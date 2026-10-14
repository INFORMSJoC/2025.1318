#include "slbjnv/branch_and_price/default_branch_and_price.hpp"

#include "slbjnv/column_generation/cplex_pricing_problem.hpp"
#include "slbjnv/column_generation/cplex_restricted_master_problem.hpp"
#include "slbjnv/branch_and_price/slbjnv_branching_policy.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <utility>

namespace slbjnv {
namespace {

struct PendingNode {
    NodeId id{kNoNode};
    NodeId parent{kNoNode};
    std::size_t depth{};
    std::vector<BranchConstraint> constraints;
    std::vector<BatchColumn> seed_columns;
    double inherited_upper_bound{kCplexInfinity};
};

bool is_integral(const std::vector<double>& values, double tolerance) {
    return std::all_of(values.begin(), values.end(), [tolerance](double value) {
        return std::abs(value - std::round(value)) <= tolerance;
    });
}

}  // namespace

BranchAndPriceResult DefaultBranchAndPrice::solve(const GBDInstance& instance, SearchMode mode) {
    if (mode == SearchMode::slbjnv) return solve_slbjnv(instance);
    BranchAndPriceResult result;
    if (!api_) { result.status = SolveStatus::backend_unavailable; return result; }
    instance.validate_or_throw();
    const auto started = std::chrono::steady_clock::now();
    auto elapsed = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    };

    IncumbentManager incumbent;
    std::vector<PendingNode> stack;
    stack.push_back({0, kNoNode, 0, {}, make_singleton_columns(instance), kCplexInfinity});
    result.generated_nodes = 1;
    NodeId next_node_id = 1;
    CplexPricingProblem pricing(api_);

    while (!stack.empty()) {
        if (result.explored_nodes >= config_.node_limit || elapsed() >= config_.time_limit_seconds) {
            result.status = SolveStatus::time_limit;
            break;
        }
        auto pending = std::move(stack.back());
        stack.pop_back();
        if (incumbent.has_incumbent() && pending.inherited_upper_bound <= incumbent.lower_bound() + config_.tolerances.bound) {
            continue;
        }

        CplexRestrictedMasterProblem master(instance, pending.constraints, api_);
        for (auto& column : pending.seed_columns) master.add_column(std::move(column));
        DefaultColumnGeneration column_generation(instance, config_.column_generation);
        auto relaxation = column_generation.solve(master, pricing, pending.constraints);
        ++result.explored_nodes;
        result.column_generation_iterations += relaxation.iterations;
        result.generated_columns += relaxation.generated_columns;

        BranchNode node;
        node.id = pending.id;
        node.parent_id = pending.parent;
        node.depth = pending.depth;
        node.constraints = pending.constraints;
        node.columns = master.columns();
        node.relaxation = std::move(relaxation);
        node.upper_bound = node.relaxation.master.lp.objective;
        node.global_lower_bound = incumbent.lower_bound();
        node.exploration_index = result.explored_nodes - 1;
        node.elapsed_seconds = elapsed();
        node.active = false;
        if (node.relaxation.status == SolveStatus::infeasible) {
            result.tree.add_node(std::move(node));
            continue;
        }
        if (node.relaxation.status != SolveStatus::optimal || !node.relaxation.converged) {
            result.status = node.relaxation.status;
            result.tree.add_node(std::move(node));
            break;
        }
        if (incumbent.has_incumbent() && node.upper_bound <= incumbent.lower_bound() + config_.tolerances.bound) {
            result.tree.add_node(std::move(node));
            continue;
        }

        const auto& primal = node.relaxation.master.lp.primal;
        node.integral = is_integral(primal, config_.tolerances.integrality);
        if (node.integral) {
            incumbent.update(node.id, node.upper_bound, elapsed());
            result.tree.add_node(std::move(node));
            continue;
        }

        std::vector<FractionalColumnValue> column_solution;
        for (std::size_t c = 0; c < node.columns.size(); ++c) {
            if (primal[c] > config_.tolerances.feasibility) column_solution.push_back({&node.columns[c], primal[c]});
        }
        const auto candidates = generate_all_branch_candidates(column_solution, config_.tolerances);
        if (candidates.empty()) {
            result.status = SolveStatus::numerical_error;
            result.tree.add_node(std::move(node));
            break;
        }
        const auto selected = std::min_element(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
            const auto left_distance = std::abs(left.lp_value - 0.5);
            const auto right_distance = std::abs(right.lp_value - 0.5);
            if (std::abs(left_distance - right_distance) > 1.0e-12) return left_distance < right_distance;
            return static_cast<int>(left.variable.rule) < static_cast<int>(right.variable.rule);
        });
        const auto candidate = *selected;
        node.branching_variable = candidate.variable;
        const auto left_id = next_node_id++;
        const auto right_id = next_node_id++;
        node.child_ids = {left_id, right_id};
        const auto parent_upper_bound = node.upper_bound;
        auto child_columns = node.columns;
        auto left_constraints = node.constraints;
        left_constraints.push_back({candidate.variable, BranchDirection::less_equal_zero});
        auto right_constraints = node.constraints;
        right_constraints.push_back({candidate.variable, BranchDirection::greater_equal_one});
        result.tree.add_node(std::move(node));

        // Push the >= child first so that the <= child is explored first by DFS.
        stack.push_back({right_id, pending.id, pending.depth + 1, std::move(right_constraints), child_columns, parent_upper_bound});
        stack.push_back({left_id, pending.id, pending.depth + 1, std::move(left_constraints), std::move(child_columns), parent_upper_bound});
        result.generated_nodes += 2;
        ++result.branching_decisions;
        ++result.branching_rule_counts[static_cast<std::size_t>(static_cast<int>(candidate.variable.rule) - 1)];
    }

    result.best_integer_objective = incumbent.lower_bound();
    if (stack.empty() && result.status == SolveStatus::not_solved) result.status = SolveStatus::optimal;
    if (result.status == SolveStatus::optimal) {
        result.global_upper_bound = result.best_integer_objective;
    } else {
        result.global_upper_bound = incumbent.lower_bound();
        for (const auto& pending : stack) result.global_upper_bound = std::max(result.global_upper_bound, pending.inherited_upper_bound);
    }
    result.total_runtime_seconds = elapsed();
    result.incumbent_history = incumbent.history();
    std::vector<const BranchNode*> ordered_nodes;
    ordered_nodes.reserve(result.tree.size());
    for (const auto& [id, node] : result.tree.nodes()) {
        static_cast<void>(id);
        if (node.relaxation.status != SolveStatus::not_solved) ordered_nodes.push_back(&node);
    }
    std::sort(ordered_nodes.begin(), ordered_nodes.end(), [](const auto* left, const auto* right) {
        return left->exploration_index < right->exploration_index;
    });
    double running_lower = kNegativeInfinity;
    for (const auto* node : ordered_nodes) {
        if (node->integral) running_lower = std::max(running_lower, node->upper_bound);
        result.bound_history.push_back({node->exploration_index + 1, node->elapsed_seconds,
                                        running_lower, node->upper_bound});
    }
    return result;
}

BranchAndPriceResult DefaultBranchAndPrice::solve_slbjnv(const GBDInstance& instance) {
    BranchAndPriceResult result;
    if (!api_ || !predictor_) {
        result.status = SolveStatus::backend_unavailable;
        return result;
    }
    instance.validate_or_throw();
    const auto started = std::chrono::steady_clock::now();
    auto elapsed = [&] {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    };

    IncumbentManager incumbent;
    CplexPricingProblem pricing(api_);
    std::vector<NodeId> active_ids;
    NodeId next_node_id = 1;
    std::size_t exploration_index = 0;

    auto evaluate = [&](PendingNode pending) {
        CplexRestrictedMasterProblem master(instance, pending.constraints, api_);
        for (auto& column : pending.seed_columns) master.add_column(std::move(column));
        DefaultColumnGeneration column_generation(instance, config_.column_generation);
        auto relaxation = column_generation.solve(master, pricing, pending.constraints);
        ++result.explored_nodes;
        result.column_generation_iterations += relaxation.iterations;
        result.generated_columns += relaxation.generated_columns;

        BranchNode node;
        node.id = pending.id;
        node.parent_id = pending.parent;
        node.depth = pending.depth;
        node.constraints = std::move(pending.constraints);
        node.columns = master.columns();
        node.relaxation = std::move(relaxation);
        node.upper_bound = node.relaxation.master.lp.objective;
        node.global_lower_bound = incumbent.lower_bound();
        node.exploration_index = exploration_index++;
        node.elapsed_seconds = elapsed();
        node.active = false;
        return node;
    };

    auto register_node = [&](BranchNode node) -> bool {
        if (node.relaxation.status == SolveStatus::infeasible) {
            result.tree.add_node(std::move(node));
            return true;
        }
        if (node.relaxation.status != SolveStatus::optimal || !node.relaxation.converged) {
            result.status = node.relaxation.status;
            result.tree.add_node(std::move(node));
            return false;
        }
        if (incumbent.has_incumbent()
            && node.upper_bound <= incumbent.lower_bound() + config_.tolerances.bound) {
            result.tree.add_node(std::move(node));
            return true;
        }
        node.integral = is_integral(node.relaxation.master.lp.primal,
                                    config_.tolerances.integrality);
        if (node.integral) {
            incumbent.update(node.id, node.upper_bound, elapsed());
        } else {
            node.active = true;
            active_ids.push_back(node.id);
        }
        result.tree.add_node(std::move(node));
        return true;
    };

    result.generated_nodes = 1;
    if (!register_node(evaluate({0, kNoNode, 0, {}, make_singleton_columns(instance),
                                 kCplexInfinity}))) {
        result.best_integer_objective = incumbent.lower_bound();
        return result;
    }
    SLbJNVBranchingPolicy policy(instance, result.tree, *predictor_, config_.tolerances);

    while (!active_ids.empty()) {
        if (result.explored_nodes >= config_.node_limit
            || elapsed() >= config_.time_limit_seconds) {
            result.status = SolveStatus::time_limit;
            break;
        }
        active_ids.erase(std::remove_if(active_ids.begin(), active_ids.end(), [&](NodeId id) {
            auto& node = result.tree.node(id);
            const auto prune = incumbent.has_incumbent()
                && node.upper_bound <= incumbent.lower_bound() + config_.tolerances.bound;
            if (prune) node.active = false;
            return prune;
        }), active_ids.end());
        if (active_ids.empty()) break;

        std::vector<const BranchNode*> active_nodes;
        active_nodes.reserve(active_ids.size());
        for (const auto id : active_ids) active_nodes.push_back(&result.tree.node(id));
        const auto decision = policy.select(active_nodes);
        if (!decision) {
            result.status = SolveStatus::numerical_error;
            break;
        }
        if (result.explored_nodes + 2 > config_.node_limit) {
            result.status = SolveStatus::time_limit;
            break;
        }

        const auto parent_id = decision->node_id;
        const auto parent_depth = result.tree.node(parent_id).depth;
        const auto parent_upper_bound = result.tree.node(parent_id).upper_bound;
        const auto parent_columns = result.tree.node(parent_id).columns;
        const auto parent_constraints = result.tree.node(parent_id).constraints;
        const auto left_id = next_node_id++;
        const auto right_id = next_node_id++;
        {
            auto& parent = result.tree.node(parent_id);
            parent.active = false;
            parent.branching_variable = decision->candidate.variable;
            parent.child_ids = {left_id, right_id};
        }
        active_ids.erase(std::remove(active_ids.begin(), active_ids.end(), parent_id), active_ids.end());
        result.generated_nodes += 2;
        ++result.branching_decisions;
        ++result.branching_rule_counts[
            static_cast<std::size_t>(static_cast<int>(decision->candidate.variable.rule) - 1)];

        auto left_constraints = parent_constraints;
        left_constraints.push_back({decision->candidate.variable,
                                    BranchDirection::less_equal_zero});
        auto right_constraints = parent_constraints;
        right_constraints.push_back({decision->candidate.variable,
                                     BranchDirection::greater_equal_one});
        if (!register_node(evaluate({left_id, parent_id, parent_depth + 1,
                                     std::move(left_constraints), parent_columns,
                                     parent_upper_bound}))) break;
        if (elapsed() >= config_.time_limit_seconds) {
            result.status = SolveStatus::time_limit;
            break;
        }
        if (!register_node(evaluate({right_id, parent_id, parent_depth + 1,
                                     std::move(right_constraints), parent_columns,
                                     parent_upper_bound}))) break;
    }

    result.inference_seconds = policy.inference_seconds();
    result.predicted_pairs = policy.predicted_pairs();
    result.best_integer_objective = incumbent.lower_bound();
    if (active_ids.empty() && result.status == SolveStatus::not_solved) {
        result.status = SolveStatus::optimal;
    }
    if (result.status == SolveStatus::optimal) {
        result.global_upper_bound = result.best_integer_objective;
    } else {
        result.global_upper_bound = incumbent.lower_bound();
        for (const auto id : active_ids) {
            result.global_upper_bound = std::max(result.global_upper_bound,
                                                 result.tree.node(id).upper_bound);
        }
    }
    result.total_runtime_seconds = elapsed();
    result.incumbent_history = incumbent.history();
    std::vector<const BranchNode*> ordered_nodes;
    ordered_nodes.reserve(result.tree.size());
    for (const auto& [id, node] : result.tree.nodes()) {
        static_cast<void>(id);
        if (node.relaxation.status != SolveStatus::not_solved) ordered_nodes.push_back(&node);
    }
    std::sort(ordered_nodes.begin(), ordered_nodes.end(), [](const auto* left, const auto* right) {
        return left->exploration_index < right->exploration_index;
    });
    double running_lower = kNegativeInfinity;
    for (const auto* node : ordered_nodes) {
        if (node->integral) running_lower = std::max(running_lower, node->upper_bound);
        result.bound_history.push_back({node->exploration_index + 1, node->elapsed_seconds,
                                        running_lower, node->upper_bound});
    }
    return result;
}

}  // namespace slbjnv
