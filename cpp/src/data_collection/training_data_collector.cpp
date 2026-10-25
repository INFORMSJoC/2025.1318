#include "slbjnv/data_collection/training_data_collector.hpp"

#include "slbjnv/column_generation/cplex_pricing_problem.hpp"
#include "slbjnv/column_generation/cplex_restricted_master_problem.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace slbjnv {
namespace {

struct PendingNode {
    NodeId id{kNoNode};
    std::vector<BatchColumn> seed_columns;
    double inherited_upper_bound{kCplexInfinity};
};

struct DraftRecord {
    std::size_t decision_index{};
    NodeId node_id{kNoNode};
    NodeId parent_id{kNoNode};
    BranchVariable variable;
    FeatureExtractor::FeatureVector features{};
};

bool integral(const std::vector<double>& values, double tolerance) {
    return std::all_of(values.begin(), values.end(), [tolerance](double value) {
        return std::abs(value - std::round(value)) <= tolerance;
    });
}

std::vector<FractionalColumnValue> column_solution(const BranchNode& node, double tolerance) {
    std::vector<FractionalColumnValue> result;
    const auto& primal = node.relaxation.master.lp.primal;
    for (std::size_t c = 0; c < std::min(node.columns.size(), primal.size()); ++c) {
        if (primal[c] > tolerance) result.push_back({&node.columns[c], primal[c]});
    }
    return result;
}

struct NodeRelaxation {
    ColumnGenerationResult result;
    std::vector<BatchColumn> columns;
};

NodeRelaxation solve_relaxation(const GBDInstance& instance, const std::vector<BranchConstraint>& constraints,
                                const std::vector<BatchColumn>& seeds, const std::shared_ptr<CplexCApi>& api,
                                ColumnGenerationConfig config) {
    CplexRestrictedMasterProblem master(instance, constraints, api);
    for (auto column : seeds) master.add_column(std::move(column));
    CplexPricingProblem pricing(api);
    DefaultColumnGeneration generation(instance, config);
    auto result = generation.solve(master, pricing, constraints);
    return {std::move(result), master.columns()};
}

double child_degradation(double parent_bound, const NodeRelaxation& child) {
    if (child.result.status == SolveStatus::infeasible) return std::max(1.0, std::abs(parent_bound));
    if (child.result.master.lp.status != SolveStatus::optimal || !std::isfinite(child.result.master.lp.objective)) return 0.0;
    return std::max(0.0, parent_bound - child.result.master.lp.objective);
}

BranchCandidate strong_branch(const GBDInstance& instance, const BranchNode& node,
                              std::vector<BranchCandidate> candidates,
                              const std::shared_ptr<CplexCApi>& api,
                              const StrongBranchingConfig& strong,
                              const ColumnGenerationConfig& base_cg) {
    std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
        return std::abs(left.lp_value - 0.5) < std::abs(right.lp_value - 0.5);
    });
    if (candidates.size() > strong.maximum_candidates) candidates.resize(strong.maximum_candidates);
    auto probe_config = base_cg;
    probe_config.maximum_iterations = strong.column_generation_iteration_limit;
    double best_score = -1.0;
    BranchCandidate best = candidates.front();
    for (const auto& candidate : candidates) {
        auto left_constraints = node.constraints;
        left_constraints.push_back({candidate.variable, BranchDirection::less_equal_zero});
        auto right_constraints = node.constraints;
        right_constraints.push_back({candidate.variable, BranchDirection::greater_equal_one});
        const auto left = solve_relaxation(instance, left_constraints, node.columns, api, probe_config);
        const auto right = solve_relaxation(instance, right_constraints, node.columns, api, probe_config);
        const auto down = child_degradation(node.upper_bound, left);
        const auto up = child_degradation(node.upper_bound, right);
        const auto product = std::max(1.0e-8, down) * std::max(1.0e-8, up);
        const auto score = strong.product_weight * product + (1.0 - strong.product_weight) * std::min(down, up);
        if (score > best_score + 1.0e-12) { best_score = score; best = candidate; }
    }
    return best;
}

std::vector<NodeId> subtree_nodes(const SearchTree& tree, NodeId root) {
    std::vector<NodeId> result;
    std::vector<NodeId> pending{root};
    while (!pending.empty()) {
        const auto id = pending.back();
        pending.pop_back();
        result.push_back(id);
        const auto& node = tree.node(id);
        pending.insert(pending.end(), node.child_ids.begin(), node.child_ids.end());
    }
    return result;
}

std::vector<NodeId> path_to(const SearchTree& tree, NodeId decision_node, NodeId terminal) {
    std::vector<NodeId> reversed;
    auto current = terminal;
    while (current != kNoNode) {
        reversed.push_back(current);
        if (current == decision_node) break;
        current = tree.node(current).parent_id;
    }
    if (reversed.back() != decision_node) throw std::logic_error("terminal is not in decision subtree");
    std::reverse(reversed.begin(), reversed.end());
    const auto parent = tree.node(decision_node).parent_id;
    if (parent != kNoNode) reversed.insert(reversed.begin(), parent);
    return reversed;
}

}  // namespace

TrainingCollectionResult TrainingDataCollector::collect(const GBDInstance& instance) {
    TrainingCollectionResult output;
    auto& result = output.solve_result;
    if (!api_) { result.status = SolveStatus::backend_unavailable; return output; }
    if (strong_config_.maximum_candidates == 0 || strong_config_.column_generation_iteration_limit == 0) {
        throw std::invalid_argument("strong branching limits must be positive");
    }
    if (strong_config_.product_weight < 0.0 || strong_config_.product_weight > 1.0) {
        throw std::invalid_argument("strong branching product weight must be in [0,1]");
    }
    instance.validate_or_throw();
    const auto started = std::chrono::steady_clock::now();
    auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count(); };
    IncumbentManager incumbent;
    FeatureExtractor feature_extractor;
    std::vector<DraftRecord> drafts;
    std::vector<PendingNode> stack;
    NodeId next_id = 1;
    std::size_t exploration_index = 0;
    std::size_t decision_index = 0;

    BranchNode root;
    root.id = 0;
    root.active = true;
    result.tree.add_node(std::move(root));
    stack.push_back({0, make_singleton_columns(instance), kCplexInfinity});
    result.generated_nodes = 1;

    while (!stack.empty()) {
        if (result.explored_nodes >= solve_config_.node_limit || elapsed() >= solve_config_.time_limit_seconds) {
            result.status = SolveStatus::time_limit;
            break;
        }
        auto pending = std::move(stack.back());
        stack.pop_back();
        auto& node = result.tree.node(pending.id);
        auto solved = solve_relaxation(instance, node.constraints, pending.seed_columns, api_, solve_config_.column_generation);
        node.relaxation = std::move(solved.result);
        node.columns = std::move(solved.columns);
        node.upper_bound = node.relaxation.master.lp.objective;
        node.global_lower_bound = incumbent.lower_bound();
        node.exploration_index = exploration_index++;
        node.elapsed_seconds = elapsed();
        ++result.explored_nodes;
        result.column_generation_iterations += node.relaxation.iterations;
        result.generated_columns += node.relaxation.generated_columns;

        if (node.relaxation.status == SolveStatus::infeasible) { node.active = false; continue; }
        if (node.relaxation.status != SolveStatus::optimal || !node.relaxation.converged) {
            node.active = false;
            result.status = node.relaxation.status;
            break;
        }
        node.integral = integral(node.relaxation.master.lp.primal, solve_config_.tolerances.integrality);
        if (node.integral) {
            incumbent.update(node.id, node.upper_bound, elapsed());
            node.active = false;
            continue;
        }
        if (incumbent.has_incumbent() && node.upper_bound <= incumbent.lower_bound() + solve_config_.tolerances.bound) {
            node.active = false;
            continue;
        }

        const auto values = column_solution(node, solve_config_.tolerances.feasibility);
        auto candidates = generate_all_branch_candidates(values, solve_config_.tolerances);
        if (candidates.empty()) { node.active = false; result.status = SolveStatus::numerical_error; break; }
        const auto selected = strong_branch(instance, node, std::move(candidates), api_, strong_config_,
                                            solve_config_.column_generation);
        node.branching_variable = selected.variable;
        node.decision_index = decision_index;
        drafts.push_back({decision_index++, node.id, node.parent_id, selected.variable,
                          feature_extractor.extract(instance, result.tree, node, selected)});

        const auto left_id = next_id++;
        const auto right_id = next_id++;
        node.child_ids = {left_id, right_id};
        const auto parent_id = node.id;
        const auto child_depth = node.depth + 1;
        const auto parent_bound = node.upper_bound;
        auto child_seeds = node.columns;
        auto left_constraints = node.constraints;
        left_constraints.push_back({selected.variable, BranchDirection::less_equal_zero});
        auto right_constraints = node.constraints;
        right_constraints.push_back({selected.variable, BranchDirection::greater_equal_one});
        BranchNode left_node;
        left_node.id = left_id; left_node.parent_id = parent_id; left_node.depth = child_depth;
        left_node.constraints = std::move(left_constraints); left_node.active = true;
        BranchNode right_node;
        right_node.id = right_id; right_node.parent_id = parent_id; right_node.depth = child_depth;
        right_node.constraints = std::move(right_constraints); right_node.active = true;
        node.active = false;
        result.tree.add_node(std::move(left_node));
        result.tree.add_node(std::move(right_node));
        stack.push_back({right_id, child_seeds, parent_bound});
        stack.push_back({left_id, std::move(child_seeds), parent_bound});
        result.generated_nodes += 2;
        ++result.branching_decisions;
        ++result.branching_rule_counts[static_cast<std::size_t>(static_cast<int>(selected.variable.rule) - 1)];
    }

    result.best_integer_objective = incumbent.lower_bound();
    if (stack.empty() && result.status == SolveStatus::not_solved) result.status = SolveStatus::optimal;
    result.global_upper_bound = result.status == SolveStatus::optimal ? result.best_integer_objective : kCplexInfinity;
    result.total_runtime_seconds = elapsed();
    result.incumbent_history = incumbent.history();
    std::vector<const BranchNode*> ordered_nodes;
    ordered_nodes.reserve(result.tree.size());
    for (const auto& [id, recorded] : result.tree.nodes()) {
        static_cast<void>(id);
        if (recorded.relaxation.status != SolveStatus::not_solved) ordered_nodes.push_back(&recorded);
    }
    std::sort(ordered_nodes.begin(), ordered_nodes.end(), [](const auto* left, const auto* right) {
        return left->exploration_index < right->exploration_index;
    });
    double running_lower = kNegativeInfinity;
    for (const auto* recorded : ordered_nodes) {
        if (recorded->integral) running_lower = std::max(running_lower, recorded->upper_bound);
        result.bound_history.push_back({recorded->exploration_index + 1, recorded->elapsed_seconds,
                                        running_lower, recorded->upper_bound});
    }
    if (result.status != SolveStatus::optimal) return output;

    const auto root_objective = result.tree.node(0).upper_bound;
    NodeId optimal_node = kNoNode;
    double optimal_objective = kNegativeInfinity;
    for (const auto& [id, node] : result.tree.nodes()) {
        if (!node.integral || node.relaxation.status != SolveStatus::optimal) continue;
        if (node.upper_bound > optimal_objective + solve_config_.tolerances.bound ||
            (std::abs(node.upper_bound - optimal_objective) <= solve_config_.tolerances.bound &&
             (optimal_node == kNoNode || node.exploration_index < result.tree.node(optimal_node).exploration_index))) {
            optimal_objective = node.upper_bound;
            optimal_node = id;
        }
    }
    if (optimal_node == kNoNode) throw std::logic_error("complete training tree contains no integer node");
    const auto score_denominator = static_cast<double>(result.tree.size() + 1);
    output.records.reserve(drafts.size());
    for (const auto& draft : drafts) {
        const auto subtree = subtree_nodes(result.tree, draft.node_id);
        const auto contains_optimal = std::find(subtree.begin(), subtree.end(), optimal_node) != subtree.end();
        NodeId best_integer = kNoNode;
        double best_integer_value = kNegativeInfinity;
        NodeId minimum_node = draft.node_id;
        double minimum_value = result.tree.node(draft.node_id).upper_bound;
        std::size_t above_optimal = 0;
        for (const auto id : subtree) {
            const auto& node = result.tree.node(id);
            if (node.upper_bound > optimal_objective + solve_config_.tolerances.bound) ++above_optimal;
            if (node.relaxation.status == SolveStatus::optimal && std::isfinite(node.upper_bound) &&
                node.upper_bound < minimum_value) { minimum_value = node.upper_bound; minimum_node = id; }
            if (node.integral && node.upper_bound > best_integer_value) {
                best_integer_value = node.upper_bound; best_integer = id;
            }
        }
        const auto rho1 = contains_optimal ? 2.0 : (best_integer != kNoNode ? 1.0 : 0.0);
        const auto rho2 = best_integer != kNoNode ? best_integer_value : 0.0;
        const auto target = rho1 * (1.0 + rho2 / root_objective) +
                            (1.0 - static_cast<double>(above_optimal) / score_denominator);
        const auto terminal = best_integer != kNoNode ? best_integer : minimum_node;
        auto key_nodes = path_to(result.tree, draft.node_id, terminal);
        std::vector<double> key_values{0.0};
        for (const auto id : key_nodes) key_values.push_back(result.tree.node(id).upper_bound / root_objective);
        key_values.push_back(1.0);
        output.records.push_back({instance.instance_id, draft.decision_index, draft.node_id, draft.parent_id,
                                  draft.variable, draft.features, target, std::move(key_nodes),
                                  std::move(key_values), root_objective});
    }
    result.total_runtime_seconds = elapsed();
    return output;
}

}  // namespace slbjnv
