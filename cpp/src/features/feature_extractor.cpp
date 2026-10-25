#include "slbjnv/features/feature_extractor.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace slbjnv {

const std::array<FeatureExtractor::Slice, 13>& FeatureExtractor::schema() {
    static const std::array<Slice, 13> value{{
        {"item_equip", 0, 520}, {"model_batch", 520, 5}, {"added_branch_constraints", 525, 100},
        {"node_variable_pair", 625, 100}, {"dual_values", 725, 100}, {"pricing_reduced_costs", 825, 30},
        {"distance_from_root", 855, 1}, {"leaf_depths", 856, 100}, {"current_depth", 956, 1},
        {"integer_variable_ratio", 957, 1}, {"bounds", 958, 2}, {"objective_sequence", 960, 20},
        {"lower_bound_sequence", 980, 20}
    }};
    return value;
}

namespace {

std::vector<double> min_max(const std::vector<double>& values) {
    if (values.empty()) return {};
    const auto [minimum, maximum] = std::minmax_element(values.begin(), values.end());
    const auto range = *maximum - *minimum;
    std::vector<double> normalized(values.size(), 0.0);
    if (range <= 1.0e-12) return normalized;
    std::transform(values.begin(), values.end(), normalized.begin(),
                   [&](double value) { return (value - *minimum) / range; });
    return normalized;
}

double safe_ratio(double numerator, double denominator) {
    return std::abs(denominator) <= 1.0e-12 ? 0.0 : numerator / denominator;
}

void write_slice(FeatureExtractor::FeatureVector& output, const FeatureExtractor::Slice& slice,
                 const std::vector<double>& values) {
    const auto count = std::min(slice.length, values.size());
    std::copy_n(values.begin(), count, output.begin() + static_cast<std::ptrdiff_t>(slice.offset));
}

std::vector<double> compress_to_100(const std::vector<double>& values) {
    if (values.size() <= 100) return values;
    const auto window = values.size() - 99;
    std::vector<double> result(100, 0.0);
    for (std::size_t position = 0; position < result.size(); ++position) {
        result[position] = std::accumulate(values.begin() + static_cast<std::ptrdiff_t>(position),
                                           values.begin() + static_cast<std::ptrdiff_t>(position + window), 0.0);
    }
    return result;
}

std::vector<double> item_coverage(const BranchNode& node, std::size_t item_count) {
    std::vector<double> coverage(item_count, 0.0);
    const auto& primal = node.relaxation.master.lp.primal;
    for (std::size_t c = 0; c < std::min(node.columns.size(), primal.size()); ++c) {
        for (const auto item : node.columns[c].items) coverage[static_cast<std::size_t>(item)] += primal[c];
    }
    return coverage;
}

}  // namespace

FeatureExtractor::FeatureVector FeatureExtractor::extract(const GBDInstance& instance, const SearchTree& tree,
                                                           const BranchNode& node, const BranchCandidate& candidate) const {
    FeatureVector output{};
    const auto& slices = schema();
    const auto root_objective = tree.contains(0) ? tree.node(0).upper_bound : node.upper_bound;
    const auto scale = std::abs(root_objective) > 1.0e-12 ? std::abs(root_objective) : 1.0;
    const auto maximum_benefit = std::max(1.0, std::accumulate(
        instance.items.begin(), instance.items.end(), 0.0,
        [](double total, const Item& item) { return total + item.profit; }));

    std::vector<double> weights, profits, capacities;
    for (const auto& item : instance.items) { weights.push_back(item.weight); profits.push_back(item.profit); }
    for (const auto& machine : instance.machines) capacities.push_back(machine.capacity);
    const auto normalized_weights = min_max(weights);
    const auto normalized_profits = min_max(profits);
    const auto normalized_capacities = min_max(capacities);
    std::vector<double> item_equipment;
    item_equipment.reserve(instance.items.size() * 2 + instance.machines.size());
    for (std::size_t i = 0; i < instance.items.size(); ++i) {
        item_equipment.push_back(safe_ratio(normalized_weights[i], normalized_profits[i]) +
                                 safe_ratio(normalized_profits[i], normalized_weights[i]));
    }
    item_equipment.insert(item_equipment.end(), normalized_capacities.begin(), normalized_capacities.end());
    for (std::size_t master = 0; master < instance.items.size(); ++master) {
        const auto compatible = std::count(instance.compatible_with_master[master].begin(),
                                           instance.compatible_with_master[master].end(), true);
        item_equipment.push_back(static_cast<double>(compatible) / static_cast<double>(instance.items.size()));
    }
    write_slice(output, slices[0], item_equipment);

    std::vector<double> benefits;
    benefits.reserve(node.columns.size());
    for (const auto& column : node.columns) benefits.push_back(column.benefit);
    if (!benefits.empty()) {
        auto sorted = benefits;
        std::sort(sorted.begin(), sorted.end());
        const auto mean = std::accumulate(benefits.begin(), benefits.end(), 0.0) / benefits.size();
        double variance = 0.0;
        for (const auto value : benefits) variance += (value - mean) * (value - mean);
        variance /= benefits.size();
        const auto median = sorted.size() % 2 == 0
            ? 0.5 * (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2])
            : sorted[sorted.size() / 2];
        write_slice(output, slices[1], {mean / maximum_benefit, variance / (maximum_benefit * maximum_benefit),
                                        median / maximum_benefit, sorted.front() / maximum_benefit,
                                        sorted.back() / maximum_benefit});
    }

    std::vector<double> added_constraints;
    for (const auto& constraint : node.constraints) {
        double sum = 0.0;
        for (const auto& column : node.columns) {
            if (constraint.coefficient(column) > 0.5) sum += column.benefit / maximum_benefit;
        }
        added_constraints.push_back((constraint.direction == BranchDirection::less_equal_zero ? -1.0 : 1.0) * sum);
    }
    write_slice(output, slices[2], added_constraints);

    const BranchConstraint candidate_constraint{candidate.variable, BranchDirection::greater_equal_one};
    std::vector<double> pair_benefits;
    for (const auto& column : node.columns) {
        if (candidate_constraint.coefficient(column) > 0.5) pair_benefits.push_back(column.benefit / maximum_benefit);
    }
    write_slice(output, slices[3], compress_to_100(pair_benefits));

    std::vector<double> duals = node.relaxation.first_iteration_duals.item_duals;
    duals.insert(duals.end(), node.relaxation.first_iteration_duals.machine_duals.begin(),
                 node.relaxation.first_iteration_duals.machine_duals.end());
    duals.insert(duals.end(), node.relaxation.first_iteration_duals.branch_duals.begin(),
                 node.relaxation.first_iteration_duals.branch_duals.end());
    for (auto& value : duals) value /= scale;
    write_slice(output, slices[4], duals);
    auto reduced_costs = node.relaxation.first_iteration_reduced_costs;
    for (auto& value : reduced_costs) value /= scale;
    write_slice(output, slices[5], reduced_costs);

    if (tree.contains(0)) {
        const auto root_coverage = item_coverage(tree.node(0), instance.items.size());
        const auto current_coverage = item_coverage(node, instance.items.size());
        double squared_distance = 0.0;
        for (std::size_t i = 0; i < root_coverage.size(); ++i) {
            const auto difference = root_coverage[i] - current_coverage[i];
            squared_distance += difference * difference;
        }
        output[slices[6].offset] = std::sqrt(squared_distance);
    }

    std::vector<double> leaf_depths(100, 0.0);
    for (const auto& [id, value] : tree.nodes()) {
        static_cast<void>(id);
        if (!value.active) continue;
        const auto depth = std::min<std::size_t>(100, std::max<std::size_t>(1, value.depth));
        leaf_depths[depth - 1] += 1.0;
    }
    write_slice(output, slices[7], leaf_depths);
    output[slices[8].offset] = static_cast<double>(node.depth);
    const auto& primal = node.relaxation.master.lp.primal;
    const auto integer_count = std::count_if(primal.begin(), primal.end(), [](double value) {
        return std::abs(value - std::round(value)) <= 1.0e-6;
    });
    output[slices[9].offset] = primal.empty() ? 1.0 : static_cast<double>(integer_count) / primal.size();
    output[slices[10].offset] = node.upper_bound / scale;
    output[slices[10].offset + 1] = std::isfinite(node.global_lower_bound) ? node.global_lower_bound / scale : 0.0;

    std::vector<const BranchNode*> history;
    for (const auto& [id, value] : tree.nodes()) {
        static_cast<void>(id);
        if (value.relaxation.status == SolveStatus::not_solved || value.exploration_index >= node.exploration_index) continue;
        history.push_back(&value);
    }
    std::sort(history.begin(), history.end(), [](const auto* left, const auto* right) {
        return left->exploration_index < right->exploration_index;
    });
    if (history.size() > 20) history.erase(history.begin(), history.end() - 20);
    std::vector<double> objectives, lower_bounds;
    for (const auto* value : history) {
        objectives.push_back(std::isfinite(value->upper_bound) ? value->upper_bound / scale : 0.0);
        lower_bounds.push_back(std::isfinite(value->global_lower_bound) ? value->global_lower_bound / scale : 0.0);
    }
    write_slice(output, slices[11], objectives);
    write_slice(output, slices[12], lower_bounds);
    return output;
}

}  // namespace slbjnv
