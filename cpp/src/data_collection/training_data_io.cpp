#include "slbjnv/data_collection/training_data_io.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace slbjnv {
namespace {

template <typename Range>
void write_numbers(std::ostream& output, const Range& values) {
    output << '[';
    bool first = true;
    for (const auto& value : values) {
        if (!first) output << ',';
        first = false;
        output << value;
    }
    output << ']';
}

void write_finite(std::ostream& output, double value) {
    if (std::isfinite(value)) output << value;
    else output << "null";
}

}  // namespace

void TrainingDataIO::write_records(const std::vector<TrainingRecord>& records,
                                   const std::filesystem::path& path, bool append) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, append ? std::ios::app : std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open training data file: " + path.string());
    output << std::setprecision(17);
    for (const auto& record : records) {
        output << "{\"instance_id\":" << std::quoted(record.instance_id)
               << ",\"decision_index\":" << record.decision_index
               << ",\"node_id\":" << record.node_id
               << ",\"parent_node_id\":" << record.parent_node_id
               << ",\"branching_rule\":" << static_cast<int>(record.branching_variable.rule)
               << ",\"branching_variable\":{\"first_item\":" << record.branching_variable.first_item
               << ",\"second_item\":" << record.branching_variable.second_item
               << ",\"machine\":" << record.branching_variable.machine << "}"
               << ",\"feature_vector\":";
        write_numbers(output, record.feature_vector);
        output << ",\"target_score\":" << record.target_score << ",\"key_node_sequence\":";
        write_numbers(output, record.key_node_sequence);
        output << ",\"key_value_sequence\":";
        write_numbers(output, record.key_value_sequence);
        output << ",\"root_objective\":" << record.root_objective << "}\n";
    }
}

void TrainingDataIO::write_tree(const SearchTree& tree, const std::string& instance_id,
                                const std::filesystem::path& path) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open tree file: " + path.string());
    output << std::setprecision(17);
    std::vector<const BranchNode*> nodes;
    for (const auto& [id, node] : tree.nodes()) { static_cast<void>(id); nodes.push_back(&node); }
    std::sort(nodes.begin(), nodes.end(), [](const auto* left, const auto* right) { return left->id < right->id; });
    for (const auto* node : nodes) {
        output << "{\"instance_id\":" << std::quoted(instance_id)
               << ",\"node_id\":" << node->id << ",\"parent_node_id\":" << node->parent_id
               << ",\"depth\":" << node->depth << ",\"objective\":";
        write_finite(output, node->upper_bound);
        output << ",\"global_lower_bound\":";
        write_finite(output, node->global_lower_bound);
        output << ",\"integral\":" << (node->integral ? "true" : "false")
               << ",\"status\":" << static_cast<int>(node->relaxation.status)
               << ",\"decision_index\":" << node->decision_index
               << ",\"branching_variable\":";
        if (node->branching_variable) {
            output << "{\"rule\":" << static_cast<int>(node->branching_variable->rule)
                   << ",\"first_item\":" << node->branching_variable->first_item
                   << ",\"second_item\":" << node->branching_variable->second_item
                   << ",\"machine\":" << node->branching_variable->machine << '}';
        } else {
            output << "null";
        }
        output << ",\"constraints\":[";
        for (std::size_t index = 0; index < node->constraints.size(); ++index) {
            if (index > 0) output << ',';
            const auto& constraint = node->constraints[index];
            output << "{\"rule\":" << static_cast<int>(constraint.variable.rule)
                   << ",\"first_item\":" << constraint.variable.first_item
                   << ",\"second_item\":" << constraint.variable.second_item
                   << ",\"machine\":" << constraint.variable.machine
                   << ",\"direction\":"
                   << (constraint.direction == BranchDirection::less_equal_zero ? -1 : 1) << '}';
        }
        output << "],\"children\":";
        write_numbers(output, node->child_ids);
        output << "}\n";
    }
}

}  // namespace slbjnv
