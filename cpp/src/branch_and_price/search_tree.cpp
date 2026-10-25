#include "slbjnv/branch_and_price/search_tree.hpp"

#include <stdexcept>
#include <utility>

namespace slbjnv {

BranchNode& SearchTree::add_node(BranchNode value) {
    if (value.id == kNoNode) throw std::invalid_argument("node id must be assigned");
    const auto id = value.id;
    const auto [it, inserted] = nodes_.emplace(id, std::move(value));
    if (!inserted) throw std::invalid_argument("duplicate node id");
    return it->second;
}

BranchNode& SearchTree::node(NodeId id) { return nodes_.at(id); }
const BranchNode& SearchTree::node(NodeId id) const { return nodes_.at(id); }

}  // namespace slbjnv
