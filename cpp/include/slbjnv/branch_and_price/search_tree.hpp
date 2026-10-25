#pragma once

#include "slbjnv/branch_and_price/branch_node.hpp"

#include <unordered_map>

namespace slbjnv {

class SearchTree {
public:
    BranchNode& add_node(BranchNode node);
    [[nodiscard]] BranchNode& node(NodeId id);
    [[nodiscard]] const BranchNode& node(NodeId id) const;
    [[nodiscard]] bool contains(NodeId id) const noexcept { return nodes_.find(id) != nodes_.end(); }
    [[nodiscard]] const std::unordered_map<NodeId, BranchNode>& nodes() const noexcept { return nodes_; }
    [[nodiscard]] std::size_t size() const noexcept { return nodes_.size(); }

private:
    std::unordered_map<NodeId, BranchNode> nodes_;
};

}  // namespace slbjnv
