#pragma once

#include "slbjnv/branch_and_price/branching_policy.hpp"
#include "slbjnv/branch_and_price/search_tree.hpp"

#include <array>
#include <cstddef>
#include <string_view>

namespace slbjnv {

class FeatureExtractor {
public:
    static constexpr std::size_t kDimension = 1000;
    using FeatureVector = std::array<double, kDimension>;

    struct Slice { std::string_view name; std::size_t offset; std::size_t length; };
    static const std::array<Slice, 13>& schema();
    [[nodiscard]] FeatureVector extract(const GBDInstance& instance,
                                        const SearchTree& tree,
                                        const BranchNode& node,
                                        const BranchCandidate& candidate) const;
};

}  // namespace slbjnv
