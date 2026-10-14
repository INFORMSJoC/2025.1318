#pragma once

#include "slbjnv/branch_and_price/branch_and_price.hpp"
#include "slbjnv/column_generation/default_column_generation.hpp"
#include "slbjnv/model/cplex_c_api.hpp"
#include "slbjnv/python_interface/predictor.hpp"

#include <memory>

namespace slbjnv {

struct BranchAndPriceConfig {
    NumericalTolerances tolerances;
    ColumnGenerationConfig column_generation;
    std::size_t node_limit{1000000};
    double time_limit_seconds{3600.0};
};

class DefaultBranchAndPrice final : public BranchAndPrice {
public:
    DefaultBranchAndPrice(std::shared_ptr<CplexCApi> api, BranchAndPriceConfig config = {},
                          std::shared_ptr<ScorePredictor> predictor = {})
        : api_(std::move(api)), config_(config), predictor_(std::move(predictor)) {}

    [[nodiscard]] BranchAndPriceResult solve(const GBDInstance& instance, SearchMode mode) override;

private:
    [[nodiscard]] BranchAndPriceResult solve_slbjnv(const GBDInstance& instance);

    std::shared_ptr<CplexCApi> api_;
    BranchAndPriceConfig config_;
    std::shared_ptr<ScorePredictor> predictor_;
};

}  // namespace slbjnv
