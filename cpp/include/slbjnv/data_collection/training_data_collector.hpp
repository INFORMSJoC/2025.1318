#pragma once

#include "slbjnv/branch_and_price/default_branch_and_price.hpp"
#include "slbjnv/data_collection/training_record.hpp"

namespace slbjnv {

struct StrongBranchingConfig {
    std::size_t maximum_candidates{20};
    std::size_t column_generation_iteration_limit{25};
    double product_weight{0.5};
};

struct TrainingCollectionResult {
    BranchAndPriceResult solve_result;
    std::vector<TrainingRecord> records;
};

class TrainingDataCollector {
public:
    TrainingDataCollector(std::shared_ptr<CplexCApi> api,
                          BranchAndPriceConfig solve_config = {},
                          StrongBranchingConfig strong_config = {})
        : api_(std::move(api)), solve_config_(solve_config), strong_config_(strong_config) {}

    [[nodiscard]] TrainingCollectionResult collect(const GBDInstance& instance);

private:
    std::shared_ptr<CplexCApi> api_;
    BranchAndPriceConfig solve_config_;
    StrongBranchingConfig strong_config_;
};

}  // namespace slbjnv
