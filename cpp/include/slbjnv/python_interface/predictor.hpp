#pragma once

#include "slbjnv/features/feature_extractor.hpp"

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace slbjnv {

class ScorePredictor {
public:
    virtual ~ScorePredictor() = default;
    virtual void load_model(const std::filesystem::path& model_path) = 0;
    [[nodiscard]] virtual std::vector<double> predict_scores(
        const std::vector<FeatureExtractor::FeatureVector>& candidate_features) = 0;
};

#ifdef SLBJNV_WITH_PYTHON
class PythonScorePredictor final : public ScorePredictor {
public:
    explicit PythonScorePredictor(std::filesystem::path module_search_path = {},
                                  std::string device = "cpu");
    ~PythonScorePredictor() override;
    PythonScorePredictor(const PythonScorePredictor&) = delete;
    PythonScorePredictor& operator=(const PythonScorePredictor&) = delete;
    PythonScorePredictor(PythonScorePredictor&&) noexcept;
    PythonScorePredictor& operator=(PythonScorePredictor&&) noexcept;

    void load_model(const std::filesystem::path& model_path) override;
    [[nodiscard]] std::vector<double> predict_scores(
        const std::vector<FeatureExtractor::FeatureVector>& candidate_features) override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
#endif

}  // namespace slbjnv
