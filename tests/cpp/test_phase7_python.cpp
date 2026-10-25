#include "slbjnv/python_interface/predictor.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) throw std::invalid_argument("checkpoint path is required");
    const auto module_path = std::filesystem::path(SLBJNV_SOURCE_DIR) / "python";
    slbjnv::PythonScorePredictor predictor(module_path, "cpu");
    predictor.load_model(argv[1]);
    std::vector<slbjnv::FeatureExtractor::FeatureVector> features(2);
    features[1].fill(0.25);
    const auto scores = predictor.predict_scores(features);
    assert(scores.size() == 2);
    assert(std::isfinite(scores[0]));
    assert(std::isfinite(scores[1]));
    return 0;
}
