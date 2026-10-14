#include "slbjnv/python_interface/predictor.hpp"

#include <pybind11/embed.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <utility>

namespace py = pybind11;

namespace slbjnv {
namespace {

class EmbeddedPythonRuntime {
public:
    EmbeddedPythonRuntime() : interpreter_{} {}

private:
    py::scoped_interpreter interpreter_;
};

EmbeddedPythonRuntime& runtime() {
    // PyTorch owns process-wide worker pools and CUDA state that are not safe to
    // tear down through Py_Finalize on all Windows builds. Keep the interpreter
    // alive until the operating system releases the process.
    static auto* value = new EmbeddedPythonRuntime();
    return *value;
}

std::runtime_error python_error(const char* operation, const py::error_already_set& error) {
    return std::runtime_error(std::string(operation) + ": " + error.what());
}

}  // namespace

class PythonScorePredictor::Impl {
public:
    Impl(const std::filesystem::path& module_search_path, std::string device)
        : device_(std::move(device)) {
        static_cast<void>(runtime());
        try {
            py::gil_scoped_acquire acquire;
            if (!module_search_path.empty()) {
                py::module_::import("sys").attr("path").attr("insert")(
                    0, module_search_path.string());
            }
            module_ = py::module_::import("slbjnv.predictor");
        } catch (const py::error_already_set& error) {
            throw python_error("cannot import slbjnv.predictor", error);
        }
    }

    void load(const std::filesystem::path& model_path) {
        try {
            py::gil_scoped_acquire acquire;
            module_.attr("load_model")(model_path.string(), device_);
            loaded_ = true;
        } catch (const py::error_already_set& error) {
            throw python_error("cannot load SLbJNV model", error);
        }
    }

    std::vector<double> predict(
        const std::vector<FeatureExtractor::FeatureVector>& candidate_features) {
        if (!loaded_) throw std::logic_error("SLbJNV model must be loaded before prediction");
        if (candidate_features.empty()) return {};
        try {
            py::gil_scoped_acquire acquire;
            py::list rows;
            for (const auto& features : candidate_features) {
                py::list row;
                for (const auto value : features) row.append(value);
                rows.append(std::move(row));
            }
            return module_.attr("predict_scores")(rows).cast<std::vector<double>>();
        } catch (const py::error_already_set& error) {
            throw python_error("SLbJNV batch prediction failed", error);
        }
    }

private:
    py::object module_;
    std::string device_;
    bool loaded_{};
};

PythonScorePredictor::PythonScorePredictor(std::filesystem::path module_search_path,
                                           std::string device)
    : impl_(std::make_unique<Impl>(module_search_path, std::move(device))) {}

PythonScorePredictor::~PythonScorePredictor() = default;
PythonScorePredictor::PythonScorePredictor(PythonScorePredictor&&) noexcept = default;
PythonScorePredictor& PythonScorePredictor::operator=(PythonScorePredictor&&) noexcept = default;

void PythonScorePredictor::load_model(const std::filesystem::path& model_path) {
    impl_->load(model_path);
}

std::vector<double> PythonScorePredictor::predict_scores(
    const std::vector<FeatureExtractor::FeatureVector>& candidate_features) {
    auto scores = impl_->predict(candidate_features);
    if (scores.size() != candidate_features.size()) {
        throw std::runtime_error("Python predictor returned an unexpected number of scores");
    }
    return scores;
}

}  // namespace slbjnv
