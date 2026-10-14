#include "slbjnv/data_collection/training_data_collector.hpp"
#include "slbjnv/data_collection/training_data_io.hpp"
#include "slbjnv/instance/instance_io.hpp"

#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

int main() {
    using namespace slbjnv;
    const auto root = std::filesystem::path(SLBJNV_SOURCE_DIR);
    const auto instance = GBDInstanceIO::read_file(root / "tests/fixtures/gbd_2_5_fractional.txt");
    const auto api = CplexCApi::load();
    TrainingDataCollector collector(api);
    const auto collected = collector.collect(instance);
    assert(collected.solve_result.status == SolveStatus::optimal);
    assert(collected.solve_result.best_integer_objective == 55.0);
    assert(collected.solve_result.tree.size() >= 3);
    assert(collected.solve_result.total_runtime_seconds >= 0.0);
    assert(!collected.solve_result.incumbent_history.empty());
    assert(!collected.solve_result.bound_history.empty());
    assert(!collected.records.empty());
    for (std::size_t index = 0; index < collected.records.size(); ++index) {
        const auto& record = collected.records[index];
        assert(record.instance_id == instance.instance_id);
        assert(record.decision_index == index);
        assert(record.feature_vector.size() == FeatureExtractor::kDimension);
        for (const auto value : record.feature_vector) assert(std::isfinite(value));
        assert(record.target_score > 0.0);
        assert(!record.key_node_sequence.empty());
        assert(record.key_value_sequence.front() == 0.0);
        assert(record.key_value_sequence.back() == 1.0);
        assert(record.key_value_sequence.size() == record.key_node_sequence.size() + 2);
    }
    const auto output = root / "work/phase5_test_records.txt";
    TrainingDataIO::write_records(collected.records, output);
    std::ifstream input(output);
    std::string line;
    std::size_t lines = 0;
    while (std::getline(input, line)) {
        assert(line.find("\"feature_vector\":[") != std::string::npos);
        assert(line.find("\"target_score\":") != std::string::npos);
        ++lines;
    }
    assert(lines == collected.records.size());
    return 0;
}
