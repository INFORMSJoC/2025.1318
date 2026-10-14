#include "slbjnv/instance/instance_io.hpp"

#ifdef SLBJNV_WITH_CPLEX
#include "slbjnv/data_collection/training_data_collector.hpp"
#include "slbjnv/data_collection/training_data_io.hpp"
#include "slbjnv/model/cplex_c_api.hpp"
#endif

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::filesystem::path> instance_files(const std::filesystem::path& input) {
    if (std::filesystem::is_regular_file(input)) return {input};
    if (!std::filesystem::is_directory(input)) throw std::invalid_argument("instance input does not exist");
    std::vector<std::filesystem::path> files;
    for (const auto& entry : std::filesystem::directory_iterator(input)) {
        if (entry.is_regular_file() && entry.path().extension() == ".txt") files.push_back(entry.path());
    }
    std::sort(files.begin(), files.end());
    return files;
}

#ifdef SLBJNV_WITH_CPLEX
std::pair<slbjnv::BranchAndPriceConfig, slbjnv::StrongBranchingConfig>
read_config(const std::filesystem::path& path) {
    slbjnv::BranchAndPriceConfig solve;
    slbjnv::StrongBranchingConfig strong;
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open configuration: " + path.string());
    std::string section, line;
    while (std::getline(input, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        const auto first = line.find_first_not_of(" \t\r");
        if (first == std::string::npos) continue;
        const auto colon = line.find(':', first);
        if (colon == std::string::npos) continue;
        auto key = line.substr(first, colon - first);
        auto value = line.substr(colon + 1);
        const auto value_first = value.find_first_not_of(" \t");
        if (value_first == std::string::npos) { section = key; continue; }
        value = value.substr(value_first);
        if (section == "solver" && key == "time_limit_seconds") solve.time_limit_seconds = std::stod(value);
        else if (section == "column_generation" && key == "reduced_cost_tolerance")
            solve.column_generation.reduced_cost_tolerance = std::stod(value);
        else if (section == "column_generation" && key == "maximum_iterations")
            solve.column_generation.maximum_iterations = std::stoull(value);
        else if (section == "branch_and_bound" && key == "integrality_tolerance")
            solve.tolerances.integrality = std::stod(value);
        else if (section == "branch_and_bound" && key == "feasibility_tolerance")
            solve.tolerances.feasibility = std::stod(value);
        else if (section == "branch_and_bound" && key == "bound_tolerance")
            solve.tolerances.bound = std::stod(value);
        else if (section == "branch_and_bound" && key == "node_limit") solve.node_limit = std::stoull(value);
        else if (section == "strong_branching" && key == "maximum_candidates")
            strong.maximum_candidates = std::stoull(value);
        else if (section == "strong_branching" && key == "column_generation_iteration_limit")
            strong.column_generation_iteration_limit = std::stoull(value);
        else if (section == "strong_branching" && key == "score_product_weight")
            strong.product_weight = std::stod(value);
    }
    return {solve, strong};
}
#endif

}  // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path input;
        std::filesystem::path output_directory = "data/training";
        std::filesystem::path config_path = "configs/branch_and_price.yaml";
        std::size_t limit = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string argument = argv[i];
            if (argument == "--instances" && i + 1 < argc) input = argv[++i];
            else if (argument == "--output" && i + 1 < argc) output_directory = argv[++i];
            else if (argument == "--config" && i + 1 < argc) config_path = argv[++i];
            else if (argument == "--limit" && i + 1 < argc) limit = std::stoull(argv[++i]);
            else if (argument == "--help") {
                std::cout << "Usage: generate_training_data --instances FILE_OR_DIR [--output DIR] [--config FILE] [--limit N]\n";
                return 0;
            } else throw std::invalid_argument("unknown or incomplete argument: " + argument);
        }
        if (input.empty()) throw std::invalid_argument("--instances is required");
#ifndef SLBJNV_WITH_CPLEX
        throw std::runtime_error("generate_training_data was built without CPLEX support");
#else
        auto files = instance_files(input);
        if (limit > 0 && files.size() > limit) files.resize(limit);
        if (files.empty()) throw std::invalid_argument("no instance TXT files found");
        std::mt19937 generator(42U);
        std::shuffle(files.begin(), files.end(), generator);
        const auto train_count = std::max<std::size_t>(1, static_cast<std::size_t>(std::llround(files.size() * 0.70)));
        const auto remaining = files.size() - std::min(train_count, files.size());
        const auto validation_count = std::min(remaining, static_cast<std::size_t>(std::llround(files.size() * 0.15)));
        const auto api = slbjnv::CplexCApi::load();
        const auto [solve_config, strong_config] = read_config(config_path);
        slbjnv::TrainingDataCollector collector(api, solve_config, strong_config);
        std::vector<slbjnv::TrainingRecord> training_records, validation_records;
        std::size_t total_nodes = 0;
        for (std::size_t index = 0; index < train_count + validation_count; ++index) {
            const auto instance = slbjnv::GBDInstanceIO::read_file(files[index]);
            auto collected = collector.collect(instance);
            if (collected.solve_result.status != slbjnv::SolveStatus::optimal) {
                throw std::runtime_error("training instance was not solved to completion: " + instance.instance_id);
            }
            total_nodes += collected.solve_result.tree.size();
            slbjnv::TrainingDataIO::write_tree(
                collected.solve_result.tree, instance.instance_id,
                output_directory / "trees" / (instance.instance_id + ".txt"));
            auto& destination = index < train_count ? training_records : validation_records;
            destination.insert(destination.end(),
                               std::make_move_iterator(collected.records.begin()),
                               std::make_move_iterator(collected.records.end()));
            std::cout << "collected " << instance.instance_id << " nodes="
                      << collected.solve_result.tree.size() << " samples=" << collected.records.size() << '\n';
        }
        slbjnv::TrainingDataIO::write_records(training_records, output_directory / "train.txt");
        slbjnv::TrainingDataIO::write_records(validation_records, output_directory / "validation.txt");
        std::filesystem::create_directories(output_directory);
        std::ofstream summary(output_directory / "dataset_summary.txt", std::ios::trunc);
        summary << "split_method=instance_level\nratio=70:15:15\n"
                << "total_instances=" << files.size() << '\n'
                << "training_instances=" << train_count << '\n'
                << "validation_instances=" << validation_count << '\n'
                << "test_instances=" << (files.size() - train_count - validation_count) << '\n'
                << "training_samples=" << training_records.size() << '\n'
                << "validation_samples=" << validation_records.size() << '\n'
                << "recorded_tree_nodes=" << total_nodes << '\n';
        std::cout << "CPLEX " << api->version() << " training_samples=" << training_records.size()
                  << " validation_samples=" << validation_records.size() << '\n';
        return 0;
#endif
    } catch (const std::exception& error) {
        std::cerr << "generate_training_data: " << error.what() << '\n';
        return 1;
    }
}
