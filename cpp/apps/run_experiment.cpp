#include "slbjnv/instance/instance_io.hpp"

#ifdef SLBJNV_WITH_CPLEX
#include "slbjnv/branch_and_price/default_branch_and_price.hpp"
#include "slbjnv/data_collection/training_data_collector.hpp"
#include "slbjnv/data_collection/training_data_io.hpp"
#include "slbjnv/model/cplex_c_api.hpp"
#endif
#ifdef SLBJNV_WITH_PYTHON
#include "slbjnv/python_interface/predictor.hpp"
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Arguments {
    std::filesystem::path instances;
    std::string mode{"baseline"};
    std::filesystem::path model;
    std::filesystem::path python_path{std::filesystem::path(SLBJNV_SOURCE_DIR) / "python"};
    std::filesystem::path config{"configs/branch_and_price.yaml"};
    std::filesystem::path output_directory{"results"};
    std::filesystem::path reference_file;
    std::string device{"cpu"};
    std::optional<double> reference_objective;
    std::size_t limit{};
};

std::string take_value(int& index, int argc, char** argv, const char* option) {
    if (index + 1 >= argc) throw std::invalid_argument(std::string(option) + " requires a value");
    return argv[++index];
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments result;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (argument == "--instance" || argument == "--instances")
            result.instances = take_value(i, argc, argv, argument.c_str());
        else if (argument == "--mode") result.mode = take_value(i, argc, argv, "--mode");
        else if (argument == "--model") result.model = take_value(i, argc, argv, "--model");
        else if (argument == "--python-path") result.python_path = take_value(i, argc, argv, "--python-path");
        else if (argument == "--device") result.device = take_value(i, argc, argv, "--device");
        else if (argument == "--config") result.config = take_value(i, argc, argv, "--config");
        else if (argument == "--output-dir") result.output_directory = take_value(i, argc, argv, "--output-dir");
        else if (argument == "--reference-file") result.reference_file = take_value(i, argc, argv, "--reference-file");
        else if (argument == "--reference-objective")
            result.reference_objective = std::stod(take_value(i, argc, argv, "--reference-objective"));
        else if (argument == "--limit") result.limit = std::stoull(take_value(i, argc, argv, "--limit"));
        else if (argument == "--help") {
            std::cout << "Usage: run_experiment --instances FILE_OR_DIR "
                         "[--mode baseline|collect|slbjnv|all] [--model PATH] "
                         "[--device cpu|cuda] [--config PATH] [--output-dir DIR] "
                         "[--reference-file FILE] [--reference-objective VALUE] [--limit N]\n";
            std::exit(0);
        } else throw std::invalid_argument("unknown argument: " + argument);
    }
    if (result.instances.empty()) throw std::invalid_argument("--instances is required");
    if (result.mode != "baseline" && result.mode != "collect"
        && result.mode != "slbjnv" && result.mode != "all") {
        throw std::invalid_argument("--mode must be baseline, collect, slbjnv, or all");
    }
    if ((result.mode == "slbjnv" || result.mode == "all") && result.model.empty()) {
        throw std::invalid_argument("--model is required for SLbJNV experiments");
    }
    return result;
}

std::vector<std::filesystem::path> instance_files(const std::filesystem::path& input,
                                                  std::size_t limit) {
    std::vector<std::filesystem::path> files;
    if (std::filesystem::is_regular_file(input)) files.push_back(input);
    else if (std::filesystem::is_directory(input)) {
        for (const auto& entry : std::filesystem::directory_iterator(input)) {
            if (entry.is_regular_file() && entry.path().extension() == ".txt") {
                files.push_back(entry.path());
            }
        }
        std::sort(files.begin(), files.end());
    } else throw std::invalid_argument("instance input does not exist: " + input.string());
    if (limit > 0 && files.size() > limit) files.resize(limit);
    if (files.empty()) throw std::invalid_argument("no instance TXT files found");
    return files;
}

std::map<std::string, double> read_references(const std::filesystem::path& path) {
    std::map<std::string, double> result;
    if (path.empty()) return result;
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open reference file: " + path.string());
    std::string line;
    while (std::getline(input, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        std::istringstream parser(line);
        std::string id;
        double objective{};
        if (parser >> id >> objective) result[id] = objective;
    }
    return result;
}

#ifdef SLBJNV_WITH_CPLEX
std::pair<slbjnv::BranchAndPriceConfig, slbjnv::StrongBranchingConfig>
read_solver_config(const std::filesystem::path& path) {
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
        const auto key = line.substr(first, colon - first);
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

const char* status_name(slbjnv::SolveStatus status) {
    switch (status) {
        case slbjnv::SolveStatus::optimal: return "optimal";
        case slbjnv::SolveStatus::infeasible: return "infeasible";
        case slbjnv::SolveStatus::unbounded: return "unbounded";
        case slbjnv::SolveStatus::time_limit: return "time_limit";
        case slbjnv::SolveStatus::numerical_error: return "numerical_error";
        case slbjnv::SolveStatus::backend_unavailable: return "backend_unavailable";
        default: return "not_solved";
    }
}

double relative_gap(const slbjnv::BranchAndPriceResult& result) {
    if (!std::isfinite(result.best_integer_objective)
        || !std::isfinite(result.global_upper_bound)) return std::numeric_limits<double>::infinity();
    return std::max(0.0, result.global_upper_bound - result.best_integer_objective)
        / std::max(1.0, std::abs(result.best_integer_objective));
}

struct ReferenceAttainment {
    bool attained{};
    slbjnv::NodeId node{slbjnv::kNoNode};
    double time{};
};

ReferenceAttainment reference_attainment(const slbjnv::BranchAndPriceResult& result,
                                         const std::optional<double>& reference,
                                         double tolerance) {
    if (!reference) return {};
    for (const auto& snapshot : result.incumbent_history) {
        if (snapshot.objective >= *reference - tolerance) {
            return {true, snapshot.node_id, snapshot.elapsed_seconds};
        }
    }
    return {};
}

void write_histories(const std::filesystem::path& directory, const std::string& instance_id,
                     const std::string& method, const slbjnv::BranchAndPriceResult& result) {
    const auto history_directory = directory / "histories";
    std::filesystem::create_directories(history_directory);
    std::ofstream incumbents(history_directory / (instance_id + "_" + method + "_incumbent.csv"));
    incumbents << "node_id,elapsed_seconds,objective\n" << std::setprecision(15);
    for (const auto& value : result.incumbent_history) {
        incumbents << value.node_id << ',' << value.elapsed_seconds << ',' << value.objective << '\n';
    }
    std::ofstream bounds(history_directory / (instance_id + "_" + method + "_bounds.csv"));
    bounds << "explored_nodes,elapsed_seconds,lower_bound,upper_bound\n" << std::setprecision(15);
    for (const auto& value : result.bound_history) {
        bounds << value.explored_nodes << ',' << value.elapsed_seconds << ',';
        if (std::isfinite(value.lower_bound)) bounds << value.lower_bound;
        bounds << ',';
        if (std::isfinite(value.upper_bound)) bounds << value.upper_bound;
        bounds << '\n';
    }
}

void write_result(std::ofstream& csv, std::ofstream& text, const std::filesystem::path& output_directory,
                  const std::string& instance_id, const std::string& method,
                  const slbjnv::BranchAndPriceResult& result,
                  const std::optional<double>& reference, double tolerance) {
    const auto attainment = reference_attainment(result, reference, tolerance);
    const auto certified = result.status == slbjnv::SolveStatus::optimal;
    const auto gap = relative_gap(result);
    csv << std::setprecision(15) << instance_id << ',' << method << ',' << status_name(result.status) << ',';
    if (std::isfinite(result.best_integer_objective)) csv << result.best_integer_objective;
    csv << ',';
    if (std::isfinite(result.global_upper_bound)) csv << result.global_upper_bound;
    csv << ',';
    if (std::isfinite(gap)) csv << gap;
    csv << ',' << result.explored_nodes << ',' << result.generated_nodes << ','
        << result.total_runtime_seconds << ',' << result.column_generation_iterations << ','
        << result.generated_columns << ',' << result.branching_decisions;
    for (const auto count : result.branching_rule_counts) csv << ',' << count;
    csv << ',' << result.predicted_pairs << ',' << result.inference_seconds << ',';
    if (reference) csv << *reference;
    csv << ',' << (attainment.attained ? "true" : "false") << ',';
    if (attainment.attained) csv << attainment.node;
    csv << ',';
    if (attainment.attained) csv << attainment.time;
    csv << ',' << (certified ? "true" : "false") << '\n';

    text << "instance_id=" << instance_id << '\n' << "method=" << method << '\n'
         << "status=" << status_name(result.status) << '\n'
         << "best_integer_objective=" << result.best_integer_objective << '\n'
         << "global_lower_bound=" << result.best_integer_objective << '\n'
         << "global_upper_bound=" << result.global_upper_bound << '\n'
         << "optimality_gap=" << gap << '\n'
         << "explored_nodes=" << result.explored_nodes << '\n'
         << "generated_nodes=" << result.generated_nodes << '\n'
         << "runtime_seconds=" << result.total_runtime_seconds << '\n'
         << "column_generation_iterations=" << result.column_generation_iterations << '\n'
         << "generated_columns=" << result.generated_columns << '\n'
         << "branching_decisions=" << result.branching_decisions << '\n';
    for (std::size_t i = 0; i < result.branching_rule_counts.size(); ++i) {
        text << "branching_rule_" << (i + 1) << '=' << result.branching_rule_counts[i] << '\n';
    }
    text << "predicted_pairs=" << result.predicted_pairs << '\n'
         << "inference_seconds=" << result.inference_seconds << '\n'
         << "reference_attained=" << (attainment.attained ? "true" : "false") << '\n'
         << "first_reference_node=" << attainment.node << '\n'
         << "first_reference_time=" << attainment.time << '\n'
         << "optimality_certified=" << (certified ? "true" : "false") << "\n\n";
    write_histories(output_directory, instance_id, method, result);
}
#endif

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
#ifndef SLBJNV_WITH_CPLEX
        throw std::runtime_error("run_experiment was built without CPLEX support");
#else
        const auto files = instance_files(arguments.instances, arguments.limit);
        const auto references = read_references(arguments.reference_file);
        const auto [solve_config, strong_config] = read_solver_config(arguments.config);
        const auto api = slbjnv::CplexCApi::load();
        std::shared_ptr<slbjnv::ScorePredictor> predictor;
        if (arguments.mode == "slbjnv" || arguments.mode == "all") {
#ifndef SLBJNV_WITH_PYTHON
            throw std::runtime_error("run_experiment was built without embedded Python support");
#else
            auto python_predictor = std::make_shared<slbjnv::PythonScorePredictor>(
                arguments.python_path, arguments.device);
            python_predictor->load_model(arguments.model);
            predictor = std::move(python_predictor);
#endif
        }

        std::filesystem::create_directories(arguments.output_directory);
        std::ofstream csv(arguments.output_directory / "experiment_results.csv", std::ios::trunc);
        std::ofstream text(arguments.output_directory / "experiment_results.txt", std::ios::trunc);
        if (!csv || !text) throw std::runtime_error("cannot create experiment result files");
        csv << "instance_id,method,status,best_integer_objective,global_upper_bound,optimality_gap,"
               "explored_nodes,generated_nodes,runtime_seconds,column_generation_iterations,"
               "generated_columns,branching_decisions,rule1,rule2,rule3,rule4,rule5,"
               "predicted_pairs,inference_seconds,reference_objective,reference_attained,"
               "first_reference_node,first_reference_time,optimality_certified\n";

        std::vector<std::string> methods;
        if (arguments.mode == "all") methods = {"baseline", "slbjnv"};
        else methods = {arguments.mode};
        for (const auto& file : files) {
            const auto instance = slbjnv::GBDInstanceIO::read_file(file);
            std::optional<double> reference = arguments.reference_objective;
            const auto found_reference = references.find(instance.instance_id);
            if (found_reference != references.end()) reference = found_reference->second;
            for (const auto& method : methods) {
                slbjnv::BranchAndPriceResult result;
                if (method == "collect") {
                    slbjnv::TrainingDataCollector collector(api, solve_config, strong_config);
                    auto collected = collector.collect(instance);
                    result = std::move(collected.solve_result);
                    const auto training_directory = arguments.output_directory / "training";
                    slbjnv::TrainingDataIO::write_records(
                        collected.records, training_directory / (instance.instance_id + ".txt"));
                    slbjnv::TrainingDataIO::write_tree(
                        result.tree, instance.instance_id,
                        training_directory / "trees" / (instance.instance_id + ".txt"));
                } else {
                    slbjnv::DefaultBranchAndPrice solver(api, solve_config, predictor);
                    result = solver.solve(instance, method == "slbjnv"
                        ? slbjnv::SearchMode::slbjnv : slbjnv::SearchMode::baseline);
                }
                write_result(csv, text, arguments.output_directory, instance.instance_id,
                             method, result, reference, solve_config.tolerances.bound);
                std::cout << instance.instance_id << ' ' << method
                          << " status=" << status_name(result.status)
                          << " objective=" << result.best_integer_objective
                          << " nodes=" << result.explored_nodes
                          << " seconds=" << result.total_runtime_seconds << '\n';
            }
        }
        std::cout << "CPLEX " << api->version() << " results="
                  << (arguments.output_directory / "experiment_results.csv") << '\n';
        return 0;
#endif
    } catch (const std::exception& error) {
        std::cerr << "run_experiment: " << error.what() << '\n';
        return 1;
    }
}
