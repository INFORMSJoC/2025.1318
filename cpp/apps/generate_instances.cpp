#include "slbjnv/instance/instance_generator.hpp"
#include "slbjnv/instance/instance_io.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

struct Arguments {
    std::filesystem::path config{"configs/instance_generator.yaml"};
    std::optional<std::uint64_t> seed;
    std::optional<std::filesystem::path> output_directory;
    bool help{};
};

Arguments parse_arguments(int argc, char** argv) {
    Arguments result;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto next = [&]() -> std::string {
            if (++index >= argc) throw std::invalid_argument("missing value after " + argument);
            return argv[index];
        };
        if (argument == "--config") result.config = next();
        else if (argument == "--seed") result.seed = std::stoull(next());
        else if (argument == "--output") result.output_directory = next();
        else if (argument == "--help" || argument == "-h") result.help = true;
        else throw std::invalid_argument("unknown argument: " + argument);
    }
    return result;
}

void print_help() {
    std::cout << "Usage: generate_instances [--config FILE] [--seed UINT64] [--output DIRECTORY]\n";
}

std::string instance_name(const slbjnv::InstanceSize& size, std::size_t index, std::size_t count) {
    const auto width = std::max<std::size_t>(2, std::to_string(count).size());
    std::ostringstream value;
    value << "gbd_" << size.machines << '_' << size.items << '_' << std::setw(static_cast<int>(width))
          << std::setfill('0') << index;
    return value.str();
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto arguments = parse_arguments(argc, argv);
        if (arguments.help) { print_help(); return 0; }
        auto config = slbjnv::InstanceGeneratorConfig::from_yaml_file(arguments.config);
        if (arguments.seed) config.seed = arguments.seed;
        if (arguments.output_directory) config.output_directory = *arguments.output_directory;
        slbjnv::GBDInstanceGenerator generator(config);
        std::filesystem::create_directories(config.output_directory);

        std::size_t generated = 0;
        for (const auto& size : config.sizes) {
            for (std::size_t index = 1; index <= config.instances_per_size; ++index) {
                const auto name = instance_name(size, index, config.instances_per_size);
                const auto path = config.output_directory / (name + ".txt");
                const auto instance = generator.generate(name, size);
                slbjnv::GBDInstanceIO::write_file(instance, path);
                const auto verified = slbjnv::GBDInstanceIO::read_file(path);
                if (verified.instance_id != name || verified.items.size() != size.items ||
                    verified.machines.size() != size.machines) {
                    throw std::runtime_error("round-trip verification failed for " + path.string());
                }
                ++generated;
                std::cout << path.string() << " pairs=" << verified.feasible_master_item_pair_count() << '\n';
            }
        }
        std::cout << "Generated " << generated << " instances with seed " << generator.seed() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "generate_instances: " << error.what() << '\n';
        return 1;
    }
}
