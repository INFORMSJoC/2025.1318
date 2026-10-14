#pragma once

#include "slbjnv/instance/gbd_instance.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <random>
#include <string>
#include <vector>

namespace slbjnv {

struct InstanceSize {
    std::size_t machines{};
    std::size_t items{};
};

struct IntegerRange {
    int minimum{};
    int maximum{};
};

struct InstanceGeneratorConfig {
    std::optional<std::uint64_t> seed;
    std::size_t instances_per_size{10};
    std::vector<InstanceSize> sizes;
    IntegerRange weight_range{10, 45};
    IntegerRange profit_range{5, 30};
    IntegerRange capacity_range{80, 180};
    double compatibility_probability{0.65};
    std::filesystem::path output_directory{"data/instances"};

    [[nodiscard]] std::vector<std::string> validate() const;
    void validate_or_throw() const;
    [[nodiscard]] static InstanceGeneratorConfig from_yaml_file(const std::filesystem::path& path);
};

class GBDInstanceGenerator {
public:
    explicit GBDInstanceGenerator(InstanceGeneratorConfig config);

    [[nodiscard]] const InstanceGeneratorConfig& config() const noexcept { return config_; }
    [[nodiscard]] std::uint64_t seed() const noexcept { return seed_; }
    [[nodiscard]] GBDInstance generate(std::string instance_id, const InstanceSize& size);

private:
    InstanceGeneratorConfig config_;
    std::uint64_t seed_{};
    std::mt19937_64 random_;
};

}  // namespace slbjnv
