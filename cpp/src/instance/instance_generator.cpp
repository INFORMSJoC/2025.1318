#include "slbjnv/instance/instance_generator.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace slbjnv {
namespace {

std::string trim(std::string value) {
    const auto first = std::find_if_not(value.begin(), value.end(),
                                        [](unsigned char ch) { return std::isspace(ch) != 0; });
    const auto last = std::find_if_not(value.rbegin(), value.rend(),
                                       [](unsigned char ch) { return std::isspace(ch) != 0; }).base();
    return first < last ? std::string(first, last) : std::string{};
}

std::string scalar_value(const std::string& line, const std::string& key) {
    const auto prefix = key + ':';
    if (line.rfind(prefix, 0) != 0) throw std::invalid_argument("expected YAML key: " + key);
    return trim(line.substr(prefix.size()));
}

IntegerRange parse_range(const std::string& text, const std::string& key) {
    static const std::regex expression(R"(^\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]$)");
    std::smatch match;
    if (!std::regex_match(text, match, expression)) throw std::invalid_argument("invalid range for " + key);
    return {std::stoi(match[1].str()), std::stoi(match[2].str())};
}

std::uint64_t random_seed() {
    std::random_device device;
    const auto clock_value = static_cast<std::uint64_t>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    return (static_cast<std::uint64_t>(device()) << 32U) ^ static_cast<std::uint64_t>(device()) ^ clock_value;
}

}  // namespace

std::vector<std::string> InstanceGeneratorConfig::validate() const {
    std::vector<std::string> errors;
    if (instances_per_size == 0) errors.emplace_back("instances_per_size must be positive");
    if (sizes.empty()) errors.emplace_back("at least one instance size is required");
    for (const auto& size : sizes) {
        if (size.machines == 0) errors.emplace_back("each size must contain at least one machine");
        if (size.items < 2) errors.emplace_back("each generated instance must contain at least two items");
    }
    const auto valid_range = [&errors](const IntegerRange& range, const char* name, bool allow_zero) {
        const auto lower_bound = allow_zero ? 0 : 1;
        if (range.minimum < lower_bound || range.maximum < range.minimum) {
            errors.emplace_back(std::string(name) + " is invalid");
        }
    };
    valid_range(weight_range, "weight_range", false);
    valid_range(profit_range, "profit_range", true);
    valid_range(capacity_range, "capacity_range", false);
    if (capacity_range.maximum < 2 * weight_range.minimum) {
        errors.emplace_back("capacity_range maximum must permit at least one two-item batch");
    }
    if (compatibility_probability < 0.0 || compatibility_probability > 1.0) {
        errors.emplace_back("compatibility_probability must be in [0, 1]");
    }
    if (output_directory.empty()) errors.emplace_back("output_directory must not be empty");
    return errors;
}

void InstanceGeneratorConfig::validate_or_throw() const {
    const auto errors = validate();
    if (errors.empty()) return;
    std::ostringstream message;
    message << "invalid instance generator configuration:";
    for (const auto& error : errors) message << "\n- " << error;
    throw std::invalid_argument(message.str());
}

InstanceGeneratorConfig InstanceGeneratorConfig::from_yaml_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open generator config: " + path.string());

    InstanceGeneratorConfig config;
    config.sizes.clear();
    std::string raw;
    std::size_t line_number = 0;
    bool in_sizes = false;
    static const std::regex size_expression(
        R"(^-\s*\{\s*machines\s*:\s*(\d+)\s*,\s*items\s*:\s*(\d+)\s*\}\s*$)");
    while (std::getline(input, raw)) {
        ++line_number;
        const auto comment = raw.find('#');
        const auto line = trim(raw.substr(0, comment));
        if (line.empty()) continue;
        try {
            if (line == "sizes:") { in_sizes = true; continue; }
            if (in_sizes && line.rfind('-', 0) == 0) {
                std::smatch match;
                if (!std::regex_match(line, match, size_expression)) {
                    throw std::invalid_argument("invalid inline size mapping");
                }
                config.sizes.push_back({static_cast<std::size_t>(std::stoull(match[1].str())),
                                        static_cast<std::size_t>(std::stoull(match[2].str()))});
                continue;
            }
            in_sizes = false;
            if (line.rfind("seed:", 0) == 0) {
                const auto value = scalar_value(line, "seed");
                config.seed = value == "null" ? std::nullopt
                                                : std::optional<std::uint64_t>(std::stoull(value));
            } else if (line.rfind("instances_per_size:", 0) == 0) {
                config.instances_per_size = static_cast<std::size_t>(
                    std::stoull(scalar_value(line, "instances_per_size")));
            } else if (line.rfind("weight_range:", 0) == 0) {
                config.weight_range = parse_range(scalar_value(line, "weight_range"), "weight_range");
            } else if (line.rfind("profit_range:", 0) == 0) {
                config.profit_range = parse_range(scalar_value(line, "profit_range"), "profit_range");
            } else if (line.rfind("capacity_range:", 0) == 0) {
                config.capacity_range = parse_range(scalar_value(line, "capacity_range"), "capacity_range");
            } else if (line.rfind("compatibility_probability:", 0) == 0) {
                config.compatibility_probability = std::stod(scalar_value(line, "compatibility_probability"));
            } else if (line.rfind("output_directory:", 0) == 0) {
                auto value = scalar_value(line, "output_directory");
                if (value.size() >= 2 && ((value.front() == '"' && value.back() == '"') ||
                                          (value.front() == '\'' && value.back() == '\''))) {
                    value = value.substr(1, value.size() - 2);
                }
                config.output_directory = value;
            } else {
                throw std::invalid_argument("unknown YAML key");
            }
        } catch (const std::exception& error) {
            throw std::invalid_argument(path.string() + ":" + std::to_string(line_number) + ": " + error.what());
        }
    }
    config.validate_or_throw();
    return config;
}

GBDInstanceGenerator::GBDInstanceGenerator(InstanceGeneratorConfig config)
    : config_(std::move(config)), seed_(config_.seed.value_or(random_seed())), random_(seed_) {
    config_.validate_or_throw();
}

GBDInstance GBDInstanceGenerator::generate(std::string instance_id, const InstanceSize& size) {
    if (size.machines == 0 || size.items < 2) throw std::invalid_argument("invalid requested instance size");
    if (size.items > static_cast<std::size_t>(std::numeric_limits<ItemId>::max()) ||
        size.machines > static_cast<std::size_t>(std::numeric_limits<MachineId>::max())) {
        throw std::invalid_argument("requested instance size exceeds supported identifier range");
    }
    std::uniform_int_distribution<int> weight(config_.weight_range.minimum, config_.weight_range.maximum);
    std::uniform_int_distribution<int> profit(config_.profit_range.minimum, config_.profit_range.maximum);
    std::uniform_int_distribution<int> capacity(config_.capacity_range.minimum, config_.capacity_range.maximum);
    std::bernoulli_distribution compatible(config_.compatibility_probability);

    GBDInstance instance;
    instance.instance_id = std::move(instance_id);
    // Rejection sampling preserves the configured uniform draws while ensuring
    // every master has at least one capacity-feasible partner candidate.
    bool structurally_feasible = false;
    for (std::size_t attempt = 0; attempt < 10000 && !structurally_feasible; ++attempt) {
        instance.items.clear();
        instance.machines.clear();
        instance.items.reserve(size.items);
        for (std::size_t i = 0; i < size.items; ++i) {
            instance.items.push_back({static_cast<ItemId>(i), static_cast<double>(weight(random_)),
                                      static_cast<double>(profit(random_))});
        }
        instance.machines.reserve(size.machines);
        for (std::size_t j = 0; j < size.machines; ++j) {
            instance.machines.push_back({static_cast<MachineId>(j), static_cast<double>(capacity(random_))});
        }
        const auto maximum_capacity = std::max_element(
            instance.machines.begin(), instance.machines.end(),
            [](const Machine& left, const Machine& right) { return left.capacity < right.capacity; })->capacity;
        structurally_feasible = std::all_of(instance.items.begin(), instance.items.end(),
            [&instance, maximum_capacity](const Item& master) {
                return std::any_of(instance.items.begin(), instance.items.end(),
                    [&master, maximum_capacity](const Item& candidate) {
                        return master.id != candidate.id && master.weight + candidate.weight <= maximum_capacity;
                    });
            });
    }
    if (!structurally_feasible) {
        throw std::runtime_error("could not sample a non-degenerate instance within 10000 attempts");
    }

    instance.compatible_with_master.assign(size.items, std::vector<bool>(size.items, false));
    for (std::size_t master = 0; master < size.items; ++master) {
        for (std::size_t item = 0; item < size.items; ++item) {
            instance.compatible_with_master[master][item] = master == item || compatible(random_);
        }
    }

    // Avoid singleton-only master sets even if the configured probability is very small.
    for (std::size_t master = 0; master < size.items; ++master) {
        if (instance.has_feasible_non_singleton_batch(static_cast<ItemId>(master))) continue;
        const auto maximum_capacity = std::max_element(
            instance.machines.begin(), instance.machines.end(),
            [](const Machine& left, const Machine& right) { return left.capacity < right.capacity; })->capacity;
        const auto partner = std::find_if(instance.items.begin(), instance.items.end(),
            [&instance, master, maximum_capacity](const Item& candidate) {
                return candidate.id != static_cast<ItemId>(master) &&
                       instance.items[master].weight + candidate.weight <= maximum_capacity;
            });
        if (partner == instance.items.end()) throw std::logic_error("missing feasible partner after sampling");
        instance.compatible_with_master[master][static_cast<std::size_t>(partner->id)] = true;
    }

    instance.validate_or_throw();
    for (const auto& master : instance.items) {
        if (!instance.has_feasible_non_singleton_batch(master.id)) {
            throw std::logic_error("generator failed to create a non-singleton batch for a master item");
        }
    }
    return instance;
}

}  // namespace slbjnv
