#include "slbjnv/instance/instance_generator.hpp"
#include "slbjnv/instance/instance_io.hpp"

#include <cassert>
#include <sstream>

int main() {
    using namespace slbjnv;
    InstanceGeneratorConfig config;
    config.seed = 123456789ULL;
    config.instances_per_size = 1;
    config.sizes = {{3, 12}};
    config.compatibility_probability = 0.35;
    config.validate_or_throw();

    GBDInstanceGenerator first_generator(config);
    const auto instance = first_generator.generate("round_trip", config.sizes.front());
    assert(instance.validate().empty());
    assert(instance.items.size() == 12);
    assert(instance.machines.size() == 3);
    for (const auto& item : instance.items) {
        assert(item.weight >= config.weight_range.minimum && item.weight <= config.weight_range.maximum);
        assert(item.profit >= config.profit_range.minimum && item.profit <= config.profit_range.maximum);
        assert(instance.is_compatible(item.id, item.id));
        assert(instance.has_feasible_non_singleton_batch(item.id));
    }
    for (const auto& machine : instance.machines) {
        assert(machine.capacity >= config.capacity_range.minimum && machine.capacity <= config.capacity_range.maximum);
    }
    assert(instance.feasible_master_item_pair_count() >= instance.items.size());

    std::stringstream serialized;
    GBDInstanceIO::write(instance, serialized);
    const auto restored = GBDInstanceIO::read(serialized);
    assert(restored.instance_id == instance.instance_id);
    assert(restored.items.size() == instance.items.size());
    assert(restored.machines.size() == instance.machines.size());
    assert(restored.compatible_with_master == instance.compatible_with_master);
    for (std::size_t i = 0; i < instance.items.size(); ++i) {
        assert(restored.items[i].weight == instance.items[i].weight);
        assert(restored.items[i].profit == instance.items[i].profit);
    }

    GBDInstanceGenerator second_generator(config);
    const auto repeated = second_generator.generate("round_trip", config.sizes.front());
    assert(repeated.compatible_with_master == instance.compatible_with_master);
    for (std::size_t i = 0; i < repeated.items.size(); ++i) {
        assert(repeated.items[i].weight == instance.items[i].weight);
        assert(repeated.items[i].profit == instance.items[i].profit);
    }
    return 0;
}
