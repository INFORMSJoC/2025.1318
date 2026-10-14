#include "slbjnv/instance/gbd_instance.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace slbjnv {

const Item& GBDInstance::item(ItemId id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= items.size()) {
        throw std::out_of_range("invalid item id");
    }
    return items[static_cast<std::size_t>(id)];
}

const Machine& GBDInstance::machine(MachineId id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= machines.size()) {
        throw std::out_of_range("invalid machine id");
    }
    return machines[static_cast<std::size_t>(id)];
}

bool GBDInstance::is_compatible(ItemId master, ItemId item_id) const {
    static_cast<void>(item(master));
    static_cast<void>(item(item_id));
    return compatible_with_master[static_cast<std::size_t>(master)][static_cast<std::size_t>(item_id)];
}

bool GBDInstance::has_feasible_non_singleton_batch(ItemId master) const {
    const auto& master_item = item(master);
    for (const auto& candidate : items) {
        if (candidate.id == master || !is_compatible(master, candidate.id)) continue;
        const auto combined_weight = master_item.weight + candidate.weight;
        if (std::any_of(machines.begin(), machines.end(), [combined_weight](const Machine& value) {
                return combined_weight <= value.capacity + 1.0e-8;
            })) {
            return true;
        }
    }
    return false;
}

std::size_t GBDInstance::feasible_master_item_pair_count() const {
    std::size_t count = 0;
    for (const auto& master : items) {
        for (const auto& candidate : items) {
            if (candidate.id == master.id || !is_compatible(master.id, candidate.id)) continue;
            const auto combined_weight = master.weight + candidate.weight;
            if (std::any_of(machines.begin(), machines.end(), [combined_weight](const Machine& value) {
                    return combined_weight <= value.capacity + 1.0e-8;
                })) {
                ++count;
            }
        }
    }
    return count;
}

std::vector<std::string> GBDInstance::validate() const {
    std::vector<std::string> errors;
    if (instance_id.empty()) errors.emplace_back("instance_id must not be empty");
    if (items.empty()) errors.emplace_back("at least one item is required");
    if (machines.empty()) errors.emplace_back("at least one machine is required");
    if (compatible_with_master.size() != items.size()) {
        errors.emplace_back("compatibility matrix row count must equal item count");
    }
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (items[i].id != static_cast<ItemId>(i)) errors.emplace_back("item ids must be zero-based and contiguous");
        if (items[i].weight <= 0.0) errors.emplace_back("item weights must be positive");
        if (items[i].profit < 0.0) errors.emplace_back("item profits must be nonnegative");
        if (i < compatible_with_master.size()) {
            if (compatible_with_master[i].size() != items.size()) {
                errors.emplace_back("compatibility matrix must be square");
            } else if (!compatible_with_master[i][i]) {
                errors.emplace_back("every master item must be compatible with itself");
            }
        }
    }
    for (std::size_t j = 0; j < machines.size(); ++j) {
        if (machines[j].id != static_cast<MachineId>(j)) errors.emplace_back("machine ids must be zero-based and contiguous");
        if (machines[j].capacity <= 0.0) errors.emplace_back("machine capacities must be positive");
    }
    if (!items.empty() && !machines.empty()) {
        const auto maximum_capacity = std::max_element(
            machines.begin(), machines.end(),
            [](const Machine& left, const Machine& right) { return left.capacity < right.capacity; })->capacity;
        for (const auto& value : items) {
            if (value.weight > maximum_capacity + 1.0e-8) {
                errors.emplace_back("every item must fit on at least one machine");
            }
        }
    }
    return errors;
}

void GBDInstance::validate_or_throw() const {
    const auto errors = validate();
    if (errors.empty()) return;
    std::ostringstream message;
    message << "invalid GBD instance:";
    for (const auto& error : errors) message << "\n- " << error;
    throw std::invalid_argument(message.str());
}

}  // namespace slbjnv
