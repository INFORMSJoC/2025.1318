#pragma once

#include "slbjnv/common/types.hpp"

#include <string>
#include <vector>

namespace slbjnv {

struct Item {
    ItemId id{};
    double weight{};
    double profit{};
};

struct Machine {
    MachineId id{};
    double capacity{};
};

class GBDInstance {
public:
    std::string instance_id;
    std::vector<Item> items;
    std::vector<Machine> machines;
    // compatible_with_master[k][i] is true exactly when i belongs to I_k.
    std::vector<std::vector<bool>> compatible_with_master;

    [[nodiscard]] const Item& item(ItemId id) const;
    [[nodiscard]] const Machine& machine(MachineId id) const;
    [[nodiscard]] bool is_compatible(ItemId master, ItemId item_id) const;
    [[nodiscard]] bool has_feasible_non_singleton_batch(ItemId master) const;
    [[nodiscard]] std::size_t feasible_master_item_pair_count() const;
    [[nodiscard]] std::vector<std::string> validate() const;
    void validate_or_throw() const;
};

}  // namespace slbjnv
