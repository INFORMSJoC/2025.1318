#pragma once

#include "slbjnv/common/types.hpp"
#include "slbjnv/instance/gbd_instance.hpp"

#include <string>
#include <vector>

namespace slbjnv {

struct BatchColumn {
    ColumnId id{-1};
    MachineId machine{};
    ItemId master{};
    std::vector<ItemId> items;
    double benefit{};

    [[nodiscard]] bool contains(ItemId item_id) const;
    [[nodiscard]] std::string signature() const;
    [[nodiscard]] std::vector<std::string> validate(const GBDInstance& instance) const;
};

}  // namespace slbjnv
