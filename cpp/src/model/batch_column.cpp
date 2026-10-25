#include "slbjnv/model/batch_column.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_set>

namespace slbjnv {

bool BatchColumn::contains(ItemId item_id) const {
    return std::find(items.begin(), items.end(), item_id) != items.end();
}

std::string BatchColumn::signature() const {
    auto sorted = items;
    std::sort(sorted.begin(), sorted.end());
    std::ostringstream stream;
    stream << machine << ':' << master << ':';
    for (const auto value : sorted) stream << value << ',';
    return stream.str();
}

std::vector<std::string> BatchColumn::validate(const GBDInstance& instance) const {
    std::vector<std::string> errors;
    try { static_cast<void>(instance.machine(machine)); }
    catch (...) { errors.emplace_back("unknown machine"); return errors; }
    try { static_cast<void>(instance.item(master)); }
    catch (...) { errors.emplace_back("unknown master item"); return errors; }
    if (!contains(master)) errors.emplace_back("batch must contain its master item");
    std::unordered_set<ItemId> unique;
    double weight = 0.0;
    double profit = 0.0;
    for (const auto item_id : items) {
        try {
            const auto& value = instance.item(item_id);
            if (!unique.insert(item_id).second) errors.emplace_back("batch contains duplicate item");
            if (!instance.is_compatible(master, item_id)) errors.emplace_back("item is incompatible with master");
            weight += value.weight;
            profit += value.profit;
        } catch (...) { errors.emplace_back("batch contains unknown item"); }
    }
    if (weight > instance.machine(machine).capacity + 1.0e-8) errors.emplace_back("batch exceeds machine capacity");
    if (std::abs(benefit - profit) > 1.0e-8) errors.emplace_back("GBD batch benefit must equal sum of item profits");
    return errors;
}

}  // namespace slbjnv
