#include "slbjnv/instance/instance_io.hpp"

#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace slbjnv {
namespace {

void expect(std::istream& input, const char* expected) {
    std::string token;
    if (!(input >> token) || token != expected) {
        throw std::invalid_argument(std::string("expected token '") + expected + "'");
    }
}

template <typename T>
T read_value(std::istream& input, const char* description) {
    T value{};
    if (!(input >> value)) throw std::invalid_argument(std::string("invalid ") + description);
    return value;
}

}  // namespace

void GBDInstanceIO::write(const GBDInstance& instance, std::ostream& output) {
    instance.validate_or_throw();
    output << kFormatMagic << '\n';
    output << "instance_id " << std::quoted(instance.instance_id) << '\n';
    output << "num_items " << instance.items.size() << '\n';
    output << "num_machines " << instance.machines.size() << '\n';
    output << "items\n";
    output << std::setprecision(17);
    for (const auto& item : instance.items) output << item.id << ' ' << item.weight << ' ' << item.profit << '\n';
    output << "machines\n";
    for (const auto& machine : instance.machines) output << machine.id << ' ' << machine.capacity << '\n';
    output << "compatibility\n";
    for (const auto& row : instance.compatible_with_master) {
        for (const auto value : row) output << (value ? '1' : '0');
        output << '\n';
    }
    output << "end\n";
    if (!output) throw std::runtime_error("failed while writing GBD instance");
}

void GBDInstanceIO::write_file(const GBDInstance& instance, const std::filesystem::path& path) {
    if (path.has_parent_path()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::out | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot create instance file: " + path.string());
    write(instance, output);
}

GBDInstance GBDInstanceIO::read(std::istream& input) {
    expect(input, kFormatMagic);
    expect(input, "instance_id");
    GBDInstance instance;
    if (!(input >> std::quoted(instance.instance_id))) throw std::invalid_argument("invalid instance_id");
    expect(input, "num_items");
    const auto item_count = read_value<std::size_t>(input, "num_items");
    expect(input, "num_machines");
    const auto machine_count = read_value<std::size_t>(input, "num_machines");
    if (item_count == 0 || item_count > static_cast<std::size_t>(std::numeric_limits<ItemId>::max())) {
        throw std::invalid_argument("num_items is outside the supported range");
    }
    if (machine_count == 0 || machine_count > static_cast<std::size_t>(std::numeric_limits<MachineId>::max())) {
        throw std::invalid_argument("num_machines is outside the supported range");
    }

    expect(input, "items");
    instance.items.reserve(item_count);
    for (std::size_t i = 0; i < item_count; ++i) {
        instance.items.push_back({read_value<ItemId>(input, "item id"),
                                  read_value<double>(input, "item weight"),
                                  read_value<double>(input, "item profit")});
    }
    expect(input, "machines");
    instance.machines.reserve(machine_count);
    for (std::size_t j = 0; j < machine_count; ++j) {
        instance.machines.push_back({read_value<MachineId>(input, "machine id"),
                                     read_value<double>(input, "machine capacity")});
    }
    expect(input, "compatibility");
    instance.compatible_with_master.assign(item_count, std::vector<bool>(item_count));
    for (std::size_t master = 0; master < item_count; ++master) {
        const auto row = read_value<std::string>(input, "compatibility row");
        if (row.size() != item_count) throw std::invalid_argument("compatibility row has incorrect length");
        for (std::size_t item = 0; item < item_count; ++item) {
            if (row[item] != '0' && row[item] != '1') {
                throw std::invalid_argument("compatibility matrix contains a non-binary value");
            }
            instance.compatible_with_master[master][item] = row[item] == '1';
        }
    }
    expect(input, "end");
    std::string trailing;
    if (input >> trailing) throw std::invalid_argument("unexpected data after end marker");
    instance.validate_or_throw();
    return instance;
}

GBDInstance GBDInstanceIO::read_file(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open instance file: " + path.string());
    try {
        return read(input);
    } catch (const std::exception& error) {
        throw std::invalid_argument("invalid instance file " + path.string() + ": " + error.what());
    }
}

}  // namespace slbjnv
