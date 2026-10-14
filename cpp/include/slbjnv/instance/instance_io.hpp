#pragma once

#include "slbjnv/instance/gbd_instance.hpp"

#include <filesystem>
#include <iosfwd>

namespace slbjnv {

class GBDInstanceIO {
public:
    static constexpr const char* kFormatMagic = "SLBJNV_GBD_INSTANCE_V1";

    static void write(const GBDInstance& instance, std::ostream& output);
    static void write_file(const GBDInstance& instance, const std::filesystem::path& path);
    [[nodiscard]] static GBDInstance read(std::istream& input);
    [[nodiscard]] static GBDInstance read_file(const std::filesystem::path& path);
};

}  // namespace slbjnv
