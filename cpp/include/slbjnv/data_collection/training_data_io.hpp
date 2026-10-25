#pragma once

#include "slbjnv/data_collection/training_record.hpp"

#include <filesystem>
#include <vector>

namespace slbjnv {

class TrainingDataIO {
public:
    static void write_records(const std::vector<TrainingRecord>& records,
                              const std::filesystem::path& path,
                              bool append = false);
    static void write_tree(const SearchTree& tree, const std::string& instance_id,
                           const std::filesystem::path& path);
};

}  // namespace slbjnv
