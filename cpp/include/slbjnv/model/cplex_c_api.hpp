#pragma once

#include <memory>
#include <string>

namespace slbjnv {

// Opaque CPLEX handles. Keeping the C API behind this adapter avoids the
// incompatible MSVC C++ ABI used by Concert on Windows and works with MinGW.
using CplexEnv = void*;
using CplexProblem = void*;

class CplexCApi {
public:
    static std::shared_ptr<CplexCApi> load(const std::string& explicit_dll = {});
    ~CplexCApi();

    CplexCApi(const CplexCApi&) = delete;
    CplexCApi& operator=(const CplexCApi&) = delete;

    [[nodiscard]] const std::string& version() const noexcept { return version_; }
    [[nodiscard]] CplexEnv environment() const noexcept { return environment_; }

    CplexProblem create_problem(const char* name) const;
    void free_problem(CplexProblem& problem) const noexcept;
    void check(int status, const char* operation) const;

    int (*chgobjsen)(CplexEnv, CplexProblem, int){};
    int (*newrows)(CplexEnv, CplexProblem, int, const double*, const char*, const double*, char**){};
    int (*newcols)(CplexEnv, CplexProblem, int, const double*, const double*, const double*, const char*, char**){};
    int (*chgcoeflist)(CplexEnv, CplexProblem, int, const int*, const int*, const double*){};
    int (*lpopt)(CplexEnv, CplexProblem){};
    int (*mipopt)(CplexEnv, CplexProblem){};
    int (*getstat)(CplexEnv, CplexProblem){};
    int (*getobjval)(CplexEnv, CplexProblem, double*){};
    int (*getx)(CplexEnv, CplexProblem, double*, int, int){};
    int (*getpi)(CplexEnv, CplexProblem, double*, int, int){};
    int (*chgctype)(CplexEnv, CplexProblem, int, const int*, const char*){};
    int (*setintparam)(CplexEnv, int, int){};
    int (*setdblparam)(CplexEnv, int, double){};

private:
    CplexCApi() = default;
    void* library_{};
    CplexEnv environment_{};
    std::string version_;
    int (*close_environment_)(CplexEnv*){};
    CplexProblem (*create_problem_)(CplexEnv, int*, const char*){};
    int (*free_problem_)(CplexEnv, CplexProblem*){};
};

inline constexpr int kCplexMaximize = -1;
inline constexpr double kCplexInfinity = 1.0e20;
inline constexpr int kCplexParamScreenOutput = 1035;
inline constexpr int kCplexParamTimeLimit = 1039;

}  // namespace slbjnv
