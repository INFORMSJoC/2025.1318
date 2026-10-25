#include "slbjnv/model/cplex_c_api.hpp"

#include <windows.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace slbjnv {
namespace {

template <typename Function>
Function resolve(HMODULE library, const char* name) {
    const auto address = GetProcAddress(library, name);
    if (address == nullptr) throw std::runtime_error(std::string("CPLEX symbol not found: ") + name);
    static_assert(sizeof(Function) == sizeof(address));
    Function function{};
    std::memcpy(&function, &address, sizeof(function));
    return function;
}

std::vector<std::filesystem::path> candidate_dlls(const std::string& explicit_dll) {
    std::vector<std::filesystem::path> result;
    if (!explicit_dll.empty()) result.emplace_back(explicit_dll);
    if (const char* root = std::getenv("CPLEX_STUDIO_DIR")) {
        result.emplace_back(std::filesystem::path(root) / "cplex/bin/x64_win64/cplex12100.dll");
    }
#ifdef SLBJNV_CPLEX_DLL_PATH
    result.emplace_back(SLBJNV_CPLEX_DLL_PATH);
#endif
    result.emplace_back("C:/Program Files/IBM/ILOG/CPLEX_Enterprise_Server1210/CPLEX_Studio/cplex/bin/x64_win64/cplex12100.dll");
    return result;
}

}  // namespace

std::shared_ptr<CplexCApi> CplexCApi::load(const std::string& explicit_dll) {
    std::filesystem::path selected;
    HMODULE library = nullptr;
    for (const auto& candidate : candidate_dlls(explicit_dll)) {
        if (!std::filesystem::exists(candidate)) continue;
        library = LoadLibraryW(candidate.wstring().c_str());
        if (library != nullptr) { selected = candidate; break; }
    }
    if (library == nullptr) {
        throw std::runtime_error("CPLEX runtime DLL was not found; set CPLEX_STUDIO_DIR or SLBJNV_CPLEX_DLL");
    }

    auto api = std::shared_ptr<CplexCApi>(new CplexCApi());
    api->library_ = library;
    try {
        using Open = CplexEnv (*)(int*);
        using Version = const char* (*)(CplexEnv);
        const auto open = resolve<Open>(library, "CPXopenCPLEX");
        api->close_environment_ = resolve<decltype(api->close_environment_)>(library, "CPXcloseCPLEX");
        api->create_problem_ = resolve<decltype(api->create_problem_)>(library, "CPXcreateprob");
        api->free_problem_ = resolve<decltype(api->free_problem_)>(library, "CPXfreeprob");
        api->chgobjsen = resolve<decltype(api->chgobjsen)>(library, "CPXchgobjsen");
        api->newrows = resolve<decltype(api->newrows)>(library, "CPXnewrows");
        api->newcols = resolve<decltype(api->newcols)>(library, "CPXnewcols");
        api->chgcoeflist = resolve<decltype(api->chgcoeflist)>(library, "CPXchgcoeflist");
        api->lpopt = resolve<decltype(api->lpopt)>(library, "CPXlpopt");
        api->mipopt = resolve<decltype(api->mipopt)>(library, "CPXmipopt");
        api->getstat = resolve<decltype(api->getstat)>(library, "CPXgetstat");
        api->getobjval = resolve<decltype(api->getobjval)>(library, "CPXgetobjval");
        api->getx = resolve<decltype(api->getx)>(library, "CPXgetx");
        api->getpi = resolve<decltype(api->getpi)>(library, "CPXgetpi");
        api->chgctype = resolve<decltype(api->chgctype)>(library, "CPXchgctype");
        api->setintparam = resolve<decltype(api->setintparam)>(library, "CPXsetintparam");
        api->setdblparam = resolve<decltype(api->setdblparam)>(library, "CPXsetdblparam");

        int status = 0;
        api->environment_ = open(&status);
        if (api->environment_ == nullptr || status != 0) {
            throw std::runtime_error("CPXopenCPLEX failed with status " + std::to_string(status));
        }
        api->check(api->setintparam(api->environment_, kCplexParamScreenOutput, 0), "disable CPLEX screen output");
        const auto version = resolve<Version>(library, "CPXversion")(api->environment_);
        api->version_ = version == nullptr ? "unknown" : version;
    } catch (...) {
        if (api->environment_ != nullptr && api->close_environment_ != nullptr) api->close_environment_(&api->environment_);
        FreeLibrary(library);
        api->library_ = nullptr;
        throw;
    }
    return api;
}

CplexCApi::~CplexCApi() {
    if (environment_ != nullptr && close_environment_ != nullptr) close_environment_(&environment_);
    if (library_ != nullptr) FreeLibrary(static_cast<HMODULE>(library_));
}

CplexProblem CplexCApi::create_problem(const char* name) const {
    int status = 0;
    auto problem = create_problem_(environment_, &status, name);
    if (problem == nullptr || status != 0) check(status == 0 ? -1 : status, "create problem");
    return problem;
}

void CplexCApi::free_problem(CplexProblem& problem) const noexcept {
    if (problem != nullptr) free_problem_(environment_, &problem);
}

void CplexCApi::check(int status, const char* operation) const {
    if (status == 0) return;
    std::ostringstream message;
    message << operation << " failed with CPLEX status " << status;
    throw std::runtime_error(message.str());
}

}  // namespace slbjnv
