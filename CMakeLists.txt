cmake_minimum_required(VERSION 3.21)
project(SLbJNV_BranchAndPrice VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(SLBJNV_ENABLE_CPLEX "Enable the CPLEX solver backend" OFF)
option(SLBJNV_ENABLE_HIGHS "Enable the HiGHS solver backend" OFF)
option(SLBJNV_ENABLE_PYTHON "Enable embedded Python inference through pybind11" OFF)
option(SLBJNV_BUILD_TESTS "Build unit tests" ON)

if(SLBJNV_ENABLE_CPLEX AND SLBJNV_ENABLE_HIGHS)
    message(FATAL_ERROR "Select exactly one LP/MIP backend")
endif()

add_library(slbjnv_core
    cpp/src/instance/gbd_instance.cpp
    cpp/src/instance/instance_generator.cpp
    cpp/src/instance/instance_io.cpp
    cpp/src/model/batch_column.cpp
    cpp/src/branching/branch_constraint.cpp
    cpp/src/branching/branching_rule.cpp
    cpp/src/branch_and_price/search_tree.cpp
    cpp/src/branch_and_price/incumbent_manager.cpp
    cpp/src/branch_and_price/slbjnv_branching_policy.cpp
    cpp/src/features/feature_extractor.cpp
)
target_include_directories(slbjnv_core PUBLIC cpp/include)

if(MSVC)
    target_compile_options(slbjnv_core PRIVATE /W4 /permissive-)
else()
    target_compile_options(slbjnv_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Make MinGW executables portable across CLion, CTest, and clean Windows shells.
if(MINGW OR CMAKE_CXX_COMPILER MATCHES "mingw.*g\\+\\+|mingw64.*/bin/g\\+\\+")
    add_link_options(-static-libgcc -static-libstdc++)
endif()

if(SLBJNV_ENABLE_CPLEX)
    if(NOT WIN32)
        message(FATAL_ERROR "The bundled dynamic CPLEX adapter currently targets Windows")
    endif()
    set(SLBJNV_CPLEX_DLL "" CACHE FILEPATH "Path to the CPLEX callable-library DLL")
    if(NOT SLBJNV_CPLEX_DLL)
        find_file(_SLBJNV_CPLEX_DLL_AUTO
            NAMES cplex12100.dll cplex2211.dll cplex2210.dll cplex2010.dll
            HINTS "$ENV{CPLEX_STUDIO_DIR}"
                  "C:/Program Files/IBM/ILOG/CPLEX_Enterprise_Server1210/CPLEX_Studio"
            PATH_SUFFIXES cplex/bin/x64_win64 bin/x64_win64)
        if(_SLBJNV_CPLEX_DLL_AUTO)
            set(SLBJNV_CPLEX_DLL "${_SLBJNV_CPLEX_DLL_AUTO}" CACHE FILEPATH
                "Path to the CPLEX callable-library DLL" FORCE)
        endif()
    endif()
    if(NOT SLBJNV_CPLEX_DLL)
        message(FATAL_ERROR "CPLEX runtime DLL not found; set SLBJNV_CPLEX_DLL")
    endif()
    message(STATUS "CPLEX callable library: ${SLBJNV_CPLEX_DLL}")
    target_sources(slbjnv_core PRIVATE
        cpp/src/model/cplex_c_api.cpp
        cpp/src/column_generation/cplex_restricted_master_problem.cpp
        cpp/src/column_generation/cplex_pricing_problem.cpp
        cpp/src/column_generation/default_column_generation.cpp
        cpp/src/branch_and_price/default_branch_and_price.cpp
        cpp/src/data_collection/training_data_collector.cpp
        cpp/src/data_collection/training_data_io.cpp)
    target_compile_definitions(slbjnv_core PUBLIC SLBJNV_WITH_CPLEX=1)
    target_compile_definitions(slbjnv_core PRIVATE SLBJNV_CPLEX_DLL_PATH="${SLBJNV_CPLEX_DLL}")
elseif(SLBJNV_ENABLE_HIGHS)
    find_package(highs CONFIG REQUIRED)
    target_compile_definitions(slbjnv_core PUBLIC SLBJNV_WITH_HIGHS=1)
    target_link_libraries(slbjnv_core PUBLIC highs::highs)
endif()

if(SLBJNV_ENABLE_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    target_sources(slbjnv_core PRIVATE cpp/src/python_interface/python_score_predictor.cpp)
    target_compile_definitions(slbjnv_core PUBLIC SLBJNV_WITH_PYTHON=1)
    target_link_libraries(slbjnv_core PUBLIC pybind11::embed)
endif()

foreach(app IN ITEMS generate_instances generate_training_data run_experiment)
    add_executable(${app} cpp/apps/${app}.cpp)
    target_link_libraries(${app} PRIVATE slbjnv_core)
endforeach()
target_compile_definitions(run_experiment PRIVATE SLBJNV_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

if(SLBJNV_BUILD_TESTS)
    include(CTest)
    add_executable(test_core_contracts tests/cpp/test_core_contracts.cpp)
    target_link_libraries(test_core_contracts PRIVATE slbjnv_core)
    add_test(NAME core_contracts COMMAND test_core_contracts)

    add_executable(test_instance_generation tests/cpp/test_instance_generation.cpp)
    target_link_libraries(test_instance_generation PRIVATE slbjnv_core)
    add_test(NAME instance_generation COMMAND test_instance_generation)

    foreach(rule RANGE 1 5)
        add_executable(test_branching_rule_${rule} tests/cpp/test_branching_rule_${rule}.cpp)
        target_link_libraries(test_branching_rule_${rule} PRIVATE slbjnv_core)
        add_test(NAME branching_rule_${rule} COMMAND test_branching_rule_${rule})
    endforeach()

    add_executable(test_phase7_policy tests/cpp/test_phase7_policy.cpp)
    target_link_libraries(test_phase7_policy PRIVATE slbjnv_core)
    target_compile_definitions(test_phase7_policy PRIVATE SLBJNV_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    add_test(NAME phase7_joint_policy COMMAND test_phase7_policy)

    if(SLBJNV_ENABLE_CPLEX)
        add_executable(test_phase3 tests/cpp/test_phase3.cpp)
        target_link_libraries(test_phase3 PRIVATE slbjnv_core)
        target_compile_definitions(test_phase3 PRIVATE SLBJNV_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
        add_test(NAME phase3_exact_solver COMMAND test_phase3)

        add_executable(test_phase5 tests/cpp/test_phase5.cpp)
        target_link_libraries(test_phase5 PRIVATE slbjnv_core)
        target_compile_definitions(test_phase5 PRIVATE SLBJNV_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
        add_test(NAME phase5_training_data COMMAND test_phase5)
    endif()

    if(SLBJNV_ENABLE_PYTHON)
        set(_SLBJNV_PHASE7_MODEL "${CMAKE_BINARY_DIR}/phase7_test_model.pt")
        add_test(NAME phase7_create_checkpoint
            COMMAND ${Python_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tests/python/create_test_checkpoint.py
                    ${_SLBJNV_PHASE7_MODEL})
        set_tests_properties(phase7_create_checkpoint PROPERTIES FIXTURES_SETUP phase7_model)
        add_executable(test_phase7_python tests/cpp/test_phase7_python.cpp)
        target_link_libraries(test_phase7_python PRIVATE slbjnv_core)
        target_compile_definitions(test_phase7_python PRIVATE SLBJNV_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
        add_test(NAME phase7_python_predictor COMMAND test_phase7_python ${_SLBJNV_PHASE7_MODEL})
        set_tests_properties(phase7_python_predictor PROPERTIES FIXTURES_REQUIRED phase7_model)
    endif()
endif()
