cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(GMC-GEN LANGUAGES CXX C)

# Enable different parts of the system
option(WITH_TESTS "Enable Tests" YES) # depends on extended_cblas
option(WITH_EXPERIMENTS "Enable Experiments" YES) 

if (WITH_EXPERIMENTS)
    set(WITH_ARMADILLO YES)
endif()


set(BLA_VENDOR OpenBLAS)
find_package(BLAS REQUIRED)

find_package(OpenMP)

# If the host is in HPC2N, adapt the paths for FMT and Armadillo
cmake_host_system_information(RESULT machine_name QUERY HOSTNAME)
if (${machine_name} MATCHES "hpc2n")
    set(fmt_DIR "/proj/nobackup/gmc_bench/Projects/fmt/build")
    set(ARMADILLO_INCLUDE_DIR "$ENV{HOME}/include")
    set(ARMADILLO_LIBRARY "$ENV{HOME}/lib/libarmadillo.so.14.6.1")
endif()

if (WITH_ARMADILLO)
    find_package(Armadillo REQUIRED)
    if (ARMADILLO_FOUND)
        message(STATUS "Armadillo library in: ${ARMADILLO_LIBRARY}")
    else()
        message(STATUS "ERROR: Could not find Armadillo")
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(comp_flags -Wall -O3 -march=native)# -O3 -march=native

# create a dir for models
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/models") # dir for models

set(TMP_DIR "${CMAKE_CURRENT_BINARY_DIR}/tmp") # set the tmp dir
file(MAKE_DIRECTORY "${TMP_DIR}") # create it
# @warning: we copy the file with the expressions used in experiments.
file(COPY ${CMAKE_SOURCE_DIR}/tmp/expr_exp2.txt DESTINATION ${TMP_DIR})

file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/times") # dir for generated timings
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/times/vars")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/times/arma")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated_code")

add_subdirectory(src)
add_subdirectory(cblas_extended)
add_subdirectory(code_generator)

if(WITH_TESTS)
    add_subdirectory(test)
endif()

if(WITH_EXPERIMENTS)
    add_subdirectory(experiments)
endif()

