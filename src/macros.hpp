#ifndef MACROS_H
#define MACROS_H

/* Stringizing operator */
#define STR(s) #s

/* ALGORITHM MACROS */
#define PREFIX_TEMP "M"
#define PREFIX_COPY "cpy_"

/* LIVE MATRIX CLASS MACROS */
#define MATRIX dMatrix
#define ROWS rows
#define COLS cols
#define STRIDE stride
#define DATA data
#define TRANSPOSE transpose
#define DEALLOCATE deallocate

/* GENERATOR MACROS */
#define EXTENSION_HEADER ".hpp"          // extension of header file
#define EXTENSION_UNIT ".cpp"            // extension of a compilation unit
#define EXTENSION_DRIVER EXTENSION_UNIT  // extension of driver code

#define PATH_GENERATOR "generated_code"
#define PATH_VERIFICATION "."
#define PREFIX_FILENAME "algorithm_"
#define PREFIX_FUNC_NAME PREFIX_FILENAME
#define PREFIX_FLOPS_FUNC "flops_"
#define SINGLE_FILENAME "GMC_code"
#define DRIVER_FILENAME "driver"

#define USING "using"
#define ALG_PTR AlgPtr
#define FLOPS_PTR FlopsPtr
#define VEC_NAME functions_vector
#define SELECT_FUNC selection

#define PREFIX_GUARD "GMC_CODE"

#define PREAMBLE_GENERATED                                          \
  "#include \"../../cblas_extended/cblas_extended.hpp\"\n#include " \
  "\"../../src/macros.hpp\"\n#include "                             \
  "\"../../src/utils/dMatrix.hpp\"\n#include "                      \
  "<openblas/cblas.h>\n"

#define PREAMBLE_DRIVER                                 \
  "#include \"../../src/features.hpp\"\n#include "      \
  "\"../../src/utils/matrix_generator.hpp\"\n#include " \
  "\"../../src/utils/dMatrix.hpp\"\n\n#include <random>\n\n"

#define SMALLEST_DIM 5
#define LARGEST_DIM 15

/* TEST GENERATOR MACROS */
#define PATH_TEST "../generated_algorithms.cpp"
#define DS_INCLUDE "#include <unordered_map>\n"
#define VARIANT_INCLUDE "#include \"../src/variant.hpp\"\n"
#define USING_VARIANT "using cg::Variant;\n"
#define FUNC_TYPE FunctionType
#define FP_TYPEDEF(FT, M) "using " #FT " = " #M " (*)(" #M "&, " #M "&);"
#define DS_TYPE(FT) "std::unordered_map<Variant, " #FT ", std::hash<Variant>> "
#define DS_DECLARE(FT, NAME) DS_TYPE(FT) #NAME ";\n"
#define DS_NAME map

#define PREAMBLE_TEST                                                    \
  DS_INCLUDE                                                             \
  "\n" PREAMBLE_GENERATED VARIANT_INCLUDE "\n" USING_VARIANT FP_TYPEDEF( \
      FUNC_TYPE, MATRIX) "\n"

#define COST_VAR "cost"

#include "macros_cblas.hpp"

#endif