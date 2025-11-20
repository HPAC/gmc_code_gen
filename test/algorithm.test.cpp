/**
 * @file algorithm.test.cpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief Example of cg::Algorithm usage. Executing this generates a file in the
 * path `generated_code/test_algorithm.cpp`. The generated code only contains
 * the function that implements the ordering specified by the permutation.
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "../src/algorithm.hpp"

#include <fstream>
#include <iostream>
#include <vector>

#include "../src/matcher.hpp"
#include "../src/matrix.hpp"
#include "../src/settings_kernels.hpp"

using namespace cg;

int main(int argc, char* argv[]) {
  Matrix A("A", Structure::Dense, Property::None, Trans::Y, Inversion::N);
  Matrix B("B", Structure::Upper, Property::None, Trans::Y, Inversion::N);
  Matrix C("C", Structure::Upper, Property::None, Trans::Y, Inversion::N);
  Matrix D("D", Structure::Dense, Property::None, Trans::N, Inversion::N);
  std::vector<Matrix> symbolic_chain = {A, B, C, D};
  std::vector<unsigned> permutation = {1, 2, 3};  // left to right evaluation

  const cg::Matcher matcher(cg::all_kernels);
  Algorithm alg(symbolic_chain, permutation, &matcher);
  std::cout << "flops: " << alg.computeFLOPs({2, 2, 2, 2, 2}) << "\n";

  std::string function_name = "algorithm_123";
  std::cout << "\n\nGenerated cost function:\n\n";
  std::cout << alg.generateCostFunction(function_name);
  std::cout << "\n\nGenerated algorithm:\n\n";
  std::cout << alg.generateCode(function_name, true);
}