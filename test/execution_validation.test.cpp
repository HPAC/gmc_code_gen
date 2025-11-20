/**
 * @file execution_validation.test.cpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief Creates chains with 4 matrices where the third and last matrices have
 * all possible combinations of features. The program checks that the numeric
 * results across algorithms for the same chain are the same.
 * @version 0.1
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <chrono>
#include <iostream>
#include <string>
#include <vector>

#include "../src/algorithm.hpp"
#include "../src/analyzer.hpp"
#include "../src/frontend/parser.hpp"
#include "../src/generator.hpp"
#include "../src/instance_generator.hpp"
#include "../src/matrix.hpp"
#include "../src/utils/common.hpp"
#include "../src/utils/dMatrix.hpp"
#include "../src/utils/matrix_generator.hpp"
#include "../src/variant.hpp"

using namespace cg;

int main() {
  Matrix A("A", Structure::Dense, Property::None, Trans::N, Inversion::N);
  Matrix B("B", Structure::Dense, Property::None, Trans::N, Inversion::N);
  MatrixChain symb_chain = {A, B};

  RangeMatrix full_range_matrix = {all_structures, all_properties, all_trans,
                                   all_inv};
  RangeVariant full_range = {full_range_matrix, full_range_matrix};
  auto all_variants = full_range.generateVariants();

  Generator generator;
  InstanceGenerator inst_gnrtor(5, 15);
  Instance s;
  std::vector<dMatrix> live_chain;
  dMatrix result_0, result_x;
  std::vector<Algorithm> Algs;
  double error;

  for (const auto& var : all_variants) {
    symb_chain.push_back(Matrix{"C", var.left});
    symb_chain.push_back(Matrix{"D", var.right});
    std::cout << var << "\n";

    generator.setMatrices(symb_chain);
    Algs = generator.getAlgorithms();
    s = inst_gnrtor.rndInstance(symb_chain);
    live_chain = MatrixGenerator::generateChain(symb_chain, s);

    Algs[0].assignChain(live_chain);
    result_0 = Algs[0].execute();
    Algs[0].clean();

    for (auto& a : Algs) {
      a.assignChain(live_chain);
      result_x = a.execute();
      a.clean();

      error = MatrixGenerator::compareMatrix(result_0, result_x);
      if (error > 1e-9) {
        std::cout << "Algorithm " << a.getPermutation()
                  << " - error = " << error << " ❌\n";
        std::cin.get();
      } else {
        std::cout << "Algorithm " << a.getPermutation()
                  << " - error = " << error << " ✅\n";
      }
    }
    std::cout << "\n\n";

    symb_chain.pop_back();
    symb_chain.pop_back();
  }

  return EXIT_SUCCESS;
}
