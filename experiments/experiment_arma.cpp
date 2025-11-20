/**
 * @file experiment_arma.cpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief
 * @version 0.1
 *
 * @copyright Copyright (c) 2025
 *
 *
 */

#include "experiment_arma.hpp"

#include <armadillo>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>

#include "../src/analyzer.hpp"
#include "../src/definitions.hpp"
#include "../src/experiment_util.hpp"
#include "../src/utils/common.hpp"
#include "../src/utils/matrix_generator.hpp"

using cg::Instance;

arma::mat copy2ArmaMat(cg::Matrix& sy_mat, dMatrix& live_mat) {
  arma::mat matrix(live_mat.data, live_mat.rows, live_mat.cols);

  if (sy_mat.isSymmetric()) {
    matrix = symmatl(matrix);
  } else if (sy_mat.isLower()) {
    matrix = trimatl(matrix);
  } else if (sy_mat.isTriangular()) {
    matrix = trimatu(matrix);
  }
  return matrix;
}

std::vector<arma::mat> createArmaMats(cg::MatrixChain& sy_chain,
                                      const Instance& q) {
  const unsigned n_mats = q.size() - 1U;  // #mats = len(instance) - 1
  std::vector<arma::mat> mats;
  mats.resize(n_mats);

  // Generate chain with proper operands
  auto live_chain = cg::MatrixGenerator::generateChain(sy_chain, q);

  // Copy operands to arma::mat
  for (unsigned i = 0U; i < n_mats; i++) {
    mats[i] = copy2ArmaMat(sy_chain[i], live_chain[i]);
  }
  return mats;
}

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "Error: fewer arguments than required\n";
    std::cerr << "Usage: ./prog <id> <nreps> <fname_times_vars> <fname_out>\n";
    exit(-1);
  }
  const unsigned ID = std::stoi(argv[1]);
  const unsigned nreps = std::stoi(argv[2]);
  std::string fname_times = argv[3];
  std::string fname_out = argv[4];

  std::ifstream is;
  is.open(fname_times);
  if (is.fail()) {
    std::cerr << "Cannot open file " << fname_times << '\n';
    exit(-1);
  }

  unsigned shapeID, n_algs, N, tmp1, len_instance;
  is >> shapeID;
  is >> n_algs;
  is >> N;
  is >> tmp1;
  is >> len_instance;

  std::vector<cg::Instance> Q;
  Q.resize(N);
  for (auto& instance : Q) {
    instance.resize(len_instance);
    for (unsigned i = 0U; i < len_instance; i++) is >> instance[i];
  }
  is.close();

  auto sy_chain = cg::ID2chain(len_instance - 1U, shapeID);
  std::vector<double> times(nreps * N);

  // i: instance id
  for (unsigned i = 0U; i < Q.size(); i++) {
    std::cout << i << '\n';
    std::vector<arma::mat> arma_mats = createArmaMats(sy_chain, Q[i]);
    for (unsigned rep = 0; rep < nreps; rep++) {
      // This is fixed for 7 matrices.
      arma::mat A = arma_mats[0];
      arma::mat B = arma_mats[1];
      arma::mat C = arma_mats[2];
      arma::mat D = arma_mats[3];
      arma::mat E = arma_mats[4];
      arma::mat F = arma_mats[5];
      arma::mat G = arma_mats[6];
      auto t0 = std::chrono::high_resolution_clock::now();
      // Computation takes place here
      arma::mat X = VEC_EXPR[ID](A, B, C, D, E, F, G);
      auto t1 = std::chrono::high_resolution_clock::now();
      double elapsed = std::chrono::duration<double>(t1 - t0).count();
      times[rep + i * nreps] = elapsed;
    }
  }

  // Write timings to file
  cg::RawTimes raw_times_arma;
  raw_times_arma.shapeID = shapeID;
  raw_times_arma.size_instance = len_instance;
  raw_times_arma.M = 1;
  raw_times_arma.N = N;
  raw_times_arma.reps = nreps;
  raw_times_arma.Q = std::move(Q);
  raw_times_arma.times = std::move(times);
  cg::writeRawTimes(fname_out, raw_times_arma);
}