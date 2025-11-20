/**
 * @file generate_models.cpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief This program generates the models for every kernel in the system. The
 * path where files are written can be found in `src/models/settings_models.hpp`
 * @version 0.1
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>

#include "../src/algorithm.hpp"
#include "../src/analyzer.hpp"
#include "../src/matrix.hpp"
#include "../src/models/base1d.hpp"
#include "../src/models/base2d.hpp"
#include "../src/models/base3d.hpp"
#include "../src/models/model1d.hpp"
#include "../src/models/model2d.hpp"
#include "../src/models/model3d.hpp"
#include "../src/models/model_common.hpp"
#include "../src/settings_kernels.hpp"
#include "../src/utils/matrix_generator.hpp"

using namespace cg;
using namespace cg::mdl;

MatrixChain modifyChain(const uint8_t key, const MatrixChain& base_chain) {
  MatrixChain chain = base_chain;

  // not entirely sure whether this works for all kernels.
  if (checkBit(key, 3)) chain[0].flipUplo();

  if (checkBit(key, 2)) chain[1].flipUplo();

  if (checkBit(key, 1)) chain[0].T();

  if (checkBit(key, 0)) chain[1].T();

  if (checkBit(key, 4)) std::swap(chain[0], chain[1]);

  return chain;
}

struct KeysNChains {
  std::vector<uint8_t> keys;
  std::vector<MatrixChain> chains;
};

KeysNChains getSymbolicVariants(const MatrixChain& base_chain,
                                const uint8_t options_kernel) {
  uint8_t options = options_kernel;
  uint8_t num_options = static_cast<uint8_t>(1 << countSetBits(options_kernel));
  uint8_t key;
  KeysNChains result;

  for (uint8_t value = 0; value < num_options; value++) {
    key = getKeyFromOptions(value, options);
    result.keys.push_back(key);
    result.chains.emplace_back(modifyChain(key, base_chain));
  }

  return result;
}

double getMedian(std::vector<double>& times) {
  std::sort(times.begin(), times.end());
  unsigned half = times.size() / 2;
  if (times.size() % 2 == 0) {
    return (times[half - 1] + times[half]) / 2.0;
  } else {
    return times[half];
  }
}

Model1D generateModel1D(const std::vector<MatrixChain>& sy_chains,
                        const std::vector<uint8_t>& keys, Matcher& matcher,
                        const std::vector<unsigned>& sizes_m,
                        const unsigned n_rep, const uint8_t options) {
  Instance instance;
  std::vector<dMatrix> live_chain;
  std::vector<double> times_instance;
  double median_time;

  const unsigned M = sizes_m.size();
  std::vector<double> perf_vec(M);

  // the model must be initialised with the options.
  Model1D model(options);

  // loop over the chains and keys.
  for (unsigned v = 0; v < sy_chains.size(); v++) {
    Algorithm alg(sy_chains[v], {1}, &matcher);
    for (unsigned i = 0; i < M; i++) {
      instance = {sizes_m[i], sizes_m[i], sizes_m[i]};
      live_chain = MatrixGenerator::generateChain(sy_chains[v], instance);
      times_instance = timeAlg(alg, live_chain, n_rep);
      median_time = getMedian(times_instance);
      perf_vec[i] = alg.computeFLOPs(instance) / median_time;
    }
    model.emplaceBase(keys[v], sizes_m, perf_vec);
  }
  return model;
}

Model2D generateModel2D(const std::vector<MatrixChain>& sy_chains,
                        const std::vector<uint8_t>& keys, Matcher& matcher,
                        const std::vector<unsigned>& sizes_m,
                        const std::vector<unsigned>& sizes_n,
                        const unsigned n_rep, const uint8_t options) {
  Instance instance;
  std::vector<dMatrix> live_chain;
  std::vector<double> times_instance;
  double median_time;

  const unsigned M = sizes_m.size();
  const unsigned N = sizes_n.size();
  std::vector<double> perf_vec(M * N);

  // The model must be initialised with the options.
  Model2D model(options);
  bool side = false;

  // loop over the chains and keys.
  for (unsigned v = 0; v < sy_chains.size(); v++) {
    Algorithm alg(sy_chains[v], {1}, &matcher);
    // check whether the key's 5th bit is set for this combination of args.
    side = checkBit(keys[v], 4);
    for (unsigned i = 0; i < M; i++) {    // size_m
      for (unsigned j = 0; j < N; j++) {  // size_n
        if (side) {
          instance = {sizes_m[i], sizes_n[j], sizes_n[j]};
        } else {
          instance = {sizes_m[i], sizes_m[i], sizes_n[j]};
        }
        live_chain = MatrixGenerator::generateChain(sy_chains[v], instance);
        times_instance = timeAlg(alg, live_chain, n_rep);
        median_time = getMedian(times_instance);
        perf_vec[i * N + j] = alg.computeFLOPs(instance) / median_time;
      }
    }
    model.emplaceBase(keys[v], sizes_m, sizes_n, perf_vec);
  }
  return model;
}

Model3D generateModel3D(const std::vector<MatrixChain>& sy_chains,
                        const std::vector<uint8_t>& keys, Matcher& matcher,
                        const std::vector<unsigned>& sizes_m,
                        const std::vector<unsigned>& sizes_k,
                        const std::vector<unsigned>& sizes_n,
                        const unsigned n_rep, const uint8_t options) {
  Instance instance;
  std::vector<dMatrix> live_chain;
  std::vector<double> times_instance;
  double median_time;

  const unsigned M = sizes_m.size();
  const unsigned K = sizes_k.size();
  const unsigned N = sizes_n.size();
  std::vector<double> perf_vec(M * K * N);

  // The model must be initialised with the options.
  Model3D model(options);

  // loop over the chains and keys.
  for (unsigned v = 0U; v < sy_chains.size(); v++) {
    Algorithm alg(sy_chains[v], {1U}, &matcher);
    for (unsigned i = 0U; i < M; i++) {      // size m
      for (unsigned k = 0U; k < K; k++) {    // size k
        for (unsigned j = 0U; j < N; j++) {  // size n
          instance = {sizes_m[i], sizes_k[k], sizes_n[j]};
          live_chain = MatrixGenerator::generateChain(sy_chains[v], instance);
          times_instance = timeAlg(alg, live_chain, n_rep);
          median_time = getMedian(times_instance);
          // mapping from (m,k,n) to linear vector.
          perf_vec[i * K * N + k * N + j] =
              alg.computeFLOPs(instance) / median_time;
        }  // size n
      }  // size k
    }  // size m
    model.emplaceBase(keys[v], sizes_m, sizes_k, sizes_n, perf_vec);
  }
  return model;
}

int main(int argc, char** argv) {
  unsigned n_rep;

  if (argc < 2) {
    std::cerr << "Usage: ./generate_models n_rep\n";
    exit(-1);
  } else {
    n_rep = std::stoi(argv[1]);
  }

  cg::Matcher matcher(cg::all_kernels);
  std::vector<unsigned> sizes_m = {50U, 100U, 300U, 500U, 700U, 1000U};
  std::vector<unsigned> sizes_k = {50U, 100U, 300U, 500U, 700U, 1000U};
  std::vector<unsigned> sizes_n = {50U, 100U, 300U, 500U, 700U, 1000U};

  cg::Matrix A, B;
  cg::MatrixChain sy_chain;
  std::string filename;
  uint8_t kernel_args;
  KeysNChains knc;
  cg::mdl::Model1D model;

  // ============================ SYSYMM ============================
  A = {"A",      Structure::Symmetric_U, Property::None,
       Trans::N, Inversion::N,           true};
  B = {"B",      Structure::Symmetric_U, Property::None,
       Trans::N, Inversion::N,           true};
  sy_chain = {A, B};
  filename = kernel_sysymm.getPathModel();
  kernel_args = kernel_sysymm.getArgs();

  std::cout << "Generating model for SYSYMM... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model = generateModel1D(knc.chains, knc.keys, matcher, sizes_m, n_rep,
                          kernel_args);
  model.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ TRSYMM ============================
  A = {"A", Structure::Upper, Property::None, Trans::N, Inversion::N, true};
  B = {"B",      Structure::Symmetric_U, Property::None,
       Trans::N, Inversion::N,           true};
  sy_chain = {A, B};
  filename = kernel_trsymm.getPathModel();
  kernel_args = kernel_trsymm.getArgs();

  std::cout << "Generating model for TRSYMM... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model = generateModel1D(knc.chains, knc.keys, matcher, sizes_m, n_rep,
                          kernel_args);
  model.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ DISYMM ============================
  A = {"A", Structure::Diagonal, Property::None, Trans::N, Inversion::N, true};
  B = {"B",      Structure::Symmetric_U, Property::None,
       Trans::N, Inversion::N,           true};
  sy_chain = {A, B};
  filename = kernel_disymm.getPathModel();
  kernel_args = kernel_disymm.getArgs();

  std::cout << "Generating model for DISYMM... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model = generateModel1D(knc.chains, knc.keys, matcher, sizes_m, n_rep,
                          kernel_args);
  model.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ TRTRMM ============================
  A = {"A", Structure::Upper, Property::None, Trans::N, Inversion::N, true};
  B = {"B", Structure::Upper, Property::None, Trans::N, Inversion::N, true};
  sy_chain = {A, B};
  filename = kernel_trtrmm.getPathModel();
  kernel_args = kernel_trtrmm.getArgs();

  std::cout << "Generating model for TRTRMM... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model = generateModel1D(knc.chains, knc.keys, matcher, sizes_m, n_rep,
                          kernel_args);
  model.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ DITRMM ============================
  A = {"A", Structure::Diagonal, Property::None, Trans::N, Inversion::N, true};
  B = {"B", Structure::Upper, Property::None, Trans::N, Inversion::N, true};
  sy_chain = {A, B};
  filename = kernel_ditrmm.getPathModel();
  kernel_args = kernel_ditrmm.getArgs();

  std::cout << "Generating model for DITRMM... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model = generateModel1D(knc.chains, knc.keys, matcher, sizes_m, n_rep,
                          kernel_args);
  model.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ DIDIMM ============================
  A = {"A", Structure::Diagonal, Property::None, Trans::N, Inversion::N, true};
  B = {"B", Structure::Diagonal, Property::None, Trans::N, Inversion::N, true};
  sy_chain = {A, B};
  filename = kernel_didimm.getPathModel();
  kernel_args = kernel_didimm.getArgs();

  std::cout << "Generating model for DIDIMM... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model = generateModel1D(knc.chains, knc.keys, matcher, sizes_m, n_rep,
                          kernel_args);
  model.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ GESYSV ============================
  A = {"A", Structure::Dense, Property::FullRank, Trans::N, Inversion::Y, true};
  B = {"B",      Structure::Symmetric_U, Property::None,
       Trans::N, Inversion::N,           true};
  sy_chain = {A, B};
  filename = kernel_gesysv.getPathModel();
  kernel_args = kernel_gesysv.getArgs();

  std::cout << "Generating model for GESYSV... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model = generateModel1D(knc.chains, knc.keys, matcher, sizes_m, n_rep,
                          kernel_args);
  model.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ GETRSV ============================
  A = {"A", Structure::Dense, Property::FullRank, Trans::N, Inversion::Y, true};
  B = {"B", Structure::Upper, Property::None, Trans::N, Inversion::N, true};
  sy_chain = {A, B};
  filename = kernel_getrsv.getPathModel();
  kernel_args = kernel_getrsv.getArgs();

  std::cout << "Generating model for GETRSV... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model = generateModel1D(knc.chains, knc.keys, matcher, sizes_m, n_rep,
                          kernel_args);
  model.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ GEDISV ============================
  A = {"A", Structure::Dense, Property::FullRank, Trans::N, Inversion::Y, true};
  B = {"B", Structure::Diagonal, Property::None, Trans::N, Inversion::N, true};
  sy_chain = {A, B};
  filename = kernel_gedisv.getPathModel();
  kernel_args = kernel_gedisv.getArgs();

  std::cout << "Generating model for GEDISV... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model = generateModel1D(knc.chains, knc.keys, matcher, sizes_m, n_rep,
                          kernel_args);
  model.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ SYSYSV ============================
  A = {"A",      Structure::Symmetric_U, Property::FullRank,
       Trans::N, Inversion::Y,           true};
  B = {"B",      Structure::Symmetric_U, Property::None,
       Trans::N, Inversion::N,           true};
  sy_chain = {A, B};
  filename = kernel_sysysv.getPathModel();
  kernel_args = kernel_sysysv.getArgs();

  std::cout << "Generating model for SYSYSV... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model = generateModel1D(knc.chains, knc.keys, matcher, sizes_m, n_rep,
                          kernel_args);
  model.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ SYTRSV ============================
  A = {"A",      Structure::Symmetric_U, Property::FullRank,
       Trans::N, Inversion::Y,           true};
  B = {"B", Structure::Upper, Property::None, Trans::N, Inversion::N, true};
  sy_chain = {A, B};
  filename = kernel_sytrsv.getPathModel();
  kernel_args = kernel_sytrsv.getArgs();

  std::cout << "Generating model for SYTRSV... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model = generateModel1D(knc.chains, knc.keys, matcher, sizes_m, n_rep,
                          kernel_args);
  model.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ SYDISV ============================
  A = {"A",      Structure::Symmetric_U, Property::FullRank,
       Trans::N, Inversion::Y,           true};
  B = {"B", Structure::Diagonal, Property::None, Trans::N, Inversion::N, true};
  sy_chain = {A, B};
  filename = kernel_sydisv.getPathModel();
  kernel_args = kernel_sydisv.getArgs();

  std::cout << "Generating model for SYDISV... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model = generateModel1D(knc.chains, knc.keys, matcher, sizes_m, n_rep,
                          kernel_args);
  model.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ TRSYSV ============================
  A = {"A", Structure::Upper, Property::FullRank, Trans::N, Inversion::Y, true};
  B = {"B",      Structure::Symmetric_U, Property::None,
       Trans::N, Inversion::N,           true};
  sy_chain = {A, B};
  filename = kernel_trsysv.getPathModel();
  kernel_args = kernel_trsysv.getArgs();

  std::cout << "Generating model for TRSYSV... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model = generateModel1D(knc.chains, knc.keys, matcher, sizes_m, n_rep,
                          kernel_args);
  model.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ TRTRSV ============================
  A = {"A", Structure::Upper, Property::FullRank, Trans::N, Inversion::Y, true};
  B = {"B", Structure::Upper, Property::None, Trans::N, Inversion::N, true};
  sy_chain = {A, B};
  filename = kernel_trtrsv.getPathModel();
  kernel_args = kernel_trtrsv.getArgs();

  std::cout << "Generating model for TRTRSV... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model = generateModel1D(knc.chains, knc.keys, matcher, sizes_m, n_rep,
                          kernel_args);
  model.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ TRDISV ============================
  A = {"A", Structure::Upper, Property::FullRank, Trans::N, Inversion::Y, true};
  B = {"B", Structure::Diagonal, Property::None, Trans::N, Inversion::N, true};
  sy_chain = {A, B};
  filename = kernel_trdisv.getPathModel();
  kernel_args = kernel_trdisv.getArgs();

  std::cout << "Generating model for TRDISV... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model = generateModel1D(knc.chains, knc.keys, matcher, sizes_m, n_rep,
                          kernel_args);
  model.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ DISYSV ============================
  A = {"A",      Structure::Diagonal, Property::FullRank,
       Trans::N, Inversion::Y,        true};
  B = {"B",      Structure::Symmetric_U, Property::None,
       Trans::N, Inversion::N,           true};
  sy_chain = {A, B};
  filename = kernel_disysv.getPathModel();
  kernel_args = kernel_disysv.getArgs();

  std::cout << "Generating model for DISYSV... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model = generateModel1D(knc.chains, knc.keys, matcher, sizes_m, n_rep,
                          kernel_args);
  model.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ DITRSM ============================
  A = {"A",      Structure::Diagonal, Property::FullRank,
       Trans::N, Inversion::Y,        true};
  B = {"B", Structure::Upper, Property::None, Trans::N, Inversion::N, true};
  sy_chain = {A, B};
  filename = kernel_ditrsv.getPathModel();
  kernel_args = kernel_ditrsv.getArgs();

  std::cout << "Generating model for DITRSM... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model = generateModel1D(knc.chains, knc.keys, matcher, sizes_m, n_rep,
                          kernel_args);
  model.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ DIDISM ============================
  A = {"A",      Structure::Diagonal, Property::FullRank,
       Trans::N, Inversion::Y,        true};
  B = {"B", Structure::Diagonal, Property::None, Trans::N, Inversion::N, true};
  sy_chain = {A, B};
  filename = kernel_didisv.getPathModel();
  kernel_args = kernel_didisv.getArgs();

  std::cout << "Generating model for DIDISM... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model = generateModel1D(knc.chains, knc.keys, matcher, sizes_m, n_rep,
                          kernel_args);
  model.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ POSYSV ============================
  A = {"A",      Structure::Symmetric_U, Property::SPD,
       Trans::N, Inversion::Y,           true};
  B = {"B",      Structure::Symmetric_U, Property::None,
       Trans::N, Inversion::N,           true};
  sy_chain = {A, B};
  filename = kernel_posysv.getPathModel();
  kernel_args = kernel_posysv.getArgs();

  std::cout << "Generating model for POSYSV... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model = generateModel1D(knc.chains, knc.keys, matcher, sizes_m, n_rep,
                          kernel_args);
  model.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ POTRSV ============================
  A = {"A",      Structure::Symmetric_U, Property::SPD,
       Trans::N, Inversion::Y,           true};
  B = {"B", Structure::Upper, Property::None, Trans::N, Inversion::N, true};
  sy_chain = {A, B};
  filename = kernel_potrsv.getPathModel();
  kernel_args = kernel_potrsv.getArgs();

  std::cout << "Generating model for POTRSV... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model = generateModel1D(knc.chains, knc.keys, matcher, sizes_m, n_rep,
                          kernel_args);
  model.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ PODISV ============================
  A = {"A",      Structure::Symmetric_U, Property::SPD,
       Trans::N, Inversion::Y,           true};
  B = {"B", Structure::Diagonal, Property::None, Trans::N, Inversion::N, true};
  sy_chain = {A, B};
  filename = kernel_podisv.getPathModel();
  kernel_args = kernel_podisv.getArgs();

  std::cout << "Generating model for PODISV... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model = generateModel1D(knc.chains, knc.keys, matcher, sizes_m, n_rep,
                          kernel_args);
  model.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // inline std::vector<Kernel*> all_1DKernels = {
  //     &kernel_sysymm, &kernel_trsymm, &kernel_disymm, &kernel_trtrmm,
  //     &kernel_ditrmm, &kernel_didimm, &kernel_gesysv, &kernel_getrsv,
  //     &kernel_gedisv, &kernel_sysysv, &kernel_sytrsv, &kernel_sydisv,
  //     &kernel_trsysv, &kernel_trtrsv, &kernel_trdisv, &kernel_disysm,
  //     &kernel_ditrsm, &kernel_didism, &kernel_posysv, &kernel_potrsv,
  //     &kernel_podisv};

  Model2D model2d;

  // ============================ SYMM ============================
  A = {"A",      Structure::Symmetric_U, Property::None,
       Trans::N, Inversion::N,           true};
  B = {"B", Structure::Dense, Property::None, Trans::N, Inversion::N, true};
  sy_chain = {A, B};
  filename = kernel_symm.getPathModel();
  kernel_args = kernel_symm.getArgs();

  std::cout << "Generating model for SYMM... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model2d = generateModel2D(knc.chains, knc.keys, matcher, sizes_m, sizes_n,
                            n_rep, kernel_args);
  model2d.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ TRMM ============================
  A = {"A", Structure::Upper, Property::None, Trans::N, Inversion::N, true};
  B = {"B", Structure::Dense, Property::None, Trans::N, Inversion::N, true};
  sy_chain = {A, B};
  filename = kernel_trmm.getPathModel();
  kernel_args = kernel_trmm.getArgs();

  std::cout << "Generating model for TRMM... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model2d = generateModel2D(knc.chains, knc.keys, matcher, sizes_m, sizes_n,
                            n_rep, kernel_args);
  model2d.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ DIMM ============================
  A = {"A", Structure::Diagonal, Property::None, Trans::N, Inversion::N, true};
  B = {"B", Structure::Dense, Property::None, Trans::N, Inversion::N, true};
  sy_chain = {A, B};
  filename = kernel_dimm.getPathModel();
  kernel_args = kernel_dimm.getArgs();

  std::cout << "Generating model for DIMM... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model2d = generateModel2D(knc.chains, knc.keys, matcher, sizes_m, sizes_n,
                            n_rep, kernel_args);
  model2d.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ GEGESV ============================
  A = {"A", Structure::Dense, Property::FullRank, Trans::N, Inversion::Y, true};
  B = {"B", Structure::Dense, Property::None, Trans::N, Inversion::N, true};
  sy_chain = {A, B};
  filename = kernel_gegesv.getPathModel();
  kernel_args = kernel_gegesv.getArgs();

  std::cout << "Generating model for GEGESV... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model2d = generateModel2D(knc.chains, knc.keys, matcher, sizes_m, sizes_n,
                            n_rep, kernel_args);
  model2d.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ SYGESV ============================
  A = {"A",      Structure::Symmetric_U, Property::FullRank,
       Trans::N, Inversion::Y,           true};
  B = {"B", Structure::Dense, Property::None, Trans::N, Inversion::N, true};
  sy_chain = {A, B};
  filename = kernel_sygesv.getPathModel();
  kernel_args = kernel_sygesv.getArgs();

  std::cout << "Generating model for SYGESV... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model2d = generateModel2D(knc.chains, knc.keys, matcher, sizes_m, sizes_n,
                            n_rep, kernel_args);
  model2d.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ TRSM ============================
  A = {"A", Structure::Upper, Property::FullRank, Trans::N, Inversion::Y, true};
  B = {"B", Structure::Dense, Property::None, Trans::N, Inversion::N, true};
  sy_chain = {A, B};
  filename = kernel_trsm.getPathModel();
  kernel_args = kernel_trsm.getArgs();

  std::cout << "Generating model for TRSM... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model2d = generateModel2D(knc.chains, knc.keys, matcher, sizes_m, sizes_n,
                            n_rep, kernel_args);
  model2d.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ DISM ============================
  A = {"A",      Structure::Diagonal, Property::FullRank,
       Trans::N, Inversion::Y,        true};
  B = {"B", Structure::Dense, Property::None, Trans::N, Inversion::N, true};
  sy_chain = {A, B};
  filename = kernel_disv.getPathModel();
  kernel_args = kernel_disv.getArgs();

  std::cout << "Generating model for DISM... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model2d = generateModel2D(knc.chains, knc.keys, matcher, sizes_m, sizes_n,
                            n_rep, kernel_args);
  model2d.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // ============================ POGESV ============================
  A = {"A",      Structure::Symmetric_U, Property::SPD,
       Trans::N, Inversion::Y,           true};
  B = {"B", Structure::Dense, Property::None, Trans::N, Inversion::N, true};
  sy_chain = {A, B};
  filename = kernel_pogesv.getPathModel();
  kernel_args = kernel_pogesv.getArgs();

  std::cout << "Generating model for POGESV... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model2d = generateModel2D(knc.chains, knc.keys, matcher, sizes_m, sizes_n,
                            n_rep, kernel_args);
  model2d.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // inline std::vector<Kernel*> all_2DKernels = {
  //     &kernel_symm,   &kernel_trmm, &kernel_dimm, &kernel_gegesv,
  //     &kernel_sygesv, &kernel_trsm, &kernel_dism, &kernel_pogesv};

  Model3D model3d;

  // ============================ GEMM ============================
  A = {"A", Structure::Dense, Property::None, Trans::N, Inversion::N, true};
  B = {"B", Structure::Dense, Property::None, Trans::N, Inversion::N, true};
  sy_chain = {A, B};
  filename = kernel_gemm.getPathModel();
  kernel_args = kernel_gemm.getArgs();

  std::cout << "Generating model for GEMM... " << std::flush;
  knc = getSymbolicVariants(sy_chain, kernel_args);
  model3d = generateModel3D(knc.chains, knc.keys, matcher, sizes_m, sizes_k,
                            sizes_n, n_rep, kernel_args);
  model3d.write(filename);
  std::cout << "generated and dumped to " << filename << '\n';

  // inline std::vector<kernel*> all_3DKernels = {&kernel_gemm};
  // ==============================================================
}

// inline std::vector<Kernel*> all_kernels = {
//     &kernel_gemm,   &kernel_symm,   &kernel_trmm,   &kernel_dimm,
//     &kernel_sysymm, &kernel_trsymm, &kernel_disymm, &kernel_trtrmm,
//     &kernel_ditrmm, &kernel_didimm, &kernel_gegesv, &kernel_gesysv,
//     &kernel_getrsv, &kernel_gedisv, &kernel_sygesv, &kernel_sysysv,
//     &kernel_sytrsv, &kernel_sydisv, &kernel_trsm,   &kernel_trsysv,
//     &kernel_trtrsv, &kernel_trdisv, &kernel_dism,   &kernel_disysm,
//     &kernel_ditrsm, &kernel_didism, &kernel_pogesv, &kernel_posysv,
//     &kernel_potrsv, &kernel_podisv};
