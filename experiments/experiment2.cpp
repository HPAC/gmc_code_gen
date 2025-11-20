#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "../src/algorithm.hpp"
#include "../src/analyzer.hpp"
#include "../src/definitions.hpp"
#include "../src/experiment_util.hpp"
#include "../src/generator.hpp"
#include "../src/instance_generator.hpp"
#include "../src/utils/common.hpp"

using cg::operator<<;
using metricFunc = std::function<double(const std::vector<double>&)>;

struct InfoSet {
  double _fvalue;
  unsigned _set_size;

  InfoSet(const unsigned set_size, const double fvalue)
      : _fvalue{fvalue}, _set_size{set_size} {}
};

std::ostream& operator<<(std::ostream& os, const InfoSet& info_set) {
  os << info_set._set_size << ' ' << info_set._fvalue;
  return os;
}

struct ResultValidation {
  std::vector<unsigned> _shapeIDs;
  std::vector<InfoSet> _info_Es_max;
  std::vector<InfoSet> _info_Es1_max;
  std::vector<InfoSet> _info_Es2_max;
  std::vector<InfoSet> _info_ltr_max;
  std::vector<InfoSet> _info_arma_max;
  std::vector<InfoSet> _info_Es_avg;
  std::vector<InfoSet> _info_Es1_avg;
  std::vector<InfoSet> _info_Es2_avg;
  std::vector<InfoSet> _info_ltr_avg;
  std::vector<InfoSet> _info_arma_avg;

  ResultValidation(const unsigned n_shapes) {
    _shapeIDs.reserve(n_shapes);
    _info_Es_max.reserve(n_shapes);
    _info_Es1_max.reserve(n_shapes);
    _info_Es2_max.reserve(n_shapes);
    _info_ltr_max.reserve(n_shapes);
    _info_arma_max.reserve(n_shapes);
    _info_Es_avg.reserve(n_shapes);
    _info_Es1_avg.reserve(n_shapes);
    _info_Es2_avg.reserve(n_shapes);
    _info_ltr_avg.reserve(n_shapes);
    _info_arma_avg.reserve(n_shapes);
  }

  void print(std::ostream& os) {
    os << _shapeIDs.size() << '\n';
    for (unsigned i = 0U; i < _shapeIDs.size(); i++) {
      os << _shapeIDs[i] << ' ' << _info_Es_max[i] << ' ' << _info_Es1_max[i]
         << ' ' << _info_Es2_max[i] << ' ' << _info_ltr_max[i] << ' '
         << _info_arma_max[i] << ' ' << _info_Es_avg[i] << ' '
         << _info_Es1_avg[i] << ' ' << _info_Es2_avg[i] << ' '
         << _info_ltr_avg[i] << ' ' << _info_arma_avg[i] << '\n';
    }
  }

  void storeMax(const InfoSet Es, const InfoSet Es1, const InfoSet Es2,
                const InfoSet ltr, const InfoSet arma) {
    _info_Es_max.push_back(Es);
    _info_Es1_max.push_back(Es1);
    _info_Es2_max.push_back(Es2);
    _info_ltr_max.push_back(ltr);
    _info_arma_max.push_back(arma);
  }

  void storeAvg(const InfoSet Es, const InfoSet Es1, const InfoSet Es2,
                const InfoSet ltr, const InfoSet arma) {
    _info_Es_avg.push_back(Es);
    _info_Es1_avg.push_back(Es1);
    _info_Es2_avg.push_back(Es2);
    _info_ltr_avg.push_back(ltr);
    _info_arma_avg.push_back(arma);
  }
};

/**
 * This struct holds the comparsion on a per-instance basis of different
 * generated sets and Armadillo to the optimal. To select sets we use:
 *    - One training cost: either FLOPs or predictions from models.
 *    - One objective function: either average or maximum penalty.
 * Each instance of this struct holds results only for one training cost, each
 * is eventually dumped to a different file.
 */
struct InfoInstances {
  std::vector<double> _Es_max;
  std::vector<double> _Es1_max;
  std::vector<double> _Es2_max;
  std::vector<double> _ltr_max;
  std::vector<double> _Es_avg;
  std::vector<double> _Es1_avg;
  std::vector<double> _Es2_avg;
  std::vector<double> _ltr_avg;
  std::vector<double> _arma;

  InfoInstances(const unsigned total_instances) {
    _Es_max.reserve(total_instances);
    _Es1_max.reserve(total_instances);
    _Es2_max.reserve(total_instances);
    _ltr_max.reserve(total_instances);
    _Es_avg.reserve(total_instances);
    _Es1_avg.reserve(total_instances);
    _Es2_avg.reserve(total_instances);
    _ltr_avg.reserve(total_instances);
    _arma.reserve(total_instances);
  }

  void storeRatioToOptimalMax(const std::vector<double>& ratios_Es,
                              const std::vector<double>& ratios_Es1,
                              const std::vector<double>& ratios_Es2,
                              const std::vector<double>& ratios_LtR) {
    // All input vectors should have equal size.
    const unsigned n = static_cast<unsigned>(ratios_Es.size());
    for (unsigned i = 0U; i < n; i++) {
      _Es_max.push_back(ratios_Es[i]);
      _Es1_max.push_back(ratios_Es1[i]);
      _Es2_max.push_back(ratios_Es2[i]);
      _ltr_max.push_back(ratios_LtR[i]);
    }
  }

  void storeRatioToOptimalAvg(const std::vector<double>& ratios_Es,
                              const std::vector<double>& ratios_Es1,
                              const std::vector<double>& ratios_Es2,
                              const std::vector<double>& ratios_LtR) {
    // All input vectors should have equal size.
    const unsigned n = static_cast<unsigned>(ratios_Es.size());
    for (unsigned i = 0U; i < n; i++) {
      _Es_avg.push_back(ratios_Es[i]);
      _Es1_avg.push_back(ratios_Es1[i]);
      _Es2_avg.push_back(ratios_Es2[i]);
      _ltr_avg.push_back(ratios_LtR[i]);
    }
  }

  void storeRatioToOptimalArma(const std::vector<double>& ratios_arma) {
    const unsigned n = static_cast<unsigned>(ratios_arma.size());
    for (unsigned i = 0U; i < n; i++) {
      _arma.push_back(ratios_arma[i]);
    }
  }

  void print(std::ostream& os) {
    for (unsigned i = 0U; i < _Es_max.size(); i++) {
      os << _Es_max[i] << ' ' << _Es1_max[i] << ' ' << _Es2_max[i] << ' '
         << _ltr_max[i] << ' ' << _Es_avg[i] << ' ' << _Es1_avg[i] << ' '
         << _Es2_avg[i] << ' ' << _ltr_avg[i] << ' ' << _arma[i] << '\n';
    }
  }
};

/**
 * @brief Entry point.
 *
 * @arg 1: Number of instances in training.
 * @arg 2: number of shapes.
 * @arg 3: base path and name for files with timings.
 * @arg 4: base path and name for files with timings in Armadillo.
 * @arg 5: base path for output files. Four files are created:
 *          - Results from set selection with FLOPs
 *          - Results from set selection with models
 *          - Comparison of arma timings vs set timings with FLOPs
 *          - Comparison of arma timings vs set timings with models
 */
int main(int argc, char** argv) {
  unsigned N_t, n_shapes;
  std::string base_fname_times_vars;
  std::string base_fname_times_arma;
  std::string base_path_out;

  if (argc < 6) {
    std::cerr << "Error. Arguments: <N_t> <n_shapes> <base_fname_times_vars> "
                 "<base_fname_times_arma> <base_fname_out>\n"
                 "Check code for interpretation of arguments\n";
    exit(-1);
  } else {
    N_t = std::stoi(argv[1]);
    n_shapes = std::stoi(argv[2]);
    base_fname_times_vars = argv[3];
    base_fname_times_arma = argv[4];
    base_path_out = argv[5];
  }
  auto t0 = std::chrono::high_resolution_clock::now();
  const unsigned ARMA_SET_SIZE = 1U;

  std::vector<metricFunc> obj_funcs = {cg::maxPenalty, cg::avgPenalty};
  cg::InstanceGenerator inst_gnrtor(50U, 1000U);
  cg::Matcher matcher(cg::all_kernels);
  matcher.loadModels();

  // Data-structure to save results on shapes
  ResultValidation results_FLOPs(n_shapes);
  ResultValidation results_models(n_shapes);
  // create vector with results
  std::vector<ResultValidation> results = {std::move(results_FLOPs),
                                           std::move(results_models)};

  // @warning: makes this a command line argument?
  const unsigned num_instances_per_shape = 1000U;
  InfoInstances ratios_w_FLOPs(n_shapes * num_instances_per_shape);
  InfoInstances ratios_w_models(n_shapes * num_instances_per_shape);
  std::vector<InfoInstances> result_ratios = {std::move(ratios_w_FLOPs),
                                              std::move(ratios_w_models)};

  // file numbering starts at 0
  for (unsigned i = 0U; i < n_shapes; i++) {
    std::cout << "processing file: " << i << std::endl;
    std::string fname_times_vars =
        base_fname_times_vars + std::to_string(i) + ".txt";
    std::string fname_times_arma =
        base_fname_times_arma + std::to_string(i) + ".txt";

    // Read arma timings
    auto raw_times_arma = cg::readRawTimes(fname_times_arma);
    auto timings_arma = cg::compressTimes(raw_times_arma);

    // Read timings from all variants
    auto raw_times = cg::readRawTimes(fname_times_vars);
    const unsigned n = raw_times.size_instance - 1U;
    const unsigned shapeID = raw_times.shapeID;
    const unsigned M = raw_times.M;
    const unsigned N_v = raw_times.N;

    // Generate symbolic variants
    auto chain = cg::ID2chain(n, shapeID);
    auto Q_t = inst_gnrtor.rndInstances(chain, N_t);
    cg::Generator generator(chain);
    auto A = generator.getAlgorithms();

    auto times = cg::compressTimes(raw_times);  // times of validation instances
    auto penalty_v = times;                     // times converted to penalties
    cg::cost2penalty(M, N_v, penalty_v);
    std::vector<double> optimal_times(N_v);
    cg::getMinA(M, N_v, times, optimal_times);  // optimal times in all vars

    auto penalty_matrix_flops =
        cg::FLOPsOnInstances(A, Q_t);  // FLOPs of training instances
    cg::cost2penalty(M, N_t, penalty_matrix_flops);  // penalties Q_t w/FLOPs

    auto penalty_matrix_models =
        cg::predTimeOnInstances(A, Q_t);  // time estimation training instances
    cg::cost2penalty(M, N_t, penalty_matrix_models);  // penalties Q_t w/models

    // create vector with penalty_matrices
    std::vector<std::vector<double>> penalties_t = {
        std::move(penalty_matrix_flops), std::move(penalty_matrix_models)};

    std::set<unsigned> Es_free_dims;
    std::vector<std::vector<unsigned>> coupled_dims;
    auto free_dims_idx = cg::getFreeDimsIdx(chain, coupled_dims);
    insertFanDimIdx(A, n, Es_free_dims, free_dims_idx);
    auto C = cg::dimId2AlgId(A, n, coupled_dims);

    for (unsigned training_id = 0U; training_id < penalties_t.size();
         training_id++) {
      // Add the current shape to the results
      results[training_id]._shapeIDs.push_back(shapeID);
      // Get a reference to the penalties in training
      const std::vector<double>& penalty_t = penalties_t[training_id];

      for (unsigned funcID = 0U; funcID < obj_funcs.size(); funcID++) {
        auto F = obj_funcs[funcID];
        // Create Es on FLOPs and evaluate on validation set
        auto Es = cg::findOptimalFanningSet(M, N_t, penalties_t[0],
                                            Es_free_dims, C, F);
        double F_Es = cg::evaluateSet(M, N_v, penalty_v, Es, F);

        // Expand Es to +1 on training set and evaluate on validation set
        unsigned K1 = Es.size() + 1U;
        auto Es1 = cg::SEGreedyWPenalty(M, N_t, K1, F, Es, penalty_t);
        double F_Es1 = cg::evaluateSet(M, N_v, penalty_v, Es1, F);

        // Expand Es to +2 on training set and evaluate on validation set
        unsigned K2 = Es1.size() + 1U;
        auto Es2 = cg::SEGreedyWPenalty(M, N_t, K2, F, Es1, penalty_t);
        double F_Es2 = cg::evaluateSet(M, N_v, penalty_v, Es2, F);

        // Create set with LtR and evaluate on validation set
        std::set<unsigned> LtR = {0};  // 0 is the index of the LtR algorithm
        double F_LtR = cg::evaluateSet(M, N_v, penalty_v, LtR, F);

        double F_arma = cg::evaluateArma(M, N_v, timings_arma, times, F);

        // Compute ratios of generated sets and Arma to the optimum.
        // Ratio Es to optimum
        std::vector<double> timings_Es(N_v);
        cg::getMinSet(M, N_v, times, Es, timings_Es);
        std::vector<double> ratios_Es(N_v);
        std::transform(timings_Es.begin(), timings_Es.end(),
                       optimal_times.begin(), ratios_Es.begin(),
                       std::divides<double>());

        // Ratio Es1 to optimum
        std::vector<double> timings_Es1(N_v);
        cg::getMinSet(M, N_v, times, Es1, timings_Es1);
        std::vector<double> ratios_Es1(N_v);
        std::transform(timings_Es1.begin(), timings_Es1.end(),
                       optimal_times.begin(), ratios_Es1.begin(),
                       std::divides<double>());

        // Ratio Es2 to optimum
        std::vector<double> timings_Es2(N_v);
        cg::getMinSet(M, N_v, times, Es2, timings_Es2);
        std::vector<double> ratios_Es2(N_v);
        std::transform(timings_Es2.begin(), timings_Es2.end(),
                       optimal_times.begin(), ratios_Es2.begin(),
                       std::divides<double>());

        // Ratio LtR to optimum
        std::vector<double> timings_LtR(N_v);
        cg::getMinSet(M, N_v, times, LtR, timings_LtR);
        std::vector<double> ratios_LtR(N_v);
        std::transform(timings_LtR.begin(), timings_LtR.end(),
                       optimal_times.begin(), ratios_LtR.begin(),
                       std::divides<double>());

        // Ratio Arma to optimum
        std::vector<double> ratios_arma(N_v);
        std::transform(timings_arma.begin(), timings_arma.end(),
                       optimal_times.begin(), ratios_arma.begin(),
                       std::divides<double>());

        // Store results
        if (funcID == 0U) {
          results[training_id].storeMax(
              {static_cast<unsigned>(Es.size()), F_Es},
              {static_cast<unsigned>(Es1.size()), F_Es1},
              {static_cast<unsigned>(Es2.size()), F_Es2},
              {static_cast<unsigned>(LtR.size()), F_LtR},
              {ARMA_SET_SIZE, F_arma});

          result_ratios[training_id].storeRatioToOptimalMax(
              ratios_Es, ratios_Es1, ratios_Es2, ratios_LtR);
        } else {
          results[training_id].storeAvg(
              {static_cast<unsigned>(Es.size()), F_Es},
              {static_cast<unsigned>(Es1.size()), F_Es1},
              {static_cast<unsigned>(Es2.size()), F_Es2},
              {static_cast<unsigned>(LtR.size()), F_LtR},
              {ARMA_SET_SIZE, F_arma});

          result_ratios[training_id].storeRatioToOptimalAvg(
              ratios_Es, ratios_Es1, ratios_Es2, ratios_LtR);
          result_ratios[training_id].storeRatioToOptimalArma(ratios_arma);
        }
      }
    }
  }

  // file where results of sets trained with FLOPs are output
  std::string fname_result_flops = base_path_out + "/result_set_flops.txt";
  std::ofstream ofile_sets_flops;
  ofile_sets_flops.open(fname_result_flops);
  if (ofile_sets_flops.fail()) {
    std::cerr << "Cannot open " << fname_result_flops << '\n';
    exit(-1);
  }
  results[0].print(ofile_sets_flops);
  ofile_sets_flops.close();

  // file where results of sets trained with models are output
  std::string fname_result_models = base_path_out + "/result_set_models.txt";
  std::ofstream ofile_sets_models;
  ofile_sets_models.open(fname_result_models);
  if (ofile_sets_models.fail()) {
    std::cerr << "Cannot open " << fname_result_models << '\n';
    exit(-1);
  }
  results[1].print(ofile_sets_models);
  ofile_sets_models.close();

  // write out comparison between armadillo and generated sets with flops
  std::string fname_comparison_flops =
      base_path_out + "/ratios_optimal_sets_wflops.txt";
  std::ofstream ofile_cmp_arma_flops;
  ofile_cmp_arma_flops.open(fname_comparison_flops);
  if (ofile_cmp_arma_flops.fail()) {
    std::cerr << "Cannot open " << fname_comparison_flops << '\n';
    exit(-1);
  }
  result_ratios[0].print(ofile_cmp_arma_flops);
  ofile_cmp_arma_flops.close();

  // write out comparison between armadillo and generated sets with models
  std::string fname_comparison_models =
      base_path_out + "/ratios_optimal_sets_wmodels.txt";
  std::ofstream ofile_cmp_arma_models;
  ofile_cmp_arma_models.open(fname_comparison_models);
  if (ofile_cmp_arma_models.fail()) {
    std::cerr << "Cannot open " << fname_comparison_models << '\n';
    exit(-1);
  }
  result_ratios[1].print(ofile_cmp_arma_models);
  ofile_cmp_arma_models.close();

  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "elapsed: " << std::chrono::duration<double>(t1 - t0).count()
            << '\n';
}