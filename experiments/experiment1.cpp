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
  float _fvalue;
  unsigned _set_size;

  InfoSet(const float fvalue, const unsigned set_size)
      : _fvalue{fvalue}, _set_size{set_size} {}
};

std::ostream& operator<<(std::ostream& os, const InfoSet& info_set) {
  os << info_set._set_size << ' ' << info_set._fvalue;
  return os;
}

struct SummaryShapes {
  unsigned _procID;
  std::array<unsigned, 2U> _proc_range;
  std::vector<unsigned> _shapeIDs;
  std::vector<InfoSet> _info_Es_max;
  std::vector<InfoSet> _info_Es1_max;
  std::vector<InfoSet> _info_Es2_max;
  std::vector<InfoSet> _info_ltr_max;
  std::vector<InfoSet> _info_Es_avg;
  std::vector<InfoSet> _info_Es1_avg;
  std::vector<InfoSet> _info_Es2_avg;
  std::vector<InfoSet> _info_ltr_avg;

  SummaryShapes(const unsigned n_shapes, const unsigned procID,
                const std::array<unsigned, 2U>& proc_range) {
    _procID = procID;
    _proc_range = proc_range;
    _shapeIDs.reserve(n_shapes);
    _info_Es_max.reserve(n_shapes);
    _info_Es1_max.reserve(n_shapes);
    _info_Es2_max.reserve(n_shapes);
    _info_ltr_max.reserve(n_shapes);
    _info_Es_avg.reserve(n_shapes);
    _info_Es1_avg.reserve(n_shapes);
    _info_Es2_avg.reserve(n_shapes);
    _info_ltr_avg.reserve(n_shapes);
  }

  void print(std::ostream& os) {
    os << _procID << ' ' << _proc_range[0] << ' ' << _proc_range[1] << ' '
       << _shapeIDs.size() << '\n';
    for (unsigned i = 0U; i < _shapeIDs.size(); i++) {
      os << _shapeIDs[i] << ' ' << _info_Es_max[i] << ' ' << _info_Es1_max[i]
         << ' ' << _info_Es2_max[i] << ' ' << _info_ltr_max[i] << ' '
         << _info_Es_avg[i] << ' ' << _info_Es1_avg[i] << ' '
         << _info_Es2_avg[i] << ' ' << _info_ltr_avg[i] << '\n';
    }
  }
};

struct ValuesInstances {
  std::vector<float> _Es;
  std::vector<float> _Es1;
  std::vector<float> _Es2;
  std::vector<float> _ltr;

  ValuesInstances(const unsigned n_shapes, const unsigned N_v) {
    // total_instances is an upper bound. For some procs it will be fully used,
    // not for all.
    size_t total_instances = n_shapes * N_v;
    _Es.reserve(total_instances);
    _Es1.reserve(total_instances);
    _Es2.reserve(total_instances);
    _ltr.reserve(total_instances);
  }

  void printToFile(std::ofstream& ofile) {
    size_t size_v = _Es.size();
    ofile.write(reinterpret_cast<const char*>(_Es.data()),
                size_v * sizeof(float));
    ofile.write(reinterpret_cast<const char*>(_Es1.data()),
                size_v * sizeof(float));
    ofile.write(reinterpret_cast<const char*>(_Es2.data()),
                size_v * sizeof(float));
    ofile.write(reinterpret_cast<const char*>(_ltr.data()),
                size_v * sizeof(float));
  }

  void storeShapeInstances(std::vector<double>& Es, std::vector<double>& Es1,
                           std::vector<double>& Es2, std::vector<double>& ltr) {
    size_t size_input = Es.size();
    for (size_t i = 0UL; i < size_input; i++) {
      _Es.push_back(static_cast<float>(Es[i]));
      _Es1.push_back(static_cast<float>(Es1[i]));
      _Es2.push_back(static_cast<float>(Es2[i]));
      _ltr.push_back(static_cast<float>(ltr[i]));
    }
  }
};

/**
 * @brief Entry point.
 *
 * @arg 1: length of the chains.
 * @arg 2: Number of instances in training.
 * @arg 3: Number of instances in validation.
 * @arg 4: process id.
 * @arg 5: total number of processes. Together with the previous arg
 * determine the range of shapes covered.
 * @arg 6: path where results are dumped. Relative to working directory.
 *
 */
int main(int argc, char** argv) {
  unsigned n, N_t, N_v, procID, n_proc;
  std::string path_out;

  if (argc < 7) {
    std::cerr
        << "Error. Arguments: <n> <N_t> <N_v> <procID> <n_proc> <path_out>"
           "\nCheck the code for interpretation of the arguments\n";
    exit(-1);
  } else {
    n = std::stoi(argv[1]);
    N_t = std::stoi(argv[2]);
    N_v = std::stoi(argv[3]);
    procID = static_cast<unsigned>(std::stoi(argv[4]));
    n_proc = static_cast<unsigned>(std::stoi(argv[5]));
    path_out = argv[6];
  }
  auto t0 = std::chrono::high_resolution_clock::now();

  const unsigned M = cg::catalan(n - 1U);
  std::vector<metricFunc> obj_funcs = {cg::maxPenalty, cg::avgPenalty};

  cg::InstanceGenerator inst_gnrtor(2U, 1000U);
  unsigned max_value = cg::getMaxValue(n);
  auto ranges = cg::getProcRange(max_value, procID, n_proc);
  unsigned n_shapes = ranges[1] - ranges[0];

  // Data-structures to store summaries and values on instances
  SummaryShapes summaries(n_shapes, procID, ranges);
  ValuesInstances values_instances(n_shapes, N_v);

  for (unsigned shapeID = ranges[0]; shapeID < ranges[1]; shapeID++) {
    if (cg::isSquareShape(n, shapeID)) continue;

    summaries._shapeIDs.push_back(shapeID);
    auto chain = cg::ID2chain(n, shapeID);            // symbolic chain
    auto Q_t = inst_gnrtor.rndInstances(chain, N_t);  // instances training
    auto Q_v = inst_gnrtor.rndInstances(chain, N_v);  // instances validation
    cg::Generator generator(chain);
    auto A = generator.getAlgorithms();  // set of all variants

    auto ratio_matrix_t = cg::FLOPsOnInstances(A, Q_t);
    cg::cost2RatioOverOptimal(M, N_t, ratio_matrix_t);

    auto ratio_matrix_v = cg::FLOPsOnInstances(A, Q_v);
    cg::cost2RatioOverOptimal(M, N_v, ratio_matrix_v);

    std::set<unsigned> Es_free_dims;
    std::vector<std::vector<unsigned>> coupled_dims;
    auto free_dims_idx = cg::getFreeDimsIdx(chain, coupled_dims);
    insertFanDimIdx(A, n, Es_free_dims, free_dims_idx);
    auto C = cg::dimId2AlgId(A, n, coupled_dims);

    // for each objective function (Fmax and Favg):
    for (unsigned funcID = 0U; funcID < obj_funcs.size(); funcID++) {
      auto F = obj_funcs[funcID];

      // Create Es on training set with objective function F
      auto Es =
          cg::findOptimalFanningSet(M, N_t, ratio_matrix_t, Es_free_dims, C, F);
      // Evaluate Es on validation set of instances
      double F_Es = cg::evaluateSet(M, N_v, ratio_matrix_v, Es, F);

      // Expand Es to +1 on training set of instances
      unsigned K1 = Es.size() + 1U;
      auto Es1 = cg::SEGreedyWPenalty(M, N_t, K1, F, Es, ratio_matrix_t);
      // Evaluate Es+1 on validation set of instances
      double F_Es1 = cg::evaluateSet(M, N_v, ratio_matrix_v, Es1, F);

      // Expand Es to +2 on training set of instances
      unsigned K2 = Es1.size() + 1U;
      auto Es2 = cg::SEGreedyWPenalty(M, N_t, K2, F, Es1, ratio_matrix_t);
      // Evaluate Es+2 on validation set of instances
      double F_Es2 = cg::evaluateSet(M, N_v, ratio_matrix_v, Es2, F);

      // Create set with LtR
      std::set<unsigned> LtR = {0};  // 0 is the index of the LtR algorithm
      // Evaluate LtR on validation set of instances
      double F_LtR = cg::evaluateSet(M, N_v, ratio_matrix_v, LtR, F);

      // Store summary of current shape
      if (funcID == 0U) {
        summaries._info_Es_max.push_back(
            {static_cast<float>(F_Es), static_cast<unsigned>(Es.size())});
        summaries._info_Es1_max.push_back(
            {static_cast<float>(F_Es1), static_cast<unsigned>(Es1.size())});
        summaries._info_Es2_max.push_back(
            {static_cast<float>(F_Es2), static_cast<unsigned>(Es2.size())});
        summaries._info_ltr_max.push_back(
            {static_cast<float>(F_LtR), static_cast<unsigned>(LtR.size())});
      } else {
        summaries._info_Es_avg.push_back(
            {static_cast<float>(F_Es), static_cast<unsigned>(Es.size())});
        summaries._info_Es1_avg.push_back(
            {static_cast<float>(F_Es1), static_cast<unsigned>(Es1.size())});
        summaries._info_Es2_avg.push_back(
            {static_cast<float>(F_Es2), static_cast<unsigned>(Es2.size())});
        summaries._info_ltr_avg.push_back(
            {static_cast<float>(F_LtR), static_cast<unsigned>(LtR.size())});

        // Compute the ratios/penalties per-instance and store them
        std::vector<double> ratios_Es(static_cast<size_t>(N_v));
        std::vector<double> ratios_Es1(static_cast<size_t>(N_v));
        std::vector<double> ratios_Es2(static_cast<size_t>(N_v));
        std::vector<double> ratios_ltr(static_cast<size_t>(N_v));
        cg::getMinSet(M, N_v, ratio_matrix_v, Es, ratios_Es);
        cg::getMinSet(M, N_v, ratio_matrix_v, Es1, ratios_Es1);
        cg::getMinSet(M, N_v, ratio_matrix_v, Es2, ratios_Es2);
        cg::getMinSet(M, N_v, ratio_matrix_v, LtR, ratios_ltr);
        values_instances.storeShapeInstances(ratios_Es, ratios_Es1, ratios_Es2,
                                             ratios_ltr);
      }
    }
  }

  std::string fname_summary =
      path_out + "/summary_proc_" + std::to_string(procID) + ".txt";
  std::ofstream ofile_summary;
  ofile_summary.open(fname_summary);
  if (ofile_summary.fail()) {
    std::cerr << "Cannot open " << fname_summary << '\n';
    exit(-1);
  }
  summaries.print(ofile_summary);
  ofile_summary.close();
  std::cout << "summaries dumped to " << fname_summary << std::endl;

  std::string fname_instances =
      path_out + "/instances_proc_" + std::to_string(procID) + ".bin";
  std::ofstream ofile_instances;
  ofile_instances.open(fname_instances);
  if (ofile_instances.fail()) {
    std::cerr << "Cannot open " << fname_instances << '\n';
    exit(-1);
  }
  values_instances.printToFile(ofile_instances);
  ofile_instances.close();
  std::cout << "per instance values dumped to " << fname_instances << std::endl;

  auto t1 = std::chrono::high_resolution_clock::now();
  std::cout << "elapsed: " << std::chrono::duration<double>(t1 - t0).count()
            << '\n';
}