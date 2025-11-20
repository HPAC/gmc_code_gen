#include "analyzer.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <set>
#include <vector>

#include "algorithm.hpp"
#include "definitions.hpp"
#include "utils/common.hpp"
#include "utils/dMatrix.hpp"
#include "utils/matrix_generator.hpp"

using std::vector;

namespace cg {

vector<double> FLOPsOnInstance(vector<Algorithm>& A, const Instance& q) {
  const unsigned M = A.size();
  vector<double> flops(static_cast<size_t>(M));

  for (unsigned i = 0; i < M; i++) flops[i] = A[i].computeFLOPs(q);
  return flops;
}

vector<double> FLOPsOnInstances(vector<Algorithm>& A,
                                const vector<Instance>& Q) {
  const unsigned M = A.size();
  const unsigned N = Q.size();
  vector<double> flops_matrix(static_cast<size_t>(M) * static_cast<size_t>(N));

  for (unsigned i = 0U; i < M; i++) {
    for (unsigned j = 0U; j < N; j++) {
      flops_matrix[i * N + j] = A[i].computeFLOPs(Q[j]);
    }
  }
  return flops_matrix;
}

vector<double> predTimeOnInstances(vector<Algorithm>& A,
                                   const vector<Instance>& Q) {
  const unsigned M = A.size();
  const unsigned N = Q.size();
  vector<double> time_matrix(static_cast<size_t>(M) * static_cast<size_t>(N));

  for (unsigned i = 0U; i < M; i++) {
    for (unsigned j = 0U; j < N; j++) {
      time_matrix[i * N + j] = A[i].predictTime(Q[j]);
    }
  }
  return time_matrix;
}

vector<double> timeAlg(Algorithm& alg, const vector<dMatrix>& live_chain,
                       const unsigned n_rep) {
  vector<double> times_instance(n_rep);
  for (unsigned rep = 0; rep < n_rep; rep++) {
    alg.assignChain(live_chain);  // initialize live matrix chain in algorithm
    auto t0 = std::chrono::high_resolution_clock::now();
    alg.execute();
    auto t1 = std::chrono::high_resolution_clock::now();
    times_instance[rep] = std::chrono::duration<double>(t1 - t0).count();
    alg.clean();  // remove matrices in the algorithm
  }
  return times_instance;
}

vector<double> avgTimings(const MatrixChain& chain, vector<Algorithm>& A,
                          const vector<Instance>& Q, unsigned n_rep) {
  unsigned M = A.size();
  unsigned N = Q.size();
  vector<double> time_matrix(static_cast<size_t>(M) * static_cast<size_t>(N));

  vector<dMatrix> live_chain;
  vector<double> times_algorithm;

  for (unsigned j = 0U; j < N; j++) {
    // Create matrices with features in chain and with the sizes in S[j]
    live_chain = MatrixGenerator::generateChain(chain, Q[j]);
    for (unsigned i = 0U; i < M; i++) {
      auto times_algorithm = timeAlg(A[i], live_chain, n_rep);
      time_matrix[i * N + j] =
          std::accumulate(times_algorithm.begin(), times_algorithm.end(), 0.0) /
          static_cast<double>(times_algorithm.size());
    }  // loop on algorithms
  }  // loop on instances
  return time_matrix;
}

vector<double> timeAlgsOnInstances(const MatrixChain& chain,
                                   vector<Algorithm>& A,
                                   const vector<Instance>& Q, unsigned reps) {
  const unsigned M = A.size();
  const unsigned N = Q.size();
  vector<double> times(M * N * reps);

  vector<dMatrix> live_chain;
  vector<double>::iterator it_times = times.begin();

  for (unsigned j = 0U; j < N; j++) {
    std::cout << "Computing instance: " << j << std::endl;
    live_chain = MatrixGenerator::generateChain(chain, Q[j]);
    for (unsigned i = 0U; i < M; i++) {
      auto times_alg = timeAlg(A[i], live_chain, reps);
      std::copy(times_alg.begin(), times_alg.end(),
                it_times + (i * N + j) * reps);
    }  // loop on algorithms
  }  // loop on instances
  return times;
}

Permutation getEssentialPerm(const unsigned N, const unsigned min_dim) {
  Permutation perm;
  perm.reserve(N - 1);

  for (unsigned i = min_dim - 1; i > 0 and i < N; --i) perm.push_back(i);
  for (unsigned i = min_dim + 1; i < N; ++i) perm.push_back(i);

  if (perm.size() < N - 1) perm.push_back(min_dim);
  return perm;
}

vector<Permutation> getEssentialPerms(const unsigned N) {
  vector<Permutation> essentials;
  unsigned n_dims = N + 1;
  essentials.reserve(n_dims);
  for (unsigned i = 0; i < n_dims; ++i)
    essentials.push_back(getEssentialPerm(N, i));

  return essentials;
}

std::set<unsigned> fanningSet(const MatrixChain& chain,
                              const vector<Algorithm>& A) {
  std::set<unsigned> E;
  vector<cg::Permutation> essential_perms = cg::getEssentialPerms(chain.size());
  for (const auto& perm : essential_perms) E.insert(cg::getID(A, perm));
  return E;
}

std::set<unsigned> getEssentialSetMin(const MatrixChain& chain,
                                      const vector<Algorithm>& A) {
  std::set<unsigned> E;
  const unsigned chain_N = chain.size();
  vector<unsigned> indices(chain_N + 1U);
  countFreeDimensions(chain, indices);

  unsigned count = 0U;
  E.insert(getID(A, getEssentialPerm(chain_N, 0U)));

  for (unsigned i = 1U; i < indices.size(); i++) {
    if (indices[i] != count) {
      E.insert(getID(A, getEssentialPerm(chain_N, i)));
      count++;
    }
  }
  return E;
}

unsigned countFreeDimensions(const MatrixChain& chain,
                             vector<unsigned>& indices) {
  unsigned free_dims = 1U;  // there must be at least a free dimension.
  indices[0] = 0;
  for (unsigned i = 0; i < chain.size(); ++i) {
    if (chain[i].isDense() and !chain[i].isInvertible()) {
      indices[i + 1] = free_dims;
      free_dims++;
    } else
      indices[i + 1] = indices[i];
  }
  return free_dims;
}

vector<unsigned> getFreeDimsIdx(const MatrixChain& chain,
                                vector<vector<unsigned>>& C) {
  vector<unsigned> free_idx;  // idx of free dims.

  // aux tmp vector. This vector will hold either:
  //    - sequences of dims idx that are coupled.
  //    - a single dim idx that is free.
  vector<unsigned> v_x = {0U};
  C.clear();

  for (unsigned i = 0U; i < chain.size(); i++) {
    if (chain[i].isDense() and !chain[i].isInvertible()) {
      if (v_x.size() == 1U)  // add dimension to free_idx.
        free_idx.push_back(v_x[0]);
      else  // add coupled dimensions to C.
        C.push_back(v_x);
      v_x.clear();
      v_x.push_back(i + 1U);
    } else
      v_x.push_back(i + 1U);
  }

  if (v_x.size() == 1U)
    free_idx.push_back(v_x[0]);
  else
    C.push_back(v_x);

  return free_idx;
}

void insertFanDimIdx(const vector<Algorithm>& A, const unsigned n,
                     std::set<unsigned>& Z,
                     const vector<unsigned>& free_dims_idx) {
  for (const auto& dim : free_dims_idx)
    Z.insert(cg::getID(A, cg::getEssentialPerm(n, dim)));
}

vector<vector<unsigned>> dimId2AlgId(
    const vector<Algorithm>& A, const unsigned n,
    const vector<vector<unsigned>>& coupled_dims_id) {
  vector<vector<unsigned>> coupled_alg_ids;
  vector<unsigned> vec;
  for (const auto& v : coupled_dims_id) {
    vec.clear();
    for (const auto& dim : v)
      vec.push_back(cg::getID(A, cg::getEssentialPerm(n, dim)));
    coupled_alg_ids.push_back(vec);
  }
  return coupled_alg_ids;
}

Instance getInstance(const Instance& base_instance, vector<unsigned> indices) {
  Instance instance(indices.size());

  for (unsigned i = 0; i < indices.size(); ++i) {
    instance[i] = base_instance[indices[i]];
  }
  return instance;
}

unsigned getID(const vector<Algorithm>& algs, const Permutation& perm) {
  auto iter = std::find_if(
      algs.begin(), algs.end(),
      [perm](const Algorithm& alg) { return alg.getPermutation() == perm; });
  return std::distance(algs.begin(), iter);
}

vector<unsigned> getIDs(const vector<Algorithm>& algs,
                        const vector<Permutation>& perms) {
  vector<unsigned> IDs;
  IDs.reserve(perms.size());
  for (const auto& perm : perms) {
    IDs.push_back(getID(algs, perm));
  }
  return IDs;
}

std::set<unsigned> SEGreedy(
    const vector<Algorithm>& A, const vector<Instance>& S, const unsigned K,
    const std::function<double(const vector<double>&)>& F,
    const std::set<unsigned>& Z0, const vector<double>& cost_matrix) {
  std::set<unsigned> Z = Z0;
  const unsigned M = A.size();
  const unsigned N = S.size();

  // this matrix is in row-major format, where each row corresponds to
  // a different algorithm. This takes roughly 0.164067s, for M=132, N=10^5.
  vector<double> penalty_matrix(static_cast<size_t>(M) *
                                static_cast<size_t>(N));
  getPenaltyMatrix(M, N, cost_matrix, penalty_matrix);

  vector<double> penalty_Z(static_cast<size_t>(N));
  getMinSet(M, N, penalty_matrix, Z, penalty_Z);
  vector<double> penalty_Z_U_min(static_cast<size_t>(N));
  vector<double> penalty_Z_U_can(static_cast<size_t>(N));

  double min_F;
  double val_F = Z.empty() ? std::numeric_limits<double>::max() : F(penalty_Z);

  int id_selected = 0;
  while (Z.size() < K and id_selected != -1) {
    id_selected =
        selectCandidate(M, N, penalty_matrix, penalty_Z, penalty_Z_U_min,
                        penalty_Z_U_can, Z, F, val_F, min_F);
    if (id_selected != -1) {
      Z.insert(static_cast<unsigned>(id_selected));
      std::swap(penalty_Z, penalty_Z_U_min);
      val_F = min_F;
    }
  }
  printMetrics(penalty_Z);  // should this be printed here?
  return Z;
}

std::set<unsigned> SEGreedyWPenalty(
    const unsigned M, const unsigned N_t, const unsigned K,
    const std::function<double(const vector<double>&)>& F,
    const std::set<unsigned>& Z0, const vector<double>& penalty_matrix_t) {
  std::set<unsigned> Z = Z0;

  vector<double> penalty_Z(static_cast<size_t>(N_t));
  vector<double> penalty_Z_U_min(static_cast<size_t>(N_t));
  vector<double> penalty_Z_U_can(static_cast<size_t>(N_t));
  getMinSet(M, N_t, penalty_matrix_t, Z, penalty_Z);

  double min_F;
  double val_F = Z.empty() ? std::numeric_limits<double>::max() : F(penalty_Z);

  int id_selected = 0;
  while (Z.size() < K and id_selected != -1) {
    id_selected =
        selectCandidate(M, N_t, penalty_matrix_t, penalty_Z, penalty_Z_U_min,
                        penalty_Z_U_can, Z, F, val_F, min_F);
    if (id_selected != -1) {
      Z.insert(static_cast<unsigned>(id_selected));
      std::swap(penalty_Z, penalty_Z_U_min);
      val_F = min_F;
    }
  }
  return Z;
}

// @todo create a similar function that takes penalty matrices as input (or
// modify this one).
InfoGreedy SEGreedyStepVal(
    const unsigned M, const unsigned N_t, const unsigned N_v, const unsigned K,
    const std::function<double(const vector<double>&)>& F,
    std::set<unsigned>& Z, const vector<double>& train_matrix,
    const vector<double>& val_matrix) {
  InfoGreedy info_greedy;

  vector<double> penalty_matrix_t(static_cast<size_t>(M * N_t));
  getPenaltyMatrix(M, N_t, train_matrix, penalty_matrix_t);
  vector<double> penalty_matrix_v(static_cast<size_t>(M * N_v));
  getPenaltyMatrix(M, N_v, val_matrix, penalty_matrix_v);

  vector<double> penalty_Z_t(static_cast<size_t>(N_t));
  vector<double> penalty_Z_U_min(static_cast<size_t>(N_t));
  vector<double> penalty_Z_U_can(static_cast<size_t>(N_t));
  getMinSet(M, N_t, penalty_matrix_t, Z, penalty_Z_t);

  // penalty of Z in validation (time).
  vector<double> penalty_Z_v(static_cast<size_t>(N_v));

  double min_F;
  double value_F =
      Z.empty() ? std::numeric_limits<double>::max() : F(penalty_Z_t);

  if (!Z.empty()) {
    getMinSet(M, N_v, penalty_matrix_v, Z, penalty_Z_v);
    info_greedy.recordStep(M, Z.size(), computeAllF(penalty_Z_v));
  }

  int id_candidate;
  while (Z.size() < K and id_candidate != -1) {
    // Select single candidate that diminishes F the most in training.
    id_candidate =
        selectCandidate(M, N_t, penalty_matrix_t, penalty_Z_t, penalty_Z_U_min,
                        penalty_Z_U_can, Z, F, value_F, min_F);
    if (id_candidate != -1) {
      Z.insert(static_cast<unsigned>(id_candidate));
      std::swap(penalty_Z_t, penalty_Z_U_min);
      value_F = min_F;
      // Only a new algorithm is included. We can minimize between the penalty
      // of the previous set and the penalty of the new algorithm.
      getMinSet(M, N_v, penalty_matrix_v, Z, penalty_Z_v);
      info_greedy.recordStep(id_candidate, Z.size(), computeAllF(penalty_Z_v));
    }
  }
  info_greedy.recordSet(Z);

  return info_greedy;
}

InfoGreedy SEGreedyStep(const unsigned M, const unsigned N, const unsigned K,
                        const std::function<double(const vector<double>&)>& F,
                        std::set<unsigned>& Z,
                        const vector<double>& penalty_matrix) {
  InfoGreedy info_greedy;

  vector<double> penalty_Z(static_cast<size_t>(N));
  vector<double> penalty_Z_U_min(static_cast<size_t>(N));
  vector<double> penalty_Z_U_can(static_cast<size_t>(N));
  getMinSet(M, N, penalty_matrix, Z, penalty_Z);
  if (!Z.empty()) info_greedy.recordStep(M, Z.size(), computeAllF(penalty_Z));

  double min_F;
  double value_F =
      Z.empty() ? std::numeric_limits<double>::max() : F(penalty_Z);

  int id_selected = 0;
  while (Z.size() < K and id_selected != -1) {
    id_selected =
        selectCandidate(M, N, penalty_matrix, penalty_Z, penalty_Z_U_min,
                        penalty_Z_U_can, Z, F, value_F, min_F);
    if (id_selected != -1) {
      value_F = min_F;
      Z.insert(static_cast<unsigned>(id_selected));
      std::swap(penalty_Z, penalty_Z_U_min);
      info_greedy.recordStep(id_selected, Z.size(), computeAllF(penalty_Z));
    }
  }
  info_greedy.recordSet(Z);

  return info_greedy;
}

double penalty(const double min_A, const double min_Z) {
  double penalty_value = (min_Z / min_A) - 1.0;
  if (penalty_value < 0.0001) penalty_value = 0.0;
  return penalty_value;
}

void getMinA(const unsigned M, const unsigned N,
             const vector<double>& cost_matrix, vector<double>& min_A) {
  for (unsigned j = 0U; j < N; j++) {
    min_A[j] = std::numeric_limits<double>::max();
    for (unsigned i = 0; i < M; i++) {
      if (cost_matrix[i * N + j] < min_A[j]) min_A[j] = cost_matrix[i * N + j];
    }
  }
}

void getMinSet(const unsigned M, const unsigned N,
               const vector<double>& cost_matrix, const std::set<unsigned>& Z,
               vector<double>& min_Z) {
  for (unsigned j = 0U; j < N; j++) {
    min_Z[j] = std::numeric_limits<double>::max();
    for (const auto& id : Z) {
      if (cost_matrix[id * N + j] < min_Z[j])
        min_Z[j] = cost_matrix[id * N + j];
    }
  }
}

void minRanges(const unsigned N, vector<double>::const_iterator& it_A,
               vector<double>::const_iterator& it_B,
               vector<double>::iterator& it_C) {
  for (unsigned i = 0U; i < N; i++)
    it_C[i] = (it_A[i] < it_B[i]) ? it_A[i] : it_B[i];
}

void getPenaltyZ(const vector<double>& min_A, const vector<double>& min_Z,
                 vector<double>& penalty_Z) {
  for (unsigned j = 0U; j < penalty_Z.size(); j++) {
    penalty_Z[j] = penalty(min_A[j], min_Z[j]);
  }
}

void getPenaltyMatrix(const unsigned M, const unsigned N,
                      const vector<double>& cost_matrix,
                      vector<double>& penalty_matrix) {
  vector<double> min_A(static_cast<size_t>(N));
  getMinA(M, N, cost_matrix, min_A);

  for (unsigned i = 0U; i < M; i++) {
    for (unsigned j = 0U; j < N; j++) {
      penalty_matrix[i * N + j] = penalty(min_A[i], cost_matrix[i * N + j]);
    }
  }
}

void cost2penalty(const unsigned M, const unsigned N,
                  vector<double>& cost_matrix) {
  for (unsigned j = 0U; j < N; j++) {
    double min_A = std::numeric_limits<double>::max();
    for (unsigned i = 0U; i < M; i++) {
      if (cost_matrix[i * N + j] < min_A) min_A = cost_matrix[i * N + j];
    }

    for (unsigned i = 0U; i < M; i++)
      cost_matrix[i * N + j] = penalty(min_A, cost_matrix[i * N + j]);
  }
}

void cost2RatioOverOptimal(const unsigned M, const unsigned N,
                           vector<double>& cost_matrix) {
  for (unsigned j = 0U; j < N; j++) {
    double min_A = std::numeric_limits<double>::max();
    for (unsigned i = 0U; i < M; i++) {
      if (cost_matrix[i * N + j] < min_A) min_A = cost_matrix[i * N + j];
    }

    for (unsigned i = 0U; i < M; i++)
      cost_matrix[i * N + j] = cost_matrix[i * N + j] / min_A;
  }
}

double maxPenalty(const vector<double>& penalty) {
  double max_penalty = -1.0;
  for (unsigned i = 0; i < penalty.size(); i++)
    if (penalty[i] > max_penalty) max_penalty = penalty[i];

  return max_penalty;
}

double freqPenalty(const vector<double>& penalty) {
  unsigned count_nnz = 0U;
  for (unsigned i = 0; i < penalty.size(); i++)
    if (penalty[i] > 0.0) count_nnz++;

  return static_cast<double>(count_nnz) / static_cast<double>(penalty.size());
}

double avgPenalty(const vector<double>& penalty) {
  double avg_penalty = 0.0;
  for (unsigned i = 0; i < penalty.size(); i++) avg_penalty += penalty[i];

  return avg_penalty / static_cast<double>(penalty.size());
}

double avgNnzPenalty(const vector<double>& penalty) {
  double nnz_penalty = 0.0;
  unsigned count_nnz = 0U;
  for (unsigned i = 0U; i < penalty.size(); i++) {
    if (penalty[i] > 0.0) {
      nnz_penalty += penalty[i];
      count_nnz++;
    }
  }
  if (count_nnz == 0U) count_nnz = 1U;  // avoid division by zero.
  return nnz_penalty / static_cast<double>(count_nnz);
}

double sqPenalty(const vector<double>& penalty) {
  double avg_penalty = 0.0;
  for (unsigned i = 0; i < penalty.size(); i++) {
    avg_penalty += penalty[i] * penalty[i];
  }
  return avg_penalty / static_cast<double>(penalty.size());
}

void printMetrics(const vector<double>& penalty) {
  double max_penalty = maxPenalty(penalty);
  double avg_penalty = avgPenalty(penalty);
  double freq_penalty = freqPenalty(penalty);

  std::cout << "\n================================================\n"
            << "max_penalty: " << max_penalty << "\n"
            << "freq_penalty: " << freq_penalty << "\n"
            << "avg_penalty: " << avg_penalty << "\n"
            << "avg_penalty (only non-zero): " << avg_penalty / freq_penalty
            << "\n================================================\n\n";
}

std::array<double, 4U> computeAllF(const vector<double>& penalty) {
  std::array<double, 4U> result;
  result[0] = maxPenalty(penalty);
  result[1] = freqPenalty(penalty);
  result[2] = avgPenalty(penalty);
  result[3] = avgNnzPenalty(penalty);
  return result;
}

void printSetPerformance(const unsigned M, const unsigned N,
                         const std::set<unsigned>& Z,
                         const vector<double>& cost_matrix) {
  vector<double> min_A(static_cast<size_t>(N));
  getMinA(M, N, cost_matrix, min_A);

  vector<double> min_Z(static_cast<size_t>(N));
  getMinSet(M, N, cost_matrix, Z, min_Z);

  vector<double> penalty_Z(static_cast<size_t>(N));
  getPenaltyZ(min_A, min_Z, penalty_Z);

  printMetrics(penalty_Z);
}

int selectCandidate(const unsigned M, const unsigned N,
                    const vector<double>& penalty_matrix,
                    const vector<double>& penalty_Z,
                    vector<double>& penalty_Z_U_min,
                    vector<double>& penalty_Z_U_can,
                    const std::set<unsigned>& Z,
                    const std::function<double(const vector<double>&)>& F,
                    double val_F, double& min_F) {
  int id_selected = -1;
  min_F = val_F;
  double tmp_val_F;
  auto it_Z = penalty_Z.begin();
  auto it_Z_U_can = penalty_Z_U_can.begin();

  for (unsigned id = 0U; id < M; id++) {
    auto it = Z.find(id);
    if (it == Z.end()) {  // if the element is not in the set...
      auto it_penalty_matrix = penalty_matrix.begin() + (id * N);
      minRanges(N, it_Z, it_penalty_matrix, it_Z_U_can);
      tmp_val_F = F(penalty_Z_U_can);

      if (tmp_val_F < min_F) {
        min_F = tmp_val_F;
        std::swap(penalty_Z_U_can, penalty_Z_U_min);
        it_Z_U_can = penalty_Z_U_can.begin();  // reassign iterator to first
                                               // element of swapped buffer.
        id_selected = static_cast<int>(id);
      }
    }
  }
  return id_selected;
}

double evalUnion(const unsigned M, const unsigned N, const unsigned id,
                 const vector<double>& cost_matrix, const vector<double>& min_A,
                 vector<double>& min_Z, vector<double>& penalty_Z,
                 std::set<unsigned>& Z,
                 const std::function<double(const vector<double>&)>& F) {
  Z.insert(id);  // Include the algorithm in the set.

  // Get the value of F(Z \cup id).
  getMinSet(M, N, cost_matrix, Z, min_Z);
  getPenaltyZ(min_A, min_Z, penalty_Z);
  double value_F = F(penalty_Z);

  // Return the set to initial condition --> erase id.
  Z.erase(id);
  return value_F;
}

std::set<unsigned> getOptimalFanningSet(
    const unsigned M, const unsigned N, const vector<double>& penalty_matrix,
    const std::set<unsigned>& Z, const vector<vector<unsigned>>& C,
    const std::function<double(const vector<double>&)>& F,
    std::array<double, 4U>& metrics) {
  std::set<unsigned> Z1 = Z;
  std::set<unsigned> Z2;  // Z2: set whose union with Z1 minimises F the most.
  std::set<unsigned> Z_tmp;  // Z_tmp: temporary set to be tested.

  vector<double> penalty_Z1(N);     // Temporary penalty_Z
  vector<double> penalty_U(N);      // penalty of union
  vector<double> penalty_Z_tmp(N);  // penalty of candidate set

  getMinSet(M, N, penalty_matrix, Z1, penalty_Z1);
  vector<double> penalty_min = penalty_Z1;  // penalty of best union

  double min_F = Z.empty() ? std::numeric_limits<double>::max() : F(penalty_Z1);
  double value_F = std::numeric_limits<double>::max();

  vector<unsigned> indices_C(C.size());
  // Get total number of combinations.
  unsigned n_combs = C.empty() ? 0U : 1U;
  for (unsigned i = 0U; i < C.size(); i++) n_combs *= C[i].size();

  vector<double>::const_iterator it_Z1 = penalty_Z1.begin();
  vector<double>::const_iterator it_Z_tmp = penalty_Z_tmp.begin();
  vector<double>::iterator it_Z_Union = penalty_U.begin();

  // Try all combinations. Keep the one that minimizes F the most.
  for (unsigned i = 0U; i < n_combs; i++) {
    Z_tmp = formCandidateSet(C, indices_C);
    getMinSet(M, N, penalty_matrix, Z_tmp, penalty_Z_tmp);
    minRanges(N, it_Z1, it_Z_tmp, it_Z_Union);
    value_F = F(penalty_U);

    if (value_F < min_F) {
      std::swap(Z2, Z_tmp);
      min_F = value_F;
      std::swap(penalty_U, penalty_min);
      it_Z_Union = penalty_U.begin();  // reassign iterator to first element of
                                       // swapped buffer.
    }
  }
  metrics = computeAllF(penalty_min);
  for (const auto& alg : Z2) Z1.insert(alg);  // insert elements of Z2 into Z1.

  return Z1;
}

std::set<unsigned> findOptimalFanningSet(
    const unsigned M, const unsigned N_t,
    const vector<double>& penalty_matrix_t, const std::set<unsigned>& Z,
    const vector<vector<unsigned>>& C,
    const std::function<double(const vector<double>&)>& F) {
  std::set<unsigned> Z1 = Z;
  std::set<unsigned> Z2;  // Z2: set whose union with Z1 minimizes F the most.
  std::set<unsigned> Z_tmp;  // Z_tmp: temporary set to be tested.

  vector<double> penalty_Z1(N_t);     // Temporary penalty_Z
  vector<double> penalty_U(N_t);      // penalty of union
  vector<double> penalty_Z_tmp(N_t);  // penalty of candidate set

  getMinSet(M, N_t, penalty_matrix_t, Z1, penalty_Z1);
  vector<double> penalty_min = penalty_Z1;  // penalty of best union

  double min_F = Z.empty() ? std::numeric_limits<double>::max() : F(penalty_Z1);
  double value_F = std::numeric_limits<double>::max();

  vector<unsigned> indices_C(C.size());
  // Get total number of combinations.
  unsigned n_combs = C.empty() ? 0U : 1U;
  for (unsigned i = 0U; i < C.size(); i++) n_combs *= C[i].size();

  vector<double>::const_iterator it_Z1 = penalty_Z1.begin();
  vector<double>::const_iterator it_Z_tmp = penalty_Z_tmp.begin();
  vector<double>::iterator it_Z_Union = penalty_U.begin();

  // Try all combinations. Keep the one that minimizes F the most.
  for (unsigned i = 0U; i < n_combs; i++) {
    Z_tmp = formCandidateSet(C, indices_C);
    getMinSet(M, N_t, penalty_matrix_t, Z_tmp, penalty_Z_tmp);
    minRanges(N_t, it_Z1, it_Z_tmp, it_Z_Union);
    value_F = F(penalty_U);

    if (value_F < min_F) {
      std::swap(Z2, Z_tmp);
      min_F = value_F;
      std::swap(penalty_U, penalty_min);
      it_Z_Union = penalty_U.begin();  // reassign iterator to first element of
                                       // swapped buffer.
    }
  }
  for (const auto& alg : Z2) Z1.insert(alg);  // insert elements of Z2 into Z1.

  return Z1;
}

double evaluateSet(const unsigned M, const unsigned N_v,
                   const vector<double>& penalty_matrix_v,
                   const std::set<unsigned>& Z,
                   const std::function<double(const vector<double>&)>& F) {
  vector<double> penalty_Z(N_v);
  getMinSet(M, N_v, penalty_matrix_v, Z, penalty_Z);
  return F(penalty_Z);
}

double evaluateArma(const unsigned M, const unsigned N_v,
                    const vector<double>& times_arma,
                    const vector<double>& times_vars,
                    const std::function<double(const vector<double>&)>& F) {
  vector<double> penalty_arma;
  penalty_arma.reserve(N_v);

  for (unsigned j = 0U; j < N_v; j++) {
    double min_A = std::numeric_limits<double>::max();
    for (unsigned i = 0U; i < M; i++) {
      if (times_vars[i * N_v + j] < min_A) min_A = times_vars[i * N_v + j];
    }

    penalty_arma.push_back(penalty(min_A, times_arma[j]));
  }
  return F(penalty_arma);
}

std::array<double, 4U> getOptimalFanningMetrics(
    const unsigned M, const unsigned N, const vector<double>& penalty_matrix,
    const std::set<unsigned>& Z, const vector<vector<unsigned>>& C,
    const std::function<double(const vector<double>&)>& F) {
  std::set<unsigned> Z1 = Z;
  std::set<unsigned> Z2;     // Z2: set that minimizes F the most.
  std::set<unsigned> Z_tmp;  // Z_tmp: temporary set to be tested.

  vector<double> penalty_Z1(N);     // Temporary penalty_Z
  vector<double> penalty_U(N);      // penalty of union.
  vector<double> penalty_Z_tmp(N);  // penalty of candidate set.
  vector<double> penalty_min(N);    // penalty of best union.
  getMinSet(M, N, penalty_matrix, Z1, penalty_Z1);
  getMinSet(M, N, penalty_matrix, Z1, penalty_min);
  double min_F = Z.empty() ? std::numeric_limits<double>::max() : F(penalty_Z1);
  double value_F = std::numeric_limits<double>::max();

  // Get total number of combinations.
  vector<unsigned> indices_C(C.size());
  unsigned n_combs = C.empty() ? 0U : 1U;
  for (unsigned i = 0U; i < C.size(); i++) n_combs *= C[i].size();

  vector<double>::const_iterator it_Z1 = penalty_Z1.begin();
  vector<double>::const_iterator it_Z_tmp = penalty_Z_tmp.begin();
  vector<double>::iterator it_Z_Union = penalty_U.begin();

  // Try all combinations. Keep the one that minimises F the most.
  for (unsigned i = 0U; i < n_combs; i++) {
    Z_tmp = formCandidateSet(C, indices_C);
    getMinSet(M, N, penalty_matrix, Z_tmp, penalty_Z_tmp);
    minRanges(N, it_Z1, it_Z_tmp, it_Z_Union);
    value_F = F(penalty_U);

    if (value_F < min_F) {
      Z2 = Z_tmp;
      min_F = value_F;
      std::swap(penalty_U, penalty_min);
      it_Z_Union = penalty_U.begin();
    }
  }
  auto ret_val = computeAllF(penalty_min);

  return ret_val;
}

std::set<unsigned> formCandidateSet(const vector<vector<unsigned>>& C,
                                    const vector<unsigned>& indices_C) {
  std::set<unsigned> X;
  for (unsigned i = 0; i < C.size(); i++) {
    X.insert(C[i][indices_C[i]]);
  }
  return X;
};

void updateCandidateIndices(const vector<vector<unsigned>>& C,
                            vector<unsigned>& indices_C, unsigned set_id) {
  indices_C[set_id]++;
  if (indices_C[set_id] == C[set_id].size()) {
    indices_C[set_id] = 0U;
    updateCandidateIndices(C, indices_C, (set_id + 1U) % C.size());
  }
}

void writeRawTimes(std::ostream& os, const RawTimes& raw_times) {
  const unsigned shapeID = raw_times.shapeID;
  const unsigned M = raw_times.M;
  const unsigned N = raw_times.N;
  const unsigned reps = raw_times.reps;
  const unsigned size_instance = raw_times.size_instance;

  os << shapeID << '\n';
  os << M << ' ' << N << ' ' << reps << ' ' << size_instance << '\n';
  os << raw_times.Q << '\n';

  for (unsigned i = 0U; i < M; i++) {
    for (unsigned j = 0U; j < N; j++) {
      for (unsigned k = 0U; k < reps; k++) {
        os << raw_times.times[i * N * reps + j * reps + k] << ' ';
        if ((j == N - 1U) and (k == reps - 1)) os << '\n';
      }  // loop on reps
    }  // loop on instances
  }  // loop on algorithms
}

void writeRawTimes(const std::string& fname, const RawTimes& raw_times) {
  std::ofstream ofile;
  ofile.open(fname);
  if (ofile.fail()) {
    std::cerr << "Error: Cannot open file " << fname << '\n';
    exit(-1);
  }
  writeRawTimes(ofile, raw_times);
  ofile.close();
}

RawTimes readRawTimes(std::ifstream& ifile) {
  RawTimes raw_times;
  unsigned M, N, reps, size_instance;

  ifile >> raw_times.shapeID;
  ifile >> M;
  ifile >> N;
  ifile >> reps;
  ifile >> size_instance;

  raw_times.M = M;
  raw_times.N = N;
  raw_times.reps = reps;
  raw_times.size_instance = size_instance;

  raw_times.Q.reserve(N);

  for (unsigned i = 0U; i < N; i++) {
    Instance instance(size_instance);  // new instance with size_instance elms.
    for (unsigned j = 0; j < size_instance; j++) {
      ifile >> instance[j];
    }
    raw_times.Q.emplace_back(std::move(instance));
  }

  vector<double> times(M * N * reps);  // to be moved
  for (unsigned i = 0U; i < M; i++) {
    for (unsigned j = 0U; j < N; j++) {
      for (unsigned k = 0U; k < reps; k++) {
        ifile >> times[i * N * reps + j * reps + k];
      }  // loop on reps
    }  // loop on instances
  }  // loop on algorithms
  raw_times.times = std::move(times);
  return raw_times;
}

RawTimes readRawTimes(const std::string& fname) {
  std::ifstream ifile;
  ifile.open(fname);
  if (ifile.fail()) {
    std::cerr << "Error: cannot open " << fname << '\n';
    exit(-1);
  }
  auto raw_times = readRawTimes(ifile);
  ifile.close();
  return raw_times;
}

double getMedian(const unsigned n, std::vector<double>::iterator it_v) {
  double ret_val;
  if (n % 2 == 0) {
    std::nth_element(it_v, it_v + (n / 2 - 1), it_v + n);
    std::nth_element(it_v, it_v + n / 2, it_v + n);
    ret_val = (it_v[n / 2 - 1] + it_v[n / 2]) / 2.0;
  } else {
    std::nth_element(it_v, it_v + n / 2, it_v + n);
    ret_val = it_v[n / 2];
  }
  return ret_val;
}

vector<double> compressTimes(RawTimes& raw_times) {
  const unsigned M = raw_times.M;        // number of algorithms
  const unsigned N = raw_times.N;        // number of instances
  const unsigned reps = raw_times.reps;  // reps per time measurement
  vector<double> res_times(M * N);       // resulting times

  vector<double>::iterator it_rt = raw_times.times.begin();
  for (unsigned i = 0U; i < M; i++) {
    for (unsigned j = 0U; j < N; j++) {
      res_times[i * N + j] = getMedian(reps, it_rt + i * reps * N + j * reps);
    }  // loop on instances
  }  // loop on algorithms
  return res_times;
}

void writeCostFile(std::ostream& os, const unsigned M, const unsigned N,
                   const vector<Instance>& Q, const vector<double>& cost_matrix,
                   const std::string& expr_file) {
  os << expr_file << '\n';
  os << M << ' ' << N << ' ' << Q[0].size() << '\n';
  os << Q << '\n';

  for (unsigned i = 0U; i < M; i++) {
    for (unsigned j = 0U; j < N; j++) {
      os << cost_matrix[i * N + j] << ' ';
      if (j == N - 1U) os << '\n';
    }  // loop on instances
  }  // loop on algorithms
}

void writeCostFile(const std::string& fname, const unsigned M, const unsigned N,
                   const vector<Instance>& Q, const vector<double>& cost_matrix,
                   const std::string& expr_file) {
  std::ofstream ofile;
  ofile.open(fname);
  if (ofile.fail()) {
    std::cerr << "Error: Cannot open file " << fname << '\n';
    exit(-1);
  }
  writeCostFile(ofile, M, N, Q, cost_matrix, expr_file);
  ofile.close();
}

vector<double> readCostMatrix(std::istream& ifile, vector<Instance>& Q,
                              std::string& expr_file) {
  unsigned M, N, size_instance;
  ifile >> expr_file;
  ifile >> M;
  ifile >> N;
  ifile >> size_instance;

  Q.clear();  // empty the set of instances.
  Q.reserve(N);

  for (unsigned i = 0U; i < N; i++) {
    Instance instance(size_instance);  // new instance with size_instance elms.
    for (unsigned j = 0; j < size_instance; j++) {
      ifile >> instance[j];
    }
    Q.emplace_back(std::move(instance));
  }

  vector<double> cost_matrix(M * N);  // to return to caller
  for (unsigned i = 0U; i < M; i++) {
    for (unsigned j = 0U; j < N; j++) {
      ifile >> cost_matrix[i * N + j];
    }  // loop on instances
  }  // loop on algorithms
  return cost_matrix;
}

vector<double> readCostMatrix(const std::string& fname, vector<Instance>& Q,
                              std::string& expr_file) {
  std::ifstream ifile;
  ifile.open(fname);
  if (ifile.fail()) {
    std::cerr << "Error: Cannot open file " << fname << '\n';
    exit(-1);
  }
  auto cost_matrix = readCostMatrix(ifile, Q, expr_file);
  ifile.close();
  return cost_matrix;
}

void InfoGreedy::recordStep(const unsigned alg_id, const unsigned size_Z,
                            const std::array<double, 4U>& value_Fs) {
  _chosen_alg.push_back(alg_id);
  _sizes_Z.push_back(size_Z);
  _max_P.push_back(value_Fs[0]);
  _freq_P.push_back(value_Fs[1]);
  _avg_P.push_back(value_Fs[2]);
  _avgNnz_P.push_back(value_Fs[3]);
}

void InfoGreedy::recordSet(const std::set<unsigned>& Z) { _chosen_set = Z; }

void InfoGreedy::recordSet(std::set<unsigned>&& Z) {
  _chosen_set = std::move(Z);
}

std::ostream& operator<<(std::ostream& os, const cg::InfoGreedy& info) {
  os << info._chosen_set << '\n'
     << info._chosen_alg << '\n'
     << info._sizes_Z << '\n'
     << info._max_P << '\n'
     << info._freq_P << '\n'
     << info._avg_P << '\n'
     << info._avgNnz_P << '\n';
  return os;
}

}  // namespace cg