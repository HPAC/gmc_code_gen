#ifndef ANALYZER_H
#define ANALYZER_H

#include <array>
#include <functional>
#include <random>
#include <set>
#include <vector>

#include "algorithm.hpp"
#include "definitions.hpp"

namespace cg {

struct InfoGreedy {
  std::string _obj_function{};          // name of the function to minimise.
  std::set<unsigned> _chosen_set{};     // reports the final chosen set.
  std::vector<unsigned> _chosen_alg{};  // tracks the chosen alg in each step.
  std::vector<unsigned> _sizes_Z{};     // tracks the size of the set.
  std::vector<double> _max_P{};         // tracks the max penalty in each step.
  std::vector<double> _freq_P{};    // tracks the freq of penalty in each step.
  std::vector<double> _avg_P{};     // tracks the avg penalty in each step.
  std::vector<double> _avgNnz_P{};  // tracks the avg-nnz penalty in each step.

  InfoGreedy() = default;

  // Ctors.
  InfoGreedy(const InfoGreedy& other) = default;

  InfoGreedy(InfoGreedy&& other)
      : _obj_function(std::move(other._obj_function)),
        _chosen_set(std::move(other._chosen_set)),
        _chosen_alg(std::move(other._chosen_alg)),
        _sizes_Z(std::move(other._sizes_Z)),
        _max_P(std::move(other._max_P)),
        _freq_P(std::move(other._freq_P)),
        _avg_P(std::move(other._avg_P)),
        _avgNnz_P(std::move(other._avgNnz_P)) {}

  // Assignment operators.
  InfoGreedy& operator=(const InfoGreedy& other) = default;

  InfoGreedy& operator=(InfoGreedy&& other) {
    _obj_function = std::move(other._obj_function);
    _chosen_set = std::move(other._chosen_set);
    _chosen_alg = std::move(other._chosen_alg);
    _sizes_Z = std::move(other._sizes_Z);
    _max_P = std::move(other._max_P);
    _freq_P = std::move(other._freq_P);
    _avg_P = std::move(other._avg_P);
    _avgNnz_P = std::move(other._avgNnz_P);
    return *this;
  }

  ~InfoGreedy() = default;

  void recordStep(const unsigned alg_id, const unsigned size_Z,
                  const std::array<double, 4U>& value_Fs);

  void recordSet(const std::set<unsigned>& Z);

  void recordSet(std::set<unsigned>&& Z);
};  // struct InfoGreedy

std::ostream& operator<<(std::ostream& os, const cg::InfoGreedy& info);

/**
 * @brief Computes FLOPs for all input algorithms for one instance.
 *
 * @param algorithms      vector<Algorithm>.
 * @param q               Instance.
 * @return vector<double> FLOPs of algorithms on instance.
 */
std::vector<double> FLOPsOnInstance(std::vector<Algorithm>& algorithms,
                                    const Instance& q);

/**
 * @brief Computes the FLOPs of the algorithms in A for all instances in Q.
 * The result is in a std::vector<double> with M * N elements, where values of
 * the same algorithm are consecutive.
 *
 * @param A             Vector of algorithms. Size: M.
 * @param Q             Vector of instances.  Size: N.
 * @return std::vector<double>
 */
std::vector<double> FLOPsOnInstances(std::vector<Algorithm>& A,
                                     const std::vector<Instance>& Q);

/**
 * @brief Predicts time for the algorithms in A for all instances in Q.
 *
 * The result is in a single std::vector<double> with M * N elements, where
 * values of the same algorithm are consecutive. Predictions use performance
 * models.
 *
 * @param A       Vector of algorithms. Size: M.
 * @param Q       Vector of instances.  Size: N.
 * @return std::vector<double>
 */
std::vector<double> predTimeOnInstances(std::vector<Algorithm>& A,
                                        const std::vector<Instance>& Q);

/**
 * @brief Times an algorithm on a particular instance n_rep times.
 *
 * @param alg               Algorithm to execute and time.
 * @param live_chain        Real matrix chain to feed the Algorithm.
 * @param n_rep             Number of times to repeat the execution.
 * @return std::vector<double>  vector with the execution times.
 */
std::vector<double> timeAlg(Algorithm& alg,
                            const std::vector<dMatrix>& live_chain,
                            const unsigned n_rep);

/**
 * @brief Measures avg time taken to execute each algorithm on every instance.
 * The result is in a single std::vector<double> with M * N elements, where
 * values for the same algorithm are consecutive.
 *
 * @param chain        Symbolic chain.
 * @param A            Vector with the algorithms. A.size() = M.
 * @param Q            Vector with the instances. Q.size() = N.
 * @param n_rep        Times to repeat each execution.
 * @return std::vector<double>
 */
std::vector<double> avgTimings(const MatrixChain& chain,
                               std::vector<Algorithm>& A,
                               const std::vector<Instance>& Q, unsigned n_rep);

/**
 * @brief Makes \p reps measurements of the time taken to execute each algorithm
 * on each instance. The result is a single std::vector<double> with M * N *
 * reps elements, where values for the same instance and algorithm are
 * consecutive in memory and values for the same algorithm are consecutive.
 *
 * Caution: The returned vector can be large in memory.
 *
 * @param chain         Symbolic chain.
 * @param A             Vector with the algorithms. A.size() = M.
 * @param Q             Vector with the instances. Q.size() = N.
 * @param reps          Times to repeat each execution.
 * @return std::vector<double>
 */
std::vector<double> timeAlgsOnInstances(const MatrixChain& chain,
                                        std::vector<Algorithm>& A,
                                        const std::vector<Instance>& Q,
                                        unsigned reps);

/**
 * @brief Returns the essential permutation fanning out from index min_dim.
 *
 * @param N             length of the chain.
 * @param min_dim       id of the dimension to fan out from.
 * @return Permutation  Permutation - order of computation.
 */
Permutation getEssentialPerm(const unsigned N, const unsigned min_dim);

/**
 * @brief Returns all essential permutations for a chain of length N.
 *
 * @param N                     length of the chain.
 * @return vector<Permutation>  Permutaions - orders of computation.
 */
std::vector<Permutation> getEssentialPerms(const unsigned N);

/**
 * @brief Returns the entire set of fanning-out algorithms.
 *
 * @param chain         Symbolic matrix chain
 * @param A             Vector with all algorithms
 * @return std::set<unsigned>
 */
std::set<unsigned> fanningSet(const MatrixChain& chain,
                              const std::vector<Algorithm>& A);

/**
 * @brief Returns a set of fanning-out algorithms with reduced cardinality.
 *
 * The resulting set includes the algorithm that fans out from the leftmost
 * dimension (i.e., the dimension with the smallest index) when dimensions are
 * coupled.
 *
 * @param chain         Symbolic matrix chain
 * @param A             Vector with all algorithms
 * @return std::set<unsigned>
 */
std::set<unsigned> getEssentialSetMin(const MatrixChain& chain,
                                      const std::vector<Algorithm>& A);

/**
 * @brief Counts the number of free dimensions and sets the mapping from chain
 * dimensions to free dimensions.
 *
 * @param chain      matrix chain of length N.
 * @param indices    vector of size (N + 1) that maps chain dimensions to free
 * dimensions.
 * @return unsigned  count of free dimensions in the chain.
 */
unsigned countFreeDimensions(const MatrixChain& chain,
                             std::vector<unsigned>& indices);

/**
 * @brief Returns the non-coupled dimensions in the chain. Places the coupled
 * dimensions in different vectors in C.
 *
 * @param chain  The matrix chain.
 * @param C      Holds the coupled dimensions. Dimensions that are coupled are
 * placed in the same vector.
 * @return std::vector<unsigned> non-coupled dimensions --> their respective
 * fanning out algorithms must be in every set of algorithms if the set is to
 * have a bounded penalty.
 */
std::vector<unsigned> getFreeDimsIdx(const MatrixChain& chain,
                                     std::vector<std::vector<unsigned>>& C);

/**
 * @brief Inserts in Z the identifiers of the algorithms that fan out from the
 * dimensions whose indices are in free_dims_idx.
 *
 * @param A              Vector with all the algorithms.
 * @param n              Length of the chain.
 * @param Z              Set where the algorithm identifiers are inserted.
 * @param free_dims_idx  Vector with dimension indices.
 */
void insertFanDimIdx(const std::vector<Algorithm>& A, const unsigned n,
                     std::set<unsigned>& Z,
                     const std::vector<unsigned>& free_dims_idx);

/**
 * @brief Creates a vector of vectors where the elements of each underlying
 * vector are identifiers of algorithms that fan out from coupled dimensions.
 *
 * @param A                 Vector with all the algorithms.
 * @param n                 Length of the chain.
 * @param coupled_dims_id   Vector of vectors where the elements of each
 * underlying vector are indices of coupled dimensions.
 * @return std::vector<std::vector<unsigned>>
 */
std::vector<std::vector<unsigned>> dimId2AlgId(
    const std::vector<Algorithm>& A, const unsigned n,
    const std::vector<std::vector<unsigned>>& coupled_dims_id);

/**
 * @brief Generates an instance by indirection from indices on
 * base_instance. Expands the base_instance to form an instance of size N+1.
 *
 * @param base_instance     Values of free dimensions in the chain.
 * @param indices           Maps chain dimensions to free dimensions.
 * @return Instance         generated instance.
 */
Instance getInstance(const Instance& base_instance,
                     std::vector<unsigned> indices);

/**
 * @brief Returns the ID of the algorithm with permutation perm.
 *
 * @param algs       vector<Algorithm> the algorithms where to look for perm.
 * @param perm       Permutation the permutation of interest.
 * @return unsigned  the ID of the algorithm with permutation perm.
 */
unsigned getID(const std::vector<Algorithm>& algs, const Permutation& perm);

/**
 * @brief Returns the IDs of the algorithms with permutations in perms.
 *
 * @param algs              the algorithms where to look for perm in perms.
 * @param perms             the permutations of interest.
 * @return vector<unsigned> IDs of the algorithms with permutation in perms.
 */
std::vector<unsigned> getIDs(const std::vector<Algorithm>& algs,
                             const std::vector<Permutation>& perms);

/**
 * @brief Greedy function to solve the set selection with cardinality
 * constraint problem.
 *
 * The function takes in a set of algorithms(A), a set of instances(S), a
 * budget(K), an objective function(F), an initial chosen set of algorithms(Z0),
 * and a cost matrix. The function includes algorithms in Z greedily. This is,
 * in each iteration the function includes the algorithm that minimises F the
 * most. The function only considers the inclusion of one algorithm at a time.
 *
 * F is a function from a vector (R^N) to R -- builds upon the penalty.
 * Z0 may be empty, though it is commonly initialised with the fanning out set
 * of algorithms.
 * The cost matrix holds the pre-computed cost of all algorithms
 * in A on the instances in S.
 *
 * @param A             vector<Algorithm>.
 * @param S             vector<Instance>.
 * @param K             budget - maximum number of algorithms to select.
 * @param F             std::function<double(const std::vector<double>&)>
 * @param Z0            set of alg identifiers.
 * @param cost_matrix   cost of all algorithms in A on the instances in S.
 * @return std::set<unsigned> augmented set of algorithms
 */
std::set<unsigned> SEGreedy(
    const std::vector<Algorithm>& A, const std::vector<Instance>& S,
    const unsigned K,
    const std::function<double(const std::vector<double>&)>& F,
    const std::set<unsigned>& Z0, const std::vector<double>& cost_matrix);

/**
 * @brief Greedy function to expand a set of algorithms.
 *
 * @param M                 Number of algorithms.
 * @param N_t               Number of instances in the training set.
 * @param K                 Budget - maximum number of algorithms to select.
 * @param F                 Objective function
 * @param Z0                Initial set of algIDs to expand.
 * @param penalty_matrix_t  cost of al algorithms on all instances.
 * @return std::set<unsigned> expanded set of algorithms.
 */
std::set<unsigned> SEGreedyWPenalty(
    const unsigned M, const unsigned N_t, const unsigned K,
    const std::function<double(const std::vector<double>&)>& F,
    const std::set<unsigned>& Z0, const std::vector<double>& penalty_matrix_t);

/**
 * @brief Greedy function to solve the set selection with cardinality
 * constraint problem.
 *
 * This function makes decisions based on the train_matrix and computes
 * penalties on the val_matrix (which must contain the execution times).
 *
 * @param M             Number of algorithms.
 * @param N_train       Number of instances in the matrix used for
 * selection.
 * @param N_val         Number of instances in the matrix used for
 * validation.
 * @param K             Maximum number of algorithms to have in Z.
 * @param F             Function to minimise while greedily choosing algs.
 * @param Z             set of alg identifiers. Begin: initial set. End:
 * chosen set.
 * @param train_matrix  cost of all algorithms on the train instances.
 * @param val_matrix    execution times of all algorithms on the validation
 * instances.
 * @return InfoGreedy   A struct that keeps track of how metrics and the set
 * evolve as the greedy selection is executed.
 */
InfoGreedy SEGreedyStepVal(
    const unsigned M, const unsigned N_t, const unsigned N_v, const unsigned K,
    const std::function<double(const std::vector<double>&)>& F,
    std::set<unsigned>& Z, const std::vector<double>& train_matrix,
    const std::vector<double>& val_matrix);

/**
 * @brief Greedy function that grows the input set of algorithms making a
 * choice in every step that minimises the objective function (F) the most.
 *
 * The penalty matrix is in row-major format. It holds the penalty of every
 * algorithm for every instance. One row corresponds to an algorithm, while one
 * column corresponds to an instance.
 *
 * @param M   Number of algorithms.
 * @param N   Number of instances.
 * @param K   Budget. Maximum number of algorithms in the chosen set.
 * @param F   Objective function to minimise.
 * @param Z   Input/output set of algorithms.
 * @param penalty_matrix  matrix in row-major format that holds the penalty of
 * every algorithm for every instance. One row corresponds to an algorithm.
 * @return InfoGreedy
 */
InfoGreedy SEGreedyStep(
    const unsigned M, const unsigned N, const unsigned K,
    const std::function<double(const std::vector<double>&)>& F,
    std::set<unsigned>& Z, const std::vector<double>& penalty_matrix);

/**
 * @brief Measure of how far two algorithms' costs are.
 *
 * @param min_A     The minimum in the overall set of algorithms.
 * @param min_Z     The minimum in the chosen set of algorithms.
 * @return double   min_Z / min_A - 1.0
 */
double penalty(const double min_A, const double min_Z);

/**
 * @brief Finds the minimum cost on each instance.
 *
 * @param M             number of algorithms.
 * @param N             number of instances.
 * @param cost_matrix   cost of every pair (alg,instance).
 * @param min_A         vector with the minimum cost for each instance in S.
 */
void getMinA(const unsigned M, const unsigned N,
             const std::vector<double>& cost_matrix,
             std::vector<double>& min_A);

/**
 * @brief Finds the minimum cost amongst the algorithms in a set on each
 * instance.
 *
 * @param M             number of algorithms.
 * @param N             number of instances.
 * @param cost_matrix   vector with cost of every pair (alg,instance).
 * @param Z             set of indices of algorithms.
 * @param min_Z         vector with the minimum cost in Z for each instance.
 */
void getMinSet(const unsigned M, const unsigned N,
               const std::vector<double>& cost_matrix,
               const std::set<unsigned>& Z, std::vector<double>& min_Z);

/**
 * @brief Element-wise min of the ranges determined by it_A and it_B and places
 * the result in the range determined by it_C.
 *
 * it_C[i] = min(it_A[i], it_B[i]).
 *
 * @param N       Number of elements.
 * @param it_A    First range.
 * @param it_B    Second range.
 * @param it_C    Resulting range.
 */
void minRanges(const unsigned N, std::vector<double>::const_iterator& it_A,
               std::vector<double>::const_iterator& it_B,
               std::vector<double>::iterator& it_C);

/**
 * @brief Computes the penalty on each instance for the set Z.
 *
 * @param min_A         vector with the minimum cost for each instance in S.
 * @param min_Z         vector with the minimum cost in Z for each instance.
 * @param penalty_Z     vector with the penalty of Z on each instance.
 */
void getPenaltyZ(const std::vector<double>& min_A,
                 const std::vector<double>& min_Z,
                 std::vector<double>& penalty_Z);

/**
 * @brief Fills penalty_matrix with the penalty for every algorithm and every
 * instance. The input cost_matrix holds the cost for every algorithm and every
 * instance.
 *
 * Both matrices are in row-major format where each row corresponds to a
 * different algorithm. In the same column one will find the cost for the
 * different algorithms for one single instance.
 *
 * @param M                 Number of algorithms.
 * @param N                 Number of instances.
 * @param cost_matrix       Vector with the cost of every pair (alg,instance).
 * Costs of the same algorithm are consecutive.
 * @param penalty_matrix    Vector with the penalty of every algorithm on every
 * instance. Values of the same algorithm are consecutive.
 */
void getPenaltyMatrix(const unsigned M, const unsigned N,
                      const std::vector<double>& cost_matrix,
                      std::vector<double>& penalty_matrix);

/**
 * @brief Converts a cost matrix into a penalty matrix.
 *
 * The cost matrix is overwritten with the penalty for every algorithm and every
 * instance. The cost matrix has sizes M x N. The matrix is in row-major format
 * where each row corresponds to a different algorithm.
 *
 * @param M             Number of algorithms.
 * @param N             Number of instances.
 * @param cost_matrix   Cost and penalty, at input and output, respectively, for
 * every algorithm and every instance.
 */
void cost2penalty(const unsigned M, const unsigned N,
                  std::vector<double>& cost_matrix);
/**
 * @brief Converts a cost matrix into a penalty with ratios computed over the
 * optimal on a per-instance basis.
 *
 * This function is essentially the same as cost2penalty. The difference is in
 * the computation of penalties/ratios.
 *
 * @param M             Number of algorithms.
 * @param N             Number of instances.
 * @param cost_matrix   Cost and ratios, at input and output, respectively, for
 * every algorithm and every instance.
 */
void cost2RatioOverOptimal(const unsigned M, const unsigned N,
                           std::vector<double>& cost_matrix);

/**
 * @brief Computes the maximum penalty in the vector penalty.
 *
 * @param penalty   vector with penalties.
 * @return double   Maximum penalty.
 */
double maxPenalty(const std::vector<double>& penalty);

/**
 * @brief Computes the number instances with non-zero penalty.
 *
 * @param penalty   vector with penalties.
 * @return double   Number of instances with non-zero penalty.
 */
double freqPenalty(const std::vector<double>& penalty);

/**
 * @brief Computes the average penalty.
 *
 * @param penalty   vector with penalties.
 * @return double   Average penalty in buffer.
 */
double avgPenalty(const std::vector<double>& penalty);

/**
 * @brief Computes the average non-zero penalty.
 *
 * @param penalty   vector with penalties.
 * @return double   Average non-zero penalty in buffer.
 */
double avgNnzPenalty(const std::vector<double>& penalty);

/**
 * @brief Computes the average squared penalty in the buffer. This objective
 * function is (supposed to be) used as a trade-off between minimizing the
 * maximum and the average penalty.
 *
 * @param penalty   Pointer to allocated buffer.
 * @return double   Average squared penalty in buffer.
 */
double sqPenalty(const std::vector<double>& penalty);

/**
 * @brief Prints metrics - used in SetExpansion for user communication.
 *
 * @param penalty   vector with penalties.
 */
void printMetrics(const std::vector<double>& penalty);

/**
 * @brief Computes the value of 4 functions (max, freq, avg, nnz-avg) for a
 * buffer containing the penalty in each instance.
 *
 * @param N             Number of instances.
 * @param penalty_Z     Buffer with the penalty in every instance.
 * @return std::array<double, 4U> [0]:max; [1]:freq; [2]:avg; [3]:nnz-avg.
 */
std::array<double, 4U> computeAllF(const std::vector<double>& penalty);

/**
 * @brief Computes the penalty of Z and invokes printMetrics.
 *
 * @param M             Number of algorithms.
 * @param N             Number of instances.
 * @param Z             Set of algorithms.
 * @param cost_matrix   vector<double> with the cost every (alg,instance).
 */
void printSetPerformance(const unsigned M, const unsigned N,
                         const std::set<unsigned>& Z,
                         const std::vector<double>& cost_matrix);

/**
 * @brief Greedily selects an algorithm to insert in the set Z.
 *
 * This function is invoked by SEGreedy, SEGreedyStep, and SEGreedyStepVal.
 *
 * @param M               Algorithms in A.
 * @param N               Instances in Q.
 * @param penalty_matrix  Penalty of every (alg,instance).
 * @param penalty_Z       Penalty of the current Z in every instance.
 * @param penalty_Z_U_min Penalty of the currently best found Z \cup X.
 * @param penalty_Z_U_can Vector to be constantly overwritten. Holds the penalty
 * of Z \cup X, where X \notin Z.
 * @param Z               Set of already chosen algorithms.
 * @param F               Objective function.
 * @param val_F           Current value of F for Z.
 * @param min_F           Best value of F for Z \cup X, where X \notin Z.
 * @return int            Index of the algorithm to include in Z.
 */
int selectCandidate(const unsigned M, const unsigned N,
                    const std::vector<double>& penalty_matrix,
                    const std::vector<double>& penalty_Z,
                    std::vector<double>& penalty_Z_U_min,
                    std::vector<double>& penalty_Z_U_can,
                    const std::set<unsigned>& Z,
                    const std::function<double(const std::vector<double>&)>& F,
                    double val_F, double& min_F);

/**
 * @brief Computes the value of F if the algorithm with id were included in the
 * current set.
 *
 * @param M             number of algorithms in A.
 * @param N             number of instances in S.
 * @param cost_matrix   cost for all algorithms in A on every instance in S.
 * @param min_A         array with the minimal cost for each instance.
 * @param min_Z         array with the minimum cost across Z for each instance.
 * @param penalty_Z     array with the penalty of Z for each instance.
 * @param Z             set of already chosen algorithms.
 * @param id            id of the algorithm to test.
 * @param F             objective function.
 * @return double       value of F(Z \cup id).
 */
double evalUnion(const unsigned M, const unsigned N, const unsigned id,
                 const std::vector<double>& cost_matrix,
                 const std::vector<double>& min_A, std::vector<double>& min_Z,
                 std::vector<double>& penalty_Z, std::set<unsigned>& Z,
                 const std::function<double(const std::vector<double>&)>& F);

/**
 * @brief Brute-force algorithm that finds the set of fanning algorithms
 * that minimizes F the most.
 *
 * The function takes in a penalty_matrix of sizes M x N in row-major format,
 * where each row corresponds to an algorithm and each column corresponds to an
 * instance. Z, the input set of algorithms, contains algorithms that fan out
 * from dimensions that are not coupled to any other. C, the vector of sets,
 * contains a number of sets of fanning out algorithms. A set in C contains
 * algorithms that fan out from coupled dimensions. The resulting set will
 * contain the elements in Z and one element from each set in C. Thus, the
 * resulting set will have a bound on the penalty. 'metrics' is an array of 4
 * doubles that holds the value of different objective functions when the
 * function returns.
 *
 * @param M              Number of algorithms.
 * @param N              Number of instances.
 * @param penalty_matrix Holds the cost of all algorithms on all instances.
 * @param Z              Set of algorithms that fan out from de-coupled
 * dimensions.
 * @param C              Vector of sets of algorithms. Each set contains
 * algorithms that fan out from coupled dimensions.
 * @param F              Objective function to minimize.
 * @param metrics        Values of several functions for the best set.
 * @return std::set<unsigned> resulting set.
 */
std::set<unsigned> getOptimalFanningSet(
    const unsigned M, const unsigned N,
    const std::vector<double>& penalty_matrix, const std::set<unsigned>& Z,
    const std::vector<std::vector<unsigned>>& C,
    const std::function<double(const std::vector<double>&)>& F,
    std::array<double, 4U>& metrics);

/**
 * @brief Brute-force algorithm that finds the set of fanning algorithms that
 * minimizes F the most on a training set of instances.
 *
 * Takes in a penalty_matrix_t of sizes M x N_t in row-major format, where each
 * row corresponds to an algorithm and each column corresponds to an instance. Z
 * is the input set of algorithms, should contain algorithms that fan out from
 * dimensions that are not coupled to any other. C is a vector of vectors,
 * contains a number of vectors with IDs of fanning-out algorithms. Each vector
 * in C contains algorithms that fan out from coupled dimensions. The resulting
 * contains the elements in Z and one element from each vector in C. Thus, the
 * resulting set will have a bound on the penalty.
 *
 * @param M                 Number of algorithms.
 * @param N_t               Number of instances in the training set.
 * @param penalty_matrix_t  Holds the cost of all algorithms on all instances.
 * @param Z                 Set with algs that fan out from non-coupled
 * dimensions.
 * @param C                 Vector of vectors with algorithms IDs. Each vector
 * contains algorithms that fan out from coupled dimensions.
 * @param F                 Objective function to minimize.
 * @return std::set<unsigned> resulting set.
 */
std::set<unsigned> findOptimalFanningSet(
    const unsigned M, const unsigned N_t,
    const std::vector<double>& penalty_matrix_t, const std::set<unsigned>& Z,
    const std::vector<std::vector<unsigned>>& C,
    const std::function<double(const std::vector<double>&)>& F);

/**
 * @brief Measures the performance of the set of algorithms Z on a validation
 * set of instances.
 *
 * For each instance, finds the penalty of the set, creates an array with all
 * penalties and computes the value of F on it.
 *
 * @param M                 Number of algorithms.
 * @param N_v               Number of instances in the validation set.
 * @param penalty_matrix_v  Holds the cost of all algorithms on all instances.
 * @param Z                 Set of algorithms.
 * @param F                 Objective function to measure.
 * @return double           F(Z)
 */
double evaluateSet(const unsigned M, const unsigned N_v,
                   const std::vector<double>& penalty_matrix_v,
                   const std::set<unsigned>& Z,
                   const std::function<double(const std::vector<double>&)>& F);

/**
 * @brief Measures the performance of Armadillo on a validation set of
 * instances.
 *
 * The timings for armadillo aer in the vector times_arma, whereas the timings
 * for all algorithms are in times_vars. times_arma is of size 1 * N_v, whereas
 * times_vars is of size M * N_v.
 *
 * @param M                 Number of algorithms.
 * @param N_v               Number of instances in the validation set.
 * @param times_arma        Holds the timings of all instances using Armadillo.
 * @param times_vars        Holds timings of all algorithms on all instances.
 * @param F                 Objective function to measure.
 * @return double           F(Arma).
 */
double evaluateArma(const unsigned M, const unsigned N_v,
                    const std::vector<double>& times_arma,
                    const std::vector<double>& times_vars,
                    const std::function<double(const std::vector<double>&)>& F);

/**
 * @brief Computes the metrics of the optimal fanning-out set with minimal
 * cardinality to be essential.
 *
 * @param M                 Number of algorithms.
 * @param N                 Number of instances.
 * @param penalty_matrix    Penalty for every algorithm and every instance.
 * @param Z                 Set of fanning-out algs from free dimensions.
 * @param C                 Vector of sets of algorithms. Each set contains
 * ids of algorithms that fan out from coupled dimensions.
 * @param F                 Objective function to minimise.
 * @return std::array<double, 4U> the value of all considered functions
 * (max, freq, avg, nnz-avg) for the optimal set.
 */
std::array<double, 4U> getOptimalFanningMetrics(
    const unsigned M, const unsigned N,
    const std::vector<double>& penalty_matrix, const std::set<unsigned>& Z,
    const std::vector<std::vector<unsigned>>& C,
    const std::function<double(const std::vector<double>&)>& F);

/**
 * @brief Forms a set taking one element from each vector in C. The taken
 * elements are indicated by indices_C.
 *
 * @param C          vector with vectors of alg ids. Each vector of alg ids
 * contains fanning out algs with coupled dimensions.
 * @param indices_C  indices of the elements in C that are taken to form a
 * set.
 * @return std::set<unsigned>
 */
std::set<unsigned> formCandidateSet(const std::vector<std::vector<unsigned>>& C,
                                    const std::vector<unsigned>& indices_C);

/**
 * @brief Updates the indices of C. This is used to generate all combinations of
 * sets of algorithms across the different sets in C.
 *
 * @param C         vector with vectors of algs ids.
 * @param indices_C indices of the elements in C that are taken to form a set.
 * @param set_id    indicates the set in C whose index is updated.
 */
void updateCandidateIndices(const std::vector<std::vector<unsigned>>& C,
                            std::vector<unsigned>& indices_C, unsigned set_id);

// Contains information on raw timings of algorithms
struct RawTimes {
  unsigned shapeID;
  unsigned size_instance;     // entries in an instance
  unsigned M;                 // number of algorithms
  unsigned N;                 // number of instances
  unsigned reps;              // number of repetitions
  std::vector<Instance> Q;    // Instances
  std::vector<double> times;  // times
};

/**
 * @brief Writes a file with raw timings of algorithms on instances.
 *
 * @param os          stream where results are dumped
 * @param raw_times   struct with all the information
 */
void writeRawTimes(std::ostream& os, const RawTimes& raw_times);

/**
 * @brief Overload that takes a path_to_file instead of a stream
 *
 * @param fname       path to file where results are dumped
 * @param raw_times   struct with all the information
 */
void writeRawTimes(const std::string& fname, const RawTimes& raw_times);

/**
 * @brief Reads a file with raw timings and returns it.
 *
 * @param ifile       input stream at the beginning of the file.
 * @return RawTimes
 */
RawTimes readRawTimes(std::ifstream& ifile);

/**
 * @brief Reads a file with raw timings and returns it.
 *
 * @param fname       file with raw timings.
 * @return RawTimes
 */
RawTimes readRawTimes(const std::string& fname);

/**
 * @brief Computes the median of a number of consecutive values in a vector.
 *
 * @param n       Number of elements to compute the median.
 * @param it_v    Iterator to the first element.
 * @return double
 */
double getMedian(const unsigned n, std::vector<double>::iterator it_v);

/**
 * @brief Computes a representative value of the times for each pair (algorithm,
 * instance).
 *
 * @param raw_times   struct with all the necessary information.
 * @return std::vector<double>
 */
std::vector<double> compressTimes(RawTimes& raw_times);

/**
 * @brief Writes a file with info about costs of algorithms on instances.
 *
 * The format of the file is, per-line, the following:
 *      expr_file
 *      M N size_instance
 *      contents_of_S
 *      A M-by-N matrix where each row corresponds to some algorithm.
 *
 * The resulting file has in each row the costs in all instances in S of
 * some algorithm.
 *
 * @param os          stream where to output results
 * @param M           number of algorithms
 * @param N           number of instances
 * @param Q           vector of instances
 * @param cost_matrix cost of every pair (alg,instance)
 * @param expr_file   file where the expression is defined (following
 * grammar)
 */
void writeCostFile(std::ostream& os, const unsigned M, const unsigned N,
                   const std::vector<Instance>& Q,
                   const std::vector<double>& cost_matrix,
                   const std::string& expr_file);

/**
 * @brief Overload that takes a path_to_file where results are output.
 *
 * @param fname       path to file where results are dumped
 * @param M           number of algorithms
 * @param N           number of instances
 * @param Q           vector of instances
 * @param cost_matrix cost of every pair (alg,instance)
 * @param expr_file   file where the expression is defined (following grammar)
 */
void writeCostFile(const std::string& fname, const unsigned M, const unsigned N,
                   const std::vector<Instance>& Q,
                   const std::vector<double>& cost_matrix,
                   const std::string& expr_file);

/**
 * @brief Reads a file with info of costs of algorithms on instances.
 *
 * The file must have the structured:
 *      expr_file
 *      M N size_instance
 *      instances
 *      matrix with cost of algorithms on instances
 *
 * The matrix has in one row the cost of a single algorithms on all instances.
 *
 * @param ifile       input stream
 * @param Q           vector of instances to be overwritten
 * @param expr_file   (overwritten) name of the file that describes the chain
 * @return std::vector<double> vector with the M-by-N matrix with costs
 */
std::vector<double> readCostMatrix(std::istream& ifile,
                                   std::vector<Instance>& Q,
                                   std::string& expr_file);

/**
 * @brief Overload that takes a path to file from where results are read
 *
 * @param fname       path to file with the results to read
 * @param Q           vector of instances to overwrite
 * @param expr_file   name of the file with the expression, to overwrite
 * @return std::vector<double>
 */
std::vector<double> readCostMatrix(const std::string& fname,
                                   std::vector<Instance>& Q,
                                   std::string& expr_file);

}  // namespace cg

#endif