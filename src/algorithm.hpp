/**
 * @file algorithm.hpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief Implements the class `Algorithm`.
 * @version 0.1
 *
 * An algorithm is identified by its permutation. The permutation determines
 * the order in which operations take place. Each operation is binary. Takes two
 * operands and produces a single result. The `matcher` (of which a reference is
 * taken at Algorithm construction) gives a mapping from pairs of operands to
 * kernels that can compute the subexpression they form.
 *
 * One algorithm is represented as a single tree in flattened form. This is, a
 * tree is an std::vector. The elements therein contain references (per index)
 * to other elements.
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef ALGORITHM_H
#define ALGORITHM_H

#include <string>
#include <vector>

#include "definitions.hpp"
#include "macros.hpp"
#include "matcher.hpp"
#include "matrix.hpp"
#include "node.hpp"
#include "utils/dMatrix.hpp"

namespace cg {

class Algorithm {
 private:
  Permutation permutation;  // algorithm's identifier -- order of computation.
  std::vector<Node> tree;   // representation of the algorithm.

 public:
  Algorithm() = delete;

  Algorithm(const MatrixChain& chain, const Permutation& permutation,
            const Matcher* const matcher);

  // Algorithm(const Algorithm& other) = delete;

  // Algorithm(Algorithm&& other);

  ~Algorithm() = default;

  /**
   * @brief Generates code in the form of a function for the algorithm.
   *
   * @param function_name string - name of the generated function.
   * @param trans_cleanup bool - determines whether an explicit transpose should
   * be performed if the result of the algorithm is the transpose of the desired
   * result.
   * @return string - code implementation of the algorithm.
   */
  std::string generateCode(const std::string& function_name,
                           const bool trans_cleanup);

  /**
   * @brief Generates code in the form of a function that computes the FLOP
   * count of the algorithm.
   *
   * @param function_name string - name of the generated function.
   * @return string - code implementation of the FLOP count.
   */
  std::string generateCostFunction(const std::string& function_name);

  /**
   * @brief Computes the algorithm's FLOP count for the given instance.
   *
   * @param instance vector<unsigned> - sizes of the matrices.
   * @return double - FLOP.
   */
  double computeFLOPs(const Instance& instance);

  /**
   * @brief Predicts the time to execute the algorithm for the given instance.
   *
   * @param instance vector<unsigned> - sizes of the matrices.
   * @return double - predicted time.
   */
  double predictTime(const Instance& instance);

  /**
   * @brief Takes the input dMatrix objects and places a copy into the
   * corresponding nodes.
   *
   * @param chain
   */
  void assignChain(const std::vector<dMatrix>& chain);

  /**
   * @brief Executes the algorithm. Requires the dMatrix chain to be assigned.
   *
   * @return dMatrix
   */
  dMatrix execute();

  /**
   * @brief Deletes all dMatrix objects pointed to by all nodes.
   *
   */
  void clean();

  Permutation getPermutation() const;

  std::string getSignature(const std::string& ret_type,
                           const std::string& function_name);

  bool isResultInverted() const;

 private:
  /**
   * @brief Creates the n input nodes in the tree.
   *
   * @param chain vector<Matrix>.
   */
  void createInputNodes(const MatrixChain& chain);

  /**
   * @brief Builds the algorithm's representation.
   *
   * @param matcher Matcher - used to map pairs of matrices to kernels.
   */
  void buildTree(const Matcher* const matcher);

  /**
   * @brief Assigns sizes to input nodes and propagates sizes to the rest.
   *
   * @param instance
   */
  void assignSizes(const Instance& instance);

  /**
   * @brief Generates the last piece of code for generated functions.
   *
   * @param var string - name of the variable to return; "" if no return.
   * @return string - last piece of code.
   */
  std::string getClosure(const std::string& var = "") const;

  /**
   * @brief Returns the root of a subtree.
   *
   * @param id int8_t - identifier of the node of which to get the root.
   * @return int8_t - identifier of the root.
   */
  int8_t getRoot(const int8_t id) const;
};

}  // namespace cg

#endif