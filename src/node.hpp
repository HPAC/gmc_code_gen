/**
 * @file node.hpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief Header of `Node`. The class defined here is to be used by `Algorithm`.
 * @version 0.1
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef NODE_H
#define NODE_H

#include <string>

#include "kernel.hpp"
#include "matcher.hpp"
#include "matrix.hpp"
#include "memory"
#include "utils/dMatrix.hpp"

namespace cg {

/**
 * @brief Node in the expression tree of an `Algorithm`.
 *
 * The class represents a `Node` in an `Algorithm`s expression tree. A `Node`
 * contains references to the `Node` above (`top`), and the `left` and `right`
 * operands. The operands are also `Node`. The tree is built bottom to top, with
 * the initial chain being at the bottom and the end result at the top.
 *
 * `left`, `right`, and `top` are index-based references to other `Nodes` in the
 * flattened tree in `Algorithm`.
 *
 * `kernel_ptr` is a pointer to the `kernel` that computes the association of
 * the operands in `left` and `right`, giving the result in `top`.
 *
 * `matrix_ptr` is a unique_ptr used to allocate a matrix when the `Algorithm`
 * is to be executed.
 *
 */
class Node {
 private:
  bool transposed{false};  // keep track of original transposition
  int8_t left{-1}, right{-1}, top{-1};
  Kernel* kernel_ptr{nullptr};
  std::unique_ptr<dMatrix> matrix_ptr;

 public:
  Matrix matrix{};

  // Default constructor deleted.
  Node() = delete;

  // Constructor for input nodes.
  Node(const Matrix& matrix)
      : transposed{matrix.isTransposed()}, matrix{matrix} {};

  // Constructor for intermediate nodes.
  Node(const int8_t left, const int8_t right, const Matcher* const matcher_ptr,
       const std::string& temp_name, std::vector<Node>& tree);

  Node(const Node& other);

  Node(Node&& other);

  Node& operator=(const Node& rhs);

  Node& operator=(Node&& rhs);

  ~Node() = default;

  inline int8_t getTop() const noexcept { return top; }

  inline bool getTransposed() const noexcept { return transposed; }

  inline void setTop(const int8_t& top) noexcept { this->top = top; }

  /**
   * @brief Generates code for the association of the node.
   *
   * @param tree          The `Algorithm`s flattened tree.
   * @return std::string  Generated code.
   */
  std::string generateCode(std::vector<Node>& tree) const;

  /**
   * @brief Generates code to explicitly transpose the final result if the
   * resulting operand is transposed.
   *
   * @return std::string
   */
  std::string generateCleanup() const;

  /**
   * @brief Generates code that computes the cost in FLOPs of the association.
   *
   * @param tree
   * @return std::string
   */
  std::string generateCost(std::vector<Node>& tree) const;

  /**
   * @brief Computes the cost in FLOPs of the association.
   *
   * @param tree
   * @return double
   */
  double computeFLOPs(std::vector<Node>& tree);

  /**
   * @brief Estimates execution time of the association using models.
   *
   * @param tree
   * @return double
   */
  double predictTime(std::vector<Node>& tree);

  /**
   * @brief Infers the sizes of the result (`matrix`) based on the sizes of
   * `left` and `right`.
   *
   * @param tree
   */
  void propagateSizes(std::vector<Node>& tree);

  /**
   * @brief Makes matrix_ptr point to a copy of the passed dMatrix object.
   *
   * @param mat
   */
  void setLiveMatrix(const dMatrix& mat);

  /**
   * @brief Deletes matrix_ptr.
   *
   */
  void deleteMatrix();

  /**
   * @brief Returns a rvalue-reference to the unique_ptr to the live matrix.
   *
   * @return std::unique_ptr<dMatrix>&&
   */
  std::unique_ptr<dMatrix>&& getLiveMatrixPtr();

  /**
   * @brief Invokes the kernel that computes the association.
   *
   * @param tree
   */
  void execute(std::vector<Node>& tree);

 private:
  /**
   * @brief Rewrites the association if convenient or needed to guarantee there
   * is a kernel that can compute it.
   *
   * The rewrite is usually from `X := inv(A) * inv(B)` to `inv(X) := B * A`
   *
   * @param tree
   */
  void rewrite(std::vector<Node>& tree);

  /**
   * @brief Performs the actual rewrite on the operands and the result.
   *
   * @param tree
   */
  void invert(std::vector<Node>& tree);

  /**
   * @brief Decides whether rewriting the association is beneficial, based on
   * heuristics.
   *
   * @param A       left operand.
   * @param B       right operand.
   * @return true   if a rewrite is to be performed.
   * @return false  if no rewrite is to be performed.
   */
  bool chooseInvert(const Matrix& A, const Matrix& B);

  /**
   * @brief Analogous of `invert()` for transposition.
   *
   * @param tree
   */
  void transpose(std::vector<Node>& tree);

  /**
   * @brief Assigns the (symbolic) dimensions of the result from the operands
   * (`left` and `right`).
   *
   * @param tree
   */
  void propagateDimensions(std::vector<Node>& tree);

  /**
   * @brief Infers the features of the result based on those of the operands.
   *
   * @param tree
   */
  void propagateFeatures(std::vector<Node>& tree);

  /**
   * @brief One of the two functions used by `propagateFeatures`.
   *
   * @param tree
   * @return Structure
   */
  Structure deduceStructure(std::vector<Node>& tree);

  /**
   * @brief The other function used by `propagateFeatures`.
   *
   * @param tree
   * @return Property
   */
  Property deduceProperty(std::vector<Node>& tree);
};

}  // namespace cg

#endif