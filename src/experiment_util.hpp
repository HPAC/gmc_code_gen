/**
 * @file experiment_util.hpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief Functions for programs that execute experiments with 10 combinations
 * of features.
 * @version 0.1
 *
 * @copyright Copyright (c) 2024
 *
 */

#include <array>
#include <string>

#include "definitions.hpp"
#include "matrix.hpp"

namespace cg {

/**
 * @brief Produces a matrix with given features based on `digit`.
 *
 * @param digit       Integer value representing one of 10 matrix combinations.
 * @param id_matrix   Integer value with the value of the matrix in the chain.
 * @return cg::Matrix The produced symbolic matrix.
 */
cg::Matrix ID2matrix(const unsigned digit, const unsigned id_matrix);

/**
 * @brief Produces a chain with given features.
 *
 * Produces a chain of length `chain_N` where the operands have the features
 * specified by `id_shape`.
 *
 * @param chain_N           Length of the chain.
 * @param id_shape          Integer representing the features of the operands.
 * @return cg::MatrixChain  Symbolic matrix chain.
 */
cg::MatrixChain ID2chain(const unsigned chain_N, const unsigned id_shape);

/**
 * @brief Returns the range of shapes for each processor.
 *
 * This function is to be used in conjunction with sbatch jobs. An array of jobs
 * is to be created. Each jobID is passed to the calling program as `proc_id`.
 * The `proc_id` and the total number of processors (`proc_num`) are used
 * together to determine the shapes upon which each processor will perform
 * experiments.
 *
 * @param max_value                 Total number of shapes
 * @param proc_id                   Processor identifier (jobID)
 * @param proc_num                  Total number of processors (jobs)
 * @return std::array<unsigned, 2U> Contiguous range of shapes
 */
std::array<unsigned, 2U> getProcRange(const unsigned max_value,
                                      const unsigned proc_id,
                                      const unsigned proc_num);

/**
 * @brief Checks whether a given shape is square.
 *
 * This function is used in the experiments for the paper sent to CGO. A shape
 * is square if all sizes are necessarily equal, due to properties of its
 * operands. In the paper, we impose shapes to never be square, since at least
 * one matrix must be general.
 *
 * @param n         Length of the chain
 * @param id_shape  Integer -- sequence identifier.
 * @return true/false
 */
bool isSquareShape(const unsigned n, const unsigned id_shape);

/**
 * @brief Returns the largest shapeID for a given length of chain (`n`).
 *
 * @param n          Length of chain
 * @return unsigned
 */
unsigned getMaxValue(const unsigned n);

}  // namespace cg