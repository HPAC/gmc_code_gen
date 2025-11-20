/**
 * @file timings_10comb.cpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief Generates timings through algorithm::execute() for specified shapes.
 * @version 0.1
 *
 * @copyright Copyright (c) 2025
 *
 * This program generates the timings through algorithm::execute() for a
 * user-specified shape and user-specified number of instances. The user also
 * specifies the number of times each algorithm is run on each instance. All
 * timings are dumped to the output file (i.e., there is no statistical
 * processing done). A number of expressions is given in @p expr_file, where
 * each chain is represented by a single unsigned integer, where each digit
 * indicates a different combination of features for each matrix.
 *
 * @param exprID: identifier of the expression -- must be >= 1.
 * @param n: length of the chain
 * @param N_s: number of random instances where to run each algorithm
 * @param reps: repetitions each algorithm is run on each instance
 * @param expr_file: file with expresssions
 * @param out_file: path to base of file where results are created.
 *
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <string>

#include "../src/algorithm.hpp"
#include "../src/analyzer.hpp"
#include "../src/definitions.hpp"
#include "../src/experiment_util.hpp"
#include "../src/generator.hpp"
#include "../src/instance_generator.hpp"
#include "../src/matrix.hpp"

// Read the ShapeID from the file with expressions
unsigned readShapeID(const std::string& expr_file, const unsigned exprID) {
  std::ifstream ifile;
  ifile.open(expr_file, std::ifstream::in);
  if (ifile.fail()) {
    std::cerr << "Cannot open " << expr_file << "\n";
    exit(-1);
  }

  // loop to skip exprID lines
  for (unsigned i = 0U; i < exprID; i++)
    ifile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

  std::string line;
  ifile >> line;
  ifile.close();
  unsigned shapeID = std::stoi(line);
  return shapeID;
}

int main(int argc, char** argv) {
  unsigned exprID, n, N_s, reps;
  std::string expr_file, out_file;

  if (argc < 7) {
    std::cerr << "Error. Arguments: <exprID> <n> <N_s> <reps> <expr_file> "
                 "<out_file>\n"
              << "Check code for interpretation of arguments.\n";
    exit(-1);
  }
  exprID = std::stoi(argv[1]);
  n = std::stoi(argv[2]);
  N_s = std::stoi(argv[3]);
  reps = std::stoi(argv[4]);
  expr_file = argv[5];
  out_file = argv[6];

  unsigned shapeID = readShapeID(expr_file, exprID);  // Get shapeID from file
  cg::MatrixChain chain = cg::ID2chain(n, shapeID);   // symbolic chain
  cg::Generator generator(chain);                 // Pass it to the generator.
  auto A = generator.getAlgorithms();             // Get the algorithms.
  cg::InstanceGenerator inst_gnrtor(50U, 1000U);  // instance generator

  cg::RawTimes raw_times;
  raw_times.shapeID = shapeID;
  raw_times.size_instance = n + 1U;
  raw_times.M = A.size();
  raw_times.N = N_s;
  raw_times.reps = reps;
  raw_times.Q = inst_gnrtor.rndInstances(chain, N_s);  // generate instances

  auto t0 = std::chrono::high_resolution_clock::now();
  raw_times.times = cg::timeAlgsOnInstances(chain, A, raw_times.Q, reps);
  auto t1 = std::chrono::high_resolution_clock::now();
  double elapsed = std::chrono::duration<double>(t1 - t0).count();

  std::cout << "Time taken to compute timings: " << elapsed << '\n';

  // Timings are dumped to the output file.
  cg::writeRawTimes(out_file, raw_times);
}
