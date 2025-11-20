/**
 * @file generate_chains.test.cpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief Randomly generates shapes of chains to be used in the experiments with
 * execution time.
 * @version 0.1
 *
 * @copyright Copyright (c) 2025
 *
 * The generation guarantees at least one matrix is general rectangular. The
 * probability of a general rectangular matrix is 0.5 (TH_RECT), while the rest
 * of the probability is uniformly distributed amongst the other features. The
 * generated shapes are represented by a number, where each digit represents the
 * features of one matrix.
 *
 * @arg 1: n - length of the chain.
 * @arg 2: n_chains - number of shapes to generate.
 * @arg 3: out_file - file where to write the generated shapes.
 *
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <vector>

const double TH_RECT = 0.5;

void initThresholds(const double th_rect, std::vector<double>& thresholds) {
  double step = (1.0 - th_rect) / 9.0;
  for (unsigned i = 0U; i < 10U; i++)
    thresholds.emplace_back(th_rect + i * step);
}

unsigned randomFeatures(std::vector<double>& thresholds) {
  static std::random_device rand_dev;
  static std::mt19937 mt_rand_engine(rand_dev());
  static std::uniform_real_distribution<double> dist(0.0, 1.0);

  double x = dist(mt_rand_engine);
  auto iter = std::lower_bound(thresholds.begin(), thresholds.end(), x);
  return std::distance(thresholds.begin(), iter);
}

unsigned randomShape(const unsigned n, std::vector<double>& thresholds) {
  unsigned shape = 0U;
  for (unsigned i = 0U; i < n; i++) {
    shape += randomFeatures(thresholds) *
             static_cast<unsigned>(std::pow(10.0, static_cast<double>(i)));
  }
  return shape;
}

bool isSquareShape(const unsigned n, const unsigned id_shape) {
  bool square_shape = true;
  unsigned digit;
  float power_10;

  for (unsigned i = 0U; i < n; i++) {
    power_10 = std::pow(10.0, static_cast<float>(i));
    digit = id_shape / static_cast<unsigned>(power_10);  // fix this
    digit %= 10U;
    if (digit == 0U) square_shape = false;
  }

  return square_shape;
}

int main(int argc, char** argv) {
  unsigned n_shapes, n;
  std::string out_file;
  if (argc < 4) {
    std::cerr << "Usage: ./generate_chains.test <n> <n_shapes> <out_file>\n";
    return EXIT_FAILURE;
  }
  n = std::stoi(argv[1]);
  n_shapes = std::stoi(argv[2]);
  out_file = argv[3];

  std::vector<double> thresholds;
  initThresholds(TH_RECT, thresholds);

  std::vector<unsigned> shapes;
  while (shapes.size() < n_shapes) {
    unsigned new_shape = randomShape(n, thresholds);
    if (std::find(shapes.begin(), shapes.end(), new_shape) == shapes.end() and
        !isSquareShape(n, new_shape)) {
      shapes.emplace_back(new_shape);
    }
  }

  std::ofstream ofile;
  ofile.open(out_file);
  if (ofile.fail()) {
    std::cerr << "Cannot open " << out_file << '\n';
    exit(-1);
  }
  for (const auto& shape : shapes) ofile << shape << '\n';
  ofile.close();
}