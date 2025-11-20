/**
 * @file model_loading.test.cpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief Simple test to check that models are loaded correctly. Models should
 * be in the path indicated in `src/models/settings_models.hpp`.
 * @version 0.1
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <iostream>

#include "../src/matcher.hpp"
#include "../src/settings_kernels.hpp"

int main() {
  cg::Matcher matcher(cg::all_kernels);
  matcher.loadModels();
  std::cout << "Successfully loaded models.\n";
}