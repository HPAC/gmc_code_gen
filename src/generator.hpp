#ifndef GENERATOR_H
#define GENERATOR_H

#include <set>
#include <string>
#include <vector>

// #include "analyzer.hpp"
#include "algorithm.hpp"
#include "definitions.hpp"
#include "macros.hpp"
#include "matcher.hpp"
#include "matrix.hpp"
#include "settings_kernels.hpp"

namespace cg {

/**
 * ConfigGenerator
 *
 * @param path_output       dir where the generated code is placed -- relative
 * to current dir
 * @param prefix_function   Naming convention for functions.
 * @param prefix_flops      Naming convention for flop functions.
 * @param prefix_filename   Naming convention for different files.
 * @param filename          Name of the file where multiple algorithms are
 * printed.
 * @param driver_filename   Name of the driver file generated.
 * @param parenth_only      If true, generates only the parenthesisations.
 * @param chain_information Enables generating information of the chain
 * @param generate_algs     Enables generation of the algorithms
 * @param generate_flop     Enables generation of flop functions
 * @param trans_cleanup     Forces final transposition in the generated code
 */
struct ConfigGenerator {
  std::string path_output{PATH_GENERATOR};
  std::string path_verification{PATH_VERIFICATION};
  std::string prefix_function{PREFIX_FUNC_NAME};
  std::string prefix_flops{PREFIX_FLOPS_FUNC};
  std::string prefix_filename{PREFIX_FILENAME};
  std::string filename{SINGLE_FILENAME};
  std::string driver_filename{DRIVER_FILENAME};

  bool parenth_only = true;
  bool chain_information = true;
  bool generate_algs = true;
  bool generate_flop = true;
  bool trans_cleanup = true;

  std::vector<Kernel*> supported_kernels = all_kernels;
};

class Generator {
 public:
  /*  Data Members  */
  ConfigGenerator settings;
  Matcher matcher{settings.supported_kernels};

  MatrixChain chain;
  std::vector<Permutation> permutations;

  /*  Member Functions  */
  Generator() = default;

  Generator(const MatrixChain& in_chain);

  ~Generator() = default;

  void setMatrices(const MatrixChain& in_chain);

  void setPermutations(const std::vector<Permutation>& permutations);

  void setKernelsMatcher(const std::vector<Kernel*>& kernels);

  std::vector<Algorithm> getAlgorithms() const;

  /**
   * @brief Generate code for a single permutation in an individual file.
   *
   * @param ofile       the output file manager
   * @param permutation vector with the permutation. Element's range: [1,
   * chain.size()-1]
   */
  void algorithmToFile(const Permutation& permutation) const;

  void setToFiles(const std::vector<Permutation>& permutations) const;

  /**
   * @brief Generates code for a set of permutations in an individual file.
   *
   * @param permutations    vector of permutations
   * @param filename        file where the code must be written
   * @param function_prefix prefix of alg-funcs names (permutation is appended)
   */
  void setToOneFile(const std::vector<Permutation>& permutations) const;

  void allToFiles() const;

  void allToOneFile(const int generation_id = -1) const;

  void generateDriver(const int generation_id = -1) const;

  void loadModels() const;

  void generate(const unsigned K, const unsigned N_t,
                const std::string& flops_or_models, const unsigned id_F) const;

  void generateCodeSet(std::vector<Algorithm>& A,
                       const std::set<unsigned>& Z) const;

 private:
  void checkAndPush(const Matrix& mat);

  // void updatePermutation();

  std::string getName(const std::string& prefix,
                      const Permutation permutation) const;

  std::string getFilename(const int generation_id = -1) const;

  std::string getFilename(const Permutation& permutation,
                          const int generation_id = -1) const;

  std::string getDriverFilename(const int generation_id = -1) const;

  std::string getGuardHeader(const int generation_id) const;

  std::string getArguments() const;

  std::string getNamedArguments() const;

  std::string getArgumentsInvocation(const std::string PREFIX = "") const;

  std::string getBodySelection() const;

  std::string informationChain() const;
};

size_t factorial(const size_t n);

size_t catalan(const size_t n);

}  // namespace cg

#endif
