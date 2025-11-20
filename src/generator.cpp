#include "generator.hpp"

#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "algorithm.hpp"
#include "analyzer.hpp"
#include "definitions.hpp"
#include "instance_generator.hpp"
#include "matcher.hpp"
#include "matrix.hpp"
#include "permutation.hpp"

namespace cg {

Generator::Generator(const MatrixChain& in_chain) {
  for (const auto& mat : in_chain) checkAndPush(mat);
}

void Generator::setMatrices(const MatrixChain& in_chain) {
  // keep permutations iff chains are of the same length
  if (chain.size() != in_chain.size()) permutations.clear();

  chain.clear();
  for (const auto& mat : in_chain) checkAndPush(mat);
}

void Generator::setPermutations(const std::vector<Permutation>& permutations) {
  this->permutations = permutations;
}

void Generator::setKernelsMatcher(const std::vector<Kernel*>& kernels) {
  if (!matcher.isEmpty()) matcher.clear();
  matcher.addSetKernels(kernels);
}

std::vector<Algorithm> Generator::getAlgorithms() const {
  unsigned n_algs = (settings.parenth_only) ? catalan(chain.size() - 1UL)
                                            : factorial(chain.size() - 1UL);

  std::vector<Algorithm> algorithms;
  algorithms.reserve(n_algs);

  Permutation permutation(chain.size() - 1U);
  std::iota(permutation.begin(), permutation.end(), 1U);

  if (settings.parenth_only) {
    do {
      if (PermutationTransformer::canonicalize(permutation) == permutation) {
        algorithms.emplace_back(chain, permutation, &matcher);
      }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
  } else {
    do {
      algorithms.emplace_back(chain, permutation, &matcher);
    } while (std::next_permutation(permutation.begin(), permutation.end()));
  }

  return algorithms;
}

void Generator::algorithmToFile(const Permutation& permutation) const {
  std::ofstream ofile;
  if (settings.generate_algs or settings.generate_flop) {
    std::string filename = getFilename(permutation) + EXTENSION_UNIT;
    ofile.open(filename);
    if (ofile.fail()) {
      std::cerr << "Error opening file " << filename << '\n';
      std::cerr << "\t>> Remember to create the dir " << settings.path_output
                << '\n';
      exit(-1);
    }
    ofile << PREAMBLE_GENERATED << '\n' << "\n\n";
    if (settings.chain_information) ofile << informationChain();
  }

  Algorithm algorithm(chain, permutation, &matcher);

  if (settings.generate_flop)
    ofile << algorithm.generateCostFunction(
        getName(settings.prefix_flops, permutation));

  if (settings.generate_algs)
    ofile << algorithm.generateCode(
        getName(settings.prefix_function, permutation), settings.trans_cleanup);

  if (settings.generate_algs or settings.generate_flop) ofile.close();
}

void Generator::setToFiles(const std::vector<Permutation>& permutations) const {
  for (const auto& perm : permutations) {
    algorithmToFile(perm);
  }
}

void Generator::setToOneFile(
    const std::vector<Permutation>& permutations) const {
  std::ofstream ofile;
  if (settings.generate_algs or settings.generate_flop) {
    std::string filename = getFilename() + EXTENSION_UNIT;
    ofile.open(filename);
    if (ofile.fail()) {
      std::cerr << "Error opening file " << filename << '\n';
      std::cerr << "\t>> Remember to create the dir " << settings.path_output
                << '\n';
      exit(-1);
    }
    ofile << PREAMBLE_GENERATED << '\n' << "\n\n";
    if (settings.chain_information) ofile << informationChain();
  }

  for (const auto& perm : permutations) {
    Algorithm algorithm(chain, perm, &matcher);

    if (settings.generate_flop)
      ofile << algorithm.generateCostFunction(
          getName(settings.prefix_flops, perm));

    if (settings.generate_algs)
      ofile << algorithm.generateCode(getName(settings.prefix_function, perm),
                                      settings.trans_cleanup);
  }

  if (settings.generate_algs or settings.generate_flop) ofile.close();
}

void Generator::allToFiles() const {
  Permutation permutation(chain.size() - 1U);
  std::iota(permutation.begin(), permutation.end(), 1U);

  do {
    if (settings.parenth_only) {
      if (PermutationTransformer::canonicalize(permutation) == permutation) {
        algorithmToFile(permutation);
      }
    } else {
      algorithmToFile(permutation);
    }
  } while (std::next_permutation(permutation.begin(), permutation.end()));
}

void Generator::allToOneFile(const int generation_id) const {
  std::ofstream unit_file;
  std::ofstream header_file;

  std::string filename = getFilename(generation_id);
  std::string filename_unit = filename + EXTENSION_UNIT;
  std::string filename_header = filename + EXTENSION_HEADER;

  unit_file.open(settings.path_output + "/" + filename_unit);
  if (unit_file.fail()) {
    std::cerr << "Error opening file " << filename_unit << '\n';
    std::cerr << "\t>> Remember to create the dir " << settings.path_output
              << " within the current dir\n";
    exit(-1);
  }
  unit_file << fmt::format("#include \"{}\"\n", filename_header);
  unit_file << PREAMBLE_GENERATED << "\n\n";

  header_file.open(settings.path_output + "/" + filename_header);
  if (header_file.fail()) {
    std::cerr << "Error opening file " << filename_header << '\n';
    std::cerr << "\t>> Remember to create the dir " << settings.path_output
              << " within the current dir\n";
    exit(-1);
  }

  header_file << getGuardHeader(generation_id) << "\n\n";
  header_file << PREAMBLE_GENERATED << "\n\n";
  header_file << fmt::format("{} {} = {} (*)({});\n", USING, STR(ALG_PTR),
                             STR(MATRIX), getArguments());
  header_file << fmt::format("{} {} = double (*)({});\n\n", USING,
                             STR(FLOPS_PTR), getArguments());

  if (settings.chain_information) header_file << informationChain() << "\n";

  std::ostringstream oss;
  oss << fmt::format("inline const std::vector<std::pair<{}, {}>> {} {{",
                     STR(FLOPS_PTR), STR(ALG_PTR), STR(VEC_NAME));

  Permutation permutation(chain.size() - 1U);
  std::iota(permutation.begin(), permutation.end(), 1U);

  bool first = true;
  do {
    if (PermutationTransformer::canonicalize(permutation) == permutation) {
      Algorithm algorithm(chain, permutation, &matcher);
      std::string flop_function_name =
          getName(settings.prefix_flops, permutation);
      std::string alg_function_name =
          getName(settings.prefix_function, permutation);

      if (first) {
        first = false;
      } else {
        oss << ", ";
      }
      oss << "{" << flop_function_name << ", " << alg_function_name << "}";

      header_file << algorithm.getSignature("double", flop_function_name)
                  << ";\n";
      header_file << algorithm.getSignature(STR(MATRIX), alg_function_name)
                  << ";\n\n";

      unit_file << algorithm.generateCostFunction(flop_function_name);

      unit_file << algorithm.generateCode(alg_function_name,
                                          settings.trans_cleanup)
                << "\n";
    }
  } while (std::next_permutation(permutation.begin(), permutation.end()));

  oss << "};\n";
  header_file << oss.str();

  std::string signature_selection = fmt::format(
      "{} {}({})", STR(ALG_PTR), STR(SELECT_FUNC), getNamedArguments());

  header_file << signature_selection << ";\n\n";
  header_file << "#endif";

  unit_file << signature_selection;
  unit_file << getBodySelection();

  unit_file.close();
  header_file.close();
}

void Generator::generateDriver(const int generation_id) const {
  std::ofstream ofile;
  std::ostringstream oss;

  std::string name_rand_gen = "random_gen";
  std::string name_dist = "dist";
  std::string invocation_dist = name_dist + "(" + name_rand_gen + ")";

  oss << fmt::format("#include \"{}\"\n",
                     getFilename(generation_id) + EXTENSION_HEADER);
  oss << PREAMBLE_DRIVER;
  oss << "int main() {\n";
  oss << "std::random_device rd;\n";
  oss << "std::mt19937 " << name_rand_gen << "(rd());\n";
  oss << "std::uniform_int_distribution<unsigned>" << name_dist << "("
      << SMALLEST_DIM << ", " << LARGEST_DIM << ");\n\n";

  char name_dim = 'k';
  std::string type_dim = "const unsigned";
  unsigned last_free_dim = 0U;
  oss << type_dim << " " << name_dim << last_free_dim << " = "
      << invocation_dist << ";\n";

  std::vector<unsigned> dimensions(chain.size() + 1);
  std::iota(dimensions.begin(), dimensions.end(), 0U);

  for (unsigned i = 1; i < dimensions.size(); i++) {
    if (!chain[i - 1].isDense() or chain[i - 1].isInvertible()) {
      dimensions[i] = last_free_dim;
      oss << "// " << type_dim << " " << name_dim << i << " = " << name_dim
          << last_free_dim << ";\n";
    } else {
      last_free_dim = i;
      oss << type_dim << " " << name_dim << last_free_dim << " = "
          << invocation_dist << ";\n";
    }
  }
  oss << "\n";

  std::string gen_matrix = "generateMatrix";
  int idx_rows = 0;
  int idx_columns = 0;
  for (unsigned i = 0; i < chain.size(); i++) {
    idx_rows = chain[i].isTransposed() ? i + 1 : i;
    idx_columns = chain[i].isTransposed() ? i : i + 1;

    oss << STR(MATRIX) << " " << chain[i].getName() << " = " << gen_matrix
        << "(" << to_qualified_string(chain[i]) << ", " << name_dim
        << dimensions[idx_rows] << ", " << name_dim << dimensions[idx_columns]
        << ");\n";
    oss << STR(MATRIX) << " " << PREFIX_COPY << chain[i].getName() << " = "
        << chain[i].getName() << ";\n";
  }

  oss << fmt::format("{} result = {}[0].second({});\n", STR(MATRIX),
                     STR(VEC_NAME), getArgumentsInvocation(PREFIX_COPY));
  oss << fmt::format("for (auto& f : {}) {{\n", STR(VEC_NAME));
  for (unsigned i = 0; i < chain.size(); i++) {
    oss << PREFIX_COPY << chain[i].getName() << " = " << chain[i].getName()
        << ";\n";
  }
  oss << fmt::format("if (compareMatrix(result, f.second({})) > 1e-10)\n",
                     getArgumentsInvocation(PREFIX_COPY));
  oss << "std::cout << \"❌\";\nelse\nstd::cout << \"✅\";\n}\n";
  oss << "std::cout << \"\\n\";\n}\n";

  ofile.open(settings.path_verification + "/" +
             getDriverFilename(generation_id) + EXTENSION_DRIVER);
  ofile << oss.str();
  ofile.close();
}

void Generator::loadModels() const { matcher.loadModels(); }

void Generator::generate(const unsigned K, const unsigned N_t,
                         const std::string& flops_or_models,
                         const unsigned id_F) const {
  using metricFunc = std::function<double(const std::vector<double>&)>;

  auto A = this->getAlgorithms();
  const unsigned M = A.size();
  const unsigned n = chain.size();
  std::vector<metricFunc> obj_funcs = {maxPenalty, avgPenalty};
  cg::InstanceGenerator inst_gnrtor(50U, 1000U);
  auto Q_t = inst_gnrtor.rndInstances(chain, N_t);
  std::vector<double> penalty_matrix;

  if (flops_or_models == "flops" or flops_or_models == "FLOPs")
    penalty_matrix = FLOPsOnInstances(A, Q_t);
  else {
    loadModels();
    penalty_matrix = predTimeOnInstances(A, Q_t);
  }
  cost2penalty(M, N_t, penalty_matrix);

  std::set<unsigned> Es_free_dims;
  std::vector<std::vector<unsigned>> coupled_dims;
  auto free_dims_idx = cg::getFreeDimsIdx(chain, coupled_dims);
  insertFanDimIdx(A, n, Es_free_dims, free_dims_idx);
  auto C = dimId2AlgId(A, n, coupled_dims);
  auto F = obj_funcs[id_F];
  auto Z =
      cg::findOptimalFanningSet(M, N_t, penalty_matrix, Es_free_dims, C, F);

  if (static_cast<unsigned>(Z.size()) < K)
    Z = SEGreedyWPenalty(M, N_t, K, F, Z, penalty_matrix);

  this->generateCodeSet(A, Z);
}

void Generator::generateCodeSet(std::vector<Algorithm>& A,
                                const std::set<unsigned>& Z) const {
  std::vector<unsigned> id_algs;
  for (const auto& id : Z) id_algs.push_back(id);

  std::ofstream unit_file;
  std::ofstream header_file;

  std::string filename = getFilename();
  std::string filename_unit = filename + EXTENSION_UNIT;
  std::string filename_header = filename + EXTENSION_HEADER;

  unit_file.open(settings.path_output + "/" + filename_unit);
  if (unit_file.fail()) {
    std::cerr << "Error opening file " << filename_unit << '\n';
    exit(-1);
  }
  unit_file << fmt::format("#include \"{}\"\n", filename_header);
  unit_file << PREAMBLE_GENERATED << "\n\n";

  header_file.open(settings.path_output + "/" + filename_header);
  if (header_file.fail()) {
    std::cerr << "Error opening file " << filename_header << '\n';
    exit(-1);
  }
  header_file << getGuardHeader(1) << "\n\n";
  header_file << PREAMBLE_GENERATED << "\n\n";
  header_file << fmt::format("{} {} = {} (*)({});\n", USING, STR(ALG_PTR),
                             STR(MATRIX), getArguments());
  header_file << fmt::format("{} {} = double (*)({});\n\n", USING,
                             STR(FLOPS_PTR), getArguments());

  if (settings.chain_information) header_file << informationChain() << "\n";

  std::ostringstream oss;
  oss << fmt::format("inline const std::vector<std::pair<{}, {}>> {} {{",
                     STR(FLOPS_PTR), STR(ALG_PTR), STR(VEC_NAME));
  bool first = true;
  unsigned i = 0;
  do {
    auto permutation = A[id_algs[i]].getPermutation();
    std::string flops_func_name = getName(settings.prefix_flops, permutation);
    std::string alg_func_name = getName(settings.prefix_function, permutation);

    if (first) {
      first = false;
    } else {
      oss << ", ";
    }
    oss << "{" << flops_func_name << ", " << alg_func_name << "}";

    header_file << A[id_algs[i]].getSignature("double", flops_func_name)
                << ";\n";
    header_file << A[id_algs[i]].getSignature(STR(MATRIX), alg_func_name)
                << ";\n\n";

    unit_file << A[id_algs[i]].generateCostFunction(flops_func_name);

    unit_file << A[id_algs[i]].generateCode(alg_func_name,
                                            settings.trans_cleanup)
              << "\n";

    i++;
  } while (i < id_algs.size());

  oss << "};\n";
  header_file << oss.str();

  std::string signature_selection = fmt::format(
      "{} {}({})", STR(ALG_PTR), STR(SELECT_FUNC), getNamedArguments());
  header_file << signature_selection << ";\n\n";
  header_file << "#endif";

  unit_file << signature_selection;
  unit_file << getBodySelection();

  unit_file.close();
  header_file.close();

  std::cout << "output files in " << settings.path_output << ": \n\t"
            << filename_unit << "\n\t" << filename_header << "\n";
}

void Generator::checkAndPush(const Matrix& mat) {
  // @todo try-catch
  if (mat.isLegal()) {
    if (!mat.isIdentity()) {
      chain.emplace_back(mat);
      chain.back().simplify();
    } else {
      std::cout << "matrix is identity: " << mat << "\n";
      // @todo modifyPermutation();
      // updatePermutation();
    }
  } else {
    std::cerr << mat.name << " has an illegal combination of features\n";
    std::cerr << mat << "\n";
    exit(-1);
  }
}

std::string Generator::getName(const std::string& prefix,
                               const Permutation permutation) const {
  std::string func_name = prefix;
  for (const auto& i : permutation) func_name += std::to_string(i);
  return func_name;
}

std::string Generator::getFilename(const int generation_id) const {
  if (generation_id == -1) {
    return settings.filename;
  } else {
    return fmt::format("{}_{}", settings.filename, generation_id);
  }
}

std::string Generator::getFilename(const Permutation& permutation,
                                   const int generation_id) const {
  if (generation_id == -1) {
    return getName(settings.prefix_filename, permutation);
  } else {
    return fmt::format("{}_{}", getName(settings.prefix_filename, permutation),
                       generation_id);
  }
}

std::string Generator::getDriverFilename(const int generation_id) const {
  if (generation_id == -1) {
    return settings.driver_filename;
  } else {
    return fmt::format("{}_{}", settings.driver_filename, generation_id);
  }
}

std::string Generator::getGuardHeader(const int generation_id) const {
  if (generation_id == -1) {
    return fmt::format("#ifndef {0}\n#define {0}", PREFIX_GUARD);
  } else {
    return fmt::format("#ifndef {0}_{1}\n#define {0}_{1}", PREFIX_GUARD,
                       generation_id);
  }
}

std::string Generator::getArguments() const {
  std::ostringstream oss;
  for (unsigned i = 0; i < chain.size(); i++) {
    oss << STR(MATRIX) << "&";
    if (i != chain.size() - 1) oss << ", ";
  }
  return oss.str();
}

std::string Generator::getNamedArguments() const {
  std::ostringstream oss;
  for (unsigned i = 0; i < chain.size(); i++) {
    oss << STR(MATRIX) << "& " << chain[i].getName();
    if (i != chain.size() - 1) oss << ", ";
  }
  return oss.str();
}

std::string Generator::getArgumentsInvocation(const std::string PREFIX) const {
  std::ostringstream oss;
  for (unsigned i = 0; i < chain.size(); i++) {
    oss << PREFIX << chain[i].getName();
    if (i != chain.size() - 1) oss << ", ";
  }

  return oss.str();
}

std::string Generator::getBodySelection() const {
  std::ostringstream oss;
  oss << "{\n"
      << STR(ALG_PTR)
      << " selected = nullptr;\ndouble min = 1e20;\ndouble aux;\n"
      << "for (auto& f : " << STR(VEC_NAME) << ") {\n"
      << "aux = f.first(" << getArgumentsInvocation() << ");\n"
      << "if (aux < min) {\n"
      << "min = aux;\n"
      << "selected = f.second;}\n}\n"
      << "return selected;\n}";
  return oss.str();
}

std::string Generator::informationChain() const {
  std::string code = "/*\n";

  for (const auto& mat : chain)
    code += fmt::format(" *  {}: {}\n", mat.getName(), to_string(mat));

  code += " */\n";

  return code;
}

size_t factorial(const size_t n) {
  if (n == 1UL)
    return 1UL;
  else
    return n * factorial(n - 1UL);
}

size_t catalan(const size_t n) {
  return factorial(2UL * n) / (factorial(n + 1UL) * factorial(n));
}

}  // namespace cg
