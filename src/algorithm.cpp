#include "algorithm.hpp"

#include <fmt/core.h>

#include <algorithm>
#include <string>
#include <vector>

#include "definitions.hpp"
#include "macros.hpp"
#include "matcher.hpp"
#include "matrix.hpp"
#include "node.hpp"
#include "utils/dMatrix.hpp"

namespace cg {

Algorithm::Algorithm(const MatrixChain& chain, const Permutation& permutation,
                     const Matcher* const matcher)
    : permutation{permutation} {
  tree.reserve(2U * chain.size() - 1U);
  createInputNodes(chain);
  buildTree(matcher);
}

std::string Algorithm::generateCode(const std::string& function_name,
                                    const bool trans_cleanup) {
  std::string code = getSignature(STR(MATRIX), function_name);
  code += "{\n";
  for (unsigned i = permutation.size() + 1U; i < tree.size(); ++i) {
    code += tree[i].generateCode(tree);
    code += "\n";
  }

  if (trans_cleanup) code += tree.back().generateCleanup();

  code += getClosure(tree.back().matrix.getName());
  return code;
}

std::string Algorithm::generateCostFunction(const std::string& function_name) {
  std::string code = getSignature("double", function_name);
  code += "{\n";
  code += fmt::format("double {}{{}};\n", COST_VAR);

  for (unsigned i = permutation.size() + 1; i < tree.size(); ++i) {
    code += tree[i].generateCost(tree);
  }

  code += getClosure(COST_VAR);
  return code;
}

double Algorithm::computeFLOPs(const Instance& instance) {
  assignSizes(instance);
  double flops = 0.0;
  for (unsigned i = permutation.size() + 1U; i < tree.size(); ++i) {
    flops += tree[i].computeFLOPs(tree);
  }

  return flops;
}

double Algorithm::predictTime(const Instance& instance) {
  assignSizes(instance);

  double time = 0.0;
  for (unsigned i = permutation.size() + 1U; i < tree.size(); ++i) {
    time += tree[i].predictTime(tree);
  }
  return time;
}

void Algorithm::assignChain(const std::vector<dMatrix>& live_chain) {
  for (unsigned i = 0U; i < live_chain.size(); ++i) {
    tree[i].setLiveMatrix(live_chain[i]);
  }
}

dMatrix Algorithm::execute() {
  for (unsigned i = permutation.size() + 1U; i < tree.size(); ++i) {
    tree[i].execute(tree);
  }
  auto mat_ptr = tree.back().getLiveMatrixPtr();
  dMatrix result = std::move(*mat_ptr);
  mat_ptr.release();
  return result;
}

void Algorithm::clean() {
  for (unsigned i = 0U; i < tree.size(); ++i) tree[i].deleteMatrix();
}

Permutation Algorithm::getPermutation() const { return permutation; }

void Algorithm::createInputNodes(const MatrixChain& chain) {
  for (unsigned i = 0U; i < chain.size(); ++i) tree.emplace_back(chain[i]);
}

void Algorithm::buildTree(const Matcher* const matcher) {
  int8_t left, right;

  for (const auto& p : permutation) {
    left = getRoot(p - 1);
    right = getRoot(p);

    tree.emplace_back(static_cast<int8_t>(left), static_cast<int8_t>(right),
                      matcher, PREFIX_TEMP + std::to_string(p), tree);

    tree[left].setTop(tree.size() - 1);
    tree[right].setTop(tree.size() - 1);
  }
}

void Algorithm::assignSizes(const Instance& instance) {
  for (unsigned i = 0; i <= permutation.size(); ++i) {
    if (tree[i].getTransposed()) {
      tree[i].matrix.nrows = instance[i + 1];
      tree[i].matrix.ncols = instance[i];
    } else {
      tree[i].matrix.nrows = instance[i];
      tree[i].matrix.ncols = instance[i + 1];
    }
  }
  for (unsigned i = permutation.size() + 1; i < tree.size(); ++i) {
    tree[i].propagateSizes(tree);
  }
}

std::string Algorithm::getSignature(const std::string& ret_type,
                                    const std::string& function_name) {
  std::string code = fmt::format("{} {}(", ret_type, function_name);

  for (unsigned i = 0; i <= permutation.size(); ++i) {
    code += fmt::format("{}& {}{}", STR(MATRIX), tree[i].matrix.getName(),
                        (i == permutation.size()) ? ")" : ", ");
  }
  return code;
}

std::string Algorithm::getClosure(const std::string& var) const {
  if (var == "")
    return fmt::format("}}\n");
  else
    return fmt::format("return {};\n}}\n", var);
}

int8_t Algorithm::getRoot(const int8_t id) const {
  auto top = tree[id].getTop();
  return (top == -1) ? id : getRoot(top);
}

bool Algorithm::isResultInverted() const {
  return tree.back().matrix.isInverted();
}

}  // namespace cg