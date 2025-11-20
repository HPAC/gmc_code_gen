#include "permutation.hpp"

#include <algorithm>
#include <vector>

namespace cg {

std::vector<PermutationTransformer::InfoEntry>
    PermutationTransformer::table_info;

std::vector<unsigned> PermutationTransformer::canonicalize(
    const std::vector<unsigned>& perm) {
  clearTable();
  table_info.resize(perm.size());
  buildRepresentation(perm);
  return buildPermutation(perm);
}

void PermutationTransformer::buildRepresentation(
    const std::vector<unsigned>& perm) {
  for (const auto& p : perm) {
    table_info[p - 1].computed = true;
    addDependencies(p);
  }
}

void PermutationTransformer::addDependencies(const unsigned& p) {
  // add dependencies left
  bool stop = false;
  for (int i = p - 2; i >= 0 and !stop; --i) {
    if (!table_info[i].computed) {
      table_info[i].right = p;
      stop = true;
    }
  }

  // add dependencies right
  stop = false;
  for (int i = p; i < static_cast<int>(table_info.size()) and !stop; ++i) {
    if (!table_info[i].computed) {
      table_info[i].left = p;
      stop = true;
    }
  }
}

std::vector<unsigned> PermutationTransformer::buildPermutation(
    const std::vector<unsigned>& perm) {
  std::vector<unsigned> canonical_perm{};

  buildRecursive(perm.back(), canonical_perm);
  std::reverse(canonical_perm.begin(), canonical_perm.end());
  return canonical_perm;
}

void PermutationTransformer::buildRecursive(
    const unsigned p, std::vector<unsigned>& canonical_perm) {
  canonical_perm.push_back(p);

  if (table_info[p - 1].right != -1)
    buildRecursive(table_info[p - 1].right, canonical_perm);

  if (table_info[p - 1].left != -1)
    buildRecursive(table_info[p - 1].left, canonical_perm);
}

void PermutationTransformer::clearTable() {
  for (auto& entry : table_info) {
    entry.computed = false;
    entry.left = -1;
    entry.right = -1;
  }
}

}  // namespace cg
