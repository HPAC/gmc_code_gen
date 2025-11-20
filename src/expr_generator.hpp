#ifndef EXPR_GENERATOR_H
#define EXPR_GENERATOR_H

#include <random>
#include <string>
#include <vector>

#include "definitions.hpp"
#include "macros_expr_gen.hpp"
#include "matrix.hpp"

namespace cg {

class ExprGenerator {
 public:
  double th_structure[4] = {TH_GE, TH_SY, TH_TR,
                            TH_DI};                  // Dense, SY, TR, Diagonal
  double th_property[3] = {TH_NONE, TH_FR, TH_SPD};  // None, FR, SPD
  double th_trans[2] = {TH_NTRANS, TH_TRANS};        // No-trans, trans
  double th_inversion[2] = {TH_NINV, TH_INV};        // No-inv, inv

  std::uniform_real_distribution<double> dist_real{};
  std::mt19937 random_generator{};

  ExprGenerator();

  ~ExprGenerator() = default;

  MatrixChain generateChain(const unsigned n);

  Features rndFeatures();

  Structure rndStructure();

  Property rndProperty();

  Trans rndTrans();

  Inversion rndInversion();
};

}  // namespace cg

#endif