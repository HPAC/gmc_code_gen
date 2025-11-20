#include "expr_generator.hpp"

#include <random>
#include <vector>

#include "definitions.hpp"
#include "macros.hpp"
#include "matrix.hpp"

namespace cg {

ExprGenerator::ExprGenerator() {
  std::random_device rd;
  random_generator = std::mt19937(rd());
  dist_real = std::uniform_real_distribution<double>(0.0, 1.0);
}

MatrixChain ExprGenerator::generateChain(const unsigned n) {
  MatrixChain chain{};
  char name = 'A';

  for (unsigned i = 0; i < n; i++) {
    chain.emplace_back(std::string(1, name), rndFeatures());
    name++;
  }

  return chain;
}

Features ExprGenerator::rndFeatures() {
  Features features;
  features.structure = rndStructure();
  features.property = rndProperty();
  if (features.property == Property::SPD and !features.isSymmetric()) {
    features.property = Property::FullRank;
  } else if (features.isUnit() and !features.isInvertible()) {
    features.property = Property::FullRank;
  }

  features.trans = rndTrans();
  if ((features.isSymmetric() or features.isDiagonal()) and
      features.isTransposed()) {
    features.trans = Trans::N;
  }

  features.inversion = rndInversion();
  if (features.isInverted() and !features.isInvertible()) {
    features.inversion = Inversion::N;
  }

  return features;
}

Structure ExprGenerator::rndStructure() {
  Structure str;
  double val = dist_real(random_generator);

  if (val < TH_GE) {
    str = Structure::Dense;

  } else if (val < TH_SY) {
    val = dist_real(random_generator);

    if (val < 0.5)
      str = Structure::Symmetric_L;
    else
      str = Structure::Symmetric_U;

  } else if (val < TH_TR) {
    val = dist_real(random_generator);
    if (val < 0.25)
      str = Structure::Lower;
    else if (val < 0.5)
      str = Structure::UnitLower;
    else if (val < 0.75)
      str = Structure::Upper;
    else
      str = Structure::UnitUpper;

  } else {
    str = Structure::Diagonal;
  }

  return str;
}

Property ExprGenerator::rndProperty() {
  Property ppty;
  double val = dist_real(random_generator);

  if (val < TH_NONE) {
    ppty = Property::None;
  } else if (val < TH_FR) {
    ppty = Property::FullRank;
  } else {
    ppty = Property::SPD;
  }

  return ppty;
}

Trans ExprGenerator::rndTrans() {
  Trans trans;
  double val = dist_real(random_generator);

  if (val < TH_NTRANS)
    trans = Trans::N;
  else
    trans = Trans::Y;

  return trans;
}

Inversion ExprGenerator::rndInversion() {
  Inversion inversion;
  double val = dist_real(random_generator);

  if (val < TH_NINV)
    inversion = Inversion::N;
  else
    inversion = Inversion::Y;

  return inversion;
}

}  // namespace cg
