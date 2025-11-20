/**
 * @file features.cpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief Definition of basic features of symbolic matrices. The main class is
 * `Features`, which contains the following: `Structure`, `Property`,
 * `Transposition`, `Inversion`.
 * @version 0.1
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "features.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>

namespace cg {

std::ostream& operator<<(std::ostream& os, const Structure& structure) {
  switch (structure) {
    case Structure::Dense:
      os << "Dense";
      break;

    case Structure::Symmetric_L:
      os << "SymmetricLower";
      break;

    case Structure::Symmetric_U:
      os << "SymmetricUpper";
      break;

    case Structure::Lower:
      os << "Lower";
      break;

    case Structure::UnitLower:
      os << "UnitLower";
      break;

    case Structure::Upper:
      os << "Upper";
      break;

    case Structure::UnitUpper:
      os << "UnitUpper";
      break;

    case Structure::Diagonal:
      os << "Diagonal";
      break;

    default:
      break;
  }
  return os;
}

Structure transposeStructure(const Structure& structure) {
  switch (structure) {
    case Structure::Lower:
      return Structure::Upper;
      break;

    case Structure::UnitLower:
      return Structure::UnitUpper;
      break;

    case Structure::Upper:
      return Structure::Lower;
      break;

    case Structure::UnitUpper:
      return Structure::UnitLower;
      break;

    default:
      return structure;
  }
}

/* table for propagation of structures.
 |  1  2  3  4  5  6  7
-|---------------------
1|  1  1  1  1  1  1  1
2|  1  1  1  1  1  1  1
3|  1  1  3  3  1  1  3
4|  1  1  3  4  1  1  3
5|  1  1  1  1  5  5  5
6|  1  1  1  1  5  6  5
7|  1  1  3  3  5  5  7
*/

Structure operator*(const Structure s1, const Structure s2) {
  Structure result{Structure::Dense};
  Structure temp_s1 = s1;
  Structure temp_s2 = s2;

  if (temp_s1 > temp_s2) std::swap(temp_s1, temp_s2);

  switch (temp_s1) {
    case Structure::Dense:
      result = Structure::Dense;
      break;

    case Structure::Symmetric_L:
      result = Structure::Dense;
      break;

    case Structure::Symmetric_U:
      result = Structure::Dense;
      break;

    case Structure::Lower:
      if (temp_s2 == Structure::Lower or temp_s2 == Structure::UnitLower or
          temp_s2 == Structure::Diagonal)
        result = Structure::Lower;
      else if (temp_s2 == Structure::Upper or temp_s2 == Structure::UnitUpper)
        result = Structure::Dense;
      break;

    case Structure::UnitLower:
      if (temp_s2 == Structure::UnitLower)
        result = Structure::UnitLower;
      else if (temp_s2 == Structure::Upper or temp_s2 == Structure::UnitUpper)
        result = Structure::Dense;
      else if (temp_s2 == Structure::Diagonal)
        result = Structure::Lower;
      break;

    case Structure::Upper:
      result = Structure::Upper;
      break;

    case Structure::UnitUpper:
      if (temp_s2 == Structure::UnitUpper)
        result = Structure::UnitUpper;
      else if (temp_s2 == Structure::Diagonal)
        result = Structure::Upper;
      break;

    case Structure::Diagonal:
      result = Structure::Diagonal;
      break;

    default:
      break;
  }
  return result;
}

std::ostream& operator<<(std::ostream& os, const Property& property) {
  switch (property) {
    case Property::None:
      os << "None";
      break;

    case Property::FullRank:
      os << "FullRank";
      break;

    case Property::SPD:
      os << "SPD";
      break;

    case Property::Orthogonal:
      os << "Orthogonal";
      break;

    default:
      break;
  }

  return os;
}

/* table for propagation of properties
 |  1  2  3  4
-|------------
1|  1  1  1  1
2|  1  2  2  2
3|  1  2  2  2
4|  1  2  2  4
*/

Property operator*(const Property p1, const Property p2) {
  Property result{Property::None};
  Property temp_p1 = p1;
  Property temp_p2 = p2;

  if (temp_p1 > temp_p2) std::swap(temp_p1, temp_p2);

  switch (temp_p1) {
    case Property::None:
      result = Property::None;
      break;

    case Property::FullRank:
      result = Property::FullRank;
      break;

    case Property::SPD:
      result = Property::FullRank;
      break;

    case Property::Orthogonal:
      result = Property::Orthogonal;
      break;

    default:
      break;
  }

  return result;
}

std::ostream& operator<<(std::ostream& os, const Trans& trans) {
  if (trans == Trans::N)
    os << "N";
  else
    os << "Y";

  return os;
}

std::ostream& operator<<(std::ostream& os, const Inversion& inv) {
  if (inv == Inversion::N)
    os << "N";
  else
    os << "Y";

  return os;
}

bool Features::operator==(const Features& rhs) const {
  return (structure == rhs.structure) and (property == rhs.property) and
         (trans == rhs.trans) and (inversion == rhs.inversion);
}

void Features::flipUplo() {
  if (structure == Structure::Upper)
    structure = Structure::Lower;
  else if (structure == Structure::Lower)
    structure = Structure::Upper;
  else if (structure == Structure::Symmetric_U)
    structure = Structure::Symmetric_L;
  else if (structure == Structure::Symmetric_L)
    structure = Structure::Symmetric_U;
}

void Features::T() {
  trans = (trans == Trans::Y) ? Trans::N : Trans::Y;
  simplify();
}

void Features::inv() {
  inversion = (inversion == Inversion::Y) ? Inversion::N : Inversion::Y;
  simplify();
}

void Features::simplify() {
  if (property == Property::Orthogonal and isInverted()) {
    this->inversion = Inversion::N;
    this->trans = (this->trans == Trans::N) ? Trans::Y : Trans::N;
  }

  // Diagonal and SYM are immune to TRANS
  if ((isSymmetric() or isDiagonal()) and trans == Trans::Y) {
    trans = Trans::N;
  }
}

bool Features::isDense() const { return structure == Structure::Dense; }

bool Features::isSymmetric() const {
  return (structure == Structure::Symmetric_L) or
         (structure == Structure::Symmetric_U);
}

bool Features::isSymmetricLower() const {
  return (structure == Structure::Symmetric_L);
}

bool Features::isTriangular() const {
  return (std::find(triangular.begin(), triangular.end(), structure) !=
          triangular.end());
}

bool Features::isLower() const {
  return (structure == Structure::Lower) or (structure == Structure::UnitLower);
}

bool Features::isUnit() const {
  return (structure == Structure::UnitLower) or
         (structure == Structure::UnitUpper);
}

bool Features::isDiagonal() const { return (structure == Structure::Diagonal); }

bool Features::isSPD() const { return (property == Property::SPD); }

bool Features::isOrthogonal() const {
  return (property == Property::Orthogonal);
}

bool Features::isInvertible() const { return (property != Property::None); }

bool Features::isTransposed() const { return (trans == Trans::Y); }

bool Features::isInverted() const { return (inversion == Inversion::Y); }

bool Features::isLegal() const {
  bool legal = true;

  // By definition SPD must include SYM
  if (property == Property::SPD and !(isSymmetric() or isDiagonal()))
    legal = false;

  // A matrix without any invertible property cannot be inverted.
  else if (isInverted() and !isInvertible())
    legal = false;

  else if (isUnit() and !isInvertible())
    legal = false;

  return legal;
}

bool Features::isIdentity() const {
  bool identity = false;

  if (isDiagonal() and property == Property::Orthogonal)
    identity = true;

  else if (isTriangular() and property == Property::Orthogonal)
    identity = true;

  return identity;
}

std::ostream& operator<<(std::ostream& os, const Features& features) {
  os << "<" << features.structure << "," << features.property << ","
     << features.trans << "," << features.inversion << ">";
  return os;
}

std::string to_string(const Features& features) {
  std::ostringstream ss;
  ss << features;
  return ss.str();
}

std::string to_qualified_string(const Features& features) {
  std::ostringstream ss;
  ss << "{cg::Structure::" << features.structure
     << ", cg::Property::" << features.property
     << ", cg::Trans::" << features.trans
     << ", cg::Inversion::" << features.inversion << "}";
  return ss.str();
}

}  // namespace cg
