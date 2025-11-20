/**
 * @file features.hpp
 * @author FranLS7 (flopz@cs.umu.se)
 * @brief Definition of basic features of symbolic matrices. The main class is
 * `Features`, which contains the following: `Structure`, `Property`,
 * `Transposition`, `Inversion`.
 * @version 0.1
 *
 * @copyright Copyright (c) 2023
 *
 */

#ifndef FEATURES_H
#define FEATURES_H

#include <iostream>
#include <vector>

namespace cg {

/* ===============================================================
   ========================== STRUCTURE ==========================
   =============================================================== */
enum class Structure {
  Dense,
  Symmetric_L,
  Symmetric_U,
  Lower,
  UnitLower,
  Upper,
  UnitUpper,
  Diagonal
};

/* Sets of structures -- used for generation of covered variants by kernels */
inline const std::vector<Structure> all_structures{
    Structure::Dense,     Structure::Symmetric_L, Structure::Symmetric_U,
    Structure::Lower,     Structure::UnitLower,   Structure::Upper,
    Structure::UnitUpper, Structure::Diagonal};

inline const std::vector<Structure> triangular{
    Structure::Lower, Structure::UnitLower, Structure::Upper,
    Structure::UnitUpper};

inline const std::vector<Structure> non_triangular{
    Structure::Dense, Structure::Symmetric_L, Structure::Symmetric_U};

std::ostream& operator<<(std::ostream& os, const Structure& structure);

Structure transposeStructure(const Structure& structure);

/**
 * @brief Infers the structure resulting from multiplying two structures.
 *
 * @param s1 Structure of the first operand
 * @param s2 Structure of the second operand
 * @return Structure of the product
 */
Structure operator*(const Structure s1, const Structure s2);

/* ===============================================================
   ========================== PROPERTY ===========================
   =============================================================== */
enum class Property { None, FullRank, Orthogonal, SPD };

inline const std::vector<Property> all_properties{
    Property::None, Property::FullRank, Property::Orthogonal, Property::SPD};

std::ostream& operator<<(std::ostream& os, const Property& property);

/**
 * @brief Deduces the property resulting from multiplying two properties.
 *
 * @param p1 Property of the first operand
 * @param p2 Property of the second operand
 * @return Property of the product
 */
Property operator*(const Property p1, const Property p2);

/* ===============================================================
   ======================= TRANSPOSITION =========================
   =============================================================== */
enum class Trans { N, Y };

inline const std::vector<Trans> all_trans{Trans::N, Trans::Y};

std::ostream& operator<<(std::ostream& os, const Trans& trans);

/* ===============================================================
   ========================= INVERSION ===========================
   =============================================================== */
enum class Inversion { N, Y };

inline const std::vector<Inversion> all_inv{Inversion::N, Inversion::Y};

std::ostream& operator<<(std::ostream& os, const Inversion& inv);

/* ===============================================================
   ========================== FEATURES ===========================
   =============================================================== */
class Features {
 public:
  /* Data Members */
  Structure structure = Structure::Dense;
  Property property = Property::None;
  Trans trans = Trans::N;
  Inversion inversion = Inversion::N;

 public:
  /**
   * @brief Default Constructor
   */
  Features() = default;

  /**
   * @brief Parametrized Constructor
   *
   * @param structure Features::structure
   * @param property  Features::property
   * @param trans     Features::trans
   * @param inversion Features::inversion
   */
  Features(const Structure structure, const Property property,
           const Trans trans, const Inversion inversion)
      : structure{structure},
        property{property},
        trans{trans},
        inversion{inversion} {}

  /**
   * @brief Copy Constructor
   *
   * @param other Features object
   */
  Features(const Features& other) = default;

  /**
   * @brief Copy Assignment Constructor
   *
   * @param rhs Feature object
   */
  Features& operator=(const Features& rhs) = default;

  /**
   * @brief Equality operator
   *
   * @param rhs     Feature object
   * @return true if every data member holds the same value
   */
  bool operator==(const Features& rhs) const;

  void flipUplo();

  void T();

  void inv();

  void simplify();

  bool isDense() const;

  bool isSymmetric() const;

  bool isSymmetricLower() const;

  bool isTriangular() const;

  bool isLower() const;

  bool isUnit() const;

  bool isDiagonal() const;

  bool isSPD() const;

  bool isOrthogonal() const;

  bool isInvertible() const;

  bool isTransposed() const;

  bool isInverted() const;

  bool isLegal() const;

  bool isIdentity() const;

  friend std::ostream& operator<<(std::ostream& os, const Features& features);

  friend std::string to_string(const Features& features);

  friend std::string to_qualified_string(const Features& features);

  friend class std::hash<Features>;
};

}  // namespace cg

template <class T>
inline void hash_combine(std::size_t& seed, const T& v) noexcept {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

template <>
struct std::hash<cg::Features> {
  std::size_t operator()(const cg::Features& features) const noexcept {
    std::size_t seed = 0;
    hash_combine(seed, features.structure);
    hash_combine(seed, features.property);
    hash_combine(seed, features.trans);
    hash_combine(seed, features.inversion);

    return seed;
  }
};

#endif