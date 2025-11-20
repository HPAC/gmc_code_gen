#include "model_common.hpp"

#include <cstdint>

namespace cg {

namespace mdl {

uint8_t countSetBits(uint8_t n) {
  // could also use: return __builtin_popcount(n) with GCC.
  uint8_t count = 0;
  while (n != 0) {
    count += static_cast<uint8_t>(n & 1);
    n >>= 1;
  }
  return count;
}

void setBit(uint8_t& n, uint8_t pos) { n |= (0b1 << pos); }

void setBitLR(uint8_t& n) { setBit(n, 4); }

void setBitUploA(uint8_t& n) { setBit(n, 3); }

void setBitUploB(uint8_t& n) { setBit(n, 2); }

void setBitTransA(uint8_t& n) { setBit(n, 1); }

void setBitTransB(uint8_t& n) { setBit(n, 0); }

bool checkBit(const uint8_t n, uint8_t pos) {
  return static_cast<bool>(n & (1 << pos));
}

void flipBit(uint8_t& n, uint8_t pos) { n ^= (0b1 << pos); }

uint8_t getKeyFromOptions(uint8_t value, uint8_t options) {
  uint8_t key = 0U;

  // there are only 5 different options (bits that can be used).
  for (uint8_t idx_option = 0; idx_option < 5; idx_option++) {
    if (options & (1 << idx_option))  // mask out the bit in idx_option.
      key |= (value & (1 << idx_option));
    else
      value <<= 1;
  }

  return key;
}

}  // namespace mdl

}  // namespace cg
