#ifndef MODEL_COMMON_H
#define MODEL_COMMON_H

#include <cstdint>

namespace cg {

namespace mdl {

/**
 * uint8_t contains information about an option in the 5 LSB.
 * - [4]-1: side (0:left; 1:right)
 * - [3]-8: uploA (0:upper; 1:lower)
 * - [2]-4: uploB (0:upper; 1:lower)
 * - [1]-2: transA (0:no; 1:trans)
 * - [0]-1: transB (0:no; 1:trans)
 */

uint8_t countSetBits(uint8_t n);

void setBit(uint8_t& n, uint8_t pos);

void setBitLR(uint8_t& n);

void setBitUploA(uint8_t& n);

void setBitUploB(uint8_t& n);

void setBitTransA(uint8_t& n);

void setBitTransB(uint8_t& n);

bool checkBit(const uint8_t n, uint8_t pos);

void flipBit(uint8_t& n, uint8_t pos);

uint8_t getKeyFromOptions(uint8_t value, uint8_t options);

// unsigned getModelID(const uint8_t& options, const uint8_t& call_info);

}  // namespace mdl

}  // namespace cg

#endif