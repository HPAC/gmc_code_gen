#ifndef LEXER_H
#define LEXER_H

#include <fstream>
#include <iostream>
#include <string>

#include "token_codes.hpp"

namespace cg::fe {
/* Character classes */
#define LETTER 0
#define DIGIT 1
#define UNDERSCORE 2
#define UNKNOWN 99

class Lexer {
 private:
  std::ifstream ifile;  // input stream to read and process
  int charClass{-1};    // completely internal information
  char nextChar{' '};   // completely internal information

 public:
  std::string lexeme;  // carries information to pass to the parser
  int nextToken{-1};   // carries information to pass to the parser

  Lexer(const std::string& fname);

  ~Lexer();

  /**
   * @brief Adds a character to the lexeme. Checks whether the lexeme is too
   * long
   */
  void addChar();

  /**
   * @brief Gets the next character from the input stream and determines its
   * character class
   */
  void getChar();

  /**
   * @brief Checks whether a lexeme is a reserved word
   *
   * @return true   if the lexeme is a reserved word
   * @return false  otherwise
   */
  bool lookupReserved();

  /**
   * @brief Checks whether a character is one of: *<>(),;
   *
   * If the character is none of these, then it's considered EOF
   */
  void lookup();

  /**
   * @brief Jumps empty characters (whitespaces, linejumps, etc) to get the next
   * character and place it in nextChar
   */
  void getNonBlank();

  /**
   * @brief Checks whether the lexeme is one of the structures reserved words
   *
   * @return true   if the lexeme denotes a structure
   * @return false  otherwise
   */
  bool isStructureRW();  //  whether a lexeme is one of the structures RWs

  /**
   * @brief Checks whether the lexeme is one of the properties' reserved words
   *
   * @return true   if the lexeme denotes a property
   * @return false  otherwise
   */
  bool isPropertyRW();  // whether a lexeme is one of the properties RWs

  /**
   * @brief Returns the next token type and updates the lexeme
   *
   * @return int - the token code
   */
  int lex();
};

}  // namespace cg::fe

#endif