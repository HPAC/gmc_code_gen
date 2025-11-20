#include "lexer.hpp"

#include <fstream>
#include <iostream>
#include <string>

namespace cg::fe {

Lexer::Lexer(const std::string& fname) {
  ifile.open(fname);
  if (ifile.fail()) {
    std::cout << "Error opening file\n";
    exit(-1);
  } else
    lex();
}

Lexer::~Lexer() { ifile.close(); }

void Lexer::addChar() {
  if (lexeme.size() <= 99) {
    lexeme += nextChar;
  } else {
    std::cout << "Error - lexeme too long\n";
  }
}

void Lexer::getChar() {
  ifile.get(nextChar);
  if (!ifile.eof()) {
    if (std::isalpha(nextChar))
      charClass = LETTER;
    else if (std::isdigit(nextChar))
      charClass = DIGIT;
    else if (nextChar == '_')
      charClass = UNDERSCORE;
    else
      charClass = UNKNOWN;
  } else
    charClass = EOF;
}

bool Lexer::lookupReserved() {
  bool reserved = false;
  if (lexeme == "Matrix") {
    reserved = true;
    nextToken = TK_MATRIX;
  } else if (isStructureRW()) {
    reserved = true;
    nextToken = STRUCTURE;
  } else if (isPropertyRW()) {
    reserved = true;
    nextToken = PROPERTY;
  } else if (lexeme == "trans") {
    reserved = true;
    nextToken = TRANS_OP;
  } else if (lexeme == "inv") {
    reserved = true;
    nextToken = INV_OP;
  }
  return reserved;
}

void Lexer::lookup() {
  switch (nextChar) {
    case '*':
      addChar();
      nextToken = MULT_OP;
      break;

    case '<':
      addChar();
      nextToken = LEFT_ANGLE;
      break;

    case '>':
      addChar();
      nextToken = RIGHT_ANGLE;
      break;

    case '(':
      addChar();
      nextToken = LEFT_PARENTH;
      break;

    case ')':
      addChar();
      nextToken = RIGHT_PARENTH;
      break;

    case ',':
      addChar();
      nextToken = COMMA;
      break;

    case ';':
      addChar();
      nextToken = SEMICOLON;
      break;

    default:
      addChar();
      nextToken = EOF;
      break;
  }
}

void Lexer::getNonBlank() {
  while (std::isspace(nextChar) and !ifile.eof()) getChar();
}

bool Lexer::isStructureRW() {
  if (lexeme == "Dense" || lexeme == "SymmetricLower" ||
      lexeme == "SymmetricUpper" || lexeme == "Lower" ||
      lexeme == "UnitLower" || lexeme == "Upper" || lexeme == "UnitUpper" ||
      lexeme == "Diagonal")
    return true;
  else
    return false;
}

bool Lexer::isPropertyRW() {
  if (lexeme == "None" || lexeme == "FullRank" || lexeme == "SPD" ||
      lexeme == "Orthogonal")
    return true;
  else
    return false;
}

int Lexer::lex() {
  lexeme.clear();
  getNonBlank();
  switch (charClass) {
    case LETTER:
      addChar();
      getChar();
      while (charClass == LETTER || charClass == DIGIT ||
             charClass == UNDERSCORE) {
        addChar();
        getChar();
      }
      // if it's a reserved word, mark it as such and fix the token value
      // if it's not a reserved word, then it's an identifier
      if (!lookupReserved()) {
        nextToken = IDENT;
      }
      break;

    case UNKNOWN:
      lookup();
      getChar();
      break;

    case EOF:
      nextToken = EOF;
      lexeme = "EOF";
      break;

    default:
      break;
  }
  return nextToken;
}

}  // namespace cg::fe