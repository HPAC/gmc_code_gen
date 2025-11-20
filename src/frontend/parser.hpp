#ifndef PARSER_H
#define PARSER_H

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../definitions.hpp"
#include "../features.hpp"
#include "../matrix.hpp"
#include "lexer.hpp"
#include "token_codes.hpp"

namespace cg::fe {

struct DeclarationNode {
  std::string ID{""};
  std::string structure{""};
  std::string property{""};

  DeclarationNode() {}

  DeclarationNode(const std::string& id) : ID{id} {}

  DeclarationNode(const std::string& id, const std::string& structure,
                  const std::string& property)
      : ID{id}, structure{structure}, property{property} {}
};

// this can serve as the symbol table
struct DeclarationsNode {
  std::vector<DeclarationNode*> declarations;

  void add(DeclarationNode* declaration) {
    declarations.push_back(declaration);
  }
};

struct OperandNode {
  std::string ID{""};
  std::string unit_op{""};
  OperandNode* child{nullptr};

  ~OperandNode() {
    if (child) delete child;
  }
};

struct ExpressionNode {
  std::vector<OperandNode*> operands;

  void add(OperandNode* operand) { operands.push_back(operand); }
};

struct ProgramNode {
  DeclarationsNode* declarations{nullptr};
  ExpressionNode* expression{nullptr};

  ProgramNode() : declarations{nullptr}, expression{nullptr} {}

  ~ProgramNode() {
    delete declarations;
    delete expression;
  }
};

class Parser {
 private:
  std::unordered_map<std::string, cg::Features> symbolTable;
  Lexer lexer;
  ProgramNode* AST;

 public:
  Parser(const std::string& fname);

  ~Parser();  // delete AST

  MatrixChain walker();

 private:
  bool accept(int token);

  bool expect(int token);

  std::string unitary_operator();

  OperandNode* operand();

  ExpressionNode* expression();

  DeclarationNode* declaration();

  DeclarationsNode* declarations();

  ProgramNode* program();

  std::string chaseOperand(bool& transposed, bool& inversed,
                           const OperandNode* node);

  Structure stringToStructure(const std::string& str);

  Property stringToProperty(const std::string& str);
};

void printDeclMatrix(std::ostream& os, const Matrix& matrix);

void printMatrixInExpr(std::ostream& os, const Matrix& matrix);

void printChain(std::ostream& os, const MatrixChain& chain);

}  // namespace cg::fe

#endif