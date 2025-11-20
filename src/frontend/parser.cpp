#include "parser.hpp"

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "lexer.hpp"

namespace cg::fe {

Parser::Parser(const std::string& fname) : lexer{fname} {
  AST = program();  // start the parsing
}

Parser::~Parser() { delete AST; }

MatrixChain Parser::walker() {
  MatrixChain chain{};
  for (auto& operand : AST->expression->operands) {
    bool transposed = false;
    bool inversed = false;
    std::string name = chaseOperand(transposed, inversed, operand);
    if (symbolTable.find(name) == symbolTable.end()) {
      std::cout << name << " is not declared. Not added to the chain.\n";
    } else {
      auto features_operand = symbolTable[name];
      if (transposed) features_operand.trans = Trans::Y;
      if (inversed) features_operand.inversion = Inversion::Y;
      chain.emplace_back(name, features_operand);
    }
  }
  return chain;
}

bool Parser::accept(int token) {
  if (lexer.nextToken == token) {
    lexer.lex();
    return true;
  } else
    return false;
}

bool Parser::expect(int token) {
  if (accept(token))
    return true;
  else {
    std::cout << "\033[0;31mERROR: unexpected symbol\033[0m" << token << "\n";
    return false;
  }
}

std::string Parser::unitary_operator() {
  std::string result{};
  if (accept(TRANS_OP)) {
    result = "trans";
  } else if (accept(INV_OP)) {
    result = "inv";
  }
  return result;
}

OperandNode* Parser::operand() {
  OperandNode* operand_node = new OperandNode;
  if (lexer.nextToken == IDENT) {
    operand_node->ID = lexer.lexeme;
    // consult the symbol table to check that the operand is here
    accept(IDENT);
  } else {
    operand_node->unit_op = unitary_operator();
    expect(LEFT_PARENTH);
    operand_node->child = operand();
    expect(RIGHT_PARENTH);
  }

  return operand_node;
}

ExpressionNode* Parser::expression() {
  ExpressionNode* expression_node = new ExpressionNode;
  do {
    expression_node->add(operand());
  } while (accept(MULT_OP));
  expect(SEMICOLON);

  return expression_node;
}

DeclarationNode* Parser::declaration() {
  DeclarationNode* declaration_node = new DeclarationNode;
  expect(TK_MATRIX);
  if (lexer.nextToken == IDENT) {
    declaration_node->ID = lexer.lexeme;
    accept(IDENT);
  } else {
    std::cout << "\033[0;31mERROR: expected IDENT and got token "
              << lexer.nextToken << "\033[0m\n";
  }
  expect(LEFT_ANGLE);
  if (lexer.nextToken == STRUCTURE) {
    declaration_node->structure = lexer.lexeme;
    accept(STRUCTURE);
  } else {
    std::cout << "\033[0;31mERROR: expected STRUCTURE and got token "
              << lexer.nextToken << "\033[0m\n";
  }

  expect(COMMA);

  if (lexer.nextToken == PROPERTY) {
    declaration_node->property = lexer.lexeme;
    accept(PROPERTY);
  } else {
    std::cout << "\033[0;31mERROR: expected PROPERTY and got token "
              << lexer.nextToken << "\033[0m\n";
  }

  expect(RIGHT_ANGLE);
  cg::Features features(stringToStructure(declaration_node->structure),
                        stringToProperty(declaration_node->property),
                        cg::Trans::N, cg::Inversion::N);
  symbolTable.emplace(std::make_pair(declaration_node->ID, features));
  return declaration_node;
}

DeclarationsNode* Parser::declarations() {
  DeclarationsNode* declarations_node = new DeclarationsNode;
  do {
    declarations_node->add(declaration());
    expect(SEMICOLON);
  } while (lexer.nextToken == TK_MATRIX);
  // check that next is a MATRIX without consuming it
  return declarations_node;
}

ProgramNode* Parser::program() {
  ProgramNode* program_node = new ProgramNode;
  program_node->declarations = declarations();
  program_node->expression = expression();

  return program_node;
}

std::string Parser::chaseOperand(bool& transposed, bool& inversed,
                                 const OperandNode* node) {
  if (node->child == nullptr)
    return node->ID;
  else {
    if (node->unit_op == "trans")
      transposed = !transposed;
    else if (node->unit_op == "inv")
      inversed = !inversed;
    return chaseOperand(transposed, inversed, node->child);
  }
}

Structure Parser::stringToStructure(const std::string& str) {
  Structure structure{};
  if (str == "Dense")
    structure = Structure::Dense;
  else if (str == "SymmetricLower")
    structure = Structure::Symmetric_L;
  else if (str == "SymmetricUpper")
    structure = Structure::Symmetric_U;
  else if (str == "Lower")
    structure = Structure::Lower;
  else if (str == "UnitLower")
    structure = Structure::UnitLower;
  else if (str == "Upper")
    structure = Structure::Upper;
  else if (str == "UnitUpper")
    structure = Structure::UnitUpper;
  else if (str == "Diagonal")
    structure = Structure::Diagonal;
  return structure;
}

Property Parser::stringToProperty(const std::string& str) {
  Property property{};
  if (str == "None")
    property = Property::None;
  else if (str == "FullRank")
    property = Property::FullRank;
  else if (str == "SPD")
    property = Property::SPD;
  else if (str == "Orthogonal")
    property = Property::Orthogonal;
  return property;
}

void printDeclMatrix(std::ostream& os, const Matrix& matrix) {
  os << "Matrix " << matrix.name << " <" << matrix.structure << ", "
     << matrix.property << ">;";
}

void printMatrixInExpr(std::ostream& os, const Matrix& matrix) {
  if (matrix.isTransposed()) {
    os << "trans(";

    if (matrix.isInverted())
      os << "inv(" << matrix.name << ")";
    else
      os << matrix.name;

    os << ")";
  }

  else {
    if (matrix.isInverted()) {
      os << "inv(" << matrix.name << ")";
    } else {
      os << matrix.name;
    }
  }
}

void printChain(std::ostream& os, const MatrixChain& chain) {
  for (const auto& matrix : chain) {
    printDeclMatrix(os, matrix);
    os << "\n";
  }
  os << "\n";

  for (unsigned i = 0; i < chain.size(); i++) {
    printMatrixInExpr(os, chain[i]);
    if (i < chain.size() - 1)
      os << " * ";
    else
      os << ";";
  }
  os << '\n';
}

}  // namespace cg::fe
