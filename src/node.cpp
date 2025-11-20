#include "node.hpp"

#include <string>

#include "kernel.hpp"
#include "matcher.hpp"
#include "matrix.hpp"
#include "utils/dMatrix.hpp"
#include "variant.hpp"

namespace cg {

Node::Node(const int8_t left, const int8_t right,
           const Matcher* const matcher_ptr, const std::string& temp_name,
           std::vector<Node>& tree)
    : left{left}, right{right} {
  matrix.setName(temp_name);
  matrix.setModifiable(true);

  rewrite(tree);  // rewrite the expression so there's a match
  kernel_ptr = matcher_ptr->getKernel({tree[left].matrix, tree[right].matrix});

  if (kernel_ptr->tweakTransposition(tree[left].matrix, tree[right].matrix)) {
    transpose(tree);
  }
  propagateFeatures(tree);  // get the structure and property of the product
  propagateDimensions(tree);
  kernel_ptr->deduceName(tree[left].matrix, tree[right].matrix, matrix);
}

Node::Node(const Node& other) {
  transposed = other.transposed;
  left = other.left;
  right = other.right;
  top = other.top;
  kernel_ptr = other.kernel_ptr;
  matrix_ptr = std::make_unique<dMatrix>(*other.matrix_ptr);
}

Node::Node(Node&& other) {
  transposed = other.transposed;

  left = other.left;
  other.left = -1;

  right = other.right;
  other.right = -1;

  top = other.top;
  other.top = -1;

  kernel_ptr = other.kernel_ptr;
  other.kernel_ptr = nullptr;

  matrix_ptr = std::move(other.matrix_ptr);
}

Node& Node::operator=(const Node& rhs) {
  transposed = rhs.transposed;
  left = rhs.left;
  right = rhs.right;
  top = rhs.top;
  kernel_ptr = rhs.kernel_ptr;
  matrix_ptr = std::make_unique<dMatrix>(*rhs.matrix_ptr);
  return *this;
}

Node& Node::operator=(Node&& rhs) {
  transposed = rhs.transposed;

  left = rhs.left;
  rhs.left = -1;

  right = rhs.right;
  rhs.right = -1;

  top = rhs.top;
  rhs.top = -1;

  kernel_ptr = rhs.kernel_ptr;
  rhs.kernel_ptr = nullptr;

  matrix_ptr = std::move(rhs.matrix_ptr);
  return *this;
}

std::string Node::generateCode(std::vector<Node>& tree) const {
  return kernel_ptr->generateCode(tree[left].matrix, tree[right].matrix,
                                  matrix);
}

std::string Node::generateCleanup() const {
  return (matrix.isTransposed()) ? kernel_ptr->generateTranspose(matrix) : "";
}

std::string Node::generateCost(std::vector<Node>& tree) const {
  return kernel_ptr->generateCost(tree[left].matrix, tree[right].matrix,
                                  matrix);
}

double Node::computeFLOPs(std::vector<Node>& tree) {
  return kernel_ptr->computeFLOPs(tree[left].matrix, tree[right].matrix,
                                  matrix);
}

double Node::predictTime(std::vector<Node>& tree) {
  return kernel_ptr->predictTime(tree[left].matrix, tree[right].matrix, matrix);
}

void Node::propagateSizes(std::vector<Node>& tree) {
  matrix.nrows = (!tree[left].matrix.isTransposed()) ? tree[left].matrix.nrows
                                                     : tree[left].matrix.ncols;

  matrix.ncols = (!tree[right].matrix.isTransposed())
                     ? tree[right].matrix.ncols
                     : tree[right].matrix.nrows;
}

void Node::setLiveMatrix(const dMatrix& mat) {
  matrix_ptr = std::make_unique<dMatrix>(mat);
}

void Node::deleteMatrix() { matrix_ptr.reset(); }

std::unique_ptr<dMatrix>&& Node::getLiveMatrixPtr() {
  return std::move(matrix_ptr);
}

void Node::execute(std::vector<Node>& tree) {
  // the first value says whether a new matrix is needed to hold the result. The
  // second value says whether the matrix has to be explicitly zeroed.
  auto infoNewMatrix = kernel_ptr->needsNewMatrix();

  if (infoNewMatrix[0] and matrix_ptr == nullptr) {
    unsigned rows = (!tree[left].matrix.isTransposed())
                        ? tree[left].matrix_ptr->rows
                        : tree[left].matrix_ptr->cols;
    unsigned cols = (!tree[right].matrix.isTransposed())
                        ? tree[right].matrix_ptr->cols
                        : tree[right].matrix_ptr->rows;
    matrix_ptr = std::make_unique<dMatrix>(rows, cols);
    if (infoNewMatrix[1]) matrix_ptr->zero();
  } else {
    // create an empty dMatrix that will be overwritten (move assignment)
    matrix_ptr = std::make_unique<dMatrix>();
  }

  kernel_ptr->execute(tree[left].matrix, tree[right].matrix,
                      *tree[left].matrix_ptr, *tree[right].matrix_ptr,
                      *matrix_ptr);
}

void Node::rewrite(std::vector<Node>& tree) {
  Matrix& operand_L = tree[left].matrix;
  Matrix& operand_R = tree[right].matrix;

  // if both operands are inverted, invert the whole pair and propagate the inv.
  if (operand_L.isInverted() and operand_R.isInverted()) {
    invert(tree);
  } else if (operand_L.isInverted() or operand_R.isInverted()) {
    // one is inverted, the other is not.
    if (chooseInvert(operand_L, operand_R)) {
      invert(tree);
    }
  }
}

void Node::invert(std::vector<Node>& tree) {
  matrix.inv();
  tree[left].matrix.inv();
  tree[right].matrix.inv();
  std::swap(left, right);
}

/* table for propagation of inverse.
   |  1  2  3  4  5  6  7  8 | RHS
  -|-------------------------
  1|  N  N  N  Y  Y  Y  Y  Y
  2|  N  N  N  Y  Y  Y  Y  Y
  3|  N  N  N  Y  Y  Y  Y  Y
  4|  N  N  N  N  N  N  N  Y
  5|  N  N  N  N  N  N  N  Y
  6|  N  N  N  N  N  N  N  Y
  7|  N  N  N  N  N  N  N  Y
  8|  N  N  N  N  N  N  N  N
---|-------------------------
LHS
*/

bool Node::chooseInvert(const Matrix& A, const Matrix& B) {
  const Matrix& lhs = A.isInverted() ? A : B;
  const Matrix& rhs = A.isInverted() ? B : A;

  // if rhs is invertible, we check whether it's beneficial to invert the pair.
  if (rhs.isInvertible()) {
    if (lhs.structure >= rhs.structure)
      return false;
    else if (lhs.isTriangular() and rhs.isTriangular())
      return false;
    else if (lhs.isDense() and rhs.isSymmetric())
      return false;
    else if (lhs.isSymmetric() and rhs.isSymmetric())
      return false;
    else
      return true;
  } else {
    return false;
  }
}

void Node::transpose(std::vector<Node>& tree) {
  matrix.T();
  tree[left].matrix.T();
  tree[right].matrix.T();
  std::swap(left, right);
}

void Node::propagateDimensions(std::vector<Node>& tree) {
  if (tree[left].matrix.isTransposed())
    matrix.setRowName(tree[left].matrix.getColName());
  else
    matrix.setRowName(tree[left].matrix.getRowName());

  if (tree[right].matrix.isTransposed())
    matrix.setColName(tree[right].matrix.getRowName());
  else
    matrix.setColName(tree[right].matrix.getColName());
}

void Node::propagateFeatures(std::vector<Node>& tree) {
  matrix.structure = deduceStructure(tree);
  matrix.property = deduceProperty(tree);
}

Structure Node::deduceStructure(std::vector<Node>& tree) {
  auto structure_L = tree[left].matrix.structure;
  if (tree[left].matrix.isTransposed())
    structure_L = transposeStructure(structure_L);

  auto structure_R = tree[right].matrix.structure;
  if (tree[right].matrix.isTransposed())
    structure_R = transposeStructure(structure_R);

  return structure_L * structure_R;
}

Property Node::deduceProperty(std::vector<Node>& tree) {
  return tree[left].matrix.property * tree[right].matrix.property;
}

}  // namespace cg
