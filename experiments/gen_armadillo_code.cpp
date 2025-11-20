#include <fmt/core.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "../src/experiment_util.hpp"

std::vector<unsigned> readShapesFromFile(const std::string& fname,
                                         const unsigned n_shapes);

std::string armaSignature(const unsigned chain_N, const unsigned id);

std::string ID2ArmaCode(const unsigned chain_N, const unsigned shapeID);

std::string digit2Arma(const unsigned digit, const unsigned id_matrix);

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Error. Arguments: ./prog <expr_file> <n_shapes> <dir_out>\n";
    exit(-1);
  }
  std::string expr_file = argv[1];
  unsigned n_shapes = std::stoi(argv[2]);
  std::string dir_out = argv[3];

  std::string fname_header = dir_out + "arma_code.hpp";
  std::string fname_unit = dir_out + "arma_code.cpp";

  std::ofstream code_header, code_unit;
  std::ostringstream vector_header;

  // Open header file
  code_header.open(fname_header);
  if (code_header.fail()) {
    std::cerr << "Error: Cannot open file " << fname_header << '\n';
    exit(-1);
  }
  // Preamble header file
  code_header << "#ifndef CG_ARMA_CODE\n#define CG_ARMA_CODE\n\n";
  code_header << "#include <armadillo>\n#include <vector>\n\n";
  code_header
      << "using expr_ptr = arma::mat (*)(arma::mat&, arma::mat&, "
         "arma::mat&, arma::mat&, arma::mat&, arma::mat&, arma::mat&);\n\n";

  // Declaration of vector with function pointers (to be in header file)
  vector_header << "inline const std::vector<expr_ptr> VEC_EXPR {";

  // Open source file
  code_unit.open(fname_unit);
  if (code_unit.fail()) {
    std::cerr << "Error: Cannot open file " << fname_unit << '\n';
    exit(-1);
  }
  // Preamble source file
  code_unit << "#include <armadillo>\n\n";
  code_unit << "#include \"arma_code.hpp\"\n\n";

  // Read the first n_shapes; put them in a vector
  auto vec_shapes = readShapesFromFile(expr_file, n_shapes);

  for (unsigned id = 0U; id < n_shapes; id++) {
    std::string signature = armaSignature(7U, id);
    unsigned shapeID = vec_shapes[id];

    // Write signature to header
    code_header << signature << ";\n";

    // Handle code for vector in header
    if (id != 0) {
      vector_header << ", ";
    }
    vector_header << "fn" << id;

    // Write code in source
    code_unit << signature << "{\n";
    code_unit << ID2ArmaCode(7U, shapeID);
    code_unit << "}\n\n";
  }

  vector_header << "};";
  code_header << vector_header.str() << "\n\n#endif";

  code_header.close();
  code_unit.close();
}

std::vector<unsigned> readShapesFromFile(const std::string& fname,
                                         const unsigned n_shapes) {
  std::ifstream is;
  is.open(fname);
  if (is.fail()) {
    std::cerr << "Error: Cannot open file " << fname << '\n';
    exit(-1);
  }
  std::vector<unsigned> vec_shapeID;
  vec_shapeID.reserve(n_shapes);
  unsigned shapeID;

  for (unsigned i = 0U; i < n_shapes; i++) {
    is >> shapeID;
    vec_shapeID.emplace_back(shapeID);
  }
  is.close();
  return vec_shapeID;
}

std::string armaSignature(const unsigned chain_N, const unsigned id) {
  std::ostringstream oss;
  oss << "arma::mat fn" << id << "(";
  char base_var = 'A';
  for (unsigned mat_id = 0U; mat_id < chain_N; mat_id++) {
    char var = base_var + mat_id;
    oss << "arma::mat& " << (var);
    if (mat_id < chain_N - 1) {
      oss << ", ";
    }
  }
  oss << ")";
  return oss.str();
}

std::string ID2ArmaCode(const unsigned chain_N, const unsigned shapeID) {
  std::ostringstream code;
  code << "\treturn ";

  unsigned digit;
  float power_10;
  for (unsigned i = 0U; i < chain_N; i++) {
    power_10 = std::pow(10.0, static_cast<float>(chain_N - 1 - i));
    digit = shapeID / static_cast<unsigned>(power_10);
    digit %= 10U;
    if (i == 0U) {
      code << digit2Arma(digit, i);
    } else {
      code << " * " << digit2Arma(digit, i);
    }
  }
  code << ";\n";
  return code.str();
}

std::string digit2Arma(const unsigned digit, const unsigned id_matrix) {
  std::string code{""};
  char name = 'A' + id_matrix;

  if (digit == 1U || digit == 8U || digit == 9U) {
    code = fmt::format("inv({})", name);
  } else if (digit == 3U) {
    code = fmt::format("inv_sympd({})", name);
  } else {
    code = fmt::format("{}", name);
  }

  return code;
}