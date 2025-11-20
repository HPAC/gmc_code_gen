#include <iostream>
#include <sstream>
#include <string>

#include "../src/frontend/parser.hpp"
#include "../src/generator.hpp"

using namespace cg;

using metricFunc = std::function<double(const std::vector<double>&)>;

std::string helpString() {
  std::ostringstream oss;
  oss << "<fname_input>: Path to the file with the GMC\n"
      << "<K>: Maximum number of variants to generate\n"
      << "<N_t>: Number of instances to use"
      << "<flops/models>: Whether to use FLOPs or performance models for "
         "expansion\n"
      << "<id_F>: objective function to use in expansion. 0 - maximum penalty. "
         "1 - avg penalty\n";
  return oss.str();
}

int main(int argc, char** argv) {
  std::string fname_input;
  unsigned K;    // maximum number of variants to generate
  unsigned N_t;  // number of instances to use in expansion
  std::string flops_or_models;
  unsigned id_F;  // cost function to use while expanding

  if (argc < 6) {
    std::cerr
        << "Error. Arguments: <fname_input> <K> <N_t> <flops/models> <id_F>\n";
    std::cerr << helpString();
    exit(-1);
  }
  fname_input = argv[1];
  K = std::stoi(argv[2]);
  N_t = std::stoi(argv[3]);
  flops_or_models = argv[4];
  id_F = std::stoi(argv[5]);

  fe::Parser parser(fname_input);
  MatrixChain chain = parser.walker();
  Generator generator(chain);
  generator.generate(K, N_t, flops_or_models, id_F);
}