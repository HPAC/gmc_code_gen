#include "dtrtrmm.cpp"

#include <omp.h>
#include <openblas/cblas.h>

#include <chrono>
#include <iostream>

double* generateUpperTriangular(int M) {
  double* matrix = (double*)malloc(M * M * sizeof(double));
  for (unsigned j = 0; j < M; ++j) {
    for (unsigned i = 0; i <= j; ++i) matrix[j * M + i] = drand48();
    for (unsigned i = j + 1; i < M; ++i) matrix[j * M + i] = 0.0;
  }
  return matrix;
}

double* zeros(int M) {
  double* matrix = (double*)malloc(M * M * sizeof(double));
  for (unsigned j = 0; j < M; ++j) {
    for (unsigned i = 0; i < M; ++i) matrix[j * M + i] = 0.0;
  }
  return matrix;
}

double norm_diff(int M, double* A, double* B) {
  double acc = 0;
  for (unsigned j = 0; j < M; ++j) {
    for (unsigned i = 0; i < M; ++i) {
      // std::cout << i << std::endl;
      acc += (A[j * M + i] - B[j * M + i]) * (A[j * M + i] - B[j * M + i]);
    }
  }
  return acc;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cout << "Usage: ./x size n_iter\n";
    exit(-1);
  }

  int M = std::atoi(argv[1]);
  int n_iter = std::atoi(argv[2]);
  double* A = generateUpperTriangular(M);
  double* B = generateUpperTriangular(M);
  double* C = zeros(M);
  openblas_set_num_threads(1);  // not working for some reason

  auto start = std::chrono::high_resolution_clock::now();
  for (unsigned it = 0; it < n_iter; it++) {
    dtrtrmm('U', 'U', 'N', 'N', 'N', 'N', M, 1.0, A, M, B, M, C, M);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::cout << "Average time TRTRMM: "
            << std::chrono::duration<double>(end - start).count() / n_iter
            << std::endl;

  double flops_trmm = static_cast<double>(M) * static_cast<double>(M) *
                      static_cast<double>(M) / 1e9;
  start = std::chrono::high_resolution_clock::now();

  for (unsigned it = 0; it < n_iter; it++) {
    cblas_dtrmm(CblasColMajor, CblasLeft, CblasUpper, CblasNoTrans,
                CblasNonUnit, M, M, 1.0, A, M, B, M);
  }
  end = std::chrono::high_resolution_clock::now();
  double avg_time_trmm =
      std::chrono::duration<double>(end - start).count() / n_iter;
  std::cout << "Average time TRMM: " << avg_time_trmm << std::endl;
  std::cout << "Performance TRMM: " << flops_trmm / avg_time_trmm
            << " GFlops/s\n";  // around 120 GFlops with 6 cores on MBP

  // This is only valid when n_iter is 1, since B is overwritten in each TRMM
  if (n_iter == 1) {
    std::cout << "Norm diff: " << norm_diff(M, C, B) << std::endl;
  }

  start = std::chrono::high_resolution_clock::now();
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, M, M, 1.0, A, M, B,
              M, 0.0, C, M);
  end = std::chrono::high_resolution_clock::now();
  double time_gemm = std::chrono::duration<double>(end - start).count();
  std::cout << "Time GEMM: " << time_gemm << std::endl;
  std::cout << "Performance GEMM: " << flops_trmm * 2.0 / time_gemm
            << std::endl;

  free(A);
  free(B);
  free(C);
}
