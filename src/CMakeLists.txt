find_package(fmt REQUIRED)

add_subdirectory(models)

add_library(codeGen SHARED
            macros.hpp
            algorithm.cpp
            analyzer.cpp
            features.cpp
            generator.cpp
            kernel.cpp
            matcher.cpp
            matrix.cpp
            node.cpp
            expr_generator.cpp
            instance_generator.cpp
            permutation.cpp
            variant.cpp
            experiment_util.cpp
            # Kernel representations
            kernels/didimm.cpp
            kernels/didisv.cpp
            kernels/dimm.cpp
            kernels/disv.cpp
            kernels/disymm.cpp
            kernels/disysv.cpp
            kernels/ditrmm.cpp
            kernels/ditrsv.cpp
            kernels/gedisv.cpp
            kernels/gegesv.cpp
            kernels/gemm.cpp
            kernels/gesysv.cpp
            kernels/getrsv.cpp
            kernels/podisv.cpp
            kernels/pogesv.cpp
            kernels/posysv.cpp
            kernels/potrsv.cpp
            kernels/sydisv.cpp
            kernels/sygesv.cpp
            kernels/symm.cpp
            kernels/sysymm.cpp
            kernels/sysysv.cpp
            kernels/sytrsv.cpp
            kernels/trdisv.cpp
            kernels/trmm.cpp
            kernels/trsm.cpp
            kernels/trsymm.cpp
            kernels/trsysv.cpp
            kernels/trtrmm.cpp
            kernels/trtrsv.cpp
            #  Frontend
            frontend/lexer.cpp
            frontend/parser.cpp
            #  Utils
            utils/dMatrix.cpp
            utils/matrix_generator.cpp
            utils/common.cpp
            )
target_link_libraries(codeGen PUBLIC 
                                cblas_extended 
                                ${BLAS_LIBRARIES} 
                                models 
                              PRIVATE 
                                fmt::fmt-header-only)
                                
target_compile_options(codeGen PUBLIC 
                                 ${comp_flags})
