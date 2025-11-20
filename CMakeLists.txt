add_library(cblas_extended SHARED
            # cblas Mult kernels
            cblas_ddidimm.cpp
            cblas_ddimm.cpp
            cblas_ddisymm.cpp
            cblas_dditrmm.cpp
            cblas_dsysymm.cpp
            cblas_dtrsymm.cpp
            cblas_dtrtrmm.cpp

            # cblas system solver kernels
            cblas_ddidisv.cpp
            cblas_ddisv.cpp
            cblas_ddisysv.cpp
            cblas_dditrsv.cpp
            cblas_dgedisv.cpp
            cblas_dgegesv.cpp
            cblas_dgesysv.cpp
            cblas_dgetrsv.cpp
            cblas_dpodisv.cpp
            cblas_dpogesv.cpp
            cblas_dposysv.cpp
            cblas_dpotrsv.cpp
            cblas_dsydisv.cpp
            cblas_dsygesv.cpp
            cblas_dsysysv.cpp
            cblas_dsytrsv.cpp
            cblas_dtrdisv.cpp
            cblas_dtrsysv.cpp
            cblas_dtrtrsv.cpp

            # Mult kernels
            base/ddidimm.cpp
            base/ddimm.cpp
            base/ddisymm.cpp
            base/dditrmm.cpp
            base/dsysymm.cpp
            base/dtrsymm.cpp
            base/dtrtrmm.cpp

            # System solver kernels
            base/ddidisv.cpp
            base/ddisv.cpp
            base/ddisysv.cpp
            base/dditrsv.cpp
            base/dgedisv.cpp
            base/dgegesv.cpp
            base/dgesysv.cpp
            base/dgetrsv.cpp
            base/dpodisv.cpp
            base/dpogesv.cpp
            base/dposysv.cpp
            base/dpotrsv.cpp
            base/dsydisv.cpp
            base/dsygesv.cpp
            base/dsysysv.cpp
            base/dsytrsv.cpp
            base/dtrdisv.cpp
            base/dtrsysv.cpp
            base/dtrtrsv.cpp
            
            # Aux functions
            base/sy2full.cpp
            base/dge_trans.cpp
            base/dlacpy.cpp
            base/dlaswp2.cpp
            base/dlaswp3.cpp
            base/dlaswp4.cpp
            base/dbdigesv.cpp
           )

if (OpenMP_CXX_FOUND)
    target_link_libraries(cblas_extended PUBLIC ${BLAS_LIBRARIES} PRIVATE OpenMP::OpenMP_CXX)
elseif()
    target_link_libraries(cblas_extended PUBLIC ${BLAS_LIBRARIES})
endif()

target_compile_options(cblas_extended PUBLIC ${comp_flags})


