#ifndef MACROS_CBLAS_H
#define MACROS_CBLAS_H

/* CBLAS ARGUMENTS MACROS */
#define CG_COL_MAJOR "CblasColMajor"
#define CG_ROW_MAJOR "CblasRowMajor"

#define CG_TRANS "CblasTrans"
#define CG_NO_TRANS "CblasNoTrans"

#define CG_LEFT "CblasLeft"
#define CG_RIGHT "CblasRight"

#define CG_LOWER "CblasLower"
#define CG_UPPER "CblasUpper"

#define CG_UNIT "CblasUnit"
#define CG_NO_UNIT "CblasNonUnit"

/* MULTIPLICATION FUNCTION NAMES */
#define CBLAS_DGEMM "cblas_dgemm"
#define CBLAS_DSYMM "cblas_dsymm"
#define CBLAS_DTRMM "cblas_dtrmm"
#define CBLAS_DDIMM "cblas_ddimm"
#define CBLAS_DSYSYMM "cblas_dsysymm"
#define CBLAS_DTRSYMM "cblas_dtrsymm"
#define CBLAS_DDISYMM "cblas_ddisymm"
#define CBLAS_DTRTRMM "cblas_dtrtrmm"
#define CBLAS_DDITRMM "cblas_dditrmm"
#define CBLAS_DDIDIMM "cblas_ddidimm"

/* SYSTEM SOLVER FUNCTION NAMES */
#define CBLAS_DDISV "cblas_ddisv"
#define CBLAS_DTRSM "cblas_dtrsm"
#define CBLAS_DDIDISV "cblas_ddidisv"
#define CBLAS_DDITRSV "cblas_dditrsv"
#define CBLAS_DDISYSV "cblas_ddisysv"
#define CBLAS_DGEGESV "cblas_dgegesv"
#define CBLAS_DGESYSV "cblas_dgesysv"
#define CBLAS_DGETRSV "cblas_dgetrsv"
#define CBLAS_DGEDISV "cblas_dgedisv"
#define CBLAS_DTRSYSV "cblas_dtrsysv"
#define CBLAS_DTRTRSV "cblas_dtrtrsv"
#define CBLAS_DTRDISV "cblas_dtrdisv"
#define CBLAS_DSYGESV "cblas_dsygesv"
#define CBLAS_DSYSYSV "cblas_dsysysv"
#define CBLAS_DSYTRSV "cblas_dsytrsv"
#define CBLAS_DSYDISV "cblas_dsydisv"
#define CBLAS_DPOGESV "cblas_dpogesv"
#define CBLAS_DPOSYSV "cblas_dposysv"
#define CBLAS_DPOTRSV "cblas_dpotrsv"
#define CBLAS_DPODISV "cblas_dpodisv"

/* MULT KERNEL SHORT NAMES */
#define CG_GEMM "GEMM"
#define CG_SYMM "SYMM"
#define CG_TRMM "TRMM"
#define CG_DIMM "DIMM"
#define CG_SYSYMM "SYSYMM"
#define CG_TRSYMM "TRSYMM"
#define CG_DISYMM "DISYMM"
#define CG_TRTRMM "TRTRMM"
#define CG_DITRMM "DITRMM"
#define CG_DIDIMM "DIDIMM"

/* SYSTEM SOLVER KERNEL SHORT NAMES */
#define CG_TRSM "TRSM"
#define CG_DISV "DISV"
#define CG_DIDISV "DIDISV"
#define CG_DITRSV "DITRSV"
#define CG_DISYSV "DISYSV"
#define CG_GEGESV "GEGESV"
#define CG_GESYSV "GESYSV"
#define CG_GETRSV "GETRSV"
#define CG_GEDISV "GEDISV"
#define CG_TRSYSV "TRSYSV"
#define CG_TRTRSV "TRTRSV"
#define CG_TRDISV "TRDISV"
#define CG_SYGESV "SYGESV"
#define CG_SYSYSV "SYSYSV"
#define CG_SYTRSV "SYTRSV"
#define CG_SYDISV "SYDISV"
#define CG_POGESV "POGESV"
#define CG_POSYSV "POSYSV"
#define CG_POTRSV "POTRSV"
#define CG_PODISV "PODISV"

/* SETTINGS */
#if 1
#define LAYOUT CG_COL_MAJOR
#else
#define LAYOUT CG_ROW_MAJOR
#endif

#endif