#ifndef SETTINGS_KERNELS_H
#define SETTINGS_KERNELS_H

#include <vector>

#include "kernel.hpp"

/* Multiplications */
#include "kernels/didimm.hpp"
#include "kernels/dimm.hpp"
#include "kernels/disymm.hpp"
#include "kernels/ditrmm.hpp"
#include "kernels/gemm.hpp"
#include "kernels/symm.hpp"
#include "kernels/sysymm.hpp"
#include "kernels/trmm.hpp"
#include "kernels/trsymm.hpp"
#include "kernels/trtrmm.hpp"

/* System Solvers */
#include "kernels/didisv.hpp"
#include "kernels/disv.hpp"
#include "kernels/disysv.hpp"
#include "kernels/ditrsv.hpp"
#include "kernels/gedisv.hpp"
#include "kernels/gegesv.hpp"
#include "kernels/gesysv.hpp"
#include "kernels/getrsv.hpp"
#include "kernels/podisv.hpp"
#include "kernels/pogesv.hpp"
#include "kernels/posysv.hpp"
#include "kernels/potrsv.hpp"
#include "kernels/sydisv.hpp"
#include "kernels/sygesv.hpp"
#include "kernels/sysysv.hpp"
#include "kernels/sytrsv.hpp"
#include "kernels/trdisv.hpp"
#include "kernels/trsm.hpp"
#include "kernels/trsysv.hpp"
#include "kernels/trtrsv.hpp"

namespace cg {

/* Multiplications */
inline KernelGemm kernel_gemm(0x03);
inline KernelSymm kernel_symm(0x18);
inline KernelTrmm kernel_trmm(0x1A);
inline KernelDimm kernel_dimm(0x10);
inline KernelSysymm kernel_sysymm(0x0C);
inline KernelTrsymm kernel_trsymm(0x1E);
inline KernelDisymm kernel_disymm(0x14);
inline KernelTrtrmm kernel_trtrmm(0x0F);
inline KernelDitrmm kernel_ditrmm(0x14);
inline KernelDidimm kernel_didimm(0x00);

/* System Solvers */
inline KernelGegesv kernel_gegesv(0x12);
inline KernelGesysv kernel_gesysv(0x16);
inline KernelGetrsv kernel_getrsv(0x16);
inline KernelGedisv kernel_gedisv(0x12);
inline KernelSygesv kernel_sygesv(0x18);
inline KernelSysysv kernel_sysysv(0x1C);
inline KernelSytrsv kernel_sytrsv(0x1C);
inline KernelSydisv kernel_sydisv(0x18);
inline KernelTrsm kernel_trsm(0x1A);
inline KernelTrsysv kernel_trsysv(0x1E);
inline KernelTrtrsv kernel_trtrsv(0x1E);
inline KernelTrdisv kernel_trdisv(0x1A);
inline KernelDisv kernel_disv(0x10);
inline KernelDisysv kernel_disysv(0x14);
inline KernelDitrsv kernel_ditrsv(0x14);
inline KernelDidisv kernel_didisv(0x00);
inline KernelPogesv kernel_pogesv(0x18);
inline KernelPosysv kernel_posysv(0x1C);
inline KernelPotrsv kernel_potrsv(0x1C);
inline KernelPodisv kernel_podisv(0x18);

inline std::vector<Kernel*> all_kernels = {
    &kernel_gemm,   &kernel_symm,   &kernel_trmm,   &kernel_dimm,
    &kernel_sysymm, &kernel_trsymm, &kernel_disymm, &kernel_trtrmm,
    &kernel_ditrmm, &kernel_didimm, &kernel_gegesv, &kernel_gesysv,
    &kernel_getrsv, &kernel_gedisv, &kernel_sygesv, &kernel_sysysv,
    &kernel_sytrsv, &kernel_sydisv, &kernel_trsm,   &kernel_trsysv,
    &kernel_trtrsv, &kernel_trdisv, &kernel_disv,   &kernel_disysv,
    &kernel_ditrsv, &kernel_didisv, &kernel_pogesv, &kernel_posysv,
    &kernel_potrsv, &kernel_podisv};

}  // namespace cg

#endif