#include "Integration/Registration.h"

#include "Conversion/ProtonGPUToLLVM/Passes.h"
#if !defined(FLAGPRISM_BACKEND_TIANSHU) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) && !defined(FLAGPRISM_BACKEND_MTHREADS)
#include "Conversion/ProtonGPUToLLVM/ProtonAMDGPUToLLVM/Passes.h"
#include "Conversion/ProtonGPUToLLVM/ProtonNvidiaGPUToLLVM/Passes.h"
#endif
#include "Conversion/ProtonToProtonGPU/Passes.h"
#include "Dialect/Proton/IR/Dialect.h"
#include "Dialect/ProtonGPU/IR/Dialect.h"
#include "Dialect/ProtonGPU/Transforms/Passes.h"

#include "mlir/IR/DialectRegistry.h"

namespace mlir::triton::proton {

void registerFlagTreeProtonPassesAndDialects(mlir::DialectRegistry &registry) {
  registerConvertProtonToProtonGPU();
#if !defined(FLAGPRISM_BACKEND_TIANSHU) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) && !defined(FLAGPRISM_BACKEND_MTHREADS)
  gpu::registerConvertProtonNvidiaGPUToLLVM();
  gpu::registerConvertProtonAMDGPUToLLVM();
#endif
  gpu::registerAllocateProtonSharedMemoryPass();
  gpu::registerAllocateProtonGlobalScratchBufferPass();
  gpu::registerScheduleBufferStorePass();
#if !defined(FLAGPRISM_BACKEND_TIANSHU) &&                                     \
    !defined(FLAGPRISM_BACKEND_ASCEND) && !defined(FLAGPRISM_BACKEND_MTHREADS)
  gpu::registerAddSchedBarriersPass();
#endif
  registry.insert<ProtonDialect, gpu::ProtonGPUDialect>();
}

} // namespace mlir::triton::proton
