#include "Profiler/Vendor/Adapter.h"

#if !defined(FLAGPRISM_BACKEND_TIANSHU) && !defined(FLAGPRISM_BACKEND_MTHREADS) && \
    !defined(FLAGPRISM_BACKEND_NVIDIA)
#include "Profiler/Vendor/CannAdapter.h"
#endif
#if !defined(FLAGPRISM_BACKEND_ASCEND) && !defined(FLAGPRISM_BACKEND_MTHREADS) && \
    !defined(FLAGPRISM_BACKEND_NVIDIA)
#include "Profiler/Vendor/TianshuAdapter.h"
#endif
#if !defined(FLAGPRISM_BACKEND_ASCEND) && !defined(FLAGPRISM_BACKEND_TIANSHU) && \
    !defined(FLAGPRISM_BACKEND_NVIDIA)
#include "Profiler/Vendor/MthreadsAdapter.h"
#endif
#if !defined(FLAGPRISM_BACKEND_ASCEND) && !defined(FLAGPRISM_BACKEND_TIANSHU) && \
    !defined(FLAGPRISM_BACKEND_MTHREADS)
#include "Profiler/Vendor/NvidiaAdapter.h"
#endif
#include "Utility/String.h"

namespace proton {

const VendorAdapter *VendorAdapterRegistry::find(const std::string &name) {
  auto lower = toLower(name);
#if !defined(FLAGPRISM_BACKEND_TIANSHU) && !defined(FLAGPRISM_BACKEND_MTHREADS) && \
    !defined(FLAGPRISM_BACKEND_NVIDIA)
  if (lower == "cann") {
    return &CannAdapter::instance();
  }
#endif
#if !defined(FLAGPRISM_BACKEND_ASCEND) && !defined(FLAGPRISM_BACKEND_MTHREADS) && \
    !defined(FLAGPRISM_BACKEND_NVIDIA)
  if (lower == "tianshu" || lower == "corex" || lower == "iluvatar") {
    return &TianshuAdapter::instance();
  }
#endif
#if !defined(FLAGPRISM_BACKEND_ASCEND) && !defined(FLAGPRISM_BACKEND_TIANSHU) && \
    !defined(FLAGPRISM_BACKEND_NVIDIA)
  if (lower == "mthreads" || lower == "musa") {
    return &MthreadsAdapter::instance();
  }
#endif
#if !defined(FLAGPRISM_BACKEND_ASCEND) && !defined(FLAGPRISM_BACKEND_TIANSHU) && \
    !defined(FLAGPRISM_BACKEND_MTHREADS)
  // FlagPrism: expose NVIDIA under an explicit vendor name while retaining
  // the existing "cupti" profiler entry point for compatibility.
  if (lower == "nvidia" || lower == "cuda") {
    return &NvidiaAdapter::instance();
  }
#endif
  return nullptr;
}

std::vector<std::string> VendorAdapterRegistry::names() {
  std::vector<std::string> result;
#if !defined(FLAGPRISM_BACKEND_TIANSHU) && !defined(FLAGPRISM_BACKEND_MTHREADS) && \
    !defined(FLAGPRISM_BACKEND_NVIDIA)
  result.push_back("cann");
#endif
#if !defined(FLAGPRISM_BACKEND_ASCEND) && !defined(FLAGPRISM_BACKEND_MTHREADS) && \
    !defined(FLAGPRISM_BACKEND_NVIDIA)
  result.push_back("tianshu");
#endif
#if !defined(FLAGPRISM_BACKEND_ASCEND) && !defined(FLAGPRISM_BACKEND_TIANSHU) && \
    !defined(FLAGPRISM_BACKEND_NVIDIA)
  result.push_back("mthreads");
#endif
#if !defined(FLAGPRISM_BACKEND_ASCEND) && !defined(FLAGPRISM_BACKEND_TIANSHU) && \
    !defined(FLAGPRISM_BACKEND_MTHREADS)
  result.push_back("nvidia");
#endif
  return result;
}

} // namespace proton
