#include "Device.h"
#if defined(FLAGPRISM_BACKEND_ASCEND)
#include "Driver/Ascend/AscendApi.h"
#endif
#if defined(FLAGPRISM_BACKEND_MTHREADS)
#include "Driver/Mthreads/MthreadsApi.h"
#endif
#if FLAGTREE_PROFILER_GPU_RUNTIME
#include "Driver/GPU/CudaApi.h"
#include "Driver/GPU/HipApi.h"
#endif

#include "Utility/Errors.h"

namespace proton {

Device getDevice(DeviceType type, uint64_t index) {
#if FLAGTREE_PROFILER_GPU_RUNTIME
  if (type == DeviceType::CUDA) {
    return cuda::getDevice(index);
  }
  if (type == DeviceType::HIP) {
    return hip::getDevice(index);
  }
#endif
  if (type == DeviceType::ASCEND) {
#if defined(FLAGPRISM_BACKEND_ASCEND)
    return ascend::getDevice(index);
#else
    throw std::runtime_error("DeviceType ASCEND is not enabled");
#endif
  }
#if defined(FLAGPRISM_BACKEND_MTHREADS)
  if (type == DeviceType::MTHREADS) {
    return mthreads::getDevice(index);
  }
#endif
  throw std::runtime_error("DeviceType not supported");
}

const std::string getDeviceTypeString(DeviceType type) {
  if (type == DeviceType::CUDA) {
    return DeviceTraits<DeviceType::CUDA>::name;
  } else if (type == DeviceType::HIP) {
    return DeviceTraits<DeviceType::HIP>::name;
  } else if (type == DeviceType::ASCEND) {
    return DeviceTraits<DeviceType::ASCEND>::name;
  } else if (type == DeviceType::MTHREADS) {
    return DeviceTraits<DeviceType::MTHREADS>::name;
  }
  throw std::runtime_error("DeviceType not supported");
}

} // namespace proton
