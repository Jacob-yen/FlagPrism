#ifndef PROTON_DRIVER_MTHREADS_API_H_
#define PROTON_DRIVER_MTHREADS_API_H_

#include "Device.h"

#include <cstdint>

namespace proton {
namespace mthreads {

// Query MUSA through its driver ABI without making the profiler depend on a
// particular MUSA SDK installation at link time.
Device getDevice(uint64_t index);

} // namespace mthreads
} // namespace proton

#endif // PROTON_DRIVER_MTHREADS_API_H_
