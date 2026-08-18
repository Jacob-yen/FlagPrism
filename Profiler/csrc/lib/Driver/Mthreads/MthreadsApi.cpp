#include "Driver/Mthreads/MthreadsApi.h"

#include <cstdlib>
#include <string>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#endif

namespace proton {
namespace mthreads {
namespace {

#if defined(__linux__)
using Result = int;
using DeviceHandle = int;
using FnInit = Result (*)(unsigned int);
using FnDeviceGet = Result (*)(DeviceHandle *, int);
using FnDeviceGetName = Result (*)(char *, int, DeviceHandle);
using FnDeviceGetAttribute = Result (*)(int *, int, DeviceHandle);

constexpr int kClockRate = 13;
constexpr int kMemoryClockRate = 36;
constexpr int kBusWidth = 37;
constexpr int kMultiprocessorCount = 16;

struct Api {
  void *library = nullptr;
  FnInit init = nullptr;
  FnDeviceGet deviceGet = nullptr;
  FnDeviceGetName deviceGetName = nullptr;
  FnDeviceGetAttribute deviceGetAttribute = nullptr;

  ~Api() {
    if (library) {
      dlclose(library);
    }
  }

  bool load() {
    if (library) {
      return true;
    }
    const char *env =
        std::getenv("FLAGTREE_PROFILER_MTHREADS_DRIVER_LIBRARY");
    std::vector<std::string> candidates;
    if (env && *env) {
      candidates.emplace_back(env);
    }
    candidates.emplace_back("libmusa.so.1");
    candidates.emplace_back("libmusa.so");
    candidates.emplace_back("/usr/local/musa/lib/libmusa.so");
    for (const auto &candidate : candidates) {
      library = dlopen(candidate.c_str(), RTLD_LOCAL | RTLD_LAZY);
      if (library) {
        break;
      }
    }
    if (!library) {
      return false;
    }
    init = reinterpret_cast<FnInit>(dlsym(library, "muInit"));
    deviceGet = reinterpret_cast<FnDeviceGet>(dlsym(library, "muDeviceGet"));
    deviceGetName = reinterpret_cast<FnDeviceGetName>(
        dlsym(library, "muDeviceGetName"));
    deviceGetAttribute = reinterpret_cast<FnDeviceGetAttribute>(
        dlsym(library, "muDeviceGetAttribute"));
    return init != nullptr && deviceGet != nullptr;
  }
};

Api &api() {
  static Api instance;
  return instance;
}
#endif

} // namespace

Device getDevice(uint64_t index) {
  std::string arch = "musa";
  uint64_t clockRate = 0;
  uint64_t memoryClockRate = 0;
  uint64_t busWidth = 0;
  uint64_t numSms = 0;

#if defined(__linux__)
  auto &driver = api();
  if (driver.load() && driver.init(0) == 0) {
    DeviceHandle device = 0;
    if (driver.deviceGet(&device, static_cast<int>(index)) == 0) {
      if (driver.deviceGetName) {
        char name[256] = {};
        if (driver.deviceGetName(name, sizeof(name), device) == 0 && name[0]) {
          arch = name;
        }
      }
      if (driver.deviceGetAttribute) {
        int value = 0;
        if (driver.deviceGetAttribute(&value, kClockRate, device) == 0 &&
            value > 0) {
          clockRate = static_cast<uint64_t>(value);
        }
        value = 0;
        if (driver.deviceGetAttribute(&value, kMemoryClockRate, device) == 0 &&
            value > 0) {
          memoryClockRate = static_cast<uint64_t>(value);
        }
        value = 0;
        if (driver.deviceGetAttribute(&value, kBusWidth, device) == 0 &&
            value > 0) {
          busWidth = static_cast<uint64_t>(value);
        }
        value = 0;
        if (driver.deviceGetAttribute(&value, kMultiprocessorCount, device) ==
                0 &&
            value > 0) {
          numSms = static_cast<uint64_t>(value);
        }
      }
    }
  }
#endif

  return Device(DeviceType::MTHREADS, index, clockRate, memoryClockRate,
                busWidth, numSms, arch);
}

} // namespace mthreads
} // namespace proton
