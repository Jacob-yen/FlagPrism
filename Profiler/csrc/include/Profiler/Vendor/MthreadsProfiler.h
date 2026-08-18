#ifndef PROTON_PROFILER_MTHREADS_PROFILER_H_
#define PROTON_PROFILER_MTHREADS_PROFILER_H_

#include "Context/Context.h"
#include "Data/Artifacts.h"
#include "Profiler/Profiler.h"
#include "Profiler/Vendor/Mode.h"
#include "Utility/Singleton.h"

#include <cstdint>
#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace proton {

class MthreadsProfiler final : public Profiler,
                               public OpInterface,
                               public Singleton<MthreadsProfiler> {
public:
  MthreadsProfiler() = default;
  ~MthreadsProfiler() override = default;

  static VendorProfileArtifact importMthreadsOutput(
      const SessionProfileMetadata &metadata,
      const VendorProfilePlan &plan);

private:
  void startOp(const Scope &scope) override;
  void stopOp(const Scope &scope) override;
  void doStart() override;
  void doFlush() override;
  void doStop() override;
  void doSetMode(const std::vector<std::string> &modeAndOptions) override;

  std::vector<RuntimeTraceEventKey> drainRuntimeEvents();
  std::string drainNativeOutputPath();
  std::string drainNativeCaptureError();

  struct NativeKernelEvent {
    std::string name;
    uint64_t startTimeNs = 0;
    uint64_t endTimeNs = 0;
    uint64_t gridId = 0;
    uint32_t deviceId = 0;
    uint32_t streamId = 0;
    uint32_t correlationId = 0;
    int32_t gridX = 0;
    int32_t gridY = 0;
    int32_t gridZ = 0;
    int32_t blockX = 0;
    int32_t blockY = 0;
    int32_t blockZ = 0;
    uint16_t registersPerThread = 0;
    int32_t staticSharedMemory = 0;
    int32_t dynamicSharedMemory = 0;
    uint32_t localMemoryPerThread = 0;
    uint32_t localMemoryTotal = 0;
    bool callbackEvent = false;
  };

  static void requestMuptiBuffer(uint8_t **buffer, size_t *size,
                                size_t *maxNumRecords);
  static void handleMuptiCallback(void *userdata, uint32_t domain,
                                  uint32_t callbackId, const void *data);
  static void completeMuptiBuffer(void *context, uint32_t streamId,
                                  uint8_t *buffer, size_t size,
                                  size_t validSize);
  void consumeMuptiBuffer(uint8_t *buffer, size_t validSize);
  bool startMuptiCapture();
  void flushMuptiCapture(bool forced);
  void stopMuptiCapture();
  void writeMuptiOutput();

  std::mutex mutex;
  std::unordered_map<size_t, uint64_t> opStartTimesNs;
  std::vector<RuntimeTraceEventKey> runtimeEvents;
  std::vector<NativeKernelEvent> nativeKernelEvents;
  std::unordered_map<uint32_t, NativeKernelEvent> callbackLaunches;
  uint32_t deviceId = 0;
  std::string importPath;
  std::string nativeOutputPath;
  std::string lastNativeOutputPath;
  std::string nativeCaptureError;
  bool nativeCaptureRequested = false;
  bool nativeActivityRequested = false;
  bool nativeCaptureActive = false;
  bool nativeActivityEnabled = false;
  bool nativeCallbacksRegistered = false;
  bool nativeSubscriberRegistered = false;
  size_t nativeBufferRequests = 0;
  size_t nativeBufferCompletions = 0;
  size_t nativeActivityRecords = 0;
};

} // namespace proton

#endif // PROTON_PROFILER_MTHREADS_PROFILER_H_
