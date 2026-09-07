#include "Profiler/Vendor/MthreadsProfiler.h"

#include "Utility/String.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <map>
#include <new>
#include <optional>
#include <set>
#include <sstream>

#if defined(FLAGTREE_MTHREADS_HAS_MUPTI)
#include <dlfcn.h>
#include <mupti_activity.h>
#include <mupti_callbacks.h>
#endif

namespace proton {
namespace {

#if defined(FLAGTREE_MTHREADS_HAS_MUPTI)

constexpr size_t kMuptiActivityBufferSize = 8 * 1024 * 1024;

struct MthreadsMuptiApi {
  using RegisterCallbacks = MUptiResult (*)(MUpti_BuffersCallbackRequestFunc,
                                            MUpti_BuffersCallbackCompleteFunc);
  using ActivityEnable = MUptiResult (*)(MUpti_ActivityKind);
  using ActivityDisable = MUptiResult (*)(MUpti_ActivityKind);
  using ActivityFlushAll = MUptiResult (*)(uint32_t);
  using ActivityGetNextRecord = MUptiResult (*)(uint8_t *, size_t,
                                                MUpti_Activity **);
  using ActivityGetNumDroppedRecords = MUptiResult (*)(MUcontext, uint32_t,
                                                       size_t *);
  using ActivityEnableLaunchAttributes = MUptiResult (*)(uint8_t);
  using Subscribe = MUptiResult (*)(MUpti_SubscriberHandle *,
                                    MUpti_CallbackFunc, void *);
  using Unsubscribe = MUptiResult (*)(MUpti_SubscriberHandle);
  using EnableDomain = MUptiResult (*)(uint32_t, MUpti_SubscriberHandle,
                                       MUpti_CallbackDomain);

  void *library = nullptr;
  RegisterCallbacks registerCallbacks = nullptr;
  ActivityEnable activityEnable = nullptr;
  ActivityDisable activityDisable = nullptr;
  ActivityFlushAll activityFlushAll = nullptr;
  ActivityGetNextRecord activityGetNextRecord = nullptr;
  ActivityGetNumDroppedRecords activityGetNumDroppedRecords = nullptr;
  ActivityEnableLaunchAttributes activityEnableLaunchAttributes = nullptr;
  Subscribe subscribe = nullptr;
  Unsubscribe unsubscribe = nullptr;
  EnableDomain enableDomain = nullptr;
  MUpti_SubscriberHandle subscriber = nullptr;

  // MUPTI can retain an incomplete activity buffer until process teardown.
  // Do not dlclose it from a static destructor: MUSA 4.3 may release that
  // buffer again during its own shutdown, which results in a double free.
  ~MthreadsMuptiApi() = default;

  bool load() {
    if (library) {
      return registerCallbacks && activityEnable && activityDisable &&
             activityFlushAll && activityGetNextRecord;
    }

    std::vector<std::string> candidates;
    if (const char *env =
            std::getenv("FLAGTREE_PROFILER_MTHREADS_MUPTI_LIBRARY")) {
      if (*env) {
        candidates.emplace_back(env);
      }
    }
    if (const char *musaHome = std::getenv("MUSA_HOME")) {
      if (*musaHome) {
        candidates.emplace_back(std::string(musaHome) + "/lib/libmupti.so");
        candidates.emplace_back(std::string(musaHome) + "/lib64/libmupti.so");
      }
    }
    candidates.emplace_back("libmupti.so.1");
    candidates.emplace_back("libmupti.so");
    candidates.emplace_back("/usr/local/musa/lib/libmupti.so");

    for (const auto &candidate : candidates) {
      library = dlopen(candidate.c_str(), RTLD_LOCAL | RTLD_LAZY);
      if (library) {
        break;
      }
    }
    if (!library) {
      return false;
    }

    registerCallbacks = reinterpret_cast<RegisterCallbacks>(
        dlsym(library, "muptiActivityRegisterCallbacks"));
    activityEnable =
        reinterpret_cast<ActivityEnable>(dlsym(library, "muptiActivityEnable"));
    activityDisable = reinterpret_cast<ActivityDisable>(
        dlsym(library, "muptiActivityDisable"));
    activityFlushAll = reinterpret_cast<ActivityFlushAll>(
        dlsym(library, "muptiActivityFlushAll"));
    activityGetNextRecord = reinterpret_cast<ActivityGetNextRecord>(
        dlsym(library, "muptiActivityGetNextRecord"));
    activityGetNumDroppedRecords =
        reinterpret_cast<ActivityGetNumDroppedRecords>(
            dlsym(library, "muptiActivityGetNumDroppedRecords"));
    activityEnableLaunchAttributes =
        reinterpret_cast<ActivityEnableLaunchAttributes>(
            dlsym(library, "muptiActivityEnableLaunchAttributes"));
    subscribe = reinterpret_cast<Subscribe>(dlsym(library, "muptiSubscribe"));
    unsubscribe =
        reinterpret_cast<Unsubscribe>(dlsym(library, "muptiUnsubscribe"));
    enableDomain =
        reinterpret_cast<EnableDomain>(dlsym(library, "muptiEnableDomain"));
    return registerCallbacks && activityEnable && activityDisable &&
           activityFlushAll && activityGetNextRecord;
  }
};

MthreadsMuptiApi &muptiApi() {
  static MthreadsMuptiApi api;
  return api;
}

#endif

uint64_t nowNs() {
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now().time_since_epoch())
          .count());
}

std::string normalizeColumn(std::string value) {
  std::string result;
  for (char ch : toLower(trim(value))) {
    if (std::isalnum(static_cast<unsigned char>(ch))) {
      result.push_back(ch);
    }
  }
  return result;
}

std::vector<std::string> splitCsvLine(const std::string &line) {
  std::vector<std::string> fields;
  std::string field;
  bool quoted = false;
  for (size_t i = 0; i < line.size(); ++i) {
    const char ch = line[i];
    if (ch == '"') {
      if (quoted && i + 1 < line.size() && line[i + 1] == '"') {
        field.push_back('"');
        ++i;
      } else {
        quoted = !quoted;
      }
    } else if (ch == ',' && !quoted) {
      fields.push_back(trim(field));
      field.clear();
    } else {
      field.push_back(ch);
    }
  }
  fields.push_back(trim(field));
  return fields;
}

std::optional<size_t> findColumn(const std::vector<std::string> &headers,
                                 std::initializer_list<const char *> names) {
  for (size_t i = 0; i < headers.size(); ++i) {
    const auto column = normalizeColumn(headers[i]);
    for (const char *name : names) {
      if (column == normalizeColumn(name)) {
        return i;
      }
    }
  }
  return std::nullopt;
}

std::string cell(const std::vector<std::string> &row,
                 const std::optional<size_t> &index) {
  if (!index || *index >= row.size()) {
    return {};
  }
  return trim(row[*index]);
}

std::optional<uint64_t> parseU64(const std::string &raw) {
  const auto value = trim(raw);
  if (value.empty()) {
    return std::nullopt;
  }
  char *end = nullptr;
  errno = 0;
  const auto parsed = std::strtoull(value.c_str(), &end, 0);
  if (errno != 0 || end == value.c_str() || *end != '\0') {
    return std::nullopt;
  }
  return static_cast<uint64_t>(parsed);
}

std::optional<double> parseDouble(const std::string &raw) {
  const auto value = trim(raw);
  if (value.empty()) {
    return std::nullopt;
  }
  char *end = nullptr;
  errno = 0;
  const auto parsed = std::strtod(value.c_str(), &end);
  if (errno != 0 || end == value.c_str() || *end != '\0') {
    return std::nullopt;
  }
  return parsed;
}

std::optional<MetricValueType> parseMetric(const std::string &raw) {
  if (auto integer = parseU64(raw)) {
    return integer;
  }
  if (auto number = parseDouble(raw)) {
    return number;
  }
  const auto value = trim(raw);
  if (!value.empty()) {
    return value;
  }
  return std::nullopt;
}

uint64_t parseTimeNs(const std::string &raw, const std::string &header) {
  auto number = parseDouble(raw);
  if (!number || *number <= 0) {
    return 0;
  }
  const auto normalized = normalizeColumn(header);
  if (normalized.find("ns") != std::string::npos) {
    return static_cast<uint64_t>(*number);
  }
  if (normalized.find("ms") != std::string::npos) {
    return static_cast<uint64_t>(*number * 1'000'000.0);
  }
  return static_cast<uint64_t>(*number * 1'000.0);
}

bool sameKernel(const std::string &lhs, const std::string &rhs) {
  const auto left = toLower(trim(lhs));
  const auto right = toLower(trim(rhs));
  return !left.empty() && !right.empty() &&
         (left == right || left.find(right) != std::string::npos ||
          right.find(left) != std::string::npos);
}

std::vector<std::filesystem::path>
collectMthreadsFiles(const SessionProfileMetadata &metadata) {
  std::vector<std::filesystem::path> roots;
  for (const auto *key :
       {"mupti_import_path", "mthreads_import_path", "vendor_import_path",
        "output_path", "mupti_output_path", "mthreads_output_path"}) {
    auto it = metadata.config.find(key);
    if (it != metadata.config.end() && !trim(it->second).empty()) {
      roots.emplace_back(trim(it->second));
    }
  }
  if (const char *env = std::getenv("FLAGTREE_PROFILER_MTHREADS_IMPORT_PATH")) {
    if (*env) {
      roots.emplace_back(env);
    }
  }

  std::set<std::string> seen;
  std::vector<std::filesystem::path> files;
  auto add = [&](const std::filesystem::path &path) {
    if (seen.insert(path.lexically_normal().string()).second) {
      files.push_back(path);
    }
  };
  for (const auto &root : roots) {
    std::error_code ec;
    if (!std::filesystem::exists(root, ec)) {
      continue;
    }
    if (std::filesystem::is_regular_file(root, ec)) {
      if (toLower(root.extension().string()) == ".csv") {
        add(root);
      }
      continue;
    }
    std::filesystem::recursive_directory_iterator it(
        root, std::filesystem::directory_options::skip_permission_denied, ec);
    for (const std::filesystem::recursive_directory_iterator end; it != end;
         it.increment(ec)) {
      if (ec) {
        ec.clear();
        continue;
      }
      if (it->is_regular_file(ec) &&
          toLower(it->path().extension().string()) == ".csv") {
        add(it->path());
      }
    }
  }
  return files;
}

void importCsv(const std::filesystem::path &file,
               const std::vector<RuntimeTraceEventKey> &runtimeEvents,
               std::vector<bool> &usedRuntimeEvents,
               VendorProfileArtifact &artifact,
               const std::string &source = "mupti_csv") {
  std::ifstream input(file);
  if (!input.is_open()) {
    artifact.degradeReasons.push_back("Failed to open MUPTI CSV: " +
                                      file.string());
    return;
  }
  std::string line;
  if (!std::getline(input, line)) {
    return;
  }
  const auto headers = splitCsvLine(line);
  const auto nameIndex =
      findColumn(headers, {"kernel_name", "kernelname", "op_name", "name"});
  const auto startIndex = findColumn(
      headers, {"start_time_ns", "start_time_us", "start_time", "start"});
  const auto endIndex =
      findColumn(headers, {"end_time_ns", "end_time_us", "end_time", "end"});
  const auto durationIndex = findColumn(
      headers, {"duration_ns", "duration_us", "duration_ms", "duration"});
  const auto deviceIndex = findColumn(headers, {"device_id", "device"});
  const auto streamIndex = findColumn(headers, {"stream_id", "stream"});
  const auto taskIndex = findColumn(headers, {"kernel_id", "task_id", "task"});
  const auto correlationIndex =
      findColumn(headers, {"correlation_id", "correlation", "corr_id"});
  const auto sourceIndex =
      findColumn(headers, {"capture_source", "source", "record_source"});
  if (!nameIndex) {
    artifact.degradeReasons.push_back("MUPTI CSV has no kernel name column: " +
                                      file.string());
    return;
  }

  while (std::getline(input, line)) {
    const auto row = splitCsvLine(line);
    if (row.empty()) {
      continue;
    }
    VendorMetricAssociation association;
    association.source = cell(row, sourceIndex);
    if (association.source.empty()) {
      association.source = source;
    }
    association.runtimeEvent.opName = cell(row, nameIndex);
    association.runtimeEvent.deviceId =
        parseU64(cell(row, deviceIndex)).value_or(0);
    association.runtimeEvent.streamId =
        parseU64(cell(row, streamIndex)).value_or(0);
    association.runtimeEvent.taskId =
        parseU64(cell(row, taskIndex)).value_or(0);
    association.runtimeEvent.correlationId =
        parseU64(cell(row, correlationIndex)).value_or(0);
    if (startIndex) {
      association.runtimeEvent.startTimeNs =
          parseTimeNs(cell(row, startIndex), headers[*startIndex]);
    }
    if (endIndex) {
      association.runtimeEvent.endTimeNs =
          parseTimeNs(cell(row, endIndex), headers[*endIndex]);
    }
    if (association.runtimeEvent.endTimeNs == 0 && durationIndex) {
      association.runtimeEvent.endTimeNs =
          association.runtimeEvent.startTimeNs +
          parseTimeNs(cell(row, durationIndex), headers[*durationIndex]);
    }
    if (association.runtimeEvent.endTimeNs >=
            association.runtimeEvent.startTimeNs &&
        association.runtimeEvent.endTimeNs != 0 &&
        association.runtimeEvent.startTimeNs != 0) {
      association.metrics["mthreads.duration_ns"] =
          association.runtimeEvent.endTimeNs -
          association.runtimeEvent.startTimeNs;
    }
    for (size_t i = 0; i < headers.size() && i < row.size(); ++i) {
      if (i == *nameIndex || (startIndex && i == *startIndex) ||
          (endIndex && i == *endIndex) ||
          (durationIndex && i == *durationIndex)) {
        continue;
      }
      if (auto value = parseMetric(row[i])) {
        const auto metricName = normalizeColumn(headers[i]);
        if (!metricName.empty()) {
          association.metrics["mthreads." + metricName] = *value;
        }
      }
    }
    association.metrics["mthreads.source_file"] = file.string();

    for (size_t i = 0; i < runtimeEvents.size(); ++i) {
      if (!usedRuntimeEvents[i] && sameKernel(association.runtimeEvent.opName,
                                              runtimeEvents[i].opName)) {
        usedRuntimeEvents[i] = true;
        if (association.runtimeEvent.scopeId == 0) {
          association.runtimeEvent.scopeId = runtimeEvents[i].scopeId;
        }
        if (association.runtimeEvent.startTimeNs == 0) {
          association.runtimeEvent.startTimeNs = runtimeEvents[i].startTimeNs;
        }
        if (association.runtimeEvent.endTimeNs == 0) {
          association.runtimeEvent.endTimeNs = runtimeEvents[i].endTimeNs;
        }
        association.state = VendorMetricState::Collected;
        break;
      }
    }
    if (association.state != VendorMetricState::Collected) {
      association.state = VendorMetricState::Unmatched;
      association.note = "MUPTI row did not match a FlagPrism launch event";
    }
    artifact.associations.push_back(std::move(association));
  }
}

bool parseBoolOption(const std::string &value) {
  const auto normalized = toLower(trim(value));
  return normalized == "1" || normalized == "true" || normalized == "on" ||
         normalized == "yes";
}

} // namespace

void MthreadsProfiler::requestMuptiBuffer(uint8_t **buffer, size_t *size,
                                          size_t *maxNumRecords) {
  if (!buffer || !size || !maxNumRecords) {
    return;
  }
#if defined(FLAGTREE_MTHREADS_HAS_MUPTI)
  {
    std::lock_guard<std::mutex> lock(MthreadsProfiler::instance().mutex);
    ++MthreadsProfiler::instance().nativeBufferRequests;
  }
  *size = kMuptiActivityBufferSize;
  *maxNumRecords = 0;
  *buffer = new (std::nothrow) uint8_t[*size];
#else
  *size = 0;
  *maxNumRecords = 0;
  *buffer = nullptr;
#endif
}

void MthreadsProfiler::handleMuptiCallback(void *userdata, uint32_t domain,
                                           uint32_t callbackId,
                                           const void *data) {
  (void)userdata;
  (void)callbackId;
#if defined(FLAGTREE_MTHREADS_HAS_MUPTI)
  if (!data || (domain != MUPTI_CB_DOMAIN_DRIVER_API &&
                domain != MUPTI_CB_DOMAIN_RUNTIME_API)) {
    return;
  }
  const auto *callback = static_cast<const MUpti_CallbackData *>(data);
  if (!callback->functionName ||
      toLower(callback->functionName).find("launch") == std::string::npos) {
    return;
  }

  auto &profiler = MthreadsProfiler::instance();
  const auto correlationId = callback->correlationId;
  if (callback->callbackSite == MUPTI_API_ENTER) {
    NativeKernelEvent event;
    event.name = callback->symbolName && *callback->symbolName
                     ? callback->symbolName
                     : callback->functionName;
    event.startTimeNs = nowNs();
    event.correlationId = correlationId;
    event.callbackEvent = true;
    std::lock_guard<std::mutex> lock(profiler.mutex);
    event.deviceId = profiler.deviceId;
    profiler.callbackLaunches[correlationId] = std::move(event);
    return;
  }
  if (callback->callbackSite != MUPTI_API_EXIT) {
    return;
  }

  NativeKernelEvent event;
  {
    std::lock_guard<std::mutex> lock(profiler.mutex);
    auto it = profiler.callbackLaunches.find(correlationId);
    if (it != profiler.callbackLaunches.end()) {
      event = std::move(it->second);
      profiler.callbackLaunches.erase(it);
    } else {
      event.name = callback->symbolName && *callback->symbolName
                       ? callback->symbolName
                       : callback->functionName;
      event.startTimeNs = nowNs();
      event.correlationId = correlationId;
      event.callbackEvent = true;
    }
    event.deviceId = profiler.deviceId;
    event.endTimeNs = nowNs();
    profiler.nativeKernelEvents.push_back(std::move(event));
  }
#else
  (void)domain;
  (void)data;
#endif
}

void MthreadsProfiler::completeMuptiBuffer(void *context, uint32_t streamId,
                                           uint8_t *buffer, size_t size,
                                           size_t validSize) {
  (void)context;
  (void)streamId;
  (void)size;
#if defined(FLAGTREE_MTHREADS_HAS_MUPTI)
  if (buffer) {
    {
      std::lock_guard<std::mutex> lock(MthreadsProfiler::instance().mutex);
      ++MthreadsProfiler::instance().nativeBufferCompletions;
    }
    MthreadsProfiler::instance().consumeMuptiBuffer(buffer, validSize);
    delete[] buffer;
  }
#else
  (void)buffer;
  (void)validSize;
#endif
}

void MthreadsProfiler::consumeMuptiBuffer(uint8_t *buffer, size_t validSize) {
#if defined(FLAGTREE_MTHREADS_HAS_MUPTI)
  if (!buffer || validSize == 0) {
    return;
  }
  auto &api = muptiApi();
  if (!api.load()) {
    return;
  }

  // MUPTI treats record as an in/out cursor. Initialize it once and preserve
  // every returned value so the next call advances through the buffer.
  MUpti_Activity *record = nullptr;
  while (true) {
    const auto result = api.activityGetNextRecord(buffer, validSize, &record);
    if (result != MUPTI_SUCCESS) {
      break;
    }
    if (!record || (record->kind != MUPTI_ACTIVITY_KIND_KERNEL &&
                    record->kind != MUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL)) {
      continue;
    }

    const auto *kernel =
        reinterpret_cast<const MUpti_ActivityKernel6 *>(record);
    NativeKernelEvent event;
    event.name = kernel->name ? kernel->name : "";
    event.startTimeNs = kernel->start;
    event.endTimeNs = kernel->end;
    event.gridId = static_cast<uint64_t>(kernel->gridId);
    event.deviceId = kernel->deviceId;
    event.streamId = kernel->streamId;
    event.correlationId = kernel->correlationId;
    event.gridX = kernel->gridX;
    event.gridY = kernel->gridY;
    event.gridZ = kernel->gridZ;
    event.blockX = kernel->blockX;
    event.blockY = kernel->blockY;
    event.blockZ = kernel->blockZ;
    event.registersPerThread = kernel->registersPerThread;
    event.staticSharedMemory = kernel->staticSharedMemory;
    event.dynamicSharedMemory = kernel->dynamicSharedMemory;
    event.localMemoryPerThread = kernel->localMemoryPerThread;
    event.localMemoryTotal = kernel->localMemoryTotal;
    {
      std::lock_guard<std::mutex> lock(mutex);
      ++nativeActivityRecords;
      nativeKernelEvents.push_back(std::move(event));
    }
  }
#else
  (void)buffer;
  (void)validSize;
#endif
}

bool MthreadsProfiler::startMuptiCapture() {
#if defined(FLAGTREE_MTHREADS_HAS_MUPTI)
  auto &api = muptiApi();
  if (!api.load()) {
    std::lock_guard<std::mutex> lock(mutex);
    nativeCaptureError = "MUPTI library or activity symbols were not found";
    return false;
  }
  bool subscriberEnabled = nativeSubscriberRegistered;
  if (!subscriberEnabled && api.subscribe && api.enableDomain) {
    MUpti_SubscriberHandle subscriber = nullptr;
    const auto result =
        api.subscribe(&subscriber,
                      reinterpret_cast<MUpti_CallbackFunc>(
                          &MthreadsProfiler::handleMuptiCallback),
                      nullptr);
    if (result == MUPTI_SUCCESS && subscriber) {
      const auto driverResult =
          api.enableDomain(1, subscriber, MUPTI_CB_DOMAIN_DRIVER_API);
      const auto runtimeResult =
          api.enableDomain(1, subscriber, MUPTI_CB_DOMAIN_RUNTIME_API);
      if (driverResult == MUPTI_SUCCESS || runtimeResult == MUPTI_SUCCESS) {
        api.subscriber = subscriber;
        nativeSubscriberRegistered = true;
        subscriberEnabled = true;
      } else if (api.unsubscribe) {
        api.unsubscribe(subscriber);
      }
    }
  }

  bool activityEnabled = false;
  int activityResult = MUPTI_ERROR_NOT_SUPPORTED;
  // MUSA 4.3's MT-Perf activity queue can retain an incomplete buffer and
  // corrupt it during process teardown. The MUPTI launch callback is stable
  // and is the default native path; activity capture is opt-in for runtimes
  // where the hardware activity stream is known to be functional.
  if (nativeActivityRequested || !subscriberEnabled) {
    const auto kernelResult = api.activityEnable(MUPTI_ACTIVITY_KIND_KERNEL);
    activityResult = static_cast<int>(kernelResult);
    if (kernelResult == MUPTI_SUCCESS && !nativeCallbacksRegistered) {
      const auto result = api.registerCallbacks(
          &MthreadsProfiler::requestMuptiBuffer,
          reinterpret_cast<MUpti_BuffersCallbackCompleteFunc>(
              &MthreadsProfiler::completeMuptiBuffer));
      if (result == MUPTI_SUCCESS) {
        nativeCallbacksRegistered = true;
        activityEnabled = true;
      } else {
        api.activityDisable(MUPTI_ACTIVITY_KIND_KERNEL);
      }
    } else if (kernelResult == MUPTI_SUCCESS) {
      activityEnabled = true;
    }
    if (activityEnabled && api.activityEnableLaunchAttributes) {
      api.activityEnableLaunchAttributes(1);
    }
  }

  if (!activityEnabled && !subscriberEnabled) {
    std::lock_guard<std::mutex> lock(mutex);
    nativeCaptureError =
        "MUPTI native capture could not be enabled (activity=" +
        std::to_string(activityResult) + ")";
    return false;
  }
  std::lock_guard<std::mutex> lock(mutex);
  nativeCaptureActive = true;
  nativeActivityEnabled = activityEnabled;
  return true;
#else
  std::lock_guard<std::mutex> lock(mutex);
  nativeCaptureError = "FlagPrism was built without MUPTI headers";
  return false;
#endif
}

void MthreadsProfiler::flushMuptiCapture(bool forced) {
#if defined(FLAGTREE_MTHREADS_HAS_MUPTI)
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (!nativeCaptureActive || !nativeActivityEnabled) {
      return;
    }
  }
  auto &api = muptiApi();
  if (api.load()) {
    api.activityFlushAll(forced ? MUPTI_ACTIVITY_FLAG_FLUSH_FORCED : 0);
  }
#else
  (void)forced;
#endif
}

void MthreadsProfiler::stopMuptiCapture() {
#if defined(FLAGTREE_MTHREADS_HAS_MUPTI)
  // Session::deactivate() has already performed the final completed-record
  // flush after synchronizing device work. A second flush during doStop can
  // wait forever in the MUSA 4.3 MUPTI runtime, so only disable activity here.
  bool activityEnabled = false;
  MUpti_SubscriberHandle subscriber = nullptr;
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (!nativeCaptureActive) {
      return;
    }
    activityEnabled = nativeActivityEnabled;
    if (nativeSubscriberRegistered) {
      subscriber = muptiApi().subscriber;
    }
    nativeActivityEnabled = false;
    nativeSubscriberRegistered = false;
    nativeCaptureActive = false;
  }
  auto &api = muptiApi();
  if (api.load() && activityEnabled) {
    api.activityDisable(MUPTI_ACTIVITY_KIND_KERNEL);
  }
  if (api.load() && subscriber && api.unsubscribe) {
    api.unsubscribe(subscriber);
    api.subscriber = nullptr;
  }
#endif
}

void MthreadsProfiler::writeMuptiOutput() {
  std::vector<NativeKernelEvent> events;
  std::string outputPath;
  {
    std::lock_guard<std::mutex> lock(mutex);
    events = nativeKernelEvents;
    outputPath = nativeOutputPath;
    lastNativeOutputPath.clear();
  }
  if (outputPath.empty()) {
    return;
  }
  if (events.empty()) {
    std::lock_guard<std::mutex> lock(mutex);
    nativeCaptureError =
        std::string(
            "MUPTI capture produced no completed kernel activity records ") +
        "(buffer_requests=" + std::to_string(nativeBufferRequests) +
        ", buffer_completions=" + std::to_string(nativeBufferCompletions) +
        ", activity_records=" + std::to_string(nativeActivityRecords) + ")";
    return;
  }

  std::error_code ec;
  const std::filesystem::path path(outputPath);
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path(), ec);
  }
  std::ofstream output(path);
  if (!output.is_open()) {
    std::lock_guard<std::mutex> lock(mutex);
    nativeCaptureError = "Failed to write MUPTI CSV: " + outputPath;
    return;
  }

  auto writeCsvString = [&](const std::string &value) {
    output << '"';
    for (const char ch : value) {
      if (ch == '"') {
        output << "\"\"";
      } else {
        output << ch;
      }
    }
    output << '"';
  };
  output << "kernel_name,start_time_ns,end_time_ns,device_id,stream_id,"
            "correlation_id,grid_id,grid_x,grid_y,grid_z,block_x,block_y,"
            "block_z,registers_per_thread,static_shared_memory,"
            "dynamic_shared_memory,local_memory_per_thread,"
            "local_memory_total,capture_source\n";

  std::set<std::string> seen;
  std::set<uint32_t> activityCorrelations;
  for (const auto &event : events) {
    if (!event.callbackEvent && event.correlationId != 0) {
      activityCorrelations.insert(event.correlationId);
    }
  }
  size_t written = 0;
  for (const auto &event : events) {
    if (event.callbackEvent && event.correlationId != 0 &&
        activityCorrelations.count(event.correlationId) != 0) {
      continue;
    }
    const auto key = event.name + ":" + std::to_string(event.startTimeNs) +
                     ":" + std::to_string(event.endTimeNs) + ":" +
                     std::to_string(event.correlationId);
    if (!seen.insert(key).second) {
      continue;
    }
    writeCsvString(event.name);
    output << ',' << event.startTimeNs << ',' << event.endTimeNs << ','
           << event.deviceId << ',' << event.streamId << ','
           << event.correlationId << ',' << event.gridId << ',' << event.gridX
           << ',' << event.gridY << ',' << event.gridZ << ',' << event.blockX
           << ',' << event.blockY << ',' << event.blockZ << ','
           << event.registersPerThread << ',' << event.staticSharedMemory << ','
           << event.dynamicSharedMemory << ',' << event.localMemoryPerThread
           << ',' << event.localMemoryTotal << ',';
    writeCsvString(event.callbackEvent ? "mupti_callback" : "mupti_activity");
    output << '\n';
    ++written;
  }
  if (output.good() && written != 0) {
    std::lock_guard<std::mutex> lock(mutex);
    lastNativeOutputPath = outputPath;
  }
}

void MthreadsProfiler::startOp(const Scope &scope) {
  std::lock_guard<std::mutex> lock(mutex);
  opStartTimesNs[scope.scopeId] = nowNs();
}

void MthreadsProfiler::stopOp(const Scope &scope) {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = opStartTimesNs.find(scope.scopeId);
  if (it == opStartTimesNs.end()) {
    return;
  }
  RuntimeTraceEventKey event;
  event.scopeId = scope.scopeId;
  event.opName = scope.name;
  event.deviceId = deviceId;
  event.startTimeNs = it->second;
  event.endTimeNs = nowNs();
  runtimeEvents.push_back(std::move(event));
  opStartTimesNs.erase(it);
}

void MthreadsProfiler::doStart() {
  {
    std::lock_guard<std::mutex> lock(mutex);
    opStartTimesNs.clear();
    runtimeEvents.clear();
    nativeKernelEvents.clear();
    callbackLaunches.clear();
    lastNativeOutputPath.clear();
    nativeCaptureError.clear();
    nativeBufferRequests = 0;
    nativeBufferCompletions = 0;
    nativeActivityRecords = 0;
  }
  if (nativeCaptureRequested) {
    startMuptiCapture();
  }
}

void MthreadsProfiler::doFlush() {
  // MUSA 4.3 may spin indefinitely when a completed activity buffer is
  // flushed without MUPTI_ACTIVITY_FLAG_FLUSH_FORCED.
  flushMuptiCapture(true);
}

void MthreadsProfiler::doStop() {
  stopMuptiCapture();
  writeMuptiOutput();
  std::lock_guard<std::mutex> lock(mutex);
  opStartTimesNs.clear();
}

void MthreadsProfiler::doSetMode(
    const std::vector<std::string> &modeAndOptions) {
  std::lock_guard<std::mutex> lock(mutex);
  deviceId = 0;
  importPath.clear();
  nativeOutputPath.clear();
  nativeCaptureError.clear();
  nativeCaptureRequested = false;
  nativeActivityRequested = false;
  bool explicitNativeCapture = false;
  for (const auto &raw : modeAndOptions) {
    const auto token = trim(raw);
    const auto separator = token.find('=');
    if (separator == std::string::npos) {
      continue;
    }
    const auto key = toLower(trim(token.substr(0, separator)));
    const auto value = trim(token.substr(separator + 1));
    if (key == "device_id") {
      deviceId = static_cast<uint32_t>(parseU64(value).value_or(0));
    } else if (key == "vendor_metrics") {
      nativeCaptureRequested = !trim(value).empty();
    } else if (key == "mupti_capture" || key == "native_capture") {
      explicitNativeCapture = true;
      nativeCaptureRequested = parseBoolOption(value);
    } else if (key == "mupti_activity" || key == "activity_capture") {
      nativeActivityRequested = parseBoolOption(value);
    } else if (key == "mupti_output_path" || key == "mthreads_output_path" ||
               key == "output_path") {
      nativeOutputPath = value;
    } else if (key == "mupti_import_path" || key == "mthreads_import_path" ||
               key == "vendor_import_path") {
      importPath = value;
    }
  }
  if (nativeOutputPath.empty()) {
    if (const char *env =
            std::getenv("FLAGTREE_PROFILER_MTHREADS_OUTPUT_PATH")) {
      if (*env) {
        nativeOutputPath = env;
      }
    }
  }
  if (!importPath.empty() && nativeOutputPath.empty() &&
      !explicitNativeCapture) {
    nativeCaptureRequested = false;
  }
  if (nativeCaptureRequested && nativeOutputPath.empty()) {
    std::error_code ec;
    auto directory = std::filesystem::temp_directory_path(ec);
    if (ec) {
      directory = std::filesystem::current_path(ec);
    }
    nativeOutputPath = (directory / ("flagtree_mthreads_mupti_" +
                                     std::to_string(nowNs()) + ".csv"))
                           .string();
  }
}

std::vector<RuntimeTraceEventKey> MthreadsProfiler::drainRuntimeEvents() {
  std::lock_guard<std::mutex> lock(mutex);
  auto events = runtimeEvents;
  runtimeEvents.clear();
  return events;
}

std::string MthreadsProfiler::drainNativeOutputPath() {
  std::lock_guard<std::mutex> lock(mutex);
  auto path = lastNativeOutputPath;
  lastNativeOutputPath.clear();
  return path;
}

std::string MthreadsProfiler::drainNativeCaptureError() {
  std::lock_guard<std::mutex> lock(mutex);
  auto error = nativeCaptureError;
  nativeCaptureError.clear();
  return error;
}

VendorProfileArtifact
MthreadsProfiler::importMthreadsOutput(const SessionProfileMetadata &metadata,
                                       const VendorProfilePlan &plan) {
  VendorProfileArtifact artifact;
  artifact.backend = metadata.backend;
  artifact.requestedMetrics = plan.requested.vendorMetrics;
  artifact.enabledMetrics = plan.enabledVendorMetrics;

  auto &profiler = MthreadsProfiler::instance();
  const auto runtimeEvents = profiler.drainRuntimeEvents();
  const auto nativeOutputPath = profiler.drainNativeOutputPath();
  auto files = collectMthreadsFiles(metadata);
  if (!nativeOutputPath.empty()) {
    const auto normalizedNativePath =
        std::filesystem::path(nativeOutputPath).lexically_normal().string();
    bool alreadyIncluded = false;
    for (const auto &file : files) {
      if (file.lexically_normal().string() == normalizedNativePath) {
        alreadyIncluded = true;
        break;
      }
    }
    if (!alreadyIncluded) {
      files.emplace_back(nativeOutputPath);
    }
  }
  std::vector<bool> usedRuntimeEvents(runtimeEvents.size(), false);
  for (const auto &file : files) {
    artifact.rawInputs.push_back(file.string());
    const auto source =
        !nativeOutputPath.empty() && file.lexically_normal().string() ==
                                         std::filesystem::path(nativeOutputPath)
                                             .lexically_normal()
                                             .string()
            ? "mupti_activity"
            : "mupti_csv";
    importCsv(file, runtimeEvents, usedRuntimeEvents, artifact, source);
  }

  if (const auto error = profiler.drainNativeCaptureError(); !error.empty()) {
    artifact.degradeReasons.push_back(error);
  }

  // MUPTI is an optional file importer. Keep the runtime-base profiler useful
  // even when no MUPTI export was requested or available.
  for (size_t i = 0; i < runtimeEvents.size(); ++i) {
    if (usedRuntimeEvents[i]) {
      continue;
    }
    VendorMetricAssociation association;
    association.runtimeEvent = runtimeEvents[i];
    association.state = VendorMetricState::Collected;
    association.source = "runtime_base_fallback";
    association.metrics["mthreads.runtime"] = std::string("host_timing");
    artifact.associations.push_back(std::move(association));
  }
  if (files.empty() && !plan.enabledVendorMetrics.empty()) {
    artifact.degradeReasons.push_back("No MUPTI activity export was found; "
                                      "runtime-base host timing was retained.");
  }
  return artifact;
}

} // namespace proton
