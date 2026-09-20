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
#include <limits>
#include <map>
#include <new>
#include <optional>
#include <set>
#include <sstream>

#if defined(FLAGTREE_MTHREADS_HAS_MUPTI)
#include <dlfcn.h>
#include <mupti.h>
#include <musa_runtime_api.h>
#endif

namespace proton {
namespace {

// TODO(FlagPrism): Enable and validate MCU integration when a compatible Moore
// Threads MCU, MUSA SDK, and driver test environment is available.
constexpr bool kMcuIntegrationEnabled = false;

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

struct MthreadsMusaApi {
  using Result = int;
  using Device = int;
  using DriverInit = Result (*)(unsigned int);
  using DriverDeviceGet = Result (*)(Device *, int);
  using DriverDeviceGetAttribute = Result (*)(int *, int, Device);
  using DriverFuncGetAttribute = Result (*)(int *, int, MUfunction);
  using DriverOccupancy = Result (*)(int *, MUfunction, int, size_t);
  using RuntimeFuncGetAttributes = Result (*)(musaFuncAttributes *,
                                              const void *);
  using RuntimeOccupancy = Result (*)(int *, const void *, int, size_t,
                                      unsigned int);

  void *driverLibrary = nullptr;
  void *runtimeLibrary = nullptr;
  DriverInit driverInit = nullptr;
  DriverDeviceGet driverDeviceGet = nullptr;
  DriverDeviceGetAttribute driverDeviceGetAttribute = nullptr;
  DriverFuncGetAttribute driverFuncGetAttribute = nullptr;
  DriverOccupancy driverOccupancy = nullptr;
  RuntimeFuncGetAttributes runtimeFuncGetAttributes = nullptr;
  RuntimeOccupancy runtimeOccupancy = nullptr;
  bool loadAttempted = false;

  // Keep the driver/runtime loaded until process exit. Some MUSA runtimes keep
  // function pointers in callbacks that can outlive the profiler session.
  ~MthreadsMusaApi() = default;

  bool load() {
    if (loadAttempted) {
      return driverLibrary != nullptr || runtimeLibrary != nullptr;
    }
    loadAttempted = true;

    std::vector<std::string> driverCandidates;
    if (const char *env =
            std::getenv("FLAGTREE_PROFILER_MTHREADS_DRIVER_LIBRARY")) {
      if (*env) {
        driverCandidates.emplace_back(env);
      }
    }
    driverCandidates.emplace_back("libmusa.so.1");
    driverCandidates.emplace_back("libmusa.so");
    driverCandidates.emplace_back("/usr/lib/x86_64-linux-gnu/libmusa.so.1");
    for (const auto &candidate : driverCandidates) {
      driverLibrary = dlopen(candidate.c_str(), RTLD_LOCAL | RTLD_LAZY);
      if (driverLibrary) {
        break;
      }
    }
    if (driverLibrary) {
      driverInit = reinterpret_cast<DriverInit>(dlsym(driverLibrary, "muInit"));
      driverDeviceGet = reinterpret_cast<DriverDeviceGet>(
          dlsym(driverLibrary, "muDeviceGet"));
      driverDeviceGetAttribute = reinterpret_cast<DriverDeviceGetAttribute>(
          dlsym(driverLibrary, "muDeviceGetAttribute"));
      driverFuncGetAttribute = reinterpret_cast<DriverFuncGetAttribute>(
          dlsym(driverLibrary, "muFuncGetAttribute"));
      driverOccupancy = reinterpret_cast<DriverOccupancy>(
          dlsym(driverLibrary, "muOccupancyMaxActiveBlocksPerMultiprocessor"));
    }

    std::vector<std::string> runtimeCandidates;
    if (const char *musaHome = std::getenv("MUSA_HOME")) {
      if (*musaHome) {
        runtimeCandidates.emplace_back(std::string(musaHome) +
                                       "/lib/libmusart.so");
        runtimeCandidates.emplace_back(std::string(musaHome) +
                                       "/lib64/libmusart.so");
      }
    }
    runtimeCandidates.emplace_back("libmusart.so.4");
    runtimeCandidates.emplace_back("libmusart.so");
    runtimeCandidates.emplace_back("/usr/local/musa/lib/libmusart.so");
    for (const auto &candidate : runtimeCandidates) {
      runtimeLibrary = dlopen(candidate.c_str(), RTLD_LOCAL | RTLD_LAZY);
      if (runtimeLibrary) {
        break;
      }
    }
    if (runtimeLibrary) {
      runtimeFuncGetAttributes = reinterpret_cast<RuntimeFuncGetAttributes>(
          dlsym(runtimeLibrary, "musaFuncGetAttributes"));
      runtimeOccupancy = reinterpret_cast<RuntimeOccupancy>(
          dlsym(runtimeLibrary,
                "musaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags"));
    }

    return driverLibrary != nullptr || runtimeLibrary != nullptr;
  }
};

MthreadsMusaApi &musaApi() {
  static MthreadsMusaApi api;
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

std::string normalizeMetricKey(std::string value) {
  std::string result;
  bool pendingSeparator = false;
  for (char ch : toLower(trim(value))) {
    if (std::isalnum(static_cast<unsigned char>(ch)) || ch == '.' ||
        ch == '_') {
      if (pendingSeparator && !result.empty() && result.back() != '_' &&
          ch != '.' && ch != '_') {
        result.push_back('_');
      }
      result.push_back(ch);
      pendingSeparator = false;
    } else {
      pendingSeparator = true;
    }
  }
  while (!result.empty() && result.back() == '_') {
    result.pop_back();
  }
  return result;
}

std::string canonicalMetricName(const std::string &header) {
  static const std::map<std::string, std::string> names = {
      {"threadsperblock", "threads_per_block"},
      {"activeblockspersm", "active_blocks_per_sm"},
      {"activewarpspersm", "active_warps_per_sm"},
      {"maxwarpspersm", "max_warps_per_sm"},
      {"theoreticaloccupancypct", "theoretical_occupancy_pct"},
      {"occupancysource", "occupancy_source"},
      {"registersperthread", "registers_per_thread"},
      {"registersperblock", "registers_per_block"},
      {"registerspersm", "registers_per_sm"},
      {"registersperblockpct", "registers_per_block_pct"},
      {"staticsharedmemory", "static_shared_memory"},
      {"dynamicsharedmemory", "dynamic_shared_memory"},
      {"sharedmemoryperblock", "shared_memory_per_block"},
      {"sharedmemorypersm", "shared_memory_per_sm"},
      {"sharedmemoryperblockpct", "shared_memory_per_block_pct"},
      {"residentregisterspersm", "resident_registers_per_sm"},
      {"residentregisterutilizationpct", "resident_register_utilization_pct"},
      {"residentsharedmemorypersm", "resident_shared_memory_per_sm"},
      {"residentsharedmemoryutilizationpct",
       "resident_shared_memory_utilization_pct"},
      {"residentthreadspersm", "resident_threads_per_sm"},
      {"maxblockspersm", "max_blocks_per_sm"},
      {"localmemoryperthread", "local_memory_per_thread"},
      {"localmemorytotal", "local_memory_total"},
      {"smcount", "sm_count"},
      {"deviceclockratekhz", "device_clock_rate_khz"},
      {"peakmemorybandwidthgbps", "peak_memory_bandwidth_gbps"},
      {"estimatedelapsedcycles", "estimated_elapsed_cycles"},
      {"deviceid", "device_id"},
      {"streamid", "stream_id"},
      {"correlationid", "correlation_id"},
      {"gridid", "grid_id"},
      {"gridx", "grid_x"},
      {"gridy", "grid_y"},
      {"gridz", "grid_z"},
      {"blockx", "block_x"},
      {"blocky", "block_y"},
      {"blockz", "block_z"},
      {"capturesource", "capture_source"},
  };
  const auto normalized = normalizeColumn(header);
  const auto it = names.find(normalized);
  return it == names.end() ? normalizeMetricKey(header) : it->second;
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
       {"mupti_import_path", "mthreads_import_path", "mcu_import_path",
        "vendor_import_path", "output_path", "mupti_output_path",
        "mthreads_output_path"}) {
    if (!kMcuIntegrationEnabled && std::string_view(key) == "mcu_import_path") {
      continue;
    }
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
  const auto metricIndex =
      findColumn(headers, {"metric_name", "metric", "counter_name", "counter"});
  const auto valueIndex =
      findColumn(headers, {"metric_value", "value", "counter_value"});
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
      const auto duration = association.runtimeEvent.endTimeNs -
                            association.runtimeEvent.startTimeNs;
      if (association.source == "mupti_callback") {
        association.metrics["mthreads.launch_api_duration_ns"] = duration;
      } else {
        association.metrics["mthreads.kernel_duration_ns"] = duration;
      }
    }
    if (metricIndex && valueIndex) {
      const auto metricName = canonicalMetricName(cell(row, metricIndex));
      if (!metricName.empty()) {
        if (auto value = parseMetric(cell(row, valueIndex))) {
          association.metrics["mthreads." + metricName] = *value;
        }
      }
    }
    for (size_t i = 0; i < headers.size() && i < row.size(); ++i) {
      if (i == *nameIndex || (startIndex && i == *startIndex) ||
          (endIndex && i == *endIndex) ||
          (durationIndex && i == *durationIndex) ||
          (metricIndex && i == *metricIndex) ||
          (valueIndex && i == *valueIndex)) {
        continue;
      }
      if (auto value = parseMetric(row[i])) {
        const auto metricName = canonicalMetricName(headers[i]);
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

bool artifactHasMetric(const VendorProfileArtifact &artifact,
                       const std::string &capability) {
  for (const auto &association : artifact.associations) {
    for (const auto &[name, value] : association.metrics) {
      (void)value;
      if (capability == "launch_stats" &&
          (name == "mthreads.threads_per_block" || name == "mthreads.grid_x")) {
        return true;
      }
      if (capability == "occupancy" &&
          name == "mthreads.theoretical_occupancy_pct") {
        return true;
      }
      if (capability == "resource_usage" &&
          (name == "mthreads.registers_per_thread" ||
           name == "mthreads.resident_register_utilization_pct" ||
           name == "mthreads.resident_shared_memory_utilization_pct")) {
        return true;
      }
      if (capability == "peak_memory_bandwidth" &&
          name == "mthreads.peak_memory_bandwidth_gbps") {
        return true;
      }
      if (capability == "estimated_cycles" &&
          name == "mthreads.estimated_elapsed_cycles") {
        return true;
      }
      if (capability == "instruction_count" &&
          (name.find("instruction") != std::string::npos ||
           name.find("__inst_") != std::string::npos)) {
        return true;
      }
      if (capability == "cycles" && name.find("cycle") != std::string::npos &&
          name.find("estimated") == std::string::npos) {
        return true;
      }
      if (capability == "memory_bandwidth" &&
          (name.find("throughput") != std::string::npos ||
           (name.find("bytes") != std::string::npos &&
            name.find("per_second") != std::string::npos) ||
           (name.find("bandwidth") != std::string::npos &&
            name.find("peak_memory_bandwidth") == std::string::npos))) {
        return true;
      }
      if (capability == "sm_utilization" &&
          (name.find("sm_efficiency") != std::string::npos ||
           name.find("sm__throughput") != std::string::npos ||
           name.find("mp__throughput") != std::string::npos ||
           name.find("mp_utilization") != std::string::npos)) {
        return true;
      }
      if (capability == "hardware_counters" &&
          association.source == "mcu_csv" && name.rfind("mthreads.", 0) == 0 &&
          name != "mthreads.source_file") {
        return true;
      }
    }
  }
  return false;
}

void reportMissingEnabledMetrics(VendorProfileArtifact &artifact) {
  for (const auto &metric : artifact.enabledMetrics) {
    if (!artifactHasMetric(artifact, metric)) {
      artifact.degradeReasons.push_back(
          "MThreads metric '" + metric +
          "' was enabled but the capture produced no value for it.");
    }
  }
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
    auto &musa = musaApi();
    musa.load();
    if (domain == MUPTI_CB_DOMAIN_DRIVER_API && callback->functionParams &&
        (callbackId == MUPTI_DRIVER_TRACE_CBID_muLaunchKernel ||
         callbackId == MUPTI_DRIVER_TRACE_CBID_muLaunchKernel_ptsz)) {
      // The regular and per-thread-default-stream parameter layouts match.
      const auto *launch =
          static_cast<const muLaunchKernel_params *>(callback->functionParams);
      event.gridX = static_cast<int32_t>(launch->gridDimX);
      event.gridY = static_cast<int32_t>(launch->gridDimY);
      event.gridZ = static_cast<int32_t>(launch->gridDimZ);
      event.blockX = static_cast<int32_t>(launch->blockDimX);
      event.blockY = static_cast<int32_t>(launch->blockDimY);
      event.blockZ = static_cast<int32_t>(launch->blockDimZ);
      event.dynamicSharedMemory = static_cast<int32_t>(launch->sharedMemBytes);
      event.streamId =
          static_cast<uint32_t>(reinterpret_cast<uintptr_t>(launch->hStream));
      if (musa.driverFuncGetAttribute) {
        int value = 0;
        if (musa.driverFuncGetAttribute(&value, MU_FUNC_ATTRIBUTE_NUM_REGS,
                                        launch->f) == 0 &&
            value > 0) {
          event.registersPerThread = static_cast<uint16_t>(value);
        }
        value = 0;
        if (musa.driverFuncGetAttribute(
                &value, MU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, launch->f) == 0 &&
            value > 0) {
          event.staticSharedMemory = value;
        }
        value = 0;
        if (musa.driverFuncGetAttribute(
                &value, MU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, launch->f) == 0 &&
            value > 0) {
          event.localMemoryPerThread = static_cast<uint32_t>(value);
        }
      }
      if (musa.driverOccupancy) {
        int activeBlocks = 0;
        const auto blockSize = event.blockX * event.blockY * event.blockZ;
        if (blockSize > 0 &&
            musa.driverOccupancy(&activeBlocks, launch->f, blockSize,
                                 launch->sharedMemBytes) == 0 &&
            activeBlocks > 0) {
          event.activeBlocksPerSm = static_cast<uint32_t>(activeBlocks);
          event.occupancySource = "musa_driver_occupancy_api";
        }
      }
    } else if (domain == MUPTI_CB_DOMAIN_RUNTIME_API &&
               callback->functionParams &&
               (callbackId == MUPTI_RUNTIME_TRACE_CBID_musaLaunchKernel_v7000 ||
                callbackId ==
                    MUPTI_RUNTIME_TRACE_CBID_musaLaunchKernel_ptsz_v7000)) {
      // The regular and per-thread-default-stream parameter layouts match.
      const auto *launch = static_cast<const musaLaunchKernel_v7000_params *>(
          callback->functionParams);
      event.gridX = static_cast<int32_t>(launch->gridDim.x);
      event.gridY = static_cast<int32_t>(launch->gridDim.y);
      event.gridZ = static_cast<int32_t>(launch->gridDim.z);
      event.blockX = static_cast<int32_t>(launch->blockDim.x);
      event.blockY = static_cast<int32_t>(launch->blockDim.y);
      event.blockZ = static_cast<int32_t>(launch->blockDim.z);
      event.dynamicSharedMemory = static_cast<int32_t>(launch->sharedMem);
      event.streamId =
          static_cast<uint32_t>(reinterpret_cast<uintptr_t>(launch->stream));
      if (musa.runtimeFuncGetAttributes) {
        musaFuncAttributes attributes{};
        if (musa.runtimeFuncGetAttributes(&attributes, launch->func) == 0) {
          event.registersPerThread =
              static_cast<uint16_t>(std::max(attributes.numRegs, 0));
          event.staticSharedMemory =
              static_cast<int32_t>(attributes.sharedSizeBytes);
          event.localMemoryPerThread =
              static_cast<uint32_t>(attributes.localSizeBytes);
        }
      }
      if (musa.runtimeOccupancy) {
        int activeBlocks = 0;
        const auto blockSize = event.blockX * event.blockY * event.blockZ;
        if (blockSize > 0 &&
            musa.runtimeOccupancy(&activeBlocks, launch->func, blockSize,
                                  launch->sharedMem, 0) == 0 &&
            activeBlocks > 0) {
          event.activeBlocksPerSm = static_cast<uint32_t>(activeBlocks);
          event.occupancySource = "musa_runtime_occupancy_api";
        }
      }
    }
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
      // Nested runtime/driver callbacks can share a correlation ID. The inner
      // driver callback owns the precise launch record, so do not fabricate a
      // second exit-only runtime record after it has consumed the entry.
      return;
    }
    event.deviceId = profiler.deviceId;
    event.endTimeNs = nowNs();
    profiler.nativeKernelEvents.push_back(std::move(event));
  }
#else
  (void)callbackId;
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
    if (event.startTimeNs == 0 || event.endTimeNs <= event.startTimeNs) {
      // MUSA 4.3 can return activity records after MT-Perf connection
      // failures, but their zero timestamps are not device measurements.
      // Ignore them so the valid callback fallback is not suppressed.
      std::lock_guard<std::mutex> lock(mutex);
      ++nativeInvalidActivityRecords;
      continue;
    }
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

void MthreadsProfiler::queryDeviceLimits() {
  warpSize = 0;
  multiprocessorCount = 0;
  maxThreadsPerMultiprocessor = 0;
  maxBlocksPerMultiprocessor = 0;
  registersPerMultiprocessor = 0;
  sharedMemoryPerMultiprocessor = 0;
  clockRateKhz = 0;
  memoryClockRateKhz = 0;
  memoryBusWidthBits = 0;
#if defined(FLAGTREE_MTHREADS_HAS_MUPTI)
  auto &musa = musaApi();
  if (!musa.load() || !musa.driverInit || !musa.driverDeviceGet ||
      !musa.driverDeviceGetAttribute || musa.driverInit(0) != 0) {
    return;
  }
  MthreadsMusaApi::Device device = 0;
  if (musa.driverDeviceGet(&device, static_cast<int>(deviceId)) != 0) {
    return;
  }
  auto query = [&](int attribute) -> uint64_t {
    int value = 0;
    if (musa.driverDeviceGetAttribute(&value, attribute, device) != 0 ||
        value <= 0) {
      return 0;
    }
    return static_cast<uint64_t>(value);
  };
  warpSize = static_cast<uint32_t>(query(MU_DEVICE_ATTRIBUTE_WARP_SIZE));
  multiprocessorCount =
      static_cast<uint32_t>(query(MU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT));
  maxThreadsPerMultiprocessor = static_cast<uint32_t>(
      query(MU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR));
  maxBlocksPerMultiprocessor = static_cast<uint32_t>(
      query(MU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR));
  registersPerMultiprocessor =
      query(MU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR);
  sharedMemoryPerMultiprocessor =
      query(MU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR);
  clockRateKhz = query(MU_DEVICE_ATTRIBUTE_CLOCK_RATE);
  memoryClockRateKhz = query(MU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE);
  memoryBusWidthBits = query(MU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH);
#endif
}

void MthreadsProfiler::enrichNativeEvent(NativeKernelEvent &event) const {
  const auto threadsPerBlock =
      static_cast<uint64_t>(std::max(event.blockX, 0)) *
      static_cast<uint64_t>(std::max(event.blockY, 0)) *
      static_cast<uint64_t>(std::max(event.blockZ, 0));
  if (event.localMemoryTotal == 0 && event.localMemoryPerThread != 0 &&
      threadsPerBlock != 0) {
    event.localMemoryTotal = static_cast<uint32_t>(
        std::min<uint64_t>(event.localMemoryPerThread * threadsPerBlock,
                           std::numeric_limits<uint32_t>::max()));
  }
}

void MthreadsProfiler::writeMuptiOutput() {
  std::vector<NativeKernelEvent> events;
  std::string outputPath;
  bool activityRequested = false;
  size_t activityRecords = 0;
  size_t invalidActivityRecords = 0;
  {
    std::lock_guard<std::mutex> lock(mutex);
    events = nativeKernelEvents;
    outputPath = nativeOutputPath;
    activityRequested = nativeActivityRequested;
    activityRecords = nativeActivityRecords;
    invalidActivityRecords = nativeInvalidActivityRecords;
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
        ", activity_records=" + std::to_string(nativeActivityRecords) +
        ", invalid_activity_records=" +
        std::to_string(nativeInvalidActivityRecords) + ")";
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
  auto writeOptionalInteger = [&](bool available, uint64_t value) {
    if (available) {
      output << value;
    }
  };
  auto writeOptionalDouble = [&](bool available, double value) {
    if (available) {
      output << value;
    }
  };
  output << "kernel_name,start_time_ns,end_time_ns,device_id,stream_id,"
            "correlation_id,grid_id,grid_x,grid_y,grid_z,block_x,block_y,"
            "block_z,registers_per_thread,static_shared_memory,"
            "dynamic_shared_memory,local_memory_per_thread,"
            "local_memory_total,threads_per_block,active_blocks_per_sm,"
            "active_warps_per_sm,max_warps_per_sm,"
            "theoretical_occupancy_pct,occupancy_source,"
            "registers_per_block,registers_per_sm,"
            "registers_per_block_pct,shared_memory_per_block,"
            "shared_memory_per_sm,shared_memory_per_block_pct,"
            "resident_registers_per_sm,resident_register_utilization_pct,"
            "resident_shared_memory_per_sm,"
            "resident_shared_memory_utilization_pct,resident_threads_per_sm,"
            "max_blocks_per_sm,sm_count,device_clock_rate_khz,"
            "peak_memory_bandwidth_gbps,"
            "estimated_elapsed_cycles,capture_source\n";

  std::set<std::string> seen;
  std::set<uint32_t> activityCorrelations;
  std::unordered_map<uint32_t, NativeKernelEvent> callbackByCorrelation;
  for (const auto &event : events) {
    if (!event.callbackEvent && event.correlationId != 0) {
      activityCorrelations.insert(event.correlationId);
    } else if (event.callbackEvent && event.correlationId != 0) {
      callbackByCorrelation[event.correlationId] = event;
    }
  }
  size_t written = 0;
  for (const auto &rawEvent : events) {
    auto event = rawEvent;
    if (event.callbackEvent && event.correlationId != 0 &&
        activityCorrelations.count(event.correlationId) != 0) {
      continue;
    }
    if (!event.callbackEvent && event.correlationId != 0) {
      const auto callback = callbackByCorrelation.find(event.correlationId);
      if (callback != callbackByCorrelation.end()) {
        const auto &launch = callback->second;
        event.activeBlocksPerSm = launch.activeBlocksPerSm;
        event.occupancySource = launch.occupancySource;
        if (event.gridX == 0) {
          event.gridX = launch.gridX;
          event.gridY = launch.gridY;
          event.gridZ = launch.gridZ;
        }
        if (event.blockX == 0) {
          event.blockX = launch.blockX;
          event.blockY = launch.blockY;
          event.blockZ = launch.blockZ;
        }
      }
    }
    enrichNativeEvent(event);
    const auto key = event.name + ":" + std::to_string(event.startTimeNs) +
                     ":" + std::to_string(event.endTimeNs) + ":" +
                     std::to_string(event.correlationId);
    if (!seen.insert(key).second) {
      continue;
    }
    const auto threadsPerBlock =
        static_cast<uint64_t>(std::max(event.blockX, 0)) *
        static_cast<uint64_t>(std::max(event.blockY, 0)) *
        static_cast<uint64_t>(std::max(event.blockZ, 0));
    const auto activeWarpsPerSm =
        warpSize == 0 ? 0
                      : (static_cast<uint64_t>(event.activeBlocksPerSm) *
                             threadsPerBlock +
                         warpSize - 1) /
                            warpSize;
    const auto maxWarpsPerSm =
        warpSize == 0 ? 0 : maxThreadsPerMultiprocessor / warpSize;
    const auto occupancyPct =
        maxThreadsPerMultiprocessor == 0
            ? 0.0
            : 100.0 * static_cast<double>(event.activeBlocksPerSm) *
                  static_cast<double>(threadsPerBlock) /
                  static_cast<double>(maxThreadsPerMultiprocessor);
    const auto registersPerBlock =
        static_cast<uint64_t>(event.registersPerThread) * threadsPerBlock;
    const auto registersPerBlockPct =
        registersPerMultiprocessor == 0
            ? 0.0
            : 100.0 * static_cast<double>(registersPerBlock) /
                  static_cast<double>(registersPerMultiprocessor);
    const auto sharedMemoryPerBlock =
        static_cast<uint64_t>(std::max(event.staticSharedMemory, 0)) +
        static_cast<uint64_t>(std::max(event.dynamicSharedMemory, 0));
    const auto sharedMemoryPerBlockPct =
        sharedMemoryPerMultiprocessor == 0
            ? 0.0
            : 100.0 * static_cast<double>(sharedMemoryPerBlock) /
                  static_cast<double>(sharedMemoryPerMultiprocessor);
    const auto residentThreadsPerSm =
        static_cast<uint64_t>(event.activeBlocksPerSm) * threadsPerBlock;
    const auto residentRegistersPerSm =
        static_cast<uint64_t>(event.activeBlocksPerSm) * registersPerBlock;
    const auto residentSharedMemoryPerSm =
        static_cast<uint64_t>(event.activeBlocksPerSm) * sharedMemoryPerBlock;
    const auto residentRegisterUtilizationPct =
        registersPerMultiprocessor == 0
            ? 0.0
            : std::min(100.0,
                       100.0 * static_cast<double>(residentRegistersPerSm) /
                           static_cast<double>(registersPerMultiprocessor));
    const auto residentSharedMemoryUtilizationPct =
        sharedMemoryPerMultiprocessor == 0
            ? 0.0
            : std::min(100.0,
                       100.0 * static_cast<double>(residentSharedMemoryPerSm) /
                           static_cast<double>(sharedMemoryPerMultiprocessor));
    const auto peakMemoryBandwidthGbps =
        memoryClockRateKhz == 0 || memoryBusWidthBits == 0
            ? 0.0
            : static_cast<double>(memoryClockRateKhz) *
                  static_cast<double>(memoryBusWidthBits) / 4'000'000.0;
    const auto estimatedElapsedCycles =
        event.callbackEvent || event.endTimeNs <= event.startTimeNs ||
                clockRateKhz == 0
            ? 0.0
            : static_cast<double>(event.endTimeNs - event.startTimeNs) *
                  static_cast<double>(clockRateKhz) / 1'000'000.0;
    const bool launchShapeAvailable = threadsPerBlock != 0;
    const bool occupancyAvailable = event.activeBlocksPerSm != 0 &&
                                    maxThreadsPerMultiprocessor != 0 &&
                                    warpSize != 0;
    const bool registersAvailable =
        event.registersPerThread != 0 && launchShapeAvailable;
    const bool sharedMemoryLimitAvailable =
        sharedMemoryPerMultiprocessor != 0 && launchShapeAvailable;
    const bool peakBandwidthAvailable =
        memoryClockRateKhz != 0 && memoryBusWidthBits != 0;
    const bool estimatedCyclesAvailable = !event.callbackEvent &&
                                          event.endTimeNs > event.startTimeNs &&
                                          clockRateKhz != 0;
    writeCsvString(event.name);
    output << ',' << event.startTimeNs << ',' << event.endTimeNs << ','
           << event.deviceId << ',' << event.streamId << ','
           << event.correlationId << ',' << event.gridId << ',' << event.gridX
           << ',' << event.gridY << ',' << event.gridZ << ',' << event.blockX
           << ',' << event.blockY << ',' << event.blockZ << ','
           << event.registersPerThread << ',' << event.staticSharedMemory << ','
           << event.dynamicSharedMemory << ',' << event.localMemoryPerThread
           << ',' << event.localMemoryTotal << ',';
    writeOptionalInteger(launchShapeAvailable, threadsPerBlock);
    output << ',';
    writeOptionalInteger(occupancyAvailable, event.activeBlocksPerSm);
    output << ',';
    writeOptionalInteger(occupancyAvailable, activeWarpsPerSm);
    output << ',';
    writeOptionalInteger(occupancyAvailable, maxWarpsPerSm);
    output << ',';
    writeOptionalDouble(occupancyAvailable, occupancyPct);
    output << ',';
    writeCsvString(event.occupancySource);
    output << ',';
    writeOptionalInteger(registersAvailable, registersPerBlock);
    output << ',';
    writeOptionalInteger(registersAvailable && registersPerMultiprocessor != 0,
                         registersPerMultiprocessor);
    output << ',';
    writeOptionalDouble(registersAvailable && registersPerMultiprocessor != 0,
                        registersPerBlockPct);
    output << ',';
    writeOptionalInteger(sharedMemoryLimitAvailable, sharedMemoryPerBlock);
    output << ',';
    writeOptionalInteger(sharedMemoryLimitAvailable,
                         sharedMemoryPerMultiprocessor);
    output << ',';
    writeOptionalDouble(sharedMemoryLimitAvailable, sharedMemoryPerBlockPct);
    output << ',';
    writeOptionalInteger(occupancyAvailable && registersAvailable,
                         residentRegistersPerSm);
    output << ',';
    writeOptionalDouble(occupancyAvailable && registersAvailable &&
                            registersPerMultiprocessor != 0,
                        residentRegisterUtilizationPct);
    output << ',';
    writeOptionalInteger(occupancyAvailable && sharedMemoryLimitAvailable,
                         residentSharedMemoryPerSm);
    output << ',';
    writeOptionalDouble(occupancyAvailable && sharedMemoryLimitAvailable,
                        residentSharedMemoryUtilizationPct);
    output << ',';
    writeOptionalInteger(occupancyAvailable, residentThreadsPerSm);
    output << ',';
    writeOptionalInteger(maxBlocksPerMultiprocessor != 0,
                         maxBlocksPerMultiprocessor);
    output << ',';
    writeOptionalInteger(multiprocessorCount != 0, multiprocessorCount);
    output << ',';
    writeOptionalInteger(clockRateKhz != 0, clockRateKhz);
    output << ',';
    writeOptionalDouble(peakBandwidthAvailable, peakMemoryBandwidthGbps);
    output << ',';
    writeOptionalDouble(estimatedCyclesAvailable, estimatedElapsedCycles);
    output << ',';
    writeCsvString(event.callbackEvent ? "mupti_callback" : "mupti_activity");
    output << '\n';
    ++written;
  }
  if (output.good() && written != 0) {
    std::lock_guard<std::mutex> lock(mutex);
    lastNativeOutputPath = outputPath;
    if (activityRequested && activityRecords == 0 &&
        invalidActivityRecords != 0 && nativeCaptureError.empty()) {
      nativeCaptureError =
          "MUPTI activity returned " + std::to_string(invalidActivityRecords) +
          " record(s) without valid device timestamps; callback launch data "
          "was retained.";
    }
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
    nativeInvalidActivityRecords = 0;
  }
  queryDeviceLimits();
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
               (kMcuIntegrationEnabled && key == "mcu_import_path") ||
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
  const bool mcuImport =
      kMcuIntegrationEnabled &&
      plan.requested.adapterOptions.count("mcu_import_path") != 0;
  for (const auto &file : files) {
    artifact.rawInputs.push_back(file.string());
    const auto source =
        !nativeOutputPath.empty() && file.lexically_normal().string() ==
                                         std::filesystem::path(nativeOutputPath)
                                             .lexically_normal()
                                             .string()
            ? "mupti_activity"
            : (mcuImport ? "mcu_csv" : "mupti_csv");
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
  reportMissingEnabledMetrics(artifact);
  return artifact;
}

} // namespace proton
