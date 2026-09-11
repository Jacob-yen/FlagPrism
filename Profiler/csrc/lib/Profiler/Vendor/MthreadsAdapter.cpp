#include "Profiler/Vendor/MthreadsAdapter.h"

#include "Profiler/Vendor/MthreadsProfiler.h"
#include "Utility/String.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>

namespace proton {
namespace {

// TODO(FlagPrism): Enable and validate MCU integration when a compatible Moore
// Threads MCU, MUSA SDK, and driver test environment is available.
constexpr bool kMcuIntegrationEnabled = false;

std::string join(const std::vector<std::string> &items) {
  std::ostringstream stream;
  for (size_t i = 0; i < items.size(); ++i) {
    if (i != 0) {
      stream << ",";
    }
    stream << items[i];
  }
  return stream.str();
}

bool optionEnabled(const std::map<std::string, std::string> &options,
                   const std::string &name) {
  const auto it = options.find(name);
  if (it == options.end()) {
    return false;
  }
  const auto value = toLower(trim(it->second));
  return value == "1" || value == "true" || value == "on" || value == "yes";
}

bool hasMcuImportPath(const std::map<std::string, std::string> &options) {
  const auto it = options.find("mcu_import_path");
  return it != options.end() && !trim(it->second).empty();
}

std::string canonicalMetric(std::string metric) {
  metric = toLower(trim(metric));
  if (metric == "launchstats") {
    return "launch_stats";
  }
  if (metric == "resource_utilization" || metric == "resources") {
    return "resource_usage";
  }
  if (metric == "peak_bandwidth") {
    return "peak_memory_bandwidth";
  }
  if (metric == "instructions" || metric == "instruction") {
    return "instruction_count";
  }
  if (metric == "cycle_count") {
    return "cycles";
  }
  if (metric == "bandwidth") {
    return "memory_bandwidth";
  }
  if (metric == "sm_usage" || metric == "mp_utilization") {
    return "sm_utilization";
  }
  if (metric == "counters") {
    return "hardware_counters";
  }
  return metric;
}

} // namespace

std::string MthreadsMetricsImporter::getName() const {
  return "mupti_importer";
}

VendorProfileArtifact
MthreadsMetricsImporter::import(const SessionProfileMetadata &metadata,
                                const VendorProfilePlan &plan) const {
  auto artifact = MthreadsProfiler::importMthreadsOutput(metadata, plan);
  artifact.backend = metadata.backend;
  artifact.importer = getName();
  artifact.requestedMetrics = plan.requested.vendorMetrics;
  artifact.enabledMetrics = plan.enabledVendorMetrics;
  artifact.degradeReasons.insert(artifact.degradeReasons.begin(),
                                 plan.degradeReasons.begin(),
                                 plan.degradeReasons.end());
  return artifact;
}

const MthreadsAdapter &MthreadsAdapter::instance() {
  static const MthreadsAdapter adapter;
  return adapter;
}

std::string MthreadsAdapter::getName() const { return "mthreads"; }

DeviceType MthreadsAdapter::getDeviceType() const {
  return DeviceType::MTHREADS;
}

std::vector<std::string> MthreadsAdapter::getSupportedVendorMetrics() const {
  std::vector<std::string> metrics = {
      "launch_stats", "occupancy", "resource_usage", "peak_memory_bandwidth",
      "estimated_cycles"};
  if (kMcuIntegrationEnabled) {
    metrics.insert(metrics.end(),
                   {"instruction_count", "cycles", "memory_bandwidth",
                    "sm_utilization", "hardware_counters"});
  }
  return metrics;
}

VendorProfilePlan
MthreadsAdapter::makePlan(const VendorProfileOptions &options) const {
  VendorProfilePlan plan;
  plan.requested = options;
  plan.runtimeBaseEnabled = true;

  auto &requested = plan.requested;
  if (requested.adapterOptions.count("mupti_import_path") == 0) {
    const char *env = std::getenv("FLAGTREE_PROFILER_MTHREADS_IMPORT_PATH");
    if (env && *env) {
      requested.adapterOptions["mupti_import_path"] = env;
    }
  }
  if (requested.adapterOptions.count("mupti_output_path") == 0) {
    const char *env = std::getenv("FLAGTREE_PROFILER_MTHREADS_OUTPUT_PATH");
    if (env && *env) {
      requested.adapterOptions["mupti_output_path"] = env;
    }
  }

  const std::vector<std::string> mcuMetrics = {
      "instruction_count", "cycles", "memory_bandwidth", "sm_utilization",
      "hardware_counters"};
  const bool mcuRequested =
      std::any_of(requested.vendorMetrics.begin(),
                  requested.vendorMetrics.end(), [&](const auto &request) {
                    const auto name = canonicalMetric(request.name);
                    return std::find(mcuMetrics.begin(), mcuMetrics.end(),
                                     name) != mcuMetrics.end();
                  }) ||
      optionEnabled(requested.adapterOptions, "mcu_external") ||
      hasMcuImportPath(requested.adapterOptions);
  if (!kMcuIntegrationEnabled) {
    requested.adapterOptions.erase("mcu_external");
    requested.adapterOptions.erase("mcu_import_path");
  }
  const auto supported = getSupportedVendorMetrics();
  const auto &adapterOptions = requested.adapterOptions;
  const bool externalMcu =
      kMcuIntegrationEnabled &&
      (optionEnabled(adapterOptions, "mcu_external") ||
       hasMcuImportPath(adapterOptions));
  const bool activityCapture =
      optionEnabled(adapterOptions, "mupti_activity") || externalMcu;
  std::vector<std::string> unsupportedMetrics;
  std::vector<std::string> unavailableMetrics;
  for (const auto &request : requested.vendorMetrics) {
    const auto metric = canonicalMetric(request.name);
    const bool requiresMcu = std::find(mcuMetrics.begin(), mcuMetrics.end(),
                                       metric) != mcuMetrics.end();
    const bool known =
        std::find(supported.begin(), supported.end(), metric) !=
            supported.end() ||
        requiresMcu;
    const bool available = known && (!requiresMcu || externalMcu) &&
                           (metric != "estimated_cycles" || activityCapture);
    if (available) {
      if (std::find(plan.enabledVendorMetrics.begin(),
                    plan.enabledVendorMetrics.end(),
                    metric) == plan.enabledVendorMetrics.end()) {
        plan.enabledVendorMetrics.push_back(metric);
      }
    } else {
      plan.disabledVendorMetrics.push_back(request.name);
      if (known) {
        unavailableMetrics.push_back(metric);
      } else {
        unsupportedMetrics.push_back(request.name);
      }
    }
  }
  if (!unsupportedMetrics.empty()) {
    plan.degradeReasons.push_back("Unsupported MThreads vendor metrics: " +
                                  join(unsupportedMetrics));
  }
  if (!kMcuIntegrationEnabled && mcuRequested) {
    plan.degradeReasons.push_back(
        "MThreads MCU integration is frozen pending validation on a compatible "
        "Moore Threads environment.");
  } else if (!externalMcu && mcuRequested) {
    plan.degradeReasons.push_back(
        "MThreads hardware-counter metrics require the Moore Perf Compute "
        "wrapper or an MCU CSV import path.");
  }
  if (!activityCapture &&
      std::find(unavailableMetrics.begin(), unavailableMetrics.end(),
                "estimated_cycles") != unavailableMetrics.end()) {
    plan.degradeReasons.push_back(
        "MThreads estimated_cycles requires MUPTI activity timestamps or an "
        "imported vendor export.");
  }
  // With no import path, SessionManager supplies a per-session
  // mupti_output_path. MthreadsProfiler then captures native MUPTI launch
  // records, writes that CSV during doStop(), and the importer consumes it
  // below. An explicit mupti_import_path remains supported as import-only
  // mode.
  return plan;
}

Profiler *MthreadsAdapter::getRuntimeProfiler() const {
  return &MthreadsProfiler::instance();
}

std::unique_ptr<VendorMetricsImporter> MthreadsAdapter::createImporter() const {
  return std::make_unique<MthreadsMetricsImporter>();
}

} // namespace proton
