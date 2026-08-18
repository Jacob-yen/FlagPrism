#include "Profiler/Vendor/MthreadsAdapter.h"

#include "Profiler/Vendor/MthreadsProfiler.h"
#include "Utility/String.h"

#include <algorithm>
#include <cstdlib>
#include <sstream>

namespace proton {
namespace {

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

} // namespace

std::string MthreadsMetricsImporter::getName() const {
  return "mupti_importer";
}

VendorProfileArtifact MthreadsMetricsImporter::import(
    const SessionProfileMetadata &metadata,
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
  return {"launch_stats", "occupancy", "instruction", "memory",
          "throughput"};
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

  const auto supported = getSupportedVendorMetrics();
  for (const auto &request : requested.vendorMetrics) {
    auto metric = toLower(trim(request.name));
    if (metric == "launchstats") {
      metric = "launch_stats";
    }
    if (std::find(supported.begin(), supported.end(), metric) !=
        supported.end()) {
      if (std::find(plan.enabledVendorMetrics.begin(),
                    plan.enabledVendorMetrics.end(), metric) ==
          plan.enabledVendorMetrics.end()) {
        plan.enabledVendorMetrics.push_back(metric);
      }
    } else {
      plan.disabledVendorMetrics.push_back(request.name);
    }
  }
  if (!plan.disabledVendorMetrics.empty()) {
    plan.degradeReasons.push_back(
        "Unsupported MThreads vendor metrics: " +
        join(plan.disabledVendorMetrics));
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

std::unique_ptr<VendorMetricsImporter>
MthreadsAdapter::createImporter() const {
  return std::make_unique<MthreadsMetricsImporter>();
}

} // namespace proton
