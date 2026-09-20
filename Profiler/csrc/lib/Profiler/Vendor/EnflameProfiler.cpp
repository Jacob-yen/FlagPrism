#include "Profiler/Vendor/EnflameProfiler.h"
#include "Context/Context.h"
#include "Profiler/Profiler.h"
#include <cstdlib>
#include <mutex>
#include <set>
#include <stdexcept>
#include <tops/tops_runtime.h>
#include <topspti_activity.h>
#include <topspti_callbacks.h>
#include <unordered_map>

namespace proton {
namespace {
void check(TopsptiResult result, const char *operation) {
  if (result != TOPSPTI_SUCCESS)
    throw std::runtime_error(std::string(operation) +
                             " failed: TOPSPTI status " +
                             std::to_string(static_cast<int>(result)));
}
thread_local std::vector<RuntimeTraceEventKey> activeScopes;
class EnflameProfiler final : public Profiler,
                              public OpInterface,
                              public Singleton<EnflameProfiler> {
public:
  VendorProfileArtifact collect(const SessionProfileMetadata &metadata,
                                const VendorProfilePlan &plan) {
    std::lock_guard<std::mutex> lock(eventsMutex);
    VendorProfileArtifact artifact;
    artifact.backend = metadata.backend;
    artifact.importer = "topspti_activity";
    artifact.requestedMetrics = plan.requested.vendorMetrics;
    for (auto event : events) {
      auto it = launches.find(event.correlationId);
      // Activity buffers arrive asynchronously. Attribute each launch to the
      // sessions active at API entry, not at buffer completion/import time.
      if (it == launches.end() ||
          !it->second.sessions.count(metadata.sessionName))
        continue;
      if (it->second.scope.scopeId) {
        event.scopeId = it->second.scope.scopeId;
        event.opName = it->second.scope.opName;
      }
      VendorMetricAssociation association;
      association.runtimeEvent = std::move(event);
      association.source = "topspti_activity";
      association.state = VendorMetricState::Collected;
      artifact.associations.push_back(std::move(association));
    }
    // Retain events across pause/resume; the next start begins a fresh capture.
    return artifact;
  }

private:
  std::mutex eventsMutex;
  std::vector<RuntimeTraceEventKey> events;
  struct Launch {
    RuntimeTraceEventKey scope;
    std::set<std::string> sessions;
  };
  std::unordered_map<uint32_t, Launch> launches;
  Topspti_SubscriberHandle subscriber = nullptr;
  bool enabled = false;
  std::string callbackError;
  void startOp(const Scope &scope) override {
    RuntimeTraceEventKey event;
    event.scopeId = scope.scopeId;
    event.opName = scope.name;
    activeScopes.push_back(std::move(event));
  }
  void stopOp(const Scope &) override {
    if (!activeScopes.empty())
      activeScopes.pop_back();
  }
  static void callback(void *, Topspti_CallbackDomain, Topspti_CallbackId,
                       const void *raw) {
    if (!raw)
      return;
    auto *data = static_cast<const Topspti_CallbackData *>(raw);
    if (data->callbackSite != TOPSPTI_API_ENTER)
      return;
    auto &self = instance();
    Launch launch;
    if (!activeScopes.empty())
      launch.scope = activeScopes.back();
    {
      // Keep Data alive while copying its identity: unregisterData takes the
      // exclusive lock before a finalized session can destroy these objects.
      std::shared_lock<std::shared_mutex> lock(self.mutex);
      for (auto *data : self.dataSet)
        launch.sessions.insert(data->getPath());
    }
    std::lock_guard<std::mutex> lock(self.eventsMutex);
    self.launches[data->correlationId] = std::move(launch);
  }
  static void request(uint8_t **buffer, size_t *size, size_t *maxRecords) {
    *size = 8 * 1024 * 1024;
    *maxRecords = 0;
    *buffer = static_cast<uint8_t *>(std::malloc(*size));
    if (!*buffer)
      *size = 0;
  }
  static void complete(uint8_t *buffer, size_t, size_t valid) {
    if (!valid) {
      std::free(buffer);
      return;
    }
    auto &self = instance();
    std::lock_guard<std::mutex> lock(self.eventsMutex);
    Topspti_Activity *record = nullptr;
    TopsptiResult result;
    while ((result = topsptiActivityGetNextRecord(buffer, valid, &record)) ==
           TOPSPTI_SUCCESS) {
      if (record->kind != TOPSPTI_ACTIVITY_KIND_KERNEL)
        continue;
      auto *kernel = reinterpret_cast<Topspti_ActivityKernel *>(record);
      if (!kernel->start || kernel->end <= kernel->start) {
        self.callbackError = "TOPSPTI kernel has invalid device timestamps";
        continue;
      }
      RuntimeTraceEventKey event;
      event.opName = kernel->name ? kernel->name : "enflame_kernel";
      event.startTimeNs = kernel->start;
      event.endTimeNs = kernel->end;
      event.deviceId = kernel->deviceId;
      event.streamId = kernel->streamId;
      event.correlationId = kernel->correlationId;
      event.taskId = kernel->gridId;
      self.events.push_back(std::move(event));
    }
    if (result != TOPSPTI_ERROR_MAX_LIMIT_REACHED)
      self.callbackError =
          "TOPSPTI could not decode activity buffer: " + std::to_string(result);
    std::free(buffer);
  }
  void doStart() override {
    {
      std::lock_guard<std::mutex> lock(eventsMutex);
      events.clear();
      launches.clear();
      callbackError.clear();
    }
    check(topsptiActivityRegisterCallbacks(request, complete),
          "register activity callbacks");
    check(topsptiSubscribe(&subscriber, callback, nullptr),
          "subscribe callbacks");
    try {
      check(topsptiEnableDomain(1, subscriber, TOPSPTI_CB_DOMAIN_RUNTIME_API),
            "enable runtime callbacks");
      check(topsptiEnableDomain(1, subscriber, TOPSPTI_CB_DOMAIN_DRIVER_API),
            "enable driver callbacks");
      check(topsptiActivityEnable(TOPSPTI_ACTIVITY_KIND_KERNEL),
            "enable kernel activities");
      enabled = true;
    } catch (...) {
      topsptiUnsubscribe(subscriber);
      subscriber = nullptr;
      throw;
    }
  }
  void doFlush() override {
    if (!enabled)
      return;
    if (topsDeviceSynchronize() != topsSuccess)
      throw std::runtime_error(
          "topsDeviceSynchronize failed during TOPSPTI flush");
    check(topsptiActivityFlushAll(TOPSPTI_ACTIVITY_FLAG_FLUSH_FORCED),
          "flush activities");
    size_t dropped = 0;
    check(topsptiActivityGetNumDroppedRecords(&dropped),
          "get dropped activities");
    std::lock_guard<std::mutex> lock(eventsMutex);
    if (dropped)
      throw std::runtime_error("TOPSPTI dropped " + std::to_string(dropped) +
                               " records");
    if (!callbackError.empty())
      throw std::runtime_error(callbackError);
  }
  void doStop() override {
    if (enabled) {
      check(topsptiActivityDisable(TOPSPTI_ACTIVITY_KIND_KERNEL),
            "disable kernel activities");
      enabled = false;
    }
    if (subscriber) {
      check(topsptiUnsubscribe(subscriber), "unsubscribe callbacks");
      subscriber = nullptr;
    }
  }
};
class EnflameImporter final : public VendorMetricsImporter {
public:
  std::string getName() const override { return "topspti_activity"; }
  VendorProfileArtifact import(const SessionProfileMetadata &metadata,
                               const VendorProfilePlan &plan) const override {
    return EnflameProfiler::instance().collect(metadata, plan);
  }
};
} // namespace
VendorProfilePlan
EnflameAdapter::makePlan(const VendorProfileOptions &options) const {
  VendorProfilePlan plan;
  plan.requested = options;
  plan.runtimeBaseEnabled = options.runtimeBaseEnabled;
  for (const auto &metric : options.vendorMetrics) {
    if (metric.required)
      throw std::runtime_error(
          "Enflame TOPSPTI hardware counter unsupported: " + metric.name);
    plan.disabledVendorMetrics.push_back(metric.name);
    plan.degradeReasons.push_back("unsupported hardware counter: " +
                                  metric.name);
  }
  return plan;
}
Profiler *EnflameAdapter::getRuntimeProfiler() const {
  return &EnflameProfiler::instance();
}
std::unique_ptr<VendorMetricsImporter> EnflameAdapter::createImporter() const {
  return std::make_unique<EnflameImporter>();
}
} // namespace proton
