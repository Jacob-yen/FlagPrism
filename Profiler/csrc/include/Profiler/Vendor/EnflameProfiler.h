#ifndef FLAGPRISM_ENFLAME_PROFILER_H
#define FLAGPRISM_ENFLAME_PROFILER_H
#include "Profiler/Vendor/Adapter.h"
#include "Utility/Singleton.h"
namespace proton {
class EnflameAdapter final : public VendorAdapter,
                             public Singleton<EnflameAdapter> {
public:
  std::string getName() const override { return "enflame"; }
  DeviceType getDeviceType() const override { return DeviceType::ENFLAME; }
  std::vector<std::string> getSupportedVendorMetrics() const override {
    return {};
  }
  VendorProfilePlan makePlan(const VendorProfileOptions &) const override;
  Profiler *getRuntimeProfiler() const override;
  std::unique_ptr<VendorMetricsImporter> createImporter() const override;
};
} // namespace proton
#endif
