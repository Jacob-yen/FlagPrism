#ifndef PROTON_PROFILER_MTHREADS_ADAPTER_H_
#define PROTON_PROFILER_MTHREADS_ADAPTER_H_

#include "Profiler/Vendor/Adapter.h"

namespace proton {

class MthreadsMetricsImporter final : public VendorMetricsImporter {
public:
  std::string getName() const override;

  VendorProfileArtifact import(const SessionProfileMetadata &metadata,
                               const VendorProfilePlan &plan) const override;
};

class MthreadsAdapter final : public VendorAdapter {
public:
  static const MthreadsAdapter &instance();

  std::string getName() const override;
  DeviceType getDeviceType() const override;
  std::vector<std::string> getSupportedVendorMetrics() const override;
  VendorProfilePlan
  makePlan(const VendorProfileOptions &options) const override;
  Profiler *getRuntimeProfiler() const override;
  std::unique_ptr<VendorMetricsImporter> createImporter() const override;

private:
  MthreadsAdapter() = default;
};

} // namespace proton

#endif // PROTON_PROFILER_MTHREADS_ADAPTER_H_
