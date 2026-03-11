#ifndef SLICE_GYM_ENV_H
#define SLICE_GYM_ENV_H

#include "ns3/nr-mac-scheduler-tdma-ai.h"
#include "ns3/opengym-env.h"

#include <array>
#include <map>
#include <vector>

namespace ns3
{

/**
 * OpenGym environment used to control 5G-LENA slice allocations.
 *
 * Observation layout (15 elements):
 *  [0:3]   PRB fraction per slice
 *  [3:6]   throughput fraction per slice
 *  [6:9]   latency fraction per slice
 *  [9:12]  queue occupancy per slice
 *  [12:15] UE fraction per slice
 */
class NrSliceGymEnv : public OpenGymEnv
{
public:
  static TypeId GetTypeId();
  NrSliceGymEnv();
  ~NrSliceGymEnv() override;

  enum SliceId : uint8_t
  {
    EMBB = 0,
    URLLC = 1,
    MMTC = 2,
    NUM_SLICES = 3
  };

  struct SliceMetrics
  {
    double throughputMbps{0.0};
    double latencyMs{0.0};
    double queueOccupancy{0.0};
    double satisfaction{0.0};
    uint32_t slaViolations{0};
  };

  void SetScheduler(Ptr<NrMacSchedulerTdmaAi> scheduler);
  void SetUeRntisBySlice(const std::array<std::vector<uint16_t>, NUM_SLICES>& ueRntisBySlice);
  void SetTotalPrbs(uint32_t totalPrbs);
  void SetStepTime(Time stepTime);
  void SetEpisodeSteps(uint32_t episodeSteps);
  void UpdateSliceMetrics(const std::array<SliceMetrics, NUM_SLICES>& metrics);

  // OpenGymEnv interface.
  Ptr<OpenGymSpace> GetObservationSpace() override;
  Ptr<OpenGymSpace> GetActionSpace() override;
  Ptr<OpenGymDataContainer> GetObservation() override;
  float GetReward() override;
  bool GetGameOver() override;
  std::string GetExtraInfo() override;
  bool ExecuteActions(Ptr<OpenGymDataContainer> action) override;

  void NotifyStep();

private:
  using ActionDelta = std::array<int8_t, NUM_SLICES>;

  void BuildActionTable();
  void ApplyPrbAllocation();
  void NormalizePrbs();
  float ComputeReward() const;
  double ComputeJainFairness() const;

  Ptr<NrMacSchedulerTdmaAi> m_scheduler;
  std::array<std::vector<uint16_t>, NUM_SLICES> m_ueRntisBySlice;
  std::array<SliceMetrics, NUM_SLICES> m_metrics;
  std::array<uint32_t, NUM_SLICES> m_prbAlloc;
  std::vector<ActionDelta> m_actionTable;

  uint32_t m_totalPrbs{25};
  uint32_t m_episodeSteps{1000};
  uint32_t m_currentStep{0};

  std::array<double, NUM_SLICES> m_maxThroughput{{10.0, 1.0, 0.1}};
  std::array<double, NUM_SLICES> m_maxLatency{{50.0, 1.0, 500.0}};
  uint32_t m_maxUes{35};
  Time m_stepTime{MilliSeconds(100)};
};

} // namespace ns3

#endif // SLICE_GYM_ENV_H
