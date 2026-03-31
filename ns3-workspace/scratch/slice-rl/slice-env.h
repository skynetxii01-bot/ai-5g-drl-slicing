#ifndef SLICE_ENV_H
#define SLICE_ENV_H

#include "ns3/core-module.h"
#include "ns3/opengym-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/nr-module.h"
#include "ns3/net-device-container.h"

#include <array>
#include <unordered_map>

namespace ns3 {

class NrSliceGymEnv : public OpenGymEnv
{
public:
  static TypeId GetTypeId();
  NrSliceGymEnv();
  ~NrSliceGymEnv() override;

  struct SimConfig
  {
    uint32_t totalPrbs{25};
    double simTimeS{100.0};
    double stepS{0.1};
    uint32_t maxUes{35};
    std::array<uint32_t, 3> initPrb{{10, 8, 7}};
    std::array<double, 3> maxThrMbps{{100.0, 10.0, 2.0}};
    std::array<double, 3> maxLatMs{{50.0, 10.0, 500.0}};
    uint32_t maxSteps{1000};
  };

  void Initialize(const SimConfig &cfg,
                  Ptr<NrHelper> nrHelper,
                  const NetDeviceContainer &gnbDevs,
                  const std::array<NetDeviceContainer, 3> &ueDevsBySlice);

  void BuildImsiSliceMap();
  void AttachFlowMonitor(Ptr<FlowMonitor> fm);

  void OnSchedulerNotify(const std::vector<NrMacSchedulerUeInfoAi::LcObservation> &obs,
                         bool gameOver,
                         float reward,
                         const std::string &extra,
                         const NrMacSchedulerUeInfoAi::UpdateAllUeWeightsFn &updateFn);

  // OpenGymEnv implementation
  Ptr<OpenGymSpace> GetObservationSpace() override;
  Ptr<OpenGymSpace> GetActionSpace() override;
  Ptr<OpenGymDataContainer> GetObservation() override;
  float GetReward() override;
  bool GetGameOver() override;
  std::string GetExtraInfo() override;
  bool ExecuteActions(Ptr<OpenGymDataContainer> action) override;

private:
  enum SliceId : uint8_t
  {
    EMBB = 0,
    URLLC = 1,
    MMTC = 2
  };

  struct SliceMetrics
  {
    double thrMbps{0.0};
    double latMs{0.0};
    double queueOcc{0.0};
    double loss{0.0};
    uint32_t ueCount{0};
    uint32_t packets{0};
  };

  void ScheduleStep();
  void AggregateFlowStats();
  void UpdateRewardAndDone();
  void ApplySliceWeights();
  void EnforceConstraints();
  void BuildRntiToSliceMapLazily();
  std::array<int32_t, 3> DecodeAction(uint32_t actionId) const;
  double JainIndex(const std::array<double, 3> &x) const;

  SimConfig m_cfg;
  Ptr<NrHelper> m_nrHelper;
  NetDeviceContainer m_gnbDevs;
  std::array<NetDeviceContainer, 3> m_ueDevsBySlice;

  Ptr<FlowMonitor> m_flowMonitor;

  std::unordered_map<uint64_t, SliceId> m_imsiToSlice;
  std::unordered_map<uint16_t, SliceId> m_rntiToSlice;
  NrMacSchedulerUeInfoAi::UpdateAllUeWeightsFn m_updateWeightsFn;

  std::array<uint32_t, 3> m_prbAlloc;
  std::array<SliceMetrics, 3> m_metrics;

  float m_lastReward{0.0f};
  bool m_gameOver{false};
  std::string m_extraInfo;
  uint32_t m_stepCounter{0};
};

} // namespace ns3

#endif
