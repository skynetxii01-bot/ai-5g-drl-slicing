#pragma once

#if __has_include("ns3/opengym-module.h")
#define HAVE_OPENGYM

#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/network-module.h"
#include "ns3/nr-module.h"
#include "ns3/opengym-module.h"

#include <array>
#include <limits>
#include <unordered_map>

namespace ns3
{

class NrSliceGymEnv : public OpenGymEnv
{
  public:
    static constexpr uint32_t kSliceCount  = 3;
    static constexpr uint32_t kObsSize     = 15;
    static constexpr uint32_t kActionCount = 27;

    enum SliceId : uint8_t
    {
        EMBB  = 0,
        URLLC = 1,
        MMTC  = 2
    };

    // -----------------------------------------------------------------------
    // Config — all simulation parameters that govern the RL environment.
    //
    // CRITICAL: maxUes must be UPPER BOUNDS exceeding the simulated UE counts,
    // NOT the simulated counts themselves.
    //
    // Reason: obs[12:15] = m_uesPerSlice[s] / maxUes[s].  If maxUes equals the
    // simulated count, this ratio is identically 1.0 for every slice, every
    // step, making those three observation dimensions constant and uninformative.
    // Setting maxUes to a plausible system-level upper bound (e.g. 2× simulated)
    // produces fractional values that carry real density information.
    //
    // Defaults below match the values set in slice-rl-sim.cc at runtime:
    //   embbUes=10  → maxUes[EMBB]=20   → obs[12]=0.50
    //   urllcUes=5  → maxUes[URLLC]=10  → obs[13]=0.50
    //   mmtcUes=20  → maxUes[MMTC]=50   → obs[14]=0.40
    //
    // Note on UeWeightsMap RNTI constraint:
    //   The 5G-LENA AI scheduler API uses uint8_t as the RNTI key in
    //   UeWeightsMap, limiting the AI weight map to 255 UEs per gNB.
    //   This scenario uses 35 UEs and is unaffected. A runtime guard in
    //   ApplySliceWeights() logs an error and skips any UE whose RNTI
    //   exceeds 255, preventing silent key truncation.
    // -----------------------------------------------------------------------
    struct Config
    {
        uint16_t totalPrbs{25};
        Time stepInterval{MilliSeconds(100)};
        Time simTime{Seconds(100)};
        std::array<uint16_t, kSliceCount> initialPrbAlloc{10, 8, 7};

        // Upper bounds — must EXCEED simulated UE counts (see note above).
        // Default: {20, 10, 50} for eMBB, URLLC, mMTC respectively.
        // Was incorrectly {10, 5, 20} (equal to simulated counts → obs always 1.0).
        std::array<uint32_t, kSliceCount> maxUes{20, 10, 50};

        std::array<double, kSliceCount> maxThrMbps{100.0, 10.0, 2.0};
        std::array<double, kSliceCount> maxLatMs{50.0, 15.0, 500.0};
        std::array<double, kSliceCount> minThrMbps{10.0, 1.0, 0.1};
    };

    NrSliceGymEnv();
    ~NrSliceGymEnv() override;

    static TypeId GetTypeId();

    void Initialize(const Config& cfg,
                    const Ptr<NrHelper>& nrHelper,
                    const NetDeviceContainer& gnbDevs,
                    const std::array<NetDeviceContainer, kSliceCount>& ueDevsBySlice);

    void SetFlowMonitor(const Ptr<FlowMonitor>& flowMonitor,
                        const Ptr<Ipv4FlowClassifier>& flowClassifier);

    void BuildImsiSliceMap();

    void OnSchedulerNotify(
        const std::vector<NrMacSchedulerUeInfoAi::LcObservation>& observations,
        bool isGameOver,
        float reward,
        const std::string& extraInfo,
        const NrMacSchedulerUeInfoAi::UpdateAllUeWeightsFn& updateWeightsFn);

    // OpenGymEnv API
    Ptr<OpenGymSpace>      GetActionSpace()      override;
    Ptr<OpenGymSpace>      GetObservationSpace() override;
    bool                   GetGameOver()         override;
    Ptr<OpenGymDataContainer> GetObservation()   override;
    float                  GetReward()           override;
    std::string            GetExtraInfo()        override;
    bool ExecuteActions(Ptr<OpenGymDataContainer> action) override;

  private:
    void ScheduleStep();
    void AggregateFlowStats();
    void AggregateHolDelay();
    void ApplySliceWeights();
    void EnforceConstraints();

    uint8_t ResolveSliceFromPort(uint16_t port) const;
    void    TryBuildRntiSliceMap();

    Config m_cfg{};

    Ptr<NrHelper>         m_nrHelper;
    NetDeviceContainer    m_gnbDevs;
    std::array<NetDeviceContainer, kSliceCount> m_ueDevsBySlice;

    Ptr<FlowMonitor>        m_flowMonitor;
    Ptr<Ipv4FlowClassifier> m_flowClassifier;

    std::unordered_map<uint64_t, uint8_t> m_imsiToSlice;
    std::unordered_map<uint16_t, uint8_t> m_rntiToSlice;
    std::unordered_map<uint32_t, uint64_t> m_prevRxPackets{};
    std::unordered_map<uint32_t, double>   m_prevDelaySum{};

    bool     m_rntiMapReady{false};
    bool     m_gameOver{false};
    bool     m_initialized{false};
    uint32_t m_stepCount{0};

    float       m_reward{0.0F};
    std::string m_extraInfo;

    NrMacSchedulerUeInfoAi::UpdateAllUeWeightsFn m_updateWeightsFn;
    std::vector<NrMacSchedulerUeInfoAi::LcObservation> m_lastLcObservations;

    std::array<uint16_t, kSliceCount> m_prbAlloc{10, 8, 7};
    std::array<float,    kObsSize>    m_observation{};

    std::array<double, kSliceCount> m_thrMbps{};
    std::array<double, kSliceCount> m_latMs{};
    std::array<double, kSliceCount> m_holNorm{};      // HOL delay normalised to [0,1]
    std::array<uint32_t, kSliceCount> m_uesPerSlice{};

    // Flow-level cumulative stat trackers (delta computation each step).
    // These maps grow with the number of unique flow IDs assigned by FlowMonitor.
    // With 35 UEs and 3 traffic types the entry count is bounded at O(100).
    std::unordered_map<uint32_t, uint64_t> m_prevRxBytes{};
};

} // namespace ns3

#endif // HAVE_OPENGYM
