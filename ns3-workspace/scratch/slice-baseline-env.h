#pragma once

#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/network-module.h"
#include "ns3/nr-module.h"

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace ns3
{

/**
 * NrSliceBaselineEnv — standalone 3-slice PRB baseline evaluator.
 *
 * Runs multiple named C++ policies sequentially on one continuous NS-3
 * simulation.  No OpenGym, no ZMQ, no Python ports required.
 *
 * State machine:
 *   StartPolicy(0) → [episodes × maxSteps ScheduleStep() calls] →
 *   EndEpisode() → StartPolicy(1) → ... → WriteResults() → m_allDone = true
 *
 * Output JSON keys match evaluate.py exactly so results are directly
 * comparable with DRL agent evaluation.
 */
class NrSliceBaselineEnv : public Object
{
  public:
    static constexpr uint32_t kSliceCount  = 3;
    static constexpr uint32_t kObsSize     = 18;
    static constexpr uint32_t kActionCount = 27;

    enum SliceId : uint8_t
    {
        EMBB  = 0,
        URLLC = 1,
        MMTC  = 2
    };

    // C++ policy function: receives 18-float obs, returns action in [0, 26]
    using PolicyFn = std::function<int(const std::array<float, kObsSize>&)>;

    // -----------------------------------------------------------------------
    // Config — mirrors NrSliceGymEnv::Config exactly
    // -----------------------------------------------------------------------
    struct Config
    {
        uint16_t totalPrbs{25};
        Time     stepInterval{MilliSeconds(100)};
        Time     simTime{Seconds(3010)};
        std::array<uint16_t, kSliceCount> initialPrbAlloc{10, 8, 7};
        std::array<uint32_t, kSliceCount> maxUes{20, 10, 50};
        std::array<double,   kSliceCount> maxThrMbps{100.0, 25.0, 8.0};
        std::array<double,   kSliceCount> maxLatMs{50.0, 15.0, 500.0};
        std::array<double,   kSliceCount> minThrMbps{10.0, 1.0, 0.1};
    };

    // -----------------------------------------------------------------------
    // Per-policy result — stores raw episode vectors; means/stds written to JSON
    // -----------------------------------------------------------------------
    struct PolicyResult
    {
        std::string name;
        std::vector<double> rewards;
        std::vector<double> slaRates, slaEmbb, slaUrllc, slaMmtc;
        std::vector<double> embbThrMbps, urllcThrMbps, mmtcThrMbps;
        std::vector<double> embbLatMs, urllcLatMs, mmtcLatMs; // NaN when always inactive
        std::vector<double> holEmbb, holUrllc, holMmtc;
        std::vector<double> effEmbb, effUrllc, effMmtc;
        std::vector<double> prbEmbb, prbUrllc, prbMmtc;
        std::vector<double> rwdThr, rwdSla, rwdEff, rwdViol, activeSlices;
    };

    NrSliceBaselineEnv();
    ~NrSliceBaselineEnv() override = default;

    static TypeId GetTypeId();

    // ── Configuration (call before Initialize) ──────────────────────────────
    void SetEpisodes(int n)                  { m_numEpisodes = n; }
    void SetMaxSteps(int n)                  { m_maxSteps    = n; }
    void SetOutputPath(const std::string& p) { m_outputPath  = p; }
    void AddPolicy(const std::string& name, PolicyFn fn);

    // ── NS-3 setup ──────────────────────────────────────────────────────────
    void Initialize(const Config&                                      cfg,
                    const Ptr<NrHelper>&                               nrHelper,
                    const NetDeviceContainer&                          gnbDevs,
                    const std::array<NetDeviceContainer, kSliceCount>& ueDevsBySlice);

    void SetFlowMonitor(const Ptr<FlowMonitor>&        fm,
                        const Ptr<Ipv4FlowClassifier>& fc);

    // ── Scheduler callback — same signature as NrSliceGymEnv ───────────────
    void OnSchedulerNotify(
        const std::vector<NrMacSchedulerUeInfoAi::LcObservation>& observations,
        bool                                                        isGameOver,
        float                                                       reward,
        const std::string&                                          extraInfo,
        const NrMacSchedulerUeInfoAi::UpdateAllUeWeightsFn&        updateWeightsFn);

    const std::vector<PolicyResult>& GetResults() const { return m_results; }

  private:
    // ── NS-3 simulation internals (copied from slice-env.cc) ────────────────
    void    ScheduleStep();
    void    AggregateFlowStats();
    void    AggregateHolDelay();
    void    ApplySliceWeights();
    void    EnforceConstraints();
    void    BuildImsiSliceMap();
    void    TryBuildRntiSliceMap();
    uint8_t ResolveSliceFromPort(uint16_t port) const;
    uint8_t ResolveSliceFromAddress(const Ipv4Address& dst) const;

    // ── Episode / policy state machine ──────────────────────────────────────
    void StartPolicy(size_t idx);
    void StartEpisode();
    void EndEpisode();
    void ResetStepAccumulators();
    void WriteResults() const;

    // ── Config & NS-3 objects ───────────────────────────────────────────────
    Config             m_cfg{};
    Ptr<NrHelper>      m_nrHelper;
    NetDeviceContainer m_gnbDevs;
    std::array<NetDeviceContainer, kSliceCount> m_ueDevsBySlice;
    Ptr<FlowMonitor>        m_flowMonitor;
    Ptr<Ipv4FlowClassifier> m_flowClassifier;

    // ── Policy registry ─────────────────────────────────────────────────────
    std::vector<std::string>  m_policyNames;
    std::vector<PolicyFn>     m_policies;
    std::vector<PolicyResult> m_results;
    size_t m_curPolicyIdx{0};
    int    m_curEpisode{0};
    int    m_curStep{0};
    int    m_numEpisodes{10};
    int    m_maxSteps{1000};
    bool   m_allDone{false};
    std::string m_outputPath{"results/baseline_cpp.json"};

    // ── Scheduler callback state ─────────────────────────────────────────────
    NrMacSchedulerUeInfoAi::UpdateAllUeWeightsFn            m_updateWeightsFn;
    std::vector<NrMacSchedulerUeInfoAi::LcObservation>      m_lastLcObs;

    // ── RNTI / IMSI maps ────────────────────────────────────────────────────
    std::unordered_map<uint64_t, uint8_t>                    m_imsiToSlice;
    std::unordered_map<Ipv4Address, uint8_t, Ipv4AddressHash> m_ipToSlice;
    std::unordered_map<uint16_t, uint8_t>                    m_rntiToSlice;
    bool m_rntiMapReady{false};

    // ── PRB / scheduling state ───────────────────────────────────────────────
    std::array<uint16_t, kSliceCount> m_prbAlloc{10, 8, 7};
    std::array<uint32_t, kSliceCount> m_uesPerSlice{};

    // ── Observation vector ───────────────────────────────────────────────────
    std::array<float, kObsSize> m_obs{};

    // ── Per-step raw metrics ─────────────────────────────────────────────────
    std::array<double,   kSliceCount> m_thrMbps{};
    std::array<double,   kSliceCount> m_latMs{};
    std::array<double,   kSliceCount> m_holNorm{};
    std::array<double,   kSliceCount> m_holSumMs{};
    std::array<uint32_t, kSliceCount> m_holSamples{};
    std::array<double,   kSliceCount> m_servedDemandRatio{};
    std::array<uint8_t,  kSliceCount> m_demandActive{};

    // ── Flow-level delta trackers ────────────────────────────────────────────
    std::unordered_map<uint32_t, uint64_t> m_prevRxBytes{};
    std::unordered_map<uint32_t, uint64_t> m_prevRxPackets{};
    std::unordered_map<uint32_t, double>   m_prevDelaySum{};

    // ── Episode-level accumulators (reset each StartEpisode) ────────────────
    double m_epReward{0.0};
    double m_epSla{0.0}, m_epSlaE{0.0}, m_epSlaU{0.0}, m_epSlaM{0.0};
    double m_epEThr{0.0}, m_epUThr{0.0}, m_epMThr{0.0};
    double m_epELat{0.0}; int m_epELatN{0};
    double m_epULat{0.0}; int m_epULatN{0};
    double m_epMLat{0.0}; int m_epMLatN{0};
    double m_epHolE{0.0}, m_epHolU{0.0}, m_epHolM{0.0};
    double m_epEffE{0.0}, m_epEffU{0.0}, m_epEffM{0.0};
    double m_epPrbE{0.0}, m_epPrbU{0.0}, m_epPrbM{0.0};
    double m_epRwdThr{0.0}, m_epRwdSla{0.0}, m_epRwdEff{0.0};
    double m_epRwdViol{0.0}, m_epActive{0.0};

    bool     m_initialized{false};
    uint32_t m_globalStep{0}; // across all policies/episodes
};

} // namespace ns3
