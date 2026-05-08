#include "slice-env.h"

#ifdef HAVE_OPENGYM

#include "ns3/integer.h"
#include "ns3/nr-ue-mac.h"
#include "ns3/nr-ue-net-device.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("NrSliceGymEnv");
NS_OBJECT_ENSURE_REGISTERED(NrSliceGymEnv);

namespace
{
constexpr std::array<std::array<int8_t, NrSliceGymEnv::kSliceCount>, NrSliceGymEnv::kActionCount>
    kActionDelta = {{{-1, -1, -1},
                     {-1, -1,  0},
                     {-1, -1,  1},
                     {-1,  0, -1},
                     {-1,  0,  0},
                     {-1,  0,  1},
                     {-1,  1, -1},
                     {-1,  1,  0},
                     {-1,  1,  1},
                     { 0, -1, -1},
                     { 0, -1,  0},
                     { 0, -1,  1},
                     { 0,  0, -1},
                     { 0,  0,  0},
                     { 0,  0,  1},
                     { 0,  1, -1},
                     { 0,  1,  0},
                     { 0,  1,  1},
                     { 1, -1, -1},
                     { 1, -1,  0},
                     { 1, -1,  1},
                     { 1,  0, -1},
                     { 1,  0,  0},
                     { 1,  0,  1},
                     { 1,  1, -1},
                     { 1,  1,  0},
                     { 1,  1,  1}}};

float
Clamp01(double v)
{
    return static_cast<float>(std::max(0.0, std::min(1.0, v)));
}

// #region agent log
inline void
DebugLog(const std::string& runId,
         const std::string& hypothesisId,
         const std::string& location,
         const std::string& message,
         const std::string& data)
{
    const std::string line =
        "{\"sessionId\":\"73bf42\",\"runId\":\"" + runId +
        "\",\"hypothesisId\":\"" + hypothesisId +
        "\",\"location\":\"" + location +
        "\",\"message\":\"" + message +
        "\",\"data\":" + data +
        ",\"timestamp\":" + std::to_string(Simulator::Now().GetMilliSeconds()) + "}\n";
    {
        std::ofstream out("/home/skynetxii/5g-project/ns-allinone-3.45/debug-73bf42.log",
                          std::ios::app);
        if (out.is_open())
        {
            out << line;
        }
    }
    {
        std::ofstream out("/tmp/debug-73bf42.log", std::ios::app);
        if (out.is_open())
        {
            out << line;
        }
    }
}
// #endregion
} // namespace

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

NrSliceGymEnv::NrSliceGymEnv()
{
    m_observation.fill(0.0F);
}

NrSliceGymEnv::~NrSliceGymEnv() = default;

TypeId
NrSliceGymEnv::GetTypeId()
{
    static TypeId tid = TypeId("ns3::NrSliceGymEnv")
                            .SetParent<OpenGymEnv>()
                            .AddConstructor<NrSliceGymEnv>();
    return tid;
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

void
NrSliceGymEnv::Initialize(const Config& cfg,
                          const Ptr<NrHelper>& nrHelper,
                          const NetDeviceContainer& gnbDevs,
                          const std::array<NetDeviceContainer, kSliceCount>& ueDevsBySlice)
{
    m_cfg           = cfg;
    m_nrHelper      = nrHelper;
    m_gnbDevs       = gnbDevs;
    m_ueDevsBySlice = ueDevsBySlice;
    m_prbAlloc      = cfg.initialPrbAlloc;

    m_uesPerSlice = {
        ueDevsBySlice[EMBB].GetN(),
        ueDevsBySlice[URLLC].GetN(),
        ueDevsBySlice[MMTC].GetN()
    };

    BuildImsiSliceMap();

    m_initialized = true;
    m_stepCount   = 0;
    Simulator::Schedule(m_cfg.stepInterval, &NrSliceGymEnv::ScheduleStep, this);
}

void
NrSliceGymEnv::SetFlowMonitor(const Ptr<FlowMonitor>& flowMonitor,
                              const Ptr<Ipv4FlowClassifier>& flowClassifier)
{
    m_flowMonitor    = flowMonitor;
    m_flowClassifier = flowClassifier;
}

// ---------------------------------------------------------------------------
// RNTI / IMSI map construction
// ---------------------------------------------------------------------------

void
NrSliceGymEnv::BuildImsiSliceMap()
{
    m_imsiToSlice.clear();

    for (uint8_t s = 0; s < kSliceCount; ++s)
    {
        for (uint32_t i = 0; i < m_ueDevsBySlice[s].GetN(); ++i)
        {
            Ptr<NrUeNetDevice> ueDev =
                DynamicCast<NrUeNetDevice>(m_ueDevsBySlice[s].Get(i));
            if (!ueDev)
            {
                continue;
            }
            m_imsiToSlice[ueDev->GetImsi()] = s;
        }
    }
}

void
NrSliceGymEnv::TryBuildRntiSliceMap()
{
    if (m_rntiMapReady)
    {
        return;
    }

    m_rntiToSlice.clear();
    bool allResolved = true;

    for (uint8_t s = 0; s < kSliceCount; ++s)
    {
        for (uint32_t i = 0; i < m_ueDevsBySlice[s].GetN(); ++i)
        {
            Ptr<NrUeNetDevice> ueDev =
                DynamicCast<NrUeNetDevice>(m_ueDevsBySlice[s].Get(i));
            if (!ueDev || !ueDev->GetMac(0))
            {
                allResolved = false;
                continue;
            }

            const uint16_t rnti = ueDev->GetMac(0)->GetRnti();
            if (rnti == std::numeric_limits<uint16_t>::max())
            {
                allResolved = false;
                continue;
            }

            m_rntiToSlice[rnti] = s;
        }
    }

    m_rntiMapReady = allResolved && !m_rntiToSlice.empty();
}

// ---------------------------------------------------------------------------
// Scheduler callback
// ---------------------------------------------------------------------------

void
NrSliceGymEnv::OnSchedulerNotify(
    const std::vector<NrMacSchedulerUeInfoAi::LcObservation>& observations,
    bool isGameOver,
    float reward,
    const std::string& extraInfo,
    const NrMacSchedulerUeInfoAi::UpdateAllUeWeightsFn& updateWeightsFn)
{
    m_lastLcObservations = observations;
    m_gameOver           = isGameOver;
    m_reward             = reward;
    m_extraInfo          = extraInfo;
    m_updateWeightsFn    = updateWeightsFn;

    TryBuildRntiSliceMap();             // P2-2 fixed: single call, For some reason it was called twice here, TWICE !!!
    if (!m_rntiMapReady)
    {
        NS_LOG_WARN("OnSchedulerNotify: RNTI map not ready — dropping "
                    << observations.size() << " HOL observations this callback.");
    }

    if (m_rntiMapReady)      
    {
        for ( const auto & obs : observations)       // P0-1 fixed: single loop, O(N), it used to be 2 loops the outer one over shadows the inner one 
 
        {
            auto it = m_rntiToSlice. find (obs.rnti);
            if (it == m_rntiToSlice.end ())
            {
                continue ;
            }
            const uint8_t slice = it->second;
 
            if (slice >= kSliceCount)
            {
                continue ;
            }
            m_holSumMs[slice] += static_cast < double >(obs.holDelay);
            ++m_holSamples[slice];
        }
    }
    ApplySliceWeights();
}

// ---------------------------------------------------------------------------
// OpenGymEnv API
// ---------------------------------------------------------------------------

Ptr<OpenGymSpace>
NrSliceGymEnv::GetActionSpace()
{
    return CreateObject<OpenGymDiscreteSpace>(kActionCount);
}

Ptr<OpenGymSpace>
NrSliceGymEnv::GetObservationSpace()
{
    const float low  = 0.0F;
    const float high = 1.0F;
    std::vector<uint32_t> shape = {kObsSize};
    return CreateObject<OpenGymBoxSpace>(low, high, shape, TypeNameGet<float>());
}

bool
NrSliceGymEnv::GetGameOver()
{
    return m_gameOver;
}

Ptr<OpenGymDataContainer>
NrSliceGymEnv::GetObservation()
{
    std::vector<uint32_t> shape = {kObsSize};
    Ptr<OpenGymBoxContainer<float>> box =
        CreateObject<OpenGymBoxContainer<float>>(shape);

    for (uint32_t i = 0; i < kObsSize; ++i)
    {
        box->AddValue(m_observation[i]);
    }

    return box;
}

float
NrSliceGymEnv::GetReward()
{
    return m_reward;
}

std::string
NrSliceGymEnv::GetExtraInfo()
{
    return m_extraInfo;
}

bool
NrSliceGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
    Ptr<OpenGymDiscreteContainer> discrete =
        DynamicCast<OpenGymDiscreteContainer>(action);
    if (!discrete)
    {
        NS_LOG_WARN("NrSliceGymEnv received non-discrete action container");
        return false;
    }

    const uint32_t actionId = static_cast<uint32_t>(discrete->GetValue());
    if (actionId >= kActionCount)
    {
        NS_LOG_WARN("Action id out of range: " << actionId);
        return false;
    }

    for (uint32_t s = 0; s < kSliceCount; ++s)
    {
        const int32_t updated =
            static_cast<int32_t>(m_prbAlloc[s]) + kActionDelta[actionId][s];
        m_prbAlloc[s] = static_cast<uint16_t>(std::max(1, updated));
    }

    EnforceConstraints();
    ApplySliceWeights();
    // #region agent log
    DebugLog("baseline",
             "H4",
             "slice-env.cc:ExecuteActions",
             "Action decoded and PRB allocation updated",
             "{\"actionId\":" + std::to_string(actionId) +
                 ",\"prb\":" +
                 "[" + std::to_string(m_prbAlloc[0]) + "," +
                 std::to_string(m_prbAlloc[1]) + "," +
                 std::to_string(m_prbAlloc[2]) + "]}");
    // #endregion
    return true;
}

// ---------------------------------------------------------------------------
// EnforceConstraints — keeps sum(m_prbAlloc) == totalPrbs with min=1 per slice
// ---------------------------------------------------------------------------

void
NrSliceGymEnv::EnforceConstraints()
{
    for (auto& prb : m_prbAlloc)
    {
        prb = std::max<uint16_t>(1, prb);
    }

    int32_t diff = static_cast<int32_t>(m_cfg.totalPrbs) -
                   static_cast<int32_t>(m_prbAlloc[0] + m_prbAlloc[1] + m_prbAlloc[2]);

    while (diff > 0)
    {
        const uint8_t minSlice = static_cast<uint8_t>(
            std::distance(m_prbAlloc.begin(),
                          std::min_element(m_prbAlloc.begin(), m_prbAlloc.end())));
        ++m_prbAlloc[minSlice];
        --diff;
    }

    int safetyCounter = 0;
    while (diff < 0)
    {
        // Rotate the starting slice index by m_stepCount to avoid always
        // removing from eMBB (index 0) when slices are tied for maximum.
        bool removed = false;
        for (uint8_t i = 0; i < kSliceCount && diff < 0; ++i)
        {
            const uint8_t idx = (m_stepCount + i) % kSliceCount;
            if (m_prbAlloc[idx] > 1)
            {
                --m_prbAlloc[idx];
                ++diff;
                removed = true;
            }
        }
        if (!removed || ++safetyCounter > 100)
        {
            NS_LOG_WARN("NrSliceGymEnv::EnforceConstraints: "
                        "cannot converge, resetting to initialPrbAlloc");
            m_prbAlloc = m_cfg.initialPrbAlloc;
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// ApplySliceWeights — pushes PRB-proportional weights to the AI scheduler
// ---------------------------------------------------------------------------

void
NrSliceGymEnv::ApplySliceWeights()
{
    if (!m_updateWeightsFn)
    {
        return;
    }

    if (m_lastLcObservations.empty())
    {
        return;
    }

    TryBuildRntiSliceMap();

    NrMacSchedulerUeInfoAi::UeWeightsMap weightsMap;

    for (const auto& obs : m_lastLcObservations)
    {
        const uint16_t rnti16 = obs.rnti;
        const uint8_t  lcId   = obs.lcId;

        auto it = m_rntiToSlice.find(rnti16);
        if (it == m_rntiToSlice.end())
        {
            continue;
        }

        const uint8_t slice = it->second;
        if (slice >= kSliceCount)
        {
            continue;
        }

        const double sliceFrac =
            static_cast<double>(m_prbAlloc[slice]) / m_cfg.totalPrbs;
        const double ueCount =
            static_cast<double>(std::max<uint32_t>(1, m_uesPerSlice[slice]));
        const double weight = sliceFrac / ueCount;

        // UeWeightsMap uses uint8_t keys — a 5G-LENA API constraint.
        // Guard against silent wrap-around if RNTI ever exceeds 255.
        // With 35 UEs this never triggers, but the failure would be
        // undetectable without this check.
        if (rnti16 > 255)
        {
            NS_LOG_ERROR("ApplySliceWeights: RNTI " << rnti16
                         << " exceeds uint8_t range — skipping UE to prevent "
                         << "key collision in UeWeightsMap. This is a 5G-LENA "
                         << "API limitation (UeWeightsMap outer key is uint8_t).");
            continue;
        }

        weightsMap[static_cast<uint8_t>(rnti16)][lcId] = weight;
    }

    if (!weightsMap.empty())
    {
        m_updateWeightsFn(weightsMap);
    }
}

// ---------------------------------------------------------------------------
// AggregateHolDelay — fills m_holNorm[s] from MAC scheduler BSR observations.
//
// P1-3 VERIFICATION: The first 20 steps emit NS_LOG_INFO lines that print the
// raw holDelay value (uint16_t from m_rlcTransmissionQueueHolDelay) alongside
// the normalised result. To activate:
//
//   NS_LOG=NrSliceGymEnv=info ./run_simulation_copy2.sh dqn --episodes 1
//   holDelay is in ms CONFIERMED MULTIPLY TIMES
//   Expected values (unit  = milliseconds):
//   URLLC active period: holDelay in range [1, 15] ms  → holNorm in [0.07, 1.0]
//   URLLC silent period: no observations arrive         → sample-and-hold
//   eMBB active period:  holDelay in range [1, 50] ms  → holNorm in [0.02, 1.0]
//
// ---------------------------------------------------------------------------

void
NrSliceGymEnv::AggregateHolDelay()
{
     // Ensure RNTI map is ready before consuming accumulated observations.
    TryBuildRntiSliceMap();

    // If no fresh scheduler observations arrived this step, preserve the
    // previous m_holNorm values (sample-and-hold). Resetting to zero would
    // confuse the agent — zero means "no congestion," but the real meaning
    // here is "no new data." The two cases must be distinguishable.
     if (!m_rntiMapReady)
    {
        return;   // keep previous m_holNorm — do NOT fill(0.0)
    }



    for (uint8_t s = 0; s < kSliceCount; ++s)
    {    m_schedulerActiveThisStep[s] = (m_holSamples[s] > 0);
         if ( m_holSamples [s] == 0 )
        {
            // No UEs from this slice had observations this callback —
            // keep previous value rather than zeroing.
            continue;
        }

        const double meanHolMs = m_holSumMs[s] / static_cast < double >(m_holSamples [s]);
            // --- TEMPORARY DIAGNOSTIC — remove before final training run ---
        if (m_stepCount <= 20)
        {
            NS_LOG_INFO("HOL_DIAG step=" << m_stepCount
                << " slice=" << static_cast<int>(s)
                << " samples=" << m_holSamples[s]
                << " raw_sum=" << m_holSumMs[s]
                << " meanHolMs=" << meanHolMs
                << " holNorm=" << Clamp01(meanHolMs / std::max(1e-9, 2.0 * m_cfg.maxLatMs[s]))); //Unit is milliseconds ( found )
        }
// --- END DIAGNOSTIC ---

        m_holNorm[s] = Clamp01(meanHolMs / std::max(1e-9, 2.0 * m_cfg.maxLatMs[s]));
        m_holSumMs[s] = 0.0 ;
        m_holSamples[s] = 0 ;
    }
}

// ---------------------------------------------------------------------------
// Flow statistics aggregation
// ---------------------------------------------------------------------------

// Returns kSliceCount (== 3) as a sentinel for "unknown slice."
// Callers must check for this value and skip the flow.
uint8_t
NrSliceGymEnv::ResolveSliceFromPort(uint16_t port) const
{
    if (port >= 1000 && port <= 1999) return EMBB;
    if (port >= 2000 && port <= 2999) return URLLC;
    if (port >= 3000 && port <= 3999) return MMTC;

    // Port outside all known slice ranges. Could be NS-3 internal control
    // traffic, ARP, or routing messages. Return sentinel to skip this flow.
    NS_LOG_DEBUG("ResolveSliceFromPort: unknown port " << port
                 << " — flow will be excluded from slice statistics.");
    return kSliceCount;   // sentinel: caller must check slice < kSliceCount
}

uint8_t
NrSliceGymEnv::ResolveSliceFromAddress(const Ipv4Address& dst) const
{
    auto it = m_ipToSlice.find(dst);
    if (it == m_ipToSlice.end())
    {
        return kSliceCount;
    }
    return it->second;
}

void
NrSliceGymEnv::AggregateFlowStats()
{
    m_thrMbps.fill(0.0);
    

    if (!m_flowMonitor || !m_flowClassifier)
    {
        return;
    }

    const auto stats = m_flowMonitor->GetFlowStats();

    std::array<double,   kSliceCount> totalDeltaDelayMs{0.0, 0.0, 0.0};
    std::array<uint64_t, kSliceCount> totalDeltaPkts{0, 0, 0};

    for (const auto& [flowId, st] : stats)
    {
        Ipv4FlowClassifier::FiveTuple fiveTuple =
            m_flowClassifier->FindFlow(flowId);

        uint8_t slice = ResolveSliceFromAddress(fiveTuple.destinationAddress);
            if (slice >= kSliceCount)
            {
             slice = ResolveSliceFromPort(fiveTuple.destinationPort);
            }

        if (slice >= kSliceCount)
        {
            continue;   // unknown port — skip to avoid polluting slice stats
        }

        // --- Throughput ---
        const uint64_t prevBytes  = m_prevRxBytes.count(flowId)
                                  ? m_prevRxBytes[flowId] : 0;
        const uint64_t deltaBytes = (st.rxBytes >= prevBytes)
                                  ? (st.rxBytes - prevBytes) : st.rxBytes;
        m_prevRxBytes[flowId]     = st.rxBytes;

        m_thrMbps[slice] +=
            (static_cast<double>(deltaBytes) * 8.0 / 1e6) /
            std::max(1e-9, m_cfg.stepInterval.GetSeconds());

        // --- Latency ---
        const uint64_t prevPkts  = m_prevRxPackets.count(flowId)
                                 ? m_prevRxPackets[flowId] : 0;
        const double   prevDelay = m_prevDelaySum.count(flowId)
                                 ? m_prevDelaySum[flowId]  : 0.0;
        const uint64_t deltaPkts  = (st.rxPackets >= prevPkts)
                                  ? (st.rxPackets - prevPkts) : 0;
        const double   deltaDelay = (st.delaySum.GetSeconds() >= prevDelay)
                                  ? (st.delaySum.GetSeconds() - prevDelay) : 0.0;

        m_prevRxPackets[flowId] = st.rxPackets;
        m_prevDelaySum[flowId]  = st.delaySum.GetSeconds();

        if (deltaPkts > 0)
        {
            totalDeltaDelayMs[slice] += deltaDelay * 1e3;
            totalDeltaPkts[slice]    += deltaPkts;
        }
    }

    for (uint8_t s = 0; s < kSliceCount; ++s)
    {
        if (totalDeltaPkts[s] > 0)
        {
            m_latMs[s] = totalDeltaDelayMs[s] /
                         static_cast<double>(totalDeltaPkts[s]);
        }
    }
}

// ---------------------------------------------------------------------------
// Scheduled step — observation + reward + notify
// ---------------------------------------------------------------------------

void
NrSliceGymEnv::ScheduleStep()
{
    if (!m_initialized)
    {
        return;
    }

    ++m_stepCount;
    AggregateFlowStats();
    AggregateHolDelay();
    // #region agent log
    DebugLog("baseline",
             "H1",
             "slice-env.cc:ScheduleStep:raw_metrics",
             "Raw throughput and latency before normalization",
             "{\"step\":" + std::to_string(m_stepCount) +
                 ",\"thr_mbps\":[" + std::to_string(m_thrMbps[0]) + "," +
                 std::to_string(m_thrMbps[1]) + "," +
                 std::to_string(m_thrMbps[2]) + "],\"lat_ms\":[" +
                 std::to_string(m_latMs[0]) + "," + std::to_string(m_latMs[1]) + "," +
                 std::to_string(m_latMs[2]) + "]}");
    // #endregion
        // Clear stale latency for slices with no active traffic this step.
    // m_latMs[s] is only updated when packets arrive (AggregateFlowStats),
    // so during off-periods it holds the value from the last active step.
    // m_thrMbps[s] IS reset to 0 every step, so it is the correct gate.
    // Without this, obs[6:9] contradicts obs[15:18]: agent sees nonzero
    // latency simultaneously with demand_active=0.
    for (uint8_t s = 0; s < kSliceCount; ++s)
    {
        if (m_thrMbps[s] <= 0.001)
        {
            m_latMs[s] = 0.0;
        }
    }


    // Build normalised observation vector.
    //
    // obs[0:3]   prb_frac  — current PRB allocation as fraction of totalPrbs
    // obs[3:6]   throughput — normalised by per-slice maxThrMbps
    // obs[6:9]   latency    — normalised by 2×maxLatMs (SLA boundary at 0.5)
    // obs[9:12]  hol_delay  — HOL delay normalised by 2 × maxLatMs (forward-looking).
    //                         SLA boundary maps to 0.5, matching obs[6:9].
    // obs[12:15] prb_efficiency — (thr/maxThr) / (prb/totalPrbs) / 5.0, clipped to [0, 1].
    //                             0.0 = no throughput per PRB (silent or starved).
    //                             0.5 = SLA-level throughput at fair PRB share.
    //                             1.0 = full throughput at minimum PRB allocation.
    //                             Orthogonal to obs[0:3] and obs[3:6] individually.


    for (uint8_t s = 0; s < kSliceCount; ++s)
    {
        m_observation[s]      = Clamp01(static_cast<double>(m_prbAlloc[s]) /
                                        m_cfg.totalPrbs);
        m_observation[3 + s]  = Clamp01(m_thrMbps[s] / m_cfg.maxThrMbps[s]);
        m_observation[6 + s]  = Clamp01(m_latMs[s] / (2.0 * m_cfg.maxLatMs[s]));
        m_observation[9 + s]  = m_holNorm[s];
        {
            const double prbFracSafe = static_cast<double>(m_prbAlloc[s]) /
                                       static_cast<double>(m_cfg.totalPrbs);
            const double thrNormVal  = m_thrMbps[s] /
                                       std::max(1e-9, m_cfg.maxThrMbps[s]);
            const double eff         = thrNormVal / std::max(1e-9, prbFracSafe);
            m_observation[12 + s]   = Clamp01(eff / 5.0);
        }
        // used to be "m_observation[12 + s] = Clamp01(static_cast<double>(m_uesPerSlice[s]) / m_cfg.maxUes[s]);"
    }
    // #region agent log
    DebugLog("baseline",
             "H2",
             "slice-env.cc:ScheduleStep:obs",
             "Observation vector after normalization",
             "{\"step\":" + std::to_string(m_stepCount) +
                 ",\"obs\":[" + std::to_string(m_observation[0]) + "," +
                 std::to_string(m_observation[1]) + "," + std::to_string(m_observation[2]) +
                 "," + std::to_string(m_observation[3]) + "," +
                 std::to_string(m_observation[4]) + "," + std::to_string(m_observation[5]) +
                 "," + std::to_string(m_observation[6]) + "," +
                 std::to_string(m_observation[7]) + "," + std::to_string(m_observation[8]) +
                 "," + std::to_string(m_observation[9]) + "," +
                 std::to_string(m_observation[10]) + "," +
                 std::to_string(m_observation[11]) + "," +
                 std::to_string(m_observation[12]) + "," +
                 std::to_string(m_observation[13]) + "," +
                 std::to_string(m_observation[14]) + "," +
                 std::to_string(m_observation[15]) + "," +
                 std::to_string(m_observation[16]) + "," +
                 std::to_string(m_observation[17]) + "]}");
    // #endregion

    // -----------------------------------------------------------------------
    // Reward Function v2: throughput utility + continuous SLA margin +
    //            PRB-efficiency Jain - heavy penalty per SLA violation
    // -----------------------------------------------------------------------
    double   thrNormAvg    = 0.0;
    double   slaMarginAvg  = 0.0;
    uint32_t slaViolations = 0;
    double   effScore      = 0.0;
    uint32_t activeSlices  = 0;

    m_servedDemandRatio.fill(0.0);
    m_demandActive.fill(0);


    for (uint8_t s = 0; s < kSliceCount; ++s)
    {
        const double thrNorm = m_thrMbps[s] / std::max(1e-9, m_cfg.maxThrMbps[s]);
        const double prbFrac =
            static_cast<double>(m_prbAlloc[s]) /
            static_cast<double>(m_cfg.totalPrbs);
        const bool demandActive = (m_thrMbps[s] > 0.001);

        m_demandActive[s] = demandActive ? 1 : 0;

        if (!demandActive)
        {
            continue;
        }
    
        const bool slaSat =
            (m_thrMbps[s] >= m_cfg.minThrMbps[s]) &&
            (m_latMs[s]   <= m_cfg.maxLatMs[s]);

        const double thrMargin = std::tanh(std::max(0.0,                                             // thrMargin, latMargin now ∈ [0,1) — no overlap with violationRate penalty
            (m_thrMbps[s] - m_cfg.minThrMbps[s]) / std::max(1e-9, m_cfg.minThrMbps[s])));            //    std::max(1e-9, m_cfg.minThrMbps[s]));
        const double latMargin = std::tanh(std::max(0.0,                                             //  const double latMargin = std::tanh((m_cfg.maxLatMs[s] - m_latMs[s]) /
            ( m_cfg.maxLatMs[s] - m_latMs[s]) / std::max(1e-9, m_cfg.maxLatMs[s])));                  //     std::max(1e-9, m_cfg.maxLatMs[s]));
                
        
        slaViolations += slaSat ? 0 : 1;

        const double thrNormSoft = std::tanh(std::max(0.0, thrNorm));
        thrNormAvg   += thrNormSoft;
        slaMarginAvg += 0.5 * thrMargin + 0.5 * latMargin;
        const double eff = thrNormSoft / std::max(1e-9, prbFrac);
        // eff / 25.0: normalised by theoretical maximum (full thr on 1 PRB).
        // std::min cap prevents a single over-efficient slice from dominating
        // the average when n > 1.
        effScore += std::min(1.0, eff / 25.0);
        ++activeSlices;

        m_servedDemandRatio[s] = std::min(2.0, m_thrMbps[s] / std::max(1e-9, m_cfg.minThrMbps[s]));
    }
        // Promoted to outer scope so m_extraInfo can reference them below.
        // Defaults to 0.0 cover the activeSlices==0 case cleanly.
        double slaMarginNorm = 0.0;
        double effNorm       = 0.0;
        double violationRate = 0.0;
    

        if (activeSlices == 0)
    {
    // All slices simultaneously inactive — no service rendered, no learning signal.
    // Emit zero rather than the +0.45 produced by the Jain fallback.
        m_reward = 0.0F;
    }
        else
    {
        const uint32_t n         = activeSlices;  // guaranteed >= 1
        thrNormAvg              /= n;
        slaMarginAvg            /= n;
        // Also update slaMarginNorm since it no longer goes below 0:
        slaMarginNorm = slaMarginAvg ;  // already in [0,1) — no +1 offset needed         <--   // assign, not declare
        // effNorm: mean per-slice PRB efficiency, normalised to [0, 1].
        // Has gradient for any activeSlices >= 1, unlike Jain which
        // degenerates to 1.0 when n = 1 regardless of actual efficiency.
        effNorm       = effScore / static_cast<double>(n);                               // <-- // assign, not declare

        violationRate = static_cast<double>(slaViolations) /                             //<-- // assign, not declare
                                 static_cast<double>(n);

        // Weights adjusted so each term contributes its stated fraction
        // of the achievable maximum, accounting for tanh saturation:
        //   thrNormAvg  max = tanh(1.0) = 0.762  → w=0.451 → 40% of peak
        //   slaMarginNorm max = 0.881             → w=0.292 → 30% of peak
        //   effNorm     max = 1.000               → w=0.257 → 30% of peak
        m_reward = static_cast<float>(
        0.451 * thrNormAvg    +
        0.292 * slaMarginNorm +
        0.257 * effNorm       -
        1.20  * violationRate);
        // #region agent log
        DebugLog("baseline",
                 "H3",
                 "slice-env.cc:ScheduleStep:reward",
                 "Reward decomposition terms",
                 "{\"step\":" + std::to_string(m_stepCount) +
                     ",\"activeSlices\":" + std::to_string(activeSlices) +
                     ",\"thrNormAvg\":" + std::to_string(thrNormAvg) +
                     ",\"slaMarginNorm\":" + std::to_string(slaMarginNorm) +
                     ",\"effNorm\":" + std::to_string(effNorm) +
                     ",\"slaViolations\":" + std::to_string(slaViolations) +
                     ",\"violationRate\":" + std::to_string(violationRate) +
                     ",\"reward\":" + std::to_string(m_reward) + "}");
        // #endregion
    }

             m_extraInfo =
        std::string("{") +
        // ── demand / traffic state ──────────────────────────────────
        "\"demand_active\":["        + std::to_string(m_demandActive[0])      + "," +
                                       std::to_string(m_demandActive[1])      + "," +
                                       std::to_string(m_demandActive[2])      + "]," +
        "\"served_demand_ratio\":["  + std::to_string(m_servedDemandRatio[0]) + "," +
                                       std::to_string(m_servedDemandRatio[1]) + "," +
                                       std::to_string(m_servedDemandRatio[2]) + "]," +
        // ── raw per-slice metrics (physical units) ──────────────────
        "\"lat_ms\":["               + std::to_string(m_latMs[0])             + "," +
                                       std::to_string(m_latMs[1])             + "," +
                                       std::to_string(m_latMs[2])             + "]," +
        "\"hol_norm\":["             + std::to_string(m_holNorm[0])           + "," +
                                       std::to_string(m_holNorm[1])           + "," +
                                       std::to_string(m_holNorm[2])           + "]," +
        // ── reward decomposition terms ──────────────────────────────
        "\"reward_terms\":{"                                                          +
            "\"thr_norm_avg\":"      + std::to_string(thrNormAvg)             + "," +
            "\"sla_margin_norm\":"   + std::to_string(slaMarginNorm)          + "," +
            "\"eff_norm\":"          + std::to_string(effNorm)                + "," +
            "\"violation_rate\":"    + std::to_string(violationRate)          + "," +
            "\"active_slices\":"     + std::to_string(activeSlices)           +
        "}," +
        // ── static sim config (sent every step for Python cross-check) ─
        "\"cfg\":{"                                                                    +
            "\"max_thr_mbps\":["     + std::to_string(m_cfg.maxThrMbps[0])    + "," +
                                       std::to_string(m_cfg.maxThrMbps[1])    + "," +
                                       std::to_string(m_cfg.maxThrMbps[2])    + "]," +
            "\"min_thr_mbps\":["     + std::to_string(m_cfg.minThrMbps[0])    + "," +
                                       std::to_string(m_cfg.minThrMbps[1])    + "," +
                                       std::to_string(m_cfg.minThrMbps[2])    + "]," +
            "\"max_lat_ms\":["       + std::to_string(m_cfg.maxLatMs[0])      + "," +
                                       std::to_string(m_cfg.maxLatMs[1])      + "," +
                                       std::to_string(m_cfg.maxLatMs[2])      + "]"  +
        "}"  +
        "}";
    

    for (uint8_t s = 0; s < kSliceCount; ++s) //added for the binary flags (each slice has a flag to tell if its on or off)
    m_observation[15 + s] = static_cast<float>(m_demandActive[s]);
    // #region agent log
    DebugLog("baseline",
             "H5",
             "slice-env.cc:ScheduleStep:extraInfo",
             "Cross-layer payload and demand flags",
             "{\"step\":" + std::to_string(m_stepCount) +
                 ",\"demandActive\":[" + std::to_string(m_demandActive[0]) + "," +
                 std::to_string(m_demandActive[1]) + "," + std::to_string(m_demandActive[2]) +
                 "],\"servedDemandRatio\":[" + std::to_string(m_servedDemandRatio[0]) + "," +
                 std::to_string(m_servedDemandRatio[1]) + "," +
                 std::to_string(m_servedDemandRatio[2]) + "]}");
    // #endregion

    m_gameOver = Simulator::Now() >= m_cfg.simTime;

    Notify();

    if (!m_gameOver)
    {
        Simulator::Schedule(m_cfg.stepInterval,
                            &NrSliceGymEnv::ScheduleStep, this);
    }
}

} // namespace ns3

#endif // HAVE_OPENGYM
