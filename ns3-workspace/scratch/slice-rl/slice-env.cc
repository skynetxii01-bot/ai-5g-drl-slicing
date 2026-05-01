#include "slice-env.h"

#ifdef HAVE_OPENGYM

#include "ns3/integer.h"
#include "ns3/nr-ue-mac.h"
#include "ns3/nr-ue-net-device.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("NrSliceGymEnv");
NS_OBJECT_ENSURE_REGISTERED(NrSliceGymEnv);

namespace
{
constexpr std::array<std::array<int8_t, NrSliceGymEnv::kSliceCount>, NrSliceGymEnv::kActionCount>
    kActionDelta = {{{-1, -1, -1},
                     {-1, -1, 0},
                     {-1, -1, 1},
                     {-1, 0, -1},
                     {-1, 0, 0},
                     {-1, 0, 1},
                     {-1, 1, -1},
                     {-1, 1, 0},
                     {-1, 1, 1},
                     {0, -1, -1},
                     {0, -1, 0},
                     {0, -1, 1},
                     {0, 0, -1},
                     {0, 0, 0},
                     {0, 0, 1},
                     {0, 1, -1},
                     {0, 1, 0},
                     {0, 1, 1},
                     {1, -1, -1},
                     {1, -1, 0},
                     {1, -1, 1},
                     {1, 0, -1},
                     {1, 0, 0},
                     {1, 0, 1},
                     {1, 1, -1},
                     {1, 1, 0},
                     {1, 1, 1}}};

float
Clamp01(double v)
{
    return static_cast<float>(std::max(0.0, std::min(1.0, v)));
}
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
    m_cfg          = cfg;
    m_nrHelper     = nrHelper;
    m_gnbDevs      = gnbDevs;
    m_ueDevsBySlice = ueDevsBySlice;
    m_prbAlloc     = cfg.initialPrbAlloc;

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
    m_flowMonitor     = flowMonitor;
    m_flowClassifier  = flowClassifier;
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

    TryBuildRntiSliceMap();
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
    return true;
}

// ---------------------------------------------------------------------------
// BUG-03 FIX — EnforceConstraints
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
        const uint8_t maxSlice = static_cast<uint8_t>(
            std::distance(m_prbAlloc.begin(),
                          std::max_element(m_prbAlloc.begin(), m_prbAlloc.end())));

        if (m_prbAlloc[maxSlice] <= 1 || ++safetyCounter > 100)
        {
            NS_LOG_WARN("NrSliceGymEnv::EnforceConstraints: "
                        "cannot converge, resetting to initialPrbAlloc");
            m_prbAlloc = m_cfg.initialPrbAlloc;
            break;
        }

        --m_prbAlloc[maxSlice];
        ++diff;
    }
}

// ---------------------------------------------------------------------------
// BUG-02 FIX — ApplySliceWeights
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

        const uint8_t rnti8 = static_cast<uint8_t>(rnti16);
        weightsMap[rnti8][lcId] = weight;
    }

    if (!weightsMap.empty())
    {
        m_updateWeightsFn(weightsMap);
    }
}

// ---------------------------------------------------------------------------
// Flow statistics aggregation
// ---------------------------------------------------------------------------

uint8_t
NrSliceGymEnv::ResolveSliceFromPort(uint16_t port) const
{
    if (port >= 1000 && port <= 1999)
    {
        return EMBB;
    }
    if (port >= 2000 && port <= 2999)
    {
        return URLLC;
    }
    return MMTC;
}

void
NrSliceGymEnv::AggregateFlowStats()
{
    m_thrMbps.fill(0.0);
    m_latMs.fill(0.0);
    m_queueOcc.fill(0.0);

    if (!m_flowMonitor || !m_flowClassifier)
    {
        return;
    }

    const auto stats = m_flowMonitor->GetFlowStats();
    std::array<uint32_t, kSliceCount> packetsPerSlice{0, 0, 0};

    for (const auto& [flowId, st] : stats)
    {
        Ipv4FlowClassifier::FiveTuple fiveTuple =
            m_flowClassifier->FindFlow(flowId);
        const uint8_t slice =
            ResolveSliceFromPort(fiveTuple.destinationPort);

        const uint64_t prev       = m_prevRxBytes.count(flowId) ? m_prevRxBytes[flowId] : 0;
        const uint64_t deltaBytes = (st.rxBytes >= prev) ? (st.rxBytes - prev) : st.rxBytes;
        m_prevRxBytes[flowId]     = st.rxBytes;
        const double thrMbps =
            (m_cfg.stepInterval.GetSeconds() > 0.0)
                ? (static_cast<double>(deltaBytes) * 8.0 / 1e6) /
                      std::max(1e-9, m_cfg.stepInterval.GetSeconds())
                : 0.0;
        m_thrMbps[slice] += thrMbps;

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
             const double meanDelayMs = (deltaDelay * 1e3) / deltaPkts;
             m_latMs[slice] += meanDelayMs;
            ++packetsPerSlice[slice];
        }

        const uint64_t tx = st.txPackets;
        const uint64_t rx = st.rxPackets;
        if (tx > 0)
        {
            m_queueOcc[slice] +=
                static_cast<double>(tx - rx) / static_cast<double>(tx);
        }
    }

    for (uint8_t s = 0; s < kSliceCount; ++s)
    {
        if (packetsPerSlice[s] > 0)
        {
            m_latMs[s] /= packetsPerSlice[s];
        }
        m_queueOcc[s] = std::min(1.0, std::max(0.0, m_queueOcc[s]));
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

    // Build normalised observation vector
    for (uint8_t s = 0; s < kSliceCount; ++s)
    {
        m_observation[s]      = Clamp01(static_cast<double>(m_prbAlloc[s]) /
                                        m_cfg.totalPrbs);
        m_observation[3 + s]  = Clamp01(m_thrMbps[s] / m_cfg.maxThrMbps[s]);
        m_observation[6 + s]  = Clamp01(m_latMs[s] / (2.0 * m_cfg.maxLatMs[s]));
        m_observation[9 + s]  = Clamp01(m_queueOcc[s]);
        m_observation[12 + s] = Clamp01(static_cast<double>(m_uesPerSlice[s]) /
                                         m_cfg.maxUes[s]);
    }

    // -----------------------------------------------------------------------
    // Reward Function v2: throughput utility + continuous SLA margin +
    //            PRB-efficiency Jain - heavy penalty per SLA violation
    //
    // Key improvements over v1:
    //   - Jain computed on (thrNorm/prbFrac) not raw Mbps: penalises
    //     degenerate allocations like (1,1,23) even when SLA is met.
    //   - Continuous SLA margin replaces binary satNorm: agent receives
    //     gradient above AND below the SLA threshold, not just a cliff.
    //   - All three positive terms normalised to [0,1]; max reward = 1.0/step.
    // -----------------------------------------------------------------------
    double thrNormAvg   = 0.0;
    double slaMarginAvg = 0.0;
    uint32_t slaViolations = 0;
    std::array<double, kSliceCount> efficiency{};

    for (uint8_t s = 0; s < kSliceCount; ++s)
    {
        const double thrNorm =
            std::min(1.0, m_thrMbps[s] / std::max(1e-9, m_cfg.maxThrMbps[s]));
        const double prbFrac =
            static_cast<double>(m_prbAlloc[s]) /
            static_cast<double>(m_cfg.totalPrbs);

        // SLA check (unchanged semantics)
        const bool slaSat =
            (m_thrMbps[s] >= m_cfg.minThrMbps[s]) &&
            (m_latMs[s]   <= m_cfg.maxLatMs[s]);

        // Continuous SLA margin: +1 = comfortably above, 0 = exactly at
        // threshold, -1 = far below. Clamped to [-1, +1].
        const double thrMargin = std::max(-1.0, std::min(1.0,
            (m_thrMbps[s] - m_cfg.minThrMbps[s]) /
            std::max(1e-9, m_cfg.minThrMbps[s])));
        const double latMargin = std::max(-1.0, std::min(1.0,
            (m_cfg.maxLatMs[s] - m_latMs[s]) /
            std::max(1e-9, m_cfg.maxLatMs[s])));

        thrNormAvg   += thrNorm;
        slaMarginAvg += 0.5 * thrMargin + 0.5 * latMargin;  // in [-1,+1]
        // Inactive mMTC (IoT duty-cycle off-period) is physics, not a policy
        // failure. Exclude from violation penalty; keep in Jain fairness above.
        const bool mmtcInactive = (s == MMTC && m_thrMbps[s] < 0.001);
        slaViolations += (slaSat || mmtcInactive) ? 0 : 1;

        // PRB efficiency: normalised throughput delivered per unit PRB
        // fraction. High for slices that use their PRBs well; low for
        // slices that hoard PRBs without delivering throughput.
        efficiency[s] = thrNorm / std::max(1e-9, prbFrac);
    }

    thrNormAvg   /= kSliceCount;
    slaMarginAvg /= kSliceCount;   // in [-1, +1]

    // Map slaMarginAvg from [-1,+1] to [0,1] so all positive terms share
    // the same scale and the formula is easy to reason about.
    const double slaMarginNorm = (slaMarginAvg + 1.0) * 0.5;

    // Jain's fairness index on PRB efficiency (not raw throughput).
    // Balanced allocations score near 1.0; degenerate ones score << 1.0.
    const double esum =
        std::accumulate(efficiency.begin(), efficiency.end(), 0.0);
    double esum2 = 0.0;
    for (double e : efficiency)
    {
        esum2 += e * e;
    }
    const double effJain =
        (esum2 > 0.0) ? ((esum * esum) / (kSliceCount * esum2)) : 0.0;

    m_reward = static_cast<float>(
        0.35 * thrNormAvg    +
        0.30 * slaMarginNorm +
        0.35 * effJain       -
        2.0  * static_cast<double>(slaViolations));

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
