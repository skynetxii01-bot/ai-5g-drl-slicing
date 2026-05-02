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

void
NrSliceGymEnv::AggregateHolDelay()
{
    // Ensure RNTI map is ready before iterating observations.
    TryBuildRntiSliceMap();

    // If no fresh scheduler observations arrived this step, preserve the
    // previous m_holNorm values (sample-and-hold). Resetting to zero would
    // confuse the agent — zero means "no congestion," but the real meaning
    // here is "no new data." The two cases must be distinguishable.
    if (m_lastLcObservations.empty() || !m_rntiMapReady)
    {
        return;   // keep previous m_holNorm — do NOT fill(0.0)
    }

    // Only zero-fill when we have fresh data to replace it with.
    m_holNorm.fill(0.0);

    std::array<uint32_t, kSliceCount> lcCount{0, 0, 0};
    std::array<double,   kSliceCount> holSum {0.0, 0.0, 0.0};

    for (const auto& obs : m_lastLcObservations)
    {
        auto it = m_rntiToSlice.find(obs.rnti);
        if (it == m_rntiToSlice.end())
        {
            continue;
        }

        const uint8_t slice = it->second;
        if (slice >= kSliceCount)
        {
            continue;
        }

        holSum[slice] += static_cast<double>(obs.holDelay);
        ++lcCount[slice];
    }

    for (uint8_t s = 0; s < kSliceCount; ++s)
    {
        if (lcCount[s] == 0)
        {
            // No UEs from this slice had observations this callback —
            // keep previous value rather than zeroing.
            continue;
        }

        const double meanHolMs = holSum[s] / static_cast<double>(lcCount[s]);
        m_holNorm[s] = Clamp01(meanHolMs / std::max(1e-9, m_cfg.maxLatMs[s]));
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

void
NrSliceGymEnv::AggregateFlowStats()
{
    m_thrMbps.fill(0.0);
    m_latMs.fill(0.0);
    // Note: m_queueOcc removed — observation channel [9:12] now uses
    // m_holNorm (HOL-delay proxy) computed in AggregateHolDelay().

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

        const uint8_t slice = ResolveSliceFromPort(fiveTuple.destinationPort);
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
        const uint64_t prevPkts   = m_prevRxPackets.count(flowId)
                                  ? m_prevRxPackets[flowId] : 0;
        const double   prevDelay  = m_prevDelaySum.count(flowId)
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

    // Finalise per-slice latency as mean delay across all received packets.
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

    // Build normalised observation vector
    for (uint8_t s = 0; s < kSliceCount; ++s)
    {
        m_observation[s]      = Clamp01(static_cast<double>(m_prbAlloc[s]) /
                                        m_cfg.totalPrbs);
        m_observation[3 + s]  = Clamp01(m_thrMbps[s] / m_cfg.maxThrMbps[s]);
        m_observation[6 + s]  = Clamp01(m_latMs[s] / (2.0 * m_cfg.maxLatMs[s]));
        // obs[9:12] — HOL-delay congestion proxy, normalised by per-slice maxLatMs.
        // Rises before packet drops occur, giving the agent a pre-emptive signal.
        m_observation[9 + s]  = m_holNorm[s];
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
double esum  = 0.0;
double esum2 = 0.0;
uint32_t activeSlices = 0;

for (uint8_t s = 0; s < kSliceCount; ++s)
{
    const double thrNorm =
        std::min(1.0, m_thrMbps[s] / std::max(1e-9, m_cfg.maxThrMbps[s]));
    const double prbFrac =
        static_cast<double>(m_prbAlloc[s]) /
        static_cast<double>(m_cfg.totalPrbs);

    const bool slaSat =
        (m_thrMbps[s] >= m_cfg.minThrMbps[s]) &&
        (m_latMs[s]   <= m_cfg.maxLatMs[s]);

    const double thrMargin = std::max(-1.0, std::min(1.0,
        (m_thrMbps[s] - m_cfg.minThrMbps[s]) /
        std::max(1e-9, m_cfg.minThrMbps[s])));
    const double latMargin = std::max(-1.0, std::min(1.0,
        (m_cfg.maxLatMs[s] - m_latMs[s]) /
        std::max(1e-9, m_cfg.maxLatMs[s])));

	// A slice is "inactive" when the traffic source is in an exponential
	// off-period and genuinely not transmitting. In this case a zero-
	// throughput reading is not a policy failure — it is the correct
	// physical behavior of the OnOff traffic model. Penalising the agent
	// for off-period steps would introduce uncontrollable noise into the
	// reward signal and bias the policy to over-allocate PRBs to silent
	// slices.
	//
	// Duty cycles: mMTC ~10% (on=1s, off=9s), URLLC ~20% (on=50ms, off=200ms).
	// Both are frequently silent within a 100ms step window.
	// eMBB ~67% (on=2s, off=1s) — rarely fully silent, no exclusion needed.
	const bool mmtcInactive  = (s == MMTC  && m_thrMbps[s] < 0.001);
	const bool urllcInactive = (s == URLLC && m_thrMbps[s] < 0.001);
	const bool sliceInactive = mmtcInactive || urllcInactive;

	slaViolations += (slaSat || sliceInactive) ? 0 : 1;

	if (!sliceInactive)
	{
    	 thrNormAvg   += thrNorm;
   		 slaMarginAvg += 0.5 * thrMargin + 0.5 * latMargin;
   		 const double eff = thrNorm / std::max(1e-9, prbFrac);
		 esum  += eff;
   		 esum2 += eff * eff;
   		 ++activeSlices;
	}
}

	const uint32_t n = std::max(1u, activeSlices);
	thrNormAvg   /= n;
	slaMarginAvg /= n;
	const double slaMarginNorm = (slaMarginAvg + 1.0) * 0.5;
	const double effJain = (esum2 > 0.0)
	    ? ((esum * esum) / (n * esum2))
	    : 1.0;   // default fair when no active slices

    

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
