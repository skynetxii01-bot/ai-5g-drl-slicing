#include "slice-baseline-env.h"

#include "ns3/integer.h"
#include "ns3/nr-ue-mac.h"
#include "ns3/nr-ue-net-device.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("NrSliceBaselineEnv");
NS_OBJECT_ENSURE_REGISTERED(NrSliceBaselineEnv);

// ---------------------------------------------------------------------------
// Anonymous helpers
// ---------------------------------------------------------------------------
namespace
{

// Identical action-delta table to slice-env.cc
constexpr std::array<std::array<int8_t, NrSliceBaselineEnv::kSliceCount>,
                     NrSliceBaselineEnv::kActionCount>
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

// ── Statistics helpers ──────────────────────────────────────────────────────

double
VecMean(const std::vector<double>& v)
{
    if (v.empty()) return 0.0;
    double s = 0.0;
    for (double x : v) s += x;
    return s / static_cast<double>(v.size());
}

double
VecStd(const std::vector<double>& v)
{
    if (v.size() < 2) return 0.0;
    const double m  = VecMean(v);
    double       sq = 0.0;
    for (double x : v) sq += (x - m) * (x - m);
    return std::sqrt(sq / static_cast<double>(v.size()));
}

double
NanMean(const std::vector<double>& v)
{
    int    n = 0;
    double s = 0.0;
    for (double x : v)
    {
        if (!std::isnan(x))
        {
            s += x;
            ++n;
        }
    }
    return (n > 0) ? s / n : std::numeric_limits<double>::quiet_NaN();
}

// ── JSON helpers ─────────────────────────────────────────────────────────────

std::string
Jd(const std::string& key, double val)
{
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6);
    if (std::isnan(val))
        ss << "\"" << key << "\": null";
    else
        ss << "\"" << key << "\": " << val;
    return ss.str();
}

std::string
Ji(const std::string& key, int val)
{
    return "\"" + key + "\": " + std::to_string(val);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Constructor / TypeId
// ---------------------------------------------------------------------------

NrSliceBaselineEnv::NrSliceBaselineEnv()
{
    m_obs.fill(0.0f);
    m_thrMbps.fill(0.0);
    m_latMs.fill(0.0);
    m_holNorm.fill(0.0);
    m_holSumMs.fill(0.0);
    m_holSamples.fill(0);
    m_demandActive.fill(0);
    m_servedDemandRatio.fill(0.0);
}

TypeId
NrSliceBaselineEnv::GetTypeId()
{
    static TypeId tid = TypeId("ns3::NrSliceBaselineEnv")
                            .SetParent<Object>()
                            .AddConstructor<NrSliceBaselineEnv>();
    return tid;
}

// ---------------------------------------------------------------------------
// Policy registry
// ---------------------------------------------------------------------------

void
NrSliceBaselineEnv::AddPolicy(const std::string& name, PolicyFn fn)
{
    m_policyNames.push_back(name);
    m_policies.push_back(std::move(fn));
    m_results.emplace_back();
    m_results.back().name = name;
}

// ---------------------------------------------------------------------------
// NS-3 setup
// ---------------------------------------------------------------------------

void
NrSliceBaselineEnv::SetFlowMonitor(const Ptr<FlowMonitor>&        fm,
                                   const Ptr<Ipv4FlowClassifier>& fc)
{
    m_flowMonitor    = fm;
    m_flowClassifier = fc;
}

void
NrSliceBaselineEnv::Initialize(
    const Config&                                      cfg,
    const Ptr<NrHelper>&                               nrHelper,
    const NetDeviceContainer&                          gnbDevs,
    const std::array<NetDeviceContainer, kSliceCount>& ueDevsBySlice)
{
    m_cfg           = cfg;
    m_nrHelper      = nrHelper;
    m_gnbDevs       = gnbDevs;
    m_ueDevsBySlice = ueDevsBySlice;
    m_prbAlloc      = cfg.initialPrbAlloc;

    m_uesPerSlice = {ueDevsBySlice[EMBB].GetN(),
                     ueDevsBySlice[URLLC].GetN(),
                     ueDevsBySlice[MMTC].GetN()};

    BuildImsiSliceMap();

    m_initialized = true;
    m_globalStep  = 0;

    NS_LOG_INFO("NrSliceBaselineEnv: " << m_policies.size() << " policies, "
                                       << m_numEpisodes << " ep × " << m_maxSteps << " steps.");

    StartPolicy(0);
    Simulator::Schedule(m_cfg.stepInterval, &NrSliceBaselineEnv::ScheduleStep, this);
}

// ---------------------------------------------------------------------------
// RNTI / IMSI maps
// ---------------------------------------------------------------------------

void
NrSliceBaselineEnv::BuildImsiSliceMap()
{
    m_imsiToSlice.clear();
    for (uint8_t s = 0; s < kSliceCount; ++s)
    {
        for (uint32_t i = 0; i < m_ueDevsBySlice[s].GetN(); ++i)
        {
            Ptr<NrUeNetDevice> dev = DynamicCast<NrUeNetDevice>(m_ueDevsBySlice[s].Get(i));
            if (dev)
                m_imsiToSlice[dev->GetImsi()] = s;
        }
    }
}

void
NrSliceBaselineEnv::TryBuildRntiSliceMap()
{
    if (m_rntiMapReady)
        return;

    m_rntiToSlice.clear();
    bool allResolved = true;

    for (uint8_t s = 0; s < kSliceCount; ++s)
    {
        for (uint32_t i = 0; i < m_ueDevsBySlice[s].GetN(); ++i)
        {
            Ptr<NrUeNetDevice> dev = DynamicCast<NrUeNetDevice>(m_ueDevsBySlice[s].Get(i));
            if (!dev || !dev->GetMac(0))
            {
                allResolved = false;
                continue;
            }
            const uint16_t rnti = dev->GetMac(0)->GetRnti();
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

uint8_t
NrSliceBaselineEnv::ResolveSliceFromPort(uint16_t port) const
{
    if (port >= 1000 && port <= 1999) return EMBB;
    if (port >= 2000 && port <= 2999) return URLLC;
    if (port >= 3000 && port <= 3999) return MMTC;
    return kSliceCount; // sentinel
}

uint8_t
NrSliceBaselineEnv::ResolveSliceFromAddress(const Ipv4Address& dst) const
{
    auto it = m_ipToSlice.find(dst);
    return (it != m_ipToSlice.end()) ? it->second : kSliceCount;
}

// ---------------------------------------------------------------------------
// Scheduler callback
// ---------------------------------------------------------------------------

void
NrSliceBaselineEnv::OnSchedulerNotify(
    const std::vector<NrMacSchedulerUeInfoAi::LcObservation>& observations,
    bool /* isGameOver */,
    float /* reward */,
    const std::string& /* extraInfo */,
    const NrMacSchedulerUeInfoAi::UpdateAllUeWeightsFn& updateWeightsFn)
{
    m_lastLcObs       = observations;
    m_updateWeightsFn = updateWeightsFn;

    TryBuildRntiSliceMap();

    if (m_rntiMapReady)
    {
        for (const auto& obs : observations)
        {
            auto it = m_rntiToSlice.find(obs.rnti);
            if (it == m_rntiToSlice.end() || it->second >= kSliceCount)
                continue;
            const uint8_t s = it->second;
            m_holSumMs[s]  += static_cast<double>(obs.holDelay);
            ++m_holSamples[s];
        }
    }

    ApplySliceWeights();
}

// ---------------------------------------------------------------------------
// Flow stats — identical logic to slice-env.cc AggregateFlowStats()
// ---------------------------------------------------------------------------

void
NrSliceBaselineEnv::AggregateFlowStats()
{
    m_thrMbps.fill(0.0);
    if (!m_flowMonitor || !m_flowClassifier)
        return;

    const auto stats = m_flowMonitor->GetFlowStats();

    std::array<double,   kSliceCount> totalDeltaDelayMs{0.0, 0.0, 0.0};
    std::array<uint64_t, kSliceCount> totalDeltaPkts{0, 0, 0};

    for (const auto& [flowId, st] : stats)
    {
        Ipv4FlowClassifier::FiveTuple ft = m_flowClassifier->FindFlow(flowId);

        uint8_t slice = ResolveSliceFromAddress(ft.destinationAddress);
        if (slice >= kSliceCount)
            slice = ResolveSliceFromPort(ft.destinationPort);
        if (slice >= kSliceCount)
            continue;

        // Throughput delta
        const uint64_t prevBytes  = m_prevRxBytes.count(flowId) ? m_prevRxBytes[flowId] : 0;
        const uint64_t deltaBytes = (st.rxBytes >= prevBytes) ? (st.rxBytes - prevBytes)
                                                              : st.rxBytes;
        m_prevRxBytes[flowId] = st.rxBytes;
        m_thrMbps[slice] += (static_cast<double>(deltaBytes) * 8.0 / 1e6) /
                             std::max(1e-9, m_cfg.stepInterval.GetSeconds());

        // Latency delta
        const uint64_t prevPkts  = m_prevRxPackets.count(flowId) ? m_prevRxPackets[flowId] : 0;
        const double   prevDelay = m_prevDelaySum.count(flowId)   ? m_prevDelaySum[flowId]  : 0.0;
        const uint64_t deltaPkts  = (st.rxPackets >= prevPkts) ? (st.rxPackets - prevPkts) : 0;
        const double   deltaDelay = (st.delaySum.GetSeconds() >= prevDelay)
                                        ? (st.delaySum.GetSeconds() - prevDelay)
                                        : 0.0;
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
            m_latMs[s] = totalDeltaDelayMs[s] / static_cast<double>(totalDeltaPkts[s]);
        // Clear stale latency for inactive slices (mirrors slice-env.cc)
        if (m_thrMbps[s] <= 0.001)
            m_latMs[s] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// HOL delay — identical logic to slice-env.cc AggregateHolDelay()
// ---------------------------------------------------------------------------

void
NrSliceBaselineEnv::AggregateHolDelay()
{
    TryBuildRntiSliceMap();
    if (!m_rntiMapReady)
        return;

    for (uint8_t s = 0; s < kSliceCount; ++s)
    {
        if (m_holSamples[s] == 0)
            continue; // sample-and-hold — keep previous value

        const double meanHolMs =
            m_holSumMs[s] / static_cast<double>(m_holSamples[s]);
        m_holNorm[s]    = Clamp01(meanHolMs / std::max(1e-9, 2.0 * m_cfg.maxLatMs[s]));
        m_holSumMs[s]   = 0.0;
        m_holSamples[s] = 0;
    }
}

// ---------------------------------------------------------------------------
// EnforceConstraints — identical to slice-env.cc
// ---------------------------------------------------------------------------

void
NrSliceBaselineEnv::EnforceConstraints()
{
    for (auto& prb : m_prbAlloc)
        prb = std::max<uint16_t>(1, prb);

    int32_t diff =
        static_cast<int32_t>(m_cfg.totalPrbs) -
        static_cast<int32_t>(m_prbAlloc[0] + m_prbAlloc[1] + m_prbAlloc[2]);

    while (diff > 0)
    {
        const uint8_t minSlice = static_cast<uint8_t>(
            std::distance(m_prbAlloc.begin(),
                          std::min_element(m_prbAlloc.begin(), m_prbAlloc.end())));
        ++m_prbAlloc[minSlice];
        --diff;
    }

    int safety = 0;
    while (diff < 0)
    {
        bool removed = false;
        for (uint8_t i = 0; i < kSliceCount && diff < 0; ++i)
        {
            const uint8_t idx = (m_globalStep + i) % kSliceCount;
            if (m_prbAlloc[idx] > 1)
            {
                --m_prbAlloc[idx];
                ++diff;
                removed = true;
            }
        }
        if (!removed || ++safety > 100)
        {
            NS_LOG_WARN("EnforceConstraints: cannot converge — resetting to initialPrbAlloc");
            m_prbAlloc = m_cfg.initialPrbAlloc;
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// ApplySliceWeights — identical to slice-env.cc
// ---------------------------------------------------------------------------

void
NrSliceBaselineEnv::ApplySliceWeights()
{
    if (!m_updateWeightsFn || m_lastLcObs.empty())
        return;

    TryBuildRntiSliceMap();

    NrMacSchedulerUeInfoAi::UeWeightsMap weightsMap;

    for (const auto& obs : m_lastLcObs)
    {
        auto it = m_rntiToSlice.find(obs.rnti);
        if (it == m_rntiToSlice.end() || it->second >= kSliceCount)
            continue;

        const uint8_t s = it->second;
        const double weight =
            (static_cast<double>(m_prbAlloc[s]) / m_cfg.totalPrbs) /
            std::max(1.0, static_cast<double>(m_uesPerSlice[s]));

        if (obs.rnti > 255)
        {
            NS_LOG_ERROR("ApplySliceWeights: RNTI " << obs.rnti << " > 255 — skipping.");
            continue;
        }
        weightsMap[static_cast<uint8_t>(obs.rnti)][obs.lcId] = weight;
    }

    if (!weightsMap.empty())
        m_updateWeightsFn(weightsMap);
}

// ---------------------------------------------------------------------------
// Episode / policy state machine
// ---------------------------------------------------------------------------

void
NrSliceBaselineEnv::StartPolicy(size_t idx)
{
    m_curPolicyIdx = idx;
    m_curEpisode   = 0;
    // Reset PRB so every policy starts from the same initial state
    m_prbAlloc = m_cfg.initialPrbAlloc;
    // Clear HOL so stale values from previous policy don't bleed in
    m_holNorm.fill(0.0);
    m_holSumMs.fill(0.0);
    m_holSamples.fill(0);

    std::cout << "\n=== Policy: " << m_policyNames[idx]
              << "  (" << m_numEpisodes << " ep × " << m_maxSteps << " steps) ===\n";

    StartEpisode();
}

void
NrSliceBaselineEnv::StartEpisode()
{
    ResetStepAccumulators();
    m_curStep = 0;
}

void
NrSliceBaselineEnv::ResetStepAccumulators()
{
    m_epReward = m_epSla = m_epSlaE = m_epSlaU = m_epSlaM = 0.0;
    m_epEThr = m_epUThr = m_epMThr = 0.0;
    m_epELat = 0.0;  m_epELatN = 0;
    m_epULat = 0.0;  m_epULatN = 0;
    m_epMLat = 0.0;  m_epMLatN = 0;
    m_epHolE = m_epHolU = m_epHolM = 0.0;
    m_epEffE = m_epEffU = m_epEffM = 0.0;
    m_epPrbE = m_epPrbU = m_epPrbM = 0.0;
    m_epRwdThr = m_epRwdSla = m_epRwdEff = m_epRwdViol = m_epActive = 0.0;
}

void
NrSliceBaselineEnv::EndEpisode()
{
    const int n = std::max(1, m_curStep);
    PolicyResult& r = m_results[m_curPolicyIdx];

    r.rewards.push_back(m_epReward);
    r.slaRates.push_back(m_epSla   / n);
    r.slaEmbb.push_back(m_epSlaE   / n);
    r.slaUrllc.push_back(m_epSlaU  / n);
    r.slaMmtc.push_back(m_epSlaM   / n);
    r.embbThrMbps.push_back(m_epEThr / n);
    r.urllcThrMbps.push_back(m_epUThr / n);
    r.mmtcThrMbps.push_back(m_epMThr / n);
    r.embbLatMs.push_back(m_epELatN > 0 ? m_epELat / m_epELatN
                                        : std::numeric_limits<double>::quiet_NaN());
    r.urllcLatMs.push_back(m_epULatN > 0 ? m_epULat / m_epULatN
                                         : std::numeric_limits<double>::quiet_NaN());
    r.mmtcLatMs.push_back(m_epMLatN > 0 ? m_epMLat / m_epMLatN
                                        : std::numeric_limits<double>::quiet_NaN());
    r.holEmbb.push_back(m_epHolE  / n);
    r.holUrllc.push_back(m_epHolU  / n);
    r.holMmtc.push_back(m_epHolM   / n);
    r.effEmbb.push_back(m_epEffE  / n);
    r.effUrllc.push_back(m_epEffU  / n);
    r.effMmtc.push_back(m_epEffM   / n);
    r.prbEmbb.push_back(m_epPrbE  / n);
    r.prbUrllc.push_back(m_epPrbU  / n);
    r.prbMmtc.push_back(m_epPrbM   / n);
    r.rwdThr.push_back(m_epRwdThr  / n);
    r.rwdSla.push_back(m_epRwdSla  / n);
    r.rwdEff.push_back(m_epRwdEff  / n);
    r.rwdViol.push_back(m_epRwdViol / n);
    r.activeSlices.push_back(m_epActive / n);

    std::cout << std::fixed << std::setprecision(3)
              << "  ep " << (m_curEpisode + 1) << "/" << m_numEpisodes
              << "  reward=" << m_epReward
              << "  sla="    << m_epSla / n
              << "  eMBB="   << m_epEThr / n << "Mbps"
              << "  viol="   << m_epRwdViol / n << "\n";

    ++m_curEpisode;

    if (m_curEpisode >= m_numEpisodes)
    {
        // This policy is finished
        if (m_curPolicyIdx + 1 < m_policies.size())
        {
            StartPolicy(m_curPolicyIdx + 1);
        }
        else
        {
            // All policies done — write results and stop scheduling
            WriteResults();
            m_allDone = true;
        }
    }
    else
    {
        StartEpisode();
    }
}

// ---------------------------------------------------------------------------
// Main ScheduleStep — mirrors slice-env.cc ScheduleStep() logic exactly,
// replacing the Notify() call with a direct C++ policy invocation.
// ---------------------------------------------------------------------------

void
NrSliceBaselineEnv::ScheduleStep()
{
    if (!m_initialized || m_allDone)
        return;

    ++m_globalStep;

    AggregateFlowStats();
    AggregateHolDelay();

    // ── Build observation vector (identical to slice-env.cc) ──────────────
    for (uint8_t s = 0; s < kSliceCount; ++s)
    {
        m_obs[s]     = Clamp01(static_cast<double>(m_prbAlloc[s]) / m_cfg.totalPrbs);
        m_obs[3 + s] = Clamp01(m_thrMbps[s] / m_cfg.maxThrMbps[s]);
        m_obs[6 + s] = Clamp01(m_latMs[s]   / (2.0 * m_cfg.maxLatMs[s]));
        m_obs[9 + s] = m_holNorm[s];

        {
            const double prbFrac    = static_cast<double>(m_prbAlloc[s]) / m_cfg.totalPrbs;
            const double thrNorm    = m_thrMbps[s] / std::max(1e-9, m_cfg.maxThrMbps[s]);
            const double thrNormSft = std::tanh(std::max(0.0, thrNorm));
            m_obs[12 + s] = Clamp01(thrNormSft / std::max(1e-9, prbFrac) / 25.0);
        }
    }

    // ── Reward computation (identical to slice-env.cc) ────────────────────
    double   thrNormAvg    = 0.0;
    double   slaMarginAvg  = 0.0;
    double   effNorm       = 0.0;
    double   violationRate = 0.0;
    double   totalThrMbps  = 0.0;
    uint32_t slaViolations = 0;
    uint32_t activeSlices  = 0;

    m_demandActive.fill(0);
    m_servedDemandRatio.fill(0.0);

    for (uint8_t s = 0; s < kSliceCount; ++s)
    {
        const bool demandActive = (m_thrMbps[s] > 0.001);
        m_demandActive[s]       = demandActive ? 1 : 0;
        if (!demandActive)
            continue;

        const double thrNorm =
            m_thrMbps[s] / std::max(1e-9, m_cfg.maxThrMbps[s]);
        const bool slaSat =
            (m_thrMbps[s] >= m_cfg.minThrMbps[s]) &&
            (m_latMs[s]   <= m_cfg.maxLatMs[s]);

        const double thrMargin = std::tanh(std::max(
            0.0, (m_thrMbps[s] - m_cfg.minThrMbps[s]) /
                     std::max(1e-9, m_cfg.minThrMbps[s])));
        const double latMargin = std::tanh(std::max(
            0.0, (m_cfg.maxLatMs[s] - m_latMs[s]) /
                     std::max(1e-9, m_cfg.maxLatMs[s])));

        slaViolations += slaSat ? 0 : 1;
        thrNormAvg    += std::tanh(std::max(0.0, thrNorm));
        slaMarginAvg  += 0.5 * thrMargin + 0.5 * latMargin;
        totalThrMbps  += m_thrMbps[s];
        ++activeSlices;

        m_servedDemandRatio[s] =
            std::min(2.0, m_thrMbps[s] / std::max(1e-9, m_cfg.minThrMbps[s]));
    }

    double reward = 0.0;
    if (activeSlices > 0)
    {
        const double n = static_cast<double>(activeSlices);
        thrNormAvg    /= n;
        slaMarginAvg  /= n;
        violationRate  = static_cast<double>(slaViolations) / n;

        if (totalThrMbps > 1e-9)
        {
            double alignScore = 0.0;
            for (uint8_t s = 0; s < kSliceCount; ++s)
            {
                if (m_demandActive[s] == 0)
                    continue;
                const double demandFrac = m_thrMbps[s] / totalThrMbps;
                const double prbFrac =
                    static_cast<double>(m_prbAlloc[s]) / m_cfg.totalPrbs;
                alignScore += 1.0 - std::abs(prbFrac - demandFrac);
            }
            effNorm = alignScore / n;
        }

        reward = 0.451 * thrNormAvg  +
                 0.292 * slaMarginAvg +
                 0.257 * effNorm      -
                 1.20  * violationRate;
    }

    // demand_active flags → obs[15:18]
    for (uint8_t s = 0; s < kSliceCount; ++s)
        m_obs[15 + s] = static_cast<float>(m_demandActive[s]);

    // ── SLA rate (mirrors Python compute_sla_rates) ───────────────────────
    int    slaDen = 0, slaSatCnt = 0;
    double slaE = 1.0, slaU = 1.0, slaM = 1.0;
    {
        double perSlice[3] = {1.0, 1.0, 1.0};
        for (uint8_t s = 0; s < kSliceCount; ++s)
        {
            if (m_demandActive[s] == 0)
                continue;
            ++slaDen;
            // latency SLA boundary: obs[6+s] <= 0.5  (matches Python)
            const bool ok = (m_thrMbps[s] >= m_cfg.minThrMbps[s]) &&
                            (static_cast<double>(m_obs[6 + s]) <= 0.5);
            perSlice[s] = ok ? 1.0 : 0.0;
            if (ok) ++slaSatCnt;
        }
        slaE = perSlice[EMBB];
        slaU = perSlice[URLLC];
        slaM = perSlice[MMTC];
    }
    const double slaRate =
        (slaDen == 0) ? 1.0 : static_cast<double>(slaSatCnt) / slaDen;

    // ── Call the current C++ policy ────────────────────────────────────────
    const int action = m_policies[m_curPolicyIdx](m_obs);
    NS_ASSERT_MSG(action >= 0 && action < static_cast<int>(kActionCount),
                  "Policy returned out-of-range action: " << action);

    // ── Apply action ───────────────────────────────────────────────────────
    for (uint32_t s = 0; s < kSliceCount; ++s)
    {
        const int32_t updated =
            static_cast<int32_t>(m_prbAlloc[s]) + kActionDelta[action][s];
        m_prbAlloc[s] = static_cast<uint16_t>(std::max(1, updated));
    }
    EnforceConstraints();
    ApplySliceWeights();

    // ── Accumulate episode metrics ─────────────────────────────────────────
    m_epReward += reward;
    m_epSla    += slaRate;
    m_epSlaE   += slaE;
    m_epSlaU   += slaU;
    m_epSlaM   += slaM;

    m_epEThr += m_thrMbps[EMBB];
    m_epUThr += m_thrMbps[URLLC];
    m_epMThr += m_thrMbps[MMTC];

    if (m_thrMbps[EMBB]  > 0.001) { m_epELat += m_latMs[EMBB];  ++m_epELatN; }
    if (m_thrMbps[URLLC] > 0.001) { m_epULat += m_latMs[URLLC]; ++m_epULatN; }
    if (m_thrMbps[MMTC]  > 0.001) { m_epMLat += m_latMs[MMTC];  ++m_epMLatN; }

    m_epHolE += m_holNorm[EMBB];
    m_epHolU += m_holNorm[URLLC];
    m_epHolM += m_holNorm[MMTC];

    m_epEffE += static_cast<double>(m_obs[12]);
    m_epEffU += static_cast<double>(m_obs[13]);
    m_epEffM += static_cast<double>(m_obs[14]);

    // PRB allocation in integer PRBs (not fraction), to match Python output
    m_epPrbE += static_cast<double>(m_prbAlloc[EMBB]);
    m_epPrbU += static_cast<double>(m_prbAlloc[URLLC]);
    m_epPrbM += static_cast<double>(m_prbAlloc[MMTC]);

    m_epRwdThr  += thrNormAvg;
    m_epRwdSla  += slaMarginAvg;
    m_epRwdEff  += effNorm;
    m_epRwdViol += violationRate;
    m_epActive  += static_cast<double>(activeSlices);

    ++m_curStep;

    // ── Episode boundary ───────────────────────────────────────────────────
    if (m_curStep >= m_maxSteps)
        EndEpisode(); // may set m_allDone = true

    // ── Schedule next step ─────────────────────────────────────────────────
    if (!m_allDone)
        Simulator::Schedule(m_cfg.stepInterval, &NrSliceBaselineEnv::ScheduleStep, this);
}

// ---------------------------------------------------------------------------
// JSON output — keys match evaluate.py exactly
// ---------------------------------------------------------------------------

void
NrSliceBaselineEnv::WriteResults() const
{
    std::cout << "\n[NrSliceBaselineEnv] Writing results to: " << m_outputPath << "\n";

    std::ostringstream js;
    js << std::fixed << std::setprecision(6);
    js << "{\n";

    for (size_t p = 0; p < m_results.size(); ++p)
    {
        const PolicyResult& r = m_results[p];

        js << "  \"" << r.name << "\": {\n";

        auto M  = [](const std::vector<double>& v) { return VecMean(v); };
        auto S  = [](const std::vector<double>& v) { return VecStd(v); };
        auto NM = [](const std::vector<double>& v) { return NanMean(v); };

        // clang-format off
        js << "    " << Jd("mean_reward",             M(r.rewards))          << ",\n"
           << "    " << Jd("std_reward",              S(r.rewards))          << ",\n"
           << "    " << Jd("mean_sla_rate",           M(r.slaRates))         << ",\n"
           << "    " << Jd("std_sla_rate",            S(r.slaRates))         << ",\n"
           << "    " << Jd("mean_sla_embb",           M(r.slaEmbb))          << ",\n"
           << "    " << Jd("std_sla_embb",            S(r.slaEmbb))          << ",\n"
           << "    " << Jd("mean_sla_urllc",          M(r.slaUrllc))         << ",\n"
           << "    " << Jd("std_sla_urllc",           S(r.slaUrllc))         << ",\n"
           << "    " << Jd("mean_sla_mmtc",           M(r.slaMmtc))          << ",\n"
           << "    " << Jd("std_sla_mmtc",            S(r.slaMmtc))          << ",\n"
           << "    " << Jd("mean_embb_thr_mbps",      M(r.embbThrMbps))      << ",\n"
           << "    " << Jd("std_embb_thr_mbps",       S(r.embbThrMbps))      << ",\n"
           << "    " << Jd("mean_urllc_thr_mbps",     M(r.urllcThrMbps))     << ",\n"
           << "    " << Jd("std_urllc_thr_mbps",      S(r.urllcThrMbps))     << ",\n"
           << "    " << Jd("mean_mmtc_thr_mbps",      M(r.mmtcThrMbps))      << ",\n"
           << "    " << Jd("std_mmtc_thr_mbps",       S(r.mmtcThrMbps))      << ",\n"
           << "    " << Jd("mean_embb_lat_ms",        NM(r.embbLatMs))       << ",\n"
           << "    " << Jd("std_embb_lat_ms",         S(r.embbLatMs))        << ",\n"
           << "    " << Jd("mean_urllc_lat_ms",       NM(r.urllcLatMs))      << ",\n"
           << "    " << Jd("std_urllc_lat_ms",        S(r.urllcLatMs))       << ",\n"
           << "    " << Jd("mean_mmtc_lat_ms",        NM(r.mmtcLatMs))       << ",\n"
           << "    " << Jd("std_mmtc_lat_ms",         S(r.mmtcLatMs))        << ",\n"
           << "    " << Jd("mean_hol_embb",           M(r.holEmbb))          << ",\n"
           << "    " << Jd("mean_hol_urllc",          M(r.holUrllc))         << ",\n"
           << "    " << Jd("mean_hol_mmtc",           M(r.holMmtc))          << ",\n"
           << "    " << Jd("mean_eff_embb",           M(r.effEmbb))          << ",\n"
           << "    " << Jd("mean_eff_urllc",          M(r.effUrllc))         << ",\n"
           << "    " << Jd("mean_eff_mmtc",           M(r.effMmtc))          << ",\n"
           << "    " << Jd("mean_prb_embb",           M(r.prbEmbb))          << ",\n"
           << "    " << Jd("std_prb_embb",            S(r.prbEmbb))          << ",\n"
           << "    " << Jd("mean_prb_urllc",          M(r.prbUrllc))         << ",\n"
           << "    " << Jd("std_prb_urllc",           S(r.prbUrllc))         << ",\n"
           << "    " << Jd("mean_prb_mmtc",           M(r.prbMmtc))          << ",\n"
           << "    " << Jd("std_prb_mmtc",            S(r.prbMmtc))          << ",\n"
           << "    " << Jd("mean_rwd_thr_norm",       M(r.rwdThr))           << ",\n"
           << "    " << Jd("mean_rwd_sla_margin",     M(r.rwdSla))           << ",\n"
           << "    " << Jd("mean_rwd_eff_norm",       M(r.rwdEff))           << ",\n"
           << "    " << Jd("mean_rwd_violation_rate", M(r.rwdViol))          << ",\n"
           << "    " << Jd("mean_active_slices",      M(r.activeSlices))     << ",\n"
           << "    " << Ji("episodes", static_cast<int>(r.rewards.size()))   << "\n";
        // clang-format on

        js << "  }" << (p + 1 < m_results.size() ? "," : "") << "\n";
    }
    js << "}\n";

    // Ensure output directory exists
    {
        const std::string& path = m_outputPath;
        const size_t       sl   = path.rfind('/');
        if (sl != std::string::npos)
            ::system(("mkdir -p " + path.substr(0, sl)).c_str());
    }

    std::ofstream f(m_outputPath);
    if (f.is_open())
    {
        f << js.str();
        std::cout << "[NrSliceBaselineEnv] Results written to " << m_outputPath << "\n";
    }
    else
    {
        std::cerr << "[NrSliceBaselineEnv] ERROR: cannot write to " << m_outputPath << "\n";
    }

    // ── Summary table ─────────────────────────────────────────────────────
    constexpr int col = 14;
    std::cout << "\n" << std::string(90, '=') << "\n"
              << std::left  << std::setw(col) << "Policy"
              << std::right << std::setw(9)   << "Reward"
              << std::setw(8)  << "SLA%"
              << std::setw(10) << "eMBB_Mbps"
              << std::setw(10) << "URLLC_ms"
              << std::setw(10) << "ViolRate" << "\n"
              << std::string(90, '-') << "\n";

    for (const auto& r : m_results)
    {
        std::cout << std::left  << std::setw(col) << r.name
                  << std::right << std::fixed     << std::setprecision(3)
                  << std::setw(9)  << VecMean(r.rewards)
                  << std::setw(7)  << VecMean(r.slaRates) * 100.0 << "%"
                  << std::setw(10) << VecMean(r.embbThrMbps)
                  << std::setw(10) << NanMean(r.urllcLatMs)
                  << std::setw(10) << VecMean(r.rwdViol) << "\n";
    }
    std::cout << std::string(90, '=') << "\n";
}

} // namespace ns3
