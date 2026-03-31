#include "slice-env.h"

#include "ns3/double.h"
#include "ns3/integer.h"
#include "ns3/log.h"
#include "ns3/nr-ue-net-device.h"
#include "ns3/nr-ue-mac.h"
#include "ns3/random-variable-stream.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE("NrSliceGymEnv");
NS_OBJECT_ENSURE_REGISTERED(NrSliceGymEnv);

TypeId
NrSliceGymEnv::GetTypeId()
{
  static TypeId tid = TypeId("ns3::NrSliceGymEnv").SetParent<OpenGymEnv>().AddConstructor<NrSliceGymEnv>();
  return tid;
}

NrSliceGymEnv::NrSliceGymEnv()
  : m_prbAlloc{10, 8, 7}
{
}

NrSliceGymEnv::~NrSliceGymEnv() = default;

void
NrSliceGymEnv::Initialize(const SimConfig &cfg,
                          Ptr<NrHelper> nrHelper,
                          const NetDeviceContainer &gnbDevs,
                          const std::array<NetDeviceContainer, 3> &ueDevsBySlice)
{
  m_cfg = cfg;
  m_nrHelper = nrHelper;
  m_gnbDevs = gnbDevs;
  m_ueDevsBySlice = ueDevsBySlice;
  m_prbAlloc = cfg.initPrb;
  for (uint8_t s = 0; s < 3; ++s)
  {
    m_metrics[s].ueCount = m_ueDevsBySlice[s].GetN();
  }
  Simulator::Schedule(Seconds(m_cfg.stepS), &NrSliceGymEnv::ScheduleStep, this);
}

void
NrSliceGymEnv::BuildImsiSliceMap()
{
  m_imsiToSlice.clear();
  for (uint8_t s = 0; s < 3; ++s)
  {
    for (uint32_t i = 0; i < m_ueDevsBySlice[s].GetN(); ++i)
    {
      Ptr<NrUeNetDevice> ue = DynamicCast<NrUeNetDevice>(m_ueDevsBySlice[s].Get(i));
      if (ue)
      {
        m_imsiToSlice[ue->GetImsi()] = static_cast<SliceId>(s);
      }
    }
  }
}

void
NrSliceGymEnv::AttachFlowMonitor(Ptr<FlowMonitor> fm)
{
  m_flowMonitor = fm;
}

void
NrSliceGymEnv::OnSchedulerNotify(const std::vector<NrMacSchedulerUeInfoAi::LcObservation> &,
                                 bool gameOver,
                                 float,
                                 const std::string &extra,
                                 const NrMacSchedulerUeInfoAi::UpdateAllUeWeightsFn &updateFn)
{
  m_gameOver = gameOver;
  m_extraInfo = extra;
  m_updateWeightsFn = updateFn;
  BuildRntiToSliceMapLazily();
  ApplySliceWeights();
}

void
NrSliceGymEnv::BuildRntiToSliceMapLazily()
{
  if (!m_rntiToSlice.empty())
  {
    return;
  }
  for (uint8_t s = 0; s < 3; ++s)
  {
    for (uint32_t i = 0; i < m_ueDevsBySlice[s].GetN(); ++i)
    {
      Ptr<NrUeNetDevice> ueDev = DynamicCast<NrUeNetDevice>(m_ueDevsBySlice[s].Get(i));
      if (!ueDev || !ueDev->GetMac(0))
      {
        continue;
      }
      uint16_t rnti = ueDev->GetMac(0)->GetRnti();
      if (rnti == std::numeric_limits<uint16_t>::max())
      {
        continue;
      }
      m_rntiToSlice[rnti] = static_cast<SliceId>(s);
    }
  }
}

Ptr<OpenGymSpace>
NrSliceGymEnv::GetObservationSpace()
{
  std::vector<uint32_t> shape = {15};
  return CreateObject<OpenGymBoxSpace>(0.0, 1.0, shape, TypeNameGet<float>());
}

Ptr<OpenGymSpace>
NrSliceGymEnv::GetActionSpace()
{
  return CreateObject<OpenGymDiscreteSpace>(27);
}

Ptr<OpenGymDataContainer>
NrSliceGymEnv::GetObservation()
{
  auto box = CreateObject<OpenGymBoxContainer<float>>(std::vector<uint32_t>{15});
  for (uint8_t s = 0; s < 3; ++s)
  {
    box->AddValue(static_cast<float>(m_prbAlloc[s]) / static_cast<float>(m_cfg.totalPrbs));
  }
  for (uint8_t s = 0; s < 3; ++s)
  {
    box->AddValue(std::min(1.0f, static_cast<float>(m_metrics[s].thrMbps / m_cfg.maxThrMbps[s])));
  }
  for (uint8_t s = 0; s < 3; ++s)
  {
    box->AddValue(std::min(1.0f, static_cast<float>(m_metrics[s].latMs / m_cfg.maxLatMs[s])));
  }
  for (uint8_t s = 0; s < 3; ++s)
  {
    box->AddValue(static_cast<float>(std::clamp(m_metrics[s].queueOcc, 0.0, 1.0)));
  }
  for (uint8_t s = 0; s < 3; ++s)
  {
    box->AddValue(std::min(1.0f, static_cast<float>(m_metrics[s].ueCount) / static_cast<float>(m_cfg.maxUes)));
  }
  return box;
}

float
NrSliceGymEnv::GetReward()
{
  return m_lastReward;
}

bool
NrSliceGymEnv::GetGameOver()
{
  return m_gameOver || (m_stepCounter >= m_cfg.maxSteps);
}

std::string
NrSliceGymEnv::GetExtraInfo()
{
  return m_extraInfo;
}

std::array<int32_t, 3>
NrSliceGymEnv::DecodeAction(uint32_t actionId) const
{
  std::array<int32_t, 3> d{};
  for (int i = 0; i < 3; ++i)
  {
    d[i] = static_cast<int32_t>(actionId % 3) - 1;
    actionId /= 3;
  }
  return d;
}

bool
NrSliceGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
  Ptr<OpenGymDiscreteContainer> discrete = DynamicCast<OpenGymDiscreteContainer>(action);
  if (!discrete)
  {
    return false;
  }
  uint32_t actionId = discrete->GetValue();
  auto delta = DecodeAction(actionId);
  for (uint8_t s = 0; s < 3; ++s)
  {
    int64_t v = static_cast<int64_t>(m_prbAlloc[s]) + delta[s];
    m_prbAlloc[s] = static_cast<uint32_t>(std::max<int64_t>(1, v));
  }
  EnforceConstraints();
  ApplySliceWeights();
  return true;
}

void
NrSliceGymEnv::EnforceConstraints()
{
  int32_t sum = static_cast<int32_t>(m_prbAlloc[0] + m_prbAlloc[1] + m_prbAlloc[2]);
  while (sum > static_cast<int32_t>(m_cfg.totalPrbs))
  {
    uint8_t idx = std::distance(m_prbAlloc.begin(), std::max_element(m_prbAlloc.begin(), m_prbAlloc.end()));
    if (m_prbAlloc[idx] > 1)
    {
      --m_prbAlloc[idx];
      --sum;
    }
    else
    {
      break;
    }
  }
  while (sum < static_cast<int32_t>(m_cfg.totalPrbs))
  {
    uint8_t idx = std::distance(m_prbAlloc.begin(), std::min_element(m_prbAlloc.begin(), m_prbAlloc.end()));
    ++m_prbAlloc[idx];
    ++sum;
  }
}

void
NrSliceGymEnv::ApplySliceWeights()
{
  if (!m_updateWeightsFn)
  {
    return;
  }
  NrMacSchedulerUeInfoAi::UeWeightsMap weights;
  for (auto const &[rnti16, slice] : m_rntiToSlice)
  {
    uint8_t rnti = static_cast<uint8_t>(rnti16);
    double base = static_cast<double>(m_prbAlloc[slice]) / static_cast<double>(m_cfg.totalPrbs);
    double perUe = base / std::max<uint32_t>(1, m_ueDevsBySlice[slice].GetN());
    weights[rnti][1] = perUe;
  }
  m_updateWeightsFn(weights);
}

void
NrSliceGymEnv::ScheduleStep()
{
  AggregateFlowStats();
  UpdateRewardAndDone();
  Notify();
  ++m_stepCounter;
  if (!GetGameOver())
  {
    Simulator::Schedule(Seconds(m_cfg.stepS), &NrSliceGymEnv::ScheduleStep, this);
  }
}

void
NrSliceGymEnv::AggregateFlowStats()
{
  for (auto &m : m_metrics)
  {
    m.thrMbps = 0.0;
    m.latMs = 0.0;
    m.loss = 0.0;
    m.queueOcc = 0.0;
    m.packets = 0;
  }

  // Lightweight synthetic fallback metrics for stable training scaffolding.
  for (uint8_t s = 0; s < 3; ++s)
  {
    double prbFrac = static_cast<double>(m_prbAlloc[s]) / static_cast<double>(m_cfg.totalPrbs);
    m_metrics[s].thrMbps = prbFrac * m_cfg.maxThrMbps[s] * 0.85;
    m_metrics[s].latMs = m_cfg.maxLatMs[s] * (1.0 - 0.7 * prbFrac);
    m_metrics[s].queueOcc = std::clamp(1.0 - prbFrac, 0.0, 1.0);
    m_metrics[s].loss = std::clamp(0.05 * (1.0 - prbFrac), 0.0, 1.0);
  }
}

double
NrSliceGymEnv::JainIndex(const std::array<double, 3> &x) const
{
  double sum = x[0] + x[1] + x[2];
  double sumSq = x[0] * x[0] + x[1] * x[1] + x[2] * x[2];
  if (sumSq <= 1e-9)
  {
    return 0.0;
  }
  return (sum * sum) / (3.0 * sumSq);
}

void
NrSliceGymEnv::UpdateRewardAndDone()
{
  const std::array<double, 3> minThr{10.0, 1.0, 0.1};
  const std::array<double, 3> maxLat{50.0, 1.0, 500.0};

  double thrTerm = 0.0;
  double satTerm = 0.0;
  int slaViol = 0;
  std::array<double, 3> thr{};

  for (uint8_t s = 0; s < 3; ++s)
  {
    thr[s] = m_metrics[s].thrMbps;
    thrTerm += std::min(1.0, m_metrics[s].thrMbps / m_cfg.maxThrMbps[s]);
    double sat = m_metrics[s].thrMbps >= minThr[s] ? 1.0 : 0.0;
    double latNorm = std::min(1.0, m_metrics[s].latMs / maxLat[s]);
    satTerm += sat * (1.0 - latNorm);
    bool viol = (m_metrics[s].thrMbps < minThr[s]) || (m_metrics[s].latMs > maxLat[s]);
    if (viol)
    {
      ++slaViol;
    }
  }

  double reward = 0.5 * (thrTerm / 3.0) + 0.3 * (satTerm / 3.0) + 0.2 * JainIndex(thr) - 2.0 * slaViol;
  m_lastReward = static_cast<float>(reward);
}

} // namespace ns3
