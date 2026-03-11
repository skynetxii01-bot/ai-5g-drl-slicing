#include "slice-env.h"

#include "ns3/log.h"
#include "ns3/open-gym-module.h"

#include <algorithm>

namespace ns3
{

NS_LOG_COMPONENT_DEFINE("NrSliceGymEnv");
NS_OBJECT_ENSURE_REGISTERED(NrSliceGymEnv);

TypeId
NrSliceGymEnv::GetTypeId()
{
  static TypeId tid = TypeId("ns3::NrSliceGymEnv").SetParent<OpenGymEnv>().AddConstructor<NrSliceGymEnv>();
  return tid;
}

NrSliceGymEnv::NrSliceGymEnv()
{
  m_prbAlloc = {9, 8, 8};
  BuildActionTable();
}

NrSliceGymEnv::~NrSliceGymEnv() = default;

void
NrSliceGymEnv::SetScheduler(Ptr<NrMacSchedulerTdmaAi> scheduler)
{
  m_scheduler = scheduler;
}

void
NrSliceGymEnv::SetUeRntisBySlice(const std::array<std::vector<uint16_t>, NUM_SLICES>& ueRntisBySlice)
{
  m_ueRntisBySlice = ueRntisBySlice;
}

void
NrSliceGymEnv::SetTotalPrbs(uint32_t totalPrbs)
{
  m_totalPrbs = totalPrbs;
  NormalizePrbs();
}

void
NrSliceGymEnv::SetStepTime(Time stepTime)
{
  m_stepTime = stepTime;
}

void
NrSliceGymEnv::SetEpisodeSteps(uint32_t episodeSteps)
{
  m_episodeSteps = episodeSteps;
}

void
NrSliceGymEnv::UpdateSliceMetrics(const std::array<SliceMetrics, NUM_SLICES>& metrics)
{
  m_metrics = metrics;
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
  std::vector<uint32_t> shape = {15};
  auto box = CreateObject<OpenGymBoxContainer<float>>(shape);

  for (uint8_t s = 0; s < NUM_SLICES; ++s)
  {
    box->AddValue(static_cast<float>(static_cast<double>(m_prbAlloc[s]) / m_totalPrbs));
  }
  for (uint8_t s = 0; s < NUM_SLICES; ++s)
  {
    box->AddValue(static_cast<float>(std::min(1.0, m_metrics[s].throughputMbps / m_maxThroughput[s])));
  }
  for (uint8_t s = 0; s < NUM_SLICES; ++s)
  {
    box->AddValue(static_cast<float>(std::min(1.0, m_metrics[s].latencyMs / m_maxLatency[s])));
  }
  for (uint8_t s = 0; s < NUM_SLICES; ++s)
  {
    box->AddValue(static_cast<float>(std::clamp(m_metrics[s].queueOccupancy, 0.0, 1.0)));
  }
  for (uint8_t s = 0; s < NUM_SLICES; ++s)
  {
    box->AddValue(static_cast<float>(static_cast<double>(m_ueRntisBySlice[s].size()) / m_maxUes));
  }

  return box;
}

float
NrSliceGymEnv::GetReward()
{
  return ComputeReward();
}

bool
NrSliceGymEnv::GetGameOver()
{
  return m_currentStep >= m_episodeSteps;
}

std::string
NrSliceGymEnv::GetExtraInfo()
{
  return "slice_step=" + std::to_string(m_currentStep);
}

bool
NrSliceGymEnv::ExecuteActions(Ptr<OpenGymDataContainer> action)
{
  auto discrete = DynamicCast<OpenGymDiscreteContainer>(action);
  if (!discrete)
  {
    NS_LOG_ERROR("Action container type mismatch");
    return false;
  }

  const uint32_t actionId = std::min<uint32_t>(26, discrete->GetValue());
  const auto& delta = m_actionTable[actionId];

  for (uint8_t s = 0; s < NUM_SLICES; ++s)
  {
    const int32_t next = static_cast<int32_t>(m_prbAlloc[s]) + delta[s];
    m_prbAlloc[s] = static_cast<uint32_t>(std::max(1, next));
  }

  NormalizePrbs();
  ApplyPrbAllocation();
  return true;
}

void
NrSliceGymEnv::NotifyStep()
{
  ++m_currentStep;
  Notify();
}

void
NrSliceGymEnv::BuildActionTable()
{
  for (int8_t de = -1; de <= 1; ++de)
  {
    for (int8_t du = -1; du <= 1; ++du)
    {
      for (int8_t dm = -1; dm <= 1; ++dm)
      {
        m_actionTable.push_back({de, du, dm});
      }
    }
  }
}

void
NrSliceGymEnv::ApplyPrbAllocation()
{
  if (!m_scheduler)
  {
    return;
  }

  NrMacSchedulerTdmaAi::UeWeightsMap weights;

  for (uint8_t s = 0; s < NUM_SLICES; ++s)
  {
    const auto& ues = m_ueRntisBySlice[s];
    if (ues.empty())
    {
      continue;
    }

    const double perUeWeight = (static_cast<double>(m_prbAlloc[s]) / static_cast<double>(m_totalPrbs)) /
                               static_cast<double>(ues.size());

    for (auto rnti : ues)
    {
      weights[rnti][4] = perUeWeight; // lcId=4 default DRB
    }
  }

  m_scheduler->UpdateAllUeWeightsDl(weights, std::vector<uint16_t>());
}

void
NrSliceGymEnv::NormalizePrbs()
{
  uint32_t sum = m_prbAlloc[0] + m_prbAlloc[1] + m_prbAlloc[2];
  while (sum > m_totalPrbs)
  {
    for (uint8_t s = 0; s < NUM_SLICES && sum > m_totalPrbs; ++s)
    {
      if (m_prbAlloc[s] > 1)
      {
        --m_prbAlloc[s];
        --sum;
      }
    }
  }
  while (sum < m_totalPrbs)
  {
    for (uint8_t s = 0; s < NUM_SLICES && sum < m_totalPrbs; ++s)
    {
      ++m_prbAlloc[s];
      ++sum;
    }
  }
}

float
NrSliceGymEnv::ComputeReward() const
{
  double throughputTerm = 0.0;
  double latencyTerm = 0.0;
  uint32_t violations = 0;

  for (uint8_t s = 0; s < NUM_SLICES; ++s)
  {
    const double thrNorm = std::min(1.0, m_metrics[s].throughputMbps / m_maxThroughput[s]);
    const double latNorm = std::min(1.0, m_metrics[s].latencyMs / m_maxLatency[s]);
    throughputTerm += thrNorm;
    latencyTerm += m_metrics[s].satisfaction * (1.0 - latNorm);
    violations += m_metrics[s].slaViolations;
  }

  const double fairness = ComputeJainFairness();
  const double reward = 0.5 * throughputTerm + 0.3 * latencyTerm + 0.2 * fairness - 2.0 * violations;
  return static_cast<float>(reward);
}

double
NrSliceGymEnv::ComputeJainFairness() const
{
  double sum = 0.0;
  double sqSum = 0.0;
  for (const auto& m : m_metrics)
  {
    sum += m.throughputMbps;
    sqSum += m.throughputMbps * m.throughputMbps;
  }
  if (sqSum == 0.0)
  {
    return 0.0;
  }
  return (sum * sum) / (NUM_SLICES * sqSum);
}

} // namespace ns3
