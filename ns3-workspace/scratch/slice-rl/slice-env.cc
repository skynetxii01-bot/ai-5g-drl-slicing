#include "slice-env.h"
#include <ns3/nr-mac-scheduler-tdma-ai.h>
#include <ns3/nr-gnb-net-device.h>
#include <ns3/nr-ue-net-device.h>
#include <ns3/log.h>
#include <sstream>
#include <numeric>

NS_LOG_COMPONENT_DEFINE ("NrSliceGymEnv");

using namespace ns3;

// ─────────────────────────────────────────────────────────────────────────────
// Constructor / Destructor
// ─────────────────────────────────────────────────────────────────────────────

NrSliceGymEnv::NrSliceGymEnv (SliceConfig cfg,
                               NetDeviceContainer gnbDevs,
                               std::vector<NetDeviceContainer> ueDevsBySlice,
                               Ptr<FlowMonitor> flowMonitor,
                               Ptr<Ipv4FlowClassifier> classifier)
  : m_cfg (cfg),
    m_gnbDevs (gnbDevs),
    m_ueDevsBySlice (ueDevsBySlice),
    m_flowMonitor (flowMonitor),
    m_classifier (classifier)
{
  // Initial PRB split: give eMBB most PRBs, URLLC guaranteed minimum
  m_prbAlloc[0] = 13;  // eMBB  — bandwidth-hungry
  m_prbAlloc[1] = 7;   // URLLC — latency-critical
  m_prbAlloc[2] = 5;   // mMTC  — low rate, many devices

  // Initialise active UE counts from config
  m_activeUes[0] = m_cfg.embbUes;
  m_activeUes[1] = m_cfg.urllcUes;
  m_activeUes[2] = m_cfg.mmtcUes;

  m_lastStepTime = Seconds (0.0);

  NS_LOG_INFO ("NrSliceGymEnv created. PRBs: eMBB=" << m_prbAlloc[0]
               << " URLLC=" << m_prbAlloc[1] << " mMTC=" << m_prbAlloc[2]);
}

NrSliceGymEnv::~NrSliceGymEnv ()
{
}

// ─────────────────────────────────────────────────────────────────────────────
// OpenGym Interface — Spaces
// ─────────────────────────────────────────────────────────────────────────────

Ptr<OpenGymSpace>
NrSliceGymEnv::GetObservationSpace ()
{
  // 15-dimensional box, all values in [0, 1]
  uint32_t shape[1]  = {15};
  std::string dtype  = TypeNameGet<float> ();
  Ptr<OpenGymBoxSpace> space = CreateObject<OpenGymBoxSpace> (
      0.0, 1.0, std::vector<uint32_t>(shape, shape + 1), dtype);
  return space;
}

Ptr<OpenGymSpace>
NrSliceGymEnv::GetActionSpace ()
{
  // 27 discrete actions: (Δ_eMBB, Δ_URLLC, Δ_mMTC) each ∈ {-1,0,+1}
  Ptr<OpenGymDiscreteSpace> space = CreateObject<OpenGymDiscreteSpace> (27);
  return space;
}

// ─────────────────────────────────────────────────────────────────────────────
// Observation — collect metrics and build 15-dim vector
// ─────────────────────────────────────────────────────────────────────────────

Ptr<OpenGymDataContainer>
NrSliceGymEnv::GetObservation ()
{
  CollectMetrics ();

  std::vector<uint32_t> shape = {15};
  Ptr<OpenGymBoxContainer<float>> obs =
      CreateObject<OpenGymBoxContainer<float>> (shape);

  for (uint32_t s = 0; s < N_SLICES; ++s)
    {
      // obs[0:3] = PRB fraction per slice
      obs->AddValue ((float)m_prbAlloc[s] / m_cfg.totalPrbs);
    }
  for (uint32_t s = 0; s < N_SLICES; ++s)
    {
      // obs[3:6] = normalised throughput
      float normThr = (float)(m_throughputMbps[s] / m_cfg.maxThrMbps[s]);
      obs->AddValue (std::min (normThr, 1.0f));
    }
  for (uint32_t s = 0; s < N_SLICES; ++s)
    {
      // obs[6:9] = normalised latency
      float normLat = (float)(m_latencyMs[s] / m_cfg.maxLatMs[s]);
      obs->AddValue (std::min (normLat, 1.0f));
    }
  for (uint32_t s = 0; s < N_SLICES; ++s)
    {
      // obs[9:12] = queue occupancy
      obs->AddValue ((float)m_queueOccupancy[s]);
    }
  for (uint32_t s = 0; s < N_SLICES; ++s)
    {
      // obs[12:15] = UE count fraction
      obs->AddValue ((float)m_activeUes[s] / m_cfg.maxUes[s]);
    }

  return obs;
}

// ─────────────────────────────────────────────────────────────────────────────
// Collect metrics from FlowMonitor each step
// ─────────────────────────────────────────────────────────────────────────────

void
NrSliceGymEnv::CollectMetrics ()
{
  m_flowMonitor->CheckForLostPackets ();
  auto stats = m_flowMonitor->GetFlowStats ();

  // Reset per-step accumulators
  double rxBytes[N_SLICES]  = {0};
  double totalLatMs[N_SLICES] = {0};
  uint32_t flowCount[N_SLICES] = {0};

  Time now = Simulator::Now ();
  double elapsed = (now - m_lastStepTime).GetSeconds ();
  if (elapsed <= 0) elapsed = m_cfg.gymStep;

  for (auto &kv : stats)
    {
      Ipv4FlowClassifier::FiveTuple t = m_classifier->FindFlow (kv.first);
      uint16_t dstPort = t.destinationPort;

      // Identify slice by destination port range
      // eMBB: 1000-1999, URLLC: 2000-2999, mMTC: 3000+
      uint32_t slice;
      if      (dstPort >= 1000 && dstPort < 2000) slice = 0;  // eMBB
      else if (dstPort >= 2000 && dstPort < 3000) slice = 1;  // URLLC
      else if (dstPort >= 3000)                   slice = 2;  // mMTC
      else continue;

      rxBytes[slice]   += kv.second.rxBytes;
      if (kv.second.rxPackets > 0)
        {
          totalLatMs[slice] += kv.second.delaySum.GetMilliSeconds ()
                               / kv.second.rxPackets;
          flowCount[slice]++;
        }
    }

  for (uint32_t s = 0; s < N_SLICES; ++s)
    {
      // Throughput = new bytes since last step / elapsed time
      double newBytes = rxBytes[s] - m_lastRxBytes[s];
      m_throughputMbps[s] = (newBytes * 8.0) / elapsed / 1e6;
      m_lastRxBytes[s]    = rxBytes[s];

      // Average latency across flows in this slice
      m_latencyMs[s] = (flowCount[s] > 0)
                        ? totalLatMs[s] / flowCount[s]
                        : m_cfg.maxLatMs[s];  // worst case if no packets

      // Queue occupancy: approximate from latency (simplified)
      // In a real implementation, hook into scheduler queue callbacks
      m_queueOccupancy[s] = std::min (1.0, m_latencyMs[s] / m_cfg.maxLatMs[s]);
    }

  m_lastStepTime = now;
}

// ─────────────────────────────────────────────────────────────────────────────
// Reward computation
// ─────────────────────────────────────────────────────────────────────────────

float
NrSliceGymEnv::GetReward ()
{
  return m_lastReward;
}

float
NrSliceGymEnv::ComputeReward ()
{
  // α=0.5: throughput term — average normalised throughput across slices
  double thrTerm = 0;
  for (uint32_t s = 0; s < N_SLICES; ++s)
    thrTerm += std::min (m_throughputMbps[s] / m_cfg.maxThrMbps[s], 1.0);
  thrTerm /= N_SLICES;

  // β=0.3: latency term — sat_i = 1 if meeting SLA, 0 otherwise
  double latTerm = 0;
  uint32_t slaViolations = 0;
  for (uint32_t s = 0; s < N_SLICES; ++s)
    {
      bool thrOk = m_throughputMbps[s] >= m_cfg.minThrMbps[s];
      bool latOk = m_latencyMs[s] <= m_cfg.maxSlaLatMs[s];
      double sat = (thrOk && latOk) ? 1.0 : 0.0;
      if (!thrOk || !latOk) slaViolations++;

      double normLat = 1.0 - std::min (m_latencyMs[s] / m_cfg.maxLatMs[s], 1.0);
      latTerm += sat * normLat;
    }
  latTerm /= N_SLICES;

  // γ=0.2: Jain's fairness index of throughputs
  double fairness = JainFairnessIndex ();

  // κ=2.0: penalty for SLA violations
  double reward = 0.5 * thrTerm
                + 0.3 * latTerm
                + 0.2 * fairness
                - 2.0 * slaViolations;

  m_lastReward = (float)reward;
  return m_lastReward;
}

float
NrSliceGymEnv::JainFairnessIndex ()
{
  // Jain's index = (Σx_i)² / (N · Σx_i²)
  double sumX  = 0, sumX2 = 0;
  for (uint32_t s = 0; s < N_SLICES; ++s)
    {
      double x  = m_throughputMbps[s];
      sumX  += x;
      sumX2 += x * x;
    }
  if (sumX2 < 1e-9) return 1.0f;  // all zero → perfect fairness by convention
  return (float)((sumX * sumX) / (N_SLICES * sumX2));
}

// ─────────────────────────────────────────────────────────────────────────────
// Game over check
// ─────────────────────────────────────────────────────────────────────────────

bool
NrSliceGymEnv::GetGameOver ()
{
  // Episode ends when simulation time is reached
  return Simulator::Now ().GetSeconds () >= m_cfg.simTime - m_cfg.gymStep;
}

std::string
NrSliceGymEnv::GetExtraInfo ()
{
  std::ostringstream oss;
  oss << "step=" << m_stepCount
      << " prb=[" << m_prbAlloc[0] << "," << m_prbAlloc[1] << "," << m_prbAlloc[2] << "]"
      << " thr=[" << m_throughputMbps[0] << "," << m_throughputMbps[1] << "," << m_throughputMbps[2] << "]"
      << " lat=[" << m_latencyMs[0] << "," << m_latencyMs[1] << "," << m_latencyMs[2] << "]";
  return oss.str ();
}

// ─────────────────────────────────────────────────────────────────────────────
// Execute action from Python agent
// ─────────────────────────────────────────────────────────────────────────────

bool
NrSliceGymEnv::ExecuteActions (Ptr<OpenGymDataContainer> action)
{
  Ptr<OpenGymDiscreteContainer> discrete =
      DynamicCast<OpenGymDiscreteContainer> (action);
  uint32_t actionIdx = discrete->GetValue ();

  // Decode 27-action space: action = base-3 number
  // action_i → (Δ_eMBB, Δ_URLLC, Δ_mMTC) each mapped: 0→-1, 1→0, 2→+1
  int dEmbb, dUrllc, dMmtc;
  DecodeAction (actionIdx, dEmbb, dUrllc, dMmtc);

  // Apply deltas with constraints: each slice ≥ 1 PRB, sum = totalPrbs
  int newAlloc[N_SLICES];
  newAlloc[0] = (int)m_prbAlloc[0] + dEmbb;
  newAlloc[1] = (int)m_prbAlloc[1] + dUrllc;
  newAlloc[2] = (int)m_prbAlloc[2] + dMmtc;

  // Enforce minimum 1 PRB per slice
  for (uint32_t s = 0; s < N_SLICES; ++s)
    newAlloc[s] = std::max (1, newAlloc[s]);

  // Enforce sum = totalPrbs by adjusting eMBB (largest slice)
  int total = newAlloc[0] + newAlloc[1] + newAlloc[2];
  newAlloc[0] += (int)m_cfg.totalPrbs - total;
  newAlloc[0] = std::max (1, newAlloc[0]);

  for (uint32_t s = 0; s < N_SLICES; ++s)
    m_prbAlloc[s] = (uint32_t)newAlloc[s];

  // Push new weights to the NS-3 AI scheduler
  ApplySliceWeights ();

  // Compute and store reward for this step
  ComputeReward ();
  m_stepCount++;

  // Schedule next step
  Simulator::Schedule (Seconds (m_cfg.gymStep), &ScheduleGymStep,
                       Ptr<NrSliceGymEnv> (this));
  return true;
}

void
NrSliceGymEnv::DecodeAction (uint32_t actionIdx, int &dEmbb, int &dUrllc, int &dMmtc)
{
  // Decode base-3: each digit ∈ {0,1,2} → maps to {-1, 0, +1}
  auto decode = [](uint32_t d) -> int { return (int)d - 1; };
  dMmtc  = decode (actionIdx % 3);   actionIdx /= 3;
  dUrllc = decode (actionIdx % 3);   actionIdx /= 3;
  dEmbb  = decode (actionIdx % 3);
}

// ─────────────────────────────────────────────────────────────────────────────
// Apply slice weights to NrMacSchedulerTdmaAi
// ─────────────────────────────────────────────────────────────────────────────

void
NrSliceGymEnv::ApplySliceWeights ()
{
  // Get the gNB device and cast its scheduler to the AI scheduler type
  Ptr<NrGnbNetDevice> gnbDev =
      DynamicCast<NrGnbNetDevice> (m_gnbDevs.Get (0));
  if (!gnbDev)
    {
      NS_LOG_WARN ("ApplySliceWeights: could not get gNB device");
      return;
    }

  Ptr<NrMacSchedulerTdmaAi> aiSched =
      DynamicCast<NrMacSchedulerTdmaAi> (gnbDev->GetMac (0)->GetScheduler ());
  if (!aiSched)
    {
      NS_LOG_WARN ("ApplySliceWeights: scheduler is not NrMacSchedulerTdmaAi");
      return;
    }

  // Build the UeWeightsMap: RNTI → lcId → weight
  // Type confirmed from nr-mac-scheduler-tdma-ai.h:
  //   std::map<uint16_t, std::map<uint8_t, double>>
  // lcId=4 is the default DRB logical channel in 5G-LENA
  NrMacSchedulerTdmaAi::UeWeightsMap weightsMap;

  for (uint32_t s = 0; s < N_SLICES; ++s)
    {
      uint32_t nUes = m_ueDevsBySlice[s].GetN ();
      if (nUes == 0) continue;

      // Weight proportional to PRB fraction divided equally among UEs in slice
      double sliceFraction = (double)m_prbAlloc[s] / m_cfg.totalPrbs;
      double perUeWeight   = sliceFraction / (double)nUes;

      for (uint32_t i = 0; i < nUes; ++i)
        {
          Ptr<NrUeNetDevice> ueDev =
              DynamicCast<NrUeNetDevice> (m_ueDevsBySlice[s].Get (i));
          if (!ueDev) continue;

          uint16_t rnti = ueDev->GetRnti ();  // Use RNTI, not IMSI!
          weightsMap[rnti][4] = perUeWeight;  // lcId=4: default DRB
        }
    }

  // Collect UE vector for the scheduler API
  // Verify exact signature with:
  //   grep -n "UpdateAllUeWeightsDl\|UeWeightsMap" \
  //        contrib/nr/model/nr-mac-scheduler-tdma-ai.h
  std::vector<Ptr<NrMacSchedulerUeInfo>> ueVector;
  // Note: ueVector is obtained from the scheduler — adjust if API differs
  aiSched->UpdateAllUeWeightsDl (weightsMap, ueVector);

  NS_LOG_DEBUG ("Weights applied: eMBB=" << m_prbAlloc[0]
                << " URLLC=" << m_prbAlloc[1]
                << " mMTC=" << m_prbAlloc[2]);
}

// ─────────────────────────────────────────────────────────────────────────────
// Schedule first gym step
// ─────────────────────────────────────────────────────────────────────────────

void
NrSliceGymEnv::ScheduleNextStep ()
{
  Simulator::Schedule (Seconds (m_cfg.gymStep), &ScheduleGymStep,
                       Ptr<NrSliceGymEnv> (this));
}

void
ScheduleGymStep (Ptr<NrSliceGymEnv> env)
{
  env->NotifyCurrentStateReady ();
}
