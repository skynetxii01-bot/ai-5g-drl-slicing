#ifndef SLICE_ENV_H
#define SLICE_ENV_H

#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/nr-module.h>
#include <ns3/opengym-module.h>
#include <ns3/flow-monitor-module.h>
#include <map>
#include <vector>
#include <cmath>
#include <fstream>

using namespace ns3;

// Number of slices (eMBB=0, URLLC=1, mMTC=2)
static const uint32_t N_SLICES = 3;

/**
 * \brief Configuration struct for simulation parameters
 *
 * Holds all parameters needed to configure the 5G NR simulation.
 * Values come from configs/config.yaml via command line arguments.
 */
struct SliceConfig
{
  uint32_t totalPrbs   = 25;       // Total Physical Resource Blocks
  double   simTime     = 100.0;    // Simulation time in seconds
  double   gymStep     = 0.1;      // Gym step interval in seconds (100ms)
  uint32_t gymPort     = 5555;     // ZMQ port for Python connection
  uint32_t seed        = 42;       // Random seed

  // UE counts per slice
  uint32_t embbUes     = 10;
  uint32_t urllcUes    = 5;
  uint32_t mmtcUes     = 20;

  // Radio parameters
  double   centralFreq = 3.5e9;   // 3.5 GHz carrier frequency
  double   bandwidth   = 20e6;    // 20 MHz bandwidth
  uint32_t numerology  = 1;       // μ=1 → 30 kHz subcarrier spacing

  // SLA targets per slice (3GPP TS 22.261)
  // eMBB: 10 Mbps min throughput, 50ms max latency
  // URLLC: 1 Mbps min throughput, 1ms max latency (hardest!)
  // mMTC: 0.1 Mbps min throughput, 500ms max latency
  double maxThrMbps[N_SLICES]  = {100.0, 10.0, 1.0};   // normalisation denominators
  double maxLatMs[N_SLICES]    = {50.0,  1.0,  500.0};  // normalisation denominators
  double minThrMbps[N_SLICES]  = {10.0,  1.0,  0.1};    // SLA minimum throughput
  double maxSlaLatMs[N_SLICES] = {50.0,  1.0,  500.0};  // SLA maximum latency
  uint32_t maxUes[N_SLICES]    = {10,    5,    20};      // max UEs per slice
};

/**
 * \class NrSliceGymEnv
 * \brief OpenGym environment for 5G NR network slicing DRL
 *
 * This class connects NS-3 simulation to a Python DRL agent via the
 * OpenGym (ns3-gym) interface. The agent learns to allocate PRBs across
 * three network slices (eMBB, URLLC, mMTC) to maximise a reward signal
 * that balances throughput, latency, fairness, and SLA compliance.
 *
 * OBSERVATION SPACE (15-dim float, all in [0,1]):
 *   obs[0:3]  = PRB fraction per slice         (allocated PRBs / total PRBs)
 *   obs[3:6]  = Normalised throughput           (Mbps / max_thr_mbps)
 *   obs[6:9]  = Normalised latency              (ms / max_lat_ms)
 *   obs[9:12] = Queue occupancy                 (0=empty, 1=full)
 *   obs[12:15]= UE count fraction               (active UEs / max UEs)
 *
 * ACTION SPACE (27 discrete):
 *   action = (Δ_eMBB, Δ_URLLC, Δ_mMTC), each Δ ∈ {-1, 0, +1}
 *   Constraints: each slice ≥ 1 PRB, sum = 25 PRBs always
 *
 * REWARD:
 *   R = 0.5·mean(thr_norm) + 0.3·mean(sat·lat_norm) + 0.2·Jain - 2.0·SLA_violations
 */
class NrSliceGymEnv : public OpenGymEnv
{
public:
  NrSliceGymEnv (SliceConfig cfg,
                 NetDeviceContainer gnbDevs,
                 std::vector<NetDeviceContainer> ueDevsBySlice,
                 Ptr<FlowMonitor> flowMonitor,
                 Ptr<Ipv4FlowClassifier> classifier);

  virtual ~NrSliceGymEnv ();

  // Required OpenGym interface methods
  Ptr<OpenGymSpace>   GetObservationSpace () override;
  Ptr<OpenGymSpace>   GetActionSpace () override;
  Ptr<OpenGymDataContainer> GetObservation () override;
  float               GetReward () override;
  bool                GetGameOver () override;
  std::string         GetExtraInfo () override;
  bool                ExecuteActions (Ptr<OpenGymDataContainer> action) override;

  // Schedule the first gym step
  void ScheduleNextStep ();

private:
  // Internal helper methods
  void   CollectMetrics ();           // Read FlowMonitor stats each step
  void   ApplySliceWeights ();        // Push PRB weights to NR scheduler
  float  ComputeReward ();            // Compute reward signal
  float  JainFairnessIndex ();        // Jain's fairness index of throughputs
  void   DecodeAction (uint32_t actionIdx,
                       int &dEmbb, int &dUrllc, int &dMmtc);

  SliceConfig                  m_cfg;
  NetDeviceContainer           m_gnbDevs;
  std::vector<NetDeviceContainer> m_ueDevsBySlice;
  Ptr<FlowMonitor>             m_flowMonitor;
  Ptr<Ipv4FlowClassifier>      m_classifier;

  // Current PRB allocation per slice [eMBB, URLLC, mMTC]
  uint32_t m_prbAlloc[N_SLICES] = {9, 8, 8};

  // Metrics collected each step
  double m_throughputMbps[N_SLICES]  = {0};
  double m_latencyMs[N_SLICES]       = {0};
  double m_queueOccupancy[N_SLICES]  = {0};
  uint32_t m_activeUes[N_SLICES]     = {0};

  // Byte counters for throughput calculation
  uint64_t m_lastRxBytes[N_SLICES]   = {0};
  Time     m_lastStepTime;

  uint32_t m_stepCount = 0;
  float    m_lastReward = 0.0f;
};

// Helper: schedule periodic gym steps
void ScheduleGymStep (Ptr<NrSliceGymEnv> env);

#endif /* SLICE_ENV_H */
