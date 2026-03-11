/**
 * slice-rl-sim.cc
 *
 * 5G NR Network Slicing DRL Simulation
 * ======================================
 * Main NS-3 simulation file for the project:
 * "AI-Powered Dynamic Resource Allocation in 5G NR Network Slicing
 *  Using Deep Reinforcement Learning"
 *
 * HOW TO RUN (two terminals required):
 *   Terminal 1 (NS-3 first!):
 *     ./ns3 run "scratch/slice-rl/slice-rl-sim --gymPort=5555 --seed=42"
 *   Terminal 2 (within 10 seconds):
 *     python3 training/train.py --agent dqn --port 5555 --seed 42
 *
 * TOPOLOGY:
 *   RemoteHost ─── PGW ─── gNB ─── 35 UEs (10 eMBB + 5 URLLC + 20 mMTC)
 */

#include "slice-env.h"

#include <ns3/core-module.h>
#include <ns3/network-module.h>
#include <ns3/internet-module.h>
#include <ns3/mobility-module.h>
#include <ns3/applications-module.h>
#include <ns3/point-to-point-module.h>
#include <ns3/nr-module.h>
#include <ns3/flow-monitor-module.h>
#include <ns3/opengym-module.h>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("SliceRlSim");

int
main (int argc, char *argv[])
{
  SliceConfig cfg;

  // ── Command line arguments ────────────────────────────────────────────────
  CommandLine cmd;
  cmd.AddValue ("simTime",   "Simulation time (s)",          cfg.simTime);
  cmd.AddValue ("gymPort",   "ZMQ port for Python agent",    cfg.gymPort);
  cmd.AddValue ("seed",      "Random seed",                  cfg.seed);
  cmd.AddValue ("embbUes",   "Number of eMBB UEs",           cfg.embbUes);
  cmd.AddValue ("urllcUes",  "Number of URLLC UEs",          cfg.urllcUes);
  cmd.AddValue ("mmtcUes",   "Number of mMTC UEs",           cfg.mmtcUes);
  cmd.Parse (argc, argv);

  // ── Random seed ───────────────────────────────────────────────────────────
  RngSeedManager::SetSeed (cfg.seed);
  RngSeedManager::SetRun (1);

  std::cout << "=== 5G NR Slice RL Simulation ===" << std::endl;
  std::cout << "SimTime=" << cfg.simTime << "s  Port=" << cfg.gymPort
            << "  UEs: eMBB=" << cfg.embbUes << " URLLC=" << cfg.urllcUes
            << " mMTC=" << cfg.mmtcUes << std::endl;

  // ── Create nodes ──────────────────────────────────────────────────────────
  uint32_t totalUes = cfg.embbUes + cfg.urllcUes + cfg.mmtcUes;

  NodeContainer gnbNode;   gnbNode.Create (1);
  NodeContainer ueNodes;   ueNodes.Create (totalUes);
  NodeContainer remoteHostNode; remoteHostNode.Create (1);

  // ── Mobility ──────────────────────────────────────────────────────────────
  MobilityHelper mobility;

  // gNB fixed at centre
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (gnbNode);
  gnbNode.Get (0)->GetObject<MobilityModel> ()->SetPosition (Vector (0, 0, 10));

  // UEs distributed randomly in 500m radius (Urban Micro scenario)
  mobility.SetPositionAllocator ("ns3::UniformDiscPositionAllocator",
                                 "X",   DoubleValue (0),
                                 "Y",   DoubleValue (0),
                                 "rho", DoubleValue (500));
  mobility.SetMobilityModel ("ns3::RandomWalk2dMobilityModel",
                             "Bounds", RectangleValue (Rectangle (-500, 500, -500, 500)),
                             "Speed",  StringValue ("ns3::UniformRandomVariable[Min=0.5|Max=2.0]"));
  mobility.Install (ueNodes);

  // Remote host is stationary (backhaul server)
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (remoteHostNode);

  // ── NR Helper setup ───────────────────────────────────────────────────────
  Ptr<NrHelper> nrHelper = CreateObject<NrHelper> ();
  Ptr<NrPointToPointEpcHelper> epcHelper = CreateObject<NrPointToPointEpcHelper> ();
  nrHelper->SetEpcHelper (epcHelper);

  // Use AI scheduler — this is what the DRL agent controls
  nrHelper->SetSchedulerTypeId (NrMacSchedulerTdmaAi::GetTypeId ());

  // ── Channel configuration (5G-LENA v4.1 API: NrChannelHelper) ────────────
  NrChannelHelper channelHelper;
  channelHelper.ConfigureFactories ("UMi");  // Urban Micro Street Canyon

  // Configure spectrum band
  BandwidthPartInfoPtrVector allBwps;
  CcBwpCreator ccBwpCreator;
  const uint8_t numCcPerBand = 1;

  CcBwpCreator::SimpleOperationBandConf bandConf (
      cfg.centralFreq,
      cfg.bandwidth,
      numCcPerBand,
      BandwidthPartInfo::UMi_StreetCanyon);

  OperationBandInfo band = ccBwpCreator.CreateOperationBandContiguousCc (bandConf);
  channelHelper.AssignChannelsToBands ({band});

  for (const auto &bwp : band.GetBwps ())
    allBwps.push_back (bwp);

  nrHelper->InitializeOperationBand (&band);

  // Set numerology μ=1 (30 kHz SCS, suited for URLLC)
  nrHelper->SetGnbPhyAttribute ("Numerology", UintegerValue (cfg.numerology));

  // Antenna configuration (MIMO)
  nrHelper->SetGnbAntennaAttribute ("NumRows",    UintegerValue (4));
  nrHelper->SetGnbAntennaAttribute ("NumColumns", UintegerValue (8));
  nrHelper->SetUeAntennaAttribute  ("NumRows",    UintegerValue (2));
  nrHelper->SetUeAntennaAttribute  ("NumColumns", UintegerValue (4));

  // ── Install NR devices ────────────────────────────────────────────────────
  NetDeviceContainer gnbDev = nrHelper->InstallGnbDevice (gnbNode, allBwps);
  NetDeviceContainer ueDev  = nrHelper->InstallUeDevice  (ueNodes, allBwps);

  // Update device configuration after install
  nrHelper->UpdateDeviceConfigs ({gnbDev, ueDev});

  // ── Internet stack ────────────────────────────────────────────────────────
  InternetStackHelper internet;
  internet.Install (remoteHostNode);
  internet.Install (ueNodes);

  // Connect remote host to EPC via point-to-point link
  auto [remoteHostAddr, ifaces] = epcHelper->SetupRemoteHost (remoteHostNode.Get (0));

  // Assign IP addresses to UEs
  Ipv4InterfaceContainer ueIpIface = epcHelper->AssignUeIpv4Address (ueDev);

  // Attach all UEs to the gNB
  for (uint32_t i = 0; i < ueNodes.GetN (); ++i)
    nrHelper->AttachToGnb (ueDev.Get (i), gnbDev.Get (0));

  // ── Organise UEs by slice ─────────────────────────────────────────────────
  // IMSIs: 1-10 = eMBB, 11-15 = URLLC, 16-35 = mMTC
  std::vector<NetDeviceContainer> ueDevsBySlice (N_SLICES);
  uint32_t embbEnd  = cfg.embbUes;
  uint32_t urllcEnd = cfg.embbUes + cfg.urllcUes;

  for (uint32_t i = 0; i < totalUes; ++i)
    {
      if      (i < embbEnd)  ueDevsBySlice[0].Add (ueDev.Get (i));  // eMBB
      else if (i < urllcEnd) ueDevsBySlice[1].Add (ueDev.Get (i));  // URLLC
      else                   ueDevsBySlice[2].Add (ueDev.Get (i));  // mMTC
    }

  // ── Install applications ──────────────────────────────────────────────────
  // Traffic parameters per slice:
  //   eMBB:  CBR UDP 2 Mbps/UE,   1448B, ports 1000+i
  //   URLLC: CBR UDP 100 kbps/UE, 20B,   ports 2000+i
  //   mMTC:  CBR UDP 10 kbps/UE,  128B,  ports 3000+i

  struct SliceTraffic { DataRate rate; uint32_t pktSize; uint16_t basePort; };
  SliceTraffic trafficCfg[N_SLICES] = {
    { DataRate ("2Mbps"),    1448, 1000 },  // eMBB
    { DataRate ("100Kbps"),  20,   2000 },  // URLLC
    { DataRate ("10Kbps"),   128,  3000 },  // mMTC
  };

  uint32_t ueOffset = 0;
  for (uint32_t s = 0; s < N_SLICES; ++s)
    {
      uint32_t nUes = ueDevsBySlice[s].GetN ();
      for (uint32_t i = 0; i < nUes; ++i)
        {
          uint16_t port = trafficCfg[s].basePort + i;
          Ipv4Address ueAddr = ueIpIface.GetAddress (ueOffset + i);

          // Downlink: server sends to UE
          PacketSinkHelper sink ("ns3::UdpSocketFactory",
                                 InetSocketAddress (ueAddr, port));
          ApplicationContainer sinkApp = sink.Install (ueNodes.Get (ueOffset + i));
          sinkApp.Start (Seconds (0.5));
          sinkApp.Stop  (Seconds (cfg.simTime));

          // Uplink: UE sends to remote host
          OnOffHelper onoff ("ns3::UdpSocketFactory",
                             InetSocketAddress (remoteHostAddr, port));
          onoff.SetConstantRate (trafficCfg[s].rate, trafficCfg[s].pktSize);
          ApplicationContainer clientApp = onoff.Install (ueNodes.Get (ueOffset + i));
          clientApp.Start (Seconds (1.0));
          clientApp.Stop  (Seconds (cfg.simTime));
        }
      ueOffset += nUes;
    }

  // ── Flow Monitor ──────────────────────────────────────────────────────────
  FlowMonitorHelper flowHelper;
  Ptr<FlowMonitor> flowMonitor = flowHelper.InstallAll ();
  Ptr<Ipv4FlowClassifier> classifier =
      DynamicCast<Ipv4FlowClassifier> (flowHelper.GetClassifier ());

  // ── OpenGym environment ───────────────────────────────────────────────────
  // This creates the ZMQ socket that Python connects to
  Ptr<OpenGymInterface> openGym = CreateObject<OpenGymInterface> (cfg.gymPort);

  Ptr<NrSliceGymEnv> gymEnv = CreateObject<NrSliceGymEnv> (
      cfg, gnbDev, ueDevsBySlice, flowMonitor, classifier);
  gymEnv->SetOpenGymInterface (openGym);

  // Schedule first gym step after warm-up period
  gymEnv->ScheduleNextStep ();

  // ── Run simulation ────────────────────────────────────────────────────────
  Simulator::Stop (Seconds (cfg.simTime));
  std::cout << "Simulation running... waiting for Python agent on port "
            << cfg.gymPort << std::endl;
  Simulator::Run ();

  // Save flow monitor results
  flowMonitor->SerializeToXmlFile ("results/flow-monitor.xml", true, true);

  Simulator::Destroy ();
  std::cout << "Simulation complete." << std::endl;
  return 0;
}
