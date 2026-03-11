#include "slice-env.h"

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/network-module.h"
#include "ns3/nr-channel-helper.h"
#include "ns3/nr-helper.h"
#include "ns3/nr-module.h"
#include "ns3/opengym-module.h"
#include "ns3/point-to-point-helper.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("SliceRlSim");

static std::array<NrSliceGymEnv::SliceMetrics, NrSliceGymEnv::NUM_SLICES> g_metrics;

static void
StepEnv(Ptr<NrSliceGymEnv> env)
{
  // In a production study, update g_metrics from FlowMonitor/trace callbacks.
  // We keep bounded synthetic values so the example remains self-contained.
  for (uint8_t s = 0; s < NrSliceGymEnv::NUM_SLICES; ++s)
  {
    g_metrics[s].throughputMbps = std::max(0.01, g_metrics[s].throughputMbps + (0.02 * (s + 1)));
    g_metrics[s].latencyMs = std::max(0.1, g_metrics[s].latencyMs);
    g_metrics[s].queueOccupancy = std::min(1.0, g_metrics[s].queueOccupancy);
  }
  env->UpdateSliceMetrics(g_metrics);
  env->NotifyStep();
  Simulator::Schedule(MilliSeconds(100), &StepEnv, env);
}

int
main(int argc, char* argv[])
{
  uint32_t gymPort = 5555;
  uint32_t seed = 42;
  CommandLine cmd;
  cmd.AddValue("gymPort", "Port where NS-3 OpenGym server binds", gymPort);
  cmd.AddValue("seed", "Random seed", seed);
  cmd.Parse(argc, argv);

  RngSeedManager::SetSeed(seed);
  RngSeedManager::SetRun(1);

  const uint32_t embbUes = 10;
  const uint32_t urllcUes = 5;
  const uint32_t mmtcUes = 20;
  const uint32_t totalUes = embbUes + urllcUes + mmtcUes;
  const double simTime = 100.0;

  Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();
  Ptr<NrPointToPointEpcHelper> epcHelper = CreateObject<NrPointToPointEpcHelper>();
  nrHelper->SetEpcHelper(epcHelper);

  Ptr<IdealBeamformingHelper> idealBeamformingHelper = CreateObject<IdealBeamformingHelper>();
  nrHelper->SetBeamformingHelper(idealBeamformingHelper);

  auto channelHelper = CreateObject<NrChannelHelper>();
  channelHelper->SetChannelConditionModelAttribute("UpdatePeriod", TimeValue(MilliSeconds(0)));
  channelHelper->SetPathlossAttribute("Scenario", StringValue("UMi-StreetCanyon"));
  nrHelper->SetChannelHelper(channelHelper);

  nrHelper->SetSchedulerTypeId(TypeId::LookupByName("ns3::NrMacSchedulerTdmaAi"));

  // v4.1-style channel helper flow (using current NR helper APIs).
  // The exact helper internals are resolved by NR helper at runtime.
  BandwidthPartInfoPtrVector allBwps = nrHelper->CreateBandwidthParts(3.5e9, 20e6, 1);

  NodeContainer gnbNodes;
  gnbNodes.Create(1);
  NodeContainer ueNodes;
  ueNodes.Create(totalUes);

  MobilityHelper mobility;
  mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  mobility.Install(gnbNodes);
  mobility.Install(ueNodes);

  NetDeviceContainer gnbNetDevices = nrHelper->InstallGnbDevice(gnbNodes, allBwps);
  NetDeviceContainer ueNetDevices = nrHelper->InstallUeDevice(ueNodes, allBwps);

  InternetStackHelper internet;
  internet.Install(ueNodes);

  Ptr<Node> remoteHost = epcHelper->SetupRemoteHost();

  Ipv4InterfaceContainer ueIpIface;
  ueIpIface = epcHelper->AssignUeIpv4Address(NetDeviceContainer(ueNetDevices));

  for (uint32_t i = 0; i < totalUes; ++i)
  {
    nrHelper->AttachToClosestEnb(ueNetDevices.Get(i), gnbNetDevices);
  }

  uint16_t embbBasePort = 1000;
  uint16_t urllcBasePort = 2000;
  uint16_t mmtcBasePort = 3000;

  ApplicationContainer clientApps;
  ApplicationContainer serverApps;

  std::array<std::vector<uint16_t>, NrSliceGymEnv::NUM_SLICES> ueRntisBySlice;

  for (uint32_t i = 0; i < totalUes; ++i)
  {
    uint16_t port = 0;
    DataRate rate("10kb/s");
    uint32_t pktSize = 128;
    uint8_t slice = NrSliceGymEnv::MMTC;

    if (i < embbUes)
    {
      port = embbBasePort + i;
      rate = DataRate("2Mb/s");
      pktSize = 1448;
      slice = NrSliceGymEnv::EMBB;
    }
    else if (i < embbUes + urllcUes)
    {
      port = urllcBasePort + (i - embbUes);
      rate = DataRate("100kb/s");
      pktSize = 20;
      slice = NrSliceGymEnv::URLLC;
    }
    else
    {
      port = mmtcBasePort + (i - embbUes - urllcUes);
      rate = DataRate("10kb/s");
      pktSize = 128;
      slice = NrSliceGymEnv::MMTC;
    }

    UdpServerHelper server(port);
    serverApps.Add(server.Install(ueNodes.Get(i)));

    UdpClientHelper client(ueIpIface.GetAddress(i), port);
    client.SetAttribute("Interval", TimeValue(Seconds(static_cast<double>(pktSize * 8) / rate.GetBitRate())));
    client.SetAttribute("PacketSize", UintegerValue(pktSize));
    client.SetAttribute("MaxPackets", UintegerValue(1000000000));
    clientApps.Add(client.Install(remoteHost));

    Ptr<NrUeNetDevice> ueDev = DynamicCast<NrUeNetDevice>(ueNetDevices.Get(i));
    if (ueDev)
    {
      ueRntisBySlice[slice].push_back(ueDev->GetRrc()->GetRnti());
    }
  }

  serverApps.Start(Seconds(0.1));
  clientApps.Start(Seconds(0.2));
  serverApps.Stop(Seconds(simTime));
  clientApps.Stop(Seconds(simTime));

  Ptr<NrGnbNetDevice> gnb = DynamicCast<NrGnbNetDevice>(gnbNetDevices.Get(0));
  Ptr<NrMacSchedulerTdmaAi> scheduler = DynamicCast<NrMacSchedulerTdmaAi>(gnb->GetMac(0)->GetScheduler());

  Ptr<NrSliceGymEnv> env = CreateObject<NrSliceGymEnv>();
  env->SetScheduler(scheduler);
  env->SetUeRntisBySlice(ueRntisBySlice);
  env->SetTotalPrbs(25);
  env->SetStepTime(MilliSeconds(100));
  env->SetEpisodeSteps(1000);

  OpenGymInterface openGymInterface(gymPort);
  openGymInterface.SetOpenGymEnv(env);

  g_metrics[0] = {12.0, 20.0, 0.2, 1.0, 0};
  g_metrics[1] = {1.2, 2.0, 0.1, 0.7, 1};
  g_metrics[2] = {0.2, 200.0, 0.3, 1.0, 0};

  Simulator::Schedule(MilliSeconds(100), &StepEnv, env);
  Simulator::Stop(Seconds(simTime));
  Simulator::Run();
  Simulator::Destroy();

  return 0;
}
