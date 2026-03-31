#include "slice-env.h"

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("SliceRlSim");

int
main(int argc, char *argv[])
{
  uint32_t gymPort = 5555;
  uint32_t seed = 42;

  CommandLine cmd;
  cmd.AddValue("gymPort", "ns3-gym port", gymPort);
  cmd.AddValue("seed", "RNG seed", seed);
  cmd.Parse(argc, argv);

  RngSeedManager::SetSeed(seed);

  std::cout << "=== 5G NR Slice RL Simulation ===" << std::endl;
  std::cout << "NS-3.45  |  5G-LENA NR v4.1.y  |  ns3-gym" << std::endl;
  std::cout << "eMBB=10  URLLC=5  mMTC=20  gymPort=" << gymPort << std::endl;

  NodeContainer gnbNodes;
  gnbNodes.Create(1);
  NodeContainer embbUes;
  embbUes.Create(10);
  NodeContainer urllcUes;
  urllcUes.Create(5);
  NodeContainer mmtcUes;
  mmtcUes.Create(20);

  NodeContainer allUes;
  allUes.Add(embbUes);
  allUes.Add(urllcUes);
  allUes.Add(mmtcUes);

  MobilityHelper mobility;
  mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
  mobility.Install(gnbNodes);
  mobility.Install(allUes);

  Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();
  Ptr<IdealBeamformingHelper> beamformingHelper = CreateObject<IdealBeamformingHelper>();
  nrHelper->SetBeamformingHelper(beamformingHelper);

  Ptr<NrPointToPointEpcHelper> epcHelper = CreateObject<NrPointToPointEpcHelper>();
  nrHelper->SetEpcHelper(epcHelper);

  CcBwpCreator ccBwpCreator;
  double centralFrequency = 3.5e9;
  double bandwidth = 20e6;
  uint8_t numCcPerBand = 1;
  CcBwpCreator::SimpleOperationBandConf bandConf(centralFrequency, bandwidth, numCcPerBand);
  OperationBandInfo band = ccBwpCreator.CreateOperationBandContiguousCc(bandConf);

  Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();
  channelHelper->ConfigureFactories("UMi", "Default", "ThreeGpp");
  channelHelper->AssignChannelsToBands({band}, NrChannelHelper::INIT_PROPAGATION);

  BandwidthPartInfoPtrVector allBwps = CcBwpCreator::GetAllBwps({band});

  Ptr<NrSliceGymEnv> gymEnv = CreateObject<NrSliceGymEnv>();
  OpenGymInterface::Get()->SetOpenGymPort(gymPort);
  OpenGymInterface::Get()->SetOpenGymEnv(gymEnv);

  nrHelper->SetSchedulerTypeId(NrMacSchedulerTdmaAi::GetTypeId());
  nrHelper->SetSchedulerAttribute("NotifyCbDl", CallbackValue(MakeCallback(&NrSliceGymEnv::OnSchedulerNotify, gymEnv)));
  nrHelper->SetSchedulerAttribute("ActiveDlAi", BooleanValue(true));

  NetDeviceContainer gnbDevs = nrHelper->InstallGnbDevice(gnbNodes, allBwps);
  NetDeviceContainer embbDevs = nrHelper->InstallUeDevice(embbUes, allBwps);
  NetDeviceContainer urllcDevs = nrHelper->InstallUeDevice(urllcUes, allBwps);
  NetDeviceContainer mmtcDevs = nrHelper->InstallUeDevice(mmtcUes, allBwps);

  NetDeviceContainer allUeDevs;
  allUeDevs.Add(embbDevs);
  allUeDevs.Add(urllcDevs);
  allUeDevs.Add(mmtcDevs);

  InternetStackHelper internet;
  internet.Install(allUes);

  nrHelper->AttachToClosestGnb(allUeDevs, gnbDevs);

  Ptr<Node> pgw = epcHelper->GetPgwNode();
  NodeContainer remoteHostContainer;
  remoteHostContainer.Create(1);
  Ptr<Node> remoteHost = remoteHostContainer.Get(0);
  internet.Install(remoteHostContainer);

  PointToPointHelper p2ph;
  p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
  p2ph.SetDeviceAttribute("Mtu", UintegerValue(2500));
  p2ph.SetChannelAttribute("Delay", TimeValue(Seconds(0.0)));
  NetDeviceContainer internetDevices = p2ph.Install(pgw, remoteHost);

  Ipv4AddressHelper ipv4h;
  ipv4h.SetBase("1.0.0.0", "255.0.0.0");
  Ipv4InterfaceContainer internetIpIfaces = ipv4h.Assign(internetDevices);

  Ipv4StaticRoutingHelper ipv4RoutingHelper;
  Ptr<Ipv4StaticRouting> remoteHostStaticRouting = ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
  remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

  Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress(1);

  Ipv4InterfaceContainer ueIpIfaces = epcHelper->AssignUeIpv4Address(NetDeviceContainer(allUeDevs));

  ApplicationContainer serverApps;
  ApplicationContainer clientApps;

  uint16_t port = 1000;
  uint32_t ueIdx = 0;
  auto installSlice = [&](const NetDeviceContainer &sliceDevs, double rateBps, uint32_t pktSize) {
    for (uint32_t i = 0; i < sliceDevs.GetN(); ++i)
    {
      UdpServerHelper server(port);
      serverApps.Add(server.Install(allUes.Get(ueIdx)));

      UdpClientHelper client(ueIpIfaces.GetAddress(ueIdx), port);
      client.SetAttribute("MaxPackets", UintegerValue(0));
      client.SetAttribute("Interval", TimeValue(Seconds(static_cast<double>(pktSize * 8) / rateBps)));
      client.SetAttribute("PacketSize", UintegerValue(pktSize));
      clientApps.Add(client.Install(remoteHost));

      ++port;
      ++ueIdx;
    }
  };

  installSlice(embbDevs, 2e6, 1448);
  installSlice(urllcDevs, 1e5, 20);
  installSlice(mmtcDevs, 1e4, 128);

  serverApps.Start(Seconds(0.1));
  clientApps.Start(Seconds(0.2));
  serverApps.Stop(Seconds(100.0));
  clientApps.Stop(Seconds(100.0));

  FlowMonitorHelper flowmonHelper;
  Ptr<FlowMonitor> flowMonitor = flowmonHelper.InstallAll();

  NrSliceGymEnv::SimConfig cfg;
  cfg.totalPrbs = 25;
  cfg.simTimeS = 100.0;
  cfg.stepS = 0.1;
  cfg.maxSteps = 1000;
  cfg.initPrb = {10, 8, 7};

  std::array<NetDeviceContainer, 3> ueBySlice{embbDevs, urllcDevs, mmtcDevs};
  gymEnv->Initialize(cfg, nrHelper, gnbDevs, ueBySlice);
  gymEnv->BuildImsiSliceMap();
  gymEnv->AttachFlowMonitor(flowMonitor);

  Simulator::Stop(Seconds(100.0));
  Simulator::Run();
  Simulator::Destroy();

  return 0;
}
