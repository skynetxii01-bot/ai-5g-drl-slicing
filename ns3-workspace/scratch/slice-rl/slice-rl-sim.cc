#include "slice-env.h"

#if __has_include("ns3/opengym-module.h")
#define HAVE_OPENGYM
#endif

#ifdef HAVE_OPENGYM

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/nr-module.h"
#include "ns3/opengym-module.h"
#include "ns3/point-to-point-module.h"

#include <array>
#include <iostream>

using namespace ns3;

namespace
{
// On/Off traffic model — replaces constant-rate UdpClient.
// During "on" periods the sender transmits at peakRate; during "off" periods
// it is silent. Exponential on/off times make each episode statistically unique
// (provided SetRun(seed) is called with a different seed per episode).
// Average throughput ≈ peakRate × onMeanSec / (onMeanSec + offMeanSec).
void
InstallOnOffTraffic(const Ptr<Node>& remoteHost,
                    const NodeContainer& sliceUes,
                    const Ipv4InterfaceContainer& ueIfaces,
                    uint16_t basePort,
                    uint32_t packetSize,
                    const DataRate& peakRate,
                    double onMeanSec,
                    double offMeanSec,
                    Time appStart,
                    Time appStop)
{
    const std::string onDist =
        "ns3::ExponentialRandomVariable[Mean=" + std::to_string(onMeanSec) + "]";
    const std::string offDist =
        "ns3::ExponentialRandomVariable[Mean=" + std::to_string(offMeanSec) + "]";

    for (uint32_t i = 0; i < sliceUes.GetN(); ++i)
    {
        const uint16_t port = basePort + i;

        // Receiver on UE side
        UdpServerHelper server(port);
        ApplicationContainer serverApp = server.Install(sliceUes.Get(i));
        serverApp.Start(appStart);
        serverApp.Stop(appStop);

        // On/Off sender on remote host
        OnOffHelper onoff("ns3::UdpSocketFactory",
                          InetSocketAddress(ueIfaces.GetAddress(i), port));
        onoff.SetAttribute("DataRate", DataRateValue(peakRate));
        onoff.SetAttribute("PacketSize", UintegerValue(packetSize));
        onoff.SetAttribute("OnTime", StringValue(onDist));
        onoff.SetAttribute("OffTime", StringValue(offDist));

        ApplicationContainer clientApp = onoff.Install(remoteHost);
        clientApp.Start(appStart + MilliSeconds(50));
        clientApp.Stop(appStop);
    }
}
} // namespace

int
main(int argc, char* argv[])
{
    uint32_t gymPort = 5555;
    uint32_t seed = 42;
    double simTimeSeconds = 100.0;

    CommandLine cmd(__FILE__);
    cmd.AddValue("gymPort", "OpenGym TCP port", gymPort);
    cmd.AddValue("seed", "Simulation RNG seed", seed);
    cmd.AddValue("simTime", "Simulation time [s]", simTimeSeconds);
    cmd.Parse(argc, argv);

    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(seed);

    constexpr uint32_t gnbCount = 1;
    constexpr uint32_t embbUes = 10;
    constexpr uint32_t urllcUes = 5;
    constexpr uint32_t mmtcUes = 20;
    constexpr uint8_t numerology = 1;
    constexpr double centralFrequency = 3.5e9;
    constexpr double bandwidth = 20e6;
    constexpr uint16_t totalPrbs = 25;

    const Time appStart = Seconds(0.2);
    const Time appStop = Seconds(simTimeSeconds - 0.05);

    std::cout << "=== 5G NR Slice RL Simulation ===\n"
              << "NS-3.45  |  5G-LENA NR v4.1.y  |  ns3-gym\n"
              << "eMBB=" << embbUes << "  URLLC=" << urllcUes << "  mMTC=" << mmtcUes
              << "  gymPort=" << gymPort << std::endl;

    NodeContainer gnbNodes;
    gnbNodes.Create(gnbCount);

    NodeContainer embbNodes;
    embbNodes.Create(embbUes);

    NodeContainer urllcNodes;
    urllcNodes.Create(urllcUes);

    NodeContainer mmtcNodes;
    mmtcNodes.Create(mmtcUes);

    NodeContainer allUeNodes;
    allUeNodes.Add(embbNodes);
    allUeNodes.Add(urllcNodes);
    allUeNodes.Add(mmtcNodes);

    MobilityHelper gnbMobility;
    gnbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    gnbMobility.Install(gnbNodes);
    gnbNodes.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(0.0, 0.0, 10.0));

    MobilityHelper ueMobility;
    ueMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    ueMobility.Install(allUeNodes);

    for (uint32_t i = 0; i < embbNodes.GetN(); ++i)
    {
        embbNodes.Get(i)->GetObject<MobilityModel>()->SetPosition(Vector(20.0 + i * 3.0, 0.0, 1.5));
    }
    for (uint32_t i = 0; i < urllcNodes.GetN(); ++i)
    {
        urllcNodes.Get(i)->GetObject<MobilityModel>()->SetPosition(Vector(30.0 + i * 2.5, 8.0, 1.5));
    }
    for (uint32_t i = 0; i < mmtcNodes.GetN(); ++i)
    {
        mmtcNodes.Get(i)->GetObject<MobilityModel>()->SetPosition(Vector(45.0 + i * 1.5, -10.0, 1.5));
    }

    Ptr<NrPointToPointEpcHelper> epcHelper = CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> beamformingHelper = CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();

    nrHelper->SetEpcHelper(epcHelper);
    nrHelper->SetBeamformingHelper(beamformingHelper);
    beamformingHelper->SetAttribute("BeamformingMethod",
                                    TypeIdValue(DirectPathBeamforming::GetTypeId()));

    nrHelper->SetSchedulerTypeId(NrMacSchedulerOfdmaAi::GetTypeId());

    Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface>(gymPort);
    Ptr<NrSliceGymEnv> gymEnv = CreateObject<NrSliceGymEnv>();
    gymEnv->SetOpenGymInterface(openGymInterface);

    nrHelper->SetSchedulerAttribute("NotifyCbDl",
                                    CallbackValue(
                                        MakeCallback(&NrSliceGymEnv::OnSchedulerNotify, gymEnv)));
    nrHelper->SetSchedulerAttribute("ActiveDlAi", BooleanValue(true));

    nrHelper->SetUeAntennaAttribute("NumRows", UintegerValue(1));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(1));
    nrHelper->SetGnbAntennaAttribute("NumRows", UintegerValue(4));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(4));

    CcBwpCreator ccBwpCreator;
    CcBwpCreator::SimpleOperationBandConf bandConf(centralFrequency, bandwidth, 1);
    bandConf.m_numBwp = 1;
    OperationBandInfo band = ccBwpCreator.CreateOperationBandContiguousCc(bandConf);

    Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();
    channelHelper->ConfigureFactories("UMi", "Default", "ThreeGpp");
    channelHelper->AssignChannelsToBands({band}, NrChannelHelper::INIT_PROPAGATION);

    BandwidthPartInfoPtrVector allBwps = CcBwpCreator::GetAllBwps({band});

    NetDeviceContainer gnbDevs = nrHelper->InstallGnbDevice(gnbNodes, allBwps);
    NetDeviceContainer embbDevs = nrHelper->InstallUeDevice(embbNodes, allBwps);
    NetDeviceContainer urllcDevs = nrHelper->InstallUeDevice(urllcNodes, allBwps);
    NetDeviceContainer mmtcDevs = nrHelper->InstallUeDevice(mmtcNodes, allBwps);

    NetDeviceContainer allUeDevs;
    allUeDevs.Add(embbDevs);
    allUeDevs.Add(urllcDevs);
    allUeDevs.Add(mmtcDevs);

    nrHelper->GetGnbPhy(gnbDevs.Get(0), 0)->SetAttribute("Numerology", UintegerValue(numerology));
    nrHelper->GetGnbPhy(gnbDevs.Get(0), 0)->SetAttribute("TxPower", DoubleValue(30.0));

    InternetStackHelper internet;
    internet.Install(allUeNodes);

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
    Ptr<Ipv4StaticRouting> remoteHostStaticRouting =
        ipv4RoutingHelper.GetStaticRouting(remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(Ipv4Address("7.0.0.0"),
                                                Ipv4Mask("255.0.0.0"),
                                                1);

    Ipv4InterfaceContainer embbIfaces = epcHelper->AssignUeIpv4Address(embbDevs);
    Ipv4InterfaceContainer urllcIfaces = epcHelper->AssignUeIpv4Address(urllcDevs);
    Ipv4InterfaceContainer mmtcIfaces = epcHelper->AssignUeIpv4Address(mmtcDevs);

    nrHelper->AttachToClosestGnb(allUeDevs, gnbDevs);

    for (uint32_t i = 0; i < allUeNodes.GetN(); ++i)
    {
        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(allUeNodes.Get(i)->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    Ipv4Address remoteHostAddr = internetIpIfaces.GetAddress(1);
    (void)remoteHostAddr;

    // eMBB: video-like bursts.  Peak=10Mbps, on~2s, off~1s → avg≈6.63 Mbps/UE
    //       Duty cycle 67 % — frequently above SLA threshold, creating real PRB pressure.
    InstallOnOffTraffic(remoteHost, embbNodes, embbIfaces,
                        1000, 1448, DataRate("10Mbps"),
                        2.0 /*onMean s*/, 1.0 /*offMean s*/,
                        appStart, appStop);

    // URLLC: event-driven bursts.  Peak=5Mbps, on~50ms, off~200ms → avg≈1 Mbps/UE
    //        Very short on-periods keep latency tight during bursts.
    InstallOnOffTraffic(remoteHost, urllcNodes, urllcIfaces,
                        2000, 20, DataRate("5Mbps"),
                        0.05 /*onMean s*/, 0.20 /*offMean s*/,
                        appStart, appStop);

    // mMTC: IoT sensor duty cycle.  Peak=2Mbps, on~1s, off~9s → avg≈0.2 Mbps/UE
    //        Long sleep periods mean mMTC demand is sparse and variable.
    InstallOnOffTraffic(remoteHost, mmtcNodes, mmtcIfaces,
                        3000, 128, DataRate("2Mbps"),
                        1.0 /*onMean s*/, 9.0 /*offMean s*/,
                        appStart, appStop);

    FlowMonitorHelper flowmonHelper;
    Ptr<FlowMonitor> monitor = flowmonHelper.InstallAll();
    Ptr<Ipv4FlowClassifier> classifier = DynamicCast<Ipv4FlowClassifier>(flowmonHelper.GetClassifier());

    NrSliceGymEnv::Config cfg;
    cfg.totalPrbs = totalPrbs;
    cfg.stepInterval = MilliSeconds(100);
    cfg.simTime = Seconds(simTimeSeconds);
    cfg.initialPrbAlloc = {10, 8, 7};
    cfg.maxUes = {embbUes, urllcUes, mmtcUes};
    cfg.maxThrMbps = {100.0, 10.0, 2.0};
    cfg.maxLatMs = {50.0, 15.0, 500.0};
    cfg.minThrMbps = {10.0, 1.0, 0.1};

    std::array<NetDeviceContainer, NrSliceGymEnv::kSliceCount> ueDevsBySlice{embbDevs, urllcDevs,
                                                                               mmtcDevs};
    gymEnv->SetFlowMonitor(monitor, classifier);
    gymEnv->Initialize(cfg, nrHelper, gnbDevs, ueDevsBySlice);
    

    Simulator::Stop(Seconds(simTimeSeconds + 0.1));
    Simulator::Run();
    monitor->SerializeToXmlFile("slice-rl-flowmon.xml", true, true);
    Simulator::Destroy();

    return 0;
}

#else

int
main(int argc, char* argv[])
{
    (void)argc;
    (void)argv;
    return 1;
}

#endif // HAVE_OPENGYM
