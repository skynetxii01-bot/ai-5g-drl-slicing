#include "slice-env.h"

#if __has_include("ns3/opengym-module.h")
// Header-level detection only — the macro HAVE_OPENGYM is defined inside slice-env.h
// via the same __has_include check. This empty block is intentional and harmless.
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
#include <fstream>
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
    // #region agent log
    {
        const std::string line =
            std::string("{\"sessionId\":\"73bf42\",\"runId\":\"baseline\",\"hypothesisId\":\"H6\",") +
            "\"location\":\"slice-rl-sim.cc:main:start\"," +
            "\"message\":\"C++ main entered\"," +
            "\"data\":{\"gymPort\":" + std::to_string(gymPort) + ",\"simTime\":" +
            std::to_string(simTimeSeconds) + ",\"seed\":" + std::to_string(seed) +
            "},\"timestamp\":0}\n";
        std::ofstream out1("/home/skynetxii/5g-project/ns-allinone-3.45/debug-73bf42.log",
                           std::ios::app);
        if (out1.is_open())
        {
            out1 << line;
        }
        std::ofstream out2("/tmp/debug-73bf42.log", std::ios::app);
        if (out2.is_open())
        {
            out2 << line;
        }
    }
    // #endregion

    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(seed);

    constexpr uint32_t gnbCount  = 1;
    constexpr uint32_t embbUes  = 20;   // video streaming users
    constexpr uint32_t urllcUes = 15;   // factory automation devices
    constexpr uint32_t mmtcUes  = 40;   // IoT sensors
    constexpr uint8_t  numerology = 1;
    constexpr double   centralFrequency = 3.5e9;
    constexpr double   bandwidth        = 20e6;
    constexpr uint16_t totalPrbs = 51;   // 3GPP TS 38.101-1 Table 5.3.2-1: 20MHz + μ=1

    const Time appStart = Seconds(0.2);
    const Time appStop  = Seconds(simTimeSeconds - 0.05);

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
        embbNodes.Get(i)->GetObject<MobilityModel>()->SetPosition(
            Vector(20.0 + i * 3.0, 0.0, 1.5));
    }
    for (uint32_t i = 0; i < urllcNodes.GetN(); ++i)
    {
        urllcNodes.Get(i)->GetObject<MobilityModel>()->SetPosition(
            Vector(30.0 + i * 2.5, 8.0, 1.5));
    }
    for (uint32_t i = 0; i < mmtcNodes.GetN(); ++i)
    {
        mmtcNodes.Get(i)->GetObject<MobilityModel>()->SetPosition(
            Vector(45.0 + i * 1.5, -10.0, 1.5));
    }

    Ptr<NrPointToPointEpcHelper> epcHelper        = CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper>  beamformingHelper = CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper>                nrHelper          = CreateObject<NrHelper>();

    nrHelper->SetEpcHelper(epcHelper);
    nrHelper->SetBeamformingHelper(beamformingHelper);
    beamformingHelper->SetAttribute("BeamformingMethod",
                                    TypeIdValue(DirectPathBeamforming::GetTypeId()));

    nrHelper->SetSchedulerTypeId(NrMacSchedulerOfdmaAi::GetTypeId());

    Ptr<OpenGymInterface> openGymInterface = CreateObject<OpenGymInterface>(gymPort);
    Ptr<NrSliceGymEnv>    gymEnv           = CreateObject<NrSliceGymEnv>();
    gymEnv->SetOpenGymInterface(openGymInterface);

    nrHelper->SetSchedulerAttribute("NotifyCbDl",
                                    CallbackValue(
                                        MakeCallback(&NrSliceGymEnv::OnSchedulerNotify, gymEnv)));
    nrHelper->SetSchedulerAttribute("ActiveDlAi", BooleanValue(true));

    nrHelper->SetUeAntennaAttribute("NumRows",    UintegerValue(1));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(1));
    nrHelper->SetGnbAntennaAttribute("NumRows",    UintegerValue(4));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(4));

    CcBwpCreator ccBwpCreator;
    CcBwpCreator::SimpleOperationBandConf bandConf(centralFrequency, bandwidth, 1);
    bandConf.m_numBwp = 1;
    OperationBandInfo band = ccBwpCreator.CreateOperationBandContiguousCc(bandConf);

    Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();
    channelHelper->ConfigureFactories("UMi", "Default", "ThreeGpp");
    channelHelper->AssignChannelsToBands({band}, NrChannelHelper::INIT_PROPAGATION);

    BandwidthPartInfoPtrVector allBwps = CcBwpCreator::GetAllBwps({band});

    NetDeviceContainer gnbDevs  = nrHelper->InstallGnbDevice(gnbNodes,  allBwps);
    NetDeviceContainer embbDevs  = nrHelper->InstallUeDevice(embbNodes,  allBwps);
    NetDeviceContainer urllcDevs = nrHelper->InstallUeDevice(urllcNodes, allBwps);
    NetDeviceContainer mmtcDevs  = nrHelper->InstallUeDevice(mmtcNodes,  allBwps);

    NetDeviceContainer allUeDevs;
    allUeDevs.Add(embbDevs);
    allUeDevs.Add(urllcDevs);
    allUeDevs.Add(mmtcDevs);

    nrHelper->GetGnbPhy(gnbDevs.Get(0), 0)->SetAttribute("Numerology", UintegerValue(numerology));
    nrHelper->GetGnbPhy(gnbDevs.Get(0), 0)->SetAttribute("TxPower",    DoubleValue(30.0));

    InternetStackHelper internet;
    internet.Install(allUeNodes);

    Ptr<Node> pgw = epcHelper->GetPgwNode();
    NodeContainer remoteHostContainer;
    remoteHostContainer.Create(1);
    Ptr<Node> remoteHost = remoteHostContainer.Get(0);
    internet.Install(remoteHostContainer);

    PointToPointHelper p2ph;
    p2ph.SetDeviceAttribute("DataRate", DataRateValue(DataRate("100Gb/s")));
    p2ph.SetDeviceAttribute("Mtu",      UintegerValue(2500));
    p2ph.SetChannelAttribute("Delay",   TimeValue(Seconds(0.0)));
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

    Ipv4InterfaceContainer embbIfaces  = epcHelper->AssignUeIpv4Address(embbDevs);
    Ipv4InterfaceContainer urllcIfaces = epcHelper->AssignUeIpv4Address(urllcDevs);
    Ipv4InterfaceContainer mmtcIfaces  = epcHelper->AssignUeIpv4Address(mmtcDevs);

    nrHelper->AttachToClosestGnb(allUeDevs, gnbDevs);

    for (uint32_t i = 0; i < allUeNodes.GetN(); ++i)
    {
        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(allUeNodes.Get(i)->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    // eMBB: 5QI=8, video streaming. Peak=12Mbps, on~2s, off~0.3s → duty=87%
    // Expected aggregate: 20 UEs × 12 Mbps × 0.87 = 208 Mbps → ~38 PRBs
    InstallOnOffTraffic(remoteHost, embbNodes, embbIfaces,
                        1000, 1448, DataRate("12Mbps"),
                         2.0, 0.3,
                         appStart, appStop);

    // URLLC: 5QI=83, factory automation. Peak=5Mbps, on~20ms, off~80ms → duty=20%
    // packetSize 20→100 bytes: 20B at 5Mbps = 31250 pkt/s (pathological), 100B = 6250 pkt/s
    // Expected aggregate: 15 UEs × 5 Mbps × 0.20 = 15 Mbps → ~5 PRBs avg, ~25 PRBs at burst
    InstallOnOffTraffic(remoteHost, urllcNodes, urllcIfaces,
                         2000, 100, DataRate("5Mbps"),
                         0.02, 0.08,
                        appStart, appStop);

    // mMTC: NB-IoT, smart metering. Peak=1Mbps, on~0.5s, off~19.5s → duty=2.5%
    // Expected aggregate: 40 UEs × 1 Mbps × 0.025 = 1 Mbps → background noise
    InstallOnOffTraffic(remoteHost, mmtcNodes, mmtcIfaces,
                        3000, 64, DataRate("1Mbps"),
                        0.5, 19.5,
                        appStart, appStop);

    FlowMonitorHelper flowmonHelper;
    Ptr<FlowMonitor>         monitor    = flowmonHelper.InstallAll();
    Ptr<Ipv4FlowClassifier>  classifier =
        DynamicCast<Ipv4FlowClassifier>(flowmonHelper.GetClassifier());

    NrSliceGymEnv::Config cfg;
    cfg.totalPrbs       = totalPrbs;           // = 51
    cfg.stepInterval = MilliSeconds(100);
    cfg.simTime      = Seconds(simTimeSeconds);
    cfg.initialPrbAlloc = {25, 15, 11}; // sums to 51, proportional starting point

    // P0-1 FIX: maxUes must be UPPER BOUNDS that EXCEED the simulated counts.
    // See slice-env.h Config for full rationale.
    cfg.maxUes          = {30, 25, 60};   /// upper bounds > actual counts

    // -----------------------------------------------------------------------
    // P0-A FIX: mMTC maxThrMbps raised from 2.0 → 8.0 Mbps.
    //
    // Root cause: with 20 mMTC UEs at 2 Mbps peak and 10% duty cycle, the
    // expected simultaneous active UEs ≈ 2, yielding expected aggregate
    // throughput ≈ 4 Mbps.  The old cap of 2.0 Mbps (= one UE peak) caused
    // Clamp01(thr / maxThr) to saturate at 1.0 for ~86% of active-period
    // steps, making obs[5] a near-binary on/off signal rather than a
    // continuous gradient.
    //
    // New value: 8.0 Mbps = 2 × expected aggregate.
    //   - Off-period (thr ≈ 0):         obs[5] ≈ 0.0
    //   - Single active UE (thr ≈ 2):   obs[5] ≈ 0.25
    //   - Expected load  (thr ≈ 4):     obs[5] ≈ 0.50
    //   - Double load    (thr ≈ 8):     obs[5] = 1.0  (saturation at 2× expected)
    //
    // This matches the treatment of eMBB (maxThr=100 >> expected≈66) and
    // URLLC (maxThr=10 >> expected≈5), where maxThr is set well above the
    // expected aggregate to avoid saturation.
    // -----------------------------------------------------------------------
    cfg.maxThrMbps      = {240.0, 75.0, 8.0}; // 20×12, 15×5, keep mMTC
    cfg.maxLatMs        = {50.0,  10.0, 300.0};// URLLC: 3GPP 5QI=83 exact
    cfg.minThrMbps      = {50.0,  2.0,  0.1}; // eMBB: 50, URLLC: 2, mMTC: keep

    std::array<NetDeviceContainer, NrSliceGymEnv::kSliceCount> ueDevsBySlice{
        embbDevs, urllcDevs, mmtcDevs};
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
    // ns3-gym (opengym-module) was not found at compile time.
    // Rebuild NS-3 with the opengym contrib module enabled.
    return 1;
}

#endif // HAVE_OPENGYM
