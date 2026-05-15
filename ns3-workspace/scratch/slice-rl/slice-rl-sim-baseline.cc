/**
 * slice-rl-sim-baseline.cc
 *
 * Standalone baseline evaluator for the 3-slice 5G NR PRB-control scenario.
 * Runs Random, Round-Robin, and Greedy-PF policies sequentially on one
 * continuous NS-3 simulation — no OpenGym, no ZMQ, no Python ports required.
 *
 * NS-3 setup is identical to slice-rl-sim.cc (same UE counts, traffic
 * parameters, channel model) so results are directly comparable to DRL agents
 * evaluated via evaluate.py.
 *
 * Build:
 *   cd ~/5g-project/ns-allinone-3.45/ns-3.45
 *   ./ns3 build scratch_slice-rl_slice-rl-sim-baseline
 *
 * Run:
 *   ./ns3 run "scratch/slice-rl/slice-rl-sim-baseline \
 *       --episodes=10 --maxSteps=1000 --seed=99 \
 *       --out=results/baseline_cpp.json"
 *
 * simTime is computed automatically:
 *   3 policies × episodes × maxSteps × 100ms + 10s buffer
 */

#include "slice-baseline-env.h"

#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-helper.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"

#include <array>
#include <cmath>
#include <iostream>

using namespace ns3;

// ---------------------------------------------------------------------------
// Traffic helper — identical to slice-rl-sim.cc
// ---------------------------------------------------------------------------
namespace
{
void
InstallOnOffTraffic(const Ptr<Node>&              remoteHost,
                    const NodeContainer&           sliceUes,
                    const Ipv4InterfaceContainer&  ueIfaces,
                    uint16_t                       basePort,
                    uint32_t                       packetSize,
                    const DataRate&                peakRate,
                    double                         onMeanSec,
                    double                         offMeanSec,
                    Time                           appStart,
                    Time                           appStop)
{
    const std::string onDist =
        "ns3::ExponentialRandomVariable[Mean=" + std::to_string(onMeanSec) + "]";
    const std::string offDist =
        "ns3::ExponentialRandomVariable[Mean=" + std::to_string(offMeanSec) + "]";

    for (uint32_t i = 0; i < sliceUes.GetN(); ++i)
    {
        const uint16_t port = basePort + i;

        UdpServerHelper server(port);
        ApplicationContainer serverApp = server.Install(sliceUes.Get(i));
        serverApp.Start(appStart);
        serverApp.Stop(appStop);

        OnOffHelper onoff("ns3::UdpSocketFactory",
                          InetSocketAddress(ueIfaces.GetAddress(i), port));
        onoff.SetAttribute("DataRate",    DataRateValue(peakRate));
        onoff.SetAttribute("PacketSize",  UintegerValue(packetSize));
        onoff.SetAttribute("OnTime",      StringValue(onDist));
        onoff.SetAttribute("OffTime",     StringValue(offDist));

        ApplicationContainer clientApp = onoff.Install(remoteHost);
        clientApp.Start(appStart + MilliSeconds(50));
        clientApp.Stop(appStop);
    }
}

// ---------------------------------------------------------------------------
// Baseline policies — C++ implementations matching evaluate.py exactly
// ---------------------------------------------------------------------------

// action 13 == (0,0,0) delta — no change
constexpr int ACTION_NO_CHANGE = 13;

int
ActionFromDelta(int dEmbb, int dUrllc, int dMmtc)
{
    return (dEmbb + 1) * 9 + (dUrllc + 1) * 3 + (dMmtc + 1);
}

// Random — uniform over 27 actions
// Uses NS-3 UniformRandomVariable so the seed is controlled by RngSeedManager.
int
RandomPolicy(const std::array<float, 18>& /* obs */)
{
    static Ptr<UniformRandomVariable> rng = []() {
        auto r = CreateObject<UniformRandomVariable>();
        r->SetAttribute("Min", DoubleValue(0));
        r->SetAttribute("Max", DoubleValue(26));
        return r;
    }();
    return static_cast<int>(std::round(rng->GetValue()));
}

// Round-Robin — target PRB split 9 / 8 / 8
int
RoundRobinPolicy(const std::array<float, 18>& obs)
{
    const float target[3]  = {9.0f / 25.0f, 8.0f / 25.0f, 8.0f / 25.0f};
    const float deficit[3] = {target[0] - obs[0],
                               target[1] - obs[1],
                               target[2] - obs[2]};

    int receiver = 0, donor = 0;
    for (int i = 1; i < 3; ++i)
    {
        if (deficit[i] > deficit[receiver]) receiver = i;
        if (deficit[i] < deficit[donor])    donor    = i;
    }

    const float threshold = 1.0f / 25.0f;
    if (deficit[receiver] < threshold && std::abs(deficit[donor]) < threshold)
        return ACTION_NO_CHANGE;

    int delta[3] = {0, 0, 0};
    delta[receiver] =  1;
    delta[donor]    = -1;
    return ActionFromDelta(delta[0], delta[1], delta[2]);
}

// Greedy-PF — move one PRB from the highest-efficiency active slice
//             to the lowest-efficiency active slice
int
GreedyPFPolicy(const std::array<float, 18>& obs)
{
    const float prbFrac[3] = {obs[0] + 1e-9f, obs[1] + 1e-9f, obs[2] + 1e-9f};
    const float thrNorm[3] = {obs[3], obs[4], obs[5]};
    const float eff[3]     = {thrNorm[0] / prbFrac[0],
                               thrNorm[1] / prbFrac[1],
                               thrNorm[2] / prbFrac[2]};
    const bool  active[3]  = {thrNorm[0] > 1e-4f,
                               thrNorm[1] > 1e-4f,
                               thrNorm[2] > 1e-4f};

    int activeCount = 0;
    for (int i = 0; i < 3; ++i)
        if (active[i]) ++activeCount;
    if (activeCount < 2)
        return ACTION_NO_CHANGE;

    int   donor = -1, receiver = -1;
    float maxE  = -1e9f, minE = 1e9f;
    for (int i = 0; i < 3; ++i)
    {
        if (!active[i]) continue;
        if (eff[i] > maxE) { maxE = eff[i]; donor    = i; }
        if (eff[i] < minE) { minE = eff[i]; receiver = i; }
    }
    if (donor == receiver)
        return ACTION_NO_CHANGE;

    int delta[3] = {0, 0, 0};
    delta[donor]    = -1;
    delta[receiver] =  1;
    return ActionFromDelta(delta[0], delta[1], delta[2]);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int
main(int argc, char* argv[])
{
    // ── Command line ────────────────────────────────────────────────────────
    int         episodes   = 10;
    int         maxSteps   = 1000;
    uint32_t    seed       = 99;
    std::string outPath    = "results/baseline_cpp.json";

    CommandLine cmd(__FILE__);
    cmd.AddValue("episodes",  "Episodes per policy",          episodes);
    cmd.AddValue("maxSteps",  "Max steps per episode",        maxSteps);
    cmd.AddValue("seed",      "RNG seed",                     seed);
    cmd.AddValue("out",       "Output JSON path",             outPath);
    cmd.Parse(argc, argv);

    // simTime = 3 policies × episodes × maxSteps × 100ms step + 10s buffer
    const double simTimeSeconds =
        3.0 * static_cast<double>(episodes) *
              static_cast<double>(maxSteps) * 0.1 + 10.0;

    std::cout << "=== 5G NR Slice Baseline Evaluation ===\n"
              << "Policies: Random, Round-Robin, Greedy-PF\n"
              << "Episodes: " << episodes << "  MaxSteps: " << maxSteps
              << "  Seed: " << seed << "\n"
              << "SimTime:  " << simTimeSeconds << " s\n"
              << "Output:   " << outPath << "\n\n";

    // ── RNG ─────────────────────────────────────────────────────────────────
    RngSeedManager::SetSeed(1);
    RngSeedManager::SetRun(seed);

    // ── Topology constants — identical to slice-rl-sim.cc ──────────────────
    constexpr uint32_t gnbCount         = 1;
    constexpr uint32_t embbUes          = 10;
    constexpr uint32_t urllcUes         = 5;
    constexpr uint32_t mmtcUes          = 20;
    constexpr uint8_t  numerology       = 1;
    constexpr double   centralFrequency = 3.5e9;
    constexpr double   bandwidth        = 20e6;
    constexpr uint16_t totalPrbs        = 25;

    const Time appStart = Seconds(0.2);
    const Time appStop  = Seconds(simTimeSeconds - 0.05);

    // ── Nodes ───────────────────────────────────────────────────────────────
    NodeContainer gnbNodes;  gnbNodes.Create(gnbCount);
    NodeContainer embbNodes; embbNodes.Create(embbUes);
    NodeContainer urllcNodes;urllcNodes.Create(urllcUes);
    NodeContainer mmtcNodes; mmtcNodes.Create(mmtcUes);

    NodeContainer allUeNodes;
    allUeNodes.Add(embbNodes);
    allUeNodes.Add(urllcNodes);
    allUeNodes.Add(mmtcNodes);

    // ── Mobility — identical to slice-rl-sim.cc ─────────────────────────────
    MobilityHelper gnbMobility;
    gnbMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    gnbMobility.Install(gnbNodes);
    gnbNodes.Get(0)->GetObject<MobilityModel>()->SetPosition(Vector(0.0, 0.0, 10.0));

    MobilityHelper ueMobility;
    ueMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    ueMobility.Install(allUeNodes);

    for (uint32_t i = 0; i < embbNodes.GetN(); ++i)
        embbNodes.Get(i)->GetObject<MobilityModel>()->SetPosition(
            Vector(20.0 + i * 3.0, 0.0, 1.5));
    for (uint32_t i = 0; i < urllcNodes.GetN(); ++i)
        urllcNodes.Get(i)->GetObject<MobilityModel>()->SetPosition(
            Vector(30.0 + i * 2.5, 8.0, 1.5));
    for (uint32_t i = 0; i < mmtcNodes.GetN(); ++i)
        mmtcNodes.Get(i)->GetObject<MobilityModel>()->SetPosition(
            Vector(45.0 + i * 1.5, -10.0, 1.5));

    // ── NR helpers ──────────────────────────────────────────────────────────
    Ptr<NrPointToPointEpcHelper> epcHelper =
        CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> beamformingHelper =
        CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nrHelper = CreateObject<NrHelper>();

    nrHelper->SetEpcHelper(epcHelper);
    nrHelper->SetBeamformingHelper(beamformingHelper);
    beamformingHelper->SetAttribute(
        "BeamformingMethod",
        TypeIdValue(DirectPathBeamforming::GetTypeId()));

    nrHelper->SetSchedulerTypeId(NrMacSchedulerOfdmaAi::GetTypeId());

    // ── Baseline env ─────────────────────────────────────────────────────────
    Ptr<NrSliceBaselineEnv> baselineEnv = CreateObject<NrSliceBaselineEnv>();
    baselineEnv->SetEpisodes(episodes);
    baselineEnv->SetMaxSteps(maxSteps);
    baselineEnv->SetOutputPath(outPath);

    // Register the scheduler callback BEFORE InstallGnbDevice
    nrHelper->SetSchedulerAttribute(
        "NotifyCbDl",
        CallbackValue(
            MakeCallback(&NrSliceBaselineEnv::OnSchedulerNotify, baselineEnv)));
    nrHelper->SetSchedulerAttribute("ActiveDlAi", BooleanValue(true));

    // ── Antenna ──────────────────────────────────────────────────────────────
    nrHelper->SetUeAntennaAttribute("NumRows",    UintegerValue(1));
    nrHelper->SetUeAntennaAttribute("NumColumns", UintegerValue(1));
    nrHelper->SetGnbAntennaAttribute("NumRows",    UintegerValue(4));
    nrHelper->SetGnbAntennaAttribute("NumColumns", UintegerValue(4));

    // ── Channel ──────────────────────────────────────────────────────────────
    CcBwpCreator ccBwpCreator;
    CcBwpCreator::SimpleOperationBandConf bandConf(centralFrequency, bandwidth, 1);
    bandConf.m_numBwp = 1;
    OperationBandInfo band =
        ccBwpCreator.CreateOperationBandContiguousCc(bandConf);

    Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();
    channelHelper->ConfigureFactories("UMi", "Default", "ThreeGpp");
    channelHelper->AssignChannelsToBands({band}, NrChannelHelper::INIT_PROPAGATION);

    BandwidthPartInfoPtrVector allBwps = CcBwpCreator::GetAllBwps({band});

    // ── Devices ──────────────────────────────────────────────────────────────
    NetDeviceContainer gnbDevs   = nrHelper->InstallGnbDevice(gnbNodes,   allBwps);
    NetDeviceContainer embbDevs  = nrHelper->InstallUeDevice(embbNodes,   allBwps);
    NetDeviceContainer urllcDevs = nrHelper->InstallUeDevice(urllcNodes,  allBwps);
    NetDeviceContainer mmtcDevs  = nrHelper->InstallUeDevice(mmtcNodes,   allBwps);

    NetDeviceContainer allUeDevs;
    allUeDevs.Add(embbDevs);
    allUeDevs.Add(urllcDevs);
    allUeDevs.Add(mmtcDevs);

    nrHelper->GetGnbPhy(gnbDevs.Get(0), 0)->SetAttribute(
        "Numerology", UintegerValue(numerology));
    nrHelper->GetGnbPhy(gnbDevs.Get(0), 0)->SetAttribute(
        "TxPower", DoubleValue(30.0));

    // ── Internet stack & routing ──────────────────────────────────────────
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
        ipv4RoutingHelper.GetStaticRouting(
            remoteHost->GetObject<Ipv4>());
    remoteHostStaticRouting->AddNetworkRouteTo(
        Ipv4Address("7.0.0.0"), Ipv4Mask("255.0.0.0"), 1);

    Ipv4InterfaceContainer embbIfaces  = epcHelper->AssignUeIpv4Address(embbDevs);
    Ipv4InterfaceContainer urllcIfaces = epcHelper->AssignUeIpv4Address(urllcDevs);
    Ipv4InterfaceContainer mmtcIfaces  = epcHelper->AssignUeIpv4Address(mmtcDevs);

    nrHelper->AttachToClosestGnb(allUeDevs, gnbDevs);

    for (uint32_t i = 0; i < allUeNodes.GetN(); ++i)
    {
        Ptr<Ipv4StaticRouting> ueStaticRouting =
            ipv4RoutingHelper.GetStaticRouting(
                allUeNodes.Get(i)->GetObject<Ipv4>());
        ueStaticRouting->SetDefaultRoute(
            epcHelper->GetUeDefaultGatewayAddress(), 1);
    }

    // ── Traffic — identical to slice-rl-sim.cc ───────────────────────────
    // eMBB: video-like.  10 Mbps peak, on~2s, off~1s → avg ≈ 6.6 Mbps/UE
    InstallOnOffTraffic(remoteHost, embbNodes, embbIfaces,
                        1000, 1448, DataRate("10Mbps"),
                        2.0, 1.0, appStart, appStop);

    // URLLC: event-driven.  5 Mbps peak, on~50ms, off~200ms → avg ≈ 1 Mbps/UE
    InstallOnOffTraffic(remoteHost, urllcNodes, urllcIfaces,
                        2000, 20, DataRate("5Mbps"),
                        0.05, 0.20, appStart, appStop);

    // mMTC: IoT duty cycle.  2 Mbps peak, on~1s, off~9s → avg ≈ 0.2 Mbps/UE
    InstallOnOffTraffic(remoteHost, mmtcNodes, mmtcIfaces,
                        3000, 128, DataRate("2Mbps"),
                        1.0, 9.0, appStart, appStop);

    // ── Flow monitor ─────────────────────────────────────────────────────
    FlowMonitorHelper flowmonHelper;
    Ptr<FlowMonitor>        monitor    = flowmonHelper.InstallAll();
    Ptr<Ipv4FlowClassifier> classifier =
        DynamicCast<Ipv4FlowClassifier>(flowmonHelper.GetClassifier());

    // ── Env config — identical values to slice-rl-sim.cc ────────────────
    NrSliceBaselineEnv::Config cfg;
    cfg.totalPrbs       = totalPrbs;
    cfg.stepInterval    = MilliSeconds(100);
    cfg.simTime         = Seconds(simTimeSeconds);
    cfg.initialPrbAlloc = {10, 8, 7};
    cfg.maxUes          = {20, 10, 50};
    cfg.maxThrMbps      = {100.0, 25.0, 8.0};
    cfg.maxLatMs        = {50.0,  15.0, 500.0};
    cfg.minThrMbps      = {10.0,  1.0,  0.1};

    // ── Register policies ────────────────────────────────────────────────
    baselineEnv->AddPolicy("Random",      RandomPolicy);
    baselineEnv->AddPolicy("Round-Robin", RoundRobinPolicy);
    baselineEnv->AddPolicy("Greedy-PF",   GreedyPFPolicy);

    // ── Wire everything up & run ─────────────────────────────────────────
    std::array<NetDeviceContainer, NrSliceBaselineEnv::kSliceCount> ueDevsBySlice{
        embbDevs, urllcDevs, mmtcDevs};

    baselineEnv->SetFlowMonitor(monitor, classifier);
    baselineEnv->Initialize(cfg, nrHelper, gnbDevs, ueDevsBySlice);

    Simulator::Stop(Seconds(simTimeSeconds + 1.0));
    Simulator::Run();
    Simulator::Destroy();

    return 0;
}
