# AI-Powered Dynamic Resource Allocation in 5G NR Network Slicing

This project implements Deep Reinforcement Learning based resource allocation
for 5G NR network slicing using:

- ns-3.45
- 5G-LENA v4.1
- ns3-opengym
- DRL agents (DQN, PPO, R2D2)

The system dynamically allocates PRBs between three slices:

- eMBB
- URLLC
- mMTC

Architecture:

ns-3 simulation → OpenGym interface → DRL agent → resource scheduler

