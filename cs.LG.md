# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Graph Reinforcement Learning for Calibration-Aware Quantum Circuit Routing](https://arxiv.org/abs/2606.12816) | 该论文提出了一种利用实时校准数据通过图强化学习进行量子电路路由的方法，在中小型量子电路上显著提升了保真度，平均精确保真度达到0.727，远超基线方法。 |
| [^2] | [SymQNet: Amortized Acquisition for Low-Latency Adaptive Hamiltonian Learning](https://arxiv.org/abs/2606.12808) | 本文提出SymQNet，一种通过离线学习后验条件采集策略并在线快速执行，从而大幅降低自适应哈密顿学习延迟的摊销强化学习方法。 |

# 详细

[^1]: 面向校准感知的量子电路路由的图强化学习

    Graph Reinforcement Learning for Calibration-Aware Quantum Circuit Routing

    [https://arxiv.org/abs/2606.12816](https://arxiv.org/abs/2606.12816)

    该论文提出了一种利用实时校准数据通过图强化学习进行量子电路路由的方法，在中小型量子电路上显著提升了保真度，平均精确保真度达到0.727，远超基线方法。

    

    arXiv:2606.12816v3 公告类型：替换-交叉 摘要：量子电路路由是为噪声中等规模量子处理器编译程序的关键步骤。即使通过标准开销指标看起来高效的路径，当它们经过校准不良的耦合器时，仍然可能损失保真度。我们研究了一种校准感知的图强化学习路由器，它利用同一天的IBM Heron r2校准数据来选择硬件边缘的SWAP操作。我们使用近端策略优化训练该策略，并通过九个慕尼黑量子工具包（MQT）基准电路和三个校准快照，以精确模拟的保真度对其进行评估。在这些评估中，合并的平均精确保真度为0.727，而SABRE-best20为0.440，目标感知SABRE为0.481。我们观察到，保真度的提升伴随着更高的路由双量子比特计数，并且集中在5量子比特和8量子比特电路族中；在固定的树状动作图下，所有10量子比特族都更倾向于SABRE-best20。总体而言，我们的方法在中小型电路上显著优于基线。

    arXiv:2606.12816v3 Announce Type: replace-cross  Abstract: Quantum circuit routing is a key step in compiling programs for noisy intermediate-scale quantum processors. Routes that appear efficient by standard overhead metrics can still lose fidelity when they pass through poorly calibrated couplers. We study a calibration-aware graph reinforcement-learning router that uses same-day IBM Heron r2 calibration data to choose hardware-edge SWAPs. We train the policy with proximal policy optimization and evaluate it with exact simulated fidelity across nine Munich Quantum Toolkit (MQT) Bench circuits and three calibration snapshots. Across these evaluations, pooled mean exact fidelity is $0.727$, compared with $0.440$ for SABRE-best20 and $0.481$ for target-aware SABRE. We observed that fidelity gains came with higher routed two-qubit counts and were concentrated in 5 qubit and 8 qubit circuit families; under the fixed tree action graph, all 10 qubit families favored SABRE-best20. Overall, o
    
[^2]: SymQNet：面向低延迟自适应哈密顿学习的摊销采集方法

    SymQNet: Amortized Acquisition for Low-Latency Adaptive Hamiltonian Learning

    [https://arxiv.org/abs/2606.12808](https://arxiv.org/abs/2606.12808)

    本文提出SymQNet，一种通过离线学习后验条件采集策略并在线快速执行，从而大幅降低自适应哈密顿学习延迟的摊销强化学习方法。

    

    自适应哈密顿学习是校准和表征量子器件的核心。在自适应控制器中，选择下一个实验本身就是一个计算过程。贝叶斯设计规则在每次后验更新后都会重新计算，这一步骤可能需要数秒时间。在数百次实验轮次中，这些秒数会累积成为自适应过程显著的时钟时间成本。我们提出了SymQNet，一种用于低延迟自适应哈密顿学习的摊销强化学习方法。SymQNet离线学习一个基于后验条件的采集策略，然后在线使用快速策略前向传播，同时保留贝叶斯后验反馈。在横向场伊辛模型基准测试中，与有界费舍尔信息搜索和有界两步贝叶斯主动学习（BALD）相比，SymQNet显著降低了采集延迟。在五个量子比特上，相对于这两种方法，其仅采集决策延迟分别降低了47.1倍和72.6倍。

    arXiv:2606.12808v3 Announce Type: replace-cross  Abstract: Adaptive Hamiltonian learning is central to calibrating and characterizing quantum devices. In an adaptive controller, choosing the next experiment is itself a computation. Bayesian design rules are recomputed after every posterior update, and that step can take seconds. Across hundreds of shots, those seconds become a significant wall-clock cost for adaptivity. We introduce SymQNet, an amortized reinforcement-learning approach for low-latency adaptive Hamiltonian learning. SymQNet learns a posterior-conditioned acquisition policy offline, then uses a fast policy forward pass online while retaining Bayesian posterior feedback. On transverse-field Ising benchmarks, SymQNet substantially reduces acquisition latency relative to bounded Fisher-information search and bounded two-step Bayesian active learning by disagreement (BALD). At five qubits, it reduces acquisition-only decision latency by $47.1\times$ and $72.6\times$ relative
    

