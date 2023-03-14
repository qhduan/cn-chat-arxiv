# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference.](http://arxiv.org/abs/2303.06182) | 本文提出了三种优化技术来缓解混合专家（MoE）模型在推理时的低效率，包括动态门控、专家缓冲和专家负载平衡。这些技术可以显著提高执行时间和减少内存使用。 |
| [^2] | [MOELA: A Multi-Objective Evolutionary/Learning Design Space Exploration Framework for 3D Heterogeneous Manycore Platforms.](http://arxiv.org/abs/2303.06169) | MOELA是一个多目标设计空间探索框架，结合了进化搜索和学习搜索，用于优化3D NoC启用的异构多核系统中的多个目标，相比现有技术，MOELA可以提高解决方案的查找速度，提高Pareto Hypervolume（PHV）和能量延迟乘积（EDP）。 |

# 详细

[^1]: 迈向MoE部署：缓解混合专家（MoE）推理中的低效率

    Towards MoE Deployment: Mitigating Inefficiencies in Mixture-of-Expert (MoE) Inference. (arXiv:2303.06182v1 [cs.DC])

    [http://arxiv.org/abs/2303.06182](http://arxiv.org/abs/2303.06182)

    本文提出了三种优化技术来缓解混合专家（MoE）模型在推理时的低效率，包括动态门控、专家缓冲和专家负载平衡。这些技术可以显著提高执行时间和减少内存使用。

    This paper proposes three optimization techniques to mitigate inefficiencies in Mixture-of-Experts (MoE) models during inference, including dynamic gating, expert buffering, and expert load balancing. These techniques can significantly improve execution time and reduce memory usage.

    混合专家（MoE）模型最近在计算机视觉和自然语言处理的广泛任务中取得了最先进的性能。它们在训练期间有效地扩展了模型容量，同时增加的计算成本很小。然而，由于其庞大的模型大小和复杂的通信模式，部署这样的模型进行推理是困难的。在这项工作中，我们提供了两个MoE工作负载的特征化，即语言建模（LM）和机器翻译（MT），并确定了它们在部署时的低效率来源。我们提出了三种优化技术来缓解低效率的来源，即（1）动态门控，（2）专家缓冲和（3）专家负载平衡。我们展示了动态门控可以使LM的执行时间提高1.25-4倍，MT编码器提高2-5倍，MT解码器提高1.09-1.5倍。它还可以将LM的内存使用减少高达1.36倍，MT的内存使用减少高达1.1倍。

    Mixture-of-Experts (MoE) models have recently gained steam in achieving the state-of-the-art performance in a wide range of tasks in computer vision and natural language processing. They effectively expand the model capacity while incurring a minimal increase in computation cost during training. However, deploying such models for inference is difficult due to their large model size and complex communication pattern. In this work, we provide a characterization of two MoE workloads, namely Language Modeling (LM) and Machine Translation (MT) and identify their sources of inefficiencies at deployment.  We propose three optimization techniques to mitigate sources of inefficiencies, namely (1) Dynamic gating, (2) Expert Buffering, and (3) Expert load balancing. We show that dynamic gating improves execution time by 1.25-4$\times$ for LM, 2-5$\times$ for MT Encoder and 1.09-1.5$\times$ for MT Decoder. It also reduces memory usage by up to 1.36$\times$ for LM and up to 1.1$\times$ for MT. We f
    
[^2]: MOELA：用于3D异构多核平台的多目标进化/学习设计空间探索框架

    MOELA: A Multi-Objective Evolutionary/Learning Design Space Exploration Framework for 3D Heterogeneous Manycore Platforms. (arXiv:2303.06169v1 [cs.LG])

    [http://arxiv.org/abs/2303.06169](http://arxiv.org/abs/2303.06169)

    MOELA是一个多目标设计空间探索框架，结合了进化搜索和学习搜索，用于优化3D NoC启用的异构多核系统中的多个目标，相比现有技术，MOELA可以提高解决方案的查找速度，提高Pareto Hypervolume（PHV）和能量延迟乘积（EDP）。

    MOELA is a multi-objective design space exploration framework that combines evolutionary-based search with learning-based local search to optimize multiple objectives in 3D NoC enabled heterogeneous manycore systems. Compared to state-of-the-art approaches, MOELA increases the speed of finding solutions, improves Pareto Hypervolume (PHV) and energy-delay-product (EDP).

    为了支持深度机器学习和图处理等新兴应用，需要能够集成多个处理元件（PE）的3D网络芯片（NoC）启用的异构多核平台。然而，由于巨大的设计空间和长时间的评估时间，设计具有多个目标的这种复杂系统可能具有挑战性。为了优化这样的系统，我们提出了一种新的多目标设计空间探索框架MOELA，它将基于进化的搜索的优点与基于学习的局部搜索相结合，以快速确定PE和通信链路的放置位置，以优化3D NoC启用的异构多核系统中的多个目标（例如延迟，吞吐量和能量）。与现有技术相比，MOELA可以将解决方案的查找速度提高多达128倍，在5个目标的情况下，可以将Pareto Hypervolume（PHV）提高多达12.14倍，并将能量延迟乘积（EDP）提高多达7.7％。

    To enable emerging applications such as deep machine learning and graph processing, 3D network-on-chip (NoC) enabled heterogeneous manycore platforms that can integrate many processing elements (PEs) are needed. However, designing such complex systems with multiple objectives can be challenging due to the huge associated design space and long evaluation times. To optimize such systems, we propose a new multi-objective design space exploration framework called MOELA that combines the benefits of evolutionary-based search with a learning-based local search to quickly determine PE and communication link placement to optimize multiple objectives (e.g., latency, throughput, and energy) in 3D NoC enabled heterogeneous manycore systems. Compared to state-of-the-art approaches, MOELA increases the speed of finding solutions by up to 128x, leads to a better Pareto Hypervolume (PHV) by up to 12.14x and improves energy-delay-product (EDP) by up to 7.7% in a 5-objective scenario.
    

