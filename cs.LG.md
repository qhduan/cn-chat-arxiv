# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CROP: Conservative Reward for Model-based Offline Policy Optimization.](http://arxiv.org/abs/2310.17245) | CROP提出了一种保守奖励的模型训练方法用于基于模型的离线策略优化，通过同时最小化估计误差和随机动作奖励来实现保守的奖励估计。 |
| [^2] | [InstructionGPT-4: A 200-Instruction Paradigm for Fine-Tuning MiniGPT-4.](http://arxiv.org/abs/2308.12067) | InstructionGPT-4通过仅使用200个例子进行微调，在多模式指令数据质量度量和选择器的帮助下，在各种评估任务中优于原始的MiniGPT-4。 |
| [^3] | [PePNet: A Periodicity-Perceived Workload Prediction Network Supporting Rare Occurrence of Heavy Workload.](http://arxiv.org/abs/2308.01917) | PePNet是一种支持罕见重负载的工作负载预测网络，通过周期性感知机制和融合多尺度序列学习的能力提高了整体特别是重负载的准确性。 |
| [^4] | [Incentivizing Honesty among Competitors in Collaborative Learning and Optimization.](http://arxiv.org/abs/2305.16272) | 这项研究提出了一个模型来描述在协作学习中竞争对手的不诚实行为，提出了机制来激励诚实沟通，并确保学习质量与全面合作相当。 |
| [^5] | [SIMGA: A Simple and Effective Heterophilous Graph Neural Network with Efficient Global Aggregation.](http://arxiv.org/abs/2305.09958) | 本文章提出了一种简单有效的异质图神经网络模型SIMGA，它通过SimRank全局聚合来解决异质性节点聚合的问题，具有接近于线性的传播效率，同时具有良好的有效性和可扩展性。 |

# 详细

[^1]: CROP: 保守奖励用于基于模型的离线策略优化

    CROP: Conservative Reward for Model-based Offline Policy Optimization. (arXiv:2310.17245v1 [cs.LG])

    [http://arxiv.org/abs/2310.17245](http://arxiv.org/abs/2310.17245)

    CROP提出了一种保守奖励的模型训练方法用于基于模型的离线策略优化，通过同时最小化估计误差和随机动作奖励来实现保守的奖励估计。

    

    离线强化学习旨在使用收集到的数据进行策略优化，而无需进行在线交互。基于模型的方法在解决离线强化学习挑战方面特别有吸引力，因为它们能够通过使用模型生成数据来缓解离线数据的限制。之前的研究表明，在策略优化过程中将保守性引入模型或Q函数可以有效缓解离线强化学习中普遍存在的分布漂移问题。然而，关于奖励估计中保守性的影响的研究仍然不足。本文提出了一种新颖的基于模型的离线强化学习算法CROP，该算法在模型训练中保守地估计奖励。为了实现保守的奖励估计，CROP同时最小化估计误差和随机动作的奖励。理论分析表明，这种保守的奖励机制导致...（文章摘要未完，下同）

    Offline reinforcement learning (RL) aims to optimize policy using collected data without online interactions. Model-based approaches are particularly appealing for addressing offline RL challenges due to their capability to mitigate the limitations of offline data through data generation using models. Prior research has demonstrated that introducing conservatism into the model or Q-function during policy optimization can effectively alleviate the prevalent distribution drift problem in offline RL. However, the investigation into the impacts of conservatism in reward estimation is still lacking. This paper proposes a novel model-based offline RL algorithm, Conservative Reward for model-based Offline Policy optimization (CROP), which conservatively estimates the reward in model training. To achieve a conservative reward estimation, CROP simultaneously minimizes the estimation error and the reward of random actions. Theoretical analysis shows that this conservative reward mechanism leads 
    
[^2]: InstructionGPT-4: 一个200指令范式用于微调MiniGPT-4

    InstructionGPT-4: A 200-Instruction Paradigm for Fine-Tuning MiniGPT-4. (arXiv:2308.12067v1 [cs.LG])

    [http://arxiv.org/abs/2308.12067](http://arxiv.org/abs/2308.12067)

    InstructionGPT-4通过仅使用200个例子进行微调，在多模式指令数据质量度量和选择器的帮助下，在各种评估任务中优于原始的MiniGPT-4。

    

    多模式大型语言模型通过两阶段的训练过程获取其遵循指令的能力：在图像-文本对上进行预训练，然后在监督式视觉-语言指令数据上进行微调。最近的研究表明，即使只有有限量的高质量遵循指令数据，大型语言模型也能取得令人满意的结果。在本文中，我们介绍了InstructionGPT-4，它经过微调的数据集只包含200个例子，约占MiniGPT-4对齐数据集中使用的遵循指令数据的6%。我们首先提出了几个用于评估多模式指令数据质量的度量指标。基于这些度量指标，我们提出了一个简单而有效的数据选择器，自动识别和过滤低质量的视觉-语言数据。通过采用这种方法，InstructionGPT-4在各种评估（如视觉问答、GPT-4偏好）上优于原始的MiniGPT-4。总体而言，我们的研究发现...

    Multimodal large language models acquire their instruction-following capabilities through a two-stage training process: pre-training on image-text pairs and fine-tuning on supervised vision-language instruction data. Recent studies have shown that large language models can achieve satisfactory results even with a limited amount of high-quality instruction-following data. In this paper, we introduce InstructionGPT-4, which is fine-tuned on a small dataset comprising only 200 examples, amounting to approximately 6% of the instruction-following data used in the alignment dataset for MiniGPT-4. We first propose several metrics to access the quality of multimodal instruction data. Based on these metrics, we present a simple and effective data selector to automatically identify and filter low-quality vision-language data. By employing this method, InstructionGPT-4 outperforms the original MiniGPT-4 on various evaluations (e.g., visual question answering, GPT-4 preference). Overall, our findi
    
[^3]: PePNet: 一种支持罕见重负载的周期性感知工作负载预测网络

    PePNet: A Periodicity-Perceived Workload Prediction Network Supporting Rare Occurrence of Heavy Workload. (arXiv:2308.01917v1 [cs.DC])

    [http://arxiv.org/abs/2308.01917](http://arxiv.org/abs/2308.01917)

    PePNet是一种支持罕见重负载的工作负载预测网络，通过周期性感知机制和融合多尺度序列学习的能力提高了整体特别是重负载的准确性。

    

    云提供商可以从准确的工作负载预测中获得巨大的好处。然而，云服务器的工作负载高度变化，有时会发生重负载突发事件，这使得工作负载预测具有挑战性。目前有两种主要的工作负载预测方法：统计方法和基于神经网络的方法。前者依赖于强大的数学假设，当预测高度变化的工作负载时，其准确性较低。而后者在整体准确性上更高，但容易受到重负载和常见负载之间数据不平衡的影响，这会影响神经网络模型对重负载的预测准确性。无论是统计方法的整体不准确性还是基于神经网络的模型对重负载的不准确性都会导致服务级别协议的违规。因此，我们提出了PePNet来提高整体特别是重负载预测的准确性。它具有两个独特的特点：周期性感知机制和融合多尺度序列学习的能力。

    Cloud providers can greatly benefit from accurate workload prediction. However, the workload of cloud servers is highly variable, with occasional heavy workload bursts. This makes workload prediction challenging.  There are mainly two categories of workload prediction methods: statistical methods and neural-network-based ones. The former ones rely on strong mathematical assumptions and have reported low accuracy when predicting highly variable workload. The latter ones offer higher overall accuracy, yet they are vulnerable to data imbalance between heavy workload and common one. This impairs the prediction accuracy of neural network-based models on heavy workload.  Either the overall inaccuracy of statistic methods or the heavy-workload inaccuracy of neural-network-based models can cause service level agreement violations.  Thus, we propose PePNet to improve overall especially heavy workload prediction accuracy. It has two distinctive characteristics:  (i) A Periodicity-Perceived Mecha
    
[^4]: 在协同学习和优化中激励竞争对手诚实行为的研究

    Incentivizing Honesty among Competitors in Collaborative Learning and Optimization. (arXiv:2305.16272v1 [cs.LG])

    [http://arxiv.org/abs/2305.16272](http://arxiv.org/abs/2305.16272)

    这项研究提出了一个模型来描述在协作学习中竞争对手的不诚实行为，提出了机制来激励诚实沟通，并确保学习质量与全面合作相当。

    

    协同学习技术能够让机器学习模型的训练比仅利用单一数据源的模型效果更好。然而，在许多情况下，潜在的参与者是下游任务中的竞争对手，如每个都希望通过提供最佳推荐来吸引客户的公司。这可能会激励不诚实的更新，损害其他参与者的模型，从而可能破坏协作的好处。在这项工作中，我们制定了一个模型来描述这种交互，并在该框架内研究了两个学习任务：单轮均值估计和强凸目标的多轮 SGD。对于一类自然的参与者行为，我们发现理性的客户会被激励强烈地操纵他们的更新，从而防止学习。然后，我们提出了机制来激励诚实沟通，并确保学习质量与全面合作相当。最后，我们通过实验证明了这一点。

    Collaborative learning techniques have the potential to enable training machine learning models that are superior to models trained on a single entity's data. However, in many cases, potential participants in such collaborative schemes are competitors on a downstream task, such as firms that each aim to attract customers by providing the best recommendations. This can incentivize dishonest updates that damage other participants' models, potentially undermining the benefits of collaboration. In this work, we formulate a game that models such interactions and study two learning tasks within this framework: single-round mean estimation and multi-round SGD on strongly-convex objectives. For a natural class of player actions, we show that rational clients are incentivized to strongly manipulate their updates, preventing learning. We then propose mechanisms that incentivize honest communication and ensure learning quality comparable to full cooperation. Lastly, we empirically demonstrate the
    
[^5]: SIMGA：一种简单有效的异质图神经网络结构与高效的全局聚合

    SIMGA: A Simple and Effective Heterophilous Graph Neural Network with Efficient Global Aggregation. (arXiv:2305.09958v1 [cs.LG])

    [http://arxiv.org/abs/2305.09958](http://arxiv.org/abs/2305.09958)

    本文章提出了一种简单有效的异质图神经网络模型SIMGA，它通过SimRank全局聚合来解决异质性节点聚合的问题，具有接近于线性的传播效率，同时具有良好的有效性和可扩展性。

    

    图神经网络在图学习领域取得了巨大成功，但遇到异质性时会出现性能下降，即因为局部和统一聚合而导致的相邻节点不相似。现有的异质性图神经网络中，试图整合全局聚合的尝试通常需要迭代地维护和更新全图信息，对于一个具有 $n$ 个节点的图，这需要 $\mathcal{O}(n^2)$ 的计算效率，从而导致对大型图的扩展性较差。在本文中，我们提出了 SIMGA，一种将 SimRank 结构相似度测量作为全局聚合的 GNN 结构。 SIMGA 的设计简单，且在效率和有效性方面都有着有 promising 的结果。SIMGA 的简单性使其成为第一个可以实现接近于线性的 $n$ 传播效率的异质性 GNN 模型。我们从理论上证明了它的有效性，将 SimRank 视为 GNN 的一种新解释，并证明了汇聚节点表示的有效性。

    Graph neural networks (GNNs) realize great success in graph learning but suffer from performance loss when meeting heterophily, i.e. neighboring nodes are dissimilar, due to their local and uniform aggregation. Existing attempts in incoorporating global aggregation for heterophilous GNNs usually require iteratively maintaining and updating full-graph information, which entails $\mathcal{O}(n^2)$ computation efficiency for a graph with $n$ nodes, leading to weak scalability to large graphs. In this paper, we propose SIMGA, a GNN structure integrating SimRank structural similarity measurement as global aggregation. The design of SIMGA is simple, yet it leads to promising results in both efficiency and effectiveness. The simplicity of SIMGA makes it the first heterophilous GNN model that can achieve a propagation efficiency near-linear to $n$. We theoretically demonstrate its effectiveness by treating SimRank as a new interpretation of GNN and prove that the aggregated node representation
    

