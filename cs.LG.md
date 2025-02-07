# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Efficient Online Unlearning via Hessian-Free Recollection of Individual Data Statistics](https://arxiv.org/abs/2404.01712) | 通过提出的Hessian-free在线遗忘方法，实现了近乎瞬时的在线遗忘，仅需要进行矢量加法操作。 |
| [^2] | [Scenario-Based Curriculum Generation for Multi-Agent Autonomous Driving](https://arxiv.org/abs/2403.17805) | 提出了MATS-Gym，一个用于在CARLA中训练智能体的多智能体交通场景框架，能够自动生成具有可变智能体数量的交通场景并整合了各种现有的交通场景描述方法。 |
| [^3] | [Partitioned Neural Network Training via Synthetic Intermediate Labels](https://arxiv.org/abs/2403.11204) | 该研究提出了一种通过将模型分区到不同GPU上，并生成合成中间标签来训练各个部分的方法，以缓解大规模神经网络训练中的内存和计算压力。 |
| [^4] | [Robust Decision Aggregation with Adversarial Experts](https://arxiv.org/abs/2403.08222) | 论文考虑了在既有真实专家又有对抗性专家的情况下的二元决策聚合问题，提出了设计鲁棒聚合器以最小化遗憾的方法，并证明了当真实专家是对称的且对抗性专家不太多时，截尾均值是最优的。 |
| [^5] | [Fast Ergodic Search with Kernel Functions](https://arxiv.org/abs/2403.01536) | 提出了一种使用核函数的快速遍历搜索方法，其在搜索空间维度上具有线性复杂度，可以推广到李群，并且通过数值测试展示比现有算法快两个数量级。 |
| [^6] | [EBBS: An Ensemble with Bi-Level Beam Search for Zero-Shot Machine Translation](https://arxiv.org/abs/2403.00144) | 提出了一种集成方法EBBS，配合新颖的双层束搜索算法，能够优于直接和通过第三语言进行的翻译，并实现知识蒸馏来提高推理效率。 |
| [^7] | [Divide and Conquer: Provably Unveiling the Pareto Front with Multi-Objective Reinforcement Learning](https://arxiv.org/abs/2402.07182) | 这项研究介绍了一个名为IPRO的算法，利用分解任务为一系列单目标问题方法，可可靠地揭示多目标强化学习中实现最优表现的策略的帕累托前沿，同时提供收敛保证和未发现解的距离上限。 |
| [^8] | [Rates of Convergence in the Central Limit Theorem for Markov Chains, with an Application to TD Learning](https://arxiv.org/abs/2401.15719) | 本研究证明了一个非渐近的中心极限定理，并通过应用于TD学习，展示了其实际应用的可行性。 |
| [^9] | [Expressive Modeling Is Insufficient for Offline RL: A Tractable Inference Perspective.](http://arxiv.org/abs/2311.00094) | 本文指出，在离线强化学习任务中，除了表达性强的序列模型，可处理性也起着重要的作用。由于离线数据收集策略和环境动态的随机性，需要精确且高效地回答各种概率查询，以找到有奖励的动作。基于此，本文提出了Trifle（离线强化学习的可处理推理）方法，利用现代可处理概率模型来解决这个问题。 |
| [^10] | [MimicTouch: Learning Human's Control Strategy with Multi-Modal Tactile Feedback.](http://arxiv.org/abs/2310.16917) | MimicTouch是一种新的框架，能够模仿人类的触觉引导控制策略，通过收集来自人类示范者的多模态触觉数据集，来学习并执行复杂任务。 |
| [^11] | [Layer-wise Feedback Propagation.](http://arxiv.org/abs/2308.12053) | 本文提出了一种名为“层级反馈传播（LFP）”的新型神经网络预测器训练方法，通过利用可解释性细化与层级相关性传播（LRP）相结合，根据每个连接对任务的贡献分配奖励，该方法克服了传统梯度下降方法存在的问题。对于各种模型和数据集，LFP取得了与梯度下降相当的性能。 |
| [^12] | [Finding Optimal Diverse Feature Sets with Alternative Feature Selection.](http://arxiv.org/abs/2307.11607) | 本文引入了替代特征选择的概念，将其形式化为优化问题，并通过约束定义了替代特征集，使用户可以控制替代的数量和差异性。我们证明了该问题的NP-hard性，并讨论了如何将传统特征选择方法作为目标集成。实验证明替代特征集确实可以具有高预测质量，同时分析了几个影响因素。 |
| [^13] | [Performative Prediction with Neural Networks.](http://arxiv.org/abs/2304.06879) | 本文提出了执行预测的框架，通过找到具有执行稳定性的分类器来适用于数据分布。通过假设数据分布相对于模型的预测值可Lipschitz连续，使得我们能够放宽对损失函数的假设要求。 |
| [^14] | [Rule Generation for Classification: Scalability, Interpretability, and Fairness.](http://arxiv.org/abs/2104.10751) | 这项研究介绍了一种新的基于规则的分类优化方法，利用列生成线性规划实现可扩展性，并通过分配成本系数和引入额外约束解决了解释性和公平性问题。该方法在局部解释性和公平性之间取得了良好的平衡。 |

# 详细

[^1]: 通过免Hessian重新整合个体数据统计实现高效在线遗忘

    Efficient Online Unlearning via Hessian-Free Recollection of Individual Data Statistics

    [https://arxiv.org/abs/2404.01712](https://arxiv.org/abs/2404.01712)

    通过提出的Hessian-free在线遗忘方法，实现了近乎瞬时的在线遗忘，仅需要进行矢量加法操作。

    

    机器遗忘旨在通过使模型能够选择性地忘记特定数据来维护数据所有者的被遗忘权利。最近的方法表明，一种数据遗忘的方法是通过预先计算和存储携带二阶信息的统计数据，以改进计算和内存效率。然而，它们依赖于苛刻的假设，而且计算/存储受到模型参数维度的诅咒，这使得难以应用到大多数深度神经网络中。在本工作中，我们提出了一种免Hessian在线遗忘方法。我们建议为每个数据点维护一个统计向量，通过重新训练和学习模型之间的差异的仿射随机递归逼近来计算。我们提出的算法实现了近乎瞬时的在线遗忘，因为它只需要进行矢量加法操作。基于重新收集遗忘数据统计的策略，

    arXiv:2404.01712v1 Announce Type: cross  Abstract: Machine unlearning strives to uphold the data owners' right to be forgotten by enabling models to selectively forget specific data. Recent methods suggest that one approach of data forgetting is by precomputing and storing statistics carrying second-order information to improve computational and memory efficiency. However, they rely on restrictive assumptions and the computation/storage suffer from the curse of model parameter dimensionality, making it challenging to apply to most deep neural networks. In this work, we propose a Hessian-free online unlearning method. We propose to maintain a statistical vector for each data point, computed through affine stochastic recursion approximation of the difference between retrained and learned models. Our proposed algorithm achieves near-instantaneous online unlearning as it only requires a vector addition operation. Based on the strategy that recollecting statistics for forgetting data, the p
    
[^2]: 多智能体自主驾驶场景驱动的课程生成

    Scenario-Based Curriculum Generation for Multi-Agent Autonomous Driving

    [https://arxiv.org/abs/2403.17805](https://arxiv.org/abs/2403.17805)

    提出了MATS-Gym，一个用于在CARLA中训练智能体的多智能体交通场景框架，能够自动生成具有可变智能体数量的交通场景并整合了各种现有的交通场景描述方法。

    

    多样化和复杂训练场景的自动化生成在许多复杂学习任务中是重要的。特别是在现实世界的应用领域，如自主驾驶，自动生成课程被认为对获得强健和通用策略至关重要。然而，在充满挑战的仿真环境中，为交通场景中的多个异构智能体进行设计通常被认为是一项繁琐且耗时的任务。在我们的工作中，我们引入了MATS-Gym，一个用于在高保真驾驶模拟器CARLA中训练智能体的多智能体交通场景框架。MATS-Gym是一个用于自主驾驶的多智能体训练框架，使用部分场景规范生成具有可变智能体数量的交通场景。这篇论文将各种现有的交通场景描述方法统一到一个单一的训练框架中，并演示了如何将其与其他自主驾驶算法集成。

    arXiv:2403.17805v1 Announce Type: cross  Abstract: The automated generation of diverse and complex training scenarios has been an important ingredient in many complex learning tasks. Especially in real-world application domains, such as autonomous driving, auto-curriculum generation is considered vital for obtaining robust and general policies. However, crafting traffic scenarios with multiple, heterogeneous agents is typically considered as a tedious and time-consuming task, especially in more complex simulation environments. In our work, we introduce MATS-Gym, a Multi-Agent Traffic Scenario framework to train agents in CARLA, a high-fidelity driving simulator. MATS-Gym is a multi-agent training framework for autonomous driving that uses partial scenario specifications to generate traffic scenarios with variable numbers of agents. This paper unifies various existing approaches to traffic scenario description into a single training framework and demonstrates how it can be integrated wi
    
[^3]: 通过合成中间标签进行分区神经网络训练

    Partitioned Neural Network Training via Synthetic Intermediate Labels

    [https://arxiv.org/abs/2403.11204](https://arxiv.org/abs/2403.11204)

    该研究提出了一种通过将模型分区到不同GPU上，并生成合成中间标签来训练各个部分的方法，以缓解大规模神经网络训练中的内存和计算压力。

    

    大规模神经网络架构的普及，特别是深度学习模型，对资源密集型训练提出了挑战。 GPU 内存约束已经成为训练这些庞大模型的一个明显瓶颈。现有策略，包括数据并行、模型并行、流水线并行和完全分片数据并行，提供了部分解决方案。 特别是模型并行允许将整个模型分布在多个 GPU 上，但随后的这些分区之间的数据通信减慢了训练速度。此外，为在每个 GPU 上存储辅助参数所需的大量内存开销增加了计算需求。 本研究主张不使用整个模型进行训练，而是将模型分区到 GPU 上，并生成合成中间标签来训练各个部分。 通过随机过程生成的这些标签减缓了训练中的内存和计算压力。

    arXiv:2403.11204v1 Announce Type: cross  Abstract: The proliferation of extensive neural network architectures, particularly deep learning models, presents a challenge in terms of resource-intensive training. GPU memory constraints have become a notable bottleneck in training such sizable models. Existing strategies, including data parallelism, model parallelism, pipeline parallelism, and fully sharded data parallelism, offer partial solutions. Model parallelism, in particular, enables the distribution of the entire model across multiple GPUs, yet the ensuing data communication between these partitions slows down training. Additionally, the substantial memory overhead required to store auxiliary parameters on each GPU compounds computational demands. Instead of using the entire model for training, this study advocates partitioning the model across GPUs and generating synthetic intermediate labels to train individual segments. These labels, produced through a random process, mitigate me
    
[^4]: 具有对抗性专家的鲁棒决策聚合

    Robust Decision Aggregation with Adversarial Experts

    [https://arxiv.org/abs/2403.08222](https://arxiv.org/abs/2403.08222)

    论文考虑了在既有真实专家又有对抗性专家的情况下的二元决策聚合问题，提出了设计鲁棒聚合器以最小化遗憾的方法，并证明了当真实专家是对称的且对抗性专家不太多时，截尾均值是最优的。

    

    我们考虑了在既有真实专家又有对抗性专家的情况下的二元决策聚合问题。真实专家将会如实报告他们的私人信号，并获得适当的激励，而对抗性专家可以任意报告。决策者需要设计一个鲁棒的聚合器，根据专家的报告来预测世界的真实状态。决策者不了解具体的信息结构，即信号、状态以及对抗性专家的策略的联合分布。我们希望找到在最坏信息结构下最小化遗憾的最优聚合器。遗憾被定义为聚合器和一个基准之间的期望损失差，该基准根据联合分布和真实专家的报告做出最优决策。我们证明了当真实专家是对称的且对抗性专家不太多时，截尾均值是最优的。

    arXiv:2403.08222v1 Announce Type: cross  Abstract: We consider a binary decision aggregation problem in the presence of both truthful and adversarial experts. The truthful experts will report their private signals truthfully with proper incentive, while the adversarial experts can report arbitrarily. The decision maker needs to design a robust aggregator to forecast the true state of the world based on the reports of experts. The decision maker does not know the specific information structure, which is a joint distribution of signals, states, and strategies of adversarial experts. We want to find the optimal aggregator minimizing regret under the worst information structure. The regret is defined by the difference in expected loss between the aggregator and a benchmark who makes the optimal decision given the joint distribution and reports of truthful experts.   We prove that when the truthful experts are symmetric and adversarial experts are not too numerous, the truncated mean is opt
    
[^5]: 使用核函数的快速遍历搜索

    Fast Ergodic Search with Kernel Functions

    [https://arxiv.org/abs/2403.01536](https://arxiv.org/abs/2403.01536)

    提出了一种使用核函数的快速遍历搜索方法，其在搜索空间维度上具有线性复杂度，可以推广到李群，并且通过数值测试展示比现有算法快两个数量级。

    

    遍历搜索使得对信息分布进行最佳探索成为可能，同时保证了对搜索空间的渐近覆盖。然而，当前的方法通常在搜索空间维度上具有指数计算复杂度，并且局限于欧几里得空间。我们引入了一种计算高效的遍历搜索方法。我们的贡献是双重的。首先，我们开发了基于核的遍历度量，并将其从欧几里得空间推广到李群上。我们正式证明了所建议的度量与标准遍历度量一致，同时保证了在搜索空间维度上具有线性复杂度。其次，我们推导了非线性系统的核遍历度量的一阶最优性条件，这使得轨迹优化变得更加高效。全面的数值基准测试表明，所提出的方法至少比现有最先进的算法快两个数量级。

    arXiv:2403.01536v1 Announce Type: cross  Abstract: Ergodic search enables optimal exploration of an information distribution while guaranteeing the asymptotic coverage of the search space. However, current methods typically have exponential computation complexity in the search space dimension and are restricted to Euclidean space. We introduce a computationally efficient ergodic search method. Our contributions are two-fold. First, we develop a kernel-based ergodic metric and generalize it from Euclidean space to Lie groups. We formally prove the proposed metric is consistent with the standard ergodic metric while guaranteeing linear complexity in the search space dimension. Secondly, we derive the first-order optimality condition of the kernel ergodic metric for nonlinear systems, which enables efficient trajectory optimization. Comprehensive numerical benchmarks show that the proposed method is at least two orders of magnitude faster than the state-of-the-art algorithm. Finally, we d
    
[^6]: EBBS: 一个具有双层束搜索的集成方法用于零翻译机器翻译

    EBBS: An Ensemble with Bi-Level Beam Search for Zero-Shot Machine Translation

    [https://arxiv.org/abs/2403.00144](https://arxiv.org/abs/2403.00144)

    提出了一种集成方法EBBS，配合新颖的双层束搜索算法，能够优于直接和通过第三语言进行的翻译，并实现知识蒸馏来提高推理效率。

    

    当我们用特定的翻译方向训练多语言模型时，零翻译的能力就会出现；模型可以直接在未见过的方向进行翻译。另外，零翻译也可以通过第三种语言（例如英语）来实现。在我们的工作中，我们发现直接和通过第三种语言进行的翻译都存在噪音，并且表现不尽如人意。我们提出了EBBS，一个具有新颖的双层束搜索算法的集成方法，其中每个集成组件在下层逐步探索自己的预测，但它们通过上层的“软投票”机制进行同步。在两个流行的多语言翻译数据集上的结果表明，EBBS始终优于直接和通过第三种语言进行的翻译，以及现有的集成技术。此外，我们可以将集成的知识传回到多语言模型中，以提高推理效率；值得注意的是，我们的E

    arXiv:2403.00144v1 Announce Type: cross  Abstract: The ability of zero-shot translation emerges when we train a multilingual model with certain translation directions; the model can then directly translate in unseen directions. Alternatively, zero-shot translation can be accomplished by pivoting through a third language (e.g., English). In our work, we observe that both direct and pivot translations are noisy and achieve less satisfactory performance. We propose EBBS, an ensemble method with a novel bi-level beam search algorithm, where each ensemble component explores its own prediction step by step at the lower level but they are synchronized by a "soft voting" mechanism at the upper level. Results on two popular multilingual translation datasets show that EBBS consistently outperforms direct and pivot translations as well as existing ensemble techniques. Further, we can distill the ensemble's knowledge back to the multilingual model to improve inference efficiency; profoundly, our E
    
[^7]: 分而治之：用多目标强化学习可靠地揭示帕累托前沿

    Divide and Conquer: Provably Unveiling the Pareto Front with Multi-Objective Reinforcement Learning

    [https://arxiv.org/abs/2402.07182](https://arxiv.org/abs/2402.07182)

    这项研究介绍了一个名为IPRO的算法，利用分解任务为一系列单目标问题方法，可可靠地揭示多目标强化学习中实现最优表现的策略的帕累托前沿，同时提供收敛保证和未发现解的距离上限。

    

    在多目标强化学习中，获取在不同偏好下实现最优表现的策略的帕累托前沿是一个重大挑战。我们引入了迭代帕累托参考优化（IPRO），这是一个原则性算法，它将找到帕累托前沿的任务分解成一系列具有各种解决方法的单目标问题。这使我们能够在每个步骤中建立收敛保证并提供未发现帕累托最优解的距离上限。实证评估表明，IPRO能够与需要额外领域知识的方法相匹配或优于它们。通过利用问题特定的单目标求解器，我们的方法也有望在多目标强化学习之外的应用中发挥作用，比如路径规划和优化问题。

    A significant challenge in multi-objective reinforcement learning is obtaining a Pareto front of policies that attain optimal performance under different preferences. We introduce Iterated Pareto Referent Optimisation (IPRO), a principled algorithm that decomposes the task of finding the Pareto front into a sequence of single-objective problems for which various solution methods exist. This enables us to establish convergence guarantees while providing an upper bound on the distance to undiscovered Pareto optimal solutions at each step. Empirical evaluations demonstrate that IPRO matches or outperforms methods that require additional domain knowledge. By leveraging problem-specific single-objective solvers, our approach also holds promise for applications beyond multi-objective reinforcement learning, such as in pathfinding and optimisation.
    
[^8]: 关于马尔可夫链中心极限定理的收敛速度，及其在TD学习中的应用

    Rates of Convergence in the Central Limit Theorem for Markov Chains, with an Application to TD Learning

    [https://arxiv.org/abs/2401.15719](https://arxiv.org/abs/2401.15719)

    本研究证明了一个非渐近的中心极限定理，并通过应用于TD学习，展示了其实际应用的可行性。

    

    我们使用Stein方法证明了一个非渐近的、矢量值鞅差的中心极限定理，并利用泊松方程将结果推广到马尔可夫链的函数。然后我们展示了这些结果可以应用于建立基于平均的非渐近的TD学习的中心极限定理。

    We prove a non-asymptotic central limit theorem for vector-valued martingale differences using Stein's method, and use Poisson's equation to extend the result to functions of Markov Chains. We then show that these results can be applied to establish a non-asymptotic central limit theorem for Temporal Difference (TD) learning with averaging.
    
[^9]: 表达建模对于离线强化学习不足：可处理的推理角度

    Expressive Modeling Is Insufficient for Offline RL: A Tractable Inference Perspective. (arXiv:2311.00094v1 [cs.LG])

    [http://arxiv.org/abs/2311.00094](http://arxiv.org/abs/2311.00094)

    本文指出，在离线强化学习任务中，除了表达性强的序列模型，可处理性也起着重要的作用。由于离线数据收集策略和环境动态的随机性，需要精确且高效地回答各种概率查询，以找到有奖励的动作。基于此，本文提出了Trifle（离线强化学习的可处理推理）方法，利用现代可处理概率模型来解决这个问题。

    

    离线强化学习任务中，一种流行的范例是先将离线轨迹拟合到一个序列模型中，然后通过该模型提示高期望回报的动作。虽然普遍认为表达性更强的序列模型可以带来更好的性能，但本文强调了可处理性，即精确而高效地回答各种概率查询的能力，同样起着重要的作用。具体而言，由于离线数据收集策略和环境动态带来的基本随机性，需要进行高度非平凡的条件/约束生成，以引出有奖励的动作。虽然仍然可以近似处理这些查询，但我们观察到这种粗糙的估计显著削弱了表达性强的序列模型带来的好处。为了解决这个问题，本文提出了Trifle（离线强化学习的可处理推理），它利用了现代可处理概率模型（TPM）来弥合这个差距。

    A popular paradigm for offline Reinforcement Learning (RL) tasks is to first fit the offline trajectories to a sequence model, and then prompt the model for actions that lead to high expected return. While a common consensus is that more expressive sequence models imply better performance, this paper highlights that tractability, the ability to exactly and efficiently answer various probabilistic queries, plays an equally important role. Specifically, due to the fundamental stochasticity from the offline data-collection policies and the environment dynamics, highly non-trivial conditional/constrained generation is required to elicit rewarding actions. While it is still possible to approximate such queries, we observe that such crude estimates significantly undermine the benefits brought by expressive sequence models. To overcome this problem, this paper proposes Trifle (Tractable Inference for Offline RL), which leverages modern Tractable Probabilistic Models (TPMs) to bridge the gap b
    
[^10]: MimicTouch: 使用多模态触觉反馈学习人类的控制策略

    MimicTouch: Learning Human's Control Strategy with Multi-Modal Tactile Feedback. (arXiv:2310.16917v1 [cs.RO])

    [http://arxiv.org/abs/2310.16917](http://arxiv.org/abs/2310.16917)

    MimicTouch是一种新的框架，能够模仿人类的触觉引导控制策略，通过收集来自人类示范者的多模态触觉数据集，来学习并执行复杂任务。

    

    在机器人技术和人工智能领域，触觉处理的整合变得越来越重要，特别是在学习执行像对准和插入这样复杂任务时。然而，现有研究主要依赖机器人遥操作数据和强化学习，忽视了人类受触觉反馈引导下的控制策略所提供的丰富见解。为了利用人类感觉，现有的从人类学习的方法主要利用视觉反馈，常常忽视了人类本能地利用触觉反馈完成复杂操作的宝贵经验。为了填补这一空白，我们引入了一种新框架"MimicTouch"，模仿人类的触觉引导控制策略。在这个框架中，我们首先从人类示范者那里收集多模态触觉数据集，包括人类触觉引导的控制策略来完成任务。接下来的步骤涉及指令的传递，其中机器人通过模仿人类的触觉引导策略来执行任务。

    In robotics and artificial intelligence, the integration of tactile processing is becoming increasingly pivotal, especially in learning to execute intricate tasks like alignment and insertion. However, existing works focusing on tactile methods for insertion tasks predominantly rely on robot teleoperation data and reinforcement learning, which do not utilize the rich insights provided by human's control strategy guided by tactile feedback. For utilizing human sensations, methodologies related to learning from humans predominantly leverage visual feedback, often overlooking the invaluable tactile feedback that humans inherently employ to finish complex manipulations. Addressing this gap, we introduce "MimicTouch", a novel framework that mimics human's tactile-guided control strategy. In this framework, we initially collect multi-modal tactile datasets from human demonstrators, incorporating human tactile-guided control strategies for task completion. The subsequent step involves instruc
    
[^11]: 层级反馈传播

    Layer-wise Feedback Propagation. (arXiv:2308.12053v1 [cs.LG])

    [http://arxiv.org/abs/2308.12053](http://arxiv.org/abs/2308.12053)

    本文提出了一种名为“层级反馈传播（LFP）”的新型神经网络预测器训练方法，通过利用可解释性细化与层级相关性传播（LRP）相结合，根据每个连接对任务的贡献分配奖励，该方法克服了传统梯度下降方法存在的问题。对于各种模型和数据集，LFP取得了与梯度下降相当的性能。

    

    本文提出了一种称为“层级反馈传播（LFP）”的新型神经网络预测器训练方法，该方法利用可解释性，具体而言是层级相关性传播（LRP），根据每个连接对解决给定任务的贡献独立分配奖励。这与传统的梯度下降方法不同，梯度下降方法是朝向估计的损失最小值更新参数。LFP在模型中传播奖励信号，而无需梯度计算。它增强接收到正反馈的结构，同时降低接收到负反馈的结构的影响。我们从理论和实证的角度证明了LFP的收敛性，并展示了它在各种模型和数据集上实现与梯度下降相当的性能。值得注意的是，LFP克服了梯度方法的某些局限性，例如对有意义的导数的依赖。我们进一步研究了LFP如何解决梯度方法相关问题的限制。

    In this paper, we present Layer-wise Feedback Propagation (LFP), a novel training approach for neural-network-like predictors that utilizes explainability, specifically Layer-wise Relevance Propagation(LRP), to assign rewards to individual connections based on their respective contributions to solving a given task. This differs from traditional gradient descent, which updates parameters towards anestimated loss minimum. LFP distributes a reward signal throughout the model without the need for gradient computations. It then strengthens structures that receive positive feedback while reducingthe influence of structures that receive negative feedback. We establish the convergence of LFP theoretically and empirically, and demonstrate its effectiveness in achieving comparable performance to gradient descent on various models and datasets. Notably, LFP overcomes certain limitations associated with gradient-based methods, such as reliance on meaningful derivatives. We further investigate how 
    
[^12]: 利用替代特征选择找到最优的多样特征集

    Finding Optimal Diverse Feature Sets with Alternative Feature Selection. (arXiv:2307.11607v1 [cs.LG])

    [http://arxiv.org/abs/2307.11607](http://arxiv.org/abs/2307.11607)

    本文引入了替代特征选择的概念，将其形式化为优化问题，并通过约束定义了替代特征集，使用户可以控制替代的数量和差异性。我们证明了该问题的NP-hard性，并讨论了如何将传统特征选择方法作为目标集成。实验证明替代特征集确实可以具有高预测质量，同时分析了几个影响因素。

    

    特征选择是获取小型、可解释且高精度预测模型的一种常见方法。传统的特征选择方法通常只能得到一个特征集，这在某些场景下可能不足够。例如，用户可能对寻找具有相似预测质量但提供不同数据解释的替代特征集感兴趣。在本文中，我们引入了替代特征选择，并将其形式化为一个优化问题。特别地，我们通过约束定义了替代特征，并使用户可以控制替代的数量和差异性。接下来，我们分析了这个优化问题的复杂性并展示了其NP-hard性质。进一步地，我们讨论了如何将传统的特征选择方法作为目标集成。最后，我们使用30个分类数据集评估了替代特征选择的效果。我们观察到替代特征集确实可能具有较高的预测质量，并分析了几个影响这一结果的因素。

    Feature selection is popular for obtaining small, interpretable, yet highly accurate prediction models. Conventional feature-selection methods typically yield one feature set only, which might not suffice in some scenarios. For example, users might be interested in finding alternative feature sets with similar prediction quality, offering different explanations of the data. In this article, we introduce alternative feature selection and formalize it as an optimization problem. In particular, we define alternatives via constraints and enable users to control the number and dissimilarity of alternatives. Next, we analyze the complexity of this optimization problem and show NP-hardness. Further, we discuss how to integrate conventional feature-selection methods as objectives. Finally, we evaluate alternative feature selection with 30 classification datasets. We observe that alternative feature sets may indeed have high prediction quality, and we analyze several factors influencing this ou
    
[^13]: 神经网络下的执行预测

    Performative Prediction with Neural Networks. (arXiv:2304.06879v1 [cs.LG])

    [http://arxiv.org/abs/2304.06879](http://arxiv.org/abs/2304.06879)

    本文提出了执行预测的框架，通过找到具有执行稳定性的分类器来适用于数据分布。通过假设数据分布相对于模型的预测值可Lipschitz连续，使得我们能够放宽对损失函数的假设要求。

    

    执行预测是一种学习模型并影响其预测数据的框架。本文旨在找到分类器，使其具有执行稳定性，即适用于其产生的数据分布的最佳分类器。在使用重复风险最小化方法找到具有执行稳定性的分类器的标准收敛结果中，假设数据分布对于模型参数是可Lipschitz连续的。在这种情况下，损失必须对这些参数强凸和平滑；否则，该方法将在某些问题上发散。然而本文则假设数据分布是相对于模型的预测值可Lipschitz连续的，这是执行系统的更加自然的假设。结果，我们能够显著放宽对损失函数的假设要求。作为一个说明，我们介绍了一种建模真实数据分布的重采样过程，并使用其来实证执行稳定性相对于其他目标的效益。

    Performative prediction is a framework for learning models that influence the data they intend to predict. We focus on finding classifiers that are performatively stable, i.e. optimal for the data distribution they induce. Standard convergence results for finding a performatively stable classifier with the method of repeated risk minimization assume that the data distribution is Lipschitz continuous to the model's parameters. Under this assumption, the loss must be strongly convex and smooth in these parameters; otherwise, the method will diverge for some problems. In this work, we instead assume that the data distribution is Lipschitz continuous with respect to the model's predictions, a more natural assumption for performative systems. As a result, we are able to significantly relax the assumptions on the loss function. In particular, we do not need to assume convexity with respect to the model's parameters. As an illustration, we introduce a resampling procedure that models realisti
    
[^14]: 分类规则生成：可扩展性，解释性和公平性

    Rule Generation for Classification: Scalability, Interpretability, and Fairness. (arXiv:2104.10751v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2104.10751](http://arxiv.org/abs/2104.10751)

    这项研究介绍了一种新的基于规则的分类优化方法，利用列生成线性规划实现可扩展性，并通过分配成本系数和引入额外约束解决了解释性和公平性问题。该方法在局部解释性和公平性之间取得了良好的平衡。

    

    我们引入了一种新的基于规则的分类优化方法，具有约束条件。所提出的方法利用列生成线性规划，因此可扩展到大型数据集。所得定价子问题被证明是NP难问题。我们采用基于决策树的启发式方法，并解决了一个代理定价子问题以加速。该方法返回一组规则以及它们的最优权重，指示每个规则对学习的重要性。我们通过为规则分配成本系数和引入额外约束来解决解释性和公平性问题。具体而言，我们关注局部解释性，并将公平性的一般分离准则推广到多个敏感属性和类别。我们在一系列数据集上测试了所提出方法的性能，并提供了一个案例研究来详细阐述其不同方面。所提出的基于规则的学习方法在局部解释性和公平性之间达到了良好的平衡点。

    We introduce a new rule-based optimization method for classification with constraints. The proposed method leverages column generation for linear programming, and hence, is scalable to large datasets. The resulting pricing subproblem is shown to be NP-Hard. We recourse to a decision tree-based heuristic and solve a proxy pricing subproblem for acceleration. The method returns a set of rules along with their optimal weights indicating the importance of each rule for learning. We address interpretability and fairness by assigning cost coefficients to the rules and introducing additional constraints. In particular, we focus on local interpretability and generalize separation criterion in fairness to multiple sensitive attributes and classes. We test the performance of the proposed methodology on a collection of datasets and present a case study to elaborate on its different aspects. The proposed rule-based learning method exhibits a good compromise between local interpretability and fairn
    

