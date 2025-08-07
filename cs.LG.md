# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adaptive Coded Federated Learning: Privacy Preservation and Straggler Mitigation](https://arxiv.org/abs/2403.14905) | ACFL提出了一种新的自适应编码联邦学习方法，通过在训练之前采用个性化的数据上传到中央服务器来生成全局编码数据集，以解决原有固定权重生成全局编码数据集时可能导致学习性能下降的问题。 |
| [^2] | [Probabilistic Actor-Critic: Learning to Explore with PAC-Bayes Uncertainty](https://arxiv.org/abs/2402.03055) | 概率演员-评论家算法（PAC）通过在评论家中建模和推断不确定性，以改进强化学习中的连续控制性能，并实现自适应的探索策略。 |
| [^3] | [Distributional Reinforcement Learning with Online Risk-awareness Adaption](https://arxiv.org/abs/2310.05179) | 本论文提出了一个新的分布式强化学习框架，可以通过在线风险适应性调整来量化不确定性，并动态选择认知风险水平。 |
| [^4] | [NACHOS: Neural Architecture Search for Hardware Constrained Early Exit Neural Networks.](http://arxiv.org/abs/2401.13330) | NACHOS 是一种面向硬件受限的早期退出神经网络的神经架构搜索方法，可以自动化设计早期退出神经网络并考虑骨干和早期退出分类器之间的关系。 |
| [^5] | [Thompson Exploration with Best Challenger Rule in Best Arm Identification.](http://arxiv.org/abs/2310.00539) | 本文提出了一种新的策略，将Thompson采样与最佳候选规则相结合，用于解决最佳臂识别问题。该策略在渐近情况下是最优的，并在一般的多臂赌博机问题中达到接近最优的性能。 |
| [^6] | [Symmetry & Critical Points for Symmetric Tensor Decompositions Problems.](http://arxiv.org/abs/2306.07886) | 本文研究了将一个实对称张量分解成秩为1项之和的非凸优化问题，得到了精确的分析估计，并发现了各种阻碍局部优化方法的几何障碍和由于对称性导致的丰富的临界点集合。 |

# 详细

[^1]: 自适应编码联邦学习：隐私保护与慢节点缓解

    Adaptive Coded Federated Learning: Privacy Preservation and Straggler Mitigation

    [https://arxiv.org/abs/2403.14905](https://arxiv.org/abs/2403.14905)

    ACFL提出了一种新的自适应编码联邦学习方法，通过在训练之前采用个性化的数据上传到中央服务器来生成全局编码数据集，以解决原有固定权重生成全局编码数据集时可能导致学习性能下降的问题。

    

    在本文中，我们讨论了在存在慢节点情况下的联邦学习问题。针对这一问题，我们提出了一种编码联邦学习框架，其中中央服务器聚合来自非慢节点的梯度和来自隐私保护全局编码数据集的梯度，以减轻慢节点的负面影响。然而，在聚合这些梯度时，固定权重在迭代中一直被应用，忽略了全局编码数据集的生成过程以及训练模型随着迭代的动态性。这一疏漏可能导致学习性能下降。为克服这一缺陷，我们提出了一种名为自适应编码联邦学习（ACFL）的新方法。在ACFL中，在训练之前，每个设备向中央服务器上传一个带有附加噪声的编码本地数据集，以生成符合隐私保护要求的全局编码数据集。在...

    arXiv:2403.14905v1 Announce Type: cross  Abstract: In this article, we address the problem of federated learning in the presence of stragglers. For this problem, a coded federated learning framework has been proposed, where the central server aggregates gradients received from the non-stragglers and gradient computed from a privacy-preservation global coded dataset to mitigate the negative impact of the stragglers. However, when aggregating these gradients, fixed weights are consistently applied across iterations, neglecting the generation process of the global coded dataset and the dynamic nature of the trained model over iterations. This oversight may result in diminished learning performance. To overcome this drawback, we propose a new method named adaptive coded federated learning (ACFL). In ACFL, before the training, each device uploads a coded local dataset with additive noise to the central server to generate a global coded dataset under privacy preservation requirements. During
    
[^2]: 概率演员-评论家：学习以PAC-Bayes不确定性进行探索

    Probabilistic Actor-Critic: Learning to Explore with PAC-Bayes Uncertainty

    [https://arxiv.org/abs/2402.03055](https://arxiv.org/abs/2402.03055)

    概率演员-评论家算法（PAC）通过在评论家中建模和推断不确定性，以改进强化学习中的连续控制性能，并实现自适应的探索策略。

    

    我们引入了概率演员-评论家（PAC），这是一种新颖的强化学习算法，通过缓解探索与利用的平衡问题，改进了连续控制性能。PAC通过将随机策略和评论家无缝融合，创建了评论家不确定性估计和演员训练之间的动态协同作用。我们的PAC算法的关键贡献在于通过Probably Approximately Correct-Bayesian（PAC-Bayes）分析，明确建模和推断评论家的认知不确定性。这种对评论家不确定性的融入使PAC能够在学习过程中自适应调整其探索策略，指导演员的决策过程。与现有技术中的固定或预定的探索方案相比，PAC表现出更好的效果。通过PAC-Bayes分析引导的随机策略和评论家之间的协同作用，是向深度强化学习中更具自适应性和有效性的探索策略迈出的关键一步。

    We introduce Probabilistic Actor-Critic (PAC), a novel reinforcement learning algorithm with improved continuous control performance thanks to its ability to mitigate the exploration-exploitation trade-off. PAC achieves this by seamlessly integrating stochastic policies and critics, creating a dynamic synergy between the estimation of critic uncertainty and actor training. The key contribution of our PAC algorithm is that it explicitly models and infers epistemic uncertainty in the critic through Probably Approximately Correct-Bayesian (PAC-Bayes) analysis. This incorporation of critic uncertainty enables PAC to adapt its exploration strategy as it learns, guiding the actor's decision-making process. PAC compares favorably against fixed or pre-scheduled exploration schemes of the prior art. The synergy between stochastic policies and critics, guided by PAC-Bayes analysis, represents a fundamental step towards a more adaptive and effective exploration strategy in deep reinforcement lear
    
[^3]: 具有在线风险感知适应性的分布式强化学习

    Distributional Reinforcement Learning with Online Risk-awareness Adaption

    [https://arxiv.org/abs/2310.05179](https://arxiv.org/abs/2310.05179)

    本论文提出了一个新的分布式强化学习框架，可以通过在线风险适应性调整来量化不确定性，并动态选择认知风险水平。

    

    在实际应用中使用强化学习（RL）需要考虑次优结果，这取决于代理人对不确定环境的熟悉程度。本文介绍了一个新的框架，Distributional RL with Online Risk Adaption（DRL-ORA），可以综合量化不确定性并动态选择认知风险水平，通过在线解决总变差最小化问题。风险水平选择可以通过使用Follow-The-Leader类型算法进行网格搜索来有效实现。

    arXiv:2310.05179v2 Announce Type: replace  Abstract: The use of reinforcement learning (RL) in practical applications requires considering sub-optimal outcomes, which depend on the agent's familiarity with the uncertain environment. Dynamically adjusting the level of epistemic risk over the course of learning can tactically achieve reliable optimal policy in safety-critical environments and tackle the sub-optimality of a static risk level. In this work, we introduce a novel framework, Distributional RL with Online Risk Adaption (DRL-ORA), which can quantify the aleatory and epistemic uncertainties compositely and dynamically select the epistemic risk levels via solving a total variation minimization problem online. The risk level selection can be efficiently achieved through grid search using a Follow-The-Leader type algorithm, and its offline oracle is related to "satisficing measure" (in the decision analysis community) under a special modification of the loss function. We show multi
    
[^4]: NACHOS: 硬件受限的早期退出神经网络的神经架构搜索

    NACHOS: Neural Architecture Search for Hardware Constrained Early Exit Neural Networks. (arXiv:2401.13330v1 [cs.LG])

    [http://arxiv.org/abs/2401.13330](http://arxiv.org/abs/2401.13330)

    NACHOS 是一种面向硬件受限的早期退出神经网络的神经架构搜索方法，可以自动化设计早期退出神经网络并考虑骨干和早期退出分类器之间的关系。

    

    早期退出神经网络（EENNs）为标准的深度神经网络（DNN）配备早期退出分类器（EECs），在处理的中间点上提供足够的分类置信度时进行预测。这在效果和效率方面带来了许多好处。目前，EENNs的设计是由专家手动完成的，这是一项复杂和耗时的任务，需要考虑许多方面，包括正确的放置、阈值设置和EECs的计算开销。因此，研究正在探索使用神经架构搜索（NAS）自动化设计EENNs。目前，文献中提出了几个完整的NAS解决方案用于EENNs，并且一个完全自动化的综合设计策略，同时考虑骨干和EECs仍然是一个未解决的问题。为此，本研究呈现了面向硬件受限的早期退出神经网络的神经架构搜索（NACHOS）。

    Early Exit Neural Networks (EENNs) endow astandard Deep Neural Network (DNN) with Early Exit Classifiers (EECs), to provide predictions at intermediate points of the processing when enough confidence in classification is achieved. This leads to many benefits in terms of effectiveness and efficiency. Currently, the design of EENNs is carried out manually by experts, a complex and time-consuming task that requires accounting for many aspects, including the correct placement, the thresholding, and the computational overhead of the EECs. For this reason, the research is exploring the use of Neural Architecture Search (NAS) to automatize the design of EENNs. Currently, few comprehensive NAS solutions for EENNs have been proposed in the literature, and a fully automated, joint design strategy taking into consideration both the backbone and the EECs remains an open problem. To this end, this work presents Neural Architecture Search for Hardware Constrained Early Exit Neural Networks (NACHOS),
    
[^5]: 最佳候选规则下的Thompson探索在最佳臂识别中的应用

    Thompson Exploration with Best Challenger Rule in Best Arm Identification. (arXiv:2310.00539v1 [stat.ML])

    [http://arxiv.org/abs/2310.00539](http://arxiv.org/abs/2310.00539)

    本文提出了一种新的策略，将Thompson采样与最佳候选规则相结合，用于解决最佳臂识别问题。该策略在渐近情况下是最优的，并在一般的多臂赌博机问题中达到接近最优的性能。

    

    本文研究了在经典单参数指数模型下，固定置信度下的最佳臂识别（BAI）问题。针对这个问题，目前已有很多策略被提出，但大多数需要在每一轮解决一个最优化问题和/或者需要探索一个臂至少一定次数，除非是针对高斯模型的限制。为了解决这些限制，我们提出了一种新的策略，将Thompson采样与一个计算效率高的方法——最佳候选规则相结合。虽然Thompson采样最初被考虑用于最大化累积奖励，但我们证明它也可以自然地用于在BAI中探索臂而不强迫最大化奖励。我们证明了我们的策略在任意两臂赌博机问题上是渐近最优的，并且在一般的$K$臂赌博机问题上（$K\geq 3$）达到接近最优的性能。然而，在数值实验中，我们的策略与现有方法相比表现出了竞争性的性能。

    This paper studies the fixed-confidence best arm identification (BAI) problem in the bandit framework in the canonical single-parameter exponential models. For this problem, many policies have been proposed, but most of them require solving an optimization problem at every round and/or are forced to explore an arm at least a certain number of times except those restricted to the Gaussian model. To address these limitations, we propose a novel policy that combines Thompson sampling with a computationally efficient approach known as the best challenger rule. While Thompson sampling was originally considered for maximizing the cumulative reward, we demonstrate that it can be used to naturally explore arms in BAI without forcing it. We show that our policy is asymptotically optimal for any two-armed bandit problems and achieves near optimality for general $K$-armed bandit problems for $K\geq 3$. Nevertheless, in numerical experiments, our policy shows competitive performance compared to as
    
[^6]: 对称张量分解问题的对称性与临界点

    Symmetry & Critical Points for Symmetric Tensor Decompositions Problems. (arXiv:2306.07886v1 [math.OC])

    [http://arxiv.org/abs/2306.07886](http://arxiv.org/abs/2306.07886)

    本文研究了将一个实对称张量分解成秩为1项之和的非凸优化问题，得到了精确的分析估计，并发现了各种阻碍局部优化方法的几何障碍和由于对称性导致的丰富的临界点集合。

    

    本文考虑了将一个实对称张量分解成秩为1项之和的非凸优化问题。利用其丰富的对称结构，导出Puiseux级数表示的一系列临界点，并获得了关于临界值和Hessian谱的精确分析估计。这些结果揭示了各种几何障碍，阻碍了局部优化方法的使用，最后，利用一个牛顿多面体论证了固定对称性的所有临界点的完全枚举，并证明了与全局最小值的集合相比，由于对称性的存在，临界点的集合可能会显示出组合的丰富性。

    We consider the non-convex optimization problem associated with the decomposition of a real symmetric tensor into a sum of rank one terms. Use is made of the rich symmetry structure to derive Puiseux series representations of families of critical points, and so obtain precise analytic estimates on the critical values and the Hessian spectrum. The sharp results make possible an analytic characterization of various geometric obstructions to local optimization methods, revealing in particular a complex array of saddles and local minima which differ by their symmetry, structure and analytic properties. A desirable phenomenon, occurring for all critical points considered, concerns the index of a point, i.e., the number of negative Hessian eigenvalues, increasing with the value of the objective function. Lastly, a Newton polytope argument is used to give a complete enumeration of all critical points of fixed symmetry, and it is shown that contrarily to the set of global minima which remains 
    

