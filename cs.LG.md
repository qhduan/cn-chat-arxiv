# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Heavy-Ball Q-Learning with Residual Weighting Correction](https://arxiv.org/abs/2606.27112) | 本文提出了一种带残差加权校正的重球Q学习方法，通过切换线性系统视角证明了其收敛性和加速效果，并扩展到了线性函数逼近场景。 |
| [^2] | [RolloutPipe: Overlapping Pipelined Rollout and Training in Disaggregated On-Policy LLM Reinforcement Learning](https://arxiv.org/abs/2606.26997) | 该论文提出了RolloutPipe框架，通过将固定权重的推出过程转化为完整的组流水线，实现了推出与训练在分离式RLVR系统中的高效重叠，解决了同步系统的GPU空闲问题和异步系统的数据过时问题。 |
| [^3] | [Multipath Adaptive Gated Bottleneck Latent ODE with Raman Data Fusion for Cell Culture Process Forecasting](https://arxiv.org/abs/2606.26520) | 提出了一种结合门控瓶颈潜在常微分方程与多路径即时微调的自适应框架，通过变量级门控和掩码感知瓶颈机制有效处理高维稀疏数据，实现了对细胞培养过程的多日早期预测。 |
| [^4] | [Autodata: An agentic data scientist to create high quality synthetic data](https://arxiv.org/abs/2606.25996) | 本文提出了一种名为Autodata的通用方法，通过训练AI智能体作为自主数据科学家，并对其进行元优化，从而在多个任务上生成比传统方法更高质量的合成数据，显著提升模型性能。 |
| [^5] | [CausalChaos! Dataset for Comprehensive Causal Action Question Answering Over Longer Causal Chains Grounded in Dynamic Visual Scenes](https://arxiv.org/abs/2404.01299) | 利用卡通图像构建的CausalChaos!数据集，包含更长因果链的因果问答，通过动态互动和视觉展示挑战性因果关系，为模型提供了更多具挑战性且明确定义的因果关系。 |
| [^6] | [Graph Unitary Message Passing](https://arxiv.org/abs/2403.11199) | 提出了一种名为GUMP的图单元消息传递方法，通过应用单元邻接矩阵来缓解图神经网络中的过度压缩问题。 |
| [^7] | [Replicability is Asymptotically Free in Multi-armed Bandits](https://arxiv.org/abs/2402.07391) | 本论文研究在多臂赌博机问题中，通过引入探索-再确定算法和连续淘汰算法，以及谨慎选择置信区间的幅度，实现了可复制性，并证明了当时间界足够大时，可复制算法的额外代价是不必要的。 |
| [^8] | [Debiasing and a local analysis for population clustering using semidefinite programming.](http://arxiv.org/abs/2401.10927) | 本文研究了使用半正定规划进行人群聚类的问题，并提出了计算高效的算法。这些算法可以根据小样本数据的原始种群将数据分为两组，适用于种群之间差异较小的情况。 |
| [^9] | [Learning to Visually Connect Actions and their Effects.](http://arxiv.org/abs/2401.10805) | 该论文提出了视觉连接动作和其效果的概念（CATE），用于视频理解。研究表明，不同的任务形式产生了捕捉直观动作特性的表示，但模型表现不佳，人类的表现明显优于它们。该研究为未来的努力奠定了基础，并希望能激发出高级形式和模型的灵感。 |
| [^10] | [Proportionally Representative Clustering.](http://arxiv.org/abs/2304.13917) | 本文提出了一个新的公平性准则——比例代表性公平性（PRF），并设计了有效的算法满足该准则。 |
| [^11] | [Distribution-free Deviation Bounds of Learning via Model Selection with Cross-validation Risk Estimation.](http://arxiv.org/abs/2303.08777) | 本文提出通过模型选择和交叉验证风险估计来学习的一般方法，并建立了无分布偏差界，比经验风险最小化方法更紧密，在一些情况下表现更优。 |

# 详细

[^1]: 带残差加权校正的重球Q学习

    Heavy-Ball Q-Learning with Residual Weighting Correction

    [https://arxiv.org/abs/2606.27112](https://arxiv.org/abs/2606.27112)

    本文提出了一种带残差加权校正的重球Q学习方法，通过切换线性系统视角证明了其收敛性和加速效果，并扩展到了线性函数逼近场景。

    

    本文提出了一种用于强化学习的校正重球Q学习方法，并证明了其收敛性。同时，文章识别了该方法在理论上保证比标准Q学习收敛更快的条件。随后，相同的构造被扩展到线性函数逼近的Q学习中，并推导出了类似的收敛性和加速结论。该分析基于Q学习算法的切换线性系统表示以及相关切换族的联合谱半径。这种切换线性系统视角在标准Q学习分析中并不常用，它为理解重球动量如何加速Q学习提供了补充框架和新的见解。

    arXiv:2606.27112v1 Announce Type: cross  Abstract: This paper proposes a corrected heavy-ball Q-learning method for reinforcement learning (RL) and establishes its convergence. It also identifies conditions under which the method is theoretically guaranteed to converge faster than standard Q-learning. The same construction is then extended to Q-learning with linear function approximation, where analogous convergence and acceleration statements are derived. The analysis is based on a switched linear system (SLS) representation of Q-learning algorithms and on the joint spectral radius (JSR) of the associated switching families. This SLS viewpoint is not commonly used in standard analyses of Q-learning, and it provides a complementary framework and new insight into how heavy-ball momentum can accelerate Q-learning.
    
[^2]: RolloutPipe：在分离式在线策略大语言模型强化学习中重叠流水线式推出与训练

    RolloutPipe: Overlapping Pipelined Rollout and Training in Disaggregated On-Policy LLM Reinforcement Learning

    [https://arxiv.org/abs/2606.26997](https://arxiv.org/abs/2606.26997)

    该论文提出了RolloutPipe框架，通过将固定权重的推出过程转化为完整的组流水线，实现了推出与训练在分离式RLVR系统中的高效重叠，解决了同步系统的GPU空闲问题和异步系统的数据过时问题。

    

    arXiv:2606.26997v1 公告类型：交叉 摘要：大语言模型（LLM）在推理方面的后训练越来越依赖于可验证奖励的强化学习（RLVR），模型通过数学、逻辑和科学任务中的真实反馈进行学习。为了实现灵活的资源分配并支持异构训练设置，现代RLVR系统采用分离式架构，将推出生成与策略训练解耦到独立的GPU池中。然而，现有的同步在线策略GRPO（组相对策略优化）RLVR系统在开始训练前完成整个推出过程，导致推出仍在进行时训练器GPU池处于空闲状态。异步RL管道重叠了这两个阶段，但代价是使用过时数据进行训练。为应对这些挑战，我们提出了RolloutPipe，一种用于分离式RLVR系统的后训练框架，它将固定权重的推出转变为完整的组流水线，其中可训练组...

    arXiv:2606.26997v1 Announce Type: cross  Abstract: Large language model (LLM) post-training for reasoning increasingly relies on reinforcement learning with verifiable rewards (RLVR), where models learn from ground-truth feedback on mathematical, logical, and scientific tasks. To enable flexible resource allocation and support heterogeneous training setups, modern RLVR systems adopt disaggregated architectures that decouple rollout generation and policy training across independent GPU pools. However, existing synchronous on-policy GRPO (Group Relative Policy Optimization) RLVR systems finish an entire rollout before starting training, leaving the trainer GPU pool idle while rollout is still ongoing. Asynchronous RL pipelines overlap the two stages, but at the cost of training on stale data. To address these challenges, we propose RolloutPipe, a post-training framework for disaggregated RLVR systems, which turns the fixed-weight rollout into a complete-group pipeline where trainable gro
    
[^3]: 基于多路径自适应门控瓶颈潜在常微分方程与拉曼数据融合的细胞培养过程预测方法

    Multipath Adaptive Gated Bottleneck Latent ODE with Raman Data Fusion for Cell Culture Process Forecasting

    [https://arxiv.org/abs/2606.26520](https://arxiv.org/abs/2606.26520)

    提出了一种结合门控瓶颈潜在常微分方程与多路径即时微调的自适应框架，通过变量级门控和掩码感知瓶颈机制有效处理高维稀疏数据，实现了对细胞培养过程的多日早期预测。

    

    哺乳动物细胞培养过程是许多生物制药生产的基础，但保持过程稳定运行十分困难：关键工艺参数会随时间漂移，而偏离规格的趋势往往在确认时已为时过晚，无法及时干预。早期的多日预测能够实现对补料、采样和控制的及时调整，但生物过程预测面临诸多挑战：测量数据稀疏且采样时间不规则，不同细胞系和培养基的操作条件存在异质性，且初始行为几乎相同的运行过程可能走向截然不同的未来结果。为此，我们提出了一种自适应框架，将门控瓶颈潜在常微分方程与多路径即时微调相结合。门控瓶颈潜在常微分方程通过可学习的变量级门控和掩码感知瓶颈机制对标准潜在常微分方程进行增强，能够压缩高维稀疏输入，从而在有限数据条件下提升学习效果。

    arXiv:2606.26520v1 Announce Type: cross  Abstract: Mammalian cell-culture processes underpin the manufacture of many biopharmaceuticals, yet keeping a run on track is hard: critical process parameters drift over days, and an off-specification trend is often confirmed too late to intervene. Early-stage, multi-day forecasts could enable timely adjustment of feeding, sampling, and control, but bioprocess forecasting is challenging because measurements are sparse and irregularly sampled, operating conditions are heterogeneous across cell lines and media, and runs with near-identical early behaviour can diverge into different futures. We propose an adaptive framework combining a Gated Bottleneck Latent Ordinary Differential Equation (GB-Latent ODE) with Multi-Path Just-In-Time Fine Tuning (MP-JIT-FT). The GB-Latent ODE augments the stan dard Latent ODE with learnable variable-wise gating and a mask-aware bottleneck that compress high-dimensional sparse inputs, improving learning under limit
    
[^4]: Autodata：一个用于创建高质量合成数据的自主数据科学家

    Autodata: An agentic data scientist to create high quality synthetic data

    [https://arxiv.org/abs/2606.25996](https://arxiv.org/abs/2606.25996)

    本文提出了一种名为Autodata的通用方法，通过训练AI智能体作为自主数据科学家，并对其进行元优化，从而在多个任务上生成比传统方法更高质量的合成数据，显著提升模型性能。

    

    arXiv:2606.25996v2 公告类型：替换 摘要：我们介绍了Autodata，一种通用方法，使AI智能体能够充当数据科学家，构建高质量的训练和评估数据。我们展示了如何训练（元优化）这样一个数据科学家智能体，使其学会创建更强大的数据。我们描述了总体框架以及一个具体的实践实现——自主自我指令（Agentic Self-Instruct）。我们在计算机科学研究任务、法律推理任务和数学对象推理任务上进行了实验，与经典的合成数据集创建方法相比，我们获得了更好的结果。此外，对数据科学家智能体本身进行元优化带来了更大的性能提升。自主数据创建提供了一种将增加的推理计算转化为更高质量模型训练的方法。总体而言，我们相信这一方向有潜力改变我们构建AI数据的方式。

    arXiv:2606.25996v2 Announce Type: replace  Abstract: We introduce Autodata, a general method that enables AI agents to act as data scientists who build high quality training and evaluation data. We show how to train (meta-optimize) such a data scientist agent, so that it learns to create even stronger data. We describe the overall formulation, and a specific practical implementation, Agentic Self-Instruct. We conduct experiments on computer science research tasks, legal reasoning tasks and reasoning with mathematical objects, where we obtain improved results compared to classical synthetic dataset creation methods. Further, meta-optimizing the data scientist agent itself delivers an even larger performance uplift. Agentic data creation provides a way to convert increased inference compute into higher quality model training. Overall, we believe this direction has the potential to change the way we build AI data.
    
[^5]: CausalChaos!数据集：基于动态视觉场景中更长因果链的全面因果行动问答

    CausalChaos! Dataset for Comprehensive Causal Action Question Answering Over Longer Causal Chains Grounded in Dynamic Visual Scenes

    [https://arxiv.org/abs/2404.01299](https://arxiv.org/abs/2404.01299)

    利用卡通图像构建的CausalChaos!数据集，包含更长因果链的因果问答，通过动态互动和视觉展示挑战性因果关系，为模型提供了更多具挑战性且明确定义的因果关系。

    

    因果视频问答（QA）越来越受到关注，然而现有数据集在因果推理分析方面往往缺乏深度。为了填补这一空白，我们利用卡通的独特属性构建了CausalChaos!，这是一个新颖且具有挑战性的因果问答（Why-QA）数据集，基于标志性的“猫和老鼠”卡通系列。我们的数据集通过周到的问题和多层次答案，包含着嵌入动态互动和视觉中的更长因果链，同时动画原理允许动画师创造定义明确、明了的因果关系。这些因素使模型能够解决更具挑战性但明确定义的因果关系。我们还引入了硬负采样，包括CausalConfusion版本。虽然模型表现良好，但仍有很大改进空间，特别是在开放式答案方面。我们确定了更为先进/明确的因果关系建模和联合建模等改进方向。

    arXiv:2404.01299v1 Announce Type: cross  Abstract: Causal video question answering (QA) has garnered increasing interest, yet existing datasets often lack depth in causal reasoning analysis. To address this gap, we capitalize on the unique properties of cartoons and construct CausalChaos!, a novel, challenging causal Why-QA dataset built upon the iconic "Tom and Jerry" cartoon series. With thoughtful questions and multi-level answers, our dataset contains much longer causal chains embedded in dynamic interactions and visuals, at the same time principles of animation allows animators to create well-defined, unambiguous causal relationships. These factors allow models to solve more challenging, yet well-defined causal relationships. We also introduce hard negative mining, including CausalConfusion version. While models perform well, there is much room for improvement, especially, on open-ended answers. We identify more advanced/explicit causal relationship modeling and joint modeling of 
    
[^6]: 图单元消息传递

    Graph Unitary Message Passing

    [https://arxiv.org/abs/2403.11199](https://arxiv.org/abs/2403.11199)

    提出了一种名为GUMP的图单元消息传递方法，通过应用单元邻接矩阵来缓解图神经网络中的过度压缩问题。

    

    消息传递机制是图神经网络在各种应用中取得成功的原因，但也带来了过度压缩的问题。最近的研究通过改善图谱的重连技术、破坏图中的结构偏见来抵制过度压缩，然而在过度压缩度量方面对过度压缩的改进有所限制。受到单元RNN的启发，我们提出了图单元消息传递（GUMP），通过应用单元邻接矩阵进行消息传递来缓解图神经网络中的过度压缩问题。为设计GUMP，首先提出了一种转换方法，使普通图具有单元邻接矩阵并保持其结构偏差。然后，通过利用单元邻接矩阵的固有结构实现单位化投影算法获得单元邻接矩阵，并允许GUMP是置换等变的。实验结果表明了GUMP在改善各种应用任务上性能的有效性。

    arXiv:2403.11199v1 Announce Type: cross  Abstract: Message passing mechanism contributes to the success of GNNs in various applications, but also brings the oversquashing problem. Recent works combat oversquashing by improving the graph spectrums with rewiring techniques, disrupting the structural bias in graphs, and having limited improvement on oversquashing in terms of oversquashing measure. Motivated by unitary RNN, we propose Graph Unitary Message Passing (GUMP) to alleviate oversquashing in GNNs by applying unitary adjacency matrix for message passing. To design GUMP, a transformation is first proposed to make general graphs have unitary adjacency matrix and keep its structural bias. Then, unitary adjacency matrix is obtained with a unitary projection algorithm, which is implemented by utilizing the intrinsic structure of unitary adjacency matrix and allows GUMP to be permutation-equivariant. Experimental results show the effectiveness of GUMP in improving the performance on vari
    
[^7]: 在多臂赌博机中，可复制性渐进自由

    Replicability is Asymptotically Free in Multi-armed Bandits

    [https://arxiv.org/abs/2402.07391](https://arxiv.org/abs/2402.07391)

    本论文研究在多臂赌博机问题中，通过引入探索-再确定算法和连续淘汰算法，以及谨慎选择置信区间的幅度，实现了可复制性，并证明了当时间界足够大时，可复制算法的额外代价是不必要的。

    

    本研究受可复制的机器学习需求的推动，研究了随机多臂赌博机问题。特别地，我们考虑了一个可复制算法，确保算法的操作序列不受数据集固有随机性的影响。我们观察到，现有算法所需的遗憾值比不可复制算法多$O(1/\rho^2)$倍，其中$\rho$是非复制程度。然而，我们证明了当给定的$\rho$下时间界$T$足够大时，此额外代价是不必要的，前提是谨慎选择置信区间的幅度。我们引入了一个先探索后决策的算法，在决策之前均匀选择动作。此外，我们还研究了一个连续淘汰算法，在每个阶段结束时淘汰次优动作。为了确保这些算法的可复制性，我们将随机性引入决策制定中。

    This work is motivated by the growing demand for reproducible machine learning. We study the stochastic multi-armed bandit problem. In particular, we consider a replicable algorithm that ensures, with high probability, that the algorithm's sequence of actions is not affected by the randomness inherent in the dataset. We observe that existing algorithms require $O(1/\rho^2)$ times more regret than nonreplicable algorithms, where $\rho$ is the level of nonreplication. However, we demonstrate that this additional cost is unnecessary when the time horizon $T$ is sufficiently large for a given $\rho$, provided that the magnitude of the confidence bounds is chosen carefully. We introduce an explore-then-commit algorithm that draws arms uniformly before committing to a single arm. Additionally, we examine a successive elimination algorithm that eliminates suboptimal arms at the end of each phase. To ensure the replicability of these algorithms, we incorporate randomness into their decision-ma
    
[^8]: 使用半正定规划的去偏和局部分析进行人群聚类

    Debiasing and a local analysis for population clustering using semidefinite programming. (arXiv:2401.10927v1 [stat.ML])

    [http://arxiv.org/abs/2401.10927](http://arxiv.org/abs/2401.10927)

    本文研究了使用半正定规划进行人群聚类的问题，并提出了计算高效的算法。这些算法可以根据小样本数据的原始种群将数据分为两组，适用于种群之间差异较小的情况。

    

    本文考虑了从混合的2个次高斯分布中抽取的小数据样本的分区问题。我们分析了同一作者提出的计算高效的算法，将数据根据其原始种群大致分为两组，给定一个小样本。本文的研究动机是将个体根据其原始种群使用p个标记进行聚类，当任意两个种群之间的差异很小时。我们基于整数二次规划的半正定松弛形式构建，该规划问题本质上是在一个图上找到最大割，其中割中的边权重表示基于它们的p个特征的两个节点之间的不相似度得分。我们用Δ^2:=pγ来表示两个中心（均值向量）之间的ℓ_2^2距离，即μ^(1), μ^(2)∈ℝ^p。目标是在交换精度和计算效率之间提供全面的权衡。

    In this paper, we consider the problem of partitioning a small data sample of size $n$ drawn from a mixture of $2$ sub-gaussian distributions. In particular, we analyze computational efficient algorithms proposed by the same author, to partition data into two groups approximately according to their population of origin given a small sample. This work is motivated by the application of clustering individuals according to their population of origin using $p$ markers, when the divergence between any two of the populations is small. We build upon the semidefinite relaxation of an integer quadratic program that is formulated essentially as finding the maximum cut on a graph, where edge weights in the cut represent dissimilarity scores between two nodes based on their $p$ features. Here we use $\Delta^2 :=p \gamma$ to denote the $\ell_2^2$ distance between two centers (mean vectors), namely, $\mu^{(1)}$, $\mu^{(2)}$ $\in$ $\mathbb{R}^p$. The goal is to allow a full range of tradeoffs between
    
[^9]: 学习视觉连接动作和其效果

    Learning to Visually Connect Actions and their Effects. (arXiv:2401.10805v1 [cs.CV])

    [http://arxiv.org/abs/2401.10805](http://arxiv.org/abs/2401.10805)

    该论文提出了视觉连接动作和其效果的概念（CATE），用于视频理解。研究表明，不同的任务形式产生了捕捉直观动作特性的表示，但模型表现不佳，人类的表现明显优于它们。该研究为未来的努力奠定了基础，并希望能激发出高级形式和模型的灵感。

    

    在这项工作中，我们引入了视觉连接动作和其效果（CATE）的新概念，用于视频理解。CATE可以在任务规划和从示范中学习等领域中应用。我们提出了不同基于CATE的任务形式，如动作选择和动作指定，其中视频理解模型以语义和细粒度的方式连接动作和效果。我们观察到不同的形式产生了捕捉直观动作特性的表示。我们还设计了各种基线模型用于动作选择和动作指定。尽管任务具有直观性，但我们观察到模型困难重重，人类表现明显优于它们。本研究旨在为未来的努力奠定基础，展示了连接视频理解中动作和效果的灵活性和多功能性，希望能激发出高级形式和模型的灵感。

    In this work, we introduce the novel concept of visually Connecting Actions and Their Effects (CATE) in video understanding. CATE can have applications in areas like task planning and learning from demonstration. We propose different CATE-based task formulations, such as action selection and action specification, where video understanding models connect actions and effects at semantic and fine-grained levels. We observe that different formulations produce representations capturing intuitive action properties. We also design various baseline models for action selection and action specification. Despite the intuitive nature of the task, we observe that models struggle, and humans outperform them by a large margin. The study aims to establish a foundation for future efforts, showcasing the flexibility and versatility of connecting actions and effects in video understanding, with the hope of inspiring advanced formulations and models.
    
[^10]: 比例代表性聚类

    Proportionally Representative Clustering. (arXiv:2304.13917v1 [cs.LG])

    [http://arxiv.org/abs/2304.13917](http://arxiv.org/abs/2304.13917)

    本文提出了一个新的公平性准则——比例代表性公平性（PRF），并设计了有效的算法满足该准则。

    

    近年来，机器学习领域对公平概念的形式化表述越来越受关注。本文关注于聚类，是无监督机器学习中最基础的任务之一。我们提出了一个新的公平性准则——比例代表性公平性（PRF），我们认为该概念以一种更有说服力的方式达到了文献中几个现存概念的理由。但现有的公平聚类算法不能满足我们的公平性概念。我们设计了高效的算法，以满足无约束聚类和离散聚类问题的PRF。

    In recent years, there has been a surge in effort to formalize notions of fairness in machine learning. We focus on clustering -- one of the fundamental tasks in unsupervised machine learning. We propose a new axiom that captures proportional representation fairness (PRF). We make a case that the concept achieves the raison d'{\^{e}}tre of several existing concepts in the literature in an arguably more convincing manner. Our fairness concept is not satisfied by existing fair clustering algorithms. We design efficient algorithms to achieve PRF both for unconstrained and discrete clustering problems.
    
[^11]: 模型选择配合交叉验证风险估计的无分布偏差界学习方法

    Distribution-free Deviation Bounds of Learning via Model Selection with Cross-validation Risk Estimation. (arXiv:2303.08777v1 [stat.ML])

    [http://arxiv.org/abs/2303.08777](http://arxiv.org/abs/2303.08777)

    本文提出通过模型选择和交叉验证风险估计来学习的一般方法，并建立了无分布偏差界，比经验风险最小化方法更紧密，在一些情况下表现更优。

    

    交叉验证方法的风险估计和模型选择在统计学和机器学习中得到了广泛应用。然而，学习通过模型选择与交叉验证风险估计的理论性质的理解在其广泛使用面前相当缺乏。在这个背景下，本文将学习通过模型选择与交叉验证风险估计作为一种经典统计学习理论中的一般系统学习框架，并建立了基于VC维的无分布偏差边界，给出了结果的详细证明，并考虑了有界和无界的损失函数。我们还推导出在整个假设空间中，学习通过模型选择的偏差界比通过经验风险最小化学习的偏差界更紧密的条件，支持在一些情况下经验上观察到的模型选择框架的更好性能。

    Cross-validation techniques for risk estimation and model selection are widely used in statistics and machine learning. However, the understanding of the theoretical properties of learning via model selection with cross-validation risk estimation is quite low in face of its widespread use. In this context, this paper presents learning via model selection with cross-validation risk estimation as a general systematic learning framework within classical statistical learning theory and establishes distribution-free deviation bounds in terms of VC dimension, giving detailed proofs of the results and considering both bounded and unbounded loss functions. We also deduce conditions under which the deviation bounds of learning via model selection are tighter than that of learning via empirical risk minimization in the whole hypotheses space, supporting the better performance of model selection frameworks observed empirically in some instances.
    

