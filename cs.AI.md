# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models](https://arxiv.org/abs/2403.19647) | 该论文介绍了一种新方法，即稀疏特征电路，可以在语言模型中发现和编辑可解释的因果图，为我们提供了对未预料机制的详细理解和包含了用于提高分类器泛化能力的SHIFT方法。 |
| [^2] | [Debiased Offline Representation Learning for Fast Online Adaptation in Non-stationary Dynamics](https://arxiv.org/abs/2402.11317) | 提出了一种名为DORA的新方法，通过信息瓶颈原理在离线设置中学习适应性策略，解决了动态编码与环境数据之间的互信息与与行为策略的互信息之间的难题 |
| [^3] | [Counterfactual Influence in Markov Decision Processes](https://arxiv.org/abs/2402.08514) | 马尔可夫决策过程中的反事实推理问题是一个基本问题，我们提出了一种算法来构建反事实模型，并通过比较反事实和干预分布来对影响进行形式化的特征化。 |
| [^4] | [Online POMDP Planning with Anytime Deterministic Guarantees.](http://arxiv.org/abs/2310.01791) | 本文中，我们推导出在线POMDP规划中一个简化解决方案与理论上最优解之间的确定性关系，以解决目前近似算法只能提供概率性和通常呈现渐进性保证的限制。 |
| [^5] | [Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures.](http://arxiv.org/abs/2307.15220) | 通过观看手术视频讲座，我们提出了一种新方法，SurgVLP，通过利用手术视频讲座中的语音和视觉信息进行多模态表示学习，并解决了手术相关语言挑战。 |

# 详细

[^1]: 稀疏特征电路：在语言模型中发现和编辑可解释的因果图

    Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models

    [https://arxiv.org/abs/2403.19647](https://arxiv.org/abs/2403.19647)

    该论文介绍了一种新方法，即稀疏特征电路，可以在语言模型中发现和编辑可解释的因果图，为我们提供了对未预料机制的详细理解和包含了用于提高分类器泛化能力的SHIFT方法。

    

    我们介绍了用于发现和应用稀疏特征电路的方法。这些电路是人类可解释特征的因果相关子网络，用于解释语言模型行为。 在先前的工作中确定的电路由多义且难以解释的单元组成，例如注意力头或神经元，使它们不适用于许多下游应用。 相比之下，稀疏特征电路实现了对未预料机制的详细理解。 由于它们基于细粒度单元，稀疏特征电路对下游任务非常有用：我们 introduc了SHIFT，通过切除人类判断为任务不相关的特征，从而提高分类器的泛化能力。 最后，我们通过发现成千上万个稀疏特征电路来展示一个完全无监督且可扩展的可解释性管线，用于自动发现的模型行为。

    arXiv:2403.19647v1 Announce Type: cross  Abstract: We introduce methods for discovering and applying sparse feature circuits. These are causally implicated subnetworks of human-interpretable features for explaining language model behaviors. Circuits identified in prior work consist of polysemantic and difficult-to-interpret units like attention heads or neurons, rendering them unsuitable for many downstream applications. In contrast, sparse feature circuits enable detailed understanding of unanticipated mechanisms. Because they are based on fine-grained units, sparse feature circuits are useful for downstream tasks: We introduce SHIFT, where we improve the generalization of a classifier by ablating features that a human judges to be task-irrelevant. Finally, we demonstrate an entirely unsupervised and scalable interpretability pipeline by discovering thousands of sparse feature circuits for automatically discovered model behaviors.
    
[^2]: 针对非静态动态的快速在线调整的去偏置离线表示学习

    Debiased Offline Representation Learning for Fast Online Adaptation in Non-stationary Dynamics

    [https://arxiv.org/abs/2402.11317](https://arxiv.org/abs/2402.11317)

    提出了一种名为DORA的新方法，通过信息瓶颈原理在离线设置中学习适应性策略，解决了动态编码与环境数据之间的互信息与与行为策略的互信息之间的难题

    

    开发能够适应非静态环境的策略对于现实世界的强化学习应用至关重要。然而，在仅有一组有限的预先收集的轨迹的离线设置中学习这种适应性策略存在显著挑战。为了解决这个问题，我们引入了一种名为去偏置离线表示快速在线调整（DORA）的新方法。DORA融入了信息瓶颈原理，最大化了动态编码与环境数据之间的互信息，同时最小化了动态编码与行为策略的互信息。我们提出了DORA的实际实现，利用

    arXiv:2402.11317v1 Announce Type: cross  Abstract: Developing policies that can adjust to non-stationary environments is essential for real-world reinforcement learning applications. However, learning such adaptable policies in offline settings, with only a limited set of pre-collected trajectories, presents significant challenges. A key difficulty arises because the limited offline data makes it hard for the context encoder to differentiate between changes in the environment dynamics and shifts in the behavior policy, often leading to context misassociations. To address this issue, we introduce a novel approach called Debiased Offline Representation for fast online Adaptation (DORA). DORA incorporates an information bottleneck principle that maximizes mutual information between the dynamics encoding and the environmental data, while minimizing mutual information between the dynamics encoding and the actions of the behavior policy. We present a practical implementation of DORA, leverag
    
[^3]: 马尔可夫决策过程中的反事实影响

    Counterfactual Influence in Markov Decision Processes

    [https://arxiv.org/abs/2402.08514](https://arxiv.org/abs/2402.08514)

    马尔可夫决策过程中的反事实推理问题是一个基本问题，我们提出了一种算法来构建反事实模型，并通过比较反事实和干预分布来对影响进行形式化的特征化。

    

    我们的工作解决了马尔可夫决策过程（MDPs）中反事实推理的一个基本问题。给定一个MDP路径τ，这种推理允许我们得出描述τ的反事实路径τ'，描述了在与τ中观察到的动作序列不同的情况下，τ的“如果是这种情况”的版本。然而，由于反事实的状态和动作随时间发生偏离，观察τ可能不再影响反事实世界，这意味着分析不再针对个体观察结果，而是产生干预性结果而非反事实结果。尽管这个问题特别影响流行的Gumbel-max结构因果模型，这种模型用于MDP反事实，但直到现在一直被忽视。在这项工作中，我们引入了一个基于比较反事实和干预分布的影响的形式特征化。我们设计了一个算法来构建反事实模型。

    Our work addresses a fundamental problem in the context of counterfactual inference for Markov Decision Processes (MDPs). Given an MDP path $\tau$, this kind of inference allows us to derive counterfactual paths $\tau'$ describing what-if versions of $\tau$ obtained under different action sequences than those observed in $\tau$. However, as the counterfactual states and actions deviate from the observed ones over time, the observation $\tau$ may no longer influence the counterfactual world, meaning that the analysis is no longer tailored to the individual observation, resulting in interventional outcomes rather than counterfactual ones. Even though this issue specifically affects the popular Gumbel-max structural causal model used for MDP counterfactuals, it has remained overlooked until now. In this work, we introduce a formal characterisation of influence based on comparing counterfactual and interventional distributions. We devise an algorithm to construct counterfactual models that
    
[^4]: 具有任意确定性保证的在线POMDP规划

    Online POMDP Planning with Anytime Deterministic Guarantees. (arXiv:2310.01791v1 [cs.AI])

    [http://arxiv.org/abs/2310.01791](http://arxiv.org/abs/2310.01791)

    本文中，我们推导出在线POMDP规划中一个简化解决方案与理论上最优解之间的确定性关系，以解决目前近似算法只能提供概率性和通常呈现渐进性保证的限制。

    

    在现实场景中，自主智能体经常遇到不确定性并基于不完整信息做出决策。在不确定性下的规划可以使用部分可观察的马尔科夫决策过程（POMDP）进行数学建模。然而，寻找POMDP的最优规划在计算上是昂贵的，只有在小规模任务中可行。近年来，近似算法（如树搜索和基于采样的方法）已经成为解决较大问题的先进POMDP求解器。尽管这些算法有效，但它们仅提供概率性和通常呈现渐进性保证，这是由于它们依赖于采样的缘故。为了解决这些限制，我们推导出一个简化解决方案与理论上最优解之间的确定性关系。首先，我们推导出选择一组观测以在计算每个后验节点时分支的边界。

    Autonomous agents operating in real-world scenarios frequently encounter uncertainty and make decisions based on incomplete information. Planning under uncertainty can be mathematically formalized using partially observable Markov decision processes (POMDPs). However, finding an optimal plan for POMDPs can be computationally expensive and is feasible only for small tasks. In recent years, approximate algorithms, such as tree search and sample-based methodologies, have emerged as state-of-the-art POMDP solvers for larger problems. Despite their effectiveness, these algorithms offer only probabilistic and often asymptotic guarantees toward the optimal solution due to their dependence on sampling. To address these limitations, we derive a deterministic relationship between a simplified solution that is easier to obtain and the theoretically optimal one. First, we derive bounds for selecting a subset of the observations to branch from while computing a complete belief at each posterior nod
    
[^5]: 通过观看数百个手术视频讲座学习多模态表示

    Learning Multi-modal Representations by Watching Hundreds of Surgical Video Lectures. (arXiv:2307.15220v1 [cs.CV])

    [http://arxiv.org/abs/2307.15220](http://arxiv.org/abs/2307.15220)

    通过观看手术视频讲座，我们提出了一种新方法，SurgVLP，通过利用手术视频讲座中的语音和视觉信息进行多模态表示学习，并解决了手术相关语言挑战。

    

    最近在外科计算机视觉应用方面的进展主要依靠完全监督方法，主要使用视觉数据。这些方法依赖于手动注释的手术视频来预测一组固定的对象类别，限制了它们在未见手术程序和后续任务上的通用性。在这项工作中，我们提出了一个观点，即通过开放的手术电子学习平台提供的手术视频讲座可以为多模态表示学习提供有效的监督信号，而无需依赖手动注释。我们通过使用多个互补的自动语音识别系统生成文本转录来解决手术视频讲座中存在的手术相关语言挑战。然后，我们提出了一种新的方法，SurgVLP - 手术视觉语言预训练，用于多模态表示学习。SurgVLP构建了一种新的对比学习目标，将视频剪辑嵌入与相应的文本嵌入对齐。

    Recent advancements in surgical computer vision applications have been driven by fully-supervised methods, primarily using only visual data. These methods rely on manually annotated surgical videos to predict a fixed set of object categories, limiting their generalizability to unseen surgical procedures and downstream tasks. In this work, we put forward the idea that the surgical video lectures available through open surgical e-learning platforms can provide effective supervisory signals for multi-modal representation learning without relying on manual annotations. We address the surgery-specific linguistic challenges present in surgical video lectures by employing multiple complementary automatic speech recognition systems to generate text transcriptions. We then present a novel method, SurgVLP - Surgical Vision Language Pre-training, for multi-modal representation learning. SurgVLP constructs a new contrastive learning objective to align video clip embeddings with the corresponding m
    

