# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Dyna-LfLH: Learning Agile Navigation in Dynamic Environments from Learned Hallucination](https://arxiv.org/abs/2403.17231) | 提出了一种新的自监督学习方法Dyna-LfLH，通过学习幻觉中的动态环境，安全地学习地面机器人在动态环境中灵活导航。 |
| [^2] | [Towards Efficient Risk-Sensitive Policy Gradient: An Iteration Complexity Analysis](https://arxiv.org/abs/2403.08955) | 本文对风险敏感策略梯度方法进行了迭代复杂度分析，发现其能够通过使用指数效用函数达到较低的迭代复杂度。 |
| [^3] | [Combining Evidence Across Filtrations](https://arxiv.org/abs/2402.09698) | 这篇论文研究了合并使用不同过滤器计算的e进程的方法，探讨了其在顺序推理中的应用。 |
| [^4] | [Adversarial Robustness Through Artifact Design](https://arxiv.org/abs/2402.04660) | 该研究提出了一种通过艺术设计实现对抗性鲁棒性的方法，通过微小更改现有规范来抵御对抗性示例的影响。 |
| [^5] | [Graph Elimination Networks.](http://arxiv.org/abs/2401.01233) | 本文提出了图消除网络（GENs），其通过消除邻域传播过程中的冗余来解决图神经网络（GNN）在深层次上性能下降的问题。GENs可以增强节点对远距离邻域的感知，并扩展网络传播的深度。 |
| [^6] | [ODTlearn: A Package for Learning Optimal Decision Trees for Prediction and Prescription.](http://arxiv.org/abs/2307.15691) | ODTlearn是一个开源的Python包，用于学习预测和处方的最优决策树。它提供了多种优化方法，并支持各种问题和算法的扩展。 |
| [^7] | [A Flexible Framework for Incorporating Patient Preferences Into Q-Learning.](http://arxiv.org/abs/2307.12022) | 这个论文提出了一种称为潜在效用Q学习的方法，能够将患者偏好纳入复合结果的动态治疗方案中，解决了传统方法对时间点和结果数量的限制，能够实现强大的性能。 |
| [^8] | [K-Tensors: Clustering Positive Semi-Definite Matrices.](http://arxiv.org/abs/2306.06534) | 本文介绍了一种针对正半定矩阵的自一致性聚类算法（K-张量），通过考虑其特征结构，能够有效地将正半定矩阵进行分区。 |
| [^9] | [Sampling, Diffusions, and Stochastic Localization.](http://arxiv.org/abs/2305.10690) | 这篇论文介绍了扩散和随机定位的关系，证明了标准去噪扩散是一种随机定位，并提出了一种在对数步骤内从 Ising 模型的 Gibbs 测度中进行采样的算法。 |
| [^10] | [Heterogeneous Directed Hypergraph Neural Network over abstract syntax tree (AST) for Code Classification.](http://arxiv.org/abs/2305.04228) | 本研究提出了使用异构有向超图表示AST，并使用异构有向超图神经网络处理图形进行代码分类，超过了现有方法。 |
| [^11] | [Memory-adaptive Depth-wise Heterogenous Federated Learning.](http://arxiv.org/abs/2303.04887) | 这项研究介绍了一种名为FeDepth的内存自适应深度学习解决方案，它根据每个客户端的内存预算将完整模型自适应地分解成块，并依次训练这些块，以解决联邦学习中异构设备的内存限制问题。 |

# 详细

[^1]: Dyna-LfLH:从学到的幻觉中学会在动态环境中学习灵活导航

    Dyna-LfLH: Learning Agile Navigation in Dynamic Environments from Learned Hallucination

    [https://arxiv.org/abs/2403.17231](https://arxiv.org/abs/2403.17231)

    提出了一种新的自监督学习方法Dyna-LfLH，通过学习幻觉中的动态环境，安全地学习地面机器人在动态环境中灵活导航。

    

    这篇论文提出了一种自监督学习方法，用于安全地学习地面机器人的运动规划器，以在密集且动态的障碍物环境中导航。针对高度混乱、快速移动、难以预测的障碍物，传统的运动规划器可能无法跟上有限的机载计算。对于基于学习的规划器，很难获取高质量的演示以进行模仿学习，同时强化学习在探索过程中由于高碰撞概率而效率低下。为了安全有效地提供训练数据，LfH方法基于过去成功的导航经验在相对简单或完全开放的环境中综合困难的导航环境，但遗憾的是无法解决动态障碍物问题。在我们的新方法Dyna-LfLH中，我们设计并学习了一种新颖的潜在分布和样本。

    arXiv:2403.17231v1 Announce Type: cross  Abstract: This paper presents a self-supervised learning method to safely learn a motion planner for ground robots to navigate environments with dense and dynamic obstacles. When facing highly-cluttered, fast-moving, hard-to-predict obstacles, classical motion planners may not be able to keep up with limited onboard computation. For learning-based planners, high-quality demonstrations are difficult to acquire for imitation learning while reinforcement learning becomes inefficient due to the high probability of collision during exploration. To safely and efficiently provide training data, the Learning from Hallucination (LfH) approaches synthesize difficult navigation environments based on past successful navigation experiences in relatively easy or completely open ones, but unfortunately cannot address dynamic obstacles. In our new Dynamic Learning from Learned Hallucination (Dyna-LfLH), we design and learn a novel latent distribution and sample
    
[^2]: 朝向高效的风险敏感策略梯度：一个迭代复杂度分析

    Towards Efficient Risk-Sensitive Policy Gradient: An Iteration Complexity Analysis

    [https://arxiv.org/abs/2403.08955](https://arxiv.org/abs/2403.08955)

    本文对风险敏感策略梯度方法进行了迭代复杂度分析，发现其能够通过使用指数效用函数达到较低的迭代复杂度。

    

    强化学习在各种应用中表现出色，使得自主智能体能够通过与环境的互动学习最佳策略。然而，传统的强化学习框架在迭代复杂度和鲁棒性方面经常面临挑战。风险敏感强化学习平衡了期望回报和风险，具有产生概率鲁棒策略的潜力，但其迭代复杂度分析尚未得到充分探讨。在本研究中，我们针对风险敏感策略梯度方法进行了彻底的迭代复杂度分析，重点关注REINFORCE算法并采用指数效用函数。我们获得了一个$\mathcal{O}(\epsilon^{-2})$的迭代复杂度，以达到$\epsilon$-近似的一阶稳定点（FOSP）。我们研究了风险敏感算法是否可以比风险中性算法实现更好的迭代复杂度。

    arXiv:2403.08955v1 Announce Type: cross  Abstract: Reinforcement Learning (RL) has shown exceptional performance across various applications, enabling autonomous agents to learn optimal policies through interaction with their environments. However, traditional RL frameworks often face challenges in terms of iteration complexity and robustness. Risk-sensitive RL, which balances expected return and risk, has been explored for its potential to yield probabilistically robust policies, yet its iteration complexity analysis remains underexplored. In this study, we conduct a thorough iteration complexity analysis for the risk-sensitive policy gradient method, focusing on the REINFORCE algorithm and employing the exponential utility function. We obtain an iteration complexity of $\mathcal{O}(\epsilon^{-2})$ to reach an $\epsilon$-approximate first-order stationary point (FOSP). We investigate whether risk-sensitive algorithms can achieve better iteration complexity compared to their risk-neutr
    
[^3]: 合并不同过滤器中的证据

    Combining Evidence Across Filtrations

    [https://arxiv.org/abs/2402.09698](https://arxiv.org/abs/2402.09698)

    这篇论文研究了合并使用不同过滤器计算的e进程的方法，探讨了其在顺序推理中的应用。

    

    在任何时刻有效的顺序推理中，已知任何可接受的推理方法必须基于测试鞅和它们的组合广义化，称为e进程，它们是非负进程，其在任何任意停时的期望上界不超过一。e进程量化了针对复合零假设的一系列结果的累积证据。本文研究了使用不同信息集（即过滤器）计算的e进程的合并方法，针对一个零假设。尽管在相同过滤器上构建的e进程可以轻松地合并（例如，通过平均），但在不同过滤器上构建的e进程不能那么容易地合并，因为它们在较粗的过滤器中的有效性不能转换为在更细的过滤器中的有效性。我们讨论了文献中三个具体例子：可交换性测试，独立性测试等。

    arXiv:2402.09698v1 Announce Type: cross  Abstract: In anytime-valid sequential inference, it is known that any admissible inference procedure must be based on test martingales and their composite generalization, called e-processes, which are nonnegative processes whose expectation at any arbitrary stopping time is upper-bounded by one. An e-process quantifies the accumulated evidence against a composite null hypothesis over a sequence of outcomes. This paper studies methods for combining e-processes that are computed using different information sets, i.e., filtrations, for a null hypothesis. Even though e-processes constructed on the same filtration can be combined effortlessly (e.g., by averaging), e-processes constructed on different filtrations cannot be combined as easily because their validity in a coarser filtration does not translate to validity in a finer filtration. We discuss three concrete examples of such e-processes in the literature: exchangeability tests, independence te
    
[^4]: 通过艺术设计提高对抗性鲁棒性

    Adversarial Robustness Through Artifact Design

    [https://arxiv.org/abs/2402.04660](https://arxiv.org/abs/2402.04660)

    该研究提出了一种通过艺术设计实现对抗性鲁棒性的方法，通过微小更改现有规范来抵御对抗性示例的影响。

    

    对抗性示例的出现给机器学习带来了挑战。为了阻碍对抗性示例，大多数防御方法都改变了模型的训练方式（如对抗性训练）或推理过程（如随机平滑）。尽管这些方法显著提高了模型的对抗性鲁棒性，但模型仍然极易受到对抗性示例的影响。在某些领域如交通标志识别中，我们发现对象是按照规范来设计（如标志规范）。为了改善对抗性鲁棒性，我们提出了一种新颖的方法。具体来说，我们提供了一种重新定义规范的方法，对现有规范进行微小的更改，以防御对抗性示例。我们将艺术设计问题建模为一个鲁棒优化问题，并提出了基于梯度和贪婪搜索的方法来解决它。我们在交通标志识别领域对我们的方法进行了评估，使其能够改变交通标志中的象形图标（即标志内的符号）。

    Adversarial examples arose as a challenge for machine learning. To hinder them, most defenses alter how models are trained (e.g., adversarial training) or inference is made (e.g., randomized smoothing). Still, while these approaches markedly improve models' adversarial robustness, models remain highly susceptible to adversarial examples. Identifying that, in certain domains such as traffic-sign recognition, objects are implemented per standards specifying how artifacts (e.g., signs) should be designed, we propose a novel approach for improving adversarial robustness. Specifically, we offer a method to redefine standards, making minor changes to existing ones, to defend against adversarial examples. We formulate the problem of artifact design as a robust optimization problem, and propose gradient-based and greedy search methods to solve it. We evaluated our approach in the domain of traffic-sign recognition, allowing it to alter traffic-sign pictograms (i.e., symbols within the signs) a
    
[^5]: 图消除网络

    Graph Elimination Networks. (arXiv:2401.01233v1 [cs.LG])

    [http://arxiv.org/abs/2401.01233](http://arxiv.org/abs/2401.01233)

    本文提出了图消除网络（GENs），其通过消除邻域传播过程中的冗余来解决图神经网络（GNN）在深层次上性能下降的问题。GENs可以增强节点对远距离邻域的感知，并扩展网络传播的深度。

    

    图神经网络（GNN）广泛应用于各个领域，但在深层次上表现不佳。现有研究通常将这个问题归因于节点过度平滑，即在多轮传播之后，节点表示变得无法区分。在本文中，我们深入研究了GNN的邻域传播机制，并发现GNN在深层次上性能下降的真正根本原因在于邻域特征传播的无效性。这种传播在每一步传播中导致节点当前表示的指数增长，使得捕捉长距离节点之间的有价值依赖关系变得极具挑战性。为了解决这个问题，我们引入了图消除网络（GENs），它使用一种特定的算法在邻域传播过程中消除冗余。我们证明了GENs可以增强节点对远距离邻域的感知，并扩展网络传播的深度。

    Graph Neural Networks (GNNs) are widely applied across various domains, yet they perform poorly in deep layers. Existing research typically attributes this problem to node over-smoothing, where node representations become indistinguishable after multiple rounds of propagation. In this paper, we delve into the neighborhood propagation mechanism of GNNs and discover that the real root cause of GNNs' performance degradation in deep layers lies in ineffective neighborhood feature propagation. This propagation leads to an exponential growth of a node's current representation at every propagation step, making it extremely challenging to capture valuable dependencies between long-distance nodes. To address this issue, we introduce Graph Elimination Networks (GENs), which employ a specific algorithm to eliminate redundancies during neighborhood propagation. We demonstrate that GENs can enhance nodes' perception of distant neighborhoods and extend the depth of network propagation. Extensive exp
    
[^6]: ODTlearn: 一个用于学习预测和处方的最优决策树的包

    ODTlearn: A Package for Learning Optimal Decision Trees for Prediction and Prescription. (arXiv:2307.15691v1 [stat.ML])

    [http://arxiv.org/abs/2307.15691](http://arxiv.org/abs/2307.15691)

    ODTlearn是一个开源的Python包，用于学习预测和处方的最优决策树。它提供了多种优化方法，并支持各种问题和算法的扩展。

    

    ODTLearn是一个开源的Python包，提供了基于混合整数优化(MIO)框架的高风险预测和处方任务的最优决策树学习方法。该包的当前版本提供了学习最优分类树、公平最优分类树、鲁棒最优分类树和从观测数据学习最优处方树的实现。我们设计了该包以便于维护和扩展，当引入新的最优决策树问题类、重构策略和解决算法时，可以轻松更新。为此，该包遵循面向对象的设计原则，并支持商业(Gurobi)和开源(COIN-OR branch and cut)求解器。包的文档和详细用户指南可以在https://d3m-research-group.github.io/odtlearn/找到。

    ODTLearn is an open-source Python package that provides methods for learning optimal decision trees for high-stakes predictive and prescriptive tasks based on the mixed-integer optimization (MIO) framework proposed in Aghaei et al. (2019) and several of its extensions. The current version of the package provides implementations for learning optimal classification trees, optimal fair classification trees, optimal classification trees robust to distribution shifts, and optimal prescriptive trees from observational data. We have designed the package to be easy to maintain and extend as new optimal decision tree problem classes, reformulation strategies, and solution algorithms are introduced. To this end, the package follows object-oriented design principles and supports both commercial (Gurobi) and open source (COIN-OR branch and cut) solvers. The package documentation and an extensive user guide can be found at https://d3m-research-group.github.io/odtlearn/. Additionally, users can view
    
[^7]: 将患者偏好纳入Q学习的灵活框架

    A Flexible Framework for Incorporating Patient Preferences Into Q-Learning. (arXiv:2307.12022v1 [cs.LG])

    [http://arxiv.org/abs/2307.12022](http://arxiv.org/abs/2307.12022)

    这个论文提出了一种称为潜在效用Q学习的方法，能够将患者偏好纳入复合结果的动态治疗方案中，解决了传统方法对时间点和结果数量的限制，能够实现强大的性能。

    

    在现实世界的医疗问题中，通常存在多个竞争性的关注点，如治疗疗效和副作用严重程度。然而，用于估计动态治疗方案 (DTRs) 的统计方法通常假设只有一个关注点，而处理复合结果的方法很少，存在重要限制，包括对单个时间点和两个结果的限制、无法纳入患者的自述偏好以及有限的理论保证。为此，我们提出了一个新的方法来解决这些限制，我们称之为潜在效用Q学习(LUQ-Learning)。LUQ-Learning采用潜在模型方法，自然地将Q学习扩展到复合结果设置，并为每个患者选择理想的结果权衡。与之前的方法不同，我们的框架允许任意数量的时间点和结果，纳入陈述的偏好，并实现强大的渐近性能。

    In real-world healthcare problems, there are often multiple competing outcomes of interest, such as treatment efficacy and side effect severity. However, statistical methods for estimating dynamic treatment regimes (DTRs) usually assume a single outcome of interest, and the few methods that deal with composite outcomes suffer from important limitations. This includes restrictions to a single time point and two outcomes, the inability to incorporate self-reported patient preferences and limited theoretical guarantees. To this end, we propose a new method to address these limitations, which we dub Latent Utility Q-Learning (LUQ-Learning). LUQ-Learning uses a latent model approach to naturally extend Q-learning to the composite outcome setting and adopt the ideal trade-off between outcomes to each patient. Unlike previous approaches, our framework allows for an arbitrary number of time points and outcomes, incorporates stated preferences and achieves strong asymptotic performance with rea
    
[^8]: K-Tensors：对正半定矩阵进行聚类

    K-Tensors: Clustering Positive Semi-Definite Matrices. (arXiv:2306.06534v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.06534](http://arxiv.org/abs/2306.06534)

    本文介绍了一种针对正半定矩阵的自一致性聚类算法（K-张量），通过考虑其特征结构，能够有效地将正半定矩阵进行分区。

    

    本文介绍了一种新颖的自一致性聚类算法（K-Tensors），用于基于它们的特征结构将正半定矩阵进行分区。由于正半定矩阵可以在 p≥2 的空间中表示为椭球体，因此保持它们的结构信息以进行有效的聚类至关重要。然而，传统的矩阵聚类算法常常涉及将矩阵向量化，导致关键结构信息的丢失。为了解决这个问题，我们提出了一种基于正半定矩阵结构信息的距离度量来进行聚类。这种距离度量使得聚类算法能够考虑正半定矩阵与它们在由一组正半定矩阵定义的正交向量张成的共同空间上的投影之间的差异。这是一种创新的聚类方法。

    This paper introduces a novel self-consistency clustering algorithm ($K$-Tensors) designed for {partitioning a distribution of} positive-semidefinite matrices based on their eigenstructures. As positive semi-definite matrices can be represented as ellipsoids in $\Re^p$, $p \ge 2$, it is critical to maintain their structural information to perform effective clustering. However, traditional clustering algorithms {applied to matrices} often {involve vectorization of} the matrices, resulting in a loss of essential structural information. To address this issue, we propose a distance metric {for clustering} that is specifically based on the structural information of positive semi-definite matrices. This distance metric enables the clustering algorithm to consider the differences between positive semi-definite matrices and their projections onto {a} common space spanned by \thadJulyTen{orthonormal vectors defined from a set of} positive semi-definite matrices. This innovative approach to clus
    
[^9]: 采样，扩散和随机定位

    Sampling, Diffusions, and Stochastic Localization. (arXiv:2305.10690v1 [cs.LG])

    [http://arxiv.org/abs/2305.10690](http://arxiv.org/abs/2305.10690)

    这篇论文介绍了扩散和随机定位的关系，证明了标准去噪扩散是一种随机定位，并提出了一种在对数步骤内从 Ising 模型的 Gibbs 测度中进行采样的算法。

    

    扩散是从高维分布中抽样的成功技术，可以明确给出或从样本集中学习。它们实现了一个扩散过程，其端点是目标分布的样本，漂移通常表示为神经网络。随机定位是在高维中证明马尔科夫链和其他函数不等式混合的成功技术。[EAMS2022]中引入了随机定位的算法版本，以获得从某些统计力学模型中抽样的算法。本文有三个目标：（i）将[EAMS2022]的构造推广到其他随机定位过程；（ii）澄清扩散和随机定位之间的联系。特别是，我们展示了标准去噪扩散是随机定位，但其他通过所提出的视角自然提出的示例；（iii）描述从这种联系中得出的一些见解；特别是，我们提出了一种新的算法，可以在对数步骤内从 Ising 模型的 Gibbs 测度中进行采样。

    Diffusions are a successful technique to sample from high-dimensional distributions can be either explicitly given or learnt from a collection of samples. They implement a diffusion process whose endpoint is a sample from the target distribution and whose drift is typically represented as a neural network. Stochastic localization is a successful technique to prove mixing of Markov Chains and other functional inequalities in high dimension. An algorithmic version of stochastic localization was introduced in [EAMS2022], to obtain an algorithm that samples from certain statistical mechanics models.  This notes have three objectives: (i) Generalize the construction [EAMS2022] to other stochastic localization processes; (ii) Clarify the connection between diffusions and stochastic localization. In particular we show that standard denoising diffusions are stochastic localizations but other examples that are naturally suggested by the proposed viewpoint; (iii) Describe some insights that foll
    
[^10]: 基于抽象语法树的异构有向超图神经网络用于代码分类

    Heterogeneous Directed Hypergraph Neural Network over abstract syntax tree (AST) for Code Classification. (arXiv:2305.04228v2 [cs.SE] UPDATED)

    [http://arxiv.org/abs/2305.04228](http://arxiv.org/abs/2305.04228)

    本研究提出了使用异构有向超图表示AST，并使用异构有向超图神经网络处理图形进行代码分类，超过了现有方法。

    

    代码分类是程序理解和自动编码中的一个难题。由于程序的模糊语法和复杂语义，大多数现有研究使用基于抽象语法树（AST）和图神经网络（GNN）的技术创建代码表示用于代码分类。这些技术利用代码的结构和语义信息，但只考虑节点之间的成对关系，忽略了AST中节点之间已经存在的高阶相关性，可能导致代码结构信息的丢失。本研究提出使用异构有向超图（HDHG）表示AST，并使用异构有向超图神经网络（HDHGN）处理图形。HDHG保留了节点之间的高阶相关性，并更全面地编码了AST的语义和结构信息。HDHGN通过聚合不同节点的特征并使用不同的函数对其进行处理来对AST进行建模。在四个数据集上的实验表明，HDHG和HDHGN在代码分类任务中超越了现有方法。

    Code classification is a difficult issue in program understanding and automatic coding. Due to the elusive syntax and complicated semantics in programs, most existing studies use techniques based on abstract syntax tree (AST) and graph neural network (GNN) to create code representations for code classification. These techniques utilize the structure and semantic information of the code, but they only take into account pairwise associations and neglect the high-order correlations that already exist between nodes in the AST, which may result in the loss of code structural information. On the other hand, while a general hypergraph can encode high-order data correlations, it is homogeneous and undirected which will result in a lack of semantic and structural information such as node types, edge types, and directions between child nodes and parent nodes when modeling AST. In this study, we propose to represent AST as a heterogeneous directed hypergraph (HDHG) and process the graph by hetero
    
[^11]: 可变深度异构联邦学习的内存自适应模型

    Memory-adaptive Depth-wise Heterogenous Federated Learning. (arXiv:2303.04887v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.04887](http://arxiv.org/abs/2303.04887)

    这项研究介绍了一种名为FeDepth的内存自适应深度学习解决方案，它根据每个客户端的内存预算将完整模型自适应地分解成块，并依次训练这些块，以解决联邦学习中异构设备的内存限制问题。

    

    联邦学习是一种有前途的范式，允许多个客户端在不共享本地数据的情况下协同训练模型。然而，在联邦学习中存在异构设备，如手机和物联网设备的内存能力不同，会限制模型能够训练的规模和性能。主要解决内存限制的方法集中在减少宽度的技术上，即不同客户端在本地训练减宽度的子网络，然后服务器聚合这些子网络。由于处理聚合阶段中不同子网络宽度变化的负面影响，这些方法产生的全局模型会受到性能的降低。在本文中，我们介绍了一种称为FeDepth的内存自适应深度学习解决方案，它根据每个客户端的内存预算将完整模型自适应地分解成块，并依次训练这些块，以获取更好的性能和可扩展性。

    Federated learning is a promising paradigm that allows multiple clients to collaboratively train a model without sharing the local data. However, the presence of heterogeneous devices in federated learning, such as mobile phones and IoT devices with varying memory capabilities, would limit the scale and hence the performance of the model could be trained. The mainstream approaches to address memory limitations focus on width-slimming techniques, where different clients train subnetworks with reduced widths locally and then the server aggregates the subnetworks. The global model produced from these methods suffers from performance degradation due to the negative impact of the actions taken to handle the varying subnetwork widths in the aggregation phase. In this paper, we introduce a memory-adaptive depth-wise learning solution in FL called FeDepth, which adaptively decomposes the full model into blocks according to the memory budgets of each client and trains blocks sequentially to obt
    

