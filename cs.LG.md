# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Online Local False Discovery Rate Control: A Resource Allocation Approach](https://arxiv.org/abs/2402.11425) | 该研究提出了一种在线局部虚发现率控制的资源分配方法，实现了$O(\sqrt{T})$的后悔率，并指出这种后悔率在一般情况下是不可改进的。 |
| [^2] | [Multi-View Hypercomplex Learning for Breast Cancer Screening](https://arxiv.org/abs/2204.05798) | 本文提出了一种基于参数化超复数神经网络的多视图乳腺癌分类方法，能够模拟并利用乳房X光检查的不同视图之间的相关性，从而提高肿瘤识别效果。 |
| [^3] | [Shortcutting Cross-Validation: Efficiently Deriving Column-Wise Centered and Scaled Training Set $\mathbf{X}^\mathbf{T}\mathbf{X}$ and $\mathbf{X}^\mathbf{T}\mathbf{Y}$ Without Full Recomputation of Matrix Products or Statistical Moments.](http://arxiv.org/abs/2401.13185) | 本文提出了三种高效计算训练集$\mathbf{X}^\mathbf{T}\mathbf{X}$和$\mathbf{X}^\mathbf{T}\mathbf{Y}$的算法，相比于以前的工作，这些算法能够显著加速交叉验证，而无需重新计算矩阵乘积或统计量。 |
| [^4] | [Leveraging World Model Disentanglement in Value-Based Multi-Agent Reinforcement Learning.](http://arxiv.org/abs/2309.04615) | 本文提出了一种新的基于模型的多智能体强化学习方法，通过使用模块化的世界模型，减少了多智能体系统中训练的样本复杂性，并成功预测了联合动作价值函数。 |
| [^5] | [Escaping the Sample Trap: Fast and Accurate Epistemic Uncertainty Estimation with Pairwise-Distance Estimators.](http://arxiv.org/abs/2308.13498) | 本文介绍了使用配对距离估计器对集成模型进行认识不确定性估计的新方法，相比于常用的深度学习方法，该方法能够更快速、更准确地在更大的空间和更高维度上估计认识不确定性。 |
| [^6] | [Data-driven Piecewise Affine Decision Rules for Stochastic Programming with Covariate Information.](http://arxiv.org/abs/2304.13646) | 本研究提出一种嵌入非凸分段仿射决策规则的经验风险最小化方法，用于学习特征与最优决策之间的直接映射。所提出的方法可用于广泛的非凸型SP问题，并且在数值研究中表现出优越的性能。 |
| [^7] | [Pre-Training Representations of Binary Code Using Contrastive Learning.](http://arxiv.org/abs/2210.05102) | 提出了一种使用对比学习预训练二进制代码表示的方法，可以将源代码和注释信息纳入二进制代码的表示学习中，对于反向工程和计算机安全任务有重要意义。 |

# 详细

[^1]: 在线局部虚发现率控制：一种资源分配方法

    Online Local False Discovery Rate Control: A Resource Allocation Approach

    [https://arxiv.org/abs/2402.11425](https://arxiv.org/abs/2402.11425)

    该研究提出了一种在线局部虚发现率控制的资源分配方法，实现了$O(\sqrt{T})$的后悔率，并指出这种后悔率在一般情况下是不可改进的。

    

    我们考虑在线局部虚发现率（FDR）控制问题，其中多个测试被顺序进行，目标是最大化总期望的发现次数。我们将问题形式化为一种在线资源分配问题，涉及接受/拒绝决策，从高层次来看，这可以被视为一个带有额外不确定性的在线背包问题，即随机预算补充。我们从一般的到达分布开始，并提出了一个简单的策略，实现了$O(\sqrt{T})$的后悔。我们通过展示这种后悔率在一般情况下是不可改进的来补充这一结果。然后我们将焦点转向离散到达分布。我们发现许多现有的在线资源分配文献中的重新解决启发式虽然在典型设置中实现了有界的损失，但可能会造成$\Omega(\sqrt{T})$甚至$\Omega(T)$的后悔。通过观察到典型策略往往太过

    arXiv:2402.11425v1 Announce Type: cross  Abstract: We consider the problem of online local false discovery rate (FDR) control where multiple tests are conducted sequentially, with the goal of maximizing the total expected number of discoveries. We formulate the problem as an online resource allocation problem with accept/reject decisions, which from a high level can be viewed as an online knapsack problem, with the additional uncertainty of random budget replenishment. We start with general arrival distributions and propose a simple policy that achieves a $O(\sqrt{T})$ regret. We complement the result by showing that such regret rate is in general not improvable. We then shift our focus to discrete arrival distributions. We find that many existing re-solving heuristics in the online resource allocation literature, albeit achieve bounded loss in canonical settings, may incur a $\Omega(\sqrt{T})$ or even a $\Omega(T)$ regret. With the observation that canonical policies tend to be too op
    
[^2]: 多视图超复数学习用于乳腺癌筛查

    Multi-View Hypercomplex Learning for Breast Cancer Screening

    [https://arxiv.org/abs/2204.05798](https://arxiv.org/abs/2204.05798)

    本文提出了一种基于参数化超复数神经网络的多视图乳腺癌分类方法，能够模拟并利用乳房X光检查的不同视图之间的相关性，从而提高肿瘤识别效果。

    

    传统上，用于乳腺癌分类的深度学习方法执行单视图分析。然而，由于乳腺X-ray图像中包含的相关性，放射科医生同时分析组成乳房X光摄影检查的所有四个视图，这为识别肿瘤提供了关键信息。鉴于此，一些研究已经开始提出多视图方法。然而，在这样的现有架构中，乳房X光图像被独立的卷积分支处理为独立的图像，从而失去了它们之间的相关性。为了克服这些局限性，在本文中，我们提出了一种基于参数化超复数神经网络的多视图乳腺癌分类方法。由于超复数代数特性，我们的网络能够建模并利用组成乳房X光检查的不同视图之间的现有相关性，从而模拟阅片过程。

    arXiv:2204.05798v3 Announce Type: replace-cross  Abstract: Traditionally, deep learning methods for breast cancer classification perform a single-view analysis. However, radiologists simultaneously analyze all four views that compose a mammography exam, owing to the correlations contained in mammography views, which present crucial information for identifying tumors. In light of this, some studies have started to propose multi-view methods. Nevertheless, in such existing architectures, mammogram views are processed as independent images by separate convolutional branches, thus losing correlations among them. To overcome such limitations, in this paper, we propose a methodological approach for multi-view breast cancer classification based on parameterized hypercomplex neural networks. Thanks to hypercomplex algebra properties, our networks are able to model, and thus leverage, existing correlations between the different views that comprise a mammogram, thus mimicking the reading process
    
[^3]: 简化交叉验证：高效地计算不需要全量重新计算矩阵乘积或统计量的列向中心化和标准化训练集$\mathbf{X}^\mathbf{T}\mathbf{X}$和$\mathbf{X}^\mathbf{T}\mathbf{Y}$

    Shortcutting Cross-Validation: Efficiently Deriving Column-Wise Centered and Scaled Training Set $\mathbf{X}^\mathbf{T}\mathbf{X}$ and $\mathbf{X}^\mathbf{T}\mathbf{Y}$ Without Full Recomputation of Matrix Products or Statistical Moments. (arXiv:2401.13185v1 [cs.LG])

    [http://arxiv.org/abs/2401.13185](http://arxiv.org/abs/2401.13185)

    本文提出了三种高效计算训练集$\mathbf{X}^\mathbf{T}\mathbf{X}$和$\mathbf{X}^\mathbf{T}\mathbf{Y}$的算法，相比于以前的工作，这些算法能够显著加速交叉验证，而无需重新计算矩阵乘积或统计量。

    

    交叉验证是一种广泛使用的评估预测模型在未知数据上表现的技术。许多预测模型，如基于核的偏最小二乘（PLS）模型，需要仅使用输入矩阵$\mathbf{X}$和输出矩阵$\mathbf{Y}$中的训练集样本来计算$\mathbf{X}^{\mathbf{T}}\mathbf{X}$和$\mathbf{X}^{\mathbf{T}}\mathbf{Y}$。在这项工作中，我们提出了三种高效计算这些矩阵的算法。第一种算法不需要列向预处理。第二种算法允许以训练集均值为中心化点进行列向中心化。第三种算法允许以训练集均值和标准差为中心化点和标准化点进行列向中心化和标准化。通过证明正确性和优越的计算复杂度，它们相比于直接交叉验证和以前的快速交叉验证工作，提供了显著的交叉验证加速，而无需数据泄露。它们适合并行计算。

    Cross-validation is a widely used technique for assessing the performance of predictive models on unseen data. Many predictive models, such as Kernel-Based Partial Least-Squares (PLS) models, require the computation of $\mathbf{X}^{\mathbf{T}}\mathbf{X}$ and $\mathbf{X}^{\mathbf{T}}\mathbf{Y}$ using only training set samples from the input and output matrices, $\mathbf{X}$ and $\mathbf{Y}$, respectively. In this work, we present three algorithms that efficiently compute these matrices. The first one allows no column-wise preprocessing. The second one allows column-wise centering around the training set means. The third one allows column-wise centering and column-wise scaling around the training set means and standard deviations. Demonstrating correctness and superior computational complexity, they offer significant cross-validation speedup compared with straight-forward cross-validation and previous work on fast cross-validation - all without data leakage. Their suitability for paralle
    
[^4]: 利用世界模型分解在基于值的多智能体强化学习中的应用

    Leveraging World Model Disentanglement in Value-Based Multi-Agent Reinforcement Learning. (arXiv:2309.04615v1 [cs.LG])

    [http://arxiv.org/abs/2309.04615](http://arxiv.org/abs/2309.04615)

    本文提出了一种新的基于模型的多智能体强化学习方法，通过使用模块化的世界模型，减少了多智能体系统中训练的样本复杂性，并成功预测了联合动作价值函数。

    

    本文提出了一种新颖的基于模型的多智能体强化学习方法，名为Value Decomposition Framework with Disentangled World Model，旨在解决在相同环境中多个智能体达成共同目标时的样本复杂性问题。由于多智能体系统的可扩展性和非平稳性问题，无模型方法依赖于大量样本进行训练。相反地，我们使用模块化的世界模型，包括动作条件、无动作和静态分支，来解开环境动态并根据过去的经验产生想象中的结果，而不是直接从真实环境中采样。我们使用变分自动编码器和变分图自动编码器来学习世界模型的潜在表示，将其与基于值的框架合并，以预测联合动作价值函数并优化整体训练目标。我们提供实验结果。

    In this paper, we propose a novel model-based multi-agent reinforcement learning approach named Value Decomposition Framework with Disentangled World Model to address the challenge of achieving a common goal of multiple agents interacting in the same environment with reduced sample complexity. Due to scalability and non-stationarity problems posed by multi-agent systems, model-free methods rely on a considerable number of samples for training. In contrast, we use a modularized world model, composed of action-conditioned, action-free, and static branches, to unravel the environment dynamics and produce imagined outcomes based on past experience, without sampling directly from the real environment. We employ variational auto-encoders and variational graph auto-encoders to learn the latent representations for the world model, which is merged with a value-based framework to predict the joint action-value function and optimize the overall training objective. We present experimental results 
    
[^5]: 逃离样本陷阱：使用配对距离估计器快速准确地估计认识不确定性

    Escaping the Sample Trap: Fast and Accurate Epistemic Uncertainty Estimation with Pairwise-Distance Estimators. (arXiv:2308.13498v1 [cs.LG])

    [http://arxiv.org/abs/2308.13498](http://arxiv.org/abs/2308.13498)

    本文介绍了使用配对距离估计器对集成模型进行认识不确定性估计的新方法，相比于常用的深度学习方法，该方法能够更快速、更准确地在更大的空间和更高维度上估计认识不确定性。

    

    本文介绍了一种使用配对距离估计器（PaiDEs）对集成模型进行认识不确定性估计的新方法。这些估计器利用模型组件之间的配对距离来建立熵的边界，并将这些边界作为基于信息准则的估计值。与最近基于样本的蒙特卡洛估计器用于认识不确定性估计的深度学习方法不同，PaiDEs能够在更大的空间（最多100倍）上以更快的速度（最多100倍）估计认识不确定性，并在更高维度上具有更准确的性能。为了验证我们的方法，我们进行了一系列用于评估认识不确定性估计的实验：一维正弦数据，摆动物体（Pendulum-v0），跳跃机器人（Hopper-v2），蚂蚁机器人（Ant-v2）和人形机器人（Humanoid-v2）。对于每个实验设置，我们应用了主动学习框架来展示PaiDEs在认识不确定性估计中的优势。

    This work introduces a novel approach for epistemic uncertainty estimation for ensemble models using pairwise-distance estimators (PaiDEs). These estimators utilize the pairwise-distance between model components to establish bounds on entropy and uses said bounds as estimates for information-based criterion. Unlike recent deep learning methods for epistemic uncertainty estimation, which rely on sample-based Monte Carlo estimators, PaiDEs are able to estimate epistemic uncertainty up to 100$\times$ faster, over a larger space (up to 100$\times$) and perform more accurately in higher dimensions. To validate our approach, we conducted a series of experiments commonly used to evaluate epistemic uncertainty estimation: 1D sinusoidal data, Pendulum-v0, Hopper-v2, Ant-v2 and Humanoid-v2. For each experimental setting, an Active Learning framework was applied to demonstrate the advantages of PaiDEs for epistemic uncertainty estimation.
    
[^6]: 基于数据驱动的分段仿射决策规则用于带协变信息的随机规划

    Data-driven Piecewise Affine Decision Rules for Stochastic Programming with Covariate Information. (arXiv:2304.13646v1 [math.OC])

    [http://arxiv.org/abs/2304.13646](http://arxiv.org/abs/2304.13646)

    本研究提出一种嵌入非凸分段仿射决策规则的经验风险最小化方法，用于学习特征与最优决策之间的直接映射。所提出的方法可用于广泛的非凸型SP问题，并且在数值研究中表现出优越的性能。

    

    本文针对带协变信息的随机规划，提出了一种嵌入非凸分段仿射决策规则(PADR)的经验风险最小化(ERM)方法，旨在学习特征与最优决策之间的直接映射。我们建立了基于PADR的ERM模型的非渐近一致性结果，可用于无约束问题，以及约束问题的渐近一致性结果。为了解决非凸和非可微的ERM问题，我们开发了一个增强的随机主导下降算法，并建立了沿（复合强）方向稳定性的渐近收敛以及复杂性分析。我们表明，所提出的PADR-based ERM方法适用于广泛的非凸型SP问题，并具有理论一致性保证和计算可处理性。数值研究表明，在各种设置下，PADR-based ERM方法相对于最先进的方法具有优越的性能。

    Focusing on stochastic programming (SP) with covariate information, this paper proposes an empirical risk minimization (ERM) method embedded within a nonconvex piecewise affine decision rule (PADR), which aims to learn the direct mapping from features to optimal decisions. We establish the nonasymptotic consistency result of our PADR-based ERM model for unconstrained problems and asymptotic consistency result for constrained ones. To solve the nonconvex and nondifferentiable ERM problem, we develop an enhanced stochastic majorization-minimization algorithm and establish the asymptotic convergence to (composite strong) directional stationarity along with complexity analysis. We show that the proposed PADR-based ERM method applies to a broad class of nonconvex SP problems with theoretical consistency guarantees and computational tractability. Our numerical study demonstrates the superior performance of PADR-based ERM methods compared to state-of-the-art approaches under various settings,
    
[^7]: 使用对比学习预训练二进制代码表示

    Pre-Training Representations of Binary Code Using Contrastive Learning. (arXiv:2210.05102v2 [cs.SE] UPDATED)

    [http://arxiv.org/abs/2210.05102](http://arxiv.org/abs/2210.05102)

    提出了一种使用对比学习预训练二进制代码表示的方法，可以将源代码和注释信息纳入二进制代码的表示学习中，对于反向工程和计算机安全任务有重要意义。

    

    编译后的软件以可执行的二进制代码形式交付。开发人员编写源代码来表达软件的语义，但编译器将其转换为CPU可以直接执行的二进制格式。因此，二进制代码分析对于反向工程和计算机安全任务等没有源代码的应用程序至关重要。然而，与包含丰富语义信息的源代码和自然语言不同，二进制代码通常难以理解和分析。虽然现有的工作使用AI模型辅助源代码分析，但很少有研究考虑二进制代码。在本文中，我们提出了一种将源代码和注释信息纳入二进制代码进行表示学习的对比学习模型，称为COMBO。具体而言，我们在COMBO中提出了三个组件：（1）用于冷启动预训练的主要对比学习方法，（2）用于将源代码和注释信息插入到二进制代码中的单纯插值方法。

    Compiled software is delivered as executable binary code. Developers write source code to express the software semantics, but the compiler converts it to a binary format that the CPU can directly execute. Therefore, binary code analysis is critical to applications in reverse engineering and computer security tasks where source code is not available. However, unlike source code and natural language that contain rich semantic information, binary code is typically difficult for human engineers to understand and analyze. While existing work uses AI models to assist source code analysis, few studies have considered binary code. In this paper, we propose a COntrastive learning Model for Binary cOde Analysis, or COMBO, that incorporates source code and comment information into binary code during representation learning. Specifically, we present three components in COMBO: (1) a primary contrastive learning method for cold-start pre-training, (2) a simplex interpolation method to incorporate so
    

