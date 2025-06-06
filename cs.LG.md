# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Uncertainty-Based Extensible Codebook for Discrete Federated Learning in Heterogeneous Data Silos](https://arxiv.org/abs/2402.18888) | 提出了一种基于不确定性的可拓展编码本的联邦学习框架，用于应对异构数据孤岛中模型适应新分布的挑战 |
| [^2] | [A Comparative Study of Conventional and Tripolar EEG for High-Performance Reach-to-Grasp BCI Systems](https://arxiv.org/abs/2402.09448) | 比较传统EEG与三极EEG在高性能到颤抓握BCI系统中的有效性，包括信噪比、空间分辨率、ERPs和小波时频分析。 |
| [^3] | [ViewFusion: Learning Composable Diffusion Models for Novel View Synthesis](https://arxiv.org/abs/2402.02906) | ViewFusion 是一种用于新视角合成的最新端到端生成方法，具有无与伦比的灵活性，通过同时应用扩散去噪和像素加权掩模的方法解决了先前方法的局限性。 |
| [^4] | [Causal Discovery from Conditionally Stationary Time Series](https://arxiv.org/abs/2110.06257) | 该论文提出了一种State-Dependent Causal Inference（SDCI）方法，可以处理一类宽泛的非平稳时间序列，成功地回复出潜在的因果依赖关系。 |
| [^5] | [High-Dimensional Independence Testing via Maximum and Average Distance Correlations](https://arxiv.org/abs/2001.01095) | 本文介绍并研究了利用最大和平均距离相关性进行高维度独立性检测的方法，并提出了一种快速卡方检验的程序。该方法适用于欧氏距离和高斯核，具有较好的实证表现和广泛的应用场景。 |
| [^6] | [Hi-Core: Hierarchical Knowledge Transfer for Continual Reinforcement Learning.](http://arxiv.org/abs/2401.15098) | Hi-Core提出了一种新的框架，通过层次化的知识迁移来增强连续强化学习。该框架包括利用大型语言模型的推理能力设定目标的高层策略制定和通过强化学习按照高层目标导向的低层策略学习。在实验中，Hi-Core展现了较强的知识迁移能力。 |
| [^7] | [Learning from Topology: Cosmological Parameter Estimation from the Large-scale Structure.](http://arxiv.org/abs/2308.02636) | 该论文提出了一种基于持续同调和神经网络的方法，利用大尺度结构的拓扑信息估计宇宙参数。通过参数恢复测试，发现该方法比传统的贝叶斯推断方法更准确和精确。 |
| [^8] | [Entropy-based Training Methods for Scalable Neural Implicit Sampler.](http://arxiv.org/abs/2306.04952) | 本文提出了一种高效且可扩展的神经隐式采样器，并引入了KL训练法和Fisher训练法来训练它，实现了低计算成本下生成大批量样本。 |
| [^9] | [Policy learning "without'' overlap: Pessimism and generalized empirical Bernstein's inequality.](http://arxiv.org/abs/2212.09900) | 本文提出了一种新的离线策略学习算法，它不需要统一交叠假设，而是利用价值的下限置信区间（LCBs）优化策略，因此能够适应允许行为策略演变和倾向性减弱的情况。 |

# 详细

[^1]: 基于不确定性的离散异构数据孤岛中可拓展编码本的联邦学习

    Uncertainty-Based Extensible Codebook for Discrete Federated Learning in Heterogeneous Data Silos

    [https://arxiv.org/abs/2402.18888](https://arxiv.org/abs/2402.18888)

    提出了一种基于不确定性的可拓展编码本的联邦学习框架，用于应对异构数据孤岛中模型适应新分布的挑战

    

    旨在利用广泛分布的数据集进行联邦学习(FL)面临着一个关键挑战：不同孤岛间数据的异构性。我们发现从FL导出的模型在应用于具有陌生分布的数据孤岛时会表现出明显增加的不确定性。因此，我们提出了一种创新而简单的迭代框架，称为基于不确定性的可拓展编码本联邦学习(UEFL)。该框架动态地将潜在特征映射到可训练的离散向量，评估不确定性，并针对表现出高不确定性的孤岛特别地扩展离散化词典或编码本。

    arXiv:2402.18888v1 Announce Type: new  Abstract: Federated learning (FL), aimed at leveraging vast distributed datasets, confronts a crucial challenge: the heterogeneity of data across different silos. While previous studies have explored discrete representations to enhance model generalization across minor distributional shifts, these approaches often struggle to adapt to new data silos with significantly divergent distributions. In response, we have identified that models derived from FL exhibit markedly increased uncertainty when applied to data silos with unfamiliar distributions. Consequently, we propose an innovative yet straightforward iterative framework, termed Uncertainty-Based Extensible-Codebook Federated Learning (UEFL). This framework dynamically maps latent features to trainable discrete vectors, assesses the uncertainty, and specifically extends the discretization dictionary or codebook for silos exhibiting high uncertainty. Our approach aims to simultaneously enhance a
    
[^2]: 普通EEG与三极EEG在高性能到颤抓握BCI系统中的比较研究

    A Comparative Study of Conventional and Tripolar EEG for High-Performance Reach-to-Grasp BCI Systems

    [https://arxiv.org/abs/2402.09448](https://arxiv.org/abs/2402.09448)

    比较传统EEG与三极EEG在高性能到颤抓握BCI系统中的有效性，包括信噪比、空间分辨率、ERPs和小波时频分析。

    

    本研究旨在比较传统EEG与三极EEG在提升运动障碍个体的BCI应用方面的有效性。重点是解读和解码各种抓握动作，如力握和精确握持。目标是确定哪种EEG技术在处理和翻译与抓握相关的脑电信号方面更为有效。研究涉及对十名健康参与者进行实验，参与者进行了两种不同的握持运动：力握和精确握持，无运动条件作为基线。我们的研究在解码抓握动作方面对EEG和三极EEG进行了全面比较。该比较涵盖了几个关键参数，包括信噪比（SNR）、通过功能连接的空间分辨率、ERPs和小波时频分析。此外，我们的研究还涉及从...

    arXiv:2402.09448v1 Announce Type: cross  Abstract: This study aims to enhance BCI applications for individuals with motor impairments by comparing the effectiveness of tripolar EEG (tEEG) with conventional EEG. The focus is on interpreting and decoding various grasping movements, such as power grasp and precision grasp. The goal is to determine which EEG technology is more effective in processing and translating grasp related neural signals. The approach involved experimenting on ten healthy participants who performed two distinct grasp movements: power grasp and precision grasp, with a no movement condition serving as the baseline. Our research presents a thorough comparison between EEG and tEEG in decoding grasping movements. This comparison spans several key parameters, including signal to noise ratio (SNR), spatial resolution via functional connectivity, ERPs, and wavelet time frequency analysis. Additionally, our study involved extracting and analyzing statistical features from th
    
[^3]: ViewFusion: 学习可组合的扩散模型用于新视角合成

    ViewFusion: Learning Composable Diffusion Models for Novel View Synthesis

    [https://arxiv.org/abs/2402.02906](https://arxiv.org/abs/2402.02906)

    ViewFusion 是一种用于新视角合成的最新端到端生成方法，具有无与伦比的灵活性，通过同时应用扩散去噪和像素加权掩模的方法解决了先前方法的局限性。

    

    深度学习为新视角合成这个老问题提供了丰富的新方法，从基于神经辐射场（NeRF）的方法到端到端的风格架构。每种方法都具有特定的优势，但也具有特定的适用性限制。这项工作引入了ViewFusion，这是一种具有无与伦比的灵活性的最新端到端生成方法，用于新视角合成。ViewFusion同时对场景的任意数量的输入视角应用扩散去噪步骤，然后将每个视角得到的噪声梯度与（推断得到的）像素加权掩模相结合，确保对于目标场景的每个区域，只考虑最具信息量的输入视角。我们的方法通过以下方式解决了先前方法的几个局限性：（1）可训练且能够泛化到多个场景和物体类别，（2）在训练和测试时自适应地采用可变数量的无姿态视图，（3）生成高质量的合成图像。

    Deep learning is providing a wealth of new approaches to the old problem of novel view synthesis, from Neural Radiance Field (NeRF) based approaches to end-to-end style architectures. Each approach offers specific strengths but also comes with specific limitations in their applicability. This work introduces ViewFusion, a state-of-the-art end-to-end generative approach to novel view synthesis with unparalleled flexibility. ViewFusion consists in simultaneously applying a diffusion denoising step to any number of input views of a scene, then combining the noise gradients obtained for each view with an (inferred) pixel-weighting mask, ensuring that for each region of the target scene only the most informative input views are taken into account. Our approach resolves several limitations of previous approaches by (1) being trainable and generalizing across multiple scenes and object classes, (2) adaptively taking in a variable number of pose-free views at both train and test time, (3) gene
    
[^4]: 从有条件平稳时间序列中进行因果发现

    Causal Discovery from Conditionally Stationary Time Series

    [https://arxiv.org/abs/2110.06257](https://arxiv.org/abs/2110.06257)

    该论文提出了一种State-Dependent Causal Inference（SDCI）方法，可以处理一类宽泛的非平稳时间序列，成功地回复出潜在的因果依赖关系。

    

    因果发现，即从观测数据推断潜在的因果关系，已被证明对AI系统具有极大挑战。在时间序列建模背景下，传统的因果发现方法主要考虑具有完全观测变量和/或来自平稳时间序列的数据的受限场景。我们开发了一种因果发现方法来处理一类宽泛的非平稳时间序列，即在条件上是平稳的条件平稳时间序列，其中非平稳行为被建模为在一组（可能是隐藏的）状态变量上的平稳性。命名为State-Dependent Causal Inference（SDCI），我们的方法能够可证地回复出潜在的因果依赖关系，证明在完全观察到的状态下，并在存在隐藏状态时经验性地实现。后者通过对合成线性系统和非线性粒子相互作用数据的实验进行验证，SDCI实现了优于基线因果发现方法的性能。

    arXiv:2110.06257v2 Announce Type: replace  Abstract: Causal discovery, i.e., inferring underlying causal relationships from observational data, has been shown to be highly challenging for AI systems. In time series modeling context, traditional causal discovery methods mainly consider constrained scenarios with fully observed variables and/or data from stationary time-series. We develop a causal discovery approach to handle a wide class of non-stationary time-series that are conditionally stationary, where the non-stationary behaviour is modeled as stationarity conditioned on a set of (possibly hidden) state variables. Named State-Dependent Causal Inference (SDCI), our approach is able to recover the underlying causal dependencies, provably with fully-observed states and empirically with hidden states. The latter is confirmed by experiments on synthetic linear system and nonlinear particle interaction data, where SDCI achieves superior performance over baseline causal discovery methods
    
[^5]: 高维度独立性检测: 通过最大和平均距离相关性

    High-Dimensional Independence Testing via Maximum and Average Distance Correlations

    [https://arxiv.org/abs/2001.01095](https://arxiv.org/abs/2001.01095)

    本文介绍并研究了利用最大和平均距离相关性进行高维度独立性检测的方法，并提出了一种快速卡方检验的程序。该方法适用于欧氏距离和高斯核，具有较好的实证表现和广泛的应用场景。

    

    本文介绍并研究了利用最大和平均距离相关性进行多元独立性检测的方法。我们在高维环境中表征了它们相对于边际相关维度数量的一致性特性，评估了每个检验统计量的优势，检查了它们各自的零分布，并提出了一种基于快速卡方检验的检测程序。得出的检验是非参数的，并适用于欧氏距离和高斯核作为底层度量。为了更好地理解所提出的测试的实际使用情况，我们在各种多元相关场景中评估了最大距离相关性、平均距离相关性和原始距离相关性的实证表现，同时进行了一个真实数据实验，以检测人类血浆中不同癌症类型和肽水平的存在。

    This paper introduces and investigates the utilization of maximum and average distance correlations for multivariate independence testing. We characterize their consistency properties in high-dimensional settings with respect to the number of marginally dependent dimensions, assess the advantages of each test statistic, examine their respective null distributions, and present a fast chi-square-based testing procedure. The resulting tests are non-parametric and applicable to both Euclidean distance and the Gaussian kernel as the underlying metric. To better understand the practical use cases of the proposed tests, we evaluate the empirical performance of the maximum distance correlation, average distance correlation, and the original distance correlation across various multivariate dependence scenarios, as well as conduct a real data experiment to test the presence of various cancer types and peptide levels in human plasma.
    
[^6]: Hi-Core: 面向连续强化学习的层次化知识迁移

    Hi-Core: Hierarchical Knowledge Transfer for Continual Reinforcement Learning. (arXiv:2401.15098v1 [cs.LG])

    [http://arxiv.org/abs/2401.15098](http://arxiv.org/abs/2401.15098)

    Hi-Core提出了一种新的框架，通过层次化的知识迁移来增强连续强化学习。该框架包括利用大型语言模型的推理能力设定目标的高层策略制定和通过强化学习按照高层目标导向的低层策略学习。在实验中，Hi-Core展现了较强的知识迁移能力。

    

    连续强化学习（Continual Reinforcement Learning, CRL）赋予强化学习智能体从一系列任务中学习的能力，保留先前的知识并利用它来促进未来的学习。然而，现有的方法往往专注于在类似任务之间传输低层次的知识，忽视了人类认知控制的层次结构，导致在各种任务之间的知识迁移不足。为了增强高层次的知识迁移，我们提出了一种名为Hi-Core (Hierarchical knowledge transfer for Continual reinforcement learning)的新框架，它由两层结构组成：1) 利用大型语言模型（Large Language Model, LLM）的强大推理能力设定目标的高层策略制定和2) 通过强化学习按照高层目标导向的低层策略学习。此外，构建了一个知识库（策略库）来存储可以用于层次化知识迁移的策略。在MiniGr实验中进行了实验。

    Continual reinforcement learning (CRL) empowers RL agents with the ability to learn from a sequence of tasks, preserving previous knowledge and leveraging it to facilitate future learning. However, existing methods often focus on transferring low-level knowledge across similar tasks, which neglects the hierarchical structure of human cognitive control, resulting in insufficient knowledge transfer across diverse tasks. To enhance high-level knowledge transfer, we propose a novel framework named Hi-Core (Hierarchical knowledge transfer for Continual reinforcement learning), which is structured in two layers: 1) the high-level policy formulation which utilizes the powerful reasoning ability of the Large Language Model (LLM) to set goals and 2) the low-level policy learning through RL which is oriented by high-level goals. Moreover, the knowledge base (policy library) is constructed to store policies that can be retrieved for hierarchical knowledge transfer. Experiments conducted in MiniGr
    
[^7]: 从拓扑学中学习：大尺度结构的宇宙参数估计

    Learning from Topology: Cosmological Parameter Estimation from the Large-scale Structure. (arXiv:2308.02636v1 [astro-ph.CO])

    [http://arxiv.org/abs/2308.02636](http://arxiv.org/abs/2308.02636)

    该论文提出了一种基于持续同调和神经网络的方法，利用大尺度结构的拓扑信息估计宇宙参数。通过参数恢复测试，发现该方法比传统的贝叶斯推断方法更准确和精确。

    

    宇宙大尺度结构的拓扑包含着有关基础宇宙参数的宝贵信息。虽然持续同调可以提取这种拓扑信息，但如何最佳地从这个工具中进行参数估计仍然是一个开放的问题。为了解决这个问题，我们提出了一个神经网络模型，用于将持续图像映射到宇宙参数。通过参数恢复测试，我们证明我们的模型能够准确而精确地估计，明显优于传统的贝叶斯推断方法。

    The topology of the large-scale structure of the universe contains valuable information on the underlying cosmological parameters. While persistent homology can extract this topological information, the optimal method for parameter estimation from the tool remains an open question. To address this, we propose a neural network model to map persistence images to cosmological parameters. Through a parameter recovery test, we demonstrate that our model makes accurate and precise estimates, considerably outperforming conventional Bayesian inference approaches.
    
[^8]: 基于熵的训练方法用于可扩展的神经隐式采样器

    Entropy-based Training Methods for Scalable Neural Implicit Sampler. (arXiv:2306.04952v1 [stat.ML])

    [http://arxiv.org/abs/2306.04952](http://arxiv.org/abs/2306.04952)

    本文提出了一种高效且可扩展的神经隐式采样器，并引入了KL训练法和Fisher训练法来训练它，实现了低计算成本下生成大批量样本。

    

    高效地从非标准目标分布中采样是科学计算和机器学习中的一个基本问题。传统方法如马尔科夫蒙特卡洛（MCMC）可保证从这些分布中渐进无偏采样，但在处理高维目标时计算效率低下，需要多次迭代生成一批样本。本文提出了一种高效且可扩展的神经隐式采样器，通过利用直接将易于采样的潜在向量映射到目标样本的神经变换，可以在低计算成本下生成大批量样本。为了训练神经隐式采样器，我们引入了两种新方法：KL训练法和Fisher训练法。前者最小化Kullback-Leibler散度，而后者则最小化Fisher散度。

    Efficiently sampling from un-normalized target distributions is a fundamental problem in scientific computing and machine learning. Traditional approaches like Markov Chain Monte Carlo (MCMC) guarantee asymptotically unbiased samples from such distributions but suffer from computational inefficiency, particularly when dealing with high-dimensional targets, as they require numerous iterations to generate a batch of samples. In this paper, we propose an efficient and scalable neural implicit sampler that overcomes these limitations. Our sampler can generate large batches of samples with low computational costs by leveraging a neural transformation that directly maps easily sampled latent vectors to target samples without the need for iterative procedures. To train the neural implicit sampler, we introduce two novel methods: the KL training method and the Fisher training method. The former minimizes the Kullback-Leibler divergence, while the latter minimizes the Fisher divergence. By empl
    
[^9]: 无交叠策略学习：悲观和广义经验Bernstein不等式

    Policy learning "without'' overlap: Pessimism and generalized empirical Bernstein's inequality. (arXiv:2212.09900v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.09900](http://arxiv.org/abs/2212.09900)

    本文提出了一种新的离线策略学习算法，它不需要统一交叠假设，而是利用价值的下限置信区间（LCBs）优化策略，因此能够适应允许行为策略演变和倾向性减弱的情况。

    

    本文研究了离线策略学习，旨在利用先前收集到的观测（来自于固定的或是适应演变的行为策略）来学习给定类别中的最优个性化决策规则。现有的策略学习方法依赖于一个统一交叠假设，即离线数据集中探索所有个性化特征的所有动作的倾向性下界。换句话说，这些方法的性能取决于离线数据集中最坏的倾向性。由于数据收集过程不受控制，在许多情况下，这种假设可能不太现实，特别是当允许行为策略随时间演变并且倾向性减弱时。为此，本文提出了一种新的算法，它优化策略价值的下限置信区间（LCBs）——而不是点估计。LCBs通过量化增强倒数倾向权重的估计不确定性来构建。

    This paper studies offline policy learning, which aims at utilizing observations collected a priori (from either fixed or adaptively evolving behavior policies) to learn the optimal individualized decision rule in a given class. Existing policy learning methods rely on a uniform overlap assumption, i.e., the propensities of exploring all actions for all individual characteristics are lower bounded in the offline dataset. In other words, the performance of these methods depends on the worst-case propensity in the offline dataset. As one has no control over the data collection process, this assumption can be unrealistic in many situations, especially when the behavior policies are allowed to evolve over time with diminishing propensities.  In this paper, we propose a new algorithm that optimizes lower confidence bounds (LCBs) -- instead of point estimates -- of the policy values. The LCBs are constructed by quantifying the estimation uncertainty of the augmented inverse propensity weight
    

