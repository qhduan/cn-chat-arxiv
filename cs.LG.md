# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text](https://arxiv.org/abs/2403.14773) | StreamingT2V是一种自回归方法，用于生成长视频，可以产生80、240、600、1200帧甚至更多帧的视频，并具有平滑的过渡。 |
| [^2] | [Deep Learning Based Dynamics Identification and Linearization of Orbital Problems using Koopman Theory](https://arxiv.org/abs/2403.08965) | 通过深度学习和库普曼理论，提出了一种数据驱动框架，可以同时识别“两体问题”和“圆限制三体问题”的动力学，并将其全局线性化成线性时不变系统。 |
| [^3] | [a-DCF: an architecture agnostic metric with application to spoofing-robust speaker verification](https://arxiv.org/abs/2403.01355) | 提出了一种架构无关的检测成本函数（a-DCF），适用于评估抵御欺骗攻击的自动说话人验证（ASV）解决方案。 |
| [^4] | [Randomization Can Reduce Both Bias and Variance: A Case Study in Random Forests](https://arxiv.org/abs/2402.12668) | 随机森林相对于装袋法具有减少偏差的能力，在揭示数据模式和高信噪比情况下表现更好的特点，为随机森林在不同信噪比环境下的成功提供了解释和实用见解。 |
| [^5] | [Resilience of the quadratic Littlewood-Offord problem](https://arxiv.org/abs/2402.10504) | 论文研究了二次Littlewood-Offord问题的统计鲁棒性，估计了对抗性噪声对二次Radamecher混沌的影响，并提供了对二次和双线性Rademacher混沌的统计鲁棒性的下限估计。 |
| [^6] | [Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space](https://arxiv.org/abs/2402.09063) | 该论文提出了一种新的嵌入空间攻击方法，针对开源LLMs进行攻击，绕过模型对齐并在遗忘的情况下提取信息，比传统的离散攻击更高效。 |
| [^7] | [Investigating Generalization Behaviours of Generative Flow Networks](https://arxiv.org/abs/2402.05309) | 本研究通过实证验证了生成流网络(GFlowNets)的一些泛化机制假设，发现它们学习逼近的函数具有隐含的基础结构，有助于泛化。同时，GFlowNets对于离线和离策略训练敏感，但隐含学习的奖励对训练分布的变化具有鲁棒性。 |
| [^8] | [Block Majorization Minimization with Extrapolation and Application to $\beta$-NMF.](http://arxiv.org/abs/2401.06646) | 本文提出了一种使用外推的块主导极小化方法（BMMe）来解决多凸优化问题，并将其应用于$\beta$-NMF。通过使用独特的自适应更新规则来更新外推参数，该方法在实验中展现出显著的加速效果。 |
| [^9] | [Strategic Client Selection to Address Non-IIDness in HAPS-enabled FL Networks.](http://arxiv.org/abs/2401.05308) | 该研究介绍了一种针对高空平台站（HAPS）使能的垂直异构网络中数据分布不均问题的战略客户选择策略，通过利用用户的网络流量行为预测和分类，优先选择数据呈现相似模式的客户参与，以提高联合学习（FL）模型的训练效果。 |
| [^10] | [Deep Variational Multivariate Information Bottleneck -- A Framework for Variational Losses.](http://arxiv.org/abs/2310.03311) | 该论文介绍了一个基于信息理论的统一原理，用于重新推导和推广现有的变分降维方法，并设计新的方法。通过将多变量信息瓶颈解释为两个贝叶斯网络的权衡，该框架引入了一个在压缩数据和保留信息之间的权衡参数。 |
| [^11] | [Maximum Likelihood Estimation of Latent Variable Structural Equation Models: A Neural Network Approach.](http://arxiv.org/abs/2309.14073) | 本研究提出了一种新的图形结构，用于在线性和高斯性假设下稳定的潜变量结构方程模型。我们证明了计算该模型的最大似然估计等价于训练一个神经网络，并实现了一个基于GPU的算法来进行计算。 |
| [^12] | [H2O+: An Improved Framework for Hybrid Offline-and-Online RL with Dynamics Gaps.](http://arxiv.org/abs/2309.12716) | H2O+是一种改进的混合离线和在线强化学习框架，通过综合考虑真实和模拟环境的动力学差距，同时利用有限的离线数据和不完美的模拟器进行策略学习，并在广泛的仿真和实际机器人实验中展示了卓越的性能和灵活性。 |

# 详细

[^1]: StreamingT2V: 一种一致、动态和可扩展的基于文本的长视频生成方法

    StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text

    [https://arxiv.org/abs/2403.14773](https://arxiv.org/abs/2403.14773)

    StreamingT2V是一种自回归方法，用于生成长视频，可以产生80、240、600、1200帧甚至更多帧的视频，并具有平滑的过渡。

    

    arXiv:2403.14773v1 公告类型: 交叉 摘要: 文本到视频的扩散模型可以生成遵循文本指令的高质量视频，使得创建多样化和个性化内容变得更加容易。然而，现有方法大多集中在生成高质量的短视频（通常为16或24帧），当天真地扩展到长视频合成的情况时，通常会出现硬裁剪。为了克服这些限制，我们引入了StreamingT2V，这是一种自回归方法，用于生成80、240、600、1200或更多帧的长视频，具有平滑的过渡。主要组件包括：（i）一种名为条件注意力模块（CAM）的短期记忆块，通过注意机制将当前生成条件设置为先前块提取的特征，实现一致的块过渡，（ii）一种名为外观保存模块的长期记忆块，从第一个视频块中提取高级场景和对象特征，以防止th

    arXiv:2403.14773v1 Announce Type: cross  Abstract: Text-to-video diffusion models enable the generation of high-quality videos that follow text instructions, making it easy to create diverse and individual content. However, existing approaches mostly focus on high-quality short video generation (typically 16 or 24 frames), ending up with hard-cuts when naively extended to the case of long video synthesis. To overcome these limitations, we introduce StreamingT2V, an autoregressive approach for long video generation of 80, 240, 600, 1200 or more frames with smooth transitions. The key components are:(i) a short-term memory block called conditional attention module (CAM), which conditions the current generation on the features extracted from the previous chunk via an attentional mechanism, leading to consistent chunk transitions, (ii) a long-term memory block called appearance preservation module, which extracts high-level scene and object features from the first video chunk to prevent th
    
[^2]: 基于深度学习和库普曼理论的轨道问题动力学识别与线性化

    Deep Learning Based Dynamics Identification and Linearization of Orbital Problems using Koopman Theory

    [https://arxiv.org/abs/2403.08965](https://arxiv.org/abs/2403.08965)

    通过深度学习和库普曼理论，提出了一种数据驱动框架，可以同时识别“两体问题”和“圆限制三体问题”的动力学，并将其全局线性化成线性时不变系统。

    

    航空航天工程和科学领域中对“两体问题”和“圆限制三体问题”的研究非常重要，因为它们有助于描述天体和人造卫星的运动。随着对卫星和卫星编队飞行的需求日益增长，对这些系统进行快速有效的控制变得越来越重要。我们提出了一个数据驱动框架，通过基于深度学习的库普曼理论实现“两体问题”和“圆限制三体问题”的同时系统识别和全局线性化，即通过纯数据驱动训练深度神经网络来发现线性库普曼算子，并将其全局线性化为线性时不变系统（LTI）系统。

    arXiv:2403.08965v1 Announce Type: cross  Abstract: The study of the Two-Body and Circular Restricted Three-Body Problems in the field of aerospace engineering and sciences is deeply important because they help describe the motion of both celestial and artificial satellites. With the growing demand for satellites and satellite formation flying, fast and efficient control of these systems is becoming ever more important. Global linearization of these systems allows engineers to employ methods of control in order to achieve these desired results. We propose a data-driven framework for simultaneous system identification and global linearization of both the Two-Body Problem and Circular Restricted Three-Body Problem via deep learning-based Koopman Theory, i.e., a framework that can identify the underlying dynamics and globally linearize it into a linear time-invariant (LTI) system. The linear Koopman operator is discovered through purely data-driven training of a Deep Neural Network with a 
    
[^3]: a-DCF：一种与架构无关的度量，适用于抵御欺骗攻击的说话人验证

    a-DCF: an architecture agnostic metric with application to spoofing-robust speaker verification

    [https://arxiv.org/abs/2403.01355](https://arxiv.org/abs/2403.01355)

    提出了一种架构无关的检测成本函数（a-DCF），适用于评估抵御欺骗攻击的自动说话人验证（ASV）解决方案。

    

    欺骗检测目前是一个主流研究课题。标准度量可以用来评估孤立欺骗检测解决方案的性能，也有一些提出来支持它们在与说话人检测结合时的评估，但存在已知的缺陷或者限制了结合说话人和欺骗检测器的架构方法。本文提出了一种架构无关的检测成本函数（a-DCF）。作为广泛用于评估自动说话人验证（ASV）性能的原始DCF的推广，a-DCF旨在用于评估抵御欺骗攻击的ASV。与DCF类似，a-DCF从Bayes风险的角度反映了决策的代价，其中明确定义了类先验和检测成本模型。我们通过对架构异构的抵御欺骗攻击的ASV解决方案进行基准评估，展示了a-DCF的优点。

    arXiv:2403.01355v1 Announce Type: cross  Abstract: Spoofing detection is today a mainstream research topic. Standard metrics can be applied to evaluate the performance of isolated spoofing detection solutions and others have been proposed to support their evaluation when they are combined with speaker detection. These either have well-known deficiencies or restrict the architectural approach to combine speaker and spoof detectors. In this paper, we propose an architecture-agnostic detection cost function (a-DCF). A generalisation of the original DCF used widely for the assessment of automatic speaker verification (ASV), the a-DCF is designed for the evaluation of spoofing-robust ASV. Like the DCF, the a-DCF reflects the cost of decisions in a Bayes risk sense, with explicitly defined class priors and detection cost model. We demonstrate the merit of the a-DCF through the benchmarking evaluation of architecturally-heterogeneous spoofing-robust ASV solutions.
    
[^4]: 随机化既可以减少偏差又可以减少方差：随机森林的案例研究

    Randomization Can Reduce Both Bias and Variance: A Case Study in Random Forests

    [https://arxiv.org/abs/2402.12668](https://arxiv.org/abs/2402.12668)

    随机森林相对于装袋法具有减少偏差的能力，在揭示数据模式和高信噪比情况下表现更好的特点，为随机森林在不同信噪比环境下的成功提供了解释和实用见解。

    

    我们研究了往往被忽视的现象，首次在\cite{breiman2001random}中指出，即随机森林似乎比装袋法减少了偏差。受\cite{mentch2020randomization}一篇有趣的论文的启发，其中作者认为随机森林减少了有效自由度，并且只有在低信噪比（SNR）环境下才能胜过装袋集成，我们探讨了随机森林如何能够揭示被装袋法忽视的数据模式。我们在实证中证明，在存在这种模式的情况下，随机森林不仅可以减小偏差还能减小方差，并且当信噪比高时随机森林的表现愈发好于装袋集成。我们的观察为解释随机森林在各种信噪比情况下的真实世界成功提供了见解，并增进了我们对随机森林与装袋集成在每次分割注入的随机化方面的差异的理解。我们的调查结果还提供了实用见解。

    arXiv:2402.12668v1 Announce Type: cross  Abstract: We study the often overlooked phenomenon, first noted in \cite{breiman2001random}, that random forests appear to reduce bias compared to bagging. Motivated by an interesting paper by \cite{mentch2020randomization}, where the authors argue that random forests reduce effective degrees of freedom and only outperform bagging ensembles in low signal-to-noise ratio (SNR) settings, we explore how random forests can uncover patterns in the data missed by bagging. We empirically demonstrate that in the presence of such patterns, random forests reduce bias along with variance and increasingly outperform bagging ensembles when SNR is high. Our observations offer insights into the real-world success of random forests across a range of SNRs and enhance our understanding of the difference between random forests and bagging ensembles with respect to the randomization injected into each split. Our investigations also yield practical insights into the 
    
[^5]: 二次Littlewood-Offord问题的弹性

    Resilience of the quadratic Littlewood-Offord problem

    [https://arxiv.org/abs/2402.10504](https://arxiv.org/abs/2402.10504)

    论文研究了二次Littlewood-Offord问题的统计鲁棒性，估计了对抗性噪声对二次Radamecher混沌的影响，并提供了对二次和双线性Rademacher混沌的统计鲁棒性的下限估计。

    

    我们研究了高维数据的统计鲁棒性。我们的结果提供了关于对抗性噪声对二次Radamecher混沌$\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi}$反集中特性的影响的估计，其中$M$是一个固定的（高维）矩阵，$\boldsymbol{\xi}$是一个共形Rademacher向量。具体来说，我们探讨了$\boldsymbol{\xi}$能够承受多少对抗性符号翻转而不“膨胀”$\sup_{x\in \mathbb{R}} \mathbb{P} \left\{\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi} = x\right\}$，从而“去除”原始分布导致更“有粒度”和对抗性偏倚的分布。我们的结果为二次和双线性Rademacher混沌的统计鲁棒性提供了下限估计；这些结果在关键区域被证明是渐近紧的。

    arXiv:2402.10504v1 Announce Type: cross  Abstract: We study the statistical resilience of high-dimensional data. Our results provide estimates as to the effects of adversarial noise over the anti-concentration properties of the quadratic Radamecher chaos $\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi}$, where $M$ is a fixed (high-dimensional) matrix and $\boldsymbol{\xi}$ is a conformal Rademacher vector. Specifically, we pursue the question of how many adversarial sign-flips can $\boldsymbol{\xi}$ sustain without "inflating" $\sup_{x\in \mathbb{R}} \mathbb{P} \left\{\boldsymbol{\xi}^{\mathsf{T}} M \boldsymbol{\xi} = x\right\}$ and thus "de-smooth" the original distribution resulting in a more "grainy" and adversarially biased distribution. Our results provide lower bound estimations for the statistical resilience of the quadratic and bilinear Rademacher chaos; these are shown to be asymptotically tight across key regimes.
    
[^6]: 软提示威胁：通过嵌入空间对开源LLMs进行安全对齐攻击和遗忘

    Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space

    [https://arxiv.org/abs/2402.09063](https://arxiv.org/abs/2402.09063)

    该论文提出了一种新的嵌入空间攻击方法，针对开源LLMs进行攻击，绕过模型对齐并在遗忘的情况下提取信息，比传统的离散攻击更高效。

    

    当前对LLMs的敌对鲁棒性研究专注于自然语言空间中的离散输入操纵，这些操纵可以直接转移到闭源模型中。然而，这种方法忽视了开源模型的持续进展。随着开源模型能力的提升，确保其安全性也变得越来越重要。然而，针对开源LLMs的攻击，利用完全模型访问权限的方式仍然很少被探索。我们填补了这一研究空白，并提出了嵌入空间攻击，直接攻击输入令牌的连续嵌入表示。我们发现，嵌入空间攻击比离散攻击或模型微调更有效地绕过模型对齐并触发有害行为。此外，我们在遗忘的背景下提出了一种新的威胁模型，并展示了嵌入空间攻击在从未经学习的LLMs中提取应该删除的信息方面的能力。

    arXiv:2402.09063v1 Announce Type: new Abstract: Current research in adversarial robustness of LLMs focuses on discrete input manipulations in the natural language space, which can be directly transferred to closed-source models. However, this approach neglects the steady progression of open-source models. As open-source models advance in capability, ensuring their safety also becomes increasingly imperative. Yet, attacks tailored to open-source LLMs that exploit full model access remain largely unexplored. We address this research gap and propose the embedding space attack, which directly attacks the continuous embedding representation of input tokens. We find that embedding space attacks circumvent model alignments and trigger harmful behaviors more efficiently than discrete attacks or model fine-tuning. Furthermore, we present a novel threat model in the context of unlearning and show that embedding space attacks can extract supposedly deleted information from unlearned LLMs across m
    
[^7]: 研究生成流网络的泛化行为

    Investigating Generalization Behaviours of Generative Flow Networks

    [https://arxiv.org/abs/2402.05309](https://arxiv.org/abs/2402.05309)

    本研究通过实证验证了生成流网络(GFlowNets)的一些泛化机制假设，发现它们学习逼近的函数具有隐含的基础结构，有助于泛化。同时，GFlowNets对于离线和离策略训练敏感，但隐含学习的奖励对训练分布的变化具有鲁棒性。

    

    生成流网络（GFlowNets，GFNs）是一种用于学习离散空间上非归一化概率质量函数的生成框架。自从它们问世以来，GFlowNets在学习生成模型方面表现出色，特别适用于训练期间大部分离散空间未被访问的应用。这使一些人假设当GFlowNets与深度神经网络（DNNs）配对时，具有良好的泛化性能。本文通过实证验证了GFlowNets的一些泛化机制假设。特别地，我们发现GFlowNets学习逼近的函数具有隐含的基础结构，有助于泛化。我们还发现GFlowNets对于离线和离策略训练很敏感，然而，GFlowNets隐含学习的奖励对训练分布的变化具有鲁棒性。

    Generative Flow Networks (GFlowNets, GFNs) are a generative framework for learning unnormalized probability mass functions over discrete spaces. Since their inception, GFlowNets have proven to be useful for learning generative models in applications where the majority of the discrete space is unvisited during training. This has inspired some to hypothesize that GFlowNets, when paired with deep neural networks (DNNs), have favourable generalization properties. In this work, we empirically verify some of the hypothesized mechanisms of generalization of GFlowNets. In particular, we find that the functions that GFlowNets learn to approximate have an implicit underlying structure which facilitate generalization. We also find that GFlowNets are sensitive to being trained offline and off-policy; however, the reward implicitly learned by GFlowNets is robust to changes in the training distribution.
    
[^8]: 使用外推的块主导极小化方法和应用于$\beta$-NMF

    Block Majorization Minimization with Extrapolation and Application to $\beta$-NMF. (arXiv:2401.06646v1 [cs.LG])

    [http://arxiv.org/abs/2401.06646](http://arxiv.org/abs/2401.06646)

    本文提出了一种使用外推的块主导极小化方法（BMMe）来解决多凸优化问题，并将其应用于$\beta$-NMF。通过使用独特的自适应更新规则来更新外推参数，该方法在实验中展现出显著的加速效果。

    

    我们提出了一种使用外推的块主导极小化方法（BMMe）来解决一类多凸优化问题。BMMe的外推参数使用一种新颖的自适应更新规则来更新。通过将块主导极小化重新表述为块镜像下降方法，并在每次迭代中自适应更新Bregman散度，我们建立了BMMe的子序列收敛性。我们使用这种方法设计了高效的算法来处理$\beta$-NMF中的非负矩阵分解问题，其中$\beta\in [1,2]$。这些算法是使用外推的乘法更新，并从我们的新结果中获得了收敛性保证。我们还通过大量实验实证了BMMe在$\beta$-NMF中的显著加速效果。

    We propose a Block Majorization Minimization method with Extrapolation (BMMe) for solving a class of multi-convex optimization problems. The extrapolation parameters of BMMe are updated using a novel adaptive update rule. By showing that block majorization minimization can be reformulated as a block mirror descent method, with the Bregman divergence adaptively updated at each iteration, we establish subsequential convergence for BMMe. We use this method to design efficient algorithms to tackle nonnegative matrix factorization problems with the $\beta$-divergences ($\beta$-NMF) for $\beta\in [1,2]$. These algorithms, which are multiplicative updates with extrapolation, benefit from our novel results that offer convergence guarantees. We also empirically illustrate the significant acceleration of BMMe for $\beta$-NMF through extensive experiments.
    
[^9]: 面对HAPS使能的FL网络中的非独立同分布问题，战略客户选择的研究

    Strategic Client Selection to Address Non-IIDness in HAPS-enabled FL Networks. (arXiv:2401.05308v1 [cs.NI])

    [http://arxiv.org/abs/2401.05308](http://arxiv.org/abs/2401.05308)

    该研究介绍了一种针对高空平台站（HAPS）使能的垂直异构网络中数据分布不均问题的战略客户选择策略，通过利用用户的网络流量行为预测和分类，优先选择数据呈现相似模式的客户参与，以提高联合学习（FL）模型的训练效果。

    

    在由高空平台站（HAPS）使能的垂直异构网络中部署联合学习（FL）为各种不同通信和计算能力的客户提供了参与的机会。这种多样性不仅提高了FL模型的训练精度，还加快了其收敛速度。然而，在这些广阔的网络中应用FL存在显著的非独立同分布问题。这种数据异质性往往导致收敛速度较慢和模型训练性能的降低。我们的研究引入了一种针对此问题的客户选择策略，利用用户网络流量行为进行预测和分类。该策略通过战略性选择数据呈现相似模式的客户参与，同时优先考虑用户隐私。

    The deployment of federated learning (FL) within vertical heterogeneous networks, such as those enabled by high-altitude platform station (HAPS), offers the opportunity to engage a wide array of clients, each endowed with distinct communication and computational capabilities. This diversity not only enhances the training accuracy of FL models but also hastens their convergence. Yet, applying FL in these expansive networks presents notable challenges, particularly the significant non-IIDness in client data distributions. Such data heterogeneity often results in slower convergence rates and reduced effectiveness in model training performance. Our study introduces a client selection strategy tailored to address this issue, leveraging user network traffic behaviour. This strategy involves the prediction and classification of clients based on their network usage patterns while prioritizing user privacy. By strategically selecting clients whose data exhibit similar patterns for participation
    
[^10]: 深度变分多变量信息瓶颈--一种变分损失的框架

    Deep Variational Multivariate Information Bottleneck -- A Framework for Variational Losses. (arXiv:2310.03311v1 [cs.LG])

    [http://arxiv.org/abs/2310.03311](http://arxiv.org/abs/2310.03311)

    该论文介绍了一个基于信息理论的统一原理，用于重新推导和推广现有的变分降维方法，并设计新的方法。通过将多变量信息瓶颈解释为两个贝叶斯网络的权衡，该框架引入了一个在压缩数据和保留信息之间的权衡参数。

    

    变分降维方法以其高精度、生成能力和鲁棒性而闻名。这些方法有很多理论上的证明。在这里，我们介绍了一种基于信息理论的统一原理，重新推导和推广了现有的变分方法，并设计了新的方法。我们的框架基于多变量信息瓶颈的解释，其中两个贝叶斯网络相互权衡。我们将第一个网络解释为编码器图，它指定了在压缩数据时要保留的信息。我们将第二个网络解释为解码器图，它为数据指定了一个生成模型。使用这个框架，我们重新推导了现有的降维方法，如深度变分信息瓶颈(DVIB)、beta变分自编码器(beta-VAE)和深度变分规范相关分析(DVCCA)。该框架自然地引入了一个在压缩数据和保留信息之间的权衡参数。

    Variational dimensionality reduction methods are known for their high accuracy, generative abilities, and robustness. These methods have many theoretical justifications. Here we introduce a unifying principle rooted in information theory to rederive and generalize existing variational methods and design new ones. We base our framework on an interpretation of the multivariate information bottleneck, in which two Bayesian networks are traded off against one another. We interpret the first network as an encoder graph, which specifies what information to keep when compressing the data. We interpret the second network as a decoder graph, which specifies a generative model for the data. Using this framework, we rederive existing dimensionality reduction methods such as the deep variational information bottleneck (DVIB), beta variational auto-encoders (beta-VAE), and deep variational canonical correlation analysis (DVCCA). The framework naturally introduces a trade-off parameter between compr
    
[^11]: 潜变量结构方程模型的最大似然估计：一种神经网络方法

    Maximum Likelihood Estimation of Latent Variable Structural Equation Models: A Neural Network Approach. (arXiv:2309.14073v1 [stat.ML])

    [http://arxiv.org/abs/2309.14073](http://arxiv.org/abs/2309.14073)

    本研究提出了一种新的图形结构，用于在线性和高斯性假设下稳定的潜变量结构方程模型。我们证明了计算该模型的最大似然估计等价于训练一个神经网络，并实现了一个基于GPU的算法来进行计算。

    

    我们提出了一种在线性和高斯性假设下稳定的结构方程模型的图形结构。我们展示了计算这个模型的最大似然估计等价于训练一个神经网络。我们实现了一个基于GPU的算法来计算这些模型的最大似然估计。

    We propose a graphical structure for structural equation models that is stable under marginalization under linearity and Gaussianity assumptions. We show that computing the maximum likelihood estimation of this model is equivalent to training a neural network. We implement a GPU-based algorithm that computes the maximum likelihood estimation of these models.
    
[^12]: H2O+: 一种改进的混合离线和在线强化学习框架，用于动力学差距问题

    H2O+: An Improved Framework for Hybrid Offline-and-Online RL with Dynamics Gaps. (arXiv:2309.12716v1 [cs.LG])

    [http://arxiv.org/abs/2309.12716](http://arxiv.org/abs/2309.12716)

    H2O+是一种改进的混合离线和在线强化学习框架，通过综合考虑真实和模拟环境的动力学差距，同时利用有限的离线数据和不完美的模拟器进行策略学习，并在广泛的仿真和实际机器人实验中展示了卓越的性能和灵活性。

    

    在没有高精度模拟环境或大量离线数据的情况下，使用强化学习（RL）解决实际复杂任务可能相当具有挑战性。在非完美模拟环境中训练的在线RL代理可能会受到严重的模拟与现实问题。虽然离线RL方法可以绕过对模拟器的需求，但往往对离线数据集的大小和质量提出了苛刻的要求。最近出现的混合离线和在线RL提供了一个有吸引力的框架，可以同时使用有限的离线数据和不完美的模拟器进行可转移策略学习。本文提出了一种名为H2O+的新算法，该算法在桥接不同的离线和在线学习方法的同时，也考虑了真实和模拟环境之间的动力学差距。通过广泛的仿真和实际机器人实验，我们证明了H2O+在性能和灵活性上优于先进的跨域在线方法

    Solving real-world complex tasks using reinforcement learning (RL) without high-fidelity simulation environments or large amounts of offline data can be quite challenging. Online RL agents trained in imperfect simulation environments can suffer from severe sim-to-real issues. Offline RL approaches although bypass the need for simulators, often pose demanding requirements on the size and quality of the offline datasets. The recently emerged hybrid offline-and-online RL provides an attractive framework that enables joint use of limited offline data and imperfect simulator for transferable policy learning. In this paper, we develop a new algorithm, called H2O+, which offers great flexibility to bridge various choices of offline and online learning methods, while also accounting for dynamics gaps between the real and simulation environment. Through extensive simulation and real-world robotics experiments, we demonstrate superior performance and flexibility over advanced cross-domain online
    

