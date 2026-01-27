# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [The Double-Edged Sword of Input Perturbations to Robust Accurate Fairness](https://arxiv.org/abs/2404.01356) | 该论文研究了深度神经网络对敌对输入扰动的敏感性，提出了新的鲁棒准确公平性定义，并介绍了一种敌对攻击方法和相应的解决方案。 |
| [^2] | [Sound event localization and classification using WASN in Outdoor Environment](https://arxiv.org/abs/2403.20130) | 本文提出了一种基于深度学习的方法，利用多种特征和注意机制来估计声源的位置和类别，包括引入"声音地图"特征、使用Gammatone滤波器生成更适合室外环境的声学特征，以及集成注意机制来学习通道间的关系。 |
| [^3] | [Theoretical Hardness and Tractability of POMDPs in RL with Partial Online State Information.](http://arxiv.org/abs/2306.08762) | 本论文研究了具有部分在线状态信息的POMDP问题的理论困难性和可计算性。作者通过建立下界得出一个惊人的难度结果：除非具有完整的在线状态信息，否则需要指数级的样本复杂度才能得到POMDP的最优策略解。然而，作者还发现了具有部分在线状态信息下的可计算POMDP类别，并提出了新的算法来证明其接近最优性。 |
| [^4] | [Efficient Asynchronize Stochastic Gradient Algorithm with Structured Data.](http://arxiv.org/abs/2305.08001) | 本论文针对具有 Kronecker 结构的训练数据，提出了一种高效的异步随机梯度算法，可以在数据维度的次线性时间内完成每次迭代。 |
| [^5] | [GFlowNet Foundations.](http://arxiv.org/abs/2111.09266) | GFlowNets是一种生成流网络方法，用于在主动学习环境中采样多样化的候选集。它们具有估计联合概率分布和边际分布的能力，可以表示关于复合对象（如集合和图）的分布。通过单次训练的生成传递，GFlowNets分摊了计算昂贵的MCMC方法的工作。 |

# 详细

[^1]: 输入扰动对鲁棒准确公平性的双刃剑

    The Double-Edged Sword of Input Perturbations to Robust Accurate Fairness

    [https://arxiv.org/abs/2404.01356](https://arxiv.org/abs/2404.01356)

    该论文研究了深度神经网络对敌对输入扰动的敏感性，提出了新的鲁棒准确公平性定义，并介绍了一种敌对攻击方法和相应的解决方案。

    

    深度神经网络(DNNs)被认为对敌对输入扰动敏感，导致预测的准确性或个体公平性降低。为了共同表征预测准确性和个体公平性对敌对扰动的敏感性，我们引入了一个名为鲁棒准确公平性的新定义。鲁棒准确公平性要求当实例及其相似对应物受到输入扰动时，预测与地面事实一致。我们提出一种敌对攻击方法RAFair，以暴露DNN中的虚假或偏见敌对缺陷，这些缺陷会欺骗准确性或损害个体公平性。然后，我们展示这样的敌对实例可以通过精心设计的良性扰动有效地解决，从而使它们的预测准确而公平。我们的工作探讨了输入对准确公平性的双刃剑。

    arXiv:2404.01356v1 Announce Type: cross  Abstract: Deep neural networks (DNNs) are known to be sensitive to adversarial input perturbations, leading to a reduction in either prediction accuracy or individual fairness. To jointly characterize the susceptibility of prediction accuracy and individual fairness to adversarial perturbations, we introduce a novel robustness definition termed robust accurate fairness. Informally, robust accurate fairness requires that predictions for an instance and its similar counterparts consistently align with the ground truth when subjected to input perturbations. We propose an adversarial attack approach dubbed RAFair to expose false or biased adversarial defects in DNN, which either deceive accuracy or compromise individual fairness. Then, we show that such adversarial instances can be effectively addressed by carefully designed benign perturbations, correcting their predictions to be accurate and fair. Our work explores the double-edged sword of input 
    
[^2]: 在室外环境中使用WASN进行声事件定位和分类

    Sound event localization and classification using WASN in Outdoor Environment

    [https://arxiv.org/abs/2403.20130](https://arxiv.org/abs/2403.20130)

    本文提出了一种基于深度学习的方法，利用多种特征和注意机制来估计声源的位置和类别，包括引入"声音地图"特征、使用Gammatone滤波器生成更适合室外环境的声学特征，以及集成注意机制来学习通道间的关系。

    

    基于深度学习的声事件定位和分类是无线声学传感器网络中的新兴研究领域。然而，当前的声事件定位和分类方法通常依赖于单个麦克风阵列，容易受到信号衰减和环境噪音的影响，这限制了它们的监测范围。此外，使用多个麦克风阵列的方法通常只关注源定位，忽略了声事件分类方面。在本文中，我们提出了一种基于深度学习的方法，利用多种特征和注意机制来估计声源的位置和类别。我们引入了"声音地图"特征，以捕获多个频段的空间信息。我们还使用Gammatone滤波器生成更适合室外环境的声学特征。此外，我们集成了注意机制来学习通道间的关系。

    arXiv:2403.20130v1 Announce Type: cross  Abstract: Deep learning-based sound event localization and classification is an emerging research area within wireless acoustic sensor networks. However, current methods for sound event localization and classification typically rely on a single microphone array, making them susceptible to signal attenuation and environmental noise, which limits their monitoring range. Moreover, methods using multiple microphone arrays often focus solely on source localization, neglecting the aspect of sound event classification. In this paper, we propose a deep learning-based method that employs multiple features and attention mechanisms to estimate the location and class of sound source. We introduce a Soundmap feature to capture spatial information across multiple frequency bands. We also use the Gammatone filter to generate acoustic features more suitable for outdoor environments. Furthermore, we integrate attention mechanisms to learn channel-wise relationsh
    
[^3]: 在具有部分在线状态信息的强化学习中，POMDP的理论难度和可计算性

    Theoretical Hardness and Tractability of POMDPs in RL with Partial Online State Information. (arXiv:2306.08762v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2306.08762](http://arxiv.org/abs/2306.08762)

    本论文研究了具有部分在线状态信息的POMDP问题的理论困难性和可计算性。作者通过建立下界得出一个惊人的难度结果：除非具有完整的在线状态信息，否则需要指数级的样本复杂度才能得到POMDP的最优策略解。然而，作者还发现了具有部分在线状态信息下的可计算POMDP类别，并提出了新的算法来证明其接近最优性。

    

    部分可观察的马尔可夫决策过程（POMDP）被广泛应用于捕捉许多现实世界的应用。然而，现有的理论结果已经表明，在一般的POMDP中学习可能是不可计算的，主要挑战在于缺乏潜在的状态信息。一个关键的基本问题是有多少在线状态信息（OSI）足以实现可计算性。在本文中，我们建立了一个下界，揭示了一个惊人的难度结果：除非我们具有完整的OSI，否则我们需要指数级的采样复杂度才能获得POMDP的$\epsilon$-最优策略解。尽管如此，受到我们下界设计的关键见解的启发，我们发现即使只有部分OSI，也存在重要的可计算的POMDP类别。特别地，对于具有部分OSI的两个新颖的POMDP类别，我们通过建立新的遗憾上下界证明了新的算法是接近最优的。

    Partially observable Markov decision processes (POMDPs) have been widely applied to capture many real-world applications. However, existing theoretical results have shown that learning in general POMDPs could be intractable, where the main challenge lies in the lack of latent state information. A key fundamental question here is how much online state information (OSI) is sufficient to achieve tractability. In this paper, we establish a lower bound that reveals a surprising hardness result: unless we have full OSI, we need an exponentially scaling sample complexity to obtain an $\epsilon$-optimal policy solution for POMDPs. Nonetheless, inspired by the key insights in our lower bound design, we find that there exist important tractable classes of POMDPs even with only partial OSI. In particular, for two novel classes of POMDPs with partial OSI, we provide new algorithms that are proved to be near-optimal by establishing new regret upper and lower bounds.
    
[^4]: 具有结构化数据的高效异步随机梯度算法

    Efficient Asynchronize Stochastic Gradient Algorithm with Structured Data. (arXiv:2305.08001v1 [cs.LG])

    [http://arxiv.org/abs/2305.08001](http://arxiv.org/abs/2305.08001)

    本论文针对具有 Kronecker 结构的训练数据，提出了一种高效的异步随机梯度算法，可以在数据维度的次线性时间内完成每次迭代。

    

    深度学习因其良好的泛化而在许多领域取得了显著的成功。但是，快速训练具有大量层数的神经网络一直是一个具有挑战性的问题。现有的研究利用局部敏感哈希技术或某些数据结构的空间划分来减轻每次迭代的训练成本。在本研究中，我们尝试从输入数据点的角度加速每次迭代中的计算。具体而言，针对一个两层全连接神经网络，当训练数据具有一些特殊属性，例如 Kronecker 结构时，每次迭代可以在数据维度的次线性时间内完成。

    Deep learning has achieved impressive success in a variety of fields because of its good generalization. However, it has been a challenging problem to quickly train a neural network with a large number of layers. The existing works utilize the locality-sensitive hashing technique or some data structures on space partitioning to alleviate the training cost in each iteration. In this work, we try accelerating the computations in each iteration from the perspective of input data points. Specifically, for a two-layer fully connected neural network, when the training data have some special properties, e.g., Kronecker structure, each iteration can be completed in sublinear time in the data dimension.
    
[^5]: GFlowNet基础

    GFlowNet Foundations. (arXiv:2111.09266v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2111.09266](http://arxiv.org/abs/2111.09266)

    GFlowNets是一种生成流网络方法，用于在主动学习环境中采样多样化的候选集。它们具有估计联合概率分布和边际分布的能力，可以表示关于复合对象（如集合和图）的分布。通过单次训练的生成传递，GFlowNets分摊了计算昂贵的MCMC方法的工作。

    

    生成流网络（GFlowNets）被引入为在主动学习环境中采样多样化的候选集的方法，其训练目标使其近似按照给定的奖励函数进行采样。本文展示了GFlowNets的一些额外的理论性质。它们可以用于估计联合概率分布和相应的边际分布，其中一些变量未指定，特别是可以表示关于复合对象（如集合和图）的分布。GFlowNets通过单次训练的生成传递来分摊通常由计算昂贵的MCMC方法完成的工作。它们还可以用于估计分区函数和自由能，给定一个子集（子图）的超集（超图）的条件概率，以及给定一个集合（图）的所有超集（超图）的边际分布。我们介绍了一些变体，使得可以估计熵的值。

    Generative Flow Networks (GFlowNets) have been introduced as a method to sample a diverse set of candidates in an active learning context, with a training objective that makes them approximately sample in proportion to a given reward function. In this paper, we show a number of additional theoretical properties of GFlowNets. They can be used to estimate joint probability distributions and the corresponding marginal distributions where some variables are unspecified and, of particular interest, can represent distributions over composite objects like sets and graphs. GFlowNets amortize the work typically done by computationally expensive MCMC methods in a single but trained generative pass. They could also be used to estimate partition functions and free energies, conditional probabilities of supersets (supergraphs) given a subset (subgraph), as well as marginal distributions over all supersets (supergraphs) of a given set (graph). We introduce variations enabling the estimation of entro
    

