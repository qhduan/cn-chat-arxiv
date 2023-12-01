# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Contextualized Policy Recovery: Modeling and Interpreting Medical Decisions with Adaptive Imitation Learning.](http://arxiv.org/abs/2310.07918) | 本论文提出了一种上下文化政策恢复方法用于建模复杂的医疗决策过程，以解决现有模型在准确性和可解释性之间的权衡问题。该方法将决策策略拆分为上下文特定策略，通过多任务学习来实现建模，并提供复杂行为的简洁描述。 |
| [^2] | [On Double-Descent in Reinforcement Learning with LSTD and Random Features.](http://arxiv.org/abs/2310.05518) | 本文研究了在强化学习中网络大小和L2正则化对性能的影响，并观察到了双下降现象。通过使用随机特征和懒惰训练策略，在参数和状态数无限大的情况下研究了正则化的最小二乘时间差分算法，得出了其收敛性和最优性，并阐述了双下降现象在该算法中的影响。 |
| [^3] | [On Sparse Modern Hopfield Model.](http://arxiv.org/abs/2309.12673) | 本文介绍了稀疏的现代 Hopfield 模型，通过引入稀疏能量函数和稀疏记忆检索动力学，实现了对稀疏注意机制的一步近似。相比密集模型，稀疏模型的记忆检索误差上界更紧凑，具有明确的稀疏优势条件。同时，稀疏的现代 Hopfield 模型还保持了其密集对应物的稳健理论性质。 |
| [^4] | [Generative Models for Anomaly Detection and Design-Space Dimensionality Reduction in Shape Optimization.](http://arxiv.org/abs/2308.04051) | 该论文提出了一种新的形状优化方法，通过降低设计空间维度和建模数据的生成过程，实现了提高全局优化算法效率和生成无几何异常的高质量设计。 |
| [^5] | [Fantastic Weights and How to Find Them: Where to Prune in Dynamic Sparse Training.](http://arxiv.org/abs/2306.12230) | 该论文对动态稀疏训练中的剪枝位置进行了实证分析，发现在低密度范围内，最简单的大小方法提供了最佳性能。 |
| [^6] | [Exploring the Carbon Footprint of Hugging Face's ML Models: A Repository Mining Study.](http://arxiv.org/abs/2305.11164) | 本论文通过分析Hugging Face上1,417个ML模型及相关数据集的碳足迹测量情况，提出了有关如何报告和优化ML模型的碳效率的见解和建议。 |
| [^7] | [Discovering Communication Pattern Shifts in Large-Scale Networks using Encoder Embedding and Vertex Dynamics.](http://arxiv.org/abs/2305.02381) | 本文提出了一种新的方法称为“时间编码嵌入”，可以以线性复杂度有效地嵌入大量图数据，并利用此方法在大型组织的通信网络中检测出了个体顶点、顶点社区和整体图结构的通信模式变化。 |
| [^8] | [Some Fundamental Aspects about Lipschitz Continuity of Neural Network Functions.](http://arxiv.org/abs/2302.10886) | 本文深入研究和描述神经网络实现的函数的Lipschitz行为，在多种设置下进行实证研究，并揭示了神经网络函数Lipschitz连续性的基本和有趣的特性，其中最引人注目的是在Lipschitz常数的上限和下限中识别出了明显的双下降趋势。 |
| [^9] | [A survey on online active learning.](http://arxiv.org/abs/2302.08893) | 在线主动学习是一种机器学习范式，旨在从数据流中选择最具信息量的数据点进行标注。本文综述了在线主动学习的最新进展、基于流的主动学习的主要挑战和机遇、用于选择信息样本的策略以及该范式中不同的评估指标。 |
| [^10] | [On the Convergence of the ELBO to Entropy Sums.](http://arxiv.org/abs/2209.03077) | 本论文研究了 ELBO 收敛到熵和的问题，证明了对于一类广泛的生成模型，ELBO 在所有学习的稳定点处都等于一系列熵的和，为无监督学习的学习算法的基本属性提供了深入的洞察。 |

# 详细

[^1]: 上下文化政策恢复：通过自适应模仿学习对医疗决策进行建模和解释

    Contextualized Policy Recovery: Modeling and Interpreting Medical Decisions with Adaptive Imitation Learning. (arXiv:2310.07918v1 [cs.LG])

    [http://arxiv.org/abs/2310.07918](http://arxiv.org/abs/2310.07918)

    本论文提出了一种上下文化政策恢复方法用于建模复杂的医疗决策过程，以解决现有模型在准确性和可解释性之间的权衡问题。该方法将决策策略拆分为上下文特定策略，通过多任务学习来实现建模，并提供复杂行为的简洁描述。

    

    可解释的策略学习旨在从观察到的行为中估计可理解的决策策略；然而，现有模型在准确性和可解释性之间存在权衡。这种权衡限制了基于数据驱动的对人类决策过程的解释，例如，审计医疗决策的偏见和次优实践，我们需要决策过程的模型，能够提供复杂行为的简洁描述。现有方法基本上由于将潜在决策过程表示为通用策略而负担了这种权衡，而实际上人类决策是动态的，可以随上下文信息而大幅改变。因此，我们提出了上下文化政策恢复（CPR），将建模复杂决策过程的问题重新定义为多任务学习问题，其中复杂决策策略由特定上下文的策略组成。CPR将每个上下文特定策略建模为线性的观察-动作映射

    Interpretable policy learning seeks to estimate intelligible decision policies from observed actions; however, existing models fall short by forcing a tradeoff between accuracy and interpretability. This tradeoff limits data-driven interpretations of human decision-making process. e.g. to audit medical decisions for biases and suboptimal practices, we require models of decision processes which provide concise descriptions of complex behaviors. Fundamentally, existing approaches are burdened by this tradeoff because they represent the underlying decision process as a universal policy, when in fact human decisions are dynamic and can change drastically with contextual information. Thus, we propose Contextualized Policy Recovery (CPR), which re-frames the problem of modeling complex decision processes as a multi-task learning problem in which complex decision policies are comprised of context-specific policies. CPR models each context-specific policy as a linear observation-to-action mapp
    
[^2]: 关于使用LSTD和随机特征的强化学习中的双下降现象

    On Double-Descent in Reinforcement Learning with LSTD and Random Features. (arXiv:2310.05518v1 [cs.LG] CROSS LISTED)

    [http://arxiv.org/abs/2310.05518](http://arxiv.org/abs/2310.05518)

    本文研究了在强化学习中网络大小和L2正则化对性能的影响，并观察到了双下降现象。通过使用随机特征和懒惰训练策略，在参数和状态数无限大的情况下研究了正则化的最小二乘时间差分算法，得出了其收敛性和最优性，并阐述了双下降现象在该算法中的影响。

    

    时间差分算法在深度强化学习中被广泛使用，其性能受神经网络大小的影响。然而，在监督学习中过参数化和其带来的好处已经得到了很好的理解，但是在强化学习中情况则不太清楚。本文通过理论分析探讨了网络大小和L2正则化对性能的影响，并将参数个数与访问状态个数之比定义为关键因素，当该比值大于1时称为过参数化。此外，我们观察到了双下降现象，即在参数/状态比为1附近会突然性能下降。通过利用随机特征和懒惰训练策略，我们在无限大的参数和状态数下研究了正则化的最小二乘时间差分算法。我们推导了其收敛性和最优性，并阐述了双下降现象在该算法中的影响。

    Temporal Difference (TD) algorithms are widely used in Deep Reinforcement Learning (RL). Their performance is heavily influenced by the size of the neural network. While in supervised learning, the regime of over-parameterization and its benefits are well understood, the situation in RL is much less clear. In this paper, we present a theoretical analysis of the influence of network size and $l_2$-regularization on performance. We identify the ratio between the number of parameters and the number of visited states as a crucial factor and define over-parameterization as the regime when it is larger than one. Furthermore, we observe a double-descent phenomenon, i.e., a sudden drop in performance around the parameter/state ratio of one. Leveraging random features and the lazy training regime, we study the regularized Least-Square Temporal Difference (LSTD) algorithm in an asymptotic regime, as both the number of parameters and states go to infinity, maintaining a constant ratio. We derive 
    
[^3]: 关于稀疏的现代 Hopfield 模型

    On Sparse Modern Hopfield Model. (arXiv:2309.12673v1 [cs.LG])

    [http://arxiv.org/abs/2309.12673](http://arxiv.org/abs/2309.12673)

    本文介绍了稀疏的现代 Hopfield 模型，通过引入稀疏能量函数和稀疏记忆检索动力学，实现了对稀疏注意机制的一步近似。相比密集模型，稀疏模型的记忆检索误差上界更紧凑，具有明确的稀疏优势条件。同时，稀疏的现代 Hopfield 模型还保持了其密集对应物的稳健理论性质。

    

    我们介绍了稀疏的现代 Hopfield 模型作为现代 Hopfield 模型的一种扩展。与其密集的对应物一样，稀疏的现代 Hopfield 模型具备一种记忆检索动力学，其一步近似对应于稀疏的注意机制。从理论上讲，我们的关键贡献是通过稀疏熵正则化器的凸共轭导出了封闭形式的稀疏 Hopfield 能量。在此基础上，我们从稀疏能量函数中推导出稀疏记忆检索动力学，并展示了它的一步近似等价于稀疏结构化注意力。重要的是，我们提供了一个依赖于稀疏度的记忆检索误差上界，该上界在证明上要比其密集对应物更紧凑。因此，我们确定并讨论了稀疏优势出现的条件。此外，我们还表明稀疏的现代 Hopfield 模型保持了其密集对应物的稳健理论性质，包括快速的固定点收敛。

    We introduce the sparse modern Hopfield model as a sparse extension of the modern Hopfield model. Like its dense counterpart, the sparse modern Hopfield model equips a memory-retrieval dynamics whose one-step approximation corresponds to the sparse attention mechanism. Theoretically, our key contribution is a principled derivation of a closed-form sparse Hopfield energy using the convex conjugate of the sparse entropic regularizer. Building upon this, we derive the sparse memory retrieval dynamics from the sparse energy function and show its one-step approximation is equivalent to the sparse-structured attention. Importantly, we provide a sparsity-dependent memory retrieval error bound which is provably tighter than its dense analog. The conditions for the benefits of sparsity to arise are therefore identified and discussed. In addition, we show that the sparse modern Hopfield model maintains the robust theoretical properties of its dense counterpart, including rapid fixed point conver
    
[^4]: 形状优化中的异常检测和设计空间维度降低的生成模型

    Generative Models for Anomaly Detection and Design-Space Dimensionality Reduction in Shape Optimization. (arXiv:2308.04051v1 [stat.ML])

    [http://arxiv.org/abs/2308.04051](http://arxiv.org/abs/2308.04051)

    该论文提出了一种新的形状优化方法，通过降低设计空间维度和建模数据的生成过程，实现了提高全局优化算法效率和生成无几何异常的高质量设计。

    

    我们的工作提出了一种新颖的形状优化方法，其两个目标是提高全局优化算法的效率，同时在优化过程中生成没有几何异常的高质量设计。通过减少定义新的减少子空间的原始设计变量的数量，并使用概率线性潜变量模型来建模数据的底层生成过程，如因子分析和概率主成分分析，来实现这一目标。我们展示了当形状修改方法是线性的且设计变量在均匀随机采样时，数据近似服从高斯分布，这是由于直接应用了中心极限定理。利用马氏距离来衡量模型不确定性，并且论文证明异常设计往往具有较高的该度量值。

    Our work presents a novel approach to shape optimization, that has the twofold objective to improve the efficiency of global optimization algorithms while promoting the generation of high-quality designs during the optimization process free of geometrical anomalies. This is accomplished by reducing the number of the original design variables defining a new reduced subspace where the geometrical variance is maximized and modeling the underlying generative process of the data via probabilistic linear latent variable models such as Factor Analysis and Probabilistic Principal Component Analysis. We show that the data follows approximately a Gaussian distribution when the shape modification method is linear and the design variables are sampled uniformly at random, due to the direct application of the central limit theorem. The model uncertainty is measured in terms of Mahalanobis distance, and the paper demonstrates that anomalous designs tend to exhibit a high value of this metric. This en
    
[^5]: 奇妙的权重及其查找方法：动态稀疏训练中的剪枝位置

    Fantastic Weights and How to Find Them: Where to Prune in Dynamic Sparse Training. (arXiv:2306.12230v1 [cs.LG])

    [http://arxiv.org/abs/2306.12230](http://arxiv.org/abs/2306.12230)

    该论文对动态稀疏训练中的剪枝位置进行了实证分析，发现在低密度范围内，最简单的大小方法提供了最佳性能。

    

    动态稀疏训练（DST）是一个快速发展的研究领域，旨在通过在训练过程中调整神经网络的拓扑结构来优化其稀疏初始化。已经证明，在特定条件下，DST能够胜过密集模型。该框架的关键组成部分是剪枝和生长标准，这些标准在训练过程中被反复应用以调整网络的稀疏连接。虽然生长标准对DST性能的影响相对较好地研究了，但剪枝标准的影响仍然被忽视。为解决这个问题，我们设计并进行了对各种剪枝标准的广泛实证分析，以更好地了解它们对 DST 解决方案动态的影响。令人惊讶的是，我们发现大多数研究方法都产生类似的结果。在低密度范围内，最简单的技术——基于大小的方法，提供了最佳性能。

    Dynamic Sparse Training (DST) is a rapidly evolving area of research that seeks to optimize the sparse initialization of a neural network by adapting its topology during training. It has been shown that under specific conditions, DST is able to outperform dense models. The key components of this framework are the pruning and growing criteria, which are repeatedly applied during the training process to adjust the network's sparse connectivity. While the growing criterion's impact on DST performance is relatively well studied, the influence of the pruning criterion remains overlooked. To address this issue, we design and perform an extensive empirical analysis of various pruning criteria to better understand their effect on the dynamics of DST solutions. Surprisingly, we find that most of the studied methods yield similar results. The differences become more significant in the low-density regime, where the best performance is predominantly given by the simplest technique: magnitude-based
    
[^6]: 探索抱抱脸ML模型的碳足迹：一项存储库挖掘研究

    Exploring the Carbon Footprint of Hugging Face's ML Models: A Repository Mining Study. (arXiv:2305.11164v1 [cs.LG])

    [http://arxiv.org/abs/2305.11164](http://arxiv.org/abs/2305.11164)

    本论文通过分析Hugging Face上1,417个ML模型及相关数据集的碳足迹测量情况，提出了有关如何报告和优化ML模型的碳效率的见解和建议。

    

    机器学习(ML)系统的崛起加剧了它们的碳足迹，这是由于其增加的能力和模型大小所致。然而，目前对ML模型的碳足迹如何实际测量、报告和评估的认识相对较少。因此，本论文旨在分析在Hugging Face上1,417个ML模型和相关数据集的碳足迹测量情况，Hugging Face是最受欢迎的预训练ML模型的存储库。目标是提供有关如何报告和优化ML模型的碳效率的见解和建议。该研究包括Hugging Face Hub API上有关碳排放的第一项存储库挖掘研究。本研究旨在回答两个研究问题：(1) ML模型的创建者如何在Hugging Face Hub上测量和报告碳排放？(2) 哪些方面影响了训练ML模型的碳排放？该研究得出了几个关键发现。其中包括碳排放报告模式比例的逐步下降等。

    The rise of machine learning (ML) systems has exacerbated their carbon footprint due to increased capabilities and model sizes. However, there is scarce knowledge on how the carbon footprint of ML models is actually measured, reported, and evaluated. In light of this, the paper aims to analyze the measurement of the carbon footprint of 1,417 ML models and associated datasets on Hugging Face, which is the most popular repository for pretrained ML models. The goal is to provide insights and recommendations on how to report and optimize the carbon efficiency of ML models. The study includes the first repository mining study on the Hugging Face Hub API on carbon emissions. This study seeks to answer two research questions: (1) how do ML model creators measure and report carbon emissions on Hugging Face Hub?, and (2) what aspects impact the carbon emissions of training ML models? The study yielded several key findings. These include a decreasing proportion of carbon emissions-reporting mode
    
[^7]: 利用编码嵌入和顶点动态发现大规模网络中的通信模式变化

    Discovering Communication Pattern Shifts in Large-Scale Networks using Encoder Embedding and Vertex Dynamics. (arXiv:2305.02381v1 [cs.SI])

    [http://arxiv.org/abs/2305.02381](http://arxiv.org/abs/2305.02381)

    本文提出了一种新的方法称为“时间编码嵌入”，可以以线性复杂度有效地嵌入大量图数据，并利用此方法在大型组织的通信网络中检测出了个体顶点、顶点社区和整体图结构的通信模式变化。

    

    分析大规模时间序列网络数据（如社交媒体和电子邮件通信）仍然是图分析方法学面临的重大挑战。本文介绍了一种称为“时间编码嵌入”的新方法，可以以线性复杂度有效地嵌入大量的图数据。我们将该方法应用于一家大型机构跨越2019年至2020年的匿名时间序列通信网络，由超过10万个顶点和8000万个边组成。我们的方法在标准计算机上仅需10秒即可嵌入数据，并能够检测个体顶点、顶点社区和整体图结构的通信模式变化。通过支持理论和综合研究，我们证明了我们的方法在随机图模型下的理论健全性和数值效果。

    The analysis of large-scale time-series network data, such as social media and email communications, remains a significant challenge for graph analysis methodology. In particular, the scalability of graph analysis is a critical issue hindering further progress in large-scale downstream inference. In this paper, we introduce a novel approach called "temporal encoder embedding" that can efficiently embed large amounts of graph data with linear complexity. We apply this method to an anonymized time-series communication network from a large organization spanning 2019-2020, consisting of over 100 thousand vertices and 80 million edges. Our method embeds the data within 10 seconds on a standard computer and enables the detection of communication pattern shifts for individual vertices, vertex communities, and the overall graph structure. Through supporting theory and synthesis studies, we demonstrate the theoretical soundness of our approach under random graph models and its numerical effecti
    
[^8]: 神经网络函数的Lipschitz连续性的一些基本方面

    Some Fundamental Aspects about Lipschitz Continuity of Neural Network Functions. (arXiv:2302.10886v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.10886](http://arxiv.org/abs/2302.10886)

    本文深入研究和描述神经网络实现的函数的Lipschitz行为，在多种设置下进行实证研究，并揭示了神经网络函数Lipschitz连续性的基本和有趣的特性，其中最引人注目的是在Lipschitz常数的上限和下限中识别出了明显的双下降趋势。

    

    Lipschitz连续性是任何预测模型的一个简单但关键的功能性质，它处于模型的稳健性、泛化性和对抗性脆弱性的核心。本文旨在深入研究和描述神经网络实现的函数的Lipschitz行为。因此，我们通过耗尽最简单和最一般的下限和上限的极限，在各种不同设置下进行实证研究（即，体系结构、损失、优化器、标签噪音等），虽然这一选择主要是受计算难度结果的驱动，但它也非常丰富，并揭示了神经网络函数Lipschitz连续性的几个基本和有趣的特性，我们还补充了适当的理论论证。

    Lipschitz continuity is a simple yet crucial functional property of any predictive model for it lies at the core of the model's robustness, generalisation, as well as adversarial vulnerability. Our aim is to thoroughly investigate and characterise the Lipschitz behaviour of the functions realised by neural networks. Thus, we carry out an empirical investigation in a range of different settings (namely, architectures, losses, optimisers, label noise, and more) by exhausting the limits of the simplest and the most general lower and upper bounds. Although motivated primarily by computational hardness results, this choice nevertheless turns out to be rather resourceful and sheds light on several fundamental and intriguing traits of the Lipschitz continuity of neural network functions, which we also supplement with suitable theoretical arguments. As a highlight of this investigation, we identify a striking double descent trend in both upper and lower bounds to the Lipschitz constant with in
    
[^9]: 在线主动学习综述

    A survey on online active learning. (arXiv:2302.08893v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.08893](http://arxiv.org/abs/2302.08893)

    在线主动学习是一种机器学习范式，旨在从数据流中选择最具信息量的数据点进行标注。本文综述了在线主动学习的最新进展、基于流的主动学习的主要挑战和机遇、用于选择信息样本的策略以及该范式中不同的评估指标。

    

    在线主动学习是一种机器学习范式，旨在从数据流中选择最具信息量的数据点进行标注。近年来，随着数据仅以未标记形式可用的实际应用日益增多，最小化与收集标记观测相关的成本问题引起了广泛关注。标注每个观测可以耗费大量的时间和成本，使得获取大量标记数据变得困难。为了克服这个问题，许多主动学习策略已经提出，旨在选择最具信息量的观测进行标记，以提高机器学习模型的性能。这些方法可以广泛地分为两类：静态基于池的和基于流的主动学习。基于池的主动学习涉及从封闭的未标记数据池中选择一部分观测，已成为许多调查和文献综述的重点。然而，随着在线数据流的不断增加，基于流的主动学习策略变得更加吸引人，因为它们允许模型适应新进数据。在本综述中，我们综述了在线主动学习的最新进展，讨论了基于流的主动学习的主要挑战和机遇、用于选择信息样本的策略以及该范例中不同的评估指标。

    Online active learning is a paradigm in machine learning that aims to select the most informative data points to label from a data stream. The problem of minimizing the cost associated with collecting labeled observations has gained a lot of attention in recent years, particularly in real-world applications where data is only available in an unlabeled form. Annotating each observation can be time-consuming and costly, making it difficult to obtain large amounts of labeled data. To overcome this issue, many active learning strategies have been proposed in the last decades, aiming to select the most informative observations for labeling in order to improve the performance of machine learning models. These approaches can be broadly divided into two categories: static pool-based and stream-based active learning. Pool-based active learning involves selecting a subset of observations from a closed pool of unlabeled data, and it has been the focus of many surveys and literature reviews. Howev
    
[^10]: 关于ELBO收敛到熵和的研究

    On the Convergence of the ELBO to Entropy Sums. (arXiv:2209.03077v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2209.03077](http://arxiv.org/abs/2209.03077)

    本论文研究了 ELBO 收敛到熵和的问题，证明了对于一类广泛的生成模型，ELBO 在所有学习的稳定点处都等于一系列熵的和，为无监督学习的学习算法的基本属性提供了深入的洞察。

    

    变分下界（又称ELBO或自由能）是许多经典和新颖的无监督学习算法的核心目标。学习算法可以改变模型参数，使变分下界增加。通常，学习进行到参数收敛到接近学习动态的稳定点值。在本文的理论贡献中，我们证明了（对于一类非常广泛的生成模型），变分下界在所有学习的稳定点处均等于一系列熵的和。对于具有一组潜在变量和一组观测变量的标准机器学习模型，这个和包括三个熵: (A) 变分分布的熵（平均熵），(B) 模型先验分布的负熵和 (C) 可观测分布的（期望）负熵。所得到的结果适用于包括：有限数量的数据点，在学习的任意阶段和各种不同的生成模型等真实条件。本研究为无监督学习的学习算法的基本属性提供了深入洞察，是对优化推理和学习的理论分析的第一步。

    The variational lower bound (a.k.a. ELBO or free energy) is the central objective for many established as well as many novel algorithms for unsupervised learning. Learning algorithms change model parameters such that the variational lower bound increases. Learning usually proceeds until parameters have converged to values close to a stationary point of the learning dynamics. In this purely theoretical contribution, we show that (for a very large class of generative models) the variational lower bound is at all stationary points of learning equal to a sum of entropies. For standard machine learning models with one set of latents and one set observed variables, the sum consists of three entropies: (A) the (average) entropy of the variational distributions, (B) the negative entropy of the model's prior distribution, and (C) the (expected) negative entropy of the observable distributions. The obtained result applies under realistic conditions including: finite numbers of data points, at an
    

