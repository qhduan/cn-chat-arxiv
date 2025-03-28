# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models](https://arxiv.org/abs/2403.19647) | 该论文介绍了一种新方法，即稀疏特征电路，可以在语言模型中发现和编辑可解释的因果图，为我们提供了对未预料机制的详细理解和包含了用于提高分类器泛化能力的SHIFT方法。 |
| [^2] | [Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning](https://arxiv.org/abs/2403.18886) | 提出了一种名为SEMA的新型微调方法，旨在通过自我扩展预训练模型与模块化适配，实现持续学习过程中的最小遗忘，解决先前针对静态模型架构情况下存在的过多参数分配或适应性不足等问题。 |
| [^3] | [Non-asymptotic Convergence of Discrete-time Diffusion Models: New Approach and Improved Rate](https://arxiv.org/abs/2402.13901) | 本文提出了离散时间扩散模型的新方法，改进了对更大类的分布的收敛保证，并提高了具有有界支撑的分布的收敛速率。 |
| [^4] | [Debiased Offline Representation Learning for Fast Online Adaptation in Non-stationary Dynamics](https://arxiv.org/abs/2402.11317) | 提出了一种名为DORA的新方法，通过信息瓶颈原理在离线设置中学习适应性策略，解决了动态编码与环境数据之间的互信息与与行为策略的互信息之间的难题 |
| [^5] | [Evaluating Membership Inference Attacks and Defenses in Federated Learning](https://arxiv.org/abs/2402.06289) | 这篇论文评估了联邦学习中成员推断攻击和防御的情况。评估揭示了两个重要发现：多时序的模型信息有助于提高攻击的有效性，多空间的模型信息有助于提高攻击的效果。这篇论文还评估了两种防御机制的效用和隐私权衡。 |
| [^6] | [Efficient Solvers for Partial Gromov-Wasserstein](https://arxiv.org/abs/2402.03664) | 本文提出了两个基于Frank-Wolfe算法的新的高效求解器来解决偏差Gromov-Wasserstein问题，并且证明了PGW问题构成了度量测度空间的度量。 |
| [^7] | [Faster Rates for Switchback Experiments](https://arxiv.org/abs/2312.15574) | 本研究提出了一种更快速的Switchback实验方法，通过使用整个时间块，以 $\sqrt{\log T/T}$ 的速率估计全局平均处理效应。 |
| [^8] | [The CTU Prague Relational Learning Repository](https://arxiv.org/abs/1511.03086) | 支持机器学习研究使用多关系数据的布拉格捷克技术大学关系学习资源库，包含大量SQL数据库，并由getML提供支持。 |
| [^9] | [Precise Asymptotic Generalization for Multiclass Classification with Overparameterized Linear Models.](http://arxiv.org/abs/2306.13255) | 本文研究了高斯协变量下的过参数化线性模型在多类分类问题中的泛化能力，成功解决了之前的猜想，并提出的新下界具有信息论中的强对偶定理的性质。 |
| [^10] | [Graph-based Time-Series Anomaly Detection: A Survey.](http://arxiv.org/abs/2302.00058) | 本文综述了基于图的时间序列异常检测，主要探讨了图表示学习的潜力和最先进的图异常检测技术在时间序列中的应用。 |

# 详细

[^1]: 稀疏特征电路：在语言模型中发现和编辑可解释的因果图

    Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models

    [https://arxiv.org/abs/2403.19647](https://arxiv.org/abs/2403.19647)

    该论文介绍了一种新方法，即稀疏特征电路，可以在语言模型中发现和编辑可解释的因果图，为我们提供了对未预料机制的详细理解和包含了用于提高分类器泛化能力的SHIFT方法。

    

    我们介绍了用于发现和应用稀疏特征电路的方法。这些电路是人类可解释特征的因果相关子网络，用于解释语言模型行为。 在先前的工作中确定的电路由多义且难以解释的单元组成，例如注意力头或神经元，使它们不适用于许多下游应用。 相比之下，稀疏特征电路实现了对未预料机制的详细理解。 由于它们基于细粒度单元，稀疏特征电路对下游任务非常有用：我们 introduc了SHIFT，通过切除人类判断为任务不相关的特征，从而提高分类器的泛化能力。 最后，我们通过发现成千上万个稀疏特征电路来展示一个完全无监督且可扩展的可解释性管线，用于自动发现的模型行为。

    arXiv:2403.19647v1 Announce Type: cross  Abstract: We introduce methods for discovering and applying sparse feature circuits. These are causally implicated subnetworks of human-interpretable features for explaining language model behaviors. Circuits identified in prior work consist of polysemantic and difficult-to-interpret units like attention heads or neurons, rendering them unsuitable for many downstream applications. In contrast, sparse feature circuits enable detailed understanding of unanticipated mechanisms. Because they are based on fine-grained units, sparse feature circuits are useful for downstream tasks: We introduce SHIFT, where we improve the generalization of a classifier by ablating features that a human judges to be task-irrelevant. Finally, we demonstrate an entirely unsupervised and scalable interpretability pipeline by discovering thousands of sparse feature circuits for automatically discovered model behaviors.
    
[^2]: 使用混合适配器进行预训练模型的自我扩展以实现持续学习

    Self-Expansion of Pre-trained Models with Mixture of Adapters for Continual Learning

    [https://arxiv.org/abs/2403.18886](https://arxiv.org/abs/2403.18886)

    提出了一种名为SEMA的新型微调方法，旨在通过自我扩展预训练模型与模块化适配，实现持续学习过程中的最小遗忘，解决先前针对静态模型架构情况下存在的过多参数分配或适应性不足等问题。

    

    持续学习旨在从连续到达的数据流中学习，最大限度地减少先前学到的知识的遗忘。本文提出了一种名为SEMA的新型微调方法，称为自我扩展预训练模型与模块化适配，自动决定...（摘要未完整）

    arXiv:2403.18886v1 Announce Type: new  Abstract: Continual learning aims to learn from a stream of continuously arriving data with minimum forgetting of previously learned knowledge. While previous works have explored the effectiveness of leveraging the generalizable knowledge from pre-trained models in continual learning, existing parameter-efficient fine-tuning approaches focus on the use of a predetermined or task-wise set of adapters or prompts. However, these approaches still suffer from forgetting due to task interference on jointly used parameters or restricted flexibility. The reliance on a static model architecture may lead to the allocation of excessive parameters that are not essential or, conversely, inadequate adaptation for downstream tasks, given that the scale and distribution of incoming data are unpredictable in continual learning. We propose Self-Expansion of pre-trained models with Modularized Adaptation (SEMA), a novel fine-tuning approach which automatically decid
    
[^3]: 离散时间扩散模型的非渐近收敛：新方法和改进速率

    Non-asymptotic Convergence of Discrete-time Diffusion Models: New Approach and Improved Rate

    [https://arxiv.org/abs/2402.13901](https://arxiv.org/abs/2402.13901)

    本文提出了离散时间扩散模型的新方法，改进了对更大类的分布的收敛保证，并提高了具有有界支撑的分布的收敛速率。

    

    最近，去噪扩散模型作为一种强大的生成技术出现，将噪声转化为数据。理论上主要研究了连续时间扩散模型的收敛性保证，并且仅在文献中对具有有界支撑的分布的离散时间扩散模型进行了获得。本文为更大类的分布建立了离散时间扩散模型的收敛性保证，并进一步改进了对具有有界支撑的分布的收敛速率。特别地，首先为具有有限二阶矩的平滑和一般（可能非光滑）分布建立了收敛速率。然后将结果专门应用于一些有明确参数依赖关系的有趣分布类别，包括具有Lipschitz分数、高斯混合分布和具有有界支撑的分布。

    arXiv:2402.13901v1 Announce Type: new  Abstract: The denoising diffusion model emerges recently as a powerful generative technique that converts noise into data. Theoretical convergence guarantee has been mainly studied for continuous-time diffusion models, and has been obtained for discrete-time diffusion models only for distributions with bounded support in the literature. In this paper, we establish the convergence guarantee for substantially larger classes of distributions under discrete-time diffusion models and further improve the convergence rate for distributions with bounded support. In particular, we first establish the convergence rates for both smooth and general (possibly non-smooth) distributions having finite second moment. We then specialize our results to a number of interesting classes of distributions with explicit parameter dependencies, including distributions with Lipschitz scores, Gaussian mixture distributions, and distributions with bounded support. We further 
    
[^4]: 针对非静态动态的快速在线调整的去偏置离线表示学习

    Debiased Offline Representation Learning for Fast Online Adaptation in Non-stationary Dynamics

    [https://arxiv.org/abs/2402.11317](https://arxiv.org/abs/2402.11317)

    提出了一种名为DORA的新方法，通过信息瓶颈原理在离线设置中学习适应性策略，解决了动态编码与环境数据之间的互信息与与行为策略的互信息之间的难题

    

    开发能够适应非静态环境的策略对于现实世界的强化学习应用至关重要。然而，在仅有一组有限的预先收集的轨迹的离线设置中学习这种适应性策略存在显著挑战。为了解决这个问题，我们引入了一种名为去偏置离线表示快速在线调整（DORA）的新方法。DORA融入了信息瓶颈原理，最大化了动态编码与环境数据之间的互信息，同时最小化了动态编码与行为策略的互信息。我们提出了DORA的实际实现，利用

    arXiv:2402.11317v1 Announce Type: cross  Abstract: Developing policies that can adjust to non-stationary environments is essential for real-world reinforcement learning applications. However, learning such adaptable policies in offline settings, with only a limited set of pre-collected trajectories, presents significant challenges. A key difficulty arises because the limited offline data makes it hard for the context encoder to differentiate between changes in the environment dynamics and shifts in the behavior policy, often leading to context misassociations. To address this issue, we introduce a novel approach called Debiased Offline Representation for fast online Adaptation (DORA). DORA incorporates an information bottleneck principle that maximizes mutual information between the dynamics encoding and the environmental data, while minimizing mutual information between the dynamics encoding and the actions of the behavior policy. We present a practical implementation of DORA, leverag
    
[^5]: 在联邦学习中评估成员推断攻击和防御

    Evaluating Membership Inference Attacks and Defenses in Federated Learning

    [https://arxiv.org/abs/2402.06289](https://arxiv.org/abs/2402.06289)

    这篇论文评估了联邦学习中成员推断攻击和防御的情况。评估揭示了两个重要发现：多时序的模型信息有助于提高攻击的有效性，多空间的模型信息有助于提高攻击的效果。这篇论文还评估了两种防御机制的效用和隐私权衡。

    

    成员推断攻击(MIAs)对于隐私保护的威胁在联邦学习中日益增长。半诚实的攻击者，例如服务器，可以根据观察到的模型信息确定一个特定样本是否属于目标客户端。本文对现有的MIAs和相应的防御策略进行了评估。我们对MIAs的评估揭示了两个重要发现。首先，结合多个通信轮次的模型信息(多时序)相比于利用单个时期的模型信息提高了MIAs的整体有效性。其次，在非目标客户端(Multi-spatial)中融入模型显著提高了MIAs的效果，特别是当客户端的数据是同质的时候。这凸显了在MIAs中考虑时序和空间模型信息的重要性。接下来，我们通过隐私-效用权衡评估了两种类型的防御机制对MIAs的有效性。

    Membership Inference Attacks (MIAs) pose a growing threat to privacy preservation in federated learning. The semi-honest attacker, e.g., the server, may determine whether a particular sample belongs to a target client according to the observed model information. This paper conducts an evaluation of existing MIAs and corresponding defense strategies. Our evaluation on MIAs reveals two important findings about the trend of MIAs. Firstly, combining model information from multiple communication rounds (Multi-temporal) enhances the overall effectiveness of MIAs compared to utilizing model information from a single epoch. Secondly, incorporating models from non-target clients (Multi-spatial) significantly improves the effectiveness of MIAs, particularly when the clients' data is homogeneous. This highlights the importance of considering the temporal and spatial model information in MIAs. Next, we assess the effectiveness via privacy-utility tradeoff for two type defense mechanisms against MI
    
[^6]: 高效求解偏差Gromov-Wasserstein问题

    Efficient Solvers for Partial Gromov-Wasserstein

    [https://arxiv.org/abs/2402.03664](https://arxiv.org/abs/2402.03664)

    本文提出了两个基于Frank-Wolfe算法的新的高效求解器来解决偏差Gromov-Wasserstein问题，并且证明了PGW问题构成了度量测度空间的度量。

    

    偏差Gromov-Wasserstein（PGW）问题可以比较具有不均匀质量的度量空间中的测度，从而实现这些空间之间的不平衡和部分匹配。本文证明了PGW问题可以转化为Gromov-Wasserstein问题的一个变种，类似于把偏差最优运输问题转化为最优运输问题。这个转化导致了两个新的求解器，基于Frank-Wolfe算法，数学和计算上等价，提供了高效的PGW问题解决方案。我们进一步证明了PGW问题构成了度量测度空间的度量。最后，我们通过与现有基线方法在形状匹配和正样本未标记学习问题上的计算时间和性能比较，验证了我们提出的求解器的有效性。

    The partial Gromov-Wasserstein (PGW) problem facilitates the comparison of measures with unequal masses residing in potentially distinct metric spaces, thereby enabling unbalanced and partial matching across these spaces. In this paper, we demonstrate that the PGW problem can be transformed into a variant of the Gromov-Wasserstein problem, akin to the conversion of the partial optimal transport problem into an optimal transport problem. This transformation leads to two new solvers, mathematically and computationally equivalent, based on the Frank-Wolfe algorithm, that provide efficient solutions to the PGW problem. We further establish that the PGW problem constitutes a metric for metric measure spaces. Finally, we validate the effectiveness of our proposed solvers in terms of computation time and performance on shape-matching and positive-unlabeled learning problems, comparing them against existing baselines.
    
[^7]: 更快速的Switchback实验方法

    Faster Rates for Switchback Experiments

    [https://arxiv.org/abs/2312.15574](https://arxiv.org/abs/2312.15574)

    本研究提出了一种更快速的Switchback实验方法，通过使用整个时间块，以 $\sqrt{\log T/T}$ 的速率估计全局平均处理效应。

    

    Switchback实验设计中，一个单独的单元（例如整个系统）在交替的时间块中暴露于一个随机处理，处理并行处理了跨单元和时间干扰问题。Hu和Wager（2022）最近提出了一种截断块起始的处理效应估计器，并在Markov条件下证明了用于估计全局平均处理效应（GATE）的$T^{-1/3}$速率，他们声称这个速率是最优的，并建议将注意力转向不同（且依赖设计）的估计量，以获得更快的速率。对于相同的设计，我们提出了一种替代估计器，使用整个块，并惊人地证明，在相同的假设下，它实际上达到了原始的设计独立GATE估计量的$\sqrt{\log T/T}$的估计速率。

    Switchback experimental design, wherein a single unit (e.g., a whole system) is exposed to a single random treatment for interspersed blocks of time, tackles both cross-unit and temporal interference. Hu and Wager (2022) recently proposed a treatment-effect estimator that truncates the beginnings of blocks and established a $T^{-1/3}$ rate for estimating the global average treatment effect (GATE) in a Markov setting with rapid mixing. They claim this rate is optimal and suggest focusing instead on a different (and design-dependent) estimand so as to enjoy a faster rate. For the same design we propose an alternative estimator that uses the whole block and surprisingly show that it in fact achieves an estimation rate of $\sqrt{\log T/T}$ for the original design-independent GATE estimand under the same assumptions.
    
[^8]: 布拉格捷克技术大学关系学习资源库

    The CTU Prague Relational Learning Repository

    [https://arxiv.org/abs/1511.03086](https://arxiv.org/abs/1511.03086)

    支持机器学习研究使用多关系数据的布拉格捷克技术大学关系学习资源库，包含大量SQL数据库，并由getML提供支持。

    

    布拉格关系学习资源库的目的是支持具有多关系数据的机器学习研究。此资源库目前包含148个SQL数据库，托管在位于\url{https://relational-data.org}的公共MySQL服务器上。服务器由getML提供，以支持关系机器学习社区（\url{www.getml.com}）。可搜索的元数据库提供元数据（例如数据库中的表数量、表中的行列数、自关联数量）。

    arXiv:1511.03086v2 Announce Type: replace  Abstract: The aim of the Prague Relational Learning Repository is to support machine learning research with multi-relational data. The repository currently contains 148 SQL databases hosted on a public MySQL server located at \url{https://relational-data.org}. The server is provided by getML to support the relational machine learning community (\url{www.getml.com}). A searchable meta-database provides metadata (e.g., the number of tables in the database, the number of rows and columns in the tables, the number of self-relationships).
    
[^9]: 过参数化线性模型下多类分类的渐进泛化精度研究

    Precise Asymptotic Generalization for Multiclass Classification with Overparameterized Linear Models. (arXiv:2306.13255v1 [cs.LG])

    [http://arxiv.org/abs/2306.13255](http://arxiv.org/abs/2306.13255)

    本文研究了高斯协变量下的过参数化线性模型在多类分类问题中的泛化能力，成功解决了之前的猜想，并提出的新下界具有信息论中的强对偶定理的性质。

    

    本文研究了在具有高斯协变量双层模型下，过参数化线性模型在多类分类中的渐进泛化问题，其中数据点数、特征和类别数都同时增长。我们完全解决了Subramanian等人在'22年所提出的猜想，与预测的泛化区间相匹配。此外，我们的新的下界类似于信息论中的强对偶定理：它们能够确立误分类率逐渐趋近于0或1.我们紧密的结果的一个令人惊讶的结果是，最小范数插值分类器在最小范数插值回归器最优的范围内，可以在渐进上次优。我们分析的关键在于一种新的Hanson-Wright不等式变体，该变体在具有稀疏标签的多类问题中具有广泛的适用性。作为应用，我们展示了相同类型分析在几种不同类型的分类模型上的结果。

    We study the asymptotic generalization of an overparameterized linear model for multiclass classification under the Gaussian covariates bi-level model introduced in Subramanian et al.~'22, where the number of data points, features, and classes all grow together. We fully resolve the conjecture posed in Subramanian et al.~'22, matching the predicted regimes for generalization. Furthermore, our new lower bounds are akin to an information-theoretic strong converse: they establish that the misclassification rate goes to 0 or 1 asymptotically. One surprising consequence of our tight results is that the min-norm interpolating classifier can be asymptotically suboptimal relative to noninterpolating classifiers in the regime where the min-norm interpolating regressor is known to be optimal.  The key to our tight analysis is a new variant of the Hanson-Wright inequality which is broadly useful for multiclass problems with sparse labels. As an application, we show that the same type of analysis 
    
[^10]: 基于图的时间序列异常检测：综述(arXiv：2302.00058v2 [cs.LG]更新)

    Graph-based Time-Series Anomaly Detection: A Survey. (arXiv:2302.00058v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.00058](http://arxiv.org/abs/2302.00058)

    本文综述了基于图的时间序列异常检测，主要探讨了图表示学习的潜力和最先进的图异常检测技术在时间序列中的应用。

    

    随着技术的发展，许多系统持续收集大量时间序列数据，如电子商务、网络安全、车辆维护和医疗监测等领域，时间序列异常检测已成为重要的任务。但由于需要同时考虑变量内部和变量间的依赖性，这一任务非常具有挑战性。近年来，基于图的方法在解决该领域的难题方面取得了重要进展。本综述全面而最新地回顾了基于图的时间序列异常检测(G-TSAD)。首先探讨了图表示学习在时间序列数据中的巨大潜力，然后在时间序列背景下回顾了最先进的图异常检测技术，并讨论了它们的优点和缺点。最后，讨论了这些技术如何应用于实际系统中。

    With the recent advances in technology, a wide range of systems continue to collect a large amount of data over time and thus generate time series. Time-Series Anomaly Detection (TSAD) is an important task in various time-series applications such as e-commerce, cybersecurity, vehicle maintenance, and healthcare monitoring. However, this task is very challenging as it requires considering both the intra-variable dependency and the inter-variable dependency, where a variable can be defined as an observation in time series data. Recent graph-based approaches have made impressive progress in tackling the challenges of this field. In this survey, we conduct a comprehensive and up-to-date review of Graph-based TSAD (G-TSAD). First, we explore the significant potential of graph representation learning for time-series data. Then, we review state-of-the-art graph anomaly detection techniques in the context of time series and discuss their strengths and drawbacks. Finally, we discuss the technic
    

