# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Learning for Optimal Dynamic Treatment Regimes with Observational Data](https://arxiv.org/abs/2404.00221) | 学习利用观测数据提出了一种逐步双重强健方法，通过向后归纳解决了最佳动态治疗方案的问题 |
| [^2] | [AdaTrans: Feature-wise and Sample-wise Adaptive Transfer Learning for High-dimensional Regression](https://arxiv.org/abs/2403.13565) | 提出了一种针对高维回归的自适应迁移学习方法，可以根据可迁移结构自适应检测和聚合特征和样本的可迁移结构。 |
| [^3] | [Interpretable Machine Learning for Survival Analysis](https://arxiv.org/abs/2403.10250) | 可解释的机器学习在生存分析中的应用促进了透明度和公平性，揭示了模型的潜在偏见和限制，并提供了更符合数学原理的特征影响和风险因素预测方法。 |
| [^4] | [When Your AI Deceives You: Challenges with Partial Observability of Human Evaluators in Reward Learning](https://arxiv.org/abs/2402.17747) | RLHF在考虑部分观察性时可能导致策略欺骗性地夸大性能或过度辩护行为，我们提出了数学条件来解决这些问题，并警告不要盲目应用RLHF在部分可观测情况下。 |
| [^5] | [Gradient descent induces alignment between weights and the empirical NTK for deep non-linear networks](https://arxiv.org/abs/2402.05271) | 了解神经网络从输入-标签对中提取统计信息的机制是监督学习中最重要的未解决问题之一。前人的研究表明，在训练过程中，权重的格拉姆矩阵与模型的平均梯度外积成正比，这被称为神经特征分析（NFA）。本研究解释了这种相关性的出现，并发现NFA等价于权重矩阵的左奇异结构与与这些权重相关的经验神经切线核的显著成分之间的对齐。在早期训练阶段，可以通过解析的方式预测NFA的发展速度。 |
| [^6] | [Interpretable Multi-Source Data Fusion Through Latent Variable Gaussian Process](https://arxiv.org/abs/2402.04146) | 这篇论文提出了一种基于潜变量高斯过程的多源数据融合框架，用于解决多个数据源之间质量和全面性差异给系统优化带来的问题。 |
| [^7] | [Bayesian Quantile Regression with Subset Selection: A Posterior Summarization Perspective.](http://arxiv.org/abs/2311.02043) | 本研究提出了一种基于贝叶斯决策分析的方法，对于任何贝叶斯回归模型，可以得到每个条件分位数的最佳和可解释的线性估计值和不确定性量化。该方法是一种适用于特定分位数子集选择的有效工具。 |
| [^8] | [Model-Agnostic Covariate-Assisted Inference on Partially Identified Causal Effects.](http://arxiv.org/abs/2310.08115) | 提出了一种模型不可知的推断方法，在部分可辨识的因果估计中应用广泛。该方法基于最优输运问题的对偶理论，能够适应随机实验和观测研究，并且具有统一有效和双重鲁棒性。 |
| [^9] | [A Model-Agnostic Graph Neural Network for Integrating Local and Global Information.](http://arxiv.org/abs/2309.13459) | MaGNet是一种模型无关的图神经网络框架，能够顺序地整合不同顺序的信息，并通过识别有影响力的紧凑图结构提供有意义且可解释的结果。 |
| [^10] | [Model-based Clustering using Non-parametric Hidden Markov Models.](http://arxiv.org/abs/2309.12238) | 本文研究了使用非参数隐马尔可夫模型进行基于模型的聚类时的贝叶斯风险，并提出了相应的聚类方法。通过研究分类的贝叶斯风险和聚类的贝叶斯风险之间的关系，确定了聚类任务的难度。同时，在插值分类器和在线设置中的结果也得到了证明。模拟实验验证了这些发现。 |
| [^11] | [Optimal and Fair Encouragement Policy Evaluation and Learning.](http://arxiv.org/abs/2309.07176) | 本研究探讨了在关键领域中针对鼓励政策的最优和公平评估以及学习的问题，研究发现在人类不遵循治疗建议的情况下，最优策略规则只是建议。同时，针对治疗的异质性和公平考虑因素，决策者的权衡和决策规则也会发生变化。在社会服务领域，研究显示存在一个使用差距问题，那些最有可能受益的人却无法获得这些益服务。 |

# 详细

[^1]: 利用观测数据进行强健学习以获得最佳动态治疗方案

    Robust Learning for Optimal Dynamic Treatment Regimes with Observational Data

    [https://arxiv.org/abs/2404.00221](https://arxiv.org/abs/2404.00221)

    学习利用观测数据提出了一种逐步双重强健方法，通过向后归纳解决了最佳动态治疗方案的问题

    

    许多公共政策和医疗干预涉及其治疗分配中的动态性，治疗通常依据先前治疗的历史和相关特征对每个阶段的效果具有异质性。本文研究了统计学习最佳动态治疗方案(DTR)，根据个体的历史指导每个阶段的最佳治疗分配。我们提出了一种基于观测数据的逐步双重强健方法，在顺序可忽略性假设下学习最佳DTR。该方法通过向后归纳解决了顺序治疗分配问题，在每一步中，我们结合倾向评分和行动值函数(Q函数)的估计量，构建了政策价值的增强反向概率加权估计量。

    arXiv:2404.00221v1 Announce Type: cross  Abstract: Many public policies and medical interventions involve dynamics in their treatment assignments, where treatments are sequentially assigned to the same individuals across multiple stages, and the effect of treatment at each stage is usually heterogeneous with respect to the history of prior treatments and associated characteristics. We study statistical learning of optimal dynamic treatment regimes (DTRs) that guide the optimal treatment assignment for each individual at each stage based on the individual's history. We propose a step-wise doubly-robust approach to learn the optimal DTR using observational data under the assumption of sequential ignorability. The approach solves the sequential treatment assignment problem through backward induction, where, at each step, we combine estimators of propensity scores and action-value functions (Q-functions) to construct augmented inverse probability weighting estimators of values of policies 
    
[^2]: AdaTrans：针对高维回归的特征自适应与样本自适应迁移学习

    AdaTrans: Feature-wise and Sample-wise Adaptive Transfer Learning for High-dimensional Regression

    [https://arxiv.org/abs/2403.13565](https://arxiv.org/abs/2403.13565)

    提出了一种针对高维回归的自适应迁移学习方法，可以根据可迁移结构自适应检测和聚合特征和样本的可迁移结构。

    

    我们考虑高维背景下的迁移学习问题，在该问题中，特征维度大于样本大小。为了学习可迁移的信息，该信息可能在特征或源样本之间变化，我们提出一种自适应迁移学习方法，可以检测和聚合特征-wise (F-AdaTrans)或样本-wise (S-AdaTrans)可迁移结构。我们通过采用一种新颖的融合惩罚方法，结合权重，可以根据可迁移结构进行调整。为了选择权重，我们提出了一个在理论上建立，数据驱动的过程，使得 F-AdaTrans 能够选择性地将可迁移的信号与目标融合在一起，同时滤除非可迁移的信号，S-AdaTrans则可以获得每个源样本传递的信息的最佳组合。我们建立了非渐近速率，可以在特殊情况下恢复现有的近最小似乎最优速率。效果证明...

    arXiv:2403.13565v1 Announce Type: cross  Abstract: We consider the transfer learning problem in the high dimensional setting, where the feature dimension is larger than the sample size. To learn transferable information, which may vary across features or the source samples, we propose an adaptive transfer learning method that can detect and aggregate the feature-wise (F-AdaTrans) or sample-wise (S-AdaTrans) transferable structures. We achieve this by employing a novel fused-penalty, coupled with weights that can adapt according to the transferable structure. To choose the weight, we propose a theoretically informed, data-driven procedure, enabling F-AdaTrans to selectively fuse the transferable signals with the target while filtering out non-transferable signals, and S-AdaTrans to obtain the optimal combination of information transferred from each source sample. The non-asymptotic rates are established, which recover existing near-minimax optimal rates in special cases. The effectivene
    
[^3]: 可解释的机器学习用于生存分析

    Interpretable Machine Learning for Survival Analysis

    [https://arxiv.org/abs/2403.10250](https://arxiv.org/abs/2403.10250)

    可解释的机器学习在生存分析中的应用促进了透明度和公平性，揭示了模型的潜在偏见和限制，并提供了更符合数学原理的特征影响和风险因素预测方法。

    

    随着黑盒机器学习模型的传播和快速进步，可解释的机器学习（IML）领域或可解释的人工智能（XAI）在过去十年中变得越来越重要。 这在生存分析领域尤为重要，其中采用IML技术促进了透明度、问责制和公平性，特别是在临床决策过程、有针对性疗法的开发、干预或其他医学或与医疗保健相关的环境中。 具体来说，可解释性可以揭示生存模型的潜在偏见和局限性，并提供更符合数学原理的方法来理解哪些特征对预测有影响或构成风险因素。 然而，缺乏即时可用的IML方法可能已经阻碍了医学从业者和公共卫生政策制定者充分利用机器学习的潜力。

    arXiv:2403.10250v1 Announce Type: cross  Abstract: With the spread and rapid advancement of black box machine learning models, the field of interpretable machine learning (IML) or explainable artificial intelligence (XAI) has become increasingly important over the last decade. This is particularly relevant for survival analysis, where the adoption of IML techniques promotes transparency, accountability and fairness in sensitive areas, such as clinical decision making processes, the development of targeted therapies, interventions or in other medical or healthcare related contexts. More specifically, explainability can uncover a survival model's potential biases and limitations and provide more mathematically sound ways to understand how and which features are influential for prediction or constitute risk factors. However, the lack of readily available IML methods may have deterred medical practitioners and policy makers in public health from leveraging the full potential of machine lea
    
[^4]: 当你的AI欺骗你：在奖励学习中人类评估者部分可观测性的挑战

    When Your AI Deceives You: Challenges with Partial Observability of Human Evaluators in Reward Learning

    [https://arxiv.org/abs/2402.17747](https://arxiv.org/abs/2402.17747)

    RLHF在考虑部分观察性时可能导致策略欺骗性地夸大性能或过度辩护行为，我们提出了数学条件来解决这些问题，并警告不要盲目应用RLHF在部分可观测情况下。

    

    强化学习从人类反馈（RLHF）的过去分析假设人类完全观察到环境。当人类反馈仅基于部分观察时会发生什么？我们对两种失败情况进行了正式定义：欺骗和过度辩护。通过将人类建模为对轨迹信念的Boltzmann-理性，我们证明了RLHF保证会导致策略欺骗性地夸大其性能、为了留下印象而过度辩护或者两者兼而有之的条件。为了帮助解决这些问题，我们数学地刻画了环境部分可观测性如何转化为（缺乏）学到的回报函数中的模糊性。在某些情况下，考虑环境部分可观测性使得在理论上可能恢复回报函数和最优策略，而在其他情况下，存在不可减少的模糊性。我们警告不要盲目应用RLHF在部分可观测情况下。

    arXiv:2402.17747v1 Announce Type: cross  Abstract: Past analyses of reinforcement learning from human feedback (RLHF) assume that the human fully observes the environment. What happens when human feedback is based only on partial observations? We formally define two failure cases: deception and overjustification. Modeling the human as Boltzmann-rational w.r.t. a belief over trajectories, we prove conditions under which RLHF is guaranteed to result in policies that deceptively inflate their performance, overjustify their behavior to make an impression, or both. To help address these issues, we mathematically characterize how partial observability of the environment translates into (lack of) ambiguity in the learned return function. In some cases, accounting for partial observability makes it theoretically possible to recover the return function and thus the optimal policy, while in other cases, there is irreducible ambiguity. We caution against blindly applying RLHF in partially observa
    
[^5]: 梯度下降引发了深度非线性网络权重与经验NTK之间的对齐

    Gradient descent induces alignment between weights and the empirical NTK for deep non-linear networks

    [https://arxiv.org/abs/2402.05271](https://arxiv.org/abs/2402.05271)

    了解神经网络从输入-标签对中提取统计信息的机制是监督学习中最重要的未解决问题之一。前人的研究表明，在训练过程中，权重的格拉姆矩阵与模型的平均梯度外积成正比，这被称为神经特征分析（NFA）。本研究解释了这种相关性的出现，并发现NFA等价于权重矩阵的左奇异结构与与这些权重相关的经验神经切线核的显著成分之间的对齐。在早期训练阶段，可以通过解析的方式预测NFA的发展速度。

    

    理解神经网络从输入-标签对中提取统计信息的机制是监督学习中最重要的未解决问题之一。先前的研究已经确定，在一般结构的训练神经网络中，权重的格拉姆矩阵与模型的平均梯度外积成正比，这个说法被称为神经特征分析（NFA）。然而，这些数量在训练过程中如何相关尚不清楚。在这项工作中，我们解释了这种相关性的出现。我们发现NFA等价于权重矩阵的左奇异结构与与这些权重相关的经验神经切线核的显著成分之间的对齐。我们证明了先前研究中引入的NFA是由隔离这种对齐的中心化NFA驱动的。我们还展示了在早期训练阶段，可以通过解析的方式预测NFA的发展速度。

    Understanding the mechanisms through which neural networks extract statistics from input-label pairs is one of the most important unsolved problems in supervised learning. Prior works have identified that the gram matrices of the weights in trained neural networks of general architectures are proportional to the average gradient outer product of the model, in a statement known as the Neural Feature Ansatz (NFA). However, the reason these quantities become correlated during training is poorly understood. In this work, we explain the emergence of this correlation. We identify that the NFA is equivalent to alignment between the left singular structure of the weight matrices and a significant component of the empirical neural tangent kernels associated with those weights. We establish that the NFA introduced in prior works is driven by a centered NFA that isolates this alignment. We show that the speed of NFA development can be predicted analytically at early training times in terms of sim
    
[^6]: 可解释的多源数据融合通过潜变量高斯过程

    Interpretable Multi-Source Data Fusion Through Latent Variable Gaussian Process

    [https://arxiv.org/abs/2402.04146](https://arxiv.org/abs/2402.04146)

    这篇论文提出了一种基于潜变量高斯过程的多源数据融合框架，用于解决多个数据源之间质量和全面性差异给系统优化带来的问题。

    

    随着人工智能（AI）和机器学习（ML）的出现，各个科学和工程领域已经利用数据驱动的替代模型来建模来自大量信息源（数据）的复杂系统。这种增加导致了开发出用于执行特定功能的优越系统所需的成本和时间的显著降低。这样的替代模型往往广泛地融合多个数据来源，可能是发表的论文、专利、开放资源库或其他资源。然而，对于已知和未知的信息来源的基础物理参数的质量和全面性的差异，可能对系统优化过程产生后续影响，却没有得到充分的关注。为了解决这个问题，提出了一种基于潜变量高斯过程（LVGP）的多源数据融合框架。

    With the advent of artificial intelligence (AI) and machine learning (ML), various domains of science and engineering communites has leveraged data-driven surrogates to model complex systems from numerous sources of information (data). The proliferation has led to significant reduction in cost and time involved in development of superior systems designed to perform specific functionalities. A high proposition of such surrogates are built extensively fusing multiple sources of data, may it be published papers, patents, open repositories, or other resources. However, not much attention has been paid to the differences in quality and comprehensiveness of the known and unknown underlying physical parameters of the information sources that could have downstream implications during system optimization. Towards resolving this issue, a multi-source data fusion framework based on Latent Variable Gaussian Process (LVGP) is proposed. The individual data sources are tagged as a characteristic cate
    
[^7]: 基于子集选择的贝叶斯分位回归：后验总结视角

    Bayesian Quantile Regression with Subset Selection: A Posterior Summarization Perspective. (arXiv:2311.02043v1 [stat.ME])

    [http://arxiv.org/abs/2311.02043](http://arxiv.org/abs/2311.02043)

    本研究提出了一种基于贝叶斯决策分析的方法，对于任何贝叶斯回归模型，可以得到每个条件分位数的最佳和可解释的线性估计值和不确定性量化。该方法是一种适用于特定分位数子集选择的有效工具。

    

    分位回归是一种强大的工具，用于推断协变量如何影响响应分布的特定分位数。现有方法要么分别估计每个感兴趣分位数的条件分位数，要么使用半参数或非参数模型估计整个条件分布。前者经常产生不适合实际数据的模型，并且不在分位数之间共享信息，而后者则以复杂且受限制的模型为特点，难以解释和计算效率低下。此外，这两种方法都不适合于特定分位数的子集选择。相反，我们从贝叶斯决策分析的角度出发，提出了线性分位估计、不确定性量化和子集选择的基本问题。对于任何贝叶斯回归模型，我们为每个基于模型的条件分位数推导出最佳和可解释的线性估计值和不确定性量化。我们的方法引入了一种分位数聚焦的方法。

    Quantile regression is a powerful tool for inferring how covariates affect specific percentiles of the response distribution. Existing methods either estimate conditional quantiles separately for each quantile of interest or estimate the entire conditional distribution using semi- or non-parametric models. The former often produce inadequate models for real data and do not share information across quantiles, while the latter are characterized by complex and constrained models that can be difficult to interpret and computationally inefficient. Further, neither approach is well-suited for quantile-specific subset selection. Instead, we pose the fundamental problems of linear quantile estimation, uncertainty quantification, and subset selection from a Bayesian decision analysis perspective. For any Bayesian regression model, we derive optimal and interpretable linear estimates and uncertainty quantification for each model-based conditional quantile. Our approach introduces a quantile-focu
    
[^8]: 模型不可知的辅助推断方法在部分可辨识因果效应上的应用

    Model-Agnostic Covariate-Assisted Inference on Partially Identified Causal Effects. (arXiv:2310.08115v1 [econ.EM])

    [http://arxiv.org/abs/2310.08115](http://arxiv.org/abs/2310.08115)

    提出了一种模型不可知的推断方法，在部分可辨识的因果估计中应用广泛。该方法基于最优输运问题的对偶理论，能够适应随机实验和观测研究，并且具有统一有效和双重鲁棒性。

    

    很多因果估计是部分可辨识的，因为它们依赖于潜在结果之间的不可观察联合分布。基于前处理协变量的分层可以获得更明确的部分可辨识性范围；然而，除非协变量为离散且支撑度相对较小，否则这种方法通常需要对给定协变量的潜在结果的条件分布进行一致估计。因此，现有的方法在模型错误或一致性假设被违反时可能失败。在本研究中，我们提出了一种基于最优输运问题的对偶理论的统一且模型不可知的推断方法，适用于广泛类别的部分可辨识估计。在随机实验中，我们的方法可以结合任何对条件分布的估计，并提供统一有效的推断，即使初始估计是任意不准确的。此外，我们的方法在观测研究中也是双重鲁棒的。

    Many causal estimands are only partially identifiable since they depend on the unobservable joint distribution between potential outcomes. Stratification on pretreatment covariates can yield sharper partial identification bounds; however, unless the covariates are discrete with relatively small support, this approach typically requires consistent estimation of the conditional distributions of the potential outcomes given the covariates. Thus, existing approaches may fail under model misspecification or if consistency assumptions are violated. In this study, we propose a unified and model-agnostic inferential approach for a wide class of partially identified estimands, based on duality theory for optimal transport problems. In randomized experiments, our approach can wrap around any estimates of the conditional distributions and provide uniformly valid inference, even if the initial estimates are arbitrarily inaccurate. Also, our approach is doubly robust in observational studies. Notab
    
[^9]: 模型无关的图神经网络用于整合局部和全局信息的研究

    A Model-Agnostic Graph Neural Network for Integrating Local and Global Information. (arXiv:2309.13459v1 [stat.ML])

    [http://arxiv.org/abs/2309.13459](http://arxiv.org/abs/2309.13459)

    MaGNet是一种模型无关的图神经网络框架，能够顺序地整合不同顺序的信息，并通过识别有影响力的紧凑图结构提供有意义且可解释的结果。

    

    图神经网络（GNNs）在各种以图为重点的任务中取得了令人满意的性能。尽管取得了成功，但现有的GNN存在两个重要限制：由于黑盒特性，结果缺乏可解释性；无法学习不同顺序的表示。为了解决这些问题，我们提出了一种新的模型无关的图神经网络（MaGNet）框架，能够顺序地整合不同顺序的信息，从高阶邻居中提取知识，并通过识别有影响力的紧凑图结构提供有意义且可解释的结果。特别地，MaGNet由两个组件组成：图拓扑下复杂关系的潜在表示的估计模型和识别有影响力的节点、边和重要节点特征的解释模型。从理论上，我们通过经验Rademacher复杂度建立了MaGNet的泛化误差界，并展示了其强大的能力。

    Graph Neural Networks (GNNs) have achieved promising performance in a variety of graph-focused tasks. Despite their success, existing GNNs suffer from two significant limitations: a lack of interpretability in results due to their black-box nature, and an inability to learn representations of varying orders. To tackle these issues, we propose a novel Model-agnostic Graph Neural Network (MaGNet) framework, which is able to sequentially integrate information of various orders, extract knowledge from high-order neighbors, and provide meaningful and interpretable results by identifying influential compact graph structures. In particular, MaGNet consists of two components: an estimation model for the latent representation of complex relationships under graph topology, and an interpretation model that identifies influential nodes, edges, and important node features. Theoretically, we establish the generalization error bound for MaGNet via empirical Rademacher complexity, and showcase its pow
    
[^10]: 使用非参数隐马尔可夫模型的基于模型的聚类

    Model-based Clustering using Non-parametric Hidden Markov Models. (arXiv:2309.12238v1 [math.ST])

    [http://arxiv.org/abs/2309.12238](http://arxiv.org/abs/2309.12238)

    本文研究了使用非参数隐马尔可夫模型进行基于模型的聚类时的贝叶斯风险，并提出了相应的聚类方法。通过研究分类的贝叶斯风险和聚类的贝叶斯风险之间的关系，确定了聚类任务的难度。同时，在插值分类器和在线设置中的结果也得到了证明。模拟实验验证了这些发现。

    

    非参数隐马尔可夫模型（HMM）由于其依赖结构，可以在不指定群组分布的情况下进行基于模型的聚类。本文研究了在使用HMM进行聚类时的贝叶斯风险，并提出了相应的聚类方法。首先，我们给出了将分类的贝叶斯风险与聚类的贝叶斯风险联系起来的结果，用以确定聚类任务的难度的关键数量。我们还在独立同分布的框架下证明了这一结果，这可能具有独立的兴趣。然后我们研究了插值分类器的过度风险。所有这些结果都被证明在在线设置中仍然有效，在该设置下，观测结果被顺序聚类。模拟实验证明了我们的发现。

    Thanks to their dependency structure, non-parametric Hidden Markov Models (HMMs) are able to handle model-based clustering without specifying group distributions. The aim of this work is to study the Bayes risk of clustering when using HMMs and to propose associated clustering procedures. We first give a result linking the Bayes risk of classification and the Bayes risk of clustering, which we use to identify the key quantity determining the difficulty of the clustering task. We also give a proof of this result in the i.i.d. framework, which might be of independent interest. Then we study the excess risk of the plugin classifier. All these results are shown to remain valid in the online setting where observations are clustered sequentially. Simulations illustrate our findings.
    
[^11]: 最优和公平的鼓励政策评估与学习

    Optimal and Fair Encouragement Policy Evaluation and Learning. (arXiv:2309.07176v1 [cs.LG])

    [http://arxiv.org/abs/2309.07176](http://arxiv.org/abs/2309.07176)

    本研究探讨了在关键领域中针对鼓励政策的最优和公平评估以及学习的问题，研究发现在人类不遵循治疗建议的情况下，最优策略规则只是建议。同时，针对治疗的异质性和公平考虑因素，决策者的权衡和决策规则也会发生变化。在社会服务领域，研究显示存在一个使用差距问题，那些最有可能受益的人却无法获得这些益服务。

    

    在关键领域中，强制个体接受治疗通常是不可能的，因此在人类不遵循治疗建议的情况下，最优策略规则只是建议。在这些领域中，接受治疗的个体可能存在异质性，治疗效果也可能存在异质性。虽然最优治疗规则可以最大化整个人群的因果结果，但在鼓励的情况下，对于访问平等限制或其他公平考虑因素可能是相关的。例如，在社会服务领域，一个持久的难题是那些最有可能从中受益的人中那些获益服务的使用差距。当决策者对访问和平均结果都有分配偏好时，最优决策规则会发生变化。我们研究了因果识别、统计方差减少估计和稳健估计的最优治疗规则，包括在违反阳性条件的情况下。

    In consequential domains, it is often impossible to compel individuals to take treatment, so that optimal policy rules are merely suggestions in the presence of human non-adherence to treatment recommendations. In these same domains, there may be heterogeneity both in who responds in taking-up treatment, and heterogeneity in treatment efficacy. While optimal treatment rules can maximize causal outcomes across the population, access parity constraints or other fairness considerations can be relevant in the case of encouragement. For example, in social services, a persistent puzzle is the gap in take-up of beneficial services among those who may benefit from them the most. When in addition the decision-maker has distributional preferences over both access and average outcomes, the optimal decision rule changes. We study causal identification, statistical variance-reduced estimation, and robust estimation of optimal treatment rules, including under potential violations of positivity. We c
    

