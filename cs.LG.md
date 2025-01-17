# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Statistical Efficiency of Distributional Temporal Difference](https://arxiv.org/abs/2403.05811) | 该论文分析了分布式时间差分的统计效率和有限样本性能。 |
| [^2] | [Wasserstein Gradient Flows for Moreau Envelopes of f-Divergences in Reproducing Kernel Hilbert Spaces](https://arxiv.org/abs/2402.04613) | 本文研究了在再生核希尔伯特空间中使用Moreau包络来对测度f-差异进行正则化的方法，并利用该方法分析了Wasserstein梯度流。 |
| [^3] | [Gaussian Processes with Linear Multiple Kernel: Spectrum Design and Distributed Learning for Multi-Dimensional Data.](http://arxiv.org/abs/2309.08201) | 本文研究了高斯过程与线性多核在多维数据上的应用，提出了一种新的格点谱混合核公式，减少了超参数数量，同时保留了优化结构和逼近能力。通过引入分布式算法，使大规模超参数优化变得可行。 |
| [^4] | [Language Aligned Visual Representations Predict Human Behavior in Naturalistic Learning Tasks.](http://arxiv.org/abs/2306.09377) | 语言对齐的视觉表示方式比纯视觉表示方式更有效地预测人类在自然学习任务中的行为。 |
| [^5] | [PeFLL: A Lifelong Learning Approach to Personalized Federated Learning.](http://arxiv.org/abs/2306.05515) | PeFLL是个性化联邦学习的一种新方法，通过联合训练嵌入网络和超网络，PeFLL能够学习输出特定于每个客户端的模型，并且在其它新出现的客户端上表现良好。 |
| [^6] | [Graphy Analysis Using a GPU-based Parallel Algorithm: Quantum Clustering.](http://arxiv.org/abs/2305.14641) | 本文介绍了一种新方法将量子聚类应用于图结构中，使用基于GPU的并行算法来计算潜在值。实验结果表明该方法具有优越性能。 |
| [^7] | [Global Convergence Rate Analysis of Nonsmooth Nonconvex-Nonconcave Minimax Optimization.](http://arxiv.org/abs/2209.10825) | 本论文提出了一种名为smoothed PLDA的算法来有效处理广泛的结构化非光滑非凸非凹极小极大问题，并证明了其具有全局收敛性，复杂度为O(epsilon^(-2/3))。 |

# 详细

[^1]: 分布式时间差分的统计效率

    Statistical Efficiency of Distributional Temporal Difference

    [https://arxiv.org/abs/2403.05811](https://arxiv.org/abs/2403.05811)

    该论文分析了分布式时间差分的统计效率和有限样本性能。

    

    分布式强化学习(DRL)关注的是返回的完整分布，而不仅仅是均值，在各个领域取得了经验成功。领域DRL中的核心任务之一是分布式策略评估，涉及估计给定策略pi的返回分布η^pi。相应地提出了分布时间差分(TD)算法，这是经典RL文献中时间差分算法的延伸。在表格案例中，citet{rowland2018analysis}和citet{rowland2023analysis}分别证明了两个分布式TD实例即分类时间差分算法(CTD)和分位数时间差分算法(QTD)的渐近收敛。在这篇论文中，我们进一步分析了分布式TD的有限样本性能。为了促进理论分析，我们提出了一个非参数的 dis

    arXiv:2403.05811v1 Announce Type: cross  Abstract: Distributional reinforcement learning (DRL), which cares about the full distribution of returns instead of just the mean, has achieved empirical success in various domains. One of the core tasks in the field of DRL is distributional policy evaluation, which involves estimating the return distribution $\eta^\pi$ for a given policy $\pi$. A distributional temporal difference (TD) algorithm has been accordingly proposed, which is an extension of the temporal difference algorithm in the classic RL literature. In the tabular case, \citet{rowland2018analysis} and \citet{rowland2023analysis} proved the asymptotic convergence of two instances of distributional TD, namely categorical temporal difference algorithm (CTD) and quantile temporal difference algorithm (QTD), respectively. In this paper, we go a step further and analyze the finite-sample performance of distributional TD. To facilitate theoretical analysis, we propose non-parametric dis
    
[^2]: 在再生核希尔伯特空间中的Moreau包络的f-差异的Wasserstein梯度流

    Wasserstein Gradient Flows for Moreau Envelopes of f-Divergences in Reproducing Kernel Hilbert Spaces

    [https://arxiv.org/abs/2402.04613](https://arxiv.org/abs/2402.04613)

    本文研究了在再生核希尔伯特空间中使用Moreau包络来对测度f-差异进行正则化的方法，并利用该方法分析了Wasserstein梯度流。

    

    大多数常用的测度f-差异，例如Kullback-Leibler差异，对于所涉及的测度的支持存在限制。解决办法是通过与特征核K相关的平方最大均值差异(MMD)对f-差异进行正则化。在本文中，我们使用所谓的核均值嵌入来显示相应的正则化可以重写为与K相关的再生核希尔伯特空间中某些函数的Moreau包络。然后，我们利用关于希尔伯特空间中Moreau包络的众所周知的结果来证明MMD正则化的f-差异及其梯度的属性。随后，我们使用我们的研究结果来分析受MMD正则化的f-差异的Wasserstein梯度流。最后，我们考虑从经验测度开始的Wasserstein梯度流，并提供使用Tsallis-$\alpha$差异的概念性数值示例的证明。

    Most commonly used $f$-divergences of measures, e.g., the Kullback-Leibler divergence, are subject to limitations regarding the support of the involved measures. A remedy consists of regularizing the $f$-divergence by a squared maximum mean discrepancy (MMD) associated with a characteristic kernel $K$. In this paper, we use the so-called kernel mean embedding to show that the corresponding regularization can be rewritten as the Moreau envelope of some function in the reproducing kernel Hilbert space associated with $K$. Then, we exploit well-known results on Moreau envelopes in Hilbert spaces to prove properties of the MMD-regularized $f$-divergences and, in particular, their gradients. Subsequently, we use our findings to analyze Wasserstein gradient flows of MMD-regularized $f$-divergences. Finally, we consider Wasserstein gradient flows starting from empirical measures and provide proof-of-the-concept numerical examples with Tsallis-$\alpha$ divergences.
    
[^3]: 高斯过程与线性多核：频谱设计和多维数据的分布式学习

    Gaussian Processes with Linear Multiple Kernel: Spectrum Design and Distributed Learning for Multi-Dimensional Data. (arXiv:2309.08201v1 [cs.LG])

    [http://arxiv.org/abs/2309.08201](http://arxiv.org/abs/2309.08201)

    本文研究了高斯过程与线性多核在多维数据上的应用，提出了一种新的格点谱混合核公式，减少了超参数数量，同时保留了优化结构和逼近能力。通过引入分布式算法，使大规模超参数优化变得可行。

    

    高斯过程（GPs）已成为机器学习和信号处理的重要技术。GP建模的关键组成部分是核函数的选择，线性多核（LMKs）因其强大的建模能力和可解释性而成为一个吸引人的核函数类。本文重点研究格点谱混合（GSM）核，它是一种可以近似任意平稳核的LMK。具体来说，我们提出了一种新的GSM核公式，用于多维数据，相比现有公式减少了超参数的数量，同时保留了有利的优化结构和逼近能力。此外，为了使GSM核中的大规模超参数优化变得可行，我们首先引入了分布式SCA（DSCA）算法。在此基础上，我们基于交替方向乘子法（ADMM）框架提出了双重分布式SCA（D$^2$SCA）算法，使我们能够合作地进行优化。

    Gaussian processes (GPs) have emerged as a prominent technique for machine learning and signal processing. A key component in GP modeling is the choice of kernel, and linear multiple kernels (LMKs) have become an attractive kernel class due to their powerful modeling capacity and interpretability. This paper focuses on the grid spectral mixture (GSM) kernel, an LMK that can approximate arbitrary stationary kernels. Specifically, we propose a novel GSM kernel formulation for multi-dimensional data that reduces the number of hyper-parameters compared to existing formulations, while also retaining a favorable optimization structure and approximation capability. In addition, to make the large-scale hyper-parameter optimization in the GSM kernel tractable, we first introduce the distributed SCA (DSCA) algorithm. Building on this, we propose the doubly distributed SCA (D$^2$SCA) algorithm based on the alternating direction method of multipliers (ADMM) framework, which allows us to cooperativ
    
[^4]: 对齐语言的视觉表示预测人类在自然学习任务中的行为

    Language Aligned Visual Representations Predict Human Behavior in Naturalistic Learning Tasks. (arXiv:2306.09377v1 [cs.LG])

    [http://arxiv.org/abs/2306.09377](http://arxiv.org/abs/2306.09377)

    语言对齐的视觉表示方式比纯视觉表示方式更有效地预测人类在自然学习任务中的行为。

    

    人类具备识别和概括自然物体相关特征的能力，在各种情境中有所帮助。为了研究这种现象并确定最有效的表示方式以预测人类行为，我们进行了两个涉及类别学习和奖励学习的实验。我们的实验使用逼真的图像作为刺激物，并要求参与者基于所有试验的新型刺激物作出准确的决策，因此需要泛化。在两个任务中，底层规则是使用人类相似性判断提取的刺激维度生成的简单线性函数。值得注意的是，参与者在几次试验内就成功地确定了相关的刺激特征，证明了有效的泛化。我们进行了广泛的模型比较，评估了各种深度学习模型的表示对人类选择的逐次预测准确性。有趣的是，自然语言处理任务（如语言建模和机器翻译）训练的模型表示优于视觉任务训练的模型表示，表明对齐语言的视觉表示可能更有效地预测人类在自然学习任务中的行为。

    Humans possess the ability to identify and generalize relevant features of natural objects, which aids them in various situations. To investigate this phenomenon and determine the most effective representations for predicting human behavior, we conducted two experiments involving category learning and reward learning. Our experiments used realistic images as stimuli, and participants were tasked with making accurate decisions based on novel stimuli for all trials, thereby necessitating generalization. In both tasks, the underlying rules were generated as simple linear functions using stimulus dimensions extracted from human similarity judgments. Notably, participants successfully identified the relevant stimulus features within a few trials, demonstrating effective generalization. We performed an extensive model comparison, evaluating the trial-by-trial predictive accuracy of diverse deep learning models' representations of human choices. Intriguingly, representations from models train
    
[^5]: 一种个性化联邦学习的终身学习方法

    PeFLL: A Lifelong Learning Approach to Personalized Federated Learning. (arXiv:2306.05515v1 [cs.LG])

    [http://arxiv.org/abs/2306.05515](http://arxiv.org/abs/2306.05515)

    PeFLL是个性化联邦学习的一种新方法，通过联合训练嵌入网络和超网络，PeFLL能够学习输出特定于每个客户端的模型，并且在其它新出现的客户端上表现良好。

    

    个性化联邦学习（pFL）已成为应对参与客户端数据分布的统计异质性挑战的常用方法。pFL不是学习单个全局模型，而是旨在学习每个客户端的个体模型，同时仍然利用其他客户端可用的数据。在这项工作中，我们提出了PeFLL，这是一种根植于终身学习的新型pFL方法，不仅在训练阶段存在的客户端上表现良好，而且在未来可能出现的客户端上也表现良好。PeFLL通过联合训练嵌入网络和超网络来学习输出特定于客户端的模型。嵌入网络学习以一种反映它们之间相似性的潜在描述符空间中表示客户端。超网络学习从这个潜在空间到可能的客户模型空间的映射。我们的实验证明，与先前的方法相比，PeFLL产生了更高准确率的模型。

    Personalized federated learning (pFL) has emerged as a popular approach to dealing with the challenge of statistical heterogeneity between the data distributions of the participating clients. Instead of learning a single global model, pFL aims to learn an individual model for each client while still making use of the data available at other clients. In this work, we present PeFLL, a new pFL approach rooted in lifelong learning that performs well not only on clients present during its training phase, but also on any that may emerge in the future. PeFLL learns to output client specific models by jointly training an embedding network and a hypernetwork. The embedding network learns to represent clients in a latent descriptor space in a way that reflects their similarity to each other. The hypernetwork learns a mapping from this latent space to the space of possible client models. We demonstrate experimentally that PeFLL produces models of superior accuracy compared to previous methods, es
    
[^6]: 使用基于GPU的并行算法进行图分析：量子聚类

    Graphy Analysis Using a GPU-based Parallel Algorithm: Quantum Clustering. (arXiv:2305.14641v1 [cs.LG])

    [http://arxiv.org/abs/2305.14641](http://arxiv.org/abs/2305.14641)

    本文介绍了一种新方法将量子聚类应用于图结构中，使用基于GPU的并行算法来计算潜在值。实验结果表明该方法具有优越性能。

    

    本文介绍了一种将量子聚类应用于图结构的新方法。量子聚类（QC）是一种新的基于密度的无监督学习方法，通过构建潜在函数来确定聚类中心。在该方法中，我们使用图梯度下降算法来找到聚类中心。GPU并行化用于计算潜在值。我们还对五个广泛使用的数据集进行了实验，并使用四个指标进行了评估。结果显示该方法具有优越的性能。最后，我们讨论了$\sigma$对实验结果的影响。

    The article introduces a new method for applying Quantum Clustering to graph structures. Quantum Clustering (QC) is a novel density-based unsupervised learning method that determines cluster centers by constructing a potential function. In this method, we use the Graph Gradient Descent algorithm to find the centers of clusters. GPU parallelization is utilized for computing potential values. We also conducted experiments on five widely used datasets and evaluated using four indicators. The results show superior performance of the method. Finally, we discuss the influence of $\sigma$ on the experimental results.
    
[^7]: 非光滑非凸非凹极小极大优化的全局收敛率分析

    Global Convergence Rate Analysis of Nonsmooth Nonconvex-Nonconcave Minimax Optimization. (arXiv:2209.10825v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2209.10825](http://arxiv.org/abs/2209.10825)

    本论文提出了一种名为smoothed PLDA的算法来有效处理广泛的结构化非光滑非凸非凹极小极大问题，并证明了其具有全局收敛性，复杂度为O(epsilon^(-2/3))。

    

    在过去的十年中，非凸非凹极小极大优化引起了广泛关注。然而，大多数现有的工作集中在梯度下降-上升（GDA）算法的各种变体上，这些算法仅适用于平滑的非凸凹场景。为了解决这个局限性，我们提出了一种新算法，名为平滑的近端线性下降上升（smoothed PLDA），可以有效地处理广泛的结构化非光滑非凸非凹极小极大问题。具体而言，我们考虑原始函数具有非光滑复合结构，对偶函数具有Kurdyka-L{o}jasiewicz（K\L{}）性质的情况。我们引入了一种新的收敛分析框架来分析smoothed PLDA算法，其中关键组件是我们最新开发的非光滑原始误差界和对偶误差界属性。利用这个框架，我们证明了smoothed PLDA可以在具有非光滑复合原始函数和KL对偶函数的广泛极小极大问题中找到$\varepsilon$-game-stationary点和$\varepsilon$-最优化稳定点，其复杂度为$\mathcal{O}(\varepsilon^{-2/3})$。

    Nonconvex-nonconcave minimax optimization has gained widespread interest over the last decade. However, most existing work focuses on variants of gradient descent-ascent (GDA) algorithms, which are only applicable in smooth nonconvex-concave settings. To address this limitation, we propose a novel algorithm named smoothed proximal linear descent-ascent (smoothed PLDA), which can effectively handle a broad range of structured nonsmooth nonconvex-nonconcave minimax problems. Specifically, we consider the setting where the primal function has a nonsmooth composite structure and the dual function possesses the Kurdyka-\L{}ojasiewicz (K\L{}) property with exponent $\theta \in [0,1)$. We introduce a novel convergence analysis framework for smoothed PLDA, the key components of which are our newly developed nonsmooth primal error bound and dual error bound properties. Using this framework, we show that smoothed PLDA can find both $\epsilon$-game-stationary points and $\epsilon$-optimization-st
    

