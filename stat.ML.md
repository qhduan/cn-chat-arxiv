# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Probabilistic Model to explain Self-Supervised Representation Learning](https://rss.arxiv.org/abs/2402.01399) | 该论文提出了一个概率模型来解释自监督表示学习的机制，并展示了鉴别性自监督算法在表示中近似诱导潜变量结构的统一理论框架。 |
| [^2] | [Distributional Reinforcement Learning by Sinkhorn Divergence](https://rss.arxiv.org/abs/2202.00769) | 本文提出了SinkhornDRL方法，使用Sinkhorn散度来减小当前和目标Bellman回报分布之间的差异，并通过理论证明和实证实验展示了该方法的优越性。 |
| [^3] | [A Novel Gaussian Min-Max Theorem and its Applications](https://arxiv.org/abs/2402.07356) | 本文介绍了一个新的高斯最小最大定理，扩展了经典定理对于独立但非恒定分布的情况。此外，该定理在高维统计学、机器学习、非光滑优化和信号处理等领域有广泛的应用。 |
| [^4] | [An accelerated first-order regularized momentum descent ascent algorithm for stochastic nonconvex-concave minimax problems.](http://arxiv.org/abs/2310.15448) | 本文提出了一种加速的算法来解决随机非凸-凹极小极大问题，并证明了该算法迭代复杂度达到了最佳已知界限。 |
| [^5] | [Variance-Aware Regret Bounds for Stochastic Contextual Dueling Bandits.](http://arxiv.org/abs/2310.00968) | 本文提出了一种新的算法，用于解决对决争夺中固有不确定性的问题，算法具有计算效率和方差感知遗憾界限。 |
| [^6] | [Random Function Descent.](http://arxiv.org/abs/2305.01377) | 本文提出了随机函数下降(RFD)算法，可以在随机环境中计算出步长并且与贝叶斯优化中的梯度下降算法相同。在合成基准测试中，RFD算法比未调整的Adam方法表现更好，提出的heuristic扩展可与调整后的Adam方法相媲美。 |

# 详细

[^1]: 解释自监督表示学习的概率模型

    A Probabilistic Model to explain Self-Supervised Representation Learning

    [https://rss.arxiv.org/abs/2402.01399](https://rss.arxiv.org/abs/2402.01399)

    该论文提出了一个概率模型来解释自监督表示学习的机制，并展示了鉴别性自监督算法在表示中近似诱导潜变量结构的统一理论框架。

    

    自监督学习（SSL）通过利用辅助的无监督任务，例如对语义相关样本进行分类，如不同的数据增强或模态来学习表示。在众多SSL方法中，对比方法（例如SimCLR，CLIP和VicREG）因学习到的表示在下游性能上接近有监督学习而受到关注。然而，这些方法背后的机制的理论理解仍然存在困难。我们提出了一个生成潜变量模型来表示数据，并展示了几类具有鉴别性的自监督算法（包括对比方法）近似诱导其表示中的潜变量结构，从而提供了一个统一的理论框架。我们还证明了与互信息和投影头的相关性。通过生成式地拟合我们的模型（如SimVE），在常见的基准测试上（例如FashionMNIST，CIFAR10，CelebA），性能优于之前的VAE方法。

    Self-supervised learning (SSL) learns representations by leveraging an auxiliary unsupervised task, such as classifying semantically related samples, e.g. different data augmentations or modalities. Of the many approaches to SSL, contrastive methods, e.g. SimCLR, CLIP and VicREG, have gained attention for learning representations that achieve downstream performance close to that of supervised learning. However, a theoretical understanding of the mechanism behind these methods eludes. We propose a generative latent variable model for the data and show that several families of discriminative self-supervised algorithms, including contrastive methods, approximately induce its latent structure over representations, providing a unifying theoretical framework. We also justify links to mutual information and the use of a projection head. Fitting our model generatively, as SimVE, improves performance over previous VAE methods on common benchmarks (e.g. FashionMNIST, CIFAR10, CelebA), narrows th
    
[^2]: 使用Sinkhorn散度的分布式强化学习

    Distributional Reinforcement Learning by Sinkhorn Divergence

    [https://rss.arxiv.org/abs/2202.00769](https://rss.arxiv.org/abs/2202.00769)

    本文提出了SinkhornDRL方法，使用Sinkhorn散度来减小当前和目标Bellman回报分布之间的差异，并通过理论证明和实证实验展示了该方法的优越性。

    

    分布式强化学习的实证成功高度依赖于分布表示和分布散度的选择。本文提出了Sinkhorn分布强化学习 (SinkhornDRL)的方法，该方法从回报分布中学习无限制的统计量，并利用Sinkhorn散度来减小当前和目标Bellman回报分布之间的差异。从理论上来讲，我们证明了SinkhornDRL的收缩性质，与Sinkhorn散度在Wasserstein距离和最大均值差异 (MMD)之间的插值性质一致。我们还建立了Sinkhorn散度与带有正则化Moment Matching行为的正则化MMD之间的等价关系，从而解释了SinkhornDRL的优越性。在实证上，我们展示了SinkhornDRL在Atari游戏套件上始终表现比现有算法更好或可比。

    The empirical success of distributional reinforcement learning~(RL) highly depends on the distribution representation and the choice of distribution divergence. In this paper, we propose \textit{Sinkhorn distributional RL~(SinkhornDRL)} that learns unrestricted statistics from return distributions and leverages Sinkhorn divergence to minimize the difference between current and target Bellman return distributions. Theoretically, we prove the contraction properties of SinkhornDRL, consistent with the interpolation nature of Sinkhorn divergence between Wasserstein distance and Maximum Mean Discrepancy~(MMD). We also establish the equivalence between Sinkhorn divergence and a regularized MMD with a regularized Moment Matching behavior, contributing to explaining the superiority of SinkhornDRL. Empirically, we show that SinkhornDRL is consistently better or comparable to existing algorithms on the Atari games suite.
    
[^3]: 一个新的高斯最小最大定理及其应用

    A Novel Gaussian Min-Max Theorem and its Applications

    [https://arxiv.org/abs/2402.07356](https://arxiv.org/abs/2402.07356)

    本文介绍了一个新的高斯最小最大定理，扩展了经典定理对于独立但非恒定分布的情况。此外，该定理在高维统计学、机器学习、非光滑优化和信号处理等领域有广泛的应用。

    

    Gordon的一个著名结果允许比较两个高斯过程的最小最大行为，如果满足某些不等式条件。这个结果的结果包括高斯最小最大（GMT）和凸高斯最小最大（CGMT）定理，这些定理在高维统计学、机器学习、非光滑优化和信号处理方面有广泛的应用。目前为止，没有发现满足这些不等式的其他一对高斯过程。在本文中，我们确定了这样一对新的高斯过程。由此得到的定理将经典的GMT定理和CGMT定理从基本过程中的底层高斯矩阵具有iid行的情况扩展到具有独立但非恒定分布的情况。新的CGMT定理应用于多源高斯回归问题，以及属于的二元分类问题。

    A celebrated result by Gordon allows one to compare the min-max behavior of two Gaussian processes if certain inequality conditions are met. The consequences of this result include the Gaussian min-max (GMT) and convex Gaussian min-max (CGMT) theorems which have had far-reaching implications in high-dimensional statistics, machine learning, non-smooth optimization, and signal processing. Both theorems rely on a pair of Gaussian processes, first identified by Slepian, that satisfy Gordon's comparison inequalities. To date, no other pair of Gaussian processes satisfying these inequalities has been discovered. In this paper, we identify such a new pair. The resulting theorems extend the classical GMT and CGMT Theorems from the case where the underlying Gaussian matrix in the primary process has iid rows to where it has independent but non-identically-distributed ones. The new CGMT is applied to the problems of multi-source Gaussian regression, as well as to binary classification of genera
    
[^4]: 一种加速的一阶正则动量下降算法用于随机非凸-凹极小极大问题

    An accelerated first-order regularized momentum descent ascent algorithm for stochastic nonconvex-concave minimax problems. (arXiv:2310.15448v1 [math.OC])

    [http://arxiv.org/abs/2310.15448](http://arxiv.org/abs/2310.15448)

    本文提出了一种加速的算法来解决随机非凸-凹极小极大问题，并证明了该算法迭代复杂度达到了最佳已知界限。

    

    随机非凸极小极大问题近年来在机器学习、信号处理等领域引起了广泛关注。本文提出了一种加速的一阶正则动量下降算法（FORMDA）用于解决随机非凸-凹极小极大问题。证明了该算法的迭代复杂度为$\tilde{\mathcal{O}}(\varepsilon ^{-6.5})$以达到$\varepsilon$-稳定点，这在目标函数稳定性下实现了解决随机非凸-凹极小极大问题的最佳已知复杂度界限。

    Stochastic nonconvex minimax problems have attracted wide attention in machine learning, signal processing and many other fields in recent years. In this paper, we propose an accelerated first-order regularized momentum descent ascent algorithm (FORMDA) for solving stochastic nonconvex-concave minimax problems. The iteration complexity of the algorithm is proved to be $\tilde{\mathcal{O}}(\varepsilon ^{-6.5})$ to obtain an $\varepsilon$-stationary point, which achieves the best-known complexity bound for single-loop algorithms to solve the stochastic nonconvex-concave minimax problems under the stationarity of the objective function.
    
[^5]: 随机情境对决争夺决策的方差感知遗憾界限

    Variance-Aware Regret Bounds for Stochastic Contextual Dueling Bandits. (arXiv:2310.00968v1 [cs.LG])

    [http://arxiv.org/abs/2310.00968](http://arxiv.org/abs/2310.00968)

    本文提出了一种新的算法，用于解决对决争夺中固有不确定性的问题，算法具有计算效率和方差感知遗憾界限。

    

    对决争夺是一个重要的决策框架，涉及到偏好反馈的决策，这是一个适用于人机交互的各种应用场景的有价值特性，例如排名、信息检索和推荐系统。虽然在对决争夺中已经做出了大量的努力来最小化累计遗憾，但目前研究中存在一个明显的空白，即遗憾界限未考虑到对决手表间成对比较的固有不确定性。直观地说，更大的不确定性意味着问题的难度更高。为了填补这个空白，本文研究了情境对决争夺的问题，其中决策手表的二元对比是由广义线性模型（GLM）生成的。我们提出了一种新的SupLinUCB类型的算法，这个算法具有计算效率和一个感知方差遗憾界限$\tilde O\big(d\sqrt{\sum_{t=1}^T\sigma_t^2} + d\big)$，其中$\sigma_t$是每轮成对比较的方差。

    Dueling bandits is a prominent framework for decision-making involving preferential feedback, a valuable feature that fits various applications involving human interaction, such as ranking, information retrieval, and recommendation systems. While substantial efforts have been made to minimize the cumulative regret in dueling bandits, a notable gap in the current research is the absence of regret bounds that account for the inherent uncertainty in pairwise comparisons between the dueling arms. Intuitively, greater uncertainty suggests a higher level of difficulty in the problem. To bridge this gap, this paper studies the problem of contextual dueling bandits, where the binary comparison of dueling arms is generated from a generalized linear model (GLM). We propose a new SupLinUCB-type algorithm that enjoys computational efficiency and a variance-aware regret bound $\tilde O\big(d\sqrt{\sum_{t=1}^T\sigma_t^2} + d\big)$, where $\sigma_t$ is the variance of the pairwise comparison in round
    
[^6]: 随机函数下降法

    Random Function Descent. (arXiv:2305.01377v1 [math.OC])

    [http://arxiv.org/abs/2305.01377](http://arxiv.org/abs/2305.01377)

    本文提出了随机函数下降(RFD)算法，可以在随机环境中计算出步长并且与贝叶斯优化中的梯度下降算法相同。在合成基准测试中，RFD算法比未调整的Adam方法表现更好，提出的heuristic扩展可与调整后的Adam方法相媲美。

    

    虽然梯度下降方法在机器学习中十分常见，但是选择正确的步长经常需要进行“超参数调整”。这是因为回溯程序如Armijo's准则依赖于每个步骤中的质量评估，而这些评估在随机情况下不可用。由于优化方案可以用Taylor逼近来解释，我们将Taylor逼近替换为条件期望（最佳的$L^2$估计），提出了“随机函数下降”（RFD）。 在Bayesian优化中常见的一些轻微假设的情况下，我们证明了RFD与梯度下降算法是相同的，但是在随机情况下具有可计算的步长。我们在合成基准测试中比未调整的Adam方法表现更好。为了缩小与调整后的Adam算法之间的性能差距，我们提出了一种启发式扩展，可与调整后的Adam方法相媲美。

    While gradient based methods are ubiquitous in machine learning, selecting the right step size often requires "hyperparameter tuning". This is because backtracking procedures like Armijo's rule depend on quality evaluations in every step, which are not available in a stochastic context. Since optimization schemes can be motivated using Taylor approximations, we replace the Taylor approximation with the conditional expectation (the best $L^2$ estimator) and propose "Random Function Descent" (RFD). Under light assumptions common in Bayesian optimization, we prove that RFD is identical to gradient descent, but with calculable step sizes, even in a stochastic context. We beat untuned Adam in synthetic benchmarks. To close the performance gap to tuned Adam, we propose a heuristic extension competitive with tuned Adam.
    

