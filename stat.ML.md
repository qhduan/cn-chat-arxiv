# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal Transport for Domain Adaptation through Gaussian Mixture Models](https://arxiv.org/abs/2403.13847) | 通过高斯混合模型进行域自适应的最优输运，可以实现源域和目标域混合成分之间的匹配，从而在失效诊断中取得最先进的性能。 |
| [^2] | [DISCOUNT: Distributional Counterfactual Explanation With Optimal Transport.](http://arxiv.org/abs/2401.13112) | 本文提出了使用最优传输进行分布式对抗解释的方法DISCOUNT，将对抗解释的概念扩展到整个输入输出分布，并通过统计置信度来支撑这一方法。 |
| [^3] | [Bayesian Optimization with Hidden Constraints via Latent Decision Models.](http://arxiv.org/abs/2310.18449) | 本文介绍了一种基于潜在决策模型的贝叶斯优化方法，通过利用变分自编码器学习可行决策的分布，在原始空间和潜在空间之间实现了双向映射，从而解决了公共决策制定中的隐藏约束问题。 |
| [^4] | [Fr\'echet Statistics Based Change Point Detection in Multivariate Hawkes Process.](http://arxiv.org/abs/2308.06769) | 本文提出了一种基于Frechet统计的方法，用于在多变量Hawkes过程中检测变点。通过将点过程分成窗口，并利用核矩阵来重构有符号的拉普拉斯矩阵，我们的方法能够准确地检测和描述多变量Hawkes过程因果结构中的变化，具有潜在的金融和神经科学等领域应用价值。 |

# 详细

[^1]: 通过高斯混合模型进行域自适应的最优输运

    Optimal Transport for Domain Adaptation through Gaussian Mixture Models

    [https://arxiv.org/abs/2403.13847](https://arxiv.org/abs/2403.13847)

    通过高斯混合模型进行域自适应的最优输运，可以实现源域和目标域混合成分之间的匹配，从而在失效诊断中取得最先进的性能。

    

    在这篇论文中，我们探讨了通过最优输运进行域自适应的方法。我们提出了一种新颖的方法，即通过高斯混合模型对数据分布进行建模。这种策略使我们能够通过等价的离散问题解决连续最优输运。最优输运解决方案为我们提供了源域和目标域混合成分之间的匹配。通过这种匹配，我们可以在域之间映射数据点，或者将标签从源域组件转移到目标域。我们在失效诊断的两个域自适应基准测试中进行了实验，结果表明我们的方法具有最先进的性能。

    arXiv:2403.13847v1 Announce Type: cross  Abstract: In this paper we explore domain adaptation through optimal transport. We propose a novel approach, where we model the data distributions through Gaussian mixture models. This strategy allows us to solve continuous optimal transport through an equivalent discrete problem. The optimal transport solution gives us a matching between source and target domain mixture components. From this matching, we can map data points between domains, or transfer the labels from the source domain components towards the target domain. We experiment with 2 domain adaptation benchmarks in fault diagnosis, showing that our methods have state-of-the-art performance.
    
[^2]: DISCOUNT: 使用最优传输进行分布式对抗解释

    DISCOUNT: Distributional Counterfactual Explanation With Optimal Transport. (arXiv:2401.13112v1 [cs.AI])

    [http://arxiv.org/abs/2401.13112](http://arxiv.org/abs/2401.13112)

    本文提出了使用最优传输进行分布式对抗解释的方法DISCOUNT，将对抗解释的概念扩展到整个输入输出分布，并通过统计置信度来支撑这一方法。

    

    对抗解释是在黑盒决策模型中提供洞察力和可解释性的事实方法，通过确定导致不同结果的替代输入实例来实现。本文将对抗解释的概念扩展到分布上下文，从个体数据点扩大到整个输入输出分布，命名为分布式对抗解释。在分布式对抗解释中，我们的重点转向分析事实和对抗的分布属性，类似于评估个体实例及其结果决策的经典方法。我们利用最优传输来构建一个机会约束优化问题，旨在导出与事实对应的对抗分布，以统计置信度做支撑。我们提出的优化方法DISCOUNT在输入和输出分布之间平衡这种置信度。

    Counterfactual Explanations (CE) is the de facto method for providing insight and interpretability in black-box decision-making models by identifying alternative input instances that lead to different outcomes. This paper extends the concept of CEs to a distributional context, broadening the scope from individual data points to entire input and output distributions, named Distributional Counterfactual Explanation (DCE). In DCE, our focus shifts to analyzing the distributional properties of the factual and counterfactual, drawing parallels to the classical approach of assessing individual instances and their resulting decisions. We leverage Optimal Transport (OT) to frame a chance-constrained optimization problem, aiming to derive a counterfactual distribution that closely aligns with its factual counterpart, substantiated by statistical confidence. Our proposed optimization method, DISCOUNT, strategically balances this confidence across both input and output distributions. This algorit
    
[^3]: 基于潜在决策模型的具有隐藏约束的贝叶斯优化方法

    Bayesian Optimization with Hidden Constraints via Latent Decision Models. (arXiv:2310.18449v1 [stat.ML])

    [http://arxiv.org/abs/2310.18449](http://arxiv.org/abs/2310.18449)

    本文介绍了一种基于潜在决策模型的贝叶斯优化方法，通过利用变分自编码器学习可行决策的分布，在原始空间和潜在空间之间实现了双向映射，从而解决了公共决策制定中的隐藏约束问题。

    

    贝叶斯优化（BO）已经成为解决复杂决策问题的强大工具，尤其在公共政策领域如警察划区方面。然而，由于定义可行区域的复杂性和决策的高维度，其在公共决策制定中的广泛应用受到了阻碍。本文介绍了一种新的贝叶斯优化方法——隐藏约束潜在空间贝叶斯优化（HC-LSBO），该方法集成了潜在决策模型。该方法利用变分自编码器来学习可行决策的分布，实现了原始决策空间与较低维度的潜在空间之间的双向映射。通过这种方式，HC-LSBO捕捉了公共决策制定中固有的隐藏约束的细微差别，在潜在空间中进行优化的同时，在原始空间中评估目标。我们通过对合成数据集和真实数据集进行数值实验来验证我们的方法，特别关注大规模问题。

    Bayesian optimization (BO) has emerged as a potent tool for addressing intricate decision-making challenges, especially in public policy domains such as police districting. However, its broader application in public policymaking is hindered by the complexity of defining feasible regions and the high-dimensionality of decisions. This paper introduces the Hidden-Constrained Latent Space Bayesian Optimization (HC-LSBO), a novel BO method integrated with a latent decision model. This approach leverages a variational autoencoder to learn the distribution of feasible decisions, enabling a two-way mapping between the original decision space and a lower-dimensional latent space. By doing so, HC-LSBO captures the nuances of hidden constraints inherent in public policymaking, allowing for optimization in the latent space while evaluating objectives in the original space. We validate our method through numerical experiments on both synthetic and real data sets, with a specific focus on large-scal
    
[^4]: 基于Frechet统计的多变量Hawkes过程中的变点检测

    Fr\'echet Statistics Based Change Point Detection in Multivariate Hawkes Process. (arXiv:2308.06769v1 [stat.ML])

    [http://arxiv.org/abs/2308.06769](http://arxiv.org/abs/2308.06769)

    本文提出了一种基于Frechet统计的方法，用于在多变量Hawkes过程中检测变点。通过将点过程分成窗口，并利用核矩阵来重构有符号的拉普拉斯矩阵，我们的方法能够准确地检测和描述多变量Hawkes过程因果结构中的变化，具有潜在的金融和神经科学等领域应用价值。

    

    本文提出了一种使用Frechet统计方法对多变量Hawkes过程中的因果网络进行变点检测的新方法。我们的方法将点过程分成重叠的窗口，在每个窗口中估计核矩阵，并通过将核矩阵视为因果网络的邻接矩阵来重构有符号的拉普拉斯矩阵。通过在模拟和真实加密货币数据集上进行实验，我们证明了我们的方法的有效性。我们的结果显示，我们的方法能够准确地检测和描述多变量Hawkes过程因果结构中的变化，并在金融和神经科学等领域具有潜在的应用价值。所提出的方法是对点过程设置中Frechet统计之前工作的扩展，并对多变量点过程的变点检测领域做出了重要贡献。

    This paper proposes a new approach for change point detection in causal networks of multivariate Hawkes processes using Frechet statistics. Our method splits the point process into overlapping windows, estimates kernel matrices in each window, and reconstructs the signed Laplacians by treating the kernel matrices as the adjacency matrices of the causal network. We demonstrate the effectiveness of our method through experiments on both simulated and real-world cryptocurrency datasets. Our results show that our method is capable of accurately detecting and characterizing changes in the causal structure of multivariate Hawkes processes, and may have potential applications in fields such as finance and neuroscience. The proposed method is an extension of previous work on Frechet statistics in point process settings and represents an important contribution to the field of change point detection in multivariate point processes.
    

