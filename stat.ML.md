# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On Independent Samples Along the Langevin Diffusion and the Unadjusted Langevin Algorithm](https://arxiv.org/abs/2402.17067) | 在该论文中，我们研究了朗之凡扩散和未调整朗之凡算法中随机变量独立化速率的收敛性，证明了在目标函数强对数凹和平滑的情况下，互信息会以指数速率收敛于$0$。 |
| [^2] | [Distributional Reduction: Unifying Dimensionality Reduction and Clustering with Gromov-Wasserstein Projection](https://arxiv.org/abs/2402.02239) | 本文提出了一种新的分布约简方法，利用格罗莫夫-瓦瑟斯坦投影统一了降维和聚类，通过优化问题同时解决降维和聚类，实验证明了该方法在多个领域表现出卓越性能。 |
| [^3] | [Robust Multi-Modal Density Estimation.](http://arxiv.org/abs/2401.10566) | 本文提出了一种名为ROME的鲁棒多模态密度估计方法，该方法利用聚类将多模态样本集分割成多个单模态样本集，并通过简单的KDE估计来估计整体分布。这种方法解决了多模态、非正态和高相关分布估计的挑战。 |

# 详细

[^1]: 关于朗之凡扩散和未调整朗之凡算法中独立样本的研究

    On Independent Samples Along the Langevin Diffusion and the Unadjusted Langevin Algorithm

    [https://arxiv.org/abs/2402.17067](https://arxiv.org/abs/2402.17067)

    在该论文中，我们研究了朗之凡扩散和未调整朗之凡算法中随机变量独立化速率的收敛性，证明了在目标函数强对数凹和平滑的情况下，互信息会以指数速率收敛于$0$。

    

    我们研究了马尔可夫链中初始和当前随机变量独立化的速率，重点关注连续时间中的朗之凡扩散和离散时间中的未调整朗之凡算法（ULA）。我们通过它们的互信息度量随机变量之间的依赖关系。对于朗之凡扩散，我们展示了当目标函数强对数凹时，互信息以指数速率收敛于$0$，当目标函数弱对数凹时，以多项式速率收敛。这些速率类似于在类似条件下朗之凡扩散的混合时间。对于ULA，我们展示了当目标函数强对数凹且光滑时，互信息以指数速率收敛于$0$。我们通过发展这些马尔可夫链的互信息版本的混合时间分析来证明我们的结果。我们还提供了基于朗之凡扩散的强数据处理不等式的替代证明。

    arXiv:2402.17067v1 Announce Type: cross  Abstract: We study the rate at which the initial and current random variables become independent along a Markov chain, focusing on the Langevin diffusion in continuous time and the Unadjusted Langevin Algorithm (ULA) in discrete time. We measure the dependence between random variables via their mutual information. For the Langevin diffusion, we show the mutual information converges to $0$ exponentially fast when the target is strongly log-concave, and at a polynomial rate when the target is weakly log-concave. These rates are analogous to the mixing time of the Langevin diffusion under similar assumptions. For the ULA, we show the mutual information converges to $0$ exponentially fast when the target is strongly log-concave and smooth. We prove our results by developing the mutual version of the mixing time analyses of these Markov chains. We also provide alternative proofs based on strong data processing inequalities for the Langevin diffusion 
    
[^2]: 分布约简：用格罗莫夫-瓦瑟斯坦投影统一降维和聚类

    Distributional Reduction: Unifying Dimensionality Reduction and Clustering with Gromov-Wasserstein Projection

    [https://arxiv.org/abs/2402.02239](https://arxiv.org/abs/2402.02239)

    本文提出了一种新的分布约简方法，利用格罗莫夫-瓦瑟斯坦投影统一了降维和聚类，通过优化问题同时解决降维和聚类，实验证明了该方法在多个领域表现出卓越性能。

    

    无监督学习旨在捕捉潜在的大规模和高维数据集的结构。传统上，这涉及使用降维方法将数据投影到可解释的空间上，或将数据点组织成有意义的聚类。在实践中，这些方法通常是按顺序使用的，而不能保证聚类与降维相一致。在这项工作中，我们提出了一个新的观点：使用分布。通过利用最优输运的工具，特别是格罗莫夫-瓦瑟斯坦距离，我们将聚类和降维统一为一个称为分布约简的单一框架。这使我们能够通过单个优化问题同时解决聚类和降维。通过全面的实验证明了我们方法的多功能性和解释性，并表明它在各种图像和基因组数据集上优于现有方法。

    Unsupervised learning aims to capture the underlying structure of potentially large and high-dimensional datasets. Traditionally, this involves using dimensionality reduction methods to project data onto interpretable spaces or organizing points into meaningful clusters. In practice, these methods are used sequentially, without guaranteeing that the clustering aligns well with the conducted dimensionality reduction. In this work, we offer a fresh perspective: that of distributions. Leveraging tools from optimal transport, particularly the Gromov-Wasserstein distance, we unify clustering and dimensionality reduction into a single framework called distributional reduction. This allows us to jointly address clustering and dimensionality reduction with a single optimization problem. Through comprehensive experiments, we highlight the versatility and interpretability of our method and show that it outperforms existing approaches across a variety of image and genomics datasets.
    
[^3]: 鲁棒的多模态密度估计

    Robust Multi-Modal Density Estimation. (arXiv:2401.10566v1 [cs.LG])

    [http://arxiv.org/abs/2401.10566](http://arxiv.org/abs/2401.10566)

    本文提出了一种名为ROME的鲁棒多模态密度估计方法，该方法利用聚类将多模态样本集分割成多个单模态样本集，并通过简单的KDE估计来估计整体分布。这种方法解决了多模态、非正态和高相关分布估计的挑战。

    

    多模态概率预测模型的发展引发了对综合评估指标的需求。虽然有几个指标可以表征机器学习模型的准确性（例如，负对数似然、Jensen-Shannon散度），但这些指标通常作用于概率密度上。因此，将它们应用于纯粹基于样本的预测模型需要估计底层密度函数。然而，常见的方法如核密度估计（KDE）已被证明在鲁棒性方面存在不足，而更复杂的方法在多模态估计问题中尚未得到评估。在本文中，我们提出了一种非参数的密度估计方法ROME（RObust Multi-modal density Estimator），它解决了估计多模态、非正态和高相关分布的挑战。ROME利用聚类将多模态样本集分割成多个单模态样本集，然后结合简单的KDE估计来得到总体的估计结果。

    Development of multi-modal, probabilistic prediction models has lead to a need for comprehensive evaluation metrics. While several metrics can characterize the accuracy of machine-learned models (e.g., negative log-likelihood, Jensen-Shannon divergence), these metrics typically operate on probability densities. Applying them to purely sample-based prediction models thus requires that the underlying density function is estimated. However, common methods such as kernel density estimation (KDE) have been demonstrated to lack robustness, while more complex methods have not been evaluated in multi-modal estimation problems. In this paper, we present ROME (RObust Multi-modal density Estimator), a non-parametric approach for density estimation which addresses the challenge of estimating multi-modal, non-normal, and highly correlated distributions. ROME utilizes clustering to segment a multi-modal set of samples into multiple uni-modal ones and then combines simple KDE estimates obtained for i
    

