# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Learning for Optimal Dynamic Treatment Regimes with Observational Data](https://arxiv.org/abs/2404.00221) | 学习利用观测数据提出了一种逐步双重强健方法，通过向后归纳解决了最佳动态治疗方案的问题 |
| [^2] | [Multistep Consistency Models](https://arxiv.org/abs/2403.06807) | 本文提出了多步一致性模型，通过在一致性模型和扩散模型之间插值，实现了采样速度和采样质量的平衡。 |
| [^3] | [Mixed-Output Gaussian Process Latent Variable Models](https://arxiv.org/abs/2402.09122) | 本文提出了一种基于高斯过程潜变量模型的贝叶斯非参数方法，可以用于信号分离，并且能够处理包含纯组分信号加权和的情况，适用于光谱学和其他领域的多种应用。 |
| [^4] | [Contextual Combinatorial Bandits with Probabilistically Triggered Arms.](http://arxiv.org/abs/2303.17110) | 本文研究了带有概率触发臂的情境组合赌博机，在不同条件下设计了C$^2$-UCB-T算法和VAC$^2$-UCB算法，并分别导出了对应的遗憾值上限，为相关应用提供了理论支持。 |
| [^5] | [Sufficient Invariant Learning for Distribution Shift.](http://arxiv.org/abs/2210.13533) | 本文研究了分布转移情况下的充分不变学习，观察到之前的工作只学习了部分不变特征，我们提出了学习充分不变特征的重要性，并指出在分布转移时，从训练集中学习的部分不变特征可能不适用于测试集，限制了性能提升。 |

# 详细

[^1]: 利用观测数据进行强健学习以获得最佳动态治疗方案

    Robust Learning for Optimal Dynamic Treatment Regimes with Observational Data

    [https://arxiv.org/abs/2404.00221](https://arxiv.org/abs/2404.00221)

    学习利用观测数据提出了一种逐步双重强健方法，通过向后归纳解决了最佳动态治疗方案的问题

    

    许多公共政策和医疗干预涉及其治疗分配中的动态性，治疗通常依据先前治疗的历史和相关特征对每个阶段的效果具有异质性。本文研究了统计学习最佳动态治疗方案(DTR)，根据个体的历史指导每个阶段的最佳治疗分配。我们提出了一种基于观测数据的逐步双重强健方法，在顺序可忽略性假设下学习最佳DTR。该方法通过向后归纳解决了顺序治疗分配问题，在每一步中，我们结合倾向评分和行动值函数(Q函数)的估计量，构建了政策价值的增强反向概率加权估计量。

    arXiv:2404.00221v1 Announce Type: cross  Abstract: Many public policies and medical interventions involve dynamics in their treatment assignments, where treatments are sequentially assigned to the same individuals across multiple stages, and the effect of treatment at each stage is usually heterogeneous with respect to the history of prior treatments and associated characteristics. We study statistical learning of optimal dynamic treatment regimes (DTRs) that guide the optimal treatment assignment for each individual at each stage based on the individual's history. We propose a step-wise doubly-robust approach to learn the optimal DTR using observational data under the assumption of sequential ignorability. The approach solves the sequential treatment assignment problem through backward induction, where, at each step, we combine estimators of propensity scores and action-value functions (Q-functions) to construct augmented inverse probability weighting estimators of values of policies 
    
[^2]: 多步一致性模型

    Multistep Consistency Models

    [https://arxiv.org/abs/2403.06807](https://arxiv.org/abs/2403.06807)

    本文提出了多步一致性模型，通过在一致性模型和扩散模型之间插值，实现了采样速度和采样质量的平衡。

    

    扩散模型相对容易训练，但生成样本需要许多步骤。一致性模型更难训练，但可以在一个步骤中生成样本。本文提出了多步一致性模型：通过一致性模型和TRACT的统一，可以在一致性模型和扩散模型之间进行插值：在采样速度和采样质量之间取得平衡。具体来说，1步一致性模型是传统的一致性模型，而我们展示了$\infty$步一致性模型是扩散模型。多步一致性模型在实践中表现良好。将样本预算从单步增加到2-8步，我们可以更轻松地训练模型，生成更高质量的样本，同时保留大部分采样速度优势。在Imagenet 64上8步达到1.4的FID，在Imagenet128上8步达到2.1的FID。

    arXiv:2403.06807v1 Announce Type: new  Abstract: Diffusion models are relatively easy to train but require many steps to generate samples. Consistency models are far more difficult to train, but generate samples in a single step.   In this paper we propose Multistep Consistency Models: A unification between Consistency Models (Song et al., 2023) and TRACT (Berthelot et al., 2023) that can interpolate between a consistency model and a diffusion model: a trade-off between sampling speed and sampling quality. Specifically, a 1-step consistency model is a conventional consistency model whereas we show that a $\infty$-step consistency model is a diffusion model.   Multistep Consistency Models work really well in practice. By increasing the sample budget from a single step to 2-8 steps, we can train models more easily that generate higher quality samples, while retaining much of the sampling speed benefits. Notable results are 1.4 FID on Imagenet 64 in 8 step and 2.1 FID on Imagenet128 in 8 
    
[^3]: 混合输出高斯过程潜变量模型

    Mixed-Output Gaussian Process Latent Variable Models

    [https://arxiv.org/abs/2402.09122](https://arxiv.org/abs/2402.09122)

    本文提出了一种基于高斯过程潜变量模型的贝叶斯非参数方法，可以用于信号分离，并且能够处理包含纯组分信号加权和的情况，适用于光谱学和其他领域的多种应用。

    

    本文提出了一种贝叶斯非参数的信号分离方法，其中信号可以根据潜变量变化。我们的主要贡献是增加了高斯过程潜变量模型（GPLVMs），以包括每个数据点由已知数量的纯组分信号的加权和组成的情况，并观察多个输入位置。我们的框架允许使用各种关于每个观测权重的先验。这种灵活性使我们能够表示包括用于估计分数组成的总和为一约束和用于分类的二进制权重的用例。我们的贡献对于光谱学尤其相关，因为改变条件可能导致基础纯组分信号在样本之间变化。为了展示对光谱学和其他领域的适用性，我们考虑了几个应用：一个具有不同温度的近红外光谱数据集。

    arXiv:2402.09122v1 Announce Type: cross Abstract: This work develops a Bayesian non-parametric approach to signal separation where the signals may vary according to latent variables. Our key contribution is to augment Gaussian Process Latent Variable Models (GPLVMs) to incorporate the case where each data point comprises the weighted sum of a known number of pure component signals, observed across several input locations. Our framework allows the use of a range of priors for the weights of each observation. This flexibility enables us to represent use cases including sum-to-one constraints for estimating fractional makeup, and binary weights for classification. Our contributions are particularly relevant to spectroscopy, where changing conditions may cause the underlying pure component signals to vary from sample to sample. To demonstrate the applicability to both spectroscopy and other domains, we consider several applications: a near-infrared spectroscopy data set with varying temper
    
[^4]: 带有概率触发臂的情境组合赌博机

    Contextual Combinatorial Bandits with Probabilistically Triggered Arms. (arXiv:2303.17110v1 [cs.LG])

    [http://arxiv.org/abs/2303.17110](http://arxiv.org/abs/2303.17110)

    本文研究了带有概率触发臂的情境组合赌博机，在不同条件下设计了C$^2$-UCB-T算法和VAC$^2$-UCB算法，并分别导出了对应的遗憾值上限，为相关应用提供了理论支持。

    

    本研究探讨了在捕捉广泛应用范围的一系列平滑条件下的带有概率触发臂的情境组合赌博机(C$^2$MAB-T)，例如情境级联赌博机和情境最大化赌博机。在模拟触发概率(TPM)的条件下，我们设计了C$^2$-UCB-T算法，并提出了一种新的分析方法，实现了一个$\tilde{O}(d\sqrt{KT})$的遗憾值上限，消除了一个可能指数级增长的因子$O(1/p_{\min})$，其中$d$是情境的维数，$p_{\min}$是能被触发的任何臂的最小正概率，批大小$K$是每轮能被触发的臂的最大数量。在方差调制(VM)或触发概率和方差调制(TPVM)条件下，我们提出了一种新的方差自适应算法VAC$^2$-UCB，并导出了一个$\tilde{O}(d\sqrt{T})$的遗憾值上限，该上限与批大小$K$无关。作为一个有价值的副产品，我们发现我们的一个...

    We study contextual combinatorial bandits with probabilistically triggered arms (C$^2$MAB-T) under a variety of smoothness conditions that capture a wide range of applications, such as contextual cascading bandits and contextual influence maximization bandits. Under the triggering probability modulated (TPM) condition, we devise the C$^2$-UCB-T algorithm and propose a novel analysis that achieves an $\tilde{O}(d\sqrt{KT})$ regret bound, removing a potentially exponentially large factor $O(1/p_{\min})$, where $d$ is the dimension of contexts, $p_{\min}$ is the minimum positive probability that any arm can be triggered, and batch-size $K$ is the maximum number of arms that can be triggered per round. Under the variance modulated (VM) or triggering probability and variance modulated (TPVM) conditions, we propose a new variance-adaptive algorithm VAC$^2$-UCB and derive a regret bound $\tilde{O}(d\sqrt{T})$, which is independent of the batch-size $K$. As a valuable by-product, we find our a
    
[^5]: 分布转移的充分不变学习

    Sufficient Invariant Learning for Distribution Shift. (arXiv:2210.13533v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.13533](http://arxiv.org/abs/2210.13533)

    本文研究了分布转移情况下的充分不变学习，观察到之前的工作只学习了部分不变特征，我们提出了学习充分不变特征的重要性，并指出在分布转移时，从训练集中学习的部分不变特征可能不适用于测试集，限制了性能提升。

    

    机器学习算法在各种应用中展现出了卓越的性能。然而，在训练集和测试集的分布不同的情况下，保证性能仍然具有挑战性。为了改善分布转移情况下的性能，已经提出了一些方法，通过学习跨组或领域的不变特征来提高性能。然而，我们观察到之前的工作只部分地学习了不变特征。虽然先前的工作侧重于有限的不变特征，但我们首次提出了充分不变特征的重要性。由于只有训练集是经验性的，从训练集中学习得到的部分不变特征可能不存在于分布转移时的测试集中。因此，分布转移情况下的性能提高可能受到限制。本文认为从训练集中学习充分的不变特征对于分布转移情况至关重要。

    Machine learning algorithms have shown remarkable performance in diverse applications. However, it is still challenging to guarantee performance in distribution shifts when distributions of training and test datasets are different. There have been several approaches to improve the performance in distribution shift cases by learning invariant features across groups or domains. However, we observe that the previous works only learn invariant features partially. While the prior works focus on the limited invariant features, we first raise the importance of the sufficient invariant features. Since only training sets are given empirically, the learned partial invariant features from training sets might not be present in the test sets under distribution shift. Therefore, the performance improvement on distribution shifts might be limited. In this paper, we argue that learning sufficient invariant features from the training set is crucial for the distribution shift case. Concretely, we newly 
    

