# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Theoretical and experimental study of SMOTE: limitations and comparisons of rebalancing strategies](https://arxiv.org/abs/2402.03819) | SMOTE是一种处理不平衡数据集的常用重新平衡策略，它通过复制原始少数样本来重新生成原始分布。本研究证明了SMOTE的密度在少数样本分布的边界附近逐渐减小，从而验证了BorderLine SMOTE策略的合理性。此外，研究还提出了两种新的SMOTE相关策略，并与其他重新平衡方法进行了比较。最终发现，在数据集极度不平衡的情况下，SMOTE、提出的方法或欠采样程序是最佳的策略。 |
| [^2] | [MCMC-Correction of Score-Based Diffusion Models for Model Composition.](http://arxiv.org/abs/2307.14012) | 本文提出了一种修正基于得分的扩散模型的方法，使其能够与各种MCMC方法结合，从而实现模型组合和进行更好的采样。 |
| [^3] | [Infinite-Dimensional Diffusion Models.](http://arxiv.org/abs/2302.10130) | 该论文提出了一种在无限维度中直接制定扩散基于的生成模型的方法，相比于传统的先离散化再应用扩散模型的方法，这种方法能够避免参数细化导致算法性能下降，同时提供了维度无关的距离界限，为无限维扩散模型设计提供了准则。 |

# 详细

[^1]: SMOTE的理论和实验研究：关于重新平衡策略的限制和比较

    Theoretical and experimental study of SMOTE: limitations and comparisons of rebalancing strategies

    [https://arxiv.org/abs/2402.03819](https://arxiv.org/abs/2402.03819)

    SMOTE是一种处理不平衡数据集的常用重新平衡策略，它通过复制原始少数样本来重新生成原始分布。本研究证明了SMOTE的密度在少数样本分布的边界附近逐渐减小，从而验证了BorderLine SMOTE策略的合理性。此外，研究还提出了两种新的SMOTE相关策略，并与其他重新平衡方法进行了比较。最终发现，在数据集极度不平衡的情况下，SMOTE、提出的方法或欠采样程序是最佳的策略。

    

    SMOTE（Synthetic Minority Oversampling Technique）是处理不平衡数据集常用的重新平衡策略。我们证明了在渐进情况下，SMOTE（默认参数）通过简单复制原始少数样本来重新生成原始分布。我们还证明了在少数样本分布的支持边界附近，SMOTE的密度会减小，从而验证了常见的BorderLine SMOTE策略。随后，我们提出了两种新的SMOTE相关策略，并将它们与现有的重新平衡方法进行了比较。我们发现，只有当数据集极度不平衡时才需要重新平衡策略。对于这种数据集，SMOTE、我们提出的方法或欠采样程序是最佳的策略。

    Synthetic Minority Oversampling Technique (SMOTE) is a common rebalancing strategy for handling imbalanced data sets. Asymptotically, we prove that SMOTE (with default parameter) regenerates the original distribution by simply copying the original minority samples. We also prove that SMOTE density vanishes near the boundary of the support of the minority distribution, therefore justifying the common BorderLine SMOTE strategy. Then we introduce two new SMOTE-related strategies, and compare them with state-of-the-art rebalancing procedures. We show that rebalancing strategies are only required when the data set is highly imbalanced. For such data sets, SMOTE, our proposals, or undersampling procedures are the best strategies.
    
[^2]: MCMC-修正基于得分的扩散模型用于模型组合

    MCMC-Correction of Score-Based Diffusion Models for Model Composition. (arXiv:2307.14012v1 [stat.ML])

    [http://arxiv.org/abs/2307.14012](http://arxiv.org/abs/2307.14012)

    本文提出了一种修正基于得分的扩散模型的方法，使其能够与各种MCMC方法结合，从而实现模型组合和进行更好的采样。

    

    扩散模型可以用得分或能量函数来参数化。能量参数化具有更好的理论特性，主要是它可以通过在提议样本中总能量的变化基于Metropolis-Hastings修正步骤来进行扩展采样过程。然而，它似乎产生了稍微较差的性能，更重要的是，由于基于得分的扩散模型的普遍流行，现有的预训练能量参数化模型的可用性受到限制。这种限制削弱了模型组合的目的，即将预训练模型组合起来从新分布中进行采样。然而，我们的提议建议保留得分参数化，而是通过对得分函数进行线积分来计算基于能量的接受概率。这使我们能够重用现有的扩散模型，并将反向过程与各种马尔可夫链蒙特卡罗（MCMC）方法组合起来。

    Diffusion models can be parameterised in terms of either a score or an energy function. The energy parameterisation has better theoretical properties, mainly that it enables an extended sampling procedure with a Metropolis--Hastings correction step, based on the change in total energy in the proposed samples. However, it seems to yield slightly worse performance, and more importantly, due to the widespread popularity of score-based diffusion, there are limited availability of off-the-shelf pre-trained energy-based ones. This limitation undermines the purpose of model composition, which aims to combine pre-trained models to sample from new distributions. Our proposal, however, suggests retaining the score parameterization and instead computing the energy-based acceptance probability through line integration of the score function. This allows us to re-use existing diffusion models and still combine the reverse process with various Markov-Chain Monte Carlo (MCMC) methods. We evaluate our 
    
[^3]: 无限维扩散模型

    Infinite-Dimensional Diffusion Models. (arXiv:2302.10130v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.10130](http://arxiv.org/abs/2302.10130)

    该论文提出了一种在无限维度中直接制定扩散基于的生成模型的方法，相比于传统的先离散化再应用扩散模型的方法，这种方法能够避免参数细化导致算法性能下降，同时提供了维度无关的距离界限，为无限维扩散模型设计提供了准则。

    

    扩散模型对于许多应用领域都产生了深远的影响，包括那些数据本质上是无限维的领域，如图像或时间序列。标准方法是首先离散化数据，然后将扩散模型应用于离散的数据。然而，这种方法在细化离散化参数时通常会导致算法性能下降。在本文中，我们直接在无限维度中制定基于扩散的生成模型，并将其应用于函数的生成建模。我们证明了我们的公式在无限维度环境中是良好定义的，并提供了从样本到目标测度的维度无关的距离界限。利用我们的理论，我们还制定了无限维扩散模型设计的准则。对于图像分布，这些准则与当前用于扩散模型的经典选择一致。对于其他分布...

    Diffusion models have had a profound impact on many application areas, including those where data are intrinsically infinite-dimensional, such as images or time series. The standard approach is first to discretize and then to apply diffusion models to the discretized data. While such approaches are practically appealing, the performance of the resulting algorithms typically deteriorates as discretization parameters are refined. In this paper, we instead directly formulate diffusion-based generative models in infinite dimensions and apply them to the generative modeling of functions. We prove that our formulations are well posed in the infinite-dimensional setting and provide dimension-independent distance bounds from the sample to the target measure. Using our theory, we also develop guidelines for the design of infinite-dimensional diffusion models. For image distributions, these guidelines are in line with the canonical choices currently made for diffusion models. For other distribut
    

