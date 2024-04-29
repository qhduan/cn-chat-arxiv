# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nonparametric consistency for maximum likelihood estimation and clustering based on mixtures of elliptically-symmetric distributions](https://arxiv.org/abs/2311.06108) | 展示了椭圆对称分布混合的最大似然估计的一致性，为基于非参数分布的聚类提供了理论依据。 |
| [^2] | [Conditional Variational Diffusion Models.](http://arxiv.org/abs/2312.02246) | 该论文提出了一种新的条件变分扩散模型，通过学习调度作为训练过程的一部分，解决了扩散模型的敏感性问题，并且能够适应不同的应用场景，提供高质量的解决方案。 |
| [^3] | [Outlier-Robust Tensor Low-Rank Representation for Data Clustering.](http://arxiv.org/abs/2307.09055) | 本文提出了一种异常鲁棒张量低秩表示方法，用于同时检测异常值和进行数据聚类。该方法基于张量奇异值分解（t-SVD）代数框架，并在较弱条件下具有恢复干净数据的行空间和检测异常值的可证明性能保证。此外，还提出了扩展方法以处理数据部分缺失的情况。 |
| [^4] | [Adversarial Estimation of Riesz Representers.](http://arxiv.org/abs/2101.00009) | 我们提出了一个敌对框架，使用通用函数空间来估计Riesz Representer，并且证明了非渐近均方速率以及渐近正态性的条件。这个条件使得在机器学习中进行推断时无需样本分割，并且能够提高有限样本性能。 |

# 详细

[^1]: 基于椭圆对称分布混合的最大似然估计和聚类的非参数一致性

    Nonparametric consistency for maximum likelihood estimation and clustering based on mixtures of elliptically-symmetric distributions

    [https://arxiv.org/abs/2311.06108](https://arxiv.org/abs/2311.06108)

    展示了椭圆对称分布混合的最大似然估计的一致性，为基于非参数分布的聚类提供了理论依据。

    

    该论文展示了椭圆对称分布混合的最大似然估计器对其总体版本的一致性，其中潜在分布P是非参数的，并不一定属于估计器所基于的混合类别。当P是足够分离但非参数的分布混合时，表明了估计器的总体版本的组分对应于P的良好分离组分。这为在P具有良好分离子总体的情况下使用这样的估计器进行聚类分析提供了一些理论上的理据，即使这些子总体与混合模型所假设的不同。

    arXiv:2311.06108v2 Announce Type: replace-cross  Abstract: The consistency of the maximum likelihood estimator for mixtures of elliptically-symmetric distributions for estimating its population version is shown, where the underlying distribution $P$ is nonparametric and does not necessarily belong to the class of mixtures on which the estimator is based. In a situation where $P$ is a mixture of well enough separated but nonparametric distributions it is shown that the components of the population version of the estimator correspond to the well separated components of $P$. This provides some theoretical justification for the use of such estimators for cluster analysis in case that $P$ has well separated subpopulations even if these subpopulations differ from what the mixture model assumes.
    
[^2]: 条件变分扩散模型

    Conditional Variational Diffusion Models. (arXiv:2312.02246v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.02246](http://arxiv.org/abs/2312.02246)

    该论文提出了一种新的条件变分扩散模型，通过学习调度作为训练过程的一部分，解决了扩散模型的敏感性问题，并且能够适应不同的应用场景，提供高质量的解决方案。

    

    逆问题旨在从观测中确定参数，这是工程和科学中的一个关键任务。最近，生成模型，特别是扩散模型，因其能够产生逼真的解决方案和良好的数学特性而在这一领域中越来越受欢迎。尽管取得了成功，但扩散模型的一个重要缺点是对方差调度的选择敏感，该调度控制着扩散过程的动态。为特定应用程序微调这个调度是至关重要的，但时间成本高昂，并且不能保证最优结果。我们提出了一种新颖的方法，将学习调度作为训练过程的一部分。我们的方法支持对数据的概率条件，提供高质量的解决方案，并且具有灵活性，能够在最小的开销下适应不同的应用。这种方法在两个不相关的逆问题中进行了测试：超分辨率显微镜和定量相位成像，结果表明比较或更好。

    Inverse problems aim to determine parameters from observations, a crucial task in engineering and science. Lately, generative models, especially diffusion models, have gained popularity in this area for their ability to produce realistic solutions and their good mathematical properties. Despite their success, an important drawback of diffusion models is their sensitivity to the choice of variance schedule, which controls the dynamics of the diffusion process. Fine-tuning this schedule for specific applications is crucial but time-costly and does not guarantee an optimal result. We propose a novel approach for learning the schedule as part of the training process. Our method supports probabilistic conditioning on data, provides high-quality solutions, and is flexible, proving able to adapt to different applications with minimum overhead. This approach is tested in two unrelated inverse problems: super-resolution microscopy and quantitative phase imaging, yielding comparable or superior 
    
[^3]: 异常鲁棒张量低秩表示用于数据聚类

    Outlier-Robust Tensor Low-Rank Representation for Data Clustering. (arXiv:2307.09055v1 [stat.ML])

    [http://arxiv.org/abs/2307.09055](http://arxiv.org/abs/2307.09055)

    本文提出了一种异常鲁棒张量低秩表示方法，用于同时检测异常值和进行数据聚类。该方法基于张量奇异值分解（t-SVD）代数框架，并在较弱条件下具有恢复干净数据的行空间和检测异常值的可证明性能保证。此外，还提出了扩展方法以处理数据部分缺失的情况。

    

    低秩张量分析在许多实际应用中受到广泛关注。然而，张量数据经常受到异常值或样本特定的污染。如何恢复被异常值损坏的张量数据并进行数据聚类仍然是一个具有挑战性的问题。本文基于张量奇异值分解（t-SVD）代数框架，提出了一种用于同时检测异常值和张量数据聚类的异常鲁棒张量低秩表示（OR-TLRR）方法。该方法受到最近提出的满足一定条件的可逆线性变换引起的张量张量积的启发。对于带有任意异常值污染的张量观测，OR-TLRR在较弱条件下能够确切恢复干净数据的行空间并检测异常值。此外，还提出了OR-TLRR的扩展来处理数据部分缺失的情况。

    Low-rank tensor analysis has received widespread attention with many practical applications. However, the tensor data are often contaminated by outliers or sample-specific corruptions. How to recover the tensor data that are corrupted by outliers and perform data clustering remains a challenging problem. This paper develops an outlier-robust tensor low-rank representation (OR-TLRR) method for simultaneous outlier detection and tensor data clustering based on the tensor singular value decomposition (t-SVD) algebraic framework. It is motivated by the recently proposed tensor-tensor product induced by invertible linear transforms that satisfy certain conditions. For tensor observations with arbitrary outlier corruptions, OR-TLRR has provable performance guarantee for exactly recovering the row space of clean data and detecting outliers under mild conditions. Moreover, an extension of OR-TLRR is also proposed to handle the case when parts of the data are missing. Finally, extensive experim
    
[^4]: 对Riesz Representer的敌对估计

    Adversarial Estimation of Riesz Representers. (arXiv:2101.00009v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2101.00009](http://arxiv.org/abs/2101.00009)

    我们提出了一个敌对框架，使用通用函数空间来估计Riesz Representer，并且证明了非渐近均方速率以及渐近正态性的条件。这个条件使得在机器学习中进行推断时无需样本分割，并且能够提高有限样本性能。

    

    许多因果和结构参数是基于底层回归的线性泛函。Riesz Representer是半参数线性泛函渐近方差的关键组成部分。我们提出了一个敌对框架，使用通用函数空间来估计Riesz Representer。我们证明了一个非渐近均方速率，其中涉及一个称为临界半径的抽象量，然后将其专门应用于神经网络、随机森林和再生核希尔伯特空间作为主要案例。此外，我们使用临界半径理论来证明了渐近正态性，而不需要样本分割，揭示了一种“复杂度-速率鲁棒性”条件。这个条件具有实际后果：在几个机器学习设置中，可以实现无需样本分割的推断，这可能会提高有限样本性能。我们的估计器在高度非线性的模拟中实现了名义覆盖率。

    Many causal and structural parameters are linear functionals of an underlying regression. The Riesz representer is a key component in the asymptotic variance of a semiparametrically estimated linear functional. We propose an adversarial framework to estimate the Riesz representer using general function spaces. We prove a nonasymptotic mean square rate in terms of an abstract quantity called the critical radius, then specialize it for neural networks, random forests, and reproducing kernel Hilbert spaces as leading cases. Furthermore, we use critical radius theory -- in place of Donsker theory -- to prove asymptotic normality without sample splitting, uncovering a ``complexity-rate robustness'' condition. This condition has practical consequences: inference without sample splitting is possible in several machine learning settings, which may improve finite sample performance compared to sample splitting. Our estimators achieve nominal coverage in highly nonlinear simulations where previo
    

