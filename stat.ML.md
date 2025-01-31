# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Semi-Supervised Deep Sobolev Regression: Estimation, Variable Selection and Beyond.](http://arxiv.org/abs/2401.04535) | 我们提出了一种半监督深度Sobolev回归器，利用深度神经网络进行梯度范数正则化，可以同时估计回归函数和其梯度，即使存在显著领域变化。这在半监督学习中利用无标签数据方面具有可证优势。 |
| [^2] | [Kernel Learning in Ridge Regression "Automatically" Yields Exact Low Rank Solution.](http://arxiv.org/abs/2310.11736) | 该论文研究了核岭回归问题中核学习的低秩解的性质。在只有低维子空间对响应变量有解释能力的情况下，通过该方法可以自动得到精确的低秩解，无需额外的正则化。 |
| [^3] | [Efficient Methods for Non-stationary Online Learning.](http://arxiv.org/abs/2309.08911) | 这项工作提出了一种针对非平稳在线学习的高效方法，通过降低每轮投影的数量来优化动态遗憾和自适应遗憾的计算复杂性。 |
| [^4] | [Virtual Human Generative Model: Masked Modeling Approach for Learning Human Characteristics.](http://arxiv.org/abs/2306.10656) | 本论文提出了一种名为VHGM的深度生成模型，基于掩码建模的方法来学习健康属性、生活方式和人格之间的关系。通过使用异构表格数据集，VHGM有效地学习了超过1,800个属性。该模型具有潜在的应用前景，例如用于医疗属性的虚拟测量和生活方式的假设验证。 |
| [^5] | [Dynamic treatment effects: high-dimensional inference under model misspecification.](http://arxiv.org/abs/2111.06818) | 本文提出了一种新的鲁棒估计方法来解决动态治疗效应估计中的挑战，提高了在模型错误下的高维环境中的估计鲁棒性和可靠性。 |

# 详细

[^1]: 半监督深度Sobolev回归: 估计、变量选择及其他

    Semi-Supervised Deep Sobolev Regression: Estimation, Variable Selection and Beyond. (arXiv:2401.04535v1 [stat.ML])

    [http://arxiv.org/abs/2401.04535](http://arxiv.org/abs/2401.04535)

    我们提出了一种半监督深度Sobolev回归器，利用深度神经网络进行梯度范数正则化，可以同时估计回归函数和其梯度，即使存在显著领域变化。这在半监督学习中利用无标签数据方面具有可证优势。

    

    我们提出了SDORE，一种半监督深度Sobolev回归器，用于非参数估计潜在的回归函数及其梯度。SDORE使用深度神经网络来最小化经验风险，并采用梯度范数正则化，允许对无标签数据计算梯度范数。我们对SDORE的收敛速度进行了全面分析，并建立了回归函数的最小化最优速率。重要的是，在存在显著领域变化的情况下，我们还推导出了关联的插值梯度估计器的收敛速度。这些理论结果为选择正则化参数和确定神经网络的大小提供了有价值的先验指导，并展示了在半监督学习中利用无标签数据的可证优势。据我们所知，SDORE是第一个同时估计回归函数及其梯度的可证神经网络方法，具有多样化的应用。

    We propose SDORE, a semi-supervised deep Sobolev regressor, for the nonparametric estimation of the underlying regression function and its gradient. SDORE employs deep neural networks to minimize empirical risk with gradient norm regularization, allowing computation of the gradient norm on unlabeled data. We conduct a comprehensive analysis of the convergence rates of SDORE and establish a minimax optimal rate for the regression function. Crucially, we also derive a convergence rate for the associated plug-in gradient estimator, even in the presence of significant domain shift. These theoretical findings offer valuable prior guidance for selecting regularization parameters and determining the size of the neural network, while showcasing the provable advantage of leveraging unlabeled data in semi-supervised learning. To the best of our knowledge, SDORE is the first provable neural network-based approach that simultaneously estimates the regression function and its gradient, with diverse
    
[^2]: 在Ridge回归中，核学习“自动”给出精确的低秩解

    Kernel Learning in Ridge Regression "Automatically" Yields Exact Low Rank Solution. (arXiv:2310.11736v1 [math.ST])

    [http://arxiv.org/abs/2310.11736](http://arxiv.org/abs/2310.11736)

    该论文研究了核岭回归问题中核学习的低秩解的性质。在只有低维子空间对响应变量有解释能力的情况下，通过该方法可以自动得到精确的低秩解，无需额外的正则化。

    

    我们考虑形式为$(x,x') \mapsto \phi(\|x-x'\|^2_\Sigma)$且由参数$\Sigma$参数化的核函数。对于这样的核函数，我们研究了核岭回归问题的变体，它同时优化了预测函数和再现核希尔伯特空间的参数$\Sigma$。从这个核岭回归问题中学到的$\Sigma$的特征空间可以告诉我们协变量空间中哪些方向对预测是重要的。假设协变量只通过低维子空间（中心均值子空间）对响应变量有非零的解释能力，我们发现有很高的概率下有限样本核学习目标的全局最小化者也是低秩的。更具体地说，最小化$\Sigma$的秩有很高的概率被中心均值子空间的维度所限制。这个现象很有趣，因为低秩特性是在没有使用任何对$\Sigma$的显式正则化的情况下实现的，例如核范数正则化等。

    We consider kernels of the form $(x,x') \mapsto \phi(\|x-x'\|^2_\Sigma)$ parametrized by $\Sigma$. For such kernels, we study a variant of the kernel ridge regression problem which simultaneously optimizes the prediction function and the parameter $\Sigma$ of the reproducing kernel Hilbert space. The eigenspace of the $\Sigma$ learned from this kernel ridge regression problem can inform us which directions in covariate space are important for prediction.  Assuming that the covariates have nonzero explanatory power for the response only through a low dimensional subspace (central mean subspace), we find that the global minimizer of the finite sample kernel learning objective is also low rank with high probability. More precisely, the rank of the minimizing $\Sigma$ is with high probability bounded by the dimension of the central mean subspace. This phenomenon is interesting because the low rankness property is achieved without using any explicit regularization of $\Sigma$, e.g., nuclear
    
[^3]: 非平稳在线学习的高效方法

    Efficient Methods for Non-stationary Online Learning. (arXiv:2309.08911v1 [cs.LG])

    [http://arxiv.org/abs/2309.08911](http://arxiv.org/abs/2309.08911)

    这项工作提出了一种针对非平稳在线学习的高效方法，通过降低每轮投影的数量来优化动态遗憾和自适应遗憾的计算复杂性。

    

    非平稳在线学习近年来引起了广泛关注。特别是在非平稳环境中，动态遗憾和自适应遗憾被提出作为在线凸优化的两个原则性性能度量。为了优化它们，通常采用两层在线集成，由于非平稳性的固有不确定性，其中维护一组基学习器，并采用元算法在运行过程中跟踪最佳学习器。然而，这种两层结构引发了关于计算复杂性的担忧 -这些方法通常同时维护$\mathcal{O}(\log T)$个基学习器，对于一个$T$轮在线游戏，因此每轮执行多次投影到可行域上，当域很复杂时，这成为计算瓶颈。在本文中，我们提出了优化动态遗憾和自适应遗憾的高效方法，将每轮的投影次数从$\mathcal{O}(\log T)$降低到...

    Non-stationary online learning has drawn much attention in recent years. In particular, dynamic regret and adaptive regret are proposed as two principled performance measures for online convex optimization in non-stationary environments. To optimize them, a two-layer online ensemble is usually deployed due to the inherent uncertainty of the non-stationarity, in which a group of base-learners are maintained and a meta-algorithm is employed to track the best one on the fly. However, the two-layer structure raises the concern about the computational complexity -- those methods typically maintain $\mathcal{O}(\log T)$ base-learners simultaneously for a $T$-round online game and thus perform multiple projections onto the feasible domain per round, which becomes the computational bottleneck when the domain is complicated. In this paper, we present efficient methods for optimizing dynamic regret and adaptive regret, which reduce the number of projections per round from $\mathcal{O}(\log T)$ t
    
[^4]: 虚拟人类生成模型：基于掩码建模的方法来学习人类特征

    Virtual Human Generative Model: Masked Modeling Approach for Learning Human Characteristics. (arXiv:2306.10656v1 [cs.LG])

    [http://arxiv.org/abs/2306.10656](http://arxiv.org/abs/2306.10656)

    本论文提出了一种名为VHGM的深度生成模型，基于掩码建模的方法来学习健康属性、生活方式和人格之间的关系。通过使用异构表格数据集，VHGM有效地学习了超过1,800个属性。该模型具有潜在的应用前景，例如用于医疗属性的虚拟测量和生活方式的假设验证。

    

    识别医疗属性、生活方式和人格之间的关系对于理解和改善身体和精神状况至关重要。本文提出了一种名为虚拟人类生成模型（VHGM）的机器学习模型，用于估计有关医疗保健、生活方式和个性的属性。VHGM是一个深度生成模型，使用掩码建模训练，在已知属性的条件下学习属性的联合分布。利用异构表格数据集，VHGM高效地学习了超过1,800个属性。我们数值评估了VHGM及其训练技术的性能。作为VHGM的概念验证，我们提出了几个应用程序，演示了用户情境，例如医疗属性的虚拟测量和生活方式的假设验证。

    Identifying the relationship between healthcare attributes, lifestyles, and personality is vital for understanding and improving physical and mental conditions. Machine learning approaches are promising for modeling their relationships and offering actionable suggestions. In this paper, we propose Virtual Human Generative Model (VHGM), a machine learning model for estimating attributes about healthcare, lifestyles, and personalities. VHGM is a deep generative model trained with masked modeling to learn the joint distribution of attributes conditioned on known ones. Using heterogeneous tabular datasets, VHGM learns more than 1,800 attributes efficiently. We numerically evaluate the performance of VHGM and its training techniques. As a proof-of-concept of VHGM, we present several applications demonstrating user scenarios, such as virtual measurements of healthcare attributes and hypothesis verifications of lifestyles.
    
[^5]: 动态治疗效应：模型错误下的高维推断

    Dynamic treatment effects: high-dimensional inference under model misspecification. (arXiv:2111.06818v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2111.06818](http://arxiv.org/abs/2111.06818)

    本文提出了一种新的鲁棒估计方法来解决动态治疗效应估计中的挑战，提高了在模型错误下的高维环境中的估计鲁棒性和可靠性。

    

    估计动态治疗效应在各个学科中都是至关重要的，可以提供有关干预的时变因果影响的微妙见解。然而，由于“维数灾难”和时变混杂的存在，这种估计存在着挑战，可能导致估计偏误。此外，正确地规定日益增多的治疗分配和多重暴露的结果模型似乎过于复杂。鉴于这些挑战，双重鲁棒性的概念，在允许模型错误的情况下，是非常有价值的，然而在实际应用中并没有实现。本文通过提出新的鲁棒估计方法来解决这个问题，同时对治疗分配和结果模型进行鲁棒估计。我们提出了一种“序列模型双重鲁棒性”的解决方案，证明了当每个时间暴露都是双重鲁棒性的时，可以在多个时间点上实现双重鲁棒性。这种方法提高了高维环境下动态治疗效应估计的鲁棒性和可靠性。

    Estimating dynamic treatment effects is essential across various disciplines, offering nuanced insights into the time-dependent causal impact of interventions. However, this estimation presents challenges due to the "curse of dimensionality" and time-varying confounding, which can lead to biased estimates. Additionally, correctly specifying the growing number of treatment assignments and outcome models with multiple exposures seems overly complex. Given these challenges, the concept of double robustness, where model misspecification is permitted, is extremely valuable, yet unachieved in practical applications. This paper introduces a new approach by proposing novel, robust estimators for both treatment assignments and outcome models. We present a "sequential model double robust" solution, demonstrating that double robustness over multiple time points can be achieved when each time exposure is doubly robust. This approach improves the robustness and reliability of dynamic treatment effe
    

