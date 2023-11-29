# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Almost Equivariance via Lie Algebra Convolutions.](http://arxiv.org/abs/2310.13164) | 本文研究了几乎等变性的主题，并提供了一个不同于现有定义的几乎等变性定义，并通过利用李群的李代数给出了在模型中编码几乎等变性的实用方法。 |
| [^2] | [Kernel Learning in Ridge Regression "Automatically" Yields Exact Low Rank Solution.](http://arxiv.org/abs/2310.11736) | 该论文研究了核岭回归问题中核学习的低秩解的性质。在只有低维子空间对响应变量有解释能力的情况下，通过该方法可以自动得到精确的低秩解，无需额外的正则化。 |
| [^3] | [Lion Secretly Solves Constrained Optimization: As Lyapunov Predicts.](http://arxiv.org/abs/2310.05898) | Lion是通过程序搜索发现的新优化器，在训练大型AI模型方面表现出有希望的结果，具有更高的内存效率。尽管其理论基础不明确，但基于连续时间和离散时间分析，我们证明Lion是一种理论上新颖且有原则的方法，可在最小化一般损失函数的同时强制执行边界约束。 |
| [^4] | [Fantastic Generalization Measures are Nowhere to be Found.](http://arxiv.org/abs/2309.13658) | 本论文研究了过参数化情况下的泛化界限问题，通过分析多个界限发现在这种情况下无法找到紧致的界限来解释神经网络的出色性能。 |
| [^5] | [Geometry-Aware Adaptation for Pretrained Models.](http://arxiv.org/abs/2307.12226) | 本论文提出了一种简单的方法，利用标签之间的距离关系来调整已训练的模型，以可靠地预测新类别或改善零样本预测的性能，而无需额外的训练。 |
| [^6] | [Kernelized Reinforcement Learning with Order Optimal Regret Bounds.](http://arxiv.org/abs/2306.07745) | 该论文提出了一种称为$\pi$-KRVI的乐观修改方法，并使用核岭回归进行强化学习中的非线性函数逼近。论文证明了在一般设置下第一个最优遗憾保证，并相对于现有最优结果实现了显着的多项式低差距。 |
| [^7] | [Curve Your Enthusiasm: Concurvity Regularization in Differentiable Generalized Additive Models.](http://arxiv.org/abs/2305.11475) | 本文提供了一种共曲抑制正则化器，用于应对广义加性模型易受共错性的问题，通过惩罚非线性转换的特征变量的成对相关性，增强了模型的解释性。 |
| [^8] | [Constrained Optimization of Rank-One Functions with Indicator Variables.](http://arxiv.org/abs/2303.18158) | 本文提出了一种基于透视重构技术的紧凑扩展公式，用于解决涉及指标变量限制下决策变量支持集合的秩一凸函数的最小化问题。 |

# 详细

[^1]: 几乎等变性通过李代数卷积

    Almost Equivariance via Lie Algebra Convolutions. (arXiv:2310.13164v1 [cs.LG])

    [http://arxiv.org/abs/2310.13164](http://arxiv.org/abs/2310.13164)

    本文研究了几乎等变性的主题，并提供了一个不同于现有定义的几乎等变性定义，并通过利用李群的李代数给出了在模型中编码几乎等变性的实用方法。

    

    最近，在机器学习中，模型相对于群作用的等变性已成为一个重要的研究课题。然而，赋予一个架构具体的群等变性对模型所期望看到的数据变换类型施加了强大的先验。严格等变模型强制执行对称性，但真实世界的数据并不总是符合这样的严格等变性，可能是因为数据中的噪声或仅编码了近似或部分对称性的潜在物理定律。在这种情况下，严格等变性的先验实际上可能过于强大，导致模型在真实数据上表现不佳。因此，在这项工作中，我们研究了一个相关的主题，即几乎等变性。我们提供了一个与当前文献中现有定义不同的几乎等变性定义，并通过利用李群的李代数给出了在模型中编码几乎等变性的实用方法。

    Recently, the equivariance of models with respect to a group action has become an important topic of research in machine learning. However, imbuing an architecture with a specific group equivariance imposes a strong prior on the types of data transformations that the model expects to see. While strictly-equivariant models enforce symmetries, real-world data does not always conform to such strict equivariances, be it due to noise in the data or underlying physical laws that encode only approximate or partial symmetries. In such cases, the prior of strict equivariance can actually prove too strong and cause models to underperform on real-world data. Therefore, in this work we study a closely related topic, that of almost equivariance. We provide a definition of almost equivariance that differs from those extant in the current literature and give a practical method for encoding almost equivariance in models by appealing to the Lie algebra of a Lie group. Specifically, we define Lie algebr
    
[^2]: 在Ridge回归中，核学习“自动”给出精确的低秩解

    Kernel Learning in Ridge Regression "Automatically" Yields Exact Low Rank Solution. (arXiv:2310.11736v1 [math.ST])

    [http://arxiv.org/abs/2310.11736](http://arxiv.org/abs/2310.11736)

    该论文研究了核岭回归问题中核学习的低秩解的性质。在只有低维子空间对响应变量有解释能力的情况下，通过该方法可以自动得到精确的低秩解，无需额外的正则化。

    

    我们考虑形式为$(x,x') \mapsto \phi(\|x-x'\|^2_\Sigma)$且由参数$\Sigma$参数化的核函数。对于这样的核函数，我们研究了核岭回归问题的变体，它同时优化了预测函数和再现核希尔伯特空间的参数$\Sigma$。从这个核岭回归问题中学到的$\Sigma$的特征空间可以告诉我们协变量空间中哪些方向对预测是重要的。假设协变量只通过低维子空间（中心均值子空间）对响应变量有非零的解释能力，我们发现有很高的概率下有限样本核学习目标的全局最小化者也是低秩的。更具体地说，最小化$\Sigma$的秩有很高的概率被中心均值子空间的维度所限制。这个现象很有趣，因为低秩特性是在没有使用任何对$\Sigma$的显式正则化的情况下实现的，例如核范数正则化等。

    We consider kernels of the form $(x,x') \mapsto \phi(\|x-x'\|^2_\Sigma)$ parametrized by $\Sigma$. For such kernels, we study a variant of the kernel ridge regression problem which simultaneously optimizes the prediction function and the parameter $\Sigma$ of the reproducing kernel Hilbert space. The eigenspace of the $\Sigma$ learned from this kernel ridge regression problem can inform us which directions in covariate space are important for prediction.  Assuming that the covariates have nonzero explanatory power for the response only through a low dimensional subspace (central mean subspace), we find that the global minimizer of the finite sample kernel learning objective is also low rank with high probability. More precisely, the rank of the minimizing $\Sigma$ is with high probability bounded by the dimension of the central mean subspace. This phenomenon is interesting because the low rankness property is achieved without using any explicit regularization of $\Sigma$, e.g., nuclear
    
[^3]: 狮子秘密地解决受限制优化问题：正如李雅普诺夫所预测的。

    Lion Secretly Solves Constrained Optimization: As Lyapunov Predicts. (arXiv:2310.05898v1 [cs.LG])

    [http://arxiv.org/abs/2310.05898](http://arxiv.org/abs/2310.05898)

    Lion是通过程序搜索发现的新优化器，在训练大型AI模型方面表现出有希望的结果，具有更高的内存效率。尽管其理论基础不明确，但基于连续时间和离散时间分析，我们证明Lion是一种理论上新颖且有原则的方法，可在最小化一般损失函数的同时强制执行边界约束。

    

    通过程序搜索发现的新优化器Lion（进化的符号动量）在训练大型AI模型方面显示出有希望的结果。它在训练效果上与AdamW相当或更好，并具有更高的内存效率。正如我们可以从随机搜索程序的结果中期待的，Lion集成了几个现有算法的元素，包括符号动量、独立的权重衰减、Polak和Nesterov动量，但又不属于任何现有的理论基础优化器类别。因此，尽管Lion作为广泛任务的通用优化器表现良好，但其理论基础仍然不明确。这种缺乏理论的明确性限制了进一步增强和扩展Lion的可能性。本文旨在揭开Lion的神秘面纱。基于连续时间和离散时间分析，我们证明Lion是一种理论上新颖且有原则的方法，可在最小化一般损失函数$f(x)$的同时强制执行边界约束。

    Lion (Evolved Sign Momentum), a new optimizer discovered through program search, has shown promising results in training large AI models. It performs comparably or favorably to AdamW but with greater memory efficiency. As we can expect from the results of a random search program, Lion incorporates elements from several existing algorithms, including signed momentum, decoupled weight decay, Polak, and Nesterov momentum, but does not fit into any existing category of theoretically grounded optimizers. Thus, even though Lion appears to perform well as a general-purpose optimizer for a wide range of tasks, its theoretical basis remains uncertain. This lack of theoretical clarity limits opportunities to further enhance and expand Lion's efficacy.  This work aims to demystify Lion. Based on both continuous-time and discrete-time analysis, we demonstrate that Lion is a theoretically novel and principled approach for minimizing a general loss function $f(x)$ while enforcing a bound constraint 
    
[^4]: 无法找到出色的泛化度量方法

    Fantastic Generalization Measures are Nowhere to be Found. (arXiv:2309.13658v1 [cs.LG])

    [http://arxiv.org/abs/2309.13658](http://arxiv.org/abs/2309.13658)

    本论文研究了过参数化情况下的泛化界限问题，通过分析多个界限发现在这种情况下无法找到紧致的界限来解释神经网络的出色性能。

    

    过去的文献中提出了许多泛化界限作为解释神经网络在过参数化情况下泛化能力的潜在方法。然而，这些界限都不是紧致的。例如，在他们的论文“Fantastic Generalization Measures and Where to Find Them”中，Jiang等人（2020）检查了十几个泛化界限，并通过实验证明没有一个能够解释神经网络卓越的性能。这引出了一个问题，即是否有可能找到紧致的泛化界限。我们考虑了文献中常见的两种泛化界限：（1）依赖于训练集和学习算法输出的界限。文献中有多个这种类型的界限（例如基于范数和基于间隔的界限），但我们证明在过参数化的情况下，没有这样的界限能够一致地紧致；（2）依赖于训练集和测试集的界限。

    Numerous generalization bounds have been proposed in the literature as potential explanations for the ability of neural networks to generalize in the overparameterized setting. However, none of these bounds are tight. For instance, in their paper ``Fantastic Generalization Measures and Where to Find Them'', Jiang et al. (2020) examine more than a dozen generalization bounds, and show empirically that none of them imply guarantees that can explain the remarkable performance of neural networks. This raises the question of whether tight generalization bounds are at all possible. We consider two types of generalization bounds common in the literature: (1) bounds that depend on the training set and the output of the learning algorithm. There are multiple bounds of this type in the literature (e.g., norm-based and margin-based bounds), but we prove mathematically that no such bound can be uniformly tight in the overparameterized setting; (2) bounds that depend on the training set and on the 
    
[^5]: 面向预训练模型的几何感知自适应技术

    Geometry-Aware Adaptation for Pretrained Models. (arXiv:2307.12226v1 [cs.LG])

    [http://arxiv.org/abs/2307.12226](http://arxiv.org/abs/2307.12226)

    本论文提出了一种简单的方法，利用标签之间的距离关系来调整已训练的模型，以可靠地预测新类别或改善零样本预测的性能，而无需额外的训练。

    

    机器学习模型，包括著名的零样本模型，通常在仅具有较小比例标签空间的数据集上进行训练。这些标签空间通常使用度量来衡量标签之间的距离关系。我们提出了一种简单的方法来利用这些信息，将已训练的模型调整以可靠地预测新类别，或者在零样本预测的情况下改善性能，而无需额外的训练。我们的技术是标准预测规则的替代方案，在其中将argmax替换为Fréchet平均值。我们为这种方法提供了全面的理论分析，研究了（i）学习理论结果，权衡标签空间直径、样本复杂性和模型维度，（ii）表征可能预测任何未观察到的类别的所有情景的特征，（iii）一种最优的主动学习式下一类别选择过程，以获取最佳的训练类别。

    Machine learning models -- including prominent zero-shot models -- are often trained on datasets whose labels are only a small proportion of a larger label space. Such spaces are commonly equipped with a metric that relates the labels via distances between them. We propose a simple approach to exploit this information to adapt the trained model to reliably predict new classes -- or, in the case of zero-shot prediction, to improve its performance -- without any additional training. Our technique is a drop-in replacement of the standard prediction rule, swapping argmax with the Fr\'echet mean. We provide a comprehensive theoretical analysis for this approach, studying (i) learning-theoretic results trading off label space diameter, sample complexity, and model dimension, (ii) characterizations of the full range of scenarios in which it is possible to predict any unobserved class, and (iii) an optimal active learning-like next class selection procedure to obtain optimal training classes f
    
[^6]: 核化强化学习及其近似方法的优化

    Kernelized Reinforcement Learning with Order Optimal Regret Bounds. (arXiv:2306.07745v1 [cs.LG])

    [http://arxiv.org/abs/2306.07745](http://arxiv.org/abs/2306.07745)

    该论文提出了一种称为$\pi$-KRVI的乐观修改方法，并使用核岭回归进行强化学习中的非线性函数逼近。论文证明了在一般设置下第一个最优遗憾保证，并相对于现有最优结果实现了显着的多项式低差距。

    

    强化学习（RL）在各种具有复杂模型和大状态-行为空间的实际场景中显示出了实证的成功。但是，现有的分析结果通常集中于具有少量状态-行为或简单模型（例如线性建模状态-行为值函数）的设置。 为了推导有效处理更广泛值函数的大状态-行为空间的RL策略，一些最新工作考虑使用核岭回归进行非线性函数逼近。 我们提出了称为$\pi$-KRVI的方法，它是最小二乘值迭代的一种乐观修改，当状态-行为值函数由RKHS表示时。我们证明了在一般设置下第一个最优遗憾保证。我们的结果显示，在许多具有高度非光滑内核（例如神经切向内核或某些Mat\'ern内核）的情况下，相对于现有最优结果，存在显着的多项式低差距。

    Reinforcement learning (RL) has shown empirical success in various real world settings with complex models and large state-action spaces. The existing analytical results, however, typically focus on settings with a small number of state-actions or simple models such as linearly modeled state-action value functions. To derive RL policies that efficiently handle large state-action spaces with more general value functions, some recent works have considered nonlinear function approximation using kernel ridge regression. We propose $\pi$-KRVI, an optimistic modification of least-squares value iteration, when the state-action value function is represented by an RKHS. We prove the first order-optimal regret guarantees under a general setting. Our results show a significant polynomial in the number of episodes improvement over the state of the art. In particular, with highly non-smooth kernels (such as Neural Tangent kernel or some Mat\'ern kernels) the existing results lead to trivial (superl
    
[^7]: 曲线上扬：在可微广义加性模型中的共曲抑制正则化

    Curve Your Enthusiasm: Concurvity Regularization in Differentiable Generalized Additive Models. (arXiv:2305.11475v1 [cs.LG])

    [http://arxiv.org/abs/2305.11475](http://arxiv.org/abs/2305.11475)

    本文提供了一种共曲抑制正则化器，用于应对广义加性模型易受共错性的问题，通过惩罚非线性转换的特征变量的成对相关性，增强了模型的解释性。

    

    最近，由于广义加性模型（GAM）可表达目标变量为特征的非线性变换和解释性，其再次受到欢迎。尽管GAM目前备受热捧，但其易受共错性，即特征之间的（可能是非线性的）依赖性迄今为止大多被忽视。在这里，我们展示了共错性如何严重破坏GAM的解释性，并提出了一个解决方法：一个在非线性转换的特征变量的成对相关性上进行惩罚的概念简单但有效的正则化器。该过程适用于任何可微的加性模型，如神经加性模型或神经预言。并且通过消除自我抵消的特征贡献的歧义，增强了解释性。我们在合成和真实时间序列和表格数据集上验证了我们的正则化器的有效性。

    Generalized Additive Models (GAMs) have recently experienced a resurgence in popularity due to their interpretability, which arises from expressing the target value as a sum of non-linear transformations of the features. Despite the current enthusiasm for GAMs, their susceptibility to concurvity - i.e., (possibly non-linear) dependencies between the features - has hitherto been largely overlooked. Here, we demonstrate how concurvity can severly impair the interpretability of GAMs and propose a remedy: a conceptually simple, yet effective regularizer which penalizes pairwise correlations of the non-linearly transformed feature variables. This procedure is applicable to any differentiable additive model, such as Neural Additive Models or NeuralProphet, and enhances interpretability by eliminating ambiguities due to self-canceling feature contributions. We validate the effectiveness of our regularizer in experiments on synthetic as well as real-world datasets for time-series and tabular d
    
[^8]: 指标变量限制下秩一函数的约束优化

    Constrained Optimization of Rank-One Functions with Indicator Variables. (arXiv:2303.18158v1 [math.OC])

    [http://arxiv.org/abs/2303.18158](http://arxiv.org/abs/2303.18158)

    本文提出了一种基于透视重构技术的紧凑扩展公式，用于解决涉及指标变量限制下决策变量支持集合的秩一凸函数的最小化问题。

    

    在各种机器学习应用中，涉及到通过约束来建模决策变量支持集合的秩一凸函数的最小化的优化问题。这些问题通常采用指标变量来识别连续变量的支持。本文通过透视重构技术研究了这些问题的紧凑扩展公式。与大多数先前的研究依赖于支持函数参数和离散规划技术以提供凸包结果不同，我们提出了一种构造方法，利用透视函数引起的隐藏圆锥结构。为此，我们首先针对每个圆锥约束涉及独立连续变量的线性函数和一组二元变量的一般圆锥混合二进制集合建立了一个凸包结果。然后，我们展示了与应对epi相关的集合的扩展表示形式。

    Optimization problems involving minimization of a rank-one convex function over constraints modeling restrictions on the support of the decision variables emerge in various machine learning applications. These problems are often modeled with indicator variables for identifying the support of the continuous variables. In this paper we investigate compact extended formulations for such problems through perspective reformulation techniques. In contrast to the majority of previous work that relies on support function arguments and disjunctive programming techniques to provide convex hull results, we propose a constructive approach that exploits a hidden conic structure induced by perspective functions. To this end, we first establish a convex hull result for a general conic mixed-binary set in which each conic constraint involves a linear function of independent continuous variables and a set of binary variables. We then demonstrate that extended representations of sets associated with epi
    

