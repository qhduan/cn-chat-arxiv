# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Theoretical and experimental study of SMOTE: limitations and comparisons of rebalancing strategies](https://arxiv.org/abs/2402.03819) | SMOTE是一种处理不平衡数据集的常用重新平衡策略，它通过复制原始少数样本来重新生成原始分布。本研究证明了SMOTE的密度在少数样本分布的边界附近逐渐减小，从而验证了BorderLine SMOTE策略的合理性。此外，研究还提出了两种新的SMOTE相关策略，并与其他重新平衡方法进行了比较。最终发现，在数据集极度不平衡的情况下，SMOTE、提出的方法或欠采样程序是最佳的策略。 |
| [^2] | [The objective function equality property of infoGAN for two-layer network.](http://arxiv.org/abs/2310.00443) | 这项研究证明了在infoGAN中，辨别器和生成器的样本数量趋向无穷时，两个目标函数变得等价。 |
| [^3] | [Why Clean Generalization and Robust Overfitting Both Happen in Adversarial Training.](http://arxiv.org/abs/2306.01271) | 对抗训练是训练深度神经网络抗击对抗扰动的标准方法, 其学习机制导致干净泛化和强健过拟合现象同时发生。 |
| [^4] | [Estimate-Then-Optimize Versus Integrated-Estimation-Optimization: A Stochastic Dominance Perspective.](http://arxiv.org/abs/2304.06833) | 本文提出，当模型类足够丰富以涵盖真实情况时，非线性问题的“先估计再优化”方法优于集成方法，包括优化间隙的渐进优势的均值，所有其他时刻和整个渐进分布。 |

# 详细

[^1]: SMOTE的理论和实验研究：关于重新平衡策略的限制和比较

    Theoretical and experimental study of SMOTE: limitations and comparisons of rebalancing strategies

    [https://arxiv.org/abs/2402.03819](https://arxiv.org/abs/2402.03819)

    SMOTE是一种处理不平衡数据集的常用重新平衡策略，它通过复制原始少数样本来重新生成原始分布。本研究证明了SMOTE的密度在少数样本分布的边界附近逐渐减小，从而验证了BorderLine SMOTE策略的合理性。此外，研究还提出了两种新的SMOTE相关策略，并与其他重新平衡方法进行了比较。最终发现，在数据集极度不平衡的情况下，SMOTE、提出的方法或欠采样程序是最佳的策略。

    

    SMOTE（Synthetic Minority Oversampling Technique）是处理不平衡数据集常用的重新平衡策略。我们证明了在渐进情况下，SMOTE（默认参数）通过简单复制原始少数样本来重新生成原始分布。我们还证明了在少数样本分布的支持边界附近，SMOTE的密度会减小，从而验证了常见的BorderLine SMOTE策略。随后，我们提出了两种新的SMOTE相关策略，并将它们与现有的重新平衡方法进行了比较。我们发现，只有当数据集极度不平衡时才需要重新平衡策略。对于这种数据集，SMOTE、我们提出的方法或欠采样程序是最佳的策略。

    Synthetic Minority Oversampling Technique (SMOTE) is a common rebalancing strategy for handling imbalanced data sets. Asymptotically, we prove that SMOTE (with default parameter) regenerates the original distribution by simply copying the original minority samples. We also prove that SMOTE density vanishes near the boundary of the support of the minority distribution, therefore justifying the common BorderLine SMOTE strategy. Then we introduce two new SMOTE-related strategies, and compare them with state-of-the-art rebalancing procedures. We show that rebalancing strategies are only required when the data set is highly imbalanced. For such data sets, SMOTE, our proposals, or undersampling procedures are the best strategies.
    
[^2]: infoGAN的两层网络的目标函数等式属性

    The objective function equality property of infoGAN for two-layer network. (arXiv:2310.00443v1 [stat.ML])

    [http://arxiv.org/abs/2310.00443](http://arxiv.org/abs/2310.00443)

    这项研究证明了在infoGAN中，辨别器和生成器的样本数量趋向无穷时，两个目标函数变得等价。

    

    信息最大化生成对抗网络(infoGAN)可以理解为涉及两个网络(辨别器和生成器)的极小化极大问题，其中包含了互信息函数。infoGAN包括多种组件，包括潜在变量、互信息和目标函数。本研究证明，在辨别器和生成器样本数量趋向无穷时，infoGAN中的两个目标函数变得等价。这种等价关系是通过考虑目标函数的经验版本和总体版本之间的差异来建立的。这个差异的界限由辨别器和生成器函数类的Rademacher复杂度决定。此外，使用具有Lipschitz和非递减激活函数的两层网络来验证这个等式。

    Information Maximizing Generative Adversarial Network (infoGAN) can be understood as a minimax problem involving two networks: discriminators and generators with mutual information functions. The infoGAN incorporates various components, including latent variables, mutual information, and objective function. This research demonstrates that the two objective functions in infoGAN become equivalent as the discriminator and generator sample size approaches infinity. This equivalence is established by considering the disparity between the empirical and population versions of the objective function. The bound on this difference is determined by the Rademacher complexity of the discriminator and generator function class. Furthermore, the utilization of a two-layer network for both the discriminator and generator, featuring Lipschitz and non-decreasing activation functions, validates this equality
    
[^3]: 为什么在对抗训练中会同时出现干净泛化和强健过拟合现象？

    Why Clean Generalization and Robust Overfitting Both Happen in Adversarial Training. (arXiv:2306.01271v1 [cs.LG])

    [http://arxiv.org/abs/2306.01271](http://arxiv.org/abs/2306.01271)

    对抗训练是训练深度神经网络抗击对抗扰动的标准方法, 其学习机制导致干净泛化和强健过拟合现象同时发生。

    

    对抗训练是训练深度神经网络抗击对抗扰动的标准方法。与在标准深度学习环境中出现惊人的干净泛化能力类似，通过对抗训练训练的神经网络也能很好地泛化到未见过的干净数据。然而，与干净泛化不同的是，尽管对抗训练能够实现低鲁棒训练误差，仍存在显著的鲁棒泛化距离，这促使我们探索在学习过程中导致干净泛化和强健过拟合现象同时发生的机制。本文提供了对抗训练中这种现象的理论理解。首先，我们提出了对抗训练的理论框架，分析了特征学习过程，解释了对抗训练如何导致网络学习者进入到干净泛化和强健过拟合状态。具体来说，我们证明了，通过迫使学习器成为强预测网络，对抗训练将导致干净泛化和鲁棒过拟合现象同时发生。

    Adversarial training is a standard method to train deep neural networks to be robust to adversarial perturbation. Similar to surprising $\textit{clean generalization}$ ability in the standard deep learning setting, neural networks trained by adversarial training also generalize well for $\textit{unseen clean data}$. However, in constrast with clean generalization, while adversarial training method is able to achieve low $\textit{robust training error}$, there still exists a significant $\textit{robust generalization gap}$, which promotes us exploring what mechanism leads to both $\textit{clean generalization and robust overfitting (CGRO)}$ during learning process. In this paper, we provide a theoretical understanding of this CGRO phenomenon in adversarial training. First, we propose a theoretical framework of adversarial training, where we analyze $\textit{feature learning process}$ to explain how adversarial training leads network learner to CGRO regime. Specifically, we prove that, u
    
[^4]: 评估-优化方法与集成评估优化法：基于随机优势的观点

    Estimate-Then-Optimize Versus Integrated-Estimation-Optimization: A Stochastic Dominance Perspective. (arXiv:2304.06833v1 [stat.ML])

    [http://arxiv.org/abs/2304.06833](http://arxiv.org/abs/2304.06833)

    本文提出，当模型类足够丰富以涵盖真实情况时，非线性问题的“先估计再优化”方法优于集成方法，包括优化间隙的渐进优势的均值，所有其他时刻和整个渐进分布。

    

    在数据驱动的随机优化中，除了需要优化任务，还需要从数据中估计潜在分布的模型参数。最近的文献表明，通过选择导致最佳经验目标性能的模型参数，可以集成估计和优化过程。当模型被错误地指定时，这种集成方法可以很容易地显示出优于简单的“先估计再优化”的方法。本文认为，在模型类足够丰富以涵盖真实情况的情况下，对于非线性问题，两种方法之间的性能排序在强烈的意义下被颠倒。在受限条件和当上下文特征可用时，类似的结果也成立。

    In data-driven stochastic optimization, model parameters of the underlying distribution need to be estimated from data in addition to the optimization task. Recent literature suggests the integration of the estimation and optimization processes, by selecting model parameters that lead to the best empirical objective performance. Such an integrated approach can be readily shown to outperform simple ``estimate then optimize" when the model is misspecified. In this paper, we argue that when the model class is rich enough to cover the ground truth, the performance ordering between the two approaches is reversed for nonlinear problems in a strong sense. Simple ``estimate then optimize" outperforms the integrated approach in terms of stochastic dominance of the asymptotic optimality gap, i,e, the mean, all other moments, and the entire asymptotic distribution of the optimality gap is always better. Analogous results also hold under constrained settings and when contextual features are availa
    

