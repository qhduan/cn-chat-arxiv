# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Data-Efficient Operator Learning via Unsupervised Pretraining and In-Context Learning](https://arxiv.org/abs/2402.15734) | 该论文提出了一种通过无监督预训练和上下文学习方法实现PDE运算符学习的高效方式，以提高数据效率并改善模型的外域性能。 |
| [^2] | [Online Control of Linear Systems with Unbounded and Degenerate Noise](https://arxiv.org/abs/2402.10252) | 这项研究揭示了在线控制问题中，对于凸成本，可以实现 $ \widetilde{O}(\sqrt{T}) $ 的遗憾界，甚至在存在无界噪声的情况下；同时，在成本具有强凸性时，可以在不需要噪声协方差是非退化的情况下建立 $ O({\rm poly} (\log T)) $ 的遗憾界。 |
| [^3] | [Generalization Bounds for Heavy-Tailed SDEs through the Fractional Fokker-Planck Equation](https://arxiv.org/abs/2402.07723) | 本论文通过分数阻尼库仑方程证明了重尾SDE的高概率泛化界限，并且相对于参数维度，界限的依赖性要好于p。 |
| [^4] | [On Calibration and Conformal Prediction of Deep Classifiers](https://arxiv.org/abs/2402.05806) | 本文研究了温度缩放对符合预测方法的影响，通过实证研究发现，校准对自适应C方法产生了有害的影响。 |
| [^5] | [Distribution-Free Inference for the Regression Function of Binary Classification.](http://arxiv.org/abs/2308.01835) | 本文提出了一种分布无关的方法来推断二元分类问题中的回归函数，通过构建置信区间来解决该问题，相关算法经过验证具有可靠性。 |
| [^6] | [Kernel $\epsilon$-Greedy for Contextual Bandits.](http://arxiv.org/abs/2306.17329) | 本文提出了基于核的$\epsilon$-贪心策略应用于情境脉冲中的方法，通过在线加权核岭回归估计器实现对奖励函数的估计，并证明了其一致性和依赖于RKHS维度的次线性后悔率，在有限维RKHS的边际条件下实现了最优后悔率。 |
| [^7] | [A Meta-Learning Method for Estimation of Causal Excursion Effects to Assess Time-Varying Moderation.](http://arxiv.org/abs/2306.16297) | 这项研究介绍了一种元学习方法，用于评估因果偏离效应，以评估干预效果随时间的变化或通过个体特征、环境或过去的反应来调节。目前的数据分析方法需要预先指定观察到的高维历史的特征来构建重要干扰参数的工作模型，而机器学习算法可以自动进行特征构建，但其朴素应用存在问题。 |
| [^8] | [Efficient distributed representations beyond negative sampling.](http://arxiv.org/abs/2303.17475) | 本文介绍了一种高效的分布式表示（嵌入）学习方法，通过线性时间估计softmax归一化常数来实现学习过程，该方法优于负采样方法并在多项测试中验证了其有效性。 |

# 详细

[^1]: 通过无监督预训练和上下文学习实现高效的运算符学习

    Data-Efficient Operator Learning via Unsupervised Pretraining and In-Context Learning

    [https://arxiv.org/abs/2402.15734](https://arxiv.org/abs/2402.15734)

    该论文提出了一种通过无监督预训练和上下文学习方法实现PDE运算符学习的高效方式，以提高数据效率并改善模型的外域性能。

    

    近年来，人们见证了将机器学习方法与物理领域特定洞察力相结合，以解决基于偏微分方程（PDEs）的科学问题的潜力。然而，由于数据密集，这些方法仍然需要大量PDE数据。 这重新引入了对昂贵的数值PDE解决方案的需求，部分削弱了避免这些昂贵模拟的原始目标。 在这项工作中，为了寻求数据效率，我们设计了用于PDE运算符学习的无监督预训练和上下文学习方法。 为了减少对带有模拟解的训练数据的需求，我们使用基于重构的代理任务在未标记的PDE数据上预训练神经运算符。 为了提高超出分布性能，我们进一步帮助神经运算符灵活地利用上下文学习方法，而无需额外的训练成本或设计。 在各种PD上进行了大量实证评估

    arXiv:2402.15734v1 Announce Type: new  Abstract: Recent years have witnessed the promise of coupling machine learning methods and physical domain-specific insight for solving scientific problems based on partial differential equations (PDEs). However, being data-intensive, these methods still require a large amount of PDE data. This reintroduces the need for expensive numerical PDE solutions, partially undermining the original goal of avoiding these expensive simulations. In this work, seeking data efficiency, we design unsupervised pretraining and in-context learning methods for PDE operator learning. To reduce the need for training data with simulated solutions, we pretrain neural operators on unlabeled PDE data using reconstruction-based proxy tasks. To improve out-of-distribution performance, we further assist neural operators in flexibly leveraging in-context learning methods, without incurring extra training costs or designs. Extensive empirical evaluations on a diverse set of PD
    
[^2]: 具有无界和退化噪声的线性系统在线控制

    Online Control of Linear Systems with Unbounded and Degenerate Noise

    [https://arxiv.org/abs/2402.10252](https://arxiv.org/abs/2402.10252)

    这项研究揭示了在线控制问题中，对于凸成本，可以实现 $ \widetilde{O}(\sqrt{T}) $ 的遗憾界，甚至在存在无界噪声的情况下；同时，在成本具有强凸性时，可以在不需要噪声协方差是非退化的情况下建立 $ O({\rm poly} (\log T)) $ 的遗憾界。

    

    本文研究了在可能存在无界和退化噪声的情况下控制线性系统的问题，其中成本函数未知，被称为在线控制问题。与现有的仅假设噪声有界性的研究不同，我们揭示了对于凸成本，即使在存在无界噪声的情况下也可以实现 $ \widetilde{O}(\sqrt{T}) $ 的遗憾界，其中 $ T $ 表示时间跨度。此外，当成本具有强凸性时，我们建立了一个 $ O({\rm poly} (\log T)) $ 的遗憾界，而不需要噪声协方差是非退化的假设，这在文献中是必需的。消除噪声秩的关键是与噪声协方差相关联的系统转化。这同时实现了在线控制算法的参数减少。

    arXiv:2402.10252v1 Announce Type: cross  Abstract: This paper investigates the problem of controlling a linear system under possibly unbounded and degenerate noise with unknown cost functions, known as an online control problem. In contrast to the existing work, which assumes the boundedness of noise, we reveal that for convex costs, an $ \widetilde{O}(\sqrt{T}) $ regret bound can be achieved even for unbounded noise, where $ T $ denotes the time horizon. Moreover, when the costs are strongly convex, we establish an $ O({\rm poly} (\log T)) $ regret bound without the assumption that noise covariance is non-degenerate, which has been required in the literature. The key ingredient in removing the rank assumption on noise is a system transformation associated with the noise covariance. This simultaneously enables the parameter reduction of an online control algorithm.
    
[^3]: 通过分数阻尼库仑方程证明重尾SDEs的泛化界限

    Generalization Bounds for Heavy-Tailed SDEs through the Fractional Fokker-Planck Equation

    [https://arxiv.org/abs/2402.07723](https://arxiv.org/abs/2402.07723)

    本论文通过分数阻尼库仑方程证明了重尾SDE的高概率泛化界限，并且相对于参数维度，界限的依赖性要好于p。

    

    过去几年来，理解重尾随机优化算法的泛化性能引起了越来越多的关注。在利用重尾随机微分方程作为代理来阐明随机优化器的有趣方面时，先前的工作要么提供预期的泛化界限，要么引入了不可计算的信息论术语。为了解决这些缺点，在本文中，我们证明了重尾SDE的高概率泛化界限，这些界限不含任何非平凡的信息论术语。为了实现这个目标，我们基于估计与所谓的分数阻尼库仑方程相关联的熵流，开发了新的证明技术（这是一种控制相应重尾SDE分布演化的偏微分方程）。除了获得高概率界限之外，我们还展示了我们的界限相对于参数维度的依赖性要好于p。

    Understanding the generalization properties of heavy-tailed stochastic optimization algorithms has attracted increasing attention over the past years. While illuminating interesting aspects of stochastic optimizers by using heavy-tailed stochastic differential equations as proxies, prior works either provided expected generalization bounds, or introduced non-computable information theoretic terms. Addressing these drawbacks, in this work, we prove high-probability generalization bounds for heavy-tailed SDEs which do not contain any nontrivial information theoretic terms. To achieve this goal, we develop new proof techniques based on estimating the entropy flows associated with the so-called fractional Fokker-Planck equation (a partial differential equation that governs the evolution of the distribution of the corresponding heavy-tailed SDE). In addition to obtaining high-probability bounds, we show that our bounds have a better dependence on the dimension of parameters as compared to p
    
[^4]: 关于深度分类器的校准和符合预测研究

    On Calibration and Conformal Prediction of Deep Classifiers

    [https://arxiv.org/abs/2402.05806](https://arxiv.org/abs/2402.05806)

    本文研究了温度缩放对符合预测方法的影响，通过实证研究发现，校准对自适应C方法产生了有害的影响。

    

    在许多分类应用中，深度神经网络（DNN）基于分类器的预测需要伴随一些置信度指示。针对这个目标，有两种流行的后处理方法：1）校准：修改分类器的softmax值，使其最大值（与预测相关）更好地估计正确概率；和2）符合预测（CP）：设计一个基于softmax值的分数，从中产生一组预测，具有理论上保证正确类别边际覆盖的特性。尽管在实践中两种指示都可能是需要的，但到目前为止它们之间的相互作用尚未得到研究。为了填补这一空白，在本文中，我们研究了温度缩放，这是最常见的校准技术，对重要的CP方法的影响。我们首先进行了一项广泛的实证研究，其中显示了一些重要的洞察，其中包括令人惊讶的发现，即校准对流行的自适应C方法产生了有害的影响。

    In many classification applications, the prediction of a deep neural network (DNN) based classifier needs to be accompanied with some confidence indication. Two popular post-processing approaches for that aim are: 1) calibration: modifying the classifier's softmax values such that their maximum (associated with the prediction) better estimates the correctness probability; and 2) conformal prediction (CP): devising a score (based on the softmax values) from which a set of predictions with theoretically guaranteed marginal coverage of the correct class is produced. While in practice both types of indications can be desired, so far the interplay between them has not been investigated. Toward filling this gap, in this paper we study the effect of temperature scaling, arguably the most common calibration technique, on prominent CP methods. We start with an extensive empirical study that among other insights shows that, surprisingly, calibration has a detrimental effect on popular adaptive C
    
[^5]: 分布无关推断二元分类的回归函数

    Distribution-Free Inference for the Regression Function of Binary Classification. (arXiv:2308.01835v1 [stat.ML])

    [http://arxiv.org/abs/2308.01835](http://arxiv.org/abs/2308.01835)

    本文提出了一种分布无关的方法来推断二元分类问题中的回归函数，通过构建置信区间来解决该问题，相关算法经过验证具有可靠性。

    

    二元分类的一个关键对象是回归函数，即给定输入的类别标签的条件期望。通过回归函数，不仅可以定义贝叶斯最优分类器，还可以编码对应的错误分类概率。本文提出了一种重采样框架，用于构建精确、分布无关且非渐近保证的真实回归函数的置信区间，根据用户选择的置信水平。然后，提出了特定的算法来演示该框架。证明了构建的置信区间是强一致的，也就是说，任何错误的模型最终被排除的概率为1。排除的程度也通过可能近似正确类型的界限进行了量化。最后，通过数值实验验证了算法，并将方法与近似渐近置信椭圆进行了比较。

    One of the key objects of binary classification is the regression function, i.e., the conditional expectation of the class labels given the inputs. With the regression function not only a Bayes optimal classifier can be defined, but it also encodes the corresponding misclassification probabilities. The paper presents a resampling framework to construct exact, distribution-free and non-asymptotically guaranteed confidence regions for the true regression function for any user-chosen confidence level. Then, specific algorithms are suggested to demonstrate the framework. It is proved that the constructed confidence regions are strongly consistent, that is, any false model is excluded in the long run with probability one. The exclusion is quantified with probably approximately correct type bounds, as well. Finally, the algorithms are validated via numerical experiments, and the methods are compared to approximate asymptotic confidence ellipsoids.
    
[^6]: 基于核的$\epsilon$-贪心策略在情境脉冲中的应用

    Kernel $\epsilon$-Greedy for Contextual Bandits. (arXiv:2306.17329v1 [stat.ML])

    [http://arxiv.org/abs/2306.17329](http://arxiv.org/abs/2306.17329)

    本文提出了基于核的$\epsilon$-贪心策略应用于情境脉冲中的方法，通过在线加权核岭回归估计器实现对奖励函数的估计，并证明了其一致性和依赖于RKHS维度的次线性后悔率，在有限维RKHS的边际条件下实现了最优后悔率。

    

    我们考虑了情境脉冲中的基于核的$\epsilon$-贪心策略。更具体地说，在有限数量的臂的情况下，我们认为平均奖励函数位于再生核希尔伯特空间（RKHS）中。我们提出了一种用于奖励函数的在线加权核岭回归估计器。在对探索概率序列$\{\epsilon_t\}_t$和正则化参数$\{\lambda_t\}_t$的一些条件下，我们证明了所提出的估计器的一致性。我们还证明，对于任何核和相应的RKHS的选择，我们可以实现依赖于RKHS内在维度的次线性后悔率。此外，在有限维RKHS的边际条件下，我们实现了$\sqrt{T}$的最优后悔率。

    We consider a kernelized version of the $\epsilon$-greedy strategy for contextual bandits. More precisely, in a setting with finitely many arms, we consider that the mean reward functions lie in a reproducing kernel Hilbert space (RKHS). We propose an online weighted kernel ridge regression estimator for the reward functions. Under some conditions on the exploration probability sequence, $\{\epsilon_t\}_t$, and choice of the regularization parameter, $\{\lambda_t\}_t$, we show that the proposed estimator is consistent. We also show that for any choice of kernel and the corresponding RKHS, we achieve a sub-linear regret rate depending on the intrinsic dimensionality of the RKHS. Furthermore, we achieve the optimal regret rate of $\sqrt{T}$ under a margin condition for finite-dimensional RKHS.
    
[^7]: 一种用于评估时变调节因素的因果偏离效应估计的元学习方法

    A Meta-Learning Method for Estimation of Causal Excursion Effects to Assess Time-Varying Moderation. (arXiv:2306.16297v1 [stat.ME])

    [http://arxiv.org/abs/2306.16297](http://arxiv.org/abs/2306.16297)

    这项研究介绍了一种元学习方法，用于评估因果偏离效应，以评估干预效果随时间的变化或通过个体特征、环境或过去的反应来调节。目前的数据分析方法需要预先指定观察到的高维历史的特征来构建重要干扰参数的工作模型，而机器学习算法可以自动进行特征构建，但其朴素应用存在问题。

    

    可穿戴技术和智能手机提供的数字化健康干预的双重革命显著增加了移动健康（mHealth）干预在各个健康科学领域的可及性和采纳率。顺序随机实验称为微随机试验（MRTs）已经越来越受欢迎，用于实证评估这些mHealth干预组成部分的有效性。MRTs产生了一类新的因果估计量，称为“因果偏离效应”，使健康科学家能够评估干预效果随时间的变化或通过个体特征、环境或过去的反应来调节。然而，目前用于估计因果偏离效应的数据分析方法需要预先指定观察到的高维历史的特征来构建重要干扰参数的工作模型。虽然机器学习算法在自动特征构建方面具有优势，但其朴素应用导致了问题。

    Twin revolutions in wearable technologies and smartphone-delivered digital health interventions have significantly expanded the accessibility and uptake of mobile health (mHealth) interventions across various health science domains. Sequentially randomized experiments called micro-randomized trials (MRTs) have grown in popularity to empirically evaluate the effectiveness of these mHealth intervention components. MRTs have given rise to a new class of causal estimands known as "causal excursion effects", which enable health scientists to assess how intervention effectiveness changes over time or is moderated by individual characteristics, context, or responses in the past. However, current data analysis methods for estimating causal excursion effects require pre-specified features of the observed high-dimensional history to construct a working model of an important nuisance parameter. While machine learning algorithms are ideal for automatic feature construction, their naive application
    
[^8]: 超越负采样的高效分布式表示方法

    Efficient distributed representations beyond negative sampling. (arXiv:2303.17475v1 [cs.LG])

    [http://arxiv.org/abs/2303.17475](http://arxiv.org/abs/2303.17475)

    本文介绍了一种高效的分布式表示（嵌入）学习方法，通过线性时间估计softmax归一化常数来实现学习过程，该方法优于负采样方法并在多项测试中验证了其有效性。

    

    本文介绍了一种高效的学习分布式表示（也称为嵌入）的方法。该方法通过最小化一个类似于Word2Vec算法中引入并在多个工作中采用的目标函数来实现。优化计算的瓶颈是softmax归一化常数的计算，这需要与样本大小呈二次比例的操作数。这种复杂度不适用于大型数据集，所以负采样是一个常见的解决方法，可以在与样本大小线性相关的时间内获得分布式表示。然而，负采样会改变损失函数，因此解决的是与最初提出的不同的优化问题。我们的贡献在于展示如何通过线性时间估计softmax归一化常数，从而设计了一种有效的优化策略来学习分布式表示。我们使用不同的数据集进行测试，并展示了我们的方法在嵌入质量和训练时间方面优于负采样。

    This article describes an efficient method to learn distributed representations, also known as embeddings. This is accomplished minimizing an objective function similar to the one introduced in the Word2Vec algorithm and later adopted in several works. The optimization computational bottleneck is the calculation of the softmax normalization constants for which a number of operations scaling quadratically with the sample size is required. This complexity is unsuited for large datasets and negative sampling is a popular workaround, allowing one to obtain distributed representations in linear time with respect to the sample size. Negative sampling consists, however, in a change of the loss function and hence solves a different optimization problem from the one originally proposed. Our contribution is to show that the sotfmax normalization constants can be estimated in linear time, allowing us to design an efficient optimization strategy to learn distributed representations. We test our ap
    

