# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Scalable Learning of Item Response Theory Models](https://arxiv.org/abs/2403.00680) | 该论文提出了一种从大数据中学习项目反应理论模型中的潜在变量的方法，利用这些模型与逻辑回归之间的相似性来提高计算的效率和可伸缩性。 |
| [^2] | [Causal Scoring: A Framework for Effect Estimation, Effect Ordering, and Effect Classification](https://arxiv.org/abs/2206.12532) | 本文引入了因果评分作为一种新型方法，支持决策制定，提供洞察力，并可用于效应估计、效应排序和效应分类。 |
| [^3] | [Inverting estimating equations for causal inference on quantiles.](http://arxiv.org/abs/2401.00987) | 本文提出了一种基于反转估计方程的通用方法，将从估计潜在结果均值的因果推断解决方案推广到其分位数，同时给出了潜在结果均值和分位数的有效影响函数的一般构造和联系。 |
| [^4] | [Bottleneck Structure in Learned Features: Low-Dimension vs Regularity Tradeoff.](http://arxiv.org/abs/2305.19008) | 本研究揭示了深度学习神经网路学习输入低维度表示和最小化特征映射中的复杂性/不规则性之间的权衡，控制了规律性，并利用理论工具证明了瓶颈结构的存在。 |
| [^5] | [Wasserstein Gaussianization and Efficient Variational Bayes for Robust Bayesian Synthetic Likelihood.](http://arxiv.org/abs/2305.14746) | 本文提出了一种称为Wasserstein高斯化转换的转换方法，用于针对贝叶斯合成似然(BSL)中汇总统计量的分布问题。将Wasserstein高斯化转换与鲁棒BSL和高效的变分贝叶斯方法相结合，开发了一种高效可靠的适用于无似然问题的近似贝叶斯推断方法。 |
| [^6] | [On Model Compression for Neural Networks: Framework, Algorithm, and Convergence Guarantee.](http://arxiv.org/abs/2303.06815) | 本文提出了一个框架和算法，从非凸优化的角度来进行神经网络模型压缩。算法解决了梯度消失/爆炸问题，并保证了收敛性。 |
| [^7] | [Depth Degeneracy in Neural Networks: Vanishing Angles in Fully Connected ReLU Networks on Initialization.](http://arxiv.org/abs/2302.09712) | 本文研究了深度神经网络中的深度退化现象，在全连接ReLU网络初始化时，两个输入之间的角度会趋近于0。通过使用组合展开，得到了其趋向于0的速度的精确公式，并验证了这些结果。 |

# 详细

[^1]: 可扩展的项目反应理论模型学习

    Scalable Learning of Item Response Theory Models

    [https://arxiv.org/abs/2403.00680](https://arxiv.org/abs/2403.00680)

    该论文提出了一种从大数据中学习项目反应理论模型中的潜在变量的方法，利用这些模型与逻辑回归之间的相似性来提高计算的效率和可伸缩性。

    

    项目反应理论（IRT）模型旨在评估 $n$ 名考生的潜在能力以及 $m$ 个测验项目的隐含难度特征，这些项目是从表明其对应答案质量的分类数据中得出的。传统的心理测量评估基于相对较少的考生和项目，例如一个由 $200$ 名学生解决包含 $10$ 道题目的考试的班级。而近年来的全球大规模评估，如PISA，或互联网研究，可能导致参与者数量显著增加。此外，在机器学习领域，算法扮演考生角色，数据分析问题扮演项目角色，$n$ 和 $m$ 都可能变得非常大，挑战计算的效率和可伸缩性。为了从大数据中学习IRT模型中的潜在变量，我们利用这些模型与逻辑回归之间的相似性，后者可以使用s准确地近似。

    arXiv:2403.00680v1 Announce Type: new  Abstract: Item Response Theory (IRT) models aim to assess latent abilities of $n$ examinees along with latent difficulty characteristics of $m$ test items from categorical data that indicates the quality of their corresponding answers. Classical psychometric assessments are based on a relatively small number of examinees and items, say a class of $200$ students solving an exam comprising $10$ problems. More recent global large scale assessments such as PISA, or internet studies, may lead to significantly increased numbers of participants. Additionally, in the context of Machine Learning where algorithms take the role of examinees and data analysis problems take the role of items, both $n$ and $m$ may become very large, challenging the efficiency and scalability of computations. To learn the latent variables in IRT models from large data, we leverage the similarity of these models to logistic regression, which can be approximated accurately using s
    
[^2]: 因果评分：效应估计、效应排序和效应分类的框架

    Causal Scoring: A Framework for Effect Estimation, Effect Ordering, and Effect Classification

    [https://arxiv.org/abs/2206.12532](https://arxiv.org/abs/2206.12532)

    本文引入了因果评分作为一种新型方法，支持决策制定，提供洞察力，并可用于效应估计、效应排序和效应分类。

    

    本文将因果评分引入到决策制定的背景中作为一种新颖方法，涉及估计支持决策制定的得分，从而提供因果效应的洞察力。我们提出了这些评分的三种有价值的因果解释：效应估计（EE）、效应排序（EO）和效应分类（EC）。在EE解释中，因果评分代表了效应本身。EO解释暗示评分可以作为效应大小的代理，可以根据其因果效应对个体进行排序。EC解释通过预定义的阈值，使个体分为高效应和低效应类别。我们通过两个关键结果展示了这些替代因果解释（EO和EC）的价值。

    arXiv:2206.12532v4 Announce Type: replace-cross  Abstract: This paper introduces causal scoring as a novel approach to frame causal estimation in the context of decision making. Causal scoring entails the estimation of scores that support decision making by providing insights into causal effects. We present three valuable causal interpretations of these scores: effect estimation (EE), effect ordering (EO), and effect classification (EC). In the EE interpretation, the causal score represents the effect itself. The EO interpretation implies that the score can serve as a proxy for the magnitude of the effect, enabling the sorting of individuals based on their causal effects. The EC interpretation enables the classification of individuals into high- and low-effect categories using a predefined threshold. We demonstrate the value of these alternative causal interpretations (EO and EC) through two key results. First, we show that aligning the statistical modeling with the desired causal inte
    
[^3]: 反转估计方程对潜在结果分位数的因果推断

    Inverting estimating equations for causal inference on quantiles. (arXiv:2401.00987v1 [stat.ME])

    [http://arxiv.org/abs/2401.00987](http://arxiv.org/abs/2401.00987)

    本文提出了一种基于反转估计方程的通用方法，将从估计潜在结果均值的因果推断解决方案推广到其分位数，同时给出了潜在结果均值和分位数的有效影响函数的一般构造和联系。

    

    因果推断文献经常关注潜在结果的均值估计，而潜在结果的分位数可能包含重要的额外信息。我们提出了一种基于反转估计方程的通用方法，将从估计潜在结果均值的广泛类别的因果推断解决方案推广到其分位数。我们假设存在一个可用来确定基于阈值变换的潜在结果均值的确定矩函数，并在此基础上提出了潜在结果分位数的估计方程的便利构造。此外，我们还给出了潜在结果均值和分位数的有效影响函数的一般构造，并确定了它们之间的联系。我们通过有效影响函数推导出分位数目标的估计器，并在使用参数模型或数据自适应机器学习方法时开发其渐近性质。

    The causal inference literature frequently focuses on estimating the mean of the potential outcome, whereas the quantiles of the potential outcome may carry important additional information. We propose a universal approach, based on the inverse estimating equations, to generalize a wide class of causal inference solutions from estimating the mean of the potential outcome to its quantiles. We assume that an identifying moment function is available to identify the mean of the threshold-transformed potential outcome, based on which a convenient construction of the estimating equation of quantiles of potential outcome is proposed. In addition, we also give a general construction of the efficient influence functions of the mean and quantiles of potential outcomes, and identify their connection. We motivate estimators for the quantile estimands with the efficient influence function, and develop their asymptotic properties when either parametric models or data-adaptive machine learners are us
    
[^4]: 学习特征中的瓶颈结构：低维度与规律性的权衡

    Bottleneck Structure in Learned Features: Low-Dimension vs Regularity Tradeoff. (arXiv:2305.19008v1 [cs.LG])

    [http://arxiv.org/abs/2305.19008](http://arxiv.org/abs/2305.19008)

    本研究揭示了深度学习神经网路学习输入低维度表示和最小化特征映射中的复杂性/不规则性之间的权衡，控制了规律性，并利用理论工具证明了瓶颈结构的存在。

    

    先前研究表明，具有大深度$L$和$L_{2}$正则化的DNN偏向于学习输入的低维表示，可以解释为最小化学习函数$f$的秩$R^{(0)}(f)$的概念，其被推测为瓶颈秩。我们计算了这个结果的有限深度修正，揭示了一个度量$R^{(1)}$的规律性，它控制了雅可比矩阵$\left|Jf(x)\right|_{+}$的伪行列式并在组合和加法下是次可加的。这使得网络可以在学习低维表示和最小化特征映射中的复杂性/不规则性之间保持平衡，从而学习“正确”的内部尺寸。我们还展示了大学习速率如何控制学习函数的规律性。最后，我们使用这些理论工具证明了瓶颈结构在$L\to\infty$时在学习特征中的猜想：对于大深度，几乎所有的隐藏表示都集中在...

    Previous work has shown that DNNs with large depth $L$ and $L_{2}$-regularization are biased towards learning low-dimensional representations of the inputs, which can be interpreted as minimizing a notion of rank $R^{(0)}(f)$ of the learned function $f$, conjectured to be the Bottleneck rank. We compute finite depth corrections to this result, revealing a measure $R^{(1)}$ of regularity which bounds the pseudo-determinant of the Jacobian $\left|Jf(x)\right|_{+}$ and is subadditive under composition and addition. This formalizes a balance between learning low-dimensional representations and minimizing complexity/irregularity in the feature maps, allowing the network to learn the `right' inner dimension. We also show how large learning rates also control the regularity of the learned function. Finally, we use these theoretical tools to prove the conjectured bottleneck structure in the learned features as $L\to\infty$: for large depths, almost all hidden representations concentrates aroun
    
[^5]: Wasserstein高斯转换和鲁棒贝叶斯合成似然的高效变分贝叶斯方法

    Wasserstein Gaussianization and Efficient Variational Bayes for Robust Bayesian Synthetic Likelihood. (arXiv:2305.14746v1 [stat.CO])

    [http://arxiv.org/abs/2305.14746](http://arxiv.org/abs/2305.14746)

    本文提出了一种称为Wasserstein高斯化转换的转换方法，用于针对贝叶斯合成似然(BSL)中汇总统计量的分布问题。将Wasserstein高斯化转换与鲁棒BSL和高效的变分贝叶斯方法相结合，开发了一种高效可靠的适用于无似然问题的近似贝叶斯推断方法。

    

    贝叶斯合成似然(BSL)方法是一种广泛使用的无似然贝叶斯推断工具。该方法假定某些汇总统计量服从正态分布，在许多应用中可能是不正确的。我们提出了一种称为Wasserstein高斯化转换的转换方法，使用Wasserstein梯度流将汇总统计量的分布近似转换为正态分布。BSL隐含地要求模拟汇总统计量在工作模型下与观察到的汇总统计量兼容。近期已开发了一种鲁棒的BSL变体来实现这一点。我们将Wasserstein高斯化转换与鲁棒BSL以及高效的变分贝叶斯过程结合起来，开发了一种高效可靠的适用于无似然问题的近似贝叶斯推断方法。

    The Bayesian Synthetic Likelihood (BSL) method is a widely-used tool for likelihood-free Bayesian inference. This method assumes that some summary statistics are normally distributed, which can be incorrect in many applications. We propose a transformation, called the Wasserstein Gaussianization transformation, that uses a Wasserstein gradient flow to approximately transform the distribution of the summary statistics into a Gaussian distribution. BSL also implicitly requires compatibility between simulated summary statistics under the working model and the observed summary statistics. A robust BSL variant which achieves this has been developed in the recent literature. We combine the Wasserstein Gaussianization transformation with robust BSL, and an efficient Variational Bayes procedure for posterior approximation, to develop a highly efficient and reliable approximate Bayesian inference method for likelihood-free problems.
    
[^6]: 关于神经网络模型压缩的框架、算法和收敛保证

    On Model Compression for Neural Networks: Framework, Algorithm, and Convergence Guarantee. (arXiv:2303.06815v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.06815](http://arxiv.org/abs/2303.06815)

    本文提出了一个框架和算法，从非凸优化的角度来进行神经网络模型压缩。算法解决了梯度消失/爆炸问题，并保证了收敛性。

    

    模型压缩对于部署神经网络（NN）至关重要，特别是在许多应用程序中计算设备的内存和存储有限的情况下。本文关注两种神经网络模型压缩技术：低秩逼近和权重裁剪，这些技术目前非常流行。然而，使用低秩逼近和权重裁剪训练神经网络总是会遭受显著的准确性损失和收敛问题。本文提出了一个全面的框架，从非凸优化的新视角设计了适当的目标函数来进行模型压缩。然后，我们引入了一种块坐标下降（BCD）算法NN-BCD来解决非凸优化问题。我们算法的一个优点是可以获得具有闭式形式的高效迭代方案，从而避免了梯度消失/爆炸的问题。此外，我们的算法利用了Kurdyka-{\L}ojasiewicz (K{\L})性质，保证了算法的收敛性。

    Model compression is a crucial part of deploying neural networks (NNs), especially when the memory and storage of computing devices are limited in many applications. This paper focuses on two model compression techniques: low-rank approximation and weight pruning in neural networks, which are very popular nowadays. However, training NN with low-rank approximation and weight pruning always suffers significant accuracy loss and convergence issues. In this paper, a holistic framework is proposed for model compression from a novel perspective of nonconvex optimization by designing an appropriate objective function. Then, we introduce NN-BCD, a block coordinate descent (BCD) algorithm to solve the nonconvex optimization. One advantage of our algorithm is that an efficient iteration scheme can be derived with closed-form, which is gradient-free. Therefore, our algorithm will not suffer from vanishing/exploding gradient problems. Furthermore, with the Kurdyka-{\L}ojasiewicz (K{\L}) property o
    
[^7]: 神经网络中的深度退化：全连接ReLU网络初始化时，消失角度的现象 (arXiv:2302.09712v2 [stat.ML] 更新版)

    Depth Degeneracy in Neural Networks: Vanishing Angles in Fully Connected ReLU Networks on Initialization. (arXiv:2302.09712v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.09712](http://arxiv.org/abs/2302.09712)

    本文研究了深度神经网络中的深度退化现象，在全连接ReLU网络初始化时，两个输入之间的角度会趋近于0。通过使用组合展开，得到了其趋向于0的速度的精确公式，并验证了这些结果。

    

    尽管深度神经网络在各种任务上表现出色，但许多其性质仍未被理论上理解，其中一个谜团是深度退化现象：网络层数越深，初始化时网络越接近于常数函数。在本文中，我们研究了ReLU神经网络两个输入之间随着层数变化的角度演变情况。通过使用组合展开，我们找到了它随深度增加趋向于0的速度的精确公式，这些公式捕捉了微观波动。我们用Monte Carlo实验验证了我们的理论结果，并证明了结果准确地近似了有限网络的行为。这些公式以通过ReLU函数的相关高斯变量的混合矩形式给出。我们还发现了一个令人惊讶的组合现象。

    Despite remarkable performance on a variety of tasks, many properties of deep neural networks are not yet theoretically understood. One such mystery is the depth degeneracy phenomenon: the deeper you make your network, the closer your network is to a constant function on initialization. In this paper, we examine the evolution of the angle between two inputs to a ReLU neural network as a function of the number of layers. By using combinatorial expansions, we find precise formulas for how fast this angle goes to zero as depth increases. These formulas capture microscopic fluctuations that are not visible in the popular framework of infinite width limits, and leads to qualitatively different predictions. We validate our theoretical results with Monte Carlo experiments and show that our results accurately approximate finite network behaviour. The formulas are given in terms of the mixed moments of correlated Gaussians passed through the ReLU function. We also find a surprising combinatoria
    

