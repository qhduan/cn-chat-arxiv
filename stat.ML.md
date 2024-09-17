# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Conformal Off-Policy Prediction for Multi-Agent Systems](https://arxiv.org/abs/2403.16871) | 这项工作介绍了MA-COPP，这是第一个解决涉及多智能体系统的离策略预测问题的一致预测方法。 |
| [^2] | [Assumption-lean and Data-adaptive Post-Prediction Inference](https://arxiv.org/abs/2311.14220) | 这项工作介绍了一种假设简化和数据自适应的后预测推断（POP-Inf）过程，可以有效且有力地基于机器学习预测结果进行统计推断。 |
| [^3] | [A Survey on Statistical Theory of Deep Learning: Approximation, Training Dynamics, and Generative Models.](http://arxiv.org/abs/2401.07187) | 该论文综述了深度学习的统计理论，包括近似方法、训练动态和生成模型。在非参数框架中，结果揭示了神经网络过度风险的快速收敛速率，以及如何通过梯度方法训练网络以找到良好的泛化解决方案。 |
| [^4] | [Generative neural networks for characteristic functions.](http://arxiv.org/abs/2401.04778) | 本论文研究了利用生成神经网络模拟特征函数的问题，并通过构建一个普适且无需假设的生成神经网络来解决。研究基于最大均值差异度量，并提出了有关逼近质量的有限样本保证。 |
| [^5] | [Energy based diffusion generator for efficient sampling of Boltzmann distributions.](http://arxiv.org/abs/2401.02080) | 介绍了一种称为基于能量的扩散生成器的新型采样器，用于从任意目标分布中生成样本，并通过扩散模型和广义哈密顿动力学提高采样性能。在各种复杂分布函数上的实证评估中表现出优越性。 |
| [^6] | [Learning Capacity: A Measure of the Effective Dimensionality of a Model.](http://arxiv.org/abs/2305.17332) | 学习能力是一种度量模型有效维度的方法，它可以帮助我们判断是否需要获取更多数据或者寻找新的体系结构以提高性能。 |
| [^7] | [Sarah Frank-Wolfe: Methods for Constrained Optimization with Best Rates and Practical Features.](http://arxiv.org/abs/2304.11737) | 本论文介绍了两种新的随机FW有限和最小化算法变体，适用于凸函数和非凸函数，且具有最佳收敛保证。同时两种方法不需要永久收集大批数据和全确定性梯度。 |
| [^8] | [High-dimensional and universally consistent k-sample tests.](http://arxiv.org/abs/1910.08883) | 本文证明了独立性测试实现了普遍一致的k样本检验，并且发现非参数独立性测试通常比多元方差分析(MANOVA)测试在高斯分布情况下表现更好。 |

# 详细

[^1]: 多智能体系统的一致离策略预测

    Conformal Off-Policy Prediction for Multi-Agent Systems

    [https://arxiv.org/abs/2403.16871](https://arxiv.org/abs/2403.16871)

    这项工作介绍了MA-COPP，这是第一个解决涉及多智能体系统的离策略预测问题的一致预测方法。

    

    离策略预测（OPP），即仅使用在一个正常（行为）策略下收集的数据来预测目标策略的结果，在数据驱动的安全关键系统分析中是一个重要问题，在这种系统中，部署新策略可能是不安全的。为了实现可信的离策略预测，最近关于一致离策略预测（COPP）的工作利用一致预测框架来在目标过程下推导带有概率保证的预测区域。现有的COPP方法可以考虑由策略切换引起的分布偏移，但仅限于单智能体系统和标量结果（例如，奖励）。在这项工作中，我们介绍了MA-COPP，这是第一个解决涉及多智能体系统的OPP问题的一致预测方法，在一个或多个“自我”智能体改变策略时为所有智能体轨迹推导联合预测区域。与单智能体场景不同，这种情况下

    arXiv:2403.16871v1 Announce Type: cross  Abstract: Off-Policy Prediction (OPP), i.e., predicting the outcomes of a target policy using only data collected under a nominal (behavioural) policy, is a paramount problem in data-driven analysis of safety-critical systems where the deployment of a new policy may be unsafe. To achieve dependable off-policy predictions, recent work on Conformal Off-Policy Prediction (COPP) leverage the conformal prediction framework to derive prediction regions with probabilistic guarantees under the target process. Existing COPP methods can account for the distribution shifts induced by policy switching, but are limited to single-agent systems and scalar outcomes (e.g., rewards). In this work, we introduce MA-COPP, the first conformal prediction method to solve OPP problems involving multi-agent systems, deriving joint prediction regions for all agents' trajectories when one or more "ego" agents change their policies. Unlike the single-agent scenario, this se
    
[^2]: 假设简化和数据自适应的后预测推断

    Assumption-lean and Data-adaptive Post-Prediction Inference

    [https://arxiv.org/abs/2311.14220](https://arxiv.org/abs/2311.14220)

    这项工作介绍了一种假设简化和数据自适应的后预测推断（POP-Inf）过程，可以有效且有力地基于机器学习预测结果进行统计推断。

    

    现代科学研究面临的主要挑战是黄金标准数据的有限可用性，而获取这些数据既耗费时间又费力。随着机器学习（ML）的快速发展，科学家们依赖于ML算法使用易得的协变量来预测这些黄金标准结果。然而，这些预测结果常常直接用于后续的统计分析中，忽略了预测过程引入的不精确性和异质性。这可能导致虚假的正面结果和无效的科学结论。在这项工作中，我们介绍了一种假设简化和数据自适应的后预测推断（POP-Inf）过程，它允许基于ML预测结果进行有效和有力的推断。它的“假设简化”属性保证在广泛的统计量上不基于ML预测做出可靠的统计推断。它的“数据自适应”特性保证了相较于现有方法的效率提高。

    A primary challenge facing modern scientific research is the limited availability of gold-standard data which can be both costly and labor-intensive to obtain. With the rapid development of machine learning (ML), scientists have relied on ML algorithms to predict these gold-standard outcomes with easily obtained covariates. However, these predicted outcomes are often used directly in subsequent statistical analyses, ignoring imprecision and heterogeneity introduced by the prediction procedure. This will likely result in false positive findings and invalid scientific conclusions. In this work, we introduce an assumption-lean and data-adaptive Post-Prediction Inference (POP-Inf) procedure that allows valid and powerful inference based on ML-predicted outcomes. Its "assumption-lean" property guarantees reliable statistical inference without assumptions on the ML-prediction, for a wide range of statistical quantities. Its "data-adaptive'" feature guarantees an efficiency gain over existing
    
[^3]: 深度学习的统计理论综述：近似，训练动态和生成模型

    A Survey on Statistical Theory of Deep Learning: Approximation, Training Dynamics, and Generative Models. (arXiv:2401.07187v1 [stat.ML])

    [http://arxiv.org/abs/2401.07187](http://arxiv.org/abs/2401.07187)

    该论文综述了深度学习的统计理论，包括近似方法、训练动态和生成模型。在非参数框架中，结果揭示了神经网络过度风险的快速收敛速率，以及如何通过梯度方法训练网络以找到良好的泛化解决方案。

    

    在这篇文章中，我们从三个角度回顾了关于神经网络统计理论的文献。第一部分回顾了在回归或分类的非参数框架下关于神经网络过度风险的结果。这些结果依赖于神经网络的显式构造，以及采用了近似理论的工具，导致过度风险的快速收敛速率。通过这些构造，可以用样本大小、数据维度和函数平滑性来表达网络的宽度和深度。然而，他们的基本分析仅适用于深度神经网络高度非凸的全局极小值点。这促使我们在第二部分回顾神经网络的训练动态。具体而言，我们回顾了那些试图回答“基于梯度方法训练的神经网络如何找到能够在未见数据上有良好泛化性能的解”的论文。尤其是两个知名的

    In this article, we review the literature on statistical theories of neural networks from three perspectives. In the first part, results on excess risks for neural networks are reviewed in the nonparametric framework of regression or classification. These results rely on explicit constructions of neural networks, leading to fast convergence rates of excess risks, in that tools from the approximation theory are adopted. Through these constructions, the width and depth of the networks can be expressed in terms of sample size, data dimension, and function smoothness. Nonetheless, their underlying analysis only applies to the global minimizer in the highly non-convex landscape of deep neural networks. This motivates us to review the training dynamics of neural networks in the second part. Specifically, we review papers that attempt to answer ``how the neural network trained via gradient-based methods finds the solution that can generalize well on unseen data.'' In particular, two well-know
    
[^4]: 利用生成神经网络模拟特征函数

    Generative neural networks for characteristic functions. (arXiv:2401.04778v1 [stat.ML])

    [http://arxiv.org/abs/2401.04778](http://arxiv.org/abs/2401.04778)

    本论文研究了利用生成神经网络模拟特征函数的问题，并通过构建一个普适且无需假设的生成神经网络来解决。研究基于最大均值差异度量，并提出了有关逼近质量的有限样本保证。

    

    在这项工作中，我们提供了一个模拟算法来从一个（多元）特征函数中模拟，该特征函数仅以黑盒格式可访问。我们构建了一个生成神经网络，其损失函数利用最大均值差异度量的特定表示，直接结合目标特征函数。这种构造具有普遍性，不依赖于维度，并且不需要对给定特征函数进行任何假设。此外，还得出了关于最大均值差异度量的逼近质量的有限样本保证。该方法在一个短期模拟研究中进行了说明。

    In this work, we provide a simulation algorithm to simulate from a (multivariate) characteristic function, which is only accessible in a black-box format. We construct a generative neural network, whose loss function exploits a specific representation of the Maximum-Mean-Discrepancy metric to directly incorporate the targeted characteristic function. The construction is universal in the sense that it is independent of the dimension and that it does not require any assumptions on the given characteristic function. Furthermore, finite sample guarantees on the approximation quality in terms of the Maximum-Mean Discrepancy metric are derived. The method is illustrated in a short simulation study.
    
[^5]: 基于能量的扩散生成器用于高效采样Boltzmann分布

    Energy based diffusion generator for efficient sampling of Boltzmann distributions. (arXiv:2401.02080v1 [cs.LG])

    [http://arxiv.org/abs/2401.02080](http://arxiv.org/abs/2401.02080)

    介绍了一种称为基于能量的扩散生成器的新型采样器，用于从任意目标分布中生成样本，并通过扩散模型和广义哈密顿动力学提高采样性能。在各种复杂分布函数上的实证评估中表现出优越性。

    

    我们介绍了一种称为基于能量的扩散生成器的新型采样器，用于从任意目标分布中生成样本。采样模型采用类似变分自编码器的结构，利用解码器将来自简单分布的潜在变量转换为逼近目标分布的随机变量，并设计了基于扩散模型的编码器。利用扩散模型对复杂分布的强大建模能力，我们可以获得生成样本和目标分布之间的Kullback-Leibler散度的准确变分估计。此外，我们提出了基于广义哈密顿动力学的解码器，进一步提高采样性能。通过实证评估，我们展示了我们的方法在各种复杂分布函数上的有效性，展示了其相对于现有方法的优越性。

    We introduce a novel sampler called the energy based diffusion generator for generating samples from arbitrary target distributions. The sampling model employs a structure similar to a variational autoencoder, utilizing a decoder to transform latent variables from a simple distribution into random variables approximating the target distribution, and we design an encoder based on the diffusion model. Leveraging the powerful modeling capacity of the diffusion model for complex distributions, we can obtain an accurate variational estimate of the Kullback-Leibler divergence between the distributions of the generated samples and the target. Moreover, we propose a decoder based on generalized Hamiltonian dynamics to further enhance sampling performance. Through empirical evaluation, we demonstrate the effectiveness of our method across various complex distribution functions, showcasing its superiority compared to existing methods.
    
[^6]: 学习能力：模型有效维度的度量方式

    Learning Capacity: A Measure of the Effective Dimensionality of a Model. (arXiv:2305.17332v1 [cs.LG])

    [http://arxiv.org/abs/2305.17332](http://arxiv.org/abs/2305.17332)

    学习能力是一种度量模型有效维度的方法，它可以帮助我们判断是否需要获取更多数据或者寻找新的体系结构以提高性能。

    

    我们利用热力学和推理之间的正式对应关系，将样本数量视为反温度，定义了一种“学习能力”，这是模型有效维度的度量方式。我们发现，对于许多在典型数据集上训练的深度网络，学习能力仅占参数数量的一小部分，取决于用于训练的样本数量，并且在数值上与从PAC-Bayesian框架获得的能力概念一致。学习能力作为测试误差的函数不会出现双峰下降。我们展示了模型的学习能力在非常小和非常大的样本大小处饱和，这提供了指导，说明是否应该获取更多数据或者寻找新的体系结构以提高性能。我们展示了如何使用学习能力来理解有效维数，即使是非参数模型，如随机森林。

    We exploit a formal correspondence between thermodynamics and inference, where the number of samples can be thought of as the inverse temperature, to define a "learning capacity'' which is a measure of the effective dimensionality of a model. We show that the learning capacity is a tiny fraction of the number of parameters for many deep networks trained on typical datasets, depends upon the number of samples used for training, and is numerically consistent with notions of capacity obtained from the PAC-Bayesian framework. The test error as a function of the learning capacity does not exhibit double descent. We show that the learning capacity of a model saturates at very small and very large sample sizes; this provides guidelines, as to whether one should procure more data or whether one should search for new architectures, to improve performance. We show how the learning capacity can be used to understand the effective dimensionality, even for non-parametric models such as random fores
    
[^7]: Sarah Frank-Wolfe：具有最佳速率和实用特点的约束优化方法

    Sarah Frank-Wolfe: Methods for Constrained Optimization with Best Rates and Practical Features. (arXiv:2304.11737v1 [math.OC])

    [http://arxiv.org/abs/2304.11737](http://arxiv.org/abs/2304.11737)

    本论文介绍了两种新的随机FW有限和最小化算法变体，适用于凸函数和非凸函数，且具有最佳收敛保证。同时两种方法不需要永久收集大批数据和全确定性梯度。

    

    Frank-Wolfe（FW）方法是解决机器学习应用中出现的结构化约束优化问题的流行方法。近年来，受到大数据集的启发，FW的随机版本变得更加流行，因为计算全梯度代价过高。本文介绍了两种新的FW随机有限和最小化算法变体。我们的算法既适用于凸函数又适用于非凸函数。我们的方法不存在永久收集大批数据的问题，这是许多投影无约束随机方法的共同问题。此外，我们的第二种方法既不需要大批量的数据也不需要全确定性梯度，这是许多有限和问题技术的典型弱点。我们方法的更快收敛速度在实践中得到了验证。

    The Frank-Wolfe (FW) method is a popular approach for solving optimization problems with structured constraints that arise in machine learning applications. In recent years, stochastic versions of FW have gained popularity, motivated by large datasets for which the computation of the full gradient is prohibitively expensive. In this paper, we present two new variants of the FW algorithms for stochastic finite-sum minimization. Our algorithms have the best convergence guarantees of existing stochastic FW approaches for both convex and non-convex objective functions. Our methods do not have the issue of permanently collecting large batches, which is common to many stochastic projection-free approaches. Moreover, our second approach does not require either large batches or full deterministic gradients, which is a typical weakness of many techniques for finite-sum problems. The faster theoretical rates of our approaches are confirmed experimentally.
    
[^8]: 高维度和普遍一致的k样本检验

    High-dimensional and universally consistent k-sample tests. (arXiv:1910.08883v4 [stat.ML] UPDATED)

    [http://arxiv.org/abs/1910.08883](http://arxiv.org/abs/1910.08883)

    本文证明了独立性测试实现了普遍一致的k样本检验，并且发现非参数独立性测试通常比多元方差分析(MANOVA)测试在高斯分布情况下表现更好。

    

    k样本检验问题涉及确定$k$组数据点是否都来自同一个分布。尽管多元方差分析(MANOVA)是生物医学中常用的k样本检验方法，但它依赖于强大且通常不合适的参数假设。此外，独立性测试和k样本测试密切相关，一些普遍一致的高维独立性测试，如距离相关(Discrepancy)和Hilbert-Schmidt独立性准则(Hsic)，具有坚实的理论和实证性质。在本文中，我们证明了独立性测试实现了普遍一致的k样本检验，并且k样本统计量，如Energy和Maximum Mean Discrepancy(MMD)，与Discrepancy完全等价。对非参数独立性测试的实证评估表明，它们通常比流行的MANOVA测试表现更好，即使在高斯分布的场景中也是如此。

    The k-sample testing problem involves determining whether $k$ groups of data points are each drawn from the same distribution. The standard method for k-sample testing in biomedicine is Multivariate analysis of variance (MANOVA), despite that it depends on strong, and often unsuitable, parametric assumptions. Moreover, independence testing and k-sample testing are closely related, and several universally consistent high-dimensional independence tests such as distance correlation (Dcorr) and Hilbert-Schmidt-Independence-Criterion (Hsic) enjoy solid theoretical and empirical properties. In this paper, we prove that independence tests achieve universally consistent k-sample testing and that k-sample statistics such as Energy and Maximum Mean Discrepancy (MMD) are precisely equivalent to Dcorr. An empirical evaluation of nonparametric independence tests showed that they generally perform better than the popular MANOVA test, even in Gaussian distributed scenarios. The evaluation included se
    

