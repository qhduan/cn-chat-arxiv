# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Transfer Learning Bayesian Optimization to Design Competitor DNA Molecules for Use in Diagnostic Assays](https://arxiv.org/abs/2402.17704) | 通过将转移学习的代理模型与贝叶斯优化相结合，本文展示了如何通过在优化任务之间共享信息来减少实验的总数，并且演示了在设计用于扩增基因诊断测定的DNA竞争对手时实验数量的减少。 |
| [^2] | [Stochastic Gradient Descent for Additive Nonparametric Regression](https://arxiv.org/abs/2401.00691) | 本文介绍了一种用于训练加性模型的随机梯度下降算法，具有良好的内存存储和计算要求。在规范很好的情况下，通过仔细选择学习率，可以实现最小和最优的风险。 |
| [^3] | [Community Detection and Classification Guarantees Using Embeddings Learned by Node2Vec.](http://arxiv.org/abs/2310.17712) | 本研究通过分析Node2Vec学习到的嵌入的理论属性，证明了在（经过度修正的）随机块模型中，使用k-means聚类方法对这些嵌入进行社区恢复是弱一致的。实验证明这一结果，并探讨了嵌入在节点和链接预测任务中的应用。 |
| [^4] | [Implicit regularization via soft ascent-descent.](http://arxiv.org/abs/2310.10006) | 本研究提出了一种通过软化的逐点机制（SoftAD）来实现正则化的方法，该方法具有更好的鲁棒性，可以减少超参数的影响，并保留上升-下降效应。 |
| [^5] | [Online Tensor Learning: Computational and Statistical Trade-offs, Adaptivity and Optimal Regret.](http://arxiv.org/abs/2306.03372) | 本文提出了在线黎曼梯度下降算法，用于在在线情况下估计潜在的低秩张量。其中，我们在处理连续或分类变量时提供了灵活的方法，并在在线情况下尝试了两个具体的应用，即在线张量补全和在线二元张量学习。我们还建立了逐个条目的精确错误界限，这是在在线张量补全中首次纳入噪声。我们观察到，在存在噪声的情况下，计算和统计方面存在着令人惊讶的权衡。 |
| [^6] | [Imprecise Bayesian Neural Networks.](http://arxiv.org/abs/2302.09656) | 在机器学习和人工智能领域，该论文提出了一种新的算法——不精确的贝叶斯神经网络(IBNNs)。这种算法使用可信区间先验分布集合和似然分布集合进行训练，相比标准的BNNs，可以区分先验和后验的不确定性并量化。此外，IBNNs在贝叶斯灵敏度分析方面具有更强的鲁棒性，并且对分布变化也更加鲁棒。 |
| [^7] | [Targeted Separation and Convergence with Kernel Discrepancies.](http://arxiv.org/abs/2209.12835) | 通过核差异度量，我们推导出了新的充分必要条件，实现了将目标分离出来，以及控制对目标的弱收敛性。此外，我们在$\mathbb{R}^d$上使用了这些结果来扩展了核Stein差异分离和收敛控制的已知条件，并开发了能够精确度量目标的弱收敛性的核差异度量。 |

# 详细

[^1]: 将贝叶斯优化应用于转移学习以设计用于诊断测定的竞争对手DNA分子

    Transfer Learning Bayesian Optimization to Design Competitor DNA Molecules for Use in Diagnostic Assays

    [https://arxiv.org/abs/2402.17704](https://arxiv.org/abs/2402.17704)

    通过将转移学习的代理模型与贝叶斯优化相结合，本文展示了如何通过在优化任务之间共享信息来减少实验的总数，并且演示了在设计用于扩增基因诊断测定的DNA竞争对手时实验数量的减少。

    

    随着工程生物分子设备的兴起，定制生物序列的需求不断增加。通常，为了特定应用需要制作许多类似的生物序列，这意味着需要进行大量甚至昂贵的实验来优化这些序列。本文提出了一个转移学习设计实验工作流程，使这种开发变得可行。通过将转移学习代理模型与贝叶斯优化相结合，我们展示了如何通过在优化任务之间共享信息来减少实验的总数。我们演示了使用用于扩增基因诊断测定中使用的DNA竞争对手开发数据来减少实验数量。我们使用交叉验证来比较不同转移学习模型的预测准确性，然后比较这些模型在单一目标和惩罚优化下的性能。

    arXiv:2402.17704v1 Announce Type: cross  Abstract: With the rise in engineered biomolecular devices, there is an increased need for tailor-made biological sequences. Often, many similar biological sequences need to be made for a specific application meaning numerous, sometimes prohibitively expensive, lab experiments are necessary for their optimization. This paper presents a transfer learning design of experiments workflow to make this development feasible. By combining a transfer learning surrogate model with Bayesian optimization, we show how the total number of experiments can be reduced by sharing information between optimization tasks. We demonstrate the reduction in the number of experiments using data from the development of DNA competitors for use in an amplification-based diagnostic assay. We use cross-validation to compare the predictive accuracy of different transfer learning models, and then compare the performance of the models for both single objective and penalized opti
    
[^2]: 添加非参数回归的随机梯度下降

    Stochastic Gradient Descent for Additive Nonparametric Regression

    [https://arxiv.org/abs/2401.00691](https://arxiv.org/abs/2401.00691)

    本文介绍了一种用于训练加性模型的随机梯度下降算法，具有良好的内存存储和计算要求。在规范很好的情况下，通过仔细选择学习率，可以实现最小和最优的风险。

    

    本文介绍了一种用于训练加性模型的迭代算法，该算法具有良好的内存存储和计算要求。该算法可以看作是对组件函数的截断基扩展的系数应用随机梯度下降的函数对应物。我们证明了得到的估计量满足一个奥拉克不等式，允许模型错误规范。在规范很好的情况下，通过在训练的三个不同阶段仔细选择学习率，我们证明了其风险在数据维度和训练样本大小的依赖方面是最小和最优的。通过在两个实际数据集上将该方法与传统的反向拟合进行比较，我们进一步说明了计算优势。

    This paper introduces an iterative algorithm for training additive models that enjoys favorable memory storage and computational requirements. The algorithm can be viewed as the functional counterpart of stochastic gradient descent, applied to the coefficients of a truncated basis expansion of the component functions. We show that the resulting estimator satisfies an oracle inequality that allows for model mis-specification. In the well-specified setting, by choosing the learning rate carefully across three distinct stages of training, we demonstrate that its risk is minimax optimal in terms of the dependence on the dimensionality of the data and the size of the training sample. We further illustrate the computational benefits by comparing the approach with traditional backfitting on two real-world datasets.
    
[^3]: 使用Node2Vec学习到的嵌入进行社区检测和分类的保证

    Community Detection and Classification Guarantees Using Embeddings Learned by Node2Vec. (arXiv:2310.17712v1 [stat.ML])

    [http://arxiv.org/abs/2310.17712](http://arxiv.org/abs/2310.17712)

    本研究通过分析Node2Vec学习到的嵌入的理论属性，证明了在（经过度修正的）随机块模型中，使用k-means聚类方法对这些嵌入进行社区恢复是弱一致的。实验证明这一结果，并探讨了嵌入在节点和链接预测任务中的应用。

    

    将大型网络的节点嵌入到欧几里得空间中是现代机器学习中的常见目标，有各种工具可用。这些嵌入可以用作社区检测/节点聚类或链接预测等任务的特征，其性能达到了最先进水平。除了谱聚类方法之外，对于其他常用的学习嵌入方法，缺乏理论上的理解。在这项工作中，我们考察了由node2vec学习到的嵌入的理论属性。我们的主要结果表明，对node2vec生成的嵌入向量应用k-means聚类可以对（经过度修正的）随机块模型中的节点进行弱一致的社区恢复。我们还讨论了这些嵌入在节点和链接预测任务中的应用。我们通过实验证明了这个结果，并研究了它与网络数据的其他嵌入工具之间的关系。

    Embedding the nodes of a large network into an Euclidean space is a common objective in modern machine learning, with a variety of tools available. These embeddings can then be used as features for tasks such as community detection/node clustering or link prediction, where they achieve state of the art performance. With the exception of spectral clustering methods, there is little theoretical understanding for other commonly used approaches to learning embeddings. In this work we examine the theoretical properties of the embeddings learned by node2vec. Our main result shows that the use of k-means clustering on the embedding vectors produced by node2vec gives weakly consistent community recovery for the nodes in (degree corrected) stochastic block models. We also discuss the use of these embeddings for node and link prediction tasks. We demonstrate this result empirically, and examine how this relates to other embedding tools for network data.
    
[^4]: 通过软上升-下降实现隐式正则化

    Implicit regularization via soft ascent-descent. (arXiv:2310.10006v1 [stat.ML])

    [http://arxiv.org/abs/2310.10006](http://arxiv.org/abs/2310.10006)

    本研究提出了一种通过软化的逐点机制（SoftAD）来实现正则化的方法，该方法具有更好的鲁棒性，可以减少超参数的影响，并保留上升-下降效应。

    

    随着模型变得越来越大和复杂，通过最小的试错来实现更好的离线泛化对机器学习工作流程的可靠性和经济性至关重要。作为寻求“平坦”局部最小值的众所周知的启发式方法的代理，梯度正则化是一条自然的途径，一阶近似方法如Floding和Sharpness-Aware Minimization (SAM) 已经受到了相当大的关注，但它们的性能严重依赖于超参数（洪水阈值和邻域半径），这些超参数不容易事先确定。为了开发一个对错误超参数更具韧性的过程，受Flooding中使用的硬阈值“上升-下降”开关装置的启发，我们提出了一种软化的逐点机制，称为SoftAD，它对边界上的点进行降权，限制异常值的影响，并保留上升-下降效应。我们将形式的平稳性保证与Flooding进行对比。

    As models grow larger and more complex, achieving better off-sample generalization with minimal trial-and-error is critical to the reliability and economy of machine learning workflows. As a proxy for the well-studied heuristic of seeking "flat" local minima, gradient regularization is a natural avenue, and first-order approximations such as Flooding and sharpness-aware minimization (SAM) have received significant attention, but their performance depends critically on hyperparameters (flood threshold and neighborhood radius, respectively) that are non-trivial to specify in advance. In order to develop a procedure which is more resilient to misspecified hyperparameters, with the hard-threshold "ascent-descent" switching device used in Flooding as motivation, we propose a softened, pointwise mechanism called SoftAD that downweights points on the borderline, limits the effects of outliers, and retains the ascent-descent effect. We contrast formal stationarity guarantees with those for Flo
    
[^5]: 在线张量学习：计算和统计权衡，适应性和最优遗憾

    Online Tensor Learning: Computational and Statistical Trade-offs, Adaptivity and Optimal Regret. (arXiv:2306.03372v1 [stat.ML])

    [http://arxiv.org/abs/2306.03372](http://arxiv.org/abs/2306.03372)

    本文提出了在线黎曼梯度下降算法，用于在在线情况下估计潜在的低秩张量。其中，我们在处理连续或分类变量时提供了灵活的方法，并在在线情况下尝试了两个具体的应用，即在线张量补全和在线二元张量学习。我们还建立了逐个条目的精确错误界限，这是在在线张量补全中首次纳入噪声。我们观察到，在存在噪声的情况下，计算和统计方面存在着令人惊讶的权衡。

    

    我们研究了一个广义框架，用于在线情况下估计潜在的低秩张量，包括线性和广义线性模型。该框架提供了一种处理连续或分类变量的灵活方法。此外，我们研究了两个具体的应用：在线张量补全和在线二元张量学习。为了应对这些挑战，我们提出了在线黎曼梯度下降算法，在所有应用程序中都可以根据适当的条件线性收敛并恢复低秩组件。此外，我们为在线张量补全建立了精确的逐个条目错误界限。值得注意的是，我们的工作代表了首次尝试在在线低秩张量恢复任务中纳入噪声的努力。有趣的是，我们观察到在存在噪声的情况下，在计算和统计方面存在着令人惊讶的权衡。增加步长可以加快收敛，但会导致更高的统计误差。

    We investigate a generalized framework for estimating latent low-rank tensors in an online setting, encompassing both linear and generalized linear models. This framework offers a flexible approach for handling continuous or categorical variables. Additionally, we investigate two specific applications: online tensor completion and online binary tensor learning. To address these challenges, we propose the online Riemannian gradient descent algorithm, which demonstrates linear convergence and the ability to recover the low-rank component under appropriate conditions in all applications. Furthermore, we establish a precise entry-wise error bound for online tensor completion. Notably, our work represents the first attempt to incorporate noise in the online low-rank tensor recovery task. Intriguingly, we observe a surprising trade-off between computational and statistical aspects in the presence of noise. Increasing the step size accelerates convergence but leads to higher statistical error
    
[^6]: 不精确的贝叶斯神经网络

    Imprecise Bayesian Neural Networks. (arXiv:2302.09656v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.09656](http://arxiv.org/abs/2302.09656)

    在机器学习和人工智能领域，该论文提出了一种新的算法——不精确的贝叶斯神经网络(IBNNs)。这种算法使用可信区间先验分布集合和似然分布集合进行训练，相比标准的BNNs，可以区分先验和后验的不确定性并量化。此外，IBNNs在贝叶斯灵敏度分析方面具有更强的鲁棒性，并且对分布变化也更加鲁棒。

    

    在机器学习和人工智能中, 确定不确定性和鲁棒性是重要的目标。虽然贝叶斯神经网络使得预测中的不确定性能够被评估，不同来源的不确定性是无法区分的。我们提出了不精确的贝叶斯神经网络（IBNNs），它们可以概括和克服标准BNNs的某些缺点。标准BNNs使用单一的先验分布和似然分布进行训练，而IBNNs使用可信区间先验分布和似然分布进行训练。它们允许区分先验和后验不确定性，并对其进行量化。此外，IBNNs在贝叶斯灵敏度分析方面具有鲁棒性，并且对分布变化比标准BNNs更加鲁棒。它们还可以用于计算具有PAC样本复杂性的结果集。我们将IBNNs应用于两个案例研究：一个是为了人工胰腺控制模拟血糖和胰岛素动力学，另一个是运动规划。

    Uncertainty quantification and robustness to distribution shifts are important goals in machine learning and artificial intelligence. Although Bayesian neural networks (BNNs) allow for uncertainty in the predictions to be assessed, different sources of uncertainty are indistinguishable. We present imprecise Bayesian neural networks (IBNNs); they generalize and overcome some of the drawbacks of standard BNNs. These latter are trained using a single prior and likelihood distributions, whereas IBNNs are trained using credal prior and likelihood sets. They allow to distinguish between aleatoric and epistemic uncertainties, and to quantify them. In addition, IBNNs are robust in the sense of Bayesian sensitivity analysis, and are more robust than BNNs to distribution shift. They can also be used to compute sets of outcomes that enjoy PAC-like properties. We apply IBNNs to two case studies. One, to model blood glucose and insulin dynamics for artificial pancreas control, and two, for motion p
    
[^7]: 通过核差异实现有针对性的分离与收敛

    Targeted Separation and Convergence with Kernel Discrepancies. (arXiv:2209.12835v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2209.12835](http://arxiv.org/abs/2209.12835)

    通过核差异度量，我们推导出了新的充分必要条件，实现了将目标分离出来，以及控制对目标的弱收敛性。此外，我们在$\mathbb{R}^d$上使用了这些结果来扩展了核Stein差异分离和收敛控制的已知条件，并开发了能够精确度量目标的弱收敛性的核差异度量。

    

    最大均值差异（MMDs）如核Stein差异（KSD）已经成为广泛应用的中心，包括假设检验、采样器选择、分布近似和变分推断。在每个设置中，这些基于核的差异度量需要实现（i）将目标P与其他概率测度分离，甚至（ii）控制对P的弱收敛。在本文中，我们推导了确保（i）和（ii）的新的充分必要条件。对于可分的度量空间上的MMDs，我们描述了分离Bochner可嵌入测度的核，并引入简单的条件来分离所有具有无界核的测度和用有界核来控制收敛。我们利用这些结果在$\mathbb{R}^d$上大大扩展了KSD分离和收敛控制的已知条件，并开发了首个能够精确度量对P的弱收敛的KSDs。在这个过程中，我们强调了我们的结果的影响。

    Maximum mean discrepancies (MMDs) like the kernel Stein discrepancy (KSD) have grown central to a wide range of applications, including hypothesis testing, sampler selection, distribution approximation, and variational inference. In each setting, these kernel-based discrepancy measures are required to (i) separate a target P from other probability measures or even (ii) control weak convergence to P. In this article we derive new sufficient and necessary conditions to ensure (i) and (ii). For MMDs on separable metric spaces, we characterize those kernels that separate Bochner embeddable measures and introduce simple conditions for separating all measures with unbounded kernels and for controlling convergence with bounded kernels. We use these results on $\mathbb{R}^d$ to substantially broaden the known conditions for KSD separation and convergence control and to develop the first KSDs known to exactly metrize weak convergence to P. Along the way, we highlight the implications of our res
    

