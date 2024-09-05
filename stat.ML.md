# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fast and interpretable Support Vector Classification based on the truncated ANOVA decomposition](https://arxiv.org/abs/2402.02438) | 基于截断ANOVA分解的快速可解释支持向量分类法能够通过使用特征映射和少量维度的多变量基函数来快速且准确地进行高维散乱数据的分类。 |
| [^2] | [Adaptive operator learning for infinite-dimensional Bayesian inverse problems.](http://arxiv.org/abs/2310.17844) | 该论文提出了一种自适应操作员学习框架，通过使用贪婪算法选择自适应点对预训练的近似模型进行微调，逐渐减少建模误差。这种方法可以在准确性和效率之间取得平衡，有助于有效解决贝叶斯逆问题中的计算问题。 |
| [^3] | [Simultaneous Dimensionality Reduction: A Data Efficient Approach for Multimodal Representations Learning.](http://arxiv.org/abs/2310.04458) | 该论文介绍了一种数据高效的多模态表示学习方法，探索了独立降维和同时降维两种方法，并通过生成线性模型评估了其相对准确性和数据集大小要求。 |
| [^4] | [Sample Complexity of Variance-reduced Distributionally Robust Q-learning.](http://arxiv.org/abs/2305.18420) | 本文提出了两种新颖的无模型算法，为动态决策面对分布变化问题提供了鲁棒的解决方案，并通过将Q-learning与方差减少技术相结合，实现了样本复杂度的有效控制。 |
| [^5] | [Small noise analysis for Tikhonov and RKHS regularizations.](http://arxiv.org/abs/2305.11055) | 该研究建立了一个小噪声分析框架，揭示了传统L2正则化范数的潜在不稳定性，并提出了一种自适应分数阶RKHS正则化器类来解决不稳定性，这些正则化器始终产生最佳的收敛速率。 |

# 详细

[^1]: 基于截断ANOVA分解的快速可解释支持向量分类法

    Fast and interpretable Support Vector Classification based on the truncated ANOVA decomposition

    [https://arxiv.org/abs/2402.02438](https://arxiv.org/abs/2402.02438)

    基于截断ANOVA分解的快速可解释支持向量分类法能够通过使用特征映射和少量维度的多变量基函数来快速且准确地进行高维散乱数据的分类。

    

    支持向量机（SVM）是在散乱数据上进行分类的重要工具，在高维空间中通常需要处理许多数据点。我们提出使用基于三角函数或小波的特征映射来解决SVM的原始形式。在小维度设置中，快速傅里叶变换（FFT）和相关方法是处理所考虑基函数的强大工具。随着维度的增长，由于维数灾难，传统的基于FFT的方法变得低效。因此，我们限制自己使用多变量基函数，每个基函数只依赖于少数几个维度。这是由于效应的稀疏性和最近关于函数从散乱数据中的截断方差分解的重建的结果所带来的动机，使得生成的模型在特征的重要性以及它们的耦合方面具有可解释性。

    Support Vector Machines (SVMs) are an important tool for performing classification on scattered data, where one usually has to deal with many data points in high-dimensional spaces. We propose solving SVMs in primal form using feature maps based on trigonometric functions or wavelets. In small dimensional settings the Fast Fourier Transform (FFT) and related methods are a powerful tool in order to deal with the considered basis functions. For growing dimensions the classical FFT-based methods become inefficient due to the curse of dimensionality. Therefore, we restrict ourselves to multivariate basis functions, each one of them depends only on a small number of dimensions. This is motivated by the well-known sparsity of effects and recent results regarding the reconstruction of functions from scattered data in terms of truncated analysis of variance (ANOVA) decomposition, which makes the resulting model even interpretable in terms of importance of the features as well as their coupling
    
[^2]: 自适应操作员学习用于无限维贝叶斯逆问题

    Adaptive operator learning for infinite-dimensional Bayesian inverse problems. (arXiv:2310.17844v1 [math.NA])

    [http://arxiv.org/abs/2310.17844](http://arxiv.org/abs/2310.17844)

    该论文提出了一种自适应操作员学习框架，通过使用贪婪算法选择自适应点对预训练的近似模型进行微调，逐渐减少建模误差。这种方法可以在准确性和效率之间取得平衡，有助于有效解决贝叶斯逆问题中的计算问题。

    

    贝叶斯逆问题(BIPs)中的基本计算问题源于需要重复进行正向模型评估的要求。减少这种成本的一种常见策略是通过操作员学习使用计算效率高的近似方法替代昂贵的模型模拟，这受到了深度学习的最新进展的启发。然而，直接使用近似模型可能引入建模误差，加剧了逆问题已经存在的病态性。因此，在有效实施这些方法中，平衡准确性和效率至关重要。为此，我们开发了一个自适应操作员学习框架，可以通过强制在局部区域中准确拟合的代理逐渐减少建模误差。这是通过使用贪婪算法选择的自适应点在反演过程中对预训练的近似模型进行微调来实现的，该算法只需要少量的正向模型评估。

    The fundamental computational issues in Bayesian inverse problems (BIPs) governed by partial differential equations (PDEs) stem from the requirement of repeated forward model evaluations. A popular strategy to reduce such cost is to replace expensive model simulations by computationally efficient approximations using operator learning, motivated by recent progresses in deep learning. However, using the approximated model directly may introduce a modeling error, exacerbating the already ill-posedness of inverse problems. Thus, balancing between accuracy and efficiency is essential for the effective implementation of such approaches. To this end, we develop an adaptive operator learning framework that can reduce modeling error gradually by forcing the surrogate to be accurate in local areas. This is accomplished by fine-tuning the pre-trained approximate model during the inversion process with adaptive points selected by a greedy algorithm, which requires only a few forward model evaluat
    
[^3]: 同时降维：一种数据高效的多模态表示学习方法

    Simultaneous Dimensionality Reduction: A Data Efficient Approach for Multimodal Representations Learning. (arXiv:2310.04458v1 [stat.ML])

    [http://arxiv.org/abs/2310.04458](http://arxiv.org/abs/2310.04458)

    该论文介绍了一种数据高效的多模态表示学习方法，探索了独立降维和同时降维两种方法，并通过生成线性模型评估了其相对准确性和数据集大小要求。

    

    本文探索了两种主要的降维方法：独立降维(IDR)和同时降维(SDR)。在IDR方法中，每个模态都被独立压缩，力图保留每个模态内的尽可能多的变化。相反，在SDR中，同时压缩模态以最大化减少描述之间的协变性，同时对保留单个变化的程度不太关注。典型的例子包括偏最小二乘法和典型相关分析。虽然这些降维方法是统计学的主要方法，但它们的相对精度和数据集大小要求尚不清楚。我们引入了一个生成线性模型来合成具有已知方差和协方差结构的多模态数据，以研究这些问题。我们评估了协方差的重构准确性。

    We explore two primary classes of approaches to dimensionality reduction (DR): Independent Dimensionality Reduction (IDR) and Simultaneous Dimensionality Reduction (SDR). In IDR methods, of which Principal Components Analysis is a paradigmatic example, each modality is compressed independently, striving to retain as much variation within each modality as possible. In contrast, in SDR, one simultaneously compresses the modalities to maximize the covariation between the reduced descriptions while paying less attention to how much individual variation is preserved. Paradigmatic examples include Partial Least Squares and Canonical Correlations Analysis. Even though these DR methods are a staple of statistics, their relative accuracy and data set size requirements are poorly understood. We introduce a generative linear model to synthesize multimodal data with known variance and covariance structures to examine these questions. We assess the accuracy of the reconstruction of the covariance s
    
[^4]: 方差减少的分布式鲁棒Q-learning的样本复杂度

    Sample Complexity of Variance-reduced Distributionally Robust Q-learning. (arXiv:2305.18420v1 [cs.LG])

    [http://arxiv.org/abs/2305.18420](http://arxiv.org/abs/2305.18420)

    本文提出了两种新颖的无模型算法，为动态决策面对分布变化问题提供了鲁棒的解决方案，并通过将Q-learning与方差减少技术相结合，实现了样本复杂度的有效控制。

    

    在强化学习的理论和应用中，面对分布转移的动态决策是基本问题，因为数据收集所基于的环境分布可能会不同于模型部署所基于的分布。本文提出了两种新颖的无模型算法，即分布式鲁棒Q-learning和它的方差减少对应算法，能够高效地学习鲁棒策略，尽管会面对分布变化。这些算法旨在将带有Kullback-Leibler不确定性集的无限时域$\gamma$-折扣鲁棒马尔科夫决策过程的$q$-函数以元素$\epsilon$-精度有效逼近。进一步地，方差减少的分布式鲁棒Q-learning将同步Q-learning与方差减少技术相结合，以增强其性能，并且我们建立了它达到$ \tilde O(|S||A|(1-\gamma)^{-4}\epsilon^{-4}$的最小最大样本复杂度上界。

    Dynamic decision making under distributional shifts is of fundamental interest in theory and applications of reinforcement learning: The distribution of the environment on which the data is collected can differ from that of the environment on which the model is deployed. This paper presents two novel model-free algorithms, namely the distributionally robust Q-learning and its variance-reduced counterpart, that can effectively learn a robust policy despite distributional shifts. These algorithms are designed to efficiently approximate the $q$-function of an infinite-horizon $\gamma$-discounted robust Markov decision process with Kullback-Leibler uncertainty set to an entry-wise $\epsilon$-degree of precision. Further, the variance-reduced distributionally robust Q-learning combines the synchronous Q-learning with variance-reduction techniques to enhance its performance. Consequently, we establish that it attains a minmax sample complexity upper bound of $\tilde O(|S||A|(1-\gamma)^{-4}\e
    
[^5]: Tikhonov和RKHS正则化的小噪声分析

    Small noise analysis for Tikhonov and RKHS regularizations. (arXiv:2305.11055v1 [stat.ML])

    [http://arxiv.org/abs/2305.11055](http://arxiv.org/abs/2305.11055)

    该研究建立了一个小噪声分析框架，揭示了传统L2正则化范数的潜在不稳定性，并提出了一种自适应分数阶RKHS正则化器类来解决不稳定性，这些正则化器始终产生最佳的收敛速率。

    

    正则化在机器学习和反问题中起着至关重要的作用。然而，各种正则化范数的基本比较分析仍然未解决。我们建立了一个小噪声分析框架，以评估Tikhonov和RKHS正则化范数在高斯噪声的不适定线性反问题中的效果。该框架研究了正则化估计器在小噪声极限下的收敛速率，并揭示了传统L2正则化的潜在不稳定性。我们通过提出一种创新的自适应分数阶RKHS正则化器类来解决这种不稳定性，通过调整分数光滑度参数，该类覆盖了L2 Tikhonov和RKHS正则化器。一个令人惊奇的观点是，通过这些分数阶RKHS进行过度平滑始终产生最佳的收敛速率，但最佳的超参数可能衰减得太快而无法在实践中进行选择。

    Regularization plays a pivotal role in ill-posed machine learning and inverse problems. However, the fundamental comparative analysis of various regularization norms remains open. We establish a small noise analysis framework to assess the effects of norms in Tikhonov and RKHS regularizations, in the context of ill-posed linear inverse problems with Gaussian noise. This framework studies the convergence rates of regularized estimators in the small noise limit and reveals the potential instability of the conventional L2-regularizer. We solve such instability by proposing an innovative class of adaptive fractional RKHS regularizers, which covers the L2 Tikhonov and RKHS regularizations by adjusting the fractional smoothness parameter. A surprising insight is that over-smoothing via these fractional RKHSs consistently yields optimal convergence rates, but the optimal hyper-parameter may decay too fast to be selected in practice.
    

