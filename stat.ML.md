# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stochastic Modified Flows for Riemannian Stochastic Gradient Descent](https://arxiv.org/abs/2402.03467) | 本文研究了黎曼随机梯度下降（RSGD）的收敛速度，并介绍了一种基于随机修改流（RSMF）的扩散逼近方法，该方法可以提高对RSGD的近似精度。 |
| [^2] | [Improving the Accuracy and Interpretability of Random Forests via Forest Pruning.](http://arxiv.org/abs/2401.05535) | 通过森林修剪方法，本研究提出了一种兼顾随机森林准确性和决策树可解释性的方法。实验证明，在大多数情景下，这种方法能够显著提高随机森林的性能。 |
| [^3] | [Sparse Bayesian Multidimensional Item Response Theory.](http://arxiv.org/abs/2310.17820) | 本文开发了一种可扩展的贝叶斯EM算法，用于从二元和有序项目响应中估计稀疏因子载荷，并通过贝叶斯非参数方法解决了未知潜在因子维度的问题。 |

# 详细

[^1]: 基于随机修改流的黎曼随机梯度下降

    Stochastic Modified Flows for Riemannian Stochastic Gradient Descent

    [https://arxiv.org/abs/2402.03467](https://arxiv.org/abs/2402.03467)

    本文研究了黎曼随机梯度下降（RSGD）的收敛速度，并介绍了一种基于随机修改流（RSMF）的扩散逼近方法，该方法可以提高对RSGD的近似精度。

    

    我们对黎曼随机梯度下降（RSGD）收敛速度给出了定量估计，并将其与黎曼梯度流和扩散过程——黎曼随机修改流（RSMF）进行了比较。利用随机微分几何工具，我们证明在小学习率范围内，RSGD可以近似为由无穷维维纳过程驱动的RSMF的解。RSMF考虑到了RSGD的随机波动，从而提高了与确定性黎曼梯度流的逼近顺序。RSGD使用了重传递映射的概念，即对指数映射的一种成本效益近似，我们对扩散逼近的弱误差进行了定量界定，在重传递映射、流形几何和梯度的随机估计的假设下证明了这些界定。

    We give quantitative estimates for the rate of convergence of Riemannian stochastic gradient descent (RSGD) to Riemannian gradient flow and to a diffusion process, the so-called Riemannian stochastic modified flow (RSMF). Using tools from stochastic differential geometry we show that, in the small learning rate regime, RSGD can be approximated by the solution to the RSMF driven by an infinite-dimensional Wiener process. The RSMF accounts for the random fluctuations of RSGD and, thereby, increases the order of approximation compared to the deterministic Riemannian gradient flow. The RSGD is build using the concept of a retraction map, that is, a cost efficient approximation of the exponential map, and we prove quantitative bounds for the weak error of the diffusion approximation under assumptions on the retraction map, the geometry of the manifold, and the random estimators of the gradient.
    
[^2]: 通过森林修剪提高随机森林的准确性和可解释性

    Improving the Accuracy and Interpretability of Random Forests via Forest Pruning. (arXiv:2401.05535v1 [stat.ML])

    [http://arxiv.org/abs/2401.05535](http://arxiv.org/abs/2401.05535)

    通过森林修剪方法，本研究提出了一种兼顾随机森林准确性和决策树可解释性的方法。实验证明，在大多数情景下，这种方法能够显著提高随机森林的性能。

    

    接近几十年的发展之后，随机森林仍然在各种学习问题中提供最先进的准确性，在这方面超越了决策树甚至神经网络等替代机器学习算法。然而，作为一种集成方法，随机森林在解释性方面往往比决策树表现不佳。在本研究中，我们提出了一种事后方法，旨在兼顾随机森林的准确性和决策树的可解释性。为此，我们提出了两种森林修剪方法，以在给定的随机森林内找到最佳子森林，然后在适用的情况下将选定的树合并为一棵。我们的第一种方法依赖于约束穷举搜索，而第二种方法基于LASSO方法的改进。在合成和真实世界数据集上进行的大量实验证明，在大多数情景下，这两种方法中至少有一种能够显著提高随机森林的准确性和可解释性。

    Decades after their inception, random forests continue to provide state-of-the-art accuracy in a variety of learning problems, outperforming in this respect alternative machine learning algorithms such as decision trees or even neural networks. However, being an ensemble method, the one aspect where random forests tend to severely underperform decision trees is interpretability. In the present work, we propose a post-hoc approach that aims to have the best of both worlds: the accuracy of random forests and the interpretability of decision trees. To this end, we present two forest-pruning methods to find an optimal sub-forest within a given random forest, and then, when applicable, combine the selected trees into one. Our first method relies on constrained exhaustive search, while our second method is based on an adaptation of the LASSO methodology. Extensive experiments over synthetic and real world datasets show that, in the majority of scenarios, at least one of the two methods propo
    
[^3]: 稀疏贝叶斯多维项目反应理论

    Sparse Bayesian Multidimensional Item Response Theory. (arXiv:2310.17820v1 [stat.ME])

    [http://arxiv.org/abs/2310.17820](http://arxiv.org/abs/2310.17820)

    本文开发了一种可扩展的贝叶斯EM算法，用于从二元和有序项目响应中估计稀疏因子载荷，并通过贝叶斯非参数方法解决了未知潜在因子维度的问题。

    

    多变量项目反应理论（MIRT）被应用研究人员广泛使用，以寻找问卷数据中响应模式背后的可解释（稀疏）解释。然而，在实践中，对于这种稀疏性发现工具的需求尚未得到满足。本文提出了一种用于二元和有序项目MIRT的贝叶斯平台，其需要最少的调整，并且由于其可并行化的特性，在相对较大的数据集上具有良好的可扩展性。MIRT模型的贝叶斯方法传统上依赖于MCMC模拟，在实践中可能既费时又难以通过额外的阈值设定实现精确的稀疏恢复。在本文中，我们开发了一种可扩展的贝叶斯EM算法，用于从二元和有序项目响应中估计稀疏因子载荷。我们利用贝叶斯非参数方法解决了未知潜在因子维度的看似不可逾越的问题，从而使得可以估计因子的数量。通过旋转可以实现稀疏性。

    Multivariate Item Response Theory (MIRT) is sought-after widely by applied researchers looking for interpretable (sparse) explanations underlying response patterns in questionnaire data. There is, however, an unmet demand for such sparsity discovery tools in practice. Our paper develops a Bayesian platform for binary and ordinal item MIRT which requires minimal tuning and scales well on relatively large datasets due to its parallelizable features. Bayesian methodology for MIRT models has traditionally relied on MCMC simulation, which cannot only be slow in practice, but also often renders exact sparsity recovery impossible without additional thresholding. In this work, we develop a scalable Bayesian EM algorithm to estimate sparse factor loadings from binary and ordinal item responses. We address the seemingly insurmountable problem of unknown latent factor dimensionality with tools from Bayesian nonparametrics which enable estimating the number of factors. Rotations to sparsity throug
    

