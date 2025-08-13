# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Neural Operator Variational Inference based on Regularized Stein Discrepancy for Deep Gaussian Processes.](http://arxiv.org/abs/2309.12658) | 基于正则化Stein差异的神经算子变分推断用于深度高斯过程，通过使用神经生成器获得取样器以及使用蒙特卡罗估计和子采样随机优化技术解决极小极大问题，提高了深度高斯过程模型的表达能力和推断效果。 |

# 详细

[^1]: 基于正则化Stein差异的神经算子变分推断用于深度高斯过程

    Neural Operator Variational Inference based on Regularized Stein Discrepancy for Deep Gaussian Processes. (arXiv:2309.12658v1 [cs.LG])

    [http://arxiv.org/abs/2309.12658](http://arxiv.org/abs/2309.12658)

    基于正则化Stein差异的神经算子变分推断用于深度高斯过程，通过使用神经生成器获得取样器以及使用蒙特卡罗估计和子采样随机优化技术解决极小极大问题，提高了深度高斯过程模型的表达能力和推断效果。

    

    深度高斯过程（DGP）模型提供了一种强大的非参数贝叶斯推断方法，但精确推断通常是难以求解的，这促使我们使用各种近似方法。然而，现有的方法，如均值场高斯假设，限制了DGP模型的表达能力和效果，而随机逼近可能计算代价高昂。为解决这些挑战，我们引入了基于神经算子的变分推断（NOVI）用于深度高斯过程。NOVI使用神经生成器获得取样器，并在L2空间中最小化生成分布和真实后验之间的正则化Stein差异。我们使用蒙特卡罗估计和子采样随机优化技术解决了极小极大问题。我们证明了通过将Fisher散度与常数相乘来控制方法引入的偏差，从而实现了鲁棒的误差控制，确保了算法的稳定性和精确性。

    Deep Gaussian Process (DGP) models offer a powerful nonparametric approach for Bayesian inference, but exact inference is typically intractable, motivating the use of various approximations. However, existing approaches, such as mean-field Gaussian assumptions, limit the expressiveness and efficacy of DGP models, while stochastic approximation can be computationally expensive. To tackle these challenges, we introduce Neural Operator Variational Inference (NOVI) for Deep Gaussian Processes. NOVI uses a neural generator to obtain a sampler and minimizes the Regularized Stein Discrepancy in L2 space between the generated distribution and true posterior. We solve the minimax problem using Monte Carlo estimation and subsampling stochastic optimization techniques. We demonstrate that the bias introduced by our method can be controlled by multiplying the Fisher divergence with a constant, which leads to robust error control and ensures the stability and precision of the algorithm. Our experim
    

