# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Asymptotic Dynamics of Alternating Minimization for Non-Convex Optimization](https://arxiv.org/abs/2402.04751) | 本文研究了交替极小化在非凸优化中的渐近动力学特性，通过复制方法追踪演化过程，发现其可用二维离散随机过程有效描述，存在记忆依赖性。该理论框架适用于分析各种迭代算法。 |
| [^2] | [Provably Stable Feature Rankings with SHAP and LIME.](http://arxiv.org/abs/2401.15800) | 这项研究提出了一种通过利用多重假设检验的思想，来设计可靠地排名机器学习模型中最重要特征的特征归因方法，旨在解决SHAP和LIME等常用方法由于随机采样导致的高度不稳定性问题。实验证明了该方法的有效性和计算效率。 |

# 详细

[^1]: 非凸优化中Alternating Minimization的渐近动力学

    Asymptotic Dynamics of Alternating Minimization for Non-Convex Optimization

    [https://arxiv.org/abs/2402.04751](https://arxiv.org/abs/2402.04751)

    本文研究了交替极小化在非凸优化中的渐近动力学特性，通过复制方法追踪演化过程，发现其可用二维离散随机过程有效描述，存在记忆依赖性。该理论框架适用于分析各种迭代算法。

    

    本研究探讨了在具有正态分布协变量的双线性非凸函数优化中应用交替极小化的渐近动力学。我们采用统计物理学中的复制方法，通过多步方法精确追踪算法的演变。研究结果表明，动力学可以有效地用一个二维离散随机过程来描述，每一步都依赖于所有先前的时间步长，揭示了过程中的记忆依赖性。本文开发的理论框架广泛适用于各种迭代算法的分析，超越了交替极小化的范围。

    This study investigates the asymptotic dynamics of alternating minimization applied to optimize a bilinear non-convex function with normally distributed covariates. We employ the replica method from statistical physics in a multi-step approach to precisely trace the algorithm's evolution. Our findings indicate that the dynamics can be described effectively by a two--dimensional discrete stochastic process, where each step depends on all previous time steps, revealing a memory dependency in the procedure. The theoretical framework developed in this work is broadly applicable for the analysis of various iterative algorithms, extending beyond the scope of alternating minimization.
    
[^2]: 使用SHAP和LIME进行可证明稳定的特征排名

    Provably Stable Feature Rankings with SHAP and LIME. (arXiv:2401.15800v1 [stat.ML])

    [http://arxiv.org/abs/2401.15800](http://arxiv.org/abs/2401.15800)

    这项研究提出了一种通过利用多重假设检验的思想，来设计可靠地排名机器学习模型中最重要特征的特征归因方法，旨在解决SHAP和LIME等常用方法由于随机采样导致的高度不稳定性问题。实验证明了该方法的有效性和计算效率。

    

    特征归因是了解机器学习模型预测的普遍工具。然而，用于评分输入变量的常用方法，如SHAP和LIME，由于随机采样而具有高度不稳定性。借鉴多重假设检验的思想，我们设计了能够以高概率正确排名最重要特征的归因方法。我们的算法RankSHAP保证$K$个最高Shapley值具有超过$1-\alpha$的正确排序概率。实证结果证明了其有效性和令人印象深刻的计算效率。我们还在之前的工作基础上为LIME提供了类似的结果，确保以正确顺序选择最重要的特征。

    Feature attributions are ubiquitous tools for understanding the predictions of machine learning models. However, popular methods for scoring input variables such as SHAP and LIME suffer from high instability due to random sampling. Leveraging ideas from multiple hypothesis testing, we devise attribution methods that correctly rank the most important features with high probability. Our algorithm RankSHAP guarantees that the $K$ highest Shapley values have the proper ordering with probability exceeding $1-\alpha$. Empirical results demonstrate its validity and impressive computational efficiency. We also build on previous work to yield similar results for LIME, ensuring the most important features are selected in the right order.
    

