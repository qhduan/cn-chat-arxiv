# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Non-asymptotic Convergence of Discrete-time Diffusion Models: New Approach and Improved Rate](https://arxiv.org/abs/2402.13901) | 本文提出了离散时间扩散模型的新方法，改进了对更大类的分布的收敛保证，并提高了具有有界支撑的分布的收敛速率。 |
| [^2] | [Efficient Solvers for Partial Gromov-Wasserstein](https://arxiv.org/abs/2402.03664) | 本文提出了两个基于Frank-Wolfe算法的新的高效求解器来解决偏差Gromov-Wasserstein问题，并且证明了PGW问题构成了度量测度空间的度量。 |
| [^3] | [Precise Asymptotic Generalization for Multiclass Classification with Overparameterized Linear Models.](http://arxiv.org/abs/2306.13255) | 本文研究了高斯协变量下的过参数化线性模型在多类分类问题中的泛化能力，成功解决了之前的猜想，并提出的新下界具有信息论中的强对偶定理的性质。 |

# 详细

[^1]: 离散时间扩散模型的非渐近收敛：新方法和改进速率

    Non-asymptotic Convergence of Discrete-time Diffusion Models: New Approach and Improved Rate

    [https://arxiv.org/abs/2402.13901](https://arxiv.org/abs/2402.13901)

    本文提出了离散时间扩散模型的新方法，改进了对更大类的分布的收敛保证，并提高了具有有界支撑的分布的收敛速率。

    

    最近，去噪扩散模型作为一种强大的生成技术出现，将噪声转化为数据。理论上主要研究了连续时间扩散模型的收敛性保证，并且仅在文献中对具有有界支撑的分布的离散时间扩散模型进行了获得。本文为更大类的分布建立了离散时间扩散模型的收敛性保证，并进一步改进了对具有有界支撑的分布的收敛速率。特别地，首先为具有有限二阶矩的平滑和一般（可能非光滑）分布建立了收敛速率。然后将结果专门应用于一些有明确参数依赖关系的有趣分布类别，包括具有Lipschitz分数、高斯混合分布和具有有界支撑的分布。

    arXiv:2402.13901v1 Announce Type: new  Abstract: The denoising diffusion model emerges recently as a powerful generative technique that converts noise into data. Theoretical convergence guarantee has been mainly studied for continuous-time diffusion models, and has been obtained for discrete-time diffusion models only for distributions with bounded support in the literature. In this paper, we establish the convergence guarantee for substantially larger classes of distributions under discrete-time diffusion models and further improve the convergence rate for distributions with bounded support. In particular, we first establish the convergence rates for both smooth and general (possibly non-smooth) distributions having finite second moment. We then specialize our results to a number of interesting classes of distributions with explicit parameter dependencies, including distributions with Lipschitz scores, Gaussian mixture distributions, and distributions with bounded support. We further 
    
[^2]: 高效求解偏差Gromov-Wasserstein问题

    Efficient Solvers for Partial Gromov-Wasserstein

    [https://arxiv.org/abs/2402.03664](https://arxiv.org/abs/2402.03664)

    本文提出了两个基于Frank-Wolfe算法的新的高效求解器来解决偏差Gromov-Wasserstein问题，并且证明了PGW问题构成了度量测度空间的度量。

    

    偏差Gromov-Wasserstein（PGW）问题可以比较具有不均匀质量的度量空间中的测度，从而实现这些空间之间的不平衡和部分匹配。本文证明了PGW问题可以转化为Gromov-Wasserstein问题的一个变种，类似于把偏差最优运输问题转化为最优运输问题。这个转化导致了两个新的求解器，基于Frank-Wolfe算法，数学和计算上等价，提供了高效的PGW问题解决方案。我们进一步证明了PGW问题构成了度量测度空间的度量。最后，我们通过与现有基线方法在形状匹配和正样本未标记学习问题上的计算时间和性能比较，验证了我们提出的求解器的有效性。

    The partial Gromov-Wasserstein (PGW) problem facilitates the comparison of measures with unequal masses residing in potentially distinct metric spaces, thereby enabling unbalanced and partial matching across these spaces. In this paper, we demonstrate that the PGW problem can be transformed into a variant of the Gromov-Wasserstein problem, akin to the conversion of the partial optimal transport problem into an optimal transport problem. This transformation leads to two new solvers, mathematically and computationally equivalent, based on the Frank-Wolfe algorithm, that provide efficient solutions to the PGW problem. We further establish that the PGW problem constitutes a metric for metric measure spaces. Finally, we validate the effectiveness of our proposed solvers in terms of computation time and performance on shape-matching and positive-unlabeled learning problems, comparing them against existing baselines.
    
[^3]: 过参数化线性模型下多类分类的渐进泛化精度研究

    Precise Asymptotic Generalization for Multiclass Classification with Overparameterized Linear Models. (arXiv:2306.13255v1 [cs.LG])

    [http://arxiv.org/abs/2306.13255](http://arxiv.org/abs/2306.13255)

    本文研究了高斯协变量下的过参数化线性模型在多类分类问题中的泛化能力，成功解决了之前的猜想，并提出的新下界具有信息论中的强对偶定理的性质。

    

    本文研究了在具有高斯协变量双层模型下，过参数化线性模型在多类分类中的渐进泛化问题，其中数据点数、特征和类别数都同时增长。我们完全解决了Subramanian等人在'22年所提出的猜想，与预测的泛化区间相匹配。此外，我们的新的下界类似于信息论中的强对偶定理：它们能够确立误分类率逐渐趋近于0或1.我们紧密的结果的一个令人惊讶的结果是，最小范数插值分类器在最小范数插值回归器最优的范围内，可以在渐进上次优。我们分析的关键在于一种新的Hanson-Wright不等式变体，该变体在具有稀疏标签的多类问题中具有广泛的适用性。作为应用，我们展示了相同类型分析在几种不同类型的分类模型上的结果。

    We study the asymptotic generalization of an overparameterized linear model for multiclass classification under the Gaussian covariates bi-level model introduced in Subramanian et al.~'22, where the number of data points, features, and classes all grow together. We fully resolve the conjecture posed in Subramanian et al.~'22, matching the predicted regimes for generalization. Furthermore, our new lower bounds are akin to an information-theoretic strong converse: they establish that the misclassification rate goes to 0 or 1 asymptotically. One surprising consequence of our tight results is that the min-norm interpolating classifier can be asymptotically suboptimal relative to noninterpolating classifiers in the regime where the min-norm interpolating regressor is known to be optimal.  The key to our tight analysis is a new variant of the Hanson-Wright inequality which is broadly useful for multiclass problems with sparse labels. As an application, we show that the same type of analysis 
    

