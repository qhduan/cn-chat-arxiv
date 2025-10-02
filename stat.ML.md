# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Fisher-Rao Gradient of the Evidence Lower Bound.](http://arxiv.org/abs/2307.11249) | 本文研究了证据下界的Fisher-Rao梯度，揭示了它与目标分布的Kullback-Leibler散度梯度的关系，进一步证明了最小化主要目标函数与最大化ELBO的等价性。 |
| [^2] | [Learning linear dynamical systems under convex constraints.](http://arxiv.org/abs/2303.15121) | 本文考虑在给定凸约束下学习线性动态系统，通过解出受约束的最小二乘估计，提出新的非渐进误差界，并应用于稀疏矩阵等情境，改进了现有统计方法。 |

# 详细

[^1]: 关于证据下界的Fisher-Rao梯度研究

    On the Fisher-Rao Gradient of the Evidence Lower Bound. (arXiv:2307.11249v1 [cs.LG])

    [http://arxiv.org/abs/2307.11249](http://arxiv.org/abs/2307.11249)

    本文研究了证据下界的Fisher-Rao梯度，揭示了它与目标分布的Kullback-Leibler散度梯度的关系，进一步证明了最小化主要目标函数与最大化ELBO的等价性。

    

    本文研究了证据下界（ELBO）的Fisher-Rao梯度，也称为自然梯度，它在变分自动编码器理论、Helmholtz机和自由能原理中起着关键作用。ELBO的自然梯度与目标分布的Kullback-Leibler散度的自然梯度相关，后者是学习的主要目标函数。基于信息几何中梯度的不变性特性，提供了关于底层模型的条件，确保最小化主要目标函数与最大化ELBO的等价性。

    This article studies the Fisher-Rao gradient, also referred to as the natural gradient, of the evidence lower bound, the ELBO, which plays a crucial role within the theory of the Variational Autonecoder, the Helmholtz Machine and the Free Energy Principle. The natural gradient of the ELBO is related to the natural gradient of the Kullback-Leibler divergence from a target distribution, the prime objective function of learning. Based on invariance properties of gradients within information geometry, conditions on the underlying model are provided that ensure the equivalence of minimising the prime objective function and the maximisation of the ELBO.
    
[^2]: 在凸约束下学习线性动态系统

    Learning linear dynamical systems under convex constraints. (arXiv:2303.15121v1 [math.ST])

    [http://arxiv.org/abs/2303.15121](http://arxiv.org/abs/2303.15121)

    本文考虑在给定凸约束下学习线性动态系统，通过解出受约束的最小二乘估计，提出新的非渐进误差界，并应用于稀疏矩阵等情境，改进了现有统计方法。

    

    我们考虑从单个轨迹中识别线性动态系统的问题。最近的研究主要关注未对系统矩阵 $A^* \in \mathbb{R}^{n \times n}$ 进行结构假设的情况，并对普通最小二乘 (OLS) 估计器进行了详细分析。我们假设可用先前的 $A^*$ 的结构信息，可以在包含 $A^*$ 的凸集 $\mathcal{K}$ 中捕获。对于随后的受约束最小二乘估计的解，我们推导出 Frobenius 范数下依赖于 $\mathcal{K}$ 在 $A^*$ 处切锥的局部大小的非渐进误差界。为了说明这一结果的有用性，我们将其实例化为以下设置：(i) $\mathcal{K}$ 是 $\mathbb{R}^{n \times n}$ 中的 $d$ 维子空间，或者 (ii) $A^*$ 是 $k$ 稀疏的，$\mathcal{K}$ 是适当缩放的 $\ell_1$ 球。在 $d, k \ll n^2$ 的区域中，我们的误差界对于相同的统计和噪声假设比 OLS 估计器获得了改进。

    We consider the problem of identification of linear dynamical systems from a single trajectory. Recent results have predominantly focused on the setup where no structural assumption is made on the system matrix $A^* \in \mathbb{R}^{n \times n}$, and have consequently analyzed the ordinary least squares (OLS) estimator in detail. We assume prior structural information on $A^*$ is available, which can be captured in the form of a convex set $\mathcal{K}$ containing $A^*$. For the solution of the ensuing constrained least squares estimator, we derive non-asymptotic error bounds in the Frobenius norm which depend on the local size of the tangent cone of $\mathcal{K}$ at $A^*$. To illustrate the usefulness of this result, we instantiate it for the settings where, (i) $\mathcal{K}$ is a $d$ dimensional subspace of $\mathbb{R}^{n \times n}$, or (ii) $A^*$ is $k$-sparse and $\mathcal{K}$ is a suitably scaled $\ell_1$ ball. In the regimes where $d, k \ll n^2$, our bounds improve upon those obta
    

