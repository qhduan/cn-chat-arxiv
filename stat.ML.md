# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Federated Transfer Learning with Differential Privacy](https://arxiv.org/abs/2403.11343) | 本文提出了具有差分隐私的联邦迁移学习框架，通过利用多个异构源数据集的信息来增强对目标数据集的学习，同时考虑隐私约束。 |
| [^2] | [Sparse Gaussian Graphical Models with Discrete Optimization: Computational and Statistical Perspectives.](http://arxiv.org/abs/2307.09366) | 本文提出了基于离散优化的稀疏高斯图模型学习问题的新方法，并提供了大规模求解器来获取良好的原始解。 |
| [^3] | [Importance Sparsification for Sinkhorn Algorithm.](http://arxiv.org/abs/2306.06581) | Spar-Sink是一种重要性稀疏化方法，能够有效近似熵正则化最优传输和不平衡最优传输问题，并且在实验中表现优异。 |
| [^4] | [Piecewise Deterministic Markov Processes for Bayesian Neural Networks.](http://arxiv.org/abs/2302.08724) | 本文介绍了基于分段确定性马尔可夫过程的贝叶斯神经网络推理方法，通过引入新的自适应稀疏方案，实现了对困难采样问题的加速处理。实验证明，这种方法在计算上可行，并能提高预测准确性、MCMC混合性能，并提供更有信息量的不确定性测量。 |

# 详细

[^1]: 具有差分隐私的联邦迁移学习

    Federated Transfer Learning with Differential Privacy

    [https://arxiv.org/abs/2403.11343](https://arxiv.org/abs/2403.11343)

    本文提出了具有差分隐私的联邦迁移学习框架，通过利用多个异构源数据集的信息来增强对目标数据集的学习，同时考虑隐私约束。

    

    联邦学习越来越受到欢迎，数据异构性和隐私性是两个突出的挑战。在本文中，我们在联邦迁移学习框架内解决了这两个问题，旨在通过利用来自多个异构源数据集的信息来增强对目标数据集的学习，同时遵守隐私约束。我们严格制定了\textit{联邦差分隐私}的概念，为每个数据集提供隐私保证，而无需假设有一个受信任的中央服务器。在这个隐私约束下，我们研究了三个经典的统计问题，即单变量均值估计、低维线性回归和高维线性回归。通过研究极小值率并确定这些问题的隐私成本，我们展示了联邦差分隐私是已建立的局部和中央模型之间的一种中间隐私模型。

    arXiv:2403.11343v1 Announce Type: new  Abstract: Federated learning is gaining increasing popularity, with data heterogeneity and privacy being two prominent challenges. In this paper, we address both issues within a federated transfer learning framework, aiming to enhance learning on a target data set by leveraging information from multiple heterogeneous source data sets while adhering to privacy constraints. We rigorously formulate the notion of \textit{federated differential privacy}, which offers privacy guarantees for each data set without assuming a trusted central server. Under this privacy constraint, we study three classical statistical problems, namely univariate mean estimation, low-dimensional linear regression, and high-dimensional linear regression. By investigating the minimax rates and identifying the costs of privacy for these problems, we show that federated differential privacy is an intermediate privacy model between the well-established local and central models of 
    
[^2]: 稀疏高斯图模型的离散优化：计算和统计角度

    Sparse Gaussian Graphical Models with Discrete Optimization: Computational and Statistical Perspectives. (arXiv:2307.09366v1 [cs.LG])

    [http://arxiv.org/abs/2307.09366](http://arxiv.org/abs/2307.09366)

    本文提出了基于离散优化的稀疏高斯图模型学习问题的新方法，并提供了大规模求解器来获取良好的原始解。

    

    我们考虑了学习基于无向高斯图模型的稀疏图的问题，这是统计机器学习中的一个关键问题。给定来自具有p个变量的多元高斯分布的n个样本，目标是估计p×p的逆协方差矩阵（也称为精度矩阵），假设它是稀疏的（即具有少数非零条目）。我们提出了GraphL0BnB这一新的估计方法，它基于伪似然函数的l0惩罚版本，而大多数早期方法都是基于l1松弛。我们的估计方法可以被形式化为一个凸混合整数规划（MIP），使用现成的商用求解器在大规模计算时可能很难计算。为了解决MIP问题，我们提出了一个定制的非线性分支定界（BnB）框架，用于使用定制的一阶方法来解决节点放松问题。作为我们BnB框架的副产品，我们提出了用于获得独立兴趣的良好原始解的大规模求解器。

    We consider the problem of learning a sparse graph underlying an undirected Gaussian graphical model, a key problem in statistical machine learning. Given $n$ samples from a multivariate Gaussian distribution with $p$ variables, the goal is to estimate the $p \times p$ inverse covariance matrix (aka precision matrix), assuming it is sparse (i.e., has a few nonzero entries). We propose GraphL0BnB, a new estimator based on an $\ell_0$-penalized version of the pseudolikelihood function, while most earlier approaches are based on the $\ell_1$-relaxation. Our estimator can be formulated as a convex mixed integer program (MIP) which can be difficult to compute at scale using off-the-shelf commercial solvers. To solve the MIP, we propose a custom nonlinear branch-and-bound (BnB) framework that solves node relaxations with tailored first-order methods. As a by-product of our BnB framework, we propose large-scale solvers for obtaining good primal solutions that are of independent interest. We d
    
[^3]: Sinkhorn算法的重要性稀疏化

    Importance Sparsification for Sinkhorn Algorithm. (arXiv:2306.06581v1 [stat.ML])

    [http://arxiv.org/abs/2306.06581](http://arxiv.org/abs/2306.06581)

    Spar-Sink是一种重要性稀疏化方法，能够有效近似熵正则化最优传输和不平衡最优传输问题，并且在实验中表现优异。

    

    Sinkhorn算法被广泛应用于近似求解最优传输（OT）和不平衡最优传输（UOT）问题。但由于高计算复杂度，其实际应用受到限制。为减轻计算负担，我们提出了一种新的重要性稀疏化方法Spar-Sink，用于高效近似熵正则化OT和UOT解。具体来说，我们的方法利用未知最优传输计划的自然上界确定有效的采样概率，并构建稀疏的核矩阵以加速Sinkhorn迭代，将每次迭代的计算成本从$ O（n ^ 2）$降低到$\widetilde {O（n）}$适用于样本大小为$ n $的情况。理论上，我们证明了对于温和正则性条件下，所提出的OT和UOT问题的估计量是一致的。在各种合成数据上的实验表明，在估计误差方面，Spar-Sink优于主流竞争对手。

    Sinkhorn algorithm has been used pervasively to approximate the solution to optimal transport (OT) and unbalanced optimal transport (UOT) problems. However, its practical application is limited due to the high computational complexity. To alleviate the computational burden, we propose a novel importance sparsification method, called Spar-Sink, to efficiently approximate entropy-regularized OT and UOT solutions. Specifically, our method employs natural upper bounds for unknown optimal transport plans to establish effective sampling probabilities, and constructs a sparse kernel matrix to accelerate Sinkhorn iterations, reducing the computational cost of each iteration from $O(n^2)$ to $\widetilde{O}(n)$ for a sample of size $n$. Theoretically, we show the proposed estimators for the regularized OT and UOT problems are consistent under mild regularity conditions. Experiments on various synthetic data demonstrate Spar-Sink outperforms mainstream competitors in terms of both estimation erro
    
[^4]: 基于分段确定性马尔可夫过程的贝叶斯神经网络研究

    Piecewise Deterministic Markov Processes for Bayesian Neural Networks. (arXiv:2302.08724v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.08724](http://arxiv.org/abs/2302.08724)

    本文介绍了基于分段确定性马尔可夫过程的贝叶斯神经网络推理方法，通过引入新的自适应稀疏方案，实现了对困难采样问题的加速处理。实验证明，这种方法在计算上可行，并能提高预测准确性、MCMC混合性能，并提供更有信息量的不确定性测量。

    

    现代贝叶斯神经网络（BNNs）的推理通常依赖于变分推断处理，这要求违反了独立性和后验形式的假设。传统的MCMC方法避免了这些假设，但由于无法适应似然的子采样，导致计算量增加。新的分段确定性马尔可夫过程（PDMP）采样器允许子采样，但引入了模型特定的不均匀泊松过程（IPPs），从中采样困难。本研究引入了一种新的通用自适应稀疏方案，用于从这些IPPs中进行采样，并展示了如何加速将PDMPs应用于BNNs推理。实验表明，使用这些方法进行推理在计算上是可行的，可以提高预测准确性、MCMC混合性能，并与其他近似推理方案相比，提供更有信息量的不确定性测量。

    Inference on modern Bayesian Neural Networks (BNNs) often relies on a variational inference treatment, imposing violated assumptions of independence and the form of the posterior. Traditional MCMC approaches avoid these assumptions at the cost of increased computation due to its incompatibility to subsampling of the likelihood. New Piecewise Deterministic Markov Process (PDMP) samplers permit subsampling, though introduce a model specific inhomogenous Poisson Process (IPPs) which is difficult to sample from. This work introduces a new generic and adaptive thinning scheme for sampling from these IPPs, and demonstrates how this approach can accelerate the application of PDMPs for inference in BNNs. Experimentation illustrates how inference with these methods is computationally feasible, can improve predictive accuracy, MCMC mixing performance, and provide informative uncertainty measurements when compared against other approximate inference schemes.
    

