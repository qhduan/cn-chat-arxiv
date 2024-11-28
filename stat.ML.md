# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Linear quadratic control of nonlinear systems with Koopman operator learning and the Nystr\"om method](https://arxiv.org/abs/2403.02811) | 本文将Koopman算子框架与核方法相结合，通过Nyström逼近实现了对非线性动态系统的有效控制，其理论贡献在于推导出Nyström逼近效果的理论保证。 |
| [^2] | [Multiscale Hodge Scattering Networks for Data Analysis](https://arxiv.org/abs/2311.10270) | 提出了多尺度霍奇散射网络（MHSNs），利用多尺度基础词典和卷积结构，生成对节点排列不变的特征。 |
| [^3] | [Simplifying GNN Performance with Low Rank Kernel Models.](http://arxiv.org/abs/2310.05250) | 本文提出了一种用于简化GNN性能的低秩内核模型，通过应用传统的非参数估计方法在谱域中取代过于复杂的GNN架构，并在多个图类型的半监督节点分类基准测试中取得了最先进的性能。 |

# 详细

[^1]: 具有Koopman算子学习和Nyström方法的非线性系统的线性二次控制

    Linear quadratic control of nonlinear systems with Koopman operator learning and the Nystr\"om method

    [https://arxiv.org/abs/2403.02811](https://arxiv.org/abs/2403.02811)

    本文将Koopman算子框架与核方法相结合，通过Nyström逼近实现了对非线性动态系统的有效控制，其理论贡献在于推导出Nyström逼近效果的理论保证。

    

    在本文中，我们研究了Koopman算子框架如何与核方法相结合以有效控制非线性动力系统。虽然核方法通常具有很大的计算需求，但我们展示了随机子空间（Nyström逼近）如何实现巨大的计算节约，同时保持精度。我们的主要技术贡献在于推导出关于Nyström逼近效果的理论保证。更具体地说，我们研究了线性二次调节器问题，证明了对于最优控制问题的相关解的近似Riccati算子和调节器目标都以$ m^{-1/2} $的速率收敛，其中$ m $是随机子空间大小。理论发现得到了数值实验的支持。

    arXiv:2403.02811v1 Announce Type: cross  Abstract: In this paper, we study how the Koopman operator framework can be combined with kernel methods to effectively control nonlinear dynamical systems. While kernel methods have typically large computational requirements, we show how random subspaces (Nystr\"om approximation) can be used to achieve huge computational savings while preserving accuracy. Our main technical contribution is deriving theoretical guarantees on the effect of the Nystr\"om approximation. More precisely, we study the linear quadratic regulator problem, showing that both the approximated Riccati operator and the regulator objective, for the associated solution of the optimal control problem, converge at the rate $m^{-1/2}$, where $m$ is the random subspace size. Theoretical findings are complemented by numerical experiments corroborating our results.
    
[^2]: 用于数据分析的多尺度霍奇散射网络

    Multiscale Hodge Scattering Networks for Data Analysis

    [https://arxiv.org/abs/2311.10270](https://arxiv.org/abs/2311.10270)

    提出了多尺度霍奇散射网络（MHSNs），利用多尺度基础词典和卷积结构，生成对节点排列不变的特征。

    

    我们提出了一种新的散射网络，用于在单纯复合仿射上测量的信号，称为\emph{多尺度霍奇散射网络}（MHSNs）。我们的构造基于单纯复合仿射上的多尺度基础词典，即$\kappa$-GHWT和$\kappa$-HGLET，我们最近为给定单纯复合仿射中的维度$\kappa \in \mathbb{N}$推广了基于节点的广义哈-沃什变换（GHWT）和分层图拉普拉斯特征变换（HGLET）。$\kappa$-GHWT和$\kappa$-HGLET都形成冗余集合（即词典）的多尺度基础向量和给定信号的相应扩展系数。我们的MHSNs使用类似于卷积神经网络（CNN）的分层结构来级联词典系数模的矩。所得特征对单纯复合仿射的重新排序不变（即节点排列的置换

    arXiv:2311.10270v2 Announce Type: replace  Abstract: We propose new scattering networks for signals measured on simplicial complexes, which we call \emph{Multiscale Hodge Scattering Networks} (MHSNs). Our construction is based on multiscale basis dictionaries on simplicial complexes, i.e., the $\kappa$-GHWT and $\kappa$-HGLET, which we recently developed for simplices of dimension $\kappa \in \mathbb{N}$ in a given simplicial complex by generalizing the node-based Generalized Haar-Walsh Transform (GHWT) and Hierarchical Graph Laplacian Eigen Transform (HGLET). The $\kappa$-GHWT and the $\kappa$-HGLET both form redundant sets (i.e., dictionaries) of multiscale basis vectors and the corresponding expansion coefficients of a given signal. Our MHSNs use a layered structure analogous to a convolutional neural network (CNN) to cascade the moments of the modulus of the dictionary coefficients. The resulting features are invariant to reordering of the simplices (i.e., node permutation of the u
    
[^3]: 用低秩内核模型简化GNN性能

    Simplifying GNN Performance with Low Rank Kernel Models. (arXiv:2310.05250v1 [cs.LG])

    [http://arxiv.org/abs/2310.05250](http://arxiv.org/abs/2310.05250)

    本文提出了一种用于简化GNN性能的低秩内核模型，通过应用传统的非参数估计方法在谱域中取代过于复杂的GNN架构，并在多个图类型的半监督节点分类基准测试中取得了最先进的性能。

    

    我们重新审视了最近的谱GNN方法对半监督节点分类（SSNC）的应用。我们认为许多当前的GNN架构可能过于精细设计。相反，简单的非参数估计传统方法，在谱域中应用，可以取代许多受深度学习启发的GNN设计。这些传统技术似乎非常适合各种图类型，在许多常见的SSNC基准测试中达到了最先进的性能。此外，我们还展示了最近在GNN方法方面的性能改进可能部分归因于评估惯例的变化。最后，我们对与GNN谱过滤技术相关的各种超参数进行了消融研究。

    We revisit recent spectral GNN approaches to semi-supervised node classification (SSNC). We posit that many of the current GNN architectures may be over-engineered. Instead, simpler, traditional methods from nonparametric estimation, applied in the spectral domain, could replace many deep-learning inspired GNN designs. These conventional techniques appear to be well suited for a variety of graph types reaching state-of-the-art performance on many of the common SSNC benchmarks. Additionally, we show that recent performance improvements in GNN approaches may be partially attributed to shifts in evaluation conventions. Lastly, an ablative study is conducted on the various hyperparameters associated with GNN spectral filtering techniques. Code available at: https://github.com/lucianoAvinas/lowrank-gnn-kernels
    

