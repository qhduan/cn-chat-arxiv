# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On diffusion models for amortized inference: Benchmarking and improving stochastic control and sampling](https://arxiv.org/abs/2402.05098) | 本研究探讨了训练扩散模型以从给定分布中采样的问题，并针对随机控制和采样提出了一种新的探索策略，通过基准测试比较了不同推断方法的相对优劣，并对过去的工作提出了质疑。 |
| [^2] | [A Random Matrix Approach to Low-Multilinear-Rank Tensor Approximation](https://arxiv.org/abs/2402.03169) | 该研究采用随机矩阵方法，在低多线性秩张量逼近中展示了对种植的低秩信号的估计，并根据大维谱行为和信噪比准确预测了重建性能，并给出了HOOI收敛的充分条件。 |
| [^3] | [Inhomogeneous graph trend filtering via a l2,0 cardinality penalty.](http://arxiv.org/abs/2304.05223) | 本文提出了一种基于L2，0基数惩罚的图趋势过滤（GTF）模型，可同时进行k-means聚类和基于图的最小割，以估计在节点之间具有不均匀平滑水平的分段平滑图信号，并在降噪、支持恢复和半监督分类任务上表现更好，比现有方法更高效地处理大型数据集。 |

# 详细

[^1]: 关于分散推断模型的扩散模型：基准测试和改进随机控制和采样

    On diffusion models for amortized inference: Benchmarking and improving stochastic control and sampling

    [https://arxiv.org/abs/2402.05098](https://arxiv.org/abs/2402.05098)

    本研究探讨了训练扩散模型以从给定分布中采样的问题，并针对随机控制和采样提出了一种新的探索策略，通过基准测试比较了不同推断方法的相对优劣，并对过去的工作提出了质疑。

    

    我们研究了训练扩散模型以从给定的非标准化密度或能量函数分布中采样的问题。我们对几种扩散结构推断方法进行了基准测试，包括基于模拟的变分方法和离策略方法（连续生成流网络）。我们的结果揭示了现有算法的相对优势，同时对过去的研究提出了一些质疑。我们还提出了一种新颖的离策略方法探索策略，基于目标空间中的局部搜索和回放缓冲区的使用，并证明它可以改善各种目标分布上的样本质量。我们研究的采样方法和基准测试的代码已公开在https://github.com/GFNOrg/gfn-diffusion，作为未来在分散推断模型上工作的基础。

    We study the problem of training diffusion models to sample from a distribution with a given unnormalized density or energy function. We benchmark several diffusion-structured inference methods, including simulation-based variational approaches and off-policy methods (continuous generative flow networks). Our results shed light on the relative advantages of existing algorithms while bringing into question some claims from past work. We also propose a novel exploration strategy for off-policy methods, based on local search in the target space with the use of a replay buffer, and show that it improves the quality of samples on a variety of target distributions. Our code for the sampling methods and benchmarks studied is made public at https://github.com/GFNOrg/gfn-diffusion as a base for future work on diffusion models for amortized inference.
    
[^2]: 低多线性秩张量逼近的随机矩阵方法

    A Random Matrix Approach to Low-Multilinear-Rank Tensor Approximation

    [https://arxiv.org/abs/2402.03169](https://arxiv.org/abs/2402.03169)

    该研究采用随机矩阵方法，在低多线性秩张量逼近中展示了对种植的低秩信号的估计，并根据大维谱行为和信噪比准确预测了重建性能，并给出了HOOI收敛的充分条件。

    

    本研究从计算阈值附近的一般尖峰张量模型，对种植的低秩信号估计进行了全面的认识。依靠大型随机矩阵理论的标准工具，我们表征了数据张量的展开的大维谱行为，并展示了决定主要信号方向可检测性的相关信噪比。这些结果可以准确地预测在非平凡区域的截断多线性奇异值分解(MLSVD)的重建性能。这一点尤其重要，因为它作为更高阶正交迭代(HOOI)方案的初始化，其收敛到最佳低多线性秩逼近完全取决于其初始化。我们给出了HOOI收敛的充分条件，并证明在大维极限下收敛前的迭代次数趋于1。

    This work presents a comprehensive understanding of the estimation of a planted low-rank signal from a general spiked tensor model near the computational threshold. Relying on standard tools from the theory of large random matrices, we characterize the large-dimensional spectral behavior of the unfoldings of the data tensor and exhibit relevant signal-to-noise ratios governing the detectability of the principal directions of the signal. These results allow to accurately predict the reconstruction performance of truncated multilinear SVD (MLSVD) in the non-trivial regime. This is particularly important since it serves as an initialization of the higher-order orthogonal iteration (HOOI) scheme, whose convergence to the best low-multilinear-rank approximation depends entirely on its initialization. We give a sufficient condition for the convergence of HOOI and show that the number of iterations before convergence tends to $1$ in the large-dimensional limit.
    
[^3]: 基于L2，0基数惩罚的不均匀图趋势过滤。

    Inhomogeneous graph trend filtering via a l2,0 cardinality penalty. (arXiv:2304.05223v1 [cs.LG])

    [http://arxiv.org/abs/2304.05223](http://arxiv.org/abs/2304.05223)

    本文提出了一种基于L2，0基数惩罚的图趋势过滤（GTF）模型，可同时进行k-means聚类和基于图的最小割，以估计在节点之间具有不均匀平滑水平的分段平滑图信号，并在降噪、支持恢复和半监督分类任务上表现更好，比现有方法更高效地处理大型数据集。

    

    我们研究了在图上估计分段平滑信号的方法，并提出了一种$\ell_{2,0}$-范数惩罚图趋势过滤（GTF）模型，以估计在节点之间具有不均匀平滑水平的分段平滑图信号。我们证明了所提出的GTF模型同时是基于节点上的信号的k-means聚类和基于图的最小割，其中聚类和割共享相同的分配矩阵。我们提出了两种方法来解决所提出的GTF模型：一种是基于谱分解的方法，另一种是基于模拟退火的方法。在合成和现实数据集的实验中，我们展示了所提出的GTF模型在降噪、支持恢复和半监督分类任务上表现更好，且比现有方法更高效地解决了大型数据集的问题。

    We study estimation of piecewise smooth signals over a graph. We propose a $\ell_{2,0}$-norm penalized Graph Trend Filtering (GTF) model to estimate piecewise smooth graph signals that exhibits inhomogeneous levels of smoothness across the nodes. We prove that the proposed GTF model is simultaneously a k-means clustering on the signal over the nodes and a minimum graph cut on the edges of the graph, where the clustering and the cut share the same assignment matrix. We propose two methods to solve the proposed GTF model: a spectral decomposition method and a method based on simulated annealing. In the experiment on synthetic and real-world datasets, we show that the proposed GTF model has a better performances compared with existing approaches on the tasks of denoising, support recovery and semi-supervised classification. We also show that the proposed GTF model can be solved more efficiently than existing models for the dataset with a large edge set.
    

