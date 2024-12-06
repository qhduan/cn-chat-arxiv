# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Equity through Access: A Case for Small-scale Deep Learning](https://arxiv.org/abs/2403.12562) | 通过引入PePR分数，研究人员展示了在资源有限的情况下，利用131种独特的DL架构在医学图像任务中的可行性。 |
| [^2] | [An SDP-based Branch-and-Cut Algorithm for Biclustering](https://arxiv.org/abs/2403.11351) | 提出了一个基于SDP的分支定界算法，用于解决$k$-最密不相交双团问题。 |
| [^3] | [Reliable uncertainty with cheaper neural network ensembles: a case study in industrial parts classification](https://arxiv.org/abs/2403.10182) | 研究在工业零部件分类中探讨了利用更便宜的神经网络集成实现可靠的不确定性估计的方法 |
| [^4] | [Tuning the perplexity for and computing sampling-based t-SNE embeddings.](http://arxiv.org/abs/2308.15513) | 本文通过采样的方法改进了大数据集下t-SNE嵌入的质量和计算速度。 |
| [^5] | [Data-driven Piecewise Affine Decision Rules for Stochastic Programming with Covariate Information.](http://arxiv.org/abs/2304.13646) | 本研究提出一种嵌入非凸分段仿射决策规则的经验风险最小化方法，用于学习特征与最优决策之间的直接映射。所提出的方法可用于广泛的非凸型SP问题，并且在数值研究中表现出优越的性能。 |

# 详细

[^1]: 通过获取赋权：支持小规模深度学习的案例

    Equity through Access: A Case for Small-scale Deep Learning

    [https://arxiv.org/abs/2403.12562](https://arxiv.org/abs/2403.12562)

    通过引入PePR分数，研究人员展示了在资源有限的情况下，利用131种独特的DL架构在医学图像任务中的可行性。

    

    深度学习（DL）的最新进展得益于大规模数据和计算力的提升。这些大规模资源被用于训练日益庞大的模型，而这些模型在计算、数据、能源和碳排放方面消耗巨大。这些成本正在成为研究人员和从业者面临的新型准入障碍，特别是对于那些在全球南方地区资源有限的人。在这项工作中，我们全面审视了现有视觉任务的DL模型，并展示了它们在资源有限的环境中的实用性。为了考虑DL模型的资源消耗，我们引入了一个衡量性能与资源单元的新指标，我们称之为PePR分数。通过使用131种独特的DL架构（跨度从1M到130M个可训练参数）和三个医学图像数据集，我们获取了有关性能和资源之间关系的趋势。

    arXiv:2403.12562v1 Announce Type: cross  Abstract: The recent advances in deep learning (DL) have been accelerated by access to large-scale data and compute. These large-scale resources have been used to train progressively larger models which are resource intensive in terms of compute, data, energy, and carbon emissions. These costs are becoming a new type of entry barrier to researchers and practitioners with limited access to resources at such scale, particularly in the Global South. In this work, we take a comprehensive look at the landscape of existing DL models for vision tasks and demonstrate their usefulness in settings where resources are limited. To account for the resource consumption of DL models, we introduce a novel measure to estimate the performance per resource unit, which we call the PePR score. Using a diverse family of 131 unique DL architectures (spanning 1M to 130M trainable parameters) and three medical image datasets, we capture trends about the performance-reso
    
[^2]: 基于SDP的二分图聚类分支定界算法

    An SDP-based Branch-and-Cut Algorithm for Biclustering

    [https://arxiv.org/abs/2403.11351](https://arxiv.org/abs/2403.11351)

    提出了一个基于SDP的分支定界算法，用于解决$k$-最密不相交双团问题。

    

    二分图聚类，也称为共聚类、块聚类或双向聚类，涉及将数据矩阵的行和列同时聚类成不同的组，使得同一组内的行和列显示出相似的模式。作为二分图聚类的模型问题，我们考虑$k$-最密不相交双团问题，其目标是在给定加权完全二分图中识别 $k$ 个不相交的完全二部子图（称为双团），使它们的密度之和最大化。为了解决这个问题，我们提出了一个定制的分支定界算法。对于上界例程，我们考虑半定规划放松并提出了用于加强界限的有效不等式。我们使用一种一阶方法以切平面方式解决这个放松问题。对于下界，我们设计了一个利用解决方案的最大权匹配舍入过程。

    arXiv:2403.11351v1 Announce Type: cross  Abstract: Biclustering, also called co-clustering, block clustering, or two-way clustering, involves the simultaneous clustering of both the rows and columns of a data matrix into distinct groups, such that the rows and columns within a group display similar patterns. As a model problem for biclustering, we consider the $k$-densest-disjoint biclique problem, whose goal is to identify $k$ disjoint complete bipartite subgraphs (called bicliques) of a given weighted complete bipartite graph such that the sum of their densities is maximized. To address this problem, we present a tailored branch-and-cut algorithm. For the upper bound routine, we consider a semidefinite programming relaxation and propose valid inequalities to strengthen the bound. We solve this relaxation in a cutting-plane fashion using a first-order method. For the lower bound, we design a maximum weight matching rounding procedure that exploits the solution of the relaxation solved
    
[^3]: 用更便宜的神经网络集成实现可靠的不确定性：工业零部件分类案例研究

    Reliable uncertainty with cheaper neural network ensembles: a case study in industrial parts classification

    [https://arxiv.org/abs/2403.10182](https://arxiv.org/abs/2403.10182)

    研究在工业零部件分类中探讨了利用更便宜的神经网络集成实现可靠的不确定性估计的方法

    

    在运筹学(OR)中，预测模型经常会遇到数据分布与训练数据分布不同的场景。近年来，神经网络(NNs)在图像分类等领域的出色性能使其在OR中备受关注。然而，当面对OOD数据时，NNs往往会做出自信但不正确的预测。不确定性估计为自信的模型提供了一个解决方案，当输出应(不应)被信任时进行通信。因此，在OR领域中，NNs中的可靠不确定性量化至关重要。由多个独立NNs组成的深度集合已经成为一种有前景的方法，不仅提供强大的预测准确性，还能可靠地估计不确定性。然而，它们的部署由于较大的计算需求而具有挑战性。最近的基础研究提出了更高效的NN集成，即sna

    arXiv:2403.10182v1 Announce Type: new  Abstract: In operations research (OR), predictive models often encounter out-of-distribution (OOD) scenarios where the data distribution differs from the training data distribution. In recent years, neural networks (NNs) are gaining traction in OR for their exceptional performance in fields such as image classification. However, NNs tend to make confident yet incorrect predictions when confronted with OOD data. Uncertainty estimation offers a solution to overconfident models, communicating when the output should (not) be trusted. Hence, reliable uncertainty quantification in NNs is crucial in the OR domain. Deep ensembles, composed of multiple independent NNs, have emerged as a promising approach, offering not only strong predictive accuracy but also reliable uncertainty estimation. However, their deployment is challenging due to substantial computational demands. Recent fundamental research has proposed more efficient NN ensembles, namely the sna
    
[^4]: 调整困惑度并计算基于采样的t-SNE嵌入

    Tuning the perplexity for and computing sampling-based t-SNE embeddings. (arXiv:2308.15513v1 [cs.LG])

    [http://arxiv.org/abs/2308.15513](http://arxiv.org/abs/2308.15513)

    本文通过采样的方法改进了大数据集下t-SNE嵌入的质量和计算速度。

    

    高维数据分析常用的管道利用二维可视化，例如通过t分布邻近随机嵌入（t-SNE）。但在处理大数据集时，应用这些可视化技术会生成次优的嵌入，因为超参数不适用于大数据。将这些参数增加通常不起作用，因为计算对于实际工作流程来说太昂贵。本文中，我们认为基于采样的嵌入方法可以解决这些问题。我们展示了必须谨慎选择超参数，取决于采样率和预期的最终嵌入。此外，我们展示了该方法如何加速计算并提高嵌入的质量。

    Widely used pipelines for the analysis of high-dimensional data utilize two-dimensional visualizations. These are created, e.g., via t-distributed stochastic neighbor embedding (t-SNE). When it comes to large data sets, applying these visualization techniques creates suboptimal embeddings, as the hyperparameters are not suitable for large data. Cranking up these parameters usually does not work as the computations become too expensive for practical workflows. In this paper, we argue that a sampling-based embedding approach can circumvent these problems. We show that hyperparameters must be chosen carefully, depending on the sampling rate and the intended final embedding. Further, we show how this approach speeds up the computation and increases the quality of the embeddings.
    
[^5]: 基于数据驱动的分段仿射决策规则用于带协变信息的随机规划

    Data-driven Piecewise Affine Decision Rules for Stochastic Programming with Covariate Information. (arXiv:2304.13646v1 [math.OC])

    [http://arxiv.org/abs/2304.13646](http://arxiv.org/abs/2304.13646)

    本研究提出一种嵌入非凸分段仿射决策规则的经验风险最小化方法，用于学习特征与最优决策之间的直接映射。所提出的方法可用于广泛的非凸型SP问题，并且在数值研究中表现出优越的性能。

    

    本文针对带协变信息的随机规划，提出了一种嵌入非凸分段仿射决策规则(PADR)的经验风险最小化(ERM)方法，旨在学习特征与最优决策之间的直接映射。我们建立了基于PADR的ERM模型的非渐近一致性结果，可用于无约束问题，以及约束问题的渐近一致性结果。为了解决非凸和非可微的ERM问题，我们开发了一个增强的随机主导下降算法，并建立了沿（复合强）方向稳定性的渐近收敛以及复杂性分析。我们表明，所提出的PADR-based ERM方法适用于广泛的非凸型SP问题，并具有理论一致性保证和计算可处理性。数值研究表明，在各种设置下，PADR-based ERM方法相对于最先进的方法具有优越的性能。

    Focusing on stochastic programming (SP) with covariate information, this paper proposes an empirical risk minimization (ERM) method embedded within a nonconvex piecewise affine decision rule (PADR), which aims to learn the direct mapping from features to optimal decisions. We establish the nonasymptotic consistency result of our PADR-based ERM model for unconstrained problems and asymptotic consistency result for constrained ones. To solve the nonconvex and nondifferentiable ERM problem, we develop an enhanced stochastic majorization-minimization algorithm and establish the asymptotic convergence to (composite strong) directional stationarity along with complexity analysis. We show that the proposed PADR-based ERM method applies to a broad class of nonconvex SP problems with theoretical consistency guarantees and computational tractability. Our numerical study demonstrates the superior performance of PADR-based ERM methods compared to state-of-the-art approaches under various settings,
    

