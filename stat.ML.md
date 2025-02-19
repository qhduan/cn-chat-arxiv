# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [$G$-Mapper: Learning a Cover in the Mapper Construction.](http://arxiv.org/abs/2309.06634) | 本论文介绍了一种基于统计检验和聚类算法的优化Mapper图覆盖的方法，通过分割覆盖选择生成了保留数据集本质的Mapper图。 |
| [^2] | [repliclust: Synthetic Data for Cluster Analysis.](http://arxiv.org/abs/2303.14301) | repliclust 是一个 Python 包，用于生成具有聚类的合成数据集，基于数据集的原型，提供了放置集群中心、采样集群形状、选择每个集群的数据点数量以及为集群分配概率分布的算法。 |
| [^3] | [Statistical Inference of Constrained Stochastic Optimization via Sketched Sequential Quadratic Programming.](http://arxiv.org/abs/2205.13687) | 本篇论文提出了一种用于等式约束的随机非线性优化问题的统计推断方法，通过基于草图的顺序二次规划（StoSQP）进行求解，并且允许自适应选择随机步长和使用高效随机迭代求解器来降低计算成本。 |

# 详细

[^1]: $G$-Mapper：学习Mapper构造中的覆盖

    $G$-Mapper: Learning a Cover in the Mapper Construction. (arXiv:2309.06634v1 [cs.LG])

    [http://arxiv.org/abs/2309.06634](http://arxiv.org/abs/2309.06634)

    本论文介绍了一种基于统计检验和聚类算法的优化Mapper图覆盖的方法，通过分割覆盖选择生成了保留数据集本质的Mapper图。

    

    Mapper算法是拓扑数据分析(TDA)中一种反映给定数据集结构的可视化技术。Mapper算法需要调整多个参数以生成一个"好看的"Mapper图。该论文关注于选择覆盖参数。我们提出了一种通过根据正态性的统计检验反复分割覆盖来优化Mapper图的算法。我们的算法基于$G$-means聚类，通过迭代地进行Anderson-Darling检验来寻找$k$-means中最佳的簇数。我们的分割过程利用高斯混合模型，根据给定数据的分布精心选择覆盖。对于合成和真实数据集的实验表明，我们的算法生成的覆盖使Mapper图保留了数据集的本质。

    The Mapper algorithm is a visualization technique in topological data analysis (TDA) that outputs a graph reflecting the structure of a given dataset. The Mapper algorithm requires tuning several parameters in order to generate a "nice" Mapper graph. The paper focuses on selecting the cover parameter. We present an algorithm that optimizes the cover of a Mapper graph by splitting a cover repeatedly according to a statistical test for normality. Our algorithm is based on $G$-means clustering which searches for the optimal number of clusters in $k$-means by conducting iteratively the Anderson-Darling test. Our splitting procedure employs a Gaussian mixture model in order to choose carefully the cover based on the distribution of a given data. Experiments for synthetic and real-world datasets demonstrate that our algorithm generates covers so that the Mapper graphs retain the essence of the datasets.
    
[^2]: repliclust：聚类分析的合成数据

    repliclust: Synthetic Data for Cluster Analysis. (arXiv:2303.14301v1 [cs.LG])

    [http://arxiv.org/abs/2303.14301](http://arxiv.org/abs/2303.14301)

    repliclust 是一个 Python 包，用于生成具有聚类的合成数据集，基于数据集的原型，提供了放置集群中心、采样集群形状、选择每个集群的数据点数量以及为集群分配概率分布的算法。

    

    我们介绍了 repliclust（来自于 repli-cate 和 clust-er），这是一个用于生成具有聚类的合成数据集的 Python 包。我们的方法基于数据集的原型，即高级几何描述，用户可以从中创建许多不同的数据集，并具有所需的几何特性。我们软件的架构是模块化和面向对象的，将数据生成分解成放置集群中心的算法、采样集群形状的算法、选择每个集群的数据点数量的算法以及为集群分配概率分布的算法。repliclust.org 项目网页提供了简明的用户指南和全面的文档。

    We present repliclust (from repli-cate and clust-er), a Python package for generating synthetic data sets with clusters. Our approach is based on data set archetypes, high-level geometric descriptions from which the user can create many different data sets, each possessing the desired geometric characteristics. The architecture of our software is modular and object-oriented, decomposing data generation into algorithms for placing cluster centers, sampling cluster shapes, selecting the number of data points for each cluster, and assigning probability distributions to clusters. The project webpage, repliclust.org, provides a concise user guide and thorough documentation.
    
[^3]: 通过基于草图的顺序二次规划对约束的随机优化进行统计推断

    Statistical Inference of Constrained Stochastic Optimization via Sketched Sequential Quadratic Programming. (arXiv:2205.13687v3 [math.OC] UPDATED)

    [http://arxiv.org/abs/2205.13687](http://arxiv.org/abs/2205.13687)

    本篇论文提出了一种用于等式约束的随机非线性优化问题的统计推断方法，通过基于草图的顺序二次规划（StoSQP）进行求解，并且允许自适应选择随机步长和使用高效随机迭代求解器来降低计算成本。

    

    我们考虑对等式约束的随机非线性优化问题进行统计推断。我们开发了一种全在线随机顺序二次规划（StoSQP）方法来解决这些问题，可以将其视为将牛顿法应用于一阶最优性条件（即KKT条件）。受最近数值二阶方法设计的启发，我们允许StoSQP自适应地选择任意随机步长$ \bar {\ alpha} _t $，只要$ \ beta _t \ leq \ bar {\ alpha} _t \ leq \ beta _t + \ chi _t $，其中 $ \ beta_t $ 和 $ \ chi_t = o(\beta_t) $ 是某些控制序列。为了降低二阶方法的主要计算成本，我们还允许StoSQP通过使用草图技术的高效随机迭代求解器来不精确地解决二次规划问题。值得注意的是，我们不要求逼近误差随着迭代的进行而减小。对于开发的方法，我们证明在温和的假设（i）下，它的计算复杂度最多为$ O(1 / \ ep）$。

    We consider statistical inference of equality-constrained stochastic nonlinear optimization problems. We develop a fully online stochastic sequential quadratic programming (StoSQP) method to solve the problems, which can be regarded as applying Newton's method to the first-order optimality conditions (i.e., the KKT conditions). Motivated by recent designs of numerical second-order methods, we allow StoSQP to adaptively select any random stepsize $\bar{\alpha}_t$, as long as $\beta_t\leq \bar{\alpha}_t \leq \beta_t+\chi_t$, for some control sequences $\beta_t$ and $\chi_t=o(\beta_t)$. To reduce the dominant computational cost of second-order methods, we additionally allow StoSQP to inexactly solve quadratic programs via efficient randomized iterative solvers that utilize sketching techniques. Notably, we do not require the approximation error to diminish as iteration proceeds. For the developed method, we show that under mild assumptions (i) computationally, it can take at most $O(1/\ep
    

