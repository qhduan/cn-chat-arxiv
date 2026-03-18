# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Exact and efficient solutions of the LMC Multitask Gaussian Process model.](http://arxiv.org/abs/2310.12032) | LMC多任务高斯过程模型的精确解决方案表明，只需对噪声模型进行温和假设，即可实现高效计算。通过引入完整参数化的“投影LMC”模型和边缘似然函数表达式，展示了该方法相对于未经处理的方法的优异性能。 |
| [^2] | [From structure mining to unsupervised exploration of atomic octahedral networks.](http://arxiv.org/abs/2306.12272) | 该论文提出了一种自动化的化学直觉算法，可以对配位八面体网络进行几何解析、量化和分类，并利用无监督机器学习对无机框架聚型进行分类。作者发现了ABO$_{3}$钙钛矿多形体系中的轴向倾斜趋势，并揭示了Pauling的第三条规则违反和设计原则的更新。 |
| [^3] | [UNREAL:Unlabeled Nodes Retrieval and Labeling for Heavily-imbalanced Node Classification.](http://arxiv.org/abs/2303.10371) | 本文提出了一种用于重度不平衡节点分类的迭代过采样方法UNREAL，通过添加未标记节点而不是合成节点，解决了特征和邻域生成的难题，并利用节点嵌入空间中的无监督学习进行几何排名来有效地校准伪标签分配。 |

# 详细

[^1]: LMC多任务高斯过程模型的精确和高效解决方案

    Exact and efficient solutions of the LMC Multitask Gaussian Process model. (arXiv:2310.12032v1 [cs.LG])

    [http://arxiv.org/abs/2310.12032](http://arxiv.org/abs/2310.12032)

    LMC多任务高斯过程模型的精确解决方案表明，只需对噪声模型进行温和假设，即可实现高效计算。通过引入完整参数化的“投影LMC”模型和边缘似然函数表达式，展示了该方法相对于未经处理的方法的优异性能。

    

    线性共同关联模型（LMC）是一种非常通用的多任务高斯过程模型，用于回归或分类。虽然其表达能力和概念简单性很有吸引力，但朴素实现在数据点数量和任务数量方面具有立方复杂度，使得对大多数应用来说，必须进行近似处理。然而，最近的研究表明，在某些条件下，该模型的潜在过程可以解耦，导致仅与所述过程数量呈线性复杂度。我们在这里扩展了这些结果，从最一般的假设中展示了在LMC的高效精确计算所需的唯一条件是对噪声模型进行温和假设。我们引入了结果的完整参数化“投影LMC”模型，并给出了边缘似然函数的表达式，以实现高效的优化。我们对合成数据进行了参数研究，展示了我们方法相对于未经处理的方法的优异性能。

    The Linear Model of Co-regionalization (LMC) is a very general model of multitask gaussian process for regression or classification. While its expressivity and conceptual simplicity are appealing, naive implementations have cubic complexity in the number of datapoints and number of tasks, making approximations mandatory for most applications. However, recent work has shown that under some conditions the latent processes of the model can be decoupled, leading to a complexity that is only linear in the number of said processes. We here extend these results, showing from the most general assumptions that the only condition necessary to an efficient exact computation of the LMC is a mild hypothesis on the noise model. We introduce a full parametrization of the resulting \emph{projected LMC} model, and an expression of the marginal likelihood enabling efficient optimization. We perform a parametric study on synthetic data to show the excellent performance of our approach, compared to an unr
    
[^2]: 从结构挖掘到无监督探索八面体原子网络

    From structure mining to unsupervised exploration of atomic octahedral networks. (arXiv:2306.12272v1 [cond-mat.mtrl-sci])

    [http://arxiv.org/abs/2306.12272](http://arxiv.org/abs/2306.12272)

    该论文提出了一种自动化的化学直觉算法，可以对配位八面体网络进行几何解析、量化和分类，并利用无监督机器学习对无机框架聚型进行分类。作者发现了ABO$_{3}$钙钛矿多形体系中的轴向倾斜趋势，并揭示了Pauling的第三条规则违反和设计原则的更新。

    

    原子中心的配位八面体网络通常出现在无机和混合固态材料中。表征它们的空间排列和特性对于许多材料系列将结构与性质联系起来至关重要。传统的逐案例检查方法在发现大型数据集中的趋势和相似之处方面变得不可行。在这里，我们运用化学直觉自动化地进行几何解析、量化和分类配位八面体网络。我们在ABO$_{3}$钙钛矿多形体系中发现了轴向倾斜趋势，有助于检测氧化态变化。此外，我们开发了一个尺度不变的编码方案来表示这些网络，并与人类辅助的无监督机器学习相结合，可以对混合碘铅酸盐(A$_x$Pb$_y$I$_z$)中的无机框架聚型进行分类。因此，我们揭示了Pauling的第三条规则违反和设计原则ereotype0的更新。

    Networks of atom-centered coordination octahedra commonly occur in inorganic and hybrid solid-state materials. Characterizing their spatial arrangements and characteristics is crucial for relating structures to properties for many materials families. The traditional method using case-by-case inspection becomes prohibitive for discovering trends and similarities in large datasets. Here, we operationalize chemical intuition to automate the geometric parsing, quantification, and classification of coordination octahedral networks. We find axis-resolved tilting trends in ABO$_{3}$ perovskite polymorphs, which assist in detecting oxidation state changes. Moreover, we develop a scale-invariant encoding scheme to represent these networks, which, combined with human-assisted unsupervised machine learning, allows us to taxonomize the inorganic framework polytypes in hybrid iodoplumbates (A$_x$Pb$_y$I$_z$). Consequently, we uncover a violation of Pauling's third rule and the design principles und
    
[^3]: UNREAL: 用于重度不平衡节点分类的未标记节点检索和标记方法

    UNREAL:Unlabeled Nodes Retrieval and Labeling for Heavily-imbalanced Node Classification. (arXiv:2303.10371v1 [cs.LG])

    [http://arxiv.org/abs/2303.10371](http://arxiv.org/abs/2303.10371)

    本文提出了一种用于重度不平衡节点分类的迭代过采样方法UNREAL，通过添加未标记节点而不是合成节点，解决了特征和邻域生成的难题，并利用节点嵌入空间中的无监督学习进行几何排名来有效地校准伪标签分配。

    

    在现实世界的节点分类任务中，极度倾斜的标签分布很常见。如果不合适地处理，这对少数类别的GNNs性能会有极大的影响。由于其实用性，最近一系列的研究都致力于解决这个难题。现有的过采样方法通过产生“假”的少数节点和合成其特征和局部拓扑来平滑标签分布，这在很大程度上忽略了图上未标记节点的丰富信息。在本文中，我们提出了UNREAL，一种迭代过采样方法。第一个关键区别在于，我们只添加未标记节点而不是合成节点，这消除了特征和邻域生成的挑战。为了选择要添加的未标记节点，我们提出了几何排名来对未标记节点进行排名。几何排名利用节点嵌入空间中的无监督学习来有效地校准伪标签分配。最后，我们确定了问题。

    Extremely skewed label distributions are common in real-world node classification tasks. If not dealt with appropriately, it significantly hurts the performance of GNNs in minority classes. Due to its practical importance, there have been a series of recent research devoted to this challenge. Existing over-sampling techniques smooth the label distribution by generating ``fake'' minority nodes and synthesizing their features and local topology, which largely ignore the rich information of unlabeled nodes on graphs. In this paper, we propose UNREAL, an iterative over-sampling method. The first key difference is that we only add unlabeled nodes instead of synthetic nodes, which eliminates the challenge of feature and neighborhood generation. To select which unlabeled nodes to add, we propose geometric ranking to rank unlabeled nodes. Geometric ranking exploits unsupervised learning in the node embedding space to effectively calibrates pseudo-label assignment. Finally, we identify the issu
    

