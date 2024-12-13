# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [HGCC: Enhancing Hyperbolic Graph Convolution Networks on Heterogeneous Collaborative Graph for Recommendation.](http://arxiv.org/abs/2304.02961) | 该论文提出了一种名为HGCC的协同过滤模型，它通过添加幂律偏差来保持协同图的长尾性质，并直接聚合邻居节点，以提高推荐性能。 |

# 详细

[^1]: HGCC：提高异构协同图上的超几何图卷积网络用于推荐

    HGCC: Enhancing Hyperbolic Graph Convolution Networks on Heterogeneous Collaborative Graph for Recommendation. (arXiv:2304.02961v1 [cs.IR])

    [http://arxiv.org/abs/2304.02961](http://arxiv.org/abs/2304.02961)

    该论文提出了一种名为HGCC的协同过滤模型，它通过添加幂律偏差来保持协同图的长尾性质，并直接聚合邻居节点，以提高推荐性能。

    

    由于推荐任务中用户-物品交互数据的自然幂律分布特性，超几何空间建模已被引入协同过滤方法中。其中，超几何GCN结合了GCN和超几何空间的优势，并取得了令人惊讶的性能。然而，这些方法仅在设计中部分利用了超几何空间的特性，由于完全随机的嵌入初始化和不精确的切线空间聚合。此外，这些工作中使用的数据主要集中在仅用户-物品交互数据中，这进一步限制了模型的性能。本文提出了一种超几何GCN协同过滤模型HGCC，它改进了现有的超几何GCN结构，用于协同过滤并纳入了附加信息。它通过在节点嵌入初始化时添加幂律偏差来保持协同图的长尾性质；然后，它直接聚合邻居节点，以提高推荐性能。

    Due to the naturally power-law distributed nature of user-item interaction data in recommendation tasks, hyperbolic space modeling has recently been introduced into collaborative filtering methods. Among them, hyperbolic GCN combines the advantages of GCN and hyperbolic space and achieves a surprising performance. However, these methods only partially exploit the nature of hyperbolic space in their designs due to completely random embedding initialization and an inaccurate tangent space aggregation. In addition, the data used in these works mainly focus on user-item interaction data only, which further limits the performance of the models. In this paper, we propose a hyperbolic GCN collaborative filtering model, HGCC, which improves the existing hyperbolic GCN structure for collaborative filtering and incorporates side information. It keeps the long-tailed nature of the collaborative graph by adding power law prior to node embedding initialization; then, it aggregates neighbors directl
    

