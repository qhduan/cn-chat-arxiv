# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Spectral-based Graph Neutral Networks for Complementary Item Recommendation.](http://arxiv.org/abs/2401.02130) | 本文提出了一种基于频谱的图神经网络方法（SComGNN）用于模拟和理解商品间的互补关系，以在推荐系统中准确和及时地推荐后续商品。 |

# 详细

[^1]: 基于频谱的图神经网络用于互补商品推荐

    Spectral-based Graph Neutral Networks for Complementary Item Recommendation. (arXiv:2401.02130v1 [cs.IR])

    [http://arxiv.org/abs/2401.02130](http://arxiv.org/abs/2401.02130)

    本文提出了一种基于频谱的图神经网络方法（SComGNN）用于模拟和理解商品间的互补关系，以在推荐系统中准确和及时地推荐后续商品。

    

    模拟互补关系极大地帮助推荐系统在购买一个商品后准确和及时地推荐后续的商品。与传统的相似关系不同，具有互补关系的商品可能会连续购买（例如iPhone和AirPods Pro），它们不仅共享相关性，还展现出不相似性。由于这两个属性是相反的，建模互补关系具有挑战性。先前尝试利用这些关系的方法要么忽视了或过度简化了不相似性属性，导致建模无效并且无法平衡这两个属性。由于图神经网络（GNNs）可以在频谱域中捕捉节点之间的相关性和不相似性，我们可以利用基于频谱的GNNs有效地理解和建模互补关系。在本研究中，我们提出了一种新方法，称为基于频谱的互补图神经网络（SComGNN），利用这一方法可以比较好地利用互补关系。

    Modeling complementary relationships greatly helps recommender systems to accurately and promptly recommend the subsequent items when one item is purchased. Unlike traditional similar relationships, items with complementary relationships may be purchased successively (such as iPhone and Airpods Pro), and they not only share relevance but also exhibit dissimilarity. Since the two attributes are opposites, modeling complementary relationships is challenging. Previous attempts to exploit these relationships have either ignored or oversimplified the dissimilarity attribute, resulting in ineffective modeling and an inability to balance the two attributes. Since Graph Neural Networks (GNNs) can capture the relevance and dissimilarity between nodes in the spectral domain, we can leverage spectral-based GNNs to effectively understand and model complementary relationships. In this study, we present a novel approach called Spectral-based Complementary Graph Neural Networks (SComGNN) that utilize
    

