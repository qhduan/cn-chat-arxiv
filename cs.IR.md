# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Continuous Input Embedding Size Search For Recommender Systems.](http://arxiv.org/abs/2304.03501) | 提出了一种新的方法CONTINUOUS，可以对潜在因子模型进行连续嵌入大小搜索，它通过将嵌入大小选择建模为连续变量解决了先前工作中的挑战，并在三个基准数据集上的实验中证实了它的有效性和高效性。 |

# 详细

[^1]: 推荐系统的连续输入嵌入大小搜索

    Continuous Input Embedding Size Search For Recommender Systems. (arXiv:2304.03501v1 [cs.IR])

    [http://arxiv.org/abs/2304.03501](http://arxiv.org/abs/2304.03501)

    提出了一种新的方法CONTINUOUS，可以对潜在因子模型进行连续嵌入大小搜索，它通过将嵌入大小选择建模为连续变量解决了先前工作中的挑战，并在三个基准数据集上的实验中证实了它的有效性和高效性。

    

    潜在因子模型是现今推荐系统最流行的基础，其性能卓越。潜在因子模型通过对用户和项目进行表示，用于对成对相似度的计算。所有嵌入向量传统上都被限制在一个相对较大的统一大小（例如256维）。随着当代电子商务中用户和项目目录指数级增长，这种设计显然变得效率低下。为了促进轻量级推荐，强化学习（RL）最近开辟了一些机会，用于识别不同用户/项目的不同嵌入大小。然而，受到搜索效率和学习最优RL策略的限制，现有的基于RL的方法被限制为高度离散的预定义嵌入大小选项。这导致了一个被广泛忽视的潜力，可以在给定计算预算下引入更细的粒度来获得更好的推荐效果。在本文中，我们提出了一种新方法，称为CONTINUOUS，可以对潜在因子模型进行连续嵌入大小搜索。CONTINUOUS通过将嵌入大小选择建模为连续变量和制定可微优化问题的形式来解决之前工作的挑战。在三个基准数据集上的实验证实了CONTINUOUS优于基线的优越性，验证了动态优化嵌入大小的有效性和高效性。

    Latent factor models are the most popular backbones for today's recommender systems owing to their prominent performance. Latent factor models represent users and items as real-valued embedding vectors for pairwise similarity computation, and all embeddings are traditionally restricted to a uniform size that is relatively large (e.g., 256-dimensional). With the exponentially expanding user base and item catalog in contemporary e-commerce, this design is admittedly becoming memory-inefficient. To facilitate lightweight recommendation, reinforcement learning (RL) has recently opened up opportunities for identifying varying embedding sizes for different users/items. However, challenged by search efficiency and learning an optimal RL policy, existing RL-based methods are restricted to highly discrete, predefined embedding size choices. This leads to a largely overlooked potential of introducing finer granularity into embedding sizes to obtain better recommendation effectiveness under a giv
    

