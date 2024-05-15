# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [One Backpropagation in Two Tower Recommendation Models](https://arxiv.org/abs/2403.18227) | 该论文提出了一种在两塔推荐模型中使用单次反向传播更新策略的方法，挑战了现有算法中平等对待用户和物品的假设。 |

# 详细

[^1]: 两塔推荐模型中的单次反向传播

    One Backpropagation in Two Tower Recommendation Models

    [https://arxiv.org/abs/2403.18227](https://arxiv.org/abs/2403.18227)

    该论文提出了一种在两塔推荐模型中使用单次反向传播更新策略的方法，挑战了现有算法中平等对待用户和物品的假设。

    

    最近几年，已经看到为了减轻信息过载而开发两塔推荐模型的广泛研究。这种模型中可以识别出四个构建模块，分别是用户-物品编码、负采样、损失计算和反向传播更新。据我们所知，现有算法仅研究了前三个模块，却忽略了反向传播模块。他们都采用某种形式的双反向传播策略，基于一个隐含的假设，即在训练阶段平等对待用户和物品。在本文中，我们挑战了这种平等训练假设，并提出了一种新颖的单次反向传播更新策略，这种策略保留了物品编码塔的正常梯度反向传播，但削减了用户编码塔的反向传播。相反，我们提出了一种移动聚合更新策略来更新每个训练周期中的用户编码。

    arXiv:2403.18227v1 Announce Type: new  Abstract: Recent years have witnessed extensive researches on developing two tower recommendation models for relieving information overload. Four building modules can be identified in such models, namely, user-item encoding, negative sampling, loss computing and back-propagation updating. To the best of our knowledge, existing algorithms have researched only on the first three modules, yet neglecting the backpropagation module. They all adopt a kind of two backpropagation strategy, which are based on an implicit assumption of equally treating users and items in the training phase. In this paper, we challenge such an equal training assumption and propose a novel one backpropagation updating strategy, which keeps the normal gradient backpropagation for the item encoding tower, but cuts off the backpropagation for the user encoding tower. Instead, we propose a moving-aggregation updating strategy to update a user encoding in each training epoch. Exce
    

