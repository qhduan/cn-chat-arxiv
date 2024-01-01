# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fast Slate Policy Optimization: Going Beyond Plackett-Luce.](http://arxiv.org/abs/2308.01566) | 本文介绍了一种快速Slate策略优化方法，通过提出一种新的策略类，可以在大规模决策系统中有效地优化任意奖励函数，结果表明该方法在百万级别动作空间问题上具有很好的效果。 |

# 详细

[^1]: 快速Slate策略优化：超越Plackett-Luce

    Fast Slate Policy Optimization: Going Beyond Plackett-Luce. (arXiv:2308.01566v1 [cs.LG])

    [http://arxiv.org/abs/2308.01566](http://arxiv.org/abs/2308.01566)

    本文介绍了一种快速Slate策略优化方法，通过提出一种新的策略类，可以在大规模决策系统中有效地优化任意奖励函数，结果表明该方法在百万级别动作空间问题上具有很好的效果。

    

    大规模机器学习系统中一个越来越重要的构建模块是返回Slate，即给定一个查询返回有序的项目列表。该技术的应用包括搜索、信息检索和推荐系统。当行动空间很大时，决策系统会限制在特定结构中以快速完成在线查询。本文解决了这些大规模决策系统在给定任意奖励函数下的优化问题。我们将这个学习问题转化为策略优化框架，并提出了一种新的策略类，它源于决策函数的一种新颖放松。这导致了一个简单而高效的学习算法，可以扩展到大规模的动作空间。我们将我们的方法与常用的Plackett-Luce策略类进行比较，并展示了我们的方法在动作空间大小达到百万级别的问题上的有效性。

    An increasingly important building block of large scale machine learning systems is based on returning slates; an ordered lists of items given a query. Applications of this technology include: search, information retrieval and recommender systems. When the action space is large, decision systems are restricted to a particular structure to complete online queries quickly. This paper addresses the optimization of these large scale decision systems given an arbitrary reward function. We cast this learning problem in a policy optimization framework and propose a new class of policies, born from a novel relaxation of decision functions. This results in a simple, yet efficient learning algorithm that scales to massive action spaces. We compare our method to the commonly adopted Plackett-Luce policy class and demonstrate the effectiveness of our approach on problems with action space sizes in the order of millions.
    

