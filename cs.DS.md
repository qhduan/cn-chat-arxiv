# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LP-Duality Theory and the Cores of Games.](http://arxiv.org/abs/2302.07627) | 本文使用LP对偶理论的构建块解决博弈核心研究中的基本差距，包括定义新的游戏、查找公平核心分配的有效算法以及在分配游戏的推广中鼓励多样性和避免过度代表。 |
| [^2] | [A Generalization of the Shortest Path Problem to Graphs with Multiple Edge-Cost Estimates.](http://arxiv.org/abs/2208.11489) | 本文提出了一个广义的加权有向图框架，其中可以多次计算（估计）边缘权重，以提高准确性和运行时间成本，解决了最短路径问题的不确定性。 |

# 详细

[^1]: LP对偶理论与博弈核心

    LP-Duality Theory and the Cores of Games. (arXiv:2302.07627v5 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2302.07627](http://arxiv.org/abs/2302.07627)

    本文使用LP对偶理论的构建块解决博弈核心研究中的基本差距，包括定义新的游戏、查找公平核心分配的有效算法以及在分配游戏的推广中鼓励多样性和避免过度代表。

    This paper addresses basic gaps in the study of game cores using building blocks from LP-duality theory, including defining new games, finding efficient algorithms for fair core imputations, and promoting diversity and avoiding over-representation in a generalization of the assignment game.

    LP对偶理论从早期到现在一直在博弈核心研究中扮演着重要角色。然而，尽管这项工作已经广泛开展，但仍存在基本差距。我们使用LP对偶理论的以下构建块来解决这些差距：1.完全单调性（TUM）。2.互补松弛条件和严格互补性。我们对TUM的探索导致定义新的游戏，表征它们的核心，并提供使用核心分配的新方法，以强制执行自然应用中出现的约束条件。后者包括：1.查找最小最大公平、最大最小公平和公平核心分配的有效算法。2.在分配游戏的推广中鼓励多样性和避免过度代表。互补性使我们能够证明分配游戏及其推广的核心分配的新属性。

    LP-duality theory has played a central role in the study of the core, right from its early days to the present time. However, despite the extensive nature of this work, basic gaps still remain. We address these gaps using the following building blocks from LP-duality theory: 1. Total unimodularity (TUM). 2. Complementary slackness conditions and strict complementarity. Our exploration of TUM leads to defining new games, characterizing their cores and giving novel ways of using core imputations to enforce constraints that arise naturally in applications of these games. The latter include: 1. Efficient algorithms for finding min-max fair, max-min fair and equitable core imputations. 2. Encouraging diversity and avoiding over-representation in a generalization of the assignment game. Complementarity enables us to prove new properties of core imputations of the assignment game and its generalizations.
    
[^2]: 具有多个边缘成本估计的图的最短路径问题的推广

    A Generalization of the Shortest Path Problem to Graphs with Multiple Edge-Cost Estimates. (arXiv:2208.11489v3 [cs.DS] UPDATED)

    [http://arxiv.org/abs/2208.11489](http://arxiv.org/abs/2208.11489)

    本文提出了一个广义的加权有向图框架，其中可以多次计算（估计）边缘权重，以提高准确性和运行时间成本，解决了最短路径问题的不确定性。

    This paper presents a generalized framework for weighted directed graphs, where edge weight can be computed (estimated) multiple times, at increasing accuracy and run-time expense, solving the uncertainty of the shortest path problem.

    图中的最短路径问题是AI理论和应用的基石。现有算法通常忽略边缘权重计算时间。在本文中，我们提出了一个广义的加权有向图框架，其中可以多次计算（估计）边缘权重，以提高准确性和运行时间成本。这引发了一个广义的最短路径问题，优化路径成本及其不确定性的不同方面。我们提出了一个完整的任何时候解决方案算法，实证证明了其功效。

    The shortest path problem in graphs is a cornerstone of AI theory and applications. Existing algorithms generally ignore edge weight computation time. In this paper we present a generalized framework for weighted directed graphs, where edge weight can be computed (estimated) multiple times, at increasing accuracy and run-time expense. This raises a generalized shortest path problem that optimize different aspects of path cost and its uncertainty. We present a complete anytime solution algorithm for the generalized problem, and empirically demonstrate its efficacy.
    

