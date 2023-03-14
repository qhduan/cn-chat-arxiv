# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Weighted Notions of Fairness with Binary Supermodular Chores.](http://arxiv.org/abs/2303.06212) | 本文研究了在具有二元超模成本函数的代理之间分配不可分割家务的问题，并提出了一个通用的公平分配框架，适用于这类估值函数。该框架可以有效地计算满足加权公平性概念的分配。 |
| [^2] | [LP-Duality Theory and the Cores of Games.](http://arxiv.org/abs/2302.07627) | 本文使用LP对偶理论的构建块解决博弈核心研究中的基本差距，包括定义新的游戏、查找公平核心分配的有效算法以及在分配游戏的推广中鼓励多样性和避免过度代表。 |
| [^3] | [Learning Sparse Graphon Mean Field Games.](http://arxiv.org/abs/2209.03880) | 本文提出了一种新型的GMFG公式，称为LPGMFG，它利用$L^p$图形的图形理论概念，提供了一种机器学习工具，以有效且准确地近似解决稀疏网络问题，特别是幂律网络。我们推导出理论存在和收敛保证，并给出了实证例子，证明了我们的学习方法的准确性。 |
| [^4] | [Bayesian Optimization-based Combinatorial Assignment.](http://arxiv.org/abs/2208.14698) | 本文提出了一种基于贝叶斯优化的组合分配（BOCA）机制，通过将捕获模型不确定性的方法集成到迭代组合拍卖机制中，解决了组合分配领域中先前工作的主要缺点，能够更好地引导代理提供信息。 |
| [^5] | [A Dataset on Malicious Paper Bidding in Peer Review.](http://arxiv.org/abs/2207.02303) | 本文提供了一份关于同行评审中恶意投标的数据集，填补了这一领域缺乏公开数据的空白。 |

# 详细

[^1]: 带有二元超模矩阵的公平性加权概念

    Weighted Notions of Fairness with Binary Supermodular Chores. (arXiv:2303.06212v1 [cs.GT])

    [http://arxiv.org/abs/2303.06212](http://arxiv.org/abs/2303.06212)

    本文研究了在具有二元超模成本函数的代理之间分配不可分割家务的问题，并提出了一个通用的公平分配框架，适用于这类估值函数。该框架可以有效地计算满足加权公平性概念的分配。

    This paper studies the problem of allocating indivisible chores among agents with binary supermodular cost functions and presents a general framework for fair allocation with this class of valuation functions. The framework allows for efficient computation of allocations that satisfy weighted notions of fairness.

    我们研究了在具有二元超模成本函数的代理之间分配不可分割家务的问题。换句话说，每个家务的边际成本为$0$或$1$，并且家务呈现出递增的边际成本（或递减的边际效用）。在本文中，我们结合了Viswanathan和Zick（2022）以及Barman等人（2023）的技术，提出了一个通用的公平分配框架，适用于这类估值函数。我们的框架允许我们推广Barman等人（2023）的结果，并有效地计算满足加权公平性概念（如加权leximin或min加权$p$-mean malfare，其中$p \ge 1$）的分配。

    We study the problem of allocating indivisible chores among agents with binary supermodular cost functions. In other words, each chore has a marginal cost of $0$ or $1$ and chores exhibit increasing marginal costs (or decreasing marginal utilities). In this note, we combine the techniques of Viswanathan and Zick (2022) and Barman et al. (2023) to present a general framework for fair allocation with this class of valuation functions. Our framework allows us to generalize the results of Barman et al. (2023) and efficiently compute allocations which satisfy weighted notions of fairness like weighted leximin or min weighted $p$-mean malfare for any $p \ge 1$.
    
[^2]: LP对偶理论与博弈核心

    LP-Duality Theory and the Cores of Games. (arXiv:2302.07627v5 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2302.07627](http://arxiv.org/abs/2302.07627)

    本文使用LP对偶理论的构建块解决博弈核心研究中的基本差距，包括定义新的游戏、查找公平核心分配的有效算法以及在分配游戏的推广中鼓励多样性和避免过度代表。

    This paper addresses basic gaps in the study of game cores using building blocks from LP-duality theory, including defining new games, finding efficient algorithms for fair core imputations, and promoting diversity and avoiding over-representation in a generalization of the assignment game.

    LP对偶理论从早期到现在一直在博弈核心研究中扮演着重要角色。然而，尽管这项工作已经广泛开展，但仍存在基本差距。我们使用LP对偶理论的以下构建块来解决这些差距：1.完全单调性（TUM）。2.互补松弛条件和严格互补性。我们对TUM的探索导致定义新的游戏，表征它们的核心，并提供使用核心分配的新方法，以强制执行自然应用中出现的约束条件。后者包括：1.查找最小最大公平、最大最小公平和公平核心分配的有效算法。2.在分配游戏的推广中鼓励多样性和避免过度代表。互补性使我们能够证明分配游戏及其推广的核心分配的新属性。

    LP-duality theory has played a central role in the study of the core, right from its early days to the present time. However, despite the extensive nature of this work, basic gaps still remain. We address these gaps using the following building blocks from LP-duality theory: 1. Total unimodularity (TUM). 2. Complementary slackness conditions and strict complementarity. Our exploration of TUM leads to defining new games, characterizing their cores and giving novel ways of using core imputations to enforce constraints that arise naturally in applications of these games. The latter include: 1. Efficient algorithms for finding min-max fair, max-min fair and equitable core imputations. 2. Encouraging diversity and avoiding over-representation in a generalization of the assignment game. Complementarity enables us to prove new properties of core imputations of the assignment game and its generalizations.
    
[^3]: 学习稀疏图形均场博弈

    Learning Sparse Graphon Mean Field Games. (arXiv:2209.03880v3 [cs.MA] UPDATED)

    [http://arxiv.org/abs/2209.03880](http://arxiv.org/abs/2209.03880)

    本文提出了一种新型的GMFG公式，称为LPGMFG，它利用$L^p$图形的图形理论概念，提供了一种机器学习工具，以有效且准确地近似解决稀疏网络问题，特别是幂律网络。我们推导出理论存在和收敛保证，并给出了实证例子，证明了我们的学习方法的准确性。

    This paper proposes a novel formulation of GMFGs, called LPGMFG, which leverages the graph theoretical concept of $L^p$ graphons and provides a machine learning tool to efficiently and accurately approximate solutions for sparse network problems, especially power law networks. The paper derives theoretical existence and convergence guarantees and gives empirical examples that demonstrate the accuracy of the learning method.

    尽管多智能体强化学习（MARL）领域在过去几年中取得了相当大的进展，但解决具有大量代理的系统仍然是一个难题。图形均场博弈（GMFG）使得可以对否则难以处理的MARL问题进行可扩展的分析。由于图形的数学结构，这种方法仅限于描述许多现实世界网络（如幂律图）的稠密图形，这是不足的。我们的论文介绍了GMFG的新型公式，称为LPGMFG，它利用$L^p$图形的图形理论概念，并提供了一种机器学习工具，以有效且准确地近似解决稀疏网络问题。这尤其包括在各种应用领域中经验观察到的幂律网络，这些网络无法被标准图形所捕捉。我们推导出理论存在和收敛保证，并给出了实证例子，证明了我们的学习方法的准确性。

    Although the field of multi-agent reinforcement learning (MARL) has made considerable progress in the last years, solving systems with a large number of agents remains a hard challenge. Graphon mean field games (GMFGs) enable the scalable analysis of MARL problems that are otherwise intractable. By the mathematical structure of graphons, this approach is limited to dense graphs which are insufficient to describe many real-world networks such as power law graphs. Our paper introduces a novel formulation of GMFGs, called LPGMFGs, which leverages the graph theoretical concept of $L^p$ graphons and provides a machine learning tool to efficiently and accurately approximate solutions for sparse network problems. This especially includes power law networks which are empirically observed in various application areas and cannot be captured by standard graphons. We derive theoretical existence and convergence guarantees and give empirical examples that demonstrate the accuracy of our learning ap
    
[^4]: 基于贝叶斯优化的组合分配

    Bayesian Optimization-based Combinatorial Assignment. (arXiv:2208.14698v5 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2208.14698](http://arxiv.org/abs/2208.14698)

    本文提出了一种基于贝叶斯优化的组合分配（BOCA）机制，通过将捕获模型不确定性的方法集成到迭代组合拍卖机制中，解决了组合分配领域中先前工作的主要缺点，能够更好地引导代理提供信息。

    This paper proposes a Bayesian optimization-based combinatorial assignment (BOCA) mechanism, which addresses the main shortcoming of prior work in the combinatorial assignment domain by integrating a method for capturing model uncertainty into an iterative combinatorial auction mechanism, and can better elicit information from agents.

    本文研究组合分配领域，包括组合拍卖和课程分配。该领域的主要挑战是随着物品数量的增加，捆绑空间呈指数增长。为了解决这个问题，最近有几篇论文提出了基于机器学习的偏好引导算法，旨在从代理中仅引导出最重要的信息。然而，这些先前工作的主要缺点是它们没有对尚未引导出的捆绑值的机制不确定性进行建模。本文通过提出一种基于贝叶斯优化的组合分配（BOCA）机制来解决这个缺点。我们的关键技术贡献是将捕获模型不确定性的方法集成到迭代组合拍卖机制中。具体而言，我们设计了一种新的方法来估计可用于定义获取函数以确定下一个查询的上限不确定性界限。这使得机制能够

    We study the combinatorial assignment domain, which includes combinatorial auctions and course allocation. The main challenge in this domain is that the bundle space grows exponentially in the number of items. To address this, several papers have recently proposed machine learning-based preference elicitation algorithms that aim to elicit only the most important information from agents. However, the main shortcoming of this prior work is that it does not model a mechanism's uncertainty over values for not yet elicited bundles. In this paper, we address this shortcoming by presenting a Bayesian optimization-based combinatorial assignment (BOCA) mechanism. Our key technical contribution is to integrate a method for capturing model uncertainty into an iterative combinatorial auction mechanism. Concretely, we design a new method for estimating an upper uncertainty bound that can be used to define an acquisition function to determine the next query to the agents. This enables the mechanism 
    
[^5]: 一份关于同行评审中恶意投标的数据集

    A Dataset on Malicious Paper Bidding in Peer Review. (arXiv:2207.02303v2 [cs.CR] UPDATED)

    [http://arxiv.org/abs/2207.02303](http://arxiv.org/abs/2207.02303)

    本文提供了一份关于同行评审中恶意投标的数据集，填补了这一领域缺乏公开数据的空白。

    This paper provides a dataset on malicious paper bidding in peer review, filling the gap of lack of publicly-available data in this field.

    在会议同行评审中，评审人通常被要求对每篇提交的论文提供“投标”，以表达他们对审查该论文的兴趣。然后，一种论文分配算法使用这些投标（以及其他数据）来计算评审人对论文的高质量分配。然而，这个过程已经被恶意评审人利用，他们会有策略地投标，以非道德的方式操纵论文分配，从而严重破坏同行评审过程。例如，这些评审人可能会试图被分配到朋友的论文中，作为一种交换条件。解决这个问题的一个关键障碍是缺乏任何公开可用的关于恶意投标的数据。在这项工作中，我们收集并公开发布了一个新的数据集，以填补这一空白，该数据集是从一个模拟会议活动中收集的，参与者被要求诚实或恶意地投标。我们进一步提供了对投标行为的描述性分析。

    In conference peer review, reviewers are often asked to provide "bids" on each submitted paper that express their interest in reviewing that paper. A paper assignment algorithm then uses these bids (along with other data) to compute a high-quality assignment of reviewers to papers. However, this process has been exploited by malicious reviewers who strategically bid in order to unethically manipulate the paper assignment, crucially undermining the peer review process. For example, these reviewers may aim to get assigned to a friend's paper as part of a quid-pro-quo deal. A critical impediment towards creating and evaluating methods to mitigate this issue is the lack of any publicly-available data on malicious paper bidding. In this work, we collect and publicly release a novel dataset to fill this gap, collected from a mock conference activity where participants were instructed to bid either honestly or maliciously. We further provide a descriptive analysis of the bidding behavior, inc
    

