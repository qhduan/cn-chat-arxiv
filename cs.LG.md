# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diffusion on language model embeddings for protein sequence generation](https://arxiv.org/abs/2403.03726) | 使用DiMA模型，在蛋白语言模型嵌入进行扩散来生成氨基酸序列，比传统解决方案表现更好，并通过设计选择的影响来量化其优越性能。 |
| [^2] | [Moco: A Learnable Meta Optimizer for Combinatorial Optimization](https://arxiv.org/abs/2402.04915) | Moco是一个可学习的组合优化元优化器，通过学习图神经网络来更新解决方案构建过程，并能够适应不同的情况和计算预算。 |
| [^3] | [(Ir)rationality in AI: State of the Art, Research Challenges and Open Questions](https://arxiv.org/abs/2311.17165) | 这篇论文调查了人工智能中理性与非理性的概念，提出了未解问题。重点讨论了行为在某些情况下的非理性行为可能是最优的情况。已经提出了一些方法来处理非理性代理，但仍存在挑战和问题。 |
| [^4] | [Universally Robust Graph Neural Networks by Preserving Neighbor Similarity.](http://arxiv.org/abs/2401.09754) | 本文通过保持邻居相似性实现了普适鲁棒的图神经网络，并在异类图上探索了图神经网络的脆弱性。理论上证明了负分类损失的更新与基于邻居特征的成对相似性呈负相关，解释了图攻击者连接不相似节点对的行为。通过这种方法，我们新颖地提出了一种解决方案。 |

# 详细

[^1]: 蛋白质序列生成的语言模型嵌入扩散

    Diffusion on language model embeddings for protein sequence generation

    [https://arxiv.org/abs/2403.03726](https://arxiv.org/abs/2403.03726)

    使用DiMA模型，在蛋白语言模型嵌入进行扩散来生成氨基酸序列，比传统解决方案表现更好，并通过设计选择的影响来量化其优越性能。

    

    蛋白设计需要对蛋白质宇宙固有复杂性的深入了解。尽管许多工作倾向于有条件的生成或专注于特定蛋白质家族，但无条件生成的基础任务仍未得到充分探索和重视。在这里，我们探索这个关键领域，引入了DiMA，这是一个利用从蛋白语言模型ESM-2衍生的嵌入进行连续扩散以生成氨基酸序列的模型。DiMA超越了包括自回归变换器和离散扩散模型在内的主要解决方案，我们定量地说明了导致其卓越性能的设计选择所带来的影响。我们使用各种指标跨多种形式广泛评估生成序列的质量、多样性、分布相似性和生物相关性。我们的方法始终产生新颖、多样化的蛋白质序列，精准

    arXiv:2403.03726v1 Announce Type: cross  Abstract: Protein design requires a deep understanding of the inherent complexities of the protein universe. While many efforts lean towards conditional generation or focus on specific families of proteins, the foundational task of unconditional generation remains underexplored and undervalued. Here, we explore this pivotal domain, introducing DiMA, a model that leverages continuous diffusion on embeddings derived from the protein language model, ESM-2, to generate amino acid sequences. DiMA surpasses leading solutions, including autoregressive transformer-based and discrete diffusion models, and we quantitatively illustrate the impact of the design choices that lead to its superior performance. We extensively evaluate the quality, diversity, distribution similarity, and biological relevance of the generated sequences using multiple metrics across various modalities. Our approach consistently produces novel, diverse protein sequences that accura
    
[^2]: Moco: 一种可学习的组合优化元优化器

    Moco: A Learnable Meta Optimizer for Combinatorial Optimization

    [https://arxiv.org/abs/2402.04915](https://arxiv.org/abs/2402.04915)

    Moco是一个可学习的组合优化元优化器，通过学习图神经网络来更新解决方案构建过程，并能够适应不同的情况和计算预算。

    

    相关的组合优化问题（COPs）通常是NP难的。过去，这些问题主要是通过人工设计的启发式方法来解决的，但是神经网络的进展促使人们开发了从数据中学习启发式方法的通用方法。许多方法利用神经网络直接构建解决方案，但在推理时无法进一步改进已经构建的解决方案。我们的方法Moco学习了一个图神经网络，根据从当前搜索状态提取的特征来更新解决方案构建过程。这种元训练过程以搜索过程中找到的最佳解决方案为目标，给定搜索预算等信息。这使得Moco能够适应不同的情况，例如不同的计算预算。Moco是一个完全可学习的元优化器，不使用任何特定问题的局部搜索或分解。我们在旅行商问题（TSP）和最大最小费用流问题中测试了Moco。

    Relevant combinatorial optimization problems (COPs) are often NP-hard. While they have been tackled mainly via handcrafted heuristics in the past, advances in neural networks have motivated the development of general methods to learn heuristics from data. Many approaches utilize a neural network to directly construct a solution, but are limited in further improving based on already constructed solutions at inference time. Our approach, Moco, learns a graph neural network that updates the solution construction procedure based on features extracted from the current search state. This meta training procedure targets the overall best solution found during the search procedure given information such as the search budget. This allows Moco to adapt to varying circumstances such as different computational budgets. Moco is a fully learnable meta optimizer that does not utilize any problem specific local search or decomposition. We test Moco on the Traveling Salesman Problem (TSP) and Maximum In
    
[^3]: (非)理性在人工智能中的应用：现状、研究挑战和未解之问

    (Ir)rationality in AI: State of the Art, Research Challenges and Open Questions

    [https://arxiv.org/abs/2311.17165](https://arxiv.org/abs/2311.17165)

    这篇论文调查了人工智能中理性与非理性的概念，提出了未解问题。重点讨论了行为在某些情况下的非理性行为可能是最优的情况。已经提出了一些方法来处理非理性代理，但仍存在挑战和问题。

    

    理性概念在人工智能领域中占据着重要地位。无论是模拟人类推理还是追求有限最优性，我们通常希望使人工智能代理尽可能理性。尽管这个概念在人工智能中非常核心，但对于什么构成理性代理并没有统一的定义。本文调查了人工智能中的理性与非理性，并提出了这个领域的未解问题。在其他领域对理性的理解对其在人工智能中的概念产生了影响，特别是经济学、哲学和心理学方面的研究。着重考虑人工智能代理的行为，我们探讨了在某些情境中非理性行为可能是最优的情况。关于处理非理性代理的方法已经得到了一些发展，包括识别和交互等方面的研究，然而，在这个领域的工作仍然存在一些挑战和问题。

    arXiv:2311.17165v2 Announce Type: replace Abstract: The concept of rationality is central to the field of artificial intelligence. Whether we are seeking to simulate human reasoning, or the goal is to achieve bounded optimality, we generally seek to make artificial agents as rational as possible. Despite the centrality of the concept within AI, there is no unified definition of what constitutes a rational agent. This article provides a survey of rationality and irrationality in artificial intelligence, and sets out the open questions in this area. The understanding of rationality in other fields has influenced its conception within artificial intelligence, in particular work in economics, philosophy and psychology. Focusing on the behaviour of artificial agents, we consider irrational behaviours that can prove to be optimal in certain scenarios. Some methods have been developed to deal with irrational agents, both in terms of identification and interaction, however work in this area re
    
[^4]: 通过保持邻居相似性实现普适鲁棒的图神经网络

    Universally Robust Graph Neural Networks by Preserving Neighbor Similarity. (arXiv:2401.09754v1 [cs.LG])

    [http://arxiv.org/abs/2401.09754](http://arxiv.org/abs/2401.09754)

    本文通过保持邻居相似性实现了普适鲁棒的图神经网络，并在异类图上探索了图神经网络的脆弱性。理论上证明了负分类损失的更新与基于邻居特征的成对相似性呈负相关，解释了图攻击者连接不相似节点对的行为。通过这种方法，我们新颖地提出了一种解决方案。

    

    尽管图神经网络在学习关系数据方面取得了巨大成功，但已经广泛研究发现，图神经网络在同类图上容易受到结构攻击的影响。受此启发，我们提出了一系列鲁棒模型，以增强图神经网络在同类图上的对抗鲁棒性。然而，关于异类图上的脆弱性仍然存在许多未解之谜。为了弥合这一差距，本文开始探索图神经网络在异类图上的脆弱性，并在理论上证明了负分类损失的更新与基于邻居特征的幂和聚合的成对相似性呈负相关。这一理论证明解释了实证观察，即图攻击者倾向于基于邻居特征而不是个体特征连接不相似节点对，无论是在同类图还是异类图上。通过这种方式，我们新颖地引入了一种方法

    Despite the tremendous success of graph neural networks in learning relational data, it has been widely investigated that graph neural networks are vulnerable to structural attacks on homophilic graphs. Motivated by this, a surge of robust models is crafted to enhance the adversarial robustness of graph neural networks on homophilic graphs. However, the vulnerability based on heterophilic graphs remains a mystery to us. To bridge this gap, in this paper, we start to explore the vulnerability of graph neural networks on heterophilic graphs and theoretically prove that the update of the negative classification loss is negatively correlated with the pairwise similarities based on the powered aggregated neighbor features. This theoretical proof explains the empirical observations that the graph attacker tends to connect dissimilar node pairs based on the similarities of neighbor features instead of ego features both on homophilic and heterophilic graphs. In this way, we novelly introduce a
    

