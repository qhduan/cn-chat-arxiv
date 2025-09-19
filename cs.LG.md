# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Utility Optimization via a GAN Approach](https://arxiv.org/abs/2403.15243) | 提出了一种生成对抗网络（GAN）方法，用于解决一般和现实设置下的稳健效用优化问题，该方法在实证研究中表现出色，能在没有已知最佳策略的情况下胜过所有其他参考策略 |
| [^2] | [Differentially private multivariate medians](https://arxiv.org/abs/2210.06459) | 差分私有多变量中位数的有限样本性能保证为常用深度函数提供了尖锐的结果，表明重尾位置估计的成本超过了隐私保护成本。 |
| [^3] | [Fast Multipole Attention: A Divide-and-Conquer Attention Mechanism for Long Sequences.](http://arxiv.org/abs/2310.11960) | 提出了一种名为快速多极化注意力的新型注意力机制，它使用分治策略将注意力的时间和内存复杂度从O(n^2)降低到O(n log n)或O(n)，同时保持了全局感知范围。 |
| [^4] | [Heterogeneous Directed Hypergraph Neural Network over abstract syntax tree (AST) for Code Classification.](http://arxiv.org/abs/2305.04228) | 本研究提出了使用异构有向超图表示AST，并使用异构有向超图神经网络处理图形进行代码分类，超过了现有方法。 |

# 详细

[^1]: 通过GAN方法实现稳健效用优化

    Robust Utility Optimization via a GAN Approach

    [https://arxiv.org/abs/2403.15243](https://arxiv.org/abs/2403.15243)

    提出了一种生成对抗网络（GAN）方法，用于解决一般和现实设置下的稳健效用优化问题，该方法在实证研究中表现出色，能在没有已知最佳策略的情况下胜过所有其他参考策略

    

    稳健效用优化使投资者能够以结构化方式处理市场不确定性，旨在最大化最坏情况的结果。在这项工作中，我们提出了一种生成对抗网络（GAN）方法，（近似地）解决一般和现实设置下的稳健效用优化问题。特别地，我们通过神经网络（NN）对投资者和市场进行建模，并在极小极大零和博弈中训练它们。这种方法适用于任何连续效用函数，并在具有交易成本的现实市场设置中，只能使用市场的可观察信息。大量实证研究显示了我们方法的多功能性。每当存在最佳参考策略时，我们的方法都能与之媲美，在没有已知最佳策略的（许多）设置中，我们的方法胜过所有其他参考策略。此外，我们可以从研究中得出结论

    arXiv:2403.15243v1 Announce Type: cross  Abstract: Robust utility optimization enables an investor to deal with market uncertainty in a structured way, with the goal of maximizing the worst-case outcome. In this work, we propose a generative adversarial network (GAN) approach to (approximately) solve robust utility optimization problems in general and realistic settings. In particular, we model both the investor and the market by neural networks (NN) and train them in a mini-max zero-sum game. This approach is applicable for any continuous utility function and in realistic market settings with trading costs, where only observable information of the market can be used. A large empirical study shows the versatile usability of our method. Whenever an optimal reference strategy is available, our method performs on par with it and in the (many) settings without known optimal strategy, our method outperforms all other reference strategies. Moreover, we can conclude from our study that the tr
    
[^2]: 差分私有多变量中位数

    Differentially private multivariate medians

    [https://arxiv.org/abs/2210.06459](https://arxiv.org/abs/2210.06459)

    差分私有多变量中位数的有限样本性能保证为常用深度函数提供了尖锐的结果，表明重尾位置估计的成本超过了隐私保护成本。

    

    现代数据分析需要满足严格隐私保证的统计工具。众所周知，对污染的鲁棒性与差分隐私有关。尽管如此，使用多元中位数进行差分私有和鲁棒的多元位置估计尚未得到系统研究。我们为差分私有多元深度中位数开发了新颖的有限样本性能保证，这些保证基本上是尖锐的。我们的结果涵盖了常用的深度函数，如半平面（或Tukey）深度，空间深度和集成双深度。我们展示了在柯西边际下，重尾位置估计的代价超过了隐私的代价。我们在高达d = 100的维度上使用高斯污染模型进行了数值演示，并将其与最先进的私有均值估计算法进行了比较。作为我们研究的一个副产品，

    arXiv:2210.06459v2 Announce Type: replace-cross  Abstract: Statistical tools which satisfy rigorous privacy guarantees are necessary for modern data analysis. It is well-known that robustness against contamination is linked to differential privacy. Despite this fact, using multivariate medians for differentially private and robust multivariate location estimation has not been systematically studied. We develop novel finite-sample performance guarantees for differentially private multivariate depth-based medians, which are essentially sharp. Our results cover commonly used depth functions, such as the halfspace (or Tukey) depth, spatial depth, and the integrated dual depth. We show that under Cauchy marginals, the cost of heavy-tailed location estimation outweighs the cost of privacy. We demonstrate our results numerically using a Gaussian contamination model in dimensions up to d = 100, and compare them to a state-of-the-art private mean estimation algorithm. As a by-product of our inv
    
[^3]: 快速多极化注意力：一种用于长序列的分治注意力机制

    Fast Multipole Attention: A Divide-and-Conquer Attention Mechanism for Long Sequences. (arXiv:2310.11960v1 [cs.CL])

    [http://arxiv.org/abs/2310.11960](http://arxiv.org/abs/2310.11960)

    提出了一种名为快速多极化注意力的新型注意力机制，它使用分治策略将注意力的时间和内存复杂度从O(n^2)降低到O(n log n)或O(n)，同时保持了全局感知范围。

    

    基于Transformer的模型已在许多领域取得了最先进的性能。然而，自注意力对于输入长度的二次复杂度限制了Transformer模型在长序列上的适用性。为了解决这个问题，我们提出了快速多极化注意力，一种使用分治策略来减少注意力时间和内存复杂度的新型注意力机制，将长度为n的序列的注意力复杂度从O(n^2)降低到O(n log n)或O(n)，同时保持了全局感知范围。这种分层方法将查询、键和值分为O(log n)级的分辨率，较远距离的组群越来越大，并学习计算组群数量的权重。因此，以高效分层的方式在较低的分辨率中考虑远离彼此的标记之间的相互作用。快速多极化注意力的总体复杂度为O(n)或O(n log n)。

    Transformer-based models have achieved state-of-the-art performance in many areas. However, the quadratic complexity of self-attention with respect to the input length hinders the applicability of Transformer-based models to long sequences. To address this, we present Fast Multipole Attention, a new attention mechanism that uses a divide-and-conquer strategy to reduce the time and memory complexity of attention for sequences of length $n$ from $\mathcal{O}(n^2)$ to $\mathcal{O}(n \log n)$ or $O(n)$, while retaining a global receptive field. The hierarchical approach groups queries, keys, and values into $\mathcal{O}( \log n)$ levels of resolution, where groups at greater distances are increasingly larger in size and the weights to compute group quantities are learned. As such, the interaction between tokens far from each other is considered in lower resolution in an efficient hierarchical manner. The overall complexity of Fast Multipole Attention is $\mathcal{O}(n)$ or $\mathcal{O}(n \
    
[^4]: 基于抽象语法树的异构有向超图神经网络用于代码分类

    Heterogeneous Directed Hypergraph Neural Network over abstract syntax tree (AST) for Code Classification. (arXiv:2305.04228v2 [cs.SE] UPDATED)

    [http://arxiv.org/abs/2305.04228](http://arxiv.org/abs/2305.04228)

    本研究提出了使用异构有向超图表示AST，并使用异构有向超图神经网络处理图形进行代码分类，超过了现有方法。

    

    代码分类是程序理解和自动编码中的一个难题。由于程序的模糊语法和复杂语义，大多数现有研究使用基于抽象语法树（AST）和图神经网络（GNN）的技术创建代码表示用于代码分类。这些技术利用代码的结构和语义信息，但只考虑节点之间的成对关系，忽略了AST中节点之间已经存在的高阶相关性，可能导致代码结构信息的丢失。本研究提出使用异构有向超图（HDHG）表示AST，并使用异构有向超图神经网络（HDHGN）处理图形。HDHG保留了节点之间的高阶相关性，并更全面地编码了AST的语义和结构信息。HDHGN通过聚合不同节点的特征并使用不同的函数对其进行处理来对AST进行建模。在四个数据集上的实验表明，HDHG和HDHGN在代码分类任务中超越了现有方法。

    Code classification is a difficult issue in program understanding and automatic coding. Due to the elusive syntax and complicated semantics in programs, most existing studies use techniques based on abstract syntax tree (AST) and graph neural network (GNN) to create code representations for code classification. These techniques utilize the structure and semantic information of the code, but they only take into account pairwise associations and neglect the high-order correlations that already exist between nodes in the AST, which may result in the loss of code structural information. On the other hand, while a general hypergraph can encode high-order data correlations, it is homogeneous and undirected which will result in a lack of semantic and structural information such as node types, edge types, and directions between child nodes and parent nodes when modeling AST. In this study, we propose to represent AST as a heterogeneous directed hypergraph (HDHG) and process the graph by hetero
    

