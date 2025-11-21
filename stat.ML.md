# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bipartite Graph Variational Auto-Encoder with Fair Latent Representation to Account for Sampling Bias in Ecological Networks](https://arxiv.org/abs/2403.02011) | 本研究提出了一种公平潜在表示的二分图变分自动编码器方法，以解决生态网络中的抽样偏差问题，通过在损失函数中引入额外的HSIC惩罚项，确保了潜在空间结构与连续变量的独立性。 |
| [^2] | [Decentralized Bilevel Optimization over Graphs: Loopless Algorithmic Update and Transient Iteration Complexity](https://arxiv.org/abs/2402.03167) | 本文提出了一种单循环的去中心化双级优化算法（D-SOBA），首次阐明了网络拓扑和数据异构性对去中心化双级算法的共同影响。D-SOBA在渐近速率、渐近梯度/海森复杂性和瞬态梯度/海森复杂性方面达到了最先进水平。 |

# 详细

[^1]: 公平潜在表示的二分图变分自动编码器，以解决生态网络中的抽样偏差问题

    Bipartite Graph Variational Auto-Encoder with Fair Latent Representation to Account for Sampling Bias in Ecological Networks

    [https://arxiv.org/abs/2403.02011](https://arxiv.org/abs/2403.02011)

    本研究提出了一种公平潜在表示的二分图变分自动编码器方法，以解决生态网络中的抽样偏差问题，通过在损失函数中引入额外的HSIC惩罚项，确保了潜在空间结构与连续变量的独立性。

    

    我们提出一种方法，使用图嵌入来表示二分网络，以解决研究生态网络所面临的挑战，比如连接植物和传粉者等网络，需考虑许多协变量，尤其要控制抽样偏差。我们将变分图自动编码器方法调整为二分情况，从而能够在潜在空间中生成嵌入，其中两组节点的位置基于它们的连接概率。我们将在社会学中常考虑的公平性框架转化为生态学中的抽样偏差问题。通过在损失函数中添加Hilbert-Schmidt独立准则（HSIC）作为额外惩罚项，我们确保潜在空间结构与连续变量（与抽样过程相关）无关。最后，我们展示了我们的方法如何改变我们对生态网络的理解。

    arXiv:2403.02011v1 Announce Type: cross  Abstract: We propose a method to represent bipartite networks using graph embeddings tailored to tackle the challenges of studying ecological networks, such as the ones linking plants and pollinators, where many covariates need to be accounted for, in particular to control for sampling bias. We adapt the variational graph auto-encoder approach to the bipartite case, which enables us to generate embeddings in a latent space where the two sets of nodes are positioned based on their probability of connection. We translate the fairness framework commonly considered in sociology in order to address sampling bias in ecology. By incorporating the Hilbert-Schmidt independence criterion (HSIC) as an additional penalty term in the loss we optimize, we ensure that the structure of the latent space is independent of continuous variables, which are related to the sampling process. Finally, we show how our approach can change our understanding of ecological n
    
[^2]: 图上的去中心化双级优化: 无环算法更新和瞬态迭代复杂性

    Decentralized Bilevel Optimization over Graphs: Loopless Algorithmic Update and Transient Iteration Complexity

    [https://arxiv.org/abs/2402.03167](https://arxiv.org/abs/2402.03167)

    本文提出了一种单循环的去中心化双级优化算法（D-SOBA），首次阐明了网络拓扑和数据异构性对去中心化双级算法的共同影响。D-SOBA在渐近速率、渐近梯度/海森复杂性和瞬态梯度/海森复杂性方面达到了最先进水平。

    

    随机双级优化（SBO）在处理嵌套结构方面的多样性使其在机器学习中变得越来越重要。为了解决大规模SBO，去中心化方法作为有效的范例出现，其中节点与直接相邻节点进行通信，无需中央服务器，从而提高通信效率和增强算法的稳健性。然而，当前的去中心化SBO算法面临挑战，包括昂贵的内部循环更新和对网络拓扑、数据异构性和嵌套双级算法结构的影响不明确。在本文中，我们引入了一种单循环的去中心化SBO（D-SOBA）算法，并建立了其瞬态迭代复杂性，首次澄清了网络拓扑和数据异构性对去中心化双级算法的共同影响。D-SOBA实现了最先进的渐近速率、渐近梯度/海森复杂性和瞬态梯度/海森复杂性。

    Stochastic bilevel optimization (SBO) is becoming increasingly essential in machine learning due to its versatility in handling nested structures. To address large-scale SBO, decentralized approaches have emerged as effective paradigms in which nodes communicate with immediate neighbors without a central server, thereby improving communication efficiency and enhancing algorithmic robustness. However, current decentralized SBO algorithms face challenges, including expensive inner-loop updates and unclear understanding of the influence of network topology, data heterogeneity, and the nested bilevel algorithmic structures. In this paper, we introduce a single-loop decentralized SBO (D-SOBA) algorithm and establish its transient iteration complexity, which, for the first time, clarifies the joint influence of network topology and data heterogeneity on decentralized bilevel algorithms. D-SOBA achieves the state-of-the-art asymptotic rate, asymptotic gradient/Hessian complexity, and transien
    

