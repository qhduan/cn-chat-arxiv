# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Communication-Efficient Multimodal Federated Learning: Joint Modality and Client Selection.](http://arxiv.org/abs/2401.16685) | 本文提出了一种新的多模态联邦学习方法，通过联合模态和客户选择来解决多样的模态集合和通信限制的挑战。 |
| [^2] | [Optimal Low-Rank Matrix Completion: Semidefinite Relaxations and Eigenvector Disjunctions.](http://arxiv.org/abs/2305.12292) | 该论文通过重新表述低秩矩阵填补问题为投影矩阵的非凸问题，实现了能够确定最优解的分离分支定界方案，并且通过新颖和紧密的凸松弛方法，使得最优性差距相对于现有方法减少了两个数量级。 |
| [^3] | [Assessment of Reinforcement Learning for Macro Placement.](http://arxiv.org/abs/2302.11014) | 本论文提供了基于强化学习的宏观布局方法以及Circuit Training (CT)实现的开源代码和评估。研究人员评估了CT相对于多个可替代的宏观布局方法，并进行了学术性混合尺寸布局基准测试和消融和稳定性研究，为未来的相关研究提供了方向。 |
| [^4] | [Increasing Fairness via Combination with Learning Guarantees.](http://arxiv.org/abs/2301.10813) | 该论文提出了一种公平质量度量方法，名为判别风险，旨在反映个体和群体公平性。此外，研究者还讨论了公平性是否可以在理论上得到保证。 |

# 详细

[^1]: 通信高效的多模态联邦学习：联合模态和客户选择

    Communication-Efficient Multimodal Federated Learning: Joint Modality and Client Selection. (arXiv:2401.16685v1 [cs.LG])

    [http://arxiv.org/abs/2401.16685](http://arxiv.org/abs/2401.16685)

    本文提出了一种新的多模态联邦学习方法，通过联合模态和客户选择来解决多样的模态集合和通信限制的挑战。

    

    多模态联邦学习旨在丰富在客户端收集多模态测量的联邦学习环境中的模型训练。然而，多模态联邦学习面临一些尚未解决的关键挑战，特别是在异构网络环境中：(i)每个客户端收集的模态集合将是多样的，(ii)通信限制阻止客户端将其所有本地训练的模态模型上传到服务器。在本文中，我们提出了多模态联邦学习与联合模态和客户选择(mmFedMC)，一种新的联邦学习方法，可以解决多模态环境中的上述挑战。联合选择算法包含两个主要组成部分：(a)为每个客户端设计的模态选择方法，根据Shapley值分析评估模态的影响，根据通信开销的模态模型大小，结合模态模型更新频率（称为最近更新）作为权重，以增强模态选择的效果。

    Multimodal federated learning (FL) aims to enrich model training in FL settings where clients are collecting measurements across multiple modalities. However, key challenges to multimodal FL remain unaddressed, particularly in heterogeneous network settings where: (i) the set of modalities collected by each client will be diverse, and (ii) communication limitations prevent clients from uploading all their locally trained modality models to the server. In this paper, we propose multimodal Federated learning with joint Modality and Client selection (mmFedMC), a new FL methodology that can tackle the above-mentioned challenges in multimodal settings. The joint selection algorithm incorporates two main components: (a) A modality selection methodology for each client, which weighs (i) the impact of the modality, gauged by Shapley value analysis, (ii) the modality model size as a gauge of communication overhead, against (iii) the frequency of modality model updates, denoted recency, to enhan
    
[^2]: 最优低秩矩阵填补：半定松弛和特征向量分离

    Optimal Low-Rank Matrix Completion: Semidefinite Relaxations and Eigenvector Disjunctions. (arXiv:2305.12292v1 [cs.LG])

    [http://arxiv.org/abs/2305.12292](http://arxiv.org/abs/2305.12292)

    该论文通过重新表述低秩矩阵填补问题为投影矩阵的非凸问题，实现了能够确定最优解的分离分支定界方案，并且通过新颖和紧密的凸松弛方法，使得最优性差距相对于现有方法减少了两个数量级。

    

    低秩矩阵填补的目的是计算一个复杂度最小的矩阵，以尽可能准确地恢复给定的一组观测数据，并且具有众多应用，如产品推荐。不幸的是，现有的解决低秩矩阵填补的方法是启发式的，虽然高度可扩展并且通常能够确定高质量的解决方案，但不具备任何最优性保证。我们通过将低秩问题重新表述为投影矩阵的非凸问题，并实现一种分离分支定界方案来重新审视矩阵填补问题，以实现最优性导向。此外，我们通过将低秩矩阵分解为一组秩一矩阵的和，并通过 Shor 松弛来激励每个秩一矩阵中的每个 2*2 小矩阵的行列式为零，从而推导出一种新颖且通常很紧的凸松弛类。在数值实验中，相对于最先进的启发式方法，我们的新凸松弛方法将最优性差距减少了两个数量级。

    Low-rank matrix completion consists of computing a matrix of minimal complexity that recovers a given set of observations as accurately as possible, and has numerous applications such as product recommendation. Unfortunately, existing methods for solving low-rank matrix completion are heuristics that, while highly scalable and often identifying high-quality solutions, do not possess any optimality guarantees. We reexamine matrix completion with an optimality-oriented eye, by reformulating low-rank problems as convex problems over the non-convex set of projection matrices and implementing a disjunctive branch-and-bound scheme that solves them to certifiable optimality. Further, we derive a novel and often tight class of convex relaxations by decomposing a low-rank matrix as a sum of rank-one matrices and incentivizing, via a Shor relaxation, that each two-by-two minor in each rank-one matrix has determinant zero. In numerical experiments, our new convex relaxations decrease the optimali
    
[^3]: 强化学习在宏观布局中的评估

    Assessment of Reinforcement Learning for Macro Placement. (arXiv:2302.11014v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.11014](http://arxiv.org/abs/2302.11014)

    本论文提供了基于强化学习的宏观布局方法以及Circuit Training (CT)实现的开源代码和评估。研究人员评估了CT相对于多个可替代的宏观布局方法，并进行了学术性混合尺寸布局基准测试和消融和稳定性研究，为未来的相关研究提供了方向。

    

    我们提供了Google Brain深度强化学习方法在宏观布局及其Circuit Training (CT)实现的开放透明实现和评估，并在GitHub中实现了CT的关键"黑盒"元素，澄清了CT与Nature论文之间的差异。我们开发并发布了新的对开放实现的测试用例。我们评估了CT及多个可替代的宏观布局方法，所有的评估流程和相关脚本都在GitHub上公开。我们的实验还包括了学术性混合尺寸布局基准测试，以及消融和稳定性研究。我们评论了Nature和CT的影响，以及未来研究的方向。

    We provide open, transparent implementation and assessment of Google Brain's deep reinforcement learning approach to macro placement and its Circuit Training (CT) implementation in GitHub. We implement in open source key "blackbox" elements of CT, and clarify discrepancies between CT and Nature paper. New testcases on open enablements are developed and released. We assess CT alongside multiple alternative macro placers, with all evaluation flows and related scripts public in GitHub. Our experiments also encompass academic mixed-size placement benchmarks, as well as ablation and stability studies. We comment on the impact of Nature and CT, as well as directions for future research.
    
[^4]: 通过学习保证提高公平性

    Increasing Fairness via Combination with Learning Guarantees. (arXiv:2301.10813v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.10813](http://arxiv.org/abs/2301.10813)

    该论文提出了一种公平质量度量方法，名为判别风险，旨在反映个体和群体公平性。此外，研究者还讨论了公平性是否可以在理论上得到保证。

    

    随着机器学习系统在越来越多的现实场景中得到广泛应用，对于隐藏在机器学习模型中的潜在歧视的担忧正在增加。许多技术已经被开发出来以增强公平性，包括常用的群体公平性度量和几种结合集成学习的公平感知方法。然而，现有的公平度量只能关注其中之一，即群体公平性或个体公平性，它们之间的硬性兼容性暗示了即使其中之一得到满足，仍可能存在偏见。此外，现有的提升公平性的机制通常只提供经验结果来证明其有效性，但很少有论文讨论公平性是否可以在理论上得到保证。为了解决这些问题，本文提出了一种公平质量度量方法——判别风险，以反映个体和群体公平性两个方面。此外，我们还研究了p...

    The concern about underlying discrimination hidden in ML models is increasing, as ML systems have been widely applied in more and more real-world scenarios and any discrimination hidden in them will directly affect human life. Many techniques have been developed to enhance fairness including commonly-used group fairness measures and several fairness-aware methods combining ensemble learning. However, existing fairness measures can only focus on one aspect -- either group or individual fairness, and the hard compatibility among them indicates a possibility of remaining biases even if one of them is satisfied. Moreover, existing mechanisms to boost fairness usually present empirical results to show validity, yet few of them discuss whether fairness can be boosted with certain theoretical guarantees. To address these issues, we propose a fairness quality measure named discriminative risk in this paper to reflect both individual and group fairness aspects. Furthermore, we investigate the p
    

