# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reinforcement Learning-based Counter-Misinformation Response Generation: A Case Study of COVID-19 Vaccine Misinformation.](http://arxiv.org/abs/2303.06433) | 本研究使用强化学习算法创建了反虚假信息响应生成模型，以帮助普通用户有效纠正虚假信息。 |
| [^2] | [Space-Invariant Projection in Streaming Network Embedding.](http://arxiv.org/abs/2303.06293) | 本文提供了一个最大新节点数量的阈值，该阈值使节点嵌入空间保持近似等效，并提出了一种生成框架，称为空间不变投影（SIP），使任意静态MF嵌入方案能够快速嵌入动态网络中的新节点。 |
| [^3] | [Multi-Frequency Joint Community Detection and Phase Synchronization.](http://arxiv.org/abs/2206.12276) | 本文提出了两种简单而高效的算法，利用MLE公式并从多个频率的信息中受益，用于解决具有相对相位的随机块模型上的联合社区检测和相位同步问题。 |
| [^4] | [Developing a Trusted Human-AI Network for Humanitarian Benefit.](http://arxiv.org/abs/2112.11191) | 本文提出了一种可信的人工智能通信网络，将通信协议、区块链技术和信息融合与AI集成，以改善冲突通信，为人道主义利益提供可问责信息交换。 |
| [^5] | [Adversarially Regularized Graph Attention Networks for Inductive Learning on Partially Labeled Graphs.](http://arxiv.org/abs/2106.03393) | 本文提出了一种对抗正则化图注意力模型，用于在部分标记的图中分类新添加的节点，通过聚合其相邻节点的信息生成节点的表示，从而自然地推广到以前未见过的节点。 |

# 详细

[^1]: 基于强化学习的反虚假信息响应生成：以COVID-19疫苗虚假信息为例

    Reinforcement Learning-based Counter-Misinformation Response Generation: A Case Study of COVID-19 Vaccine Misinformation. (arXiv:2303.06433v1 [cs.SI])

    [http://arxiv.org/abs/2303.06433](http://arxiv.org/abs/2303.06433)

    本研究使用强化学习算法创建了反虚假信息响应生成模型，以帮助普通用户有效纠正虚假信息。

    This study creates a counter-misinformation response generation model using reinforcement learning algorithm to empower ordinary users to effectively correct misinformation.

    在线虚假信息的传播威胁着公共卫生、民主和更广泛的社会。本文旨在创建一个反虚假信息响应生成模型，以赋予用户有效纠正虚假信息的能力。本研究创建了两个虚假信息和反虚假信息数据集，使用强化学习算法训练模型，使其能够生成有效的反虚假信息响应。

    The spread of online misinformation threatens public health, democracy, and the broader society. While professional fact-checkers form the first line of defense by fact-checking popular false claims, they do not engage directly in conversations with misinformation spreaders. On the other hand, non-expert ordinary users act as eyes-on-the-ground who proactively counter misinformation -- recent research has shown that 96% counter-misinformation responses are made by ordinary users. However, research also found that 2/3 times, these responses are rude and lack evidence. This work seeks to create a counter-misinformation response generation model to empower users to effectively correct misinformation. This objective is challenging due to the absence of datasets containing ground-truth of ideal counter-misinformation responses, and the lack of models that can generate responses backed by communication theories. In this work, we create two novel datasets of misinformation and counter-misinfo
    
[^2]: 流式网络嵌入中的空间不变投影

    Space-Invariant Projection in Streaming Network Embedding. (arXiv:2303.06293v1 [cs.SI])

    [http://arxiv.org/abs/2303.06293](http://arxiv.org/abs/2303.06293)

    本文提供了一个最大新节点数量的阈值，该阈值使节点嵌入空间保持近似等效，并提出了一种生成框架，称为空间不变投影（SIP），使任意静态MF嵌入方案能够快速嵌入动态网络中的新节点。

    This paper provides a threshold for the maximum number of new nodes that keep the node embedding space approximately equivalent, and proposes a generation framework called Space-Invariant Projection (SIP) to enable fast embedding of new nodes in dynamic networks using any static MF-based embedding scheme.

    动态网络中新到达的节点会逐渐使节点嵌入空间漂移，因此需要重新训练节点嵌入和下游模型。然而，很少有人在理论或实验中考虑这些新节点的确切阈值大小，即使这些新节点的大小低于某个阈值，节点嵌入空间也很难被维护。本文从矩阵扰动理论的角度提供了一个最大新节点数量的阈值，该阈值使节点嵌入空间保持近似等效，并经过了实证验证。因此，理论上保证了当新到达节点的数量低于此阈值时，这些新节点的嵌入可以快速从原始节点的嵌入中导出。因此，提出了一种生成框架，称为空间不变投影（SIP），使任意静态MF嵌入方案能够快速嵌入动态网络中的新节点。SIP的时间复杂度与网络大小成线性关系。

    Newly arriving nodes in dynamics networks would gradually make the node embedding space drifted and the retraining of node embedding and downstream models indispensable. An exact threshold size of these new nodes, below which the node embedding space will be predicatively maintained, however, is rarely considered in either theory or experiment. From the view of matrix perturbation theory, a threshold of the maximum number of new nodes that keep the node embedding space approximately equivalent is analytically provided and empirically validated. It is therefore theoretically guaranteed that as the size of newly arriving nodes is below this threshold, embeddings of these new nodes can be quickly derived from embeddings of original nodes. A generation framework, Space-Invariant Projection (SIP), is accordingly proposed to enables arbitrary static MF-based embedding schemes to embed new nodes in dynamics networks fast. The time complexity of SIP is linear with the network size. By combinin
    
[^3]: 多频联合社区检测和相位同步

    Multi-Frequency Joint Community Detection and Phase Synchronization. (arXiv:2206.12276v2 [cs.SI] UPDATED)

    [http://arxiv.org/abs/2206.12276](http://arxiv.org/abs/2206.12276)

    本文提出了两种简单而高效的算法，利用MLE公式并从多个频率的信息中受益，用于解决具有相对相位的随机块模型上的联合社区检测和相位同步问题。

    This paper proposes two simple and efficient algorithms that leverage the MLE formulation and benefit from the information across multiple frequencies to solve the joint community detection and phase synchronization problem on the stochastic block model with relative phase.

    本文研究了具有相对相位的随机块模型上的联合社区检测和相位同步问题，其中每个节点都与一个未知的相位角相关联。这个问题具有多种实际应用，旨在同时恢复簇结构和相关的相位角。我们通过仔细研究其最大似然估计（MLE）公式，展示了这个问题呈现出“多频”结构，而现有方法并非源于这个角度。为此，提出了两种简单而高效的算法，利用MLE公式并从多个频率的信息中受益。前者是基于新颖的多频列主元QR分解的谱方法。应用于观测矩阵的前几个特征向量的分解提供了有关簇结构和相关相位角的关键信息。第二种方法是迭代的多频率方法。

    This paper studies the joint community detection and phase synchronization problem on the stochastic block model with relative phase, where each node is associated with an unknown phase angle. This problem, with a variety of real-world applications, aims to recover the cluster structure and associated phase angles simultaneously. We show this problem exhibits a ``multi-frequency'' structure by closely examining its maximum likelihood estimation (MLE) formulation, whereas existing methods are not originated from this perspective. To this end, two simple yet efficient algorithms that leverage the MLE formulation and benefit from the information across multiple frequencies are proposed. The former is a spectral method based on the novel multi-frequency column-pivoted QR factorization. The factorization applied to the top eigenvectors of the observation matrix provides key information about the cluster structure and associated phase angles. The second approach is an iterative multi-frequen
    
[^4]: 为人道主义利益开发可信的人工智能网络

    Developing a Trusted Human-AI Network for Humanitarian Benefit. (arXiv:2112.11191v3 [cs.CY] UPDATED)

    [http://arxiv.org/abs/2112.11191](http://arxiv.org/abs/2112.11191)

    本文提出了一种可信的人工智能通信网络，将通信协议、区块链技术和信息融合与AI集成，以改善冲突通信，为人道主义利益提供可问责信息交换。

    This paper proposes a trusted human-AI communication network that integrates communication protocols, blockchain technology, and information fusion with AI to improve conflict communications for accountable information exchange regarding protected entities, critical infrastructure, and humanitarian signals and status updates for humans and machines in conflicts.

    人工智能（AI）将越来越多地在冲突中以数字和物理方式参与，但缺乏与人类进行人道主义目的的可信通信。本文考虑将通信协议（“白旗协议”）、分布式账本“区块链”技术和信息融合与AI集成，以改善冲突通信，称为“受保护的保证理解情况和实体”PAUSE。这样一个可信的人工智能通信网络可以提供关于受保护实体、关键基础设施、人道主义信号和人类和机器在冲突中的状态更新的可问责信息交换。我们研究了几个现实的潜在案例研究，将这些技术集成到一个可信的人工智能网络中，以实现人道主义利益，包括实时映射冲突区域的平民和战斗人员，为避免事故做准备，并使用网络管理错误信息。

    Artificial intelligences (AI) will increasingly participate digitally and physically in conflicts, yet there is a lack of trused communications with humans for humanitarian purposes. In this paper we consider the integration of a communications protocol (the 'whiteflag protocol'), distributed ledger 'blockchain' technology, and information fusion with AI, to improve conflict communications called 'protected assurance understanding situation and entitities' PAUSE. Such a trusted human-AI communication network could provide accountable information exchange regarding protected entities, critical infrastructure, humanitiarian signals and status updates for humans and machines in conflicts. We examine several realistic potential case studies for the integration of these technologies into a trusted human-AI network for humanitarian benefit including mapping a conflict zone with civilians and combatants in real time, preparation to avoid incidents and using the network to manage misinformatio
    
[^5]: 对抗正则化图注意力网络用于部分标记图的归纳学习

    Adversarially Regularized Graph Attention Networks for Inductive Learning on Partially Labeled Graphs. (arXiv:2106.03393v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2106.03393](http://arxiv.org/abs/2106.03393)

    本文提出了一种对抗正则化图注意力模型，用于在部分标记的图中分类新添加的节点，通过聚合其相邻节点的信息生成节点的表示，从而自然地推广到以前未见过的节点。

    This paper proposes an adversarially regularized graph attention model for classifying newly added nodes in a partially labeled graph, which generates the representation of a node by aggregating information from its neighboring nodes and naturally generalizes to previously unseen nodes. Adversarial training is employed to improve the model's robustness and generalization ability.

    在实际应用中，数据标记的高成本经常导致节点标记短缺。为了提高节点分类准确性，基于图的半监督学习利用丰富的未标记节点与稀缺的可用标记节点一起训练。然而，大多数现有方法在模型训练期间需要所有节点的信息，包括要预测的节点，这在具有新添加节点的动态图中不实用。为了解决这个问题，提出了一种对抗正则化图注意力模型，用于在部分标记的图中分类新添加的节点。设计了一种基于注意力的聚合器，通过聚合其相邻节点的信息生成节点的表示，从而自然地推广到以前未见过的节点。此外，采用对抗性训练，通过强制节点表示匹配先验分布来提高模型的鲁棒性和泛化能力。在真实世界的数据集上进行了实验。

    The high cost of data labeling often results in node label shortage in real applications. To improve node classification accuracy, graph-based semi-supervised learning leverages the ample unlabeled nodes to train together with the scarce available labeled nodes. However, most existing methods require the information of all nodes, including those to be predicted, during model training, which is not practical for dynamic graphs with newly added nodes. To address this issue, an adversarially regularized graph attention model is proposed to classify newly added nodes in a partially labeled graph. An attention-based aggregator is designed to generate the representation of a node by aggregating information from its neighboring nodes, thus naturally generalizing to previously unseen nodes. In addition, adversarial training is employed to improve the model's robustness and generalization ability by enforcing node representations to match a prior distribution. Experiments on real-world datasets
    

