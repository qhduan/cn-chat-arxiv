# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Search and Society: Reimagining Information Access for Radical Futures](https://arxiv.org/abs/2403.17901) | 社区应该重新将研究议程聚焦于社会需求，消除公平性、问责制、透明度和道德研究与信息检索其他领域之间的人为隔离，积极设定研究议程，激励构建明确陈述的社会技术想象力所启发的系统类型。 |
| [^2] | [LARA: Linguistic-Adaptive Retrieval-Augmented LLMs for Multi-Turn Intent Classification](https://arxiv.org/abs/2403.16504) | LARA是一个Linguistic-Adaptive Retrieval-Augmented Language Models（语言自适应检索增强LLMs），旨在通过结合微调过的较小模型与检索增强机制来提高多语言多轮意图分类任务的准确性，从而改善对话背景的理解。 |
| [^3] | [Understanding and Guiding Weakly Supervised Entity Alignment with Potential Isomorphism Propagation](https://arxiv.org/abs/2402.03025) | 本文通过传播视角分析了弱监督的实体对齐任务，并提出一种潜在同构传播操作符来增强知识图谱之间的邻域信息传播。通过验证，发现基于聚合的实体对齐模型中的潜在对齐实体具有同构子图。 |
| [^4] | [Fine-Grained Embedding Dimension Optimization During Training for Recommender Systems.](http://arxiv.org/abs/2401.04408) | 本文提出了一种细粒度嵌入维度优化方法（FIITED），能够在推荐系统的训练过程中根据嵌入向量的重要性不断调整其维度，并设计了一种虚拟哈希索引哈希表的嵌入存储系统以有效节省内存。 |
| [^5] | [Distributionally Robust Unsupervised Dense Retrieval Training on Web Graphs.](http://arxiv.org/abs/2310.16605) | 本论文提出了一种无监督的密集检索模型Web-DRO，它利用网络结构进行聚类并重新加权，在无监督场景中显著提高了检索效果。群组分布鲁棒优化方法指导模型对高对比损失的群组分配更多权重，在训练过程中更加关注最坏情况。实验结果表明，结合URL信息的网络图训练能达到最佳的聚类性能。 |
| [^6] | [EHI: End-to-end Learning of Hierarchical Index for Efficient Dense Retrieval.](http://arxiv.org/abs/2310.08891) | EHI是一种端到端学习的层次索引方法，用于高效密集检索。它同时学习嵌入和ANNS结构，通过使用密集路径嵌入来捕获索引的语义信息，以优化检索性能。 |
| [^7] | [Pure Message Passing Can Estimate Common Neighbor for Link Prediction.](http://arxiv.org/abs/2309.00976) | 这篇论文提出了一种纯粹的消息传递方法，用于估计共同邻居进行链路预测。该方法通过利用输入向量的正交性来捕捉联合结构特征，提出了一种新的链路预测模型MPLP，该模型利用准正交向量估计链路级结构特征，同时保留了节点级复杂性。 |

# 详细

[^1]: 搜索与社会：重新构想激进未来的信息获取

    Search and Society: Reimagining Information Access for Radical Futures

    [https://arxiv.org/abs/2403.17901](https://arxiv.org/abs/2403.17901)

    社区应该重新将研究议程聚焦于社会需求，消除公平性、问责制、透明度和道德研究与信息检索其他领域之间的人为隔离，积极设定研究议程，激励构建明确陈述的社会技术想象力所启发的系统类型。

    

    arXiv: 2403.17901v1 公告类型：新摘要：信息检索（IR）技术和研究正在经历变革。我们认为社区应该抓住这个机会，重新将研究议程聚焦于社会需求，同时消除IR的公平性、问责制、透明度和道德研究与IR其他领域之间的人为隔离。社区不应采取试图减轻新兴技术可能带来社会害处的反应性策略，而应该积极设定研究议程，激励我们构建各种明确陈述的社会技术想象力所启发的系统类型。支撑信息获取技术设计和开发的社会技术想象力需要明确表达，我们需要在这些不同视角的背景下发展变革理论。我们的指导未来想象力必须受到其他学术领域的启发。

    arXiv:2403.17901v1 Announce Type: new  Abstract: Information retrieval (IR) technologies and research are undergoing transformative changes. It is our perspective that the community should accept this opportunity to re-center our research agendas on societal needs while dismantling the artificial separation between the work on fairness, accountability, transparency, and ethics in IR and the rest of IR research. Instead of adopting a reactionary strategy of trying to mitigate potential social harms from emerging technologies, the community should aim to proactively set the research agenda for the kinds of systems we should build inspired by diverse explicitly stated sociotechnical imaginaries. The sociotechnical imaginaries that underpin the design and development of information access technologies needs to be explicitly articulated, and we need to develop theories of change in context of these diverse perspectives. Our guiding future imaginaries must be informed by other academic field
    
[^2]: LARA：语言自适应检索增强LLMs用于多轮意图分类

    LARA: Linguistic-Adaptive Retrieval-Augmented LLMs for Multi-Turn Intent Classification

    [https://arxiv.org/abs/2403.16504](https://arxiv.org/abs/2403.16504)

    LARA是一个Linguistic-Adaptive Retrieval-Augmented Language Models（语言自适应检索增强LLMs），旨在通过结合微调过的较小模型与检索增强机制来提高多语言多轮意图分类任务的准确性，从而改善对话背景的理解。

    

    鉴于大型语言模型(LLMs)取得的显著成就，研究人员已经在文本分类任务中采用了上下文学习。然而，这些研究侧重于单语言、单轮分类任务。本文介绍了LARA（Linguistic-Adaptive Retrieval-Augmented Language Models），旨在增强多语言多轮分类任务的准确性，以适应聊天机器人交互中的众多意图。由于会话背景的复杂性和不断发展的性质，多轮意图分类尤为具有挑战性。LARA通过将微调过的较小模型与检索增强机制结合，嵌入LLMs的架构中来解决这些问题。这种整合使LARA能够动态利用过去的对话和相关意图，从而提高对上下文的理解。此外，我们的自适应检索技术增强了跨语言的能力。

    arXiv:2403.16504v1 Announce Type: new  Abstract: Following the significant achievements of large language models (LLMs), researchers have employed in-context learning for text classification tasks. However, these studies focused on monolingual, single-turn classification tasks. In this paper, we introduce LARA (Linguistic-Adaptive Retrieval-Augmented Language Models), designed to enhance accuracy in multi-turn classification tasks across six languages, accommodating numerous intents in chatbot interactions. Multi-turn intent classification is notably challenging due to the complexity and evolving nature of conversational contexts. LARA tackles these issues by combining a fine-tuned smaller model with a retrieval-augmented mechanism, integrated within the architecture of LLMs. This integration allows LARA to dynamically utilize past dialogues and relevant intents, thereby improving the understanding of the context. Furthermore, our adaptive retrieval techniques bolster the cross-lingual
    
[^3]: 通过潜在同构传播理解和引导弱监督的实体对齐

    Understanding and Guiding Weakly Supervised Entity Alignment with Potential Isomorphism Propagation

    [https://arxiv.org/abs/2402.03025](https://arxiv.org/abs/2402.03025)

    本文通过传播视角分析了弱监督的实体对齐任务，并提出一种潜在同构传播操作符来增强知识图谱之间的邻域信息传播。通过验证，发现基于聚合的实体对齐模型中的潜在对齐实体具有同构子图。

    

    弱监督的实体对齐是使用有限数量的种子对齐，在不同知识图谱之间识别等价实体的任务。尽管在基于聚合的弱监督实体对齐方面取得了重大进展，但在这种设置下的基本机制仍未被探索。在本文中，我们提出了一种传播视角来分析弱监督实体对齐，并解释了现有的基于聚合的实体对齐模型。我们的理论分析揭示了这些模型实质上是寻找用于对实体相似度进行传播的操作符。我们进一步证明，尽管不同知识图谱之间存在结构异质性，基于聚合的实体对齐模型中的潜在对齐实体具有同构子图，这是实体对齐的核心前提，但尚未被研究。利用这一洞见，我们引入了潜在同构传播操作符来增强跨知识图谱的邻域信息传播。我们开发了一个通用的实体对齐框架PipEA，实现了效果显著的实验结果。

    Weakly Supervised Entity Alignment (EA) is the task of identifying equivalent entities across diverse knowledge graphs (KGs) using only a limited number of seed alignments. Despite substantial advances in aggregation-based weakly supervised EA, the underlying mechanisms in this setting remain unexplored. In this paper, we present a propagation perspective to analyze weakly supervised EA and explain the existing aggregation-based EA models. Our theoretical analysis reveals that these models essentially seek propagation operators for pairwise entity similarities. We further prove that, despite the structural heterogeneity of different KGs, the potentially aligned entities within aggregation-based EA models have isomorphic subgraphs, which is the core premise of EA but has not been investigated. Leveraging this insight, we introduce a potential isomorphism propagation operator to enhance the propagation of neighborhood information across KGs. We develop a general EA framework, PipEA, inco
    
[^4]: 在推荐系统的训练过程中优化细粒度嵌入维度

    Fine-Grained Embedding Dimension Optimization During Training for Recommender Systems. (arXiv:2401.04408v1 [cs.IR])

    [http://arxiv.org/abs/2401.04408](http://arxiv.org/abs/2401.04408)

    本文提出了一种细粒度嵌入维度优化方法（FIITED），能够在推荐系统的训练过程中根据嵌入向量的重要性不断调整其维度，并设计了一种虚拟哈希索引哈希表的嵌入存储系统以有效节省内存。

    

    现代深度学习推荐模型中的大型嵌入表在训练和推断过程中需要过大的内存。为了减小训练时的内存占用，本文提出了一种细粒度嵌入维度优化方法 (FIITED)。根据嵌入向量的重要性不同，FIITED在训练过程中连续调整每个嵌入向量的维度，将更重要的嵌入向量分配更长的维度，并能够适应数据的动态变化。同时，本文设计了一种基于虚拟哈希的物理索引哈希表的嵌入存储系统，以实现嵌入维度的调整并有效地节省内存。对两个行业模型的实验表明，FIITED能够将嵌入的大小减小超过65%，同时保持训练模型的质量，比现有的一种在训练过程中进行嵌入修剪的方法节省更多内存。

    Huge embedding tables in modern Deep Learning Recommender Models (DLRM) require prohibitively large memory during training and inference. Aiming to reduce the memory footprint of training, this paper proposes FIne-grained In-Training Embedding Dimension optimization (FIITED). Given the observation that embedding vectors are not equally important, FIITED adjusts the dimension of each individual embedding vector continuously during training, assigning longer dimensions to more important embeddings while adapting to dynamic changes in data. A novel embedding storage system based on virtually-hashed physically-indexed hash tables is designed to efficiently implement the embedding dimension adjustment and effectively enable memory saving. Experiments on two industry models show that FIITED is able to reduce the size of embeddings by more than 65% while maintaining the trained model's quality, saving significantly more memory than a state-of-the-art in-training embedding pruning method. On p
    
[^5]: 基于网络图的分布鲁棒无监督密集检索训练

    Distributionally Robust Unsupervised Dense Retrieval Training on Web Graphs. (arXiv:2310.16605v1 [cs.IR])

    [http://arxiv.org/abs/2310.16605](http://arxiv.org/abs/2310.16605)

    本论文提出了一种无监督的密集检索模型Web-DRO，它利用网络结构进行聚类并重新加权，在无监督场景中显著提高了检索效果。群组分布鲁棒优化方法指导模型对高对比损失的群组分配更多权重，在训练过程中更加关注最坏情况。实验结果表明，结合URL信息的网络图训练能达到最佳的聚类性能。

    

    本文介绍了Web-DRO，一种基于网络结构进行聚类并在对比训练期间重新加权的无监督密集检索模型。具体而言，我们首先利用网络图链接并对锚点-文档对进行对比训练，训练一个嵌入模型用于聚类。然后，我们使用群组分布鲁棒优化方法来重新加权不同的锚点-文档对群组，这指导模型将更多权重分配给对比损失更高的群组，并在训练过程中更加关注最坏情况。在MS MARCO和BEIR上的实验表明，我们的模型Web-DRO在无监督场景中显著提高了检索效果。对聚类技术的比较表明，结合URL信息的网络图训练能达到最佳的聚类性能。进一步分析证实了群组权重的稳定性和有效性，表明了一致的模型偏好以及对有价值文档的有效加权。

    This paper introduces Web-DRO, an unsupervised dense retrieval model, which clusters documents based on web structures and reweights the groups during contrastive training. Specifically, we first leverage web graph links and contrastively train an embedding model for clustering anchor-document pairs. Then we use Group Distributional Robust Optimization to reweight different clusters of anchor-document pairs, which guides the model to assign more weights to the group with higher contrastive loss and pay more attention to the worst case during training. Our experiments on MS MARCO and BEIR show that our model, Web-DRO, significantly improves the retrieval effectiveness in unsupervised scenarios. A comparison of clustering techniques shows that training on the web graph combining URL information reaches optimal performance on clustering. Further analysis confirms that group weights are stable and valid, indicating consistent model preferences as well as effective up-weighting of valuable 
    
[^6]: EHI: 高效密集检索的层次索引的端到端学习

    EHI: End-to-end Learning of Hierarchical Index for Efficient Dense Retrieval. (arXiv:2310.08891v1 [cs.LG])

    [http://arxiv.org/abs/2310.08891](http://arxiv.org/abs/2310.08891)

    EHI是一种端到端学习的层次索引方法，用于高效密集检索。它同时学习嵌入和ANNS结构，通过使用密集路径嵌入来捕获索引的语义信息，以优化检索性能。

    

    密集嵌入式检索现已成为语义搜索和排名问题的行业标准，如获取给定查询的相关网络文档。这些技术使用了两个阶段的过程：(a)对比学习来训练双编码器以嵌入查询和文档，以及(b)近似最近邻搜索(ANNS)以查找给定查询的相似文档。这两个阶段是不相交的；学得的嵌入可能不适合ANNS方法，反之亦然，导致性能不佳。在这项工作中，我们提出了一种名为端到端层次索引(EHI)的方法，它同时学习嵌入和ANNS结构以优化检索性能。EHI使用标准的双编码器模型来嵌入查询和文档，同时学习一个倒排文件索引(IVF)风格的树状结构以实现高效的ANNS。为了确保离散基于树的ANNS结构的稳定和高效学习，EHI引入了密集路径嵌入的概念，用来捕获索引的语义信息。

    Dense embedding-based retrieval is now the industry standard for semantic search and ranking problems, like obtaining relevant web documents for a given query. Such techniques use a two-stage process: (a) contrastive learning to train a dual encoder to embed both the query and documents and (b) approximate nearest neighbor search (ANNS) for finding similar documents for a given query. These two stages are disjoint; the learned embeddings might be ill-suited for the ANNS method and vice-versa, leading to suboptimal performance. In this work, we propose End-to-end Hierarchical Indexing -- EHI -- that jointly learns both the embeddings and the ANNS structure to optimize retrieval performance. EHI uses a standard dual encoder model for embedding queries and documents while learning an inverted file index (IVF) style tree structure for efficient ANNS. To ensure stable and efficient learning of discrete tree-based ANNS structure, EHI introduces the notion of dense path embedding that capture
    
[^7]: 纯粹的消息传递可以估计共同邻居进行链路预测

    Pure Message Passing Can Estimate Common Neighbor for Link Prediction. (arXiv:2309.00976v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2309.00976](http://arxiv.org/abs/2309.00976)

    这篇论文提出了一种纯粹的消息传递方法，用于估计共同邻居进行链路预测。该方法通过利用输入向量的正交性来捕捉联合结构特征，提出了一种新的链路预测模型MPLP，该模型利用准正交向量估计链路级结构特征，同时保留了节点级复杂性。

    

    消息传递神经网络（MPNN）已成为图表示学习中的事实标准。然而，在链路预测方面，它们往往表现不佳，被简单的启发式算法如共同邻居（CN）所超越。这种差异源于一个根本限制：尽管MPNN在节点级表示方面表现出色，但在编码链路预测中至关重要的联合结构特征（如CN）方面则遇到困难。为了弥合这一差距，我们认为通过利用输入向量的正交性，纯粹的消息传递确实可以捕捉到联合结构特征。具体而言，我们研究了MPNN在近似CN启发式算法方面的能力。基于我们的发现，我们引入了一种新的链路预测模型——消息传递链路预测器（MPLP）。MPLP利用准正交向量估计链路级结构特征，同时保留节点级复杂性。此外，我们的方法表明利用消息传递捕捉结构特征能够改善链路预测性能。

    Message Passing Neural Networks (MPNNs) have emerged as the {\em de facto} standard in graph representation learning. However, when it comes to link prediction, they often struggle, surpassed by simple heuristics such as Common Neighbor (CN). This discrepancy stems from a fundamental limitation: while MPNNs excel in node-level representation, they stumble with encoding the joint structural features essential to link prediction, like CN. To bridge this gap, we posit that, by harnessing the orthogonality of input vectors, pure message-passing can indeed capture joint structural features. Specifically, we study the proficiency of MPNNs in approximating CN heuristics. Based on our findings, we introduce the Message Passing Link Predictor (MPLP), a novel link prediction model. MPLP taps into quasi-orthogonal vectors to estimate link-level structural features, all while preserving the node-level complexities. Moreover, our approach demonstrates that leveraging message-passing to capture stru
    

