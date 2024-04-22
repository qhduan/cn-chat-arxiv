# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [High-Frequency-aware Hierarchical Contrastive Selective Coding for Representation Learning on Text-attributed Graphs](https://arxiv.org/abs/2402.16240) | 提出了一种名为HASH-CODE的高频感知分层对比选择编码方法，将图神经网络（GNNs）和预训练语言模型（PLMs）相结合，解决了在文本属性图上节点表示学习中的挑战。 |
| [^2] | [Interactive Question Answering Systems: Literature Review](https://arxiv.org/abs/2209.01621) | 交互式问答系统是问答和对话系统的结合，用户可以用自然语言提问并与系统动态交互，获得更精确的结果。 |
| [^3] | [Self-supervised Learning of Rotation-invariant 3D Point Set Features using Transformer and its Self-distillation.](http://arxiv.org/abs/2308.04725) | 本文提出了一种使用Transformer和自蒸馏的自监督学习框架，用于从大量无标签的3D点集中获取准确且旋转不变的3D点集特征。 |

# 详细

[^1]: 针对文本属性图的高频感知分层对比选择编码用于表示学习

    High-Frequency-aware Hierarchical Contrastive Selective Coding for Representation Learning on Text-attributed Graphs

    [https://arxiv.org/abs/2402.16240](https://arxiv.org/abs/2402.16240)

    提出了一种名为HASH-CODE的高频感知分层对比选择编码方法，将图神经网络（GNNs）和预训练语言模型（PLMs）相结合，解决了在文本属性图上节点表示学习中的挑战。

    

    我们研究了在文本属性图（TAGs）上的节点表示学习，其中节点关联有文本信息。尽管最近关于图神经网络（GNNs）和预训练语言模型（PLMs）的研究展示了它们在编码网络和文本信号方面的强大能力，但对于精细地将这两种模型耦合在TAGs上的注意力较少。具体而言，现有的GNNs很少以一种情境化的方式对每个节点中的文本进行建模；现有的PLMs由于其序列架构，几乎无法应用于表征图结构。为了解决这些挑战，我们提出了HASH-CODE，一种高频感知的谱分层对比选择编码方法，将GNNs和PLMs整合到统一模型中。与之前的“级联架构”不同，直接在PLM之上添加GNN层的方法不同，我们的HASH-CODE依靠五个自监督优化目标，以促进彻底的相互学习。

    arXiv:2402.16240v1 Announce Type: new  Abstract: We investigate node representation learning on text-attributed graphs (TAGs), where nodes are associated with text information. Although recent studies on graph neural networks (GNNs) and pretrained language models (PLMs) have exhibited their power in encoding network and text signals, respectively, less attention has been paid to delicately coupling these two types of models on TAGs. Specifically, existing GNNs rarely model text in each node in a contextualized way; existing PLMs can hardly be applied to characterize graph structures due to their sequence architecture. To address these challenges, we propose HASH-CODE, a High-frequency Aware Spectral Hierarchical Contrastive Selective Coding method that integrates GNNs and PLMs into a unified model. Different from previous "cascaded architectures" that directly add GNN layers upon a PLM, our HASH-CODE relies on five self-supervised optimization objectives to facilitate thorough mutual e
    
[^2]: 交互式问答系统：文献综述

    Interactive Question Answering Systems: Literature Review

    [https://arxiv.org/abs/2209.01621](https://arxiv.org/abs/2209.01621)

    交互式问答系统是问答和对话系统的结合，用户可以用自然语言提问并与系统动态交互，获得更精确的结果。

    

    arXiv:2209.01621v2 公告类型: 替换-跨  摘要: 问答系统被公认为在网络上寻求信息的流行且有效的手段。在这种系统中，信息寻找者可以通过用自然语言提出问题来获得简洁的回答。交互式问答是最近提出的并越来越流行的解决方案，位于问答和对话系统的交集处。一方面，用户可以用普通语言提问并找到她问题的实际回答；另一方面，如果初始请求中存在多个可能的回复、很少或模棱两可，系统可以将问答会话延长为对话。通过允许用户提出更多问题，交互式问答使用户能够动态地与系统交互并获得更精确的结果。本综述提供了交互式问答系统的详细概述。

    arXiv:2209.01621v2 Announce Type: replace-cross  Abstract: Question answering systems are recognized as popular and frequently effective means of information seeking on the web. In such systems, information seekers can receive a concise response to their query by presenting their questions in natural language. Interactive question answering is a recently proposed and increasingly popular solution that resides at the intersection of question answering and dialogue systems. On the one hand, the user can ask questions in normal language and locate the actual response to her inquiry; on the other hand, the system can prolong the question-answering session into a dialogue if there are multiple probable replies, very few, or ambiguities in the initial request. By permitting the user to ask more questions, interactive question answering enables users to dynamically interact with the system and receive more precise results. This survey offers a detailed overview of the interactive question-ans
    
[^3]: 使用Transformer和自蒸馏的自监督学习旋转不变的3D点集特征

    Self-supervised Learning of Rotation-invariant 3D Point Set Features using Transformer and its Self-distillation. (arXiv:2308.04725v1 [cs.CV])

    [http://arxiv.org/abs/2308.04725](http://arxiv.org/abs/2308.04725)

    本文提出了一种使用Transformer和自蒸馏的自监督学习框架，用于从大量无标签的3D点集中获取准确且旋转不变的3D点集特征。

    

    在分析3D点集数据中，3D物体的旋转不变性是一个重要的属性。传统的具有旋转不变性的3D点集深度神经网络通常通过使用有标签的3D点集作为训练样本，通过监督学习获取准确的3D形状特征。然而，由于3D点集数据的快速增长和标注的高成本，需要一个从大量无标签的3D点集中学习旋转不变的3D形状特征的框架。本文提出了一种新颖的自监督学习框架，用于在对象级别获取准确且旋转不变的3D点集特征。我们提出的轻量级深度神经网络架构将输入的3D点集分解为多个全局尺度的区域（称为tokens），这些区域保留了构成3D对象的局部形状的空间布局。我们使用自注意机制来改进tokens，并将它们聚合成每个3D点集的表达性旋转不变特征。我们的深度神经网络通过自蒸馏机制进行有效训练。

    Invariance against rotations of 3D objects is an important property in analyzing 3D point set data. Conventional 3D point set DNNs having rotation invariance typically obtain accurate 3D shape features via supervised learning by using labeled 3D point sets as training samples. However, due to the rapid increase in 3D point set data and the high cost of labeling, a framework to learn rotation-invariant 3D shape features from numerous unlabeled 3D point sets is required. This paper proposes a novel self-supervised learning framework for acquiring accurate and rotation-invariant 3D point set features at object-level. Our proposed lightweight DNN architecture decomposes an input 3D point set into multiple global-scale regions, called tokens, that preserve the spatial layout of partial shapes composing the 3D object. We employ a self-attention mechanism to refine the tokens and aggregate them into an expressive rotation-invariant feature per 3D point set. Our DNN is effectively trained by u
    

