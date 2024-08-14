# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Empirical Study of Training ID-Agnostic Multi-modal Sequential Recommenders](https://arxiv.org/abs/2403.17372) | 通过研究现有的多模态相关的顺序推荐方法，提炼出视觉编码器、文本编码器、多模态融合模块和顺序架构这四个核心组件。 |
| [^2] | [Enhancing Real-World Complex Network Representations with Hyperedge Augmentation](https://arxiv.org/abs/2402.13033) | 提出了一种新颖的图增强方法Hyperedge Augmentation (HyperAug)，通过构建直接从原始数据形成的虚拟超边，以解决现实世界复杂网络表示中高阶节点关系的问题 |
| [^3] | [Large language models can rate news outlet credibility.](http://arxiv.org/abs/2304.00228) | 本文评估了 ChatGPT 是否能够评估新闻机构的可信度，结果表明 ChatGPT 可以为不同语言和讽刺性资源的新闻机构提供评级及其背景说明，并且这些评级与人类专家的评级相关。LLMs可以成为事实检查应用程序中可信度评级的经济参考。 |

# 详细

[^1]: 训练独立于ID的多模态顺序推荐器的实证研究

    An Empirical Study of Training ID-Agnostic Multi-modal Sequential Recommenders

    [https://arxiv.org/abs/2403.17372](https://arxiv.org/abs/2403.17372)

    通过研究现有的多模态相关的顺序推荐方法，提炼出视觉编码器、文本编码器、多模态融合模块和顺序架构这四个核心组件。

    

    顺序推荐旨在基于历史交互来预测未来用户-物品交互。许多顺序推荐方法集中在用户ID和物品ID上，人类通过多模态信号（如文本和图像）感知世界的方式启发了研究人员探索如何构建不使用ID的多模态信息的顺序推荐。然而，多模态学习的复杂性体现在不同的特征提取器、融合方法和预训练模型中。因此，设计一个简单且通用的多模态顺序推荐（MMSR）框架仍然是一个巨大挑战。我们系统总结了现有的多模态相关的顺序推荐方法，并将精华提炼成四个核心组件：视觉编码器、文本编码器、多模态融合模块和顺序架构。沿着这些维度，我们剖析了模型设计，并回答了以下问题

    arXiv:2403.17372v1 Announce Type: new  Abstract: Sequential Recommendation (SR) aims to predict future user-item interactions based on historical interactions. While many SR approaches concentrate on user IDs and item IDs, the human perception of the world through multi-modal signals, like text and images, has inspired researchers to delve into constructing SR from multi-modal information without using IDs. However, the complexity of multi-modal learning manifests in diverse feature extractors, fusion methods, and pre-trained models. Consequently, designing a simple and universal \textbf{M}ulti-\textbf{M}odal \textbf{S}equential \textbf{R}ecommendation (\textbf{MMSR}) framework remains a formidable challenge. We systematically summarize the existing multi-modal related SR methods and distill the essence into four core components: visual encoder, text encoder, multimodal fusion module, and sequential architecture. Along these dimensions, we dissect the model designs, and answer the foll
    
[^2]: 用超边增强改进现实世界复杂网络表示

    Enhancing Real-World Complex Network Representations with Hyperedge Augmentation

    [https://arxiv.org/abs/2402.13033](https://arxiv.org/abs/2402.13033)

    提出了一种新颖的图增强方法Hyperedge Augmentation (HyperAug)，通过构建直接从原始数据形成的虚拟超边，以解决现实世界复杂网络表示中高阶节点关系的问题

    

    arXiv:2402.13033v1 公告类型: 新摘要: 图增强方法在改进图神经网络（GNNs）的性能和增强泛化能力中起着至关重要的作用。现有的图增强方法主要扰动图结构，通常限于成对节点关系。这些方法无法完全解决真实世界大规模网络的复杂性，这些网络通常涉及高阶节点关系，而不仅仅是成对关系。同时，由于缺乏可用于形成高阶边的数据，真实世界图数据集主要被建模为简单图。因此，将高阶边重新配置为图增强策略的一部分是一个有前途的研究路径，可解决前述问题。在本文中，我们提出了超边增强（HyperAug），一种新颖的图增强方法，直接从原始数据构建虚拟超边，并产生辅助节点。

    arXiv:2402.13033v1 Announce Type: new  Abstract: Graph augmentation methods play a crucial role in improving the performance and enhancing generalisation capabilities in Graph Neural Networks (GNNs). Existing graph augmentation methods mainly perturb the graph structures and are usually limited to pairwise node relations. These methods cannot fully address the complexities of real-world large-scale networks that often involve higher-order node relations beyond only being pairwise. Meanwhile, real-world graph datasets are predominantly modelled as simple graphs, due to the scarcity of data that can be used to form higher-order edges. Therefore, reconfiguring the higher-order edges as an integration into graph augmentation strategies lights up a promising research path to address the aforementioned issues. In this paper, we present Hyperedge Augmentation (HyperAug), a novel graph augmentation method that constructs virtual hyperedges directly form the raw data, and produces auxiliary nod
    
[^3]: 大型语言模型可评估新闻机构的可信度。

    Large language models can rate news outlet credibility. (arXiv:2304.00228v1 [cs.CL])

    [http://arxiv.org/abs/2304.00228](http://arxiv.org/abs/2304.00228)

    本文评估了 ChatGPT 是否能够评估新闻机构的可信度，结果表明 ChatGPT 可以为不同语言和讽刺性资源的新闻机构提供评级及其背景说明，并且这些评级与人类专家的评级相关。LLMs可以成为事实检查应用程序中可信度评级的经济参考。

    

    虽然大型语言模型（LLMs）在各种自然语言处理任务中表现出色，但它们容易产生幻象。现代最先进的聊天机器人，如新的 Bing，尝试通过直接从互联网收集信息来解决这个问题。在这种情况下，区分值得信赖的信息源对于向用户提供适当的准确性背景至关重要。本文评估了知名的LLM ChatGPT是否能够评估新闻机构的可信度。在适当的指导下，ChatGPT可以为不同语言和讽刺性资源的新闻机构提供评级及其背景说明。我们的结果表明，这些评级与人类专家的评级相关（Spearmam's $\rho=0.54, p<0.001$）。这些发现表明，LLMs可以成为事实检查应用程序中可信度评级的经济参考。未来的LLMs应增强它们的对齐性。

    Although large language models (LLMs) have shown exceptional performance in various natural language processing tasks, they are prone to hallucinations. State-of-the-art chatbots, such as the new Bing, attempt to mitigate this issue by gathering information directly from the internet to ground their answers. In this setting, the capacity to distinguish trustworthy sources is critical for providing appropriate accuracy contexts to users. Here we assess whether ChatGPT, a prominent LLM, can evaluate the credibility of news outlets. With appropriate instructions, ChatGPT can provide ratings for a diverse set of news outlets, including those in non-English languages and satirical sources, along with contextual explanations. Our results show that these ratings correlate with those from human experts (Spearmam's $\rho=0.54, p<0.001$). These findings suggest that LLMs could be an affordable reference for credibility ratings in fact-checking applications. Future LLMs should enhance their align
    

