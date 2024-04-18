# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GOLF: Goal-Oriented Long-term liFe tasks supported by human-AI collaboration](https://arxiv.org/abs/2403.17089) | 该研究提出了GOLF框架，通过目标导向和长期规划增强LLMs的能力，以协助用户处理重要的生活决策。 |
| [^2] | [Leveraging Foundation Models for Content-Based Medical Image Retrieval in Radiology](https://arxiv.org/abs/2403.06567) | 基于内容的医学图像检索中，利用基础模型作为特征提取器，无需微调即可取得与专门模型竞争的性能，尤其在检索病理特征方面具有较大困难。 |
| [^3] | [JMLR: Joint Medical LLM and Retrieval Training for Enhancing Reasoning and Professional Question Answering Capability](https://arxiv.org/abs/2402.17887) | JMLR通过联合训练信息检索系统和大型语言模型，在医学领域提高问题回答系统性能，降低计算资源需求，增强模型利用医疗知识进行推理和回答问题的能力。 |
| [^4] | [GenSERP: Large Language Models for Whole Page Presentation](https://arxiv.org/abs/2402.14301) | 该论文提出了GenSERP框架，利用大型语言模型动态整理搜索结果并根据用户查询生成连贯的搜索引擎结果页面。 |
| [^5] | [Retrieval-Augmented Generation: Is Dense Passage Retrieval Retrieving?](https://arxiv.org/abs/2402.11035) | DPR微调预训练网络以增强查询和相关文本数据之间的嵌入对齐，发现训练中知识去中心化，但也揭示了模型内部知识的局限性 |
| [^6] | [UMAIR-FPS: User-aware Multi-modal Animation Illustration Recommendation Fusion with Painting Style](https://arxiv.org/abs/2402.10381) | UMAIR-FPS提出了一种新的用户感知多模态动画插画推荐系统，通过融合图像绘画风格特征和语义特征来增强表示。 |
| [^7] | [Causal Intervention for Fairness in Multi-behavior Recommendation](https://arxiv.org/abs/2209.04589) | 通过考虑多种用户行为来减轻流行度偏见，处理了多行为推荐中的公平问题。 |
| [^8] | [Gradient Flow of Energy: A General and Efficient Approach for Entity Alignment Decoding.](http://arxiv.org/abs/2401.12798) | 这篇论文介绍了一种能够解决实体对齐解码问题的新方法，该方法通过最小化能量来优化解码过程，以实现图同质性，并且仅依赖于实体嵌入，具有较高的通用性和效率。 |
| [^9] | [Recommender Systems in the Era of Large Language Models (LLMs).](http://arxiv.org/abs/2307.02046) | 大型语言模型在推荐系统中的应用已经带来了显著的改进，克服了传统DNN方法的限制，并提供了强大的语言理解、生成、推理和泛化能力。 |

# 详细

[^1]: GOLF：目标导向的长期生活任务，由人工智能协作支持

    GOLF: Goal-Oriented Long-term liFe tasks supported by human-AI collaboration

    [https://arxiv.org/abs/2403.17089](https://arxiv.org/abs/2403.17089)

    该研究提出了GOLF框架，通过目标导向和长期规划增强LLMs的能力，以协助用户处理重要的生活决策。

    

    ChatGPT等大型语言模型（LLMs）的出现彻底改变了人工智能交互和信息获取过程。利用LLMs作为搜索引擎的替代方案，用户现在可以访问根据其查询定制的摘要信息，显著减少了在导航大量信息资源时所带来的认知负荷。这种转变凸显了LLMs在重新定义信息获取范式方面的潜力。基于任务焦点信息检索和LLMs的任务规划能力，本研究将LLMs的能力范围扩展到支持用户导航长期和重要的生活任务。它引入了GOLF框架（目标导向的长期生活任务），侧重于增强LLMs通过目标定向和长期规划来协助用户做出重要的生活决策。该方法论包含了一个全面的类比实验

    arXiv:2403.17089v1 Announce Type: cross  Abstract: The advent of ChatGPT and similar large language models (LLMs) has revolutionized the human-AI interaction and information-seeking process. Leveraging LLMs as an alternative to search engines, users can now access summarized information tailored to their queries, significantly reducing the cognitive load associated with navigating vast information resources. This shift underscores the potential of LLMs in redefining information access paradigms. Drawing on the foundation of task-focused information retrieval and LLMs' task planning ability, this research extends the scope of LLM capabilities beyond routine task automation to support users in navigating long-term and significant life tasks. It introduces the GOLF framework (Goal-Oriented Long-term liFe tasks), which focuses on enhancing LLMs' ability to assist in significant life decisions through goal orientation and long-term planning. The methodology encompasses a comprehensive simul
    
[^2]: 利用基础模型进行放射学中基于内容的医学图像检索

    Leveraging Foundation Models for Content-Based Medical Image Retrieval in Radiology

    [https://arxiv.org/abs/2403.06567](https://arxiv.org/abs/2403.06567)

    基于内容的医学图像检索中，利用基础模型作为特征提取器，无需微调即可取得与专门模型竞争的性能，尤其在检索病理特征方面具有较大困难。

    

    Content-based image retrieval（CBIR）有望显著改善放射学中的诊断辅助和医学研究。我们提出利用视觉基础模型作为强大且多功能的现成特征提取器，用于基于内容的医学图像检索。通过在涵盖四种模态和161种病理学的160万张2D放射图像的全面数据集上对这些模型进行基准测试，我们发现弱监督模型表现优异，P@1可达0.594。这种性能不仅与专门化模型竞争，而且无需进行微调。我们的分析进一步探讨了检索病理学与解剖结构的挑战，表明准确检索病理特征更具挑战性。

    arXiv:2403.06567v1 Announce Type: cross  Abstract: Content-based image retrieval (CBIR) has the potential to significantly improve diagnostic aid and medical research in radiology. Current CBIR systems face limitations due to their specialization to certain pathologies, limiting their utility. In response, we propose using vision foundation models as powerful and versatile off-the-shelf feature extractors for content-based medical image retrieval. By benchmarking these models on a comprehensive dataset of 1.6 million 2D radiological images spanning four modalities and 161 pathologies, we identify weakly-supervised models as superior, achieving a P@1 of up to 0.594. This performance not only competes with a specialized model but does so without the need for fine-tuning. Our analysis further explores the challenges in retrieving pathological versus anatomical structures, indicating that accurate retrieval of pathological features presents greater difficulty. Despite these challenges, our
    
[^3]: JMLR：联合医疗LLM和检索训练以增强推理和专业问题回答能力

    JMLR: Joint Medical LLM and Retrieval Training for Enhancing Reasoning and Professional Question Answering Capability

    [https://arxiv.org/abs/2402.17887](https://arxiv.org/abs/2402.17887)

    JMLR通过联合训练信息检索系统和大型语言模型，在医学领域提高问题回答系统性能，降低计算资源需求，增强模型利用医疗知识进行推理和回答问题的能力。

    

    随着医疗数据的爆炸性增长和人工智能技术的快速发展，精准医学已经成为增强医疗服务质量和效率的关键。在这种背景下，大型语言模型（LLMs）在医疗知识获取和问题回答系统中发挥越来越重要的作用。为了进一步提高这些系统在医学领域的性能，我们介绍了一种创新方法，在微调阶段同时训练信息检索（IR）系统和LLM。我们称之为联合医疗LLM和检索训练（JMLR）的方法旨在克服传统模型在处理医学问题回答任务时面临的挑战。通过采用同步训练机制，JMLR减少了对计算资源的需求，并增强了模型利用医疗知识进行推理和回答问题的能力。

    arXiv:2402.17887v1 Announce Type: new  Abstract: With the explosive growth of medical data and the rapid development of artificial intelligence technology, precision medicine has emerged as a key to enhancing the quality and efficiency of healthcare services. In this context, Large Language Models (LLMs) play an increasingly vital role in medical knowledge acquisition and question-answering systems. To further improve the performance of these systems in the medical domain, we introduce an innovative method that jointly trains an Information Retrieval (IR) system and an LLM during the fine-tuning phase. This approach, which we call Joint Medical LLM and Retrieval Training (JMLR), is designed to overcome the challenges faced by traditional models in handling medical question-answering tasks. By employing a synchronized training mechanism, JMLR reduces the demand for computational resources and enhances the model's ability to leverage medical knowledge for reasoning and answering question
    
[^4]: GenSERP: 用于整个页面呈现的大型语言模型

    GenSERP: Large Language Models for Whole Page Presentation

    [https://arxiv.org/abs/2402.14301](https://arxiv.org/abs/2402.14301)

    该论文提出了GenSERP框架，利用大型语言模型动态整理搜索结果并根据用户查询生成连贯的搜索引擎结果页面。

    

    大语言模型（LLMs）的出现为最小化搜索引擎结果页面（SERP）的组织工作带来了机会。本文提出了GenSERP，这是一个利用LLMs和视觉在少样本设置中动态组织中间搜索结果的框架，包括生成的聊天答案、网站摘要、多媒体数据、知识面板等，并根据用户的查询以连贯的SERP布局呈现。

    arXiv:2402.14301v1 Announce Type: cross  Abstract: The advent of large language models (LLMs) brings an opportunity to minimize the effort in search engine result page (SERP) organization. In this paper, we propose GenSERP, a framework that leverages LLMs with vision in a few-shot setting to dynamically organize intermediate search results, including generated chat answers, website snippets, multimedia data, knowledge panels into a coherent SERP layout based on a user's query. Our approach has three main stages: (1) An information gathering phase where the LLM continuously orchestrates API tools to retrieve different types of items, and proposes candidate layouts based on the retrieved items, until it's confident enough to generate the final result. (2) An answer generation phase where the LLM populates the layouts with the retrieved content. In this phase, the LLM adaptively optimize the ranking of items and UX configurations of the SERP. Consequently, it assigns a location on the pag
    
[^5]: 密集通道检索：密集通道检索是否在检索中？

    Retrieval-Augmented Generation: Is Dense Passage Retrieval Retrieving?

    [https://arxiv.org/abs/2402.11035](https://arxiv.org/abs/2402.11035)

    DPR微调预训练网络以增强查询和相关文本数据之间的嵌入对齐，发现训练中知识去中心化，但也揭示了模型内部知识的局限性

    

    密集通道检索（DPR）是改进大型语言模型（LLM）性能的检索增强生成（RAG）范式中的第一步。 DPR微调预训练网络，以增强查询和相关文本数据之间的嵌入对齐。对DPR微调的深入理解将需要从根本上释放该方法的全部潜力。在这项工作中，我们通过使用探针、层激活分析和模型编辑的组合，机械地探索了DPR训练模型。我们的实验证明，DPR训练使网络中存储知识的方式去中心化，创建了访问相同信息的多个路径。我们还发现了这种训练风格的局限性：预训练模型的内部知识限制了检索模型可以检索的内容。这些发现为密集检索提出了一些可能的方向：（1）暴露DPR训练过程

    arXiv:2402.11035v1 Announce Type: new  Abstract: Dense passage retrieval (DPR) is the first step in the retrieval augmented generation (RAG) paradigm for improving the performance of large language models (LLM). DPR fine-tunes pre-trained networks to enhance the alignment of the embeddings between queries and relevant textual data. A deeper understanding of DPR fine-tuning will be required to fundamentally unlock the full potential of this approach. In this work, we explore DPR-trained models mechanistically by using a combination of probing, layer activation analysis, and model editing. Our experiments show that DPR training decentralizes how knowledge is stored in the network, creating multiple access pathways to the same information. We also uncover a limitation in this training style: the internal knowledge of the pre-trained model bounds what the retrieval model can retrieve. These findings suggest a few possible directions for dense retrieval: (1) expose the DPR training process 
    
[^6]: UMAIR-FPS：带绘画风格的用户感知多模态动画插画推荐融合

    UMAIR-FPS: User-aware Multi-modal Animation Illustration Recommendation Fusion with Painting Style

    [https://arxiv.org/abs/2402.10381](https://arxiv.org/abs/2402.10381)

    UMAIR-FPS提出了一种新的用户感知多模态动画插画推荐系统，通过融合图像绘画风格特征和语义特征来增强表示。

    

    高质量基于人工智能的图像生成模型的快速进步产生了大量的动漫插画。在海量数据中向用户推荐插画已成为一项具有挑战性和受欢迎的任务。然而，现有的动漫推荐系统侧重于文本特征，但仍需要整合图像特征。此外，大多数多模态推荐研究受到紧密耦合数据集的限制，限制了其对动漫插画的适用性。我们提出了带绘画风格的用户感知多模态动画插画推荐融合（UMAIR-FPS）来解决这些问题。在特征提取阶段，对于图像特征，我们首次结合图像绘画风格特征与语义特征来构建双输出图像编码器以增强表示。对于文本特征，我们基于Fine-tuning Sentence-Transformers获得文本嵌入，通过整合领域知识

    arXiv:2402.10381v1 Announce Type: cross  Abstract: The rapid advancement of high-quality image generation models based on AI has generated a deluge of anime illustrations. Recommending illustrations to users within massive data has become a challenging and popular task. However, existing anime recommendation systems have focused on text features but still need to integrate image features. In addition, most multi-modal recommendation research is constrained by tightly coupled datasets, limiting its applicability to anime illustrations. We propose the User-aware Multi-modal Animation Illustration Recommendation Fusion with Painting Style (UMAIR-FPS) to tackle these gaps. In the feature extract phase, for image features, we are the first to combine image painting style features with semantic features to construct a dual-output image encoder for enhancing representation. For text features, we obtain text embeddings based on fine-tuning Sentence-Transformers by incorporating domain knowledg
    
[^7]: 多行为推荐中的公平因果干预

    Causal Intervention for Fairness in Multi-behavior Recommendation

    [https://arxiv.org/abs/2209.04589](https://arxiv.org/abs/2209.04589)

    通过考虑多种用户行为来减轻流行度偏见，处理了多行为推荐中的公平问题。

    

    推荐系统通常从各种用户行为中学习用户兴趣，包括点击和点击后的行为（例如，点赞和收藏）。然而，这些行为不可避免地表现出流行度偏差，导致一些不公平问题：1）对于相似质量的物品，更受欢迎的物品会获得更多曝光；2）更糟糕的是，流行度较低的受欢迎物品可能会获得更多曝光。现有工作在减轻流行度偏差方面盲目消除偏见，通常忽略物品质量的影响。我们认为，不同用户行为之间（例如转化率）的关系实际上反映了物品质量。因此，为了处理不公平问题，我们提出通过考虑多种用户行为来减轻流行度偏见。在这项工作中，我们研究了多行为推荐中交互生成过程背后的因果关系。

    arXiv:2209.04589v2 Announce Type: replace-cross  Abstract: Recommender systems usually learn user interests from various user behaviors, including clicks and post-click behaviors (e.g., like and favorite). However, these behaviors inevitably exhibit popularity bias, leading to some unfairness issues: 1) for items with similar quality, more popular ones get more exposure; and 2) even worse the popular items with lower popularity might receive more exposure. Existing work on mitigating popularity bias blindly eliminates the bias and usually ignores the effect of item quality. We argue that the relationships between different user behaviors (e.g., conversion rate) actually reflect the item quality. Therefore, to handle the unfairness issues, we propose to mitigate the popularity bias by considering multiple user behaviors.   In this work, we examine causal relationships behind the interaction generation procedure in multi-behavior recommendation. Specifically, we find that: 1) item popula
    
[^8]: 能量的梯度流：实体对齐解码的通用高效方法

    Gradient Flow of Energy: A General and Efficient Approach for Entity Alignment Decoding. (arXiv:2401.12798v1 [cs.IR])

    [http://arxiv.org/abs/2401.12798](http://arxiv.org/abs/2401.12798)

    这篇论文介绍了一种能够解决实体对齐解码问题的新方法，该方法通过最小化能量来优化解码过程，以实现图同质性，并且仅依赖于实体嵌入，具有较高的通用性和效率。

    

    实体对齐（EA）是在集成多源知识图谱（KGs）中的关键过程，旨在识别这些图谱中的等价实体对。大多数现有方法将EA视为图表示学习任务，专注于增强图编码器。然而，在EA中解码过程-对于有效的操作和对齐准确性至关重要-得到了有限的关注，并且仍然针对特定数据集和模型架构进行定制，需要实体和额外的显式关系嵌入。这种特殊性限制了它的适用性，尤其是在基于GNN的模型中。为了填补这一空白，我们引入了一种新颖、通用和高效的EA解码方法，仅依赖于实体嵌入。我们的方法通过最小化狄利克雷能量来优化解码过程，在图内引导梯度流，以促进图同质性。梯度流的离散化产生了一种快速可扩展的方法，称为三元特征。

    Entity alignment (EA), a pivotal process in integrating multi-source Knowledge Graphs (KGs), seeks to identify equivalent entity pairs across these graphs. Most existing approaches regard EA as a graph representation learning task, concentrating on enhancing graph encoders. However, the decoding process in EA - essential for effective operation and alignment accuracy - has received limited attention and remains tailored to specific datasets and model architectures, necessitating both entity and additional explicit relation embeddings. This specificity limits its applicability, particularly in GNN-based models. To address this gap, we introduce a novel, generalized, and efficient decoding approach for EA, relying solely on entity embeddings. Our method optimizes the decoding process by minimizing Dirichlet energy, leading to the gradient flow within the graph, to promote graph homophily. The discretization of the gradient flow produces a fast and scalable approach, termed Triple Feature
    
[^9]: 大语言模型时代的推荐系统 (LLMs)

    Recommender Systems in the Era of Large Language Models (LLMs). (arXiv:2307.02046v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2307.02046](http://arxiv.org/abs/2307.02046)

    大型语言模型在推荐系统中的应用已经带来了显著的改进，克服了传统DNN方法的限制，并提供了强大的语言理解、生成、推理和泛化能力。

    

    随着电子商务和网络应用的繁荣，推荐系统（RecSys）已经成为我们日常生活中重要的组成部分，为用户提供个性化建议以满足其喜好。尽管深度神经网络（DNN）通过模拟用户-物品交互和整合文本侧信息在提升推荐系统方面取得了重要进展，但是DNN方法仍然存在一些限制，例如理解用户兴趣、捕捉文本侧信息的困难，以及在不同推荐场景中泛化和推理能力的不足等。与此同时，大型语言模型（LLMs）的出现（例如ChatGPT和GPT4）在自然语言处理（NLP）和人工智能（AI）领域引起了革命，因为它们在语言理解和生成的基本职责上有着卓越的能力，同时具有令人印象深刻的泛化和推理能力。

    With the prosperity of e-commerce and web applications, Recommender Systems (RecSys) have become an important component of our daily life, providing personalized suggestions that cater to user preferences. While Deep Neural Networks (DNNs) have made significant advancements in enhancing recommender systems by modeling user-item interactions and incorporating textual side information, DNN-based methods still face limitations, such as difficulties in understanding users' interests and capturing textual side information, inabilities in generalizing to various recommendation scenarios and reasoning on their predictions, etc. Meanwhile, the emergence of Large Language Models (LLMs), such as ChatGPT and GPT4, has revolutionized the fields of Natural Language Processing (NLP) and Artificial Intelligence (AI), due to their remarkable abilities in fundamental responsibilities of language understanding and generation, as well as impressive generalization and reasoning capabilities. As a result, 
    

