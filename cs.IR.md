# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Knowledge-Enhanced Recommendation with User-Centric Subgraph Network](https://arxiv.org/abs/2403.14377) | 提出了一种基于知识增强的用户中心子图网络推荐方法，通过将用户-物品交互信息和知识图中的附加信息结合到子图学习中，实现了有效的个性化推荐。 |
| [^2] | [FIT-RAG: Black-Box RAG with Factual Information and Token Reduction](https://arxiv.org/abs/2403.14374) | FIT-RAG使用事实信息和令牌减少解决了黑匣子RAG系统中存在的事实信息忽视和令牌浪费问题 |
| [^3] | [Understanding the Ranking Loss for Recommendation with Sparse User Feedback](https://arxiv.org/abs/2403.14144) | 排序损失与二元交叉熵损失相结合可以提高点击率预测性能，特别是在稀疏正反馈情况下，通过生成更大的负样本梯度来改善分类能力。 |
| [^4] | [M3: A Multi-Task Mixed-Objective Learning Framework for Open-Domain Multi-Hop Dense Sentence Retrieval](https://arxiv.org/abs/2403.14074) | M3是一个多任务混合目标学习框架，旨在解决仅依赖对比学习可能导致的次优检索性能问题，并取得了在FEVER数据集上的最先进性能。 |
| [^5] | [Discrete Semantic Tokenization for Deep CTR Prediction](https://arxiv.org/abs/2403.08206) | 提出了一种新型的语义标记范式并引入离散语义标记化方法UIST，用于用户和项目表示，旨在将项目内容信息整合到点击率（CTR）预测模型中，实现快速训练和推断，并在保持内存占用的同时提高效率。 |
| [^6] | [EasyInstruct: An Easy-to-use Instruction Processing Framework for Large Language Models](https://arxiv.org/abs/2402.03049) | EasyInstruct是一个易于使用的用于大型语言模型的指令处理框架，通过模块化指令生成、选择和提示，并考虑它们的组合和交互，使指令处理更加方便和高效。 |
| [^7] | [Pushing the Limits: Concurrency Detection in Acyclic Sound Free-Choice Workflow Nets in $O(P^2 + T^2)$](https://arxiv.org/abs/2401.16097) | 本文提出了一种新的并发性检测算法，能够在$O(P^2 + T^2)$的时间复杂度下有效地检测无环无竞争自由选择工作流网中的并发性。 |
| [^8] | [RecMind: Large Language Model Powered Agent For Recommendation](https://arxiv.org/abs/2308.14296) | RecMind是一种LLM驱动的自主推荐代理，通过Self-Inspiring算法提高了规划能力，能够为零-shot个性化推荐提供支持。 |
| [^9] | [TensorBank:Tensor Lakehouse for Foundation Model Training.](http://arxiv.org/abs/2309.02094) | TensorBank是一个基于Tensor的湖仓库，能够以高速从云对象存储流式传输张量到GPU内存，并通过使用分层统计指标进行查询加速。 |

# 详细

[^1]: 基于知识增强的用户中心子图网络推荐

    Knowledge-Enhanced Recommendation with User-Centric Subgraph Network

    [https://arxiv.org/abs/2403.14377](https://arxiv.org/abs/2403.14377)

    提出了一种基于知识增强的用户中心子图网络推荐方法，通过将用户-物品交互信息和知识图中的附加信息结合到子图学习中，实现了有效的个性化推荐。

    

    推荐系统在当今各种平台上得到广泛实施，根据用户的偏好向他们推荐相关的物品。依赖用户-物品交互矩阵的经典方法存在局限性，特别是在新物品缺乏交互数据的情况下。基于知识图（KG）的推荐系统已经成为一种有前途的解决方案。然而，大多数基于KG的方法采用节点嵌入，这种方法不能为不同用户提供个性化推荐，也无法很好地推广到新物品。为了解决这些局限性，我们提出了基于知识增强的用户中心子图网络（KUCNet），这是一种利用图神经网络（GNN）进行有效推荐的子图学习方法。KUCNet为每个用户-物品对构建一个U-I子图，该子图捕获了用户-物品交互的历史信息和KG中提供的附加信息。基于注意力机制的GNN进入...

    arXiv:2403.14377v1 Announce Type: cross  Abstract: Recommendation systems, as widely implemented nowadays on various platforms, recommend relevant items to users based on their preferences. The classical methods which rely on user-item interaction matrices has limitations, especially in scenarios where there is a lack of interaction data for new items. Knowledge graph (KG)-based recommendation systems have emerged as a promising solution. However, most KG-based methods adopt node embeddings, which do not provide personalized recommendations for different users and cannot generalize well to the new items. To address these limitations, we propose Knowledge-enhanced User-Centric subgraph Network (KUCNet), a subgraph learning approach with graph neural network (GNN) for effective recommendation. KUCNet constructs a U-I subgraph for each user-item pair that captures both the historical information of user-item interactions and the side information provided in KG. An attention-based GNN is d
    
[^2]: FIT-RAG: 具有事实信息和令牌减少的黑匣子RAG

    FIT-RAG: Black-Box RAG with Factual Information and Token Reduction

    [https://arxiv.org/abs/2403.14374](https://arxiv.org/abs/2403.14374)

    FIT-RAG使用事实信息和令牌减少解决了黑匣子RAG系统中存在的事实信息忽视和令牌浪费问题

    

    由于参数数量庞大，对大型语言模型（LLMs）进行微调以更新长尾或过时知识在许多应用中都是不切实际的。为了避免微调，我们可以将LLM视为一个黑匣子（即，冻结LLM的参数）并增加一个检索增强生成（RAG）系统，即黑匣子RAG。最近，黑匣子RAG在知识密集型任务中取得了成功并引起了广泛关注。现有的黑匣子RAG方法通常微调检索器以迎合LLMs的偏好，并将所有检索到的文档串联在一起作为输入，但存在两个问题:（1）对事实信息的忽视。LLM偏好的文档可能不包含给定问题的事实信息，这可能会误导检索器，并损害黑匣子RAG的有效性;（2）令牌的浪费。简单地将所有检索到的文档串联在一起会带来...

    arXiv:2403.14374v1 Announce Type: new  Abstract: Due to the extraordinarily large number of parameters, fine-tuning Large Language Models (LLMs) to update long-tail or out-of-date knowledge is impractical in lots of applications. To avoid fine-tuning, we can alternatively treat a LLM as a black-box (i.e., freeze the parameters of the LLM) and augment it with a Retrieval-Augmented Generation (RAG) system, namely black-box RAG. Recently, black-box RAG has achieved success in knowledge-intensive tasks and has gained much attention. Existing black-box RAG methods typically fine-tune the retriever to cater to LLMs' preferences and concatenate all the retrieved documents as the input, which suffers from two issues: (1) Ignorance of Factual Information. The LLM preferred documents may not contain the factual information for the given question, which can mislead the retriever and hurt the effectiveness of black-box RAG; (2) Waste of Tokens. Simply concatenating all the retrieved documents brin
    
[^3]: 了解带有稀疏用户反馈的推荐排序损失

    Understanding the Ranking Loss for Recommendation with Sparse User Feedback

    [https://arxiv.org/abs/2403.14144](https://arxiv.org/abs/2403.14144)

    排序损失与二元交叉熵损失相结合可以提高点击率预测性能，特别是在稀疏正反馈情况下，通过生成更大的负样本梯度来改善分类能力。

    

    arXiv:2403.14144v1 公告类型: 新的 摘要: 在在线广告领域，点击率（CTR）预测具有重要意义。虽然许多现有方法将其视为二元分类问题，并利用二元交叉熵（BCE）作为优化目标，但最近的进展表明，将BCE损失与排序损失相结合可以显著提高性能。然而，这种组合损失的完整功效尚未完全理解。在本文中，我们揭示了在存在稀疏正反馈场景（如CTR预测）中与BCE损失相关的一个新挑战：负样本的梯度消失问题。随后，我们介绍了一个新的视角，强调了排序损失在CTR预测中的有效性，突出了它在负样本上生成更大的梯度，从而减轻了它们的优化问题，并导致了改善的分类能力。我们的观点得到了大量支持。

    arXiv:2403.14144v1 Announce Type: new  Abstract: Click-through rate (CTR) prediction holds significant importance in the realm of online advertising. While many existing approaches treat it as a binary classification problem and utilize binary cross entropy (BCE) as the optimization objective, recent advancements have indicated that combining BCE loss with ranking loss yields substantial performance improvements. However, the full efficacy of this combination loss remains incompletely understood. In this paper, we uncover a new challenge associated with BCE loss in scenarios with sparse positive feedback, such as CTR prediction: the gradient vanishing for negative samples. Subsequently, we introduce a novel perspective on the effectiveness of ranking loss in CTR prediction, highlighting its ability to generate larger gradients on negative samples, thereby mitigating their optimization issues and resulting in improved classification ability. Our perspective is supported by extensive the
    
[^4]: M3: 用于开放领域多跳密集句子检索的多任务混合目标学习框架

    M3: A Multi-Task Mixed-Objective Learning Framework for Open-Domain Multi-Hop Dense Sentence Retrieval

    [https://arxiv.org/abs/2403.14074](https://arxiv.org/abs/2403.14074)

    M3是一个多任务混合目标学习框架，旨在解决仅依赖对比学习可能导致的次优检索性能问题，并取得了在FEVER数据集上的最先进性能。

    

    在最近的研究中，对比学习已被证明是一种非常有效的表示学习方法，广泛用于密集检索。然而，我们发现仅依赖对比学习可能会导致次优的检索性能。另一方面，尽管许多检索数据集支持各种超越对比学习的学习目标，但在多任务学习场景中高效地组合它们可能具有挑战性。在本文中，我们介绍了M3，这是一个先进的递归多跳密集句子检索系统，它建立在一种新颖的多任务混合目标方法之上，用于密集文本表示学习，解决了上述挑战。我们的方法在大规模开放领域事实验证基准数据集FEVER上取得了最先进的性能。代码和数据可在以下链接获取: https://github.com/TonyBY/M3

    arXiv:2403.14074v1 Announce Type: cross  Abstract: In recent research, contrastive learning has proven to be a highly effective method for representation learning and is widely used for dense retrieval. However, we identify that relying solely on contrastive learning can lead to suboptimal retrieval performance. On the other hand, despite many retrieval datasets supporting various learning objectives beyond contrastive learning, combining them efficiently in multi-task learning scenarios can be challenging. In this paper, we introduce M3, an advanced recursive Multi-hop dense sentence retrieval system built upon a novel Multi-task Mixed-objective approach for dense text representation learning, addressing the aforementioned challenges. Our approach yields state-of-the-art performance on a large-scale open-domain fact verification benchmark dataset, FEVER. Code and data are available at: https://github.com/TonyBY/M3
    
[^5]: 用于深度CTR预测的离散语义标记化

    Discrete Semantic Tokenization for Deep CTR Prediction

    [https://arxiv.org/abs/2403.08206](https://arxiv.org/abs/2403.08206)

    提出了一种新型的语义标记范式并引入离散语义标记化方法UIST，用于用户和项目表示，旨在将项目内容信息整合到点击率（CTR）预测模型中，实现快速训练和推断，并在保持内存占用的同时提高效率。

    

    将项目内容信息整合到点击率（CTR）预测模型中仍然是一个挑战，尤其是在工业场景下的时间和空间约束下。传统的内容编码范式将用户和项目编码器直接整合到CTR模型中，优先考虑空间而非时间。相反，基于嵌入的范式将项目和用户语义转换为潜在嵌入，然后对其进行缓存，优先考虑空间而非时间。本文介绍了一种新型的语义标记范式，并提出了一种用于用户和项目表示的离散语义标记化方法，即UIST。UIST实现了快速的训练和推断，同时保持了保守的内存占用。具体而言，UIST将密集嵌入向量量化为较短的离散标记，并采用分层混合推断模块来衡量每个用户-项目标记对的贡献。我们在新闻数据集上的实验结果表明，UIST在提高效率的同时降低了内存消耗。

    arXiv:2403.08206v1 Announce Type: new  Abstract: Incorporating item content information into click-through rate (CTR) prediction models remains a challenge, especially with the time and space constraints of industrial scenarios. The content-encoding paradigm, which integrates user and item encoders directly into CTR models, prioritizes space over time. In contrast, the embedding-based paradigm transforms item and user semantics into latent embeddings and then caches them, prioritizes space over time. In this paper, we introduce a new semantic-token paradigm and propose a discrete semantic tokenization approach, namely UIST, for user and item representation. UIST facilitates swift training and inference while maintaining a conservative memory footprint. Specifically, UIST quantizes dense embedding vectors into discrete tokens with shorter lengths and employs a hierarchical mixture inference module to weigh the contribution of each user--item token pair. Our experimental results on news 
    
[^6]: EasyInstruct：一个易于使用的用于大型语言模型的指令处理框架

    EasyInstruct: An Easy-to-use Instruction Processing Framework for Large Language Models

    [https://arxiv.org/abs/2402.03049](https://arxiv.org/abs/2402.03049)

    EasyInstruct是一个易于使用的用于大型语言模型的指令处理框架，通过模块化指令生成、选择和提示，并考虑它们的组合和交互，使指令处理更加方便和高效。

    

    近年来，指令调整已经引起了越来越多的关注，并成为增强大型语言模型（LLMs）能力的一种关键技术。为了构建高质量的指令数据集，已经提出了许多指令处理方法，旨在在数据数量和数据质量之间达到精巧的平衡。然而，由于各种指令处理方法之间仍然存在不一致，目前没有标准的开源指令处理实现框架可供社区使用，这使得从业者无法进一步开发和推进。为了促进指令处理的研究和开发，我们提出了EasyInstruct，一个易于使用的用于LLMs的指令处理框架，它将指令生成、选择和提示模块化，并考虑它们的组合和交互。EasyInstruct已经在https://github.com/zjunlp/EasyInstruct上公开发布，并得到了积极维护。

    In recent years, instruction tuning has gained increasing attention and emerged as a crucial technique to enhance the capabilities of Large Language Models (LLMs). To construct high-quality instruction datasets, many instruction processing approaches have been proposed, aiming to achieve a delicate balance between data quantity and data quality. Nevertheless, due to inconsistencies that persist among various instruction processing methods, there is no standard open-source instruction processing implementation framework available for the community, which hinders practitioners from further developing and advancing. To facilitate instruction processing research and development, we present EasyInstruct, an easy-to-use instruction processing framework for LLMs, which modularizes instruction generation, selection, and prompting, while also considering their combination and interaction. EasyInstruct is publicly released and actively maintained at https://github.com/zjunlp/EasyInstruct, along 
    
[^7]: 在$O(P^2 + T^2)$的时间复杂度下检测无环无竞争自由选择工作流网中的并发性

    Pushing the Limits: Concurrency Detection in Acyclic Sound Free-Choice Workflow Nets in $O(P^2 + T^2)$

    [https://arxiv.org/abs/2401.16097](https://arxiv.org/abs/2401.16097)

    本文提出了一种新的并发性检测算法，能够在$O(P^2 + T^2)$的时间复杂度下有效地检测无环无竞争自由选择工作流网中的并发性。

    

    并发性是Petri网描述和模拟复杂系统行为的重要方面。知道哪些位置和变迁可以并行执行有助于理解网并启用分析技术和计算其他属性，如因果关系、排他性等。本文在$O\big((P+T)TP^2\big)$的时间内为活跃有界网和$O\big(P(P+T)^2\big)$的时间内为活跃有界自由选择网开发了算法，计算出所有并发位置。尽管这些算法具有相当不错的计算复杂度，但大量并发节点对可能导致长时间计算。

    arXiv:2401.16097v2 Announce Type: replace-cross  Abstract: Concurrency is an important aspect of Petri nets to describe and simulate the behavior of complex systems. Knowing which places and transitions could be executed in parallel helps to understand nets and enables analysis techniques and the computation of other properties, such as causality, exclusivity, etc.. All techniques based on concurrency detection depend on the efficiency of this detection methodology. Kovalyov and Esparza have developed algorithms that compute all concurrent places in $O\big((P+T)TP^2\big)$ for live and bounded nets (where $P$ and $T$ are the numbers of places and transitions) and in $O\big(P(P+T)^2\big)$ for live and bounded free-choice nets. Although these algorithms have a reasonably good computational complexity, large numbers of concurrent pairs of nodes may still lead to long computation times. This paper complements the palette of concurrency detection algorithms with the Concurrent Paths (CP) alg
    
[^8]: RecMind：大型语言模型驱动的推荐代理

    RecMind: Large Language Model Powered Agent For Recommendation

    [https://arxiv.org/abs/2308.14296](https://arxiv.org/abs/2308.14296)

    RecMind是一种LLM驱动的自主推荐代理，通过Self-Inspiring算法提高了规划能力，能够为零-shot个性化推荐提供支持。

    

    推荐系统（RS）通过深度学习取得了显著进展，但当前RS方法通常在特定任务数据集上训练和微调模型，限制了它们对新推荐任务的泛化能力以及利用外部知识的能力，因为模型规模和数据大小的限制。因此，我们设计了一种LLM驱动的自主推荐代理RecMind，能够利用外部知识，利用谨慎规划的工具为零-shot个性化推荐提供支持。我们提出了一种Self-Inspiring算法来提高规划能力。在每个中间步骤，LLM自我激励以考虑所有先前探索过的状态来规划下一步。这一机制极大地提高了模型理解和利用历史信息规划推荐的能力。我们评估了RecMind在各种推荐场景中的性能。

    arXiv:2308.14296v2 Announce Type: replace-cross  Abstract: While the recommendation system (RS) has advanced significantly through deep learning, current RS approaches usually train and fine-tune models on task-specific datasets, limiting their generalizability to new recommendation tasks and their ability to leverage external knowledge due to model scale and data size constraints. Thus, we designed an LLM-powered autonomous recommender agent, RecMind, which is capable of leveraging external knowledge, utilizing tools with careful planning to provide zero-shot personalized recommendations. We propose a Self-Inspiring algorithm to improve the planning ability. At each intermediate step, the LLM self-inspires to consider all previously explored states to plan for the next step. This mechanism greatly improves the model's ability to comprehend and utilize historical information in planning for recommendation. We evaluate RecMind's performance in various recommendation scenarios. Our exper
    
[^9]: TensorBank: 基于Tensor的湖仓库用于基础模型训练

    TensorBank:Tensor Lakehouse for Foundation Model Training. (arXiv:2309.02094v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2309.02094](http://arxiv.org/abs/2309.02094)

    TensorBank是一个基于Tensor的湖仓库，能够以高速从云对象存储流式传输张量到GPU内存，并通过使用分层统计指标进行查询加速。

    

    随着基础模型在自然语言之外的领域的兴起，存储和流式处理高维数据成为基础模型训练的关键需求。在本文中，我们介绍了TensorBank，一个能够基于复杂关系查询从云对象存储（COS）流式传输张量到GPU内存的百亿级张量湖仓库。我们使用分层统计指标（HSI）来加速查询。我们的架构允许使用HTTP范围读取来直接访问块级别的张量。一旦在GPU内存中，数据可以使用PyTorch转换进行转换。我们提供了一个通用的PyTorch数据集类型，配有相应的数据集工厂，用于将关系查询和请求的转换作为一个实例进行翻译。通过使用HSI，可以跳过不相关的块，而无需读取它们，因为这些索引包含不同层次分辨率级别上内容的统计信息。这是一个基于开放标准的有主观观点的架构。

    Storing and streaming high dimensional data for foundation model training became a critical requirement with the rise of foundation models beyond natural language. In this paper we introduce TensorBank, a petabyte scale tensor lakehouse capable of streaming tensors from Cloud Object Store (COS) to GPU memory at wire speed based on complex relational queries. We use Hierarchical Statistical Indices (HSI) for query acceleration. Our architecture allows to directly address tensors on block level using HTTP range reads. Once in GPU memory, data can be transformed using PyTorch transforms. We provide a generic PyTorch dataset type with a corresponding dataset factory translating relational queries and requested transformations as an instance. By making use of the HSI, irrelevant blocks can be skipped without reading them as those indices contain statistics on their content at different hierarchical resolution levels. This is an opinionated architecture powered by open standards and making h
    

