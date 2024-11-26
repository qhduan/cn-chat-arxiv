# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Effective Two-Stage Knowledge Transfer for Multi-Entity Cross-Domain Recommendation](https://arxiv.org/abs/2402.19101) | 该研究提出了一种有效的两阶段跨实体跨域推荐知识传输方法，解决了多实体推荐中源实体数据分布不同和特征模式不对齐等重要问题。 |
| [^2] | [Word4Per: Zero-shot Composed Person Retrieval](https://arxiv.org/abs/2311.16515) | 提出了一个新任务：组合人员检索（CPR），旨在联合利用图像和文本信息进行目标人员检索，引入零样本组合人员检索（ZS-CPR）解决了CPR问题，提出了一个两阶段学习框架Word4Per。 |
| [^3] | [Policy-Gradient Training of Language Models for Ranking.](http://arxiv.org/abs/2310.04407) | 该论文提出了一种用于排序的语言模型的策略梯度训练算法Neural PG-RANK，通过将大规模语言模型实例化为Plackett-Luce排名策略，实现了对检索模型的原则性、端到端训练。 |

# 详细

[^1]: 有效的两阶段跨实体跨域推荐知识传输

    Effective Two-Stage Knowledge Transfer for Multi-Entity Cross-Domain Recommendation

    [https://arxiv.org/abs/2402.19101](https://arxiv.org/abs/2402.19101)

    该研究提出了一种有效的两阶段跨实体跨域推荐知识传输方法，解决了多实体推荐中源实体数据分布不同和特征模式不对齐等重要问题。

    

    近年来，电子商务平台上的推荐内容变得越来越丰富 -- 单个用户反馈可能包含多个实体，如销售产品、短视频和内容帖子。为了解决多实体推荐问题，一个直观的解决方案是采用基于共享网络的架构进行联合训练。这一想法是将一个类型实体（源实体）中提取的知识传输到另一个类型实体（目标实体）中。

    arXiv:2402.19101v1 Announce Type: cross  Abstract: In recent years, the recommendation content on e-commerce platforms has become increasingly rich -- a single user feed may contain multiple entities, such as selling products, short videos, and content posts. To deal with the multi-entity recommendation problem, an intuitive solution is to adopt the shared-network-based architecture for joint training. The idea is to transfer the extracted knowledge from one type of entity (source entity) to another (target entity). However, different from the conventional same-entity cross-domain recommendation, multi-entity knowledge transfer encounters several important issues: (1) data distributions of the source entity and target entity are naturally different, making the shared-network-based joint training susceptible to the negative transfer issue, (2) more importantly, the corresponding feature schema of each entity is not exactly aligned (e.g., price is an essential feature for selling product
    
[^2]: Word4Per: Zero-shot组合人员检索

    Word4Per: Zero-shot Composed Person Retrieval

    [https://arxiv.org/abs/2311.16515](https://arxiv.org/abs/2311.16515)

    提出了一个新任务：组合人员检索（CPR），旨在联合利用图像和文本信息进行目标人员检索，引入零样本组合人员检索（ZS-CPR）解决了CPR问题，提出了一个两阶段学习框架Word4Per。

    

    寻找特定人员具有极大的社会效益和安全价值，通常涉及视觉和文本信息的结合。本文提出了一个全新的任务，称为组合人员检索（CPR），旨在联合利用图像和文本信息进行目标人员检索。然而，监督CPR需要昂贵的手动注释数据集，而目前没有可用资源。为了解决这个问题，我们首先引入了零样本组合人员检索（ZS-CPR），利用现有的领域相关数据解决了CPR问题而不需要昂贵的注释。其次，为了学习ZS-CPR模型，我们提出了一个两阶段学习框架，即Word4Per，其中包含一个轻量级的文本反转网络。

    arXiv:2311.16515v2 Announce Type: replace-cross  Abstract: Searching for specific person has great social benefits and security value, and it often involves a combination of visual and textual information. Conventional person retrieval methods, whether image-based or text-based, usually fall short in effectively harnessing both types of information, leading to the loss of accuracy. In this paper, a whole new task called Composed Person Retrieval (CPR) is proposed to jointly utilize both image and text information for target person retrieval. However, the supervised CPR requires very costly manual annotation dataset, while there are currently no available resources. To mitigate this issue, we firstly introduce the Zero-shot Composed Person Retrieval (ZS-CPR), which leverages existing domain-related data to resolve the CPR problem without expensive annotations. Secondly, to learn ZS-CPR model, we propose a two-stage learning framework, Word4Per, where a lightweight Textual Inversion Netw
    
[^3]: 用于排序的语言模型的策略梯度训练

    Policy-Gradient Training of Language Models for Ranking. (arXiv:2310.04407v1 [cs.CL])

    [http://arxiv.org/abs/2310.04407](http://arxiv.org/abs/2310.04407)

    该论文提出了一种用于排序的语言模型的策略梯度训练算法Neural PG-RANK，通过将大规模语言模型实例化为Plackett-Luce排名策略，实现了对检索模型的原则性、端到端训练。

    

    文本检索在将事实知识纳入到语言处理流程中的决策过程中起着关键作用，从聊天式网页搜索到问答系统。当前最先进的文本检索模型利用预训练的大规模语言模型（LLM）以达到有竞争力的性能，但通过典型的对比损失训练基于LLM的检索器需要复杂的启发式算法，包括选择困难的负样本和使用额外的监督作为学习信号。这种依赖于启发式算法的原因是对比损失本身是启发式的，不能直接优化处理流程末端决策质量的下游指标。为了解决这个问题，我们引入了神经PG-RANK，一种新的训练算法，通过将LLM实例化为Plackett-Luce排名策略，学习排序。神经PG-RANK为检索模型的端到端训练提供了一种原则性方法，作为更大的决策系统的一部分进行训练。

    Text retrieval plays a crucial role in incorporating factual knowledge for decision making into language processing pipelines, ranging from chat-based web search to question answering systems. Current state-of-the-art text retrieval models leverage pre-trained large language models (LLMs) to achieve competitive performance, but training LLM-based retrievers via typical contrastive losses requires intricate heuristics, including selecting hard negatives and using additional supervision as learning signals. This reliance on heuristics stems from the fact that the contrastive loss itself is heuristic and does not directly optimize the downstream metrics of decision quality at the end of the processing pipeline. To address this issue, we introduce Neural PG-RANK, a novel training algorithm that learns to rank by instantiating a LLM as a Plackett-Luce ranking policy. Neural PG-RANK provides a principled method for end-to-end training of retrieval models as part of larger decision systems vi
    

