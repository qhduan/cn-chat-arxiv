# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LARA: Linguistic-Adaptive Retrieval-Augmented LLMs for Multi-Turn Intent Classification](https://arxiv.org/abs/2403.16504) | LARA是一个Linguistic-Adaptive Retrieval-Augmented Language Models（语言自适应检索增强LLMs），旨在通过结合微调过的较小模型与检索增强机制来提高多语言多轮意图分类任务的准确性，从而改善对话背景的理解。 |
| [^2] | [MultiContrievers: Analysis of Dense Retrieval Representations](https://arxiv.org/abs/2402.15925) | 该论文对稠密检索器的信息捕获进行了分析，探讨了其与语言模型的比较、信息提取的可行性以及提取性与性能、性别偏见的关系。 |
| [^3] | [Retrieval-Augmented Generation: Is Dense Passage Retrieval Retrieving?](https://arxiv.org/abs/2402.11035) | DPR微调预训练网络以增强查询和相关文本数据之间的嵌入对齐，发现训练中知识去中心化，但也揭示了模型内部知识的局限性 |

# 详细

[^1]: LARA：语言自适应检索增强LLMs用于多轮意图分类

    LARA: Linguistic-Adaptive Retrieval-Augmented LLMs for Multi-Turn Intent Classification

    [https://arxiv.org/abs/2403.16504](https://arxiv.org/abs/2403.16504)

    LARA是一个Linguistic-Adaptive Retrieval-Augmented Language Models（语言自适应检索增强LLMs），旨在通过结合微调过的较小模型与检索增强机制来提高多语言多轮意图分类任务的准确性，从而改善对话背景的理解。

    

    鉴于大型语言模型(LLMs)取得的显著成就，研究人员已经在文本分类任务中采用了上下文学习。然而，这些研究侧重于单语言、单轮分类任务。本文介绍了LARA（Linguistic-Adaptive Retrieval-Augmented Language Models），旨在增强多语言多轮分类任务的准确性，以适应聊天机器人交互中的众多意图。由于会话背景的复杂性和不断发展的性质，多轮意图分类尤为具有挑战性。LARA通过将微调过的较小模型与检索增强机制结合，嵌入LLMs的架构中来解决这些问题。这种整合使LARA能够动态利用过去的对话和相关意图，从而提高对上下文的理解。此外，我们的自适应检索技术增强了跨语言的能力。

    arXiv:2403.16504v1 Announce Type: new  Abstract: Following the significant achievements of large language models (LLMs), researchers have employed in-context learning for text classification tasks. However, these studies focused on monolingual, single-turn classification tasks. In this paper, we introduce LARA (Linguistic-Adaptive Retrieval-Augmented Language Models), designed to enhance accuracy in multi-turn classification tasks across six languages, accommodating numerous intents in chatbot interactions. Multi-turn intent classification is notably challenging due to the complexity and evolving nature of conversational contexts. LARA tackles these issues by combining a fine-tuned smaller model with a retrieval-augmented mechanism, integrated within the architecture of LLMs. This integration allows LARA to dynamically utilize past dialogues and relevant intents, thereby improving the understanding of the context. Furthermore, our adaptive retrieval techniques bolster the cross-lingual
    
[^2]: MultiContrievers: 稠密检索表示的分析

    MultiContrievers: Analysis of Dense Retrieval Representations

    [https://arxiv.org/abs/2402.15925](https://arxiv.org/abs/2402.15925)

    该论文对稠密检索器的信息捕获进行了分析，探讨了其与语言模型的比较、信息提取的可行性以及提取性与性能、性别偏见的关系。

    

    稠密检索器将源文档压缩为（可能是有损的）向量表示，然而目前对于失去和保留的信息以及它们如何影响下游任务的分析较少。我们进行了首次对比稠密检索器捕获的信息与它们基于的语言模型（如BERT与Contriever）之间的分析。我们使用25个MultiBert检查点作为随机初始化来训练MultiContrievers，这是一组25个contriever模型。我们测试特定信息（如性别和职业）是否可以从类似维基百科的文档的contriever向量中提取。我们通过信息论探测来衡量这种可提取性。然后我们研究了可提取性与性能、性别偏见之间的关系，以及这些结果对许多随机初始化和数据洗牌的敏感性。我们发现（1）contriever模型有显著增加的可提取性

    arXiv:2402.15925v1 Announce Type: cross  Abstract: Dense retrievers compress source documents into (possibly lossy) vector representations, yet there is little analysis of what information is lost versus preserved, and how it affects downstream tasks. We conduct the first analysis of the information captured by dense retrievers compared to the language models they are based on (e.g., BERT versus Contriever). We use 25 MultiBert checkpoints as randomized initialisations to train MultiContrievers, a set of 25 contriever models. We test whether specific pieces of information -- such as gender and occupation -- can be extracted from contriever vectors of wikipedia-like documents. We measure this extractability via information theoretic probing. We then examine the relationship of extractability to performance and gender bias, as well as the sensitivity of these results to many random initialisations and data shuffles. We find that (1) contriever models have significantly increased extracta
    
[^3]: 密集通道检索：密集通道检索是否在检索中？

    Retrieval-Augmented Generation: Is Dense Passage Retrieval Retrieving?

    [https://arxiv.org/abs/2402.11035](https://arxiv.org/abs/2402.11035)

    DPR微调预训练网络以增强查询和相关文本数据之间的嵌入对齐，发现训练中知识去中心化，但也揭示了模型内部知识的局限性

    

    密集通道检索（DPR）是改进大型语言模型（LLM）性能的检索增强生成（RAG）范式中的第一步。 DPR微调预训练网络，以增强查询和相关文本数据之间的嵌入对齐。对DPR微调的深入理解将需要从根本上释放该方法的全部潜力。在这项工作中，我们通过使用探针、层激活分析和模型编辑的组合，机械地探索了DPR训练模型。我们的实验证明，DPR训练使网络中存储知识的方式去中心化，创建了访问相同信息的多个路径。我们还发现了这种训练风格的局限性：预训练模型的内部知识限制了检索模型可以检索的内容。这些发现为密集检索提出了一些可能的方向：（1）暴露DPR训练过程

    arXiv:2402.11035v1 Announce Type: new  Abstract: Dense passage retrieval (DPR) is the first step in the retrieval augmented generation (RAG) paradigm for improving the performance of large language models (LLM). DPR fine-tunes pre-trained networks to enhance the alignment of the embeddings between queries and relevant textual data. A deeper understanding of DPR fine-tuning will be required to fundamentally unlock the full potential of this approach. In this work, we explore DPR-trained models mechanistically by using a combination of probing, layer activation analysis, and model editing. Our experiments show that DPR training decentralizes how knowledge is stored in the network, creating multiple access pathways to the same information. We also uncover a limitation in this training style: the internal knowledge of the pre-trained model bounds what the retrieval model can retrieve. These findings suggest a few possible directions for dense retrieval: (1) expose the DPR training process 
    

