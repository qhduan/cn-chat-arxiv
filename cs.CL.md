# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [FP8-LM: Training FP8 Large Language Models.](http://arxiv.org/abs/2310.18313) | 本文提出了一种用于训练大语言模型的新型FP8自动混合精度框架，能够在不影响模型准确性的情况下显著减少内存使用并提高训练速度。 |
| [^2] | [GraphGPT: Graph Instruction Tuning for Large Language Models.](http://arxiv.org/abs/2310.13023) | 本论文提出了GraphGPT框架，它是一种面向图结构知识的大型语言模型，通过图指令调优实现高度泛化，即使在没有下游图数据的情况下也能在不同的下游数据集和任务上取得很好的效果。 |
| [^3] | [Recurrent Neural Language Models as Probabilistic Finite-state Automata.](http://arxiv.org/abs/2310.05161) | 本文研究了循环神经网络语言模型（RNN LMs）作为概率有限状态自动机的能力，并发现它们只能表示有限状态模型所能表达的概率分布的一个严格子集。 |
| [^4] | [GPT-Fathom: Benchmarking Large Language Models to Decipher the Evolutionary Path towards GPT-4 and Beyond.](http://arxiv.org/abs/2309.16583) | GPT-Fathom是一个用于评估大型语言模型的开源套件，它系统评估了10多个主要的语言模型，并提供了从GPT-3到GPT-4演化路径的宝贵见解。 |
| [^5] | [Question-Answering Approach to Evaluate Legal Summaries.](http://arxiv.org/abs/2309.15016) | 本文提出了一种利用GPT-4进行问答的法律摘要评估方法，通过生成一组问题-答案对来覆盖参考摘要中的主要信息，并通过GPT-4对参考摘要和生成摘要的答案进行评分，证明了该方法可以作为衡量摘要质量的有效工具。 |
| [^6] | [LLMR: Real-time Prompting of Interactive Worlds using Large Language Models.](http://arxiv.org/abs/2309.12276) | LLMR是一个用于实时创建和修改交互式混合现实体验的框架，通过利用大型语言模型和新颖的策略，它能够解决训练数据稀缺和设计目标复杂的问题，并在性能上超过标准的GPT-4。我们展示了LLMR的跨平台互操作性，并通过评估和用户研究证明了其对于生成和编辑各种对象、工具和场景的能力。 |
| [^7] | [Narrowing the Gap between Supervised and Unsupervised Sentence Representation Learning with Large Language Model.](http://arxiv.org/abs/2309.06453) | 本文通过实验比较了监督和无监督句子表示学习在训练过程中的行为，并探讨了如何缩小性能差距。 |
| [^8] | [SeaEval for Multilingual Foundation Models: From Cross-Lingual Alignment to Cultural Reasoning.](http://arxiv.org/abs/2309.04766) | SeaEval是一个评估多语言基础模型的基准测试，研究了模型在自然语言理解、推理以及对文化实践、细微差别和价值观的理解能力上的表现。重要发现包括模型在给出改写指令时行为各异，受到暴露偏差的影响，对于语义等价的多语言查询的回答不一致，以及模型在情感相关问题上的一致性不同。 |
| [^9] | [Exploring Transformer Extrapolation.](http://arxiv.org/abs/2307.10156) | 本文通过数学和实证分析，发现只要RPE的指数序列收敛，Transformer就具有长度外推的能力。从中导出了两种实践方法，并提出了一种新的理论感受野(TRF)来测量RPE的感受野。 |
| [^10] | [Communicative Agents for Software Development.](http://arxiv.org/abs/2307.07924) | 本文介绍了一种创新的软件开发范式，利用大型语言模型(LLMs)在整个软件开发过程中实现自然语言交流，消除了每个阶段需要专门模型的需求。该范式使用ChatDev作为一个虚拟聊天驱动的软件开发公司，通过设计、编码、测试和文档化四个阶段的代理人团队促进协作。 |
| [^11] | [PoetryDiffusion: Towards Joint Semantic and Metrical Manipulation in Poetry Generation.](http://arxiv.org/abs/2306.08456) | 本文提出了PoetryDiffusion模型，利用扩散模型生成诗歌，同时考虑了语义和韵律方面的控制，具有较高的实用性和创新性。 |
| [^12] | [ArtGPT-4: Artistic Vision-Language Understanding with Adapter-enhanced MiniGPT-4.](http://arxiv.org/abs/2305.07490) | ArtGPT-4是一种基于适配器增强的MiniGPT-4模型，专注于解决图像理解方面的问题，能够在短时间内训练出具备良好视觉语言理解能力的多模态模型。 |
| [^13] | [GPT-4 Technical Report.](http://arxiv.org/abs/2303.08774) | GPT-4是一个大规模多模态模型，可以接收图像和文本输入并产生文本输出，能够在各种专业和学术基准测试中表现出人类水平的表现，包括通过模拟的律师考试。该项目的核心组件是开发基础设施和优化方法，可在广泛的规模范围内表现预测性。 |
| [^14] | [Meta-Referential Games to Learn Compositional Learning Behaviours.](http://arxiv.org/abs/2207.08012) | 本论文提出了一种元元反游戏学习的方法来解决组合学习行为的问题，通过解决绑定问题来支持人工智能代理展示组合学习行为的能力。 |
| [^15] | [Graphmax for Text Generation.](http://arxiv.org/abs/2101.00153) | 本论文提出了图形最大化函数，用于任务特定的文本生成。该函数结合了语言模型的全局知识和特定场景语料库的局部知识，通过正则化的方式应用于传统的softmax函数，以充分利用共现信息，提高生成文本的主题一致性。 |

# 详细

[^1]: FP8-LM：训练FP8大语言模型

    FP8-LM: Training FP8 Large Language Models. (arXiv:2310.18313v1 [cs.LG])

    [http://arxiv.org/abs/2310.18313](http://arxiv.org/abs/2310.18313)

    本文提出了一种用于训练大语言模型的新型FP8自动混合精度框架，能够在不影响模型准确性的情况下显著减少内存使用并提高训练速度。

    

    本文探讨了用于高效训练大语言模型（LLMs）的FP8低比特数据格式。我们的关键洞察是，在LLM训练中，大多数变量（如梯度和优化器状态）可以使用低精度数据格式，而不会影响模型准确性，并且不需要改变超参数。具体地，我们提出了一种新的FP8自动混合精度框架用于训练LLMs。该框架为LLM的混合精度和分布式并行训练提供了三个级别的FP8利用。它逐步引入8位梯度，优化器状态和分布式学习。实验结果表明，在H100 GPU平台上训练GPT-175B模型期间，我们的FP8混合精度训练框架不仅实现了显著的42%的真实内存使用减少，而且比广泛采用的BF16框架（即Megatron-LM）运行速度快64%，比Nvidia Transformer Engine快17%。

    In this paper, we explore FP8 low-bit data formats for efficient training of large language models (LLMs). Our key insight is that most variables, such as gradients and optimizer states, in LLM training can employ low-precision data formats without compromising model accuracy and requiring no changes to hyper-parameters. Specifically, we propose a new FP8 automatic mixed-precision framework for training LLMs. This framework offers three levels of FP8 utilization to streamline mixed-precision and distributed parallel training for LLMs. It gradually incorporates 8-bit gradients, optimizer states, and distributed learning in an incremental manner. Experiment results show that, during the training of GPT-175B model on H100 GPU platform, our FP8 mixed-precision training framework not only achieved a remarkable 42% reduction in real memory usage but also ran 64% faster than the widely adopted BF16 framework (i.e., Megatron-LM), surpassing the speed of Nvidia Transformer Engine by 17%. This l
    
[^2]: GraphGPT: 大型语言模型的图指令调优

    GraphGPT: Graph Instruction Tuning for Large Language Models. (arXiv:2310.13023v1 [cs.CL])

    [http://arxiv.org/abs/2310.13023](http://arxiv.org/abs/2310.13023)

    本论文提出了GraphGPT框架，它是一种面向图结构知识的大型语言模型，通过图指令调优实现高度泛化，即使在没有下游图数据的情况下也能在不同的下游数据集和任务上取得很好的效果。

    

    通过图节点之间的递归信息交换和聚合，图神经网络（GNN）在理解图结构方面取得了进展。为了提高模型的健壮性，自监督学习（SSL）已经成为一种有前途的数据增强方法。然而，现有的用于生成预训练图嵌入的方法通常依赖于对特定下游任务标签进行微调，这限制了它们在标记数据稀缺或不可用的情况下的可用性。为了解决这个问题，我们的研究重点是提升图模型在具有挑战性的零样本学习场景中的泛化能力。受大型语言模型（LLM）的成功启发，我们的目标是开发一种面向图结构知识的LLM，即使没有来自下游图数据的任何信息，也能在不同的下游数据集和任务上实现高度泛化。在这项工作中，我们提出了GraphGPT框架，通过图指令调优将LLM与图结构知识对齐。

    Graph Neural Networks (GNNs) have advanced graph structure understanding via recursive information exchange and aggregation among graph nodes. To improve model robustness, self-supervised learning (SSL) has emerged as a promising approach for data augmentation. However, existing methods for generating pre-trained graph embeddings often rely on fine-tuning with specific downstream task labels, which limits their usability in scenarios where labeled data is scarce or unavailable. To address this, our research focuses on advancing the generalization capabilities of graph models in challenging zero-shot learning scenarios. Inspired by the success of large language models (LLMs), we aim to develop a graph-oriented LLM that can achieve high generalization across diverse downstream datasets and tasks, even without any information available from the downstream graph data. In this work, we present the GraphGPT framework that aligns LLMs with graph structural knowledge with a graph instruction t
    
[^3]: 循环神经语言模型作为概率有限状态自动机

    Recurrent Neural Language Models as Probabilistic Finite-state Automata. (arXiv:2310.05161v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.05161](http://arxiv.org/abs/2310.05161)

    本文研究了循环神经网络语言模型（RNN LMs）作为概率有限状态自动机的能力，并发现它们只能表示有限状态模型所能表达的概率分布的一个严格子集。

    

    通过以容易理解的形式来研究语言模型（LMs）可以使我们精确地描述它们的能力和局限性。先前的研究已经考察了循环神经网络（RNN）语言模型在识别无权重形式语言的能力。然而，LMs并不描述无权重形式语言，而是定义了对字符串的概率分布。在本研究中，我们研究了RNN LMs可以表示哪些类的概率分布，这使得我们可以更直接地陈述它们的能力。我们证明了简单的RNN等价于概率有限状态自动机的一个子类，因此只能模拟有限状态模型所能表达的概率分布的一个严格子集。此外，我们研究了用RNNs表示有限状态LMs的空间复杂度。我们证明了，为了表示一个任意确定的有限状态LMs，其中有$N$个状态且字符集为$\Sigma$的RNN requir

    Studying language models (LMs) in terms of well-understood formalisms allows us to precisely characterize their abilities and limitations. Previous work has investigated the representational capacity of recurrent neural network (RNN) LMs in terms of their capacity to recognize unweighted formal languages. However, LMs do not describe unweighted formal languages -- rather, they define probability distributions over strings. In this work, we study what classes of such probability distributions RNN LMs can represent, which allows us to make more direct statements about their capabilities. We show that simple RNNs are equivalent to a subclass of probabilistic finite-state automata, and can thus model a strict subset of probability distributions expressible by finite-state models. Furthermore, we study the space complexity of representing finite-state LMs with RNNs. We show that, to represent an arbitrary deterministic finite-state LM with $N$ states over an alphabet $\Sigma$, an RNN requir
    
[^4]: GPT-Fathom：评估大型语言模型以解析GPT-4及其后续版本的演化路径的基准测试

    GPT-Fathom: Benchmarking Large Language Models to Decipher the Evolutionary Path towards GPT-4 and Beyond. (arXiv:2309.16583v1 [cs.CL])

    [http://arxiv.org/abs/2309.16583](http://arxiv.org/abs/2309.16583)

    GPT-Fathom是一个用于评估大型语言模型的开源套件，它系统评估了10多个主要的语言模型，并提供了从GPT-3到GPT-4演化路径的宝贵见解。

    

    随着大型语言模型（LLMs）的快速进展，人们迫切需要一个全面的评估套件来评估它们的能力和局限性。现有的LLM排行榜通常参考其他论文中报告的得分，设置和提示不一致，这可能无意间鼓励选择有利的设置和提示以获得更好的结果。在这项工作中，我们引入了GPT-Fathom，这是一个基于OpenAI Evals构建的开源和可重复的LLM评估套件。我们在对齐的环境设置下系统评估了10多个主要的LLMs以及OpenAI的传统模型在20多个精选基准测试中的表现，涵盖了7个能力类别。我们对OpenAI早期模型的回顾性研究为我们揭示了从GPT-3到GPT-4的演化路径提供了宝贵的见解。目前，社区渴望了解GPT-3如何逐步改进到GPT-4，包括像添加代码数据是否提高了LLM的推理能力以及LLM能力的哪些方面等技术细节。

    With the rapid advancement of large language models (LLMs), there is a pressing need for a comprehensive evaluation suite to assess their capabilities and limitations. Existing LLM leaderboards often reference scores reported in other papers without consistent settings and prompts, which may inadvertently encourage cherry-picking favored settings and prompts for better results. In this work, we introduce GPT-Fathom, an open-source and reproducible LLM evaluation suite built on top of OpenAI Evals. We systematically evaluate 10+ leading LLMs as well as OpenAI's legacy models on 20+ curated benchmarks across 7 capability categories, all under aligned settings. Our retrospective study on OpenAI's earlier models offers valuable insights into the evolutionary path from GPT-3 to GPT-4. Currently, the community is eager to know how GPT-3 progressively improves to GPT-4, including technical details like whether adding code data improves LLM's reasoning capability, which aspects of LLM capabili
    
[^5]: 用问答方法评估法律摘要

    Question-Answering Approach to Evaluate Legal Summaries. (arXiv:2309.15016v1 [cs.CL])

    [http://arxiv.org/abs/2309.15016](http://arxiv.org/abs/2309.15016)

    本文提出了一种利用GPT-4进行问答的法律摘要评估方法，通过生成一组问题-答案对来覆盖参考摘要中的主要信息，并通过GPT-4对参考摘要和生成摘要的答案进行评分，证明了该方法可以作为衡量摘要质量的有效工具。

    

    传统的评估指标如ROUGE仅比较参考摘要和生成摘要之间的词汇重叠，而不考虑论点结构，而这对于法律摘要非常重要。在本文中，我们提出了一种新颖的法律摘要评估框架，利用GPT-4生成一组问题-答案对，涵盖参考摘要中的主要点和信息。然后，利用生成摘要回答参考摘要中的问题。最后，GPT-4对参考摘要和生成摘要的答案进行评分。我们检查了GPT-4评分与人工评分之间的相关性。结果表明，利用GPT-4的问答方法可以作为衡量摘要质量的有用工具。

    Traditional evaluation metrics like ROUGE compare lexical overlap between the reference and generated summaries without taking argumentative structure into account, which is important for legal summaries. In this paper, we propose a novel legal summarization evaluation framework that utilizes GPT-4 to generate a set of question-answer pairs that cover main points and information in the reference summary. GPT-4 is then used to generate answers based on the generated summary for the questions from the reference summary. Finally, GPT-4 grades the answers from the reference summary and the generated summary. We examined the correlation between GPT-4 grading with human grading. The results suggest that this question-answering approach with GPT-4 can be a useful tool for gauging the quality of the summary.
    
[^6]: LLMR：使用大型语言模型实时提示交互式世界的框架

    LLMR: Real-time Prompting of Interactive Worlds using Large Language Models. (arXiv:2309.12276v1 [cs.HC])

    [http://arxiv.org/abs/2309.12276](http://arxiv.org/abs/2309.12276)

    LLMR是一个用于实时创建和修改交互式混合现实体验的框架，通过利用大型语言模型和新颖的策略，它能够解决训练数据稀缺和设计目标复杂的问题，并在性能上超过标准的GPT-4。我们展示了LLMR的跨平台互操作性，并通过评估和用户研究证明了其对于生成和编辑各种对象、工具和场景的能力。

    

    我们提出了用于混合现实场景的大型语言模型(LLMR)，这是一个框架，用于实时创建和修改交互式混合现实体验。LLMR利用了新颖的策略来解决训练数据稀缺或设计目标需要合成内部动态、直观分析或高级交互的困难情况。我们的框架依赖于文本交互和Unity游戏引擎。通过融合场景理解、任务规划、自我调试和内存管理技术，LLMR在平均错误率上比标准的GPT-4提高了4倍。我们展示了LLMR与几个示例世界的跨平台互操作性，并通过多个创建和修改任务对其进行了评估，以展示它能够生成和编辑各种对象、工具和场景。最后，我们进行了一个有多样性的可用性研究（N=11），揭示了参与者对该系统有积极的体验，并愿意再次使用它。

    We present Large Language Model for Mixed Reality (LLMR), a framework for the real-time creation and modification of interactive Mixed Reality experiences using LLMs. LLMR leverages novel strategies to tackle difficult cases where ideal training data is scarce, or where the design goal requires the synthesis of internal dynamics, intuitive analysis, or advanced interactivity. Our framework relies on text interaction and the Unity game engine. By incorporating techniques for scene understanding, task planning, self-debugging, and memory management, LLMR outperforms the standard GPT-4 by 4x in average error rate. We demonstrate LLMR's cross-platform interoperability with several example worlds, and evaluate it on a variety of creation and modification tasks to show that it can produce and edit diverse objects, tools, and scenes. Finally, we conducted a usability study (N=11) with a diverse set that revealed participants had positive experiences with the system and would use it again.
    
[^7]: 缩小监督和无监督句子表示学习的差距：大规模语言模型

    Narrowing the Gap between Supervised and Unsupervised Sentence Representation Learning with Large Language Model. (arXiv:2309.06453v1 [cs.CL])

    [http://arxiv.org/abs/2309.06453](http://arxiv.org/abs/2309.06453)

    本文通过实验比较了监督和无监督句子表示学习在训练过程中的行为，并探讨了如何缩小性能差距。

    

    句子表示学习是自然语言处理中的一项基本任务，对比学习的句子嵌入（CSE）作为主流技术具有出色的性能。然而，在CSE中有一个有趣的现象，即监督和无监督方法之间存在显著的性能差距，即使它们的句子编码器和损失函数相同。本文通过实证实验回答“发生了什么导致了性能差距”和“如何缩小性能差距”的问题。我们首先通过彻底比较监督和无监督CSE在各自的训练过程中的行为来回答“发生了什么”这个问题。

    Sentence Representation Learning (SRL) is a fundamental task in Natural Language Processing (NLP), with Contrastive learning of Sentence Embeddings (CSE) as the mainstream technique due to its superior performance. An intriguing phenomenon in CSE is the significant performance gap between supervised and unsupervised methods, even when their sentence encoder and loss function are the same. Previous works attribute this performance gap to differences in two representation properties (alignment and uniformity). However, alignment and uniformity only measure the results, which means they cannot answer "What happens during the training process that leads to the performance gap?" and "How can the performance gap be narrowed?". In this paper, we conduct empirical experiments to answer these "What" and "How" questions. We first answer the "What" question by thoroughly comparing the behavior of supervised and unsupervised CSE during their respective training processes. From the comparison, We o
    
[^8]: SeaEval多语言基础模型：从跨语言对齐到文化推理

    SeaEval for Multilingual Foundation Models: From Cross-Lingual Alignment to Cultural Reasoning. (arXiv:2309.04766v1 [cs.CL])

    [http://arxiv.org/abs/2309.04766](http://arxiv.org/abs/2309.04766)

    SeaEval是一个评估多语言基础模型的基准测试，研究了模型在自然语言理解、推理以及对文化实践、细微差别和价值观的理解能力上的表现。重要发现包括模型在给出改写指令时行为各异，受到暴露偏差的影响，对于语义等价的多语言查询的回答不一致，以及模型在情感相关问题上的一致性不同。

    

    我们提出了一种用于多语言基础模型的SeaEval基准测试。除了表征这些模型如何理解和推理自然语言外，我们还研究了它们对文化实践、细微差别和价值观的理解能力。除了标准的准确度指标，我们还调查了基础模型在语义和多语言性维度上的脆弱性。我们的分析涵盖了开源和闭源模型，从而得到了在经典的自然语言处理任务、推理和文化理解方面的实证结果。重要发现包括：（1）大多数模型在给出改写指令时的行为各异；（2）许多模型仍然受到暴露偏差的影响（如位置偏差、大多数标签偏差）；（3）对于根源于事实、科学和常识知识的问题，预期在语义上等价的多语言查询应该得到一致的回答。然而，大多数模型在这些查询上表现出令人意外的不一致性；（4）多语言情况下，模型对于情感相关的问题表现出不同程度的一致性。

    We present SeaEval, a benchmark for multilingual foundation models. In addition to characterizing how these models understand and reason with natural language, we also investigate how well they comprehend cultural practices, nuances, and values. Alongside standard accuracy metrics, we investigate the brittleness of foundation models in the dimensions of semantics and multilinguality. Our analyses span both open-sourced and closed models, leading to empirical results across classic NLP tasks, reasoning, and cultural comprehension. Key findings indicate (1) Most models exhibit varied behavior when given paraphrased instructions. (2) Many models still suffer from exposure bias (e.g., positional bias, majority label bias). (3) For questions rooted in factual, scientific, and commonsense knowledge, consistent responses are expected across multilingual queries that are semantically equivalent. Yet, most models surprisingly demonstrate inconsistent performance on these queries. (4) Multilingu
    
[^9]: 探索Transformer外推

    Exploring Transformer Extrapolation. (arXiv:2307.10156v1 [cs.CL])

    [http://arxiv.org/abs/2307.10156](http://arxiv.org/abs/2307.10156)

    本文通过数学和实证分析，发现只要RPE的指数序列收敛，Transformer就具有长度外推的能力。从中导出了两种实践方法，并提出了一种新的理论感受野(TRF)来测量RPE的感受野。

    

    长度外推近期引起了相当大的关注，因为它允许transformers在训练中使用的序列长度之外进行测试。先前的研究表明，通过使用精心设计的相对位置编码(RPEs)可以实现这一属性。虽然这些方法在各种文集上表现良好，但对于长度外推的条件尚未得到研究。本文试图通过彻底的数学和实证分析确定哪种类型的RPEs可以实现长度外推。我们发现只要对应于RPE的指数收敛的序列，transformer一定具有这个属性。从这些条件中导出了两种实践方法，并在各种文集上进行了语言建模任务的研究。作为条件衍生的额外好处，我们推导出了一种新的理论感受野(TRF)，可以在不进行任何训练步骤的情况下测量RPE的感受野。进行了大量的实验。

    Length extrapolation has attracted considerable attention recently since it allows transformers to be tested on longer sequences than those used in training. Previous research has shown that this property can be attained by using carefully designed Relative Positional Encodings (RPEs). While these methods perform well on a variety of corpora, the conditions for length extrapolation have yet to be investigated. This paper attempts to determine what types of RPEs allow for length extrapolation through a thorough mathematical and empirical analysis. We discover that a transformer is certain to possess this property as long as the series that corresponds to the RPE's exponential converges. Two practices are derived from the conditions and examined in language modeling tasks on a variety of corpora. As a bonus from the conditions, we derive a new Theoretical Receptive Field (TRF) to measure the receptive field of RPEs without taking any training steps. Extensive experiments are conducted on
    
[^10]: 软件开发中的交流型代理

    Communicative Agents for Software Development. (arXiv:2307.07924v1 [cs.SE])

    [http://arxiv.org/abs/2307.07924](http://arxiv.org/abs/2307.07924)

    本文介绍了一种创新的软件开发范式，利用大型语言模型(LLMs)在整个软件开发过程中实现自然语言交流，消除了每个阶段需要专门模型的需求。该范式使用ChatDev作为一个虚拟聊天驱动的软件开发公司，通过设计、编码、测试和文档化四个阶段的代理人团队促进协作。

    

    软件工程是一个以微妙的直觉和咨询为特征的领域，决策过程复杂。深度学习的最新进展已经开始通过在软件开发的各个阶段实施精心设计来革新软件工程实践。在本文中，我们提出了一种创新的范式，通过自然语言交流，在整个软件开发过程中利用大型语言模型(LLMs)，简化和统一关键流程，从而消除了在每个阶段需要专门的模型的需要。这个范式的核心是ChatDev，一个虚拟的聊天驱动软件开发公司，它模仿了已经建立的瀑布模型，将开发过程细分为四个不同的时间阶段：设计、编码、测试和文档化。每个阶段都涉及一个团队的代理人，如程序员、代码审查人员和测试工程师，促进协作。

    Software engineering is a domain characterized by intricate decision-making processes, often relying on nuanced intuition and consultation. Recent advancements in deep learning have started to revolutionize software engineering practices through elaborate designs implemented at various stages of software development. In this paper, we present an innovative paradigm that leverages large language models (LLMs) throughout the entire software development process, streamlining and unifying key processes through natural language communication, thereby eliminating the need for specialized models at each phase. At the core of this paradigm lies ChatDev, a virtual chat-powered software development company that mirrors the established waterfall model, meticulously dividing the development process into four distinct chronological stages: designing, coding, testing, and documenting. Each stage engages a team of agents, such as programmers, code reviewers, and test engineers, fostering collaborativ
    
[^11]: PoetryDiffusion: 实现诗歌生成中的语义和韵律结合

    PoetryDiffusion: Towards Joint Semantic and Metrical Manipulation in Poetry Generation. (arXiv:2306.08456v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2306.08456](http://arxiv.org/abs/2306.08456)

    本文提出了PoetryDiffusion模型，利用扩散模型生成诗歌，同时考虑了语义和韵律方面的控制，具有较高的实用性和创新性。

    

    可控制文本生成是自然语言生成中具有挑战性和意义重大的领域。尤其是诗歌生成是一个典型的领域，对文本生成有着明确定义和严格的条件，是评估当前方法学的理想实验场。过去的研究成功地控制了诗歌生成的语义或韵律方面，但同时解决这两个方面仍然是一个挑战。在本文中，我们首次使用扩散模型来生成十四行诗和中国宋词，以应对这些挑战。就语义而言，我们的PoetryDiffusion模型基于扩散模型生成完整的句子或诗歌，全面考虑句子信息的整体性。这种方法增强了语义表达，使其与自回归模型和大型语言模型有所区别。就韵律控制而言，扩散生成和其约束控制模块的分离特性使我们能够灵活地控制韵律。

    Controllable text generation is a challenging and meaningful field in natural language generation (NLG). Especially, poetry generation is a typical one with well-defined and strict conditions for text generation which is an ideal playground for the assessment of current methodologies. While prior works succeeded in controlling either semantic or metrical aspects of poetry generation, simultaneously addressing both remains a challenge. In this paper, we pioneer the use of the Diffusion model for generating sonnets and Chinese SongCi poetry to tackle such challenges. In terms of semantics, our PoetryDiffusion model, built upon the Diffusion model, generates entire sentences or poetry by comprehensively considering the entirety of sentence information. This approach enhances semantic expression, distinguishing it from autoregressive and large language models (LLMs). For metrical control, the separation feature of diffusion generation and its constraint control module enable us to flexibly
    
[^12]: ArtGPT-4: 基于适配器增强的MiniGPT-4模型的艺术视觉语言理解

    ArtGPT-4: Artistic Vision-Language Understanding with Adapter-enhanced MiniGPT-4. (arXiv:2305.07490v1 [cs.CL])

    [http://arxiv.org/abs/2305.07490](http://arxiv.org/abs/2305.07490)

    ArtGPT-4是一种基于适配器增强的MiniGPT-4模型，专注于解决图像理解方面的问题，能够在短时间内训练出具备良好视觉语言理解能力的多模态模型。

    

    近年来，大型语言模型在自然语言处理领域取得了显著进展，比如ChatGPT和GPT-4等模型在多种语言任务上取得了惊人的能力。但是，对这样的大规模模型进行训练是具有挑战性的，而找到与模型规模匹配的数据集通常也很困难。微调和使用新方法训练参数较少的模型已经成为克服这些挑战的有效方法。MiniGPT-4模型便是其中之一，该模型通过运用新颖的预训练模型和革新性的培训策略实现了与GPT-4相当的视觉语言理解能力。但是，该模型在图像理解方面仍然面临一些挑战，特别是在艺术图片方面。ArtGPT-4是一种新型的多模态模型，旨在应对这些局限。ArtGPT-4使用Tesla A100设备对图像-文本对进行训练，仅用了约200GB的数据，在2小时内就能展示出图像。

    In recent years, large language models (LLMs) have made significant progress in natural language processing (NLP), with models like ChatGPT and GPT-4 achieving impressive capabilities in various linguistic tasks. However, training models on such a large scale is challenging, and finding datasets that match the model's scale is often difficult. Fine-tuning and training models with fewer parameters using novel methods have emerged as promising approaches to overcome these challenges. One such model is MiniGPT-4, which achieves comparable vision-language understanding to GPT-4 by leveraging novel pre-training models and innovative training strategies. However, the model still faces some challenges in image understanding, particularly in artistic pictures. A novel multimodal model called ArtGPT-4 has been proposed to address these limitations. ArtGPT-4 was trained on image-text pairs using a Tesla A100 device in just 2 hours, using only about 200 GB of data. The model can depict images wit
    
[^13]: GPT-4技术报告

    GPT-4 Technical Report. (arXiv:2303.08774v1 [cs.CL])

    [http://arxiv.org/abs/2303.08774](http://arxiv.org/abs/2303.08774)

    GPT-4是一个大规模多模态模型，可以接收图像和文本输入并产生文本输出，能够在各种专业和学术基准测试中表现出人类水平的表现，包括通过模拟的律师考试。该项目的核心组件是开发基础设施和优化方法，可在广泛的规模范围内表现预测性。

    

    我们报告了GPT-4的开发，它是一个可以接受图像和文本输入并产生文本输出的大规模多模态模型。虽然在许多现实场景中不如人类，但GPT-4在各种专业和学术基准测试中表现出人类水平的表现，包括通过模拟的律师考试，成绩排名在前10％左右。GPT-4是一个基于Transformer的模型，预训练用于预测文档中的下一个标记。后训练对齐过程提高了事实性和符合期望行为的性能指标。项目的核心组件是开发基础设施和优化方法，可在广泛的规模范围内表现预测性。这使我们能够准确预测GPT-4的某些性能方面，而这些性能是基于使用不超过GPT-4计算能力的1/1,000的模型训练的。

    We report the development of GPT-4, a large-scale, multimodal model which can accept image and text inputs and produce text outputs. While less capable than humans in many real-world scenarios, GPT-4 exhibits human-level performance on various professional and academic benchmarks, including passing a simulated bar exam with a score around the top 10% of test takers. GPT-4 is a Transformer-based model pre-trained to predict the next token in a document. The post-training alignment process results in improved performance on measures of factuality and adherence to desired behavior. A core component of this project was developing infrastructure and optimization methods that behave predictably across a wide range of scales. This allowed us to accurately predict some aspects of GPT-4's performance based on models trained with no more than 1/1,000th the compute of GPT-4.
    
[^14]: 元元反游戏学习组合学习行为

    Meta-Referential Games to Learn Compositional Learning Behaviours. (arXiv:2207.08012v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2207.08012](http://arxiv.org/abs/2207.08012)

    本论文提出了一种元元反游戏学习的方法来解决组合学习行为的问题，通过解决绑定问题来支持人工智能代理展示组合学习行为的能力。

    

    人类利用组合性从过去的经验中推广到新颖的经验。我们假设我们的经验可以分解为基本的原子组件，这些组件可以以新颖的方式重新组合，以支持我们参与新颖经验的能力。我们将这视为学习以组合方式泛化的能力，并将利用这种能力的行为称为组合学习行为（CLBs）。学习CLBs的一个核心问题是解决绑定问题（BP）。尽管这是人类轻松完成的智能壮举，但对于现有技术的人工智能代理来说并非如此。因此，为了构建能够与人类合作的人工智能代理，我们建议开发一个新的基准来研究代理商通过解决BP的领域无关版本来展示CLBs的能力。我们受到指代游戏的语言涌现和基础架构框架的启发，提出了一个元学习扩展方案

    Human beings use compositionality to generalise from past experiences to novel experiences. We assume a separation of our experiences into fundamental atomic components that can be recombined in novel ways to support our ability to engage with novel experiences. We frame this as the ability to learn to generalise compositionally, and we will refer to behaviours making use of this ability as compositional learning behaviours (CLBs). A central problem to learning CLBs is the resolution of a binding problem (BP). While it is another feat of intelligence that human beings perform with ease, it is not the case for state-of-the-art artificial agents. Thus, in order to build artificial agents able to collaborate with human beings, we propose to develop a novel benchmark to investigate agents' abilities to exhibit CLBs by solving a domain-agnostic version of the BP. We take inspiration from the language emergence and grounding framework of referential games and propose a meta-learning extensio
    
[^15]: 图形最大化用于文本生成

    Graphmax for Text Generation. (arXiv:2101.00153v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2101.00153](http://arxiv.org/abs/2101.00153)

    本论文提出了图形最大化函数，用于任务特定的文本生成。该函数结合了语言模型的全局知识和特定场景语料库的局部知识，通过正则化的方式应用于传统的softmax函数，以充分利用共现信息，提高生成文本的主题一致性。

    

    在文本生成中，一个大型语言模型（LM）仅基于上下文中先前选择的内容，使用softmax函数选择每个新词。然而，基于特定场景语料库的并发词的链接统计信息对选择下一个词是有价值的，可以帮助确保生成文本的主题与当前任务相一致。为了充分利用共现信息，我们提出了一种用于任务特定文本生成的图形最大化函数。使用基于图的正则化，图形最大化使最终词的选择由LM的全局知识和特定场景语料库的局部知识共同确定。传统的softmax函数通过图总变化（GTV）项进行正则化，将局部知识融入到LM中，并鼓励模型考虑特定场景语料库中单词之间的统计关系。所提出的图形最大化功能多样且易于使用。

    In text generation, a large language model (LM) makes a choice of each new word based only on the former selection of its context using the softmax function. Nevertheless, the link statistics information of concurrent words based on a scene-specific corpus is valuable in choosing the next word, which can help to ensure the topic of the generated text to be aligned with the current task. To fully explore the co-occurrence information,we propose a graphmax function for task-specific text generation. Using the graph-based regularization, graphmax enables the final word choice to be determined by both the global knowledge from the LM and the local knowledge from the scene-specific corpus. The traditional softmax function is regularized with a graph total variation (GTV) term, which incorporates the local knowledge into the LM and encourages the model to consider the statistical relationships between words in a scene-specific corpus. The proposed graphmax is versatile and can be readily plu
    

