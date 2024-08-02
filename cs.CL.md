# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Hybrid Strategy for Chat Transcript Summarization](https://rss.arxiv.org/abs/2402.01510) | 这篇论文介绍了一种混合策略用于聊天记录摘要化，该策略首先结合了抽取和生成式摘要化技术，然后通过强化学习优化摘要的质量。这种方法在大规模部署的聊天记录摘要化中表现出了很好的效果。 |
| [^2] | [The opportunities and risks of large language models in mental health](https://arxiv.org/abs/2403.14814) | 大型语言模型在心理健康领域有望提供新颖的解决方案，但应注意其应用可能带来的风险，并积极采取策略减轻这些风险。 |
| [^3] | [A Hybrid Intelligence Method for Argument Mining](https://arxiv.org/abs/2403.09713) | 提出了一种混合(人类+AI)方法HyEnA，用于从意见文本中提取论点，结合了自动化处理速度和人类理解推理能力，在公民反馈语料库上取得了更高的覆盖率和准确率。 |
| [^4] | [Multitask Multilingual Model Adaptation with Featurized Low-Rank Mixtures](https://arxiv.org/abs/2402.17934) | 提出了一种名为FLix的新型参数高效微调方法，适用于多任务多语言调整，通过关联每个独特数据集特征与其低秩权重更新参数，实现了更好的泛化能力和性能表现。 |
| [^5] | [ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition](https://arxiv.org/abs/2402.15220) | ChunkAttention是一种前缀感知的自注意力模块，通过将键/值张量分解为较小的块并结构化到辅助前缀树中，实现了在运行时改善内存利用率的KV缓存，同时设计了两阶段分区算法以提高自注意力计算中的数据局部性。 |
| [^6] | [DefInt: A Default-interventionist Framework for Efficient Reasoning with Hybrid Large Language Models](https://arxiv.org/abs/2402.02563) | DefInt提出了一种默认干预框架，通过默认使用较小规模的语言模型生成推理思路，然后通过反思推理干预解决复杂推理问题，从而提高混合大型语言模型的效率和准确性。 |
| [^7] | [KGLens: A Parameterized Knowledge Graph Solution to Assess What an LLM Does and Doesn't Know](https://arxiv.org/abs/2312.11539) | KGLens 是一个旨在衡量知识图与大型语言模型（LLMs）之间对齐程度的框架，帮助找出LLMs相对于知识图的知识不足之处。 |
| [^8] | [Improving Small Language Models' Mathematical Reasoning via Equation-of-Thought Distillation.](http://arxiv.org/abs/2401.11864) | 本研究提出了思维方程蒸馏（EoTD）技术和集合思维蒸馏（ETD）框架，通过构建基于方程的表示和使用多个思维过程的推理数据集来改进小型语言模型（SLMs）的数学推理能力，实验结果表明，EoTD和ETD显著提升了SLMs的推理能力。 |
| [^9] | [Narrowing the Knowledge Evaluation Gap: Open-Domain Question Answering with Multi-Granularity Answers.](http://arxiv.org/abs/2401.04695) | 本研究提出了一种名为GRANOLA QA的评估设置，在开放领域问答中使用多粒度答案来评估预测的答案的准确性和信息量。作者提出了一种简单的方法来丰富现有数据集，并创建了一个多粒度版本的数据集。实验结果表明... |
| [^10] | [A Survey of Text Watermarking in the Era of Large Language Models.](http://arxiv.org/abs/2312.07913) | 本文综述了大语言模型时代的文本水印技术，包括不同技术的概述和比较、算法评估方法、应用场景以及当前挑战和未来发展方向。 |
| [^11] | [Instructive Dialogue Summarization with Query Aggregations.](http://arxiv.org/abs/2310.10981) | 传统的对话摘要方法无法考虑用户的特定兴趣，而指导对话摘要的引入可以帮助扩展对话摘要模型的能力。我们提出了一个三步方法来合成高质量的查询摘要三元组，并通过在多个数据集上训练一个统一模型来扩展对话摘要模型的能力。 |
| [^12] | [Psychoacoustic Challenges Of Speech Enhancement On VoIP Platforms.](http://arxiv.org/abs/2310.07161) | 本研究在VoIP通信领域中探索了声学转换的复杂性，并通过分析心理声学指标，揭示了语音增强对VoIP系统的影响。 |
| [^13] | [Geolocation Predicting of Tweets Using BERT-Based Models.](http://arxiv.org/abs/2303.07865) | 该论文提出基于BERT模型的推文地理位置预测方法，可以实现全球和美国上的中位误差分别小于30公里和15公里的定位精度。 |

# 详细

[^1]: 一种混合策略用于聊天记录摘要化

    A Hybrid Strategy for Chat Transcript Summarization

    [https://rss.arxiv.org/abs/2402.01510](https://rss.arxiv.org/abs/2402.01510)

    这篇论文介绍了一种混合策略用于聊天记录摘要化，该策略首先结合了抽取和生成式摘要化技术，然后通过强化学习优化摘要的质量。这种方法在大规模部署的聊天记录摘要化中表现出了很好的效果。

    

    文本摘要化是将一段文本压缩成较少句子的过程，同时保留其内容。在这个背景下，聊天记录是客户（来电者）和客服人员之间的数字或在线对话的文本副本。本文提出了一种本地开发的混合方法，首先结合抽取和生成式摘要化技术，对缺乏标点或未标点的聊天记录进行压缩，产生更易读的带标点摘要，然后通过强化学习优化摘要的整体质量。广泛的测试、评估、比较和验证证明了这种方法在没有手动生成的参考摘要的情况下，对于大规模部署的聊天记录摘要化的有效性。

    Text summarization is the process of condensing a piece of text to fewer sentences, while still preserving its content. Chat transcript, in this context, is a textual copy of a digital or online conversation between a customer (caller) and agent(s). This paper presents an indigenously (locally) developed hybrid method that first combines extractive and abstractive summarization techniques in compressing ill-punctuated or un-punctuated chat transcripts to produce more readable punctuated summaries and then optimizes the overall quality of summarization through reinforcement learning. Extensive testing, evaluations, comparisons, and validation have demonstrated the efficacy of this approach for large-scale deployment of chat transcript summarization, in the absence of manually generated reference (annotated) summaries.
    
[^2]: 大型语言模型在心理健康领域的机会和风险

    The opportunities and risks of large language models in mental health

    [https://arxiv.org/abs/2403.14814](https://arxiv.org/abs/2403.14814)

    大型语言模型在心理健康领域有望提供新颖的解决方案，但应注意其应用可能带来的风险，并积极采取策略减轻这些风险。

    

    全球心理健康问题的发生率正在上升，人们越来越意识到现有的心理保健模式无法充分扩展以满足需求。随着大型语言模型（LLMs）的出现，人们对它们具有创造新颖、大规模解决方案以支持心理健康的承诺感到乐观。尽管它们还处于初期阶段，LLMs已被应用于与心理健康相关的任务。本综述总结了已有文献中关于利用LLMs提供心理健康教育、评估和干预的努力，并突出了每个领域中产生积极影响的关键机会。然后，我们强调了将LLMs应用于心理健康领域所伴随的风险，并鼓励采用策略来减轻这些风险。对于心理健康支持的迫切需求必须与负责任的心理健康LLMs的开发、测试和部署相平衡。特别关键的是确保心理健康...

    arXiv:2403.14814v1 Announce Type: cross  Abstract: Global rates of mental health concerns are rising and there is increasing realization that existing models of mental healthcare will not adequately expand to meet the demand. With the emergence of large language models (LLMs) has come great optimism regarding their promise to create novel, large-scale solutions to support mental health. Despite their nascence, LLMs have already been applied to mental health-related tasks. In this review, we summarize the extant literature on efforts to use LLMs to provide mental health education, assessment, and intervention and highlight key opportunities for positive impact in each area. We then highlight risks associated with LLMs application to mental health and encourage adoption of strategies to mitigate these risks. The urgent need for mental health support must be balanced with responsible development, testing, and deployment of mental health LLMs. Especially critical is ensuring that mental he
    
[^3]: 一种用于论证挖掘的混合智能方法

    A Hybrid Intelligence Method for Argument Mining

    [https://arxiv.org/abs/2403.09713](https://arxiv.org/abs/2403.09713)

    提出了一种混合(人类+AI)方法HyEnA，用于从意见文本中提取论点，结合了自动化处理速度和人类理解推理能力，在公民反馈语料库上取得了更高的覆盖率和准确率。

    

    大规模调查工具能够收集公民反馈意见语料库。从庞大且嘈杂的意见集中提取关键论点有助于快速准确地理解意见。完全自动化的方法可以提取论点，但(1)需要大规模标记数据集，导致较高的注释成本; (2)对已知观点效果良好，但对新颖观点效果欠佳。我们提出了HyEnA，一种混合(人类+AI)方法，用于从主观文本中提取论点，结合了自动化处理的速度和人类的理解和推理能力。我们在三个公民反馈语料库上评估了HyEnA。我们发现，一方面，与一组各种意见进行比较时，HyEnA在高覆盖率和准确率方面优于最先进的自动化方法，证实了人类洞察的必要性。另一方面，HyEnA需要较少的人力工作量，且不会牺牲质量。

    arXiv:2403.09713v1 Announce Type: new  Abstract: Large-scale survey tools enable the collection of citizen feedback in opinion corpora. Extracting the key arguments from a large and noisy set of opinions helps in understanding the opinions quickly and accurately. Fully automated methods can extract arguments but (1) require large labeled datasets that induce large annotation costs and (2) work well for known viewpoints, but not for novel points of view. We propose HyEnA, a hybrid (human + AI) method for extracting arguments from opinionated texts, combining the speed of automated processing with the understanding and reasoning capabilities of humans. We evaluate HyEnA on three citizen feedback corpora. We find that, on the one hand, HyEnA achieves higher coverage and precision than a state-of-the-art automated method when compared to a common set of diverse opinions, justifying the need for human insight. On the other hand, HyEnA requires less human effort and does not compromise quali
    
[^4]: 用特征化低秩混合进行多任务多语言模型适应

    Multitask Multilingual Model Adaptation with Featurized Low-Rank Mixtures

    [https://arxiv.org/abs/2402.17934](https://arxiv.org/abs/2402.17934)

    提出了一种名为FLix的新型参数高效微调方法，适用于多任务多语言调整，通过关联每个独特数据集特征与其低秩权重更新参数，实现了更好的泛化能力和性能表现。

    

    预训练大型语言模型（LLMs）适应数十甚至数百种人类语言的各种下游任务在计算上是昂贵的。参数高效微调（PEFT）通过只调整少量参数显著减少了适应成本。然而，直接将像 LoRA（Hu 等人，2022）这样的 PEFT 方法应用于不同数据集混合可能导致性能次优，原因在于有限的参数容量和不同数据集之间的负面互相影响。在这项工作中，我们提出了特征化低秩混合（FLix），这是一种针对有效的多任务多语言调整的新型 PEFT 方法。FLix将每个独特数据集特征（例如数据集的语言或任务）与其自己的低秩权重更新参数相关联。通过为每个数据集组合特定于特征的参数，FLix能够适应多种数据集混合，并更好地泛化到未见数据集。我们的实验表明，FLix 可以在提供更好性能的同时显著减少适应成本。

    arXiv:2402.17934v1 Announce Type: cross  Abstract: Adapting pretrained large language models (LLMs) to various downstream tasks in tens or hundreds of human languages is computationally expensive. Parameter-efficient fine-tuning (PEFT) significantly reduces the adaptation cost, by tuning only a small amount of parameters. However, directly applying PEFT methods such as LoRA (Hu et al., 2022) on diverse dataset mixtures could lead to suboptimal performance due to limited parameter capacity and negative interference among different datasets. In this work, we propose Featurized Low-rank Mixtures (FLix), a novel PEFT method designed for effective multitask multilingual tuning. FLix associates each unique dataset feature, such as the dataset's language or task, with its own low-rank weight update parameters. By composing feature-specific parameters for each dataset, FLix can accommodate diverse dataset mixtures and generalize better to unseen datasets. Our experiments show that FLix leads t
    
[^5]: ChunkAttention: 具有前缀感知KV缓存和两阶段分区的高效自注意力

    ChunkAttention: Efficient Self-Attention with Prefix-Aware KV Cache and Two-Phase Partition

    [https://arxiv.org/abs/2402.15220](https://arxiv.org/abs/2402.15220)

    ChunkAttention是一种前缀感知的自注意力模块，通过将键/值张量分解为较小的块并结构化到辅助前缀树中，实现了在运行时改善内存利用率的KV缓存，同时设计了两阶段分区算法以提高自注意力计算中的数据局部性。

    

    自注意力是大型语言模型（LLMs）的重要组成部分，但对于长序列来说是推理延迟的一个显著来源。在多租户LLMs服务场景中，通过利用多个LLM请求在前缀中共享系统提示的概率，可以优化自注意力的计算和内存操作成本。本文介绍了ChunkAttention，一种具有前缀感知的自注意力模块，可以在运行时检测多个请求之间匹配的提示前缀，并共享它们的键/值张量以改进KV缓存的内存利用率。这是通过将整体键/值张量分解为较小的块，并将它们结构化到辅助前缀树中来实现的。因此，在基于前缀树的KV缓存之上，我们设计了一个高效的自注意力内核，其中实现了两阶段分区算法，以改善自注意力计算中的数据局部性。

    arXiv:2402.15220v1 Announce Type: cross  Abstract: Self-attention is an essential component of large language models(LLMs) but a significant source of inference latency for long sequences. In multi-tenant LLMs serving scenarios, the compute and memory operation cost of self-attention can be optimized by using the probability that multiple LLM requests have shared system prompts in prefixes. In this paper, we introduce ChunkAttention, a prefix-aware self-attention module that can detect matching prompt prefixes across multiple requests and share their key/value tensors in memory at runtime to improve the memory utilization of KV cache. This is achieved by breaking monolithic key/value tensors into smaller chunks and structuring them into the auxiliary prefix tree. Consequently, on top of the prefix-tree based KV cache, we design an efficient self-attention kernel, where a two-phase partition algorithm is implemented to improve the data locality during self-attention computation in the p
    
[^6]: DefInt：一种用于高效处理混合大型语言模型推理的默认干预框架

    DefInt: A Default-interventionist Framework for Efficient Reasoning with Hybrid Large Language Models

    [https://arxiv.org/abs/2402.02563](https://arxiv.org/abs/2402.02563)

    DefInt提出了一种默认干预框架，通过默认使用较小规模的语言模型生成推理思路，然后通过反思推理干预解决复杂推理问题，从而提高混合大型语言模型的效率和准确性。

    

    大型语言模型（LLMs）在各种任务中展示出令人印象深刻的新能力，但在处理复杂推理问题方面仍面临挑战。以往的研究如连锁推理（CoT）和思维树（ToT）主要关注提高准确性，但忽视了不断增加的标记成本，这对于具有巨大解空间的开放性实际任务来说可能特别问题。受人类认知的双过程理论的启发，我们提出了一种默认干预框架（DefInt），以释放混合LLMs的协同潜力。默认情况下，DefInt使用较小规模的语言模型生成低成本的推理思路，类似于“系统1”产生的快速直觉。如果这些直觉被认为低置信度，则DefInt将调用放大的语言模型的反思推理作为“系统2”的干预，可以覆盖默认思考并纠正推理过程。实验在五个实际数据集上展示了DefInt论文中的有效性。

    Large language models (LLMs) have shown impressive emergent abilities in a wide range of tasks, but still face challenges in handling complex reasoning problems. Previous works like chain-of-thought (CoT) and tree-of-thoughts(ToT) have predominately focused on enhancing accuracy, but overlook the rapidly increasing token cost, which could be particularly problematic for open-ended real-world tasks with huge solution spaces. Motivated by the dual process theory of human cognition, we propose a Default-Interventionist framework (DefInt) to unleash the synergistic potential of hybrid LLMs. By default, DefInt uses smaller-scale language models to generate low-cost reasoning thoughts, which resembles the fast intuitions produced by System 1. If the intuitions are considered with low confidence, DefInt will invoke the reflective reasoning of scaled-up language models as the intervention of System 2, which can override the default thoughts and rectify the reasoning process. Experiments on fiv
    
[^7]: KGLens：一个参数化知识图解决方案，用于评估LLM知道和不知道的内容

    KGLens: A Parameterized Knowledge Graph Solution to Assess What an LLM Does and Doesn't Know

    [https://arxiv.org/abs/2312.11539](https://arxiv.org/abs/2312.11539)

    KGLens 是一个旨在衡量知识图与大型语言模型（LLMs）之间对齐程度的框架，帮助找出LLMs相对于知识图的知识不足之处。

    

    衡量知识图（KG）与大型语言模型（LLMs）之间的对齐程度是评估事实性并识别LLMs的知识盲点的有效方法。然而，这种方法面临两个主要挑战，包括将KGs转化为自然语言和高效评估这些广泛且复杂的结构。在本文中，我们提出了KGLens--一个旨在衡量KGs和LLMs之间对齐程度，并找出LLMs相对于KGs的知识缺陷的新颖框架。KGLens具有一个图引导的问题生成器，用于将KGs转化为自然语言，以及一个基于参数化KG结构的精心设计的采样策略，以加快KG的遍历。我们使用来自Wikidata的三个领域特定KG进行实验，这些KG包括超过19,000条边，700个关系和21,000个实体。我们跨越8个LLMs的分析表明，KGLens不仅

    arXiv:2312.11539v2 Announce Type: replace  Abstract: Measuring the alignment between a Knowledge Graph (KG) and Large Language Models (LLMs) is an effective method to assess the factualness and identify the knowledge blind spots of LLMs. However, this approach encounters two primary challenges including the translation of KGs into natural language and the efficient evaluation of these extensive and complex structures. In this paper, we present KGLens--a novel framework aimed at measuring the alignment between KGs and LLMs, and pinpointing the LLMs' knowledge deficiencies relative to KGs. KGLens features a graph-guided question generator for converting KGs into natural language, along with a carefully designed sampling strategy based on parameterized KG structure to expedite KG traversal. We conducted experiments using three domain-specific KGs from Wikidata, which comprise over 19,000 edges, 700 relations, and 21,000 entities. Our analysis across eight LLMs reveals that KGLens not only
    
[^8]: 通过思维方程蒸馏改进小型语言模型的数学推理能力

    Improving Small Language Models' Mathematical Reasoning via Equation-of-Thought Distillation. (arXiv:2401.11864v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2401.11864](http://arxiv.org/abs/2401.11864)

    本研究提出了思维方程蒸馏（EoTD）技术和集合思维蒸馏（ETD）框架，通过构建基于方程的表示和使用多个思维过程的推理数据集来改进小型语言模型（SLMs）的数学推理能力，实验结果表明，EoTD和ETD显著提升了SLMs的推理能力。

    

    本研究解决了将先进的大型语言模型（LLMs）的数学推理能力压缩到具有小于十亿参数的小型语言模型（SLMs）中的挑战，同时不损害性能。我们引入了一种新颖的思维方程蒸馏（EoTD）技术，将推理过程封装为基于方程的表示，构建了一个EoTD数据集来对SLMs进行微调。此外，我们提出了集合思维蒸馏（ETD）框架，以提升SLMs的推理性能。这包括创建一个包含多个思维过程（包括思维链、思维程序和思维方程）的推理数据集，并将其用于微调。我们的实验证明，EoTD显著提升了SLMs的推理能力，而ETD使这些模型实现了最先进的推理性能。

    This work addresses the challenge of democratizing advanced Large Language Models (LLMs) by compressing their mathematical reasoning capabilities into sub-billion parameter Small Language Models (SLMs) without compromising performance. We introduce Equation-of-Thought Distillation (EoTD), a novel technique that encapsulates the reasoning process into equation-based representations to construct an EoTD dataset for fine-tuning SLMs. Additionally, we propose the Ensemble Thoughts Distillation (ETD) framework to enhance the reasoning performance of SLMs. This involves creating a reasoning dataset with multiple thought processes, including Chain-of-Thought (CoT), Program-of-Thought (PoT), and Equation-of-Thought (EoT), and using it for fine-tuning. Our experimental findings demonstrate that EoTD significantly boosts the reasoning abilities of SLMs, while ETD enables these models to achieve state-of-the-art reasoning performance.
    
[^9]: 缩小知识评估差距：多层次答案的开放领域问答

    Narrowing the Knowledge Evaluation Gap: Open-Domain Question Answering with Multi-Granularity Answers. (arXiv:2401.04695v1 [cs.CL])

    [http://arxiv.org/abs/2401.04695](http://arxiv.org/abs/2401.04695)

    本研究提出了一种名为GRANOLA QA的评估设置，在开放领域问答中使用多粒度答案来评估预测的答案的准确性和信息量。作者提出了一种简单的方法来丰富现有数据集，并创建了一个多粒度版本的数据集。实验结果表明...

    

    事实性问题通常可以以不同层次的粒度正确回答。例如，“1961年8月4日”和“1961年”都是对“巴拉克·奥巴马是在什么时候出生的？”这个问题的正确答案。然而，标准的问答评估协议并没有明确考虑这一点，而是将预测的答案与单一粒度层次的答案进行比较。在这项工作中，我们提出了GRANOLA QA，一种新颖的评估设置，其中预测的答案根据准确性和信息量与一组多粒度答案进行评估。我们提出了一种简单的方法来丰富现有数据集的多粒度答案，并创建了GRANOLA-EQ，一个多粒度版本的EntityQuestions数据集。我们在GRANOLA-EQ上评估了一系列解码方法，包括一种新的算法，称为Decoding with Response Aggregation (DRAG)，该算法旨在将响应的粒度与模型的不确定性对齐。我们的实验显示...

    Factual questions typically can be answered correctly at different levels of granularity. For example, both ``August 4, 1961'' and ``1961'' are correct answers to the question ``When was Barack Obama born?''. Standard question answering (QA) evaluation protocols, however, do not explicitly take this into account and compare a predicted answer against answers of a single granularity level. In this work, we propose GRANOLA QA, a novel evaluation setting where a predicted answer is evaluated in terms of accuracy and informativeness against a set of multi-granularity answers. We present a simple methodology for enriching existing datasets with multi-granularity answers, and create GRANOLA-EQ, a multi-granularity version of the EntityQuestions dataset. We evaluate a range of decoding methods on GRANOLA-EQ, including a new algorithm, called Decoding with Response Aggregation (DRAG), that is geared towards aligning the response granularity with the model's uncertainty. Our experiments show th
    
[^10]: 大语言模型时代文本水印技术综述

    A Survey of Text Watermarking in the Era of Large Language Models. (arXiv:2312.07913v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2312.07913](http://arxiv.org/abs/2312.07913)

    本文综述了大语言模型时代的文本水印技术，包括不同技术的概述和比较、算法评估方法、应用场景以及当前挑战和未来发展方向。

    

    文本水印算法在版权保护中起着至关重要的作用，然而其能力和应用场景一直受限。大语言模型的最新发展为文本水印技术的进步打开了新的机会。大语言模型不仅通过其文本理解和生成能力增强了文本水印算法的能力，还需要使用文本水印算法来保护自身的版权。本文对当前文本水印技术的现状进行了全面的调查，包括四个主要方面：（1）不同文本水印技术的概述和比较；（2）文本水印算法的评估方法，包括成功率、对文本质量的影响、鲁棒性和防篡改性；（3）文本水印技术的潜在应用场景；（4）当前挑战和未来发展方向。

    Text watermarking algorithms play a crucial role in the copyright protection of textual content, yet their capabilities and application scenarios have been limited historically. The recent developments in large language models (LLMs) have opened new opportunities for the advancement of text watermarking techniques. LLMs not only enhance the capabilities of text watermarking algorithms through their text understanding and generation abilities but also necessitate the use of text watermarking algorithms for their own copyright protection. This paper conducts a comprehensive survey of the current state of text watermarking technology, covering four main aspects: (1) an overview and comparison of different text watermarking techniques; (2) evaluation methods for text watermarking algorithms, including their success rates, impact on text quality, robustness, and unforgeability; (3) potential application scenarios for text watermarking technology; (4) current challenges and future directions
    
[^11]: 使用查询聚合的指导性对话摘要

    Instructive Dialogue Summarization with Query Aggregations. (arXiv:2310.10981v1 [cs.CL])

    [http://arxiv.org/abs/2310.10981](http://arxiv.org/abs/2310.10981)

    传统的对话摘要方法无法考虑用户的特定兴趣，而指导对话摘要的引入可以帮助扩展对话摘要模型的能力。我们提出了一个三步方法来合成高质量的查询摘要三元组，并通过在多个数据集上训练一个统一模型来扩展对话摘要模型的能力。

    

    传统的对话摘要方法直接生成摘要，不考虑用户的特定兴趣。这在用户更加关注特定主题或方面的情况下会带来挑战。随着指导调优语言模型的进步，我们引入了指导对话来扩展对话摘要模型的能力集。为了克服指导性对话摘要数据的稀缺性，我们提出了一种三步方法来合成高质量的基于查询的摘要三元组。这个过程包括以摘要为锚点的查询生成、查询过滤和基于查询的摘要生成。通过在三个摘要数据集上训练一个统一的模型InstructDS（指导性对话摘要），我们扩展了对话摘要模型的能力。我们在包括对话摘要和对话阅读理解的四个数据集上对我们的方法进行评估。

    Conventional dialogue summarization methods directly generate summaries and do not consider user's specific interests. This poses challenges in cases where the users are more focused on particular topics or aspects. With the advancement of instruction-finetuned language models, we introduce instruction-tuning to dialogues to expand the capability set of dialogue summarization models. To overcome the scarcity of instructive dialogue summarization data, we propose a three-step approach to synthesize high-quality query-based summarization triples. This process involves summary-anchored query generation, query filtering, and query-based summary generation. By training a unified model called InstructDS (Instructive Dialogue Summarization) on three summarization datasets with multi-purpose instructive triples, we expand the capability of dialogue summarization models. We evaluate our method on four datasets, including dialogue summarization and dialogue reading comprehension. Experimental re
    
[^12]: VoIP平台上语音增强的心理声学挑战

    Psychoacoustic Challenges Of Speech Enhancement On VoIP Platforms. (arXiv:2310.07161v1 [cs.SD])

    [http://arxiv.org/abs/2310.07161](http://arxiv.org/abs/2310.07161)

    本研究在VoIP通信领域中探索了声学转换的复杂性，并通过分析心理声学指标，揭示了语音增强对VoIP系统的影响。

    

    在VoIP（互联网语音传输协议）通信中，由声学转换引入的复杂性需要进行严格的分析。本研究基于对专有发送端降噪效果的探索，对Google Meets和Zoom等平台进行了细致评估。研究利用Deep Noise Suppression（DNS）2020数据集，确保了针对各种降噪设置和接收器接口的结构化考察。通过将Oaxaca分解引入到声学-语音扰动分析中，本研究引入了一种方法论的创新，该分解通常是经济计量学工具，在此处重新用于分析VoIP系统中的声学-语音扰动。为了进一步确定这些转换的影响，利用心理声学指标，特别是PESQ和STOI，来提供对语音改变的全面理解。总体而言，所获得的观点突出显示了VoIP影响的声学动力学的复杂景观。

    Within the ambit of VoIP (Voice over Internet Protocol) telecommunications, the complexities introduced by acoustic transformations merit rigorous analysis. This research, rooted in the exploration of proprietary sender-side denoising effects, meticulously evaluates platforms such as Google Meets and Zoom. The study draws upon the Deep Noise Suppression (DNS) 2020 dataset, ensuring a structured examination tailored to various denoising settings and receiver interfaces. A methodological novelty is introduced via the Oaxaca decomposition, traditionally an econometric tool, repurposed herein to analyze acoustic-phonetic perturbations within VoIP systems. To further ground the implications of these transformations, psychoacoustic metrics, specifically PESQ and STOI, were harnessed to furnish a comprehensive understanding of speech alterations. Cumulatively, the insights garnered underscore the intricate landscape of VoIP-influenced acoustic dynamics. In addition to the primary findings, a 
    
[^13]: 基于BERT模型的推文地理位置预测

    Geolocation Predicting of Tweets Using BERT-Based Models. (arXiv:2303.07865v1 [cs.CL])

    [http://arxiv.org/abs/2303.07865](http://arxiv.org/abs/2303.07865)

    该论文提出基于BERT模型的推文地理位置预测方法，可以实现全球和美国上的中位误差分别小于30公里和15公里的定位精度。

    

    该研究旨在解决推文/用户地理位置预测任务，并提供了处理文本大数据地理标记的灵活方法。该方法采用基于神经网络的自然语言处理来估计坐标对（经度，纬度）和二维高斯混合模型（GMM）。提出的模型的范围已经在Twitter数据集上使用预训练的BERT模型进行调整。性能指标表明，对于在推文内容和元数据上训练和评估的模型，全球范围内的中位误差小于30公里，美国范围内的中位误差小于15公里。

    This research is aimed to solve the tweet/user geolocation prediction task and provide a flexible methodology for the geotagging of textual big data. The suggested approach implements neural networks for natural language processing (NLP) to estimate the location as coordinate pairs (longitude, latitude) and two-dimensional Gaussian Mixture Models (GMMs). The scope of proposed models has been finetuned on a Twitter dataset using pretrained Bidirectional Encoder Representations from Transformers (BERT) as base models. Performance metrics show a median error of fewer than 30 km on a worldwide-level, and fewer than 15 km on the US-level datasets for the models trained and evaluated on text features of tweets' content and metadata context.
    

