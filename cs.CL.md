# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PATCH -- Psychometrics-AssisTed benCHmarking of Large Language Models: A Case Study of Mathematics Proficiency](https://arxiv.org/abs/2404.01799) | 该论文提出了一种新的框架PATCH，用于将心理测量领域的知识整合到大型语言模型的基准测试中，以解决现有基准测试存在的测量质量、项目级别评估和参考人群等问题。 |
| [^2] | [The Larger the Better? Improved LLM Code-Generation via Budget Reallocation](https://arxiv.org/abs/2404.00725) | 较小的语言模型可以在相同预算下产生可靠的改进，但在无法进行单元测试的情况下，较小的模型选择排名次于较大模型的单个输出。 |
| [^3] | [AutoRE: Document-Level Relation Extraction with Large Language Models](https://arxiv.org/abs/2403.14888) | AutoRE 是一种端到端的文档级关系抽取模型，采用了一种名为RHF的新颖关系抽取范式，可有效处理分布在文档中的多个关系和三元组事实。 |
| [^4] | [A Unified Framework for Model Editing](https://arxiv.org/abs/2403.14236) | 这个统一框架结合了“定位和编辑”模型编辑技术，最大化保留某些向量表示并记忆新事实信息。 |
| [^5] | [Optimal Block-Level Draft Verification for Accelerating Speculative Decoding](https://arxiv.org/abs/2403.10444) | 提出了一种更好的草稿验证算法，通过将验证步骤制定为块级最优传输问题，实现了额外的墙钟速度提升，而不增加额外的计算成本和草稿标记 |
| [^6] | [Identifying Semantic Induction Heads to Understand In-Context Learning](https://arxiv.org/abs/2402.13055) | 该研究通过分析注意力头的操作，揭示了结合了句法依赖和知识图关系的语义感应头的出现，从而更好地理解了大型语言模型的上下文学习能力。 |
| [^7] | [Understanding and Mitigating the Threat of Vec2Text to Dense Retrieval Systems](https://arxiv.org/abs/2402.12784) | 本文研究了Vec2Text对密集检索系统的威胁以及如何缓解，通过对距离度量、池化函数、瓶颈预训练等方面进行深入分析，以获得对密集检索系统中文本可恢复性和检索效果权衡关键元素的更深入理解。 |
| [^8] | [Chain-of-Layer: Iteratively Prompting Large Language Models for Taxonomy Induction from Limited Examples](https://arxiv.org/abs/2402.07386) | 本文介绍了一种称为Chain-of-Layer的上下文学习框架，用于从给定的实体集中归纳分类体系。通过引入基于集成的排名过滤器来减少错误，Chain-of-Layer在四个实际基准测试中实现了最先进的性能。 |
| [^9] | [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750) | 该论文提出了一种无需调整的非对称2位量化KV缓存技术，以解决存储注意力键和值的内存需求增加和推断速度受限问题。 |
| [^10] | [Adapting Large Language Models via Reading Comprehension](https://arxiv.org/abs/2309.09530) | 通过将原始语料库转化为阅读理解文本来调整大型语言模型，使其在多个领域的各种任务中性能始终得到提升。 |
| [^11] | [Brand Network Booster: A New System for Improving Brand Connectivity.](http://arxiv.org/abs/2309.16228) | 本文介绍了一种新的决策支持系统，用于深入分析语义网络，为品牌形象的更好探索和连接性的改进提供洞察力。 |
| [^12] | [The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants.](http://arxiv.org/abs/2308.16884) | Belebele是一个包含122种语言变体的多选机器阅读理解数据集，可用于评估文本模型在高、中和低资源语言中的性能。尽管英语为中心的大型语言模型在跨语言转移方面表现良好，但小型多语言遮蔽语言模型在其他语言上表现更佳。 |
| [^13] | [LyricWhiz: Robust Multilingual Zero-shot Lyrics Transcription by Whispering to ChatGPT.](http://arxiv.org/abs/2306.17103) | LyricWhiz是一种鲁棒、多语言、零射击的自动歌词转录方法，通过使用Whisper作为"耳朵"和GPT-4作为"大脑"，它在各种数据集上实现了最先进的性能，同时还实现了在多种语言中进行歌词转录的能力，并创建了第一个大规模多语言歌词转录数据集。 |

# 详细

[^1]: PATCH -- 大型语言模型的心理测量辅助基准测试：数学能力的案例研究

    PATCH -- Psychometrics-AssisTed benCHmarking of Large Language Models: A Case Study of Mathematics Proficiency

    [https://arxiv.org/abs/2404.01799](https://arxiv.org/abs/2404.01799)

    该论文提出了一种新的框架PATCH，用于将心理测量领域的知识整合到大型语言模型的基准测试中，以解决现有基准测试存在的测量质量、项目级别评估和参考人群等问题。

    

    许多现有的大型（多模态）语言模型（LLMs）基准测试着重于衡量LLMs的学术能力，通常也对比较模型性能与人类考试者感兴趣。尽管这些基准测试对LLMs的发展至关重要，但它们存在一些限制，包括有问题的测量质量（例如，它们是否以可靠的方式衡量所需的内容？）、缺乏项目级别的质量评估（例如，有些项目是否比其他更重要或更困难？）以及人类人口参照模糊（例如，模型可以与谁进行比较？）。为了应对这些挑战，我们提出利用心理测量学领域的知识——一门致力于测量潜在变量如学术能力的领域——来进行LLMs基准测试的心理测量辅助方法。我们的主要贡献有三点。首先，我们介绍了PATCH：一种用于大型语言模型的心理测量辅助基准测试的新框架。

    arXiv:2404.01799v1 Announce Type: new  Abstract: Many existing benchmarks of large (multimodal) language models (LLMs) focus on measuring LLMs' academic proficiency, often with also an interest in comparing model performance with human test takers. While these benchmarks have proven key to the development of LLMs, they suffer from several limitations, including questionable measurement quality (e.g., Do they measure what they are supposed to in a reliable way?), lack of quality assessment on the item level (e.g., Are some items more important or difficult than others?) and unclear human population reference (e.g., To whom can the model be compared?). In response to these challenges, we propose leveraging knowledge from psychometrics - a field dedicated to the measurement of latent variables like academic proficiency - into LLM benchmarking. We make three primary contributions. First, we introduce PATCH: a novel framework for Psychometrics-AssisTed benCHmarking of LLMs. PATCH addresses 
    
[^2]: 越大越好吗？通过预算重新分配改进LLM代码生成

    The Larger the Better? Improved LLM Code-Generation via Budget Reallocation

    [https://arxiv.org/abs/2404.00725](https://arxiv.org/abs/2404.00725)

    较小的语言模型可以在相同预算下产生可靠的改进，但在无法进行单元测试的情况下，较小的模型选择排名次于较大模型的单个输出。

    

    人们普遍认为，大型语言模型(LLMs)比较小的模型更好。然而，更大的模型在推断过程中也需要更多的时间和计算资源。这就引出了一个问题：当两个模型在相同的预算下运行时会发生什么？（例如，计算资源，运行时间）。为了解决这个问题，我们分析了各种大小的代码生成LLMs，并进行比较，例如运行一个70B模型一次与从13B模型生成五个输出并选择一个的情况。我们的研究结果表明，在标准单元测试设置中，反复使用较小的模型可以产生一致的改进，在五个任务中最高可达15%的增益。另一方面，在无法进行单元测试的情况下，从较小模型中基于排名的候选选择表现不及来自较大模型的单个输出。我们的结果突显了使用较小模型而非较大模型的潜力。

    arXiv:2404.00725v1 Announce Type: cross  Abstract: It is a common belief that large language models (LLMs) are better than smaller-sized ones. However, larger models also require significantly more time and compute during inference. This begs the question: what happens when both models operate under the same budget? (e.g., compute, run-time). To address this question, we analyze code generation LLMs of various sizes and make comparisons such as running a 70B model once vs. generating five outputs from a 13B model and selecting one. Our findings reveal that, in a standard unit-test setup, the repeated use of smaller models can yield consistent improvements, with gains of up to 15% across five tasks. On the other hand, in scenarios where unit-tests are unavailable, a ranking-based selection of candidates from the smaller model falls short of the performance of a single output from larger ones. Our results highlight the potential of using smaller models instead of larger ones, and the imp
    
[^3]: AutoRE：使用大型语言模型进行文档级关系抽取

    AutoRE: Document-Level Relation Extraction with Large Language Models

    [https://arxiv.org/abs/2403.14888](https://arxiv.org/abs/2403.14888)

    AutoRE 是一种端到端的文档级关系抽取模型，采用了一种名为RHF的新颖关系抽取范式，可有效处理分布在文档中的多个关系和三元组事实。

    

    大型语言模型(LLMs)展示了在理解和生成文本方面的异常能力，这激励着许多研究人员利用它们进行信息抽取(IE)任务，包括关系抽取(RE)。然而，大多数现有方法主要设计用于句子级关系抽取(SentRE)任务，这通常涵盖了单个句子内的一组关系和三元组事实。此外，一些方法采用将关系作为候选选择集成到提示模板中的方式，导致在处理分布在给定文档中的多个关系和三元组事实时效率低下，性能亚优，并在处理文档级关系抽取(DocRE)任务时面临独特挑战。为了克服这些限制，我们介绍了AutoRE，这是一个端到端的DocRE模型，采用了一种名为RHF(Re

    arXiv:2403.14888v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have demonstrated exceptional abilities in comprehending and generating text, motivating numerous researchers to utilize them for Information Extraction (IE) purposes, including Relation Extraction (RE). Nonetheless, most existing methods are predominantly designed for Sentence-level Relation Extraction (SentRE) tasks, which typically encompass a restricted set of relations and triplet facts within a single sentence. Furthermore, certain approaches resort to treating relations as candidate choices integrated into prompt templates, leading to inefficient processing and suboptimal performance when tackling Document-Level Relation Extraction (DocRE) tasks, which entail handling multiple relations and triplet facts distributed across a given document, posing distinct challenges. To overcome these limitations, we introduce AutoRE, an end-to-end DocRE model that adopts a novel RE extraction paradigm named RHF (Re
    
[^4]: 一个统一的模型编辑框架

    A Unified Framework for Model Editing

    [https://arxiv.org/abs/2403.14236](https://arxiv.org/abs/2403.14236)

    这个统一框架结合了“定位和编辑”模型编辑技术，最大化保留某些向量表示并记忆新事实信息。

    

    模型编辑是一个不断发展的领域，专注于更新模型中嵌入的知识。在各种方法中，ROME和MEMIT作为主要的“定位和编辑”模型编辑技术脱颖而出。而MEMIT可以批量编辑记忆，ROME则一次只能改变一个事实。本文引入了一个统一的框架，将ROME和MEMIT纳入一个单一的概念框架，优化同一目标，我们称之为“保存-记忆”目标。该目标旨在在记忆新事实信息的同时保留某些选定向量的表示。具体来说，ROME使用等式约束优化此目标，而MEMIT采用更灵活的最小二乘约束。除了批量编辑外，MEMIT还可以在多个层面编辑模型。我们将编辑的分布从多个层面分开，区别于优化目标。

    arXiv:2403.14236v1 Announce Type: cross  Abstract: Model editing is a growing area focused on updating the knowledge embedded within models. Among the various methodologies, ROME and MEMIT stand out as leading "locate-and-edit" model editing techniques. While MEMIT enables batched editing of memories, ROME is limited to changing one fact at a time. This paper introduces a unifying framework that brings ROME and MEMIT under a single conceptual umbrella, optimizing for the same goal, which we call the "preservation-memorization" objective. This objective aims to preserve the representations of certain selected vectors while memorizing the representations of new factual information. Specifically, ROME optimizes this objective using an equality constraint, whereas MEMIT employs a more flexible least-square constraint. In addition to making batched edits, MEMIT also edits the model at multiple layers. We disentangle the distribution of edits to multiple layers from the optimization objectiv
    
[^5]: 用于加速推测解码的最佳块级草稿验证

    Optimal Block-Level Draft Verification for Accelerating Speculative Decoding

    [https://arxiv.org/abs/2403.10444](https://arxiv.org/abs/2403.10444)

    提出了一种更好的草稿验证算法，通过将验证步骤制定为块级最优传输问题，实现了额外的墙钟速度提升，而不增加额外的计算成本和草稿标记

    

    推测解码已被证明是在推理过程中加速大型语言模型（LLMs）无损加速的有效方法。 在每次迭代中，算法首先使用一个较小的模型起草一块标记。这些标记然后由大型模型并行验证，只有一部分标记将被保留，以确保最终输出遵循大型模型的分布。 在以往的所有推测解码工作中，起草验证是独立地逐个标记执行的。 在本工作中，我们提出了一个更好的起草验证算法，可提供额外的墙钟加速，而不需要额外的计算成本和起草标记。 我们首先将起草验证步骤制定为一个块级最优传输问题。 块级制定允许我们考虑更广泛的起草验证算法，并在一个起草中预期获得更多接受的标记数量

    arXiv:2403.10444v1 Announce Type: cross  Abstract: Speculative decoding has shown to be an effective method for lossless acceleration of large language models (LLMs) during inference. In each iteration, the algorithm first uses a smaller model to draft a block of tokens. The tokens are then verified by the large model in parallel and only a subset of tokens will be kept to guarantee that the final output follows the distribution of the large model. In all of the prior speculative decoding works, the draft verification is performed token-by-token independently. In this work, we propose a better draft verification algorithm that provides additional wall-clock speedup without incurring additional computation cost and draft tokens. We first formulate the draft verification step as a block-level optimal transport problem. The block-level formulation allows us to consider a wider range of draft verification algorithms and obtain a higher number of accepted tokens in expectation in one draft 
    
[^6]: 识别语义感应头以理解上下文学习

    Identifying Semantic Induction Heads to Understand In-Context Learning

    [https://arxiv.org/abs/2402.13055](https://arxiv.org/abs/2402.13055)

    该研究通过分析注意力头的操作，揭示了结合了句法依赖和知识图关系的语义感应头的出现，从而更好地理解了大型语言模型的上下文学习能力。

    

    虽然大型语言模型(LLMs)已经展示出卓越的性能，但它们推理逻辑的不透明性引发了对其可靠性的担忧。为了更好地理解LLMs，我们对注意力头的操作进行了详细分析，并旨在更好地理解LLMs的上下文学习。具体而言，我们研究了注意力头是否编码了自然语言中存在的两种类型的关系：从句子中解析的句法依赖和知识图中的关系。我们发现某些注意力头表现出一种模式，即当关注头标记时，它们会回忆起尾标记，并增加这些尾标记的输出逻辑。更重要的是，这种语义感应头的制定与语言模型上下文学习能力的出现存在密切关联。语义注意力头的研究推动了我们的

    arXiv:2402.13055v1 Announce Type: cross  Abstract: Although large language models (LLMs) have demonstrated remarkable performance, the lack of transparency in their inference logic raises concerns about their trustworthiness. To gain a better understanding of LLMs, we conduct a detailed analysis of the operations of attention heads and aim to better understand the in-context learning of LLMs. Specifically, we investigate whether attention heads encode two types of relationships between tokens present in natural languages: the syntactic dependency parsed from sentences and the relation within knowledge graphs. We find that certain attention heads exhibit a pattern where, when attending to head tokens, they recall tail tokens and increase the output logits of those tail tokens. More crucially, the formulation of such semantic induction heads has a close correlation with the emergence of the in-context learning ability of language models. The study of semantic attention heads advances our
    
[^7]: 理解和缓解Vec2Text对密集检索系统的威胁

    Understanding and Mitigating the Threat of Vec2Text to Dense Retrieval Systems

    [https://arxiv.org/abs/2402.12784](https://arxiv.org/abs/2402.12784)

    本文研究了Vec2Text对密集检索系统的威胁以及如何缓解，通过对距离度量、池化函数、瓶颈预训练等方面进行深入分析，以获得对密集检索系统中文本可恢复性和检索效果权衡关键元素的更深入理解。

    

    引入Vec2Text技术，一种用于反转文本嵌入的技术，引发了人们对密集检索系统中存在严重隐私问题的担忧，包括那些使用OpenAI和Cohere提供的文本嵌入的系统。这种威胁来自于一个恶意攻击者通过访问文本嵌入来重构原始文本。本文研究了影响使用Vec2Text恢复文本的嵌入模型的各个方面。我们的探索涉及距离度量、池化函数、瓶颈预训练、加噪声训练、嵌入量化和嵌入维度等因素，这些因素在原始Vec2Text论文中尚未被讨论。通过对这些因素的深入分析，我们旨在更深入地了解影响密集检索系统中文本可恢复性和检索效果之间权衡的关键因素。

    arXiv:2402.12784v1 Announce Type: cross  Abstract: The introduction of Vec2Text, a technique for inverting text embeddings, has raised serious privacy concerns within dense retrieval systems utilizing text embeddings, including those provided by OpenAI and Cohere. This threat comes from the ability for a malicious attacker with access to text embeddings to reconstruct the original text.   In this paper, we investigate various aspects of embedding models that could influence the recoverability of text using Vec2Text. Our exploration involves factors such as distance metrics, pooling functions, bottleneck pre-training, training with noise addition, embedding quantization, and embedding dimensions -- aspects not previously addressed in the original Vec2Text paper. Through a thorough analysis of these factors, our aim is to gain a deeper understanding of the critical elements impacting the trade-offs between text recoverability and retrieval effectiveness in dense retrieval systems. This a
    
[^8]: Chain-of-Layer：通过有限示例迭代引导大型语言模型进行分类体系归纳

    Chain-of-Layer: Iteratively Prompting Large Language Models for Taxonomy Induction from Limited Examples

    [https://arxiv.org/abs/2402.07386](https://arxiv.org/abs/2402.07386)

    本文介绍了一种称为Chain-of-Layer的上下文学习框架，用于从给定的实体集中归纳分类体系。通过引入基于集成的排名过滤器来减少错误，Chain-of-Layer在四个实际基准测试中实现了最先进的性能。

    

    自动分类体系归纳对于网络搜索、推荐系统和问答系统非常重要。手动整理分类体系需要大量人力成本，因此自动构建分类体系非常有需求。本文介绍了一种称为Chain-of-Layer的上下文学习框架，用于从给定的实体集中归纳分类体系。Chain-of-Layer将任务分解为每一层选择相关候选实体，并逐步从上到下构建分类体系。为了减少错误，我们引入了基于集成的排名过滤器，在每一次迭代中减少生成的虚构内容。通过大量实验证明，Chain-of-Layer在四个实际基准测试中实现了最先进的性能。

    Automatic taxonomy induction is crucial for web search, recommendation systems, and question answering. Manual curation of taxonomies is expensive in terms of human effort, making automatic taxonomy construction highly desirable. In this work, we introduce Chain-of-Layer which is an in-context learning framework designed to induct taxonomies from a given set of entities. Chain-of-Layer breaks down the task into selecting relevant candidate entities in each layer and gradually building the taxonomy from top to bottom. To minimize errors, we introduce the Ensemble-based Ranking Filter to reduce the hallucinated content generated at each iteration. Through extensive experiments, we demonstrate that Chain-of-Layer achieves state-of-the-art performance on four real-world benchmarks.
    
[^9]: KIVI：一种无需调整的非对称2位量化KV缓存技术

    KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

    [https://arxiv.org/abs/2402.02750](https://arxiv.org/abs/2402.02750)

    该论文提出了一种无需调整的非对称2位量化KV缓存技术，以解决存储注意力键和值的内存需求增加和推断速度受限问题。

    

    高效地为大型语言模型（LLMs）提供服务需要将许多请求批量处理以减少每个请求的成本。然而，存储注意力键和值以避免重新计算的键值（KV）缓存显著增加了内存需求，并成为速度和内存使用的新瓶颈。这种内存需求随着批处理大小和上下文长度的增加而增加。此外，推断速度受到KV缓存大小的限制，因为GPU的SRAM必须从主GPU内存中加载整个KV缓存以生成每个标记，导致计算核心在此过程中处于空闲状态。减小KV缓存大小的一个直接而有效的解决方案是量化，通过减少KV缓存所需的总字节数来实现。然而，目前缺乏对KV缓存元素分布进行深入研究以了解KV缓存量化的难度和限制。为了弥补这一空白，我们开展了一项全面的元素分布研究。。。

    Efficiently serving large language models (LLMs) requires batching many requests together to reduce the cost per request. Yet, the key-value (KV) cache, which stores attention keys and values to avoid re-computations, significantly increases memory demands and becomes the new bottleneck in speed and memory usage. This memory demand increases with larger batch sizes and longer context lengths. Additionally, the inference speed is limited by the size of KV cache, as the GPU's SRAM must load the entire KV cache from the main GPU memory for each token generated, causing the computational core to be idle during this process. A straightforward and effective solution to reduce KV cache size is quantization, which decreases the total bytes taken by KV cache. However, there is a lack of in-depth studies that explore the element distribution of KV cache to understand the hardness and limitation of KV cache quantization. To fill the gap, we conducted a comprehensive study on the element distribut
    
[^10]: 通过阅读理解调整大型语言模型

    Adapting Large Language Models via Reading Comprehension

    [https://arxiv.org/abs/2309.09530](https://arxiv.org/abs/2309.09530)

    通过将原始语料库转化为阅读理解文本来调整大型语言模型，使其在多个领域的各种任务中性能始终得到提升。

    

    我们探讨了在特定领域语料库上持续预训练对大型语言模型的影响，发现在原始语料库上进行训练赋予模型领域知识，但极大地损害了其回答问题的能力。受人类通过阅读理解学习的启发，即阅读后练习提高基于所学知识回答问题的能力，我们提出了一种将原始语料库转化为阅读理解文本的简单方法。每个原始文本都会被一系列与其内容相关的任务丰富。我们的方法非常可扩展，适用于任何预训练语料库，能够在三个不同领域（生物医学、金融和法律）的各种任务中持续提升性能。值得注意的是，我们的7B语言模型在竞争中表现出色，能与规模更大的领域特定模型（如BloombergGPT-50B）相媲美。此外，我们证明了领域特定模型可以带来更好的效果。

    arXiv:2309.09530v2 Announce Type: replace  Abstract: We explore how continued pre-training on domain-specific corpora influences large language models, revealing that training on the raw corpora endows the model with domain knowledge, but drastically hurts its prompting ability for question answering. Taken inspiration from human learning via reading comprehension--practice after reading improves the ability to answer questions based on the learned knowledge--we propose a simple method for transforming raw corpora into reading comprehension texts. Each raw text is enriched with a series of tasks related to its content. Our method, highly scalable and applicable to any pre-training corpora, consistently enhances performance across various tasks in three different domains: biomedicine, finance, and law. Notably, our 7B language model achieves competitive performance with domain-specific models of much larger scales, such as BloombergGPT-50B. Furthermore, we demonstrate that domain-specif
    
[^11]: 品牌网络增强器：提升品牌连接性的新系统

    Brand Network Booster: A New System for Improving Brand Connectivity. (arXiv:2309.16228v1 [cs.SI])

    [http://arxiv.org/abs/2309.16228](http://arxiv.org/abs/2309.16228)

    本文介绍了一种新的决策支持系统，用于深入分析语义网络，为品牌形象的更好探索和连接性的改进提供洞察力。

    

    本文介绍了一种新的决策支持系统，用于深入分析语义网络，为品牌形象的更好探索和连接性的改进提供洞察力。在网络分析方面，我们通过解决扩展版的最大连接度改进问题来实现这一目标，其中包括考虑敌对节点、约束预算和加权网络的可能性 - 通过添加链接或增加现有连接的权重来实现连接性的改进。我们结合两个案例研究来展示这个新系统，并讨论其性能。我们的工具和方法对于网络学者和支持市场营销和传播管理者的战略决策过程都很有用。

    This paper presents a new decision support system offered for an in-depth analysis of semantic networks, which can provide insights for a better exploration of a brand's image and the improvement of its connectivity. In terms of network analysis, we show that this goal is achieved by solving an extended version of the Maximum Betweenness Improvement problem, which includes the possibility of considering adversarial nodes, constrained budgets, and weighted networks - where connectivity improvement can be obtained by adding links or increasing the weight of existing connections. We present this new system together with two case studies, also discussing its performance. Our tool and approach are useful both for network scholars and for supporting the strategic decision-making processes of marketing and communication managers.
    
[^12]: Belebele基准数据集：122种语言变体的并行阅读理解数据集

    The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants. (arXiv:2308.16884v1 [cs.CL])

    [http://arxiv.org/abs/2308.16884](http://arxiv.org/abs/2308.16884)

    Belebele是一个包含122种语言变体的多选机器阅读理解数据集，可用于评估文本模型在高、中和低资源语言中的性能。尽管英语为中心的大型语言模型在跨语言转移方面表现良好，但小型多语言遮蔽语言模型在其他语言上表现更佳。

    

    我们提出了Belebele，一个包含122种语言变体的多选机器阅读理解（MRC）数据集。该数据集极大地扩展了自然语言理解（NLU）基准的语言覆盖范围，使得可以评估文本模型在高、中和低资源语言中的性能。每个问题都基于Flores-200数据集中的一个短篇文章，并提供了四个多选答案。问题经过精心策划，以区分具有不同通用语言理解水平的模型。单独的英语数据集已经足够困难，可以挑战最先进的语言模型。由于完全并行，该数据集可以直接比较所有语言的模型性能。我们使用该数据集评估多语言遮蔽语言模型（MLMs）和大型语言模型（LLMs）的能力。我们展示了广泛的结果，并发现尽管英语为中心的LLMs之间存在显著的跨语言转移，但小型MLMs在其他语言上的表现相对较好。

    We present Belebele, a multiple-choice machine reading comprehension (MRC) dataset spanning 122 language variants. Significantly expanding the language coverage of natural language understanding (NLU) benchmarks, this dataset enables the evaluation of text models in high-, medium-, and low-resource languages. Each question is based on a short passage from the Flores-200 dataset and has four multiple-choice answers. The questions were carefully curated to discriminate between models with different levels of general language comprehension. The English dataset on its own proves difficult enough to challenge state-of-the-art language models. Being fully parallel, this dataset enables direct comparison of model performance across all languages. We use this dataset to evaluate the capabilities of multilingual masked language models (MLMs) and large language models (LLMs). We present extensive results and find that despite significant cross-lingual transfer in English-centric LLMs, much small
    
[^13]: LyricWhiz: 通过向ChatGPT耳语进行鲁棒的多语言零射击歌词转录

    LyricWhiz: Robust Multilingual Zero-shot Lyrics Transcription by Whispering to ChatGPT. (arXiv:2306.17103v1 [cs.CL])

    [http://arxiv.org/abs/2306.17103](http://arxiv.org/abs/2306.17103)

    LyricWhiz是一种鲁棒、多语言、零射击的自动歌词转录方法，通过使用Whisper作为"耳朵"和GPT-4作为"大脑"，它在各种数据集上实现了最先进的性能，同时还实现了在多种语言中进行歌词转录的能力，并创建了第一个大规模多语言歌词转录数据集。

    

    我们介绍了一种名为LyricWhiz的鲁棒、多语言、零射击的自动歌词转录方法，该方法在各种歌词转录数据集上实现了最先进的性能，即使在具有挑战性的流派如摇滚和金属中也是如此。我们的全新、无需训练的方法利用了Whisper，一种弱监督的鲁棒语音识别模型，以及GPT-4，当今最性能卓越的基于聊天的大型语言模型。在该方法中，Whisper充当“耳朵”，负责转录语音，而GPT-4则作为“大脑”，作为一种具有强大性能的上下文输出选择和校正的注释器。我们的实验结果表明，与现有方法相比，LyricWhiz在英语中显著降低了词错误率，并且可以有效地转录多种语言的歌词。此外，我们使用LyricWhiz创建了第一个具有CC-BY-NC-SA版权许可的公开可用的大规模多语言歌词转录数据集，基于MTG-Jamendo，并提供了h

    We introduce LyricWhiz, a robust, multilingual, and zero-shot automatic lyrics transcription method achieving state-of-the-art performance on various lyrics transcription datasets, even in challenging genres such as rock and metal. Our novel, training-free approach utilizes Whisper, a weakly supervised robust speech recognition model, and GPT-4, today's most performant chat-based large language model. In the proposed method, Whisper functions as the "ear" by transcribing the audio, while GPT-4 serves as the "brain," acting as an annotator with a strong performance for contextualized output selection and correction. Our experiments show that LyricWhiz significantly reduces Word Error Rate compared to existing methods in English and can effectively transcribe lyrics across multiple languages. Furthermore, we use LyricWhiz to create the first publicly available, large-scale, multilingual lyrics transcription dataset with a CC-BY-NC-SA copyright license, based on MTG-Jamendo, and offer a h
    

