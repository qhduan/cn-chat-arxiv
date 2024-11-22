# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Large Language Models on Graphs: A Comprehensive Survey](https://rss.arxiv.org/abs/2312.02783) | 这篇论文对在图上的大型语言模型进行了全面调查，研究了纯图形、文本属性图形和文本配对图形三个不同场景下的应用情况，并探讨了基于图形的推理能力是否可以推广到大型语言模型上。 |
| [^2] | [Linguacodus: A Synergistic Framework for Transformative Code Generation in Machine Learning Pipelines](https://arxiv.org/abs/2403.11585) | Linguacodus是一种创新框架，通过部署动态流水线和精细调整的大型语言模型，实现了将自然语言任务描述转换为代码的自动化过程，极大地推进了机器学习应用的发展。 |
| [^3] | [Design2Code: How Far Are We From Automating Front-End Engineering?](https://arxiv.org/abs/2403.03163) | 生成式人工智能在多模态理解和代码生成方面取得了突破，提出了Design2Code任务并进行了全面基准测试，展示了多模态LLMs直接将视觉设计转换为代码实现的能力。 |
| [^4] | [REAR: A Relevance-Aware Retrieval-Augmented Framework for Open-Domain Question Answering](https://arxiv.org/abs/2402.17497) | 提出了一种名为REAR的新方法，旨在解决大型语言模型在检索增强生成中无法准确评估检索文档相关性的问题，通过增强对检索文档相关性的自我意识，能够自适应地利用外部知识。 |
| [^5] | [Probing Multimodal Large Language Models for Global and Local Semantic Representation](https://arxiv.org/abs/2402.17304) | 通过研究发现，多模态大型语言模型的中间层能够更好地编码全局语义信息，在视觉-语言任务中表现出更好的性能。顶层可能过多关注局部信息，导致理解全局信息的能力下降。 |
| [^6] | [CodeArt: Better Code Models by Attention Regularization When Symbols Are Lacking](https://arxiv.org/abs/2402.11842) | 提出一种在符号缺失时通过注意力规范化改进代码模型的新方法，使用程序分析提取上下文并利用注意力掩码方法，同时利用自注意力机制学习关注度的重要性 |
| [^7] | [PaDeLLM-NER: Parallel Decoding in Large Language Models for Named Entity Recognition](https://arxiv.org/abs/2402.04838) | 本研究提出了PaDeLLM-NER，一种能够在大型语言模型中实现并行解码，从而显著减少命名实体识别的生成延迟，同时保持预测质量和性能。 |
| [^8] | [Schema-Driven Information Extraction from Heterogeneous Tables](https://arxiv.org/abs/2305.14336) | 本文探讨了大型语言模型在通过引入基于模式的信息提取任务进行多领域表格数据处理时的竞争性表现，而无需特定流水线或标签，同时保持成本效率。 |
| [^9] | [CAMELL: Confidence-based Acquisition Model for Efficient Self-supervised Active Learning with Label Validation.](http://arxiv.org/abs/2310.08944) | CAMELL是一个适用于序列多输出问题的主动学习框架，通过仅需专家标注序列的一小部分、自监督和标签验证机制来解决监督神经方法对大规模标注数据集的依赖限制。 |
| [^10] | [BODEGA: Benchmark for Adversarial Example Generation in Credibility Assessment.](http://arxiv.org/abs/2303.08032) | BODEGA是一个基准测试，用于模拟真实的内容管理场景，在四个误传检测任务上测试受害模型和攻击方法。测试结果表明，在某些情况下，即使进行微小的文本修改，也可以欺骗最准确的分类器。 |

# 详细

[^1]: 在图上的大型语言模型：一项全面调查

    Large Language Models on Graphs: A Comprehensive Survey

    [https://rss.arxiv.org/abs/2312.02783](https://rss.arxiv.org/abs/2312.02783)

    这篇论文对在图上的大型语言模型进行了全面调查，研究了纯图形、文本属性图形和文本配对图形三个不同场景下的应用情况，并探讨了基于图形的推理能力是否可以推广到大型语言模型上。

    

    大型语言模型（LLMs），如GPT4和LLaMA，由于其强大的文本编码/解码能力和新发现的紧急能力（例如推理）在自然语言处理方面取得了显著的进展。虽然LLMs主要设计用于处理纯文本，但在许多现实场景中，文本数据与图形形式的丰富结构信息相关联（例如学术网络和电子商务网络），或者图形数据与丰富的文本信息配对（例如带有描述的分子）。此外，尽管LLMs已经展示了其基于纯文本的推理能力，但尚未探索此类能力是否可以推广到图形上（即基于图形的推理）。在本文中，我们对在图上的大型语言模型相关场景和技术进行了系统回顾。我们首先总结了采用LLMs在图形上的潜在场景，分为纯图形、文本属性图形和文本配对图形三个类别。

    Large language models (LLMs), such as GPT4 and LLaMA, are creating significant advancements in natural language processing, due to their strong text encoding/decoding ability and newly found emergent capability (e.g., reasoning). While LLMs are mainly designed to process pure texts, there are many real-world scenarios where text data is associated with rich structure information in the form of graphs (e.g., academic networks, and e-commerce networks) or scenarios where graph data is paired with rich textual information (e.g., molecules with descriptions). Besides, although LLMs have shown their pure text-based reasoning ability, it is underexplored whether such ability can be generalized to graphs (i.e., graph-based reasoning). In this paper, we provide a systematic review of scenarios and techniques related to large language models on graphs. We first summarize potential scenarios of adopting LLMs on graphs into three categories, namely pure graphs, text-attributed graphs, and text-pa
    
[^2]: Linguacodus：一种在机器学习流水线中进行变革性代码生成的协同框架

    Linguacodus: A Synergistic Framework for Transformative Code Generation in Machine Learning Pipelines

    [https://arxiv.org/abs/2403.11585](https://arxiv.org/abs/2403.11585)

    Linguacodus是一种创新框架，通过部署动态流水线和精细调整的大型语言模型，实现了将自然语言任务描述转换为代码的自动化过程，极大地推进了机器学习应用的发展。

    

    在不断发展的机器学习领域中，将自然语言描述无缝转化为可执行代码仍然是一个巨大的挑战。本文介绍了Linguacodus，这是一个创新性框架，旨在通过部署一个动态流水线，通过高级数据塑形指令，将自然语言任务描述迭代地转换为代码来应对这一挑战。Linguacodus的核心是一个经过精细调整的大型语言模型（LLM），能够评估各种问题的多样解决方案，并为特定任务选择最合适的解决方案。本文详细介绍了精细调整过程，并阐明了如何将自然语言描述转化为功能性代码。Linguacodus代表了自动化代码生成的重大飞跃，有效地弥合了任务描述和可执行代码之间的差距。它对推进跨不同领域的机器学习应用具有巨大潜力。

    arXiv:2403.11585v1 Announce Type: cross  Abstract: In the ever-evolving landscape of machine learning, seamless translation of natural language descriptions into executable code remains a formidable challenge. This paper introduces Linguacodus, an innovative framework designed to tackle this challenge by deploying a dynamic pipeline that iteratively transforms natural language task descriptions into code through high-level data-shaping instructions. The core of Linguacodus is a fine-tuned large language model (LLM), empowered to evaluate diverse solutions for various problems and select the most fitting one for a given task. This paper details the fine-tuning process, and sheds light on how natural language descriptions can be translated into functional code. Linguacodus represents a substantial leap towards automated code generation, effectively bridging the gap between task descriptions and executable code. It holds great promise for advancing machine learning applications across div
    
[^3]: Design2Code：我们离自动化前端工程有多远？

    Design2Code: How Far Are We From Automating Front-End Engineering?

    [https://arxiv.org/abs/2403.03163](https://arxiv.org/abs/2403.03163)

    生成式人工智能在多模态理解和代码生成方面取得了突破，提出了Design2Code任务并进行了全面基准测试，展示了多模态LLMs直接将视觉设计转换为代码实现的能力。

    

    近年来，生成式人工智能在多模态理解和代码生成方面取得了突飞猛进的进展，实现了前所未有的能力。这可以实现一种新的前端开发范式，其中多模态LLMs可能直接将视觉设计转换为代码实现。本文将这一过程形式化为Design2Code任务，并进行全面基准测试。我们手动策划了一个包含484个多样化真实网页的基准测试用例，并开发了一套自动评估指标，以评估当前多模态LLMs能否生成直接渲染为给定参考网页的代码实现，以输入为屏幕截图。我们还结合了全面的人工评估。我们开发了一套多模态提示方法，并展示了它们在GPT-4V和Gemini Pro Vision上的有效性。我们进一步对一个开源的Design2Code-18B模型进行了微调。

    arXiv:2403.03163v1 Announce Type: new  Abstract: Generative AI has made rapid advancements in recent years, achieving unprecedented capabilities in multimodal understanding and code generation. This can enable a new paradigm of front-end development, in which multimodal LLMs might directly convert visual designs into code implementations. In this work, we formalize this as a Design2Code task and conduct comprehensive benchmarking. Specifically, we manually curate a benchmark of 484 diverse real-world webpages as test cases and develop a set of automatic evaluation metrics to assess how well current multimodal LLMs can generate the code implementations that directly render into the given reference webpages, given the screenshots as input. We also complement automatic metrics with comprehensive human evaluations. We develop a suite of multimodal prompting methods and show their effectiveness on GPT-4V and Gemini Pro Vision. We further finetune an open-source Design2Code-18B model that su
    
[^4]: REAR：一种面向开放域问答的关注度感知检索增强框架

    REAR: A Relevance-Aware Retrieval-Augmented Framework for Open-Domain Question Answering

    [https://arxiv.org/abs/2402.17497](https://arxiv.org/abs/2402.17497)

    提出了一种名为REAR的新方法，旨在解决大型语言模型在检索增强生成中无法准确评估检索文档相关性的问题，通过增强对检索文档相关性的自我意识，能够自适应地利用外部知识。

    

    考虑到有限的内部参数化知识，检索增强生成（RAG）被广泛用于扩展大型语言模型（LLMs）的知识范围。尽管在RAG研究上进行了大量努力，但在现有方法中，LLMs 无法准确评估检索文档的相关性，因此很可能导致对外部知识（即检索文档）的误导甚至错误利用。为解决这一问题，本文提出了 REAR，一种面向开放域问答（QA）的关注度感知检索增强方法。作为关键动机，我们旨在增强LLMs对来源相关性的自我意识，以便在RAG系统中自适应地利用外部知识。特别是，我们开发了一种新的基于LLM的RAG系统架构，通过整合一个精确评估检索文档相关性的特别设计的排名头。此外，我们提出了一种改进的训练方法。

    arXiv:2402.17497v1 Announce Type: new  Abstract: Considering the limited internal parametric knowledge, retrieval-augmented generation (RAG) has been widely used to extend the knowledge scope of large language models (LLMs). Despite the extensive efforts on RAG research, in existing methods, LLMs cannot precisely assess the relevance of retrieved documents, thus likely leading to misleading or even incorrect utilization of external knowledge (i.e., retrieved documents). To address this issue, in this paper, we propose REAR, a RElevance-Aware Retrieval-augmented approach for open-domain question answering (QA). As the key motivation, we aim to enhance the self-awareness of source relevance for LLMs, so as to adaptively utilize external knowledge in RAG systems. Specially, we develop a new architecture for LLM based RAG system, by incorporating a specially designed rank head that precisely assesses the relevance of retrieved documents. Furthermore, we propose an improved training method 
    
[^5]: 探究多模态大型语言模型对全局和局部语义表示的影响

    Probing Multimodal Large Language Models for Global and Local Semantic Representation

    [https://arxiv.org/abs/2402.17304](https://arxiv.org/abs/2402.17304)

    通过研究发现，多模态大型语言模型的中间层能够更好地编码全局语义信息，在视觉-语言任务中表现出更好的性能。顶层可能过多关注局部信息，导致理解全局信息的能力下降。

    

    大型语言模型的成功启发了研究人员将其优秀的表示能力转移到其他模态。最近的一些研究利用图像描述对齐数据集训练多模态大型语言模型（MLLMs），在图像到文本任务中取得了最新的性能表现。然而，很少有研究探讨MLLMs是否真正理解完整的图像信息，即全局信息，或者它们只能捕捉一些局部对象信息。本研究发现模型的中间层可以编码更多全局语义信息，其表示向量在视觉-语言蕴涵任务上表现更好，而不是顶层。我们通过目标检测任务进一步探究模型的局部语义表示。我们得出的结论是顶层可能过多专注于局部信息，导致减弱了对全局信息的理解能力。

    arXiv:2402.17304v1 Announce Type: cross  Abstract: The success of large language models has inspired researchers to transfer their exceptional representing ability to other modalities. Several recent works leverage image-caption alignment datasets to train multimodal large language models (MLLMs), which achieve state-of-the-art performance on image-to-text tasks. However, there are very few studies exploring whether MLLMs truly understand the complete image information, i.e., global information, or if they can only capture some local object information. In this study, we find that the intermediate layers of models can encode more global semantic information, whose representation vectors perform better on visual-language entailment tasks, rather than the topmost layers. We further probe models for local semantic representation through object detection tasks. And we draw a conclusion that the topmost layers may excessively focus on local information, leading to a diminished ability to en
    
[^6]: CodeArt：当符号缺失时通过注意力规范化改进代码模型

    CodeArt: Better Code Models by Attention Regularization When Symbols Are Lacking

    [https://arxiv.org/abs/2402.11842](https://arxiv.org/abs/2402.11842)

    提出一种在符号缺失时通过注意力规范化改进代码模型的新方法，使用程序分析提取上下文并利用注意力掩码方法，同时利用自注意力机制学习关注度的重要性

    

    基于Transformer的代码模型在许多软件工程任务中表现出色。然而，当符号缺失或者不具信息量时，它们的有效性会下降。这是因为模型可能没有学会在没有符号的情况下正确地关注相关性/上下文。我们提出了一种新的方法，在符号缺失时预训练通用代码模型。我们观察到，在这种情况下，程序会退化为用非常原始的语言编写的内容。因此，我们建议使用程序分析来事先提取上下文（而不是像传统模型中依赖符号和掩码语言建模）。然后，我们利用一种新颖的注意力掩码方法，只允许模型关注这些上下文，例如双向程序依赖传递闭包和令牌共现。与此同时，内在的自注意力机制被用于学习允许的关注度哪些更重要。

    arXiv:2402.11842v1 Announce Type: cross  Abstract: Transformer based code models have impressive performance in many software engineering tasks. However, their effectiveness degrades when symbols are missing or not informative. The reason is that the model may not learn to pay attention to the right correlations/contexts without the help of symbols. We propose a new method to pre-train general code models when symbols are lacking. We observe that in such cases, programs degenerate to something written in a very primitive language. We hence propose to use program analysis to extract contexts a priori (instead of relying on symbols and masked language modeling as in vanilla models). We then leverage a novel attention masking method to only allow the model attending to these contexts, e.g., bi-directional program dependence transitive closures and token co-occurrences. In the meantime, the inherent self-attention mechanism is utilized to learn which of the allowed attentions are more impo
    
[^7]: PaDeLLM-NER：大型语言模型中的并行解码用于命名实体识别

    PaDeLLM-NER: Parallel Decoding in Large Language Models for Named Entity Recognition

    [https://arxiv.org/abs/2402.04838](https://arxiv.org/abs/2402.04838)

    本研究提出了PaDeLLM-NER，一种能够在大型语言模型中实现并行解码，从而显著减少命名实体识别的生成延迟，同时保持预测质量和性能。

    

    本研究旨在使用大型语言模型（LLMs）减少命名实体识别（NER）的生成延迟。LLMs的高延迟的主要原因是顺序解码过程，该过程自回归地生成NER的所有标签和提及，显著增加了序列长度。为此，我们引入了PaDeLLM-NER（Parallel Decoding in LLM for NE），这是一种无需额外模块或架构修改即可无缝集成到现有生成模型框架中的方法。PaDeLLM-NER允许同时解码所有提及，从而减少生成延迟。实验结果显示，PaDeLLM-NER的推理速度显著提高，对英语和中文来说比自回归方法快1.76到10.22倍。与各种数据集上的最先进性能相媲美，同时维持了预测质量。

    In this study, we aim to reduce generation latency for Named Entity Recognition (NER) with Large Language Models (LLMs). The main cause of high latency in LLMs is the sequential decoding process, which autoregressively generates all labels and mentions for NER, significantly increase the sequence length. To this end, we introduce Parallel Decoding in LLM for NE} (PaDeLLM-NER), a approach that integrates seamlessly into existing generative model frameworks without necessitating additional modules or architectural modifications. PaDeLLM-NER allows for the simultaneous decoding of all mentions, thereby reducing generation latency. Experiments reveal that PaDeLLM-NER significantly increases inference speed that is 1.76 to 10.22 times faster than the autoregressive approach for both English and Chinese. Simultaneously it maintains the quality of predictions as evidenced by the performance that is on par with the state-of-the-art across various datasets.
    
[^8]: 来自异构表格的基于模式的信息提取

    Schema-Driven Information Extraction from Heterogeneous Tables

    [https://arxiv.org/abs/2305.14336](https://arxiv.org/abs/2305.14336)

    本文探讨了大型语言模型在通过引入基于模式的信息提取任务进行多领域表格数据处理时的竞争性表现，而无需特定流水线或标签，同时保持成本效率。

    

    在本文中，我们探讨了大型语言模型是否能够支持高效地从表格中提取信息的问题。我们引入了基于模式的信息提取，这是一项将表格数据转换为按照人类编写的模式组织的记录的新任务。为了评估各种LLM在这一任务上的能力，我们提出了一个基准，包括来自四个不同领域的表格：机器学习论文、化学文献、材料科学期刊和网页。我们利用这个带有注释的表格集合来评估开源和基于API的语言模型从涵盖多种领域和数据格式的表格中提取信息的能力。我们的实验表明，即使不需要任务特定的流水线或标签，也可以实现出人意料的竞争性表现，F1分数范围从74.2到96.1，同时保持成本效率。此外，通过详细的消融研究

    arXiv:2305.14336v3 Announce Type: replace  Abstract: In this paper, we explore the question of whether large language models can support cost-efficient information extraction from tables. We introduce schema-driven information extraction, a new task that transforms tabular data into structured records following a human-authored schema. To assess various LLM's capabilities on this task, we present a benchmark comprised of tables from four diverse domains: machine learning papers, chemistry literature, material science journals, and webpages. We use this collection of annotated tables to evaluate the ability of open-source and API-based language models to extract information from tables covering diverse domains and data formats. Our experiments demonstrate that surprisingly competitive performance can be achieved without requiring task-specific pipelines or labels, achieving F1 scores ranging from 74.2 to 96.1, while maintaining cost efficiency. Moreover, through detailed ablation studie
    
[^9]: CAMELL：基于置信度的高效自监督主动学习与标签验证获取模型

    CAMELL: Confidence-based Acquisition Model for Efficient Self-supervised Active Learning with Label Validation. (arXiv:2310.08944v1 [cs.CL])

    [http://arxiv.org/abs/2310.08944](http://arxiv.org/abs/2310.08944)

    CAMELL是一个适用于序列多输出问题的主动学习框架，通过仅需专家标注序列的一小部分、自监督和标签验证机制来解决监督神经方法对大规模标注数据集的依赖限制。

    

    在序列任务中，受大规模且精确标注数据集的依赖限制，监督神经方法受到阻碍。标注质量随着从专家标注向众包标注的转变而逐渐恶化。为了解决这些挑战，我们提出了CAMELL（Confidence-based Acquisition Model for Efficient self-supervised active Learning with Label validation），这是一个针对序列多输出问题量身定制的基于池化的主动学习框架。CAMELL具有三个核心特点：(1)仅要求专家标注所选序列的一小部分，(2)为其余序列提供自监督，(3)采用标签验证机制，防止错误标签污染数据集并影响模型性能。我们在序列任务中对CAMELL进行了评估，特别强调对话信念跟踪，这是一个受限制的任务。

    Supervised neural approaches are hindered by their dependence on large, meticulously annotated datasets, a requirement that is particularly cumbersome for sequential tasks. The quality of annotations tends to deteriorate with the transition from expert-based to crowd-sourced labelling. To address these challenges, we present \textbf{CAMELL} (Confidence-based Acquisition Model for Efficient self-supervised active Learning with Label validation), a pool-based active learning framework tailored for sequential multi-output problems. CAMELL possesses three core features: (1) it requires expert annotators to label only a fraction of a chosen sequence, (2) it facilitates self-supervision for the remainder of the sequence, and (3) it employs a label validation mechanism to prevent erroneous labels from contaminating the dataset and harming model performance. We evaluate CAMELL on sequential tasks, with a special emphasis on dialogue belief tracking, a task plagued by the constraints of limited
    
[^10]: BODEGA: 针对可信度评估中对抗性样本生成的基准测试

    BODEGA: Benchmark for Adversarial Example Generation in Credibility Assessment. (arXiv:2303.08032v1 [cs.CL])

    [http://arxiv.org/abs/2303.08032](http://arxiv.org/abs/2303.08032)

    BODEGA是一个基准测试，用于模拟真实的内容管理场景，在四个误传检测任务上测试受害模型和攻击方法。测试结果表明，在某些情况下，即使进行微小的文本修改，也可以欺骗最准确的分类器。

    

    文本分类方法被广泛应用于检测不可信内容，如假新闻、社交媒体机器人、宣传等。较为准确的模型（可能基于深度神经网络）有助于管理公共电子平台，并经常导致内容创建者面临提交拒绝或已发布文本的撤下。为了避免进一步被检测，内容创建者尝试产生一个稍微修改过的文本版本（即攻击对抗性样本），利用分类器的弱点导致不同的输出。本文介绍了BODEGA：一个基准测试，用于在模拟内容管理的真实用例中测试受害模型和攻击方法在四个误传检测任务上的表现。我们还系统地测试了受欢迎的文本分类器对可用攻击技术的鲁棒性，并发现在某些情况下，即使在文本中进行微小的修改也可以欺骗最准确的分类器。

    Text classification methods have been widely investigated as a way to detect content of low credibility: fake news, social media bots, propaganda, etc. Quite accurate models (likely based on deep neural networks) help in moderating public electronic platforms and often cause content creators to face rejection of their submissions or removal of already published texts. Having the incentive to evade further detection, content creators try to come up with a slightly modified version of the text (known as an attack with an adversarial example) that exploit the weaknesses of classifiers and result in a different output. Here we introduce BODEGA: a benchmark for testing both victim models and attack methods on four misinformation detection tasks in an evaluation framework designed to simulate real use-cases of content moderation. We also systematically test the robustness of popular text classifiers against available attacking techniques and discover that, indeed, in some cases barely signif
    

