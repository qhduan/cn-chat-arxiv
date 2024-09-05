# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Measuring and Modeling "Culture" in LLMs: A Survey](https://arxiv.org/abs/2403.15412) | 这项研究调查了39篇最新论文，旨在研究大型语言模型中的文化表达和包容性，发现当前研究未对“文化”进行定义，而是在特定设计的数据集上对模型进行探究，研究了某些“文化”的方面，留下许多未被探究的有趣和重要方面，如语义领域和关于性。 |
| [^2] | [Simple and Scalable Strategies to Continually Pre-train Large Language Models](https://arxiv.org/abs/2403.08763) | 通过简单和可扩展的学习率调整、重放数据的方法，可以在不重新训练的情况下，持续预训练大型语言模型以匹配完全重新训练时的性能。 |
| [^3] | [Persian Slang Text Conversion to Formal and Deep Learning of Persian Short Texts on Social Media for Sentiment Classification](https://arxiv.org/abs/2403.06023) | 通过提供PSC工具将波斯语俚语文本转换为正式文本，结合深度学习方法进行波斯语短文本的情感学习。 |
| [^4] | [AI-generated text boundary detection with RoFT](https://arxiv.org/abs/2311.08349) | 使用RoFT进行人工智能生成文本边界检测的研究揭示了基于困惑度的方法在跨领域和跨模型设置中更加稳健。 |
| [^5] | [Transformers as Recognizers of Formal Languages: A Survey on Expressivity.](http://arxiv.org/abs/2311.00208) | 本文对transformers在形式语言识别领域的相关研究进行了全面调查，为理解其表达能力提供了一个统一的框架。 |
| [^6] | [Correction with Backtracking Reduces Hallucination in Summarization.](http://arxiv.org/abs/2310.16176) | 本文介绍了一种简单而有效的技术，CoBa，用于减少摘要中的幻觉。该方法通过测量条件词概率和上下文词距离的统计信息进行幻觉检测，并通过直观的回溯法进行减轻。实验证明，CoBa在减少摘要幻觉方面是有效且高效的。 |
| [^7] | [Large Language Models for Information Retrieval: A Survey.](http://arxiv.org/abs/2308.07107) | 本综述将大型语言模型（LLMs）在信息检索中的发展进行了综述，探讨了其在捕捉上下文信号和语义细微之处方面的优势和挑战，以及与传统检索方法的结合的重要性。 |
| [^8] | [A Sentence is Worth a Thousand Pictures: Can Large Language Models Understand Human Language?.](http://arxiv.org/abs/2308.00109) | 本文分析了大型语言模型作为理论信息丰富表示和非理论强大机械工具的贡献，并指出当前的模型发展和利用中仍然缺乏关键能力。 |
| [^9] | [CADGE: Context-Aware Dialogue Generation Enhanced with Graph-Structured Knowledge Aggregation.](http://arxiv.org/abs/2305.06294) | 本文提出了一种基于上下文感知的图注意力模型，可以将上下文增强的知识聚合过程与相关知识图的全局特征有效融合，将增强的图结构知识集成到基于上下文感知的对话生成模型中。实验证明，该模型在自动度量和人类评估方面均优于现有方法。 |

# 详细

[^1]: 在LLMs中测量和建模“文化”：一项调查

    Towards Measuring and Modeling "Culture" in LLMs: A Survey

    [https://arxiv.org/abs/2403.15412](https://arxiv.org/abs/2403.15412)

    这项研究调查了39篇最新论文，旨在研究大型语言模型中的文化表达和包容性，发现当前研究未对“文化”进行定义，而是在特定设计的数据集上对模型进行探究，研究了某些“文化”的方面，留下许多未被探究的有趣和重要方面，如语义领域和关于性。

    

    我们呈现了对39篇最新论文的调查，旨在研究大型语言模型中的文化表达和包容性。我们观察到，没有一篇研究定义“文化”，这是一个复杂、多层面的概念；相反，它们在一些特别设计的数据集上对模型进行探究，这些数据集代表了某些“文化”的方面。我们将这些方面称为文化的代理，并将它们组织在人口统计、语义和语言文化交互代理的三个维度上。我们还对采用的探查方法进行了分类。我们的分析表明，只有“文化”的某些方面，如价值观和目标，被研究了，留下了几个其他有趣且重要的方面，特别是大量语义领域和关于性（Hershcovich等人，2022）的未被探究。另外两个关键的空白是目前方法的鲁棒性和情境性的缺乏。基于这些观察结果，

    arXiv:2403.15412v1 Announce Type: cross  Abstract: We present a survey of 39 recent papers that aim to study cultural representation and inclusion in large language models. We observe that none of the studies define "culture," which is a complex, multifaceted concept; instead, they probe the models on some specially designed datasets which represent certain aspects of "culture." We call these aspects the proxies of cultures, and organize them across three dimensions of demographic, semantic and linguistic-cultural interaction proxies. We also categorize the probing methods employed. Our analysis indicates that only certain aspects of "culture," such as values and objectives, have been studied, leaving several other interesting and important facets, especially the multitude of semantic domains (Thompson et al., 2020) and aboutness (Hershcovich et al., 2022), unexplored. Two other crucial gaps are the lack of robustness and situatedness of the current methods. Based on these observations
    
[^2]: 持续预训练大型语言模型的简单可扩展策略

    Simple and Scalable Strategies to Continually Pre-train Large Language Models

    [https://arxiv.org/abs/2403.08763](https://arxiv.org/abs/2403.08763)

    通过简单和可扩展的学习率调整、重放数据的方法，可以在不重新训练的情况下，持续预训练大型语言模型以匹配完全重新训练时的性能。

    

    大型语言模型（LLMs）通常在数十亿的标记上进行常规预训练，一旦有新数据可用就重新开始该过程。一个更有效率的解决方案是持续预训练这些模型，与重新训练相比能节省大量计算资源。然而，新数据引起的分布转移通常会导致在以前数据上降低性能或无法适应新数据。在本工作中，我们展示了一种简单且可扩展的学习率（LR）重新升温、LR重新衰减和重放上一数据的组合足以与完全从头开始重新训练在所有可用数据上的性能相匹配，从最终损失和语言模型（LM）评估基准的角度衡量。具体而言，我们展示了在两个常用的LLM预训练数据集（英语→英语）之间的弱但现实的分布转移以及更强烈的分布转移（英语→德语）下的情况。

    arXiv:2403.08763v1 Announce Type: cross  Abstract: Large language models (LLMs) are routinely pre-trained on billions of tokens, only to start the process over again once new data becomes available. A much more efficient solution is to continually pre-train these models, saving significant compute compared to re-training. However, the distribution shift induced by new data typically results in degraded performance on previous data or poor adaptation to the new data. In this work, we show that a simple and scalable combination of learning rate (LR) re-warming, LR re-decaying, and replay of previous data is sufficient to match the performance of fully re-training from scratch on all available data, as measured by final loss and language model (LM) evaluation benchmarks. Specifically, we show this for a weak but realistic distribution shift between two commonly used LLM pre-training datasets (English$\rightarrow$English) and a stronger distribution shift (English$\rightarrow$German) at th
    
[^3]: 波斯语俚语文本转换为正式文本以及社交媒体上波斯语短文本的深度学习用于情感分类

    Persian Slang Text Conversion to Formal and Deep Learning of Persian Short Texts on Social Media for Sentiment Classification

    [https://arxiv.org/abs/2403.06023](https://arxiv.org/abs/2403.06023)

    通过提供PSC工具将波斯语俚语文本转换为正式文本，结合深度学习方法进行波斯语短文本的情感学习。

    

    缺乏适合分析波斯语会话文本的工具使得对这些文本（包括情感分析）的各种分析变得困难。本研究尝试通过提供PSC（波斯语俚语转换器），将会话文本转换为正式文本，并结合最新和最佳的深度学习方法，使机器更容易理解这些文本，更好地进行波斯语短文本的情感学习。

    arXiv:2403.06023v1 Announce Type: new  Abstract: The lack of a suitable tool for the analysis of conversational texts in the Persian language has made various analyses of these texts, including Sentiment Analysis, difficult. In this research, we tried to make the understanding of these texts easier for the machine by providing PSC, Persian Slang Converter, a tool for converting conversational texts into formal ones, and by using the most up-to-date and best deep learning methods along with the PSC, the sentiment learning of short Persian language texts for the machine in a better way. be made More than 10 million unlabeled texts from various social networks and movie subtitles (as Conversational texts) and about 10 million news texts (as formal texts) have been used for training unsupervised models and formal implementation of the tool. 60,000 texts from the comments of Instagram social network users with positive, negative, and neutral labels are considered supervised data for trainin
    
[^4]: 使用RoFT进行人工智能生成文本边界检测

    AI-generated text boundary detection with RoFT

    [https://arxiv.org/abs/2311.08349](https://arxiv.org/abs/2311.08349)

    使用RoFT进行人工智能生成文本边界检测的研究揭示了基于困惑度的方法在跨领域和跨模型设置中更加稳健。

    

    由于大语言模型的快速发展，人们越来越经常遇到可能一开始是由人类编写但之后是由机器生成的文本。检测这些文本中人类编写和机器生成部分之间的边界是一个具有挑战性且在文献中尚未受到足够关注的问题。我们试图填补这一差距，并研究几种方法来将最先进的人工文本检测分类器调整为边界检测设置。我们将所有检测器推向极限，在包含多个主题的短文本的Real or Fake文本基准集上进行测试，并包括各种语言模型的生成。我们利用这种多样性深入研究所有检测器在跨领域和跨模型设置中的鲁棒性，以提供未来研究的基线和见解。特别地，我们发现基于困惑度的边界检测方法倾向于更加稳健。

    arXiv:2311.08349v2 Announce Type: replace  Abstract: Due to the rapid development of large language models, people increasingly often encounter texts that may start as written by a human but continue as machine-generated. Detecting the boundary between human-written and machine-generated parts of such texts is a challenging problem that has not received much attention in literature. We attempt to bridge this gap and examine several ways to adapt state of the art artificial text detection classifiers to the boundary detection setting. We push all detectors to their limits, using the Real or Fake text benchmark that contains short texts on several topics and includes generations of various language models. We use this diversity to deeply examine the robustness of all detectors in cross-domain and cross-model settings to provide baselines and insights for future research. In particular, we find that perplexity-based approaches to boundary detection tend to be more robust to peculiarities 
    
[^5]: Transformers作为形式语言识别器：关于表达能力的调查

    Transformers as Recognizers of Formal Languages: A Survey on Expressivity. (arXiv:2311.00208v1 [cs.LG])

    [http://arxiv.org/abs/2311.00208](http://arxiv.org/abs/2311.00208)

    本文对transformers在形式语言识别领域的相关研究进行了全面调查，为理解其表达能力提供了一个统一的框架。

    

    随着transformers在自然语言处理中的重要性日益突出，一些研究人员开始从理论上探讨它们能否解决问题，将问题视为形式语言。探索这类问题将有助于比较transformers与其他模型以及不同变种之间的差异，适用于各种任务。近年来，在这个子领域的工作取得了相当大的进展。本文对这方面的工作进行了全面调查，记录了不同结果背后的各种假设，并提供了一个统一的框架，以协调看似相互矛盾的研究结果。

    As transformers have gained prominence in natural language processing, some researchers have investigated theoretically what problems they can and cannot solve, by treating problems as formal languages. Exploring questions such as this will help to compare transformers with other models, and transformer variants with one another, for various tasks. Work in this subarea has made considerable progress in recent years. Here, we undertake a comprehensive survey of this work, documenting the diverse assumptions that underlie different results and providing a unified framework for harmonizing seemingly contradictory findings.
    
[^6]: 通过回溯法纠正，减少摘要中的幻觉

    Correction with Backtracking Reduces Hallucination in Summarization. (arXiv:2310.16176v1 [cs.CL])

    [http://arxiv.org/abs/2310.16176](http://arxiv.org/abs/2310.16176)

    本文介绍了一种简单而有效的技术，CoBa，用于减少摘要中的幻觉。该方法通过测量条件词概率和上下文词距离的统计信息进行幻觉检测，并通过直观的回溯法进行减轻。实验证明，CoBa在减少摘要幻觉方面是有效且高效的。

    

    摘要生成旨在生成源文件的自然语言摘要，既简洁又保留重要元素。尽管最近取得了一些进展，但神经文本摘要模型容易产生幻觉（或更准确地说是混淆），即生成的摘要包含源文件中没有根据的细节。在本文中，我们引入了一种简单而有效的技术，CoBa，用于减少摘要中的幻觉。该方法基于两个步骤：幻觉检测和减轻。我们展示了通过测量有关条件词概率和上下文词距离的简单统计信息可以实现前者。此外，我们还证明了直观的回溯法在减轻幻觉方面的惊人效果。我们在三个文本摘要基准数据集上对所提出的方法进行了全面评估。结果表明，CoBa在减少摘要幻觉方面是有效且高效的。

    Abstractive summarization aims at generating natural language summaries of a source document that are succinct while preserving the important elements. Despite recent advances, neural text summarization models are known to be susceptible to hallucinating (or more correctly confabulating), that is to produce summaries with details that are not grounded in the source document. In this paper, we introduce a simple yet efficient technique, CoBa, to reduce hallucination in abstractive summarization. The approach is based on two steps: hallucination detection and mitigation. We show that the former can be achieved through measuring simple statistics about conditional word probabilities and distance to context words. Further, we demonstrate that straight-forward backtracking is surprisingly effective at mitigation. We thoroughly evaluate the proposed method with prior art on three benchmark datasets for text summarization. The results show that CoBa is effective and efficient in reducing hall
    
[^7]: 信息检索中的大型语言模型：一项综述

    Large Language Models for Information Retrieval: A Survey. (arXiv:2308.07107v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2308.07107](http://arxiv.org/abs/2308.07107)

    本综述将大型语言模型（LLMs）在信息检索中的发展进行了综述，探讨了其在捕捉上下文信号和语义细微之处方面的优势和挑战，以及与传统检索方法的结合的重要性。

    

    作为信息获取的主要手段，信息检索（IR）系统，如搜索引擎，已经融入到我们的日常生活中。这些系统还作为对话、问答和推荐系统的组成部分。IR的发展轨迹从基于词项的方法起步，逐渐发展成与先进的神经模型相融合。尽管神经模型擅长捕捉复杂的上下文信号和语义细微之处，从而改变了IR的格局，但它们仍然面临着数据稀缺、可解释性以及生成上下文合理但潜在不准确响应的挑战。这种演变需要传统方法（如基于词项的稀疏检索方法与快速响应）和现代神经架构（如具有强大语言理解能力的语言模型）的结合。与此同时，大型语言模型（LLMs），如ChatGPT和GPT-4的出现，引起了一场革命

    As a primary means of information acquisition, information retrieval (IR) systems, such as search engines, have integrated themselves into our daily lives. These systems also serve as components of dialogue, question-answering, and recommender systems. The trajectory of IR has evolved dynamically from its origins in term-based methods to its integration with advanced neural models. While the neural models excel at capturing complex contextual signals and semantic nuances, thereby reshaping the IR landscape, they still face challenges such as data scarcity, interpretability, and the generation of contextually plausible yet potentially inaccurate responses. This evolution requires a combination of both traditional methods (such as term-based sparse retrieval methods with rapid response) and modern neural architectures (such as language models with powerful language understanding capacity). Meanwhile, the emergence of large language models (LLMs), typified by ChatGPT and GPT-4, has revolu
    
[^8]: 一句话胜千张图片：大型语言模型能理解人类语言吗？

    A Sentence is Worth a Thousand Pictures: Can Large Language Models Understand Human Language?. (arXiv:2308.00109v1 [cs.CL])

    [http://arxiv.org/abs/2308.00109](http://arxiv.org/abs/2308.00109)

    本文分析了大型语言模型作为理论信息丰富表示和非理论强大机械工具的贡献，并指出当前的模型发展和利用中仍然缺乏关键能力。

    

    人工智能应用在依赖于下一个单词预测的语言相关任务中表现出巨大潜力。当前一代大型语言模型被认为能够达到类人语言表现，并且它们的应用被誉为人工通用智能的关键步骤，同时也是对人类语言认知和神经基础的重大进展的理解。我们分析了大型语言模型作为目标系统的理论信息丰富表示与非理论强大机械工具的贡献，并确定了当前发展和利用这些模型所缺失的关键能力。

    Artificial Intelligence applications show great potential for language-related tasks that rely on next-word prediction. The current generation of large language models have been linked to claims about human-like linguistic performance and their applications are hailed both as a key step towards Artificial General Intelligence and as major advance in understanding the cognitive, and even neural basis of human language. We analyze the contribution of large language models as theoretically informative representations of a target system vs. atheoretical powerful mechanistic tools, and we identify the key abilities that are still missing from the current state of development and exploitation of these models.
    
[^9]: CADGE：基于图结构知识聚合的上下文感知对话生成

    CADGE: Context-Aware Dialogue Generation Enhanced with Graph-Structured Knowledge Aggregation. (arXiv:2305.06294v1 [cs.CL])

    [http://arxiv.org/abs/2305.06294](http://arxiv.org/abs/2305.06294)

    本文提出了一种基于上下文感知的图注意力模型，可以将上下文增强的知识聚合过程与相关知识图的全局特征有效融合，将增强的图结构知识集成到基于上下文感知的对话生成模型中。实验证明，该模型在自动度量和人类评估方面均优于现有方法。

    

    常识知识（commonsense knowledge）对于自然语言处理任务来说至关重要。现有的方法通常将图知识与传统的图神经网络（GNNs）相结合，导致文本和图知识编码过程在串行流水线中被分离。我们认为，这些分离的表示学习阶段可能对神经网络学习包含在两种输入知识类型中的整体上下文是次优的。在本文中，我们提出了一种新颖的基于上下文感知的图注意力模型（Context-aware GAT），它可以基于上下文增强的知识聚合过程有效地融合相关知识图的全局特征。具体地，我们的框架利用了一种新颖的表示学习方法来处理异构特征——将图知识与文本相结合。据我们所知，这是第一次尝试在连接子图上分层应用图知识聚合以及上下文信息，并将增强的图结构知识集成到基于上下文感知的对话生成模型中。我们在两个基准数据集上的实验证明，所提出的模型在自动度量和人类评估方面均优于现有方法。

    Commonsense knowledge is crucial to many natural language processing tasks. Existing works usually incorporate graph knowledge with conventional graph neural networks (GNNs), leading to the text and graph knowledge encoding processes being separated in a serial pipeline. We argue that these separate representation learning stages may be suboptimal for neural networks to learn the overall context contained in both types of input knowledge. In this paper, we propose a novel context-aware graph-attention model (Context-aware GAT), which can effectively incorporate global features of relevant knowledge graphs based on a context-enhanced knowledge aggregation process. Specifically, our framework leverages a novel representation learning approach to process heterogeneous features - combining flattened graph knowledge with text. To the best of our knowledge, this is the first attempt at hierarchically applying graph knowledge aggregation on a connected subgraph in addition to contextual infor
    

