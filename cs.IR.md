# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Transforming LLMs into Cross-modal and Cross-lingual RetrievalSystems](https://arxiv.org/abs/2404.01616) | 提出使用LLMs初始化多模态DE检索系统，实现在102种语言中匹配语音和文本的能力，无需在LLM预训练期间使用语音数据，且相比先前系统取得10%的Recall@1绝对改进 |
| [^2] | [Planning and Editing What You Retrieve for Enhanced Tool Learning](https://arxiv.org/abs/2404.00450) | 该论文提出了一种新颖的模型，结合了“规划与检索”和“编辑与确认”范式，通过神经检索模块和LLM-based查询规划器提高了工具利用的效果。 |
| [^3] | [EulerFormer: Sequential User Behavior Modeling with Complex Vector Attention](https://arxiv.org/abs/2403.17729) | EulerFormer提出了一种具有复杂向量注意力的新型转换器变体，统一了语义差异和位置差异的理论框架。 |
| [^4] | [BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives](https://arxiv.org/abs/2402.14151) | BIRCO基准评估基于大型语言模型的信息检索系统对多方面用户目标的检索能力，发现新的检索协议和更强大的模型是解决复杂用户需求的必要条件。 |
| [^5] | [Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens](https://arxiv.org/abs/2401.17377) | 这项研究展示了n-gram语言模型的价值，并介绍了一个名为infini-gram的引擎，它可以以毫秒级的延迟计算任意n的n-gram概率，使得在神经大型语言模型中对文本进行更准确的分析成为可能。 |
| [^6] | [Retrieval-Augmented Generative Agent for Reaction Condition Recommendation in Chemical Synthesis.](http://arxiv.org/abs/2311.10776) | 本研究提出了一种转变性的人工智能代理，利用检索增强生成（RAG）技术自动化化学中的反应条件推荐（RCR）任务，通过模拟专家化学家的策略，使用大型语言模型（LLM）和新反应指纹，显著优于传统人工智能。此系统可以减轻化学家的工作负担，使他们能够更专注于更基础和创造性的科学问题。 |
| [^7] | [Incorporating Recklessness to Collaborative Filtering based Recommender Systems.](http://arxiv.org/abs/2308.02058) | 本文提出了一种将鲁莽行为引入基于矩阵分解的推荐系统学习过程的方法，通过控制风险水平来提高预测的数量和质量。 |

# 详细

[^1]: 将LLMs转化为跨模态和跨语言检索系统

    Transforming LLMs into Cross-modal and Cross-lingual RetrievalSystems

    [https://arxiv.org/abs/2404.01616](https://arxiv.org/abs/2404.01616)

    提出使用LLMs初始化多模态DE检索系统，实现在102种语言中匹配语音和文本的能力，无需在LLM预训练期间使用语音数据，且相比先前系统取得10%的Recall@1绝对改进

    

    大型语言模型（LLMs）是在仅基于文本数据进行训练的，这超出了具有配对语音和文本数据的语言范围。同时，基于双编码器（DE）的检索系统将查询和文档投影到相同的嵌入空间中，并在检索和双语文本挖掘中展示了成功。为了在许多语言中匹配语音和文本，我们建议使用LLMs初始化多模态DE检索系统。与传统方法不同，我们的系统在LLM预训练期间不需要语音数据，并且可以利用LLM的多语言文本理解能力来匹配检索训练期间看不见的语言中的语音和文本。我们的多模态LLM-based检索系统能够在102种语言中匹配语音和文本，尽管只在21种语言上进行了训练。我们的系统优于先前专门在所有102种语言上训练的系统。在这些语言中，我们在Recall@1上实现了10％的绝对改进。

    arXiv:2404.01616v1 Announce Type: new  Abstract: Large language models (LLMs) are trained on text-only data that go far beyond the languages with paired speech and text data. At the same time, Dual Encoder (DE) based retrieval systems project queries and documents into the same embedding space and have demonstrated their success in retrieval and bi-text mining. To match speech and text in many languages, we propose using LLMs to initialize multi-modal DE retrieval systems. Unlike traditional methods, our system doesn't require speech data during LLM pre-training and can exploit LLM's multilingual text understanding capabilities to match speech and text in languages unseen during retrieval training. Our multi-modal LLM-based retrieval system is capable of matching speech and text in 102 languages despite only training on 21 languages. Our system outperforms previous systems trained explicitly on all 102 languages. We achieve a 10% absolute improvement in Recall@1 averaged across these l
    
[^2]: 规划和编辑检索以增强工具学习

    Planning and Editing What You Retrieve for Enhanced Tool Learning

    [https://arxiv.org/abs/2404.00450](https://arxiv.org/abs/2404.00450)

    该论文提出了一种新颖的模型，结合了“规划与检索”和“编辑与确认”范式，通过神经检索模块和LLM-based查询规划器提高了工具利用的效果。

    

    最近在将外部工具与大型语言模型（LLMs）集成方面取得的进展打开了新的领域，应用范围涵盖数学推理、代码生成器和智能助手。然而，现有方法依赖简单的一次性检索策略，无法有效准确地筛选相关工具。本文介绍了一种新颖的“规划与检索（P&R）”和“编辑与确认（E&G）”范式的模型，包括了神经检索模块和基于LLM的查询规划器，以增强工具利用的效果。

    arXiv:2404.00450v1 Announce Type: new  Abstract: Recent advancements in integrating external tools with Large Language Models (LLMs) have opened new frontiers, with applications in mathematical reasoning, code generators, and smart assistants. However, existing methods, relying on simple one-time retrieval strategies, fall short on effectively and accurately shortlisting relevant tools. This paper introduces a novel \modelname (\modelmeaning) approach, encompassing ``Plan-and-Retrieve (P\&R)'' and ``Edit-and-Ground (E\&G)'' paradigms. The P\&R paradigm consists of a neural retrieval module for shortlisting relevant tools and an LLM-based query planner that decomposes complex queries into actionable tasks, enhancing the effectiveness of tool utilization. The E\&G paradigm utilizes LLMs to enrich tool descriptions based on user scenarios, bridging the gap between user queries and tool functionalities. Experiment results demonstrate that these paradigms significantly improve the recall an
    
[^3]: EulerFormer：具有复杂向量注意力的顺序用户行为建模

    EulerFormer: Sequential User Behavior Modeling with Complex Vector Attention

    [https://arxiv.org/abs/2403.17729](https://arxiv.org/abs/2403.17729)

    EulerFormer提出了一种具有复杂向量注意力的新型转换器变体，统一了语义差异和位置差异的理论框架。

    

    为了捕捉用户偏好，转换器模型被广泛应用于建模顺序用户行为数据。转换器架构的核心在于自注意力机制，它计算序列中的成对注意力分数。由于排列等变性的特性，位置编码用于增强令牌表示之间的注意力。在这种设定下，成对注意力分数可以通过语义差异和位置差异两者衍生出来。然而，先前的研究经常以不同方式建模两种不同类型的差异测量，这可能限制了序列建模的表达能力。为了解决这个问题，本文提出了一种名为EulerFormer的具有复杂向量注意力的新型转换器变体，提供了一个统一的理论框架来表述语义差异和位置差异。 EulerFormer包含两个关键技术改进。

    arXiv:2403.17729v1 Announce Type: cross  Abstract: To capture user preference, transformer models have been widely applied to model sequential user behavior data. The core of transformer architecture lies in the self-attention mechanism, which computes the pairwise attention scores in a sequence. Due to the permutation-equivariant nature, positional encoding is used to enhance the attention between token representations. In this setting, the pairwise attention scores can be derived by both semantic difference and positional difference. However, prior studies often model the two kinds of difference measurements in different ways, which potentially limits the expressive capacity of sequence modeling. To address this issue, this paper proposes a novel transformer variant with complex vector attention, named EulerFormer, which provides a unified theoretical framework to formulate both semantic difference and positional difference. The EulerFormer involves two key technical improvements. Fi
    
[^4]: BIRCO：具有复杂目标的信息检索任务基准

    BIRCO: A Benchmark of Information Retrieval Tasks with Complex Objectives

    [https://arxiv.org/abs/2402.14151](https://arxiv.org/abs/2402.14151)

    BIRCO基准评估基于大型语言模型的信息检索系统对多方面用户目标的检索能力，发现新的检索协议和更强大的模型是解决复杂用户需求的必要条件。

    

    我们提出了具有复杂目标的信息检索(IR)任务基准(BIRCO)。 BIRCO评估IR系统根据多方面用户目标检索文档的能力。 该基准的复杂性和紧凑大小使其适用于评估基于大型语言模型(LLM)的信息检索系统。 我们提出了一个模块化框架，用于研究可能影响LLM在检索任务上的性能的因素，并确定了一个简单的基线模型，该模型与或优于现有方法和更复杂的替代方案。 没有一种方法在所有基准任务上均达到令人满意的性能，这表明需要更强大的模型和新的检索协议来解决复杂的用户需求。

    arXiv:2402.14151v1 Announce Type: cross  Abstract: We present the Benchmark of Information Retrieval (IR) tasks with Complex Objectives (BIRCO). BIRCO evaluates the ability of IR systems to retrieve documents given multi-faceted user objectives. The benchmark's complexity and compact size make it suitable for evaluating large language model (LLM)-based information retrieval systems. We present a modular framework for investigating factors that may influence LLM performance on retrieval tasks, and identify a simple baseline model which matches or outperforms existing approaches and more complex alternatives. No approach achieves satisfactory performance on all benchmark tasks, suggesting that stronger models and new retrieval protocols are necessary to address complex user needs.
    
[^5]: 无限-gram：将无限n-gram语言模型扩展到万亿标记

    Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens

    [https://arxiv.org/abs/2401.17377](https://arxiv.org/abs/2401.17377)

    这项研究展示了n-gram语言模型的价值，并介绍了一个名为infini-gram的引擎，它可以以毫秒级的延迟计算任意n的n-gram概率，使得在神经大型语言模型中对文本进行更准确的分析成为可能。

    

    在神经大型语言模型（LLM）时代，n-gram语言模型还具有相关性吗？我们的答案是肯定的，并且我们展示了它们在文本分析和改进神经LLM方面的价值。然而，这需要在两个方面对n-gram模型进行现代化。首先，我们将它们与神经LLM相同的数据规模训练- 1.4万亿个标记。这是迄今为止构建的最大的n-gram模型。其次，现有的n-gram模型使用的n很小，这妨碍了它们的性能；相反，我们允许n可以是任意大的，通过引入一个新的无限-gram LM与回退。我们开发了一个名为infini-gram的引擎，它可以通过后缀数组计算无限-gram（以及任意n的n-gram）概率，并且具有毫秒级的延迟，而无需预先计算n-gram计数表（这将非常昂贵）。无限-gram框架和infini-gram引擎使我们能够对人类写作和机器生成的文本进行许多新颖和有意思的分析：我们发现无限-gram LM...

    Are n-gram language models still relevant in this era of neural large language models (LLMs)? Our answer is yes, and we show their values in both text analysis and improving neural LLMs. Yet this necessitates modernizing n-gram models in two aspects. First, we train them at the same data scale as neural LLMs -- 1.4 trillion tokens. This is the largest n-gram model ever built. Second, existing n-gram models use small n which hinders their performance; we instead allow n to be arbitrarily large, by introducing a new $\infty$-gram LM with backoff. Instead of pre-computing n-gram count tables (which would be very expensive), we develop an engine named infini-gram -- powered by suffix arrays -- that can compute $\infty$-gram (as well as n-gram with arbitrary n) probabilities with millisecond-level latency. The $\infty$-gram framework and infini-gram engine enable us to conduct many novel and interesting analyses of human-written and machine-generated text: we find that the $\infty$-gram LM 
    
[^6]: 在化学合成中的反应条件推荐中，检索增强生成代理

    Retrieval-Augmented Generative Agent for Reaction Condition Recommendation in Chemical Synthesis. (arXiv:2311.10776v3 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2311.10776](http://arxiv.org/abs/2311.10776)

    本研究提出了一种转变性的人工智能代理，利用检索增强生成（RAG）技术自动化化学中的反应条件推荐（RCR）任务，通过模拟专家化学家的策略，使用大型语言模型（LLM）和新反应指纹，显著优于传统人工智能。此系统可以减轻化学家的工作负担，使他们能够更专注于更基础和创造性的科学问题。

    

    最近的人工智能研究为化学社会中的自动化化学反应铺平了一个有前途的未来。本研究提出了一种转变性的人工智能代理，利用检索增强生成（RAG）技术自动化化学中的反应条件推荐（RCR）任务。通过模拟专家化学家的搜索和分析策略，该代理使用大型语言模型（LLM）来查询分子数据库，并从在线文献中提取关键数据。此外，该人工智能代理还配备了我们为RCR任务开发的新反应指纹。由于RAG技术的使用，我们的代理使用更新的在线数据库作为知识源，显著优于仅受其训练数据固定知识限制的传统人工智能。由此产生的系统可以显著减轻化学家的工作负担，使他们能够更专注于更基础和创造性的科学问题。这一重大进展将计算技术与化学社会更紧密联系起来。

    Recent artificial intelligence (AI) research plots a promising future of automatic chemical reactions within the chemistry society. This study presents a transformative AI agent that automates the reaction condition recommendation (RCR) task in chemistry using retrieval-augmented generation (RAG) technology. By emulating expert chemists search and analysis strategies, the agent employs large language models (LLMs) to interrogate molecular databases and distill critical data from online literature. Further, the AI agent is equipped with our novel reaction fingerprint developed for the RCR task. Thanks to the RAG technology, our agent uses updated online databases as knowledge sources, significantly outperforming conventional AIs confined to the fixed knowledge within its training data. The resulting system can significantly reduce chemists workload, allowing them to focus on more fundamental and creative scientific problems. This significant advancement brings closer computational techn
    
[^7]: 整合鲁莽行为到基于协同过滤的推荐系统中

    Incorporating Recklessness to Collaborative Filtering based Recommender Systems. (arXiv:2308.02058v1 [cs.IR])

    [http://arxiv.org/abs/2308.02058](http://arxiv.org/abs/2308.02058)

    本文提出了一种将鲁莽行为引入基于矩阵分解的推荐系统学习过程的方法，通过控制风险水平来提高预测的数量和质量。

    

    包含可靠性测量的推荐系统往往在预测中更加保守，因为它们需要保持可靠性。这导致了这些系统可以提供的覆盖范围和新颖性的显著下降。在本文中，我们提出了在矩阵分解型推荐系统的学习过程中加入一项新的项，称为鲁莽行为，它可以控制在做出关于预测可靠性的决策时所希望的风险水平。实验结果表明，鲁莽行为不仅允许进行风险调控，还提高了推荐系统提供的预测的数量和质量。

    Recommender systems that include some reliability measure of their predictions tend to be more conservative in forecasting, due to their constraint to preserve reliability. This leads to a significant drop in the coverage and novelty that these systems can provide. In this paper, we propose the inclusion of a new term in the learning process of matrix factorization-based recommender systems, called recklessness, which enables the control of the risk level desired when making decisions about the reliability of a prediction. Experimental results demonstrate that recklessness not only allows for risk regulation but also improves the quantity and quality of predictions provided by the recommender system.
    

