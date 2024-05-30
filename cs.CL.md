# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deconstructing In-Context Learning: Understanding Prompts via Corruption](https://arxiv.org/abs/2404.02054) | 大型语言模型的能力在上下文中学习已经导致AI助手的急剧增长，其鲁棒性部分归因于对齐技术，然而这些助手使用的预训练模型在这方面却较为脆弱，构建高质量的骨干模型仍然是一个挑战。 |
| [^2] | [Going Beyond Word Matching: Syntax Improves In-context Example Selection for Machine Translation](https://arxiv.org/abs/2403.19285) | 本文提出了一种基于句法的机器翻译上下文例句选择方法，通过计算依存树之间的句法相似性，结合词级和句法水平标准选择例句，实验结果表明语法可以有效提升机器翻译上下文学习质量。 |
| [^3] | [Disambiguate Entity Matching through Relation Discovery with Large Language Models](https://arxiv.org/abs/2403.17344) | 通过定义实体之间的关系，解决实体匹配中的歧义问题 |
| [^4] | [Text clustering with LLM embeddings](https://arxiv.org/abs/2403.15112) | 研究表明，LLM嵌入能够捕捉结构化语言的细微差别，BERT在性能上领先于轻量级选项，增加嵌入维度和摘要技术并不一致地提高聚类效率 |
| [^5] | [Unfamiliar Finetuning Examples Control How Language Models Hallucinate](https://arxiv.org/abs/2403.05612) | 本文研究了大型语言模型如何产生幻觉，并提出通过调整微调示例的监督来控制其对不熟悉输入的预测。作者开发了一种基于RL的方法，更可靠地减轻了长篇生成任务中的幻觉。 |
| [^6] | [From One to Many: Expanding the Scope of Toxicity Mitigation in Language Models](https://arxiv.org/abs/2403.03893) | 该研究拓展了语言模型中毒性缓解的范围，涵盖了多语言环境，通过翻译数据评估和增强缓解技术，比较了不同缓解方法，并探讨了模型大小和数据量对缓解效果的影响。 |
| [^7] | [WebCiteS: Attributed Query-Focused Summarization on Chinese Web Search Results with Citations](https://arxiv.org/abs/2403.01774) | WebCiteS提出了一个带引文的查询焦点摘要任务，并发布了包含7k人工注释摘要及引文的中文数据集，以处理归因中存在的问题。 |
| [^8] | [Training-Free Long-Context Scaling of Large Language Models](https://arxiv.org/abs/2402.17463) | 提出了一种名为Dual Chunk Attention (DCA)的方法，可以使Llama2 70B在不需要持续训练的情况下支持超过100k令牌的上下文窗口，能够在长上下文任务中取得与微调模型相媲美甚至更好的性能。 |
| [^9] | [Less is More: Mitigating Multimodal Hallucination from an EOS Decision Perspective](https://arxiv.org/abs/2402.14545) | 本文研究了大型多模态模型中存在的多模态幻觉问题，发现通过模型基于视觉感知作出适当的EOS决策，可以减少持续输出，提出了两种缓解方法。 |
| [^10] | [Towards Building Multilingual Language Model for Medicine](https://arxiv.org/abs/2402.13963) | 本文提出了为医学领域构建多语言语言模型的三个关键贡献:构建了新的多语言医学语料库MMedC，提出了多语言医学多选问答基准MMedBench，并且通过在MMedC上进一步训练获得了性能优越的MMedLM 2模型。 |
| [^11] | [Benchmarking Knowledge Boundary for Large Language Model: A Different Perspective on Model Evaluation](https://arxiv.org/abs/2402.11493) | 引入了知识边界的概念，以涵盖语言模型内的无提示和有提示敏感性知识，通过避免提示敏感性，使得语言模型评估更可靠和稳健。 |
| [^12] | [LLMs as Bridges: Reformulating Grounded Multimodal Named Entity Recognition](https://arxiv.org/abs/2402.09989) | 本文提出了RiVEG，一个统一的框架，通过利用大型语言模型（LLMs）作为连接桥梁，将多模态命名实体识别重新构建为联合任务，解决了命名实体无法确定和指代表达与命名实体之间的区别的问题。 |
| [^13] | [SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks](https://arxiv.org/abs/2402.09025) | SLEB是一种通过消除冗余的Transformer块来优化LLM流程的新方法，它成功加速了LLM的推理过程。 |
| [^14] | [Beyond the Limits: A Survey of Techniques to Extend the Context Length in Large Language Models](https://arxiv.org/abs/2402.02244) | 这篇论文综述了近期为扩展大型语言模型中上下文长度而设计的技术和方法，并回顾了包括架构修改在内的多种技术，使得语言模型可以更有效地理解长上下文。 |
| [^15] | [Building Guardrails for Large Language Models](https://arxiv.org/abs/2402.01822) | 本文旨在为大型语言模型构建防护措施，并倡导采用系统化方法，通过与多学科团队合作来确定精确的技术要求，以减轻LLM的风险，并全面考虑不同LLM应用的多样化上下文。 |
| [^16] | [InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining](https://arxiv.org/abs/2310.07713) | InstructRetro是目前规模最大的使用检索预训练的LLM，扩展了基础模型Retro 48B，通过指令调优在各种零样例任务上取得显著改进。 |
| [^17] | [Evolutionary Computation in the Era of Large Language Model: Survey and Roadmap.](http://arxiv.org/abs/2401.10034) | 该论文调查了大语言模型和进化计算之间的相互作用，并提出了在黑盒设置下进一步提升大语言模型性能的优化框架，以及将大语言模型与进化算法结合应用于各种任务的方法。 |
| [^18] | [Code Simulation Challenges for Large Language Models.](http://arxiv.org/abs/2401.09074) | 大型语言模型在模拟计算机代码和算法执行方面遇到挑战，性能随着代码长度的增加而迅速下降。在处理短程序或标准过程时，它们能以低错误率按顺序执行指令，但对于复杂的程序，特别是包含关键路径和冗余指令的程序，模拟效果较差。我们提出了一种逐行模拟代码执行的方法来解决这个问题。 |
| [^19] | [Learn to Refuse: Making Large Language Models More Controllable and Reliable through Knowledge Scope Limitation and Refusal Mechanism.](http://arxiv.org/abs/2311.01041) | 本文提出了一种学会拒绝（L2R）的简单而有效的解决方案，通过引入拒绝机制，使大型语言模型（LLMs）能够识别和拒绝难以回答的问题，从而提高模型的可控性和可靠性。 |
| [^20] | [Debiasing Algorithm through Model Adaptation.](http://arxiv.org/abs/2310.18913) | 本论文提出了一种通过模型适应来检测和减轻语言模型中性别偏见的方法，并证明了该方法能够显著减少偏见同时保持模型性能。 |
| [^21] | [Why Can Large Language Models Generate Correct Chain-of-Thoughts?.](http://arxiv.org/abs/2310.13571) | 本文研究了大型语言模型如何生成连贯的思维链条，并通过建立几何收敛速率的框架来解释它与真实语言来源之间的相似性。这一研究结果为大型语言模型在推理任务中的性能提升提供了理论支持。 |
| [^22] | [Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models.](http://arxiv.org/abs/2307.01379) | 本论文研究了大型语言模型（LLMs）自动生成的关键词不平等问题，发现在估计不确定性时，重要的令牌和含有有限语义的句子被同等或更加重视。为了解决这个问题，提出了共同转移关注点来更好地估计不确定性。 |
| [^23] | [Think Before You Act: Decision Transformers with Internal Working Memory.](http://arxiv.org/abs/2305.16338) | 该论文提出了具有内部工作记忆模块的决策Transformer方法，以解决使用大型语言模型的决策代理在处理新任务上性能低下的问题。所提出的方法改善了训练效率和泛化能力，并进一步增强了转化决策制定代理对新任务的适应性。 |
| [^24] | [UP5: Unbiased Foundation Model for Fairness-aware Recommendation.](http://arxiv.org/abs/2305.12090) | 本研究提出了一种新颖的基础模型UP5，它采用反事实公平促进技术来消除大型语言模型中的偏见，从而实现面向公平性的推荐。 |

# 详细

[^1]: 拆解上下文学习: 通过破坏理解提示

    Deconstructing In-Context Learning: Understanding Prompts via Corruption

    [https://arxiv.org/abs/2404.02054](https://arxiv.org/abs/2404.02054)

    大型语言模型的能力在上下文中学习已经导致AI助手的急剧增长，其鲁棒性部分归因于对齐技术，然而这些助手使用的预训练模型在这方面却较为脆弱，构建高质量的骨干模型仍然是一个挑战。

    

    大型语言模型（LLMs）根据提供的提示“在上下文中学习”的能力已经导致它们的使用数量急剧增长，最终导致AI助手如ChatGPT、Claude和Bard的大量出现。这些AI助手被认为对提示的轻微修改具有鲁棒性，主要是由于使用了人类反馈的对齐技术。相比之下，它们使用作为骨干的基础预训练LLMs被认为在这方面比较脆弱。构建高质量的骨干模型仍然是一个核心挑战，评估其质量的常见方法是进行少样本评估。这种评估以对轻微提示修改和特定上下文示例选择的高度敏感而臭名昭著。先前的研究已经考察了修改提示的不同元素如何影响模型性能。然而，这些较早的研究往往集中在有限数量的具体提示上。

    arXiv:2404.02054v1 Announce Type: new  Abstract: The ability of large language models (LLMs) to "learn in context" based on the provided prompt has led to an explosive growth in their use, culminating in the proliferation of AI assistants such as ChatGPT, Claude, and Bard. These AI assistants are known to be robust to minor prompt modifications, mostly due to alignment techniques that use human feedback. In contrast, the underlying pre-trained LLMs they use as a backbone are known to be brittle in this respect. Building high-quality backbone models remains a core challenge, and a common approach to assessing their quality is to conduct few-shot evaluation. Such evaluation is notorious for being highly sensitive to minor prompt modifications, as well as the choice of specific in-context examples. Prior work has examined how modifying different elements of the prompt can affect model performance. However, these earlier studies tended to concentrate on a limited number of specific prompt 
    
[^2]: 超越词语匹配：句法改善上下文例句选择以提高机器翻译质量

    Going Beyond Word Matching: Syntax Improves In-context Example Selection for Machine Translation

    [https://arxiv.org/abs/2403.19285](https://arxiv.org/abs/2403.19285)

    本文提出了一种基于句法的机器翻译上下文例句选择方法，通过计算依存树之间的句法相似性，结合词级和句法水平标准选择例句，实验结果表明语法可以有效提升机器翻译上下文学习质量。

    

    arXiv:2403.19285v1 公告类型: 新的 摘要: 在大语言模型（LLMs）时代，上下文学习（ICL）是一种流行的提示策略，其中展示了一些示例以唤起LLMs在给定任务上的能力。如何选择信息量大的例句仍然是一个悬而未决的问题。先前关于机器翻译（MT）的上下文例句选择的作品侧重于表面的词级特征，而忽略了深层次的句法层次知识。在本文中，我们提出了一种基于语法的机器翻译上下文例句选择方法，通过使用多项式距离计算依赖树之间的句法相似性。此外，我们提出了一种综合策略，将通过词级和句法水平标准选择的例句进行组合。对英语和6种常见语言之间的实验结果表明，语法可以有效提升MT的ICL，获得了在12个翻译方向中11个方向上最高的COMET分数。

    arXiv:2403.19285v1 Announce Type: new  Abstract: In-context learning (ICL) is the trending prompting strategy in the era of large language models (LLMs), where a few examples are demonstrated to evoke LLMs' power for a given task. How to select informative examples remains an open issue. Previous works on in-context example selection for machine translation (MT) focus on superficial word-level features while ignoring deep syntax-level knowledge. In this paper, we propose a syntax-based in-context example selection method for MT, by computing the syntactic similarity between dependency trees using Polynomial Distance. In addition, we propose an ensemble strategy combining examples selected by both word-level and syntax-level criteria. Experimental results between English and 6 common languages indicate that syntax can effectively enhancing ICL for MT, obtaining the highest COMET scores on 11 out of 12 translation directions.
    
[^3]: 通过大型语言模型进行关系发现的实体匹配消歧

    Disambiguate Entity Matching through Relation Discovery with Large Language Models

    [https://arxiv.org/abs/2403.17344](https://arxiv.org/abs/2403.17344)

    通过定义实体之间的关系，解决实体匹配中的歧义问题

    

    实体匹配是数据集成和清洗中的一个关键挑战，对于模糊连接和数据重复消除等任务至关重要。传统方法集中在克服模糊术语表示，例如编辑距离、Jaccard相似性，以及最近的嵌入和深度神经网络，包括来自大型语言模型（LLMs）如GPT的进展。然而，实体匹配中的核心挑战超越了术语模糊性，而是在定义何为“匹配”时的歧义，特别是在与外部数据库集成时。这种歧义是由于实体之间在细节和粒度方面存在差异引起的，这使得确切匹配变得复杂。我们提出了一种新方法，将焦点从纯粹识别语义相似性转变为理解和定义实体之间的“关系”作为解决匹配中的歧义至关重要。通过预定义一组与任务相关的关系，可帮助解决匹配中的歧义。

    arXiv:2403.17344v1 Announce Type: cross  Abstract: Entity matching is a critical challenge in data integration and cleaning, central to tasks like fuzzy joins and deduplication. Traditional approaches have focused on overcoming fuzzy term representations through methods such as edit distance, Jaccard similarity, and more recently, embeddings and deep neural networks, including advancements from large language models (LLMs) like GPT. However, the core challenge in entity matching extends beyond term fuzziness to the ambiguity in defining what constitutes a "match," especially when integrating with external databases. This ambiguity arises due to varying levels of detail and granularity among entities, complicating exact matches. We propose a novel approach that shifts focus from purely identifying semantic similarities to understanding and defining the "relations" between entities as crucial for resolving ambiguities in matching. By predefining a set of relations relevant to the task at
    
[^4]: 使用LLM嵌入进行文本聚类

    Text clustering with LLM embeddings

    [https://arxiv.org/abs/2403.15112](https://arxiv.org/abs/2403.15112)

    研究表明，LLM嵌入能够捕捉结构化语言的细微差别，BERT在性能上领先于轻量级选项，增加嵌入维度和摘要技术并不一致地提高聚类效率

    

    文本聚类是组织不断增长的数字内容的重要方法，有助于结构化和发现未分类数据中的隐藏模式。在这项研究中，我们调查了不同文本嵌入（特别是大型语言模型LLMs中使用的）和聚类算法如何影响文本数据集的聚类方式。进行了一系列实验以评估嵌入是如何影响聚类结果的，以及通过摘要进行降维和嵌入大小调整的作用。结果显示，LLM嵌入在捕获结构化语言的细微差别方面表现出色，而BERT在性能上领先于轻量级选项。此外，我们发现增加嵌入维度和摘要技术并不一致地提高聚类效率，这表明这些策略需要仔细分析才能在实际模型中使用。这些结果突出了一种

    arXiv:2403.15112v1 Announce Type: cross  Abstract: Text clustering is an important approach for organising the growing amount of digital content, helping to structure and find hidden patterns in uncategorised data. In this research, we investigated how different textual embeddings - particularly those used in large language models (LLMs) - and clustering algorithms affect how text datasets are clustered. A series of experiments were conducted to assess how embeddings influence clustering results, the role played by dimensionality reduction through summarisation, and embedding size adjustment. Results reveal that LLM embeddings excel at capturing the nuances of structured language, while BERT leads the lightweight options in performance. In addition, we find that increasing embedding dimensionality and summarisation techniques do not uniformly improve clustering efficiency, suggesting that these strategies require careful analysis to use in real-life models. These results highlight a co
    
[^5]: 不熟悉的微调示例控制语言模型如何产生幻觉

    Unfamiliar Finetuning Examples Control How Language Models Hallucinate

    [https://arxiv.org/abs/2403.05612](https://arxiv.org/abs/2403.05612)

    本文研究了大型语言模型如何产生幻觉，并提出通过调整微调示例的监督来控制其对不熟悉输入的预测。作者开发了一种基于RL的方法，更可靠地减轻了长篇生成任务中的幻觉。

    

    大型语言模型（LLMs）倾向于生成听起来令人信服但事实不正确的响应，特别是当在不熟悉的概念上进行查询时。本文探讨了调整后的LLMs如何产生幻觉的基本机制。我们的调查揭示了一个有趣的模式：随着输入变得更不熟悉，LLMs的输出倾向于默认为"含糊其词"的预测，其形式受微调数据中不熟悉示例监督方式的影响。因此，通过策略性地修改这些示例的监督，我们可以控制LLM对不熟悉输入的预测（例如，教会它们说“我不知道”）。基于这些原则，我们开发了一种RL方法，通过解决奖励模型幻觉带来的挑战，更可靠地减轻长篇生成任务的幻觉。我们通过在MMLU上的多选QA中进行一系列受控实验来验证我们的发现。

    arXiv:2403.05612v1 Announce Type: cross  Abstract: Large language models (LLMs) have a tendency to generate plausible-sounding yet factually incorrect responses, especially when queried on unfamiliar concepts. In this work, we explore the underlying mechanisms that govern how finetuned LLMs hallucinate. Our investigation reveals an interesting pattern: as inputs become more unfamiliar, LLM outputs tend to default towards a ``hedged'' prediction, whose form is determined by how the unfamiliar examples in the finetuning data are supervised. Thus, by strategically modifying these examples' supervision, we can control LLM predictions for unfamiliar inputs (e.g., teach them to say ``I don't know''). Based on these principles, we develop an RL approach that more reliably mitigates hallucinations for long-form generation tasks, by tackling the challenges presented by reward model hallucinations. We validate our findings with a series of controlled experiments in multiple-choice QA on MMLU, as
    
[^6]: 从单一到多样：拓展语言模型中毒性缓解的范围

    From One to Many: Expanding the Scope of Toxicity Mitigation in Language Models

    [https://arxiv.org/abs/2403.03893](https://arxiv.org/abs/2403.03893)

    该研究拓展了语言模型中毒性缓解的范围，涵盖了多语言环境，通过翻译数据评估和增强缓解技术，比较了不同缓解方法，并探讨了模型大小和数据量对缓解效果的影响。

    

    迄今为止，语言模型中的毒性缓解几乎完全集中在单语言环境中。随着语言模型拥抱多语言能力，我们的安全措施跟上步伐至关重要。我们意识到了这一研究空白，我们的方法将传统的毒性缓解范围扩展到应对多语言带来的复杂性。在缺乏跨语言的足够标注数据集的情况下，我们使用翻译数据来评估和增强我们的缓解技术。我们还在静态和持续毒性缓解场景下比较了微调缓解方法和检索增强技术。这使我们能够检验翻译质量和跨语言转移对毒性缓解的影响。我们还探讨了模型大小和数据数量如何影响这些缓解工作的成功。涵盖了九种语言，我们的研究代表了广泛的语言学领域。

    arXiv:2403.03893v1 Announce Type: cross  Abstract: To date, toxicity mitigation in language models has almost entirely been focused on single-language settings. As language models embrace multilingual capabilities, it's crucial our safety measures keep pace. Recognizing this research gap, our approach expands the scope of conventional toxicity mitigation to address the complexities presented by multiple languages. In the absence of sufficient annotated datasets across languages, we employ translated data to evaluate and enhance our mitigation techniques. We also compare finetuning mitigation approaches against retrieval-augmented techniques under both static and continual toxicity mitigation scenarios. This allows us to examine the effects of translation quality and the cross-lingual transfer on toxicity mitigation. We also explore how model size and data quantity affect the success of these mitigation efforts. Covering nine languages, our study represents a broad array of linguistic f
    
[^7]: WebCiteS: 在中国网页搜索结果上进行带引文的查询焦点摘要

    WebCiteS: Attributed Query-Focused Summarization on Chinese Web Search Results with Citations

    [https://arxiv.org/abs/2403.01774](https://arxiv.org/abs/2403.01774)

    WebCiteS提出了一个带引文的查询焦点摘要任务，并发布了包含7k人工注释摘要及引文的中文数据集，以处理归因中存在的问题。

    

    arXiv:2403.01774v1 声明类型：新摘要：增强大型语言模型（LLMs）中的归因是一项关键任务。一个可行的方法是使LLMs能够引用支持其生成的外部来源。然而，该领域现有数据集和评估方法仍存在明显限制。在这项工作中，我们制定了带引文的查询焦点摘要（AQFS）任务，并提出了WebCiteS，这是一个包含7k人工注释摘要及引文的中文数据集。WebCiteS源自现实用户查询和网页搜索结果，为模型训练和评估提供了宝贵资源。之前关于归因评估的工作未能区分基于事实错误和引文错误。他们亦未能自动验证那些部分依赖多个来源的句子。我们通过开发详细的度量标准并使自动评估器能够将句子分解为子主张以解决这些问题。

    arXiv:2403.01774v1 Announce Type: new  Abstract: Enhancing the attribution in large language models (LLMs) is a crucial task. One feasible approach is to enable LLMs to cite external sources that support their generations. However, existing datasets and evaluation methods in this domain still exhibit notable limitations. In this work, we formulate the task of attributed query-focused summarization (AQFS) and present WebCiteS, a Chinese dataset featuring 7k human-annotated summaries with citations. WebCiteS derives from real-world user queries and web search results, offering a valuable resource for model training and evaluation. Prior works in attribution evaluation do not differentiate between groundedness errors and citation errors. They also fall short in automatically verifying sentences that draw partial support from multiple sources. We tackle these issues by developing detailed metrics and enabling the automatic evaluator to decompose the sentences into sub-claims for fine-grain
    
[^8]: 无须训练的大语言模型长上下文扩展

    Training-Free Long-Context Scaling of Large Language Models

    [https://arxiv.org/abs/2402.17463](https://arxiv.org/abs/2402.17463)

    提出了一种名为Dual Chunk Attention (DCA)的方法，可以使Llama2 70B在不需要持续训练的情况下支持超过100k令牌的上下文窗口，能够在长上下文任务中取得与微调模型相媲美甚至更好的性能。

    

    大语言模型（LLMs）在处理和生成连贯文本时，当输入令牌数量超过它们的预训练长度时，其能力会明显减弱。鉴于使用更长序列进行大规模模型微调的昂贵开销，我们提出了Dual Chunk Attention（DCA），它使Llama2 70B能够支持超过100k令牌的上下文窗口，而无需持续训练。通过将长序列的注意力计算分解为基于块的模块，DCA成功捕获了相同块内（Intra-Chunk）和不同块之间（Inter-Chunk）令牌的相对位置信息，并能与Flash Attention无缝集成。除了其惊人的外推能力外，DCA在实际长上下文任务上实现了与或甚至优于微调模型相当的性能。与专有模型相比，我们的无须训练的70B模型取得了

    arXiv:2402.17463v1 Announce Type: new  Abstract: The ability of Large Language Models (LLMs) to process and generate coherent text is markedly weakened when the number of input tokens exceeds their pretraining length. Given the expensive overhead of finetuning large-scale models with longer sequences, we propose Dual Chunk Attention (DCA), which enables Llama2 70B to support context windows of more than 100k tokens without continual training. By decomposing the attention computation for long sequences into chunk-based modules, DCA manages to effectively capture the relative positional information of tokens within the same chunk (Intra-Chunk) and across distinct chunks (Inter-Chunk), as well as integrates seamlessly with Flash Attention. In addition to its impressive extrapolation capability, DCA achieves performance on practical long-context tasks that is comparable to or even better than that of finetuned models. When compared with proprietary models, our training-free 70B model attai
    
[^9]: 减少是有益的：从EOS决策角度缓解多模态幻觉

    Less is More: Mitigating Multimodal Hallucination from an EOS Decision Perspective

    [https://arxiv.org/abs/2402.14545](https://arxiv.org/abs/2402.14545)

    本文研究了大型多模态模型中存在的多模态幻觉问题，发现通过模型基于视觉感知作出适当的EOS决策，可以减少持续输出，提出了两种缓解方法。

    

    大型多模态模型（LMMs）经常遭受多模态幻觉，即它们可能创造出在视觉输入中并不存在的内容。本文探讨了这个问题的一个新角度：过于详细的训练数据妨碍了模型及时终止生成，导致超出视觉感知限制的持续输出。通过研究模型如何通过EOS（特殊的句子结尾标记）来决定终止生成，我们发现模型通过将生成的文本与图像进行比较来评估整个序列的完整性。这一观察表明，模型具有基于其视觉感知进行适当EOS决策的潜力，以避免过长的输出。为了利用这种潜力，我们探讨了两种缓解多模态幻觉的方法：通过学习常规指示实现模型减少幻觉的训练目标

    arXiv:2402.14545v1 Announce Type: new  Abstract: Large Multimodal Models (LMMs) often suffer from multimodal hallucinations, wherein they may create content that is not present in the visual inputs. In this paper, we explore a new angle of this issue: overly detailed training data hinders the model's ability to timely terminate generation, leading to continued outputs beyond visual perception limits. By investigating how the model decides to terminate generation with EOS, the special end-of-sentence token, we find that the model assesses the completeness of the entire sequence by comparing the generated text with the image. This observation suggests that the model possesses an inherent potential of making proper EOS decisions based on its visual perception to avoid overly lengthy outputs. To take advantage of such potential, we explore two methods to mitigate multimodal hallucinations: a training objective that enables the model to reduce hallucinations by learning from regular instruc
    
[^10]: 为医学构建多语言语言模型

    Towards Building Multilingual Language Model for Medicine

    [https://arxiv.org/abs/2402.13963](https://arxiv.org/abs/2402.13963)

    本文提出了为医学领域构建多语言语言模型的三个关键贡献:构建了新的多语言医学语料库MMedC，提出了多语言医学多选问答基准MMedBench，并且通过在MMedC上进一步训练获得了性能优越的MMedLM 2模型。

    

    本文旨在开发一种面向医学的开源多语言语言模型，使得更广泛的语言多样性受众受益。我们的工作主要贡献体现在以下几个方面:首先，针对多语言医学特定适应性，我们构建了一个新的多语言医学语料库，包含大约25.5B个tokens，覆盖了6种主要语言，被称为MMedC，这使得现有通用LLM能够进行自回归训练。其次，为了监测医学领域多语言LLM的发展，我们提出了一个新的带有解释的多语言医学多选问答基准，称为MMedBench；第三，我们评估了一些流行的开源大型语言模型(LLMs)在我们的基准上的表现，以及那些在MMedC上进一步进行自回归训练的模型，最终，我们的最终模型，命名为MMedLM 2，仅有7B参数，取得了卓越的性能。

    arXiv:2402.13963v1 Announce Type: new  Abstract: In this paper, we aim to develop an open-source, multilingual language model for medicine, that the benefits a wider, linguistically diverse audience from different regions. In general, we present the contribution from the following aspects: first, for multilingual medical-specific adaptation, we construct a new multilingual medical corpus, that contains approximately 25.5B tokens encompassing 6 main languages, termed as MMedC, that enables auto-regressive training for existing general LLMs. second, to monitor the development of multilingual LLMs in medicine, we propose a new multilingual medical multi-choice question-answering benchmark with rationale, termed as MMedBench; third, we have assessed a number of popular, opensource large language models (LLMs) on our benchmark, along with those further auto-regressive trained on MMedC, as a result, our final model, termed as MMedLM 2, with only 7B parameters, achieves superior performance c
    
[^11]: 基于大语言模型的知识边界基准：对模型评估的另一种视角

    Benchmarking Knowledge Boundary for Large Language Model: A Different Perspective on Model Evaluation

    [https://arxiv.org/abs/2402.11493](https://arxiv.org/abs/2402.11493)

    引入了知识边界的概念，以涵盖语言模型内的无提示和有提示敏感性知识，通过避免提示敏感性，使得语言模型评估更可靠和稳健。

    

    最近几年，在大语言模型的发展中取得了实质性进展，在各种任务中取得了显著的性能。为了评估语言模型的知识能力，先前的研究提出了许多基于问答对的基准。我们认为，使用固定问题或有限的释义作为查询来评估语言模型是不可靠和全面的，因为语言模型对提示很敏感。因此，我们引入了一个名为知识边界的新概念，以包含语言模型内的无提示和有提示敏感性知识。知识边界避免了语言模型评估中的提示敏感性，使其更可靠和稳健。为了探索给定模型的知识边界，我们提出了带有语义约束的投影梯度下降方法，这是一种旨在识别每个部分的最佳提示的新算法。

    arXiv:2402.11493v1 Announce Type: new  Abstract: In recent years, substantial advancements have been made in the development of large language models, achieving remarkable performance across diverse tasks. To evaluate the knowledge ability of language models, previous studies have proposed lots of benchmarks based on question-answering pairs. We argue that it is not reliable and comprehensive to evaluate language models with a fixed question or limited paraphrases as the query, since language models are sensitive to prompt. Therefore, we introduce a novel concept named knowledge boundary to encompass both prompt-agnostic and prompt-sensitive knowledge within language models. Knowledge boundary avoids prompt sensitivity in language model evaluations, rendering them more dependable and robust. To explore the knowledge boundary for a given model, we propose projected gradient descent method with semantic constraints, a new algorithm designed to identify the optimal prompt for each piece o
    
[^12]: LLMs作为桥梁：重新构建基于多模态图像的命名实体识别

    LLMs as Bridges: Reformulating Grounded Multimodal Named Entity Recognition

    [https://arxiv.org/abs/2402.09989](https://arxiv.org/abs/2402.09989)

    本文提出了RiVEG，一个统一的框架，通过利用大型语言模型（LLMs）作为连接桥梁，将多模态命名实体识别重新构建为联合任务，解决了命名实体无法确定和指代表达与命名实体之间的区别的问题。

    

    Grounded Multimodal Named Entity Recognition (GMNER) 是一个新兴的多模态任务，旨在识别命名实体、实体类型及其对应的视觉区域。GMNER任务具有两个挑战性质：1）社交媒体中图像和文本之间的弱相关性导致大部分命名实体难以确定；2）常用于类似任务的粗粒度指代表达与细粒度命名实体之间存在明显区别。本文提出了RiVEG，一个统一的框架，通过利用大型语言模型（LLMs）作为连接桥梁，将GMNER重新构建为联合MNER-VE-VG任务。这种重新构建带来了两个好处：1）保持了最佳的MNER性能，消除了使用目标检测方法预提取区域特征的需求，自然解决了这两个挑战。

    arXiv:2402.09989v1 Announce Type: cross  Abstract: Grounded Multimodal Named Entity Recognition (GMNER) is a nascent multimodal task that aims to identify named entities, entity types and their corresponding visual regions. GMNER task exhibits two challenging properties: 1) The weak correlation between image-text pairs in social media results in a significant portion of named entities being ungroundable. 2) There exists a distinction between coarse-grained referring expressions commonly used in similar tasks (e.g., phrase localization, referring expression comprehension) and fine-grained named entities. In this paper, we propose RiVEG, a unified framework that reformulates GMNER into a joint MNER-VE-VG task by leveraging large language models (LLMs) as a connecting bridge. This reformulation brings two benefits: 1) It maintains the optimal MNER performance and eliminates the need for employing object detection methods to pre-extract regional features, thereby naturally addressing two m
    
[^13]: SLEB: 通过冗余验证和消除Transformer块优化LLM的流程

    SLEB: Streamlining LLMs through Redundancy Verification and Elimination of Transformer Blocks

    [https://arxiv.org/abs/2402.09025](https://arxiv.org/abs/2402.09025)

    SLEB是一种通过消除冗余的Transformer块来优化LLM流程的新方法，它成功加速了LLM的推理过程。

    

    大型语言模型（LLM）在各种自然语言处理任务中证明了其高效性。然而，它们庞大的参数数量给实际部署带来了重大挑战。精简，一种旨在减小LLM大小和复杂度的技术，通过从网络中删除冗余组件提供了潜在解决方案。尽管精简有希望，但现有方法往往难以实现显著的端到端LLM推理加速。本文中，我们引入了SLEB，一种通过消除冗余的Transformer块来优化LLM流程的新方法。我们选择Transformer块作为精简的基本单位，因为LLM在相邻块的输出之间具有块级别的冗余和高相似性。这个选择使我们能够有效地增强LLM的处理速度。我们的实验证明，SLEB成功加速了LLM的推理过程。

    arXiv:2402.09025v1 Announce Type: new Abstract: Large language models (LLMs) have proven to be highly effective across various natural language processing tasks. However, their large number of parameters poses significant challenges for practical deployment. Pruning, a technique aimed at reducing the size and complexity of LLMs, offers a potential solution by removing redundant components from the network. Despite the promise of pruning, existing methods often struggle to achieve substantial end-to-end LLM inference speedup. In this paper, we introduce SLEB, a novel approach designed to streamline LLMs by eliminating redundant transformer blocks. We choose the transformer block as the fundamental unit for pruning, because LLMs exhibit block-level redundancy with high similarity between the outputs of neighboring blocks. This choice allows us to effectively enhance the processing speed of LLMs. Our experimental results demonstrate that SLEB successfully accelerates LLM inference without
    
[^14]: 超越极限：扩展大型语言模型中上下文长度的技术综述

    Beyond the Limits: A Survey of Techniques to Extend the Context Length in Large Language Models

    [https://arxiv.org/abs/2402.02244](https://arxiv.org/abs/2402.02244)

    这篇论文综述了近期为扩展大型语言模型中上下文长度而设计的技术和方法，并回顾了包括架构修改在内的多种技术，使得语言模型可以更有效地理解长上下文。

    

    近期，大型语言模型（LLMs）展现出了令人惊异的能力，包括理解上下文、进行逻辑推理和生成响应。然而，这是以严格的计算和内存要求为代价的，限制了它们有效支持长输入序列的能力。本综述全面回顾了最近为扩展LLMs序列长度而设计的技术和方法，从而增强其对长上下文理解的能力。具体而言，我们回顾和分类了各种技术，包括修改位置编码和修改注意机制等架构修改，旨在增强对更长序列的处理，同时避免计算需求的成比例增加。本研究探讨的多样方法可以在LLMs的不同阶段（即训练、微调和推理）中利用。这使得LLMs可以有效地处理长序列并提升对长上下文的理解能力。

    Recently, large language models (LLMs) have shown remarkable capabilities including understanding context, engaging in logical reasoning, and generating responses. However, this is achieved at the expense of stringent computational and memory requirements, hindering their ability to effectively support long input sequences. This survey provides an inclusive review of the recent techniques and methods devised to extend the sequence length in LLMs, thereby enhancing their capacity for long-context understanding. In particular, we review and categorize a wide range of techniques including architectural modifications, such as modified positional encoding and altered attention mechanisms, which are designed to enhance the processing of longer sequences while avoiding a proportional increase in computational requirements. The diverse methodologies investigated in this study can be leveraged across different phases of LLMs, i.e., training, fine-tuning and inference. This enables LLMs to effic
    
[^15]: 为大型语言模型构建防护措施

    Building Guardrails for Large Language Models

    [https://arxiv.org/abs/2402.01822](https://arxiv.org/abs/2402.01822)

    本文旨在为大型语言模型构建防护措施，并倡导采用系统化方法，通过与多学科团队合作来确定精确的技术要求，以减轻LLM的风险，并全面考虑不同LLM应用的多样化上下文。

    

    随着大型语言模型（LLM）越来越多地融入我们的日常生活中，识别和减轻它们的风险变得至关重要，特别是当这些风险对人类用户和社会产生深远影响时。防护措施，即过滤LLM的输入或输出，已经成为一种核心的安全技术。本文深入研究了当前的开源解决方案（Llama Guard，Nvidia NeMo，Guardrails AI），讨论了构建更完整解决方案的挑战和路径。基于前期研究的有力证据，我们倡导采用系统化方法构建LLM的防护措施，全面考虑不同LLM应用的多样化上下文。我们建议通过与多学科团队的合作，采用社会技术方法来确定精确的技术要求，探索面向需求复杂性的先进神经符号实现，并开展验证和测试。

    As Large Language Models (LLMs) become more integrated into our daily lives, it is crucial to identify and mitigate their risks, especially when the risks can have profound impacts on human users and societies. Guardrails, which filter the inputs or outputs of LLMs, have emerged as a core safeguarding technology. This position paper takes a deep look at current open-source solutions (Llama Guard, Nvidia NeMo, Guardrails AI), and discusses the challenges and the road towards building more complete solutions. Drawing on robust evidence from previous research, we advocate for a systematic approach to construct guardrails for LLMs, based on comprehensive consideration of diverse contexts across various LLMs applications. We propose employing socio-technical methods through collaboration with a multi-disciplinary team to pinpoint precise technical requirements, exploring advanced neural-symbolic implementations to embrace the complexity of the requirements, and developing verification and t
    
[^16]: InstructRetro: 检索增强的预训练中指令调优

    InstructRetro: Instruction Tuning post Retrieval-Augmented Pretraining

    [https://arxiv.org/abs/2310.07713](https://arxiv.org/abs/2310.07713)

    InstructRetro是目前规模最大的使用检索预训练的LLM，扩展了基础模型Retro 48B，通过指令调优在各种零样例任务上取得显著改进。

    

    使用检索增强技术对自回归大型语言模型（LLM）进行预训练可以提高困惑度和事实准确性。然而，现有的预训练检索增强LLM的规模仍然有限（如Retro具有75亿个参数），这限制了指令调优和零样例泛化的效果。本文介绍了Retro 48B，这是目前规模最大的使用检索预训练的LLM。具体来说，我们使用检索技术从1.2万亿个标记中继续预训练一个43B的GPT模型，并借助Retro方法将其扩展到4800亿个参数。值得注意的是，所得到的基础模型Retro 48B在困惑度方面显著优于仅使用1.2万亿个标记进行训练的43B GPT模型，且只增加了2.58%的GPU使用时间，展示了该方法的显著扩展潜力。在对Retro进行指令调优后，InstructRetro在各种零样例任务上表现出显著的改进。

    Pretraining auto-regressive large language models (LLMs) with retrieval demonstrates better perplexity and factual accuracy by leveraging external databases. However, the size of existing pretrained retrieval-augmented LLM is still limited (e.g., Retro has 7.5B parameters), which limits the effectiveness of instruction tuning and zero-shot generalization. In this work, we introduce Retro 48B, the largest LLM pretrained with retrieval. Specifically, we continue to pretrain a 43B GPT model on additional 100 billion tokens using the Retro augmentation method by retrieving from 1.2 trillion tokens. Notably, the obtained foundation model, Retro 48B, largely outperforms the counterpart GPT 43B trained on 1.2T tokens in terms of perplexity with only 2.58% additional GPU hours, demonstrating the significant scaling potential of the method. After instruction tuning on Retro, InstructRetro demonstrates significant improvement over the instruction tuned GPT on a wide range of zero-shot tasks. Spe
    
[^17]: 大语言模型时代的进化计算：调查与路线图

    Evolutionary Computation in the Era of Large Language Model: Survey and Roadmap. (arXiv:2401.10034v1 [cs.NE])

    [http://arxiv.org/abs/2401.10034](http://arxiv.org/abs/2401.10034)

    该论文调查了大语言模型和进化计算之间的相互作用，并提出了在黑盒设置下进一步提升大语言模型性能的优化框架，以及将大语言模型与进化算法结合应用于各种任务的方法。

    

    大型语言模型（LLMs）是基于Transformer架构，在多样的数据上进行大规模预训练的，它们不仅在自然语言处理领域引起了革命，还将其能力扩展到了各个领域，迈向了人工通用智能的重要一步。尽管进化算法（EAs）与LLMs在目标和方法论上存在差异，但它们之间的相互作用揭示了有趣的相似之处，特别是在他们共同的优化性质、黑盒特性和处理复杂问题的能力方面。与此同时，进化算法不仅可以为LLM在黑盒设置下提供优化框架，还可以在应用中为LLM赋予灵活的全局搜索和迭代机制。另一方面，LLM丰富的领域知识使得进化算法可以进行更智能的搜索，而其文本处理能力则有助于将进化算法应用于各种任务。基于它们的互补优势，本文提出了一份调查和路线图。

    Large Language Models (LLMs), built upon Transformer-based architectures with massive pretraining on diverse data, have not only revolutionized natural language processing but also extended their prowess to various domains, marking a significant stride towards artificial general intelligence. The interplay between LLMs and Evolutionary Algorithms (EAs), despite differing in objectives and methodologies, reveals intriguing parallels, especially in their shared optimization nature, black-box characteristics, and proficiency in handling complex problems. Meanwhile, EA can not only provide an optimization framework for LLM's further enhancement under black-box settings but also empower LLM with flexible global search and iterative mechanism in applications. On the other hand, LLM's abundant domain knowledge enables EA to perform smarter searches, while its text processing capability assist in deploying EA across various tasks. Based on their complementary advantages, this paper presents a 
    
[^18]: 大型语言模型中的代码模拟挑战

    Code Simulation Challenges for Large Language Models. (arXiv:2401.09074v1 [cs.LG])

    [http://arxiv.org/abs/2401.09074](http://arxiv.org/abs/2401.09074)

    大型语言模型在模拟计算机代码和算法执行方面遇到挑战，性能随着代码长度的增加而迅速下降。在处理短程序或标准过程时，它们能以低错误率按顺序执行指令，但对于复杂的程序，特别是包含关键路径和冗余指令的程序，模拟效果较差。我们提出了一种逐行模拟代码执行的方法来解决这个问题。

    

    我们调查了大型语言模型（LLMs）在模拟计算机代码和算法执行方面的能力。我们首先研究了直线程序，并展示了当前LLMs在处理这样简单的程序时表现出的性能较差——性能随着代码长度的增加而迅速下降。接着，我们研究了LLMs在模拟包含关键路径和冗余指令的程序方面的能力。我们还通过排序算法和嵌套循环超越了直线程序的模拟，并展示了程序的计算复杂性直接影响LLMs模拟其执行的能力。我们观察到LLMs只有在处理短程序或标准过程时才能以低错误率按顺序执行指令。LLMs的代码模拟与它们的模式识别和记忆能力存在矛盾：在记忆对任务有害的情况下，我们提出了一种新的提示方法，逐行模拟代码的执行。

    We investigate the extent to which Large Language Models (LLMs) can simulate the execution of computer code and algorithms. We begin by looking straight line programs, and show that current LLMs demonstrate poor performance even with such simple programs -- performance rapidly degrades with the length of code. We then investigate the ability of LLMs to simulate programs that contain critical paths and redundant instructions. We also go beyond straight line program simulation with sorting algorithms and nested loops, and we show the computational complexity of a routine directly affects the ability of an LLM to simulate its execution. We observe that LLMs execute instructions sequentially and with a low error margin only for short programs or standard procedures. LLMs' code simulation is in tension with their pattern recognition and memorisation capabilities: on tasks where memorisation is detrimental, we propose a novel prompting method to simulate code execution line by line. Empirica
    
[^19]: 学会拒绝：通过知识范围限制和拒绝机制使大型语言模型更可控和可靠

    Learn to Refuse: Making Large Language Models More Controllable and Reliable through Knowledge Scope Limitation and Refusal Mechanism. (arXiv:2311.01041v1 [cs.CL])

    [http://arxiv.org/abs/2311.01041](http://arxiv.org/abs/2311.01041)

    本文提出了一种学会拒绝（L2R）的简单而有效的解决方案，通过引入拒绝机制，使大型语言模型（LLMs）能够识别和拒绝难以回答的问题，从而提高模型的可控性和可靠性。

    

    大型语言模型（LLMs）展示了令人印象深刻的语言理解和生成能力，使它们能够回答各个领域的广泛问题。然而，这些模型并不完美，经常产生含有错误或错误信息的回答。这些不准确性，通常称为幻觉，使得LLMs在许多场景中不可靠甚至不可用。本文的重点是在LLMs中缓解幻觉问题，特别是在问答环境中。我们探索了一种拒绝机制，指导LLMs拒绝回答具有挑战性的问题以避免错误。我们提出了一个简单而有效的解决方案Learn to Refuse (L2R)，它将拒绝机制纳入到LLMs中，使其能够识别和拒绝那些它们难以回答的问题。为了实现这一点，我们利用结构化知识库来表示所有LLMs所需要的知识。

    Large language models (LLMs) have demonstrated impressive language understanding and generation capabilities, enabling them to answer a wide range of questions across various domains. However, these models are not flawless and often produce responses that contain errors or misinformation. These inaccuracies, commonly referred to as hallucinations, render LLMs unreliable and even unusable in many scenarios. In this paper, our focus is on mitigating the issue of hallucination in LLMs, particularly in the context of question-answering. Instead of attempting to answer all questions, we explore a refusal mechanism that instructs LLMs to refuse to answer challenging questions in order to avoid errors. We then propose a simple yet effective solution called Learn to Refuse (L2R), which incorporates the refusal mechanism to enable LLMs to recognize and refuse to answer questions that they find difficult to address. To achieve this, we utilize a structured knowledge base to represent all the LLM
    
[^20]: 通过模型适应来去除偏见算法

    Debiasing Algorithm through Model Adaptation. (arXiv:2310.18913v1 [cs.CL])

    [http://arxiv.org/abs/2310.18913](http://arxiv.org/abs/2310.18913)

    本论文提出了一种通过模型适应来检测和减轻语言模型中性别偏见的方法，并证明了该方法能够显著减少偏见同时保持模型性能。

    

    大型语言模型正在成为各种语言任务的首选解决方案。然而，随着容量的增长，模型很容易依赖训练数据中存在的偏见和刻板印象所产生的虚假相关性。本研究提出了一种新颖的方法来检测和减轻语言模型中的性别偏见。我们进行因果分析，以识别问题模型组件，并发现中上层前馈层最容易传递偏见。根据分析结果，我们通过线性投影将这些层乘以模型进行适应。我们的方法DAMA通过各种度量指标明显减少了偏见，同时保持模型在后续任务中的性能。我们发布了我们的方法和模型的代码，通过重新训练，保持了LLaMA的最先进性能，同时偏见显著减少。

    Large language models are becoming the go-to solution for various language tasks. However, with growing capacity, models are prone to rely on spurious correlations stemming from biases and stereotypes present in the training data. This work proposes a novel method for detecting and mitigating gender bias in language models. We perform causal analysis to identify problematic model components and discover that mid-upper feed-forward layers are most prone to convey biases. Based on the analysis results, we adapt the model by multiplying these layers by a linear projection. Our titular method, DAMA, significantly decreases bias as measured by diverse metrics while maintaining the model's performance on downstream tasks. We release code for our method and models, which retrain LLaMA's state-of-the-art performance while being significantly less biased.
    
[^21]: 大型语言模型为何能生成正确的思维链条？

    Why Can Large Language Models Generate Correct Chain-of-Thoughts?. (arXiv:2310.13571v1 [cs.CL])

    [http://arxiv.org/abs/2310.13571](http://arxiv.org/abs/2310.13571)

    本文研究了大型语言模型如何生成连贯的思维链条，并通过建立几何收敛速率的框架来解释它与真实语言来源之间的相似性。这一研究结果为大型语言模型在推理任务中的性能提升提供了理论支持。

    

    本文深入研究了大型语言模型（LLM）的能力，特别关注推动对思维链条引发能力的理论理解。我们研究了如何有效地诱导LLM生成连贯的思维链条。为了实现这一目标，我们引入了一个针对自然语言生成的两级分层图模型。在这个框架下，我们建立了一个有说服力的几何收敛速率，用于衡量LLM生成的思维链条与真实语言来源的思维链条之间的相似性。我们的研究结果为LLM能够产生正确的思维序列（可能）解释了在需要推理能力的任务中性能提升的能力提供了理论上的解释。

    This paper delves into the capabilities of large language models (LLMs), specifically focusing on advancing the theoretical comprehension of chain-of-thought prompting. We investigate how LLMs can be effectively induced to generate a coherent chain of thoughts. To achieve this, we introduce a two-level hierarchical graphical model tailored for natural language generation. Within this framework, we establish a compelling geometrical convergence rate that gauges the likelihood of an LLM-generated chain of thoughts compared to those originating from the true language. Our findings provide a theoretical justification for the ability of LLMs to produce the correct sequence of thoughts (potentially) explaining performance gains in tasks demanding reasoning skills.
    
[^22]: 将关注点转移到相关性上: 探索大型语言模型的不确定性估计

    Shifting Attention to Relevance: Towards the Uncertainty Estimation of Large Language Models. (arXiv:2307.01379v1 [cs.CL])

    [http://arxiv.org/abs/2307.01379](http://arxiv.org/abs/2307.01379)

    本论文研究了大型语言模型（LLMs）自动生成的关键词不平等问题，发现在估计不确定性时，重要的令牌和含有有限语义的句子被同等或更加重视。为了解决这个问题，提出了共同转移关注点来更好地估计不确定性。

    

    虽然大型语言模型（LLMs）在自然语言生成方面表现出了巨大的潜力，但是对于模型生成的不确定性的特征化仍然具有挑战性，即用户何时可以信任模型的输出。我们的研究基于一些启发性的事实，即在自回归的LLMs中，令牌在反映生成的含义方面是不平等的，即一些令牌比其他令牌更相关（或更具代表性），然而在估计不确定性时所有的令牌被等值对待。这是由于语言冗余，其中大部分情况下，只需要几个关键词就足以传达一个长句的含义。我们将这些不平等称为生成的不平等，并研究它们如何影响不确定性的估计。我们的结果揭示，相当数量的令牌和包含有限语义的句子，在估计不确定性时被同等或甚至更加重视。为了解决由生成的不平等引起的这些偏差，我们提出了共同转移关注点来更好地估计不确定性。

    Although Large Language Models (LLMs) have shown great potential in Natural Language Generation, it is still challenging to characterize the uncertainty of model generations, i.e., when users could trust model outputs. Our research is derived from the heuristic facts that tokens are created unequally in reflecting the meaning of generations by auto-regressive LLMs, i.e., some tokens are more relevant (or representative) than others, yet all the tokens are equally valued when estimating uncertainty. It is because of the linguistic redundancy where mostly a few keywords are sufficient to convey the meaning of a long sentence. We name these inequalities as generative inequalities and investigate how they affect uncertainty estimation. Our results reveal that considerable tokens and sentences containing limited semantics are weighted equally or even heavily when estimating uncertainty. To tackle these biases posed by generative inequalities, we propose to jointly Shifting Attention to more
    
[^23]: 深思熟虑：具有内部工作记忆的决策Transformer

    Think Before You Act: Decision Transformers with Internal Working Memory. (arXiv:2305.16338v1 [cs.LG])

    [http://arxiv.org/abs/2305.16338](http://arxiv.org/abs/2305.16338)

    该论文提出了具有内部工作记忆模块的决策Transformer方法，以解决使用大型语言模型的决策代理在处理新任务上性能低下的问题。所提出的方法改善了训练效率和泛化能力，并进一步增强了转化决策制定代理对新任务的适应性。

    

    基于大型语言模型（LLM）的决策制定代理已经展示了跨越多个任务的泛化能力。然而，它们的性能依赖于大规模的数据和计算。我们认为，这种低效性源于遗忘现象，即模型通过参数记忆其行为，在训练过程中。因此，新任务的训练可能会降低模型在先前任务上的性能。与LLM的隐式记忆机制不同，人脑利用分布式存储器存储记忆，以有效地管理和组织多种技能，减轻了遗忘现象。因此，我们建议使用内部工作记忆模块来存储、融合和检索不同下游任务的信息。评估结果表明，所提出的方法改善了Atari游戏和元世界物体操作任务的训练效率和泛化能力。此外，我们证明了记忆微调进一步增强了转化决策制定代理对新任务的适应性。

    Large language model (LLM)-based decision-making agents have shown the ability to generalize across multiple tasks. However, their performance relies on massive data and compute. We argue that this inefficiency stems from the forgetting phenomenon, in which a model memorizes its behaviors in parameters throughout training. As a result, training on a new task may deteriorate the model's performance on previous tasks. In contrast to LLMs' implicit memory mechanism, the human brain utilizes distributed memory storage, which helps manage and organize multiple skills efficiently, mitigating the forgetting phenomenon. Thus inspired, we propose an internal working memory module to store, blend, and retrieve information for different downstream tasks. Evaluation results show that the proposed method improves training efficiency and generalization in both Atari games and meta-world object manipulation tasks. Moreover, we demonstrate that memory fine-tuning further enhances the adaptability of t
    
[^24]: UP5: 面向公平性推荐的无偏基础模型

    UP5: Unbiased Foundation Model for Fairness-aware Recommendation. (arXiv:2305.12090v1 [cs.IR])

    [http://arxiv.org/abs/2305.12090](http://arxiv.org/abs/2305.12090)

    本研究提出了一种新颖的基础模型UP5，它采用反事实公平促进技术来消除大型语言模型中的偏见，从而实现面向公平性的推荐。

    

    基于大型语言模型（LLM）等基础模型的最新进展，已将它们推到了推荐系统（RS）的前沿。此外，RS中的公平性很关键，因为许多用户将其用于决策和需求履行。然而，目前尚缺乏对推荐基础模型展示公平性水平和公平处理不同用户群组的适当方法的理解。本文侧重于用户方面的不公平问题，并通过彻底检查表明，LLMs中存在不公平性，导致不公平的推荐结果。为了消除LLM中的偏差以实现面向公平性的推荐，我们引入了一种基于反事实公平促进技术的新型无偏P5（UP5）基础模型。CFP包括两个子模块：个性化前缀提示和Prompt混合，从而增强了个体敏感属性的公平性。

    Recent advancements in foundation models such as large language models (LLM) have propelled them to the forefront of recommender systems (RS). Moreover, fairness in RS is critical since many users apply it for decision-making and demand fulfillment. However, at present, there is a lack of understanding regarding the level of fairness exhibited by recommendation foundation models and the appropriate methods for equitably treating different groups of users in foundation models. In this paper, we focus on user-side unfairness problem and show through a thorough examination that there is unfairness involved in LLMs that lead to unfair recommendation results. To eliminate bias from LLM for fairness-aware recommendation, we introduce a novel Unbiased P5 (UP5) foundation model based on Counterfactually-Fair-Prompting (CFP) techniques. CFP includes two sub-modules: a personalized prefix prompt that enhances fairness with respect to individual sensitive attributes, and a Prompt Mixture that int
    

