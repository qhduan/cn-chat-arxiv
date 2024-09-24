# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DRAGIN: Dynamic Retrieval Augmented Generation based on the Real-time Information Needs of Large Language Models](https://arxiv.org/abs/2403.10081) | 提出了一种新框架DRAGIN，旨在解决大型语言模型在文本生成过程中动态检索和生成中存在的问题。 |
| [^2] | [Large language models in biomedical natural language processing: benchmarks, baselines, and recommendations.](http://arxiv.org/abs/2305.16326) | 本文研究了GPT-3和GPT-4在生物医学自然语言处理中的表现，分析了它们可能产生的错误类型，并提供了使用这些模型的建议。 |

# 详细

[^1]: DRAGIN：基于大型语言模型实时信息需求的动态检索增强生成

    DRAGIN: Dynamic Retrieval Augmented Generation based on the Real-time Information Needs of Large Language Models

    [https://arxiv.org/abs/2403.10081](https://arxiv.org/abs/2403.10081)

    提出了一种新框架DRAGIN，旨在解决大型语言模型在文本生成过程中动态检索和生成中存在的问题。

    

    动态检索增强生成（RAG）范式在大型语言模型（LLMs）的文本生成过程中主动决定何时以及何时检索。该范式的两个关键元素是确定激活检索模块的最佳时机（决定何时检索）以及一旦触发检索，制定适当的查询（确定要检索什么）。然而，当前动态RAG方法在两个方面都存在不足。首先，决定何时进行检索的策略通常依赖于静态规则。此外，决定要检索什么的策略通常局限于LLM的最近一句或最后几个标记，而LLM的实时信息需求可能跨越整个上下文。为克服这些局限性，我们引入了一个新框架DRAGIN， 即基于LLMs实时信息需求的动态检索增强生成。

    arXiv:2403.10081v1 Announce Type: new  Abstract: Dynamic retrieval augmented generation (RAG) paradigm actively decides when and what to retrieve during the text generation process of Large Language Models (LLMs). There are two key elements of this paradigm: identifying the optimal moment to activate the retrieval module (deciding when to retrieve) and crafting the appropriate query once retrieval is triggered (determining what to retrieve). However, current dynamic RAG methods fall short in both aspects. Firstly, the strategies for deciding when to retrieve often rely on static rules. Moreover, the strategies for deciding what to retrieve typically limit themselves to the LLM's most recent sentence or the last few tokens, while the LLM's real-time information needs may span across the entire context. To overcome these limitations, we introduce a new framework, DRAGIN, i.e., Dynamic Retrieval Augmented Generation based on the real-time Information Needs of LLMs. Our framework is specif
    
[^2]: 生物医学自然语言处理中的大型语言模型: 基准、基线和建议

    Large language models in biomedical natural language processing: benchmarks, baselines, and recommendations. (arXiv:2305.16326v1 [cs.CL])

    [http://arxiv.org/abs/2305.16326](http://arxiv.org/abs/2305.16326)

    本文研究了GPT-3和GPT-4在生物医学自然语言处理中的表现，分析了它们可能产生的错误类型，并提供了使用这些模型的建议。

    

    生物医学文献呈指数级增长，手动筛选和提取知识变得困难。自动从生物医学文献中提取信息的生物医学自然语言处理（BioNLP）技术有助于减轻这种负担。近年来，如GPT-3和GPT-4等大型语言模型（LLMs）因其卓越的性能而受到重视。但是，它们在BioNLP任务中的有效性以及对方法开发和下游用户的影响仍未得到研究。本研究（1）在四个应用程序中在八个BioNLP数据集中建立了GPT-3和GPT-4在零-shot和一-shot设置下的基准表现，包括命名实体识别，关系提取，多标签文档分类和语义相似性和推理；（2）审查了LLMs产生的错误，并将错误分为三种类型：缺失，不一致和不需要的人工内容；（3）提出了使用LLMs的建议。

    Biomedical literature is growing rapidly, making it challenging to curate and extract knowledge manually. Biomedical natural language processing (BioNLP) techniques that can automatically extract information from biomedical literature help alleviate this burden. Recently, large Language Models (LLMs), such as GPT-3 and GPT-4, have gained significant attention for their impressive performance. However, their effectiveness in BioNLP tasks and impact on method development and downstream users remain understudied. This pilot study (1) establishes the baseline performance of GPT-3 and GPT-4 at both zero-shot and one-shot settings in eight BioNLP datasets across four applications: named entity recognition, relation extraction, multi-label document classification, and semantic similarity and reasoning, (2) examines the errors produced by the LLMs and categorized the errors into three types: missingness, inconsistencies, and unwanted artificial content, and (3) provides suggestions for using L
    

