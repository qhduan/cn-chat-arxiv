# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Detecting Generated Native Ads in Conversational Search](https://arxiv.org/abs/2402.04889) | 本论文研究了LLM是否可以用作对抗生成式原生广告的对策，并通过构建广告倾向查询数据集和带自动整合广告的生成答案数据集进行实验证明。 |
| [^2] | [Faithful Path Language Modelling for Explainable Recommendation over Knowledge Graph.](http://arxiv.org/abs/2310.16452) | 本文提出了一个名为PEARLM的方法，通过语言建模开展基于路径的知识图谱推荐，解决了现有方法中对预训练知识图谱嵌入的依赖以及未充分利用实体和关系之间相互依赖性的问题，还避免了生成不准确的解释。实验结果表明，与现有方法相比，我们的方法效果显著。 |
| [^3] | [Group Membership Bias.](http://arxiv.org/abs/2308.02887) | 研究分析了群组偏见对排名质量的影响，指出在不纠正群组偏见的情况下，所谓的公平排名不真正公平。 |

# 详细

[^1]: 发现对话式搜索中的生成式原生广告

    Detecting Generated Native Ads in Conversational Search

    [https://arxiv.org/abs/2402.04889](https://arxiv.org/abs/2402.04889)

    本论文研究了LLM是否可以用作对抗生成式原生广告的对策，并通过构建广告倾向查询数据集和带自动整合广告的生成答案数据集进行实验证明。

    

    对话式搜索引擎如YouChat和Microsoft Copilot使用大型语言模型（LLM）为查询生成答案。将此技术用于生成并整合广告，而不是将广告与有机搜索结果分开放置，只是一小步。这种类型的广告类似于原生广告和产品放置，两者都是非常有效的微妙和操纵性广告形式。在考虑到与LLM相关的高计算成本时，信息搜索者将很可能在不久的将来面临这种LLM技术的使用，因此供应商需要开发可持续的商业模式。本文研究了LLM是否也可以用作对抗生成式原生广告的对策，即阻止它们。为此，我们编制了一个大型的广告倾向查询数据集和带自动整合广告的生成答案数据集进行实验。

    Conversational search engines such as YouChat and Microsoft Copilot use large language models (LLMs) to generate answers to queries. It is only a small step to also use this technology to generate and integrate advertising within these answers - instead of placing ads separately from the organic search results. This type of advertising is reminiscent of native advertising and product placement, both of which are very effective forms of subtle and manipulative advertising. It is likely that information seekers will be confronted with such use of LLM technology in the near future, especially when considering the high computational costs associated with LLMs, for which providers need to develop sustainable business models. This paper investigates whether LLMs can also be used as a countermeasure against generated native ads, i.e., to block them. For this purpose we compile a large dataset of ad-prone queries and of generated answers with automatically integrated ads to experiment with fin
    
[^2]: 可解释的基于路径的知识图推荐中的忠实路径语言建模

    Faithful Path Language Modelling for Explainable Recommendation over Knowledge Graph. (arXiv:2310.16452v1 [cs.IR])

    [http://arxiv.org/abs/2310.16452](http://arxiv.org/abs/2310.16452)

    本文提出了一个名为PEARLM的方法，通过语言建模开展基于路径的知识图谱推荐，解决了现有方法中对预训练知识图谱嵌入的依赖以及未充分利用实体和关系之间相互依赖性的问题，还避免了生成不准确的解释。实验结果表明，与现有方法相比，我们的方法效果显著。

    

    针对知识图谱中的路径推理方法在提高推荐系统透明度方面的潜力，本文提出了一种名为PEARLM的新方法，该方法通过语言建模有效捕获用户行为和产品端知识。我们的方法通过语言模型直接从知识图谱上的路径中学习知识图谱嵌入，并将实体和关系统一在同一优化空间中。序列解码的约束保证了路径对知识图谱的忠实性。在两个数据集上的实验证明了我们方法与现有最先进方法的有效性。

    Path reasoning methods over knowledge graphs have gained popularity for their potential to improve transparency in recommender systems. However, the resulting models still rely on pre-trained knowledge graph embeddings, fail to fully exploit the interdependence between entities and relations in the KG for recommendation, and may generate inaccurate explanations. In this paper, we introduce PEARLM, a novel approach that efficiently captures user behaviour and product-side knowledge through language modelling. With our approach, knowledge graph embeddings are directly learned from paths over the KG by the language model, which also unifies entities and relations in the same optimisation space. Constraints on the sequence decoding additionally guarantee path faithfulness with respect to the KG. Experiments on two datasets show the effectiveness of our approach compared to state-of-the-art baselines. Source code and datasets: AVAILABLE AFTER GETTING ACCEPTED.
    
[^3]: 群组成员偏见

    Group Membership Bias. (arXiv:2308.02887v1 [cs.IR])

    [http://arxiv.org/abs/2308.02887](http://arxiv.org/abs/2308.02887)

    研究分析了群组偏见对排名质量的影响，指出在不纠正群组偏见的情况下，所谓的公平排名不真正公平。

    

    当从用户交互中学习排名时，搜索和推荐系统必须解决用户行为中的偏见问题，以提供高质量的排名。在排名文献中最近研究的一种偏见类型是敏感属性（如性别）对用户对项目效用的判断产生的影响。例如，在寻找某个专业领域时，一些用户可能对男性候选人比女性候选人更有偏见。我们将这种偏见称为群组成员偏见或群组偏见。越来越多的人希望获得不仅具有高效用性而且对个人和敏感群体也公平的排名。基于价值的公平度量依赖于项目的估计价值或效用。在群组偏见的情况下，敏感群体的效用被低估，因此，在不纠正这种偏见的情况下，所谓的公平排名并不真正公平。首先，本文分析了群组偏见对排名质量以及两个众所周知的情况的影响

    When learning to rank from user interactions, search and recommendation systems must address biases in user behavior to provide a high-quality ranking. One type of bias that has recently been studied in the ranking literature is when sensitive attributes, such as gender, have an impact on a user's judgment about an item's utility. For example, in a search for an expertise area, some users may be biased towards clicking on male candidates over female candidates. We call this type of bias group membership bias or group bias for short. Increasingly, we seek rankings that not only have high utility but are also fair to individuals and sensitive groups. Merit-based fairness measures rely on the estimated merit or utility of the items. With group bias, the utility of the sensitive groups is under-estimated, hence, without correcting for this bias, a supposedly fair ranking is not truly fair. In this paper, first, we analyze the impact of group bias on ranking quality as well as two well-know
    

