# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards LLM-RecSys Alignment with Textual ID Learning](https://arxiv.org/abs/2403.19021) | 通过提出IDGen，将每个推荐项目表示为独特、简洁、语义丰富的文本ID，从而使得基于大型语言模型的推荐更好地与自然语言生成对齐。 |
| [^2] | [A Survey on Cross-Domain Sequential Recommendation.](http://arxiv.org/abs/2401.04971) | 跨领域序列推荐通过集成和学习多个领域的交互信息，将用户偏好建模从平面转向立体。文章对CDSR问题进行了定义和分析，提供了从宏观和微观两个视角的系统概述。对于不同领域间的模型，总结了多层融合结构和融合桥梁。对于现有模型，讨论了基础技术和辅助学习技术。展示了公开数据集和实验结果，并给出了未来发展的见解。 |
| [^3] | [Large language models can accurately predict searcher preferences.](http://arxiv.org/abs/2309.10621) | 大型语言模型可以通过从真实用户那里获取高质量的第一方数据来准确预测搜索者的偏好。 |

# 详细

[^1]: 朝向LLM-RecSys对齐与文本ID学习的方向

    Towards LLM-RecSys Alignment with Textual ID Learning

    [https://arxiv.org/abs/2403.19021](https://arxiv.org/abs/2403.19021)

    通过提出IDGen，将每个推荐项目表示为独特、简洁、语义丰富的文本ID，从而使得基于大型语言模型的推荐更好地与自然语言生成对齐。

    

    基于大型语言模型(LLMs)的生成式推荐已经将传统的基于排名的推荐方式转变为文本生成范例。然而，与固有操作人类词汇的标准NLP任务相反，目前生成式推荐领域的研究在如何在文本生成范式中以简洁而有意义的ID表示有效编码推荐项目方面存在困难。为了更好地对齐LLMs与推荐需求，我们提出了IDGen，使用人类语言标记将每个项目表示为独特、简洁、语义丰富、与平台无关的文本ID。这通过在基于LLM的推荐系统旁训练文本ID生成器来实现，使个性化推荐能够无缝集成到自然语言生成中。值得注意的是，由于用户历史记录以自然语言表达并与原始数据集解耦，我们的方法提出了潜在的

    arXiv:2403.19021v1 Announce Type: cross  Abstract: Generative recommendation based on Large Language Models (LLMs) have transformed the traditional ranking-based recommendation style into a text-to-text generation paradigm. However, in contrast to standard NLP tasks that inherently operate on human vocabulary, current research in generative recommendations struggles to effectively encode recommendation items within the text-to-text framework using concise yet meaningful ID representations. To better align LLMs with recommendation needs, we propose IDGen, representing each item as a unique, concise, semantically rich, platform-agnostic textual ID using human language tokens. This is achieved by training a textual ID generator alongside the LLM-based recommender, enabling seamless integration of personalized recommendations into natural language generation. Notably, as user history is expressed in natural language and decoupled from the original dataset, our approach suggests the potenti
    
[^2]: 跨领域序列推荐的综述

    A Survey on Cross-Domain Sequential Recommendation. (arXiv:2401.04971v1 [cs.IR])

    [http://arxiv.org/abs/2401.04971](http://arxiv.org/abs/2401.04971)

    跨领域序列推荐通过集成和学习多个领域的交互信息，将用户偏好建模从平面转向立体。文章对CDSR问题进行了定义和分析，提供了从宏观和微观两个视角的系统概述。对于不同领域间的模型，总结了多层融合结构和融合桥梁。对于现有模型，讨论了基础技术和辅助学习技术。展示了公开数据集和实验结果，并给出了未来发展的见解。

    

    跨领域序列推荐（CDSR）通过在不同粒度（从序列间到序列内，从单领域到跨领域）上集成和学习来自多个领域的交互信息，将用户偏好建模从平面转向了立体。本综述文章中，我们首先使用四维张量定义了CDSR问题，并分析了其在多维度降维下的多类型输入表示。接下来，我们从整体和细节两个视角提供了系统的概述。从整体视角，我们总结了各个模型在不同领域间的多层融合结构，并讨论了它们的融合桥梁。从细节视角，我们着重讨论了现有模型的基础技术，并解释了辅助学习技术。最后，我们展示了可用的公开数据集和代表性的实验结果，并提供了对未来发展的一些见解。

    Cross-domain sequential recommendation (CDSR) shifts the modeling of user preferences from flat to stereoscopic by integrating and learning interaction information from multiple domains at different granularities (ranging from inter-sequence to intra-sequence and from single-domain to cross-domain).In this survey, we initially define the CDSR problem using a four-dimensional tensor and then analyze its multi-type input representations under multidirectional dimensionality reductions. Following that, we provide a systematic overview from both macro and micro views. From a macro view, we abstract the multi-level fusion structures of various models across domains and discuss their bridges for fusion. From a micro view, focusing on the existing models, we specifically discuss the basic technologies and then explain the auxiliary learning technologies. Finally, we exhibit the available public datasets and the representative experimental results as well as provide some insights into future d
    
[^3]: 大型语言模型能够准确预测搜索者的偏好

    Large language models can accurately predict searcher preferences. (arXiv:2309.10621v1 [cs.IR])

    [http://arxiv.org/abs/2309.10621](http://arxiv.org/abs/2309.10621)

    大型语言模型可以通过从真实用户那里获取高质量的第一方数据来准确预测搜索者的偏好。

    

    相关性标签是评估和优化搜索系统的关键。获取大量相关性标签通常需要第三方标注人员，但存在低质量数据的风险。本论文介绍了一种改进标签质量的替代方法，通过从真实用户那里获得仔细反馈来获取高质量的第一方数据。

    Relevance labels, which indicate whether a search result is valuable to a searcher, are key to evaluating and optimising search systems. The best way to capture the true preferences of users is to ask them for their careful feedback on which results would be useful, but this approach does not scale to produce a large number of labels. Getting relevance labels at scale is usually done with third-party labellers, who judge on behalf of the user, but there is a risk of low-quality data if the labeller doesn't understand user needs. To improve quality, one standard approach is to study real users through interviews, user studies and direct feedback, find areas where labels are systematically disagreeing with users, then educate labellers about user needs through judging guidelines, training and monitoring. This paper introduces an alternate approach for improving label quality. It takes careful feedback from real users, which by definition is the highest-quality first-party gold data that 
    

