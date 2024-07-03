# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Dynamic Q&A of Clinical Documents with Large Language Models.](http://arxiv.org/abs/2401.10733) | 本研究介绍了一种使用大型语言模型进行临床文档动态问答的自然语言接口。通过Langchain和Transformer-based LLMs驱动的聊天机器人，用户可以用自然语言查询临床笔记并获得相关答案。实验结果显示Wizard Vicuna具有出色的准确性，但计算要求较高。模型优化方案提高了约48倍的延迟。然而，模型产生幻象和多样化医疗案例评估的限制仍然存在。解决这些挑战对于发掘临床笔记的价值和推进基于AI的临床决策至关重要。 |
| [^2] | [A Survey on Popularity Bias in Recommender Systems.](http://arxiv.org/abs/2308.01118) | 这篇综述论文讨论了推荐系统中的流行偏差问题，并回顾了现有的方法来检测、量化和减少流行偏差。它同时提供了计算度量的概述和主要技术方法的回顾。 |

# 详细

[^1]: 使用大型语言模型进行临床文档的动态问答

    Dynamic Q&A of Clinical Documents with Large Language Models. (arXiv:2401.10733v1 [cs.IR])

    [http://arxiv.org/abs/2401.10733](http://arxiv.org/abs/2401.10733)

    本研究介绍了一种使用大型语言模型进行临床文档动态问答的自然语言接口。通过Langchain和Transformer-based LLMs驱动的聊天机器人，用户可以用自然语言查询临床笔记并获得相关答案。实验结果显示Wizard Vicuna具有出色的准确性，但计算要求较高。模型优化方案提高了约48倍的延迟。然而，模型产生幻象和多样化医疗案例评估的限制仍然存在。解决这些挑战对于发掘临床笔记的价值和推进基于AI的临床决策至关重要。

    

    电子健康记录（EHR）中收录了临床笔记中的重要患者数据。随着这些笔记数量和复杂度的增加，手动提取变得具有挑战性。本研究利用大型语言模型（LLMs）引入了一种自然语言接口，用于对临床笔记进行动态问答。我们的聊天机器人由Langchain和基于Transformer的LLMs驱动，允许用户用自然语言发出查询，并从临床笔记中获得相关答案。通过使用各种嵌入模型和先进的LLMs进行实验，结果表明Wizard Vicuna在准确性方面表现优异，尽管计算要求较高。模型优化，包括权重量化，将延迟提高了约48倍。有希望的结果显示了临床笔记中的价值潜力，但仍存在模型产生幻象和有限的多样化医疗案例评估等挑战。解决这些问题对于发掘临床笔记的价值和推动AI驱动的临床决策至关重要。

    Electronic health records (EHRs) house crucial patient data in clinical notes. As these notes grow in volume and complexity, manual extraction becomes challenging. This work introduces a natural language interface using large language models (LLMs) for dynamic question-answering on clinical notes. Our chatbot, powered by Langchain and transformer-based LLMs, allows users to query in natural language, receiving relevant answers from clinical notes. Experiments, utilizing various embedding models and advanced LLMs, show Wizard Vicuna's superior accuracy, albeit with high compute demands. Model optimization, including weight quantization, improves latency by approximately 48 times. Promising results indicate potential, yet challenges such as model hallucinations and limited diverse medical case evaluations remain. Addressing these gaps is crucial for unlocking the value in clinical notes and advancing AI-driven clinical decision-making.
    
[^2]: 推荐系统中的流行偏差综述

    A Survey on Popularity Bias in Recommender Systems. (arXiv:2308.01118v1 [cs.IR])

    [http://arxiv.org/abs/2308.01118](http://arxiv.org/abs/2308.01118)

    这篇综述论文讨论了推荐系统中的流行偏差问题，并回顾了现有的方法来检测、量化和减少流行偏差。它同时提供了计算度量的概述和主要技术方法的回顾。

    

    推荐系统以个性化的方式帮助人们找到相关内容。这些系统的一个主要承诺是能够增加目录中较少知名的物品的可见性。然而，现有研究表明，在许多情况下，现今的推荐算法反而表现出流行偏差，即它们在推荐中经常关注相当流行的物品。这种偏差不仅可能导致短期内对消费者和提供者的推荐价值有限，而且还可能引起不希望的强化效应。在本文中，我们讨论了流行偏差的潜在原因，并回顾了现有的检测、量化和减少推荐系统中流行偏差的方法。因此，我们的综述既包括了文献中使用的计算度量的概述，也包括了减少偏差的主要技术方法的回顾。我们还对这些方法进行了批判性讨论。

    Recommender systems help people find relevant content in a personalized way. One main promise of such systems is that they are able to increase the visibility of items in the long tail, i.e., the lesser-known items in a catalogue. Existing research, however, suggests that in many situations today's recommendation algorithms instead exhibit a popularity bias, meaning that they often focus on rather popular items in their recommendations. Such a bias may not only lead to limited value of the recommendations for consumers and providers in the short run, but it may also cause undesired reinforcement effects over time. In this paper, we discuss the potential reasons for popularity bias and we review existing approaches to detect, quantify and mitigate popularity bias in recommender systems. Our survey therefore includes both an overview of the computational metrics used in the literature as well as a review of the main technical approaches to reduce the bias. We furthermore critically discu
    

