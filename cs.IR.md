# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TREC iKAT 2023: The Interactive Knowledge Assistance Track Overview.](http://arxiv.org/abs/2401.01330) | TREC iKAT 2023是一个交互式的知识辅助任务，旨在开发适应用户交互和上下文的会话搜索代理。该任务还强调决策搜索任务，用户通过筛选数据和信息来进行决策和执行动作。 |
| [^2] | [LLaRA: Aligning Large Language Models with Sequential Recommenders.](http://arxiv.org/abs/2312.02445) | LLaRA是一个将传统推荐器和大型语言模型相结合的框架，通过使用一种新颖的混合方法来代表项目，在顺序推荐中充分利用了传统推荐器的用户行为知识和LLMs的世界知识。 |
| [^3] | [Caseformer: Pre-training for Legal Case Retrieval.](http://arxiv.org/abs/2311.00333) | 本文提出了一种新颖的预训练方法，名为Caseformer，在法律案例检索中解决了标注数据不足的问题，能够更好地理解和捕捉法律语料库中的关键知识和数据结构。 |
| [^4] | [A statistical significance testing approach for measuring term burstiness with applications to domain-specific terminology extraction.](http://arxiv.org/abs/2310.15790) | 我们提出了一种统计显著性测试方法，用于测量专业术语抽取中的术语爆发性。我们的方法基于多项式语言模型，通过启发式公式得到近似测试P值。此外，我们还推导了逆文档频率与逆收集频率之间的关系。 |
| [^5] | [Large Search Model: Redefining Search Stack in the Era of LLMs.](http://arxiv.org/abs/2310.14587) | 本文介绍了一个称为大型搜索模型的框架，通过将所有搜索任务统一为一个大型语言模型(LLM)，重新定义了传统的搜索堆栈。这个框架利用了LLM的强大语言理解和推理能力，有潜力提高搜索结果的质量，同时简化现有的繁琐的搜索堆栈。 |
| [^6] | [Ranking In Generalized Linear Bandits.](http://arxiv.org/abs/2207.00109) | 本文研究了广义线性Bandits中的排名问题，设计了UCB和Thompson Sampling类型算法来解决该问题，并对位置和物品之间的依赖关系进行了建模。研究结果在位置依赖性和排名问题与图论的连接等方面进行了推广。 |

# 详细

[^1]: TREC iKAT 2023: 交互式知识辅助任务概述

    TREC iKAT 2023: The Interactive Knowledge Assistance Track Overview. (arXiv:2401.01330v1 [cs.IR])

    [http://arxiv.org/abs/2401.01330](http://arxiv.org/abs/2401.01330)

    TREC iKAT 2023是一个交互式的知识辅助任务，旨在开发适应用户交互和上下文的会话搜索代理。该任务还强调决策搜索任务，用户通过筛选数据和信息来进行决策和执行动作。

    

    会话式信息查询是一个关键的研究领域，之前的工作也有很大的贡献。TREC交互式知识辅助任务（iKAT）建立在TREC会话辅助任务（CAsT）的基础上。然而，iKAT着重于创建和研究可以根据用户之前的交互和当前情境自适应响应的会话搜索代理。挑战在于使会话搜索代理能够将个性化的上下文信息融入到相应中，以高效地引导用户获取相关信息。iKAT还着重于决策搜索任务，即用户通过数据和信息筛选来衡量各种选择，以达到结论或执行动作。这些任务在日常信息搜索决策中普遍存在，无论是旅游、健康还是购物等，通常涉及一组高级信息操作符，其中查询或问题可能会

    Conversational Information Seeking stands as a pivotal research area with significant contributions from previous works. The TREC Interactive Knowledge Assistance Track (iKAT) builds on the foundational work of the TREC Conversational Assistance Track (CAsT). However, iKAT distinctively emphasizes the creation and research of conversational search agents that adapt responses based on user's prior interactions and present context. The challenge lies in enabling Conversational Search Agents (CSA) to incorporate this personalized context to efficiency and effectively guide users through the relevant information to them. iKAT also emphasizes decisional search tasks, where users sift through data and information to weigh up options in order to reach a conclusion or perform an action. These tasks, prevalent in everyday information-seeking decisions -- be it related to travel, health, or shopping -- often revolve around a subset of high-level information operators where queries or questions a
    
[^2]: LLaRA: 使用顺序推荐器对齐大型语言模型

    LLaRA: Aligning Large Language Models with Sequential Recommenders. (arXiv:2312.02445v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2312.02445](http://arxiv.org/abs/2312.02445)

    LLaRA是一个将传统推荐器和大型语言模型相结合的框架，通过使用一种新颖的混合方法来代表项目，在顺序推荐中充分利用了传统推荐器的用户行为知识和LLMs的世界知识。

    

    顺序推荐旨在根据用户的历史交互预测与用户偏好相匹配的后续项目。随着大型语言模型 (LLMs) 的发展，人们对于将LLMs 应用于顺序推荐并将其视为语言建模任务的潜力越来越感兴趣。之前的工作中，使用ID索引或文本索引来表示文本提示中的项目，并将提示输入LLMs，但无法全面融合世界知识或展示足够的顺序理解能力。为了充分发挥传统推荐器（可以编码用户行为知识）和LLMs（具有项目的世界知识）的互补优势，我们提出了LLaRA - 一种大型语言和推荐助手框架。具体而言，LLaRA使用一种新颖的混合方法，将传统推荐器的基于ID的项目嵌入与文本项目特征整合到LLM的输入提示中。

    Sequential recommendation aims to predict the subsequent items matching user preference based on her/his historical interactions. With the development of Large Language Models (LLMs), there is growing interest in exploring the potential of LLMs for sequential recommendation by framing it as a language modeling task. Prior works represent items in the textual prompts using either ID indexing or text indexing and feed the prompts into LLMs, but falling short of either encapsulating comprehensive world knowledge or exhibiting sufficient sequential understanding. To harness the complementary strengths of traditional recommenders (which encode user behavioral knowledge) and LLMs (which possess world knowledge about items), we propose LLaRA -- a Large Language and Recommendation Assistant framework. Specifically, LLaRA represents items in LLM's input prompts using a novel hybrid approach that integrates ID-based item embeddings from traditional recommenders with textual item features. Viewin
    
[^3]: Caseformer: 法律案例检索的预训练

    Caseformer: Pre-training for Legal Case Retrieval. (arXiv:2311.00333v1 [cs.IR])

    [http://arxiv.org/abs/2311.00333](http://arxiv.org/abs/2311.00333)

    本文提出了一种新颖的预训练方法，名为Caseformer，在法律案例检索中解决了标注数据不足的问题，能够更好地理解和捕捉法律语料库中的关键知识和数据结构。

    

    法律案例检索旨在帮助法律工作者找到与他们手头案件相关的案例，这对于保证公平和正义的法律判决非常重要。尽管最近神经检索方法在开放域检索任务（例如网络搜索）方面取得了显著的改进，但是由于对标注数据的渴望，这些方法在法律案例检索中并没有显示出优势。由于需要领域专业知识，对法律领域进行大规模训练数据的标注是困难的，因此传统的基于词汇匹配的搜索技术，如TF-IDF、BM25和查询似然，仍然在法律案例检索系统中盛行。虽然以前的研究已经设计了一些针对开放域任务中IR模型的预训练方法，但是由于无法理解和捕捉法律语料库中的关键知识和数据结构，这些方法在法律案例检索中通常是次优的。为此，我们提出了一种新颖的预训练方法。

    Legal case retrieval aims to help legal workers find relevant cases related to their cases at hand, which is important for the guarantee of fairness and justice in legal judgments. While recent advances in neural retrieval methods have significantly improved the performance of open-domain retrieval tasks (e.g., Web search), their advantages have not been observed in legal case retrieval due to their thirst for annotated data. As annotating large-scale training data in legal domains is prohibitive due to the need for domain expertise, traditional search techniques based on lexical matching such as TF-IDF, BM25, and Query Likelihood are still prevalent in legal case retrieval systems. While previous studies have designed several pre-training methods for IR models in open-domain tasks, these methods are usually suboptimal in legal case retrieval because they cannot understand and capture the key knowledge and data structures in the legal corpus. To this end, we propose a novel pre-trainin
    
[^4]: 一种用于测量专业术语抽取中术语爆发性的统计显著性测试方法

    A statistical significance testing approach for measuring term burstiness with applications to domain-specific terminology extraction. (arXiv:2310.15790v1 [cs.IR])

    [http://arxiv.org/abs/2310.15790](http://arxiv.org/abs/2310.15790)

    我们提出了一种统计显著性测试方法，用于测量专业术语抽取中的术语爆发性。我们的方法基于多项式语言模型，通过启发式公式得到近似测试P值。此外，我们还推导了逆文档频率与逆收集频率之间的关系。

    

    专业术语抽取是文本分析中的重要任务。当语料库中一个术语的出现集中在少数几个文件中时，可称之为“爆发性”。作为内容丰富的术语，爆发性术语非常适合用于主题描述，并且是技术术语的自然候选词。文献中提出了多种术语爆发性的测量方法。然而，在文本分析中，包括与术语爆发性相关的统计显著性测试范式尚未得到充分探索。为了探索这个领域，我们的主要贡献是提出了一种基于多项式语言模型的术语爆发性统计显著性的精确测试方法。由于计算成本过高，我们还提出了一个启发式公式，用于近似测试P值。作为补充的理论贡献，我们推导了一种未经报道的逆文档频率与逆收集频率的关系。

    Domain-specific terminology extraction is an important task in text analysis. A term in a corpus is said to be "bursty" when its occurrences are concentrated in few out of many documents. Being content rich, bursty terms are highly suited for subject matter characterization, and serve as natural candidates for identifying with technical terminology. Multiple measures of term burstiness have been proposed in the literature. However, the statistical significance testing paradigm has remained underexplored in text analysis, including in relation to term burstiness. To test these waters, we propose as our main contribution a multinomial language model-based exact test of statistical significance for term burstiness. Due to its prohibitive computational cost, we advance a heuristic formula designed to serve as a proxy for test P-values. As a complementary theoretical contribution, we derive a previously unreported relationship connecting the inverse document frequency and inverse collection
    
[^5]: 大型搜索模型：重新定义LLM时代的搜索堆栈

    Large Search Model: Redefining Search Stack in the Era of LLMs. (arXiv:2310.14587v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2310.14587](http://arxiv.org/abs/2310.14587)

    本文介绍了一个称为大型搜索模型的框架，通过将所有搜索任务统一为一个大型语言模型(LLM)，重新定义了传统的搜索堆栈。这个框架利用了LLM的强大语言理解和推理能力，有潜力提高搜索结果的质量，同时简化现有的繁琐的搜索堆栈。

    

    现代搜索引擎是由不同组件构建的堆栈，包括查询理解、检索、多阶段排名和问答等。这些组件通常是独立优化和部署的。本文介绍了一个新的概念性框架，称为大型搜索模型，通过将所有任务统一为一个大型语言模型(LLM)来重新定义传统的搜索堆栈。所有任务都被表述为自回归文本生成问题，通过使用自然语言提示可以定制任务。这个提出的框架利用了LLM的强大语言理解和推理能力，有潜力提高搜索结果的质量，同时简化现有的繁琐的搜索堆栈。为了验证这个框架的可行性，我们展示了一系列概念验证实验，并讨论了实现这种方法所面临的潜在挑战。

    Modern search engines are built on a stack of different components, including query understanding, retrieval, multi-stage ranking, and question answering, among others. These components are often optimized and deployed independently. In this paper, we introduce a novel conceptual framework called large search model, which redefines the conventional search stack by unifying search tasks with one large language model (LLM). All tasks are formulated as autoregressive text generation problems, allowing for the customization of tasks through the use of natural language prompts. This proposed framework capitalizes on the strong language understanding and reasoning capabilities of LLMs, offering the potential to enhance search result quality while simultaneously simplifying the existing cumbersome search stack. To substantiate the feasibility of this framework, we present a series of proof-of-concept experiments and discuss the potential challenges associated with implementing this approach w
    
[^6]: 广义线性Bandits中的排名问题研究

    Ranking In Generalized Linear Bandits. (arXiv:2207.00109v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2207.00109](http://arxiv.org/abs/2207.00109)

    本文研究了广义线性Bandits中的排名问题，设计了UCB和Thompson Sampling类型算法来解决该问题，并对位置和物品之间的依赖关系进行了建模。研究结果在位置依赖性和排名问题与图论的连接等方面进行了推广。

    

    我们研究了广义线性Bandits中的排名问题。在每个时刻，学习代理选择一个有序的物品列表，并观察随机结果。在推荐系统中，显示一个有序的最具吸引力的物品列表并不总是最优的，因为位置和物品之间存在复杂的奖励函数。一个非常简单的例子是当所有最具吸引力的物品都来自同一类别时缺乏多样性。我们对有序列表中的位置和物品之间的依赖关系进行建模，并设计了用于解决这个问题的UCB和Thompson Sampling类型的算法。我们的工作在几个方向上推广了现有的研究，包括位置依赖性，其中位置折扣是一个特例，并将排名问题与图论相联系。

    We study the ranking problem in generalized linear bandits. At each time, the learning agent selects an ordered list of items and observes stochastic outcomes. In recommendation systems, displaying an ordered list of the most attractive items is not always optimal as both position and item dependencies result in a complex reward function. A very naive example is the lack of diversity when all the most attractive items are from the same category. We model the position and item dependencies in the ordered list and design UCB and Thompson Sampling type algorithms for this problem. Our work generalizes existing studies in several directions, including position dependencies where position discount is a particular case, and connecting the ranking problem to graph theory.
    

