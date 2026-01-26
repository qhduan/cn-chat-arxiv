# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Fine-Tuning, Prompting, In-Context Learning and Instruction-Tuning: How Many Labelled Samples Do We Need?](https://arxiv.org/abs/2402.12819) | 专门模型通常只需少量标记样本（100-1000个）就能与通用模型持平甚至更好，取决于任务的复杂性和结果的变化。 |
| [^2] | [Entire Chain Uplift Modeling with Context-Enhanced Learning for Intelligent Marketing](https://arxiv.org/abs/2402.03379) | 全链路上升建模方法ECUP旨在解决链路偏差和处理不适应问题，在线营销中有重要的应用价值。 |
| [^3] | [Failures of Contingent Thinking.](http://arxiv.org/abs/2007.07703) | 本文提供了分析智能体错误解读或错误感知真实决策问题的理论框架，并提出了一种行为定义来评估智能体因果思维水平，同时提供了一种策略来识别智能体在缺乏完全理性的情况下的信念。 |

# 详细

[^1]: 微调、提示、上下文学习和指导微调：我们需要多少标记样本？

    Fine-Tuning, Prompting, In-Context Learning and Instruction-Tuning: How Many Labelled Samples Do We Need?

    [https://arxiv.org/abs/2402.12819](https://arxiv.org/abs/2402.12819)

    专门模型通常只需少量标记样本（100-1000个）就能与通用模型持平甚至更好，取决于任务的复杂性和结果的变化。

    

    当解决具有有限标记数据的任务时，研究人员可以选择使用通用的大型语言模型而不进行进一步更新，或者使用少量示例来调整专门的较小模型。 当有足够的标记可用时，专门的模型在许多自然语言处理任务上表现优于通用模型。 在这项工作中，我们旨在调查专门模型需要多少标记样本才能实现这种出色的性能，同时考虑结果的变化。观察提示、上下文学习、微调和指导微调的行为，识别它们在增加不同复杂性任务的标记训练样本数量时的收支平衡点，我们发现专门模型通常只需少量样本（100-1000个）就能与通用模型持平甚至更好。 同时，所需的标记数据量强烈依赖于任务的复杂性和结果的变化。

    arXiv:2402.12819v1 Announce Type: cross  Abstract: When solving a task with limited labelled data, researchers can either use a general large language model without further update, or use the few examples to tune a specialised smaller model. When enough labels are available, the specialised models outperform the general ones on many NLP tasks. In this work, we aim to investigate how many labelled samples are required for the specialised models to achieve this superior performance, while taking the results variance into consideration. Observing the behaviour of prompting, in-context learning, fine-tuning and instruction-tuning, identifying their break-even points when increasing number of labelled training samples across three tasks of varying complexity, we find that the specialised models often need only few samples ($100-1000$) to be on par or better than the general ones. At the same time, the amount of required labelled data strongly depends on the task complexity and results varia
    
[^2]: 全链路上升建模与上下文增强学习用于智能营销

    Entire Chain Uplift Modeling with Context-Enhanced Learning for Intelligent Marketing

    [https://arxiv.org/abs/2402.03379](https://arxiv.org/abs/2402.03379)

    全链路上升建模方法ECUP旨在解决链路偏差和处理不适应问题，在线营销中有重要的应用价值。

    

    上升建模在在线营销中非常重要，它旨在通过预测个体处理效果（ITE）来准确衡量不同策略（如优惠券或折扣）对不同用户的影响。在电子商务环境中，用户行为遵循确定的顺序链路，包括展示、点击和转化。营销策略在这个链路中的每个阶段都会产生不同的上升效应，影响着点击率和转化率等指标。尽管其实用性，现有研究忽视了特定处理中所有阶段的相互影响，并未充分利用处理信息，可能给后续的营销决策引入了重大偏差。本文将这两个问题称为链路偏差问题和处理不适应问题。本文介绍了一种用于解决这些问题的具有上下文增强学习的全链路上升方法（ECUP）。ECUP包括两个主要组成部分：

    Uplift modeling, vital in online marketing, seeks to accurately measure the impact of various strategies, such as coupons or discounts, on different users by predicting the Individual Treatment Effect (ITE). In an e-commerce setting, user behavior follows a defined sequential chain, including impression, click, and conversion. Marketing strategies exert varied uplift effects at each stage within this chain, impacting metrics like click-through and conversion rate. Despite its utility, existing research has neglected to consider the inter-task across all stages impacts within a specific treatment and has insufficiently utilized the treatment information, potentially introducing substantial bias into subsequent marketing decisions. We identify these two issues as the chain-bias problem and the treatment-unadaptive problem. This paper introduces the Entire Chain UPlift method with context-enhanced learning (ECUP), devised to tackle these issues. ECUP consists of two primary components: 1)
    
[^3]: 《因果思维失败》

    Failures of Contingent Thinking. (arXiv:2007.07703v3 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2007.07703](http://arxiv.org/abs/2007.07703)

    本文提供了分析智能体错误解读或错误感知真实决策问题的理论框架，并提出了一种行为定义来评估智能体因果思维水平，同时提供了一种策略来识别智能体在缺乏完全理性的情况下的信念。

    

    本文提供了一个理论框架来分析一个错误解读或错误感知真实决策问题的智能体。我们展示了一系列行为在实验环境中观察到的表现为无法理解含义，换句话说，无法正确考虑各种与关键支付相关的情况之间的逻辑关系。我们提出了对感知因果关系的行为定义，从而提供了一种引导技术，并展示了一个智能体对因果关系的解释确定了其行为的主观状态空间。通过分析这个状态空间，我们描述了驱动经验现象的逻辑复杂性的不同基准。我们区分了静态和动态的理性。因此，我们的框架既提供了评估智能体因果思维水平的方法，又提供了在没有完全理性的情况下识别其信念的策略。

    In this paper, we provide a theoretical framework to analyze an agent who misinterprets or misperceives the true decision problem she faces. We show that a wide range of behavior observed in experimental settings manifest as failures to perceive implications, in other words, to properly account for the logical relationships between various payoff relevant contingencies. We present a behavioral definition of perceived implication, thereby providing an elicitation technique, and show that an agent's account of implication identifies a subjective state-space that underlies her behavior. By analyzing this state-space, we characterize distinct benchmarks of logical sophistication that drive empirical phenomena. We disentangle static and dynamic rationality. Thus, our framework delivers both a methodology for assessing an agent's level of contingent thinking and a strategy for identifying her beliefs in the absence full rationality.
    

