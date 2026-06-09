# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Survey on Large Language Model-Based Game Agents](https://arxiv.org/abs/2404.02039) | LLM和MLLM的进步为游戏智能体提供了强大的人类决策能力，本文全面综述了基于LLM的游戏智能体的概念架构、方法论和未来研究方向 |
| [^2] | [Investigating the Histogram Loss in Regression](https://arxiv.org/abs/2402.13425) | 学习整个分布在回归中的性能提升主要来自于优化的改进，而不是学习更好的表示。 |
| [^3] | [Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook.](http://arxiv.org/abs/2310.10196) | 这篇综述探讨了大型模型在时间序列和时空数据中的应用。它们不仅带来了增强的模式识别和推理能力，还为人工通用智能打下了基础。 |

# 详细

[^1]: 基于大型语言模型的游戏智能体综述

    A Survey on Large Language Model-Based Game Agents

    [https://arxiv.org/abs/2404.02039](https://arxiv.org/abs/2404.02039)

    LLM和MLLM的进步为游戏智能体提供了强大的人类决策能力，本文全面综述了基于LLM的游戏智能体的概念架构、方法论和未来研究方向

    

    游戏智能体的发展在推动人工通用智能（AGI）方面扮演着关键角色。LLM及其多模态对应物（MLLM）的进展为游戏智能体在复杂的电脑游戏环境中具备类似人类决策能力提供了前所未有的机会。本文从整体视角全面概述了基于LLM的游戏智能体。首先，我们介绍了以感知、记忆、思维、角色扮演、行动和学习为中心的LLM游戏智能体的概念架构。其次，我们调查了文献中已有的代表性LLM游戏智能体，涉及到六类游戏中的方法论和适应能力，包括冒险、沟通、竞争、合作、模拟以及创造与探索游戏。最后，我们展望了未来研究的方向。

    arXiv:2404.02039v1 Announce Type: new  Abstract: The development of game agents holds a critical role in advancing towards Artificial General Intelligence (AGI). The progress of LLMs and their multimodal counterparts (MLLMs) offers an unprecedented opportunity to evolve and empower game agents with human-like decision-making capabilities in complex computer game environments. This paper provides a comprehensive overview of LLM-based game agents from a holistic viewpoint. First, we introduce the conceptual architecture of LLM-based game agents, centered around six essential functional components: perception, memory, thinking, role-playing, action, and learning. Second, we survey existing representative LLM-based game agents documented in the literature with respect to methodologies and adaptation agility across six genres of games, including adventure, communication, competition, cooperation, simulation, and crafting & exploration games. Finally, we present an outlook of future research
    
[^2]: 在回归中探讨直方图损失

    Investigating the Histogram Loss in Regression

    [https://arxiv.org/abs/2402.13425](https://arxiv.org/abs/2402.13425)

    学习整个分布在回归中的性能提升主要来自于优化的改进，而不是学习更好的表示。

    

    越来越常见的是，在回归中训练神经网络来建模整个分布，即使只需要均值来进行预测。 这种额外的建模通常会带来性能增益，但背后的原因尚不完全清楚。 本文研究了回归中的一种最新方法，即直方图损失，该方法通过最小化目标分布和灵活直方图预测之间的交叉熵来学习目标变量的条件分布。 我们设计了理论和实证分析，以确定为什么以及何时会出现性能增益，以及损失的不同组件如何为此做出贡献。 我们的结果表明，在这种设置中学习分布的好处来自于优化的改进，而不是学习更好的表示。 然后，我们展示了直方图损失在常见的深度学习应用中的可行性。

    arXiv:2402.13425v1 Announce Type: cross  Abstract: It is becoming increasingly common in regression to train neural networks that model the entire distribution even if only the mean is required for prediction. This additional modeling often comes with performance gain and the reasons behind the improvement are not fully known. This paper investigates a recent approach to regression, the Histogram Loss, which involves learning the conditional distribution of the target variable by minimizing the cross-entropy between a target distribution and a flexible histogram prediction. We design theoretical and empirical analyses to determine why and when this performance gain appears, and how different components of the loss contribute to it. Our results suggest that the benefits of learning distributions in this setup come from improvements in optimization rather than learning a better representation. We then demonstrate the viability of the Histogram Loss in common deep learning applications wi
    
[^3]: 时间序列和时空数据的大型模型：综述与展望

    Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook. (arXiv:2310.10196v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.10196](http://arxiv.org/abs/2310.10196)

    这篇综述探讨了大型模型在时间序列和时空数据中的应用。它们不仅带来了增强的模式识别和推理能力，还为人工通用智能打下了基础。

    

    时间数据，特别是时间序列和时空数据，在现实世界的应用中非常普遍。它们捕捉动态系统的测量数据，并由物理和虚拟传感器大量产生。分析这些数据类型对于利用它们所包含的丰富信息以及受益于各种下游任务非常重要。近年来，大型语言和其他基础模型的突破推动了这些模型在时间序列和时空数据挖掘中的增加使用。这样的方法不仅能够实现跨不同领域的增强模式识别和推理，还为能够理解和处理常见时间数据的人工通用智能打下了基础。在此综述中，我们提供了针对时间序列和时空数据定制的大型模型的全面和最新综述，包括数据类型、模型类别、模型范围和应用领域/任务。我们的目标是

    Temporal data, notably time series and spatio-temporal data, are prevalent in real-world applications. They capture dynamic system measurements and are produced in vast quantities by both physical and virtual sensors. Analyzing these data types is vital to harnessing the rich information they encompass and thus benefits a wide range of downstream tasks. Recent advances in large language and other foundational models have spurred increased use of these models in time series and spatio-temporal data mining. Such methodologies not only enable enhanced pattern recognition and reasoning across diverse domains but also lay the groundwork for artificial general intelligence capable of comprehending and processing common temporal data. In this survey, we offer a comprehensive and up-to-date review of large models tailored (or adapted) for time series and spatio-temporal data, spanning four key facets: data types, model categories, model scopes, and application areas/tasks. Our objective is to 
    

