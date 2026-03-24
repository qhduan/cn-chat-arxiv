# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Towards Urban General Intelligence: A Review and Outlook of Urban Foundation Models](https://arxiv.org/abs/2402.01749) | 本文综述了城市基础模型在智能城市发展中的重要性和潜力，并提出了一个以数据为中心的分类方法。这个新兴领域面临着一些挑战，如缺乏清晰的定义和系统性的综述，需要进一步的研究和解决方案。 |
| [^2] | [Policy Optimization over General State and Action Spaces.](http://arxiv.org/abs/2211.16715) | 本文提出了一种新方法并引入了函数近似来解决通用状态和动作空间上的强化学习问题，同时介绍了一种新的策略双平均法。 |

# 详细

[^1]: 迈向城市智能：城市基础模型综述与展望

    Towards Urban General Intelligence: A Review and Outlook of Urban Foundation Models

    [https://arxiv.org/abs/2402.01749](https://arxiv.org/abs/2402.01749)

    本文综述了城市基础模型在智能城市发展中的重要性和潜力，并提出了一个以数据为中心的分类方法。这个新兴领域面临着一些挑战，如缺乏清晰的定义和系统性的综述，需要进一步的研究和解决方案。

    

    机器学习技术现已成为智能城市服务进步的核心，对提高城市环境的效率、可持续性和宜居性起到至关重要的作用。最近出现的ChatGPT等基础模型在机器学习和人工智能领域标志着一个革命性的转变。它们在上下文理解、问题解决和适应各种任务方面的无与伦比的能力表明，将这些模型整合到城市领域中可能对智能城市的发展产生变革性影响。尽管对城市基础模型（UFMs）的兴趣日益增长，但这个新兴领域面临着一些挑战，如缺乏清晰的定义、系统性的综述和可普遍化的解决方案。为此，本文首先介绍了UFM的概念，并讨论了构建它们所面临的独特挑战。然后，我们提出了一个以数据为中心的分类方法，对当前与UFM相关的工作进行了分类。

    Machine learning techniques are now integral to the advancement of intelligent urban services, playing a crucial role in elevating the efficiency, sustainability, and livability of urban environments. The recent emergence of foundation models such as ChatGPT marks a revolutionary shift in the fields of machine learning and artificial intelligence. Their unparalleled capabilities in contextual understanding, problem solving, and adaptability across a wide range of tasks suggest that integrating these models into urban domains could have a transformative impact on the development of smart cities. Despite growing interest in Urban Foundation Models~(UFMs), this burgeoning field faces challenges such as a lack of clear definitions, systematic reviews, and universalizable solutions. To this end, this paper first introduces the concept of UFM and discusses the unique challenges involved in building them. We then propose a data-centric taxonomy that categorizes current UFM-related works, base
    
[^2]: 通用状态和动作空间上的策略优化

    Policy Optimization over General State and Action Spaces. (arXiv:2211.16715v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.16715](http://arxiv.org/abs/2211.16715)

    本文提出了一种新方法并引入了函数近似来解决通用状态和动作空间上的强化学习问题，同时介绍了一种新的策略双平均法。

    

    通用状态和动作空间上的强化学习问题异常困难。本文提出了一种新方法，并引入了函数近似来解决这个问题。同时，还提出了一种新的策略双平均法。这些方法都可以应用于不同类型的RL问题。

    Reinforcement learning (RL) problems over general state and action spaces are notoriously challenging. In contrast to the tableau setting, one can not enumerate all the states and then iteratively update the policies for each state. This prevents the application of many well-studied RL methods especially those with provable convergence guarantees. In this paper, we first present a substantial generalization of the recently developed policy mirror descent method to deal with general state and action spaces. We introduce new approaches to incorporate function approximation into this method, so that we do not need to use explicit policy parameterization at all. Moreover, we present a novel policy dual averaging method for which possibly simpler function approximation techniques can be applied. We establish linear convergence rate to global optimality or sublinear convergence to stationarity for these methods applied to solve different classes of RL problems under exact policy evaluation. 
    

