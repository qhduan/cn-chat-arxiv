# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robustly estimating heterogeneity in factorial data using Rashomon Partitions](https://arxiv.org/abs/2404.02141) | 通过使用拉细孟划分集，我们能够在因子数据中稳健地估计异质性，并将因子空间划分成协变量组合的“池”，以便区分结果的差异。 |
| [^2] | [Collective Evaluation Problem](https://arxiv.org/abs/2402.16309) | 本研究探讨了集体评估函数的属性，并确定了确保满足四种不同组合属性的集体评估函数存在的评估能力配置文件的必要和充分条件。 |
| [^3] | [LLM Voting: Human Choices and AI Collective Decision Making](https://arxiv.org/abs/2402.01766) | 本文研究了大型语言模型（LLMs），特别是OpenAI的GPT4和LLaMA2的投票行为，并揭示了LLMs与人类在决策和偏见方面的差异。研究发现，在投票辅助中使用LLMs可能会导致更同质化的集体结果，强调了谨慎将LLMs整合到民主过程中的必要性。 |
| [^4] | [Significance Bands for Local Projections.](http://arxiv.org/abs/2306.03073) | 本文表明，在局部投影分析中，应该使用显著性带来评估措施对结果的影响，而不是使用常用的置信度带。 |

# 详细

[^1]: 使用拉细孟划分在因子数据中稳健估计异质性

    Robustly estimating heterogeneity in factorial data using Rashomon Partitions

    [https://arxiv.org/abs/2404.02141](https://arxiv.org/abs/2404.02141)

    通过使用拉细孟划分集，我们能够在因子数据中稳健地估计异质性，并将因子空间划分成协变量组合的“池”，以便区分结果的差异。

    

    许多统计分析，无论是在观测数据还是随机对照试验中，都会问：感兴趣的结果如何随可观察协变量组合变化？不同的药物组合如何影响健康结果，科技采纳如何依赖激励和人口统计学？我们的目标是将这个因子空间划分成协变量组合的“池”，在这些池中结果会发生差异（但池内部不会发生），而现有方法要么寻找一个单一的“最优”分割，要么从可能分割的整个集合中抽样。这两种方法都忽视了这样一个事实：特别是在协变量之间存在相关结构的情况下，可能以许多种方式划分协变量空间，在统计上是无法区分的，尽管对政策或科学有着非常不同的影响。我们提出了一种名为拉细孟划分集的替代视角

    arXiv:2404.02141v1 Announce Type: cross  Abstract: Many statistical analyses, in both observational data and randomized control trials, ask: how does the outcome of interest vary with combinations of observable covariates? How do various drug combinations affect health outcomes, or how does technology adoption depend on incentives and demographics? Our goal is to partition this factorial space into ``pools'' of covariate combinations where the outcome differs across the pools (but not within a pool). Existing approaches (i) search for a single ``optimal'' partition under assumptions about the association between covariates or (ii) sample from the entire set of possible partitions. Both these approaches ignore the reality that, especially with correlation structure in covariates, many ways to partition the covariate space may be statistically indistinguishable, despite very different implications for policy or science. We develop an alternative perspective, called Rashomon Partition Set
    
[^2]: 集体评估问题

    Collective Evaluation Problem

    [https://arxiv.org/abs/2402.16309](https://arxiv.org/abs/2402.16309)

    本研究探讨了集体评估函数的属性，并确定了确保满足四种不同组合属性的集体评估函数存在的评估能力配置文件的必要和充分条件。

    

    这项研究聚焦于通过收集来自多个个体的评估来评估有限的替代方案的情况，其中一些个体可能不评估特定的替代方案。个体（可）评估的替代方案子集的集合被称为评估能力配置文件。针对给定的评估能力配置文件，我们定义了一个集体评估函数，其输入是个体对他们评估的替代方案子集的评估顺序。我们研究了集体评估函数的特性，这些特性是对先前研究中引入的特性的修改。我们确定了关于评估能力配置文件的充分必要条件，以确保存在满足这些特性中四种不同组合的集体评估函数。

    arXiv:2402.16309v1 Announce Type: new  Abstract: This study focuses on situations where a finite set of alternatives is evaluated by collecting evaluations from several individuals, some of whom may not evaluate specific alternatives. The collection of subsets of alternatives that individuals (can) evaluate is referred to as an evaluability profile. For a given evaluability profile, we define a collective evaluation function whose inputs are the evaluation orders of individuals on the subsets of alternatives that they evaluate. We investigate the properties of collective evaluation functions, which are modifications of those introduced in previous studies. We identify the necessary and sufficient conditions on the evaluability profile that ensure the existence of collective evaluation functions satisfying four different combinations of these properties.
    
[^3]: LLM投票：人类选择和AI集体决策

    LLM Voting: Human Choices and AI Collective Decision Making

    [https://arxiv.org/abs/2402.01766](https://arxiv.org/abs/2402.01766)

    本文研究了大型语言模型（LLMs），特别是OpenAI的GPT4和LLaMA2的投票行为，并揭示了LLMs与人类在决策和偏见方面的差异。研究发现，在投票辅助中使用LLMs可能会导致更同质化的集体结果，强调了谨慎将LLMs整合到民主过程中的必要性。

    

    本文研究了大型语言模型（LLMs），特别是OpenAI的GPT4和LLaMA2的投票行为，并与人类投票模式进行了对比。我们的方法包括进行人类投票实验以建立人类偏好的基准，并与LLM代理进行平行实验。研究聚焦于集体结果和个体偏好，揭示了人类和LLMs之间在决策和固有偏见方面的差异。我们观察到LLMs在偏好多样性和一致性之间存在权衡，相比人类选民的多样偏好，LLMs有更趋向于一致选择的倾向。这一发现表明，在投票辅助中使用LLMs可能会导致更同质化的集体结果，强调了谨慎将LLMs整合到民主过程中的必要性。

    This paper investigates the voting behaviors of Large Language Models (LLMs), particularly OpenAI's GPT4 and LLaMA2, and their alignment with human voting patterns. Our approach included a human voting experiment to establish a baseline for human preferences and a parallel experiment with LLM agents. The study focused on both collective outcomes and individual preferences, revealing differences in decision-making and inherent biases between humans and LLMs. We observed a trade-off between preference diversity and alignment in LLMs, with a tendency towards more uniform choices as compared to the diverse preferences of human voters. This finding indicates that LLMs could lead to more homogenized collective outcomes when used in voting assistance, underscoring the need for cautious integration of LLMs into democratic processes.
    
[^4]: 局部投影的显著性带

    Significance Bands for Local Projections. (arXiv:2306.03073v1 [econ.EM])

    [http://arxiv.org/abs/2306.03073](http://arxiv.org/abs/2306.03073)

    本文表明，在局部投影分析中，应该使用显著性带来评估措施对结果的影响，而不是使用常用的置信度带。

    

    冲击反应函数描述了刺激或治疗后结果变量的动态演变。一个常见的兴趣假设是治疗是否影响了结果。我们表明，最好使用显著性带来评估这个假设，而不是依赖于通常显示的置信度带。在零假设下，我们展示了使用LM原则可以使用标准统计软件轻松构建显著性带，并且在图形化显示冲击反应时应当作为常规报告。

    An impulse response function describes the dynamic evolution of an outcome variable following a stimulus or treatment. A common hypothesis of interest is whether the treatment affects the outcome. We show that this hypothesis is best assessed using significance bands rather than relying on commonly displayed confidence bands. Under the null hypothesis, we show that significance bands are trivial to construct with standard statistical software using the LM principle, and should be reported as a matter of routine when displaying impulse responses graphically.
    

