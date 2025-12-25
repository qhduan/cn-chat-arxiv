# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Eliciting Risk Aversion with Inverse Reinforcement Learning via Interactive Questioning.](http://arxiv.org/abs/2308.08427) | 本文提出了一个新的方法，通过交互式问答来识别代理人的风险规避。在一期情景和无限期情景下，我们通过要求代理人展示她的最优策略来回答问题，使用随机设计的问题来识别代理人的风险规避。这个方法可以通过一个有限的候选集有效地识别出代理人的风险规避。 |

# 详细

[^1]: 借助交互式问答通过逆强化学习来引导风险规避

    Eliciting Risk Aversion with Inverse Reinforcement Learning via Interactive Questioning. (arXiv:2308.08427v1 [stat.ML])

    [http://arxiv.org/abs/2308.08427](http://arxiv.org/abs/2308.08427)

    本文提出了一个新的方法，通过交互式问答来识别代理人的风险规避。在一期情景和无限期情景下，我们通过要求代理人展示她的最优策略来回答问题，使用随机设计的问题来识别代理人的风险规避。这个方法可以通过一个有限的候选集有效地识别出代理人的风险规避。

    

    本文提出了一个新颖的框架，利用交互式问答来识别代理人的风险规避。我们的研究在两种情景中进行：一期情景和无限期情景。在一期情景中，我们假设代理人的风险规避由状态的成本函数和失真风险度量所表征。在无限期情景中，我们用一个额外的成分，折扣因子，来建模风险规避。假设我们可以访问一个包含代理人真实风险规避的有限候选集，我们证明通过要求代理人在各种环境中展示她的最优政策来回答问题，这可以有效地识别代理人的风险规避。具体而言，我们证明了当问题的数量趋近无穷大并且问题是随机设计的时候，可以识别出代理人的风险规避。我们还开发了一个算法用于设计最优问题，并提供了实证证据来支持我们的方法。

    This paper proposes a novel framework for identifying an agent's risk aversion using interactive questioning. Our study is conducted in two scenarios: a one-period case and an infinite horizon case. In the one-period case, we assume that the agent's risk aversion is characterized by a cost function of the state and a distortion risk measure. In the infinite horizon case, we model risk aversion with an additional component, a discount factor. Assuming the access to a finite set of candidates containing the agent's true risk aversion, we show that asking the agent to demonstrate her optimal policies in various environment, which may depend on their previous answers, is an effective means of identifying the agent's risk aversion. Specifically, we prove that the agent's risk aversion can be identified as the number of questions tends to infinity, and the questions are randomly designed. We also develop an algorithm for designing optimal questions and provide empirical evidence that our met
    

