# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Seemingly unrelated Bayesian additive regression trees for cost-effectiveness analyses in healthcare](https://arxiv.org/abs/2404.02228) | 提出了适用于医疗保健成本效益分析的多元贝叶斯可加回归树扩展，克服了现有模型的局限性，可以处理多个相关结果变量的回归和分类分析 |
| [^2] | [Calibrating doubly-robust estimators with unbalanced treatment assignment](https://arxiv.org/abs/2403.01585) | 提出了一个简单的DML估计器扩展，通过对概率得分建模进行欠采样，并校准分数以匹配原始分布，以解决处理分配不平衡问题。 |
| [^3] | [Social Environment Design](https://arxiv.org/abs/2402.14090) | 该论文提出了一种新的研究议程，介绍了社会环境设计作为一种用于自动化政策制定的AI通用框架，旨在捕捉一般经济环境，通过AI模拟系统分析政府和经济政策，并强调未来基于AI的政策制定研究中的关键挑战。 |
| [^4] | [Persuading a Learning Agent](https://arxiv.org/abs/2402.09721) | 在一个重复的贝叶斯说服问题中，即使没有承诺能力，委托人可以通过使用上下文无遗憾学习算法来实现与经典无学习模型中具有承诺的委托人的最优效用无限接近的效果；在代理人使用上下文无交换遗憾学习算法的情况下，委托人无法获得比具有承诺的无学习模型中的最优效用更高的效用。 |
| [^5] | [Algorithmic Persuasion Through Simulation](https://arxiv.org/abs/2311.18138) | 通过模拟接收者行为的贝叶斯劝导问题中，发送者设计了一个最优消息策略并设计了一个多项式时间查询算法，以优化其预期效用。 |

# 详细

[^1]: 基于贝叶斯可加回归树的医疗保健成本效益分析

    Seemingly unrelated Bayesian additive regression trees for cost-effectiveness analyses in healthcare

    [https://arxiv.org/abs/2404.02228](https://arxiv.org/abs/2404.02228)

    提出了适用于医疗保健成本效益分析的多元贝叶斯可加回归树扩展，克服了现有模型的局限性，可以处理多个相关结果变量的回归和分类分析

    

    近年来的理论结果和模拟证据表明，贝叶斯可加回归树是一种非常有效的非参数回归方法。受到在卫生经济学中的成本效益分析的启发，我们提出了适用于具有多个相关结果变量的回归和分类分析的BART的多元扩展。我们的框架通过允许每个个体响应与不同树组相关联，同时处理结果之间的依赖关系，克服了现有多元BART模型的一些主要局限性。在连续结果的情况下，我们的模型本质上是表面无关回归的非参数版本。同样，我们针对二元结果的建议是非参数概括

    arXiv:2404.02228v1 Announce Type: cross  Abstract: In recent years, theoretical results and simulation evidence have shown Bayesian additive regression trees to be a highly-effective method for nonparametric regression. Motivated by cost-effectiveness analyses in health economics, where interest lies in jointly modelling the costs of healthcare treatments and the associated health-related quality of life experienced by a patient, we propose a multivariate extension of BART applicable in regression and classification analyses with several correlated outcome variables. Our framework overcomes some key limitations of existing multivariate BART models by allowing each individual response to be associated with different ensembles of trees, while still handling dependencies between the outcomes. In the case of continuous outcomes, our model is essentially a nonparametric version of seemingly unrelated regression. Likewise, our proposal for binary outcomes is a nonparametric generalisation of
    
[^2]: 使用不平衡的处理分配校准双重稳健估计器

    Calibrating doubly-robust estimators with unbalanced treatment assignment

    [https://arxiv.org/abs/2403.01585](https://arxiv.org/abs/2403.01585)

    提出了一个简单的DML估计器扩展，通过对概率得分建模进行欠采样，并校准分数以匹配原始分布，以解决处理分配不平衡问题。

    

    机器学习方法，尤其是双机器学习（DML）估计器（Chernozhukov等，2018），越来越受欢迎地用于估计平均处理效应（ATE）。然而，数据集通常表现出处理分配不平衡，只有少数观测值被处理，导致稳健概率得分估计不稳定。我们提出了DML估计器的简单扩展，该扩展对概率得分建模进行了欠采样，并校准分数以匹配原始分布。本文提供了理论结果表明，该估计器保留了DML估计器的渐近特性。模拟研究说明了估计器的有限样本性能。

    arXiv:2403.01585v1 Announce Type: new  Abstract: Machine learning methods, particularly the double machine learning (DML) estimator (Chernozhukov et al., 2018), are increasingly popular for the estimation of the average treatment effect (ATE). However, datasets often exhibit unbalanced treatment assignments where only a few observations are treated, leading to unstable propensity score estimations. We propose a simple extension of the DML estimator which undersamples data for propensity score modeling and calibrates scores to match the original distribution. The paper provides theoretical results showing that the estimator retains the DML estimator's asymptotic properties. A simulation study illustrates the finite sample performance of the estimator.
    
[^3]: 社会环境设计

    Social Environment Design

    [https://arxiv.org/abs/2402.14090](https://arxiv.org/abs/2402.14090)

    该论文提出了一种新的研究议程，介绍了社会环境设计作为一种用于自动化政策制定的AI通用框架，旨在捕捉一般经济环境，通过AI模拟系统分析政府和经济政策，并强调未来基于AI的政策制定研究中的关键挑战。

    

    人工智能（AI）作为一种用于改善政府和经济政策制定的技术具有潜力。本文提出了一个新的研究议程，介绍了社会环境设计，这是一种用于自动化政策制定的AI通用框架，与强化学习、经济与计算社会选择社区相连接。该框架旨在捕捉一般经济环境，包括对政策目标的投票，并为通过AI模拟对政府和经济政策进行系统分析提供指导。我们强调了未来基于AI的政策制定研究中的关键开放问题。通过解决这些挑战，我们希望实现各种社会福利目标，从而促进更具道德和负责任的决策制定。

    arXiv:2402.14090v1 Announce Type: new  Abstract: Artificial Intelligence (AI) holds promise as a technology that can be used to improve government and economic policy-making. This paper proposes a new research agenda towards this end by introducing Social Environment Design, a general framework for the use of AI for automated policy-making that connects with the Reinforcement Learning, EconCS, and Computational Social Choice communities. The framework seeks to capture general economic environments, includes voting on policy objectives, and gives a direction for the systematic analysis of government and economic policy through AI simulation. We highlight key open problems for future research in AI-based policy-making. By solving these challenges, we hope to achieve various social welfare objectives, thereby promoting more ethical and responsible decision making.
    
[^4]: 说服一位学习代理

    Persuading a Learning Agent

    [https://arxiv.org/abs/2402.09721](https://arxiv.org/abs/2402.09721)

    在一个重复的贝叶斯说服问题中，即使没有承诺能力，委托人可以通过使用上下文无遗憾学习算法来实现与经典无学习模型中具有承诺的委托人的最优效用无限接近的效果；在代理人使用上下文无交换遗憾学习算法的情况下，委托人无法获得比具有承诺的无学习模型中的最优效用更高的效用。

    

    我们研究了一个重复的贝叶斯说服问题（更一般地，任何具有完全信息的广义委托-代理问题），其中委托人没有承诺能力，代理人使用算法来学习如何对委托人的信号做出响应。我们将这个问题简化为一个一次性的广义委托-代理问题，代理人近似地最佳响应。通过这个简化，我们可以证明：如果代理人使用上下文无遗憾学习算法，则委托人可以保证其效用与经典无学习模型中具有承诺的委托人的最优效用之间可以无限接近；如果代理人使用上下文无交换遗憾学习算法，则委托人无法获得比具有承诺的无学习模型中的最优效用更高的效用。委托人在学习模型与非学习模型中可以获得的效用之间的差距是有界的。

    arXiv:2402.09721v1 Announce Type: cross  Abstract: We study a repeated Bayesian persuasion problem (and more generally, any generalized principal-agent problem with complete information) where the principal does not have commitment power and the agent uses algorithms to learn to respond to the principal's signals. We reduce this problem to a one-shot generalized principal-agent problem with an approximately-best-responding agent. This reduction allows us to show that: if the agent uses contextual no-regret learning algorithms, then the principal can guarantee a utility that is arbitrarily close to the principal's optimal utility in the classic non-learning model with commitment; if the agent uses contextual no-swap-regret learning algorithms, then the principal cannot obtain any utility significantly more than the optimal utility in the non-learning model with commitment. The difference between the principal's obtainable utility in the learning model and the non-learning model is bound
    
[^5]: 通过模拟进行算法性劝导

    Algorithmic Persuasion Through Simulation

    [https://arxiv.org/abs/2311.18138](https://arxiv.org/abs/2311.18138)

    通过模拟接收者行为的贝叶斯劝导问题中，发送者设计了一个最优消息策略并设计了一个多项式时间查询算法，以优化其预期效用。

    

    我们研究了一个贝叶斯劝导问题，其中发送者希望说服接收者采取二元行为，例如购买产品。发送者了解世界的（二元）状态，比如产品质量是高还是低，但是对接收者的信念和效用只有有限的信息。受到客户调查、用户研究和生成式人工智能的最新进展的启发，我们允许发送者通过查询模拟接收者的行为来了解更多关于接收者的信息。在固定数量的查询之后，发送者承诺一个消息策略，接收者根据收到的消息来最大化她的预期效用来采取行动。我们对发送者在任何接收者类型分布下的最优消息策略进行了表征。然后，我们设计了一个多项式时间查询算法，优化了这个贝叶斯劝导游戏中发送者的预期效用。

    arXiv:2311.18138v2 Announce Type: replace-cross Abstract: We study a Bayesian persuasion problem where a sender wants to persuade a receiver to take a binary action, such as purchasing a product. The sender is informed about the (binary) state of the world, such as whether the quality of the product is high or low, but only has limited information about the receiver's beliefs and utilities. Motivated by customer surveys, user studies, and recent advances in generative AI, we allow the sender to learn more about the receiver by querying an oracle that simulates the receiver's behavior. After a fixed number of queries, the sender commits to a messaging policy and the receiver takes the action that maximizes her expected utility given the message she receives. We characterize the sender's optimal messaging policy given any distribution over receiver types. We then design a polynomial-time querying algorithm that optimizes the sender's expected utility in this Bayesian persuasion game. We 
    

