# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Constructing a type-adjustable mechanism to yield Pareto-optimal outcomes.](http://arxiv.org/abs/2309.01096) | 本文提出一种可调节代理人类型的机制，通过选择最优的控制因素作为公共信息，设计者能够获得帕累托最优结果。 |
| [^2] | [GDP nowcasting with artificial neural networks: How much does long-term memory matter?.](http://arxiv.org/abs/2304.05805) | 通过比较四种人工神经网络和动态因子模型对美国GDP季度增长的预测表现，研究发现在平衡经济增长期间，更长的输入序列能够实现更准确的预测，但是这种效果会在不到两年的时间内消失。在经济动荡时期，长期记忆的效果变得明显。 |
| [^3] | [Moderation in instant runoff voting.](http://arxiv.org/abs/2303.09734) | 研究证明，相对于传统多数票投票，瞬时排名投票在对称分布的选民偏好条件下具有中和效应，可以避免选出极端候选人。 |
| [^4] | [Fictitious Play Outperforms Counterfactual Regret Minimization.](http://arxiv.org/abs/2001.11165) | 本研究比较了两种算法在近似多人博弈Nash均衡方面的表现，结果发现Fictitious Play比Counterfactual Regret Minimization更优秀。 |

# 详细

[^1]: 构建一种可调节类型机制以获得帕累托最优结果

    Constructing a type-adjustable mechanism to yield Pareto-optimal outcomes. (arXiv:2309.01096v1 [econ.TH])

    [http://arxiv.org/abs/2309.01096](http://arxiv.org/abs/2309.01096)

    本文提出一种可调节代理人类型的机制，通过选择最优的控制因素作为公共信息，设计者能够获得帕累托最优结果。

    

    在机制设计理论中，代理人的类型被描述为他们的私人信息，设计者可以揭示一些公共信息以影响代理人的类型，从而获得更多的回报。传统上，每个代理人的私人类型和公共信息分别被表示为随机变量。本文提出了一种类型可调节机制，其中每个代理人的私人类型被表示为两个参数的函数，即他的内在因素和外部控制因素。每个代理人的内在因素被建模为私人随机变量，而外部控制因素被建模为设计者优化问题的解。类型可调节机制的优点是，通过选择控制因素的最优值作为公共信息，设计者可以获得对她自己和所有代理人都有利的帕累托最优结果。相比之下，在具有相互依赖值的拍卖中，公共信息的使用限制了最优结果的实现。

    In mechanism design theory, agents' types are described as their private information, and the designer may reveal some public information to affect agents' types in order to obtain more payoffs. Traditionally, both each agent's private type and the public information are represented as a random variable respectively. In this paper, we propose a type-adjustable mechanism where each agent's private type is represented as a function of two parameters, \emph{i.e.}, his intrinsic factor and an external control factor. Each agent's intrinsic factor is modeled as a private random variable, and the external control factor is modeled as a solution of the designer's optimization problem. The advantage of the type-adjustable mechanism is that by choosing an optimal value of control factor as public information, the designer may obtain Pareto-optimal outcomes, beneficial not only to herself but also to all agents. As a comparison, in an auction with interdependent values where the public informati
    
[^2]: 用人工神经网络预测国内生产总值：长期记忆有多大的作用？

    GDP nowcasting with artificial neural networks: How much does long-term memory matter?. (arXiv:2304.05805v1 [econ.EM])

    [http://arxiv.org/abs/2304.05805](http://arxiv.org/abs/2304.05805)

    通过比较四种人工神经网络和动态因子模型对美国GDP季度增长的预测表现，研究发现在平衡经济增长期间，更长的输入序列能够实现更准确的预测，但是这种效果会在不到两年的时间内消失。在经济动荡时期，长期记忆的效果变得明显。

    

    在本研究中，我们将不同的统计模型应用于美国经济季度国内生产总值（GDP）增长预测。使用每月的FRED-MD数据库，我们比较了动态因子模型（DFM）和四个人工神经网络（ANNs）的预测表现：多层感知机（MLP）、一维卷积神经网络（1D CNN）、长短期记忆网络（LSTM）和门控循环单元（GRU）。实证分析呈现了两个不同评估周期的结果。第一个周期（2010年第1季度至2019年第4季度）具有平衡的经济增长，而第二个周期（2010年第1季度至2022年第3季度）还包括COVID-19衰退期间的时间。根据我们的结果，更长的输入序列在平衡经济增长期间能够实现更准确的预测。然而，在一个相对较低的阈值值（约六个季度或十八个月）以后，这种效应会消失。在经济动荡期（如COVID-19衰退期间），长期记忆的效果会变得较为明显。

    In our study, we apply different statistical models to nowcast quarterly GDP growth for the US economy. Using the monthly FRED-MD database, we compare the nowcasting performance of the dynamic factor model (DFM) and four artificial neural networks (ANNs): the multilayer perceptron (MLP), the one-dimensional convolutional neural network (1D CNN), the long short-term memory network (LSTM), and the gated recurrent unit (GRU). The empirical analysis presents the results from two distinctively different evaluation periods. The first (2010:Q1 -- 2019:Q4) is characterized by balanced economic growth, while the second (2010:Q1 -- 2022:Q3) also includes periods of the COVID-19 recession. According to our results, longer input sequences result in more accurate nowcasts in periods of balanced economic growth. However, this effect ceases above a relatively low threshold value of around six quarters (eighteen months). During periods of economic turbulence (e.g., during the COVID-19 recession), long
    
[^3]: 瞬时排名投票中的中和性分析

    Moderation in instant runoff voting. (arXiv:2303.09734v1 [cs.MA])

    [http://arxiv.org/abs/2303.09734](http://arxiv.org/abs/2303.09734)

    研究证明，相对于传统多数票投票，瞬时排名投票在对称分布的选民偏好条件下具有中和效应，可以避免选出极端候选人。

    

    近年来，瞬时排名投票（IRV）作为传统多数票投票的一种替代方式备受欢迎。支持者声称IRV相对于多数票投票的好处之一是它倾向于中间派：它产生比多数票更为温和的胜者，因此可以成为解决极化问题的有用工具。然而，对于这种说法，很少有理论支持，现有的证据都是基于模拟和案例研究的。在这项工作中，我们在一维欧几里得模型中的选民偏好条件下，证明了IRV相对于传统的多数票投票具有一定的中和效应。我们的结果表明，只要选民的分布是对称的，并且不太集中于极端，IRV就不会选出超过分布尾部某个阈值之外的候选人，而多数票则可能会。对于均匀分布，我们提供了推导出多数票和IRV精确分布的方法。

    Instant runoff voting (IRV) has gained popularity in recent years as an alternative to traditional plurality voting. Advocates of IRV claim that one of its benefits relative to plurality voting is its tendency toward moderation: that it produces more moderate winners than plurality and could therefore be a useful tool for addressing polarization. However, there is little theoretical backing for this claim, and existing evidence has focused on simulations and case studies. In this work, we prove that IRV has a moderating effect relative to traditional plurality voting in a specific sense, developed in a 1-dimensional Euclidean model of voter preferences. Our results show that as long as voters are symmetrically distributed and not too concentrated at the extremes, IRV will not elect a candidate that is beyond a certain threshold in the tails of the distribution, while plurality can. For the uniform distribution, we provide an approach for deriving the exact distributions of the pluralit
    
[^4]: Fictitious Play优于Counterfactual Regret Minimization

    Fictitious Play Outperforms Counterfactual Regret Minimization. (arXiv:2001.11165v7 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2001.11165](http://arxiv.org/abs/2001.11165)

    本研究比较了两种算法在近似多人博弈Nash均衡方面的表现，结果发现Fictitious Play比Counterfactual Regret Minimization更优秀。

    

    本文比较了两种广受欢迎的算法——Fictitious Play和Counterfactual Regret Minimization在近似多人博弈Nash均衡方面的表现。虽然Counterfactual Regret Minimization在多人扑克中取得了较大成功并被认为是更优秀的算法，但我们展示了Fictitious Play在各种类别和规模的游戏中都可以带来更好的Nash均衡近似效果。

    We compare the performance of two popular algorithms, fictitious play and counterfactual regret minimization, in approximating Nash equilibrium in multiplayer games. Despite recent success of counterfactual regret minimization in multiplayer poker and conjectures of its superiority, we show that fictitious play leads to improved Nash equilibrium approximation over a variety of game classes and sizes.
    

