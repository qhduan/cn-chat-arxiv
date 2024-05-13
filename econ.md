# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Counterfactual Sensitivity in Quantitative Spatial Models](https://arxiv.org/abs/2311.14032) | 本文提出了一种用于定量空间模型中处理反事实不确定性的方法，利用经验贝叶斯方法量化不确定性，并在两个应用程序中发现了有关反事实的非平凡不确定性。 |
| [^2] | [Multi-agent Deep Reinforcement Learning for Dynamic Pricing by Fast-charging Electric Vehicle Hubs in ccompetition.](http://arxiv.org/abs/2401.15108) | 本文提出了一个多智能体深度强化学习的方法，应用于快速充电电动车中心的动态定价竞争。通过预测性购买电力需求和设定竞争性价格策略，充电站可以在竞争中进行有效定价。 |

# 详细

[^1]: 定量空间模型中的反事实敏感性

    Counterfactual Sensitivity in Quantitative Spatial Models

    [https://arxiv.org/abs/2311.14032](https://arxiv.org/abs/2311.14032)

    本文提出了一种用于定量空间模型中处理反事实不确定性的方法，利用经验贝叶斯方法量化不确定性，并在两个应用程序中发现了有关反事实的非平凡不确定性。

    

    定量空间模型中的反事实是当前世界状态和模型参数的函数。当前做法将当前世界状态视为完全可观测，但我们有充分理由相信存在测量误差。本文提供了一种工具，用于在当前世界状态存在测量误差时量化关于反事实的不确定性。我推荐一种经验贝叶斯方法用于不确定性量化，这既实用又在理论上被证明了。我将所提出的方法应用于Adao, Costinot和Donaldson (2017)以及Allen和Arkolakis (2022)的应用中，并发现有关反事实的非平凡不确定性。

    arXiv:2311.14032v2 Announce Type: replace  Abstract: Counterfactuals in quantitative spatial models are functions of the current state of the world and the model parameters. Current practice treats the current state of the world as perfectly observed, but there is good reason to believe that it is measured with error. This paper provides tools for quantifying uncertainty about counterfactuals when the current state of the world is measured with error. I recommend an empirical Bayes approach to uncertainty quantification, which is both practical and theoretically justified. I apply the proposed method to the applications in Adao, Costinot, and Donaldson (2017) and Allen and Arkolakis (2022) and find non-trivial uncertainty about counterfactuals.
    
[^2]: 多智能体深度强化学习在竞争中为快速充电电动车中心的动态定价中的应用

    Multi-agent Deep Reinforcement Learning for Dynamic Pricing by Fast-charging Electric Vehicle Hubs in ccompetition. (arXiv:2401.15108v1 [cs.LG])

    [http://arxiv.org/abs/2401.15108](http://arxiv.org/abs/2401.15108)

    本文提出了一个多智能体深度强化学习的方法，应用于快速充电电动车中心的动态定价竞争。通过预测性购买电力需求和设定竞争性价格策略，充电站可以在竞争中进行有效定价。

    

    快速充电站将成为全球新建交通电气化基础设施的一部分。这些充电站将承载许多直流快速充电设备，仅可供电动车辆充电使用。类似于汽油加油站，同一地区的快速充电站将根据竞争调整价格以吸引同一群电动车主。这些充电站将与电力网络进行交互，通过预测性购买在前一天电力市场上的电力需求，并在实时市场上满足差额需求。充电站可能配备补充电池储能系统用于套利。本文针对充电站竞争中开发了一个两步数据驱动的动态定价方法。首先通过求解随机的前一天电力需求模型得到纳入承诺，然后通过将游戏建模为竞争来得到充电站的价格策略。

    Fast-charging hubs for electric vehicles will soon become part of the newly built infrastructure for transportation electrification across the world. These hubs are expected to host many DC fast-charging stations and will admit EVs only for charging. Like the gasoline refueling stations, fast-charging hubs in a neighborhood will dynamically vary their prices to compete for the same pool of EV owners. These hubs will interact with the electric power network by making purchase commitments for a significant part of their power needs in the day-ahead (DA) electricity market and meeting the difference from the real-time (RT) market. Hubs may have supplemental battery storage systems (BSS), which they will use for arbitrage. In this paper, we develop a two-step data-driven dynamic pricing methodology for hubs in price competition. We first obtain the DA commitment by solving a stochastic DA commitment model. Thereafter we obtain the hub pricing strategies by modeling the game as a competitiv
    

