# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [ABIDES-Economist: Agent-Based Simulation of Economic Systems with Learning Agents](https://arxiv.org/abs/2402.09563) | ABIDES-Economist是一个多智能体模拟器，用于经济系统，具有学习代理、规则性策略和基于现实数据的设计。它提供了一种使用强化学习策略的模拟环境，并可以模拟和分析各种经济情景。 |
| [^2] | [Inference in Cluster Randomized Trials with Matched Pairs.](http://arxiv.org/abs/2211.14903) | 本文研究了在匹配对簇随机试验中进行统计推断的问题，提出了加权均值差估计量和方差估计量，建立了基于这些估计量的渐近精确性，并探讨了常用的两种测试程序的特性。 |

# 详细

[^1]: ABIDES-Economist: 具有学习代理的经济系统的基于代理的模拟

    ABIDES-Economist: Agent-Based Simulation of Economic Systems with Learning Agents

    [https://arxiv.org/abs/2402.09563](https://arxiv.org/abs/2402.09563)

    ABIDES-Economist是一个多智能体模拟器，用于经济系统，具有学习代理、规则性策略和基于现实数据的设计。它提供了一种使用强化学习策略的模拟环境，并可以模拟和分析各种经济情景。

    

    我们介绍了一个多智能体模拟器，用于由异质家庭、异质公司、中央银行和政府代理组成的经济系统，该系统可以受到外生的随机冲击。代理之间的互动定义了经济中商品的生产和消费以及资金的流动。每个代理可以根据固定的、规则性的策略行动，也可以通过与模拟器中其他代理的互动来学习自己的策略。我们通过选择基于经济文献的代理异质性参数，并将其行动空间设计与美国的实际数据相一致，来使我们的模拟器具备现实基础。我们的模拟器通过为经济系统定义 OpenAI Gym 风格的环境，促进了代理使用强化学习策略的能力。通过模拟和分析两种假设的（但有趣的）经济情景，我们展示了我们模拟器的实用性。

    arXiv:2402.09563v1 Announce Type: cross  Abstract: We introduce a multi-agent simulator for economic systems comprised of heterogeneous Households, heterogeneous Firms, Central Bank and Government agents, that could be subjected to exogenous, stochastic shocks. The interaction between agents defines the production and consumption of goods in the economy alongside the flow of money. Each agent can be designed to act according to fixed, rule-based strategies or learn their strategies using interactions with others in the simulator. We ground our simulator by choosing agent heterogeneity parameters based on economic literature, while designing their action spaces in accordance with real data in the United States. Our simulator facilitates the use of reinforcement learning strategies for the agents via an OpenAI Gym style environment definition for the economic system. We demonstrate the utility of our simulator by simulating and analyzing two hypothetical (yet interesting) economic scenar
    
[^2]: 匹配对簇随机试验中的推断

    Inference in Cluster Randomized Trials with Matched Pairs. (arXiv:2211.14903v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2211.14903](http://arxiv.org/abs/2211.14903)

    本文研究了在匹配对簇随机试验中进行统计推断的问题，提出了加权均值差估计量和方差估计量，建立了基于这些估计量的渐近精确性，并探讨了常用的两种测试程序的特性。

    

    本文研究了在匹配对簇随机试验中进行推断的问题。匹配对设计意味着根据基线簇级协变量对一组簇进行匹配，然后在每对簇中随机选择一个簇进行处理。我们研究了加权均值差估计量的大样本行为，并根据匹配过程是否匹配簇大小，导出了两组不同的结果。然后，我们提出了一个方差估计量，无论匹配过程是匹配簇大小还是不匹配簇大小，都是一致的。结合这些结果，建立了基于这些估计量的检验的渐近精确性。接下来，我们考虑了基于线性回归构造的$t$测试的两种常见测试程序的特性，并声称这两种程序通常是保守的。

    This paper considers the problem of inference in cluster randomized trials where treatment status is determined according to a "matched pairs'' design. Here, by a cluster randomized experiment, we mean one in which treatment is assigned at the level of the cluster; by a "matched pairs'' design we mean that a sample of clusters is paired according to baseline, cluster-level covariates and, within each pair, one cluster is selected at random for treatment. We study the large sample behavior of a weighted difference-in-means estimator and derive two distinct sets of results depending on if the matching procedure does or does not match on cluster size. We then propose a variance estimator which is consistent in either case. Combining these results establishes the asymptotic exactness of tests based on these estimators. Next, we consider the properties of two common testing procedures based on $t$-tests constructed from linear regressions, and argue that both are generally conservative in o
    

