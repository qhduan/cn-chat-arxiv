# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Road Graph Generator: Mapping roads at construction sites from GPS data](https://arxiv.org/abs/2402.09919) | 本研究提出了一种通过分析GPS轨迹来绘制建筑工地道路地图的方法，通过识别关键的交叉口并连接它们，生成道路图，为规划和任务分配提供支持。 |
| [^2] | [Avoiding Catastrophe in Continuous Spaces by Asking for Help](https://arxiv.org/abs/2402.08062) | 在连续空间中，通过寻求帮助来避免灾难。引入了一种上下文多臂赌博问题的变体，目标是最小化灾难发生的概率。提出了一种算法，在连续1D状态空间和相对简单的回报函数下，遗憾和向导师查询率都趋近于0。 |
| [^3] | [Beyond Scale: the Diversity Coefficient as a Data Quality Metric Demonstrates LLMs are Pre-trained on Formally Diverse Data.](http://arxiv.org/abs/2306.13840) | 本论文提出使用多样性系数作为LLM预训练数据质量的指标，研究表明公开可用的LLM数据集的多样性系数很高。 |
| [^4] | [Kernel Density Bayesian Inverse Reinforcement Learning.](http://arxiv.org/abs/2303.06827) | KD-BIRL是一种核密度贝叶斯逆强化学习方法，通过直接逼近似然函数来学习代理的奖励函数，克服了学习点估计的缺点，并适用于复杂和无限环境。 |

# 详细

[^1]: 道路图生成器：从GPS数据中生成建筑工地道路地图

    Road Graph Generator: Mapping roads at construction sites from GPS data

    [https://arxiv.org/abs/2402.09919](https://arxiv.org/abs/2402.09919)

    本研究提出了一种通过分析GPS轨迹来绘制建筑工地道路地图的方法，通过识别关键的交叉口并连接它们，生成道路图，为规划和任务分配提供支持。

    

    在本文中，我们提出了一种从GPS轨迹中推测道路以绘制建筑工地地图的方法。这项任务由于建筑机械的不规则和非标准运动模式与已建立道路上的 typcial 车辆交通显著不同，因此面临着独特的挑战。我们的方法首先识别道路网络中作为关键决策点的交叉口，然后连接它们以形成一个图，随后可以用于规划和任务分配。我们通过在挪威的一个实际建筑工地绘制道路来证明我们方法的有效性。

    arXiv:2402.09919v1 Announce Type: new  Abstract: We present a method for road inference from GPS trajectories to map construction sites. This task introduces a unique challenge due to the erratic and non-standard movement patterns of construction machinery, which diverge significantly from typical vehicular traffic on established roads. Our method first identifies intersections in the road network that serve as critical decision points, and later connects them with edges, producing a graph, which subsequently can be used for planning and task-allocation. We demonstrate the effectiveness of our approach by mapping roads at a real-life construction site in Norway.
    
[^2]: 避免连续空间中的灾难：通过寻求帮助

    Avoiding Catastrophe in Continuous Spaces by Asking for Help

    [https://arxiv.org/abs/2402.08062](https://arxiv.org/abs/2402.08062)

    在连续空间中，通过寻求帮助来避免灾难。引入了一种上下文多臂赌博问题的变体，目标是最小化灾难发生的概率。提出了一种算法，在连续1D状态空间和相对简单的回报函数下，遗憾和向导师查询率都趋近于0。

    

    大多数具有正式遗憾保证的强化学习算法假设所有错误都是可逆的，并依赖于尝试所有可能的选项。当一些错误是无法修复甚至是灾难性的时，这种方法会导致糟糕的结果。我们提出了一种上下文多臂赌博问题的变体，在这个问题中，目标是最小化发生灾难的概率。具体而言，我们假设每轮的回报代表了在该轮避免灾难的概率，并尝试最大化回报的乘积（总体避免灾难的概率）。为了给 agent 一些成功的机会，我们允许有限次向导师提问，并假设回报函数为 Lipschitz 连续的。我们提出了一种算法，当时间跨度增长时，它的遗憾和向导师查询率都趋近于 0，假设是一个连续的 1D 状态空间和相对"简单"的回报函数。我们还提供了一个匹配的下界：在没有简单性假设的情况下，任何算法要么不断查询异常的行为，要么每次查询完全相同的行为。

    Most reinforcement learning algorithms with formal regret guarantees assume all mistakes are reversible and rely on essentially trying all possible options. This approach leads to poor outcomes when some mistakes are irreparable or even catastrophic. We propose a variant of the contextual bandit problem where the goal is to minimize the chance of catastrophe. Specifically, we assume that the payoff each round represents the chance of avoiding catastrophe that round, and try to maximize the product of payoffs (the overall chance of avoiding catastrophe). To give the agent some chance of success, we allow a limited number of queries to a mentor and assume a Lipschitz continuous payoff function. We present an algorithm whose regret and rate of querying the mentor both approach 0 as the time horizon grows, assuming a continuous 1D state space and a relatively "simple" payoff function. We also provide a matching lower bound: without the simplicity assumption: any algorithm either constantly
    
[^3]: 超越规模：多样性系数作为数据质量指标证明了LLMs是在形式多样的数据上预先训练的

    Beyond Scale: the Diversity Coefficient as a Data Quality Metric Demonstrates LLMs are Pre-trained on Formally Diverse Data. (arXiv:2306.13840v1 [cs.CL])

    [http://arxiv.org/abs/2306.13840](http://arxiv.org/abs/2306.13840)

    本论文提出使用多样性系数作为LLM预训练数据质量的指标，研究表明公开可用的LLM数据集的多样性系数很高。

    

    当前，预先训练强大的大语言模型(LLMs)的趋势主要集中在模型和数据集规模的扩大。然而，预先训练数据的质量对于训练强大的LLMs来说是一个重要因素，但它是一个模糊的概念，尚未完全表征。因此，我们使用最近提出的Task2Vec多样性系数来基于数据质量的形式方面，超越规模本身。具体而言，我们测量公开可用的预先训练数据集的多样性系数，以证明它们的形式多样性高于理论的下限和上限。此外，为了建立对多样性系数的信心，我们进行可解释性实验，并发现该系数与多样性的直观属性相吻合，例如，随着潜在概念数量的增加，它增加。我们得出结论，多样性系数是可靠的，表明公开可用的LLM数据集的多样性系数很高，并推测它可以作为预训练LLMs模型的数据质量指标。

    Current trends to pre-train capable Large Language Models (LLMs) mostly focus on scaling of model and dataset size. However, the quality of pre-training data is an important factor for training powerful LLMs, yet it is a nebulous concept that has not been fully characterized. Therefore, we use the recently proposed Task2Vec diversity coefficient to ground and understand formal aspects of data quality, to go beyond scale alone. Specifically, we measure the diversity coefficient of publicly available pre-training datasets to demonstrate that their formal diversity is high when compared to theoretical lower and upper bounds. In addition, to build confidence in the diversity coefficient, we conduct interpretability experiments and find that the coefficient aligns with intuitive properties of diversity, e.g., it increases as the number of latent concepts increases. We conclude the diversity coefficient is reliable, show it's high for publicly available LLM datasets, and conjecture it can be
    
[^4]: 核密度贝叶斯逆强化学习

    Kernel Density Bayesian Inverse Reinforcement Learning. (arXiv:2303.06827v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.06827](http://arxiv.org/abs/2303.06827)

    KD-BIRL是一种核密度贝叶斯逆强化学习方法，通过直接逼近似然函数来学习代理的奖励函数，克服了学习点估计的缺点，并适用于复杂和无限环境。

    

    逆强化学习（IRL）是一种通过观察代理行为来推断其奖励函数的强大框架，但学习奖励函数的点估计可能会误导，因为可能有多个函数能够很好地描述代理的行为。贝叶斯逆强化学习采用贝叶斯方法模拟候选奖励函数的分布，克服了学习点估计的缺点。然而，一些贝叶斯逆强化学习算法使用Q值函数代替似然函数。由此得到的后验计算量大，理论保证少，并且Q值函数通常对似然函数的逼近效果较差。我们引入了核密度贝叶斯逆强化学习（KD-BIRL），该方法使用条件核密度估计直接逼近似然函数，提供了一个高效的框架，在经过改进的奖励函数参数化下，适用于具有复杂和无限的环境。

    Inverse reinforcement learning~(IRL) is a powerful framework to infer an agent's reward function by observing its behavior, but IRL algorithms that learn point estimates of the reward function can be misleading because there may be several functions that describe an agent's behavior equally well. A Bayesian approach to IRL models a distribution over candidate reward functions, alleviating the shortcomings of learning a point estimate. However, several Bayesian IRL algorithms use a $Q$-value function in place of the likelihood function. The resulting posterior is computationally intensive to calculate, has few theoretical guarantees, and the $Q$-value function is often a poor approximation for the likelihood. We introduce kernel density Bayesian IRL (KD-BIRL), which uses conditional kernel density estimation to directly approximate the likelihood, providing an efficient framework that, with a modified reward function parameterization, is applicable to environments with complex and infin
    

