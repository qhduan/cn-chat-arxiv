# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Macroeconomic Policies based on Microfoundations: A Stackelberg Mean Field Game Approach](https://arxiv.org/abs/2403.12093) | 本研究提出了基于Stackelberg Mean Field Game的方法，可以有效地学习宏观经济政策，并在模型预训练和无模型Stackelberg均场强化学习算法的基础上取得了实验结果表明其优越性。 |
| [^2] | [Iterative Estimation of Nonparametric Regressions with Continuous Endogenous Variables and Discrete Instruments](https://arxiv.org/abs/1905.07812) | 提出了一种简单的迭代程序来估计具有连续内生变量和离散工具的非参数回归模型，并展示了一些渐近性质。 |
| [^3] | [How do we measure trade elasticity for services?.](http://arxiv.org/abs/2401.08594) | 本研究通过汇率变动来识别服务贸易的弹性，解决了因内生性问题而无法找到工具变量的困境。通过应用于多种不同的有形商品，评估了该方法的性能。 |
| [^4] | [School Choice with Multiple Priorities.](http://arxiv.org/abs/2308.04780) | 本研究提出了一个具有多个优先级的学校选择模型，引入了一种名为M-fairness的公平性概念，并介绍了一种利用效率调整延迟接受算法的机制，该机制是学生最优M稳定的，改进群体最优M稳定的，并且对改进是有响应的。 |
| [^5] | [The Strong Maximum Circulation Algorithm: A New Method for Aggregating Preference Rankings.](http://arxiv.org/abs/2307.15702) | 强大的最大环算法提出了一种集成偏好排序的新方法，通过删除投票图中的最大环路，得出与投票结果一致的唯一排序结果。 |
| [^6] | [Non-diversified portfolios with subjective expected utility.](http://arxiv.org/abs/2304.08059) | 研究表明，即使风险厌恶的投资者通常采用的策略是多样化投资组合，分配所有资金到单一的资产或世界状态的极端投资组合也很常见。这种行为与风险厌恶的主观期望效用最大化相容。当有限个极端资产需求在此种信念下是合理化的时候，它们可以同时通过一些效用指数的方式合理化。 |
| [^7] | [Both invariant principles implied by Marx's law of value are necessary and sufficient to solve the transformation problem through Morishima's formalism.](http://arxiv.org/abs/2303.11471) | 通过Morishima的形式主义，我们证明了从马克思的不变原则出发解决转型问题的必要性和充分性。 |

# 详细

[^1]: 基于微观基础的宏观经济政策学习：一种斯塔克尔贝格均场博弈方法

    Learning Macroeconomic Policies based on Microfoundations: A Stackelberg Mean Field Game Approach

    [https://arxiv.org/abs/2403.12093](https://arxiv.org/abs/2403.12093)

    本研究提出了基于Stackelberg Mean Field Game的方法，可以有效地学习宏观经济政策，并在模型预训练和无模型Stackelberg均场强化学习算法的基础上取得了实验结果表明其优越性。

    

    有效的宏观经济政策在促进经济增长和社会稳定方面起着至关重要的作用。本文基于Stackelberg Mean Field Game（SMFG）模型，将最优宏观经济政策问题建模，其中政府作为政策制定的领导者，大规模家庭动态响应为追随者。这种建模方法捕捉了政府和大规模家庭之间的非对称动态博弈，并可以解释地评估基于微观基础的宏观经济政策效果，这是现有方法难以实现的。我们还提出了一种解决SMFG的方法，将真实数据进行预训练，并结合一种无模型的Stackelberg均场强化学习（SMFRL）算法，该算法可以独立于先前的环境知识和转变运行。我们的实验结果展示了SMFG方法在经济政策方面优于其他方法的优越性。

    arXiv:2403.12093v1 Announce Type: cross  Abstract: Effective macroeconomic policies play a crucial role in promoting economic growth and social stability. This paper models the optimal macroeconomic policy problem based on the \textit{Stackelberg Mean Field Game} (SMFG), where the government acts as the leader in policy-making, and large-scale households dynamically respond as followers. This modeling method captures the asymmetric dynamic game between the government and large-scale households, and interpretably evaluates the effects of macroeconomic policies based on microfoundations, which is difficult for existing methods to achieve. We also propose a solution for SMFGs, incorporating pre-training on real data and a model-free \textit{Stackelberg mean-field reinforcement learning }(SMFRL) algorithm, which operates independently of prior environmental knowledge and transitions. Our experimental results showcase the superiority of the SMFG method over other economic policies in terms 
    
[^2]: 连续内生变量和离散工具的非参数回归的迭代估计

    Iterative Estimation of Nonparametric Regressions with Continuous Endogenous Variables and Discrete Instruments

    [https://arxiv.org/abs/1905.07812](https://arxiv.org/abs/1905.07812)

    提出了一种简单的迭代程序来估计具有连续内生变量和离散工具的非参数回归模型，并展示了一些渐近性质。

    

    我们考虑了一个具有连续内生独立变量的非参数回归模型，当只有与误差项独立的离散工具可用时。虽然这个框架在应用研究中非常相关，但其实现很麻烦，因为回归函数成为了非线性积分方程的解。我们提出了一个简单的迭代过程来估计这样的模型，并展示了一些其渐近性质。在一个模拟实验中，我们讨论了在工具变量为二进制时其实现细节。我们总结了一个实证应用，其中我们研究了美国几个县的房价对污染的影响。

    arXiv:1905.07812v2 Announce Type: replace  Abstract: We consider a nonparametric regression model with continuous endogenous independent variables when only discrete instruments are available that are independent of the error term. While this framework is very relevant for applied research, its implementation is cumbersome, as the regression function becomes the solution to a nonlinear integral equation. We propose a simple iterative procedure to estimate such models and showcase some of its asymptotic properties. In a simulation experiment, we discuss the details of its implementation in the case when the instrumental variable is binary. We conclude with an empirical application in which we examine the effect of pollution on house prices in a short panel of U.S. counties.
    
[^3]: 如何测量服务贸易弹性？

    How do we measure trade elasticity for services?. (arXiv:2401.08594v1 [econ.GN])

    [http://arxiv.org/abs/2401.08594](http://arxiv.org/abs/2401.08594)

    本研究通过汇率变动来识别服务贸易的弹性，解决了因内生性问题而无法找到工具变量的困境。通过应用于多种不同的有形商品，评估了该方法的性能。

    

    本文试图通过汇率变动来识别出贸易弹性，以应用于服务贸易的情况，其中实际交易统计数据中的物理交易被掩盖。回归分析用于估计弹性，涉及到解释变量通过潜在的供应方程泄露到误差项中，导致了内生性问题，无法找到工具变量。我们的识别策略是利用归一化条件，使得供应参数能够被识别，以及需求和供应方程系统的约化形式方程。通过应用于几种不同的有形商品，我们评估了所提出的方法的性能，这些商品的基准贸易弹性可以通过利用其物理交易信息进行估计。

    This paper is about our attempt of identifying trade elasticities through the variations in the exchange rate, for possible applications to the case of services whose physical transactions are veiled in the trade statistics. The regression analysis to estimate the elasticity entails a situation where the explanatory variable is leaked into the error term through the latent supply equation, causing an endogeneity problem for which an instrumental variable cannot be found. Our identification strategy is to utilize the normalizing condition, which enables the supply parameter to be identified, along with the reduced-form equation of the system of demand and supply equations. We evaluate the performances of the method proposed by applying to several different tangible goods, whose benchmark trade elasticities are estimable by utilizing the information on their physical transactions.
    
[^4]: 具有多个优先级的学校选择模型

    School Choice with Multiple Priorities. (arXiv:2308.04780v1 [econ.TH])

    [http://arxiv.org/abs/2308.04780](http://arxiv.org/abs/2308.04780)

    本研究提出了一个具有多个优先级的学校选择模型，引入了一种名为M-fairness的公平性概念，并介绍了一种利用效率调整延迟接受算法的机制，该机制是学生最优M稳定的，改进群体最优M稳定的，并且对改进是有响应的。

    

    本研究考虑了一种模型，在这种模型中，学校可能对学生有多个优先级顺序，这些顺序可能相互矛盾。例如，在学校选择系统中，由于兄弟姐妹优先级和步行区域优先级并存，基于它们的优先级顺序可能存在冲突。在这种情况下，可能找不到满足所有优先级顺序的匹配。我们引入了一种名为M-fairness的新颖公平性概念来研究这样的市场。此外，我们重点研究了一种更具体的情况，即所有学校都有两个优先级顺序，并且对于某个学生群体，每所学校的一个优先级顺序是另一优先级顺序的改进。一个说明性例子是具有基于优先级的积极行动政策的学校选择匹配市场。我们引入了一种利用效率调整延迟接受算法的机制，并证明该机制是学生最优M稳定的，改进群体最优M稳定的，并且对改进是有响应的。

    This study considers a model where schools may have multiple priority orders on students, which may be inconsistent with each other. For example, in school choice systems, since the sibling priority and the walk zone priority coexist, the priority orders based on them would be conflicting. In that case, there may be no matching that respect to all priority orders. We introduce a novel fairness notion called M-fairness to examine such markets. Further, we focus on a more specific situation where all schools have two priority orders, and for a certain group of students, a priority order of each school is an improvement of the other priority order of the school. An illustrative example is the school choice matching market with a priority-based affirmative action policy. We introduce a mechanism that utilizes the efficiency adjusted deferred acceptance algorithm and show that the mechanism is student optimally M-stable, improved-group optimally M-stable and responsive to improvements.
    
[^5]: 强大的最大环算法：一种集成偏好排序的新方法

    The Strong Maximum Circulation Algorithm: A New Method for Aggregating Preference Rankings. (arXiv:2307.15702v1 [cs.SI])

    [http://arxiv.org/abs/2307.15702](http://arxiv.org/abs/2307.15702)

    强大的最大环算法提出了一种集成偏好排序的新方法，通过删除投票图中的最大环路，得出与投票结果一致的唯一排序结果。

    

    我们提出了一种基于优化的方法，用于在每个决策者或选民对一对选择进行偏好表达的情况下集成偏好。挑战在于在一些冲突的投票情况下，尽可能与投票结果一致地得出一个排序。只有不包含环路的投票集合才是非冲突的，并且可以在选择之间引发一个部分顺序。我们的方法是基于这样一个观察：构成一个环路的投票集合可以被视为平局。然后，方法是从投票图中删除环路的并集，并根据剩余部分确定集成偏好。我们引入了强大的最大环路，它由一组环路的并集形成，删除它可以保证在引发的部分顺序中获得唯一结果。此外，它还包含在消除任何最大环路后剩下的所有集成偏好。与之相反的是，wel

    We present a new optimization-based method for aggregating preferences in settings where each decision maker, or voter, expresses preferences over pairs of alternatives. The challenge is to come up with a ranking that agrees as much as possible with the votes cast in cases when some of the votes conflict. Only a collection of votes that contains no cycles is non-conflicting and can induce a partial order over alternatives. Our approach is motivated by the observation that a collection of votes that form a cycle can be treated as ties. The method is then to remove unions of cycles of votes, or circulations, from the vote graph and determine aggregate preferences from the remainder.  We introduce the strong maximum circulation which is formed by a union of cycles, the removal of which guarantees a unique outcome in terms of the induced partial order. Furthermore, it contains all the aggregate preferences remaining following the elimination of any maximum circulation. In contrast, the wel
    
[^6]: 非多样化投资组合与主观期望效用

    Non-diversified portfolios with subjective expected utility. (arXiv:2304.08059v1 [econ.TH])

    [http://arxiv.org/abs/2304.08059](http://arxiv.org/abs/2304.08059)

    研究表明，即使风险厌恶的投资者通常采用的策略是多样化投资组合，分配所有资金到单一的资产或世界状态的极端投资组合也很常见。这种行为与风险厌恶的主观期望效用最大化相容。当有限个极端资产需求在此种信念下是合理化的时候，它们可以同时通过一些效用指数的方式合理化。

    

    虽然投资组合多样化是风险厌恶的投资者通常采用的策略，但为了单一资产/世界状态分配所有资金的极端投资组合也很常见。这种资产需求行为与风险厌恶的主观期望效用最大化相容，假设将每个状态分配一个严格正的概率。我们表明，只要有限个极端资产需求在此种信念下是合理化的，它们就可以同时在同样信念下通过：(i)绝对风险厌恶的常数；递减的绝对风险厌恶/递增的相对风险厌恶(DARA/IRRA)；风险中性；所有财富水平下的追求风险的效用指数；(ii)于某个严格正的固定初期财富水平下明显不同的DARA/IRRA效用指数类；和(iii)在有界财富下递减的相对风险厌恶效用指数。我们还表明，在这种情况下，可观察数据允许给出尖锐的限制。

    Although portfolio diversification is the typical strategy followed by risk-averse investors, extreme portfolios that allocate all funds to a single asset/state of the world are common too. Such asset-demand behavior is compatible with risk-averse subjective expected utility maximization under beliefs that assign a strictly positive probability to every state. We show that whenever finitely many extreme asset demands are rationalizable in this way under such beliefs, they are simultaneously rationalizable under the same beliefs by: (i) constant absolute risk aversion; decreasing absolute risk aversion/increasing relative risk aversion (DARA/IRRA); risk-neutral; and ris-kseeking utility indices at all wealth levels; (ii) a distinct class of DARA/IRRA utility indices at some strictly positive fixed initial wealth; and (iii) decreasing relative risk aversion utility indices under bounded wealth. We also show that, in such situations, the observable data allow for sharp bounds to be given 
    
[^7]: 用于解决马克思价值法则下的转型问题的不变原理——从森岛正雄形式主义出发

    Both invariant principles implied by Marx's law of value are necessary and sufficient to solve the transformation problem through Morishima's formalism. (arXiv:2303.11471v1 [econ.TH])

    [http://arxiv.org/abs/2303.11471](http://arxiv.org/abs/2303.11471)

    通过Morishima的形式主义，我们证明了从马克思的不变原则出发解决转型问题的必要性和充分性。

    

    Michio Morishima的方法可以正确确定商品的单位价值，即它所含的商品价值（输入）和生产所需的劳动量之和。但是，只有当商品满足涉及整个经济体系各个产业部门之间互相关联的有偿的社会需求时，它们才能以市场生产价格售出。这一条件充分体现了马克思的基本等式，这些等式源于价值法则，并构成适用于整个经济体系的不变量。这些等式是确定市场生产价格必需的。我们证明这些等式还能够用于解决一个没有固定资本的简单再生产系统的转型问题，从Morishima的形式主义出发并回归到更接近马克思所用形式主义。

    The unit value of a commodity that Michio Morishima's method and its variations enable to determine correctly, is the sum of the value of the commodities it contains (inputs) and the quantity of labor required for its production. However, goods are sold at their market production price only when they meet a solvent social need that involves the entire economy with its interconnections between the different industrial sectors. This condition gives full meaning to Marx's fundamental equalities, which derive from the law of value and constitute invariants that apply to the economy as a whole. These equalities are necessary to determine market production prices. We demonstrate that they also enable to solve the transformation problem for a simple reproduction system without fixed capital by starting from Morishima's formalism and returning to a formalism closer to that used by Marx.
    

