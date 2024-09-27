# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Enhancing Rolling Horizon Production Planning Through Stochastic Optimization Evaluated by Means of Simulation](https://arxiv.org/abs/2402.14506) | 本文通过将更新的客户需求整合到滚动视界规划周期中，使用场景-based的随机规划来解决有能力批量规模问题，在离散事件模拟-优化框架中不断调整生产计划，提升了生产计划的滚动视界。 |
| [^2] | [Model Averaging and Double Machine Learning.](http://arxiv.org/abs/2401.01645) | 本文介绍了一种将双机器学习和模型平均化相结合的方法，用于估计结构参数。研究表明，这种方法比起常见的基于单一学习器的替代方法更加鲁棒，适用于处理部分未知的函数形式。 |
| [^3] | [Optimal Scoring for Dynamic Information Acquisition.](http://arxiv.org/abs/2310.19147) | 这篇论文研究了一个委托人通过雇佣代理人以动态方式获取信息的问题，在条件允许的情况下，委托人无法通过一次性收集所有信息之后再从代理人那里获取单个报告来获得更好的结果，并且在较强的条件违反下，静态合同是次优的。与此同时，该论文还展示了在动态环境中，即使代理人对状态预测错误，最优合同也可能会给予代理人严格正的基本奖励。 |
| [^4] | [The Power of Simple Menus in Robust Selling Mechanisms.](http://arxiv.org/abs/2310.17392) | 本研究致力于寻找简单的销售机制，以有效对冲市场模糊性。我们发现，只有有限数量的价格随机化的销售机制已经能够获得无限选项的最优稳健机制所实现的显著效益。 |

# 详细

[^1]: 通过模拟评估的随机优化来增强滚动视界生产计划

    Enhancing Rolling Horizon Production Planning Through Stochastic Optimization Evaluated by Means of Simulation

    [https://arxiv.org/abs/2402.14506](https://arxiv.org/abs/2402.14506)

    本文通过将更新的客户需求整合到滚动视界规划周期中，使用场景-based的随机规划来解决有能力批量规模问题，在离散事件模拟-优化框架中不断调整生产计划，提升了生产计划的滚动视界。

    

    生产计划必须考虑到生产系统中的不确定性，源自需求预测的波动。因此，本文集中在将更新的客户需求整合到滚动视界规划周期中。我们使用基于场景的随机规划来解决滚动视界环境下的有能力批量规模问题。该环境通过离散事件模拟优化框架复制，其中对定期解决的优化问题，利用最新的需求信息不断调整生产计划。我们评估了随机优化方法，并将其性能与使用预期需求数据作为输入解决确定性批量规模模型以及标准物料需求计划（MRP）进行了比较。在模拟研究中，我们分析了与预测相关的三种不同客户行为，以及四个水平的需求波动。

    arXiv:2402.14506v1 Announce Type: new  Abstract: Production planning must account for uncertainty in a production system, arising from fluctuating demand forecasts. Therefore, this article focuses on the integration of updated customer demand into the rolling horizon planning cycle. We use scenario-based stochastic programming to solve capacitated lot sizing problems under stochastic demand in a rolling horizon environment. This environment is replicated using a discrete event simulation-optimization framework, where the optimization problem is periodically solved, leveraging the latest demand information to continually adjust the production plan. We evaluate the stochastic optimization approach and compare its performance to solving a deterministic lot sizing model, using expected demand figures as input, as well as to standard Material Requirements Planning (MRP). In the simulation study, we analyze three different customer behaviors related to forecasting, along with four levels of 
    
[^2]: 模型平均化和双机器学习

    Model Averaging and Double Machine Learning. (arXiv:2401.01645v1 [econ.EM])

    [http://arxiv.org/abs/2401.01645](http://arxiv.org/abs/2401.01645)

    本文介绍了一种将双机器学习和模型平均化相结合的方法，用于估计结构参数。研究表明，这种方法比起常见的基于单一学习器的替代方法更加鲁棒，适用于处理部分未知的函数形式。

    

    本文讨论了将双重/无偏机器学习（DDML）与stacking（一种模型平均化方法，用于结合多个候选学习器）相结合，用于估计结构参数。我们引入了两种新的DDML stacking方法：短stacking利用DDML的交叉拟合步骤大大减少了计算负担，而汇总stacking可以在交叉拟合的折叠上强制执行通用 stacking权重。通过经过校准的模拟研究和两个应用程序，即估计引用和工资中的性别差距，我们展示了DDML与stacking相比基于单个预选学习器的常见替代方法对于部分未知的函数形式更加鲁棒。我们提供了实现我们方案的Stata和R软件。

    This paper discusses pairing double/debiased machine learning (DDML) with stacking, a model averaging method for combining multiple candidate learners, to estimate structural parameters. We introduce two new stacking approaches for DDML: short-stacking exploits the cross-fitting step of DDML to substantially reduce the computational burden and pooled stacking enforces common stacking weights over cross-fitting folds. Using calibrated simulation studies and two applications estimating gender gaps in citations and wages, we show that DDML with stacking is more robust to partially unknown functional forms than common alternative approaches based on single pre-selected learners. We provide Stata and R software implementing our proposals.
    
[^3]: 动态信息获取的最佳评分

    Optimal Scoring for Dynamic Information Acquisition. (arXiv:2310.19147v1 [econ.TH])

    [http://arxiv.org/abs/2310.19147](http://arxiv.org/abs/2310.19147)

    这篇论文研究了一个委托人通过雇佣代理人以动态方式获取信息的问题，在条件允许的情况下，委托人无法通过一次性收集所有信息之后再从代理人那里获取单个报告来获得更好的结果，并且在较强的条件违反下，静态合同是次优的。与此同时，该论文还展示了在动态环境中，即使代理人对状态预测错误，最优合同也可能会给予代理人严格正的基本奖励。

    

    一个委托人试图通过雇佣一个代理人来使用泊松信息到达技术随着时间来获取关于二进制状态的信息。代理人会私下了解这个状态，而委托人无法观察到代理人的努力选择。委托人可以根据代理人的报告序列和实现的状态，以一个固定价值的奖品来奖励代理人。我们确定了一些条件，每个条件都确保委托人无法比在获取了所有信息后从代理人那里提取单个报告更好。我们还证明了在这些条件足够强烈的违反下，这样的静态合同是次优的。我们将我们的解决方案与代理人一次性获取所有信息的情况进行对比；值得注意的是，在动态环境中，即使代理人对状态的预测是错误的，最优合同可能会向代理人提供严格正的基本奖励。

    A principal seeks to learn about a binary state and can do so by enlisting an agent to acquire information over time using a Poisson information arrival technology. The agent learns about this state privately, and his effort choices are unobserved by the principal. The principal can reward the agent with a prize of fixed value as a function of the agent's sequence of reports and the realized state. We identify conditions that each individually ensure that the principal cannot do better than by eliciting a single report from the agent after all information has been acquired. We also show that such a static contract is suboptimal under sufficiently strong violations of these conditions. We contrast our solution to the case where the agent acquires information "all at once;" notably, the optimal contract in the dynamic environment may provide strictly positive base rewards to the agent even if his prediction about the state is incorrect.
    
[^4]: 简单菜单在稳健销售机制中的力量

    The Power of Simple Menus in Robust Selling Mechanisms. (arXiv:2310.17392v1 [econ.TH])

    [http://arxiv.org/abs/2310.17392](http://arxiv.org/abs/2310.17392)

    本研究致力于寻找简单的销售机制，以有效对冲市场模糊性。我们发现，只有有限数量的价格随机化的销售机制已经能够获得无限选项的最优稳健机制所实现的显著效益。

    

    我们研究了一个稳健销售问题，其中卖方试图将一个物品卖给一个买方，但对买方的估值分布存在不确定性。现有文献表明，稳健机制设计比稳健确定性定价提供了更强的理论保证。同时，稳健机制设计的卓越性能以卖方提供具有无限选项的菜单，每个选项与抽奖和买方选择的付款方式相配。鉴于此，我们的研究的主要重点是寻找可以有效对冲市场模糊性的简单销售机制。我们表明，一个具有小菜单大小（或有限数量的价格随机化）的销售机制已经能够获得无限选项的最优稳健机制所实现的显著效益。特别是，我们发展了一个通用框架来研究稳健销售机制问题。

    We study a robust selling problem where a seller attempts to sell one item to a buyer but is uncertain about the buyer's valuation distribution. Existing literature indicates that robust mechanism design provides a stronger theoretical guarantee than robust deterministic pricing. Meanwhile, the superior performance of robust mechanism design comes at the expense of implementation complexity given that the seller offers a menu with an infinite number of options, each coupled with a lottery and a payment for the buyer's selection. In view of this, the primary focus of our research is to find simple selling mechanisms that can effectively hedge against market ambiguity. We show that a selling mechanism with a small menu size (or limited randomization across a finite number of prices) is already capable of deriving significant benefits achieved by the optimal robust mechanism with infinite options. In particular, we develop a general framework to study the robust selling mechanism problem 
    

