# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Deep Learning Based Measure of Name Concentration Risk](https://arxiv.org/abs/2403.16525) | 提出了一种基于深度学习的方法，用于量化贷款组合中的姓名集中风险，通过重要性抽样的蒙特卡洛模拟训练神经网络，展示了其相比现有方法在评估小型和集中组合中的姓名集中风险方面的准确性和优越性能。 |
| [^2] | [Reinforcement Learning for Financial Index Tracking.](http://arxiv.org/abs/2308.02820) | 本论文提出了针对金融指数跟踪问题的第一个具有动态性的离散时间无穷期模型，它克服了现有模型的一些局限，可以精确计算交易成本，同时考虑了跟踪误差和交易成本之间的权衡，并能有效利用长时间段的数据。我们使用深度强化学习方法解决该模型，解决了由于数据限制导致的问题。 |

# 详细

[^1]: 基于深度学习的姓名集中风险度量方法

    Deep Learning Based Measure of Name Concentration Risk

    [https://arxiv.org/abs/2403.16525](https://arxiv.org/abs/2403.16525)

    提出了一种基于深度学习的方法，用于量化贷款组合中的姓名集中风险，通过重要性抽样的蒙特卡洛模拟训练神经网络，展示了其相比现有方法在评估小型和集中组合中的姓名集中风险方面的准确性和优越性能。

    

    我们提出了一种新的基于深度学习的方法，用于量化贷款组合中的姓名集中风险。我们的方法针对小型组合进行了定制，允许损失的精算定义和按市场价值核算定义。我们的神经网络的训练依赖于重要性抽样的蒙特卡洛模拟，我们明确为CreditRisk${+}$和基于评级的CreditMetrics模型制定了这一过程。基于模拟和真实数据的数值结果显示了我们新方法的准确性，以及与现有分析方法相比，在评估小型和集中组合中的姓名集中风险方面表现出的卓越性能。

    arXiv:2403.16525v1 Announce Type: new  Abstract: We propose a new deep learning approach for the quantification of name concentration risk in loan portfolios. Our approach is tailored for small portfolios and allows for both an actuarial as well as a mark-to-market definition of loss. The training of our neural network relies on Monte Carlo simulations with importance sampling which we explicitly formulate for the CreditRisk${+}$ and the ratings-based CreditMetrics model. Numerical results based on simulated as well as real data demonstrate the accuracy of our new approach and its superior performance compared to existing analytical methods for assessing name concentration risk in small and concentrated portfolios.
    
[^2]: 针对金融指数跟踪的强化学习

    Reinforcement Learning for Financial Index Tracking. (arXiv:2308.02820v1 [q-fin.PM])

    [http://arxiv.org/abs/2308.02820](http://arxiv.org/abs/2308.02820)

    本论文提出了针对金融指数跟踪问题的第一个具有动态性的离散时间无穷期模型，它克服了现有模型的一些局限，可以精确计算交易成本，同时考虑了跟踪误差和交易成本之间的权衡，并能有效利用长时间段的数据。我们使用深度强化学习方法解决该模型，解决了由于数据限制导致的问题。

    

    我们提出了第一个离散时间无穷期动态形式的金融指数跟踪问题，同时考虑到基于收益的跟踪误差和基于价值的跟踪误差。该模型克服了现有模型的局限性，包括不仅限于价格的市场信息变量的时间动态性，可以精确计算交易成本，考虑跟踪误差和交易成本之间的权衡，可以有效利用长时间段的数据等。该模型还引入了现金注入或提取的新的决策变量。我们提出了使用Banach不动点迭代求解投资组合再平衡方程的方法，可以准确计算实践中指定为交易量的非线性函数的交易成本。我们还提出了扩展深度强化学习（RL）方法来解决动态模型。我们的RL方法解决了由数据限制引起的问题。

    We propose the first discrete-time infinite-horizon dynamic formulation of the financial index tracking problem under both return-based tracking error and value-based tracking error. The formulation overcomes the limitations of existing models by incorporating the intertemporal dynamics of market information variables not limited to prices, allowing exact calculation of transaction costs, accounting for the tradeoff between overall tracking error and transaction costs, allowing effective use of data in a long time period, etc. The formulation also allows novel decision variables of cash injection or withdraw. We propose to solve the portfolio rebalancing equation using a Banach fixed point iteration, which allows to accurately calculate the transaction costs specified as nonlinear functions of trading volumes in practice. We propose an extension of deep reinforcement learning (RL) method to solve the dynamic formulation. Our RL method resolves the issue of data limitation resulting fro
    

