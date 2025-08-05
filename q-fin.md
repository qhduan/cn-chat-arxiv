# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Incentivizing Data Sharing for Energy Forecasting: Analytics Markets with Correlated Data.](http://arxiv.org/abs/2310.06000) | 该论文开发了一个考虑相关性的分析市场，通过采用Shapley值的归因策略来分配收入，促进了数据共享以提高能源预测的准确性。 |
| [^2] | [An analysis of least squares regression and neural networks approximation for the pricing of swing options.](http://arxiv.org/abs/2307.04510) | 该论文分析了最小二乘回归和神经网络逼近方法在摇摆期权定价中的应用，并证明了适当选择回归函数时，这两种方法都能够实现精确的定价。 |

# 详细

[^1]: 鼓励数据共享以进行能源预测：具有相关数据的分析市场

    Incentivizing Data Sharing for Energy Forecasting: Analytics Markets with Correlated Data. (arXiv:2310.06000v1 [econ.GN])

    [http://arxiv.org/abs/2310.06000](http://arxiv.org/abs/2310.06000)

    该论文开发了一个考虑相关性的分析市场，通过采用Shapley值的归因策略来分配收入，促进了数据共享以提高能源预测的准确性。

    

    准确地预测不确定的电力产量对于电力市场的社会福利具有益处，可以减少平衡资源的需求。将这种预测描述为一项分析任务，当前文献提出了以分析市场作为激励手段来改善精度的数据共享方法，例如利用时空相关性。挑战在于，当相关数据用作预测的输入特征时，重叠信息的价值在于收入分配方面使市场设计复杂化，因为这种价值在本质上是组合的。我们为风力预测应用开发了一个考虑相关性的分析市场。为了分配收入，我们采用了基于Shapley值的归因策略，将代理人的特征视为玩家，将他们的相互作用视为一个特征函数博弈。我们说明了描述这种博弈的多种选项，每个选项都有因果细微差别，影响着特征相关时的市场行为。

    Reliably forecasting uncertain power production is beneficial for the social welfare of electricity markets by reducing the need for balancing resources. Describing such forecasting as an analytics task, the current literature proposes analytics markets as an incentive for data sharing to improve accuracy, for instance by leveraging spatio-temporal correlations. The challenge is that, when used as input features for forecasting, correlated data complicates the market design with respect to the revenue allocation, as the value of overlapping information is inherently combinatorial. We develop a correlation-aware analytics market for a wind power forecasting application. To allocate revenue, we adopt a Shapley value-based attribution policy, framing the features of agents as players and their interactions as a characteristic function game. We illustrate that there are multiple options to describe such a game, each having causal nuances that influence market behavior when features are cor
    
[^2]: 最小二乘回归和神经网络逼近用于摇摆期权定价的分析

    An analysis of least squares regression and neural networks approximation for the pricing of swing options. (arXiv:2307.04510v1 [q-fin.MF])

    [http://arxiv.org/abs/2307.04510](http://arxiv.org/abs/2307.04510)

    该论文分析了最小二乘回归和神经网络逼近方法在摇摆期权定价中的应用，并证明了适当选择回归函数时，这两种方法都能够实现精确的定价。

    

    最小二乘回归最初是用于美式期权定价的，但现在已经扩展到了摇摆期权定价。摇摆期权的价格可以看作是一个反向动态规划原理的解，其中包含一个称为继续值的条件期望。使用最小二乘回归逼近继续值的方法涉及两个级别的逼近。首先，继续值被一个有限集合的$m$个平方可积函数（回归函数）生成的子空间的正交投影所替代，得到摇摆价值函数的第一个近似值$V^m$。在本文中，我们证明，通过选择合适的回归函数，$V^m$当$m \to + \infty$时收敛到摇摆实际价格$V$。当回归函数被神经网络替代时，也证明了类似的结果。对于这两种方法（最小二乘或神经网络），我们分析了第二级别的逼近。

    Least Squares regression was first introduced for the pricing of American-style options, but it has since been expanded to include swing options pricing. The swing options price may be viewed as a solution to a Backward Dynamic Programming Principle, which involves a conditional expectation known as the continuation value. The approximation of the continuation value using least squares regression involves two levels of approximation. First, the continuation value is replaced by an orthogonal projection over a subspace spanned by a finite set of $m$ squared-integrable functions (regression functions) yielding a first approximation $V^m$ of the swing value function. In this paper, we prove that, with well-chosen regression functions, $V^m$ converges to the swing actual price $V$ as $m \to + \infty$. A similar result is proved when the regression functions are replaced by neural networks. For both methods (least squares or neural networks), we analyze the second level of approximation inv
    

