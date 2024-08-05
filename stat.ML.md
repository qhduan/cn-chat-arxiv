# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DASA: Delay-Adaptive Multi-Agent Stochastic Approximation](https://arxiv.org/abs/2403.17247) | DASA算法是第一个收敛速度仅依赖于混合时间和平均延迟的算法，同时在马尔科夫采样下实现N倍的收敛加速。 |
| [^2] | [Mixed moving average field guided learning for spatio-temporal data.](http://arxiv.org/abs/2301.00736) | 本论文提出了一种理论引导机器学习方法，采用广义贝叶斯算法进行混合移动平均场引导的时空数据建模，可以进行因果未来预测。 |

# 详细

[^1]: DASA: 延迟自适应多智能体随机逼近

    DASA: Delay-Adaptive Multi-Agent Stochastic Approximation

    [https://arxiv.org/abs/2403.17247](https://arxiv.org/abs/2403.17247)

    DASA算法是第一个收敛速度仅依赖于混合时间和平均延迟的算法，同时在马尔科夫采样下实现N倍的收敛加速。

    

    我们考虑一种设置，其中$N$个智能体旨在通过并行操作并与中央服务器通信来加速一个常见的随机逼近（SA）问题。我们假定上行传输到服务器的传输受到异步和潜在无界时变延迟的影响。为了减轻延迟和落后者的影响，同时又能获得分布式计算的好处，我们提出了一种名为DASA的延迟自适应多智能体随机逼近算法。我们对DASA进行了有限时间分析，假设智能体的随机观测过程是独立马尔科夫链。与现有结果相比，DASA是第一个其收敛速度仅取决于混合时间$tmix$和平均延迟$\tau_{avg}$，同时在马尔科夫采样下实现N倍的收敛加速的算法。我们的工作对于各种SA应用是相关的。

    arXiv:2403.17247v1 Announce Type: new  Abstract: We consider a setting in which $N$ agents aim to speedup a common Stochastic Approximation (SA) problem by acting in parallel and communicating with a central server. We assume that the up-link transmissions to the server are subject to asynchronous and potentially unbounded time-varying delays. To mitigate the effect of delays and stragglers while reaping the benefits of distributed computation, we propose \texttt{DASA}, a Delay-Adaptive algorithm for multi-agent Stochastic Approximation. We provide a finite-time analysis of \texttt{DASA} assuming that the agents' stochastic observation processes are independent Markov chains. Significantly advancing existing results, \texttt{DASA} is the first algorithm whose convergence rate depends only on the mixing time $\tmix$ and on the average delay $\tau_{avg}$ while jointly achieving an $N$-fold convergence speedup under Markovian sampling. Our work is relevant for various SA applications, inc
    
[^2]: 混合移动平均场引导的时空数据学习

    Mixed moving average field guided learning for spatio-temporal data. (arXiv:2301.00736v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2301.00736](http://arxiv.org/abs/2301.00736)

    本论文提出了一种理论引导机器学习方法，采用广义贝叶斯算法进行混合移动平均场引导的时空数据建模，可以进行因果未来预测。

    

    受到混合移动平均场的影响，时空数据的建模是一个多功能的技巧。但是，它们的预测分布通常不可访问。在这个建模假设下，我们定义了一种新的理论引导机器学习方法，采用广义贝叶斯算法进行预测。我们采用Lipschitz预测器（例如线性模型或前馈神经网络），并通过最小化沿空间和时间维度串行相关的数据的新型PAC贝叶斯界限来确定一个随机估计值。进行因果未来预测是我们方法的一个亮点，因为它适用于具有短期和长期相关性的数据。最后，我们通过展示线性预测器和模拟STOU过程的时空数据的示例来展示学习方法的性能。

    Influenced mixed moving average fields are a versatile modeling class for spatio-temporal data. However, their predictive distribution is not generally accessible. Under this modeling assumption, we define a novel theory-guided machine learning approach that employs a generalized Bayesian algorithm to make predictions. We employ a Lipschitz predictor, for example, a linear model or a feed-forward neural network, and determine a randomized estimator by minimizing a novel PAC Bayesian bound for data serially correlated along a spatial and temporal dimension. Performing causal future predictions is a highlight of our methodology as its potential application to data with short and long-range dependence. We conclude by showing the performance of the learning methodology in an example with linear predictors and simulated spatio-temporal data from an STOU process.
    

