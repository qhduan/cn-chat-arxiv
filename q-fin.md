# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Anomalous diffusion and price impact in the fluid-limit of an order book.](http://arxiv.org/abs/2310.06079) | 本文扩展了一个数值模拟方法，用于研究订单簿中金融市场订单的异常扩散和价格影响。我们通过模拟连续到达、取消和扩散的订单以及信息冲击，研究了不同情况下的冲击响应和风格特点。结果表明，数值方法能生成波纹效应，并提出了在扩散动力学存在的情况下使用非均匀采样作为首选的模拟方法。 |
| [^2] | [Responses of Unemployment to Productivity Changes for a General Matching Technology.](http://arxiv.org/abs/2307.05843) | 这篇论文探讨了一般匹配技术下失业对生产力变化的响应。研究表明，失业对生产力变化的响应取决于可用于创造就业机会的资源，而不是匹配技术本身的类型。 |
| [^3] | [A Unified Framework for Fast Large-Scale Portfolio Optimization.](http://arxiv.org/abs/2303.12751) | 该论文提出了一个用于快速大规模组合优化的统一框架，并以协方差矩阵估计器的样本外组合表现为例，展示了其中正则化规范项对模型表现的重要性。 |
| [^4] | [Inventories, Demand Shocks Propagation and Amplification in Supply Chains.](http://arxiv.org/abs/2205.03862) | 本文通过研究供应链上的产业地位和存货对需求冲击传导效应的作用，发现冲击在上游产业中有放大效应，并找到了存货在解释产出弹性上的重要作用。 |

# 详细

[^1]: 订单簿流限根据非均匀采样和非线性扩散条件下的异常扩散和价格影响

    Anomalous diffusion and price impact in the fluid-limit of an order book. (arXiv:2310.06079v1 [q-fin.CP])

    [http://arxiv.org/abs/2310.06079](http://arxiv.org/abs/2310.06079)

    本文扩展了一个数值模拟方法，用于研究订单簿中金融市场订单的异常扩散和价格影响。我们通过模拟连续到达、取消和扩散的订单以及信息冲击，研究了不同情况下的冲击响应和风格特点。结果表明，数值方法能生成波纹效应，并提出了在扩散动力学存在的情况下使用非均匀采样作为首选的模拟方法。

    

    我们扩展了一个离散时间随机游走（DTRW）的数值模拟方法，用于模拟模拟订单簿中金融市场订单的异常扩散。在这里，我们使用Sibuya等待时间的随机游走来包括一个时间相关的随机强迫函数，该函数具有非均匀采样的订单簿事件之间的时间。这个模型模拟了订单簿的流限，模拟了订单的连续到达、取消和扩散，以及信息冲击的影响。我们研究了不同强迫函数和模型参数下经历异常扩散的订单的冲击响应和风格特点。具体地说，我们展示了闪电限价单和市价单的价格影响，并展示了数值方法如何生成价格影响中的波纹。我们使用三次样条插值生成平滑的价格影响曲线。这项工作推广了在扩散动力学存在的情况下使用非均匀采样作为首选的模拟方法。

    We extend a Discrete Time Random Walk (DTRW) numerical scheme to simulate the anomalous diffusion of financial market orders in a simulated order book. Here using random walks with Sibuya waiting times to include a time-dependent stochastic forcing function with non-uniformly sampled times between order book events in the setting of fractional diffusion. This models the fluid limit of an order book by modelling the continuous arrival, cancellation and diffusion of orders in the presence of information shocks. We study the impulse response and stylised facts of orders undergoing anomalous diffusion for different forcing functions and model parameters. Concretely, we demonstrate the price impact for flash limit-orders and market orders and show how the numerical method generate kinks in the price impact. We use cubic spline interpolation to generate smoothed price impact curves. The work promotes the use of non-uniform sampling in the presence of diffusive dynamics as the preferred simul
    
[^2]: 对于一般匹配技术的生产力变化的失业响应

    Responses of Unemployment to Productivity Changes for a General Matching Technology. (arXiv:2307.05843v1 [econ.GN])

    [http://arxiv.org/abs/2307.05843](http://arxiv.org/abs/2307.05843)

    这篇论文探讨了一般匹配技术下失业对生产力变化的响应。研究表明，失业对生产力变化的响应取决于可用于创造就业机会的资源，而不是匹配技术本身的类型。

    

    工人离职、寻找工作、接受工作并用工资支付消费。公司招聘工人来填补职位空缺，但搜索摩擦使得公司无法立即雇佣可用工人。失业持续存在。这些特征由Diamond-Mortensen-Pissarides建模框架描述。在这类模型中，失业对生产力变化的响应取决于可用于创造就业机会的资源。然而，这种特征是在匹配被Cobb-Douglas技术参数化时得出的。对于一个典型的DMP模型，我(1)证明只要初始职位空缺产生正的余额，就会存在唯一的稳态平衡;(2)对一般匹配技术的失业对生产力变化的响应进行了描述;(3)展示了一个不是Cobb-Douglas的匹配技术意味着失业对生产力变化的响应更大，这与可用于就业创造的资源无关。

    Workers separate from jobs, search for jobs, accept jobs, and fund consumption with their wages. Firms recruit workers to fill vacancies, but search frictions prevent firms from instantly hiring available workers. Unemployment persists. These features are described by the Diamond--Mortensen--Pissarides modeling framework. In this class of models, how unemployment responds to productivity changes depends on resources that can be allocated to job creation. Yet, this characterization has been made when matching is parameterized by a Cobb--Douglas technology. For a canonical DMP model, I (1) demonstrate that a unique steady-state equilibrium will exist as long as the initial vacancy yields a positive surplus; (2) characterize responses of unemployment to productivity changes for a general matching technology; and (3) show how a matching technology that is not Cobb--Douglas implies unemployment responds more to productivity changes, which is independent of resources available for job creati
    
[^3]: 快速大规模组合优化的统一框架

    A Unified Framework for Fast Large-Scale Portfolio Optimization. (arXiv:2303.12751v1 [q-fin.PM])

    [http://arxiv.org/abs/2303.12751](http://arxiv.org/abs/2303.12751)

    该论文提出了一个用于快速大规模组合优化的统一框架，并以协方差矩阵估计器的样本外组合表现为例，展示了其中正则化规范项对模型表现的重要性。

    

    我们开发了一个统一的框架，用于具有缩减和正则化的快速大规模组合优化，针对不同的目标，如最小方差、平均方差和最大夏普比率，以及组合权重的各种约束条件。对于所有优化问题，我们推导出相应的二次规划问题，并在开源Python库中实现它们。我们使用所提出的框架评估了流行的协方差矩阵估计器的样本外组合表现，例如样本协方差矩阵、线性和非线性缩减估计器以及仪器化主成分分析（IPCA）的协方差矩阵。我们使用了65年间平均上市公司达585家的美国市场的月度收益率，以及用于IPCA模型的94个月度特定公司特征。我们展示了组合规范项的正则化极大地改善了IPCA模型在组合优化中的表现，结果表现良好。

    We develop a unified framework for fast large-scale portfolio optimization with shrinkage and regularization for different objectives such as minimum variance, mean-variance, and maximum Sharpe ratio with various constraints on the portfolio weights. For all of the optimization problems, we derive the corresponding quadratic programming problems and implement them in an open-source Python library. We use the proposed framework to evaluate the out-of-sample portfolio performance of popular covariance matrix estimators such as sample covariance matrix, linear and nonlinear shrinkage estimators, and the covariance matrix from the instrumented principal component analysis (IPCA). We use 65 years of monthly returns from (on average) 585 largest companies in the US market, and 94 monthly firm-specific characteristics for the IPCA model. We show that the regularization of the portfolio norms greatly benefits the performance of the IPCA model in portfolio optimization, resulting in outperforma
    
[^4]: 存货、需求冲击传导和供应链中的放大效应

    Inventories, Demand Shocks Propagation and Amplification in Supply Chains. (arXiv:2205.03862v4 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2205.03862](http://arxiv.org/abs/2205.03862)

    本文通过研究供应链上的产业地位和存货对需求冲击传导效应的作用，发现冲击在上游产业中有放大效应，并找到了存货在解释产出弹性上的重要作用。

    

    本文研究了产业在供应链中的地位对决定最终需求冲击传导效应的作用。通过基于目标特定的最终需求冲击和目标份额的变化来设计转移份额，本文发现冲击在上游放大。定量上，上游行业对最终需求冲击的反应是终端商品生产者的三倍。为了组织简化形式的结果，我开发了一个包括存货的可操作生产网络模型，并研究了网络属性和存货的周期性如何相互作用来决定最终需求冲击是放大还是耗散。我通过直接估计模型预测的产出增长和需求冲击之间的关系来验证该机制，中介网络位置和存货。我发现存货在解释异质性产出弹性方面扮演了重要角色。最后，我使用模型定量研究了存货和网络属性如何塑造生产过程中的波动性。

    I study the role of industries' position in supply chains in shaping the transmission of final demand shocks. First, I use a shift-share design based on destination-specific final demand shocks and destination shares to show that shocks amplify upstream. Quantitatively, upstream industries respond to final demand shocks up to three times as much as final goods producers. To organize the reduced form results, I develop a tractable production network model with inventories and study how the properties of the network and the cyclicality of inventories interact to determine whether final demand shocks amplify or dissipate upstream. I test the mechanism by directly estimating the model-implied relationship between output growth and demand shocks, mediated by network position and inventories. I find evidence of the role of inventories in explaining heterogeneous output elasticities. Finally, I use the model to quantitatively study how inventories and network properties shape the volatility o
    

