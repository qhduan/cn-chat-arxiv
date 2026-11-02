# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Functional Tensor Trains for High-Order Greeks and Exposure Tails of Multi-Asset Structured Products](https://arxiv.org/abs/2610.42119) | 本文提出 hp 函数张量列车（hp-FTT）方法，通过求解反向定价递归并以秩一形式精确处理最差型障碍事件，实现了多资产结构化产品（如最差型自动赎回票据）的高阶（三阶）希腊字母与潜在未来风险敞口尾部分位数的无采样噪声、高效且具理论误差保证的计算。 |
| [^2] | [When is the deepest drawdown also the longest?](https://arxiv.org/abs/2610.41990) | 该研究揭示了回撤深度与持续时间的渐近关系：对均值为正的独立同分布增量随机游走，轻尾损失下最深与最长回撤区间几乎必然分离（最深回撤的相对时长收敛于显式常数ρ，高斯模型中为1/2），而重尾损失下一次大损失同时主导深度与时长，使两者重合的概率趋于1。 |
| [^3] | [A New GAUSS Software Component for Implementing Dynamic Tests of Contagion with an Application](https://arxiv.org/abs/2610.41740) | 本文开发了一个新的GAUSS软件组件，用于实施动态传染检验，能够在危机或大流行期间测量金融市场之间随时间变化的溢出效应，并采用杠杆自助法模拟技术生成可靠的临界值以确保推断的准确性。 |
| [^4] | [QBX: A Compiler for Well-Formed, Auditable Alpha Programs](https://arxiv.org/abs/2610.41689) | QBX 通过有限版本化的块注册表、port-DAG 语言和携带证据的中间表示，构建了一个良构、可审计且可复现的 Alpha 程序编译框架，能在编译期拒绝前视偏差并实现程序去重、增量评估与试验溯源。 |
| [^5] | [Expected shortfall estimation and capital coverage under overlapping scenarios](https://arxiv.org/abs/2610.41624) | 论文揭示了基于重叠多日收益构建的短样本估计预期损失时，因独立抽样假设失配会使估计值系统性偏低、削弱资本覆盖，并在一致性风险估计框架下系统比较了六种基于次序统计量的ES估计器的表现。 |
| [^6] | [Why delivered costs understate prices in an integrated global hydrogen market](https://arxiv.org/abs/2610.41447) | 该研究利用81区域全球氢能市场模型量化发现，交付成本会系统性低估氢能市场价格，2045年平均低估63欧元/兆瓦时（占单一买方成本的52%），且其中大部分源于市场相互作用导致的边际价格与平均成本之差，而非采购成本上升，因此氢能战略应将交付成本基准与市场价格区分开来。 |
| [^7] | [Optimal climate policy in incomplete electricity markets](https://arxiv.org/abs/2610.41103) | 本文发现电力市场的不完全性使可再生能源补贴和碳价格具有保险协同效益，从而提高了最优补贴水平和碳价格。 |
| [^8] | [Neural Network Methods in the 2BSDE Formalism for Solving the High-Dimensional Black--Scholes Equation with Transaction Costs](https://arxiv.org/abs/2610.41019) | 本文提出用单个ResNet神经网络求解带Leland交易成本的高维Black-Scholes方程对应的2BSDE系统，同时学习PDE的解和曲率矩阵的自由度，在D=10和100维下对几何篮子上的衍生品定价均实现了小于0.1的时间平均RMSE。 |
| [^9] | [Smart charging without rebound peaks: Wholesale price feedback limits distribution grid peaks from price-responsive electric vehicles](https://arxiv.org/abs/2610.34133) | 本文指出，传统研究将批发价格视为外生变量从而夸大了电动汽车“回弹峰值”问题，在考虑批发价格与充电负荷之间的均衡反馈后，配电网同步峰值增幅可从260%大幅降至5%。 |
| [^10] | [Tracking High-Frequency Trading Activity in China and Its Effects on Market Quality](https://arxiv.org/abs/2610.33400) | 本文提出以主动精确匹配订单作为高频交易活动的新型订单级代理指标，并利用深交所开盘时订单传播延迟的跨通道差异提供因果证据，研究高频交易对市场质量的影响。 |
| [^11] | [Admissible Portfolio Optimization: Information Constraints, Conditional Efficient Frontiers, and the Price of Causal Identification](https://arxiv.org/abs/2610.00147) | 该论文将均值-方差组合优化的条件信息本身作为决策变量，施加事先声明的可容许性约束（可获得性、保持无套利的信息流扩张、统计可分性与跨状态不变性），从而在源头排除前瞻偏差与共同因子误判，并证明可容许信息价值定理及“按决策损失选择信息会引入前瞻信息”的失效条件。 |
| [^12] | [Climate aware lending allocation under NGFS scenarios - A Monte Carlo approach](https://arxiv.org/abs/2610.00051) | 本文提出一个将NGFS短期气候情景与蒙特卡洛模拟相结合的结构化信用风险框架，把气候转型与物理风险转化为借款人违约概率的变化，并通过约束优化为气候风险下的贷款配置提供决策依据。 |
| [^13] | [Joint Exclusivity](https://arxiv.org/abs/2604.17490) | 本文提出了一种新的“联合互斥性”概念，允许非负随机向量中所有分量同时为正被禁止，但保留灵活的边际分布，并给出了存在性准则及构造方法。 |
| [^14] | [A Controlled Comparison of Deep Learning Architectures for Multi-Horizon Financial Forecasting: Evidence from 918 Experiments](https://arxiv.org/abs/2603.16886) | 本研究构建了一个包含918个受控实验的五阶段基准框架，在控制超参数、随机初始化、预测跨度和资产特性等因素的前提下，系统比较了Transformer、MLP、卷积和循环网络等九种深度学习架构在12种金融资产、4小时和24小时两种预测跨度下的表现。 |
| [^15] | [A Nested Factor Model for Equity Markets: Reconciling Multifractal Stock Returns and Rough Index Volatilities](https://arxiv.org/abs/2505.02678) | 本文通过嵌套因子模型，用粗糙的主波动率模式和超粗糙的多分形残差，成功解释了股票指数Hurst指数大于个股的反直觉现象。 |
| [^16] | [Matching-LLM: Historically Anchored Synthetic Prediction of Public Opinion and Elections](https://arxiv.org/abs/2411.01582) | Matching-LLM框架通过将大语言模型生成的响应与历史人类行为数据加权融合，在调查响应重构和2024年美国总统选举预测中表现出色，预测共和党获得310张选举人票（实际为312张），准确性优于仅依赖单一数据源的方法。 |

# 详细

[^1]: 面向多资产结构化产品高阶希腊字母与风险敞口尾部的函数张量列车

    Functional Tensor Trains for High-Order Greeks and Exposure Tails of Multi-Asset Structured Products

    [https://arxiv.org/abs/2610.42119](https://arxiv.org/abs/2610.42119)

    本文提出 hp 函数张量列车（hp-FTT）方法，通过求解反向定价递归并以秩一形式精确处理最差型障碍事件，实现了多资产结构化产品（如最差型自动赎回票据）的高阶（三阶）希腊字母与潜在未来风险敞口尾部分位数的无采样噪声、高效且具理论误差保证的计算。

    

    arXiv:2610.42119v1 公告类型： 新论文 摘要： 最差型自动赎回结构化票据的风险管理需要高达三阶的价格敏感度（希腊字母）以及潜在未来风险敞口，即票据未来价值的尾部分位数。在蒙特卡洛方法中，每增加一阶求导都会使采样噪声乘以一个随观察频率增长而增大的因子，而嵌套模拟则会将其内层路径的噪声转化为分位数估计的偏差。我们转而求解反向定价递归，将延续价值存储为 hp 函数张量列车（hp-FTT），其分段多项式核心在障碍位处进行切分，从而使最差型障碍事件能够以秩一的形式被精确处理。对于高斯转移，我们证明了达到误差 ε 所需的多项式阶数仅按 √(log(1/ε)) 的速度增长。一次求解即可通过将似然比权重与所存储的价值进行积分，在任意现货价格下得到所有希腊字母且无采样噪声，并且它所存储的未来日期的价值……

    arXiv:2610.42119v1 Announce Type: new  Abstract: Risk management of worst-of autocallables needs price sensitivities (Greeks) up to third order and the potential future exposure, a tail quantile of the note's future value. In Monte Carlo, each order of differentiation multiplies the sampling noise by a factor that grows with the observation frequency, and nested simulation turns the noise of its inner paths into a bias of the quantile. We instead solve the backward pricing recursion, storing the continuation value as an $hp$ functional tensor train (hp-FTT) whose piecewise-polynomial cores are split at the barriers, so that worst-of barrier events act exactly, with rank one. For Gaussian transitions we show that the polynomial degree sufficient for an error $\varepsilon$ grows only as $\sqrt{\log(1/\varepsilon)}$. One solve then gives every Greek at any spot without sampling noise, by integrating the likelihood-ratio weight against the stored value, and the values it stores at future d
    
[^2]: 最深的回撤何时也是最长的回撤？

    When is the deepest drawdown also the longest?

    [https://arxiv.org/abs/2610.41990](https://arxiv.org/abs/2610.41990)

    该研究揭示了回撤深度与持续时间的渐近关系：对均值为正的独立同分布增量随机游走，轻尾损失下最深与最长回撤区间几乎必然分离（最深回撤的相对时长收敛于显式常数ρ，高斯模型中为1/2），而重尾损失下一次大损失同时主导深度与时长，使两者重合的概率趋于1。

    

    最深的回撤是否需要最长的时间来恢复？对于具有独立同分布增量且均值为正且有限的随机游走，我们发现：在轻尾损失下，最深与最长的回撤区间最终几乎必然不同；而在重尾损失下，两者重合的概率趋于1。更精确地说，当损失具有有限指数矩时，最深回撤区间的持续时间相对于 n 个区间中最长者，依概率收敛到一个显式常数 ρ∈(0,1)，而其按持续时间降序排列的排名为 n^(1-ρ+o_P(1))。这些极限对所有固定数目的最深区间均成立；在高斯模型中，ρ=1/2，与漂移和方差无关。这种分离反映了“达到较大深度”与“维持长时间区间”所需指数代价的不同。对于正则变化的重尾损失，一次共同的大损失同时主导深度与持续时间：持续时间之比几乎必然趋于1，而对于（后文截断）……

    arXiv:2610.41990v1 Announce Type: cross  Abstract: Does the deepest drawdown take the longest to recover? For random walks with iid increments of positive finite mean, we find that the deepest and longest episodes are eventually distinct almost surely under light-tailed losses, but coincide with probability tending to one under heavy-tailed losses. More precisely, when losses have a finite exponential moment, the deepest episode's duration, relative to the longest among $n$ episodes, converges in probability to an explicit $\rho\in(0,1)$, while its descending duration rank is $n^{1-\rho+o_{\mathbb{P}}(1)}$. These limits hold for every fixed number of leading depths; in the Gaussian model, $\rho=1/2$, independently of drift and variance. The separation reflects the different exponential costs of reaching a large depth and sustaining a long episode. For regularly varying losses, a common large loss governs both depth and duration: the duration ratio tends to one almost surely, and for ev
    
[^3]: 一个用于实施传染动态检验的新型GAUSS软件组件及其应用

    A New GAUSS Software Component for Implementing Dynamic Tests of Contagion with an Application

    [https://arxiv.org/abs/2610.41740](https://arxiv.org/abs/2610.41740)

    本文开发了一个新的GAUSS软件组件，用于实施动态传染检验，能够在危机或大流行期间测量金融市场之间随时间变化的溢出效应，并采用杠杆自助法模拟技术生成可靠的临界值以确保推断的准确性。

    

    本文的主要目标是介绍作者开发的一个新的GAUSS软件组件，用于实施传染的动态检验。这些检验旨在提供检验金融危机期间或大流行疫情期间（如我们最近经历的那种情况）金融市场之间溢出效应的可能性。与具有固定潜在变化日期的标准传染检验不同，本文实施的检验是动态的，即在整个时间段内测量潜在的传染效应。为了在实施检验时生成可靠的临界值，采用了杠杆自助法模拟技术，以确保所进行的推断是准确的。由于该方法较新，需要创建软件组件来进行相关检验，本文即完成了这一工作。该模块非常易于实现……

    arXiv:2610.41740v1 Announce Type: new  Abstract: The main goal of this article is to present a new software component created by the authors in GAUSS for implementing dynamic tests for contagion. These tests are aimed at providing the possibilities of testing for spiller-over effects between financial markets during a crisis or during a pandemic situation such as the one that we experienced recently. Unlike standard tests for contagion that have fixed dates for a potential shift, the tests implemented here are dynamic in the sense that the potential contagion effect is measured across the time span. The leverage bootstrap simulation technique is utilized in order to generate critical values that are reliable when the tests are implemented in order to make sure that the conducted inference is accurate. Since the approach is new, software component needs to be created for conducting the underlying tests, which is accomplished via this article. The module is very easy to implement by the 
    
[^4]: QBX：一个面向良构、可审计 Alpha 程序的编译器

    QBX: A Compiler for Well-Formed, Auditable Alpha Programs

    [https://arxiv.org/abs/2610.41689](https://arxiv.org/abs/2610.41689)

    QBX 通过有限版本化的块注册表、port-DAG 语言和携带证据的中间表示，构建了一个良构、可审计且可复现的 Alpha 程序编译框架，能在编译期拒绝前视偏差并实现程序去重、增量评估与试验溯源。

    

    公式化 Alpha 挖掘系统在候选因子生成方式上日益多样化，然而每个系统通常将自身的表示、评估器和选择器耦合为定制的流水线，其结果难以复现或比较。我们提出 QBX，一个定义了良构、可审计 Alpha 程序空间的编译器。一个有限的、带版本管理的块注册表和整体策略级的 port-DAG 语言定义了候选空间；编译器要么以结构化诊断信息拒绝提案，要么输出携带证据的中间表示（IR），其中包含类型信息、时间可用性效应、可执行链接以及 Merkle 风格的内容标识。编译器封闭的操作将搜索限制在被许可的空间内，节点标识支持增量评估，而上下文绑定的根标识管理程序去重与试验溯源。可用性效应系统能够拒绝契约可见的前视偏差，同时允许合法的时点预测。在中国A股……（原文摘要在此处截断）

    arXiv:2610.41689v1 Announce Type: cross  Abstract: Formulaic alpha miners increasingly differ in how they generate candidates, yet each system typically couples its own representation, evaluator, and selector into a bespoke pipeline whose results are difficult to reproduce or compare. We present QBX, a compiler that defines a well-formed, auditable space of alpha programs. A finite, versioned block registry and whole-strategy port-DAG language define the candidate space; the compiler either rejects a proposal with structured diagnostics or emits an evidence-carrying IR with types, time-availability effects, executable linkage, and Merkle-style content identities. Compiler-closed actions keep search inside the admitted space, node identities support incremental evaluation, and context-bound root identities govern program deduplication and trial provenance. The availability-effect system rejects contract-visible look-ahead while admitting legal point-in-time forecasts.   On Chinese A-sha
    
[^5]: 重叠情景下的预期损失估计与资本覆盖

    Expected shortfall estimation and capital coverage under overlapping scenarios

    [https://arxiv.org/abs/2610.41624](https://arxiv.org/abs/2610.41624)

    论文揭示了基于重叠多日收益构建的短样本估计预期损失时，因独立抽样假设失配会使估计值系统性偏低、削弱资本覆盖，并在一致性风险估计框架下系统比较了六种基于次序统计量的ES估计器的表现。

    

    预期损失通常是基于由重叠多日收益率构建的损益情景短样本进行估计的，而应用于此类数据的非参数估计器却是在独立抽样假设下设计的。我们研究了这种不匹配如何影响估计精度和资本覆盖，以及它如何与底层损益序列中的序列依赖性和厚尾特征相互作用。在一致性风险估计框架内，我们基于拟合度、偏差和资本覆盖指标比较了六种基于次序统计量的预期损失估计器。遵循交易账簿基本审查的要求，我们以一年期估计窗口和2.5%置信水平的预期损失作为基准设置。我们将模拟与涵盖十二个风险因子和九十九个股票投资组合的市场数据相结合。研究表明，重叠抽样不仅仅是增加噪声，还会使预期损失估计系统性地下移，从而削弱资本覆盖。

    arXiv:2610.41624v1 Announce Type: new  Abstract: Expected shortfall (ES) is routinely estimated from a short sample of profit-and-loss (P&L) scenarios built from overlapping multi-day returns, while the non-parametric estimators applied to such data are designed under independent sampling. We study how this mismatch affects estimation accuracy and capital coverage, and how it interacts with serial dependence and heavy tails in the underlying P&L. Working within the coherent risk estimation framework, we compare six order-statistic-based ES estimators on fit, bias, and capital-coverage metrics. Following the Fundamental Review of the Trading Book (FRTB), we use a one-year estimation window and ES at the $2.5\%$ confidence level as a benchmark setup. We combine simulations with market data on twelve risk factors and ninety-nine equity portfolios. We show that overlapping sampling does not merely add noise but shifts ES estimates systematically downward, thereby weakening capital coverage
    
[^6]: 为什么在一体化全球氢能市场中交付成本会低估价格

    Why delivered costs understate prices in an integrated global hydrogen market

    [https://arxiv.org/abs/2610.41447](https://arxiv.org/abs/2610.41447)

    该研究利用81区域全球氢能市场模型量化发现，交付成本会系统性低估氢能市场价格，2045年平均低估63欧元/兆瓦时（占单一买方成本的52%），且其中大部分源于市场相互作用导致的边际价格与平均成本之差，而非采购成本上升，因此氢能战略应将交付成本基准与市场价格区分开来。

    

    交付成本估算常被用作氢能进口价格的代理指标。然而，它衡量的是平均供应链成本，而边际价格衡量的是新增一单位产品的系统价值。市场相互作用可能使两者产生偏离，从而导致长期价格假设出现偏差。我们利用一个涵盖81个区域的全球氢能市场模型来量化这一差距，比较各区域在仅有自身需求活跃时与所有区域需求均活跃时的成本与价格。在2045年，78个区域的价格-成本差距为正值，按需求加权计算平均为63欧元/兆瓦时（占单一买方成本的52%）。其中仅12欧元/兆瓦时反映了竞争性需求导致的平均采购成本上升，其余51欧元/兆瓦时则是全球市场价格与交付组合平均成本之间的差额。在若干区域，批发价格上涨的同时平均交付成本却在下降。氢能战略应将交付成本基准与市场价格区分开来。

    arXiv:2610.41447v1 Announce Type: cross  Abstract: Delivered-cost estimates are often used as proxies for hydrogen import prices. Yet they measure average supply-chain costs, whereas marginal prices measure the system value of an additional unit. Market interactions can make the two diverge, potentially biasing long-term price assumptions. We quantify this gap using an 81-region global hydrogen-market model, comparing each region's costs and prices when only its demand is active and when all regions' demands are active. The 2045 price-cost gap is positive in 78 regions and averages 63 EUR/MWh (52% of the single-buyer cost) on a demand-weighted basis. Only 12 EUR/MWh reflects higher average sourcing costs due to competing demand. The remaining 51 EUR/MWh is the difference between the global-market price and the delivered portfolio's average cost. In several regions, wholesale prices rise while average delivered costs fall. Hydrogen strategies should distinguish delivered-cost benchmarks
    
[^7]: 不完全电力市场中的最优气候政策

    Optimal climate policy in incomplete electricity markets

    [https://arxiv.org/abs/2610.41103](https://arxiv.org/abs/2610.41103)

    本文发现电力市场的不完全性使可再生能源补贴和碳价格具有保险协同效益，从而提高了最优补贴水平和碳价格。

    

    电力市场中不完整的长期合约安排会使投资者和消费者面临低效率的风险水平。本文研究了市场不完全性如何影响气候政策的最优选择。我们构建了一个博弈论模型，在天然气价格和需求不确定且合约不完整的情形下，预测电力市场参与者对给定政策的反应，从而进行最优政策制定。在说明性实验中，我们观察到当市场不完整时，可再生能源补贴和碳价格会带来一种保险方面的协同效益。这是因为，相对于完全市场基准，市场不完全性会将投资从可再生能源转向天然气，并使消费者面临电力价格年际波动带来的未保险风险。通过激励可再生能源投资，补贴和碳价格间接抵消了这些扭曲。因此，市场不完全性提高了最优的补贴水平和碳价格。

    arXiv:2610.41103v1 Announce Type: cross  Abstract: Incomplete long-term contracting in electricity markets can expose investors and consumers to inefficient levels of risk. This paper investigates how incompleteness impacts the optimal choice of climate policy. We develop a game-theoretic model of optimal policy-making in anticipation of how electricity market agents react to a given policy when gas prices and demand are uncertain and contracting is incomplete. In illustrative experiments, we observe that renewable subsidies and carbon prices induce an insurance co-benefit when markets are incomplete. This is because, relative to a complete market benchmark, incompleteness shifts investment from renewables toward gas and exposes consumers to uninsured risk from inter-annual volatility in electricity prices. By incentivizing investment in renewables, subsidies and carbon prices indirectly counteract these distortions. Consequently, incompleteness increases the optimal subsidy and carbon
    
[^8]: 2BSDE框架下的神经网络方法：求解带交易成本的高维Black-Scholes方程

    Neural Network Methods in the 2BSDE Formalism for Solving the High-Dimensional Black--Scholes Equation with Transaction Costs

    [https://arxiv.org/abs/2610.41019](https://arxiv.org/abs/2610.41019)

    本文提出用单个ResNet神经网络求解带Leland交易成本的高维Black-Scholes方程对应的2BSDE系统，同时学习PDE的解和曲率矩阵的自由度，在D=10和100维下对几何篮子上的衍生品定价均实现了小于0.1的时间平均RMSE。

    

    针对采用Leland交易成本模型（该模型纳入了曲率符号）的$(D+1)$维Black-Scholes偏微分方程（PDE），本文建立了一个二阶倒向随机微分方程（2BSDE）系统。该2BSDE系统以“速度”（曲率的导数）作为输入。该系统通过单个ResNet神经网络求解，该网络同时学习PDE的解以及曲率矩阵中的自由度。该网络用于学习$D=10,100$情形下几何篮子上的看涨期权和蝶式价差等衍生品的价格，在两种情形下均实现了小于0.1的时间平均RMSE。

    arXiv:2610.41019v1 Announce Type: cross  Abstract: A 2BSDE system is written down for a $(D+1)$-dimensional Black--Scholes partial differential equation (PDE) with the Leland transaction cost model which incorporates the sign of the curvature. The 2BSDE system takes input from the ``speed'' (derivative of the curvature). This system is solved using a single ResNet neural network which learns both the solution to the PDE as well as the degrees of freedom in the curvature matrix. The network is used to learn the price of derivatives such as call options and butterfly spreads on geometric baskets for $D=10,100$ achieving a time-averaged RMSE of less than $0.1$ in each case.
    
[^9]: 无回弹峰值的智能充电：批发价格反馈限制了价格响应型电动汽车造成的配电网峰值

    Smart charging without rebound peaks: Wholesale price feedback limits distribution grid peaks from price-responsive electric vehicles

    [https://arxiv.org/abs/2610.34133](https://arxiv.org/abs/2610.34133)

    本文指出，传统研究将批发价格视为外生变量从而夸大了电动汽车“回弹峰值”问题，在考虑批发价格与充电负荷之间的均衡反馈后，配电网同步峰值增幅可从260%大幅降至5%。

    

    关于价格优化的电动汽车充电的研究一致发现，常见的价格信号会导致整个车队在最便宜的时段同步充电，从而在低压电网中产生新的“回弹”负荷峰值。几乎所有这些研究都将批发价格视为外生变量，即不受转移的电动汽车负荷的影响。本文认为这一假设夸大了问题的严重性：在低价时段充电会抬高这些时段的批发价格，在均衡状态下这将使充电分散到不同时段。我基于德国2025年数据校准的实时价格，模拟了价格优化的充电行为。即使对电力需求的增加进行调整，当地电网的同步进口峰值仍上升260%。在加入考虑价格-数量反馈的均衡批发价格模型后，这一增幅降至仅5%。在允许车网互动（V2G）的情况下，出口峰值分别上升115%和下降6%。

    arXiv:2610.34133v1 Announce Type: new  Abstract: Studies on price-optimized EV charging consistently find that the common price signal synchronizes charging of the entire fleet of cars in the cheapest hours and creates new "rebound" load peaks in low-voltage grids. Almost all of these studies treat wholesale prices as exogenous, i.e., unaffected by the shifting EV load. This paper argues that this assumption overstates the problem: charging in low-price hours raises the wholesale price in those hours, which spreads charging across hours in equilibrium. I simulate price-optimized charging under real-time prices calibrated to German 2025 data. Even when adjusting for the increase in electricity demand, the coincident import peak of the local grid rises by 260%. After adding a model of equilibrium wholesale prices that takes the price-quantity feedback into account, the increase is reduced to a mere 5%. Allowing for vehicle-to-grid, the export peak rises by 115% and falls by 6%, respectiv
    
[^10]: 追踪中国高频交易活动及其对市场质量的影响

    Tracking High-Frequency Trading Activity in China and Its Effects on Market Quality

    [https://arxiv.org/abs/2610.33400](https://arxiv.org/abs/2610.33400)

    本文提出以主动精确匹配订单作为高频交易活动的新型订单级代理指标，并利用深交所开盘时订单传播延迟的跨通道差异提供因果证据，研究高频交易对市场质量的影响。

    

    我们开发了一种基于精确下单的新型订单级高频交易（HFT）活动代理指标，并研究其对市场质量的影响。由于报价每秒更新多次，毫秒级的订单簿访问使高频交易者更倾向于提交主动精确匹配订单——即价格恰好等于当时对手方最优报价的限价订单。基于深圳证券交易所的数据，我们验证了该代理指标的有效性：主动精确匹配订单对报价更新的反应更快、撤销更迅速，在订单簿深度失衡有利且买卖价差较窄时集中出现，并能预测100毫秒至十分钟内的中间价收益。我们利用市场开盘时订单传播延迟的跨通道差异构建双重差分设计，结果表明更低的延迟会增加主动精确匹配订单的占比。这一变化提供了较为可信的因果证据，表明延迟降低所引发的高频交易提升了短周期……（摘要原文在此处截断）

    arXiv:2610.33400v1 Announce Type: new  Abstract: We develop a new order-level proxy for high-frequency trading (HFT) activity based on precise order placement and study its effects on market quality. With quotes updating multiple times per second, millisecond-level order-book access makes high-frequency traders more likely to submit active exact-match orders, defined as limit orders priced exactly at the prevailing best opposite-side quote. Using Shenzhen Stock Exchange data, we validate the proxy: active exact-match orders respond faster to quote updates, cancel more rapidly, concentrate when depth imbalance is favorable and spreads are narrow, and predict mid-quote returns from 100 milliseconds through ten minutes. A difference-in-differences design exploiting cross-channel variation in order dissemination latency at the market open shows that lower latency increases active exact-match shares. This variation yields plausibly causal evidence that latency-induced HFT raises short-horiz
    
[^11]: 可容许投资组合优化：信息约束、条件有效前沿与因果识别的代价

    Admissible Portfolio Optimization: Information Constraints, Conditional Efficient Frontiers, and the Price of Causal Identification

    [https://arxiv.org/abs/2610.00147](https://arxiv.org/abs/2610.00147)

    该论文将均值-方差组合优化的条件信息本身作为决策变量，施加事先声明的可容许性约束（可获得性、保持无套利的信息流扩张、统计可分性与跨状态不变性），从而在源头排除前瞻偏差与共同因子误判，并证明可容许信息价值定理及“按决策损失选择信息会引入前瞻信息”的失效条件。

    

    均值-方差投资组合选择将条件信息视为给定，仅对权重进行优化，因此关于该信息的两类错误会在不被察觉的情况下传入组合：使用决策时点尚不可获得的变量，以及将共同变动误当作特质变动。我们将条件信息本身变为一个决策变量，使其服从事先声明的硬性可容许性约束：可获得性、保持无套利的决策信息流扩张、统计可分性，以及（对于干预性论断而言）跨所声明状态（regime）的不变性。通过一个不含任何风险-收益数量的字典序可容许性排序来选出最优信息类，并在该信息类内部求解经典问题。我们证明了存在性、在重编码下的不变性，以及一个可容许信息价值定理，该定理的失效条件表明：基于决策损失的选择只要遇到前瞻信息就会将其纳入；两阶段解……

    arXiv:2610.00147v1 Announce Type: new  Abstract: Mean--variance portfolio choice takes the conditioning information as given and optimizes over weights, so two errors about that information pass into the portfolio unseen: using variables unavailable at the decision time and treating common variation as idiosyncratic. We make the conditioning information a decision variable subject to hard admissibility constraints declared in advance: availability, a no-arbitrage-preserving enlargement of the decision filtration, statistical separation and, for interventional claims, invariance across declared regimes. A lexicographic admissibility order in which no risk--return quantity enters selects an optimal information class, and the classical problem is solved inside it. We prove existence, invariance under recodings, and a value-of-admissible-information theorem whose failure conditions show that selection by decision loss admits look-ahead information whenever present; the two-stage solution a
    
[^12]: NGFS情景下气候感知的贷款配置——一种蒙特卡洛方法

    Climate aware lending allocation under NGFS scenarios - A Monte Carlo approach

    [https://arxiv.org/abs/2610.00051](https://arxiv.org/abs/2610.00051)

    本文提出一个将NGFS短期气候情景与蒙特卡洛模拟相结合的结构化信用风险框架，把气候转型与物理风险转化为借款人违约概率的变化，并通过约束优化为气候风险下的贷款配置提供决策依据。

    

    本文开发了一个框架，用于在3-5年的时间范围内评估暴露于气候风险的信贷组合中的贷款组合预算。该方法利用NGFS的短期情景，将转型风险和物理风险路径转化为债务人违约风险的变化。企业经营表现被建模于结构化信用风险框架之中，其中依赖于情景的气候冲击会调整企业价值动态的漂移项。当企业价值跌破负债阈值时即发生违约。联合违约通过蒙特卡洛方法进行模拟，其相关性通过主成分分析加以刻画。由此得到的违约概率估计为约束优化提供输入，用于研究各风险暴露之间的贷款配置。气候情景的选择会影响模型隐含的违约概率，以及债务人和各行业之间由此产生的风险-收益权衡。结果显示违约概率分布存在情景特定的差异……

    arXiv:2610.00051v1 Announce Type: new  Abstract: This paper develops a framework to assess loan portfolio budget in credit portfolios exposed to climate risk over a 3-5 year horizon. Using short-term scenarios from the NGFS, the approach translates transition and physical risk pathways into changes in obligor default risk. Firm performance is modeled within a structural credit risk framework where scenario-dependent climate shocks adjust the drift of firm value dynamics. Defaults occur when firm value falls below a liability threshold. Joint defaults are simulated using Monte Carlo methods with dependence captured via principal components. Resulting default-probability estimates inform a constrained optimization used to examine lending allocation across exposures. Climate scenario choice affects model-implied default probabilities and the resulting risk-return tradeoff across obligors and sectors. The results show scenario-specific differences in default-probability distributions, uppe
    
[^13]: 联合互斥性

    Joint Exclusivity

    [https://arxiv.org/abs/2604.17490](https://arxiv.org/abs/2604.17490)

    本文提出了一种新的“联合互斥性”概念，允许非负随机向量中所有分量同时为正被禁止，但保留灵活的边际分布，并给出了存在性准则及构造方法。

    

    arXiv:2604.17490v2 公告类型：替换-交叉 摘要：我们引入了“联合互斥性”（JE），这是非负随机向量互斥性的一种灵活扩展，它仅禁止所有$n$个分量同时为正。与互斥性不同，JE允许广泛的边际分布，同时保持尖锐的存在性准则。我们证明了具有指定边际分布$F_1,\ldots,F_n$的JE随机向量存在，当且仅当\[ \sum_{i=1}^n \overline{F}_i(0)\le n-1. \] 我们还提供了一种基于面的典型构造，每个面内可自由指定copula，并表明三元情况可以优雅地简化为一个单参数族。最后，我们通过Wang型扭曲和介于互斥性与JE之间的$m$-互斥性层次扩展了该框架。

    arXiv:2604.17490v2 Announce Type: replace-cross  Abstract: We introduce \emph{joint exclusivity} (JE), a flexible extension of mutual exclusivity for non-negative random vectors that prohibits only the simultaneous positivity of all $n$ components. Unlike mutual exclusivity, JE admits a broad class of marginals while retaining a sharp existence criterion. We prove that a JE random vector with prescribed marginals $F_1,\ldots,F_n$ exists if and only if \[ \sum_{i=1}^n \overline{F}_i(0)\le n-1. \] We also provide a canonical face-based construction with freely specified copulas within each face, and show that the trivariate case can elegantly be reduced to a one-parameter family. Finally, we extend the framework through Wang-type distortions and an $m$-exclusivity hierarchy interpolating between mutual exclusivity and JE.
    
[^14]: 深度学习架构在多时间跨度金融预测中的受控比较：来自918个实验的证据

    A Controlled Comparison of Deep Learning Architectures for Multi-Horizon Financial Forecasting: Evidence from 918 Experiments

    [https://arxiv.org/abs/2603.16886](https://arxiv.org/abs/2603.16886)

    本研究构建了一个包含918个受控实验的五阶段基准框架，在控制超参数、随机初始化、预测跨度和资产特性等因素的前提下，系统比较了Transformer、MLP、卷积和循环网络等九种深度学习架构在12种金融资产、4小时和24小时两种预测跨度下的表现。

    

    摘要：对用于金融预测的深度学习架构进行可靠比较，需要受控的评估方式，将架构本身的影响与超参数调优、随机初始化、预测时间跨度和资产特性等因素分离。我们提出了一个五阶段基准测试框架，采用等预算贝叶斯超参数优化、类别级配置冻结、三种子最终训练、留出指标聚合以及基于排名的统计验证。该基准评估了来自Transformer、MLP/线性、卷积和循环网络四大类别的九种架构，基于五个OHLCV特征进行直接多步预测，覆盖12种资产（四种加密货币、四种外汇、四种股指类工具），每种资产使用30,000个小时级窗口按时间顺序以70/15/15划分数据集，并在4小时和24小时两种预测跨度上进行测试。该基准测试共包含918次运行（270次优化试验，648次最终训练运行）。在24个资产-跨度组合点上，Modern

    arXiv:2603.16886v2 Announce Type: replace-cross  Abstract: Reliable comparison of deep learning architectures for financial forecasting requires controlled evaluation that separates architectural effects from hyperparameter tuning, random initialization, forecast horizon, and asset characteristics. We present a five-stage benchmark using equal-budget Bayesian hyperparameter optimization, category-level configuration freezing, three-seed final training, held-out metric aggregation, and rank-based statistical validation. Nine architectures from Transformer, MLP/linear, convolutional, and recurrent families are evaluated with direct multi-step prediction from five OHLCV features across 12 assets (four cryptocurrency, four foreign exchange, four equity-index instruments), using chronological 70/15/15 splits of 30,000 hourly windows per asset at 4- and 24-hour horizons. The benchmark includes 918 runs (270 optimization trials, 648 final training runs). Across 24 asset-horizon points, Modern
    
[^15]: 股票市场的嵌套因子模型：调和多分形股票收益与粗糙指数波动率

    A Nested Factor Model for Equity Markets: Reconciling Multifractal Stock Returns and Rough Index Volatilities

    [https://arxiv.org/abs/2505.02678](https://arxiv.org/abs/2505.02678)

    本文通过嵌套因子模型，用粗糙的主波动率模式和超粗糙的多分形残差，成功解释了股票指数Hurst指数大于个股的反直觉现象。

    

    嵌套因子模型由Chicheportiche等人提出，用于表示股票之间的非线性相关性。股票收益由标准因子模型解释，但因子和残差的（对数）波动率本身被分解为因子模式，其中有一个共同的占主导地位的波动率模式影响市场因子、行业因子以及残差。本文考虑单因子情况，其中唯一的占主导地位的对数波动率模式是粗糙的，Hurst指数H≈0.11，而对数波动率残差是“超粗糙”或“多分形”的，H≈0。我们证明，这种构造自然解释了Wu等人报告的一个令人惊讶的风格化事实：即股票指数的Hurst指数相对于个股的Hurst指数更大。我们提出了一种统计方法，从股票收益动态中估计Hurst因子指数。

    arXiv:2505.02678v4 Announce Type: replace  Abstract: The Nested factor model was introduced by Chicheportiche et al. to represent non-linear correlations between stocks. Stock returns are explained by a standard factor model, but the (log)-volatilities of factors and residuals are themselves decomposed into factor modes, with a common dominant volatility mode affecting both market and sector factors but also residuals. Here, we consider the case of a single factor where the only dominant log-volatility mode is rough, with a Hurst exponent $H \simeq 0.11$ and the log-volatility residuals are ''super-rough'' or ''multifractal'', with $H \simeq 0$. We demonstrate that such a construction naturally accounts for the somewhat surprising stylized fact reported by Wu et al. , where it has been observed that the Hurst exponents of stock indexes are large compared to those of individual stocks. We propose a statistical procedure to estimate the Hurst factor exponent from the stock returns dynami
    
[^16]: Matching-LLM：基于历史锚定的公众舆论与选举合成预测

    Matching-LLM: Historically Anchored Synthetic Prediction of Public Opinion and Elections

    [https://arxiv.org/abs/2411.01582](https://arxiv.org/abs/2411.01582)

    Matching-LLM框架通过将大语言模型生成的响应与历史人类行为数据加权融合，在调查响应重构和2024年美国总统选举预测中表现出色，预测共和党获得310张选举人票（实际为312张），准确性优于仅依赖单一数据源的方法。

    

    大语言模型（LLM）为预测公众舆论和选举行为提供了新的机会，但其合成响应可能无法完全反映人类态度及其异质性。本研究提出了Matching-LLM，这是一个将LLM生成的响应与历史人类行为数据相结合的加权框架。利用世界价值观调查、美国全国选举研究以及德国选举调查数据，我们评估了该方法在重构调查响应和预测选举结果方面的表现。结果表明，相较于仅使用LLM的方法，Matching-LLM通常能提高准确性，包括在12项未纳入校准的调查问题比较中有11项表现更优，尽管它并不能持续超越仅使用历史数据的方法。在2024年美国总统选举中，Matching-LLM预测共和党获得310张选举人票，与实际的312张相比，在选举人票准确性上优于两个单源基准方法。

    arXiv:2411.01582v3 Announce Type: replace  Abstract: Large language models (LLMs) offer new opportunities for predicting public opinion and electoral behavior, but their synthetic responses may not fully reflect human attitudes and heterogeneity. This study introduces Matching-LLM, a weighted framework that combines LLM-generated responses with historical human behavioral data. Using the World Values Survey, American National Election Studies, and German election survey data, we evaluate its performance in reconstructing survey responses and predicting election outcomes. Results show that Matching-LLM generally improves accuracy relative to LLM-only, including 11 of 12 comparisons on survey questions excluded from calibration, although it does not consistently outperform History-only. In the 2024 U.S. presidential election, Matching-LLM predicts 310 Republican electoral votes, compared with the actual 312, outperforming both single-source benchmarks in electoral-vote accuracy. A supple
    

