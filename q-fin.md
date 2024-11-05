# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On Merton's Optimal Portfolio Problem under Sporadic Bankruptcy](https://arxiv.org/abs/2403.15923) | 本文研究了在股票市场遵循几何布朗运动并且遵循恒定利率连续复利的无风险资产的情况下，考虑股票可能在一定随机时间内破产的梅顿最优投资组合问题。通过新型哈密顿-雅可比-贝尔曼方程，将问题形式化为受控马尔可夫扩散，并且严格推导了一个新版本的梅顿比率，最终通过验证定理验证了该模型在现实世界中的适用性。 |
| [^2] | [The contribution of US broadband infrastructure subsidy and investment programs to GDP using input-output modeling](https://arxiv.org/abs/2311.02431) | 这篇论文研究了美国宽带基础设施补贴和投资计划对国内生产总值的贡献。通过投入产出模型探索了宽带支出对宏观经济的影响。 |
| [^3] | [Neural Hawkes: Non-Parametric Estimation in High Dimension and Causality Analysis in Cryptocurrency Markets.](http://arxiv.org/abs/2401.09361) | 这篇论文提出了一种名为神经霍克斯的方法，它能够在高维度情况下进行非参数估计，并用于加密货币市场的因果分析。该方法通过解决第二类Fredholm积分方程来推断标记霍克斯核，并使用物理信息神经网络进行数值计算。该方法在模拟数据上得到了广泛的验证，并应用于分析加密货币市场的微观结构和因果关系。 |

# 详细

[^1]: 关于间歇性破产下梅顿最优投资组合问题

    On Merton's Optimal Portfolio Problem under Sporadic Bankruptcy

    [https://arxiv.org/abs/2403.15923](https://arxiv.org/abs/2403.15923)

    本文研究了在股票市场遵循几何布朗运动并且遵循恒定利率连续复利的无风险资产的情况下，考虑股票可能在一定随机时间内破产的梅顿最优投资组合问题。通过新型哈密顿-雅可比-贝尔曼方程，将问题形式化为受控马尔可夫扩散，并且严格推导了一个新版本的梅顿比率，最终通过验证定理验证了该模型在现实世界中的适用性。

    

    考虑一个遵循几何布朗运动的股票市场和以恒定利率连续复利的无风险资产。假设股票可以破产，即在一些外生随机时间（独立于股价）内失去全部价值，这被建模为齐次泊松过程的第一到达时间，我们研究了梅顿的最优投资组合问题，旨在最大化在预先选择的有限到期时间内总财富的期望对数效用。首先，我们提出了一种基于新型哈密顿-雅可比-贝尔曼方程的启发式推导。然后，我们将问题形式化为经典的受控马尔可夫扩散，具有一种新类型的终端和运行成本。采用贝尔曼的动态规划原理严格推导出了梅顿比率的新版本，并用适当类型的验证定理进行验证。一个现实世界的例子将后者的比率与经典梅顿的r进行了比较。

    arXiv:2403.15923v1 Announce Type: new  Abstract: Consider a stock market following a geometric Brownian motion and a riskless asset continuously compounded at a constant rate. Assuming the stock can go bankrupt, i.e., lose all of its value, at some exogenous random time (independent of the stock price) modeled as the first arrival time of a homogeneous Poisson process, we study the Merton's optimal portfolio problem consisting of maximizing the expected logarithmic utility of the total wealth at a preselected finite maturity time. First, we present a heuristic derivation based on a new type of Hamilton-Jacobi-Bellman equation. Then, we formally reduce the problem to a classical controlled Markovian diffusion with a new type of terminal and running costs. A new version of Merton's ratio is rigorously derived using Bellman's dynamic programming principle and validated with a suitable type of verification theorem. A real-world example comparing the latter ratio to the classical Merton's r
    
[^2]: 美国宽带基础设施补贴和投资计划对国内生产总值的贡献：基于投入产出模型

    The contribution of US broadband infrastructure subsidy and investment programs to GDP using input-output modeling

    [https://arxiv.org/abs/2311.02431](https://arxiv.org/abs/2311.02431)

    这篇论文研究了美国宽带基础设施补贴和投资计划对国内生产总值的贡献。通过投入产出模型探索了宽带支出对宏观经济的影响。

    

    尽管被公认为一种重要的福利，超过五分之一的美国人口没有订阅固定宽带服务。例如，尽管年收入超过7万美元的市民中只有不到4%没有宽带，但收入低于2万美元的市民中有26%没有宽带。为了解决这个问题，拜登政府通过《两党基础设施法》实施了有史以来规模最大的宽带投资计划，旨在解决这种差距，并将宽带连接扩展到所有公民。支持者表示，这将从根本上减少美国的数字鸿沟。然而，批评者认为，该计划导致经济周期的后期出现了前所未有的借贷，留下了很少的财政余地。因此，在这篇论文中，我们将研究宽带的可用性、采用情况和需求，然后构建一个投入产出模型，以探索宽带支出对国内生产总值（GDP）的宏观经济影响。最后，我们量化了部门间的影响。

    More than one-fifth of the US population does not subscribe to a fixed broadband service despite being a recognized merit good. For example, less than 4% of citizens earning more than US \$70k annually do not have broadband, compared to 26% of those earning below US \$20k annually. To address this, the Biden Administration has undertaken one of the largest broadband investment programs ever via The Bipartisan Infrastructure Law, with the aim of addressing this disparity and expanding broadband connectivity to all citizens. Proponents state this will reduce the US digital divide once-and-for-all. However, detractors say the program leads to unprecedented borrowing at a late stage of the economic cycle, leaving little fiscal headroom. Subsequently, in this paper, we examine broadband availability, adoption, and need and then construct an input-output model to explore the macroeconomic impacts of broadband spending in Gross Domestic Product (GDP) terms. Finally, we quantify inter-sectoral
    
[^3]: 神经霍克斯：高维度的非参数估计和加密货币市场的因果分析

    Neural Hawkes: Non-Parametric Estimation in High Dimension and Causality Analysis in Cryptocurrency Markets. (arXiv:2401.09361v1 [q-fin.TR])

    [http://arxiv.org/abs/2401.09361](http://arxiv.org/abs/2401.09361)

    这篇论文提出了一种名为神经霍克斯的方法，它能够在高维度情况下进行非参数估计，并用于加密货币市场的因果分析。该方法通过解决第二类Fredholm积分方程来推断标记霍克斯核，并使用物理信息神经网络进行数值计算。该方法在模拟数据上得到了广泛的验证，并应用于分析加密货币市场的微观结构和因果关系。

    

    我们提出了一种新颖的标记霍克斯核推断方法，称之为基于矩的神经霍克斯估计方法。霍克斯过程可以通过它们的一阶和二阶统计特性完全描述，通过第二类Fredholm积分方程。利用最新的物理信息神经网络求解偏微分方程的进展，我们提出了一种在高维度下求解这个积分方程的数值过程。结合适应性训练流程，我们给出了一组通用的超参数，能够在各种核形状范围内产生稳健的结果。我们在模拟数据上进行了大量的数值验证。最后，我们提出了两个应用该方法于加密货币市场微观结构分析。在第一个应用中，我们提取了BTC-USD交易到达率对成交量的影响，第二个应用中，我们分析了15种加密货币之间的因果关系及其方向。

    We propose a novel approach to marked Hawkes kernel inference which we name the moment-based neural Hawkes estimation method. Hawkes processes are fully characterized by their first and second order statistics through a Fredholm integral equation of the second kind. Using recent advances in solving partial differential equations with physics-informed neural networks, we provide a numerical procedure to solve this integral equation in high dimension. Together with an adapted training pipeline, we give a generic set of hyperparameters that produces robust results across a wide range of kernel shapes. We conduct an extensive numerical validation on simulated data. We finally propose two applications of the method to the analysis of the microstructure of cryptocurrency markets. In a first application we extract the influence of volume on the arrival rate of BTC-USD trades and in a second application we analyze the causality relationships and their directions amongst a universe of 15 crypto
    

