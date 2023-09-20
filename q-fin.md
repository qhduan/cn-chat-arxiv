# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PAMS: Platform for Artificial Market Simulations.](http://arxiv.org/abs/2309.10729) | PAMS是一种Python-based模拟器，可与深度学习集成，用于模拟人工市场。它通过一个代理人的研究展示了它的有效性。 |
| [^2] | [Mean Absolute Directional Loss as a New Loss Function for Machine Learning Problems in Algorithmic Investment Strategies.](http://arxiv.org/abs/2309.10546) | 这项研究提出了一种新的损失函数(Mean Absolute Directional Loss，MADL)，用于解决在机器学习模型中应用于算法投资策略中的金融时间序列预测问题。通过在两种不同资产类别的数据上验证，MADL函数能够提供更好的超参数选择，并获得更有效的投资策略。 |
| [^3] | [Derivatives Sensitivities Computation under Heston Model on GPU.](http://arxiv.org/abs/2309.10477) | 本报告研究了在GPU上使用Heston模型计算欧式和亚式期权的希腊字母。我们实现了精确模拟方法的基准和一种新的GPU方法，结果表明，新方法相对于基准可以提供多达200倍的加速。然而，用GPU方法估计Rho的准确性不如CPU方法。这项研究揭示了GPU在使用数值方法计算衍生品敏感度方面的潜力。 |
| [^4] | [Human-AI Interactions and Societal Pitfalls.](http://arxiv.org/abs/2309.10448) | 本研究研究了人工智能与人类互动中面临的同质化和偏见问题，提出了改善人工智能与人类互动的解决办法，实现个性化输出而不牺牲生产力。 |
| [^5] | [OPUS: An Integrated Assessment Model for Satellites and Orbital Debris.](http://arxiv.org/abs/2309.10252) | OPUS是一个综合评估模型，结合了轨道人口的天体力学和空间活动者的经济行为，帮助决策者评估卫星和轨道残骸管理政策的有效性和可能的结果。 |
| [^6] | [Comparing effects of price limit and circuit breaker in stock exchanges by an agent-based model.](http://arxiv.org/abs/2309.10220) | 本研究通过基于智能体的模型比较了股票交易所中的价格限制和熔断机制，发现在参数相同的情况下，它们基本上具有相同的效果。然而，当限制时间范围小于取消时间范围时，价格限制的效果较差。 |
| [^7] | [Primal-Dual $\ell_0$-Constrained Sparse Index Tracking.](http://arxiv.org/abs/2309.10152) | 本文提出了一种新的稀疏指数跟踪问题的形式化，使用了$\ell_0$-范数约束，可以轻松控制投资组合中资产数量的上限。 |
| [^8] | [Can political gridlock undermine checks and balances? A lab experiment.](http://arxiv.org/abs/2309.10080) | 本研究通过实验研究发现，当政治僵局存在时，人们更倾向于削弱制衡机制，但并不仅仅是在改革有益时削弱，也包括在改革有害时削弱。 |
| [^9] | [Sizing Strategies for Algorithmic Trading in Volatile Markets: A Study of Backtesting and Risk Mitigation Analysis.](http://arxiv.org/abs/2309.09094) | 本文研究了金融交易中不同的规模模型，并回测了这些模型在高波动性市场中的效果。研究发现通过合理控制头寸规模，可以降低危机事件中的VaR模型。研究还使用了多种数据和模型进行了验证。 |
| [^10] | [On Sparse Grid Interpolation for American Option Pricing with Multiple Underlying Assets.](http://arxiv.org/abs/2309.08287) | 本文提出了一种基于稀疏网格插值的方法，用于定价包含多种标的资产的美式期权。通过动态规划和静态稀疏网格插值技术，我们能够高效地计算美式期权的继续价值函数，并通过减少插值点的数量实现计算效率的提高。数值实验结果表明该方法在定价美式算术和几何篮子看跌期权方面表现出色。 |
| [^11] | [ChatGPT Informed Graph Neural Network for Stock Movement Prediction.](http://arxiv.org/abs/2306.03763) | 该研究介绍了一种新的框架，利用ChatGPT技术增强图神经网络，能够从财经新闻中提取出不断变化的网络结构，并用于股票价格预测，获得了超过基于深度学习的最新基准的表现，提示了ChatGPT在文本推断和金融预测方面的潜力。 |
| [^12] | [The Emergence of Economic Rationality of GPT.](http://arxiv.org/abs/2305.12763) | 本文研究了GPT在经济理性方面的能力，通过指示其在4个领域中做出预算决策，发现GPT决策基本上是合理的，并且比人类更具有理性。 |
| [^13] | [Robust Detection of Lead-Lag Relationships in Lagged Multi-Factor Models.](http://arxiv.org/abs/2305.06704) | 该论文提出了一种基于聚类的鲁棒检测滞后多因子模型中的领先滞后关系方法，并使用各种聚类技术和相似度度量方法实现了对领先滞后估计的聚合，从而强化了对原始宇宙中的一致关系的识别。 |

# 详细

[^1]: PAMS: 人工市场模拟平台

    PAMS: Platform for Artificial Market Simulations. (arXiv:2309.10729v1 [q-fin.CP])

    [http://arxiv.org/abs/2309.10729](http://arxiv.org/abs/2309.10729)

    PAMS是一种Python-based模拟器，可与深度学习集成，用于模拟人工市场。它通过一个代理人的研究展示了它的有效性。

    

    本文介绍了一种新的人工市场模拟平台，PAMS: Platform for Artificial Market Simulations。PAMS是基于Python开发的模拟器，可以轻松与深度学习集成，实现各种需要用户简单修改的模拟。在本文中，我们通过使用通过深度学习预测未来价格的代理人来展示PAMS的有效性。

    This paper presents a new artificial market simulation platform, PAMS: Platform for Artificial Market Simulations. PAMS is developed as a Python-based simulator that is easily integrated with deep learning and enabling various simulation that requires easy users' modification. In this paper, we demonstrate PAMS effectiveness through a study using agents predicting future prices by deep learning.
    
[^2]: 机器学习问题中用于算法投资策略的新损失函数——平均绝对方向损失

    Mean Absolute Directional Loss as a New Loss Function for Machine Learning Problems in Algorithmic Investment Strategies. (arXiv:2309.10546v1 [q-fin.CP])

    [http://arxiv.org/abs/2309.10546](http://arxiv.org/abs/2309.10546)

    这项研究提出了一种新的损失函数(Mean Absolute Directional Loss，MADL)，用于解决在机器学习模型中应用于算法投资策略中的金融时间序列预测问题。通过在两种不同资产类别的数据上验证，MADL函数能够提供更好的超参数选择，并获得更有效的投资策略。

    

    本文研究了在算法投资策略（AIS）构建中，用于金融时间序列预测的机器学习模型的适当损失函数的问题。我们提出了平均绝对方向损失（MADL）函数，解决了传统预测误差函数在从预测中提取信息以创建有效的买卖信号方面的重要问题。最后，我们基于两种不同资产类别（加密货币：比特币和大宗商品：原油）的数据表明，新的损失函数使我们能够选择更好的LSTM模型超参数，并在样本外数据上获得更有效的投资策略，相对于风险调整回报指标。

    This paper investigates the issue of an adequate loss function in the optimization of machine learning models used in the forecasting of financial time series for the purpose of algorithmic investment strategies (AIS) construction. We propose the Mean Absolute Directional Loss (MADL) function, solving important problems of classical forecast error functions in extracting information from forecasts to create efficient buy/sell signals in algorithmic investment strategies. Finally, based on the data from two different asset classes (cryptocurrencies: Bitcoin and commodities: Crude Oil), we show that the new loss function enables us to select better hyperparameters for the LSTM model and obtain more efficient investment strategies, with regard to risk-adjusted return metrics on the out-of-sample data.
    
[^3]: 在GPU上计算Heston模型下的衍生品敏感性

    Derivatives Sensitivities Computation under Heston Model on GPU. (arXiv:2309.10477v1 [q-fin.CP])

    [http://arxiv.org/abs/2309.10477](http://arxiv.org/abs/2309.10477)

    本报告研究了在GPU上使用Heston模型计算欧式和亚式期权的希腊字母。我们实现了精确模拟方法的基准和一种新的GPU方法，结果表明，新方法相对于基准可以提供多达200倍的加速。然而，用GPU方法估计Rho的准确性不如CPU方法。这项研究揭示了GPU在使用数值方法计算衍生品敏感度方面的潜力。

    

    本报告研究了在GPU上计算Heston随机波动模型下欧式和亚式期权的希腊字母。我们首先实现了Broadie和Kaya提出的精确模拟方法，并将其用作精度和速度的基准。然后，我们提出了一种在GPU上使用Milstein离散化方法计算希腊字母的新方法。我们的结果表明，与精确模拟实现相比，所提出的方法可以提供多达200倍的加速，并且可以用于欧式和亚式期权。然而，用GPU方法估计Rho的准确性不及CPU方法。总的来说，我们的研究证明了使用数值方法计算衍生品敏感度时GPU的潜力。

    This report investigates the computation of option Greeks for European and Asian options under the Heston stochastic volatility model on GPU. We first implemented the exact simulation method proposed by Broadie and Kaya and used it as a baseline for precision and speed. We then proposed a novel method for computing Greeks using the Milstein discretisation method on GPU. Our results show that the proposed method provides a speed-up up to 200x compared to the exact simulation implementation and that it can be used for both European and Asian options. However, the accuracy of the GPU method for estimating Rho is inferior to the CPU method. Overall, our study demonstrates the potential of GPU for computing derivatives sensitivies with numerical methods.
    
[^4]: 人工智能与人类互动以及社会陷阱

    Human-AI Interactions and Societal Pitfalls. (arXiv:2309.10448v1 [cs.AI])

    [http://arxiv.org/abs/2309.10448](http://arxiv.org/abs/2309.10448)

    本研究研究了人工智能与人类互动中面临的同质化和偏见问题，提出了改善人工智能与人类互动的解决办法，实现个性化输出而不牺牲生产力。

    

    当与生成式人工智能（AI）合作时，用户可能会看到生产力的提升，但AI生成的内容可能不完全符合他们的偏好。为了研究这种影响，我们引入了一个贝叶斯框架，其中异质用户选择与AI共享多少信息，面临输出保真度和通信成本之间的权衡。我们展示了这些个体决策与AI训练之间的相互作用可能导致社会挑战。输出可能变得更加同质化，特别是当AI在AI生成的内容上进行训练时。而任何AI的偏见可能成为社会偏见。解决同质化和偏见问题的办法是改进人工智能与人类的互动，实现个性化输出而不牺牲生产力。

    When working with generative artificial intelligence (AI), users may see productivity gains, but the AI-generated content may not match their preferences exactly. To study this effect, we introduce a Bayesian framework in which heterogeneous users choose how much information to share with the AI, facing a trade-off between output fidelity and communication cost. We show that the interplay between these individual-level decisions and AI training may lead to societal challenges. Outputs may become more homogenized, especially when the AI is trained on AI-generated content. And any AI bias may become societal bias. A solution to the homogenization and bias issues is to improve human-AI interactions, enabling personalized outputs without sacrificing productivity.
    
[^5]: OPUS：用于卫星和轨道残骸的综合评估模型

    OPUS: An Integrated Assessment Model for Satellites and Orbital Debris. (arXiv:2309.10252v1 [physics.space-ph])

    [http://arxiv.org/abs/2309.10252](http://arxiv.org/abs/2309.10252)

    OPUS是一个综合评估模型，结合了轨道人口的天体力学和空间活动者的经济行为，帮助决策者评估卫星和轨道残骸管理政策的有效性和可能的结果。

    

    如何管理越来越多的卫星在轨道上成为了一个日益引人瞩目的公共政策挑战，包括大型星座。虽然提出了许多政策倡议试图从不同角度解决这个问题，但是缺乏分析工具来帮助决策者评估这些不同提议的有效性和可能的适得其反的结果。为了解决这个问题，本文总结了开发一个实验性综合评估模型-Orbital Debris Propagators Unified with Economic Systems (OPUS)的工作，该模型将轨道人口的天体力学和空间活动者的经济行为结合起来。对于给定的参数集，模型首先利用给定的天体动力学传播器评估轨道中物体的状态。然后利用一组用户定义的经济和政策参数（例如，发射价格，处置规定）模拟参与者对经济激励的反应。

    An increasingly salient public policy challenge is how to manage the growing number of satellites in orbit, including large constellations. Many policy initiatives have been proposed that attempt to address the problem from different angles, but there is a paucity of analytical tools to help policymakers evaluate the efficacy of these different proposals and any potential counterproductive outcomes. To help address this problem, this paper summarizes work done to develop an experimental integrated assessment model -Orbital Debris Propagators Unified with Economic Systems (OPUS) -- that combines both astrodynamics of the orbital population and economic behavior of space actors. For a given set of parameters, the model first utilizes a given astrodynamic propagator to assess the state of objects in orbit. It then uses a set of user-defined economic and policy parameters -- e.g. launch prices, disposal regulations -- to model how actors will respond to the economic incentives created by
    
[^6]: 通过基于智能体的模型比较股票交易所中的价格限制与熔断机制的效果

    Comparing effects of price limit and circuit breaker in stock exchanges by an agent-based model. (arXiv:2309.10220v1 [q-fin.CP])

    [http://arxiv.org/abs/2309.10220](http://arxiv.org/abs/2309.10220)

    本研究通过基于智能体的模型比较了股票交易所中的价格限制和熔断机制，发现在参数相同的情况下，它们基本上具有相同的效果。然而，当限制时间范围小于取消时间范围时，价格限制的效果较差。

    

    防止市场价格迅速下跌对于避免金融危机至关重要。为此，一些股票交易所实施了价格限制或熔断机制，并对哪种规定最有效地防止价格的快速和大幅波动进行了密集调查。本研究使用一个金融市场的基于智能体的人工市场模型来探讨这个问题。研究结果显示，在参数相同的情况下，价格限制和熔断机制基本上具有相同的效果。然而，当限制时间范围小于取消时间范围时，价格限制的效果较差。价格限制会导致许多卖单在较低限价处积累，当低限价在已积累的卖单被取消之前被改变时，会导致不同价格的卖单积累。这些积累的卖单实际上会形成一个阻碍市场下跌的壁垒。

    The prevention of rapidly and steeply falling market prices is vital to avoid financial crisis. To this end, some stock exchanges implement a price limit or a circuit breaker, and there has been intensive investigation into which regulation best prevents rapid and large variations in price. In this study, we examine this question using an artificial market model that is an agent-based model for a financial market. Our findings show that the price limit and the circuit breaker basically have the same effect when the parameters, limit price range and limit time range, are the same. However, the price limit is less effective when limit the time range is smaller than the cancel time range. With the price limit, many sell orders are accumulated around the lower limit price, and when the lower limit price is changed before the accumulated sell orders are cancelled, it leads to the accumulation of sell orders of various prices. These accumulated sell orders essentially act as a wall against b
    
[^7]: Primal-Dual $\ell_0$-约束稀疏指数跟踪

    Primal-Dual $\ell_0$-Constrained Sparse Index Tracking. (arXiv:2309.10152v1 [q-fin.PM])

    [http://arxiv.org/abs/2309.10152](http://arxiv.org/abs/2309.10152)

    本文提出了一种新的稀疏指数跟踪问题的形式化，使用了$\ell_0$-范数约束，可以轻松控制投资组合中资产数量的上限。

    

    稀疏指数跟踪是一种重要的被动投资组合管理策略，它通过构建稀疏投资组合来跟踪金融指数。相比于全仓投资组合，稀疏投资组合在降低交易成本和避免不流动资产方面更具优势。为了强制投资组合的稀疏性，传统研究提出了基于$\ell_p$-范数正则化的公式，作为$\ell_0$-范数正则化的连续替代。尽管这样的公式可以用来构建稀疏投资组合，但在实际投资中却不易使用，因为细致的参数调整来指定投资组合中资产数量的上限是艰难且耗时的。本文提出了一种新的稀疏指数跟踪问题的形式化，使用了$\ell_0$-范数约束，从而可以轻松控制投资组合中资产数量的上限。此外，我们的形式化允许在投资组合稀疏性和换手率之间进行选择。

    Sparse index tracking is one of the prominent passive portfolio management strategies that construct a sparse portfolio to track a financial index. A sparse portfolio is desirable over a full portfolio in terms of transaction cost reduction and avoiding illiquid assets. To enforce the sparsity of the portfolio, conventional studies have proposed formulations based on $\ell_p$-norm regularizations as a continuous surrogate of the $\ell_0$-norm regularization. Although such formulations can be used to construct sparse portfolios, they are not easy to use in actual investments because parameter tuning to specify the exact upper bound on the number of assets in the portfolio is delicate and time-consuming. In this paper, we propose a new problem formulation of sparse index tracking using an $\ell_0$-norm constraint that enables easy control of the upper bound on the number of assets in the portfolio. In addition, our formulation allows the choice between portfolio sparsity and turnover spa
    
[^8]: 政治僵局会削弱制衡吗？实验研究。

    Can political gridlock undermine checks and balances? A lab experiment. (arXiv:2309.10080v1 [econ.GN])

    [http://arxiv.org/abs/2309.10080](http://arxiv.org/abs/2309.10080)

    本研究通过实验研究发现，当政治僵局存在时，人们更倾向于削弱制衡机制，但并不仅仅是在改革有益时削弱，也包括在改革有害时削弱。

    

    如果制衡旨在保护公民免受政府滥用权力的侵害，为什么有时它们会削弱制衡机制呢？本实验通过实验室实验来探讨这个问题，实验对象需要在两种不同的决策规则之间选择：具有和没有制衡机制。如果不受制衡的行政部门能够推动由立法机构阻挠的改革，选民可能更倾向于支持无制衡的行政部门。与我们的预测相一致，当政治僵局存在时，我们发现实验对象更倾向于削弱制衡机制。然而，实验对象不仅在改革有益时削弱制衡机制，也在改革有害时削弱它们。

    If checks and balances are aimed at protecting citizens from the government's abuse of power, why do they sometimes weaken them? We address this question in a laboratory experiment in which subjects choose between two decision rules: with and without checks and balances. Voters may prefer an unchecked executive if that enables a reform that, otherwise, is blocked by the legislature. Consistent with our predictions, we find that subjects are more likely to weaken checks and balances when there is political gridlock. However, subjects weaken the controls not only when the reform is beneficial but also when it is harmful.
    
[^9]: 在波动市场中的算法交易的规模策略：一项基于回测和风险缓解分析的研究

    Sizing Strategies for Algorithmic Trading in Volatile Markets: A Study of Backtesting and Risk Mitigation Analysis. (arXiv:2309.09094v1 [q-fin.CP])

    [http://arxiv.org/abs/2309.09094](http://arxiv.org/abs/2309.09094)

    本文研究了金融交易中不同的规模模型，并回测了这些模型在高波动性市场中的效果。研究发现通过合理控制头寸规模，可以降低危机事件中的VaR模型。研究还使用了多种数据和模型进行了验证。

    

    回测是一种金融风险评估方法，帮助分析我们的交易算法在过去的市场中的表现。高波动性情况一直是算法交易员面临的挑战。本文研究了金融交易中的不同规模模型，并回测了在高波动性情况下如何使用规模模型降低VaR模型。因此，本文试图展示如何通过短期和长期头寸规模控制高波动性的危机事件。本文还研究了股票与AR、ARIMA、LSTM、GARCH以及ETF数据。

    Backtest is a way of financial risk evaluation which helps to analyze how our trading algorithm would work in markets with past time frame. The high volatility situation has always been a critical situation which creates challenges for algorithmic traders. The paper investigates different models of sizing in financial trading and backtest to high volatility situations to understand how sizing models can lower the models of VaR during crisis events. Hence it tries to show that how crisis events with high volatility can be controlled using short and long positional size. The paper also investigates stocks with AR, ARIMA, LSTM, GARCH with ETF data.
    
[^10]: 关于稀疏网格插值在多标的资产美式期权定价中的应用

    On Sparse Grid Interpolation for American Option Pricing with Multiple Underlying Assets. (arXiv:2309.08287v1 [math.NA])

    [http://arxiv.org/abs/2309.08287](http://arxiv.org/abs/2309.08287)

    本文提出了一种基于稀疏网格插值的方法，用于定价包含多种标的资产的美式期权。通过动态规划和静态稀疏网格插值技术，我们能够高效地计算美式期权的继续价值函数，并通过减少插值点的数量实现计算效率的提高。数值实验结果表明该方法在定价美式算术和几何篮子看跌期权方面表现出色。

    

    本文提出了一种基于高效积分和稀疏网格的多项式插值方法，用于定价包含多种标的资产的美式期权。该方法首先利用动态规划的思想对美式期权进行定价，然后使用静态稀疏网格对每个时间步长的继续价值函数进行插值。为了提高效率，我们首先通过缩放tanh映射将定义域从$\mathbb{R}^d$转换到$(-1,1)^d$，然后通过一个气泡函数消除在$(-1,1)^d$上的边界奇异性，并同时显著减少插值点的数量。我们严格证明了通过适当选择气泡函数，所得到的函数在一定阶数的混合导数上具有有界性，从而为使用稀疏网格提供了理论基础。数值实验结果表明，该方法在美式算术和几何篮子看跌期权定价中效果显著。

    In this work, we develop a novel efficient quadrature and sparse grid based polynomial interpolation method to price American options with multiple underlying assets. The approach is based on first formulating the pricing of American options using dynamic programming, and then employing static sparse grids to interpolate the continuation value function at each time step. To achieve high efficiency, we first transform the domain from $\mathbb{R}^d$ to $(-1,1)^d$ via a scaled tanh map, and then remove the boundary singularity of the resulting multivariate function over $(-1,1)^d$ by a bubble function and simultaneously, to significantly reduce the number of interpolation points. We rigorously establish that with a proper choice of the bubble function, the resulting function has bounded mixed derivatives up to a certain order, which provides theoretical underpinnings for the use of sparse grids. Numerical experiments for American arithmetic and geometric basket put options with the number
    
[^11]: ChatGPT信息的图神经网络用于股票价格预测

    ChatGPT Informed Graph Neural Network for Stock Movement Prediction. (arXiv:2306.03763v1 [q-fin.ST])

    [http://arxiv.org/abs/2306.03763](http://arxiv.org/abs/2306.03763)

    该研究介绍了一种新的框架，利用ChatGPT技术增强图神经网络，能够从财经新闻中提取出不断变化的网络结构，并用于股票价格预测，获得了超过基于深度学习的最新基准的表现，提示了ChatGPT在文本推断和金融预测方面的潜力。

    

    ChatGPT已在各种自然语言处理（NLP）任务中展示了出色的能力。然而，它从时间文本数据（尤其是财经新闻）推断动态网络结构的潜力仍是一个未开发的领域。在这项研究中，我们介绍了一个新的框架，利用ChatGPT的图推断能力来增强图神经网络（GNN）。我们的框架巧妙地从文本数据中提取出不断变化的网络结构，并将这些网络结构融合到图神经网络中，进行后续的预测任务。股票价格预测的实验结果表明，我们的模型始终优于基于深度学习的最新基准。此外，基于我们模型的产出构建的组合展示出更高的年化累计回报、更低的波动性和最大回撤。这种卓越表现突显了ChatGPT用于基于文本的网络推断和金融预测应用的潜力。

    ChatGPT has demonstrated remarkable capabilities across various natural language processing (NLP) tasks. However, its potential for inferring dynamic network structures from temporal textual data, specifically financial news, remains an unexplored frontier. In this research, we introduce a novel framework that leverages ChatGPT's graph inference capabilities to enhance Graph Neural Networks (GNN). Our framework adeptly extracts evolving network structures from textual data, and incorporates these networks into graph neural networks for subsequent predictive tasks. The experimental results from stock movement forecasting indicate our model has consistently outperformed the state-of-the-art Deep Learning-based benchmarks. Furthermore, the portfolios constructed based on our model's outputs demonstrate higher annualized cumulative returns, alongside reduced volatility and maximum drawdown. This superior performance highlights the potential of ChatGPT for text-based network inferences and 
    
[^12]: GPT的经济理性出现

    The Emergence of Economic Rationality of GPT. (arXiv:2305.12763v1 [econ.GN])

    [http://arxiv.org/abs/2305.12763](http://arxiv.org/abs/2305.12763)

    本文研究了GPT在经济理性方面的能力，通过指示其在4个领域中做出预算决策，发现GPT决策基本上是合理的，并且比人类更具有理性。

    

    随着像GPT这样的大型语言模型越来越普遍，评估它们在语言处理之外的能力至关重要。本文通过指示GPT在风险、时间、社交和食品偏好的四个领域中进行预算决策来研究GPT的经济理性。我们通过评估GPT决策与古典揭示偏好理论中的效用最大化一致性来衡量经济理性。我们发现GPT在每个领域的决策基本上是合理的，并且表现出比文献报道的人类更高的理性得分。我们还发现，理性得分对于随机程度和人口统计学设置（如年龄和性别）是稳健的，但对基于选择情境的语言框架的上下文敏感。这些结果表明了LLM作出良好决策的潜力，以及需要进一步了解它们的能力、局限性和基本机制。

    As large language models (LLMs) like GPT become increasingly prevalent, it is essential that we assess their capabilities beyond language processing. This paper examines the economic rationality of GPT by instructing it to make budgetary decisions in four domains: risk, time, social, and food preferences. We measure economic rationality by assessing the consistency of GPT decisions with utility maximization in classic revealed preference theory. We find that GPT decisions are largely rational in each domain and demonstrate higher rationality scores than those of humans reported in the literature. We also find that the rationality scores are robust to the degree of randomness and demographic settings such as age and gender, but are sensitive to contexts based on the language frames of the choice situations. These results suggest the potential of LLMs to make good decisions and the need to further understand their capabilities, limitations, and underlying mechanisms.
    
[^13]: 滞后多因子模型中领先滞后关系的鲁棒检测

    Robust Detection of Lead-Lag Relationships in Lagged Multi-Factor Models. (arXiv:2305.06704v1 [stat.ML])

    [http://arxiv.org/abs/2305.06704](http://arxiv.org/abs/2305.06704)

    该论文提出了一种基于聚类的鲁棒检测滞后多因子模型中的领先滞后关系方法，并使用各种聚类技术和相似度度量方法实现了对领先滞后估计的聚合，从而强化了对原始宇宙中的一致关系的识别。

    

    在多元时间序列系统中，通过发现数据中固有的领先滞后关系，可以获得关键信息，这指的是两个相对时间互移的时间序列之间的依赖关系，可以用于控制、预测或聚类。我们开发了一种基于聚类的方法，用于鲁棒检测滞后多因子模型中的领先滞后关系。在我们的框架中，所设想的管道接收一组时间序列作为输入，并使用滑动窗口方法从每个输入时间序列中提取一组子序列时间序列。然后，我们应用各种聚类技术（例如K-means++和谱聚类），采用各种成对相似性度量，包括非线性的相似性度量。一旦聚类被提取出来，跨聚类的领先滞后估计被聚合起来，以增强对原始宇宙中一致关系的识别。由于多

    In multivariate time series systems, key insights can be obtained by discovering lead-lag relationships inherent in the data, which refer to the dependence between two time series shifted in time relative to one another, and which can be leveraged for the purposes of control, forecasting or clustering. We develop a clustering-driven methodology for the robust detection of lead-lag relationships in lagged multi-factor models. Within our framework, the envisioned pipeline takes as input a set of time series, and creates an enlarged universe of extracted subsequence time series from each input time series, by using a sliding window approach. We then apply various clustering techniques (e.g, K-means++ and spectral clustering), employing a variety of pairwise similarity measures, including nonlinear ones. Once the clusters have been extracted, lead-lag estimates across clusters are aggregated to enhance the identification of the consistent relationships in the original universe. Since multi
    

