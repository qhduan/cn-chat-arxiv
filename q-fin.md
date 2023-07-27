# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Modeling Inverse Demand Function with Explainable Dual Neural Networks.](http://arxiv.org/abs/2307.14322) | 本研究提出了一种新的双重神经网络结构，用于建模金融传染中的反向需求函数。该方法能够通过预测资产清算和推导均衡价格来解决实际场景中的数据缺失问题。 |
| [^2] | [Derivative Pricing using Quantum Signal Processing.](http://arxiv.org/abs/2307.14310) | 这项研究介绍了一种基于量子信号处理的方法，将金融衍生品的回报直接编码到量子幅度中，减少了对昂贵的量子算术的需求，并显著降低了所需的量子资源。 |
| [^3] | [Socioeconomic agents as active matter in nonequilibrium Sakoda-Schelling models.](http://arxiv.org/abs/2307.14270) | 该研究通过考虑Sakoda-Schelling模型中的职业模型，揭示了社会经济代理人模型中的非平衡动力学，并在平均场近似下将其映射为主动物质描述。通过研究非互惠性互动，展示了非稳态的宏观行为。这一研究为地理相关的基于代理人的模型提供了统一的框架，有助于同时考虑人口和价格动态。 |
| [^4] | [Interest rate convexity in a Gaussian framework.](http://arxiv.org/abs/2307.14218) | 本文研究了高斯框架下的利率凸性，提出了由高斯Volterra过程驱动的短期利率模型，并给出了凸性调整的显式公式。 |
| [^5] | [Macroscopic Market Making.](http://arxiv.org/abs/2307.14129) | 这项研究提出了宏观市场做市模型，通过连续过程对委托进行建模，填补了市场做市和最优执行问题之间的间隙，并分析了委托流对策略的影响。 |
| [^6] | [Capital Structure Theories and its Practice, A study with reference to select NSE listed public sectors banks, India.](http://arxiv.org/abs/2307.14049) | 本研究通过研究印度公共部门银行的资本结构行为变化，结合流行的资本结构理论，验证了这些理论对银行绩效的适用性。 |
| [^7] | [American options in time-dependent one-factor models: Semi-analytic pricing, numerical methods and ML support.](http://arxiv.org/abs/2307.13870) | 本文填补了时间依赖的一因子模型中美式期权定价的空白，并提出了一种称为"广义积分变换"方法的半解析定价方法，该方法在计算效率、精度和稳定性方面与前向有限差分求解器相当，同时扩展到了其他模型中。 |
| [^8] | [Multi-Factor Inception: What to Do with All of These Features?.](http://arxiv.org/abs/2307.13832) | MFIN是一个多资产和多因素系统交易框架，通过学习特征并优化投资组合夏普比率的头寸大小，提供了比规则策略更好的表现。 |
| [^9] | [Sports Betting: an application of neural networks and modern portfolio theory to the English Premier League.](http://arxiv.org/abs/2307.13807) | 本文提出了一种将神经网络模型与投资组合优化相结合的新方法，通过研究英超联赛数据，成功实现了超过初始财富135.8%的惊人利润的体育博彩策略优化。 |
| [^10] | [Liquidity fragmentation on decentralized exchanges.](http://arxiv.org/abs/2307.13772) | 本研究通过分析Uniswap的数据发现，去中心化交易所中的固定交易成本导致了小型流动性提供者受到不成比例的影响，结果是低费用池和高费用池之间的流动性供应分裂，给大型流动性提供者和小型流动性提供者带来不同的影响。 |
| [^11] | [Is Kyle's equilibrium model stable?.](http://arxiv.org/abs/2307.09392) | 该论文证明了在Kyle的均衡模型中，当只有一个或两个交易时稳定，而当有三个或更多交易时不稳定。这些结果是独立于所有的Kyle输入参数的。 |
| [^12] | [Liquidity Premium and Liquidity-Adjusted Return and Volatility: illustrated with a Liquidity-Adjusted Mean Variance Framework and its Application on a Portfolio of Crypto Assets.](http://arxiv.org/abs/2306.15807) | 这项研究创建了创新技术来度量加密资产的流动性溢价，并开发了流动性调整的模型来提高投资组合的预测性能。 |
| [^13] | [A Novel Deep Reinforcement Learning Based Automated Stock Trading System Using Cascaded LSTM Networks.](http://arxiv.org/abs/2212.02721) | 本文提出了一种使用级联LSTM网络的基于深度强化学习的自动股票交易系统，通过对股票数据进行特征提取和策略函数训练，我们的模型在累积回报和夏普比率方面优于基准模型，特别在中国股市这一新兴市场中表现更为突出。 |

# 详细

[^1]: 用可解释的双重神经网络建模反向需求函数

    Modeling Inverse Demand Function with Explainable Dual Neural Networks. (arXiv:2307.14322v1 [q-fin.CP])

    [http://arxiv.org/abs/2307.14322](http://arxiv.org/abs/2307.14322)

    本研究提出了一种新的双重神经网络结构，用于建模金融传染中的反向需求函数。该方法能够通过预测资产清算和推导均衡价格来解决实际场景中的数据缺失问题。

    

    金融传染被广泛认为是金融系统的基本风险。特别强大的是基于价格的传染，其中公司的强制清算压低资产价格并传播金融压力，使危机能够在看似无关的实体之间广泛蔓延。目前，价格影响是通过外生的反向需求函数建模的。然而，在实际场景中，通常只能观察到初始冲击和最终的均衡资产价格，导致实际资产清算在很大程度上被隐藏。这些缺失的数据给校准现有模型带来了显著的局限性。为了解决这些挑战，我们引入了一种新的双重神经网络结构，它在两个连续的阶段中运作：第一个神经网络将初始冲击映射到预测的资产清算中，第二个网络利用这些清算来推导出结果均衡价格。这种数据驱动的方法可以捕捉线性和非线性程度更高的需求函数。

    Financial contagion has been widely recognized as a fundamental risk to the financial system. Particularly potent is price-mediated contagion, wherein forced liquidations by firms depress asset prices and propagate financial stress, enabling crises to proliferate across a broad spectrum of seemingly unrelated entities. Price impacts are currently modeled via exogenous inverse demand functions. However, in real-world scenarios, only the initial shocks and the final equilibrium asset prices are typically observable, leaving actual asset liquidations largely obscured. This missing data presents significant limitations to calibrating the existing models. To address these challenges, we introduce a novel dual neural network structure that operates in two sequential stages: the first neural network maps initial shocks to predicted asset liquidations, and the second network utilizes these liquidations to derive resultant equilibrium prices. This data-driven approach can capture both linear an
    
[^2]: 使用量子信号处理进行衍生品定价

    Derivative Pricing using Quantum Signal Processing. (arXiv:2307.14310v1 [quant-ph])

    [http://arxiv.org/abs/2307.14310](http://arxiv.org/abs/2307.14310)

    这项研究介绍了一种基于量子信号处理的方法，将金融衍生品的回报直接编码到量子幅度中，减少了对昂贵的量子算术的需求，并显著降低了所需的量子资源。

    

    在量子计算机上进行金融衍生品定价通常包括量子算术组件，这些组件对应的电路所需的量子资源很重。在本文中，我们介绍了一种基于量子信号处理（QSP）的方法，直接将金融衍生品的回报编码到量子幅度中，减轻了昂贵的量子算术对量子电路的负担。与现有文献中的当前最先进方法相比，我们发现对于实际感兴趣的衍生品合约，应用QSP显著减小了所有度量标准下所需的资源，尤其是T门的总数约减16倍，逻辑量子比特数约减4倍。此外，我们估计达到量子优势所需的逻辑时钟速率也降低了约5倍。总体而言，我们发现达到量子优势所需的逻辑量子比特为4.7k个，而能够执行10^9个T门的量子设备所需。

    Pricing financial derivatives on quantum computers typically includes quantum arithmetic components which contribute heavily to the quantum resources required by the corresponding circuits. In this manuscript, we introduce a method based on Quantum Signal Processing (QSP) to encode financial derivative payoffs directly into quantum amplitudes, alleviating the quantum circuits from the burden of costly quantum arithmetic. Compared to current state-of-the-art approaches in the literature, we find that for derivative contracts of practical interest, the application of QSP significantly reduces the required resources across all metrics considered, most notably the total number of T-gates by $\sim 16$x and the number of logical qubits by $\sim 4$x. Additionally, we estimate that the logical clock rate needed for quantum advantage is also reduced by a factor of $\sim 5$x. Overall, we find that quantum advantage will require $4.7$k logical qubits, and quantum devices that can execute $10^9$ T
    
[^3]: 非平衡的Sakoda-Schelling模型中的社会经济代理人作为主动物质

    Socioeconomic agents as active matter in nonequilibrium Sakoda-Schelling models. (arXiv:2307.14270v1 [cond-mat.stat-mech])

    [http://arxiv.org/abs/2307.14270](http://arxiv.org/abs/2307.14270)

    该研究通过考虑Sakoda-Schelling模型中的职业模型，揭示了社会经济代理人模型中的非平衡动力学，并在平均场近似下将其映射为主动物质描述。通过研究非互惠性互动，展示了非稳态的宏观行为。这一研究为地理相关的基于代理人的模型提供了统一的框架，有助于同时考虑人口和价格动态。

    

    代理人的决策规则对于社会经济代理人模型有多么稳健？我们通过考虑一种类似Sakoda-Schelling模型的职业模型来解决这个问题，该模型在历史上被引入以揭示人类群体之间的隔离动力学。对于大类的效用函数和决策规则，我们确定了代理人动力学的非平衡性，同时恢复了类似平衡相分离的现象学。在平均场近似下，我们展示了该模型在一定程度上可以被映射为主动物质场描述（Active Model B）。最后，我们考虑了两个人群之间的非互惠性互动，并展示了它们如何导致非稳态的宏观行为。我们相信我们的方法提供了一个统一的框架，进一步研究地理相关的基于代理人的模型，尤其是在场论方法中同时考虑人口和价格动态的研究。

    How robust are socioeconomic agent-based models with respect to the details of the agents' decision rule? We tackle this question by considering an occupation model in the spirit of the Sakoda-Schelling model, historically introduced to shed light on segregation dynamics among human groups. For a large class of utility functions and decision rules, we pinpoint the nonequilibrium nature of the agent dynamics, while recovering the equilibrium-like phase separation phenomenology. Within the mean field approximation we show how the model can be mapped, to some extent, onto an active matter field description (Active Model B). Finally, we consider non-reciprocal interactions between two populations, and show how they can lead to non-steady macroscopic behavior. We believe our approach provides a unifying framework to further study geography-dependent agent-based models, notably paving the way for joint consideration of population and price dynamics within a field theoretic approach.
    
[^4]: 高斯框架下的利率凸性

    Interest rate convexity in a Gaussian framework. (arXiv:2307.14218v1 [q-fin.PR])

    [http://arxiv.org/abs/2307.14218](http://arxiv.org/abs/2307.14218)

    本文研究了高斯框架下的利率凸性，提出了由高斯Volterra过程驱动的短期利率模型，并给出了凸性调整的显式公式。

    

    本文的贡献有两个：我们定义并研究了由一般高斯Volterra过程驱动的短期利率模型的性质，并在准确定义凸性调整的基础上，推导了其显式的公式。

    The contributions of this paper are twofold: we define and investigate the properties of a short rate model driven by a general Gaussian Volterra process and, after defining precisely a notion of convexity adjustment, derive explicit formulae for it.
    
[^5]: 宏观市场做市模型

    Macroscopic Market Making. (arXiv:2307.14129v1 [q-fin.MF])

    [http://arxiv.org/abs/2307.14129](http://arxiv.org/abs/2307.14129)

    这项研究提出了宏观市场做市模型，通过连续过程对委托进行建模，填补了市场做市和最优执行问题之间的间隙，并分析了委托流对策略的影响。

    

    我们提出了基于连续过程的宏观市场做市模型，与离散点过程相比。该模型旨在填补市场做市和最优执行问题之间的差距，同时阐明委托流对策略的影响。我们通过三个问题演示了我们的模型。这项研究从马尔可夫到非马尔可夫噪声，从线性到非线性强度函数，涵盖了有界和无界系数的综合分析。在数学上，其贡献在于最优控制的存在和唯一性，由Hamilton-Jacobi-Bellman方程或（非）利普希茨前向-后向随机微分方程的良定义性保证。

    We propose the macroscopic market making model \`a la Avellaneda-Stoikov, using continuous processes for orders instead of discrete point processes. The model intends to bridge a gap between market making and optimal execution problems, while shedding light on the influence of order flows on the strategy. We demonstrate our model through three problems. The study provides a comprehensive analysis from Markovian to non-Markovian noises and from linear to non-linear intensity functions, encompassing both bounded and unbounded coefficients. Mathematically, the contribution lies in the existence and uniqueness of the optimal control, guaranteed by the well-posedness of the Hamilton-Jacobi-Bellman equation or the (non-)Lipschitz forward-backward stochastic differential equation.
    
[^6]: 资本结构理论及其实践——以印度选定上市的国家证券交易所公共部门银行为例的研究

    Capital Structure Theories and its Practice, A study with reference to select NSE listed public sectors banks, India. (arXiv:2307.14049v1 [q-fin.GN])

    [http://arxiv.org/abs/2307.14049](http://arxiv.org/abs/2307.14049)

    本研究通过研究印度公共部门银行的资本结构行为变化，结合流行的资本结构理论，验证了这些理论对银行绩效的适用性。

    

    在现代市场中影响公司定位和绩效的各种因素中，公司的资本结构有其自己的方式来表达自己的重要性。随着技术的快速变化，公司被推向一种使资本管理过程繁重的范式。因此，资本结构变化的研究为投资者提供了对公司行为和内在目标的深入了解。这些变化会因不同行业的公司而异。本研究考虑了银行业，该行业根据印度的运作规定有一个独特的资本结构。本文研究了一些公共部门银行的资本结构行为变化。从流行的资本结构理论中制定了一个理论框架，并相应地推导出假设。主要目的是验证选择银行在2011年至2022年的实时表现与不同理论的一致性。使用统计技术，如回归分析、相关分析和差异检验等，对数据进行了分析。

    Among the various factors affecting the firms positioning and performance in modern day markets, capital structure of the firm has its own way of expressing itself as a crucial one. With the rapid changes in technology, firms are being pushed onto a paradigm that is burdening the capital management process. Hence the study of capital structure changes gives the investors an insight into firm's behavior and intrinsic goals. These changes will vary for firms in different sectors. This work considers the banking sector, which has a unique capital structure for the given regulations of its operations in India. The capital structure behavioral changes in a few public sector banks are studied in this paper. A theoretical framework has been developed from the popular capital structure theories and hypotheses are derived from them accordingly. The main idea is to validate different theories with real time performance of the select banks from 2011 to 2022. Using statistical techniques like regr
    
[^7]: 美式期权在时间依赖的一因子模型中的半解析定价、数值方法和机器学习支持

    American options in time-dependent one-factor models: Semi-analytic pricing, numerical methods and ML support. (arXiv:2307.13870v1 [q-fin.CP])

    [http://arxiv.org/abs/2307.13870](http://arxiv.org/abs/2307.13870)

    本文填补了时间依赖的一因子模型中美式期权定价的空白，并提出了一种称为"广义积分变换"方法的半解析定价方法，该方法在计算效率、精度和稳定性方面与前向有限差分求解器相当，同时扩展到了其他模型中。

    

    在一种时间依赖的Ornstein-Uhlenbeck模型中，介绍了美式期权的半解析定价方法。论文指出，为了获得这些定价，需要数值地求解一个非线性的第二类Volterra积分方程来找到行权边界（它是时间的函数）。一旦完成这一步骤，期权价格就可以得到。论文还证明了，计算上，这种方法与前向有限差分求解器一样高效，同时提供更好的精度和稳定性。后来，作者（还与Peter Carr和Alex Lipton合作）将这种方法显著扩展到各种时间依赖的一因子和随机波动率模型，用于定价障碍期权。然而，对于美式期权，尽管可能存在，但这并没有在任何地方明确报告过。本文的目标是填补这个空白，并讨论哪种数值方法（包括...）

    Semi-analytical pricing of American options in a time-dependent Ornstein-Uhlenbeck model was presented in [Carr, Itkin, 2020]. It was shown that to obtain these prices one needs to solve (numerically) a nonlinear Volterra integral equation of the second kind to find the exercise boundary (which is a function of the time only). Once this is done, the option prices follow. It was also shown that computationally this method is as efficient as the forward finite difference solver while providing better accuracy and stability. Later this approach called "the Generalized Integral transform" method has been significantly extended by the authors (also, in cooperation with Peter Carr and Alex Lipton) to various time-dependent one factor, and stochastic volatility models as applied to pricing barrier options. However, for American options, despite possible, this was not explicitly reported anywhere. In this paper our goal is to fill this gap and also discuss which numerical method (including tho
    
[^8]: 多因素入门：如何处理这些特征？

    Multi-Factor Inception: What to Do with All of These Features?. (arXiv:2307.13832v1 [q-fin.TR])

    [http://arxiv.org/abs/2307.13832](http://arxiv.org/abs/2307.13832)

    MFIN是一个多资产和多因素系统交易框架，通过学习特征并优化投资组合夏普比率的头寸大小，提供了比规则策略更好的表现。

    

    加密货币交易是一个新兴的研究领域，在行业中越来越受到采用。由于其去中心化的特性，许多描述加密货币的指标可以通过简单的谷歌搜索获得，并且通常会定期更新，至少每天更新一次。这为数据驱动的系统交易研究提供了一个有前途的机会，可以通过额外的特征（如哈希率或谷歌趋势）来增强有限的历史数据。然而，一个自然而然的问题出现了：如何有效地选择和处理这些特征？在本文中，我们引入了多因素入门网络(MFIN)，这是一个端到端的多资产和多因素系统交易框架。MFIN将深度入门网络(DIN)扩展到多因素环境中。与DIN类似，MFIN模型可以自动从回报数据中学习特征，并输出优化投资组合夏普比率的头寸大小。与一系列基于规则的动量和回归策略相比，MFIN可以提供更好的表现。

    Cryptocurrency trading represents a nascent field of research, with growing adoption in industry. Aided by its decentralised nature, many metrics describing cryptocurrencies are accessible with a simple Google search and update frequently, usually at least on a daily basis. This presents a promising opportunity for data-driven systematic trading research, where limited historical data can be augmented with additional features, such as hashrate or Google Trends. However, one question naturally arises: how to effectively select and process these features? In this paper, we introduce Multi-Factor Inception Networks (MFIN), an end-to-end framework for systematic trading with multiple assets and factors. MFINs extend Deep Inception Networks (DIN) to operate in a multi-factor context. Similar to DINs, MFIN models automatically learn features from returns data and output position sizes that optimise portfolio Sharpe ratio. Compared to a range of rule-based momentum and reversion strategies, M
    
[^9]: 体育博彩：神经网络和现代投资组合理论在英超联赛中的应用

    Sports Betting: an application of neural networks and modern portfolio theory to the English Premier League. (arXiv:2307.13807v1 [q-fin.PM])

    [http://arxiv.org/abs/2307.13807](http://arxiv.org/abs/2307.13807)

    本文提出了一种将神经网络模型与投资组合优化相结合的新方法，通过研究英超联赛数据，成功实现了超过初始财富135.8%的惊人利润的体育博彩策略优化。

    

    本文提出了一种在体育博彩中优化投注策略的新方法，该方法将冯·诺依曼-莫根斯特恩期望效用理论、深度学习技术和凯利标准的先进公式相结合。通过将神经网络模型与投资组合优化相结合，我们的方法在2020/2021英超联赛的后半段相对于初始财富获得了惊人的利润，达到了135.8%。我们研究了完整和受限策略，评估了它们的绩效、风险管理和多样化。我们开发了一个深度神经网络模型来预测比赛结果，解决了变量有限等挑战。我们的研究在体育博彩和预测建模领域提供了有价值的洞察和实际应用。

    This paper presents a novel approach for optimizing betting strategies in sports gambling by integrating Von Neumann-Morgenstern Expected Utility Theory, deep learning techniques, and advanced formulations of the Kelly Criterion. By combining neural network models with portfolio optimization, our method achieved remarkable profits of 135.8% relative to the initial wealth during the latter half of the 20/21 season of the English Premier League. We explore complete and restricted strategies, evaluating their performance, risk management, and diversification. A deep neural network model is developed to forecast match outcomes, addressing challenges such as limited variables. Our research provides valuable insights and practical applications in the field of sports betting and predictive modeling.
    
[^10]: 去中心化交易所的流动性分裂

    Liquidity fragmentation on decentralized exchanges. (arXiv:2307.13772v1 [q-fin.TR])

    [http://arxiv.org/abs/2307.13772](http://arxiv.org/abs/2307.13772)

    本研究通过分析Uniswap的数据发现，去中心化交易所中的固定交易成本导致了小型流动性提供者受到不成比例的影响，结果是低费用池和高费用池之间的流动性供应分裂，给大型流动性提供者和小型流动性提供者带来不同的影响。

    

    我们研究了去中心化交易所中流动性供应的规模经济性，重点关注固定交易成本（例如燃气价格）对流动性提供者（LPs）的影响。小型LPs受到固定成本的不成比例影响，导致低费用池和高费用池之间的流动性供应分裂。通过分析Uniswap的数据，我们发现高费用池吸引了56%的流动性供应，但只执行了35%的交易量。大型（机构）LPs主导低费用池，经常根据大量交易进行调整。相反，小型（零售）LPs趋于高费用池，为了减轻较小的流动性管理成本而接受较低的执行概率。

    We study economies of scale in liquidity provision on decentralized exchanges, focusing on the impact of fixed transaction costs such as gas prices on liquidity providers (LPs). Small LPs are disproportionately affected by the fixed cost, resulting in liquidity supply fragmentation between low- and high-fee pools. Analyzing Uniswap data, we find that high-fee pools attract 56% of liquidity supply but execute only 35% of trading volume. Large (institutional) LPs dominate low-fee pools, frequently adjusting positions in response to substantial trading volume. In contrast, small (retail) LPs converge to high-fee pools, accepting lower execution probabilities to mitigate smaller liquidity management costs.
    
[^11]: Kyle的均衡模型是否稳定？

    Is Kyle's equilibrium model stable?. (arXiv:2307.09392v1 [q-fin.TR])

    [http://arxiv.org/abs/2307.09392](http://arxiv.org/abs/2307.09392)

    该论文证明了在Kyle的均衡模型中，当只有一个或两个交易时稳定，而当有三个或更多交易时不稳定。这些结果是独立于所有的Kyle输入参数的。

    

    在Kyle (1985)的动态离散时间交易环境中，我们证明了当只有一个或两个交易时，Kyle的均衡模型是稳定的。对于三个或更多的交易时机，我们证明了Kyle的均衡是不稳定的。这些理论结果被证明与Kyle的所有输入参数无关。

    In the dynamic discrete-time trading setting of Kyle (1985), we prove that Kyle's equilibrium model is stable when there are one or two trading times. For three or more trading times, we prove that Kyle's equilibrium is not stable. These theoretical results are proven to hold irrespectively of all Kyle's input parameters.
    
[^12]: 流动性溢价和流动性调整收益与波动性：以流动性调整的均值方差框架及其在加密资产投资组合上的应用为例

    Liquidity Premium and Liquidity-Adjusted Return and Volatility: illustrated with a Liquidity-Adjusted Mean Variance Framework and its Application on a Portfolio of Crypto Assets. (arXiv:2306.15807v1 [q-fin.PM])

    [http://arxiv.org/abs/2306.15807](http://arxiv.org/abs/2306.15807)

    这项研究创建了创新技术来度量加密资产的流动性溢价，并开发了流动性调整的模型来提高投资组合的预测性能。

    

    我们建立了创新的流动性溢价Beta度量方法，并应用于选定的加密资产，同时对个别资产的收益进行流动性调整并建模，以及对投资组合的波动性进行流动性调整和建模。在高流动性情况下，这两个模型都表现出较强的可预测性，这使得流动性调整的均值方差 (LAMV) 框架在投资组合表现上比正常的均值方差 (RMV) 框架具有明显的优势。

    We establish innovative measures of liquidity premium Beta on both asset and portfolio levels, and corresponding liquidity-adjusted return and volatility, for selected crypto assets. We develop liquidity-adjusted ARMA-GARCH/EGARCH representation to model the liquidity-adjusted return of individual assets, and liquidity-adjusted VECM/VAR-DCC/ADCC structure to model the liquidity-adjusted variance of portfolio. Both models exhibit improved predictability at high liquidity, which affords a liquidity-adjusted mean-variance (LAMV) framework a clear advantage over its regular mean variance (RMV) counterpart in portfolio performance.
    
[^13]: 使用级联LSTM网络的一种新型基于深度强化学习的自动股票交易系统

    A Novel Deep Reinforcement Learning Based Automated Stock Trading System Using Cascaded LSTM Networks. (arXiv:2212.02721v2 [q-fin.CP] UPDATED)

    [http://arxiv.org/abs/2212.02721](http://arxiv.org/abs/2212.02721)

    本文提出了一种使用级联LSTM网络的基于深度强化学习的自动股票交易系统，通过对股票数据进行特征提取和策略函数训练，我们的模型在累积回报和夏普比率方面优于基准模型，特别在中国股市这一新兴市场中表现更为突出。

    

    越来越多的股票交易策略是使用深度强化学习（DRL）算法构建的，但是DRL方法最初在游戏界广泛使用，直接适应金融数据的低信噪比和不均匀性会导致性能不足。为了捕捉隐藏的信息，本文提出了一种使用级联LSTM的DRL股票交易系统，首先使用LSTM从股票日常数据中提取时间序列特征，然后将提取的特征馈给代理进行训练，同时强化学习中的策略函数也使用另一个LSTM进行训练。在美国市场的DJI和中国股市的SSE50上的实验证明，我们的模型在累积回报和夏普比率方面优于以前的基准模型，并且在中国股市这一新兴市场中优势更为显著。这表明我们提出的方法是一种有前景的构建自动股票交易系统的方式。

    More and more stock trading strategies are constructed using deep reinforcement learning (DRL) algorithms, but DRL methods originally widely used in the gaming community are not directly adaptable to financial data with low signal-to-noise ratios and unevenness, and thus suffer from performance shortcomings. In this paper, to capture the hidden information, we propose a DRL based stock trading system using cascaded LSTM, which first uses LSTM to extract the time-series features from stock daily data, and then the features extracted are fed to the agent for training, while the strategy functions in reinforcement learning also use another LSTM for training. Experiments in DJI in the US market and SSE50 in the Chinese stock market show that our model outperforms previous baseline models in terms of cumulative returns and Sharp ratio, and this advantage is more significant in the Chinese stock market, a merging market. It indicates that our proposed method is a promising way to build a aut
    

