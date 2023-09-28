# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Implementing portfolio risk management and hedging in practice.](http://arxiv.org/abs/2309.15767) | 本文介绍了一种实践中的跨资产投资组合风险管理和对冲方法，利用（近似）二次规划的凸优化设置，解决了连续时间随机控制框架的不可取之处，并特别关注了经济概念和数学表示之间的对应关系。 |
| [^2] | [Hedging Properties of Algorithmic Investment Strategies using Long Short-Term Memory and Time Series models for Equity Indices.](http://arxiv.org/abs/2309.15640) | 本文提出了一种使用长短期记忆和时间序列模型构建算法投资策略的对冲方法，并通过利用不同类型的投资策略来对冲风险资产组合。实证结果显示，该方法在金融市场的动荡时期具有多样化的潜力。 |
| [^3] | [To better understand realized ecosystem services: An integrated analysis framework of supply, demand, flow and use.](http://arxiv.org/abs/2309.15574) | 该研究提出了一个供给-需求-流动-利用（SDFU）框架来理解实现生态系统服务（ES）。研究应用该框架分析了城市绿地公园中的野生浆果供给，传粉和娱乐等ES，揭示了ES的实际利用情况，以及供给受限、需求受限和供需平衡类型的ES。研究还讨论了ES的尺度特征、时间动态和空间特征，以及未来研究的关键问题。 |
| [^4] | [Startup success prediction and VC portfolio simulation using CrunchBase data.](http://arxiv.org/abs/2309.15552) | 本研究提出了一个使用CrunchBase数据来预测创业成功和模拟VC投资组合的新颖深度学习模型，并通过全面回溯算法对模型在历史数据上的表现进行了评估。 |
| [^5] | [Systemic risk in financial networks: the effects of asymptotic independence.](http://arxiv.org/abs/2309.15511) | 该论文研究了金融网络中的系统性风险，特别关注了渐近尾部独立性的影响。通过建立一般维度的互相渐近尾部独立性概念，并与传统的成对渐近独立性进行比较，提供了金融风险建模的新视角。此外，通过构建银行和资产的双分图展示了资产组合的多样渐近尾部独立性行为，并提供了条件尾部风险的精确渐近表达式。 |
| [^6] | [Enumerating the climate impact of disequilibrium in critical mineral supply.](http://arxiv.org/abs/2309.15368) | 根据研究，符合提议的尾气排放标准需要在2027年至2032年间用至少1021万辆新的内燃机汽车替换为EV。根据现有可利用的矿产储量，制造足够的EVs在大多数电池化学品中是可行的，并且可以减少高达4573万吨的CO2e。 |
| [^7] | [The importance of quality in austere times: University competitiveness and grant income.](http://arxiv.org/abs/2309.15309) | 这项研究利用英国科学资助紧缩政策的自然实验，发现传统的大学竞争力测量指标无法准确反映竞争力。通过使用一种基于复杂性科学的替代指标，研究人员揭示了大学在科学领域的高度动态参与方式，并发现研究竞争力对资助收入具有影响。紧缩政策放宽后，英国大学的地位和资助收入发生了变化。 |
| [^8] | [Quantum Analysis of Continuous Time Stochastic Process.](http://arxiv.org/abs/2208.02364) | 本文提出了一种通用框架，用于在量子计算机上高效地准备连续时间随机过程路径。基于压缩状态准备方法，可以高效地提取重要的路径相关和历史敏感信息，并实现二次加速。 |

# 详细

[^1]: 实践中的投资组合风险管理和对冲实施

    Implementing portfolio risk management and hedging in practice. (arXiv:2309.15767v1 [q-fin.PM])

    [http://arxiv.org/abs/2309.15767](http://arxiv.org/abs/2309.15767)

    本文介绍了一种实践中的跨资产投资组合风险管理和对冲方法，利用（近似）二次规划的凸优化设置，解决了连续时间随机控制框架的不可取之处，并特别关注了经济概念和数学表示之间的对应关系。

    

    在学术文献中，投资组合风险管理和对冲通常通过连续时间的随机控制和Hamilton--Jacobi--Bellman~(HJB)方程的语言来阐述。然而，在实践中，连续时间框架的随机控制可能由于各种商业原因而不可取。在本文中，我们提出了一种简单的方法来思考跨资产投资组合风险管理和对冲，提供了一些实施细节，同时很少涉足（近似）二次规划~(QP)的凸优化设置之外。我们特别关注经济概念与数学表示之间的对应关系；使我们能够同时处理多个资产类别和风险模型的抽象；产生方程的尺寸分析；以及我们推导中的假设。我们展示了如何使用CVXOPT求解所得到的QP问题。

    In academic literature portfolio risk management and hedging are often versed in the language of stochastic control and Hamilton--Jacobi--Bellman~(HJB) equations in continuous time. In practice the continuous-time framework of stochastic control may be undesirable for various business reasons. In this work we present a straightforward approach for thinking of cross-asset portfolio risk management and hedging, providing some implementation details, while rarely venturing outside the convex optimisation setting of (approximate) quadratic programming~(QP). We pay particular attention to the correspondence between the economic concepts and their mathematical representations; the abstractions enabling us to handle multiple asset classes and risk models at once; the dimensional analysis of the resulting equations; and the assumptions inherent in our derivations. We demonstrate how to solve the resulting QPs with CVXOPT.
    
[^2]: 用长短期记忆和时间序列模型进行算法投资策略对冲的对冲特性

    Hedging Properties of Algorithmic Investment Strategies using Long Short-Term Memory and Time Series models for Equity Indices. (arXiv:2309.15640v1 [q-fin.PM])

    [http://arxiv.org/abs/2309.15640](http://arxiv.org/abs/2309.15640)

    本文提出了一种使用长短期记忆和时间序列模型构建算法投资策略的对冲方法，并通过利用不同类型的投资策略来对冲风险资产组合。实证结果显示，该方法在金融市场的动荡时期具有多样化的潜力。

    

    本文提出了一种在金融市场受金融动荡影响时对冲风险资产组合的新方法。我们引入了一种全新的多元算法投资策略（AIS）的分散化方法，该方法不是在单个资产的级别上进行，而是在基于这些资产的价格的级别上进行。我们采用四种不同的理论模型（LSTM - 长短期记忆、ARIMA-GARCH - 自回归移动平均 - 广义自回归条件异方差、动量和反向交易）来生成价格预测，然后利用这些预测产生单个和复合的AIS的投资信号。通过这种方式，我们能够验证由各种资产（能源商品、贵金属、加密货币或软商品）组成的不同类型的投资策略在对冲用于股票指数（S&P 500指数）的组合AIS中的多样化潜力。

    This paper proposes a novel approach to hedging portfolios of risky assets when financial markets are affected by financial turmoils. We introduce a completely novel approach to diversification activity not on the level of single assets but on the level of ensemble algorithmic investment strategies (AIS) built based on the prices of these assets. We employ four types of diverse theoretical models (LSTM - Long Short-Term Memory, ARIMA-GARCH Autoregressive Integrated Moving Average - Generalized Autoregressive Conditional Heteroskedasticity, momentum, and contrarian) to generate price forecasts, which are then used to produce investment signals in single and complex AIS. In such a way, we are able to verify the diversification potential of different types of investment strategies consisting of various assets (energy commodities, precious metals, cryptocurrencies, or soft commodities) in hedging ensemble AIS built for equity indices (S&P 500 index). Empirical data used in this study cov
    
[^3]: 更好地理解实现生态系统服务: 供给、需求、流动和利用的集成分析框架

    To better understand realized ecosystem services: An integrated analysis framework of supply, demand, flow and use. (arXiv:2309.15574v1 [econ.GN])

    [http://arxiv.org/abs/2309.15574](http://arxiv.org/abs/2309.15574)

    该研究提出了一个供给-需求-流动-利用（SDFU）框架来理解实现生态系统服务（ES）。研究应用该框架分析了城市绿地公园中的野生浆果供给，传粉和娱乐等ES，揭示了ES的实际利用情况，以及供给受限、需求受限和供需平衡类型的ES。研究还讨论了ES的尺度特征、时间动态和空间特征，以及未来研究的关键问题。

    

    实现生态系统服务（ES）是社会实际利用ES的情况，与潜在ES相比，更直接关联到人类福祉。然而，目前缺乏一个通用的分析框架来理解实现了多少ES。在本研究中，我们首先提出了一个供给-需求-流动-利用（SDFU）的框架，将ES的供给、需求、流动和利用进行整合，并将这些概念区分为不同的方面（例如，潜在和实际的ES需求，供给的出口和进口流动等）。然后，我们将该框架应用于典型城市绿地公园中的三个ES的例子（即野生浆果供给，传粉和娱乐）。我们展示了该框架如何评估ES的实际利用情况，并识别出供给受限、需求受限和供需平衡类型的ES。我们还讨论了实现ES的尺度特征、时间动态和空间特征，以及未来研究中的一些关键问题。

    Realized ecosystem services (ES) are the actual use of ES by societies, which is more directly linked to human well-being than potential ES. However, there is a lack of a general analysis framework to understand how much ES was realized. In this study, we first proposed a Supply-Demand-Flow-Use (SDFU) framework that integrates the supply, demand, flow, and use of ES and differentiates these concepts into different aspects (e.g., potential vs. actual ES demand, export and import flows of supply, etc.). Then, we applied the framework to three examples of ES that can be found in typical urban green parks (i.e., wild berry supply, pollination, and recreation). We showed how the framework could assess the actual use of ES and identify the supply-limited, demand-limited, and supply-demand-balanced types of realized ES. We also discussed the scaling features, temporal dynamics, and spatial characteristics of realized ES, as well as some critical questions for future studies. Although facing c
    
[^4]: 使用CrunchBase数据预测创业公司成功和风险投资组合模拟

    Startup success prediction and VC portfolio simulation using CrunchBase data. (arXiv:2309.15552v1 [cs.LG])

    [http://arxiv.org/abs/2309.15552](http://arxiv.org/abs/2309.15552)

    本研究提出了一个使用CrunchBase数据来预测创业成功和模拟VC投资组合的新颖深度学习模型，并通过全面回溯算法对模型在历史数据上的表现进行了评估。

    

    预测创业公司的成功对于创业生态系统的不稳定性而言是一项巨大的挑战。借助CrunchBase等广泛数据库的出现，结合可用的开放数据，可以应用机器学习和人工智能进行更准确的预测分析。本文聚焦于创业公司在B轮和C轮投资阶段，旨在预测关键的成功里程碑，如实现首次公开募股（IPO），达到独角兽地位，或成功实施并购。我们提出了一种新颖的深度学习模型来预测创业公司的成功，整合了各种因素，如资金指标、创始人特征和行业类别。我们研究的一个独特特点是使用了一种全面的回溯算法来模拟风险投资的投资过程。这种模拟允许对我们模型的性能进行针对历史数据的强大评估。

    Predicting startup success presents a formidable challenge due to the inherently volatile landscape of the entrepreneurial ecosystem. The advent of extensive databases like Crunchbase jointly with available open data enables the application of machine learning and artificial intelligence for more accurate predictive analytics. This paper focuses on startups at their Series B and Series C investment stages, aiming to predict key success milestones such as achieving an Initial Public Offering (IPO), attaining unicorn status, or executing a successful Merger and Acquisition (M\&A). We introduce novel deep learning model for predicting startup success, integrating a variety of factors such as funding metrics, founder features, industry category. A distinctive feature of our research is the use of a comprehensive backtesting algorithm designed to simulate the venture capital investment process. This simulation allows for a robust evaluation of our model's performance against historical data
    
[^5]: 金融网络中的系统性风险: 渐近独立的影响

    Systemic risk in financial networks: the effects of asymptotic independence. (arXiv:2309.15511v1 [q-fin.RM])

    [http://arxiv.org/abs/2309.15511](http://arxiv.org/abs/2309.15511)

    该论文研究了金融网络中的系统性风险，特别关注了渐近尾部独立性的影响。通过建立一般维度的互相渐近尾部独立性概念，并与传统的成对渐近独立性进行比较，提供了金融风险建模的新视角。此外，通过构建银行和资产的双分图展示了资产组合的多样渐近尾部独立性行为，并提供了条件尾部风险的精确渐近表达式。

    

    系统性风险度量对于评估复杂金融系统的稳定性至关重要。经验证据表明，各种金融资产的回报呈重尾行为；此外，这些回报经常表现出渐近尾部独立性，即极端值不太可能同时发生。令人惊讶的是，大于两个维度的渐近尾部独立性在理论上和金融风险建模上都受到了有限的关注。在本文中，我们建立了一般 $d$ 维度的互相渐近尾部独立性概念，并将其与传统的成对渐近独立性概念进行比较。此外，我们使用银行和资产的双分图构建了一个金融网络模型，其中资产组合可能存在重尾风险资产的重叠，展示了不同的渐近尾部独立性行为。对于此类模型，我们提供了各种条件尾部风险的精确渐近表达式。

    Systemic risk measurements are important for the assessment of stability of complex financial systems. Empirical evidence indicates that returns from various financial assets have a heavy-tailed behavior; moreover, such returns often exhibit asymptotic tail independence, i.e., extreme values are less likely to occur simultaneously. Surprisingly, asymptotic tail independence in dimensions larger than two has received limited attention both theoretically, and as well for financial risk modeling. In this paper, we establish the notion of mutual asymptotic tail independence for general $d$-dimensions and compare it with the traditional notion of pairwise asymptotic independence. Furthermore, we consider a financial network model using a bipartite graph of banks and assets with portfolios of possibly overlapping heavy-tailed risky assets exhibiting various asymptotic tail (in)dependence behavior. For such models we provide precise asymptotic expressions for a variety of conditional tail ris
    
[^6]: 枚举关键矿物供应失衡对气候影响的论文

    Enumerating the climate impact of disequilibrium in critical mineral supply. (arXiv:2309.15368v1 [econ.GN])

    [http://arxiv.org/abs/2309.15368](http://arxiv.org/abs/2309.15368)

    根据研究，符合提议的尾气排放标准需要在2027年至2032年间用至少1021万辆新的内燃机汽车替换为EV。根据现有可利用的矿产储量，制造足够的EVs在大多数电池化学品中是可行的，并且可以减少高达4573万吨的CO2e。

    

    最近提出的尾气排放标准旨在在美国显著增加电动汽车（EV）的销量。我们的研究考察了在EV矿物供应链存在潜在限制的情况下，是否能够实现这一增长。我们估计了一个模型，反映了国际采购规则、主要电池化学品的矿物质强度的异质性，以及长期的电网脱碳努力。我们的努力得出了五个关键发现。首先，要符合提议的标准，需要在2027年至2032年之间将至少1021万辆新的内燃机汽车替换为EV。其次，基于经济上可行和地质上可利用的矿产储量，制造足够的EVs在大多数电池化学品中是可行的，并且可以根据所采用的化学品减少高达4573万吨的CO2e。第三，美国及其盟友的矿产生产能力限制了在2027年至2032年之间总共生产509万个EV电池。

    Recently proposed tailpipe emissions standards aim to significant increases in electric vehicle (EV) sales in the United States. Our work examines whether this increase is achievable given potential constraints in EV mineral supply chains. We estimate a model that reflects international sourcing rules, heterogeneity in the mineral intensity of predominant battery chemistries, and long-run grid decarbonization efforts. Our efforts yield five key findings. First, compliance with the proposed standard necessitates replacing at least 10.21 million new ICEVs with EVs between 2027 and 2032. Second, based on economically viable and geologically available mineral reserves, manufacturing sufficient EVs is plausible across most battery chemistries and could, subject to the chemistry leveraged, reduce up to 457.3 million total tons of CO2e. Third, mineral production capacities of the US and its allies constrain battery production to a total of 5.09 million EV batteries between 2027 and 2032, well
    
[^7]: 紧缩时期质量的重要性：大学竞争力和资助收入

    The importance of quality in austere times: University competitiveness and grant income. (arXiv:2309.15309v1 [econ.GN])

    [http://arxiv.org/abs/2309.15309](http://arxiv.org/abs/2309.15309)

    这项研究利用英国科学资助紧缩政策的自然实验，发现传统的大学竞争力测量指标无法准确反映竞争力。通过使用一种基于复杂性科学的替代指标，研究人员揭示了大学在科学领域的高度动态参与方式，并发现研究竞争力对资助收入具有影响。紧缩政策放宽后，英国大学的地位和资助收入发生了变化。

    

    在2009年之后，许多政府实施了紧缩措施，常常限制科学资助。这些限制是否进一步使资助收入向精英科学家和大学倾斜？增加的资金竞争是否削弱了参与度？英国科学资助机构在紧缩期响应中显著减少了资助数目和总资金，但令人惊讶的是在2015年大选之后，科学资助限制得到了放宽。利用这个自然实验，我们显示常规的大学竞争力衡量指标是竞争力的不良代理。从复杂性科学中得出的大学竞争力替代指标捕捉了大学在科学领域中的高度动态参与方式。基于2006年至2020年间的43,430项英国资助项目数据集，我们分析了英国大学的排名，并研究了研究竞争力对资助收入的影响。当紧缩政策于2015年放宽时，英国大学的地位和资助收入发生了变化。

    After 2009 many governments implemented austerity measures, often restricting science funding. Did such restrictions further skew grant income towards elite scientists and universities? And did increased competition for funding undermine participation? UK science funding agencies significantly reduced numbers of grants and total grant funding in response to austerity, but surprisingly restrictions of science funding were relaxed after the 2015 general election. Exploiting this natural experiment, we show that conventional measures of university competitiveness are poor proxies for competitiveness. An alternative measure of university competitiveness, drawn from complexity science, captures the highly dynamical way in which universities engage in scientific subjects. Building on a data set of 43,430 UK funded grants between 2006 and 2020, we analyse rankings of UK universities and investigate the effect of research competitiveness on grant income. When austerity was relaxed in 2015 the 
    
[^8]: 连续时间随机过程的量子分析

    Quantum Analysis of Continuous Time Stochastic Process. (arXiv:2208.02364v3 [quant-ph] UPDATED)

    [http://arxiv.org/abs/2208.02364](http://arxiv.org/abs/2208.02364)

    本文提出了一种通用框架，用于在量子计算机上高效地准备连续时间随机过程路径。基于压缩状态准备方法，可以高效地提取重要的路径相关和历史敏感信息，并实现二次加速。

    

    连续时间随机过程是一种主流的数学工具，用于模拟具有广泛应用的金融、统计、物理和时间序列分析中的随机世界，然而，连续时间随机过程的模拟和分析对于经典计算机来说是一个具有挑战性的问题。本文建立了一个通用框架，可以在量子计算机上高效地准备连续时间随机过程的路径。通过我们的压缩状态准备方法，关键参数存留时间的量子比特数和电路深度都得到了优化，存储和计算资源指数级降低。所需的信息，包括对金融问题至关重要的路径相关和历史敏感信息，可以从压缩取样路径中高效地提取，并且进一步实现了二次加速。此外，这种提取方法对于那些不连续的跳跃更加敏感。

    The continuous time stochastic process is a mainstream mathematical instrument modeling the random world with a wide range of applications involving finance, statistics, physics, and time series analysis, while the simulation and analysis of the continuous time stochastic process is a challenging problem for classical computers. In this work, a general framework is established to prepare the path of a continuous time stochastic process in a quantum computer efficiently. The storage and computation resource is exponentially reduced on the key parameter of holding time, as the qubit number and the circuit depth are both optimized via our compressed state preparation method. The desired information, including the path-dependent and history-sensitive information that is essential for financial problems, can be extracted efficiently from the compressed sampling path, and admits a further quadratic speed-up. Moreover, this extraction method is more sensitive to those discontinuous jumps capt
    

