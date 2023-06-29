# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Expectile Quadrangle and Applications.](http://arxiv.org/abs/2306.16351) | 该论文研究了期望值风险度量在基本风险四边形理论中的应用，并重点关注了包含期望值作为统计量和风险度量的四边形。 |
| [^2] | [A closed form model-free approximation for the Initial Margin of option portfolios.](http://arxiv.org/abs/2306.16346) | 该论文提出了一种闭式无模型逼近方法来计算期权组合的初始保证金。通过对VaR的近期逼近公式的推导，并考虑了神经SDE模型，取得了比传统方法更满意的结果。 |
| [^3] | [Continuous-Time q-learning for McKean-Vlasov Control Problems.](http://arxiv.org/abs/2306.16208) | 本文研究了连续时间q-learning在熵正则化强化学习框架下用于McKean-Vlasov控制问题，并揭示了两种不同的q函数的存在及其积分表示。 |
| [^4] | [Application of spin glass ideas in social sciences, economics and finance.](http://arxiv.org/abs/2306.16165) | 这篇论文探讨了自旋玻璃思想在社会科学、经济学和金融领域的应用，强调了经济系统的复杂性以及在理解和描述这种系统时的困难。 |
| [^5] | [Analysis of Indian foreign exchange markets: A Multifractal Detrended Fluctuation Analysis (MFDFA) approach.](http://arxiv.org/abs/2306.16162) | 本研究通过多重分形分析了印度外汇市场的汇率数据，并发现尾部的厚尾和长程相关性是导致多重分形特征的主要来源。 |
| [^6] | [Non-parametric online market regime detection and regime clustering for multidimensional and path-dependent data structures.](http://arxiv.org/abs/2306.15835) | 本研究提出了一种非参数的在线市场制度检测和制度聚类方法，适用于多维和路径依赖的数据结构。该方法利用基于路径空间的最大均值差异相似度度量进行路径样本检验，并优化了针对新进数据少的情况的反应速度。同时，该方法也适用于高维度、非马尔可夫以及自相关性的数据结构。 |
| [^7] | [Liquidity Premium and Liquidity-Adjusted Return and Volatility: illustrated with a Liquidity-Adjusted Mean Variance Framework and its Application on a Portfolio of Crypto Assets.](http://arxiv.org/abs/2306.15807) | 这项研究创建了创新技术来度量加密资产的流动性溢价，并开发了流动性调整的模型来提高投资组合的预测性能。 |
| [^8] | [The Shifting Attention of Political Leaders: Evidence from Two Centuries of Presidential Speeches.](http://arxiv.org/abs/2209.00540) | 本研究使用两个世纪的总统演讲数据，通过自然语言处理算法研究了总统政策优先事项的动态和决定因素。研究发现在1819年至2022年期间，总统的关注重点从军事干预和国家能力的发展逐渐转向建设实体资本和人力资本。总统的年龄和性别等特征预测了主要政策问题。这些发现拓展了我们对总统关注动态和塑造因素的理解。 |
| [^9] | [Planning ride-pooling services with detour restrictions for spatially heterogeneous demand: A multi-zone queuing network approach.](http://arxiv.org/abs/2208.02219) | 本研究提出了一个多区域排队网络模型来优化空间异质需求下的拼车服务设计，通过细分区域和避免明显绕行，实现了稳态拼车操作，并通过非线性规划解决方案优化了车辆部署、路径规划和再平衡操作。 |
| [^10] | [Racial Disparities in Debt Collection.](http://arxiv.org/abs/1910.02570) | 本文研究发现，无论控制了信用评分和其他相关信用属性，黑人和西班牙裔借款人比白人借款人更有可能遭受债务收集判决。这种判决差距在有大量发薪日贷款机构、没有收入的家庭占比高以及受过高等教育程度低的地区更为明显。缩小种族财富差距可以显著减少这种判决的种族差异。 |

# 详细

[^1]: 期望值四边形及其应用

    Expectile Quadrangle and Applications. (arXiv:2306.16351v1 [q-fin.RM])

    [http://arxiv.org/abs/2306.16351](http://arxiv.org/abs/2306.16351)

    该论文研究了期望值风险度量在基本风险四边形理论中的应用，并重点关注了包含期望值作为统计量和风险度量的四边形。

    

    本文探讨了在基本风险四边形（Fundamental Risk Quadrangle，简称FRQ）理论框架下的“期望风险度量”概念。根据FRQ理论，一个四边形由与随机变量相关的四个随机函数组成：“误差”、“遗憾”、“风险”和“偏差”。这些函数通过一种随机函数（称为“统计量”）相互关联。期望值作为一种风险度量，类似于VaR（分位数）和CVaR（超分位数），可用于风险管理。虽然基于VaR和CVaR统计的四边形已得到广泛使用，但本文专注于最近提出的基于期望值的四边形。本文旨在对这些期望值四边形的性质进行严格的研究，特别强调了一个包含期望值作为统计量和风险度量的四边形。

    The paper explores the concept of the \emph{expectile risk measure} within the framework of the Fundamental Risk Quadrangle (FRQ) theory. According to the FRQ theory, a quadrangle comprises four stochastic functions associated with a random variable: ``error'', ``regret'', ``risk'', and ``deviation''. These functions are interconnected through a stochastic function known as the ``statistic''. Expectile is a risk measure that, similar to VaR (quantile) and CVaR (superquantile), can be employed in risk management. While quadrangles based on VaR and CVaR statistics are well-established and widely used, the paper focuses on the recently proposed quadrangles based on expectile. The aim of this paper is to rigorously examine the properties of these Expectile Quadrangles, with particular emphasis on a quadrangle that encompasses expectile as both a statistic and a measure of risk.
    
[^2]: 期权组合的初始保证金的一种闭式无模型逼近

    A closed form model-free approximation for the Initial Margin of option portfolios. (arXiv:2306.16346v1 [q-fin.RM])

    [http://arxiv.org/abs/2306.16346](http://arxiv.org/abs/2306.16346)

    该论文提出了一种闭式无模型逼近方法来计算期权组合的初始保证金。通过对VaR的近期逼近公式的推导，并考虑了神经SDE模型，取得了比传统方法更满意的结果。

    

    中央清算交易对手方（CCP）在减轻交易所交易期权的交易对手风险中起着基础性的作用。CCP通过从其成员收取初始保证金来对自己的成员的投资组合的清算可能造成的损失进行覆盖。本文分析了计算期权初始保证金的行业最前沿技术，其核心组件通常基于VaR或预期损失风险度量。我们在无模型设置下推导出VaR的近期逼近公式。这种创新的公式具有良好的特性，在我们的数值实验中表现出比传统的基于滤波历史模拟的VaR更令人满意的性能。此外，我们考虑了[Cohen et al., arXiv:2202.07148, 2022]提出的归一化看涨期权定价的神经SDE模型，并在该模型中获得了VaR的准显式公式和短期VaR的闭式公式，由于其条件特性。

    Central clearing counterparty houses (CCPs) play a fundamental role in mitigating the counterparty risk for exchange traded options. CCPs cover for possible losses during the liquidation of a defaulting member's portfolio by collecting initial margins from their members. In this article we analyze the current state of the art in the industry for computing initial margins for options, whose core component is generally based on a VaR or Expected Shortfall risk measure. We derive an approximation formula for the VaR at short horizons in a model-free setting. This innovating formula has promising features and behaves in a much more satisfactory way than the classical Filtered Historical Simulation-based VaR in our numerical experiments. In addition, we consider the neural-SDE model for normalized call prices proposed by [Cohen et al., arXiv:2202.07148, 2022] and obtain a quasi-explicit formula for the VaR and a closed formula for the short term VaR in this model, due to its conditional aff
    
[^3]: 连续时间q-learning用于McKean-Vlasov控制问题

    Continuous-Time q-learning for McKean-Vlasov Control Problems. (arXiv:2306.16208v1 [cs.LG])

    [http://arxiv.org/abs/2306.16208](http://arxiv.org/abs/2306.16208)

    本文研究了连续时间q-learning在熵正则化强化学习框架下用于McKean-Vlasov控制问题，并揭示了两种不同的q函数的存在及其积分表示。

    

    本文研究了q-learning，在熵正则化强化学习框架下，用于连续时间的McKean-Vlasov控制问题。与Jia和Zhou（2022c）的单个代理控制问题不同，代理之间的均场相互作用使得q函数的定义更加复杂，我们揭示了自然产生两种不同q函数的情况：（i）被称为集成q函数（用$q$表示），作为Gu、Guo、Wei和Xu（2023）引入的集成Q函数的一阶近似，可以通过涉及测试策略的弱鞅条件进行学习；（ii）作为策略改进迭代中所使用的实质q函数（用$q_e$表示）。我们证明了这两个q函数在所有测试策略下通过积分表示相关联。基于集成q函数的弱鞅条件和我们提出的搜索方法，我们设计了算法来学习两个q函数以解决Mckean-Vlasov控制问题。

    This paper studies the q-learning, recently coined as the continuous-time counterpart of Q-learning by Jia and Zhou (2022c), for continuous time Mckean-Vlasov control problems in the setting of entropy-regularized reinforcement learning. In contrast to the single agent's control problem in Jia and Zhou (2022c), the mean-field interaction of agents render the definition of q-function more subtle, for which we reveal that two distinct q-functions naturally arise: (i) the integrated q-function (denoted by $q$) as the first-order approximation of the integrated Q-function introduced in Gu, Guo, Wei and Xu (2023) that can be learnt by a weak martingale condition involving test policies; and (ii) the essential q-function (denoted by $q_e$) that is employed in the policy improvement iterations. We show that two q-functions are related via an integral representation under all test policies. Based on the weak martingale condition of the integrated q-function and our proposed searching method of
    
[^4]: 自旋玻璃思想在社会科学、经济学和金融领域的应用

    Application of spin glass ideas in social sciences, economics and finance. (arXiv:2306.16165v1 [physics.soc-ph])

    [http://arxiv.org/abs/2306.16165](http://arxiv.org/abs/2306.16165)

    这篇论文探讨了自旋玻璃思想在社会科学、经济学和金融领域的应用，强调了经济系统的复杂性以及在理解和描述这种系统时的困难。

    

    经典经济学建立了一套基于代表性代理人思想的方法来预测明年的国内生产总值、通货膨胀和汇率等精确数字。然而，很少有人会不同意经济是一个复杂的系统，有着大量强烈异质性的不同类型单位（企业、银行、家庭、公共机构）和不同规模的交互作用单元。现在，经济学的主要问题正是这种大杂烩微观单位的新兴组织、合作和协调。把它们视为一个“代表性”企业或家庭显然会冒着舍弃精华的风险。如同我们从统计物理学中学到的那样，理解和描述这种新兴性质可能很困难。由于具有不同符号、异质性和非线性的反馈环路，宏观性质往往很难预料。特别是在这些情况下，这些情况一般都是无法预测的。

    Classical economics has developed an arsenal of methods, based on the idea of representative agents, to come up with precise numbers for next year's GDP, inflation and exchange rates, among (many) other things. Few, however, will disagree with the fact that the economy is a complex system, with a large number of strongly heterogeneous, interacting units of different types (firms, banks, households, public institutions) and different sizes.  Now, the main issue in economics is precisely the emergent organization, cooperation and coordination of such a motley crowd of micro-units. Treating them as a unique ``representative'' firm or household clearly risks throwing the baby with the bathwater. As we have learnt from statistical physics, understanding and characterizing such emergent properties can be difficult. Because of feedback loops of different signs, heterogeneities and non-linearities, the macro-properties are often hard to anticipate. In particular, these situations generically l
    
[^5]: 印度外汇市场的分析：基于多重分形去趋势波动分析（MFDFA）方法

    Analysis of Indian foreign exchange markets: A Multifractal Detrended Fluctuation Analysis (MFDFA) approach. (arXiv:2306.16162v1 [q-fin.ST])

    [http://arxiv.org/abs/2306.16162](http://arxiv.org/abs/2306.16162)

    本研究通过多重分形分析了印度外汇市场的汇率数据，并发现尾部的厚尾和长程相关性是导致多重分形特征的主要来源。

    

    本研究分析了印度卢比兑美元（USD）、英镑（GBP）、欧元（Euro）和日元（Yen）的日常外汇汇率在1999年1月6日至2018年7月24日期间的多重分形谱。我们观察到所有四种汇率的对数收益率时间序列都呈现出多重分形特征。接下来，我们研究了观察到的多重分形的来源。为此，我们通过两种方式对回报序列进行转换：a）随机洗牌原始对数回报时间序列；b）对未改变的序列应用相位随机化过程。我们的结果表明，在美元的情况下，多重分形主要来源于尾部较厚。对于英镑和欧元，我们发现观测值与概率分布的厚尾之间存在长程相关性，从而产生了观测到的多重分形特征；而在日元的情况下，多重分形的来源则是尾部较厚。

    The multifractal spectra of daily foreign exchange rates for US dollar (USD), the British Pound (GBP), the Euro (Euro) and the Japanese Yen (Yen) with respect to the Indian Rupee are analysed for the period 6th January 1999 to 24th July 2018. We observe that the time series of logarithmic returns of all the four exchange rates exhibit features of multifractality. Next, we research the source of the observed multifractality. For this, we transform the return series in two ways: a) We randomly shuffle the original time series of logarithmic returns and b) We apply the process of phase randomisation on the unchanged series. Our results indicate in the case of the US dollar the source of multifractality is mainly the fat tail. For the British Pound and the Euro, we see the long-range correlations between the observations and the thick tails of the probability distribution give rise to the observed multifractal features, while in the case of the Japanese Yen, the origin of the multifractal 
    
[^6]: 非参数的在线市场制度检测和制度聚类方法对于多维和路径依赖的数据结构

    Non-parametric online market regime detection and regime clustering for multidimensional and path-dependent data structures. (arXiv:2306.15835v1 [stat.ML])

    [http://arxiv.org/abs/2306.15835](http://arxiv.org/abs/2306.15835)

    本研究提出了一种非参数的在线市场制度检测和制度聚类方法，适用于多维和路径依赖的数据结构。该方法利用基于路径空间的最大均值差异相似度度量进行路径样本检验，并优化了针对新进数据少的情况的反应速度。同时，该方法也适用于高维度、非马尔可夫以及自相关性的数据结构。

    

    本研究提出了一种非参数的在线市场制度检测方法，用于多维数据结构，利用基于路径空间上的最大均值差异相似度度量的路径样本检验。该相似度度量已经在最近的小规模数据环境的生成模型中作为鉴别器进行了发展和应用，并在此进行了优化，以适应新进数据量特别少的情况，以加快反应速度。在同样的原则下，我们还提出了一种基于路径的制度聚类方法，扩展了我们之前的工作。所提出的制度聚类技术被设计为前期市场分析工具，可以识别出大致相似的市场活动期间，但新的结果同时适用于基于路径的、高维度的和非马尔可夫的设置，以及表现出自相关性的数据结构。

    In this work we present a non-parametric online market regime detection method for multidimensional data structures using a path-wise two-sample test derived from a maximum mean discrepancy-based similarity metric on path space that uses rough path signatures as a feature map. The latter similarity metric has been developed and applied as a discriminator in recent generative models for small data environments, and has been optimised here to the setting where the size of new incoming data is particularly small, for faster reactivity.  On the same principles, we also present a path-wise method for regime clustering which extends our previous work. The presented regime clustering techniques were designed as ex-ante market analysis tools that can identify periods of approximatively similar market activity, but the new results also apply to path-wise, high dimensional-, and to non-Markovian settings as well as to data structures that exhibit autocorrelation.  We demonstrate our clustering t
    
[^7]: 流动性溢价和流动性调整收益与波动性：以流动性调整的均值方差框架及其在加密资产投资组合上的应用为例

    Liquidity Premium and Liquidity-Adjusted Return and Volatility: illustrated with a Liquidity-Adjusted Mean Variance Framework and its Application on a Portfolio of Crypto Assets. (arXiv:2306.15807v1 [q-fin.PM])

    [http://arxiv.org/abs/2306.15807](http://arxiv.org/abs/2306.15807)

    这项研究创建了创新技术来度量加密资产的流动性溢价，并开发了流动性调整的模型来提高投资组合的预测性能。

    

    我们建立了创新的流动性溢价Beta度量方法，并应用于选定的加密资产，同时对个别资产的收益进行流动性调整并建模，以及对投资组合的波动性进行流动性调整和建模。在高流动性情况下，这两个模型都表现出较强的可预测性，这使得流动性调整的均值方差 (LAMV) 框架在投资组合表现上比正常的均值方差 (RMV) 框架具有明显的优势。

    We establish innovative measures of liquidity premium Beta on both asset and portfolio levels, and corresponding liquidity-adjusted return and volatility, for selected crypto assets. We develop liquidity-adjusted ARMA-GARCH/EGARCH representation to model the liquidity-adjusted return of individual assets, and liquidity-adjusted VECM/VAR-DCC/ADCC structure to model the liquidity-adjusted variance of portfolio. Both models exhibit improved predictability at high liquidity, which affords a liquidity-adjusted mean-variance (LAMV) framework a clear advantage over its regular mean variance (RMV) counterpart in portfolio performance.
    
[^8]: 政治领导人的关注转移：来自两个世纪总统演讲的证据

    The Shifting Attention of Political Leaders: Evidence from Two Centuries of Presidential Speeches. (arXiv:2209.00540v3 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2209.00540](http://arxiv.org/abs/2209.00540)

    本研究使用两个世纪的总统演讲数据，通过自然语言处理算法研究了总统政策优先事项的动态和决定因素。研究发现在1819年至2022年期间，总统的关注重点从军事干预和国家能力的发展逐渐转向建设实体资本和人力资本。总统的年龄和性别等特征预测了主要政策问题。这些发现拓展了我们对总统关注动态和塑造因素的理解。

    

    我们使用自然语言处理算法对10个拉丁美洲国家的两个世纪900多次总统演讲的新数据集进行研究，以研究总统政策优先事项的动态和决定因素。我们证明，大多数演讲内容可以用一组紧凑的政策问题来描述，其相对构成在1819年至2022年之间发生了缓慢但实质性的转变。总统的关注重点最初集中在军事干预和国家能力的发展上。关注逐渐转向通过投资于基础设施和公共服务来建设实体资本，并最终转向通过教育、卫生和社会安全网的投资来建设人力资本。我们确定了总统的年龄和性别等总统级特征如何预测主要政策问题。我们的发现提供了关于总统关注的动态和塑造因素的新见解，拓展了我们对这一问题的理解。

    We use natural-language-processing algorithms on a novel dataset of over 900 presidential speeches from ten Latin American countries spanning two centuries to study the dynamics and determinants of presidential policy priorities. We show that most speech content can be characterized by a compact set of policy issues whose relative composition exhibited slow yet substantial shifts over 1819-2022. Presidential attention initially centered on military interventions and the development of state capacity. Attention gradually evolved towards building physical capital through investments in infrastructure and public services and finally turned towards building human capital through investments in education, health, and social safety nets. We characterize the way in which president-level characteristics, like age and gender, predict the main policy issues. Our findings offer novel insights into the dynamics of presidential attention and the factors that shape it, expanding our understanding of
    
[^9]: 为空间异质需求规划限制不便绕行的拼车服务：基于多区域排队网络的方法

    Planning ride-pooling services with detour restrictions for spatially heterogeneous demand: A multi-zone queuing network approach. (arXiv:2208.02219v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2208.02219](http://arxiv.org/abs/2208.02219)

    本研究提出了一个多区域排队网络模型来优化空间异质需求下的拼车服务设计，通过细分区域和避免明显绕行，实现了稳态拼车操作，并通过非线性规划解决方案优化了车辆部署、路径规划和再平衡操作。

    

    本研究提出了一个用于服务异质需求的稳态拼车排队网络模型，并在此基础上优化拼车服务的设计。通过将研究区域划分为一组相对均匀的区域来解决空间异质性，并引入一组条件以避免乘客之间的明显绕行。然后，我们开发了一个广义的多区域排队网络模型，描述了车辆在每个区域内和相邻区域之间状态的转换，以及如何由闲置或部分占用的车辆为乘客提供服务。基于排队网络模型构建了一个大型方程系统，用于分析评估稳态系统性能。然后，我们制定了一个带约束的非线性规划问题，以优化拼车服务的设计，例如区域级车辆部署、车辆路径规划和车辆再平衡操作。一个定制的解决方案方法被提出来。

    This study presents a multi-zone queuing network model for steady-state ride-pooling operations that serve heterogeneous demand, and then builds upon this model to optimize the design of ride-pooling services. Spatial heterogeneity is addressed by partitioning the study region into a set of relatively homogeneous zones, and a set of criteria are imposed to avoid significant detours among matched passengers. A generalized multi-zone queuing network model is then developed to describe how vehicles' states transition within each zone and across neighboring zones, and how passengers are served by idle or partially occupied vehicles. A large system of equations is constructed based on the queuing network model to analytically evaluate steady-state system performance. Then, we formulate a constrained nonlinear program to optimize the design of ride-pooling services, such as zone-level vehicle deployment, vehicle routing paths, and vehicle rebalancing operations. A customized solution approac
    
[^10]: 债务收集中的种族差异

    Racial Disparities in Debt Collection. (arXiv:1910.02570v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/1910.02570](http://arxiv.org/abs/1910.02570)

    本文研究发现，无论控制了信用评分和其他相关信用属性，黑人和西班牙裔借款人比白人借款人更有可能遭受债务收集判决。这种判决差距在有大量发薪日贷款机构、没有收入的家庭占比高以及受过高等教育程度低的地区更为明显。缩小种族财富差距可以显著减少这种判决的种族差异。

    

    本文表明，即使在控制了信用评分和其他相关信用属性后，黑人和西班牙裔借款人比白人借款人更有可能遭受债务收集判决。在有大量发薪日贷款机构、没有收入的家庭占比高以及受过高等教育程度低的地区，种族间的判决差距更加明显。州级的种族歧视措施不能解释判断差距，社区级别的争议判决占比或有律师代表的案件之前的差异也不能解释判断差距。一个简单的计算估计表明，缩小种族财富差距可以显著减少债务收集判决的种族差异。

    This paper shows that black and Hispanic borrowers are 39% more likely to experience a debt collection judgment than white borrowers, even after controlling for credit scores and other relevant credit attributes. The racial gap in judgments is more pronounced in areas with a high density of payday lenders, a high share of income-less households, and low levels of tertiary education. State-level measures of racial discrimination cannot explain the judgment gap, nor can neighborhood-level differences in the previous share of contested judgments or cases with attorney representation. A back-of-the-envelope calculation suggests that closing the racial wealth gap could significantly reduce the racial disparity in debt collection judgments.
    

