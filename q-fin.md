# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning to Learn Financial Networks for Optimising Momentum Strategies.](http://arxiv.org/abs/2308.12212) | 这篇论文提出了一个端到端的机器学习框架L2GMOM，它可以同时学习金融网络和优化网络动量策略的交易信号，解决了传统方法依赖昂贵数据库和金融专业知识的问题，并提供了高度可解释的前向传播架构。 |
| [^2] | [Investigating Short-Term Dynamics in Green Bond Markets.](http://arxiv.org/abs/2308.12179) | 本文研究了绿色债券市场短期动态的影响。通过使用自激励过程和连续时间移动平均模型，研究了债券价格的高频动态，并考虑了价格上行和下行运动的交叉效应。实证结果表明，在与利率公告相关的时间段中，特别是对于能源市场中的发行人，存在差异。 |
| [^3] | [Retail Demand Forecasting: A Comparative Study for Multivariate Time Series.](http://arxiv.org/abs/2308.11939) | 本研究通过将客户需求的时间序列数据与宏观经济变量相结合，开发并比较了各种回归和机器学习模型，以准确预测零售需求。 |
| [^4] | [Discrimination and Constraints: Evidence from The Voice.](http://arxiv.org/abs/2308.11922) | 通过The Voice电视节目的盲选实验，发现雇佣过程中的性别偏见导致了劳动力市场的不平等，当选手是异性教练的接受者时，有更大的可能被选中。 |
| [^5] | [Optimal Robust Reinsurance with Multiple Insurers.](http://arxiv.org/abs/2308.11828) | 本研究探讨了一种面对多种模型不确定性的再保险商策略，其通过设计最大化预期财富的再保险合同，并在保险商模型的重心失真下定价，解决了连续时间的领导者-追随者博弈问题。 |
| [^6] | [The Impact of Stocks on Correlations of Crop Yields and Prices and on Revenue Insurance Premiums using Semiparametric Quantile Regression.](http://arxiv.org/abs/2308.11805) | 本文使用半参数分位回归研究了股票对作物产量和价格相关性以及收益保险费用的影响。通过惩罚B样条来估计储存条件下的联合分布，并通过模拟研究验证了该方法的有效性。应用该方法进行估计后发现玉米和大豆在美国的储存条件下具有相关性，并计算出相应的收益保险费用。 |
| [^7] | [Designing an attack-defense game: how to increase robustness of financial transaction models via a competition.](http://arxiv.org/abs/2308.11406) | 通过设计一款攻防游戏，我们研究了使用序列金融数据的神经网络模型的对抗攻击和防御的现状和动态，并且通过分析比赛动态，回答了隐藏模型免受恶意用户攻击的重要性以及需要多长时间才能破解模型的问题。 |
| [^8] | [Price Discovery for Derivatives.](http://arxiv.org/abs/2302.13426) | 本研究提供了一个基本理论，研究了带有高阶信息的期权市场中价格的发现机制。与此同时，该研究还以内幕交易的形式呈现了其中的特例，给出了通货膨胀需求、价格冲击和信息效率的闭式解。 |
| [^9] | [Mapping intra firm trade in the automotive sector: a network approach.](http://arxiv.org/abs/2202.00409) | 本研究提出了一种多层次方法，利用企业和国家层面的数据构建了一组汽车生产链各个环节的国家内部企业贸易网络。通过图形模式检测方法，提取了潜在的国家层面的企业内贸易联系。 |

# 详细

[^1]: 学习学习金融网络以优化动力策略

    Learning to Learn Financial Networks for Optimising Momentum Strategies. (arXiv:2308.12212v1 [q-fin.PM])

    [http://arxiv.org/abs/2308.12212](http://arxiv.org/abs/2308.12212)

    这篇论文提出了一个端到端的机器学习框架L2GMOM，它可以同时学习金融网络和优化网络动量策略的交易信号，解决了传统方法依赖昂贵数据库和金融专业知识的问题，并提供了高度可解释的前向传播架构。

    

    网络动量提供了一种新型的风险溢价，它利用金融网络中资产之间的相互关联来预测未来的回报。然而，目前构建金融网络的过程依赖于昂贵的数据库和金融专业知识，限制了小型和学术机构的可访问性。此外，传统方法将网络构建和投资组合优化视为单独的任务，可能会影响最优投资组合的表现。为了解决这些挑战，我们提出了L2GMOM，一个端到端的机器学习框架，可同时学习金融网络和优化网络动量策略的交易信号。L2GMOM模型是一个具有高度可解释前向传播架构的神经网络，它是从算法展开中推导出来的。L2GMOM具有灵活性，并可以使用不同的投资组合绩效损失函数进行训练，例如负夏普比率。在回测中。

    Network momentum provides a novel type of risk premium, which exploits the interconnections among assets in a financial network to predict future returns. However, the current process of constructing financial networks relies heavily on expensive databases and financial expertise, limiting accessibility for small-sized and academic institutions. Furthermore, the traditional approach treats network construction and portfolio optimisation as separate tasks, potentially hindering optimal portfolio performance. To address these challenges, we propose L2GMOM, an end-to-end machine learning framework that simultaneously learns financial networks and optimises trading signals for network momentum strategies. The model of L2GMOM is a neural network with a highly interpretable forward propagation architecture, which is derived from algorithm unrolling. The L2GMOM is flexible and can be trained with diverse loss functions for portfolio performance, e.g. the negative Sharpe ratio. Backtesting on 
    
[^2]: 研究绿色债券市场的短期动态

    Investigating Short-Term Dynamics in Green Bond Markets. (arXiv:2308.12179v1 [q-fin.TR])

    [http://arxiv.org/abs/2308.12179](http://arxiv.org/abs/2308.12179)

    本文研究了绿色债券市场短期动态的影响。通过使用自激励过程和连续时间移动平均模型，研究了债券价格的高频动态，并考虑了价格上行和下行运动的交叉效应。实证结果表明，在与利率公告相关的时间段中，特别是对于能源市场中的发行人，存在差异。

    

    本论文从交易活动的角度研究了债券市场中"绿色"标签的影响。研究认为，收益动态变化中的跳跃具有特定的记忆性质，可以通过自激励过程很好地表示。具体而言，我们使用霍克斯过程，并通过连续时间移动平均模型描述强度，研究债券价格的高频动态。我们还引入了模型的双变量扩展，处理价格上行和下行运动的交叉效应。实证结果表明，如果考虑与利率公告相关的时间段，尤其是在能源市场中运营的发行人的情况下，差异会出现。

    The paper investigates the effect of the label green in bond markets from the lens of the trading activity. The idea is that jumps in the dynamics of returns have a specific memory nature that can be well represented through a self-exciting process. Specifically, using Hawkes processes where the intensity is described through a continuous time moving average model, we study the high-frequency dynamics of bond prices. We also introduce a bivariate extension of the model that deals with the cross-effect of upward and downward price movements. Empirical results suggest that differences emerge if we consider periods with relevant interest rate announcements, especially in the case of an issuer operating in the energy market.
    
[^3]: 零售需求预测：多元时间序列的比较研究

    Retail Demand Forecasting: A Comparative Study for Multivariate Time Series. (arXiv:2308.11939v1 [cs.LG])

    [http://arxiv.org/abs/2308.11939](http://arxiv.org/abs/2308.11939)

    本研究通过将客户需求的时间序列数据与宏观经济变量相结合，开发并比较了各种回归和机器学习模型，以准确预测零售需求。

    

    在零售行业中，准确的需求预测是财务绩效和供应链效率的重要决定因素。随着全球市场日益互联互通，企业开始采用先进的预测模型来获取竞争优势。然而，现有文献主要关注历史销售数据，忽略了宏观经济条件对消费者支出行为的重要影响。在本研究中，我们通过将客户需求的时间序列数据与消费者物价指数（CPI）、消费者信心指数（ICS）和失业率等宏观经济变量相结合，弥补了这一差距。利用这个综合数据集，我们开发并比较了各种回归和机器学习模型，以准确预测零售需求。

    Accurate demand forecasting in the retail industry is a critical determinant of financial performance and supply chain efficiency. As global markets become increasingly interconnected, businesses are turning towards advanced prediction models to gain a competitive edge. However, existing literature mostly focuses on historical sales data and ignores the vital influence of macroeconomic conditions on consumer spending behavior. In this study, we bridge this gap by enriching time series data of customer demand with macroeconomic variables, such as the Consumer Price Index (CPI), Index of Consumer Sentiment (ICS), and unemployment rates. Leveraging this comprehensive dataset, we develop and compare various regression and machine learning models to predict retail demand accurately.
    
[^4]: 歧视和限制：来自The Voice的证据

    Discrimination and Constraints: Evidence from The Voice. (arXiv:2308.11922v1 [econ.GN])

    [http://arxiv.org/abs/2308.11922](http://arxiv.org/abs/2308.11922)

    通过The Voice电视节目的盲选实验，发现雇佣过程中的性别偏见导致了劳动力市场的不平等，当选手是异性教练的接受者时，有更大的可能被选中。

    

    性别歧视在雇佣过程中是导致劳动力市场不平等的一个重要因素。然而，关于雇佣经理人的性别偏见在这些不平等中起到了多大程度的作用，目前还没有太多证据。本文利用The Voice电视节目的独特数据集，将其作为一个实验，以识别选择过程中自身的性别偏见。在第一轮电视台面试中，有四位著名录音艺术家担任导师，他们会盲选（椅子背对舞台）以避免看到选手。通过差异法估计策略，可以证明教练（雇佣人）和艺术家的性别是外生的，我发现当选手是异性教练的接受者时，他们更有4.5个百分点（11％）的可能性被选中。我还使用Athey等人（2018）的机器学习方法，包括团队性别组成的异质性。

    Gender discrimination in the hiring process is one significant factor contributing to labor market disparities. However, there is little evidence on the extent to which gender bias by hiring managers is responsible for these disparities. In this paper, I exploit a unique dataset of blind auditions of The Voice television show as an experiment to identify own gender bias in the selection process. The first televised stage audition, in which four noteworthy recording artists are coaches, listens to the contestants blindly (chairs facing away from the stage) to avoid seeing the contestant. Using a difference-in-differences estimation strategy, a coach (hiring person) is demonstrably exogenous with respect to the artist's gender, I find that artists are 4.5 percentage points (11 percent) more likely to be selected when they are the recipients of an opposite-gender coach. I also utilize the machine-learning approach in Athey et al. (2018) to include heterogeneity from team gender compositio
    
[^5]: 多家保险商的最优鲁棒再保险研究

    Optimal Robust Reinsurance with Multiple Insurers. (arXiv:2308.11828v1 [q-fin.RM])

    [http://arxiv.org/abs/2308.11828](http://arxiv.org/abs/2308.11828)

    本研究探讨了一种面对多种模型不确定性的再保险商策略，其通过设计最大化预期财富的再保险合同，并在保险商模型的重心失真下定价，解决了连续时间的领导者-追随者博弈问题。

    

    我们研究了一家面临多个模型不确定性的再保险商。再保险商向$n$家保险商提供合同，其索赔遵循不同的复合泊松过程。由于再保险商对于保险商的索赔严重程度分布和频率存在不确定性，他们设计了最大化预期财富的再保险合同，同时还需要承担一个熵惩罚。而保险商则寻求在无歧义的情况下最大化他们的预期效用。我们解决了这个连续时间的领导者-追随者博弈问题，得到再保险商在保险商模型的重心失真下的定价方式。我们将这些结果应用到比例再保险和超额损失再保险合同，并进行了数值解析。此外，我们还解决了再保险商在模糊情况下最大化他们的预期效用的相关问题，并比较了解的不同。

    We study a reinsurer who faces multiple sources of model uncertainty. The reinsurer offers contracts to $n$ insurers whose claims follow different compound Poisson processes. As the reinsurer is uncertain about the insurers' claim severity distributions and frequencies, they design reinsurance contracts that maximise their expected wealth subject to an entropy penalty. Insurers meanwhile seek to maximise their expected utility without ambiguity. We solve this continuous-time Stackelberg game for general reinsurance contracts and find that the reinsurer prices under a distortion of the barycentre of the insurers' models. We apply our results to proportional reinsurance and excess-of-loss reinsurance contracts, and illustrate the solutions numerically. Furthermore, we solve the related problem where the reinsurer maximises, still under ambiguity, their expected utility and compare the solutions.
    
[^6]: 股票对作物产量和价格相关性以及收益保险费用的影响：使用半参数分位回归研究

    The Impact of Stocks on Correlations of Crop Yields and Prices and on Revenue Insurance Premiums using Semiparametric Quantile Regression. (arXiv:2308.11805v1 [econ.GN])

    [http://arxiv.org/abs/2308.11805](http://arxiv.org/abs/2308.11805)

    本文使用半参数分位回归研究了股票对作物产量和价格相关性以及收益保险费用的影响。通过惩罚B样条来估计储存条件下的联合分布，并通过模拟研究验证了该方法的有效性。应用该方法进行估计后发现玉米和大豆在美国的储存条件下具有相关性，并计算出相应的收益保险费用。

    

    作物产量和收获价格常常被认为是负相关的，因此通过稳定收益来作为自然风险管理对冲。储存理论认为，该相关性是从前几年的库存中得出的一个递增函数。储存条件下的二阶矩对于短缺期间的价格波动和对冲需求具有影响，而空间上变化的产量-价格相关结构对于谁从商品支持政策中受益具有影响。在本文中，我们提出使用惩罚B样条的半参数分位回归（SQR）来估计储存条件下的产量和价格的联合分布。通过全面的模拟研究验证的提出方法，允许使用SQR从真实的联合分布中抽样。然后，将其应用于估计美国玉米和大豆的储存条件下的相关性和收益保险费用。对于这两种作物，Cornbelt c

    Crop yields and harvest prices are often considered to be negatively correlated, thus acting as a natural risk management hedge through stabilizing revenues. Storage theory gives reason to believe that the correlation is an increasing function of stocks carried over from previous years. Stock-conditioned second moments have implications for price movements during shortages and for hedging needs, while spatially varying yield-price correlation structures have implications for who benefits from commodity support policies. In this paper, we propose to use semi-parametric quantile regression (SQR) with penalized B-splines to estimate a stock-conditioned joint distribution of yield and price. The proposed method, validated through a comprehensive simulation study, enables sampling from the true joint distribution using SQR. Then it is applied to approximate stock-conditioned correlation and revenue insurance premium for both corn and soybeans in the United States. For both crops, Cornbelt c
    
[^7]: 设计一款攻防游戏：通过竞争来增加金融交易模型的鲁棒性

    Designing an attack-defense game: how to increase robustness of financial transaction models via a competition. (arXiv:2308.11406v1 [cs.LG])

    [http://arxiv.org/abs/2308.11406](http://arxiv.org/abs/2308.11406)

    通过设计一款攻防游戏，我们研究了使用序列金融数据的神经网络模型的对抗攻击和防御的现状和动态，并且通过分析比赛动态，回答了隐藏模型免受恶意用户攻击的重要性以及需要多长时间才能破解模型的问题。

    

    鉴于金融领域恶意攻击风险不断升级和由此引发的严重损害，对机器学习模型的对抗策略和鲁棒的防御机制有深入的理解至关重要。随着银行日益广泛采用更精确但潜在脆弱的神经网络，这一威胁变得更加严重。我们旨在调查使用序列金融数据作为输入的神经网络模型的对抗攻击和防御的当前状态和动态。为了实现这一目标，我们设计了一个比赛，允许对现代金融交易数据中的问题进行逼真而详细的研究。参与者直接竞争，因此可能的攻击和防御在接近真实条件下进行了检验。我们的主要贡献是分析比赛动态，回答了隐藏模型免受恶意用户攻击的重要性以及需要多长时间才能破解模型的问题。

    Given the escalating risks of malicious attacks in the finance sector and the consequential severe damage, a thorough understanding of adversarial strategies and robust defense mechanisms for machine learning models is critical. The threat becomes even more severe with the increased adoption in banks more accurate, but potentially fragile neural networks. We aim to investigate the current state and dynamics of adversarial attacks and defenses for neural network models that use sequential financial data as the input.  To achieve this goal, we have designed a competition that allows realistic and detailed investigation of problems in modern financial transaction data. The participants compete directly against each other, so possible attacks and defenses are examined in close-to-real-life conditions. Our main contributions are the analysis of the competition dynamics that answers the questions on how important it is to conceal a model from malicious users, how long does it take to break i
    
[^8]: 期权的价格发现

    Price Discovery for Derivatives. (arXiv:2302.13426v5 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2302.13426](http://arxiv.org/abs/2302.13426)

    本研究提供了一个基本理论，研究了带有高阶信息的期权市场中价格的发现机制。与此同时，该研究还以内幕交易的形式呈现了其中的特例，给出了通货膨胀需求、价格冲击和信息效率的闭式解。

    

    本文通过一个模型，考虑了私有信息和高阶信息对期权市场价格的影响。模型允许有私有信息的交易者在状态-索赔集市场上交易。等价的期权形式下，我们考虑了拥有关于基础资产收益的分布的私有信息，并允许交易任意期权组合的操纵者。我们得出了通货膨胀需求、价格冲击和信息效率的闭式解，这些解提供了关于内幕交易的高阶信息，如任何给定的时刻交易期权策略，并将这些策略泛化到了波动率交易等实践领域。

    We obtain a basic theory of price discovery across derivative markets with respect to higher-order information, using a model where an agent with general private information regarding state probabilities is allowed to trade arbitrary portfolios of state-contingent claims. In an equivalent options formulation, the informed agent has private information regarding arbitrary aspects of the payoff distribution of an underlying asset and is allowed to trade arbitrary option portfolios. We characterize, in closed form, the informed demand, price impact, and information efficiency of prices. Our results offer a theory of insider trading on higher moments of the underlying payoff as a special case. The informed demand formula prescribes option strategies for trading on any given moment and extends those used in practice for, e.g. volatility trading.
    
[^9]: 在汽车行业中的企业内贸易的映射: 一种网络方法

    Mapping intra firm trade in the automotive sector: a network approach. (arXiv:2202.00409v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2202.00409](http://arxiv.org/abs/2202.00409)

    本研究提出了一种多层次方法，利用企业和国家层面的数据构建了一组汽车生产链各个环节的国家内部企业贸易网络。通过图形模式检测方法，提取了潜在的国家层面的企业内贸易联系。

    

    企业内贸易描述的是附属企业之间的贸易，随着全球生产的分散化，它变得越来越重要。然而，关于全球企业内贸易模式的统计数据普遍不可得。本研究提出了一种新颖的多层次方法，将企业和国家层面的数据结合起来，构建了一组汽车生产链各个环节的国家内部企业贸易网络。在宏观层面上构建了一种多层次网络，其中包括国际贸易网络、微观层面上的企业所有权网络以及将两者连接起来的企业-国家隶属网络。采用图形模式检测方法来筛选这些网络，提取潜在的国家内部企业贸易联系，其中模式（或子结构）是由贸易相连的两个国家，每个国家都附属于一个企业，而这两个企业又通过所有权联系在一起。图形模式检测用于提取潜在的国家层面的企业内贸易联系。

    Intra-firm trade describes the trade between affiliated firms and is increasingly important as global production is fragmented. However, statistics and data on global intra-firm trade patterns are widely unavailable. This study proposes a novel multilevel approach combining firm and country level data to construct a set of country intra-firm trade networks for various segments of the automotive production chain. A multilevel network is constructed with a network of international trade at the macro level, a firm ownership network at the micro level and a firm-country affiliation network linking the two, at the meso level. A motif detection approach is used to filter these networks to extract potential intra-firm trade ties between countries, where the motif (or substructure) is two countries linked by trade, each affiliated with a firm, and these two firms linked by ownership. The motif detection is used to extract potential country level intra-firm trade ties. An Exponential Random Gra
    

