# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A time-stepping deep gradient flow method for option pricing in (rough) diffusion models](https://arxiv.org/abs/2403.00746) | 提出了一种时间步进深度梯度流方法，用于处理（粗糙）扩散模型中的期权定价问题，保证了对大金额水平下期权价格的渐近行为和先验上下界。 |
| [^2] | [Dimensionality reduction techniques to support insider trading detection](https://arxiv.org/abs/2403.00707) | 提出了一种无监督机器学习方法，利用主成分分析和自动编码器作为降维技术，用于支持市场监控以识别潜在内幕交易活动。 |
| [^3] | [Modelling Global Fossil CO2 Emissions with a Lognormal Distribution: A Climate Policy Tool](https://arxiv.org/abs/2403.00653) | 本研究探索了使用对数正态分布作为框架来理解和预测CO2排放，以测试一个更简单的分布是否仍然可以为政策制定者提供有意义的见解。 |
| [^4] | [Assessing the Efficacy of Heuristic-Based Address Clustering for Bitcoin](https://arxiv.org/abs/2403.00523) | 该研究评估了基于启发式的地址聚类在比特币中的有效性，介绍了一种“聚类比率”指标来量化启发式实现的实体数量减少，为分析目的选择特定启发式提供了重要依据。 |
| [^5] | [Volatility-based strategy on Chinese equity index ETF options](https://arxiv.org/abs/2403.00474) | 该研究通过基于波动率的策略调整仓位和暴露，改进了基于中国股票指数ETF期权的交易模型，在市场波动中实现更大的上涨捕捉和更小的回撤。 |
| [^6] | [Idiosyncratic Risk, Government Debt and Inflation](https://arxiv.org/abs/2403.00471) | 公共债务扩张可能会提高自然利率并导致通货膨胀，特别是在活跃的货币政策下，持续升高的公共债务可能会使实现通货紧缩的最后一“英里”变得更加困难。 |
| [^7] | [ARED: Argentina Real Estate Dataset](https://arxiv.org/abs/2403.00273) | ARED是专为阿根廷市场设计的综合房地产价格预测数据集系列，尽管零版只包含短期信息，但展示了市场层面上的时间相关现象。 |
| [^8] | [Optimal positioning in derivative securities in incomplete markets](https://arxiv.org/abs/2403.00139) | 本文分析了在不完全市场中使用衍生成为静态对冲的最优问题，并通过变分方法找到了解析解，其中指出当每个基础资产都有普通期权可用时，最优解与Lipschitz映射的不动点相关联。 |
| [^9] | [Randomized Control in Performance Analysis and Empirical Asset Pricing](https://arxiv.org/abs/2403.00009) | 文章在实证资产定价和绩效评估中应用了随机对照技术，利用几何随机游走方法构建了符合投资者约束条件的随机投资组合，探索了因子溢价与绩效之间的关系。 |
| [^10] | [Local sensitivity analysis of heating degree day and cooling degree day temperature derivatives prices](https://arxiv.org/abs/2403.00006) | 研究了供暖日和制冷日温度衍生价格相对于去季节性温度微扰的局部敏感性，并通过连续时间自回归过程模型进行分析 |
| [^11] | [Limit Order Book Simulations: A Review](https://arxiv.org/abs/2402.17359) | 本综述研究了当前先进的各种限价订单簿（LOB）模拟模型，在方法学分类的基础上提供了流行风格事实的整体视图，重点研究了模型中的价格冲击现象。 |
| [^12] | [Firm Entry and Exit with Unbounded Productivity Growth](https://arxiv.org/abs/1910.14023) | 本文通过去除生产率的有界性假设，在更一般的情境下提供了平衡态存在的确切特征，并给出了企业规模分布具有幂律尾部的充分条件。 |
| [^13] | [Modeling the yield curve of Burundian bond market by parametric models.](http://arxiv.org/abs/2310.00321) | 本文研究了布隆迪债券市场的收益率曲线建模，并发现Nelson-Siegel模型是最佳选择。 |
| [^14] | [Global universal approximation of functional input maps on weighted spaces.](http://arxiv.org/abs/2306.03303) | 本文提出了功能性输入神经网络，可以在带权重空间上完成全局函数逼近。这一方法适用于连续函数的推广，还可用于路径空间函数的逼近，同时也可以逼近线性函数签名。 |

# 详细

[^1]: 一种针对（粗糙）扩散模型中期权定价的时间步进深度梯度流方法

    A time-stepping deep gradient flow method for option pricing in (rough) diffusion models

    [https://arxiv.org/abs/2403.00746](https://arxiv.org/abs/2403.00746)

    提出了一种时间步进深度梯度流方法，用于处理（粗糙）扩散模型中的期权定价问题，保证了对大金额水平下期权价格的渐近行为和先验上下界。

    

    我们开发了一种新颖的深度学习方法，用于在扩散模型中定价欧式期权，可以高效处理由于粗糙波动率模型的马尔可夫逼近而导致的高维问题。期权定价的偏微分方程被重新表述为能量最小化问题，该问题通过深度人工神经网络以时间步进的方式进行近似。所提出的方案符合期权价格在大金额水平上的渐近行为，并遵守期权价格的先验已知上下界。通过一系列数值示例评估了所提方法的准确性和效率，特别关注了提升Heston模型。

    arXiv:2403.00746v1 Announce Type: cross  Abstract: We develop a novel deep learning approach for pricing European options in diffusion models, that can efficiently handle high-dimensional problems resulting from Markovian approximations of rough volatility models. The option pricing partial differential equation is reformulated as an energy minimization problem, which is approximated in a time-stepping fashion by deep artificial neural networks. The proposed scheme respects the asymptotic behavior of option prices for large levels of moneyness, and adheres to a priori known bounds for option prices. The accuracy and efficiency of the proposed method is assessed in a series of numerical examples, with particular focus in the lifted Heston model.
    
[^2]: 降维技术用于支持内幕交易检测

    Dimensionality reduction techniques to support insider trading detection

    [https://arxiv.org/abs/2403.00707](https://arxiv.org/abs/2403.00707)

    提出了一种无监督机器学习方法，利用主成分分析和自动编码器作为降维技术，用于支持市场监控以识别潜在内幕交易活动。

    

    识别市场滥用是一项非常复杂的活动，需要分析大量复杂的数据集。我们提出了一种无监督机器学习方法，用于上下文异常检测，可以支持旨在识别潜在内幕交易活动的市场监控。该方法基于重建范式，采用主成分分析和自动编码器作为降维技术。该方法的唯一输入是每位投资者在对我们具有价格敏感事件（PSE）的资产上的交易位置。在确定与交易配置文件相关的重建错误后，我们会施加几个条件，以识别那些行为可疑的投资者，其行为可能涉及与PSE有关的内幕交易。作为案例研究，我们将我们的方法应用于围绕收购要约的意大利股票的投资者解析数据。

    arXiv:2403.00707v1 Announce Type: new  Abstract: Identification of market abuse is an extremely complicated activity that requires the analysis of large and complex datasets. We propose an unsupervised machine learning method for contextual anomaly detection, which allows to support market surveillance aimed at identifying potential insider trading activities. This method lies in the reconstruction-based paradigm and employs principal component analysis and autoencoders as dimensionality reduction techniques. The only input of this method is the trading position of each investor active on the asset for which we have a price sensitive event (PSE). After determining reconstruction errors related to the trading profiles, several conditions are imposed in order to identify investors whose behavior could be suspicious of insider trading related to the PSE. As a case study, we apply our method to investor resolved data of Italian stocks around takeover bids.
    
[^3]: 用对数正态分布模拟全球化石二氧化碳排放：一种气候政策工具

    Modelling Global Fossil CO2 Emissions with a Lognormal Distribution: A Climate Policy Tool

    [https://arxiv.org/abs/2403.00653](https://arxiv.org/abs/2403.00653)

    本研究探索了使用对数正态分布作为框架来理解和预测CO2排放，以测试一个更简单的分布是否仍然可以为政策制定者提供有意义的见解。

    

    二氧化碳（CO2）排放已经成为一个关键问题，对环境、人类健康和全球经济产生了深远影响。大气CO2水平的稳步增加主要是由人类活动，如燃烧化石燃料和森林砍伐所致，已成为气候变化及其相关灾难性影响的重要贡献者。为了应对这一紧迫挑战，需要协调全球努力，这需要深刻理解排放模式和趋势。本文探讨了统计建模，特别是对数正态分布，作为理解和预测CO2排放的框架。我们基于先前研究，提出了一种复杂的排放分布，并试图检验一个更简单的分布是否仍然可以为决策者提供有意义的见解。我们利用三个全面数据库的数据，并分析了六个候选分布。

    arXiv:2403.00653v1 Announce Type: new  Abstract: Carbon dioxide (CO2) emissions have emerged as a critical issue with profound impacts on the environment, human health, and the global economy. The steady increase in atmospheric CO2 levels, largely due to human activities such as burning fossil fuels and deforestation, has become a major contributor to climate change and its associated catastrophic effects. To tackle this pressing challenge, a coordinated global effort is needed, which necessitates a deep understanding of emissions patterns and trends. In this paper, we explore the use of statistical modelling, specifically the lognormal distribution, as a framework for comprehending and predicting CO2 emissions. We build on prior research that suggests a complex distribution of emissions and seek to test the hypothesis that a simpler distribution can still offer meaningful insights for policy-makers. We utilize data from three comprehensive databases and analyse six candidate distribut
    
[^4]: 评估基于启发式地址聚类在比特币中的有效性

    Assessing the Efficacy of Heuristic-Based Address Clustering for Bitcoin

    [https://arxiv.org/abs/2403.00523](https://arxiv.org/abs/2403.00523)

    该研究评估了基于启发式的地址聚类在比特币中的有效性，介绍了一种“聚类比率”指标来量化启发式实现的实体数量减少，为分析目的选择特定启发式提供了重要依据。

    

    在比特币区块链中探索交易涉及研究数亿个实体之间比特币的转移。然而，研究如此庞大数量的实体往往是不切实际且耗费资源的。因此，实体聚类往往是大多数分析研究的首要步骤。这一过程通常采用基于这些实体的实践和行为的启发式。在这项研究中，我们深入研究了两种广泛使用的启发式，同时介绍了四种新颖的启发式。我们的贡献包括引入“聚类比率”，这是一种旨在量化给定启发式所实现的实体数量减少的指标。评估这种减少比率在证明选择特定启发式进行分析目的时起着重要作用。鉴于比特币系统的动态性，其特点是不断增加

    arXiv:2403.00523v1 Announce Type: new  Abstract: Exploring transactions within the Bitcoin blockchain entails examining the transfer of bitcoins among several hundred million entities. However, it is often impractical and resource-consuming to study such a vast number of entities. Consequently, entity clustering serves as an initial step in most analytical studies. This process often employs heuristics grounded in the practices and behaviors of these entities. In this research, we delve into the examination of two widely used heuristics, alongside the introduction of four novel ones. Our contribution includes the introduction of the \textit{clustering ratio}, a metric designed to quantify the reduction in the number of entities achieved by a given heuristic. The assessment of this reduction ratio plays an important role in justifying the selection of a specific heuristic for analytical purposes. Given the dynamic nature of the Bitcoin system, characterized by a continuous increase in t
    
[^5]: 基于中国股票指数ETF期权的波动率策略

    Volatility-based strategy on Chinese equity index ETF options

    [https://arxiv.org/abs/2403.00474](https://arxiv.org/abs/2403.00474)

    该研究通过基于波动率的策略调整仓位和暴露，改进了基于中国股票指数ETF期权的交易模型，在市场波动中实现更大的上涨捕捉和更小的回撤。

    

    最近几年，中国衍生品市场得到了快速发展，标准化衍生品交易量已达到相当规模。本研究收集了上海证券交易所交易的所有ETF期权的每日数据，并从一个简单的短波动率策略开始。该策略在2018年之前表现良好，提供了显著超额回报，超过了买入持有基准。然而，2018年之后，该策略开始恶化，没有显示明显的风险调整回报。根据策略表现与市场波动性之间的关系讨论，我们通过根据波动率预测调整仓位和暴露的方法（如波动率动能和GARCH）改进了模型。新模型在不同方面表现得更好，可以在市场波动中实现更大的上涨捕捉和更小的回撤。这项研究展示了潜力。

    arXiv:2403.00474v1 Announce Type: new  Abstract: In recent years, there has been quick developments of derivative markets in China and standardized derivative trading have reached considerable volumes. In this research, we collect all the daily data of ETF options traded at Shanghai Stock Exchange and start with a simple short-volatility strategy. The strategy delivers nice performance before 2018, providing significant excess return over the buy and hold benchmark. However, after 2018, this strategy starts to deteriorate and no obvious risk-adjusted return is shown. Based on the discussion of relationship between the strategy's performance and market's volatility, we improve the model by adjusting positions and exposure according to volatility forecasts using methods such as volatility momentum and GARCH. The new models have improved performance in different ways, where larger upside capture and smaller drawbacks can be achieved in market fluctuation. This research has shown potential
    
[^6]: 特质风险、政府债务和通货膨胀

    Idiosyncratic Risk, Government Debt and Inflation

    [https://arxiv.org/abs/2403.00471](https://arxiv.org/abs/2403.00471)

    公共债务扩张可能会提高自然利率并导致通货膨胀，特别是在活跃的货币政策下，持续升高的公共债务可能会使实现通货紧缩的最后一“英里”变得更加困难。

    

    公共债务对价格稳定有何重要性？如果私营部门为了保险特质风险而有用，政府债务扩张可能会提高自然利率并导致通货膨胀。正如我在一个易处理的模型中展示的那样，这在存在积极的泰勒规则的情况下成立，并且并不需要未来财政巩固的缺席。进一步使用一个完整的2资产HANK模型进行分析，揭示了这一机制的定量影响在很大程度上取决于资产市场的结构：在标准假设下，公共债务对自然利率的影响要么过于强大，要么过于弱。采用简明的方法来克服这个问题，我的框架表明公共债务对活跃的货币政策下通货膨胀产生相关影响：特别是，持续升高的公共债务可能使实现通货紧缩的最后一“英里”变得更加困难，除非央行明确考虑其影响。

    arXiv:2403.00471v1 Announce Type: new  Abstract: How does public debt matter for price stability? If it is useful for the private sector to insure idiosyncratic risk, government debt expansions can increase the natural rate of interest and create inflation. As I demonstrate using a tractable model, this holds in the presence of an active Taylor rule and does not require the absence of future fiscal consolidation. Further analysis using a full-blown 2-asset HANK model reveals the quantitative magnitude of the mechanism to crucially depend on the structure of the asset market: under standard assumptions, the effect of public debt on the natural rate is either overly strong or overly weak. Employing a parsimonious way to overcome this issue, my framework suggests relevant effects of public debt on inflation under active monetary policy: In particular, persistently elevated public debt may make it harder to go the last "mile of disinflation" unless central banks explicitly take its effect 
    
[^7]: ARED: 阿根廷房地产数据集

    ARED: Argentina Real Estate Dataset

    [https://arxiv.org/abs/2403.00273](https://arxiv.org/abs/2403.00273)

    ARED是专为阿根廷市场设计的综合房地产价格预测数据集系列，尽管零版只包含短期信息，但展示了市场层面上的时间相关现象。

    

    阿根廷房地产市场是一个独特的案例研究，其特点是过去几十年间不稳定且迅速变化的宏观经济环境。尽管存在一些用于价格预测的数据集，但缺乏专门针对阿根廷的混合多模态数据集。本文介绍了ARED的第一版，这是专为阿根廷市场设计的综合房地产价格预测数据集系列。这个版本仅包含2024年1月至2月的信息。尽管这个零版只捕获了短短的时间范围（44天），但时间相关的现象主要发生在市场层面上（整个市场）。然而，未来版本的数据集很可能会包含历史数据。ARED中的每个列表都包含描述性特征和可变长度的图像集。

    arXiv:2403.00273v1 Announce Type: new  Abstract: The Argentinian real estate market presents a unique case study characterized by its unstable and rapidly shifting macroeconomic circumstances over the past decades. Despite the existence of a few datasets for price prediction, there is a lack of mixed modality datasets specifically focused on Argentina. In this paper, the first edition of ARED is introduced. A comprehensive real estate price prediction dataset series, designed for the Argentinian market. This edition contains information solely for Jan-Feb 2024. It was found that despite the short time range captured by this zeroth edition (44 days), time dependent phenomena has been occurring mostly on a market level (market as a whole). Nevertheless future editions of this dataset, will most likely contain historical data. Each listing in ARED comprises descriptive features, and variable-length sets of images.
    
[^8]: 不完全市场中衍生证券的最优定位

    Optimal positioning in derivative securities in incomplete markets

    [https://arxiv.org/abs/2403.00139](https://arxiv.org/abs/2403.00139)

    本文分析了在不完全市场中使用衍生成为静态对冲的最优问题，并通过变分方法找到了解析解，其中指出当每个基础资产都有普通期权可用时，最优解与Lipschitz映射的不动点相关联。

    

    本文分析了在不完全市场中使用衍生成为静态对冲的最优问题。投资者被假设对两个基础资产有风险敞口。对冲工具为写在单个基础资产上的普通期权。对冲问题被规定为一个效用最大化问题，通过该问题确定最优静态对冲的形式。在我们的结果中，通过指数、幂/对数和二次效用的变分方法找到了最优解的半解析解。当每个基础资产都有普通期权可用时，最优解与Lipschitz映射的不动点相关联。在指数效用的情况下，只有一个这样的不动点，并且映射的后续迭代收敛于它。

    arXiv:2403.00139v1 Announce Type: new  Abstract: This paper analyzes a problem of optimal static hedging using derivatives in incomplete markets. The investor is assumed to have a risk exposure to two underlying assets. The hedging instruments are vanilla options written on a single underlying asset. The hedging problem is formulated as a utility maximization problem whereby the form of the optimal static hedge is determined. Among our results, a semi-analytical solution for the optimizer is found through variational methods for exponential, power/logarithmic, and quadratic utility. When vanilla options are available for each underlying asset, the optimal solution is related to the fixed points of a Lipschitz map. In the case of exponential utility, there is only one such fixed point, and subsequent iterations of the map converge to it.
    
[^9]: 随机对照在绩效分析和实证资产定价中的应用

    Randomized Control in Performance Analysis and Empirical Asset Pricing

    [https://arxiv.org/abs/2403.00009](https://arxiv.org/abs/2403.00009)

    文章在实证资产定价和绩效评估中应用了随机对照技术，利用几何随机游走方法构建了符合投资者约束条件的随机投资组合，探索了因子溢价与绩效之间的关系。

    

    本文探讨了随机对照技术在实证资产定价和绩效评估中的应用。它介绍了几何随机游走，一种马尔可夫链蒙特卡洛方法的范例，以构建灵活的控制组，即符合投资者约束条件的随机投资组合。基于抽样的方法使得能够在实践环境中探索学术研究的因子溢价与绩效之间的关系。在一个实证应用中，本研究评估了在强烈约束设置下捕获与MSCI Diversified Multifactor指数的投资者指南所示相对应的规模、价值、质量和动量溢价的潜力。此外，本文强调了传统随机投资组合在绩效评估中推断的问题，展示了与高维几何复杂性相关的挑战。

    arXiv:2403.00009v1 Announce Type: new  Abstract: The present article explores the application of randomized control techniques in empirical asset pricing and performance evaluation. It introduces geometric random walks, a class of Markov chain Monte Carlo methods, to construct flexible control groups in the form of random portfolios adhering to investor constraints. The sampling-based methods enable an exploration of the relationship between academically studied factor premia and performance in a practical setting. In an empirical application, the study assesses the potential to capture premias associated with size, value, quality, and momentum within a strongly constrained setup, exemplified by the investor guidelines of the MSCI Diversified Multifactor index. Additionally, the article highlights issues with the more traditional use case of random portfolios for drawing inferences in performance evaluation, showcasing challenges related to the intricacies of high-dimensional geometry.
    
[^10]: 对供暖日和制冷日温度衍生价格的局部敏感性分析

    Local sensitivity analysis of heating degree day and cooling degree day temperature derivatives prices

    [https://arxiv.org/abs/2403.00006](https://arxiv.org/abs/2403.00006)

    研究了供暖日和制冷日温度衍生价格相对于去季节性温度微扰的局部敏感性，并通过连续时间自回归过程模型进行分析

    

    我们研究了采暖日（HDD）和冷却日（CDD）温度期货和期权价格对去季节性温度或其衍生物之一的微扰的局部敏感性，该微扰的阶数由连续时间自回归过程确定，该过程对HDD和CDD指数中去季节性温度进行建模。我们还考虑了一个实证案例，其中将一个自回归阶数为3的CAR过程拟合到纽约温度，并对这些金融合约的局部敏感性进行研究，然后进行结果的后续分析。

    arXiv:2403.00006v1 Announce Type: new  Abstract: We study the local sensitivity of heating degree day (HDD) and cooling degree day (CDD) temperature futures and option prices with respect to perturbations in the deseasonalized temperature or in one of its derivatives up to a certain order determined by the continuous-time autoregressive process modelling the deseasonalized temperature in the HDD and CDD indexes. We also consider an empirical case where a CAR process of autoregressive order 3 is fitted to New York temperatures and we perform a study of the local sensitivity of these financial contracts and a posterior analysis of the results.
    
[^11]: 限价订单簿模拟：一项综述

    Limit Order Book Simulations: A Review

    [https://arxiv.org/abs/2402.17359](https://arxiv.org/abs/2402.17359)

    本综述研究了当前先进的各种限价订单簿（LOB）模拟模型，在方法学分类的基础上提供了流行风格事实的整体视图，重点研究了模型中的价格冲击现象。

    

    限价订单簿（LOBs）作为买家和卖家在金融市场中相互交互的机制。对LOB进行建模和模拟通常是校准和微调算法交易研究中开发的自动交易策略时的必要步骤。近年来，人工智能革命和更快、更便宜的计算能力的可用性使得建模和模拟变得更加丰富，甚至使用现代人工智能技术。在这项综述中，我们考察了当前最先进的各种LOB模拟模型。我们在方法论基础上对这些模型进行分类，并提供了文献中用于测试模型的流行风格事实的整体视图。此外，我们重点研究模型中价格冲击的存在，因为这是算法交易中一个更为关键的现象之一。最后，我们进行了一项比较研究。

    arXiv:2402.17359v1 Announce Type: new  Abstract: Limit Order Books (LOBs) serve as a mechanism for buyers and sellers to interact with each other in the financial markets. Modelling and simulating LOBs is quite often necessary} for calibrating and fine-tuning the automated trading strategies developed in algorithmic trading research. The recent AI revolution and availability of faster and cheaper compute power has enabled the modelling and simulations to grow richer and even use modern AI techniques. In this review we \highlight{examine} the various kinds of LOB simulation models present in the current state of the art. We provide a classification of the models on the basis of their methodology and provide an aggregate view of the popular stylized facts used in the literature to test the models. We additionally provide a focused study of price impact's presence in the models since it is one of the more crucial phenomena to model in algorithmic trading. Finally, we conduct a comparative
    
[^12]: 具有无界生产率增长的企业进入与退出

    Firm Entry and Exit with Unbounded Productivity Growth

    [https://arxiv.org/abs/1910.14023](https://arxiv.org/abs/1910.14023)

    本文通过去除生产率的有界性假设，在更一般的情境下提供了平衡态存在的确切特征，并给出了企业规模分布具有幂律尾部的充分条件。

    

    在Hopenhayn（1992）的进入退出模型中，生产率是有界的，这意味着预测的企业规模分布无法与数据中可观测到的幂律尾部相匹配。在本文中，我们去除了有界性假设，并在这种更一般的情境下，对平衡态存在的确切特征提供了描述，以及基于将生产视为Lyapunov函数的存在新的充分条件。我们还提供了进入率和总供给的新表示。最后，我们证明在非常广泛的生产率增长设定下，企业规模分布具有幂律尾部。

    arXiv:1910.14023v2 Announce Type: replace  Abstract: In Hopenhayn's (1992) entry-exit model productivity is bounded, implying that the predicted firm size distribution cannot match the power law tail observable in the data. In this paper we remove the boundedness assumption and, in this more general setting, provide an exact characterization of existence of stationary equilibria, as well as a novel sufficient condition for existence based on treating production as a Lyapunov function. We also provide new representations of the rate of entry and aggregate supply. Finally, we prove that the firm size distribution has a power law tail under a very broad set of productivity growth specifications.
    
[^13]: 使用参数模型对布隆迪债券市场的收益率曲线建模

    Modeling the yield curve of Burundian bond market by parametric models. (arXiv:2310.00321v1 [q-fin.GN])

    [http://arxiv.org/abs/2310.00321](http://arxiv.org/abs/2310.00321)

    本文研究了布隆迪债券市场的收益率曲线建模，并发现Nelson-Siegel模型是最佳选择。

    

    利率期限结构（收益率曲线）是金融分析中的重要组成部分，影响各种投资和风险管理决策。它被中央银行用于实施和监控货币政策。该工具反映了投资者对通胀预期和风险的看法。收益率曲线上报的利率是所有资产估值的基石。为了为布隆迪的金融市场提供这样的工具，我们从布隆迪央行的网站收集了国库证券的拍卖报告。然后，我们计算了零息票利率，并应用Nelson-Siegel和Svensson模型估计了保险精算利率。本文对这两个主要参数化收益率曲线模型进行了严格的比较分析，并发现Nelson-Siegel模型是建模布隆迪收益率曲线的最佳选择。这些研究结果对于收益率曲线建模的知识体系有所贡献，提升了其精确性和适用性。

    The term structure of interest rates (yield curve) is a critical facet of financial analytics, impacting various investment and risk management decisions. It is used by the central bank to conduct and monitor its monetary policy. That instrument reflects the anticipation of inflation and the risk by investors. The rates reported on yield curve are the cornerstone of valuation of all assets. To provide such tool for Burundi financial market, we collected the auction reports of treasury securities from the website of the Central Bank of Burundi. Then, we computed the zero-coupon rates, and estimated actuarial rates of return by applying the Nelson-Siegel and Svensson models. This paper conducts a rigorous comparative analysis of these two prominent parametric yield curve models and finds that the Nelson-Siegel model is the optimal choice for modeling the Burundian yield curve. The findings contribute to the body of knowledge on yield curve modeling, enhancing its precision and applicabil
    
[^14]: 带权重空间上功能性输入映射的全局普适逼近

    Global universal approximation of functional input maps on weighted spaces. (arXiv:2306.03303v1 [stat.ML])

    [http://arxiv.org/abs/2306.03303](http://arxiv.org/abs/2306.03303)

    本文提出了功能性输入神经网络，可以在带权重空间上完成全局函数逼近。这一方法适用于连续函数的推广，还可用于路径空间函数的逼近，同时也可以逼近线性函数签名。

    

    我们引入了所谓的功能性输入神经网络，定义在可能是无限维带权重空间上，其值也在可能是无限维的输出空间中。为此，我们使用一个加性族作为隐藏层映射，以及一个非线性激活函数应用于每个隐藏层。依靠带权重空间上的Stone-Weierstrass定理，我们可以证明连续函数的推广的全局普适逼近结果，超越了常规紧集逼近。这特别适用于通过功能性输入神经网络逼近（非先见之明的）路径空间函数。作为带权Stone-Weierstrass定理的进一步应用，我们证明了线性函数签名的全局普适逼近结果。我们还在这个设置中引入了高斯过程回归的观点，并展示了签名内核的再生核希尔伯特空间是某些高斯过程的Cameron-Martin空间。

    We introduce so-called functional input neural networks defined on a possibly infinite dimensional weighted space with values also in a possibly infinite dimensional output space. To this end, we use an additive family as hidden layer maps and a non-linear activation function applied to each hidden layer. Relying on Stone-Weierstrass theorems on weighted spaces, we can prove a global universal approximation result for generalizations of continuous functions going beyond the usual approximation on compact sets. This then applies in particular to approximation of (non-anticipative) path space functionals via functional input neural networks. As a further application of the weighted Stone-Weierstrass theorem we prove a global universal approximation result for linear functions of the signature. We also introduce the viewpoint of Gaussian process regression in this setting and show that the reproducing kernel Hilbert space of the signature kernels are Cameron-Martin spaces of certain Gauss
    

