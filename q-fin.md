# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Probability Distributions of Intraday Electricity Prices.](http://arxiv.org/abs/2310.02867) | 该论文提出了一种利用机器学习方法对电力日内价格概率进行预测的新方法，该方法通过学习数据中的经验分布选择最佳分布，并利用分布神经网络学习复杂模式，优于现有的基准模型。 |
| [^2] | [Resolving a Clearing Member's Default, A Radner Equilibrium Approach.](http://arxiv.org/abs/2310.02608) | 本研究通过使用Radner均衡方法，评估了清算成员违约的对冲和清算成本，并提供了相应的解析和数值解。这为中央对手方（CCP）在决定在哪个市场对冲和拍卖或清算违约的投资组合时提供了理性的决策依据。 |
| [^3] | [Bitcoin versus S&P 500 Index: Return and Risk Analysis.](http://arxiv.org/abs/2310.02436) | 该论文分析了比特币和标普500指数的回报分布，并评估了它们的尾部概率。研究发现，标普500回报呈现尖峰状分布，而比特币回报呈现重尾状分布。 |
| [^4] | [Signature Methods in Stochastic Portfolio Theory.](http://arxiv.org/abs/2310.02322) | 线性路径函数投资组合是一种通用的投资组合类别，可以通过签名投资组合来一致逼近市场权重的连续、可能路径相关的投资组合函数，并在多类非马尔科夫市场模型中任意好地逼近增长最优投资组合。 |
| [^5] | [The Price of Empire: Unrest Location and Sovereign Risk in Tsarist Russia.](http://arxiv.org/abs/2309.06885) | 该论文研究了政治动荡和主权风险对于地理辽阔的国家的影响，并发现在帝国边疆地区发生的动荡更容易增加风险。研究结果对于我们理解当前事件有启示，也提醒着我们在维护国家稳定与吸引外国投资方面所面临的挑战。 |
| [^6] | [Mean-field equilibrium price formation with exponential utility.](http://arxiv.org/abs/2304.07108) | 本文研究了多个投资者在初始财富、风险规避参数以及终止时间的随机负债方面存在差异时的均衡价格形成问题，通过一个新的均场反向随机微分方程（BSDE）的解来表征风险股票的均衡风险溢价过程，并证明其清除市场。 |
| [^7] | [Pricing cyber-insurance for systems via maturity models.](http://arxiv.org/abs/2302.04734) | 本篇论文提出了一种使用安全成熟度模型的方法，以评估组织的安全水平并确定网络保险的适当保费。 |

# 详细

[^1]: 学习电力日内价格概率分布

    Learning Probability Distributions of Intraday Electricity Prices. (arXiv:2310.02867v1 [econ.GN])

    [http://arxiv.org/abs/2310.02867](http://arxiv.org/abs/2310.02867)

    该论文提出了一种利用机器学习方法对电力日内价格概率进行预测的新方法，该方法通过学习数据中的经验分布选择最佳分布，并利用分布神经网络学习复杂模式，优于现有的基准模型。

    

    我们提出了一种新颖的机器学习方法，用于对小时级电力日内价格进行概率预测。与最近在数据丰富的概率预测方面的进展不同，该方法是非参数的，并从数据中学习到所有可能的经验分布中选择最佳分布。我们提出的模型是一种具有单调调整惩罚的多输出神经网络。这样的分布神经网络可以从数据丰富的环境中学习到电力价格的复杂模式，并且优于最先进的基准模型。

    We propose a novel machine learning approach to probabilistic forecasting of hourly intraday electricity prices. In contrast to recent advances in data-rich probabilistic forecasting that approximate the distributions with some features such as moments, our method is non-parametric and selects the best distribution from all possible empirical distributions learned from the data. The model we propose is a multiple output neural network with a monotonicity adjusting penalty. Such a distributional neural network can learn complex patterns in electricity prices from data-rich environments and it outperforms state-of-the-art benchmarks.
    
[^2]: 解决结算成员违约问题的Radner均衡方法

    Resolving a Clearing Member's Default, A Radner Equilibrium Approach. (arXiv:2310.02608v1 [q-fin.RM])

    [http://arxiv.org/abs/2310.02608](http://arxiv.org/abs/2310.02608)

    本研究通过使用Radner均衡方法，评估了清算成员违约的对冲和清算成本，并提供了相应的解析和数值解。这为中央对手方（CCP）在决定在哪个市场对冲和拍卖或清算违约的投资组合时提供了理性的决策依据。

    

    对于构成投资银行对冲投资组合主要部分的标准衍生品，通过中央对手方（CCP）进行中央清算已成为主导。CCP的关键任务之一是提供高效和适当的清算成员违约解决程序。当清算成员违约时，CCP可以对其头寸进行对冲和拍卖或清算。拍卖的对手方信用风险成本已经在Bastide、Crépey、Drapeau和Tadese（2023）中用XVA指标进行了分析。在这项工作中，我们评估对冲或清算的成本。通过比较违约前后的市场均衡，使用Radner均衡方法对投资组合配置和价格发现进行了研究。我们展示了Radner均衡的独特存在，并为椭圆分布市场中的后者提供了解析和数值解。使用这些工具，CCP可以理性地决定在哪个市场对冲和拍卖或清算违约的投资组合。

    For vanilla derivatives that constitute the bulk of investment banks' hedging portfolios, central clearing through central counterparties (CCPs) has become hegemonic. A key mandate of a CCP is to provide an efficient and proper clearing member default resolution procedure. When a clearing member defaults, the CCP can hedge and auction or liquidate its positions. The counterparty credit risk cost of auctioning has been analyzed in terms of XVA metrics in Bastide, Cr{\'e}pey, Drapeau, and Tadese (2023). In this work we assess the costs of hedging or liquidating. This is done by comparing pre- and post-default market equilibria, using a Radner equilibrium approach for portfolio allocation and price discovery in each case. We show that the Radner equilibria uniquely exist and we provide both analytical and numerical solutions for the latter in elliptically distributed markets. Using such tools, a CCP could decide rationally on which market to hedge and auction or liquidate defaulted portfo
    
[^3]: 比特币与标普500指数：回报和风险分析

    Bitcoin versus S&P 500 Index: Return and Risk Analysis. (arXiv:2310.02436v1 [q-fin.ST])

    [http://arxiv.org/abs/2310.02436](http://arxiv.org/abs/2310.02436)

    该论文分析了比特币和标普500指数的回报分布，并评估了它们的尾部概率。研究发现，标普500回报呈现尖峰状分布，而比特币回报呈现重尾状分布。

    

    标普500指数被认为是金融市场中最受欢迎的交易工具。随着加密货币在过去几年的崛起，比特币也在受到越来越多的关注和应用。该论文旨在分析比特币和标普500指数的日回报分布，并通过两个财务风险指标评估其尾部概率。作为方法论，我们使用比特币和标普500指数的日回报数据，通过将快速分数傅里叶（FRFT）算法与12点规则复合牛顿-科茨积分相结合的先进快速分数傅里叶变换（FRFT）方案来拟合七参数广义调和稳定（GTS）分布。研究结果表明，标普500回报分布的主要特点是尖峰状，而比特币回报分布的主要特点是重尾状。GTS分布显示，$80.05\%$的标普500回报在$-1.06\%$和$1.23\%$之间，而只有$40.32\%$的比特币回报在这个范围内。

    The S&P 500 index is considered the most popular trading instrument in financial markets. With the rise of cryptocurrencies over the past years, Bitcoin has also grown in popularity and adoption. The paper aims to analyze the daily return distribution of the Bitcoin and S&P 500 index and assess their tail probabilities through two financial risk measures. As a methodology, We use Bitcoin and S&P 500 Index daily return data to fit The seven-parameter General Tempered Stable (GTS) distribution using the advanced Fast Fractional Fourier transform (FRFT) scheme developed by combining the Fast Fractional Fourier (FRFT) algorithm and the 12-point rule Composite Newton-Cotes Quadrature. The findings show that peakedness is the main characteristic of the S&P 500 return distribution, whereas heavy-tailedness is the main characteristic of the Bitcoin return distribution. The GTS distribution shows that $80.05\%$ of S&P 500 returns are within $-1.06\%$ and $1.23\%$ against only $40.32\%$ of Bitco
    
[^4]: 随机投资组合理论中的签名方法

    Signature Methods in Stochastic Portfolio Theory. (arXiv:2310.02322v1 [q-fin.MF])

    [http://arxiv.org/abs/2310.02322](http://arxiv.org/abs/2310.02322)

    线性路径函数投资组合是一种通用的投资组合类别，可以通过签名投资组合来一致逼近市场权重的连续、可能路径相关的投资组合函数，并在多类非马尔科夫市场模型中任意好地逼近增长最优投资组合。

    

    在随机投资组合理论的背景下，我们引入了一种新颖的投资组合类别，称之为线性路径函数投资组合。这些投资组合是由某些线性函数的转化所确定的特征映射的非预测路径函数的集合。我们以市场权重的签名(排名)作为这些特征映射的主要示例。我们证明了这些投资组合在某种意义上是通用的，即市场权重的连续、可能路径相关的投资组合函数可以通过签名投资组合进行一致逼近。我们还展示了签名投资组合在几类非马尔科夫市场模型中可以任意好地逼近增长最优投资组合，并通过数值实验说明，训练得到的签名投资组合与理论增长最优投资组合非常接近。除了这些通用性特征之外，主要的数值优势 lies in the fact th...

    In the context of stochastic portfolio theory we introduce a novel class of portfolios which we call linear path-functional portfolios. These are portfolios which are determined by certain transformations of linear functions of a collections of feature maps that are non-anticipative path functionals of an underlying semimartingale. As main example for such feature maps we consider the signature of the (ranked) market weights. We prove that these portfolios are universal in the sense that every continuous, possibly path-dependent, portfolio function of the market weights can be uniformly approximated by signature portfolios. We also show that signature portfolios can approximate the growth-optimal portfolio in several classes of non-Markovian market models arbitrarily well and illustrate numerically that the trained signature portfolios are remarkably close to the theoretical growth-optimal portfolios. Besides these universality features, the main numerical advantage lies in the fact th
    
[^5]: 帝国的代价：沙俄动荡地点与主权风险

    The Price of Empire: Unrest Location and Sovereign Risk in Tsarist Russia. (arXiv:2309.06885v1 [econ.GN])

    [http://arxiv.org/abs/2309.06885](http://arxiv.org/abs/2309.06885)

    该论文研究了政治动荡和主权风险对于地理辽阔的国家的影响，并发现在帝国边疆地区发生的动荡更容易增加风险。研究结果对于我们理解当前事件有启示，也提醒着我们在维护国家稳定与吸引外国投资方面所面临的挑战。

    

    关于政治动荡和主权风险的研究忽视了动荡地点对于地理辽阔的国家主权风险的影响及其机制。在直观上，首都或附近的政治暴力似乎直接威胁到国家偿还债务的能力。然而，远离暴力地点可能会更加严重地影响政府，与抑制叛乱所带来的长期成本有关。我们利用沙俄的案例来评估俄罗斯国土内发生动荡与帝国边疆地区发生动荡时风险效应的差异。我们分析了1820年至1914年间沙俄帝国各地的动荡事件，发现动荡对帝国边疆地区的风险影响更大。与当前事件相呼应，我们发现乌克兰的动荡使风险增加最多。帝国的代价包括了向镇压动荡和获得外国投资者信任的同时维持力量投射的高额成本。

    Research on politically motivated unrest and sovereign risk overlooks whether and how unrest location matters for sovereign risk in geographically extensive states. Intuitively, political violence in the capital or nearby would seem to directly threaten the state's ability to pay its debts. However, it is possible that the effect on a government could be more pronounced the farther away the violence is, connected to the longer-term costs of suppressing rebellion. We use Tsarist Russia to assess these differences in risk effects when unrest occurs in Russian homeland territories versus more remote imperial territories. Our analysis of unrest events across the Russian imperium from 1820 to 1914 suggests that unrest increases risk more in imperial territories. Echoing current events, we find that unrest in Ukraine increases risk most. The price of empire included higher costs in projecting force to repress unrest and retain the confidence of the foreign investors financing those costs.
    
[^6]: 带指数效用函数的均值场均衡价格形成

    Mean-field equilibrium price formation with exponential utility. (arXiv:2304.07108v1 [q-fin.MF])

    [http://arxiv.org/abs/2304.07108](http://arxiv.org/abs/2304.07108)

    本文研究了多个投资者在初始财富、风险规避参数以及终止时间的随机负债方面存在差异时的均衡价格形成问题，通过一个新的均场反向随机微分方程（BSDE）的解来表征风险股票的均衡风险溢价过程，并证明其清除市场。

    

    本文研究了多位投资者在初始财富、风险规避参数以及终止时间的随机负债方面存在差异时的均衡价格形成问题。我们通过一个新的均场反向随机微分方程（BSDE）的解来表征风险股票的均衡风险溢价过程，其特征是驱动程序在随机积分和条件期望上都具有二次增长。我们证明了在多个条件下均场BSDE存在解，并且表明随着人口规模的增大，结果风险溢价进程实际上会清除市场。

    In this paper, we study a problem of equilibrium price formation among many investors with exponential utility. The investors are heterogeneous in their initial wealth, risk-averseness parameter, as well as stochastic liability at the terminal time. We characterize the equilibrium risk-premium process of the risky stocks in terms of the solution to a novel mean-field backward stochastic differential equation (BSDE), whose driver has quadratic growth both in the stochastic integrands and in their conditional expectations. We prove the existence of a solution to the mean-field BSDE under several conditions and show that the resultant risk-premium process actually clears the market in the large population limit.
    
[^7]: 基于成熟度模型的信息系统网络保险定价

    Pricing cyber-insurance for systems via maturity models. (arXiv:2302.04734v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2302.04734](http://arxiv.org/abs/2302.04734)

    本篇论文提出了一种使用安全成熟度模型的方法，以评估组织的安全水平并确定网络保险的适当保费。

    

    对于与信息技术系统相关的风险进行保险定价提出了一个综合的建议，结合运营管理、安全和经济学，提出了一个社会经济模型。该模型包括实体关系图、安全成熟度模型和经济模型，解决了一个长期以来的研究难题，即如何在设计和定价网络保险政策时捕捉组织结构。文中提出了一个新的挑战，即网络保险的数据历史有限，不能直接应用于其它险种，因此提出一个安全成熟度模型，以评估组织的安全水平并确定相应的保险费用。

    Pricing insurance for risks associated with information technology systems presents a complex modelling challenge, combining the disciplines of operations management, security, and economics. This work proposes a socioeconomic model for cyber-insurance decisions compromised of entity relationship diagrams, security maturity models, and economic models, addressing a long-standing research challenge of capturing organizational structure in the design and pricing of cyber-insurance policies. Insurance pricing is usually informed by the long experience insurance companies have of the magnitude and frequency of losses that arise in organizations based on their size, industry sector, and location. Consequently, their calculations of premia will start from a baseline determined by these considerations. A unique challenge of cyber-insurance is that data history is limited and not necessarily informative of future loss risk meaning that established actuarial methodology for other lines of insur
    

