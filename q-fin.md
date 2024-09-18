# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Crypto Inverse-Power Options and Fractional Stochastic Volatility](https://arxiv.org/abs/2403.16006) | 本文介绍了一个包含分数随机波动性的解析模型框架，特别关注逆期权，为减轻加密货币汇率风险和调整风险敞口提供了功率类型的泛化逆期权以及相应的定价-套期保值公式。 |
| [^2] | [Regime-Aware Asset Allocation: a Statistical Jump Model Approach](https://arxiv.org/abs/2402.05272) | 本文研究了制度转换对资产配置决策的影响，并比较了不同的制度识别模型。通过使用统计跳跃模型，作者开发了一种决策制度感知的资产配置策略，通过优化跳跃模型的性能指标，并在实证分析中展示了该策略相对于传统方法的表现优势。 |
| [^3] | [The law of one price in quadratic hedging and mean-variance portfolio selection.](http://arxiv.org/abs/2210.15613) | 本文证明了一价定律是一个明确均值方差组合配置框架而不会退化的最小条件，并且发现了一种新的机制，用于解释连续时间下一价定律的违规情况。 |

# 详细

[^1]: 加密逆功率期权与分数随机波动性

    Crypto Inverse-Power Options and Fractional Stochastic Volatility

    [https://arxiv.org/abs/2403.16006](https://arxiv.org/abs/2403.16006)

    本文介绍了一个包含分数随机波动性的解析模型框架，特别关注逆期权，为减轻加密货币汇率风险和调整风险敞口提供了功率类型的泛化逆期权以及相应的定价-套期保值公式。

    

    最近的经验性证据突显了在加密货币市场中价格和波动率中跃升所起到的关键作用。在本文中，我们介绍了一个分数随机波动性的解析模型框架，包括价格-波动率共跃和短期波动率依赖性。我们特别关注逆期权，包括新兴的Quanto逆期权及其功率类型的泛化，旨在减轻加密货币汇率风险，并调整固有风险敞口。针对这些逆期权导出了基于特征函数的定价-套期保值公式。然后将通用模型框架应用于不对称的拉普拉斯跃变扩散和高斯混合淡定稳定型过程，采用三种分数核，进行广泛的实证分析，包括两个独立的比特币期权数据集上的模型校准，在COVI期间和之后。

    arXiv:2403.16006v1 Announce Type: new  Abstract: Recent empirical evidence has highlighted the crucial role of jumps in both price and volatility within the cryptocurrency market. In this paper, we introduce an analytical model framework featuring fractional stochastic volatility, accommodating price--volatility co-jumps and volatility short-term dependency concurrently. We particularly focus on inverse options, including the emerging Quanto inverse options and their power-type generalizations, aimed at mitigating cryptocurrency exchange rate risk and adjusting inherent risk exposure. Characteristic function-based pricing--hedging formulas are derived for these inverse options. The general model framework is then applied to asymmetric Laplace jump-diffusions and Gaussian-mixed tempered stable-type processes, employing three types of fractional kernels, for an extensive empirical analysis involving model calibration on two independent Bitcoin options data sets, during and after the COVI
    
[^2]: 决策制度感知资产配置：基于统计跳跃模型方法的研究

    Regime-Aware Asset Allocation: a Statistical Jump Model Approach

    [https://arxiv.org/abs/2402.05272](https://arxiv.org/abs/2402.05272)

    本文研究了制度转换对资产配置决策的影响，并比较了不同的制度识别模型。通过使用统计跳跃模型，作者开发了一种决策制度感知的资产配置策略，通过优化跳跃模型的性能指标，并在实证分析中展示了该策略相对于传统方法的表现优势。

    

    本文研究了制度转换对资产配置决策的影响，重点比较了不同制度识别模型。与传统的马尔可夫转换模型不同，我们采用了统计跳跃模型，这是一种近期被提出的稳健模型，以其能够通过应用显式跳跃惩罚来捕捉持久的市场制度而闻名。我们的跳跃模型的特征集仅包括从价格序列中得出的收益和波动率特征。我们引入了一种在时间序列交叉验证框架内选择跳跃惩罚的数据驱动方法，该方法直接优化决策制度感知资产配置策略的性能指标，这一策略经历了全面的多步骤过程构建。通过使用美国主要股票指数的日回报序列进行实证分析，我们突出了采用跳跃模型相对于买入持有策略和马尔可夫转换资产配置方法的表现优势。

    This article investigates the impact of regime switching on asset allocation decisions, with a primary focus on comparing different regime identification models. In contrast to traditional Markov-switching models, we adopt the statistical jump model, a recently proposed robust model known for its ability to capture persistent market regimes by applying an explicit jump penalty. The feature set of our jump model comprises return and volatility features derived solely from the price series. We introduce a data-driven approach for selecting the jump penalty within a time-series cross-validation framework, which directly optimizes the performance metric of the regime-aware asset allocation strategy constructed following a comprehensive multi-step process. Through empirical analysis using daily return series from major US equity indices, we highlight the outperformance of employing jump models in comparison to both buy-and-hold strategies and Markov-switching asset allocation approaches. Th
    
[^3]: 二次对冲和均值方差组合选择中的一价定律

    The law of one price in quadratic hedging and mean-variance portfolio selection. (arXiv:2210.15613v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2210.15613](http://arxiv.org/abs/2210.15613)

    本文证明了一价定律是一个明确均值方差组合配置框架而不会退化的最小条件，并且发现了一种新的机制，用于解释连续时间下一价定律的违规情况。

    

    一价定律（LOP）广泛断言相同的金融流应该获取相同的价格。我们展示了当适当地制定时，LOP是一个明确均值方差组合配置框架而不会退化的最小条件。关键是，本文确定了一个新机制，即在连续时间$L^2(P)$情景下，在没有摩擦的情况下，“在可预测停时之前进行交易”，这出人意料地揭示了连续价格过程中一价定律的违规。关闭这个漏洞允许在二次上下文中给出“资产定价基本定理”的一个版本，建立了LOP经济概念和存在本地$\scr{E}$-鞅态密度的概率性特性之间的等价性。后者为扩展市场中所有平方可积的待定索赔提供唯一的价格，并在均值方差组合选择中发挥重要作用。

    The \emph{law of one price (LOP)} broadly asserts that identical financial flows should command the same price. We show that, when properly formulated, LOP is the minimal condition for a well-defined mean--variance portfolio allocation framework without degeneracy. Crucially, the paper identifies a new mechanism through which LOP can fail in a continuous-time $L^2(P)$ setting without frictions, namely `trading from just before a predictable stopping time', which surprisingly identifies LOP violations even for continuous price processes. Closing this loophole allows to give a version of the ``Fundamental Theorem of Asset Pricing'' appropriate in the quadratic context, establishing the equivalence of the economic concept of LOP with the probabilistic property of the existence of a local $\scr{E}$-martingale state price density. The latter provides unique prices for all square-integrable contingent claims in an extended market and subsequently plays an important role in mean-variance port
    

