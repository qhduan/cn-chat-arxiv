# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Which Past Matters? Training History and Multi-Forecast Horizon in Equity Transformers](https://arxiv.org/abs/2611.23126) | 本文通过十二种配置的月度滚动实验发现，在日内股票收益预测中，最优的历史训练数据选择取决于预测期限——短期预测（5-15分钟）适合近期数据采样，长期预测（60-120分钟）适合均匀采样，且单纯增加训练数据量并不能持续提升预测精度。 |
| [^2] | [Integrating market sentiments for stock price prediction: A comparative study of Bi-LSTM and multilayer perceptions](https://arxiv.org/abs/2611.23057) | 本研究以英伟达公司为例，将FinBERT和TextBlob两种工具从金融新闻中提取的情感分数融入Bi-LSTM和多层感知机模型进行股价预测，发现情感特征在特定市场阶段（尤其是停滞期）能显著提升预测准确性。 |
| [^3] | [Does the AI Reviewer Alleviate or Aggravate the Maintenance Burden? Evidence from GitHub Copilot Reviewer](https://arxiv.org/abs/2611.22793) | 本研究以GitHub Copilot审查者的引入作为自然实验，发现AI审查者短期内显著加重维护者的审查工作量，长期则导致人类贡献下降而机器人PR持续增加，且AI虽提高了人类PR的审查效率，但机器人PR的审查与合并耗时更长。 |
| [^4] | [Covariate-Indexed Inclusion Probabilities on Unbalanced Panels](https://arxiv.org/abs/2611.22730) | 本文提出两个凸模型以估计非平衡面板中协变量索引的包含概率，模型保证概率非负且对任意个体依赖结构有效，两者拟合分歧的一半可在不知真值的情况下估计较差模型的误设误差下界，并在美国股票月度失败预测中优于传统logit模型。 |
| [^5] | [Beyond Strike Remapping: Conditional-Gamma Pricing of Leveraged ETF Options under Local Volatility](https://arxiv.org/abs/2611.22604) | 本文的主要贡献是提出一种新的条件伽马定价方法：通过以标的最终收益率为条件来保留方差损耗的状态依赖性，在校准的局部波动率模型下用三个一维远期偏微分方程计算积分方差的条件矩，并匹配伽马分布，从而以闭式形式对杠杆ETF期权定价，同时保持已实现方差非负。 |
| [^6] | [Clicks, Carbon, and Risk: Climate Attention, Persistent Financial Risk, and Real-Sector Activity](https://arxiv.org/abs/2611.22517) | 本文基于互联网搜索构建气候关注度指数，发现不利型气候担忧会推高能源行业的长期波动并预示油气行业生产与就业的下滑，而转型导向型关注则降低公用事业行业的长期波动，揭示了气候情绪从金融市场向实体经济的传导机制。 |
| [^7] | [Explaining Realized Returns: Roll's $R^2$ 40 Years Later](https://arxiv.org/abs/2611.22277) | 本文利用生成式AI从《华尔街日报》市场综述中提取剔除价格相关语言的原因文本，发现这些原因在样本外能解释约30-40%的股票收益率波动，远超传统宏观经济等变量不足10%的解释力，验证了四十年前罗尔提出的R²之谜在新时代的解答。 |
| [^8] | [An axiomatic foundation for optimized certainty equivalents](https://arxiv.org/abs/2611.22119) | 本文提出“公平硬币补偿”新公理，结合强风险厌恶与归一化首次从经济公理角度刻画了优化确定性等价（OCEs），并通过附加公理进一步分别刻画了标准OCEs、预期短缺、熵风险测度及均值-方差准则。 |
| [^9] | [Thinking Harder Doesn't Lead to Better Choices: Reasoning Effort Increases the Confidence but Not the Correctness of LLM Shopping Agents](https://arxiv.org/abs/2611.22033) | 在三个购物基准测试中，增加LLM购物代理的推理努力并不能显著提升其选购产品的正确性，却会显著提高其报告的置信度——也就是说，模型“想得更多”只是变得更自信，而不是变得更正确。 |
| [^10] | [Revisiting Equity-Bond Diversification under Tail Risk and Structural Market Change: Evidence from Global ETFs and Model-Implied Option Prices](https://arxiv.org/abs/2611.21897) | 本研究对全球30只股票-债券及多资产ETF进行滚动窗口下的最小方差与CVaR组合优化，发现优化组合（尤其多空策略）虽能显著降低尾部与下行风险，但等权重基准的风险调整收益反而最高，揭示了尾部风险管控与收益表现之间的权衡。 |
| [^11] | [Who sees the sea? Sea-view potential by fa\c{c}ade and floor from cadastral LoD1 models on the Costa del Sol](https://arxiv.org/abs/2611.21672) | 该论文提出了一种基于开放地籍LoD1数据的快速海景潜力评估方法，可按建筑立面与楼层量化每栋住宅的海景可见度，并在西班牙太阳海岸15.8万余栋建筑上以不到一小时的双核运算验证了其高效性，结果与LiDAR追踪和房源照片高度一致。 |
| [^12] | [RAW: Refusal Aversion in the Wild, A Causal Measurement Method for Deployed LLMs](https://arxiv.org/abs/2611.21641) | 本文提出RAW方法，利用大语言模型在相同提示下有时拒答有时不拒答的随机性作为自然实验，首次对已部署模型的拒答进行因果测量，发现一次拒答会使约一半被测模型的用户后续再参与率下降3至17个百分点。 |
| [^13] | [Drift-invariant kernels for discount curve models](https://arxiv.org/abs/2611.09751) | 本文证明了核岭回归估计的贴现曲线在风险中性漂移下保持不变的充要条件是核函数具有特定的指数乘积形式，从而使估计曲线可作为连续时间无套利模型的状态。 |
| [^14] | [Quantifying Repair Risk in Crypto Option Surfaces](https://arxiv.org/abs/2611.00958) | 该论文提出“修复风险”概念——同一加密期权报价在满足买卖价差与无套利约束下可能存在多个可容许修复曲面，导致希腊字母与风险度量产生显著差异（经有限差分放大机制可放大至 $4\varepsilon/h^2$），并建立修复风险验证框架在 Deribit BTC/ETH 期权链上量化这种离散性。 |
| [^15] | [AI Adoption in S&P 500 Firms](https://arxiv.org/abs/2607.08920) | 该论文基于SEC 10-K年报开发了一种能够区分“深度AI采纳”与“AI炒作”的新颖度量方法，发现2025年标普500企业中11%已将AI深度整合进业务流程、另有10%在生产与服务交付中使用AI，采纳率自2016年以来已增长超过四倍。 |
| [^16] | [On the use of case estimate and transactional payment data in neural networks for individual loss reserving](https://arxiv.org/abs/2601.05274) | 该论文比较了前馈神经网络（使用汇总交易数据）与循环神经网络（使用完整赔付历史及个案估计发展历史）在个体损失准备金评估中的表现，发现在协变量数据有限的情况下，个案估计能显著提升预测效果，而赋予神经网络记忆机制的额外收益则相对有限。 |
| [^17] | [A Multivariate Generalized Cox Model for Dependent and Simultaneous Default Times](https://arxiv.org/abs/2508.05022) | 该论文提出了一个多元广义Cox模型框架，通过递增驱动过程跨越独立指数阈值来刻画相关联与同时发生的违约时间，并给出了条件联合生存律具有显式乘积形式的充要条件。 |
| [^18] | [Just After Minimum Wage Hikes: Short-Run Labor-Demand Response and Reallocation](https://arxiv.org/abs/2505.04555) | 本文利用日本即时劳动力市场数据发现，最低工资上调后净就业份额立即小幅下降约1%，原有雇主削减低于新工资下限的岗位，而新进入的雇主则在略高于新标准处发布岗位，表明市场主要通过岗位再配置进行调整。 |
| [^19] | [Algorithmic Insurance](https://arxiv.org/abs/2106.00839) | 该论文开创了“算法保险”这一新兴领域，通过条件风险价值（CVaR）将二元分类中的决策阈值选择与尾部风险敞口联系起来，证明了准确率最大化等传统方法会显著放大最坏情况损失，并据此提出了将AI系统设计决策与责任风险敞口相挂钩的保险合同框架。 |
| [^20] | [Post-Last Exit Time Process and its Application to Loss-Given-Default Distribution](https://arxiv.org/abs/2009.00868) | 通过研究后末次退出时间过程，我们建立了一个用于估计公司债务损失给定违约分布的金融模型。 |

# 详细

[^1]: 哪段历史最重要？股票Transformer模型中的训练历史与多预测期限

    Which Past Matters? Training History and Multi-Forecast Horizon in Equity Transformers

    [https://arxiv.org/abs/2611.23126](https://arxiv.org/abs/2611.23126)

    本文通过十二种配置的月度滚动实验发现，在日内股票收益预测中，最优的历史训练数据选择取决于预测期限——短期预测（5-15分钟）适合近期数据采样，长期预测（60-120分钟）适合均匀采样，且单纯增加训练数据量并不能持续提升预测精度。

    

    更多的历史数据可以改善估计效果，但同时也会引入难以迁移到当前市场状况的关系。因此，我们研究了训练数据量与历史数据选择如何与日内股票预测中的预测期限相互作用。我们部署了十二种配置，将四种采样与验证策略同6个月、12个月和24个月的训练数据量相结合，并采用固定的横截面Transformer架构与特征集。月度滚动前推实验共产生了264次重新拟合，以及从5分钟到120分钟的八个累计收益预测。评估区间覆盖2025年1月至2026年10月20日，并使用横截面Spearman信息系数，以未来五因子残差收益作为评估基准。结果显示，6个月近期数据采样在5-15分钟的单模型预测期限中表现领先，而多近期数据配置在20-30分钟期限领先，均匀采样配置则在60-120分钟期限领先。增加训练数据量并不能一致地提升预测准确率。

    arXiv:2611.23126v1 Announce Type: cross  Abstract: More historical data can improve estimation while introducing relationships that transfer poorly to current market conditions. We therefore examine how training quantity and historical selection interact with forecast horizon in intraday equity prediction. We deploy twelve configurations which combine four sampling and validation policies with 6, 12 and 24 month training quantities under a fixed cross-sectional Transformer and feature set. A monthly walk-forward experiment produces 264 refits and eight cumulative return forecasts from 5 to 120 minutes. Evaluation covers January 2025 through 20 October 2026 and uses cross-sectional Spearman information coefficients against future five-factor residual returns. Six-month recency sampling leads individual models at 5-15 minutes, while multi recency configurations lead at 20-30 minutes and uniform configurations at 60-120 minutes. Increasing training quantity does not uniformly improve accu
    
[^2]: 整合市场情绪进行股价预测：Bi-LSTM与多层感知机的比较研究

    Integrating market sentiments for stock price prediction: A comparative study of Bi-LSTM and multilayer perceptions

    [https://arxiv.org/abs/2611.23057](https://arxiv.org/abs/2611.23057)

    本研究以英伟达公司为例，将FinBERT和TextBlob两种工具从金融新闻中提取的情感分数融入Bi-LSTM和多层感知机模型进行股价预测，发现情感特征在特定市场阶段（尤其是停滞期）能显著提升预测准确性。

    

    本研究以英伟达公司为案例，探讨将情感分析与机器学习模型相结合以预测股价走势。研究使用两种情感分析工具从与英伟达相关的金融新闻标题中提取情感分数：一种是特定领域工具FinBERT，另一种是通用型工具TextBlob。这些情感分数被整合到基于双向长短期记忆网络（Bi-LSTM）和多层感知机（MLPs）的预测框架中，并与历史股价数据结合使用。本研究评估了在整个观测期内以及不同市场阶段（牛市、停滞期和强势牛市）下的预测性能。研究结果表明，情感特征在特定情境下能够提升预测准确性，尤其是在停滞期阶段，且TextBlob在特定场景中表现优于FinBERT。

    arXiv:2611.23057v1 Announce Type: new  Abstract: This study investigates the integration of sentiment analysis with machine learning models to forecast stock price movements using the Nvidia Corporation as a case study. Sentiment scores were derived from Nvidia-related financial news headlines using two sentiment analysis tools: FinBERT, a domain-specific tool, and TextBlob, a general-purpose tool. These scores were integrated into predictive frameworks based on bidirectional long short-term memory (Bi-LSTM) networks and multilayer perceptrons (MLPs) developed alongside historical stock price data. This study assesses the predictive performance over the entire observation period and across distinct market phases: bullish, stagnation, and strong bullish conditions. The findings indicate that sentiment features enhance predictive accuracy in specific contexts, particularly during stagnation phases, with TextBlob demonstrating superior performance to FinBERT in specific scenarios. In addi
    
[^3]: AI审查者是减轻还是加重维护负担？来自GitHub Copilot审查者的证据

    Does the AI Reviewer Alleviate or Aggravate the Maintenance Burden? Evidence from GitHub Copilot Reviewer

    [https://arxiv.org/abs/2611.22793](https://arxiv.org/abs/2611.22793)

    本研究以GitHub Copilot审查者的引入作为自然实验，发现AI审查者短期内显著加重维护者的审查工作量，长期则导致人类贡献下降而机器人PR持续增加，且AI虽提高了人类PR的审查效率，但机器人PR的审查与合并耗时更长。

    

    本研究考察了基于AI的代码审查者如何重塑开源软件（OSS）社区中的贡献行为和维护动态。AI审查者提供可扩展、快速且标准化的评估，它们既重塑了贡献方式，也改变了维护者的工作量。本研究利用2025年GitHub Copilot审查者在GitHub上的引入作为自然实验，探讨了AI审查如何影响维护者的工作量、审查效率以及贡献行为。研究发现，在采用AI审查者之后，仓库维护者最初面临PR（拉取请求）提交数量的急剧增加和更高的审查工作量。随着时间推移，人类贡献逐渐下降，而机器人生成的PR的增加则持续存在。有趣的是，生存分析显示，AI审查者提高了人类PR贡献的审查效率，而机器人PR的审查和合并则需要更长时间。研究结果表明，尽管AI审查者增加了……（原文摘要在此处截断）

    arXiv:2611.22793v1 Announce Type: new  Abstract: This study examines how AI-based code reviewers reshape contribution behavior and maintenance dynamics in open-source software (OSS) communities. AI reviewers provide scalable, fast, and standardized evaluations. They reshape contributions as well as maintainers' workload. Leveraging the introduction of GitHub Copilot Reviewer on GitHub as a natural experiment in 2025, we investigate how AI review affects maintainers' workload and review efficiency as well as contribution behavior. We find that repository maintainers initially face a sharp increase in PR submissions and a higher review workload after the adoption of the AI reviewer. Over time, human contributions decline while the increase in bot-generated PRs is sustained. Interestingly, survival analysis reveals that AI reviewers increase review efficiency for human PR contributions, whereas bot PRs take longer to review and merge. Our findings suggest that while AI reviewers increase 
    
[^4]: 非平衡面板上以协变量为索引的包含概率

    Covariate-Indexed Inclusion Probabilities on Unbalanced Panels

    [https://arxiv.org/abs/2611.22730](https://arxiv.org/abs/2611.22730)

    本文提出两个凸模型以估计非平衡面板中协变量索引的包含概率，模型保证概率非负且对任意个体依赖结构有效，两者拟合分歧的一半可在不知真值的情况下估计较差模型的误设误差下界，并在美国股票月度失败预测中优于传统logit模型。

    

    当横截面随时间变化且个体与时期之间存在相依性时，我们估计具有给定协变量的个体属于某个集合的概率。我们提出两个凸模型，其拟合概率在构造上非负，且无论个体之间依赖关系如何均保持有效；对于序列相依面板，我们为这些模型建立了一致性、有限样本界和极限分布。其中一个模型适用于中等或高概率事件，另一个适用于罕见事件，且两者拟合结果之间分歧的一半可以在不知道真实概率的情况下估计出较差模型的误设误差下界。在美国股票月度失败（退市）数据上，罕见事件模型的预测效果优于普通logit模型和经典的失败logit模型。在使用八个经典预测变量时，大部分增益来自每个预测变量秩角度的基函数及其汇总。

    arXiv:2611.22730v1 Announce Type: cross  Abstract: We estimate the probability that an item with given covariates belongs to a set when the cross-section changes over time and items and periods are dependent. We propose two convex models whose fitted probabilities are nonnegative by construction and remain valid whatever the dependence between items, for which we establish consistency, finite-sample bounds and limit distributions for serially dependent panels. One suits events of moderate or high probability, the other rare events, and half the disagreement between their fits estimates a lower bound on the misspecification error of the worse fit, without knowledge of the true probabilities. On monthly US stock failures, the rare-event model predicts better than the ordinary logit and the classical failure logit. With the eight classical predictors, most of the gain comes from basis functions of the rank angle of each predictor and summary.
    
[^5]: 超越行权价重映射：局部波动率下杠杆ETF期权的条件伽马定价

    Beyond Strike Remapping: Conditional-Gamma Pricing of Leveraged ETF Options under Local Volatility

    [https://arxiv.org/abs/2611.22604](https://arxiv.org/abs/2611.22604)

    本文的主要贡献是提出一种新的条件伽马定价方法：通过以标的最终收益率为条件来保留方差损耗的状态依赖性，在校准的局部波动率模型下用三个一维远期偏微分方程计算积分方差的条件矩，并匹配伽马分布，从而以闭式形式对杠杆ETF期权定价，同时保持已实现方差非负。

    

    杠杆ETF（LETF）期权的定价与对冲基于LETF波动率曲面，但路径依赖的方差损耗使得该曲面难以与标的指数的波动率微笑保持一致地构建。我们研究了三种构建该曲面的方法：闭式一阶行权价重映射、一种新的条件伽马方法，以及一个不作任何分布假设的二维联合律偏微分方程（作为基准）。条件伽马方法是我们的主要贡献。它以标的最终收益率作为条件，从而保留了单个重映射行权价所舍弃的方差损耗的状态依赖性。在校准后的局部波动率模型下，三个一维远期偏微分方程给出已实现积分方差的条件均值与条件方差，而与这些矩相匹配的伽马分布保证了已实现方差的非负性，并以闭式形式为期权定价。一个同时匹配条件三阶矩的平移伽马变体……（原文摘要在此处被截断）

    arXiv:2611.22604v1 Announce Type: new  Abstract: Options on leveraged ETFs (LETFs) are priced and hedged off the LETF volatility surface, but path-dependent variance drag makes that surface difficult to build consistently with the underlying index smile. We study three methods for constructing it: a closed-form first-order strike remapping, a new conditional-Gamma method, and a two-dimensional joint-law PDE that makes no distributional assumption and serves as the benchmark. The conditional-Gamma method is our main contribution. It conditions on the terminal underlying return, which preserves the state dependence of variance drag that a single remapped strike discards. Three one-dimensional forward PDEs under a calibrated local-volatility model give the conditional mean and variance of integrated variance, and a Gamma law matched to these moments keeps realised variance non-negative and prices the option in closed form. A shifted-Gamma variant that also matches the conditional third mo
    
[^6]: 点击、碳与风险：气候关注度、持续性金融风险与实体经济活动

    Clicks, Carbon, and Risk: Climate Attention, Persistent Financial Risk, and Real-Sector Activity

    [https://arxiv.org/abs/2611.22517](https://arxiv.org/abs/2611.22517)

    本文基于互联网搜索构建气候关注度指数，发现不利型气候担忧会推高能源行业的长期波动并预示油气行业生产与就业的下滑，而转型导向型关注则降低公用事业行业的长期波动，揭示了气候情绪从金融市场向实体经济的传导机制。

    

    公众气候担忧的变化是否会改变持续性金融风险，并进而对实体经济活动产生影响？我们将美国行业股票日收益率与月度气候关注度指数（CCI）相结合，该指数基于互联网搜索强度构建，并被分解为不利型（adverse）和转型导向型（transition-oriented）两个组成部分。GARCH-MIDAS估计结果显示出显著的异质性：不利型气候担忧会推高碳密集型能源行业的持续性波动率，而转型导向型关注则倾向于降低受监管公用事业行业的持续性波动率，且各子行业之间存在重要差异。随后，我们构建了能源与公用事业之间的持续性风险差异指标，并利用局部投影方法考察其对行业实体经济活动的预测能力。与不利型气候担忧相关的新息在随后的几个季度中先于石油和天然气工业生产及就业的下降，而公用事业行业的类似反应……

    arXiv:2611.22517v1 Announce Type: new  Abstract: Do shifts in public climate concern alter persistent financial risk and subsequently matter for real activity? We combine daily U.S. sectoral equity returns with a monthly Climate Concern Index (CCI) constructed from internet search intensity and decomposed into adverse and transition-oriented components. GARCH-MIDAS estimates reveal substantial heterogeneity: adverse climate concern raises persistent volatility in carbon-intensive Energy activities, while transition-oriented attention tends to reduce persistent volatility in regulated Utilities, with important differences across sub-sectors. We then construct a persistent-risk differential between Energy and Utilities and examine its predictive content for sectoral real activity using local projections. Innovations associated with adverse climate concern precede declines in Oil and Gas industrial production and employment over the following quarters, while comparable Utilities responses
    
[^7]: 解释已实现收益：罗尔的 R² 四十年后

    Explaining Realized Returns: Roll's $R^2$ 40 Years Later

    [https://arxiv.org/abs/2611.22277](https://arxiv.org/abs/2611.22277)

    本文利用生成式AI从《华尔街日报》市场综述中提取剔除价格相关语言的原因文本，发现这些原因在样本外能解释约30-40%的股票收益率波动，远超传统宏观经济等变量不足10%的解释力，验证了四十年前罗尔提出的R²之谜在新时代的解答。

    

    即便事后回看，我们能否解释资产价格的变动？本文将罗尔（Roll, 1988）在美国金融学会（AFA）主席演讲《R²》重新解读为一项现代实证研究，结果显示：传统的宏观经济、现金流、预期以及资金流变量在样本外对美国股票月度收益率波动的解释力不足10%（且R²常为负值），并且这些变量缺乏日度观测数据。我们利用生成式人工智能从《华尔街日报》的市场综述中提取所陈述的原因，同时过滤掉与价格相关的语言。这些原因在样本外能够解释约30-40%的日度和月度收益率波动，其内容涵盖可解释的主题，包括公司新闻、宏观经济状况、货币政策、市场情绪和地缘政治。

    arXiv:2611.22277v1 Announce Type: new  Abstract: Can we explain asset price movements even with hindsight? Reinterpreting Roll's (1988) AFA Presidential Address, "$R^2$," as a modern empirical exercise, we show that conventional macroeconomic, cash-flow, expectations, and flow variables explain less than 10% (often with negative $R^2$) of monthly U.S. equity return variation out of sample and lack daily observations. We use generative AI to extract reasons stated in Wall Street Journal market summaries while filtering out price-related language. These reasons explain around 30-40% of daily and monthly return variation out of sample. Their content spans interpretable themes, including corporate news, macroeconomic conditions, monetary policy, sentiment, and geopolitics.
    
[^8]: 优化确定性等价的公理化基础

    An axiomatic foundation for optimized certainty equivalents

    [https://arxiv.org/abs/2611.22119](https://arxiv.org/abs/2611.22119)

    本文提出“公平硬币补偿”新公理，结合强风险厌恶与归一化首次从经济公理角度刻画了优化确定性等价（OCEs），并通过附加公理进一步分别刻画了标准OCEs、预期短缺、熵风险测度及均值-方差准则。

    

    优化确定性等价是一类重要的风险度量，广泛应用于优化与风险管理，涵盖了均值-方差准则、预期短缺、熵风险测度以及期望。尽管其重要性显著，此前尚无经济公理能够刻画OCEs。我们提出了一个名为“公平硬币补偿”的新公理，它与强风险厌恶和归一化共同刻画了OCEs。我们通过掷硬币对两个损失进行公平分配来诠释这一公理：参与者之间的转移支付可以消除随机分配所带来的额外风险。加入更多公理后可以得到更精确的刻画：单调性导出标准OCEs；正齐次性与谨慎性共同导出预期短缺；公平硬币独立性导出熵风险测度。我们还利用新颖的等均值随机化公理对均值-方差准则进行了公理化刻画

    arXiv:2611.22119v1 Announce Type: cross  Abstract: Optimized Certainty Equivalents (OCEs) are an important class of risk measures for optimization and risk management that includes the Mean--Variance criterion, Expected Shortfall, the Entropic Risk Measure, and expectation. Despite their importance, economic axioms have not characterized OCEs. We propose a new axiom called Fair-Coin Compensation, which characterizes OCEs together with Strong Risk Aversion and Normalization. We interpret Fair-Coin Compensation through the fair assignment of two losses by a coin toss: transfers between two participants can remove the additional risk created by the random assignment. Adding further axioms yields sharper characterizations. Monotonicity yields standard OCEs; Positive Homogeneity, together with Prudence, yields Expected Shortfall; and Fair-Coin Independence yields the Entropic Risk Measure. We also axiomatize the Mean--Variance criterion using our novel axiom of Equal-Mean Randomization, whi
    
[^9]: 更努力思考并不能带来更好的选择：推理努力提升了LLM购物代理的置信度，而非正确性

    Thinking Harder Doesn't Lead to Better Choices: Reasoning Effort Increases the Confidence but Not the Correctness of LLM Shopping Agents

    [https://arxiv.org/abs/2611.22033](https://arxiv.org/abs/2611.22033)

    在三个购物基准测试中，增加LLM购物代理的推理努力并不能显著提升其选购产品的正确性，却会显著提高其报告的置信度——也就是说，模型“想得更多”只是变得更自信，而不是变得更正确。

    

    大型语言模型（LLM）越来越多地被部署为自主购物代理，它们处理多模态商品信息（包括视觉和文本数据）并代表消费者推荐购买。人们自然会认为，思考能力更强的模型会做出更高质量的购买决策。为了验证这一观点，我们针对此类代理提出了两个问题：要求模型更努力地思考，是否会让它选出更好的产品？它所报告的置信度，能否告诉我们它出错的概率？我们在基于WebShop、亚马逊SQID和Wayfair的WANDS平台构建的三个产品选择基准上，对十个前沿模型进行了测试，以检验思考能力更强的LLM是否真的能做出更高质量的产品购买决策。在全部三个基准上，我们发现增加思考努力并不能在统计上显著提升决策质量。而同样的思考努力变化却会提高模型所声称的置信度（原文摘要在此处截断）。

    arXiv:2611.22033v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed as autonomous shopping agents that process multimodal product listings (consisting of visual and text data) and recommend purchases on behalf of consumers. Naturally, one might assume that models with more thinking power would make higher-quality purchasing decisions. To test this belief, we ask two questions about such agents: does asking a model to think harder make it choose better products, and does the confidence it reports tell us the likelihood it may be wrong? We benchmark ten frontier models on three product-choice benchmarks built from WebShop, Amazon's SQID and Wayfair's WANDS platforms to see if LLMs with more thinking power actually make better-quality product purchase decisions. Across all three benchmarks, we found that increasing the thinking effort does not increase decision quality in a statistically meaningful way. The same change in thinking effort raises stated c
    
[^10]: 尾部风险与市场结构性变化下的股票-债券多元化再审视：来自全球ETF与模型隐含期权价格的证据

    Revisiting Equity-Bond Diversification under Tail Risk and Structural Market Change: Evidence from Global ETFs and Model-Implied Option Prices

    [https://arxiv.org/abs/2611.21897](https://arxiv.org/abs/2611.21897)

    本研究对全球30只股票-债券及多资产ETF进行滚动窗口下的最小方差与CVaR组合优化，发现优化组合（尤其多空策略）虽能显著降低尾部与下行风险，但等权重基准的风险调整收益反而最高，揭示了尾部风险管控与收益表现之间的权衡。

    

    本研究使用2020至2024年期间的美元计价日收益率，考察了在美国、加拿大和澳大利亚上市的三十只股票-债券及多资产交易所交易基金（ETF）。研究采用滚动窗口框架，在仅做多与多空两种约束条件下，将最小方差和条件风险价值（CVaR）投资组合优化与等权重基准进行比较。绩效评估采用风险调整指标、下行风险、极端尾部估计以及条件波动率模型。结果显示，优化后的投资组合表现出比基准更低的实证VaR和CVaR，其中多空策略通常能带来更大的下行风险降低效果。然而，等权重基准却实现了最高的Sharpe、Sortino和Rachev比率。基于信息准则，GARCH与FIGARCH的比较结果更倾向于GARCH模型。研究还利用历史估计的正态逆高斯模型生成欧式期权价格并……（摘要原文在此处截断）

    arXiv:2611.21897v1 Announce Type: new  Abstract: This study examines thirty equity--bond and multi-asset exchange-traded funds (ETFs) listed in the United States, Canada, and Australia using daily U.S.-dollar returns over 2020--2024. A rolling-window framework compares minimum-variance and Conditional Value-at-Risk (CVaR) portfolio optimization under long-only and long--short constraints against an equally weighted benchmark. Performance is assessed using risk-adjusted measures, downside risk, extreme-tail estimation, and conditional volatility models. Optimized portfolios exhibit lower empirical VaR and CVaR than the benchmark, with long--short strategies generally providing greater downside-risk reduction. However, the equally weighted benchmark achieves the highest reported Sharpe, Sortino, and Rachev ratios. GARCH and FIGARCH comparisons favor GARCH based on information criteria. A historically estimated Normal Inverse Gaussian model further generates European option prices and mod
    
[^11]: 谁能看到海？基于地籍LoD1模型按立面与楼层评估太阳海岸的海景潜力

    Who sees the sea? Sea-view potential by fa\c{c}ade and floor from cadastral LoD1 models on the Costa del Sol

    [https://arxiv.org/abs/2611.21672](https://arxiv.org/abs/2611.21672)

    该论文提出了一种基于开放地籍LoD1数据的快速海景潜力评估方法，可按建筑立面与楼层量化每栋住宅的海景可见度，并在西班牙太阳海岸15.8万余栋建筑上以不到一小时的双核运算验证了其高效性，结果与LiDAR追踪和房源照片高度一致。

    

    视野是住宅最受重视却最少被测量的属性之一。我们提出了一种开放数据方法，仅利用带楼层数的地籍轮廓图和国家地形模型，即可估算每栋住宅建筑每个无遮挡立面在三个楼层（底层、中层和顶层，其余楼层取其最邻近层）的海景潜力。该方法在双核处理器上不到一小时内完成了西班牙太阳海岸21个市镇共158,060栋建筑的运算，发现52%的建筑从顶层可见部分海景，36%的建筑拥有开阔视野（20°或以上）。在一项基于LiDAR表面的试点追踪中，评分与地籍轮廓内的模型结果高度一致（在20°阈值下96.7%的观测点落入同一类别），但地籍未记录的构筑物平均会使顶层可见海景减少6.1–9.0°。在与93套在售住宅照片（由两家开发商的AI模型进行盲标注，判定关系由作者……）的对比中……（原文摘要在此处截断）

    arXiv:2611.21672v1 Announce Type: new  Abstract: Views are among the most valued and least measured attributes of a home. We present an open-data method that estimates the sea-view potential of every free fa\c{c}ade of every residential building at three floors (ground, middle and top; other floors take the nearest of them), from cadastral footprints with floor counts and a national terrain model. Run for 158,060 buildings of 21 municipalities on Spain's Costa del Sol in under an hour on two cores, it finds that 52 % of buildings see some sea from their top floor and 36 % have an open view (20{\deg} or more). Traced over a LiDAR surface in a pilot, scores agree with the model's inside cadastral footprints (96.7 % of viewpoints in the same class at 20{\deg}), but structures the cadastre does not record lower the sea seen from the top floor by 6.1--9.0{\deg} on average. Against the photographs of 93 homes for sale, labelled blind by AI models of two developers with ties decided by the au
    
[^12]: RAW：现实场景中的拒答回避——一种针对已部署大语言模型的因果测量方法

    RAW: Refusal Aversion in the Wild, A Causal Measurement Method for Deployed LLMs

    [https://arxiv.org/abs/2611.21641](https://arxiv.org/abs/2611.21641)

    本文提出RAW方法，利用大语言模型在相同提示下有时拒答有时不拒答的随机性作为自然实验，首次对已部署模型的拒答进行因果测量，发现一次拒答会使约一半被测模型的用户后续再参与率下降3至17个百分点。

    

    已部署的大语言模型每天处理数亿次对话，并会拒绝相当一部分用户请求，然而拒答给用户侧带来的代价——即用户是否会因此流失以及流失程度——在任何生产模型上都未曾被测量过，这对建立在这些模型之上的面向消费者的平台而言是一个盲区。在前沿实验室产品上进行拒答的A/B测试在伦理和运营层面均受限制，而对于通过固定API部署这些模型的第三方平台来说更是无法实现。我们提出了RAW方法，利用大语言模型的随机性作为自然实验：在任何现有对话日志中，面对几乎相同的提示，同一模型有时拒答、有时不拒答，这种变化使得我们能够识别拒答对用户侧行为的因果效应。将该应用于LMSYS-Chat-1M和WildChat数据集中的15个已部署大语言模型，并采用用户聚类推断，我们发现对于约一半可精确测量的模型，一次拒答会导致后续用户再参与率下降3至17个百分点。

    arXiv:2611.21641v1 Announce Type: cross  Abstract: Deployed LLMs field hundreds of millions of conversations a day and refuse a non-trivial fraction of user requests, yet the user-side cost of refusal, whether users disengage and by how much, is unmeasured on any production model: a blind spot for the consumer-facing platforms built on them. A/B-testing refusal is ethically and operationally constrained on frontier-lab products and unavailable to third-party platforms deploying these models through a fixed API. We introduce RAW, a method that uses LLM stochasticity as a natural experiment: at near-identical prompts in any existing log, the same model sometimes refuses and sometimes does not, and that variation identifies the causal user-side effect of refusal. Applying the method to 15 deployed LLMs in LMSYS-Chat-1M and WildChat, with user-clustered inference, we find that for about half of the models we can measure precisely, a refusal costs 3 to 17 pp of subsequent re-engagement; the
    
[^13]: 面向贴现曲线模型的漂移不变核函数

    Drift-invariant kernels for discount curve models

    [https://arxiv.org/abs/2611.09751](https://arxiv.org/abs/2611.09751)

    本文证明了核岭回归估计的贴现曲线在风险中性漂移下保持不变的充要条件是核函数具有特定的指数乘积形式，从而使估计曲线可作为连续时间无套利模型的状态。

    

    arXiv:2611.09751v1 公告类型：新论文。摘要：核岭回归通过最小化定价误差并加上再生核希尔伯特空间范数形式的惩罚项来估计贴现曲线。根据表示定理，估计结果位于一个有限维仿射空间中，该空间仅取决于核函数和金融工具的现金流日期。我们研究在何种条件下该空间在贴现曲线的风险中性漂移下保持不变，从而使估计出的曲线能够成为连续时间无套利模型的状态。我们证明，对于任意现金流日期网格，当且仅当核函数具有形式 $k(x,y)=\e^{b(x+y)}\big(c\,\e^{axy}+1-c\big)$ 时（希尔伯特空间维数不超过二的核函数除外），该不变性成立。对于这种核函数，估计出的曲线是平坦贴现曲线的仿射组合，其中每条平坦贴现曲线对应一个恒定的连续复利收益率。漂移不变的子空间对应于现金流日期的子网格，并且...

    arXiv:2611.09751v1 Announce Type: new  Abstract: Kernel ridge regression estimates the discount curve as the minimizer of a pricing error plus a penalty given by the norm of a reproducing kernel Hilbert space. By the representer theorem, the estimate lies in a finite-dimensional affine space that depends only on the kernel and on the cash flow dates of the instruments. We ask when this space is invariant under the risk-neutral drift of the discount curve, so that the estimated curves can be the states of a continuous-time arbitrage-free model. We show that this holds for every grid of cash flow dates if and only if the kernel is of the form $k(x,y)=\e^{b(x+y)}\big(c\,\e^{axy}+1-c\big)$, apart from kernels whose Hilbert space has dimension at most two. For this kernel the estimated curve is an affine combination of flat discount curves, each corresponding to a constant continuously compounded yield. The drift-invariant subspaces correspond to sub-grids of the cash flow dates, and the di
    
[^14]: 量化加密期权曲面中的修复风险

    Quantifying Repair Risk in Crypto Option Surfaces

    [https://arxiv.org/abs/2611.00958](https://arxiv.org/abs/2611.00958)

    该论文提出“修复风险”概念——同一加密期权报价在满足买卖价差与无套利约束下可能存在多个可容许修复曲面，导致希腊字母与风险度量产生显著差异（经有限差分放大机制可放大至 $4\varepsilon/h^2$），并建立修复风险验证框架在 Deribit BTC/ETH 期权链上量化这种离散性。

    

    期权交易台在计算希腊字母、对冲数量或风险度量之前，会将加密期权报价调整为保持在买卖价差区间内、并在各行权价间满足标准无套利约束的价格曲面。然而，对于同一报价面板，这些约束可能允许多个符合条件的曲面（我们称之为可容许修复），而不同曲面会产生不同的下游输出。我们将由此产生的变异定义为修复风险，并提出修复风险验证，这是一个用于量化希腊字母、对冲数量和风险度量离散程度的匹配框架。其机制是有限差分放大：价格距离为 $\varepsilon$ 的两个可容许修复，其局部归一化行权价曲率可相差 $4\varepsilon/h^2$，其中 $h$ 为局部归一化行权价间距。我们在 Deribit 的 BTC 和 ETH 期权链上评估修复风险，并将中点价格在同一可容许集合上的 $L_1$ 和 $L_2$ 投影作为匹配对照组。

    arXiv:2611.00958v2 Announce Type: replace  Abstract: Option desks adjust crypto option quotes into price surfaces that remain within bid-ask intervals and satisfy standard no-arbitrage restrictions across strikes before computing Greeks, hedge quantities, or risk measures. For the same quote panel, however, these constraints may admit multiple such surfaces, which we call admissible repairs, with different downstream outputs. We define the resulting variation as repair risk and propose repair-risk validation, a matched framework for quantifying dispersion in Greeks, hedge quantities, and risk measures. The mechanism is finite-difference amplification: two admissible repairs within price distance $\varepsilon$ can differ in local normalized-strike curvature by $4\varepsilon/h^2$, where $h$ is local normalized-strike spacing. We evaluate repair risk on Deribit BTC and ETH option chains, using $L_1$ and $L_2$ projections of midpoint prices onto the same admissible set as matched controls.
    
[^15]: 标普500企业的AI采纳研究

    AI Adoption in S&P 500 Firms

    [https://arxiv.org/abs/2607.08920](https://arxiv.org/abs/2607.08920)

    该论文基于SEC 10-K年报开发了一种能够区分“深度AI采纳”与“AI炒作”的新颖度量方法，发现2025年标普500企业中11%已将AI深度整合进业务流程、另有10%在生产与服务交付中使用AI，采纳率自2016年以来已增长超过四倍。

    

    arXiv:2607.08920v2 公告类型：替换 摘要：大型企业对人工智能（AI）的采纳是总体生产率提升和劳动力市场影响的重要潜在来源。我们研究了2016年至2025年期间标普500企业的AI采纳情况，并在企业层面进行采纳率估计。虽然生成式AI工具对个人和专业应用很有用，但我们的研究重点是AI在大型企业业务流程中的深度整合，这些企业是更广泛企业采纳趋势的风向标。我们开发了一种新颖的度量方法来评估深度AI采纳（并将其与AI炒作区分开来），该方法基于SEC 10-K年报文件，因为法律法规“禁止公司做出重大虚假或误导性陈述”。2025年，11%的标普500企业已将AI深度整合到其业务流程中，另有10%的企业在商品生产和服务交付中使用AI。AI的采纳率已从5（原文在此处截断，疑为“5%”）增长了超过四倍。

    arXiv:2607.08920v2 Announce Type: replace  Abstract: The adoption of artificial intelligence (AI) by large enterprises is an important potential source of aggregate productivity improvement and labor market impact. We study AI adoption of S&P 500 firms over the period 2016 to 2025, estimating adoption at the enterprise level. While generative AI tools are useful for personal and professional applications, our focus is on the deep integration of AI in the business processes of large enterprises which are bellwethers for firm adoption more broadly. We develop a novel measure to assess deep AI adoption (and distinguish it from AI hype) that is based on SEC 10-K filings, where laws and regulations ``prohibit companies from making materially false or misleading statements." In 2025, 11% of S&P 500 enterprises had AI deeply integrated into their business processes, and a further 10% were using AI in the production of goods and delivery of services. AI adoption has more than quadrupled from 5
    
[^16]: 论在个体损失准备金评估的神经网络中使用个案估计与交易赔付数据

    On the use of case estimate and transactional payment data in neural networks for individual loss reserving

    [https://arxiv.org/abs/2601.05274](https://arxiv.org/abs/2601.05274)

    该论文比较了前馈神经网络（使用汇总交易数据）与循环神经网络（使用完整赔付历史及个案估计发展历史）在个体损失准备金评估中的表现，发现在协变量数据有限的情况下，个案估计能显著提升预测效果，而赋予神经网络记忆机制的额外收益则相对有限。

    

    在精算准备金文献中，使用基于个体索赔数据训练的神经网络已日益流行。我们研究了如何在神经网络模型中最优地输入历史赔付数据。此外，个案估计也以时间序列的形式存在，我们进一步扩展了分析以评估其预测能力。在本文中，我们将基于汇总交易数据训练的前馈神经网络与能够分析索赔完整赔付历史和/或个案估计发展历史的循环神经网络进行了比较。我们通过在由SPLICE（Avanzi、Taylor和Wang，2023）模拟出的多个可比较的高度复杂数据集上训练并比较各模型的性能，从而得出结论。在协变量数据有限的环境中，我们发现有证据表明个案估计能够显著改善预测效果，但为神经网络配备记忆机制只会带来……（原文摘要至此处被截断）

    arXiv:2601.05274v2 Announce Type: replace-cross  Abstract: The use of neural networks trained on individual claims data has become increasingly popular in the actuarial reserving literature. We consider how to best input historical payment data in neural network models. Additionally, case estimates are also available in the format of a time series, and we extend our analysis to assessing their predictive power. In this paper, we compare a feed-forward neural network trained on summarised transactions to a recurrent neural network equipped to analyse a claim's entire payment history and/or case estimate development history. We draw conclusions from training and comparing the performance of the models on multiple, comparable highly complex datasets simulated from SPLICE (Avanzi, Taylor and Wang, 2023). In environments where covariate data is limited, we find evidence that case estimates will improve predictions substantially, but that equipping the neural network with memory only leads t
    
[^17]: 用于相关联和同时违约时间的多元广义Cox模型

    A Multivariate Generalized Cox Model for Dependent and Simultaneous Default Times

    [https://arxiv.org/abs/2508.05022](https://arxiv.org/abs/2508.05022)

    该论文提出了一个多元广义Cox模型框架，通过递增驱动过程跨越独立指数阈值来刻画相关联与同时发生的违约时间，并给出了条件联合生存律具有显式乘积形式的充要条件。

    

    我们提出了一个相关违约时间的框架，其中每个主体（名称）在一个递增的驱动过程跨越独立的单位指数阈值时发生违约。正如Gueye和Jeanblanc（2022）所述，驱动过程可能发生跳跃，因此违约可能发生在参考滤流的停时处，包括可预测的停时，且一次共同的跳跃可能导致零个、一个或多个同时违约。对于一般的驱动过程，我们通过向后递推导出了条件联合生存律；当且仅当对于任意一组主体，其首次违约时间的补偿子的可预测约化为确定性时，该生存律才是生存函数的显式乘积形式。此外，如果违约可避免可预测停时，则该生存律是一个致命冲击律，其风险强度可由驱动过程确定；而可预测的违约原子可能排除这一性质。例子包括Lévy从属过程，以及具有随机约化但显式生存律的Liu模型和散粒噪声驱动过程。

    arXiv:2508.05022v2 Announce Type: replace-cross  Abstract: We propose a framework for dependent default times where each name defaults when an increasing driver crosses an independent unit exponential threshold. As in Gueye and Jeanblanc (2022), the drivers may jump, so defaults may occur at stopping times of the reference filtration, including predictable ones, and a common jump may cause no, one or several simultaneous defaults. For general drivers, we derive the conditional joint survival law by a backward recursion; it is an explicit product of survival functions if and only if, for every set of names, the predictable reduction of the compensator of their first default time is deterministic. If, moreover, defaults avoid predictable stopping times, the law is a fatal shock law with hazards identified from the drivers; predictable default atoms can rule this out. Examples include L\'evy subordinators and, with random reductions but explicit laws, Liu's model and shot-noise drivers wi
    
[^18]: 最低工资上调之后：短期劳动需求响应与再配置

    Just After Minimum Wage Hikes: Short-Run Labor-Demand Response and Reallocation

    [https://arxiv.org/abs/2505.04555](https://arxiv.org/abs/2505.04555)

    本文利用日本即时劳动力市场数据发现，最低工资上调后净就业份额立即小幅下降约1%，原有雇主削减低于新工资下限的岗位，而新进入的雇主则在略高于新标准处发布岗位，表明市场主要通过岗位再配置进行调整。

    

    最低工资上调后劳动力市场如何在短期内立即调整，仍然是一个开放的、具有重要政策意义的问题。本文利用Timee数据和工资区间双重差分设计，研究了最低工资上调在日本即时（零工）劳动力市场中的短期效应。我们发现标准化的净就业份额变化为-0.010，职位空缺和总工作时长也有类似的估计结果。周度和日度估计显示，这些效应在政策实施后立即显现，随后趋于稳定。在最低工资约束力度更大的地区以及餐饮业中，负面效应更为明显。深夜班次和提供交通补贴的招聘信息其净份额效应为负，而针对特定性别和面向新手的招聘信息则没有显著的净变化。通过按雇主的招聘历史对职位空缺进行分解，我们发现原有雇主削减了低于新工资下限的招聘信息，而修订前从未发布过招聘信息的雇主则将招聘工资定在略高于新标准的水平。

    arXiv:2505.04555v4 Announce Type: replace  Abstract: How labor markets adjust immediately after minimum wage hikes remains an open, policy-relevant question. This paper studies short-run minimum-wage effects in Japan's spot labor market using Timee data and a wage-bin difference-in-differences design. We find a normalized net employment-share change of $-0.010$, with similar estimates for vacancies and total working hours. Weekly and daily estimates show that these effects emerge immediately after implementation and then stabilize. Effects are more negative where the minimum-wage bite is higher and in restaurant work. Net share effects are negative for midnight shifts and postings offering transportation reimbursement. Gender-targeted and beginner-targeted postings show no significant net change. Decomposing vacancies by employers' posting histories, we find that incumbent employers cut postings below the new floor while employers that did not post before the revision post just above i
    
[^19]: 算法保险

    Algorithmic Insurance

    [https://arxiv.org/abs/2106.00839](https://arxiv.org/abs/2106.00839)

    该论文开创了“算法保险”这一新兴领域，通过条件风险价值（CVaR）将二元分类中的决策阈值选择与尾部风险敞口联系起来，证明了准确率最大化等传统方法会显著放大最坏情况损失，并据此提出了将AI系统设计决策与责任风险敞口相挂钩的保险合同框架。

    

    当人工智能（AI）系统在医学诊断或自动驾驶等高风险领域出现错误时，一个在不同情境中相同的单一算法缺陷可能产生异质性的损失，这挑战了传统保险的假设。算法保险，即为AI造成的损害提供财务保障的新兴市场正在兴起，然而保险公司在为这类风险定价时面临困难，AI开发者也缺乏将系统设计与责任风险敞口相关联的框架。我们将二元分类中的一个关键操作选择——从业者如何设置决策阈值——与尾部风险敞口联系起来。利用条件风险价值（CVaR）来刻画极端损失，我们证明了准确率最大化等既有方法相对于尾部风险优化会显著增加最坏情况下的损失，且当阈值偏离最优值时，惩罚至少以二次方速度增长。我们提出了一种责任保险合同，要求……（原文摘要在此处截断）

    arXiv:2106.00839v4 Announce Type: replace  Abstract: When artificial intelligence (AI) systems make errors in high-stakes domains such as medical diagnosis or autonomous driving, a single algorithmic flaw across varying contexts can generate heterogeneous losses that challenge traditional insurance assumptions. Algorithmic insurance, financial coverage for AI-induced damages, is an emerging market, yet insurers struggle to price these risks and AI developers lack frameworks linking system design to liability exposure. We connect a key operational choice in binary classification, how practitioners set decision thresholds, to tail risk exposure. Using conditional value-at-risk (CVaR) to capture extreme losses, we prove that established approaches such as accuracy maximization can significantly increase worst-case losses relative to tail risk optimization, with penalties growing at least quadratically as thresholds deviate from optimal. We propose a liability insurance contract that manda
    
[^20]: 后末次退出时间过程及其在损失给定违约分布中的应用

    Post-Last Exit Time Process and its Application to Loss-Given-Default Distribution

    [https://arxiv.org/abs/2009.00868](https://arxiv.org/abs/2009.00868)

    通过研究后末次退出时间过程，我们建立了一个用于估计公司债务损失给定违约分布的金融模型。

    

    我们研究了线性扩散过程在从特定规则点最后退出时间后的情况。我们并非将该过程视为在最后退出时间时新生，而是观察整条路径，并将最后退出时间前后的原始过程分开。这不仅使我们能够识别过渡半群、边界行为、入口规律以及后末次退出时间过程的反向，而且能够建立一个用于估计公司债务损失给定违约分布的金融模型（一个一直存在的重要开放问题）。

    arXiv:2009.00868v3 Announce Type: replace  Abstract: We study a linear diffusion process after its last exit time from a certain regular point. Rather than treating the process as newly born at the last exit time, we view the whole path and separate the original process before and after the last exit time. This enables us not only to identify the transition semigroup, boundary behavior, entrance law, and reverse of the post-last exit time process, but also to establish a financial model for estimating the loss-given-default distribution of corporate debt (an all-time important open problem).
    

