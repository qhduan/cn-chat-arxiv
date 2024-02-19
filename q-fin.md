# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Modelling crypto markets by multi-agent reinforcement learning](https://arxiv.org/abs/2402.10803) | 该研究通过引入多智能体强化学习模型，成功模拟加密货币市场，弥补了之前基于零智能代理或单个自主智能体方法的不足。 |
| [^2] | [RAGIC: Risk-Aware Generative Adversarial Model for Stock Interval Construction](https://arxiv.org/abs/2402.10760) | 提出了一种新型模型RAGIC，引入序列生成用于股票区间预测，利用生成对抗网络生成未来价格序列，通过风险模块和时间模块创建风险敏感区间。 |
| [^3] | [Emoji Driven Crypto Assets Market Reactions](https://arxiv.org/abs/2402.10481) | 该研究利用GPT-4和BERT模型进行多模态情感分析，发现基于表情符号情绪的策略可以帮助避免市场下挫并稳定回报。 |
| [^4] | [The Famous American Economist H. Markowitz and Mathematical Overview of his Portfolio Selection Theory](https://arxiv.org/abs/2402.10253) | 该论文对著名美国经济学家H. Markowitz的生平及其投资组合选择理论的数学综述进行了研究。 |
| [^5] | [The Mean Field Market Model Revisited](https://arxiv.org/abs/2402.10215) | 本文提出了一种新的视角重新审视均场LIBOR市场模型，将其嵌入到经典设置中，在长时间范围内控制期限利率方差的关键特征，并展示了可以直接应用于建模各种期限利率的框架。 |
| [^6] | [The Paradox Of Just-in-Time Liquidity in Decentralized Exchanges: More Providers Can Sometimes Mean Less Liquidity](https://arxiv.org/abs/2311.18164) | 区块链去中心化交易所中“及时性流动性”提供存在悖论，即使可能降低整体市场流动性，当订单量对于流动性池深度的弹性不足时，更多的提供方有时意味着更少的流动性 |
| [^7] | [On Pricing of Discrete Asian and Lookback Options under the Heston Model](https://arxiv.org/abs/2211.03638) | 提出了一种基于数据驱动和神经网络的方法，用于在Heston模型下高效定价离散算术亚式期权和Lookback期权，相比传统方法减少计算时间高达数千倍 |
| [^8] | [Robust Estimation of Pareto's Scale Parameter from Grouped Data.](http://arxiv.org/abs/2401.14593) | 本文介绍了一种新的稳健估计方法（MTuM），用于从分组数据中估计Pareto分布的尾指数。该方法通过应用中心极限定理和模拟研究验证了其推理合理性。 |
| [^9] | [Proof of Efficient Liquidity: A Staking Mechanism for Capital Efficient Liquidity.](http://arxiv.org/abs/2401.04521) | Proof of Efficient Liquidity (PoEL)协议旨在为具有内置DeFi应用的基于PoS共识的区块链提供支持，通过有效利用预算化的权益奖励来吸引和维持流动性，并实现资本效率的最大化。 |
| [^10] | [Predict-AI-bility of how humans balance self-interest with the interest of others.](http://arxiv.org/abs/2307.12776) | 生成式AI能够准确预测人类在决策中平衡自身利益与他人利益的行为模式，但存在高估他人关注行为的倾向，这对AI的开发者和用户具有重要意义。 |
| [^11] | [Optimizing Credit Limit Adjustments Under Adversarial Goals Using Reinforcement Learning.](http://arxiv.org/abs/2306.15585) | 本研究使用强化学习技术，通过平衡最大化投资组合收入和最小化准备金的对抗目标，自动化寻找最优信用卡额度调整策略。 |
| [^12] | [Correlation between upstreamness and downstreamness in random global value chains.](http://arxiv.org/abs/2303.06603) | 本文研究了全球价值链中产业和国家的上游和下游，发现同一产业部门的上游和下游之间存在正相关性。 |

# 详细

[^1]: 通过多智能体强化学习模型对加密市场进行建模

    Modelling crypto markets by multi-agent reinforcement learning

    [https://arxiv.org/abs/2402.10803](https://arxiv.org/abs/2402.10803)

    该研究通过引入多智能体强化学习模型，成功模拟加密货币市场，弥补了之前基于零智能代理或单个自主智能体方法的不足。

    

    构建在之前的基础工作（Lussange等人，2020年）之上，本研究引入了一种多智能体强化学习（MARL）模型，模拟加密货币市场，该模型校准为 2018 年至 2022 年间不间断交易的 Binance 的153种加密货币的每日收盘价。与先前依赖于零智能代理或单个自主智能体方法的代理基础模型（ABM）或多智能体系统（MAS）不同，我们的方法依赖于赋予代理强化学习（RL）技术，以模拟加密市场。这种整合旨在通过自下而上的复杂性推理，模拟个体和集体代理，确保在这类市场近期波动剧烈且在 COVID-19 时代期间的稳健性。我们模型的一个关键特征还在于其自主代理根据两种信息源进行资产价格估值：

    arXiv:2402.10803v1 Announce Type: cross  Abstract: Building on a previous foundation work (Lussange et al. 2020), this study introduces a multi-agent reinforcement learning (MARL) model simulating crypto markets, which is calibrated to the Binance's daily closing prices of $153$ cryptocurrencies that were continuously traded between 2018 and 2022. Unlike previous agent-based models (ABM) or multi-agent systems (MAS) which relied on zero-intelligence agents or single autonomous agent methodologies, our approach relies on endowing agents with reinforcement learning (RL) techniques in order to model crypto markets. This integration is designed to emulate, with a bottom-up approach to complexity inference, both individual and collective agents, ensuring robustness in the recent volatile conditions of such markets and during the COVID-19 era. A key feature of our model also lies in the fact that its autonomous agents perform asset price valuation based on two sources of information: the mar
    
[^2]: RAGIC: 面向股票区间构建的风险感知生成对抗模型

    RAGIC: Risk-Aware Generative Adversarial Model for Stock Interval Construction

    [https://arxiv.org/abs/2402.10760](https://arxiv.org/abs/2402.10760)

    提出了一种新型模型RAGIC，引入序列生成用于股票区间预测，利用生成对抗网络生成未来价格序列，通过风险模块和时间模块创建风险敏感区间。

    

    预测股市结果的努力取得了有限的成功，这是由市场固有的随机特性受许多不可预测因素影响造成的。许多现有的预测方法侧重于单点预测，缺乏有效决策所需的深度，经常忽视市场风险。为了弥补这一差距，我们提出了一种新颖的模型RAGIC，引入序列生成用于股票区间预测，更有效地量化不确定性。我们的方法利用生成对抗网络（GAN）生成未来价格序列，注入金融市场固有的随机性。RAGIC的生成器包括一个风险模块，捕捉熟悉投资者的风险感知，以及一个时间模块，考虑历史价格趋势和季节性。这个多方面的生成器通过统计推断呈现风险敏感区间的创建，融入时间

    arXiv:2402.10760v1 Announce Type: cross  Abstract: Efforts to predict stock market outcomes have yielded limited success due to the inherently stochastic nature of the market, influenced by numerous unpredictable factors. Many existing prediction approaches focus on single-point predictions, lacking the depth needed for effective decision-making and often overlooking market risk. To bridge this gap, we propose a novel model, RAGIC, which introduces sequence generation for stock interval prediction to quantify uncertainty more effectively. Our approach leverages a Generative Adversarial Network (GAN) to produce future price sequences infused with randomness inherent in financial markets. RAGIC's generator includes a risk module, capturing the risk perception of informed investors, and a temporal module, accounting for historical price trends and seasonality. This multi-faceted generator informs the creation of risk-sensitive intervals through statistical inference, incorporating horizon
    
[^3]: 基于表情符号的加密资产市场反应

    Emoji Driven Crypto Assets Market Reactions

    [https://arxiv.org/abs/2402.10481](https://arxiv.org/abs/2402.10481)

    该研究利用GPT-4和BERT模型进行多模态情感分析，发现基于表情符号情绪的策略可以帮助避免市场下挫并稳定回报。

    

    在加密货币领域，诸如Twitter之类的社交媒体平台已经成为影响市场趋势和投资者情绪的关键因素。在我们的研究中，我们利用GPT-4和经过微调的基于BERT模型的多模态情感分析，重点关注表情符号情绪对加密货币市场的影响。通过将表情符号转化为可量化的情感数据，我们将这些见解与BTC价格和VCRIX指数等关键市场指标进行了相关联。这种方法可以用于开发旨在利用社交媒体元素识别和预测市场趋势的交易策略。关键是，我们的研究结果表明，基于表情符号情绪的策略可以有助于避免重大市场下挫，并有助于回报的稳定。这项研究强调了将先进的基于人工智能的分析整合到金融策略中的实际益处，并提供了一种新的方式来看待市场预测。

    arXiv:2402.10481v1 Announce Type: cross  Abstract: In the burgeoning realm of cryptocurrency, social media platforms like Twitter have become pivotal in influencing market trends and investor sentiments. In our study, we leverage GPT-4 and a fine-tuned transformer-based BERT model for a multimodal sentiment analysis, focusing on the impact of emoji sentiment on cryptocurrency markets. By translating emojis into quantifiable sentiment data, we correlate these insights with key market indicators like BTC Price and the VCRIX index. This approach may be fed into the development of trading strategies aimed at utilizing social media elements to identify and forecast market trends. Crucially, our findings suggest that strategies based on emoji sentiment can facilitate the avoidance of significant market downturns and contribute to the stabilization of returns. This research underscores the practical benefits of integrating advanced AI-driven analyses into financial strategies, offering a nuan
    
[^4]: 著名美国经济学家H. Markowitz及其投资组合选择理论的数学综述

    The Famous American Economist H. Markowitz and Mathematical Overview of his Portfolio Selection Theory

    [https://arxiv.org/abs/2402.10253](https://arxiv.org/abs/2402.10253)

    该论文对著名美国经济学家H. Markowitz的生平及其投资组合选择理论的数学综述进行了研究。

    

    这篇调查文章致力于介绍著名美国经济学家H. Markowitz（1927-2023）的生平。我们回顾了投资组合选择理论的主要观点，以数学完整性的角度包括所有必要的辅助细节。

    arXiv:2402.10253v1 Announce Type: new  Abstract: This survey article is dedicated to the life of the famous American economist H. Markowitz (1927--2023). We do revisit the main statements of the portfolio selection theory in terms of mathematical completeness including all the necessary auxiliary details.
    
[^5]: 《重新审视均场市场模型》

    The Mean Field Market Model Revisited

    [https://arxiv.org/abs/2402.10215](https://arxiv.org/abs/2402.10215)

    本文提出了一种新的视角重新审视均场LIBOR市场模型，将其嵌入到经典设置中，在长时间范围内控制期限利率方差的关键特征，并展示了可以直接应用于建模各种期限利率的框架。

    

    在本文中，我们对Desmettre等人在arXiv:2109.10779中引入的均场LIBOR市场模型提出了一种替代视角。我们的新方法将均场模型嵌入到一个经典设置中，但保留了在长时间范围内控制期限利率方差的关键特征。这保持了市场模型的实用性，因为可以高效地进行校准和模拟，而无需嵌套模拟。此外，我们展示了我们的框架可以直接应用于建模来自SOFR、ESTR或其他几乎无风险的隔夜短期利率的期限利率 ——这是一个至关重要的特征，因为许多IBOR利率正在逐渐被替换。这些结果由一项校准研究和一些理论论证来补充，这些理论论证可以用来估计所呈现的市场模型中出现不现实高利率的概率。

    arXiv:2402.10215v1 Announce Type: new  Abstract: In this paper, we present an alternative perspective on the mean-field LIBOR market model introduced by Desmettre et al. in arXiv:2109.10779. Our novel approach embeds the mean-field model in a classical setup, but retains the crucial feature of controlling the term rate's variances over large time horizons. This maintains the market model's practicability, since calibrations and simulations can be carried out efficiently without nested simulations. In addition, we show that our framework can be directly applied to model term rates derived from SOFR, ESTR or other nearly risk-free overnight short-term rates -- a crucial feature since many IBOR rates are gradually being replaced. These results are complemented by a calibration study and some theoretical arguments which allow to estimate the probability of unrealistically high rates in the presented market models.
    
[^6]: 区块链去中心化交易所中“及时性流动性”悖论：更多提供方有时意味着更少的流动性

    The Paradox Of Just-in-Time Liquidity in Decentralized Exchanges: More Providers Can Sometimes Mean Less Liquidity

    [https://arxiv.org/abs/2311.18164](https://arxiv.org/abs/2311.18164)

    区块链去中心化交易所中“及时性流动性”提供存在悖论，即使可能降低整体市场流动性，当订单量对于流动性池深度的弹性不足时，更多的提供方有时意味着更少的流动性

    

    我们研究了基于区块链的去中心化交易所中的“及时性流动性”提供。一种“即时性（JIT）”流动性提供者（LP）监视区块链的公共mempool中挂起的交换订单，通过在订单之前进行存款并在订单之后进行提款，以提供流动性。我们的博弈论模型揭示了即时性(LP)的存在并不总是增加流动性池深度，人们可能会期待这种情况。尽管被动LP面临着被通知的套利者的逆向选择，但即时性LP能够在提供流动性之前探测对恶性订单流的挂起订单，从而避免被逆向选择。当订单量对于流动性池深度的弹性不足时，即时性LP仅为未知订单提供流动性，并且会排挤被动LP，可能降低整体市场流动性。我们表明，使用转移即时LP部分费用收入的两层费用结构

    arXiv:2311.18164v2 Announce Type: replace  Abstract: We study Just-in-time (JIT) liquidity provision in blockchain-based decentralized exchanges. A JIT liquidity provider (LP) monitors pending swap orders in public mempools of blockchains to sandwich orders of their choice with liquidity, depositing right before and withdrawing right after the order. Our game-theoretic model with asymmetrically informed agents reveals that a JIT LP's presence does not always enhance liquidity pool depth, as one might expect. While passive LPs face adverse selection by informed arbitrageurs, a JIT LP's ability to detect pending orders for toxic order flow prior to liquidity provision lets them avoid being adversely selected. JIT LPs thus only provide liquidity to uninformed orders and crowd out passive LPs when order volume is not sufficiently elastic to pool depth, possibly reducing overall market liquidity. We show that using a two-tiered fee structure which transfers a part of a JIT LP's fee revenue 
    
[^7]: 对Heston模型下的离散亚式期权和Lookback期权定价方法的研究

    On Pricing of Discrete Asian and Lookback Options under the Heston Model

    [https://arxiv.org/abs/2211.03638](https://arxiv.org/abs/2211.03638)

    提出了一种基于数据驱动和神经网络的方法，用于在Heston模型下高效定价离散算术亚式期权和Lookback期权，相比传统方法减少计算时间高达数千倍

    

    我们提出了一种新的、数据驱动的方法，用于在Heston模型动态驱动的情况下高效定价固定和浮动执行价格的离散算术亚式期权和Lookback期权。本文提出的方法是我们先前工作的延伸，先前工作中解决了从时间积分随机桥中采样的问题。该模型依赖于七联盟方案，其中利用随机配置点的人工神经网络被用来“学习”感兴趣的随机变量的分布。该方法导致了一种稳健的蒙特卡罗定价程序。此外，在一个简化但通用的框架中提供了期权定价的半解析公式。与传统蒙特卡罗定价方案相比，该模型保证了高精确度和计算时间减少高达数千倍。

    arXiv:2211.03638v2 Announce Type: replace  Abstract: We propose a new, data-driven approach for efficient pricing of - fixed- and float-strike - discrete arithmetic Asian and Lookback options when the underlying process is driven by the Heston model dynamics. The method proposed in this article constitutes an extension of our previous work, where the problem of sampling from time-integrated stochastic bridges was addressed. The model relies on the Seven-League scheme, where artificial neural networks are employed to "learn" the distribution of the random variable of interest utilizing stochastic collocation points. The method results in a robust procedure for Monte Carlo pricing. Furthermore, semi-analytic formulae for option pricing are provided in a simplified, yet general, framework. The model guarantees high accuracy and a reduction of the computational time up to thousands of times compared to classical Monte Carlo pricing schemes.
    
[^8]: 从分组数据中稳健估计Pareto的尺度参数

    Robust Estimation of Pareto's Scale Parameter from Grouped Data. (arXiv:2401.14593v1 [stat.ME])

    [http://arxiv.org/abs/2401.14593](http://arxiv.org/abs/2401.14593)

    本文介绍了一种新的稳健估计方法（MTuM），用于从分组数据中估计Pareto分布的尾指数。该方法通过应用中心极限定理和模拟研究验证了其推理合理性。

    

    当可获取的完全观测到的从头至尾的损失严重性样本数据集存在时，存在许多稳健估计器作为最大似然估计器（MLE）的替代方案。然而，当处理分组损失严重性数据时，稳健的MLE替代方案的选择变得非常有限，只有少数方法可用，例如最小二乘法、最小Hellinger距离和最优有界影响函数。本文介绍了一种称为截断矩法的新型稳健估计技术，该方法专门用于从分组数据估计Pareto分布的尾指数。通过应用中心极限定理和通过全面的模拟研究验证了MTuM的推理合理性。

    Numerous robust estimators exist as alternatives to the maximum likelihood estimator (MLE) when a completely observed ground-up loss severity sample dataset is available. However, the options for robust alternatives to MLE become significantly limited when dealing with grouped loss severity data, with only a handful of methods like least squares, minimum Hellinger distance, and optimal bounded influence function available. This paper introduces a novel robust estimation technique, the Method of Truncated Moments (MTuM), specifically designed to estimate the tail index of a Pareto distribution from grouped data. Inferential justification of MTuM is established by employing the central limit theorem and validating them through a comprehensive simulation study.
    
[^9]: 有效流动性的证明：一种资本高效流动性的权益机制

    Proof of Efficient Liquidity: A Staking Mechanism for Capital Efficient Liquidity. (arXiv:2401.04521v1 [q-fin.GN])

    [http://arxiv.org/abs/2401.04521](http://arxiv.org/abs/2401.04521)

    Proof of Efficient Liquidity (PoEL)协议旨在为具有内置DeFi应用的基于PoS共识的区块链提供支持，通过有效利用预算化的权益奖励来吸引和维持流动性，并实现资本效率的最大化。

    

    Proof of Efficient Liquidity (PoEL)协议是为专门采用内置DeFi应用的基于Proof of Stake (PoS)共识的区块链基础设施设计的，旨在支持可持续的流动性引导和网络安全。这种创新的机制通过风险结构引擎和激励分配策略，有效利用预算化的权益奖励来吸引和维持流动性，从而实现了资本效率的最大化。该协议旨在实现两个目标：（i）通过有效吸引风险资本并最大限度地提高其在内置DeFi应用中的运营效用来创造资本，从而确保可持续性；（ii）通过在PoS机制中增加一个协同层，吸引多样化的数字资产，从而增强采用的区块链网络的经济安全性。最后，在附录中，我们还试图将这一金融激励协议推广到...

    The Proof of Efficient Liquidity (PoEL) protocol, designed for specialised Proof of Stake (PoS) consensus-based blockchain infrastructures that incorporate intrinsic DeFi applications, aims to support sustainable liquidity bootstrapping and network security. This innovative mechanism efficiently utilises budgeted staking rewards to attract and sustain liquidity through a risk structuring engine and incentive allocation strategy, both of which are designed to maximise capital efficiency. The proposed protocol seeks to serve the dual objective of - (i) capital creation, by efficiently attracting risk capital, and maximising its operational utility for intrinsic DeFi applications, thereby asserting sustainability; and (ii) enhancing the adopting blockchain network's economic security, by augmenting their staking (PoS) mechanism with a harmonious layer seeking to attract a diversity of digital assets. Finally, in the appendix, we seek to generalise the financial incentivisation protocol to
    
[^10]: 预测人类如何在自身利益与他人利益之间平衡的可预测性

    Predict-AI-bility of how humans balance self-interest with the interest of others. (arXiv:2307.12776v1 [econ.GN])

    [http://arxiv.org/abs/2307.12776](http://arxiv.org/abs/2307.12776)

    生成式AI能够准确预测人类在决策中平衡自身利益与他人利益的行为模式，但存在高估他人关注行为的倾向，这对AI的开发者和用户具有重要意义。

    

    生成式人工智能具有革命性的潜力，可以改变从日常生活到高风险场景的决策过程。然而，由于许多决策具有社会影响，为了使AI能够成为可靠的决策助手，它必须能够捕捉自身利益与他人利益之间的平衡。我们对三种最先进的聊天机器人对来自12个国家的78个实验的独裁者游戏决策进行了研究。我们发现，只有GPT-4（而不是Bard或Bing）能够正确捕捉到行为模式的定性特征，识别出三种主要的行为类别：自私的、不公平厌恶的和完全无私的。然而，GPT-4一直高估了他人关注行为，夸大了不公平厌恶和完全无私参与者的比例。这种偏见对于AI开发人员和用户具有重要意义。

    Generative artificial intelligence holds enormous potential to revolutionize decision-making processes, from everyday to high-stake scenarios. However, as many decisions carry social implications, for AI to be a reliable assistant for decision-making it is crucial that it is able to capture the balance between self-interest and the interest of others. We investigate the ability of three of the most advanced chatbots to predict dictator game decisions across 78 experiments with human participants from 12 countries. We find that only GPT-4 (not Bard nor Bing) correctly captures qualitative behavioral patterns, identifying three major classes of behavior: self-interested, inequity-averse, and fully altruistic. Nonetheless, GPT-4 consistently overestimates other-regarding behavior, inflating the proportion of inequity-averse and fully altruistic participants. This bias has significant implications for AI developers and users.
    
[^11]: 使用强化学习优化对抗目标下的信用额度调整

    Optimizing Credit Limit Adjustments Under Adversarial Goals Using Reinforcement Learning. (arXiv:2306.15585v1 [q-fin.GN])

    [http://arxiv.org/abs/2306.15585](http://arxiv.org/abs/2306.15585)

    本研究使用强化学习技术，通过平衡最大化投资组合收入和最小化准备金的对抗目标，自动化寻找最优信用卡额度调整策略。

    

    强化学习已经在很多问题中得到应用，从具有确定性环境的视频游戏到具有随机场景的投资组合和运营管理；然而，在银行问题中对这些方法的测试尝试很少。在本研究中，我们试图通过使用强化学习技术找到并自动化最优信用卡额度调整策略。具体而言，由于有历史数据可用，我们考虑每个客户的两种可能操作，即增加或保持个人当前的信用额度。为了找到这个策略，我们首先将这个决策问题形式化为一个优化问题，在其中最大化预期利润；因此，我们平衡了两个对抗目标：最大化投资组合的收入和最小化投资组合的准备金。其次，考虑到我们问题的特殊性，我们使用了离线学习策略，以基于历史数据模拟行动的影响。

    Reinforcement learning has been explored for many problems, from video games with deterministic environments to portfolio and operations management in which scenarios are stochastic; however, there have been few attempts to test these methods in banking problems. In this study, we sought to find and automatize an optimal credit card limit adjustment policy by employing reinforcement learning techniques. In particular, because of the historical data available, we considered two possible actions per customer, namely increasing or maintaining an individual's current credit limit. To find this policy, we first formulated this decision-making question as an optimization problem in which the expected profit was maximized; therefore, we balanced two adversarial goals: maximizing the portfolio's revenue and minimizing the portfolio's provisions. Second, given the particularities of our problem, we used an offline learning strategy to simulate the impact of the action based on historical data f
    
[^12]: 随机全球价值链中上游和下游之间的相关性

    Correlation between upstreamness and downstreamness in random global value chains. (arXiv:2303.06603v1 [stat.AP])

    [http://arxiv.org/abs/2303.06603](http://arxiv.org/abs/2303.06603)

    本文研究了全球价值链中产业和国家的上游和下游，发现同一产业部门的上游和下游之间存在正相关性。

    This paper studies the upstreamness and downstreamness of industries and countries in global value chains, and finds a positive correlation between upstreamness and downstreamness of the same industrial sector.

    本文关注全球价值链中产业和国家的上游和下游。上游和下游分别衡量产业部门与最终消费和初级输入之间的平均距离，并基于最常用的全球投入产出表数据库（例如世界投入产出数据库（WIOD））进行计算。最近，Antr\`as和Chor在1995-2011年的数据中报告了一个令人困惑和反直觉的发现，即（在国家层面上）上游似乎与下游呈正相关，相关斜率接近+1。这种效应随时间和跨国家稳定存在，并已得到后续分析的确认和验证。我们分析了一个简单的随机投入产出表模型，并展示了在最小和现实的结构假设下，同一产业部门的上游和下游之间存在正相关性，具有相关性。

    This paper is concerned with upstreamness and downstreamness of industries and countries in global value chains. Upstreamness and downstreamness measure respectively the average distance of an industrial sector from final consumption and from primary inputs, and they are computed from based on the most used global Input-Output tables databases, e.g., the World Input-Output Database (WIOD). Recently, Antr\`as and Chor reported a puzzling and counter-intuitive finding in data from the period 1995-2011, namely that (at country level) upstreamness appears to be positively correlated with downstreamness, with a correlation slope close to $+1$. This effect is stable over time and across countries, and it has been confirmed and validated by later analyses. We analyze a simple model of random Input/Output tables, and we show that, under minimal and realistic structural assumptions, there is a positive correlation between upstreamness and downstreamness of the same industrial sector, with corre
    

