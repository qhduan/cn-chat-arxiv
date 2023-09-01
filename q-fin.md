# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Linking microblogging sentiments to stock price movement: An application of GPT-4.](http://arxiv.org/abs/2308.16771) | 本文研究了GPT-4在通过微博情感分析预测股票价格运动方面的潜在改进，并开发了一种新的方法用于情境情感分析，结果显示GPT-4在准确性上超过了BERT和买入持有策略。 |
| [^2] | [Improving Robustness and Accuracy of Ponzi Scheme Detection on Ethereum Using Time-Dependent Features.](http://arxiv.org/abs/2308.16391) | 这篇论文提出了一种基于交易的方法来提高以太坊上庞氏骗局的检测鲁棒性和准确性。现有的方法主要基于智能合约源代码或操作码进行检测，但缺乏鲁棒性。通过分析交易数据，可以更有效地识别庞氏骗局，因为交易更难伪装。 |
| [^3] | [A new adaptive pricing framework for perpetual protocols using liquidity curves and on-chain oracles.](http://arxiv.org/abs/2308.16256) | 本文介绍了一种使用流动性曲线和链上预言机的新型自适应定价框架，可稳定和可预测地为永续合约定价，并根据当前市场条件对交易者报价费用。通过数学建模和与现有解决方案的比较，展示了该框架的操作和优势。同时，探讨了增强去中心化永续协议整体效率的其他功能。 |
| [^4] | [Hedging Forecast Combinations With an Application to the Random Forest.](http://arxiv.org/abs/2308.15384) | 本文提出了一种使用对冲进行预测组合的通用方法，并应用于随机森林。研究表明，该方法在14个数据集中相对于标准方法在组合树预测形成加权随机森林方面具有改进的外样本表现。 |
| [^5] | [Cognitive Aging and Labor Share.](http://arxiv.org/abs/2308.14982) | 该研究将劳动份额的下降与认知衰老联系起来，提出了一个新颖的宏观经济模型。模型表明，工业化导致人口老龄化，老龄消费者认知能力的下降减少了对新产出变体的需求，从而导致劳动份额的下降。 |
| [^6] | [Agree to Disagree: Measuring Hidden Dissents in FOMC Meetings.](http://arxiv.org/abs/2308.10131) | 该研究使用自我关注模块的深度学习模型，根据FOMC会议的异议记录和会议记录，测量了每位成员在每个会议中的异议程度。研究发现，尽管异议很少见，成员们经常对政策决策持保留意见。异议程度主要受到当前或预测的宏观经济数据的影响，而成员的个人特征几乎不起作用。此外，研究还发现了会议之间成员的演讲与随后会议的异议程度之间存在弱相关性。最后，研究发现，每当货币政策行动更加激进时，异议程度会增加。 |
| [^7] | [Dynamic Transportation of Economic Agents.](http://arxiv.org/abs/2303.12567) | 本文通过提出新的方法，解决了之前在某些异质性代理人不完全市场模型的宏观经济均衡解决方案的问题。 |
| [^8] | [Unbiased estimators for the Heston model with stochastic interest rates.](http://arxiv.org/abs/2301.12072) | 本研究结合了无偏估计器和具有随机利率的Heston模型，通过开发半精确的对数欧拉方案，证明了其收敛率为O(h)，适用于多种模型。 |

# 详细

[^1]: 将微博情绪与股票价格运动相关联：GPT-4的应用

    Linking microblogging sentiments to stock price movement: An application of GPT-4. (arXiv:2308.16771v1 [q-fin.ST])

    [http://arxiv.org/abs/2308.16771](http://arxiv.org/abs/2308.16771)

    本文研究了GPT-4在通过微博情感分析预测股票价格运动方面的潜在改进，并开发了一种新的方法用于情境情感分析，结果显示GPT-4在准确性上超过了BERT和买入持有策略。

    

    本文研究了GPT-4语言学习模型（LLM）在对比BERT模型中，通过对微博消息的情感分析来建模苹果和特斯拉2017年同日日常股价的潜在改进。我们记录了每日调整后的收盘价格，并将其转化为上涨或下跌的动态。利用两种LLM从Stocktwits平台上提取每日的情绪。我们开发了一种新颖的方法，通过综合提示来进行情境情感分析，以发掘现代LLM的真实能力。这使我们能够仔细提取情绪、感知优势或劣势，以及与分析公司相关性。采用逻辑回归评估提取的消息内容是否反映了股票价格的变动。结果表明，GPT-4在五个月中超过了BERT，显著提高了准确性，并且明显超过了天真的买入持有策略，达到了最高的准确性。

    This paper investigates the potential improvement of the GPT-4 Language Learning Model (LLM) in comparison to BERT for modeling same-day daily stock price movements of Apple and Tesla in 2017, based on sentiment analysis of microblogging messages. We recorded daily adjusted closing prices and translated them into up-down movements. Sentiment for each day was extracted from messages on the Stocktwits platform using both LLMs. We develop a novel method to engineer a comprehensive prompt for contextual sentiment analysis which unlocks the true capabilities of modern LLM. This enables us to carefully retrieve sentiments, perceived advantages or disadvantages, and the relevance towards the analyzed company. Logistic regression is used to evaluate whether the extracted message contents reflect stock price movements. As a result, GPT-4 exhibited substantial accuracy, outperforming BERT in five out of six months and substantially exceeding a naive buy-and-hold strategy, reaching a peak accurac
    
[^2]: 提高以太坊上庞氏骗局检测的鲁棒性和准确性的方法

    Improving Robustness and Accuracy of Ponzi Scheme Detection on Ethereum Using Time-Dependent Features. (arXiv:2308.16391v1 [cs.CR])

    [http://arxiv.org/abs/2308.16391](http://arxiv.org/abs/2308.16391)

    这篇论文提出了一种基于交易的方法来提高以太坊上庞氏骗局的检测鲁棒性和准确性。现有的方法主要基于智能合约源代码或操作码进行检测，但缺乏鲁棒性。通过分析交易数据，可以更有效地识别庞氏骗局，因为交易更难伪装。

    

    区块链的快速发展导致越来越多的资金涌入加密货币市场，也吸引了近年来网络犯罪分子的兴趣。庞氏骗局作为一种老式的欺诈行为，现在也流行于区块链上，给许多加密货币投资者造成了巨大的财务损失。现有文献中已经提出了一些庞氏骗局检测方法，其中大多数是基于智能合约的源代码或操作码进行检测的。虽然基于合约代码的方法在准确性方面表现出色，但它缺乏鲁棒性：首先，大部分以太坊上的合约源代码并不公开可用；其次，庞氏骗局开发者可以通过混淆操作码或者创造新的分配逻辑来欺骗基于合约代码的检测模型（因为这些模型仅在现有的庞氏逻辑上进行训练）。基于交易的方法可以提高检测的鲁棒性，因为与智能合约不同，交易更加难以伪装。

    The rapid development of blockchain has led to more and more funding pouring into the cryptocurrency market, which also attracted cybercriminals' interest in recent years. The Ponzi scheme, an old-fashioned fraud, is now popular on the blockchain, causing considerable financial losses to many crypto-investors. A few Ponzi detection methods have been proposed in the literature, most of which detect a Ponzi scheme based on its smart contract source code or opcode. The contract-code-based approach, while achieving very high accuracy, is not robust: first, the source codes of a majority of contracts on Ethereum are not available, and second, a Ponzi developer can fool a contract-code-based detection model by obfuscating the opcode or inventing a new profit distribution logic that cannot be detected (since these models were trained on existing Ponzi logics only). A transaction-based approach could improve the robustness of detection because transactions, unlike smart contracts, are harder t
    
[^3]: 一种使用流动性曲线和链上预言机的永续协议自适应定价框架

    A new adaptive pricing framework for perpetual protocols using liquidity curves and on-chain oracles. (arXiv:2308.16256v1 [q-fin.TR])

    [http://arxiv.org/abs/2308.16256](http://arxiv.org/abs/2308.16256)

    本文介绍了一种使用流动性曲线和链上预言机的新型自适应定价框架，可稳定和可预测地为永续合约定价，并根据当前市场条件对交易者报价费用。通过数学建模和与现有解决方案的比较，展示了该框架的操作和优势。同时，探讨了增强去中心化永续协议整体效率的其他功能。

    

    本文介绍了一种创新的机制，用于根据当前市场条件为永续合约定价和对交易者报价费用。该方法利用流动性曲线和链上预言机建立了一种新的自适应定价框架，考虑了各种因素，确保定价的稳定性和可预测性。该框架利用抛物线和S形函数报价价格和费用，考虑了流动性、活跃的多头和空头头寸以及利用率。本文通过数学建模详细解释了自适应定价框架与流动性曲线的操作，并将其与现有解决方案进行了比较。此外，我们探讨了增强去中心化永续协议整体效率的其他功能。

    This whitepaper introduces an innovative mechanism for pricing perpetual contracts and quoting fees to traders based on current market conditions. The approach employs liquidity curves and on-chain oracles to establish a new adaptive pricing framework that considers various factors, ensuring pricing stability and predictability. The framework utilizes parabolic and sigmoid functions to quote prices and fees, accounting for liquidity, active long and short positions, and utilization. This whitepaper provides a detailed explanation of how the adaptive pricing framework, in conjunction with liquidity curves, operates through mathematical modeling and compares it to existing solutions. Furthermore, we explore additional features that enhance the overall efficiency of the decentralized perpetual protocol.
    
[^4]: 使用对冲进行预测组合，并应用于随机森林

    Hedging Forecast Combinations With an Application to the Random Forest. (arXiv:2308.15384v1 [stat.ME])

    [http://arxiv.org/abs/2308.15384](http://arxiv.org/abs/2308.15384)

    本文提出了一种使用对冲进行预测组合的通用方法，并应用于随机森林。研究表明，该方法在14个数据集中相对于标准方法在组合树预测形成加权随机森林方面具有改进的外样本表现。

    

    本文提出了一个通用的、高层次的方法来生成预测组合，如果能够获得两个总体量：个体预测误差向量的均值向量和协方差矩阵，该方法能够以均方预测误差为标准，得到最优线性组合的预测。我们指出，这个问题与一个均值-方差投资组合构建问题是相同的，其中投资组合权重对应于预测组合权重。我们允许负的预测权重，并将这样的权重解释为对估计量之间的过度估计风险和不足估计风险进行对冲。这种解释直接来自投资组合类比。我们在14个数据集中演示了我们的方法相对于标准方法在组合树预测形成加权随机森林方面的外样本表现改进。

    This papers proposes a generic, high-level methodology for generating forecast combinations that would deliver the optimal linearly combined forecast in terms of the mean-squared forecast error if one had access to two population quantities: the mean vector and the covariance matrix of the vector of individual forecast errors. We point out that this problem is identical to a mean-variance portfolio construction problem, in which portfolio weights correspond to forecast combination weights. We allow negative forecast weights and interpret such weights as hedging over and under estimation risks across estimators. This interpretation follows directly as an implication of the portfolio analogy. We demonstrate our method's improved out-of-sample performance relative to standard methods in combining tree forecasts to form weighted random forests in 14 data sets.
    
[^5]: 认知衰老与劳动份额

    Cognitive Aging and Labor Share. (arXiv:2308.14982v1 [econ.GN])

    [http://arxiv.org/abs/2308.14982](http://arxiv.org/abs/2308.14982)

    该研究将劳动份额的下降与认知衰老联系起来，提出了一个新颖的宏观经济模型。模型表明，工业化导致人口老龄化，老龄消费者认知能力的下降减少了对新产出变体的需求，从而导致劳动份额的下降。

    

    劳动份额，即经济产出的工资比例，在工业化国家中不可理解地在下降。虽然许多之前的研究试图通过经济因素来解释这种下降，但我们的新颖方法将这种下降与生物因素联系起来。具体而言，我们提出了一个理论宏观经济模型，劳动份额反映了劳动力自动化现有产出和消费者需求新的依赖人力劳动的产出变体之间的动态平衡。工业化导致人口老龄化，虽然在工作年限内认知表现稳定，但之后急剧下降。因此，老龄消费者认知能力的下降减少了对新的产出变体的需求，导致劳动份额下降。我们的模型将劳动份额表达为中位数年龄的代数函数，并通过非线性随机回归在工业化经济体的历史数据上以惊人的准确性进行了验证。

    Labor share, the fraction of economic output accrued as wages, is inexplicably declining in industrialized countries. Whilst numerous prior works attempt to explain the decline via economic factors, our novel approach links the decline to biological factors. Specifically, we propose a theoretical macroeconomic model where labor share reflects a dynamic equilibrium between the workforce automating existing outputs, and consumers demanding new output variants that require human labor. Industrialization leads to an aging population, and while cognitive performance is stable in the working years it drops sharply thereafter. Consequently, the declining cognitive performance of aging consumers reduces the demand for new output variants, leading to a decline in labor share. Our model expresses labor share as an algebraic function of median age, and is validated with surprising accuracy on historical data across industrialized economies via non-linear stochastic regression.
    
[^6]: 持不同意见：测量FOMC会议中的隐藏异议

    Agree to Disagree: Measuring Hidden Dissents in FOMC Meetings. (arXiv:2308.10131v1 [econ.GN])

    [http://arxiv.org/abs/2308.10131](http://arxiv.org/abs/2308.10131)

    该研究使用自我关注模块的深度学习模型，根据FOMC会议的异议记录和会议记录，测量了每位成员在每个会议中的异议程度。研究发现，尽管异议很少见，成员们经常对政策决策持保留意见。异议程度主要受到当前或预测的宏观经济数据的影响，而成员的个人特征几乎不起作用。此外，研究还发现了会议之间成员的演讲与随后会议的异议程度之间存在弱相关性。最后，研究发现，每当货币政策行动更加激进时，异议程度会增加。

    

    基于1976年至2017年的FOMC投票异议记录和会议记录，我们开发了一个基于自我关注模块的深度学习模型，用于确定每个成员在每个会议中的异议程度。虽然异议很少见，但我们发现成员们经常对政策决策持保留意见。异议程度主要由当前或预测的宏观经济数据驱动，成员的个人特征几乎不起作用。我们还利用模型评估会议之间成员的演讲，并发现它们所揭示的异议程度与随后的会议异议程度之间存在弱相关性。最后，我们发现每当货币政策行动更加激进时，异议程度会增加。

    Based on a record of dissents on FOMC votes and transcripts of the meetings from 1976 to 2017, we develop a deep learning model based on self-attention modules to create a measure of the level of disagreement for each member in each meeting. While dissents are rare, we find that members often have reservations with the policy decision. The level of disagreement is mostly driven by current or predicted macroeconomic data, and personal characteristics of the members play almost no role. We also use our model to evaluate speeches made by members between meetings, and we find a weak correlation between the level of disagreement revealed in them and that of the following meeting. Finally, we find that the level of disagreement increases whenever monetary policy action is more aggressive.
    
[^7]: 经济主体的动态运输

    Dynamic Transportation of Economic Agents. (arXiv:2303.12567v1 [econ.GN])

    [http://arxiv.org/abs/2303.12567](http://arxiv.org/abs/2303.12567)

    本文通过提出新的方法，解决了之前在某些异质性代理人不完全市场模型的宏观经济均衡解决方案的问题。

    

    本文是在发现了一个共同的策略未能将某些异质性代理人不完全市场模型的宏观经济均衡定位到广泛引用的基准研究中而引发的。通过模仿Dumas和Lyasoff（2012）提出的方法，本文提供了一个新的描述，在面对不可保险的总体和个体风险的大量互动经济体代表的私人状态分布的运动定律。提出了一种新的算法，用于确定回报、最优私人配置和平衡状态下的人口运输，并在两个众所周知的基准研究中进行了测试。

    The paper was prompted by the surprising discovery that the common strategy, adopted in a large body of research, for producing macroeconomic equilibrium in certain heterogeneous-agent incomplete-market models fails to locate the equilibrium in a widely cited benchmark study. By mimicking the approach proposed by Dumas and Lyasoff (2012), the paper provides a novel description of the law of motion of the distribution over the range of private states of a large population of interacting economic agents faced with uninsurable aggregate and idiosyncratic risk. A new algorithm for identifying the returns, the optimal private allocations, and the population transport in the state of equilibrium is developed and is tested in two well known benchmark studies.
    
[^8]: 具有随机利率的Heston模型的无偏估计器

    Unbiased estimators for the Heston model with stochastic interest rates. (arXiv:2301.12072v2 [q-fin.CP] UPDATED)

    [http://arxiv.org/abs/2301.12072](http://arxiv.org/abs/2301.12072)

    本研究结合了无偏估计器和具有随机利率的Heston模型，通过开发半精确的对数欧拉方案，证明了其收敛率为O(h)，适用于多种模型。

    

    我们结合了Rhee和Glynn（Operations Research: 63(5), 1026-1043，2015）中的无偏估计器和具有随机利率的Heston模型。具体地，我们首先为具有随机利率的Heston模型开发了一个半精确的对数欧拉方案。然后，在一些温和的假设下，我们证明收敛率在L^2范数中是O(h)，其中h是步长。该结果适用于许多模型，如Heston-Hull-While模型，Heston-CIR模型和Heston-Black-Karasinski模型。数值实验支持我们的理论收敛率。

    We combine the unbiased estimators in Rhee and Glynn (Operations Research: 63(5), 1026-1043, 2015) and the Heston model with stochastic interest rates. Specifically, we first develop a semi-exact log-Euler scheme for the Heston model with stochastic interest rates. Then, under mild assumptions, we show that the convergence rate in the $L^2$ norm is $O(h)$, where $h$ is the step size. The result applies to a large class of models, such as the Heston-Hull-While model, the Heston-CIR model and the Heston-Black-Karasinski model. Numerical experiments support our theoretical convergence rate.
    

