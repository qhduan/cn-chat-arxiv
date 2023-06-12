# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Not to Spoof.](http://arxiv.org/abs/2306.06087) | 该论文提出了学习不进行欺骗的方法，并考虑了智能股票交易代理的欺诈行为识别和避免。 |
| [^2] | [FinGPT: Open-Source Financial Large Language Models.](http://arxiv.org/abs/2306.06031) | FinGPT是一个开源的金融大型语言模型，提供了可访问和透明的资源来开发金融LLMs，其重要性在于自动数据筛选管道和轻量级低秩适应技术。 |
| [^3] | [Agent market orders representation through a contrastive learning approach.](http://arxiv.org/abs/2306.05987) | 通过对比学习方法，本研究构建了一个自监督学习模型，用于学习代理市场订单的表示。进一步地，我们使用K均值聚类算法对代理订单的学习表示向量进行聚类，以确定每个簇中的不同行为类型。 |
| [^4] | [An Empirical Analysis of the Effect of Ballot Truncation on Ranked-Choice Electoral Outcomes.](http://arxiv.org/abs/2306.05966) | 本研究分析了1171个真实世界的排名选举数据，发现如果截选级别至少为三，则限制投票上的候选人数量很少影响选举获胜者。 |
| [^5] | [The Relationship Between Burnout Operators with the Functions of Family Tehran Banking Melli Iran Bank in 2015.](http://arxiv.org/abs/2306.05867) | 本研究探讨了伊朗国家银行员工的职业倦怠与家庭功能之间的关系。 |
| [^6] | [Interbank Decisions and Margins of Stability: an Agent-Based Stock-Flow Consistent Approach.](http://arxiv.org/abs/2306.05860) | 本篇论文以代理人模型为基础，研究了现代支付系统的运作，发现赤字银行和盈余银行之间的成熟度错配威胁了银行间市场的有效性和金融稳定，同时传统货币政策的有效性也受到了影响。 |
| [^7] | [Causality between Sentiment and Cryptocurrency Prices.](http://arxiv.org/abs/2306.05803) | 本研究通过将主题建模和情感分析相结合，建立了关于加密货币的多个叙述，并发现这些叙述与加密货币价格之间存在强大的联系。 |
| [^8] | [The far-reaching effects of bombing on fertility in mid-20th century Japan.](http://arxiv.org/abs/2306.05770) | 本研究探究了空袭对20世纪日本生育率的深远影响，并证明了战争破坏的区域影响即使在未直接受影响的地区也存在。 |
| [^9] | [Monte Carlo simulation for Barndorff-Nielsen and Shephard model under change of measure.](http://arxiv.org/abs/2306.05750) | 本文基于测度变换提出两种模拟方法，解决了Barndorff-Nielsen和Shephard模型下具有无限积极跳跃但非鞅情况下计算期权价格的问题。 |
| [^10] | [Maximally Machine-Learnable Portfolios.](http://arxiv.org/abs/2306.05568) | 本文通过 MACE 算法，以随机森林和受限岭回归优化组合权重，实现了最大程度的可预测性和盈利能力，适用于任何预测算法和预测器集，可以处理大型组合。 |
| [^11] | [Deep Attentive Survival Analysis in Limit Order Books: Estimating Fill Probabilities with Convolutional-Transformers.](http://arxiv.org/abs/2306.05479) | 本文提出了一种基于深度学习的生存分析方法，使用卷积-Transformer估计限价订单簿中放置于不同层级的限价订单的成交时间分布，相比其他方法表现显著优越。 |
| [^12] | [Equilibrium in Functional Stochastic Games with Mean-Field Interaction.](http://arxiv.org/abs/2306.05433) | 该论文提出了一种新的方法来明确推导出带有平均场相互作用的功能随机博弈的纳什均衡，同时证明了均衡的收敛性和存在的条件比有限玩家博弈的条件更少。 |
| [^13] | [Hierarchical forecasting for aggregated curves with an application to day-ahead electricity price auctions.](http://arxiv.org/abs/2305.16255) | 本文提出了一种协调方法来提高聚合曲线的预测准确性，并在德国日前电力竞拍市场进行了实证研究。 |
| [^14] | [How to handle the COS method for option pricing.](http://arxiv.org/abs/2303.16012) | 介绍了用于欧式期权定价的 Fourier余弦展开 (COS) 方法，通过指定截断范围和项数N进行逼近，文章提出明确的N的上界，对密度平滑并指数衰减的情况，COS方法的收敛阶数至少是指数收敛阶数。 |
| [^15] | [Ordered Reference Dependent Choice.](http://arxiv.org/abs/2105.12915) | 本文研究了参考依赖性选择在捕捉风险、时间和社会偏好方面的应用，从而揭示了参考依赖性选择在标准行为假设中的内在依赖性。 |

# 详细

[^1]: 学会不进行欺骗

    Learning Not to Spoof. (arXiv:2306.06087v1 [cs.LG])

    [http://arxiv.org/abs/2306.06087](http://arxiv.org/abs/2306.06087)

    该论文提出了学习不进行欺骗的方法，并考虑了智能股票交易代理的欺诈行为识别和避免。

    

    随着基于强化学习（RL）的智能交易代理越来越普及，确保RL代理遵守法律、法规和人类行为期望变得更加重要。本文考虑了一系列实验，其中智能股票交易代理最大化利润，但可能无意中学会欺骗其参与的市场。本文首先引入手动编码的欺诈代理到一个多代理市场模拟中，并学习识别欺诈活动序列。然后，本文将手动编码的欺骗交易员替换为一个简单的最大化利润的RL代理，观察它是否会独立地学习欺诈行为，并尝试避免它。

    As intelligent trading agents based on reinforcement learning (RL) gain prevalence, it becomes more important to ensure that RL agents obey laws, regulations, and human behavioral expectations. There is substantial literature concerning the aversion of obvious catastrophes like crashing a helicopter or bankrupting a trading account, but little around the avoidance of subtle non-normative behavior for which there are examples, but no programmable definition. Such behavior may violate legal or regulatory, rather than physical or monetary, constraints.  In this article, I consider a series of experiments in which an intelligent stock trading agent maximizes profit but may also inadvertently learn to spoof the market in which it participates. I first inject a hand-coded spoofing agent to a multi-agent market simulation and learn to recognize spoofing activity sequences. Then I replace the hand-coded spoofing trader with a simple profit-maximizing RL agent and observe that it independently 
    
[^2]: FinGPT：开源金融大型语言模型

    FinGPT: Open-Source Financial Large Language Models. (arXiv:2306.06031v1 [q-fin.ST])

    [http://arxiv.org/abs/2306.06031](http://arxiv.org/abs/2306.06031)

    FinGPT是一个开源的金融大型语言模型，提供了可访问和透明的资源来开发金融LLMs，其重要性在于自动数据筛选管道和轻量级低秩适应技术。

    

    大型语言模型（LLMs）展示了在各个领域革新自然语言处理任务的潜力，引起了金融领域的浓厚兴趣。获得高质量的金融数据是金融LLMs（FinLLMs）的第一个挑战。在这篇论文中，我们提出了一个针对金融领域的开源大型语言模型FinGPT。与专有模型不同，FinGPT采用数据为中心的方法，为研究人员和从业者提供可访问和透明的资源来开发他们的金融LLMs。我们强调自动数据筛选管道和轻量级低秩适应技术在建立FinGPT中的重要性。此外，我们展示了几个潜在的应用作为用户的基础，如机器顾问、算法交易和论 。

    Large language models (LLMs) have shown the potential of revolutionizing natural language processing tasks in diverse domains, sparking great interest in finance. Accessing high-quality financial data is the first challenge for financial LLMs (FinLLMs). While proprietary models like BloombergGPT have taken advantage of their unique data accumulation, such privileged access calls for an open-source alternative to democratize Internet-scale financial data.  In this paper, we present an open-source large language model, FinGPT, for the finance sector. Unlike proprietary models, FinGPT takes a data-centric approach, providing researchers and practitioners with accessible and transparent resources to develop their FinLLMs. We highlight the importance of an automatic data curation pipeline and the lightweight low-rank adaptation technique in building FinGPT. Furthermore, we showcase several potential applications as stepping stones for users, such as robo-advising, algorithmic trading, and l
    
[^3]: 一种基于对比学习方法的代理市场订单表示

    Agent market orders representation through a contrastive learning approach. (arXiv:2306.05987v1 [q-fin.ST])

    [http://arxiv.org/abs/2306.05987](http://arxiv.org/abs/2306.05987)

    通过对比学习方法，本研究构建了一个自监督学习模型，用于学习代理市场订单的表示。进一步地，我们使用K均值聚类算法对代理订单的学习表示向量进行聚类，以确定每个簇中的不同行为类型。

    

    本研究通过访问Euronext的CAC40数据中的标记订单，分析代理在市场中根据其下达的订单的行为。本研究构建了一个自监督学习模型，使用三元组损失来有效地学习代理市场订单的表示。通过获取这个学习表示，各种下游任务变得可行。本研究使用K均值聚类算法对代理订单的学习表示向量进行聚类，以确定每个簇中的不同行为类型。

    Due to the access to the labeled orders on the CAC40 data from Euronext, we are able to analyse agents' behaviours in the market based on their placed orders. In this study, we construct a self-supervised learning model using triplet loss to effectively learn the representation of agent market orders. By acquiring this learned representation, various downstream tasks become feasible. In this work, we utilise the K-means clustering algorithm on the learned representation vectors of agent orders to identify distinct behaviour types within each cluster.
    
[^4]: 一项关于截选票对排名选举结果影响的实证分析

    An Empirical Analysis of the Effect of Ballot Truncation on Ranked-Choice Electoral Outcomes. (arXiv:2306.05966v1 [econ.GN])

    [http://arxiv.org/abs/2306.05966](http://arxiv.org/abs/2306.05966)

    本研究分析了1171个真实世界的排名选举数据，发现如果截选级别至少为三，则限制投票上的候选人数量很少影响选举获胜者。

    

    在排名选举中，选民投出偏好选票，提供候选人的排名。排名选举方法通过使用选民偏好来模拟一系列的决胜选举来选择获胜者。某些使用排名选举的司法管辖区限制选民在选票上排名的候选人数量，实施我们所称的截选级别，即选民被允许排名的候选人数。在固定的选民偏好下，如果我们实施不同的截选级别，选举的获胜者可能会改变。我们使用1171个真实世界的排名选举数据库，实证分析了在排名选举中实施不同的截选级别可能产生的影响。我们的普遍发现是，如果截选级别至少为三，则限制投票上的候选人数量很少影响选举获胜者。

    In ranked-choice elections voters cast preference ballots which provide a voter's ranking of the candidates. The method of ranked-choice voting (RCV) chooses a winner by using voter preferences to simulate a series of runoff elections. Some jurisdictions which use RCV limit the number of candidates that voters can rank on the ballot, imposing what we term a truncation level, which is the number of candidates that voters are allowed to rank. Given fixed voter preferences, the winner of the election can change if we impose different truncation levels. We use a database of 1171 real-world ranked-choice elections to empirically analyze the potential effects of imposing different truncation levels in ranked-choice elections. Our general finding is that if the truncation level is at least three then restricting the number of candidates which can be ranked on the ballot rarely affects the election winner.
    
[^5]: 2015年伊朗国家银行银行家的职业倦怠与家庭功能的关系研究

    The Relationship Between Burnout Operators with the Functions of Family Tehran Banking Melli Iran Bank in 2015. (arXiv:2306.05867v1 [econ.GN])

    [http://arxiv.org/abs/2306.05867](http://arxiv.org/abs/2306.05867)

    本研究探讨了伊朗国家银行员工的职业倦怠与家庭功能之间的关系。

    

    本研究旨在探讨职业倦怠与伊朗国家银行员工家庭功能之间的关系。采用适当的科学方法，通过详细的问卷调查选择了该组织中的一些员工作为样本，并收集了有关职业倦怠和家庭功能的合适数据。样本人数为314名伊朗国家银行贷款主管，他们在银行的服务年限均超过5年，已婚，男性。使用Maslach职业倦怠问卷和家庭功能问卷进行描述性统计数据统计。研究结果表明

    In this study, the relationship between burnout and family functions of the Melli Iran Bank staff will be studied. A number of employees within the organization using appropriate scientific methods as the samples were selected by detailed questionnaire and the appropriate data is collected burnout and family functions. The method used descriptive statistical population used for this study consisted of 314 bank loan officers in branches of Melli Iran Bank of Tehran province and all the officials at the bank for >5 years of service at Melli Iran Bank branches in Tehran. They are married and men constitute the study population. The Maslach Burnout Inventory in the end internal to 0/90 alpha emotional exhaustion, depersonalization and low personal accomplishment Cronbach alpha of 0/79 and inventory by 0/71 within the last family to solve the problem 0/70, emotional response 0/51, touch 0/70, 0/69 affective involvement, roles, 0/59, 0/68 behavior is controlled. The results indicate that the
    
[^6]: 银行间决策和稳定边界: 一种基于代理人的股-流一致性方法的研究(arXiv:2306.05860v1 [econ.GN])

    Interbank Decisions and Margins of Stability: an Agent-Based Stock-Flow Consistent Approach. (arXiv:2306.05860v1 [econ.GN])

    [http://arxiv.org/abs/2306.05860](http://arxiv.org/abs/2306.05860)

    本篇论文以代理人模型为基础，研究了现代支付系统的运作，发现赤字银行和盈余银行之间的成熟度错配威胁了银行间市场的有效性和金融稳定，同时传统货币政策的有效性也受到了影响。

    

    本研究通过银行成熟度错配行为的视角，研究现代支付系统的运作，并探讨银行拒绝滚动短期银行间负债对金融稳定的影响。在基于代理人的股-流一致性框架下，银行在两个不同成熟期的银行间市场中进行交易，即隔夜市场和期限市场。我们比较了两种银行间匹配情景，以评估依赖于净稳定资金比率规定的银行特定成熟度目标，对银行间市场动态和传统货币政策有效性的影响。研究结果表明，赤字和盈余银行之间的成熟度错配损害了银行间市场的效率，并增加了对中央银行的常备措施的依赖。货币政策利率调控也变得不那么有效。该研究还揭示了银行业中的双重稳定性配置。

    This study investigates the functioning of modern payment systems through the lens of banks' maturity mismatch practices, and it examines the effects of banks' refusal to roll over short-term interbank liabilities on financial stability. Within an agent-based stock-flow consistent framework, banks can engage in two segments of the interbank market that differ in maturity, overnight and term. We compare two interbank matching scenarios to assess how bank-specific maturity targets, dependent on the dictates of the Net Stable Funding Ratio, impact the dynamics of the interbank market and the effectiveness of conventional monetary policies. The findings reveal that maturity misalignment between deficit and surplus banks compromises the interbank market's efficiency and increases reliance on the central bank's standing facilities. Monetary policy interest-rate steering practices also become less effective. The study also uncovers a dual stability-based configuration in the banking sector, r
    
[^7]: 情感和加密货币价格之间的因果关系

    Causality between Sentiment and Cryptocurrency Prices. (arXiv:2306.05803v1 [q-fin.CP])

    [http://arxiv.org/abs/2306.05803](http://arxiv.org/abs/2306.05803)

    本研究通过将主题建模和情感分析相结合，建立了关于加密货币的多个叙述，并发现这些叙述与加密货币价格之间存在强大的联系。

    

    本研究调查了微博平台（Twitter）传递的叙述与加密资产价值之间的关系。我们的研究采用了一种独特的技术，将短文本的主题建模与情感分析相结合，建立了关于加密货币的叙述。首先，我们使用了一种无监督的机器学习算法，从Twitter的大规模和嘈杂文本数据中发现潜在主题，然后我们揭示了4-5个与加密货币相关的叙述，包括与加密货币相关的金融投资、技术进步、金融和政治监管、加密资产和媒体报道。在许多情况下，我们注意到我们的叙述与加密货币价格之间存在强大的联系。我们的工作将最新的经济学创新——叙事经济学与主题建模和情感分析相结合的新领域联系起来，以关联消费者行为和叙述。

    This study investigates the relationship between narratives conveyed through microblogging platforms, namely Twitter, and the value of crypto assets. Our study provides a unique technique to build narratives about cryptocurrency by combining topic modelling of short texts with sentiment analysis. First, we used an unsupervised machine learning algorithm to discover the latent topics within the massive and noisy textual data from Twitter, and then we revealed 4-5 cryptocurrency-related narratives, including financial investment, technological advancement related to crypto, financial and political regulations, crypto assets, and media coverage. In a number of situations, we noticed a strong link between our narratives and crypto prices. Our work connects the most recent innovation in economics, Narrative Economics, to a new area of study that combines topic modelling and sentiment analysis to relate consumer behaviour to narratives.
    
[^8]: 二战期间轰炸对20世纪日本生育率的深远影响

    The far-reaching effects of bombing on fertility in mid-20th century Japan. (arXiv:2306.05770v1 [econ.GN])

    [http://arxiv.org/abs/2306.05770](http://arxiv.org/abs/2306.05770)

    本研究探究了空袭对20世纪日本生育率的深远影响，并证明了战争破坏的区域影响即使在未直接受影响的地区也存在。

    

    战争和冲突之后的生育变化在全球范围内得到了观察。本研究旨在研究区域战争破坏是否会影响战后生育，即使是未直接受到影响但靠近受损地区的地区。为了达到这个目的，我们利用了日本二战期间的空袭经历。利用1935年和1947年近畿地区的市町村级别生育数据以及城市空袭损失数据，我们发现轰炸对于15公里内城镇和乡村的战后生育率存在影响，尽管这些地区未直接受到损害。然而，间接影响的方向是混合的。估计结果表明，邻近城市的严重空袭增加了生育率，而较轻的空袭则降低了生育率。此外，拟实验法的结果表明，严重的空袭恐惧在战后期间增加了生育率。本研究为战后生育变化的文献提供了证据，即轰炸对于生育率有深远影响，即使是被战争损害间接影响的地区。

    Fertility changes after wars and conflicts have been observed worldwide. This study examines whether regional war damage affects postwar fertility even in areas that were not directly affected but were close to the damaged areas. In order to accomplish this, we exploit the air-raid experience in Japan during World War II. Using the municipality-level fertility data in the Kinki region in 1935 and 1947 and the data on damages from air raids in cities, we find the effects of bombing on postwar fertility in towns and villages within 15 kilometers, despite no direct damages. However, the direction of the indirect effects is mixed. The estimation results suggest that severe air raids in neighboring cities increased fertility, whereas minor air raids decreased it. Moreover, the results of the quasi-experimental approach indicate that intense fears of air raids increased the fertility rate in the postwar period. Our study contributes to the literature on fertility changes in the postwar perio
    
[^9]: Barndorff-Nielsen和Shephard模型在测度变换下的蒙特卡洛模拟

    Monte Carlo simulation for Barndorff-Nielsen and Shephard model under change of measure. (arXiv:2306.05750v1 [q-fin.CP])

    [http://arxiv.org/abs/2306.05750](http://arxiv.org/abs/2306.05750)

    本文基于测度变换提出两种模拟方法，解决了Barndorff-Nielsen和Shephard模型下具有无限积极跳跃但非鞅情况下计算期权价格的问题。

    

    Barndorff-Nielsen和Shephard模型是代表性的跳跃型随机波动率模型。然而，在具有无限积极跳跃但非鞅情况下，尚无方法计算期权价格。本文针对该情况，基于测度变换开发了两种模拟方法，并进行了一些数值实验。

    The Barndorff-Nielsen and Shephard model is a representative jump-type stochastic volatility model. Still, no method exists to compute option prices numerically for the non-martingale case with infinite active jumps. We develop two simulation methods for such a case under change of measure and conduct some numerical experiments.
    
[^10]: 最大机器学习组合的构建方法

    Maximally Machine-Learnable Portfolios. (arXiv:2306.05568v1 [econ.EM])

    [http://arxiv.org/abs/2306.05568](http://arxiv.org/abs/2306.05568)

    本文通过 MACE 算法，以随机森林和受限岭回归优化组合权重，实现了最大程度的可预测性和盈利能力，适用于任何预测算法和预测器集，可以处理大型组合。

    

    对于股票回报，任何形式的可预测性都可以增强调整风险后的盈利能力。本文开发了一种协作机器学习算法，优化组合权重，以使得合成证券最大程度的可预测。具体来说，我们引入了MACE，Alternating Conditional Expectations的多元扩展，通过在方程的一侧使用随机森林和受限岭回归在另一侧实现了上述目标。相较于Lo和MacKinlay的最大可预测组合方法，本文有两个关键改进。第一，它适用于任何（非线性）预测算法和预测器集。第二，它可以处理大型组合。我们进行了日频和月频的实验，并发现在使用很少的条件信息时，可预测性和盈利能力显著增加。有趣的是，可预测性在好时和坏时都存在，并且MACE成功地导航了两者。

    When it comes to stock returns, any form of predictability can bolster risk-adjusted profitability. We develop a collaborative machine learning algorithm that optimizes portfolio weights so that the resulting synthetic security is maximally predictable. Precisely, we introduce MACE, a multivariate extension of Alternating Conditional Expectations that achieves the aforementioned goal by wielding a Random Forest on one side of the equation, and a constrained Ridge Regression on the other. There are two key improvements with respect to Lo and MacKinlay's original maximally predictable portfolio approach. First, it accommodates for any (nonlinear) forecasting algorithm and predictor set. Second, it handles large portfolios. We conduct exercises at the daily and monthly frequency and report significant increases in predictability and profitability using very little conditioning information. Interestingly, predictability is found in bad as well as good times, and MACE successfully navigates
    
[^11]: 深度关注生存分析在限价订单上的应用：利用卷积-Transformer估计成交概率(arXiv:2306.05479v1[q-fin.ST])

    Deep Attentive Survival Analysis in Limit Order Books: Estimating Fill Probabilities with Convolutional-Transformers. (arXiv:2306.05479v1 [q-fin.ST])

    [http://arxiv.org/abs/2306.05479](http://arxiv.org/abs/2306.05479)

    本文提出了一种基于深度学习的生存分析方法，使用卷积-Transformer估计限价订单簿中放置于不同层级的限价订单的成交时间分布，相比其他方法表现显著优越。

    

    执行策略中的关键决定之一是选择被动（提供流动性）或积极（吸纳流动性）订单以在限价订单簿中执行交易。这取决于放置于限价订单簿中的被动限价订单成交概率。本文提出了深度学习方法来估计放置于限价订单簿中不同层级的限价订单的成交时间。我们开发了一种新的生存分析模型，将限价订单的时变特征映射为其成交时间的分布。我们的方法基于卷积-Transformer编码器和单调神经网络解码器。我们使用适当的评分规则来比较我们的方法与生存分析中的其他方法，并进行可解释性分析以了解用于计算成交概率的特征的信息量。我们的方法在生存分析文献中显著优于通常使用的方法。最后，我们进行了统计分析……（未完整翻译）

    One of the key decisions in execution strategies is the choice between a passive (liquidity providing) or an aggressive (liquidity taking) order to execute a trade in a limit order book (LOB). Essential to this choice is the fill probability of a passive limit order placed in the LOB. This paper proposes a deep learning method to estimate the filltimes of limit orders posted in different levels of the LOB. We develop a novel model for survival analysis that maps time-varying features of the LOB to the distribution of filltimes of limit orders. Our method is based on a convolutional-Transformer encoder and a monotonic neural network decoder. We use proper scoring rules to compare our method with other approaches in survival analysis, and perform an interpretability analysis to understand the informativeness of features used to compute fill probabilities. Our method significantly outperforms those typically used in survival analysis literature. Finally, we carry out a statistical analysi
    
[^12]: 带有平均场相互作用的功能随机博弈的均衡

    Equilibrium in Functional Stochastic Games with Mean-Field Interaction. (arXiv:2306.05433v1 [math.OC])

    [http://arxiv.org/abs/2306.05433](http://arxiv.org/abs/2306.05433)

    该论文提出了一种新的方法来明确推导出带有平均场相互作用的功能随机博弈的纳什均衡，同时证明了均衡的收敛性和存在的条件比有限玩家博弈的条件更少。

    

    我们考虑了一个一般的有限玩家带有平均场相互作用的随机博弈，其中线性二次成本函数包括作用于$L^2$中的控制的线性算子。我们提出了一种新的方法，通过将相关的一阶条件减少到第二类随机Fredholm方程组的系统，并推导出它们的闭形式解来明确推导出了博弈的纳什均衡。此外，通过证明随机Fredholm方程组的稳定性结果，我们推导出了$N$人博弈的均衡收敛到相应的平均场均衡。作为一个副产品，我们还推导出了平均场博弈的$\varepsilon$-纳什均衡，在这种情况下，它具有价值，因为我们表明，在平均场极限存在均衡的条件比有限玩家博弈的条件要少。最后，我们将我们的一般框架应用于解决各种例子。

    We consider a general class of finite-player stochastic games with mean-field interaction, in which the linear-quadratic cost functional includes linear operators acting on controls in $L^2$. We propose a novel approach for deriving the Nash equilibrium of the game explicitly in terms of operator resolvents, by reducing the associated first order conditions to a system of stochastic Fredholm equations of the second kind and deriving their closed form solution. Furthermore, by proving stability results for the system of stochastic Fredholm equations we derive the convergence of the equilibrium of the $N$-player game to the corresponding mean-field equilibrium. As a by-product we also derive an $\varepsilon$-Nash equilibrium for the mean-field game, which is valuable in this setting as we show that the conditions for existence of an equilibrium in the mean-field limit are less restrictive than in the finite-player game. Finally we apply our general framework to solve various examples, su
    
[^13]: 层次预测聚合曲线并应用于日前电力竞拍

    Hierarchical forecasting for aggregated curves with an application to day-ahead electricity price auctions. (arXiv:2305.16255v1 [stat.AP])

    [http://arxiv.org/abs/2305.16255](http://arxiv.org/abs/2305.16255)

    本文提出了一种协调方法来提高聚合曲线的预测准确性，并在德国日前电力竞拍市场进行了实证研究。

    

    聚合曲线在经济和金融中很常见，最突出的例子是供给和需求曲线。我们发现所有聚合曲线都具有内在的层次结构，因此可以使用层次协调方法来提高预测准确性。我们提供了聚合曲线如何构建或解构的深入理论，并得出这些方法在弱假设下是等效的结论。我们考虑了多种聚合曲线的协调方法，包括已经建立的自下而上、自上而下和线性最优协调方法。我们还提出了一种新的基准协调方法，称为“聚合-下”，其复杂度类似于自下而上和自上而下方法，但在这种设置中往往提供更好的准确性。我们对德国日前电力竞拍市场进行了实证预测研究，预测了需求和供给曲线，并对其均衡性进行了分析。

    Aggregated curves are common structures in economics and finance, and the most prominent examples are supply and demand curves. In this study, we exploit the fact that all aggregated curves have an intrinsic hierarchical structure, and thus hierarchical reconciliation methods can be used to improve the forecast accuracy. We provide an in-depth theory on how aggregated curves can be constructed or deconstructed, and conclude that these methods are equivalent under weak assumptions. We consider multiple reconciliation methods for aggregated curves, including previously established bottom-up, top-down, and linear optimal reconciliation approaches. We also present a new benchmark reconciliation method called 'aggregated-down' with similar complexity to bottom-up and top-down approaches, but it tends to provide better accuracy in this setup. We conducted an empirical forecasting study on the German day-ahead power auction market by predicting the demand and supply curves, where their equili
    
[^14]: 如何处理用于期权定价的 COS 方法

    How to handle the COS method for option pricing. (arXiv:2303.16012v1 [q-fin.CP])

    [http://arxiv.org/abs/2303.16012](http://arxiv.org/abs/2303.16012)

    介绍了用于欧式期权定价的 Fourier余弦展开 (COS) 方法，通过指定截断范围和项数N进行逼近，文章提出明确的N的上界，对密度平滑并指数衰减的情况，COS方法的收敛阶数至少是指数收敛阶数。

    

    Fourier余弦展开（COS）方法用于高效地计算欧式期权价格。要应用COS方法，必须指定两个参数：对数收益率密度的截断范围和用余弦级数逼近截断密度的项数N。如何选择截断范围已经为人所知。在这里，我们还能找到一个明确的并且有用的项数N的界限。我们还进一步表明，如果密度是平滑的并且呈指数衰减，则COS方法至少具有指数收敛阶数。但是，如果密度平滑但有重尾巴，就像在有限矩阵log稳定模型中一样，则COS方法没有指数收敛阶数。数值实验确认了理论发现。

    The Fourier cosine expansion (COS) method is used for pricing European options numerically very efficiently. To apply the COS method, one has to specify two parameters: a truncation range for the density of the log-returns and a number of terms N to approximate the truncated density by a cosine series. How to choose the truncation range is already known. Here, we are able to find an explicit and useful bound for N as well. We further show that the COS method has at least an exponential order of convergence if the density is smooth and decays exponentially. But, if the density is smooth and has heavy tails like in the Finite Moment Log Stable model, the COS method has not an exponential order of convergence. Numerical experiments confirm the theoretical findings.
    
[^15]: 有序的参考依赖性选择

    Ordered Reference Dependent Choice. (arXiv:2105.12915v3 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2105.12915](http://arxiv.org/abs/2105.12915)

    本文研究了参考依赖性选择在捕捉风险、时间和社会偏好方面的应用，从而揭示了参考依赖性选择在标准行为假设中的内在依赖性。

    

    本文研究了结构假设（比如期望效用和指数折扣）的违反如何与源自参考依赖偏好的合理性违反相连接，即使行为在参考固定情况下完全标准。针对不同的选择领域，参考依赖的广义取代任意行为假设，并且产生内生地确定参考备选方案的线性顺序，进而确定选择问题的偏好参数。通过使用例如效用函数的凹性和折扣因子的级别等已知技术，可以捕捉偏好的变化。该框架允许我们集体研究风险、时间和社会偏好，其中看似独立的反常现象通过参考依赖选择的镜头相互连接。

    This paper studies how violations of structural assumptions like expected utility and exponential discounting can be connected to rationality violations that arise from reference-dependent preferences, even if behavior is fully standard when the reference is fixed. A reference-dependent generalization of arbitrarily behavioral postulates captures changing preferences across choice domains. It gives rise to a linear order that endogenously determines reference alternatives, which in turn determines the preference parameters for a choice problem. With canonical models as backbones, preference changes are captured using known technologies like the concavity of utility functions and the levels of discount factors. The framework allows us to study risk, time, and social preferences collectively, where seemingly independent anomalies are interconnected through the lens of reference-dependent choice.
    

