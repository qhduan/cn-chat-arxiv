# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Using Machine Learning to Forecast Market Direction with Efficient Frontier Coefficients](https://arxiv.org/abs/2404.00825) | 提出了一种利用机器学习预测市场方向的新方法，该方法使用有效边界系数进行资产回报估计，并将市场预测集成到投资组合优化框架中。 |
| [^2] | [Algorithmic Collusion by Large Language Models](https://arxiv.org/abs/2404.00806) | 大型语言模型的算法定价代理在寡头市场环境中自主勾结，对消费者利益有害，其说明书中的短语变化可能增加勾结。 |
| [^3] | [From attention to profit: quantitative trading strategy based on transformer](https://arxiv.org/abs/2404.00424) | 该研究介绍了一种基于Transformer的量化交易策略，利用改进的模型架构和情感分析的迁移学习，不仅在捕捉长期依赖关系和建模数据关系方面具有优势，而且能够准确预测未来一段时间内的回报。 |
| [^4] | [Shared Hardships Strengthen Bonds: Negative Shocks, Embeddedness and Employee Retention](https://arxiv.org/abs/2404.00183) | 明确为负面冲击可以增加员工留任，通过设备相关的冲击导致员工对公司的情感承诺增加，突出了激励一致性在冲击转化为留任中的关键作用。 |
| [^5] | [Temporal Graph Networks for Graph Anomaly Detection in Financial Networks](https://arxiv.org/abs/2404.00060) | 时间图网络（TGN）在金融网络中的异常检测中表现出显著优势，适应了现代金融系统动态和复杂的特性。 |
| [^6] | [Investigating Similarities Across Decentralized Financial (DeFi) Services](https://arxiv.org/abs/2404.00034) | 该研究通过采用图表示学习算法，提出了一种方法来研究去中心化金融协议提供的服务之间的相似性，并成功将这些服务分组为具有相似功能的集群。 |
| [^7] | [Antinetwork among China A-shares](https://arxiv.org/abs/2404.00028) | 本研究首次考虑了负相关性和正相关性，构建了中国A股上市股票之间的加权时间反网络和网络，揭示了在21世纪前24年期间（反）网络之间节点度和强度、同配性系数、平均局部聚类系数以及平均最短路径长度等方面的一些重要差异 |
| [^8] | [Empowering Credit Scoring Systems with Quantum-Enhanced Machine Learning](https://arxiv.org/abs/2404.00015) | 提出了一种名为Systemic Quantum Score (SQS)的新方法，展示在金融领域生产级应用案例中相比纯经典模型更有优势，能够从较少数据点中提取模式并表现出更好性能。 |
| [^9] | [Missing Data Imputation With Granular Semantics and AI-driven Pipeline for Bankruptcy Prediction](https://arxiv.org/abs/2404.00013) | 本文介绍了一种具有精确语义的缺失数据插补方法，通过在粒空间中利用特征语义和可靠观测来预测缺失值，从而解决了破产预测中的重要挑战。 |
| [^10] | [Stress index strategy enhanced with financial news sentiment analysis for the equity markets](https://arxiv.org/abs/2404.00012) | 通过将金融压力指标与财经新闻情感分析相结合，提高了股票市场的风险控制策略表现。 |
| [^11] | [Labor Market Effects of the Venezuelan Refugee Crisis in Brazil](https://arxiv.org/abs/2302.04201) | 巴西委内瑞拉难民危机对罗赖马州劳动力市场的影响表明，虽然巴西人的月工资增加约2％，但主要受无难民参与的部门和职业影响，同时非正规市场中的移民抵消了正规市场的替代效应。 |
| [^12] | [Time-aware Metapath Feature Augmentation for Ponzi Detection in Ethereum](https://arxiv.org/abs/2210.16863) | 该研究介绍了一种用于以太坊Ponzi检测的时序元路径特征增强方法，旨在捕获交易模式信息中的时间依赖性。 |
| [^13] | [A Pomeranzian Growth Theory of the Great Divergence](https://arxiv.org/abs/2108.03110) | 基于欧洲土地限制缓解导致的大分歧假设，这个研究构建了一个增长模型，对农业和制造业部门进行了形式化，并展示了经济从马尔萨斯状态向非马尔萨斯状态转变的过程。 |
| [^14] | [Influencing Competition Through Shelf Design](https://arxiv.org/abs/2010.09227) | 货架设计对产品需求有显著影响，研究发现竞争效应通常更强，但并非总是如此。 |
| [^15] | [Productivity, Inputs Misallocation, and the Financial Crisis.](http://arxiv.org/abs/2306.08760) | 本文对定量衡量同质产业内资源错配的常规方法进行了重新评估，并提出了一种基于已识别的政策变化的比较分析策略。研究表明，金融危机对生产要素的影响导致资源配置严重失调，而政策干预可能会损害整体经济效率。 |

# 详细

[^1]: 利用机器学习预测市场方向与有效边界系数

    Using Machine Learning to Forecast Market Direction with Efficient Frontier Coefficients

    [https://arxiv.org/abs/2404.00825](https://arxiv.org/abs/2404.00825)

    提出了一种利用机器学习预测市场方向的新方法，该方法使用有效边界系数进行资产回报估计，并将市场预测集成到投资组合优化框架中。

    

    我们提出了一种新方法，用于改善投资组合优化中资产回报的估计。该方法首先使用在线决策树执行月度市场方向预测。决策树是在从投资组合理论中提取的一组新特征上进行训练的：有效边界函数系数。有效边界可以被分解为其函数形式，即平方根二次多项式，该函数的系数捕捉了组成当前时间段市场的所有成分的信息。为了使这些预测可操作，这些方向预测被集成到一个投资组合优化框架中，该框架使用基于市场预测的预期回报作为回报向量的估计。这个条件期望是使用倒数密尔斯比率来计算的，资本资产定价模型用于将市场预测转化为个体

    arXiv:2404.00825v1 Announce Type: new  Abstract: We propose a novel method to improve estimation of asset returns for portfolio optimization. This approach first performs a monthly directional market forecast using an online decision tree. The decision tree is trained on a novel set of features engineered from portfolio theory: the efficient frontier functional coefficients. Efficient frontiers can be decomposed to their functional form, a square-root second-order polynomial, and the coefficients of this function captures the information of all the constituents that compose the market in the current time period. To make these forecasts actionable, these directional forecasts are integrated to a portfolio optimization framework using expected returns conditional on the market forecast as an estimate for the return vector. This conditional expectation is calculated using the inverse Mills ratio, and the Capital Asset Pricing Model is used to translate the market forecast to individual as
    
[^2]: 大型语言模型的算法勾结

    Algorithmic Collusion by Large Language Models

    [https://arxiv.org/abs/2404.00806](https://arxiv.org/abs/2404.00806)

    大型语言模型的算法定价代理在寡头市场环境中自主勾结，对消费者利益有害，其说明书中的短语变化可能增加勾结。

    

    arXiv:2404.00806v1 公告类型:交叉摘要:算法定价的兴起引起了对算法勾结的担忧。我们对基于大型语言模型（LLMs）特别是GPT-4的算法定价代理进行实验。我们发现：（1）基于LLM的代理在定价任务上表现出色，（2）基于LLM的定价代理在寡头市场环境中自主勾结，损害消费者利益，（3）LLM说明书中看似无害短语("提示")的变化可能会增加勾结。这些结果也适用于拍卖设置。我们的发现强调了有关算法定价的反垄断监管的必要性，并发现了基于LLM的定价代理所面临的监管挑战。

    arXiv:2404.00806v1 Announce Type: cross  Abstract: The rise of algorithmic pricing raises concerns of algorithmic collusion. We conduct experiments with algorithmic pricing agents based on Large Language Models (LLMs), and specifically GPT-4. We find that (1) LLM-based agents are adept at pricing tasks, (2) LLM-based pricing agents autonomously collude in oligopoly settings to the detriment of consumers, and (3) variation in seemingly innocuous phrases in LLM instructions ("prompts") may increase collusion. These results extend to auction settings. Our findings underscore the need for antitrust regulation regarding algorithmic pricing, and uncover regulatory challenges unique to LLM-based pricing agents.
    
[^3]: 从注意力到利润：基于Transformer的量化交易策略

    From attention to profit: quantitative trading strategy based on transformer

    [https://arxiv.org/abs/2404.00424](https://arxiv.org/abs/2404.00424)

    该研究介绍了一种基于Transformer的量化交易策略，利用改进的模型架构和情感分析的迁移学习，不仅在捕捉长期依赖关系和建模数据关系方面具有优势，而且能够准确预测未来一段时间内的回报。

    

    传统量化交易实践中，应对复杂动态的金融市场一直是个持久挑战。先前的机器学习方法往往难以充分捕捉各种市场变量，经常忽视长期信息并且无法捕捉可能带来利润的基本信号。本文引入了改进的Transformer架构，并设计了一个基于该模型的新型因子。通过从情感分析进行迁移学习，所提出的模型不仅发挥了其原有的长距离依赖捕捉和建模复杂数据关系的优势，而且能够处理具有数值输入的任务，并准确预测未来一段时间内的回报。该研究收集了2010年至2019年中国资本市场4,601只股票的5,000,000多条滚动数据。研究结果证明了该模型在预测股票表现方面的卓越性能。

    arXiv:2404.00424v1 Announce Type: cross  Abstract: In traditional quantitative trading practice, navigating the complicated and dynamic financial market presents a persistent challenge. Former machine learning approaches have struggled to fully capture various market variables, often ignore long-term information and fail to catch up with essential signals that may lead the profit. This paper introduces an enhanced transformer architecture and designs a novel factor based on the model. By transfer learning from sentiment analysis, the proposed model not only exploits its original inherent advantages in capturing long-range dependencies and modelling complex data relationships but is also able to solve tasks with numerical inputs and accurately forecast future returns over a period. This work collects more than 5,000,000 rolling data of 4,601 stocks in the Chinese capital market from 2010 to 2019. The results of this study demonstrated the model's superior performance in predicting stock
    
[^4]: 分享困境增强联系：负面冲击、内嵌性和员工留任

    Shared Hardships Strengthen Bonds: Negative Shocks, Embeddedness and Employee Retention

    [https://arxiv.org/abs/2404.00183](https://arxiv.org/abs/2404.00183)

    明确为负面冲击可以增加员工留任，通过设备相关的冲击导致员工对公司的情感承诺增加，突出了激励一致性在冲击转化为留任中的关键作用。

    

    意外事件——“冲击”——是解释内嵌性和员工留任变化的动力，在劳动力流动的不断演变模型中发挥作用。大量研究努力探讨如何让受重视的员工免受不利冲击。然而，本文提供实证证据表明，当公司和员工与这些冲击相关的激励相一致时，明确为负面冲击可以增加员工留任。通过对来自21家卡车公司的466,236条通信记录和45,873次就业记录的独特数据集进行生存分析，我们展示了设备相关的冲击往往会延长就业期限。 设备冲击还产生矛盾的积极情绪，表明员工对公司的情感承诺增加。我们的结果突出了激励一致性在冲击最终如何转化为留任中的重要调节作用。

    arXiv:2404.00183v1 Announce Type: new  Abstract: Unexpected events -- "shocks" -- are the motive force in explaining changes in embeddedness and retention within the unfolding model of labor turnover. Substantial research effort has examined strategies for insulating valued employees from adverse shocks. However, this paper provides empirical evidence that unambiguously negative shocks can increase employee retention when underlying firm and employee incentives with respect to these shocks are aligned. Using survival analysis on a unique data set of 466,236 communication records and 45,873 employment spells from 21 trucking companies, we show how equipment-related shocks tend to increase the duration of employment. Equipment shocks also generate paradoxically positive sentiments that demonstrate an increase in employees' affective commitment to the firm. Our results highlight the important moderating role aligned incentives have in how shocks ultimately translate into retention. Shared
    
[^5]: 金融网络中的时间图网络用于异常检测

    Temporal Graph Networks for Graph Anomaly Detection in Financial Networks

    [https://arxiv.org/abs/2404.00060](https://arxiv.org/abs/2404.00060)

    时间图网络（TGN）在金融网络中的异常检测中表现出显著优势，适应了现代金融系统动态和复杂的特性。

    

    本文探讨了利用时间图网络（TGN）进行金融异常检测，在金融科技和数字化金融交易时代这是一个迫切需要。我们提出了一个全面的框架，利用TGN捕捉金融网络中边的动态变化，用于欺诈检测。我们比较了TGN与静态图神经网络（GNN）基线以及使用DGraph数据集进行现实金融场景下的前沿超图神经网络基线的性能。结果表明，TGN在AUC指标方面显著优于其他模型。这种优越性能突显了TGN作为检测金融欺诈的有效工具的潜力，展示了其适应现代金融系统动态和复杂性的能力。我们还在TGN框架内尝试了各种图嵌入模块，并比较了它们的有效性。

    arXiv:2404.00060v1 Announce Type: cross  Abstract: This paper explores the utilization of Temporal Graph Networks (TGN) for financial anomaly detection, a pressing need in the era of fintech and digitized financial transactions. We present a comprehensive framework that leverages TGN, capable of capturing dynamic changes in edges within financial networks, for fraud detection. Our study compares TGN's performance against static Graph Neural Network (GNN) baselines, as well as cutting-edge hypergraph neural network baselines using DGraph dataset for a realistic financial context. Our results demonstrate that TGN significantly outperforms other models in terms of AUC metrics. This superior performance underlines TGN's potential as an effective tool for detecting financial fraud, showcasing its ability to adapt to the dynamic and complex nature of modern financial systems. We also experimented with various graph embedding modules within the TGN framework and compared the effectiveness of 
    
[^6]: 探究去中心化金融（DeFi）服务之间的相似性

    Investigating Similarities Across Decentralized Financial (DeFi) Services

    [https://arxiv.org/abs/2404.00034](https://arxiv.org/abs/2404.00034)

    该研究通过采用图表示学习算法，提出了一种方法来研究去中心化金融协议提供的服务之间的相似性，并成功将这些服务分组为具有相似功能的集群。

    

    我们探讨了采用图表示学习（GRL）算法来研究去中心化金融（DeFi）协议提供的服务之间的相似性。我们使用以太坊交易数据来识别DeFi构建模块，这些是协议特定的智能合约集，它们在单个交易中以组合方式使用，并封装了执行特定金融服务（如加密资产交换或借贷）的逻辑。我们提出了一种基于智能合约属性和智能合约调用的图结构将这些模块分类进集群的方法。我们利用GRL从构建模块创建嵌入向量，并利用凝聚模型对其进行聚类。为了评估它们是否有效地分组为具有相似功能的集群，我们将它们与八个金融功能类别关联，并将此信息用作目标l

    arXiv:2404.00034v1 Announce Type: cross  Abstract: We explore the adoption of graph representation learning (GRL) algorithms to investigate similarities across services offered by Decentralized Finance (DeFi) protocols. Following existing literature, we use Ethereum transaction data to identify the DeFi building blocks. These are sets of protocol-specific smart contracts that are utilized in combination within single transactions and encapsulate the logic to conduct specific financial services such as swapping or lending cryptoassets. We propose a method to categorize these blocks into clusters based on their smart contract attributes and the graph structure of their smart contract calls. We employ GRL to create embedding vectors from building blocks and agglomerative models for clustering them. To evaluate whether they are effectively grouped in clusters of similar functionalities, we associate them with eight financial functionality categories and use this information as the target l
    
[^7]: 中国A股之间的反网络

    Antinetwork among China A-shares

    [https://arxiv.org/abs/2404.00028](https://arxiv.org/abs/2404.00028)

    本研究首次考虑了负相关性和正相关性，构建了中国A股上市股票之间的加权时间反网络和网络，揭示了在21世纪前24年期间（反）网络之间节点度和强度、同配性系数、平均局部聚类系数以及平均最短路径长度等方面的一些重要差异

    

    基于涨跌幅时间序列之间的相关关系构建的金融网络得到了广泛研究。然而，这些研究忽视了负相关性的重要性。本文首次考虑了负相关性和正相关性，并相应地构建了上海和深圳证券交易所上市股票之间的加权时间反网络和网络。对21世纪前24年的(反)网络，系统分析了节点的度和强度、同配性系数、平均局部聚类系数和平均最短路径长度。本文揭示了这些拓扑测量指标在反网络和网络之间的一些基本差异。这些差异的发现在理解

    arXiv:2404.00028v1 Announce Type: cross  Abstract: The correlation-based financial networks, constructed with the correlation relationships among the time series of fluctuations of daily logarithmic prices of stocks, are intensively studied. However, these studies ignore the importance of negative correlations. This paper is the first time to consider the negative and positive correlations separately, and accordingly to construct weighted temporal antinetwork and network among stocks listed in the Shanghai and Shenzhen stock exchanges. For (anti)networks during the first 24 years of the 21st century, the node's degree and strength, the assortativity coefficient, the average local clustering coefficient, and the average shortest path length are analyzed systematically. This paper unveils some essential differences in these topological measurements between antinetwork and network. The findings of the differences between antinetwork and network have an important role in understanding the 
    
[^8]: 利用量子增强机器学习赋能信用评分系统

    Empowering Credit Scoring Systems with Quantum-Enhanced Machine Learning

    [https://arxiv.org/abs/2404.00015](https://arxiv.org/abs/2404.00015)

    提出了一种名为Systemic Quantum Score (SQS)的新方法，展示在金融领域生产级应用案例中相比纯经典模型更有优势，能够从较少数据点中提取模式并表现出更好性能。

    

    Quantum Kernels被认为在量子机器学习的早期阶段提供了有用性。然而，在利用庞大数据集时，高度复杂的经典模型很难超越，特别是在理解力方面。尽管如此，一旦数据稀缺且倾斜，经典模型就会遇到困难。量子特征空间被预计在这样具有挑战性的情景中能够找到更好的数据特征和目标类别之间的联系，最重要的是增强了泛化能力。在这项工作中，我们提出了一种名为Systemic Quantum Score (SQS)的新方法，并提供了初步结果，表明在金融行业生产级应用案例中，SQS可能比纯经典模型具有优势。我们的具体研究表明，SQS能够从较少的数据点中提取出模式，并且在数据需求量大的算法（如XGBoost）上表现出更好的性能，带来优势。

    arXiv:2404.00015v1 Announce Type: cross  Abstract: Quantum Kernels are projected to provide early-stage usefulness for quantum machine learning. However, highly sophisticated classical models are hard to surpass without losing interpretability, particularly when vast datasets can be exploited. Nonetheless, classical models struggle once data is scarce and skewed. Quantum feature spaces are projected to find better links between data features and the target class to be predicted even in such challenging scenarios and most importantly, enhanced generalization capabilities. In this work, we propose a novel approach called Systemic Quantum Score (SQS) and provide preliminary results indicating potential advantage over purely classical models in a production grade use case for the Finance sector. SQS shows in our specific study an increased capacity to extract patterns out of fewer data points as well as improved performance over data-hungry algorithms such as XGBoost, providing advantage i
    
[^9]: 具有精确语义和AI驱动流程的缺失数据插补及破产预测

    Missing Data Imputation With Granular Semantics and AI-driven Pipeline for Bankruptcy Prediction

    [https://arxiv.org/abs/2404.00013](https://arxiv.org/abs/2404.00013)

    本文介绍了一种具有精确语义的缺失数据插补方法，通过在粒空间中利用特征语义和可靠观测来预测缺失值，从而解决了破产预测中的重要挑战。

    

    本文着重设计了一个用于预测破产的流程。缺失值、高维数据以及高度类别不平衡的数据库是该任务中的主要挑战。本文介绍了一种具有精确语义的缺失数据插补新方法。探讨了粒计算的优点以定义此方法。利用特征语义和可靠观测在低维空间、粒空间中预测缺失值。围绕每个缺失条目形成粒子，考虑到一些高度相关的特征和最可靠的最近观测以保持数据库对缺失条目的相关性和可靠性。然后在这些上下文粒子中进行跨粒子预测进行插补。也就是说，上下文粒子使得在巨大数据库中进行一小部分相关的插补成为可能。

    arXiv:2404.00013v1 Announce Type: cross  Abstract: This work focuses on designing a pipeline for the prediction of bankruptcy. The presence of missing values, high dimensional data, and highly class-imbalance databases are the major challenges in the said task. A new method for missing data imputation with granular semantics has been introduced here. The merits of granular computing have been explored here to define this method. The missing values have been predicted using the feature semantics and reliable observations in a low-dimensional space, in the granular space. The granules are formed around every missing entry, considering a few of the highly correlated features and most reliable closest observations to preserve the relevance and reliability, the context, of the database against the missing entries. An intergranular prediction is then carried out for the imputation within those contextual granules. That is, the contextual granules enable a small relevant fraction of the huge 
    
[^10]: 基于财经新闻情感分析的股票市场压力指数策略改进

    Stress index strategy enhanced with financial news sentiment analysis for the equity markets

    [https://arxiv.org/abs/2404.00012](https://arxiv.org/abs/2404.00012)

    通过将金融压力指标与财经新闻情感分析相结合，提高了股票市场的风险控制策略表现。

    

    本文介绍了一种新的股市风险-风险策略，将金融压力指标与通过ChatGPT读取和解释Bloomberg每日市场摘要进行的情感分析相结合。通过将从波动率和信贷利差推导出的市场压力预测与GPT-4推导出的财经新闻情感相结合，改进了策略的表现，表现为更高的夏普比率和降低的最大回撤。改进的表现在纳斯达克、标普500指数和六个主要股票市场中都保持一致，表明该方法在股票市场中具有普遍适用性。

    arXiv:2404.00012v1 Announce Type: cross  Abstract: This paper introduces a new risk-on risk-off strategy for the stock market, which combines a financial stress indicator with a sentiment analysis done by ChatGPT reading and interpreting Bloomberg daily market summaries. Forecasts of market stress derived from volatility and credit spreads are enhanced when combined with the financial news sentiment derived from GPT-4. As a result, the strategy shows improved performance, evidenced by higher Sharpe ratio and reduced maximum drawdowns. The improved performance is consistent across the NASDAQ, the S&P 500 and the six major equity markets, indicating that the method generalises across equities markets.
    
[^11]: 巴西委内瑞拉难民危机对劳动力市场的影响

    Labor Market Effects of the Venezuelan Refugee Crisis in Brazil

    [https://arxiv.org/abs/2302.04201](https://arxiv.org/abs/2302.04201)

    巴西委内瑞拉难民危机对罗赖马州劳动力市场的影响表明，虽然巴西人的月工资增加约2％，但主要受无难民参与的部门和职业影响，同时非正规市场中的移民抵消了正规市场的替代效应。

    

    我们利用巴西正式工人普遍使用的行政面板数据，研究了委内瑞拉危机对巴西劳动力市场的影响，重点关注边境州罗赖马。差异处理结果表明，罗赖马州的巴西人月工资增长了约2％，主要是由那些从事没有难民参与的部门和职业的人驱动的。研究发现，巴西人几乎没有失业，但发现本地工人转移到没有移民的职业。我们还发现，非正规市场中的移民抵消了正规市场中的替代效应。

    arXiv:2302.04201v2 Announce Type: replace  Abstract: We use administrative panel data on the universe of Brazilian formal workers to investigate the labor market effects of the Venezuelan crisis in Brazil, focusing on the border state of Roraima. The results using difference-in-differences show that the monthly wages of Brazilians in Roraima increased by around 2 percent, which was mostly driven by those working in sectors and occupations with no refugee involvement. The study finds negligible job displacement for Brazilians but finds evidence of native workers moving to occupations without immigrants. We also find that immigrants in the informal market offset the substitution effects in the formal market.
    
[^12]: 以太坊Ponzi检测的时序元路径特征增强

    Time-aware Metapath Feature Augmentation for Ponzi Detection in Ethereum

    [https://arxiv.org/abs/2210.16863](https://arxiv.org/abs/2210.16863)

    该研究介绍了一种用于以太坊Ponzi检测的时序元路径特征增强方法，旨在捕获交易模式信息中的时间依赖性。

    

    随着强调去中心化的Web 3.0的发展，区块链技术迎来了自己的革命，并带来了许多挑战，特别是在加密货币领域。最近，大量的犯罪行为不断在区块链上出现，如庞氏骗局和钓鱼欺诈，这严重危害了去中心化金融。现有基于图的区块链异常行为检测方法通常侧重于构建同质交易图，而没有区分节点和边的异质性，导致交易模式信息的部分丢失。尽管现有的异质建模方法可以通过元路径描述更丰富的信息，但提取的元路径通常忽视实体之间的时间依赖性，并且不反映真实行为。本文引入了时间感知元路径特征增强（TMFAug）作为一种即插即用模块，用于捕获

    arXiv:2210.16863v2 Announce Type: replace  Abstract: With the development of Web 3.0 which emphasizes decentralization, blockchain technology ushers in its revolution and also brings numerous challenges, particularly in the field of cryptocurrency. Recently, a large number of criminal behaviors continuously emerge on blockchain, such as Ponzi schemes and phishing scams, which severely endanger decentralized finance. Existing graph-based abnormal behavior detection methods on blockchain usually focus on constructing homogeneous transaction graphs without distinguishing the heterogeneity of nodes and edges, resulting in partial loss of transaction pattern information. Although existing heterogeneous modeling methods can depict richer information through metapaths, the extracted metapaths generally neglect temporal dependencies between entities and do not reflect real behavior. In this paper, we introduce Time-aware Metapath Feature Augmentation (TMFAug) as a plug-and-play module to captu
    
[^13]: 一个关于大分歧的波莫兰增长理论

    A Pomeranzian Growth Theory of the Great Divergence

    [https://arxiv.org/abs/2108.03110](https://arxiv.org/abs/2108.03110)

    基于欧洲土地限制缓解导致的大分歧假设，这个研究构建了一个增长模型，对农业和制造业部门进行了形式化，并展示了经济从马尔萨斯状态向非马尔萨斯状态转变的过程。

    

    这项研究构建了一个关于大分歧的增长模型，形式化了波莫兰兹（2000）的假设，即欧洲土地限制的缓解导致了自19世纪以来欧洲与中国经济增长之间的分歧。该模型由农业和制造业组成。农业部门从土地上生产生活必需品、来自制造业的中间品和劳动力。制造业部门通过全职制造业工人的实践成长来生产商品。家庭做出生育决策。在模型中，对土地供应的大规模外生性正面冲击使经济从马尔萨斯状态转变为非马尔萨斯状态，其中所有工人都从事农业生产，人均收入恒定，到参与农业生产的工人比例逐渐减少。

    arXiv:2108.03110v3 Announce Type: replace  Abstract: This study constructs a growth model of the Great Divergence that formalizes Pomeranz's (2000) hypothesis that the relief of land constraints in Europe has caused divergence in economic growth between Europe and China since the 19th century. The model consists of the agricultural and manufacturing sectors. The agricultural sector produces subsistence goods from land, intermediate goods from the manufacturing sector, and labor. The manufacturing sector produces goods from labor, and its productivity grows through the learning-by-doing of full-time manufacturing workers. Households make fertility decisions. In the model, a large exogenous positive shock in land supply causes the transition of the economy from the Malthusian state, in which all workers are engaged in agricultural production and per capita income is constant, to the non-Malthusian state, in which the share of workers engaged in agricultural production gradually decreases
    
[^14]: 通过货架设计影响竞争

    Influencing Competition Through Shelf Design

    [https://arxiv.org/abs/2010.09227](https://arxiv.org/abs/2010.09227)

    货架设计对产品需求有显著影响，研究发现竞争效应通常更强，但并非总是如此。

    

    货架设计决策强烈影响产品需求。特别是将产品放置在理想位置会增加需求。货架位置对需求有主要影响，但也存在基于附近产品相对位置的次要影响。直觉上，相邻位置的产品更有可能被比较，产生积极和消极效应。通过修改GEV类模型，以允许需求受竞争对手的接近程度调节，这两种效应会自然产生。

    arXiv:2010.09227v3 Announce Type: replace  Abstract: Shelf design decisions strongly influence product demand. In particular, placing products in desirable locations increases demand. This primary effect on shelf position is clear, but there is a secondary effect based on the relative positioning of nearby products. Intuitively, products located next to each other are more likely to be compared having positive and negative effects. On the one hand, locations closer to relatively strong products will be undesirable, as these strong products will draw demand from others -- an effect that is stronger for those in close proximity. On the other hand, because strong products tend to attract more traffic, locations closer to them elicit high consumer attention by increased visibility. Modifying the GEV class of models to allow demand to be moderated by competitors' proximity, these two effects emerge naturally. We found that although the competition effect is usually stronger, it is not alway
    
[^15]: 生产率、要素错配与金融危机

    Productivity, Inputs Misallocation, and the Financial Crisis. (arXiv:2306.08760v1 [econ.GN])

    [http://arxiv.org/abs/2306.08760](http://arxiv.org/abs/2306.08760)

    本文对定量衡量同质产业内资源错配的常规方法进行了重新评估，并提出了一种基于已识别的政策变化的比较分析策略。研究表明，金融危机对生产要素的影响导致资源配置严重失调，而政策干预可能会损害整体经济效率。

    

    本文重新评估了定量衡量同质产业内资源错配的常规方法，该方法通常使用一个输入边际产品的离散度来衡量。我的研究发现，这一统计量包括固有的生产率异质性和特有的生产率冲击，而不考虑所研究的输入。利用美国和欧洲制造业公司的资产负债表数据，我发现总生产要素生产率波动值占资本边际产品的方差的7％，劳动力为9％，材料输入为10％。因此，这个指数以表面价值表示时，无法识别出任何生产要素的政策引发的错配。为了克服这一局限性，我提出了一种基于已识别的政策变化的比较分析策略。这种方法允许研究人员在相对的条件下评估诱导失配，同时控制TFP波动性的差异。我发现金融危机对生产要素的影响是不均衡的，导致资源配置严重失调。通过比较不同地区和行业的金融危机前和金融危机期间，经济金融环境稳定的情况，我发现金融危机对制造业的负面影响比服务业更为明显，对欧洲业务更为严重，而且这种失调是由于金融紧缩政策引起的。这一研究表明，政策干预可能会导致资源配置失调，损害整体经济效率。

    This paper reevaluates the conventional approach to quantifying within-industry resource misallocation, typically measured by the dispersion of an input marginal product. My findings suggest that this statistic incorporates inherent productivity heterogeneity and idiosyncratic productivity shocks, irrespective of the input under scrutiny. Using balance sheet data from American and European manufacturing firms, I show that total factor productivity (TFP) volatility accounts for 7% of the variance in the marginal product of capital, 9% for labor, and 10% for material inputs. Consequently, this index, taken at face value, fails to identify policy-induced misallocation for any production input. To overcome this limitation, I propose a comparative analysis strategy driven by an identified policy variation. This approach allows the researcher to assess induced misallocation in relative terms whilst controlling for differences in TFP volatility. I show that the financial crisis had an uneven 
    

