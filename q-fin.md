# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Robust and Efficient Optimization Model for Electric Vehicle Charging Stations in Developing Countries under Electricity Uncertainty.](http://arxiv.org/abs/2307.05470) | 提出了一种鲁棒的、高效的模拟优化模型，用于解决发展中国家电动车充电站的电力不确定性问题。通过估计候选站点的服务可靠性，并运用控制变量技术，我们提供了一个高度鲁棒的解决方案，有效应对电力中断的不确定性。 |
| [^2] | [Harnessing the Potential of Volatility: Advancing GDP Prediction.](http://arxiv.org/abs/2307.05391) | 本文介绍了一种新的机器学习方法，通过将波动性作为模型权重，准确预测国内生产总值（GDP）。该方法考虑了意外冲击和事件对经济的影响，通过测试和比较表明，波动性加权的Lasso方法在准确性和鲁棒性方面优于其他方法，为决策者提供了有价值的决策工具。 |
| [^3] | [Transaction Fraud Detection via Spatial-Temporal-Aware Graph Transformer.](http://arxiv.org/abs/2307.05121) | 我们提出了一种名为STA-GT的新颖异构图神经网络用于交易欺诈检测，它能够有效学习空间-时间信息，并通过合并全局信息改进表示学习。 |
| [^4] | [Portfolio Optimization: A Comparative Study.](http://arxiv.org/abs/2307.05048) | 该论文进行了一项比较研究，研究了三种投资组合设计方法，包括均值方差投资组合（MVP）、层次风险均衡（HRP）基于投资组合和自编码器基于投资组合，并将其应用于印度国家证券交易所（NSE）上的十个主题行业的股票。结果显示，在样本外数据上，MVP投资组合的表现最佳。 |
| [^5] | [Epidemic Modeling with Generative Agents.](http://arxiv.org/abs/2307.04986) | 本研究利用生成型智能体在流行病模型中模拟了人类行为，通过模拟实验展示了智能体的行为与真实世界相似，并成功实现了流行病曲线的平坦化。该研究创造了改进动态系统建模的潜力，为表示人类思维、推理和决策提供了一种途径。 |
| [^6] | [Measuring Cause-Effect with the Variability of the Largest Eigenvalue.](http://arxiv.org/abs/2307.04953) | 本论文提出了一种用最大特征值的变异性来测量因果关系的方法，通过分析滞后相关矩阵的特征值的分布来测试时间变量之间的结构关系，并应用于分析零售经纪商每日货币流量之间的因果依赖关系。 |
| [^7] | [Modeling evidential cooperation in large worlds.](http://arxiv.org/abs/2307.04879) | 该论文研究了大规模世界中的证据合作（ECL）的问题，并提出了一个不完全信息的谈判问题的博弈论模型。通过模型，作者发现所有合作者必须最大化相同的加权效用函数之和才能达到帕累托最优结果。 |
| [^8] | [Tackling the Problem of State Dependent Execution Probability: Empirical Evidence and Order Placement.](http://arxiv.org/abs/2307.04863) | 本研究使用高频数据和生存分析展示了填充概率函数具有强烈的状态依赖性质。通过对比数字资产交易所和股票市场的结果，我们分析了小tick加密货币对和大tick资产之间的填充概率差异。研究还得出了在固定时间周期内执行问题中的最优策略。 |
| [^9] | [A note on the induction of comonotonic additive risk measures from acceptance sets.](http://arxiv.org/abs/2307.04647) | 我们提出了一种简单的方法，可以通过接受集诱导出共单增加性风险度量。这种方法适用于具有先验指定的依赖结构的风险度量。 |
| [^10] | [SoK: Blockchain Decentralization.](http://arxiv.org/abs/2205.04256) | 该论文为区块链去中心化领域的知识系统化，提出了分类法并建议使用指数来衡量和量化区块链的去中心化水平。除了共识去中心化外，其他方面的研究相对较少。这项工作为未来的研究提供了基准和指导。 |

# 详细

[^1]: 发展中国家电动车充电站的鲁棒高效优化模型——基于电力不确定性的研究

    A Robust and Efficient Optimization Model for Electric Vehicle Charging Stations in Developing Countries under Electricity Uncertainty. (arXiv:2307.05470v1 [math.OC])

    [http://arxiv.org/abs/2307.05470](http://arxiv.org/abs/2307.05470)

    提出了一种鲁棒的、高效的模拟优化模型，用于解决发展中国家电动车充电站的电力不确定性问题。通过估计候选站点的服务可靠性，并运用控制变量技术，我们提供了一个高度鲁棒的解决方案，有效应对电力中断的不确定性。

    

    全球电动车（EV）的需求日益增长，这需要发展鲁棒且可访问的充电基础设施，尤其是在电力不稳定的发展中国家。早期的充电基础设施优化研究未能严格考虑此类服务中断特征，导致基础设施设计不佳。为解决这个问题，我们提出了一个高效的基于模拟的优化模型，该模型估计候选站点的服务可靠性，并将其纳入目标函数和约束中。我们采用了控制变量（CV）方差减小技术来提高模拟效率。我们的模型提供了一个高度鲁棒的解决方案，即使在候选站点服务可靠性被低估或高估的情况下，也能缓冲不确定的电力中断。使用印度尼西亚苏拉巴亚的数据集，我们的数值实验证明了所提出模型的有效性。

    The rising demand for electric vehicles (EVs) worldwide necessitates the development of robust and accessible charging infrastructure, particularly in developing countries where electricity disruptions pose a significant challenge. Earlier charging infrastructure optimization studies do not rigorously address such service disruption characteristics, resulting in suboptimal infrastructure designs. To address this issue, we propose an efficient simulation-based optimization model that estimates candidate stations' service reliability and incorporates it into the objective function and constraints. We employ the control variates (CV) variance reduction technique to enhance simulation efficiency. Our model provides a highly robust solution that buffers against uncertain electricity disruptions, even when candidate station service reliability is subject to underestimation or overestimation. Using a dataset from Surabaya, Indonesia, our numerical experiment demonstrates that the proposed mod
    
[^2]: 发挥波动性潜力：推进国内生产总值预测

    Harnessing the Potential of Volatility: Advancing GDP Prediction. (arXiv:2307.05391v1 [econ.GN])

    [http://arxiv.org/abs/2307.05391](http://arxiv.org/abs/2307.05391)

    本文介绍了一种新的机器学习方法，通过将波动性作为模型权重，准确预测国内生产总值（GDP）。该方法考虑了意外冲击和事件对经济的影响，通过测试和比较表明，波动性加权的Lasso方法在准确性和鲁棒性方面优于其他方法，为决策者提供了有价值的决策工具。

    

    本文提出了一种新颖的机器学习方法，将波动性作为模型权重引入国内生产总值（GDP）预测中。该方法专门设计用于准确预测GDP，同时考虑到可能影响经济的意外冲击或事件。通过对实际数据进行测试，并与以往用于GDP预测的技术（如Lasso和自适应Lasso）进行比较，表明波动性加权Lasso方法在准确性和鲁棒性方面优于其他方法，为决策者和分析师在快速变化的经济环境中提供了有价值的决策工具。本研究展示了数据驱动方法如何帮助我们更好地理解经济波动，并支持更有效的经济政策制定。

    This paper presents a novel machine learning approach to GDP prediction that incorporates volatility as a model weight. The proposed method is specifically designed to identify and select the most relevant macroeconomic variables for accurate GDP prediction, while taking into account unexpected shocks or events that may impact the economy. The proposed method's effectiveness is tested on real-world data and compared to previous techniques used for GDP forecasting, such as Lasso and Adaptive Lasso. The findings show that the Volatility-weighted Lasso method outperforms other methods in terms of accuracy and robustness, providing policymakers and analysts with a valuable tool for making informed decisions in a rapidly changing economic environment. This study demonstrates how data-driven approaches can help us better understand economic fluctuations and support more effective economic policymaking.  Keywords: GDP prediction, Lasso, Volatility, Regularization, Macroeconomics Variable Sele
    
[^3]: 通过空间-时间感知图转换器进行交易欺诈检测

    Transaction Fraud Detection via Spatial-Temporal-Aware Graph Transformer. (arXiv:2307.05121v1 [cs.LG])

    [http://arxiv.org/abs/2307.05121](http://arxiv.org/abs/2307.05121)

    我们提出了一种名为STA-GT的新颖异构图神经网络用于交易欺诈检测，它能够有效学习空间-时间信息，并通过合并全局信息改进表示学习。

    

    如何获取交易的信息表示并进行欺诈交易的识别是确保金融安全的关键部分。最近的研究将图神经网络应用于交易欺诈检测问题。然而，由于结构限制，它们在有效学习空间-时间信息方面遇到了挑战。此外，很少有基于图神经网络的先前检测器意识到合并全局信息的重要性，该全局信息涵盖了相似的行为模式并为判别性表示学习提供了有价值的见解。因此，我们提出了一种新颖的异构图神经网络，称为空间-时间感知图转换器（STA-GT），用于交易欺诈检测问题。具体来说，我们设计了一种时间编码策略来捕捉时间依赖关系，并将其纳入图神经网络框架中，增强了空间-时间信息建模并改进了表达能力。

    How to obtain informative representations of transactions and then perform the identification of fraudulent transactions is a crucial part of ensuring financial security. Recent studies apply Graph Neural Networks (GNNs) to the transaction fraud detection problem. Nevertheless, they encounter challenges in effectively learning spatial-temporal information due to structural limitations. Moreover, few prior GNN-based detectors have recognized the significance of incorporating global information, which encompasses similar behavioral patterns and offers valuable insights for discriminative representation learning. Therefore, we propose a novel heterogeneous graph neural network called Spatial-Temporal-Aware Graph Transformer (STA-GT) for transaction fraud detection problems. Specifically, we design a temporal encoding strategy to capture temporal dependencies and incorporate it into the graph neural network framework, enhancing spatial-temporal information modeling and improving expressive
    
[^4]: 投资组合优化：一项比较研究

    Portfolio Optimization: A Comparative Study. (arXiv:2307.05048v1 [q-fin.PM])

    [http://arxiv.org/abs/2307.05048](http://arxiv.org/abs/2307.05048)

    该论文进行了一项比较研究，研究了三种投资组合设计方法，包括均值方差投资组合（MVP）、层次风险均衡（HRP）基于投资组合和自编码器基于投资组合，并将其应用于印度国家证券交易所（NSE）上的十个主题行业的股票。结果显示，在样本外数据上，MVP投资组合的表现最佳。

    

    投资组合优化一直是金融研究界关注的领域。设计一个盈利的投资组合是一个具有挑战性的任务，涉及到对未来股票收益和风险的精确预测。本章介绍了三种投资组合设计方法的比较研究，包括均值方差投资组合（MVP）、层次风险均衡（HRP）基于投资组合和自编码器基于投资组合。这三种投资组合设计方法应用于从印度国家证券交易所（NSE）上选择的十个主题行业的股票的历史价格。投资组合是使用2018年1月1日到2021年12月31日的股票价格数据进行设计的，并且它们的表现在2022年1月1日到2022年12月31日的样本外数据上进行了测试。对投资组合的性能进行了详细结果分析。观察到MVP投资组合在样本外数据上的表现最好，风险调整。

    Portfolio optimization has been an area that has attracted considerable attention from the financial research community. Designing a profitable portfolio is a challenging task involving precise forecasting of future stock returns and risks. This chapter presents a comparative study of three portfolio design approaches, the mean-variance portfolio (MVP), hierarchical risk parity (HRP)-based portfolio, and autoencoder-based portfolio. These three approaches to portfolio design are applied to the historical prices of stocks chosen from ten thematic sectors listed on the National Stock Exchange (NSE) of India. The portfolios are designed using the stock price data from January 1, 2018, to December 31, 2021, and their performances are tested on the out-of-sample data from January 1, 2022, to December 31, 2022. Extensive results are analyzed on the performance of the portfolios. It is observed that the performance of the MVP portfolio is the best on the out-of-sample data for the risk-adjust
    
[^5]: 用生成型智能体进行流行病建模

    Epidemic Modeling with Generative Agents. (arXiv:2307.04986v1 [cs.AI])

    [http://arxiv.org/abs/2307.04986](http://arxiv.org/abs/2307.04986)

    本研究利用生成型智能体在流行病模型中模拟了人类行为，通过模拟实验展示了智能体的行为与真实世界相似，并成功实现了流行病曲线的平坦化。该研究创造了改进动态系统建模的潜力，为表示人类思维、推理和决策提供了一种途径。

    

    本研究提供了一种新的个体层面建模范式，以解决将人类行为纳入流行病模型的重大挑战。通过在基于智能体的流行病模型中利用生成型人工智能，每个智能体都能够通过连接到大型语言模型（如ChatGPT）进行自主推理和决策。通过各种模拟实验，我们呈现了令人信服的证据，表明生成型智能体模仿了现实世界的行为，如生病时进行隔离，病例增加时进行自我隔离。总体而言，智能体展示了类似于近期流行病观察到的多次波动，然后是一段流行期。此外，智能体成功地使流行病曲线平坦化。该研究提供了一种改进动态系统建模的潜力，通过提供一种表示人类大脑、推理和决策的方法。

    This study offers a new paradigm of individual-level modeling to address the grand challenge of incorporating human behavior in epidemic models. Using generative artificial intelligence in an agent-based epidemic model, each agent is empowered to make its own reasonings and decisions via connecting to a large language model such as ChatGPT. Through various simulation experiments, we present compelling evidence that generative agents mimic real-world behaviors such as quarantining when sick and self-isolation when cases rise. Collectively, the agents demonstrate patterns akin to multiple waves observed in recent pandemics followed by an endemic period. Moreover, the agents successfully flatten the epidemic curve. This study creates potential to improve dynamic system modeling by offering a way to represent human brain, reasoning, and decision making.
    
[^6]: 用最大特征值的变异性测量因果关系

    Measuring Cause-Effect with the Variability of the Largest Eigenvalue. (arXiv:2307.04953v1 [q-fin.PM])

    [http://arxiv.org/abs/2307.04953](http://arxiv.org/abs/2307.04953)

    本论文提出了一种用最大特征值的变异性来测量因果关系的方法，通过分析滞后相关矩阵的特征值的分布来测试时间变量之间的结构关系，并应用于分析零售经纪商每日货币流量之间的因果依赖关系。

    

    我们提出了一种测试和监测时间变量之间结构关系的方法。使用滞后相关矩阵的第一个特征值的分布（Tracy-Widom分布），来测试变量之间的结构时间关系与备选假设（独立性）相对。该分布研究了最大特征值随滞后的渐近动态。通过分析具有不同滞后的$2\times 2$相关矩阵的最大特征值的标准差的时间序列，我们可以分析与Tracy-Widom分布的偏离，以测试这两个时间变量之间的结构关系，这些关系可以与因果关系相关联。我们使用不同滞后的第一个特征值的标准差作为测试和监测结构因果关系的代理。该方法被应用于分析零售经纪商每日货币流量之间的因果依赖关系。

    We present a method to test and monitor structural relationships between time variables. The distribution of the first eigenvalue for lagged correlation matrices (Tracy-Widom distribution) is used to test structural time relationships between variables against the alternative hypothesis (Independence). This distribution studies the asymptotic dynamics of the largest eigenvalue as a function of the lag in lagged correlation matrices. By analyzing the time series of the standard deviation of the greatest eigenvalue for $2\times 2$ correlation matrices with different lags we can analyze deviations from the Tracy-Widom distribution to test structural relationships between these two time variables. These relationships can be related to causality. We use the standard deviation of the first eigenvalue at different lags as a proxy for testing and monitoring structural causal relationships. The method is applied to analyse causal dependencies between daily monetary flows in a retail brokerage b
    
[^7]: 在大规模世界中建模证据合作

    Modeling evidential cooperation in large worlds. (arXiv:2307.04879v1 [econ.GN])

    [http://arxiv.org/abs/2307.04879](http://arxiv.org/abs/2307.04879)

    该论文研究了大规模世界中的证据合作（ECL）的问题，并提出了一个不完全信息的谈判问题的博弈论模型。通过模型，作者发现所有合作者必须最大化相同的加权效用函数之和才能达到帕累托最优结果。

    

    大规模世界中的证据合作（ECL）指的是人类和其他代理人可以通过与具有不同价值观的相似代理人在一个大宇宙中的因果断开部分合作而获益。合作为代理人提供了其他相似代理人可能会合作的证据，从而使所有人都从交易中获益。对于利他主义者来说，这可能是一个关键的考虑因素。我将ECL发展为一个不完全信息的谈判问题的博弈论模型。该模型融入了对他人价值观和实证情况的不确定性，并解决了选择妥协结果的问题。使用该模型，我调查了ECL存在的问题，并概述了开放的技术和哲学问题。我展示了所有合作者必须最大化相同的加权效用函数之和才能达到帕累托最优结果。然而，我反对通过对效用函数进行归一化来隐式选择妥协结果。我回顾了谈判理论和

    Evidential cooperation in large worlds (ECL) refers to the idea that humans and other agents can benefit by cooperating with similar agents with differing values in causally disconnected parts of a large universe. Cooperating provides agents with evidence that other similar agents are likely to cooperate too, resulting in gains from trade for all. This could be a crucial consideration for altruists.  I develop a game-theoretic model of ECL as an incomplete information bargaining problem. The model incorporates uncertainty about others' value systems and empirical situations, and addresses the problem of selecting a compromise outcome. Using the model, I investigate issues with ECL and outline open technical and philosophical questions.  I show that all cooperators must maximize the same weighted sum of utility functions to reach a Pareto optimal outcome. However, I argue against selecting a compromise outcome implicitly by normalizing utility functions. I review bargaining theory and a
    
[^8]: 解决状态依赖执行概率问题：经验证据和订单放置策略

    Tackling the Problem of State Dependent Execution Probability: Empirical Evidence and Order Placement. (arXiv:2307.04863v1 [q-fin.TR])

    [http://arxiv.org/abs/2307.04863](http://arxiv.org/abs/2307.04863)

    本研究使用高频数据和生存分析展示了填充概率函数具有强烈的状态依赖性质。通过对比数字资产交易所和股票市场的结果，我们分析了小tick加密货币对和大tick资产之间的填充概率差异。研究还得出了在固定时间周期内执行问题中的最优策略。

    

    订单放置策略在高频交易算法中起着至关重要的作用，其设计基于对订单簿动态的理解。利用高质量的高频数据和生存分析，我们展示了充分概率函数具有强烈的状态依赖性质。我们定义了一组微观结构特征，并训练了一个多层感知机来推断填充概率函数。我们应用了一种加权方法到损失函数中，使得模型能够从被审查数据中学习。通过比较在数字资产中心化交易所（CEXs）和股票市场上获得的数值结果，我们能够分析小tick加密货币对和大tick资产（与加密货币相对较大）的填充概率之间的差异。我们用一个固定时间周期的执行问题来说明这个模型的实际用途，其中包括是否发布限价订单或立即执行的决策，以及最优放置距离的特征。

    Order placement tactics play a crucial role in high-frequency trading algorithms and their design is based on understanding the dynamics of the order book. Using high quality high-frequency data and survival analysis, we exhibit strong state dependence properties of the fill probability function. We define a set of microstructure features and train a multi-layer perceptron to infer the fill probability function. A weighting method is applied to the loss function such that the model learns from censored data. By comparing numerical results obtained on both digital asset centralized exchanges (CEXs) and stock markets, we are able to analyze dissimilarities between the fill probability of small tick crypto pairs and large tick assets -- large, relative to cryptos. The practical use of this model is illustrated with a fixed time horizon execution problem in which both the decision to post a limit order or to immediately execute and the optimal distance of placement are characterized. We di
    
[^9]: 关于由接受集导出共单增加性风险度量的一点说明

    A note on the induction of comonotonic additive risk measures from acceptance sets. (arXiv:2307.04647v1 [q-fin.MF])

    [http://arxiv.org/abs/2307.04647](http://arxiv.org/abs/2307.04647)

    我们提出了一种简单的方法，可以通过接受集诱导出共单增加性风险度量。这种方法适用于具有先验指定的依赖结构的风险度量。

    

    我们在接受集上提出了简单的一般条件，使其诱导出的货币风险和偏差度量是共单增加性的。我们证明了，如果接受集及其补集在共单增随机变量的凸组合下是稳定的，则接受集诱导出共单增加性风险度量。该结果的推广适用于对于先验指定的依赖结构（例如完全相关的、不相关的或独立的随机变量），与随机变量相加性风险度量。

    We present simple general conditions on the acceptance sets under which their induced monetary risk and deviation measures are comonotonic additive. We show that acceptance sets induce comonotonic additive risk measures if and only if the acceptance sets and their complements are stable under convex combinations of comonotonic random variables. A generalization of this result applies to risk measures that are additive for random variables with \textit{a priori} specified dependence structures, e.g., perfectly correlated, uncorrelated, or independent random variables.
    
[^10]: SoK：区块链去中心化

    SoK: Blockchain Decentralization. (arXiv:2205.04256v4 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2205.04256](http://arxiv.org/abs/2205.04256)

    该论文为区块链去中心化领域的知识系统化，提出了分类法并建议使用指数来衡量和量化区块链的去中心化水平。除了共识去中心化外，其他方面的研究相对较少。这项工作为未来的研究提供了基准和指导。

    

    区块链通过在点对点网络中实现分布式信任，为去中心化经济提供了支持。然而，令人惊讶的是，目前还缺乏广泛接受的去中心化定义或度量标准。我们通过全面分析现有研究，探索了区块链去中心化的知识系统化（SoK）。首先，我们通过对现有研究的定性分析，在共识、网络、治理、财富和交易等五个方面建立了用于分析区块链去中心化的分类法。我们发现，除了共识去中心化以外，其他方面的研究相对较少。其次，我们提出了一种指数，通过转换香农熵来衡量和量化区块链在不同方面的去中心化水平。我们通过比较静态模拟验证了该指数的可解释性。我们还提供了其他指数的定义和讨论，包括基尼系数、中本聪系数和赫尔曼-赫尔东指数等。我们的工作概述了当前区块链去中心化的景象，并提出了一个量化的度量标准，为未来的研究提供基准。

    Blockchain empowers a decentralized economy by enabling distributed trust in a peer-to-peer network. However, surprisingly, a widely accepted definition or measurement of decentralization is still lacking. We explore a systematization of knowledge (SoK) on blockchain decentralization by comprehensively analyzing existing studies in various aspects. First, we establish a taxonomy for analyzing blockchain decentralization in the five facets of consensus, network, governance, wealth, and transaction bu qualitative analysis of existing research. We find relatively little research on aspects other than consensus decentralization. Second, we propose an index that measures and quantifies the decentralization level of blockchain across different facets by transforming Shannon entropy. We verify the explainability of the index via comparative static simulations. We also provide the definition and discussion of alternative indices including the Gini Coefficient, Nakamoto Coefficient, and Herfind
    

