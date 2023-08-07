# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Statistically consistent term structures have affine geometry.](http://arxiv.org/abs/2308.02246) | 本文研究了能源期货的全期限结构的有限维模型，并提出了一个相容性概念，要求所选择的可能收益率曲线集与所获得的扩散系数是相容的，从而推导出一组可能的收益率曲线具有仿射几何性质。 |
| [^2] | [Should we trust web-scraped data?.](http://arxiv.org/abs/2308.02231) | 本论文指出天真的网络抓取程序可能导致收集数据中的抽样偏差，并描述了来源于网络内容易变性、个性化和未索引的抽样偏差。通过例子说明了抽样偏差的普遍性和程度，并提供了克服抽样偏差的建议。 |
| [^3] | [A Non-Parametric Test of Risk Aversion.](http://arxiv.org/abs/2308.02083) | 本研究提出了一种检验期望效用和凹凸性的简单方法，结果发现几乎没有支持这两个模型。此外，研究还证明使用流行的多价格清单方法测量风险规避是不合适的，由于参数错误导致了风险规避高普遍性的现象。 |
| [^4] | [Portfolio Optimization in a Market with Hidden Gaussian Drift and Randomly Arriving Expert Opinions: Modeling and Theoretical Results.](http://arxiv.org/abs/2308.02049) | 本研究分析了金融市场中投资组合优化的问题，考虑了股票回报的隐藏高斯漂移以及随机到达的专家意见，应用卡尔曼滤波技术获得了隐藏漂移的估计，并通过动态规划方法解决了功用最大化问题。 |
| [^5] | [LOB-Based Deep Learning Models for Stock Price Trend Prediction: A Benchmark Study.](http://arxiv.org/abs/2308.01915) | 本研究通过基准研究探讨了基于LOB数据的股票价格趋势预测的15种最新DL模型的鲁棒性和泛化能力。实验证明这些模型在面对新数据时性能明显下降，对其在实际市场中的应用性提出了问题。 |
| [^6] | [Deep Policy Gradient Methods in Commodity Markets.](http://arxiv.org/abs/2308.01910) | 本论文研究了深度强化学习方法在商品交易中的应用，并通过提供流动性和减少市场波动性来稳定市场。这对于解决能源市场的不稳定和全球能源危机具有重要意义。 |
| [^7] | [When do Default Nudges Work?.](http://arxiv.org/abs/2301.08797) | 通过在瑞典Covid-19疫苗推出中的区域变化，研究了对激励有差异的群体的提示效果，结果显示提示对于个人无意义的选择更有效。 |
| [^8] | [Generative CVaR Portfolio Optimization with Attention-Powered Dynamic Factor Learning.](http://arxiv.org/abs/2301.07318) | 本论文提出了一种使用注意力驱动的动态因子学习的生成CVaR组合优化算法，通过随机变量转换作为分布建模的隐式方式，以及使用注意力-GRU网络进行动态学习和预测，捕捉了多元股票收益之间的动态依赖关系，特别是关注尾部属性。通过在每个投资日期从学习到的生成模型中模拟新样本，并进一步应用CVaR组合优化策略，实现了更明智的投资决策。 |
| [^9] | [Multiarmed Bandits Problem Under the Mean-Variance Setting.](http://arxiv.org/abs/2212.09192) | 本文将经典的多臂赌博机问题扩展到均值-方差设置，并通过考虑亚高斯臂放松了先前假设，解决了风险-回报权衡的问题。 |
| [^10] | [SoK: Blockchain Decentralization.](http://arxiv.org/abs/2205.04256) | 该论文为区块链去中心化领域的知识系统化，提出了分类法并建议使用指数来衡量和量化区块链的去中心化水平。除了共识去中心化外，其他方面的研究相对较少。这项工作为未来的研究提供了基准和指导。 |

# 详细

[^1]: 统计一致的期限结构具有仿射几何

    Statistically consistent term structures have affine geometry. (arXiv:2308.02246v1 [q-fin.MF])

    [http://arxiv.org/abs/2308.02246](http://arxiv.org/abs/2308.02246)

    本文研究了能源期货的全期限结构的有限维模型，并提出了一个相容性概念，要求所选择的可能收益率曲线集与所获得的扩散系数是相容的，从而推导出一组可能的收益率曲线具有仿射几何性质。

    

    本文涉及能源期货全期限结构的有限维模型。在选择了一组可能的收益率曲线之后，我们希望从数据中估计收益率曲线演化的动态行为。估计出的模型应该没有套利，这被认为会导致某些漂移条件。如果收益率曲线的演化模型是扩散过程，那么扩散系数的估计就是开放的。从实际角度来看，这要求所选择的可能收益率曲线集与所获得的扩散系数是相容的。在本文中，我们展示了这种相容性强制了一组可能的收益率曲线的仿射几何性质。

    This paper is concerned with finite dimensional models for the entire term structure for energy futures. As soon as a finite dimensional set of possible yield curves is chosen, one likes to estimate the dynamic behaviour of the yield curve evolution from data. The estimated model should be free of arbitrage which is known to result in some drift condition. If the yield curve evolution is modelled by a diffusion, then this leaves the diffusion coefficient open for estimation. From a practical perspective, this requires that the chosen set of possible yield curves is compatible with any obtained diffusion coefficient. In this paper, we show that this compatibility enforces an affine geometry of the set of possible yield curves.
    
[^2]: 我们应该相信网络抓取的数据吗？

    Should we trust web-scraped data?. (arXiv:2308.02231v1 [econ.GN])

    [http://arxiv.org/abs/2308.02231](http://arxiv.org/abs/2308.02231)

    本论文指出天真的网络抓取程序可能导致收集数据中的抽样偏差，并描述了来源于网络内容易变性、个性化和未索引的抽样偏差。通过例子说明了抽样偏差的普遍性和程度，并提供了克服抽样偏差的建议。

    

    实证研究人员越来越多地采用计量经济学和机器学习方法，导致了对一种数据收集方法的广泛使用：网络抓取。网络抓取指的是使用自动化计算机程序访问网站并下载其内容。本文的主要论点是，天真的网络抓取程序可能会导致收集数据中的抽样偏差。本文描述了网络抓取数据中的三种抽样偏差来源。更具体地说，抽样偏差源于网络内容的易变性（即可能发生变化）、个性化（即根据请求特征呈现）和未索引（即人口登记簿的丰富性）。通过一系列例子，我说明了抽样偏差的普遍性和程度。为了支持研究人员和审稿人，本文提供了关于对网络抓取数据的抽样偏差进行预期、检测和克服的建议。

    The increasing adoption of econometric and machine-learning approaches by empirical researchers has led to a widespread use of one data collection method: web scraping. Web scraping refers to the use of automated computer programs to access websites and download their content. The key argument of this paper is that na\"ive web scraping procedures can lead to sampling bias in the collected data. This article describes three sources of sampling bias in web-scraped data. More specifically, sampling bias emerges from web content being volatile (i.e., being subject to change), personalized (i.e., presented in response to request characteristics), and unindexed (i.e., abundance of a population register). In a series of examples, I illustrate the prevalence and magnitude of sampling bias. To support researchers and reviewers, this paper provides recommendations on anticipating, detecting, and overcoming sampling bias in web-scraped data.
    
[^3]: 风险规避的非参数检验

    A Non-Parametric Test of Risk Aversion. (arXiv:2308.02083v1 [econ.GN])

    [http://arxiv.org/abs/2308.02083](http://arxiv.org/abs/2308.02083)

    本研究提出了一种检验期望效用和凹凸性的简单方法，结果发现几乎没有支持这两个模型。此外，研究还证明使用流行的多价格清单方法测量风险规避是不合适的，由于参数错误导致了风险规避高普遍性的现象。

    

    在经济学中，通过期望效用范式内的凹凸伯努利效用来建模风险规避。我们提出了一种简单的期望效用和凹凸性的测试方法。我们发现很少支持任何一种模型：只有30%的选择符合凹凸效用，只有72名受试者中的两人符合期望效用，而其中只有一人符合经济学中的风险规避模型。我们的结果与使用流行的多价格清单方法获取的看似“风险规避”的选择的普遍现象形成对比，这一结果我们在本文中重复了。我们证明了这种方法不适用于衡量风险规避，并且它产生风险规避高普遍性的原因是参数错误。

    In economics, risk aversion is modeled via a concave Bernoulli utility within the expected-utility paradigm. We propose a simple test of expected utility and concavity. We find little support for either: only 30 percent of the choices are consistent with a concave utility, only two out of 72 subjects are consistent with expected utility, and only one of them fits the economic model of risk aversion. Our findings contrast with the preponderance of seemingly "risk-averse" choices that have been elicited using the popular multiple-price list methodology, a result we replicate in this paper. We demonstrate that this methodology is unfit to measure risk aversion, and that the high prevalence of risk aversion it produces is due to parametric misspecification.
    
[^4]: 一个具有隐藏高斯漂移和随机到达专家意见的市场中的组合优化：建模和理论结果

    Portfolio Optimization in a Market with Hidden Gaussian Drift and Randomly Arriving Expert Opinions: Modeling and Theoretical Results. (arXiv:2308.02049v1 [q-fin.PM])

    [http://arxiv.org/abs/2308.02049](http://arxiv.org/abs/2308.02049)

    本研究分析了金融市场中投资组合优化的问题，考虑了股票回报的隐藏高斯漂移以及随机到达的专家意见，应用卡尔曼滤波技术获得了隐藏漂移的估计，并通过动态规划方法解决了功用最大化问题。

    

    本文研究了在一个金融市场中，股票回报依赖于一个隐藏的高斯均值回归漂移过程的情况下，功用最大化投资者的组合选择问题。漂移的信息是通过回报和专家意见的噪声信号获得的，这些信号随机地随时间到达。到达日期被建模为一个齐次泊松过程的跳跃时间。应用卡尔曼滤波技术，我们计算出了关于观测值的条件均值和协方差的隐藏漂移的估计值。利用动态规划方法解决了功用最大化问题。我们推导出了相关的动态规划方程，并对严谨的数学证明进行了正则化论证。

    This paper investigates the optimal selection of portfolios for power utility maximizing investors in a financial market where stock returns depend on a hidden Gaussian mean reverting drift process. Information on the drift is obtained from returns and expert opinions in the form of noisy signals about the current state of the drift arriving randomly over time. The arrival dates are modeled as the jump times of a homogeneous Poisson process. Applying Kalman filter techniques we derive estimates of the hidden drift which are described by the conditional mean and covariance of the drift given the observations. The utility maximization problem is solved with dynamic programming methods. We derive the associated dynamic programming equation and study regularization arguments for a rigorous mathematical justification.
    
[^5]: 基于LOB的深度学习模型用于股票价格趋势预测：一项基准研究

    LOB-Based Deep Learning Models for Stock Price Trend Prediction: A Benchmark Study. (arXiv:2308.01915v1 [q-fin.TR])

    [http://arxiv.org/abs/2308.01915](http://arxiv.org/abs/2308.01915)

    本研究通过基准研究探讨了基于LOB数据的股票价格趋势预测的15种最新DL模型的鲁棒性和泛化能力。实验证明这些模型在面对新数据时性能明显下降，对其在实际市场中的应用性提出了问题。

    

    深度学习（DL）研究在金融行业中产生了显著影响。我们研究了基于限价订单簿（LOB）数据的股票价格趋势预测（SPTP）的十五种最新 DL模型的鲁棒性和泛化能力。为了进行这项研究，我们开发了LOBCAST，一个开源框架，包括数据预处理、DL 模型训练、评估和利润分析。我们的大量实验表明，所有模型在面对新数据时性能显著下降，从而对它们在实际市场中的适用性提出了疑问。我们的工作作为一个基准，揭示了当前方法的潜力和局限性，并为创新解决方案提供了见解。

    The recent advancements in Deep Learning (DL) research have notably influenced the finance sector. We examine the robustness and generalizability of fifteen state-of-the-art DL models focusing on Stock Price Trend Prediction (SPTP) based on Limit Order Book (LOB) data. To carry out this study, we developed LOBCAST, an open-source framework that incorporates data preprocessing, DL model training, evaluation and profit analysis. Our extensive experiments reveal that all models exhibit a significant performance drop when exposed to new data, thereby raising questions about their real-world market applicability. Our work serves as a benchmark, illuminating the potential and the limitations of current approaches and providing insight for innovative solutions.
    
[^6]: 商品市场中的深度策略梯度方法

    Deep Policy Gradient Methods in Commodity Markets. (arXiv:2308.01910v1 [q-fin.TR])

    [http://arxiv.org/abs/2308.01910](http://arxiv.org/abs/2308.01910)

    本论文研究了深度强化学习方法在商品交易中的应用，并通过提供流动性和减少市场波动性来稳定市场。这对于解决能源市场的不稳定和全球能源危机具有重要意义。

    

    能源转型增加了对间歇性能源的依赖，不稳定了能源市场并导致了前所未有的波动性，最终导致了2021年的全球能源危机。除了对生产者和消费者造成伤害外，波动的能源市场还可能危及关键的碳减排努力。交易商通过提供流动性和减少波动性在稳定市场中扮演着重要角色。已经提出了几种数学和统计模型用于预测未来收益。然而，由于金融市场的低信噪比和非平稳动态，开发这样的模型并不是易事。本论文研究了深度强化学习方法在商品交易中的有效性。将商品交易问题形式化为连续离散时间随机动力系统。该系统采用了新颖的时间离散化方案，对市场波动做出反应并自适应，提供更好的统计特性。

    The energy transition has increased the reliance on intermittent energy sources, destabilizing energy markets and causing unprecedented volatility, culminating in the global energy crisis of 2021. In addition to harming producers and consumers, volatile energy markets may jeopardize vital decarbonization efforts. Traders play an important role in stabilizing markets by providing liquidity and reducing volatility. Several mathematical and statistical models have been proposed for forecasting future returns. However, developing such models is non-trivial due to financial markets' low signal-to-noise ratios and nonstationary dynamics.  This thesis investigates the effectiveness of deep reinforcement learning methods in commodities trading. It formalizes the commodities trading problem as a continuing discrete-time stochastic dynamical system. This system employs a novel time-discretization scheme that is reactive and adaptive to market volatility, providing better statistical properties f
    
[^7]: 默认提示什么时候有效?

    When do Default Nudges Work?. (arXiv:2301.08797v3 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2301.08797](http://arxiv.org/abs/2301.08797)

    通过在瑞典Covid-19疫苗推出中的区域变化，研究了对激励有差异的群体的提示效果，结果显示提示对于个人无意义的选择更有效。

    

    提示是科学和政策上一个新兴的话题，但不同激励群体中的提示效果的证据还不足。本文利用瑞典Covid-19疫苗推出过程中的区域变化，研究了对激励有差异的群体的提示效果：对于16-17岁的孩子，Covid-19不危险，而50-59岁的人则面临着严重疾病或死亡的巨大风险。我们发现，年轻人的反应显著强烈，这与提示对于个人无意义的选择更有效的理论相一致。

    Nudging is a burgeoning topic in science and in policy, but evidence on the effectiveness of nudges among differentially-incentivized groups is lacking. This paper exploits regional variations in the roll-out of the Covid-19 vaccine in Sweden to examine the effect of a nudge on groups whose intrinsic incentives are different: 16-17-year-olds, for whom Covid-19 is not dangerous, and 50-59-year-olds, who face a substantial risk of death or severe dis-ease. We find a significantly stronger response in the younger group, consistent with the theory that nudges are more effective for choices that are not meaningful to the individual.
    
[^8]: 使用注意力驱动的动态因子学习的生成CVaR组合优化算法

    Generative CVaR Portfolio Optimization with Attention-Powered Dynamic Factor Learning. (arXiv:2301.07318v3 [q-fin.PM] UPDATED)

    [http://arxiv.org/abs/2301.07318](http://arxiv.org/abs/2301.07318)

    本论文提出了一种使用注意力驱动的动态因子学习的生成CVaR组合优化算法，通过随机变量转换作为分布建模的隐式方式，以及使用注意力-GRU网络进行动态学习和预测，捕捉了多元股票收益之间的动态依赖关系，特别是关注尾部属性。通过在每个投资日期从学习到的生成模型中模拟新样本，并进一步应用CVaR组合优化策略，实现了更明智的投资决策。

    

    动态组合构建问题需要动态建模多元股票收益的联合分布。为了实现这一目标，我们提出了一种动态生成因子模型，它使用随机变量转换作为分布建模的隐式方式，并依赖于使用注意力-GRU网络进行动态学习和预测。所提出的模型捕捉了多元股票收益之间的动态依赖关系，特别是关注尾部属性。我们还提出了一个两步迭代算法来训练模型，然后预测时变的模型参数，包括时不变的尾部参数。在每个投资日期，我们可以轻松地从学习到的生成模型中模拟新样本，并进一步使用模拟样本来进行CVaR组合优化，形成动态组合策略。对股票数据的数值实验表明，我们的模型可以带来更明智的投资，承诺更高的回报风险比，并呈现出...

    The dynamic portfolio construction problem requires dynamic modeling of the joint distribution of multivariate stock returns. To achieve this, we propose a dynamic generative factor model which uses random variable transformation as an implicit way of distribution modeling and relies on the Attention-GRU network for dynamic learning and forecasting. The proposed model captures the dynamic dependence among multivariate stock returns, especially focusing on the tail-side properties. We also propose a two-step iterative algorithm to train the model and then predict the time-varying model parameters, including the time-invariant tail parameters. At each investment date, we can easily simulate new samples from the learned generative model, and we further perform CVaR portfolio optimization with the simulated samples to form a dynamic portfolio strategy. The numerical experiment on stock data shows that our model leads to wiser investments that promise higher reward-risk ratios and present l
    
[^9]: 均值-方差设置下的多臂赌博机问题

    Multiarmed Bandits Problem Under the Mean-Variance Setting. (arXiv:2212.09192v3 [math.OC] UPDATED)

    [http://arxiv.org/abs/2212.09192](http://arxiv.org/abs/2212.09192)

    本文将经典的多臂赌博机问题扩展到均值-方差设置，并通过考虑亚高斯臂放松了先前假设，解决了风险-回报权衡的问题。

    

    经典的多臂赌博机（MAB）问题涉及一个学习者和一个包含K个独立臂的集合，每个臂都有自己的事前未知独立奖励分布。在有限次选择中的每一次，学习者选择一个臂并接收新信息。学习者经常面临一个勘探-开发困境：通过玩估计奖励最高的臂来利用当前信息，还是探索所有臂以收集更多奖励信息。设计目标旨在最大化所有回合中的期望累积奖励。然而，这样的目标并不考虑风险-回报权衡，而这在许多应用领域，特别是金融和经济领域，常常是一项基本原则。在本文中，我们在Sani等人（2012）的基础上，将经典的MAB问题扩展到均值-方差设置。具体而言，我们通过考虑亚高斯臂放松了Sani等人（2012）做出的独立臂和有界奖励的假设。

    The classical multi-armed bandit (MAB) problem involves a learner and a collection of K independent arms, each with its own ex ante unknown independent reward distribution. At each one of a finite number of rounds, the learner selects one arm and receives new information. The learner often faces an exploration-exploitation dilemma: exploiting the current information by playing the arm with the highest estimated reward versus exploring all arms to gather more reward information. The design objective aims to maximize the expected cumulative reward over all rounds. However, such an objective does not account for a risk-reward tradeoff, which is often a fundamental precept in many areas of applications, most notably in finance and economics. In this paper, we build upon Sani et al. (2012) and extend the classical MAB problem to a mean-variance setting. Specifically, we relax the assumptions of independent arms and bounded rewards made in Sani et al. (2012) by considering sub-Gaussian arms.
    
[^10]: SoK：区块链去中心化

    SoK: Blockchain Decentralization. (arXiv:2205.04256v4 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2205.04256](http://arxiv.org/abs/2205.04256)

    该论文为区块链去中心化领域的知识系统化，提出了分类法并建议使用指数来衡量和量化区块链的去中心化水平。除了共识去中心化外，其他方面的研究相对较少。这项工作为未来的研究提供了基准和指导。

    

    区块链通过在点对点网络中实现分布式信任，为去中心化经济提供了支持。然而，令人惊讶的是，目前还缺乏广泛接受的去中心化定义或度量标准。我们通过全面分析现有研究，探索了区块链去中心化的知识系统化（SoK）。首先，我们通过对现有研究的定性分析，在共识、网络、治理、财富和交易等五个方面建立了用于分析区块链去中心化的分类法。我们发现，除了共识去中心化以外，其他方面的研究相对较少。其次，我们提出了一种指数，通过转换香农熵来衡量和量化区块链在不同方面的去中心化水平。我们通过比较静态模拟验证了该指数的可解释性。我们还提供了其他指数的定义和讨论，包括基尼系数、中本聪系数和赫尔曼-赫尔东指数等。我们的工作概述了当前区块链去中心化的景象，并提出了一个量化的度量标准，为未来的研究提供基准。

    Blockchain empowers a decentralized economy by enabling distributed trust in a peer-to-peer network. However, surprisingly, a widely accepted definition or measurement of decentralization is still lacking. We explore a systematization of knowledge (SoK) on blockchain decentralization by comprehensively analyzing existing studies in various aspects. First, we establish a taxonomy for analyzing blockchain decentralization in the five facets of consensus, network, governance, wealth, and transaction bu qualitative analysis of existing research. We find relatively little research on aspects other than consensus decentralization. Second, we propose an index that measures and quantifies the decentralization level of blockchain across different facets by transforming Shannon entropy. We verify the explainability of the index via comparative static simulations. We also provide the definition and discussion of alternative indices including the Gini Coefficient, Nakamoto Coefficient, and Herfind
    

