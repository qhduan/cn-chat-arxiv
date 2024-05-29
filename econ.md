# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rationality Report Cards: Assessing the Economic Rationality of Large Language Models](https://arxiv.org/abs/2402.09552) | 本文在评估大型语言模型的经济合理性方面提出了一种方法，通过量化评分模型在各个要素上的表现并结合用户提供的评分标准，生成一份"理性报告卡"，以确定代理人是否足够可靠。 |
| [^2] | [The learning effects of subsidies to bundled goods: a semiparametric approach.](http://arxiv.org/abs/2311.01217) | 本文研究了捆绑商品补贴的学习效应，发现一次性补贴捆绑商品既有直接价格效应，也有间接学习动机，可以通过增加信息遗产来引起长期需求变化。 |
| [^3] | [On Optimal Set Estimation for Partially Identified Binary Choice Models.](http://arxiv.org/abs/2310.02414) | 本文重新考虑了部分识别模型估计中的最优性概念，并提出了在所有设计中收敛于识别区域的替代方法。 |
| [^4] | [Proofs for the New Definitions in Financial Markets.](http://arxiv.org/abs/2309.03003) | 本研究提供了金融市场新定义的证明，通过构造定理来确定新定义中的效用曲线形状。与标准理论不同，新定义中出现了严格凹性、严格凸性或线性的情况。 |
| [^5] | [Empirical Evidence for the New Definitions in Financial Markets.](http://arxiv.org/abs/2305.03468) | 研究证实了新的金融市场定义准确反映了投资者行为，并提供了投资策略方面的建议。 |
| [^6] | [On Graphical Methods in Stochastic Choice.](http://arxiv.org/abs/2303.14249) | 该论文提供了基于图论工具的随机选择概率特征描述，并在不完整领域中提供了随机理性的新刻画。 |
| [^7] | [On Robust Inference in Time Series Regression.](http://arxiv.org/abs/2203.04080) | 该论文研究了时间序列回归中的鲁棒推断问题，指出了将HAC估计技术应用于时间序列环境时的困难，并讨论了OLS参数估计不一致、HAC回归参数估计低效以及HAC条件预测低效等问题。 |

# 详细

[^1]: 理性报告卡：评估大型语言模型的经济合理性

    Rationality Report Cards: Assessing the Economic Rationality of Large Language Models

    [https://arxiv.org/abs/2402.09552](https://arxiv.org/abs/2402.09552)

    本文在评估大型语言模型的经济合理性方面提出了一种方法，通过量化评分模型在各个要素上的表现并结合用户提供的评分标准，生成一份"理性报告卡"，以确定代理人是否足够可靠。

    

    越来越多的人对将LLM用作决策"代理人"兴趣日益增加。这包括很多自由度：应该使用哪个模型；如何进行提示；是否要求其进行内省、进行思考链等。解决这些问题（更广泛地说，确定LLM代理人是否足够可靠以便获得信任）需要一种评估这种代理人经济合理性的方法论，在本文中我们提供了一个方法。我们首先对理性决策的经济文献进行了调研、将代理人应该展现的大量细粒度"要素"进行分类，并确定了它们之间的依赖关系。然后，我们提出了一个基准分布，以定量评分LLM在这些要素上的表现，并结合用户提供的评分标准，生成一份"理性报告卡"。最后，我们描述了与14种不同的LLM进行的大规模实证实验的结果。

    arXiv:2402.09552v1 Announce Type: new  Abstract: There is increasing interest in using LLMs as decision-making "agents." Doing so includes many degrees of freedom: which model should be used; how should it be prompted; should it be asked to introspect, conduct chain-of-thought reasoning, etc? Settling these questions -- and more broadly, determining whether an LLM agent is reliable enough to be trusted -- requires a methodology for assessing such an agent's economic rationality. In this paper, we provide one. We begin by surveying the economic literature on rational decision making, taxonomizing a large set of fine-grained "elements" that an agent should exhibit, along with dependencies between them. We then propose a benchmark distribution that quantitatively scores an LLMs performance on these elements and, combined with a user-provided rubric, produces a "rationality report card." Finally, we describe the results of a large-scale empirical experiment with 14 different LLMs, characte
    
[^2]: 捆绑商品补贴的学习效应：半参数方法

    The learning effects of subsidies to bundled goods: a semiparametric approach. (arXiv:2311.01217v1 [econ.EM])

    [http://arxiv.org/abs/2311.01217](http://arxiv.org/abs/2311.01217)

    本文研究了捆绑商品补贴的学习效应，发现一次性补贴捆绑商品既有直接价格效应，也有间接学习动机，可以通过增加信息遗产来引起长期需求变化。

    

    临时补贴捆绑商品是否能够通过学习其中一种商品的相对质量来引起长期需求变化？本文从理论和实证角度提供了关于此机制的证据。在理论上，我们引入了一个模型，代理人通过消费来学习必需品创新的质量。我们的结果显示，对包含该创新的捆绑商品进行一次性补贴的即时效应可以分解为直接价格效应和间接学习动机，代理人利用折扣增加信息遗产留给未来的自己。然后，我们在一个随机实验中评估了我们理论的预测，该实验在一个拼车平台上为与火车或地铁站整合的汽车行程提供了两周的折扣。考虑到我们数据的重尾特性，我们遵循Athey（2023）的方法，并基于我们的理论提出了一个半参数模型。

    Can temporary subsidies to bundles induce long-run changes in demand due to learning about the relative quality of one of its constituent goods? This paper provides theoretical and experimental evidence on the role of this mechanism. Theoretically, we introduce a model where an agent learns about the quality of an innovation on an essential good through consumption. Our results show that the contemporaneous effect of a one-off subsidy to a bundle that contains the innovation may be decomposed into a direct price effect, and an indirect learning motive, whereby an agent leverages the discount to increase the informational bequest left to her future selves. We then assess the predictions of our theory in a randomised experiment in a ridesharing platform. The experiment provided two-week discounts for car trips integrating with a train or metro station (a bundle). Given the heavy-tailed nature of our data, we follow \cite{Athey2023} and, motivated by our theory, propose a semiparametric m
    
[^3]: 关于部分识别二项选择模型的最优集估计

    On Optimal Set Estimation for Partially Identified Binary Choice Models. (arXiv:2310.02414v1 [econ.EM])

    [http://arxiv.org/abs/2310.02414](http://arxiv.org/abs/2310.02414)

    本文重新考虑了部分识别模型估计中的最优性概念，并提出了在所有设计中收敛于识别区域的替代方法。

    

    在本文中，我们重新考虑了部分识别模型估计中的最优性概念。我们以半参数二项选择模型为例，以离散协变量作为示例，说明了一般问题。该模型在一定程度上是部分识别的，例如Bierens和Hartog（1988）所示。通过实施Manski（1975）提出的最大分数程序，可以构建模型中回归系数的集合估计。对于许多设计，该方法收敛于这些参数的识别集，因此在某种意义上是最优的。但是，正如Komarova（2013）所示，对于其他情况，最大分数目标函数给出了识别集的边界区域。这激发了寻求其他优化方法的动力，这些方法在所有设计中都收敛于识别区域，并且我们提出并比较了这些方法。一个是Hodges类型的估计器，将最大分数估计器与现有程序相结合。第二个是两步法

    In this paper we reconsider the notion of optimality in estimation of partially identified models. We illustrate the general problem in the context of a semiparametric binary choice model with discrete covariates as an example of a model which is partially identified as shown in, e.g. Bierens and Hartog (1988). A set estimator for the regression coefficients in the model can be constructed by implementing the Maximum Score procedure proposed by Manski (1975). For many designs this procedure converges to the identified set for these parameters, and so in one sense is optimal. But as shown in Komarova (2013) for other cases the Maximum Score objective function gives an outer region of the identified set. This motivates alternative methods that are optimal in one sense that they converge to the identified region in all designs, and we propose and compare such procedures. One is a Hodges type estimator combining the Maximum Score estimator with existing procedures. A second is a two step e
    
[^4]: 金融市场新定义的证明

    Proofs for the New Definitions in Financial Markets. (arXiv:2309.03003v1 [q-fin.GN])

    [http://arxiv.org/abs/2309.03003](http://arxiv.org/abs/2309.03003)

    本研究提供了金融市场新定义的证明，通过构造定理来确定新定义中的效用曲线形状。与标准理论不同，新定义中出现了严格凹性、严格凸性或线性的情况。

    

    构造定理可以帮助确定金融市场新定义中构成的某些效用曲线的形状。本研究旨在为这些定理提供证明。尽管风险厌恶、风险爱好和风险中性等术语在标准理论中分别等同于严格凹性、严格凸性和线性，但某些新定义满足严格凹性或严格凸性，或线性。

    Constructing theorems can help to determine the shape of certain utility curves that make up the new definitions in financial markets. The aim of this study was to present proofs for these theorems. Although the terms of risk-averse, risk-loving, and risk-neutral are equivalent to strict concavity, strict convexity, and linearity, respectively, in standard theory, certain new definitions satisfy strict concavity or strict convexity, or linearity.
    
[^5]: 金融市场新定义的经验证据

    Empirical Evidence for the New Definitions in Financial Markets. (arXiv:2305.03468v1 [q-fin.GN])

    [http://arxiv.org/abs/2305.03468](http://arxiv.org/abs/2305.03468)

    研究证实了新的金融市场定义准确反映了投资者行为，并提供了投资策略方面的建议。

    

    本研究给出了支持金融市场新定义的经验证据。分析了1889-1978年美国金融市场投资者的风险态度，结果表明，1977年在投资综合S＆P 500指数的股票投资者是风险规避者。相反，投资美国国债的无风险资产投资者则表现出不足的风险偏爱，这可以被认为是一种风险规避行为。这些发现表明，金融市场新定义准确反映了投资者的行为，应考虑在投资策略中。

    This study presents empirical evidence to support the validity of new definitions in financial markets. The risk attitudes of investors in US financial markets from 1889-1978 are analyzed and the results indicate that equity investors who invested in the composite S&P 500 index were risk-averse in 1977. Conversely, risk-free asset investors who invested in US Treasury bills were found to exhibit not enough risk-loving behavior, which can be considered a type of risk-averse behavior. These findings suggest that the new definitions in financial markets accurately reflect the behavior of investors and should be considered in investment strategies.
    
[^6]: 关于随机选择中的图形方法

    On Graphical Methods in Stochastic Choice. (arXiv:2303.14249v1 [econ.TH])

    [http://arxiv.org/abs/2303.14249](http://arxiv.org/abs/2303.14249)

    该论文提供了基于图论工具的随机选择概率特征描述，并在不完整领域中提供了随机理性的新刻画。

    

    近年来，有大量论文使用图论工具研究随机选择。Fiornini (2004)的研究成果为该领域提供了基础，通过提供选择概率的图形表示，并显示该图的每个内部节点必须满足流入等于流出。我们发现，这种流入等于流出特性几乎是选择概率的特征。通过这种方式，我们使用图论工具对选择概率进行特征描述。作为该结果的一个应用，我们提供了不完整领域中随机理性的新刻画。

    In recent years there has been an influx of papers which use graph theoretic tools to study stochastic choice. Fiorini (2004) serves as a base for this literature by providing a graphical representation of choice probabilities and showing that every interior node of this graph must satisfy inflow equals outflow. We show that this inflow equals outflow property is almost characteristic of choice probabilities. In doing so, we characterize choice probabilities through graph theoretic tools. As an application of this result, we provide a novel characterization of stochastic rationality on an incomplete domain.
    
[^7]: 关于时间序列回归中的鲁棒推断

    On Robust Inference in Time Series Regression. (arXiv:2203.04080v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2203.04080](http://arxiv.org/abs/2203.04080)

    该论文研究了时间序列回归中的鲁棒推断问题，指出了将HAC估计技术应用于时间序列环境时的困难，并讨论了OLS参数估计不一致、HAC回归参数估计低效以及HAC条件预测低效等问题。

    

    在横截面环境中，具有异方差和自相关一致（HAC）标准误的最小二乘回归非常有用。然而，在将HAC估计技术应用于时间序列环境时，会面临几个常常被忽视的主要困难。首先，在可能出现强外生性失败的时间序列环境中，OLS参数估计可能不一致，因此即使渐近下也无法进行HAC推断。其次，大多数经济时间序列具有强自相关性，这使得HAC回归参数估计非常低效。第三，强自相关性同样使得HAC条件预测非常低效。最后，流行的HAC估计器的结构不适合捕捉通常存在于经济时间序列中的自回归自相关性，这会产生较大的规模扭曲和HAC假设检验中的功效降低，除非样本数量非常大。

    Least squares regression with heteroskedasticity and autocorrelation consistent (HAC) standard errors has proved very useful in cross section environments. However, several major difficulties, which are generally overlooked, must be confronted when transferring the HAC estimation technology to time series environments. First, in plausible time-series environments involving failure of strong exogeneity, OLS parameter estimates can be inconsistent, so that HAC inference fails even asymptotically. Second, most economic time series have strong autocorrelation, which renders HAC regression parameter estimates highly inefficient. Third, strong autocorrelation similarly renders HAC conditional predictions highly inefficient. Finally, The structure of popular HAC estimators is ill-suited for capturing the autoregressive autocorrelation typically present in economic time series, which produces large size distortions and reduced power in HACbased hypothesis testing, in all but the largest sample
    

