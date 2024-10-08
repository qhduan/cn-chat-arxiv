# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Postprocessing of point predictions for probabilistic forecasting of electricity prices: Diversity matters](https://arxiv.org/abs/2404.02270) | 将点预测转换为概率预测的电力价格后处理方法中，结合Isotonic Distributional Regression与其他两种方法的预测分布可以实现显著的性能提升。 |
| [^2] | [Numerical Claim Detection in Finance: A New Financial Dataset, Weak-Supervision Model, and Market Analysis](https://arxiv.org/abs/2402.11728) | 本研究探讨了分析师报告和盈利电话中的索赔对金融市场回报的影响，并构建了一个新的金融数据集用于索赔检测任务。提出了一种融入主题专家知识的新型弱监督模型，通过构建一种新的度量“乐观主义”展示了模型的实际效用。 |
| [^3] | [Callable convertible bonds under liquidity constraints and hybrid priorities](https://arxiv.org/abs/2111.02554) | 本文研究了在流动性约束下的可调转换债券问题，提出了一个完整解决方案，并介绍了一个非有序情况下处理的方法。 |
| [^4] | [Signature Methods in Stochastic Portfolio Theory.](http://arxiv.org/abs/2310.02322) | 线性路径函数投资组合是一种通用的投资组合类别，可以通过签名投资组合来一致逼近市场权重的连续、可能路径相关的投资组合函数，并在多类非马尔科夫市场模型中任意好地逼近增长最优投资组合。 |
| [^5] | [Inequality in Educational Attainment: Urban-Rural Comparison in the Indian Context.](http://arxiv.org/abs/2307.16238) | 该研究比较了印度15个邦在1981-2011年期间城乡识字率的差异，并研究了减少城乡教育不平等的因素。研究发现，尽管识字差距在减小，但2011年安得拉邦、中央邦、古吉拉特邦、奥里萨邦、马哈拉施特拉邦甚至卡纳塔克邦在城乡教育方面仍面临较高的不平等。此外，研究还指出，农村妇女生育率的降低和农村女性21岁后婚姻的比例较高可以减少城乡地区的识字差距。 |

# 详细

[^1]: 电力价格概率预测的点预测后处理：多样性至关重要

    Postprocessing of point predictions for probabilistic forecasting of electricity prices: Diversity matters

    [https://arxiv.org/abs/2404.02270](https://arxiv.org/abs/2404.02270)

    将点预测转换为概率预测的电力价格后处理方法中，结合Isotonic Distributional Regression与其他两种方法的预测分布可以实现显著的性能提升。

    

    依赖于电力价格的预测分布进行操作决策相较于仅基于点预测的决策可以带来显著更高的利润。然而，在学术和工业环境中开发的大多数模型仅提供点预测。为了解决这一问题，我们研究了三种将点预测转换为概率预测的后处理方法：分位数回归平均、一致性预测和最近引入的等温分布回归。我们发现，虽然等温分布回归表现最为多样化，但将其预测分布与另外两种方法结合使用，相较于具有正态分布误差的基准模型，在德国电力市场的4.5年测试期间（涵盖COVID大流行和乌克兰战争），实现了约7.5%的改进。值得注意的是，这种组合的性能与最先进的Dis

    arXiv:2404.02270v1 Announce Type: new  Abstract: Operational decisions relying on predictive distributions of electricity prices can result in significantly higher profits compared to those based solely on point forecasts. However, the majority of models developed in both academic and industrial settings provide only point predictions. To address this, we examine three postprocessing methods for converting point forecasts into probabilistic ones: Quantile Regression Averaging, Conformal Prediction, and the recently introduced Isotonic Distributional Regression. We find that while IDR demonstrates the most varied performance, combining its predictive distributions with those of the other two methods results in an improvement of ca. 7.5% compared to a benchmark model with normally distributed errors, over a 4.5-year test period in the German power market spanning the COVID pandemic and the war in Ukraine. Remarkably, the performance of this combination is at par with state-of-the-art Dis
    
[^2]: 金融领域的数字化索赔检测：一个新的金融数据集、弱监督模型和市场分析

    Numerical Claim Detection in Finance: A New Financial Dataset, Weak-Supervision Model, and Market Analysis

    [https://arxiv.org/abs/2402.11728](https://arxiv.org/abs/2402.11728)

    本研究探讨了分析师报告和盈利电话中的索赔对金融市场回报的影响，并构建了一个新的金融数据集用于索赔检测任务。提出了一种融入主题专家知识的新型弱监督模型，通过构建一种新的度量“乐观主义”展示了模型的实际效用。

    

    在本文中，我们研究了分析师报告和盈利电话中的索赔对金融市场回报的影响，将它们视为上市公司重要的季度事件。为了进行全面的分析，我们构建了一个新的金融数据集，用于金融领域的索赔检测任务。我们在该数据集上对各种语言模型进行了基准测试，并提出了一种融入主题专家（SMEs）知识的新型弱监督模型，在聚合函数中超越了现有方法。此外，我们通过构建一种新的度量“乐观主义”展示了我们提出的模型的实际效用。我们还观察到盈利惊喜和回报对我们的乐观主义度量的依赖。我们的数据集、模型和代码将在GitHub和Hugging Face上公开（遵循CC BY 4.0许可）。

    arXiv:2402.11728v1 Announce Type: new  Abstract: In this paper, we investigate the influence of claims in analyst reports and earnings calls on financial market returns, considering them as significant quarterly events for publicly traded companies. To facilitate a comprehensive analysis, we construct a new financial dataset for the claim detection task in the financial domain. We benchmark various language models on this dataset and propose a novel weak-supervision model that incorporates the knowledge of subject matter experts (SMEs) in the aggregation function, outperforming existing approaches. Furthermore, we demonstrate the practical utility of our proposed model by constructing a novel measure ``optimism". Furthermore, we observed the dependence of earnings surprise and return on our optimism measure. Our dataset, models, and code will be made publicly (under CC BY 4.0 license) available on GitHub and Hugging Face.
    
[^3]: 可调转换债券在流动性约束和混合优先级下的研究

    Callable convertible bonds under liquidity constraints and hybrid priorities

    [https://arxiv.org/abs/2111.02554](https://arxiv.org/abs/2111.02554)

    本文研究了在流动性约束下的可调转换债券问题，提出了一个完整解决方案，并介绍了一个非有序情况下处理的方法。

    

    本文研究了在由泊松信号建模的流动性约束下的可调转换债券问题。我们假设当债券持有人和公司同时停止游戏时，他们都没有绝对优先级，而是一部分$m\in[0,1]$的债券转换为公司的股票，其余被公司调用。因此，本文推广了[Liang和Sun，带泊松随机干预时间的Dynkin博弈，SIAM控制和优化杂志，57(2019)，2962-2991]中研究的特殊情况（债券持有人有优先权，$m=1$），并提出了具有流动性约束的可调转换债券问题的完整解决方案。可调转换债券是Dynkin博弈的一个例子，但不属于标准范式，因为收益不以有序方式取决于哪个代理停止游戏。我们展示了如何通过引入...

    arXiv:2111.02554v2 Announce Type: replace  Abstract: This paper investigates the callable convertible bond problem in the presence of a liquidity constraint modelled by Poisson signals. We assume that neither the bondholder nor the firm has absolute priority when they stop the game simultaneously, but instead, a proportion $m\in[0,1]$ of the bond is converted to the firm's stock and the rest is called by the firm. The paper thus generalizes the special case studied in [Liang and Sun, Dynkin games with Poisson random intervention times, SIAM Journal on Control and Optimization, 57 (2019), 2962-2991] where the bondholder has priority ($m=1$), and presents a complete solution to the callable convertible bond problem with liquidity constraint. The callable convertible bond is an example of a Dynkin game, but falls outside the standard paradigm since the payoffs do not depend in an ordered way upon which agent stops the game. We show how to deal with this non-ordered situation by introducin
    
[^4]: 随机投资组合理论中的签名方法

    Signature Methods in Stochastic Portfolio Theory. (arXiv:2310.02322v1 [q-fin.MF])

    [http://arxiv.org/abs/2310.02322](http://arxiv.org/abs/2310.02322)

    线性路径函数投资组合是一种通用的投资组合类别，可以通过签名投资组合来一致逼近市场权重的连续、可能路径相关的投资组合函数，并在多类非马尔科夫市场模型中任意好地逼近增长最优投资组合。

    

    在随机投资组合理论的背景下，我们引入了一种新颖的投资组合类别，称之为线性路径函数投资组合。这些投资组合是由某些线性函数的转化所确定的特征映射的非预测路径函数的集合。我们以市场权重的签名(排名)作为这些特征映射的主要示例。我们证明了这些投资组合在某种意义上是通用的，即市场权重的连续、可能路径相关的投资组合函数可以通过签名投资组合进行一致逼近。我们还展示了签名投资组合在几类非马尔科夫市场模型中可以任意好地逼近增长最优投资组合，并通过数值实验说明，训练得到的签名投资组合与理论增长最优投资组合非常接近。除了这些通用性特征之外，主要的数值优势 lies in the fact th...

    In the context of stochastic portfolio theory we introduce a novel class of portfolios which we call linear path-functional portfolios. These are portfolios which are determined by certain transformations of linear functions of a collections of feature maps that are non-anticipative path functionals of an underlying semimartingale. As main example for such feature maps we consider the signature of the (ranked) market weights. We prove that these portfolios are universal in the sense that every continuous, possibly path-dependent, portfolio function of the market weights can be uniformly approximated by signature portfolios. We also show that signature portfolios can approximate the growth-optimal portfolio in several classes of non-Markovian market models arbitrarily well and illustrate numerically that the trained signature portfolios are remarkably close to the theoretical growth-optimal portfolios. Besides these universality features, the main numerical advantage lies in the fact th
    
[^5]: 印度背景下教育水平的不平等：城乡比较

    Inequality in Educational Attainment: Urban-Rural Comparison in the Indian Context. (arXiv:2307.16238v1 [econ.GN])

    [http://arxiv.org/abs/2307.16238](http://arxiv.org/abs/2307.16238)

    该研究比较了印度15个邦在1981-2011年期间城乡识字率的差异，并研究了减少城乡教育不平等的因素。研究发现，尽管识字差距在减小，但2011年安得拉邦、中央邦、古吉拉特邦、奥里萨邦、马哈拉施特拉邦甚至卡纳塔克邦在城乡教育方面仍面临较高的不平等。此外，研究还指出，农村妇女生育率的降低和农村女性21岁后婚姻的比例较高可以减少城乡地区的识字差距。

    

    本文试图比较印度选定的15个邦在1981-2011年期间城乡识字率，并探讨减少城乡教育取得差异的工具。研究构建了Sopher的城乡差异识字指数，分析了印度15个邦的识字差距趋势。尽管识字差距随着时间的推移有所减小，但Sopher指数显示，2011年安得拉邦、中央邦、古吉拉特邦、奥里萨邦、马哈拉施特拉邦甚至卡纳塔克邦在城乡教育方面面临较高的不平等。此外，本研究还应用了固定效应面板数据回归技术，以确定影响城乡教育不平等的因素。模型表明，以下因素可以减少印度城乡地区之间的识字差距：农村妇女生育率低，农村女性21岁后婚姻的比例较高。

    The article tries to compare urban and rural literacy of fifteen selected Indian states during 1981 - 2011 and explores the instruments which can reduce the disparity in urban and rural educational attainment. The study constructs Sopher's urban-rural differential literacy index to analyze the trends of literacy disparity across fifteen states in India over time. Although literacy disparity has decreased over time, Sopher's index shows that the states of Andhra Pradesh, Madhya Pradesh, Gujarat, Odisha, Maharashtra and even Karnataka faced high inequality in education between urban and rural India in 2011. Additionally, the Fixed Effect panel data regression technique has been applied in the study to identify the factors which influence urban-rural inequality in education. The model shows that the following factors can reduce literacy disparity between urban and rural areas of India: low fertility rate in rural women, higher percentages of rural females marrying after the age of 21 year
    

