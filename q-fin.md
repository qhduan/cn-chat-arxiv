# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rough volatility, path-dependent PDEs and weak rates of convergence.](http://arxiv.org/abs/2304.03042) | 该研究探究了粗糙波动率模型中的路径依赖PDE问题，证明了条件期望是其唯一的经典解。并且，该研究还研究了具有Hurst参数的离散随机积分的弱收敛率，以逼近粗糙波动率模型中的对数股票价格。 |
| [^2] | [Modelling customer lifetime-value in the retail banking industry.](http://arxiv.org/abs/2304.03038) | 本研究提出了一个通用的框架，可以应用于具有长期合同和产品中心客户关系的行业来建模客户的生命周期价值，该框架可以预测任意时间范围内的CLV，并可生成基于产品的倾向模型，这在零售银行业中尤其重要。通过测试，我们证明了相对于传统算法，该模型可以提高43%的超出时间的CLV预测误差。 |
| [^3] | [Efficient OCR for Building a Diverse Digital History.](http://arxiv.org/abs/2304.02737) | 本研究使用对比训练的视觉编码器，将OCR建模为字符级图像检索问题，相比于已有架构更具样本效率和可扩展性，从而使数字历史更具代表性的文献史料得以更好地参与社区。 |
| [^4] | [Measuring Discrete Risks on Infinite Domains: Theoretical Foundations, Conditional Five Number Summaries, and Data Analyses.](http://arxiv.org/abs/2304.02723) | 本文提出了一种新的离散损失分布截断方法，将平滑分位数估计的统计推断从有限域扩展到无限域。并提出一种基于自助法的灵活方法，用于实际中的应用。此外，我们计算了构成条件五数摘要（C5NS）的五个分位数的置信区间，并计算尾部概率，结果表明，平滑分位数方法不仅分类投资组合的尾部风险性，还可对其进行分类。 |
| [^5] | [Quantifying dimensional change in stochastic portfolio theory.](http://arxiv.org/abs/2303.00858) | 本文研究了具有变化维度的股票组合理论，通过构建不同类型的自融资股票组合来解释维度变化如何影响组合回报，并对一些经典组合进行了实证分析，量化了维度变化对组合表现的影响。 |

# 详细

[^1]: 粗糙波动率，路径依赖PDE和弱收敛率的研究

    Rough volatility, path-dependent PDEs and weak rates of convergence. (arXiv:2304.03042v1 [math.PR])

    [http://arxiv.org/abs/2304.03042](http://arxiv.org/abs/2304.03042)

    该研究探究了粗糙波动率模型中的路径依赖PDE问题，证明了条件期望是其唯一的经典解。并且，该研究还研究了具有Hurst参数的离散随机积分的弱收敛率，以逼近粗糙波动率模型中的对数股票价格。

    

    在随机Volterra方程的设置中，特别是粗糙波动率模型中，我们展示了条件期望是路径依赖PDE的唯一经典解。后者由[Viens，F。，＆Zhang，J。（2019）。对分数布朗运动及其相关路径依赖PDE的鞅方法的开发而来。Ann. Appl. Probab.],。然后，我们利用这些工具研究具有Hurst参数$H \in (0,1/2)$的Riemann-Liouville分数布朗运动的平滑函数的离散随机积分的弱收敛率。这些积分逼近粗糙波动率模型中的对数股票价格。如果测试函数是二次的，则我们获得1阶弱误差率，如果测试函数是平滑的，则获得$H + 1/2$阶的误差率。

    In the setting of stochastic Volterra equations, and in particular rough volatility models, we show that conditional expectations are the unique classical solutions to path-dependent PDEs. The latter arise from the functional It\^o formula developed by [Viens, F., & Zhang, J. (2019). A martingale approach for fractional Brownian motions and related path dependent PDEs. Ann. Appl. Probab.]. We then leverage these tools to study weak rates of convergence for discretised stochastic integrals of smooth functions of a Riemann-Liouville fractional Brownian motion with Hurst parameter $H \in (0,1/2)$. These integrals approximate log-stock prices in rough volatility models. We obtain weak error rates of order 1 if the test function is quadratic and of order $H+1/2$ for smooth test functions.
    
[^2]: 在零售银行业中建模客户生命周期价值

    Modelling customer lifetime-value in the retail banking industry. (arXiv:2304.03038v1 [cs.LG])

    [http://arxiv.org/abs/2304.03038](http://arxiv.org/abs/2304.03038)

    本研究提出了一个通用的框架，可以应用于具有长期合同和产品中心客户关系的行业来建模客户的生命周期价值，该框架可以预测任意时间范围内的CLV，并可生成基于产品的倾向模型，这在零售银行业中尤其重要。通过测试，我们证明了相对于传统算法，该模型可以提高43%的超出时间的CLV预测误差。

    

    理解客户生命周期价值是培养长期客户关系的关键，但估计它远非易事。在零售银行业中，常用的方法依赖于简单的启发式算法，并未充分利用现代机器学习技术的高预测能力。我们提出了一个通用的框架来建模客户生命周期价值，该框架可应用于具有长期合同和产品中心客户关系的行业，其中零售银行就是一个例子。该框架的创新之处在于可以在任意时间范围内进行CLV预测和基于产品的倾向模型。我们还详细介绍了这个模型的实现，该模型目前已在一家大型英国放贷机构中投入生产。在测试中，相对于一种流行的基线方法，我们估计在时间外CLV预测误差方面有43%的改善。从我们的CLV模型派生的倾向模型已被用于支持客户联络营销活动。

    Understanding customer lifetime value is key to nurturing long-term customer relationships, however, estimating it is far from straightforward. In the retail banking industry, commonly used approaches rely on simple heuristics and do not take advantage of the high predictive ability of modern machine learning techniques. We present a general framework for modelling customer lifetime value which may be applied to industries with long-lasting contractual and product-centric customer relationships, of which retail banking is an example. This framework is novel in facilitating CLV predictions over arbitrary time horizons and product-based propensity models. We also detail an implementation of this model which is currently in production at a large UK lender. In testing, we estimate an 43% improvement in out-of-time CLV prediction error relative to a popular baseline approach. Propensity models derived from our CLV model have been used to support customer contact marketing campaigns. In test
    
[^3]: 建设多样化数字历史的高效OCR

    Efficient OCR for Building a Diverse Digital History. (arXiv:2304.02737v1 [cs.CV])

    [http://arxiv.org/abs/2304.02737](http://arxiv.org/abs/2304.02737)

    本研究使用对比训练的视觉编码器，将OCR建模为字符级图像检索问题，相比于已有架构更具样本效率和可扩展性，从而使数字历史更具代表性的文献史料得以更好地参与社区。

    

    每天有成千上万的用户查阅数字档案，但他们可以使用的信息并不能代表各种文献史料的多样性。典型用于光学字符识别（OCR）的序列到序列架构——联合学习视觉和语言模型——在低资源文献集合中很难扩展，因为学习语言-视觉模型需要大量标记的序列和计算。本研究将OCR建模为字符级图像检索问题，使用对比训练的视觉编码器。因为该模型只学习字符的视觉特征，它比现有架构更具样本效率和可扩展性，能够在现有解决方案失败的情况下实现准确的OCR。关键是，该模型为社区参与在使数字历史更具代表性的文献史料方面开辟了新的途径。

    Thousands of users consult digital archives daily, but the information they can access is unrepresentative of the diversity of documentary history. The sequence-to-sequence architecture typically used for optical character recognition (OCR) - which jointly learns a vision and language model - is poorly extensible to low-resource document collections, as learning a language-vision model requires extensive labeled sequences and compute. This study models OCR as a character level image retrieval problem, using a contrastively trained vision encoder. Because the model only learns characters' visual features, it is more sample efficient and extensible than existing architectures, enabling accurate OCR in settings where existing solutions fail. Crucially, the model opens new avenues for community engagement in making digital history more representative of documentary history.
    
[^4]: 无限域上离散风险的测量：理论基础、条件五数摘要和数据分析

    Measuring Discrete Risks on Infinite Domains: Theoretical Foundations, Conditional Five Number Summaries, and Data Analyses. (arXiv:2304.02723v1 [stat.AP])

    [http://arxiv.org/abs/2304.02723](http://arxiv.org/abs/2304.02723)

    本文提出了一种新的离散损失分布截断方法，将平滑分位数估计的统计推断从有限域扩展到无限域。并提出一种基于自助法的灵活方法，用于实际中的应用。此外，我们计算了构成条件五数摘要（C5NS）的五个分位数的置信区间，并计算尾部概率，结果表明，平滑分位数方法不仅分类投资组合的尾部风险性，还可对其进行分类。

    

    为了适应众多实际情况，本文将平滑分位数估计的统计推断从有限域扩展到无限域。我们通过新设计的离散损失分布截断方法完成此任务。模拟研究展示了这种方法在几种分布（如泊松分布、负二项分布及其零膨胀版本）中的应用，这些分布通常用于模拟保险业中的索赔频率。此外，我们提出了一种非常灵活的基于自助法的方法，用于实际中的应用。我们使用汽车事故数据及其修改，计算构成条件五数摘要（C5NS）的五个分位数的置信区间，并计算尾部概率。结果表明，平滑分位数方法不仅分类投资组合的尾部风险性，还可对其进行分类。

    To accommodate numerous practical scenarios, in this paper we extend statistical inference for smoothed quantile estimators from finite domains to infinite domains. We accomplish the task with the help of a newly designed truncation methodology for discrete loss distributions with infinite domains. A simulation study illustrates the methodology in the case of several distributions, such as Poisson, negative binomial, and their zero inflated versions, which are commonly used in insurance industry to model claim frequencies. Additionally, we propose a very flexible bootstrap-based approach for the use in practice. Using automobile accident data and their modifications, we compute what we have termed the conditional five number summary (C5NS) for the tail risk and construct confidence intervals for each of the five quantiles making up C5NS, and then calculate the tail probabilities. The results show that the smoothed quantile approach classifies the tail riskiness of portfolios not only m
    
[^5]: 随机组合理论中的维度变化定量化研究

    Quantifying dimensional change in stochastic portfolio theory. (arXiv:2303.00858v2 [q-fin.MF] UPDATED)

    [http://arxiv.org/abs/2303.00858](http://arxiv.org/abs/2303.00858)

    本文研究了具有变化维度的股票组合理论，通过构建不同类型的自融资股票组合来解释维度变化如何影响组合回报，并对一些经典组合进行了实证分析，量化了维度变化对组合表现的影响。

    

    本文研究了具有变化维度的股票市场组合理论中的功能生成理论。通过引入市场中的维度跳跃以及维度跳跃之间的股票资本化跳跃，我们构建了不同类型的自融资股票组合（加法、乘法和排名）在一个非常普通的环境中。我们的研究阐释了由于股票上市或退市事件以及市场上未预期的冲击所引起的维度变化如何影响组合回报。我们还对一些经典组合进行了实证分析，量化了维度变化对组合表现相对于市场的影响。

    In this paper, we develop the theory of functional generation of portfolios in an equity market with changing dimension. By introducing dimensional jumps in the market, as well as jumps in stock capitalization between the dimensional jumps, we construct different types of self-financing stock portfolios (additive, multiplicative, and rank-based) in a very general setting. Our study explains how a dimensional change caused by a listing or delisting event of a stock, and unexpected shocks in the market, affect portfolio return. We also provide empirical analyses of some classical portfolios, quantifying the impact of dimensional change in portfolio performance relative to the market.
    

