# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reparametrization and the Semiparametric Bernstein-von-Mises Theorem.](http://arxiv.org/abs/2306.03816) | 本文提出了一种参数化形式，该形式可以通过生成Neyman正交矩条件来降低对干扰参数的敏感度，从而可以用于去偏贝叶斯推断中的后验分布，同时在参数速率下对低维参数进行真实值的收缩，并在半参数效率界的方差下进行渐近正态分布。 |
| [^2] | [Uniform Inference for Cointegrated Vector Autoregressive Processes.](http://arxiv.org/abs/2306.03632) | 本文提出了一种解决协整向量自回归过程的均匀有效推断问题的方法，并应用于两个具体案例验证其有效性。 |
| [^3] | [Forecasting the Performance of US Stock Market Indices During COVID-19: RF vs LSTM.](http://arxiv.org/abs/2306.03620) | 该研究使用了两个机器学习模型（随机森林和LSTM）来预测COVID-19期间美国两个主要股票市场指数的表现，这些模型应用历史数据进行训练和预测，并使用交叉验证和一系列技术进行优化。 |
| [^4] | [Robust inference for the treatment effect variance in experiments using machine learning.](http://arxiv.org/abs/2306.03363) | 本论文提出了关于可观测处理效应方差的有效置信区间，解决了传统方法在可观测处理效应方差为零时产生的错误覆盖范围的问题，并发现了VCATE与决定谁应受治疗的问题之间的新联系。 |
| [^5] | [Quantiled conditional variance, skewness, and kurtosis by Cornish-Fisher expansion.](http://arxiv.org/abs/2302.06799) | 该论文提出了一种使用Cornish-Fisher扩展来估计条件方差、偏度和峰度的方法——QCMs，该方法不需要先前估计条件均值，并且在模拟研究中表现良好。 |
| [^6] | [In Search of Insights, Not Magic Bullets: Towards Demystification of the Model Selection Dilemma in Heterogeneous Treatment Effect Estimation.](http://arxiv.org/abs/2302.02923) | 本文研究在具有高风险应用的个性化处理效应估计中，不同模型选择标准的优点和缺点，并提出未来研究方向。 |
| [^7] | [Uniform Convergence Results for the Local Linear Regression Estimation of the Conditional Distribution.](http://arxiv.org/abs/2112.08546) | 本文研究了条件分布函数的局部线性回归估计，并推导了一致收敛性结果。这些结果对于条件分布估计是半参数估计的第一阶段时特别有用。 |
| [^8] | [Some Impossibility Results for Inference With Cluster Dependence with Large Clusters.](http://arxiv.org/abs/2109.03971) | 本文研究了在具有群体依赖结构的情况下进行推断的不可能性结果，发现当没有关于观测值依赖结构的知识时，不能一致地区分平均值，而至少有两个大聚类是进行一致的 $\sqrt{n}$-区分平均值的充分条件；同时，我们提供了长期方差一致估计的必要和充分条件，结果表明，当至少有一个大聚类时，长期方差不能一致地估计。 |
| [^9] | [Measuring Cognitive Abilities in the Wild: Validating a Population-Scale Game-Based Cognitive Assessment.](http://arxiv.org/abs/2009.05274) | 该论文提出了一种基于游戏的认知评估方法Skill Lab，利用一个流行的公民科学平台进行全面验证，在真实环境中测量了广泛的认知能力，可以同时预测8种认知能力。 |
| [^10] | [Instrument Validity for Heterogeneous Causal Effects.](http://arxiv.org/abs/2009.01995) | 本论文提出了一个通用框架和非参数检验方法来检验异质因果效应模型中工具的有效性，可以应用于治疗多值有序或无序的情况，并且可以帮助检测无效工具以避免因果效应不合理的结果。 |
| [^11] | [Orthogonal Statistical Learning.](http://arxiv.org/abs/1901.09036) | 本文提出了一个两阶段样本拆分的元算法，该算法能够在评估总体风险时考虑干扰参数，并且实现的超额风险界的影响为二次。 |

# 详细

[^1]: 重参数化与半参数Bernstein-von-Mises定理

    Reparametrization and the Semiparametric Bernstein-von-Mises Theorem. (arXiv:2306.03816v1 [math.ST])

    [http://arxiv.org/abs/2306.03816](http://arxiv.org/abs/2306.03816)

    本文提出了一种参数化形式，该形式可以通过生成Neyman正交矩条件来降低对干扰参数的敏感度，从而可以用于去偏贝叶斯推断中的后验分布，同时在参数速率下对低维参数进行真实值的收缩，并在半参数效率界的方差下进行渐近正态分布。

    

    本文考虑了部分线性模型的贝叶斯推断。我们的方法利用了回归函数的一个参数化形式，该形式专门用于估计所关心的低维参数。参数化的关键特性是生成了一个Neyman正交矩条件，这意味着对干扰参数的估计低维参数不太敏感。我们的大样本分析支持了这种说法。特别地，我们推导出充分的条件，使得低维参数的后验在参数速率下对真实值收缩，并且在半参数效率界的方差下渐近地正态分布。这些条件相对于回归模型的原始参数化允许更大类的干扰参数。总的来说，我们得出结论，一个嵌入了Neyman正交性的参数化方法可以成为半参数推断中的一个有用工具，以去偏后验分布。

    This paper considers Bayesian inference for the partially linear model. Our approach exploits a parametrization of the regression function that is tailored toward estimating a low-dimensional parameter of interest. The key property of the parametrization is that it generates a Neyman orthogonal moment condition meaning that the low-dimensional parameter is less sensitive to the estimation of nuisance parameters. Our large sample analysis supports this claim. In particular, we derive sufficient conditions under which the posterior for the low-dimensional parameter contracts around the truth at the parametric rate and is asymptotically normal with a variance that coincides with the semiparametric efficiency bound. These conditions allow for a larger class of nuisance parameters relative to the original parametrization of the regression model. Overall, we conclude that a parametrization that embeds Neyman orthogonality can be a useful device for debiasing posterior distributions in semipa
    
[^2]: 协整向量自回归过程的均匀推断

    Uniform Inference for Cointegrated Vector Autoregressive Processes. (arXiv:2306.03632v1 [math.ST])

    [http://arxiv.org/abs/2306.03632](http://arxiv.org/abs/2306.03632)

    本文提出了一种解决协整向量自回归过程的均匀有效推断问题的方法，并应用于两个具体案例验证其有效性。

    

    协整向量自回归过程的均匀有效推断因最小二乘估计的渐近分布中出现的某些不连续而一直难以证实。我们展示了如何将单变量情况下的渐近结果扩展到多维，并基于这些结果进行推断。此外，我们展示了[20]提出的新型工具变量程序（IVX）如何生成整个自回归矩阵的均匀有效置信区间。我们将这些结果应用于两个具体的例子，并在模拟实验中验证理论结果并研究有限样本性质。

    Uniformly valid inference for cointegrated vector autoregressive processes has so far proven difficult due to certain discontinuities arising in the asymptotic distribution of the least squares estimator. We show how asymptotic results from the univariate case can be extended to multiple dimensions and how inference can be based on these results. Furthermore, we show that the novel instrumental variable procedure proposed by [20] (IVX) yields uniformly valid confidence regions for the entire autoregressive matrix. The results are applied to two specific examples for which we verify the theoretical findings and investigate finite sample properties in simulation experiments.
    
[^3]: 预测COVID-19期间US股票市场指数表现：RF vs LSTM

    Forecasting the Performance of US Stock Market Indices During COVID-19: RF vs LSTM. (arXiv:2306.03620v1 [econ.EM])

    [http://arxiv.org/abs/2306.03620](http://arxiv.org/abs/2306.03620)

    该研究使用了两个机器学习模型（随机森林和LSTM）来预测COVID-19期间美国两个主要股票市场指数的表现，这些模型应用历史数据进行训练和预测，并使用交叉验证和一系列技术进行优化。

    

    美国股市在经历了2007-2009年的衰退后出现了不稳定情况。COVID-19对美国股票交易员和投资者构成了重大挑战。交易员和投资者应该跟上股市的节奏。使用考虑到大流行病影响的预测模型，可以缓解风险并提高利润。考虑到大流行病后的衰退，使用了两个机器学习模型，包括随机森林和LSTM来预测两个主要的美国股票市场指数。使用历史价格数据开发机器学习模型并预测指数回报。为了评估训练期间的模型性能，使用交叉验证。此外，超参数优化，正则化（如投放和权重衰减）和预处理可以改善机器学习技术的性能。使用高精度的机器学习技术，交易员和投资者可以预测股市行为。

    The US stock market experienced instability following the recession (2007-2009). COVID-19 poses a significant challenge to US stock traders and investors. Traders and investors should keep up with the stock market. This is to mitigate risks and improve profits by using forecasting models that account for the effects of the pandemic. With consideration of the COVID-19 pandemic after the recession, two machine learning models, including Random Forest and LSTM are used to forecast two major US stock market indices. Data on historical prices after the big recession is used for developing machine learning models and forecasting index returns. To evaluate the model performance during training, cross-validation is used. Additionally, hyperparameter optimizing, regularization, such as dropouts and weight decays, and preprocessing improve the performances of Machine Learning techniques. Using high-accuracy machine learning techniques, traders and investors can forecast stock market behavior, st
    
[^4]: 利用机器学习进行实验的处理效应方差的鲁棒性推断

    Robust inference for the treatment effect variance in experiments using machine learning. (arXiv:2306.03363v1 [econ.EM])

    [http://arxiv.org/abs/2306.03363](http://arxiv.org/abs/2306.03363)

    本论文提出了关于可观测处理效应方差的有效置信区间，解决了传统方法在可观测处理效应方差为零时产生的错误覆盖范围的问题，并发现了VCATE与决定谁应受治疗的问题之间的新联系。

    

    实验者通常收集基线数据来研究异质性。本文提出了第一个关于可观测处理效应方差的有效置信区间。传统方法在可观测处理效应方差为零时，会产生错误的覆盖范围。因此，即使不存在异质性，实际操作者也容易检测到。当边界处于局部退化状态时，所有高效估计器都具有局部退化的影响函数，并且可能不具有渐近正态性。本文解决了具有预测第一阶段的广泛类多步估计器的问题。我的置信区间考虑了极限分布中的高阶项，并且计算速度很快。我还发现了VCATE与决定谁应受治疗的问题之间的新联系。治疗目标的收益(急剧)受到VCATE平方根的一半的限制。最后，我通过模拟和重新分析马拉维的实验，证明了优异的性能。

    Experimenters often collect baseline data to study heterogeneity. I propose the first valid confidence intervals for the VCATE, the treatment effect variance explained by observables. Conventional approaches yield incorrect coverage when the VCATE is zero. As a result, practitioners could be prone to detect heterogeneity even when none exists. The reason why coverage worsens at the boundary is that all efficient estimators have a locally-degenerate influence function and may not be asymptotically normal. I solve the problem for a broad class of multistep estimators with a predictive first stage. My confidence intervals account for higher-order terms in the limiting distribution and are fast to compute. I also find new connections between the VCATE and the problem of deciding whom to treat. The gains of targeting treatment are (sharply) bounded by half the square root of the VCATE. Finally, I document excellent performance in simulation and reanalyze an experiment from Malawi.
    
[^5]: Cornish-Fisher扩展的分位条件方差、偏度和峰度

    Quantiled conditional variance, skewness, and kurtosis by Cornish-Fisher expansion. (arXiv:2302.06799v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2302.06799](http://arxiv.org/abs/2302.06799)

    该论文提出了一种使用Cornish-Fisher扩展来估计条件方差、偏度和峰度的方法——QCMs，该方法不需要先前估计条件均值，并且在模拟研究中表现良好。

    

    条件方差、偏度和峰度在时间序列分析中起着核心作用。这篇论文提出了一种利用Cornish-Fisher扩展来估计这三个条件矩的新方法——分位条件矩（QCMs）。该方法通过基于$n$个不同的估计条件分位数构建线性回归模型来计算QCMs，然后使用此回归模型的最小二乘估计量简单而同时地计算QCMs，而无需先前估计条件均值。在一定条件下，QCMs被证明是一致的，收敛速度为$n^{-1/2}$。模拟研究表明，QCMs在Cornish-Fisher扩展误差和分位数估计误差的不同情况下表现良好。

    The conditional variance, skewness, and kurtosis play a central role in time series analysis. These three conditional moments (CMs) are often studied by some parametric models but with two big issues: the risk of model mis-specification and the instability of model estimation. To avoid the above two issues, this paper proposes a novel method to estimate these three CMs by the so-called quantiled CMs (QCMs). The QCM method first adopts the idea of Cornish-Fisher expansion to construct a linear regression model, based on $n$ different estimated conditional quantiles. Next, it computes the QCMs simply and simultaneously by using the ordinary least squares estimator of this regression model, without any prior estimation of the conditional mean. Under certain conditions, the QCMs are shown to be consistent with the convergence rate $n^{-1/2}$. Simulation studies indicate that the QCMs perform well under different scenarios of Cornish-Fisher expansion errors and quantile estimation errors. I
    
[^6]: 不是神奇药丸，而是洞察力之搜寻：消除异质性处理效应估计中的模型选择困境

    In Search of Insights, Not Magic Bullets: Towards Demystification of the Model Selection Dilemma in Heterogeneous Treatment Effect Estimation. (arXiv:2302.02923v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.02923](http://arxiv.org/abs/2302.02923)

    本文研究在具有高风险应用的个性化处理效应估计中，不同模型选择标准的优点和缺点，并提出未来研究方向。

    

    个性化处理效应估计在高风险应用中经常备受关注，因此，在实践中部署估计这种效应的模型之前，需要确信已经选择了最好的机器学习工具箱中的候选模型。不幸的是，由于实践中缺乏反事实信息，通常无法依靠标准验证指标完成此任务，导致了处理效应估计文献中已知的模型选择困境。虽然最近已经研究了一些解决方案，但对不同模型选择标准的优缺点的系统理解仍然缺乏。因此，在本文中，我们并没有试图宣布全局“胜者”，而是对不同选择标准的成功和失败模式进行了实证研究。我们强调选择策略，候选估计量和用于比较它们的数据之间存在复杂的相互作用，并提出了未来研究的方向。

    Personalized treatment effect estimates are often of interest in high-stakes applications -- thus, before deploying a model estimating such effects in practice, one needs to be sure that the best candidate from the ever-growing machine learning toolbox for this task was chosen. Unfortunately, due to the absence of counterfactual information in practice, it is usually not possible to rely on standard validation metrics for doing so, leading to a well-known model selection dilemma in the treatment effect estimation literature. While some solutions have recently been investigated, systematic understanding of the strengths and weaknesses of different model selection criteria is still lacking. In this paper, instead of attempting to declare a global `winner', we therefore empirically investigate success- and failure modes of different selection criteria. We highlight that there is a complex interplay between selection strategies, candidate estimators and the data used for comparing them, an
    
[^7]: 条件分布的局部线性回归估计的一致性收敛性结果

    Uniform Convergence Results for the Local Linear Regression Estimation of the Conditional Distribution. (arXiv:2112.08546v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2112.08546](http://arxiv.org/abs/2112.08546)

    本文研究了条件分布函数的局部线性回归估计，并推导了一致收敛性结果。这些结果对于条件分布估计是半参数估计的第一阶段时特别有用。

    

    本文研究了条件分布函数 $F(y|x)$ 的局部线性回归估计。我们推导了三个一致性收敛性结果：一致偏差展开、一致收敛速率和一致渐近线性表示。上述结果中的一致性是相对于 $x$ 和 $y$ 的，因此在局部多项式回归的文献中尚未涉及。这种一致性收敛性结果在条件分布估计是半参数估计的第一阶段时特别有用。我们通过两个例子展示了这些一致性结果的实用性：$y$ 的随机等连续性条件和积分条件分布函数的估计。

    This paper examines the local linear regression (LLR) estimate of the conditional distribution function $F(y|x)$. We derive three uniform convergence results: the uniform bias expansion, the uniform convergence rate, and the uniform asymptotic linear representation. The uniformity in the above results is with respect to both $x$ and $y$ and therefore has not previously been addressed in the literature on local polynomial regression. Such uniform convergence results are especially useful when the conditional distribution estimator is the first stage of a semiparametric estimator. We demonstrate the usefulness of these uniform results with two examples: the stochastic equicontinuity condition in $y$, and the estimation of the integrated conditional distribution function.
    
[^8]: 对具有大聚类的群体依赖结构进行推断的一些不可能性结果

    Some Impossibility Results for Inference With Cluster Dependence with Large Clusters. (arXiv:2109.03971v4 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2109.03971](http://arxiv.org/abs/2109.03971)

    本文研究了在具有群体依赖结构的情况下进行推断的不可能性结果，发现当没有关于观测值依赖结构的知识时，不能一致地区分平均值，而至少有两个大聚类是进行一致的 $\sqrt{n}$-区分平均值的充分条件；同时，我们提供了长期方差一致估计的必要和充分条件，结果表明，当至少有一个大聚类时，长期方差不能一致地估计。

    

    本文关注于观测值具有群体依赖结构的情况，并提出了两个主要的不可能性结果。首先，我们证明了当只有一个大聚类时，即研究人员没有观测值依赖结构的任何知识时，不能一致地区分平均值。当聚类内的观测值满足均匀中心极限定理时，我们还证明了一个进行 $\sqrt{n}$-区分平均值的一致的充分条件是至少有两个大聚类。这个结果显示了当我们缺乏有关观测值依赖结构信息时进行推断的一些限制。我们的第二个结果为长期方差一致估计提供了必要和充分条件。我们的结果意味着，当至少有一个大聚类时，长期方差不能一致地估计。

    This paper focuses on a setting with observations having a cluster dependence structure and presents two main impossibility results. First, we show that when there is only one large cluster, i.e., the researcher does not have any knowledge on the dependence structure of the observations, it is not possible to consistently discriminate the mean. When within-cluster observations satisfy the uniform central limit theorem, we also show that a sufficient condition for consistent $\sqrt{n}$-discrimination of the mean is that we have at least two large clusters. This result shows some limitations for inference when we lack information on the dependence structure of observations. Our second result provides a necessary and sufficient condition for the cluster structure that the long run variance is consistently estimable. Our result implies that when there is at least one large cluster, the long run variance is not consistently estimable.
    
[^9]: 在真实环境中测量认知能力：验证一种面向人群的基于游戏的认知评估方法

    Measuring Cognitive Abilities in the Wild: Validating a Population-Scale Game-Based Cognitive Assessment. (arXiv:2009.05274v5 [physics.soc-ph] UPDATED)

    [http://arxiv.org/abs/2009.05274](http://arxiv.org/abs/2009.05274)

    该论文提出了一种基于游戏的认知评估方法Skill Lab，利用一个流行的公民科学平台进行全面验证，在真实环境中测量了广泛的认知能力，可以同时预测8种认知能力。

    

    个体认知表型的快速测量具有革命性的潜力，可在个性化学习、就业实践和精准精神病学等广泛领域得到应用。为了超越传统实验室实验所带来的限制，人们正在努力增加生态效度和参与者多样性，以捕捉普通人群中认知能力和行为的个体差异的全部范围。基于此，我们开发了Skill Lab，一种新型的基于游戏的工具，它在提供引人入胜的故事情节的同时评估广泛的认知能力。Skill Lab由六个小游戏和14个已知的认知能力任务组成。利用一个流行的公民科学平台（N = 10725），我们在真实环境中进行了一项全面的基于游戏的认知评估。基于游戏和验证任务的数据，我们构建了可靠的模型，同时预测八种认知能力。

    Rapid individual cognitive phenotyping holds the potential to revolutionize domains as wide-ranging as personalized learning, employment practices, and precision psychiatry. Going beyond limitations imposed by traditional lab-based experiments, new efforts have been underway towards greater ecological validity and participant diversity to capture the full range of individual differences in cognitive abilities and behaviors across the general population. Building on this, we developed Skill Lab, a novel game-based tool that simultaneously assesses a broad suite of cognitive abilities while providing an engaging narrative. Skill Lab consists of six mini-games as well as 14 established cognitive ability tasks. Using a popular citizen science platform (N = 10725), we conducted a comprehensive validation in the wild of a game-based cognitive assessment suite. Based on the game and validation task data, we constructed reliable models to simultaneously predict eight cognitive abilities based 
    
[^10]: 异质因果效应的工具有效性检验

    Instrument Validity for Heterogeneous Causal Effects. (arXiv:2009.01995v5 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2009.01995](http://arxiv.org/abs/2009.01995)

    本论文提出了一个通用框架和非参数检验方法来检验异质因果效应模型中工具的有效性，可以应用于治疗多值有序或无序的情况，并且可以帮助检测无效工具以避免因果效应不合理的结果。

    

    本文提出了一个通用框架，用于检验异质因果效应模型中的工具有效性。这个推广包括了治疗可以是多值有序或无序的情况。基于一系列可检验的推论，我们提出了一个非参数检验，被证明具有渐进的尺寸控制和一致性。与文献中的检验相比，我们的检验可以在更普遍的情况下应用，并且可能实现了功率提升。检验过程中的工具无效有助于检测可能导致因果效应不合理的无效工具。通过模拟提供了测试在有限样本上表现良好的证据。我们重新审视了对于学校回报率的实证研究，以展示所提出的检验在实践中的应用。同时，我们提供了一个扩展的连续映射定理和扩展的 delta 方法，这可能是独立的感兴趣的研究内容，来建立测试统计量的渐近分布。

    This paper provides a general framework for testing instrument validity in heterogeneous causal effect models. The generalization includes the cases where the treatment can be multivalued ordered or unordered. Based on a series of testable implications, we propose a nonparametric test which is proved to be asymptotically size controlled and consistent. Compared to the tests in the literature, our test can be applied in more general settings and may achieve power improvement. Refutation of instrument validity by the test helps detect invalid instruments that may yield implausible results on causal effects. Evidence that the test performs well on finite samples is provided via simulations. We revisit the empirical study on return to schooling to demonstrate application of the proposed test in practice. An extended continuous mapping theorem and an extended delta method, which may be of independent interest, are provided to establish the asymptotic distribution of the test statistic under
    
[^11]: 正交统计学习

    Orthogonal Statistical Learning. (arXiv:1901.09036v4 [math.ST] UPDATED)

    [http://arxiv.org/abs/1901.09036](http://arxiv.org/abs/1901.09036)

    本文提出了一个两阶段样本拆分的元算法，该算法能够在评估总体风险时考虑干扰参数，并且实现的超额风险界的影响为二次。

    

    我们在一个统计学习的设置下提供了关于非渐近超额风险保证，其中目标参数所评估的总体风险取决于必须从数据中估计的未知干扰参数。我们分析了一个两阶段样本拆分的元算法，该算法将任意估计目标参数和干扰参数的算法作为输入。我们表明，如果总体风险满足一个称为Neyman正交性的条件，则干扰估计误差对元算法实现的超额风险界的影响为二次。我们的定理不关心用于目标和干扰的特定算法，只做出了有关它们各自性能的假设。这样，就可以利用现有机器学习的大量结果，为带有干扰组成的学习提供新的保证。此外，通过关注超额风险而不是参数估计，我们可以提供一个弱化的速率。

    We provide non-asymptotic excess risk guarantees for statistical learning in a setting where the population risk with respect to which we evaluate the target parameter depends on an unknown nuisance parameter that must be estimated from data. We analyze a two-stage sample splitting meta-algorithm that takes as input arbitrary estimation algorithms for the target parameter and nuisance parameter. We show that if the population risk satisfies a condition called Neyman orthogonality, the impact of the nuisance estimation error on the excess risk bound achieved by the meta-algorithm is of second order. Our theorem is agnostic to the particular algorithms used for the target and nuisance and only makes an assumption on their individual performance. This enables the use of a plethora of existing results from machine learning to give new guarantees for learning with a nuisance component. Moreover, by focusing on excess risk rather than parameter estimation, we can provide rates under weaker a
    

