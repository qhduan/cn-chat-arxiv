# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Learning Probability Distributions of Intraday Electricity Prices.](http://arxiv.org/abs/2310.02867) | 该论文提出了一种利用机器学习方法对电力日内价格概率进行预测的新方法，该方法通过学习数据中的经验分布选择最佳分布，并利用分布神经网络学习复杂模式，优于现有的基准模型。 |
| [^2] | [Moran's I Lasso for models with spatially correlated data.](http://arxiv.org/abs/2310.02773) | 本文提出了一种新的基于Lasso的估计器，利用Moran统计量来解决特征向量选择问题，该方法在性能和计算速度上有明显优势。 |
| [^3] | [Affirmative Action in India: Restricted Strategy Space, Complex Constraints, and Direct Mechanism Design.](http://arxiv.org/abs/2310.02660) | 印度通过预留制度实施复杂的平权行动计划，包括垂直和水平预留，以解决历史上被边缘化群体的社会经济失衡问题。 |
| [^4] | [On Optimal Set Estimation for Partially Identified Binary Choice Models.](http://arxiv.org/abs/2310.02414) | 本文重新考虑了部分识别模型估计中的最优性概念，并提出了在所有设计中收敛于识别区域的替代方法。 |
| [^5] | [The Price of Empire: Unrest Location and Sovereign Risk in Tsarist Russia.](http://arxiv.org/abs/2309.06885) | 该论文研究了政治动荡和主权风险对于地理辽阔的国家的影响，并发现在帝国边疆地区发生的动荡更容易增加风险。研究结果对于我们理解当前事件有启示，也提醒着我们在维护国家稳定与吸引外国投资方面所面临的挑战。 |
| [^6] | [Mean-field equilibrium price formation with exponential utility.](http://arxiv.org/abs/2304.07108) | 本文研究了多个投资者在初始财富、风险规避参数以及终止时间的随机负债方面存在差异时的均衡价格形成问题，通过一个新的均场反向随机微分方程（BSDE）的解来表征风险股票的均衡风险溢价过程，并证明其清除市场。 |
| [^7] | [Pricing cyber-insurance for systems via maturity models.](http://arxiv.org/abs/2302.04734) | 本篇论文提出了一种使用安全成熟度模型的方法，以评估组织的安全水平并确定网络保险的适当保费。 |
| [^8] | [High Dimensional Generalised Penalised Least Squares.](http://arxiv.org/abs/2207.07055) | 本文提出了GLS Lasso方法用于高维线性模型的推断，支持强混合变量和误差过程的分布和重尾特征，并采用去偏Lasso方法进行感兴趣的参数均匀推断，蒙特卡罗结果表明该方法比传统方法更有效。 |
| [^9] | [Machine Learning Inference on Inequality of Opportunity.](http://arxiv.org/abs/2206.05235) | 通过机器学习在预测结果和计算预测的不平等指数的两个步骤中可能存在偏差，我们提出了一种简单的去偏IOp估计器，并提供了第一个有效的IOp推论理论。我们在欧洲报告了首个无偏的收入IOp度量，发现母亲的教育和父亲的职业是最重要的解释因素。插值估计器对机器学习算法非常敏感，而去偏IOp估计器则具有鲁棒性。 |

# 详细

[^1]: 学习电力日内价格概率分布

    Learning Probability Distributions of Intraday Electricity Prices. (arXiv:2310.02867v1 [econ.GN])

    [http://arxiv.org/abs/2310.02867](http://arxiv.org/abs/2310.02867)

    该论文提出了一种利用机器学习方法对电力日内价格概率进行预测的新方法，该方法通过学习数据中的经验分布选择最佳分布，并利用分布神经网络学习复杂模式，优于现有的基准模型。

    

    我们提出了一种新颖的机器学习方法，用于对小时级电力日内价格进行概率预测。与最近在数据丰富的概率预测方面的进展不同，该方法是非参数的，并从数据中学习到所有可能的经验分布中选择最佳分布。我们提出的模型是一种具有单调调整惩罚的多输出神经网络。这样的分布神经网络可以从数据丰富的环境中学习到电力价格的复杂模式，并且优于最先进的基准模型。

    We propose a novel machine learning approach to probabilistic forecasting of hourly intraday electricity prices. In contrast to recent advances in data-rich probabilistic forecasting that approximate the distributions with some features such as moments, our method is non-parametric and selects the best distribution from all possible empirical distributions learned from the data. The model we propose is a multiple output neural network with a monotonicity adjusting penalty. Such a distributional neural network can learn complex patterns in electricity prices from data-rich environments and it outperforms state-of-the-art benchmarks.
    
[^2]: Moran的I Lasso用于具有空间相关数据的模型

    Moran's I Lasso for models with spatially correlated data. (arXiv:2310.02773v1 [econ.EM])

    [http://arxiv.org/abs/2310.02773](http://arxiv.org/abs/2310.02773)

    本文提出了一种新的基于Lasso的估计器，利用Moran统计量来解决特征向量选择问题，该方法在性能和计算速度上有明显优势。

    

    本文提出了一种基于Lasso的估计器，利用Moran统计量中嵌入的信息来开发一种选择过程，称为Moran的I Lasso (Mi-Lasso)，以解决特征向量空间滤波（ESF）特征向量选择问题。ESF使用空间加权矩阵的子集来有效地考虑经典线性回归框架中遗漏的横截面相关项，因此不需要研究者明确指定底层结构模型的空间部分。我们推导了性能界限，并展示了一致性特征向量选择的必要条件。该估计器的主要优点是直观、理论基础扎实，而且比基于交叉验证或任何提出的前向逐步过程的Lasso方法快得多。我们的主要模拟结果表明，所提出的选择过程在有限样本中表现良好。与现有的选择过程相比，我们发现Mi-Lasso在选择准确性和计算速度方面表现优异。

    This paper proposes a Lasso-based estimator which uses information embedded in the Moran statistic to develop a selection procedure called Moran's I Lasso (Mi-Lasso) to solve the Eigenvector Spatial Filtering (ESF) eigenvector selection problem. ESF uses a subset of eigenvectors from a spatial weights matrix to efficiently account for any omitted cross-sectional correlation terms in a classical linear regression framework, thus does not require the researcher to explicitly specify the spatial part of the underlying structural model. We derive performance bounds and show the necessary conditions for consistent eigenvector selection. The key advantages of the proposed estimator are that it is intuitive, theoretically grounded, and substantially faster than Lasso based on cross-validation or any proposed forward stepwise procedure. Our main simulation results show the proposed selection procedure performs well in finite samples. Compared to existing selection procedures, we find Mi-Lasso 
    
[^3]: 印度的平权行动：限制策略空间、复杂约束和直接机制设计

    Affirmative Action in India: Restricted Strategy Space, Complex Constraints, and Direct Mechanism Design. (arXiv:2310.02660v1 [econ.TH])

    [http://arxiv.org/abs/2310.02660](http://arxiv.org/abs/2310.02660)

    印度通过预留制度实施复杂的平权行动计划，包括垂直和水平预留，以解决历史上被边缘化群体的社会经济失衡问题。

    

    自1950年以来，印度通过一个精心设计的预留制度，实施了一项复杂的平权行动计划。该制度融合了垂直和水平的预留，以解决历史上被边缘化群体的社会经济失衡问题。垂直预留为公共教育机构和政府就业岗位指定了特定的配额，供应安排给“计划种姓”、“计划部落”、“其他落后阶层”和“经济弱势群体”。同时，在每个垂直类别内，还采用了水平预留，为其他子群体，如妇女和残疾人，分配职位。在教育招生中，法律框架建议未被垂直类别占满的职位恢复为非预留状态。此外，我们还记录到，来自垂直预留类别的个人对于机构-垂直类别职位对有更复杂的偏好，尽管当局要求他们按优先顺序申请这些职位。

    Since 1950, India has instituted an intricate affirmative action program through a meticulously designed reservation system. This system incorporates vertical and horizontal reservations to address historically marginalized groups' socioeconomic imbalances. Vertical reservations designate specific quotas of available positions in publicly funded educational institutions and government employment for Scheduled Castes, Scheduled Tribes, Other Backward Classes, and Economically Weaker Sections. Concurrently, horizontal reservations are employed within each vertical category to allocate positions for additional subgroups, such as women and individuals with disabilities. In educational admissions, the legal framework recommended that unfilled positions reserved for the OBC category revert to unreserved status. Moreover, we document that individuals from vertically reserved categories have more complicated preferences over institution-vertical category position pairs, even though authorities
    
[^4]: 关于部分识别二项选择模型的最优集估计

    On Optimal Set Estimation for Partially Identified Binary Choice Models. (arXiv:2310.02414v1 [econ.EM])

    [http://arxiv.org/abs/2310.02414](http://arxiv.org/abs/2310.02414)

    本文重新考虑了部分识别模型估计中的最优性概念，并提出了在所有设计中收敛于识别区域的替代方法。

    

    在本文中，我们重新考虑了部分识别模型估计中的最优性概念。我们以半参数二项选择模型为例，以离散协变量作为示例，说明了一般问题。该模型在一定程度上是部分识别的，例如Bierens和Hartog（1988）所示。通过实施Manski（1975）提出的最大分数程序，可以构建模型中回归系数的集合估计。对于许多设计，该方法收敛于这些参数的识别集，因此在某种意义上是最优的。但是，正如Komarova（2013）所示，对于其他情况，最大分数目标函数给出了识别集的边界区域。这激发了寻求其他优化方法的动力，这些方法在所有设计中都收敛于识别区域，并且我们提出并比较了这些方法。一个是Hodges类型的估计器，将最大分数估计器与现有程序相结合。第二个是两步法

    In this paper we reconsider the notion of optimality in estimation of partially identified models. We illustrate the general problem in the context of a semiparametric binary choice model with discrete covariates as an example of a model which is partially identified as shown in, e.g. Bierens and Hartog (1988). A set estimator for the regression coefficients in the model can be constructed by implementing the Maximum Score procedure proposed by Manski (1975). For many designs this procedure converges to the identified set for these parameters, and so in one sense is optimal. But as shown in Komarova (2013) for other cases the Maximum Score objective function gives an outer region of the identified set. This motivates alternative methods that are optimal in one sense that they converge to the identified region in all designs, and we propose and compare such procedures. One is a Hodges type estimator combining the Maximum Score estimator with existing procedures. A second is a two step e
    
[^5]: 帝国的代价：沙俄动荡地点与主权风险

    The Price of Empire: Unrest Location and Sovereign Risk in Tsarist Russia. (arXiv:2309.06885v1 [econ.GN])

    [http://arxiv.org/abs/2309.06885](http://arxiv.org/abs/2309.06885)

    该论文研究了政治动荡和主权风险对于地理辽阔的国家的影响，并发现在帝国边疆地区发生的动荡更容易增加风险。研究结果对于我们理解当前事件有启示，也提醒着我们在维护国家稳定与吸引外国投资方面所面临的挑战。

    

    关于政治动荡和主权风险的研究忽视了动荡地点对于地理辽阔的国家主权风险的影响及其机制。在直观上，首都或附近的政治暴力似乎直接威胁到国家偿还债务的能力。然而，远离暴力地点可能会更加严重地影响政府，与抑制叛乱所带来的长期成本有关。我们利用沙俄的案例来评估俄罗斯国土内发生动荡与帝国边疆地区发生动荡时风险效应的差异。我们分析了1820年至1914年间沙俄帝国各地的动荡事件，发现动荡对帝国边疆地区的风险影响更大。与当前事件相呼应，我们发现乌克兰的动荡使风险增加最多。帝国的代价包括了向镇压动荡和获得外国投资者信任的同时维持力量投射的高额成本。

    Research on politically motivated unrest and sovereign risk overlooks whether and how unrest location matters for sovereign risk in geographically extensive states. Intuitively, political violence in the capital or nearby would seem to directly threaten the state's ability to pay its debts. However, it is possible that the effect on a government could be more pronounced the farther away the violence is, connected to the longer-term costs of suppressing rebellion. We use Tsarist Russia to assess these differences in risk effects when unrest occurs in Russian homeland territories versus more remote imperial territories. Our analysis of unrest events across the Russian imperium from 1820 to 1914 suggests that unrest increases risk more in imperial territories. Echoing current events, we find that unrest in Ukraine increases risk most. The price of empire included higher costs in projecting force to repress unrest and retain the confidence of the foreign investors financing those costs.
    
[^6]: 带指数效用函数的均值场均衡价格形成

    Mean-field equilibrium price formation with exponential utility. (arXiv:2304.07108v1 [q-fin.MF])

    [http://arxiv.org/abs/2304.07108](http://arxiv.org/abs/2304.07108)

    本文研究了多个投资者在初始财富、风险规避参数以及终止时间的随机负债方面存在差异时的均衡价格形成问题，通过一个新的均场反向随机微分方程（BSDE）的解来表征风险股票的均衡风险溢价过程，并证明其清除市场。

    

    本文研究了多位投资者在初始财富、风险规避参数以及终止时间的随机负债方面存在差异时的均衡价格形成问题。我们通过一个新的均场反向随机微分方程（BSDE）的解来表征风险股票的均衡风险溢价过程，其特征是驱动程序在随机积分和条件期望上都具有二次增长。我们证明了在多个条件下均场BSDE存在解，并且表明随着人口规模的增大，结果风险溢价进程实际上会清除市场。

    In this paper, we study a problem of equilibrium price formation among many investors with exponential utility. The investors are heterogeneous in their initial wealth, risk-averseness parameter, as well as stochastic liability at the terminal time. We characterize the equilibrium risk-premium process of the risky stocks in terms of the solution to a novel mean-field backward stochastic differential equation (BSDE), whose driver has quadratic growth both in the stochastic integrands and in their conditional expectations. We prove the existence of a solution to the mean-field BSDE under several conditions and show that the resultant risk-premium process actually clears the market in the large population limit.
    
[^7]: 基于成熟度模型的信息系统网络保险定价

    Pricing cyber-insurance for systems via maturity models. (arXiv:2302.04734v2 [econ.GN] UPDATED)

    [http://arxiv.org/abs/2302.04734](http://arxiv.org/abs/2302.04734)

    本篇论文提出了一种使用安全成熟度模型的方法，以评估组织的安全水平并确定网络保险的适当保费。

    

    对于与信息技术系统相关的风险进行保险定价提出了一个综合的建议，结合运营管理、安全和经济学，提出了一个社会经济模型。该模型包括实体关系图、安全成熟度模型和经济模型，解决了一个长期以来的研究难题，即如何在设计和定价网络保险政策时捕捉组织结构。文中提出了一个新的挑战，即网络保险的数据历史有限，不能直接应用于其它险种，因此提出一个安全成熟度模型，以评估组织的安全水平并确定相应的保险费用。

    Pricing insurance for risks associated with information technology systems presents a complex modelling challenge, combining the disciplines of operations management, security, and economics. This work proposes a socioeconomic model for cyber-insurance decisions compromised of entity relationship diagrams, security maturity models, and economic models, addressing a long-standing research challenge of capturing organizational structure in the design and pricing of cyber-insurance policies. Insurance pricing is usually informed by the long experience insurance companies have of the magnitude and frequency of losses that arise in organizations based on their size, industry sector, and location. Consequently, their calculations of premia will start from a baseline determined by these considerations. A unique challenge of cyber-insurance is that data history is limited and not necessarily informative of future loss risk meaning that established actuarial methodology for other lines of insur
    
[^8]: 高维广义惩罚最小二乘的推断方法

    High Dimensional Generalised Penalised Least Squares. (arXiv:2207.07055v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2207.07055](http://arxiv.org/abs/2207.07055)

    本文提出了GLS Lasso方法用于高维线性模型的推断，支持强混合变量和误差过程的分布和重尾特征，并采用去偏Lasso方法进行感兴趣的参数均匀推断，蒙特卡罗结果表明该方法比传统方法更有效。

    

    本文针对高维线性模型中的串行相关误差开展推断研究。我们在变量和误差过程强混合的假设下探讨了Lasso方法，并允许它们的分布尾部更加厚重。虽然在这种情况下Lasso估计量表现不佳，但我们通过GLS Lasso估计感兴趣的参数，并在更一般的条件下扩展了Lasso的渐近特性。我们的理论结果表明，对于平稳相关过程的非渐近界限更严格，而Lasso在一般条件下的速度随着$T, p \to \infty$而变慢。此外，我们采用去偏Lasso方法，对感兴趣的参数进行均匀推断。蒙特卡罗结果支持所提出的估计量，因为它相对传统方法具有显着的效率提高。

    In this paper we develop inference for high dimensional linear models, with serially correlated errors. We examine Lasso under the assumption of strong mixing in the covariates and error process, allowing for fatter tails in their distribution. While the Lasso estimator performs poorly under such circumstances, we estimate via GLS Lasso the parameters of interest and extend the asymptotic properties of the Lasso under more general conditions. Our theoretical results indicate that the non-asymptotic bounds for stationary dependent processes are sharper, while the rate of Lasso under general conditions appears slower as $T,p\to \infty$. Further we employ the debiased Lasso to perform inference uniformly on the parameters of interest. Monte Carlo results support the proposed estimator, as it has significant efficiency gains over traditional methods.
    
[^9]: 机器学习在机会不平等上的推论

    Machine Learning Inference on Inequality of Opportunity. (arXiv:2206.05235v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2206.05235](http://arxiv.org/abs/2206.05235)

    通过机器学习在预测结果和计算预测的不平等指数的两个步骤中可能存在偏差，我们提出了一种简单的去偏IOp估计器，并提供了第一个有效的IOp推论理论。我们在欧洲报告了首个无偏的收入IOp度量，发现母亲的教育和父亲的职业是最重要的解释因素。插值估计器对机器学习算法非常敏感，而去偏IOp估计器则具有鲁棒性。

    

    机会平等已经成为分配公正的重要理念。实证上，机会不平等(IOp)通过两个步骤进行测量：首先，根据个人情况预测一个结果（如收入）；然后，计算预测的不平等指数（如基尼系数）。机器学习方法在第一步非常有用。然而，在IOp的第二步中，它们可能会导致相当大的偏差，因为偏差-方差权衡允许偏差渗入。我们提出了一个简单的，抵消了这种机器学习偏差的IOp估计器，并提供了第一个有效的IOp推论理论。我们在模拟中展示了改进的性能，并报道了欧洲的首个无偏收入IOp度量。母亲的教育和父亲的职业是最重要的解释因素。插值估计器对机器学习算法非常敏感，而抵消偏差的IOp估计器则具有鲁棒性。这些结果还扩展到了一般的U-统计设置。

    Equality of opportunity has emerged as an important ideal of distributive justice. Empirically, Inequality of Opportunity (IOp) is measured in two steps: first, an outcome (e.g., income) is predicted given individual circumstances; and second, an inequality index (e.g., Gini) of the predictions is computed. Machine Learning (ML) methods are tremendously useful in the first step. However, they can cause sizable biases in IOp since the bias-variance trade-off allows the bias to creep in the second step. We propose a simple debiased IOp estimator robust to such ML biases and provide the first valid inferential theory for IOp. We demonstrate improved performance in simulations and report the first unbiased measures of income IOp in Europe. Mother's education and father's occupation are the circumstances that explain the most. Plug-in estimators are very sensitive to the ML algorithm, while debiased IOp estimators are robust. These results are extended to a general U-statistics setting.
    

