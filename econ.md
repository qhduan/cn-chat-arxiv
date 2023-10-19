# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Survey calibration for causal inference: a simple method to balance covariate distributions.](http://arxiv.org/abs/2310.11969) | 本文提出一种简单的方法用于平衡因果推断中观察性研究中的协变量分布，通过修改熵平衡方法和协变量平衡倾向评分方法实现对处理组和对照组的分布平衡。 |
| [^2] | [Machine Learning for Staggered Difference-in-Differences and Dynamic Treatment Effect Heterogeneity.](http://arxiv.org/abs/2310.11962) | 本文提出了一种机器学习差异-差异（MLDID）方法，结合了两种非参数差异-差异方法，用于研究错位采用环境中的治疗效果异质性。我们通过模拟实验证明MLDID能够准确识别出治疗效果异质性的预测因素，并利用该方法评估了巴西家庭健康计划对婴儿死亡率的异质影响，发现贫困人口和城市地区比其他群体更快地受到政策影响。 |
| [^3] | [Revenue sharing at music streaming platforms.](http://arxiv.org/abs/2310.11861) | 该论文研究了音乐流媒体平台上如何分享订阅费用的问题。作者提供了比例分成和以用户为中心的分成两种方法的基础，并提出了一系列折中方法，以更好地协调艺术家、粉丝和流媒体服务的利益。 |
| [^4] | [Trimmed Mean Group Estimation of Average Treatment Effects in Ultra Short T Panels under Correlated Heterogeneity.](http://arxiv.org/abs/2310.11680) | 本文在相关异质性下，提出了一种修剪均值组（TMG）估计器，可以在面板数据时间维度很小的情况下以不规则的速度保持一致性。该方法具有较好的性质和性能，并提供了相关异质性的检验方法。通过实证应用展示了该方法的实用性。 |
| [^5] | [Global Factors in Non-core Bank Funding and Exchange Rate Flexibility.](http://arxiv.org/abs/2310.11552) | 全球因素对发达经济体银行体系中非核心与核心资金比率的波动起主导作用，汇率灵活性能够在2008-2009年以外的时期减小这种影响。 |
| [^6] | [The Sponge Cake Dilemma over the Nile: Achieving Fairness in Resource Allocation through Rawlsian Theory and Algorithms.](http://arxiv.org/abs/2310.11472) | 通过融合Rawlsian理论和算法手段，本文研究了水资源分配中的公平性问题。蛋糕切割模型在考虑战略因素的同时，寻找与Rawlsian公平原则相一致的解决方案，为解决尼罗河水资源争端提供了有价值的方法。 |
| [^7] | [\"Uber wissenschaftliche Exzellenz und Wettbewerb.](http://arxiv.org/abs/2310.09588) | 本文讨论了科学研究和竞争背景下卓越的概念。 |
| [^8] | [On Optimal Set Estimation for Partially Identified Binary Choice Models.](http://arxiv.org/abs/2310.02414) | 本文重新考虑了部分识别模型估计中的最优性概念，并提出了在所有设计中收敛于识别区域的替代方法。 |
| [^9] | [Multi-period static hedging of European options.](http://arxiv.org/abs/2310.01104) | 本文研究了基于单因素马尔可夫框架的多期欧式期权的静态对冲，并对相对性能进行了实验比较。 |
| [^10] | [Another Look at the Linear Probability Model and Nonlinear Index Models.](http://arxiv.org/abs/2308.15338) | 本文重新评估了使用线性模型来近似二元结果的响应概率的部分效应的利弊，并重点研究了平均部分效应而不是底层线性指数的参数。通过模拟实验，发现OLS近似或不近似APE的各种情况，并提出了一种实用的方法来减少OLS的有限样本偏差，结果表明该方法与非线性最小二乘（NLS）估计数值上相等。 |
| [^11] | [The far-reaching effects of bombing on fertility in mid-20th century Japan.](http://arxiv.org/abs/2306.05770) | 本研究探究了空袭对20世纪日本生育率的深远影响，并证明了战争破坏的区域影响即使在未直接受影响的地区也存在。 |
| [^12] | [Monotonicity Anomalies in Scottish Local Government Elections.](http://arxiv.org/abs/2305.17741) | 本研究分析了苏格兰1,079个地方政府STV选举，发现其中41次出现某种单调性异常，且这些异常率与之前的经验研究相似，远低于大多数理论研究发现的异常率。 |
| [^13] | [On Robust Inference in Time Series Regression.](http://arxiv.org/abs/2203.04080) | 该论文研究了时间序列回归中的鲁棒推断问题，指出了将HAC估计技术应用于时间序列环境时的困难，并讨论了OLS参数估计不一致、HAC回归参数估计低效以及HAC条件预测低效等问题。 |

# 详细

[^1]: 调查校准的因果推断方法：一种平衡协变量分布的简单方法

    Survey calibration for causal inference: a simple method to balance covariate distributions. (arXiv:2310.11969v1 [stat.ME])

    [http://arxiv.org/abs/2310.11969](http://arxiv.org/abs/2310.11969)

    本文提出一种简单的方法用于平衡因果推断中观察性研究中的协变量分布，通过修改熵平衡方法和协变量平衡倾向评分方法实现对处理组和对照组的分布平衡。

    

    本文提出了一种基于观察性研究的因果推断的简单方法，用于平衡协变量的分布。该方法可以平衡任意数量的分位数（例如中位数、四分位数或十分位数），如果必要的话还可以平衡均值。所提出的方法基于校正估计器的理论（Deville和Sarndal，1992），尤其是Harms和Duchesne（2006）提出的分位数校准估计器。通过修改熵平衡方法和协变量平衡倾向评分方法，可以平衡处理组和对照组的分布。该方法不需要数值积分、核密度估计或对分布的任何假设；可以通过利用现有的渐近理论得到有效的估计。模拟研究的结果表明，该方法可以有效地估计被处理组的平均处理效应（ATT）、平均处理效应（ATE）和处理组的总处理效应（ATE）等。

    This paper proposes a simple method for balancing distributions of covariates for causal inference based on observational studies. The method makes it possible to balance an arbitrary number of quantiles (e.g., medians, quartiles, or deciles) together with means if necessary. The proposed approach is based on the theory of calibration estimators (Deville and S\"arndal 1992), in particular, calibration estimators for quantiles, proposed by Harms and Duchesne (2006). By modifying the entropy balancing method and the covariate balancing propensity score method, it is possible to balance the distributions of the treatment and control groups. The method does not require numerical integration, kernel density estimation or assumptions about the distributions; valid estimates can be obtained by drawing on existing asymptotic theory. Results of a simulation study indicate that the method efficiently estimates average treatment effects on the treated (ATT), the average treatment effect (ATE), th
    
[^2]: 用机器学习方法研究错位差异-差异和动态治疗效果异质性

    Machine Learning for Staggered Difference-in-Differences and Dynamic Treatment Effect Heterogeneity. (arXiv:2310.11962v1 [econ.EM])

    [http://arxiv.org/abs/2310.11962](http://arxiv.org/abs/2310.11962)

    本文提出了一种机器学习差异-差异（MLDID）方法，结合了两种非参数差异-差异方法，用于研究错位采用环境中的治疗效果异质性。我们通过模拟实验证明MLDID能够准确识别出治疗效果异质性的预测因素，并利用该方法评估了巴西家庭健康计划对婴儿死亡率的异质影响，发现贫困人口和城市地区比其他群体更快地受到政策影响。

    

    我们结合了两种最近提出的非参数差异-差异方法，将它们扩展到使用机器学习方法在错位采用环境中研究治疗效果异质性。所提出的方法，机器学习差异-差异（MLDID），允许估计处理组的时变条件平均治疗效应，从而可以对治疗效果异质性的驱动因素进行详细推断。我们进行模拟实验来评估MLDID的性能，并发现它能准确识别出治疗效果异质性的真实预测因素。然后，我们使用MLDID来评估巴西家庭健康计划对婴儿死亡率的异质影响，并发现贫困人口和城市地区比其他子群体更快地受到政策影响。

    We combine two recently proposed nonparametric difference-in-differences methods, extending them to enable the examination of treatment effect heterogeneity in the staggered adoption setting using machine learning. The proposed method, machine learning difference-in-differences (MLDID), allows for estimation of time-varying conditional average treatment effects on the treated, which can be used to conduct detailed inference on drivers of treatment effect heterogeneity. We perform simulations to evaluate the performance of MLDID and find that it accurately identifies the true predictors of treatment effect heterogeneity. We then use MLDID to evaluate the heterogeneous impacts of Brazil's Family Health Program on infant mortality, and find those in poverty and urban locations experienced the impact of the policy more quickly than other subgroups.
    
[^3]: 音乐流媒体平台上的分成问题

    Revenue sharing at music streaming platforms. (arXiv:2310.11861v1 [econ.TH])

    [http://arxiv.org/abs/2310.11861](http://arxiv.org/abs/2310.11861)

    该论文研究了音乐流媒体平台上如何分享订阅费用的问题。作者提供了比例分成和以用户为中心的分成两种方法的基础，并提出了一系列折中方法，以更好地协调艺术家、粉丝和流媒体服务的利益。

    

    我们研究了在音乐流媒体平台上分享订阅费用的问题，其中包括直接、公理和博弈论基础。我们提供了两种在实践中广泛使用的方法的基础：比例分成和以用户为中心的分成。前者根据艺术家的总播放次数来奖励他们。后者按照用户所播放的艺术家来将用户的订阅费平均分配给艺术家。我们还提供了一系列方法的基础，这些方法在前两种方法之间进行权衡，解决了音乐行业对探索更好地协调艺术家、粉丝和流媒体服务利益的新流媒体模型的关切。

    We study the problem of sharing the revenues raised from subscriptions to music streaming platforms among content providers. We provide direct, axiomatic and game-theoretical foundations for two focal (and somewhat polar) methods widely used in practice: pro-rata and user-centric. The former rewards artists proportionally to their number of total streams. With the latter, each user's subscription fee is proportionally divided among the artists streamed by that user. We also provide foundations for a family of methods compromising among the previous two, which addresses the rising concern in the music industry to explore new streaming models that better align the interests of artists, fans and streaming services.
    
[^4]: 在相关异质性下，短期面板中关于平均处理效应的修剪均值组估计方法

    Trimmed Mean Group Estimation of Average Treatment Effects in Ultra Short T Panels under Correlated Heterogeneity. (arXiv:2310.11680v1 [econ.EM])

    [http://arxiv.org/abs/2310.11680](http://arxiv.org/abs/2310.11680)

    本文在相关异质性下，提出了一种修剪均值组（TMG）估计器，可以在面板数据时间维度很小的情况下以不规则的速度保持一致性。该方法具有较好的性质和性能，并提供了相关异质性的检验方法。通过实证应用展示了该方法的实用性。

    

    在相关异质性下，常用的两路固定效应估计方法存在偏差并可能导致误导性推断。本文提出了一种新的修剪均值组估计器（TMG estimator），即使面板的时间维度与回归变量数目一样小，也能以不规则的n^{1/3}速度保持一致性。本文还提供了适用于具有时间效应的面板的扩展方法，并提出了一种相关异质性的豪斯曼式检验。通过蒙特卡洛实验，研究了TMG估计器（带有和不带有时间效应）在小样本情况下的性质，结果表明其性能令人满意，优于文献中提出的其他修剪估计器。同时，所提出的相关异质性检验显示出正确的大小和令人满意的功效。通过实证应用，展示了TMG方法的实用性。

    Under correlated heterogeneity, the commonly used two-way fixed effects estimator is biased and can lead to misleading inference. This paper proposes a new trimmed mean group (TMG) estimator which is consistent at the irregular rate of n^{1/3} even if the time dimension of the panel is as small as the number of its regressors. Extensions to panels with time effects are provided, and a Hausman-type test of correlated heterogeneity is proposed. Small sample properties of the TMG estimator (with and without time effects) are investigated by Monte Carlo experiments and shown to be satisfactory and perform better than other trimmed estimators proposed in the literature. The proposed test of correlated heterogeneity is also shown to have the correct size and satisfactory power. The utility of the TMG approach is illustrated with an empirical application.
    
[^5]: 非核心银行资金和汇率灵活性中的全球因素

    Global Factors in Non-core Bank Funding and Exchange Rate Flexibility. (arXiv:2310.11552v1 [econ.GN])

    [http://arxiv.org/abs/2310.11552](http://arxiv.org/abs/2310.11552)

    全球因素对发达经济体银行体系中非核心与核心资金比率的波动起主导作用，汇率灵活性能够在2008-2009年以外的时期减小这种影响。

    

    我们展示了发达经济体银行体系中非核心与核心资金比率的波动由少数几个既有实物性又有金融性质的全球因素驱动，国家特定因素没有发挥重要作用。汇率灵活性有助于减小非核心与核心比率受到全球因素的影响，但仅在重大全球金融震荡期间（如2008-2009年）明显起作用。

    We show that fluctuations in the ratio of non-core to core funding in the banking systems of advanced economies are driven by a handful of global factors of both real and financial natures, with country-specific factors playing no significant roles. Exchange rate flexibility helps insulate the non-core to core ratio from such global factors but only significantly so outside periods of major global financial disruptions, as in 2008-2009.
    
[^6]: 尼罗河上的海绵蛋糕困境：通过Rawlsian理论和算法实现资源分配的公平性

    The Sponge Cake Dilemma over the Nile: Achieving Fairness in Resource Allocation through Rawlsian Theory and Algorithms. (arXiv:2310.11472v1 [econ.GN])

    [http://arxiv.org/abs/2310.11472](http://arxiv.org/abs/2310.11472)

    通过融合Rawlsian理论和算法手段，本文研究了水资源分配中的公平性问题。蛋糕切割模型在考虑战略因素的同时，寻找与Rawlsian公平原则相一致的解决方案，为解决尼罗河水资源争端提供了有价值的方法。

    

    本文通过整合规范和实证的视角，研究了水资源争端。约翰·罗尔斯的正义理论提供了道德指导，维护所有流域国家的合理获取权利。然而，通过蛋糕切割模型进行实证分析揭示了现实世界中的战略约束。虽然罗尔斯定义了期望的目标，但蛋糕切割提供了基于实际行为的算法手段。尼罗河流域争端说明了这种综合。罗尔斯认为水资源具有天然权利，但无限制的竞争可能导致垄断。在自利等局限性存在的情况下，他的原则本身无法防止不利的结果。这就是蛋糕切割在存在偏见主张的情况下提供价值的地方。其模型确定了与罗尔斯公平原则相符的安排，并纳入了战略考虑。本文详细介绍了蛋糕切割理论，回顾了水资源冲突文献，研究了尼罗河案例，探讨了合作与非合作的情况。

    This article examines water disputes through an integrated framework combining normative and positive perspectives. John Rawls' theory of justice provides moral guidance, upholding rights to reasonable access for all riparian states. However, positive analysis using cake-cutting models reveals real-world strategic constraints. While Rawls defines desired ends, cake-cutting offers algorithmic means grounded in actual behaviors. The Nile River basin dispute illustrates this synthesis. Rawls suggests inherent rights to water, but unrestricted competition could enable monopoly. His principles alone cannot prevent unfavorable outcomes, given limitations like self-interest. This is where cake-cutting provides value despite biased claims. Its models identify arrangements aligning with Rawlsian fairness while incorporating strategic considerations. The article details the cake-cutting theory, reviews water conflicts literature, examines the Nile case, explores cooperative vs. non-cooperative g
    
[^7]: "关于科学卓越与竞争的论文"

    \"Uber wissenschaftliche Exzellenz und Wettbewerb. (arXiv:2310.09588v1 [econ.GN])

    [http://arxiv.org/abs/2310.09588](http://arxiv.org/abs/2310.09588)

    本文讨论了科学研究和竞争背景下卓越的概念。

    

    追求卓越似乎是学术界的真正目标。卓越是什么意思？能否衡量卓越？本文讨论了在研究和竞争的背景下卓越的概念。

    The pursuit of excellence seems to be the True North of academia. What is meant by excellence? Can excellence be measured? This article discusses the concept of excellence in the context of research and competition.
    
[^8]: 关于部分识别二项选择模型的最优集估计

    On Optimal Set Estimation for Partially Identified Binary Choice Models. (arXiv:2310.02414v1 [econ.EM])

    [http://arxiv.org/abs/2310.02414](http://arxiv.org/abs/2310.02414)

    本文重新考虑了部分识别模型估计中的最优性概念，并提出了在所有设计中收敛于识别区域的替代方法。

    

    在本文中，我们重新考虑了部分识别模型估计中的最优性概念。我们以半参数二项选择模型为例，以离散协变量作为示例，说明了一般问题。该模型在一定程度上是部分识别的，例如Bierens和Hartog（1988）所示。通过实施Manski（1975）提出的最大分数程序，可以构建模型中回归系数的集合估计。对于许多设计，该方法收敛于这些参数的识别集，因此在某种意义上是最优的。但是，正如Komarova（2013）所示，对于其他情况，最大分数目标函数给出了识别集的边界区域。这激发了寻求其他优化方法的动力，这些方法在所有设计中都收敛于识别区域，并且我们提出并比较了这些方法。一个是Hodges类型的估计器，将最大分数估计器与现有程序相结合。第二个是两步法

    In this paper we reconsider the notion of optimality in estimation of partially identified models. We illustrate the general problem in the context of a semiparametric binary choice model with discrete covariates as an example of a model which is partially identified as shown in, e.g. Bierens and Hartog (1988). A set estimator for the regression coefficients in the model can be constructed by implementing the Maximum Score procedure proposed by Manski (1975). For many designs this procedure converges to the identified set for these parameters, and so in one sense is optimal. But as shown in Komarova (2013) for other cases the Maximum Score objective function gives an outer region of the identified set. This motivates alternative methods that are optimal in one sense that they converge to the identified region in all designs, and we propose and compare such procedures. One is a Hodges type estimator combining the Maximum Score estimator with existing procedures. A second is a two step e
    
[^9]: 多期欧式期权的静态对冲

    Multi-period static hedging of European options. (arXiv:2310.01104v1 [q-fin.MF])

    [http://arxiv.org/abs/2310.01104](http://arxiv.org/abs/2310.01104)

    本文研究了基于单因素马尔可夫框架的多期欧式期权的静态对冲，并对相对性能进行了实验比较。

    

    本文考虑了在基础资产价格遵循单因素马尔可夫框架的情况下对欧式期权进行对冲。Carr和Wu [1]在这样的设置下，导出了给定期权与在同一资产上写的一系列较短期限期权之间的跨度关系。在本文中，我们将他们的方法扩展到同时包括多个短期到期的期权。然后，我们使用高斯求积方法通过有限的一组短期期权确定对冲误差的实际实现。我们对\textit{Black-Scholes}和\textit{Merton Jump Diffusion}模型进行了广泛的实验，展示了这两种方法的比较性能。

    We consider the hedging of European options when the price of the underlying asset follows a single-factor Markovian framework. By working in such a setting, Carr and Wu \cite{carr2014static} derived a spanning relation between a given option and a continuum of shorter-term options written on the same asset. In this paper, we have extended their approach to simultaneously include options over multiple short maturities. We then show a practical implementation of this with a finite set of shorter-term options to determine the hedging error using a Gaussian Quadrature method. We perform a wide range of experiments for both the \textit{Black-Scholes} and \textit{Merton Jump Diffusion} models, illustrating the comparative performance of the two methods.
    
[^10]: 对线性概率模型和非线性指数模型的再思考

    Another Look at the Linear Probability Model and Nonlinear Index Models. (arXiv:2308.15338v1 [econ.EM])

    [http://arxiv.org/abs/2308.15338](http://arxiv.org/abs/2308.15338)

    本文重新评估了使用线性模型来近似二元结果的响应概率的部分效应的利弊，并重点研究了平均部分效应而不是底层线性指数的参数。通过模拟实验，发现OLS近似或不近似APE的各种情况，并提出了一种实用的方法来减少OLS的有限样本偏差，结果表明该方法与非线性最小二乘（NLS）估计数值上相等。

    

    我们重新考虑了使用线性模型来近似二元结果的响应概率的部分效应的利弊。特别地，我们研究了Horrace和Oaxaca (2006)中的斜坡模型，但我们的重点是平均部分效应(APE)，而不是底层线性指数的参数。我们使用现有的理论结果来验证线性投影参数（这些参数总是被普通最小二乘法（OLS）一致估计）可能与指数参数不同，但在某些情况下仍然与APE相同。使用模拟，我们描述了OLS近似或不近似APE的其他情况，并发现在[0,1]范围内具有大比例的拟合值既不是必要的，也不是充分的。减少OLS有限样本偏差的实用方法是迭代修剪超出单位间隔的观测值，我们发现这产生的估计量在数值上等同于非线性最小二乘（NLS）估计。

    We reconsider the pros and cons of using a linear model to approximate partial effects on a response probability for a binary outcome. In particular, we study the ramp model in Horrace and Oaxaca (2006), but focus on average partial effects (APE) rather than the parameters of the underlying linear index. We use existing theoretical results to verify that the linear projection parameters (which are always consistently estimated by ordinary least squares (OLS)) may differ from the index parameters, yet still be identical to the APEs in some cases. Using simulations, we describe other cases where OLS either does or does not approximate the APEs, and we find that having a large fraction of fitted values in [0,1] is neither necessary nor sufficient. A practical approach to reduce the finite sample bias of OLS is to iteratively trim the observations with fitted values outside the unit interval, which we find produces estimates numerically equivalent to nonlinear least squares (NLS) estimatio
    
[^11]: 二战期间轰炸对20世纪日本生育率的深远影响

    The far-reaching effects of bombing on fertility in mid-20th century Japan. (arXiv:2306.05770v1 [econ.GN])

    [http://arxiv.org/abs/2306.05770](http://arxiv.org/abs/2306.05770)

    本研究探究了空袭对20世纪日本生育率的深远影响，并证明了战争破坏的区域影响即使在未直接受影响的地区也存在。

    

    战争和冲突之后的生育变化在全球范围内得到了观察。本研究旨在研究区域战争破坏是否会影响战后生育，即使是未直接受到影响但靠近受损地区的地区。为了达到这个目的，我们利用了日本二战期间的空袭经历。利用1935年和1947年近畿地区的市町村级别生育数据以及城市空袭损失数据，我们发现轰炸对于15公里内城镇和乡村的战后生育率存在影响，尽管这些地区未直接受到损害。然而，间接影响的方向是混合的。估计结果表明，邻近城市的严重空袭增加了生育率，而较轻的空袭则降低了生育率。此外，拟实验法的结果表明，严重的空袭恐惧在战后期间增加了生育率。本研究为战后生育变化的文献提供了证据，即轰炸对于生育率有深远影响，即使是被战争损害间接影响的地区。

    Fertility changes after wars and conflicts have been observed worldwide. This study examines whether regional war damage affects postwar fertility even in areas that were not directly affected but were close to the damaged areas. In order to accomplish this, we exploit the air-raid experience in Japan during World War II. Using the municipality-level fertility data in the Kinki region in 1935 and 1947 and the data on damages from air raids in cities, we find the effects of bombing on postwar fertility in towns and villages within 15 kilometers, despite no direct damages. However, the direction of the indirect effects is mixed. The estimation results suggest that severe air raids in neighboring cities increased fertility, whereas minor air raids decreased it. Moreover, the results of the quasi-experimental approach indicate that intense fears of air raids increased the fertility rate in the postwar period. Our study contributes to the literature on fertility changes in the postwar perio
    
[^12]: 苏格兰地方政府选举中的单调性异常现象

    Monotonicity Anomalies in Scottish Local Government Elections. (arXiv:2305.17741v1 [econ.GN])

    [http://arxiv.org/abs/2305.17741](http://arxiv.org/abs/2305.17741)

    本研究分析了苏格兰1,079个地方政府STV选举，发现其中41次出现某种单调性异常，且这些异常率与之前的经验研究相似，远低于大多数理论研究发现的异常率。

    

    单一可转移选票（STV）投票方法用于选举排名选举中的多个候选人。STV的一个弱点是它未能通过与单调性和无表决悖论相关的多个公平标准。我们分析了苏格兰1,079个地方政府STV选举，以估计现实世界选举中此类单调性异常的频率，并将结果与关于此类异常出现率的先前经验和理论研究进行比较。在1079次选举中，我们发现41次出现某种单调性异常。我们通常发现异常率与之前的经验研究相似，远低于大多数理论研究发现的异常率。我们发现的大多数STV异常都是第一次在实际选举中记录到的。

    The single transferable vote (STV) voting method is used to elect multiple candidates in ranked-choice elections. One weakness of STV is that it fails multiple fairness criteria related to monotonicity and no-show paradoxes. We analyze 1,079 local government STV elections in Scotland to estimate the frequency of such monotonicity anomalies in real-world elections, and compare our results with prior empirical and theoretical research about the rates at which such anomalies occur. In 41 of the 1079 elections we found some kind of monotonicity anomaly. We generally find that the rates of anomalies are similar to prior empirical research and much lower than what most theoretical research has found. Most of the STV anomalies we find are the first of their kind to be documented in real-world elections.
    
[^13]: 关于时间序列回归中的鲁棒推断

    On Robust Inference in Time Series Regression. (arXiv:2203.04080v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2203.04080](http://arxiv.org/abs/2203.04080)

    该论文研究了时间序列回归中的鲁棒推断问题，指出了将HAC估计技术应用于时间序列环境时的困难，并讨论了OLS参数估计不一致、HAC回归参数估计低效以及HAC条件预测低效等问题。

    

    在横截面环境中，具有异方差和自相关一致（HAC）标准误的最小二乘回归非常有用。然而，在将HAC估计技术应用于时间序列环境时，会面临几个常常被忽视的主要困难。首先，在可能出现强外生性失败的时间序列环境中，OLS参数估计可能不一致，因此即使渐近下也无法进行HAC推断。其次，大多数经济时间序列具有强自相关性，这使得HAC回归参数估计非常低效。第三，强自相关性同样使得HAC条件预测非常低效。最后，流行的HAC估计器的结构不适合捕捉通常存在于经济时间序列中的自回归自相关性，这会产生较大的规模扭曲和HAC假设检验中的功效降低，除非样本数量非常大。

    Least squares regression with heteroskedasticity and autocorrelation consistent (HAC) standard errors has proved very useful in cross section environments. However, several major difficulties, which are generally overlooked, must be confronted when transferring the HAC estimation technology to time series environments. First, in plausible time-series environments involving failure of strong exogeneity, OLS parameter estimates can be inconsistent, so that HAC inference fails even asymptotically. Second, most economic time series have strong autocorrelation, which renders HAC regression parameter estimates highly inefficient. Third, strong autocorrelation similarly renders HAC conditional predictions highly inefficient. Finally, The structure of popular HAC estimators is ill-suited for capturing the autoregressive autocorrelation typically present in economic time series, which produces large size distortions and reduced power in HACbased hypothesis testing, in all but the largest sample
    

