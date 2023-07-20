# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Reconciling the Theory of Factor Sequences.](http://arxiv.org/abs/2307.10067) | 该论文研究了动态因子序列和静态因子序列之间的区别，强调了忽略弱共同成分对于因子模型在结构分析和预测中的应用可能产生的重大影响，同时指出动态共同成分才能被解释为潜在经济变量。 |
| [^2] | [Asymptotic equivalence of Principal Component and Quasi Maximum Likelihood estimators in Large Approximate Factor Models.](http://arxiv.org/abs/2307.09864) | 在大型近似因子模型中，我们证明主成分估计的因子载荷与准极大似然估计的载荷等价，同时这两种估计也与不可行最小二乘估计的载荷等价。我们还证明了准极大似然估计的协方差矩阵与不可行最小二乘的协方差矩阵等价，从而可以简化准极大似然估计的置信区间的估计过程。所有结果都适用于假设异方差跨截面的一般情况。 |
| [^3] | [Power to the teens? A model of parents' and teens' collective labor supply.](http://arxiv.org/abs/2307.09634) | 本研究探讨了青少年和父母在家庭决策过程中的共同作用，发现了父母和青少年之间的讨价还价过程导致了性别差距，并提出了一个集体家庭模型。这一结果表明，儿子参加学校的机会成本比女儿更高，公共政策必须考虑到这种性别差距。 |
| [^4] | [Reparametrization and the Semiparametric Bernstein-von-Mises Theorem.](http://arxiv.org/abs/2306.03816) | 本文提出了一种参数化形式，该形式可以通过生成Neyman正交矩条件来降低对干扰参数的敏感度，从而可以用于去偏贝叶斯推断中的后验分布，同时在参数速率下对低维参数进行真实值的收缩，并在半参数效率界的方差下进行渐近正态分布。 |
| [^5] | [Dynamic Transportation of Economic Agents.](http://arxiv.org/abs/2303.12567) | 本文通过提出新的方法，解决了之前在某些异质性代理人不完全市场模型的宏观经济均衡解决方案的问题。 |
| [^6] | [Cointegration with Occasionally Binding Constraints.](http://arxiv.org/abs/2211.09604) | 本研究在非线性协整性的背景下，探讨了在时变约束下如何生成共同的线性和非线性随机趋势的问题。通过使用被审查和弯曲的结构化VAR模型，我们提供了对于时间序列受到阈值型非线性性质影响的全面描述。 |
| [^7] | [On Estimation and Inference of Large Approximate Dynamic Factor Models via the Principal Component Analysis.](http://arxiv.org/abs/2211.01921) | 本文研究了大型近似动态因子模型的主成分分析估计和推断，提供了渐近结果的替代推导，并且通过经典的样本协方差矩阵进行估计，得出它们等价于OLS的结论。 |
| [^8] | [Beta-Sorted Portfolios.](http://arxiv.org/abs/2208.10974) | 该论文对Beta分类组合投资组合进行了研究，通过将过程形式化为一个由非参数第一步和Beta自适应投资组合构建组成的两步非参数估计器，解释了该估计算法的关键特征，并提供了条件以确保一致性和渐近正态性。 |
| [^9] | [Unconditional Effects of General Policy Interventions.](http://arxiv.org/abs/2201.02292) | 该论文研究了通用政策干预的无条件效应，包括位置-比例换算和同时换算。使用简单的半参数估计器对无条件政策参数进行估计，并应用于明塞尔方程以研究其效果。 |
| [^10] | [A Design-Based Perspective on Synthetic Control Methods.](http://arxiv.org/abs/2101.09398) | 本文从设计角度研究了合成控制（SC）方法，提出了一个修改的无偏合成控制（MUSC）估计量，在随机分配下保证无偏，并且其均方根误差低于其他常见估计器。 |
| [^11] | [An Automatic Finite-Sample Robustness Metric: When Can Dropping a Little Data Make a Big Difference?.](http://arxiv.org/abs/2011.14999) | 该论文提出了一种自动的有限样本稳健性度量方法，用于评估应用计量经济学结论对样本的小部分删除的敏感性。 |
| [^12] | [Identification and Estimation of Unconditional Policy Effects of an Endogenous Binary Treatment: an Unconditional MTE Approach.](http://arxiv.org/abs/2010.15864) | 本文介绍了一种基于函数影响函数的无条件边际处理效应（MTE）的新方法，研究了处理状态为二元且内生时策略效应的识别与估计。我们证明了无条件策略效应可以表示为对于对待自己的处理状态处于决策困惑的个体而言新定义的无条件MTE的加权平均。对于感兴趣的分位数，我们引入了无条件工具分位数估计（UNIQUE）并证明了其一致性和渐近分布。在实证应用中，我们估计了通过更高的学费补贴引起的大学入学状态变化对工资分布的分位数的影响。 |

# 详细

[^1]: 调和因子序列理论

    Reconciling the Theory of Factor Sequences. (arXiv:2307.10067v1 [econ.EM])

    [http://arxiv.org/abs/2307.10067](http://arxiv.org/abs/2307.10067)

    该论文研究了动态因子序列和静态因子序列之间的区别，强调了忽略弱共同成分对于因子模型在结构分析和预测中的应用可能产生的重大影响，同时指出动态共同成分才能被解释为潜在经济变量。

    

    因子序列是随机的双序列$(y_{it}: i \in \mathbb N, t \in \mathbb Z)$，以时间和横截面为索引，具有所谓的因子结构。该名词由Forni等人于2001年提出，引入了动态因子序列。我们展示了动态因子序列和静态因子序列之间的区别，静态因子序列是计量经济学因子分析中最常见的工作模型，基于Chamberlain和Rothschild （1983），Stock和Watson（2002）和Bai和Ng（2002）。区别在于我们所称的弱共同成分，该成分由潜在无限多个弱因子所构成。忽略弱共同成分可能对结构分析和预测中因子模型的应用产生重大影响。我们还展示了动态因子序列的动态共同成分在一般条件下是因果从属于输出的。因此，只有动态共同成分才能作为潜在经济变量的解释。

    Factor Sequences are stochastic double sequences $(y_{it}: i \in \mathbb N, t \in \mathbb Z)$ indexed in time and cross-section which have a so called factor structure. The name was coined by Forni et al. 2001, who introduced dynamic factor sequences. We show the difference between dynamic factor sequences and static factor sequences which are the most common workhorse model of econometric factor analysis building on Chamberlain and Rothschild (1983), Stock and Watson (2002) and Bai and Ng (2002). The difference consists in what we call the weak common component which is spanned by a potentially infinite number of weak factors. Ignoring the weak common component can have substantial consequences for applications of factor models in structural analysis and forecasting. We also show that the dynamic common component of a dynamic factor sequence is causally subordinated to the output under general conditions. As a consequence only the dynamic common component can be interpreted as the pro
    
[^2]: 大型近似因子模型中主成分和准极大似然估计量的渐近等价性分析

    Asymptotic equivalence of Principal Component and Quasi Maximum Likelihood estimators in Large Approximate Factor Models. (arXiv:2307.09864v1 [econ.EM])

    [http://arxiv.org/abs/2307.09864](http://arxiv.org/abs/2307.09864)

    在大型近似因子模型中，我们证明主成分估计的因子载荷与准极大似然估计的载荷等价，同时这两种估计也与不可行最小二乘估计的载荷等价。我们还证明了准极大似然估计的协方差矩阵与不可行最小二乘的协方差矩阵等价，从而可以简化准极大似然估计的置信区间的估计过程。所有结果都适用于假设异方差跨截面的一般情况。

    

    我们证明在一个$n$维的稳定时间序列向量的近似因子模型中，通过主成分估计的因子载荷在$n\to\infty$时与准极大似然估计得到的载荷等价。这两种估计量在$n\to\infty$时也与如果观察到因子时的不可行最小二乘估计等价。我们还证明了准极大似然估计的渐近协方差矩阵的传统三明治形式与不可行最小二乘的简单渐近协方差矩阵等价。这提供了一种简单的方法来估计准极大似然估计的渐近置信区间，而不需要估计复杂的海森矩阵和费谢尔信息矩阵。所有结果均适用于假设异方差跨截面的一般情况。

    We prove that in an approximate factor model for an $n$-dimensional vector of stationary time series the factor loadings estimated via Principal Components are asymptotically equivalent, as $n\to\infty$, to those estimated by Quasi Maximum Likelihood. Both estimators are, in turn, also asymptotically equivalent, as $n\to\infty$, to the unfeasible Ordinary Least Squares estimator we would have if the factors were observed. We also show that the usual sandwich form of the asymptotic covariance matrix of the Quasi Maximum Likelihood estimator is asymptotically equivalent to the simpler asymptotic covariance matrix of the unfeasible Ordinary Least Squares. This provides a simple way to estimate asymptotic confidence intervals for the Quasi Maximum Likelihood estimator without the need of estimating the Hessian and Fisher information matrices whose expressions are very complex. All our results hold in the general case in which the idiosyncratic components are cross-sectionally heteroskedast
    
[^3]: 青少年权力? 父母和青少年的共同劳动力供应模型

    Power to the teens? A model of parents' and teens' collective labor supply. (arXiv:2307.09634v1 [econ.GN])

    [http://arxiv.org/abs/2307.09634](http://arxiv.org/abs/2307.09634)

    本研究探讨了青少年和父母在家庭决策过程中的共同作用，发现了父母和青少年之间的讨价还价过程导致了性别差距，并提出了一个集体家庭模型。这一结果表明，儿子参加学校的机会成本比女儿更高，公共政策必须考虑到这种性别差距。

    

    青少年在成长环境中的需求和资源的限制下做出了改变一生的决策。家庭行为模型通常将决策权委托给青少年或他们的父母，忽略了家庭中的共同决策过程。本研究利用2011年至2019年的哥斯达黎加Encuesta Nacional de Hogares的数据和有条件的现金转移计划，展示了青少年和父母如何共同分配时间和收入。首先，通过边际处理效应框架，展示了家庭对转移的性别差异响应。其次，解释了结果中的性别差距是由父母和青少年之间的讨价还价过程导致的。研究提出了一个集体家庭模型，并显示儿子与父母合作讨价的，而女儿却不是。这一结果意味着儿子上学的机会成本比女儿高。针对青少年的公共政策必须考虑到这种性别差距才能有效果。

    Teens make life-changing decisions while constrained by the needs and resources of the households they grow up in. Household behavior models frequently delegate decision-making to the teen or their parents, ignoring joint decision-making in the household. I show that teens and parents allocate time and income jointly by using data from the Costa Rican Encuesta Nacional de Hogares from 2011 to 2019 and a conditional cash transfer program. First, I present gender differences in household responses to the transfer using a marginal treatment effect framework. Second, I explain how the gender gap from the results is due to the bargaining process between parents and teens. I propose a collective household model and show that sons bargain cooperatively with their parents while daughters do not. This result implies that sons have a higher opportunity cost of attending school than daughters. Public policy targeting teens must account for this gender disparity to be effective.
    
[^4]: 重参数化与半参数Bernstein-von-Mises定理

    Reparametrization and the Semiparametric Bernstein-von-Mises Theorem. (arXiv:2306.03816v1 [math.ST])

    [http://arxiv.org/abs/2306.03816](http://arxiv.org/abs/2306.03816)

    本文提出了一种参数化形式，该形式可以通过生成Neyman正交矩条件来降低对干扰参数的敏感度，从而可以用于去偏贝叶斯推断中的后验分布，同时在参数速率下对低维参数进行真实值的收缩，并在半参数效率界的方差下进行渐近正态分布。

    

    本文考虑了部分线性模型的贝叶斯推断。我们的方法利用了回归函数的一个参数化形式，该形式专门用于估计所关心的低维参数。参数化的关键特性是生成了一个Neyman正交矩条件，这意味着对干扰参数的估计低维参数不太敏感。我们的大样本分析支持了这种说法。特别地，我们推导出充分的条件，使得低维参数的后验在参数速率下对真实值收缩，并且在半参数效率界的方差下渐近地正态分布。这些条件相对于回归模型的原始参数化允许更大类的干扰参数。总的来说，我们得出结论，一个嵌入了Neyman正交性的参数化方法可以成为半参数推断中的一个有用工具，以去偏后验分布。

    This paper considers Bayesian inference for the partially linear model. Our approach exploits a parametrization of the regression function that is tailored toward estimating a low-dimensional parameter of interest. The key property of the parametrization is that it generates a Neyman orthogonal moment condition meaning that the low-dimensional parameter is less sensitive to the estimation of nuisance parameters. Our large sample analysis supports this claim. In particular, we derive sufficient conditions under which the posterior for the low-dimensional parameter contracts around the truth at the parametric rate and is asymptotically normal with a variance that coincides with the semiparametric efficiency bound. These conditions allow for a larger class of nuisance parameters relative to the original parametrization of the regression model. Overall, we conclude that a parametrization that embeds Neyman orthogonality can be a useful device for debiasing posterior distributions in semipa
    
[^5]: 经济主体的动态运输

    Dynamic Transportation of Economic Agents. (arXiv:2303.12567v1 [econ.GN])

    [http://arxiv.org/abs/2303.12567](http://arxiv.org/abs/2303.12567)

    本文通过提出新的方法，解决了之前在某些异质性代理人不完全市场模型的宏观经济均衡解决方案的问题。

    

    本文是在发现了一个共同的策略未能将某些异质性代理人不完全市场模型的宏观经济均衡定位到广泛引用的基准研究中而引发的。通过模仿Dumas和Lyasoff（2012）提出的方法，本文提供了一个新的描述，在面对不可保险的总体和个体风险的大量互动经济体代表的私人状态分布的运动定律。提出了一种新的算法，用于确定回报、最优私人配置和平衡状态下的人口运输，并在两个众所周知的基准研究中进行了测试。

    The paper was prompted by the surprising discovery that the common strategy, adopted in a large body of research, for producing macroeconomic equilibrium in certain heterogeneous-agent incomplete-market models fails to locate the equilibrium in a widely cited benchmark study. By mimicking the approach proposed by Dumas and Lyasoff (2012), the paper provides a novel description of the law of motion of the distribution over the range of private states of a large population of interacting economic agents faced with uninsurable aggregate and idiosyncratic risk. A new algorithm for identifying the returns, the optimal private allocations, and the population transport in the state of equilibrium is developed and is tested in two well known benchmark studies.
    
[^6]: 时变约束下的协整性问题

    Cointegration with Occasionally Binding Constraints. (arXiv:2211.09604v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2211.09604](http://arxiv.org/abs/2211.09604)

    本研究在非线性协整性的背景下，探讨了在时变约束下如何生成共同的线性和非线性随机趋势的问题。通过使用被审查和弯曲的结构化VAR模型，我们提供了对于时间序列受到阈值型非线性性质影响的全面描述。

    

    在非线性协整性的文献中，一个长期的悬而未决问题是关于（非线性）向量自回归如何在时间序列的短期和长期动力学的统一描述中生成“非线性协整性”，即这些序列共享共同的非线性随机趋势。我们在被审查和弯曲的结构化VAR模型（CKSVAR）的背景下考虑了这个问题，该模型提供了一个灵活但易于处理的框架，用于建模受阈值型非线性性质影响的时间序列，例如由于偶尔出现的约束所导致的零下限（ZLB）。我们通过单位根和对通常秩条件的适当推广，提供了这一模型中如何产生共同线性和非线性随机趋势的完整特征描述，这是迄今为止对Granger-J的第一次扩展

    In the literature on nonlinear cointegration, a long-standing open problem relates to how a (nonlinear) vector autoregression, which provides a unified description of the short- and long-run dynamics of a collection of time series, can generate 'nonlinear cointegration' in the profound sense of those series sharing common nonlinear stochastic trends. We consider this problem in the setting of the censored and kinked structural VAR (CKSVAR), which provides a flexible yet tractable framework within which to model time series that are subject to threshold-type nonlinearities, such as those arising due to occasionally binding constraints, of which the zero lower bound (ZLB) on short-term nominal interest rates provides a leading example. We provide a complete characterisation of how common linear and {nonlinear stochastic trends may be generated in this model, via unit roots and appropriate generalisations of the usual rank conditions, providing the first extension to date of the Granger-J
    
[^7]: 对大型近似动态因子模型进行主成分分析的估计和推断

    On Estimation and Inference of Large Approximate Dynamic Factor Models via the Principal Component Analysis. (arXiv:2211.01921v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2211.01921](http://arxiv.org/abs/2211.01921)

    本文研究了大型近似动态因子模型的主成分分析估计和推断，提供了渐近结果的替代推导，并且通过经典的样本协方差矩阵进行估计，得出它们等价于OLS的结论。

    

    我们提供了一种对大型近似因子模型主成分估计器的渐近结果的替代推导。结果是在最小的假设集合下导出的，特别地，我们只需要存在4阶矩。在时间序列设置中给予了特别关注，这是几乎所有最新计量应用因子模型的情况。因此，估计是基于经典的$n\times n$样本协方差矩阵，而不是文献中常考虑的$T\times T$协方差矩阵。事实上，尽管这两种方法在渐近意义下等价，但前者更符合时间序列设置，并且它立即允许我们编写更直观的主成分估计渐近展开，显示它们等价于OLS，只要$\sqrt n/T\to 0$和$\sqrt T/n\to 0$，即在时间序列回归中估计载荷时假设因子已知，而因子则已知。

    We provide an alternative derivation of the asymptotic results for the Principal Components estimator of a large approximate factor model. Results are derived under a minimal set of assumptions and, in particular, we require only the existence of 4th order moments. A special focus is given to the time series setting, a case considered in almost all recent econometric applications of factor models. Hence, estimation is based on the classical $n\times n$ sample covariance matrix and not on a $T\times T$ covariance matrix often considered in the literature. Indeed, despite the two approaches being asymptotically equivalent, the former is more coherent with a time series setting and it immediately allows us to write more intuitive asymptotic expansions for the Principal Component estimators showing that they are equivalent to OLS as long as $\sqrt n/T\to 0$ and $\sqrt T/n\to 0$, that is the loadings are estimated in a time series regression as if the factors were known, while the factors a
    
[^8]: Beta分类组合投资组合研究

    Beta-Sorted Portfolios. (arXiv:2208.10974v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2208.10974](http://arxiv.org/abs/2208.10974)

    该论文对Beta分类组合投资组合进行了研究，通过将过程形式化为一个由非参数第一步和Beta自适应投资组合构建组成的两步非参数估计器，解释了该估计算法的关键特征，并提供了条件以确保一致性和渐近正态性。

    

    Beta分类组合投资组合是由与选择的风险因素具有类似协变性的资产组成的，是经济金融领域中分析(条件)预期收益模型的常用工具。尽管使用广泛，但与可比的两步回归等程序相比，对其统计性质知之甚少。我们通过将该过程作为一个由非参数第一步和Beta自适应投资组合构建组成的两步非参数估计器来形式化研究Beta分类组合投资组合回报的性质。我们的框架基于一般数据生成过程上的精确经济和统计假设，从而解释了众所周知的估计算法，并揭示了其关键特征。我们研究了单个截面和随时间聚合（例如总体均值）的Beta分类组合投资组合，提供了确保一致性和渐近正态性的条件，同时还提供了新的均一推断过程，允许不确定性。

    Beta-sorted portfolios -- portfolios comprised of assets with similar covariation to selected risk factors -- are a popular tool in empirical finance to analyze models of (conditional) expected returns. Despite their widespread use, little is known of their statistical properties in contrast to comparable procedures such as two-pass regressions. We formally investigate the properties of beta-sorted portfolio returns by casting the procedure as a two-step nonparametric estimator with a nonparametric first step and a beta-adaptive portfolios construction. Our framework rationalize the well-known estimation algorithm with precise economic and statistical assumptions on the general data generating process and characterize its key features. We study beta-sorted portfolios for both a single cross-section as well as for aggregation over time (e.g., the grand mean), offering conditions that ensure consistency and asymptotic normality along with new uniform inference procedures allowing for unc
    
[^9]: 通用政策干预的无条件效应

    Unconditional Effects of General Policy Interventions. (arXiv:2201.02292v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2201.02292](http://arxiv.org/abs/2201.02292)

    该论文研究了通用政策干预的无条件效应，包括位置-比例换算和同时换算。使用简单的半参数估计器对无条件政策参数进行估计，并应用于明塞尔方程以研究其效果。

    

    本文研究了通用政策干预的无条件效应，其中包括作为特例的位置-比例换算和同时换算。位置-比例换算是为了研究一个反事实政策，旨在改变协变量的均值或位置，同时也改变其离散度或比例。同时换算是指两个或多个协变量同时发生变化的情况。例如，一个协变量的变化以一定的比率被另一个协变量的变化所补偿。不考虑这些可能的比例或同时换算将导致对感兴趣的结果变量的潜在政策效应的错误评估。使用简单的半参数估计器对无条件政策参数进行估计，并对其渐近性质进行研究。通过蒙特卡洛模拟来研究其有限样本性能。所提出的方法应用于明塞尔方程，以研究其效果。

    This paper studies the unconditional effects of a general policy intervention, which includes location-scale shifts and simultaneous shifts as special cases. The location-scale shift is intended to study a counterfactual policy aimed at changing not only the mean or location of a covariate but also its dispersion or scale. The simultaneous shift refers to the situation where shifts in two or more covariates take place simultaneously. For example, a shift in one covariate is compensated at a certain rate by a shift in another covariate. Not accounting for these possible scale or simultaneous shifts will result in an incorrect assessment of the potential policy effects on an outcome variable of interest. The unconditional policy parameters are estimated with simple semiparametric estimators, for which asymptotic properties are studied. Monte Carlo simulations are implemented to study their finite sample performances. The proposed approach is applied to a Mincer equation to study the effe
    
[^10]: 设计角度下的合成控制方法研究

    A Design-Based Perspective on Synthetic Control Methods. (arXiv:2101.09398v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2101.09398](http://arxiv.org/abs/2101.09398)

    本文从设计角度研究了合成控制（SC）方法，提出了一个修改的无偏合成控制（MUSC）估计量，在随机分配下保证无偏，并且其均方根误差低于其他常见估计器。

    

    自从Abadie和Gardeazabal（2003）提出以来，合成控制（SC）方法已经迅速成为面板数据观测研究中估计因果效应的主要方法之一。正式的讨论通常通过假设潜在结果由一个因子模型生成来激励SC方法。本文从设计角度研究了SC方法，假设选取受试单元和周期的模型。我们表明，在随机分配下，标准的SC估计量通常是有偏的。我们提出了一个修改的无偏合成控制（MUSC）估计量，在随机分配下保证无偏，推导了它的确切、基于随机性的有限样本方差。我们还提出了一个这个方差的无偏估计量。我们在真实数据的设置中记录，随机分配下，SC类估计器的均方根误差可以大大降低，低于其他常见估计器。

    Since their introduction in Abadie and Gardeazabal (2003), Synthetic Control (SC) methods have quickly become one of the leading methods for estimating causal effects in observational studies in settings with panel data. Formal discussions often motivate SC methods by the assumption that the potential outcomes were generated by a factor model. Here we study SC methods from a design-based perspective, assuming a model for the selection of the treated unit(s) and period(s). We show that the standard SC estimator is generally biased under random assignment. We propose a Modified Unbiased Synthetic Control (MUSC) estimator that guarantees unbiasedness under random assignment and derive its exact, randomization-based, finite-sample variance. We also propose an unbiased estimator for this variance. We document in settings with real data that under random assignment, SC-type estimators can have root mean-squared errors that are substantially lower than that of other common estimators. We show
    
[^11]: 一个自动的有限样本稳健性度量：当删除少量数据可能造成很大差异时？

    An Automatic Finite-Sample Robustness Metric: When Can Dropping a Little Data Make a Big Difference?. (arXiv:2011.14999v5 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2011.14999](http://arxiv.org/abs/2011.14999)

    该论文提出了一种自动的有限样本稳健性度量方法，用于评估应用计量经济学结论对样本的小部分删除的敏感性。

    

    研究样本往往以非随机方式与推断和政策决策的目标人群不同。研究人员通常认为，这种对随机抽样的偏离——由于时间和空间上的人口变化，或者真正随机采样的困难——很小，对推断的影响也应该很小。因此，我们可能会担心我们研究的结论过于敏感，对样本数据的非常小的比例过于敏感。我们提出了一种方法来评估应用计量经济学结论对样本的小部分删除的敏感性。手动检查所有可能的小子集的影响在计算上是不可行的，因此我们使用近似方法找到最有影响力的子集合。我们的度量方法，“近似最大影响扰动”，基于经典的影响函数，并且对通常的方法（包括但不限于OLS，IV，MLE，G）可以自动计算。

    Study samples often differ from the target populations of inference and policy decisions in non-random ways. Researchers typically believe that such departures from random sampling -- due to changes in the population over time and space, or difficulties in sampling truly randomly -- are small, and their corresponding impact on the inference should be small as well. We might therefore be concerned if the conclusions of our studies are excessively sensitive to a very small proportion of our sample data. We propose a method to assess the sensitivity of applied econometric conclusions to the removal of a small fraction of the sample. Manually checking the influence of all possible small subsets is computationally infeasible, so we use an approximation to find the most influential subset. Our metric, the "Approximate Maximum Influence Perturbation," is based on the classical influence function, and is automatically computable for common methods including (but not limited to) OLS, IV, MLE, G
    
[^12]: 无条件策略影响的识别与估计：一种内生二元处理的无条件MTE方法

    Identification and Estimation of Unconditional Policy Effects of an Endogenous Binary Treatment: an Unconditional MTE Approach. (arXiv:2010.15864v4 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2010.15864](http://arxiv.org/abs/2010.15864)

    本文介绍了一种基于函数影响函数的无条件边际处理效应（MTE）的新方法，研究了处理状态为二元且内生时策略效应的识别与估计。我们证明了无条件策略效应可以表示为对于对待自己的处理状态处于决策困惑的个体而言新定义的无条件MTE的加权平均。对于感兴趣的分位数，我们引入了无条件工具分位数估计（UNIQUE）并证明了其一致性和渐近分布。在实证应用中，我们估计了通过更高的学费补贴引起的大学入学状态变化对工资分布的分位数的影响。

    

    本文研究了在处理状态为二元且内生时的策略效应的识别和估计。我们基于反映政策目标的函数的影响函数引入了一类新的无条件边际处理效应（MTE）。我们证明了无条件策略效应可以表示为对于对待自己的处理状态处于决策困惑的个体而言新定义的无条件MTE的加权平均。我们提供了识别无条件策略效应的条件。当感兴趣的函数是分位数时，我们引入了无条件工具分位数估计（UNIQUE）并证明了其一致性和渐近分布。在实证应用中，我们估计了通过更高的学费补贴引起的大学入学状态变化对工资分布的分位数的影响。

    This paper studies the identification and estimation of policy effects when treatment status is binary and endogenous. We introduce a new class of unconditional marginal treatment effects (MTE) based on the influence function of the functional underlying the policy target. We show that an unconditional policy effect can be represented as a weighted average of the newly defined unconditional MTEs over the individuals who are indifferent about their treatment status. We provide conditions for point identification of the unconditional policy effects. When a quantile is the functional of interest, we introduce the UNconditional Instrumental Quantile Estimator (UNIQUE) and establish its consistency and asymptotic distribution. In the empirical application, we estimate the effect of changing college enrollment status, induced by higher tuition subsidy, on the quantiles of the wage distribution.
    

