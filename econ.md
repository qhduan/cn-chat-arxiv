# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Manipulation Test for Multidimensional RDD](https://arxiv.org/abs/2402.10836) | 本文提出了多维RDD的操纵测试方法，通过理论模型推导出关于条件边际密度的可测试暗示，并基于统计量矢量的二次形式构建测试。 |
| [^2] | [They were robbed! Scoring by the middlemost to attenuate biased judging in boxing](https://arxiv.org/abs/2402.06594) | 通过根据裁判在每个回合中的胜者来决定胜利者的评分方法，可以减少拳击裁判的偏见对比赛结果的影响。 |
| [^3] | [Cutting Feedback in Misspecified Copula Models.](http://arxiv.org/abs/2310.03521) | 该论文介绍了一种在错配的Copula模型中限制反馈的剪切方法，并证明了在只有一个模块错配的情况下，适当的剪切后验提供了准确的不确定性量化。该方法在贝叶斯推断中具有重要的应用。 |
| [^4] | [Inference in Experiments with Matched Pairs and Imperfect Compliance.](http://arxiv.org/abs/2307.13094) | 本文研究了在不完全遵守的随机对照试验中，根据"匹配对"确定治疗状态的局部平均治疗效应的推断，并提出了一种对极限方差的一致估计器。 |
| [^5] | [Asymptotic equivalence of Principal Component and Quasi Maximum Likelihood estimators in Large Approximate Factor Models.](http://arxiv.org/abs/2307.09864) | 在大型近似因子模型中，我们证明主成分估计的因子载荷与准极大似然估计的载荷等价，同时这两种估计也与不可行最小二乘估计的载荷等价。我们还证明了准极大似然估计的协方差矩阵与不可行最小二乘的协方差矩阵等价，从而可以简化准极大似然估计的置信区间的估计过程。所有结果都适用于假设异方差跨截面的一般情况。 |
| [^6] | [COWPEA (Candidates Optimally Weighted in Proportional Election using Approval voting).](http://arxiv.org/abs/2305.08857) | 本文提出了一种新的比例代表方法，COWPEA，可根据候选人不同的权重进行最优选举，并可转换为分数或分级投票方法。 |

# 详细

[^1]: 多维RDD的操纵测试

    Manipulation Test for Multidimensional RDD

    [https://arxiv.org/abs/2402.10836](https://arxiv.org/abs/2402.10836)

    本文提出了多维RDD的操纵测试方法，通过理论模型推导出关于条件边际密度的可测试暗示，并基于统计量矢量的二次形式构建测试。

    

    Lee (2008)提出的因果推断模型适用于回归断点设计(RDD)，依赖于暗示分配（运行）变量的密度连续性的假设。这种假设的测试通常被称为操纵测试，并在应用研究中经常报告，以加强设计的有效性。多维RDD（MRDD）将RDD扩展到治疗分配取决于多个运行变量的情境。本文引入了MRDD的操纵测试。首先，它为MRDD进行因果推断的理论模型，用于推导关于运行变量的条件边际密度的可测试暗示。然后，它基于为每个边际密度单独计算的统计量矢量的二次形式构建了该假设的测试。最后，提出的测试与常用的备选程序进行了比较。

    arXiv:2402.10836v1 Announce Type: new  Abstract: The causal inference model proposed by Lee (2008) for the regression discontinuity design (RDD) relies on assumptions that imply the continuity of the density of the assignment (running) variable. The test for this implication is commonly referred to as the manipulation test and is regularly reported in applied research to strengthen the design's validity. The multidimensional RDD (MRDD) extends the RDD to contexts where treatment assignment depends on several running variables. This paper introduces a manipulation test for the MRDD. First, it develops a theoretical model for causal inference with the MRDD, used to derive a testable implication on the conditional marginal densities of the running variables. Then, it constructs the test for the implication based on a quadratic form of a vector of statistics separately computed for each marginal density. Finally, the proposed test is compared with alternative procedures commonly employed i
    
[^2]: 他们被抢劫了！通过中间值评分减少拳击裁判的偏见

    They were robbed! Scoring by the middlemost to attenuate biased judging in boxing

    [https://arxiv.org/abs/2402.06594](https://arxiv.org/abs/2402.06594)

    通过根据裁判在每个回合中的胜者来决定胜利者的评分方法，可以减少拳击裁判的偏见对比赛结果的影响。

    

    拳击长期存在着偏见评判的问题，不仅影响职业拳击赛事，也影响奥运比赛。所谓的“抢劫”现象让拳击运动的粉丝和运动员们对此产生了疑虑。为了解决这个问题，我们提出了一种简化的拳击评分调整方法：根据裁判在每个回合中的胜者来决定胜利者，而不是依靠裁判对整场比赛的总体评分。这种方法基于社会选择理论，采用多数规则和中间值聚合函数，对偏袒裁判构成协调问题，减弱其影响力。我们的模型分析和模拟结果表明，这种方法可以显著减少偏袒裁判影响比赛结果的可能性。

    Boxing has a long-standing problem with biased judging, impacting both professional and Olympic bouts. ''Robberies'', where boxers are widely seen as being denied rightful victories, threaten to drive fans and athletes away from the sport. To tackle this problem, we propose a minimalist adjustment in how boxing is scored: the winner would be decided by the majority of round-by-round victories according to the judges, rather than relying on the judges' overall bout scores. This approach, rooted in social choice theory and utilising majority rule and middlemost aggregation functions, creates a coordination problem for partisan judges and attenuates their influence. Our model analysis and simulations demonstrate the potential to significantly decrease the likelihood of a partisan judge swaying the result of a bout.
    
[^3]: 在错配的Copula模型中限制反馈的剪切方法

    Cutting Feedback in Misspecified Copula Models. (arXiv:2310.03521v1 [stat.ME])

    [http://arxiv.org/abs/2310.03521](http://arxiv.org/abs/2310.03521)

    该论文介绍了一种在错配的Copula模型中限制反馈的剪切方法，并证明了在只有一个模块错配的情况下，适当的剪切后验提供了准确的不确定性量化。该方法在贝叶斯推断中具有重要的应用。

    

    在Copula模型中，边缘分布和Copula函数被分别指定。我们将它们视为模块化贝叶斯推断框架中的两个模块，并提出通过“剪切反馈”进行修改的贝叶斯推断方法。剪切反馈限制了后验推断中潜在错配模块的影响。我们考虑两种类型的剪切方法。第一种限制了错配Copula对边缘推断的影响，这是流行的边际推断（IFM）估计的贝叶斯类似方法。第二种通过使用秩似然定义剪切模型来限制错配边缘对Copula参数推断的影响。我们证明，如果只有一个模块错配，那么适当的剪切后验在另一个模块的参数的渐近不确定性量化方面是准确的。计算剪切后验很困难，我们提出了新的变分推断方法来解决这个问题。

    In copula models the marginal distributions and copula function are specified separately. We treat these as two modules in a modular Bayesian inference framework, and propose conducting modified Bayesian inference by ``cutting feedback''. Cutting feedback limits the influence of potentially misspecified modules in posterior inference. We consider two types of cuts. The first limits the influence of a misspecified copula on inference for the marginals, which is a Bayesian analogue of the popular Inference for Margins (IFM) estimator. The second limits the influence of misspecified marginals on inference for the copula parameters by using a rank likelihood to define the cut model. We establish that if only one of the modules is misspecified, then the appropriate cut posterior gives accurate uncertainty quantification asymptotically for the parameters in the other module. Computation of the cut posteriors is difficult, and new variational inference methods to do so are proposed. The effic
    
[^4]: 匹配对和不完全遵守下的实验推断

    Inference in Experiments with Matched Pairs and Imperfect Compliance. (arXiv:2307.13094v1 [econ.EM])

    [http://arxiv.org/abs/2307.13094](http://arxiv.org/abs/2307.13094)

    本文研究了在不完全遵守的随机对照试验中，根据"匹配对"确定治疗状态的局部平均治疗效应的推断，并提出了一种对极限方差的一致估计器。

    

    本文研究了在不完全遵守的随机对照试验中，根据“匹配对”确定治疗状态的局部平均治疗效应的推断。通过“匹配对”，我们指的是从感兴趣的总体中独立和随机抽取单位，根据观察到的基线协变量进行配对，然后在每个对中，随机选择一个单位进行治疗。在对匹配质量进行的弱假设下，我们首先推导了传统的Wald（即二阶最小二乘）估计器的局部平均治疗效应的极限行为。我们进一步显示，传统的异方差性稳健估计器的极限方差通常是保守的，即其可能性极限比极限方差（通常严格地）大。因此，我们提供了一种对所需数量一致的极限方差的替代估计器。最后，我们考虑了额外观察到的基线协变量的使用。

    This paper studies inference for the local average treatment effect in randomized controlled trials with imperfect compliance where treatment status is determined according to "matched pairs." By "matched pairs," we mean that units are sampled i.i.d. from the population of interest, paired according to observed, baseline covariates and finally, within each pair, one unit is selected at random for treatment. Under weak assumptions governing the quality of the pairings, we first derive the limiting behavior of the usual Wald (i.e., two-stage least squares) estimator of the local average treatment effect. We show further that the conventional heteroskedasticity-robust estimator of its limiting variance is generally conservative in that its limit in probability is (typically strictly) larger than the limiting variance. We therefore provide an alternative estimator of the limiting variance that is consistent for the desired quantity. Finally, we consider the use of additional observed, base
    
[^5]: 大型近似因子模型中主成分和准极大似然估计量的渐近等价性分析

    Asymptotic equivalence of Principal Component and Quasi Maximum Likelihood estimators in Large Approximate Factor Models. (arXiv:2307.09864v1 [econ.EM])

    [http://arxiv.org/abs/2307.09864](http://arxiv.org/abs/2307.09864)

    在大型近似因子模型中，我们证明主成分估计的因子载荷与准极大似然估计的载荷等价，同时这两种估计也与不可行最小二乘估计的载荷等价。我们还证明了准极大似然估计的协方差矩阵与不可行最小二乘的协方差矩阵等价，从而可以简化准极大似然估计的置信区间的估计过程。所有结果都适用于假设异方差跨截面的一般情况。

    

    我们证明在一个$n$维的稳定时间序列向量的近似因子模型中，通过主成分估计的因子载荷在$n\to\infty$时与准极大似然估计得到的载荷等价。这两种估计量在$n\to\infty$时也与如果观察到因子时的不可行最小二乘估计等价。我们还证明了准极大似然估计的渐近协方差矩阵的传统三明治形式与不可行最小二乘的简单渐近协方差矩阵等价。这提供了一种简单的方法来估计准极大似然估计的渐近置信区间，而不需要估计复杂的海森矩阵和费谢尔信息矩阵。所有结果均适用于假设异方差跨截面的一般情况。

    We prove that in an approximate factor model for an $n$-dimensional vector of stationary time series the factor loadings estimated via Principal Components are asymptotically equivalent, as $n\to\infty$, to those estimated by Quasi Maximum Likelihood. Both estimators are, in turn, also asymptotically equivalent, as $n\to\infty$, to the unfeasible Ordinary Least Squares estimator we would have if the factors were observed. We also show that the usual sandwich form of the asymptotic covariance matrix of the Quasi Maximum Likelihood estimator is asymptotically equivalent to the simpler asymptotic covariance matrix of the unfeasible Ordinary Least Squares. This provides a simple way to estimate asymptotic confidence intervals for the Quasi Maximum Likelihood estimator without the need of estimating the Hessian and Fisher information matrices whose expressions are very complex. All our results hold in the general case in which the idiosyncratic components are cross-sectionally heteroskedast
    
[^6]: COWPEA（候选人按比例使用赞成投票进行最优加权）：一种新的比例代表方法

    COWPEA (Candidates Optimally Weighted in Proportional Election using Approval voting). (arXiv:2305.08857v1 [econ.TH])

    [http://arxiv.org/abs/2305.08857](http://arxiv.org/abs/2305.08857)

    本文提出了一种新的比例代表方法，COWPEA，可根据候选人不同的权重进行最优选举，并可转换为分数或分级投票方法。

    

    本文描述了一种使用赞成投票的比例代表新方法，称为COWPEA（候选人按比例使用赞成投票进行最优加权）。COWPEA在选择一定数量的候选人时，可以根据其不同的权重进行最优选举，而不是只给予固定数量相同的权重。COWPEA Lottery 是一个不确定性的版本，可以选择一定数量的候选人，并使它们拥有相等的权重。COWPEA是唯一已知通过单调性、与无关选票和普遍喜欢的候选人标准的比例方法。同时，也有方法可以将COWPEA和COWPEA Lottery转换为分数或分级投票方法。

    This paper describes a new method of proportional representation that uses approval voting, known as COWPEA (Candidates Optimally Weighted in Proportional Election using Approval voting). COWPEA optimally elects an unlimited number of candidates with potentially different weights to a body, rather than giving a fixed number equal weight. A version that elects a fixed a number of candidates with equal weight does exist, but it is non-deterministic, and is known as COWPEA Lottery. This is the only proportional method known to pass monotonicity, Independence of Irrelevant Ballots, and the Universally Liked Candidate criterion. There are also ways to convert COWPEA and COWPEA Lottery to a score or graded voting method.
    

