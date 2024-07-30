# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Privacy-Protected Spatial Autoregressive Model](https://arxiv.org/abs/2403.16773) | 提出了一种隐私保护的空间自回归模型，引入了噪声响应和协变量以满足隐私保护要求，并开发了纠正由噪声引入的偏差的技术。 |
| [^2] | [The Power of Simple Menus in Robust Selling Mechanisms.](http://arxiv.org/abs/2310.17392) | 本研究致力于寻找简单的销售机制，以有效对冲市场模糊性。我们发现，只有有限数量的价格随机化的销售机制已经能够获得无限选项的最优稳健机制所实现的显著效益。 |
| [^3] | [Improving Robust Decisions with Data.](http://arxiv.org/abs/2310.16281) | 该论文研究了如何通过数据改善鲁棒决策，并开发了简单易实现的推理方法以保证改善。 |
| [^4] | [Adaptive maximization of social welfare.](http://arxiv.org/abs/2310.09597) | 论文研究了通过适应性策略选择最大化社会福利的问题，并提供了关于遗憾的下界和算法的匹配上界。研究发现福利最大化比多臂老虎机问题更困难，但该算法达到了最优增长速率。 |
| [^5] | [Nonparametric Causal Decomposition of Group Disparities.](http://arxiv.org/abs/2306.16591) | 本论文提出了一个因果框架，通过一个中间处理变量将组差异分解成不同的组成部分，并提供了一个新的解释和改善差异的机制。该框架改进了经典的分解方法，并且可以指导政策干预。 |
| [^6] | [Bayesian Opponent Modeling in Multiplayer Imperfect-Information Games.](http://arxiv.org/abs/2212.06027) | 该论文提出了一种针对多人不完全信息博弈的贝叶斯对手建模方法，在三人 Kuhn poker 中应用这种方法可以明显超过所有的代理商，包括准确的纳什均衡策略。 |
| [^7] | [Security Issuance, Institutional Investors and Quid Pro Quo: Insights from SPACs.](http://arxiv.org/abs/2211.16643) | SPAC中的优质投资者参与可以创造价值，非优质投资者则仅参与交换条件安排，这不仅仅是代理成本还能使更多公司上市。 |
| [^8] | [Optimal Pre-Analysis Plans: Statistical Decisions Subject to Implementability.](http://arxiv.org/abs/2208.09638) | 本研究提出了一个委托-代理模型来解决先分析计划的设计问题，通过实施具有可行性的统计决策规则，发现先分析计划在实施中起着重要作用，特别是对于假设检验来说，最优拒绝规则需要预先注册有效的检验并对未报告的数据做最坏情况假设。 |
| [^9] | [The Transfer Performance of Economic Models.](http://arxiv.org/abs/2202.04796) | 该论文提供了一个易于处理的“跨领域”预测问题，并根据模型在新领域中的表现定义了模型的迁移误差。该论文比较了经济模型和黑盒算法在预测确定等价值方面的迁移能力，发现经济模型在跨领域上的泛化能力更强。 |

# 详细

[^1]: 隐私保护的空间自回归模型

    Privacy-Protected Spatial Autoregressive Model

    [https://arxiv.org/abs/2403.16773](https://arxiv.org/abs/2403.16773)

    提出了一种隐私保护的空间自回归模型，引入了噪声响应和协变量以满足隐私保护要求，并开发了纠正由噪声引入的偏差的技术。

    

    空间自回归（SAR）模型是研究网络效应的重要工具。然而，随着对数据隐私的重视增加，数据提供者经常实施隐私保护措施，使传统的SAR模型变得不适用。在本研究中，我们介绍了一种带有添加噪声响应和协变量的隐私保护的SAR模型，以满足隐私保护要求。然而，在这种情况下，由于无法建立似然函数，传统的拟最大似然估计变得不可行。为了解决这个问题，我们首先考虑了只有噪声添加响应的似然函数的显式表达。然而，由于协变量中的噪声，导数是有偏的。因此，我们开发了可以纠正噪声引入的偏差的技术。相应地，提出了一种类似牛顿-拉弗森的算法来获得估计量，从而导致一个修正的似然估计量。

    arXiv:2403.16773v1 Announce Type: cross  Abstract: Spatial autoregressive (SAR) models are important tools for studying network effects. However, with an increasing emphasis on data privacy, data providers often implement privacy protection measures that make classical SAR models inapplicable. In this study, we introduce a privacy-protected SAR model with noise-added response and covariates to meet privacy-protection requirements. However, in this scenario, the traditional quasi-maximum likelihood estimator becomes infeasible because the likelihood function cannot be formulated. To address this issue, we first consider an explicit expression for the likelihood function with only noise-added responses. However, the derivatives are biased owing to the noise in the covariates. Therefore, we develop techniques that can correct the biases introduced by noise. Correspondingly, a Newton-Raphson-type algorithm is proposed to obtain the estimator, leading to a corrected likelihood estimator. To
    
[^2]: 简单菜单在稳健销售机制中的力量

    The Power of Simple Menus in Robust Selling Mechanisms. (arXiv:2310.17392v1 [econ.TH])

    [http://arxiv.org/abs/2310.17392](http://arxiv.org/abs/2310.17392)

    本研究致力于寻找简单的销售机制，以有效对冲市场模糊性。我们发现，只有有限数量的价格随机化的销售机制已经能够获得无限选项的最优稳健机制所实现的显著效益。

    

    我们研究了一个稳健销售问题，其中卖方试图将一个物品卖给一个买方，但对买方的估值分布存在不确定性。现有文献表明，稳健机制设计比稳健确定性定价提供了更强的理论保证。同时，稳健机制设计的卓越性能以卖方提供具有无限选项的菜单，每个选项与抽奖和买方选择的付款方式相配。鉴于此，我们的研究的主要重点是寻找可以有效对冲市场模糊性的简单销售机制。我们表明，一个具有小菜单大小（或有限数量的价格随机化）的销售机制已经能够获得无限选项的最优稳健机制所实现的显著效益。特别是，我们发展了一个通用框架来研究稳健销售机制问题。

    We study a robust selling problem where a seller attempts to sell one item to a buyer but is uncertain about the buyer's valuation distribution. Existing literature indicates that robust mechanism design provides a stronger theoretical guarantee than robust deterministic pricing. Meanwhile, the superior performance of robust mechanism design comes at the expense of implementation complexity given that the seller offers a menu with an infinite number of options, each coupled with a lottery and a payment for the buyer's selection. In view of this, the primary focus of our research is to find simple selling mechanisms that can effectively hedge against market ambiguity. We show that a selling mechanism with a small menu size (or limited randomization across a finite number of prices) is already capable of deriving significant benefits achieved by the optimal robust mechanism with infinite options. In particular, we develop a general framework to study the robust selling mechanism problem 
    
[^3]: 通过数据改善鲁棒决策

    Improving Robust Decisions with Data. (arXiv:2310.16281v1 [econ.TH])

    [http://arxiv.org/abs/2310.16281](http://arxiv.org/abs/2310.16281)

    该论文研究了如何通过数据改善鲁棒决策，并开发了简单易实现的推理方法以保证改善。

    

    决策者面临由数据生成过程(DGP)控制的不确定性，这些过程可能只属于一组独立但可能非相同分布的序列。鲁棒决策在这个集合中最大化决策者对最坏情况DGP的预期收益。本文研究了如何通过数据改善这些鲁棒决策，其中改善通过真实DGP下的预期收益来衡量。本文完全描述了在所有可能的DGP下保证这种改善的时间和方式，并开发了推理方法来实现它。这些推理方法是必需的，因为本文表明，常见的推理方法（如最大似然或贝叶斯）通常无法实现这种改善。重要的是，开发的推理方法是通过对标准推理程序进行简单扩展获得的，因此在实践中很容易实现。

    A decision-maker (DM) faces uncertainty governed by a data-generating process (DGP), which is only known to belong to a set of sequences of independent but possibly non-identical distributions. A robust decision maximizes the DM's expected payoff against the worst possible DGP in this set. This paper studies how such robust decisions can be improved with data, where improvement is measured by expected payoff under the true DGP. In this paper, I fully characterize when and how such an improvement can be guaranteed under all possible DGPs and develop inference methods to achieve it. These inference methods are needed because, as this paper shows, common inference methods (e.g., maximum likelihood or Bayesian) often fail to deliver such an improvement. Importantly, the developed inference methods are given by simple augmentations to standard inference procedures, and are thus easy to implement in practice.
    
[^4]: 自适应最大化社会福利

    Adaptive maximization of social welfare. (arXiv:2310.09597v1 [econ.EM])

    [http://arxiv.org/abs/2310.09597](http://arxiv.org/abs/2310.09597)

    论文研究了通过适应性策略选择最大化社会福利的问题，并提供了关于遗憾的下界和算法的匹配上界。研究发现福利最大化比多臂老虎机问题更困难，但该算法达到了最优增长速率。

    

    我们考虑了重复选择政策以最大化社会福利的问题。福利是个人效用和公共收入的加权和。早期的结果影响后续的政策选择。效用不可观测，但可以间接推断。响应函数通过实验学习获得。我们推导出了一个关于遗憾的下界，并且对于一种Exp3算法的匹配对策对立上界。累积遗憾以$T^{2/3}$的速率增长。这意味着(i)福利最大化比多臂老虎机问题更困难（对于有限的政策集来说，增长速率为$T^{1/2}$），和(ii)我们的算法实现了最优增长速率。对于随机设置，如果社会福利是凹的，我们可以使用二分搜索算法在连续政策集上实现$T^{1/2}$的速率。我们分析了非线性收入税扩展，并概述了商品税扩展。我们将我们的设置与垄断定价（更容易）和双边交易的定价进行了比较。

    We consider the problem of repeatedly choosing policies to maximize social welfare. Welfare is a weighted sum of private utility and public revenue. Earlier outcomes inform later policies. Utility is not observed, but indirectly inferred. Response functions are learned through experimentation.  We derive a lower bound on regret, and a matching adversarial upper bound for a variant of the Exp3 algorithm. Cumulative regret grows at a rate of $T^{2/3}$. This implies that (i) welfare maximization is harder than the multi-armed bandit problem (with a rate of $T^{1/2}$ for finite policy sets), and (ii) our algorithm achieves the optimal rate. For the stochastic setting, if social welfare is concave, we can achieve a rate of $T^{1/2}$ (for continuous policy sets), using a dyadic search algorithm.  We analyze an extension to nonlinear income taxation, and sketch an extension to commodity taxation. We compare our setting to monopoly pricing (which is easier), and price setting for bilateral tra
    
[^5]: 非参数因果分解组差异

    Nonparametric Causal Decomposition of Group Disparities. (arXiv:2306.16591v1 [stat.ME])

    [http://arxiv.org/abs/2306.16591](http://arxiv.org/abs/2306.16591)

    本论文提出了一个因果框架，通过一个中间处理变量将组差异分解成不同的组成部分，并提供了一个新的解释和改善差异的机制。该框架改进了经典的分解方法，并且可以指导政策干预。

    

    我们提出了一个因果框架来将结果中的组差异分解为中间处理变量。我们的框架捕捉了基线潜在结果、处理前沿、平均处理效应和处理选择的组差异的贡献。这个框架以反事实的方式进行了数学表达，并且能够方便地指导政策干预。特别是，针对不同的处理选择进行的分解部分是特别新颖的，揭示了一种解释和改善差异的新机制。这个框架以因果术语重新定义了经典的Kitagawa-Blinder-Oaxaca分解，通过解释组差异而不是组效应来补充了因果中介分析，并解决了近期随机等化分解的概念困难。我们还提供了一个条件分解，允许研究人员在定义评估和相应的干预措施时纳入协变量。

    We propose a causal framework for decomposing a group disparity in an outcome in terms of an intermediate treatment variable. Our framework captures the contributions of group differences in baseline potential outcome, treatment prevalence, average treatment effect, and selection into treatment. This framework is counterfactually formulated and readily informs policy interventions. The decomposition component for differential selection into treatment is particularly novel, revealing a new mechanism for explaining and ameliorating disparities. This framework reformulates the classic Kitagawa-Blinder-Oaxaca decomposition in causal terms, supplements causal mediation analysis by explaining group disparities instead of group effects, and resolves conceptual difficulties of recent random equalization decompositions. We also provide a conditional decomposition that allows researchers to incorporate covariates in defining the estimands and corresponding interventions. We develop nonparametric
    
[^6]: 多人不完全信息博弈中的贝叶斯对手建模

    Bayesian Opponent Modeling in Multiplayer Imperfect-Information Games. (arXiv:2212.06027v2 [cs.GT] UPDATED)

    [http://arxiv.org/abs/2212.06027](http://arxiv.org/abs/2212.06027)

    该论文提出了一种针对多人不完全信息博弈的贝叶斯对手建模方法，在三人 Kuhn poker 中应用这种方法可以明显超过所有的代理商，包括准确的纳什均衡策略。

    

    在许多现实世界的情境中，代理商与多个对立代理商进行战略互动，对手可能采用各种策略。对于这样的情境，设计代理的标准方法是计算或逼近相关的博弈理论解，如纳什均衡，然后遵循规定的策略。然而，这样的策略忽略了对手玩游戏的任何观察，这些观察可能表明可以利用的缺点。我们提出了一种多人不完全信息博弈中的对手建模方法，通过重复交互收集对手玩游戏的观察。我们对三人 Kuhn 扑克展开了对许多真实对手和准确的纳什均衡策略的实验，结果表明我们的算法明显优于所有的代理商，包括准确的纳什均衡策略。

    In many real-world settings agents engage in strategic interactions with multiple opposing agents who can employ a wide variety of strategies. The standard approach for designing agents for such settings is to compute or approximate a relevant game-theoretic solution concept such as Nash equilibrium and then follow the prescribed strategy. However, such a strategy ignores any observations of opponents' play, which may indicate shortcomings that can be exploited. We present an approach for opponent modeling in multiplayer imperfect-information games where we collect observations of opponents' play through repeated interactions. We run experiments against a wide variety of real opponents and exact Nash equilibrium strategies in three-player Kuhn poker and show that our algorithm significantly outperforms all of the agents, including the exact Nash equilibrium strategies.
    
[^7]: SPAC的发行、机构投资者和交换条件：洞察力

    Security Issuance, Institutional Investors and Quid Pro Quo: Insights from SPACs. (arXiv:2211.16643v3 [q-fin.GN] UPDATED)

    [http://arxiv.org/abs/2211.16643](http://arxiv.org/abs/2211.16643)

    SPAC中的优质投资者参与可以创造价值，非优质投资者则仅参与交换条件安排，这不仅仅是代理成本还能使更多公司上市。

    

    证券发行受信息和代理成本的影响。然而，很难将它们分开考虑。我们考虑SPAC并分别评估这些影响。为此，我们确定产生价值相关信息的优质投资者：他们的参与与更高的SPAC成功和公告日回报相关。然而，非优质投资者仅参与交换条件安排。他们今天从发行人（优惠）获得高回报，这意味着他们更有可能参与由这些发行人发起的低回报未来交易（条件）。因此，交换条件不是纯代理成本，而是部分使更多的公司上市的转移。

    Security issuance is subject to informational and agency-related frictions. However, it is difficult to separate their effects. We consider SPACs and assess those effects separately. To this end, we identify premium investors who produce value-relevant information: their participation is associated with higher SPAC success and announcement-day returns. Non-premium investors, however, engage only in quid pro quo arrangements. Their high returns today from issuers (quid) mean they are more likely to participate in low-returns future deals initiated by those issuers (quo). Thus quid pro quo is not pure agency cost but partly a transfer that enables more firms to go public.
    
[^8]: 最优先分析计划：受限于可实施性的统计决策

    Optimal Pre-Analysis Plans: Statistical Decisions Subject to Implementability. (arXiv:2208.09638v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2208.09638](http://arxiv.org/abs/2208.09638)

    本研究提出了一个委托-代理模型来解决先分析计划的设计问题，通过实施具有可行性的统计决策规则，发现先分析计划在实施中起着重要作用，特别是对于假设检验来说，最优拒绝规则需要预先注册有效的检验并对未报告的数据做最坏情况假设。

    

    什么是先分析计划的目的，应该如何设计？我们提出了一个委托-代理模型，其中决策者依赖于分析师选择性但真实的报告。分析师具有数据访问权，且目标不一致。在这个模型中，实施统计决策规则（检验、估计量）需要一个激励兼容机制。我们首先描述了哪些决策规则可以被实施。然后描述了受限于可实施性的最优统计决策规则。我们表明实施需要先分析计划。重点放在假设检验上，我们表明最优拒绝规则预先注册了一个对于所有数据报告的有效检验，并对未报告的数据做最坏情况假设。最优检验可以通过线性规划问题的解来找到。

    What is the purpose of pre-analysis plans, and how should they be designed? We propose a principal-agent model where a decision-maker relies on selective but truthful reports by an analyst. The analyst has data access, and non-aligned objectives. In this model, the implementation of statistical decision rules (tests, estimators) requires an incentive-compatible mechanism. We first characterize which decision rules can be implemented. We then characterize optimal statistical decision rules subject to implementability. We show that implementation requires pre-analysis plans. Focussing specifically on hypothesis tests, we show that optimal rejection rules pre-register a valid test for the case when all data is reported, and make worst-case assumptions about unreported data. Optimal tests can be found as a solution to a linear-programming problem.
    
[^9]: 经济模型的迁移表现

    The Transfer Performance of Economic Models. (arXiv:2202.04796v3 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2202.04796](http://arxiv.org/abs/2202.04796)

    该论文提供了一个易于处理的“跨领域”预测问题，并根据模型在新领域中的表现定义了模型的迁移误差。该论文比较了经济模型和黑盒算法在预测确定等价值方面的迁移能力，发现经济模型在跨领域上的泛化能力更强。

    

    经济学家经常使用特定领域的数据来估计模型，例如在特定对象组中估计风险偏好或特定彩票类别上估计风险偏好。模型预测是否跨领域适用取决于估计模型是否捕捉到可推广结构。我们提供了一个易于处理的“跨领域”预测问题，并根据模型在新领域中的表现定义了模型的迁移误差。我们推导出有限样本预测区间，当领域独立同分布时，保证以用户选择的概率包含实现的迁移误差，并使用这些区间比较经济模型和黑盒算法在预测确定等价值方面的迁移能力。我们发现在这个应用程序中，我们考虑的黑盒算法在相同领域的数据上估计和测试时优于标准经济模型，但经济模型在跨领域上的泛化能力更强。

    Economists often estimate models using data from a particular domain, e.g. estimating risk preferences in a particular subject pool or for a specific class of lotteries. Whether a model's predictions extrapolate well across domains depends on whether the estimated model has captured generalizable structure. We provide a tractable formulation for this "out-of-domain" prediction problem and define the transfer error of a model based on how well it performs on data from a new domain. We derive finite-sample forecast intervals that are guaranteed to cover realized transfer errors with a user-selected probability when domains are iid, and use these intervals to compare the transferability of economic models and black box algorithms for predicting certainty equivalents. We find that in this application, the black box algorithms we consider outperform standard economic models when estimated and tested on data from the same domain, but the economic models generalize across domains better than 
    

