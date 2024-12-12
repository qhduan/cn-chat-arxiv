# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimistic and pessimistic approaches for cooperative games](https://arxiv.org/abs/2403.01442) | 本文澄清了乐观和悲观方法的解释，展示了确保联盟不超过其乐观上界至少和保证其悲观下界一样困难的关系。 |
| [^2] | [Mean-variance constrained priors have finite maximum Bayes risk in the normal location model.](http://arxiv.org/abs/2303.08653) | 该研究证明了在正态位置模型中，限制了均值方差的先验，在贝叶斯风险下具有有限的最大误差，结果适用于各种先验和方差。 |
| [^3] | [Robust Estimation and Inference in Panels with Interactive Fixed Effects.](http://arxiv.org/abs/2210.06639) | 本文研究了具有交互固定效应的面板数据中回归系数的估计和推断问题。通过采用改进的估计器和偏倚感知置信区间，我们能够解决因素弱引起的偏倚和大小失真的问题，无论因素是否强壮，都能得到统一有效的结果。 |
| [^4] | [On the Robustness of Second-Price Auctions in Prior-Independent Mechanism Design.](http://arxiv.org/abs/2204.10478) | 这篇论文研究了在先验独立机制设计中，对于没有可用先验信息的情况下，卖方如何以最坏情况的遗憾度量来衡量机制的性能。通过研究多种估值分布类别，论文总结了关于放松先验假设的先验独立机制设计的可行性。 |

# 详细

[^1]: 合作博弈的乐观与悲观方法

    Optimistic and pessimistic approaches for cooperative games

    [https://arxiv.org/abs/2403.01442](https://arxiv.org/abs/2403.01442)

    本文澄清了乐观和悲观方法的解释，展示了确保联盟不超过其乐观上界至少和保证其悲观下界一样困难的关系。

    

    合作博弈理论旨在研究如何划分一组玩家共同创造的价值。这些游戏通常通过可转让效用的特征函数形式进行研究，该形式代表了每个联盟可获得的价值。在存在外部性的情况下，有许多定义这个价值的方式。已经研究了各种模型，考虑了不同水平的玩家合作以及外部玩家对联盟价值的影响。尽管存在不同的方法，但通常，乐观和悲观方法为战略互动提供了充分的见解。本文通过提供统一框架来澄清这些方法的解释。我们表明，确保没有任何联盟获得多于他们（乐观）上界的是至少和保证他们（悲观）下界一样困难的。我们还表明，如果外部性是负面的，提供

    arXiv:2403.01442v1 Announce Type: new  Abstract: Cooperative game theory aims to study how to divide a joint value created by a set of players. These games are often studied through the characteristic function form with transferable utility, which represents the value obtainable by each coalition. In the presence of externalities, there are many ways to define this value. Various models that account for different levels of player cooperation and the influence of external players on coalition value have been studied. Although there are different approaches, typically, the optimistic and pessimistic approaches provide sufficient insights into strategic interactions. This paper clarifies the interpretation of these approaches by providing a unified framework. We show that making sure that no coalition receives more than their (optimistic) upper bounds is always at least as difficult as guaranteeing their (pessimistic) lower bounds. We also show that if externalities are negative, providin
    
[^2]: 正态位置模型中的均值-方差受限先验有有限的最大贝叶斯风险

    Mean-variance constrained priors have finite maximum Bayes risk in the normal location model. (arXiv:2303.08653v1 [math.ST])

    [http://arxiv.org/abs/2303.08653](http://arxiv.org/abs/2303.08653)

    该研究证明了在正态位置模型中，限制了均值方差的先验，在贝叶斯风险下具有有限的最大误差，结果适用于各种先验和方差。

    

    考虑一个正态位置模型，其中 $X \mid \theta \sim N(\theta, \sigma^2)$，$\sigma^2$已知。假设 $\theta \sim G_0$，其中先验 $G_0$ 具有零均值和单位方差。令 $G_1$ 为可能存在误差的零均值和单位方差的先验。我们表明，在 $G_0, G_1, \sigma^2 > 0$ 范围内，贝叶斯风险下的后验均值的平方误差有界。

    Consider a normal location model $X \mid \theta \sim N(\theta, \sigma^2)$ with known $\sigma^2$. Suppose $\theta \sim G_0$, where the prior $G_0$ has zero mean and unit variance. Let $G_1$ be a possibly misspecified prior with zero mean and unit variance. We show that the squared error Bayes risk of the posterior mean under $G_1$ is bounded, uniformly over $G_0, G_1, \sigma^2 > 0$.
    
[^3]: 具有交互固定效应的面板数据中的鲁棒估计和推断

    Robust Estimation and Inference in Panels with Interactive Fixed Effects. (arXiv:2210.06639v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2210.06639](http://arxiv.org/abs/2210.06639)

    本文研究了具有交互固定效应的面板数据中回归系数的估计和推断问题。通过采用改进的估计器和偏倚感知置信区间，我们能够解决因素弱引起的偏倚和大小失真的问题，无论因素是否强壮，都能得到统一有效的结果。

    

    本文考虑具有交互固定效应（即具有因子结构）的面板数据中回归系数的估计和推断问题。我们发现之前开发的估计器和置信区间可能在一些因素较弱的情况下存在严重的偏倚和大小失真。我们提出了具有改进收敛速度和偏倚感知置信区间的估计器，无论因素是否强壮都能保持统一有效。我们的方法采用最小化线性估计理论，在初始交互固定效应的误差上使用核范数约束来形成一个无偏估计。我们利用所得估计构建一个考虑到因素弱引起的剩余偏差的偏倚感知置信区间。在蒙特卡洛实验中，我们发现在因素较弱的情况下相较于传统方法有显著改进，并且在因素较强的情况下几乎没有估计误差的损失。

    We consider estimation and inference for a regression coefficient in panels with interactive fixed effects (i.e., with a factor structure). We show that previously developed estimators and confidence intervals (CIs) might be heavily biased and size-distorted when some of the factors are weak. We propose estimators with improved rates of convergence and bias-aware CIs that are uniformly valid regardless of whether the factors are strong or not. Our approach applies the theory of minimax linear estimation to form a debiased estimate using a nuclear norm bound on the error of an initial estimate of the interactive fixed effects. We use the obtained estimate to construct a bias-aware CI taking into account the remaining bias due to weak factors. In Monte Carlo experiments, we find a substantial improvement over conventional approaches when factors are weak, with little cost to estimation error when factors are strong.
    
[^4]: 《关于先验独立机制设计中的二价竞拍的鲁棒性》

    On the Robustness of Second-Price Auctions in Prior-Independent Mechanism Design. (arXiv:2204.10478v4 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2204.10478](http://arxiv.org/abs/2204.10478)

    这篇论文研究了在先验独立机制设计中，对于没有可用先验信息的情况下，卖方如何以最坏情况的遗憾度量来衡量机制的性能。通过研究多种估值分布类别，论文总结了关于放松先验假设的先验独立机制设计的可行性。

    

    经典的贝叶斯机制设计依赖于公共先验假设，但在实践中往往无法获得这样的先验知识。我们研究了放松这个假设的先验独立机制的设计：卖方向n个买家出售一个不可分割的物品，买家的估值从一个对卖方和买家都未知的联合分布中抽取；买家不需要形成对竞争对手的信念，而卖方假设该分布是从一个指定类中对抗性选择的。我们通过最坏情况的遗憾度量性能，即预期收入与实际机制收入之间的差异。我们研究了一组广泛的估值分布类，涵盖了各种可能的依赖关系：独立同分布（i.i.d.）分布、i.i.d.分布的混合分布、关联和可交换分布、可交换分布等。

    Classical Bayesian mechanism design relies on the common prior assumption, but such prior is often not available in practice. We study the design of prior-independent mechanisms that relax this assumption: the seller is selling an indivisible item to $n$ buyers such that the buyers' valuations are drawn from a joint distribution that is unknown to both the buyers and the seller; buyers do not need to form beliefs about competitors, and the seller assumes the distribution is adversarially chosen from a specified class. We measure performance through the worst-case regret, or the difference between the expected revenue achievable with perfect knowledge of buyers' valuations and the actual mechanism revenue.  We study a broad set of classes of valuation distributions that capture a wide spectrum of possible dependencies: independent and identically distributed (i.i.d.) distributions, mixtures of i.i.d. distributions, affiliated and exchangeable distributions, exchangeable distributions, a
    

