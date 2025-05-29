# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal convex $M$-estimation via score matching](https://arxiv.org/abs/2403.16688) | 该论文提出了一种通过得分匹配实现最佳凸$M$-估计的方法，在线性回归中能够达到最佳的渐近方差，并且在计算上高效，证明具有所有凸$M$-估计中最小的渐近协方差。 |
| [^2] | [LC-Tsalis-INF: Generalized Best-of-Both-Worlds Linear Contextual Bandits](https://arxiv.org/abs/2403.03219) | 该研究提出了一种广义最佳双赢线性背景强化型赌博机算法，能够在次优性差距受到下界限制时遗憾为$O(\log(T))$。同时引入了边缘条件来描述次优性差距对问题难度的影响。 |
| [^3] | [Causal Inference with Differentially Private (Clustered) Outcomes.](http://arxiv.org/abs/2308.00957) | 本文提出了一种新的差分隐私机制"Cluster-DP"，它在保证隐私的同时利用数据的聚类结构，从而实现了更强的隐私保证和较低的方差，可以用于进行因果分析。 |

# 详细

[^1]: 通过得分匹配实现最佳凸$M$-估计

    Optimal convex $M$-estimation via score matching

    [https://arxiv.org/abs/2403.16688](https://arxiv.org/abs/2403.16688)

    该论文提出了一种通过得分匹配实现最佳凸$M$-估计的方法，在线性回归中能够达到最佳的渐近方差，并且在计算上高效，证明具有所有凸$M$-估计中最小的渐近协方差。

    

    在线性回归的背景下，我们构建了一个数据驱动的凸损失函数，通过该函数进行经验风险最小化可以在回归系数的下游估计中实现最佳的渐近方差。我们的半参数方法旨在最佳逼近噪声分布对数密度的导数。在总体层面上，这个拟合过程是对得分匹配的非参数拓展，对应于根据Fisher散度进行噪声分布的对数凹映射。该过程在计算上是高效的，我们证明我们的程序达到了所有凸$M$-估计中最小的渐近协方差。作为非对数凹设置的一个例子，对于柯西误差，最佳凸损失函数类似于Huber函数，并且我们的过程相对于oracle最大似然估计器实现了大于0.87的渐近效率。

    arXiv:2403.16688v1 Announce Type: cross  Abstract: In the context of linear regression, we construct a data-driven convex loss function with respect to which empirical risk minimisation yields optimal asymptotic variance in the downstream estimation of the regression coefficients. Our semiparametric approach targets the best decreasing approximation of the derivative of the log-density of the noise distribution. At the population level, this fitting process is a nonparametric extension of score matching, corresponding to a log-concave projection of the noise distribution with respect to the Fisher divergence. The procedure is computationally efficient, and we prove that our procedure attains the minimal asymptotic covariance among all convex $M$-estimators. As an example of a non-log-concave setting, for Cauchy errors, the optimal convex loss function is Huber-like, and our procedure yields an asymptotic efficiency greater than 0.87 relative to the oracle maximum likelihood estimator o
    
[^2]: LC-Tsalis-INF: 广义最佳双赢线性背景强化型赌博机

    LC-Tsalis-INF: Generalized Best-of-Both-Worlds Linear Contextual Bandits

    [https://arxiv.org/abs/2403.03219](https://arxiv.org/abs/2403.03219)

    该研究提出了一种广义最佳双赢线性背景强化型赌博机算法，能够在次优性差距受到下界限制时遗憾为$O(\log(T))$。同时引入了边缘条件来描述次优性差距对问题难度的影响。

    

    本研究考虑具有独立同分布（i.i.d.）背景的线性背景强化型赌博机问题。在这个问题中，现有研究提出了最佳双赢（BoBW）算法，其遗憾在随机区域中满足$O(\log^2(T))$，其中$T$为回合数，其次优性差距由正常数下界，同时在对抗性区域中满足$O(\sqrt{T})$。然而，对$T$的依赖仍有改进空间，并且次优性差距的假设可以放宽。针对这个问题，本研究提出了一个算法，当次优性差距受到下界限制时，其遗憾满足$O(\log(T))$。此外，我们引入了一个边缘条件，即对次优性差距的一个更温和的假设。该条件使用参数$\beta \in (0, \infty]$表征与次优性差距相关的问题难度。然后我们证明该算法的遗憾满足$O\left(\

    arXiv:2403.03219v1 Announce Type: new  Abstract: This study considers the linear contextual bandit problem with independent and identically distributed (i.i.d.) contexts. In this problem, existing studies have proposed Best-of-Both-Worlds (BoBW) algorithms whose regrets satisfy $O(\log^2(T))$ for the number of rounds $T$ in a stochastic regime with a suboptimality gap lower-bounded by a positive constant, while satisfying $O(\sqrt{T})$ in an adversarial regime. However, the dependency on $T$ has room for improvement, and the suboptimality-gap assumption can be relaxed. For this issue, this study proposes an algorithm whose regret satisfies $O(\log(T))$ in the setting when the suboptimality gap is lower-bounded. Furthermore, we introduce a margin condition, a milder assumption on the suboptimality gap. That condition characterizes the problem difficulty linked to the suboptimality gap using a parameter $\beta \in (0, \infty]$. We then show that the algorithm's regret satisfies $O\left(\
    
[^3]: 具有差分隐私(分组)结果的因果推断

    Causal Inference with Differentially Private (Clustered) Outcomes. (arXiv:2308.00957v1 [stat.ML])

    [http://arxiv.org/abs/2308.00957](http://arxiv.org/abs/2308.00957)

    本文提出了一种新的差分隐私机制"Cluster-DP"，它在保证隐私的同时利用数据的聚类结构，从而实现了更强的隐私保证和较低的方差，可以用于进行因果分析。

    

    从随机实验中估计因果效应只有在参与者同意透露他们可能敏感的响应时才可行。在确保隐私的许多方法中，标签差分隐私是一种广泛使用的算法隐私保证度量，可以鼓励参与者分享响应而不会面临去匿名化的风险。许多差分隐私机制会向原始数据集中注入噪音来实现这种隐私保证，这会增加大多数统计估计量的方差，使得精确测量因果效应变得困难：从差分隐私数据进行因果分析存在着固有的隐私-方差权衡。为了实现更强隐私保证的较低方差，我们提出了一种新的差分隐私机制"Cluster-DP"，它利用数据的任何给定的聚类结构，同时仍然允许对因果效应进行估计。

    Estimating causal effects from randomized experiments is only feasible if participants agree to reveal their potentially sensitive responses. Of the many ways of ensuring privacy, label differential privacy is a widely used measure of an algorithm's privacy guarantee, which might encourage participants to share responses without running the risk of de-anonymization. Many differentially private mechanisms inject noise into the original data-set to achieve this privacy guarantee, which increases the variance of most statistical estimators and makes the precise measurement of causal effects difficult: there exists a fundamental privacy-variance trade-off to performing causal analyses from differentially private data. With the aim of achieving lower variance for stronger privacy guarantees, we suggest a new differential privacy mechanism, "Cluster-DP", which leverages any given cluster structure of the data while still allowing for the estimation of causal effects. We show that, depending 
    

