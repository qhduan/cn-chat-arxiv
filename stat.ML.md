# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Vanilla Bayesian Optimization Performs Great in High Dimension](https://arxiv.org/abs/2402.02229) | 本文研究了高维情况下贝叶斯优化算法的问题，并提出了一种改进方法，通过对先验假设进行简单的缩放，使普通贝叶斯优化在高维任务中表现出色。 |
| [^2] | [Minimax Optimal Submodular Optimization with Bandit Feedback.](http://arxiv.org/abs/2310.18465) | 这项工作研究了带有Bandit反馈的极小极大次模优化问题，在这个问题中，我们建立了第一个最小最大下限，并提出了一个能够与下限遗憾相匹配的算法。 |
| [^3] | [TNDDR: Efficient and doubly robust estimation of COVID-19 vaccine effectiveness under the test-negative design.](http://arxiv.org/abs/2310.04578) | 我们提出了一种高效且双重鲁棒的估计器TNDDR，用于在阴性测试设计下估计COVID-19疫苗的有效性，可有效解决选择偏差问题，并结合机器学习技术进行辅助函数估计。 |
| [^4] | [AdaStop: sequential testing for efficient and reliable comparisons of Deep RL Agents.](http://arxiv.org/abs/2306.10882) | AdaStop是一种基于多组序列测试的新统计测试方法，可用于比较多个深度强化学习算法来解决实验结果可复制性的问题。 |

# 详细

[^1]: 高维情况下，普通贝叶斯优化算法表现出色

    Vanilla Bayesian Optimization Performs Great in High Dimension

    [https://arxiv.org/abs/2402.02229](https://arxiv.org/abs/2402.02229)

    本文研究了高维情况下贝叶斯优化算法的问题，并提出了一种改进方法，通过对先验假设进行简单的缩放，使普通贝叶斯优化在高维任务中表现出色。

    

    长期以来，高维问题一直被认为是贝叶斯优化算法的软肋。受到维度噪音的刺激，许多算法旨在通过对目标应用各种简化假设来提高其性能。本文通过识别导致普通贝叶斯优化在高维任务中不适用的退化现象，并进一步展示了现有算法如何通过降低模型复杂度来应对这些退化现象。此外，我们还提出了一种对普通贝叶斯优化算法中典型先验假设的改进方法，该方法在不对目标施加结构性限制的情况下将复杂性降低到可管理的水平。我们的修改方法——通过维度对高斯过程长度先验进行简单的缩放——揭示了标准贝叶斯优化在高维情况下的显著改进，明确表明其效果远远超出以往的预期。

    High-dimensional problems have long been considered the Achilles' heel of Bayesian optimization algorithms. Spurred by the curse of dimensionality, a large collection of algorithms aim to make it more performant in this setting, commonly by imposing various simplifying assumptions on the objective. In this paper, we identify the degeneracies that make vanilla Bayesian optimization poorly suited to high-dimensional tasks, and further show how existing algorithms address these degeneracies through the lens of lowering the model complexity. Moreover, we propose an enhancement to the prior assumptions that are typical to vanilla Bayesian optimization algorithms, which reduces the complexity to manageable levels without imposing structural restrictions on the objective. Our modification - a simple scaling of the Gaussian process lengthscale prior with the dimensionality - reveals that standard Bayesian optimization works drastically better than previously thought in high dimensions, clearly
    
[^2]: 带有Bandit反馈的极小极大次模优化问题

    Minimax Optimal Submodular Optimization with Bandit Feedback. (arXiv:2310.18465v1 [cs.LG])

    [http://arxiv.org/abs/2310.18465](http://arxiv.org/abs/2310.18465)

    这项工作研究了带有Bandit反馈的极小极大次模优化问题，在这个问题中，我们建立了第一个最小最大下限，并提出了一个能够与下限遗憾相匹配的算法。

    

    我们考虑在随机Bandit反馈下，最大化一个单调次模集函数$f：2 ^ {[n]} \rightarrow [0,1]$。具体来说，$f$对于学习者是未知的，但是在每个时间$t=1,\dots,T$，学习者选择一个集合$S_t \subset [n]$，其中$|S_t|\leq k$，并接收奖励$f(S_t)+\eta_t$，其中$\eta_t$是均值为零的次高斯噪声。目标是在$T$次中使得学习者对于带有$|S_*|=k$的最大$f(S_*)$的($1-e^{-1}$)近似的最小遗憾，通过对$f$的贪婪最大化来达到。到目前为止，文献中最好的遗憾边界按照$k n^{1/3} T^{2/3}$的比例缩放。通过将每个集合简单地视为一个唯一的arm，可以推断出$\sqrt{{n \choose k} T}$也是可实现的。在这项工作中，我们建立了这种情况下的第一个极小极大下限，其按照$\mathcal{O}(\min_{i \le k}(in^{1/3}T^{2/3} + \sqrt{n^{k-i}T}))$的比例缩放。此外，我们提出了一个能够与下限遗憾相匹配的算法。

    We consider maximizing a monotonic, submodular set function $f: 2^{[n]} \rightarrow [0,1]$ under stochastic bandit feedback. Specifically, $f$ is unknown to the learner but at each time $t=1,\dots,T$ the learner chooses a set $S_t \subset [n]$ with $|S_t| \leq k$ and receives reward $f(S_t) + \eta_t$ where $\eta_t$ is mean-zero sub-Gaussian noise. The objective is to minimize the learner's regret over $T$ times with respect to ($1-e^{-1}$)-approximation of maximum $f(S_*)$ with $|S_*| = k$, obtained through greedy maximization of $f$. To date, the best regret bound in the literature scales as $k n^{1/3} T^{2/3}$. And by trivially treating every set as a unique arm one deduces that $\sqrt{ {n \choose k} T }$ is also achievable. In this work, we establish the first minimax lower bound for this setting that scales like $\mathcal{O}(\min_{i \le k}(in^{1/3}T^{2/3} + \sqrt{n^{k-i}T}))$. Moreover, we propose an algorithm that is capable of matching the lower bound regret.
    
[^3]: TNDDR: 高效且双重鲁棒的COVID-19疫苗有效性估计在阴性测试设计下

    TNDDR: Efficient and doubly robust estimation of COVID-19 vaccine effectiveness under the test-negative design. (arXiv:2310.04578v1 [stat.ME])

    [http://arxiv.org/abs/2310.04578](http://arxiv.org/abs/2310.04578)

    我们提出了一种高效且双重鲁棒的估计器TNDDR，用于在阴性测试设计下估计COVID-19疫苗的有效性，可有效解决选择偏差问题，并结合机器学习技术进行辅助函数估计。

    

    尽管阴性测试设计（TND）常用于监测季节性流感疫苗有效性（VE），但最近已成为COVID-19疫苗监测的重要组成部分，但由于结果相关抽样，它容易受到选择偏差的影响。一些研究已经解决了TND下因果参数的可鉴别性和估计问题，但尚未研究非参数估计器在无混杂性假设下的效率边界。我们提出了一种称为TNDDR（TND双重鲁棒）的一步双重鲁棒和局部高效估计器,它利用样本分割，并可以结合机器学习技术来估计辅助函数。我们推导了结果边际期望的高效影响函数（EIF），探索了von Mises展开，并建立了TNDDR的n的平方根一致性、渐近正态性和双重鲁棒性的条件。

    While the test-negative design (TND), which is routinely used for monitoring seasonal flu vaccine effectiveness (VE), has recently become integral to COVID-19 vaccine surveillance, it is susceptible to selection bias due to outcome-dependent sampling. Some studies have addressed the identifiability and estimation of causal parameters under the TND, but efficiency bounds for nonparametric estimators of the target parameter under the unconfoundedness assumption have not yet been investigated. We propose a one-step doubly robust and locally efficient estimator called TNDDR (TND doubly robust), which utilizes sample splitting and can incorporate machine learning techniques to estimate the nuisance functions. We derive the efficient influence function (EIF) for the marginal expectation of the outcome under a vaccination intervention, explore the von Mises expansion, and establish the conditions for $\sqrt{n}-$consistency, asymptotic normality and double robustness of TNDDR. The proposed TND
    
[^4]: AdaStop：用于深度强化学习代理比较的高效可靠序列测试

    AdaStop: sequential testing for efficient and reliable comparisons of Deep RL Agents. (arXiv:2306.10882v1 [cs.LG])

    [http://arxiv.org/abs/2306.10882](http://arxiv.org/abs/2306.10882)

    AdaStop是一种基于多组序列测试的新统计测试方法，可用于比较多个深度强化学习算法来解决实验结果可复制性的问题。

    

    许多深度强化学习实验结果的可复现性受到质疑。为了解决这个可复现性危机，我们提出了一种理论上可靠的方法，用于比较多个深度强化学习算法。由于一个深度强化学习算法的一次执行性能是随机的，所以需要进行独立的多次执行来精确评估它。当比较多个强化学习算法时，一个主要问题是需要进行多少次执行，并且如何确保这样比较的结果在理论上是可靠的。深度强化学习的研究人员通常使用少于5个独立执行来比较算法：我们认为这通常是不够的。而且，当同时比较几个算法时，每个比较的误差都会累积，必须采用多重测试程序来考虑这些误差，以维持低误差保证。为了以统计学上的可靠方式解决这个问题，我们介绍了AdaStop，这是一种基于多组序列测试的新统计测试方法。

    The reproducibility of many experimental results in Deep Reinforcement Learning (RL) is under question. To solve this reproducibility crisis, we propose a theoretically sound methodology to compare multiple Deep RL algorithms. The performance of one execution of a Deep RL algorithm is random so that independent executions are needed to assess it precisely. When comparing several RL algorithms, a major question is how many executions must be made and how can we assure that the results of such a comparison is theoretically sound. Researchers in Deep RL often use less than 5 independent executions to compare algorithms: we claim that this is not enough in general. Moreover, when comparing several algorithms at once, the error of each comparison accumulates and must be taken into account with a multiple tests procedure to preserve low error guarantees. To address this problem in a statistically sound way, we introduce AdaStop, a new statistical test based on multiple group sequential tests
    

