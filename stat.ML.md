# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Monte Carlo with kernel-based Gibbs measures: Guarantees for probabilistic herding](https://arxiv.org/abs/2402.11736) | 该论文研究了一种联合概率分布，其支持趋于最小化最坏情况误差，证明了它在最坏情况积分误差集中不等式上优于i.i.d.蒙特卡罗。 |
| [^2] | [Optimal Multi-Distribution Learning.](http://arxiv.org/abs/2312.05134) | 本论文提出了一种最优化多分布学习的方法，通过自适应采样来实现数据高效的学习。针对Vapnik-Chervonenkis (VC)维数为d的假设类，算法可以生成一个ε-最优随机假设，并且样本复杂度与最佳下界保持一致。同时，该算法的思想和理论还被进一步扩展以适应Rademacher类。最终提出的算法是奥拉克尔高效的，仅访问假设类。 |
| [^3] | [Blending Imitation and Reinforcement Learning for Robust Policy Improvement.](http://arxiv.org/abs/2310.01737) | 本文提出了一种融合模仿学习和强化学习的方法，根据在线评估结果交替使用二者，以提高样本效率和学习效果。 |
| [^4] | [Thompson Exploration with Best Challenger Rule in Best Arm Identification.](http://arxiv.org/abs/2310.00539) | 本文提出了一种新的策略，将Thompson采样与最佳候选规则相结合，用于解决最佳臂识别问题。该策略在渐近情况下是最优的，并在一般的多臂赌博机问题中达到接近最优的性能。 |
| [^5] | [A Meta-Learning Method for Estimation of Causal Excursion Effects to Assess Time-Varying Moderation.](http://arxiv.org/abs/2306.16297) | 这项研究介绍了一种元学习方法，用于评估因果偏离效应，以评估干预效果随时间的变化或通过个体特征、环境或过去的反应来调节。目前的数据分析方法需要预先指定观察到的高维历史的特征来构建重要干扰参数的工作模型，而机器学习算法可以自动进行特征构建，但其朴素应用存在问题。 |
| [^6] | [Active Policy Improvement from Multiple Black-box Oracles.](http://arxiv.org/abs/2306.10259) | 本研究提出了MAPS和MAPS-SE两个算法，可在多黑盒预言情况下，采用模仿学习并主动选择和改进最优预言，显著提升了性能。 |

# 详细

[^1]: 基于核Gibbs测度的蒙特卡罗：概率随机放牧的保证

    Monte Carlo with kernel-based Gibbs measures: Guarantees for probabilistic herding

    [https://arxiv.org/abs/2402.11736](https://arxiv.org/abs/2402.11736)

    该论文研究了一种联合概率分布，其支持趋于最小化最坏情况误差，证明了它在最坏情况积分误差集中不等式上优于i.i.d.蒙特卡罗。

    

    Kernel herding属于一类确定性的四位数法，旨在通过再生核希尔伯特空间（RKHS）上的最坏情况积分误差。尽管有很强的实验支持，但在通常情况下，即RKHS是无限维时，证明这种最坏情况误差以比标准积分节点数量的平方根更快的速率减少是困难的。在这篇理论论文中，我们研究了一个关于积分节点的联合概率分布，其支持趋于最小化与核放牧相同的最坏情况误差。我们证明它优于i.i.d.蒙特卡罗，意味着在最坏情况积分误差上具有更紧的集中不等式。尽管尚未提高速率，但这表明了研究Gibbs测度的数学工具可以帮助理解核放牧及其变体在计算上的改进程度

    arXiv:2402.11736v1 Announce Type: new  Abstract: Kernel herding belongs to a family of deterministic quadratures that seek to minimize the worst-case integration error over a reproducing kernel Hilbert space (RKHS). In spite of strong experimental support, it has revealed difficult to prove that this worst-case error decreases at a faster rate than the standard square root of the number of quadrature nodes, at least in the usual case where the RKHS is infinite-dimensional. In this theoretical paper, we study a joint probability distribution over quadrature nodes, whose support tends to minimize the same worst-case error as kernel herding. We prove that it does outperform i.i.d. Monte Carlo, in the sense of coming with a tighter concentration inequality on the worst-case integration error. While not improving the rate yet, this demonstrates that the mathematical tools of the study of Gibbs measures can help understand to what extent kernel herding and its variants improve on computation
    
[^2]: 最优化多分布学习

    Optimal Multi-Distribution Learning. (arXiv:2312.05134v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2312.05134](http://arxiv.org/abs/2312.05134)

    本论文提出了一种最优化多分布学习的方法，通过自适应采样来实现数据高效的学习。针对Vapnik-Chervonenkis (VC)维数为d的假设类，算法可以生成一个ε-最优随机假设，并且样本复杂度与最佳下界保持一致。同时，该算法的思想和理论还被进一步扩展以适应Rademacher类。最终提出的算法是奥拉克尔高效的，仅访问假设类。

    

    多分布学习（MDL）旨在学习一个共享模型，使得在k个不同的数据分布下，最小化最坏情况风险，已成为适应健壮性、公平性、多组合作等需求的统一框架。实现数据高效的MDL需要在学习过程中进行自适应采样，也称为按需采样。然而，最优样本复杂度的上下界之间存在较大差距。针对Vapnik-Chervonenkis（VC）维数为d的假设类，我们提出了一种新颖的算法，可生成一个ε-最优随机假设，其样本复杂度接近于（d+k）/ε^2（在某些对数因子中），与已知的最佳下界匹配。我们的算法思想和理论被进一步扩展，以适应Rademacher类。提出的算法是奥拉克尔高效的，仅仅访问假设类

    Multi-distribution learning (MDL), which seeks to learn a shared model that minimizes the worst-case risk across $k$ distinct data distributions, has emerged as a unified framework in response to the evolving demand for robustness, fairness, multi-group collaboration, etc. Achieving data-efficient MDL necessitates adaptive sampling, also called on-demand sampling, throughout the learning process. However, there exist substantial gaps between the state-of-the-art upper and lower bounds on the optimal sample complexity. Focusing on a hypothesis class of Vapnik-Chervonenkis (VC) dimension $d$, we propose a novel algorithm that yields an $varepsilon$-optimal randomized hypothesis with a sample complexity on the order of $(d+k)/\varepsilon^2$ (modulo some logarithmic factor), matching the best-known lower bound. Our algorithmic ideas and theory have been further extended to accommodate Rademacher classes. The proposed algorithms are oracle-efficient, which access the hypothesis class solely
    
[^3]: 融合模仿学习和强化学习以实现鲁棒策略改进

    Blending Imitation and Reinforcement Learning for Robust Policy Improvement. (arXiv:2310.01737v1 [cs.LG])

    [http://arxiv.org/abs/2310.01737](http://arxiv.org/abs/2310.01737)

    本文提出了一种融合模仿学习和强化学习的方法，根据在线评估结果交替使用二者，以提高样本效率和学习效果。

    

    虽然强化学习在性能上表现出色，但其样本复杂度仍然是一个重大障碍，限制了其在各个领域的广泛应用。模仿学习利用神经网络优化样本效率，但通常受到所使用的专家示范的质量限制。本文介绍了一种融合模仿学习和强化学习的方法，该方法根据在线评估结果交替使用二者，有效地提高了学习效率。这种算法能够从多种黑盒专家示范中学习和改进。

    While reinforcement learning (RL) has shown promising performance, its sample complexity continues to be a substantial hurdle, restricting its broader application across a variety of domains. Imitation learning (IL) utilizes oracles to improve sample efficiency, yet it is often constrained by the quality of the oracles deployed. which actively interleaves between IL and RL based on an online estimate of their performance. RPI draws on the strengths of IL, using oracle queries to facilitate exploration, an aspect that is notably challenging in sparse-reward RL, particularly during the early stages of learning. As learning unfolds, RPI gradually transitions to RL, effectively treating the learned policy as an improved oracle. This algorithm is capable of learning from and improving upon a diverse set of black-box oracles. Integral to RPI are Robust Active Policy Selection (RAPS) and Robust Policy Gradient (RPG), both of which reason over whether to perform state-wise imitation from the o
    
[^4]: 最佳候选规则下的Thompson探索在最佳臂识别中的应用

    Thompson Exploration with Best Challenger Rule in Best Arm Identification. (arXiv:2310.00539v1 [stat.ML])

    [http://arxiv.org/abs/2310.00539](http://arxiv.org/abs/2310.00539)

    本文提出了一种新的策略，将Thompson采样与最佳候选规则相结合，用于解决最佳臂识别问题。该策略在渐近情况下是最优的，并在一般的多臂赌博机问题中达到接近最优的性能。

    

    本文研究了在经典单参数指数模型下，固定置信度下的最佳臂识别（BAI）问题。针对这个问题，目前已有很多策略被提出，但大多数需要在每一轮解决一个最优化问题和/或者需要探索一个臂至少一定次数，除非是针对高斯模型的限制。为了解决这些限制，我们提出了一种新的策略，将Thompson采样与一个计算效率高的方法——最佳候选规则相结合。虽然Thompson采样最初被考虑用于最大化累积奖励，但我们证明它也可以自然地用于在BAI中探索臂而不强迫最大化奖励。我们证明了我们的策略在任意两臂赌博机问题上是渐近最优的，并且在一般的$K$臂赌博机问题上（$K\geq 3$）达到接近最优的性能。然而，在数值实验中，我们的策略与现有方法相比表现出了竞争性的性能。

    This paper studies the fixed-confidence best arm identification (BAI) problem in the bandit framework in the canonical single-parameter exponential models. For this problem, many policies have been proposed, but most of them require solving an optimization problem at every round and/or are forced to explore an arm at least a certain number of times except those restricted to the Gaussian model. To address these limitations, we propose a novel policy that combines Thompson sampling with a computationally efficient approach known as the best challenger rule. While Thompson sampling was originally considered for maximizing the cumulative reward, we demonstrate that it can be used to naturally explore arms in BAI without forcing it. We show that our policy is asymptotically optimal for any two-armed bandit problems and achieves near optimality for general $K$-armed bandit problems for $K\geq 3$. Nevertheless, in numerical experiments, our policy shows competitive performance compared to as
    
[^5]: 一种用于评估时变调节因素的因果偏离效应估计的元学习方法

    A Meta-Learning Method for Estimation of Causal Excursion Effects to Assess Time-Varying Moderation. (arXiv:2306.16297v1 [stat.ME])

    [http://arxiv.org/abs/2306.16297](http://arxiv.org/abs/2306.16297)

    这项研究介绍了一种元学习方法，用于评估因果偏离效应，以评估干预效果随时间的变化或通过个体特征、环境或过去的反应来调节。目前的数据分析方法需要预先指定观察到的高维历史的特征来构建重要干扰参数的工作模型，而机器学习算法可以自动进行特征构建，但其朴素应用存在问题。

    

    可穿戴技术和智能手机提供的数字化健康干预的双重革命显著增加了移动健康（mHealth）干预在各个健康科学领域的可及性和采纳率。顺序随机实验称为微随机试验（MRTs）已经越来越受欢迎，用于实证评估这些mHealth干预组成部分的有效性。MRTs产生了一类新的因果估计量，称为“因果偏离效应”，使健康科学家能够评估干预效果随时间的变化或通过个体特征、环境或过去的反应来调节。然而，目前用于估计因果偏离效应的数据分析方法需要预先指定观察到的高维历史的特征来构建重要干扰参数的工作模型。虽然机器学习算法在自动特征构建方面具有优势，但其朴素应用导致了问题。

    Twin revolutions in wearable technologies and smartphone-delivered digital health interventions have significantly expanded the accessibility and uptake of mobile health (mHealth) interventions across various health science domains. Sequentially randomized experiments called micro-randomized trials (MRTs) have grown in popularity to empirically evaluate the effectiveness of these mHealth intervention components. MRTs have given rise to a new class of causal estimands known as "causal excursion effects", which enable health scientists to assess how intervention effectiveness changes over time or is moderated by individual characteristics, context, or responses in the past. However, current data analysis methods for estimating causal excursion effects require pre-specified features of the observed high-dimensional history to construct a working model of an important nuisance parameter. While machine learning algorithms are ideal for automatic feature construction, their naive application
    
[^6]: 多黑盒预言下的主动策略改进

    Active Policy Improvement from Multiple Black-box Oracles. (arXiv:2306.10259v1 [cs.LG])

    [http://arxiv.org/abs/2306.10259](http://arxiv.org/abs/2306.10259)

    本研究提出了MAPS和MAPS-SE两个算法，可在多黑盒预言情况下，采用模仿学习并主动选择和改进最优预言，显著提升了性能。

    

    强化学习在各种复杂领域中取得了重大进展，但是通过强化学习确定有效策略往往需要进行广泛的探索，而模仿学习旨在通过使用专家演示来指导探索，缓解这个问题。在真实世界情境下，人们通常只能接触到多个次优的黑盒预言，而不是单个最优的预言，这些预言不能在所有状态下普遍优于彼此，这给主动决定在哪种状态下使用哪种预言以及如何改进各自估计值函数提出了挑战。本文介绍了一个可行的解决方案，即MAPS和MAPS-SE算法。

    Reinforcement learning (RL) has made significant strides in various complex domains. However, identifying an effective policy via RL often necessitates extensive exploration. Imitation learning aims to mitigate this issue by using expert demonstrations to guide exploration. In real-world scenarios, one often has access to multiple suboptimal black-box experts, rather than a single optimal oracle. These experts do not universally outperform each other across all states, presenting a challenge in actively deciding which oracle to use and in which state. We introduce MAPS and MAPS-SE, a class of policy improvement algorithms that perform imitation learning from multiple suboptimal oracles. In particular, MAPS actively selects which of the oracles to imitate and improve their value function estimates, and MAPS-SE additionally leverages an active state exploration criterion to determine which states one should explore. We provide a comprehensive theoretical analysis and demonstrate that MAP
    

