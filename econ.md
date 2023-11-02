# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On Gaussian Process Priors in Conditional Moment Restriction Models.](http://arxiv.org/abs/2311.00662) | 本文研究了在非参数条件矩限制模型中，对于一段未知函数的准贝叶斯估计和不确定性量化。通过推导高斯过程先验的收缩速率，并满足伯恩斯坦-冯·米塞斯定理的条件，证明了最优加权准贝叶斯可信集具有精确的渐近经典主义覆盖率。 |
| [^2] | [Personalized Assignment to One of Many Treatment Arms via Regularized and Clustered Joint Assignment Forests.](http://arxiv.org/abs/2311.00577) | 提出了一种个性化分配至多个治疗组的方法，通过正则化和聚类优化治疗分配，实现了更好的效果估计和个性化效益。 |
| [^3] | [Bounds on Treatment Effects under Stochastic Monotonicity Assumption in Sample Selection Models.](http://arxiv.org/abs/2311.00439) | 本文研究了在样本选择模型中处理效应的部分识别问题，当排除限制失效和选择效应的单调性假设不完全成立时。我们的方法允许在单调性不严格成立的情况下获得有用的界限。 |
| [^4] | [Semiparametric Discrete Choice Models for Bundles.](http://arxiv.org/abs/2311.00013) | 本文提出了两种方法来估计半参数的离散选择模型，一种是基于匹配的核加权秩估计器，另一种是多指标最小绝对偏差估计器。这些方法都能考虑扰动的相关性，并可以在备选和个体特定的回归变量上估计偏好参数。 |
| [^5] | [Smootheness-Adaptive Dynamic Pricing with Nonparametric Demand Learning.](http://arxiv.org/abs/2310.07558) | 这项研究提出了一种具有非参数需求学习和平滑自适应的动态定价算法，通过使用自相似条件实现了最小化极限遗憾。 |
| [^6] | [A Reexamination of Proof Approaches for the Impossibility Theorem.](http://arxiv.org/abs/2309.06753) | 本研究使用逻辑的证明演算提出了对不可能性定理的证明方法，该方法保证了对所有可能社会福利函数的所有配置进行了检验，从而证明了该定理。 |
| [^7] | [Market Design for Dynamic Pricing and Pooling in Capacitated Networks.](http://arxiv.org/abs/2307.03994) | 本研究提出了一种用于动态定价和汇聚网络的市场设计，通过设置边缘价格激励代理商共享有限网络容量。在考虑了整数和网络约束以及代理商异质偏好的情况下，我们提供了充分条件，保证市场均衡的存在和多项式时间计算，并识别了实现最大效用的特定市场均衡。 |
| [^8] | [Fast Inference for Quantile Regression with Tens of Millions of Observations.](http://arxiv.org/abs/2209.14502) | 本文提出了一个快速推断框架，利用随机次梯度下降更新处理数量级达数千万观测值的分位数回归问题。通过顺序处理数据、聚合参数估计和计算关键统计量的解路径，该方法在处理超大规模数据集时具有较低的计算资源和内存需求。 |
| [^9] | [A Vector Monotonicity Assumption for Multiple Instruments.](http://arxiv.org/abs/2009.00553) | 本文提出了向量单调性假设的观点，该假设对于多个工具变量组合的研究具有重要意义。通过假设所有工具变量的治疗接受程度是单调的，我们可以得到一些因果参数的点识别，并提供了相应的估计方法。 |

# 详细

[^1]: 关于高斯过程先验在条件矩限制模型中的应用

    On Gaussian Process Priors in Conditional Moment Restriction Models. (arXiv:2311.00662v1 [econ.EM])

    [http://arxiv.org/abs/2311.00662](http://arxiv.org/abs/2311.00662)

    本文研究了在非参数条件矩限制模型中，对于一段未知函数的准贝叶斯估计和不确定性量化。通过推导高斯过程先验的收缩速率，并满足伯恩斯坦-冯·米塞斯定理的条件，证明了最优加权准贝叶斯可信集具有精确的渐近经典主义覆盖率。

    

    本文研究了在非参数条件矩限制模型中，对于一段未知函数的准贝叶斯估计和不确定性量化。我们推导了一类高斯过程先验的收缩速率，并提供了满足伯恩斯坦-冯·米塞斯定理的条件。作为结果，我们证明了最优加权准贝叶斯可信集具有精确的渐近经典主义覆盖率。这扩展了关于参数广义矩估计（GMM）模型的最优加权准贝叶斯可信集的频率有效性的经典结果。

    This paper studies quasi-Bayesian estimation and uncertainty quantification for an unknown function that is identified by a nonparametric conditional moment restriction model. We derive contraction rates for a class of Gaussian process priors and provide conditions under which a Bernstein-von Mises theorem holds for the quasi-posterior distribution. As a consequence, we show that optimally-weighted quasi-Bayes credible sets have exact asymptotic frequentist coverage. This extends classical result on the frequentist validity of optimally weighted quasi-Bayes credible sets for parametric generalized method of moments (GMM) models.
    
[^2]: 通过正则化和聚类联合分配森林进行个性化分配至多个治疗组

    Personalized Assignment to One of Many Treatment Arms via Regularized and Clustered Joint Assignment Forests. (arXiv:2311.00577v1 [stat.ML])

    [http://arxiv.org/abs/2311.00577](http://arxiv.org/abs/2311.00577)

    提出了一种个性化分配至多个治疗组的方法，通过正则化和聚类优化治疗分配，实现了更好的效果估计和个性化效益。

    

    我们考虑从随机对照试验中学习个性化的分配至多个治疗组。由于过多的方差，对于每个治疗组分别估计异质治疗效果的标准方法在这种情况下可能表现不佳。相反，我们提出了一种汇总治疗组信息的方法：首先，我们考虑基于贪婪递归分区的正则化森林分配算法，该算法可缩小不同治疗组之间的效果估计。其次，我们通过聚类方案将治疗组与具有一致相似结果的组合起来，增强了我们的算法。在模拟研究中，我们将这些方法与分别预测每个治疗组结果的方法进行了比较，并记录了通过正则化和聚类直接优化治疗分配带来的收益。在一个理论模型中，我们说明治疗组数量较多时找到最佳组的困难，而通过正则化和聚类可以实现个性化的明显效益。

    We consider learning personalized assignments to one of many treatment arms from a randomized controlled trial. Standard methods that estimate heterogeneous treatment effects separately for each arm may perform poorly in this case due to excess variance. We instead propose methods that pool information across treatment arms: First, we consider a regularized forest-based assignment algorithm based on greedy recursive partitioning that shrinks effect estimates across arms. Second, we augment our algorithm by a clustering scheme that combines treatment arms with consistently similar outcomes. In a simulation study, we compare the performance of these approaches to predicting arm-wise outcomes separately, and document gains of directly optimizing the treatment assignment with regularization and clustering. In a theoretical model, we illustrate how a high number of treatment arms makes finding the best arm hard, while we can achieve sizable utility gains from personalization by regularized 
    
[^3]: 在样本选择模型中的随机单调性假设下的处理效应界限

    Bounds on Treatment Effects under Stochastic Monotonicity Assumption in Sample Selection Models. (arXiv:2311.00439v1 [econ.EM])

    [http://arxiv.org/abs/2311.00439](http://arxiv.org/abs/2311.00439)

    本文研究了在样本选择模型中处理效应的部分识别问题，当排除限制失效和选择效应的单调性假设不完全成立时。我们的方法允许在单调性不严格成立的情况下获得有用的界限。

    

    本文讨论了在样本选择模型中处理效应的部分识别，当排除限制失效和选择效应的单调性假设不能完全成立时，这两个都是应用现有方法的主要挑战。我们的方法基于Lee（2009）的过程，他考虑了单调性假设下的部分识别，但我们只假设了随机（更弱）版本的单调性，这依赖于预先指定的参数$ \vartheta $，代表研究人员对单调性合理性的信念。在这种假设下，我们证明即使单调行为模型不严格成立，我们仍然可以获得有用的界限。当实证研究人员预计人口中只有一小部分人会在选择中表现出单调行为时，我们的方法非常有用；它还可以用于进行敏感性分析或检验识别能力。

    This paper discusses the partial identification of treatment effects in sample selection models when the exclusion restriction fails and the monotonicity assumption in the selection effect does not hold exactly, both of which are key challenges in applying the existing methodologies. Our approach builds on Lee's (2009) procedure, who considers partial identification under the monotonicity assumption, but we assume only a stochastic (and weaker) version of monotonicity, which depends on a prespecified parameter $\vartheta$ that represents researchers' belief in the plausibility of the monotonicity. Under this assumption, we show that we can still obtain useful bounds even when the monotonic behavioral model does not strictly hold. Our procedure is useful when empirical researchers anticipate that a small fraction of the population will not behave monotonically in selection; it can also be an effective tool for performing sensitivity analysis or examining the identification power of the 
    
[^4]: 可供选择的半参数离散选择模型

    Semiparametric Discrete Choice Models for Bundles. (arXiv:2311.00013v1 [econ.EM])

    [http://arxiv.org/abs/2311.00013](http://arxiv.org/abs/2311.00013)

    本文提出了两种方法来估计半参数的离散选择模型，一种是基于匹配的核加权秩估计器，另一种是多指标最小绝对偏差估计器。这些方法都能考虑扰动的相关性，并可以在备选和个体特定的回归变量上估计偏好参数。

    

    我们提出了两种方法来估计半参数的离散选择模型。我们的第一种方法是基于匹配的识别策略的核加权秩估计器。我们建立了其完整的渐近性质，并证明了非参数自助法的推断有效性。然后，我们介绍了一种新的多指标最小绝对偏差（LAD）估计器作为替代方法，其主要优点是能够在备选和个体特定的回归变量上估计偏好参数。这两种方法都能够考虑选择之间任意的扰动相关性，前者还可以考虑人际异方差性。我们还证明了这些程序背后的识别策略可以自然地推广到面板数据设置，从而产生类似的局部极大值估计器和用于估计带有固定效应的捆绑选择模型的LAD估计器。我们推导出了这些估计器的极限分布。

    We propose two approaches to estimate semiparametric discrete choice models for bundles. Our first approach is a kernel-weighted rank estimator based on a matching-based identification strategy. We establish its complete asymptotic properties and prove the validity of the nonparametric bootstrap for inference. We then introduce a new multi-index least absolute deviations (LAD) estimator as an alternative, of which the main advantage is its capacity to estimate preference parameters on both alternative- and agent-specific regressors. Both methods can account for arbitrary correlation in disturbances across choices, with the former also allowing for interpersonal heteroskedasticity. We also demonstrate that the identification strategy underlying these procedures can be extended naturally to panel data settings, producing an analogous localized maximum score estimator and a LAD estimator for estimating bundle choice models with fixed effects. We derive the limiting distribution of the for
    
[^5]: 具有非参数需求学习的平滑自适应动态定价

    Smootheness-Adaptive Dynamic Pricing with Nonparametric Demand Learning. (arXiv:2310.07558v1 [stat.ML])

    [http://arxiv.org/abs/2310.07558](http://arxiv.org/abs/2310.07558)

    这项研究提出了一种具有非参数需求学习和平滑自适应的动态定价算法，通过使用自相似条件实现了最小化极限遗憾。

    

    我们研究了需求函数为非参数和Holder平滑的动态定价问题，并且我们专注于适应未知的Holder平滑参数β的能力。传统上，最优的动态定价算法严重依赖于对β的了解，以达到一个最小化极限遗憾的效果，即O(T^((β+1)/(2β+1)))。然而，我们通过证明没有定价策略能够在不知道β的情况下自适应地达到这个最小化极限遗憾，突显了这个动态定价问题的适应性挑战。受到不可能性结果的启发，我们提出了一种自相似条件来实现适应性。重要的是，我们证明了自相似条件不会损害问题本身的复杂性，因为它保持了渐近遗憾下界Ω(T^((β+1)/(2β+1)))。此外，我们开发了一种平滑自适应的动态定价算法，并理论上证明了该算法的有效性。

    We study the dynamic pricing problem where the demand function is nonparametric and H\"older smooth, and we focus on adaptivity to the unknown H\"older smoothness parameter $\beta$ of the demand function. Traditionally the optimal dynamic pricing algorithm heavily relies on the knowledge of $\beta$ to achieve a minimax optimal regret of $\widetilde{O}(T^{\frac{\beta+1}{2\beta+1}})$. However, we highlight the challenge of adaptivity in this dynamic pricing problem by proving that no pricing policy can adaptively achieve this minimax optimal regret without knowledge of $\beta$. Motivated by the impossibility result, we propose a self-similarity condition to enable adaptivity. Importantly, we show that the self-similarity condition does not compromise the problem's inherent complexity since it preserves the regret lower bound $\Omega(T^{\frac{\beta+1}{2\beta+1}})$. Furthermore, we develop a smoothness-adaptive dynamic pricing algorithm and theoretically prove that the algorithm achieves t
    
[^6]: 重新审视不可能性定理的证明方法

    A Reexamination of Proof Approaches for the Impossibility Theorem. (arXiv:2309.06753v1 [econ.TH])

    [http://arxiv.org/abs/2309.06753](http://arxiv.org/abs/2309.06753)

    本研究使用逻辑的证明演算提出了对不可能性定理的证明方法，该方法保证了对所有可能社会福利函数的所有配置进行了检验，从而证明了该定理。

    

    决定性集合和关键选民方法已被用于证明阿罗不可能性定理。这些方法的证明仅考虑了所有可能社会福利函数的子集，并审查了这些函数域的部分。因此，这两个思路都不能有效证明该定理。本研究提出了一种基于逻辑的证明演算。在前提、公理和定理的条件之间进行有效的演绎推理，并得出独裁的结论，保证了对所有可能社会福利函数的所有配置进行了检验，从而证明了该定理。

    The decisive-set and pivotal-voter approaches have been used for proving Arrow's impossibility theorem. Proofs by these approaches consider only subsets of all possible social welfare functions and examine parts of the domain of these functions. Hence, both ideas are not effective to prove the theorem. This study presents a proof using a proof calculus in logic. A valid deductive inference between the premises, the axioms and conditions of the theorem, and the conclusion, dictatorship, guarantees that every profile of all possible social welfare functions is examined, thereby the theorem is established.
    
[^7]: 动态定价和汇聚网络的市场设计

    Market Design for Dynamic Pricing and Pooling in Capacitated Networks. (arXiv:2307.03994v1 [cs.GT])

    [http://arxiv.org/abs/2307.03994](http://arxiv.org/abs/2307.03994)

    本研究提出了一种用于动态定价和汇聚网络的市场设计，通过设置边缘价格激励代理商共享有限网络容量。在考虑了整数和网络约束以及代理商异质偏好的情况下，我们提供了充分条件，保证市场均衡的存在和多项式时间计算，并识别了实现最大效用的特定市场均衡。

    

    我们研究了一种市场机制，通过设置边缘价格来激励战略性代理商组织旅行，以有效共享有限的网络容量。该市场允许代理商组成团队共享旅行，做出出发时间和路线选择的决策，并支付边缘价格和其他成本。我们发展了一种新的方法来分析市场均衡的存在和计算，建立在组合拍卖和动态网络流理论的基础上。我们的方法解决了市场均衡特征化中的挑战，包括：（a）共享有限边缘容量中旅行的动态流量所引发的整数和网络约束；（b）战略性代理商的异质和私人偏好。我们提供了关于网络拓扑和代理商偏好的充分条件，以确保市场均衡的存在和多项式时间计算。我们确定了一个特定的市场均衡，实现了所有代理商的最大效用，并且与经典的最大流最小割问题等价。

    We study a market mechanism that sets edge prices to incentivize strategic agents to organize trips that efficiently share limited network capacity. This market allows agents to form groups to share trips, make decisions on departure times and route choices, and make payments to cover edge prices and other costs. We develop a new approach to analyze the existence and computation of market equilibrium, building on theories of combinatorial auctions and dynamic network flows. Our approach tackles the challenges in market equilibrium characterization arising from: (a) integer and network constraints on the dynamic flow of trips in sharing limited edge capacity; (b) heterogeneous and private preferences of strategic agents. We provide sufficient conditions on the network topology and agents' preferences that ensure the existence and polynomial-time computation of market equilibrium. We identify a particular market equilibrium that achieves maximum utilities for all agents, and is equivalen
    
[^8]: 处理数量级达数千万观测值的分位数回归的快速推断方法

    Fast Inference for Quantile Regression with Tens of Millions of Observations. (arXiv:2209.14502v5 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2209.14502](http://arxiv.org/abs/2209.14502)

    本文提出了一个快速推断框架，利用随机次梯度下降更新处理数量级达数千万观测值的分位数回归问题。通过顺序处理数据、聚合参数估计和计算关键统计量的解路径，该方法在处理超大规模数据集时具有较低的计算资源和内存需求。

    

    大数据分析在经济研究中开辟了新的道路，但是分析具有数千万观测值的数据集的挑战是巨大的。传统的基于极值估计的计量经济学方法需要大量的计算资源和内存，这些资源通常并不容易获得。本文针对应用于“超大规模”数据集（例如美国十年一次的人口普查数据）的线性分位数回归，提出了一个快速推断框架，利用随机次梯度下降（S-subGD）更新。推断过程按顺序处理横截面数据：(i) 对每个新观测值进行参数估计的更新，(ii) 将其作为 $\textit{Polyak-Ruppert}$ 平均值进行聚合，(iii) 使用解路径仅计算用于推断的关键统计量。该方法借鉴了时间序列回归的思想，通过随机缩放创建一个渐近可靠的统计量。我们提出的检验统计量是在完全在线的情况下计算的。

    Big data analytics has opened new avenues in economic research, but the challenge of analyzing datasets with tens of millions of observations is substantial. Conventional econometric methods based on extreme estimators require large amounts of computing resources and memory, which are often not readily available. In this paper, we focus on linear quantile regression applied to "ultra-large" datasets, such as U.S. decennial censuses. A fast inference framework is presented, utilizing stochastic subgradient descent (S-subGD) updates. The inference procedure handles cross-sectional data sequentially: (i) updating the parameter estimate with each incoming "new observation", (ii) aggregating it as a $\textit{Polyak-Ruppert}$ average, and (iii) computing a pivotal statistic for inference using only a solution path. The methodology draws from time-series regression to create an asymptotically pivotal statistic through random scaling. Our proposed test statistic is calculated in a fully online
    
[^9]: 多个工具变量的向量单调性假设

    A Vector Monotonicity Assumption for Multiple Instruments. (arXiv:2009.00553v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2009.00553](http://arxiv.org/abs/2009.00553)

    本文提出了向量单调性假设的观点，该假设对于多个工具变量组合的研究具有重要意义。通过假设所有工具变量的治疗接受程度是单调的，我们可以得到一些因果参数的点识别，并提供了相应的估计方法。

    

    当研究人员将多个工具变量组合用于单一二元治疗时，本地平均治疗效应（LATE）框架的单调性假设可能变得过于限制性：它要求所有单位在分别改变工具变量方向时具有相同的反应方向。相比之下，我所称的向量单调性仅仅假设所有工具变量的治疗接受程度是单调的，它是Mogstad等人引入的部分单调性假设的特殊情况。在工具变量为二元变量时，我刻画了在向量单调性下被点识别的因果参数类。该类包括对任何一种方式对工具变量集合做出响应的单位的平均治疗效应，或对给定子集做出响应的单位的平均治疗效应。识别结果是建设性的，并提供了对已识别的治疗效应参数的简单估计器。

    When a researcher combines multiple instrumental variables for a single binary treatment, the monotonicity assumption of the local average treatment effects (LATE) framework can become restrictive: it requires that all units share a common direction of response even when separate instruments are shifted in opposing directions. What I call vector monotonicity, by contrast, simply assumes treatment uptake to be monotonic in all instruments, representing a special case of the partial monotonicity assumption introduced by Mogstad et al. (2021). I characterize the class of causal parameters that are point identified under vector monotonicity, when the instruments are binary. This class includes, for example, the average treatment effect among units that are in any way responsive to the collection of instruments, or those that are responsive to a given subset of them. The identification results are constructive and yield a simple estimator for the identified treatment effect parameters. An e
    

