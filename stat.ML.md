# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Beyond Lengthscales: No-regret Bayesian Optimisation With Unknown Hyperparameters Of Any Type](https://rss.arxiv.org/abs/2402.01632) | 这篇论文提出了一种新的贝叶斯优化算法，可以处理具有任意类型未知超参数的情况，并具有无遗憾特性。 |
| [^2] | [Adaptive Optimization for Prediction with Missing Data](https://rss.arxiv.org/abs/2402.01543) | 本文提出了一种针对缺失数据预测的自适应优化方法，通过自适应线性回归模型来适应观测特征集，并将填充规则和回归模型同时学习，相比顺序学习方法，在数据非完全随机缺失情况下，方法实现了2-10%的准确性改进。 |
| [^3] | [On Convergence of Adam for Stochastic Optimization under Relaxed Assumptions](https://arxiv.org/abs/2402.03982) | 本文研究了在宽松假设下的随机优化中Adam算法的收敛性。我们引入了一个全面的噪声模型，并证明了在这个模型下，Adam算法可以以较高的概率高效地寻找到一个稳定点。与其他随机一阶算法相比，Adam算法具有更好的自适应性能，无需调整步长和问题参数。 |
| [^4] | [A flexible Bayesian g-formula for causal survival analyses with time-dependent confounding](https://arxiv.org/abs/2402.02306) | 本文提出了一种更灵活的贝叶斯g形式估计器，用于具有时变混杂的因果生存分析。它采用贝叶斯附加回归树来模拟时变生成组件，并引入了纵向平衡分数以降低模型错误规范引起的偏差。 |
| [^5] | [Score-based Causal Representation Learning: Linear and General Transformations](https://arxiv.org/abs/2402.00849) | 这篇论文提出了一种基于得分的算法类，用于干预范围内的因果表示学习，涵盖了线性和一般转化。算法保证了可识别性和实现性，并且通过创造性地将得分函数与因果表示学习相结合。 |
| [^6] | [Causal Rule Learning: Enhancing the Understanding of Heterogeneous Treatment Effect via Weighted Causal Rules.](http://arxiv.org/abs/2310.06746) | 通过因果规则学习，我们可以利用加权因果规则来估计和加强对异质治疗效应的理解。 |
| [^7] | [Learning under Selective Labels with Heterogeneous Decision-makers: An Instrumental Variable Approach.](http://arxiv.org/abs/2306.07566) | 本文提出了一种处理选择性标记数据的学习问题的方法。通过利用历史决策由一组异质决策者做出的事实，我们建立了一种有原理的工具变量框架，并提出了一种加权学习方法，用于学习预测规则。 |

# 详细

[^1]: 超越尺度：具有任意类型未知超参数的无遗憾贝叶斯优化

    Beyond Lengthscales: No-regret Bayesian Optimisation With Unknown Hyperparameters Of Any Type

    [https://rss.arxiv.org/abs/2402.01632](https://rss.arxiv.org/abs/2402.01632)

    这篇论文提出了一种新的贝叶斯优化算法，可以处理具有任意类型未知超参数的情况，并具有无遗憾特性。

    

    贝叶斯优化需要拟合高斯过程模型，而拟合高斯过程模型需要指定超参数 - 大部分理论文献假设这些超参数是已知的。之前的理论研究通常假设数据在空间中均匀填充，而常用的高斯过程超参数的最大似然估计器只有在这种情况下才是一致的。然而，在贝叶斯优化中，数据不一定满足这种均匀填充的条件。由于无法保证超参数估计的正确性，并且这些超参数可以显著影响高斯过程拟合，因此对具有未知超参数的贝叶斯优化进行理论分析非常具有挑战性。之前提出的具有无遗憾特性的算法仅能处理特殊情况下的未知长度尺度、再生核希尔伯特空间范数，并且仅适用于频率派的情况。我们提出了一种新的算法，命名为HE-GP-UCB，它是第一个具有无遗憾特性的算法，在具有未知超参数的情况下实现了贝叶斯优化。

    Bayesian optimisation requires fitting a Gaussian process model, which in turn requires specifying hyperparameters - most of the theoretical literature assumes those hyperparameters are known. The commonly used maximum likelihood estimator for hyperparameters of the Gaussian process is consistent only if the data fills the space uniformly, which does not have to be the case in Bayesian optimisation. Since no guarantees exist regarding the correctness of hyperparameter estimation, and those hyperparameters can significantly affect the Gaussian process fit, theoretical analysis of Bayesian optimisation with unknown hyperparameters is very challenging. Previously proposed algorithms with the no-regret property were only able to handle the special case of unknown lengthscales, reproducing kernel Hilbert space norm and applied only to the frequentist case. We propose a novel algorithm, HE-GP-UCB, which is the first algorithm enjoying the no-regret property in the case of unknown hyperparame
    
[^2]: 针对缺失数据预测的自适应优化方法

    Adaptive Optimization for Prediction with Missing Data

    [https://rss.arxiv.org/abs/2402.01543](https://rss.arxiv.org/abs/2402.01543)

    本文提出了一种针对缺失数据预测的自适应优化方法，通过自适应线性回归模型来适应观测特征集，并将填充规则和回归模型同时学习，相比顺序学习方法，在数据非完全随机缺失情况下，方法实现了2-10%的准确性改进。

    

    在训练具有缺失条目的预测模型时，最常用和多功能的方法是一种流水线技术，首先填充缺失条目，然后计算预测结果。本文将缺失数据预测视为一个两阶段的自适应优化问题，并提出了一种新的模型类别，自适应线性回归模型，其中回归系数能够适应观测特征集。我们表明一些自适应线性回归模型等同于同时学习填充规则和下游线性回归模型而不是顺序学习。我们利用这种联合填充-回归的解释将我们的框架推广到非线性模型。在数据非完全随机缺失的情况下，我们的方法在样外准确性方面实现了2-10%的改进。

    When training predictive models on data with missing entries, the most widely used and versatile approach is a pipeline technique where we first impute missing entries and then compute predictions. In this paper, we view prediction with missing data as a two-stage adaptive optimization problem and propose a new class of models, adaptive linear regression models, where the regression coefficients adapt to the set of observed features. We show that some adaptive linear regression models are equivalent to learning an imputation rule and a downstream linear regression model simultaneously instead of sequentially. We leverage this joint-impute-then-regress interpretation to generalize our framework to non-linear models. In settings where data is strongly not missing at random, our methods achieve a 2-10% improvement in out-of-sample accuracy.
    
[^3]: 在宽松假设下关于随机优化中Adam收敛性的研究

    On Convergence of Adam for Stochastic Optimization under Relaxed Assumptions

    [https://arxiv.org/abs/2402.03982](https://arxiv.org/abs/2402.03982)

    本文研究了在宽松假设下的随机优化中Adam算法的收敛性。我们引入了一个全面的噪声模型，并证明了在这个模型下，Adam算法可以以较高的概率高效地寻找到一个稳定点。与其他随机一阶算法相比，Adam算法具有更好的自适应性能，无需调整步长和问题参数。

    

    适应性动量评估（Adam）算法在训练各种深度学习任务中非常有效。尽管如此，在非凸光滑场景下，特别是在可能存在无界梯度和仿射方差噪声的情况下，对于Adam的理论理解仍然有限。在本文中，我们研究了在这些具有挑战性条件下的普通Adam算法。我们引入了一个全面的噪声模型，该模型控制着仿射方差噪声、有界噪声和次高斯噪声。我们证明了在这个通用噪声模型下，Adam算法可以以$\mathcal{O}(\text{poly}(\log T)/\sqrt{T})$的概率高效地寻找到一个稳定点，其中$T$表示总迭代次数，与随机一阶算法的更底效率相匹配。更重要的是，我们揭示了在相同条件下，Adam算法无需调整步长和任何问题参数，具有比随机梯度下降更好的自适应性能。

    The Adaptive Momentum Estimation (Adam) algorithm is highly effective in training various deep learning tasks. Despite this, there's limited theoretical understanding for Adam, especially when focusing on its vanilla form in non-convex smooth scenarios with potential unbounded gradients and affine variance noise. In this paper, we study vanilla Adam under these challenging conditions. We introduce a comprehensive noise model which governs affine variance noise, bounded noise and sub-Gaussian noise. We show that Adam can find a stationary point with a $\mathcal{O}(\text{poly}(\log T)/\sqrt{T})$ rate in high probability under this general noise model where $T$ denotes total number iterations, matching the lower rate of stochastic first-order algorithms up to logarithm factors. More importantly, we reveal that Adam is free of tuning step-sizes with any problem-parameters, yielding a better adaptation property than the Stochastic Gradient Descent under the same conditions. We also provide 
    
[^4]: 弹性贝叶斯g形式在具有时变混杂的因果生存分析中的应用

    A flexible Bayesian g-formula for causal survival analyses with time-dependent confounding

    [https://arxiv.org/abs/2402.02306](https://arxiv.org/abs/2402.02306)

    本文提出了一种更灵活的贝叶斯g形式估计器，用于具有时变混杂的因果生存分析。它采用贝叶斯附加回归树来模拟时变生成组件，并引入了纵向平衡分数以降低模型错误规范引起的偏差。

    

    在具有时间至事件结果的纵向观察性研究中，因果分析的常见目标是在研究群体中估计在假设干预情景下的因果生存曲线。g形式是这种分析的一个特别有用的工具。为了增强传统的参数化g形式方法，我们开发了一种更灵活的贝叶斯g形式估计器。该估计器同时支持纵向预测和因果推断。它在模拟时变生成组件的建模中引入了贝叶斯附加回归树，旨在减轻由于模型错误规范造成的偏差。具体而言，我们引入了一类更通用的离散生存数据g形式。这些公式可以引入纵向平衡分数，这在处理越来越多的时变混杂因素时是一种有效的降维方法。

    In longitudinal observational studies with a time-to-event outcome, a common objective in causal analysis is to estimate the causal survival curve under hypothetical intervention scenarios within the study cohort. The g-formula is a particularly useful tool for this analysis. To enhance the traditional parametric g-formula approach, we developed a more adaptable Bayesian g-formula estimator. This estimator facilitates both longitudinal predictive and causal inference. It incorporates Bayesian additive regression trees in the modeling of the time-evolving generative components, aiming to mitigate bias due to model misspecification. Specifically, we introduce a more general class of g-formulas for discrete survival data. These formulas can incorporate the longitudinal balancing scores, which serve as an effective method for dimension reduction and are vital when dealing with an expanding array of time-varying confounders. The minimum sufficient formulation of these longitudinal balancing
    
[^5]: 基于得分的因果表示学习：线性和一般的转化

    Score-based Causal Representation Learning: Linear and General Transformations

    [https://arxiv.org/abs/2402.00849](https://arxiv.org/abs/2402.00849)

    这篇论文提出了一种基于得分的算法类，用于干预范围内的因果表示学习，涵盖了线性和一般转化。算法保证了可识别性和实现性，并且通过创造性地将得分函数与因果表示学习相结合。

    

    本篇论文针对一般非参数潜在因果模型和将潜在变量映射到观测变量的未知转化，研究了基于干预的因果表示学习（CRL）。研究了线性和一般的转化。这篇论文同时讨论了可识别性和实现性两个方面。可识别性是指确定算法不相关的条件，以确保恢复真实的潜在因果变量和潜在因果图。实现性是指算法方面，解决设计算法来实现可识别保证的问题。通过将得分函数（即密度函数对数的梯度）与CRL之间建立新联系，本文设计了一种得分为基础的算法类，确保了可识别性和实现性。首先，本文专注于线性转化，并展示了每个n个随机硬干预下该转化的因果表示可识别。

    This paper addresses intervention-based causal representation learning (CRL) under a general nonparametric latent causal model and an unknown transformation that maps the latent variables to the observed variables. Linear and general transformations are investigated. The paper addresses both the \emph{identifiability} and \emph{achievability} aspects. Identifiability refers to determining algorithm-agnostic conditions that ensure recovering the true latent causal variables and the latent causal graph underlying them. Achievability refers to the algorithmic aspects and addresses designing algorithms that achieve identifiability guarantees. By drawing novel connections between \emph{score functions} (i.e., the gradients of the logarithm of density functions) and CRL, this paper designs a \emph{score-based class of algorithms} that ensures both identifiability and achievability. First, the paper focuses on \emph{linear} transformations and shows that one stochastic hard intervention per n
    
[^6]: 因果规则学习：通过加权因果规则增强对异质治疗效应的理解

    Causal Rule Learning: Enhancing the Understanding of Heterogeneous Treatment Effect via Weighted Causal Rules. (arXiv:2310.06746v1 [cs.LG])

    [http://arxiv.org/abs/2310.06746](http://arxiv.org/abs/2310.06746)

    通过因果规则学习，我们可以利用加权因果规则来估计和加强对异质治疗效应的理解。

    

    解释性是利用机器学习方法估计异质治疗效应时的关键问题，特别是对于医疗应用来说，常常需要做出高风险决策。受到解释性的预测性、描述性、相关性框架的启发，我们提出了因果规则学习，该方法通过找到描述潜在子群的精细因果规则集来估计和增强我们对异质治疗效应的理解。因果规则学习包括三个阶段：规则发现、规则选择和规则分析。在规则发现阶段，我们利用因果森林生成一组具有相应子群平均治疗效应的因果规则池。选择阶段使用D-学习方法从这些规则中选择子集，将个体水平的治疗效应作为子群水平效应的线性组合进行解构。这有助于回答之前文献忽视的问题：如果一个个体同时属于多个不同的治疗子群，会怎么样呢？

    Interpretability is a key concern in estimating heterogeneous treatment effects using machine learning methods, especially for healthcare applications where high-stake decisions are often made. Inspired by the Predictive, Descriptive, Relevant framework of interpretability, we propose causal rule learning which finds a refined set of causal rules characterizing potential subgroups to estimate and enhance our understanding of heterogeneous treatment effects. Causal rule learning involves three phases: rule discovery, rule selection, and rule analysis. In the rule discovery phase, we utilize a causal forest to generate a pool of causal rules with corresponding subgroup average treatment effects. The selection phase then employs a D-learning method to select a subset of these rules to deconstruct individual-level treatment effects as a linear combination of the subgroup-level effects. This helps to answer an ignored question by previous literature: what if an individual simultaneously bel
    
[^7]: 学习选择标签下的异质决策者：一种工具变量方法

    Learning under Selective Labels with Heterogeneous Decision-makers: An Instrumental Variable Approach. (arXiv:2306.07566v1 [stat.ML])

    [http://arxiv.org/abs/2306.07566](http://arxiv.org/abs/2306.07566)

    本文提出了一种处理选择性标记数据的学习问题的方法。通过利用历史决策由一组异质决策者做出的事实，我们建立了一种有原理的工具变量框架，并提出了一种加权学习方法，用于学习预测规则。

    

    我们研究了在选择性标记数据下的学习问题。这种问题在历史决策导致结果仅部分标记时出现。标记数据分布可能与整体人群有显著差异，特别是当历史决策和目标结果可以同时受某些未观察到的因素影响时。因此，仅基于标记数据进行学习可能会导致在整体人群中的严重偏差。我们的论文通过利用许多应用中历史决策由一组异质决策者做出的事实来解决此挑战。具体而言，我们在一个有原理的工具变量框架下分析了这种设置。我们建立了满足观察到的数据时任何给定预测规则的全体风险的点识别条件，并在点识别失败时提供了尖锐的风险界限。我们进一步提出了一种加权学习方法，用于学习预测规则。

    We study the problem of learning with selectively labeled data, which arises when outcomes are only partially labeled due to historical decision-making. The labeled data distribution may substantially differ from the full population, especially when the historical decisions and the target outcome can be simultaneously affected by some unobserved factors. Consequently, learning with only the labeled data may lead to severely biased results when deployed to the full population. Our paper tackles this challenge by exploiting the fact that in many applications the historical decisions were made by a set of heterogeneous decision-makers. In particular, we analyze this setup in a principled instrumental variable (IV) framework. We establish conditions for the full-population risk of any given prediction rule to be point-identified from the observed data and provide sharp risk bounds when the point identification fails. We further propose a weighted learning approach that learns prediction ru
    

