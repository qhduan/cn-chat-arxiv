# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multivariate Probabilistic Time Series Forecasting with Correlated Errors](https://rss.arxiv.org/abs/2402.01000) | 本文提出了一种方法，基于低秩加对角线参数化协方差矩阵，可以有效地刻画时间序列预测中误差的自相关性，并具有复杂度低、校准预测准确性高等优点。 |
| [^2] | [Optimal Flow Matching: Learning Straight Trajectories in Just One Step](https://arxiv.org/abs/2403.13117) | 该论文提出了一种新颖的最优流匹配方法，能够在一步中学习实现二次成本下的直线 OT 位移。 |
| [^3] | [Learning high-dimensional targets by two-parameter models and gradient flow](https://arxiv.org/abs/2402.17089) | 通过提出两参数模型和梯度流学习高维目标的理论可能性，研究发现在特定条件下存在大量不可学习目标，并且这些目标的集合不密集，具有一定拓扑性质的子集中也存在不可学习目标。最终，发现使用层次过程构建的主要定理模型在数学表达上并非由单一初等函数表示。 |
| [^4] | [Is K-fold cross validation the best model selection method for Machine Learning?.](http://arxiv.org/abs/2401.16407) | K折交叉验证在机器学习中是常用的模型选择方法，但在处理小样本数据集和异质数据源时存在困难。 |
| [^5] | [Bayesian Nonparametrics meets Data-Driven Robust Optimization.](http://arxiv.org/abs/2401.15771) | 本文提出了一种将贝叶斯非参数方法与最新的决策理论模型相结合的鲁棒优化准则，通过这种方法，可以在线性回归问题中获得有稳定性和优越性能的结果。 |
| [^6] | [Beyond Regrets: Geometric Metrics for Bayesian Optimization.](http://arxiv.org/abs/2401.01981) | 本论文提出了四个新的几何度量，可以比较贝叶斯优化算法在考虑查询点和全局最优解的几何特性时的性能。 |
| [^7] | [TomOpt: Differential optimisation for task- and constraint-aware design of particle detectors in the context of muon tomography.](http://arxiv.org/abs/2309.14027) | TomOpt是一个软件包，用于优化宇宙射线μ子断层扫描设计中的微粒探测器的几何布局和规格。它利用可微分编程模拟μ子与探测器和扫描体积的相互作用，并通过损失最小化的优化循环进行推断感知优化。 |
| [^8] | [PAC Neural Prediction Set Learning to Quantify the Uncertainty of Generative Language Models.](http://arxiv.org/abs/2307.09254) | 本文提出了一种使用神经网络来量化生成式语言模型不确定性的PAC神经预测集学习方法，通过在多种语言数据集和模型上的实验证明，相比于标准基准方法，我们的方法平均提高了63％的量化不确定性。 |
| [^9] | [Encoding Domain Expertise into Multilevel Models for Source Location.](http://arxiv.org/abs/2305.08657) | 本文提出了一种基于贝叶斯多级模型的方法，可以将群体数据视为整体来考虑，并将领域专业知识和物理知识编码到模型中，以实现源位置的定位。 |
| [^10] | [Pseudo-Labeling for Kernel Ridge Regression under Covariate Shift.](http://arxiv.org/abs/2302.10160) | 该论文提出了一种关于核岭回归的协变量转移策略，通过使用伪标签进行模型选择，能够适应不同特征分布下的学习，实现均方误差最小化。 |

# 详细

[^1]: 多元概率时间序列预测与相关误差

    Multivariate Probabilistic Time Series Forecasting with Correlated Errors

    [https://rss.arxiv.org/abs/2402.01000](https://rss.arxiv.org/abs/2402.01000)

    本文提出了一种方法，基于低秩加对角线参数化协方差矩阵，可以有效地刻画时间序列预测中误差的自相关性，并具有复杂度低、校准预测准确性高等优点。

    

    建模误差之间的相关性与模型能够准确量化概率时间序列预测中的预测不确定性密切相关。最近的多元模型在考虑误差之间的同时相关性方面取得了显著进展，然而，对于统计简化的目的，对这些误差的常见假设是它们在时间上是独立的。然而，实际观测往往偏离了这个假设，因为误差通常由于各种因素（如排除时间相关的协变量）而表现出显著的自相关性。在这项工作中，我们提出了一种基于低秩加对角线参数化协方差矩阵的高效方法，可以有效地刻画误差的自相关性。所提出的方法具有几个可取的特性：复杂度不随时间序列数目增加，得到的协方差可以用于校准预测，且具有较好的性能。

    Modeling the correlations among errors is closely associated with how accurately the model can quantify predictive uncertainty in probabilistic time series forecasting. Recent multivariate models have made significant progress in accounting for contemporaneous correlations among errors, while a common assumption on these errors is that they are temporally independent for the sake of statistical simplicity. However, real-world observations often deviate from this assumption, since errors usually exhibit substantial autocorrelation due to various factors such as the exclusion of temporally correlated covariates. In this work, we propose an efficient method, based on a low-rank-plus-diagonal parameterization of the covariance matrix, which can effectively characterize the autocorrelation of errors. The proposed method possesses several desirable properties: the complexity does not scale with the number of time series, the resulting covariance can be used for calibrating predictions, and i
    
[^2]: 最优流匹配：在一步中学习直线轨迹

    Optimal Flow Matching: Learning Straight Trajectories in Just One Step

    [https://arxiv.org/abs/2403.13117](https://arxiv.org/abs/2403.13117)

    该论文提出了一种新颖的最优流匹配方法，能够在一步中学习实现二次成本下的直线 OT 位移。

    

    在过去几年中，流匹配方法在生成建模中得到了蓬勃发展。社区追求的一个引人注目的属性是能够学习具有直线轨迹的流，这些轨迹实现了最优输运（OT）置换。直线性对于快速集成学习流的路径至关重要。不幸的是，大多数现有的流直线化方法都基于非平凡的迭代过程，在训练过程中积累误差或利用启发式小批量OT近似。为解决这一问题，我们开发了一种新颖的最优流匹配方法，仅通过一次流匹配步骤即可为二次成本恢复直线OT置换。

    arXiv:2403.13117v1 Announce Type: cross  Abstract: Over the several recent years, there has been a boom in development of flow matching methods for generative modeling. One intriguing property pursued by the community is the ability to learn flows with straight trajectories which realize the optimal transport (OT) displacements. Straightness is crucial for fast integration of the learned flow's paths. Unfortunately, most existing flow straightening methods are based on non-trivial iterative procedures which accumulate the error during training or exploit heuristic minibatch OT approximations. To address this issue, we develop a novel optimal flow matching approach which recovers the straight OT displacement for the quadratic cost in just one flow matching step.
    
[^3]: 通过两参数模型和梯度流学习高维目标

    Learning high-dimensional targets by two-parameter models and gradient flow

    [https://arxiv.org/abs/2402.17089](https://arxiv.org/abs/2402.17089)

    通过提出两参数模型和梯度流学习高维目标的理论可能性，研究发现在特定条件下存在大量不可学习目标，并且这些目标的集合不密集，具有一定拓扑性质的子集中也存在不可学习目标。最终，发现使用层次过程构建的主要定理模型在数学表达上并非由单一初等函数表示。

    

    我们探讨了当$W<d$时，通过梯度流（GF）以$W$参数模型学习$d$维目标的理论可能性，必然存在GF-不可学习目标的大子集。特别是，可学习目标的集合在$\mathbb R^d$中不是密集的，任何形同$W$维球面的$\mathbb R^d$子集包含不可学习目标。最后，我们观察到在几乎保证二参数学习的主要定理中，所述模型是通过层次过程构建的，因此不能用单个初等函数表达。我们展示了这种限制在本质上是必要的，因为这种可学习性对于许多初等函数类的可学习性是被排除的。

    arXiv:2402.17089v1 Announce Type: cross  Abstract: We explore the theoretical possibility of learning $d$-dimensional targets with $W$-parameter models by gradient flow (GF) when $W<d$ there is necessarily a large subset of GF-non-learnable targets. In particular, the set of learnable targets is not dense in $\mathbb R^d$, and any subset of $\mathbb R^d$ homeomorphic to the $W$-dimensional sphere contains non-learnable targets. Finally, we observe that the model in our main theorem on almost guaranteed two-parameter learning is constructed using a hierarchical procedure and as a result is not expressible by a single elementary function. We show that this limitation is essential in the sense that such learnability can be ruled out for a large class of elementary functions.
    
[^4]: K折交叉验证是否是机器学习中最好的模型选择方法？

    Is K-fold cross validation the best model selection method for Machine Learning?. (arXiv:2401.16407v1 [stat.ML])

    [http://arxiv.org/abs/2401.16407](http://arxiv.org/abs/2401.16407)

    K折交叉验证在机器学习中是常用的模型选择方法，但在处理小样本数据集和异质数据源时存在困难。

    

    机器学习作为一种能够紧凑表示复杂模式的技术，具有显著的预测推理潜力。K折交叉验证（CV）是确定机器学习结果是否是随机生成的最常用方法，并经常优于传统的假设检验。这种改进利用了直接从机器学习分类中获得的度量，比如准确性，这些度量没有参数描述。为了在机器学习流程中进行频率分析，可以添加排列测试或来自数据分区（即折叠）的简单统计量来估计置信区间。不幸的是，无论是参数化还是非参数化测试都无法解决围绕分割小样本数据集和来自异质数据源的学习固有问题。机器学习严重依赖学习参数和数据在折叠中的分布，这重新概括了熟悉的困难情况。

    As a technique that can compactly represent complex patterns, machine learning has significant potential for predictive inference. K-fold cross-validation (CV) is the most common approach to ascertaining the likelihood that a machine learning outcome is generated by chance and frequently outperforms conventional hypothesis testing. This improvement uses measures directly obtained from machine learning classifications, such as accuracy, that do not have a parametric description. To approach a frequentist analysis within machine learning pipelines, a permutation test or simple statistics from data partitions (i.e. folds) can be added to estimate confidence intervals. Unfortunately, neither parametric nor non-parametric tests solve the inherent problems around partitioning small sample-size datasets and learning from heterogeneous data sources. The fact that machine learning strongly depends on the learning parameters and the distribution of data across folds recapitulates familiar diffic
    
[^5]: 贝叶斯非参数方法与数据驱动鲁棒优化的结合

    Bayesian Nonparametrics meets Data-Driven Robust Optimization. (arXiv:2401.15771v1 [stat.ML])

    [http://arxiv.org/abs/2401.15771](http://arxiv.org/abs/2401.15771)

    本文提出了一种将贝叶斯非参数方法与最新的决策理论模型相结合的鲁棒优化准则，通过这种方法，可以在线性回归问题中获得有稳定性和优越性能的结果。

    

    训练机器学习和统计模型通常涉及优化数据驱动的风险准则。风险通常是根据经验数据分布计算的，但由于分布不确定性，这可能导致性能不稳定和不好的样本外表现。在分布鲁棒优化的精神下，我们提出了一个新颖的鲁棒准则，将贝叶斯非参数（即狄利克雷过程）理论和最近的平滑模糊规避偏好的决策理论模型的见解相结合。首先，我们强调了与标准正则化经验风险最小化技术的新连接，其中包括岭回归和套索回归。然后，我们从理论上证明了鲁棒优化过程在有限样本和渐近统计保证方面的有利性存在。对于实际实施，我们提出并研究了基于众所周知的狄利克雷过程表示的可行近似准则。

    Training machine learning and statistical models often involves optimizing a data-driven risk criterion. The risk is usually computed with respect to the empirical data distribution, but this may result in poor and unstable out-of-sample performance due to distributional uncertainty. In the spirit of distributionally robust optimization, we propose a novel robust criterion by combining insights from Bayesian nonparametric (i.e., Dirichlet Process) theory and recent decision-theoretic models of smooth ambiguity-averse preferences. First, we highlight novel connections with standard regularized empirical risk minimization techniques, among which Ridge and LASSO regressions. Then, we theoretically demonstrate the existence of favorable finite-sample and asymptotic statistical guarantees on the performance of the robust optimization procedure. For practical implementation, we propose and study tractable approximations of the criterion based on well-known Dirichlet Process representations. 
    
[^6]: 超越遗憾：贝叶斯优化的几何度量

    Beyond Regrets: Geometric Metrics for Bayesian Optimization. (arXiv:2401.01981v1 [cs.LG])

    [http://arxiv.org/abs/2401.01981](http://arxiv.org/abs/2401.01981)

    本论文提出了四个新的几何度量，可以比较贝叶斯优化算法在考虑查询点和全局最优解的几何特性时的性能。

    

    贝叶斯优化是一种针对黑盒子目标函数的原则性优化策略。它在科学发现和实验设计等各种实际应用中的效果得到了证明。通常，贝叶斯优化的性能是通过基于遗憾的度量来评估的，如瞬时遗憾、简单遗憾和累积遗憾。这些度量仅依赖于函数评估，因此它们不考虑查询点和全局解之间的几何关系，也不考虑查询点本身。值得注意的是，它们不能区分是否成功找到了多个全局解。此外，它们也不能评估贝叶斯优化在给定搜索空间中利用和探索的能力。为了解决这些问题，我们提出了四个新的几何度量，即精确度、召回率、平均度和平均距离。这些度量使我们能够比较考虑查询点和全局最优解的几何特性的贝叶斯优化算法。

    Bayesian optimization is a principled optimization strategy for a black-box objective function. It shows its effectiveness in a wide variety of real-world applications such as scientific discovery and experimental design. In general, the performance of Bayesian optimization is assessed by regret-based metrics such as instantaneous, simple, and cumulative regrets. These metrics only rely on function evaluations, so that they do not consider geometric relationships between query points and global solutions, or query points themselves. Notably, they cannot discriminate if multiple global solutions are successfully found. Moreover, they do not evaluate Bayesian optimization's abilities to exploit and explore a search space given. To tackle these issues, we propose four new geometric metrics, i.e., precision, recall, average degree, and average distance. These metrics allow us to compare Bayesian optimization algorithms considering the geometry of both query points and global optima, or que
    
[^7]: TomOpt：在宇宙射线μ子断层扫描中面向任务和约束感知设计的微粒探测器的差分优化

    TomOpt: Differential optimisation for task- and constraint-aware design of particle detectors in the context of muon tomography. (arXiv:2309.14027v1 [physics.ins-det])

    [http://arxiv.org/abs/2309.14027](http://arxiv.org/abs/2309.14027)

    TomOpt是一个软件包，用于优化宇宙射线μ子断层扫描设计中的微粒探测器的几何布局和规格。它利用可微分编程模拟μ子与探测器和扫描体积的相互作用，并通过损失最小化的优化循环进行推断感知优化。

    

    我们描述了一个名为TomOpt的软件包，用于优化几何布局和探测器规格，以进行宇宙射线μ子的散射断层扫描设计。该软件利用可微分编程来模拟μ子与探测器和扫描体积的相互作用，推断体积属性，并进行损失最小化的优化循环。通过这样做，我们首次演示了粒子物理仪器的端到端可微分和推断感知优化。我们研究了该软件在相关基准场景上的性能，并讨论了其潜在应用。

    We describe a software package, TomOpt, developed to optimise the geometrical layout and specifications of detectors designed for tomography by scattering of cosmic-ray muons. The software exploits differentiable programming for the modeling of muon interactions with detectors and scanned volumes, the inference of volume properties, and the optimisation cycle performing the loss minimisation. In doing so, we provide the first demonstration of end-to-end-differentiable and inference-aware optimisation of particle physics instruments. We study the performance of the software on a relevant benchmark scenarios and discuss its potential applications.
    
[^8]: 用于量化生成式语言模型不确定性的PAC神经预测集学习

    PAC Neural Prediction Set Learning to Quantify the Uncertainty of Generative Language Models. (arXiv:2307.09254v1 [cs.LG])

    [http://arxiv.org/abs/2307.09254](http://arxiv.org/abs/2307.09254)

    本文提出了一种使用神经网络来量化生成式语言模型不确定性的PAC神经预测集学习方法，通过在多种语言数据集和模型上的实验证明，相比于标准基准方法，我们的方法平均提高了63％的量化不确定性。

    

    学习和量化模型的不确定性是增强模型可信度的关键任务。由于对生成虚构事实的担忧，最近兴起的生成式语言模型（GLM）特别强调可靠的不确定性量化的需求。本文提出了一种学习神经预测集模型的方法，该方法能够以可能近似正确（PAC）的方式量化GLM的不确定性。与现有的预测集模型通过标量值参数化不同，我们提出通过神经网络参数化预测集，实现更精确的不确定性量化，但仍满足PAC保证。通过在四种类型的语言数据集和六种类型的模型上展示，我们的方法相比标准基准方法平均提高了63％的量化不确定性。

    Uncertainty learning and quantification of models are crucial tasks to enhance the trustworthiness of the models. Importantly, the recent surge of generative language models (GLMs) emphasizes the need for reliable uncertainty quantification due to the concerns on generating hallucinated facts. In this paper, we propose to learn neural prediction set models that comes with the probably approximately correct (PAC) guarantee for quantifying the uncertainty of GLMs. Unlike existing prediction set models, which are parameterized by a scalar value, we propose to parameterize prediction sets via neural networks, which achieves more precise uncertainty quantification but still satisfies the PAC guarantee. We demonstrate the efficacy of our method on four types of language datasets and six types of models by showing that our method improves the quantified uncertainty by $63\%$ on average, compared to a standard baseline method.
    
[^9]: 将领域专业知识编码到多级模型中用于源位置的定位。

    Encoding Domain Expertise into Multilevel Models for Source Location. (arXiv:2305.08657v1 [stat.ML])

    [http://arxiv.org/abs/2305.08657](http://arxiv.org/abs/2305.08657)

    本文提出了一种基于贝叶斯多级模型的方法，可以将群体数据视为整体来考虑，并将领域专业知识和物理知识编码到模型中，以实现源位置的定位。

    

    在许多工业应用中，群体数据是普遍存在的。机器和基础设施越来越多地配备了传感系统，发出具有复杂相互依赖关系的遥测数据流。实际上，数据中心的监测程序倾向于将这些资产（以及各自的模型）视为不同的实体 - 独立运行并与独立数据相关联。相反，这项工作捕捉了一组系统模型之间的统计相关性和相互依赖关系。利用贝叶斯多级方法，数据的价值可以得到扩展，因为可以将人群作为一个整体来考虑，而不是作为组成部分。最有趣的是，领域专业知识和基础物理知识可以在系统、子组或人群水平上编码到模型中。我们提供了一个声发射（到达时间）映射源位置的示例，以说明多级模型如何自然地适用于表示。

    Data from populations of systems are prevalent in many industrial applications. Machines and infrastructure are increasingly instrumented with sensing systems, emitting streams of telemetry data with complex interdependencies. In practice, data-centric monitoring procedures tend to consider these assets (and respective models) as distinct -- operating in isolation and associated with independent data. In contrast, this work captures the statistical correlations and interdependencies between models of a group of systems. Utilising a Bayesian multilevel approach, the value of data can be extended, since the population can be considered as a whole, rather than constituent parts. Most interestingly, domain expertise and knowledge of the underlying physics can be encoded in the model at the system, subgroup, or population level. We present an example of acoustic emission (time-of-arrival) mapping for source location, to illustrate how multilevel models naturally lend themselves to represent
    
[^10]: 核岭回归下伪标签的协变量转移策略

    Pseudo-Labeling for Kernel Ridge Regression under Covariate Shift. (arXiv:2302.10160v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2302.10160](http://arxiv.org/abs/2302.10160)

    该论文提出了一种关于核岭回归的协变量转移策略，通过使用伪标签进行模型选择，能够适应不同特征分布下的学习，实现均方误差最小化。

    

    我们提出并分析了一种基于协变量转移的核岭回归方法。我们的目标是在目标分布上学习一个均方误差最小的回归函数，基于从目标分布采样的未标记数据和可能具有不同特征分布的已标记数据。我们将已标记数据分成两个子集，并分别进行核岭回归，以获得候选模型集合和一个填充模型。我们使用后者填充缺失的标签，然后相应地选择最佳的候选模型。我们的非渐近性过量风险界表明，在相当一般的情况下，我们的估计器能够适应目标分布以及协变量转移的结构。它能够实现渐近正态误差率直到对数因子的最小极限优化。在模型选择中使用伪标签不会产生主要负面影响。

    We develop and analyze a principled approach to kernel ridge regression under covariate shift. The goal is to learn a regression function with small mean squared error over a target distribution, based on unlabeled data from there and labeled data that may have a different feature distribution. We propose to split the labeled data into two subsets and conduct kernel ridge regression on them separately to obtain a collection of candidate models and an imputation model. We use the latter to fill the missing labels and then select the best candidate model accordingly. Our non-asymptotic excess risk bounds show that in quite general scenarios, our estimator adapts to the structure of the target distribution as well as the covariate shift. It achieves the minimax optimal error rate up to a logarithmic factor. The use of pseudo-labels in model selection does not have major negative impacts.
    

