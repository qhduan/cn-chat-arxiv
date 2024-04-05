# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Postprocessing of point predictions for probabilistic forecasting of electricity prices: Diversity matters](https://arxiv.org/abs/2404.02270) | 将点预测转换为概率预测的电力价格后处理方法中，结合Isotonic Distributional Regression与其他两种方法的预测分布可以实现显著的性能提升。 |
| [^2] | [Convergence Guarantees for RMSProp and Adam in Generalized-smooth Non-convex Optimization with Affine Noise Variance](https://arxiv.org/abs/2404.01436) | 本文提出了对于RMSProp和Adam在非凸优化中的紧致收敛性分析，首次展示了在最宽松的假设下的收敛性结果，并展示了RMSProp和Adam的迭代复杂度分别为$\mathcal O(\epsilon^{-4})$。 |
| [^3] | [LC-Tsalis-INF: Generalized Best-of-Both-Worlds Linear Contextual Bandits](https://arxiv.org/abs/2403.03219) | 该研究提出了一种广义最佳双赢线性背景强化型赌博机算法，能够在次优性差距受到下界限制时遗憾为$O(\log(T))$。同时引入了边缘条件来描述次优性差距对问题难度的影响。 |
| [^4] | [Sharp bounds for the max-sliced Wasserstein distance](https://arxiv.org/abs/2403.00666) | 对于可分希尔伯特空间上的概率测度与其经验分布之间的最大切片1-Wasserstein距离，得到了尖锐的上下界限。 |
| [^5] | [Better-than-KL PAC-Bayes Bounds](https://arxiv.org/abs/2402.09201) | 本文提出了一种更好的比KL PAC-Bayes界限方法来估计序列均值，应用于预测器泛化误差的估计。 |
| [^6] | [MMM and MMMSynth: Clustering of heterogeneous tabular data, and synthetic data generation.](http://arxiv.org/abs/2310.19454) | 该论文提出了MMM和MMMSynth算法，用于聚类异构表格数据和生成合成数据。MMM算法利用EM算法，在同类算法中表现更优，对于确定合成数据的聚类以及恢复真实数据的结构有较好的效果。 MMMSynth算法则用于从真实数据生成合成表格数据。 |
| [^7] | [DePAint: A Decentralized Safe Multi-Agent Reinforcement Learning Algorithm considering Peak and Average Constraints.](http://arxiv.org/abs/2310.14348) | 本文提出了一种分散的安全多智能体强化学习算法，能够在没有中央控制器的情况下，通过智能体之间的邻居通信来最大化累积奖励总和，并同时满足每个智能体的安全约束限制。 |
| [^8] | [Online Estimation with Rolling Validation: Adaptive Nonparametric Estimation with Stream Data.](http://arxiv.org/abs/2310.12140) | 本研究提出了一种在线估计方法，通过加权滚动验证过程来提高基本估计器的自适应收敛速度，并证明了这种方法的重要性和敏感性 |
| [^9] | [Incorporating Recklessness to Collaborative Filtering based Recommender Systems.](http://arxiv.org/abs/2308.02058) | 本文提出了一种将鲁莽行为引入基于矩阵分解的推荐系统学习过程的方法，通过控制风险水平来提高预测的数量和质量。 |
| [^10] | [Kernel Debiased Plug-in Estimation.](http://arxiv.org/abs/2306.08598) | 本文提出了一种高效、不需要实现影响函数且可计算的去偏插值估计方法。 |
| [^11] | [Long-term Forecasting with TiDE: Time-series Dense Encoder.](http://arxiv.org/abs/2304.08424) | TiDE是一种基于MLP的编码器-解码器模型，用于长期时间序列预测。它既具备线性模型的简单性和速度，又能处理协变量和非线性依赖，相较于最佳的Transformer模型，速度快5-10倍。 |
| [^12] | [Optimal inference of a generalised Potts model by single-layer transformers with factored attention.](http://arxiv.org/abs/2304.07235) | 我们将分析和数值推导结合，在基于广义 Potts 模型的数据上，对经过改进适应这种模型的self-attention机制进行训练，发现经过修改的self-attention机制可以在极限采样下准确学习Potts模型。这个“分解”注意力机制通过从数据中学习相关属性，可以提高Transformer的性能和可解释性。 |
| [^13] | [High-dimensional scaling limits and fluctuations of online least-squares SGD with smooth covariance.](http://arxiv.org/abs/2304.00707) | 本文针对在线最小二乘随机梯度下降（SGD）算法，以数据生成模型的性质为基础，建立了高维缩放极限与波动模型。在低、中、高噪声设置阶段，揭示了迭代的精确三阶段相变。 |
| [^14] | [Risk-Adaptive Approaches to Learning and Decision Making: A Survey.](http://arxiv.org/abs/2212.00856) | 本文调查了过去25年中风险测度的快速发展，介绍了其在各个领域的应用，以及与效用理论和分布鲁棒优化的关系，并指出了公平机器学习等新兴应用领域。 |
| [^15] | [Generalization in the Face of Adaptivity: A Bayesian Perspective.](http://arxiv.org/abs/2106.10761) | 本文提出面对自适应选择数据样本引起的过度拟合问题，使用噪声加算法可以提供不依赖于查询规模的误差保证，这一结果表明适应数据分析的问题在于新查询与过去查询的协方差。 |

# 详细

[^1]: 电力价格概率预测的点预测后处理：多样性至关重要

    Postprocessing of point predictions for probabilistic forecasting of electricity prices: Diversity matters

    [https://arxiv.org/abs/2404.02270](https://arxiv.org/abs/2404.02270)

    将点预测转换为概率预测的电力价格后处理方法中，结合Isotonic Distributional Regression与其他两种方法的预测分布可以实现显著的性能提升。

    

    依赖于电力价格的预测分布进行操作决策相较于仅基于点预测的决策可以带来显著更高的利润。然而，在学术和工业环境中开发的大多数模型仅提供点预测。为了解决这一问题，我们研究了三种将点预测转换为概率预测的后处理方法：分位数回归平均、一致性预测和最近引入的等温分布回归。我们发现，虽然等温分布回归表现最为多样化，但将其预测分布与另外两种方法结合使用，相较于具有正态分布误差的基准模型，在德国电力市场的4.5年测试期间（涵盖COVID大流行和乌克兰战争），实现了约7.5%的改进。值得注意的是，这种组合的性能与最先进的Dis

    arXiv:2404.02270v1 Announce Type: new  Abstract: Operational decisions relying on predictive distributions of electricity prices can result in significantly higher profits compared to those based solely on point forecasts. However, the majority of models developed in both academic and industrial settings provide only point predictions. To address this, we examine three postprocessing methods for converting point forecasts into probabilistic ones: Quantile Regression Averaging, Conformal Prediction, and the recently introduced Isotonic Distributional Regression. We find that while IDR demonstrates the most varied performance, combining its predictive distributions with those of the other two methods results in an improvement of ca. 7.5% compared to a benchmark model with normally distributed errors, over a 4.5-year test period in the German power market spanning the COVID pandemic and the war in Ukraine. Remarkably, the performance of this combination is at par with state-of-the-art Dis
    
[^2]: RMSProp和Adam在具有仿射噪声方差的广义光滑非凸优化中的收敛性保证

    Convergence Guarantees for RMSProp and Adam in Generalized-smooth Non-convex Optimization with Affine Noise Variance

    [https://arxiv.org/abs/2404.01436](https://arxiv.org/abs/2404.01436)

    本文提出了对于RMSProp和Adam在非凸优化中的紧致收敛性分析，首次展示了在最宽松的假设下的收敛性结果，并展示了RMSProp和Adam的迭代复杂度分别为$\mathcal O(\epsilon^{-4})$。

    

    本文在坐标级别广义光滑性和仿射噪声方差的最宽松假设下，为非凸优化中的RMSProp和Adam提供了首个收敛性分析。首先分析了RMSProp，它是一种具有自适应学习率但没有一阶动量的Adam的特例。具体地，为了解决自适应更新、无界梯度估计和Lipschitz常数之间的依赖挑战，我们证明了下降引理中的一阶项收敛，并且其分母由梯度范数的函数上界限制。基于这一结果，我们展示了使用适当的超参数的RMSProp收敛到一个$\epsilon$-稳定点，其迭代复杂度为$\mathcal O(\epsilon^{-4})$。然后，将我们的分析推广到Adam，额外的挑战是由于梯度与一阶动量之间的不匹配。我们提出了一个新的上界限制

    arXiv:2404.01436v1 Announce Type: cross  Abstract: This paper provides the first tight convergence analyses for RMSProp and Adam in non-convex optimization under the most relaxed assumptions of coordinate-wise generalized smoothness and affine noise variance. We first analyze RMSProp, which is a special case of Adam with adaptive learning rates but without first-order momentum. Specifically, to solve the challenges due to dependence among adaptive update, unbounded gradient estimate and Lipschitz constant, we demonstrate that the first-order term in the descent lemma converges and its denominator is upper bounded by a function of gradient norm. Based on this result, we show that RMSProp with proper hyperparameters converges to an $\epsilon$-stationary point with an iteration complexity of $\mathcal O(\epsilon^{-4})$. We then generalize our analysis to Adam, where the additional challenge is due to a mismatch between the gradient and first-order momentum. We develop a new upper bound on
    
[^3]: LC-Tsalis-INF: 广义最佳双赢线性背景强化型赌博机

    LC-Tsalis-INF: Generalized Best-of-Both-Worlds Linear Contextual Bandits

    [https://arxiv.org/abs/2403.03219](https://arxiv.org/abs/2403.03219)

    该研究提出了一种广义最佳双赢线性背景强化型赌博机算法，能够在次优性差距受到下界限制时遗憾为$O(\log(T))$。同时引入了边缘条件来描述次优性差距对问题难度的影响。

    

    本研究考虑具有独立同分布（i.i.d.）背景的线性背景强化型赌博机问题。在这个问题中，现有研究提出了最佳双赢（BoBW）算法，其遗憾在随机区域中满足$O(\log^2(T))$，其中$T$为回合数，其次优性差距由正常数下界，同时在对抗性区域中满足$O(\sqrt{T})$。然而，对$T$的依赖仍有改进空间，并且次优性差距的假设可以放宽。针对这个问题，本研究提出了一个算法，当次优性差距受到下界限制时，其遗憾满足$O(\log(T))$。此外，我们引入了一个边缘条件，即对次优性差距的一个更温和的假设。该条件使用参数$\beta \in (0, \infty]$表征与次优性差距相关的问题难度。然后我们证明该算法的遗憾满足$O\left(\

    arXiv:2403.03219v1 Announce Type: new  Abstract: This study considers the linear contextual bandit problem with independent and identically distributed (i.i.d.) contexts. In this problem, existing studies have proposed Best-of-Both-Worlds (BoBW) algorithms whose regrets satisfy $O(\log^2(T))$ for the number of rounds $T$ in a stochastic regime with a suboptimality gap lower-bounded by a positive constant, while satisfying $O(\sqrt{T})$ in an adversarial regime. However, the dependency on $T$ has room for improvement, and the suboptimality-gap assumption can be relaxed. For this issue, this study proposes an algorithm whose regret satisfies $O(\log(T))$ in the setting when the suboptimality gap is lower-bounded. Furthermore, we introduce a margin condition, a milder assumption on the suboptimality gap. That condition characterizes the problem difficulty linked to the suboptimality gap using a parameter $\beta \in (0, \infty]$. We then show that the algorithm's regret satisfies $O\left(\
    
[^4]: 最大切片Wasserstein距离的尖锐界限

    Sharp bounds for the max-sliced Wasserstein distance

    [https://arxiv.org/abs/2403.00666](https://arxiv.org/abs/2403.00666)

    对于可分希尔伯特空间上的概率测度与其经验分布之间的最大切片1-Wasserstein距离，得到了尖锐的上下界限。

    

    我们得到了关于在可分希尔伯特空间上的概率测度与从$n$个样本中获得的经验分布之间期望的最大切片1-Wasserstein距离的尖锐上下界。我们还得到了一个适用于Banach空间上的概率测度的版本。

    arXiv:2403.00666v1 Announce Type: cross  Abstract: We obtain sharp upper and lower bounds for the expected max-sliced 1-Wasserstein distance between a probability measure on a separable Hilbert space and its empirical distribution from $n$ samples. A version of this result for probability measures on Banach spaces is also obtained.
    
[^5]: 更好的比KL PAC-Bayes界限

    Better-than-KL PAC-Bayes Bounds

    [https://arxiv.org/abs/2402.09201](https://arxiv.org/abs/2402.09201)

    本文提出了一种更好的比KL PAC-Bayes界限方法来估计序列均值，应用于预测器泛化误差的估计。

    

    让$f(\theta, X_1),$ $ \dots,$ $ f(\theta, X_n)$成为一个随机元素序列，其中$f$是一个固定的标量函数，$X_1, \dots, X_n$是独立的随机变量（数据），而$\theta$是根据一些数据相关的后验分布$P_n$分布的随机参数。本文考虑了证明浓度不等式来估计序列均值的问题。这样一个问题的一个例子是对某些通过随机算法训练的预测器的泛化误差的估计，比如神经网络，其中$f$是一个损失函数。传统上，这个问题是通过PAC-Bayes分析来解决的，在这个分析中，除了后验分布，我们还选择一个能够捕捉到学习问题归纳偏差的先验分布。然后，PAC-Bayes浓度界限中的关键数量是一个能够捕捉到学习问题复杂性的分歧。

    arXiv:2402.09201v1 Announce Type: new Abstract: Let $f(\theta, X_1),$ $ \dots,$ $ f(\theta, X_n)$ be a sequence of random elements, where $f$ is a fixed scalar function, $X_1, \dots, X_n$ are independent random variables (data), and $\theta$ is a random parameter distributed according to some data-dependent posterior distribution $P_n$. In this paper, we consider the problem of proving concentration inequalities to estimate the mean of the sequence. An example of such a problem is the estimation of the generalization error of some predictor trained by a stochastic algorithm, such as a neural network where $f$ is a loss function. Classically, this problem is approached through a PAC-Bayes analysis where, in addition to the posterior, we choose a prior distribution which captures our belief about the inductive bias of the learning problem. Then, the key quantity in PAC-Bayes concentration bounds is a divergence that captures the complexity of the learning problem where the de facto stand
    
[^6]: MMM和MMMSynth：异构表格数据的聚类和合成数据生成

    MMM and MMMSynth: Clustering of heterogeneous tabular data, and synthetic data generation. (arXiv:2310.19454v1 [cs.LG])

    [http://arxiv.org/abs/2310.19454](http://arxiv.org/abs/2310.19454)

    该论文提出了MMM和MMMSynth算法，用于聚类异构表格数据和生成合成数据。MMM算法利用EM算法，在同类算法中表现更优，对于确定合成数据的聚类以及恢复真实数据的结构有较好的效果。 MMMSynth算法则用于从真实数据生成合成表格数据。

    

    我们提出了两个与异构表格数据相关的任务的新算法：聚类和合成数据生成。表格数据集通常由列中的异构数据类型（数值、有序、分类）组成，但行中可能还存在隐藏的聚类结构：例如，它们可能来自异构的（地理、社会经济、方法论）来源，因此所描述的结果变量（如疾病的存在）可能不仅依赖其他变量，还依赖于聚类上下文。此外，医学数据的共享通常受到患者隐私法律的限制，因此目前对于通过深度学习等方法从真实数据生成合成表格数据的算法非常感兴趣。我们展示了一种新颖的基于EM的聚类算法MMM（“Madras混合模型”），它在确定合成异构数据的聚类和恢复真实数据结构方面优于标准算法。基于此，我们可将MMM应用于数据合成任务的MMMSynth算法。

    We provide new algorithms for two tasks relating to heterogeneous tabular datasets: clustering, and synthetic data generation. Tabular datasets typically consist of heterogeneous data types (numerical, ordinal, categorical) in columns, but may also have hidden cluster structure in their rows: for example, they may be drawn from heterogeneous (geographical, socioeconomic, methodological) sources, such that the outcome variable they describe (such as the presence of a disease) may depend not only on the other variables but on the cluster context. Moreover, sharing of biomedical data is often hindered by patient confidentiality laws, and there is current interest in algorithms to generate synthetic tabular data from real data, for example via deep learning.  We demonstrate a novel EM-based clustering algorithm, MMM (``Madras Mixture Model''), that outperforms standard algorithms in determining clusters in synthetic heterogeneous data, and recovers structure in real data. Based on this, we
    
[^7]: DePAint：考虑峰值和平均限制的分散安全多智能体强化学习算法

    DePAint: A Decentralized Safe Multi-Agent Reinforcement Learning Algorithm considering Peak and Average Constraints. (arXiv:2310.14348v1 [cs.MA])

    [http://arxiv.org/abs/2310.14348](http://arxiv.org/abs/2310.14348)

    本文提出了一种分散的安全多智能体强化学习算法，能够在没有中央控制器的情况下，通过智能体之间的邻居通信来最大化累积奖励总和，并同时满足每个智能体的安全约束限制。

    

    安全的多智能体强化学习领域，尽管在无人机投递和车辆自动化等各个领域具有潜在应用，但仍然相对未被探索。在训练智能体学习最优策略以最大化奖励的同时考虑特定限制，特别是在训练过程中无法有中央控制器协调智能体的场景中，可能具有挑战性。本文针对分散环境下的多智能体策略优化问题进行了研究，其中智能体通过与邻居通信以最大化其累积奖励总和的同时满足每个智能体的安全约束。我们同时考虑了峰值和平均限制。在这种情况下，没有中央控制器协调智能体，奖励和约束只在每个智能体本地/私有可知。我们将问题形式化为分散约束的多智能体马尔可夫决策问题，并提出了一个算法。

    The field of safe multi-agent reinforcement learning, despite its potential applications in various domains such as drone delivery and vehicle automation, remains relatively unexplored. Training agents to learn optimal policies that maximize rewards while considering specific constraints can be challenging, particularly in scenarios where having a central controller to coordinate the agents during the training process is not feasible. In this paper, we address the problem of multi-agent policy optimization in a decentralized setting, where agents communicate with their neighbors to maximize the sum of their cumulative rewards while also satisfying each agent's safety constraints. We consider both peak and average constraints. In this scenario, there is no central controller coordinating the agents and both the rewards and constraints are only known to each agent locally/privately. We formulate the problem as a decentralized constrained multi-agent Markov Decision Problem and propose a 
    
[^8]: 在线估计与滚动验证：适应性非参数估计与数据流

    Online Estimation with Rolling Validation: Adaptive Nonparametric Estimation with Stream Data. (arXiv:2310.12140v1 [math.ST])

    [http://arxiv.org/abs/2310.12140](http://arxiv.org/abs/2310.12140)

    本研究提出了一种在线估计方法，通过加权滚动验证过程来提高基本估计器的自适应收敛速度，并证明了这种方法的重要性和敏感性

    

    由于其高效计算和竞争性的泛化能力，在线非参数估计器越来越受欢迎。一个重要的例子是随机梯度下降的变体。这些算法通常一次只取一个样本点，并立即更新感兴趣的参数估计。在这项工作中，我们考虑了这些在线算法的模型选择和超参数调整。我们提出了一种加权滚动验证过程，一种在线的留一交叉验证变体，对于许多典型的随机梯度下降估计器来说，额外的计算成本最小。类似于批量交叉验证，它可以提升基本估计器的自适应收敛速度。我们的理论分析很简单，主要依赖于一些一般的统计稳定性假设。模拟研究强调了滚动验证中发散权重在实践中的重要性，并证明了即使只有一个很小的偏差，它的敏感性也很高

    Online nonparametric estimators are gaining popularity due to their efficient computation and competitive generalization abilities. An important example includes variants of stochastic gradient descent. These algorithms often take one sample point at a time and instantly update the parameter estimate of interest. In this work we consider model selection and hyperparameter tuning for such online algorithms. We propose a weighted rolling-validation procedure, an online variant of leave-one-out cross-validation, that costs minimal extra computation for many typical stochastic gradient descent estimators. Similar to batch cross-validation, it can boost base estimators to achieve a better, adaptive convergence rate. Our theoretical analysis is straightforward, relying mainly on some general statistical stability assumptions. The simulation study underscores the significance of diverging weights in rolling validation in practice and demonstrates its sensitivity even when there is only a slim
    
[^9]: 整合鲁莽行为到基于协同过滤的推荐系统中

    Incorporating Recklessness to Collaborative Filtering based Recommender Systems. (arXiv:2308.02058v1 [cs.IR])

    [http://arxiv.org/abs/2308.02058](http://arxiv.org/abs/2308.02058)

    本文提出了一种将鲁莽行为引入基于矩阵分解的推荐系统学习过程的方法，通过控制风险水平来提高预测的数量和质量。

    

    包含可靠性测量的推荐系统往往在预测中更加保守，因为它们需要保持可靠性。这导致了这些系统可以提供的覆盖范围和新颖性的显著下降。在本文中，我们提出了在矩阵分解型推荐系统的学习过程中加入一项新的项，称为鲁莽行为，它可以控制在做出关于预测可靠性的决策时所希望的风险水平。实验结果表明，鲁莽行为不仅允许进行风险调控，还提高了推荐系统提供的预测的数量和质量。

    Recommender systems that include some reliability measure of their predictions tend to be more conservative in forecasting, due to their constraint to preserve reliability. This leads to a significant drop in the coverage and novelty that these systems can provide. In this paper, we propose the inclusion of a new term in the learning process of matrix factorization-based recommender systems, called recklessness, which enables the control of the risk level desired when making decisions about the reliability of a prediction. Experimental results demonstrate that recklessness not only allows for risk regulation but also improves the quantity and quality of predictions provided by the recommender system.
    
[^10]: 核去偏插值估计

    Kernel Debiased Plug-in Estimation. (arXiv:2306.08598v1 [stat.ME])

    [http://arxiv.org/abs/2306.08598](http://arxiv.org/abs/2306.08598)

    本文提出了一种高效、不需要实现影响函数且可计算的去偏插值估计方法。

    

    本文考虑在干扰参数存在的情况下估计标量目标参数的问题。采用非参数估计器（例如机器学习（ML）模型）替换未知干扰参数是方便的，但因存在较大偏差而效率低下。为了避免偏差-方差权衡的次优选择，现代方法会进行插值预估的去偏差操作，如有目标最小损失估计（TMLE）和双机器学习（DML）等。现有的去偏方法需要将目标参数的影响函数（IF）作为输入，然而，IF的推导需要专业知识，从而阻碍了这些方法的适应性。我们提出了一种新的去偏插入估计器的方法，它（i）高效、（ii）不需要实现IF、（iii）可计算。

    We consider the problem of estimating a scalar target parameter in the presence of nuisance parameters. Replacing the unknown nuisance parameter with a nonparametric estimator, e.g.,a machine learning (ML) model, is convenient but has shown to be inefficient due to large biases. Modern methods, such as the targeted minimum loss-based estimation (TMLE) and double machine learning (DML), achieve optimal performance under flexible assumptions by harnessing ML estimates while mitigating the plug-in bias. To avoid a sub-optimal bias-variance trade-off, these methods perform a debiasing step of the plug-in pre-estimate. Existing debiasing methods require the influence function of the target parameter as input. However, deriving the IF requires specialized expertise and thus obstructs the adaptation of these methods by practitioners. We propose a novel way to debias plug-in estimators which (i) is efficient, (ii) does not require the IF to be implemented, (iii) is computationally tractable, a
    
[^11]: 用TiDE进行长期预测：时间序列稠密编码器

    Long-term Forecasting with TiDE: Time-series Dense Encoder. (arXiv:2304.08424v1 [stat.ML])

    [http://arxiv.org/abs/2304.08424](http://arxiv.org/abs/2304.08424)

    TiDE是一种基于MLP的编码器-解码器模型，用于长期时间序列预测。它既具备线性模型的简单性和速度，又能处理协变量和非线性依赖，相较于最佳的Transformer模型，速度快5-10倍。

    

    最近的研究表明，相比于基于Transformer的方法，简单的线性模型在长期时间序列预测中表现更好。鉴于此，我们提出了一种基于多层感知机(MLP)的编码器-解码器模型，即时间序列稠密编码器(TiDE)，用于长期时间序列预测。它既享有线性模型的简单性和速度，又能处理协变量和非线性依赖。从理论上讲，我们证明了我们模型的最简线性类比在一些假设下可以达到线性动态系统(LDS)的近乎最优误差率。实证上，我们表明，我们的方法可以在流行的长期时间序列预测基准测试中匹配或胜过以前的方法，同时比最佳的基于Transformer的模型快5-10倍。

    Recent work has shown that simple linear models can outperform several Transformer based approaches in long term time-series forecasting. Motivated by this, we propose a Multi-layer Perceptron (MLP) based encoder-decoder model, Time-series Dense Encoder (TiDE), for long-term time-series forecasting that enjoys the simplicity and speed of linear models while also being able to handle covariates and non-linear dependencies. Theoretically, we prove that the simplest linear analogue of our model can achieve near optimal error rate for linear dynamical systems (LDS) under some assumptions. Empirically, we show that our method can match or outperform prior approaches on popular long-term time-series forecasting benchmarks while being 5-10x faster than the best Transformer based model.
    
[^12]: 利用分解注意力机制的单层Transformer对广义Potts模型进行最优推断

    Optimal inference of a generalised Potts model by single-layer transformers with factored attention. (arXiv:2304.07235v1 [cond-mat.dis-nn])

    [http://arxiv.org/abs/2304.07235](http://arxiv.org/abs/2304.07235)

    我们将分析和数值推导结合，在基于广义 Potts 模型的数据上，对经过改进适应这种模型的self-attention机制进行训练，发现经过修改的self-attention机制可以在极限采样下准确学习Potts模型。这个“分解”注意力机制通过从数据中学习相关属性，可以提高Transformer的性能和可解释性。

    

    Transformer 是一种革命性的神经网络，在自然语言处理和蛋白质科学方面取得了实践上的成功。它们的关键构建块是一个叫做自注意力机制的机制，它被训练用于预测句子中缺失的词。尽管Transformer在应用中取得了实践上的成功，但是自注意力机制究竟从数据中学到了什么以及它是怎么做到的还不是很清楚。本文针对从具有相互作用的位置和 Potts 颜色中提取的数据在训练的Transformer上给出了精确的分析和数值刻画。我们证明，虽然一般的transformer需要多层学习才能准确学习这个分布，但是经过小改进的自注意力机制在无限采样的极限下可以完美地学习Potts模型。我们还计算了这个修改后的自注意力机制所谓“分解”的泛化误差，并在合成数据上数值演示了我们的发现。我们的结果为解释Transformer的内在工作原理以及提高其性能和可解释性提供了新的思路。

    Transformers are the type of neural networks that has revolutionised natural language processing and protein science. Their key building block is a mechanism called self-attention which is trained to predict missing words in sentences. Despite the practical success of transformers in applications it remains unclear what self-attention learns from data, and how. Here, we give a precise analytical and numerical characterisation of transformers trained on data drawn from a generalised Potts model with interactions between sites and Potts colours. While an off-the-shelf transformer requires several layers to learn this distribution, we show analytically that a single layer of self-attention with a small modification can learn the Potts model exactly in the limit of infinite sampling. We show that this modified self-attention, that we call ``factored'', has the same functional form as the conditional probability of a Potts spin given the other spins, compute its generalisation error using t
    
[^13]: 具有平滑协方差的在线最小二乘SGD算法的高维缩放极限与波动

    High-dimensional scaling limits and fluctuations of online least-squares SGD with smooth covariance. (arXiv:2304.00707v1 [math.PR])

    [http://arxiv.org/abs/2304.00707](http://arxiv.org/abs/2304.00707)

    本文针对在线最小二乘随机梯度下降（SGD）算法，以数据生成模型的性质为基础，建立了高维缩放极限与波动模型。在低、中、高噪声设置阶段，揭示了迭代的精确三阶段相变。

    

    本文考虑数据生成模型的性质，导出了在线最小二乘随机梯度下降（SGD）算法的高维缩放极限与波动。我们将SGD迭代视为相互作用的粒子系统，其中期望相互作用由输入的协方差结构表征。我们假设对所有达到八阶矩的矩顺滑，且无需显式假设高斯性，就能够建立无穷维的普通微分方程（ODEs）或随机微分方程（SDEs）的高维缩放极限和波动。我们的结果揭示了迭代的精确三阶段相变；当噪声方差从低、中、高噪声设置时，它从轨道运动变为扩散运动，最终变为纯随机运动。在低噪声设置下，我们进一步表征了精确的波动。

    We derive high-dimensional scaling limits and fluctuations for the online least-squares Stochastic Gradient Descent (SGD) algorithm by taking the properties of the data generating model explicitly into consideration. Our approach treats the SGD iterates as an interacting particle system, where the expected interaction is characterized by the covariance structure of the input. Assuming smoothness conditions on moments of order up to eight orders, and without explicitly assuming Gaussianity, we establish the high-dimensional scaling limits and fluctuations in the form of infinite-dimensional Ordinary Differential Equations (ODEs) or Stochastic Differential Equations (SDEs). Our results reveal a precise three-step phase transition of the iterates; it goes from being ballistic, to diffusive, and finally to purely random behavior, as the noise variance goes from low, to moderate and finally to very-high noise setting. In the low-noise setting, we further characterize the precise fluctuation
    
[^14]: 学习和决策的风险自适应方法：一项调查

    Risk-Adaptive Approaches to Learning and Decision Making: A Survey. (arXiv:2212.00856v2 [math.OC] UPDATED)

    [http://arxiv.org/abs/2212.00856](http://arxiv.org/abs/2212.00856)

    本文调查了过去25年中风险测度的快速发展，介绍了其在各个领域的应用，以及与效用理论和分布鲁棒优化的关系，并指出了公平机器学习等新兴应用领域。

    

    不确定性在工程设计、统计学习和决策制定中普遍存在。由于固有的风险规避和对假设的模糊性，通常通过制定和解决使用风险和相关概念的保守优化模型来解决不确定性问题。我们对过去25年来风险测度的快速发展进行了调查。从它们在金融工程领域的起步，我们回顾了它们在几乎所有领域的工程和应用数学中的应用。风险测度扎根于凸分析，为处理不确定性提供了一个具有重要计算和理论优势的通用框架。我们描述了关键事实，列举了几种具体算法，并提供了大量参考文献供进一步阅读。该调查还回顾了与效用理论和分布鲁棒优化的联系，指出了新兴应用领域，如公平机器学习，并定义了相对测度。

    Uncertainty is prevalent in engineering design, statistical learning, and decision making broadly. Due to inherent risk-averseness and ambiguity about assumptions, it is common to address uncertainty by formulating and solving conservative optimization models expressed using measures of risk and related concepts. We survey the rapid development of risk measures over the last quarter century. From their beginning in financial engineering, we recount the spread to nearly all areas of engineering and applied mathematics. Solidly rooted in convex analysis, risk measures furnish a general framework for handling uncertainty with significant computational and theoretical advantages. We describe the key facts, list several concrete algorithms, and provide an extensive list of references for further reading. The survey recalls connections with utility theory and distributionally robust optimization, points to emerging applications areas such as fair machine learning, and defines measures of rel
    
[^15]: 面对适应性泛化：贝叶斯视角下的研究

    Generalization in the Face of Adaptivity: A Bayesian Perspective. (arXiv:2106.10761v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2106.10761](http://arxiv.org/abs/2106.10761)

    本文提出面对自适应选择数据样本引起的过度拟合问题，使用噪声加算法可以提供不依赖于查询规模的误差保证，这一结果表明适应数据分析的问题在于新查询与过去查询的协方差。

    

    自适应选择样本可能导致过度拟合，简单的噪声加算法可以避免这一问题。本文证明了噪声加算法可以提供不依赖于查询规模的误差保证，这一结果来源于更好地理解自适应数据分析的核心问题。我们表明适应数据分析的问题在于新查询与过去查询的协方差。

    Repeated use of a data sample via adaptively chosen queries can rapidly lead to overfitting, wherein the empirical evaluation of queries on the sample significantly deviates from their mean with respect to the underlying data distribution. It turns out that simple noise addition algorithms suffice to prevent this issue, and differential privacy-based analysis of these algorithms shows that they can handle an asymptotically optimal number of queries. However, differential privacy's worst-case nature entails scaling such noise to the range of the queries even for highly-concentrated queries, or introducing more complex algorithms.  In this paper, we prove that straightforward noise-addition algorithms already provide variance-dependent guarantees that also extend to unbounded queries. This improvement stems from a novel characterization that illuminates the core problem of adaptive data analysis. We show that the harm of adaptivity results from the covariance between the new query and a 
    

