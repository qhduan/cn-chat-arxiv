# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Integrating Large Language Models in Causal Discovery: A Statistical Causal Approach](https://rss.arxiv.org/abs/2402.01454) | 本文提出了一种在因果发现中集成大型语言模型的方法，通过将统计因果提示与知识增强相结合，可以使统计因果发现结果接近真实情况并进一步改进结果。 |
| [^2] | [Goal-Oriented Bayesian Optimal Experimental Design for Nonlinear Models using Markov Chain Monte Carlo](https://arxiv.org/abs/2403.18072) | 提出了一种适用于非线性模型的预测目标导向最优实验设计方法，通过最大化QoIs的期望信息增益来确定实验设计。 |
| [^3] | [A Probabilistic Approach for Alignment with Human Comparisons](https://arxiv.org/abs/2403.10771) | 通过提出的两阶段“监督微调+人类比较”框架，本文研究了如何有效利用人类比较来改善AI模型的对齐，特别是在面对嘈杂数据和高维模型时。 |
| [^4] | [Deep Horseshoe Gaussian Processes](https://arxiv.org/abs/2403.01737) | 深马蹄高斯过程Deep-HGP是一种简单的先验，采用深高斯过程并允许数据驱动选择关键长度尺度参数，对于非参数回归表现出良好的性能，实现了对未知真实回归曲线的优化回复，具有自适应的收敛速率。 |
| [^5] | [Online Estimation with Rolling Validation: Adaptive Nonparametric Estimation with Stream Data.](http://arxiv.org/abs/2310.12140) | 本研究提出了一种在线估计方法，通过加权滚动验证过程来提高基本估计器的自适应收敛速度，并证明了这种方法的重要性和敏感性 |
| [^6] | [Energy-Guided Continuous Entropic Barycenter Estimation for General Costs.](http://arxiv.org/abs/2310.01105) | 本文提出了一种基于能量导向的方法用于近似计算任意OT成本函数的连续熵OT巴氏中心，该方法具有优越的性能，并且能与基于能量的模型（EBMs）学习过程无缝连接。 |
| [^7] | [Instance-Optimal Cluster Recovery in the Labeled Stochastic Block Model.](http://arxiv.org/abs/2306.12968) | 本论文提出了一种算法，名为实例自适应聚类（IAC），它能够在标记随机块模型（LSBM）中恢复隐藏的群集。IAC包括一次谱聚类和一个迭代的基于似然的簇分配改进，不需要任何模型参数，是高效的。 |
| [^8] | [Global universal approximation of functional input maps on weighted spaces.](http://arxiv.org/abs/2306.03303) | 本文提出了功能性输入神经网络，可以在带权重空间上完成全局函数逼近。这一方法适用于连续函数的推广，还可用于路径空间函数的逼近，同时也可以逼近线性函数签名。 |
| [^9] | [Holistic Robust Data-Driven Decisions.](http://arxiv.org/abs/2207.09560) | 这篇论文提出了一种全面稳健的数据驱动公式，能够同时保护三个过拟合的源头：有限样本数据的统计误差、数据点的有限精度测量引起的数据噪声，以及被破坏的部分数据。 |
| [^10] | [Wasserstein multivariate auto-regressive models for modeling distributional time series and its application in graph learning.](http://arxiv.org/abs/2207.05442) | 本文提出了一种新的自回归模型，用于分析多元分布时间序列。并且在Wasserstein空间中建模了随机对象，提供了该模型的解的存在性和一致估计器。此方法可以应用于年龄分布和自行车共享网络的观察数据。 |
| [^11] | [Hierarchical Correlation Clustering and Tree Preserving Embedding.](http://arxiv.org/abs/2002.07756) | 本文提出了一种分层相关聚类方法，可应用于正负配对不相似度，并研究了使用此方法进行无监督表征学习的方法。 |

# 详细

[^1]: 在因果发现中集成大型语言模型: 一种统计因果方法

    Integrating Large Language Models in Causal Discovery: A Statistical Causal Approach

    [https://rss.arxiv.org/abs/2402.01454](https://rss.arxiv.org/abs/2402.01454)

    本文提出了一种在因果发现中集成大型语言模型的方法，通过将统计因果提示与知识增强相结合，可以使统计因果发现结果接近真实情况并进一步改进结果。

    

    在实际的统计因果发现（SCD）中，将领域专家知识作为约束嵌入到算法中被广泛接受，因为这对于创建一致有意义的因果模型是重要的，尽管识别背景知识的挑战被认可。为了克服这些挑战，本文提出了一种新的因果推断方法，即通过将LLM的“统计因果提示（SCP）”与SCD方法和基于知识的因果推断（KBCI）相结合，对SCD进行先验知识增强。实验证明，GPT-4可以使LLM-KBCI的输出与带有LLM-KBCI的先验知识的SCD结果接近真实情况，如果GPT-4经历了SCP，那么SCD的结果还可以进一步改善。而且，即使LLM不含有数据集的信息，LLM仍然可以通过其背景知识来改进SCD。

    In practical statistical causal discovery (SCD), embedding domain expert knowledge as constraints into the algorithm is widely accepted as significant for creating consistent meaningful causal models, despite the recognized challenges in systematic acquisition of the background knowledge. To overcome these challenges, this paper proposes a novel methodology for causal inference, in which SCD methods and knowledge based causal inference (KBCI) with a large language model (LLM) are synthesized through "statistical causal prompting (SCP)" for LLMs and prior knowledge augmentation for SCD. Experiments have revealed that GPT-4 can cause the output of the LLM-KBCI and the SCD result with prior knowledge from LLM-KBCI to approach the ground truth, and that the SCD result can be further improved, if GPT-4 undergoes SCP. Furthermore, it has been clarified that an LLM can improve SCD with its background knowledge, even if the LLM does not contain information on the dataset. The proposed approach
    
[^2]: 非线性模型的目标导向贝叶斯最优实验设计与马尔可夫链蒙特卡洛方法

    Goal-Oriented Bayesian Optimal Experimental Design for Nonlinear Models using Markov Chain Monte Carlo

    [https://arxiv.org/abs/2403.18072](https://arxiv.org/abs/2403.18072)

    提出了一种适用于非线性模型的预测目标导向最优实验设计方法，通过最大化QoIs的期望信息增益来确定实验设计。

    

    最优实验设计（OED）提供了一种系统化的方法来量化和最大化实验数据的价值。在贝叶斯方法下，传统的OED会最大化对模型参数的期望信息增益（EIG）。然而，我们通常感兴趣的不是参数本身，而是依赖于参数的非线性方式的预测感兴趣量（QoIs）。我们提出了一个适用于非线性观测和预测模型的预测目标导向OED（GO-OED）的计算框架，该框架寻求提供对QoIs的最大EIG的实验设计。具体地，我们提出了用于QoI EIG的嵌套蒙特卡洛估计器，其中采用马尔可夫链蒙特卡洛进行后验采样，利用核密度估计来评估后验预测密度及其与先验预测之间的Kullback-Leibler散度。GO-OED设计通过在设计空间中最大化EIG来获得。

    arXiv:2403.18072v1 Announce Type: cross  Abstract: Optimal experimental design (OED) provides a systematic approach to quantify and maximize the value of experimental data. Under a Bayesian approach, conventional OED maximizes the expected information gain (EIG) on model parameters. However, we are often interested in not the parameters themselves, but predictive quantities of interest (QoIs) that depend on the parameters in a nonlinear manner. We present a computational framework of predictive goal-oriented OED (GO-OED) suitable for nonlinear observation and prediction models, which seeks the experimental design providing the greatest EIG on the QoIs. In particular, we propose a nested Monte Carlo estimator for the QoI EIG, featuring Markov chain Monte Carlo for posterior sampling and kernel density estimation for evaluating the posterior-predictive density and its Kullback-Leibler divergence from the prior-predictive. The GO-OED design is then found by maximizing the EIG over the des
    
[^3]: 一种基于概率的人类比较对齐方法

    A Probabilistic Approach for Alignment with Human Comparisons

    [https://arxiv.org/abs/2403.10771](https://arxiv.org/abs/2403.10771)

    通过提出的两阶段“监督微调+人类比较”框架，本文研究了如何有效利用人类比较来改善AI模型的对齐，特别是在面对嘈杂数据和高维模型时。

    

    一个增长的趋势是将人类知识整合到学习框架中，利用微妙的人类反馈来完善AI模型。尽管取得了这些进展，但尚未开发出描述人类比较何时改善传统监督微调过程的特定条件的全面理论框架。为弥补这一差距，本文研究了有效利用人类比较来解决由嘈杂数据和高维模型引起的限制。我们提出了一个将机器学习与人类反馈通过概率二分方法联系起来的两阶段“监督微调+人类比较”（SFT+HC）框架。这两阶段框架首先通过SFT过程从带有噪声标记的数据中学习低维表示，然后利用人类比较来改进模型对齐。为了检验对齐阶段的效力，我们引入了一个新概念，称为“标签噪声到一致性”

    arXiv:2403.10771v1 Announce Type: new  Abstract: A growing trend involves integrating human knowledge into learning frameworks, leveraging subtle human feedback to refine AI models. Despite these advances, no comprehensive theoretical framework describing the specific conditions under which human comparisons improve the traditional supervised fine-tuning process has been developed. To bridge this gap, this paper studies the effective use of human comparisons to address limitations arising from noisy data and high-dimensional models. We propose a two-stage "Supervised Fine Tuning+Human Comparison" (SFT+HC) framework connecting machine learning with human feedback through a probabilistic bisection approach. The two-stage framework first learns low-dimensional representations from noisy-labeled data via an SFT procedure, and then uses human comparisons to improve the model alignment. To examine the efficacy of the alignment phase, we introduce a novel concept termed the "label-noise-to-co
    
[^4]: 深马蹄高斯过程

    Deep Horseshoe Gaussian Processes

    [https://arxiv.org/abs/2403.01737](https://arxiv.org/abs/2403.01737)

    深马蹄高斯过程Deep-HGP是一种简单的先验，采用深高斯过程并允许数据驱动选择关键长度尺度参数，对于非参数回归表现出良好的性能，实现了对未知真实回归曲线的优化回复，具有自适应的收敛速率。

    

    最近提出深高斯过程作为一种自然对象，类似于深度神经网络，可能拟合现代数据样本中存在的复杂特征，如组合结构。采用贝叶斯非参数方法，自然地利用深高斯过程作为先验分布，并将相应的后验分布用于统计推断。我们介绍了深马蹄高斯过程Deep-HGP，这是一种基于带有平方指数核的深高斯过程的新简单先验，特别是使得可以对关键长度尺度参数进行数据驱动选择。对于随机设计的非参数回归，我们展示了相应的调节后验分布以一种自适应方式，最优地在二次损失的意义下恢复未知的真回归曲线，最多只有一个对数因子。收敛速率同时对回归的平滑度和设计维度自适应。

    arXiv:2403.01737v1 Announce Type: cross  Abstract: Deep Gaussian processes have recently been proposed as natural objects to fit, similarly to deep neural networks, possibly complex features present in modern data samples, such as compositional structures. Adopting a Bayesian nonparametric approach, it is natural to use deep Gaussian processes as prior distributions, and use the corresponding posterior distributions for statistical inference. We introduce the deep Horseshoe Gaussian process Deep-HGP, a new simple prior based on deep Gaussian processes with a squared-exponential kernel, that in particular enables data-driven choices of the key lengthscale parameters. For nonparametric regression with random design, we show that the associated tempered posterior distribution recovers the unknown true regression curve optimally in terms of quadratic loss, up to a logarithmic factor, in an adaptive way. The convergence rates are simultaneously adaptive to both the smoothness of the regress
    
[^5]: 在线估计与滚动验证：适应性非参数估计与数据流

    Online Estimation with Rolling Validation: Adaptive Nonparametric Estimation with Stream Data. (arXiv:2310.12140v1 [math.ST])

    [http://arxiv.org/abs/2310.12140](http://arxiv.org/abs/2310.12140)

    本研究提出了一种在线估计方法，通过加权滚动验证过程来提高基本估计器的自适应收敛速度，并证明了这种方法的重要性和敏感性

    

    由于其高效计算和竞争性的泛化能力，在线非参数估计器越来越受欢迎。一个重要的例子是随机梯度下降的变体。这些算法通常一次只取一个样本点，并立即更新感兴趣的参数估计。在这项工作中，我们考虑了这些在线算法的模型选择和超参数调整。我们提出了一种加权滚动验证过程，一种在线的留一交叉验证变体，对于许多典型的随机梯度下降估计器来说，额外的计算成本最小。类似于批量交叉验证，它可以提升基本估计器的自适应收敛速度。我们的理论分析很简单，主要依赖于一些一般的统计稳定性假设。模拟研究强调了滚动验证中发散权重在实践中的重要性，并证明了即使只有一个很小的偏差，它的敏感性也很高

    Online nonparametric estimators are gaining popularity due to their efficient computation and competitive generalization abilities. An important example includes variants of stochastic gradient descent. These algorithms often take one sample point at a time and instantly update the parameter estimate of interest. In this work we consider model selection and hyperparameter tuning for such online algorithms. We propose a weighted rolling-validation procedure, an online variant of leave-one-out cross-validation, that costs minimal extra computation for many typical stochastic gradient descent estimators. Similar to batch cross-validation, it can boost base estimators to achieve a better, adaptive convergence rate. Our theoretical analysis is straightforward, relying mainly on some general statistical stability assumptions. The simulation study underscores the significance of diverging weights in rolling validation in practice and demonstrates its sensitivity even when there is only a slim
    
[^6]: 基于能量导向的连续熵巴氏中心估计方法及其在一般成本问题中的应用

    Energy-Guided Continuous Entropic Barycenter Estimation for General Costs. (arXiv:2310.01105v1 [cs.LG])

    [http://arxiv.org/abs/2310.01105](http://arxiv.org/abs/2310.01105)

    本文提出了一种基于能量导向的方法用于近似计算任意OT成本函数的连续熵OT巴氏中心，该方法具有优越的性能，并且能与基于能量的模型（EBMs）学习过程无缝连接。

    

    优化输运（OT）巴氏中心是一种在捕捉概率分布几何特性的同时对其进行平均的数学方法。本文提出了一种新颖的算法，用于近似计算任意OT成本函数的连续熵OT巴氏中心。我们的方法基于最近在机器学习社区中受到关注的基于弱OT的连续熵最优输运问题的对偶重构。除了创新性之外，我们的方法还具有以下若干优势特点：（i）我们建立了对恢复解的质量界限；（ii）该方法与基于能量的模型（EBMs）学习过程无缝连接，可以使用经过良好调整的算法解决感兴趣的问题；（iii）它提供了一种直观的优化方案，避免使用极小-极大、强化等复杂技巧。为了验证我们的方法，我们考虑了s

    Optimal transport (OT) barycenters are a mathematically grounded way of averaging probability distributions while capturing their geometric properties. In short, the barycenter task is to take the average of a collection of probability distributions w.r.t. given OT discrepancies. We propose a novel algorithm for approximating the continuous Entropic OT (EOT) barycenter for arbitrary OT cost functions. Our approach is built upon the dual reformulation of the EOT problem based on weak OT, which has recently gained the attention of the ML community. Beyond its novelty, our method enjoys several advantageous properties: (i) we establish quality bounds for the recovered solution; (ii) this approach seemlessly interconnects with the Energy-Based Models (EBMs) learning procedure enabling the use of well-tuned algorithms for the problem of interest; (iii) it provides an intuitive optimization scheme avoiding min-max, reinforce and other intricate technical tricks. For validation, we consider s
    
[^7]: 标记随机块模型中的最优簇恢复问题

    Instance-Optimal Cluster Recovery in the Labeled Stochastic Block Model. (arXiv:2306.12968v1 [cs.SI])

    [http://arxiv.org/abs/2306.12968](http://arxiv.org/abs/2306.12968)

    本论文提出了一种算法，名为实例自适应聚类（IAC），它能够在标记随机块模型（LSBM）中恢复隐藏的群集。IAC包括一次谱聚类和一个迭代的基于似然的簇分配改进，不需要任何模型参数，是高效的。

    

    本文考虑在有限数量的簇的情况下，用标记随机块模型（LSBM）恢复隐藏的社群，其中簇大小随着物品总数$n$的增长而线性增长。在LSBM中，为每对物品（独立地）观测到一个标签。我们的目标是设计一种有效的算法，利用观测到的标签来恢复簇。为此，我们重新审视了关于期望被任何聚类算法误分类的物品数量的实例特定下界。我们提出了实例自适应聚类（IAC），这是第一个在期望和高概率下都能匹配这些下界表现的算法。IAC由一次谱聚类算法和一个迭代的基于似然的簇分配改进组成。这种方法基于实例特定的下界，不需要任何模型参数，包括簇的数量。通过仅执行一次谱聚类，IAC在计算和存储方面都是高效的。

    We consider the problem of recovering hidden communities in the Labeled Stochastic Block Model (LSBM) with a finite number of clusters, where cluster sizes grow linearly with the total number $n$ of items. In the LSBM, a label is (independently) observed for each pair of items. Our objective is to devise an efficient algorithm that recovers clusters using the observed labels. To this end, we revisit instance-specific lower bounds on the expected number of misclassified items satisfied by any clustering algorithm. We present Instance-Adaptive Clustering (IAC), the first algorithm whose performance matches these lower bounds both in expectation and with high probability. IAC consists of a one-time spectral clustering algorithm followed by an iterative likelihood-based cluster assignment improvement. This approach is based on the instance-specific lower bound and does not require any model parameters, including the number of clusters. By performing the spectral clustering only once, IAC m
    
[^8]: 带权重空间上功能性输入映射的全局普适逼近

    Global universal approximation of functional input maps on weighted spaces. (arXiv:2306.03303v1 [stat.ML])

    [http://arxiv.org/abs/2306.03303](http://arxiv.org/abs/2306.03303)

    本文提出了功能性输入神经网络，可以在带权重空间上完成全局函数逼近。这一方法适用于连续函数的推广，还可用于路径空间函数的逼近，同时也可以逼近线性函数签名。

    

    我们引入了所谓的功能性输入神经网络，定义在可能是无限维带权重空间上，其值也在可能是无限维的输出空间中。为此，我们使用一个加性族作为隐藏层映射，以及一个非线性激活函数应用于每个隐藏层。依靠带权重空间上的Stone-Weierstrass定理，我们可以证明连续函数的推广的全局普适逼近结果，超越了常规紧集逼近。这特别适用于通过功能性输入神经网络逼近（非先见之明的）路径空间函数。作为带权Stone-Weierstrass定理的进一步应用，我们证明了线性函数签名的全局普适逼近结果。我们还在这个设置中引入了高斯过程回归的观点，并展示了签名内核的再生核希尔伯特空间是某些高斯过程的Cameron-Martin空间。

    We introduce so-called functional input neural networks defined on a possibly infinite dimensional weighted space with values also in a possibly infinite dimensional output space. To this end, we use an additive family as hidden layer maps and a non-linear activation function applied to each hidden layer. Relying on Stone-Weierstrass theorems on weighted spaces, we can prove a global universal approximation result for generalizations of continuous functions going beyond the usual approximation on compact sets. This then applies in particular to approximation of (non-anticipative) path space functionals via functional input neural networks. As a further application of the weighted Stone-Weierstrass theorem we prove a global universal approximation result for linear functions of the signature. We also introduce the viewpoint of Gaussian process regression in this setting and show that the reproducing kernel Hilbert space of the signature kernels are Cameron-Martin spaces of certain Gauss
    
[^9]: 全面稳健的数据驱动决策

    Holistic Robust Data-Driven Decisions. (arXiv:2207.09560v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2207.09560](http://arxiv.org/abs/2207.09560)

    这篇论文提出了一种全面稳健的数据驱动公式，能够同时保护三个过拟合的源头：有限样本数据的统计误差、数据点的有限精度测量引起的数据噪声，以及被破坏的部分数据。

    

    设计具有良好样本外性能的机器学习和决策的数据驱动公式是一个关键的挑战。好的样本内性能不一定能保证好的样本外性能，这被普遍认为是过拟合问题。实际的过拟合通常不能归因于单一原因，而是由多个因素同时引起的。我们在这里考虑了三个过拟合的源头：（一）统计误差，由于使用有限的样本数据而产生的误差，（二）数据噪声，当数据点只用有限精度测量时产生的噪声，（三）数据错误，即全部数据中有一小部分数据被完全破坏。我们认为，尽管现有的数据驱动公式在单独处理这三个源头时可能是稳健的，但它们不能同时提供对所有过拟合源头的全面保护。我们设计了一种新颖的数据驱动公式，可以保证这种全面保护。

    The design of data-driven formulations for machine learning and decision-making with good out-of-sample performance is a key challenge. The observation that good in-sample performance does not guarantee good out-of-sample performance is generally known as overfitting. Practical overfitting can typically not be attributed to a single cause but instead is caused by several factors all at once. We consider here three overfitting sources: (i) statistical error as a result of working with finite sample data, (ii) data noise which occurs when the data points are measured only with finite precision, and finally (iii) data misspecification in which a small fraction of all data may be wholly corrupted. We argue that although existing data-driven formulations may be robust against one of these three sources in isolation they do not provide holistic protection against all overfitting sources simultaneously. We design a novel data-driven formulation which does guarantee such holistic protection an
    
[^10]: Wasserstein多元自回归模型用于建模分布时间序列及其在图形学习中的应用

    Wasserstein multivariate auto-regressive models for modeling distributional time series and its application in graph learning. (arXiv:2207.05442v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2207.05442](http://arxiv.org/abs/2207.05442)

    本文提出了一种新的自回归模型，用于分析多元分布时间序列。并且在Wasserstein空间中建模了随机对象，提供了该模型的解的存在性和一致估计器。此方法可以应用于年龄分布和自行车共享网络的观察数据。

    

    我们提出了一种新的自回归模型，用于统计分析多元分布时间序列。感兴趣的数据包括一组在实线有界间隔上支持的概率测度的多个系列，并且被不同时间瞬间所索引。概率测度被建模为Wasserstein空间中的随机对象。我们通过在Lebesgue测度的切空间中建立自回归模型，首先对所有原始测度进行居中处理，以便它们的Fréchet平均值成为Lebesgue测度。利用迭代随机函数系统的理论，提供了这样一个模型的解的存在性、唯一性和平稳性的结果。我们还提出了模型系数的一致估计器。除了对模拟数据的分析，我们还使用两个实际数据集进行了模型演示：一个是不同国家年龄分布的观察数据集，另一个是巴黎自行车共享网络的观察数据集。

    We propose a new auto-regressive model for the statistical analysis of multivariate distributional time series. The data of interest consist of a collection of multiple series of probability measures supported over a bounded interval of the real line, and that are indexed by distinct time instants. The probability measures are modelled as random objects in the Wasserstein space. We establish the auto-regressive model in the tangent space at the Lebesgue measure by first centering all the raw measures so that their Fr\'echet means turn to be the Lebesgue measure. Using the theory of iterated random function systems, results on the existence, uniqueness and stationarity of the solution of such a model are provided. We also propose a consistent estimator for the model coefficient. In addition to the analysis of simulated data, the proposed model is illustrated with two real data sets made of observations from age distribution in different countries and bike sharing network in Paris. Final
    
[^11]: 分层相关聚类和维持树结构嵌入

    Hierarchical Correlation Clustering and Tree Preserving Embedding. (arXiv:2002.07756v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2002.07756](http://arxiv.org/abs/2002.07756)

    本文提出了一种分层相关聚类方法，可应用于正负配对不相似度，并研究了使用此方法进行无监督表征学习的方法。

    

    我们提出了一种分层相关聚类方法，扩展了著名的相关聚类方法，可以产生适用于正负配对不相似度的分层聚类。接下来，我们研究了使用这种分层相关聚类的无监督表征学习。为此，我们首先研究将相应的分层嵌入用于维持树结构嵌入和特征提取。然后，我们研究了最小最大距离度量扩展到相关聚类的方法，作为另一种表征学习范式。最后，我们在多个数据集上展示了我们方法的性能。

    We propose a hierarchical correlation clustering method that extends the well-known correlation clustering to produce hierarchical clusters applicable to both positive and negative pairwise dissimilarities. Then, in the following, we study unsupervised representation learning with such hierarchical correlation clustering. For this purpose, we first investigate embedding the respective hierarchy to be used for tree-preserving embedding and feature extraction. Thereafter, we study the extension of minimax distance measures to correlation clustering, as another representation learning paradigm. Finally, we demonstrate the performance of our methods on several datasets.
    

