# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PIP-Net: Pedestrian Intention Prediction in the Wild](https://arxiv.org/abs/2402.12810) | PIP-Net是一个新型框架，通过综合利用动态学数据和场景空间特征，采用循环和时间注意力机制解决方案，成功预测行人通过马路的意图，性能优于现有技术。 |
| [^2] | [Frequentist Guarantees of Distributed (Non)-Bayesian Inference](https://arxiv.org/abs/2311.08214) | 本文针对通过通信网络连接的代理之间的分布式(非)贝叶斯推断问题建立了频率特性，探讨了在适当假设下分布式贝叶斯推断在参数效率和不确定性量化方面的表现，以及通信图设计和大小对后验收缩率的影响。 |
| [^3] | [Provably Stable Feature Rankings with SHAP and LIME.](http://arxiv.org/abs/2401.15800) | 这项研究提出了一种通过利用多重假设检验的思想，来设计可靠地排名机器学习模型中最重要特征的特征归因方法，旨在解决SHAP和LIME等常用方法由于随机采样导致的高度不稳定性问题。实验证明了该方法的有效性和计算效率。 |
| [^4] | [Four Facets of Forecast Felicity: Calibration, Predictiveness, Randomness and Regret.](http://arxiv.org/abs/2401.14483) | 本文展示了校准和遗憾在评估预测中的概念等价性，将评估问题构建为一个预测者、一个赌徒和自然之间的博弈，并将预测的评估与结果的随机性联系起来。 |
| [^5] | [Deep Kalman Filters Can Filter.](http://arxiv.org/abs/2310.19603) | 本研究展示了一类连续时间的深度卡尔曼滤波器（DKFs），可以近似实现一类非马尔可夫和条件高斯信号过程的条件分布律，从而具有在数学金融领域中传统模型基础上的滤波问题的应用潜力。 |
| [^6] | [Model-free Posterior Sampling via Learning Rate Randomization.](http://arxiv.org/abs/2310.18186) | 本文介绍了一种随机化无模型算法RandQL，用于减小马尔科夫决策过程中的遗憾。RandQL通过学习率随机化实现乐观探索，并在实证研究中表现出色。 |
| [^7] | [Online Estimation with Rolling Validation: Adaptive Nonparametric Estimation with Stream Data.](http://arxiv.org/abs/2310.12140) | 本研究提出了一种在线估计方法，通过加权滚动验证过程来提高基本估计器的自适应收敛速度，并证明了这种方法的重要性和敏感性 |
| [^8] | [Statistical guarantees for stochastic Metropolis-Hastings.](http://arxiv.org/abs/2310.09335) | 该论文研究了针对随机Metropolis-Hastings算法的统计保证。通过引入简单的修正项，该方法可以避免计算成本上的损失，并通过分析非参数回归情景和深度神经网络回归的数值实例来证明了其在采样和可信区间方面的优势。 |
| [^9] | [On the quality of randomized approximations of Tukey's depth.](http://arxiv.org/abs/2309.05657) | 本文研究了Tukey深度的随机近似质量问题，证明了在维度较高且数据从对数凹集的均匀分布中抽样的情况下，随机算法可以正确近似最大深度和接近零的深度，而对于中间深度的点，任何好的近似都需要指数复杂度。 |
| [^10] | [The geometry of financial institutions -- Wasserstein clustering of financial data.](http://arxiv.org/abs/2305.03565) | 本文提出了一种新的算法，Wasserstein聚类，用于处理金融机构的复杂数据，有效地解决了缺失值和基于特定特征识别聚类所面临的挑战。该算法可用于监管者的监管工作，并在其领域取得了良好的效果。 |
| [^11] | [Learning from Similar Linear Representations: Adaptivity, Minimaxity, and Robustness.](http://arxiv.org/abs/2303.17765) | 本文提出了两种算法，适应相似性结构并对异常值任务具有稳健性，适用于表示多任务学习和迁移学习设置。 |

# 详细

[^1]: PIP-Net：城市中行人意图预测

    PIP-Net: Pedestrian Intention Prediction in the Wild

    [https://arxiv.org/abs/2402.12810](https://arxiv.org/abs/2402.12810)

    PIP-Net是一个新型框架，通过综合利用动态学数据和场景空间特征，采用循环和时间注意力机制解决方案，成功预测行人通过马路的意图，性能优于现有技术。

    

    精准的自动驾驶车辆（AVs）对行人意图的预测是当前该领域的一项研究挑战。在本文中，我们介绍了PIP-Net，这是一个新颖的框架，旨在预测AVs在现实世界城市场景中的行人过马路意图。我们提供了两种针对不同摄像头安装和设置设计的PIP-Net变种。利用来自行驶场景的动力学数据和空间特征，所提出的模型采用循环和时间注意力机制的解决方案，性能优于现有技术。为了增强道路用户的视觉表示及其与自车的相关性，我们引入了一个分类深度特征图，结合局部运动流特征，为场景动态提供丰富的洞察。此外，我们探讨了将摄像头的视野从一个扩展到围绕自车的三个摄像头的影响，以提升

    arXiv:2402.12810v1 Announce Type: cross  Abstract: Accurate pedestrian intention prediction (PIP) by Autonomous Vehicles (AVs) is one of the current research challenges in this field. In this article, we introduce PIP-Net, a novel framework designed to predict pedestrian crossing intentions by AVs in real-world urban scenarios. We offer two variants of PIP-Net designed for different camera mounts and setups. Leveraging both kinematic data and spatial features from the driving scene, the proposed model employs a recurrent and temporal attention-based solution, outperforming state-of-the-art performance. To enhance the visual representation of road users and their proximity to the ego vehicle, we introduce a categorical depth feature map, combined with a local motion flow feature, providing rich insights into the scene dynamics. Additionally, we explore the impact of expanding the camera's field of view, from one to three cameras surrounding the ego vehicle, leading to enhancement in the
    
[^2]: 分布式(非)贝叶斯推断的频率保证

    Frequentist Guarantees of Distributed (Non)-Bayesian Inference

    [https://arxiv.org/abs/2311.08214](https://arxiv.org/abs/2311.08214)

    本文针对通过通信网络连接的代理之间的分布式(非)贝叶斯推断问题建立了频率特性，探讨了在适当假设下分布式贝叶斯推断在参数效率和不确定性量化方面的表现，以及通信图设计和大小对后验收缩率的影响。

    

    受分析大型分散数据集的需求推动，分布式贝叶斯推断已成为跨多个领域（包括统计学、电气工程和经济学）的关键研究领域。本文针对通过通信网络连接的代理之间的分布式(非)贝叶斯推断问题建立了频率特性，如后验一致性、渐近正态性和后验收缩率。我们的结果表明，在通信图上的适当假设下，分布式贝叶斯推断保留了参数效率，同时在不确定性量化方面增强了鲁棒性。我们还通过研究设计和通信图的大小如何影响后验收缩率来探讨了统计效率和通信效率之间的权衡。此外，我们将我们的分析扩展到时变图，并将结果应用于指数f

    arXiv:2311.08214v2 Announce Type: replace-cross  Abstract: Motivated by the need to analyze large, decentralized datasets, distributed Bayesian inference has become a critical research area across multiple fields, including statistics, electrical engineering, and economics. This paper establishes Frequentist properties, such as posterior consistency, asymptotic normality, and posterior contraction rates, for the distributed (non-)Bayes Inference problem among agents connected via a communication network. Our results show that, under appropriate assumptions on the communication graph, distributed Bayesian inference retains parametric efficiency while enhancing robustness in uncertainty quantification. We also explore the trade-off between statistical efficiency and communication efficiency by examining how the design and size of the communication graph impact the posterior contraction rate. Furthermore, We extend our analysis to time-varying graphs and apply our results to exponential f
    
[^3]: 使用SHAP和LIME进行可证明稳定的特征排名

    Provably Stable Feature Rankings with SHAP and LIME. (arXiv:2401.15800v1 [stat.ML])

    [http://arxiv.org/abs/2401.15800](http://arxiv.org/abs/2401.15800)

    这项研究提出了一种通过利用多重假设检验的思想，来设计可靠地排名机器学习模型中最重要特征的特征归因方法，旨在解决SHAP和LIME等常用方法由于随机采样导致的高度不稳定性问题。实验证明了该方法的有效性和计算效率。

    

    特征归因是了解机器学习模型预测的普遍工具。然而，用于评分输入变量的常用方法，如SHAP和LIME，由于随机采样而具有高度不稳定性。借鉴多重假设检验的思想，我们设计了能够以高概率正确排名最重要特征的归因方法。我们的算法RankSHAP保证$K$个最高Shapley值具有超过$1-\alpha$的正确排序概率。实证结果证明了其有效性和令人印象深刻的计算效率。我们还在之前的工作基础上为LIME提供了类似的结果，确保以正确顺序选择最重要的特征。

    Feature attributions are ubiquitous tools for understanding the predictions of machine learning models. However, popular methods for scoring input variables such as SHAP and LIME suffer from high instability due to random sampling. Leveraging ideas from multiple hypothesis testing, we devise attribution methods that correctly rank the most important features with high probability. Our algorithm RankSHAP guarantees that the $K$ highest Shapley values have the proper ordering with probability exceeding $1-\alpha$. Empirical results demonstrate its validity and impressive computational efficiency. We also build on previous work to yield similar results for LIME, ensuring the most important features are selected in the right order.
    
[^4]: 预测的四个方面：校准、预测性、随机性和遗憾

    Four Facets of Forecast Felicity: Calibration, Predictiveness, Randomness and Regret. (arXiv:2401.14483v1 [cs.LG])

    [http://arxiv.org/abs/2401.14483](http://arxiv.org/abs/2401.14483)

    本文展示了校准和遗憾在评估预测中的概念等价性，将评估问题构建为一个预测者、一个赌徒和自然之间的博弈，并将预测的评估与结果的随机性联系起来。

    

    机器学习是关于预测的。然而，预测只有经过评估后才具有其有用性。机器学习传统上关注损失类型及其相应的遗憾。目前，机器学习社区重新对校准产生了兴趣。在这项工作中，我们展示了校准和遗憾在评估预测中的概念等价性。我们将评估问题构建为一个预测者、一个赌徒和自然之间的博弈。通过对赌徒和预测者施加直观的限制，校准和遗憾自然地成为了这个框架的一部分。此外，这个博弈将预测的评估与结果的随机性联系起来。相对于预测而言，结果的随机性等同于关于结果的好的预测。我们称这两个方面为校准和遗憾、预测性和随机性，即预测的四个方面。

    Machine learning is about forecasting. Forecasts, however, obtain their usefulness only through their evaluation. Machine learning has traditionally focused on types of losses and their corresponding regret. Currently, the machine learning community regained interest in calibration. In this work, we show the conceptual equivalence of calibration and regret in evaluating forecasts. We frame the evaluation problem as a game between a forecaster, a gambler and nature. Putting intuitive restrictions on gambler and forecaster, calibration and regret naturally fall out of the framework. In addition, this game links evaluation of forecasts to randomness of outcomes. Random outcomes with respect to forecasts are equivalent to good forecasts with respect to outcomes. We call those dual aspects, calibration and regret, predictiveness and randomness, the four facets of forecast felicity.
    
[^5]: 深度卡尔曼滤波器可以进行滤波

    Deep Kalman Filters Can Filter. (arXiv:2310.19603v1 [cs.LG])

    [http://arxiv.org/abs/2310.19603](http://arxiv.org/abs/2310.19603)

    本研究展示了一类连续时间的深度卡尔曼滤波器（DKFs），可以近似实现一类非马尔可夫和条件高斯信号过程的条件分布律，从而具有在数学金融领域中传统模型基础上的滤波问题的应用潜力。

    

    深度卡尔曼滤波器（DKFs）是一类神经网络模型，可以从序列数据中生成高斯概率测度。虽然DKFs受卡尔曼滤波器的启发，但它们缺乏与随机滤波问题的具体理论关联，从而限制了它们在传统模型基础上的滤波问题的应用，例如数学金融中的债券和期权定价模型校准。我们通过展示一类连续时间DKFs，可以近似实现一类非马尔可夫和条件高斯信号过程的条件分布律，从而解决了深度学习数学基础中的这个问题。我们的近似结果在路径的足够规则的紧致子集上一致成立，其中近似误差由在给定紧致路径集上均一地计算的最坏情况2-Wasserstein距离量化。

    Deep Kalman filters (DKFs) are a class of neural network models that generate Gaussian probability measures from sequential data. Though DKFs are inspired by the Kalman filter, they lack concrete theoretical ties to the stochastic filtering problem, thus limiting their applicability to areas where traditional model-based filters have been used, e.g.\ model calibration for bond and option prices in mathematical finance. We address this issue in the mathematical foundations of deep learning by exhibiting a class of continuous-time DKFs which can approximately implement the conditional law of a broad class of non-Markovian and conditionally Gaussian signal processes given noisy continuous-times measurements. Our approximation results hold uniformly over sufficiently regular compact subsets of paths, where the approximation error is quantified by the worst-case 2-Wasserstein distance computed uniformly over the given compact set of paths.
    
[^6]: 无模型后验采样的模型自由随机学习方法

    Model-free Posterior Sampling via Learning Rate Randomization. (arXiv:2310.18186v1 [stat.ML])

    [http://arxiv.org/abs/2310.18186](http://arxiv.org/abs/2310.18186)

    本文介绍了一种随机化无模型算法RandQL，用于减小马尔科夫决策过程中的遗憾。RandQL通过学习率随机化实现乐观探索，并在实证研究中表现出色。

    

    本文介绍了一种新颖的随机化无模型算法，Randomized Q-learning（简称RandQL），用于减小马尔科夫决策过程（MDPs）中的遗憾。据我们所知，RandQL是第一个可行的模型自由后验采样算法。我们分析了RandQL在表格和非表格度量空间设置下的性能。在表格MDPs中，RandQL实现了一个遗憾界的顺序为$\widetilde{\mathcal{O}}(\sqrt{H^{5}SAT})$，其中$H$是计划的时间长度，$S$是状态数，$A$是动作数，$T$是回合数。对于度量状态-动作空间，RandQL实现了一个遗憾界的顺序为$\widetilde{\mathcal{O}}(H^{5/2} T^{(d_z+1)/(d_z+2)})$，其中$d_z$表示缩放维度。需要注意的是，RandQL实现了乐观探索，而不使用奖励，而是依赖于学习率随机化的新思想。我们的实证研究表明，RandQL在基线探索上胜过现有方法。

    In this paper, we introduce Randomized Q-learning (RandQL), a novel randomized model-free algorithm for regret minimization in episodic Markov Decision Processes (MDPs). To the best of our knowledge, RandQL is the first tractable model-free posterior sampling-based algorithm. We analyze the performance of RandQL in both tabular and non-tabular metric space settings. In tabular MDPs, RandQL achieves a regret bound of order $\widetilde{\mathcal{O}}(\sqrt{H^{5}SAT})$, where $H$ is the planning horizon, $S$ is the number of states, $A$ is the number of actions, and $T$ is the number of episodes. For a metric state-action space, RandQL enjoys a regret bound of order $\widetilde{\mathcal{O}}(H^{5/2} T^{(d_z+1)/(d_z+2)})$, where $d_z$ denotes the zooming dimension. Notably, RandQL achieves optimistic exploration without using bonuses, relying instead on a novel idea of learning rate randomization. Our empirical study shows that RandQL outperforms existing approaches on baseline exploration en
    
[^7]: 在线估计与滚动验证：适应性非参数估计与数据流

    Online Estimation with Rolling Validation: Adaptive Nonparametric Estimation with Stream Data. (arXiv:2310.12140v1 [math.ST])

    [http://arxiv.org/abs/2310.12140](http://arxiv.org/abs/2310.12140)

    本研究提出了一种在线估计方法，通过加权滚动验证过程来提高基本估计器的自适应收敛速度，并证明了这种方法的重要性和敏感性

    

    由于其高效计算和竞争性的泛化能力，在线非参数估计器越来越受欢迎。一个重要的例子是随机梯度下降的变体。这些算法通常一次只取一个样本点，并立即更新感兴趣的参数估计。在这项工作中，我们考虑了这些在线算法的模型选择和超参数调整。我们提出了一种加权滚动验证过程，一种在线的留一交叉验证变体，对于许多典型的随机梯度下降估计器来说，额外的计算成本最小。类似于批量交叉验证，它可以提升基本估计器的自适应收敛速度。我们的理论分析很简单，主要依赖于一些一般的统计稳定性假设。模拟研究强调了滚动验证中发散权重在实践中的重要性，并证明了即使只有一个很小的偏差，它的敏感性也很高

    Online nonparametric estimators are gaining popularity due to their efficient computation and competitive generalization abilities. An important example includes variants of stochastic gradient descent. These algorithms often take one sample point at a time and instantly update the parameter estimate of interest. In this work we consider model selection and hyperparameter tuning for such online algorithms. We propose a weighted rolling-validation procedure, an online variant of leave-one-out cross-validation, that costs minimal extra computation for many typical stochastic gradient descent estimators. Similar to batch cross-validation, it can boost base estimators to achieve a better, adaptive convergence rate. Our theoretical analysis is straightforward, relying mainly on some general statistical stability assumptions. The simulation study underscores the significance of diverging weights in rolling validation in practice and demonstrates its sensitivity even when there is only a slim
    
[^8]: 针对随机Metropolis-Hastings算法的统计保证

    Statistical guarantees for stochastic Metropolis-Hastings. (arXiv:2310.09335v1 [stat.ML])

    [http://arxiv.org/abs/2310.09335](http://arxiv.org/abs/2310.09335)

    该论文研究了针对随机Metropolis-Hastings算法的统计保证。通过引入简单的修正项，该方法可以避免计算成本上的损失，并通过分析非参数回归情景和深度神经网络回归的数值实例来证明了其在采样和可信区间方面的优势。

    

    Metropolis-Hastings步骤被广泛应用于基于梯度的马尔可夫链蒙特卡洛方法中的不确定性量化中。通过对批次计算接受概率，随机Metropolis-Hastings步骤节省了计算成本，但降低了有效样本量。我们展示了通过简单的修正项可以避免这个障碍。我们研究了如果在非参数回归设置中应用改进的随机Metropolis-Hastings方法从Gibbs后验分布中采样，则链的结果稳态分布的统计属性。针对深度神经网络回归，我们证明了PAC-Bayes预言不等式，它提供了最优的收缩速率，并分析了结果可信区间的直径和高置信概率。通过在高维参数空间中的数值实例，我们说明了随机Metropolis-Hastings算法的可信区间和收缩速率确实表现出类似的行为。

    A Metropolis-Hastings step is widely used for gradient-based Markov chain Monte Carlo methods in uncertainty quantification. By calculating acceptance probabilities on batches, a stochastic Metropolis-Hastings step saves computational costs, but reduces the effective sample size. We show that this obstacle can be avoided by a simple correction term. We study statistical properties of the resulting stationary distribution of the chain if the corrected stochastic Metropolis-Hastings approach is applied to sample from a Gibbs posterior distribution in a nonparametric regression setting. Focusing on deep neural network regression, we prove a PAC-Bayes oracle inequality which yields optimal contraction rates and we analyze the diameter and show high coverage probability of the resulting credible sets. With a numerical example in a high-dimensional parameter space, we illustrate that credible sets and contraction rates of the stochastic Metropolis-Hastings algorithm indeed behave similar to 
    
[^9]: 关于Tukey深度的随机近似质量

    On the quality of randomized approximations of Tukey's depth. (arXiv:2309.05657v1 [stat.ML])

    [http://arxiv.org/abs/2309.05657](http://arxiv.org/abs/2309.05657)

    本文研究了Tukey深度的随机近似质量问题，证明了在维度较高且数据从对数凹集的均匀分布中抽样的情况下，随机算法可以正确近似最大深度和接近零的深度，而对于中间深度的点，任何好的近似都需要指数复杂度。

    

    Tukey深度（或半空间深度）是用于多元数据中心度量的广泛应用的指标。然而，在高维度下，Tukey深度的精确计算被认为是一个困难的问题。为了解决这个问题，人们提出了Tukey深度的随机近似方法。在本文中，我们探讨了这样的随机算法何时能够返回一个良好的Tukey深度近似。我们研究了数据从对数凹陷均匀分布中抽样的情况。我们证明了，如果要求算法在维度上以多项式时间运行，随机算法可以正确地近似最大深度1/2和接近零的深度。另一方面，对于任何中间深度的点，任何好的近似都需要指数复杂度。

    Tukey's depth (or halfspace depth) is a widely used measure of centrality for multivariate data. However, exact computation of Tukey's depth is known to be a hard problem in high dimensions. As a remedy, randomized approximations of Tukey's depth have been proposed. In this paper we explore when such randomized algorithms return a good approximation of Tukey's depth. We study the case when the data are sampled from a log-concave isotropic distribution. We prove that, if one requires that the algorithm runs in polynomial time in the dimension, the randomized algorithm correctly approximates the maximal depth $1/2$ and depths close to zero. On the other hand, for any point of intermediate depth, any good approximation requires exponential complexity.
    
[^10]: 金融机构的几何形态--金融数据的Wasserstein聚类

    The geometry of financial institutions -- Wasserstein clustering of financial data. (arXiv:2305.03565v1 [stat.ML])

    [http://arxiv.org/abs/2305.03565](http://arxiv.org/abs/2305.03565)

    本文提出了一种新的算法，Wasserstein聚类，用于处理金融机构的复杂数据，有效地解决了缺失值和基于特定特征识别聚类所面临的挑战。该算法可用于监管者的监管工作，并在其领域取得了良好的效果。

    

    不断增加的各种有趣对象的细节和大数据的可用性使得有必要开发将这些信息压缩成代表性和可理解的地图的方法。金融监管是一个展示这种需求的领域，因为监管机构需要从金融机构获取多样化的数据，有时是高度细粒度的，以监督和评估他们的活动。然而，处理和分析这样的数据可能是一项艰巨的任务，尤其是考虑到处理缺失值和基于特定特征识别聚类所面临的挑战。为了解决这些挑战，我们提出了一种适用于概率分布的Lloyd算法变体，并使用广义Wasserstein重心构建表示不同对象上的给定数据的度量空间，从而应对金融监管背景下监管者面临的具体挑战。我们相信这种方法在金融监管领域具有实用价值。

    The increasing availability of granular and big data on various objects of interest has made it necessary to develop methods for condensing this information into a representative and intelligible map. Financial regulation is a field that exemplifies this need, as regulators require diverse and often highly granular data from financial institutions to monitor and assess their activities. However, processing and analyzing such data can be a daunting task, especially given the challenges of dealing with missing values and identifying clusters based on specific features.  To address these challenges, we propose a variant of Lloyd's algorithm that applies to probability distributions and uses generalized Wasserstein barycenters to construct a metric space which represents given data on various objects in condensed form. By applying our method to the financial regulation context, we demonstrate its usefulness in dealing with the specific challenges faced by regulators in this domain. We beli
    
[^11]: 学习相似的线性表示：适应性、极小化、以及稳健性

    Learning from Similar Linear Representations: Adaptivity, Minimaxity, and Robustness. (arXiv:2303.17765v1 [stat.ML])

    [http://arxiv.org/abs/2303.17765](http://arxiv.org/abs/2303.17765)

    本文提出了两种算法，适应相似性结构并对异常值任务具有稳健性，适用于表示多任务学习和迁移学习设置。

    

    表示多任务学习和迁移学习在实践中取得了巨大的成功，然而对这些方法的理论理解仍然欠缺。本文旨在理解从具有相似但并非完全相同的线性表示的任务中学习，同时处理异常值任务。我们提出了两种算法，适应相似性结构并对异常值任务具有稳健性，适用于表示多任务学习和迁移学习设置，我们的算法在单任务或仅目标学习时表现优异。

    Representation multi-task learning (MTL) and transfer learning (TL) have achieved tremendous success in practice. However, the theoretical understanding of these methods is still lacking. Most existing theoretical works focus on cases where all tasks share the same representation, and claim that MTL and TL almost always improve performance. However, as the number of tasks grow, assuming all tasks share the same representation is unrealistic. Also, this does not always match empirical findings, which suggest that a shared representation may not necessarily improve single-task or target-only learning performance. In this paper, we aim to understand how to learn from tasks with \textit{similar but not exactly the same} linear representations, while dealing with outlier tasks. We propose two algorithms that are \textit{adaptive} to the similarity structure and \textit{robust} to outlier tasks under both MTL and TL settings. Our algorithms outperform single-task or target-only learning when
    

