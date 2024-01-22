# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Simulation Based Bayesian Optimization.](http://arxiv.org/abs/2401.10811) | 本文介绍了基于仿真的贝叶斯优化（SBBO）作为一种新方法，用于通过仅需基于采样的访问来优化获取函数。 |
| [^2] | [Robust Multi-Modal Density Estimation.](http://arxiv.org/abs/2401.10566) | 本文提出了一种名为ROME的鲁棒多模态密度估计方法，该方法利用聚类将多模态样本集分割成多个单模态样本集，并通过简单的KDE估计来估计整体分布。这种方法解决了多模态、非正态和高相关分布估计的挑战。 |
| [^3] | [LDReg: Local Dimensionality Regularized Self-Supervised Learning.](http://arxiv.org/abs/2401.10474) | 本文提出了一种叫做LDReg的本地维度正则化方法，用于解决自监督学习中的维度坍缩问题。通过增加局部内在维度，LDReg能够改善表示的性能。 |
| [^4] | [Cooperative Multi-Agent Graph Bandits: UCB Algorithm and Regret Analysis.](http://arxiv.org/abs/2401.10383) | 本文提出了一种解决多智能体图形赌博机问题的算法Multi-G-UCB，并通过数值实验验证了其有效性。 |
| [^5] | [Statistical Test for Attention Map in Vision Transformer.](http://arxiv.org/abs/2401.08169) | 本研究提出了一种Vision Transformer中注意力图的统计检验方法，可以将注意力作为可靠的定量证据指标用于决策，并通过p值进行统计显著性量化。 |
| [^6] | [Solution of the Probabilistic Lambert Problem: Connections with Optimal Mass Transport, Schr\"odinger Bridge and Reaction-Diffusion PDEs.](http://arxiv.org/abs/2401.07961) | 这项研究将概率Lambert问题与最优质量传输、Schr\"odinger桥和反应-扩散偏微分方程等领域连接起来，从而解决了概率Lambert问题的解的存在和唯一性，并提供了数值求解的方法。 |
| [^7] | [Let's do the time-warp-attend: Learning topological invariants of dynamical systems.](http://arxiv.org/abs/2312.09234) | 该论文提出了一个数据驱动、基于物理信息的深度学习框架，用于分类和表征动力学变化的拓扑不变特征提取，特别关注超临界霍普分歧。这个方法可以帮助预测系统的质变和常发行为变化。 |
| [^8] | [A Latent Variable Approach for Non-Hierarchical Multi-Fidelity Adaptive Sampling.](http://arxiv.org/abs/2310.03298) | 提出了一种基于潜变量的方法，用于非层次化多保真度自适应采样。该方法能够利用不同保真度模型之间的相关性以更高效地探索和利用设计空间。 |
| [^9] | [Unified Uncertainty Calibration.](http://arxiv.org/abs/2310.01202) | 该论文提出了一种统一的不确定性校准（U2C）框架，用于合并可知和认知不确定性，实现了面对困难样例时的准确预测和校准。 |
| [^10] | [Postprocessing of Ensemble Weather Forecasts Using Permutation-invariant Neural Networks.](http://arxiv.org/abs/2309.04452) | 本研究使用置换不变神经网络对集合天气预测进行后处理，不同于之前的方法，该网络将预测集合视为一组无序的成员预测，并学习对成员顺序的排列置换具有不变性的链接函数。在地表温度和风速预测的案例研究中，我们展示了最先进的预测质量。 |
| [^11] | [Learned harmonic mean estimation of the marginal likelihood with normalizing flows.](http://arxiv.org/abs/2307.00048) | 本文研究使用归一化流学习边缘似然的调和平均估计，在贝叶斯模型选择中解决了原始方法中的方差爆炸问题。 |
| [^12] | [TemperatureGAN: Generative Modeling of Regional Atmospheric Temperatures.](http://arxiv.org/abs/2306.17248) | TemperatureGAN是一个生成对抗网络，使用地面以上2m的大气温度数据，能够生成具有良好空间表示和与昼夜周期一致的时间动态的高保真样本。 |
| [^13] | [Interpreting Deep Neural Networks with the Package innsight.](http://arxiv.org/abs/2306.10822) | innsight是一个通用的R包，能够独立于深度学习库，解释来自任何R包的模型，并提供了丰富的可视化工具，以揭示深度神经网络预测的变量解释。 |
| [^14] | [$\alpha$-divergence Improves the Entropy Production Estimation via Machine Learning.](http://arxiv.org/abs/2303.02901) | 本研究通过机器学习提出了一种基于$\alpha$-散度的损失函数，在估计随机熵产生时表现出更加稳健的性能，尤其在强非平衡驱动或者缓慢动力学的情况下。选择$\alpha=-0.5$能获得最优结果。 |
| [^15] | [Are you using test log-likelihood correctly?.](http://arxiv.org/abs/2212.00219) | 使用测试对数似然进行比较可能与其他指标相矛盾，并且高测试对数似然不意味着更准确的后验近似。 |
| [^16] | [Hybrid Models for Mixed Variables in Bayesian Optimization.](http://arxiv.org/abs/2206.01409) | 本文提出了一种新型的混合模型，用于混合变量贝叶斯优化，并且在搜索和代理模型阶段都具有创新之处。数值实验证明了混合模型的优越性。 |
| [^17] | [Exploring Local Explanations of Nonlinear Models Using Animated Linear Projections.](http://arxiv.org/abs/2205.05359) | 本文介绍了一种使用动态线性投影方法来分析流行的非线性模型的局部解释的方法，探索预测变量之间的交互如何影响变量重要性估计，这对于理解模型的可解释性非常有用。 |

# 详细

[^1]: 基于仿真的贝叶斯优化

    Simulation Based Bayesian Optimization. (arXiv:2401.10811v1 [stat.ML])

    [http://arxiv.org/abs/2401.10811](http://arxiv.org/abs/2401.10811)

    本文介绍了基于仿真的贝叶斯优化（SBBO）作为一种新方法，用于通过仅需基于采样的访问来优化获取函数。

    

    贝叶斯优化是一种将先验知识与持续函数评估相结合的强大方法，用于优化黑盒函数。贝叶斯优化通过构建与协变量相关的目标函数的概率代理模型来指导未来评估点的选择。对于平滑连续的搜索空间，高斯过程经常被用作代理模型，因为它们提供对后验预测分布的解析访问，从而便于计算和优化获取函数。然而，在涉及对分类或混合协变量空间进行优化的复杂情况下，高斯过程可能不是理想的选择。本文介绍了一种名为基于仿真的贝叶斯优化（SBBO）的新方法，该方法仅需要对后验预测分布进行基于采样的访问，以优化获取函数。

    Bayesian Optimization (BO) is a powerful method for optimizing black-box functions by combining prior knowledge with ongoing function evaluations. BO constructs a probabilistic surrogate model of the objective function given the covariates, which is in turn used to inform the selection of future evaluation points through an acquisition function. For smooth continuous search spaces, Gaussian Processes (GPs) are commonly used as the surrogate model as they offer analytical access to posterior predictive distributions, thus facilitating the computation and optimization of acquisition functions. However, in complex scenarios involving optimizations over categorical or mixed covariate spaces, GPs may not be ideal.  This paper introduces Simulation Based Bayesian Optimization (SBBO) as a novel approach to optimizing acquisition functions that only requires \emph{sampling-based} access to posterior predictive distributions. SBBO allows the use of surrogate probabilistic models tailored for co
    
[^2]: 鲁棒的多模态密度估计

    Robust Multi-Modal Density Estimation. (arXiv:2401.10566v1 [cs.LG])

    [http://arxiv.org/abs/2401.10566](http://arxiv.org/abs/2401.10566)

    本文提出了一种名为ROME的鲁棒多模态密度估计方法，该方法利用聚类将多模态样本集分割成多个单模态样本集，并通过简单的KDE估计来估计整体分布。这种方法解决了多模态、非正态和高相关分布估计的挑战。

    

    多模态概率预测模型的发展引发了对综合评估指标的需求。虽然有几个指标可以表征机器学习模型的准确性（例如，负对数似然、Jensen-Shannon散度），但这些指标通常作用于概率密度上。因此，将它们应用于纯粹基于样本的预测模型需要估计底层密度函数。然而，常见的方法如核密度估计（KDE）已被证明在鲁棒性方面存在不足，而更复杂的方法在多模态估计问题中尚未得到评估。在本文中，我们提出了一种非参数的密度估计方法ROME（RObust Multi-modal density Estimator），它解决了估计多模态、非正态和高相关分布的挑战。ROME利用聚类将多模态样本集分割成多个单模态样本集，然后结合简单的KDE估计来得到总体的估计结果。

    Development of multi-modal, probabilistic prediction models has lead to a need for comprehensive evaluation metrics. While several metrics can characterize the accuracy of machine-learned models (e.g., negative log-likelihood, Jensen-Shannon divergence), these metrics typically operate on probability densities. Applying them to purely sample-based prediction models thus requires that the underlying density function is estimated. However, common methods such as kernel density estimation (KDE) have been demonstrated to lack robustness, while more complex methods have not been evaluated in multi-modal estimation problems. In this paper, we present ROME (RObust Multi-modal density Estimator), a non-parametric approach for density estimation which addresses the challenge of estimating multi-modal, non-normal, and highly correlated distributions. ROME utilizes clustering to segment a multi-modal set of samples into multiple uni-modal ones and then combines simple KDE estimates obtained for i
    
[^3]: LDReg: 本地维度正则化的自监督学习

    LDReg: Local Dimensionality Regularized Self-Supervised Learning. (arXiv:2401.10474v1 [cs.LG])

    [http://arxiv.org/abs/2401.10474](http://arxiv.org/abs/2401.10474)

    本文提出了一种叫做LDReg的本地维度正则化方法，用于解决自监督学习中的维度坍缩问题。通过增加局部内在维度，LDReg能够改善表示的性能。

    

    通过自监督学习（SSL）学习的表示可能容易出现维度坍缩，其中学习的表示子空间维度极低，因此无法表示完整的数据分布和模态。维度坍缩也被称为“填充不足”现象，是下游任务性能下降的主要原因之一。之前的工作在全局层面上研究了SSL的维度坍缩问题。在本文中，我们证明表示可以在全局上覆盖高维空间，但在局部上会坍缩。为了解决这个问题，我们提出了一种称为“本地维度正则化（LDReg）”的方法。我们的公式是基于Fisher-Rao度量的推导，用于比较和优化每个数据点在渐进小半径处的局部距离分布。通过增加局部内在维度，我们通过一系列实验证明LDReg可以改善表示。

    Representations learned via self-supervised learning (SSL) can be susceptible to dimensional collapse, where the learned representation subspace is of extremely low dimensionality and thus fails to represent the full data distribution and modalities. Dimensional collapse also known as the "underfilling" phenomenon is one of the major causes of degraded performance on downstream tasks. Previous work has investigated the dimensional collapse problem of SSL at a global level. In this paper, we demonstrate that representations can span over high dimensional space globally, but collapse locally. To address this, we propose a method called $\textit{local dimensionality regularization (LDReg)}$. Our formulation is based on the derivation of the Fisher-Rao metric to compare and optimize local distance distributions at an asymptotically small radius for each data point. By increasing the local intrinsic dimensionality, we demonstrate through a range of experiments that LDReg improves the repres
    
[^4]: 合作多智能体图形赌博机：UCB算法和遗憾分析

    Cooperative Multi-Agent Graph Bandits: UCB Algorithm and Regret Analysis. (arXiv:2401.10383v1 [cs.LG])

    [http://arxiv.org/abs/2401.10383](http://arxiv.org/abs/2401.10383)

    本文提出了一种解决多智能体图形赌博机问题的算法Multi-G-UCB，并通过数值实验验证了其有效性。

    

    本文将多智能体图形赌博机问题建模为Zhang、Johansson和Li在[CISS 57, 1-6 (2023)]中提出的图形赌博机问题的多智能体扩展。在我们的模型中，N个合作智能体在一个连通的图G上移动，图G有K个节点。抵达每个节点时，智能体观察到从一个与节点相关的概率分布中随机抽取的奖励。系统奖励被建模为智能体观测到的奖励的加权和，其中权重表达了多个智能体同时对同一节点进行采样的边际减少奖励。我们提出了一个基于上限置信区间（UCB）的学习算法，称为Multi-G-UCB，并证明了在T步内其期望遗憾被界定为$O(N\log(T)[\sqrt{KT} + DK])$，其中D是图G的直径。最后，我们通过与其他方法进行比较对算法进行了数值测试。

    In this paper, we formulate the multi-agent graph bandit problem as a multi-agent extension of the graph bandit problem introduced by Zhang, Johansson, and Li [CISS 57, 1-6 (2023)]. In our formulation, $N$ cooperative agents travel on a connected graph $G$ with $K$ nodes. Upon arrival at each node, agents observe a random reward drawn from a node-dependent probability distribution. The reward of the system is modeled as a weighted sum of the rewards the agents observe, where the weights capture the decreasing marginal reward associated with multiple agents sampling the same node at the same time. We propose an Upper Confidence Bound (UCB)-based learning algorithm, Multi-G-UCB, and prove that its expected regret over $T$ steps is bounded by $O(N\log(T)[\sqrt{KT} + DK])$, where $D$ is the diameter of graph $G$. Lastly, we numerically test our algorithm by comparing it to alternative methods.
    
[^5]: Vision Transformer中的注意力图统计检验

    Statistical Test for Attention Map in Vision Transformer. (arXiv:2401.08169v1 [stat.ML])

    [http://arxiv.org/abs/2401.08169](http://arxiv.org/abs/2401.08169)

    本研究提出了一种Vision Transformer中注意力图的统计检验方法，可以将注意力作为可靠的定量证据指标用于决策，并通过p值进行统计显著性量化。

    

    Vision Transformer（ViT）在各种计算机视觉任务中展示出了出色的性能。注意力对于ViT捕捉图像补丁之间复杂广泛的关系非常重要，使得模型可以权衡图像补丁的重要性，并帮助我们理解决策过程。然而，当将ViT的注意力用作高风险决策任务（如医学诊断）中的证据时，面临一个挑战，即注意机制可能错误地关注无关的区域。在本研究中，我们提出了一种ViT注意力的统计检验，使我们能够将注意力作为可靠的定量证据指标用于ViT的决策，并严格控制误差率。使用选择性推理框架，我们以p值的形式量化注意力的统计显著性，从而能够理论上基于假阳性检测概率量化注意力。

    The Vision Transformer (ViT) demonstrates exceptional performance in various computer vision tasks. Attention is crucial for ViT to capture complex wide-ranging relationships among image patches, allowing the model to weigh the importance of image patches and aiding our understanding of the decision-making process. However, when utilizing the attention of ViT as evidence in high-stakes decision-making tasks such as medical diagnostics, a challenge arises due to the potential of attention mechanisms erroneously focusing on irrelevant regions. In this study, we propose a statistical test for ViT's attentions, enabling us to use the attentions as reliable quantitative evidence indicators for ViT's decision-making with a rigorously controlled error rate. Using the framework called selective inference, we quantify the statistical significance of attentions in the form of p-values, which enables the theoretically grounded quantification of the false positive detection probability of attentio
    
[^6]: 概率Lambert问题的解决方案：与最优质量传输、Schr\"odinger桥和反应-扩散偏微分方程的连接

    Solution of the Probabilistic Lambert Problem: Connections with Optimal Mass Transport, Schr\"odinger Bridge and Reaction-Diffusion PDEs. (arXiv:2401.07961v1 [math.OC])

    [http://arxiv.org/abs/2401.07961](http://arxiv.org/abs/2401.07961)

    这项研究将概率Lambert问题与最优质量传输、Schr\"odinger桥和反应-扩散偏微分方程等领域连接起来，从而解决了概率Lambert问题的解的存在和唯一性，并提供了数值求解的方法。

    

    Lambert问题涉及通过速度控制在规定的飞行时间内将航天器从给定的初始位置转移到给定的终端位置，受到重力力场的限制。我们考虑了Lambert问题的概率变种，其中位置向量的端点约束的知识被它们各自的联合概率密度函数所替代。我们证明了具有端点联合概率密度约束的Lambert问题是一个广义的最优质量传输（OMT）问题，从而将这个经典的天体动力学问题与现代随机控制和随机机器学习的新兴研究领域联系起来。这个新发现的连接使我们能够严格建立概率Lambert问题的解的存在性和唯一性。同样的连接还帮助通过扩散正规化数值求解概率Lambert问题，即通过进一步的连接来利用。

    Lambert's problem concerns with transferring a spacecraft from a given initial to a given terminal position within prescribed flight time via velocity control subject to a gravitational force field. We consider a probabilistic variant of the Lambert problem where the knowledge of the endpoint constraints in position vectors are replaced by the knowledge of their respective joint probability density functions. We show that the Lambert problem with endpoint joint probability density constraints is a generalized optimal mass transport (OMT) problem, thereby connecting this classical astrodynamics problem with a burgeoning area of research in modern stochastic control and stochastic machine learning. This newfound connection allows us to rigorously establish the existence and uniqueness of solution for the probabilistic Lambert problem. The same connection also helps to numerically solve the probabilistic Lambert problem via diffusion regularization, i.e., by leveraging further connection 
    
[^7]: 做时间扭曲吧：学习动力系统的拓扑不变量

    Let's do the time-warp-attend: Learning topological invariants of dynamical systems. (arXiv:2312.09234v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2312.09234](http://arxiv.org/abs/2312.09234)

    该论文提出了一个数据驱动、基于物理信息的深度学习框架，用于分类和表征动力学变化的拓扑不变特征提取，特别关注超临界霍普分歧。这个方法可以帮助预测系统的质变和常发行为变化。

    

    科学领域中的动力系统，从电路到生态网络，当其基本参数跨越阈值时，会发生质变和常发性的行为变化，称为分歧。现有方法能够预测单个系统中即将发生的灾难，但主要基于时间序列，并且在分类不同系统的定性动力学变化和推广到真实数据方面存在困难。为了应对这一挑战，我们提出了一个数据驱动的、基于物理信息的深度学习框架，用于对动力学变化进行分类并表征分歧边界的拓扑不变特征提取。我们专注于超临界霍普分歧的典型案例，其用于模拟广泛应用的周期性动力学。我们的卷积关注方法经过了数据增强训练，鼓励学习可以用于检测分歧边界的拓扑不变量。

    Dynamical systems across the sciences, from electrical circuits to ecological networks, undergo qualitative and often catastrophic changes in behavior, called bifurcations, when their underlying parameters cross a threshold. Existing methods predict oncoming catastrophes in individual systems but are primarily time-series-based and struggle both to categorize qualitative dynamical regimes across diverse systems and to generalize to real data. To address this challenge, we propose a data-driven, physically-informed deep-learning framework for classifying dynamical regimes and characterizing bifurcation boundaries based on the extraction of topologically invariant features. We focus on the paradigmatic case of the supercritical Hopf bifurcation, which is used to model periodic dynamics across a wide range of applications. Our convolutional attention method is trained with data augmentations that encourage the learning of topological invariants which can be used to detect bifurcation boun
    
[^8]: 一种用于非层次化多保真度自适应采样的潜变量方法

    A Latent Variable Approach for Non-Hierarchical Multi-Fidelity Adaptive Sampling. (arXiv:2310.03298v1 [stat.ML])

    [http://arxiv.org/abs/2310.03298](http://arxiv.org/abs/2310.03298)

    提出了一种基于潜变量的方法，用于非层次化多保真度自适应采样。该方法能够利用不同保真度模型之间的相关性以更高效地探索和利用设计空间。

    

    多保真度（MF）方法在提高替代模型和设计优化方面越来越受欢迎，通过整合来自不同低保真度（LF）模型的数据。尽管大多数现有的MF方法假定了一个固定的数据集，但是动态分配资源在不同保真度模型之间可以实现更高的探索和利用设计空间的效率。然而，大多数现有的MF方法依赖于保真度级别的层次假设，或者无法捕捉多个保真度级别之间的相互关系并利用其来量化未来样本的价值和导航自适应采样。为了解决这个障碍，我们提出了一个基于不同保真度模型的潜变量嵌入和相关的先验-后验分析的框架，以显式地利用它们的相关性进行自适应采样。在这个框架中，每个填充采样迭代包括两个步骤：首先我们确定具有最大潜力影响的位置。

    Multi-fidelity (MF) methods are gaining popularity for enhancing surrogate modeling and design optimization by incorporating data from various low-fidelity (LF) models. While most existing MF methods assume a fixed dataset, adaptive sampling methods that dynamically allocate resources among fidelity models can achieve higher efficiency in the exploring and exploiting the design space. However, most existing MF methods rely on the hierarchical assumption of fidelity levels or fail to capture the intercorrelation between multiple fidelity levels and utilize it to quantify the value of the future samples and navigate the adaptive sampling. To address this hurdle, we propose a framework hinged on a latent embedding for different fidelity models and the associated pre-posterior analysis to explicitly utilize their correlation for adaptive sampling. In this framework, each infill sampling iteration includes two steps: We first identify the location of interest with the greatest potential imp
    
[^9]: 统一的不确定性校准

    Unified Uncertainty Calibration. (arXiv:2310.01202v1 [stat.ML])

    [http://arxiv.org/abs/2310.01202](http://arxiv.org/abs/2310.01202)

    该论文提出了一种统一的不确定性校准（U2C）框架，用于合并可知和认知不确定性，实现了面对困难样例时的准确预测和校准。

    

    为了构建健壮，公平和安全的人工智能系统，我们希望在面对困难或超出训练类别的测试样例时，分类器能够说“我不知道”。普遍的预测不确定性策略是简单的“拒绝或分类”规则：如果认知不确定性高，则放弃预测，否则进行分类。然而，这种方法不允许不同的不确定性来源相互通信，会产生未校准的预测，并且不能纠正不确定性估计中的错误。为了解决这三个问题，我们引入了统一的不确定性校准（U2C）的整体框架，用于合并可知和认知不确定性。U2C能够进行清晰的学习理论分析不确定性估计，并且在各种ImageNet基准测试中优于拒绝或分类方法。

    To build robust, fair, and safe AI systems, we would like our classifiers to say ``I don't know'' when facing test examples that are difficult or fall outside of the training classes.The ubiquitous strategy to predict under uncertainty is the simplistic \emph{reject-or-classify} rule: abstain from prediction if epistemic uncertainty is high, classify otherwise.Unfortunately, this recipe does not allow different sources of uncertainty to communicate with each other, produces miscalibrated predictions, and it does not allow to correct for misspecifications in our uncertainty estimates. To address these three issues, we introduce \emph{unified uncertainty calibration (U2C)}, a holistic framework to combine aleatoric and epistemic uncertainties. U2C enables a clean learning-theoretical analysis of uncertainty estimation, and outperforms reject-or-classify across a variety of ImageNet benchmarks.
    
[^10]: 使用置换不变神经网络对集合天气预测进行后处理

    Postprocessing of Ensemble Weather Forecasts Using Permutation-invariant Neural Networks. (arXiv:2309.04452v1 [stat.ML])

    [http://arxiv.org/abs/2309.04452](http://arxiv.org/abs/2309.04452)

    本研究使用置换不变神经网络对集合天气预测进行后处理，不同于之前的方法，该网络将预测集合视为一组无序的成员预测，并学习对成员顺序的排列置换具有不变性的链接函数。在地表温度和风速预测的案例研究中，我们展示了最先进的预测质量。

    

    统计后处理用于将原始数值天气预报的集合转化为可靠的概率预测分布。本研究中，我们考察了使用置换不变神经网络进行这一任务的方法。与以往的方法不同，通常基于集合概要统计信息并忽略集合分布的细节，我们提出的网络将预测集合视为一组无序的成员预测，并学习对成员顺序的排列置换具有不变性的链接函数。我们通过校准度和锐度评估所获得的预测分布的质量，并将模型与经典的基准方法和基于神经网络的方法进行比较。通过处理地表温度和风速预测的案例研究，我们展示了最先进的预测质量。为了加深对学习推理过程的理解，我们进一步提出了基于置换的重要性评估方法。

    Statistical postprocessing is used to translate ensembles of raw numerical weather forecasts into reliable probabilistic forecast distributions. In this study, we examine the use of permutation-invariant neural networks for this task. In contrast to previous approaches, which often operate on ensemble summary statistics and dismiss details of the ensemble distribution, we propose networks which treat forecast ensembles as a set of unordered member forecasts and learn link functions that are by design invariant to permutations of the member ordering. We evaluate the quality of the obtained forecast distributions in terms of calibration and sharpness, and compare the models against classical and neural network-based benchmark methods. In case studies addressing the postprocessing of surface temperature and wind gust forecasts, we demonstrate state-of-the-art prediction quality. To deepen the understanding of the learned inference process, we further propose a permutation-based importance
    
[^11]: 使用归一化流学习边缘似然的调和平均估计

    Learned harmonic mean estimation of the marginal likelihood with normalizing flows. (arXiv:2307.00048v1 [stat.ME])

    [http://arxiv.org/abs/2307.00048](http://arxiv.org/abs/2307.00048)

    本文研究使用归一化流学习边缘似然的调和平均估计，在贝叶斯模型选择中解决了原始方法中的方差爆炸问题。

    

    计算边缘似然（也称为贝叶斯模型证据）是贝叶斯模型选择中的一项重要任务，它提供了一种有原则的定量比较模型的方法。学习的调和平均估计器解决了原始调和平均估计边缘似然的方差爆炸问题。学习的调和平均估计器学习了一个重要性采样目标分布，该分布近似于最优分布。虽然近似不必非常准确，但确保学习分布的概率质量包含在后验分布中是至关重要的，以避免方差爆炸问题。在先前的工作中，为了确保满足这个性质，在训练模型时引入了一种专门的优化问题。在本文中，我们引入了使用归一化流来表示重要性采样目标分布。基于流的模型通过最大似然从后验样本中进行训练。

    Computing the marginal likelihood (also called the Bayesian model evidence) is an important task in Bayesian model selection, providing a principled quantitative way to compare models. The learned harmonic mean estimator solves the exploding variance problem of the original harmonic mean estimation of the marginal likelihood. The learned harmonic mean estimator learns an importance sampling target distribution that approximates the optimal distribution. While the approximation need not be highly accurate, it is critical that the probability mass of the learned distribution is contained within the posterior in order to avoid the exploding variance problem. In previous work a bespoke optimization problem is introduced when training models in order to ensure this property is satisfied. In the current article we introduce the use of normalizing flows to represent the importance sampling target distribution. A flow-based model is trained on samples from the posterior by maximum likelihood e
    
[^12]: TemperatureGAN: 区域大气温度的生成建模

    TemperatureGAN: Generative Modeling of Regional Atmospheric Temperatures. (arXiv:2306.17248v1 [cs.LG])

    [http://arxiv.org/abs/2306.17248](http://arxiv.org/abs/2306.17248)

    TemperatureGAN是一个生成对抗网络，使用地面以上2m的大气温度数据，能够生成具有良好空间表示和与昼夜周期一致的时间动态的高保真样本。

    

    随机生成器对于估计气候对各个领域的影响非常有用。在各个领域中进行气候风险的预测，例如能源系统，需要准确（与基准真实数据有统计相似性）、可靠（不产生错误样本）和高效的生成器。我们利用来自北美陆地数据同化系统的数据，引入了TemperatureGAN，这是一个以月份、位置和时间段为条件的生成对抗网络，以每小时分辨率生成地面以上2m的大气温度。我们提出了评估方法和指标来衡量生成样本的质量。我们证明TemperatureGAN能够生成具有良好空间表示和与已知昼夜周期一致的时间动态的高保真样本。

    Stochastic generators are useful for estimating climate impacts on various sectors. Projecting climate risk in various sectors, e.g. energy systems, requires generators that are accurate (statistical resemblance to ground-truth), reliable (do not produce erroneous examples), and efficient. Leveraging data from the North American Land Data Assimilation System, we introduce TemperatureGAN, a Generative Adversarial Network conditioned on months, locations, and time periods, to generate 2m above ground atmospheric temperatures at an hourly resolution. We propose evaluation methods and metrics to measure the quality of generated samples. We show that TemperatureGAN produces high-fidelity examples with good spatial representation and temporal dynamics consistent with known diurnal cycles.
    
[^13]: 利用innsight包解释深度神经网络

    Interpreting Deep Neural Networks with the Package innsight. (arXiv:2306.10822v1 [stat.ML])

    [http://arxiv.org/abs/2306.10822](http://arxiv.org/abs/2306.10822)

    innsight是一个通用的R包，能够独立于深度学习库，解释来自任何R包的模型，并提供了丰富的可视化工具，以揭示深度神经网络预测的变量解释。

    

    R包innsight提供了一个通用的工具箱，通过所谓的特征归因方法，揭示了深度神经网络预测的变量解释。除了统一的用户友好的框架外，该包在三个方面脱颖而出：首先，它通常是第一个实现神经网络特征归因方法的R包。其次，它独立于深度学习库，允许解释来自任何R包，包括keras、torch、neuralnet甚至用户定义模型的模型。尽管它很灵活，但innsight在内部从torch包的快速和高效的数组计算中受益，这建立在LibTorch（PyTorch的C++后端）上，而不需要Python依赖。最后，它提供了各种可视化工具，用于表格、信号、图像数据或这些数据的组合。此外，可以使用plotly包以交互方式呈现这些图。

    The R package innsight offers a general toolbox for revealing variable-wise interpretations of deep neural networks' predictions with so-called feature attribution methods. Aside from the unified and user-friendly framework, the package stands out in three ways: It is generally the first R package implementing feature attribution methods for neural networks. Secondly, it operates independently of the deep learning library allowing the interpretation of models from any R package, including keras, torch, neuralnet, and even custom models. Despite its flexibility, innsight benefits internally from the torch package's fast and efficient array calculations, which builds on LibTorch $-$ PyTorch's C++ backend $-$ without a Python dependency. Finally, it offers a variety of visualization tools for tabular, signal, image data or a combination of these. Additionally, the plots can be rendered interactively using the plotly package.
    
[^14]: $\alpha$-散度通过机器学习改进了熵产生估计

    $\alpha$-divergence Improves the Entropy Production Estimation via Machine Learning. (arXiv:2303.02901v2 [cond-mat.stat-mech] UPDATED)

    [http://arxiv.org/abs/2303.02901](http://arxiv.org/abs/2303.02901)

    本研究通过机器学习提出了一种基于$\alpha$-散度的损失函数，在估计随机熵产生时表现出更加稳健的性能，尤其在强非平衡驱动或者缓慢动力学的情况下。选择$\alpha=-0.5$能获得最优结果。

    

    最近几年，通过机器学习从轨迹数据估计随机熵产生（EP）的算法引起了极大兴趣。这类算法的关键是找到一个损失函数，使其最小化能够保证准确的EP估计。本研究展示了存在一类损失函数，即那些实现了$\alpha$-散度的变分表示的函数，可以用于EP估计。通过将$\alpha$固定为在-1到0之间的值，$\alpha$-NEEP（Entropy Production的神经估计器）在强非平衡驱动或者缓慢动力学的情况下表现出更为稳健的性能，而这些情况对基于Kullback-Leibler散度（$\alpha=0$）的现有方法产生不利影响。特别地，选择$\alpha=-0.5$往往能得到最优结果。为了证实我们的发现，我们还提供了一个可精确求解的EP估计问题简化模型，其损失函数为land

    Recent years have seen a surge of interest in the algorithmic estimation of stochastic entropy production (EP) from trajectory data via machine learning. A crucial element of such algorithms is the identification of a loss function whose minimization guarantees the accurate EP estimation. In this study, we show that there exists a host of loss functions, namely those implementing a variational representation of the $\alpha$-divergence, which can be used for the EP estimation. By fixing $\alpha$ to a value between $-1$ and $0$, the $\alpha$-NEEP (Neural Estimator for Entropy Production) exhibits a much more robust performance against strong nonequilibrium driving or slow dynamics, which adversely affects the existing method based on the Kullback-Leibler divergence ($\alpha = 0$). In particular, the choice of $\alpha = -0.5$ tends to yield the optimal results. To corroborate our findings, we present an exactly solvable simplification of the EP estimation problem, whose loss function land
    
[^15]: 你是否正确使用了测试对数似然？

    Are you using test log-likelihood correctly?. (arXiv:2212.00219v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2212.00219](http://arxiv.org/abs/2212.00219)

    使用测试对数似然进行比较可能与其他指标相矛盾，并且高测试对数似然不意味着更准确的后验近似。

    

    测试对数似然常被用来比较不同模型的同一数据，或者比较拟合同一概率模型的不同近似推断算法。我们通过简单的例子展示了如何基于测试对数似然的比较可能与其他目标相矛盾。具体来说，我们的例子表明：（i）达到更高测试对数似然的近似贝叶斯推断算法不必意味着能够产生更准确的后验近似，（ii）基于测试对数似然比较的预测准确性结论可能与基于均方根误差的结论不一致。

    Test log-likelihood is commonly used to compare different models of the same data or different approximate inference algorithms for fitting the same probabilistic model. We present simple examples demonstrating how comparisons based on test log-likelihood can contradict comparisons according to other objectives. Specifically, our examples show that (i) approximate Bayesian inference algorithms that attain higher test log-likelihoods need not also yield more accurate posterior approximations and (ii) conclusions about forecast accuracy based on test log-likelihood comparisons may not agree with conclusions based on root mean squared error.
    
[^16]: 混合变量贝叶斯优化中的混合模型

    Hybrid Models for Mixed Variables in Bayesian Optimization. (arXiv:2206.01409v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.01409](http://arxiv.org/abs/2206.01409)

    本文提出了一种新型的混合模型，用于混合变量贝叶斯优化，并且在搜索和代理模型阶段都具有创新之处。数值实验证明了混合模型的优越性。

    

    本文提出了一种新型的混合模型，用于处理混合变量贝叶斯优化中的定量（连续和整数）和定性（分类）类型。我们的混合模型将蒙特卡洛树搜索结构（MCTS）用于分类变量，并将高斯过程（GP）用于连续变量。在搜索阶段中，我们将频率派的上置信度树搜索（UCTS）和贝叶斯狄利克雷搜索策略进行对比，展示了树结构在贝叶斯优化中的融合。在代理模型阶段，我们的创新之处在于针对混合变量贝叶斯优化的在线核选择。我们的创新，包括动态核选择、独特的UCTS（hybridM）和贝叶斯更新策略（hybridD），将我们的混合模型定位为混合变量代理模型的进步。数值实验凸显了混合模型的优越性，凸显了它们的潜力。

    This paper presents a new type of hybrid models for Bayesian optimization (BO) adept at managing mixed variables, encompassing both quantitative (continuous and integer) and qualitative (categorical) types. Our proposed new hybrid models merge Monte Carlo Tree Search structure (MCTS) for categorical variables with Gaussian Processes (GP) for continuous ones. Addressing efficiency in searching phase, we juxtapose the original (frequentist) upper confidence bound tree search (UCTS) and the Bayesian Dirichlet search strategies, showcasing the tree architecture's integration into Bayesian optimization. Central to our innovation in surrogate modeling phase is online kernel selection for mixed-variable BO. Our innovations, including dynamic kernel selection, unique UCTS (hybridM) and Bayesian update strategies (hybridD), position our hybrid models as an advancement in mixed-variable surrogate models. Numerical experiments underscore the hybrid models' superiority, highlighting their potentia
    
[^17]: 使用动态线性投影方法探索非线性模型的局部解释

    Exploring Local Explanations of Nonlinear Models Using Animated Linear Projections. (arXiv:2205.05359v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2205.05359](http://arxiv.org/abs/2205.05359)

    本文介绍了一种使用动态线性投影方法来分析流行的非线性模型的局部解释的方法，探索预测变量之间的交互如何影响变量重要性估计，这对于理解模型的可解释性非常有用。

    

    机器学习模型的预测能力日益增强，但与参数统计模型相比，其复杂性和可解释性下降。这种折衷导致了可解释的人工智能（XAI）的出现，提供了诸如局部解释（LE）和局部变量归因（LVA）之类的方法，以揭示模型如何使用预测变量进行预测。然而，LVA通常不能有效处理预测变量之间的关联。为了理解预测变量之间的交互如何影响变量重要性估计，可以将LVA转换为线性投影，并使用径向游览。这对于学习模型如何犯错，或异常值的影响，或观测值的聚类也非常有用。本文使用各种流行的非线性模型（包括随机森林和神经网络）的示例来说明这种方法。

    The increased predictive power of machine learning models comes at the cost of increased complexity and loss of interpretability, particularly in comparison to parametric statistical models. This trade-off has led to the emergence of eXplainable AI (XAI) which provides methods, such as local explanations (LEs) and local variable attributions (LVAs), to shed light on how a model use predictors to arrive at a prediction. These provide a point estimate of the linear variable importance in the vicinity of a single observation. However, LVAs tend not to effectively handle association between predictors. To understand how the interaction between predictors affects the variable importance estimate, we can convert LVAs into linear projections and use the radial tour. This is also useful for learning how a model has made a mistake, or the effect of outliers, or the clustering of observations. The approach is illustrated with examples from categorical (penguin species, chocolate types) and quant
    

