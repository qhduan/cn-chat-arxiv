# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Sine Activated Low-Rank Matrices for Parameter Efficient Learning](https://arxiv.org/abs/2403.19243) | 整合正弦函数到低秩分解过程中，提高模型准确性的同时保持参数高效性。 |
| [^2] | [Efficient Algorithms for Regularized Nonnegative Scale-invariant Low-rank Approximation Models](https://arxiv.org/abs/2403.18517) | 通过研究称为均匀正则化尺度不变的更一般模型，揭示了低秩逼近模型中尺度不变性导致隐式正则化的效果，有助于更好理解正则化函数的作用并指导正则化超参数的选择。 |
| [^3] | [Improving Model's Interpretability and Reliability using Biomarkers](https://arxiv.org/abs/2402.12394) | 利用决策树解释基于生物标志物的诊断模型，帮助临床医生提高识别不准确预测的能力，从而增强医学诊断模型的可靠性。 |
| [^4] | [Retrieve, Merge, Predict: Augmenting Tables with Data Lakes](https://arxiv.org/abs/2402.06282) | 本文通过对数据湖中的数据发现进行深入分析，着重于表格增强，提出了准确检索连接候选人的重要性和简单合并方法的效率，以及现有解决方案的好处和局限性。 |
| [^5] | [A distribution-guided Mapper algorithm.](http://arxiv.org/abs/2401.12237) | 这项工作引入了一种名为D-Mapper的分布引导Mapper算法，使用概率模型和数据固有特征生成密度引导的覆盖，并提供增强的拓扑特征。 |
| [^6] | [Semi-Supervised Deep Sobolev Regression: Estimation, Variable Selection and Beyond.](http://arxiv.org/abs/2401.04535) | 我们提出了一种半监督深度Sobolev回归器，利用深度神经网络进行梯度范数正则化，可以同时估计回归函数和其梯度，即使存在显著领域变化。这在半监督学习中利用无标签数据方面具有可证优势。 |
| [^7] | [Efficient Methods for Non-stationary Online Learning.](http://arxiv.org/abs/2309.08911) | 这项工作提出了一种针对非平稳在线学习的高效方法，通过降低每轮投影的数量来优化动态遗憾和自适应遗憾的计算复杂性。 |
| [^8] | [Enhancing Hyperedge Prediction with Context-Aware Self-Supervised Learning.](http://arxiv.org/abs/2309.05798) | 该论文提出了一种增强超边预测的方法，通过上下文感知的节点聚合和自监督对比学习来解决超边预测中的问题。这种方法可以准确捕捉节点之间的复杂关系，并缓解数据稀疏问题。 |
| [^9] | [Computing the gradients with respect to all parameters of a quantum neural network using a single circuit.](http://arxiv.org/abs/2307.08167) | 该论文提出了一种使用单个电路计算量子神经网络所有参数梯度的方法，相比传统方法，它具有较低的电路深度和较少的编译时间，从而加速了总体运行时间。 |
| [^10] | [Virtual Human Generative Model: Masked Modeling Approach for Learning Human Characteristics.](http://arxiv.org/abs/2306.10656) | 本论文提出了一种名为VHGM的深度生成模型，基于掩码建模的方法来学习健康属性、生活方式和人格之间的关系。通过使用异构表格数据集，VHGM有效地学习了超过1,800个属性。该模型具有潜在的应用前景，例如用于医疗属性的虚拟测量和生活方式的假设验证。 |
| [^11] | [Optimal Decision Trees for Separable Objectives: Pushing the Limits of Dynamic Programming.](http://arxiv.org/abs/2305.19706) | 本研究提出了一种通用的动态规划方法来优化任何组合的可分离目标和约束条件，这种方法在可扩展性方面比通用求解器表现得更好。 |
| [^12] | [Generative Adversarial Reduced Order Modelling.](http://arxiv.org/abs/2305.15881) | 本文提出了一种基于GAN的简化建模方法GAROM，通过引入一个数据驱动的生成对抗模型，能够学习参数微分方程的解，并获得了较好的实验效果。 |
| [^13] | [MINN: Learning the dynamics of differential-algebraic equations and application to battery modeling.](http://arxiv.org/abs/2304.14422) | 本文提出了一种体系结构，生成模型集成神经网络（MINN）以允许在学习系统物理动态方面进行整合，应用于锂离子电池的电化学动力学建模，并展示了所提出的模型在解释性、精度和计算效率方面的优势。 |
| [^14] | [ClusterNet: A Perception-Based Clustering Model for Scattered Data.](http://arxiv.org/abs/2304.14185) | 这项工作介绍了ClusterNet，一种基于感知的分布式数据聚类模型，利用大规模数据集和基于点的深度学习模型，反映人类感知的聚类可分性。 |
| [^15] | [Adaptive Client Sampling in Federated Learning via Online Learning with Bandit Feedback.](http://arxiv.org/abs/2112.14332) | 本文提出一种基于赌博反馈的在线学习算法，用于自适应选择哪些客户端用于联邦学习的训练，并通过理论证明该算法可提高优化算法的收敛速度。 |
| [^16] | [Dynamic treatment effects: high-dimensional inference under model misspecification.](http://arxiv.org/abs/2111.06818) | 本文提出了一种新的鲁棒估计方法来解决动态治疗效应估计中的挑战，提高了在模型错误下的高维环境中的估计鲁棒性和可靠性。 |

# 详细

[^1]: 用正弦激活的低秩矩阵实现参数高效学习

    Sine Activated Low-Rank Matrices for Parameter Efficient Learning

    [https://arxiv.org/abs/2403.19243](https://arxiv.org/abs/2403.19243)

    整合正弦函数到低秩分解过程中，提高模型准确性的同时保持参数高效性。

    

    低秩分解已经成为在神经网络架构中增强参数效率的重要工具，在机器学习的各种应用中越来越受到关注。这些技术显著降低了参数数量，取得了简洁性和性能之间的平衡。然而，一个常见的挑战是在参数效率和模型准确性之间做出妥协，参数减少往往导致准确性不及完整秩对应模型。在这项工作中，我们提出了一个创新的理论框架，在低秩分解过程中整合了一个正弦函数。这种方法不仅保留了低秩方法的参数效率特性的好处，还增加了分解的秩，从而提高了模型的准确性。我们的方法被证明是现有低秩模型的一种适应性增强，正如其成功证实的那样。

    arXiv:2403.19243v1 Announce Type: new  Abstract: Low-rank decomposition has emerged as a vital tool for enhancing parameter efficiency in neural network architectures, gaining traction across diverse applications in machine learning. These techniques significantly lower the number of parameters, striking a balance between compactness and performance. However, a common challenge has been the compromise between parameter efficiency and the accuracy of the model, where reduced parameters often lead to diminished accuracy compared to their full-rank counterparts. In this work, we propose a novel theoretical framework that integrates a sinusoidal function within the low-rank decomposition process. This approach not only preserves the benefits of the parameter efficiency characteristic of low-rank methods but also increases the decomposition's rank, thereby enhancing model accuracy. Our method proves to be an adaptable enhancement for existing low-rank models, as evidenced by its successful 
    
[^2]: 针对正则化非负尺度不变低秩逼近模型的高效算法

    Efficient Algorithms for Regularized Nonnegative Scale-invariant Low-rank Approximation Models

    [https://arxiv.org/abs/2403.18517](https://arxiv.org/abs/2403.18517)

    通过研究称为均匀正则化尺度不变的更一般模型，揭示了低秩逼近模型中尺度不变性导致隐式正则化的效果，有助于更好理解正则化函数的作用并指导正则化超参数的选择。

    

    正则化非负低秩逼近，如稀疏的非负矩阵分解或稀疏的非负Tucker分解，是具有增强可解释性的降维模型中的一个重要分支。然而，从实践角度来看，由于这些模型的多因素特性以及缺乏支持这些选择的理论，正则化函数和正则化系数的选择，以及高效算法的设计仍然具有挑战性。本文旨在改进这些问题。通过研究一个称为均匀正则化尺度不变的更一般模型，我们证明低秩逼近模型中固有的尺度不变性导致了隐式正则化，具有意想不到的有益和有害效果。这一发现使我们能够更好地理解低秩逼近模型中正则化函数的作用，指导正则化超参数的选择。

    arXiv:2403.18517v1 Announce Type: new  Abstract: Regularized nonnegative low-rank approximations such as sparse Nonnegative Matrix Factorization or sparse Nonnegative Tucker Decomposition are an important branch of dimensionality reduction models with enhanced interpretability. However, from a practical perspective, the choice of regularizers and regularization coefficients, as well as the design of efficient algorithms, is challenging because of the multifactor nature of these models and the lack of theory to back these choices. This paper aims at improving upon these issues. By studying a more general model called the Homogeneous Regularized Scale-Invariant, we prove that the scale-invariance inherent to low-rank approximation models causes an implicit regularization with both unexpected beneficial and detrimental effects. This observation allows to better understand the effect of regularization functions in low-rank approximation models, to guide the choice of the regularization hyp
    
[^3]: 利用生物标志物提高模型的解释性和可靠性

    Improving Model's Interpretability and Reliability using Biomarkers

    [https://arxiv.org/abs/2402.12394](https://arxiv.org/abs/2402.12394)

    利用决策树解释基于生物标志物的诊断模型，帮助临床医生提高识别不准确预测的能力，从而增强医学诊断模型的可靠性。

    

    准确且具有解释性的诊断模型在医学这个安全关键领域至关重要。我们研究了我们提出的基于生物标志物的肺部超声诊断流程的可解释性，以增强临床医生的诊断能力。本研究的目标是评估决策树分类器利用生物标志物提供的解释是否能够改善用户识别模型不准确预测能力，与传统的显著性图相比。我们的研究发现表明，基于临床建立的生物标志物的决策树解释能够帮助临床医生检测到假阳性，从而提高医学诊断模型的可靠性。

    arXiv:2402.12394v1 Announce Type: cross  Abstract: Accurate and interpretable diagnostic models are crucial in the safety-critical field of medicine. We investigate the interpretability of our proposed biomarker-based lung ultrasound diagnostic pipeline to enhance clinicians' diagnostic capabilities. The objective of this study is to assess whether explanations from a decision tree classifier, utilizing biomarkers, can improve users' ability to identify inaccurate model predictions compared to conventional saliency maps. Our findings demonstrate that decision tree explanations, based on clinically established biomarkers, can assist clinicians in detecting false positives, thus improving the reliability of diagnostic models in medicine.
    
[^4]: 获取、合并、预测：通过数据湖增强表格

    Retrieve, Merge, Predict: Augmenting Tables with Data Lakes

    [https://arxiv.org/abs/2402.06282](https://arxiv.org/abs/2402.06282)

    本文通过对数据湖中的数据发现进行深入分析，着重于表格增强，提出了准确检索连接候选人的重要性和简单合并方法的效率，以及现有解决方案的好处和局限性。

    

    我们对数据湖中的数据发现进行了深入分析，重点是给定机器学习任务的表格增强。我们分析了三个主要步骤中使用的替代方法：检索可连接的表格、合并信息和预测结果表格。作为数据湖，本文使用了YADL（另一个数据湖）-我们开发的一种用于基准测试此数据发现任务的新型数据集-和Open Data US，一个被引用的真实数据湖。通过对这两个数据湖的系统性探索，我们的研究概述了准确检索连接候选人的重要性以及简单合并方法的效率。我们报告了现有解决方案的好处和局限性，旨在指导未来的研究。

    We present an in-depth analysis of data discovery in data lakes, focusing on table augmentation for given machine learning tasks. We analyze alternative methods used in the three main steps: retrieving joinable tables, merging information, and predicting with the resultant table. As data lakes, the paper uses YADL (Yet Another Data Lake) -- a novel dataset we developed as a tool for benchmarking this data discovery task -- and Open Data US, a well-referenced real data lake. Through systematic exploration on both lakes, our study outlines the importance of accurately retrieving join candidates and the efficiency of simple merging methods. We report new insights on the benefits of existing solutions and on their limitations, aiming at guiding future research in this space.
    
[^5]: 一种分布引导的Mapper算法

    A distribution-guided Mapper algorithm. (arXiv:2401.12237v1 [math.AT])

    [http://arxiv.org/abs/2401.12237](http://arxiv.org/abs/2401.12237)

    这项工作引入了一种名为D-Mapper的分布引导Mapper算法，使用概率模型和数据固有特征生成密度引导的覆盖，并提供增强的拓扑特征。

    

    动机：Mapper算法是拓扑数据分析中探索数据形状的重要工具。使用数据集作为输入，Mapper算法输出代表整个数据集拓扑特征的图形。这个图形通常被认为是数据的一个Reeb图的近似。经典的Mapper算法使用固定的区间长度和重叠比率，这可能无法揭示数据的微妙特征，尤其是当底层结构复杂时。结果：在这项工作中，我们引入了一种名为D-Mapper的分布引导Mapper算法，利用概率模型的属性和数据固有特征生成密度引导的覆盖，并提供增强的拓扑特征。我们提出的算法是一种基于概率模型的方法，可以作为非概率性方法的替代。此外，我们引入了一个度量来考虑重叠聚类的质量和扩展持续同调。

    Motivation: The Mapper algorithm is an essential tool to explore shape of data in topology data analysis. With a dataset as an input, the Mapper algorithm outputs a graph representing the topological features of the whole dataset. This graph is often regarded as an approximation of a reeb graph of data. The classic Mapper algorithm uses fixed interval lengths and overlapping ratios, which might fail to reveal subtle features of data, especially when the underlying structure is complex.  Results: In this work, we introduce a distribution guided Mapper algorithm named D-Mapper, that utilizes the property of the probability model and data intrinsic characteristics to generate density guided covers and provides enhanced topological features. Our proposed algorithm is a probabilistic model-based approach, which could serve as an alternative to non-prababilistic ones. Moreover, we introduce a metric accounting for both the quality of overlap clustering and extended persistence homology to me
    
[^6]: 半监督深度Sobolev回归: 估计、变量选择及其他

    Semi-Supervised Deep Sobolev Regression: Estimation, Variable Selection and Beyond. (arXiv:2401.04535v1 [stat.ML])

    [http://arxiv.org/abs/2401.04535](http://arxiv.org/abs/2401.04535)

    我们提出了一种半监督深度Sobolev回归器，利用深度神经网络进行梯度范数正则化，可以同时估计回归函数和其梯度，即使存在显著领域变化。这在半监督学习中利用无标签数据方面具有可证优势。

    

    我们提出了SDORE，一种半监督深度Sobolev回归器，用于非参数估计潜在的回归函数及其梯度。SDORE使用深度神经网络来最小化经验风险，并采用梯度范数正则化，允许对无标签数据计算梯度范数。我们对SDORE的收敛速度进行了全面分析，并建立了回归函数的最小化最优速率。重要的是，在存在显著领域变化的情况下，我们还推导出了关联的插值梯度估计器的收敛速度。这些理论结果为选择正则化参数和确定神经网络的大小提供了有价值的先验指导，并展示了在半监督学习中利用无标签数据的可证优势。据我们所知，SDORE是第一个同时估计回归函数及其梯度的可证神经网络方法，具有多样化的应用。

    We propose SDORE, a semi-supervised deep Sobolev regressor, for the nonparametric estimation of the underlying regression function and its gradient. SDORE employs deep neural networks to minimize empirical risk with gradient norm regularization, allowing computation of the gradient norm on unlabeled data. We conduct a comprehensive analysis of the convergence rates of SDORE and establish a minimax optimal rate for the regression function. Crucially, we also derive a convergence rate for the associated plug-in gradient estimator, even in the presence of significant domain shift. These theoretical findings offer valuable prior guidance for selecting regularization parameters and determining the size of the neural network, while showcasing the provable advantage of leveraging unlabeled data in semi-supervised learning. To the best of our knowledge, SDORE is the first provable neural network-based approach that simultaneously estimates the regression function and its gradient, with diverse
    
[^7]: 非平稳在线学习的高效方法

    Efficient Methods for Non-stationary Online Learning. (arXiv:2309.08911v1 [cs.LG])

    [http://arxiv.org/abs/2309.08911](http://arxiv.org/abs/2309.08911)

    这项工作提出了一种针对非平稳在线学习的高效方法，通过降低每轮投影的数量来优化动态遗憾和自适应遗憾的计算复杂性。

    

    非平稳在线学习近年来引起了广泛关注。特别是在非平稳环境中，动态遗憾和自适应遗憾被提出作为在线凸优化的两个原则性性能度量。为了优化它们，通常采用两层在线集成，由于非平稳性的固有不确定性，其中维护一组基学习器，并采用元算法在运行过程中跟踪最佳学习器。然而，这种两层结构引发了关于计算复杂性的担忧 -这些方法通常同时维护$\mathcal{O}(\log T)$个基学习器，对于一个$T$轮在线游戏，因此每轮执行多次投影到可行域上，当域很复杂时，这成为计算瓶颈。在本文中，我们提出了优化动态遗憾和自适应遗憾的高效方法，将每轮的投影次数从$\mathcal{O}(\log T)$降低到...

    Non-stationary online learning has drawn much attention in recent years. In particular, dynamic regret and adaptive regret are proposed as two principled performance measures for online convex optimization in non-stationary environments. To optimize them, a two-layer online ensemble is usually deployed due to the inherent uncertainty of the non-stationarity, in which a group of base-learners are maintained and a meta-algorithm is employed to track the best one on the fly. However, the two-layer structure raises the concern about the computational complexity -- those methods typically maintain $\mathcal{O}(\log T)$ base-learners simultaneously for a $T$-round online game and thus perform multiple projections onto the feasible domain per round, which becomes the computational bottleneck when the domain is complicated. In this paper, we present efficient methods for optimizing dynamic regret and adaptive regret, which reduce the number of projections per round from $\mathcal{O}(\log T)$ t
    
[^8]: 增强上下文感知自监督学习的超边预测

    Enhancing Hyperedge Prediction with Context-Aware Self-Supervised Learning. (arXiv:2309.05798v1 [cs.LG])

    [http://arxiv.org/abs/2309.05798](http://arxiv.org/abs/2309.05798)

    该论文提出了一种增强超边预测的方法，通过上下文感知的节点聚合和自监督对比学习来解决超边预测中的问题。这种方法可以准确捕捉节点之间的复杂关系，并缓解数据稀疏问题。

    

    超图可以自然地建模群组关系（例如，一组共同购买物品的用户），hyperedge预测是预测未来或未观察到的超边的任务，在许多实际应用中都非常重要。然而，目前的研究中很少探讨以下挑战：（C1）如何聚合每个超边候选中的节点以准确预测超边？（C2）如何缓解超边预测中固有的数据稀疏问题？为了同时解决这两个挑战，本文提出了一种新颖的超边预测框架CASH，它采用了（1）上下文感知节点聚合，精确捕捉每个超边中节点之间的复杂关系，用于解决挑战（C1），以及（2）自监督对比学习在超边预测上下文中增强超图表示，以应对挑战（C2）。此外，针对挑战（C2），我们提出了超边感知的数据增强方法。

    Hypergraphs can naturally model group-wise relations (e.g., a group of users who co-purchase an item) as hyperedges. Hyperedge prediction is to predict future or unobserved hyperedges, which is a fundamental task in many real-world applications (e.g., group recommendation). Despite the recent breakthrough of hyperedge prediction methods, the following challenges have been rarely studied: (C1) How to aggregate the nodes in each hyperedge candidate for accurate hyperedge prediction? and (C2) How to mitigate the inherent data sparsity problem in hyperedge prediction? To tackle both challenges together, in this paper, we propose a novel hyperedge prediction framework (CASH) that employs (1) context-aware node aggregation to precisely capture complex relations among nodes in each hyperedge for (C1) and (2) self-supervised contrastive learning in the context of hyperedge prediction to enhance hypergraph representations for (C2). Furthermore, as for (C2), we propose a hyperedge-aware augmenta
    
[^9]: 使用单个电路计算量子神经网络所有参数的梯度

    Computing the gradients with respect to all parameters of a quantum neural network using a single circuit. (arXiv:2307.08167v2 [quant-ph] UPDATED)

    [http://arxiv.org/abs/2307.08167](http://arxiv.org/abs/2307.08167)

    该论文提出了一种使用单个电路计算量子神经网络所有参数梯度的方法，相比传统方法，它具有较低的电路深度和较少的编译时间，从而加速了总体运行时间。

    

    在使用参数平移规则计算量子神经网络的梯度时，需要对网络的单个可调参数计算两次代价函数。当参数总数较高时，需要调整和运行多次用于计算的量子电路。在这里，我们提出了一种仅使用一个电路计算所有梯度的方法，它具有较低的电路深度和较少的经典寄存器。我们还在真实量子硬件和模拟器上进行了实验证明，我们的方法具有电路编译时间明显缩短的优势，从而加速了总体运行时间。

    When computing the gradients of a quantum neural network using the parameter-shift rule, the cost function needs to be calculated twice for the gradient with respect to a single adjustable parameter of the network. When the total number of parameters is high, the quantum circuit for the computation has to be adjusted and run for many times. Here we propose an approach to compute all the gradients using a single circuit only, with a much reduced circuit depth and less classical registers. We also demonstrate experimentally, on both real quantum hardware and simulator, that our approach has the advantages that the circuit takes a significantly shorter time to compile than the conventional approach, resulting in a speedup on the total runtime.
    
[^10]: 虚拟人类生成模型：基于掩码建模的方法来学习人类特征

    Virtual Human Generative Model: Masked Modeling Approach for Learning Human Characteristics. (arXiv:2306.10656v1 [cs.LG])

    [http://arxiv.org/abs/2306.10656](http://arxiv.org/abs/2306.10656)

    本论文提出了一种名为VHGM的深度生成模型，基于掩码建模的方法来学习健康属性、生活方式和人格之间的关系。通过使用异构表格数据集，VHGM有效地学习了超过1,800个属性。该模型具有潜在的应用前景，例如用于医疗属性的虚拟测量和生活方式的假设验证。

    

    识别医疗属性、生活方式和人格之间的关系对于理解和改善身体和精神状况至关重要。本文提出了一种名为虚拟人类生成模型（VHGM）的机器学习模型，用于估计有关医疗保健、生活方式和个性的属性。VHGM是一个深度生成模型，使用掩码建模训练，在已知属性的条件下学习属性的联合分布。利用异构表格数据集，VHGM高效地学习了超过1,800个属性。我们数值评估了VHGM及其训练技术的性能。作为VHGM的概念验证，我们提出了几个应用程序，演示了用户情境，例如医疗属性的虚拟测量和生活方式的假设验证。

    Identifying the relationship between healthcare attributes, lifestyles, and personality is vital for understanding and improving physical and mental conditions. Machine learning approaches are promising for modeling their relationships and offering actionable suggestions. In this paper, we propose Virtual Human Generative Model (VHGM), a machine learning model for estimating attributes about healthcare, lifestyles, and personalities. VHGM is a deep generative model trained with masked modeling to learn the joint distribution of attributes conditioned on known ones. Using heterogeneous tabular datasets, VHGM learns more than 1,800 attributes efficiently. We numerically evaluate the performance of VHGM and its training techniques. As a proof-of-concept of VHGM, we present several applications demonstrating user scenarios, such as virtual measurements of healthcare attributes and hypothesis verifications of lifestyles.
    
[^11]: 可分目标的最优决策树：推动动态规划的极限

    Optimal Decision Trees for Separable Objectives: Pushing the Limits of Dynamic Programming. (arXiv:2305.19706v1 [cs.LG])

    [http://arxiv.org/abs/2305.19706](http://arxiv.org/abs/2305.19706)

    本研究提出了一种通用的动态规划方法来优化任何组合的可分离目标和约束条件，这种方法在可扩展性方面比通用求解器表现得更好。

    

    决策树的全局优化在准确性，大小和人类可理解性方面表现出良好的前景。然而，许多方法仍然依赖于通用求解器，可扩展性仍然是一个问题。动态规划方法已被证明具有更好的可扩展性，因为它们通过将子树作为独立的子问题解决来利用树结构。然而，这仅适用于可以分别优化子树的任务。我们详细研究了这种关系，并展示了实现这种可分离约束和目标任意组合的动态规划方法。在四个应用领域的实验表明了这种方法的普适性，同时也比通用求解器具有更好的可扩展性。

    Global optimization of decision trees has shown to be promising in terms of accuracy, size, and consequently human comprehensibility. However, many of the methods used rely on general-purpose solvers for which scalability remains an issue. Dynamic programming methods have been shown to scale much better because they exploit the tree structure by solving subtrees as independent subproblems. However, this only works when an objective can be optimized separately for subtrees. We explore this relationship in detail and show necessary and sufficient conditions for such separability and generalize previous dynamic programming approaches into a framework that can optimize any combination of separable objectives and constraints. Experiments on four application domains show the general applicability of this framework, while outperforming the scalability of general-purpose solvers by a large margin.
    
[^12]: 基于生成对抗网络的简化建模方法

    Generative Adversarial Reduced Order Modelling. (arXiv:2305.15881v1 [cs.LG])

    [http://arxiv.org/abs/2305.15881](http://arxiv.org/abs/2305.15881)

    本文提出了一种基于GAN的简化建模方法GAROM，通过引入一个数据驱动的生成对抗模型，能够学习参数微分方程的解，并获得了较好的实验效果。

    

    本文提出了一种新的基于生成对抗网络（GAN）的简化建模方法——GAROM。GAN在多个深度学习领域得到广泛应用，但在简化建模中的应用却鲜有研究。我们将GAN和ROM框架相结合，引入了一种数据驱动的生成对抗模型，能够学习参数微分方程的解。我们将鉴别器网络建模为自编码器，提取输入的相关特征，并将微分方程参数作为生成器和鉴别器网络的输入条件。我们展示了如何将该方法应用于推断问题，提供了实验证据证明了模型的泛化能力，并进行了方法的收敛性研究。

    In this work, we present GAROM, a new approach for reduced order modelling (ROM) based on generative adversarial networks (GANs). GANs have the potential to learn data distribution and generate more realistic data. While widely applied in many areas of deep learning, little research is done on their application for ROM, i.e. approximating a high-fidelity model with a simpler one. In this work, we combine the GAN and ROM framework, by introducing a data-driven generative adversarial model able to learn solutions to parametric differential equations. The latter is achieved by modelling the discriminator network as an autoencoder, extracting relevant features of the input, and applying a conditioning mechanism to the generator and discriminator networks specifying the differential equation parameters. We show how to apply our methodology for inference, provide experimental evidence of the model generalisation, and perform a convergence study of the method.
    
[^13]: MINN：学习微分代数方程的动态和应用于电池建模

    MINN: Learning the dynamics of differential-algebraic equations and application to battery modeling. (arXiv:2304.14422v1 [cs.LG])

    [http://arxiv.org/abs/2304.14422](http://arxiv.org/abs/2304.14422)

    本文提出了一种体系结构，生成模型集成神经网络（MINN）以允许在学习系统物理动态方面进行整合，应用于锂离子电池的电化学动力学建模，并展示了所提出的模型在解释性、精度和计算效率方面的优势。

    

    整合基于物理和基于数据的方法已经成为建模可持续能源系统的常见方法。但是，现有的文献主要集中在生成用于替代基于物理模型的数据驱动替代模型上。这些模型通常以速度为代价换取精度，但缺乏基于物理模型的泛化性、适应性和可解释性，而这些特点在优化和控制实际动态系统的建模中通常是不可或缺的。在本文中，我们提出了一种新的体系结构来生成模型集成神经网络（MINN），以允许在学习系统物理动态方面进行整合。获得的混合模型解决了控制导向建模中一个尚未解决的研究问题，即如何同时获得物理洞察力、数字精度和计算可行性的最优简化模型。我们将所提出的神经网络架构应用于锂离子电池的电化学动力学建模，并展示了所提出的模型在解释性、精度和计算效率方面的优势。

    The concept of integrating physics-based and data-driven approaches has become popular for modeling sustainable energy systems. However, the existing literature mainly focuses on the data-driven surrogates generated to replace physics-based models. These models often trade accuracy for speed but lack the generalisability, adaptability, and interpretability inherent in physics-based models, which are often indispensable in the modeling of real-world dynamic systems for optimization and control purposes. In this work, we propose a novel architecture for generating model-integrated neural networks (MINN) to allow integration on the level of learning physics-based dynamics of the system. The obtained hybrid model solves an unsettled research problem in control-oriented modeling, i.e., how to obtain an optimally simplified model that is physically insightful, numerically accurate, and computationally tractable simultaneously. We apply the proposed neural network architecture to model the el
    
[^14]: ClusterNet：一种基于感知的分布式数据聚类模型

    ClusterNet: A Perception-Based Clustering Model for Scattered Data. (arXiv:2304.14185v1 [cs.LG])

    [http://arxiv.org/abs/2304.14185](http://arxiv.org/abs/2304.14185)

    这项工作介绍了ClusterNet，一种基于感知的分布式数据聚类模型，利用大规模数据集和基于点的深度学习模型，反映人类感知的聚类可分性。

    

    散点图中的聚类分离是一个通常由广泛使用的聚类技术（例如k-means或DBSCAN）来解决的任务。然而，由于这些算法基于非感知度量，它们的输出经常不能反映出人类聚类感知。为了弥合人类聚类感知和机器计算聚类之间的差距，我们提出了一种直接处理分布式数据的学习策略。为了在这些数据上学习感知聚类分离，我们进行了一项众包大规模数据集的工作，其中包括384个人群工作者对双变量数据的7,320个点聚类从属进行了标记。基于这些数据，我们能够训练ClusterNet，这是一个基于点的深度学习模型，被训练成反映人类感知的聚类可分性。为了在人类注释的数据上训练ClusterNet，我们省略了在2D画布上渲染散点图，而是使用了一个PointNet++架构，使其能够直接推理点云。在这项工作中，我们建立了一种基于感知的分布式数据聚类模型，ClusterNet。

    Cluster separation in scatterplots is a task that is typically tackled by widely used clustering techniques, such as for instance k-means or DBSCAN. However, as these algorithms are based on non-perceptual metrics, their output often does not reflect human cluster perception. To bridge the gap between human cluster perception and machine-computed clusters, we propose a learning strategy which directly operates on scattered data. To learn perceptual cluster separation on this data, we crowdsourced a large scale dataset, consisting of 7,320 point-wise cluster affiliations for bivariate data, which has been labeled by 384 human crowd workers. Based on this data, we were able to train ClusterNet, a point-based deep learning model, trained to reflect human perception of cluster separability. In order to train ClusterNet on human annotated data, we omit rendering scatterplots on a 2D canvas, but rather use a PointNet++ architecture enabling inference on point clouds directly. In this work, w
    
[^15]: 基于赌博反馈的在线学习自适应客户端采样在联邦学习中的应用

    Adaptive Client Sampling in Federated Learning via Online Learning with Bandit Feedback. (arXiv:2112.14332v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2112.14332](http://arxiv.org/abs/2112.14332)

    本文提出一种基于赌博反馈的在线学习算法，用于自适应选择哪些客户端用于联邦学习的训练，并通过理论证明该算法可提高优化算法的收敛速度。

    

    由于通信成本高，联邦学习（FL）系统需要采样一部分客户端参与每一轮训练。因此，客户端采样在FL系统中具有重要作用，它影响用于训练机器学习模型的优化算法的收敛速度。尽管具有重要性，但有效采样客户端方面的研究很有限。在本文中，我们将客户端采样建模为在带有赌博反馈的在线学习任务，使用在线随机镜像下降（OSMD）算法来最小化采样方差。然后，我们在理论上展示了我们的采样方法如何提高优化算法的收敛速度。为了处理OSMD中依赖于未知问题参数的调整参数，我们使用在线集成方法和翻倍技巧。我们证明了相对于任何采样序列的动态遗憾界。遗憾界取决于比较序列的总变化。

    Due to the high cost of communication, federated learning (FL) systems need to sample a subset of clients that are involved in each round of training. As a result, client sampling plays an important role in FL systems as it affects the convergence rate of optimization algorithms used to train machine learning models. Despite its importance, there is limited work on how to sample clients effectively. In this paper, we cast client sampling as an online learning task with bandit feedback, which we solve with an online stochastic mirror descent (OSMD) algorithm designed to minimize the sampling variance. We then theoretically show how our sampling method can improve the convergence speed of optimization algorithms. To handle the tuning parameters in OSMD that depend on the unknown problem parameters, we use the online ensemble method and doubling trick. We prove a dynamic regret bound relative to any sampling sequence. The regret bound depends on the total variation of the comparator seque
    
[^16]: 动态治疗效应：模型错误下的高维推断

    Dynamic treatment effects: high-dimensional inference under model misspecification. (arXiv:2111.06818v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2111.06818](http://arxiv.org/abs/2111.06818)

    本文提出了一种新的鲁棒估计方法来解决动态治疗效应估计中的挑战，提高了在模型错误下的高维环境中的估计鲁棒性和可靠性。

    

    估计动态治疗效应在各个学科中都是至关重要的，可以提供有关干预的时变因果影响的微妙见解。然而，由于“维数灾难”和时变混杂的存在，这种估计存在着挑战，可能导致估计偏误。此外，正确地规定日益增多的治疗分配和多重暴露的结果模型似乎过于复杂。鉴于这些挑战，双重鲁棒性的概念，在允许模型错误的情况下，是非常有价值的，然而在实际应用中并没有实现。本文通过提出新的鲁棒估计方法来解决这个问题，同时对治疗分配和结果模型进行鲁棒估计。我们提出了一种“序列模型双重鲁棒性”的解决方案，证明了当每个时间暴露都是双重鲁棒性的时，可以在多个时间点上实现双重鲁棒性。这种方法提高了高维环境下动态治疗效应估计的鲁棒性和可靠性。

    Estimating dynamic treatment effects is essential across various disciplines, offering nuanced insights into the time-dependent causal impact of interventions. However, this estimation presents challenges due to the "curse of dimensionality" and time-varying confounding, which can lead to biased estimates. Additionally, correctly specifying the growing number of treatment assignments and outcome models with multiple exposures seems overly complex. Given these challenges, the concept of double robustness, where model misspecification is permitted, is extremely valuable, yet unachieved in practical applications. This paper introduces a new approach by proposing novel, robust estimators for both treatment assignments and outcome models. We present a "sequential model double robust" solution, demonstrating that double robustness over multiple time points can be achieved when each time exposure is doubly robust. This approach improves the robustness and reliability of dynamic treatment effe
    

