# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Conformal online model aggregation](https://arxiv.org/abs/2403.15527) | 该论文提出了一种基于投票的在线依从模型聚合方法，可以根据过去表现调整模型权重。 |
| [^2] | [An Ordering of Divergences for Variational Inference with Factorized Gaussian Approximations](https://arxiv.org/abs/2403.13748) | 不同的散度排序可以通过它们的变分近似误估不确定性的各种度量，并且因子化近似无法同时匹配这些度量中的任意两个 |
| [^3] | [Diffusion Models as Constrained Samplers for Optimization with Unknown Constraints](https://arxiv.org/abs/2402.18012) | 使用扩散模型在数据流形内进行优化，通过在目标函数定义的Boltzmann分布和扩散模型学习的数据分布的乘积上进行抽样来解决具有未知约束的优化问题。 |
| [^4] | [FIRE: A Failure-Adaptive Reinforcement Learning Framework for Edge Computing Migrations](https://arxiv.org/abs/2209.14399) | 提出了一个面向边缘计算迁移的故障自适应强化学习框架 FIRE，引入ImRE算法，通过在边缘计算数字孪生环境中训练RL策略来适应罕见事件，解决了RL框架在处理偶发服务器故障方面的挑战。 |
| [^5] | [Deep Neural Decision Forest: A Novel Approach for Predicting Recovery or Decease of COVID-19 Patients with Clinical and RT-PCR.](http://arxiv.org/abs/2311.13925) | 该研究介绍了一种利用临床和RT-PCR数据结合深度学习算法来预测COVID-19患者康复或死亡风险的新方法。 |
| [^6] | [Exploit the antenna response consistency to define the alignment criteria for CSI data.](http://arxiv.org/abs/2310.06328) | 本论文提出了一个解决方案，利用天线响应一致性（ARC）来定义适当的对准标准，以解决在WiFi人体活动识别中的自我监督学习算法在CSI数据上无法达到预期性能的问题。 |
| [^7] | [On Memorization and Privacy Risks of Sharpness Aware Minimization.](http://arxiv.org/abs/2310.00488) | 本研究通过对过度参数化模型中的数据记忆的剖析，揭示了尖锐意识最小化算法在非典型数据点上实现的泛化收益。同时，也发现了与此算法相关的更高隐私风险，并提出了缓解策略，以达到更理想的准确度与隐私权衡。 |
| [^8] | [A Double Machine Learning Approach to Combining Experimental and Observational Data.](http://arxiv.org/abs/2307.01449) | 这种双机器学习方法将实验和观测研究结合起来，能够测试假设的违反情况并一致估计处理效应。它提供了半参数高效的处理效应估计器。这种方法在实际环境中是可行的。 |
| [^9] | [A Graph Neural Network Approach to Nanosatellite Task Scheduling: Insights into Learning Mixed-Integer Models.](http://arxiv.org/abs/2303.13773) | 本研究提出基于GNN的纳米卫星任务调度方法，以更好地优化服务质量，解决ONTS问题的复杂性。 |
| [^10] | [HUMAP: Hierarchical Uniform Manifold Approximation and Projection.](http://arxiv.org/abs/2106.07718) | HUMAP是一种新的层次降维技术，能够在层次探索中保留心理地图并在多个数据集和数据类型上具有优越性。 |

# 详细

[^1]: 依从在线模型聚合

    Conformal online model aggregation

    [https://arxiv.org/abs/2403.15527](https://arxiv.org/abs/2403.15527)

    该论文提出了一种基于投票的在线依从模型聚合方法，可以根据过去表现调整模型权重。

    

    依从预测为机器学习模型提供了一种合理的不确定性量化概念，而不需要做出强烈的分布假设。它适用于任何黑盒预测模型，并将点预测转换成具有预定义边际覆盖保证的集预测。然而，依从预测只在事先确定底层机器学习模型的情况下起作用。依从预测中相对较少涉及的问题是模型选择和/或聚合：对于给定的问题，应该如何依从化众多预测方法（随机森林、神经网络、正则化线性模型等）？本文提出了一种新的依从模型聚合方法，用于在线设置，该方法基于将来自多个算法的预测集进行投票，其中根据过去表现调整模型上的权重。

    arXiv:2403.15527v1 Announce Type: cross  Abstract: Conformal prediction equips machine learning models with a reasonable notion of uncertainty quantification without making strong distributional assumptions. It wraps around any black-box prediction model and converts point predictions into set predictions that have a predefined marginal coverage guarantee. However, conformal prediction only works if we fix the underlying machine learning model in advance. A relatively unaddressed issue in conformal prediction is that of model selection and/or aggregation: for a given problem, which of the plethora of prediction methods (random forests, neural nets, regularized linear models, etc.) should we conformalize? This paper proposes a new approach towards conformal model aggregation in online settings that is based on combining the prediction sets from several algorithms by voting, where weights on the models are adapted over time based on past performance.
    
[^2]: 变分推断中因子化高斯近似的差异排序

    An Ordering of Divergences for Variational Inference with Factorized Gaussian Approximations

    [https://arxiv.org/abs/2403.13748](https://arxiv.org/abs/2403.13748)

    不同的散度排序可以通过它们的变分近似误估不确定性的各种度量，并且因子化近似无法同时匹配这些度量中的任意两个

    

    在变分推断（VI）中，给定一个难以处理的分布$p$，问题是从一些更易处理的族$\mathcal{Q}$中计算最佳近似$q$。通常情况下，这种近似是通过最小化Kullback-Leibler (KL)散度来找到的。然而，存在其他有效的散度选择，当$\mathcal{Q}$不包含$p$时，每个散度都支持不同的解决方案。我们分析了在高斯的密集协方差矩阵被对角协方差矩阵的高斯近似所影响的VI结果中，散度选择如何影响VI结果。在这种设置中，我们展示了不同的散度可以通过它们的变分近似误估不确定性的各种度量，如方差、精度和熵，进行\textit{排序}。我们还得出一个不可能定理，表明无法通过因子化近似同时匹配这些度量中的任意两个；因此

    arXiv:2403.13748v1 Announce Type: cross  Abstract: Given an intractable distribution $p$, the problem of variational inference (VI) is to compute the best approximation $q$ from some more tractable family $\mathcal{Q}$. Most commonly the approximation is found by minimizing a Kullback-Leibler (KL) divergence. However, there exist other valid choices of divergences, and when $\mathcal{Q}$ does not contain~$p$, each divergence champions a different solution. We analyze how the choice of divergence affects the outcome of VI when a Gaussian with a dense covariance matrix is approximated by a Gaussian with a diagonal covariance matrix. In this setting we show that different divergences can be \textit{ordered} by the amount that their variational approximations misestimate various measures of uncertainty, such as the variance, precision, and entropy. We also derive an impossibility theorem showing that no two of these measures can be simultaneously matched by a factorized approximation; henc
    
[^3]: 扩散模型作为具有未知约束的优化约束抽样器

    Diffusion Models as Constrained Samplers for Optimization with Unknown Constraints

    [https://arxiv.org/abs/2402.18012](https://arxiv.org/abs/2402.18012)

    使用扩散模型在数据流形内进行优化，通过在目标函数定义的Boltzmann分布和扩散模型学习的数据分布的乘积上进行抽样来解决具有未知约束的优化问题。

    

    处理现实世界的优化问题在分析客观函数或约束不可用时变得尤为具有挑战性。虽然许多研究已经解决了未知目标的问题，但有限研究关注了约束条件未明确给出的情况。忽略这些约束可能导致在实践中不现实的虚假解决方案。为了处理这种未知约束，我们建议使用扩散模型在数据流形内进行优化。为了将优化过程限制在数据流形内，我们将原始优化问题重新构造为通过客观函数定义的Boltzmann分布和扩散模型学习的数据分布的乘积的抽样问题。为了增强抽样效率，我们提出了一个两阶段框架，以引导扩散过程进行预热，然后是Langevin动态。

    arXiv:2402.18012v1 Announce Type: cross  Abstract: Addressing real-world optimization problems becomes particularly challenging when analytic objective functions or constraints are unavailable. While numerous studies have addressed the issue of unknown objectives, limited research has focused on scenarios where feasibility constraints are not given explicitly. Overlooking these constraints can lead to spurious solutions that are unrealistic in practice. To deal with such unknown constraints, we propose to perform optimization within the data manifold using diffusion models. To constrain the optimization process to the data manifold, we reformulate the original optimization problem as a sampling problem from the product of the Boltzmann distribution defined by the objective function and the data distribution learned by the diffusion model. To enhance sampling efficiency, we propose a two-stage framework that begins with a guided diffusion process for warm-up, followed by a Langevin dyna
    
[^4]: FIRE：面向边缘计算迁移的故障自适应强化学习框架

    FIRE: A Failure-Adaptive Reinforcement Learning Framework for Edge Computing Migrations

    [https://arxiv.org/abs/2209.14399](https://arxiv.org/abs/2209.14399)

    提出了一个面向边缘计算迁移的故障自适应强化学习框架 FIRE，引入ImRE算法，通过在边缘计算数字孪生环境中训练RL策略来适应罕见事件，解决了RL框架在处理偶发服务器故障方面的挑战。

    

    在边缘计算中，用户服务配置文件由于用户移动而进行迁移。已经提出了强化学习（RL）框架来进行迁移，通常是在模拟数据上进行训练。然而，现有的RL框架忽视了偶发的服务器故障，尽管罕见，但会影响到像自动驾驶和实时障碍检测等对延迟敏感的应用。因此，这些（罕见事件）故障虽然在历史训练数据中没有得到充分代表，却对基于数据驱动的RL算法构成挑战。由于在实际应用中调整故障频率进行训练是不切实际的，我们引入了FIRE，这是一个通过在边缘计算数字孪生环境中训练RL策略来适应罕见事件的框架。我们提出了ImRE，一种基于重要性抽样的Q-learning算法，它根据罕见事件对值函数的影响进行比例抽样。FIRE考虑了延迟、迁移、故障和备份pl

    arXiv:2209.14399v2 Announce Type: replace-cross  Abstract: In edge computing, users' service profiles are migrated due to user mobility. Reinforcement learning (RL) frameworks have been proposed to do so, often trained on simulated data. However, existing RL frameworks overlook occasional server failures, which although rare, impact latency-sensitive applications like autonomous driving and real-time obstacle detection. Nevertheless, these failures (rare events), being not adequately represented in historical training data, pose a challenge for data-driven RL algorithms. As it is impractical to adjust failure frequency in real-world applications for training, we introduce FIRE, a framework that adapts to rare events by training a RL policy in an edge computing digital twin environment. We propose ImRE, an importance sampling-based Q-learning algorithm, which samples rare events proportionally to their impact on the value function. FIRE considers delay, migration, failure, and backup pl
    
[^5]: 深度神经决策森林：一种用于预测COVID-19患者康复或死亡的新方法，结合临床和RT-PCR数据

    Deep Neural Decision Forest: A Novel Approach for Predicting Recovery or Decease of COVID-19 Patients with Clinical and RT-PCR. (arXiv:2311.13925v2 [eess.IV] UPDATED)

    [http://arxiv.org/abs/2311.13925](http://arxiv.org/abs/2311.13925)

    该研究介绍了一种利用临床和RT-PCR数据结合深度学习算法来预测COVID-19患者康复或死亡风险的新方法。

    

    尽管世界卫生组织宣布大流行已经结束，但COVID-19仍然被视为一种地方性疾病。这次大流行以前所未有的方式打乱了人们的生活并导致广泛的发病率和死亡率。因此，紧急医生有必要确定高风险死亡患者，以便优先考虑医院设备的分配，尤其是在医疗资源有限的地区。尽管存在哪种数据最准确的预测的问题，但患者收集到的数据对于预测COVID-19病例的结果是有益的。因此，本研究旨在实现两个主要目标。首先，我们想要检查深度学习算法是否能够预测患者的死亡率。其次，我们研究了临床和RT-PCR对预测的影响，以确定哪个更可靠。我们定义了四个不同特征集的阶段，并使用可解释的深度学习方法构建了相应的模型。

    COVID-19 continues to be considered an endemic disease in spite of the World Health Organization's declaration that the pandemic is over. This pandemic has disrupted people's lives in unprecedented ways and caused widespread morbidity and mortality. As a result, it is important for emergency physicians to identify patients with a higher mortality risk in order to prioritize hospital equipment, especially in areas with limited medical services. The collected data from patients is beneficial to predict the outcome of COVID-19 cases, although there is a question about which data makes the most accurate predictions. Therefore, this study aims to accomplish two main objectives. First, we want to examine whether deep learning algorithms can predict a patient's morality. Second, we investigated the impact of Clinical and RT-PCR on prediction to determine which one is more reliable. We defined four stages with different feature sets and used interpretable deep learning methods to build appropr
    
[^6]: 利用天线响应一致性定义CSI数据的对准标准

    Exploit the antenna response consistency to define the alignment criteria for CSI data. (arXiv:2310.06328v1 [cs.LG])

    [http://arxiv.org/abs/2310.06328](http://arxiv.org/abs/2310.06328)

    本论文提出了一个解决方案，利用天线响应一致性（ARC）来定义适当的对准标准，以解决在WiFi人体活动识别中的自我监督学习算法在CSI数据上无法达到预期性能的问题。

    

    自我监督学习（SSL）用于基于WiFi的人体活动识别（HAR）由于能够解决标注数据不足的挑战而具有很大的潜力。然而，直接将原本设计用于其他领域的SSL算法，特别是对比学习，移植到CSI数据上往往无法达到预期的性能。我们将这个问题归因于对准标准不当，这破坏了特征空间和输入空间之间的语义距离一致性。为了解决这个挑战，我们引入了``Anetenna Response Consistency (ARC)''作为定义合适对准标准的解决方案。ARC的设计在保留输入空间的语义信息的同时，引入了对现实世界噪声的鲁棒性。我们从CSI数据结构的角度分析了ARC，并展示了其最优解导致了从输入CSI数据到特征映射中的动作向量的直接映射。

    Self-supervised learning (SSL) for WiFi-based human activity recognition (HAR) holds great promise due to its ability to address the challenge of insufficient labeled data. However, directly transplanting SSL algorithms, especially contrastive learning, originally designed for other domains to CSI data, often fails to achieve the expected performance. We attribute this issue to the inappropriate alignment criteria, which disrupt the semantic distance consistency between the feature space and the input space. To address this challenge, we introduce \textbf{A}netenna \textbf{R}esponse \textbf{C}onsistency (ARC) as a solution to define proper alignment criteria. ARC is designed to retain semantic information from the input space while introducing robustness to real-world noise. We analyze ARC from the perspective of CSI data structure, demonstrating that its optimal solution leads to a direct mapping from input CSI data to action vectors in the feature map. Furthermore, we provide extensi
    
[^7]: 关于尖锐意识最小化的记忆和隐私风险研究

    On Memorization and Privacy Risks of Sharpness Aware Minimization. (arXiv:2310.00488v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.00488](http://arxiv.org/abs/2310.00488)

    本研究通过对过度参数化模型中的数据记忆的剖析，揭示了尖锐意识最小化算法在非典型数据点上实现的泛化收益。同时，也发现了与此算法相关的更高隐私风险，并提出了缓解策略，以达到更理想的准确度与隐私权衡。

    

    在许多最近的研究中，设计寻求神经网络损失优化中更平坦的极值的算法成为焦点，因为有经验证据表明这会在许多数据集上导致更好的泛化性能。在这项工作中，我们通过过度参数化模型中的数据记忆视角来剖析这些性能收益。我们定义了一个新的度量指标，帮助我们确定相对于普通SGD，寻求更平坦极值的算法在哪些数据点上表现更好。我们发现，尖锐意识最小化（SAM）所实现的泛化收益在非典型数据点上特别显著，这需要记忆。这一认识帮助我们揭示与SAM相关的更高的隐私风险，并通过详尽的实证评估进行验证。最后，我们提出缓解策略，以实现更理想的准确度与隐私权衡。

    In many recent works, there is an increased focus on designing algorithms that seek flatter optima for neural network loss optimization as there is empirical evidence that it leads to better generalization performance in many datasets. In this work, we dissect these performance gains through the lens of data memorization in overparameterized models. We define a new metric that helps us identify which data points specifically do algorithms seeking flatter optima do better when compared to vanilla SGD. We find that the generalization gains achieved by Sharpness Aware Minimization (SAM) are particularly pronounced for atypical data points, which necessitate memorization. This insight helps us unearth higher privacy risks associated with SAM, which we verify through exhaustive empirical evaluations. Finally, we propose mitigation strategies to achieve a more desirable accuracy vs privacy tradeoff.
    
[^8]: 将实验数据与观测数据结合的双机器学习方法

    A Double Machine Learning Approach to Combining Experimental and Observational Data. (arXiv:2307.01449v1 [stat.ME])

    [http://arxiv.org/abs/2307.01449](http://arxiv.org/abs/2307.01449)

    这种双机器学习方法将实验和观测研究结合起来，能够测试假设的违反情况并一致估计处理效应。它提供了半参数高效的处理效应估计器。这种方法在实际环境中是可行的。

    

    实验和观测研究通常由于无法测试的假设而缺乏有效性。我们提出了一种双机器学习方法，将实验和观测研究结合起来，使从业人员能够测试假设违反情况并一致估计处理效应。我们的框架在较轻的假设下测试外部效度和可忽视性的违反情况。当只有一个假设被违反时，我们提供半参数高效的处理效应估计器。然而，我们的无免费午餐定理强调了准确识别违反的假设对一致的处理效应估计的必要性。我们通过三个实际案例研究展示了我们方法的适用性，并突出了其在实际环境中的相关性。

    Experimental and observational studies often lack validity due to untestable assumptions. We propose a double machine learning approach to combine experimental and observational studies, allowing practitioners to test for assumption violations and estimate treatment effects consistently. Our framework tests for violations of external validity and ignorability under milder assumptions. When only one assumption is violated, we provide semi-parametrically efficient treatment effect estimators. However, our no-free-lunch theorem highlights the necessity of accurately identifying the violated assumption for consistent treatment effect estimation. We demonstrate the applicability of our approach in three real-world case studies, highlighting its relevance for practical settings.
    
[^9]: 基于图神经网络的纳米卫星任务调度方法：学习混合整数模型的洞见

    A Graph Neural Network Approach to Nanosatellite Task Scheduling: Insights into Learning Mixed-Integer Models. (arXiv:2303.13773v1 [cs.LG])

    [http://arxiv.org/abs/2303.13773](http://arxiv.org/abs/2303.13773)

    本研究提出基于GNN的纳米卫星任务调度方法，以更好地优化服务质量，解决ONTS问题的复杂性。

    

    本研究探讨如何利用图神经网络（GNN）更有效地调度纳米卫星任务。在离线纳米卫星任务调度（ONTS）问题中，目标是找到在轨道上执行任务的最佳安排，同时考虑服务质量（QoS）方面的考虑因素，如优先级，最小和最大激活事件，执行时间框架，周期和执行窗口，以及卫星电力资源和能量收集和管理的复杂性的约束。ONTS问题已经使用传统的数学公式和精确方法进行了处理，但是它们在问题的挑战性案例中的适用性有限。本研究考察了在这种情况下使用GNN的方法，该方法已经成功应用于许多优化问题，包括旅行商问题，调度问题和设施放置问题。在本文中，我们将ONTS问题的MILP实例完全表示成二分图网络结构来应用GNN。

    This study investigates how to schedule nanosatellite tasks more efficiently using Graph Neural Networks (GNN). In the Offline Nanosatellite Task Scheduling (ONTS) problem, the goal is to find the optimal schedule for tasks to be carried out in orbit while taking into account Quality-of-Service (QoS) considerations such as priority, minimum and maximum activation events, execution time-frames, periods, and execution windows, as well as constraints on the satellite's power resources and the complexity of energy harvesting and management. The ONTS problem has been approached using conventional mathematical formulations and precise methods, but their applicability to challenging cases of the problem is limited. This study examines the use of GNNs in this context, which has been effectively applied to many optimization problems, including traveling salesman problems, scheduling problems, and facility placement problems. Here, we fully represent MILP instances of the ONTS problem in biparti
    
[^10]: HUMAP：层次统一流形逼近与投影

    HUMAP: Hierarchical Uniform Manifold Approximation and Projection. (arXiv:2106.07718v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2106.07718](http://arxiv.org/abs/2106.07718)

    HUMAP是一种新的层次降维技术，能够在层次探索中保留心理地图并在多个数据集和数据类型上具有优越性。

    

    数据降维技术有助于分析人员理解高维空间中的模式。这些技术通常以散点图形式呈现，应用于各种科学领域，并促进集群和数据样本之间的相似性分析。针对包含许多粒度或遵循信息可视化准则的数据集的分析，层次降维技术是最适合的方法，因为它们先前呈现了主要结构并可以按需提供详细信息。然而，当前的层次降维技术并不能完全解决文献中存在的问题，因为它们不能在层次级别之间保持投影心理地图，也不适用于大多数数据类型。本文介绍了一种新的层次降维技术HUMAP，旨在灵活地保留本地和全局结构以及心理地图，在层次探索中具有优越性，并在多个数据集和数据类型上提供了实证证据。

    Dimensionality reduction (DR) techniques help analysts understand patterns in high-dimensional spaces. These techniques, often represented by scatter plots, are employed in diverse science domains and facilitate similarity analysis among clusters and data samples. For datasets containing many granularities or when analysis follows the information visualization mantra, hierarchical DR techniques are the most suitable approach since they present major structures beforehand and details on demand. However, current hierarchical DR techniques are not fully capable of addressing literature problems because they do not preserve the projection mental map across hierarchical levels or are not suitable for most data types. This work presents HUMAP, a novel hierarchical dimensionality reduction technique designed to be flexible in preserving local and global structures and the mental map throughout hierarchical exploration. We provide empirical evidence of our technique's superiority compared with
    

