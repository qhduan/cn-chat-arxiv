# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A High Dimensional Model for Adversarial Training: Geometry and Trade-Offs](https://arxiv.org/abs/2402.05674) | 本论文研究了高维模型中的对抗训练，引入了一个可处理的数学模型，并给出了对抗性经验风险最小化器的充分统计的精确渐近描述。研究结果表明存在可以防御而不惩罚准确性的方向，揭示了防御非鲁棒特征的优势。 |
| [^2] | [Learning Operators with Stochastic Gradient Descent in General Hilbert Spaces](https://arxiv.org/abs/2402.04691) | 本研究在一般希尔伯特空间中使用随机梯度下降（SGD）学习算子，提出了适用于目标算子的规则条件，并建立了SGD算法的收敛速度上界，同时展示了对于非线性算子学习的有效性及线性近似收敛特性。 |
| [^3] | [Deep Kalman Filters Can Filter.](http://arxiv.org/abs/2310.19603) | 本研究展示了一类连续时间的深度卡尔曼滤波器（DKFs），可以近似实现一类非马尔可夫和条件高斯信号过程的条件分布律，从而具有在数学金融领域中传统模型基础上的滤波问题的应用潜力。 |
| [^4] | [Causal Similarity-Based Hierarchical Bayesian Models.](http://arxiv.org/abs/2310.12595) | 本文提出了一种基于因果相似性的分层贝叶斯模型，通过学习如何从具有相似因果机制的训练任务中汇集数据来提高机器学习算法对新任务的泛化能力。 |
| [^5] | [Efficient Link Prediction via GNN Layers Induced by Negative Sampling.](http://arxiv.org/abs/2310.09516) | 本研究提出了一种新颖的GNN架构，通过负采样诱导了正边和负边的正向传递，以更加灵活而稳定地进行链接预测。 |
| [^6] | [Simulation-based Inference for Cardiovascular Models.](http://arxiv.org/abs/2307.13918) | 本研究将心血管模型的逆问题作为统计推理进行解决，在体外进行了五个生物标记物的不确定性分析，展示了模拟推理的能力。 |
| [^7] | [Neural Likelihood Surfaces for Spatial Processes with Computationally Intensive or Intractable Likelihoods.](http://arxiv.org/abs/2305.04634) | 研究提出了使用CNN学习空间过程的似然函数。即使在没有确切似然函数的情况下，通过分类任务进行的神经网络的训练，可以隐式地学习似然函数。使用Platt缩放可以提高神经似然面的准确性。 |

# 详细

[^1]: 高维模型的对抗训练：几何和权衡

    A High Dimensional Model for Adversarial Training: Geometry and Trade-Offs

    [https://arxiv.org/abs/2402.05674](https://arxiv.org/abs/2402.05674)

    本论文研究了高维模型中的对抗训练，引入了一个可处理的数学模型，并给出了对抗性经验风险最小化器的充分统计的精确渐近描述。研究结果表明存在可以防御而不惩罚准确性的方向，揭示了防御非鲁棒特征的优势。

    

    本研究在高维情况下，即维度$d$和数据点数$n$与固定比例$\alpha = n / d$发散的上下文中，研究了基于边际的线性分类器中的对抗训练。我们引入了一个可处理的数学模型，可以研究数据和对抗攻击者几何之间的相互作用，同时捕捉到对抗鲁棒性文献中观察到的核心现象。我们的主要理论贡献是在通用的凸且非递增损失函数下，对于对抗性经验风险最小化器的充分统计的精确渐近描述。我们的结果使我们能够精确地刻画数据中与更高的泛化/鲁棒性权衡相关的方向，由一个鲁棒性度量和一个有用性度量定义。特别地，我们揭示了存在一些方向，可以进行防御而不惩罚准确性。最后，我们展示了防御非鲁棒特征的优势。

    This work investigates adversarial training in the context of margin-based linear classifiers in the high-dimensional regime where the dimension $d$ and the number of data points $n$ diverge with a fixed ratio $\alpha = n / d$. We introduce a tractable mathematical model where the interplay between the data and adversarial attacker geometries can be studied, while capturing the core phenomenology observed in the adversarial robustness literature. Our main theoretical contribution is an exact asymptotic description of the sufficient statistics for the adversarial empirical risk minimiser, under generic convex and non-increasing losses. Our result allow us to precisely characterise which directions in the data are associated with a higher generalisation/robustness trade-off, as defined by a robustness and a usefulness metric. In particular, we unveil the existence of directions which can be defended without penalising accuracy. Finally, we show the advantage of defending non-robust featu
    
[^2]: 在一般希尔伯特空间中使用随机梯度下降学习算子

    Learning Operators with Stochastic Gradient Descent in General Hilbert Spaces

    [https://arxiv.org/abs/2402.04691](https://arxiv.org/abs/2402.04691)

    本研究在一般希尔伯特空间中使用随机梯度下降（SGD）学习算子，提出了适用于目标算子的规则条件，并建立了SGD算法的收敛速度上界，同时展示了对于非线性算子学习的有效性及线性近似收敛特性。

    

    本研究探讨了利用随机梯度下降（SGD）在一般希尔伯特空间中学习算子的方法。我们提出了针对目标算子的弱和强规则条件，以描述其内在结构和复杂性。在这些条件下，我们建立了SGD算法的收敛速度的上界，并进行了极小值下界分析，进一步说明我们的收敛分析和规则条件定量地刻画了使用SGD算法解决算子学习问题的可行性。值得强调的是，我们的收敛分析对于非线性算子学习仍然有效。我们证明了SGD估计器将收敛于非线性目标算子的最佳线性近似。此外，将我们的分析应用于基于矢量值和实值再生核希尔伯特空间的算子学习问题，产生了新的收敛结果，从而完善了现有文献的结论。

    This study investigates leveraging stochastic gradient descent (SGD) to learn operators between general Hilbert spaces. We propose weak and strong regularity conditions for the target operator to depict its intrinsic structure and complexity. Under these conditions, we establish upper bounds for convergence rates of the SGD algorithm and conduct a minimax lower bound analysis, further illustrating that our convergence analysis and regularity conditions quantitatively characterize the tractability of solving operator learning problems using the SGD algorithm. It is crucial to highlight that our convergence analysis is still valid for nonlinear operator learning. We show that the SGD estimator will converge to the best linear approximation of the nonlinear target operator. Moreover, applying our analysis to operator learning problems based on vector-valued and real-valued reproducing kernel Hilbert spaces yields new convergence results, thereby refining the conclusions of existing litera
    
[^3]: 深度卡尔曼滤波器可以进行滤波

    Deep Kalman Filters Can Filter. (arXiv:2310.19603v1 [cs.LG])

    [http://arxiv.org/abs/2310.19603](http://arxiv.org/abs/2310.19603)

    本研究展示了一类连续时间的深度卡尔曼滤波器（DKFs），可以近似实现一类非马尔可夫和条件高斯信号过程的条件分布律，从而具有在数学金融领域中传统模型基础上的滤波问题的应用潜力。

    

    深度卡尔曼滤波器（DKFs）是一类神经网络模型，可以从序列数据中生成高斯概率测度。虽然DKFs受卡尔曼滤波器的启发，但它们缺乏与随机滤波问题的具体理论关联，从而限制了它们在传统模型基础上的滤波问题的应用，例如数学金融中的债券和期权定价模型校准。我们通过展示一类连续时间DKFs，可以近似实现一类非马尔可夫和条件高斯信号过程的条件分布律，从而解决了深度学习数学基础中的这个问题。我们的近似结果在路径的足够规则的紧致子集上一致成立，其中近似误差由在给定紧致路径集上均一地计算的最坏情况2-Wasserstein距离量化。

    Deep Kalman filters (DKFs) are a class of neural network models that generate Gaussian probability measures from sequential data. Though DKFs are inspired by the Kalman filter, they lack concrete theoretical ties to the stochastic filtering problem, thus limiting their applicability to areas where traditional model-based filters have been used, e.g.\ model calibration for bond and option prices in mathematical finance. We address this issue in the mathematical foundations of deep learning by exhibiting a class of continuous-time DKFs which can approximately implement the conditional law of a broad class of non-Markovian and conditionally Gaussian signal processes given noisy continuous-times measurements. Our approximation results hold uniformly over sufficiently regular compact subsets of paths, where the approximation error is quantified by the worst-case 2-Wasserstein distance computed uniformly over the given compact set of paths.
    
[^4]: 基于因果相似性的分层贝叶斯模型

    Causal Similarity-Based Hierarchical Bayesian Models. (arXiv:2310.12595v1 [cs.LG])

    [http://arxiv.org/abs/2310.12595](http://arxiv.org/abs/2310.12595)

    本文提出了一种基于因果相似性的分层贝叶斯模型，通过学习如何从具有相似因果机制的训练任务中汇集数据来提高机器学习算法对新任务的泛化能力。

    

    机器学习的关键挑战是对新数据的泛化能力。本研究探讨了对由相关任务组成的数据集进行泛化的问题，这些任务可能在因果机制上存在差异。例如，复杂疾病的观察性医学数据在不同患者间具有疾病因果机制的异质性，这给需要对训练数据集之外的新患者进行泛化的机器学习算法带来了挑战。常用的处理异质性数据集的方法包括为整个数据集学习一个全局模型，为每个任务的数据学习本地模型，或者利用分层、元学习和多任务学习方法从汇集的多个任务的数据中学习泛化。本文提出了基于因果相似性的分层贝叶斯模型，通过学习如何从具有相似因果机制的训练任务中汇集数据来提高对新任务的泛化能力。我们应用这种通用建模方法

    The key challenge underlying machine learning is generalisation to new data. This work studies generalisation for datasets consisting of related tasks that may differ in causal mechanisms. For example, observational medical data for complex diseases suffers from heterogeneity in causal mechanisms of disease across patients, creating challenges for machine learning algorithms that need to generalise to new patients outside of the training dataset. Common approaches for learning supervised models with heterogeneous datasets include learning a global model for the entire dataset, learning local models for each tasks' data, or utilising hierarchical, meta-learning and multi-task learning approaches to learn how to generalise from data pooled across multiple tasks. In this paper we propose causal similarity-based hierarchical Bayesian models to improve generalisation to new tasks by learning how to pool data from training tasks with similar causal mechanisms. We apply this general modelling
    
[^5]: 通过负采样诱导的GNN层进行高效的链接预测

    Efficient Link Prediction via GNN Layers Induced by Negative Sampling. (arXiv:2310.09516v1 [cs.LG])

    [http://arxiv.org/abs/2310.09516](http://arxiv.org/abs/2310.09516)

    本研究提出了一种新颖的GNN架构，通过负采样诱导了正边和负边的正向传递，以更加灵活而稳定地进行链接预测。

    

    链接预测中的图神经网络(GNN)可以大致分为两大类。第一类是基于节点的结构，为每个节点预先计算个体嵌入，并通过简单的解码器进行组合以进行预测。尽管在推理时非常高效（因为节点嵌入只计算一次并反复重用），但模型表达能力有限，导致无法区分对候选边有贡献的同构节点，从而影响准确性。与之相反，第二类方法则依赖于形成针对每个边的子图嵌入，以丰富两两关系的表示，从而消除同构节点，提高准确性，但代价是增加了模型复杂度。为了更好地权衡这个取舍，我们提出了一种新颖的GNN架构，其中的正向传递明确依赖于正边（通常情况下）和负边（我们方法的独特之处），以提供更灵活但仍稳定的信号。

    Graph neural networks (GNNs) for link prediction can loosely be divided into two broad categories. First, \emph{node-wise} architectures pre-compute individual embeddings for each node that are later combined by a simple decoder to make predictions. While extremely efficient at inference time (since node embeddings are only computed once and repeatedly reused), model expressiveness is limited such that isomorphic nodes contributing to candidate edges may not be distinguishable, compromising accuracy. In contrast, \emph{edge-wise} methods rely on the formation of edge-specific subgraph embeddings to enrich the representation of pair-wise relationships, disambiguating isomorphic nodes to improve accuracy, but with the cost of increased model complexity. To better navigate this trade-off, we propose a novel GNN architecture whereby the \emph{forward pass} explicitly depends on \emph{both} positive (as is typical) and negative (unique to our approach) edges to inform more flexible, yet sti
    
[^6]: 基于模拟的推理用于心血管模型

    Simulation-based Inference for Cardiovascular Models. (arXiv:2307.13918v1 [stat.ML])

    [http://arxiv.org/abs/2307.13918](http://arxiv.org/abs/2307.13918)

    本研究将心血管模型的逆问题作为统计推理进行解决，在体外进行了五个生物标记物的不确定性分析，展示了模拟推理的能力。

    

    在过去的几十年中，血流动力学模拟器不断发展，已成为研究体外心血管系统的首选工具。虽然这样的工具通常用于从生理参数模拟全身血流动力学，但解决将波形映射回合理的生理参数的逆问题仍然有很大的潜力和挑战。受模拟推理（SBI）的进展的启发，我们将这个逆问题作为统计推理来处理。与其他方法不同，SBI为感兴趣的参数提供了后验分布，提供了关于个体测量的不确定性的多维表示。我们通过对比几种测量模态来展示这种能力，进行了五个临床感兴趣的生物标志物的体外不确定性分析。除了确认已知事实，比如估计心率的可行性，我们的研究还突出了…

    Over the past decades, hemodynamics simulators have steadily evolved and have become tools of choice for studying cardiovascular systems in-silico. While such tools are routinely used to simulate whole-body hemodynamics from physiological parameters, solving the corresponding inverse problem of mapping waveforms back to plausible physiological parameters remains both promising and challenging. Motivated by advances in simulation-based inference (SBI), we cast this inverse problem as statistical inference. In contrast to alternative approaches, SBI provides \textit{posterior distributions} for the parameters of interest, providing a \textit{multi-dimensional} representation of uncertainty for \textit{individual} measurements. We showcase this ability by performing an in-silico uncertainty analysis of five biomarkers of clinical interest comparing several measurement modalities. Beyond the corroboration of known facts, such as the feasibility of estimating heart rate, our study highlight
    
[^7]: 空间过程的神经似然面

    Neural Likelihood Surfaces for Spatial Processes with Computationally Intensive or Intractable Likelihoods. (arXiv:2305.04634v1 [stat.ME])

    [http://arxiv.org/abs/2305.04634](http://arxiv.org/abs/2305.04634)

    研究提出了使用CNN学习空间过程的似然函数。即使在没有确切似然函数的情况下，通过分类任务进行的神经网络的训练，可以隐式地学习似然函数。使用Platt缩放可以提高神经似然面的准确性。

    

    在空间统计中，当拟合空间过程到真实世界的数据时，快速准确的参数估计和可靠的不确定性量化手段可能是一项具有挑战性的任务，因为似然函数可能评估缓慢或难以处理。 在本研究中，我们提出使用卷积神经网络（CNN）学习空间过程的似然函数。通过特定设计的分类任务，我们的神经网络隐式地学习似然函数，即使在没有显式可用的确切似然函数的情况下也可以实现。一旦在分类任务上进行了训练，我们的神经网络使用Platt缩放进行校准，从而提高了神经似然面的准确性。为了展示我们的方法，我们比较了来自神经似然面的最大似然估计和近似置信区间与两个不同空间过程（高斯过程和对数高斯Cox过程）的相应精确或近似的似然函数构成的等效物。

    In spatial statistics, fast and accurate parameter estimation coupled with a reliable means of uncertainty quantification can be a challenging task when fitting a spatial process to real-world data because the likelihood function might be slow to evaluate or intractable. In this work, we propose using convolutional neural networks (CNNs) to learn the likelihood function of a spatial process. Through a specifically designed classification task, our neural network implicitly learns the likelihood function, even in situations where the exact likelihood is not explicitly available. Once trained on the classification task, our neural network is calibrated using Platt scaling which improves the accuracy of the neural likelihood surfaces. To demonstrate our approach, we compare maximum likelihood estimates and approximate confidence regions constructed from the neural likelihood surface with the equivalent for exact or approximate likelihood for two different spatial processes: a Gaussian Pro
    

