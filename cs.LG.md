# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Holo-VQVAE: VQ-VAE for phase-only holograms](https://arxiv.org/abs/2404.01330) | Holo-VQVAE是一种针对仅相位全息图的新型生成框架，结合了矢量量化变分自动编码器的结构，通过集成角谱方法来学习图像域，在全息图生成中实现了从复杂分布中直接生成多样化全息内容。 |
| [^2] | [Accelerated Inference and Reduced Forgetting: The Dual Benefits of Early-Exit Networks in Continual Learning](https://arxiv.org/abs/2403.07404) | 早期退出网络在持续学习中展现出降低遗忘和在资源利用上表现优异的特点 |
| [^3] | [FlexLLM: A System for Co-Serving Large Language Model Inference and Parameter-Efficient Finetuning](https://arxiv.org/abs/2402.18789) | FlexLLM是第一个可以在同一迭代中共同提供推理和参数高效微调请求的系统，通过引入标记级微调机制实现共享GPU资源的高效利用 |
| [^4] | [Prismatic: Interactive Multi-View Cluster Analysis of Concept Stocks](https://arxiv.org/abs/2402.08978) | Prismatic是一个集成历史数据分析和业务关系知识的交互式多视角概念股集群分析系统，通过多视角集群分析方法，丰富了数据驱动的集群，并提供了细致的业务相关性理解。 |
| [^5] | [Nonparametric Estimation via Variance-Reduced Sketching.](http://arxiv.org/abs/2401.11646) | 本文提出了一种名为Variance-Reduced Sketching的框架，用于在高维度中估计密度函数和非参数回归函数。该方法通过将函数概念化为矩阵，并采用草图技术来降低维度灾难引起的方差，展示了鲁棒性能和显著改进。 |
| [^6] | [Generating synthetic data for neural operators.](http://arxiv.org/abs/2401.02398) | 该论文提出了一种生成神经算子的合成数据的新方法，为训练网络提供不需要数值求解PDE的数据。 |
| [^7] | [Learning the hub graphical Lasso model with the structured sparsity via an efficient algorithm.](http://arxiv.org/abs/2308.08852) | 通过双重交替方向乘子法 (ADMM) 和半平滑牛顿 (SSN) 基于增广对偶法 (ALM) 的方法，我们提出了一个高效算法来学习具有结构稀疏性的核心图形Lasso模型，该算法能够在大维度的任务中节省超过70\%的执行时间，并且具有较高的性能。 |
| [^8] | [An Adaptive Method for Weak Supervision with Drifting Data.](http://arxiv.org/abs/2306.01658) | 本文在非稳态设置中提出了一种自适应弱监督方法，该方法可以推断序列数据的未知标签，并适应时间漂移，而无需假设漂移幅度。 |
| [^9] | [Convergence of a Normal Map-based Prox-SGD Method under the KL Inequality.](http://arxiv.org/abs/2305.05828) | 本文提出了一种新的随机正态映射算法用于非凸复合型优化问题，并证明其收敛性质。该方法扩展了基本Proximal随机梯度法的更有限的收敛保证。 |
| [^10] | [Adaptive Estimation of Random Vectors with Bandit Feedback: A mean-squared error viewpoint.](http://arxiv.org/abs/2203.16810) | 本文研究了在每轮仅观察部分未知协方差的高斯向量情况下，通过均方误差估计的顺序学习问题，并提出了连续消除算法的一种变体。同时，导出了样本复杂性的极小值下界。 |

# 详细

[^1]: Holo-VQVAE：用于仅相位全息图的VQ-VAE

    Holo-VQVAE: VQ-VAE for phase-only holograms

    [https://arxiv.org/abs/2404.01330](https://arxiv.org/abs/2404.01330)

    Holo-VQVAE是一种针对仅相位全息图的新型生成框架，结合了矢量量化变分自动编码器的结构，通过集成角谱方法来学习图像域，在全息图生成中实现了从复杂分布中直接生成多样化全息内容。

    

    Holography stands at the forefront of visual technology innovation, offering immersive, three-dimensional visualizations through the manipulation of light wave amplitude and phase. Contemporary research in hologram generation has predominantly focused on image-to-hologram conversion, producing holograms from existing images. These approaches, while effective, inherently limit the scope of innovation and creativity in hologram generation. In response to this limitation, we present Holo-VQVAE, a novel generative framework tailored for phase-only holograms (POHs). Holo-VQVAE leverages the architecture of Vector Quantized Variational AutoEncoders, enabling it to learn the complex distributions of POHs. Furthermore, it integrates the Angular Spectrum Method into the training process, facilitating learning in the image domain. This framework allows for the generation of unseen, diverse holographic content directly from its intricately learned distributions.

    arXiv:2404.01330v1 Announce Type: cross  Abstract: Holography stands at the forefront of visual technology innovation, offering immersive, three-dimensional visualizations through the manipulation of light wave amplitude and phase. Contemporary research in hologram generation has predominantly focused on image-to-hologram conversion, producing holograms from existing images. These approaches, while effective, inherently limit the scope of innovation and creativity in hologram generation. In response to this limitation, we present Holo-VQVAE, a novel generative framework tailored for phase-only holograms (POHs). Holo-VQVAE leverages the architecture of Vector Quantized Variational AutoEncoders, enabling it to learn the complex distributions of POHs. Furthermore, it integrates the Angular Spectrum Method into the training process, facilitating learning in the image domain. This framework allows for the generation of unseen, diverse holographic content directly from its intricately learne
    
[^2]: 提高推理速度和减少遗忘：早期退出网络在持续学习中的双重好处

    Accelerated Inference and Reduced Forgetting: The Dual Benefits of Early-Exit Networks in Continual Learning

    [https://arxiv.org/abs/2403.07404](https://arxiv.org/abs/2403.07404)

    早期退出网络在持续学习中展现出降低遗忘和在资源利用上表现优异的特点

    

    arXiv:2403.07404v1 公告类型: 跨界 摘要: 受深度神经网络能源高效利用需求驱动，早期退出方法备受关注。这些策略通过在网络早期做出决定，实现快速预测，从而节省计算时间和资源。然而，迄今为止，早期退出网络仅针对静态数据分布进行了开发，限制了它们在具有持续非静态数据的实际场景中的应用。本研究旨在探讨早期退出网络的持续学习。我们改编现有的持续学习方法以适应早期退出架构，并研究它们在持续设置中的行为。我们注意到，早期网络层表现出减少遗忘，即使使用的资源显著更少，也能胜过标准网络。此外，我们分析任务最近性偏差对早期退出推理的影响，并提出任务...

    arXiv:2403.07404v1 Announce Type: cross  Abstract: Driven by the demand for energy-efficient employment of deep neural networks, early-exit methods have experienced a notable increase in research attention. These strategies allow for swift predictions by making decisions early in the network, thereby conserving computation time and resources. However, so far the early-exit networks have only been developed for stationary data distributions, which restricts their application in real-world scenarios with continuous non-stationary data. This study aims to explore the continual learning of the early-exit networks. We adapt existing continual learning methods to fit with early-exit architectures and investigate their behavior in the continual setting. We notice that early network layers exhibit reduced forgetting and can outperform standard networks even when using significantly fewer resources. Furthermore, we analyze the impact of task-recency bias on early-exit inference and propose Task
    
[^3]: FlexLLM：一种用于共同提供大型语言模型推理和参数高效微调的系统

    FlexLLM: A System for Co-Serving Large Language Model Inference and Parameter-Efficient Finetuning

    [https://arxiv.org/abs/2402.18789](https://arxiv.org/abs/2402.18789)

    FlexLLM是第一个可以在同一迭代中共同提供推理和参数高效微调请求的系统，通过引入标记级微调机制实现共享GPU资源的高效利用

    

    Parameter-efficient finetuning（PEFT）是一种广泛使用的技术，用于为不同任务调整大型语言模型。通常，服务提供商会为用户创建单独的系统，以执行PEFT模型微调和推理任务。这是因为现有系统无法处理包含推理和PEFT微调请求混合的工作负载。因此，共享的GPU资源利用不足，导致效率低下。为解决这一问题，我们提出了FlexLLM，这是第一个可以在同一迭代中为推理和参数高效微调请求提供服务的系统。我们的系统利用这两个任务的互补性质，并利用共享的GPU资源来共同运行它们，使用一种称为共同提供的方法。为实现这一目标，FlexLLM引入了一种新颖的标记级微调机制，将序列的微调计算分解为更小的标记级计算，并使用依赖并行化。

    arXiv:2402.18789v1 Announce Type: cross  Abstract: Parameter-efficient finetuning (PEFT) is a widely used technique to adapt large language models for different tasks. Service providers typically create separate systems for users to perform PEFT model finetuning and inference tasks. This is because existing systems cannot handle workloads that include a mix of inference and PEFT finetuning requests. As a result, shared GPU resources are underutilized, leading to inefficiencies. To address this problem, we present FlexLLM, the first system that can serve inference and parameter-efficient finetuning requests in the same iteration. Our system leverages the complementary nature of these two tasks and utilizes shared GPU resources to run them jointly, using a method called co-serving. To achieve this, FlexLLM introduces a novel token-level finetuning mechanism, which breaks down the finetuning computation of a sequence into smaller token-level computations and uses dependent parallelization
    
[^4]: Prismatic:交互式多视角概念股集群分析

    Prismatic: Interactive Multi-View Cluster Analysis of Concept Stocks

    [https://arxiv.org/abs/2402.08978](https://arxiv.org/abs/2402.08978)

    Prismatic是一个集成历史数据分析和业务关系知识的交互式多视角概念股集群分析系统，通过多视角集群分析方法，丰富了数据驱动的集群，并提供了细致的业务相关性理解。

    

    arXiv:2402.08978v1 公告类型:跨领域 摘要:金融集群分析使投资者能够发现投资替代品，并避免承担过高的风险。然而，这种分析任务面临许多挑战，如大量的两两比较、时间跨度的动态相关性以及从业务关系知识中得出推论的模糊性。我们提出了Prismatic，一种可视化分析系统，它整合了历史性能的定量分析和业务关系知识的定性分析，以交互方式对相关业务进行集群分析。Prismatic具有三个集群生成过程：动态集群生成、基于知识的集群探索和基于相关性的集群验证。利用多视角集群分析方法，它通过知识驱动的相似性丰富了数据驱动的集群，提供了对业务相关性的细致理解。通过良好协调的可视化视图，Prismatic便于了解企业的关联性。

    arXiv:2402.08978v1 Announce Type: cross Abstract: Financial cluster analysis allows investors to discover investment alternatives and avoid undertaking excessive risks. However, this analytical task faces substantial challenges arising from many pairwise comparisons, the dynamic correlations across time spans, and the ambiguity in deriving implications from business relational knowledge. We propose Prismatic, a visual analytics system that integrates quantitative analysis of historical performance and qualitative analysis of business relational knowledge to cluster correlated businesses interactively. Prismatic features three clustering processes: dynamic cluster generation, knowledge-based cluster exploration, and correlation-based cluster validation. Utilizing a multi-view clustering approach, it enriches data-driven clusters with knowledge-driven similarity, providing a nuanced understanding of business correlations. Through well-coordinated visual views, Prismatic facilitates a com
    
[^5]: 通过方差降低的草图进行非参数估计

    Nonparametric Estimation via Variance-Reduced Sketching. (arXiv:2401.11646v1 [stat.ML])

    [http://arxiv.org/abs/2401.11646](http://arxiv.org/abs/2401.11646)

    本文提出了一种名为Variance-Reduced Sketching的框架，用于在高维度中估计密度函数和非参数回归函数。该方法通过将函数概念化为矩阵，并采用草图技术来降低维度灾难引起的方差，展示了鲁棒性能和显著改进。

    

    非参数模型在各个科学和工程领域中备受关注。经典的核方法在低维情况下具有数值稳定性和统计可靠性，但在高维情况下由于维度灾难变得不够适用。在本文中，我们引入了一个名为Variance-Reduced Sketching（VRS）的新框架，专门用于在降低维度灾难的同时在高维度中估计密度函数和非参数回归函数。我们的框架将多变量函数概念化为无限大小的矩阵，并借鉴了数值线性代数文献中的一种新的草图技术来降低估计问题中的方差。我们通过一系列的模拟实验和真实数据应用展示了VRS的鲁棒性能。值得注意的是，在许多密度估计问题中，VRS相较于现有的神经网络估计器和经典的核方法表现出显著的改进。

    Nonparametric models are of great interest in various scientific and engineering disciplines. Classical kernel methods, while numerically robust and statistically sound in low-dimensional settings, become inadequate in higher-dimensional settings due to the curse of dimensionality. In this paper, we introduce a new framework called Variance-Reduced Sketching (VRS), specifically designed to estimate density functions and nonparametric regression functions in higher dimensions with a reduced curse of dimensionality. Our framework conceptualizes multivariable functions as infinite-size matrices, and facilitates a new sketching technique motivated by numerical linear algebra literature to reduce the variance in estimation problems. We demonstrate the robust numerical performance of VRS through a series of simulated experiments and real-world data applications. Notably, VRS shows remarkable improvement over existing neural network estimators and classical kernel methods in numerous density 
    
[^6]: 生成神经算子的合成数据

    Generating synthetic data for neural operators. (arXiv:2401.02398v1 [cs.LG])

    [http://arxiv.org/abs/2401.02398](http://arxiv.org/abs/2401.02398)

    该论文提出了一种生成神经算子的合成数据的新方法，为训练网络提供不需要数值求解PDE的数据。

    

    近期文献中的许多发展展示了深度学习在获取偏微分方程（PDEs）的数值解方面的潜力，这超出了当前数值求解器的能力。然而，数据驱动的神经算子都存在同样的问题：训练网络所需的数据依赖于传统的数值求解器，如有限差分或有限元等。本文提出了一种新方法，用于生成合成的函数训练数据，而无需数值求解PDE。我们的方法很简单：我们从已知解位于的经典理论解空间（例如$H_0^1(\Omega)$）中抽取大量独立同分布的“随机函数”$u_j$，然后将每个随机解方案代入方程并获得相应的右侧函数$f_j$，将$(f_j, u_j)_{j=1}^N$作为监督训练数据。

    Numerous developments in the recent literature show the promising potential of deep learning in obtaining numerical solutions to partial differential equations (PDEs) beyond the reach of current numerical solvers. However, data-driven neural operators all suffer from the same problem: the data needed to train a network depends on classical numerical solvers such as finite difference or finite element, among others. In this paper, we propose a new approach to generating synthetic functional training data that does not require solving a PDE numerically. The way we do this is simple: we draw a large number $N$ of independent and identically distributed `random functions' $u_j$ from the underlying solution space (e.g., $H_0^1(\Omega)$) in which we know the solution lies according to classical theory. We then plug each such random candidate solution into the equation and get a corresponding right-hand side function $f_j$ for the equation, and consider $(f_j, u_j)_{j=1}^N$ as supervised trai
    
[^7]: 通过高效算法学习具有结构稀疏性的核心图形Lasso模型

    Learning the hub graphical Lasso model with the structured sparsity via an efficient algorithm. (arXiv:2308.08852v1 [math.OC])

    [http://arxiv.org/abs/2308.08852](http://arxiv.org/abs/2308.08852)

    通过双重交替方向乘子法 (ADMM) 和半平滑牛顿 (SSN) 基于增广对偶法 (ALM) 的方法，我们提出了一个高效算法来学习具有结构稀疏性的核心图形Lasso模型，该算法能够在大维度的任务中节省超过70\%的执行时间，并且具有较高的性能。

    

    图形模型在从生物分析到推荐系统等众多任务中展现出了良好的性能。然而，具有核心节点的图形模型在数据维度较大时计算上存在困难。为了高效估计核心图形模型，我们提出了一个两阶段算法。所提出的算法首先通过双重交替方向乘子法 (ADMM) 生成一个良好的初始点，然后使用半平滑牛顿 (SSN) 基于增广对偶法 (ALM) 的方法进行热启动，以计算出能够在实际任务中精确到足够程度的解。广义雅可比矩阵的稀疏结构确保了该算法能够非常高效地获得一个良好的解。在合成数据和真实数据的全面实验中，该算法明显优于现有的最先进算法。特别是在某些高维任务中，它可以节省超过70\%的执行时间，同时仍然可以达到很好的性能。

    Graphical models have exhibited their performance in numerous tasks ranging from biological analysis to recommender systems. However, graphical models with hub nodes are computationally difficult to fit, particularly when the dimension of the data is large. To efficiently estimate the hub graphical models, we introduce a two-phase algorithm. The proposed algorithm first generates a good initial point via a dual alternating direction method of multipliers (ADMM), and then warm starts a semismooth Newton (SSN) based augmented Lagrangian method (ALM) to compute a solution that is accurate enough for practical tasks. The sparsity structure of the generalized Jacobian ensures that the algorithm can obtain a nice solution very efficiently. Comprehensive experiments on both synthetic data and real data show that it obviously outperforms the existing state-of-the-art algorithms. In particular, in some high dimensional tasks, it can save more than 70\% of the execution time, meanwhile still ach
    
[^8]: 一种用于漂移数据的自适应弱监督方法

    An Adaptive Method for Weak Supervision with Drifting Data. (arXiv:2306.01658v1 [cs.LG])

    [http://arxiv.org/abs/2306.01658](http://arxiv.org/abs/2306.01658)

    本文在非稳态设置中提出了一种自适应弱监督方法，该方法可以推断序列数据的未知标签，并适应时间漂移，而无需假设漂移幅度。

    

    我们提出了一种在非稳态设置中具有正式质量保证的自适应方法，用于对数据序列进行弱监督标记。我们的目标是通过使用提供每个数据点正确分类的独立嘈杂信号的弱监督源来推断未知标签。这种情况包括众包和编程式弱监督。我们重点研究非稳态情况，在这种情况下，弱监督源的精度可能会随时间漂移，例如由于底层数据分布的变化。由于漂移，旧数据可能会提供误导性信息来推断当前数据点的标签。以往的工作依赖于先验对漂移幅度的假设，以决定使用多少过去的数据。相比之下，我们的算法不需要任何漂移假设，而是根据输入进行自适应。特别地，在每个步骤中，我们的算法保证弱监督源当前准确度的估计。

    We introduce an adaptive method with formal quality guarantees for weak supervision in a non-stationary setting. Our goal is to infer the unknown labels of a sequence of data by using weak supervision sources that provide independent noisy signals of the correct classification for each data point. This setting includes crowdsourcing and programmatic weak supervision. We focus on the non-stationary case, where the accuracy of the weak supervision sources can drift over time, e.g., because of changes in the underlying data distribution. Due to the drift, older data could provide misleading information to infer the label of the current data point. Previous work relied on a priori assumptions on the magnitude of the drift to decide how much data to use from the past. Comparatively, our algorithm does not require any assumptions on the drift, and it adapts based on the input. In particular, at each step, our algorithm guarantees an estimation of the current accuracies of the weak supervisio
    
[^9]: 基于正态映射的Prox-SGD方法在KL不等式下的收敛性

    Convergence of a Normal Map-based Prox-SGD Method under the KL Inequality. (arXiv:2305.05828v1 [math.OC])

    [http://arxiv.org/abs/2305.05828](http://arxiv.org/abs/2305.05828)

    本文提出了一种新的随机正态映射算法用于非凸复合型优化问题，并证明其收敛性质。该方法扩展了基本Proximal随机梯度法的更有限的收敛保证。

    

    本文提出了一种新颖的随机正态映射算法（$\mathsf{norM}\text{-}\mathsf{SGD}$）用于非凸复合型优化问题，并讨论了其收敛性质。使用基于时间窗口的策略，首先分析了$\mathsf{norM}\text{-}\mathsf{SGD}$的全局收敛行为，并证明了所生成的迭代序列$\{\boldsymbol{x}^k\}_k$的每个累积点几乎确定地和期望上都对应于一个稳定点。所得结果在标准假设下成立，并扩展了基本Proximal随机梯度法的更有限的收敛保证。此外，基于著名的Kurdyka-{\L}ojasiewicz（KL）分析框架，我们为迭代序列$\{\boldsymbol{x}^k\}_k$提供了新的逐点收敛结果，并得出了取决于基础KL指数$\boldsymbol{\theta}$和步长动态$\{\alpha_k\}_k$的收敛速率。

    In this paper, we present a novel stochastic normal map-based algorithm ($\mathsf{norM}\text{-}\mathsf{SGD}$) for nonconvex composite-type optimization problems and discuss its convergence properties. Using a time window-based strategy, we first analyze the global convergence behavior of $\mathsf{norM}\text{-}\mathsf{SGD}$ and it is shown that every accumulation point of the generated sequence of iterates $\{\boldsymbol{x}^k\}_k$ corresponds to a stationary point almost surely and in an expectation sense. The obtained results hold under standard assumptions and extend the more limited convergence guarantees of the basic proximal stochastic gradient method. In addition, based on the well-known Kurdyka-{\L}ojasiewicz (KL) analysis framework, we provide novel point-wise convergence results for the iterates $\{\boldsymbol{x}^k\}_k$ and derive convergence rates that depend on the underlying KL exponent $\boldsymbol{\theta}$ and the step size dynamics $\{\alpha_k\}_k$. Specifically, for the 
    
[^10]: 自适应估计带有赌博反馈的随机向量：从均方误差视角来看

    Adaptive Estimation of Random Vectors with Bandit Feedback: A mean-squared error viewpoint. (arXiv:2203.16810v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2203.16810](http://arxiv.org/abs/2203.16810)

    本文研究了在每轮仅观察部分未知协方差的高斯向量情况下，通过均方误差估计的顺序学习问题，并提出了连续消除算法的一种变体。同时，导出了样本复杂性的极小值下界。

    

    本文考虑在每轮观察仅有$ m < K $个未知协方差的高斯$ K $向量的问题下，通过均方误差（MSE）估计顺序学习。我们首先建立了MSE估计的集中界限。然后，我们使用赌博反馈的方法重新构建估计问题，并提出了一种连续消除算法的变体。我们还导出了一个极小值下界，以了解该问题样本复杂性的基本限制。

    We consider the problem of sequentially learning to estimate, in the mean squared error (MSE) sense, a Gaussian $K$-vector of unknown covariance by observing only $m < K$ of its entries in each round. We first establish a concentration bound for MSE estimation. We then frame the estimation problem with bandit feedback, and propose a variant of the successive elimination algorithm. We also derive a minimax lower bound to understand the fundamental limit on the sample complexity of this problem.
    

