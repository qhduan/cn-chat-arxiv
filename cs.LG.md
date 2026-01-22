# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [GSINA: Improving Subgraph Extraction for Graph Invariant Learning via Graph Sinkhorn Attention](https://arxiv.org/abs/2402.07191) | 本文提出了一种改进的图不变学习方法，通过稀疏性、软性和可微性原则来提取不变子图，从而提高图学习的泛化性能。 |
| [^2] | [Towards an AI Accountability Policy.](http://arxiv.org/abs/2307.13658) | 这份白皮书是对美国国家电信和信息管理局的“AI问责政策评论请求”的回应，提出了一组相互关联的AI问责政策建议。 |
| [^3] | [Learning minimal representations of stochastic processes with variational autoencoders.](http://arxiv.org/abs/2307.11608) | 本文引入了一种无监督机器学习方法，使用变分自动编码器确定最小参数集，有效描述随机过程动力学，并生成能准确复制预期随机行为的新轨迹。 |
| [^4] | [A Finite Expression Method for Solving High-Dimensional Committor Problems.](http://arxiv.org/abs/2306.12268) | 本文提出了一种用于解决高维Committor问题的有限表达式方法(FEX)，该方法通过深度神经网络学习最优非线性函数和系数值，能够显著提高计算效果。 |
| [^5] | [Benchmark data to study the influence of pre-training on explanation performance in MR image classification.](http://arxiv.org/abs/2306.12150) | 本研究提出了一个MRI分类任务的基准数据集，用于评估不同模型的解释性能。实验结果表明，XAI方法并不一定比简单模型提供更好的解释，且CNN的解释能力取决于底层数据的复杂性和标签的质量。 |
| [^6] | [Finite Expression Methods for Discovering Physical Laws from Data.](http://arxiv.org/abs/2305.08342) | 本文介绍了一种名为"有限表达法" (FEX) 的深度符号学习方法，通过学习动态数据中PDE解的导数，发现控制方程的解析表达式。相对于其他现有方法，FEX在多种问题上表现出更好的数值性能，包括时变的PDE问题和具有时变系数的非线性动力系统。 |
| [^7] | [Learning from Discriminatory Training Data.](http://arxiv.org/abs/1912.08189) | 本文提出了一种公平学习方法，该方法能够在可能带有歧视的数据集上进行训练，且能够在公平的测试数据集上表现良好，且该方法可在消除歧视的情况下使用，并在受保护群体之间取得平衡。 |

# 详细

[^1]: GSINA: 通过图Sinkhorn Attention改进图不变学习中的子图提取

    GSINA: Improving Subgraph Extraction for Graph Invariant Learning via Graph Sinkhorn Attention

    [https://arxiv.org/abs/2402.07191](https://arxiv.org/abs/2402.07191)

    本文提出了一种改进的图不变学习方法，通过稀疏性、软性和可微性原则来提取不变子图，从而提高图学习的泛化性能。

    

    图不变学习(GIL)是一种有效的方法，用于在不同分布变化下发现图数据与其标签之间的不变关系，以解决各种图学习任务。最近的GIL研究主要集中在从输入图中提取不变子图，作为规则化策略来提高图学习的泛化性能。然而，这些方法在获取不变子图方面也存在各种限制。本文分析了现有工作的缺点，并提出了提取不变子图的相应原则：1）稀疏性，以过滤掉变异特征；2）软性，以获得更广泛的解空间；和3）可微性，以进行端到端优化。为了在一次操作中满足这些原则，我们利用最优传输(OT)理论，并提出了一种新颖的图注意机制，称为图Sinkhorn Attention（G)

    Graph invariant learning (GIL) has been an effective approach to discovering the invariant relationships between graph data and its labels for different graph learning tasks under various distribution shifts. Many recent endeavors of GIL focus on extracting the invariant subgraph from the input graph for prediction as a regularization strategy to improve the generalization performance of graph learning. Despite their success, such methods also have various limitations in obtaining their invariant subgraphs. In this paper, we provide in-depth analyses of the drawbacks of existing works and propose corresponding principles of our invariant subgraph extraction: 1) the sparsity, to filter out the variant features, 2) the softness, for a broader solution space, and 3) the differentiability, for a soundly end-to-end optimization. To meet these principles in one shot, we leverage the Optimal Transport (OT) theory and propose a novel graph attention mechanism called Graph Sinkhorn Attention (G
    
[^2]: 关于AI问责政策的探索

    Towards an AI Accountability Policy. (arXiv:2307.13658v1 [cs.CY])

    [http://arxiv.org/abs/2307.13658](http://arxiv.org/abs/2307.13658)

    这份白皮书是对美国国家电信和信息管理局的“AI问责政策评论请求”的回应，提出了一组相互关联的AI问责政策建议。

    

    这份白皮书是对美国国家电信和信息管理局的“AI问责政策评论请求”作出的回应。在回答相关问题的关键句子末尾，提供了要求评论的问题编号的上标。该白皮书提出了一组相互关联的AI问责政策建议。

    This white paper is a response to the "AI Accountability Policy Request for Comments" by the National Telecommunications and Information Administration of the United States. The question numbers for which comments were requested are provided in superscripts at the end of key sentences answering the respective questions. The white paper offers a set of interconnected recommendations for an AI accountability policy.
    
[^3]: 使用变分自动编码器学习随机过程的最小表示

    Learning minimal representations of stochastic processes with variational autoencoders. (arXiv:2307.11608v1 [cond-mat.soft])

    [http://arxiv.org/abs/2307.11608](http://arxiv.org/abs/2307.11608)

    本文引入了一种无监督机器学习方法，使用变分自动编码器确定最小参数集，有效描述随机过程动力学，并生成能准确复制预期随机行为的新轨迹。

    

    随机过程在科学中有许多应用，因为它们广泛用于模拟各种自然现象。由于其固有的随机性和不确定性，它们很难进行表征。在这里，我们引入了一种无监督机器学习方法，用于确定有效描述随机过程动力学所需的最小参数集。我们的方法建立在扩展的β-变分自动编码器架构上。通过与典型扩散模型相对应的模拟数据集，我们展示了它在提取能准确描述这些动力学的最小相关参数方面的有效性。此外，该方法可以生成忠实复制预期随机行为的新轨迹。总体而言，我们的方法使得能够自动发现描述随机过程的未知参数，从而增进对各个领域中复杂现象的理解。

    Stochastic processes have found numerous applications in science, as they are broadly used to model a variety of natural phenomena. Due to their intrinsic randomness and uncertainty, they are however difficult to characterize. Here, we introduce an unsupervised machine learning approach to determine the minimal set of parameters required to effectively describe the dynamics of a stochastic process. Our method builds upon an extended $\beta$-variational autoencoder architecture. By means of simulated datasets corresponding to paradigmatic diffusion models, we showcase its effectiveness in extracting the minimal relevant parameters that accurately describe these dynamics. Furthermore, the method enables the generation of new trajectories that faithfully replicate the expected stochastic behavior. Overall, our approach enables for the autonomous discovery of unknown parameters describing stochastic processes, hence enhancing our comprehension of complex phenomena across various fields.
    
[^4]: 一种用于解决高维Committor问题的有限表达式方法

    A Finite Expression Method for Solving High-Dimensional Committor Problems. (arXiv:2306.12268v1 [math.NA])

    [http://arxiv.org/abs/2306.12268](http://arxiv.org/abs/2306.12268)

    本文提出了一种用于解决高维Committor问题的有限表达式方法(FEX)，该方法通过深度神经网络学习最优非线性函数和系数值，能够显著提高计算效果。

    

    转移路径理论（TPT）是一种数学框架，用于量化从选定的亚稳态$A$到$B$之间的稀有转移事件。TPT的核心是Committor函数，其描述了从相空间的任何起始点到达亚稳态$B$之前到达$A$的概率。计算出Committor之后，可以立即找到转换通道和转换速率。Committor是具有适当边界条件的反向Kolmogorov方程的解。然而，在高维情况下，由于需要网格化整个环境空间，解决Committor是一项具有挑战性的任务。在这项工作中，我们探索了有限表达式方法（FEX，Liang和Yang（2022））作为计算Committor的工具。FEX通过涉及一定数量的非线性函数和二进制算术运算的固定有限代数表达式来逼近Committor。最佳的非线性函数、二进制运算和数值系数值通过深度神经网络从训练数据中学习到。我们通过解决多个高维Committor问题，其中包括高达400个维度，展示了FEX的有效性，并且表明FEX显著优于传统的数值方法，如有限元方法和有限差分方法。

    Transition path theory (TPT) is a mathematical framework for quantifying rare transition events between a pair of selected metastable states $A$ and $B$. Central to TPT is the committor function, which describes the probability to hit the metastable state $B$ prior to $A$ from any given starting point of the phase space. Once the committor is computed, the transition channels and the transition rate can be readily found. The committor is the solution to the backward Kolmogorov equation with appropriate boundary conditions. However, solving it is a challenging task in high dimensions due to the need to mesh a whole region of the ambient space. In this work, we explore the finite expression method (FEX, Liang and Yang (2022)) as a tool for computing the committor. FEX approximates the committor by an algebraic expression involving a fixed finite number of nonlinear functions and binary arithmetic operations. The optimal nonlinear functions, the binary operations, and the numerical coeffi
    
[^5]: 基于预训练的影响因素研究医学图像分类解释性能的基准数据

    Benchmark data to study the influence of pre-training on explanation performance in MR image classification. (arXiv:2306.12150v1 [cs.CV])

    [http://arxiv.org/abs/2306.12150](http://arxiv.org/abs/2306.12150)

    本研究提出了一个MRI分类任务的基准数据集，用于评估不同模型的解释性能。实验结果表明，XAI方法并不一定比简单模型提供更好的解释，且CNN的解释能力取决于底层数据的复杂性和标签的质量。

    

    卷积神经网络（CNN）常常在医学预测任务中被成功地应用，通常与迁移学习相结合，在训练数据不足时能够提高性能。然而，由于CNN产生的模型高度复杂且通常不提供任何有关其预测机制的信息，这促使了“可解释性”人工智能（XAI）领域的研究。本文提出了一个基准数据集，用于在MRI分类任务中定量评估解释性能。通过这个基准数据集，我们可以了解迁移学习对解释质量的影响。实验结果表明，应用于基于迁移学习的CNN的流行XAI方法并不一定比简单模型提供更好的解释，并且CNN提供有意义解释的能力严重依赖于底层数据的复杂性和标签的质量。

    Convolutional Neural Networks (CNNs) are frequently and successfully used in medical prediction tasks. They are often used in combination with transfer learning, leading to improved performance when training data for the task are scarce. The resulting models are highly complex and typically do not provide any insight into their predictive mechanisms, motivating the field of 'explainable' artificial intelligence (XAI). However, previous studies have rarely quantitatively evaluated the 'explanation performance' of XAI methods against ground-truth data, and transfer learning and its influence on objective measures of explanation performance has not been investigated. Here, we propose a benchmark dataset that allows for quantifying explanation performance in a realistic magnetic resonance imaging (MRI) classification task. We employ this benchmark to understand the influence of transfer learning on the quality of explanations. Experimental results show that popular XAI methods applied to t
    
[^6]: 从数据中发现物理定律的有限表达方法

    Finite Expression Methods for Discovering Physical Laws from Data. (arXiv:2305.08342v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.08342](http://arxiv.org/abs/2305.08342)

    本文介绍了一种名为"有限表达法" (FEX) 的深度符号学习方法，通过学习动态数据中PDE解的导数，发现控制方程的解析表达式。相对于其他现有方法，FEX在多种问题上表现出更好的数值性能，包括时变的PDE问题和具有时变系数的非线性动力系统。

    

    非线性动力学是科学和工程领域中普遍存在的现象。然而，从有限数据中推导出描述非线性动力学的解析表达式仍然具有挑战性。本文将介绍一种新颖的深度符号学习方法，称为"有限表达法" (FEX)，通过学习动态数据中的偏微分方程（PDE）解的导数，利用FEX在包含有限集的解析表达式的函数空间中发现控制方程。我们的数值结果表明，相对于其他现有方法（如PDE-Net, SINDy, GP 和 SPL），我们的FEX在多种问题上表现出更好的数值性能，包括时变的PDE问题和具有时变系数的非线性动力系统。此外，结果突显了FEX的灵活性和鲁棒性。

    Nonlinear dynamics is a pervasive phenomenon observed in scientific and engineering disciplines. However, the task of deriving analytical expressions to describe nonlinear dynamics from limited data remains challenging. In this paper, we shall present a novel deep symbolic learning method called the "finite expression method" (FEX) to discover governing equations within a function space containing a finite set of analytic expressions, based on observed dynamic data. The key concept is to employ FEX to generate analytical expressions of the governing equations by learning the derivatives of partial differential equation (PDE) solutions through convolutions. Our numerical results demonstrate that our FEX surpasses other existing methods (such as PDE-Net, SINDy, GP, and SPL) in terms of numerical performance across a range of problems, including time-dependent PDE problems and nonlinear dynamical systems with time-varying coefficients. Moreover, the results highlight FEX's flexibility and
    
[^7]: 从带有歧视性质的训练数据中学习

    Learning from Discriminatory Training Data. (arXiv:1912.08189v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/1912.08189](http://arxiv.org/abs/1912.08189)

    本文提出了一种公平学习方法，该方法能够在可能带有歧视的数据集上进行训练，且能够在公平的测试数据集上表现良好，且该方法可在消除歧视的情况下使用，并在受保护群体之间取得平衡。

    

    监督学习系统是通过历史数据训练的，如果这些数据受到歧视性质的影响，那么该系统可能会在保护组中产生歧视。本文提出了公平学习的方法，即使在潜在的歧视性的数据集上训练，也将在公平的测试数据集上表现良好。这样的数据集转变为特定公平学习方法的应用方案。例如，消除直接歧视可以被表示为特定的数据集转变问题。对于这种情况，我们提出了一种学习方法，该方法在盲目训练包含直接加性歧视的数据集的同时，在公平数据集上可以证明最小化模型误差。该方法与现有的法律体系兼容，并通过在受保护群体之间取得平衡来解决广泛讨论的受保护群体交叉的问题。从技术上讲，该方法应用了概率干预，并具有因果和反事实公式。

    Supervised learning systems are trained using historical data and, if the data was tainted by discrimination, they may unintentionally learn to discriminate against protected groups. We propose that fair learning methods, despite training on potentially discriminatory datasets, shall perform well on fair test datasets. Such dataset shifts crystallize application scenarios for specific fair learning methods. For instance, the removal of direct discrimination can be represented as a particular dataset shift problem. For this scenario, we propose a learning method that provably minimizes model error on fair datasets, while blindly training on datasets poisoned with direct additive discrimination. The method is compatible with existing legal systems and provides a solution to the widely discussed issue of protected groups' intersectionality by striking a balance between the protected groups. Technically, the method applies probabilistic interventions, has causal and counterfactual formulat
    

