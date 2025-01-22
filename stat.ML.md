# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Global Safe Sequential Learning via Efficient Knowledge Transfer](https://arxiv.org/abs/2402.14402) | 提出了考虑转移安全的全局顺序学习方法，以加速安全学习，并通过预先计算源组件来减少额外的计算负载。 |
| [^2] | [Behind the Myth of Exploration in Policy Gradients](https://arxiv.org/abs/2402.00162) | 本论文提出了对政策梯度算法中探索项的新分析方法，区分了其平滑学习目标和增加梯度估计的两种不同作用。同时，详细讨论和实证了基于熵奖励的探索策略的局限性，并开辟了未来对这些策略设计和分析的研究方向。 |
| [^3] | [Analysis of tidal flows through the Strait of Gibraltar using Dynamic Mode Decomposition.](http://arxiv.org/abs/2311.01377) | 本研究利用动态模态分解分析直布罗陀海峡的潮流，揭示了其复杂的海洋亚中尺度特征以及物理机制，并提出了改进方法来增强分析的稳健性和准确性。 |
| [^4] | [PAC Neural Prediction Set Learning to Quantify the Uncertainty of Generative Language Models.](http://arxiv.org/abs/2307.09254) | 本文提出了一种使用神经网络来量化生成式语言模型不确定性的PAC神经预测集学习方法，通过在多种语言数据集和模型上的实验证明，相比于标准基准方法，我们的方法平均提高了63％的量化不确定性。 |
| [^5] | [On Neural Networks as Infinite Tree-Structured Probabilistic Graphical Models.](http://arxiv.org/abs/2305.17583) | 本文提出了一种创新方法，通过构建与神经网络完全对应的无限树状PGMs来解决深度神经网络(DNNs)缺乏PGMs的精确语义和明确定义的概率解释的问题。研究发现DNNs在前向传播时确实执行PGM推断的近似，这与现有研究不同，它阐明了DNNs对PGMs中的精确推理的更直接近似，潜在的好处包括改进DNNs的教学和解释，以及能够合并PGMs和DNNs的算法。 |
| [^6] | [Conditional Generative Modeling is All You Need for Marked Temporal Point Processes.](http://arxiv.org/abs/2305.12569) | 本文提出了一种从标记时间点过程中提取其统计直觉的事件生成模型，通过条件生成器以历史观察作为输入，生成可能发生的高质量随后事件。该模型具有高效、灵活和表示能力等方面的优势。 |
| [^7] | [Explainable Performance: Measuring the Driving Forces of Predictive Performance.](http://arxiv.org/abs/2212.05866) | XPER方法能衡量输入特征对模型预测性能的具体贡献，并可用于处理异质性问题，构建同质化个体群体，从而提高预测精度。 |

# 详细

[^1]: 全局安全顺序学习通过高效知识转移

    Global Safe Sequential Learning via Efficient Knowledge Transfer

    [https://arxiv.org/abs/2402.14402](https://arxiv.org/abs/2402.14402)

    提出了考虑转移安全的全局顺序学习方法，以加速安全学习，并通过预先计算源组件来减少额外的计算负载。

    

    arXiv:2402.14402v1 公告类型: 新摘要: 顺序学习方法例如主动学习和贝叶斯优化选择最具信息量的数据来学习一个任务。在许多医学或工程应用中，数据选择受先验未知的安全条件限制。一条有前途的安全学习方法利用高斯过程（GPs）来建模安全概率，并在具有较高安全置信度的区域中进行数据选择。然而，准确的安全建模需要先验知识或消耗数据。此外，安全置信度集中在给定的观测值周围，导致局部探索。由于在安全关键实验中通常存在可转移的源知识，我们提出考虑转移安全顺序学习来加速安全学习。我们进一步考虑先计算源组件，以减少引入源数据带来的额外计算负载。

    arXiv:2402.14402v1 Announce Type: new  Abstract: Sequential learning methods such as active learning and Bayesian optimization select the most informative data to learn about a task. In many medical or engineering applications, the data selection is constrained by a priori unknown safety conditions. A promissing line of safe learning methods utilize Gaussian processes (GPs) to model the safety probability and perform data selection in areas with high safety confidence. However, accurate safety modeling requires prior knowledge or consumes data. In addition, the safety confidence centers around the given observations which leads to local exploration. As transferable source knowledge is often available in safety critical experiments, we propose to consider transfer safe sequential learning to accelerate the learning of safety. We further consider a pre-computation of source components to reduce the additional computational load that is introduced by incorporating source data. In this pap
    
[^2]: 政策梯度探索背后的神话

    Behind the Myth of Exploration in Policy Gradients

    [https://arxiv.org/abs/2402.00162](https://arxiv.org/abs/2402.00162)

    本论文提出了对政策梯度算法中探索项的新分析方法，区分了其平滑学习目标和增加梯度估计的两种不同作用。同时，详细讨论和实证了基于熵奖励的探索策略的局限性，并开辟了未来对这些策略设计和分析的研究方向。

    

    政策梯度算法是解决具有连续状态和动作空间的控制问题的有效强化学习方法。为了计算接近最优的策略，在实践中必须在学习目标中包含探索项。尽管这些项的有效性通常通过对探索环境的内在需求进行证明，但我们提出了一种新的分析方法，区分了这些技术的两种不同含义。首先，它们使得平滑学习目标成为可能，并在保持全局最大值的同时消除了局部最优解。其次，它们修改了梯度估计，增加了随机参数更新最终提供最优策略的概率。基于这些效应，我们讨论并实证了基于熵奖励的探索策略，突出了其局限性，并为设计和分析这些策略的未来研究开辟了新方向。

    Policy-gradient algorithms are effective reinforcement learning methods for solving control problems with continuous state and action spaces. To compute near-optimal policies, it is essential in practice to include exploration terms in the learning objective. Although the effectiveness of these terms is usually justified by an intrinsic need to explore environments, we propose a novel analysis and distinguish two different implications of these techniques. First, they make it possible to smooth the learning objective and to eliminate local optima while preserving the global maximum. Second, they modify the gradient estimates, increasing the probability that the stochastic parameter update eventually provides an optimal policy. In light of these effects, we discuss and illustrate empirically exploration strategies based on entropy bonuses, highlighting their limitations and opening avenues for future works in the design and analysis of such strategies.
    
[^3]: 利用动态模态分解分析直布罗陀海峡的潮流

    Analysis of tidal flows through the Strait of Gibraltar using Dynamic Mode Decomposition. (arXiv:2311.01377v1 [math.DS])

    [http://arxiv.org/abs/2311.01377](http://arxiv.org/abs/2311.01377)

    本研究利用动态模态分解分析直布罗陀海峡的潮流，揭示了其复杂的海洋亚中尺度特征以及物理机制，并提出了改进方法来增强分析的稳健性和准确性。

    

    直布罗陀海峡是一个由地形、潮汐力、不稳定性和非线性水力过程影响的复杂海洋亚中尺度特征区域，所有这些都受非线性流体运动方程管控。本研究旨在通过3D MIT通用环流模型模拟，包括波浪、涡旋和旋回，揭示这些现象背后的物理机制。为了实现这一目标，我们采用动态模态分解（DMD）将模拟快照分解成Koopman模态，具有不同的指数增长/衰减率和振荡频率。我们的目标包括评估DMD在捕捉已知特征、揭示新元素、排名模态和探索降阶的效果。我们还引入了一些修改来增强DMD的稳健性、数值精度和特征值的稳健性。DMD分析产生了对流动模式、内波形成和直布罗陀海峡动力学的全面了解。

    The Strait of Gibraltar is a region characterized by intricate oceanic sub-mesoscale features, influenced by topography, tidal forces, instabilities, and nonlinear hydraulic processes, all governed by the nonlinear equations of fluid motion. In this study, we aim to uncover the underlying physics of these phenomena within 3D MIT general circulation model simulations, including waves, eddies, and gyres. To achieve this, we employ Dynamic Mode Decomposition (DMD) to break down simulation snapshots into Koopman modes, with distinct exponential growth/decay rates and oscillation frequencies. Our objectives encompass evaluating DMD's efficacy in capturing known features, unveiling new elements, ranking modes, and exploring order reduction. We also introduce modifications to enhance DMD's robustness, numerical accuracy, and robustness of eigenvalues. DMD analysis yields a comprehensive understanding of flow patterns, internal wave formation, and the dynamics of the Strait of Gibraltar, its m
    
[^4]: 用于量化生成式语言模型不确定性的PAC神经预测集学习

    PAC Neural Prediction Set Learning to Quantify the Uncertainty of Generative Language Models. (arXiv:2307.09254v1 [cs.LG])

    [http://arxiv.org/abs/2307.09254](http://arxiv.org/abs/2307.09254)

    本文提出了一种使用神经网络来量化生成式语言模型不确定性的PAC神经预测集学习方法，通过在多种语言数据集和模型上的实验证明，相比于标准基准方法，我们的方法平均提高了63％的量化不确定性。

    

    学习和量化模型的不确定性是增强模型可信度的关键任务。由于对生成虚构事实的担忧，最近兴起的生成式语言模型（GLM）特别强调可靠的不确定性量化的需求。本文提出了一种学习神经预测集模型的方法，该方法能够以可能近似正确（PAC）的方式量化GLM的不确定性。与现有的预测集模型通过标量值参数化不同，我们提出通过神经网络参数化预测集，实现更精确的不确定性量化，但仍满足PAC保证。通过在四种类型的语言数据集和六种类型的模型上展示，我们的方法相比标准基准方法平均提高了63％的量化不确定性。

    Uncertainty learning and quantification of models are crucial tasks to enhance the trustworthiness of the models. Importantly, the recent surge of generative language models (GLMs) emphasizes the need for reliable uncertainty quantification due to the concerns on generating hallucinated facts. In this paper, we propose to learn neural prediction set models that comes with the probably approximately correct (PAC) guarantee for quantifying the uncertainty of GLMs. Unlike existing prediction set models, which are parameterized by a scalar value, we propose to parameterize prediction sets via neural networks, which achieves more precise uncertainty quantification but still satisfies the PAC guarantee. We demonstrate the efficacy of our method on four types of language datasets and six types of models by showing that our method improves the quantified uncertainty by $63\%$ on average, compared to a standard baseline method.
    
[^5]: 关于神经网络作为无限树状概率图模型的论文研究

    On Neural Networks as Infinite Tree-Structured Probabilistic Graphical Models. (arXiv:2305.17583v1 [stat.ML])

    [http://arxiv.org/abs/2305.17583](http://arxiv.org/abs/2305.17583)

    本文提出了一种创新方法，通过构建与神经网络完全对应的无限树状PGMs来解决深度神经网络(DNNs)缺乏PGMs的精确语义和明确定义的概率解释的问题。研究发现DNNs在前向传播时确实执行PGM推断的近似，这与现有研究不同，它阐明了DNNs对PGMs中的精确推理的更直接近似，潜在的好处包括改进DNNs的教学和解释，以及能够合并PGMs和DNNs的算法。

    

    深度神经网络(DNNs)缺乏概率图模型(PGMs)的精确语义和明确定义的概率解释。本文提出了一种创新方法，通过构建与神经网络完全对应的无限树状PGMs来解决这个问题。我们的研究揭示了DNNs在前向传播期间确实执行PGM推断的近似，这与曾经的神经网络描述为核机器或无限大小的高斯过程的现有研究不同，它阐明了DNNs对PGMs中的精确推理的更直接近似。潜在的好处包括改进DNNs的教学和解释，以及能够合并PGMs和DNNs的算法。

    Deep neural networks (DNNs) lack the precise semantics and definitive probabilistic interpretation of probabilistic graphical models (PGMs). In this paper, we propose an innovative solution by constructing infinite tree-structured PGMs that correspond exactly to neural networks. Our research reveals that DNNs, during forward propagation, indeed perform approximations of PGM inference that are precise in this alternative PGM structure. Not only does our research complement existing studies that describe neural networks as kernel machines or infinite-sized Gaussian processes, it also elucidates a more direct approximation that DNNs make to exact inference in PGMs. Potential benefits include improved pedagogy and interpretation of DNNs, and algorithms that can merge the strengths of PGMs and DNNs.
    
[^6]: 有条件生成模型是标记时间点过程的必备工具。

    Conditional Generative Modeling is All You Need for Marked Temporal Point Processes. (arXiv:2305.12569v1 [stat.ML])

    [http://arxiv.org/abs/2305.12569](http://arxiv.org/abs/2305.12569)

    本文提出了一种从标记时间点过程中提取其统计直觉的事件生成模型，通过条件生成器以历史观察作为输入，生成可能发生的高质量随后事件。该模型具有高效、灵活和表示能力等方面的优势。

    

    近年来，生成建模的进步使得从上下文信息中生成高质量内容成为可能，但一个关键问题仍然存在：如何教模型知道何时生成内容？为了回答这个问题，本研究提出了一种新的事件生成模型，从标记时间点过程中提取其统计直觉，并提供了一个干净、灵活和计算效率高的解决方案，适用于涉及多维标记的各种应用。我们旨在捕捉点过程的分布而不需明确指定条件强度或概率密度。我们使用一个条件生成器，以事件历史为输入并生成在先前观察到的事件下，可能发生的高质量随后事件。所提出的框架提供了一系列利益，包括在学习模型和生成样本方面的异常效率以及相当大的表示能力来捕捉。

    Recent advancements in generative modeling have made it possible to generate high-quality content from context information, but a key question remains: how to teach models to know when to generate content? To answer this question, this study proposes a novel event generative model that draws its statistical intuition from marked temporal point processes, and offers a clean, flexible, and computationally efficient solution for a wide range of applications involving multi-dimensional marks. We aim to capture the distribution of the point process without explicitly specifying the conditional intensity or probability density. Instead, we use a conditional generator that takes the history of events as input and generates the high-quality subsequent event that is likely to occur given the prior observations. The proposed framework offers a host of benefits, including exceptional efficiency in learning the model and generating samples, as well as considerable representational power to capture
    
[^7]: 可解释的性能：衡量预测性能的驱动力

    Explainable Performance: Measuring the Driving Forces of Predictive Performance. (arXiv:2212.05866v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2212.05866](http://arxiv.org/abs/2212.05866)

    XPER方法能衡量输入特征对模型预测性能的具体贡献，并可用于处理异质性问题，构建同质化个体群体，从而提高预测精度。

    

    我们引入了XPER（eXplainable PERformance）方法来衡量输入特征对模型预测性能的具体贡献。我们的方法在理论上基于Shapley值，既不依赖于模型，也不依赖于性能度量。此外，XPER可在模型级别或个体级别实现。我们证明XPER具有标准解释性方法（SHAP）的特殊情况。在贷款违约预测应用中，我们展示了如何利用XPER处理异质性问题，并显著提高样本外性能。为此，我们通过基于个体XPER值对他们进行聚类来构建同质化的个体群体。我们发现估计群体特定的模型比一个模型适用于所有个体具有更高的预测精度。

    We introduce the XPER (eXplainable PERformance) methodology to measure the specific contribution of the input features to the predictive performance of a model. Our methodology is theoretically grounded on Shapley values and is both model-agnostic and performance metric-agnostic. Furthermore, XPER can be implemented either at the model level or at the individual level. We demonstrate that XPER has as a special case the standard explainability method in machine learning (SHAP). In a loan default forecasting application, we show how XPER can be used to deal with heterogeneity issues and significantly boost out-of-sample performance. To do so, we build homogeneous groups of individuals by clustering them based on their individual XPER values. We find that estimating group-specific models yields a much higher predictive accuracy than with a one-fits-all model.
    

