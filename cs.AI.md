# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Geometry-induced Implicit Regularization in Deep ReLU Neural Networks](https://arxiv.org/abs/2402.08269) | 通过研究参数变化时输出集合的几何特征，我们发现在深度ReLU神经网络的优化过程中存在几何引导的隐式正则化现象。 |
| [^2] | [On the Convergence Rate of the Stochastic Gradient Descent (SGD) and application to a modified policy gradient for the Multi Armed Bandit](https://arxiv.org/abs/2402.06388) | 该论文证明了当学习速率按照逆时间衰减规则时，随机梯度下降（SGD）的收敛速度，并应用于修改的带有L2正则化的策略梯度多臂赌博机（MAB）的收敛性分析。 |
| [^3] | [Beyond Expectations: Learning with Stochastic Dominance Made Practical](https://arxiv.org/abs/2402.02698) | 这项工作首次尝试建立了一个随机优势学习的通用框架，并推广了随机优势的概念以使其能够在任意两个随机变量之间进行比较。同时，我们还开发了一种有效的计算方法来处理连续性评估的问题。 |
| [^4] | [A Survey on 3D Skeleton Based Person Re-Identification: Approaches, Designs, Challenges, and Future Directions.](http://arxiv.org/abs/2401.15296) | 本文通过对当前基于3D骨架的人员再识别方法、模型设计、挑战和未来方向的系统调研，填补了相关研究总结的空白。 |
| [^5] | [HCVP: Leveraging Hierarchical Contrastive Visual Prompt for Domain Generalization.](http://arxiv.org/abs/2401.09716) | HCVP是一种基于层次对比视觉提示的领域泛化方法，通过引导模型将不变特征与特定特征分离，提高了泛化性能。 |
| [^6] | [GRACE: Discriminator-Guided Chain-of-Thought Reasoning.](http://arxiv.org/abs/2305.14934) | GRACE是一种判别器引导的思维链推理的逐步解码方法，通过使用一个正确性判别器来评分下一步候选，解决了语言模型在多步推理中容易得到错误答案的问题。在多个数学和符号推理任务中，GRACE相较于其他方法在性能上有明显的提升。 |
| [^7] | [On Pitfalls of $\textit{RemOve-And-Retrain}$: Data Processing Inequality Perspective.](http://arxiv.org/abs/2304.13836) | 本论文评估了RemOve-And-Retrain（ROAR）协议的可靠性。研究结果表明，ROAR基准测试中的属性可能有更少的有关决策的重要信息，这种偏差称为毛糙度偏差，并提醒人们不要在ROAR指标上进行盲目的依赖。 |

# 详细

[^1]: 深度ReLU神经网络中的几何引导隐式正则化

    Geometry-induced Implicit Regularization in Deep ReLU Neural Networks

    [https://arxiv.org/abs/2402.08269](https://arxiv.org/abs/2402.08269)

    通过研究参数变化时输出集合的几何特征，我们发现在深度ReLU神经网络的优化过程中存在几何引导的隐式正则化现象。

    

    众所周知，具有比训练样本更多参数的神经网络不会过拟合。隐式正则化现象在优化过程中出现，对“好”的网络有利。因此，如果我们不考虑所有可能的网络，而只考虑“好”的网络，参数数量就不是一个足够衡量复杂性的指标。为了更好地理解在优化过程中哪些网络受到青睐，我们研究了参数变化时输出集合的几何特征。当输入固定时，我们证明了这个集合的维度会发生变化，并且局部维度，即批次功能维度，几乎总是由隐藏层中的激活模式决定。我们证明了批次功能维度对网络参数化的对称性（神经元排列和正向缩放）是不变的。实证上，我们证实了在优化过程中批次功能维度会下降。因此，优化过程具有隐式正则化的效果。

    It is well known that neural networks with many more parameters than training examples do not overfit. Implicit regularization phenomena, which are still not well understood, occur during optimization and 'good' networks are favored. Thus the number of parameters is not an adequate measure of complexity if we do not consider all possible networks but only the 'good' ones. To better understand which networks are favored during optimization, we study the geometry of the output set as parameters vary. When the inputs are fixed, we prove that the dimension of this set changes and that the local dimension, called batch functional dimension, is almost surely determined by the activation patterns in the hidden layers. We prove that the batch functional dimension is invariant to the symmetries of the network parameterization: neuron permutations and positive rescalings. Empirically, we establish that the batch functional dimension decreases during optimization. As a consequence, optimization l
    
[^2]: 关于随机梯度下降（SGD）的收敛速度及其在修改的多臂赌博机上的策略梯度应用

    On the Convergence Rate of the Stochastic Gradient Descent (SGD) and application to a modified policy gradient for the Multi Armed Bandit

    [https://arxiv.org/abs/2402.06388](https://arxiv.org/abs/2402.06388)

    该论文证明了当学习速率按照逆时间衰减规则时，随机梯度下降（SGD）的收敛速度，并应用于修改的带有L2正则化的策略梯度多臂赌博机（MAB）的收敛性分析。

    

    我们提出了一个自包含的证明，证明了当学习速率遵循逆时间衰减规则时，随机梯度下降（SGD）的收敛速度；接下来，我们将这些结果应用于带有L2正则化的修改的策略梯度多臂赌博机（MAB）的收敛性分析。

    We present a self-contained proof of the convergence rate of the Stochastic Gradient Descent (SGD) when the learning rate follows an inverse time decays schedule; we next apply the results to the convergence of a modified form of policy gradient Multi-Armed Bandit (MAB) with $L2$ regularization.
    
[^3]: 超越期望: 现实中实现随机优势学习

    Beyond Expectations: Learning with Stochastic Dominance Made Practical

    [https://arxiv.org/abs/2402.02698](https://arxiv.org/abs/2402.02698)

    这项工作首次尝试建立了一个随机优势学习的通用框架，并推广了随机优势的概念以使其能够在任意两个随机变量之间进行比较。同时，我们还开发了一种有效的计算方法来处理连续性评估的问题。

    

    随机优势模型对决策时具有风险厌恶偏好的不确定结果进行建模，相比于仅仅依赖期望值，自然地捕捉了底层不确定性的内在结构。尽管在理论上具有吸引力，但随机优势在机器学习中的应用却很少，主要是由于以下挑战：$\textbf{i)}$ 随机优势的原始概念仅提供了$\textit{部分序}$，因此不能作为最优性准则；和 $\textbf{ii)}$ 由于评估随机优势的连续性本质，目前还缺乏高效的计算方法。在这项工作中，我们首次尝试建立一个与随机优势学习相关的通用框架。我们首先将随机优势概念推广，使得任意两个随机变量之间的比较成为可能。接下来我们开发了一个有效的计算方法，以解决评估随机优势的连续性问题。

    Stochastic dominance models risk-averse preferences for decision making with uncertain outcomes, which naturally captures the intrinsic structure of the underlying uncertainty, in contrast to simply resorting to the expectations. Despite theoretically appealing, the application of stochastic dominance in machine learning has been scarce, due to the following challenges: $\textbf{i)}$, the original concept of stochastic dominance only provides a $\textit{partial order}$, therefore, is not amenable to serve as an optimality criterion; and $\textbf{ii)}$, an efficient computational recipe remains lacking due to the continuum nature of evaluating stochastic dominance.%, which barriers its application for machine learning.   In this work, we make the first attempt towards establishing a general framework of learning with stochastic dominance. We first generalize the stochastic dominance concept to enable feasible comparisons between any arbitrary pair of random variables. We next develop a 
    
[^4]: 基于3D骨架的人员再识别：方法、设计、挑战和未来方向的综述

    A Survey on 3D Skeleton Based Person Re-Identification: Approaches, Designs, Challenges, and Future Directions. (arXiv:2401.15296v1 [cs.CV])

    [http://arxiv.org/abs/2401.15296](http://arxiv.org/abs/2401.15296)

    本文通过对当前基于3D骨架的人员再识别方法、模型设计、挑战和未来方向的系统调研，填补了相关研究总结的空白。

    

    通过3D骨架进行人员再识别是一个重要的新兴研究领域，引起了模式识别社区的极大兴趣。近年来，针对骨架建模和特征学习中突出问题，已经提出了许多具有独特优势的基于3D骨架的人员再识别（SRID）方法。尽管近年来取得了一些进展，但据我们所知，目前还没有对这些研究及其挑战进行综合总结。因此，本文通过对当前SRID方法、模型设计、挑战和未来方向的系统调研，试图填补这一空白。具体而言，我们首先定义了SRID问题，并提出了一个SRID研究的分类体系，总结了常用的基准数据集、常用的模型架构，并对不同方法的特点进行了分析评价。然后，我们详细阐述了SRID模型的设计原则。

    Person re-identification via 3D skeletons is an important emerging research area that triggers great interest in the pattern recognition community. With distinctive advantages for many application scenarios, a great diversity of 3D skeleton based person re-identification (SRID) methods have been proposed in recent years, effectively addressing prominent problems in skeleton modeling and feature learning. Despite recent advances, to the best of our knowledge, little effort has been made to comprehensively summarize these studies and their challenges. In this paper, we attempt to fill this gap by providing a systematic survey on current SRID approaches, model designs, challenges, and future directions. Specifically, we first formulate the SRID problem, and propose a taxonomy of SRID research with a summary of benchmark datasets, commonly-used model architectures, and an analytical review of different methods' characteristics. Then, we elaborate on the design principles of SRID models fro
    
[^5]: HCVP: 基于层次对比视觉提示的领域泛化方法

    HCVP: Leveraging Hierarchical Contrastive Visual Prompt for Domain Generalization. (arXiv:2401.09716v1 [cs.CV])

    [http://arxiv.org/abs/2401.09716](http://arxiv.org/abs/2401.09716)

    HCVP是一种基于层次对比视觉提示的领域泛化方法，通过引导模型将不变特征与特定特征分离，提高了泛化性能。

    

    领域泛化（DG）旨在通过学习不变特征来创建在未知场景中表现出色的机器学习模型。然而，在DG中，将模型限制在固定结构或统一参数化中以包含不变特征的主流实践可能会不可避免地融合特定方面。这种方法难以对领域间变化进行细微区分，可能对某些领域存在偏见，从而阻碍了对域不变特征的精确学习。鉴于此，我们引入了一种新方法，旨在为模型提供领域级和任务特定的特征。该方法旨在更有效地引导模型将不变特征与特定特征分离，从而提高泛化性能。在领域泛化范式中，借鉴了视觉提示的新趋势，我们的工作引入了一种新颖的“HCVP”（层次对比视觉提示）方法。

    Domain Generalization (DG) endeavors to create machine learning models that excel in unseen scenarios by learning invariant features. In DG, the prevalent practice of constraining models to a fixed structure or uniform parameterization to encapsulate invariant features can inadvertently blend specific aspects. Such an approach struggles with nuanced differentiation of inter-domain variations and may exhibit bias towards certain domains, hindering the precise learning of domain-invariant features. Recognizing this, we introduce a novel method designed to supplement the model with domain-level and task-specific characteristics. This approach aims to guide the model in more effectively separating invariant features from specific characteristics, thereby boosting the generalization. Building on the emerging trend of visual prompts in the DG paradigm, our work introduces the novel \textbf{H}ierarchical \textbf{C}ontrastive \textbf{V}isual \textbf{P}rompt (HCVP) methodology. This represents 
    
[^6]: GRACE: 判别器引导的思维链推理

    GRACE: Discriminator-Guided Chain-of-Thought Reasoning. (arXiv:2305.14934v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.14934](http://arxiv.org/abs/2305.14934)

    GRACE是一种判别器引导的思维链推理的逐步解码方法，通过使用一个正确性判别器来评分下一步候选，解决了语言模型在多步推理中容易得到错误答案的问题。在多个数学和符号推理任务中，GRACE相较于其他方法在性能上有明显的提升。

    

    在多步推理的背景下，例如使用思维链，语言模型往往会对错误的步骤分配较高的可能性。因此，优化解决方案可能性的解码策略往往会产生错误的解决方案。为了解决这个问题，我们提出了一种称为GRACE的引导思维链推理的逐步解码方法，该方法通过一个正确性判别器训练来引导解码过程产生正确的推理步骤。GRACE使用一个在正确和错误步骤上进行对比损失训练的判别器，该判别器在解码过程中基于正确性对下一步候选进行评分。重要的是，GRACE只需要从语言模型中采样，而不需要进行语言模型的训练或微调。我们使用FLAN-T5和LLaMA系列的模型，对四个数学和两个符号推理任务进行了GRACE的评估，在大多数设置中，与贪婪解码、验证器和自一致性相比，GRACE展现出了显著的性能提升。

    In the context of multi-step reasoning, e.g., with chain-of-thought, language models (LMs) can easily assign a high likelihood to incorrect steps. As a result, decoding strategies that optimize for solution likelihood often yield incorrect solutions. To address this issue, we propose Guiding chain-of-thought ReAsoning with a CorrectnEss Discriminator (GRACE), a stepwise decoding approach that steers the decoding process towards producing correct reasoning steps. GRACE employs a discriminator trained with a contrastive loss over correct and incorrect steps, which is used during decoding to score next-step candidates based on their correctness. Importantly, GRACE only requires sampling from the LM, without the need for LM training or fine-tuning. Using models from FLAN-T5 and LLaMA families, we evaluate GRACE over four math and two symbolic reasoning tasks, where it exhibits substantial performance gains compared to greedy decoding, verifiers, and self-consistency in most settings. When 
    
[^7]: 论RemOve-And-Retrain的陷阱：数据处理不等式的视角

    On Pitfalls of $\textit{RemOve-And-Retrain}$: Data Processing Inequality Perspective. (arXiv:2304.13836v1 [cs.LG])

    [http://arxiv.org/abs/2304.13836](http://arxiv.org/abs/2304.13836)

    本论文评估了RemOve-And-Retrain（ROAR）协议的可靠性。研究结果表明，ROAR基准测试中的属性可能有更少的有关决策的重要信息，这种偏差称为毛糙度偏差，并提醒人们不要在ROAR指标上进行盲目的依赖。

    

    本文评估了RemOve-And-Retrain（ROAR）协议的可靠性，该协议用于测量特征重要性估计的性能。我们从理论背景和实证实验中发现，具有较少有关决策功能的信息的属性在ROAR基准测试中表现更好，与ROAR的原始目的相矛盾。这种现象也出现在最近提出的变体RemOve-And-Debias（ROAD）中，我们提出了ROAR归因度量中毛糙度偏差的一致趋势。我们的结果提醒人们不要盲目依赖ROAR的性能评估指标。

    This paper assesses the reliability of the RemOve-And-Retrain (ROAR) protocol, which is used to measure the performance of feature importance estimates. Our findings from the theoretical background and empirical experiments indicate that attributions that possess less information about the decision function can perform better in ROAR benchmarks, conflicting with the original purpose of ROAR. This phenomenon is also observed in the recently proposed variant RemOve-And-Debias (ROAD), and we propose a consistent trend of blurriness bias in ROAR attribution metrics. Our results caution against uncritical reliance on ROAR metrics.
    

