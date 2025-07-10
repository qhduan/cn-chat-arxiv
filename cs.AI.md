# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Semantic Augmentation in Images using Language](https://arxiv.org/abs/2404.02353) | 深度学习模型需要大规模标记数据集，本文提出利用生成图像增强数据集以改进模型跨领域泛化能力。 |
| [^2] | [Geometric Constraints in Deep Learning Frameworks: A Survey](https://arxiv.org/abs/2403.12431) | 本调查研究了几何约束和深度学习框架之间的重合部分，比较了深度估计等问题中集成在深度学习框架中的几何强制约束。 |
| [^3] | [Rethinking Class-incremental Learning in the Era of Large Pre-trained Models via Test-Time Adaptation.](http://arxiv.org/abs/2310.11482) | 本研究提出了一种名为“增量学习的测试时适应”的方法，通过在测试实例上进行微调，避免了在每个新任务上进行训练，从而在增量学习中实现了预训练模型的稳定性和可塑性的平衡。 |
| [^4] | [Stepwise functional refoundation of relational concept analysis.](http://arxiv.org/abs/2310.06441) | 逐步功能重构的关系概念分析（RCA）是形式概念分析的扩展，通过定义良构解决方案的空间和相关函数，解决了RCA在循环依赖数据上返回单一概念格家族的问题。 |
| [^5] | [Provably Efficient Learning in Partially Observable Contextual Bandit.](http://arxiv.org/abs/2308.03572) | 本文研究了在部分可观察情境轮盘赌中的转移学习问题，提出了一种通过优化问题识别行为和奖励因果效应的方法，并利用因果约束来改进轮盘赌算法。 |

# 详细

[^1]: 利用语言在图像中进行语义增强

    Semantic Augmentation in Images using Language

    [https://arxiv.org/abs/2404.02353](https://arxiv.org/abs/2404.02353)

    深度学习模型需要大规模标记数据集，本文提出利用生成图像增强数据集以改进模型跨领域泛化能力。

    

    深度学习模型需要非常庞大的标记数据集进行监督学习，缺乏这些数据集会导致过拟合并限制其泛化到现实世界示例的能力。最近扩散模型的进展使得能够基于文本输入生成逼真的图像。利用用于训练这些扩散模型的大规模数据集，我们提出一种利用生成的图像来增强现有数据集的技术。本文探讨了各种有效数据增强策略，以提高深度学习模型的跨领域泛化能力。

    arXiv:2404.02353v1 Announce Type: cross  Abstract: Deep Learning models are incredibly data-hungry and require very large labeled datasets for supervised learning. As a consequence, these models often suffer from overfitting, limiting their ability to generalize to real-world examples. Recent advancements in diffusion models have enabled the generation of photorealistic images based on textual inputs. Leveraging the substantial datasets used to train these diffusion models, we propose a technique to utilize generated images to augment existing datasets. This paper explores various strategies for effective data augmentation to improve the out-of-domain generalization capabilities of deep learning models.
    
[^2]: 深度学习框架中的几何约束：一项调查

    Geometric Constraints in Deep Learning Frameworks: A Survey

    [https://arxiv.org/abs/2403.12431](https://arxiv.org/abs/2403.12431)

    本调查研究了几何约束和深度学习框架之间的重合部分，比较了深度估计等问题中集成在深度学习框架中的几何强制约束。

    

    Stereophotogrammetry是一种新兴的场景理解技术。其起源可以追溯到至少19世纪，当时人们开始研究使用照片来测量世界的物理属性。自那时以来，已经探索了成千上万种方法。经典几何技术的Shape from Stereo建立在使用几何来定义场景和摄像机几何的约束，然后解决非线性方程组。更近期的工作采用了完全不同的方法，使用端到端的深度学习而没有明确建模几何。在这项调查中，我们探讨了基于几何和基于深度学习框架的重叠部分。我们比较和对比了集成到深度学习框架中用于深度估计或其他密切相关问题的几何强制约束。我们提出了一种新的分类法，用于描述现代深度学习中使用的普遍几何约束。

    arXiv:2403.12431v1 Announce Type: cross  Abstract: Stereophotogrammetry is an emerging technique of scene understanding. Its origins go back to at least the 1800s when people first started to investigate using photographs to measure the physical properties of the world. Since then, thousands of approaches have been explored. The classic geometric techniques of Shape from Stereo is built on using geometry to define constraints on scene and camera geometry and then solving the non-linear systems of equations. More recent work has taken an entirely different approach, using end-to-end deep learning without any attempt to explicitly model the geometry. In this survey, we explore the overlap for geometric-based and deep learning-based frameworks. We compare and contrast geometry enforcing constraints integrated into a deep learning framework for depth estimation or other closely related problems. We present a new taxonomy for prevalent geometry enforcing constraints used in modern deep lear
    
[^3]: 在大型预训练模型时代重新思考增量学习的测试时适应方法

    Rethinking Class-incremental Learning in the Era of Large Pre-trained Models via Test-Time Adaptation. (arXiv:2310.11482v1 [cs.CV])

    [http://arxiv.org/abs/2310.11482](http://arxiv.org/abs/2310.11482)

    本研究提出了一种名为“增量学习的测试时适应”的方法，通过在测试实例上进行微调，避免了在每个新任务上进行训练，从而在增量学习中实现了预训练模型的稳定性和可塑性的平衡。

    

    增量学习是一个具有挑战性的任务，涉及持续学习将类别划分到新任务中，同时不会遗忘先前学到的信息。大型预训练模型的出现加快了增量学习的进展，因为高度可传输的预训练模型表示使得在调整一小组参数时，与从头开始训练的传统增量学习方法相比，可以获得最先进的性能。然而，对每个任务进行反复微调会破坏预训练模型的丰富表示，并导致遗忘之前的任务。为了在增量学习中在预训练模型的稳定性和可塑性之间取得平衡，我们提出了一种新颖的方法，即通过直接在测试实例上进行测试时适应。具体而言，我们提出了“增量学习的测试时适应”（TTACIL），它首先在每个测试实例上对预训练模型的层归一化参数进行微调。

    Class-incremental learning (CIL) is a challenging task that involves continually learning to categorize classes into new tasks without forgetting previously learned information. The advent of the large pre-trained models (PTMs) has fast-tracked the progress in CIL due to the highly transferable PTM representations, where tuning a small set of parameters results in state-of-the-art performance when compared with the traditional CIL methods that are trained from scratch. However, repeated fine-tuning on each task destroys the rich representations of the PTMs and further leads to forgetting previous tasks. To strike a balance between the stability and plasticity of PTMs for CIL, we propose a novel perspective of eliminating training on every new task and instead performing test-time adaptation (TTA) directly on the test instances. Concretely, we propose "Test-Time Adaptation for Class-Incremental Learning" (TTACIL) that first fine-tunes Layer Norm parameters of the PTM on each test instan
    
[^4]: 逐步功能重构的关系概念分析

    Stepwise functional refoundation of relational concept analysis. (arXiv:2310.06441v1 [cs.AI])

    [http://arxiv.org/abs/2310.06441](http://arxiv.org/abs/2310.06441)

    逐步功能重构的关系概念分析（RCA）是形式概念分析的扩展，通过定义良构解决方案的空间和相关函数，解决了RCA在循环依赖数据上返回单一概念格家族的问题。

    

    关系概念分析（RCA）是形式概念分析的扩展，允许同时处理多个相关的语境。它被设计用于从数据中学习描述逻辑理论，并在各种应用中使用。关于RCA的一个令人困惑的观察是，尽管数据存在循环依赖关系，它返回一个单一的概念格家族，其他解决方案可能被认为是可接受的。RCA的语义以操作方式提供，对此问题并没有提供明确的解释。在本报告中，我们将这些可接受的解决方案定义为属于初始语境确定的空间的概念格家族（良构），不能扩展新属性（饱和），并且仅涉及该家族的概念（自支持）。我们通过定义良构解决方案的空间以及该空间上的两个函数（一个扩张函数和一个收缩函数），采用功能视图来描述RCA过程。我们展示了可接受的解决方案…

    Relational concept analysis (RCA) is an extension of formal concept analysis allowing to deal with several related contexts simultaneously. It has been designed for learning description logic theories from data and used within various applications. A puzzling observation about RCA is that it returns a single family of concept lattices although, when the data feature circular dependencies, other solutions may be considered acceptable. The semantics of RCA, provided in an operational way, does not shed light on this issue. In this report, we define these acceptable solutions as those families of concept lattices which belong to the space determined by the initial contexts (well-formed), cannot scale new attributes (saturated), and refer only to concepts of the family (self-supported). We adopt a functional view on the RCA process by defining the space of well-formed solutions and two functions on that space: one expansive and the other contractive. We show that the acceptable solutions a
    
[^5]: 在部分可观察情境轮盘赌中的可证效率学习

    Provably Efficient Learning in Partially Observable Contextual Bandit. (arXiv:2308.03572v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2308.03572](http://arxiv.org/abs/2308.03572)

    本文研究了在部分可观察情境轮盘赌中的转移学习问题，提出了一种通过优化问题识别行为和奖励因果效应的方法，并利用因果约束来改进轮盘赌算法。

    

    本文研究了在部分可观察情境轮盘赌中的转移学习问题，其中代理人仅有来自其他代理人的有限知识，并且对隐藏的混淆因素只有部分信息。我们将该问题转化为通过优化问题来识别或部分识别行为和奖励之间的因果效应。为了解决这些优化问题，我们将未知分布的原始功能约束离散化为线性约束，并通过顺序解线性规划来采样兼容的因果模型，以考虑估计误差得到因果约束。我们的采样算法为适当的采样分布提供了理想的收敛结果。然后，我们展示了如何将因果约束应用于改进经典的轮盘赌算法，并以行动集和函数空间规模为参考改变了遗憾值。值得注意的是，在允许我们处理一般情境分布的函数逼近任务中

    In this paper, we investigate transfer learning in partially observable contextual bandits, where agents have limited knowledge from other agents and partial information about hidden confounders. We first convert the problem to identifying or partially identifying causal effects between actions and rewards through optimization problems. To solve these optimization problems, we discretize the original functional constraints of unknown distributions into linear constraints, and sample compatible causal models via sequentially solving linear programmings to obtain causal bounds with the consideration of estimation error. Our sampling algorithms provide desirable convergence results for suitable sampling distributions. We then show how causal bounds can be applied to improving classical bandit algorithms and affect the regrets with respect to the size of action sets and function spaces. Notably, in the task with function approximation which allows us to handle general context distributions
    

