# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Physics-Based Causal Reasoning for Safe & Robust Next-Best Action Selection in Robot Manipulation Tasks](https://arxiv.org/abs/2403.14488) | 该论文提出了一个基于物理因果推理的框架，用于机器人在部分可观察的环境中进行概率推理，成功预测积木塔稳定性并选择下一最佳动作。 |
| [^2] | [FluoroSAM: A Language-aligned Foundation Model for X-ray Image Segmentation](https://arxiv.org/abs/2403.08059) | FluoroSAM是用于X光图像的分割的语言对齐基础模型，提供了一种在X光成像领域具有广泛适用性的自动图像分析工具。 |
| [^3] | [Do Concept Bottleneck Models Obey Locality?.](http://arxiv.org/abs/2401.01259) | 本文研究了概念瓶颈模型（CBMs）是否能够正确捕捉到概念之间的条件独立程度，通过分析对于概念局部性之外特征的变化如何影响概念的预测。 |
| [^4] | [A Survey on Explainable Reinforcement Learning: Concepts, Algorithms, Challenges.](http://arxiv.org/abs/2211.06665) | 该综述调查了可解释性强化学习方法，介绍了模型解释、奖励解释、状态解释和任务解释方法，并探讨了解释强化学习的概念、算法和挑战。 |

# 详细

[^1]: 基于物理学因果推理的机器人操作任务中安全稳健的下一最佳动作选择

    Physics-Based Causal Reasoning for Safe & Robust Next-Best Action Selection in Robot Manipulation Tasks

    [https://arxiv.org/abs/2403.14488](https://arxiv.org/abs/2403.14488)

    该论文提出了一个基于物理因果推理的框架，用于机器人在部分可观察的环境中进行概率推理，成功预测积木塔稳定性并选择下一最佳动作。

    

    安全高效的物体操作是许多真实世界机器人应用的关键推手。然而，这种挑战在于机器人操作必须对一系列传感器和执行器的不确定性具有稳健性。本文提出了一个基于物理知识和因果推理的框架，用于让机器人在部分可观察的环境中对候选动作进行概率推理，以完成一个积木堆叠任务。我们将刚体系统动力学的基于物理学的仿真与因果贝叶斯网络（CBN）结合起来，定义了机器人决策过程的因果生成概率模型。通过基于仿真的蒙特卡洛实验，我们展示了我们的框架成功地能够：(1) 高准确度地预测积木塔的稳定性（预测准确率：88.6%）；和，(2) 为积木堆叠任务选择一个近似的下一最佳动作，供整合的机器人系统执行，实现94.2%的任务成功率。

    arXiv:2403.14488v1 Announce Type: cross  Abstract: Safe and efficient object manipulation is a key enabler of many real-world robot applications. However, this is challenging because robot operation must be robust to a range of sensor and actuator uncertainties. In this paper, we present a physics-informed causal-inference-based framework for a robot to probabilistically reason about candidate actions in a block stacking task in a partially observable setting. We integrate a physics-based simulation of the rigid-body system dynamics with a causal Bayesian network (CBN) formulation to define a causal generative probabilistic model of the robot decision-making process. Using simulation-based Monte Carlo experiments, we demonstrate our framework's ability to successfully: (1) predict block tower stability with high accuracy (Pred Acc: 88.6%); and, (2) select an approximate next-best action for the block stacking task, for execution by an integrated robot system, achieving 94.2% task succe
    
[^2]: FluoroSAM: 用于X光图像分割的语言对齐基础模型

    FluoroSAM: A Language-aligned Foundation Model for X-ray Image Segmentation

    [https://arxiv.org/abs/2403.08059](https://arxiv.org/abs/2403.08059)

    FluoroSAM是用于X光图像的分割的语言对齐基础模型，提供了一种在X光成像领域具有广泛适用性的自动图像分析工具。

    

    自动X光图像分割将加速诊断和介入精准医学领域的研究和发展。先前的研究已经提出了适用于解决特定图像分析问题的特定任务模型，但这些模型的效用受限于特定任务领域，要拓展到更广泛的应用则需要额外的数据、标签和重新训练工作。最近，基础模型（FMs） - 训练在大量高度变化数据上的机器学习模型因此使得广泛适用性成为可能 - 已经成为自动图像分析的有希望的工具。现有的用于医学图像分析的FMs聚焦于对象被明显可见边界清晰定义的场景和模式，如内窥镜手术工具分割。相比之下，X光成像通常没有提供这种清晰的边界或结构先验。在X光图像形成期间，复杂的三维

    arXiv:2403.08059v1 Announce Type: cross  Abstract: Automated X-ray image segmentation would accelerate research and development in diagnostic and interventional precision medicine. Prior efforts have contributed task-specific models capable of solving specific image analysis problems, but the utility of these models is restricted to their particular task domain, and expanding to broader use requires additional data, labels, and retraining efforts. Recently, foundation models (FMs) -- machine learning models trained on large amounts of highly variable data thus enabling broad applicability -- have emerged as promising tools for automated image analysis. Existing FMs for medical image analysis focus on scenarios and modalities where objects are clearly defined by visually apparent boundaries, such as surgical tool segmentation in endoscopy. X-ray imaging, by contrast, does not generally offer such clearly delineated boundaries or structure priors. During X-ray image formation, complex 3D
    
[^3]: 概念瓶颈模型是否遵循局部性？

    Do Concept Bottleneck Models Obey Locality?. (arXiv:2401.01259v1 [cs.LG])

    [http://arxiv.org/abs/2401.01259](http://arxiv.org/abs/2401.01259)

    本文研究了概念瓶颈模型（CBMs）是否能够正确捕捉到概念之间的条件独立程度，通过分析对于概念局部性之外特征的变化如何影响概念的预测。

    

    概念基础学习通过解释其预测结果使用人可理解的概念，改善了深度学习模型的可解释性。在这种范式下训练的深度学习模型严重依赖于神经网络能够学习独立于其他概念的给定概念的存在或不存在。然而，最近的研究强烈暗示这种假设可能在概念瓶颈模型（CBMs）这一典型的基于概念的可解释架构中不能成立。本文中，我们研究了当这些概念既在空间上（通过它们的值完全由固定子集的特征定义）又在语义上（通过它们的值仅与预定义的固定子集的概念相关联）定位时，CBMs是否正确捕捉到概念之间的条件独立程度。为了理解局部性，我们分析了概念之外的特征变化对概念预测的影响。

    Concept-based learning improves a deep learning model's interpretability by explaining its predictions via human-understandable concepts. Deep learning models trained under this paradigm heavily rely on the assumption that neural networks can learn to predict the presence or absence of a given concept independently of other concepts. Recent work, however, strongly suggests that this assumption may fail to hold in Concept Bottleneck Models (CBMs), a quintessential family of concept-based interpretable architectures. In this paper, we investigate whether CBMs correctly capture the degree of conditional independence across concepts when such concepts are localised both spatially, by having their values entirely defined by a fixed subset of features, and semantically, by having their values correlated with only a fixed subset of predefined concepts. To understand locality, we analyse how changes to features outside of a concept's spatial or semantic locality impact concept predictions. Our
    
[^4]: 关于可解释性强化学习的综述：概念、算法和挑战

    A Survey on Explainable Reinforcement Learning: Concepts, Algorithms, Challenges. (arXiv:2211.06665v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.06665](http://arxiv.org/abs/2211.06665)

    该综述调查了可解释性强化学习方法，介绍了模型解释、奖励解释、状态解释和任务解释方法，并探讨了解释强化学习的概念、算法和挑战。

    

    强化学习是一种流行的机器学习范式，智能代理与环境进行交互以实现长期目标。在深度学习的复兴推动下，深度强化学习在各种复杂控制任务中取得了巨大成功。尽管取得了令人鼓舞的结果，基于深度神经网络的主干结构被普遍视为黑盒子，阻碍了从业者在安全性和可靠性至关重要的真实场景中信任和使用训练代理。为了缓解这个问题，大量的文献致力于揭示智能代理的内部工作原理，通过构建内在可解释性或事后可解释性。在本综述中，我们对现有的可解释性强化学习方法进行了全面的回顾，并引入了一个新的分类法，将先前的工作明确地分为模型解释、奖励解释、状态解释和任务解释方法。

    Reinforcement Learning (RL) is a popular machine learning paradigm where intelligent agents interact with the environment to fulfill a long-term goal. Driven by the resurgence of deep learning, Deep RL (DRL) has witnessed great success over a wide spectrum of complex control tasks. Despite the encouraging results achieved, the deep neural network-based backbone is widely deemed as a black box that impedes practitioners to trust and employ trained agents in realistic scenarios where high security and reliability are essential. To alleviate this issue, a large volume of literature devoted to shedding light on the inner workings of the intelligent agents has been proposed, by constructing intrinsic interpretability or post-hoc explainability. In this survey, we provide a comprehensive review of existing works on eXplainable RL (XRL) and introduce a new taxonomy where prior works are clearly categorized into model-explaining, reward-explaining, state-explaining, and task-explaining methods
    

