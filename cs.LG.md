# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Counterfactual Fairness through Transforming Data Orthogonal to Bias](https://arxiv.org/abs/2403.17852) | 提出了一种新颖的数据预处理算法，正交于偏见（OB），通过确保数据与敏感变量不相关，实现机器学习应用中的反事实公平性。 |
| [^2] | [Physics-Based Causal Reasoning for Safe & Robust Next-Best Action Selection in Robot Manipulation Tasks](https://arxiv.org/abs/2403.14488) | 该论文提出了一个基于物理因果推理的框架，用于机器人在部分可观察的环境中进行概率推理，成功预测积木塔稳定性并选择下一最佳动作。 |
| [^3] | [Dual-Channel Multiplex Graph Neural Networks for Recommendation](https://arxiv.org/abs/2403.11624) | 该研究提出了一种名为双通道多重图神经网络（DCMGNN）的新型推荐框架，能够有效解决现有推荐方法中存在的多通路关系行为模式建模和对目标关系影响忽略的问题。 |
| [^4] | [FluoroSAM: A Language-aligned Foundation Model for X-ray Image Segmentation](https://arxiv.org/abs/2403.08059) | FluoroSAM是用于X光图像的分割的语言对齐基础模型，提供了一种在X光成像领域具有广泛适用性的自动图像分析工具。 |
| [^5] | [Flexible infinite-width graph convolutional networks and the importance of representation learning](https://arxiv.org/abs/2402.06525) | 本文讨论了神经网络高斯过程（NNGP）在理论上的局限，提出图卷积深度内核机（graph convolutional deep kernel machine）来研究图分类任务中的表示学习问题。 |
| [^6] | [Do Concept Bottleneck Models Obey Locality?.](http://arxiv.org/abs/2401.01259) | 本文研究了概念瓶颈模型（CBMs）是否能够正确捕捉到概念之间的条件独立程度，通过分析对于概念局部性之外特征的变化如何影响概念的预测。 |
| [^7] | [SA-Solver: Stochastic Adams Solver for Fast Sampling of Diffusion Models.](http://arxiv.org/abs/2309.05019) | 本文提出了一种改进的高效随机亚当方法SA-Solver，用于解扩散随机微分方程以生成高质量的数据，实验结果显示它在少步采样中相较于现有最先进的方法有改进或可比的性能，并达到了SOTA FID分数。 |
| [^8] | [Variational quantum regression algorithm with encoded data structure.](http://arxiv.org/abs/2307.03334) | 本文介绍了一个具有编码数据结构的变分量子回归算法，在量子机器学习中具有模型解释性，并能有效地处理互连度较高的量子比特。算法通过压缩编码和数字-模拟门操作，大大提高了在噪声中尺度量子计算机上的运行时间复杂度。 |
| [^9] | [Efficient uniform approximation using Random Vector Functional Link networks.](http://arxiv.org/abs/2306.17501) | 本文研究了使用随机向量功能连接网络进行高效统一逼近的方法，证明了具有ReLU激活函数的RVFL网络可以逼近利普希茨连续函数，前提是隐藏层相对于输入维度是指数级宽度的。这是第一个证明了$L_\infty$逼近误差和高斯内部权重条件下的结果，给出了非渐进性的隐藏层节点数量下界。 |
| [^10] | [A Survey on Explainable Reinforcement Learning: Concepts, Algorithms, Challenges.](http://arxiv.org/abs/2211.06665) | 该综述调查了可解释性强化学习方法，介绍了模型解释、奖励解释、状态解释和任务解释方法，并探讨了解释强化学习的概念、算法和挑战。 |

# 详细

[^1]: 通过将数据转化为与偏见正交的方式实现反事实公平性

    Counterfactual Fairness through Transforming Data Orthogonal to Bias

    [https://arxiv.org/abs/2403.17852](https://arxiv.org/abs/2403.17852)

    提出了一种新颖的数据预处理算法，正交于偏见（OB），通过确保数据与敏感变量不相关，实现机器学习应用中的反事实公平性。

    

    机器学习模型在解决各个领域的复杂问题中展现出了卓越的能力。然而，这些模型有时可能表现出有偏见的决策，导致不同群体之间的待遇不平等。尽管公平性方面的研究已经很广泛，但多元连续敏感变量对决策结果的微妙影响尚未得到充分研究。我们引入了一种新颖的数据预处理算法，即正交于偏见（OB），旨在消除连续敏感变量的影响，从而促进机器学习应用中的反事实公平性。我们的方法基于结构因果模型（SCM）中联合正态分布的假设，证明了通过确保数据与敏感变量不相关即可实现反事实公平性。OB算法与模型无关，适用于多种机器学习应用。

    arXiv:2403.17852v1 Announce Type: new  Abstract: Machine learning models have shown exceptional prowess in solving complex issues across various domains. Nonetheless, these models can sometimes exhibit biased decision-making, leading to disparities in treatment across different groups. Despite the extensive research on fairness, the nuanced effects of multivariate and continuous sensitive variables on decision-making outcomes remain insufficiently studied. We introduce a novel data pre-processing algorithm, Orthogonal to Bias (OB), designed to remove the influence of a group of continuous sensitive variables, thereby facilitating counterfactual fairness in machine learning applications. Our approach is grounded in the assumption of a jointly normal distribution within a structural causal model (SCM), proving that counterfactual fairness can be achieved by ensuring the data is uncorrelated with sensitive variables. The OB algorithm is model-agnostic, catering to a wide array of machine 
    
[^2]: 基于物理学因果推理的机器人操作任务中安全稳健的下一最佳动作选择

    Physics-Based Causal Reasoning for Safe & Robust Next-Best Action Selection in Robot Manipulation Tasks

    [https://arxiv.org/abs/2403.14488](https://arxiv.org/abs/2403.14488)

    该论文提出了一个基于物理因果推理的框架，用于机器人在部分可观察的环境中进行概率推理，成功预测积木塔稳定性并选择下一最佳动作。

    

    安全高效的物体操作是许多真实世界机器人应用的关键推手。然而，这种挑战在于机器人操作必须对一系列传感器和执行器的不确定性具有稳健性。本文提出了一个基于物理知识和因果推理的框架，用于让机器人在部分可观察的环境中对候选动作进行概率推理，以完成一个积木堆叠任务。我们将刚体系统动力学的基于物理学的仿真与因果贝叶斯网络（CBN）结合起来，定义了机器人决策过程的因果生成概率模型。通过基于仿真的蒙特卡洛实验，我们展示了我们的框架成功地能够：(1) 高准确度地预测积木塔的稳定性（预测准确率：88.6%）；和，(2) 为积木堆叠任务选择一个近似的下一最佳动作，供整合的机器人系统执行，实现94.2%的任务成功率。

    arXiv:2403.14488v1 Announce Type: cross  Abstract: Safe and efficient object manipulation is a key enabler of many real-world robot applications. However, this is challenging because robot operation must be robust to a range of sensor and actuator uncertainties. In this paper, we present a physics-informed causal-inference-based framework for a robot to probabilistically reason about candidate actions in a block stacking task in a partially observable setting. We integrate a physics-based simulation of the rigid-body system dynamics with a causal Bayesian network (CBN) formulation to define a causal generative probabilistic model of the robot decision-making process. Using simulation-based Monte Carlo experiments, we demonstrate our framework's ability to successfully: (1) predict block tower stability with high accuracy (Pred Acc: 88.6%); and, (2) select an approximate next-best action for the block stacking task, for execution by an integrated robot system, achieving 94.2% task succe
    
[^3]: 双通道多重图神经网络用于推荐

    Dual-Channel Multiplex Graph Neural Networks for Recommendation

    [https://arxiv.org/abs/2403.11624](https://arxiv.org/abs/2403.11624)

    该研究提出了一种名为双通道多重图神经网络（DCMGNN）的新型推荐框架，能够有效解决现有推荐方法中存在的多通路关系行为模式建模和对目标关系影响忽略的问题。

    

    高效的推荐系统在准确捕捉反映个人偏好的用户和项目属性方面发挥着至关重要的作用。一些现有的推荐技术已经开始将重点转向在真实世界的推荐场景中对用户和项目之间的各种类型交互关系进行建模，例如在线购物平台上的点击、标记收藏和购买。然而，这些方法仍然面临两个重要的缺点：(1) 不足的建模和利用用户和项目之间多通路关系形成的各种行为模式对表示学习的影响，以及(2) 忽略了行为模式中不同关系对推荐系统场景中目标关系的影响。在本研究中，我们介绍了一种新颖的推荐框架，即双通道多重图神经网络（DCMGNN），该框架解决了上述挑战。

    arXiv:2403.11624v1 Announce Type: cross  Abstract: Efficient recommender systems play a crucial role in accurately capturing user and item attributes that mirror individual preferences. Some existing recommendation techniques have started to shift their focus towards modeling various types of interaction relations between users and items in real-world recommendation scenarios, such as clicks, marking favorites, and purchases on online shopping platforms. Nevertheless, these approaches still grapple with two significant shortcomings: (1) Insufficient modeling and exploitation of the impact of various behavior patterns formed by multiplex relations between users and items on representation learning, and (2) ignoring the effect of different relations in the behavior patterns on the target relation in recommender system scenarios. In this study, we introduce a novel recommendation framework, Dual-Channel Multiplex Graph Neural Network (DCMGNN), which addresses the aforementioned challenges
    
[^4]: FluoroSAM: 用于X光图像分割的语言对齐基础模型

    FluoroSAM: A Language-aligned Foundation Model for X-ray Image Segmentation

    [https://arxiv.org/abs/2403.08059](https://arxiv.org/abs/2403.08059)

    FluoroSAM是用于X光图像的分割的语言对齐基础模型，提供了一种在X光成像领域具有广泛适用性的自动图像分析工具。

    

    自动X光图像分割将加速诊断和介入精准医学领域的研究和发展。先前的研究已经提出了适用于解决特定图像分析问题的特定任务模型，但这些模型的效用受限于特定任务领域，要拓展到更广泛的应用则需要额外的数据、标签和重新训练工作。最近，基础模型（FMs） - 训练在大量高度变化数据上的机器学习模型因此使得广泛适用性成为可能 - 已经成为自动图像分析的有希望的工具。现有的用于医学图像分析的FMs聚焦于对象被明显可见边界清晰定义的场景和模式，如内窥镜手术工具分割。相比之下，X光成像通常没有提供这种清晰的边界或结构先验。在X光图像形成期间，复杂的三维

    arXiv:2403.08059v1 Announce Type: cross  Abstract: Automated X-ray image segmentation would accelerate research and development in diagnostic and interventional precision medicine. Prior efforts have contributed task-specific models capable of solving specific image analysis problems, but the utility of these models is restricted to their particular task domain, and expanding to broader use requires additional data, labels, and retraining efforts. Recently, foundation models (FMs) -- machine learning models trained on large amounts of highly variable data thus enabling broad applicability -- have emerged as promising tools for automated image analysis. Existing FMs for medical image analysis focus on scenarios and modalities where objects are clearly defined by visually apparent boundaries, such as surgical tool segmentation in endoscopy. X-ray imaging, by contrast, does not generally offer such clearly delineated boundaries or structure priors. During X-ray image formation, complex 3D
    
[^5]: 灵活的无限宽图卷积网络及表示学习的重要性

    Flexible infinite-width graph convolutional networks and the importance of representation learning

    [https://arxiv.org/abs/2402.06525](https://arxiv.org/abs/2402.06525)

    本文讨论了神经网络高斯过程（NNGP）在理论上的局限，提出图卷积深度内核机（graph convolutional deep kernel machine）来研究图分类任务中的表示学习问题。

    

    理解神经网络的一种常见理论方法是进行无限宽度限制，此时输出成为高斯过程（GP）分布。这被称为神经网络高斯过程（NNGP）。然而，NNGP内核是固定的，只能通过少量超参数进行调节，消除了任何表示学习的可能性。这与有限宽度的神经网络形成对比，后者通常被认为能够表现良好，正是因为它们能够学习表示。因此，简化神经网络以使其在理论上可处理的同时，NNGP可能会消除使其工作良好的因素（表示学习）。这激发了我们对一系列图分类任务中表示学习是否必要的理解。我们开发了一个精确的工具来完成这个任务，即图卷积深度内核机（graph convolutional deep kernel machine）。这与NNGP非常相似，因为它是无限宽度限制并使用内核，但它带有一个“旋钮”来控制表示学习的程度。

    A common theoretical approach to understanding neural networks is to take an infinite-width limit, at which point the outputs become Gaussian process (GP) distributed. This is known as a neural network Gaussian process (NNGP). However, the NNGP kernel is fixed, and tunable only through a small number of hyperparameters, eliminating any possibility of representation learning. This contrasts with finite-width NNs, which are often believed to perform well precisely because they are able to learn representations. Thus in simplifying NNs to make them theoretically tractable, NNGPs may eliminate precisely what makes them work well (representation learning). This motivated us to understand whether representation learning is necessary in a range of graph classification tasks. We develop a precise tool for this task, the graph convolutional deep kernel machine. This is very similar to an NNGP, in that it is an infinite width limit and uses kernels, but comes with a `knob' to control the amount 
    
[^6]: 概念瓶颈模型是否遵循局部性？

    Do Concept Bottleneck Models Obey Locality?. (arXiv:2401.01259v1 [cs.LG])

    [http://arxiv.org/abs/2401.01259](http://arxiv.org/abs/2401.01259)

    本文研究了概念瓶颈模型（CBMs）是否能够正确捕捉到概念之间的条件独立程度，通过分析对于概念局部性之外特征的变化如何影响概念的预测。

    

    概念基础学习通过解释其预测结果使用人可理解的概念，改善了深度学习模型的可解释性。在这种范式下训练的深度学习模型严重依赖于神经网络能够学习独立于其他概念的给定概念的存在或不存在。然而，最近的研究强烈暗示这种假设可能在概念瓶颈模型（CBMs）这一典型的基于概念的可解释架构中不能成立。本文中，我们研究了当这些概念既在空间上（通过它们的值完全由固定子集的特征定义）又在语义上（通过它们的值仅与预定义的固定子集的概念相关联）定位时，CBMs是否正确捕捉到概念之间的条件独立程度。为了理解局部性，我们分析了概念之外的特征变化对概念预测的影响。

    Concept-based learning improves a deep learning model's interpretability by explaining its predictions via human-understandable concepts. Deep learning models trained under this paradigm heavily rely on the assumption that neural networks can learn to predict the presence or absence of a given concept independently of other concepts. Recent work, however, strongly suggests that this assumption may fail to hold in Concept Bottleneck Models (CBMs), a quintessential family of concept-based interpretable architectures. In this paper, we investigate whether CBMs correctly capture the degree of conditional independence across concepts when such concepts are localised both spatially, by having their values entirely defined by a fixed subset of features, and semantically, by having their values correlated with only a fixed subset of predefined concepts. To understand locality, we analyse how changes to features outside of a concept's spatial or semantic locality impact concept predictions. Our
    
[^7]: SA-Solver：用于快速采样扩散模型的随机亚当求解器

    SA-Solver: Stochastic Adams Solver for Fast Sampling of Diffusion Models. (arXiv:2309.05019v1 [cs.LG])

    [http://arxiv.org/abs/2309.05019](http://arxiv.org/abs/2309.05019)

    本文提出了一种改进的高效随机亚当方法SA-Solver，用于解扩散随机微分方程以生成高质量的数据，实验结果显示它在少步采样中相较于现有最先进的方法有改进或可比的性能，并达到了SOTA FID分数。

    

    扩散概率模型在生成任务中取得了相当大的成功。由于从扩散概率模型中进行采样相当于解扩散随机微分方程或常微分方程，这是一项耗时的工作，因此提出了许多基于改进的微分方程求解器的快速采样方法。这些技术中的大部分方法都考虑解扩散常微分方程，因为它具有更好的效率。然而，随机采样可以在生成多样化和高质量数据方面提供额外的优势。在这项工作中，我们从两个方面进行了对随机采样的综合分析：方差控制的扩散随机微分方程和线性多步扩散随机微分方程求解器。基于我们的分析，我们提出了SA-Solver，它是一种改进的高效随机亚当方法，用于解扩散随机微分方程以生成高质量的数据。我们的实验结果显示，SA-Solver实现了：1）在少步采样中与现有最先进的采样方法相比，有改进或可比性能；2）SOTA FID分数。

    Diffusion Probabilistic Models (DPMs) have achieved considerable success in generation tasks. As sampling from DPMs is equivalent to solving diffusion SDE or ODE which is time-consuming, numerous fast sampling methods built upon improved differential equation solvers are proposed. The majority of such techniques consider solving the diffusion ODE due to its superior efficiency. However, stochastic sampling could offer additional advantages in generating diverse and high-quality data. In this work, we engage in a comprehensive analysis of stochastic sampling from two aspects: variance-controlled diffusion SDE and linear multi-step SDE solver. Based on our analysis, we propose SA-Solver, which is an improved efficient stochastic Adams method for solving diffusion SDE to generate data with high quality. Our experiments show that SA-Solver achieves: 1) improved or comparable performance compared with the existing state-of-the-art sampling methods for few-step sampling; 2) SOTA FID scores o
    
[^8]: 具有编码数据结构的变分量子回归算法

    Variational quantum regression algorithm with encoded data structure. (arXiv:2307.03334v1 [quant-ph])

    [http://arxiv.org/abs/2307.03334](http://arxiv.org/abs/2307.03334)

    本文介绍了一个具有编码数据结构的变分量子回归算法，在量子机器学习中具有模型解释性，并能有效地处理互连度较高的量子比特。算法通过压缩编码和数字-模拟门操作，大大提高了在噪声中尺度量子计算机上的运行时间复杂度。

    

    变分量子算法(VQAs)被广泛应用于解决实际问题，如组合优化、量子化学模拟、量子机器学习和噪声量子计算机上的量子错误纠正。对于变分量子机器学习，尚未开发出将模型解释性内嵌到算法中的变分算法。本文构建了一个量子回归算法，并确定了变分参数与学习回归系数之间的直接关系，同时采用了将数据直接编码为反映经典数据表结构的量子幅度的电路。该算法特别适用于互连度较高的量子比特。通过压缩编码和数字-模拟门操作，运行时间复杂度在数据输入量编码的情况下对数级更有优势，显著提升了噪声中尺度量子计算机的性能。

    Variational quantum algorithms (VQAs) prevail to solve practical problems such as combinatorial optimization, quantum chemistry simulation, quantum machine learning, and quantum error correction on noisy quantum computers. For variational quantum machine learning, a variational algorithm with model interpretability built into the algorithm is yet to be exploited. In this paper, we construct a quantum regression algorithm and identify the direct relation of variational parameters to learned regression coefficients, while employing a circuit that directly encodes the data in quantum amplitudes reflecting the structure of the classical data table. The algorithm is particularly suitable for well-connected qubits. With compressed encoding and digital-analog gate operation, the run time complexity is logarithmically more advantageous than that for digital 2-local gate native hardware with the number of data entries encoded, a decent improvement in noisy intermediate-scale quantum computers a
    
[^9]: 使用随机向量功能连接网络进行高效统一逼近

    Efficient uniform approximation using Random Vector Functional Link networks. (arXiv:2306.17501v1 [stat.ML])

    [http://arxiv.org/abs/2306.17501](http://arxiv.org/abs/2306.17501)

    本文研究了使用随机向量功能连接网络进行高效统一逼近的方法，证明了具有ReLU激活函数的RVFL网络可以逼近利普希茨连续函数，前提是隐藏层相对于输入维度是指数级宽度的。这是第一个证明了$L_\infty$逼近误差和高斯内部权重条件下的结果，给出了非渐进性的隐藏层节点数量下界。

    

    随机向量功能连接(RVFL)网络是一个具有随机内部权重和偏置的二层神经网络。由于这种架构只需要学习外部权重，学习过程可以简化为线性优化任务，从而避免了非凸优化问题的困扰。在本文中，我们证明了具有ReLU激活函数的RVFL网络可以逼近利普希茨连续函数，前提是其隐藏层相对于输入维度是指数级宽度的。尽管之前已经证明了以$L_2$方式可以实现这样的逼近，但我们证明了在$L_\infty$逼近误差和高斯内部权重情况下的可行性。据我们所知，这是第一个这样的结果。我们给出了非渐进性的隐藏层节点数量的下界，取决于目标函数的利普希茨常数、期望的准确度和输入维度等因素。我们的证明方法根植于概率论。

    A Random Vector Functional Link (RVFL) network is a depth-2 neural network with random inner weights and biases. As only the outer weights of such architectures need to be learned, the learning process boils down to a linear optimization task, allowing one to sidestep the pitfalls of nonconvex optimization problems. In this paper, we prove that an RVFL with ReLU activation functions can approximate Lipschitz continuous functions provided its hidden layer is exponentially wide in the input dimension. Although it has been established before that such approximation can be achieved in $L_2$ sense, we prove it for $L_\infty$ approximation error and Gaussian inner weights. To the best of our knowledge, our result is the first of this kind. We give a nonasymptotic lower bound for the number of hidden layer nodes, depending on, among other things, the Lipschitz constant of the target function, the desired accuracy, and the input dimension. Our method of proof is rooted in probability theory an
    
[^10]: 关于可解释性强化学习的综述：概念、算法和挑战

    A Survey on Explainable Reinforcement Learning: Concepts, Algorithms, Challenges. (arXiv:2211.06665v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.06665](http://arxiv.org/abs/2211.06665)

    该综述调查了可解释性强化学习方法，介绍了模型解释、奖励解释、状态解释和任务解释方法，并探讨了解释强化学习的概念、算法和挑战。

    

    强化学习是一种流行的机器学习范式，智能代理与环境进行交互以实现长期目标。在深度学习的复兴推动下，深度强化学习在各种复杂控制任务中取得了巨大成功。尽管取得了令人鼓舞的结果，基于深度神经网络的主干结构被普遍视为黑盒子，阻碍了从业者在安全性和可靠性至关重要的真实场景中信任和使用训练代理。为了缓解这个问题，大量的文献致力于揭示智能代理的内部工作原理，通过构建内在可解释性或事后可解释性。在本综述中，我们对现有的可解释性强化学习方法进行了全面的回顾，并引入了一个新的分类法，将先前的工作明确地分为模型解释、奖励解释、状态解释和任务解释方法。

    Reinforcement Learning (RL) is a popular machine learning paradigm where intelligent agents interact with the environment to fulfill a long-term goal. Driven by the resurgence of deep learning, Deep RL (DRL) has witnessed great success over a wide spectrum of complex control tasks. Despite the encouraging results achieved, the deep neural network-based backbone is widely deemed as a black box that impedes practitioners to trust and employ trained agents in realistic scenarios where high security and reliability are essential. To alleviate this issue, a large volume of literature devoted to shedding light on the inner workings of the intelligent agents has been proposed, by constructing intrinsic interpretability or post-hoc explainability. In this survey, we provide a comprehensive review of existing works on eXplainable RL (XRL) and introduce a new taxonomy where prior works are clearly categorized into model-explaining, reward-explaining, state-explaining, and task-explaining methods
    

