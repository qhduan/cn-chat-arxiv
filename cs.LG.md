# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Simplicity Bias of Transformers to Learn Low Sensitivity Functions](https://arxiv.org/abs/2403.06925) | Transformers在不同数据模态上具有低敏感性，这种简单性偏差有助于解释其在视觉和语言任务中的优越性能。 |
| [^2] | [Opening Cabinets and Drawers in the Real World using a Commodity Mobile Manipulator](https://arxiv.org/abs/2402.17767) | 实现了一个端到端系统，使商品移动操作器成功在以前未见的真实世界环境中打开橱柜和抽屉，感知误差是主要挑战。 |
| [^3] | [For Better or For Worse? Learning Minimum Variance Features With Label Augmentation](https://arxiv.org/abs/2402.06855) | 本研究分析了标签增强方法中标签增强的作用。研究证明，在线性可分数据上使用标签增强训练的线性模型只能学习到最小方差特征，而标准训练可以学习到更高方差特征。此外，标签平滑和Mixup对于训练数据的对抗扰动可能不太鲁棒。 |
| [^4] | [Sym-Q: Adaptive Symbolic Regression via Sequential Decision-Making](https://arxiv.org/abs/2402.05306) | Sym-Q是一个基于强化学习的模型，通过将符号回归重新定义为顺序决策任务来解决现有模型在泛化性和适应性方面的挑战。通过利用监督演示和奖励信号，Sym-Q能够根据拟合精度的质量改进表达式。 |
| [^5] | [Efficient Solvers for Partial Gromov-Wasserstein](https://arxiv.org/abs/2402.03664) | 本文提出了两个基于Frank-Wolfe算法的新的高效求解器来解决偏差Gromov-Wasserstein问题，并且证明了PGW问题构成了度量测度空间的度量。 |
| [^6] | [Dual Interior-Point Optimization Learning](https://arxiv.org/abs/2402.02596) | 本文介绍了双内点优化学习和双超梯度学习两种方法，用于学习带有有界变量的参数线性规划的对偶可行解。这些方法通过预测约束对应的对偶变量，确保对偶可行性，并且能够提供高保真度的对偶可行解和有效的对偶界限。 |
| [^7] | [The Developmental Landscape of In-Context Learning](https://arxiv.org/abs/2402.02364) | 在transformers模型中，我们展示了在上下文学习中的离散发展阶段，并引入了两种方法来检测这些阶段的关键里程碑。我们使用行为和结构度量验证了这些方法的有效性。 |
| [^8] | [Vision-LLMs Can Fool Themselves with Self-Generated Typographic Attacks](https://arxiv.org/abs/2402.00626) | 这项研究深入研究了大规模视觉语言模型（LVLM）对于自动生成的排版攻击的易受攻击性，并引入了一种新的、更有效的自动生成的排版攻击方法，为此设计了一个独特的测试基准。通过使用该基准，研究发现排版攻击对LVLM构成了重大威胁。 |
| [^9] | [Rendering Wireless Environments Useful for Gradient Estimators: A Zero-Order Stochastic Federated Learning Method](https://arxiv.org/abs/2401.17460) | 提出了一种新颖的零阶随机联邦学习方法，通过利用无线通信通道的特性，在学习算法中考虑了无线通道，避免了资源的浪费和分析难度。 |
| [^10] | [On Rademacher Complexity-based Generalization Bounds for Deep Learning](https://arxiv.org/abs/2208.04284) | 该论文研究了基于Rademacher复杂度的方法在对卷积神经网络进行少类别图像分类时生成非空泛化界限。其中的关键技术贡献是发展了针对函数空间和具有一般Lipschitz激活函数的CNNs的新的Talagrand压缩引理。 |
| [^11] | [AI Oversight and Human Mistakes: Evidence from Centre Court.](http://arxiv.org/abs/2401.16754) | 人工智能系统在纠正人类错误方面起到了积极作用，但此举也潜在导致心理成本，并影响人的决策。通过研究网球比赛中的Hawk-Eye审查系统，我们发现引入AI监督后，裁判员的错误率下降，心理成本导致他们更倾向于将球判为进界，从而产生了类型错判的转变。 |
| [^12] | [The Calibration Gap between Model and Human Confidence in Large Language Models.](http://arxiv.org/abs/2401.13835) | 该论文研究了大型语言模型在传达置信度方面模型和人类之间存在的差距，并发现默认解释会导致用户过高估计模型置信度和准确性。 |
| [^13] | [Multi-modal Multi-kernel Graph Learning for Autism Prediction and Biomarker Discovery.](http://arxiv.org/abs/2303.03388) | 本文提出了一种名为MMKGL的新方法，能够解决多模态集成中各模态之间的负面影响，并从多个图中提取异质信息，以进行自闭症的预测和生物标志物的发现。 |
| [^14] | [Global Convergence Rate of Deep Equilibrium Models with General Activations.](http://arxiv.org/abs/2302.05797) | 该论文研究了具有一般激活函数的深度平衡模型（DEQ）的全局收敛速度，证明了梯度下降以线性收敛速度收敛到全局最优解，并解决了限制平衡点Gram矩阵最小特征值的挑战。 |

# 详细

[^1]: Transformers学习低敏感性函数的简单性偏差

    Simplicity Bias of Transformers to Learn Low Sensitivity Functions

    [https://arxiv.org/abs/2403.06925](https://arxiv.org/abs/2403.06925)

    Transformers在不同数据模态上具有低敏感性，这种简单性偏差有助于解释其在视觉和语言任务中的优越性能。

    

    Transformers在许多任务中取得了最先进的准确性和鲁棒性，但对它们具有的归纳偏差以及这些偏差如何与其他神经网络架构不同的理解仍然难以捉摸。本文中，我们将模型对输入中的随机更改的敏感性概念化为一种简单性偏差的概念，这为解释transformers在不同数据模态上的简单性和谱偏差提供了统一的度量标准。我们展示了transformers在视觉和语言任务中比其他替代架构（如LSTMs、MLPs和CNNs）具有更低的敏感性。我们还展示了低敏感性偏差与改进性能的相关性。

    arXiv:2403.06925v1 Announce Type: cross  Abstract: Transformers achieve state-of-the-art accuracy and robustness across many tasks, but an understanding of the inductive biases that they have and how those biases are different from other neural network architectures remains elusive. Various neural network architectures such as fully connected networks have been found to have a simplicity bias towards simple functions of the data; one version of this simplicity bias is a spectral bias to learn simple functions in the Fourier space. In this work, we identify the notion of sensitivity of the model to random changes in the input as a notion of simplicity bias which provides a unified metric to explain the simplicity and spectral bias of transformers across different data modalities. We show that transformers have lower sensitivity than alternative architectures, such as LSTMs, MLPs and CNNs, across both vision and language tasks. We also show that low-sensitivity bias correlates with impro
    
[^2]: 在现实世界中使用商品移动操作器打开橱柜和抽屉

    Opening Cabinets and Drawers in the Real World using a Commodity Mobile Manipulator

    [https://arxiv.org/abs/2402.17767](https://arxiv.org/abs/2402.17767)

    实现了一个端到端系统，使商品移动操作器成功在以前未见的真实世界环境中打开橱柜和抽屉，感知误差是主要挑战。

    

    在这项工作中，我们构建了一个端到端系统，使商品移动操作器（Stretch RE2）能够在多样的以前未见的真实世界环境中拉开橱柜和抽屉。我们在31个不同的物体和13个不同真实世界环境中进行了4天的实际测试。我们的系统在零击打下，对在未知环境中新颖的橱柜和抽屉的打开率达到61%。对失败模式的分析表明，感知误差是我们系统面临的最重要挑战。

    arXiv:2402.17767v1 Announce Type: cross  Abstract: Pulling open cabinets and drawers presents many difficult technical challenges in perception (inferring articulation parameters for objects from onboard sensors), planning (producing motion plans that conform to tight task constraints), and control (making and maintaining contact while applying forces on the environment). In this work, we build an end-to-end system that enables a commodity mobile manipulator (Stretch RE2) to pull open cabinets and drawers in diverse previously unseen real world environments. We conduct 4 days of real world testing of this system spanning 31 different objects from across 13 different real world environments. Our system achieves a success rate of 61% on opening novel cabinets and drawers in unseen environments zero-shot. An analysis of the failure modes suggests that errors in perception are the most significant challenge for our system. We will open source code and models for others to replicate and bui
    
[^3]: 更好还是更差？通过标签增强学习最小方差特征

    For Better or For Worse? Learning Minimum Variance Features With Label Augmentation

    [https://arxiv.org/abs/2402.06855](https://arxiv.org/abs/2402.06855)

    本研究分析了标签增强方法中标签增强的作用。研究证明，在线性可分数据上使用标签增强训练的线性模型只能学习到最小方差特征，而标准训练可以学习到更高方差特征。此外，标签平滑和Mixup对于训练数据的对抗扰动可能不太鲁棒。

    

    在过去的十年中，数据增强对于成功地训练深度学习模型在分类任务上发挥了关键作用。数据增强技术中的一个重要子类-包括标签平滑和Mixup-涉及在模型训练过程中修改输入数据和输入标签。在这项工作中，我们分析了此类方法中标签增强的作用。我们证明了在线性可分数据上使用标签增强训练的线性模型只能学习到最小方差特征，而标准训练（包括权重衰减）可以学习到更高方差特征。我们的结果的一个重要后果是消极的：与标准训练相比，标签平滑和Mixup对于训练数据的对抗扰动可能不太鲁棒。我们通过对合成数据和图像分类基准的一系列实验证明了我们的理论与实践的一致性。

    Data augmentation has been pivotal in successfully training deep learning models on classification tasks over the past decade. An important subclass of data augmentation techniques - which includes both label smoothing and Mixup - involves modifying not only the input data but also the input label during model training. In this work, we analyze the role played by the label augmentation aspect of such methods. We prove that linear models on linearly separable data trained with label augmentation learn only the minimum variance features in the data, while standard training (which includes weight decay) can learn higher variance features. An important consequence of our results is negative: label smoothing and Mixup can be less robust to adversarial perturbations of the training data when compared to standard training. We verify that our theory reflects practice via a range of experiments on synthetic data and image classification benchmarks.
    
[^4]: Sym-Q：通过顺序决策进行自适应符号回归

    Sym-Q: Adaptive Symbolic Regression via Sequential Decision-Making

    [https://arxiv.org/abs/2402.05306](https://arxiv.org/abs/2402.05306)

    Sym-Q是一个基于强化学习的模型，通过将符号回归重新定义为顺序决策任务来解决现有模型在泛化性和适应性方面的挑战。通过利用监督演示和奖励信号，Sym-Q能够根据拟合精度的质量改进表达式。

    

    符号回归具有从实证数据中揭示潜在数学和物理关系的巨大潜力。虽然现有的基于Transformer的模型在这个领域取得了显著成功，但它们在泛化性和适应性方面面临挑战。通常，当输出表达式不足以适应实验数据时，这些模型缺乏有效的机制来适应或修改表达式。这种缺乏灵活性限制了它们在实际场景中的应用，特别是在发现未知的物理或生物关系方面。受到人类专家如何改进和调整表达式的启发，我们引入了一种新颖的基于强化学习的模型Symbolic Q-network（Sym-Q），将符号回归重新定义为顺序决策任务。Sym-Q利用监督演示并根据奖励信号来改进表达式，奖励信号指示拟合精度的质量。它独特的能力可以处理复杂性。

    Symbolic regression holds great potential for uncovering underlying mathematical and physical relationships from empirical data. While existing transformer-based models have recently achieved significant success in this domain, they face challenges in terms of generalizability and adaptability. Typically, in cases where the output expressions do not adequately fit experimental data, the models lack efficient mechanisms to adapt or modify the expression. This inflexibility hinders their application in real-world scenarios, particularly in discovering unknown physical or biological relationships. Inspired by how human experts refine and adapt expressions, we introduce Symbolic Q-network (Sym-Q), a novel reinforcement learning-based model that redefines symbolic regression as a sequential decision-making task. Sym-Q leverages supervised demonstrations and refines expressions based on reward signals indicating the quality of fitting precision. Its distinctive ability to manage the complexi
    
[^5]: 高效求解偏差Gromov-Wasserstein问题

    Efficient Solvers for Partial Gromov-Wasserstein

    [https://arxiv.org/abs/2402.03664](https://arxiv.org/abs/2402.03664)

    本文提出了两个基于Frank-Wolfe算法的新的高效求解器来解决偏差Gromov-Wasserstein问题，并且证明了PGW问题构成了度量测度空间的度量。

    

    偏差Gromov-Wasserstein（PGW）问题可以比较具有不均匀质量的度量空间中的测度，从而实现这些空间之间的不平衡和部分匹配。本文证明了PGW问题可以转化为Gromov-Wasserstein问题的一个变种，类似于把偏差最优运输问题转化为最优运输问题。这个转化导致了两个新的求解器，基于Frank-Wolfe算法，数学和计算上等价，提供了高效的PGW问题解决方案。我们进一步证明了PGW问题构成了度量测度空间的度量。最后，我们通过与现有基线方法在形状匹配和正样本未标记学习问题上的计算时间和性能比较，验证了我们提出的求解器的有效性。

    The partial Gromov-Wasserstein (PGW) problem facilitates the comparison of measures with unequal masses residing in potentially distinct metric spaces, thereby enabling unbalanced and partial matching across these spaces. In this paper, we demonstrate that the PGW problem can be transformed into a variant of the Gromov-Wasserstein problem, akin to the conversion of the partial optimal transport problem into an optimal transport problem. This transformation leads to two new solvers, mathematically and computationally equivalent, based on the Frank-Wolfe algorithm, that provide efficient solutions to the PGW problem. We further establish that the PGW problem constitutes a metric for metric measure spaces. Finally, we validate the effectiveness of our proposed solvers in terms of computation time and performance on shape-matching and positive-unlabeled learning problems, comparing them against existing baselines.
    
[^6]: 双内点优化学习

    Dual Interior-Point Optimization Learning

    [https://arxiv.org/abs/2402.02596](https://arxiv.org/abs/2402.02596)

    本文介绍了双内点优化学习和双超梯度学习两种方法，用于学习带有有界变量的参数线性规划的对偶可行解。这些方法通过预测约束对应的对偶变量，确保对偶可行性，并且能够提供高保真度的对偶可行解和有效的对偶界限。

    

    本文引入了双内点学习（DIPL）和双超梯度学习（DSL），以学习带有有界变量的参数线性规划的对偶可行解，这在许多行业中都是普遍存在的。DIPL模拟了一种新颖的对偶内点算法，而DSL则模拟了经典的对偶超梯度上升算法。通过预测与约束关联的对偶变量，DIPL和DSL保证对偶可行性，然后利用对于约束界限的对偶的灵活性。DIPL和DSL通过提供质量证明来补充现有的原始学习方法。实验证明，它们能够为大规模最优功率流问题产生高保真度的对偶可行解，并在0.5%的优化差距下提供有效的对偶界限。

    This paper introduces Dual Interior Point Learning (DIPL) and Dual Supergradient Learning (DSL) to learn dual feasible solutions to parametric linear programs with bounded variables, which are pervasive across many industries. DIPL mimics a novel dual interior point algorithm while DSL mimics classical dual supergradient ascent. DIPL and DSL ensure dual feasibility by predicting dual variables associated with the constraints then exploiting the flexibility of the duals of the bound constraints. DIPL and DSL complement existing primal learning methods by providing a certificate of quality. They are shown to produce high-fidelity dual-feasible solutions to large-scale optimal power flow problems providing valid dual bounds under 0.5% optimality gap.
    
[^7]: 在上下文中学习的发展景观

    The Developmental Landscape of In-Context Learning

    [https://arxiv.org/abs/2402.02364](https://arxiv.org/abs/2402.02364)

    在transformers模型中，我们展示了在上下文学习中的离散发展阶段，并引入了两种方法来检测这些阶段的关键里程碑。我们使用行为和结构度量验证了这些方法的有效性。

    

    我们展示了在transformers中，当它们通过语言建模或线性回归任务进行训练时，上下文学习是如何以离散的发展阶段出现的。我们引入了两种方法来检测分隔这些阶段的关键里程碑，通过探测参数空间和函数空间中种群损失的几何特征。我们使用一系列行为和结构度量研究这些新方法揭示的阶段，以建立它们的有效性。

    We show that in-context learning emerges in transformers in discrete developmental stages, when they are trained on either language modeling or linear regression tasks. We introduce two methods for detecting the milestones that separate these stages, by probing the geometry of the population loss in both parameter space and function space. We study the stages revealed by these new methods using a range of behavioral and structural metrics to establish their validity.
    
[^8]: Vision-LLMs通过自动生成的排版攻击可以自欺欺人

    Vision-LLMs Can Fool Themselves with Self-Generated Typographic Attacks

    [https://arxiv.org/abs/2402.00626](https://arxiv.org/abs/2402.00626)

    这项研究深入研究了大规模视觉语言模型（LVLM）对于自动生成的排版攻击的易受攻击性，并引入了一种新的、更有效的自动生成的排版攻击方法，为此设计了一个独特的测试基准。通过使用该基准，研究发现排版攻击对LVLM构成了重大威胁。

    

    最近，在大规模视觉语言模型（LVLM）方面取得了重大进展；这是一种利用大型预训练语言模型的全新类别的视觉语言模型。然而，LVLM对于涉及将误导性文本叠加到图像上的从排版攻击的容易受攻击性却没有研究。此外，先前的排版攻击依赖于从预定义类别集合中随机选择一个误导性类别。然而，随机选择的类别可能不是最有效的攻击类别。为了解决这些问题，我们首先引入了一种独特设计的新颖基准来测试LVLM对排版攻击的容易受攻击性。此外，我们介绍了一种新而更有效的排版攻击：自动生成的排版攻击。实际上，我们的方法通过简单地提示GPT-4V等模型利用其强大的语言能力推荐一种排版攻击来为给定的图像生成攻击。使用我们的新颖基准，我们发现排版攻击对LVLM构成了重大威胁。

    Recently, significant progress has been made on Large Vision-Language Models (LVLMs); a new class of VL models that make use of large pre-trained language models. Yet, their vulnerability to Typographic attacks, which involve superimposing misleading text onto an image remain unstudied. Furthermore, prior work typographic attacks rely on sampling a random misleading class from a predefined set of classes. However, the random chosen class might not be the most effective attack. To address these issues, we first introduce a novel benchmark uniquely designed to test LVLMs vulnerability to typographic attacks. Furthermore, we introduce a new and more effective typographic attack: Self-Generated typographic attacks. Indeed, our method, given an image, make use of the strong language capabilities of models like GPT-4V by simply prompting them to recommend a typographic attack. Using our novel benchmark, we uncover that typographic attacks represent a significant threat against LVLM(s). Furth
    
[^9]: 使无线环境对梯度估计器有用：一种零阶随机联邦学习方法

    Rendering Wireless Environments Useful for Gradient Estimators: A Zero-Order Stochastic Federated Learning Method

    [https://arxiv.org/abs/2401.17460](https://arxiv.org/abs/2401.17460)

    提出了一种新颖的零阶随机联邦学习方法，通过利用无线通信通道的特性，在学习算法中考虑了无线通道，避免了资源的浪费和分析难度。

    

    联邦学习（FL）是一种新颖的机器学习方法，允许多个边缘设备协同训练模型，而无需公开原始数据。然而，当设备和服务器通过无线信道通信时，该方法面临着通信和计算瓶颈。通过利用一个通信高效的框架，我们提出了一种新颖的零阶（ZO）方法，采用一点梯度估计器，利用无线通信通道的特性，而无需知道通道状态系数。这是第一种将无线通道包含在学习算法本身中的方法，而不是浪费资源来分析和消除其影响。这项工作的两个主要困难是，在FL中，目标函数通常不是凸的，这使得将FL扩展到ZO方法具有挑战性，以及包括影响的难度。

    Federated learning (FL) is a novel approach to machine learning that allows multiple edge devices to collaboratively train a model without disclosing their raw data. However, several challenges hinder the practical implementation of this approach, especially when devices and the server communicate over wireless channels, as it suffers from communication and computation bottlenecks in this case. By utilizing a communication-efficient framework, we propose a novel zero-order (ZO) method with a one-point gradient estimator that harnesses the nature of the wireless communication channel without requiring the knowledge of the channel state coefficient. It is the first method that includes the wireless channel in the learning algorithm itself instead of wasting resources to analyze it and remove its impact. The two main difficulties of this work are that in FL, the objective function is usually not convex, which makes the extension of FL to ZO methods challenging, and that including the impa
    
[^10]: 基于Rademacher复杂度的深度学习一般化界限研究

    On Rademacher Complexity-based Generalization Bounds for Deep Learning

    [https://arxiv.org/abs/2208.04284](https://arxiv.org/abs/2208.04284)

    该论文研究了基于Rademacher复杂度的方法在对卷积神经网络进行少类别图像分类时生成非空泛化界限。其中的关键技术贡献是发展了针对函数空间和具有一般Lipschitz激活函数的CNNs的新的Talagrand压缩引理。

    

    我们展示了基于Rademacher复杂度的方法可以生成对卷积神经网络（CNNs）进行分类少量类别图像非空泛化界限。新的Talagrand压缩引理的发展对于高维映射函数空间和具有一般Lipschitz激活函数的CNNs是一个关键技术贡献。我们的结果表明，Rademacher复杂度不依赖于CNNs的网络长度，特别是对于诸如ReLU，Leaky ReLU，Parametric Rectifier Linear Unit，Sigmoid和Tanh等特定类型的激活函数。

    We show that the Rademacher complexity-based approach can generate non-vacuous generalisation bounds on Convolutional Neural Networks (CNNs) for classifying a small number of classes of images. The development of new Talagrand's contraction lemmas for high-dimensional mappings between function spaces and CNNs for general Lipschitz activation functions is a key technical contribution. Our results show that the Rademacher complexity does not depend on the network length for CNNs with some special types of activation functions such as ReLU, Leaky ReLU, Parametric Rectifier Linear Unit, Sigmoid, and Tanh.
    
[^11]: AI监督和人类错误：来自中心法庭的证据

    AI Oversight and Human Mistakes: Evidence from Centre Court. (arXiv:2401.16754v1 [cs.LG])

    [http://arxiv.org/abs/2401.16754](http://arxiv.org/abs/2401.16754)

    人工智能系统在纠正人类错误方面起到了积极作用，但此举也潜在导致心理成本，并影响人的决策。通过研究网球比赛中的Hawk-Eye审查系统，我们发现引入AI监督后，裁判员的错误率下降，心理成本导致他们更倾向于将球判为进界，从而产生了类型错判的转变。

    

    在机器学习算法不断提升的驱动下，人工智能（AI）系统已经开始在许多场合用于纠正人类错误。我们提供了首个实地证据，证明这种AI监督会产生心理成本，影响人的决策。我们调查了AI监督发生的最高可见性场景之一：顶级网球比赛中裁判的Hawk-Eye审查。我们发现，引入Hawk-Eye审查后，裁判的整体错误率降低，符合心理成本被AI否定的合理忽视现象。我们还发现，裁判增加了对球入内的判定率，从而产生了从II类错误（将球判为出界，实际上是进界）到I类错误（将球判为进界，实际上是出界）的转变。通过对理性不注意的裁判模型进行心理成本的结构估计，我们的结果表明，由于AI否定的心理成本，裁判员降低了错误判定的风险并提高了球入内的判定率。

    Powered by the increasing predictive capabilities of machine learning algorithms, artificial intelligence (AI) systems have begun to be used to overrule human mistakes in many settings. We provide the first field evidence this AI oversight carries psychological costs that can impact human decision-making. We investigate one of the highest visibility settings in which AI oversight has occurred: the Hawk-Eye review of umpires in top tennis tournaments. We find that umpires lowered their overall mistake rate after the introduction of Hawk-Eye review, in line with rational inattention given psychological costs of being overruled by AI. We also find that umpires increased the rate at which they called balls in, which produced a shift from making Type II errors (calling a ball out when in) to Type I errors (calling a ball in when out). We structurally estimate the psychological costs of being overruled by AI using a model of rational inattentive umpires, and our results suggest that because 
    
[^12]: 语言模型中模型和人类置信度之间的校准差距

    The Calibration Gap between Model and Human Confidence in Large Language Models. (arXiv:2401.13835v1 [cs.LG])

    [http://arxiv.org/abs/2401.13835](http://arxiv.org/abs/2401.13835)

    该论文研究了大型语言模型在传达置信度方面模型和人类之间存在的差距，并发现默认解释会导致用户过高估计模型置信度和准确性。

    

    为了使大型语言模型（LLM）能够获得人类的信任，它们需要在某种意义上实现良好的校准，即能够准确评估和传达它们的预测正确的可能性。最近的研究关注了LLM内部置信度评估的质量，但问题仍然是LLM能够如何将这种内部模型置信度传达给人类用户。本文探讨了人类对LLM响应的外部置信度与模型内部置信度之间的差距。通过涉及多项选择题的实验，我们系统地检查了人类用户识别LLM输出可信度的能力。我们的研究重点分为两个方面：（1）评估用户对真实LLM置信度的感知和（2）调查个性化解释对该感知的影响。研究结果显示，LLM的默认解释往往会导致用户过高估计模型的置信度和准确性。通过修改解释的方式可以减小这种误差。

    For large language models (LLMs) to be trusted by humans they need to be well-calibrated in the sense that they can accurately assess and communicate how likely it is that their predictions are correct. Recent work has focused on the quality of internal LLM confidence assessments, but the question remains of how well LLMs can communicate this internal model confidence to human users. This paper explores the disparity between external human confidence in an LLM's responses and the internal confidence of the model. Through experiments involving multiple-choice questions, we systematically examine human users' ability to discern the reliability of LLM outputs. Our study focuses on two key areas: (1) assessing users' perception of true LLM confidence and (2) investigating the impact of tailored explanations on this perception. The research highlights that default explanations from LLMs often lead to user overestimation of both the model's confidence and its' accuracy. By modifying the expl
    
[^13]: 基于多模态多核图学习的自闭症预测与生物标志物发现

    Multi-modal Multi-kernel Graph Learning for Autism Prediction and Biomarker Discovery. (arXiv:2303.03388v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.03388](http://arxiv.org/abs/2303.03388)

    本文提出了一种名为MMKGL的新方法，能够解决多模态集成中各模态之间的负面影响，并从多个图中提取异质信息，以进行自闭症的预测和生物标志物的发现。

    

    基于图学习的多模态集成和分类是疾病预测中最具挑战性的障碍之一。我们提出了一种名为MMKGL的新方法来有效抵消多模态集成过程中各模态之间负面影响，并从图中提取异质信息。具体地，我们提出了多模态图嵌入模块，并通过自适应学习生成多个图，然后提出多核图学习模块，从多模态图中提取异质信息。在不同层次上聚合多模态图中的信息，实现了对自闭症的预测和生物标志物的发现。

    Due to its complexity, graph learning-based multi-modal integration and classification is one of the most challenging obstacles for disease prediction. To effectively offset the negative impact between modalities in the process of multi-modal integration and extract heterogeneous information from graphs, we propose a novel method called MMKGL (Multi-modal Multi-Kernel Graph Learning). For the problem of negative impact between modalities, we propose a multi-modal graph embedding module to construct a multi-modal graph. Different from conventional methods that manually construct static graphs for all modalities, each modality generates a separate graph by adaptive learning, where a function graph and a supervision graph are introduced for optimization during the multi-graph fusion embedding process. We then propose a multi-kernel graph learning module to extract heterogeneous information from the multi-modal graph. The information in the multi-modal graph at different levels is aggregat
    
[^14]: 具有一般激活函数的深度平衡模型的全局收敛速度

    Global Convergence Rate of Deep Equilibrium Models with General Activations. (arXiv:2302.05797v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.05797](http://arxiv.org/abs/2302.05797)

    该论文研究了具有一般激活函数的深度平衡模型（DEQ）的全局收敛速度，证明了梯度下降以线性收敛速度收敛到全局最优解，并解决了限制平衡点Gram矩阵最小特征值的挑战。

    

    在最近的一篇论文中，Ling等人研究了具有ReLU激活函数的过参数化深度平衡模型（DEQ）。他们证明了对于二次损失函数，梯度下降方法以线性收敛速度收敛到全局最优解。本文表明，对于具有任何具有有界一阶和二阶导数的激活函数的DEQ，该事实仍然成立。由于新的激活函数通常是非线性的，限制平衡点的Gram矩阵的最小特征值尤其具有挑战性。为了完成这个任务，我们需要创建一个新的总体Gram矩阵，并开发一种具有Hermite多项式展开的新形式的双重激活函数。

    In a recent paper, Ling et al. investigated the over-parametrized Deep Equilibrium Model (DEQ) with ReLU activation. They proved that the gradient descent converges to a globally optimal solution at a linear convergence rate for the quadratic loss function. This paper shows that this fact still holds for DEQs with any general activation that has bounded first and second derivatives. Since the new activation function is generally non-linear, bounding the least eigenvalue of the Gram matrix of the equilibrium point is particularly challenging. To accomplish this task, we need to create a novel population Gram matrix and develop a new form of dual activation with Hermite polynomial expansion.
    

