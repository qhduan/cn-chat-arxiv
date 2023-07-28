# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Incrementally-Computable Neural Networks: Efficient Inference for Dynamic Inputs.](http://arxiv.org/abs/2307.14988) | 本论文介绍了一种增量计算的神经网络方法，通过离散化中间值并过滤不必要的修改，实现了对动态输入的高效推理。 |
| [^2] | [Multi-Source Domain Adaptation through Dataset Dictionary Learning in Wasserstein Space.](http://arxiv.org/abs/2307.14953) | 本文提出了一种基于字典学习和最优传输的MSDA框架，通过将每个域表示为字典原子的Wasserstein重心来缓解数据分布偏移。根据该字典，提出了两种新的MSDA方法，分别基于目标域标记样本的重构和在原子分布上学习的分类器的集成。在多个基准测试集上进行的实验证明，这些方法在分类任务上取得了显著的改进效果。 |
| [^3] | [Kernelised Normalising Flows.](http://arxiv.org/abs/2307.14839) | 本文提出了一种新颖的核化归一化流范式，称为Ferumal流，它将核函数集成到归一化流的框架中。相对于基于神经网络的流，核化流可以在低数据环境中产生竞争力或优越的结果，同时保持参数效率。 |
| [^4] | [Speed Limits for Deep Learning.](http://arxiv.org/abs/2307.14653) | 研究使用随机热力学方法，根据权重分布间的Wasserstein-2距离和熵产生速率，提供了对深度学习网络从初始状态到完全训练的最大速度限制。通过应用于线性和可线性化的神经网络，结果表明，在某些缩放假设下，学习在某种程度上是最优的。 |
| [^5] | [Linear Convergence of Black-Box Variational Inference: Should We Stick the Landing?.](http://arxiv.org/abs/2307.14642) | 本文证明了带有控制变量的黑盒变分推断在完美变分族规范下以几何速度收敛，为BBVI提供了收敛性保证，同时提出了对熵梯度估计器的改进，对比了STL估计器，并给出了明确的非渐近复杂度保证。 |
| [^6] | [Imitating Complex Trajectories: Bridging Low-Level Stability and High-Level Behavior.](http://arxiv.org/abs/2307.14619) | 本文提出了一个理论框架，研究了在非线性动态系统中模仿复杂专家演示的行为。通过稳定模仿策略并确保准确估计演示者分布，可以使模仿者与演示者的轨迹分布相近。 |
| [^7] | [Optimal Estimation in Mixed-Membership Stochastic Block Models.](http://arxiv.org/abs/2307.14530) | 本论文研究了重叠社区检测问题，在混合成员随机块模型的基础上提出了一个新的估计器，并建立了估计误差的极小下界。 |
| [^8] | [How to Scale Your EMA.](http://arxiv.org/abs/2307.13813) | 本研究提供了在存在模型EMA的情况下进行优化的缩放规则，以保持训练动态的一致性。这对于实际机器学习中的权衡批量大小和墙钟时间非常重要。模型EMA能够提高模型的性能以及稳定训练过程，并为自监督学习提供学习信号。 |
| [^9] | [A Bayesian approach to quantifying uncertainties and improving generalizability in traffic prediction models.](http://arxiv.org/abs/2307.05946) | 本研究提出了一种贝叶斯循环神经网络框架，通过引入归一化处理，实现交通预测模型中的不确定性量化和更高的泛化能力。 |
| [^10] | [Automating Model Comparison in Factor Graphs.](http://arxiv.org/abs/2306.05965) | 本文基于自定义混合节点 Forney 样式的因子图消息传递，实现了高效自动化贝叶斯模型平均、选择和组合，并缩短了模型设计周期。 |
| [^11] | [Spectral learning of Bernoulli linear dynamical systems models.](http://arxiv.org/abs/2303.02060) | 该论文提出了一种用于快速、高效拟合概率波-伯努利潜在线性动力系统模型的谱学习方法。这种方法通过转换样本矩的方式将传统的子空间识别方法扩展到了伯努利设置中，得到了一个鲁棒的固定成本估计器。在数据有限的情况下，谱估计可以为Laplace-EM拟合提供良好的初始化。 |
| [^12] | [Causal Lifting and Link Prediction.](http://arxiv.org/abs/2302.01198) | 本文开发了第一个能够处理链路预测中路径依赖的因果模型，并介绍了因果提升的概念，通过有限的干预数据识别因果链路预测查询。 |
| [^13] | [Statistical process monitoring of artificial neural networks.](http://arxiv.org/abs/2209.07436) | 这篇论文提出了一种基于人工神经网络生成的数据潜在特征表示的监控方法，以确定数据流开始变得非平稳的时间。该方法通过应用基于数据深度计算和归一化排名的多元控制图进行监测，并与各种基准方法进行了比较。 |
| [^14] | [Neural Point Estimation for Fast Optimal Likelihood-Free Inference.](http://arxiv.org/abs/2208.12942) | 本文介绍了一种快速、无需似然函数、易于进行基于自举的不确定性量化的推断工具——神经点估计器，并通过模拟研究和实际案例分析证明其可以在弱识别和高参数化模型中进行快速且最优的参数估计。 |
| [^15] | [Neural Networks for Scalar Input and Functional Output.](http://arxiv.org/abs/2208.05776) | 该论文提出了一种解决标量输入和函数输出之间回归问题的方法，使用前馈神经网络预测函数响应。该方法适用于大量预测变量或非线性关系，并可以控制预测曲线的平滑程度。在实验中验证了方法的有效性。 |
| [^16] | [Algorithmic Gaussianization through Sketching: Converting Data into Sub-gaussian Random Designs.](http://arxiv.org/abs/2206.10291) | 该论文通过引入杠杆分数稀疏（LESS）嵌入的草图技术，提供了一种算法框架，实现了对数据分布的高斯化，从而能够高效地构建与次高斯随机设计几乎无法区分的数据草图。 |
| [^17] | [Dynamic covariate balancing: estimating treatment effects over time with potential local projections.](http://arxiv.org/abs/2103.01280) | 本文提出了一种通过动态协变量平衡方法，基于过去历史上潜在结果期望的局部投影，估计面板数据中动态变化的治疗效果，并考虑结果和时间变化的协变量与治疗轨迹的关系以及治疗效应的异质性。研究结果表明该方法具有良好的渐近性质和数值特性，在实证应用中具有优势。 |
| [^18] | [On the Generalization Effects of Linear Transformations in Data Augmentation.](http://arxiv.org/abs/2005.00695) | 这项研究考虑了一类线性转换，并研究了其在过参数化线性回归设置中对岭估计量的影响。研究发现，能够保持数据标签的转换可以通过扩大训练数据的张量来改善估计结果；而混合数据的转换则通过起到正则化作用来改善估计结果。此外，通过在MNIST数据集上进行验证，研究者提出了一个增强方案，该方案通过模型对转换后数据的不确定性进行搜索转换空间，并在图像和文本数据集上验证了其有效性。 |

# 详细

[^1]: 增量计算的神经网络：处理动态输入的高效推理方法

    Incrementally-Computable Neural Networks: Efficient Inference for Dynamic Inputs. (arXiv:2307.14988v1 [cs.LG])

    [http://arxiv.org/abs/2307.14988](http://arxiv.org/abs/2307.14988)

    本论文介绍了一种增量计算的神经网络方法，通过离散化中间值并过滤不必要的修改，实现了对动态输入的高效推理。

    

    深度学习在处理动态输入（例如传感器数据或用户输入）时常面临着高效处理的挑战。本论文提出了一种增量计算的方法，通过重复使用计算来适应输入变化，以解决这个问题。我们使用向量量化来离散化网络中的中间值，并过滤噪声和不必要的隐藏神经元修改，从而促进值的重用。我们将此方法应用于Transformer架构，创建了一个高效的增量推理算法。

    Deep learning often faces the challenge of efficiently processing dynamic inputs, such as sensor data or user inputs. For example, an AI writing assistant is required to update its suggestions in real time as a document is edited. Re-running the model each time is expensive, even with compression techniques like knowledge distillation, pruning, or quantization. Instead, we take an incremental computing approach, looking to reuse calculations as the inputs change. However, the dense connectivity of conventional architectures poses a major obstacle to incremental computation, as even minor input changes cascade through the network and restrict information reuse. To address this, we use vector quantization to discretize intermediate values in the network, which filters out noisy and unnecessary modifications to hidden neurons, facilitating the reuse of their values. We apply this approach to the transformers architecture, creating an efficient incremental inference algorithm with complexi
    
[^2]: 在Wasserstein空间中通过数据集字典学习进行多源域自适应

    Multi-Source Domain Adaptation through Dataset Dictionary Learning in Wasserstein Space. (arXiv:2307.14953v1 [cs.LG])

    [http://arxiv.org/abs/2307.14953](http://arxiv.org/abs/2307.14953)

    本文提出了一种基于字典学习和最优传输的MSDA框架，通过将每个域表示为字典原子的Wasserstein重心来缓解数据分布偏移。根据该字典，提出了两种新的MSDA方法，分别基于目标域标记样本的重构和在原子分布上学习的分类器的集成。在多个基准测试集上进行的实验证明，这些方法在分类任务上取得了显著的改进效果。

    

    本文旨在解决多源域自适应（MSDA）问题，该问题旨在在从多个标记的源域转移知识到未标记的目标域时缓解数据分布偏移。我们提出了一种基于字典学习和最优传输的新型MSDA框架。我们将MSDA中的每个域解释为经验分布。因此，我们将每个域表达为字典原子的Wasserstein重心，这些原子是经验分布。我们提出了一种新的通过小批量学习的算法DaDiL：（i）原子分布；（ii）重心坐标矩阵。根据我们的字典，我们提出了两种新的MSDA方法：DaDiL-R，基于目标域标记样本的重构；DaDiL-E，基于在原子分布上学习的分类器的集成。我们在3个基准测试集中评估了我们的方法：Caltech-Office、Office 31和CRWU，在分类上改进了以前的最先进技术3.15％、2.29％和7.71％。

    This paper seeks to solve Multi-Source Domain Adaptation (MSDA), which aims to mitigate data distribution shifts when transferring knowledge from multiple labeled source domains to an unlabeled target domain. We propose a novel MSDA framework based on dictionary learning and optimal transport. We interpret each domain in MSDA as an empirical distribution. As such, we express each domain as a Wasserstein barycenter of dictionary atoms, which are empirical distributions. We propose a novel algorithm, DaDiL, for learning via mini-batches: (i) atom distributions; (ii) a matrix of barycentric coordinates. Based on our dictionary, we propose two novel methods for MSDA: DaDil-R, based on the reconstruction of labeled samples in the target domain, and DaDiL-E, based on the ensembling of classifiers learned on atom distributions. We evaluate our methods in 3 benchmarks: Caltech-Office, Office 31, and CRWU, where we improved previous state-of-the-art by 3.15%, 2.29%, and 7.71% in classification 
    
[^3]: 核化归一化流

    Kernelised Normalising Flows. (arXiv:2307.14839v1 [stat.ML])

    [http://arxiv.org/abs/2307.14839](http://arxiv.org/abs/2307.14839)

    本文提出了一种新颖的核化归一化流范式，称为Ferumal流，它将核函数集成到归一化流的框架中。相对于基于神经网络的流，核化流可以在低数据环境中产生竞争力或优越的结果，同时保持参数效率。

    

    归一化流是以其可逆的架构而被描述的生成模型。然而，可逆性要求对其表达能力施加限制，需要大量的参数和创新的架构设计来达到满意的结果。虽然基于流的模型主要依赖于基于神经网络的转换来实现表达能力，但替代的转换方法却受到了有限的关注。在这项工作中，我们提出了一种新颖的核化归一化流范式，称为Ferumal流，它将核函数集成到框架中。我们的结果表明，相比于基于神经网络的流，核化流可以产生有竞争力或优越的结果，同时保持参数效率。核化流在低数据环境中表现出色，可以在数据稀缺的应用中进行灵活的非参数密度估计。

    Normalising Flows are generative models characterised by their invertible architecture. However, the requirement of invertibility imposes constraints on their expressiveness, necessitating a large number of parameters and innovative architectural designs to achieve satisfactory outcomes. Whilst flow-based models predominantly rely on neural-network-based transformations for expressive designs, alternative transformation methods have received limited attention. In this work, we present Ferumal flow, a novel kernelised normalising flow paradigm that integrates kernels into the framework. Our results demonstrate that a kernelised flow can yield competitive or superior results compared to neural network-based flows whilst maintaining parameter efficiency. Kernelised flows excel especially in the low-data regime, enabling flexible non-parametric density estimation in applications with sparse data availability.
    
[^4]: 深度学习的速度限制

    Speed Limits for Deep Learning. (arXiv:2307.14653v1 [stat.ML])

    [http://arxiv.org/abs/2307.14653](http://arxiv.org/abs/2307.14653)

    研究使用随机热力学方法，根据权重分布间的Wasserstein-2距离和熵产生速率，提供了对深度学习网络从初始状态到完全训练的最大速度限制。通过应用于线性和可线性化的神经网络，结果表明，在某些缩放假设下，学习在某种程度上是最优的。

    

    现阶段的神经网络需要极大的计算能力才能进行训练。因此很自然地想知道它们是否被最优化地训练。在本文中，我们应用了最近在随机热力学中的一个进展，允许根据它们的Wasserstein-2距离的比率和连接它们的动态过程的熵产生速率，对从初始权重分布到完全训练的网络的最大速度进行界定。考虑了梯度流和Langevin训练动力学，我们为线性和可线性化的神经网络（例如神经切向核(NTK)）提供了这些速度限制的解析表达式。值得注意的是，如果对NTK谱和标签的谱分解做出一些合理的缩放假设，学习在某种程度上是最优化的。我们的结果与在CIFAR-10上使用卷积神经网络(CNNs)和全连接神经网络(FCNs)进行的小规模实验一致，显示了

    State-of-the-art neural networks require extreme computational power to train. It is therefore natural to wonder whether they are optimally trained. Here we apply a recent advancement in stochastic thermodynamics which allows bounding the speed at which one can go from the initial weight distribution to the final distribution of the fully trained network, based on the ratio of their Wasserstein-2 distance and the entropy production rate of the dynamical process connecting them. Considering both gradient-flow and Langevin training dynamics, we provide analytical expressions for these speed limits for linear and linearizable neural networks e.g. Neural Tangent Kernel (NTK). Remarkably, given some plausible scaling assumptions on the NTK spectra and spectral decomposition of the labels -- learning is optimal in a scaling sense. Our results are consistent with small-scale experiments with Convolutional Neural Networks (CNNs) and Fully Connected Neural networks (FCNs) on CIFAR-10, showing a
    
[^5]: 黑盒变分推断的线性收敛性：我们应该坚持到底吗？

    Linear Convergence of Black-Box Variational Inference: Should We Stick the Landing?. (arXiv:2307.14642v1 [stat.ML])

    [http://arxiv.org/abs/2307.14642](http://arxiv.org/abs/2307.14642)

    本文证明了带有控制变量的黑盒变分推断在完美变分族规范下以几何速度收敛，为BBVI提供了收敛性保证，同时提出了对熵梯度估计器的改进，对比了STL估计器，并给出了明确的非渐近复杂度保证。

    

    我们证明了带有控制变量的黑盒变分推断（BBVI），特别是着陆稳定（STL）估计器，在完美变分族规范下收敛于几何（传统上称为“线性”）速度。特别地，我们证明了STL估计器的梯度方差的二次界限，该界限包括了误指定的变分族。结合先前关于二次方差条件的工作，这直接暗示了在使用投影随机梯度下降的情况下BBVI的收敛性。我们还改进了现有对于正常封闭形式熵梯度估计器的分析，这使得我们能够将其与STL估计器进行比较，并为两者提供明确的非渐进复杂度保证。

    We prove that black-box variational inference (BBVI) with control variates, particularly the sticking-the-landing (STL) estimator, converges at a geometric (traditionally called "linear") rate under perfect variational family specification. In particular, we prove a quadratic bound on the gradient variance of the STL estimator, one which encompasses misspecified variational families. Combined with previous works on the quadratic variance condition, this directly implies convergence of BBVI with the use of projected stochastic gradient descent. We also improve existing analysis on the regular closed-form entropy gradient estimators, which enables comparison against the STL estimator and provides explicit non-asymptotic complexity guarantees for both.
    
[^6]: 模仿复杂轨迹：桥接低层稳定性与高层行为

    Imitating Complex Trajectories: Bridging Low-Level Stability and High-Level Behavior. (arXiv:2307.14619v1 [cs.LG])

    [http://arxiv.org/abs/2307.14619](http://arxiv.org/abs/2307.14619)

    本文提出了一个理论框架，研究了在非线性动态系统中模仿复杂专家演示的行为。通过稳定模仿策略并确保准确估计演示者分布，可以使模仿者与演示者的轨迹分布相近。

    

    我们提出了一个理论框架来研究在非线性动态系统中模仿随机、非马尔可夫、潜在多模态（即“复杂”）专家演示的行为。我们的框架使用低层控制器（无论是学习的还是隐含的）来稳定围绕专家演示的模仿策略。我们证明，在（a）合适的低层稳定性保证和（b）学习策略的随机连续性属性（我们称之为“总变差连续性”）（TVC）的情况下，一个精确估计演示者状态分布上的行动的模仿者会与演示者对整个轨迹的分布相近。然后，我们证明可以通过将流行的数据增强规则与一种新颖的算法技巧相结合（即在执行时添加增强噪声）来确保TVC并且最小程度上降低精度。我们将我们的保证实例化为由扩散模型参数化的策略，并证明如果学习者准确地估计了演示者的分布，则最终完成这种实例化。

    We propose a theoretical framework for studying the imitation of stochastic, non-Markovian, potentially multi-modal (i.e. "complex" ) expert demonstrations in nonlinear dynamical systems. Our framework invokes low-level controllers either learned or implicit in position-command control - to stabilize imitation policies around expert demonstrations. We show that with (a) a suitable low-level stability guarantee and (b) a stochastic continuity property of the learned policy we call "total variation continuity" (TVC), an imitator that accurately estimates actions on the demonstrator's state distribution closely matches the demonstrator's distribution over entire trajectories. We then show that TVC can be ensured with minimal degradation of accuracy by combining a popular data-augmentation regimen with a novel algorithmic trick: adding augmentation noise at execution time. We instantiate our guarantees for policies parameterized by diffusion models and prove that if the learner accuratel
    
[^7]: 混合成员随机块模型中的最优估计

    Optimal Estimation in Mixed-Membership Stochastic Block Models. (arXiv:2307.14530v1 [stat.ML])

    [http://arxiv.org/abs/2307.14530](http://arxiv.org/abs/2307.14530)

    本论文研究了重叠社区检测问题，在混合成员随机块模型的基础上提出了一个新的估计器，并建立了估计误差的极小下界。

    

    社区检测是现代网络科学中最关键的问题之一。其应用可以在各个领域找到，从蛋白质建模到社交网络分析。最近，出现了许多论文研究重叠社区检测问题，即网络中的每个节点可能属于多个社区。在本文中，我们考虑了由Airoldi等人（2008）首次提出的混合成员随机块模型（MMSB）。MMSB在图中对重叠社区结构提供了相当一般的设置。本文的核心问题是在观察到的网络中重建社区之间的关系。我们比较了不同的方法，并建立了估计误差的极小下界。然后，我们提出了一个与这个下界匹配的新估计器。理论结果在对所考虑的模型的相当普遍条件下得到证明。最后，我们通过一系列实验来说明这个理论。

    Community detection is one of the most critical problems in modern network science. Its applications can be found in various fields, from protein modeling to social network analysis. Recently, many papers appeared studying the problem of overlapping community detection, where each node of a network may belong to several communities. In this work, we consider Mixed-Membership Stochastic Block Model (MMSB) first proposed by Airoldi et al. (2008). MMSB provides quite a general setting for modeling overlapping community structure in graphs. The central question of this paper is to reconstruct relations between communities given an observed network. We compare different approaches and establish the minimax lower bound on the estimation error. Then, we propose a new estimator that matches this lower bound. Theoretical results are proved under fairly general conditions on the considered model. Finally, we illustrate the theory in a series of experiments.
    
[^8]: 如何扩展您的EMA（arXiv:2307.13813v1 [stat.ML]）

    How to Scale Your EMA. (arXiv:2307.13813v1 [stat.ML])

    [http://arxiv.org/abs/2307.13813](http://arxiv.org/abs/2307.13813)

    本研究提供了在存在模型EMA的情况下进行优化的缩放规则，以保持训练动态的一致性。这对于实际机器学习中的权衡批量大小和墙钟时间非常重要。模型EMA能够提高模型的性能以及稳定训练过程，并为自监督学习提供学习信号。

    

    在实际机器学习中，保持训练动态在批量大小之间的一致性是一种重要工具，它能够在批量大小和墙钟时间之间进行权衡。这种权衡通常通过一个缩放规则来实现，例如，在随机梯度下降中，应该将学习率与批量大小呈线性关系。另一个实际机器学习的重要工具是模型指数移动平均（EMA），它是一个不接收梯度信息的模型副本，而是以一定的动量跟随其目标模型。这个模型EMA可以提高监督学习的稳健性和泛化性能，稳定伪标记，为自监督学习提供学习信号。之前的研究将模型EMA与优化分开处理，导致批量大小之间存在不同的训练动态和较低的模型性能。在这项工作中，我们提供了在存在模型EMA的情况下进行优化的缩放规则，并展示了其效果。

    Preserving training dynamics across batch sizes is an important tool for practical machine learning as it enables the trade-off between batch size and wall-clock time. This trade-off is typically enabled by a scaling rule, for example, in stochastic gradient descent, one should scale the learning rate linearly with the batch size. Another important tool for practical machine learning is the model Exponential Moving Average (EMA), which is a model copy that does not receive gradient information, but instead follows its target model with some momentum. This model EMA can improve the robustness and generalization properties of supervised learning, stabilize pseudo-labeling, and provide a learning signal for Self-Supervised Learning (SSL). Prior works have treated the model EMA separately from optimization, leading to different training dynamics across batch sizes and lower model performance. In this work, we provide a scaling rule for optimization in the presence of model EMAs and demonst
    
[^9]: 一种贝叶斯方法用于量化交通预测模型中的不确定性和改善泛化能力

    A Bayesian approach to quantifying uncertainties and improving generalizability in traffic prediction models. (arXiv:2307.05946v1 [cs.LG])

    [http://arxiv.org/abs/2307.05946](http://arxiv.org/abs/2307.05946)

    本研究提出了一种贝叶斯循环神经网络框架，通过引入归一化处理，实现交通预测模型中的不确定性量化和更高的泛化能力。

    

    交通数据预测的深度学习模型可以通过多层架构对复杂函数进行优化建模，但这些方法的一个主要缺点是大多数方法不提供带有不确定性估计的预测结果，而这对于交通运营和控制是必需的。本研究提出了一种贝叶斯循环神经网络框架，通过引入谱归一化到其隐藏层，实现交通预测中的不确定性量化和更高的泛化能力。我们的论文表明，归一化通过控制模型的复杂性并减少对训练数据的过度拟合风险，改善了深度神经网络的泛化性能。

    Deep-learning models for traffic data prediction can have superior performance in modeling complex functions using a multi-layer architecture. However, a major drawback of these approaches is that most of these approaches do not offer forecasts with uncertainty estimates, which are essential for traffic operations and control. Without uncertainty estimates, it is difficult to place any level of trust to the model predictions, and operational strategies relying on overconfident predictions can lead to worsening traffic conditions. In this study, we propose a Bayesian recurrent neural network framework for uncertainty quantification in traffic prediction with higher generalizability by introducing spectral normalization to its hidden layers. In our paper, we have shown that normalization alters the training process of deep neural networks by controlling the model's complexity and reducing the risk of overfitting to the training data. This, in turn, helps improve the generalization perfor
    
[^10]: 在因子图中自动进行模型比较

    Automating Model Comparison in Factor Graphs. (arXiv:2306.05965v1 [cs.LG])

    [http://arxiv.org/abs/2306.05965](http://arxiv.org/abs/2306.05965)

    本文基于自定义混合节点 Forney 样式的因子图消息传递，实现了高效自动化贝叶斯模型平均、选择和组合，并缩短了模型设计周期。

    

    在文献中，贝叶斯状态和参数估计已经被有效自动化，但对于模型比较尚未如此，因此仍需要容易出错和耗时的手动推导。因此，模型比较经常被忽视和忽略，尽管它很重要。本文通过在Forney样式的因子图上使用自定义混合节点上的消息传递来高效地自动化贝叶斯模型平均、选择和组合。进而可使用缩放因子同时执行参数和状态推断以及模型比较。这种方法缩短了模型设计周期，同时允许简单地扩展到分层和时间模型先验，以适应建模复杂的时变过程。

    Bayesian state and parameter estimation have been automated effectively in the literature, however, this has not yet been the case for model comparison, which therefore still requires error-prone and time-consuming manual derivations. As a result, model comparison is often overlooked and ignored, despite its importance. This paper efficiently automates Bayesian model averaging, selection, and combination by message passing on a Forney-style factor graph with a custom mixture node. Parameter and state inference, and model comparison can then be executed simultaneously using message passing with scale factors. This approach shortens the model design cycle and allows for the straightforward extension to hierarchical and temporal model priors to accommodate for modeling complicated time-varying processes.
    
[^11]: Bernoulli线性动力系统模型的谱学习

    Spectral learning of Bernoulli linear dynamical systems models. (arXiv:2303.02060v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2303.02060](http://arxiv.org/abs/2303.02060)

    该论文提出了一种用于快速、高效拟合概率波-伯努利潜在线性动力系统模型的谱学习方法。这种方法通过转换样本矩的方式将传统的子空间识别方法扩展到了伯努利设置中，得到了一个鲁棒的固定成本估计器。在数据有限的情况下，谱估计可以为Laplace-EM拟合提供良好的初始化。

    

    具有Bernoulli观测的潜在线性动力系统为识别二进制时间序列数据的时间动态提供了强大的建模框架，这些数据在二进制决策和离散随机过程（例如离散神经尖峰训练）等各种情况下产生。在这里，我们开发了一种快速有效的概率波/Bernoulli潜在线性动态系统（LDS）模型的谱学习方法。我们的方法通过对第一和第二个样本矩的转换将传统的子空间识别方法扩展到Bernoulli设置中。这导致了一个健壮的固定成本估计器，避免了局部最优解的危险以及期望最大化（EM）算法等迭代拟合过程的长时间计算。在数据有限或数据的统计结构不满足假设的情况下，我们证明了谱估计为Laplace-EM拟合提供了良好的初始化。

    Latent linear dynamical systems with Bernoulli observations provide a powerful modeling framework for identifying the temporal dynamics underlying binary time series data, which arise in a variety of contexts such as binary decision-making and discrete stochastic processes (e.g., binned neural spike trains). Here we develop a spectral learning method for fast, efficient fitting of probit-Bernoulli latent linear dynamical system (LDS) models. Our approach extends traditional subspace identification methods to the Bernoulli setting via a transformation of the first and second sample moments. This results in a robust, fixed-cost estimator that avoids the hazards of local optima and the long computation time of iterative fitting procedures like the expectation-maximization (EM) algorithm. In regimes where data is limited or assumptions about the statistical structure of the data are not met, we demonstrate that the spectral estimate provides a good initialization for Laplace-EM fitting. Fi
    
[^12]: 因果提升与链路预测

    Causal Lifting and Link Prediction. (arXiv:2302.01198v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.01198](http://arxiv.org/abs/2302.01198)

    本文开发了第一个能够处理链路预测中路径依赖的因果模型，并介绍了因果提升的概念，通过有限的干预数据识别因果链路预测查询。

    

    现有的链路预测因果模型假设存在一组固有的节点因子，即在节点出生时就定义的固有特征，它们控制着图中链路的因果演化。然而，在某些因果任务中，链路形成是路径依赖的：链路干预的结果取决于现有的链路。不幸的是，这些现有的因果方法并不适用于路径依赖的链路形成，因为链路之间的级联功能依赖（由路径依赖性产生）要么无法识别，要么需要大量不切实际的控制变量。为了克服这个问题，我们开发了第一个能够处理链路预测中路径依赖的因果模型。在这项工作中，我们引入了因果提升的概念，这是一种独立于图的因果模型的不变性，可以利用有限的干预数据来识别因果链路预测查询。此外，我们展示了结构对两个节点之间嵌入的低维表示的方式。

    Existing causal models for link prediction assume an underlying set of inherent node factors -- an innate characteristic defined at the node's birth -- that governs the causal evolution of links in the graph. In some causal tasks, however, link formation is path-dependent: The outcome of link interventions depends on existing links. Unfortunately, these existing causal methods are not designed for path-dependent link formation, as the cascading functional dependencies between links (arising from path dependence) are either unidentifiable or require an impractical number of control variables. To overcome this, we develop the first causal model capable of dealing with path dependencies in link prediction. In this work we introduce the concept of causal lifting, an invariance in causal models of independent interest that, on graphs, allows the identification of causal link prediction queries using limited interventional data. Further, we show how structural pairwise embeddings exhibit low
    
[^13]: 人工神经网络的统计过程监控

    Statistical process monitoring of artificial neural networks. (arXiv:2209.07436v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2209.07436](http://arxiv.org/abs/2209.07436)

    这篇论文提出了一种基于人工神经网络生成的数据潜在特征表示的监控方法，以确定数据流开始变得非平稳的时间。该方法通过应用基于数据深度计算和归一化排名的多元控制图进行监测，并与各种基准方法进行了比较。

    

    基于人工智能的模型的快速发展要求创新的监控技术，这些技术可以以低计算成本实时操作。在机器学习中，特别是考虑到人工神经网络（ANNs），模型通常是以监督方式进行训练的。因此，在模型部署期间，输入和输出之间学习到的关系必须保持有效。如果这个平稳性假设成立，我们可以得出结论，ANN提供准确的预测。否则，需要重新训练或重建模型。我们建议考虑由ANN生成的数据的潜在特征表示（称为“嵌入”），以确定数据流开始变得非平稳的时间。具体而言，我们通过应用基于数据深度计算和归一化排名的多元控制图来监测嵌入。我们将引入的方法与各种ANN基准方法进行了比较。

    The rapid advancement of models based on artificial intelligence demands innovative monitoring techniques which can operate in real time with low computational costs. In machine learning, especially if we consider artificial neural networks (ANNs), the models are often trained in a supervised manner. Consequently, the learned relationship between the input and the output must remain valid during the model's deployment. If this stationarity assumption holds, we can conclude that the ANN provides accurate predictions. Otherwise, the retraining or rebuilding of the model is required. We propose considering the latent feature representation of the data (called "embedding") generated by the ANN to determine the time when the data stream starts being nonstationary. In particular, we monitor embeddings by applying multivariate control charts based on the data depth calculation and normalized ranks. The performance of the introduced method is compared with benchmark approaches for various ANN 
    
[^14]: 快速最优无似然推断的神经点估计

    Neural Point Estimation for Fast Optimal Likelihood-Free Inference. (arXiv:2208.12942v3 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2208.12942](http://arxiv.org/abs/2208.12942)

    本文介绍了一种快速、无需似然函数、易于进行基于自举的不确定性量化的推断工具——神经点估计器，并通过模拟研究和实际案例分析证明其可以在弱识别和高参数化模型中进行快速且最优的参数估计。

    

    神经点估计器是一种将数据映射到参数点估计的神经网络。它们快速、无需似然函数，并且由于它们的平均特性，易于进行基于自举的不确定性量化。本文旨在提高统计学家对于这种相对较新的推断工具的认识，并通过提供用户友好的开源软件来促进其采用。我们还关注了从重复数据进行推断的广泛问题，在神经设置中使用排列不变神经网络来解决这个问题。通过广泛的模拟研究，我们展示了这些神经点估计器可以快速且最优地（从贝叶斯意义上）在弱识别和高参数化模型中进行估计，并且相对容易。我们通过对红海极端海表温度分析来证明它们的适用性，在训练之后，我们获得了参数估计和基于自举的置信区间。

    Neural point estimators are neural networks that map data to parameter point estimates. They are fast, likelihood free and, due to their amortised nature, amenable to fast bootstrap-based uncertainty quantification. In this paper, we aim to increase the awareness of statisticians to this relatively new inferential tool, and to facilitate its adoption by providing user-friendly open-source software. We also give attention to the ubiquitous problem of making inference from replicated data, which we address in the neural setting using permutation-invariant neural networks. Through extensive simulation studies we show that these neural point estimators can quickly and optimally (in a Bayes sense) estimate parameters in weakly-identified and highly-parameterised models with relative ease. We demonstrate their applicability through an analysis of extreme sea-surface temperature in the Red Sea where, after training, we obtain parameter estimates and bootstrap-based confidence intervals from h
    
[^15]: 标量输入和函数输出的神经网络

    Neural Networks for Scalar Input and Functional Output. (arXiv:2208.05776v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2208.05776](http://arxiv.org/abs/2208.05776)

    该论文提出了一种解决标量输入和函数输出之间回归问题的方法，使用前馈神经网络预测函数响应。该方法适用于大量预测变量或非线性关系，并可以控制预测曲线的平滑程度。在实验中验证了方法的有效性。

    

    在一组标量预测变量上回归函数响应可以是一项具有挑战性的任务，特别是当有大量预测变量或者预测变量与响应之间的关系是非线性的时候。在这项工作中，我们提出了一个解决方案：使用前馈神经网络（NN）预测标量输入下的函数响应。首先，我们将函数响应转化为有限维度表示，并构建一个输出该表示的神经网络。然后，我们提出通过目标函数修改神经网络的输出，并引入不同的目标函数来进行网络训练。所提出的模型适用于均匀和不均匀间隔的数据，并可以进一步应用平滑惩罚项来控制预测曲线的平滑程度。实现这些特性的困难在于定义可以进行反向传播的目标函数。在我们的实验中，我们展示了我们的方法在多个数据集上的有效性。

    The regression of a functional response on a set of scalar predictors can be a challenging task, especially if there is a large number of predictors, or the relationship between those predictors and the response is nonlinear. In this work, we propose a solution to this problem: a feed-forward neural network (NN) designed to predict a functional response using scalar inputs. First, we transform the functional response to a finite-dimensional representation and construct an NN that outputs this representation. Then, we propose to modify the output of an NN via the objective function and introduce different objective functions for network training. The proposed models are suited for both regularly and irregularly spaced data, and a roughness penalty can be further applied to control the smoothness of the predicted curve. The difficulty in implementing both those features lies in the definition of objective functions that can be back-propagated. In our experiments, we demonstrate that our 
    
[^16]: 通过草图技术实现算法高斯化：将数据转换为次高斯随机设计

    Algorithmic Gaussianization through Sketching: Converting Data into Sub-gaussian Random Designs. (arXiv:2206.10291v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.10291](http://arxiv.org/abs/2206.10291)

    该论文通过引入杠杆分数稀疏（LESS）嵌入的草图技术，提供了一种算法框架，实现了对数据分布的高斯化，从而能够高效地构建与次高斯随机设计几乎无法区分的数据草图。

    

    算法高斯化是使用随机草图方法或采样方法生成大型数据集的较小表示时可能出现的现象：对于某些任务，观察到的这些草图表示具有许多稳健性能特征，这些特征在数据样本来自次高斯随机设计时已被确认存在，而次高斯随机设计是数据分布的一个强大统计模型。然而，这种现象仅在特定任务和度量标准上进行了研究，或者依赖于计算昂贵的方法。我们通过提供一种通过平均来高斯化数据分布的算法框架来解决这个问题，并证明可以高效地构建与次高斯随机设计在总变异距离上几乎无法区分的数据草图。特别地，依赖于最近引入的一种称为杠杆分数稀疏（LESS）嵌入的草图技术，我们展示了可以构建一个n逼真的高斯化数据草图。

    Algorithmic Gaussianization is a phenomenon that can arise when using randomized sketching or sampling methods to produce smaller representations of large datasets: For certain tasks, these sketched representations have been observed to exhibit many robust performance characteristics that are known to occur when a data sample comes from a sub-gaussian random design, which is a powerful statistical model of data distributions. However, this phenomenon has only been studied for specific tasks and metrics, or by relying on computationally expensive methods. We address this by providing an algorithmic framework for gaussianizing data distributions via averaging, proving that it is possible to efficiently construct data sketches that are nearly indistinguishable (in terms of total variation distance) from sub-gaussian random designs. In particular, relying on a recently introduced sketching technique called Leverage Score Sparsified (LESS) embeddings, we show that one can construct an $n\ti
    
[^17]: 动态协变量平衡：基于潜在局部投影的治疗效果随时间估计

    Dynamic covariate balancing: estimating treatment effects over time with potential local projections. (arXiv:2103.01280v3 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2103.01280](http://arxiv.org/abs/2103.01280)

    本文提出了一种通过动态协变量平衡方法，基于过去历史上潜在结果期望的局部投影，估计面板数据中动态变化的治疗效果，并考虑结果和时间变化的协变量与治疗轨迹的关系以及治疗效应的异质性。研究结果表明该方法具有良好的渐近性质和数值特性，在实证应用中具有优势。

    

    本文研究了面板数据中治疗历史的估计和推断问题，特别是在治疗在时间上动态变化的情况下。我们提出了一种方法，允许治疗根据高维协变量、过去的结果和治疗动态分配，同时考虑结果和时间变化的协变量与治疗轨迹的关系，以及治疗效应的异质性。我们的方法通过在过去历史上对潜在结果期望进行递归投影，然后通过平衡动态可观测特征来控制偏差。我们研究了估计量的渐近性质和数值特性，并在实证应用中展示了该方法的优势。

    This paper studies the estimation and inference of treatment histories in panel data settings when treatments change dynamically over time.  We propose a method that allows for (i) treatments to be assigned dynamically over time based on high-dimensional covariates, past outcomes and treatments; (ii) outcomes and time-varying covariates to depend on treatment trajectories; (iii) heterogeneity of treatment effects.  Our approach recursively projects potential outcomes' expectations on past histories. It then controls the bias by balancing dynamically observable characteristics. We study the asymptotic and numerical properties of the estimator and illustrate the benefits of the procedure in an empirical application.
    
[^18]: 关于数据增强中线性转换的泛化效果的研究

    On the Generalization Effects of Linear Transformations in Data Augmentation. (arXiv:2005.00695v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2005.00695](http://arxiv.org/abs/2005.00695)

    这项研究考虑了一类线性转换，并研究了其在过参数化线性回归设置中对岭估计量的影响。研究发现，能够保持数据标签的转换可以通过扩大训练数据的张量来改善估计结果；而混合数据的转换则通过起到正则化作用来改善估计结果。此外，通过在MNIST数据集上进行验证，研究者提出了一个增强方案，该方案通过模型对转换后数据的不确定性进行搜索转换空间，并在图像和文本数据集上验证了其有效性。

    

    数据增强是一种在图像和文本分类任务等应用中提高性能的强大技术。然而，对于各种增强方法为何有效以及其工作原理的严格理解还很有限。在本研究中，我们考虑了一类线性转换，并研究了其在过参数化线性回归设置中对岭估计量的影响。首先，我们通过扩大训练数据的张量来展示了能够保持数据标签的转换会改善估计结果。其次，我们通过混合数据的转换展示了对估计量起到了正则化的作用。最后，我们通过MNIST数据集验证了我们的理论洞见。基于这些洞见，我们提出了一个通过模型对转换后的数据的不确定性来搜索转换空间的增强方案。我们在图像和文本数据集上验证了我们提出的方案。例如，我们的方法在使用Wide-ResNet对CIFAR-100数据集上优于随机采样方法1.24%。

    Data augmentation is a powerful technique to improve performance in applications such as image and text classification tasks. Yet, there is little rigorous understanding of why and how various augmentations work. In this work, we consider a family of linear transformations and study their effects on the ridge estimator in an over-parametrized linear regression setting. First, we show that transformations that preserve the labels of the data can improve estimation by enlarging the span of the training data. Second, we show that transformations that mix data can improve estimation by playing a regularization effect. Finally, we validate our theoretical insights on MNIST. Based on the insights, we propose an augmentation scheme that searches over the space of transformations by how uncertain the model is about the transformed data. We validate our proposed scheme on image and text datasets. For example, our method outperforms random sampling methods by 1.24% on CIFAR-100 using Wide-ResNet
    

