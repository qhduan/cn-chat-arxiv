# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts](https://arxiv.org/abs/2403.10568) | 本文提出了MoPE技术，通过解开提示以自适应捕获数据集级和实例级特征，引入了混合Prompt专家来增强表达能力，并且在多模态融合中表现出更大的表达能力和可扩展性。 |
| [^2] | [Directional Smoothness and Gradient Methods: Convergence and Adaptivity](https://arxiv.org/abs/2403.04081) | 提出了一种新的次优性界限方法以解决梯度下降问题，利用方向平滑度开发上界并获得收敛保证，在实验中证明优于传统基于L平滑度的理论。 |
| [^3] | [Fast, Scale-Adaptive, and Uncertainty-Aware Downscaling of Earth System Model Fields with Generative Foundation Models](https://arxiv.org/abs/2403.02774) | 通过学习一致性模型，在不需要重新训练的情况下高效、准确地降尺度任意地球系统模型模拟，并产生概率性降尺度场。 |
| [^4] | [Remove that Square Root: A New Efficient Scale-Invariant Version of AdaGrad](https://arxiv.org/abs/2403.02648) | KATE是一种新的优化算法，提出了一种与AdaGrad标度不变的适应方法，并在广义线性模型和一般的非凸问题中证明了其标度不变性。数值实验结果表明，KATE在各种场景中均优于AdaGrad并与Adam性能匹配/超越。 |
| [^5] | [Invariant Test-Time Adaptation for Vision-Language Model Generalization](https://arxiv.org/abs/2403.00376) | 本文提出了一个测试时提示调优范式，通过优化可学习的提示，迫使模型利用真正的因果不变特征，以解决视觉-语言模型在特定任务需求上无法有效利用预训练特征的挑战。 |
| [^6] | [On diffusion models for amortized inference: Benchmarking and improving stochastic control and sampling](https://arxiv.org/abs/2402.05098) | 本研究探讨了训练扩散模型以从给定分布中采样的问题，并针对随机控制和采样提出了一种新的探索策略，通过基准测试比较了不同推断方法的相对优劣，并对过去的工作提出了质疑。 |
| [^7] | [A Random Matrix Approach to Low-Multilinear-Rank Tensor Approximation](https://arxiv.org/abs/2402.03169) | 该研究采用随机矩阵方法，在低多线性秩张量逼近中展示了对种植的低秩信号的估计，并根据大维谱行为和信噪比准确预测了重建性能，并给出了HOOI收敛的充分条件。 |
| [^8] | [ACPO: AI-Enabled Compiler-Driven Program Optimization](https://arxiv.org/abs/2312.09982) | 该论文提出了ACPO框架，通过机器学习模型提供给LLVM简单全面的工具，以实现编译器驱动的程序优化。 |
| [^9] | [Divergences between Language Models and Human Brains](https://arxiv.org/abs/2311.09308) | 该论文系统地探索了语言模型（LMs）和人类大脑在语言处理方面的差异，发现在社交/情感智能和物理常识领域，LMs无法很好地捕捉到人类的表现，但在这些领域对LMs进行微调可以提高其性能。 |
| [^10] | [End-To-End Set-Based Training for Neural Network Verification.](http://arxiv.org/abs/2401.14961) | 本论文提出了一种端到端基于集合的训练方法，用于训练鲁棒性神经网络进行形式化验证，并证明该方法能够简化验证过程并有效训练出易于验证的神经网络。 |
| [^11] | [A Closer Look at AUROC and AUPRC under Class Imbalance.](http://arxiv.org/abs/2401.06091) | 通过数学分析，研究发现AUROC和AUPRC在类别不平衡情况下可以以概率术语简洁地相关联。相比于人们普遍认为的AUPRC优越性，结果表明AUPRC并不如人们预期的有优势，并且可能是一种有害的指标。研究还通过分析大量文献验证了这一结论。 |
| [^12] | [Set-based Neural Network Encoding.](http://arxiv.org/abs/2305.16625) | 提出了一种能够集合化地编码神经网络参数的神经网络权重编码方法，并引入了一种逐层编码方案来考虑神经网络的分层计算结构。同时引入了“pad-chunk-encode”流水线进行神经网络层的高效编码处理，还提出了新的神经网络泛化性能预测任务。 |
| [^13] | [Inhomogeneous graph trend filtering via a l2,0 cardinality penalty.](http://arxiv.org/abs/2304.05223) | 本文提出了一种基于L2，0基数惩罚的图趋势过滤（GTF）模型，可同时进行k-means聚类和基于图的最小割，以估计在节点之间具有不均匀平滑水平的分段平滑图信号，并在降噪、支持恢复和半监督分类任务上表现更好，比现有方法更高效地处理大型数据集。 |

# 详细

[^1]: MoPE：通过Prompt专家混合实现参数高效和可扩展的多模态融合

    MoPE: Parameter-Efficient and Scalable Multimodal Fusion via Mixture of Prompt Experts

    [https://arxiv.org/abs/2403.10568](https://arxiv.org/abs/2403.10568)

    本文提出了MoPE技术，通过解开提示以自适应捕获数据集级和实例级特征，引入了混合Prompt专家来增强表达能力，并且在多模态融合中表现出更大的表达能力和可扩展性。

    

    Prompt调整已经证明在融合多模态任务的单模基础模型时具有参数效率性。然而，其有限的适应性和表达能力导致性能不佳与其他调整方法相比。本文通过将简单提示解开以自适应地捕获数据集级和实例级特征来解决这个问题。建立在这种解开的基础上，我们引入了Prompt专家的混合（MoPE）技术来增强表达能力。MoPE利用多模态配对先验在每个实例基础上路由最有效的提示。与简单提示相比，我们基于MoPE的条件提示对多模态融合具有更大的表达能力，在训练数据和可训练参数总数上具有更好的扩展性。我们还研究了一个专家路由的正则化项，导致专家的不断发展专长，不同专家专注于不同的特征。

    arXiv:2403.10568v1 Announce Type: cross  Abstract: Prompt-tuning has demonstrated parameter-efficiency in fusing unimodal foundation models for multimodal tasks. However, its limited adaptivity and expressiveness lead to suboptimal performance when compared with other tuning methods. In this paper, we address this issue by disentangling the vanilla prompts to adaptively capture dataset-level and instance-level features. Building upon this disentanglement, we introduce the mixture of prompt experts (MoPE) technique to enhance expressiveness. MoPE leverages multimodal pairing priors to route the most effective prompt on a per-instance basis. Compared to vanilla prompting, our MoPE-based conditional prompting exhibits greater expressiveness for multimodal fusion, scaling better with the training data and the overall number of trainable parameters. We also study a regularization term for expert routing, leading to emergent expert specialization, where different experts focus on different c
    
[^2]: 方向平滑度与梯度方法：收敛性和自适应性

    Directional Smoothness and Gradient Methods: Convergence and Adaptivity

    [https://arxiv.org/abs/2403.04081](https://arxiv.org/abs/2403.04081)

    提出了一种新的次优性界限方法以解决梯度下降问题，利用方向平滑度开发上界并获得收敛保证，在实验中证明优于传统基于L平滑度的理论。

    

    我们为梯度下降（GD）开发了一种新的次优性界限，其取决于沿优化路径的目标条件性，而不是全局最坏情况常数。我们证明的关键是方向平滑度，这是我们用来开发目标上界的梯度变化度量。最小化这些上界需要解决隐式方程以获得一系列强适应的步长；我们展示了对于凸二次函数, 这些方程很容易求解，并为两种经典步长提供了新的保证。对于一般函数, 我们证明Polyak步长和归一化GD获得快速的、路径相关的速率，尽管不使用方向平滑度的任何知识。 Logistic回归实验表明，我们的收敛保证比基于L平滑度的经典理论更紧密。

    arXiv:2403.04081v1 Announce Type: new  Abstract: We develop new sub-optimality bounds for gradient descent (GD) that depend on the conditioning of the objective along the path of optimization, rather than on global, worst-case constants. Key to our proofs is directional smoothness, a measure of gradient variation that we use to develop upper-bounds on the objective. Minimizing these upper-bounds requires solving implicit equations to obtain a sequence of strongly adapted step-sizes; we show that these equations are straightforward to solve for convex quadratics and lead to new guarantees for two classical step-sizes. For general functions, we prove that the Polyak step-size and normalized GD obtain fast, path-dependent rates despite using no knowledge of the directional smoothness. Experiments on logistic regression show our convergence guarantees are tighter than the classical theory based on L-smoothness.
    
[^3]: 快速、自适应尺度和具有不确定性意识的地球系统模型场降尺度与生成基础模型

    Fast, Scale-Adaptive, and Uncertainty-Aware Downscaling of Earth System Model Fields with Generative Foundation Models

    [https://arxiv.org/abs/2403.02774](https://arxiv.org/abs/2403.02774)

    通过学习一致性模型，在不需要重新训练的情况下高效、准确地降尺度任意地球系统模型模拟，并产生概率性降尺度场。

    

    精确和高分辨率的地球系统模型(ESM)模拟对于评估人为气候变化对生态和社会经济影响至关重要，但计算成本过高。最近的机器学习方法在ESM模拟的降尺度中表现出色，优于最先进的统计方法。然而，现有方法对每个ESM都需要计算昂贵的重新训练，并且在训练期间未见过的气候预测效果差。我们通过学习一个一致性模型(CM)，以零样本方式高效准确地降尺度任意ESM模拟来解决这些缺点。我们的基础模型方法以只受观测参考数据限制的分辨率产生概率性降尺度场。我们展示了CM在维持高可控性的同时以较低的计算成本优于最先进的扩散模型。

    arXiv:2403.02774v1 Announce Type: cross  Abstract: Accurate and high-resolution Earth system model (ESM) simulations are essential to assess the ecological and socio-economic impacts of anthropogenic climate change, but are computationally too expensive. Recent machine learning approaches have shown promising results in downscaling ESM simulations, outperforming state-of-the-art statistical approaches. However, existing methods require computationally costly retraining for each ESM and extrapolate poorly to climates unseen during training. We address these shortcomings by learning a consistency model (CM) that efficiently and accurately downscales arbitrary ESM simulations without retraining in a zero-shot manner. Our foundation model approach yields probabilistic downscaled fields at resolution only limited by the observational reference data. We show that the CM outperforms state-of-the-art diffusion models at a fraction of computational cost while maintaining high controllability on
    
[^4]: 移除平方根：一种新的高效标度不变版本的AdaGrad

    Remove that Square Root: A New Efficient Scale-Invariant Version of AdaGrad

    [https://arxiv.org/abs/2403.02648](https://arxiv.org/abs/2403.02648)

    KATE是一种新的优化算法，提出了一种与AdaGrad标度不变的适应方法，并在广义线性模型和一般的非凸问题中证明了其标度不变性。数值实验结果表明，KATE在各种场景中均优于AdaGrad并与Adam性能匹配/超越。

    

    自适应方法在机器学习中非常流行，因为它们可以降低学习速率调整的成本。本文引入了一种名为KATE的新型优化算法，它提出了一个著名的AdaGrad算法的标度不变适应。我们证明了KATE在广义线性模型案例中的标度不变性。此外，对于一般的光滑非凸问题，我们为KATE建立了一个收敛速率为$O \left(\frac{\log T}{\sqrt{T}} \right)$，与AdaGrad和Adam的最佳收敛速率相匹配。我们还通过不同问题的数值实验将KATE与其他最先进的自适应算法Adam和AdaGrad进行了比较，包括在真实数据上进行图像分类和文本分类等复杂机器学习任务。结果表明，在所有考虑到的场景中，KATE始终胜过AdaGrad，并且在性能上匹配/超越Adam。

    arXiv:2403.02648v1 Announce Type: cross  Abstract: Adaptive methods are extremely popular in machine learning as they make learning rate tuning less expensive. This paper introduces a novel optimization algorithm named KATE, which presents a scale-invariant adaptation of the well-known AdaGrad algorithm. We prove the scale-invariance of KATE for the case of Generalized Linear Models. Moreover, for general smooth non-convex problems, we establish a convergence rate of $O \left(\frac{\log T}{\sqrt{T}} \right)$ for KATE, matching the best-known ones for AdaGrad and Adam. We also compare KATE to other state-of-the-art adaptive algorithms Adam and AdaGrad in numerical experiments with different problems, including complex machine learning tasks like image classification and text classification on real data. The results indicate that KATE consistently outperforms AdaGrad and matches/surpasses the performance of Adam in all considered scenarios.
    
[^5]: 视觉-语言模型泛化的不变测试时适应性

    Invariant Test-Time Adaptation for Vision-Language Model Generalization

    [https://arxiv.org/abs/2403.00376](https://arxiv.org/abs/2403.00376)

    本文提出了一个测试时提示调优范式，通过优化可学习的提示，迫使模型利用真正的因果不变特征，以解决视觉-语言模型在特定任务需求上无法有效利用预训练特征的挑战。

    

    arXiv:2403.00376v1 公告类型: 交叉摘要: 视觉-语言基础模型在大量图像-文本配对数据集上的可扩展性使其在众多下游任务中展现出卓越成功。然而，这些模型在应用于长尾任务（如细粒度图像分类）时显示出明显局限，这是由于“决策捷径”导致了它们的泛化能力受限。本文发现CLIP模型具有丰富的特征集，涵盖了既有的\textit{期望不变因果特征}又有的\textit{不希望的决策捷径}。此外，CLIP在下游任务中的表现不佳源自其无法有效利用预训练特征以符合特定任务要求。为解决这一挑战，本文引入一种测试时提示调优范式，优化一个可学习的提示，从而促使模型利用真正的因果不变特征。

    arXiv:2403.00376v1 Announce Type: cross  Abstract: Vision-language foundation models have exhibited remarkable success across a multitude of downstream tasks due to their scalability on extensive image-text paired datasets. However, these models display significant limitations when applied to long-tail tasks, such as fine-grained image classification, as a result of "decision shortcuts" that hinders their generalization capabilities. In this work, we find that the CLIP model possesses a rich set of features, encompassing both \textit{desired invariant causal features} and \textit{undesired decision shortcuts}. Moreover, the underperformance of CLIP on downstream tasks originates from its inability to effectively utilize pre-trained features in accordance with specific task requirements. To address this challenge, this paper introduces a test-time prompt tuning paradigm that optimizes a learnable prompt, thereby compelling the model to exploit genuine causal invariant features while dis
    
[^6]: 关于分散推断模型的扩散模型：基准测试和改进随机控制和采样

    On diffusion models for amortized inference: Benchmarking and improving stochastic control and sampling

    [https://arxiv.org/abs/2402.05098](https://arxiv.org/abs/2402.05098)

    本研究探讨了训练扩散模型以从给定分布中采样的问题，并针对随机控制和采样提出了一种新的探索策略，通过基准测试比较了不同推断方法的相对优劣，并对过去的工作提出了质疑。

    

    我们研究了训练扩散模型以从给定的非标准化密度或能量函数分布中采样的问题。我们对几种扩散结构推断方法进行了基准测试，包括基于模拟的变分方法和离策略方法（连续生成流网络）。我们的结果揭示了现有算法的相对优势，同时对过去的研究提出了一些质疑。我们还提出了一种新颖的离策略方法探索策略，基于目标空间中的局部搜索和回放缓冲区的使用，并证明它可以改善各种目标分布上的样本质量。我们研究的采样方法和基准测试的代码已公开在https://github.com/GFNOrg/gfn-diffusion，作为未来在分散推断模型上工作的基础。

    We study the problem of training diffusion models to sample from a distribution with a given unnormalized density or energy function. We benchmark several diffusion-structured inference methods, including simulation-based variational approaches and off-policy methods (continuous generative flow networks). Our results shed light on the relative advantages of existing algorithms while bringing into question some claims from past work. We also propose a novel exploration strategy for off-policy methods, based on local search in the target space with the use of a replay buffer, and show that it improves the quality of samples on a variety of target distributions. Our code for the sampling methods and benchmarks studied is made public at https://github.com/GFNOrg/gfn-diffusion as a base for future work on diffusion models for amortized inference.
    
[^7]: 低多线性秩张量逼近的随机矩阵方法

    A Random Matrix Approach to Low-Multilinear-Rank Tensor Approximation

    [https://arxiv.org/abs/2402.03169](https://arxiv.org/abs/2402.03169)

    该研究采用随机矩阵方法，在低多线性秩张量逼近中展示了对种植的低秩信号的估计，并根据大维谱行为和信噪比准确预测了重建性能，并给出了HOOI收敛的充分条件。

    

    本研究从计算阈值附近的一般尖峰张量模型，对种植的低秩信号估计进行了全面的认识。依靠大型随机矩阵理论的标准工具，我们表征了数据张量的展开的大维谱行为，并展示了决定主要信号方向可检测性的相关信噪比。这些结果可以准确地预测在非平凡区域的截断多线性奇异值分解(MLSVD)的重建性能。这一点尤其重要，因为它作为更高阶正交迭代(HOOI)方案的初始化，其收敛到最佳低多线性秩逼近完全取决于其初始化。我们给出了HOOI收敛的充分条件，并证明在大维极限下收敛前的迭代次数趋于1。

    This work presents a comprehensive understanding of the estimation of a planted low-rank signal from a general spiked tensor model near the computational threshold. Relying on standard tools from the theory of large random matrices, we characterize the large-dimensional spectral behavior of the unfoldings of the data tensor and exhibit relevant signal-to-noise ratios governing the detectability of the principal directions of the signal. These results allow to accurately predict the reconstruction performance of truncated multilinear SVD (MLSVD) in the non-trivial regime. This is particularly important since it serves as an initialization of the higher-order orthogonal iteration (HOOI) scheme, whose convergence to the best low-multilinear-rank approximation depends entirely on its initialization. We give a sufficient condition for the convergence of HOOI and show that the number of iterations before convergence tends to $1$ in the large-dimensional limit.
    
[^8]: ACPO: AI-Enabled Compiler-Driven Program Optimization

    ACPO: AI-Enabled Compiler-Driven Program Optimization

    [https://arxiv.org/abs/2312.09982](https://arxiv.org/abs/2312.09982)

    该论文提出了ACPO框架，通过机器学习模型提供给LLVM简单全面的工具，以实现编译器驱动的程序优化。

    

    该论文提出了ACPO：AI-Enabled Compiler-driven Program Optimization，这是一个新颖的框架，为LLVM提供简单全面的工具，以从应用机器学习模型来进行不同的优化通路中获益。首先展示了ACPO的高层视图、类层次结构和功能，然后通过将循环展开和函数内联传递的ML使能化，展示了ACPO的一些用例，描述了ACPO如何发挥作用。

    arXiv:2312.09982v2 Announce Type: replace-cross  Abstract: The key to performance optimization of a program is to decide correctly when a certain transformation should be applied by a compiler. This is an ideal opportunity to apply machine-learning models to speed up the tuning process; while this realization has been around since the late 90s, only recent advancements in ML enabled a practical application of ML to compilers as an end-to-end framework.   This paper presents ACPO: \textbf{\underline{A}}I-Enabled \textbf{\underline{C}}ompiler-driven \textbf{\underline{P}}rogram \textbf{\underline{O}}ptimization; a novel framework to provide LLVM with simple and comprehensive tools to benefit from employing ML models for different optimization passes. We first showcase the high-level view, class hierarchy, and functionalities of ACPO and subsequently, demonstrate a couple of use cases of ACPO by ML-enabling the Loop Unroll and Function Inlining passes and describe how ACPO can be leverage
    
[^9]: 语言模型与人脑的差异

    Divergences between Language Models and Human Brains

    [https://arxiv.org/abs/2311.09308](https://arxiv.org/abs/2311.09308)

    该论文系统地探索了语言模型（LMs）和人类大脑在语言处理方面的差异，发现在社交/情感智能和物理常识领域，LMs无法很好地捕捉到人类的表现，但在这些领域对LMs进行微调可以提高其性能。

    

    机器和人类是否以相似的方式处理语言？最近的研究暗示肯定，发现大脑信号可以通过语言模型（LMs）的内部表示有效地进行预测。尽管这样的结果被认为反映了LMs和人类大脑之间的共享计算原理，但LMs和人类在语言表示和使用上也存在明显的差异。在这项工作中，我们通过检查LM表示和人类大脑对语言的响应之间的差异，通过采用两个数据集对受试者阅读和听叙述故事的方式，系统地探索了人类和机器语言处理之间的分歧。通过数据驱动的方法，我们确定了两个领域，即社交/情感智能和物理常识，这些领域在LMs中无法很好地捕捉到。然后，我们使用人类行为实验验证了这些领域，并证明在这些领域对LMs进行微调可以改善其性能。

    Do machines and humans process language in similar ways? Recent research has hinted in the affirmative, finding that brain signals can be effectively predicted using the internal representations of language models (LMs). Although such results are thought to reflect shared computational principles between LMs and human brains, there are also clear differences in how LMs and humans represent and use language. In this work, we systematically explore the divergences between human and machine language processing by examining the differences between LM representations and human brain responses to language as measured by Magnetoencephalography (MEG) across two datasets in which subjects read and listened to narrative stories. Using a data-driven approach, we identify two domains that are not captured well by LMs: social/emotional intelligence and physical commonsense. We then validate these domains with human behavioral experiments and show that fine-tuning LMs on these domains can improve th
    
[^10]: 神经网络验证的端到端基于集合的训练方法

    End-To-End Set-Based Training for Neural Network Verification. (arXiv:2401.14961v1 [cs.LG])

    [http://arxiv.org/abs/2401.14961](http://arxiv.org/abs/2401.14961)

    本论文提出了一种端到端基于集合的训练方法，用于训练鲁棒性神经网络进行形式化验证，并证明该方法能够简化验证过程并有效训练出易于验证的神经网络。

    

    神经网络容易受到对抗性攻击，即微小的输入扰动可能导致神经网络输出产生重大变化。安全关键环境需要对输入扰动具有鲁棒性的神经网络。然而，训练和形式化验证鲁棒性神经网络是具有挑战性的。我们首次采用端到端基于集合的训练方法来解决这个挑战，该训练方法能够训练出可进行形式化验证的鲁棒性神经网络。我们的训练方法能够大大简化已训练神经网络的后续形式化鲁棒性验证过程。相比于以往的研究主要关注增强神经网络训练的对抗性攻击，我们的方法利用基于集合的计算来训练整个扰动输入集合上的神经网络。此外，我们证明我们的基于集合的训练方法可以有效训练出易于验证的鲁棒性神经网络。

    Neural networks are vulnerable to adversarial attacks, i.e., small input perturbations can result in substantially different outputs of a neural network. Safety-critical environments require neural networks that are robust against input perturbations. However, training and formally verifying robust neural networks is challenging. We address this challenge by employing, for the first time, a end-to-end set-based training procedure that trains robust neural networks for formal verification. Our training procedure drastically simplifies the subsequent formal robustness verification of the trained neural network. While previous research has predominantly focused on augmenting neural network training with adversarial attacks, our approach leverages set-based computing to train neural networks with entire sets of perturbed inputs. Moreover, we demonstrate that our set-based training procedure effectively trains robust neural networks, which are easier to verify. In many cases, set-based trai
    
[^11]: AUROC和AUPRC在类不平衡下的深入研究

    A Closer Look at AUROC and AUPRC under Class Imbalance. (arXiv:2401.06091v1 [cs.LG])

    [http://arxiv.org/abs/2401.06091](http://arxiv.org/abs/2401.06091)

    通过数学分析，研究发现AUROC和AUPRC在类别不平衡情况下可以以概率术语简洁地相关联。相比于人们普遍认为的AUPRC优越性，结果表明AUPRC并不如人们预期的有优势，并且可能是一种有害的指标。研究还通过分析大量文献验证了这一结论。

    

    在机器学习中，一个广泛的观点是，在二分类任务中，面积受限制的准确率曲线（AUPRC）比受试者工作特征曲线下的面积（AUROC）更好地用于模型比较，尤其是在存在类别不平衡的情况下。本文通过新颖的数学分析挑战了这一观点，并说明了AUROC和AUPRC可以以概率术语简洁地相关联。我们证明了AUPRC并不如人们普遍认为的在类别不平衡的情况下更优，甚至可能是一种有害的指标，因为它倾向于过分偏向于在正样本较为频繁的子群中改善模型。这种偏差可能会无意中增加算法的差异。在这些洞见的推动下，我们对现有的机器学习文献进行了彻底的回顾，并利用大型语言模型对arXiv上的150多万篇论文进行了分析。我们的调查重点是验证和证明声称的AUPRC优越性的普遍性。

    In machine learning (ML), a widespread adage is that the area under the precision-recall curve (AUPRC) is a superior metric for model comparison to the area under the receiver operating characteristic (AUROC) for binary classification tasks with class imbalance. This paper challenges this notion through novel mathematical analysis, illustrating that AUROC and AUPRC can be concisely related in probabilistic terms. We demonstrate that AUPRC, contrary to popular belief, is not superior in cases of class imbalance and might even be a harmful metric, given its inclination to unduly favor model improvements in subpopulations with more frequent positive labels. This bias can inadvertently heighten algorithmic disparities. Prompted by these insights, a thorough review of existing ML literature was conducted, utilizing large language models to analyze over 1.5 million papers from arXiv. Our investigation focused on the prevalence and substantiation of the purported AUPRC superiority. The result
    
[^12]: 集合化的神经网络编码

    Set-based Neural Network Encoding. (arXiv:2305.16625v1 [cs.LG])

    [http://arxiv.org/abs/2305.16625](http://arxiv.org/abs/2305.16625)

    提出了一种能够集合化地编码神经网络参数的神经网络权重编码方法，并引入了一种逐层编码方案来考虑神经网络的分层计算结构。同时引入了“pad-chunk-encode”流水线进行神经网络层的高效编码处理，还提出了新的神经网络泛化性能预测任务。

    

    我们提出了一种利用集合到集合和集合到向量函数来有效编码神经网络参数，进行泛化性能预测的神经网络权重编码方法。与之前需要对不同架构编写自定义编码模型的方法不同，我们的方法能够对混合架构和不同参数大小的模型动态编码。此外，我们的 SNE（集合化神经网络编码器）通过使用一种逐层编码方案，考虑神经网络的分层计算结构。最终将所有层次编码合并到一起，以获取神经网络编码矢量。我们还引入了“pad-chunk-encode”流水线来有效地编码神经网络层，该流水线可根据计算和内存限制进行调整。我们还引入了两个用于神经网络泛化性能预测的新任务：跨数据集和架构适应性预测。

    We propose an approach to neural network weight encoding for generalization performance prediction that utilizes set-to-set and set-to-vector functions to efficiently encode neural network parameters. Our approach is capable of encoding neural networks in a modelzoo of mixed architecture and different parameter sizes as opposed to previous approaches that require custom encoding models for different architectures. Furthermore, our \textbf{S}et-based \textbf{N}eural network \textbf{E}ncoder (SNE) takes into consideration the hierarchical computational structure of neural networks by utilizing a layer-wise encoding scheme that culminates to encoding all layer-wise encodings to obtain the neural network encoding vector. Additionally, we introduce a \textit{pad-chunk-encode} pipeline to efficiently encode neural network layers that is adjustable to computational and memory constraints. We also introduce two new tasks for neural network generalization performance prediction: cross-dataset a
    
[^13]: 基于L2，0基数惩罚的不均匀图趋势过滤。

    Inhomogeneous graph trend filtering via a l2,0 cardinality penalty. (arXiv:2304.05223v1 [cs.LG])

    [http://arxiv.org/abs/2304.05223](http://arxiv.org/abs/2304.05223)

    本文提出了一种基于L2，0基数惩罚的图趋势过滤（GTF）模型，可同时进行k-means聚类和基于图的最小割，以估计在节点之间具有不均匀平滑水平的分段平滑图信号，并在降噪、支持恢复和半监督分类任务上表现更好，比现有方法更高效地处理大型数据集。

    

    我们研究了在图上估计分段平滑信号的方法，并提出了一种$\ell_{2,0}$-范数惩罚图趋势过滤（GTF）模型，以估计在节点之间具有不均匀平滑水平的分段平滑图信号。我们证明了所提出的GTF模型同时是基于节点上的信号的k-means聚类和基于图的最小割，其中聚类和割共享相同的分配矩阵。我们提出了两种方法来解决所提出的GTF模型：一种是基于谱分解的方法，另一种是基于模拟退火的方法。在合成和现实数据集的实验中，我们展示了所提出的GTF模型在降噪、支持恢复和半监督分类任务上表现更好，且比现有方法更高效地解决了大型数据集的问题。

    We study estimation of piecewise smooth signals over a graph. We propose a $\ell_{2,0}$-norm penalized Graph Trend Filtering (GTF) model to estimate piecewise smooth graph signals that exhibits inhomogeneous levels of smoothness across the nodes. We prove that the proposed GTF model is simultaneously a k-means clustering on the signal over the nodes and a minimum graph cut on the edges of the graph, where the clustering and the cut share the same assignment matrix. We propose two methods to solve the proposed GTF model: a spectral decomposition method and a method based on simulated annealing. In the experiment on synthetic and real-world datasets, we show that the proposed GTF model has a better performances compared with existing approaches on the tasks of denoising, support recovery and semi-supervised classification. We also show that the proposed GTF model can be solved more efficiently than existing models for the dataset with a large edge set.
    

