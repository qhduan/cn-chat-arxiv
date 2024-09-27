# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Stochastic Quasi-Newton Method for Non-convex Optimization with Non-uniform Smoothness](https://arxiv.org/abs/2403.15244) | 论文提出了一种针对非凸优化问题的随机拟牛顿方法，适用于具有非均匀平滑度的情况，其创新之处在于引入了$(L_0, L_1)$-平滑度，相比传统的$L$-平滑度，能更好地捕捉平滑度与梯度范数之间的正相关关系。 |
| [^2] | [Efficient Combinatorial Optimization via Heat Diffusion](https://arxiv.org/abs/2403.08757) | 通过热扩散实现了高效的组合优化，克服了现有方法在搜索全局最优时效率有限的问题。 |
| [^3] | [Optimal Parallelization Strategies for Active Flow Control in Deep Reinforcement Learning-Based Computational Fluid Dynamics](https://arxiv.org/abs/2402.11515) | 该研究专注于优化深度强化学习在流体力学中主动流控制中的并行设置，通过拆解DRL框架、进行扩展性基准测试、提出混合并行化配置并优化多环境DRL训练中的I/O操作，提出了有效的并行化策略。 |
| [^4] | [Generative Adversarial Bayesian Optimization for Surrogate Objectives](https://arxiv.org/abs/2402.06532) | 提出了生成对抗贝叶斯优化（GABO）算法，通过使用自适应源批评家正则化，将优化轨迹限制在代理函数可靠的区域内，解决了离线模型基于策略优化中代理模型预测不准确的问题。在多个离线优化任务中，GABO表现优于现有基准方法。 |
| [^5] | [Efficient Solvers for Partial Gromov-Wasserstein](https://arxiv.org/abs/2402.03664) | 本文提出了两个基于Frank-Wolfe算法的新的高效求解器来解决偏差Gromov-Wasserstein问题，并且证明了PGW问题构成了度量测度空间的度量。 |
| [^6] | [High-Quality Image Restoration Following Human Instructions.](http://arxiv.org/abs/2401.16468) | 本论文提出了一种使用人类编写的指令来指导图像恢复模型的方法，并在多个恢复任务上取得了最先进的结果，为基于文本指导的图像恢复和增强研究提供了一个新的基准。 |
| [^7] | [Enhanced Distribution Alignment for Post-Training Quantization of Diffusion Models.](http://arxiv.org/abs/2401.04585) | 本文提出了一种扩展分布对齐方法以解决后训练量化对于弥散模型的分布不匹配问题，该方法在低延迟应用中具有较高的潜力，并且能有效提升性能。 |
| [^8] | [Assumption violations in causal discovery and the robustness of score matching.](http://arxiv.org/abs/2310.13387) | 本文在不同背景条件下对最近的因果发现方法在观察性独立同分布数据上的实际性能进行了基准测试，发现基于score matching的方法在具有挑战性的场景中表现出令人惊讶的性能。 |
| [^9] | [Initialization Bias of Fourier Neural Operator: Revisiting the Edge of Chaos.](http://arxiv.org/abs/2310.06379) | 本文研究了Fourier神经操作符(FNO)的初始化偏差，提出了一种FNO版本的He初始化方案，通过模式截断和密集连接网络相似的特点，解决了训练不稳定的负初始化偏差问题。 |
| [^10] | [Discrete, compositional, and symbolic representations through attractor dynamics.](http://arxiv.org/abs/2310.01807) | 这项工作探讨了如何通过模拟吸引子动力学来更加神经可行地实现离散化，从而将连续的表示空间划分为对应于符号序列的分区。通过引入符号空间结构，可以在丰富的感知输入的吸引子支持表示空间中实现组合性。 |
| [^11] | [Learning to Receive Help: Intervention-Aware Concept Embedding Models.](http://arxiv.org/abs/2309.16928) | 这项研究提出了一种干预感知的概念嵌入模型，用于提高神经架构对概念干预的响应性，并解决了概念干预顺序和模型架构的依赖性的问题。 |
| [^12] | [EPTQ: Enhanced Post-Training Quantization via Label-Free Hessian.](http://arxiv.org/abs/2309.11531) | 本文提出了一种名为EPTQ的增强后训练量化方法，该方法通过自适应加权层和无标签Hessian近似技术实现了最先进的结果。 |
| [^13] | [Bivariate DeepKriging for Large-scale Spatial Interpolation of Wind Fields.](http://arxiv.org/abs/2307.08038) | 本文提出了一种名为双变量深度克里金的方法，它利用空间相关的深度神经网络(DNN)和嵌入层以及基于自助法和集成DNN的无分布不确定性量化方法，用于大规模空间插值风场的预测和估计。 |
| [^14] | [Realising Synthetic Active Inference Agents, Part II: Variational Message Updates.](http://arxiv.org/abs/2306.02733) | 本文讨论了解决广义自由能（FE）目标的合成主动推理代理的变分信息更新和消息传递算法，通过对T形迷宫导航任务的模拟比较，表明AIF可引起认知行为。 |
| [^15] | [Denial-of-Service or Fine-Grained Control: Towards Flexible Model Poisoning Attacks on Federated Learning.](http://arxiv.org/abs/2304.10783) | 本文提出了一种灵活的联邦学习模型毒化攻击策略，既可以实现拒绝服务(Dos)目标，也可以精确控制全局准确性，具有高效和隐形的特点。 |
| [^16] | [Bayesian Matrix Decomposition and Applications.](http://arxiv.org/abs/2302.11337) | 本书旨在介绍贝叶斯矩阵分解的概念和工具，并总结了贝叶斯矩阵分解方法在不同领域的应用。 |

# 详细

[^1]: 非凸优化的随机拟牛顿方法与非均匀平滑度

    A Stochastic Quasi-Newton Method for Non-convex Optimization with Non-uniform Smoothness

    [https://arxiv.org/abs/2403.15244](https://arxiv.org/abs/2403.15244)

    论文提出了一种针对非凸优化问题的随机拟牛顿方法，适用于具有非均匀平滑度的情况，其创新之处在于引入了$(L_0, L_1)$-平滑度，相比传统的$L$-平滑度，能更好地捕捉平滑度与梯度范数之间的正相关关系。

    

    传统优化算法的经典收敛分析依赖于广泛采用的均匀平滑度假设。然而，最近的实验研究表明，许多机器学习问题表现出非均匀平滑度，这意味着平滑度因子是模型参数的函数，而不是一个普遍常数。尤其是观察到，平滑度随着训练轨迹中的梯度范数增长。受这一现象的启发，最近引入的$(L_0, L_1)$-平滑度是一个比传统的$L$-平滑度更一般的概念，它捕捉了平滑度与梯度范数之间的这种正相关关系。在这种非均匀平滑度下，现有文献通过利用梯度裁剪技术设计了随机一阶算法，以获得找到$\epsilon$-近似解的$\mathcal{O}(\epsilon^{-3})$样本复杂度。

    arXiv:2403.15244v1 Announce Type: new  Abstract: Classical convergence analyses for optimization algorithms rely on the widely-adopted uniform smoothness assumption. However, recent experimental studies have demonstrated that many machine learning problems exhibit non-uniform smoothness, meaning the smoothness factor is a function of the model parameter instead of a universal constant. In particular, it has been observed that the smoothness grows with respect to the gradient norm along the training trajectory. Motivated by this phenomenon, the recently introduced $(L_0, L_1)$-smoothness is a more general notion, compared to traditional $L$-smoothness, that captures such positive relationship between smoothness and gradient norm. Under this type of non-uniform smoothness, existing literature has designed stochastic first-order algorithms by utilizing gradient clipping techniques to obtain the optimal $\mathcal{O}(\epsilon^{-3})$ sample complexity for finding an $\epsilon$-approximate fi
    
[^2]: 通过热扩散实现高效的组合优化

    Efficient Combinatorial Optimization via Heat Diffusion

    [https://arxiv.org/abs/2403.08757](https://arxiv.org/abs/2403.08757)

    通过热扩散实现了高效的组合优化，克服了现有方法在搜索全局最优时效率有限的问题。

    

    论文探讨了通过热扩散来实现高效的组合优化。针对现有方法只能在每次迭代中访问解空间的一小部分这一限制，提出了一种框架来解决一般的组合优化问题，并且在一系列最具挑战性和广泛遇到的组合优化中展现出卓越性能。

    arXiv:2403.08757v1 Announce Type: cross  Abstract: Combinatorial optimization problems are widespread but inherently challenging due to their discrete nature.The primary limitation of existing methods is that they can only access a small fraction of the solution space at each iteration, resulting in limited efficiency for searching the global optimal. To overcome this challenge, diverging from conventional efforts of expanding the solver's search scope, we focus on enabling information to actively propagate to the solver through heat diffusion. By transforming the target function while preserving its optima, heat diffusion facilitates information flow from distant regions to the solver, providing more efficient navigation. Utilizing heat diffusion, we propose a framework for solving general combinatorial optimization problems. The proposed methodology demonstrates superior performance across a range of the most challenging and widely encountered combinatorial optimizations. Echoing rec
    
[^3]: 深度强化学习在流体力学中主动流控制中的最佳并行化策略

    Optimal Parallelization Strategies for Active Flow Control in Deep Reinforcement Learning-Based Computational Fluid Dynamics

    [https://arxiv.org/abs/2402.11515](https://arxiv.org/abs/2402.11515)

    该研究专注于优化深度强化学习在流体力学中主动流控制中的并行设置，通过拆解DRL框架、进行扩展性基准测试、提出混合并行化配置并优化多环境DRL训练中的I/O操作，提出了有效的并行化策略。

    

    深度强化学习（DRL）已被证明是处理高动态和非线性主动流控制（AFC）问题的一种有前途的方法。然而，与训练DRL模型相关的计算成本构成了重要的性能瓶颈。为了应对这一挑战并在高性能计算架构上实现有效的扩展，本研究侧重于优化并行设置中的基于DRL的算法。我们验证了用于AFC问题的现有最先进的DRL框架，并讨论了其效率瓶颈。随后，通过拆解整体框架，并为各个组件进行广泛的可扩展性基准测试，我们研究了各种混合并行化配置，并提出了有效的并行化策略。此外，我们优化了多环境DRL训练中的输入/输出（I/O）操作，以解决与数据移动相关的关键开销。

    arXiv:2402.11515v1 Announce Type: new  Abstract: Deep Reinforcement Learning (DRL) has emerged as a promising approach for handling highly dynamic and nonlinear Active Flow Control (AFC) problems. However, the computational cost associated with training DRL models presents a significant performance bottleneck. To address this challenge and enable efficient scaling on high-performance computing architectures, this study focuses on optimizing DRL-based algorithms in parallel settings. We validate an existing state-of-the-art DRL framework used for AFC problems and discuss its efficiency bottlenecks. Subsequently, by deconstructing the overall framework and conducting extensive scalability benchmarks for individual components, we investigate various hybrid parallelization configurations and propose efficient parallelization strategies. Moreover, we refine input/output (I/O) operations in multi-environment DRL training to tackle critical overhead associated with data movement. Finally, we 
    
[^4]: 生成对抗贝叶斯优化用于代理目标

    Generative Adversarial Bayesian Optimization for Surrogate Objectives

    [https://arxiv.org/abs/2402.06532](https://arxiv.org/abs/2402.06532)

    提出了生成对抗贝叶斯优化（GABO）算法，通过使用自适应源批评家正则化，将优化轨迹限制在代理函数可靠的区域内，解决了离线模型基于策略优化中代理模型预测不准确的问题。在多个离线优化任务中，GABO表现优于现有基准方法。

    

    离线基于模型的策略优化通过在优化过程中不查询真实的目标函数来优化学习到的代理目标函数。然而，在优化过程中经常遇到代理模型预测不准确的情况。为了解决这个问题，我们提出了使用自适应源批评家正则化的生成对抗贝叶斯优化（GABO），这是一个任务不可知的贝叶斯优化框架，采用了Lipschitz有界源批评家模型来约束优化轨迹，使其在代理函数可靠的区域内。我们证明，在连续输入空间先验的一定假设下，我们的算法动态调整源批评家正则化的强度。在各种科学领域的多个离线优化任务中，GABO优于现有基准方法。我们的代码可在https://github.com/michael-s-yao/gabo 查询。

    Offline model-based policy optimization seeks to optimize a learned surrogate objective function without querying the true oracle objective during optimization. However, inaccurate surrogate model predictions are frequently encountered along the optimization trajectory. To address this limitation, we propose generative adversarial Bayesian optimization (GABO) using adaptive source critic regularization, a task-agnostic framework for Bayesian optimization that employs a Lipschitz-bounded source critic model to constrain the optimization trajectory to regions where the surrogate function is reliable. We show that under certain assumptions for the continuous input space prior, our algorithm dynamically adjusts the strength of the source critic regularization. GABO outperforms existing baselines on a number of different offline optimization tasks across a variety of scientific domains. Our code is available at https://github.com/michael-s-yao/gabo
    
[^5]: 高效求解偏差Gromov-Wasserstein问题

    Efficient Solvers for Partial Gromov-Wasserstein

    [https://arxiv.org/abs/2402.03664](https://arxiv.org/abs/2402.03664)

    本文提出了两个基于Frank-Wolfe算法的新的高效求解器来解决偏差Gromov-Wasserstein问题，并且证明了PGW问题构成了度量测度空间的度量。

    

    偏差Gromov-Wasserstein（PGW）问题可以比较具有不均匀质量的度量空间中的测度，从而实现这些空间之间的不平衡和部分匹配。本文证明了PGW问题可以转化为Gromov-Wasserstein问题的一个变种，类似于把偏差最优运输问题转化为最优运输问题。这个转化导致了两个新的求解器，基于Frank-Wolfe算法，数学和计算上等价，提供了高效的PGW问题解决方案。我们进一步证明了PGW问题构成了度量测度空间的度量。最后，我们通过与现有基线方法在形状匹配和正样本未标记学习问题上的计算时间和性能比较，验证了我们提出的求解器的有效性。

    The partial Gromov-Wasserstein (PGW) problem facilitates the comparison of measures with unequal masses residing in potentially distinct metric spaces, thereby enabling unbalanced and partial matching across these spaces. In this paper, we demonstrate that the PGW problem can be transformed into a variant of the Gromov-Wasserstein problem, akin to the conversion of the partial optimal transport problem into an optimal transport problem. This transformation leads to two new solvers, mathematically and computationally equivalent, based on the Frank-Wolfe algorithm, that provide efficient solutions to the PGW problem. We further establish that the PGW problem constitutes a metric for metric measure spaces. Finally, we validate the effectiveness of our proposed solvers in terms of computation time and performance on shape-matching and positive-unlabeled learning problems, comparing them against existing baselines.
    
[^6]: 遵循人类指令的高质量图像恢复

    High-Quality Image Restoration Following Human Instructions. (arXiv:2401.16468v1 [cs.CV])

    [http://arxiv.org/abs/2401.16468](http://arxiv.org/abs/2401.16468)

    本论文提出了一种使用人类编写的指令来指导图像恢复模型的方法，并在多个恢复任务上取得了最先进的结果，为基于文本指导的图像恢复和增强研究提供了一个新的基准。

    

    图像恢复是一个基本问题，涉及从退化观测中恢复出高质量的干净图像。全能图像恢复模型可以通过使用特定于退化类型的信息作为提示来有效地恢复各种类型和级别的退化图像，并引导恢复模型。我们提出了一种使用人类编写的指令来指导图像恢复模型的方法。在给定自然语言提示的情况下，我们的模型可以从退化图像中恢复出高质量的图像，并考虑多种退化类型。我们的方法InstructIR在图像去噪、雨水去除、去模糊、去雾和(低光)图像增强等多个恢复任务上取得了最先进的结果。InstructIR在之前的全能恢复方法上提高了1dB。此外，我们的数据集和结果为基于文本指导的图像恢复和增强的新研究提供了一个新的基准。我们提供了代码、数据集和模型。

    Image restoration is a fundamental problem that involves recovering a high-quality clean image from its degraded observation. All-In-One image restoration models can effectively restore images from various types and levels of degradation using degradation-specific information as prompts to guide the restoration model. In this work, we present the first approach that uses human-written instructions to guide the image restoration model. Given natural language prompts, our model can recover high-quality images from their degraded counterparts, considering multiple degradation types. Our method, InstructIR, achieves state-of-the-art results on several restoration tasks including image denoising, deraining, deblurring, dehazing, and (low-light) image enhancement. InstructIR improves +1dB over previous all-in-one restoration methods. Moreover, our dataset and results represent a novel benchmark for new research on text-guided image restoration and enhancement. Our code, datasets and models a
    
[^7]: 扩展分布对齐来实现弥散模型的后训练量化

    Enhanced Distribution Alignment for Post-Training Quantization of Diffusion Models. (arXiv:2401.04585v1 [cs.CV])

    [http://arxiv.org/abs/2401.04585](http://arxiv.org/abs/2401.04585)

    本文提出了一种扩展分布对齐方法以解决后训练量化对于弥散模型的分布不匹配问题，该方法在低延迟应用中具有较高的潜力，并且能有效提升性能。

    

    通过迭代噪声估计，扩散模型在图像生成任务中取得了巨大成功。然而，繁重的去噪过程和复杂的神经网络阻碍了它们在实际场景中的低延迟应用。量化可以有效降低模型复杂度，而后训练量化(PTQ)在加速去噪过程方面具有很高的潜力，并且不需要微调。不幸的是，我们发现由于不同去噪步骤中激活的高度动态分布，现有的扩散模型的PTQ方法在校准样本和重构输出两个层面上都存在分布不匹配的问题，导致性能远低于令人满意的水平，特别是在低位情况下。在本文中，我们提出了增强的分布对齐用于弥散模型的后训练量化(EDA-DM)来解决上述问题。具体来说，在校准样本层面，我们基于...[缺省]

    Diffusion models have achieved great success in image generation tasks through iterative noise estimation. However, the heavy denoising process and complex neural networks hinder their low-latency applications in real-world scenarios. Quantization can effectively reduce model complexity, and post-training quantization (PTQ), which does not require fine-tuning, is highly promising in accelerating the denoising process. Unfortunately, we find that due to the highly dynamic distribution of activations in different denoising steps, existing PTQ methods for diffusion models suffer from distribution mismatch issues at both calibration sample level and reconstruction output level, which makes the performance far from satisfactory, especially in low-bit cases. In this paper, we propose Enhanced Distribution Alignment for Post-Training Quantization of Diffusion Models (EDA-DM) to address the above issues. Specifically, at the calibration sample level, we select calibration samples based on the 
    
[^8]: 因果发现中的假设违规和score matching的鲁棒性

    Assumption violations in causal discovery and the robustness of score matching. (arXiv:2310.13387v1 [stat.ME])

    [http://arxiv.org/abs/2310.13387](http://arxiv.org/abs/2310.13387)

    本文在不同背景条件下对最近的因果发现方法在观察性独立同分布数据上的实际性能进行了基准测试，发现基于score matching的方法在具有挑战性的场景中表现出令人惊讶的性能。

    

    当领域知识有限且实验受到道德、财务或时间限制时，从业者会转向观察性因果发现方法来恢复因果结构，利用其数据的统计特性。由于没有进一步的假设，因果发现是一个不适定的问题，每个算法都有其自己的一套通常无法验证的假设，其中一些在真实数据集中很难满足。鉴于这些考虑，本文在不同背景条件下对最近的因果发现方法在观察性独立同分布数据上的实际性能进行了广泛的基准测试，允许违反每个选定方法所需的关键假设。我们的实验结果显示，在这些具有挑战性场景中，基于score matching的方法在推断的图的误报与漏报率方面表现出令人惊讶的性能，并提供了对其的理论洞察。

    When domain knowledge is limited and experimentation is restricted by ethical, financial, or time constraints, practitioners turn to observational causal discovery methods to recover the causal structure, exploiting the statistical properties of their data. Because causal discovery without further assumptions is an ill-posed problem, each algorithm comes with its own set of usually untestable assumptions, some of which are hard to meet in real datasets. Motivated by these considerations, this paper extensively benchmarks the empirical performance of recent causal discovery methods on observational i.i.d. data generated under different background conditions, allowing for violations of the critical assumptions required by each selected approach. Our experimental findings show that score matching-based methods demonstrate surprising performance in the false positive and false negative rate of the inferred graph in these challenging scenarios, and we provide theoretical insights into their
    
[^9]: Fourier神经操作符的初始化偏差：重新审视混沌边缘

    Initialization Bias of Fourier Neural Operator: Revisiting the Edge of Chaos. (arXiv:2310.06379v1 [cs.LG])

    [http://arxiv.org/abs/2310.06379](http://arxiv.org/abs/2310.06379)

    本文研究了Fourier神经操作符(FNO)的初始化偏差，提出了一种FNO版本的He初始化方案，通过模式截断和密集连接网络相似的特点，解决了训练不稳定的负初始化偏差问题。

    

    本文研究了Fourier神经操作符(FNO)的初始化偏差。建立了一个针对FNO的平均场理论，从“混沌边缘”的视角分析了随机FNO的行为。我们揭示了前向和反向传播行为表现出与FNO独特的特征，这是由模式截断引起的，同时也展示了与密集连接网络相似的特点。基于这一观察，我们还提出了一种FNO版本的He初始化方案，以减轻导致训练不稳定的负初始化偏差。实验结果显示了我们初始化方案的有效性，使得32层FNO的训练稳定，无需额外技术或显著性能下降。

    This paper investigates the initialization bias of the Fourier neural operator (FNO). A mean-field theory for FNO is established, analyzing the behavior of the random FNO from an ``edge of chaos'' perspective. We uncover that the forward and backward propagation behaviors exhibit characteristics unique to FNO, induced by mode truncation, while also showcasing similarities to those of densely connected networks. Building upon this observation, we also propose a FNO version of the He initialization scheme to mitigate the negative initialization bias leading to training instability. Experimental results demonstrate the effectiveness of our initialization scheme, enabling stable training of a 32-layer FNO without the need for additional techniques or significant performance degradation.
    
[^10]: 通过吸引子动力学实现离散、组合和符号表示

    Discrete, compositional, and symbolic representations through attractor dynamics. (arXiv:2310.01807v1 [cs.AI])

    [http://arxiv.org/abs/2310.01807](http://arxiv.org/abs/2310.01807)

    这项工作探讨了如何通过模拟吸引子动力学来更加神经可行地实现离散化，从而将连续的表示空间划分为对应于符号序列的分区。通过引入符号空间结构，可以在丰富的感知输入的吸引子支持表示空间中实现组合性。

    

    组合性是离散符号系统（如语言和程序）的重要特征，它使得这些系统尽管使用有限的符号集合，但仍具有无限的容量。它在认知科学和人工智能领域的推理中都具有很好的抽象性。然而，连续和符号处理之间的界面通常是通过算法级别上的量化或softmax采样步骤来实现的。在本研究中，我们通过模拟吸引子动力学将离散化实现得更加神经可行，这种方法将连续的表示空间划分为对应于符号序列的分区。在吸引子网络的基础上，引入了新的训练方法，我们展示了在丰富的感知输入的吸引子支持表示空间中引入符号空间结构可以产生组合性。最后，我们认为我们的模型展示了一种信息增长的过程。

    Compositionality is an important feature of discrete symbolic systems, such as language and programs, as it enables them to have infinite capacity despite a finite symbol set. It serves as a useful abstraction for reasoning in both cognitive science and in AI, yet the interface between continuous and symbolic processing is often imposed by fiat at the algorithmic level, such as by means of quantization or a softmax sampling step. In this work, we explore how discretization could be implemented in a more neurally plausible manner through the modeling of attractor dynamics that partition the continuous representation space into basins that correspond to sequences of symbols. Building on established work in attractor networks and introducing novel training methods, we show that imposing structure in the symbolic space can produce compositionality in the attractor-supported representation space of rich sensory inputs. Lastly, we argue that our model exhibits the process of an information b
    
[^11]: 学习接受帮助：干预感知的概念嵌入模型

    Learning to Receive Help: Intervention-Aware Concept Embedding Models. (arXiv:2309.16928v1 [cs.LG])

    [http://arxiv.org/abs/2309.16928](http://arxiv.org/abs/2309.16928)

    这项研究提出了一种干预感知的概念嵌入模型，用于提高神经架构对概念干预的响应性，并解决了概念干预顺序和模型架构的依赖性的问题。

    

    概念瓶颈模型（CBMs）通过使用一组高级概念构建和解释神经架构的预测，以解决其不透明性的问题。这些模型的一个特殊属性是它们允许概念干预，用户可以纠正被错误预测的概念，从而提高模型的性能。然而，最近的研究表明，干预有效性可能严重依赖于干预概念的顺序以及模型的架构和训练超参数。我们认为，这源于CBM在训练时缺乏模型适应概念干预的激励。为了解决这个问题，我们提出了干预感知的概念嵌入模型（IntCEMs），这是一种基于CBM的新型架构和训练范式，可以提高模型对测试时干预的响应性。我们的模型以端到端的方式学习了一个概念干预策略，从中可以采样有意义的干预轨迹。

    Concept Bottleneck Models (CBMs) tackle the opacity of neural architectures by constructing and explaining their predictions using a set of high-level concepts. A special property of these models is that they permit concept interventions, wherein users can correct mispredicted concepts and thus improve the model's performance. Recent work, however, has shown that intervention efficacy can be highly dependent on the order in which concepts are intervened on and on the model's architecture and training hyperparameters. We argue that this is rooted in a CBM's lack of train-time incentives for the model to be appropriately receptive to concept interventions. To address this, we propose Intervention-aware Concept Embedding models (IntCEMs), a novel CBM-based architecture and training paradigm that improves a model's receptiveness to test-time interventions. Our model learns a concept intervention policy in an end-to-end fashion from where it can sample meaningful intervention trajectories a
    
[^12]: EPTQ:通过无标签Hessian增强的后训练量化

    EPTQ: Enhanced Post-Training Quantization via Label-Free Hessian. (arXiv:2309.11531v1 [cs.CV])

    [http://arxiv.org/abs/2309.11531](http://arxiv.org/abs/2309.11531)

    本文提出了一种名为EPTQ的增强后训练量化方法，该方法通过自适应加权层和无标签Hessian近似技术实现了最先进的结果。

    

    深度神经网络的量化已成为将这些网络嵌入到最终用户设备上的关键要素。然而，当前的量化方法通常会导致准确性严重下降。本文提出了一种名为EPTQ的增强后训练量化方法。该方法基于知识蒸馏，并采用自适应加权层的方式。此外，我们提出了一种新的无标签Hessian近似技术，名为Label-Free Hessian。这种技术消除了计算Hessian所需的标记数据集的要求。自适应知识蒸馏利用Label-Free Hessian技术，在进行优化时更加关注模型的敏感部分。通过使用EPTQ，我们在各种模型、任务和数据集上实现了最先进的结果，包括ImageNet分类、COCO目标检测和用于语义分割的Pascal-VOC数据集。

    Quantization of deep neural networks (DNN) has become a key element in the efforts of embedding such networks on end-user devices. However, current quantization methods usually suffer from costly accuracy degradation. In this paper, we propose a new method for Enhanced Post Training Quantization named EPTQ. The method is based on knowledge distillation with an adaptive weighting of layers. In addition, we introduce a new label-free technique for approximating the Hessian trace of the task loss, named Label-Free Hessian. This technique removes the requirement of a labeled dataset for computing the Hessian. The adaptive knowledge distillation uses the Label-Free Hessian technique to give greater attention to the sensitive parts of the model while performing the optimization. Empirically, by employing EPTQ we achieve state-of-the-art results on a wide variety of models, tasks, and datasets, including ImageNet classification, COCO object detection, and Pascal-VOC for semantic segmentation.
    
[^13]: 大规模空间插值风场的双变量深度克里金方法

    Bivariate DeepKriging for Large-scale Spatial Interpolation of Wind Fields. (arXiv:2307.08038v1 [stat.ML])

    [http://arxiv.org/abs/2307.08038](http://arxiv.org/abs/2307.08038)

    本文提出了一种名为双变量深度克里金的方法，它利用空间相关的深度神经网络(DNN)和嵌入层以及基于自助法和集成DNN的无分布不确定性量化方法，用于大规模空间插值风场的预测和估计。

    

    高空间分辨率的风场数据对于气候、海洋和气象研究中的各种应用至关重要。由于风数据往往具有非高斯分布、高空间变异性和异质性，因此对具有两个维度速度的双变量风场进行大规模空间插值或下缩放是一项具有挑战性的任务。在空间统计学中，常用cokriging来预测双变量空间场。然而，cokriging预测器除了对高斯过程有效外，并不是最优的。此外，对于大型数据集，cokriging计算量巨大。在本文中，我们提出了一种称为双变量深度克里金的方法，它是一个由空间径向基函数构建的空间相关的深度神经网络(DNN)和嵌入层，用于双变量空间数据预测。然后，我们基于自助法和集成DNN开发了一种无分布不确定性量化方法。我们提出的方法优于传统的cokriging方法。

    High spatial resolution wind data are essential for a wide range of applications in climate, oceanographic and meteorological studies. Large-scale spatial interpolation or downscaling of bivariate wind fields having velocity in two dimensions is a challenging task because wind data tend to be non-Gaussian with high spatial variability and heterogeneity. In spatial statistics, cokriging is commonly used for predicting bivariate spatial fields. However, the cokriging predictor is not optimal except for Gaussian processes. Additionally, cokriging is computationally prohibitive for large datasets. In this paper, we propose a method, called bivariate DeepKriging, which is a spatially dependent deep neural network (DNN) with an embedding layer constructed by spatial radial basis functions for bivariate spatial data prediction. We then develop a distribution-free uncertainty quantification method based on bootstrap and ensemble DNN. Our proposed approach outperforms the traditional cokriging 
    
[^14]: 实现合成主动推理代理，第二部分：变分信息更新

    Realising Synthetic Active Inference Agents, Part II: Variational Message Updates. (arXiv:2306.02733v1 [stat.ML])

    [http://arxiv.org/abs/2306.02733](http://arxiv.org/abs/2306.02733)

    本文讨论了解决广义自由能（FE）目标的合成主动推理代理的变分信息更新和消息传递算法，通过对T形迷宫导航任务的模拟比较，表明AIF可引起认知行为。

    

    自由能原理（FEP）描述生物代理通过相应环境的生成模型最小化变分自由能（FE）。主动推理（AIF）是FEP的推论，描述了代理人通过最小化期望的FE目标来探索和利用其环境。在两篇相关论文中，我们通过自由形式Forney-style因子图（FFG）上的消息传递，描述了一种可扩展的合成AIF代理的认知方法。本文（第二部分）根据变分演算法，导出了最小化CFFG上（广义）FE目标的消息传递算法。比较了模拟Bethe和广义FE代理之间的差异，说明了合成AIF如何在T形迷宫导航任务上引起认知行为。通过对合成AIF代理的完整消息传递描述，可以推导和重用该代理在不同环境下的行为。

    The Free Energy Principle (FEP) describes (biological) agents as minimising a variational Free Energy (FE) with respect to a generative model of their environment. Active Inference (AIF) is a corollary of the FEP that describes how agents explore and exploit their environment by minimising an expected FE objective. In two related papers, we describe a scalable, epistemic approach to synthetic AIF agents, by message passing on free-form Forney-style Factor Graphs (FFGs). A companion paper (part I) introduces a Constrained FFG (CFFG) notation that visually represents (generalised) FE objectives for AIF. The current paper (part II) derives message passing algorithms that minimise (generalised) FE objectives on a CFFG by variational calculus. A comparison between simulated Bethe and generalised FE agents illustrates how synthetic AIF induces epistemic behaviour on a T-maze navigation task. With a full message passing account of synthetic AIF agents, it becomes possible to derive and reuse 
    
[^15]: 拒绝服务或细粒度控制：面向联邦学习的灵活模型毒化攻击

    Denial-of-Service or Fine-Grained Control: Towards Flexible Model Poisoning Attacks on Federated Learning. (arXiv:2304.10783v1 [cs.LG])

    [http://arxiv.org/abs/2304.10783](http://arxiv.org/abs/2304.10783)

    本文提出了一种灵活的联邦学习模型毒化攻击策略，既可以实现拒绝服务(Dos)目标，也可以精确控制全局准确性，具有高效和隐形的特点。

    

    联邦学习容易受到毒化攻击，敌对方会破坏全局聚合结果并造成拒绝服务。本文提出了一种灵活模型毒化攻击(FMPA)，旨在实现多功能攻击目标。本文考虑如下实际情景：敌对方没有关于FL系统的额外信息（例如，聚合规则或良性设备上的更新）。FMPA利用全局历史信息构建估计器，将下一轮全局模型预测为良性参考模型，并微调参考模型以获得所需的精度低和扰动小的毒化模型。FMPA不仅可以达到DoS的目标，还可以自然地扩展到启动细粒度可控攻击，从而精确降低全局准确性。本文进一步探索了FMPA在几种FL场景下的攻击性能，包括二元分类和图像分类，在不同的攻击目标和攻击知识水平下。实验结果表明，FMPA可以有效而高效地实现所需的攻击目标，同时保持隐形和不可感知。

    Federated learning (FL) is vulnerable to poisoning attacks, where adversaries corrupt the global aggregation results and cause denial-of-service (DoS). Unlike recent model poisoning attacks that optimize the amplitude of malicious perturbations along certain prescribed directions to cause DoS, we propose a Flexible Model Poisoning Attack (FMPA) that can achieve versatile attack goals. We consider a practical threat scenario where no extra knowledge about the FL system (e.g., aggregation rules or updates on benign devices) is available to adversaries. FMPA exploits the global historical information to construct an estimator that predicts the next round of the global model as a benign reference. It then fine-tunes the reference model to obtain the desired poisoned model with low accuracy and small perturbations. Besides the goal of causing DoS, FMPA can be naturally extended to launch a fine-grained controllable attack, making it possible to precisely reduce the global accuracy. Armed wi
    
[^16]: 贝叶斯矩阵分解及应用

    Bayesian Matrix Decomposition and Applications. (arXiv:2302.11337v2 [math.NA] UPDATED)

    [http://arxiv.org/abs/2302.11337](http://arxiv.org/abs/2302.11337)

    本书旨在介绍贝叶斯矩阵分解的概念和工具，并总结了贝叶斯矩阵分解方法在不同领域的应用。

    

    本书的唯一目的是为了给出贝叶斯矩阵分解概念和数学工具的自包含介绍，以便在后续章节中无缝引入矩阵分解技术及其应用。然而，我们清楚地意识到我们无法覆盖关于贝叶斯矩阵分解的所有有用和有趣的结果，并且由于讨论的范围有限，例如分析变分推理以进行优化的分离分析。我们将读者引导到贝叶斯分析领域的文献中，以便更详细地介绍相关领域。本书主要总结了重要的贝叶斯矩阵分解方法（例如实值分解、非负矩阵分解、贝叶斯插值分解）的目的和意义，以及这些方法的起源和复杂性对其应用提供的启示。数学先决条件是第一门课程。

    The sole aim of this book is to give a self-contained introduction to concepts and mathematical tools in Bayesian matrix decomposition in order to seamlessly introduce matrix decomposition techniques and their applications in subsequent sections. However, we clearly realize our inability to cover all the useful and interesting results concerning Bayesian matrix decomposition and given the paucity of scope to present this discussion, e.g., the separated analysis of variational inference for conducting the optimization. We refer the reader to literature in the field of Bayesian analysis for a more detailed introduction to the related fields.  This book is primarily a summary of purpose, significance of important Bayesian matrix decomposition methods, e.g., real-valued decomposition, nonnegative matrix factorization, Bayesian interpolative decomposition, and the origin and complexity of the methods which shed light on their applications. The mathematical prerequisite is a first course in 
    

