# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Linear Attention Sequence Parallelism](https://arxiv.org/abs/2404.02882) | 提出了一种名为线性注意力序列并行（LASP）的高效序列并行方法，针对线性注意力的语言模型进行了优化，通过设计高效的点对点通信机制和执行内核融合来降低通信开销，并实现硬件友好性。 |
| [^2] | [On the nonconvexity of some push-forward constraints and its consequences in machine learning](https://arxiv.org/abs/2403.07471) | 本文提供了关于推进约束的非凸性的理论见解，并展示了这对相关学习问题的影响。 |
| [^3] | [Towards Principled Task Grouping for Multi-Task Learning](https://arxiv.org/abs/2402.15328) | 提出了一种为多任务学习建立基于原则的任务分组方法，该方法在理论和实践上具有优势，通过灵活的数学规划形式解决了资源约束问题。 |
| [^4] | [Wavelet Analysis of Noninvasive EEG Signals Discriminates Complex and Natural Grasp Types](https://arxiv.org/abs/2402.09447) | 该研究使用小波分析技术对非侵入性脑电图信号进行解码，成功区分复杂和自然的抓握类型，并且证明了小波特征在基于脑电图的抓握区分中的有效性。 |
| [^5] | [Can Authorship Attribution Models Distinguish Speakers in Speech Transcripts?](https://arxiv.org/abs/2311.07564) | 本文研究了作者归属模型在演讲文本中区分发言人的能力，并提出了以会话演讲文本为重点的发言人归属基准。 |
| [^6] | [A Survey of Federated Unlearning: A Taxonomy, Challenges and Future Directions](https://arxiv.org/abs/2310.19218) | 联邦遗忘（FU）是解决联邦学习（FL）中数据隐私问题的战略解决方案。在发展FU方法时需要平衡隐私、安全、效用和效率的竞争性要求，以维持FL系统的效果和可用性。 |
| [^7] | [Solving Kernel Ridge Regression with Gradient Descent for a Non-Constant Kernel.](http://arxiv.org/abs/2311.01762) | 本文研究了使用梯度下降法解决非常数核的核岭回归。通过在训练过程中逐渐减小带宽，避免了超参数选择的需求，并提出了一种带宽更新方案，证明了其优于使用常数带宽的方法。 |
| [^8] | [A Stability Principle for Learning under Non-Stationarity.](http://arxiv.org/abs/2310.18304) | 本研究提出了一个适用于非稳态环境的统计学习框架，通过应用稳定性原则选择回溯窗口来最大化历史数据利用，并保持累积偏差在可接受范围内。该方法展示了对未知非稳态的适应性，遗憾界在强凸或满足Lipschitz条件下是极小化的最优解。该研究的创新点是函数相似度度量和非稳态数据序列划分技术。 |
| [^9] | [High-dimensional SGD aligns with emerging outlier eigenspaces.](http://arxiv.org/abs/2310.03010) | 本研究通过研究训练动态和经验海森矩阵以及梯度矩阵的谱的联合演化，证明了在高维混合和多层神经网络的分类任务中，SGD轨迹与海森矩阵和梯度矩阵的新兴低秩异常特征空间吻合。在多层设置中，这种对齐会在每一层发生，并且在收敛到亚优分类器时会表现出秩缺乏。 |
| [^10] | [Computing excited states of molecules using normalizing flows.](http://arxiv.org/abs/2308.16468) | 使用规范化流计算分子的激发态，通过逼近波函数并优化基函数的线性空间内的近似。该方法在计算量子系统中取得了准确和有效的结果，并在能量预测准确性和基组收敛速度方面进行了显著改善。 |
| [^11] | [Auditing Fairness by Betting.](http://arxiv.org/abs/2305.17570) | 本文提供了一种通过赌博的方式进行公平性审计的方法，相比之前的方法，这种方法具有更高的实用性和效率，能够对不断产生的数据进行连续的监控，并处理因分布漂移导致的公平性问题。 |
| [^12] | [Practitioner Motives to Select Hyperparameter Optimization Methods.](http://arxiv.org/abs/2203.01717) | 研究探讨了机器学习从业者选择超参数优化方法的动机，结果表明这基于个人目标和背景因素，调查还给出了优化模型的六个主要目标。 |

# 详细

[^1]: 线性注意力序列并行化

    Linear Attention Sequence Parallelism

    [https://arxiv.org/abs/2404.02882](https://arxiv.org/abs/2404.02882)

    提出了一种名为线性注意力序列并行（LASP）的高效序列并行方法，针对线性注意力的语言模型进行了优化，通过设计高效的点对点通信机制和执行内核融合来降低通信开销，并实现硬件友好性。

    

    序列并行（SP）作为一种处理超出单个GPU内存限制的长序列的流行策略。然而，现有的SP方法并未利用线性注意力特性，导致在基于线性注意力的语言模型中并行效率和可用性不佳。在本文中，我们介绍了线性注意力序列并行（LASP），这是一种专为基于线性注意力的语言模型量身定制的高效SP方法。具体来说，我们设计了一种高效的点对点通信机制，以利用线性注意力的右乘内核技巧，从而显着降低SP的通信开销。我们还通过执行内核融合和中间状态缓存来增强LASP的实际效率，使LASP在GPU集群上的硬件友好性得到提升。此外，我们还精心确保序列级LASP与所有类型的批级数据兼容。

    arXiv:2404.02882v1 Announce Type: cross  Abstract: Sequence Parallel (SP) serves as a prevalent strategy to handle long sequences that exceed the memory limit of a single GPU. However, existing SP methods do not take advantage of linear attention features, resulting in sub-optimal parallelism efficiency and usability for linear attention-based language models. In this paper, we introduce Linear Attention Sequence Parallel (LASP), an efficient SP method tailored to linear attention-based language models. Specifically, we design an efficient point-to-point communication mechanism to leverage the right-product kernel trick of linear attention, which sharply decreases the communication overhead of SP. We also enhance the practical efficiency of LASP by performing kernel fusion and intermediate state caching, making the implementation of LASP hardware-friendly on GPU clusters. Furthermore, we meticulously ensure the compatibility of sequence-level LASP with all types of batch-level data par
    
[^2]: 有关某些推进约束的非凸性及其在机器学习中的影响

    On the nonconvexity of some push-forward constraints and its consequences in machine learning

    [https://arxiv.org/abs/2403.07471](https://arxiv.org/abs/2403.07471)

    本文提供了关于推进约束的非凸性的理论见解，并展示了这对相关学习问题的影响。

    

    push-forward操作使人能够通过确定性映射重新分配概率测度。它在统计和优化中起着关键作用：许多学习问题（特别是来自最优输运、生成建模和算法公平性的问题）包括作为模型上的推进条件或处罚的约束。然而，文献缺乏关于这些约束的（非）凸性及其对相关学习问题的影响的一般理论见解。本文旨在填补这一空白。在第一部分中，我们提供了两组函数（将一个概率测度传输到另一个的映射；诱导不同概率测度之间相等输出分布的映射）的（非）凸性的一系列充分必要条件。这突出了对于大多数概率测度而言，这些推进约束是非凸的。在接下来，我们展示了这一结果如何暗示

    arXiv:2403.07471v1 Announce Type: cross  Abstract: The push-forward operation enables one to redistribute a probability measure through a deterministic map. It plays a key role in statistics and optimization: many learning problems (notably from optimal transport, generative modeling, and algorithmic fairness) include constraints or penalties framed as push-forward conditions on the model. However, the literature lacks general theoretical insights on the (non)convexity of such constraints and its consequences on the associated learning problems. This paper aims at filling this gap. In a first part, we provide a range of sufficient and necessary conditions for the (non)convexity of two sets of functions: the maps transporting one probability measure to another; the maps inducing equal output distributions across distinct probability measures. This highlights that for most probability measures, these push-forward constraints are not convex. In a second time, we show how this result impli
    
[^3]: 为多任务学习建立基于原则的任务分组方法

    Towards Principled Task Grouping for Multi-Task Learning

    [https://arxiv.org/abs/2402.15328](https://arxiv.org/abs/2402.15328)

    提出了一种为多任务学习建立基于原则的任务分组方法，该方法在理论和实践上具有优势，通过灵活的数学规划形式解决了资源约束问题。

    

    本文提出了一种新颖的多任务学习（MTL）中任务分组的方法，超越了现有方法，解决了关键的理论和实际限制。与之前的研究不同，我们的方法提供了一个更具理论基础的方法，不依赖于构建转移增益的限制性假设。我们还提出了一种灵活的数学规划形式，可以适应各种资源约束，从而增强了其多功能性。在各种领域进行的实验结果，包括计算机视觉数据集、组合优化基准和时间序列任务，证明了我们的方法相对于广泛的基线方法的优越性，验证了其在MTL中的有效性和普适性。

    arXiv:2402.15328v1 Announce Type: new  Abstract: This paper presents a novel approach to task grouping in Multitask Learning (MTL), advancing beyond existing methods by addressing key theoretical and practical limitations. Unlike prior studies, our approach offers a more theoretically grounded method that does not rely on restrictive assumptions for constructing transfer gains. We also propose a flexible mathematical programming formulation which can accommodate a wide spectrum of resource constraints, thus enhancing its versatility. Experimental results across diverse domains, including computer vision datasets, combinatorial optimization benchmarks and time series tasks, demonstrate the superiority of our method over extensive baselines, validating its effectiveness and general applicability in MTL.
    
[^4]: 非侵入性脑电图信号的小波分析区分复杂和自然的抓握类型

    Wavelet Analysis of Noninvasive EEG Signals Discriminates Complex and Natural Grasp Types

    [https://arxiv.org/abs/2402.09447](https://arxiv.org/abs/2402.09447)

    该研究使用小波分析技术对非侵入性脑电图信号进行解码，成功区分复杂和自然的抓握类型，并且证明了小波特征在基于脑电图的抓握区分中的有效性。

    

    该研究旨在通过对脑电图（EEG）信号进行解码，为灵巧的神经假肢开发和脑机接口（BCI）应用来区分手部抓握，特别是针对运动障碍患者。具体而言，它专注于使用一种新的基于脑电图的BCI平台和小波信号处理，区分两种复杂的自然力量和精确抓握类型以及一种中立条件作为无运动条件。小波分析涉及从小波能量系数生成时间频率和拓扑图。然后，通过使用机器学习技术和新型小波特征，我们实现了高平均准确率：多类别为85.16%，无运动 vs 力量为95.37%，无运动 vs 精确为95.40%，力量 vs 精确为88.07%，证明了这些特征在基于脑电图的抓握区分中的有效性。与先前的研究不同，我们研究的关键部分是排列特征重要性的部分。

    arXiv:2402.09447v1 Announce Type: cross  Abstract: This research aims to decode hand grasps from Electroencephalograms (EEGs) for dexterous neuroprosthetic development and Brain-Computer Interface (BCI) applications, especially for patients with motor disorders. Particularly, it focuses on distinguishing two complex natural power and precision grasps in addition to a neutral condition as a no-movement condition using a new EEG-based BCI platform and wavelet signal processing. Wavelet analysis involved generating time-frequency and topographic maps from wavelet power coefficients. Then, by using machine learning techniques with novel wavelet features, we achieved high average accuracies: 85.16% for multiclass, 95.37% for No-Movement vs Power, 95.40% for No-Movement vs Precision, and 88.07% for Power vs Precision, demonstrating the effectiveness of these features in EEG-based grasp differentiation. In contrast to previous studies, a critical part of our study was permutation feature impo
    
[^5]: 作者归属模型能否区分演讲文本中的发言人？

    Can Authorship Attribution Models Distinguish Speakers in Speech Transcripts?

    [https://arxiv.org/abs/2311.07564](https://arxiv.org/abs/2311.07564)

    本文研究了作者归属模型在演讲文本中区分发言人的能力，并提出了以会话演讲文本为重点的发言人归属基准。

    

    作者归属验证是确定两个不同书面样本是否同属一作者的任务，通常涉及对书面文本的归因。本文探讨了转录演讲的归属问题，这带来了新的挑战。其中一个主要挑战是，许多文体特征，如标点和大写，在这种情境下并不具备信息量。另一方面，转录的演讲呈现其他模式，如填充词和回应性声音（例如“嗯”，“嗯，嗯”），这些可能是不同发言人的特征性表现。我们提出了一个新的以会话演讲文本为重点的发言人归属基准。为了限制发言人与话题之间的虚假关联，我们使用会话提示和参与同一对话的发言人构建不同难度的验证试验。通过比较一系列方法，在这一新基准上建立了最新技术水平。

    arXiv:2311.07564v2 Announce Type: replace  Abstract: Authorship verification is the task of determining if two distinct writing samples share the same author and is typically concerned with the attribution of written text. In this paper, we explore the attribution of transcribed speech, which poses novel challenges. The main challenge is that many stylistic features, such as punctuation and capitalization, are not informative in this setting. On the other hand, transcribed speech exhibits other patterns, such as filler words and backchannels (e.g., 'um', 'uh-huh'), which may be characteristic of different speakers. We propose a new benchmark for speaker attribution focused on conversational speech transcripts. To limit spurious associations of speakers with topic, we employ both conversation prompts and speakers participating in the same conversation to construct verification trials of varying difficulties. We establish the state of the art on this new benchmark by comparing a suite of
    
[^6]: 联邦遗忘的综述：分类、挑战和未来方向

    A Survey of Federated Unlearning: A Taxonomy, Challenges and Future Directions

    [https://arxiv.org/abs/2310.19218](https://arxiv.org/abs/2310.19218)

    联邦遗忘（FU）是解决联邦学习（FL）中数据隐私问题的战略解决方案。在发展FU方法时需要平衡隐私、安全、效用和效率的竞争性要求，以维持FL系统的效果和可用性。

    

    随着隐私保护的联邦学习（FL）的发展，对实现被遗忘权的需求越来越大。由于FL的分散性质，实施选择性遗忘尤其具有挑战性。这种复杂性催生了一个新的领域，即联邦遗忘（FU）。FU作为解决数据隐私需求的战略解决方案，包括实施“被遗忘权”。开发FU方法的主要挑战在于在隐私、安全、效用和效率之间取得平衡，因为这些因素往往具有竞争性要求。在保持FL系统的有效性和可用性的同时，实现这些方面的最佳平衡对于遵守隐私和安全标准至关重要。本综述对现有的FU方法进行了全面分析，包括对各种评估指标的详细评论。

    The evolution of privacy-preserving Federated Learning (FL) has led to an increasing demand for implementing the right to be forgotten. The implementation of selective forgetting is particularly challenging in FL due to its decentralized nature. This complexity has given rise to a new field, Federated Unlearning (FU). FU emerges as a strategic solution to address the increasing need for data privacy, including the implementation of the `right to be forgotten'. The primary challenge in developing FU approaches lies in balancing the trade-offs in privacy, security, utility, and efficiency, as these elements often have competing requirements. Achieving an optimal equilibrium among these facets is crucial for maintaining the effectiveness and usability of FL systems while adhering to privacy and security standards. This survey provides a comprehensive analysis of existing FU methods, incorporating a detailed review of the various evaluation metrics. Furthermore, we unify these diverse meth
    
[^7]: 使用梯度下降法解决非常数核的核岭回归

    Solving Kernel Ridge Regression with Gradient Descent for a Non-Constant Kernel. (arXiv:2311.01762v1 [stat.ML])

    [http://arxiv.org/abs/2311.01762](http://arxiv.org/abs/2311.01762)

    本文研究了使用梯度下降法解决非常数核的核岭回归。通过在训练过程中逐渐减小带宽，避免了超参数选择的需求，并提出了一种带宽更新方案，证明了其优于使用常数带宽的方法。

    

    核岭回归（KRR）是线性岭回归的推广，它在数据中是非线性的，但在参数中是线性的。解决方案可以通过闭式解获得，其中包括矩阵求逆，也可以通过梯度下降迭代获得。本文研究了在训练过程中改变核函数的方法。我们从理论上探讨了这对模型复杂性和泛化性能的影响。基于我们的发现，我们提出了一种用于平移不变核的带宽更新方案，其中带宽在训练过程中逐渐减小至零，从而避免了超参数选择的需要。我们在真实和合成数据上展示了在训练过程中逐渐减小带宽的优于使用常数带宽，通过交叉验证和边缘似然最大化选择的带宽。我们还从理论和实证上证明了使用逐渐减小的带宽时，我们能够...

    Kernel ridge regression, KRR, is a generalization of linear ridge regression that is non-linear in the data, but linear in the parameters. The solution can be obtained either as a closed-form solution, which includes a matrix inversion, or iteratively through gradient descent. Using the iterative approach opens up for changing the kernel during training, something that is investigated in this paper. We theoretically address the effects this has on model complexity and generalization. Based on our findings, we propose an update scheme for the bandwidth of translational-invariant kernels, where we let the bandwidth decrease to zero during training, thus circumventing the need for hyper-parameter selection. We demonstrate on real and synthetic data how decreasing the bandwidth during training outperforms using a constant bandwidth, selected by cross-validation and marginal likelihood maximization. We also show theoretically and empirically that using a decreasing bandwidth, we are able to
    
[^8]: 学习非稳态条件下的稳定性原则

    A Stability Principle for Learning under Non-Stationarity. (arXiv:2310.18304v1 [cs.LG])

    [http://arxiv.org/abs/2310.18304](http://arxiv.org/abs/2310.18304)

    本研究提出了一个适用于非稳态环境的统计学习框架，通过应用稳定性原则选择回溯窗口来最大化历史数据利用，并保持累积偏差在可接受范围内。该方法展示了对未知非稳态的适应性，遗憾界在强凸或满足Lipschitz条件下是极小化的最优解。该研究的创新点是函数相似度度量和非稳态数据序列划分技术。

    

    我们在非稳定环境中开发了一个灵活的统计学习框架。在每个时间段，我们的方法应用稳定性原则来选择一个回溯窗口，最大限度地利用历史数据，同时将累积偏差保持在与随机误差相对可接受的范围内。我们的理论展示了该方法对未知非稳定性的适应性。当人口损失函数强凸或仅满足Lipschitz条件时，遗憾界是极小化的最优解，仅受对数因子的影响。我们的分析核心是两个新颖的组成部分：函数之间的相似度度量和将非稳态数据序列划分为准稳态片段的分割技术。

    We develop a versatile framework for statistical learning in non-stationary environments. In each time period, our approach applies a stability principle to select a look-back window that maximizes the utilization of historical data while keeping the cumulative bias within an acceptable range relative to the stochastic error. Our theory showcases the adaptability of this approach to unknown non-stationarity. The regret bound is minimax optimal up to logarithmic factors when the population losses are strongly convex, or Lipschitz only. At the heart of our analysis lie two novel components: a measure of similarity between functions and a segmentation technique for dividing the non-stationary data sequence into quasi-stationary pieces.
    
[^9]: 高维度 SGD 与新兴的异常特征空间相吻合

    High-dimensional SGD aligns with emerging outlier eigenspaces. (arXiv:2310.03010v1 [cs.LG])

    [http://arxiv.org/abs/2310.03010](http://arxiv.org/abs/2310.03010)

    本研究通过研究训练动态和经验海森矩阵以及梯度矩阵的谱的联合演化，证明了在高维混合和多层神经网络的分类任务中，SGD轨迹与海森矩阵和梯度矩阵的新兴低秩异常特征空间吻合。在多层设置中，这种对齐会在每一层发生，并且在收敛到亚优分类器时会表现出秩缺乏。

    

    我们通过随机梯度下降（SGD）和经验海森矩阵和梯度矩阵的谱的联合演化，对训练动态进行了严格的研究。我们证明在多类高维混合和1或2层神经网络的两个典型分类任务中，SGD轨迹迅速与海森矩阵和梯度矩阵的新兴低秩异常特征空间相吻合。此外，在多层设置中，这种对齐发生在每一层，最后一层的异常特征空间在训练过程中演化，并且在SGD收敛到亚优分类器时表现出秩缺乏。这为过去十年中关于在超参数化网络中训练过程中海森矩阵和信息矩阵的谱的广泛数值研究提供了丰富的预测。

    We rigorously study the joint evolution of training dynamics via stochastic gradient descent (SGD) and the spectra of empirical Hessian and gradient matrices. We prove that in two canonical classification tasks for multi-class high-dimensional mixtures and either 1 or 2-layer neural networks, the SGD trajectory rapidly aligns with emerging low-rank outlier eigenspaces of the Hessian and gradient matrices. Moreover, in multi-layer settings this alignment occurs per layer, with the final layer's outlier eigenspace evolving over the course of training, and exhibiting rank deficiency when the SGD converges to sub-optimal classifiers. This establishes some of the rich predictions that have arisen from extensive numerical studies in the last decade about the spectra of Hessian and information matrices over the course of training in overparametrized networks.
    
[^10]: 使用规范化流计算分子的激发态

    Computing excited states of molecules using normalizing flows. (arXiv:2308.16468v1 [physics.chem-ph])

    [http://arxiv.org/abs/2308.16468](http://arxiv.org/abs/2308.16468)

    使用规范化流计算分子的激发态，通过逼近波函数并优化基函数的线性空间内的近似。该方法在计算量子系统中取得了准确和有效的结果，并在能量预测准确性和基组收敛速度方面进行了显著改善。

    

    我们提出了一种新的非线性变分框架，可以同时计算量子系统的基态和激发态。我们的方法基于通过与规范化流的组合来逼近波函数，这些波函数位于基函数的线性空间中，并对其进行优化。我们通过计算三原子H$_2$S分子的大量振动态以及典型的单电子系统（包括氢原子、分子氢离子和碳原子在单激发电子近似下的基态和多个激发态）来证明我们方法的准确性和效率。结果表明，即使使用参数较少的规范化流，能量预测的准确性和基组收敛速度也有显著改善。该方法也可以被看作是对最佳捕捉底层物理的一组内禀坐标进行优化的过程。

    We present a new nonlinear variational framework for simultaneously computing ground and excited states of quantum systems. Our approach is based on approximating wavefunctions in the linear span of basis functions that are augmented and optimized \emph{via} composition with normalizing flows. The accuracy and efficiency of our approach are demonstrated in the calculations of a large number of vibrational states of the triatomic H$_2$S molecule as well as ground and several excited electronic states of prototypical one-electron systems including the hydrogen atom, the molecular hydrogen ion, and a carbon atom in a single-active-electron approximation. The results demonstrate significant improvements in the accuracy of energy predictions and accelerated basis-set convergence even when using normalizing flows with a small number of parameters. The present approach can be also seen as the optimization of a set of intrinsic coordinates that best capture the underlying physics within the gi
    
[^11]: 通过赌博进行公平性审计

    Auditing Fairness by Betting. (arXiv:2305.17570v1 [stat.ML])

    [http://arxiv.org/abs/2305.17570](http://arxiv.org/abs/2305.17570)

    本文提供了一种通过赌博的方式进行公平性审计的方法，相比之前的方法，这种方法具有更高的实用性和效率，能够对不断产生的数据进行连续的监控，并处理因分布漂移导致的公平性问题。

    

    我们提供了实用、高效、非参数方法，用于审计已部署的分类和回归模型的公平性。相比之前依赖于固定样本量的方法，我们的方法是序贯的，并允许对不断产生的数据进行连续的监控，因此非常适用于跟踪现实世界系统的公平性。我们也允许数据通过概率策略进行收集，而不是从人口中均匀采样。这使得审计可以在为其他目的收集的数据上进行。此外，该策略可以随时间改变，并且不同的子人群可以使用不同的策略。最后，我们的方法可以处理因模型变更或基础人群变更导致的分布漂移。我们的方法基于最近关于 anytime-valid 推断和博弈统计学的进展，尤其是"通过赌博进行测试"框架。这些联系确保了我们的方法具有可解释性、快速和提供统计保证。

    We provide practical, efficient, and nonparametric methods for auditing the fairness of deployed classification and regression models. Whereas previous work relies on a fixed-sample size, our methods are sequential and allow for the continuous monitoring of incoming data, making them highly amenable to tracking the fairness of real-world systems. We also allow the data to be collected by a probabilistic policy as opposed to sampled uniformly from the population. This enables auditing to be conducted on data gathered for another purpose. Moreover, this policy may change over time and different policies may be used on different subpopulations. Finally, our methods can handle distribution shift resulting from either changes to the model or changes in the underlying population. Our approach is based on recent progress in anytime-valid inference and game-theoretic statistics-the "testing by betting" framework in particular. These connections ensure that our methods are interpretable, fast, 
    
[^12]: 选择超参数优化方法的从业者动机

    Practitioner Motives to Select Hyperparameter Optimization Methods. (arXiv:2203.01717v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2203.01717](http://arxiv.org/abs/2203.01717)

    研究探讨了机器学习从业者选择超参数优化方法的动机，结果表明这基于个人目标和背景因素，调查还给出了优化模型的六个主要目标。

    

    先进的编程超参数优化方法，如贝叶斯优化，具有高样本效率，能够可靠地找到机器学习模型的最佳超参数值。然而，机器学习从业者经常应用样本效率较低的HPO方法，如网格搜索，这通常导致机器学习模型未经优化。我们怀疑，从业者选择HPO方法的原因基于个人动机，包括背景因素和个人目标。然而，从业者的动机仍然需要澄清，这妨碍了评估HPO方法以实现特定目标和以用户为中心的HPO工具的开发。为了了解从业者使用特定HPO方法的动机，我们采用混合方法，包括20个半结构化访谈和一项调查研究，共有71名机器学习专家参与，以收集访谈结果的外部有效性的证据。通过设置六个主要目标（例如，改进模型理解），

    Advanced programmatic hyperparameter optimization (HPO) methods, such as Bayesian optimization, have high sample efficiency in reproducibly finding optimal hyperparameter values of machine learning (ML) models. Yet, ML practitioners often apply less sample-efficient HPO methods, such as grid search, which often results in under-optimized ML models. As a reason for this behavior, we suspect practitioners choose HPO methods based on individual motives, consisting of contextual factors and individual goals. However, practitioners' motives still need to be clarified, hindering the evaluation of HPO methods for achieving specific goals and the user-centered development of HPO tools. To understand practitioners' motives for using specific HPO methods, we used a mixed-methods approach involving 20 semi-structured interviews and a survey study with 71 ML experts to gather evidence of the external validity of the interview results. By presenting six main goals (e.g., improving model understandi
    

