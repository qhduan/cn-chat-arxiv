# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Operator SVD with Neural Networks via Nested Low-Rank Approximation](https://arxiv.org/abs/2402.03655) | 本文提出了一个新的优化框架，使用嵌套的低秩近似方法通过神经网络实现运算符的奇异值分解。该方法通过无约束优化公式隐式高效地保持学习函数的正交性。 |
| [^2] | [Quantum Inception Score](https://arxiv.org/abs/2311.12163) | 通过量子启蒙分数，我们提出了一个用于评估量子生成模型质量的新指标，证明量子生成模型在质量上优于经典生成模型，并利用量子波动定理揭示了其物理限制。 |
| [^3] | [S4Sleep: Elucidating the design space of deep-learning-based sleep stage classification models.](http://arxiv.org/abs/2310.06715) | 本研究解析了基于深度学习的睡眠阶段分类模型的设计空间，找到了适用于不同输入表示的稳健架构，并在睡眠数据集上实现了显著的性能提升。 |
| [^4] | [A comprehensive study of spike and slab shrinkage priors for structurally sparse Bayesian neural networks.](http://arxiv.org/abs/2308.09104) | 本论文研究了在贝叶斯神经网络中使用Lasso和Horseshoe两种缩减技术进行模型压缩的方法。为了实现结构稀疏，通过提出尖峰与块组稀疏Lasso和尖峰与块组Horseshoe先验，并开发了可计算的变分推断方法。该方法可以在保持推理效率的同时实现深度神经网络的模型压缩。 |
| [^5] | [Recent Advances in Optimal Transport for Machine Learning.](http://arxiv.org/abs/2306.16156) | 最优输运在机器学习中的最新进展包括生成建模和迁移学习等领域，并且计算最优输运的发展也与机器学习实践相互影响。 |

# 详细

[^1]: 使用神经网络通过嵌套低秩近似实现运算符的奇异值分解

    Operator SVD with Neural Networks via Nested Low-Rank Approximation

    [https://arxiv.org/abs/2402.03655](https://arxiv.org/abs/2402.03655)

    本文提出了一个新的优化框架，使用嵌套的低秩近似方法通过神经网络实现运算符的奇异值分解。该方法通过无约束优化公式隐式高效地保持学习函数的正交性。

    

    在许多机器学习和科学计算问题中，计算给定线性算子的特征值分解（EVD）或找到其主要特征值和特征函数是一项基础任务。对于高维特征值问题，训练神经网络参数化特征函数被认为是传统数值线性代数技术的有希望的替代方法。本文提出了一个新的优化框架，基于截断奇异值分解的低秩近似表征，并伴随着称为嵌套的学习方法，以正确的顺序学习前L个奇异值和奇异函数。所提出的方法通过无约束优化公式隐式高效地促进了学习函数的正交性，这个公式可以很容易地通过现成的基于梯度的优化算法求解。我们展示了所提出的优化框架在使用案例中的有效性。

    Computing eigenvalue decomposition (EVD) of a given linear operator, or finding its leading eigenvalues and eigenfunctions, is a fundamental task in many machine learning and scientific computing problems. For high-dimensional eigenvalue problems, training neural networks to parameterize the eigenfunctions is considered as a promising alternative to the classical numerical linear algebra techniques. This paper proposes a new optimization framework based on the low-rank approximation characterization of a truncated singular value decomposition, accompanied by new techniques called nesting for learning the top-$L$ singular values and singular functions in the correct order. The proposed method promotes the desired orthogonality in the learned functions implicitly and efficiently via an unconstrained optimization formulation, which is easy to solve with off-the-shelf gradient-based optimization algorithms. We demonstrate the effectiveness of the proposed optimization framework for use cas
    
[^2]: 量子启蒙分数

    Quantum Inception Score

    [https://arxiv.org/abs/2311.12163](https://arxiv.org/abs/2311.12163)

    通过量子启蒙分数，我们提出了一个用于评估量子生成模型质量的新指标，证明量子生成模型在质量上优于经典生成模型，并利用量子波动定理揭示了其物理限制。

    

    受到经典生成模型在机器学习中取得巨大成功的启发，近期开始了对它们量子版本的热切探索。为了开始这一探索之旅，开发一个相关的度量标准来评估量子生成模型的质量是很重要的；在经典情况下，一个这样的例子便是启蒙分数。在本文中，我们提出了量子启蒙分数，它将质量与用于对给定数据集进行分类的量子通道的Holevo信息联系起来。我们证明，在这个提出的度量标准下，量子生成模型提供比它们的经典对应物更好的质量，因为存在着由不对称性的资源理论和纠缠所表征的量子相干性。此外，我们利用量子波动定理来表征限制量子生成模型质量的物理限制。最后，我们应用量子启蒙分数来

    arXiv:2311.12163v2 Announce Type: replace-cross  Abstract: Motivated by the great success of classical generative models in machine learning, enthusiastic exploration of their quantum version has recently started. To depart on this journey, it is important to develop a relevant metric to evaluate the quality of quantum generative models; in the classical case, one such example is the inception score. In this paper, we propose the quantum inception score, which relates the quality to the Holevo information of the quantum channel that classifies a given dataset. We prove that, under this proposed measure, the quantum generative models provide better quality than their classical counterparts because of the presence of quantum coherence, characterized by the resource theory of asymmetry, and entanglement. Furthermore, we harness the quantum fluctuation theorem to characterize the physical limitation of the quality of quantum generative models. Finally, we apply the quantum inception score 
    
[^3]: S4Sleep: 解析基于深度学习的睡眠阶段分类模型的设计空间

    S4Sleep: Elucidating the design space of deep-learning-based sleep stage classification models. (arXiv:2310.06715v1 [cs.LG])

    [http://arxiv.org/abs/2310.06715](http://arxiv.org/abs/2310.06715)

    本研究解析了基于深度学习的睡眠阶段分类模型的设计空间，找到了适用于不同输入表示的稳健架构，并在睡眠数据集上实现了显著的性能提升。

    

    对于多通道睡眠脑电图记录进行睡眠阶段打分是一项耗时且存在显著的评分人员之间差异的任务。因此，应用机器学习算法可以带来很大的益处。虽然已经为此提出了许多算法，但某些关键的架构决策并未得到系统性的探索。在本研究中，我们详细调查了广泛的编码器-预测器架构范畴内的这些设计选择。我们找到了适用于时间序列和声谱图输入表示的稳健架构。这些架构将结构化状态空间模型作为组成部分，对广泛的SHHS数据集的性能进行了统计显著的提升。这些改进通过统计和系统误差估计进行了评估。我们预计，从本研究中获得的架构洞察不仅对未来的睡眠分期研究有价值，而且对整体睡眠研究都有价值。

    Scoring sleep stages in polysomnography recordings is a time-consuming task plagued by significant inter-rater variability. Therefore, it stands to benefit from the application of machine learning algorithms. While many algorithms have been proposed for this purpose, certain critical architectural decisions have not received systematic exploration. In this study, we meticulously investigate these design choices within the broad category of encoder-predictor architectures. We identify robust architectures applicable to both time series and spectrogram input representations. These architectures incorporate structured state space models as integral components, leading to statistically significant advancements in performance on the extensive SHHS dataset. These improvements are assessed through both statistical and systematic error estimations. We anticipate that the architectural insights gained from this study will not only prove valuable for future research in sleep staging but also hol
    
[^4]: 基于尖峰与块缩减先验的结构稀疏贝叶斯神经网络的全面研究

    A comprehensive study of spike and slab shrinkage priors for structurally sparse Bayesian neural networks. (arXiv:2308.09104v1 [stat.ML])

    [http://arxiv.org/abs/2308.09104](http://arxiv.org/abs/2308.09104)

    本论文研究了在贝叶斯神经网络中使用Lasso和Horseshoe两种缩减技术进行模型压缩的方法。为了实现结构稀疏，通过提出尖峰与块组稀疏Lasso和尖峰与块组Horseshoe先验，并开发了可计算的变分推断方法。该方法可以在保持推理效率的同时实现深度神经网络的模型压缩。

    

    网络复杂度和计算效率已经成为深度学习中越来越重要的方面。稀疏深度学习通过减少过参数化的深度神经网络来恢复底层目标函数的稀疏表示，解决了这些挑战。具体而言，通过结构稀疏（如节点稀疏）压缩的深度神经架构提供了低延迟推理、更高的数据吞吐量和更低的能量消耗。在本文中，我们研究了两种广泛应用的缩减技术，Lasso和Horseshoe，在贝叶斯神经网络中进行模型压缩。为此，我们提出了基于尖峰与块组稀疏Lasso (SS-GL)和基于尖峰与块组Horseshoe (SS-GHS)先验的结构稀疏贝叶斯神经网络，并开发了可计算的变分推断，包括对伯努利变量的连续松弛。我们确定了变分推断的收缩速率。

    Network complexity and computational efficiency have become increasingly significant aspects of deep learning. Sparse deep learning addresses these challenges by recovering a sparse representation of the underlying target function by reducing heavily over-parameterized deep neural networks. Specifically, deep neural architectures compressed via structured sparsity (e.g. node sparsity) provide low latency inference, higher data throughput, and reduced energy consumption. In this paper, we explore two well-established shrinkage techniques, Lasso and Horseshoe, for model compression in Bayesian neural networks. To this end, we propose structurally sparse Bayesian neural networks which systematically prune excessive nodes with (i) Spike-and-Slab Group Lasso (SS-GL), and (ii) Spike-and-Slab Group Horseshoe (SS-GHS) priors, and develop computationally tractable variational inference including continuous relaxation of Bernoulli variables. We establish the contraction rates of the variational 
    
[^5]: 机器学习中最优输运的最新进展

    Recent Advances in Optimal Transport for Machine Learning. (arXiv:2306.16156v1 [cs.LG])

    [http://arxiv.org/abs/2306.16156](http://arxiv.org/abs/2306.16156)

    最优输运在机器学习中的最新进展包括生成建模和迁移学习等领域，并且计算最优输运的发展也与机器学习实践相互影响。

    

    最近，最优输运被提出作为机器学习中比较和操作概率分布的概率框架。这个框架源于其丰富的历史和理论，并提供了新的解决方案，如生成建模和迁移学习。在这项调查中，我们探讨了最优输运在2012年至2022年期间对机器学习的贡献，重点关注机器学习的四个子领域：有监督学习、无监督学习、迁移学习和强化学习。我们还突出了计算最优输运的最新发展，并与机器学习实践相互影响。

    Recently, Optimal Transport has been proposed as a probabilistic framework in Machine Learning for comparing and manipulating probability distributions. This is rooted in its rich history and theory, and has offered new solutions to different problems in machine learning, such as generative modeling and transfer learning. In this survey we explore contributions of Optimal Transport for Machine Learning over the period 2012 -- 2022, focusing on four sub-fields of Machine Learning: supervised, unsupervised, transfer and reinforcement learning. We further highlight the recent development in computational Optimal Transport, and its interplay with Machine Learning practice.
    

