# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Understanding deep neural networks through the lens of their non-linearity.](http://arxiv.org/abs/2310.11439) | 本文提出了一个理论上有效的解决方案，通过亲和度评分追踪深度神经网络中的非线性传播，尤其关注计算机视觉应用。实验证实了所提出方法的实用性和对广泛应用的潜力。 |
| [^2] | [Identifying Interpretable Visual Features in Artificial and Biological Neural Systems.](http://arxiv.org/abs/2310.11431) | 本文提出了一种量化视觉可解释性的自动化方法，并找到了卷积神经网络中有意义的方向。 |
| [^3] | [Butterfly Effects of SGD Noise: Error Amplification in Behavior Cloning and Autoregression.](http://arxiv.org/abs/2310.11428) | 这项研究探究了在深度神经网络中进行行为克隆训练时出现的训练不稳定性现象。我们发现，尽管小批量SGD更新对于行为克隆损失几乎没有影响，但它会导致长期奖励的剧烈振荡。我们称这种效应为梯度方差放大（GVA），并发现使用指数移动平均（EMA）可以有效减缓这种效应。这一现象在连续控制和自回归等领域具有普遍性。 |
| [^4] | [Faster Algorithms for Generalized Mean Densest Subgraph Problem.](http://arxiv.org/abs/2310.11377) | 该论文提出了更快速的算法来解决广义平均密集子图问题，其中对于$0<p<1$的情况下，标准剥离算法可以得到$2^{1/p}$的近似解。另外，引入了一种新的广义剥离算法（GENPEEL），对于$p \geq 1$，其近似保证率为$(p+1)^{1/p}$。 |
| [^5] | [Lie Group Decompositions for Equivariant Neural Networks.](http://arxiv.org/abs/2310.11366) | 本论文提出了一种基于Lie群结构和几何特性的框架，可以处理非紧致非阿贝尔的Lie群，特别关注于$\text{GL}^{+}(n, \mathbb{R})$和$\text{SL}(n, \mathbb{R})$这两个Lie群。 |
| [^6] | [Contextualized Machine Learning.](http://arxiv.org/abs/2310.11340) | 上下文化机器学习是一种用于学习异质和上下文相关效应的新范式，通过深度学习和元关系估计异质函数，并引入上下文编码器和样本特定模型来统一现有框架。 |
| [^7] | [Elucidating The Design Space of Classifier-Guided Diffusion Generation.](http://arxiv.org/abs/2310.11311) | 本论文通过研究设计空间，提出了一种新的无训练引导方案，通过利用现成的分类器来引导扩散生成，在保持灵活性的同时实现了显著的性能提升。 |
| [^8] | [Gromov-Wassertein-like Distances in the Gaussian Mixture Models Space.](http://arxiv.org/abs/2310.11256) | 本文介绍了两种在高斯混合模型空间中的Gromov-Wasserstein类型距离，分别用于评估分布之间的距离和推导最优的点分配方案。 |
| [^9] | [Learning to Sample Better.](http://arxiv.org/abs/2310.11232) | 本课程介绍了基于动态输运测度的生成建模方法的最新进展，重点讲述了如何通过数据学习这些映射，并通过正反馈循环改进蒙特卡洛采样技术。 |
| [^10] | [Federated Learning with Nonvacuous Generalisation Bounds.](http://arxiv.org/abs/2310.11203) | 这项研究提出了一种新的策略来在联邦学习中训练随机预测器，通过保护每个节点的隐私并且具有数值上非空的泛化界限，可以在保持预测性能的同时实现数据共享和保护隐私。 |
| [^11] | [A new high-resolution indoor radon map for Germany using a machine learning based probabilistic exposure model.](http://arxiv.org/abs/2310.11143) | 本研究提出了一种基于机器学习的概率暴露模型，可以更准确地估计德国室内氡气分布，并具有更高的空间分辨率。 |
| [^12] | [Sensitivity-Aware Amortized Bayesian Inference.](http://arxiv.org/abs/2310.11122) | 本文提出了一种敏感性感知的摊销贝叶斯推断方法，通过权重共享和神经网络来进行似然和先验规范的训练，以及对数据扰动和预处理程序的敏感性评估。 |
| [^13] | [Minimally Informed Linear Discriminant Analysis: training an LDA model with unlabelled data.](http://arxiv.org/abs/2310.11110) | 本文展示了在只有未标记数据的情况下，通过一些最小的先验信息，可以计算出精确的LDA投影向量。数值实验验证了这种最小信息的线性判别分析（MILDA）模型与有监督的LDA模型的性能接近。 |
| [^14] | [Resampling Stochastic Gradient Descent Cheaply for Efficient Uncertainty Quantification.](http://arxiv.org/abs/2310.11065) | 本研究提出了两种低成本重采样的方法，用于构建随机梯度下降解的置信区间，这一方法可以有效减少计算工作量，并绕过现有方法中的混合条件。 |
| [^15] | [Matrix Compression via Randomized Low Rank and Low Precision Factorization.](http://arxiv.org/abs/2310.11028) | 通过随机化的低秩和低精度因式分解，我们提出了一种矩阵压缩算法，可以有效地减小存储和处理大型矩阵所需的计算资源和内存使用。 |
| [^16] | [From Identifiable Causal Representations to Controllable Counterfactual Generation: A Survey on Causal Generative Modeling.](http://arxiv.org/abs/2310.11011) | 本文综述了因果生成建模的技术，其中分为因果表示学习和可控反事实生成两个部分，这些模型融合了因果理论，解决了深度生成模型的一些根本性缺点，并提供了分布偏移鲁棒性、公平性和互操作性等有益属性。 |
| [^17] | [WGoM: A novel model for categorical data with weighted responses.](http://arxiv.org/abs/2310.10989) | 本文提出了一种名为加权成员级别（WGoM）模型，用于解决基于分类数据的潜在类别推断问题。与现有模型相比，WGoM更通用且适用于具有连续或负响应的数据集。通过提出的算法，我们能够准确高效地估计潜在混合成员和其他WGoM参数，并且通过实验证明了该算法的性能和实用潜力。 |
| [^18] | [Latent class analysis with weighted responses.](http://arxiv.org/abs/2310.10984) | 提出了一种新的生成模型，即加权潜在类别模型（WLCM），可以用于对具有连续或负响应的真实世界数据进行潜在类别分析。 |
| [^19] | [MRI brain tumor segmentation using informative feature vectors and kernel dictionary learning.](http://arxiv.org/abs/2310.10963) | 本文提出一种基于核字典学习算法的方法，利用统计特征向量和相关性样本选择技术，在MRI中实现了脑肿瘤的有效分割。 |
| [^20] | [A Local Graph Limits Perspective on Sampling-Based GNNs.](http://arxiv.org/abs/2310.10953) | 该论文提出了一种基于局部图界限的训练大型输入图的采样型图神经网络的理论框架，通过对小样本的训练，我们可以获得与整个图训练类似的结果。这为使用采样训练GNN提供了新的理论理解，并提供了在选择最佳模型、超参数和采样算法方面更高效的方法。 |
| [^21] | [Restricted Tweedie Stochastic Block Models.](http://arxiv.org/abs/2310.10952) | 这项研究提出了一种新的随机块模型，可以处理由非负零膨胀连续边权组成的邻接矩阵，特别适用于模拟国际贸易网络。该模型结合了节点信息和动态效应，并且可以独立于社区标签进行参数估计。一个高效的两步算法被开发用于估计协变效应和社区标签。 |
| [^22] | [Surrogate Active Subspaces for Jump-Discontinuous Functions.](http://arxiv.org/abs/2310.10907) | 该论文提出了一种针对不连续函数的替代主动子空间方法，扩展了活跃子空间的应用范围，并通过数值实验验证了该方法的有效性。 |
| [^23] | [Approximation properties of slice-matching operators.](http://arxiv.org/abs/2310.10869) | 本文研究了迭代的切片匹配算法的近似性质，并探讨了与源度量、目标度量和切片方向相关的属性。研究结果包括与源度量相关的不变性属性、与目标度量相关的等变性属性以及与切片方向相关的Lipschitz连续性。此外，还给出了通过一步切片匹配方案逼近目标度量的误差界限，并研究了切片匹配算子恢复最优输运映射的情况。 |
| [^24] | [Probabilistic Classification by Density Estimation Using Gaussian Mixture Model and Masked Autoregressive Flow.](http://arxiv.org/abs/2310.10843) | 本研究提出了一种使用密度估计进行概率分类的方法，通过使用高斯混合模型和蒙特卡洛自回归流对数据的似然进行建模，并展示了这种方法优于传统的分类器。这项工作为基于联合密度估计的其他概率分类器的提出开辟了新的研究方向。 |
| [^25] | [Regularization properties of adversarially-trained linear regression.](http://arxiv.org/abs/2310.10807) | 本研究对对抗训练线性回归的正则化性质进行了研究，发现在过参数化情况下，对抗训练可以得到最小范数插值解，这一发现对理解对抗训练的效果和应用具有重要意义。 |
| [^26] | [Neural Tangent Kernels Motivate Graph Neural Networks with Cross-Covariance Graphs.](http://arxiv.org/abs/2310.10791) | 本文研究了神经切向核函数（NTKs）在图神经网络（GNNs）中的应用。我们发现优化对齐等价于优化GNN中的图表示或图移位运算符，并建立了对于两层GNN对齐的最优性的理论保证。 |
| [^27] | [Wide Neural Networks as Gaussian Processes: Lessons from Deep Equilibrium Models.](http://arxiv.org/abs/2310.10767) | 本文研究了神经网络中广义神经网络和高斯过程的对应关系，发现具有无限深度层并且宽度趋近于无穷大的神经网络收敛于高斯过程，揭示了广义神经网络的良性过拟合现象。 |
| [^28] | [Mori-Zwanzig latent space Koopman closure for nonlinear autoencoder.](http://arxiv.org/abs/2310.10745) | 本研究提出了一种名为Mori-Zwanzig自编码器（MZ-AE）的新方法，用于在低维空间中稳健地逼近Koopman算子，通过非线性自编码器和Mori-Zwanzig形式主义的集成实现对有限不变Koopman子空间的逼近，从而增强了精确性和准确预测复杂系统行为的能力。 |
| [^29] | [TacticAI: an AI assistant for football tactics.](http://arxiv.org/abs/2310.10553) | 提出了TacticAI，一种与利物浦足球俱乐部的领域专家密切合作开发和评价的AI足球战术助手。TacticAI能够通过预测和生成的方式帮助教练们分析角球情况，并为每个角球惯例选择成功可能性最高的球员配置。 |
| [^30] | [Towards the Fundamental Limits of Knowledge Transfer over Finite Domains.](http://arxiv.org/abs/2310.07838) | 本论文研究了在有限领域中从教师到学生分类器进行知识传递的统计效率，发现特权信息会加速传递，通过使用一种新颖的损失函数达到了知识传递的基本限制。 |
| [^31] | [Spectrum-Aware Adjustment: A New Debiasing Framework with Applications to Principal Components Regression.](http://arxiv.org/abs/2309.07810) | 这项研究介绍了一种新的去偏方法框架，用于解决高维线性回归中现有技术对协变量分布的限制问题。研究者们发现，现有方法在处理非高斯分布、异质性设计矩阵和缺乏可靠特征协方差估计时遇到困难。为了解决这些问题，他们提出了一种新的策略，该策略利用缩放的梯度下降步骤进行去偏校正。 |
| [^32] | [Geometry-Aware Approaches for Balancing Performance and Theoretical Guarantees in Linear Bandits.](http://arxiv.org/abs/2306.14872) | 本文提出了一种新的数据驱动技术，跟踪不确定度椭球体的几何形状，为线性赌博机算法建立实例相关的频率后悔界，并实现了平衡算法性能与理论保证的效果。 |
| [^33] | [Kernel Quadrature with Randomly Pivoted Cholesky.](http://arxiv.org/abs/2306.03955) | 本文提出了一种新的使用随机选择纯量分解算法的核求积方法，可以在达到可比的求积误差达到率的同时显著降低计算复杂度，并可以应用于任意核的复杂几何结构。 |
| [^34] | [How many samples are needed to leverage smoothness?.](http://arxiv.org/abs/2305.16014) | 本文通过研究泛化误差的新下界，探讨了学习平滑函数时需要的样本数量及其机器学习问题中的挑战。 |
| [^35] | [Posterior Inference on Infinitely Wide Bayesian Neural Networks under Weights with Unbounded Variance.](http://arxiv.org/abs/2305.10664) | 本文提出了一种新的方法进行关于具有无界方差权重的贝叶斯神经网络的后验推断，并表明后验分布集中在具有非标准超参数依赖性的稀疏促进和均值收缩先验周围。 |
| [^36] | [HINT: Hierarchical Mixture Networks For Coherent Probabilistic Forecasting.](http://arxiv.org/abs/2305.07089) | HINT是一种用于概率预测的新型模型族，能够有效、准确地进行一致性预测，通过引入Bootstrap方法并为网络加入规范化特征提取和输出规范化来保证其性能，在多个数据集上的预测精度比现有技术更高。 |
| [^37] | [The emergence of clusters in self-attention dynamics.](http://arxiv.org/abs/2305.05465) | 本文证实了当Transformer处理一系列token时，出现“领导者”的经验观察，即随着时间趋于无穷大，代表token的粒子会聚集在特定的极限对象附近，这取决于价值矩阵的谱。 |
| [^38] | [Multivariate Probabilistic CRPS Learning with an Application to Day-Ahead Electricity Prices.](http://arxiv.org/abs/2303.10019) | 本文提出一种新的多元概率CRPS学习方法，应用于日前电价预测中，相比于统一组合在CRPS方面取得了显著改进。 |
| [^39] | [A path in regression Random Forest looking for spatial dependence: a taxonomy and a systematic review.](http://arxiv.org/abs/2303.04693) | 在这项工作中，我们提出了一种分类法，根据前处理、中处理和/或后处理的时间点尝试将空间信息纳入回归随机森林中。此外，我们进行了系统回顾并分类最新采用的调整回归随机森林以适应空间相关数据的策略。 |
| [^40] | [Densely Connected $G$-invariant Deep Neural Networks with Signed Permutation Representations.](http://arxiv.org/abs/2303.04614) | 本文提出了一种具有带符号排列表示的密集连接$G$-不变深度神经网络($G$-DNN)架构，通过耦合权重，使得网络的前激活能够通过$G$的带符号排列表示进行变换，从而得到一族更丰富的$G$-不变架构。 |
| [^41] | [Towards Minimax Optimality of Model-based Robust Reinforcement Learning.](http://arxiv.org/abs/2302.05372) | 本文研究了在鲁棒强化学习中，对于仅具有对正常核心的生成模型访问权限时，获得ε-最优策略的样本复杂度。对于sa（s-）矩形不确定集合，已知最佳样本复杂度为ε^2/（H^4 * |S|^2 * |A|）（响应为ε^2/（H^4 * |S|^2 * |A|^2）），对于特定算法和基于总变差（TV）、KL或卡方散度的不确定集合。 |
| [^42] | [Sketchy: Memory-efficient Adaptive Regularization with Frequent Directions.](http://arxiv.org/abs/2302.03764) | 本论文提出了一种内存高效的自适应正则化方法，通过使用频繁方向草稿来降低矩阵预处理器的内存和计算需求。在深度学习任务中，该方法可以在保持性能的同时降低资源的使用。 |
| [^43] | [A Recipe for Well-behaved Graph Neural Approximations of Complex Dynamics.](http://arxiv.org/abs/2301.04900) | 本文介绍了一种行为良好的图神经网络近似复杂动力学的方法，包括必要的偏置和适当的神经网络结构，并提出了评估泛化能力和推断时预测置信度的方法。 |
| [^44] | [On the Overlooked Structure of Stochastic Gradients.](http://arxiv.org/abs/2212.02083) | 本文对深度学习中随机梯度的结构进行了正式的统计检验，发现逐维梯度通常呈现幂律重尾，而逐次迭代的梯度和随机梯度噪声通常不呈现幂律重尾。 |
| [^45] | [A Theoretical Analysis on Independence-driven Importance Weighting for Covariate-shift Generalization.](http://arxiv.org/abs/2111.02355) | 本文通过理论分析，将独立性驱动的重要性加权算法解释为特征选择过程，并证明了在协变量偏移泛化中的有效性。 |
| [^46] | [Optimal Model Selection in Contextual Bandits with Many Classes via Offline Oracles.](http://arxiv.org/abs/2106.06483) | 本论文研究了在随机上下文推断设置中，针对累计遗憾最小化的最优模型选择问题。通过引入渐增类别复杂性和递减边际收益条件，我们提出了一种基于新颖误配测试的算法，并展示了模型选择在奖励估计中的优势。 |
| [^47] | [Gaussian Mixture Reduction with Composite Transportation Divergence.](http://arxiv.org/abs/2002.08410) | 本文提出了一种基于复合传输散度的高斯混合简化方法，用于解决高斯混合在递归更新中阶数指数增加的推断问题。 |
| [^48] | [Improving Native Ads CTR Prediction by Large Scale Event Embedding and Recurrent Networks.](http://arxiv.org/abs/1804.09133) | 本文通过大规模事件嵌入和循环网络，提出了一种改进的CTR预测方法，在原生广告中取得了显著优势。 |

# 详细

[^1]: 通过非线性研究深度神经网络的理解

    Understanding deep neural networks through the lens of their non-linearity. (arXiv:2310.11439v1 [cs.LG])

    [http://arxiv.org/abs/2310.11439](http://arxiv.org/abs/2310.11439)

    本文提出了一个理论上有效的解决方案，通过亲和度评分追踪深度神经网络中的非线性传播，尤其关注计算机视觉应用。实验证实了所提出方法的实用性和对广泛应用的潜力。

    

    深度神经网络(DNN)的显著成功常常归因于它们的高表达能力和近似任意复杂函数的能力。事实上，DNN是高度非线性的模型，其中引入的激活函数在其中起到了重要作用。然而，尽管许多研究通过近似能力的视角研究了DNN的表达能力，但量化DNN或个别激活函数的非线性仍然是一个开放性问题。在本文中，我们提出了第一个在具体关注计算机视觉应用中追踪非线性传播的理论有效解决方案。我们提出的亲和度评分允许我们深入了解各种不同体系结构和学习范式的内部工作原理。我们提供了大量的实验结果，突出了所提出的亲和度评分的实际效用和潜在应用的可能性。

    The remarkable success of deep neural networks (DNN) is often attributed to their high expressive power and their ability to approximate functions of arbitrary complexity. Indeed, DNNs are highly non-linear models, and activation functions introduced into them are largely responsible for this. While many works studied the expressive power of DNNs through the lens of their approximation capabilities, quantifying the non-linearity of DNNs or of individual activation functions remains an open problem. In this paper, we propose the first theoretically sound solution to track non-linearity propagation in deep neural networks with a specific focus on computer vision applications. Our proposed affinity score allows us to gain insights into the inner workings of a wide range of different architectures and learning paradigms. We provide extensive experimental results that highlight the practical utility of the proposed affinity score and its potential for long-reaching applications.
    
[^2]: 在人工和生物神经系统中识别可解释的视觉特征

    Identifying Interpretable Visual Features in Artificial and Biological Neural Systems. (arXiv:2310.11431v1 [stat.ML])

    [http://arxiv.org/abs/2310.11431](http://arxiv.org/abs/2310.11431)

    本文提出了一种量化视觉可解释性的自动化方法，并找到了卷积神经网络中有意义的方向。

    

    神经网络中的单个神经元通常是“可解释的”，因为它们代表个别直观有意义的特征。然而，许多神经元表现出“混合选择性”，即它们代表多个不相关的特征。最近的假设认为，深度网络中的特征可能以“叠加”的方式表示，即由多个神经元沿非正交轴表示，因为自然数据中可解释的特征数通常大于给定网络中的神经元数。因此，我们应该能够在激活空间中找到与个别神经元不对齐的有意义的方向。在这里，我们提出了（1）一种自动化的方法来量化视觉可解释性，并通过与大量人类心理物理学对神经元可解释性的判断进行验证，以及（2）一种在网络激活空间中寻找有意义方向的方法。我们利用这些方法来发现卷积神经网络中的方向。

    Single neurons in neural networks are often ``interpretable'' in that they represent individual, intuitively meaningful features. However, many neurons exhibit $\textit{mixed selectivity}$, i.e., they represent multiple unrelated features. A recent hypothesis proposes that features in deep networks may be represented in $\textit{superposition}$, i.e., on non-orthogonal axes by multiple neurons, since the number of possible interpretable features in natural data is generally larger than the number of neurons in a given network. Accordingly, we should be able to find meaningful directions in activation space that are not aligned with individual neurons. Here, we propose (1) an automated method for quantifying visual interpretability that is validated against a large database of human psychophysics judgments of neuron interpretability, and (2) an approach for finding meaningful directions in network activation space. We leverage these methods to discover directions in convolutional neural
    
[^3]: SGD噪声的蝴蝶效应：行为克隆和自回归中的误差放大

    Butterfly Effects of SGD Noise: Error Amplification in Behavior Cloning and Autoregression. (arXiv:2310.11428v1 [cs.LG])

    [http://arxiv.org/abs/2310.11428](http://arxiv.org/abs/2310.11428)

    这项研究探究了在深度神经网络中进行行为克隆训练时出现的训练不稳定性现象。我们发现，尽管小批量SGD更新对于行为克隆损失几乎没有影响，但它会导致长期奖励的剧烈振荡。我们称这种效应为梯度方差放大（GVA），并发现使用指数移动平均（EMA）可以有效减缓这种效应。这一现象在连续控制和自回归等领域具有普遍性。

    

    本文研究了使用深度神经网络进行行为克隆的训练不稳定性。我们观察到，尽管对于行为克隆损失几乎没有影响，但在训练过程中，对策略网络的小批量SGD更新导致了长期奖励的剧烈振荡。我们通过实验证明了这些振荡的统计和计算原因，并发现它们源于小批量SGD噪声在不稳定的闭环动力学中的混沌传播。虽然SGD噪声对于单步动作预测目标是无害的，但在长期视野上它导致了灾难性的误差累积，我们称之为梯度方差放大（GVA）效应。我们发现许多标准的缓解技术不能缓解GVA，但是发现迭代的指数移动平均（EMA）在缓解GVA方面非常有效。我们通过展示连续控制和自回归中GVA的存在以及EMA减缓GVA的情况，说明了这一现象的普遍性。

    This work studies training instabilities of behavior cloning with deep neural networks. We observe that minibatch SGD updates to the policy network during training result in sharp oscillations in long-horizon rewards, despite negligibly affecting the behavior cloning loss. We empirically disentangle the statistical and computational causes of these oscillations, and find them to stem from the chaotic propagation of minibatch SGD noise through unstable closed-loop dynamics. While SGD noise is benign in the single-step action prediction objective, it results in catastrophic error accumulation over long horizons, an effect we term gradient variance amplification (GVA). We show that many standard mitigation techniques do not alleviate GVA, but find an exponential moving average (EMA) of iterates to be surprisingly effective at doing so. We illustrate the generality of this phenomenon by showing the existence of GVA and its amelioration by EMA in both continuous control and autoregressive l
    
[^4]: 更快的广义平均密集子图问题算法

    Faster Algorithms for Generalized Mean Densest Subgraph Problem. (arXiv:2310.11377v1 [cs.DS])

    [http://arxiv.org/abs/2310.11377](http://arxiv.org/abs/2310.11377)

    该论文提出了更快速的算法来解决广义平均密集子图问题，其中对于$0<p<1$的情况下，标准剥离算法可以得到$2^{1/p}$的近似解。另外，引入了一种新的广义剥离算法（GENPEEL），对于$p \geq 1$，其近似保证率为$(p+1)^{1/p}$。

    

    大图的最密子图通常指的是具有最高平均度的一些子图，这已经被Veldt等人扩展到了$p$-平均密集子图目标的家族上。$p$-平均密集子图问题寻找具有最高平均$p$次幂度的子图，而标准最密子图问题寻找具有最高平均度的子图。已经证明了当$p>1$时，标准剥离算法在广义目标上可以表现得任意差，但当$0<p<1$时不确定。在本文中，我们首次证明了标准剥离算法在$0<p<1$的情况下仍然可以得到$2^{1/p}$的近似解。Veldt等人提出了一种新的广义剥离算法（GENPEEL），对于$p \geq 1$，其近似保证率为$(p+1)^{1/p}$，时间复杂度为$O(mn)$，其中$m$和$n$分别表示图中的边数和节点数。在算法贡献方面，我们提出了一种新的算法。

    The densest subgraph of a large graph usually refers to some subgraph with the highest average degree, which has been extended to the family of $p$-means dense subgraph objectives by~\citet{veldt2021generalized}. The $p$-mean densest subgraph problem seeks a subgraph with the highest average $p$-th-power degree, whereas the standard densest subgraph problem seeks a subgraph with a simple highest average degree. It was shown that the standard peeling algorithm can perform arbitrarily poorly on generalized objective when $p>1$ but uncertain when $0<p<1$. In this paper, we are the first to show that a standard peeling algorithm can still yield $2^{1/p}$-approximation for the case $0<p < 1$. (Veldt 2021) proposed a new generalized peeling algorithm (GENPEEL), which for $p \geq 1$ has an approximation guarantee ratio $(p+1)^{1/p}$, and time complexity $O(mn)$, where $m$ and $n$ denote the number of edges and nodes in graph respectively. In terms of algorithmic contributions, we propose a ne
    
[^5]: Lie Group Decompositions for Equivariant Neural Networks. (arXiv:2310.11366v1 [cs.LG]) (等变神经网络的Lie群分解)

    Lie Group Decompositions for Equivariant Neural Networks. (arXiv:2310.11366v1 [cs.LG])

    [http://arxiv.org/abs/2310.11366](http://arxiv.org/abs/2310.11366)

    本论文提出了一种基于Lie群结构和几何特性的框架，可以处理非紧致非阿贝尔的Lie群，特别关注于$\text{GL}^{+}(n, \mathbb{R})$和$\text{SL}(n, \mathbb{R})$这两个Lie群。

    

    在训练（卷积）神经网络模型时，对几何变换的不变性和等变性被证明是非常有用的归纳偏差，特别是在低数据环境下。大部分研究集中在使用的对称群为紧致或阿贝尔群，或者两者都是。最近的研究拓展了使用的变换类别到Lie群的情况，主要通过使用其Lie代数以及群的指数和对数映射。然而，这样的方法在适用于更大的变换群时受到限制，因为根据所关心的群$G$的不同，指数映射可能不满射。当$G$既不是紧致群也不是阿贝尔群时，还会遇到进一步的限制。我们利用Lie群及其齐次空间的结构和几何特性，提出了一个可以处理这类群的框架，主要关注Lie群$G = \text{GL}^{+}(n, \mathbb{R})$和$G = \text{SL}(n, \mathbb{R}$。

    Invariance and equivariance to geometrical transformations have proven to be very useful inductive biases when training (convolutional) neural network models, especially in the low-data regime. Much work has focused on the case where the symmetry group employed is compact or abelian, or both. Recent work has explored enlarging the class of transformations used to the case of Lie groups, principally through the use of their Lie algebra, as well as the group exponential and logarithm maps. The applicability of such methods to larger transformation groups is limited by the fact that depending on the group of interest $G$, the exponential map may not be surjective. Further limitations are encountered when $G$ is neither compact nor abelian. Using the structure and geometry of Lie groups and their homogeneous spaces, we present a framework by which it is possible to work with such groups primarily focusing on the Lie groups $G = \text{GL}^{+}(n, \mathbb{R})$ and $G = \text{SL}(n, \mathbb{R}
    
[^6]: 上下文化机器学习

    Contextualized Machine Learning. (arXiv:2310.11340v1 [stat.ML])

    [http://arxiv.org/abs/2310.11340](http://arxiv.org/abs/2310.11340)

    上下文化机器学习是一种用于学习异质和上下文相关效应的新范式，通过深度学习和元关系估计异质函数，并引入上下文编码器和样本特定模型来统一现有框架。

    

    我们研究了上下文化机器学习（ML），这是一种用于学习异质和上下文相关效应的范式。上下文化ML通过将深度学习应用于上下文信息和上下文特定参数模型之间的元关系来估计异质函数。这是一种统一现有框架的变系数建模方法，包括聚类分析和队列建模，引入了两个可重用的概念：上下文编码器将样本上下文转化为模型参数，以及基于样本预测子的样本特定模型。我们回顾了开发上下文化模型的过程，从上下文化模型中进行非参数推断的方法，以及上下文化模型的可辨认条件。最后，我们介绍了开源的PyTorch软件包ContextualizedML。

    We examine Contextualized Machine Learning (ML), a paradigm for learning heterogeneous and context-dependent effects. Contextualized ML estimates heterogeneous functions by applying deep learning to the meta-relationship between contextual information and context-specific parametric models. This is a form of varying-coefficient modeling that unifies existing frameworks including cluster analysis and cohort modeling by introducing two reusable concepts: a context encoder which translates sample context into model parameters, and sample-specific model which operates on sample predictors. We review the process of developing contextualized models, nonparametric inference from contextualized models, and identifiability conditions of contextualized models. Finally, we present the open-source PyTorch package ContextualizedML.
    
[^7]: 揭示分类器引导扩散生成的设计空间

    Elucidating The Design Space of Classifier-Guided Diffusion Generation. (arXiv:2310.11311v1 [cs.LG])

    [http://arxiv.org/abs/2310.11311](http://arxiv.org/abs/2310.11311)

    本论文通过研究设计空间，提出了一种新的无训练引导方案，通过利用现成的分类器来引导扩散生成，在保持灵活性的同时实现了显著的性能提升。

    

    条件扩散生成中的引导对于样本质量和可控性非常重要。然而，现有的引导方案还有待改进。一方面，主流方法如分类器引导和无分类器引导都需要额外的标注数据训练，这既耗时又不能适应新的条件。另一方面，无训练方法如通用引导虽然更加灵活，但尚未证明具有可比性能。本研究通过对设计空间进行全面的研究，展示了通过以无训练的方式利用现成的分类器，可以在现有引导方案的基础上实现显著的性能提升，使得两者皆可兼得。我们提出了一种以校准为指导原则，通过几种预先调整的技术来更好地利用预训练的现成分类器来引导扩散生成。在ImageNet数据集上进行了大量实验。

    Guidance in conditional diffusion generation is of great importance for sample quality and controllability. However, existing guidance schemes are to be desired. On one hand, mainstream methods such as classifier guidance and classifier-free guidance both require extra training with labeled data, which is time-consuming and unable to adapt to new conditions. On the other hand, training-free methods such as universal guidance, though more flexible, have yet to demonstrate comparable performance. In this work, through a comprehensive investigation into the design space, we show that it is possible to achieve significant performance improvements over existing guidance schemes by leveraging off-the-shelf classifiers in a training-free fashion, enjoying the best of both worlds. Employing calibration as a general guideline, we propose several pre-conditioning techniques to better exploit pretrained off-the-shelf classifiers for guiding diffusion generation. Extensive experiments on ImageNet 
    
[^8]: 在高斯混合模型空间中引入了类似于Gromov-Wassertein的距离

    Gromov-Wassertein-like Distances in the Gaussian Mixture Models Space. (arXiv:2310.11256v1 [stat.ML])

    [http://arxiv.org/abs/2310.11256](http://arxiv.org/abs/2310.11256)

    本文介绍了两种在高斯混合模型空间中的Gromov-Wasserstein类型距离，分别用于评估分布之间的距离和推导最优的点分配方案。

    

    本文介绍了两种在高斯混合模型集合上的Gromov-Wasserstein类型距离。第一种距离是在高斯测度空间上两个离散分布的Gromov-Wasserstein距离。该距离可以作为Gromov-Wasserstein的替代，用于评估分布之间的距离，但不能直接推导出最优的运输方案。为了设计出这样的运输方案，我们引入了另一种在不可比较的空间中的测度之间的距离，该距离与Gromov-Wasserstein密切相关。当将允许的运输耦合限制为高斯混合模型时，这定义了另一种高斯混合模型之间的距离，可以作为Gromov-Wasserstein的另一种替代，并允许推导出最优的点分配方案。

    In this paper, we introduce two Gromov-Wasserstein-type distances on the set of Gaussian mixture models. The first one takes the form of a Gromov-Wasserstein distance between two discrete distributionson the space of Gaussian measures. This distance can be used as an alternative to Gromov-Wasserstein for applications which only require to evaluate how far the distributions are from each other but does not allow to derive directly an optimal transportation plan between clouds of points. To design a way to define such a transportation plan, we introduce another distance between measures living in incomparable spaces that turns out to be closely related to Gromov-Wasserstein. When restricting the set of admissible transportation couplings to be themselves Gaussian mixture models in this latter, this defines another distance between Gaussian mixture models that can be used as another alternative to Gromov-Wasserstein and which allows to derive an optimal assignment between points. Finally,
    
[^9]: 学习更好的采样方法

    Learning to Sample Better. (arXiv:2310.11232v1 [cs.LG])

    [http://arxiv.org/abs/2310.11232](http://arxiv.org/abs/2310.11232)

    本课程介绍了基于动态输运测度的生成建模方法的最新进展，重点讲述了如何通过数据学习这些映射，并通过正反馈循环改进蒙特卡洛采样技术。

    

    这些讲义介绍了基于动态输运测度的生成建模方法的最新进展，通过这种方法，简单基础测度的样本被映射到感兴趣目标测度的样本。特别强调这些方法在蒙特卡洛采样技术中的应用，例如重要性采样和马尔可夫链蒙特卡洛。在这种情况下，讲义展示了如何通过MC采样生成的数据变分学习这些映射，并如何利用它们通过正反馈循环改进采样。

    These lecture notes provide an introduction to recent advances in generative modeling methods based on the dynamical transportation of measures, by means of which samples from a simple base measure are mapped to samples from a target measure of interest. Special emphasis is put on the applications of these methods to Monte-Carlo (MC) sampling techniques, such as importance sampling and Markov Chain Monte-Carlo (MCMC) schemes. In this context, it is shown how the maps can be learned variationally using data generated by MC sampling, and how they can in turn be used to improve such sampling in a positive feedback loop.
    
[^10]: 具有非空泛化界限的联邦学习

    Federated Learning with Nonvacuous Generalisation Bounds. (arXiv:2310.11203v1 [cs.LG])

    [http://arxiv.org/abs/2310.11203](http://arxiv.org/abs/2310.11203)

    这项研究提出了一种新的策略来在联邦学习中训练随机预测器，通过保护每个节点的隐私并且具有数值上非空的泛化界限，可以在保持预测性能的同时实现数据共享和保护隐私。

    

    我们引入了一种新的策略来训练联邦学习中的随机预测器，在这种策略中，网络的每个节点通过发布本地预测器但对其他节点保密其训练数据集的方式来保护其隐私。然后，我们构建一个全局的随机预测器，它在PAC-Bayesian泛化界限的意义上继承了本地私有预测器的属性。我们考虑了同步情况，即所有节点共享相同的训练目标（从泛化界限导出），以及异步情况，即每个节点可以有自己的个性化训练目标。通过一系列的数值实验，我们证明了我们的方法实现了与将所有数据集共享给所有节点的批处理方法相当的预测性能。此外，这些预测器支持着在保护每个节点隐私的同时具有数值上非空的泛化界限。我们明确地计算了预测性能的增量。

    We introduce a novel strategy to train randomised predictors in federated learning, where each node of the network aims at preserving its privacy by releasing a local predictor but keeping secret its training dataset with respect to the other nodes. We then build a global randomised predictor which inherits the properties of the local private predictors in the sense of a PAC-Bayesian generalisation bound. We consider the synchronous case where all nodes share the same training objective (derived from a generalisation bound), and the asynchronous case where each node may have its own personalised training objective. We show through a series of numerical experiments that our approach achieves a comparable predictive performance to that of the batch approach where all datasets are shared across nodes. Moreover the predictors are supported by numerically nonvacuous generalisation bounds while preserving privacy for each node. We explicitly compute the increment on predictive performance an
    
[^11]: 一种基于机器学习的概率暴露模型的德国高分辨率室内氡气地图

    A new high-resolution indoor radon map for Germany using a machine learning based probabilistic exposure model. (arXiv:2310.11143v1 [stat.ML])

    [http://arxiv.org/abs/2310.11143](http://arxiv.org/abs/2310.11143)

    本研究提出了一种基于机器学习的概率暴露模型，可以更准确地估计德国室内氡气分布，并具有更高的空间分辨率。

    

    室内氡气是一种致癌的放射性气体，可以在室内积累。通常情况下，全国范围内的室内氡暴露是基于广泛的测量活动估计得来的。然而，样本的特征往往与人口特征不同，这是由于许多相关因素，如地质源氡气的可用性或楼层水平。此外，样本大小通常不允许以高空间分辨率进行暴露估计。我们提出了一种基于模型的方法，可以比纯数据方法更加现实地估计室内氡分布，并具有更高的空间分辨率。我们采用了两阶段建模方法：1）应用分位数回归森林，使用环境和建筑数据作为预测因子，估计了德国每个住宅楼的每个楼层的室内氡概率分布函数；2）使用概率蒙特卡罗抽样技术使它们组合和。

    Radon is a carcinogenic, radioactive gas that can accumulate indoors. Indoor radon exposure at the national scale is usually estimated on the basis of extensive measurement campaigns. However, characteristics of the sample often differ from the characteristics of the population due to the large number of relevant factors such as the availability of geogenic radon or floor level. Furthermore, the sample size usually does not allow exposure estimation with high spatial resolution. We propose a model-based approach that allows a more realistic estimation of indoor radon distribution with a higher spatial resolution than a purely data-based approach. We applied a two-stage modelling approach: 1) a quantile regression forest using environmental and building data as predictors was applied to estimate the probability distribution function of indoor radon for each floor level of each residential building in Germany; (2) a probabilistic Monte Carlo sampling technique enabled the combination and
    
[^12]: 敏感性感知的摊销贝叶斯推断

    Sensitivity-Aware Amortized Bayesian Inference. (arXiv:2310.11122v1 [stat.ML])

    [http://arxiv.org/abs/2310.11122](http://arxiv.org/abs/2310.11122)

    本文提出了一种敏感性感知的摊销贝叶斯推断方法，通过权重共享和神经网络来进行似然和先验规范的训练，以及对数据扰动和预处理程序的敏感性评估。

    

    贝叶斯推断是在不确定性下进行概率推理和决策的强大框架。现代贝叶斯工作流程中的基本选择涉及似然函数和先验分布的规范、后验逼近器和数据。每个选择都可以显着影响基于模型的推断和后续决策，因此需要进行敏感性分析。在这项工作中，我们提出了一种多方面的方法，将敏感性分析整合到摊销贝叶斯推断（ABI，即基于神经网络的模拟推断）中。首先，我们利用权重共享在训练过程中编码替代似然和先验规范之间的结构相似性，以最小的计算开销。其次，我们利用神经网络的快速推断来评估对各种数据扰动或预处理程序的敏感性。与大多数其他贝叶斯方法相比，这两个步骤都避免了昂贵的计算。

    Bayesian inference is a powerful framework for making probabilistic inferences and decisions under uncertainty. Fundamental choices in modern Bayesian workflows concern the specification of the likelihood function and prior distributions, the posterior approximator, and the data. Each choice can significantly influence model-based inference and subsequent decisions, thereby necessitating sensitivity analysis. In this work, we propose a multifaceted approach to integrate sensitivity analyses into amortized Bayesian inference (ABI, i.e., simulation-based inference with neural networks). First, we utilize weight sharing to encode the structural similarities between alternative likelihood and prior specifications in the training process with minimal computational overhead. Second, we leverage the rapid inference of neural networks to assess sensitivity to various data perturbations or pre-processing procedures. In contrast to most other Bayesian approaches, both steps circumvent the costly
    
[^13]: 最小信息线性判别分析：使用未标记数据训练LDA模型

    Minimally Informed Linear Discriminant Analysis: training an LDA model with unlabelled data. (arXiv:2310.11110v1 [cs.LG])

    [http://arxiv.org/abs/2310.11110](http://arxiv.org/abs/2310.11110)

    本文展示了在只有未标记数据的情况下，通过一些最小的先验信息，可以计算出精确的LDA投影向量。数值实验验证了这种最小信息的线性判别分析（MILDA）模型与有监督的LDA模型的性能接近。

    

    线性判别分析（LDA）是最古老且最流行的线性方法之一，用于有监督分类问题。本文证明，如果有一些最小的先验信息，那么可以基于未标记数据计算出LDA模型的精确投影向量。更具体地说，我们展示了只需要以下三个信息中的任意一个即可计算LDA投影向量，如果只有未标记数据可用：（1）两个类别中任意一个的类别平均值，（2）两个类别平均值之间的差异（经过缩放），或者（3）类别协方差矩阵（经过缩放）。这些理论结果在数值实验中得到了验证，证明这种最小信息的线性判别分析（MILDA）模型与有监督的LDA模型的性能非常接近。此外，我们还展示了MILDA投影向量可以通过一个封闭形式计算出来，并且计算成本与LDA相当。

    Linear Discriminant Analysis (LDA) is one of the oldest and most popular linear methods for supervised classification problems. In this paper, we demonstrate that it is possible to compute the exact projection vector from LDA models based on unlabelled data, if some minimal prior information is available. More precisely, we show that only one of the following three pieces of information is actually sufficient to compute the LDA projection vector if only unlabelled data are available: (1) the class average of one of the two classes, (2) the difference between both class averages (up to a scaling), or (3) the class covariance matrices (up to a scaling). These theoretical results are validated in numerical experiments, demonstrating that this minimally informed Linear Discriminant Analysis (MILDA) model closely matches the performance of a supervised LDA model. Furthermore, we show that the MILDA projection vector can be computed in a closed form with a computational cost comparable to LD
    
[^14]: 低成本重采样随机梯度下降用于高效不确定性量化

    Resampling Stochastic Gradient Descent Cheaply for Efficient Uncertainty Quantification. (arXiv:2310.11065v1 [stat.ML])

    [http://arxiv.org/abs/2310.11065](http://arxiv.org/abs/2310.11065)

    本研究提出了两种低成本重采样的方法，用于构建随机梯度下降解的置信区间，这一方法可以有效减少计算工作量，并绕过现有方法中的混合条件。

    

    随机梯度下降（SGD）或随机逼近在模型训练和随机优化中被广泛使用。虽然有大量关于其收敛性分析的文献，但对从SGD获得的解进行推断的研究只是最近才开始，但由于对不确定性量化的日益需求而变得重要。我们研究了两种计算上廉价的基于重采样的方法来构建SGD解的置信区间。一个方法通过从数据中进行替换重采样来使用多个但少量的SGD并行进行操作，另一个方法以在线方式进行操作。我们的方法可以被视为对已建立的Bootstrap方案进行增强，以显着减少重采样需求方面的计算工作量，同时绕过现有批处理方法中复杂的混合条件。我们通过最近的所谓低成本bootstrap思想和SGD的Berry-Esseen型边界来实现这些目标。

    Stochastic gradient descent (SGD) or stochastic approximation has been widely used in model training and stochastic optimization. While there is a huge literature on analyzing its convergence, inference on the obtained solutions from SGD has only been recently studied, yet is important due to the growing need for uncertainty quantification. We investigate two computationally cheap resampling-based methods to construct confidence intervals for SGD solutions. One uses multiple, but few, SGDs in parallel via resampling with replacement from the data, and another operates this in an online fashion. Our methods can be regarded as enhancements of established bootstrap schemes to substantially reduce the computation effort in terms of resampling requirements, while at the same time bypassing the intricate mixing conditions in existing batching methods. We achieve these via a recent so-called cheap bootstrap idea and Berry-Esseen-type bound for SGD.
    
[^15]: 矩阵压缩通过随机低秩低精度因式分解

    Matrix Compression via Randomized Low Rank and Low Precision Factorization. (arXiv:2310.11028v1 [cs.LG])

    [http://arxiv.org/abs/2310.11028](http://arxiv.org/abs/2310.11028)

    通过随机化的低秩和低精度因式分解，我们提出了一种矩阵压缩算法，可以有效地减小存储和处理大型矩阵所需的计算资源和内存使用。

    

    矩阵在各个研究领域中都非常有用，因为它们提供了一种方便的框架，可以以结构化的方式组织和操作数据。然而，现代矩阵可能包含数十亿个元素，使得它们的存储和处理对计算资源和内存使用要求很高。虽然这些矩阵非常大，但它们通常是近似低秩的。我们提出了一种算法，利用这种结构来获得任何矩阵 $\mathbf{A}$ 的低秩分解，即 $\mathbf{A} \approx \mathbf{L}\mathbf{R}$，其中 $\mathbf{L}$ 和 $\mathbf{R}$ 是低秩因子。$\mathbf{L}$ 和 $\mathbf{R}$ 中的元素总数可以显著少于 $\mathbf{A}$ 中的元素总数。此外，$\mathbf{L}$ 和 $\mathbf{R}$ 的条目被量化为低精度格式 $--$ 通过给出低秩和低精度因式分解来压缩 $\mathbf{A}$。我们的算法首先计算 $\mathbf$

    Matrices are exceptionally useful in various fields of study as they provide a convenient framework to organize and manipulate data in a structured manner. However, modern matrices can involve billions of elements, making their storage and processing quite demanding in terms of computational resources and memory usage. Although prohibitively large, such matrices are often approximately low rank. We propose an algorithm that exploits this structure to obtain a low rank decomposition of any matrix $\mathbf{A}$ as $\mathbf{A} \approx \mathbf{L}\mathbf{R}$, where $\mathbf{L}$ and $\mathbf{R}$ are the low rank factors. The total number of elements in $\mathbf{L}$ and $\mathbf{R}$ can be significantly less than that in $\mathbf{A}$. Furthermore, the entries of $\mathbf{L}$ and $\mathbf{R}$ are quantized to low precision formats $--$ compressing $\mathbf{A}$ by giving us a low rank and low precision factorization. Our algorithm first computes an approximate basis of the range space of $\mathb
    
[^16]: 从可识别的因果表示到可控的反事实生成：因果生成建模综述

    From Identifiable Causal Representations to Controllable Counterfactual Generation: A Survey on Causal Generative Modeling. (arXiv:2310.11011v1 [cs.LG])

    [http://arxiv.org/abs/2310.11011](http://arxiv.org/abs/2310.11011)

    本文综述了因果生成建模的技术，其中分为因果表示学习和可控反事实生成两个部分，这些模型融合了因果理论，解决了深度生成模型的一些根本性缺点，并提供了分布偏移鲁棒性、公平性和互操作性等有益属性。

    

    深度生成模型在数据密度估计和从有限样本中生成数据方面取得了巨大的成功。然而，这些模型存在一些根本性的缺点，如缺乏可解释性、引入虚假相关性和差劲的超出分布的外推能力。为了解决这些挑战，可以将因果理论融入深度生成建模中。结构因果模型描述了数据生成过程，并对系统中变量之间的复杂因果关系和机制进行建模。因此，结构因果模型可以与深度生成模型自然地结合。因果模型为深度生成模型提供了几个有益的属性，如分布偏移鲁棒性、公平性和互操作性。本文提供了对因果生成建模的技术综述，分为因果表示学习和可控反事实生成两个部分。

    Deep generative models have shown tremendous success in data density estimation and data generation from finite samples. While these models have shown impressive performance by learning correlations among features in the data, some fundamental shortcomings are their lack of explainability, the tendency to induce spurious correlations, and poor out-of-distribution extrapolation. In an effort to remedy such challenges, one can incorporate the theory of causality in deep generative modeling. Structural causal models (SCMs) describe data-generating processes and model complex causal relationships and mechanisms among variables in a system. Thus, SCMs can naturally be combined with deep generative models. Causal models offer several beneficial properties to deep generative models, such as distribution shift robustness, fairness, and interoperability. We provide a technical survey on causal generative modeling categorized into causal representation learning and controllable counterfactual ge
    
[^17]: WGoM：一种适用于带加权响应的分类数据的新模型

    WGoM: A novel model for categorical data with weighted responses. (arXiv:2310.10989v1 [cs.SI])

    [http://arxiv.org/abs/2310.10989](http://arxiv.org/abs/2310.10989)

    本文提出了一种名为加权成员级别（WGoM）模型，用于解决基于分类数据的潜在类别推断问题。与现有模型相比，WGoM更通用且适用于具有连续或负响应的数据集。通过提出的算法，我们能够准确高效地估计潜在混合成员和其他WGoM参数，并且通过实验证明了该算法的性能和实用潜力。

    

    Graded of Membership（GoM）模型是一种用于推断分类数据中潜在类别的强大工具，使得个体可以属于多个潜在类别。然而，该模型仅适用于具有非负整数响应的分类数据，使得它无法应用于具有连续或负响应的数据集。为了解决这个限制，本文提出了一种名为加权成员级别（WGoM）模型的新模型。与GoM相比，我们的WGoM在响应矩阵的生成上放宽了GoM的分布约束，并且比GoM更通用。我们还提出了一种估计潜在混合成员和其他WGoM参数的算法。我们推导了估计参数的误差界限，并且证明了算法的统计一致性。该算法的性能在合成和真实世界的数据集中得到了验证。结果表明我们的算法准确高效，具有很高的实用潜力。

    The Graded of Membership (GoM) model is a powerful tool for inferring latent classes in categorical data, which enables subjects to belong to multiple latent classes. However, its application is limited to categorical data with nonnegative integer responses, making it inappropriate for datasets with continuous or negative responses. To address this limitation, this paper proposes a novel model named the Weighted Grade of Membership (WGoM) model. Compared with GoM, our WGoM relaxes GoM's distribution constraint on the generation of a response matrix and it is more general than GoM. We then propose an algorithm to estimate the latent mixed memberships and the other WGoM parameters. We derive the error bounds of the estimated parameters and show that the algorithm is statistically consistent. The algorithmic performance is validated in both synthetic and real-world datasets. The results demonstrate that our algorithm is accurate and efficient, indicating its high potential for practical a
    
[^18]: 具有加权响应的潜在类别分析

    Latent class analysis with weighted responses. (arXiv:2310.10984v1 [cs.SI])

    [http://arxiv.org/abs/2310.10984](http://arxiv.org/abs/2310.10984)

    提出了一种新的生成模型，即加权潜在类别模型（WLCM），可以用于对具有连续或负响应的真实世界数据进行潜在类别分析。

    

    潜在类别模型被提议作为对社会、心理、行为和生物科学等各个领域中的分类数据进行聚类分析的强大工具。然而，潜在类别模型的一个重要限制是它只适用于具有二进制响应的数据，使其无法对具有连续或负响应的真实世界数据进行建模。在许多应用中，忽视权重会丢失掉在权重中包含的许多潜在有价值信息。为了解决这个限制，我们提出了一种新的生成模型，即加权潜在类别模型（WLCM）。我们的模型允许通过潜在类别结构从任意分布生成数据的响应矩阵。与潜在类别模型相比，我们的WLCM更加真实和通用。据我们所知，我们的WLCM是第一个用于具有加权响应的潜在类别分析的模型。我们研究了该模型的可辨识性，并提出了一种高效的算法。

    The latent class model has been proposed as a powerful tool for cluster analysis of categorical data in various fields such as social, psychological, behavioral, and biological sciences. However, one important limitation of the latent class model is that it is only suitable for data with binary responses, making it fail to model real-world data with continuous or negative responses. In many applications, ignoring the weights throws out a lot of potentially valuable information contained in the weights. To address this limitation, we propose a novel generative model, the weighted latent class model (WLCM). Our model allows data's response matrix to be generated from an arbitrary distribution with a latent class structure. In comparison to the latent class model, our WLCM is more realistic and more general. To our knowledge, our WLCM is the first model for latent class analysis with weighted responses. We investigate the identifiability of the model and propose an efficient algorithm for
    
[^19]: 使用信息特征向量和核字典学习的MRI脑肿瘤分割

    MRI brain tumor segmentation using informative feature vectors and kernel dictionary learning. (arXiv:2310.10963v1 [cs.CV])

    [http://arxiv.org/abs/2310.10963](http://arxiv.org/abs/2310.10963)

    本文提出一种基于核字典学习算法的方法，利用统计特征向量和相关性样本选择技术，在MRI中实现了脑肿瘤的有效分割。

    

    本文提出了一种基于核字典学习算法的方法，用于在磁共振成像（MRI）中分割脑肿瘤区域。从脑MRI扫描中的像素周围大小为3×3的补丁中提取一组一阶和二阶统计特征向量。利用这些特征向量分别训练两个核字典，用于健康和肿瘤组织的分割。为了提高字典的效率和减少训练时间，开发了一种基于相关性的样本选择技术，用于识别最具信息量和区分度的特征向量子集。该技术旨在通过选择一组特征向量来提供有价值的分割任务信息，从而改善字典的性能。随后，利用线性分类器基于学习到的字典来区分健康像素和异常像素。结果表明，所提出的方法优于其他现有方法。

    This paper presents a method based on a kernel dictionary learning algorithm for segmenting brain tumor regions in magnetic resonance images (MRI). A set of first-order and second-order statistical feature vectors are extracted from patches of size 3 * 3 around pixels in the brain MRI scans. These feature vectors are utilized to train two kernel dictionaries separately for healthy and tumorous tissues. To enhance the efficiency of the dictionaries and reduce training time, a correlation-based sample selection technique is developed to identify the most informative and discriminative subset of feature vectors. This technique aims to improve the performance of the dictionaries by selecting a subset of feature vectors that provide valuable information for the segmentation task. Subsequently, a linear classifier is utilized to distinguish between healthy and unhealthy pixels based on the learned dictionaries. The results demonstrate that the proposed method outperforms other existing metho
    
[^20]: 基于局部图界限的采样型图神经网络的视角

    A Local Graph Limits Perspective on Sampling-Based GNNs. (arXiv:2310.10953v1 [cs.LG])

    [http://arxiv.org/abs/2310.10953](http://arxiv.org/abs/2310.10953)

    该论文提出了一种基于局部图界限的训练大型输入图的采样型图神经网络的理论框架，通过对小样本的训练，我们可以获得与整个图训练类似的结果。这为使用采样训练GNN提供了新的理论理解，并提供了在选择最佳模型、超参数和采样算法方面更高效的方法。

    

    我们提出了一个理论框架，通过对大型输入图中的小型固定大小的采样子图进行训练，来训练图神经网络（GNN）。该框架适用于各种模型，包括常用的基于采样的GNN，如GraphSAGE和FastGCN。借助图的局部界限理论，我们证明，在温和的假设下，通过对大型输入图的小样本进行采样训练的参数与在整个图上训练相同结构的参数在ε-邻域内。我们以ε的函数推导出样本数量、图的大小和训练步骤的界限。我们的结果为训练GNN时使用采样提供了一种新颖的理论理解。它们还暗示，通过对输入图的小样本进行训练，从业者可以更高效地识别和选择最佳模型、超参数和采样算法。我们通过实验证明了我们的结果。

    We propose a theoretical framework for training Graph Neural Networks (GNNs) on large input graphs via training on small, fixed-size sampled subgraphs. This framework is applicable to a wide range of models, including popular sampling-based GNNs, such as GraphSAGE and FastGCN. Leveraging the theory of graph local limits, we prove that, under mild assumptions, parameters learned from training sampling-based GNNs on small samples of a large input graph are within an $\epsilon$-neighborhood of the outcome of training the same architecture on the whole graph. We derive bounds on the number of samples, the size of the graph, and the training steps required as a function of $\epsilon$. Our results give a novel theoretical understanding for using sampling in training GNNs. They also suggest that by training GNNs on small samples of the input graph, practitioners can identify and select the best models, hyperparameters, and sampling algorithms more efficiently. We empirically illustrate our re
    
[^21]: 限制的Tweedie随机块模型

    Restricted Tweedie Stochastic Block Models. (arXiv:2310.10952v1 [stat.ML])

    [http://arxiv.org/abs/2310.10952](http://arxiv.org/abs/2310.10952)

    这项研究提出了一种新的随机块模型，可以处理由非负零膨胀连续边权组成的邻接矩阵，特别适用于模拟国际贸易网络。该模型结合了节点信息和动态效应，并且可以独立于社区标签进行参数估计。一个高效的两步算法被开发用于估计协变效应和社区标签。

    

    随机块模型 (SBM) 是在网络中进行社区检测的广泛应用框架，其中网络结构通常由邻接矩阵表示。然而，传统的SBM不能直接应用于由非负的零膨胀连续边权组成的邻接矩阵。为了模拟国际贸易网络，其中边权表示国家之间的贸易价值，我们提出了一种基于限制Tweedie分布的创新SBM。此外，我们还结合了节点信息，如国家之间的地理距离，并考虑其对边权的动态影响。值得注意的是，我们证明在节点数足够大的情况下，估计这个协变效应时，可以独立于每个节点的社区标签，在计算我们模型参数的最大似然估计器时。这个结果使得我们能够开发一种高效的两步算法，将协变效应的估计与社区标签的估计分离开来。

    The stochastic block model (SBM) is a widely used framework for community detection in networks, where the network structure is typically represented by an adjacency matrix. However, conventional SBMs are not directly applicable to an adjacency matrix that consists of non-negative zero-inflated continuous edge weights. To model the international trading network, where edge weights represent trading values between countries, we propose an innovative SBM based on a restricted Tweedie distribution. Additionally, we incorporate nodal information, such as the geographical distance between countries, and account for its dynamic effect on edge weights. Notably, we show that given a sufficiently large number of nodes, estimating this covariate effect becomes independent of community labels of each node when computing the maximum likelihood estimator of parameters in our model. This result enables the development of an efficient two-step algorithm that separates the estimation of covariate effe
    
[^22]: 针对跳跃不连续函数的替代主动子空间

    Surrogate Active Subspaces for Jump-Discontinuous Functions. (arXiv:2310.10907v1 [stat.ML])

    [http://arxiv.org/abs/2310.10907](http://arxiv.org/abs/2310.10907)

    该论文提出了一种针对不连续函数的替代主动子空间方法，扩展了活跃子空间的应用范围，并通过数值实验验证了该方法的有效性。

    

    替代建模和活跃子空间已经成为计算科学和工程领域的强大范例。将这些技术应用于社会科学中的计算模型，突显了它们在处理离散输出的Agent-Based模型等不连续模拟器时的局限性。然而，之前的应用研究已经表明，对于这类估计器，替代计算的活跃子空间可以产生有趣的结果。但是，由于活跃子空间是通过梯度定义的，当将该方法应用于不连续模拟器时，估计的是什么量还不清楚。本文首先展示了进行此类分析时可能出现的一些病态情况。这促使我们将活跃子空间扩展到不连续函数上，澄清了在此类分析中实际估计的内容。我们还对合成测试函数进行了数值实验，比较了活跃子空间的高斯过程估计。

    Surrogate modeling and active subspaces have emerged as powerful paradigms in computational science and engineering. Porting such techniques to computational models in the social sciences brings into sharp relief their limitations in dealing with discontinuous simulators, such as Agent-Based Models, which have discrete outputs. Nevertheless, prior applied work has shown that surrogate estimates of active subspaces for such estimators can yield interesting results. But given that active subspaces are defined by way of gradients, it is not clear what quantity is being estimated when this methodology is applied to a discontinuous simulator. We begin this article by showing some pathologies that can arise when conducting such an analysis. This motivates an extension of active subspaces to discontinuous functions, clarifying what is actually being estimated in such analyses. We also conduct numerical experiments on synthetic test functions to compare Gaussian process estimates of active sub
    
[^23]: 近似性质的切片匹配算子

    Approximation properties of slice-matching operators. (arXiv:2310.10869v1 [math.NA])

    [http://arxiv.org/abs/2310.10869](http://arxiv.org/abs/2310.10869)

    本文研究了迭代的切片匹配算法的近似性质，并探讨了与源度量、目标度量和切片方向相关的属性。研究结果包括与源度量相关的不变性属性、与目标度量相关的等变性属性以及与切片方向相关的Lipschitz连续性。此外，还给出了通过一步切片匹配方案逼近目标度量的误差界限，并研究了切片匹配算子恢复最优输运映射的情况。

    

    迭代的切片匹配算法是一种有效的将源度量转换为目标度量的方法，尤其适用于高维情况。这些算法已成功应用于颜色转换和形状检索等领域，并且在正则性假设下保证收敛。在本文中，我们通过研究与源度量、目标度量和切片方向有关的一个相关的切片匹配算子，探讨了这些迭代方案的单步近似性质。特别是，我们证明了与源度量相关的不变性属性，与目标度量相关的等变性属性以及与切片方向相关的Lipschitz连续性。此外，我们还确定了通过一步切片匹配方案逼近目标度量的误差界限，并表征了切片匹配算子在恢复两个度量之间的最优输运映射的情况。

    Iterative slice-matching procedures are efficient schemes for transferring a source measure to a target measure, especially in high dimensions. These schemes have been successfully used in applications such as color transfer and shape retrieval, and are guaranteed to converge under regularity assumptions. In this paper, we explore approximation properties related to a single step of such iterative schemes by examining an associated slice-matching operator, depending on a source measure, a target measure, and slicing directions. In particular, we demonstrate an invariance property with respect to the source measure, an equivariance property with respect to the target measure, and Lipschitz continuity concerning the slicing directions. We furthermore establish error bounds corresponding to approximating the target measure by one step of the slice-matching scheme and characterize situations in which the slice-matching operator recovers the optimal transport map between two measures. We al
    
[^24]: 通过使用高斯混合模型和蒙特卡洛自回归流进行概率分类的密度估计

    Probabilistic Classification by Density Estimation Using Gaussian Mixture Model and Masked Autoregressive Flow. (arXiv:2310.10843v1 [stat.ML])

    [http://arxiv.org/abs/2310.10843](http://arxiv.org/abs/2310.10843)

    本研究提出了一种使用密度估计进行概率分类的方法，通过使用高斯混合模型和蒙特卡洛自回归流对数据的似然进行建模，并展示了这种方法优于传统的分类器。这项工作为基于联合密度估计的其他概率分类器的提出开辟了新的研究方向。

    

    密度估计是一类重要的概率机器学习问题，它用于估计数据的分布。其中一类密度估计器是混合模型，如通过期望最大化得到的高斯混合模型（GMM）。另一类密度估计器是生成模型，它们从输入的潜变量生成数据。其中一种生成模型是蒙特卡洛自回归流（MAF），它利用归一化流和自回归网络。本文中，我们将密度估计器用于分类，尽管它们通常用于估计数据的分布。我们使用密度估计器（具体来说是GMM和MAF）对数据的类别的似然进行建模。所提出的分类器优于仅使用单个高斯分布对似然进行建模的较简单的分类器，如线性判别分析。这项工作为提出基于联合密度估计的其他概率分类器开辟了研究空间。

    Density estimation, which estimates the distribution of data, is an important category of probabilistic machine learning. A family of density estimators is mixture models, such as Gaussian Mixture Model (GMM) by expectation maximization. Another family of density estimators is the generative models which generate data from input latent variables. One of the generative models is the Masked Autoregressive Flow (MAF) which makes use of normalizing flows and autoregressive networks. In this paper, we use the density estimators for classification, although they are often used for estimating the distribution of data. We model the likelihood of classes of data by density estimation, specifically using GMM and MAF. The proposed classifiers outperform simpler classifiers such as linear discriminant analysis which model the likelihood using only a single Gaussian distribution. This work opens the research door for proposing other probabilistic classifiers based on joint density estimation.
    
[^25]: 对对抗训练线性回归的正则化性质的研究

    Regularization properties of adversarially-trained linear regression. (arXiv:2310.10807v1 [stat.ML])

    [http://arxiv.org/abs/2310.10807](http://arxiv.org/abs/2310.10807)

    本研究对对抗训练线性回归的正则化性质进行了研究，发现在过参数化情况下，对抗训练可以得到最小范数插值解，这一发现对理解对抗训练的效果和应用具有重要意义。

    

    最先进的机器学习模型对于由对手构造的非常小的输入扰动可能存在漏洞。对抗训练是一种有效的防御方法。它将问题建模为一个极小极大问题，在训练数据受到最坏情况攻击时寻找最佳解决方案。线性模型是可以观察到漏洞的简单模型，也是我们研究的重点。在这种情况下，对抗训练导致一个凸优化问题，可以形式化为有限和的最小化。我们对线性回归中对抗训练的解与其他正则化方法进行了比较分析。我们的主要发现是：（A）只要最大扰动半径小于阈值，对抗训练可以得到在过参数化情况下（参数数目大于数据数目）的最小范数插值解；相反，最小范数插值器就是通过对抗训练得到的解。

    State-of-the-art machine learning models can be vulnerable to very small input perturbations that are adversarially constructed. Adversarial training is an effective approach to defend against it. Formulated as a min-max problem, it searches for the best solution when the training data were corrupted by the worst-case attacks. Linear models are among the simple models where vulnerabilities can be observed and are the focus of our study. In this case, adversarial training leads to a convex optimization problem which can be formulated as the minimization of a finite sum. We provide a comparative analysis between the solution of adversarial training in linear regression and other regularization methods. Our main findings are that: (A) Adversarial training yields the minimum-norm interpolating solution in the overparameterized regime (more parameters than data), as long as the maximum disturbance radius is smaller than a threshold. And, conversely, the minimum-norm interpolator is the solu
    
[^26]: 神经切向核函数为具有交叉协方差图的图神经网络提供了动力学

    Neural Tangent Kernels Motivate Graph Neural Networks with Cross-Covariance Graphs. (arXiv:2310.10791v1 [cs.LG])

    [http://arxiv.org/abs/2310.10791](http://arxiv.org/abs/2310.10791)

    本文研究了神经切向核函数（NTKs）在图神经网络（GNNs）中的应用。我们发现优化对齐等价于优化GNN中的图表示或图移位运算符，并建立了对于两层GNN对齐的最优性的理论保证。

    

    神经切向核函数（NTKs）提供了分析过参数化神经网络的学习和泛化行为的理论基础。对于有监督学习任务，NTK核函数的特征向量与给定数据之间的关联（在本文中称为对齐）可以控制梯度下降的收敛速度以及对未见数据的泛化能力。在这个概念的基础上，我们研究了NTKs和对齐在图神经网络（GNNs）的背景下的应用，我们的分析揭示了优化对齐等价于优化GNN中的图表示或图移位运算符。我们的结果进一步建立了对于两层GNN对齐的最优性的理论保证，这些保证由图移位运算符作为输入和输出数据之间的交叉协方差函数的函数所决定。通过对NTKs的分析得出的理论洞察力，通过我们的实验证实了这些洞察力。

    Neural tangent kernels (NTKs) provide a theoretical regime to analyze the learning and generalization behavior of over-parametrized neural networks. For a supervised learning task, the association between the eigenvectors of the NTK kernel and given data (a concept referred to as alignment in this paper) can govern the rate of convergence of gradient descent, as well as generalization to unseen data. Building upon this concept, we investigate NTKs and alignment in the context of graph neural networks (GNNs), where our analysis reveals that optimizing alignment translates to optimizing the graph representation or the graph shift operator in a GNN. Our results further establish the theoretical guarantees on the optimality of the alignment for a two-layer GNN and these guarantees are characterized by the graph shift operator being a function of the cross-covariance between the input and the output data. The theoretical insights drawn from the analysis of NTKs are validated by our experime
    
[^27]: 广义神经网络作为高斯过程：来自深度平衡模型的启示

    Wide Neural Networks as Gaussian Processes: Lessons from Deep Equilibrium Models. (arXiv:2310.10767v1 [cs.LG])

    [http://arxiv.org/abs/2310.10767](http://arxiv.org/abs/2310.10767)

    本文研究了神经网络中广义神经网络和高斯过程的对应关系，发现具有无限深度层并且宽度趋近于无穷大的神经网络收敛于高斯过程，揭示了广义神经网络的良性过拟合现象。

    

    具有宽度层的神经网络由于与高斯过程的等价性而受到极大关注，在保持泛化性能的同时完美拟合训练数据，这被称为良性过拟合。然而，现有的结果主要集中在浅层或有限深度的网络上，需要对具有无限深度层的广义神经网络进行全面分析，例如神经常微分方程(ODE)和深度平衡模型(DEQ)。在本文中，我们特别研究了深度平衡模型(DEQ)，它是一个具有共享权重矩阵的无限深度神经网络。我们的分析揭示了当DEQ层的宽度趋近于无穷大时，它收敛到一个高斯过程，从而建立了所谓的神经网络与高斯过程(NNGP)的对应关系。值得注意的是，即使深度和宽度的极限互换，在典型的无限深度多层网络中也不会观察到这种收敛。

    Neural networks with wide layers have attracted significant attention due to their equivalence to Gaussian processes, enabling perfect fitting of training data while maintaining generalization performance, known as benign overfitting. However, existing results mainly focus on shallow or finite-depth networks, necessitating a comprehensive analysis of wide neural networks with infinite-depth layers, such as neural ordinary differential equations (ODEs) and deep equilibrium models (DEQs). In this paper, we specifically investigate the deep equilibrium model (DEQ), an infinite-depth neural network with shared weight matrices across layers. Our analysis reveals that as the width of DEQ layers approaches infinity, it converges to a Gaussian process, establishing what is known as the Neural Network and Gaussian Process (NNGP) correspondence. Remarkably, this convergence holds even when the limits of depth and width are interchanged, which is not observed in typical infinite-depth Multilayer 
    
[^28]: Mori-Zwanzig潜变空间Koopman闭包用于非线性自编码器

    Mori-Zwanzig latent space Koopman closure for nonlinear autoencoder. (arXiv:2310.10745v1 [cs.LG])

    [http://arxiv.org/abs/2310.10745](http://arxiv.org/abs/2310.10745)

    本研究提出了一种名为Mori-Zwanzig自编码器（MZ-AE）的新方法，用于在低维空间中稳健地逼近Koopman算子，通过非线性自编码器和Mori-Zwanzig形式主义的集成实现对有限不变Koopman子空间的逼近，从而增强了精确性和准确预测复杂系统行为的能力。

    

    Koopman算子提供了一种吸引人的方法来实现非线性系统的全局线性化，使其成为简化复杂动力学理解的宝贵方法。虽然数据驱动的方法在逼近有限Koopman算子方面表现出了潜力，但它们面临着各种挑战，例如选择合适的可观察量、降维和准确预测复杂系统行为的能力。本研究提出了一种名为Mori-Zwanzig自编码器（MZ-AE）的新方法，用于在低维空间中稳健地逼近Koopman算子。所提出的方法利用非线性自编码器提取关键可观察量来逼近有限不变Koopman子空间，并利用Mori-Zwanzig形式主义集成非马尔可夫校正机制。因此，该方法在非线性自编码器的潜变流形中产生了动力学的封闭表示，从而提高了精确性和...

    The Koopman operator presents an attractive approach to achieve global linearization of nonlinear systems, making it a valuable method for simplifying the understanding of complex dynamics. While data-driven methodologies have exhibited promise in approximating finite Koopman operators, they grapple with various challenges, such as the judicious selection of observables, dimensionality reduction, and the ability to predict complex system behaviours accurately. This study presents a novel approach termed Mori-Zwanzig autoencoder (MZ-AE) to robustly approximate the Koopman operator in low-dimensional spaces. The proposed method leverages a nonlinear autoencoder to extract key observables for approximating a finite invariant Koopman subspace and integrates a non-Markovian correction mechanism using the Mori-Zwanzig formalism. Consequently, this approach yields a closed representation of dynamics within the latent manifold of the nonlinear autoencoder, thereby enhancing the precision and s
    
[^29]: TacticAI:一种足球战术的人工智能助手

    TacticAI: an AI assistant for football tactics. (arXiv:2310.10553v1 [cs.LG])

    [http://arxiv.org/abs/2310.10553](http://arxiv.org/abs/2310.10553)

    提出了TacticAI，一种与利物浦足球俱乐部的领域专家密切合作开发和评价的AI足球战术助手。TacticAI能够通过预测和生成的方式帮助教练们分析角球情况，并为每个角球惯例选择成功可能性最高的球员配置。

    

    辨别对手团队实施的战术关键模式并开发有效的应对方法是现代足球的核心问题。然而，以算法的方式来解决这个问题仍是一个未解决的研究挑战。为了解决这个需求，我们提出了TacticAI，一种与利物浦足球俱乐部的领域专家密切合作开发和评价的AI足球战术助手。我们专注于分析角球，因为它们给教练们提供了直接的干预和改进机会。TacticAI包含了一个预测和生成的组件，使教练能够有效地采样和探索每个角球惯例的替代球员配置，并选择那些预测成功可能性最高的。我们通过一些相关的基准任务对TacticAI进行了验证：预测接收球员和射门尝试以及推荐球员位置调整。TacticAI的实用性通过与利物浦足球领域专家进行的定性研究得到了验证。

    Identifying key patterns of tactics implemented by rival teams, and developing effective responses, lies at the heart of modern football. However, doing so algorithmically remains an open research challenge. To address this unmet need, we propose TacticAI, an AI football tactics assistant developed and evaluated in close collaboration with domain experts from Liverpool FC. We focus on analysing corner kicks, as they offer coaches the most direct opportunities for interventions and improvements. TacticAI incorporates both a predictive and a generative component, allowing the coaches to effectively sample and explore alternative player setups for each corner kick routine and to select those with the highest predicted likelihood of success. We validate TacticAI on a number of relevant benchmark tasks: predicting receivers and shot attempts and recommending player position adjustments. The utility of TacticAI is validated by a qualitative study conducted with football domain experts at Liv
    
[^30]: 探索有限领域知识传递的基本限制

    Towards the Fundamental Limits of Knowledge Transfer over Finite Domains. (arXiv:2310.07838v1 [cs.LG])

    [http://arxiv.org/abs/2310.07838](http://arxiv.org/abs/2310.07838)

    本论文研究了在有限领域中从教师到学生分类器进行知识传递的统计效率，发现特权信息会加速传递，通过使用一种新颖的损失函数达到了知识传递的基本限制。

    

    我们对通过从教师到概率化学生分类器的n个样本进行知识传递的统计效率进行了表征，其中输入空间S和标签A为有限域。我们发现，在三个渐进级别上的特权信息可以加快传递的速度。在第一级别上，只有具有困难标签的样本是已知的，最大似然估计器能够达到最小化速率sqrt(|S||A|/n)。第二级别上，除了已知的困难标签样本外，还有采样标签的教师概率可用，这将收敛速度的下界提高到|S||A|/n。然而，在第二个数据采集协议下，最小化交叉熵损失的朴素适应会导致渐近偏差的学生。我们克服了这个限制，并通过使用一种新颖的经验变体的平方误差逻辑损失来实现了基本限制。第三级别进一步赋予学生软标签。

    We characterize the statistical efficiency of knowledge transfer through $n$ samples from a teacher to a probabilistic student classifier with input space $\mathcal S$ over labels $\mathcal A$. We show that privileged information at three progressive levels accelerates the transfer. At the first level, only samples with hard labels are known, via which the maximum likelihood estimator attains the minimax rate $\sqrt{{|{\mathcal S}||{\mathcal A}|}/{n}}$. The second level has the teacher probabilities of sampled labels available in addition, which turns out to boost the convergence rate lower bound to ${{|{\mathcal S}||{\mathcal A}|}/{n}}$. However, under this second data acquisition protocol, minimizing a naive adaptation of the cross-entropy loss results in an asymptotically biased student. We overcome this limitation and achieve the fundamental limit by using a novel empirical variant of the squared error logit loss. The third level further equips the student with the soft labels (com
    
[^31]: Spectrum-Aware Adjustment: 一种新的去偏方法框架及其在主成分回归中的应用

    Spectrum-Aware Adjustment: A New Debiasing Framework with Applications to Principal Components Regression. (arXiv:2309.07810v1 [math.ST])

    [http://arxiv.org/abs/2309.07810](http://arxiv.org/abs/2309.07810)

    这项研究介绍了一种新的去偏方法框架，用于解决高维线性回归中现有技术对协变量分布的限制问题。研究者们发现，现有方法在处理非高斯分布、异质性设计矩阵和缺乏可靠特征协方差估计时遇到困难。为了解决这些问题，他们提出了一种新的策略，该策略利用缩放的梯度下降步骤进行去偏校正。

    

    我们引入了一个新的去偏方法框架，用于解决高维线性回归中现代去偏技术对协变量分布的约束问题。我们研究了特征数和样本数都很大且相近的普遍情况。在这种情况下，现代去偏技术使用自由度校正来除去正则化估计量的收缩偏差并进行推断。然而，该方法要求观测样本是独立同分布的，协变量遵循均值为零的高斯分布，并且能够获得可靠的特征协方差矩阵估计。当（i）协变量具有非高斯分布、重尾或非对称分布，（ii）设计矩阵的行呈异质性或存在依赖性，以及（iii）缺乏可靠的特征协方差估计时，这种方法就会遇到困难。为了应对这些问题，我们提出了一种新的策略，其中去偏校正是一步缩放的梯度下降步骤（适当缩放）。

    We introduce a new debiasing framework for high-dimensional linear regression that bypasses the restrictions on covariate distributions imposed by modern debiasing technology. We study the prevalent setting where the number of features and samples are both large and comparable. In this context, state-of-the-art debiasing technology uses a degrees-of-freedom correction to remove shrinkage bias of regularized estimators and conduct inference. However, this method requires that the observed samples are i.i.d., the covariates follow a mean zero Gaussian distribution, and reliable covariance matrix estimates for observed features are available. This approach struggles when (i) covariates are non-Gaussian with heavy tails or asymmetric distributions, (ii) rows of the design exhibit heterogeneity or dependencies, and (iii) reliable feature covariance estimates are lacking.  To address these, we develop a new strategy where the debiasing correction is a rescaled gradient descent step (suitably
    
[^32]: 线性赌博机中平衡性能与理论保证的几何感知方法

    Geometry-Aware Approaches for Balancing Performance and Theoretical Guarantees in Linear Bandits. (arXiv:2306.14872v1 [cs.LG])

    [http://arxiv.org/abs/2306.14872](http://arxiv.org/abs/2306.14872)

    本文提出了一种新的数据驱动技术，跟踪不确定度椭球体的几何形状，为线性赌博机算法建立实例相关的频率后悔界，并实现了平衡算法性能与理论保证的效果。

    

    本文受线性赌博机算法表现良好的实证性能与悲观理论后悔界之间的不一致性启发，提出一种新的数据驱动技术，跟踪不确定度椭球体的几何形状，为包括贪心、OFUL和汤普森抽样算法在内的广泛算法类建立实例相关的频率后悔界，在保留基本算法大部分优良特性的同时“校正”基本算法在某些实例中表现差的问题，实现了渐近最优后悔界。我们通过仿真实验验证了该方法的有效性。

    This paper is motivated by recent developments in the linear bandit literature, which have revealed a discrepancy between the promising empirical performance of algorithms such as Thompson sampling and Greedy, when compared to their pessimistic theoretical regret bounds. The challenge arises from the fact that while these algorithms may perform poorly in certain problem instances, they generally excel in typical instances. To address this, we propose a new data-driven technique that tracks the geometry of the uncertainty ellipsoid, enabling us to establish an instance-dependent frequentist regret bound for a broad class of algorithms, including Greedy, OFUL, and Thompson sampling. This result empowers us to identify and ``course-correct" instances in which the base algorithms perform poorly. The course-corrected algorithms achieve the minimax optimal regret of order $\tilde{\mathcal{O}}(d\sqrt{T})$, while retaining most of the desirable properties of the base algorithms. We present sim
    
[^33]: 随机选择纯量分解的核求积方法

    Kernel Quadrature with Randomly Pivoted Cholesky. (arXiv:2306.03955v1 [math.NA])

    [http://arxiv.org/abs/2306.03955](http://arxiv.org/abs/2306.03955)

    本文提出了一种新的使用随机选择纯量分解算法的核求积方法，可以在达到可比的求积误差达到率的同时显著降低计算复杂度，并可以应用于任意核的复杂几何结构。

    

    本文使用随机选择纯量分解的采样算法提出了一种新的重现核希尔伯特空间函数求积规则。所得的计算过程与既有的核求积方法相比，在精度和求解复杂度方面具有更好的性能。理论和实验结果表明，随机选择纯量分解的方法快速且具有可比的求积误差达到率，与基于连续体积采样、稀疏化和重组的更为昂贵的求积方案相匹配。随机选择纯量分解易于适应任意核的复杂几何结构，为核求积开辟了新的潜力。

    This paper presents new quadrature rules for functions in a reproducing kernel Hilbert space using nodes drawn by a sampling algorithm known as randomly pivoted Cholesky. The resulting computational procedure compares favorably to previous kernel quadrature methods, which either achieve low accuracy or require solving a computationally challenging sampling problem. Theoretical and numerical results show that randomly pivoted Cholesky is fast and achieves comparable quadrature error rates to more computationally expensive quadrature schemes based on continuous volume sampling, thinning, and recombination. Randomly pivoted Cholesky is easily adapted to complicated geometries with arbitrary kernels, unlocking new potential for kernel quadrature.
    
[^34]: 当前机器学习需要多少样本才能利用平滑性？

    How many samples are needed to leverage smoothness?. (arXiv:2305.16014v1 [stat.ML])

    [http://arxiv.org/abs/2305.16014](http://arxiv.org/abs/2305.16014)

    本文通过研究泛化误差的新下界，探讨了学习平滑函数时需要的样本数量及其机器学习问题中的挑战。

    

    统计学习的核心原则之一是，目标函数的平滑性可以打破维度灾难。然而，通过泰勒展开学习平滑函数需要足够接近一起的样本来获得高阶导数的有意义估计，这在数据量相对较小的机器学习问题中似乎很困难。本文通过推导广义泛化误差的新的下界，研究了常数和瞬态区域在实践中通常被忽略却发挥了主导作用的问题。

    A core principle in statistical learning is that smoothness of target functions allows to break the curse of dimensionality. However, learning a smooth function through Taylor expansions requires enough samples close to one another to get meaningful estimate of high-order derivatives, which seems hard in machine learning problems where the ratio between number of data and input dimension is relatively small. Should we really hope to break the curse of dimensionality based on Taylor expansion estimation? What happens if Taylor expansions are replaced by Fourier or wavelet expansions? By deriving a new lower bound on the generalization error, this paper investigates the role of constants and transitory regimes which are usually not depicted beyond classical learning theory statements while that play a dominant role in practice.
    
[^35]: 权重具有无界方差的无限宽贝叶斯神经网络后验推断

    Posterior Inference on Infinitely Wide Bayesian Neural Networks under Weights with Unbounded Variance. (arXiv:2305.10664v1 [stat.ML])

    [http://arxiv.org/abs/2305.10664](http://arxiv.org/abs/2305.10664)

    本文提出了一种新的方法进行关于具有无界方差权重的贝叶斯神经网络的后验推断，并表明后验分布集中在具有非标准超参数依赖性的稀疏促进和均值收缩先验周围。

    

    由Neal（1996）的经典而有影响力的作品已知，具有一层隐藏层的贝叶斯神经网络的无限宽度标度极限是一个高斯过程，当网络权重具有有界先验方差时。Neal的结果已扩展到具有多个隐藏层和卷积神经网络的网络，也具有高斯过程标度极限。高斯过程的易处理属性允许直接的后验推断和不确定性量化，相比有限宽度的网络，极大地简化了极限过程的研究。然而，具有无界方差的神经网络权重面临着独特的挑战。在这种情况下，经典的中心极限定理失效，据适当条件下的稳定$\alpha$过程的标度极限的文献较多的是前向模拟，而在这些权重下的后验推断问题仍然是一个未解决的问题。在本文中，我们提出了关于具有无界方差权重的贝叶斯神经网络后验推断的新理论洞察力。具体而言，我们建立了一种新的后验收缩速率结果，并表明后验分布集中在具有非标准超参数依赖性的稀疏促进和均值收缩先验周围。

    From the classical and influential works of Neal (1996), it is known that the infinite width scaling limit of a Bayesian neural network with one hidden layer is a Gaussian process, \emph{when the network weights have bounded prior variance}. Neal's result has been extended to networks with multiple hidden layers and to convolutional neural networks, also with Gaussian process scaling limits. The tractable properties of Gaussian processes then allow straightforward posterior inference and uncertainty quantification, considerably simplifying the study of the limit process compared to a network of finite width. Neural network weights with unbounded variance, however, pose unique challenges. In this case, the classical central limit theorem breaks down and it is well known that the scaling limit is an $\alpha$-stable process under suitable conditions. However, current literature is primarily limited to forward simulations under these processes and the problem of posterior inference under s
    
[^36]: HINT:层次混合网络用于一致概率预测

    HINT: Hierarchical Mixture Networks For Coherent Probabilistic Forecasting. (arXiv:2305.07089v1 [stat.ML])

    [http://arxiv.org/abs/2305.07089](http://arxiv.org/abs/2305.07089)

    HINT是一种用于概率预测的新型模型族，能够有效、准确地进行一致性预测，通过引入Bootstrap方法并为网络加入规范化特征提取和输出规范化来保证其性能，在多个数据集上的预测精度比现有技术更高。

    

    我们提出了一种名为"Hierarchical Mixture Networks"（HINT）的模型族，用于有效而准确的一致性预测。我们通过多元混合并使用复合似然函数进行优化来专门针对该任务进行网络特化，并通过引入Bootstrap方法加以协调。此外，我们在网络中引入了规范化特征提取和输出规范化，以应对时间序列尺度变化。与现有最先进技术相比，我们展示了在五个数据集上的8％ sCRPS增强精度。我们对模型部件进行了消融研究并广泛研究了多元混合的理论性质。 HINT的代码可以在https://github.com/Nixtla/neuralforecast上获得。

    We present the Hierarchical Mixture Networks (HINT), a model family for efficient and accurate coherent forecasting. We specialize the networks on the task via a multivariate mixture optimized with composite likelihood and made coherent via bootstrap reconciliation. Additionally, we robustify the networks to stark time series scale variations, incorporating normalized feature extraction and recomposition of output scales within their architecture. We demonstrate 8% sCRPS improved accuracy across five datasets compared to the existing state-of-the-art. We conduct ablation studies on our model's components and extensively investigate the theoretical properties of the multivariate mixture. HINT's code is available at this https://github.com/Nixtla/neuralforecast.
    
[^37]: 自注意力动态中的聚类现象

    The emergence of clusters in self-attention dynamics. (arXiv:2305.05465v1 [cs.LG])

    [http://arxiv.org/abs/2305.05465](http://arxiv.org/abs/2305.05465)

    本文证实了当Transformer处理一系列token时，出现“领导者”的经验观察，即随着时间趋于无穷大，代表token的粒子会聚集在特定的极限对象附近，这取决于价值矩阵的谱。

    

    将Transformer视为相互作用的粒子系统，当权重不随时间变化时，本文描述了学习表示的几何形状。我们展示了代表token的粒子随着时间趋于无穷大而趋向于特定的极限对象。出现的极限对象类型取决于价值矩阵的谱。此外，在一维情况下，我们证明了自我注意力矩阵收敛于低秩布尔矩阵。这些结果的组合在数学上证实了Vaswani等人的经验观察，即Transformer处理一系列token时会出现“领导者”。

    Viewing Transformers as interacting particle systems, we describe the geometry of learned representations when the weights are not time dependent. We show that particles, representing tokens, tend to cluster toward particular limiting objects as time tends to infinity. The type of limiting object that emerges depends on the spectrum of the value matrix. Additionally, in the one-dimensional case we prove that the self-attention matrix converges to a low-rank Boolean matrix. The combination of these results mathematically confirms the empirical observation made by Vaswani et al. \cite{vaswani2017attention} that \emph{leaders} appear in a sequence of tokens when processed by Transformers.
    
[^38]: 多元概率CRPS学习及其在日前电价预测中的应用

    Multivariate Probabilistic CRPS Learning with an Application to Day-Ahead Electricity Prices. (arXiv:2303.10019v1 [stat.ML])

    [http://arxiv.org/abs/2303.10019](http://arxiv.org/abs/2303.10019)

    本文提出一种新的多元概率CRPS学习方法，应用于日前电价预测中，相比于统一组合在CRPS方面取得了显著改进。

    

    本文提出了一种考虑分位数和协变量依赖关系的多元概率预测的结合方法，并通过平滑过程允许在线学习。通过维数降低和罚函数平滑等两种平滑方法来将标准CRPS学习框架推广到多元维度中。将该方法应用于预测日前电价，相比于统一组合，在CRPS方面取得了显著改进。

    This paper presents a new method for combining (or aggregating or ensembling) multivariate probabilistic forecasts, taking into account dependencies between quantiles and covariates through a smoothing procedure that allows for online learning. Two smoothing methods are discussed: dimensionality reduction using Basis matrices and penalized smoothing. The new online learning algorithm generalizes the standard CRPS learning framework into multivariate dimensions. It is based on Bernstein Online Aggregation (BOA) and yields optimal asymptotic learning properties. We provide an in-depth discussion on possible extensions of the algorithm and several nested cases related to the existing literature on online forecast combination. The methodology is applied to forecasting day-ahead electricity prices, which are 24-dimensional distributional forecasts. The proposed method yields significant improvements over uniform combination in terms of continuous ranked probability score (CRPS). We discuss 
    
[^39]: 在回归随机森林中寻找空间依赖的路径：分类和系统回顾

    A path in regression Random Forest looking for spatial dependence: a taxonomy and a systematic review. (arXiv:2303.04693v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2303.04693](http://arxiv.org/abs/2303.04693)

    在这项工作中，我们提出了一种分类法，根据前处理、中处理和/或后处理的时间点尝试将空间信息纳入回归随机森林中。此外，我们进行了系统回顾并分类最新采用的调整回归随机森林以适应空间相关数据的策略。

    

    随机森林（RF）是一种著名的数据驱动算法，在多个领域中应用广泛，因为它在建模响应变量和预测变量之间的关系时具有很大的灵活性，即使在存在强非线性关系的情况下也适用。在环境应用中，常常出现感兴趣的现象可能存在空间和/或时间依赖性，这在RF的标准版本中没有明确考虑到。在这项工作中，我们提出了一种分类法，根据它们在何时（前处理、中处理和/或后处理）尝试将空间信息纳入回归RF中来对策略进行分类。此外，我们根据《系统回顾和Meta分析首选报告项目》（PRISMA）提供的标准，对最近采用的调整回归RF以适应空间相关数据的策略进行系统回顾和分类。后者是一种可重复的方法，用于收集和处理关于特定主题的不同来源的现有文献。

    Random Forest (RF) is a well-known data-driven algorithm applied in several fields thanks to its flexibility in modeling the relationship between the response variable and the predictors, also in case of strong non-linearities. In environmental applications, it often occurs that the phenomenon of interest may present spatial and/or temporal dependence that is not taken explicitly into account by RF in its standard version. In this work, we propose a taxonomy to classify strategies according to when (Pre-, In- and/or Post-processing) they try to include the spatial information into regression RF. Moreover, we provide a systematic review and classify the most recent strategies adopted to "adjust" regression RF to spatially dependent data, based on the criteria provided by the Preferred Reporting Items for Systematic reviews and Meta-Analysis (PRISMA). The latter consists of a reproducible methodology for collecting and processing existing literature on a specified topic from different so
    
[^40]: 具有带符号排列表示的密集连接的$G$-不变深度神经网络

    Densely Connected $G$-invariant Deep Neural Networks with Signed Permutation Representations. (arXiv:2303.04614v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2303.04614](http://arxiv.org/abs/2303.04614)

    本文提出了一种具有带符号排列表示的密集连接$G$-不变深度神经网络($G$-DNN)架构，通过耦合权重，使得网络的前激活能够通过$G$的带符号排列表示进行变换，从而得到一族更丰富的$G$-不变架构。

    

    我们介绍并研究了对于有限群$G$，具有ReLU激活函数的密集连接$G$-不变深度神经网络($G$-DNN)架构。与文献中其他$G$-不变架构不同，我们所提出的$G$-DNN的前激活能够通过$G$的带符号排列表示(signed perm-reps)进行变换。此外，$G$-DNN的各个层不要求是$G$-等变的；而是通过将输入网络的前激活函数限制为$G$-等变函数的方式，在所有层之间耦合权重。结果是一族更丰富的$G$-不变架构，这在以前从未见过。我们通过权重的重新参数化推导了$G$-DNN的高效实现，并得出了一个架构“可接受”的充分必要条件——即非退化且与更小的架构不相同。我们提供了相关代码。

    We introduce and investigate, for finite groups $G$, $G$-invariant deep neural network ($G$-DNN) architectures with ReLU activation that are densely connected-- i.e., include all possible skip connections. In contrast to other $G$-invariant architectures in the literature, the preactivations of the$G$-DNNs presented here are able to transform by \emph{signed} permutation representations (signed perm-reps) of $G$. Moreover, the individual layers of the $G$-DNNs are not required to be $G$-equivariant; instead, the preactivations are constrained to be $G$-equivariant functions of the network input in a way that couples weights across all layers. The result is a richer family of $G$-invariant architectures never seen previously. We derive an efficient implementation of $G$-DNNs after a reparameterization of weights, as well as necessary and sufficient conditions for an architecture to be ``admissible''-- i.e., nondegenerate and inequivalent to smaller architectures. We include code that al
    
[^41]: 迈向模型基础鲁棒强化学习的最小最大优化

    Towards Minimax Optimality of Model-based Robust Reinforcement Learning. (arXiv:2302.05372v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.05372](http://arxiv.org/abs/2302.05372)

    本文研究了在鲁棒强化学习中，对于仅具有对正常核心的生成模型访问权限时，获得ε-最优策略的样本复杂度。对于sa（s-）矩形不确定集合，已知最佳样本复杂度为ε^2/（H^4 * |S|^2 * |A|）（响应为ε^2/（H^4 * |S|^2 * |A|^2）），对于特定算法和基于总变差（TV）、KL或卡方散度的不确定集合。

    

    我们研究了在只有对正常核心的生成模型访问权限时，获得ε-最优策略的采样复杂度。这个问题在非鲁棒情况下已经得到了广泛研究，并且已知任何应用于经验MDP的规划方法，只需要用ε^2/（H^3 * |S| * |A|）个样本来估计，均可提供ε-最优策略，从而最小最大优化。鲁棒情况下的结果更加少见。对于sa（s-）矩形不确定集合，已知最佳样本复杂度为ε^2/（H^4 * |S|^2 * |A|）（响应为ε^2/（H^4 * |S|^2 * |A|^2）），对于特定算法和基于总变差（TV）、KL或卡方散度的不确定集合。在本文中，我们考虑用Lp球定义的不确定集合（回复到TV情况），并且...

    We study the sample complexity of obtaining an $\epsilon$-optimal policy in \emph{Robust} discounted Markov Decision Processes (RMDPs), given only access to a generative model of the nominal kernel. This problem is widely studied in the non-robust case, and it is known that any planning approach applied to an empirical MDP estimated with $\tilde{\mathcal{O}}(\frac{H^3 \mid S \mid\mid A \mid}{\epsilon^2})$ samples provides an $\epsilon$-optimal policy, which is minimax optimal. Results in the robust case are much more scarce. For $sa$(resp $s$-)rectangular uncertainty sets, the best known sample complexity is $\tilde{\mathcal{O}}(\frac{H^4 \mid S \mid^2\mid A \mid}{\epsilon^2})$ (resp. $\tilde{\mathcal{O}}(\frac{H^4 \mid S \mid^2\mid A \mid^2}{\epsilon^2})$), for specific algorithms and when the uncertainty set is based on the total variation (TV), the KL or the Chi-square divergences. In this paper, we consider uncertainty sets defined with an $L_p$-ball (recovering the TV case), and
    
[^42]: Sketchy: 内存高效的自适应正则化方法与频繁方向的应用

    Sketchy: Memory-efficient Adaptive Regularization with Frequent Directions. (arXiv:2302.03764v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.03764](http://arxiv.org/abs/2302.03764)

    本论文提出了一种内存高效的自适应正则化方法，通过使用频繁方向草稿来降低矩阵预处理器的内存和计算需求。在深度学习任务中，该方法可以在保持性能的同时降低资源的使用。

    

    自适应正则化方法在许多任务中展现了卓越的性能，但在内存和运行时间方面可能受到限制。我们发现在深度学习任务中，Kronecker因子梯度协方差矩阵的谱聚焦在一个变化的小的主特征空间上，这促使我们采用低秩的草稿方法。我们描述了一种通用方法，使用频繁方向（FD）草稿来减少维护矩阵预处理器的内存和计算需求。尽管之前的方法已经探索了在二阶优化中应用FD的方法，但我们提出了一种新颖的分析方法，允许在资源需求和遗憾保证的退化之间进行高效插值: 在在线凸优化（OCO）设置中，我们使用仅$dk$的内存与完整矩阵$d^2$的内存遗憾匹配，直到在底部$d-k$的特征值上添加误差为止。

    Adaptive regularization methods that exploit more than the diagonal entries exhibit state of the art performance for many tasks, but can be prohibitive in terms of memory and running time. We find the spectra of the Kronecker-factored gradient covariance matrix in deep learning (DL) training tasks are concentrated on a small leading eigenspace that changes throughout training, motivating a low-rank sketching approach. We describe a generic method for reducing memory and compute requirements of maintaining a matrix preconditioner using the Frequent Directions (FD) sketch. While previous approaches have explored applying FD for second-order optimization, we present a novel analysis which allows efficient interpolation between resource requirements and the degradation in regret guarantees with rank $k$: in the online convex optimization (OCO) setting over dimension $d$, we match full-matrix $d^2$ memory regret using only $dk$ memory up to additive error in the bottom $d-k$ eigenvalues of 
    
[^43]: 一种行为良好的图神经近似复杂动力学的方法

    A Recipe for Well-behaved Graph Neural Approximations of Complex Dynamics. (arXiv:2301.04900v2 [cond-mat.stat-mech] UPDATED)

    [http://arxiv.org/abs/2301.04900](http://arxiv.org/abs/2301.04900)

    本文介绍了一种行为良好的图神经网络近似复杂动力学的方法，包括必要的偏置和适当的神经网络结构，并提出了评估泛化能力和推断时预测置信度的方法。

    

    数据驱动的常微分方程近似提供了一种有前景的方法来发现动力系统模型，特别是对于缺乏明确原理的复杂系统。本文着重研究了一类由网络邻接矩阵耦合的常微分方程系统描述的复杂系统。许多现实世界中的系统，包括金融、社交和神经系统，属于这类动力学模型。我们提出了使用神经网络近似这种动力系统的关键要素，包括必要的偏置和适当的神经网络结构。强调与静态监督学习的区别，我们提倡在统计学习理论的经典假设之外评估泛化能力。为了在推断时估计预测的置信度，我们引入了一个专用的空模型。通过研究各种复杂网络动力学，我们展示了神经网络的能力。

    Data-driven approximations of ordinary differential equations offer a promising alternative to classical methods in discovering a dynamical system model, particularly in complex systems lacking explicit first principles. This paper focuses on a complex system whose dynamics is described with a system of ordinary differential equations, coupled via a network adjacency matrix. Numerous real-world systems, including financial, social, and neural systems, belong to this class of dynamical models. We propose essential elements for approximating such dynamical systems using neural networks, including necessary biases and an appropriate neural architecture. Emphasizing the differences from static supervised learning, we advocate for evaluating generalization beyond classical assumptions of statistical learning theory. To estimate confidence in prediction during inference time, we introduce a dedicated null model. By studying various complex network dynamics, we demonstrate the neural network'
    
[^44]: 关于随机梯度的被忽视的结构

    On the Overlooked Structure of Stochastic Gradients. (arXiv:2212.02083v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.02083](http://arxiv.org/abs/2212.02083)

    本文对深度学习中随机梯度的结构进行了正式的统计检验，发现逐维梯度通常呈现幂律重尾，而逐次迭代的梯度和随机梯度噪声通常不呈现幂律重尾。

    

    随机梯度与深度神经网络（DNN）的优化和泛化密切相关。一些研究试图通过梯度噪声的重尾性质来解释随机优化在深度学习中的成功，而其他研究则提出了对梯度噪声的重尾假设的理论和实证证据。不幸的是，在深度学习中，用于分析随机梯度结构和重尾的正式统计检验还没有得到充分开发。在本文中，我们主要做出两个贡献。首先，我们对随机梯度和梯度噪声在参数和迭代中的分布进行了正式的统计检验。我们的统计检验发现，逐维梯度通常表现出幂律重尾，而逐次迭代的梯度和由小批量训练引起的随机梯度噪声通常不表现出幂律重尾。其次，我们进一步发现协方差特性。

    Stochastic gradients closely relate to both optimization and generalization of deep neural networks (DNNs). Some works attempted to explain the success of stochastic optimization for deep learning by the arguably heavy-tail properties of gradient noise, while other works presented theoretical and empirical evidence against the heavy-tail hypothesis on gradient noise. Unfortunately, formal statistical tests for analyzing the structure and heavy tails of stochastic gradients in deep learning are still under-explored. In this paper, we mainly make two contributions. First, we conduct formal statistical tests on the distribution of stochastic gradients and gradient noise across both parameters and iterations. Our statistical tests reveal that dimension-wise gradients usually exhibit power-law heavy tails, while iteration-wise gradients and stochastic gradient noise caused by minibatch training usually do not exhibit power-law heavy tails. Second, we further discover that the covariance spe
    
[^45]: 对于协变量偏移泛化的基于独立性驱动的重要性加权算法的理论分析

    A Theoretical Analysis on Independence-driven Importance Weighting for Covariate-shift Generalization. (arXiv:2111.02355v4 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2111.02355](http://arxiv.org/abs/2111.02355)

    本文通过理论分析，将独立性驱动的重要性加权算法解释为特征选择过程，并证明了在协变量偏移泛化中的有效性。

    

    协变量偏移泛化是分布之外（OOD）泛化中的典型情况，要求在未知的测试分布上表现良好，该分布与可访问的训练分布以协变量转移的形式有所不同。最近，稳定学习文献中的独立性驱动的重要性加权算法在处理包括回归算法和深度神经网络在内的多个学习模型上显示出了经验有效性，但它们的理论分析尚缺失。本文通过将它们解释为特征选择过程，从理论上证明了这些算法的有效性。我们首先指定了一组变量，称为最小稳定变量集，该集合是处理协变量偏移泛化的常见损失函数（如均方损失和二元交叉熵损失）的最小最优变量集。随后，我们证明在理想条件下，在这些算法下，独立性驱动的重要性加权算法可以实现这个最小稳定变量集的有效选择。

    Covariate-shift generalization, a typical case in out-of-distribution (OOD) generalization, requires a good performance on the unknown test distribution, which varies from the accessible training distribution in the form of covariate shift. Recently, independence-driven importance weighting algorithms in stable learning literature have shown empirical effectiveness to deal with covariate-shift generalization on several learning models, including regression algorithms and deep neural networks, while their theoretical analyses are missing. In this paper, we theoretically prove the effectiveness of such algorithms by explaining them as feature selection processes. We first specify a set of variables, named minimal stable variable set, that is the minimal and optimal set of variables to deal with covariate-shift generalization for common loss functions, such as the mean squared loss and binary cross-entropy loss. Afterward, we prove that under ideal conditions, independence-driven importan
    
[^46]: 在具有许多类别的上下文推断中通过离线神谕进行最优模型选择

    Optimal Model Selection in Contextual Bandits with Many Classes via Offline Oracles. (arXiv:2106.06483v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2106.06483](http://arxiv.org/abs/2106.06483)

    本论文研究了在随机上下文推断设置中，针对累计遗憾最小化的最优模型选择问题。通过引入渐增类别复杂性和递减边际收益条件，我们提出了一种基于新颖误配测试的算法，并展示了模型选择在奖励估计中的优势。

    

    在监督学习中，模型选择提供了一种无成本的保证，就好像最优平衡偏差和方差的模型是先验已知的一样。我们研究了在随机上下文推断设置中实现类似保证的可行性。最近的研究 [Marinov and Zimmert, 2021] 鉴别出没有算法能够保证无成本的遗憾界限的情况。然而，我们发现在渐增类别复杂性和随着类别复杂性增加最佳策略价值边际收益递减的温和条件下，无成本模型选择是可行的。我们的算法基于一种新颖的误配测试，我们的分析展示了模型选择在奖励估计中的优势。与先前关于上下文推断中模型选择的工作不同，我们的算法在收集更多数据时会仔细地适应逐渐演变的偏差-方差权衡。特别地，我们的算法和分析超越了适应时间复杂性的范畴。

    Model selection in supervised learning provides costless guarantees as if the model that best balances bias and variance was known a priori. We study the feasibility of similar guarantees for cumulative regret minimization in the stochastic contextual bandit setting. Recent work [Marinov and Zimmert, 2021] identifies instances where no algorithm can guarantee costless regret bounds. Nevertheless, we identify benign conditions where costless model selection is feasible: gradually increasing class complexity, and diminishing marginal returns for best-in-class policy value with increasing class complexity. Our algorithm is based on a novel misspecification test, and our analysis demonstrates the benefits of using model selection for reward estimation. Unlike prior work on model selection in contextual bandits, our algorithm carefully adapts to the evolving bias-variance trade-off as more data is collected. In particular, our algorithm and analysis go beyond adapting to the complexity of t
    
[^47]: 用复合传输散度进行高斯混合简化

    Gaussian Mixture Reduction with Composite Transportation Divergence. (arXiv:2002.08410v4 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2002.08410](http://arxiv.org/abs/2002.08410)

    本文提出了一种基于复合传输散度的高斯混合简化方法，用于解决高斯混合在递归更新中阶数指数增加的推断问题。

    

    高斯混合在密度估计、信念传播和贝叶斯滤波等各种应用中被广泛用于逼近密度函数。这些应用通常利用高斯混合作为递归更新的初始近似。这些递归过程中的一个关键挑战源于混合阶数的指数增加，导致难以求解的推断问题。为了克服这个困难，可以使用高斯混合简化（GMR）将高阶高斯混合近似为低阶混合。尽管现有的基于聚类的方法在性能和计算效率上表现良好，但它们的收敛性质和最优目标仍然未知。在本文中，我们提出了一种基于复合传输散度的新型优化GMR方法。我们开发了一个主元最小化算法来计算简化的混合，并在g中建立了其理论收敛性。

    Gaussian mixtures are widely used for approximating density functions in various applications such as density estimation, belief propagation, and Bayesian filtering. These applications often utilize Gaussian mixtures as initial approximations that are updated recursively. A key challenge in these recursive processes stems from the exponential increase in the mixture's order, resulting in intractable inference. To overcome the difficulty, the Gaussian mixture reduction (GMR), which approximates a high order Gaussian mixture by one with a lower order, can be used. Although existing clustering-based methods are known for their satisfactory performance and computational efficiency, their convergence properties and optimal targets remain unknown. In this paper, we propose a novel optimization-based GMR method based on composite transportation divergence (CTD). We develop a majorization-minimization algorithm for computing the reduced mixture and establish its theoretical convergence under g
    
[^48]: 通过大规模事件嵌入和循环网络提高原生广告的CTR预测

    Improving Native Ads CTR Prediction by Large Scale Event Embedding and Recurrent Networks. (arXiv:1804.09133v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/1804.09133](http://arxiv.org/abs/1804.09133)

    本文通过大规模事件嵌入和循环网络，提出了一种改进的CTR预测方法，在原生广告中取得了显著优势。

    

    点击率（CTR）预测对于原生广告非常重要，但由于没有直接的查询意图，因此很难。本文提出了一种大规模事件嵌入方案，通过对用户连续事件进行弱监督训练的孪生网络来编码每个用户浏览事件。CTR预测问题被建模为一个监督循环神经网络，自然地将用户历史建模为事件序列。我们提出的循环模型利用预训练的事件嵌入向量和注意层对用户历史进行建模。实验结果表明，我们的模型明显优于基线模型和一些变体。

    Click through rate (CTR) prediction is very important for Native advertisement but also hard as there is no direct query intent. In this paper we propose a large-scale event embedding scheme to encode the each user browsing event by training a Siamese network with weak supervision on the users' consecutive events. The CTR prediction problem is modeled as a supervised recurrent neural network, which naturally model the user history as a sequence of events. Our proposed recurrent models utilizing pretrained event embedding vectors and an attention layer to model the user history. Our experiments demonstrate that our model significantly outperforms the baseline and some variants.
    

