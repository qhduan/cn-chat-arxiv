# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Beyond IID weights: sparse and low-rank deep Neural Networks are also Gaussian Processes.](http://arxiv.org/abs/2310.16597) | 本文扩展了之前的研究，将证明的范围从独立同分布权重扩展到了更大的权重分布类别(PSEUDO-IID)，包括低秩和稀疏设置。作者发现使用PSEUDO-IID分布初始化的全连接和卷积网络在方差上都是等效的。这些结果可以帮助我们识别更广泛的神经网络的边界混沌状态，并进行性能调优。 |
| [^2] | [Model-adapted Fourier sampling for generative compressed sensing.](http://arxiv.org/abs/2310.04984) | 这篇论文提出了一种模型适应的采样策略，用于生成式压缩感知中的信号恢复，通过优化采样分布和新的理论恢复保证技术，能够显著减少所需测量的数量。 |
| [^3] | [Information Geometry of Wasserstein Statistics on Shapes and Affine Deformations.](http://arxiv.org/abs/2307.12508) | 在这篇论文中，我们研究了Wasserstein统计在仿射变形统计模型中的信息几何特征，比较了信息几何和Wasserstein几何的估计器的优缺点，并发现Wasserstein估计量在椭圆对称仿射变形模型中是矩估计量，在波形为高斯分布时与信息几何估计量重合。 |
| [^4] | [Tracking Most Significant Shifts in Nonparametric Contextual Bandits.](http://arxiv.org/abs/2307.05341) | 该论文研究了非参数情境赌博中的最显著变化，提出了一种只计算显著变化的方法，来解决局部性问题。 |
| [^5] | [Online Learning with Set-Valued Feedback.](http://arxiv.org/abs/2306.06247) | 本文研究了一种在线多类分类的变体，其中使用集合型反馈。通过引入新的组合维度，该论文表明确定性和随机性的在线可学习性在实现设置下不等价，并将在线多标签排名和在线多标签分类等实际学习设置作为其特定实例。 |
| [^6] | [Geometric Algebra Transformers.](http://arxiv.org/abs/2305.18415) | 本文介绍了一种通用架构几何代数变换器（GATr），用于解决几何数据问题。GATr使用投影几何代数表示输入输出和状态，具有可缩放性、表达性、多功能性。在n体建模和机器人规划的实验中，GATr相对于非几何基线表现出强大的改进。 |
| [^7] | [On the impact of activation and normalization in obtaining isometric embeddings at initialization.](http://arxiv.org/abs/2305.18399) | 本论文研究了深度神经网络中的 Gram 矩阵结构，证明了激活函数和层规范化结合使用可以在初始化时偏向指数级深度等距，从而弥补了现有理论的空白。 |
| [^8] | [The minimax risk in testing the histogram of discrete distributions for uniformity under missing ball alternatives.](http://arxiv.org/abs/2305.18111) | 研究了离散分布样本对于类别间的均匀分布拟合问题下的极小极大风险，在缺少球形替代方案的情况下进行了讨论，通过离散直方图进行检验，获得了一种具有精确刻画的检验方法，并在实证研究中表现出了显著性。 |
| [^9] | [Let the Flows Tell: Solving Graph Combinatorial Optimization Problems with GFlowNets.](http://arxiv.org/abs/2305.17010) | 本文提出了一种名为GFlowNets的机器，可以有效地解决组合优化问题，同时在训练方面进行了优化，结果表明其可以高效地找到高质量的解决方案。 |
| [^10] | [Outage Performance and Novel Loss Function for an ML-Assisted Resource Allocation: An Exact Analytical Framework.](http://arxiv.org/abs/2305.09739) | 本文关注在无法获得未来信道状态信息时，利用机器学习解决不同资源分配对应的失误概率问题。提出了一种新型理论最优、可解释的损失函数，经过模拟验证其在失误概率、学习速度和收敛等方面的表现优于一些常见的损失函数。 |
| [^11] | [Reward Teaching for Federated Multi-armed Bandits.](http://arxiv.org/abs/2305.02441) | 本论文提出了一种基于奖励教学思想的联邦多臂老虎机设计，通过隐式本地奖励调整来指导客户端朝着全局最优性，队服务端提出了老虎机学习和目标教学任务进行了优化。 |
| [^12] | [Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC.](http://arxiv.org/abs/2302.11552) | 该论文提出了一种基于能量扩散模型和MCMC的组合生成方法，旨在解决现有技术在组合生成中的失败问题，并提出了新的成功的解决方案。 |
| [^13] | [Uniform-in-time propagation of chaos for mean field Langevin dynamics.](http://arxiv.org/abs/2212.03050) | 研究了平均场 Langevin 动力学，证明了边缘分布 $L^p$ 收敛性和混沌现象的均匀时间传播。 |
| [^14] | [PopArt: Efficient Sparse Regression and Experimental Design for Optimal Sparse Linear Bandits.](http://arxiv.org/abs/2210.15345) | 本文提出了一种称为PopArt的高效稀疏线性估计方法，相比于Lasso，在许多问题中具有更紧的$\ell_1$恢复保证，并基于此推导出稀疏线性摇臂算法，具有改进的遗憾上界。同时，我们证明了在数据稀缺情况下稀疏线性摇臂的匹配下界。 |
| [^15] | [Likelihood-Free Frequentist Inference: Confidence Sets with Correct Conditional Coverage.](http://arxiv.org/abs/2107.03920) | 本文提出了无似然假设下的频率学派推断（LF2I）框架，通过结合经典统计和现代机器学习，实现了构建具有正确条件覆盖的置信区间的实用程序和诊断方法，在包括宇宙学参数推断在内的多个例子中都实现了覆盖性质得到大幅改善。 |

# 详细

[^1]: 超越独立同分布权重：稀疏和低秩深度神经网络也是高斯过程

    Beyond IID weights: sparse and low-rank deep Neural Networks are also Gaussian Processes. (arXiv:2310.16597v1 [stat.ML])

    [http://arxiv.org/abs/2310.16597](http://arxiv.org/abs/2310.16597)

    本文扩展了之前的研究，将证明的范围从独立同分布权重扩展到了更大的权重分布类别(PSEUDO-IID)，包括低秩和稀疏设置。作者发现使用PSEUDO-IID分布初始化的全连接和卷积网络在方差上都是等效的。这些结果可以帮助我们识别更广泛的神经网络的边界混沌状态，并进行性能调优。

    

    无限宽神经网络已经被证明是一个有用且可管理的数学模型，使得我们能够理解深度学习中出现的许多现象。其中一个例子是随机深层网络收敛到高斯过程，从而能够对激活函数和网络权重选择对训练动态的影响进行严格分析。在本文中，我们将Matthews等人(2018)的开创性证明扩展到更大的初始权重分布类别(我们称之为PSEUDO-IID)，其中包括独立同分布和正交权重的已有情况，以及因其计算加速优势而受到赞誉的新兴低秩和结构稀疏设置。我们证明，使用PSEUDO-IID分布初始化的全连接和卷积网络在方差上都是等效的。利用我们的结果，可以识别更广泛的神经网络的边界混沌状态，并调整它们的临界性，以增强训练性能。

    The infinitely wide neural network has been proven a useful and manageable mathematical model that enables the understanding of many phenomena appearing in deep learning. One example is the convergence of random deep networks to Gaussian processes that allows a rigorous analysis of the way the choice of activation function and network weights impacts the training dynamics. In this paper, we extend the seminal proof of Matthews et al. (2018) to a larger class of initial weight distributions (which we call PSEUDO-IID), including the established cases of IID and orthogonal weights, as well as the emerging low-rank and structured sparse settings celebrated for their computational speed-up benefits. We show that fully-connected and convolutional networks initialized with PSEUDO-IID distributions are all effectively equivalent up to their variance. Using our results, one can identify the Edge-of-Chaos for a broader class of neural networks and tune them at criticality in order to enhance the
    
[^2]: 模型适应的傅立叶采样用于生成式压缩感知

    Model-adapted Fourier sampling for generative compressed sensing. (arXiv:2310.04984v1 [cs.IT])

    [http://arxiv.org/abs/2310.04984](http://arxiv.org/abs/2310.04984)

    这篇论文提出了一种模型适应的采样策略，用于生成式压缩感知中的信号恢复，通过优化采样分布和新的理论恢复保证技术，能够显著减少所需测量的数量。

    

    我们研究了当测量矩阵是从一个酉矩阵中随机子采样得到时的生成式压缩感知问题（离散傅立叶变换是重要的特殊情况）。最近的研究表明，当每个傅立叶向量与神经网络的取值范围对齐时，只需要$O(kdn\|\boldsymbol{\alpha}\|_{\infty}^{2})$个均匀随机傅立叶测量就足以恢复输出信号。我们构建了一个模型适应的采样策略，其样本复杂度得到了改进，只需要$O(kd\|\boldsymbol{\alpha}\|_{2}^{2})$个测量。这是通过以下步骤实现的：（1）我们开发了适用于非均匀随机采样分布  的新的理论恢复保证，然后（2）优化采样分布以最小化这些保证所需的测量数量。这种技术发展提供了适用的样本复杂度。

    We study generative compressed sensing when the measurement matrix is randomly subsampled from a unitary matrix (with the DFT as an important special case). It was recently shown that $\textit{O}(kdn\| \boldsymbol{\alpha}\|_{\infty}^{2})$ uniformly random Fourier measurements are sufficient to recover signals in the range of a neural network $G:\mathbb{R}^k \to \mathbb{R}^n$ of depth $d$, where each component of the so-called local coherence vector $\boldsymbol{\alpha}$ quantifies the alignment of a corresponding Fourier vector with the range of $G$. We construct a model-adapted sampling strategy with an improved sample complexity of $\textit{O}(kd\| \boldsymbol{\alpha}\|_{2}^{2})$ measurements. This is enabled by: (1) new theoretical recovery guarantees that we develop for nonuniformly random sampling distributions and then (2) optimizing the sampling distribution to minimize the number of measurements needed for these guarantees. This development offers a sample complexity applicable
    
[^3]: 形状和仿射变形的Wasserstein统计的信息几何

    Information Geometry of Wasserstein Statistics on Shapes and Affine Deformations. (arXiv:2307.12508v1 [math.ST])

    [http://arxiv.org/abs/2307.12508](http://arxiv.org/abs/2307.12508)

    在这篇论文中，我们研究了Wasserstein统计在仿射变形统计模型中的信息几何特征，比较了信息几何和Wasserstein几何的估计器的优缺点，并发现Wasserstein估计量在椭圆对称仿射变形模型中是矩估计量，在波形为高斯分布时与信息几何估计量重合。

    

    信息几何和Wasserstein几何是介绍概率分布流形中的两个主要结构，它们捕捉了不同的特征。我们在仿射变形统计模型的Li和Zhao（2023）框架中研究了Wasserstein几何的特征，它是位置-尺度模型的多维泛化。我们比较了基于信息几何和Wasserstein几何的估计器的优点和缺点。在Wasserstein几何中，概率分布的形状和仿射变形是分离的，表明在对波形扰动具有鲁棒性的同时，会损失Fisher效率。我们证明了在椭圆对称仿射变形模型的情况下Wasserstein估计量是矩估计量。它与信息几何估计量（最大似然估计量）仅在波形为高斯分布时重合。Wasserstein效率的作用是...

    Information geometry and Wasserstein geometry are two main structures introduced in a manifold of probability distributions, and they capture its different characteristics. We study characteristics of Wasserstein geometry in the framework of Li and Zhao (2023) for the affine deformation statistical model, which is a multi-dimensional generalization of the location-scale model. We compare merits and demerits of estimators based on information geometry and Wasserstein geometry. The shape of a probability distribution and its affine deformation are separated in the Wasserstein geometry, showing its robustness against the waveform perturbation in exchange for the loss in Fisher efficiency. We show that the Wasserstein estimator is the moment estimator in the case of the elliptically symmetric affine deformation model. It coincides with the information-geometrical estimator (maximum-likelihood estimator) when and only when the waveform is Gaussian. The role of the Wasserstein efficiency is 
    
[^4]: 跟踪非参数情境赌博中最显著变化

    Tracking Most Significant Shifts in Nonparametric Contextual Bandits. (arXiv:2307.05341v1 [stat.ML])

    [http://arxiv.org/abs/2307.05341](http://arxiv.org/abs/2307.05341)

    该论文研究了非参数情境赌博中的最显著变化，提出了一种只计算显著变化的方法，来解决局部性问题。

    

    我们研究了非参数情境赌博，其中Lipschitz均值奖励函数可能随时间变化。我们首先在这个较少被理解的情境下建立了动态遗憾率的极小极大值，这些值与变化数量L和总变差V有关，两者都可以捕捉到上下文空间的所有分布变化，并且证明了目前的方法在这个情境下是次优的。接下来，我们探讨了这种情境下的适应性问题，即在不知道L或V的情况下实现极小极大值。非常重要的是，我们认为，在给定的上下文X_t处，赌博问题在上下文空间其他部分中的奖励变化不应该产生影响。因此，我们提出了一种变化的概念，我们称之为经验显著变化，更好地考虑了局部性，因此比L和V计数更少。此外，类似于最近在非平稳多臂赌博机中的工作（Suk和Kpotufe，2022），经验显著变化只计算显著变化。

    We study nonparametric contextual bandits where Lipschitz mean reward functions may change over time. We first establish the minimax dynamic regret rate in this less understood setting in terms of number of changes $L$ and total-variation $V$, both capturing all changes in distribution over context space, and argue that state-of-the-art procedures are suboptimal in this setting.  Next, we tend to the question of an adaptivity for this setting, i.e. achieving the minimax rate without knowledge of $L$ or $V$. Quite importantly, we posit that the bandit problem, viewed locally at a given context $X_t$, should not be affected by reward changes in other parts of context space $\cal X$. We therefore propose a notion of change, which we term experienced significant shifts, that better accounts for locality, and thus counts considerably less changes than $L$ and $V$. Furthermore, similar to recent work on non-stationary MAB (Suk & Kpotufe, 2022), experienced significant shifts only count the m
    
[^5]: 使用集合型反馈的在线学习

    Online Learning with Set-Valued Feedback. (arXiv:2306.06247v1 [cs.LG])

    [http://arxiv.org/abs/2306.06247](http://arxiv.org/abs/2306.06247)

    本文研究了一种在线多类分类的变体，其中使用集合型反馈。通过引入新的组合维度，该论文表明确定性和随机性的在线可学习性在实现设置下不等价，并将在线多标签排名和在线多标签分类等实际学习设置作为其特定实例。

    

    本文研究了在线多类分类的一种变体，其中学习器预测单个标签，但接收到一个标签的集合作为反馈。在该模型中，如果学习器没有输出包含在反馈集合中的标签，则会受到惩罚。我们表明，与具有单标签反馈的在线多类学习不同，在实现设置中使用集合型反馈时，确定性和随机化的在线可学习性\textit{不等价}。因此，我们提供了两个新的组合维度，分别命名为集合小石和度量破裂维度，严格描述了确定性和随机化的在线可学习性。此外，我们表明度量破裂维度在悟性设置下严格描述在线可学习性。最后，我们证明了在线多标签排名和在线多标签分类等实际学习设置是我们通用在线学习框架的具体实例。

    We study a variant of online multiclass classification where the learner predicts a single label but receives a \textit{set of labels} as feedback. In this model, the learner is penalized for not outputting a label contained in the revealed set. We show that unlike online multiclass learning with single-label feedback, deterministic and randomized online learnability are \textit{not equivalent} even in the realizable setting with set-valued feedback. Accordingly, we give two new combinatorial dimensions, named the Set Littlestone and Measure Shattering dimension, that tightly characterize deterministic and randomized online learnability respectively in the realizable setting. In addition, we show that the Measure Shattering dimension tightly characterizes online learnability in the agnostic setting. Finally, we show that practical learning settings like online multilabel ranking and online multilabel classification are specific instances of our general online learning framework.
    
[^6]: 几何代数变换器

    Geometric Algebra Transformers. (arXiv:2305.18415v1 [cs.LG])

    [http://arxiv.org/abs/2305.18415](http://arxiv.org/abs/2305.18415)

    本文介绍了一种通用架构几何代数变换器（GATr），用于解决几何数据问题。GATr使用投影几何代数表示输入输出和状态，具有可缩放性、表达性、多功能性。在n体建模和机器人规划的实验中，GATr相对于非几何基线表现出强大的改进。

    

    几何数据问题涉及计算机视觉、机器人、化学和物理领域。这些数据可以采用许多形式，例如点、方向向量、平面或变换，但迄今为止还没有一种单一的架构，可以应用于如此多种几何类型, 同时尊重它们的对称性。在本文中，我们介绍了几何代数变换器（GATr），一种用于几何数据的通用架构。GATr使用投影几何代数来表示输入、输出和隐藏状态，其提供常见几何对象的高效16维向量空间表示以及作用于它们的运算符。GATr是相对于E(3)（3D欧几里得空间的对称群）等变的。作为变换器，GATr可扩展、表达丰富且多功能。在n体建模和机器人规划的实验中，GATr相对于非几何基线均表现出强大的改进。

    Problems involving geometric data arise in a variety of fields, including computer vision, robotics, chemistry, and physics. Such data can take numerous forms, such as points, direction vectors, planes, or transformations, but to date there is no single architecture that can be applied to such a wide variety of geometric types while respecting their symmetries. In this paper we introduce the Geometric Algebra Transformer (GATr), a general-purpose architecture for geometric data. GATr represents inputs, outputs, and hidden states in the projective geometric algebra, which offers an efficient 16-dimensional vector space representation of common geometric objects as well as operators acting on them. GATr is equivariant with respect to E(3), the symmetry group of 3D Euclidean space. As a transformer, GATr is scalable, expressive, and versatile. In experiments with n-body modeling and robotic planning, GATr shows strong improvements over non-geometric baselines.
    
[^7]: 关于激活函数和规范化对初始化等距嵌入的影响

    On the impact of activation and normalization in obtaining isometric embeddings at initialization. (arXiv:2305.18399v1 [cs.LG])

    [http://arxiv.org/abs/2305.18399](http://arxiv.org/abs/2305.18399)

    本论文研究了深度神经网络中的 Gram 矩阵结构，证明了激活函数和层规范化结合使用可以在初始化时偏向指数级深度等距，从而弥补了现有理论的空白。

    

    本文探讨了深度神经网络中倒数第二个 Gram 矩阵的结构，该矩阵包含与一批输入对应的输出之间的成对内积。在几种架构中，观察到在初始化时该 Gram 矩阵会随着深度变得退化，从而严重减缓训练速度。规范化层如批处理规范化或层规范化，在防止秩崩溃问题方面起着关键作用。然而现有的理论结果无法全面覆盖广泛用于 transformer 中的层规范化和有限深度下规范化的量化偏差。为了解决这个问题，我们证明了在初始化时，结合激活函数层使用的层规范化可以使多层感知机的 Gram 矩阵偏向指数级深度等距，并使用激活函数的 Hermite 展开来量化这个速度，从而填补了现有理论的空白。

    In this paper, we explore the structure of the penultimate Gram matrix in deep neural networks, which contains the pairwise inner products of outputs corresponding to a batch of inputs. In several architectures it has been observed that this Gram matrix becomes degenerate with depth at initialization, which dramatically slows training. Normalization layers, such as batch or layer normalization, play a pivotal role in preventing the rank collapse issue. Despite promising advances, the existing theoretical results (i) do not extend to layer normalization, which is widely used in transformers, (ii) can not characterize the bias of normalization quantitatively at finite depth.  To bridge this gap, we provide a proof that layer normalization, in conjunction with activation layers, biases the Gram matrix of a multilayer perceptron towards isometry at an exponential rate with depth at initialization. We quantify this rate using the Hermite expansion of the activation function, highlighting th
    
[^8]: 在缺少球形替代方案下测试离散分布直方图均匀性的极小极大风险

    The minimax risk in testing the histogram of discrete distributions for uniformity under missing ball alternatives. (arXiv:2305.18111v1 [math.ST])

    [http://arxiv.org/abs/2305.18111](http://arxiv.org/abs/2305.18111)

    研究了离散分布样本对于类别间的均匀分布拟合问题下的极小极大风险，在缺少球形替代方案的情况下进行了讨论，通过离散直方图进行检验，获得了一种具有精确刻画的检验方法，并在实证研究中表现出了显著性。

    

    我们考虑测试一个来自许多类别的离散样本对于类别间的均匀分布拟合的问题。作为另一类替代假设，我们考虑去除半径为$\epsilon$的$\ell_p$球形替代方案，其中$p\leq 2$。我们给出了基于直方图（缺失类别、单例、碰撞的数量）的检验在样本数和维数趋向无穷大，$\epsilon\to0$时，渐进极小极大风险的一个精确刻画。例如，当$p=1$且期望样本数$n$与类别数$N$的比值很小（也称为“次线性”区域）时，渐进极小极大风险$R^*_\epsilon$趋近于$2\bar{\Phi}\left(n\epsilon^2/\sqrt{8N}\right)$，其中$\bar{\Phi}(x)$是正态残存函数。在一系列问题参数范围内的实证研究表明，这个估计在有限样本中很精确，并且我们的检验显著。

    We consider the problem of testing the fit of a discrete sample of items from many categories to the uniform distribution over the categories. As a class of alternative hypotheses, we consider the removal of an $\ell_p$ ball of radius $\epsilon$ around the uniform rate sequence for $p \leq 2$. We deliver a sharp characterization of the asymptotic minimax risk when $\epsilon \to 0$ as the number of samples and number of dimensions go to infinity, for testing based on the occurrences' histogram (number of absent categories, singletons, collisions, ...). For example, for $p=1$ and in the limit of a small expected number of samples $n$ compared to the number of categories $N$ (aka "sub-linear" regime), the minimax risk $R^*_\epsilon$ asymptotes to $2 \bar{\Phi}\left(n \epsilon^2/\sqrt{8N}\right) $, with $\bar{\Phi}(x)$ the normal survival function. Empirical studies over a range of problem parameters show that this estimate is accurate in finite samples, and that our test is significantly 
    
[^9]: 利用GFlowNets解决图形组合优化问题

    Let the Flows Tell: Solving Graph Combinatorial Optimization Problems with GFlowNets. (arXiv:2305.17010v1 [cs.LG])

    [http://arxiv.org/abs/2305.17010](http://arxiv.org/abs/2305.17010)

    本文提出了一种名为GFlowNets的机器，可以有效地解决组合优化问题，同时在训练方面进行了优化，结果表明其可以高效地找到高质量的解决方案。

    

    组合优化问题通常是NP难题，因此不适用于精确算法，这使它们成为应用机器学习方法的理想领域。这些问题中高度结构化的限制可能会直接阻碍优化或采样解决方案的空间。另一方面，GFlowNets最近被发现是一种强大的机器，可以顺序地从复合非规范化密度中有效地采样，并具有在CO中分摊此类解决方案搜索过程以及生成不同的解决方案候选项的潜力。在本文中，我们设计了适用于不同组合问题的马尔科夫决策过程（MDP），并提出训练有条件的GFlowNets从解空间中采样的策略。还开发了高效的训练技术来受益于远程信用分配。通过对各种使用合成和实际数据的不同CO任务的广泛实验，我们证明了GFlowNet策略可以有效地找到高质量的解。

    Combinatorial optimization (CO) problems are often NP-hard and thus out of reach for exact algorithms, making them a tempting domain to apply machine learning methods. The highly structured constraints in these problems can hinder either optimization or sampling directly in the solution space. On the other hand, GFlowNets have recently emerged as a powerful machinery to efficiently sample from composite unnormalized densities sequentially and have the potential to amortize such solution-searching processes in CO, as well as generate diverse solution candidates. In this paper, we design Markov decision processes (MDPs) for different combinatorial problems and propose to train conditional GFlowNets to sample from the solution space. Efficient training techniques are also developed to benefit long-range credit assignment. Through extensive experiments on a variety of different CO tasks with synthetic and realistic data, we demonstrate that GFlowNet policies can efficiently find high-quali
    
[^10]: ML辅助资源分配的失误性能和新型损失函数: 一种精确分析框架

    Outage Performance and Novel Loss Function for an ML-Assisted Resource Allocation: An Exact Analytical Framework. (arXiv:2305.09739v1 [eess.SP])

    [http://arxiv.org/abs/2305.09739](http://arxiv.org/abs/2305.09739)

    本文关注在无法获得未来信道状态信息时，利用机器学习解决不同资源分配对应的失误概率问题。提出了一种新型理论最优、可解释的损失函数，经过模拟验证其在失误概率、学习速度和收敛等方面的表现优于一些常见的损失函数。

    

    机器学习是使6G及以上通信成为可能的重要工具。本文致力于将机器学习应用于解决这些系统普遍遇到的失误概率问题。具体来说，我们考虑单用户多资源贪婪分配策略，其中一个ML二分类预测器在选择充足资源时进行辅助。当预测器遇到相信会满意的资源时，它将其分配给用户。我们的主要结果是建立了该系统失误概率的精确和渐进表达式。在此基础上，我们制定了一种理论最优的、可微的损失函数来训练预测器。我们通过大量模拟比较使用此损失函数训练的预测器和使用其他常用损失函数训练的预测器之间的性能差异。我们还探讨了所提出的损失函数对预测器透明度和无线信道结构的影响。我们的数值结果表明，所提出的损失函数在失误概率、端到端学习速度和收敛方面优于现有的基于二元交叉熵的损失。此外，我们的损失函数制定赋予了可解释的预测器训练，能够稳健地处理信道的时变性。

    Machine Learning (ML) is a popular tool that will be pivotal in enabling 6G and beyond communications. This paper focuses on applying ML solutions to address outage probability issues commonly encountered in these systems. In particular, we consider a single-user multi-resource greedy allocation strategy, where an ML binary classification predictor assists in seizing an adequate resource. With no access to future channel state information, this predictor foresees each resource's likely future outage status. When the predictor encounters a resource it believes will be satisfactory, it allocates it to the user. Critically, the goal of the predictor is to ensure that a user avoids an unsatisfactory resource since this is likely to cause an outage. Our main result establishes exact and asymptotic expressions for this system's outage probability. With this, we formulate a theoretically optimal, differentiable loss function to train our predictor. We then compare predictors trained using thi
    
[^11]: 基于奖励教学的联邦多臂老虎机设计

    Reward Teaching for Federated Multi-armed Bandits. (arXiv:2305.02441v1 [stat.ML])

    [http://arxiv.org/abs/2305.02441](http://arxiv.org/abs/2305.02441)

    本论文提出了一种基于奖励教学思想的联邦多臂老虎机设计，通过隐式本地奖励调整来指导客户端朝着全局最优性，队服务端提出了老虎机学习和目标教学任务进行了优化。

    

    目前大部分已有的联邦多臂老虎机（FMAB）设计都基于假设客户端会实现指定的设计来与服务器协作。但实际上，可能无法修改客户端现有的协议。为了应对这一挑战，该工作关注始终最大化其个体累积奖励的客户端，并引入了“奖励教学”的新思想，即通过隐式的本地奖励调整指导客户端朝着全局最优性。在这个框架下，服务器面临两个密切耦合的任务，即老虎机学习和目标教学，它们的结合非常复杂和具有挑战性。首先设计了一个名为 “Teaching-After-Learning（TAL）” 的分阶段方法，分别鼓励和限制客户端的探索。当客户端策略满足一定的温和要求时，建立了TAL的综合性能分析。通过开发新的技术方法来分析TAL的热启动和算法，我们展示了TAL可以比现有的FMAB设计带来显著的改进。

    Most of the existing federated multi-armed bandits (FMAB) designs are based on the presumption that clients will implement the specified design to collaborate with the server. In reality, however, it may not be possible to modify the client's existing protocols. To address this challenge, this work focuses on clients who always maximize their individual cumulative rewards, and introduces a novel idea of "reward teaching", where the server guides the clients towards global optimality through implicit local reward adjustments. Under this framework, the server faces two tightly coupled tasks of bandit learning and target teaching, whose combination is non-trivial and challenging. A phased approach, called Teaching-After-Learning (TAL), is first designed to encourage and discourage clients' explorations separately. General performance analyses of TAL are established when the clients' strategies satisfy certain mild requirements. With novel technical approaches developed to analyze the warm
    
[^12]: 减少、重复利用、回收：基于能量扩散模型和MCMC的组合生成

    Reduce, Reuse, Recycle: Compositional Generation with Energy-Based Diffusion Models and MCMC. (arXiv:2302.11552v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.11552](http://arxiv.org/abs/2302.11552)

    该论文提出了一种基于能量扩散模型和MCMC的组合生成方法，旨在解决现有技术在组合生成中的失败问题，并提出了新的成功的解决方案。

    

    自从扩散模型问世以来，它在许多领域中已经迅速成为生成模型的主要方法。它们可以被解释为学习一系列时变的对数概率密度函数的梯度。这种解释已经激发了基于分类器和无分类器指导的思想成为后续控制扩散模型的方法。在这项工作中，我们建立在这些想法的基础上，利用扩散模型的分数-based解释，探索了用于涉及组合生成和指导的条件、修改和重复使用扩散模型的替代方法。特别是，我们调查了为什么某些类型的组合使用当前技术失败，并介绍了一些解决方案。我们得出结论，采样者(而不是模型)对此失败负有责任，并提出了新的采样器，受MCMC的启发，使组合生成成功。此外，我们提出了一种基于能量的扩散模型参数化方法，它使得逼近目标分布更加容易。

    Since their introduction, diffusion models have quickly become the prevailing approach to generative modeling in many domains. They can be interpreted as learning the gradients of a time-varying sequence of log-probability density functions. This interpretation has motivated classifier-based and classifier-free guidance as methods for post-hoc control of diffusion models. In this work, we build upon these ideas using the score-based interpretation of diffusion models, and explore alternative ways to condition, modify, and reuse diffusion models for tasks involving compositional generation and guidance. In particular, we investigate why certain types of composition fail using current techniques and present a number of solutions. We conclude that the sampler (not the model) is responsible for this failure and propose new samplers, inspired by MCMC, which enable successful compositional generation. Further, we propose an energy-based parameterization of diffusion models which enables the 
    
[^13]: 均匀时间传播混沌的平均场 Langevin 动力学

    Uniform-in-time propagation of chaos for mean field Langevin dynamics. (arXiv:2212.03050v2 [math.PR] UPDATED)

    [http://arxiv.org/abs/2212.03050](http://arxiv.org/abs/2212.03050)

    研究了平均场 Langevin 动力学，证明了边缘分布 $L^p$ 收敛性和混沌现象的均匀时间传播。

    

    本文研究了平均场 Langevin 动力学及其相关粒子系统。通过假设能量函数的凸性，我们得出了边缘分布收敛到平均场动力学唯一不变测度的 $L^p$ 收敛性。此外，我们证明了在 $L^2$ Wasserstein 距离和相对熵两方面，混沌现象的均匀时间传播。

    We study the mean field Langevin dynamics and the associated particle system. By assuming the functional convexity of the energy, we obtain the $L^p$-convergence of the marginal distributions towards the unique invariant measure for the mean field dynamics. Furthermore, we prove the uniform-in-time propagation of chaos in both the $L^2$-Wasserstein metric and relative entropy.
    
[^14]: PopArt: 高效稀疏回归和优化稀疏线性摇臂的实验设计

    PopArt: Efficient Sparse Regression and Experimental Design for Optimal Sparse Linear Bandits. (arXiv:2210.15345v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2210.15345](http://arxiv.org/abs/2210.15345)

    本文提出了一种称为PopArt的高效稀疏线性估计方法，相比于Lasso，在许多问题中具有更紧的$\ell_1$恢复保证，并基于此推导出稀疏线性摇臂算法，具有改进的遗憾上界。同时，我们证明了在数据稀缺情况下稀疏线性摇臂的匹配下界。

    

    在稀疏线性摇臂中，学习代理按顺序选择一个动作并接收奖励反馈，而奖励函数线性依赖于动作的一些坐标的协变量。这在许多实际的顺序决策问题中都有应用。在本文中，我们提出了一种简单而计算高效的稀疏线性估计方法，称为PopArt，与Lasso（Tibshirani, 1996）相比，它在许多问题中具有更紧的$\ell_1$恢复保证。我们的界限自然地激发了一种凸实验设计准则，因此在计算上是高效的解决方法。基于我们的新估计器和设计准则，我们推导出稀疏线性摇臂算法，其在给定动作集的几何性方面相对于现有技术（Hao et al., 2020）具有改进的遗憾上界。最后，我们证明了在数据稀缺情况下稀疏线性摇臂的匹配下界，这填补了上界和下界之间的差距。

    In sparse linear bandits, a learning agent sequentially selects an action and receive reward feedback, and the reward function depends linearly on a few coordinates of the covariates of the actions. This has applications in many real-world sequential decision making problems. In this paper, we propose a simple and computationally efficient sparse linear estimation method called PopArt that enjoys a tighter $\ell_1$ recovery guarantee compared to Lasso (Tibshirani, 1996) in many problems. Our bound naturally motivates an experimental design criterion that is convex and thus computationally efficient to solve. Based on our novel estimator and design criterion, we derive sparse linear bandit algorithms that enjoy improved regret upper bounds upon the state of the art (Hao et al., 2020), especially w.r.t. the geometry of the given action set. Finally, we prove a matching lower bound for sparse linear bandits in the data-poor regime, which closes the gap between upper and lower bounds in pr
    
[^15]: 无似然假设下基于频率学派推断：具有正确条件覆盖的置信区间

    Likelihood-Free Frequentist Inference: Confidence Sets with Correct Conditional Coverage. (arXiv:2107.03920v6 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2107.03920](http://arxiv.org/abs/2107.03920)

    本文提出了无似然假设下的频率学派推断（LF2I）框架，通过结合经典统计和现代机器学习，实现了构建具有正确条件覆盖的置信区间的实用程序和诊断方法，在包括宇宙学参数推断在内的多个例子中都实现了覆盖性质得到大幅改善。

    

    许多科学领域都广泛使用计算机模拟器以隐含复杂系统的似然函数。传统的统计方法并不适用于这些称为无似然假设下推断（LFI）的情况，尤其是在渐近和低维的条件下。虽然新的机器学习方法，如归一化流，已经革新了LFI方法的样本效率和容量，但它们是否能为小样本大小产生具有正确条件覆盖的置信区间，仍然是一个开放问题。本文将经典统计和现代机器学习相结合，提出了（i）具有有限样本保证名义覆盖的内曼区间建设的实用程序，以及（ii）估计整个参数空间的条件覆盖的诊断。我们将我们的框架称为无似然假设下的频率学派推断（LF2I）。我们的框架可以使用定义测试统计量的任何方法，如似然比，因此具有广泛的适用性。我们将我们的方法应用于几个合成和实际的例子，包括宇宙学参数推断，并证明与现有的LFI方法相比，覆盖性质得到了大幅改善。

    Many areas of science make extensive use of computer simulators that implicitly encode likelihood functions of complex systems. Classical statistical methods are poorly suited for these so-called likelihood-free inference (LFI) settings, particularly outside asymptotic and low-dimensional regimes. Although new machine learning methods, such as normalizing flows, have revolutionized the sample efficiency and capacity of LFI methods, it remains an open question whether they produce confidence sets with correct conditional coverage for small sample sizes. This paper unifies classical statistics with modern machine learning to present (i) a practical procedure for the Neyman construction of confidence sets with finite-sample guarantees of nominal coverage, and (ii) diagnostics that estimate conditional coverage over the entire parameter space. We refer to our framework as likelihood-free frequentist inference (LF2I). Any method that defines a test statistic, like the likelihood ratio, can 
    

