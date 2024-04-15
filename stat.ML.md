# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Dynamical Model of Neural Scaling Laws](https://rss.arxiv.org/abs/2402.01092) | 这篇论文提出了一个动力学模型来解释神经缩放定律。通过分析梯度下降训练的随机特征模型，研究发现训练时间和模型大小的缩放具有不同的幂律指数，而计算最优缩放规则要求增加训练步数快于增加模型参数，与实证观察相一致。 |
| [^2] | [Transfer Learning with Reconstruction Loss](https://arxiv.org/abs/2404.00505) | 本文通过引入额外的重建阶段和重建损失，提出了一种具有共享模型参数和特征表示的模型训练方法，建立了共同信息的概念，用于解决相关任务。 |
| [^3] | [TaCo: Targeted Concept Removal in Output Embeddings for NLP via Information Theory and Explainability.](http://arxiv.org/abs/2312.06499) | 本论文提出了一种新颖的方法，通过对NLP模型的嵌入层级进行操作，借鉴了最新的解释性人工智能技术，通过嵌入转换来消除隐含的敏感信息，从而实现模型的公平性。 |
| [^4] | [Neural Likelihood Approximation for Integer Valued Time Series Data.](http://arxiv.org/abs/2310.12544) | 本文构建了整数值时间序列数据的神经似然近似方法，使用因果卷积并行评估整个时间序列的似然，实现了对生态学和流行病学模型进行准确推断并显著加快计算速度。 |
| [^5] | [On the Minimax Regret in Online Ranking with Top-k Feedback.](http://arxiv.org/abs/2309.02425) | 本文研究了在线排名中的最小极大后悔问题与Top-k反馈，并通过提供一个全面的极大后悔率刻画，解决了Chaudhuri和Tewari [2017]所提出的问题。 |
| [^6] | [Integrated Variational Fourier Features for Fast Spatial Modelling with Gaussian Processes.](http://arxiv.org/abs/2308.14142) | 本论文提出了一种综合变分傅里叶特征的方法，可以对广泛的平稳协方差函数进行快速空间建模，相比其他方法具有更高的性能和加速效果。 |
| [^7] | [A State-Space Perspective on Modelling and Inference for Online Skill Rating.](http://arxiv.org/abs/2308.02414) | 本文提供了对竞技体育技能评级主要方法的全面回顾，并提出了采用状态空间模型视角的建议。通过使用状态空间模型视角，玩家的技能可以表示为随时间变化的变量，而比赛结果则是唯一的观测量。该视角有助于解耦建模和推理，并促进通用推理工具的发展。在构建技能评级的状态空间模型方面，本文探讨了基本步骤，同时还讨论了滤波、平滑和参数估计等推理阶段。在面对高维场景中的计算挑战时，本文强调了所采用的近似和简化方法。该文提供了对记录的流行方法的简明总结。 |
| [^8] | [Properties of Discrete Sliced Wasserstein Losses.](http://arxiv.org/abs/2307.10352) | 本文研究了离散切割Wasserstein损失的性质，并探讨了其正则性和优化性质以及通过蒙特卡洛近似的方法。 |
| [^9] | [Kernel-Based Testing for Single-Cell Differential Analysis.](http://arxiv.org/abs/2307.08509) | 本论文提出了一种基于核方法的单细胞差异分析测试框架，可以非线性比较复杂的细胞间分子特征分布。通过利用核嵌入的变异性，我们的方法能够揭示细胞群体中隐蔽的异质性。我们展示了核测试如何克服单细胞差异分析方法的局限性，并应用于研究分化逆转的过程。 |
| [^10] | [Reinforcement Learning with Non-Cumulative Objective.](http://arxiv.org/abs/2307.04957) | 本文研究了最优控制和强化学习中非累积目标的挑战，并提出了修改现有算法的方法来优化这些目标。研究结果表明，在贝尔曼最优性方程中使用广义运算可以更好地处理非累积目标。 |
| [^11] | [Topological Interpretability for Deep-Learning.](http://arxiv.org/abs/2305.08642) | 本文提出了一种在临床和非临床文本上训练的深度学习分类模型中使用拓扑和几何数据分析技术推断主要特征的方法。 |
| [^12] | [Kullback-Leibler Maillard Sampling for Multi-armed Bandits with Bounded Rewards.](http://arxiv.org/abs/2304.14989) | 本文提出了Kullback-Leibler Maillard Sampling (KL-MS)算法，能够在有界奖励的多臂赌博机中实现KL空间的扩展，具有较好的渐近性能。 |

# 详细

[^1]: 神经缩放定律的动力学模型

    A Dynamical Model of Neural Scaling Laws

    [https://rss.arxiv.org/abs/2402.01092](https://rss.arxiv.org/abs/2402.01092)

    这篇论文提出了一个动力学模型来解释神经缩放定律。通过分析梯度下降训练的随机特征模型，研究发现训练时间和模型大小的缩放具有不同的幂律指数，而计算最优缩放规则要求增加训练步数快于增加模型参数，与实证观察相一致。

    

    在各种任务中，神经网络的性能随着训练时间、数据集大小和模型大小的增加而预测性地提高，跨多个数量级。这种现象被称为神经缩放定律。最重要的是计算最优缩放定律，它报告了在选择最佳模型大小时性能与计算数量的关系。我们分析了一个通过梯度下降进行训练和泛化的随机特征模型作为网络训练和泛化的可解模型。这个模型复现了关于神经缩放定律的许多观察结果。首先，我们的模型对于为什么训练时间和模型大小的缩放具有不同的幂律指数提出了一个预测。因此，理论预测了一种不对称的计算最优缩放规则，其中训练步数的增加速度快于模型参数的增加速度，与最近的实证观察一致。其次，观察到在训练的早期，网络会收敛到无限宽度情况下的结果。

    On a variety of tasks, the performance of neural networks predictably improves with training time, dataset size and model size across many orders of magnitude. This phenomenon is known as a neural scaling law. Of fundamental importance is the compute-optimal scaling law, which reports the performance as a function of units of compute when choosing model sizes optimally. We analyze a random feature model trained with gradient descent as a solvable model of network training and generalization. This reproduces many observations about neural scaling laws. First, our model makes a prediction about why the scaling of performance with training time and with model size have different power law exponents. Consequently, the theory predicts an asymmetric compute-optimal scaling rule where the number of training steps are increased faster than model parameters, consistent with recent empirical observations. Second, it has been observed that early in training, networks converge to their infinite-wi
    
[^2]: 具有重建损失的迁移学习

    Transfer Learning with Reconstruction Loss

    [https://arxiv.org/abs/2404.00505](https://arxiv.org/abs/2404.00505)

    本文通过引入额外的重建阶段和重建损失，提出了一种具有共享模型参数和特征表示的模型训练方法，建立了共同信息的概念，用于解决相关任务。

    

    在大多数利用神经网络进行数学优化的应用中，通常为每个特定优化目标训练一个专用模型。然而，在许多场景中，同一组问题输入上经常需要优化几个不同但相关的目标或任务。与为每个问题单独训练不同的神经网络相比，更有效的方法是利用这些目标之间的相关性，使用共享模型参数和特征表示训练多个神经网络模型。为实现这一目标，本文首先建立了共同信息的概念：解决相关任务所需的共享知识，然后提出了一种新颖的模型训练方法，通过在模型中添加一个额外的重建阶段以及相关的新重建损失。该损失用于从选择的隐藏状态开始重新构建共同信息。

    arXiv:2404.00505v1 Announce Type: cross  Abstract: In most applications of utilizing neural networks for mathematical optimization, a dedicated model is trained for each specific optimization objective. However, in many scenarios, several distinct yet correlated objectives or tasks often need to be optimized on the same set of problem inputs. Instead of independently training a different neural network for each problem separately, it would be more efficient to exploit the correlations between these objectives and to train multiple neural network models with shared model parameters and feature representations. To achieve this, this paper first establishes the concept of common information: the shared knowledge required for solving the correlated tasks, then proposes a novel approach for model training by adding into the model an additional reconstruction stage associated with a new reconstruction loss. This loss is for reconstructing the common information starting from a selected hidde
    
[^3]: TaCo：通过信息论和可解释性在NLP中的输出嵌入中实现有针对性的概念去除

    TaCo: Targeted Concept Removal in Output Embeddings for NLP via Information Theory and Explainability. (arXiv:2312.06499v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2312.06499](http://arxiv.org/abs/2312.06499)

    本论文提出了一种新颖的方法，通过对NLP模型的嵌入层级进行操作，借鉴了最新的解释性人工智能技术，通过嵌入转换来消除隐含的敏感信息，从而实现模型的公平性。

    

    自然语言处理（NLP）模型的公平性已成为一个关键问题。信息论表明，为了实现公平性，模型不应能够预测敏感变量，如性别、种族和年龄。然而，与这些变量相关的信息通常以隐式的方式出现在语言中，这给识别和减少偏见带来了挑战。为了解决这个问题，我们提出了一种新颖的方法，在NLP模型的嵌入层级上操作，独立于具体的架构。我们的方法借鉴了最近解释性人工智能技术的进展，并采用嵌入转换来消除选定变量中的隐式信息。通过直接操纵最后一层的嵌入，我们的方法能够无缝集成到现有模型中，而无需进行重大修改或重训练。在评估中，我们展示了该后处理方法显著降低了与性别相关的关联性。

    The fairness of Natural Language Processing (NLP) models has emerged as a crucial concern. Information theory indicates that to achieve fairness, a model should not be able to predict sensitive variables, such as gender, ethnicity, and age. However, information related to these variables often appears implicitly in language, posing a challenge in identifying and mitigating biases effectively. To tackle this issue, we present a novel approach that operates at the embedding level of an NLP model, independent of the specific architecture. Our method leverages insights from recent advances in XAI techniques and employs an embedding transformation to eliminate implicit information from a selected variable. By directly manipulating the embeddings in the final layer, our approach enables a seamless integration into existing models without requiring significant modifications or retraining. In evaluation, we show that the proposed post-hoc approach significantly reduces gender-related associati
    
[^4]: 整数值时间序列数据的神经似然近似

    Neural Likelihood Approximation for Integer Valued Time Series Data. (arXiv:2310.12544v1 [stat.ML])

    [http://arxiv.org/abs/2310.12544](http://arxiv.org/abs/2310.12544)

    本文构建了整数值时间序列数据的神经似然近似方法，使用因果卷积并行评估整个时间序列的似然，实现了对生态学和流行病学模型进行准确推断并显著加快计算速度。

    

    在物理和生物科学中，定义在整数值状态空间上的随机过程很常见。这些模型用于捕捉小系统的动力学，其中个体群体的个体属性不能被忽视，随机效应很重要。由于似然的复杂性，从时间序列数据中推断这些模型的参数是困难的；目前的方法基于基础模型的模拟，计算成本非常高昂，以至于难以实现。在本文中，我们使用因果卷积构建了用于整数值时间序列数据的神经似然近似方法，这使我们能够并行评估整个时间序列的似然。我们通过对一些生态学和流行病学模型进行推断来演示我们的方法，结果显示我们能够准确地近似真实的后验概率，同时在当前方法受限的情况下实现显著的计算加速。

    Stochastic processes defined on integer valued state spaces are popular within the physical and biological sciences. These models are necessary for capturing the dynamics of small systems where the individual nature of the populations cannot be ignored and stochastic effects are important. The inference of the parameters of such models, from time series data, is difficult due to intractability of the likelihood; current methods, based on simulations of the underlying model, can be so computationally expensive as to be prohibitive. In this paper we construct a neural likelihood approximation for integer valued time series data using causal convolutions, which allows us to evaluate the likelihood of the whole time series in parallel. We demonstrate our method by performing inference on a number of ecological and epidemiological models, showing that we can accurately approximate the true posterior while achieving significant computational speed ups in situations where current methods stru
    
[^5]: 在在线排名中的最小极大后悔问题与Top-k反馈

    On the Minimax Regret in Online Ranking with Top-k Feedback. (arXiv:2309.02425v1 [cs.LG])

    [http://arxiv.org/abs/2309.02425](http://arxiv.org/abs/2309.02425)

    本文研究了在线排名中的最小极大后悔问题与Top-k反馈，并通过提供一个全面的极大后悔率刻画，解决了Chaudhuri和Tewari [2017]所提出的问题。

    

    在在线排名中，学习算法按顺序对一组项目进行排名，并以相关性得分的形式接收反馈。由于获得相关性得分通常涉及人工注释，因此考虑仅对排名中的前k个项目限制反馈的部分反馈设置具有极大的兴趣。Chaudhuri和Tewari [2017]开发了一个框架来分析带有Top $k$反馈的在线排名算法。他们工作的关键要素是使用了部分监控技术。在本文中，我们进一步研究了具有Top $k$反馈的在线排名，并解决了Chaudhuri和Tewari [2017]提出的一些开放性问题。我们对所有$k$和以下排名性能度量（Pairwise Loss，Discounted Cumulative Gain和Precision@n）的Top $k$反馈模型进行了最小极大后悔率的完全刻画。此外，我们提供了一种有效的算法，该算法实现了Precision@n的最小极大后悔率。

    In online ranking, a learning algorithm sequentially ranks a set of items and receives feedback on its ranking in the form of relevance scores. Since obtaining relevance scores typically involves human annotation, it is of great interest to consider a partial feedback setting where feedback is restricted to the top-$k$ items in the rankings. Chaudhuri and Tewari [2017] developed a framework to analyze online ranking algorithms with top $k$ feedback. A key element in their work was the use of techniques from partial monitoring. In this paper, we further investigate online ranking with top $k$ feedback and solve some open problems posed by Chaudhuri and Tewari [2017]. We provide a full characterization of minimax regret rates with the top $k$ feedback model for all $k$ and for the following ranking performance measures: Pairwise Loss, Discounted Cumulative Gain, and Precision@n. In addition, we give an efficient algorithm that achieves the minimax regret rate for Precision@n.
    
[^6]: 快速空间建模的综合变分傅里叶特征

    Integrated Variational Fourier Features for Fast Spatial Modelling with Gaussian Processes. (arXiv:2308.14142v1 [stat.ML])

    [http://arxiv.org/abs/2308.14142](http://arxiv.org/abs/2308.14142)

    本论文提出了一种综合变分傅里叶特征的方法，可以对广泛的平稳协方差函数进行快速空间建模，相比其他方法具有更高的性能和加速效果。

    

    稀疏变分逼近是扩展高斯过程推理和学习至更大数据集的流行方法。对于$N$个训练点，精确推理的成本为$O(N^3)$；使用$M \ll N$特征的先进稀疏变分方法成本为$O(NM^2)$。最近，提出了使用更复杂特征的方法；这些方法在低维任务（如空间建模）中能够有很好的性能，但只使用了一类非常有限的核函数，排除了一些常用的核函数。在这项工作中，我们提出了综合傅里叶特征，将这些性能优势扩展到非常广泛的平稳协方差函数。我们从收敛分析和经验探索的角度来解释该方法和参数的选择，同时展示该方法在合成和实际空间回归任务中的实际加速效果。

    Sparse variational approximations are popular methods for scaling up inference and learning in Gaussian processes to larger datasets. For $N$ training points, exact inference has $O(N^3)$ cost; with $M \ll N$ features, state of the art sparse variational methods have $O(NM^2)$ cost. Recently, methods have been proposed using more sophisticated features; these promise $O(M^3)$ cost, with good performance in low dimensional tasks such as spatial modelling, but they only work with a very limited class of kernels, excluding some of the most commonly used. In this work, we propose integrated Fourier features, which extends these performance benefits to a very broad class of stationary covariance functions. We motivate the method and choice of parameters from a convergence analysis and empirical exploration, and show practical speedup in synthetic and real world spatial regression tasks.
    
[^7]: 对在线技能评级建模和推理的状态空间视角的研究

    A State-Space Perspective on Modelling and Inference for Online Skill Rating. (arXiv:2308.02414v1 [stat.AP])

    [http://arxiv.org/abs/2308.02414](http://arxiv.org/abs/2308.02414)

    本文提供了对竞技体育技能评级主要方法的全面回顾，并提出了采用状态空间模型视角的建议。通过使用状态空间模型视角，玩家的技能可以表示为随时间变化的变量，而比赛结果则是唯一的观测量。该视角有助于解耦建模和推理，并促进通用推理工具的发展。在构建技能评级的状态空间模型方面，本文探讨了基本步骤，同时还讨论了滤波、平滑和参数估计等推理阶段。在面对高维场景中的计算挑战时，本文强调了所采用的近似和简化方法。该文提供了对记录的流行方法的简明总结。

    

    本文全面回顾了用于竞技体育技能评级的主要方法。我们提倡采用状态空间模型视角，将玩家的技能表示为随时间变化的，比赛结果作为唯一的观测量。状态空间模型视角有助于解耦建模和推理，从而使得更加注重模型假设的方法得以突出，并促进了通用推理工具的发展。我们探讨了构建技能评级的状态空间模型的基本步骤，并讨论了推理的三个阶段：滤波、平滑和参数估计。在整个过程中，我们研究了在涉及大量玩家和比赛的高维场景中进行规模扩展的计算挑战，强调了用于有效应对这些挑战的近似和简化方法。我们提供了文献中记录的流行方法的简明总结。

    This paper offers a comprehensive review of the main methodologies used for skill rating in competitive sports. We advocate for a state-space model perspective, wherein players' skills are represented as time-varying, and match results serve as the sole observed quantities. The state-space model perspective facilitates the decoupling of modeling and inference, enabling a more focused approach highlighting model assumptions, while also fostering the development of general-purpose inference tools. We explore the essential steps involved in constructing a state-space model for skill rating before turning to a discussion on the three stages of inference: filtering, smoothing and parameter estimation. Throughout, we examine the computational challenges of scaling up to high-dimensional scenarios involving numerous players and matches, highlighting approximations and reductions used to address these challenges effectively. We provide concise summaries of popular methods documented in the lit
    
[^8]: 离散切割Wasserstein损失的性质

    Properties of Discrete Sliced Wasserstein Losses. (arXiv:2307.10352v1 [stat.ML])

    [http://arxiv.org/abs/2307.10352](http://arxiv.org/abs/2307.10352)

    本文研究了离散切割Wasserstein损失的性质，并探讨了其正则性和优化性质以及通过蒙特卡洛近似的方法。

    

    切割Wasserstein（SW）距离已成为比较概率测度的Wasserstein距离的一种流行替代方法。广泛应用包括图像处理、领域自适应和生成建模，常常需要优化一些参数以最小化SW，该参数充当离散概率测度之间的损失函数（因为具有密度的测度在数值上是无法实现的）。所有这些优化问题都存在相同的子问题，即最小化切割Wasserstein能量。在本文中，我们研究了$\mathcal{E}: Y \longmapsto \mathrm{SW}_2^2(\gamma_Y, \gamma_Z)$的属性，即两个具有与一个测度的支撑相同数量的离散均匀测度之间的SW距离作为支撑$Y \in \mathbb{R}^{n \times d}$函数的能量。我们研究了这个能量的正则性和优化性质，以及其通过蒙特卡洛近似$\mathcal{E}_p$（使用SW中的期望估计）。

    The Sliced Wasserstein (SW) distance has become a popular alternative to the Wasserstein distance for comparing probability measures. Widespread applications include image processing, domain adaptation and generative modelling, where it is common to optimise some parameters in order to minimise SW, which serves as a loss function between discrete probability measures (since measures admitting densities are numerically unattainable). All these optimisation problems bear the same sub-problem, which is minimising the Sliced Wasserstein energy. In this paper we study the properties of $\mathcal{E}: Y \longmapsto \mathrm{SW}_2^2(\gamma_Y, \gamma_Z)$, i.e. the SW distance between two uniform discrete measures with the same amount of points as a function of the support $Y \in \mathbb{R}^{n \times d}$ of one of the measures. We investigate the regularity and optimisation properties of this energy, as well as its Monte-Carlo approximation $\mathcal{E}_p$ (estimating the expectation in SW using 
    
[^9]: 基于核方法的单细胞差异分析测试

    Kernel-Based Testing for Single-Cell Differential Analysis. (arXiv:2307.08509v1 [stat.ML])

    [http://arxiv.org/abs/2307.08509](http://arxiv.org/abs/2307.08509)

    本论文提出了一种基于核方法的单细胞差异分析测试框架，可以非线性比较复杂的细胞间分子特征分布。通过利用核嵌入的变异性，我们的方法能够揭示细胞群体中隐蔽的异质性。我们展示了核测试如何克服单细胞差异分析方法的局限性，并应用于研究分化逆转的过程。

    

    单细胞技术为我们提供了关于基因表达和表观遗传修饰等分子特征的宝贵信息。然而，以控制和强有力的方式比较这些复杂分布面临着方法论上的挑战。本文提出利用基于核嵌入的核测试框架来非线性比较细胞间复杂分子特征的分布。我们的框架不仅允许对特征进行分析，还能在考虑了它们之间复杂依赖关系的情况下进行转录组或表观组的全局比较。通过使用分类器基于核嵌入的变异性来区分细胞，我们的方法可以发现在细胞群体中原本无法察觉到的异质性。我们展示了核测试方法如何克服专门用于单细胞的差异分析方法的局限性。我们还将核测试应用于研究分化逆转的过程。

    Single-cell technologies have provided valuable insights into the distribution of molecular features, such as gene expression and epigenomic modifications. However, comparing these complex distributions in a controlled and powerful manner poses methodological challenges. Here we propose to benefit from the kernel-testing framework to compare the complex cell-wise distributions of molecular features in a non-linear manner based on their kernel embedding. Our framework not only allows for feature-wise analyses but also enables global comparisons of transcriptomes or epigenomes, considering their intricate dependencies. By using a classifier to discriminate cells based on the variability of their embedding, our method uncovers heterogeneities in cell populations that would otherwise go undetected. We show that kernel testing overcomes the limitations of differential analysis methods dedicated to single-cell. Kernel testing is applied to investigate the reversion process of differentiating
    
[^10]: 非累积目标的强化学习

    Reinforcement Learning with Non-Cumulative Objective. (arXiv:2307.04957v1 [cs.LG])

    [http://arxiv.org/abs/2307.04957](http://arxiv.org/abs/2307.04957)

    本文研究了最优控制和强化学习中非累积目标的挑战，并提出了修改现有算法的方法来优化这些目标。研究结果表明，在贝尔曼最优性方程中使用广义运算可以更好地处理非累积目标。

    

    在强化学习中，目标几乎总是定义为沿过程中奖励的\emph{累积}函数。然而，在许多最优控制和强化学习问题中，尤其是在通信和网络领域中，目标并不自然地表达为奖励的求和。本文中，我们认识到各种问题中非累积目标的普遍存在，并提出了修改现有算法以优化这些目标的方法。具体来说，我们深入研究了许多最优控制和强化学习算法的基本构建模块：贝尔曼最优性方程。为了优化非累积目标，我们用与目标相对应的广义运算替换了贝尔曼更新规则中的原始求和运算。此外，我们提供了广义运算形式的足够条件以及对马尔可夫决策的假设。

    In reinforcement learning, the objective is almost always defined as a \emph{cumulative} function over the rewards along the process. However, there are many optimal control and reinforcement learning problems in various application fields, especially in communications and networking, where the objectives are not naturally expressed as summations of the rewards. In this paper, we recognize the prevalence of non-cumulative objectives in various problems, and propose a modification to existing algorithms for optimizing such objectives. Specifically, we dive into the fundamental building block for many optimal control and reinforcement learning algorithms: the Bellman optimality equation. To optimize a non-cumulative objective, we replace the original summation operation in the Bellman update rule with a generalized operation corresponding to the objective. Furthermore, we provide sufficient conditions on the form of the generalized operation as well as assumptions on the Markov decision 
    
[^11]: 深度学习的拓扑可解释性

    Topological Interpretability for Deep-Learning. (arXiv:2305.08642v1 [stat.ML])

    [http://arxiv.org/abs/2305.08642](http://arxiv.org/abs/2305.08642)

    本文提出了一种在临床和非临床文本上训练的深度学习分类模型中使用拓扑和几何数据分析技术推断主要特征的方法。

    

    随着基于人工智能的系统在日常生活中的应用越来越广泛，理解它们的决策机制的需求也相应加速。我们能够信任基于人工智能决策系统所做的统计推断的程度越来越成为一个越来越重要的问题，特别是在高风险的系统，例如刑事司法或医学诊断系统中，错误的推断可能会产生悲剧性的后果。尽管在解决涉及现实世界数据的问题方面取得了成功，但深度学习（DL）模型无法量化其预测的确定性。而且当其解决方案不正确时，通常仍然非常自信。本文介绍了一种方法，在临床和非临床文本上训练的两个DL分类模型中推断杰出特征，采用了拓扑和几何数据分析技术。我们创建了模型预测空间的图形，并通过特征和预测的相似性将输入聚类到图形的顶点中。

    With the increasing adoption of AI-based systems across everyday life, the need to understand their decision-making mechanisms is correspondingly accelerating. The level at which we can trust the statistical inferences made from AI-based decision systems is an increasing concern, especially in high-risk systems such as criminal justice or medical diagnosis, where incorrect inferences may have tragic consequences. Despite their successes in providing solutions to problems involving real-world data, deep learning (DL) models cannot quantify the certainty of their predictions. And are frequently quite confident, even when their solutions are incorrect.  This work presents a method to infer prominent features in two DL classification models trained on clinical and non-clinical text by employing techniques from topological and geometric data analysis. We create a graph of a model's prediction space and cluster the inputs into the graph's vertices by the similarity of features and prediction
    
[^12]: Kullback-Leibler Maillard采样在有界奖励的多臂赌博机问题中的应用

    Kullback-Leibler Maillard Sampling for Multi-armed Bandits with Bounded Rewards. (arXiv:2304.14989v1 [cs.LG])

    [http://arxiv.org/abs/2304.14989](http://arxiv.org/abs/2304.14989)

    本文提出了Kullback-Leibler Maillard Sampling (KL-MS)算法，能够在有界奖励的多臂赌博机中实现KL空间的扩展，具有较好的渐近性能。

    

    本文研究了奖励分布集中在区间$[0,1]$内的$K$臂数臂赌博机问题。本文提出了一种名为Kullback-Leibler Maillard Sampling (KL-MS)的新算法，它是Maillard采样在KL空间的自然扩展。实验表明，KL-MS在Bernoulli奖励时具有渐近最优性能，其最坏情况遗憾度上界为$O(\sqrt{\mu^*(1-\mu^*) K T \ln K} + K \ln T)$，其中$\mu^*$是最优臂的期望奖励，$T$是时段长度。

    We study $K$-armed bandit problems where the reward distributions of the arms are all supported on the $[0,1]$ interval. It has been a challenge to design regret-efficient randomized exploration algorithms in this setting. Maillard sampling~\cite{maillard13apprentissage}, an attractive alternative to Thompson sampling, has recently been shown to achieve competitive regret guarantees in the sub-Gaussian reward setting~\cite{bian2022maillard} while maintaining closed-form action probabilities, which is useful for offline policy evaluation. In this work, we propose the Kullback-Leibler Maillard Sampling (KL-MS) algorithm, a natural extension of Maillard sampling for achieving KL-style gap-dependent regret bound. We show that KL-MS enjoys the asymptotic optimality when the rewards are Bernoulli and has a worst-case regret bound of the form $O(\sqrt{\mu^*(1-\mu^*) K T \ln K} + K \ln T)$, where $\mu^*$ is the expected reward of the optimal arm, and $T$ is the time horizon length.
    

