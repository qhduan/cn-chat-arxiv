# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Rapid Bayesian identification of sparse nonlinear dynamics from scarce and noisy data](https://arxiv.org/abs/2402.15357) | 提出了一种快速的概率框架，称为贝叶斯-SINDy，用于从有限且嘈杂数据中学习正确的模型方程，并且对参数估计中的不确定性进行量化，特别适用于生物数据和实时系统识别。 |
| [^2] | [Chain of Thought Empowers Transformers to Solve Inherently Serial Problems](https://arxiv.org/abs/2402.12875) | 思维链赋予变压器模型执行固有串行计算的能力，提高了变压器在算术和符号推理任务中的准确性。 |
| [^3] | [Predictive Analysis for Optimizing Port Operations.](http://arxiv.org/abs/2401.14498) | 本研究开发了一种具有竞争预测和分类能力的港口运营解决方案，用于准确估计船舶在港口的总时间和延迟时间，填补了港口分析模型在这方面的空白，并为海事物流领域提供了有价值的贡献。 |
| [^4] | [Early alignment in two-layer networks training is a two-edged sword.](http://arxiv.org/abs/2401.10791) | 本文研究了两层网络训练中的早期对齐现象，发现在小初始化和一个隐藏的ReLU层网络中，神经元会在训练的早期阶段向关键方向进行对齐，导致网络稀疏表示以及梯度流在收敛时的隐含偏好。然而，这种稀疏诱导的对齐也使得训练目标的最小化变得困难。 |
| [^5] | [Improving the Accuracy and Interpretability of Random Forests via Forest Pruning.](http://arxiv.org/abs/2401.05535) | 通过森林修剪方法，本研究提出了一种兼顾随机森林准确性和决策树可解释性的方法。实验证明，在大多数情景下，这种方法能够显著提高随机森林的性能。 |
| [^6] | [Convergence Rates for Stochastic Approximation: Biased Noise with Unbounded Variance, and Applications.](http://arxiv.org/abs/2312.02828) | 本论文研究了带有无界方差的有偏噪声对随机逼近算法的收敛速度的影响，并介绍了该算法在各个领域的应用。 |
| [^7] | [Policy Learning with Distributional Welfare.](http://arxiv.org/abs/2311.15878) | 本文提出了一种针对分配福利的最优治疗分配策略，该策略根据个体治疗效应的条件分位数来决定治疗分配，并引入了鲁棒的最小最大化策略来解决对反事实结果联合分布的恢复问题。 |
| [^8] | [Discriminator Guidance for Autoregressive Diffusion Models.](http://arxiv.org/abs/2310.15817) | 本文引入了判别器引导，用于自回归扩散模型的训练，通过使用最优判别器来纠正预训练模型，并提出了一个顺序蒙特卡洛算法来应对使用次优判别器的情况。在生成分子图的任务中，判别器引导有助于提高生成性能。 |
| [^9] | [Stochastic interpolants with data-dependent couplings.](http://arxiv.org/abs/2310.03725) | 本文提出了一种使用数据依赖耦合来构建生成模型的方法，并展示了在超分辨率和修复任务中的实验效果。 |
| [^10] | [Spectrum-Aware Adjustment: A New Debiasing Framework with Applications to Principal Components Regression.](http://arxiv.org/abs/2309.07810) | 这项研究介绍了一种新的去偏方法框架，用于解决高维线性回归中现有技术对协变量分布的限制问题。研究者们发现，现有方法在处理非高斯分布、异质性设计矩阵和缺乏可靠特征协方差估计时遇到困难。为了解决这些问题，他们提出了一种新的策略，该策略利用缩放的梯度下降步骤进行去偏校正。 |
| [^11] | [Computable Stability for Persistence Rank Function Machine Learning.](http://arxiv.org/abs/2307.02904) | 本文讨论了持续性同调秩函数在拓扑数据分析中的应用，并指出秩函数相对于条码的优势在于更易于计算，并且可以直接应用函数数据分析的方法。然而，秩函数的稳定性问题尚待解决。 |
| [^12] | [Noise Stability Optimization for Flat Minima with Optimal Convergence Rates.](http://arxiv.org/abs/2306.08553) | 本文提出了一个SGD-like算法，注入随机噪声并利用分布对称性来减少方差，以寻找具有低海森矩阵迹的平坦极小值，同时提供了收敛速率分析。 |
| [^13] | [Batches Stabilize the Minimum Norm Risk in High Dimensional Overparameterized Linear Regression.](http://arxiv.org/abs/2306.08432) | 本文研究了将数据分成批次的学习算法，在高维超参数线性回归模型中提供了隐式正则化，通过适当的批量大小选择，稳定了风险行为，消除了插值点处的膨胀和双峰现象 |
| [^14] | [repliclust: Synthetic Data for Cluster Analysis.](http://arxiv.org/abs/2303.14301) | repliclust 是一个 Python 包，用于生成具有聚类的合成数据集，基于数据集的原型，提供了放置集群中心、采样集群形状、选择每个集群的数据点数量以及为集群分配概率分布的算法。 |
| [^15] | [Counterfactual inference for sequential experiments.](http://arxiv.org/abs/2202.06891) | 本文针对序列实验的反事实推断问题，提出了一个潜在因子模型，使用非参数方法对反事实均值进行估计，并建立了误差界限。 |

# 详细

[^1]: 从稀疏且嘈杂数据中快速识别稀疏非线性动力学的贝叶斯方法

    Rapid Bayesian identification of sparse nonlinear dynamics from scarce and noisy data

    [https://arxiv.org/abs/2402.15357](https://arxiv.org/abs/2402.15357)

    提出了一种快速的概率框架，称为贝叶斯-SINDy，用于从有限且嘈杂数据中学习正确的模型方程，并且对参数估计中的不确定性进行量化，特别适用于生物数据和实时系统识别。

    

    我们提出了一个快速的概率框架，用于识别控制观测数据动态的微分方程。我们将SINDy方法重新构建到贝叶斯框架中，并使用高斯逼近来加速计算。由此产生的方法，贝叶斯-SINDy，不仅量化了参数估计中的不确定性，而且在从有限且嘈杂数据中学习正确模型时更加稳健。通过使用合成和真实例子，如猞猁-野兔种群动态，我们展示了新框架在学习正确模型方程中的有效性，并比较了其与现有方法的计算和数据效率。由于贝叶斯-SINDy可以快速吸收数据并对噪声具有稳健性，因此特别适用于生物数据和控制中的实时系统识别。其概率框架还使得可以计算信息熵。

    arXiv:2402.15357v1 Announce Type: cross  Abstract: We propose a fast probabilistic framework for identifying differential equations governing the dynamics of observed data. We recast the SINDy method within a Bayesian framework and use Gaussian approximations for the prior and likelihood to speed up computation. The resulting method, Bayesian-SINDy, not only quantifies uncertainty in the parameters estimated but also is more robust when learning the correct model from limited and noisy data. Using both synthetic and real-life examples such as Lynx-Hare population dynamics, we demonstrate the effectiveness of the new framework in learning correct model equations and compare its computational and data efficiency with existing methods. Because Bayesian-SINDy can quickly assimilate data and is robust against noise, it is particularly suitable for biological data and real-time system identification in control. Its probabilistic framework also enables the calculation of information entropy, 
    
[^2]: 思维链激发变压器解决固有串行问题的能力

    Chain of Thought Empowers Transformers to Solve Inherently Serial Problems

    [https://arxiv.org/abs/2402.12875](https://arxiv.org/abs/2402.12875)

    思维链赋予变压器模型执行固有串行计算的能力，提高了变压器在算术和符号推理任务中的准确性。

    

    指导模型生成一系列中间步骤，即思维链（CoT），是提高大型语言模型（LLMs）在算术和符号推理任务上准确性的高效方法。然而，CoT背后的机制仍不清楚。这项工作通过表达性的视角提供了对解码器专用变压器的CoT能力的理论理解。在概念上，CoT赋予模型执行固有串行计算的能力，而这种能力在变压器中缺乏，特别是当深度较低时。先前的作品已经表明，在没有CoT的情况下，具有有限精度$\mathsf{poly}(n)$嵌入尺寸的恒定深度变压器只能在$\mathsf{TC}^0$中解决问题。我们首先展示了具有常数位精度的恒定深度变压器的更紧密的表达性上界，它只能解决$\mathsf{AC}^0$中的问题。

    arXiv:2402.12875v1 Announce Type: new  Abstract: Instructing the model to generate a sequence of intermediate steps, a.k.a., a chain of thought (CoT), is a highly effective method to improve the accuracy of large language models (LLMs) on arithmetics and symbolic reasoning tasks. However, the mechanism behind CoT remains unclear. This work provides a theoretical understanding of the power of CoT for decoder-only transformers through the lens of expressiveness. Conceptually, CoT empowers the model with the ability to perform inherently serial computation, which is otherwise lacking in transformers, especially when depth is low. Given input length $n$, previous works have shown that constant-depth transformers with finite precision $\mathsf{poly}(n)$ embedding size can only solve problems in $\mathsf{TC}^0$ without CoT. We first show an even tighter expressiveness upper bound for constant-depth transformers with constant-bit precision, which can only solve problems in $\mathsf{AC}^0$, a 
    
[^3]: 优化港口运营的预测分析

    Predictive Analysis for Optimizing Port Operations. (arXiv:2401.14498v1 [cs.LG])

    [http://arxiv.org/abs/2401.14498](http://arxiv.org/abs/2401.14498)

    本研究开发了一种具有竞争预测和分类能力的港口运营解决方案，用于准确估计船舶在港口的总时间和延迟时间，填补了港口分析模型在这方面的空白，并为海事物流领域提供了有价值的贡献。

    

    海运是远距离和大宗货物运输的重要物流方式。然而，这种运输模式中复杂的规划经常受到不确定性的影响，包括天气条件、货物多样性和港口动态，导致成本增加。因此，准确估计船舶在港口停留的总时间和潜在延迟变得至关重要，以便在港口运营中进行有效的规划和安排。本研究旨在开发具有竞争预测和分类能力的港口运营解决方案，用于估计船舶的总时间和延迟时间。该研究填补了港口分析模型在船舶停留和延迟时间方面的重要空白，为海事物流领域提供了有价值的贡献。所提出的解决方案旨在协助港口环境下的决策制定，并预测服务延迟。通过对巴西港口的案例研究进行验证，同时使用特征分析来理解...

    Maritime transport is a pivotal logistics mode for the long-distance and bulk transportation of goods. However, the intricate planning involved in this mode is often hindered by uncertainties, including weather conditions, cargo diversity, and port dynamics, leading to increased costs. Consequently, accurately estimating vessel total (stay) time at port and potential delays becomes imperative for effective planning and scheduling in port operations. This study aims to develop a port operation solution with competitive prediction and classification capabilities for estimating vessel Total and Delay times. This research addresses a significant gap in port analysis models for vessel Stay and Delay times, offering a valuable contribution to the field of maritime logistics. The proposed solution is designed to assist decision-making in port environments and predict service delays. This is demonstrated through a case study on Brazil ports. Additionally, feature analysis is used to understand
    
[^4]: 两层网络训练中的早期对齐是一把双刃剑

    Early alignment in two-layer networks training is a two-edged sword. (arXiv:2401.10791v1 [cs.LG])

    [http://arxiv.org/abs/2401.10791](http://arxiv.org/abs/2401.10791)

    本文研究了两层网络训练中的早期对齐现象，发现在小初始化和一个隐藏的ReLU层网络中，神经元会在训练的早期阶段向关键方向进行对齐，导致网络稀疏表示以及梯度流在收敛时的隐含偏好。然而，这种稀疏诱导的对齐也使得训练目标的最小化变得困难。

    

    使用一阶优化方法训练神经网络是深度学习成功的核心。初始化的规模是一个关键因素，因为小的初始化通常与特征学习模式相关，在这种模式下，梯度下降对简单解隐含偏好。本文提供了早期对齐阶段的普遍和量化描述，最初由Maennel等人提出。对于小初始化和一个隐藏的ReLU层网络，训练动态的早期阶段导致神经元向关键方向进行对齐。这种对齐引发了网络的稀疏表示，这与梯度流在收敛时的隐含偏好直接相关。然而，这种稀疏诱导的对齐是以在最小化训练目标方面遇到困难为代价的：我们还提供了一个简单的数据示例，其中超参数网络无法收敛到全局最小值。

    Training neural networks with first order optimisation methods is at the core of the empirical success of deep learning. The scale of initialisation is a crucial factor, as small initialisations are generally associated to a feature learning regime, for which gradient descent is implicitly biased towards simple solutions. This work provides a general and quantitative description of the early alignment phase, originally introduced by Maennel et al. (2018) . For small initialisation and one hidden ReLU layer networks, the early stage of the training dynamics leads to an alignment of the neurons towards key directions. This alignment induces a sparse representation of the network, which is directly related to the implicit bias of gradient flow at convergence. This sparsity inducing alignment however comes at the expense of difficulties in minimising the training objective: we also provide a simple data example for which overparameterised networks fail to converge towards global minima and
    
[^5]: 通过森林修剪提高随机森林的准确性和可解释性

    Improving the Accuracy and Interpretability of Random Forests via Forest Pruning. (arXiv:2401.05535v1 [stat.ML])

    [http://arxiv.org/abs/2401.05535](http://arxiv.org/abs/2401.05535)

    通过森林修剪方法，本研究提出了一种兼顾随机森林准确性和决策树可解释性的方法。实验证明，在大多数情景下，这种方法能够显著提高随机森林的性能。

    

    接近几十年的发展之后，随机森林仍然在各种学习问题中提供最先进的准确性，在这方面超越了决策树甚至神经网络等替代机器学习算法。然而，作为一种集成方法，随机森林在解释性方面往往比决策树表现不佳。在本研究中，我们提出了一种事后方法，旨在兼顾随机森林的准确性和决策树的可解释性。为此，我们提出了两种森林修剪方法，以在给定的随机森林内找到最佳子森林，然后在适用的情况下将选定的树合并为一棵。我们的第一种方法依赖于约束穷举搜索，而第二种方法基于LASSO方法的改进。在合成和真实世界数据集上进行的大量实验证明，在大多数情景下，这两种方法中至少有一种能够显著提高随机森林的准确性和可解释性。

    Decades after their inception, random forests continue to provide state-of-the-art accuracy in a variety of learning problems, outperforming in this respect alternative machine learning algorithms such as decision trees or even neural networks. However, being an ensemble method, the one aspect where random forests tend to severely underperform decision trees is interpretability. In the present work, we propose a post-hoc approach that aims to have the best of both worlds: the accuracy of random forests and the interpretability of decision trees. To this end, we present two forest-pruning methods to find an optimal sub-forest within a given random forest, and then, when applicable, combine the selected trees into one. Our first method relies on constrained exhaustive search, while our second method is based on an adaptation of the LASSO methodology. Extensive experiments over synthetic and real world datasets show that, in the majority of scenarios, at least one of the two methods propo
    
[^6]: 随机逼近的收敛速度：带有无界方差的有偏噪声和应用

    Convergence Rates for Stochastic Approximation: Biased Noise with Unbounded Variance, and Applications. (arXiv:2312.02828v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2312.02828](http://arxiv.org/abs/2312.02828)

    本论文研究了带有无界方差的有偏噪声对随机逼近算法的收敛速度的影响，并介绍了该算法在各个领域的应用。

    

    1951年罗宾斯和莫洛引入的随机逼近（SA）算法已经成为解方程$\mathbf{f}({\boldsymbol{\theta}}) = \mathbf{0}$的标准方法，当只有$\mathbf{f}(\cdot)$的带噪声测量可用时。如果对于某个函数$J(\cdot)$，$\mathbf{f}({\boldsymbol{\theta}}) = \nabla J({\boldsymbol{\theta}})$，那么SA也可以用来寻找$J(\cdot)$的一个稳定点。在每个时间$t$，当前的猜测${\boldsymbol{\theta}}_t$通过形式为$\mathbf{f}({\boldsymbol{\theta}}_t) + {\boldsymbol{\xi}}_{t+1}$的带噪声测量更新为${\boldsymbol{\theta}}_{t+1}$。在许多文献中，假设误差项${\boldsymbol{\xi}}_{t+1}$的条件均值为零，和/或者它的条件方差随$t$（而不是${\boldsymbol{\theta}}_t$）被限制。多年来，SA已经应用于各种领域，本文重点研究其中一个领域。

    The Stochastic Approximation (SA) algorithm introduced by Robbins and Monro in 1951 has been a standard method for solving equations of the form $\mathbf{f}({\boldsymbol {\theta}}) = \mathbf{0}$, when only noisy measurements of $\mathbf{f}(\cdot)$ are available. If $\mathbf{f}({\boldsymbol {\theta}}) = \nabla J({\boldsymbol {\theta}})$ for some function $J(\cdot)$, then SA can also be used to find a stationary point of $J(\cdot)$. At each time $t$, the current guess ${\boldsymbol {\theta}}_t$ is updated to ${\boldsymbol {\theta}}_{t+1}$ using a noisy measurement of the form $\mathbf{f}({\boldsymbol {\theta}}_t) + {\boldsymbol {\xi}}_{t+1}$. In much of the literature, it is assumed that the error term ${\boldsymbol {\xi}}_{t+1}$ has zero conditional mean, and/or that its conditional variance is bounded as a function of $t$ (though not necessarily with respect to ${\boldsymbol {\theta}}_t$). Over the years, SA has been applied to a variety of areas, out of which the focus in this paper i
    
[^7]: 分配福利的政策学习

    Policy Learning with Distributional Welfare. (arXiv:2311.15878v2 [stat.ME] UPDATED)

    [http://arxiv.org/abs/2311.15878](http://arxiv.org/abs/2311.15878)

    本文提出了一种针对分配福利的最优治疗分配策略，该策略根据个体治疗效应的条件分位数来决定治疗分配，并引入了鲁棒的最小最大化策略来解决对反事实结果联合分布的恢复问题。

    

    本文探讨了针对分配福利的最优治疗分配策略。大部分关于治疗选择的文献都考虑了基于条件平均治疗效应（ATE）的功利福利。虽然平均福利是直观的，但在个体异质化（例如，存在离群值）情况下可能会产生不理想的分配 - 这正是个性化治疗引入的原因之一。这个观察让我们提出了一种根据个体治疗效应的条件分位数（QoTE）来分配治疗的最优策略。根据分位数概率的选择，这个准则可以适应谨慎或粗心的决策者。确定QoTE的挑战在于其需要对反事实结果的联合分布有所了解，但即使使用实验数据，通常也很难恢复出来。因此，我们介绍了鲁棒的最小最大化策略

    In this paper, we explore optimal treatment allocation policies that target distributional welfare. Most literature on treatment choice has considered utilitarian welfare based on the conditional average treatment effect (ATE). While average welfare is intuitive, it may yield undesirable allocations especially when individuals are heterogeneous (e.g., with outliers) - the very reason individualized treatments were introduced in the first place. This observation motivates us to propose an optimal policy that allocates the treatment based on the conditional quantile of individual treatment effects (QoTE). Depending on the choice of the quantile probability, this criterion can accommodate a policymaker who is either prudent or negligent. The challenge of identifying the QoTE lies in its requirement for knowledge of the joint distribution of the counterfactual outcomes, which is generally hard to recover even with experimental data. Therefore, we introduce minimax policies that are robust 
    
[^8]: 判别器引导下的自回归扩散模型

    Discriminator Guidance for Autoregressive Diffusion Models. (arXiv:2310.15817v1 [cs.LG])

    [http://arxiv.org/abs/2310.15817](http://arxiv.org/abs/2310.15817)

    本文引入了判别器引导，用于自回归扩散模型的训练，通过使用最优判别器来纠正预训练模型，并提出了一个顺序蒙特卡洛算法来应对使用次优判别器的情况。在生成分子图的任务中，判别器引导有助于提高生成性能。

    

    我们在自回归扩散模型中引入了判别器引导。在连续扩散模型中，使用判别器引导扩散过程的方法已经被使用过，本文中，我们推导了在离散情况下使用判别器和预训练生成模型的方法。首先，我们证明使用最优判别器将纠正预训练模型，并能够从底层数据分布中精确采样。其次，为了应对使用次优判别器的实际情况，我们推导了一个顺序蒙特卡洛算法，该算法在生成过程中迭代地将判别器的预测纳入考虑。我们将这些方法应用于生成分子图的任务，并展示了判别器相较于仅使用预训练模型时的生成性能提升。

    We introduce discriminator guidance in the setting of Autoregressive Diffusion Models. The use of a discriminator to guide a diffusion process has previously been used for continuous diffusion models, and in this work we derive ways of using a discriminator together with a pretrained generative model in the discrete case. First, we show that using an optimal discriminator will correct the pretrained model and enable exact sampling from the underlying data distribution. Second, to account for the realistic scenario of using a sub-optimal discriminator, we derive a sequential Monte Carlo algorithm which iteratively takes the predictions from the discrimiator into account during the generation process. We test these approaches on the task of generating molecular graphs and show how the discriminator improves the generative performance over using only the pretrained model.
    
[^9]: 具有数据依赖耦合的随机插值。

    Stochastic interpolants with data-dependent couplings. (arXiv:2310.03725v1 [cs.LG])

    [http://arxiv.org/abs/2310.03725](http://arxiv.org/abs/2310.03725)

    本文提出了一种使用数据依赖耦合来构建生成模型的方法，并展示了在超分辨率和修复任务中的实验效果。

    

    受动态测度传输启发的生成模型（如流和扩散）构建了两个概率密度之间的连续时间映射。按照传统方法，其中一个是目标密度，只能通过样本访问，而另一个是简单的基础密度，与数据无关。在这项工作中，我们使用随机插值的框架，规范化了如何“耦合”基本密度和目标密度。这使我们能够将类别标签或连续嵌入的信息纳入到构建动态传输映射的条件生成模型中。我们展示了通过解决类似于标准独立设置的简单平方损失回归问题来学习这些传输映射。通过超分辨率和修复实验，我们证明了构建依赖耦合的有效性。

    Generative models inspired by dynamical transport of measure -- such as flows and diffusions -- construct a continuous-time map between two probability densities. Conventionally, one of these is the target density, only accessible through samples, while the other is taken as a simple base density that is data-agnostic. In this work, using the framework of stochastic interpolants, we formalize how to \textit{couple} the base and the target densities. This enables us to incorporate information about class labels or continuous embeddings to construct dynamical transport maps that serve as conditional generative models. We show that these transport maps can be learned by solving a simple square loss regression problem analogous to the standard independent setting. We demonstrate the usefulness of constructing dependent couplings in practice through experiments in super-resolution and in-painting.
    
[^10]: Spectrum-Aware Adjustment: 一种新的去偏方法框架及其在主成分回归中的应用

    Spectrum-Aware Adjustment: A New Debiasing Framework with Applications to Principal Components Regression. (arXiv:2309.07810v1 [math.ST])

    [http://arxiv.org/abs/2309.07810](http://arxiv.org/abs/2309.07810)

    这项研究介绍了一种新的去偏方法框架，用于解决高维线性回归中现有技术对协变量分布的限制问题。研究者们发现，现有方法在处理非高斯分布、异质性设计矩阵和缺乏可靠特征协方差估计时遇到困难。为了解决这些问题，他们提出了一种新的策略，该策略利用缩放的梯度下降步骤进行去偏校正。

    

    我们引入了一个新的去偏方法框架，用于解决高维线性回归中现代去偏技术对协变量分布的约束问题。我们研究了特征数和样本数都很大且相近的普遍情况。在这种情况下，现代去偏技术使用自由度校正来除去正则化估计量的收缩偏差并进行推断。然而，该方法要求观测样本是独立同分布的，协变量遵循均值为零的高斯分布，并且能够获得可靠的特征协方差矩阵估计。当（i）协变量具有非高斯分布、重尾或非对称分布，（ii）设计矩阵的行呈异质性或存在依赖性，以及（iii）缺乏可靠的特征协方差估计时，这种方法就会遇到困难。为了应对这些问题，我们提出了一种新的策略，其中去偏校正是一步缩放的梯度下降步骤（适当缩放）。

    We introduce a new debiasing framework for high-dimensional linear regression that bypasses the restrictions on covariate distributions imposed by modern debiasing technology. We study the prevalent setting where the number of features and samples are both large and comparable. In this context, state-of-the-art debiasing technology uses a degrees-of-freedom correction to remove shrinkage bias of regularized estimators and conduct inference. However, this method requires that the observed samples are i.i.d., the covariates follow a mean zero Gaussian distribution, and reliable covariance matrix estimates for observed features are available. This approach struggles when (i) covariates are non-Gaussian with heavy tails or asymmetric distributions, (ii) rows of the design exhibit heterogeneity or dependencies, and (iii) reliable feature covariance estimates are lacking.  To address these, we develop a new strategy where the debiasing correction is a rescaled gradient descent step (suitably
    
[^11]: 可计算的稳定性对于持续性秩函数机器学习的影响

    Computable Stability for Persistence Rank Function Machine Learning. (arXiv:2307.02904v1 [math.AT])

    [http://arxiv.org/abs/2307.02904](http://arxiv.org/abs/2307.02904)

    本文讨论了持续性同调秩函数在拓扑数据分析中的应用，并指出秩函数相对于条码的优势在于更易于计算，并且可以直接应用函数数据分析的方法。然而，秩函数的稳定性问题尚待解决。

    

    持续性同调条码和图是拓扑数据分析的基础。它们在许多真实数据环境中得到广泛应用，将拓扑信息的变化（通过细胞同调测量）与数据的变化相关联，然而，由于其复杂的几何结构，它们在统计环境中的使用具有挑战性。在本文中，我们重新审视了持续性同调秩函数——一种衡量“形状”的不变量，它在条码和持续性图之前被引入，并以更适合数据和计算的形式捕捉相同的信息。尤其是，由于它们是函数，当持续性同调以秩函数形式表示时，可以直接应用函数数据分析领域的技术，这是一种适用于函数的统计学领域。然而，与条码相比，秩函数的受欢迎程度较低，因为它们面临着稳定性的挑战——这是验证它们在数据分析中使用的关键性质，而这种稳定性很难保证。

    Persistent homology barcodes and diagrams are a cornerstone of topological data analysis. Widely used in many real data settings, they relate variation in topological information (as measured by cellular homology) with variation in data, however, they are challenging to use in statistical settings due to their complex geometric structure. In this paper, we revisit the persistent homology rank function -- an invariant measure of ``shape" that was introduced before barcodes and persistence diagrams and captures the same information in a form that is more amenable to data and computation. In particular, since they are functions, techniques from functional data analysis -- a domain of statistics adapted for functions -- apply directly to persistent homology when represented by rank functions. Rank functions, however, have been less popular than barcodes because they face the challenge that stability -- a property that is crucial to validate their use in data analysis -- is difficult to gua
    
[^12]: 噪声稳定优化对于具有最优收敛率的平坦极小值的影响

    Noise Stability Optimization for Flat Minima with Optimal Convergence Rates. (arXiv:2306.08553v1 [cs.LG])

    [http://arxiv.org/abs/2306.08553](http://arxiv.org/abs/2306.08553)

    本文提出了一个SGD-like算法，注入随机噪声并利用分布对称性来减少方差，以寻找具有低海森矩阵迹的平坦极小值，同时提供了收敛速率分析。

    

    本文研究通过加入加权扰动来找到平坦的极小值。给定一个非凸函数$f:\mathbb{R}^d\rightarrow \mathbb{R}$和一个$d$维分布$\mathcal{P}$，我们扰动$f$的权重，并定义$F(W)=\mathbb{E}[f({W+U})]$，其中$U$是一个从$\mathcal{P}$中随机抽取的样本。这个过程通过$f$的海森矩阵的迹来诱导正则化，以适应于小的、各向同性的高斯扰动。因此，加权扰动的函数偏向于带有低海森矩阵迹的极小值。本文提出了一种类似于SGD的算法，在计算梯度之前注入随机噪声，同时利用$\mathcal{P}$的对称性来减少方差。我们还提供了严格的分析，证明了...

    We consider finding flat, local minimizers by adding average weight perturbations. Given a nonconvex function $f: \mathbb{R}^d \rightarrow \mathbb{R}$ and a $d$-dimensional distribution $\mathcal{P}$ which is symmetric at zero, we perturb the weight of $f$ and define $F(W) = \mathbb{E}[f({W + U})]$, where $U$ is a random sample from $\mathcal{P}$. This injection induces regularization through the Hessian trace of $f$ for small, isotropic Gaussian perturbations. Thus, the weight-perturbed function biases to minimizers with low Hessian trace. Several prior works have studied settings related to this weight-perturbed function by designing algorithms to improve generalization. Still, convergence rates are not known for finding minima under the average perturbations of the function $F$. This paper considers an SGD-like algorithm that injects random noise before computing gradients while leveraging the symmetry of $\mathcal{P}$ to reduce variance. We then provide a rigorous analysis, showing
    
[^13]: 批次使高维超参数线性回归的最小规范风险稳定

    Batches Stabilize the Minimum Norm Risk in High Dimensional Overparameterized Linear Regression. (arXiv:2306.08432v1 [cs.LG])

    [http://arxiv.org/abs/2306.08432](http://arxiv.org/abs/2306.08432)

    本文研究了将数据分成批次的学习算法，在高维超参数线性回归模型中提供了隐式正则化，通过适当的批量大小选择，稳定了风险行为，消除了插值点处的膨胀和双峰现象

    

    将数据分成批次的学习算法在许多机器学习应用中很常见，通常在计算效率和性能之间提供有用的权衡。本文通过具有各向同性高斯特征的最小规范超参数线性回归模型的视角来研究批量分区的好处。我们建议最小规范估计量的自然小批量版本，并推导出其二次风险的上界，表明其与噪声水平以及过度参数化比例成反比，对于最佳批量大小的选择。与最小规范相比，我们的估计器具有稳定的风险行为，其在过度参数化比例上单调递增，消除了插值点处的膨胀和双峰现象。有趣的是，我们观察到批处理所提供的隐式正则化在一定程度上可以通过特征重叠来解释。

    Learning algorithms that divide the data into batches are prevalent in many machine-learning applications, typically offering useful trade-offs between computational efficiency and performance. In this paper, we examine the benefits of batch-partitioning through the lens of a minimum-norm overparameterized linear regression model with isotropic Gaussian features. We suggest a natural small-batch version of the minimum-norm estimator, and derive an upper bound on its quadratic risk, showing it is inversely proportional to the noise level as well as to the overparameterization ratio, for the optimal choice of batch size. In contrast to minimum-norm, our estimator admits a stable risk behavior that is monotonically increasing in the overparameterization ratio, eliminating both the blowup at the interpolation point and the double-descent phenomenon. Interestingly, we observe that this implicit regularization offered by the batch partition is partially explained by feature overlap between t
    
[^14]: repliclust：聚类分析的合成数据

    repliclust: Synthetic Data for Cluster Analysis. (arXiv:2303.14301v1 [cs.LG])

    [http://arxiv.org/abs/2303.14301](http://arxiv.org/abs/2303.14301)

    repliclust 是一个 Python 包，用于生成具有聚类的合成数据集，基于数据集的原型，提供了放置集群中心、采样集群形状、选择每个集群的数据点数量以及为集群分配概率分布的算法。

    

    我们介绍了 repliclust（来自于 repli-cate 和 clust-er），这是一个用于生成具有聚类的合成数据集的 Python 包。我们的方法基于数据集的原型，即高级几何描述，用户可以从中创建许多不同的数据集，并具有所需的几何特性。我们软件的架构是模块化和面向对象的，将数据生成分解成放置集群中心的算法、采样集群形状的算法、选择每个集群的数据点数量的算法以及为集群分配概率分布的算法。repliclust.org 项目网页提供了简明的用户指南和全面的文档。

    We present repliclust (from repli-cate and clust-er), a Python package for generating synthetic data sets with clusters. Our approach is based on data set archetypes, high-level geometric descriptions from which the user can create many different data sets, each possessing the desired geometric characteristics. The architecture of our software is modular and object-oriented, decomposing data generation into algorithms for placing cluster centers, sampling cluster shapes, selecting the number of data points for each cluster, and assigning probability distributions to clusters. The project webpage, repliclust.org, provides a concise user guide and thorough documentation.
    
[^15]: 序列实验的反事实推断

    Counterfactual inference for sequential experiments. (arXiv:2202.06891v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2202.06891](http://arxiv.org/abs/2202.06891)

    本文针对序列实验的反事实推断问题，提出了一个潜在因子模型，使用非参数方法对反事实均值进行估计，并建立了误差界限。

    

    我们考虑针对连续设计实验进行的事后统计推断，在此实验中，多个单位在多个时间点上分配治疗，并使用随时间而适应的治疗策略。我们的目标是在对适应性治疗策略做出最少的假设的情况下，为最小可能规模的反事实均值提供推断保证，即在每个单位和每个时间下，针对不同治疗的平均结果。在没有对反事实均值进行任何结构性假设的情况下，这项具有挑战性的任务是不可行的，因为未知变量比观察到的数据点还多。为了取得进展，我们引入了一个潜在因子模型用于反事实均值上，该模型作为非参数形式的非线性混合效应模型和以前工作中考虑的双线性潜在因子模型的推广。我们使用非参数方法进行估计，即最近邻的变体，并为每个单位和每个时间的反事实均值建立了非渐进高概率误差界限。

    We consider after-study statistical inference for sequentially designed experiments wherein multiple units are assigned treatments for multiple time points using treatment policies that adapt over time. Our goal is to provide inference guarantees for the counterfactual mean at the smallest possible scale -- mean outcome under different treatments for each unit and each time -- with minimal assumptions on the adaptive treatment policy. Without any structural assumptions on the counterfactual means, this challenging task is infeasible due to more unknowns than observed data points. To make progress, we introduce a latent factor model over the counterfactual means that serves as a non-parametric generalization of the non-linear mixed effects model and the bilinear latent factor model considered in prior works. For estimation, we use a non-parametric method, namely a variant of nearest neighbors, and establish a non-asymptotic high probability error bound for the counterfactual mean for ea
    

