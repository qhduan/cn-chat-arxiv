# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hierarchy of the echo state property in quantum reservoir computing](https://arxiv.org/abs/2403.02686) | 介绍了在量子储备计算中回声态性质的不同层次，包括非平稳性ESP和子系统具有ESP的子空间/子集ESP。进行了数值演示和记忆容量计算以验证这些定义。 |
| [^2] | [Low-Rank Bandits via Tight Two-to-Infinity Singular Subspace Recovery](https://arxiv.org/abs/2402.15739) | 该论文介绍了一种解决低秩环境中具有上下文信息的赌徒问题的高效算法，其中包括策略评估、最佳策略识别和遗憾最小化，并且在最佳策略识别和策略评估方面的算法几乎是极小极大最优的。 |
| [^3] | [LoRA+: Efficient Low Rank Adaptation of Large Models](https://arxiv.org/abs/2402.12354) | LoRA+通过设置不同的学习率来改进原始LoRA的低效率问题，在保持计算成本不变的情况下提高了模型性能和微调速度。 |
| [^4] | [Sampling from the Mean-Field Stationary Distribution](https://arxiv.org/abs/2402.07355) | 本文研究了从均场随机微分方程 (SDE) 的稳态分布中采样的复杂性，并提出了一种解耦的方法。该方法能够在多种情况下提供改进的保证，包括在均场区域优化某些双层神经网络的更好保证。 |
| [^5] | [On Calibration and Conformal Prediction of Deep Classifiers](https://arxiv.org/abs/2402.05806) | 本文研究了温度缩放对符合预测方法的影响，通过实证研究发现，校准对自适应C方法产生了有害的影响。 |
| [^6] | [Online estimation of the inverse of the Hessian for stochastic optimization with application to universal stochastic Newton algorithms.](http://arxiv.org/abs/2401.10923) | 该论文提出了一种在线估计Hessian矩阵逆的方法，利用Robbins-Monro过程，能够 drastical reducescomputational complexity,并发展了通用的随机牛顿算法，研究了所提方法的渐进效率。 |
| [^7] | [A Survey on Statistical Theory of Deep Learning: Approximation, Training Dynamics, and Generative Models.](http://arxiv.org/abs/2401.07187) | 该论文综述了深度学习的统计理论，包括近似方法、训练动态和生成模型。在非参数框架中，结果揭示了神经网络过度风险的快速收敛速率，以及如何通过梯度方法训练网络以找到良好的泛化解决方案。 |
| [^8] | [Convergence of flow-based generative models via proximal gradient descent in Wasserstein space.](http://arxiv.org/abs/2310.17582) | 本文通过在Wasserstein空间中应用近端梯度下降，证明了基于流的生成模型的收敛性，并提供了生成数据分布的理论保证。 |
| [^9] | [Implicit regularization of deep residual networks towards neural ODEs.](http://arxiv.org/abs/2309.01213) | 本文建立了深度残差网络向神经常微分方程的隐式正则化，通过对用梯度流训练的非线性网络的研究，证明了在网络以神经常微分方程的离散化形式初始化后，这种离散化将在整个训练过程中保持不变，并提供了收敛性的条件。 |
| [^10] | [Gotta match 'em all: Solution diversification in graph matching matched filters.](http://arxiv.org/abs/2308.13451) | 本文提出了一种在大规模背景图中查找多个嵌入的模板图的新方法，通过迭代惩罚相似度矩阵来实现多样化匹配的发现，并提出了算法加速措施。在理论验证和实验证明中，证明了该方法的可行性和实用性。 |
| [^11] | [Engression: Extrapolation for Nonlinear Regression?.](http://arxiv.org/abs/2307.00835) | Engression是一种非线性回归方法，通过使用分布回归技术和预加性噪声模型，在训练样本范围边界外也能可靠地进行外推。 |
| [^12] | [High-dimensional variable clustering based on sub-asymptotic maxima of a weakly dependent random process.](http://arxiv.org/abs/2302.00934) | 我们提出了一种基于亚渐近极大值的高维变量聚类模型，该模型利用群集间多变量随机过程的极大值的独立性定义种群水平的群集，我们还开发了一种无需预先指定群集数量的算法来恢复变量的群集。该算法在特定条件下能够有效地识别数据中的群集，并能够以多项式复杂度进行计算。我们的工作对于理解依赖过程的块最大值的非参数学习有重要意义，并且在神经科学领域有着应用潜力。 |

# 详细

[^1]: 量子储备计算中的回声态性质等级

    Hierarchy of the echo state property in quantum reservoir computing

    [https://arxiv.org/abs/2403.02686](https://arxiv.org/abs/2403.02686)

    介绍了在量子储备计算中回声态性质的不同层次，包括非平稳性ESP和子系统具有ESP的子空间/子集ESP。进行了数值演示和记忆容量计算以验证这些定义。

    

    回声态性质（ESP）代表了储备计算（RC）框架中的一个基本概念，通过对初始状态和远期输入不加歧视来确保储蓄网络的仅输出训练。然而，传统的ESP定义并未描述可能演变统计属性的非平稳系统。为解决这一问题，我们引入了两类新的ESP：\textit{非平稳ESP}，用于潜在非平稳系统，和\textit{子空间/子集ESP}，适用于具有ESP的子系统的系统。根据这些定义，我们在量子储备计算（QRC）框架中数值演示了非平稳ESP与典型哈密顿动力学和使用非线性自回归移动平均（NARMA）任务的输入编码方法之间的对应关系。我们还通过计算线性/非线性记忆容量来确认这种对应关系，以量化

    arXiv:2403.02686v1 Announce Type: cross  Abstract: The echo state property (ESP) represents a fundamental concept in the reservoir computing (RC) framework that ensures output-only training of reservoir networks by being agnostic to the initial states and far past inputs. However, the traditional definition of ESP does not describe possible non-stationary systems in which statistical properties evolve. To address this issue, we introduce two new categories of ESP: \textit{non-stationary ESP}, designed for potentially non-stationary systems, and \textit{subspace/subset ESP}, designed for systems whose subsystems have ESP. Following the definitions, we numerically demonstrate the correspondence between non-stationary ESP in the quantum reservoir computer (QRC) framework with typical Hamiltonian dynamics and input encoding methods using non-linear autoregressive moving-average (NARMA) tasks. We also confirm the correspondence by computing linear/non-linear memory capacities that quantify 
    
[^2]: 低秩赌徒通过紧绝对到无穷奇异子空间恢复

    Low-Rank Bandits via Tight Two-to-Infinity Singular Subspace Recovery

    [https://arxiv.org/abs/2402.15739](https://arxiv.org/abs/2402.15739)

    该论文介绍了一种解决低秩环境中具有上下文信息的赌徒问题的高效算法，其中包括策略评估、最佳策略识别和遗憾最小化，并且在最佳策略识别和策略评估方面的算法几乎是极小极大最优的。

    

    我们研究了具有低秩结构的情境赌徒问题，在每一轮中，如果选择了(情境，动作)对$(i,j)\in [m]\times [n]$，学习者会观察一个未知低秩奖励矩阵的$(i,j)$-th入口的嘈杂样本。连续的情境以独立同分布的方式随机生成并透露给学习者。对于这样的赌徒问题，我们提出了高效的算法用于策略评估、最佳策略识别和遗憾最小化。对于策略评估和最佳策略识别，我们展示了我们的算法几乎是极小极大最优的。例如，为了以至少$1-\delta$的概率返回一个$\varepsilon$-最佳策略，通常需要的样本数大致按照${m+n\over \varepsilon^2}\log(1/\delta)$来衡量。我们的遗憾最小化算法享有的极小极大保证按照$r^{7/4}(m+n)^{3/4}\sqrt{T}$缩放，这优于现有算法。所有提出的算法包括两个阶段：

    arXiv:2402.15739v1 Announce Type: new  Abstract: We study contextual bandits with low-rank structure where, in each round, if the (context, arm) pair $(i,j)\in [m]\times [n]$ is selected, the learner observes a noisy sample of the $(i,j)$-th entry of an unknown low-rank reward matrix. Successive contexts are generated randomly in an i.i.d. manner and are revealed to the learner. For such bandits, we present efficient algorithms for policy evaluation, best policy identification and regret minimization. For policy evaluation and best policy identification, we show that our algorithms are nearly minimax optimal. For instance, the number of samples required to return an $\varepsilon$-optimal policy with probability at least $1-\delta$ typically scales as ${m+n\over \varepsilon^2}\log(1/\delta)$. Our regret minimization algorithm enjoys minimax guarantees scaling as $r^{7/4}(m+n)^{3/4}\sqrt{T}$, which improves over existing algorithms. All the proposed algorithms consist of two phases: they
    
[^3]: LoRA+: 大规模模型的高效低秩适应性

    LoRA+: Efficient Low Rank Adaptation of Large Models

    [https://arxiv.org/abs/2402.12354](https://arxiv.org/abs/2402.12354)

    LoRA+通过设置不同的学习率来改进原始LoRA的低效率问题，在保持计算成本不变的情况下提高了模型性能和微调速度。

    

    在这篇论文中，我们展示了低秩适应（LoRA）最初由胡等人（2021年）引入，导致对具有大宽度（嵌入维度）的模型进行微调时表现亚优。这是因为LoRA中的适配器矩阵A和B使用相同的学习率进行更新。通过对大宽度网络进行缩放参数的论证，我们展示了对适配器矩阵A和B使用相同的学习率不利于有效的特征学习。然后，我们表明LoRA的这种次优性可以简单地通过为LoRA适配器矩阵A和B设置不同的学习率以及一个精心选择的比率来进行校正。我们将这个提出的算法称为LoRA$+$。在我们广泛的实验证明中，LoRA$+$在相同计算成本下提高了性能（1-2％的改进）和微调速度（最多提速约2倍）。

    arXiv:2402.12354v1 Announce Type: cross  Abstract: In this paper, we show that Low Rank Adaptation (LoRA) as originally introduced in Hu et al. (2021) leads to suboptimal finetuning of models with large width (embedding dimension). This is due to the fact that adapter matrices A and B in LoRA are updated with the same learning rate. Using scaling arguments for large width networks, we demonstrate that using the same learning rate for A and B does not allow efficient feature learning. We then show that this suboptimality of LoRA can be corrected simply by setting different learning rates for the LoRA adapter matrices A and B with a well-chosen ratio. We call this proposed algorithm LoRA$+$. In our extensive experiments, LoRA$+$ improves performance (1-2 $\%$ improvements) and finetuning speed (up to $\sim$ 2X SpeedUp), at the same computational cost as LoRA.
    
[^4]: 从均场稳态分布中采样

    Sampling from the Mean-Field Stationary Distribution

    [https://arxiv.org/abs/2402.07355](https://arxiv.org/abs/2402.07355)

    本文研究了从均场随机微分方程 (SDE) 的稳态分布中采样的复杂性，并提出了一种解耦的方法。该方法能够在多种情况下提供改进的保证，包括在均场区域优化某些双层神经网络的更好保证。

    

    我们研究了从均场随机微分方程 (SDE) 的稳态分布中采样的复杂性，或者等价地，即包含交互项的概率测度空间上的最小化函数的复杂性。我们的主要洞察是将这个问题的两个关键方面解耦：(1) 通过有限粒子系统逼近均场SDE，通过时间均匀传播混沌，和(2) 通过标准对数凹抽样器从有限粒子稳态分布中采样。我们的方法在概念上更简单，其灵活性允许结合用于算法和理论的最新技术。这导致在许多设置中提供了改进的保证，包括在均场区域优化某些双层神经网络的更好保证。

    We study the complexity of sampling from the stationary distribution of a mean-field SDE, or equivalently, the complexity of minimizing a functional over the space of probability measures which includes an interaction term.   Our main insight is to decouple the two key aspects of this problem: (1) approximation of the mean-field SDE via a finite-particle system, via uniform-in-time propagation of chaos, and (2) sampling from the finite-particle stationary distribution, via standard log-concave samplers. Our approach is conceptually simpler and its flexibility allows for incorporating the state-of-the-art for both algorithms and theory. This leads to improved guarantees in numerous settings, including better guarantees for optimizing certain two-layer neural networks in the mean-field regime.
    
[^5]: 关于深度分类器的校准和符合预测研究

    On Calibration and Conformal Prediction of Deep Classifiers

    [https://arxiv.org/abs/2402.05806](https://arxiv.org/abs/2402.05806)

    本文研究了温度缩放对符合预测方法的影响，通过实证研究发现，校准对自适应C方法产生了有害的影响。

    

    在许多分类应用中，深度神经网络（DNN）基于分类器的预测需要伴随一些置信度指示。针对这个目标，有两种流行的后处理方法：1）校准：修改分类器的softmax值，使其最大值（与预测相关）更好地估计正确概率；和2）符合预测（CP）：设计一个基于softmax值的分数，从中产生一组预测，具有理论上保证正确类别边际覆盖的特性。尽管在实践中两种指示都可能是需要的，但到目前为止它们之间的相互作用尚未得到研究。为了填补这一空白，在本文中，我们研究了温度缩放，这是最常见的校准技术，对重要的CP方法的影响。我们首先进行了一项广泛的实证研究，其中显示了一些重要的洞察，其中包括令人惊讶的发现，即校准对流行的自适应C方法产生了有害的影响。

    In many classification applications, the prediction of a deep neural network (DNN) based classifier needs to be accompanied with some confidence indication. Two popular post-processing approaches for that aim are: 1) calibration: modifying the classifier's softmax values such that their maximum (associated with the prediction) better estimates the correctness probability; and 2) conformal prediction (CP): devising a score (based on the softmax values) from which a set of predictions with theoretically guaranteed marginal coverage of the correct class is produced. While in practice both types of indications can be desired, so far the interplay between them has not been investigated. Toward filling this gap, in this paper we study the effect of temperature scaling, arguably the most common calibration technique, on prominent CP methods. We start with an extensive empirical study that among other insights shows that, surprisingly, calibration has a detrimental effect on popular adaptive C
    
[^6]: 在具有通用随机牛顿算法应用于随机优化的情况下，关于Hessian矩阵的逆的在线估计

    Online estimation of the inverse of the Hessian for stochastic optimization with application to universal stochastic Newton algorithms. (arXiv:2401.10923v1 [math.OC])

    [http://arxiv.org/abs/2401.10923](http://arxiv.org/abs/2401.10923)

    该论文提出了一种在线估计Hessian矩阵逆的方法，利用Robbins-Monro过程，能够 drastical reducescomputational complexity,并发展了通用的随机牛顿算法，研究了所提方法的渐进效率。

    

    本文针对以期望形式表示的凸函数的次优随机优化问题，提出了一种直接递归估计逆Hessian矩阵的技术，采用了Robbins-Monro过程。该方法能够大幅降低计算复杂性，并且允许开发通用的随机牛顿方法，并研究所提方法的渐近效率。这项工作扩展了在随机优化中二阶算法的应用范围。

    This paper addresses second-order stochastic optimization for estimating the minimizer of a convex function written as an expectation. A direct recursive estimation technique for the inverse Hessian matrix using a Robbins-Monro procedure is introduced. This approach enables to drastically reduces computational complexity. Above all, it allows to develop universal stochastic Newton methods and investigate the asymptotic efficiency of the proposed approach. This work so expands the application scope of secondorder algorithms in stochastic optimization.
    
[^7]: 深度学习的统计理论综述：近似，训练动态和生成模型

    A Survey on Statistical Theory of Deep Learning: Approximation, Training Dynamics, and Generative Models. (arXiv:2401.07187v1 [stat.ML])

    [http://arxiv.org/abs/2401.07187](http://arxiv.org/abs/2401.07187)

    该论文综述了深度学习的统计理论，包括近似方法、训练动态和生成模型。在非参数框架中，结果揭示了神经网络过度风险的快速收敛速率，以及如何通过梯度方法训练网络以找到良好的泛化解决方案。

    

    在这篇文章中，我们从三个角度回顾了关于神经网络统计理论的文献。第一部分回顾了在回归或分类的非参数框架下关于神经网络过度风险的结果。这些结果依赖于神经网络的显式构造，以及采用了近似理论的工具，导致过度风险的快速收敛速率。通过这些构造，可以用样本大小、数据维度和函数平滑性来表达网络的宽度和深度。然而，他们的基本分析仅适用于深度神经网络高度非凸的全局极小值点。这促使我们在第二部分回顾神经网络的训练动态。具体而言，我们回顾了那些试图回答“基于梯度方法训练的神经网络如何找到能够在未见数据上有良好泛化性能的解”的论文。尤其是两个知名的

    In this article, we review the literature on statistical theories of neural networks from three perspectives. In the first part, results on excess risks for neural networks are reviewed in the nonparametric framework of regression or classification. These results rely on explicit constructions of neural networks, leading to fast convergence rates of excess risks, in that tools from the approximation theory are adopted. Through these constructions, the width and depth of the networks can be expressed in terms of sample size, data dimension, and function smoothness. Nonetheless, their underlying analysis only applies to the global minimizer in the highly non-convex landscape of deep neural networks. This motivates us to review the training dynamics of neural networks in the second part. Specifically, we review papers that attempt to answer ``how the neural network trained via gradient-based methods finds the solution that can generalize well on unseen data.'' In particular, two well-know
    
[^8]: 在Wasserstein空间中通过近端梯度下降实现基于流的生成模型的收敛性

    Convergence of flow-based generative models via proximal gradient descent in Wasserstein space. (arXiv:2310.17582v1 [stat.ML])

    [http://arxiv.org/abs/2310.17582](http://arxiv.org/abs/2310.17582)

    本文通过在Wasserstein空间中应用近端梯度下降，证明了基于流的生成模型的收敛性，并提供了生成数据分布的理论保证。

    

    基于流的生成模型在计算数据生成和似然函数方面具有一定的优势，并且最近在实证表现上显示出竞争力。与相关基于分数扩散模型的积累理论研究相比，对于在正向（数据到噪声）和反向（噪声到数据）方向上都是确定性的流模型的分析还很少。本文通过在归一化流网络中实施Jordan-Kinderleherer-Otto（JKO）方案的所谓JKO流模型，提供了通过渐进流模型生成数据分布的理论保证。利用Wasserstein空间中近端梯度下降（GD）的指数收敛性，我们证明了通过JKO流模型生成数据的Kullback-Leibler（KL）保证为$O(\varepsilon^2)$，其中使用$N \lesssim \log (1/\varepsilon)$个JKO步骤（流中的$N$个残差块），其中$\varepsilon$是每步一阶条件的误差。

    Flow-based generative models enjoy certain advantages in computing the data generation and the likelihood, and have recently shown competitive empirical performance. Compared to the accumulating theoretical studies on related score-based diffusion models, analysis of flow-based models, which are deterministic in both forward (data-to-noise) and reverse (noise-to-data) directions, remain sparse. In this paper, we provide a theoretical guarantee of generating data distribution by a progressive flow model, the so-called JKO flow model, which implements the Jordan-Kinderleherer-Otto (JKO) scheme in a normalizing flow network. Leveraging the exponential convergence of the proximal gradient descent (GD) in Wasserstein space, we prove the Kullback-Leibler (KL) guarantee of data generation by a JKO flow model to be $O(\varepsilon^2)$ when using $N \lesssim \log (1/\varepsilon)$ many JKO steps ($N$ Residual Blocks in the flow) where $\varepsilon $ is the error in the per-step first-order condit
    
[^9]: 深度残差网络的隐式正则化与神经常微分方程的关联

    Implicit regularization of deep residual networks towards neural ODEs. (arXiv:2309.01213v1 [stat.ML])

    [http://arxiv.org/abs/2309.01213](http://arxiv.org/abs/2309.01213)

    本文建立了深度残差网络向神经常微分方程的隐式正则化，通过对用梯度流训练的非线性网络的研究，证明了在网络以神经常微分方程的离散化形式初始化后，这种离散化将在整个训练过程中保持不变，并提供了收敛性的条件。

    

    残差神经网络是先进的深度学习模型。它们的连续深度模拟称为神经常微分方程（ODE），也被广泛使用。尽管它们取得了成功，但离散模型与连续模型之间的联系仍缺乏坚实的数学基础。在本文中，我们通过建立一个针对用梯度流训练的非线性网络的深度残差网络向神经常微分方程的隐式正则化来朝着这个方向迈出了一步。我们证明，如果网络的初始化是神经常微分方程的离散化，则这种离散化在整个训练过程中保持不变。我们的结果对于有限的训练时间和训练时间趋于无穷大都成立，只要网络满足Polyak-Lojasiewicz条件。重要的是，这个条件适用于一个残差网络家族，其中残差是两层感知机，在宽度上只是线性超参数化，并且暗示了梯度流的收敛性。

    Residual neural networks are state-of-the-art deep learning models. Their continuous-depth analog, neural ordinary differential equations (ODEs), are also widely used. Despite their success, the link between the discrete and continuous models still lacks a solid mathematical foundation. In this article, we take a step in this direction by establishing an implicit regularization of deep residual networks towards neural ODEs, for nonlinear networks trained with gradient flow. We prove that if the network is initialized as a discretization of a neural ODE, then such a discretization holds throughout training. Our results are valid for a finite training time, and also as the training time tends to infinity provided that the network satisfies a Polyak-Lojasiewicz condition. Importantly, this condition holds for a family of residual networks where the residuals are two-layer perceptrons with an overparameterization in width that is only linear, and implies the convergence of gradient flow to
    
[^10]: 抓住它们：图匹配匹配滤波中的解决方案多样化

    Gotta match 'em all: Solution diversification in graph matching matched filters. (arXiv:2308.13451v1 [stat.ML])

    [http://arxiv.org/abs/2308.13451](http://arxiv.org/abs/2308.13451)

    本文提出了一种在大规模背景图中查找多个嵌入的模板图的新方法，通过迭代惩罚相似度矩阵来实现多样化匹配的发现，并提出了算法加速措施。在理论验证和实验证明中，证明了该方法的可行性和实用性。

    

    我们提出了一种在非常大的背景图中查找多个嵌入在其中的模板图的新方法。我们的方法基于Sussman等人提出的图匹配匹配滤波技术，通过在匹配滤波算法中迭代地惩罚合适的节点对相似度矩阵来实现多样化匹配的发现。此外，我们提出了算法加速，极大地提高了我们的匹配滤波方法的可扩展性。我们在相关的Erdos-Renyi图设置中对我们的方法进行了理论上的验证，显示其在温和的模型条件下能够顺序地发现多个模板。我们还通过使用模拟模型和真实世界数据集（包括人脑连接组和大型交易知识库）进行了大量实验证明了我们方法的实用性。

    We present a novel approach for finding multiple noisily embedded template graphs in a very large background graph. Our method builds upon the graph-matching-matched-filter technique proposed in Sussman et al., with the discovery of multiple diverse matchings being achieved by iteratively penalizing a suitable node-pair similarity matrix in the matched filter algorithm. In addition, we propose algorithmic speed-ups that greatly enhance the scalability of our matched-filter approach. We present theoretical justification of our methodology in the setting of correlated Erdos-Renyi graphs, showing its ability to sequentially discover multiple templates under mild model conditions. We additionally demonstrate our method's utility via extensive experiments both using simulated models and real-world dataset, include human brain connectomes and a large transactional knowledge base.
    
[^11]: Engression: 非线性回归的外推方法

    Engression: Extrapolation for Nonlinear Regression?. (arXiv:2307.00835v1 [stat.ME])

    [http://arxiv.org/abs/2307.00835](http://arxiv.org/abs/2307.00835)

    Engression是一种非线性回归方法，通过使用分布回归技术和预加性噪声模型，在训练样本范围边界外也能可靠地进行外推。

    

    外推对于许多统计学和机器学习应用至关重要，因为常常会遇到超出训练样本范围的测试数据。然而，对于非线性模型来说，外推是一个巨大的挑战。传统模型在这方面通常遇到困难：树集成模型在支持范围外提供连续的预测，而神经网络的预测往往变得不可控。这项工作旨在提供一种非线性回归方法，其可靠性在训练样本范围边界不会立即崩溃。我们的主要贡献是一种名为“engression”的新方法，它是一种预加性噪声模型的分布回归技术，其中噪声添加到协变量上并应用非线性转换。我们的实验结果表明，该模型通常适用于许多真实数据集。我们展示engression可以在一些假设下成功进行外推，例如严格限制噪声大小。

    Extrapolation is crucial in many statistical and machine learning applications, as it is common to encounter test data outside the training support. However, extrapolation is a considerable challenge for nonlinear models. Conventional models typically struggle in this regard: while tree ensembles provide a constant prediction beyond the support, neural network predictions tend to become uncontrollable. This work aims at providing a nonlinear regression methodology whose reliability does not break down immediately at the boundary of the training support. Our primary contribution is a new method called `engression' which, at its core, is a distributional regression technique for pre-additive noise models, where the noise is added to the covariates before applying a nonlinear transformation. Our experimental results indicate that this model is typically suitable for many real data sets. We show that engression can successfully perform extrapolation under some assumptions such as a strictl
    
[^12]: 基于弱相关随机过程的亚渐近极大值的高维变量聚类

    High-dimensional variable clustering based on sub-asymptotic maxima of a weakly dependent random process. (arXiv:2302.00934v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2302.00934](http://arxiv.org/abs/2302.00934)

    我们提出了一种基于亚渐近极大值的高维变量聚类模型，该模型利用群集间多变量随机过程的极大值的独立性定义种群水平的群集，我们还开发了一种无需预先指定群集数量的算法来恢复变量的群集。该算法在特定条件下能够有效地识别数据中的群集，并能够以多项式复杂度进行计算。我们的工作对于理解依赖过程的块最大值的非参数学习有重要意义，并且在神经科学领域有着应用潜力。

    

    我们提出了一种新的变量聚类模型，称为渐近独立块 (AI-block) 模型，该模型基于群集间多变量平稳混合随机过程的极大值的独立性来定义种群水平的群集。该模型类是可识别的，意味着存在一种偏序关系，允许进行统计推断。我们还提出了一种算法，无需事先指定群集的数量即可恢复变量的群集。我们的工作提供了一些理论洞察，证明了在某些条件下，我们的算法能够在计算复杂性在维度中是多项式的情况下有效地识别数据中的群集。这意味着可以非参数地学习出仅仅是亚渐近的依赖过程的块最大值的群组。为了进一步说明我们的工作的重要性，我们将我们的方法应用于神经科学数据集。

    We propose a new class of models for variable clustering called Asymptotic Independent block (AI-block) models, which defines population-level clusters based on the independence of the maxima of a multivariate stationary mixing random process among clusters. This class of models is identifiable, meaning that there exists a maximal element with a partial order between partitions, allowing for statistical inference. We also present an algorithm for recovering the clusters of variables without specifying the number of clusters \emph{a priori}. Our work provides some theoretical insights into the consistency of our algorithm, demonstrating that under certain conditions it can effectively identify clusters in the data with a computational complexity that is polynomial in the dimension. This implies that groups can be learned nonparametrically in which block maxima of a dependent process are only sub-asymptotic. To further illustrate the significance of our work, we applied our method to neu
    

