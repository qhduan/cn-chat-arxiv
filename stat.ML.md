# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Taming the Interactive Particle Langevin Algorithm -- the superlinear case](https://arxiv.org/abs/2403.19587) | 我们提出了一种新颖的 tamed interactive particle Langevin algorithms（tIPLA）类算法，能够在多项式增长情况下获得稳定且具有最佳速率的非渐近收敛误差估计。 |
| [^2] | [A Practical Guide to Statistical Distances for Evaluating Generative Models in Science](https://arxiv.org/abs/2403.12636) | 本文提供了一种实用指南，介绍了四种常用统计距离的概念，以帮助评估生成模型，无需高深的数学和统计知识。 |
| [^3] | [Stochastic Extragradient with Random Reshuffling: Improved Convergence for Variational Inequalities](https://arxiv.org/abs/2403.07148) | 该论文针对三类变分不等式问题提出了具有随机重排的随机外推法（SEG-RR），并证明其在单调情况下实现了比均匀替换采样SEG更快的收敛速度。 |
| [^4] | [Convergence of Gradient Descent for Recurrent Neural Networks: A Nonasymptotic Analysis](https://arxiv.org/abs/2402.12241) | 该论文分析了在动态系统中利用梯度下降进行监督学习的递归神经网络的性能，并证明在不需要海量过参数化的情况下，梯度下降可以达到最优性。 |
| [^5] | [Learning from higher-order statistics, efficiently: hypothesis tests, random features, and neural networks](https://arxiv.org/abs/2312.14922) | 神经网络在高维数据中发现统计模式，研究了如何高效地从高阶累积量中提取特征，并探讨了在尖峰累积量模型中的统计和计算限制。 |
| [^6] | [Adversarial Bandits against Arbitrary Strategies](https://arxiv.org/abs/2205.14839) | 该论文研究了对抗性赌博机问题，针对任意策略，提出了使用在线镜像下降方法的主控基础框架，并使用自适应学习率的OMD来减轻方差的影响，取得了较好的结果。 |
| [^7] | [Interventional and Counterfactual Inference with Diffusion Models.](http://arxiv.org/abs/2302.00860) | 本论文提出了基于扩散模型的因果模型 (DCM)，它可以在只有观测数据和因果图可用的情况下进行干预和反事实推断，其具有较好的表现。同时，论文还提供了一种分析反事实估计的方法，可以应用于更广泛的场景。 |
| [^8] | [Contextual Linear Bandits under Noisy Features: Towards Bayesian Oracles.](http://arxiv.org/abs/1703.01347) | 本论文研究了具有噪声特征的上下文线性Bandit问题。我们提出了一个算法，通过观察信息，实现了贝叶斯神谕并得到了$\tilde{O}(d\sqrt{T})$的遗憾界。 |

# 详细

[^1]: 驯服交互式粒子 Langevin 算法 -- 超线性情况

    Taming the Interactive Particle Langevin Algorithm -- the superlinear case

    [https://arxiv.org/abs/2403.19587](https://arxiv.org/abs/2403.19587)

    我们提出了一种新颖的 tamed interactive particle Langevin algorithms（tIPLA）类算法，能够在多项式增长情况下获得稳定且具有最佳速率的非渐近收敛误差估计。

    

    最近在随机优化方面的进展产生了交互式粒子 Langevin 算法（IPLA），该算法利用交互粒子系统（IPS）的概念来高效地从近似后验密度中抽样。这在期望最大化（EM）框架中变得尤为关键，其中 E 步骤在计算上具有挑战性甚至是难以处理的。尽管先前的研究侧重于梯度最多线性增长的凸情况，我们的工作将此框架扩展到包括多项式增长。采用驯服技术生成明确的离散化方案，从而产生一类稳定的、在这种非线性情况下，称为驯服交互式粒子 Langevin 算法（tIPLA）的算法。我们获得了新类在 Wasserstein-2 距离下的非渐近收敛误差估计，具有最佳速率。

    arXiv:2403.19587v1 Announce Type: cross  Abstract: Recent advances in stochastic optimization have yielded the interactive particle Langevin algorithm (IPLA), which leverages the notion of interacting particle systems (IPS) to efficiently sample from approximate posterior densities. This becomes particularly crucial within the framework of Expectation-Maximization (EM), where the E-step is computationally challenging or even intractable. Although prior research has focused on scenarios involving convex cases with gradients of log densities that grow at most linearly, our work extends this framework to include polynomial growth. Taming techniques are employed to produce an explicit discretization scheme that yields a new class of stable, under such non-linearities, algorithms which are called tamed interactive particle Langevin algorithms (tIPLA). We obtain non-asymptotic convergence error estimates in Wasserstein-2 distance for the new class under an optimal rate.
    
[^2]: 用于科学中评估生成模型的统计距离的实用指南

    A Practical Guide to Statistical Distances for Evaluating Generative Models in Science

    [https://arxiv.org/abs/2403.12636](https://arxiv.org/abs/2403.12636)

    本文提供了一种实用指南，介绍了四种常用统计距离的概念，以帮助评估生成模型，无需高深的数学和统计知识。

    

    生成模型在许多科学领域中是非常宝贵的，因为它们能够捕捉高维和复杂的分布，例如逼真的图像、蛋白质结构和连接组。本研究旨在为理解流行的统计距离概念提供一个易于理解的入口点，只需要数学和统计学的基础知识。我们专注于代表不同方法论的四种常用统计距离概念：使用低维投影（Sliced-Wasserstein; SW)、使用分类器获取距离（Classifier Two-Sample Tests; C2ST)、通过核进行嵌入（Maximum Mean Discrepancy; MMD) 或神经网络（Fr\'echet Inception Distance; FID)。我们强调每个距离背后的直觉，并解释它们的优点、可伸缩性、复杂性和缺陷。

    arXiv:2403.12636v1 Announce Type: new  Abstract: Generative models are invaluable in many fields of science because of their ability to capture high-dimensional and complicated distributions, such as photo-realistic images, protein structures, and connectomes. How do we evaluate the samples these models generate? This work aims to provide an accessible entry point to understanding popular notions of statistical distances, requiring only foundational knowledge in mathematics and statistics. We focus on four commonly used notions of statistical distances representing different methodologies: Using low-dimensional projections (Sliced-Wasserstein; SW), obtaining a distance using classifiers (Classifier Two-Sample Tests; C2ST), using embeddings through kernels (Maximum Mean Discrepancy; MMD), or neural networks (Fr\'echet Inception Distance; FID). We highlight the intuition behind each distance and explain their merits, scalability, complexity, and pitfalls. To demonstrate how these distanc
    
[^3]: 具有随机重排的随机外推法：改进变分不等式的收敛性

    Stochastic Extragradient with Random Reshuffling: Improved Convergence for Variational Inequalities

    [https://arxiv.org/abs/2403.07148](https://arxiv.org/abs/2403.07148)

    该论文针对三类变分不等式问题提出了具有随机重排的随机外推法（SEG-RR），并证明其在单调情况下实现了比均匀替换采样SEG更快的收敛速度。

    

    随机外推法（SEG）方法是解决出现在各种机器学习任务中的有限求和极小-极大优化和变分不等式问题（VIPs）的最流行算法之一。然而，现有的SEG收敛分析专注于其带替换变体，而方法的实际实现会随机重新排列分量并按顺序使用它们。与广为研究的带替换变体不同，具有随机重排的SEG（SEG-RR）缺乏已建立的理论保证。在本工作中，我们针对三类VIPs（i）强单调，（ii）仿射和（iii）单调提供了SEG-RR的收敛性分析。我们推导了SEG-RR实现比均匀带替换采样SEG具有更快收敛速度的条件。在单调设置中，我们的SEG-RR分析保证了收敛到任意精度而无需大批量大小，这是对大批量大小而言的强要求。

    arXiv:2403.07148v1 Announce Type: cross  Abstract: The Stochastic Extragradient (SEG) method is one of the most popular algorithms for solving finite-sum min-max optimization and variational inequality problems (VIPs) appearing in various machine learning tasks. However, existing convergence analyses of SEG focus on its with-replacement variants, while practical implementations of the method randomly reshuffle components and sequentially use them. Unlike the well-studied with-replacement variants, SEG with Random Reshuffling (SEG-RR) lacks established theoretical guarantees. In this work, we provide a convergence analysis of SEG-RR for three classes of VIPs: (i) strongly monotone, (ii) affine, and (iii) monotone. We derive conditions under which SEG-RR achieves a faster convergence rate than the uniform with-replacement sampling SEG. In the monotone setting, our analysis of SEG-RR guarantees convergence to an arbitrary accuracy without large batch sizes, a strong requirement needed in 
    
[^4]: 递归神经网络的梯度下降收敛性：非渐近性分析

    Convergence of Gradient Descent for Recurrent Neural Networks: A Nonasymptotic Analysis

    [https://arxiv.org/abs/2402.12241](https://arxiv.org/abs/2402.12241)

    该论文分析了在动态系统中利用梯度下降进行监督学习的递归神经网络的性能，并证明在不需要海量过参数化的情况下，梯度下降可以达到最优性。

    

    我们分析在监督学习设置下利用梯度下降训练的递归神经网络在动态系统中的表现，并证明梯度下降可以在\emph{不}需要海量过参数化的情况下达到最优性。我们进行了深入的非渐近性分析，(i)利用序列长度$T$、样本大小$n$和环境维度$d$给出了网络大小$m$和迭代复杂度$\tau$的尖锐界限，(ii)确定了动态系统中长期依赖对收敛和网络宽度界限的显着影响，这些界限由激活函数的Lipschitz连续性决定的截止点来表征。值得注意的是，这一分析揭示了一个妥善初始化的递归神经网络在$n$个样本的情况下，可以通过网络大小$m$仅对数地随$n$扩展就达到最优性。这与以前的工作形成鲜明对比，前者需要高阶多项式分布。

    arXiv:2402.12241v1 Announce Type: new  Abstract: We analyze recurrent neural networks trained with gradient descent in the supervised learning setting for dynamical systems, and prove that gradient descent can achieve optimality \emph{without} massive overparameterization. Our in-depth nonasymptotic analysis (i) provides sharp bounds on the network size $m$ and iteration complexity $\tau$ in terms of the sequence length $T$, sample size $n$ and ambient dimension $d$, and (ii) identifies the significant impact of long-term dependencies in the dynamical system on the convergence and network width bounds characterized by a cutoff point that depends on the Lipschitz continuity of the activation function. Remarkably, this analysis reveals that an appropriately-initialized recurrent neural network trained with $n$ samples can achieve optimality with a network size $m$ that scales only logarithmically with $n$. This sharply contrasts with the prior works that require high-order polynomial dep
    
[^5]: 从高阶统计量中高效学习：假设检验、随机特征和神经网络

    Learning from higher-order statistics, efficiently: hypothesis tests, random features, and neural networks

    [https://arxiv.org/abs/2312.14922](https://arxiv.org/abs/2312.14922)

    神经网络在高维数据中发现统计模式，研究了如何高效地从高阶累积量中提取特征，并探讨了在尖峰累积量模型中的统计和计算限制。

    

    神经网络擅长发现高维数据集中的统计模式。在实践中，度量三个或更多变量间的非高斯相关性的高阶累积量对神经网络的性能特别重要。但神经网络有多有效地从高阶累积量中提取特征？我们在尖峰累积量模型中探讨了这个问题，这里统计学家需要从$d$维输入的阶-$p\ge 4$累积量中恢复出一个特权方向或“尖峰”。我们首先通过分析所需样本数$n$来表征恢复尖峰的基本统计和计算限制，以强烈区分来自尖峰累积量模型和各向同性高斯输入的输入。我们发现，统计上的可区分性需要$n\gtrsim d$个样本，而在多项式时间内区分这两个分布则需要

    arXiv:2312.14922v2 Announce Type: replace-cross  Abstract: Neural networks excel at discovering statistical patterns in high-dimensional data sets. In practice, higher-order cumulants, which quantify the non-Gaussian correlations between three or more variables, are particularly important for the performance of neural networks. But how efficient are neural networks at extracting features from higher-order cumulants? We study this question in the spiked cumulant model, where the statistician needs to recover a privileged direction or "spike" from the order-$p\ge 4$ cumulants of $d$-dimensional inputs. We first characterise the fundamental statistical and computational limits of recovering the spike by analysing the number of samples $n$ required to strongly distinguish between inputs from the spiked cumulant model and isotropic Gaussian inputs. We find that statistical distinguishability requires $n\gtrsim d$ samples, while distinguishing the two distributions in polynomial time require
    
[^6]: 对抗性赌博机针对任意策略的研究

    Adversarial Bandits against Arbitrary Strategies

    [https://arxiv.org/abs/2205.14839](https://arxiv.org/abs/2205.14839)

    该论文研究了对抗性赌博机问题，针对任意策略，提出了使用在线镜像下降方法的主控基础框架，并使用自适应学习率的OMD来减轻方差的影响，取得了较好的结果。

    

    我们研究了针对任意策略的对抗性赌博机问题，其中S是问题难度的参数，该参数对于代理人来说是未知的。为了解决这个问题，我们采用了使用在线镜像下降方法（OMD）的主控基础框架。我们首先提供了一个具有简单OMD的主控基础算法，实现了$\tilde{O}(S^{1/2}K^{1/3}T^{2/3})$的结果，其中$T^{2/3}$来自损失估计器的方差。为了减轻方差的影响，我们提出使用自适应学习率的OMD，并实现了$\tilde{O}(\min\{\mathbb{E}[\sqrt{SKT\rho_T(h^\dagger)}],S\sqrt{KT}\})$的结果，其中$\rho_T(h^\dagger)$是损失估计器的方差项。

    We study the adversarial bandit problem against arbitrary strategies, in which $S$ is the parameter for the hardness of the problem and this parameter is not given to the agent. To handle this problem, we adopt the master-base framework using the online mirror descent method (OMD). We first provide a master-base algorithm with simple OMD, achieving $\tilde{O}(S^{1/2}K^{1/3}T^{2/3})$, in which $T^{2/3}$ comes from the variance of loss estimators. To mitigate the impact of the variance, we propose using adaptive learning rates for OMD and achieve $\tilde{O}(\min\{\mathbb{E}[\sqrt{SKT\rho_T(h^\dagger)}],S\sqrt{KT}\})$, where $\rho_T(h^\dagger)$ is a variance term for loss estimators.
    
[^7]: 利用扩散模型进行干预和反事实推断

    Interventional and Counterfactual Inference with Diffusion Models. (arXiv:2302.00860v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.00860](http://arxiv.org/abs/2302.00860)

    本论文提出了基于扩散模型的因果模型 (DCM)，它可以在只有观测数据和因果图可用的情况下进行干预和反事实推断，其具有较好的表现。同时，论文还提供了一种分析反事实估计的方法，可以应用于更广泛的场景。

    

    我们考虑在只有观测数据和因果图可用的因果充分设置中回答观测、干预和反事实查询的问题。利用扩散模型的最新发展，我们引入了基于扩散的因果模型 (DCM)，来学习生成独特的潜在编码的因果机制。这些编码使我们能够在干预下直接采样和进行反事实推断。扩散模型在这里是一个自然的选择，因为它们可以将每个节点编码为一个代表外生噪声的潜在表示。我们的实证评估表明，在回答因果查询方面，与现有的最先进方法相比，有显着的改进。此外，我们提供了理论结果，为分析一般编码器-解码器模型中的反事实估计提供一种方法，这对我们提出的方法以外的设置可能也有用。

    We consider the problem of answering observational, interventional, and counterfactual queries in a causally sufficient setting where only observational data and the causal graph are available. Utilizing the recent developments in diffusion models, we introduce diffusion-based causal models (DCM) to learn causal mechanisms, that generate unique latent encodings. These encodings enable us to directly sample under interventions and perform abduction for counterfactuals. Diffusion models are a natural fit here, since they can encode each node to a latent representation that acts as a proxy for exogenous noise. Our empirical evaluations demonstrate significant improvements over existing state-of-the-art methods for answering causal queries. Furthermore, we provide theoretical results that offer a methodology for analyzing counterfactual estimation in general encoder-decoder models, which could be useful in settings beyond our proposed approach.
    
[^8]: 带噪声特征的上下文线性Bandit：朝向贝叶斯神谕前进

    Contextual Linear Bandits under Noisy Features: Towards Bayesian Oracles. (arXiv:1703.01347v3 [cs.AI] UPDATED)

    [http://arxiv.org/abs/1703.01347](http://arxiv.org/abs/1703.01347)

    本论文研究了具有噪声特征的上下文线性Bandit问题。我们提出了一个算法，通过观察信息，实现了贝叶斯神谕并得到了$\tilde{O}(d\sqrt{T})$的遗憾界。

    

    我们研究了带有噪声和缺失项的上下文线性Bandit问题。为了解决噪声的挑战，我们分析了在观测噪声特征的情况下给出的贝叶斯神谕。我们的贝叶斯分析发现，最优假设可能会远离潜在的可实现函数，这取决于噪声特征，这是高度非直观的，并且在经典的无噪声设置下不会发生。这意味着经典方法不能保证非平凡的遗憾界（regret bound）。因此，我们提出了一个算法，旨在从这个模型下的观察信息中实现贝叶斯神谕，当有大量手臂时，可以实现$\tilde{O}(d\sqrt{T})$遗憾界。我们使用合成和实际数据集演示了所提出的算法。

    We study contextual linear bandit problems under feature uncertainty; they are noisy with missing entries. To address the challenges of the noise, we analyze Bayesian oracles given observed noisy features. Our Bayesian analysis finds that the optimal hypothesis can be far from the underlying realizability function, depending on the noise characteristics, which are highly non-intuitive and do not occur for classical noiseless setups. This implies that classical approaches cannot guarantee a non-trivial regret bound. Therefore, we propose an algorithm that aims at the Bayesian oracle from observed information under this model, achieving $\tilde{O}(d\sqrt{T})$ regret bound when there is a large number of arms. We demonstrate the proposed algorithm using synthetic and real-world datasets.
    

