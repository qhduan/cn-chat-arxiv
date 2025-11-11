# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diffusion Posterior Sampling is Computationally Intractable](https://arxiv.org/abs/2402.12727) | 我们证明了后验抽样在计算上是难以解决的：在加密学中最基本的假设下——单向函数存在的假设下，存在一些实例，对于这些实例，每个算法都需要超多项式时间，即使无条件抽样可以证明是快速的。 |
| [^2] | [Deeper or Wider: A Perspective from Optimal Generalization Error with Sobolev Loss](https://arxiv.org/abs/2402.00152) | 本文比较了更深的神经网络和更宽的神经网络在Sobolev损失的最优泛化误差方面的表现，研究发现神经网络的架构受多种因素影响，参数数量更多倾向于选择更宽的网络，而样本点数量和损失函数规则性更高倾向于选择更深的网络。 |
| [^3] | [Causal Dynamic Variational Autoencoder for Counterfactual Regression in Longitudinal Data.](http://arxiv.org/abs/2310.10559) | 本论文提出了一种因果动态变分自编码器（CDVAE）来解决纵向数据中的反事实回归问题。该方法假设存在未观察到的调整变量，并通过结合动态变分自编码器（DVAE）框架和使用倾向得分的加权策略来估计反事实响应。 |
| [^4] | [A General Framework for Learning under Corruption: Label Noise, Attribute Noise, and Beyond.](http://arxiv.org/abs/2307.08643) | 该研究提出了一个通用框架，在分布层面上对不同类型的数据污染模型进行了形式化分析，并通过分析贝叶斯风险的变化展示了这些污染对标准监督学习的影响。这些发现为进一步研究提供了新的方向和基础。 |

# 详细

[^1]: 扩散后验抽样在计算上是难以解决的

    Diffusion Posterior Sampling is Computationally Intractable

    [https://arxiv.org/abs/2402.12727](https://arxiv.org/abs/2402.12727)

    我们证明了后验抽样在计算上是难以解决的：在加密学中最基本的假设下——单向函数存在的假设下，存在一些实例，对于这些实例，每个算法都需要超多项式时间，即使无条件抽样可以证明是快速的。

    

    扩散模型是学习和从分布$p(x)$中抽样的一种非常有效的方法。在后验抽样中，人们还会给出一个测量模型$p(y \mid x)$和一个测量$y$，希望从$p(x \mid y)$中抽样。后验抽样对于诸如修补、超分辨率和MRI重建等任务非常有用，因此一些最近的工作已经给出了启发式近似算法；但没有一个已知能在多项式时间内收敛到正确的分布。

    arXiv:2402.12727v1 Announce Type: cross  Abstract: Diffusion models are a remarkably effective way of learning and sampling from a distribution $p(x)$. In posterior sampling, one is also given a measurement model $p(y \mid x)$ and a measurement $y$, and would like to sample from $p(x \mid y)$. Posterior sampling is useful for tasks such as inpainting, super-resolution, and MRI reconstruction, so a number of recent works have given algorithms to heuristically approximate it; but none are known to converge to the correct distribution in polynomial time.   In this paper we show that posterior sampling is \emph{computationally intractable}: under the most basic assumption in cryptography -- that one-way functions exist -- there are instances for which \emph{every} algorithm takes superpolynomial time, even though \emph{unconditional} sampling is provably fast. We also show that the exponential-time rejection sampling algorithm is essentially optimal under the stronger plausible assumption 
    
[^2]: 更深还是更宽: 从Sobolev损失的最优泛化误差角度看

    Deeper or Wider: A Perspective from Optimal Generalization Error with Sobolev Loss

    [https://arxiv.org/abs/2402.00152](https://arxiv.org/abs/2402.00152)

    本文比较了更深的神经网络和更宽的神经网络在Sobolev损失的最优泛化误差方面的表现，研究发现神经网络的架构受多种因素影响，参数数量更多倾向于选择更宽的网络，而样本点数量和损失函数规则性更高倾向于选择更深的网络。

    

    构建神经网络的架构是机器学习界一个具有挑战性的追求，到底是更深还是更宽一直是一个持续的问题。本文探索了更深的神经网络（DeNNs）和具有有限隐藏层的更宽的神经网络（WeNNs）在Sobolev损失的最优泛化误差方面的比较。通过分析研究发现，神经网络的架构可以受到多种因素的显著影响，包括样本点的数量，神经网络内的参数以及损失函数的规则性。具体而言，更多的参数倾向于选择WeNNs，而更多的样本点和更高的损失函数规则性倾向于选择DeNNs。最后，我们将这个理论应用于使用深度Ritz和物理感知神经网络（PINN）方法解决偏微分方程的问题。

    Constructing the architecture of a neural network is a challenging pursuit for the machine learning community, and the dilemma of whether to go deeper or wider remains a persistent question. This paper explores a comparison between deeper neural networks (DeNNs) with a flexible number of layers and wider neural networks (WeNNs) with limited hidden layers, focusing on their optimal generalization error in Sobolev losses. Analytical investigations reveal that the architecture of a neural network can be significantly influenced by various factors, including the number of sample points, parameters within the neural networks, and the regularity of the loss function. Specifically, a higher number of parameters tends to favor WeNNs, while an increased number of sample points and greater regularity in the loss function lean towards the adoption of DeNNs. We ultimately apply this theory to address partial differential equations using deep Ritz and physics-informed neural network (PINN) methods,
    
[^3]: 因果动态变分自编码器用于纵向数据中的反事实回归

    Causal Dynamic Variational Autoencoder for Counterfactual Regression in Longitudinal Data. (arXiv:2310.10559v1 [stat.ML])

    [http://arxiv.org/abs/2310.10559](http://arxiv.org/abs/2310.10559)

    本论文提出了一种因果动态变分自编码器（CDVAE）来解决纵向数据中的反事实回归问题。该方法假设存在未观察到的调整变量，并通过结合动态变分自编码器（DVAE）框架和使用倾向得分的加权策略来估计反事实响应。

    

    在很多实际应用中，如精准医学、流行病学、经济和市场营销中，估计随时间变化的治疗效果是相关的。许多最先进的方法要么假设了所有混杂变量的观测结果，要么试图推断未观察到的混杂变量。我们采取了不同的观点，假设存在未观察到的风险因素，即仅影响结果序列的调整变量。在无混杂性的情况下，我们以未观测到的风险因素导致的治疗反应中的未知异质性为目标，估计个体治疗效果（ITE）。我们应对了时变效应和未观察到的调整变量所带来的挑战。在学习到的调整变量的有效性和治疗效果的一般化界限的理论结果指导下，我们设计了因果DVAE（CDVAE）。该模型将动态变分自编码器（DVAE）框架与使用倾向得分的加权策略相结合，用于估计反事实响应。

    Estimating treatment effects over time is relevant in many real-world applications, such as precision medicine, epidemiology, economy, and marketing. Many state-of-the-art methods either assume the observations of all confounders or seek to infer the unobserved ones. We take a different perspective by assuming unobserved risk factors, i.e., adjustment variables that affect only the sequence of outcomes. Under unconfoundedness, we target the Individual Treatment Effect (ITE) estimation with unobserved heterogeneity in the treatment response due to missing risk factors. We address the challenges posed by time-varying effects and unobserved adjustment variables. Led by theoretical results over the validity of the learned adjustment variables and generalization bounds over the treatment effect, we devise Causal DVAE (CDVAE). This model combines a Dynamic Variational Autoencoder (DVAE) framework with a weighting strategy using propensity scores to estimate counterfactual responses. The CDVA
    
[^4]: 一个学习受到污染的通用框架：标签噪声、属性噪声等等

    A General Framework for Learning under Corruption: Label Noise, Attribute Noise, and Beyond. (arXiv:2307.08643v1 [cs.LG])

    [http://arxiv.org/abs/2307.08643](http://arxiv.org/abs/2307.08643)

    该研究提出了一个通用框架，在分布层面上对不同类型的数据污染模型进行了形式化分析，并通过分析贝叶斯风险的变化展示了这些污染对标准监督学习的影响。这些发现为进一步研究提供了新的方向和基础。

    

    数据中的污染现象很常见，并且已经在不同的污染模型下进行了广泛研究。尽管如此，对于这些模型之间的关系仍然了解有限，缺乏对污染及其对学习的影响的统一视角。在本研究中，我们通过基于马尔可夫核的一般性和详尽的框架，在分布层面上正式分析了污染模型。我们强调了标签和属性上存在的复杂联合和依赖性污染，这在现有研究中很少触及。此外，我们通过分析贝叶斯风险变化来展示这些污染如何影响标准的监督学习。我们的发现提供了对于“更复杂”污染对学习问题影响的定性洞察，并为未来的定量比较提供了基础。该框架的应用包括污染校正学习，其中包含一个子案例。

    Corruption is frequently observed in collected data and has been extensively studied in machine learning under different corruption models. Despite this, there remains a limited understanding of how these models relate such that a unified view of corruptions and their consequences on learning is still lacking. In this work, we formally analyze corruption models at the distribution level through a general, exhaustive framework based on Markov kernels. We highlight the existence of intricate joint and dependent corruptions on both labels and attributes, which are rarely touched by existing research. Further, we show how these corruptions affect standard supervised learning by analyzing the resulting changes in Bayes Risk. Our findings offer qualitative insights into the consequences of "more complex" corruptions on the learning problem, and provide a foundation for future quantitative comparisons. Applications of the framework include corruption-corrected learning, a subcase of which we 
    

