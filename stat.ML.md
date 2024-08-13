# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Causal Representation Learning from Multiple Distributions: A General Setting](https://arxiv.org/abs/2402.05052) | 本文研究了一个通用的、完全非参数的因果表示学习设置，旨在在多个分布之间学习因果关系，无需假设硬干预。通过稀疏性约束，可以从多个分布中恢复出因果关系。 |
| [^2] | [Improving and Unifying Discrete&Continuous-time Discrete Denoising Diffusion](https://arxiv.org/abs/2402.03701) | 本文提出了一种改进和统一离散和连续时间离散去噪扩散的方法。通过数学简化和推导，使得离散扩散的训练更准确易优化，并且实现了精确和加速的采样。同时，成功地统一了离散时间和连续时间离散扩散。 |
| [^3] | [Featurizing Koopman Mode Decomposition](https://arxiv.org/abs/2312.09146) | 本文提出了一种命名为FKMD的先进KMD技术，通过时间嵌入和马氏距离缩放，可以增强对高维动力系统的分析和预测，特别适用于特征未知的情况，并在丙氨酸二肽数据降维和分析Lorenz吸引子和癌症研究中细胞信号问题方面取得了显著改进。 |
| [^4] | [Deep Backtracking Counterfactuals for Causally Compliant Explanations.](http://arxiv.org/abs/2310.07665) | 本研究提供了一种实用方法，用于在深度生成组件的结构因果模型中计算回溯反事实。通过在因果模型的结构化潜在空间中解决优化问题，我们的方法能够生成反事实，并且与其他方法相比具备了多功能、模块化和符合因果关系的特点。 |
| [^5] | [Active Learning in Symbolic Regression Performance with Physical Constraints.](http://arxiv.org/abs/2305.10379) | 本文探讨了利用进化符号回归作为主动学习中的方法来提出哪些数据应该被采集，通过“委员会查询”来减少所需数据，并在重新发现已知方程所需的数据方面实现最新的结果。 |
| [^6] | [Manifold Learning by Mixture Models of VAEs for Inverse Problems.](http://arxiv.org/abs/2303.15244) | 本文提出了一种用混合VAE模型学习流形的方法，并将其用于解决逆问题，结果表现出良好的性能，可用于模糊和电阻抗层析成像。 |
| [^7] | [Policy Gradient Converges to the Globally Optimal Policy for Nearly Linear-Quadratic Regulators.](http://arxiv.org/abs/2303.08431) | 本论文研究了强化学习方法在几乎线性二次型调节器系统中找到最优策略的问题，提出了一个策略梯度算法，可以以线性速率收敛于全局最优解。 |

# 详细

[^1]: 从多个分布中进行因果表示学习：一个通用设置

    Causal Representation Learning from Multiple Distributions: A General Setting

    [https://arxiv.org/abs/2402.05052](https://arxiv.org/abs/2402.05052)

    本文研究了一个通用的、完全非参数的因果表示学习设置，旨在在多个分布之间学习因果关系，无需假设硬干预。通过稀疏性约束，可以从多个分布中恢复出因果关系。

    

    在许多问题中，测量变量（例如图像像素）只是隐藏的因果变量（例如潜在的概念或对象）的数学函数。为了在不断变化的环境中进行预测或对系统进行适当的更改，恢复隐藏的因果变量$Z_i$以及由图$\mathcal{G}_Z$表示的它们的因果关系是有帮助的。这个问题最近被称为因果表示学习。本文关注来自多个分布（来自异构数据或非平稳时间序列）的因果表示学习的通用、完全非参数的设置，不需要假设分布改变背后存在硬干预。我们旨在在这个基本情况下开发通用解决方案；作为副产品，这有助于看到其他假设（如参数因果模型或硬干预）提供的独特好处。我们证明在恢复过程中对图的稀疏性约束下，可以从多个分布中学习出因果关系。

    In many problems, the measured variables (e.g., image pixels) are just mathematical functions of the hidden causal variables (e.g., the underlying concepts or objects). For the purpose of making predictions in changing environments or making proper changes to the system, it is helpful to recover the hidden causal variables $Z_i$ and their causal relations represented by graph $\mathcal{G}_Z$. This problem has recently been known as causal representation learning. This paper is concerned with a general, completely nonparametric setting of causal representation learning from multiple distributions (arising from heterogeneous data or nonstationary time series), without assuming hard interventions behind distribution changes. We aim to develop general solutions in this fundamental case; as a by product, this helps see the unique benefit offered by other assumptions such as parametric causal models or hard interventions. We show that under the sparsity constraint on the recovered graph over
    
[^2]: 改进和统一离散和连续时间离散去噪扩散

    Improving and Unifying Discrete&Continuous-time Discrete Denoising Diffusion

    [https://arxiv.org/abs/2402.03701](https://arxiv.org/abs/2402.03701)

    本文提出了一种改进和统一离散和连续时间离散去噪扩散的方法。通过数学简化和推导，使得离散扩散的训练更准确易优化，并且实现了精确和加速的采样。同时，成功地统一了离散时间和连续时间离散扩散。

    

    离散扩散模型在自然离散数据如语言和图形上得到了广泛关注。虽然离散时间离散扩散已经建立了一段时间，但直到最近Campbell等人（2022）才引入了连续时间离散扩散的第一个框架。然而，他们的训练和采样过程与离散时间版本有很大差异，需要非平凡的近似才能进行可行性分析。本文首先介绍了一系列对变分下界的数学简化，这些简化使离散扩散的训练更加准确和易于优化。此外，我们推导出了一种简单的反向去噪公式，能够实现精确和加速的采样，更重要的是能够优雅地统一离散时间和连续时间离散扩散。通过更简单的分析公式，前向和现在也包括了后向概率可以灵活地适应任何噪声分布。

    Discrete diffusion models have seen a surge of attention with applications on naturally discrete data such as language and graphs. Although discrete-time discrete diffusion has been established for a while, only recently Campbell et al. (2022) introduced the first framework for continuous-time discrete diffusion. However, their training and sampling processes differ significantly from the discrete-time version, necessitating nontrivial approximations for tractability. In this paper, we first present a series of mathematical simplifications of the variational lower bound that enable more accurate and easy-to-optimize training for discrete diffusion. In addition, we derive a simple formulation for backward denoising that enables exact and accelerated sampling, and importantly, an elegant unification of discrete-time and continuous-time discrete diffusion. Thanks to simpler analytical formulations, both forward and now also backward probabilities can flexibly accommodate any noise distrib
    
[^3]: 对Koopman模态分解进行特征化处理

    Featurizing Koopman Mode Decomposition

    [https://arxiv.org/abs/2312.09146](https://arxiv.org/abs/2312.09146)

    本文提出了一种命名为FKMD的先进KMD技术，通过时间嵌入和马氏距离缩放，可以增强对高维动力系统的分析和预测，特别适用于特征未知的情况，并在丙氨酸二肽数据降维和分析Lorenz吸引子和癌症研究中细胞信号问题方面取得了显著改进。

    

    本文介绍了一种先进的Koopman模态分解（KMD）技术：命名为特征化Koopman模态分解（FKMD），该技术利用时间嵌入和马氏距离缩放来增强对高维动力系统的分析和预测。时间嵌入扩展了观测空间，更好地捕捉基础流形结构，而应用于核函数或随机傅里叶特征的马氏距离缩放，则根据系统的动态调整观测值。这有助于在不事先知道良好特征的情况下对KMD进行特征化处理。我们发现，FKMD中的马氏距离缩放可用于对丙氨酸二肽数据进行有效的降维。我们还展示了FKMD如何改善对高维Lorenz吸引子和癌症研究中的细胞信号问题的预测。

    arXiv:2312.09146v3 Announce Type: replace-cross  Abstract: This article introduces an advanced Koopman mode decomposition (KMD) technique -- coined Featurized Koopman Mode Decomposition (FKMD) -- that uses time embedding and Mahalanobis scaling to enhance analysis and prediction of high dimensional dynamical systems. The time embedding expands the observation space to better capture underlying manifold structure, while the Mahalanobis scaling, applied to kernel or random Fourier features, adjusts observations based on the system's dynamics. This aids in featurizing KMD in cases where good features are not a priori known. We find that the Mahalanobis scaling from FKMD can be used for effective dimensionality reduction of alanine dipeptide data. We also show that FKMD improves predictions for a high-dimensional Lorenz attractor and a cell signaling problem from cancer research.
    
[^4]: 深度回溯对因果一致解释的反事实推理

    Deep Backtracking Counterfactuals for Causally Compliant Explanations. (arXiv:2310.07665v1 [cs.AI])

    [http://arxiv.org/abs/2310.07665](http://arxiv.org/abs/2310.07665)

    本研究提供了一种实用方法，用于在深度生成组件的结构因果模型中计算回溯反事实。通过在因果模型的结构化潜在空间中解决优化问题，我们的方法能够生成反事实，并且与其他方法相比具备了多功能、模块化和符合因果关系的特点。

    

    反事实推理可以通过回答在改变情况下会观察到什么来提供有价值的见解，条件是根据实际观察。虽然经典的介入式解释已经得到了广泛研究，回溯原则被提出作为一种保持所有因果定律完整性的替代哲学，但其研究较少。在本研究中，我们介绍了在由深度生成组件组成的结构因果模型中计算回溯反事实的实用方法。为此，我们对结构分配施加了条件，通过在因果模型的结构化潜在空间中解决一个可行的约束优化问题来生成反事实。我们的方法还可以与反事实解释领域的方法进行比较。与这些方法相比，我们的方法代表了一种多功能、模块化和遵守因果的替代方案。

    Counterfactuals can offer valuable insights by answering what would have been observed under altered circumstances, conditional on a factual observation. Whereas the classical interventional interpretation of counterfactuals has been studied extensively, backtracking constitutes a less studied alternative the backtracking principle has emerged as an alternative philosophy where all causal laws are kept intact. In the present work, we introduce a practical method for computing backtracking counterfactuals in structural causal models that consist of deep generative components. To this end, we impose conditions on the structural assignments that enable the generation of counterfactuals by solving a tractable constrained optimization problem in the structured latent space of a causal model. Our formulation also facilitates a comparison with methods in the field of counterfactual explanations. Compared to these, our method represents a versatile, modular and causally compliant alternative. 
    
[^5]: 基于物理约束的符号回归中主动学习的表现

    Active Learning in Symbolic Regression Performance with Physical Constraints. (arXiv:2305.10379v1 [cs.LG])

    [http://arxiv.org/abs/2305.10379](http://arxiv.org/abs/2305.10379)

    本文探讨了利用进化符号回归作为主动学习中的方法来提出哪些数据应该被采集，通过“委员会查询”来减少所需数据，并在重新发现已知方程所需的数据方面实现最新的结果。

    

    进化符号回归（SR）是一种将符号方程拟合到数据中的方法，可以得到简洁易懂的模型。本文探讨使用SR作为主动学习中的方法来提出哪些数据应该被采集，在此过程中考虑物理约束。基于主动学习的SR通过“委员会查询”来提出下一步实验。物理约束可以在非常低的数据情况下改善所建议的方程。这些方法可以减少SR所需的数据，并在重新发现已知方程所需的数据方面实现最新的结果。

    Evolutionary symbolic regression (SR) fits a symbolic equation to data, which gives a concise interpretable model. We explore using SR as a method to propose which data to gather in an active learning setting with physical constraints. SR with active learning proposes which experiments to do next. Active learning is done with query by committee, where the Pareto frontier of equations is the committee. The physical constraints improve proposed equations in very low data settings. These approaches reduce the data required for SR and achieves state of the art results in data required to rediscover known equations.
    
[^6]: 用混合VAE模型学习流形来解决逆问题

    Manifold Learning by Mixture Models of VAEs for Inverse Problems. (arXiv:2303.15244v1 [cs.LG])

    [http://arxiv.org/abs/2303.15244](http://arxiv.org/abs/2303.15244)

    本文提出了一种用混合VAE模型学习流形的方法，并将其用于解决逆问题，结果表现出良好的性能，可用于模糊和电阻抗层析成像。

    

    在实践中，使用生成模型表示高维数据的流形已被证明具有计算效率。然而，这要求数据流形具有全局参数化。为了表示任意拓扑的流形，我们提出了学习变分自编码器的混合模型。这里，每个编码器-解码器对表示流形的一个图表。我们提出了一种损失函数来最大化似然估计模型权重，并选择一个架构，为我们提供图表及其逆的解析表达式。一旦学习了流形，我们将其用于通过将数据拟合项限制在学习的流形上来解决逆问题。为了解决所产生的最小化问题，我们在学习的流形上提出了一种黎曼梯度下降算法。我们展示了我们的方法在低维玩具例子以及模糊和电阻抗层析成像方面的性能。

    Representing a manifold of very high-dimensional data with generative models has been shown to be computationally efficient in practice. However, this requires that the data manifold admits a global parameterization. In order to represent manifolds of arbitrary topology, we propose to learn a mixture model of variational autoencoders. Here, every encoder-decoder pair represents one chart of a manifold. We propose a loss function for maximum likelihood estimation of the model weights and choose an architecture that provides us the analytical expression of the charts and of their inverses. Once the manifold is learned, we use it for solving inverse problems by minimizing a data fidelity term restricted to the learned manifold. To solve the arising minimization problem we propose a Riemannian gradient descent algorithm on the learned manifold. We demonstrate the performance of our method for low-dimensional toy examples as well as for deblurring and electrical impedance tomography on cert
    
[^7]: 政策梯度算法收敛于几乎线性二次型调节器的全局最优策略

    Policy Gradient Converges to the Globally Optimal Policy for Nearly Linear-Quadratic Regulators. (arXiv:2303.08431v1 [cs.LG])

    [http://arxiv.org/abs/2303.08431](http://arxiv.org/abs/2303.08431)

    本论文研究了强化学习方法在几乎线性二次型调节器系统中找到最优策略的问题，提出了一个策略梯度算法，可以以线性速率收敛于全局最优解。

    

    决策者只获得了非完整信息的非线性控制系统在各种应用中普遍存在。本研究探索了强化学习方法，以找到几乎线性二次型调节器系统中最优策略。我们考虑一个动态系统，结合线性和非线性组成部分，并由相同结构的策略进行管理。在假设非线性组成部分包含具有小型Lipschitz系数的内核的情况下，我们对成本函数的优化进行了表征。虽然成本函数通常是非凸的，但我们确立了全局最优解附近局部的强凸性和光滑性。此外，我们提出了一种初始化机制，以利用这些属性。在此基础上，我们设计了一个策略梯度算法，可以保证以线性速率收敛于全局最优解。

    Nonlinear control systems with partial information to the decision maker are prevalent in a variety of applications. As a step toward studying such nonlinear systems, this work explores reinforcement learning methods for finding the optimal policy in the nearly linear-quadratic regulator systems. In particular, we consider a dynamic system that combines linear and nonlinear components, and is governed by a policy with the same structure. Assuming that the nonlinear component comprises kernels with small Lipschitz coefficients, we characterize the optimization landscape of the cost function. Although the cost function is nonconvex in general, we establish the local strong convexity and smoothness in the vicinity of the global optimizer. Additionally, we propose an initialization mechanism to leverage these properties. Building on the developments, we design a policy gradient algorithm that is guaranteed to converge to the globally optimal policy with a linear rate.
    

