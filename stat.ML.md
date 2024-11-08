# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Provable Mutual Benefits from Federated Learning in Privacy-Sensitive Domains](https://arxiv.org/abs/2403.06672) | 本文研究了在隐私敏感领域中如何设计一种FL协议，既能保证隐私，又能提高模型准确性，并提供了设计出对所有参与者都有益处的协议。 |
| [^2] | [Depth Separations in Neural Networks: Separating the Dimension from the Accuracy](https://arxiv.org/abs/2402.07248) | 通过研究深度2和深度3神经网络在逼近Lipschitz目标函数时的分离性质，证明了维度诅咒也会在深度2逼近中存在，即使目标函数可以使用深度3高效表示。这为以前确定深度要求的下界提供了新的观点，并且适用于多种激活函数。 |
| [^3] | [The Implicit Bias of Gradient Noise: A Symmetry Perspective](https://arxiv.org/abs/2402.07193) | 本研究通过对对称性的存在进行分析，揭示了梯度噪声在随机梯度下降中的隐性偏见。我们发现不同类型的对称性会导致不同的学习动态，其中一类对称性可以自然收敛，而另一类对称性几乎总是发散。此外，我们的研究结果适用于没有对称性的损失函数，对于理解训练动态和解释相关实际问题具有普适性。 |
| [^4] | [Multi-Lattice Sampling of Quantum Field Theories via Neural Operator-based Flows.](http://arxiv.org/abs/2401.00828) | 本文提出了一种基于神经算子流的方法，通过近似时间相关算子，实现了在量子场论中从底层自由理论到目标理论的离散-连续归一化流。 |
| [^5] | [Latent Diffusion Model for Conditional Reservoir Facies Generation.](http://arxiv.org/abs/2311.01968) | 本研究提出了一种专门用于条件下储层相生成的潜在扩散模型，通过充分保留条件数据，生成了高保真度的储层相。它在性能上明显优于基于GANs的替代方法。 |
| [^6] | [Generative Entropic Neural Optimal Transport To Map Within and Across Spaces.](http://arxiv.org/abs/2310.09254) | 该论文介绍了生成熵神经最优传输在测度到测度映射中的应用，解决了处理非平方欧氏距离成本、确定性蒙格映射、映射跨不可比较空间和质量守恒约束等实际挑战。 |
| [^7] | [Anytime-valid t-tests and confidence sequences for Gaussian means with unknown variance.](http://arxiv.org/abs/2310.03722) | 本文提出了两种新的“e-process”和置信序列方法，分别通过替换Lai的混合方法，并分析了所得结果的宽度。 |

# 详细

[^1]: 在隐私敏感领域中从联邦学习中有可证明的互惠益处

    Provable Mutual Benefits from Federated Learning in Privacy-Sensitive Domains

    [https://arxiv.org/abs/2403.06672](https://arxiv.org/abs/2403.06672)

    本文研究了在隐私敏感领域中如何设计一种FL协议，既能保证隐私，又能提高模型准确性，并提供了设计出对所有参与者都有益处的协议。

    

    跨领域联邦学习（FL）允许数据所有者通过从彼此的私有数据集中获益来训练准确的机器学习模型。本文研究了在何时以及如何服务器可以设计一种对所有参与者都有利的FL协议的问题。我们提供了在均值估计和凸随机优化背景下存在相互有利协议的必要和充分条件。我们推导出了在对称隐私偏好下，最大化总客户效用的协议。最后，我们设计了最大化最终模型准确性的协议，并在合成实验中展示了它们的好处。

    arXiv:2403.06672v1 Announce Type: cross  Abstract: Cross-silo federated learning (FL) allows data owners to train accurate machine learning models by benefiting from each others private datasets. Unfortunately, the model accuracy benefits of collaboration are often undermined by privacy defenses. Therefore, to incentivize client participation in privacy-sensitive domains, a FL protocol should strike a delicate balance between privacy guarantees and end-model accuracy. In this paper, we study the question of when and how a server could design a FL protocol provably beneficial for all participants. First, we provide necessary and sufficient conditions for the existence of mutually beneficial protocols in the context of mean estimation and convex stochastic optimization. We also derive protocols that maximize the total clients' utility, given symmetric privacy preferences. Finally, we design protocols maximizing end-model accuracy and demonstrate their benefits in synthetic experiments.
    
[^2]: 神经网络中的深度分离：将维度与准确度分离

    Depth Separations in Neural Networks: Separating the Dimension from the Accuracy

    [https://arxiv.org/abs/2402.07248](https://arxiv.org/abs/2402.07248)

    通过研究深度2和深度3神经网络在逼近Lipschitz目标函数时的分离性质，证明了维度诅咒也会在深度2逼近中存在，即使目标函数可以使用深度3高效表示。这为以前确定深度要求的下界提供了新的观点，并且适用于多种激活函数。

    

    我们证明了深度2和深度3神经网络在逼近一个$\mathcal{O}(1)$-Lipschitz目标函数至常数精度时的指数分离，对于支持在$[0,1]^{d}$上的分布，假设权重指数有界。这解决了在\citet{safran2019depth}中提出的一个问题，并证明了维度诅咒在深度2逼近中的存在，即使在目标函数可以使用深度3高效表示的情况下也是如此。以前，将深度2和深度3分离的下界要求至少有一个Lipschitz参数、目标准确度或逼近域的大小（某种度量）与输入维度多项式地缩放，而我们保持前两者不变，并将我们的域限制在单位超立方体上。我们的下界适用于各种激活函数，并基于一种新的平均情况到最坏情况的随机自约化论证的应用，以减少

    We prove an exponential separation between depth 2 and depth 3 neural networks, when approximating an $\mathcal{O}(1)$-Lipschitz target function to constant accuracy, with respect to a distribution with support in $[0,1]^{d}$, assuming exponentially bounded weights. This addresses an open problem posed in \citet{safran2019depth}, and proves that the curse of dimensionality manifests in depth 2 approximation, even in cases where the target function can be represented efficiently using depth 3. Previously, lower bounds that were used to separate depth 2 from depth 3 required that at least one of the Lipschitz parameter, target accuracy or (some measure of) the size of the domain of approximation scale polynomially with the input dimension, whereas we fix the former two and restrict our domain to the unit hypercube. Our lower bound holds for a wide variety of activation functions, and is based on a novel application of an average- to worst-case random self-reducibility argument, to reduce
    
[^3]: 梯度噪声的隐性偏见：从对称性角度来看

    The Implicit Bias of Gradient Noise: A Symmetry Perspective

    [https://arxiv.org/abs/2402.07193](https://arxiv.org/abs/2402.07193)

    本研究通过对对称性的存在进行分析，揭示了梯度噪声在随机梯度下降中的隐性偏见。我们发现不同类型的对称性会导致不同的学习动态，其中一类对称性可以自然收敛，而另一类对称性几乎总是发散。此外，我们的研究结果适用于没有对称性的损失函数，对于理解训练动态和解释相关实际问题具有普适性。

    

    我们对随机梯度下降（SGD）在损失函数存在连续对称性时的学习动态进行了表征，说明了SGD和梯度下降之间的分歧是多么巨大。我们展示了根据对称性对学习动态的影响方式，我们可以将一族对称性分为两类。对于一类对称性，SGD自然地收敛到具有平衡和对齐梯度噪声的解。对于另一类对称性，SGD几乎总是发散的。然后，我们展示了即使损失函数中没有对称性，我们的结果依然适用并可以帮助我们理解训练动态。我们的主要结果是普遍的，它只依赖于对称性的存在，而与损失函数的细节无关。我们证明了所提出的理论对于逐步变形和平坦化提供了解释，并可以应用于常见的实际问题，如表示正则化。

    We characterize the learning dynamics of stochastic gradient descent (SGD) when continuous symmetry exists in the loss function, where the divergence between SGD and gradient descent is dramatic. We show that depending on how the symmetry affects the learning dynamics, we can divide a family of symmetry into two classes. For one class of symmetry, SGD naturally converges to solutions that have a balanced and aligned gradient noise. For the other class of symmetry, SGD will almost always diverge. Then, we show that our result remains applicable and can help us understand the training dynamics even when the symmetry is not present in the loss function. Our main result is universal in the sense that it only depends on the existence of the symmetry and is independent of the details of the loss function. We demonstrate that the proposed theory offers an explanation of progressive sharpening and flattening and can be applied to common practical problems such as representation normalization, 
    
[^4]: 基于神经算子流的量子场论多格采样方法

    Multi-Lattice Sampling of Quantum Field Theories via Neural Operator-based Flows. (arXiv:2401.00828v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2401.00828](http://arxiv.org/abs/2401.00828)

    本文提出了一种基于神经算子流的方法，通过近似时间相关算子，实现了在量子场论中从底层自由理论到目标理论的离散-连续归一化流。

    

    本文考虑从玻尔兹曼分布中采样离散场配置$\phi$的问题，其中$S$是某个量子场论连续欧几里得作用$\mathcal S$的格点离散化。我们将该密度近似视为底层函数密度$[\mathcal D\phi(x)]\mathcal Z^{-1}e^{-\mathcal S[\phi(x)]}$的学习算子实例。具体而言，我们提出了近似时间相关算子$\mathcal V_t$的方法，其时间积分提供了自由理论$[\mathcal D\phi(x)]\mathcal Z_0^{-1}e^{-\mathcal S_{0}[\phi(x)]}$的函数分布与目标理论$[\mathcal D\phi(x)]\mathcal Z^{-1}e^{-\mathcal S[\phi(x)]}$之间的映射。当选择特定的格点时，算子$\mathcal V_t$可以离散化为有限维的时间相关矢量场$V_t$，从而在离散格点上实现了连续的归一化流。

    We consider the problem of sampling discrete field configurations $\phi$ from the Boltzmann distribution $[d\phi] Z^{-1} e^{-S[\phi]}$, where $S$ is the lattice-discretization of the continuous Euclidean action $\mathcal S$ of some quantum field theory. Since such densities arise as the approximation of the underlying functional density $[\mathcal D\phi(x)] \mathcal Z^{-1} e^{-\mathcal S[\phi(x)]}$, we frame the task as an instance of operator learning. In particular, we propose to approximate a time-dependent operator $\mathcal V_t$ whose time integral provides a mapping between the functional distributions of the free theory $[\mathcal D\phi(x)] \mathcal Z_0^{-1} e^{-\mathcal S_{0}[\phi(x)]}$ and of the target theory $[\mathcal D\phi(x)]\mathcal Z^{-1}e^{-\mathcal S[\phi(x)]}$. Whenever a particular lattice is chosen, the operator $\mathcal V_t$ can be discretized to a finite dimensional, time-dependent vector field $V_t$ which in turn induces a continuous normalizing flow between fi
    
[^5]: 条件储层相生成的潜在扩散模型

    Latent Diffusion Model for Conditional Reservoir Facies Generation. (arXiv:2311.01968v1 [physics.geo-ph])

    [http://arxiv.org/abs/2311.01968](http://arxiv.org/abs/2311.01968)

    本研究提出了一种专门用于条件下储层相生成的潜在扩散模型，通过充分保留条件数据，生成了高保真度的储层相。它在性能上明显优于基于GANs的替代方法。

    

    在油气领域的田地开发和储层管理中，基于有限测量数据创建准确且地质真实的储层相至关重要。传统的两点地质统计方法虽然基础，但往往难以捕捉复杂的地质模式。多点统计方法提供了更大的灵活性，但也面临着挑战。随着生成对抗网络（GANs）的兴起和它们在不同领域的成功，人们开始倾向于使用它们进行储层相生成。然而，计算机视觉领域的最新进展显示了扩散模型相较于GANs的卓越性能。受此启发，提出了一种新颖的潜在扩散模型，专门用于条件下的储层相生成。该模型产生了高保真度的储层相，严格保留了条件数据。它明显优于基于GANs的替代方法。

    Creating accurate and geologically realistic reservoir facies based on limited measurements is crucial for field development and reservoir management, especially in the oil and gas sector. Traditional two-point geostatistics, while foundational, often struggle to capture complex geological patterns. Multi-point statistics offers more flexibility, but comes with its own challenges. With the rise of Generative Adversarial Networks (GANs) and their success in various fields, there has been a shift towards using them for facies generation. However, recent advances in the computer vision domain have shown the superiority of diffusion models over GANs. Motivated by this, a novel Latent Diffusion Model is proposed, which is specifically designed for conditional generation of reservoir facies. The proposed model produces high-fidelity facies realizations that rigorously preserve conditioning data. It significantly outperforms a GAN-based alternative.
    
[^6]: 生成熵神经最优传输在空间内外映射中的应用

    Generative Entropic Neural Optimal Transport To Map Within and Across Spaces. (arXiv:2310.09254v1 [stat.ML])

    [http://arxiv.org/abs/2310.09254](http://arxiv.org/abs/2310.09254)

    该论文介绍了生成熵神经最优传输在测度到测度映射中的应用，解决了处理非平方欧氏距离成本、确定性蒙格映射、映射跨不可比较空间和质量守恒约束等实际挑战。

    

    学习测度到测度的映射是机器学习中的一个关键任务，尤其在生成建模中占据重要地位。近年来，受最优传输理论启发的技术不断涌现。结合神经网络模型，这些方法统称为"神经最优传输"，将最优传输作为归纳偏好：这些映射应该针对给定的成本函数是最优的，能以节约的方式（通过最小化位移）在空间内或空间间移动点。这一原则在直观上是合理的，但往往面临几个实际挑战，需要调整最优传输工具箱：处理其他非平方欧氏距离成本的挑战，确定性状况下的蒙格映射公式会限制灵活性，映射在不可比较的空间中会带来多个挑战，最优传输固有的质量守恒约束可能对异常数据给予过多的重视。

    Learning measure-to-measure mappings is a crucial task in machine learning, featured prominently in generative modeling. Recent years have witnessed a surge of techniques that draw inspiration from optimal transport (OT) theory. Combined with neural network models, these methods collectively known as \textit{Neural OT} use optimal transport as an inductive bias: such mappings should be optimal w.r.t. a given cost function, in the sense that they are able to move points in a thrifty way, within (by minimizing displacements) or across spaces (by being isometric). This principle, while intuitive, is often confronted with several practical challenges that require adapting the OT toolbox: cost functions other than the squared-Euclidean cost can be challenging to handle, the deterministic formulation of Monge maps leaves little flexibility, mapping across incomparable spaces raises multiple challenges, while the mass conservation constraint inherent to OT can provide too much credit to outli
    
[^7]: 未知方差下的高斯均值的任意有效T检验和置信序列

    Anytime-valid t-tests and confidence sequences for Gaussian means with unknown variance. (arXiv:2310.03722v1 [math.ST])

    [http://arxiv.org/abs/2310.03722](http://arxiv.org/abs/2310.03722)

    本文提出了两种新的“e-process”和置信序列方法，分别通过替换Lai的混合方法，并分析了所得结果的宽度。

    

    在1976年，Lai构造了一个非平凡的均值$\mu$的高斯分布的置信序列，该分布的方差$\sigma$是未知的。他使用了关于$\sigma$的不适当（右Haar）混合和关于$\mu$的不适当（平坦）混合。在本文中，我们详细说明了他构建的细节，其中使用了广义的不可积分鞅和扩展的维尔不等式。尽管这确实产生了一个顺序T检验，但由于他的鞅不可积分，它并没有产生一个“e-process”。在本文中，我们为相同的设置开发了两个新的“e-process”和置信序列：一个是在缩减滤波器中的测试鞅，另一个是在规范数据滤波器中的“e-process”。这些分别是通过将Lai的平坦混合替换为高斯混合，并将对$\sigma$的右Haar混合替换为在零空间下的最大似然估计，就像在通用推断中一样。我们还分析了所得结果的宽度。

    In 1976, Lai constructed a nontrivial confidence sequence for the mean $\mu$ of a Gaussian distribution with unknown variance $\sigma$. Curiously, he employed both an improper (right Haar) mixture over $\sigma$ and an improper (flat) mixture over $\mu$. Here, we elaborate carefully on the details of his construction, which use generalized nonintegrable martingales and an extended Ville's inequality. While this does yield a sequential t-test, it does not yield an ``e-process'' (due to the nonintegrability of his martingale). In this paper, we develop two new e-processes and confidence sequences for the same setting: one is a test martingale in a reduced filtration, while the other is an e-process in the canonical data filtration. These are respectively obtained by swapping Lai's flat mixture for a Gaussian mixture, and swapping the right Haar mixture over $\sigma$ with the maximum likelihood estimate under the null, as done in universal inference. We also analyze the width of resulting 
    

