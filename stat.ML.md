# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [High-probability Convergence Bounds for Nonlinear Stochastic Gradient Descent Under Heavy-tailed Noise.](http://arxiv.org/abs/2310.18784) | 本研究探讨了一类非线性随机梯度下降方法的高概率收敛边界。对于具有Lipschitz连续梯度的强凸损失函数，即使噪声是重尾的，结果证明了对失败概率的对数依赖。这些结果适用于剪切、归一化和量化等任何具有有界输出的非线性函数。 |
| [^2] | [On the Temperature of Bayesian Graph Neural Networks for Conformal Prediction.](http://arxiv.org/abs/2310.11479) | 该论文探讨了将温度参数纳入贝叶斯图神经网络在一致预测中的优势，以提供有效的不确定性量化。 |
| [^3] | [Quantum Machine Learning on Near-Term Quantum Devices: Current State of Supervised and Unsupervised Techniques for Real-World Applications.](http://arxiv.org/abs/2307.00908) | 近期量子设备上的量子机器学习应用中，我们着重研究了监督和无监督学习在现实世界场景的应用。我们探究了当前量子硬件上的QML实现的限制，并提出了克服这些限制的技术。与经典对应物相比较，这些QML实现的性能得到了评估。 |
| [^4] | [Physics-informed neural networks with unknown measurement noise.](http://arxiv.org/abs/2211.15498) | 这篇论文提出了一种解决物理信息神经网络在存在非高斯噪声情况下失效的问题的方法，即通过同时训练一个能量模型来学习正确的噪声分布。通过多个例子的实验证明了该方法的改进性能。 |
| [^5] | [Grassmann Manifold Flows for Stable Shape Generation.](http://arxiv.org/abs/2211.02900) | 本文提出了一种利用Grassmann流形学习分布的方法，以生成稳定的形状。实验结果证明了该方法在生成高质量样本方面的优势。 |

# 详细

[^1]: 高概率收敛边界下的非线性随机梯度下降在重尾噪声下的研究

    High-probability Convergence Bounds for Nonlinear Stochastic Gradient Descent Under Heavy-tailed Noise. (arXiv:2310.18784v1 [cs.LG])

    [http://arxiv.org/abs/2310.18784](http://arxiv.org/abs/2310.18784)

    本研究探讨了一类非线性随机梯度下降方法的高概率收敛边界。对于具有Lipschitz连续梯度的强凸损失函数，即使噪声是重尾的，结果证明了对失败概率的对数依赖。这些结果适用于剪切、归一化和量化等任何具有有界输出的非线性函数。

    

    最近几个研究工作研究了随机梯度下降（SGD）及其剪切变体的高概率收敛。与普通的SGD相比，剪切SGD在实际中更加稳定，并且在理论上有对数依赖于失败概率的额外好处。然而，其他实际非线性SGD变体（如符号SGD、量化SGD和归一化SGD）的收敛性理解要少得多，这些方法实现了改进的通信效率或加速收敛。在本工作中，我们研究了一类广义非线性SGD方法的高概率收敛边界。对于具有Lipschitz连续梯度的强凸损失函数，即使噪声是重尾的，我们证明了对失败概率的对数依赖。与剪切SGD的结果相比，我们的结果更为一般，适用于具有有界输出的任何非线性函数，如剪切、归一化和量化。

    Several recent works have studied the convergence \textit{in high probability} of stochastic gradient descent (SGD) and its clipped variant. Compared to vanilla SGD, clipped SGD is practically more stable and has the additional theoretical benefit of logarithmic dependence on the failure probability. However, the convergence of other practical nonlinear variants of SGD, e.g., sign SGD, quantized SGD and normalized SGD, that achieve improved communication efficiency or accelerated convergence is much less understood. In this work, we study the convergence bounds \textit{in high probability} of a broad class of nonlinear SGD methods. For strongly convex loss functions with Lipschitz continuous gradients, we prove a logarithmic dependence on the failure probability, even when the noise is heavy-tailed. Strictly more general than the results for clipped SGD, our results hold for any nonlinearity with bounded (component-wise or joint) outputs, such as clipping, normalization, and quantizati
    
[^2]: 关于贝叶斯图神经网络在一致预测中的温度问题

    On the Temperature of Bayesian Graph Neural Networks for Conformal Prediction. (arXiv:2310.11479v1 [cs.LG])

    [http://arxiv.org/abs/2310.11479](http://arxiv.org/abs/2310.11479)

    该论文探讨了将温度参数纳入贝叶斯图神经网络在一致预测中的优势，以提供有效的不确定性量化。

    

    准确的不确定性量化对于图神经网络(GNNs)至关重要，特别是在高风险领域中经常使用GNNs的情况下。一致预测(CP)为任何黑盒模型提供了一个量化不确定性的有前途的框架。CP保证了一个预测集以所需的概率包含真实标签的形式的官方概率保证。然而，预测集的大小，即"低效率"，受到底层模型和数据生成过程的影响。另一方面，贝叶斯学习还基于估计的后验分布提供一个可信区域，但只有在模型正确指定的情况下，这个区域才是"良好校准"的。在一个最近的工作的基础上，该工作引入了一个缩放参数，用于从后验估计中构建有效的可信区域，我们的研究探讨了在CP框架中将一个温度参数纳入贝叶斯GNNs中的优势。

    Accurate uncertainty quantification in graph neural networks (GNNs) is essential, especially in high-stakes domains where GNNs are frequently employed. Conformal prediction (CP) offers a promising framework for quantifying uncertainty by providing $\textit{valid}$ prediction sets for any black-box model. CP ensures formal probabilistic guarantees that a prediction set contains a true label with a desired probability. However, the size of prediction sets, known as $\textit{inefficiency}$, is influenced by the underlying model and data generating process. On the other hand, Bayesian learning also provides a credible region based on the estimated posterior distribution, but this region is $\textit{well-calibrated}$ only when the model is correctly specified. Building on a recent work that introduced a scaling parameter for constructing valid credible regions from posterior estimate, our study explores the advantages of incorporating a temperature parameter into Bayesian GNNs within CP fra
    
[^3]: 近期量子装置上的量子机器学习: 监督和无监督技术在现实世界应用的现状

    Quantum Machine Learning on Near-Term Quantum Devices: Current State of Supervised and Unsupervised Techniques for Real-World Applications. (arXiv:2307.00908v1 [quant-ph])

    [http://arxiv.org/abs/2307.00908](http://arxiv.org/abs/2307.00908)

    近期量子设备上的量子机器学习应用中，我们着重研究了监督和无监督学习在现实世界场景的应用。我们探究了当前量子硬件上的QML实现的限制，并提出了克服这些限制的技术。与经典对应物相比较，这些QML实现的性能得到了评估。

    

    在过去十年中，量子硬件在速度、量子比特数量和量子体积方面取得了相当大的进展，量子体积被定义为在近期量子设备上可以有效实现的量子电路的最大规模。因此，在实际硬件上应用量子机器学习(QML)以实现量子优势已经有了很大的增长。在这篇综述中，我们主要关注在量子硬件上实现的选定监督和无监督学习应用，特别针对现实世界场景。我们探讨并强调了QML在量子硬件上的当前限制。我们深入讨论了各种克服这些限制的技术，如编码技术、基态结构、误差补偿和梯度方法。此外，我们评估了这些QML实现与它们的经典对应物之间的性能对比。

    The past decade has seen considerable progress in quantum hardware in terms of the speed, number of qubits and quantum volume which is defined as the maximum size of a quantum circuit that can be effectively implemented on a near-term quantum device. Consequently, there has also been a rise in the number of works based on the applications of Quantum Machine Learning (QML) on real hardware to attain quantum advantage over their classical counterparts. In this survey, our primary focus is on selected supervised and unsupervised learning applications implemented on quantum hardware, specifically targeting real-world scenarios. Our survey explores and highlights the current limitations of QML implementations on quantum hardware. We delve into various techniques to overcome these limitations, such as encoding techniques, ansatz structure, error mitigation, and gradient methods. Additionally, we assess the performance of these QML implementations in comparison to their classical counterparts
    
[^4]: 具有未知测量噪声的物理信息神经网络

    Physics-informed neural networks with unknown measurement noise. (arXiv:2211.15498v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2211.15498](http://arxiv.org/abs/2211.15498)

    这篇论文提出了一种解决物理信息神经网络在存在非高斯噪声情况下失效的问题的方法，即通过同时训练一个能量模型来学习正确的噪声分布。通过多个例子的实验证明了该方法的改进性能。

    

    物理信息神经网络(PINNs)是一种既能找到解决方案又能识别偏微分方程参数的灵活方法。大多数相关的研究都假设数据是无噪声的，或者是受弱高斯噪声污染的。我们展示了标准PINN框架在非高斯噪声情况下失效的问题，并提出了一种解决这个根本性问题的方法，即同时训练一个能量模型(Energy-Based Model, EBM)来学习正确的噪声分布。我们通过多个例子展示了我们方法的改进性能。

    Physics-informed neural networks (PINNs) constitute a flexible approach to both finding solutions and identifying parameters of partial differential equations. Most works on the topic assume noiseless data, or data contaminated by weak Gaussian noise. We show that the standard PINN framework breaks down in case of non-Gaussian noise. We give a way of resolving this fundamental issue and we propose to jointly train an energy-based model (EBM) to learn the correct noise distribution. We illustrate the improved performance of our approach using multiple examples.
    
[^5]: Grassmann流形流用于稳定形状生成

    Grassmann Manifold Flows for Stable Shape Generation. (arXiv:2211.02900v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2211.02900](http://arxiv.org/abs/2211.02900)

    本文提出了一种利用Grassmann流形学习分布的方法，以生成稳定的形状。实验结果证明了该方法在生成高质量样本方面的优势。

    

    最近，机器学习的研究集中在利用特定流形中的对称性作为归纳偏差的方法上。Grassmann流形提供了处理以形状空间表示的基本形状的能力，实现了稳定的形状分析。本文提出了一种新的方法，通过连续的归一化流在Grassmann流形上建立学习分布的理论基础，明确的目标是生成稳定的形状。我们的方法通过在Grassmann流形内学习和生成，有效地消除了旋转和翻转等外部变换的影响，从而实现更稳健的生成，以适应对象的基本形状信息。实验结果表明，所提出的方法能够通过捕捉数据结构生成高质量的样本。此外，所提出的方法在t方面显著优于最先进的方法。

    Recently, studies on machine learning have focused on methods that use symmetry implicit in a specific manifold as an inductive bias. Grassmann manifolds provide the ability to handle fundamental shapes represented as shape spaces, enabling stable shape analysis. In this paper, we present a novel approach in which we establish the theoretical foundations for learning distributions on the Grassmann manifold via continuous normalization flows, with the explicit goal of generating stable shapes. Our approach facilitates more robust generation by effectively eliminating the influence of extraneous transformations, such as rotations and inversions, through learning and generating within a Grassmann manifolds designed to accommodate the essential shape information of the object. The experimental results indicated that the proposed method can generate high-quality samples by capturing the data structure. Furthermore, the proposed method significantly outperformed state-of-the-art methods in t
    

