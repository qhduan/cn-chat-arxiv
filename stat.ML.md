# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Non-negative Contrastive Learning](https://arxiv.org/abs/2403.12459) | 非负对比学习(NCL)是对非负矩阵分解(NMF)的重新演绎，通过对特征施加非负约束来获得可解释的特征，保留了NMF的可解释属性，从而得到比标准对比学习(CL)更稀疏和解耦的表示 |
| [^2] | [A unified Fourier slice method to derive ridgelet transform for a variety of depth-2 neural networks](https://arxiv.org/abs/2402.15984) | 通过使用傅里叶表达式导出尖峰变换，实现了对各种现代神经网络的描述和分析。 |
| [^3] | [How to validate average calibration for machine learning regression tasks ?](https://arxiv.org/abs/2402.10043) | 本文提出了两种验证机器学习回归任务平均校准性的方法，将校准误差与平均绝对误差之间的差值和将平均平方z-分数与1进行比较。研究发现，前者对不确定性分布敏感，而后者在该方面提供了最可靠的方法。 |
| [^4] | [Stochastic Smoothed Gradient Descent Ascent for Federated Minimax Optimization.](http://arxiv.org/abs/2311.00944) | 本论文提出了一种新算法，FESS-GDA，利用平滑技术进行联邦极小极大优化。通过解决不同类型的联邦极小极大问题，我们证明了FESS-GDA的收敛性，并展示了其在实际联邦学习任务中的实际效果。 |
| [^5] | [High-probability Convergence Bounds for Nonlinear Stochastic Gradient Descent Under Heavy-tailed Noise.](http://arxiv.org/abs/2310.18784) | 本研究探讨了一类非线性随机梯度下降方法的高概率收敛边界。对于具有Lipschitz连续梯度的强凸损失函数，即使噪声是重尾的，结果证明了对失败概率的对数依赖。这些结果适用于剪切、归一化和量化等任何具有有界输出的非线性函数。 |
| [^6] | [Differential Equation Scaling Limits of Shaped and Unshaped Neural Networks.](http://arxiv.org/abs/2310.12079) | 最近的研究发现，对于具有形状激活函数的神经网络，其缩放极限可以由微分方程描述。然而，关于未经形状处理的神经网络的信息尚不明确。本文研究了两种未经形状处理的网络，发现它们也可以由类似的微分方程来描述，并给出了它们的一些特征。 |
| [^7] | [RandCom: Random Communication Skipping Method for Decentralized Stochastic Optimization.](http://arxiv.org/abs/2310.07983) | RandCom是一种去中心化的随机通信跳跃方法，能够在分布式优化中通过概率性本地更新减少通信开销，并在不同的设置中实现线性加速。 |
| [^8] | [Lion Secretly Solves Constrained Optimization: As Lyapunov Predicts.](http://arxiv.org/abs/2310.05898) | Lion是通过程序搜索发现的新优化器，在训练大型AI模型方面表现出有希望的结果，具有更高的内存效率。尽管其理论基础不明确，但基于连续时间和离散时间分析，我们证明Lion是一种理论上新颖且有原则的方法，可在最小化一般损失函数的同时强制执行边界约束。 |
| [^9] | [Learning multi-modal generative models with permutation-invariant encoders and tighter variational bounds.](http://arxiv.org/abs/2309.00380) | 本文提出了一种用于多模态数据的深度潜变量模型，并开发了更灵活的编码特征聚合方案，能够紧密地下界数据对数似然。 |
| [^10] | [Noise Stability Optimization for Flat Minima with Optimal Convergence Rates.](http://arxiv.org/abs/2306.08553) | 本文提出了一个SGD-like算法，注入随机噪声并利用分布对称性来减少方差，以寻找具有低海森矩阵迹的平坦极小值，同时提供了收敛速率分析。 |
| [^11] | [Inverse Cubature and Quadrature Kalman filters.](http://arxiv.org/abs/2303.10322) | 本文提出了逆多面积积分卡尔曼滤波器（I-CKF）和逆多项式积分卡尔曼滤波器（I-QKF），用于解决高度非线性的系统模型的逆认知问题，并在指数-平均-二次有界性意义下推导了其的随机稳定性条件。数值实验证明了其高的估计精度和递归Cram\'{e}r-Rao下界相当。 |

# 详细

[^1]: 非负对比学习

    Non-negative Contrastive Learning

    [https://arxiv.org/abs/2403.12459](https://arxiv.org/abs/2403.12459)

    非负对比学习(NCL)是对非负矩阵分解(NMF)的重新演绎，通过对特征施加非负约束来获得可解释的特征，保留了NMF的可解释属性，从而得到比标准对比学习(CL)更稀疏和解耦的表示

    

    深度表示在以黑盒方式转移到下游任务时表现出了良好的性能。然而，它们固有的不可解释性仍然是一个重大挑战，因为这些特征通常对人类理解而言是不透明的。在本文中，我们提出了非负对比学习（NCL），这是对非负矩阵分解（NMF）的复兴，旨在得出可解释的特征。NCL的力量在于强制将非负约束应用于特征，这让人想起NMF能够提取与样本集群紧密对齐的特征的能力。NCL不仅在数学上与NMF目标很好地对齐，而且保留了NMF的可解释属性，使得与标准对比学习（CL）相比，得到了更加稀疏和解耦的表示。从理论上，我们为NCL的可识别性和下游泛化性能提供了保证。从经验上看，我们展示了这些

    arXiv:2403.12459v1 Announce Type: cross  Abstract: Deep representations have shown promising performance when transferred to downstream tasks in a black-box manner. Yet, their inherent lack of interpretability remains a significant challenge, as these features are often opaque to human understanding. In this paper, we propose Non-negative Contrastive Learning (NCL), a renaissance of Non-negative Matrix Factorization (NMF) aimed at deriving interpretable features. The power of NCL lies in its enforcement of non-negativity constraints on features, reminiscent of NMF's capability to extract features that align closely with sample clusters. NCL not only aligns mathematically well with an NMF objective but also preserves NMF's interpretability attributes, resulting in a more sparse and disentangled representation compared to standard contrastive learning (CL). Theoretically, we establish guarantees on the identifiability and downstream generalization of NCL. Empirically, we show that these 
    
[^2]: 用统一的傅里叶切片方法导出一种适用于多种深度-2神经网络的尖峰变换

    A unified Fourier slice method to derive ridgelet transform for a variety of depth-2 neural networks

    [https://arxiv.org/abs/2402.15984](https://arxiv.org/abs/2402.15984)

    通过使用傅里叶表达式导出尖峰变换，实现了对各种现代神经网络的描述和分析。

    

    研究神经网络参数时，研究参数分布比研究每个神经元的参数更容易。尖峰变换是一个伪逆算子，将给定函数 $f$ 映射到参数分布 $\gamma$，使得网络 $\mathtt{NN}[\gamma]$ 能够重现 $f$，即 $\mathtt{NN}[\gamma]=f$。在欧氏空间上的深度-2全连接网络中，已发现了尖峰变换的闭合形式表达式，因此我们可以描述参数的分布。然而，对于多种现代神经网络架构，尚不知道闭合形式表达式。本文介绍了一种使用傅里叶表达式的系统方法，用于推导各种现代网络的尖峰变换，例如有限域 $\mathbb{F}_p$ 上的网络、抽象希尔伯特空间 $\mathcal{H}$ 上的群卷积网络，以及非紧致对称的全连接网络。

    arXiv:2402.15984v1 Announce Type: new  Abstract: To investigate neural network parameters, it is easier to study the distribution of parameters than to study the parameters in each neuron. The ridgelet transform is a pseudo-inverse operator that maps a given function $f$ to the parameter distribution $\gamma$ so that a network $\mathtt{NN}[\gamma]$ reproduces $f$, i.e. $\mathtt{NN}[\gamma]=f$. For depth-2 fully-connected networks on a Euclidean space, the ridgelet transform has been discovered up to the closed-form expression, thus we could describe how the parameters are distributed. However, for a variety of modern neural network architectures, the closed-form expression has not been known. In this paper, we explain a systematic method using Fourier expressions to derive ridgelet transforms for a variety of modern networks such as networks on finite fields $\mathbb{F}_p$, group convolutional networks on abstract Hilbert space $\mathcal{H}$, fully-connected networks on noncompact symm
    
[^3]: 如何验证机器学习回归任务的平均校准性？

    How to validate average calibration for machine learning regression tasks ?

    [https://arxiv.org/abs/2402.10043](https://arxiv.org/abs/2402.10043)

    本文提出了两种验证机器学习回归任务平均校准性的方法，将校准误差与平均绝对误差之间的差值和将平均平方z-分数与1进行比较。研究发现，前者对不确定性分布敏感，而后者在该方面提供了最可靠的方法。

    

    机器学习回归任务的平均校准性可以通过两种方式进行测试。一种方式是将校准误差（CE）估计为平均绝对误差（MSE）与平均方差（MV）或平均平方不确定性之间的差值。另一种方式是将平均平方z-分数或缩放误差（ZMS）与1进行比较。两种方法可能得出不同的结论，正如来自最近的机器学习不确定性量化文献中的数据集集合所示。研究表明，CE对不确定性分布非常敏感，特别是对于离群不确定性的存在，因此无法可靠地用于校准测试。相比之下，ZMS统计量不具有这种敏感性问题，在这种情况下提供了最可靠的方法。文章还讨论了对条件校准验证的影响。

    arXiv:2402.10043v1 Announce Type: cross  Abstract: Average calibration of the uncertainties of machine learning regression tasks can be tested in two ways. One way is to estimate the calibration error (CE) as the difference between the mean absolute error (MSE) and the mean variance (MV) or mean squared uncertainty. The alternative is to compare the mean squared z-scores or scaled errors (ZMS) to 1. Both approaches might lead to different conclusion, as illustrated on an ensemble of datasets from the recent machine learning uncertainty quantification literature. It is shown here that the CE is very sensitive to the distribution of uncertainties, and notably to the presence of outlying uncertainties, and that it cannot be used reliably for calibration testing. By contrast, the ZMS statistic does not present this sensitivity issue and offers the most reliable approach in this context. Implications for the validation of conditional calibration are discussed.
    
[^4]: 基于随机平滑梯度上升下降法的联邦极小极大优化研究

    Stochastic Smoothed Gradient Descent Ascent for Federated Minimax Optimization. (arXiv:2311.00944v1 [stat.ML])

    [http://arxiv.org/abs/2311.00944](http://arxiv.org/abs/2311.00944)

    本论文提出了一种新算法，FESS-GDA，利用平滑技术进行联邦极小极大优化。通过解决不同类型的联邦极小极大问题，我们证明了FESS-GDA的收敛性，并展示了其在实际联邦学习任务中的实际效果。

    

    近年来，由于其在各种机器学习任务中的广泛应用，联邦极小极大优化引起了越来越多的关注。虽然在集中非凸极小极大优化中，平滑交替梯度上升下降（Smoothed-AGDA）已经证明了其成功之处，但平滑技术在联邦设置中的作用和是否有所帮助尚未被探究。在本文中，我们提出了一种新算法，称为联邦随机平滑梯度上升下降（FESS-GDA），该算法利用平滑技术进行联邦极小极大优化。我们证明了FESS-GDA可以统一解决几类联邦极小极大问题，并为这些设置提供了新的或更好的收敛结果分析。我们展示了FESS-GDA在实际联邦学习任务中，如生成对抗网络（GANs）的训练和公平分类中的实际效率。

    In recent years, federated minimax optimization has attracted growing interest due to its extensive applications in various machine learning tasks. While Smoothed Alternative Gradient Descent Ascent (Smoothed-AGDA) has proved its success in centralized nonconvex minimax optimization, how and whether smoothing technique could be helpful in federated setting remains unexplored. In this paper, we propose a new algorithm termed Federated Stochastic Smoothed Gradient Descent Ascent (FESS-GDA), which utilizes the smoothing technique for federated minimax optimization. We prove that FESS-GDA can be uniformly used to solve several classes of federated minimax problems and prove new or better analytical convergence results for these settings. We showcase the practical efficiency of FESS-GDA in practical federated learning tasks of training generative adversarial networks (GANs) and fair classification.
    
[^5]: 高概率收敛边界下的非线性随机梯度下降在重尾噪声下的研究

    High-probability Convergence Bounds for Nonlinear Stochastic Gradient Descent Under Heavy-tailed Noise. (arXiv:2310.18784v1 [cs.LG])

    [http://arxiv.org/abs/2310.18784](http://arxiv.org/abs/2310.18784)

    本研究探讨了一类非线性随机梯度下降方法的高概率收敛边界。对于具有Lipschitz连续梯度的强凸损失函数，即使噪声是重尾的，结果证明了对失败概率的对数依赖。这些结果适用于剪切、归一化和量化等任何具有有界输出的非线性函数。

    

    最近几个研究工作研究了随机梯度下降（SGD）及其剪切变体的高概率收敛。与普通的SGD相比，剪切SGD在实际中更加稳定，并且在理论上有对数依赖于失败概率的额外好处。然而，其他实际非线性SGD变体（如符号SGD、量化SGD和归一化SGD）的收敛性理解要少得多，这些方法实现了改进的通信效率或加速收敛。在本工作中，我们研究了一类广义非线性SGD方法的高概率收敛边界。对于具有Lipschitz连续梯度的强凸损失函数，即使噪声是重尾的，我们证明了对失败概率的对数依赖。与剪切SGD的结果相比，我们的结果更为一般，适用于具有有界输出的任何非线性函数，如剪切、归一化和量化。

    Several recent works have studied the convergence \textit{in high probability} of stochastic gradient descent (SGD) and its clipped variant. Compared to vanilla SGD, clipped SGD is practically more stable and has the additional theoretical benefit of logarithmic dependence on the failure probability. However, the convergence of other practical nonlinear variants of SGD, e.g., sign SGD, quantized SGD and normalized SGD, that achieve improved communication efficiency or accelerated convergence is much less understood. In this work, we study the convergence bounds \textit{in high probability} of a broad class of nonlinear SGD methods. For strongly convex loss functions with Lipschitz continuous gradients, we prove a logarithmic dependence on the failure probability, even when the noise is heavy-tailed. Strictly more general than the results for clipped SGD, our results hold for any nonlinearity with bounded (component-wise or joint) outputs, such as clipping, normalization, and quantizati
    
[^6]: 形状和非形状神经网络的微分方程缩放极限

    Differential Equation Scaling Limits of Shaped and Unshaped Neural Networks. (arXiv:2310.12079v1 [stat.ML])

    [http://arxiv.org/abs/2310.12079](http://arxiv.org/abs/2310.12079)

    最近的研究发现，对于具有形状激活函数的神经网络，其缩放极限可以由微分方程描述。然而，关于未经形状处理的神经网络的信息尚不明确。本文研究了两种未经形状处理的网络，发现它们也可以由类似的微分方程来描述，并给出了它们的一些特征。

    

    最近对具有形状激活函数（即随着网络规模增大而缩放的激活函数）的神经网络进行的分析表明，它们具有由微分方程描述的缩放极限。然而，这些结果不预先告诉我们关于“普通”非形状网络的任何信息，其中激活函数在网络规模增大时保持不变。在本文中，我们针对两种类型的非形状网络找到了类似的基于微分方程的渐近特征描述。首先，我们证明以下两种架构在初始化时会收敛到相同的无限深度和宽度极限：（i）带有残差分支上的 $d^{-1/2}$ 因子的全连接 ResNet，其中 $d$ 是网络的深度；（ii）带有深度 $d \ll$ 宽度 $n$ 和形状 ReLU 激活函数 (activation) 的多层感知机 (MLP)，以 $d^{-1/2}$ 的速率。其次，对于初始化的非形状 MLP，我们推导了层间相关性的一阶渐近修正。特别地，如果 $\rho_\ell$ 是第 $\ell$ 层的相关性，则...

    Recent analyses of neural networks with shaped activations (i.e. the activation function is scaled as the network size grows) have led to scaling limits described by differential equations. However, these results do not a priori tell us anything about "ordinary" unshaped networks, where the activation is unchanged as the network size grows. In this article, we find similar differential equation based asymptotic characterization for two types of unshaped networks.  Firstly, we show that the following two architectures converge to the same infinite-depth-and-width limit at initialization: (i) a fully connected ResNet with a $d^{-1/2}$ factor on the residual branch, where $d$ is the network depth. (ii) a multilayer perceptron (MLP) with depth $d \ll$ width $n$ and shaped ReLU activation at rate $d^{-1/2}$.  Secondly, for an unshaped MLP at initialization, we derive the first order asymptotic correction to the layerwise correlation. In particular, if $\rho_\ell$ is the correlation at layer
    
[^7]: RandCom：去中心化随机通信跳跃方法用于分布式随机优化

    RandCom: Random Communication Skipping Method for Decentralized Stochastic Optimization. (arXiv:2310.07983v1 [cs.LG])

    [http://arxiv.org/abs/2310.07983](http://arxiv.org/abs/2310.07983)

    RandCom是一种去中心化的随机通信跳跃方法，能够在分布式优化中通过概率性本地更新减少通信开销，并在不同的设置中实现线性加速。

    

    具有随机通信跳过的分布式优化方法因其在加速通信复杂性方面具有的优势而受到越来越多的关注。然而，现有的研究主要集中在强凸确定性设置的集中式通信协议上。在本研究中，我们提出了一种名为RandCom的分布式优化方法，它采用了概率性的本地更新。我们分析了RandCom在随机非凸、凸和强凸设置中的性能，并证明了它能够通过通信概率来渐近地减少通信开销。此外，我们证明当节点数量增加时，RandCom能够实现线性加速。在随机强凸设置中，我们进一步证明了RandCom可以通过独立于网络的步长实现线性加速。此外，我们将RandCom应用于联邦学习，并提供了关于实现线性加速的潜力的积极结果。

    Distributed optimization methods with random communication skips are gaining increasing attention due to their proven benefits in accelerating communication complexity. Nevertheless, existing research mainly focuses on centralized communication protocols for strongly convex deterministic settings. In this work, we provide a decentralized optimization method called RandCom, which incorporates probabilistic local updates. We analyze the performance of RandCom in stochastic non-convex, convex, and strongly convex settings and demonstrate its ability to asymptotically reduce communication overhead by the probability of communication. Additionally, we prove that RandCom achieves linear speedup as the number of nodes increases. In stochastic strongly convex settings, we further prove that RandCom can achieve linear speedup with network-independent stepsizes. Moreover, we apply RandCom to federated learning and provide positive results concerning the potential for achieving linear speedup and
    
[^8]: 狮子秘密地解决受限制优化问题：正如李雅普诺夫所预测的。

    Lion Secretly Solves Constrained Optimization: As Lyapunov Predicts. (arXiv:2310.05898v1 [cs.LG])

    [http://arxiv.org/abs/2310.05898](http://arxiv.org/abs/2310.05898)

    Lion是通过程序搜索发现的新优化器，在训练大型AI模型方面表现出有希望的结果，具有更高的内存效率。尽管其理论基础不明确，但基于连续时间和离散时间分析，我们证明Lion是一种理论上新颖且有原则的方法，可在最小化一般损失函数的同时强制执行边界约束。

    

    通过程序搜索发现的新优化器Lion（进化的符号动量）在训练大型AI模型方面显示出有希望的结果。它在训练效果上与AdamW相当或更好，并具有更高的内存效率。正如我们可以从随机搜索程序的结果中期待的，Lion集成了几个现有算法的元素，包括符号动量、独立的权重衰减、Polak和Nesterov动量，但又不属于任何现有的理论基础优化器类别。因此，尽管Lion作为广泛任务的通用优化器表现良好，但其理论基础仍然不明确。这种缺乏理论的明确性限制了进一步增强和扩展Lion的可能性。本文旨在揭开Lion的神秘面纱。基于连续时间和离散时间分析，我们证明Lion是一种理论上新颖且有原则的方法，可在最小化一般损失函数$f(x)$的同时强制执行边界约束。

    Lion (Evolved Sign Momentum), a new optimizer discovered through program search, has shown promising results in training large AI models. It performs comparably or favorably to AdamW but with greater memory efficiency. As we can expect from the results of a random search program, Lion incorporates elements from several existing algorithms, including signed momentum, decoupled weight decay, Polak, and Nesterov momentum, but does not fit into any existing category of theoretically grounded optimizers. Thus, even though Lion appears to perform well as a general-purpose optimizer for a wide range of tasks, its theoretical basis remains uncertain. This lack of theoretical clarity limits opportunities to further enhance and expand Lion's efficacy.  This work aims to demystify Lion. Based on both continuous-time and discrete-time analysis, we demonstrate that Lion is a theoretically novel and principled approach for minimizing a general loss function $f(x)$ while enforcing a bound constraint 
    
[^9]: 用排序不变的编码器和更紧的变分边界学习多模态生成模型

    Learning multi-modal generative models with permutation-invariant encoders and tighter variational bounds. (arXiv:2309.00380v1 [stat.ML])

    [http://arxiv.org/abs/2309.00380](http://arxiv.org/abs/2309.00380)

    本文提出了一种用于多模态数据的深度潜变量模型，并开发了更灵活的编码特征聚合方案，能够紧密地下界数据对数似然。

    

    设计用于多模态数据的深度潜变量模型一直是机器学习研究中的一个重要主题。多模态变分自编码器 (VAE) 是一种常用的生成模型类别，它学习能够共同解释多种模态的潜在表示。各种客观函数已被提出用于这样的模型，往往以多模态数据对数似然的下界以及信息论方面的考虑为动机。为了对不同模态子集进行编码，我们经常使用并展示了产品型专家 (PoE) 或者混合型专家 (MoE) 聚合方案，这些方案在生成质量或者多模态一致性等方面具有不同的权衡。在本研究中，我们考虑了一个能够紧密地下界数据对数似然的变分边界。我们通过将不同模态的编码特征组合起来，开发了更灵活的聚合方案，这些方案推广了 PoE 或者 MoE 方法。

    Devising deep latent variable models for multi-modal data has been a long-standing theme in machine learning research. Multi-modal Variational Autoencoders (VAEs) have been a popular generative model class that learns latent representations which jointly explain multiple modalities. Various objective functions for such models have been suggested, often motivated as lower bounds on the multi-modal data log-likelihood or from information-theoretic considerations. In order to encode latent variables from different modality subsets, Product-of-Experts (PoE) or Mixture-of-Experts (MoE) aggregation schemes have been routinely used and shown to yield different trade-offs, for instance, regarding their generative quality or consistency across multiple modalities. In this work, we consider a variational bound that can tightly lower bound the data log-likelihood. We develop more flexible aggregation schemes that generalise PoE or MoE approaches by combining encoded features from different modali
    
[^10]: 噪声稳定优化对于具有最优收敛率的平坦极小值的影响

    Noise Stability Optimization for Flat Minima with Optimal Convergence Rates. (arXiv:2306.08553v1 [cs.LG])

    [http://arxiv.org/abs/2306.08553](http://arxiv.org/abs/2306.08553)

    本文提出了一个SGD-like算法，注入随机噪声并利用分布对称性来减少方差，以寻找具有低海森矩阵迹的平坦极小值，同时提供了收敛速率分析。

    

    本文研究通过加入加权扰动来找到平坦的极小值。给定一个非凸函数$f:\mathbb{R}^d\rightarrow \mathbb{R}$和一个$d$维分布$\mathcal{P}$，我们扰动$f$的权重，并定义$F(W)=\mathbb{E}[f({W+U})]$，其中$U$是一个从$\mathcal{P}$中随机抽取的样本。这个过程通过$f$的海森矩阵的迹来诱导正则化，以适应于小的、各向同性的高斯扰动。因此，加权扰动的函数偏向于带有低海森矩阵迹的极小值。本文提出了一种类似于SGD的算法，在计算梯度之前注入随机噪声，同时利用$\mathcal{P}$的对称性来减少方差。我们还提供了严格的分析，证明了...

    We consider finding flat, local minimizers by adding average weight perturbations. Given a nonconvex function $f: \mathbb{R}^d \rightarrow \mathbb{R}$ and a $d$-dimensional distribution $\mathcal{P}$ which is symmetric at zero, we perturb the weight of $f$ and define $F(W) = \mathbb{E}[f({W + U})]$, where $U$ is a random sample from $\mathcal{P}$. This injection induces regularization through the Hessian trace of $f$ for small, isotropic Gaussian perturbations. Thus, the weight-perturbed function biases to minimizers with low Hessian trace. Several prior works have studied settings related to this weight-perturbed function by designing algorithms to improve generalization. Still, convergence rates are not known for finding minima under the average perturbations of the function $F$. This paper considers an SGD-like algorithm that injects random noise before computing gradients while leveraging the symmetry of $\mathcal{P}$ to reduce variance. We then provide a rigorous analysis, showing
    
[^11]: 逆卡尔曼滤波器和逆多面积积分卡尔曼滤波器

    Inverse Cubature and Quadrature Kalman filters. (arXiv:2303.10322v1 [math.OC])

    [http://arxiv.org/abs/2303.10322](http://arxiv.org/abs/2303.10322)

    本文提出了逆多面积积分卡尔曼滤波器（I-CKF）和逆多项式积分卡尔曼滤波器（I-QKF），用于解决高度非线性的系统模型的逆认知问题，并在指数-平均-二次有界性意义下推导了其的随机稳定性条件。数值实验证明了其高的估计精度和递归Cram\'{e}r-Rao下界相当。

    

    反对抗系统研究的最新进展已经导致开发了逆随机滤波器，这些滤波器被防御者用来推断其对手可能学到的信息。早期的作品通过针对线性和非线性高斯状态空间模型分别提出了逆卡尔曼滤波器（I-KF）和逆扩展KF（I-EKF）来解决这个逆认知问题。然而，在实践中，很多反对抗性设置会涉及高度非线性的系统模型，其中EKF的线性化常常会失败。在本文中，我们考虑了有效的数值积分技术来解决这种非线性问题，并因此开发了逆多面积积分卡尔曼滤波器（I-CKF）和逆多项式积分卡尔曼滤波器（I-QKF）。我们在指数-平均-二次有界性意义下推导了所提出的滤波器的随机稳定性条件。数值实验证明了我们的I-CKF和I-QKF的估计精度，跟递归Cram\'{e}r-Rao下界作为基准。

    Recent developments in counter-adversarial system research have led to the development of inverse stochastic filters that are employed by a defender to infer the information its adversary may have learned. Prior works addressed this inverse cognition problem by proposing inverse Kalman filter (I-KF) and inverse extended KF (I-EKF), respectively, for linear and non-linear Gaussian state-space models. However, in practice, many counter-adversarial settings involve highly non-linear system models, wherein EKF's linearization often fails. In this paper, we consider the efficient numerical integration techniques to address such nonlinearities and, to this end, develop inverse cubature KF (I-CKF) and inverse quadrature KF (I-QKF). We derive the stochastic stability conditions for the proposed filters in the exponential-mean-squared-boundedness sense. Numerical experiments demonstrate the estimation accuracy of our I-CKF and I-QKF with the recursive Cram\'{e}r-Rao lower bound as a benchmark.
    

