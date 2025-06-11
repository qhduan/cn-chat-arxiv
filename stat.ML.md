# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Nonsmooth Nonparametric Regression via Fractional Laplacian Eigenmaps](https://arxiv.org/abs/2402.14985) | 通过分数拉普拉斯特征映射，开发了针对真实回归函数非光滑情况的非参数回归方法，成功处理了该函数类在$L_2$-分数 Sobolev 空间中的特性，并证明了误差上界为$n^{-\frac{2s}{2s+d}}$。 |
| [^2] | [Spectral invariance and maximality properties of the frequency spectrum of quantum neural networks](https://arxiv.org/abs/2402.14515) | 量子神经网络研究了频谱的极大性质，证明了在一类模型中存在极大结果，以及在一些条件下存在保持频谱的光谱不变性，解释了文献中观察到的结果对称性。 |
| [^3] | [High-Dimensional Independence Testing via Maximum and Average Distance Correlations](https://arxiv.org/abs/2001.01095) | 本文介绍并研究了利用最大和平均距离相关性进行高维度独立性检测的方法，并提出了一种快速卡方检验的程序。该方法适用于欧氏距离和高斯核，具有较好的实证表现和广泛的应用场景。 |
| [^4] | [General Loss Functions Lead to (Approximate) Interpolation in High Dimensions.](http://arxiv.org/abs/2303.07475) | 本研究提供了一般凸损失函数和超参数化阶段下梯度下降的隐含偏差的近似表征。具体而言是近似最小范数插值。 |
| [^5] | [GradSkip: Communication-Accelerated Local Gradient Methods with Better Computational Complexity.](http://arxiv.org/abs/2210.16402) | 本文研究了一类分布式优化算法，通过允许具有“次要”数据的客户端在本地执行较少的训练步骤来减轻高通信成本，这一方法可在强凸区域内实现可证明的通信加速。 |
| [^6] | [Precise High-Dimensional Asymptotics for Quantifying Heterogeneous Transfers.](http://arxiv.org/abs/2010.11750) | 本文利用随机矩阵理论在线性回归设置中，对于具有两个任务的高维情况下的常用估计量的超额风险进行了精确渐近分析。 |

# 详细

[^1]: 通过分数拉普拉斯特征映射进行非光滑非参数回归

    Nonsmooth Nonparametric Regression via Fractional Laplacian Eigenmaps

    [https://arxiv.org/abs/2402.14985](https://arxiv.org/abs/2402.14985)

    通过分数拉普拉斯特征映射，开发了针对真实回归函数非光滑情况的非参数回归方法，成功处理了该函数类在$L_2$-分数 Sobolev 空间中的特性，并证明了误差上界为$n^{-\frac{2s}{2s+d}}$。

    

    我们为真实回归函数不一定平滑的情况开发了非参数回归方法。具体来说，我们的方法使用了分数拉普拉斯，并旨在处理真实回归函数位于$L_2$-分数 Sobolev 空间（阶数为$s\in (0,1)$）的情况。该函数类是一个 Hilbert 空间，位于平方可积函数空间和一阶 Sobolev 空间之间，包括分数幂函数、分段常数或多项式函数以及尖峰函数作为典型示例。对于我们提出的方法，我们证明了关于样本内均方估计误差的上界，具有$n^{-\frac{2s}{2s+d}}$的阶，其中$d$是维数，$s$是前述顺序参数，$n$是观测数量。我们还提供了初步的实证结果，验证了所开发方法的实际表现。

    arXiv:2402.14985v1 Announce Type: cross  Abstract: We develop nonparametric regression methods for the case when the true regression function is not necessarily smooth. More specifically, our approach is using the fractional Laplacian and is designed to handle the case when the true regression function lies in an $L_2$-fractional Sobolev space with order $s\in (0,1)$. This function class is a Hilbert space lying between the space of square-integrable functions and the first-order Sobolev space consisting of differentiable functions. It contains fractional power functions, piecewise constant or polynomial functions and bump function as canonical examples. For the proposed approach, we prove upper bounds on the in-sample mean-squared estimation error of order $n^{-\frac{2s}{2s+d}}$, where $d$ is the dimension, $s$ is the aforementioned order parameter and $n$ is the number of observations. We also provide preliminary empirical results validating the practical performance of the developed
    
[^2]: 量子神经网络频谱的光谱不变性和极大性质

    Spectral invariance and maximality properties of the frequency spectrum of quantum neural networks

    [https://arxiv.org/abs/2402.14515](https://arxiv.org/abs/2402.14515)

    量子神经网络研究了频谱的极大性质，证明了在一类模型中存在极大结果，以及在一些条件下存在保持频谱的光谱不变性，解释了文献中观察到的结果对称性。

    

    量子神经网络（QNNs）是量子机器学习领域的热门方法，由于其与变分量子电路的密切联系，使其成为在噪声中间尺度量子（NISQ）设备上进行实际应用的有前途的候选方法。QNN可以表示为有限傅里叶级数，其中频率集被称为频谱。我们分析了这个频谱并证明，对于一大类模型，存在各种极大性结果。此外，我们证明在一些温和条件下，存在一个保持频谱的具有相同面积$A = RL$的模型类之间的双射，其中$R$表示量子比特数量，$L$表示层数，我们因此称之为面积保持变换下的光谱不变性。通过这个，我们解释了文献中经常观察到的在结果中$R$和$L$的对称性，并展示了最大频谱的依赖性

    arXiv:2402.14515v1 Announce Type: cross  Abstract: Quantum Neural Networks (QNNs) are a popular approach in Quantum Machine Learning due to their close connection to Variational Quantum Circuits, making them a promising candidate for practical applications on Noisy Intermediate-Scale Quantum (NISQ) devices. A QNN can be expressed as a finite Fourier series, where the set of frequencies is called the frequency spectrum. We analyse this frequency spectrum and prove, for a large class of models, various maximality results. Furthermore, we prove that under some mild conditions there exists a bijection between classes of models with the same area $A = RL$ that preserves the frequency spectrum, where $R$ denotes the number of qubits and $L$ the number of layers, which we consequently call spectral invariance under area-preserving transformations. With this we explain the symmetry in $R$ and $L$ in the results often observed in the literature and show that the maximal frequency spectrum depen
    
[^3]: 高维度独立性检测: 通过最大和平均距离相关性

    High-Dimensional Independence Testing via Maximum and Average Distance Correlations

    [https://arxiv.org/abs/2001.01095](https://arxiv.org/abs/2001.01095)

    本文介绍并研究了利用最大和平均距离相关性进行高维度独立性检测的方法，并提出了一种快速卡方检验的程序。该方法适用于欧氏距离和高斯核，具有较好的实证表现和广泛的应用场景。

    

    本文介绍并研究了利用最大和平均距离相关性进行多元独立性检测的方法。我们在高维环境中表征了它们相对于边际相关维度数量的一致性特性，评估了每个检验统计量的优势，检查了它们各自的零分布，并提出了一种基于快速卡方检验的检测程序。得出的检验是非参数的，并适用于欧氏距离和高斯核作为底层度量。为了更好地理解所提出的测试的实际使用情况，我们在各种多元相关场景中评估了最大距离相关性、平均距离相关性和原始距离相关性的实证表现，同时进行了一个真实数据实验，以检测人类血浆中不同癌症类型和肽水平的存在。

    This paper introduces and investigates the utilization of maximum and average distance correlations for multivariate independence testing. We characterize their consistency properties in high-dimensional settings with respect to the number of marginally dependent dimensions, assess the advantages of each test statistic, examine their respective null distributions, and present a fast chi-square-based testing procedure. The resulting tests are non-parametric and applicable to both Euclidean distance and the Gaussian kernel as the underlying metric. To better understand the practical use cases of the proposed tests, we evaluate the empirical performance of the maximum distance correlation, average distance correlation, and the original distance correlation across various multivariate dependence scenarios, as well as conduct a real data experiment to test the presence of various cancer types and peptide levels in human plasma.
    
[^4]: 一般损失函数在高维情况下导致（近似）插值

    General Loss Functions Lead to (Approximate) Interpolation in High Dimensions. (arXiv:2303.07475v1 [stat.ML])

    [http://arxiv.org/abs/2303.07475](http://arxiv.org/abs/2303.07475)

    本研究提供了一般凸损失函数和超参数化阶段下梯度下降的隐含偏差的近似表征。具体而言是近似最小范数插值。

    

    我们提供了一个统一的框架，适用于一般凸损失函数和超参数化阶段的二元和多元分类设置，以近似地表征梯度下降的隐含偏差。具体而言，我们展示了在高维情况下梯度下降的隐含偏差近似于最小范数插值，最小范数插值来自于对平方损失的训练。与之前专门针对指数尾损失并使用中间支持向量机公式的研究不同，我们的框架直接基于Ji和Telgarsky（2021）的原始-对偶分析方法，通过新颖的敏感度分析提供了一般凸损失的新近似等效性结果。我们的框架还恢复了二元和多元分类设置下指数尾损失的现有精确等效结果。最后，我们提供了我们技术的紧密性的证据，我们使用这些证据来演示我们的方法的有效性。

    We provide a unified framework, applicable to a general family of convex losses and across binary and multiclass settings in the overparameterized regime, to approximately characterize the implicit bias of gradient descent in closed form. Specifically, we show that the implicit bias is approximated (but not exactly equal to) the minimum-norm interpolation in high dimensions, which arises from training on the squared loss. In contrast to prior work which was tailored to exponentially-tailed losses and used the intermediate support-vector-machine formulation, our framework directly builds on the primal-dual analysis of Ji and Telgarsky (2021), allowing us to provide new approximate equivalences for general convex losses through a novel sensitivity analysis. Our framework also recovers existing exact equivalence results for exponentially-tailed losses across binary and multiclass settings. Finally, we provide evidence for the tightness of our techniques, which we use to demonstrate the ef
    
[^5]: GradSkip：具有更好计算复杂度的通信加速局部梯度方法

    GradSkip: Communication-Accelerated Local Gradient Methods with Better Computational Complexity. (arXiv:2210.16402v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.16402](http://arxiv.org/abs/2210.16402)

    本文研究了一类分布式优化算法，通过允许具有“次要”数据的客户端在本地执行较少的训练步骤来减轻高通信成本，这一方法可在强凸区域内实现可证明的通信加速。

    

    我们研究了一类分布式优化算法，旨在通过允许客户端在通信之前执行多个本地梯度类型的训练步骤来减轻高通信成本。虽然这种方法已经研究了约十年，但本地训练的加速性质在理论上还未得到完全解释。最近，Mishchenko等人(2022 International Conference on Machine Learning)取得了重大突破，证明了当本地训练得到正确执行时，会导致可证明的通信加速，在强凸区域内这一点成立，而且不依赖于任何数据相似性假设。然而，他们的方法ProxSkip要求所有客户端在每次通信轮中执行相同数量的本地训练步骤。灵感来自常识的直觉，我们通过猜测认为拥有“次要”数据的客户端应该能够用较少的本地训练步骤就能完成，而不影响整体通信

    We study a class of distributed optimization algorithms that aim to alleviate high communication costs by allowing the clients to perform multiple local gradient-type training steps prior to communication. While methods of this type have been studied for about a decade, the empirically observed acceleration properties of local training eluded all attempts at theoretical understanding. In a recent breakthrough, Mishchenko et al. (ICML 2022) proved that local training, when properly executed, leads to provable communication acceleration, and this holds in the strongly convex regime without relying on any data similarity assumptions. However, their method ProxSkip requires all clients to take the same number of local training steps in each communication round. Inspired by a common sense intuition, we start our investigation by conjecturing that clients with ``less important'' data should be able to get away with fewer local training steps without this impacting the overall communication c
    
[^6]: 量化异构转移的精确高维渐近分析

    Precise High-Dimensional Asymptotics for Quantifying Heterogeneous Transfers. (arXiv:2010.11750v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2010.11750](http://arxiv.org/abs/2010.11750)

    本文利用随机矩阵理论在线性回归设置中，对于具有两个任务的高维情况下的常用估计量的超额风险进行了精确渐近分析。

    

    最近，学习一个任务时使用来自另一个任务的样本的问题引起了广泛关注。本文提出了一个基本问题：什么时候将来自两个任务的数据合并比单独学习一个任务更好？直观上，从一个任务到另一个任务的转移效应取决于数据集的转移，如样本大小和协方差矩阵。然而，量化这种转移效应是具有挑战性的，因为我们需要比较联合学习和单任务学习之间的风险，并且一个任务是否比另一个任务具有比较优势取决于两个任务之间确切的数据集转移类型。本文利用随机矩阵理论在具有两个任务的线性回归设置中解决了这一挑战。我们给出了在高维情况下一些常用估计量的超额风险的精确渐近分析，当样本大小与特征维度成比例增加时，固定比例。精确渐近分析以样本大小的函数形式给出。

    The problem of learning one task with samples from another task has received much interest recently. In this paper, we ask a fundamental question: when is combining data from two tasks better than learning one task alone? Intuitively, the transfer effect from one task to another task depends on dataset shifts such as sample sizes and covariance matrices. However, quantifying such a transfer effect is challenging since we need to compare the risks between joint learning and single-task learning, and the comparative advantage of one over the other depends on the exact kind of dataset shift between both tasks. This paper uses random matrix theory to tackle this challenge in a linear regression setting with two tasks. We give precise asymptotics about the excess risks of some commonly used estimators in the high-dimensional regime, when the sample sizes increase proportionally with the feature dimension at fixed ratios. The precise asymptotics is provided as a function of the sample sizes 
    

