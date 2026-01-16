# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Arbitrary Polynomial Separations in Trainable Quantum Machine Learning](https://arxiv.org/abs/2402.08606) | 该论文通过构建一种层次结构的可高效训练的量子神经网络，实现了在经典序列建模任务中具有任意常数次数的多项式内存分离。每个量子神经网络单元都可以在常数时间内在量子设备上进行计算。 |
| [^2] | [The cell signaling structure function.](http://arxiv.org/abs/2401.02501) | 该论文提出了一个新的方法，在活体细胞显微镜捕捉到的五维视频中寻找细胞信号动力学时空模式，并且不需要任何先验的预期模式动力学和训练数据。该方法基于细胞信号结构函数（SSF），通过测量细胞信号状态和周围细胞质之间的核糖体强度，与当前最先进的核糖体与细胞核比值相比有了显著改进。通过归一化压缩距离（NCD）来识别相似的模式。该方法能够将输入的SSF构图表示为低维嵌入中的点，最优地捕捉模式。 |
| [^3] | [Physics-guided Noise Neural Proxy for Low-light Raw Image Denoising.](http://arxiv.org/abs/2310.09126) | 本文提出了一种新的物理引导噪声神经代理（PNNP）用于准确噪声建模和低光原始图像去噪，集成了物理引导噪声解耦、物理引导代理模型和可微分分布导向损失等高效技术。 |

# 详细

[^1]: 可训练量子机器学习中任意多项式分离

    Arbitrary Polynomial Separations in Trainable Quantum Machine Learning

    [https://arxiv.org/abs/2402.08606](https://arxiv.org/abs/2402.08606)

    该论文通过构建一种层次结构的可高效训练的量子神经网络，实现了在经典序列建模任务中具有任意常数次数的多项式内存分离。每个量子神经网络单元都可以在常数时间内在量子设备上进行计算。

    

    最近在量子机器学习领域的理论研究表明，量子神经网络（QNNs）的表达能力和可训练性之间存在一种普遍的权衡；作为这些结果的推论，实际上在表达能力上实现指数级的超越经典机器学习模型的分离是不可行的，因为这样的QNN训练时间在模型规模上是指数级的。我们通过构建一种层次结构的可高效训练的QNNs来绕开这些负面结果，在执行经典序列建模任务时，这些QNNs可以展示出任意常数次数的多项式内存分离，且在量子设备上每个单元格都可以在常数时间内进行计算。我们证明了这种分离适用于包括循环神经网络和Transformer在内的众所周知的经典网络。我们还展示了量子上下文相关性的重要性。

    Recent theoretical results in quantum machine learning have demonstrated a general trade-off between the expressive power of quantum neural networks (QNNs) and their trainability; as a corollary of these results, practical exponential separations in expressive power over classical machine learning models are believed to be infeasible as such QNNs take a time to train that is exponential in the model size. We here circumvent these negative results by constructing a hierarchy of efficiently trainable QNNs that exhibit unconditionally provable, polynomial memory separations of arbitrary constant degree over classical neural networks in performing a classical sequence modeling task. Furthermore, each unit cell of the introduced class of QNNs is computationally efficient, implementable in constant time on a quantum device. The classical networks we prove a separation over include well-known examples such as recurrent neural networks and Transformers. We show that quantum contextuality is th
    
[^2]: 细胞信号传导结构和功能

    The cell signaling structure function. (arXiv:2401.02501v1 [cs.CV])

    [http://arxiv.org/abs/2401.02501](http://arxiv.org/abs/2401.02501)

    该论文提出了一个新的方法，在活体细胞显微镜捕捉到的五维视频中寻找细胞信号动力学时空模式，并且不需要任何先验的预期模式动力学和训练数据。该方法基于细胞信号结构函数（SSF），通过测量细胞信号状态和周围细胞质之间的核糖体强度，与当前最先进的核糖体与细胞核比值相比有了显著改进。通过归一化压缩距离（NCD）来识别相似的模式。该方法能够将输入的SSF构图表示为低维嵌入中的点，最优地捕捉模式。

    

    活体细胞显微镜捕捉到的五维$(x,y,z,channel,time)$视频显示了细胞运动和信号动力学的模式。我们在这里提出一种在五维活体细胞显微镜视频中寻找细胞信号动力学时空模式的方法，该方法独特之处在于不需要预先了解预期的模式动力学以及没有训练数据。所提出的细胞信号结构函数（SSF）是一种Kolmogorov结构函数，可以通过核心区域相对于周围细胞质的核糖体强度来最优地测量细胞信号状态，相比当前最先进的核糖体与细胞核比值有了显著的改进。通过度量归一化压缩距离（NCD）来识别相似的模式。NCD是一个用于表示输入的SSF构图在低维嵌入中作为点的Hilbert空间的再生核，可以最优地捕捉模式。

    Live cell microscopy captures 5-D $(x,y,z,channel,time)$ movies that display patterns of cellular motion and signaling dynamics. We present here an approach to finding spatiotemporal patterns of cell signaling dynamics in 5-D live cell microscopy movies unique in requiring no \emph{a priori} knowledge of expected pattern dynamics, and no training data. The proposed cell signaling structure function (SSF) is a Kolmogorov structure function that optimally measures cell signaling state as nuclear intensity w.r.t. surrounding cytoplasm, a significant improvement compared to the current state-of-the-art cytonuclear ratio. SSF kymographs store at each spatiotemporal cell centroid the SSF value, or a functional output such as velocity. Patterns of similarity are identified via the metric normalized compression distance (NCD). The NCD is a reproducing kernel for a Hilbert space that represents the input SSF kymographs as points in a low dimensional embedding that optimally captures the pattern
    
[^3]: 物理引导的噪声神经代理用于低光原始图像去噪

    Physics-guided Noise Neural Proxy for Low-light Raw Image Denoising. (arXiv:2310.09126v1 [eess.IV])

    [http://arxiv.org/abs/2310.09126](http://arxiv.org/abs/2310.09126)

    本文提出了一种新的物理引导噪声神经代理（PNNP）用于准确噪声建模和低光原始图像去噪，集成了物理引导噪声解耦、物理引导代理模型和可微分分布导向损失等高效技术。

    

    低光原始图像去噪在移动摄影中起着至关重要的作用，学习方法已成为主流方法。使用合成数据训练学习方法成为替代对应真实数据的高效实用方法。然而，合成数据的质量受噪声模型精度的限制，降低了低光原始图像去噪的性能。本文提出了一个新颖的准确噪声建模框架，学习一个从暗场中获得的物理引导噪声神经代理（PNNP）。PNNP集成了三种高效技术：物理引导噪声解耦（PND），物理引导代理模型（PPM）和可微分分布导向损失（DDL）。PND将暗场解耦为不同的组分，并以灵活的方式处理不同水平的噪声，降低了噪声神经代理的复杂度。PPM通过引入物理先验有效地约束生成的噪声。

    Low-light raw image denoising plays a crucial role in mobile photography, and learning-based methods have become the mainstream approach. Training the learning-based methods with synthetic data emerges as an efficient and practical alternative to paired real data. However, the quality of synthetic data is inherently limited by the low accuracy of the noise model, which decreases the performance of low-light raw image denoising. In this paper, we develop a novel framework for accurate noise modeling that learns a physics-guided noise neural proxy (PNNP) from dark frames. PNNP integrates three efficient techniques: physics-guided noise decoupling (PND), physics-guided proxy model (PPM), and differentiable distribution-oriented loss (DDL). The PND decouples the dark frame into different components and handles different levels of noise in a flexible manner, which reduces the complexity of the noise neural proxy. The PPM incorporates physical priors to effectively constrain the generated no
    

