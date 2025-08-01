# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimal and Near-Optimal Adaptive Vector Quantization](https://arxiv.org/abs/2402.03158) | 该论文提出了最优和近似最优的自适应向量量化算法，能够优化量化过程，在时间和空间复杂度上具有改进，可扩展应用于各种机器学习任务。 |
| [^2] | [Weighted least-squares approximation with determinantal point processes and generalized volume sampling.](http://arxiv.org/abs/2312.14057) | 该论文研究了使用行列式点过程和广义体积取样进行加权最小二乘逼近的问题，提出了广义版本的体积标准化取样算法，并证明了该算法在期望上的准最优性以及在某些规范向量空间中的逼近结果。 |
| [^3] | [GANs Settle Scores!.](http://arxiv.org/abs/2306.01654) | 这篇论文提出了一种新的方法，通过变分方法来统一分析生成器的优化，并展示了在f-散度最小化和IPM GAN中生成器的最优解决方案。这种方法能够平滑分数匹配。 |
| [^4] | [Divided Attention: Unsupervised Multi-Object Discovery with Contextually Separated Slots.](http://arxiv.org/abs/2304.01430) | 该论文提出了一种新的无监督多对象发现方法，通过一种上下文分隔的槽结构来将视觉场分割为独立运动区域，并用对抗性标准来保证解码器无法重构整个光流。 |

# 详细

[^1]: 最优和近似最优的自适应向量量化

    Optimal and Near-Optimal Adaptive Vector Quantization

    [https://arxiv.org/abs/2402.03158](https://arxiv.org/abs/2402.03158)

    该论文提出了最优和近似最优的自适应向量量化算法，能够优化量化过程，在时间和空间复杂度上具有改进，可扩展应用于各种机器学习任务。

    

    量化是许多机器学习用例中的基本优化，包括压缩梯度、模型权重和激活以及数据集。最准确的量化形式是“自适应”，其中通过最小化相对于给定输入的误差来优化，而不是针对最坏情况进行优化。然而，最优的自适应量化方法在运行时间和内存需求方面被认为是不可行的。我们重新审视了自适应向量量化（AVQ）问题，并提出了一种能够在渐近改进的时间和空间复杂度下找到最优解的算法。我们还提出了一种更快速的近似最优算法，以处理大输入。我们的实验表明，我们的算法可能会在各种机器学习应用中更广泛地使用AVQ。

    Quantization is a fundamental optimization for many machine-learning use cases, including compressing gradients, model weights and activations, and datasets. The most accurate form of quantization is \emph{adaptive}, where the error is minimized with respect to a given input, rather than optimizing for the worst case. However, optimal adaptive quantization methods are considered infeasible in terms of both their runtime and memory requirements.   We revisit the Adaptive Vector Quantization (AVQ) problem and present algorithms that find optimal solutions with asymptotically improved time and space complexity. We also present an even faster near-optimal algorithm for large inputs. Our experiments show our algorithms may open the door to using AVQ more extensively in a variety of machine learning applications.
    
[^2]: 基于行列式点过程和广义体积取样的加权最小二乘逼近

    Weighted least-squares approximation with determinantal point processes and generalized volume sampling. (arXiv:2312.14057v2 [math.NA] UPDATED)

    [http://arxiv.org/abs/2312.14057](http://arxiv.org/abs/2312.14057)

    该论文研究了使用行列式点过程和广义体积取样进行加权最小二乘逼近的问题，提出了广义版本的体积标准化取样算法，并证明了该算法在期望上的准最优性以及在某些规范向量空间中的逼近结果。

    

    我们考虑使用给定的m维空间V_m中的元素，借助于一些特征映射φ，通过对随机点x_1，...，x_n处的函数进行评估，来逼近函数从L^2到函数。在回顾一些关于使用独立同分布点的最优加权最小二乘的结果之后，我们考虑使用投影行列式点过程（DPP）或体积取样的加权最小二乘。这些分布在选定的特征φ(x_i)中引入了点之间的依赖性，以促进多样性。我们首先提供了广义版本的体积标准化取样，使用样本数n = O(mlog(m))得到了期望上的准最优结果，这意味着期望的L^2误差受到一个常数乘以在L^2中的最佳逼近误差的限制。此外，进一步假设函数在某个嵌入在L^2中的规范向量空间H中，我们进一步证明了逼近的结果。

    We consider the problem of approximating a function from $L^2$ by an element of a given $m$-dimensional space $V_m$, associated with some feature map $\varphi$, using evaluations of the function at random points $x_1,\dots,x_n$. After recalling some results on optimal weighted least-squares using independent and identically distributed points, we consider weighted least-squares using projection determinantal point processes (DPP) or volume sampling. These distributions introduce dependence between the points that promotes diversity in the selected features $\varphi(x_i)$. We first provide a generalized version of volume-rescaled sampling yielding quasi-optimality results in expectation with a number of samples $n = O(m\log(m))$, that means that the expected $L^2$ error is bounded by a constant times the best approximation error in $L^2$. Also, further assuming that the function is in some normed vector space $H$ continuously embedded in $L^2$, we further prove that the approximation is
    
[^3]: GANs解决分数争议问题！

    GANs Settle Scores!. (arXiv:2306.01654v1 [cs.LG])

    [http://arxiv.org/abs/2306.01654](http://arxiv.org/abs/2306.01654)

    这篇论文提出了一种新的方法，通过变分方法来统一分析生成器的优化，并展示了在f-散度最小化和IPM GAN中生成器的最优解决方案。这种方法能够平滑分数匹配。

    

    生成对抗网络（GAN）由一个生成器和一个判别器组成，生成器被训练以学习期望数据的基础分布，而判别器则被训练以区分真实样本和生成器输出的样本。本文提出了一种统一的方法，通过变分方法来分析生成器优化。在f-散度最小化 GAN 中，我们表明最优生成器是通过将其输出分布的得分与数据分布的得分进行匹配得到的。在IPM GAN中，我们表明这个最优生成器匹配得分型函数，包括与所选IPM约束空间相关的核流场。此外，IPM-GAN优化可以看作是平滑分数匹配中的一种，其中数据和生成器分布的得分与在核函数上进行卷积处理。

    Generative adversarial networks (GANs) comprise a generator, trained to learn the underlying distribution of the desired data, and a discriminator, trained to distinguish real samples from those output by the generator. A majority of GAN literature focuses on understanding the optimality of the discriminator through integral probability metric (IPM) or divergence based analysis. In this paper, we propose a unified approach to analyzing the generator optimization through variational approach. In $f$-divergence-minimizing GANs, we show that the optimal generator is the one that matches the score of its output distribution with that of the data distribution, while in IPM GANs, we show that this optimal generator matches score-like functions, involving the flow-field of the kernel associated with a chosen IPM constraint space. Further, the IPM-GAN optimization can be seen as one of smoothed score-matching, where the scores of the data and the generator distributions are convolved with the 
    
[^4]: 分离的关注力：基于上下文分离槽的无监督多对象发现

    Divided Attention: Unsupervised Multi-Object Discovery with Contextually Separated Slots. (arXiv:2304.01430v1 [cs.CV])

    [http://arxiv.org/abs/2304.01430](http://arxiv.org/abs/2304.01430)

    该论文提出了一种新的无监督多对象发现方法，通过一种上下文分隔的槽结构来将视觉场分割为独立运动区域，并用对抗性标准来保证解码器无法重构整个光流。

    

    我们提出了一种将视觉场分割为独立运动区域的方法，不需要任何基础真值或监督。它由基于槽关注的对抗条件编码器-解码器架构组成，修改为使用图像作为上下文来解码光流，而不是尝试重构图像本身。在结果的多模式表示中，一种模式（流）将馈送给编码器以产生单独的潜在代码（槽），而另一种模式（图像）将决定解码器从槽生成第一个模式（流）。由于惯常的自编码基于最小化重构误差，并不能防止整个流被编码到一个槽中，因此我们将损失修改为基于上下文信息分离的对抗性标准。

    We introduce a method to segment the visual field into independently moving regions, trained with no ground truth or supervision. It consists of an adversarial conditional encoder-decoder architecture based on Slot Attention, modified to use the image as context to decode optical flow without attempting to reconstruct the image itself. In the resulting multi-modal representation, one modality (flow) feeds the encoder to produce separate latent codes (slots), whereas the other modality (image) conditions the decoder to generate the first (flow) from the slots. This design frees the representation from having to encode complex nuisance variability in the image due to, for instance, illumination and reflectance properties of the scene. Since customary autoencoding based on minimizing the reconstruction error does not preclude the entire flow from being encoded into a single slot, we modify the loss to an adversarial criterion based on Contextual Information Separation. The resulting min-m
    

