# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Permutation invariant functions: statistical tests, dimension reduction in metric entropy and estimation](https://arxiv.org/abs/2403.01671) | 本文研究了如何在多元概率分布中测试排列不变性、估计排列不变密度以及分析排列不变函数类的度量熵，比较了它们与没有排列不变性的函数类的差异。 |
| [^2] | [Scaling Efficient LLMs](https://arxiv.org/abs/2402.14746) | 训练得到的LLM模型通常是稀疏的，为了提高效率，研究了在训练语料上达到所需准确度的参数最少的高效LLM模型，得出了参数数量与自然训练语料规模之间的关系，并指出扩展可以揭示新技能。 |
| [^3] | [Practical Kernel Tests of Conditional Independence](https://arxiv.org/abs/2402.13196) | 提出了一种数据高效、基于核的方法，用于测试条件独立性，并提出了三种偏差控制方法来纠正测试水平。 |
| [^4] | [Data-Driven Discovery of PDEs via the Adjoint Method](https://arxiv.org/abs/2401.17177) | 本文提出了一种通过伴随方法从数据中发现潜在偏微分方程的方法，展示了在给定光滑数据集的情况下，该方法可以恢复真实的PDE，但在存在噪声的情况下，准确性与PDE-FIND方法相当。 |
| [^5] | [Faster and Accurate Neural Networks with Semantic Inference.](http://arxiv.org/abs/2310.01259) | 本研究提出了一种名为语义推理（SINF）的新框架，在减少计算负载的同时，通过聚类语义相似的类来提取子图，从而减少深度神经网络的计算负担，并在性能上有限损失。 |
| [^6] | [LieDetect: Detection of representation orbits of compact Lie groups from point clouds.](http://arxiv.org/abs/2309.03086) | LieDetect是一种从紧致Lie群的有限样本轨道中估计表示的新算法。与其他技术不同，该算法可以检索精确的表示类型，并重建其轨道，有助于识别生成该作用的Lie群。该算法适用于任何紧致Lie群，并在多个领域的应用中取得了非常准确的结果。 |
| [^7] | [Martian time-series unraveled: A multi-scale nested approach with factorial variational autoencoders.](http://arxiv.org/abs/2305.16189) | 该论文提出了一种因子高斯混合变分自动编码器，用于多尺度聚类和源分离，通过利用小波散射协方差来提供随机过程的低维表示，能够区分不同的非高斯随机过程，并在MRO数据集上展现了更好的性能。 |
| [^8] | [Unsupervised Interpretable Basis Extraction for Concept-Based Visual Explanations.](http://arxiv.org/abs/2303.10523) | 本文提出了一种无监督的方法，通过对CNN进行转换，从而更好地解释中间层的表示，提取了一个可解释性欠完备基础，并证明该方法在各种网络结构和训练数据集上都很有效。 |

# 详细

[^1]: 排列不变函数：统计检验、度量熵中的降维和估计

    Permutation invariant functions: statistical tests, dimension reduction in metric entropy and estimation

    [https://arxiv.org/abs/2403.01671](https://arxiv.org/abs/2403.01671)

    本文研究了如何在多元概率分布中测试排列不变性、估计排列不变密度以及分析排列不变函数类的度量熵，比较了它们与没有排列不变性的函数类的差异。

    

    排列不变性是机器学习中可以利用来简化复杂问题的最常见的对称性之一。近年来关于构建排列不变的机器学习架构的研究活动激增。然而，在多元概率分布中的变量如何统计测试排列不变性却鲜有研究，其中样本量允许随着维数的增长。此外，在统计理论方面，关于排列不变性如何帮助估计中降维的知识甚少。本文通过研究几个基本问题，回顾并探讨这些问题：（i）测试多元分布排列不变性的假设；（ii）估计排列不变密度；（iii）分析光滑排列不变函数类的度量熵，并将其与未强加排列不变性的对应函数类进行比较。

    arXiv:2403.01671v1 Announce Type: new  Abstract: Permutation invariance is among the most common symmetry that can be exploited to simplify complex problems in machine learning (ML). There has been a tremendous surge of research activities in building permutation invariant ML architectures. However, less attention is given to how to statistically test for permutation invariance of variables in a multivariate probability distribution where the dimension is allowed to grow with the sample size. Also, in terms of a statistical theory, little is known about how permutation invariance helps with estimation in reducing dimensions. In this paper, we take a step back and examine these questions in several fundamental problems: (i) testing the assumption of permutation invariance of multivariate distributions; (ii) estimating permutation invariant densities; (iii) analyzing the metric entropy of smooth permutation invariant function classes and compare them with their counterparts without impos
    
[^2]: 扩展高效的LLM模型

    Scaling Efficient LLMs

    [https://arxiv.org/abs/2402.14746](https://arxiv.org/abs/2402.14746)

    训练得到的LLM模型通常是稀疏的，为了提高效率，研究了在训练语料上达到所需准确度的参数最少的高效LLM模型，得出了参数数量与自然训练语料规模之间的关系，并指出扩展可以揭示新技能。

    

    训练得到的LLM模型通常是稀疏的，即大部分参数为零，这引发了关于效率的问题。为此，我们研究了高效的LLM模型，即那些在训练语料上达到所需准确度的参数最少。具体地，我们比较了当前规模下训练损失的理论和实证估计，以获得自然训练语料中独特序列数量上下界的数量。我们的结果暗示：(1)要在训练语料中表示的技能数量翻倍，需要将语料规模大约扩展三到五倍，(2)对于高效的LLM模型，参数数量$N$和自然训练语料规模$D$满足$N \sim D^{0.58}$的关系，(3)如果一个LLM模型的参数数量小于训练语料中的独特序列数量，扩展可以揭示出新的技能。

    arXiv:2402.14746v1 Announce Type: new  Abstract: Trained LLMs are typically sparse in that most of the parameters are zero, raising questions on efficiency. In response, we inquire into efficient LLMs, i.e. those with the fewest parameters that achieve the desired accuracy on a training corpus. Specifically, we compare theoretical and empirical estimates for training loss at current scale to obtain upper and lower bounds on the number of unique sequences in a natural training corpus as a function of its size. Our result implies (1) to double the number of skills represented in a training corpus, the corpus must scale roughly between three and five fold (2) for efficient LLMs, the number of parameters $N$ and the size $D$ of a natural training corpus scale as $N \sim D^{0.58}$ (3) if the number of parameters of an LLM is smaller than the number of unique sequences in the training corpus, scaling up can uncover emergent skills.
    
[^3]: 实用的条件独立性核测试

    Practical Kernel Tests of Conditional Independence

    [https://arxiv.org/abs/2402.13196](https://arxiv.org/abs/2402.13196)

    提出了一种数据高效、基于核的方法，用于测试条件独立性，并提出了三种偏差控制方法来纠正测试水平。

    

    我们描述了一种数据高效、基于核的统计测试方法，用于测试条件独立性。条件独立性测试的一个主要挑战是获得正确的测试水平（指定的错误阳性率上限），同时仍具有竞争力的测试能力。过多的假阳性是由于测试统计量中的偏差引起的，该统计量是使用非参数核岭回归获得的。我们提出了三种偏差控制方法来修正测试水平，基于数据分割、辅助数据，以及（在可能的情况下）更简单的函数类。我们展示了这些组合策略在合成数据和真实世界数据中的有效性。

    arXiv:2402.13196v1 Announce Type: new  Abstract: We describe a data-efficient, kernel-based approach to statistical testing of conditional independence. A major challenge of conditional independence testing, absent in tests of unconditional independence, is to obtain the correct test level (the specified upper bound on the rate of false positives), while still attaining competitive test power. Excess false positives arise due to bias in the test statistic, which is obtained using nonparametric kernel ridge regression. We propose three methods for bias control to correct the test level, based on data splitting, auxiliary data, and (where possible) simpler function classes. We show these combined strategies are effective both for synthetic and real-world data.
    
[^4]: 数据驱动的通过伴随方法发现偏微分方程

    Data-Driven Discovery of PDEs via the Adjoint Method

    [https://arxiv.org/abs/2401.17177](https://arxiv.org/abs/2401.17177)

    本文提出了一种通过伴随方法从数据中发现潜在偏微分方程的方法，展示了在给定光滑数据集的情况下，该方法可以恢复真实的PDE，但在存在噪声的情况下，准确性与PDE-FIND方法相当。

    

    在这项工作中，我们提出了一种通过伴随方法来发现给定数据的潜在偏微分方程（PDEs）的方法。我们的思路是以一般形式考虑参数化的PDE，并制定最小化PDE解与数据误差的优化问题。利用变分计算，我们得到了拉格朗日乘子（伴随方程）的演化方程，使我们能够直接计算出与PDE参数相关的目标函数的梯度。特别是对于一族参数化和非线性PDEs，我们展示了如何推导出相应的伴随方程。我们在这里展示了，在给定光滑数据集的情况下，所提出的伴随方法可以以机器精度恢复真实的PDE。然而，在存在噪声的情况下，伴随方法的准确性与著名的PDE-FIND（Rudy et al., 2017）方法相当。

    In this work, we present an adjoint-based method for discovering the underlying governing partial differential equations (PDEs) given data. The idea is to consider a parameterized PDE in a general form, and formulate the optimization problem that minimizes the error of PDE solution from data. Using variational calculus, we obtain an evolution equation for the Lagrange multipliers (adjoint equations) allowing us to compute the gradient of the objective function with respect to the parameters of PDEs given data in a straightforward manner. In particular, for a family of parameterized and nonlinear PDEs, we show how the corresponding adjoint equations can be derived. Here, we show that given smooth data set, the proposed adjoint method can recover the true PDE up to machine accuracy. However, in the presence of noise, the accuracy of the adjoint method becomes comparable to the famous PDE Functional Identification of Nonlinear Dynamics method known as PDE-FIND (Rudy et al., 2017). Even th
    
[^5]: 使用语义推理实现更快更准确的神经网络

    Faster and Accurate Neural Networks with Semantic Inference. (arXiv:2310.01259v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2310.01259](http://arxiv.org/abs/2310.01259)

    本研究提出了一种名为语义推理（SINF）的新框架，在减少计算负载的同时，通过聚类语义相似的类来提取子图，从而减少深度神经网络的计算负担，并在性能上有限损失。

    

    深度神经网络通常具有显著的计算负担。虽然提出了结构化剪枝和专门用于移动设备的神经网络的方法，但它们会导致明显的准确率损失。在本文中，我们利用潜在表示中的内在冗余来减少计算负载，并在性能上有限损失。我们证明，语义上相似的输入共享许多滤波器，尤其是在较早的层次上。因此，可以对语义上相似的类进行聚类，以创建特定于聚类的子图。为此，我们提出了一个名为语义推理（SINF）的新框架。简而言之，SINF（i）使用一个小的附加分类器来识别对象属于的语义聚类，并（ii）执行与该语义聚类相关的基本DNN提取的子图进行推理。为了提取每个特定于聚类的子图，我们提出了一个名为区分能力得分（DCS）的新方法，用于找到具有区分能力的子图。

    Deep neural networks (DNN) usually come with a significant computational burden. While approaches such as structured pruning and mobile-specific DNNs have been proposed, they incur drastic accuracy loss. In this paper we leverage the intrinsic redundancy in latent representations to reduce the computational load with limited loss in performance. We show that semantically similar inputs share many filters, especially in the earlier layers. Thus, semantically similar classes can be clustered to create cluster-specific subgraphs. To this end, we propose a new framework called Semantic Inference (SINF). In short, SINF (i) identifies the semantic cluster the object belongs to using a small additional classifier and (ii) executes the subgraph extracted from the base DNN related to that semantic cluster for inference. To extract each cluster-specific subgraph, we propose a new approach named Discriminative Capability Score (DCS) that finds the subgraph with the capability to discriminate amon
    
[^6]: LieDetect: 从点云中检测紧致Lie群的表示轨道

    LieDetect: Detection of representation orbits of compact Lie groups from point clouds. (arXiv:2309.03086v1 [math.OC])

    [http://arxiv.org/abs/2309.03086](http://arxiv.org/abs/2309.03086)

    LieDetect是一种从紧致Lie群的有限样本轨道中估计表示的新算法。与其他技术不同，该算法可以检索精确的表示类型，并重建其轨道，有助于识别生成该作用的Lie群。该算法适用于任何紧致Lie群，并在多个领域的应用中取得了非常准确的结果。

    

    我们提出了一种新的算法，用于从紧致Lie群的有限样本轨道中估计表示。与其他报道的技术不同，我们的方法允许检索精确的表示类型，作为不可约表示的直和。而且，对表示类型的了解可以重建其轨道，有助于识别生成该作用的Lie群。我们的算法适用于任何紧致Lie群，但只考虑了SO(2), T^d, SU(2)和SO(3)的实例化。我们推导了在Hausdorff和Wasserstein距离方面的鲁棒性的理论保证。我们的工具来自于几何测度理论，计算几何和矩阵流形上的优化。算法在高达16维的合成数据以及图像分析，谐波分析和经典力学系统的实际应用中进行了测试，取得了非常准确的结果。

    We suggest a new algorithm to estimate representations of compact Lie groups from finite samples of their orbits. Different from other reported techniques, our method allows the retrieval of the precise representation type as a direct sum of irreducible representations. Moreover, the knowledge of the representation type permits the reconstruction of its orbit, which is useful to identify the Lie group that generates the action. Our algorithm is general for any compact Lie group, but only instantiations for SO(2), T^d, SU(2) and SO(3) are considered. Theoretical guarantees of robustness in terms of Hausdorff and Wasserstein distances are derived. Our tools are drawn from geometric measure theory, computational geometry, and optimization on matrix manifolds. The algorithm is tested for synthetic data up to dimension 16, as well as real-life applications in image analysis, harmonic analysis, and classical mechanics systems, achieving very accurate results.
    
[^7]: 火星时间序列分解：一种多尺度嵌套方法中的因子变分自编码器

    Martian time-series unraveled: A multi-scale nested approach with factorial variational autoencoders. (arXiv:2305.16189v1 [cs.LG])

    [http://arxiv.org/abs/2305.16189](http://arxiv.org/abs/2305.16189)

    该论文提出了一种因子高斯混合变分自动编码器，用于多尺度聚类和源分离，通过利用小波散射协方差来提供随机过程的低维表示，能够区分不同的非高斯随机过程，并在MRO数据集上展现了更好的性能。

    

    无监督的源分离涉及通过混合操作记录的未知源信号的分解，其中对源的先验知识有限，仅可以访问信号混合数据集。这个问题本质上是不适用的，并且进一步受到时间序列数据中源展现出的多种时间尺度的挑战。为了解决这个问题，我们提出了一种无监督的多尺度聚类和源分离框架，通过利用小波散射协方差来提供随机过程的低维表示，能够区分不同的非高斯随机过程。在这个表示空间中，我们开发了一个因子高斯混合变分自动编码器，它被训练用于(1)概率地对不同时间尺度上的源进行聚类和逐层非监督源分离，(2)在每个时间尺度上提取低维表示，(3)学习源信号的因子表示，(4)在表示空间中进行采样，以生成未知源信号。我们在MRO上的三个频道的可见数据集上进行了评估，结果表明所提出的方法比目前最先进的技术具有更好的性能。

    Unsupervised source separation involves unraveling an unknown set of source signals recorded through a mixing operator, with limited prior knowledge about the sources, and only access to a dataset of signal mixtures. This problem is inherently ill-posed and is further challenged by the variety of time-scales exhibited by sources in time series data. Existing methods typically rely on a preselected window size that limits their capacity to handle multi-scale sources. To address this issue, instead of operating in the time domain, we propose an unsupervised multi-scale clustering and source separation framework by leveraging wavelet scattering covariances that provide a low-dimensional representation of stochastic processes, capable of distinguishing between different non-Gaussian stochastic processes. Nested within this representation space, we develop a factorial Gaussian-mixture variational autoencoder that is trained to (1) probabilistically cluster sources at different time-scales a
    
[^8]: 无监督解释性基础抽取用于基于概念的视觉解释

    Unsupervised Interpretable Basis Extraction for Concept-Based Visual Explanations. (arXiv:2303.10523v1 [cs.CV])

    [http://arxiv.org/abs/2303.10523](http://arxiv.org/abs/2303.10523)

    本文提出了一种无监督的方法，通过对CNN进行转换，从而更好地解释中间层的表示，提取了一个可解释性欠完备基础，并证明该方法在各种网络结构和训练数据集上都很有效。

    

    研究人员尝试用人类可以理解的概念来解释CNN图像分类器预测和中间层表示。本文提出了一种无监督后处理方法，通过查找解释像素激活的稀疏二值化转换表示的特征空间旋转来提取解释性欠完备基础。我们对现有的流行CNN进行了实验，并证明了我们方法在网络架构和训练数据集上提取解释性基础的有效性。最后，我们扩展了文献中的基础可解释性度量，并表明，当中间层表示被转换为我们方法提取的基础时，它们变得更易解释。

    An important line of research attempts to explain CNN image classifier predictions and intermediate layer representations in terms of human understandable concepts. In this work, we expand on previous works in the literature that use annotated concept datasets to extract interpretable feature space directions and propose an unsupervised post-hoc method to extract a disentangling interpretable basis by looking for the rotation of the feature space that explains sparse one-hot thresholded transformed representations of pixel activations. We do experimentation with existing popular CNNs and demonstrate the effectiveness of our method in extracting an interpretable basis across network architectures and training datasets. We make extensions to the existing basis interpretability metrics found in the literature and show that, intermediate layer representations become more interpretable when transformed to the bases extracted with our method. Finally, using the basis interpretability metrics
    

