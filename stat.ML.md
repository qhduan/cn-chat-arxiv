# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Multiscale Hodge Scattering Networks for Data Analysis](https://arxiv.org/abs/2311.10270) | 提出了多尺度霍奇散射网络（MHSNs），利用多尺度基础词典和卷积结构，生成对节点排列不变的特征。 |
| [^2] | [Clustering Three-Way Data with Outliers.](http://arxiv.org/abs/2310.05288) | 这项研究提出了一种用于聚类矩阵形式数据的方法，可以处理其中的异常值。 |
| [^3] | [A Unifying Perspective on Non-Stationary Kernels for Deeper Gaussian Processes.](http://arxiv.org/abs/2309.10068) | 本论文提出了一个统一的视角来探讨非平稳核在深层高斯过程中的应用，以提高预测性能和不确定性估计。 |
| [^4] | [Quantifying degeneracy in singular models via the learning coefficient.](http://arxiv.org/abs/2308.12108) | 这项工作介绍了一种称为学习系数的量，用于精确量化深度神经网络中的退化程度，该方法能够区分不同参数区域的退化顺序，并揭示了随机优化器对临界点的归纳偏好。 |

# 详细

[^1]: 用于数据分析的多尺度霍奇散射网络

    Multiscale Hodge Scattering Networks for Data Analysis

    [https://arxiv.org/abs/2311.10270](https://arxiv.org/abs/2311.10270)

    提出了多尺度霍奇散射网络（MHSNs），利用多尺度基础词典和卷积结构，生成对节点排列不变的特征。

    

    我们提出了一种新的散射网络，用于在单纯复合仿射上测量的信号，称为\emph{多尺度霍奇散射网络}（MHSNs）。我们的构造基于单纯复合仿射上的多尺度基础词典，即$\kappa$-GHWT和$\kappa$-HGLET，我们最近为给定单纯复合仿射中的维度$\kappa \in \mathbb{N}$推广了基于节点的广义哈-沃什变换（GHWT）和分层图拉普拉斯特征变换（HGLET）。$\kappa$-GHWT和$\kappa$-HGLET都形成冗余集合（即词典）的多尺度基础向量和给定信号的相应扩展系数。我们的MHSNs使用类似于卷积神经网络（CNN）的分层结构来级联词典系数模的矩。所得特征对单纯复合仿射的重新排序不变（即节点排列的置换

    arXiv:2311.10270v2 Announce Type: replace  Abstract: We propose new scattering networks for signals measured on simplicial complexes, which we call \emph{Multiscale Hodge Scattering Networks} (MHSNs). Our construction is based on multiscale basis dictionaries on simplicial complexes, i.e., the $\kappa$-GHWT and $\kappa$-HGLET, which we recently developed for simplices of dimension $\kappa \in \mathbb{N}$ in a given simplicial complex by generalizing the node-based Generalized Haar-Walsh Transform (GHWT) and Hierarchical Graph Laplacian Eigen Transform (HGLET). The $\kappa$-GHWT and the $\kappa$-HGLET both form redundant sets (i.e., dictionaries) of multiscale basis vectors and the corresponding expansion coefficients of a given signal. Our MHSNs use a layered structure analogous to a convolutional neural network (CNN) to cascade the moments of the modulus of the dictionary coefficients. The resulting features are invariant to reordering of the simplices (i.e., node permutation of the u
    
[^2]: 带有异常值的三元数据聚类

    Clustering Three-Way Data with Outliers. (arXiv:2310.05288v1 [stat.ML])

    [http://arxiv.org/abs/2310.05288](http://arxiv.org/abs/2310.05288)

    这项研究提出了一种用于聚类矩阵形式数据的方法，可以处理其中的异常值。

    

    矩阵变量分布是模型聚类领域的最新添加，从而可以分析具有复杂结构（如图像和时间序列）的矩阵形式数据。由于其最近的出现，关于矩阵变量数据的文献有限，对于处理这些模型中的异常值的文献更少。本文讨论了一种用于聚类矩阵变量正态数据的方法。该方法使用子集对数似然的分布，将OCLUST算法扩展到矩阵变量正态数据，并使用迭代方法检测和剪裁异常值。

    Matrix-variate distributions are a recent addition to the model-based clustering field, thereby making it possible to analyze data in matrix form with complex structure such as images and time series. Due to its recent appearance, there is limited literature on matrix-variate data, with even less on dealing with outliers in these models. An approach for clustering matrix-variate normal data with outliers is discussed. The approach, which uses the distribution of subset log-likelihoods, extends the OCLUST algorithm to matrix-variate normal data and uses an iterative approach to detect and trim outliers.
    
[^3]: 非平稳核对深层高斯过程的统一视角

    A Unifying Perspective on Non-Stationary Kernels for Deeper Gaussian Processes. (arXiv:2309.10068v1 [stat.ML])

    [http://arxiv.org/abs/2309.10068](http://arxiv.org/abs/2309.10068)

    本论文提出了一个统一的视角来探讨非平稳核在深层高斯过程中的应用，以提高预测性能和不确定性估计。

    

    高斯过程（GP）是一种流行的用于数据的随机函数近似和不确定性量化的统计技术。在过去的二十年中，由于其优越的预测能力，特别是在数据稀疏情况下，以及其固有的提供强健不确定性估计的能力，GP已被广泛应用于机器学习领域。然而，它们的性能高度依赖于核心方法的复杂定制，这往往在使用标准设置和现成软件工具时使从业者不满意。可以说，GP最重要的组成部分是核函数，它扮演协方差算子的角色。Mat\'ern类的平稳核在大多数应用研究中被使用；低效的预测性能和不现实的不确定性估计往往是其结果。非平稳核表现出更好的性能，但由于其更加复杂的属性，很少被使用。

    The Gaussian process (GP) is a popular statistical technique for stochastic function approximation and uncertainty quantification from data. GPs have been adopted into the realm of machine learning in the last two decades because of their superior prediction abilities, especially in data-sparse scenarios, and their inherent ability to provide robust uncertainty estimates. Even so, their performance highly depends on intricate customizations of the core methodology, which often leads to dissatisfaction among practitioners when standard setups and off-the-shelf software tools are being deployed. Arguably the most important building block of a GP is the kernel function which assumes the role of a covariance operator. Stationary kernels of the Mat\'ern class are used in the vast majority of applied studies; poor prediction performance and unrealistic uncertainty quantification are often the consequences. Non-stationary kernels show improved performance but are rarely used due to their more
    
[^4]: 通过学习系数量化奇异模型中的退化情况

    Quantifying degeneracy in singular models via the learning coefficient. (arXiv:2308.12108v1 [stat.ML])

    [http://arxiv.org/abs/2308.12108](http://arxiv.org/abs/2308.12108)

    这项工作介绍了一种称为学习系数的量，用于精确量化深度神经网络中的退化程度，该方法能够区分不同参数区域的退化顺序，并揭示了随机优化器对临界点的归纳偏好。

    

    深度神经网络(DNN)是具有复杂退化的奇异统计模型。本文阐述了一种称为学习系数的量，它在奇异学习理论中精确地量化了深度神经网络的退化程度。重要的是，我们将证明DNN中的退化不能仅通过计算“平坦”方向的数量来解释。我们提出了一种基于随机梯度 Langevin 动力学的局部学习系数的计算可扩展近似方法。为了验证我们的方法，我们在已知理论值的低维模型上演示了其准确性。重要的是，局部学习系数能够正确恢复感兴趣参数区域之间退化的顺序。对MNIST的实验证明，局部学习系数可以揭示随机优化器对更退化或不太退化的临界点的归纳偏好。

    Deep neural networks (DNN) are singular statistical models which exhibit complex degeneracies. In this work, we illustrate how a quantity known as the \emph{learning coefficient} introduced in singular learning theory quantifies precisely the degree of degeneracy in deep neural networks. Importantly, we will demonstrate that degeneracy in DNN cannot be accounted for by simply counting the number of "flat" directions. We propose a computationally scalable approximation of a localized version of the learning coefficient using stochastic gradient Langevin dynamics. To validate our approach, we demonstrate its accuracy in low-dimensional models with known theoretical values. Importantly, the local learning coefficient can correctly recover the ordering of degeneracy between various parameter regions of interest. An experiment on MNIST shows the local learning coefficient can reveal the inductive bias of stochastic opitmizers for more or less degenerate critical points.
    

