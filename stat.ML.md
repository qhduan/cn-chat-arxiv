# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data](https://arxiv.org/abs/2404.01413) | 本文通过比较数据取代和数据积累两种情况，发现累积数据可以防止模型崩溃。 |
| [^2] | [Estimating Causal Effects with Double Machine Learning -- A Method Evaluation](https://arxiv.org/abs/2403.14385) | 双重/无偏机器学习（DML）方法改进了因果效应估计中对非线性混淆关系的调整，摆脱传统函数形式假设，但仍然依赖于标准因果假设。 |
| [^3] | [An extended asymmetric sigmoid with Perceptron (SIGTRON) for imbalanced linear classification](https://arxiv.org/abs/2312.16043) | 本文提出了一个新的多项式参数化sigmoid函数(SIGTRON)，并且介绍了其伴随的SIC模型。相比传统的成本敏感学习模型，在给定的训练数据集接近良好平衡的条件下，所提出的SIC模型对于数据集的变化更加适应，并通过创建倾斜的超平面方程来实现。 |
| [^4] | [Heteroskedastic conformal regression.](http://arxiv.org/abs/2309.08313) | 本文研究了使用标准化和Mondrian符合规范的方法如何构建自适应的预测区间，以解决回归问题中的异方差噪声。 |
| [^5] | [Causal Inference with Differentially Private (Clustered) Outcomes.](http://arxiv.org/abs/2308.00957) | 本文提出了一种新的差分隐私机制"Cluster-DP"，它在保证隐私的同时利用数据的聚类结构，从而实现了更强的隐私保证和较低的方差，可以用于进行因果分析。 |
| [^6] | [Any-dimensional equivariant neural networks.](http://arxiv.org/abs/2306.06327) | 该论文提出了一个新的方法，利用代数拓扑中的表示稳定性，可以定义出一个可以以任意维度为输入的等变神经网络。这种方法使用方便，只需指定网络架构和等变性的组，且在任何训练过程中都可以使用。 |
| [^7] | [Quantum Kernel Mixtures for Probabilistic Deep Learning.](http://arxiv.org/abs/2305.18204) | 本文提出了一种量子核混合方法，可以用于表示连续和离散随机变量的联合概率分布。该框架允许构建可微分的模型，适用于密度估计、推理和采样，以及各种机器学习任务，包括生成建模和判别学习。 |
| [^8] | [Orthonormal Expansions for Translation-Invariant Kernels.](http://arxiv.org/abs/2206.08648) | 该论文提出了一种傅里叶分析技术，用于从$\mathscr{L}_2(\mathbb{R})$的正交基中构建平移不变核函数的正交基展开，实现了马特尔核函数、柯西核函数和高斯核函数的明确展开表达式。 |

# 详细

[^1]: 模型崩溃是否不可避免？通过累积真实和合成数据打破递归的诅咒

    Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data

    [https://arxiv.org/abs/2404.01413](https://arxiv.org/abs/2404.01413)

    本文通过比较数据取代和数据积累两种情况，发现累积数据可以防止模型崩溃。

    

    随着生成模型的激增，以及在网络规模数据上的预训练，一个及时的问题浮出水面：当这些模型被训练在它们自己生成的输出上时会发生什么？最近对模型数据反馈循环的研究发现，这样的循环可能导致模型崩溃，即性能随着每次模型拟合迭代逐渐下降，直到最新的模型变得无用。然而，最近几篇研究模型崩溃的论文都假设随着时间推移，新数据会取代旧数据，而不是假设数据会随时间累积。在本文中，我们比较了这两种情况，并表明积累数据可以防止模型崩溃。我们首先研究了一个解析可处理的设置，其中一系列线性模型拟合到先前模型的预测。先前的工作表明，如果数据被替换，测试误差会随着模型拟合迭代次数线性增加；我们扩展了这个研究探讨了数据逐渐累积的情况下会发生什么。

    arXiv:2404.01413v1 Announce Type: cross  Abstract: The proliferation of generative models, combined with pretraining on web-scale data, raises a timely question: what happens when these models are trained on their own generated outputs? Recent investigations into model-data feedback loops discovered that such loops can lead to model collapse, a phenomenon where performance progressively degrades with each model-fitting iteration until the latest model becomes useless. However, several recent papers studying model collapse assumed that new data replace old data over time rather than assuming data accumulate over time. In this paper, we compare these two settings and show that accumulating data prevents model collapse. We begin by studying an analytically tractable setup in which a sequence of linear models are fit to the previous models' predictions. Previous work showed if data are replaced, the test error increases linearly with the number of model-fitting iterations; we extend this r
    
[^2]: 用双机器学习估计因果效应--一种方法评估

    Estimating Causal Effects with Double Machine Learning -- A Method Evaluation

    [https://arxiv.org/abs/2403.14385](https://arxiv.org/abs/2403.14385)

    双重/无偏机器学习（DML）方法改进了因果效应估计中对非线性混淆关系的调整，摆脱传统函数形式假设，但仍然依赖于标准因果假设。

    

    使用观测数据估计因果效应仍然是一个非常活跃的研究领域。近年来，研究人员开发了利用机器学习放宽传统假设以估计因果效应的新框架。在本文中，我们回顾了其中一个最重要的方法-"双/无偏机器学习"（DML），并通过比较它在模拟数据上相对于更传统的统计方法的表现，然后将其应用于真实世界数据进行了实证评估。我们的研究发现表明，在DML中应用一个适当灵活的机器学习算法可以改进对各种非线性混淆关系的调整。这种优势使得可以摆脱通常在因果效应估计中必需的传统函数形式假设。然而，我们表明该方法在关于因果关系的标准假设方面仍然至关重要。

    arXiv:2403.14385v1 Announce Type: cross  Abstract: The estimation of causal effects with observational data continues to be a very active research area. In recent years, researchers have developed new frameworks which use machine learning to relax classical assumptions necessary for the estimation of causal effects. In this paper, we review one of the most prominent methods - "double/debiased machine learning" (DML) - and empirically evaluate it by comparing its performance on simulated data relative to more traditional statistical methods, before applying it to real-world data. Our findings indicate that the application of a suitably flexible machine learning algorithm within DML improves the adjustment for various nonlinear confounding relationships. This advantage enables a departure from traditional functional form assumptions typically necessary in causal effect estimation. However, we demonstrate that the method continues to critically depend on standard assumptions about causal 
    
[^3]: 一种针对不平衡线性分类的扩展非对称sigmoid和感知机(SIGTRON)

    An extended asymmetric sigmoid with Perceptron (SIGTRON) for imbalanced linear classification

    [https://arxiv.org/abs/2312.16043](https://arxiv.org/abs/2312.16043)

    本文提出了一个新的多项式参数化sigmoid函数(SIGTRON)，并且介绍了其伴随的SIC模型。相比传统的成本敏感学习模型，在给定的训练数据集接近良好平衡的条件下，所提出的SIC模型对于数据集的变化更加适应，并通过创建倾斜的超平面方程来实现。

    

    本文提出了一种新的多项式参数化sigmoid函数，称为SIGTRON，它是一种扩展的非对称sigmoid函数和感知机的结合，以及它的伴随凸模型SIGTRON-不平衡分类(SIC)模型，该模型使用了虚拟SIGTRON产生的凸损失函数。与传统的$\pi$-加权成本敏感学习模型相比，SIC模型在损失函数上没有外部的$\pi$-权重，而是在虚拟的SIGTRON产生的损失函数中有内部参数。因此，当给定的训练数据集接近良好平衡的条件时，我们展示了所提出的SIC模型对数据集的变化更加适应，比如训练集和测试集之间比例不平衡的不一致性。这种适应是通过创建一个倾斜的超平面方程来实现的。另外，我们提出了一个基于拟牛顿优化(L-BFGS)框架的虚拟凸损失，通过开发一个基于区间的二分线性搜索算法来实现。

    This article presents a new polynomial parameterized sigmoid called SIGTRON, which is an extended asymmetric sigmoid with Perceptron, and its companion convex model called SIGTRON-imbalanced classification (SIC) model that employs a virtual SIGTRON-induced convex loss function. In contrast to the conventional $\pi$-weighted cost-sensitive learning model, the SIC model does not have an external $\pi$-weight on the loss function but has internal parameters in the virtual SIGTRON-induced loss function. As a consequence, when the given training dataset is close to the well-balanced condition, we show that the proposed SIC model is more adaptive to variations of the dataset, such as the inconsistency of the scale-class-imbalance ratio between the training and test datasets. This adaptation is achieved by creating a skewed hyperplane equation. Additionally, we present a quasi-Newton optimization(L-BFGS) framework for the virtual convex loss by developing an interval-based bisection line sear
    
[^4]: 异方差拟合置信回归

    Heteroskedastic conformal regression. (arXiv:2309.08313v1 [stat.ML])

    [http://arxiv.org/abs/2309.08313](http://arxiv.org/abs/2309.08313)

    本文研究了使用标准化和Mondrian符合规范的方法如何构建自适应的预测区间，以解决回归问题中的异方差噪声。

    

    符合规范的预测以及特定的拆分符合规范的预测提供了一种无分布的方法来估计具有统计保证的预测区间。最近的研究表明，当专注于边际覆盖时，即在校准数据集上，该方法产生的预测区间平均包含预定义覆盖水平的真实值，拆分符合规范的预测可以产生最先进的预测区间。然而，这样的区间通常不是自适应的，这对于具有异方差噪声的回归问题可能是有问题的。本文试图阐明如何使用标准化和Mondrian符合规范的方法来构建自适应的预测区间。我们以系统的方式提出理论和实验结果来研究这些方法。

    Conformal prediction, and split conformal prediction as a specific implementation, offer a distribution-free approach to estimating prediction intervals with statistical guarantees. Recent work has shown that split conformal prediction can produce state-of-the-art prediction intervals when focusing on marginal coverage, i.e., on a calibration dataset the method produces on average prediction intervals that contain the ground truth with a predefined coverage level. However, such intervals are often not adaptive, which can be problematic for regression problems with heteroskedastic noise. This paper tries to shed new light on how adaptive prediction intervals can be constructed using methods such as normalized and Mondrian conformal prediction. We present theoretical and experimental results in which these methods are investigated in a systematic way.
    
[^5]: 具有差分隐私(分组)结果的因果推断

    Causal Inference with Differentially Private (Clustered) Outcomes. (arXiv:2308.00957v1 [stat.ML])

    [http://arxiv.org/abs/2308.00957](http://arxiv.org/abs/2308.00957)

    本文提出了一种新的差分隐私机制"Cluster-DP"，它在保证隐私的同时利用数据的聚类结构，从而实现了更强的隐私保证和较低的方差，可以用于进行因果分析。

    

    从随机实验中估计因果效应只有在参与者同意透露他们可能敏感的响应时才可行。在确保隐私的许多方法中，标签差分隐私是一种广泛使用的算法隐私保证度量，可以鼓励参与者分享响应而不会面临去匿名化的风险。许多差分隐私机制会向原始数据集中注入噪音来实现这种隐私保证，这会增加大多数统计估计量的方差，使得精确测量因果效应变得困难：从差分隐私数据进行因果分析存在着固有的隐私-方差权衡。为了实现更强隐私保证的较低方差，我们提出了一种新的差分隐私机制"Cluster-DP"，它利用数据的任何给定的聚类结构，同时仍然允许对因果效应进行估计。

    Estimating causal effects from randomized experiments is only feasible if participants agree to reveal their potentially sensitive responses. Of the many ways of ensuring privacy, label differential privacy is a widely used measure of an algorithm's privacy guarantee, which might encourage participants to share responses without running the risk of de-anonymization. Many differentially private mechanisms inject noise into the original data-set to achieve this privacy guarantee, which increases the variance of most statistical estimators and makes the precise measurement of causal effects difficult: there exists a fundamental privacy-variance trade-off to performing causal analyses from differentially private data. With the aim of achieving lower variance for stronger privacy guarantees, we suggest a new differential privacy mechanism, "Cluster-DP", which leverages any given cluster structure of the data while still allowing for the estimation of causal effects. We show that, depending 
    
[^6]: 任意维度等变神经网络

    Any-dimensional equivariant neural networks. (arXiv:2306.06327v1 [cs.LG])

    [http://arxiv.org/abs/2306.06327](http://arxiv.org/abs/2306.06327)

    该论文提出了一个新的方法，利用代数拓扑中的表示稳定性，可以定义出一个可以以任意维度为输入的等变神经网络。这种方法使用方便，只需指定网络架构和等变性的组，且在任何训练过程中都可以使用。

    

    传统的监督学习旨在通过将函数拟合到一组具有固定维度的输入/输出对来学习未知映射。然后，在相同维度的输入上定义拟合函数。然而，在许多情况下，未知映射以任意维度的输入作为输入；例如，定义在任意大小的图形上的图形参数和定义在任意数量粒子上的物理量。我们利用代数拓扑中的新现象——表示稳定性，来定义等变神经网络，可以使用固定维度的数据进行训练，然后在任意维度上扩展接受输入。我们的方法易于使用，只需要网络架构和等变性的组，并且可以与任何训练过程结合使用。我们提供了我们方法的简单开源实现，并提供了初步的数值实验。

    Traditional supervised learning aims to learn an unknown mapping by fitting a function to a set of input-output pairs with a fixed dimension. The fitted function is then defined on inputs of the same dimension. However, in many settings, the unknown mapping takes inputs in any dimension; examples include graph parameters defined on graphs of any size and physics quantities defined on an arbitrary number of particles. We leverage a newly-discovered phenomenon in algebraic topology, called representation stability, to define equivariant neural networks that can be trained with data in a fixed dimension and then extended to accept inputs in any dimension. Our approach is user-friendly, requiring only the network architecture and the groups for equivariance, and can be combined with any training procedure. We provide a simple open-source implementation of our methods and offer preliminary numerical experiments.
    
[^7]: 概率深度学习的量子核混合方法

    Quantum Kernel Mixtures for Probabilistic Deep Learning. (arXiv:2305.18204v1 [cs.LG])

    [http://arxiv.org/abs/2305.18204](http://arxiv.org/abs/2305.18204)

    本文提出了一种量子核混合方法，可以用于表示连续和离散随机变量的联合概率分布。该框架允许构建可微分的模型，适用于密度估计、推理和采样，以及各种机器学习任务，包括生成建模和判别学习。

    

    本文提出了一种新的概率深度学习方法——量子核混合，它是从量子密度矩阵的数学形式中推导出来的。该方法提供了一种简单而有效的机制，用于表示连续和离散随机变量的联合概率分布。该框架允许构建可微分的模型，用于密度估计、推理和采样，从而能够整合到端到端的深度神经模型中。通过这样做，我们提供了一种多功能的边际和联合概率分布表示，可以开发一种可微分的、组合的和可逆的推理过程，涵盖了广泛的机器学习任务，包括密度估计、判别学习和生成建模。我们通过两个示例来说明该框架的广泛适用性：一个图像分类模型，它可以自然地转化为条件生成模型，得益于量子核混合的表示能力。

    This paper presents a novel approach to probabilistic deep learning (PDL), quantum kernel mixtures, derived from the mathematical formalism of quantum density matrices, which provides a simpler yet effective mechanism for representing joint probability distributions of both continuous and discrete random variables. The framework allows for the construction of differentiable models for density estimation, inference, and sampling, enabling integration into end-to-end deep neural models. In doing so, we provide a versatile representation of marginal and joint probability distributions that allows us to develop a differentiable, compositional, and reversible inference procedure that covers a wide range of machine learning tasks, including density estimation, discriminative learning, and generative modeling. We illustrate the broad applicability of the framework with two examples: an image classification model, which can be naturally transformed into a conditional generative model thanks to
    
[^8]: 平移不变核函数的正交展开

    Orthonormal Expansions for Translation-Invariant Kernels. (arXiv:2206.08648v3 [math.CA] UPDATED)

    [http://arxiv.org/abs/2206.08648](http://arxiv.org/abs/2206.08648)

    该论文提出了一种傅里叶分析技术，用于从$\mathscr{L}_2(\mathbb{R})$的正交基中构建平移不变核函数的正交基展开，实现了马特尔核函数、柯西核函数和高斯核函数的明确展开表达式。

    

    我们提出了一种用于构建平移不变核函数的正交基展开的傅里叶分析技术，该技术利用$\mathscr{L}_2(\mathbb{R})$上的正交基，得到了实轴上所有半整数阶马特尔核函数、柯西核函数以及高斯核函数的明确展开表达式，分别由相关的拉盖尔函数、有理函数和厄米函数表示。

    We present a general Fourier analytic technique for constructing orthonormal basis expansions of translation-invariant kernels from orthonormal bases of $\mathscr{L}_2(\mathbb{R})$. This allows us to derive explicit expansions on the real line for (i) Mat\'ern kernels of all half-integer orders in terms of associated Laguerre functions, (ii) the Cauchy kernel in terms of rational functions, and (iii) the Gaussian kernel in terms of Hermite functions.
    

