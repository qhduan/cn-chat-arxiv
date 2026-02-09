# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Gradient Alignment with Prototype Feature for Fully Test-time Adaptation](https://arxiv.org/abs/2402.09004) | 在测试时间自适应中，我们提出了一种名为GAP的正则化方法，通过梯度对齐和原型特征，减轻了来自于错误分类伪标签熵最小化损失的不适当引导，显著改善了TTA方法。 |
| [^2] | [Keep or toss? A nonparametric score to evaluate solutions for noisy ICA](https://arxiv.org/abs/2401.08468) | 本文提出一种非参数分数来自适应选择适用于任意高斯噪声的ICA算法，并通过特征函数评估估计的混合矩阵质量，无需了解噪声分布参数。 |
| [^3] | [Bayesian Matrix Decomposition and Applications.](http://arxiv.org/abs/2302.11337) | 本书旨在介绍贝叶斯矩阵分解的概念和工具，并总结了贝叶斯矩阵分解方法在不同领域的应用。 |

# 详细

[^1]: 使用原型特征进行梯度对齐的完全测试时间自适应

    Gradient Alignment with Prototype Feature for Fully Test-time Adaptation

    [https://arxiv.org/abs/2402.09004](https://arxiv.org/abs/2402.09004)

    在测试时间自适应中，我们提出了一种名为GAP的正则化方法，通过梯度对齐和原型特征，减轻了来自于错误分类伪标签熵最小化损失的不适当引导，显著改善了TTA方法。

    

    在测试时间自适应（TTA）的背景下，我们提出了一种正则化方法，称为梯度对齐与原型特征（GAP），它可以减轻由于错误分类伪标签的熵最小化损失而导致的不适当引导。我们开发了一个梯度对齐损失，以精确地管理适应过程，确保对某些数据进行的更改不会对模型在其他数据上的性能产生负面影响。我们引入了一个类的原型特征作为负面影响的代理测量。为了使在TTA约束下GAP正则化方法可行，即模型只能访问没有标签的测试数据，我们通过两种方式修改了其公式：用分类器的权重向量近似原型特征，不使用反向传播计算梯度。我们证明了GAP在各种数据集上显著改善了TTA方法，证明了其多功能性和有效性。

    arXiv:2402.09004v1 Announce Type: cross Abstract: In context of Test-time Adaptation(TTA), we propose a regularizer, dubbed Gradient Alignment with Prototype feature (GAP), which alleviates the inappropriate guidance from entropy minimization loss from misclassified pseudo label. We developed a gradient alignment loss to precisely manage the adaptation process, ensuring that changes made for some data don't negatively impact the model's performance on other data. We introduce a prototype feature of a class as a proxy measure of the negative impact. To make GAP regularizer feasible under the TTA constraints, where model can only access test data without labels, we tailored its formula in two ways: approximating prototype features with weight vectors of the classifier, calculating gradient without back-propagation. We demonstrate GAP significantly improves TTA methods across various datasets, which proves its versatility and effectiveness.
    
[^2]: 保留还是丢弃？一种评估有噪声ICA解决方案的非参数分数

    Keep or toss? A nonparametric score to evaluate solutions for noisy ICA

    [https://arxiv.org/abs/2401.08468](https://arxiv.org/abs/2401.08468)

    本文提出一种非参数分数来自适应选择适用于任意高斯噪声的ICA算法，并通过特征函数评估估计的混合矩阵质量，无需了解噪声分布参数。

    

    独立分量分析（ICA）于20世纪80年代引入，作为盲源分离（BSS）的模型，指的是在对混合信号进行恢复时，对源信号或混合过程了解有限的情况下的过程。尽管有许多精密算法进行估计，但不同方法存在不同的缺点。在本文中，我们开发了一种非参数分数，用于自适应地选择ICA算法和任意高斯噪声。该分数的创新之处在于，它只假设数据具有有限的二阶矩，并使用特征函数来评估估计的混合矩阵的质量，而无需了解噪声分布的参数。此外，我们提出了一些新的对比函数和算法，它们具有与现有算法（如FASTICA和JADE）相同的快速计算性能，但在前者可能失败的领域中工作。尽管这些方法也可能存在缺点，

    Independent Component Analysis (ICA) was introduced in the 1980's as a model for Blind Source Separation (BSS), which refers to the process of recovering the sources underlying a mixture of signals, with little knowledge about the source signals or the mixing process. While there are many sophisticated algorithms for estimation, different methods have different shortcomings. In this paper, we develop a nonparametric score to adaptively pick the right algorithm for ICA with arbitrary Gaussian noise. The novelty of this score stems from the fact that it just assumes a finite second moment of the data and uses the characteristic function to evaluate the quality of the estimated mixing matrix without any knowledge of the parameters of the noise distribution. In addition, we propose some new contrast functions and algorithms that enjoy the same fast computability as existing algorithms like FASTICA and JADE but work in domains where the former may fail. While these also may have weaknesses,
    
[^3]: 贝叶斯矩阵分解及应用

    Bayesian Matrix Decomposition and Applications. (arXiv:2302.11337v2 [math.NA] UPDATED)

    [http://arxiv.org/abs/2302.11337](http://arxiv.org/abs/2302.11337)

    本书旨在介绍贝叶斯矩阵分解的概念和工具，并总结了贝叶斯矩阵分解方法在不同领域的应用。

    

    本书的唯一目的是为了给出贝叶斯矩阵分解概念和数学工具的自包含介绍，以便在后续章节中无缝引入矩阵分解技术及其应用。然而，我们清楚地意识到我们无法覆盖关于贝叶斯矩阵分解的所有有用和有趣的结果，并且由于讨论的范围有限，例如分析变分推理以进行优化的分离分析。我们将读者引导到贝叶斯分析领域的文献中，以便更详细地介绍相关领域。本书主要总结了重要的贝叶斯矩阵分解方法（例如实值分解、非负矩阵分解、贝叶斯插值分解）的目的和意义，以及这些方法的起源和复杂性对其应用提供的启示。数学先决条件是第一门课程。

    The sole aim of this book is to give a self-contained introduction to concepts and mathematical tools in Bayesian matrix decomposition in order to seamlessly introduce matrix decomposition techniques and their applications in subsequent sections. However, we clearly realize our inability to cover all the useful and interesting results concerning Bayesian matrix decomposition and given the paucity of scope to present this discussion, e.g., the separated analysis of variational inference for conducting the optimization. We refer the reader to literature in the field of Bayesian analysis for a more detailed introduction to the related fields.  This book is primarily a summary of purpose, significance of important Bayesian matrix decomposition methods, e.g., real-valued decomposition, nonnegative matrix factorization, Bayesian interpolative decomposition, and the origin and complexity of the methods which shed light on their applications. The mathematical prerequisite is a first course in 
    

