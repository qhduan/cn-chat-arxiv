# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Robust Learning for Optimal Dynamic Treatment Regimes with Observational Data](https://arxiv.org/abs/2404.00221) | 学习利用观测数据提出了一种逐步双重强健方法，通过向后归纳解决了最佳动态治疗方案的问题 |
| [^2] | [Delegating Data Collection in Decentralized Machine Learning.](http://arxiv.org/abs/2309.01837) | 这项研究在分散机器学习生态系统中研究了委托的数据收集问题，通过设计最优契约解决了模型质量评估的不确定性和对最优性能缺乏预先知识的挑战。 |
| [^3] | [Deep quantum neural networks form Gaussian processes.](http://arxiv.org/abs/2305.09957) | 本文证明了基于Haar随机酉或正交深量子神经网络的某些模型的输出会收敛于高斯过程。然而，这种高斯过程不能用于通过贝叶斯统计学来有效预测QNN的输出。 |
| [^4] | [Generalization on the Unseen, Logic Reasoning and Degree Curriculum.](http://arxiv.org/abs/2301.13105) | 本文研究了在逻辑推理任务中对未知数据的泛化能力，提供了网络架构在该设置下的表现证据，发现了一类网络模型在未知数据上学习了最小度插值器，并对长度普通化现象提供了解释。 |

# 详细

[^1]: 利用观测数据进行强健学习以获得最佳动态治疗方案

    Robust Learning for Optimal Dynamic Treatment Regimes with Observational Data

    [https://arxiv.org/abs/2404.00221](https://arxiv.org/abs/2404.00221)

    学习利用观测数据提出了一种逐步双重强健方法，通过向后归纳解决了最佳动态治疗方案的问题

    

    许多公共政策和医疗干预涉及其治疗分配中的动态性，治疗通常依据先前治疗的历史和相关特征对每个阶段的效果具有异质性。本文研究了统计学习最佳动态治疗方案(DTR)，根据个体的历史指导每个阶段的最佳治疗分配。我们提出了一种基于观测数据的逐步双重强健方法，在顺序可忽略性假设下学习最佳DTR。该方法通过向后归纳解决了顺序治疗分配问题，在每一步中，我们结合倾向评分和行动值函数(Q函数)的估计量，构建了政策价值的增强反向概率加权估计量。

    arXiv:2404.00221v1 Announce Type: cross  Abstract: Many public policies and medical interventions involve dynamics in their treatment assignments, where treatments are sequentially assigned to the same individuals across multiple stages, and the effect of treatment at each stage is usually heterogeneous with respect to the history of prior treatments and associated characteristics. We study statistical learning of optimal dynamic treatment regimes (DTRs) that guide the optimal treatment assignment for each individual at each stage based on the individual's history. We propose a step-wise doubly-robust approach to learn the optimal DTR using observational data under the assumption of sequential ignorability. The approach solves the sequential treatment assignment problem through backward induction, where, at each step, we combine estimators of propensity scores and action-value functions (Q-functions) to construct augmented inverse probability weighting estimators of values of policies 
    
[^2]: 委托分散机器学习中的数据收集

    Delegating Data Collection in Decentralized Machine Learning. (arXiv:2309.01837v1 [cs.LG])

    [http://arxiv.org/abs/2309.01837](http://arxiv.org/abs/2309.01837)

    这项研究在分散机器学习生态系统中研究了委托的数据收集问题，通过设计最优契约解决了模型质量评估的不确定性和对最优性能缺乏预先知识的挑战。

    

    受分散机器学习生态系统的出现的启发，我们研究了数据收集的委托问题。以契约理论为出发点，我们设计了解决两个基本机器学习挑战的最优和近似最优契约：模型质量评估的不确定性和对任何模型最优性能的缺乏知识。我们证明，通过简单的线性契约可以解决不确定性问题，即使委托人只有一个小的测试集，也能实现1-1/e的一等效用水平。此外，我们给出了委托人测试集大小的充分条件，可以达到对最优效用的逼近。为了解决对最优性能缺乏预先知识的问题，我们提出了一个凸问题，可以自适应和高效地计算最优契约。

    Motivated by the emergence of decentralized machine learning ecosystems, we study the delegation of data collection. Taking the field of contract theory as our starting point, we design optimal and near-optimal contracts that deal with two fundamental machine learning challenges: lack of certainty in the assessment of model quality and lack of knowledge regarding the optimal performance of any model. We show that lack of certainty can be dealt with via simple linear contracts that achieve 1-1/e fraction of the first-best utility, even if the principal has a small test set. Furthermore, we give sufficient conditions on the size of the principal's test set that achieves a vanishing additive approximation to the optimal utility. To address the lack of a priori knowledge regarding the optimal performance, we give a convex program that can adaptively and efficiently compute the optimal contract.
    
[^3]: 深度量子神经网络对应高斯过程

    Deep quantum neural networks form Gaussian processes. (arXiv:2305.09957v1 [quant-ph])

    [http://arxiv.org/abs/2305.09957](http://arxiv.org/abs/2305.09957)

    本文证明了基于Haar随机酉或正交深量子神经网络的某些模型的输出会收敛于高斯过程。然而，这种高斯过程不能用于通过贝叶斯统计学来有效预测QNN的输出。

    

    众所周知，从独立同分布的先验条件开始初始化的人工神经网络在隐藏层神经元数目足够大的极限下收敛到高斯过程。本文证明了量子神经网络（QNNs）也存在类似的结果。特别地，我们证明了基于Haar随机酉或正交深QNNs的某些模型的输出在希尔伯特空间维度$d$足够大时会收敛于高斯过程。由于输入状态、测量的可观测量以及酉矩阵的元素不独立等因素的作用，本文对这一结果的推导比经典情形更加微妙。我们分析的一个重要后果是，这个结果得到的高斯过程不能通过贝叶斯统计学来有效地预测QNN的输出。此外，我们的定理表明，Haar随机QNNs中的测量现象比以前认为的要更严重，我们证明了演员的集中现象。

    It is well known that artificial neural networks initialized from independent and identically distributed priors converge to Gaussian processes in the limit of large number of neurons per hidden layer. In this work we prove an analogous result for Quantum Neural Networks (QNNs). Namely, we show that the outputs of certain models based on Haar random unitary or orthogonal deep QNNs converge to Gaussian processes in the limit of large Hilbert space dimension $d$. The derivation of this result is more nuanced than in the classical case due the role played by the input states, the measurement observable, and the fact that the entries of unitary matrices are not independent. An important consequence of our analysis is that the ensuing Gaussian processes cannot be used to efficiently predict the outputs of the QNN via Bayesian statistics. Furthermore, our theorems imply that the concentration of measure phenomenon in Haar random QNNs is much worse than previously thought, as we prove that ex
    
[^4]: 对未知数据的泛化、逻辑推理和学位课程的概述

    Generalization on the Unseen, Logic Reasoning and Degree Curriculum. (arXiv:2301.13105v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.13105](http://arxiv.org/abs/2301.13105)

    本文研究了在逻辑推理任务中对未知数据的泛化能力，提供了网络架构在该设置下的表现证据，发现了一类网络模型在未知数据上学习了最小度插值器，并对长度普通化现象提供了解释。

    

    本文考虑了逻辑（布尔）函数的学习，重点在于对未知数据的泛化（GOTU）设定，这是一种强大的分布外泛化的案例。这是由于某些推理任务（例如算术/逻辑）中数据的丰富组合性质使得代表性数据采样具有挑战性，并且在GOTU下成功学习为第一个“推理”学习者展示了一个小插图。然后，我们研究了通过(S)GD训练的不同网络架构在GOTU下的表现，并提供了理论和实验证据，证明了一个类别的网络模型（包括Transformer的实例、随机特征模型和对角线线性网络）在未知数据上学习了最小度插值器。我们还提供了证据表明，其他具有更大学习速率或均场网络的实例达到了渗漏最小度解。这些发现带来了两个影响：（1）我们提供了对长度普通化的解释

    This paper considers the learning of logical (Boolean) functions with focus on the generalization on the unseen (GOTU) setting, a strong case of out-of-distribution generalization. This is motivated by the fact that the rich combinatorial nature of data in certain reasoning tasks (e.g., arithmetic/logic) makes representative data sampling challenging, and learning successfully under GOTU gives a first vignette of an 'extrapolating' or 'reasoning' learner. We then study how different network architectures trained by (S)GD perform under GOTU and provide both theoretical and experimental evidence that for a class of network models including instances of Transformers, random features models, and diagonal linear networks, a min-degree-interpolator is learned on the unseen. We also provide evidence that other instances with larger learning rates or mean-field networks reach leaky min-degree solutions. These findings lead to two implications: (1) we provide an explanation to the length genera
    

