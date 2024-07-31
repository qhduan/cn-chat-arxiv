# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction](https://arxiv.org/abs/2403.07263) | 通过两步形式预测方法，本文实现了自适应边界框不确定性的量化，保证了对象边界框不确定性区间的覆盖率，包括了错误分类的对象，同时确保边界框区间能够适应物体大小，实现更平衡的覆盖率。 |
| [^2] | [Optimizing Adaptive Experiments: A Unified Approach to Regret Minimization and Best-Arm Identification](https://arxiv.org/abs/2402.10592) | 提出了一种统一模型，同时考虑了实验内部性能和实验后结果，在优化大规模人群中的表现方面提供了尖锐理论，揭示了新颖的见解 |
| [^3] | [Scalable Kernel Logistic Regression with Nystr\"om Approximation: Theoretical Analysis and Application to Discrete Choice Modelling](https://arxiv.org/abs/2402.06763) | 本文介绍了使用Nystr\"om近似方法解决大规模数据集上核逻辑回归的可扩展性问题。研究提供了理论分析并验证了不同的地标选择方法的性能。 |
| [^4] | [Information Leakage Detection through Approximate Bayes-optimal Prediction.](http://arxiv.org/abs/2401.14283) | 本论文通过建立一个理论框架，利用统计学习理论和信息论来准确量化和检测信息泄漏，通过近似贝叶斯预测的对数损失和准确性来准确估计互信息。 |
| [^5] | [Geometric Learning with Positively Decomposable Kernels.](http://arxiv.org/abs/2310.13821) | 本文提出了使用正可分解核的几何学习方法，该方法通过在RKKS中学习而不需要访问核的分解，为非欧几里德数据的核学习提供了一条路径，并为RKKS方法提供了理论基础。 |
| [^6] | [Monte Carlo inference for semiparametric Bayesian regression.](http://arxiv.org/abs/2306.05498) | 本文介绍了一种简单、通用和高效的半参数贝叶斯回归的蒙特卡洛推断策略，可用于联合后验一致性，即使经典的似然函数是难以处理或未知的。 |
| [^7] | [Sequential Knockoffs for Variable Selection in Reinforcement Learning.](http://arxiv.org/abs/2303.14281) | 本论文介绍了一种新颖的序列 Knockoffs (SEEK)算法，用于在强化学习系统中实现变量选择，该算法估计了最小充分状态，确保学习进程良好而不会减缓。 |

# 详细

[^1]: 通过两步形式预测实现自适应边界框不确定性

    Adaptive Bounding Box Uncertainties via Two-Step Conformal Prediction

    [https://arxiv.org/abs/2403.07263](https://arxiv.org/abs/2403.07263)

    通过两步形式预测方法，本文实现了自适应边界框不确定性的量化，保证了对象边界框不确定性区间的覆盖率，包括了错误分类的对象，同时确保边界框区间能够适应物体大小，实现更平衡的覆盖率。

    

    量化模型的预测不确定性对于像自动驾驶这样的安全关键应用至关重要。我们考虑为多物体检测量化这种不确定性。具体来说，我们利用形式预测来获得具有保证覆盖率的物体边界框不确定性区间。这样做的一个挑战是边界框的预测取决于物体的类别标签。因此，我们开发了一种新颖的两步形式方法，将对预测类别标签的不确定性传播到边界框的不确定性区间中。这样，我们的形式覆盖保证的有效性更广泛，包括了被错误分类的物体，确保它们在需要最大安全保证时的实用性。此外，我们研究了新颖的集成和分位数回归形式，以确保边界框区间能够适应物体大小，从而实现更平衡的覆盖率。

    arXiv:2403.07263v1 Announce Type: cross  Abstract: Quantifying a model's predictive uncertainty is essential for safety-critical applications such as autonomous driving. We consider quantifying such uncertainty for multi-object detection. In particular, we leverage conformal prediction to obtain uncertainty intervals with guaranteed coverage for object bounding boxes. One challenge in doing so is that bounding box predictions are conditioned on the object's class label. Thus, we develop a novel two-step conformal approach that propagates uncertainty in predicted class labels into the uncertainty intervals for the bounding boxes. This broadens the validity of our conformal coverage guarantees to include incorrectly classified objects, ensuring their usefulness when maximal safety assurances are required. Moreover, we investigate novel ensemble and quantile regression formulations to ensure the bounding box intervals are adaptive to object size, leading to a more balanced coverage across
    
[^2]: 优化自适应实验：最小化后悔和最佳臂识别的统一方法

    Optimizing Adaptive Experiments: A Unified Approach to Regret Minimization and Best-Arm Identification

    [https://arxiv.org/abs/2402.10592](https://arxiv.org/abs/2402.10592)

    提出了一种统一模型，同时考虑了实验内部性能和实验后结果，在优化大规模人群中的表现方面提供了尖锐理论，揭示了新颖的见解

    

    进行自适应实验的从业者通常面临两个竞争性优先级：通过在实验过程中有效地分配治疗来降低实验成本，以及迅速收集信息以结束实验并在整个人群中实施治疗。当前，文献意见分歧，有关最小化后悔的研究独立地处理前者的优先级，而有关最佳臂识别的研究则专注于后者。本文提出了一种统一模型，考虑到实验内部性能和实验后结果。我们随后提供了一个针对大规模人群的最佳性能的尖锐理论，将文献中的经典结果统一起来。这种统一还揭示了新的见解。例如，理论揭示了类似最近提出的顶部两个Thompson抽样算法等熟悉算法可被调整以优化广泛类别的目标。

    arXiv:2402.10592v1 Announce Type: new  Abstract: Practitioners conducting adaptive experiments often encounter two competing priorities: reducing the cost of experimentation by effectively assigning treatments during the experiment itself, and gathering information swiftly to conclude the experiment and implement a treatment across the population. Currently, the literature is divided, with studies on regret minimization addressing the former priority in isolation, and research on best-arm identification focusing solely on the latter. This paper proposes a unified model that accounts for both within-experiment performance and post-experiment outcomes. We then provide a sharp theory of optimal performance in large populations that unifies canonical results in the literature. This unification also uncovers novel insights. For example, the theory reveals that familiar algorithms, like the recently proposed top-two Thompson sampling algorithm, can be adapted to optimize a broad class of obj
    
[^3]: 使用Nystr\"om近似的可扩展核逻辑回归：理论分析和离散选择建模应用

    Scalable Kernel Logistic Regression with Nystr\"om Approximation: Theoretical Analysis and Application to Discrete Choice Modelling

    [https://arxiv.org/abs/2402.06763](https://arxiv.org/abs/2402.06763)

    本文介绍了使用Nystr\"om近似方法解决大规模数据集上核逻辑回归的可扩展性问题。研究提供了理论分析并验证了不同的地标选择方法的性能。

    

    将基于核的机器学习技术应用于使用大规模数据集的离散选择建模时，经常面临存储需求和模型中涉及的大量参数的挑战。这种复杂性影响了大规模模型的高效训练。本文通过引入Nystr\"om近似方法解决了可扩展性问题，用于大规模数据集上的核逻辑回归。研究首先进行了理论分析，其中：i) 对KLR解的集合进行了描述，ii) 给出了使用Nystr\"om近似的KLR解的上界，并最后描述了专门用于Nystr\"om KLR的优化算法的特化。之后，对Nystr\"om KLR进行了计算验证。测试了四种地标选择方法，包括基本均匀采样、k-means采样策略和基于杠杆得分的两种非均匀方法。这些策略的性能进行了评估。

    The application of kernel-based Machine Learning (ML) techniques to discrete choice modelling using large datasets often faces challenges due to memory requirements and the considerable number of parameters involved in these models. This complexity hampers the efficient training of large-scale models. This paper addresses these problems of scalability by introducing the Nystr\"om approximation for Kernel Logistic Regression (KLR) on large datasets. The study begins by presenting a theoretical analysis in which: i) the set of KLR solutions is characterised, ii) an upper bound to the solution of KLR with Nystr\"om approximation is provided, and finally iii) a specialisation of the optimisation algorithms to Nystr\"om KLR is described. After this, the Nystr\"om KLR is computationally validated. Four landmark selection methods are tested, including basic uniform sampling, a k-means sampling strategy, and two non-uniform methods grounded in leverage scores. The performance of these strategi
    
[^4]: 通过近似贝叶斯最优预测检测信息泄漏

    Information Leakage Detection through Approximate Bayes-optimal Prediction. (arXiv:2401.14283v1 [stat.ML])

    [http://arxiv.org/abs/2401.14283](http://arxiv.org/abs/2401.14283)

    本论文通过建立一个理论框架，利用统计学习理论和信息论来准确量化和检测信息泄漏，通过近似贝叶斯预测的对数损失和准确性来准确估计互信息。

    

    在今天的以数据驱动的世界中，公开可获得的信息的增加加剧了信息泄漏（IL）的挑战，引发了安全问题。IL涉及通过系统的可观察信息无意地将秘密（敏感）信息暴露给未经授权的方，传统的统计方法通过估计可观察信息和秘密信息之间的互信息（MI）来检测IL，面临维度灾难、收敛、计算复杂度和MI估计错误等挑战。此外，虽然新兴的监督机器学习（ML）方法在二进制系统敏感信息的检测上有效，但缺乏一个全面的理论框架。为了解决这些限制，我们使用统计学习理论和信息论建立了一个理论框架来准确量化和检测IL。我们证明了可以通过近似贝叶斯预测的对数损失和准确性来准确估计MI。

    In today's data-driven world, the proliferation of publicly available information intensifies the challenge of information leakage (IL), raising security concerns. IL involves unintentionally exposing secret (sensitive) information to unauthorized parties via systems' observable information. Conventional statistical approaches, which estimate mutual information (MI) between observable and secret information for detecting IL, face challenges such as the curse of dimensionality, convergence, computational complexity, and MI misestimation. Furthermore, emerging supervised machine learning (ML) methods, though effective, are limited to binary system-sensitive information and lack a comprehensive theoretical framework. To address these limitations, we establish a theoretical framework using statistical learning theory and information theory to accurately quantify and detect IL. We demonstrate that MI can be accurately estimated by approximating the log-loss and accuracy of the Bayes predict
    
[^5]: 使用正可分解核的几何学习

    Geometric Learning with Positively Decomposable Kernels. (arXiv:2310.13821v1 [cs.LG])

    [http://arxiv.org/abs/2310.13821](http://arxiv.org/abs/2310.13821)

    本文提出了使用正可分解核的几何学习方法，该方法通过在RKKS中学习而不需要访问核的分解，为非欧几里德数据的核学习提供了一条路径，并为RKKS方法提供了理论基础。

    

    核方法是机器学习中强大的工具。经典的核方法基于正定核，将数据空间映射到重现核希尔伯特空间(RKHS)。对于非欧几里德数据空间，很难找到正定核。在这种情况下，我们提出使用基于重现核控制空间(RKKS)的方法，这些方法只需要具有正分解的核。我们证明了在RKKS中学习时，并不需要访问这个分解。然后我们研究了使核正可分解的条件。我们证明在可处理的正则性假设下，不变核在齐次空间上允许正分解。这使得它们比正定核更容易构造，为非欧几里德数据的核学习提供了一条路径。同样，这为RKKS方法提供了一般的理论基础。

    Kernel methods are powerful tools in machine learning. Classical kernel methods are based on positive-definite kernels, which map data spaces into reproducing kernel Hilbert spaces (RKHS). For non-Euclidean data spaces, positive-definite kernels are difficult to come by. In this case, we propose the use of reproducing kernel Krein space (RKKS) based methods, which require only kernels that admit a positive decomposition. We show that one does not need to access this decomposition in order to learn in RKKS. We then investigate the conditions under which a kernel is positively decomposable. We show that invariant kernels admit a positive decomposition on homogeneous spaces under tractable regularity assumptions. This makes them much easier to construct than positive-definite kernels, providing a route for learning with kernels for non-Euclidean data. By the same token, this provides theoretical foundations for RKKS-based methods in general.
    
[^6]: 半参数贝叶斯回归的蒙特卡洛推断

    Monte Carlo inference for semiparametric Bayesian regression. (arXiv:2306.05498v1 [stat.ME])

    [http://arxiv.org/abs/2306.05498](http://arxiv.org/abs/2306.05498)

    本文介绍了一种简单、通用和高效的半参数贝叶斯回归的蒙特卡洛推断策略，可用于联合后验一致性，即使经典的似然函数是难以处理或未知的。

    

    数据转换对于参数回归模型的广泛适用性至关重要，但对于贝叶斯分析，联合推断转换和模型参数通常需要限制性参数转换或非参数表示，这对实现和理论分析来说计算效率低下且繁琐，限制了他们在实践中的可用性。本文介绍了一种简单、通用和高效的策略，直接通过将转换与独立变量和因变量的边缘分布相连的方式来定位未知转换和所有回归模型参数的后验分布，并通过贝叶斯非参数模型使用贝叶斯自举方法。关键是，这种方法在广泛的回归模型中都可以实现(1)联合后验一致性，包括多个模型错配情况，和(2)高效的蒙特卡罗算法，即使经典的似然函数是难以处理或未知的。

    Data transformations are essential for broad applicability of parametric regression models. However, for Bayesian analysis, joint inference of the transformation and model parameters typically involves restrictive parametric transformations or nonparametric representations that are computationally inefficient and cumbersome for implementation and theoretical analysis, which limits their usability in practice. This paper introduces a simple, general, and efficient strategy for joint posterior inference of an unknown transformation and all regression model parameters. The proposed approach directly targets the posterior distribution of the transformation by linking it with the marginal distributions of the independent and dependent variables, and then deploys a Bayesian nonparametric model via the Bayesian bootstrap. Crucially, this approach delivers (1) joint posterior consistency under general conditions, including multiple model misspecifications, and (2) efficient Monte Carlo (not Ma
    
[^7]: 基于序列 Knockoffs 的强化学习变量选择

    Sequential Knockoffs for Variable Selection in Reinforcement Learning. (arXiv:2303.14281v1 [stat.ML])

    [http://arxiv.org/abs/2303.14281](http://arxiv.org/abs/2303.14281)

    本论文介绍了一种新颖的序列 Knockoffs (SEEK)算法，用于在强化学习系统中实现变量选择，该算法估计了最小充分状态，确保学习进程良好而不会减缓。

    

    在强化学习的实际应用中，通常很难获得一个既简洁又满足马尔可夫属性的状态表示，而不需要使用先验知识。因此，常规做法是构造一个比必要的要大的状态，例如将连续时间点上的测量串联起来。然而，增加状态的维数可能会减缓学习进程并使学习策略模糊不清。我们引入了一个在马尔可夫决策过程(MDP)中的最小充分状态的概念，作为原始状态下最小的子向量，使该过程仍然是MDP，并且与原始过程共享相同的最优策略。我们提出了一种新颖的序列 Knockoffs (SEEK)算法，用于估计高维复杂非线性动力学系统中的最小充分状态。在大样本中，所提出的方法控制了假发现率，并且选择所有充分的变量的概率趋近于1。

    In real-world applications of reinforcement learning, it is often challenging to obtain a state representation that is parsimonious and satisfies the Markov property without prior knowledge. Consequently, it is common practice to construct a state which is larger than necessary, e.g., by concatenating measurements over contiguous time points. However, needlessly increasing the dimension of the state can slow learning and obfuscate the learned policy. We introduce the notion of a minimal sufficient state in a Markov decision process (MDP) as the smallest subvector of the original state under which the process remains an MDP and shares the same optimal policy as the original process. We propose a novel sequential knockoffs (SEEK) algorithm that estimates the minimal sufficient state in a system with high-dimensional complex nonlinear dynamics. In large samples, the proposed method controls the false discovery rate, and selects all sufficient variables with probability approaching one. As
    

