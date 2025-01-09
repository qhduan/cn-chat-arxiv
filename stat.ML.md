# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Two-Scale Complexity Measure for Deep Learning Models.](http://arxiv.org/abs/2401.09184) | 这篇论文介绍了一种用于统计模型的新容量测量2sED，可以可靠地限制泛化误差，并且与训练误差具有很好的相关性。此外，对于深度学习模型，我们展示了如何通过逐层迭代的方法有效地近似2sED，从而处理大量参数的情况。 |
| [^2] | [Manifold Filter-Combine Networks.](http://arxiv.org/abs/2307.04056) | 这篇论文介绍了一类称为流形滤波-组合网络的大型流形神经网络。作者提出了一种基于构建数据驱动图的方法来实现这种网络，并提供了收敛到连续极限的充分条件，其收敛速度不依赖于滤波器数量。 |
| [^3] | [Learning High-Dimensional Nonparametric Differential Equations via Multivariate Occupation Kernel Functions.](http://arxiv.org/abs/2306.10189) | 本论文提出了一种线性方法，通过多元占位核函数在高维状态空间中学习非参数ODE系统，可以解决显式公式按二次方缩放的问题。这种方法在高度非线性的数据和图像数据中都具有通用性。 |
| [^4] | [Asymptotic Inference for Multi-Stage Stationary Treatment Policy with High Dimensional Features.](http://arxiv.org/abs/2301.12553) | 本研究填补了在高维特征变量存在的情况下，对于多阶段静态治疗策略本身进行推断的工作空白，提出了一种增强的估计器以提高价值函数的准确性。 |

# 详细

[^1]: 深度学习模型的两尺度复杂度测量

    A Two-Scale Complexity Measure for Deep Learning Models. (arXiv:2401.09184v1 [stat.ML])

    [http://arxiv.org/abs/2401.09184](http://arxiv.org/abs/2401.09184)

    这篇论文介绍了一种用于统计模型的新容量测量2sED，可以可靠地限制泛化误差，并且与训练误差具有很好的相关性。此外，对于深度学习模型，我们展示了如何通过逐层迭代的方法有效地近似2sED，从而处理大量参数的情况。

    

    我们引入了一种基于有效维度的统计模型新容量测量2sED。这个新的数量在对模型进行温和假设的情况下，可以可靠地限制泛化误差。此外，对于标准数据集和流行的模型架构的模拟结果表明，2sED与训练误差具有很好的相关性。对于马尔可夫模型，我们展示了如何通过逐层迭代的方法有效地从下方近似2sED，从而解决具有大量参数的深度学习模型。模拟结果表明，这种近似对不同的突出模型和数据集都很好。

    We introduce a novel capacity measure 2sED for statistical models based on the effective dimension. The new quantity provably bounds the generalization error under mild assumptions on the model. Furthermore, simulations on standard data sets and popular model architectures show that 2sED correlates well with the training error. For Markovian models, we show how to efficiently approximate 2sED from below through a layerwise iterative approach, which allows us to tackle deep learning models with a large number of parameters. Simulation results suggest that the approximation is good for different prominent models and data sets.
    
[^2]: 流形滤波-组合网络

    Manifold Filter-Combine Networks. (arXiv:2307.04056v1 [stat.ML])

    [http://arxiv.org/abs/2307.04056](http://arxiv.org/abs/2307.04056)

    这篇论文介绍了一类称为流形滤波-组合网络的大型流形神经网络。作者提出了一种基于构建数据驱动图的方法来实现这种网络，并提供了收敛到连续极限的充分条件，其收敛速度不依赖于滤波器数量。

    

    我们介绍了一类大型流形神经网络(MNNs)，我们称之为流形滤波-组合网络。这个类别包括了Wang、Ruiz和Ribeiro之前的研究中考虑的MNNs，流形散射变换(一种基于小波的神经网络模型)，以及其他有趣的之前在文献中未考虑的示例，如Kipf和Welling的图卷积网络的流形等效。然后，我们考虑了一种基于构建数据驱动图的方法，用于在没有对流形有全局知识的情况下实现这样的网络，而只能访问有限数量的样本点。我们提供了网络在样本点数趋于无穷大时能够保证收敛到其连续极限的充分条件。与之前的工作(主要关注特定的MNN结构和图构建)不同，我们的收敛速度并不依赖于使用的滤波器数量。而且，它表现出线性的收敛速度。

    We introduce a large class of manifold neural networks (MNNs) which we call Manifold Filter-Combine Networks. This class includes as special cases, the MNNs considered in previous work by Wang, Ruiz, and Ribeiro, the manifold scattering transform (a wavelet-based model of neural networks), and other interesting examples not previously considered in the literature such as the manifold equivalent of Kipf and Welling's graph convolutional network. We then consider a method, based on building a data-driven graph, for implementing such networks when one does not have global knowledge of the manifold, but merely has access to finitely many sample points. We provide sufficient conditions for the network to provably converge to its continuum limit as the number of sample points tends to infinity. Unlike previous work (which focused on specific MNN architectures and graph constructions), our rate of convergence does not explicitly depend on the number of filters used. Moreover, it exhibits line
    
[^3]: 通过多元占位核函数学习高维非参数微分方程

    Learning High-Dimensional Nonparametric Differential Equations via Multivariate Occupation Kernel Functions. (arXiv:2306.10189v1 [stat.ML])

    [http://arxiv.org/abs/2306.10189](http://arxiv.org/abs/2306.10189)

    本论文提出了一种线性方法，通过多元占位核函数在高维状态空间中学习非参数ODE系统，可以解决显式公式按二次方缩放的问题。这种方法在高度非线性的数据和图像数据中都具有通用性。

    

    从$d$维状态空间中$n$个轨迹快照中学习非参数的常微分方程（ODE）系统需要学习$d$个函数。除非具有额外的系统属性知识，例如稀疏性和对称性，否则显式的公式按二次方缩放。在这项工作中，我们提出了一种使用向量值再生核希尔伯特空间提供的隐式公式学习的线性方法。通过将ODE重写为更弱的积分形式，我们随后进行最小化并推导出我们的学习算法。最小化问题的解向量场依赖于与解轨迹相关的多元占位核函数。我们通过对高度非线性的模拟和真实数据进行实验证实了我们的方法，其中$d$可能超过100。我们进一步通过从图像数据学习非参数一阶拟线性偏微分方程展示了所提出的方法的多样性。

    Learning a nonparametric system of ordinary differential equations (ODEs) from $n$ trajectory snapshots in a $d$-dimensional state space requires learning $d$ functions of $d$ variables. Explicit formulations scale quadratically in $d$ unless additional knowledge about system properties, such as sparsity and symmetries, is available. In this work, we propose a linear approach to learning using the implicit formulation provided by vector-valued Reproducing Kernel Hilbert Spaces. By rewriting the ODEs in a weaker integral form, which we subsequently minimize, we derive our learning algorithm. The minimization problem's solution for the vector field relies on multivariate occupation kernel functions associated with the solution trajectories. We validate our approach through experiments on highly nonlinear simulated and real data, where $d$ may exceed 100. We further demonstrate the versatility of the proposed method by learning a nonparametric first order quasilinear partial differential 
    
[^4]: 多阶段静态治疗策略的高维特征渐近推断

    Asymptotic Inference for Multi-Stage Stationary Treatment Policy with High Dimensional Features. (arXiv:2301.12553v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2301.12553](http://arxiv.org/abs/2301.12553)

    本研究填补了在高维特征变量存在的情况下，对于多阶段静态治疗策略本身进行推断的工作空白，提出了一种增强的估计器以提高价值函数的准确性。

    

    动态治疗规则是一系列针对个体特征量身定制的多阶段决策函数。在实践中，一类重要的治疗策略是多阶段静态治疗策略，其使用相同的决策函数来指定治疗分配概率，在决策时基于同时包括基线变量（例如人口统计学）和时变变量（例如常规检测到的疾病生物标志物）的一组特征。虽然已经有大量文献对与动态治疗策略相关的价值函数进行有效推断，但在高维特征变量存在的情况下，对于治疗策略本身的工作还很少。我们旨在填补这项工作的空白。具体而言，我们首先基于增强的倒数权重估计器估计多阶段静态治疗策略，以提高价值函数的准确性。

    Dynamic treatment rules or policies are a sequence of decision functions over multiple stages that are tailored to individual features. One important class of treatment policies for practice, namely multi-stage stationary treatment policies, prescribe treatment assignment probabilities using the same decision function over stages, where the decision is based on the same set of features consisting of both baseline variables (e.g., demographics) and time-evolving variables (e.g., routinely collected disease biomarkers). Although there has been extensive literature to construct valid inference for the value function associated with the dynamic treatment policies, little work has been done for the policies themselves, especially in the presence of high dimensional feature variables. We aim to fill in the gap in this work. Specifically, we first estimate the multistage stationary treatment policy based on an augmented inverse probability weighted estimator for the value function to increase
    

