# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stochastic Gradient Descent for Additive Nonparametric Regression](https://arxiv.org/abs/2401.00691) | 本文介绍了一种用于训练加性模型的随机梯度下降算法，具有良好的内存存储和计算要求。在规范很好的情况下，通过仔细选择学习率，可以实现最小和最优的风险。 |
| [^2] | [Are Ensembles Getting Better all the Time?](https://arxiv.org/abs/2311.17885) | 只有当考虑的损失函数为凸函数时，集成模型一直在变得更好，当损失函数为非凸函数时，好模型的集成变得更好，坏模型的集成变得更糟。 |
| [^3] | [Generative Modelling of L\'{e}vy Area for High Order SDE Simulation.](http://arxiv.org/abs/2308.02452) | 本文提出了一种基于深度学习的模型LévyGAN，用于生成条件于布朗增量的Lévy区域的近似样本。通过“桥翻转”操作，输出的样本可以精确匹配所有奇数阶矩，解决了非高斯性质下的抽样困难问题。 |
| [^4] | [The Power of Preconditioning in Overparameterized Low-Rank Matrix Sensing.](http://arxiv.org/abs/2302.01186) | 该研究提出了ScaledGD(𝜆)方法，相较于传统梯度下降法更加鲁棒，并且在处理低秩矩阵感知问题时具有很好的表现。 |

# 详细

[^1]: 添加非参数回归的随机梯度下降

    Stochastic Gradient Descent for Additive Nonparametric Regression

    [https://arxiv.org/abs/2401.00691](https://arxiv.org/abs/2401.00691)

    本文介绍了一种用于训练加性模型的随机梯度下降算法，具有良好的内存存储和计算要求。在规范很好的情况下，通过仔细选择学习率，可以实现最小和最优的风险。

    

    本文介绍了一种用于训练加性模型的迭代算法，该算法具有良好的内存存储和计算要求。该算法可以看作是对组件函数的截断基扩展的系数应用随机梯度下降的函数对应物。我们证明了得到的估计量满足一个奥拉克不等式，允许模型错误规范。在规范很好的情况下，通过在训练的三个不同阶段仔细选择学习率，我们证明了其风险在数据维度和训练样本大小的依赖方面是最小和最优的。通过在两个实际数据集上将该方法与传统的反向拟合进行比较，我们进一步说明了计算优势。

    This paper introduces an iterative algorithm for training additive models that enjoys favorable memory storage and computational requirements. The algorithm can be viewed as the functional counterpart of stochastic gradient descent, applied to the coefficients of a truncated basis expansion of the component functions. We show that the resulting estimator satisfies an oracle inequality that allows for model mis-specification. In the well-specified setting, by choosing the learning rate carefully across three distinct stages of training, we demonstrate that its risk is minimax optimal in terms of the dependence on the dimensionality of the data and the size of the training sample. We further illustrate the computational benefits by comparing the approach with traditional backfitting on two real-world datasets.
    
[^2]: 集成模型是否一直在不断进步？

    Are Ensembles Getting Better all the Time?

    [https://arxiv.org/abs/2311.17885](https://arxiv.org/abs/2311.17885)

    只有当考虑的损失函数为凸函数时，集成模型一直在变得更好，当损失函数为非凸函数时，好模型的集成变得更好，坏模型的集成变得更糟。

    

    集成方法结合了几个基础模型的预测。本研究探讨了是否始终将更多模型纳入集成会提升其平均性能。这个问题取决于所考虑的集成类型，以及选择的预测度量。我们专注于所有集成成员被预期表现相同的情况，这是几种流行方法（如随机森林或深度集成）的情况。在这种设定下，我们表明，只有当考虑的损失函数为凸函数时，集成才会一直变得更好。更具体地说，在这种情况下，集成的平均损失是模型数量的减函数。当损失函数为非凸函数时，我们展示了一系列结果，可以总结为：好模型的集成会变得更好，坏模型的集成会变得更糟。为此，我们证明了关于尾概率单调性的新结果。

    arXiv:2311.17885v2 Announce Type: replace-cross  Abstract: Ensemble methods combine the predictions of several base models. We study whether or not including more models always improves their average performance. This question depends on the kind of ensemble considered, as well as the predictive metric chosen. We focus on situations where all members of the ensemble are a priori expected to perform as well, which is the case of several popular methods such as random forests or deep ensembles. In this setting, we show that ensembles are getting better all the time if, and only if, the considered loss function is convex. More precisely, in that case, the average loss of the ensemble is a decreasing function of the number of models. When the loss function is nonconvex, we show a series of results that can be summarised as: ensembles of good models keep getting better, and ensembles of bad models keep getting worse. To this end, we prove a new result on the monotonicity of tail probabiliti
    
[^3]: 对高阶SDE模拟的Lévy区域进行生成建模

    Generative Modelling of L\'{e}vy Area for High Order SDE Simulation. (arXiv:2308.02452v1 [stat.ML])

    [http://arxiv.org/abs/2308.02452](http://arxiv.org/abs/2308.02452)

    本文提出了一种基于深度学习的模型LévyGAN，用于生成条件于布朗增量的Lévy区域的近似样本。通过“桥翻转”操作，输出的样本可以精确匹配所有奇数阶矩，解决了非高斯性质下的抽样困难问题。

    

    众所周知，当数值模拟SDE的解时，要实现强收敛速率超过O(\sqrt{h})（其中h为步长），需要使用某些布朗运动的迭代积分，通常称为其“Lévy区域”。然而，由于其非高斯性质，对于d维布朗运动（d>2），目前没有快速近似抽样算法。本文提出了LévyGAN，一种基于深度学习的模型，用于生成条件于布朗增量的Lévy区域的近似样本。通过“桥翻转”操作，输出的样本可以精确匹配所有奇数阶矩。我们的生成器采用经过量身定制的GNN-inspired架构，强制输出分布与条件变量之间的正确依赖结构。此外，我们还结合了基于特征函数的数学原理的判别性归一化操作。

    It is well known that, when numerically simulating solutions to SDEs, achieving a strong convergence rate better than O(\sqrt{h}) (where h is the step size) requires the use of certain iterated integrals of Brownian motion, commonly referred to as its "L\'{e}vy areas". However, these stochastic integrals are difficult to simulate due to their non-Gaussian nature and for a d-dimensional Brownian motion with d > 2, no fast almost-exact sampling algorithm is known.  In this paper, we propose L\'{e}vyGAN, a deep-learning-based model for generating approximate samples of L\'{e}vy area conditional on a Brownian increment. Due to our "Bridge-flipping" operation, the output samples match all joint and conditional odd moments exactly. Our generator employs a tailored GNN-inspired architecture, which enforces the correct dependency structure between the output distribution and the conditioning variable. Furthermore, we incorporate a mathematically principled characteristic-function based discrim
    
[^4]: 预条件对超参化低秩矩阵感知的影响

    The Power of Preconditioning in Overparameterized Low-Rank Matrix Sensing. (arXiv:2302.01186v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.01186](http://arxiv.org/abs/2302.01186)

    该研究提出了ScaledGD(𝜆)方法，相较于传统梯度下降法更加鲁棒，并且在处理低秩矩阵感知问题时具有很好的表现。

    

    本文提出了ScaledGD(𝜆)方法来解决低秩矩阵感知中矩阵可能病态以及真实秩未知的问题。该方法使用超参式表示，从一个小的随机初始化开始，通过使用特定形式的阻尼预条件梯度下降来对抗超参化和病态曲率的影响。与基准梯度下降（GD）相比，尽管预处理需要轻微的计算开销，但ScaledGD（𝜆）在面对病态问题时表现出了出色的鲁棒性。在高斯设计下，ScaledGD($\lambda$) 会在仅迭代数对数级别的情况下，以线性速率收敛到真实的低秩矩阵。

    We propose $\textsf{ScaledGD($\lambda$)}$, a preconditioned gradient descent method to tackle the low-rank matrix sensing problem when the true rank is unknown, and when the matrix is possibly ill-conditioned. Using overparametrized factor representations, $\textsf{ScaledGD($\lambda$)}$ starts from a small random initialization, and proceeds by gradient descent with a specific form of damped preconditioning to combat bad curvatures induced by overparameterization and ill-conditioning. At the expense of light computational overhead incurred by preconditioners, $\textsf{ScaledGD($\lambda$)}$ is remarkably robust to ill-conditioning compared to vanilla gradient descent ($\textsf{GD}$) even with overprameterization. Specifically, we show that, under the Gaussian design, $\textsf{ScaledGD($\lambda$)}$ converges to the true low-rank matrix at a constant linear rate after a small number of iterations that scales only logarithmically with respect to the condition number and the problem dimensi
    

