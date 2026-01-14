# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Gradient-free online learning of subgrid-scale dynamics with neural emulators.](http://arxiv.org/abs/2310.19385) | 本文提出了一种利用神经仿真器在线训练亚网格参数化的算法，通过后验损失函数适应非可微分数值求解器，并通过时间积分步骤允许梯度传播。实验证明，将神经仿真器和参数化组件分别用相应的损失量进行训练是必要的，以最小化某些近似偏差的传播。 |
| [^2] | [Feed-Forward Optimization With Delayed Feedback for Neural Networks.](http://arxiv.org/abs/2304.13372) | 本文提出了一种延迟反馈的前馈神经网络优化方法F^3，使用延迟的误差信息来缩放梯度从而提高生物可行性和计算效率，具有较高的预测性能，为低能量训练和并行化提供了新思路。 |

# 详细

[^1]: 基于神经仿真器的无梯度在线学习亚网格尺度动力学

    Gradient-free online learning of subgrid-scale dynamics with neural emulators. (arXiv:2310.19385v2 [physics.comp-ph] UPDATED)

    [http://arxiv.org/abs/2310.19385](http://arxiv.org/abs/2310.19385)

    本文提出了一种利用神经仿真器在线训练亚网格参数化的算法，通过后验损失函数适应非可微分数值求解器，并通过时间积分步骤允许梯度传播。实验证明，将神经仿真器和参数化组件分别用相应的损失量进行训练是必要的，以最小化某些近似偏差的传播。

    

    本文提出了一种通用算法，用于在线训练基于机器学习的亚网格参数化，并通过后验损失函数适应非可微分数值求解器。所提出的方法利用神经仿真器训练简化状态空间求解器的近似值，然后通过时间积分步骤允许梯度传播。该算法能够在不计算原始求解器梯度的情况下恢复大部分在线策略的好处。实验证明，将神经仿真器和参数化组件分别用相应的损失量进行训练是必要的，以最小化某些近似偏差的传播。

    In this paper, we propose a generic algorithm to train machine learning-based subgrid parametrizations online, i.e., with $\textit{a posteriori}$ loss functions for non-differentiable numerical solvers. The proposed approach leverage neural emulators to train an approximation of the reduced state-space solver, which is then used to allows gradient propagation through temporal integration steps. The algorithm is able to recover most of the benefit of online strategies without having to compute the gradient of the original solver. It is demonstrated that training the neural emulator and parametrization components separately with respective loss quantities is necessary in order to minimize the propagation of some approximation bias.
    
[^2]: 延迟反馈的前馈优化神经网络

    Feed-Forward Optimization With Delayed Feedback for Neural Networks. (arXiv:2304.13372v1 [cs.LG])

    [http://arxiv.org/abs/2304.13372](http://arxiv.org/abs/2304.13372)

    本文提出了一种延迟反馈的前馈神经网络优化方法F^3，使用延迟的误差信息来缩放梯度从而提高生物可行性和计算效率，具有较高的预测性能，为低能量训练和并行化提供了新思路。

    

    反向传播长期以来一直受到生物学上的批评，因为它依赖于自然学习过程中不可行的概念。本文提出了一种替代方法来解决两个核心问题，即权重传输和更新锁定，以实现生物可行性和计算效率。我们引入了延迟反馈的前馈（F^3），通过利用延迟的误差信息作为样本级缩放因子来更准确地近似梯度，改进了先前的工作。我们发现，F^3将生物可行性训练算法和反向传播之间的预测性能差距缩小了高达96％。这证明了生物可行性训练的适用性，并为低能量训练和并行化开辟了有 promising 的新方向。

    Backpropagation has long been criticized for being biologically implausible, relying on concepts that are not viable in natural learning processes. This paper proposes an alternative approach to solve two core issues, i.e., weight transport and update locking, for biological plausibility and computational efficiency. We introduce Feed-Forward with delayed Feedback (F$^3$), which improves upon prior work by utilizing delayed error information as a sample-wise scaling factor to approximate gradients more accurately. We find that F$^3$ reduces the gap in predictive performance between biologically plausible training algorithms and backpropagation by up to 96%. This demonstrates the applicability of biologically plausible training and opens up promising new avenues for low-energy training and parallelization.
    

