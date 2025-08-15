# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hypothesis Spaces for Deep Learning](https://arxiv.org/abs/2403.03353) | 本文介绍了一种应用深度神经网络的深度学习假设空间，并构建了一个再生核Banach空间，研究了正则化学习和最小插值问题，证明了学习模型的解可以表示为线性组合。 |
| [^2] | [Continuous Tensor Relaxation for Finding Diverse Solutions in Combinatorial Optimization Problems](https://arxiv.org/abs/2402.02190) | 本研究提出了连续张量放松方法(CTRA)，用于在组合优化问题中寻找多样化的解决方案。CTRA通过对离散决策变量进行连续放松，解决了寻找多样化解决方案的挑战。 |

# 详细

[^1]: 深度学习的假设空间

    Hypothesis Spaces for Deep Learning

    [https://arxiv.org/abs/2403.03353](https://arxiv.org/abs/2403.03353)

    本文介绍了一种应用深度神经网络的深度学习假设空间，并构建了一个再生核Banach空间，研究了正则化学习和最小插值问题，证明了学习模型的解可以表示为线性组合。

    

    本文介绍了一种应用深度神经网络（DNNs）的深度学习假设空间。通过将DNN视为两个变量的函数，即物理变量和参数变量，我们考虑了DNNs的原始集合，参数变量位于由DNNs的权重矩阵和偏置决定的一组深度和宽度中。然后在弱*拓扑中完成原始DNN集合的线性跨度，以构建一个物理变量函数的Banach空间。我们证明所构造的Banach空间是一个再生核Banach空间（RKBS），并构造其再生核。通过为学习模型的解建立表达定理，我们研究了两个学习模型，正则化学习和最小插值问题在结果RKBS中。表达定理揭示了这些学习模型的解可以表示为线性组合

    arXiv:2403.03353v1 Announce Type: cross  Abstract: This paper introduces a hypothesis space for deep learning that employs deep neural networks (DNNs). By treating a DNN as a function of two variables, the physical variable and parameter variable, we consider the primitive set of the DNNs for the parameter variable located in a set of the weight matrices and biases determined by a prescribed depth and widths of the DNNs. We then complete the linear span of the primitive DNN set in a weak* topology to construct a Banach space of functions of the physical variable. We prove that the Banach space so constructed is a reproducing kernel Banach space (RKBS) and construct its reproducing kernel. We investigate two learning models, regularized learning and minimum interpolation problem in the resulting RKBS, by establishing representer theorems for solutions of the learning models. The representer theorems unfold that solutions of these learning models can be expressed as linear combination of
    
[^2]: 在组合优化问题中寻找多样化解决方案的连续张量放松方法

    Continuous Tensor Relaxation for Finding Diverse Solutions in Combinatorial Optimization Problems

    [https://arxiv.org/abs/2402.02190](https://arxiv.org/abs/2402.02190)

    本研究提出了连续张量放松方法(CTRA)，用于在组合优化问题中寻找多样化的解决方案。CTRA通过对离散决策变量进行连续放松，解决了寻找多样化解决方案的挑战。

    

    在组合优化问题中，寻找最佳解是最常见的目标。然而，在实际场景中，单一解决方案可能不适用，因为目标函数和约束条件只是原始现实世界情况的近似值。为了解决这个问题，寻找具有不同特征的多样化解决方案和约束严重性的变化成为自然的方向。这种策略提供了在后处理过程中选择合适解决方案的灵活性。然而，发现这些多样化解决方案比确定单一解决方案更具挑战性。为了克服这一挑战，本研究引入了连续张量松弛退火 (CTRA) 方法，用于基于无监督学习的组合优化求解器。CTRA通过扩展连续松弛方法，将离散决策变量转换为连续张量，同时解决了多个问题。该方法找到了不同特征的多样化解决方案和约束严重性的变化。

    Finding the best solution is the most common objective in combinatorial optimization (CO) problems. However, a single solution may not be suitable in practical scenarios, as the objective functions and constraints are only approximations of original real-world situations. To tackle this, finding (i) "heterogeneous solutions", diverse solutions with distinct characteristics, and (ii) "penalty-diversified solutions", variations in constraint severity, are natural directions. This strategy provides the flexibility to select a suitable solution during post-processing. However, discovering these diverse solutions is more challenging than identifying a single solution. To overcome this challenge, this study introduces Continual Tensor Relaxation Annealing (CTRA) for unsupervised-learning-based CO solvers. CTRA addresses various problems simultaneously by extending the continual relaxation approach, which transforms discrete decision variables into continual tensors. This method finds heterog
    

