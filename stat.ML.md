# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Weak Correlations as the Underlying Principle for Linearization of Gradient-Based Learning Systems.](http://arxiv.org/abs/2401.04013) | 本文研究了梯度下降学习算法在参数动力学中的线性结构，发现这种线性化现象是由于初始值附近假设函数的一阶和高阶导数之间的弱相关性所致。这一发现为深度学习模型的线性化提供了新的认识。 |
| [^2] | [A Differentially Private Weighted Empirical Risk Minimization Procedure and its Application to Outcome Weighted Learning.](http://arxiv.org/abs/2307.13127) | 本文提出了一种差分隐私加权经验风险最小化算法，可以在使用敏感数据的情况下保护隐私。这是第一个在权重ERM中应用差分隐私的算法，并且在一定的条件下提供了严格的DP保证。 |
| [^3] | [Convergence Guarantees for Stochastic Subgradient Methods in Nonsmooth Nonconvex Optimization.](http://arxiv.org/abs/2307.10053) | 本文研究了非平滑非凸优化中随机次梯度方法的收敛性质，并提出了一种新的框架，证明了其在单时间尺度和双时间尺度情况下的全局收敛性，包括了多种已知的SGD类型方法。对于有限和形式的目标函数，证明了这些方法能够在随机选择的步长和初始点上找到Clarke稳定点。 |

# 详细

[^1]: 以弱相关性作为梯度下降学习系统线性化的基本原则

    Weak Correlations as the Underlying Principle for Linearization of Gradient-Based Learning Systems. (arXiv:2401.04013v1 [cs.LG])

    [http://arxiv.org/abs/2401.04013](http://arxiv.org/abs/2401.04013)

    本文研究了梯度下降学习算法在参数动力学中的线性结构，发现这种线性化现象是由于初始值附近假设函数的一阶和高阶导数之间的弱相关性所致。这一发现为深度学习模型的线性化提供了新的认识。

    

    深度学习模型，如宽神经网络，可以被概念化为非线性动力学物理系统，其具有多个相互作用的自由度。在无限极限下，这些系统趋向于表现出简化的动力学。本文深入研究了基于梯度下降的学习算法，在其参数动力学中展示出与神经切向核类似的线性结构。我们发现，这种明显的线性化是因为在初始值附近，假设函数的一阶和高阶导数之间的弱相关性。这一洞见表明，这些弱相关性可能是此类系统中观察到的线性化的潜在原因。作为一个例证，我们展示了在宽度很大的神经网络中存在的这种弱相关性结构。利用线性和弱相关性之间的关系，我们推导出线性度偏离的一个界限。

    Deep learning models, such as wide neural networks, can be conceptualized as nonlinear dynamical physical systems characterized by a multitude of interacting degrees of freedom. Such systems in the infinite limit, tend to exhibit simplified dynamics. This paper delves into gradient descent-based learning algorithms, that display a linear structure in their parameter dynamics, reminiscent of the neural tangent kernel. We establish this apparent linearity arises due to weak correlations between the first and higher-order derivatives of the hypothesis function, concerning the parameters, taken around their initial values. This insight suggests that these weak correlations could be the underlying reason for the observed linearization in such systems. As a case in point, we showcase this weak correlations structure within neural networks in the large width limit. Exploiting the relationship between linearity and weak correlations, we derive a bound on deviations from linearity observed duri
    
[^2]: 一个差分隐私加权经验风险最小化算法及其在结果加权学习中的应用

    A Differentially Private Weighted Empirical Risk Minimization Procedure and its Application to Outcome Weighted Learning. (arXiv:2307.13127v1 [stat.ML])

    [http://arxiv.org/abs/2307.13127](http://arxiv.org/abs/2307.13127)

    本文提出了一种差分隐私加权经验风险最小化算法，可以在使用敏感数据的情况下保护隐私。这是第一个在权重ERM中应用差分隐私的算法，并且在一定的条件下提供了严格的DP保证。

    

    在经验风险最小化(ERM)框架中，使用包含个人信息的数据来构建预测模型是常见的做法。尽管这些模型在预测上可以非常准确，但使用敏感数据得到的结果可能容易受到隐私攻击。差分隐私(DP)是一种有吸引力的框架，可以通过提供数学上可证明的隐私损失界限来解决这些数据隐私问题。先前的工作主要集中在将DP应用于无权重的ERM中。我们考虑到了权重ERM(wERM)的重要推广。在wERM中，可以为每个个体的目标函数贡献分配不同的权重。在这个背景下，我们提出了第一个有差分隐私保障的wERM算法，并在一定的正则条件下提供了严格的理论证明。将现有的DP-ERM程序扩展到wERM为结果加权学习铺平了道路。

    It is commonplace to use data containing personal information to build predictive models in the framework of empirical risk minimization (ERM). While these models can be highly accurate in prediction, results obtained from these models with the use of sensitive data may be susceptible to privacy attacks. Differential privacy (DP) is an appealing framework for addressing such data privacy issues by providing mathematically provable bounds on the privacy loss incurred when releasing information from sensitive data. Previous work has primarily concentrated on applying DP to unweighted ERM. We consider an important generalization to weighted ERM (wERM). In wERM, each individual's contribution to the objective function can be assigned varying weights. In this context, we propose the first differentially private wERM algorithm, backed by a rigorous theoretical proof of its DP guarantees under mild regularity conditions. Extending the existing DP-ERM procedures to wERM paves a path to derivin
    
[^3]: 非平滑非凸优化中随机次梯度方法的收敛性保证

    Convergence Guarantees for Stochastic Subgradient Methods in Nonsmooth Nonconvex Optimization. (arXiv:2307.10053v1 [math.OC])

    [http://arxiv.org/abs/2307.10053](http://arxiv.org/abs/2307.10053)

    本文研究了非平滑非凸优化中随机次梯度方法的收敛性质，并提出了一种新的框架，证明了其在单时间尺度和双时间尺度情况下的全局收敛性，包括了多种已知的SGD类型方法。对于有限和形式的目标函数，证明了这些方法能够在随机选择的步长和初始点上找到Clarke稳定点。

    

    本文研究了随机梯度下降（SGD）方法及其变种在训练由非平滑激活函数构建的神经网络中的收敛性质。我们提出了一种新颖的框架，为更新动量项和变量的步长分配了不同的时间尺度。在一些温和的条件下，我们证明了我们提出的框架在单时间尺度和双时间尺度情况下的全局收敛性。我们还证明了我们提出的框架包含了很多已知的SGD类型方法，包括heavy-ball SGD、SignSGD、Lion、normalized SGD和clipped SGD。此外，当目标函数采用有限和形式时，我们基于我们提出的框架证明了这些SGD类型方法的收敛性质。特别地，在温和的假设下，我们证明了这些SGD类型方法在随机选择的步长和初始点上能够找到目标函数的Clarke稳定点。

    In this paper, we investigate the convergence properties of the stochastic gradient descent (SGD) method and its variants, especially in training neural networks built from nonsmooth activation functions. We develop a novel framework that assigns different timescales to stepsizes for updating the momentum terms and variables, respectively. Under mild conditions, we prove the global convergence of our proposed framework in both single-timescale and two-timescale cases. We show that our proposed framework encompasses a wide range of well-known SGD-type methods, including heavy-ball SGD, SignSGD, Lion, normalized SGD and clipped SGD. Furthermore, when the objective function adopts a finite-sum formulation, we prove the convergence properties for these SGD-type methods based on our proposed framework. In particular, we prove that these SGD-type methods find the Clarke stationary points of the objective function with randomly chosen stepsizes and initial points under mild assumptions. Preli
    

