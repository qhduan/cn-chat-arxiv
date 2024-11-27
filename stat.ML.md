# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [On the Convergence Rate of the Stochastic Gradient Descent (SGD) and application to a modified policy gradient for the Multi Armed Bandit](https://arxiv.org/abs/2402.06388) | 该论文证明了当学习速率按照逆时间衰减规则时，随机梯度下降（SGD）的收敛速度，并应用于修改的带有L2正则化的策略梯度多臂赌博机（MAB）的收敛性分析。 |
| [^2] | [A Multi-Grained Symmetric Differential Equation Model for Learning Protein-Ligand Binding Dynamics.](http://arxiv.org/abs/2401.15122) | 提出了一种能够促进数值MD模拟并有效模拟蛋白质-配体结合动力学的NeuralMD方法，采用物理信息多级对称框架，实现了准确建模多级蛋白质-配体相互作用。 |
| [^3] | [Finite-Time Decoupled Convergence in Nonlinear Two-Time-Scale Stochastic Approximation.](http://arxiv.org/abs/2401.03893) | 本研究探讨了非线性双时间尺度随机逼近中有限时间解耦收敛的潜力，并通过引入嵌套局部线性条件证明了其可行性。 |
| [^4] | [Universal approximation with complex-valued deep narrow neural networks.](http://arxiv.org/abs/2305.16910) | 本文研究了具有有界宽度和任意深度的复值神经网络的普适性，发现当且仅当激活函数既不是全纯的，也不是反全纯的，也不是 $\mathbb{R}$-仿射的时，深窄的复值网络具有普适逼近能力。我们还发现足够的宽度依赖于考虑的激活函数，对于一类可允许的激活函数，宽度为 $n+m+4$ 是足够的。 |
| [^5] | [Conformal Prediction is Robust to Dispersive Label Noise.](http://arxiv.org/abs/2209.14295) | 本研究研究了Conformal Prediction方法对于标签噪声具有鲁棒性。我们找出了构建可以正确覆盖无噪声真实标签的不确定性集合的条件，并提出了对具有噪声标签的一般损失函数进行正确控制的要求。实验证明，在对抗性案例之外，使用Conformal Prediction和风险控制技术可以实现对干净真实标签的保守风险。我们还提出了一种有界尺寸噪声修正的方法，以确保实现正确的真实标签风险。 |

# 详细

[^1]: 关于随机梯度下降（SGD）的收敛速度及其在修改的多臂赌博机上的策略梯度应用

    On the Convergence Rate of the Stochastic Gradient Descent (SGD) and application to a modified policy gradient for the Multi Armed Bandit

    [https://arxiv.org/abs/2402.06388](https://arxiv.org/abs/2402.06388)

    该论文证明了当学习速率按照逆时间衰减规则时，随机梯度下降（SGD）的收敛速度，并应用于修改的带有L2正则化的策略梯度多臂赌博机（MAB）的收敛性分析。

    

    我们提出了一个自包含的证明，证明了当学习速率遵循逆时间衰减规则时，随机梯度下降（SGD）的收敛速度；接下来，我们将这些结果应用于带有L2正则化的修改的策略梯度多臂赌博机（MAB）的收敛性分析。

    We present a self-contained proof of the convergence rate of the Stochastic Gradient Descent (SGD) when the learning rate follows an inverse time decays schedule; we next apply the results to the convergence of a modified form of policy gradient Multi-Armed Bandit (MAB) with $L2$ regularization.
    
[^2]: 一种多级对称微分方程模型用于学习蛋白质-配体结合动力学

    A Multi-Grained Symmetric Differential Equation Model for Learning Protein-Ligand Binding Dynamics. (arXiv:2401.15122v1 [cs.LG])

    [http://arxiv.org/abs/2401.15122](http://arxiv.org/abs/2401.15122)

    提出了一种能够促进数值MD模拟并有效模拟蛋白质-配体结合动力学的NeuralMD方法，采用物理信息多级对称框架，实现了准确建模多级蛋白质-配体相互作用。

    

    在药物发现中，蛋白质-配体结合的分子动力学（MD）模拟提供了一种强大的工具，用于预测结合亲和力，估计运输性能和探索口袋位点。通过改进数值方法以及最近通过机器学习（ML）方法增强MD模拟的效率已经有了很长的历史。然而，仍然存在一些挑战，例如准确建模扩展时间尺度的模拟。为了解决这个问题，我们提出了NeuralMD，这是第一个能够促进数值MD并提供准确的蛋白质-配体结合动力学模拟的ML辅助工具。我们提出了一个合理的方法，将一种新的物理信息多级对称框架纳入模型中。具体而言，我们提出了（1）一个使用向量框架满足群对称性并捕获多级蛋白质-配体相互作用的BindingNet模型，以及（2）一个增强的神经微分方程求解器，学习轨迹的演化。

    In drug discovery, molecular dynamics (MD) simulation for protein-ligand binding provides a powerful tool for predicting binding affinities, estimating transport properties, and exploring pocket sites. There has been a long history of improving the efficiency of MD simulations through better numerical methods and, more recently, by augmenting them with machine learning (ML) methods. Yet, challenges remain, such as accurate modeling of extended-timescale simulations. To address this issue, we propose NeuralMD, the first ML surrogate that can facilitate numerical MD and provide accurate simulations of protein-ligand binding dynamics. We propose a principled approach that incorporates a novel physics-informed multi-grained group symmetric framework. Specifically, we propose (1) a BindingNet model that satisfies group symmetry using vector frames and captures the multi-level protein-ligand interactions, and (2) an augmented neural differential equation solver that learns the trajectory und
    
[^3]: 非线性双时间尺度随机逼近中的有限时间解耦收敛

    Finite-Time Decoupled Convergence in Nonlinear Two-Time-Scale Stochastic Approximation. (arXiv:2401.03893v1 [math.OC])

    [http://arxiv.org/abs/2401.03893](http://arxiv.org/abs/2401.03893)

    本研究探讨了非线性双时间尺度随机逼近中有限时间解耦收敛的潜力，并通过引入嵌套局部线性条件证明了其可行性。

    

    在双时间尺度随机逼近中，使用不同的步长以不同的速度更新两个迭代，每次更新都会影响另一个。先前的线性双时间尺度随机逼近研究发现，这些更新的均方误差的收敛速度仅仅取决于它们各自的步长，导致了所谓的解耦收敛。然而，在非线性随机逼近中实现这种解耦收敛的可能性仍不明确。我们的研究探讨了非线性双时间尺度随机逼近中有限时间解耦收敛的潜力。我们发现，在较弱的Lipschitz条件下，传统分析无法实现解耦收敛。这一发现在数值上得到了进一步的支持。但是通过引入一个嵌套局部线性条件，我们证明了在适当选择与平滑度相关的步长的情况下，解耦收敛仍然是可行的。

    In two-time-scale stochastic approximation (SA), two iterates are updated at varying speeds using different step sizes, with each update influencing the other. Previous studies in linear two-time-scale SA have found that the convergence rates of the mean-square errors for these updates are dependent solely on their respective step sizes, leading to what is referred to as decoupled convergence. However, the possibility of achieving this decoupled convergence in nonlinear SA remains less understood. Our research explores the potential for finite-time decoupled convergence in nonlinear two-time-scale SA. We find that under a weaker Lipschitz condition, traditional analyses are insufficient for achieving decoupled convergence. This finding is further numerically supported by a counterexample. But by introducing an additional condition of nested local linearity, we show that decoupled convergence is still feasible, contingent on the appropriate choice of step sizes associated with smoothnes
    
[^4]: 带有复值的深窄神经网络的普适逼近

    Universal approximation with complex-valued deep narrow neural networks. (arXiv:2305.16910v1 [math.FA])

    [http://arxiv.org/abs/2305.16910](http://arxiv.org/abs/2305.16910)

    本文研究了具有有界宽度和任意深度的复值神经网络的普适性，发现当且仅当激活函数既不是全纯的，也不是反全纯的，也不是 $\mathbb{R}$-仿射的时，深窄的复值网络具有普适逼近能力。我们还发现足够的宽度依赖于考虑的激活函数，对于一类可允许的激活函数，宽度为 $n+m+4$ 是足够的。

    

    我们研究了具有有界宽度和任意深度的复值神经网络的普适性。在温和的假设下，我们给出了那些激活函数 $\varrho:\mathbb{CC}\to \mathbb{C}$ 的完整描述，这些函数具有这样一个属性：它们关联的网络是普适的，即能够在紧致域上逼近连续函数至任意精度。准确地说，我们表明了当且仅当它们的激活函数既不是全纯的，也不是反全纯的，也不是 $\mathbb{R}$-仿射的，深窄的复值网络是普适的。这是一个比宽度任意、深度固定的对偶设置中更大的函数类。与实值情况不同的是，足够的宽度依赖于考虑的激活函数。我们表明，宽度为 $2n+2m+5$ 总是足够的，并且通常 $\max\{2n,2m\}$ 是必要的。然而，我们证明了对于一类可允许的激活函数，宽度为 $n+m+4$ 是足够的。

    We study the universality of complex-valued neural networks with bounded widths and arbitrary depths. Under mild assumptions, we give a full description of those activation functions $\varrho:\mathbb{CC}\to \mathbb{C}$ that have the property that their associated networks are universal, i.e., are capable of approximating continuous functions to arbitrary accuracy on compact domains. Precisely, we show that deep narrow complex-valued networks are universal if and only if their activation function is neither holomorphic, nor antiholomorphic, nor $\mathbb{R}$-affine. This is a much larger class of functions than in the dual setting of arbitrary width and fixed depth. Unlike in the real case, the sufficient width differs significantly depending on the considered activation function. We show that a width of $2n+2m+5$ is always sufficient and that in general a width of $\max\{2n,2m\}$ is necessary. We prove, however, that a width of $n+m+4$ suffices for a rich subclass of the admissible acti
    
[^5]: Conformal Prediction对分散标签噪声具有稳健性

    Conformal Prediction is Robust to Dispersive Label Noise. (arXiv:2209.14295v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2209.14295](http://arxiv.org/abs/2209.14295)

    本研究研究了Conformal Prediction方法对于标签噪声具有鲁棒性。我们找出了构建可以正确覆盖无噪声真实标签的不确定性集合的条件，并提出了对具有噪声标签的一般损失函数进行正确控制的要求。实验证明，在对抗性案例之外，使用Conformal Prediction和风险控制技术可以实现对干净真实标签的保守风险。我们还提出了一种有界尺寸噪声修正的方法，以确保实现正确的真实标签风险。

    

    我们研究了对标签噪声具有鲁棒性的Conformal Prediction方法，该方法是一种用于不确定性量化的强大工具。我们的分析涵盖了回归和分类问题，对于如何构建能够正确覆盖未观察到的无噪声真实标签的不确定性集合进行了界定。我们进一步扩展了我们的理论，并提出了对于带有噪声标签正确控制一般损失函数（如假阴性比例）的要求。我们的理论和实验表明，在带有噪声标签的情况下，Conformal Prediction和风险控制技术能够实现对干净真实标签的保守风险，除了在对抗性案例中。在这种情况下，我们还可以通过对Conformal Prediction算法进行有界尺寸的噪声修正，以确保实现正确的真实标签风险，而无需考虑分数或数据的规则性。

    We study the robustness of conformal prediction, a powerful tool for uncertainty quantification, to label noise. Our analysis tackles both regression and classification problems, characterizing when and how it is possible to construct uncertainty sets that correctly cover the unobserved noiseless ground truth labels. We further extend our theory and formulate the requirements for correctly controlling a general loss function, such as the false negative proportion, with noisy labels. Our theory and experiments suggest that conformal prediction and risk-controlling techniques with noisy labels attain conservative risk over the clean ground truth labels except in adversarial cases. In such cases, we can also correct for noise of bounded size in the conformal prediction algorithm in order to ensure achieving the correct risk of the ground truth labels without score or data regularity.
    

