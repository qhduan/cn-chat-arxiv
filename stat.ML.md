# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A tutorial on learning from preferences and choices with Gaussian Processes](https://arxiv.org/abs/2403.11782) | 提供了一个使用高斯过程进行偏好学习的框架，能够将理性原则融入学习过程，涵盖了多种偏好学习模型。 |
| [^2] | [Causal Representation Learning from Multiple Distributions: A General Setting](https://arxiv.org/abs/2402.05052) | 本文研究了一个通用的、完全非参数的因果表示学习设置，旨在在多个分布之间学习因果关系，无需假设硬干预。通过稀疏性约束，可以从多个分布中恢复出因果关系。 |
| [^3] | [A General Theory for Kernel Packets: from state space model to compactly supported basis](https://arxiv.org/abs/2402.04022) | 该论文提出了一种从状态空间模型到紧支持基的核分组的通用理论，该理论可以用于降低高斯过程的训练和预测时间，并且通过适当的线性组合产生了$m$个紧支持的核分组函数。 |
| [^4] | [Global $\mathcal{L}^2$ minimization at uniform exponential rate via geometrically adapted gradient descent in Deep Learning](https://arxiv.org/abs/2311.15487) | 通过几何调整的梯度下降，在深度学习中以均匀指数速率实现全局$\mathcal{L}^2$最小化，这一方法在过参数化情况下具有明确自然的不变几何含义。 |
| [^5] | [A Survey on Uncertainty Quantification for Deep Learning: An Uncertainty Source Perspective](https://arxiv.org/abs/2302.13425) | 本研究对深度学习的不确定性量化进行了调查，从不确定性来源的角度分析不同方法，以评估DNN预测的置信度。 |
| [^6] | [Learning Sparse Codes with Entropy-Based ELBOs.](http://arxiv.org/abs/2311.01888) | 本论文提出了一种基于熵的学习目标，用于稀疏编码参数的学习，通过非平凡的后验逼近和解析的目标函数，实现了标准稀疏编码的学习，在数值实验中证明了其可行性。 |
| [^7] | [Stabilizing Estimates of Shapley Values with Control Variates.](http://arxiv.org/abs/2310.07672) | 使用控制变量的方法稳定Shapley值的估计，减少了模型解释的不确定性，适用于任何机器学习模型。 |
| [^8] | [Nearest neighbor process: weak convergence and non-asymptotic bound.](http://arxiv.org/abs/2110.15083) | 本文介绍了一种中心统计量——最近邻测度，并通过均匀中心极限定理和一种均匀的非渐近界限研究了它。该测度可能为推断提供了一种替代方法。 |

# 详细

[^1]: 使用高斯过程从偏好和选择中学习的教程

    A tutorial on learning from preferences and choices with Gaussian Processes

    [https://arxiv.org/abs/2403.11782](https://arxiv.org/abs/2403.11782)

    提供了一个使用高斯过程进行偏好学习的框架，能够将理性原则融入学习过程，涵盖了多种偏好学习模型。

    

    偏好建模位于经济学、决策理论、机器学习和统计学的交叉点。通过理解个体的偏好及其选择方式，我们可以构建更接近他们期望的产品，为跨领域的更高效、个性化应用铺平道路。此教程的目标是提供一个连贯、全面的偏好学习框架，使用高斯过程演示如何将理性原则（来自经济学和决策理论）无缝地纳入学习过程中。通过合适地定制似然函数，这一框架使得能够构建涵盖随机效用模型、辨识限制和对象和标签偏好的多重冲突效用情景的偏好学习模型。

    arXiv:2403.11782v1 Announce Type: new  Abstract: Preference modelling lies at the intersection of economics, decision theory, machine learning and statistics. By understanding individuals' preferences and how they make choices, we can build products that closely match their expectations, paving the way for more efficient and personalised applications across a wide range of domains. The objective of this tutorial is to present a cohesive and comprehensive framework for preference learning with Gaussian Processes (GPs), demonstrating how to seamlessly incorporate rationality principles (from economics and decision theory) into the learning process. By suitably tailoring the likelihood function, this framework enables the construction of preference learning models that encompass random utility models, limits of discernment, and scenarios with multiple conflicting utilities for both object- and label-preference. This tutorial builds upon established research while simultaneously introducin
    
[^2]: 从多个分布中进行因果表示学习：一个通用设置

    Causal Representation Learning from Multiple Distributions: A General Setting

    [https://arxiv.org/abs/2402.05052](https://arxiv.org/abs/2402.05052)

    本文研究了一个通用的、完全非参数的因果表示学习设置，旨在在多个分布之间学习因果关系，无需假设硬干预。通过稀疏性约束，可以从多个分布中恢复出因果关系。

    

    在许多问题中，测量变量（例如图像像素）只是隐藏的因果变量（例如潜在的概念或对象）的数学函数。为了在不断变化的环境中进行预测或对系统进行适当的更改，恢复隐藏的因果变量$Z_i$以及由图$\mathcal{G}_Z$表示的它们的因果关系是有帮助的。这个问题最近被称为因果表示学习。本文关注来自多个分布（来自异构数据或非平稳时间序列）的因果表示学习的通用、完全非参数的设置，不需要假设分布改变背后存在硬干预。我们旨在在这个基本情况下开发通用解决方案；作为副产品，这有助于看到其他假设（如参数因果模型或硬干预）提供的独特好处。我们证明在恢复过程中对图的稀疏性约束下，可以从多个分布中学习出因果关系。

    In many problems, the measured variables (e.g., image pixels) are just mathematical functions of the hidden causal variables (e.g., the underlying concepts or objects). For the purpose of making predictions in changing environments or making proper changes to the system, it is helpful to recover the hidden causal variables $Z_i$ and their causal relations represented by graph $\mathcal{G}_Z$. This problem has recently been known as causal representation learning. This paper is concerned with a general, completely nonparametric setting of causal representation learning from multiple distributions (arising from heterogeneous data or nonstationary time series), without assuming hard interventions behind distribution changes. We aim to develop general solutions in this fundamental case; as a by product, this helps see the unique benefit offered by other assumptions such as parametric causal models or hard interventions. We show that under the sparsity constraint on the recovered graph over
    
[^3]: 一种从状态空间模型到紧支持基的核分组的通用理论

    A General Theory for Kernel Packets: from state space model to compactly supported basis

    [https://arxiv.org/abs/2402.04022](https://arxiv.org/abs/2402.04022)

    该论文提出了一种从状态空间模型到紧支持基的核分组的通用理论，该理论可以用于降低高斯过程的训练和预测时间，并且通过适当的线性组合产生了$m$个紧支持的核分组函数。

    

    众所周知，高斯过程（GP）的状态空间（SS）模型公式可以将其训练和预测时间降低到O（n）（n为数据点个数）。我们证明了一个m维的GP的SS模型公式等价于我们引入的一个概念，称为通用右核分组（KP）：一种用于GP协方差函数K的变换，使得对于任意$t \leq t_1$，$0 \leq j \leq m-1$和$m+1$个连续点$t_i$，都满足$\sum_{i=0}^{m}a_iD_t^{(j)}K(t,t_i)=0$，其中${D}_t^{(j)}f(t)$表示在$t$上作用的第j阶导数。我们将这个思想扩展到了GP的向后SS模型公式，得到了下一个$m$个连续点的左核分组的概念：$\sum_{i=0}^{m}b_i{D}_t^{(j)}K(t,t_{m+i})=0$，对于任意$t\geq t_{2m}$。通过结合左右核分组，可以证明这些协方差函数的适当线性组合产生了$m$个紧支持的核分组函数：对于任意$t\not\in(t_0,t_{2m})$和$j=0,\cdots,m-1$，$\phi^{(j)}(t)=0$。

    It is well known that the state space (SS) model formulation of a Gaussian process (GP) can lower its training and prediction time both to O(n) for n data points. We prove that an $m$-dimensional SS model formulation of GP is equivalent to a concept we introduce as the general right Kernel Packet (KP): a transformation for the GP covariance function $K$ such that $\sum_{i=0}^{m}a_iD_t^{(j)}K(t,t_i)=0$ holds for any $t \leq t_1$, 0 $\leq j \leq m-1$, and $m+1$ consecutive points $t_i$, where ${D}_t^{(j)}f(t) $ denotes $j$-th order derivative acting on $t$. We extend this idea to the backward SS model formulation of the GP, leading to the concept of the left KP for next $m$ consecutive points: $\sum_{i=0}^{m}b_i{D}_t^{(j)}K(t,t_{m+i})=0$ for any $t\geq t_{2m}$. By combining both left and right KPs, we can prove that a suitable linear combination of these covariance functions yields $m$ compactly supported KP functions: $\phi^{(j)}(t)=0$ for any $t\not\in(t_0,t_{2m})$ and $j=0,\cdots,m-1$
    
[^4]: 深度学习中通过几何调整的梯度下降以均匀指数速率全局$\mathcal{L}^2$最小化

    Global $\mathcal{L}^2$ minimization at uniform exponential rate via geometrically adapted gradient descent in Deep Learning

    [https://arxiv.org/abs/2311.15487](https://arxiv.org/abs/2311.15487)

    通过几何调整的梯度下降，在深度学习中以均匀指数速率实现全局$\mathcal{L}^2$最小化，这一方法在过参数化情况下具有明确自然的不变几何含义。

    

    我们考虑在深度学习网络中广泛使用的用于最小化$\mathcal{L}^2$代价函数的梯度下降流，并引入两个改进版本；一个适用于过参数化设置，另一个适用于欠参数化设置。这两个版本都具有明确自然的不变几何含义，考虑到在过参数化设置中的拉回向量丛结构和在欠参数化设置中的推前向量丛结构。在过参数化情况下，我们证明，只要满足秩条件，改进的梯度下降的所有轨道将以均匀指数收敛速率将$\mathcal{L}^2$代价驱动到全局最小值；因此，对于任何预先指定的接近全局最小值的近似，我们可以得到先验停止时间。我们指出后者与次Riemann几何的关系。

    arXiv:2311.15487v3 Announce Type: replace-cross  Abstract: We consider the gradient descent flow widely used for the minimization of the $\mathcal{L}^2$ cost function in Deep Learning networks, and introduce two modified versions; one adapted for the overparametrized setting, and the other for the underparametrized setting. Both have a clear and natural invariant geometric meaning, taking into account the pullback vector bundle structure in the overparametrized, and the pushforward vector bundle structure in the underparametrized setting. In the overparametrized case, we prove that, provided that a rank condition holds, all orbits of the modified gradient descent drive the $\mathcal{L}^2$ cost to its global minimum at a uniform exponential convergence rate; one thereby obtains an a priori stopping time for any prescribed proximity to the global minimum. We point out relations of the latter to sub-Riemannian geometry.
    
[^5]: 对深度学习的不确定性量化进行调查：从不确定性来源的角度分析

    A Survey on Uncertainty Quantification for Deep Learning: An Uncertainty Source Perspective

    [https://arxiv.org/abs/2302.13425](https://arxiv.org/abs/2302.13425)

    本研究对深度学习的不确定性量化进行了调查，从不确定性来源的角度分析不同方法，以评估DNN预测的置信度。

    

    深度神经网络(DNNs)在计算机视觉、自然语言处理以及科学与工程领域取得了巨大成功。然而，人们也认识到DNNs有时会做出意外、错误但过于自信的预测。这可能导致在自动驾驶、医学诊断和灾难响应等高风险应用中出现严重后果。不确定性量化（UQ）旨在估计DNN预测的置信度，超越预测准确性。近年来，已经开发了许多针对DNNs的UQ方法。系统地对这些UQ方法进行分类并比较它们的优势和劣势具有极大的实际价值。然而，现有调查大多集中在从神经网络架构角度或贝叶斯角度对UQ方法进行分类，忽略了每种方法可能引入的不确定性来源。

    arXiv:2302.13425v3 Announce Type: replace  Abstract: Deep neural networks (DNNs) have achieved tremendous success in making accurate predictions for computer vision, natural language processing, as well as science and engineering domains. However, it is also well-recognized that DNNs sometimes make unexpected, incorrect, but overconfident predictions. This can cause serious consequences in high-stake applications, such as autonomous driving, medical diagnosis, and disaster response. Uncertainty quantification (UQ) aims to estimate the confidence of DNN predictions beyond prediction accuracy. In recent years, many UQ methods have been developed for DNNs. It is of great practical value to systematically categorize these UQ methods and compare their advantages and disadvantages. However, existing surveys mostly focus on categorizing UQ methodologies from a neural network architecture perspective or a Bayesian perspective and ignore the source of uncertainty that each methodology can incor
    
[^6]: 使用基于熵的ELBO学习稀疏编码

    Learning Sparse Codes with Entropy-Based ELBOs. (arXiv:2311.01888v1 [stat.ML])

    [http://arxiv.org/abs/2311.01888](http://arxiv.org/abs/2311.01888)

    本论文提出了一种基于熵的学习目标，用于稀疏编码参数的学习，通过非平凡的后验逼近和解析的目标函数，实现了标准稀疏编码的学习，在数值实验中证明了其可行性。

    

    标准概率稀疏编码假设拉普拉斯先验、从潜在到可观测的线性映射以及高斯可观测分布。我们在这里导出了一个仅基于熵的学习目标，用于标准稀疏编码的参数。这个新的变分目标具有以下特点：（A）与MAP逼近不同，它使用了概率推理的非平凡后验逼近；（B）与以前的非平凡逼近不同，这个新的目标是完全解析的；（C）该目标允许一种新的原则性的退火形式。目标的导出首先通过证明标准ELBO目标收敛到熵的和，这与具有高斯先验的生成模型的最近类似结果相匹配。然后，我们证明了ELBO等于熵的条件具有解析解，从而得到了完全解析的目标。通过数值实验证明了学习逼真性的可行性。

    Standard probabilistic sparse coding assumes a Laplace prior, a linear mapping from latents to observables, and Gaussian observable distributions. We here derive a solely entropy-based learning objective for the parameters of standard sparse coding. The novel variational objective has the following features: (A) unlike MAP approximations, it uses non-trivial posterior approximations for probabilistic inference; (B) unlike for previous non-trivial approximations, the novel objective is fully analytical; and (C) the objective allows for a novel principled form of annealing. The objective is derived by first showing that the standard ELBO objective converges to a sum of entropies, which matches similar recent results for generative models with Gaussian priors. The conditions under which the ELBO becomes equal to entropies are then shown to have analytical solutions, which leads to the fully analytical objective. Numerical experiments are used to demonstrate the feasibility of learning wit
    
[^7]: 用控制变量稳定Shapley值的估计

    Stabilizing Estimates of Shapley Values with Control Variates. (arXiv:2310.07672v1 [stat.ML])

    [http://arxiv.org/abs/2310.07672](http://arxiv.org/abs/2310.07672)

    使用控制变量的方法稳定Shapley值的估计，减少了模型解释的不确定性，适用于任何机器学习模型。

    

    Shapley值是解释黑盒机器学习模型预测最流行的工具之一。然而，它们的计算成本很高，因此采用抽样近似来减少不确定性。为了稳定这些模型解释，我们提出了一种基于控制变量的蒙特卡洛技术的方法，称为ControlSHAP。我们的方法适用于任何机器学习模型，并且几乎不需要额外的计算或建模工作。在多个高维数据集上，我们发现它可以显著减少Shapley估计的蒙特卡洛变异性。

    Shapley values are among the most popular tools for explaining predictions of blackbox machine learning models. However, their high computational cost motivates the use of sampling approximations, inducing a considerable degree of uncertainty. To stabilize these model explanations, we propose ControlSHAP, an approach based on the Monte Carlo technique of control variates. Our methodology is applicable to any machine learning model and requires virtually no extra computation or modeling effort. On several high-dimensional datasets, we find it can produce dramatic reductions in the Monte Carlo variability of Shapley estimates.
    
[^8]: 最近邻过程：弱收敛和非渐近界限

    Nearest neighbor process: weak convergence and non-asymptotic bound. (arXiv:2110.15083v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2110.15083](http://arxiv.org/abs/2110.15083)

    本文介绍了一种中心统计量——最近邻测度，并通过均匀中心极限定理和一种均匀的非渐近界限研究了它。该测度可能为推断提供了一种替代方法。

    

    介绍并研究了由给定点的最近邻所得到的经验测度——最近邻测度作为一种中心统计量。首先，在底层函数类上满足（反映最近邻算法的本地化特性的）（本地）支撑熵条件下，将相关经验过程证明为满足均匀中心极限定理。其次，在统一熵数的著名条件（通常称为Vapnik-Chervonenkis）下建立了一种均匀的非渐近界限。在均匀中心极限定理中所获得的高斯极限的协方差等于条件协方差算子（给出兴趣点）。这提示了一种可能性，即在使用相同的推理方式但仅使用最近邻而不是全部替换标准经验测度的标准方法的情况下，扩展标准方法 - 非局部。

    The empirical measure resulting from the nearest neighbors to a given point \textit{the nearest neighbor measure} - is introduced and studied as a central statistical quantity. First, the associated empirical process is shown to satisfy a uniform central limit theorem under a (local) bracketing entropy condition on the underlying class of functions (reflecting the localizing nature of the nearest neighbor algorithm). Second a uniform non-asymptotic bound is established under a well-known condition, often referred to as Vapnik-Chervonenkis, on the uniform entropy numbers. The covariance of the Gaussian limit obtained in the uniform central limit theorem is equal to the conditional covariance operator (given the point of interest). This suggests the possibility of extending standard approaches - non local - replacing simply the standard empirical measure by the nearest neighbor measure while using the same way of making inference but with the nearest neighbors only instead of the full 
    

