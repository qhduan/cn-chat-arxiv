# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Bandit Convex Optimisation](https://arxiv.org/abs/2402.06535) | 这篇论文介绍了强盗凸优化的基本框架和用于解决这一问题的多种工具。虽然没有太多创新，但通过以新颖的方式应用现有工具，获得了新的算法和改进了一些界限。 |
| [^2] | [Learning a Gaussian Mixture for Sparsity Regularization in Inverse Problems.](http://arxiv.org/abs/2401.16612) | 本研究提出了一种基于高斯混合模型的稀疏正则化方法，通过神经网络进行贝叶斯估计，有效地解决了逆问题中的稀疏建模和参数估计问题。 |
| [^3] | [Law of Balance and Stationary Distribution of Stochastic Gradient Descent.](http://arxiv.org/abs/2308.06671) | 本文证明了随机梯度下降算法中的小批量噪音会使解决方案向平衡解靠近，只要损失函数包含重新缩放对称性。利用这个结果，我们导出了对角线性网络的随机梯度流稳态分布，该分布展示了复杂的非线性现象。这些发现揭示了动态梯度下降法在训练神经网络中的工作原理。 |
| [^4] | [Second-order Conditional Gradient Sliding.](http://arxiv.org/abs/2002.08907) | 提出了一种二阶条件梯度滑动（SOCGS）算法，可以高效解决约束二次凸优化问题，并在有限次线性收敛迭代后二次收敛于原始间隙。 |

# 详细

[^1]: Bandit Convex Optimisation（强盗凸优化）

    Bandit Convex Optimisation

    [https://arxiv.org/abs/2402.06535](https://arxiv.org/abs/2402.06535)

    这篇论文介绍了强盗凸优化的基本框架和用于解决这一问题的多种工具。虽然没有太多创新，但通过以新颖的方式应用现有工具，获得了新的算法和改进了一些界限。

    

    强盗凸优化是研究零阶凸优化的基本框架。本文介绍了用于解决该问题的许多工具，包括切平面方法、内点方法、连续指数权重、梯度下降和在线牛顿步骤。解释了许多假设和设置之间的细微差别。尽管在这里没有太多真正新的东西，但一些现有工具以新颖的方式应用于获得新算法。一些界限稍微改进了一些。

    Bandit convex optimisation is a fundamental framework for studying zeroth-order convex optimisation. These notes cover the many tools used for this problem, including cutting plane methods, interior point methods, continuous exponential weights, gradient descent and online Newton step. The nuances between the many assumptions and setups are explained. Although there is not much truly new here, some existing tools are applied in novel ways to obtain new algorithms. A few bounds are improved in minor ways.
    
[^2]: 在逆问题中学习高斯混合物进行稀疏正则化

    Learning a Gaussian Mixture for Sparsity Regularization in Inverse Problems. (arXiv:2401.16612v1 [stat.ML])

    [http://arxiv.org/abs/2401.16612](http://arxiv.org/abs/2401.16612)

    本研究提出了一种基于高斯混合模型的稀疏正则化方法，通过神经网络进行贝叶斯估计，有效地解决了逆问题中的稀疏建模和参数估计问题。

    

    在逆问题中，广泛认为引入稀疏先验对解决方案具有正则化效果。这种方法是基于一个先验假设，即未知量可以在一个有限数量的显著成分的基础上适当表示，而大多数系数接近于零。这种情况在现实世界中经常出现，比如分段平滑信号。在本研究中，我们提出了一种以高斯退化混合物形式表述的概率稀疏先验，能够对于任意基进行稀疏建模。在这个前提下，我们设计了一个可以解释为线性逆问题的贝叶斯估计器的神经网络。此外，我们提出了一种有监督和无监督的训练策略来估计这个网络的参数。为了评估我们方法的有效性，我们进行了与常用的稀疏正则化方法的数值比较。

    In inverse problems, it is widely recognized that the incorporation of a sparsity prior yields a regularization effect on the solution. This approach is grounded on the a priori assumption that the unknown can be appropriately represented in a basis with a limited number of significant components, while most coefficients are close to zero. This occurrence is frequently observed in real-world scenarios, such as with piecewise smooth signals. In this study, we propose a probabilistic sparsity prior formulated as a mixture of degenerate Gaussians, capable of modeling sparsity with respect to a generic basis. Under this premise, we design a neural network that can be interpreted as the Bayes estimator for linear inverse problems. Additionally, we put forth both a supervised and an unsupervised training strategy to estimate the parameters of this network. To evaluate the effectiveness of our approach, we conduct a numerical comparison with commonly employed sparsity-promoting regularization
    
[^3]: 动态梯度下降法的平衡法则与稳态分布

    Law of Balance and Stationary Distribution of Stochastic Gradient Descent. (arXiv:2308.06671v1 [cs.LG])

    [http://arxiv.org/abs/2308.06671](http://arxiv.org/abs/2308.06671)

    本文证明了随机梯度下降算法中的小批量噪音会使解决方案向平衡解靠近，只要损失函数包含重新缩放对称性。利用这个结果，我们导出了对角线性网络的随机梯度流稳态分布，该分布展示了复杂的非线性现象。这些发现揭示了动态梯度下降法在训练神经网络中的工作原理。

    

    随机梯度下降（SGD）算法是我们用于训练神经网络的算法。然而，我们很难理解SGD如何在神经网络的非线性和退化的损失曲面中进行导航。在这项工作中，我们证明了SGD的小批量噪音可以使解决方案向平衡解靠近，只要损失函数包含一个重新缩放对称性。由于简单扩散过程和SGD动力学的差异在对称性存在时最重要，我们的理论表明，损失函数的对称性是了解SGD工作方式的重要线索。然后，我们将这个结果应用于导出具有任意深度和宽度的对角线性网络的随机梯度流的稳态分布。稳态分布展现了复杂的非线性现象，如相变、破坏的遍历性和波动反转。这些现象仅在深层网络中存在，表明了一种基本的新的加深训练理论。

    The stochastic gradient descent (SGD) algorithm is the algorithm we use to train neural networks. However, it remains poorly understood how the SGD navigates the highly nonlinear and degenerate loss landscape of a neural network. In this work, we prove that the minibatch noise of SGD regularizes the solution towards a balanced solution whenever the loss function contains a rescaling symmetry. Because the difference between a simple diffusion process and SGD dynamics is the most significant when symmetries are present, our theory implies that the loss function symmetries constitute an essential probe of how SGD works. We then apply this result to derive the stationary distribution of stochastic gradient flow for a diagonal linear network with arbitrary depth and width. The stationary distribution exhibits complicated nonlinear phenomena such as phase transitions, broken ergodicity, and fluctuation inversion. These phenomena are shown to exist uniquely in deep networks, implying a fundam
    
[^4]: 二阶条件梯度滑动

    Second-order Conditional Gradient Sliding. (arXiv:2002.08907v3 [math.OC] UPDATED)

    [http://arxiv.org/abs/2002.08907](http://arxiv.org/abs/2002.08907)

    提出了一种二阶条件梯度滑动（SOCGS）算法，可以高效解决约束二次凸优化问题，并在有限次线性收敛迭代后二次收敛于原始间隙。

    

    当需要高精度解决问题时，约束二阶凸优化算法是首选，因为它们具有局部二次收敛性。这些算法在每次迭代时需要解决一个约束二次子问题。我们提出了\emph{二阶条件梯度滑动}（SOCGS）算法，它使用一种无投影算法来近似解决约束二次子问题。当可行域是一个多面体时，该算法在有限次线性收敛迭代后二次收敛于原始间隙。进入二次收敛阶段后，SOCGS算法需通过$\mathcal{O}(\log(\log 1/\varepsilon))$次一阶和Hessian正交调用以及$\mathcal{O}(\log (1/\varepsilon) \log(\log1/\varepsilon))$次线性最小化正交调用来实现$\varepsilon$-最优解。当可行域只能通过线性优化正交调用高效访问时，此算法非常有用。

    Constrained second-order convex optimization algorithms are the method of choice when a high accuracy solution to a problem is needed, due to their local quadratic convergence. These algorithms require the solution of a constrained quadratic subproblem at every iteration. We present the \emph{Second-Order Conditional Gradient Sliding} (SOCGS) algorithm, which uses a projection-free algorithm to solve the constrained quadratic subproblems inexactly. When the feasible region is a polytope the algorithm converges quadratically in primal gap after a finite number of linearly convergent iterations. Once in the quadratic regime the SOCGS algorithm requires $\mathcal{O}(\log(\log 1/\varepsilon))$ first-order and Hessian oracle calls and $\mathcal{O}(\log (1/\varepsilon) \log(\log1/\varepsilon))$ linear minimization oracle calls to achieve an $\varepsilon$-optimal solution. This algorithm is useful when the feasible region can only be accessed efficiently through a linear optimization oracle, 
    

