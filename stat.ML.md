# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Universal Sharpness Dynamics in Neural Network Training: Fixed Point Analysis, Edge of Stability, and Route to Chaos.](http://arxiv.org/abs/2311.02076) | 本研究通过分析神经网络训练中的锐度动力学，揭示出早期锐度降低、逐渐增加锐化和稳定边界的机制，并发现增大学习率时，稳定边界流形上发生倍增混沌路径。 |
| [^2] | [Empirical Bayes Estimation with Side Information: A Nonparametric Integrative Tweedie Approach.](http://arxiv.org/abs/2308.05883) | 本研究提出了一种非参数综合 Tweedie 方法，通过利用侧信息的结构知识，在考虑了辅助数据的情况下进行正态均值的复合估计。理论分析和实证结果证明了该方法的优越性。 |
| [^3] | [Misspecification-robust likelihood-free inference in high dimensions.](http://arxiv.org/abs/2002.09377) | 通过基于贝叶斯优化的扩展方法和差异函数，我们实现了高维参数空间的鲁棒性强的误差分布自由推断。 |

# 详细

[^1]: 神经网络训练中的普适锐度动力学：固定点分析、稳定边界和混沌路径

    Universal Sharpness Dynamics in Neural Network Training: Fixed Point Analysis, Edge of Stability, and Route to Chaos. (arXiv:2311.02076v1 [cs.LG])

    [http://arxiv.org/abs/2311.02076](http://arxiv.org/abs/2311.02076)

    本研究通过分析神经网络训练中的锐度动力学，揭示出早期锐度降低、逐渐增加锐化和稳定边界的机制，并发现增大学习率时，稳定边界流形上发生倍增混沌路径。

    

    在神经网络的梯度下降动力学中，损失函数海森矩阵的最大特征值（锐度）在训练过程中展示出各种稳健的现象。这包括早期时间阶段，在训练的早期阶段锐度可能减小（降低锐度），以及后期行为，如逐渐增加的锐化和稳定边界。我们证明了一个简单的2层线性网络（UV模型），在单个训练样本上训练，展示了在真实场景中观察到的所有关键锐度现象。通过分析函数空间中动力学固定点的结构和函数更新的向量场，我们揭示了这些锐度趋势背后的机制。我们的分析揭示了：(i)早期锐度降低和逐渐增加锐化的机制，(ii)稳定边界所需的条件，以及 (iii)当学习率增加时，稳定边界流形上的倍增混沌路径.

    In gradient descent dynamics of neural networks, the top eigenvalue of the Hessian of the loss (sharpness) displays a variety of robust phenomena throughout training. This includes early time regimes where the sharpness may decrease during early periods of training (sharpness reduction), and later time behavior such as progressive sharpening and edge of stability. We demonstrate that a simple $2$-layer linear network (UV model) trained on a single training example exhibits all of the essential sharpness phenomenology observed in real-world scenarios. By analyzing the structure of dynamical fixed points in function space and the vector field of function updates, we uncover the underlying mechanisms behind these sharpness trends. Our analysis reveals (i) the mechanism behind early sharpness reduction and progressive sharpening, (ii) the required conditions for edge of stability, and (iii) a period-doubling route to chaos on the edge of stability manifold as learning rate is increased. Fi
    
[^2]: 使用侧信息的经验贝叶斯估计：一种非参数综合 Tweedie 方法

    Empirical Bayes Estimation with Side Information: A Nonparametric Integrative Tweedie Approach. (arXiv:2308.05883v1 [stat.ME])

    [http://arxiv.org/abs/2308.05883](http://arxiv.org/abs/2308.05883)

    本研究提出了一种非参数综合 Tweedie 方法，通过利用侧信息的结构知识，在考虑了辅助数据的情况下进行正态均值的复合估计。理论分析和实证结果证明了该方法的优越性。

    

    我们研究了在考虑到侧信息存在的情况下，正态均值的复合估计问题。利用经验贝叶斯框架，我们开发了一种非参数综合 Tweedie（NIT）方法，该方法将多变量辅助数据中编码的结构知识合并到复合估计的精度中。我们的方法使用凸优化工具直接估计对数密度的梯度，从而能够将结构约束纳入考虑。我们对 NIT 的渐近风险进行理论分析，并确定了 NIT 收敛到 Oracle 估计器的速率。随着辅助数据的维度增加，我们准确地量化了估计风险的改善以及收敛速度的恶化。通过对模拟数据和真实数据进行分析，我们展示了 NIT 的数值性能，证明了其优于现有方法。

    We investigate the problem of compound estimation of normal means while accounting for the presence of side information. Leveraging the empirical Bayes framework, we develop a nonparametric integrative Tweedie (NIT) approach that incorporates structural knowledge encoded in multivariate auxiliary data to enhance the precision of compound estimation. Our approach employs convex optimization tools to estimate the gradient of the log-density directly, enabling the incorporation of structural constraints. We conduct theoretical analyses of the asymptotic risk of NIT and establish the rate at which NIT converges to the oracle estimator. As the dimension of the auxiliary data increases, we accurately quantify the improvements in estimation risk and the associated deterioration in convergence rate. The numerical performance of NIT is illustrated through the analysis of both simulated and real data, demonstrating its superiority over existing methods.
    
[^3]: 高维情形下鲁棒性强的误差分布自由推断方法

    Misspecification-robust likelihood-free inference in high dimensions. (arXiv:2002.09377v3 [stat.CO] UPDATED)

    [http://arxiv.org/abs/2002.09377](http://arxiv.org/abs/2002.09377)

    通过基于贝叶斯优化的扩展方法和差异函数，我们实现了高维参数空间的鲁棒性强的误差分布自由推断。

    

    基于模拟器的统计模型的误差分布自由推断已经发展成为实践中有用的工具。然而，具有多个参数的模型仍然是逼近贝叶斯计算（ABC）推断的挑战。为了在高维参数空间中进行误差分布自由推断，我们引入了一种基于贝叶斯优化的扩展方法来概率化地逼近差异函数，这种方法适合于对参数空间的高效探索。我们的方法通过为每个参数使用单独的采集函数和差异函数来实现高维参数空间的计算可扩展性。有效的加性采集结构与指数损失-似然相结合，提供了一个对模型参数的误差模型说明的鲁棒性强的边际后验分布。

    Likelihood-free inference for simulator-based statistical models has developed rapidly from its infancy to a useful tool for practitioners. However, models with more than a handful of parameters still generally remain a challenge for the Approximate Bayesian Computation (ABC) based inference. To advance the possibilities for performing likelihood-free inference in higher dimensional parameter spaces, we introduce an extension of the popular Bayesian optimisation based approach to approximate discrepancy functions in a probabilistic manner which lends itself to an efficient exploration of the parameter space. Our approach achieves computational scalability for higher dimensional parameter spaces by using separate acquisition functions and discrepancies for each parameter. The efficient additive acquisition structure is combined with exponentiated loss -likelihood to provide a misspecification-robust characterisation of the marginal posterior distribution for all model parameters. The me
    

