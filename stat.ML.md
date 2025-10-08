# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Model-free generalized fiducial inference.](http://arxiv.org/abs/2307.12472) | 本文提出了一种无模型的统计框架，用于不准确概率预测推理的不确定性量化，并考虑了精确概率近似模型无关的不准确推理框架的特性。 |
| [^2] | [An Oblivious Stochastic Composite Optimization Algorithm for Eigenvalue Optimization Problems.](http://arxiv.org/abs/2306.17470) | 本论文提出了两种针对非光滑和光滑目标的无视觉随机镜像下降算法，不需要先验知识，并给出了相应收敛速度。 |
| [^3] | [An inexact linearized proximal algorithm for a class of DC composite optimization problems and applications.](http://arxiv.org/abs/2303.16822) | 本文提出了一种解决非凸非光滑问题的非精确线性化近端算法，并应用于鲁棒分解中的两个问题，得到了有效的数值结果。 |

# 详细

[^1]: 无模型广义基准推理

    Model-free generalized fiducial inference. (arXiv:2307.12472v1 [stat.ML])

    [http://arxiv.org/abs/2307.12472](http://arxiv.org/abs/2307.12472)

    本文提出了一种无模型的统计框架，用于不准确概率预测推理的不确定性量化，并考虑了精确概率近似模型无关的不准确推理框架的特性。

    

    鉴于机器学习中不确定性量化方法的安全可靠性的需求，本文提出并发展了一种无模型的统计框架，用于不准确概率预测推理的不确定性量化。该框架通过提供预测集的形式，实现了对第一类错误的有限样本控制，这与一致性预测集具有相同的属性，但这种新方法还提供了更灵活的不准确概率推理工具。此外，本文提出并考虑了一种精确概率近似模型无关的不准确推理框架的理论和实证特性。通过将信念/可信度度量对近似为在可信区间中的[在某种意义上最优]概率度量，是扩大在统计和机器学习社区推广不准确概率推理方法所需的关键解决方案，目前在统计和

    Motivated by the need for the development of safe and reliable methods for uncertainty quantification in machine learning, I propose and develop ideas for a model-free statistical framework for imprecise probabilistic prediction inference. This framework facilitates uncertainty quantification in the form of prediction sets that offer finite sample control of type 1 errors, a property shared with conformal prediction sets, but this new approach also offers more versatile tools for imprecise probabilistic reasoning. Furthermore, I propose and consider the theoretical and empirical properties of a precise probabilistic approximation to the model-free imprecise framework. Approximating a belief/plausibility measure pair by an [optimal in some sense] probability measure in the credal set is a critical resolution needed for the broader adoption of imprecise probabilistic approaches to inference in statistical and machine learning communities. It is largely undetermined in the statistical and
    
[^2]: 一种用于特征值优化问题的无视觉随机复合优化算法

    An Oblivious Stochastic Composite Optimization Algorithm for Eigenvalue Optimization Problems. (arXiv:2306.17470v1 [math.OC])

    [http://arxiv.org/abs/2306.17470](http://arxiv.org/abs/2306.17470)

    本论文提出了两种针对非光滑和光滑目标的无视觉随机镜像下降算法，不需要先验知识，并给出了相应收敛速度。

    

    在这项工作中，我们重新审视了使用随机化一阶方法和随机平滑解决大规模半定规划问题的问题。我们引入了两种基于互补复合设置的无视觉随机镜像下降算法。一种算法设计用于非光滑目标，而加速版本则适用于光滑目标。值得注意的是，这两种算法都不需要对目标函数的Lipschitz常数或光滑度有先验知识。对于具有$\mathcal{M}-$有界预言的非光滑情况，我们证明了一个收敛速度为$ O( {\mathcal{M}}/{\sqrt{T}} ) $的收敛速度。对于具有由$D$限制的可行集的$L$-光滑情况，我们得到了一个收敛速度为$ O( {L^2 D^2}/{(T^{2}\sqrt{T})} + {(D_0^2+\sigma^2)}/{\sqrt{T}} )$的收敛速度，其中$D_0$是到最优解的起始距离，$ \sigma^2$是随机预言方差。目前只有在假设先验知识的Lipschitz常数或t情况下才能得到这些速度。

    In this work, we revisit the problem of solving large-scale semidefinite programs using randomized first-order methods and stochastic smoothing. We introduce two oblivious stochastic mirror descent algorithms based on a complementary composite setting. One algorithm is designed for non-smooth objectives, while an accelerated version is tailored for smooth objectives. Remarkably, both algorithms work without prior knowledge of the Lipschitz constant or smoothness of the objective function. For the non-smooth case with $\mathcal{M}-$bounded oracles, we prove a convergence rate of $ O( {\mathcal{M}}/{\sqrt{T}} ) $. For the $L$-smooth case with a feasible set bounded by $D$, we derive a convergence rate of $ O( {L^2 D^2}/{(T^{2}\sqrt{T})} + {(D_0^2+\sigma^2)}/{\sqrt{T}} )$, where $D_0$ is the starting distance to an optimal solution, and $ \sigma^2$ is the stochastic oracle variance. These rates had only been obtained so far by either assuming prior knowledge of the Lipschitz constant or t
    
[^3]: 一类DC复合优化问题的非精确线性近似近端算法及应用

    An inexact linearized proximal algorithm for a class of DC composite optimization problems and applications. (arXiv:2303.16822v1 [math.OC])

    [http://arxiv.org/abs/2303.16822](http://arxiv.org/abs/2303.16822)

    本文提出了一种解决非凸非光滑问题的非精确线性化近端算法，并应用于鲁棒分解中的两个问题，得到了有效的数值结果。

    

    本文研究了一类DC复合优化问题。这类问题通常由低秩矩阵恢复的鲁棒分解模型推导而来，是凸复合优化问题和具有非光滑分量的DC规划的扩展。针对这类非凸和非光滑问题，我们提出了一种非精确线性化近端算法（iLPA）。算法中，我们利用目标函数的部分线性化，计算强凸主导的非精确最小化值。迭代序列的生成收敛于潜在函数的Kurdyka-{\L}ojasiewicz（KL）性质，如果潜在函数在极限点处具有KL指数$1/2$的KL性质，则收敛具有局部R线性速率。对于后一种假设，我们利用复合结构提供了一个可验证的条件，并阐明了与凸复合优化所使用的正则性的关系。最后，我们将所提出的非精确线性近端算法应用于解决鲁棒分解中的两个重要问题：张量鲁棒主成分分析（TRPCA）和张量鲁棒低秩张量完成（TRLRTC）。对合成和真实数据的数值结果证明了我们的算法相对于现有最新算法的有效性。

    This paper is concerned with a class of DC composite optimization problems which, as an extension of the convex composite optimization problem and the DC program with nonsmooth components, often arises from robust factorization models of low-rank matrix recovery. For this class of nonconvex and nonsmooth problems, we propose an inexact linearized proximal algorithm (iLPA) which in each step computes an inexact minimizer of a strongly convex majorization constructed by the partial linearization of their objective functions. The generated iterate sequence is shown to be convergent under the Kurdyka-{\L}ojasiewicz (KL) property of a potential function, and the convergence admits a local R-linear rate if the potential function has the KL property of exponent $1/2$ at the limit point. For the latter assumption, we provide a verifiable condition by leveraging the composite structure, and clarify its relation with the regularity used for the convex composite optimization. Finally, the propose
    

