# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Stochastic interpolants with data-dependent couplings.](http://arxiv.org/abs/2310.03725) | 本文提出了一种使用数据依赖耦合来构建生成模型的方法，并展示了在超分辨率和修复任务中的实验效果。 |
| [^2] | [The Optimal Approximation Factors in Misspecified Off-Policy Value Function Estimation.](http://arxiv.org/abs/2307.13332) | 本文研究了在线性离策略值函数估计中的逼近因子，并在多种设置下建立了最优的渐近逼近因子，这些因子决定了离策略评估的困难程度。 |
| [^3] | [Nonlinear Meta-Learning Can Guarantee Faster Rates.](http://arxiv.org/abs/2307.10870) | 非线性元学习可以保证更快的收敛速度。 |
| [^4] | [Distributed Semi-Supervised Sparse Statistical Inference.](http://arxiv.org/abs/2306.10395) | 本文提出了一种分布式半监督稀疏统计推断的高效算法，融合了有/无标签数据，为M估计和广义线性模型提供了定制去偏方法，并在模拟和真实数据应用中展示了结合无标签数据的效果。 |
| [^5] | [Optimal Estimation of Generic Dynamics by Path-Dependent Neural Jump ODEs.](http://arxiv.org/abs/2206.14284) | 本文研究了使用路径相关的神经跳跃ODE对通用动力学进行最优估计的问题，并通过实证研究支持了这些理论结果，展示了其在非马尔可夫数据和限价订单簿数据方面的优势。 |

# 详细

[^1]: 具有数据依赖耦合的随机插值。

    Stochastic interpolants with data-dependent couplings. (arXiv:2310.03725v1 [cs.LG])

    [http://arxiv.org/abs/2310.03725](http://arxiv.org/abs/2310.03725)

    本文提出了一种使用数据依赖耦合来构建生成模型的方法，并展示了在超分辨率和修复任务中的实验效果。

    

    受动态测度传输启发的生成模型（如流和扩散）构建了两个概率密度之间的连续时间映射。按照传统方法，其中一个是目标密度，只能通过样本访问，而另一个是简单的基础密度，与数据无关。在这项工作中，我们使用随机插值的框架，规范化了如何“耦合”基本密度和目标密度。这使我们能够将类别标签或连续嵌入的信息纳入到构建动态传输映射的条件生成模型中。我们展示了通过解决类似于标准独立设置的简单平方损失回归问题来学习这些传输映射。通过超分辨率和修复实验，我们证明了构建依赖耦合的有效性。

    Generative models inspired by dynamical transport of measure -- such as flows and diffusions -- construct a continuous-time map between two probability densities. Conventionally, one of these is the target density, only accessible through samples, while the other is taken as a simple base density that is data-agnostic. In this work, using the framework of stochastic interpolants, we formalize how to \textit{couple} the base and the target densities. This enables us to incorporate information about class labels or continuous embeddings to construct dynamical transport maps that serve as conditional generative models. We show that these transport maps can be learned by solving a simple square loss regression problem analogous to the standard independent setting. We demonstrate the usefulness of constructing dependent couplings in practice through experiments in super-resolution and in-painting.
    
[^2]: 在错误指定的离策略值函数估计中的最佳逼近因子

    The Optimal Approximation Factors in Misspecified Off-Policy Value Function Estimation. (arXiv:2307.13332v1 [cs.LG])

    [http://arxiv.org/abs/2307.13332](http://arxiv.org/abs/2307.13332)

    本文研究了在线性离策略值函数估计中的逼近因子，并在多种设置下建立了最优的渐近逼近因子，这些因子决定了离策略评估的困难程度。

    

    已经知道，在强化学习中的理论保证在函数逼近的错误指定中会出现乘法放大因子。然而，这些\emph{逼近因子}的性质，特别是在给定的学习问题中的最佳形式，仍然不为人所了解。在本文中，我们研究了这个问题在线性离策略值函数估计中的广泛设置中的逼近因子，其中仍有许多开放问题。我们研究了在多种设置下的逼近因子，例如加权$L_2$范数（其中加权是离线状态分布），$L_\infty$范数，状态别名的存在与否以及对状态空间的全面与部分覆盖。对于所有这些设置，我们建立了最优的渐近逼近因子（至多常数）。特别地，我们的界限确定了$L_2(\mu)$范数的两个依赖于实例的因子和$L_\infty$范数的一个因子，它们被证明决定了离策略评估的困难程度。

    Theoretical guarantees in reinforcement learning (RL) are known to suffer multiplicative blow-up factors with respect to the misspecification error of function approximation. Yet, the nature of such \emph{approximation factors} -especially their optimal form in a given learning problem -- is poorly understood. In this paper we study this question in linear off-policy value function estimation, where many open questions remain. We study the approximation factor in a broad spectrum of settings, such as with the weighted $L_2$-norm (where the weighting is the offline state distribution), the $L_\infty$ norm, the presence vs. absence of state aliasing, and full vs. partial coverage of the state space. We establish the optimal asymptotic approximation factors (up to constants) for all of these settings. In particular, our bounds identify two instance-dependent factors for the $L_2(\mu)$ norm and only one for the $L_\infty$ norm, which are shown to dictate the hardness of off-policy evalua
    
[^3]: 非线性元学习可以保证更快的收敛速度

    Nonlinear Meta-Learning Can Guarantee Faster Rates. (arXiv:2307.10870v1 [stat.ML])

    [http://arxiv.org/abs/2307.10870](http://arxiv.org/abs/2307.10870)

    非线性元学习可以保证更快的收敛速度。

    

    最近许多关于元学习的理论研究旨在利用相关任务中的相似表示结构来简化目标任务，并实现收敛速率的保证。然而，在实践中，表示往往是高度非线性的，引入了每个任务中不可简单平均的非平凡偏差。本研究通过非线性表示推导出元学习的理论保证。

    Many recent theoretical works on \emph{meta-learning} aim to achieve guarantees in leveraging similar representational structures from related tasks towards simplifying a target task. Importantly, the main aim in theory works on the subject is to understand the extent to which convergence rates -- in learning a common representation -- \emph{may scale with the number $N$ of tasks} (as well as the number of samples per task). First steps in this setting demonstrate this property when both the shared representation amongst tasks, and task-specific regression functions, are linear. This linear setting readily reveals the benefits of aggregating tasks, e.g., via averaging arguments. In practice, however, the representation is often highly nonlinear, introducing nontrivial biases in each task that cannot easily be averaged out as in the linear case. In the present work, we derive theoretical guarantees for meta-learning with nonlinear representations. In particular, assuming the shared nonl
    
[^4]: 分布式半监督稀疏统计推断

    Distributed Semi-Supervised Sparse Statistical Inference. (arXiv:2306.10395v1 [stat.ML])

    [http://arxiv.org/abs/2306.10395](http://arxiv.org/abs/2306.10395)

    本文提出了一种分布式半监督稀疏统计推断的高效算法，融合了有/无标签数据，为M估计和广义线性模型提供了定制去偏方法，并在模拟和真实数据应用中展示了结合无标签数据的效果。

    

    本文研究了分布式环境下半监督稀疏统计推断问题。我们提出了一种高效的多轮分布式去偏估计器，它融合了有标记和无标记数据，并且演示了额外的无标签数据如何帮助提高每轮迭代的统计速率。我们的方法为$M$- 估计和广义线性模型提供了量身定制的去偏方法，具体根据损失函数的特定形式而定。此外，我们的算法还可以应用于非光滑损失，例如绝对偏差损失。此外，我们的算法计算效率高，因为它只需要高维逆协方差矩阵的估计。通过模拟研究和真实数据应用，我们证明了我们的方法的有效性，并突出了结合无标签数据的好处。

    This paper is devoted to studying the semi-supervised sparse statistical inference in a distributed setup. An efficient multi-round distributed debiased estimator, which integrates both labeled and unlabelled data, is developed. We will show that the additional unlabeled data helps to improve the statistical rate of each round of iteration. Our approach offers tailored debiasing methods for $M$-estimation and generalized linear model according to the specific form of the loss function. Our method also applies to a non-smooth loss like absolute deviation loss. Furthermore, our algorithm is computationally efficient since it requires only one estimation of a high-dimensional inverse covariance matrix. We demonstrate the effectiveness of our method by presenting simulation studies and real data applications that highlight the benefits of incorporating unlabeled data.
    
[^5]: 使用路径相关的神经跳跃ODE对通用动力学进行最优估计

    Optimal Estimation of Generic Dynamics by Path-Dependent Neural Jump ODEs. (arXiv:2206.14284v4 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2206.14284](http://arxiv.org/abs/2206.14284)

    本文研究了使用路径相关的神经跳跃ODE对通用动力学进行最优估计的问题，并通过实证研究支持了这些理论结果，展示了其在非马尔可夫数据和限价订单簿数据方面的优势。

    

    本文研究了使用神经跳跃ODE（NJ-ODE）框架的路径相关扩展来预测一般随机过程的问题。虽然NJ-ODE是第一个建立起针对不规则观测时间序列预测的收敛性保证的框架，但这些结果仅适用于来自具有完整观测的It\^o扩散的数据，特别是所有坐标同时观测到的马尔可夫过程。在本研究中，我们通过利用签名变换的重构性质将这些结果推广到通用的、可能是非马尔可夫或不连续的随机过程，并通过实证研究支持了这些理论结果，在非马尔可夫数据的情况下，路径相关的NJ-ODE优于原始NJ-ODE框架。此外，我们还展示了PD-NJ-ODE可以成功应用于经典的随机滤波问题和限价订单簿（LOB）数据。

    This paper studies the problem of forecasting general stochastic processes using a path-dependent extension of the Neural Jump ODE (NJ-ODE) framework. While NJ-ODE was the first framework to establish convergence guarantees for the prediction of irregularly observed time series, these results were limited to data stemming from It\^o-diffusions with complete observations, in particular Markov processes where all coordinates are observed simultaneously. In this work, we generalise these results to generic, possibly non-Markovian or discontinuous, stochastic processes with incomplete observations, by utilising the reconstruction properties of the signature transform. These theoretical results are supported by empirical studies, where it is shown that the path-dependent NJ-ODE outperforms the original NJ-ODE framework in the case of non-Markovian data. Moreover, we show that PD-NJ-ODE can be applied successfully to classical stochastic filtering problems and to limit order book (LOB) data.
    

