# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Floralens: a Deep Learning Model for the Portuguese Native Flora](https://arxiv.org/abs/2403.12072) | 本论文开发了一种用于从公开数据集构建生物分类群数据集以及利用深度卷积神经网络推导模型的简化方法，并以葡萄牙本地植物为案例研究。 |
| [^2] | [Federated Transfer Learning with Differential Privacy](https://arxiv.org/abs/2403.11343) | 本文提出了具有差分隐私的联邦迁移学习框架，通过利用多个异构源数据集的信息来增强对目标数据集的学习，同时考虑隐私约束。 |
| [^3] | [A Multi-Agent Reinforcement Learning Framework for Evaluating the U.S. Ending the HIV Epidemic Plan.](http://arxiv.org/abs/2311.00855) | 本论文提出了一种多智能体强化学习（MARL）框架，用于评估美国终结HIV流行计划。该框架能够进行特定地区的决策分析，并考虑到地区之间的流行病学相互作用。 |
| [^4] | [Sparse Gaussian Graphical Models with Discrete Optimization: Computational and Statistical Perspectives.](http://arxiv.org/abs/2307.09366) | 本文提出了基于离散优化的稀疏高斯图模型学习问题的新方法，并提供了大规模求解器来获取良好的原始解。 |
| [^5] | [Accelerated gradient methods for nonconvex optimization: Escape trajectories from strict saddle points and convergence to local minima.](http://arxiv.org/abs/2307.07030) | 本文研究了一类加速梯度方法在非凸优化问题上的行为，包括逃离鞍点和收敛到局部极小值点的分析。研究在渐进和非渐进情况下，提出了一类新的Nesterov类型的加速方法，并回答了Nesterov加速梯度方法是否避免了严格鞍点的问题。 |
| [^6] | [Importance Sparsification for Sinkhorn Algorithm.](http://arxiv.org/abs/2306.06581) | Spar-Sink是一种重要性稀疏化方法，能够有效近似熵正则化最优传输和不平衡最优传输问题，并且在实验中表现优异。 |
| [^7] | [Neural Exploitation and Exploration of Contextual Bandits.](http://arxiv.org/abs/2305.03784) | 本文提出了一种新型的神经网络策略"EE-Net"，它用于多臂赌博机的利用和探索，在学习奖励函数的同时也适应性地学习潜在收益。 |
| [^8] | [Piecewise Deterministic Markov Processes for Bayesian Neural Networks.](http://arxiv.org/abs/2302.08724) | 本文介绍了基于分段确定性马尔可夫过程的贝叶斯神经网络推理方法，通过引入新的自适应稀疏方案，实现了对困难采样问题的加速处理。实验证明，这种方法在计算上可行，并能提高预测准确性、MCMC混合性能，并提供更有信息量的不确定性测量。 |

# 详细

[^1]: Floralens：一种用于葡萄牙本地植物的深度学习模型

    Floralens: a Deep Learning Model for the Portuguese Native Flora

    [https://arxiv.org/abs/2403.12072](https://arxiv.org/abs/2403.12072)

    本论文开发了一种用于从公开数据集构建生物分类群数据集以及利用深度卷积神经网络推导模型的简化方法，并以葡萄牙本地植物为案例研究。

    

    机器学习技术，特别是深度卷积神经网络，在许多公民科学平台中对生物物种进行基于图像的识别是至关重要的。然而，构建足够大小和样本的数据集来训练网络以及网络架构的选择本身仍然很少有文献记录，因此不容易被复制。在本文中，我们开发了一种简化的方法，用于从公开可用的研究级数据集构建生物分类群的数据集，并利用这些数据集使用谷歌的AutoML Vision云服务提供的现成深度卷积神经网络来推导模型。我们的案例研究是葡萄牙本地植物，基于由葡萄牙植物学会提供的高质量数据集，并通过添加来自iNaturalist、Pl@ntNet和Observation.org的采集数据进行扩展。我们发现通过谨慎地

    arXiv:2403.12072v1 Announce Type: cross  Abstract: Machine-learning techniques, namely deep convolutional neural networks, are pivotal for image-based identification of biological species in many Citizen Science platforms. However, the construction of critically sized and sampled datasets to train the networks and the choice of the network architectures itself remains little documented and, therefore, does not lend itself to be easily replicated. In this paper, we develop a streamlined methodology for building datasets for biological taxa from publicly available research-grade datasets and for deriving models from these datasets using off-the-shelf deep convolutional neural networks such as those provided by Google's AutoML Vision cloud service. Our case study is the Portuguese native flora, anchored in a high-quality dataset, provided by the Sociedade Portuguesa de Bot\^anica, scaled up by adding sampled data from iNaturalist, Pl@ntNet, and Observation.org. We find that with a careful
    
[^2]: 具有差分隐私的联邦迁移学习

    Federated Transfer Learning with Differential Privacy

    [https://arxiv.org/abs/2403.11343](https://arxiv.org/abs/2403.11343)

    本文提出了具有差分隐私的联邦迁移学习框架，通过利用多个异构源数据集的信息来增强对目标数据集的学习，同时考虑隐私约束。

    

    联邦学习越来越受到欢迎，数据异构性和隐私性是两个突出的挑战。在本文中，我们在联邦迁移学习框架内解决了这两个问题，旨在通过利用来自多个异构源数据集的信息来增强对目标数据集的学习，同时遵守隐私约束。我们严格制定了\textit{联邦差分隐私}的概念，为每个数据集提供隐私保证，而无需假设有一个受信任的中央服务器。在这个隐私约束下，我们研究了三个经典的统计问题，即单变量均值估计、低维线性回归和高维线性回归。通过研究极小值率并确定这些问题的隐私成本，我们展示了联邦差分隐私是已建立的局部和中央模型之间的一种中间隐私模型。

    arXiv:2403.11343v1 Announce Type: new  Abstract: Federated learning is gaining increasing popularity, with data heterogeneity and privacy being two prominent challenges. In this paper, we address both issues within a federated transfer learning framework, aiming to enhance learning on a target data set by leveraging information from multiple heterogeneous source data sets while adhering to privacy constraints. We rigorously formulate the notion of \textit{federated differential privacy}, which offers privacy guarantees for each data set without assuming a trusted central server. Under this privacy constraint, we study three classical statistical problems, namely univariate mean estimation, low-dimensional linear regression, and high-dimensional linear regression. By investigating the minimax rates and identifying the costs of privacy for these problems, we show that federated differential privacy is an intermediate privacy model between the well-established local and central models of 
    
[^3]: 一种用于评估美国终结HIV流行计划的多智能体强化学习框架

    A Multi-Agent Reinforcement Learning Framework for Evaluating the U.S. Ending the HIV Epidemic Plan. (arXiv:2311.00855v1 [cs.AI])

    [http://arxiv.org/abs/2311.00855](http://arxiv.org/abs/2311.00855)

    本论文提出了一种多智能体强化学习（MARL）框架，用于评估美国终结HIV流行计划。该框架能够进行特定地区的决策分析，并考虑到地区之间的流行病学相互作用。

    

    人类免疫缺陷病毒（HIV）是美国的主要公共卫生问题，每年有约1.2万人感染HIV，其中有3.5万人是新感染者。美国的HIV负担和护理接触存在着地理差异。2019年的终结HIV流行计划旨在到2030年将新感染人数减少90%，通过提高诊断、治疗和预防干预措施的覆盖率，并优先考虑HIV高流行地区。确定最佳干预措施的规模扩大将有助于资源分配的决策。现有的HIV决策模型要么只评估特定城市，要么评估整个国家人口，忽视地方的相互作用或差异。在本文中，我们提出了一种多智能体强化学习（MARL）模型，它能够进行特定地区的决策分析，同时考虑跨地区的流行病互动。在实验分析中，

    Human immunodeficiency virus (HIV) is a major public health concern in the United States, with about 1.2 million people living with HIV and 35,000 newly infected each year. There are considerable geographical disparities in HIV burden and care access across the U.S. The 2019 Ending the HIV Epidemic (EHE) initiative aims to reduce new infections by 90% by 2030, by improving coverage of diagnoses, treatment, and prevention interventions and prioritizing jurisdictions with high HIV prevalence. Identifying optimal scale-up of intervention combinations will help inform resource allocation. Existing HIV decision analytic models either evaluate specific cities or the overall national population, thus overlooking jurisdictional interactions or differences. In this paper, we propose a multi-agent reinforcement learning (MARL) model, that enables jurisdiction-specific decision analyses but in an environment with cross-jurisdictional epidemiological interactions. In experimental analyses, conduct
    
[^4]: 稀疏高斯图模型的离散优化：计算和统计角度

    Sparse Gaussian Graphical Models with Discrete Optimization: Computational and Statistical Perspectives. (arXiv:2307.09366v1 [cs.LG])

    [http://arxiv.org/abs/2307.09366](http://arxiv.org/abs/2307.09366)

    本文提出了基于离散优化的稀疏高斯图模型学习问题的新方法，并提供了大规模求解器来获取良好的原始解。

    

    我们考虑了学习基于无向高斯图模型的稀疏图的问题，这是统计机器学习中的一个关键问题。给定来自具有p个变量的多元高斯分布的n个样本，目标是估计p×p的逆协方差矩阵（也称为精度矩阵），假设它是稀疏的（即具有少数非零条目）。我们提出了GraphL0BnB这一新的估计方法，它基于伪似然函数的l0惩罚版本，而大多数早期方法都是基于l1松弛。我们的估计方法可以被形式化为一个凸混合整数规划（MIP），使用现成的商用求解器在大规模计算时可能很难计算。为了解决MIP问题，我们提出了一个定制的非线性分支定界（BnB）框架，用于使用定制的一阶方法来解决节点放松问题。作为我们BnB框架的副产品，我们提出了用于获得独立兴趣的良好原始解的大规模求解器。

    We consider the problem of learning a sparse graph underlying an undirected Gaussian graphical model, a key problem in statistical machine learning. Given $n$ samples from a multivariate Gaussian distribution with $p$ variables, the goal is to estimate the $p \times p$ inverse covariance matrix (aka precision matrix), assuming it is sparse (i.e., has a few nonzero entries). We propose GraphL0BnB, a new estimator based on an $\ell_0$-penalized version of the pseudolikelihood function, while most earlier approaches are based on the $\ell_1$-relaxation. Our estimator can be formulated as a convex mixed integer program (MIP) which can be difficult to compute at scale using off-the-shelf commercial solvers. To solve the MIP, we propose a custom nonlinear branch-and-bound (BnB) framework that solves node relaxations with tailored first-order methods. As a by-product of our BnB framework, we propose large-scale solvers for obtaining good primal solutions that are of independent interest. We d
    
[^5]: 加速梯度方法用于非凸优化：逃逸轨迹和收敛到局部极小值点

    Accelerated gradient methods for nonconvex optimization: Escape trajectories from strict saddle points and convergence to local minima. (arXiv:2307.07030v1 [math.OC])

    [http://arxiv.org/abs/2307.07030](http://arxiv.org/abs/2307.07030)

    本文研究了一类加速梯度方法在非凸优化问题上的行为，包括逃离鞍点和收敛到局部极小值点的分析。研究在渐进和非渐进情况下，提出了一类新的Nesterov类型的加速方法，并回答了Nesterov加速梯度方法是否避免了严格鞍点的问题。

    

    本文研究了一类广义的加速梯度方法在光滑非凸函数上的行为。通过对Polyak的重球方法和Nesterov加速梯度方法进行改进，以实现对非凸函数局部极小值的收敛，本文提出了一类Nesterov类型的加速方法，并通过渐进分析和非渐进分析对这些方法进行了严格研究，包括逃离鞍点和收敛到局部极小值点。在渐进情况下，本文回答了一个开放问题，即带有可变动量参数的Nesterov加速梯度方法（NAG）是否几乎必定避免了严格鞍点。本文还提出了两种渐进收敛和发散的度量方式，并对几种常用的标准加速方法（如NAG和Ne）进行了评估。

    This paper considers the problem of understanding the behavior of a general class of accelerated gradient methods on smooth nonconvex functions. Motivated by some recent works that have proposed effective algorithms, based on Polyak's heavy ball method and the Nesterov accelerated gradient method, to achieve convergence to a local minimum of nonconvex functions, this work proposes a broad class of Nesterov-type accelerated methods and puts forth a rigorous study of these methods encompassing the escape from saddle-points and convergence to local minima through a both asymptotic and a non-asymptotic analysis. In the asymptotic regime, this paper answers an open question of whether Nesterov's accelerated gradient method (NAG) with variable momentum parameter avoids strict saddle points almost surely. This work also develops two metrics of asymptotic rate of convergence and divergence, and evaluates these two metrics for several popular standard accelerated methods such as the NAG, and Ne
    
[^6]: Sinkhorn算法的重要性稀疏化

    Importance Sparsification for Sinkhorn Algorithm. (arXiv:2306.06581v1 [stat.ML])

    [http://arxiv.org/abs/2306.06581](http://arxiv.org/abs/2306.06581)

    Spar-Sink是一种重要性稀疏化方法，能够有效近似熵正则化最优传输和不平衡最优传输问题，并且在实验中表现优异。

    

    Sinkhorn算法被广泛应用于近似求解最优传输（OT）和不平衡最优传输（UOT）问题。但由于高计算复杂度，其实际应用受到限制。为减轻计算负担，我们提出了一种新的重要性稀疏化方法Spar-Sink，用于高效近似熵正则化OT和UOT解。具体来说，我们的方法利用未知最优传输计划的自然上界确定有效的采样概率，并构建稀疏的核矩阵以加速Sinkhorn迭代，将每次迭代的计算成本从$ O（n ^ 2）$降低到$\widetilde {O（n）}$适用于样本大小为$ n $的情况。理论上，我们证明了对于温和正则性条件下，所提出的OT和UOT问题的估计量是一致的。在各种合成数据上的实验表明，在估计误差方面，Spar-Sink优于主流竞争对手。

    Sinkhorn algorithm has been used pervasively to approximate the solution to optimal transport (OT) and unbalanced optimal transport (UOT) problems. However, its practical application is limited due to the high computational complexity. To alleviate the computational burden, we propose a novel importance sparsification method, called Spar-Sink, to efficiently approximate entropy-regularized OT and UOT solutions. Specifically, our method employs natural upper bounds for unknown optimal transport plans to establish effective sampling probabilities, and constructs a sparse kernel matrix to accelerate Sinkhorn iterations, reducing the computational cost of each iteration from $O(n^2)$ to $\widetilde{O}(n)$ for a sample of size $n$. Theoretically, we show the proposed estimators for the regularized OT and UOT problems are consistent under mild regularity conditions. Experiments on various synthetic data demonstrate Spar-Sink outperforms mainstream competitors in terms of both estimation erro
    
[^7]: 多臂赌博机的上下文利用与探索的神经网络研究

    Neural Exploitation and Exploration of Contextual Bandits. (arXiv:2305.03784v1 [cs.LG])

    [http://arxiv.org/abs/2305.03784](http://arxiv.org/abs/2305.03784)

    本文提出了一种新型的神经网络策略"EE-Net"，它用于多臂赌博机的利用和探索，在学习奖励函数的同时也适应性地学习潜在收益。

    

    本文研究利用神经网络进行上下文多臂赌博机的利用和探索。我们提出了一个名为"EE-Net"的新型神经网络利用和探索策略，它使用一个神经网络（利用网络）来学习奖励函数，另一个神经网络（探索网络）来适应性地学习相对于当前估计奖励的潜在收益。

    In this paper, we study utilizing neural networks for the exploitation and exploration of contextual multi-armed bandits. Contextual multi-armed bandits have been studied for decades with various applications. To solve the exploitation-exploration trade-off in bandits, there are three main techniques: epsilon-greedy, Thompson Sampling (TS), and Upper Confidence Bound (UCB). In recent literature, a series of neural bandit algorithms have been proposed to adapt to the non-linear reward function, combined with TS or UCB strategies for exploration. In this paper, instead of calculating a large-deviation based statistical bound for exploration like previous methods, we propose, ``EE-Net,'' a novel neural-based exploitation and exploration strategy. In addition to using a neural network (Exploitation network) to learn the reward function, EE-Net uses another neural network (Exploration network) to adaptively learn the potential gains compared to the currently estimated reward for exploration
    
[^8]: 基于分段确定性马尔可夫过程的贝叶斯神经网络研究

    Piecewise Deterministic Markov Processes for Bayesian Neural Networks. (arXiv:2302.08724v2 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2302.08724](http://arxiv.org/abs/2302.08724)

    本文介绍了基于分段确定性马尔可夫过程的贝叶斯神经网络推理方法，通过引入新的自适应稀疏方案，实现了对困难采样问题的加速处理。实验证明，这种方法在计算上可行，并能提高预测准确性、MCMC混合性能，并提供更有信息量的不确定性测量。

    

    现代贝叶斯神经网络（BNNs）的推理通常依赖于变分推断处理，这要求违反了独立性和后验形式的假设。传统的MCMC方法避免了这些假设，但由于无法适应似然的子采样，导致计算量增加。新的分段确定性马尔可夫过程（PDMP）采样器允许子采样，但引入了模型特定的不均匀泊松过程（IPPs），从中采样困难。本研究引入了一种新的通用自适应稀疏方案，用于从这些IPPs中进行采样，并展示了如何加速将PDMPs应用于BNNs推理。实验表明，使用这些方法进行推理在计算上是可行的，可以提高预测准确性、MCMC混合性能，并与其他近似推理方案相比，提供更有信息量的不确定性测量。

    Inference on modern Bayesian Neural Networks (BNNs) often relies on a variational inference treatment, imposing violated assumptions of independence and the form of the posterior. Traditional MCMC approaches avoid these assumptions at the cost of increased computation due to its incompatibility to subsampling of the likelihood. New Piecewise Deterministic Markov Process (PDMP) samplers permit subsampling, though introduce a model specific inhomogenous Poisson Process (IPPs) which is difficult to sample from. This work introduces a new generic and adaptive thinning scheme for sampling from these IPPs, and demonstrates how this approach can accelerate the application of PDMPs for inference in BNNs. Experimentation illustrates how inference with these methods is computationally feasible, can improve predictive accuracy, MCMC mixing performance, and provide informative uncertainty measurements when compared against other approximate inference schemes.
    

