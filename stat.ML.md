# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Generative Adversarial Learning of Sinkhorn Algorithm Initializations](https://rss.arxiv.org/abs/2212.00133) | 本文通过生成对抗学习的方法，训练神经网络来学习Sinkhorn算法的初始化，显著加快收敛速度，同时保持算法的可微分性和并行性，并证明了网络的普适性和独立求解能力。 |
| [^2] | [A Primal-Dual Algorithm for Faster Distributionally Robust Optimization](https://arxiv.org/abs/2403.10763) | 这种原始-对偶算法在分布鲁棒优化问题中实现了最先进的线性收敛速度。 |
| [^3] | [Reinforcement Learning from Human Feedback with Active Queries](https://arxiv.org/abs/2402.09401) | 本文提出了一种基于主动查询的强化学习方法，用于解决与人类反馈的对齐问题。通过在强化学习过程中减少人工标注偏好数据的需求，该方法具有较低的代价，并在实验中表现出较好的性能。 |
| [^4] | [A PAC-Bayesian Link Between Generalisation and Flat Minima](https://arxiv.org/abs/2402.08508) | 本研究结合了PAC-Bayes工具箱和Poincaré与Log-Sobolev不等式，提供了新的梯度项泛化界限，并突出了平坦最小值对泛化性能的积极影响。 |
| [^5] | [A duality framework for generalization analysis of random feature models and two-layer neural networks.](http://arxiv.org/abs/2305.05642) | 本文提出了一个针对随机特征模型和双层神经网络的泛化分析的对偶性框架，并证明了学习不会受到维数灾难的影响，使 RFMs 可以在核范围之外发挥作用。 |
| [^6] | [Generalisation under gradient descent via deterministic PAC-Bayes.](http://arxiv.org/abs/2209.02525) | 本文介绍了一种新的PAC-Bayesian泛化界限，适用于使用梯度下降方法或连续梯度流训练模型的优化算法，且无需随机化。 |

# 详细

[^1]: 生成对抗学习Sinkhorn算法初始化

    Generative Adversarial Learning of Sinkhorn Algorithm Initializations

    [https://rss.arxiv.org/abs/2212.00133](https://rss.arxiv.org/abs/2212.00133)

    本文通过生成对抗学习的方法，训练神经网络来学习Sinkhorn算法的初始化，显著加快收敛速度，同时保持算法的可微分性和并行性，并证明了网络的普适性和独立求解能力。

    

    Sinkhorn算法是近似求解离散概率分布之间熵正则输运（OT）距离的最先进方法。我们展示了通过训练神经网络来学习算法初始化，可以显著加快收敛速度，同时保持Sinkhorn算法的可微分性和并行性。我们通过对抗训练的方式使用第二个生成网络和自监督引导损失来训练我们的预测网络。预测网络具有普适性，能够推广到任意固定维度和成本的概率分布对，并且我们证明生成网络可以在训练过程中产生任意概率分布对。此外，我们还展示了我们的网络可以作为独立的OT求解器来近似正则化输运问题。

    The Sinkhorn algorithm is the state-of-the-art to approximate solutions of entropic optimal transport (OT) distances between discrete probability distributions. We show that meticulously training a neural network to learn initializations to the algorithm via the entropic OT dual problem can significantly speed up convergence, while maintaining desirable properties of the Sinkhorn algorithm, such as differentiability and parallelizability. We train our predictive network in an adversarial fashion using a second, generating network and a self-supervised bootstrapping loss. The predictive network is universal in the sense that it is able to generalize to any pair of distributions of fixed dimension and cost at inference, and we prove that we can make the generating network universal in the sense that it is capable of producing any pair of distributions during training. Furthermore, we show that our network can even be used as a standalone OT solver to approximate regularized transport dis
    
[^2]: 一种用于更快分布鲁棒优化问题的原始-对偶算法

    A Primal-Dual Algorithm for Faster Distributionally Robust Optimization

    [https://arxiv.org/abs/2403.10763](https://arxiv.org/abs/2403.10763)

    这种原始-对偶算法在分布鲁棒优化问题中实现了最先进的线性收敛速度。

    

    我们考虑带有闭合、凸不确定性集的惩罚分布鲁棒优化（DRO）问题，这个设置包括了实践中使用的$f$-DRO、Wasserstein-DRO和谱/$L$-风险公式。我们提出了Drago，一种随机原始-对偶算法，在强凸-强凹DRO问题上实现了最先进的线性收敛速度。该方法将随机化和循环组件与小批量结合，有效处理了DRO中原始和对偶问题的独特不对称性质。我们通过分类和回归中的数值基准支持我们的理论结果。

    arXiv:2403.10763v1 Announce Type: cross  Abstract: We consider the penalized distributionally robust optimization (DRO) problem with a closed, convex uncertainty set, a setting that encompasses the $f$-DRO, Wasserstein-DRO, and spectral/$L$-risk formulations used in practice. We present Drago, a stochastic primal-dual algorithm that achieves a state-of-the-art linear convergence rate on strongly convex-strongly concave DRO problems. The method combines both randomized and cyclic components with mini-batching, which effectively handles the unique asymmetric nature of the primal and dual problems in DRO. We support our theoretical results with numerical benchmarks in classification and regression.
    
[^3]: 使用主动查询的人类反馈强化学习

    Reinforcement Learning from Human Feedback with Active Queries

    [https://arxiv.org/abs/2402.09401](https://arxiv.org/abs/2402.09401)

    本文提出了一种基于主动查询的强化学习方法，用于解决与人类反馈的对齐问题。通过在强化学习过程中减少人工标注偏好数据的需求，该方法具有较低的代价，并在实验中表现出较好的性能。

    

    将大型语言模型（LLM）与人类偏好进行对齐，在构建现代生成模型中发挥重要作用，这可以通过从人类反馈中进行强化学习来实现。然而，尽管当前的强化学习方法表现出优越性能，但往往需要大量的人工标注偏好数据，而这种数据收集费时费力。本文受到主动学习的成功启发，通过提出查询效率高的强化学习方法来解决这个问题。我们首先将对齐问题形式化为上下文竞争二臂强盗问题，并设计了基于主动查询的近端策略优化（APPO）算法，具有$\tilde{O}(d^2/\Delta)$的遗憾界和$\tilde{O}(d^2/\Delta^2)$的查询复杂度，其中$d$是特征空间的维度，$\Delta$是所有上下文中的次优差距。然后，我们提出了ADPO，这是我们算法的实际版本，基于直接偏好优化（DPO）并将其应用于...

    arXiv:2402.09401v1 Announce Type: cross Abstract: Aligning large language models (LLM) with human preference plays a key role in building modern generative models and can be achieved by reinforcement learning from human feedback (RLHF). Despite their superior performance, current RLHF approaches often require a large amount of human-labelled preference data, which is expensive to collect. In this paper, inspired by the success of active learning, we address this problem by proposing query-efficient RLHF methods. We first formalize the alignment problem as a contextual dueling bandit problem and design an active-query-based proximal policy optimization (APPO) algorithm with an $\tilde{O}(d^2/\Delta)$ regret bound and an $\tilde{O}(d^2/\Delta^2)$ query complexity, where $d$ is the dimension of feature space and $\Delta$ is the sub-optimality gap over all the contexts. We then propose ADPO, a practical version of our algorithm based on direct preference optimization (DPO) and apply it to 
    
[^4]: 广义和平均容量之间的PAC-Bayes联结

    A PAC-Bayesian Link Between Generalisation and Flat Minima

    [https://arxiv.org/abs/2402.08508](https://arxiv.org/abs/2402.08508)

    本研究结合了PAC-Bayes工具箱和Poincaré与Log-Sobolev不等式，提供了新的梯度项泛化界限，并突出了平坦最小值对泛化性能的积极影响。

    

    现代机器学习通常使用超参数设置（训练参数数量大于数据集大小）中的预测器，它们的训练不仅产生良好的训练数据性能，而且具有良好的泛化能力。这一现象挑战了许多理论结果，并且仍然是一个未解决的问题。为了更好地理解这一现象，我们提供了涉及梯度项的新型泛化界限。为此，我们将PAC-Bayes工具箱与Poincaré和Log-Sobolev不等式相结合，避免了对预测器空间维数的显式依赖。我们的结果突出了“平坦最小值”（几乎能够最小化学习问题的邻近最小值）对泛化性能的积极影响，直接涉及到优化阶段的好处。

    Modern machine learning usually involves predictors in the overparametrised setting (number of trained parameters greater than dataset size), and their training yield not only good performances on training data, but also good generalisation capacity. This phenomenon challenges many theoretical results, and remains an open problem. To reach a better understanding, we provide novel generalisation bounds involving gradient terms. To do so, we combine the PAC-Bayes toolbox with Poincar\'e and Log-Sobolev inequalities, avoiding an explicit dependency on dimension of the predictor space. Our results highlight the positive influence of \emph{flat minima} (being minima with a neighbourhood nearly minimising the learning problem as well) on generalisation performances, involving directly the benefits of the optimisation phase.
    
[^5]: 随机特征模型和双层神经网络的泛化分析的对偶性框架

    A duality framework for generalization analysis of random feature models and two-layer neural networks. (arXiv:2305.05642v1 [stat.ML])

    [http://arxiv.org/abs/2305.05642](http://arxiv.org/abs/2305.05642)

    本文提出了一个针对随机特征模型和双层神经网络的泛化分析的对偶性框架，并证明了学习不会受到维数灾难的影响，使 RFMs 可以在核范围之外发挥作用。

    

    本文研究在高维分析中出现的自然函数空间 $\mathcal{F}_{p,\pi}$ 和 Barron 空间中学习函数的问题。通过对偶分析，我们揭示了这些空间的逼近和估计可以在某种意义下被视为等价的。这使得我们能够在研究这两种模型的泛化时更专注于更容易的逼近和估计问题。通过定义一种基于信息的复杂度来有效地控制估计误差，建立了对偶等价性。此外，我们通过对两个具体应用进行综合分析展示了我们的对偶性框架的灵活性。第一个应用是研究使用 RFMs 学习 $\mathcal{F}_{p,\pi}$ 中的函数。我们证明只要 $p>1$，学习不会受到维数灾难的影响，这意味着 RFMs 可以在核范围之外发挥作用。

    We consider the problem of learning functions in the $\mathcal{F}_{p,\pi}$ and Barron spaces, which are natural function spaces that arise in the high-dimensional analysis of random feature models (RFMs) and two-layer neural networks. Through a duality analysis, we reveal that the approximation and estimation of these spaces can be considered equivalent in a certain sense. This enables us to focus on the easier problem of approximation and estimation when studying the generalization of both models. The dual equivalence is established by defining an information-based complexity that can effectively control estimation errors. Additionally, we demonstrate the flexibility of our duality framework through comprehensive analyses of two concrete applications.  The first application is to study learning functions in $\mathcal{F}_{p,\pi}$ with RFMs. We prove that the learning does not suffer from the curse of dimensionality as long as $p>1$, implying RFMs can work beyond the kernel regime. Our 
    
[^6]: 基于确定性PAC-Bayes的梯度下降下的泛化

    Generalisation under gradient descent via deterministic PAC-Bayes. (arXiv:2209.02525v3 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2209.02525](http://arxiv.org/abs/2209.02525)

    本文介绍了一种新的PAC-Bayesian泛化界限，适用于使用梯度下降方法或连续梯度流训练模型的优化算法，且无需随机化。

    

    我们为使用梯度下降方法或连续梯度流训练模型建立了细分的PAC-Bayesian泛化界限。与PAC-Bayes设定中的标准做法相反，我们的结果适用于确定性的优化算法，而不需要任何去随机化的步骤。我们的界限是完全可计算的，取决于初始分布的密度和轨迹上训练目标的海森矩阵。我们展示了我们的框架可以应用于各种迭代优化算法，包括随机梯度下降（SGD）、动量算法和阻尼哈密顿动力学。

    We establish disintegrated PAC-Bayesian generalisation bounds for models trained with gradient descent methods or continuous gradient flows. Contrary to standard practice in the PAC-Bayesian setting, our result applies to optimisation algorithms that are deterministic, without requiring any de-randomisation step. Our bounds are fully computable, depending on the density of the initial distribution and the Hessian of the training objective over the trajectory. We show that our framework can be applied to a variety of iterative optimisation algorithms, including stochastic gradient descent (SGD), momentum-based schemes, and damped Hamiltonian dynamics.
    

