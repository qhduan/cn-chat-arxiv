# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Combining Evidence Across Filtrations](https://arxiv.org/abs/2402.09698) | 这篇论文研究了合并使用不同过滤器计算的e进程的方法，探讨了其在顺序推理中的应用。 |
| [^2] | [Measure transfer via stochastic slicing and matching.](http://arxiv.org/abs/2307.05705) | 本文研究了通过切片和配准过程定义的测量转移和逼近问题的迭代方案，并对随机切片和配准方案提供了几乎必然收敛的证明。 |
| [^3] | [Trainability barriers and opportunities in quantum generative modeling.](http://arxiv.org/abs/2305.02881) | 本文研究了量子生成模型的可训练性障碍，如荒芜高原和指数损失集中，使用隐式生成模型和明确损失会产生一种新的荒芜高原现象。最大均值差可以是低秩且可训练的或全局性且不可训练的。但是，可训练性所需的低秩损失通常不能区分高频和低频特征。 |
| [^4] | [Skeleton Regression: A Graph-Based Approach to Estimation with Manifold Structure.](http://arxiv.org/abs/2303.11786) | 这是一个处理低维流形数据的回归框架，首先通过构建图形骨架来捕捉潜在的流形几何结构，然后在其上运用非参数回归技术来估计回归函数，除了具有非参数优点之外，在处理多个流形数据，嘈杂观察时也表现出较好的鲁棒性。 |
| [^5] | [Offline Estimation of Controlled Markov Chains: Minimaxity and Sample Complexity.](http://arxiv.org/abs/2211.07092) | 本论文研究了离线估计有限控制马尔可夫链的转移概率矩阵的非参数估计器，并通过记录策略的混合特性建立了样本复杂性界限和最小化条件。结果表明，实现特定的统计风险界限涉及到混合特性的强度和样本数量之间微妙而有趣的权衡。还使用这些样本复杂性界限建立了离线评估恒定马尔可夫控制策略的相关界限。 |

# 详细

[^1]: 合并不同过滤器中的证据

    Combining Evidence Across Filtrations

    [https://arxiv.org/abs/2402.09698](https://arxiv.org/abs/2402.09698)

    这篇论文研究了合并使用不同过滤器计算的e进程的方法，探讨了其在顺序推理中的应用。

    

    在任何时刻有效的顺序推理中，已知任何可接受的推理方法必须基于测试鞅和它们的组合广义化，称为e进程，它们是非负进程，其在任何任意停时的期望上界不超过一。e进程量化了针对复合零假设的一系列结果的累积证据。本文研究了使用不同信息集（即过滤器）计算的e进程的合并方法，针对一个零假设。尽管在相同过滤器上构建的e进程可以轻松地合并（例如，通过平均），但在不同过滤器上构建的e进程不能那么容易地合并，因为它们在较粗的过滤器中的有效性不能转换为在更细的过滤器中的有效性。我们讨论了文献中三个具体例子：可交换性测试，独立性测试等。

    arXiv:2402.09698v1 Announce Type: cross  Abstract: In anytime-valid sequential inference, it is known that any admissible inference procedure must be based on test martingales and their composite generalization, called e-processes, which are nonnegative processes whose expectation at any arbitrary stopping time is upper-bounded by one. An e-process quantifies the accumulated evidence against a composite null hypothesis over a sequence of outcomes. This paper studies methods for combining e-processes that are computed using different information sets, i.e., filtrations, for a null hypothesis. Even though e-processes constructed on the same filtration can be combined effortlessly (e.g., by averaging), e-processes constructed on different filtrations cannot be combined as easily because their validity in a coarser filtration does not translate to validity in a finer filtration. We discuss three concrete examples of such e-processes in the literature: exchangeability tests, independence te
    
[^2]: 通过随机切片和配准进行测量转移

    Measure transfer via stochastic slicing and matching. (arXiv:2307.05705v1 [math.NA])

    [http://arxiv.org/abs/2307.05705](http://arxiv.org/abs/2307.05705)

    本文研究了通过切片和配准过程定义的测量转移和逼近问题的迭代方案，并对随机切片和配准方案提供了几乎必然收敛的证明。

    

    本论文研究了通过切片和配准过程定义的测量转移和逼近问题的迭代方案。类似于切片Wasserstein距离，这些方案受益于一维最优输运问题的闭式解的可用性和相关计算优势。尽管这些方案已经在数据科学应用中取得了成功，但关于它们的收敛性的结果不太多。本文的主要贡献是对随机切片和配准方案提供了几乎必然收敛的证明。该证明建立在将其解释为Wasserstein空间上的随机梯度下降方案的基础之上。同时还展示了关于逐步图像变形的数值示例。

    This paper studies iterative schemes for measure transfer and approximation problems, which are defined through a slicing-and-matching procedure. Similar to the sliced Wasserstein distance, these schemes benefit from the availability of closed-form solutions for the one-dimensional optimal transport problem and the associated computational advantages. While such schemes have already been successfully utilized in data science applications, not too many results on their convergence are available. The main contribution of this paper is an almost sure convergence proof for stochastic slicing-and-matching schemes. The proof builds on an interpretation as a stochastic gradient descent scheme on the Wasserstein space. Numerical examples on step-wise image morphing are demonstrated as well.
    
[^3]: 量子生成建模中的可训练性障碍和机遇

    Trainability barriers and opportunities in quantum generative modeling. (arXiv:2305.02881v1 [quant-ph])

    [http://arxiv.org/abs/2305.02881](http://arxiv.org/abs/2305.02881)

    本文研究了量子生成模型的可训练性障碍，如荒芜高原和指数损失集中，使用隐式生成模型和明确损失会产生一种新的荒芜高原现象。最大均值差可以是低秩且可训练的或全局性且不可训练的。但是，可训练性所需的低秩损失通常不能区分高频和低频特征。

    

    量子生成模型提供了本质高效的采样策略，因此在量子硬件上实现近期优势有很大的潜力。然而，它们的可扩展性仍存在重要问题。本文研究量子生成模型的可训练性障碍，如荒芜高原和指数损失集中。我们探索了明确和隐含模型、损失之间的相互作用，并表明使用隐式生成模型（如基于量子电路的模型）和明确损失（如KL散度）会产生一种新的荒芜高原现象。相比之下，最大均值差（MMD），作为隐式损失的一个流行例子，可以看作是一个观测量的期望值，该观测量可能是低秩且可训练的，也可能是全局性且不可训练的，具体取决于核函数的选择。然而，我们同时强调，可训练性所需的低秩损失通常不能区分高频和低频特征。

    Quantum generative models, in providing inherently efficient sampling strategies, show promise for achieving a near-term advantage on quantum hardware. Nonetheless, important questions remain regarding their scalability. In this work, we investigate the barriers to the trainability of quantum generative models posed by barren plateaus and exponential loss concentration. We explore the interplay between explicit and implicit models and losses, and show that using implicit generative models (such as quantum circuit-based models) with explicit losses (such as the KL divergence) leads to a new flavour of barren plateau. In contrast, the Maximum Mean Discrepancy (MMD), which is a popular example of an implicit loss, can be viewed as the expectation value of an observable that is either low-bodied and trainable, or global and untrainable depending on the choice of kernel. However, in parallel, we highlight that the low-bodied losses required for trainability cannot in general distinguish hig
    
[^4]: Skeleton Regression：一种基于流形结构估计的基于图形的方法。

    Skeleton Regression: A Graph-Based Approach to Estimation with Manifold Structure. (arXiv:2303.11786v1 [cs.LG])

    [http://arxiv.org/abs/2303.11786](http://arxiv.org/abs/2303.11786)

    这是一个处理低维流形数据的回归框架，首先通过构建图形骨架来捕捉潜在的流形几何结构，然后在其上运用非参数回归技术来估计回归函数，除了具有非参数优点之外，在处理多个流形数据，嘈杂观察时也表现出较好的鲁棒性。

    

    我们引入了一个新的回归框架，旨在处理围绕低维流形的复杂数据。我们的方法首先构建一个图形表示，称为骨架，以捕获潜在的几何结构。然后，我们在骨架图上定义指标，应用非参数回归技术，以及基于图形的特征转换来估计回归函数。除了包括的非参数方法外，我们还讨论了一些非参数回归器在骨架图等一般度量空间方面的限制。所提出的回归框架使我们能够避开维度灾难，具有可以处理多个流形的并集并且鲁棒性能应对加性噪声和嘈杂观察的额外优势。我们为所提出的方法提供了统计保证，并通过模拟和实际数据示例证明了其有效性。

    We introduce a new regression framework designed to deal with large-scale, complex data that lies around a low-dimensional manifold. Our approach first constructs a graph representation, referred to as the skeleton, to capture the underlying geometric structure. We then define metrics on the skeleton graph and apply nonparametric regression techniques, along with feature transformations based on the graph, to estimate the regression function. In addition to the included nonparametric methods, we also discuss the limitations of some nonparametric regressors with respect to the general metric space such as the skeleton graph. The proposed regression framework allows us to bypass the curse of dimensionality and provides additional advantages that it can handle the union of multiple manifolds and is robust to additive noise and noisy observations. We provide statistical guarantees for the proposed method and demonstrate its effectiveness through simulations and real data examples.
    
[^5]: 离线估计控制马尔可夫链：最小化和样本复杂度

    Offline Estimation of Controlled Markov Chains: Minimaxity and Sample Complexity. (arXiv:2211.07092v4 [stat.ML] UPDATED)

    [http://arxiv.org/abs/2211.07092](http://arxiv.org/abs/2211.07092)

    本论文研究了离线估计有限控制马尔可夫链的转移概率矩阵的非参数估计器，并通过记录策略的混合特性建立了样本复杂性界限和最小化条件。结果表明，实现特定的统计风险界限涉及到混合特性的强度和样本数量之间微妙而有趣的权衡。还使用这些样本复杂性界限建立了离线评估恒定马尔可夫控制策略的相关界限。

    

    在这项工作中，我们研究了有限控制马尔可夫链的转移概率矩阵的自然非参数估计器。我们考虑了一个固定数据集的离线设置，该数据集是使用所谓的记录策略收集的。我们为估计器开发了样本复杂性的界限，并建立了最小化的条件。我们的统计界限通过记录策略的混合特性来确定。我们表明，实现特定的统计风险界限涉及到混合特性的强度和样本数量之间微妙而有趣的权衡。我们在各种示例中验证了我们结果的有效性，包括遗传马尔可夫链，弱遗传非齐次马尔可夫链和具有非平稳马尔可夫、阶段性和贪婪控制的控制马尔可夫链。最后，我们使用这些样本复杂性界限来建立离线评估恒定马尔可夫控制策略的相关界限。

    In this work, we study a natural nonparametric estimator of the transition probability matrices of a finite controlled Markov chain. We consider an offline setting with a fixed dataset, collected using a so-called logging policy. We develop sample complexity bounds for the estimator and establish conditions for minimaxity. Our statistical bounds depend on the logging policy through its mixing properties. We show that achieving a particular statistical risk bound involves a subtle and interesting trade-off between the strength of the mixing properties and the number of samples. We demonstrate the validity of our results under various examples, such as ergodic Markov chains, weakly ergodic inhomogeneous Markov chains, and controlled Markov chains with non-stationary Markov, episodic, and greedy controls. Lastly, we use these sample complexity bounds to establish concomitant ones for offline evaluation of stationary Markov control policies.
    

