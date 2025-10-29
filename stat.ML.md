# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Diffusion Models Meet Contextual Bandits with Large Action Spaces](https://arxiv.org/abs/2402.10028) | 本文设计了一种利用预训练扩散模型的扩散汤普森采样方法，用于在大动作空间下进行高效的情境强化学习探索。实证评估结果表明了该方法的优越性能。 |
| [^2] | [Optimistically Tempered Online Learning](https://arxiv.org/abs/2301.07530) | 本文提出了一种乐观调节的在线学习框架和适应算法，挑战了对专家的信心假设，并通过动态遗憾界限的理论保证和实验证明了该方法的有效性。 |
| [^3] | [Minimax Optimal Transfer Learning for Kernel-based Nonparametric Regression.](http://arxiv.org/abs/2310.13966) | 本文主要研究了在再生核希尔伯特空间中的非参数回归的传递学习问题，提出了两种情况下的解决方法，并分别给出了统计性质和最优性结果。 |

# 详细

[^1]: 扩散模型与大动作空间情境强化学习的结合

    Diffusion Models Meet Contextual Bandits with Large Action Spaces

    [https://arxiv.org/abs/2402.10028](https://arxiv.org/abs/2402.10028)

    本文设计了一种利用预训练扩散模型的扩散汤普森采样方法，用于在大动作空间下进行高效的情境强化学习探索。实证评估结果表明了该方法的优越性能。

    

    由于动作空间较大，有效的探索是情境强化学习中的一个关键挑战。本文通过利用预训练的扩散模型来捕捉动作之间的相关性，设计了扩散汤普森采样（dTS）方法，实现了高效的探索。我们为dTS方法提供了理论和算法基础，并通过实证评估展示了它的优越性能。

    arXiv:2402.10028v1 Announce Type: cross  Abstract: Efficient exploration is a key challenge in contextual bandits due to the large size of their action space, where uninformed exploration can result in computational and statistical inefficiencies. Fortunately, the rewards of actions are often correlated and this can be leveraged to explore them efficiently. In this work, we capture such correlations using pre-trained diffusion models; upon which we design diffusion Thompson sampling (dTS). Both theoretical and algorithmic foundations are developed for dTS, and empirical evaluation also shows its favorable performance.
    
[^2]: 乐观调节的在线学习

    Optimistically Tempered Online Learning

    [https://arxiv.org/abs/2301.07530](https://arxiv.org/abs/2301.07530)

    本文提出了一种乐观调节的在线学习框架和适应算法，挑战了对专家的信心假设，并通过动态遗憾界限的理论保证和实验证明了该方法的有效性。

    

    乐观在线学习算法已经被开发出来，以利用专家意见，假设专家意见总是有用的。然而，我们可以合理地对这些意见与基于梯度的在线算法提供的学习信息的相关性提出质疑。在这项工作中，我们质疑对专家的信心假设，并开发了乐观调节（OT）在线学习框架以及在线算法的OT适应性。我们的算法具有动态遗憾界限的稳固理论保证，并最终验证了OT方法的有用性。

    arXiv:2301.07530v2 Announce Type: replace Abstract: Optimistic Online Learning algorithms have been developed to exploit expert advices, assumed optimistically to be always useful. However, it is legitimate to question the relevance of such advices \emph{w.r.t.} the learning information provided by gradient-based online algorithms. In this work, we challenge the confidence assumption on the expert and develop the \emph{optimistically tempered} (OT) online learning framework as well as OT adaptations of online algorithms. Our algorithms come with sound theoretical guarantees in the form of dynamic regret bounds, and we eventually provide experimental validation of the usefulness of the OT approach.
    
[^3]: 基于核非参数回归的最优极小化传递学习

    Minimax Optimal Transfer Learning for Kernel-based Nonparametric Regression. (arXiv:2310.13966v1 [stat.ML])

    [http://arxiv.org/abs/2310.13966](http://arxiv.org/abs/2310.13966)

    本文主要研究了在再生核希尔伯特空间中的非参数回归的传递学习问题，提出了两种情况下的解决方法，并分别给出了统计性质和最优性结果。

    

    近年来，传递学习在机器学习社区中受到了很大关注。它能够利用相关研究的知识来提高目标研究的泛化性能，使其具有很高的吸引力。本文主要研究在再生核希尔伯特空间中的非参数回归的传递学习问题，目的是缩小实际效果与理论保证之间的差距。具体考虑了两种情况：已知可传递的来源和未知的情况。对于已知可传递的来源情况，我们提出了一个两步核估计器，仅使用核岭回归。对于未知的情况，我们开发了一种基于高效聚合算法的新方法，可以自动检测并减轻负面来源的影响。本文提供了所需估计器的统计性质，并建立了该方法的最优性结果。

    In recent years, transfer learning has garnered significant attention in the machine learning community. Its ability to leverage knowledge from related studies to improve generalization performance in a target study has made it highly appealing. This paper focuses on investigating the transfer learning problem within the context of nonparametric regression over a reproducing kernel Hilbert space. The aim is to bridge the gap between practical effectiveness and theoretical guarantees. We specifically consider two scenarios: one where the transferable sources are known and another where they are unknown. For the known transferable source case, we propose a two-step kernel-based estimator by solely using kernel ridge regression. For the unknown case, we develop a novel method based on an efficient aggregation algorithm, which can automatically detect and alleviate the effects of negative sources. This paper provides the statistical properties of the desired estimators and establishes the 
    

