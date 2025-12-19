# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Optimization with access to auxiliary information.](http://arxiv.org/abs/2206.00395) | 本文研究了在计算目标函数梯度很昂贵或有限的情况下，给定一些辅助函数的情况下，如何最小化目标函数。作者提出了两种通用的新算法，并证明了这个框架可以受益于目标和辅助信息之间的Hessian相似性假设。 |

# 详细

[^1]: 具备辅助信息的优化

    Optimization with access to auxiliary information. (arXiv:2206.00395v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2206.00395](http://arxiv.org/abs/2206.00395)

    本文研究了在计算目标函数梯度很昂贵或有限的情况下，给定一些辅助函数的情况下，如何最小化目标函数。作者提出了两种通用的新算法，并证明了这个框架可以受益于目标和辅助信息之间的Hessian相似性假设。

    This paper investigates the fundamental optimization question of minimizing a target function with expensive or limited gradient computation, given access to some auxiliary side function with cheaper or more available gradients. The authors propose two generic new algorithms and prove that this framework can benefit from the Hessian similarity assumption between the target and side information.

    我们研究了基本的优化问题，即在计算目标函数$f(x)$的梯度很昂贵或有限的情况下，给定一些辅助函数$h(x)$的情况下，如何最小化目标函数。这个公式涵盖了许多实际相关的设置，如i）在SGD中重复使用批次，ii）迁移学习，iii）联邦学习，iv）使用压缩模型/丢弃等进行训练。我们提出了两种通用的新算法，适用于所有这些设置，并证明仅使用目标和辅助信息之间的Hessian相似性假设，我们可以从这个框架中受益。

    We investigate the fundamental optimization question of minimizing a target function $f(x)$ whose gradients are expensive to compute or have limited availability, given access to some auxiliary side function $h(x)$ whose gradients are cheap or more available. This formulation captures many settings of practical relevance such as i) re-using batches in SGD, ii) transfer learning, iii) federated learning, iv) training with compressed models/dropout, etc. We propose two generic new algorithms which are applicable in all these settings and prove using only an assumption on the Hessian similarity between the target and side information that we can benefit from this framework.
    

