# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Extremal graphical modeling with latent variables](https://arxiv.org/abs/2403.09604) | 提出了一种针对混合变量的极端图模型的学习方法，能够有效恢复条件图和潜变量数量。 |
| [^2] | [Evaluating AI systems under uncertain ground truth: a case study in dermatology.](http://arxiv.org/abs/2307.02191) | 这项研究总结了在健康领域中评估AI系统时的一个重要问题：基准事实的不确定性。现有的方法通常忽视了这一点，而该研究提出了一种使用统计模型聚合注释的框架，以更准确地评估AI系统的性能。 |
| [^3] | [Convergence of Momentum-Based Heavy Ball Method with Batch Updating and/or Approximate Gradients.](http://arxiv.org/abs/2303.16241) | 本文研究了含有批量更新和/或近似梯度的动量重球法的收敛性，由于采用了简化的梯度计算方法，大大减少了计算消耗，同时仍能保证收敛性。 |
| [^4] | [Understanding Best Subset Selection: A Tale of Two C(omplex)ities.](http://arxiv.org/abs/2301.06259) | 本文研究了最佳子集选择在高维稀疏线性回归设置中的变量选择性质，通过研究残差化特征和虚假投影的复杂性来揭示模型一致性的边界条件。 |

# 详细

[^1]: 混合变量的极端图模型

    Extremal graphical modeling with latent variables

    [https://arxiv.org/abs/2403.09604](https://arxiv.org/abs/2403.09604)

    提出了一种针对混合变量的极端图模型的学习方法，能够有效恢复条件图和潜变量数量。

    

    极端图模型编码多变量极端条件独立结构，并为量化罕见事件风险提供强大工具。我们提出了面向潜变量的可延伸图模型的可行凸规划方法，将 H\"usler-Reiss 精度矩阵分解为编码观察变量之间的图结构的稀疏部分和编码少量潜变量对观察变量的影响的低秩部分。我们提供了\texttt{eglatent}的有限样本保证，并展示它能一致地恢复条件图以及潜变量的数量。

    arXiv:2403.09604v1 Announce Type: cross  Abstract: Extremal graphical models encode the conditional independence structure of multivariate extremes and provide a powerful tool for quantifying the risk of rare events. Prior work on learning these graphs from data has focused on the setting where all relevant variables are observed. For the popular class of H\"usler-Reiss models, we propose the \texttt{eglatent} method, a tractable convex program for learning extremal graphical models in the presence of latent variables. Our approach decomposes the H\"usler-Reiss precision matrix into a sparse component encoding the graphical structure among the observed variables after conditioning on the latent variables, and a low-rank component encoding the effect of a few latent variables on the observed variables. We provide finite-sample guarantees of \texttt{eglatent} and show that it consistently recovers the conditional graph as well as the number of latent variables. We highlight the improved 
    
[^2]: 在不确定的基准事实下评估AI系统：皮肤病例研究

    Evaluating AI systems under uncertain ground truth: a case study in dermatology. (arXiv:2307.02191v1 [cs.LG])

    [http://arxiv.org/abs/2307.02191](http://arxiv.org/abs/2307.02191)

    这项研究总结了在健康领域中评估AI系统时的一个重要问题：基准事实的不确定性。现有的方法通常忽视了这一点，而该研究提出了一种使用统计模型聚合注释的框架，以更准确地评估AI系统的性能。

    

    为了安全起见，在部署之前，卫生领域的AI系统需要经过全面的评估，将其预测结果与假定为确定的基准事实进行验证。然而，实际情况并非如此，基准事实可能是不确定的。不幸的是，在标准的AI模型评估中，这一点被大部分忽视了，但是它可能会产生严重后果，如高估未来的性能。为了避免这种情况，我们测量了基准事实的不确定性，我们假设它可以分解为两个主要部分：注释不确定性是由于缺乏可靠注释，以及由于有限的观测信息而导致的固有不确定性。在确定地聚合注释时，通常会忽视这种基准事实的不确定性，例如通过多数投票或平均值来聚合。相反，我们提出了一个框架，在该框架中使用统计模型进行注释的聚合。具体而言，我们将注释的聚合框架解释为所谓可能性的后验推断。

    For safety, AI systems in health undergo thorough evaluations before deployment, validating their predictions against a ground truth that is assumed certain. However, this is actually not the case and the ground truth may be uncertain. Unfortunately, this is largely ignored in standard evaluation of AI models but can have severe consequences such as overestimating the future performance. To avoid this, we measure the effects of ground truth uncertainty, which we assume decomposes into two main components: annotation uncertainty which stems from the lack of reliable annotations, and inherent uncertainty due to limited observational information. This ground truth uncertainty is ignored when estimating the ground truth by deterministically aggregating annotations, e.g., by majority voting or averaging. In contrast, we propose a framework where aggregation is done using a statistical model. Specifically, we frame aggregation of annotations as posterior inference of so-called plausibilities
    
[^3]: 采用批量更新和/或近似梯度的动量重球法的收敛性

    Convergence of Momentum-Based Heavy Ball Method with Batch Updating and/or Approximate Gradients. (arXiv:2303.16241v1 [math.OC])

    [http://arxiv.org/abs/2303.16241](http://arxiv.org/abs/2303.16241)

    本文研究了含有批量更新和/或近似梯度的动量重球法的收敛性，由于采用了简化的梯度计算方法，大大减少了计算消耗，同时仍能保证收敛性。

    

    本文研究了1964年Polyak引入的凸优化和非凸优化中广为人知的“动量重球”法，并在多种情况下确立了其收敛性。当要求解参数的维度非常高时，更新一部分而不是所有参数可以提高优化效率，称之为“批量更新”，若与梯度法配合使用，则理论上只需计算需要更新的参数的梯度，而在实际中，通过反向传播等方法仅计算部分梯度并不能减少计算量。因此，为了在每一步中减少CPU使用量，可以使用一阶微分或近似梯度代替真实梯度。我们的分析表明，在各种假设下，采用近似梯度信息和/或批量更新的动量重球法仍然可以收敛。

    In this paper, we study the well-known "Heavy Ball" method for convex and nonconvex optimization introduced by Polyak in 1964, and establish its convergence under a variety of situations. Traditionally, most algorthms use "full-coordinate update," that is, at each step, very component of the argument is updated. However, when the dimension of the argument is very high, it is more efficient to update some but not all components of the argument at each iteration. We refer to this as "batch updating" in this paper.  When gradient-based algorithms are used together with batch updating, in principle it is sufficient to compute only those components of the gradient for which the argument is to be updated. However, if a method such as back propagation is used to compute these components, computing only some components of gradient does not offer much savings over computing the entire gradient. Therefore, to achieve a noticeable reduction in CPU usage at each step, one can use first-order diffe
    
[^4]: 了解最佳子集选择: 两种复杂性的故事

    Understanding Best Subset Selection: A Tale of Two C(omplex)ities. (arXiv:2301.06259v2 [math.ST] UPDATED)

    [http://arxiv.org/abs/2301.06259](http://arxiv.org/abs/2301.06259)

    本文研究了最佳子集选择在高维稀疏线性回归设置中的变量选择性质，通过研究残差化特征和虚假投影的复杂性来揭示模型一致性的边界条件。

    

    几十年来，最佳子集选择(BSS)主要由于计算瓶颈而困扰统计学家。然而，直到最近，现代计算突破重新点燃了对BSS的理论兴趣并导致了新的发现。最近，Guo等人表明，BSS的模型选择性能受到了鲁棒性设计依赖的边界量的控制，不像LASSO、SCAD、MCP等现代方法。在他们的理论结果的激励下，本文还研究了高维稀疏线性回归设置下最佳子集选择的变量选择性质。我们展示了除了可辨识性边界以外，下列两种复杂性度量在表征模型一致性边界条件中起着基本的作用：(a)“残差化特征”的复杂性，(b)“虚假投影”的复杂性。特别地，我们建立了一个仅依赖于可辨识性边界的简单边界条件。

    For decades, best subset selection (BSS) has eluded statisticians mainly due to its computational bottleneck. However, until recently, modern computational breakthroughs have rekindled theoretical interest in BSS and have led to new findings. Recently, \cite{guo2020best} showed that the model selection performance of BSS is governed by a margin quantity that is robust to the design dependence, unlike modern methods such as LASSO, SCAD, MCP, etc. Motivated by their theoretical results, in this paper, we also study the variable selection properties of best subset selection for high-dimensional sparse linear regression setup. We show that apart from the identifiability margin, the following two complexity measures play a fundamental role in characterizing the margin condition for model consistency: (a) complexity of \emph{residualized features}, (b) complexity of \emph{spurious projections}. In particular, we establish a simple margin condition that depends only on the identifiability mar
    

