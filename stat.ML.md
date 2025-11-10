# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Inference with Mondrian Random Forests.](http://arxiv.org/abs/2310.09702) | 本文在回归设置下给出了Mondrian随机森林的估计中心极限定理和去偏过程，使其能够进行统计推断和实现最小极大估计速率。 |
| [^2] | [Externally Valid Policy Choice.](http://arxiv.org/abs/2205.05561) | 本文研究了外部有效的个性化治疗策略的学习问题。研究表明，最大化福利的治疗策略对于实验和目标人群之间的结果分布变化具有鲁棒性。通过开发新的方法，作者提出了对结果和特征变化具有鲁棒性的策略学习方法，并强调实验人群内的治疗效果异质性对策略普适性的影响。 |

# 详细

[^1]: 带有Mondrian随机森林的推理

    Inference with Mondrian Random Forests. (arXiv:2310.09702v1 [math.ST])

    [http://arxiv.org/abs/2310.09702](http://arxiv.org/abs/2310.09702)

    本文在回归设置下给出了Mondrian随机森林的估计中心极限定理和去偏过程，使其能够进行统计推断和实现最小极大估计速率。

    

    随机森林是一种常用的分类和回归方法，在最近几年中提出了许多不同的变体。一个有趣的例子是Mondrian随机森林，其中底层树是根据Mondrian过程构建的。在本文中，我们给出了Mondrian随机森林在回归设置下的估计的中心极限定理。当与偏差表征和一致方差估计器相结合时，这允许进行渐近有效的统计推断，如构建置信区间，对未知的回归函数进行推断。我们还提供了一种去偏过程，用于Mondrian随机森林，使其能够在适当的参数调整下实现$\beta$-H\"older回归函数的最小极大估计速率，对于所有的$\beta$和任意维度。

    Random forests are popular methods for classification and regression, and many different variants have been proposed in recent years. One interesting example is the Mondrian random forest, in which the underlying trees are constructed according to a Mondrian process. In this paper we give a central limit theorem for the estimates made by a Mondrian random forest in the regression setting. When combined with a bias characterization and a consistent variance estimator, this allows one to perform asymptotically valid statistical inference, such as constructing confidence intervals, on the unknown regression function. We also provide a debiasing procedure for Mondrian random forests which allows them to achieve minimax-optimal estimation rates with $\beta$-H\"older regression functions, for all $\beta$ and in arbitrary dimension, assuming appropriate parameter tuning.
    
[^2]: 外部有效的策略选择

    Externally Valid Policy Choice. (arXiv:2205.05561v2 [econ.EM] UPDATED)

    [http://arxiv.org/abs/2205.05561](http://arxiv.org/abs/2205.05561)

    本文研究了外部有效的个性化治疗策略的学习问题。研究表明，最大化福利的治疗策略对于实验和目标人群之间的结果分布变化具有鲁棒性。通过开发新的方法，作者提出了对结果和特征变化具有鲁棒性的策略学习方法，并强调实验人群内的治疗效果异质性对策略普适性的影响。

    

    我们考虑学习个性化治疗策略的问题，这些策略是外部有效或广义化的：它们在除了实验（或训练）人群外的其他目标人群中表现良好。我们首先证明，对于实验人群而言，最大化福利的策略对于实验和目标人群之间的结果（但不是特征）分布变化具有鲁棒性。然后，我们开发了新的方法来学习对结果和特征变化具有鲁棒性的策略。在这样做时，我们强调了实验人群内的治疗效果异质性如何影响策略的普适性。我们的方法可以使用实验或观察数据（其中治疗是内生的）。我们的许多方法可以使用线性规划实现。

    We consider the problem of learning personalized treatment policies that are externally valid or generalizable: they perform well in other target populations besides the experimental (or training) population from which data are sampled. We first show that welfare-maximizing policies for the experimental population are robust to shifts in the distribution of outcomes (but not characteristics) between the experimental and target populations. We then develop new methods for learning policies that are robust to shifts in outcomes and characteristics. In doing so, we highlight how treatment effect heterogeneity within the experimental population affects the generalizability of policies. Our methods may be used with experimental or observational data (where treatment is endogenous). Many of our methods can be implemented with linear programming.
    

