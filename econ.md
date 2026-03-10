# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Partially identified heteroskedastic SVARs](https://arxiv.org/abs/2403.06879) | 该论文研究了一种利用结构性冲击方差的突变来识别SVAR的方法，提出了冲激响应的已确定集，并展示了计算方法。 |
| [^2] | [Learning from Viral Content.](http://arxiv.org/abs/2210.01267) | 本文研究了社交媒体上的学习，发现向用户展示病毒性故事可以增加信息汇集，但也可能导致大多数分享的故事是错误的稳定状态。这些误导性的稳定状态会自我维持，对平台设计和鲁棒性产生多种后果。 |

# 详细

[^1]: 部分识别的异方差SVARs

    Partially identified heteroskedastic SVARs

    [https://arxiv.org/abs/2403.06879](https://arxiv.org/abs/2403.06879)

    该论文研究了一种利用结构性冲击方差的突变来识别SVAR的方法，提出了冲激响应的已确定集，并展示了计算方法。

    

    本文研究了利用结构性冲击方差的突变来识别结构向量自回归模型（SVARs）。对于这类模型的点识别依赖于涉及约减型误差的协方差矩阵的特征值分解，并要求所有特征值是不同的。然而，在存在多重特征值的情况下，这种点识别会失败。在一个实证相关的场景中出现这种情况，例如只有一个子集的结构性冲击有了方差的突变，或者其中一组变量展示了相同幅度的方差变化。结合对结构参数和冲激响应的零或符号约束，我们得出了冲激响应的已确定集，并展示了如何计算它们。我们基于为已确定集SVARs开发的稳健贝叶斯方法进行了冲激响应函数的推论。

    arXiv:2403.06879v1 Announce Type: new  Abstract: This paper studies the identification of Structural Vector Autoregressions (SVARs) exploiting a break in the variances of the structural shocks. Point-identification for this class of models relies on an eigen-decomposition involving the covariance matrices of reduced-form errors and requires that all the eigenvalues are distinct. This point-identification, however, fails in the presence of multiplicity of eigenvalues. This occurs in an empirically relevant scenario where, for instance, only a subset of structural shocks had the break in their variances, or where a group of variables shows a variance shift of the same amount. Together with zero or sign restrictions on the structural parameters and impulse responses, we derive the identified sets for impulse responses and show how to compute them. We perform inference on the impulse response functions, building on the robust Bayesian approach developed for set identified SVARs. To illustr
    
[^2]: 从病毒性内容中学习

    Learning from Viral Content. (arXiv:2210.01267v2 [econ.TH] UPDATED)

    [http://arxiv.org/abs/2210.01267](http://arxiv.org/abs/2210.01267)

    本文研究了社交媒体上的学习，发现向用户展示病毒性故事可以增加信息汇集，但也可能导致大多数分享的故事是错误的稳定状态。这些误导性的稳定状态会自我维持，对平台设计和鲁棒性产生多种后果。

    

    我们研究了社交媒体上的学习，采用了一个均衡模型来描述用户与共享新闻故事进行交互。理性用户按顺序到达，观察到原始故事（即私有信号）和新闻推送中前辈故事的样本，然后决定分享哪些故事。观察到的故事样本取决于前辈分享的内容以及生成新闻推送的抽样算法。我们重点研究了这个算法如何选择更具病毒性（即被广泛分享）的故事的频率。向用户展示病毒性故事可以增加信息汇集，但也可能产生大多数分享故事错误的稳定状态。这些误导性的稳定状态自我持续，因为观察到错误故事的用户会形成错误的信念，从而理性地继续分享它们。最后，我们描述了平台设计和鲁棒性方面的若干后果。

    We study learning on social media with an equilibrium model of users interacting with shared news stories. Rational users arrive sequentially, observe an original story (i.e., a private signal) and a sample of predecessors' stories in a news feed, and then decide which stories to share. The observed sample of stories depends on what predecessors share as well as the sampling algorithm generating news feeds. We focus on how often this algorithm selects more viral (i.e., widely shared) stories. Showing users viral stories can increase information aggregation, but it can also generate steady states where most shared stories are wrong. These misleading steady states self-perpetuate, as users who observe wrong stories develop wrong beliefs, and thus rationally continue to share them. Finally, we describe several consequences for platform design and robustness.
    

