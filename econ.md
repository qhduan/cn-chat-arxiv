# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Large sample properties of GMM estimators under second-order identification.](http://arxiv.org/abs/2307.13475) | 本文提出了GMM估计器在二阶识别条件下的大样本性质，证明了估计器的收敛速率并给出了极限分布，但需要满足过识别条件。 |

# 详细

[^1]: GMM估计器在二阶识别下的大样本性质

    Large sample properties of GMM estimators under second-order identification. (arXiv:2307.13475v1 [econ.EM])

    [http://arxiv.org/abs/2307.13475](http://arxiv.org/abs/2307.13475)

    本文提出了GMM估计器在二阶识别条件下的大样本性质，证明了估计器的收敛速率并给出了极限分布，但需要满足过识别条件。

    

    本文翻译了Dovonon和Hall（2018）在全局识别参数向量{\phi}的p维情形中，当一阶识别条件失败但二阶识别条件成立时提出了GMM估计器的极限分布理论。他们假设一阶不识别是由于在真实值{\phi}_{0}处预期的Jacobian矩阵的秩为p-1，即存在秩缺失的情况。通过对模型重新参数化，使得Jacobian矩阵的最后一列为零，他们证明了前p-1个参数的GMM估计收敛速率为T^{-1/2}，剩下的参数{\phi}_{p}的GMM估计收敛速率为T^{-1/4}。他们还给出了T^{1/4}({\phi}_{p}-{\phi}_{0,p})的极限分布，但需要满足一个（不透明）条件，他们声称这个条件通常并不具限制性。然而，正如我们在本文中所展示的，他们的条件实际上只有在{\phi}过识别的情况下才满足。

    Dovonon and Hall (Journal of Econometrics, 2018) proposed a limiting distribution theory for GMM estimators for a p - dimensional globally identified parameter vector {\phi} when local identification conditions fail at first-order but hold at second-order. They assumed that the first-order underidentification is due to the expected Jacobian having rank p-1 at the true value {\phi}_{0}, i.e., having a rank deficiency of one. After reparametrizing the model such that the last column of the Jacobian vanishes, they showed that the GMM estimator of the first p-1 parameters converges at rate T^{-1/2} and the GMM estimator of the remaining parameter, {\phi}_{p}, converges at rate T^{-1/4}. They also provided a limiting distribution of T^{1/4}({\phi}_{p}-{\phi}_{0,p}) subject to a (non-transparent) condition which they claimed to be not restrictive in general. However, as we show in this paper, their condition is in fact only satisfied when {\phi} is overidentified and the limiting distributio
    

