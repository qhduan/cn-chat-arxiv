# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Semantically-Shifted Incremental Adapter-Tuning is A Continual ViTransformer](https://arxiv.org/abs/2403.19979) | 适配器调整方法在持续学习中展现出较优性能，提出了增量调整共享适配器和利用存储原型进行特征采样和更新的方法来增强模型学习能力。 |
| [^2] | [ConjNorm: Tractable Density Estimation for Out-of-Distribution Detection](https://arxiv.org/abs/2402.17888) | 提出了一种新颖的理论框架，基于Bregman散度，通过引入共轭约束，提出了一种\textsc{ConjNorm}方法，以在给定数据集中搜索最佳规范系数$p$来重新构想密度函数设计。 |

# 详细

[^1]: 语义转移增量适配器调整是一种持续的 ViTransformer

    Semantically-Shifted Incremental Adapter-Tuning is A Continual ViTransformer

    [https://arxiv.org/abs/2403.19979](https://arxiv.org/abs/2403.19979)

    适配器调整方法在持续学习中展现出较优性能，提出了增量调整共享适配器和利用存储原型进行特征采样和更新的方法来增强模型学习能力。

    

    类增量学习（CIL）旨在使模型能够在克服灾难性遗忘的同时持续学习新的类别。本文重新审视了在持续学习背景下的不同参数高效调整（PET）方法。我们观察到适配器调整表现优于基于提示的方法，甚至在每个学习会话中没有参数扩展的情况下也如此。受此启发，我们提出了增量调整共享适配器而不施加参数更新约束，增强骨干的学习能力。此外，我们从存储的原型中抽取特征样本来重新训练统一的分类器，进一步提高其性能。我们估计旧原型的语义转移，而无法访问过去的样本，并逐个会话更新存储的原型。我们提出的方法消除了模型的扩展和...

    arXiv:2403.19979v1 Announce Type: cross  Abstract: Class-incremental learning (CIL) aims to enable models to continuously learn new classes while overcoming catastrophic forgetting. The introduction of pre-trained models has brought new tuning paradigms to CIL. In this paper, we revisit different parameter-efficient tuning (PET) methods within the context of continual learning. We observe that adapter tuning demonstrates superiority over prompt-based methods, even without parameter expansion in each learning session. Motivated by this, we propose incrementally tuning the shared adapter without imposing parameter update constraints, enhancing the learning capacity of the backbone. Additionally, we employ feature sampling from stored prototypes to retrain a unified classifier, further improving its performance. We estimate the semantic shift of old prototypes without access to past samples and update stored prototypes session by session. Our proposed method eliminates model expansion and
    
[^2]: ConjNorm：用于异常分布检测的可处理密度估计

    ConjNorm: Tractable Density Estimation for Out-of-Distribution Detection

    [https://arxiv.org/abs/2402.17888](https://arxiv.org/abs/2402.17888)

    提出了一种新颖的理论框架，基于Bregman散度，通过引入共轭约束，提出了一种\textsc{ConjNorm}方法，以在给定数据集中搜索最佳规范系数$p$来重新构想密度函数设计。

    

    后续异常分布（OOD）检测在可靠机器学习中受到密切关注。许多工作致力于推导基于logits、距离或严格数据分布假设的评分函数，以识别得分低的OOD样本。然而，这些估计得分可能无法准确反映真实数据密度或施加不切实际的约束。为了在基于密度得分设计方面提供一个统一的视角，我们提出了一个以Bregman散度为基础的新颖理论框架，该框架将分布考虑扩展到涵盖一系列指数族分布。利用我们定理中揭示的共轭约束，我们引入了一种\textsc{ConjNorm}方法，将密度函数设计重新构想为针对给定数据集搜索最佳规范系数$p$的过程。鉴于归一化的计算挑战，我们设计了一种无偏和解析可追踪的方法

    arXiv:2402.17888v1 Announce Type: cross  Abstract: Post-hoc out-of-distribution (OOD) detection has garnered intensive attention in reliable machine learning. Many efforts have been dedicated to deriving score functions based on logits, distances, or rigorous data distribution assumptions to identify low-scoring OOD samples. Nevertheless, these estimate scores may fail to accurately reflect the true data density or impose impractical constraints. To provide a unified perspective on density-based score design, we propose a novel theoretical framework grounded in Bregman divergence, which extends distribution considerations to encompass an exponential family of distributions. Leveraging the conjugation constraint revealed in our theorem, we introduce a \textsc{ConjNorm} method, reframing density function design as a search for the optimal norm coefficient $p$ against the given dataset. In light of the computational challenges of normalization, we devise an unbiased and analytically tract
    

