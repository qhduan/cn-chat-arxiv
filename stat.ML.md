# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Neurosymbolic AI for Reasoning over Knowledge Graphs: A Survey](https://arxiv.org/abs/2302.07200) | 这项综述介绍了神经符号人工智能在知识图谱推理方面的研究。研究表明，最近的方法试图将符号推理和深度学习相结合，以生成具有解释性、竞争性能力并集成专家知识的模型。 |
| [^2] | [Querying Easily Flip-flopped Samples for Deep Active Learning.](http://arxiv.org/abs/2401.09787) | 本文提出了一种基于模型的预测不确定性度量，即最小不一致度量（LDM），用于解决复杂决策边界情况下的主动学习问题。通过查询具有最小LDM的未标记数据，可以提高深度学习模型的性能。 |
| [^3] | [Stable Estimation of Survival Causal Effects.](http://arxiv.org/abs/2310.02278) | 这篇论文研究了稳定估计生存因果效应的问题，研究表明传统的估计方法存在偏差，而近期非倾斜机器学习方法尽管理论上有吸引力，但在生存问题中存在不稳定性。 |
| [^4] | [Uniform Pessimistic Risk and Optimal Portfolio.](http://arxiv.org/abs/2303.07158) | 本文提出了一种称为统一悲观风险的综合$\alpha$-风险版本和基于风险获得最优组合的计算算法，该方法可以用于估计韩国股票的悲观最优组合模型。 |

# 详细

[^1]: 知识图谱推理的神经符号人工智能：一项综述

    Neurosymbolic AI for Reasoning over Knowledge Graphs: A Survey

    [https://arxiv.org/abs/2302.07200](https://arxiv.org/abs/2302.07200)

    这项综述介绍了神经符号人工智能在知识图谱推理方面的研究。研究表明，最近的方法试图将符号推理和深度学习相结合，以生成具有解释性、竞争性能力并集成专家知识的模型。

    

    神经符号人工智能是一个日益活跃的研究领域，它将符号推理方法与深度学习相结合，以利用它们的互补优势。随着知识图谱成为表示异构和多关系数据的一种流行方式，对图结构进行推理的方法开始遵循这种神经符号范式。传统上，这些方法要么利用基于规则的推理，要么生成代表性的数值嵌入，从中可以提取出模式。然而，最近的一些研究尝试弥合这种二元对立，提出了能够促进可解释性、保持竞争性能力并集成专家知识的模型。因此，我们调查了在知识图谱上执行神经符号推理任务的方法，并提出了一种新的分类法。具体而言，我们提出了三个主要类别：（1）逻辑信息嵌入方法，（2）基于嵌入的方法与逻辑一致的方法

    Neurosymbolic AI is an increasingly active area of research that combines symbolic reasoning methods with deep learning to leverage their complementary benefits. As knowledge graphs are becoming a popular way to represent heterogeneous and multi-relational data, methods for reasoning on graph structures have attempted to follow this neurosymbolic paradigm. Traditionally, such approaches have utilized either rule-based inference or generated representative numerical embeddings from which patterns could be extracted. However, several recent studies have attempted to bridge this dichotomy to generate models that facilitate interpretability, maintain competitive performance, and integrate expert knowledge. Therefore, we survey methods that perform neurosymbolic reasoning tasks on knowledge graphs and propose a novel taxonomy by which we can classify them. Specifically, we propose three major categories: (1) logically-informed embedding approaches, (2) embedding approaches with logical cons
    
[^2]: 查询易于翻转样本的深度主动学习

    Querying Easily Flip-flopped Samples for Deep Active Learning. (arXiv:2401.09787v1 [cs.LG])

    [http://arxiv.org/abs/2401.09787](http://arxiv.org/abs/2401.09787)

    本文提出了一种基于模型的预测不确定性度量，即最小不一致度量（LDM），用于解决复杂决策边界情况下的主动学习问题。通过查询具有最小LDM的未标记数据，可以提高深度学习模型的性能。

    

    主动学习是一种机器学习范式，旨在通过选择和查询未标记数据来提高模型的性能。一种有效的选择策略是基于模型的预测不确定性，这可以解释为样本的信息量度量。样本到决策边界的距离是一种自然的预测不确定性度量，但通常难以计算，特别是对于多类分类任务中形成的复杂决策边界。为了解决这个问题，本文提出了“最小不一致度量”（LDM），定义为预测标签不一致的最小概率，并且证明了LDM的估计器在温和假设下是渐近一致的。该估计器计算效率高，并且可以通过参数扰动轻松实现在深度学习模型中使用。基于LDM的主动学习通过查询具有最小LDM的未标记数据来执行。

    Active learning is a machine learning paradigm that aims to improve the performance of a model by strategically selecting and querying unlabeled data. One effective selection strategy is to base it on the model's predictive uncertainty, which can be interpreted as a measure of how informative a sample is. The sample's distance to the decision boundary is a natural measure of predictive uncertainty, but it is often intractable to compute, especially for complex decision boundaries formed in multiclass classification tasks. To address this issue, this paper proposes the {\it least disagree metric} (LDM), defined as the smallest probability of disagreement of the predicted label, and an estimator for LDM proven to be asymptotically consistent under mild assumptions. The estimator is computationally efficient and can be easily implemented for deep learning models using parameter perturbation. The LDM-based active learning is performed by querying unlabeled data with the smallest LDM. Exper
    
[^3]: 稳定估计生存因果效应

    Stable Estimation of Survival Causal Effects. (arXiv:2310.02278v1 [stat.ME])

    [http://arxiv.org/abs/2310.02278](http://arxiv.org/abs/2310.02278)

    这篇论文研究了稳定估计生存因果效应的问题，研究表明传统的估计方法存在偏差，而近期非倾斜机器学习方法尽管理论上有吸引力，但在生存问题中存在不稳定性。

    

    本文研究了估计生存因果效应的问题，即表征干预对生存时间的影响，例如药物是否缩短了入住ICU的时间或广告活动是否增加了顾客停留时间。 过去最流行的估计方法是基于参数化或半参数化模型（例如比例风险模型），然而，这些方法存在显著的偏差问题。最近，非倾斜机器学习方法在大数据集应用中变得越来越流行。然而，尽管这些估计方法具有吸引人的理论性质，但它们往往是不稳定的，因为非倾斜化步骤涉及使用小估计概率的倒数-估计概率的小误差可能导致倒数和结果估计量发生巨大变化。在生存问题中，这个问题更加突出。

    We study the problem of estimating survival causal effects, where the aim is to characterize the impact of an intervention on survival times, i.e., how long it takes for an event to occur. Applications include determining if a drug reduces the time to ICU discharge or if an advertising campaign increases customer dwell time. Historically, the most popular estimates have been based on parametric or semiparametric (e.g. proportional hazards) models; however, these methods suffer from problematic levels of bias. Recently debiased machine learning approaches are becoming increasingly popular, especially in applications to large datasets. However, despite their appealing theoretical properties, these estimators tend to be unstable because the debiasing step involves the use of the inverses of small estimated probabilities -- small errors in the estimated probabilities can result in huge changes in their inverses and therefore the resulting estimator. This problem is exacerbated in survival 
    
[^4]: 统一悲观风险和最优组合

    Uniform Pessimistic Risk and Optimal Portfolio. (arXiv:2303.07158v1 [q-fin.PM])

    [http://arxiv.org/abs/2303.07158](http://arxiv.org/abs/2303.07158)

    本文提出了一种称为统一悲观风险的综合$\alpha$-风险版本和基于风险获得最优组合的计算算法，该方法可以用于估计韩国股票的悲观最优组合模型。

    This paper proposes a version of integrated $\alpha$-risk called the uniform pessimistic risk and a computational algorithm to obtain an optimal portfolio based on the risk. The proposed method can be used to estimate the pessimistic optimal portfolio models for Korean stocks.

    资产配置的最优性已经在风险度量的理论分析中得到广泛讨论。悲观主义是一种超越传统最优组合模型的最有吸引力的方法之一，$\alpha$-风险在推导出广泛的悲观最优组合中起着关键作用。然而，由悲观风险评估的最优组合的估计仍然具有挑战性，因为缺乏可用的估计模型和计算算法。在本研究中，我们提出了一种称为统一悲观风险的综合$\alpha$-风险版本和基于风险获得最优组合的计算算法。此外，我们从多个分位数回归、适当的评分规则和分布鲁棒优化三个不同的方法来研究所提出的风险的理论性质。同时，统一悲观风险被应用于估计韩国股票的悲观最优组合模型。

    The optimality of allocating assets has been widely discussed with the theoretical analysis of risk measures. Pessimism is one of the most attractive approaches beyond the conventional optimal portfolio model, and the $\alpha$-risk plays a crucial role in deriving a broad class of pessimistic optimal portfolios. However, estimating an optimal portfolio assessed by a pessimistic risk is still challenging due to the absence of an available estimation model and a computational algorithm. In this study, we propose a version of integrated $\alpha$-risk called the uniform pessimistic risk and the computational algorithm to obtain an optimal portfolio based on the risk. Further, we investigate the theoretical properties of the proposed risk in view of three different approaches: multiple quantile regression, the proper scoring rule, and distributionally robust optimization. Also, the uniform pessimistic risk is applied to estimate the pessimistic optimal portfolio models for the Korean stock 
    

