# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [LMM-Assisted Breast Cancer Treatment Target Segmentation with Consistency Embedding](https://arxiv.org/abs/2311.15876) | RO-LMM是一个针对放射肿瘤学领域设计的多功能大型多模型，提出了一种Consistency Embedding Fine-Tuning（CEFTune）技术，使其能够在保持处理干净输入能力的同时提升对嘈杂输入的鲁棒性，用于放射治疗计划和目标体积分割。 |
| [^2] | [Beyond RMSE and MAE: Introducing EAUC to unmask hidden bias and unfairness in dyadic regression models.](http://arxiv.org/abs/2401.10690) | 本研究引入EAUC作为一种新的度量标准，用以揭示迪亚德回归模型中隐藏的偏见和不公平问题。传统的全局错误度量标准如RMSE和MAE无法捕捉到这种问题。 |
| [^3] | [A Simple Way to Incorporate Novelty Detection in World Models.](http://arxiv.org/abs/2310.08731) | 本文提出了一个简单的方法，通过利用世界模型幻觉状态和真实观察状态的不匹配性作为异常分数，将新颖性检测纳入世界模型强化学习代理中。在新环境中与传统方法相比，我们的工作具有优势。 |
| [^4] | [Q-based Equilibria.](http://arxiv.org/abs/2304.12647) | 该论文研究了基于Q的策略规则族中的均衡偏差（或 Qb-equilibria），即Q值在不同监测技术下的效果。 |

# 详细

[^1]: LMM辅助的一致性嵌入下乳腺癌治疗目标分割

    LMM-Assisted Breast Cancer Treatment Target Segmentation with Consistency Embedding

    [https://arxiv.org/abs/2311.15876](https://arxiv.org/abs/2311.15876)

    RO-LMM是一个针对放射肿瘤学领域设计的多功能大型多模型，提出了一种Consistency Embedding Fine-Tuning（CEFTune）技术，使其能够在保持处理干净输入能力的同时提升对嘈杂输入的鲁棒性，用于放射治疗计划和目标体积分割。

    

    人工智能的最新进展深刻影响了医学领域，为降低临床工作量提供了工具。然而，大多数人工智能模型受限于执行单模式任务，与医学专业人员所使用的综合方法形成鲜明对比。为解决这一问题，本文介绍了RO-LMM，一个专为放射肿瘤学领域设计的多功能大型多模型（LMM）。该模型涵盖了临床工作流中的一系列任务，擅长临床报告摘要、放疗治疗计划建议和计划引导的目标体积分割。为了执行连续的临床任务，我们进一步提出了一种新颖的一致性嵌入微调（CEFTune）技术，提升了LMM对嘈杂输入的鲁棒性，同时保持了处理干净输入的能力，并将该概念转化为LMM驱动的分割框架，即一致性嵌入S。

    arXiv:2311.15876v2 Announce Type: replace-cross  Abstract: Recent advancements in Artificial Intelligence (AI) have profoundly influenced medical fields, by providing tools to reduce clinical workloads. However, most AI models are constrained to execute unimodal tasks, in stark contrast to the comprehensive approaches utilized by medical professionals. To address this, here we present RO-LMM, a multi-purpose large multimodal model (LMM) tailored for the field of radiation oncology. This model covers series of tasks within clinical workflow, adept at clinical report summarization, radiation treatment plan suggestion, and plan-guided target volume segmentation. In particular, to perform consecutive clinical tasks, we further present a novel Consistency Embedding Fine-Tuning (CEFTune) technique, which boosts LMM's robustness to noisy inputs while preserving the capability of handling clean inputs, and transform this concept into LMM-driven segmentation framework as Consistency Embedding S
    
[^2]: 超越RMSE和MAE：引入EAUC来揭示偏见和不公平的迪亚德回归模型中的隐藏因素

    Beyond RMSE and MAE: Introducing EAUC to unmask hidden bias and unfairness in dyadic regression models. (arXiv:2401.10690v1 [cs.LG])

    [http://arxiv.org/abs/2401.10690](http://arxiv.org/abs/2401.10690)

    本研究引入EAUC作为一种新的度量标准，用以揭示迪亚德回归模型中隐藏的偏见和不公平问题。传统的全局错误度量标准如RMSE和MAE无法捕捉到这种问题。

    

    迪亚德回归模型用于预测一对实体的实值结果，在许多领域中都是基础的（例如，在推荐系统中预测用户对产品的评分），在许多其他领域中也有许多潜力但尚未深入探索（例如，在个性化药理学中近似确定患者的适当剂量）。本研究中，我们证明个体实体观察值分布的非均匀性导致了最先进模型中的严重偏见预测，偏向于实体的观察过去值的平均值，并在另类但同样重要的情况下提供比随机预测更差的预测能力。我们表明，全局错误度量标准如均方根误差（RMSE）和平均绝对误差（MAE）不足以捕捉到这种现象，我们将其命名为另类偏见，并引入另类-曲线下面积（EAUC）作为一个新的补充度量，可以在所有研究的模型中量化它。

    Dyadic regression models, which predict real-valued outcomes for pairs of entities, are fundamental in many domains (e.g. predicting the rating of a user to a product in Recommender Systems) and promising and under exploration in many others (e.g. approximating the adequate dosage of a drug for a patient in personalized pharmacology). In this work, we demonstrate that non-uniformity in the observed value distributions of individual entities leads to severely biased predictions in state-of-the-art models, skewing predictions towards the average of observed past values for the entity and providing worse-than-random predictive power in eccentric yet equally important cases. We show that the usage of global error metrics like Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE) is insufficient to capture this phenomenon, which we name eccentricity bias, and we introduce Eccentricity-Area Under the Curve (EAUC) as a new complementary metric that can quantify it in all studied models
    
[^3]: 一个简单的方法在世界模型中实现新颖性检测

    A Simple Way to Incorporate Novelty Detection in World Models. (arXiv:2310.08731v1 [cs.AI])

    [http://arxiv.org/abs/2310.08731](http://arxiv.org/abs/2310.08731)

    本文提出了一个简单的方法，通过利用世界模型幻觉状态和真实观察状态的不匹配性作为异常分数，将新颖性检测纳入世界模型强化学习代理中。在新环境中与传统方法相比，我们的工作具有优势。

    

    使用世界模型进行强化学习已经取得了显著的成功。然而，当世界机制或属性发生突然变化时，代理的性能和可靠性可能会显著下降。我们将视觉属性或状态转换的突变称为“新颖性”。在生成的世界模型框架中实施新颖性检测是保护部署时代理的关键任务。在本文中，我们提出了一种简单的边界方法，用于将新颖性检测纳入世界模型强化学习代理中，通过利用世界模型幻觉状态和真实观察状态的不匹配性作为异常分数。首先，我们提供了与序列决策相关的新颖性检测本体论，然后我们提供了在代理在世界模型中学习的转换分布中检测新颖性的有效方法。最后，我们展示了我们的工作在新环境中与传统方法相比的优势。

    Reinforcement learning (RL) using world models has found significant recent successes. However, when a sudden change to world mechanics or properties occurs then agent performance and reliability can dramatically decline. We refer to the sudden change in visual properties or state transitions as {\em novelties}. Implementing novelty detection within generated world model frameworks is a crucial task for protecting the agent when deployed. In this paper, we propose straightforward bounding approaches to incorporate novelty detection into world model RL agents, by utilizing the misalignment of the world model's hallucinated states and the true observed states as an anomaly score. We first provide an ontology of novelty detection relevant to sequential decision making, then we provide effective approaches to detecting novelties in a distribution of transitions learned by an agent in a world model. Finally, we show the advantage of our work in a novel environment compared to traditional ma
    
[^4]: 基于Q的均衡

    Q-based Equilibria. (arXiv:2304.12647v1 [econ.TH])

    [http://arxiv.org/abs/2304.12647](http://arxiv.org/abs/2304.12647)

    该论文研究了基于Q的策略规则族中的均衡偏差（或 Qb-equilibria），即Q值在不同监测技术下的效果。

    

    在动态环境中，Q学习是一种自适应规则，其为每个替代方案提供估计值(即Q值)，该值与之前的决策相关。一个朴素的策略是始终选择具有最高Q值的替代方案。我们考虑一族基于Q的策略规则，这些规则可能系统地支持某些替代方案而不是其他替代方案，例如包含有利合作的宽容偏差的规则。在 Compte 和 Postlewaite [2018] 的精神下，我们在这个 Q-based 规则族中寻找均衡偏差（或 Qb-equilibria）。我们研究了不同监测技术下的经典博弈。

    In dynamic environments, Q-learning is an adaptative rule that provides an estimate (a Q-value) of the continuation value associated with each alternative. A naive policy consists in always choosing the alternative with highest Q-value. We consider a family of Q-based policy rules that may systematically favor some alternatives over others, for example rules that incorporate a leniency bias that favors cooperation. In the spirit of Compte and Postlewaite [2018], we look for equilibrium biases (or Qb-equilibria) within this family of Q-based rules. We examine classic games under various monitoring technologies.
    

