# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Customizing Visual-Language Foundation Models for Multi-modal Anomaly Detection and Reasoning](https://arxiv.org/abs/2403.11083) | 该研究旨在开发一种适用于多种场景的通用异常检测模型，通过定制视觉-语言基础模型和引入多模态提示策略进行多模态异常检测和推理。 |
| [^2] | [Don't Half-listen: Capturing Key-part Information in Continual Instruction Tuning](https://arxiv.org/abs/2403.10056) | 提出了一种基于关键部分信息增益的新型连续指导调整方法，通过动态重放数据和优化训练目标，使LLMs能够捕捉任务感知信息和减轻过度拟合。 |

# 详细

[^1]: 为多模态异常检测和推理定制视觉-语言基础模型

    Customizing Visual-Language Foundation Models for Multi-modal Anomaly Detection and Reasoning

    [https://arxiv.org/abs/2403.11083](https://arxiv.org/abs/2403.11083)

    该研究旨在开发一种适用于多种场景的通用异常检测模型，通过定制视觉-语言基础模型和引入多模态提示策略进行多模态异常检测和推理。

    

    异常检测在各种工业场景中十分重要，包括生产线上异常模式的识别和用于质量控制的制造缺陷检测。本研究旨在开发一种适用于多种场景的通用异常检测模型。为实现这一目标，我们将拥有广泛知识和强大推理能力的通用视觉-语言基础模型定制为异常检测器和推理器。具体来说，我们引入了一种多模态提示策略，将领域专家的领域知识作为条件引导模型。我们的方法考虑多模态提示类型，包括任务描述、类别上下文、正常规则和参考图像。另外，我们将多模态输入表示统一为2D图像格式，使其能够

    arXiv:2403.11083v1 Announce Type: cross  Abstract: Anomaly detection is vital in various industrial scenarios, including the identification of unusual patterns in production lines and the detection of manufacturing defects for quality control. Existing techniques tend to be specialized in individual scenarios and lack generalization capacities. In this study, we aim to develop a generic anomaly detection model applicable across multiple scenarios. To achieve this, we customize generic visual-language foundation models that possess extensive knowledge and robust reasoning abilities into anomaly detectors and reasoners. Specifically, we introduce a multi-modal prompting strategy that incorporates domain knowledge from experts as conditions to guide the models. Our approach considers multi-modal prompt types, including task descriptions, class context, normality rules, and reference images. In addition, we unify the input representation of multi-modality into a 2D image format, enabling m
    
[^2]: 不要半心半意：捕捉连续指导调整中的关键部分信息

    Don't Half-listen: Capturing Key-part Information in Continual Instruction Tuning

    [https://arxiv.org/abs/2403.10056](https://arxiv.org/abs/2403.10056)

    提出了一种基于关键部分信息增益的新型连续指导调整方法，通过动态重放数据和优化训练目标，使LLMs能够捕捉任务感知信息和减轻过度拟合。

    

    arXiv:2403.10056v1 公告类型: 跨领域 摘要：大型语言模型（LLMs）的指导调整可以驱使它们在特定下游任务中产生符合人类目标的结果。然而，LLMs的连续指导调整（CIT）过程可能会带来灾难性遗忘（CF）问题，导致先前学到的能力退化。最近的方法尝试通过修改模型或重放数据来缓解CF问题，但这可能只记住指令的表面模式并在留存任务上感到困惑。在本文中，我们提出了一种基于关键部分信息增益（KPIG）的新型连续指导调整方法。我们的方法计算掩盖部分的信息增益，动态重放数据并优化训练目标，从而使LLMs能够捕捉与正确响应相关的任务感知信息，并减轻对指导中通用描述的过度拟合。此外，我们提出了两个指标，P分和V分，

    arXiv:2403.10056v1 Announce Type: cross  Abstract: Instruction tuning for large language models (LLMs) can drive them to produce results consistent with human goals in specific downstream tasks. However, the process of continual instruction tuning (CIT) for LLMs may bring about the catastrophic forgetting (CF) problem, where previously learned abilities are degraded. Recent methods try to alleviate the CF problem by modifying models or replaying data, which may only remember the surface-level pattern of instructions and get confused on held-out tasks. In this paper, we propose a novel continual instruction tuning method based on Key-part Information Gain (KPIG). Our method computes the information gain on masked parts to dynamically replay data and refine the training objective, which enables LLMs to capture task-aware information relevant to the correct response and alleviate overfitting to general descriptions in instructions. In addition, we propose two metrics, P-score and V-score,
    

