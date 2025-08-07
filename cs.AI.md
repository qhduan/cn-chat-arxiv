# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Long-term Frame-Event Visual Tracking: Benchmark Dataset and Baseline](https://arxiv.org/abs/2403.05839) | 提出了一个新的长期和大规模的帧事件单目标跟踪数据集FELT，重新训练和评估了15个基准跟踪器，并引入了关联记忆Transformer网络来解决RGB帧和事件流不完整的问题。 |
| [^2] | [Hulk: A Universal Knowledge Translator for Human-Centric Tasks](https://arxiv.org/abs/2312.01697) | Hulk是第一个多模态人类中心通用模型，能够处理2D视觉、3D视觉、基于骨架和视觉语言任务，无需任务特定微调 |

# 详细

[^1]: 长期帧事件视觉跟踪：基准数据集与基准线

    Long-term Frame-Event Visual Tracking: Benchmark Dataset and Baseline

    [https://arxiv.org/abs/2403.05839](https://arxiv.org/abs/2403.05839)

    提出了一个新的长期和大规模的帧事件单目标跟踪数据集FELT，重新训练和评估了15个基准跟踪器，并引入了关联记忆Transformer网络来解决RGB帧和事件流不完整的问题。

    

    当前基于事件/帧事件的跟踪器在短期跟踪数据集上进行评估，然而，对于真实场景的跟踪涉及长期跟踪，现有跟踪算法在这些场景中的性能仍不清楚。在本文中，我们首先提出了一个新的长期和大规模的帧事件单目标跟踪数据集，名为FELT。它包含742个视频和1,594,474个RGB帧和事件流对，并已成为迄今为止最大的帧事件跟踪数据集。我们重新训练和评估了15个基准跟踪器在我们的数据集上，以供未来研究进行比较。更重要的是，我们发现RGB帧和事件流由于挑战因素的影响和空间稀疏的事件流而自然不完整。针对这一问题，我们提出了一种新颖的关联记忆Transformer网络作为统一骨干，通过将现代Hopfield层引入多头自注意力块来进行处理。

    arXiv:2403.05839v1 Announce Type: cross  Abstract: Current event-/frame-event based trackers undergo evaluation on short-term tracking datasets, however, the tracking of real-world scenarios involves long-term tracking, and the performance of existing tracking algorithms in these scenarios remains unclear. In this paper, we first propose a new long-term and large-scale frame-event single object tracking dataset, termed FELT. It contains 742 videos and 1,594,474 RGB frames and event stream pairs and has become the largest frame-event tracking dataset to date. We re-train and evaluate 15 baseline trackers on our dataset for future works to compare. More importantly, we find that the RGB frames and event streams are naturally incomplete due to the influence of challenging factors and spatially sparse event flow. In response to this, we propose a novel associative memory Transformer network as a unified backbone by introducing modern Hopfield layers into multi-head self-attention blocks to
    
[^2]: Hulk: 一种面向人类中心任务的通用知识翻译器

    Hulk: A Universal Knowledge Translator for Human-Centric Tasks

    [https://arxiv.org/abs/2312.01697](https://arxiv.org/abs/2312.01697)

    Hulk是第一个多模态人类中心通用模型，能够处理2D视觉、3D视觉、基于骨架和视觉语言任务，无需任务特定微调

    

    人类中心感知任务，例如行人检测、基于骨架的动作识别和姿态估计，在诸如元宇宙和体育分析等广泛的工业应用中具有重要意义。近来，出现了发展旨在受益于广泛人类中心感知任务的人类中心基础模型的激增。虽然许多人类中心基础模型取得了成功，但它们没有探索用于人类中心及需要任务特定微调的3D和视觉语言任务。这些限制限制了它们在更多下游任务和情境中的应用。为了解决这些问题，我们提出了Hulk，第一个能够在无需任务特定微调的情况下处理2D视觉、3D视觉、基于骨架和视觉语言任务的多模态人类中心通用模型。实现这一目标的关键在于将各种任务特定头部压缩成两个通用头部，一个用于离散表示，如语言，

    arXiv:2312.01697v4 Announce Type: replace-cross  Abstract: Human-centric perception tasks, e.g., pedestrian detection, skeleton-based action recognition, and pose estimation, have wide industrial applications, such as metaverse and sports analysis. There is a recent surge to develop human-centric foundation models that can benefit a broad range of human-centric perception tasks. While many human-centric foundation models have achieved success, they did not explore 3D and vision-language tasks for human-centric and required task-specific finetuning. These limitations restrict their application to more downstream tasks and situations. To tackle these problems, we present Hulk, the first multimodal human-centric generalist model, capable of addressing 2D vision, 3D vision, skeleton-based, and vision-language tasks without task-specific finetuning. The key to achieving this is condensing various task-specific heads into two general heads, one for discrete representations, e.g., languages, 
    

