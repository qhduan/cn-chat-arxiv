# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Collaborative Contextualization: Bridging the Gap between Collaborative Filtering and Pre-trained Language Model.](http://arxiv.org/abs/2310.09400) | 本文介绍了一种名为CollabContext的模型，通过巧妙地将协同过滤信号与情境化表示相结合，同时保留了关键的情境语义，解决了传统推荐系统中协同信号和情境化表示之间的差距。 |
| [^2] | [Dual Feedback Attention Framework via Boundary-Aware Auxiliary and Progressive Semantic Optimization for Salient Object Detection in Optical Remote Sensing Imagery.](http://arxiv.org/abs/2303.02867) | 本文提出了一种面向光学遥感图像中显著目标检测的新方法，称为基于边界感知辅助和渐进语义优化的双反馈注意力框架 (DFA-BASO)。通过引入边界保护校准和双特征反馈补充模块，该方法能够减少信息损失、抑制噪声、增强目标的准确性和完整性。 |

# 详细

[^1]: 协作情境化：填补协同过滤和预训练语言模型之间的差距

    Collaborative Contextualization: Bridging the Gap between Collaborative Filtering and Pre-trained Language Model. (arXiv:2310.09400v1 [cs.IR])

    [http://arxiv.org/abs/2310.09400](http://arxiv.org/abs/2310.09400)

    本文介绍了一种名为CollabContext的模型，通过巧妙地将协同过滤信号与情境化表示相结合，同时保留了关键的情境语义，解决了传统推荐系统中协同信号和情境化表示之间的差距。

    

    传统的推荐系统在建模用户和物品时 heavily relied on identity representations (IDs)，而预训练语言模型 (PLM) 的兴起丰富了对情境化物品描述的建模。然而，尽管 PLM 在解决 few-shot、zero-shot 或统一建模场景方面非常有效，但常常忽视了关键的协同过滤信号。这种忽视带来了两个紧迫的挑战：(1) 协作情境化，即协同信号与情境化表示的无缝集成。(2) 在保留它们的情境语义的同时，弥合基于ID的表示和情境化表示之间的表示差距的必要性。在本文中，我们提出了CollabContext，一种新颖的模型，能够巧妙地将协同过滤信号与情境化表示结合起来，并将这些表示对齐在情境空间内，保留了重要的情境语义。实验结果表明...

    Traditional recommender systems have heavily relied on identity representations (IDs) to model users and items, while the ascendancy of pre-trained language model (PLM) encoders has enriched the modeling of contextual item descriptions. However, PLMs, although effective in addressing few-shot, zero-shot, or unified modeling scenarios, often neglect the crucial collaborative filtering signal. This neglect gives rise to two pressing challenges: (1) Collaborative Contextualization, the seamless integration of collaborative signals with contextual representations. (2) the imperative to bridge the representation gap between ID-based representations and contextual representations while preserving their contextual semantics. In this paper, we propose CollabContext, a novel model that adeptly combines collaborative filtering signals with contextual representations and aligns these representations within the contextual space, preserving essential contextual semantics. Experimental results acros
    
[^2]: 基于边界感知辅助和渐进语义优化的双反馈注意力框架用于光学遥感图像中的显著目标检测

    Dual Feedback Attention Framework via Boundary-Aware Auxiliary and Progressive Semantic Optimization for Salient Object Detection in Optical Remote Sensing Imagery. (arXiv:2303.02867v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2303.02867](http://arxiv.org/abs/2303.02867)

    本文提出了一种面向光学遥感图像中显著目标检测的新方法，称为基于边界感知辅助和渐进语义优化的双反馈注意力框架 (DFA-BASO)。通过引入边界保护校准和双特征反馈补充模块，该方法能够减少信息损失、抑制噪声、增强目标的准确性和完整性。

    

    光学遥感图像中的显著目标检测逐渐引起了人们的关注，这要归功于深度学习和自然场景图像中的显著目标检测的发展。然而，自然场景图像和光学遥感图像在许多方面是不同的，例如覆盖范围大、背景复杂以及目标类型和尺度的巨大差异。因此，需要一种专门的方法来处理光学遥感图像中的显著目标检测。此外，现有的方法没有充分关注到目标的边界，最终显著性图的完整性仍需要改进。为了解决这些问题，我们提出了一种新的方法，称为基于边界感知辅助和渐进语义优化的双反馈注意力框架（DFA-BASO）。首先，引入了边界保护校准(BPC)模块，用于减少正向传播过程中边界位置信息的丢失，并抑制低级特征中的噪声。其次，引入了双特征反馈补充(DFFC)模块，用于增强正反馈和负反馈之间的相互作用，提高显著性目标的准确性和完整性。

    Salient object detection in optical remote sensing image (ORSI-SOD) has gradually attracted attention thanks to the development of deep learning (DL) and salient object detection in natural scene image (NSI-SOD). However, NSI and ORSI are different in many aspects, such as large coverage, complex background, and large differences in target types and scales. Therefore, a new dedicated method is needed for ORSI-SOD. In addition, existing methods do not pay sufficient attention to the boundary of the object, and the completeness of the final saliency map still needs improvement. To address these issues, we propose a novel method called Dual Feedback Attention Framework via Boundary-Aware Auxiliary and Progressive Semantic Optimization (DFA-BASO). First, Boundary Protection Calibration (BPC) module is proposed to reduce the loss of edge position information during forward propagation and suppress noise in low-level features. Second, a Dual Feature Feedback Complementary (DFFC) module is pr
    

