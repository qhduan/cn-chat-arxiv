# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Knowledge Soft Integration for Multimodal Recommendation.](http://arxiv.org/abs/2305.07419) | 本文提出了一个知识软融合框架用于多模态推荐，该框架在特征提取和推荐过程中集成了领域特定的知识，解决了模型拟合偏差和性能下降的问题。 |

# 详细

[^1]: 多模态推荐中的知识软融合

    Knowledge Soft Integration for Multimodal Recommendation. (arXiv:2305.07419v1 [cs.IR])

    [http://arxiv.org/abs/2305.07419](http://arxiv.org/abs/2305.07419)

    本文提出了一个知识软融合框架用于多模态推荐，该框架在特征提取和推荐过程中集成了领域特定的知识，解决了模型拟合偏差和性能下降的问题。

    

    现代推荐系统中的主要挑战之一是如何有效地利用多模态内容实现更个性化的推荐。尽管有各种各样的解决方案，但大多数解决方案都忽略了独立特征提取过程所获得知识与下游推荐任务之间的不匹配。具体而言，多模态特征提取过程未纳入与推荐任务相关的先前知识，而推荐任务经常直接将这些多模态特征用作辅助信息。这种不匹配可能导致模型拟合偏差和性能下降，本文将其称为“知识诅咒”问题。为了解决这个问题，我们提出使用知识软融合平衡多模态特征利用和知识诅咒问题带来的负面影响。为此，我们提出了一个适用于多模态推荐的知识软融合框架，简称KSI，它将领域特定的知识集成到特征提取和推荐过程中。在三个真实数据集上的实验表明，KSI在推荐准确性和多样性方面超越了几种最先进的方法。

    One of the main challenges in modern recommendation systems is how to effectively utilize multimodal content to achieve more personalized recommendations. Despite various proposed solutions, most of them overlook the mismatch between the knowledge gained from independent feature extraction processes and downstream recommendation tasks. Specifically, multimodal feature extraction processes do not incorporate prior knowledge relevant to recommendation tasks, while recommendation tasks often directly use these multimodal features as side information. This mismatch can lead to model fitting biases and performance degradation, which this paper refers to as the \textit{curse of knowledge} problem. To address this issue, we propose using knowledge soft integration to balance the utilization of multimodal features and the curse of knowledge problem it brings about. To achieve this, we put forward a Knowledge Soft Integration framework for the multimodal recommendation, abbreviated as KSI, whic
    

