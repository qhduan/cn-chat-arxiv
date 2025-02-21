# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Confidence-aware Fine-tuning of Sequential Recommendation Systems via Conformal Prediction](https://arxiv.org/abs/2402.08976) | 本研究提出了CPFT框架，通过在顺序推荐系统中精细调整过程中结合交叉熵损失函数和基于符合性预测的损失函数，增强了推荐系统的置信度。CPFT动态生成潜在真实值的项目集合，提升了训练过程中的性能，并提高了推荐的准确性和可信度。 |

# 详细

[^1]: 通过符合性预测实现置信度感知的顺序推荐系统的精细调整

    Confidence-aware Fine-tuning of Sequential Recommendation Systems via Conformal Prediction

    [https://arxiv.org/abs/2402.08976](https://arxiv.org/abs/2402.08976)

    本研究提出了CPFT框架，通过在顺序推荐系统中精细调整过程中结合交叉熵损失函数和基于符合性预测的损失函数，增强了推荐系统的置信度。CPFT动态生成潜在真实值的项目集合，提升了训练过程中的性能，并提高了推荐的准确性和可信度。

    

    在顺序推荐系统中，通常使用交叉熵损失函数，但在训练过程中未能利用项目置信度分数。为了认识到置信度在将训练目标与评估指标对齐中的关键作用，我们提出了CPFT，这是一个多功能的框架，通过在精细调整过程中将基于符合性预测的损失函数与交叉熵损失函数相结合，增强了推荐系统的置信度。CPFT动态生成一组具有高概率包含真实值的项目，通过将验证数据纳入训练过程而不损害其在模型选择中的作用，丰富了训练过程。这种创新的方法与基于符合性预测的损失函数相结合，更专注于改善推荐集合，从而提高潜在项目预测的置信度。通过通过基于符合性预测的损失函数对项目置信度进行精细调整，CPFT显著提高了模型性能，提供了更精确和可信的推荐。

    arXiv:2402.08976v1 Announce Type: new Abstract: In Sequential Recommendation Systems, Cross-Entropy (CE) loss is commonly used but fails to harness item confidence scores during training. Recognizing the critical role of confidence in aligning training objectives with evaluation metrics, we propose CPFT, a versatile framework that enhances recommendation confidence by integrating Conformal Prediction (CP)-based losses with CE loss during fine-tuning. CPFT dynamically generates a set of items with a high probability of containing the ground truth, enriching the training process by incorporating validation data without compromising its role in model selection. This innovative approach, coupled with CP-based losses, sharpens the focus on refining recommendation sets, thereby elevating the confidence in potential item predictions. By fine-tuning item confidence through CP-based losses, CPFT significantly enhances model performance, leading to more precise and trustworthy recommendations th
    

