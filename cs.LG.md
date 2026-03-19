# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Selective Prediction for Semantic Segmentation using Post-Hoc Confidence Estimation and Its Performance under Distribution Shift](https://arxiv.org/abs/2402.10665) | 本文研究了在低资源环境中语义分割的选择性预测，提出了一种针对语义分割量身定制的新型图像级置信度测量，并通过实验证明了其有效性 |
| [^2] | [QFT: Quantized Full-parameter Tuning of LLMs with Affordable Resources.](http://arxiv.org/abs/2310.07147) | 我们提出了一种新的QFT框架，可以对LLMs进行内存高效的全参数微调，而不损害性能。 |

# 详细

[^1]: 使用事后置信度估计的选择性预测在语义分割中的性能及其在分布偏移下的表现

    Selective Prediction for Semantic Segmentation using Post-Hoc Confidence Estimation and Its Performance under Distribution Shift

    [https://arxiv.org/abs/2402.10665](https://arxiv.org/abs/2402.10665)

    本文研究了在低资源环境中语义分割的选择性预测，提出了一种针对语义分割量身定制的新型图像级置信度测量，并通过实验证明了其有效性

    

    语义分割在各种计算机视觉应用中扮演着重要角色，然而其有效性常常受到高质量标记数据的缺乏所限。为了解决这一挑战，一个常见策略是利用在不同种群上训练的模型，如公开可用的数据集。然而，这种方法导致了分布偏移问题，在兴趣种群上表现出降低的性能。在模型错误可能带来重大后果的情况下，选择性预测方法提供了一种减轻风险、减少对专家监督依赖的手段。本文研究了在资源匮乏环境下语义分割的选择性预测，着重于应用于在分布偏移下运行的预训练模型的事后置信度估计器。我们提出了一种针对语义分割量身定制的新型图像级置信度测量，并通过实验证明了其有效性。

    arXiv:2402.10665v1 Announce Type: new  Abstract: Semantic segmentation plays a crucial role in various computer vision applications, yet its efficacy is often hindered by the lack of high-quality labeled data. To address this challenge, a common strategy is to leverage models trained on data from different populations, such as publicly available datasets. This approach, however, leads to the distribution shift problem, presenting a reduced performance on the population of interest. In scenarios where model errors can have significant consequences, selective prediction methods offer a means to mitigate risks and reduce reliance on expert supervision. This paper investigates selective prediction for semantic segmentation in low-resource settings, thus focusing on post-hoc confidence estimators applied to pre-trained models operating under distribution shift. We propose a novel image-level confidence measure tailored for semantic segmentation and demonstrate its effectiveness through expe
    
[^2]: QFT: 使用可承担资源对LLMs进行量化全参数调整

    QFT: Quantized Full-parameter Tuning of LLMs with Affordable Resources. (arXiv:2310.07147v1 [cs.CL])

    [http://arxiv.org/abs/2310.07147](http://arxiv.org/abs/2310.07147)

    我们提出了一种新的QFT框架，可以对LLMs进行内存高效的全参数微调，而不损害性能。

    

    大型语言模型（LLMs）在自然语言处理任务中展示出了显著的影响。对这些预训练模型进行微调可以进一步提高性能，但由于其巨大的资源需求，这一过程具有挑战性。为此，现有的努力都集中在参数高效的微调上，不幸的是，它们没有充分发挥全参数微调的潜力。在这项工作中，我们提出了QFT，一种新颖的用于LLMs的量化全参数调整框架，可以在不损害性能的情况下实现高效的内存微调。我们的框架包括两个新颖的思想：（i）我们采用高效的Lion优化器，仅跟踪动量并具有每个参数一致的更新幅度，这对于稳健的量化是一种内在优势；（ii）我们将所有模型状态进行量化，并以整数值存储，同时提供梯度流和参数更新的方法。

    Large Language Models (LLMs) have showcased remarkable impacts across a wide spectrum of natural language processing tasks. Fine-tuning these pre-trained models on downstream datasets provides further significant performance gains, but this process has been challenging due to its extraordinary resource requirements. To this end, existing efforts focus on parameter-efficient fine-tuning, which, unfortunately, fail to capitalize on the powerful potential of full-parameter fine-tuning. In this work, we propose QFT, a novel Quantized Full-parameter Tuning framework for LLMs that enables memory-efficient fine-tuning without harming performance. Our framework incorporates two novel ideas: (i) we adopt the efficient Lion optimizer, which only keeps track of the momentum and has consistent update magnitudes for each parameter, an inherent advantage for robust quantization; and (ii) we quantize all model states and store them as integer values, and present a gradient flow and parameter update s
    

