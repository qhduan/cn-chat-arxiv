# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [SpikeNAS: A Fast Memory-Aware Neural Architecture Search Framework for Spiking Neural Network Systems](https://arxiv.org/abs/2402.11322) | SpikeNAS提出了一种快速内存感知神经架构搜索框架，旨在帮助脉冲神经网络系统快速找到在给定内存预算下高准确性的适当架构。 |
| [^2] | [EdgeQAT: Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge](https://arxiv.org/abs/2402.10787) | 本文提出了EdgeQAT，使用熵和分布引导的量化感知训练方法来优化轻量级LLMs，在边缘设备上实现推理加速。 |
| [^3] | [Dataset Distillation via the Wasserstein Metric](https://arxiv.org/abs/2311.18531) | 通过引入Wasserstein距离及其重心，我们提出一种有效的数据集精炼方法，利用先验知识提高分布匹配效果，实现了新的最先进性能。 |

# 详细

[^1]: SpikeNAS: 一种面向脉冲神经网络系统的快速内存感知神经架构搜索框架

    SpikeNAS: A Fast Memory-Aware Neural Architecture Search Framework for Spiking Neural Network Systems

    [https://arxiv.org/abs/2402.11322](https://arxiv.org/abs/2402.11322)

    SpikeNAS提出了一种快速内存感知神经架构搜索框架，旨在帮助脉冲神经网络系统快速找到在给定内存预算下高准确性的适当架构。

    

    脉冲神经网络（SNN）为解决机器学习任务提供了实现超低功耗计算的有前途的解决方案。目前，大多数SNN架构都源自人工神经网络，其神经元的架构和操作与SNN不同，或者在不考虑来自底层处理硬件的内存预算的情况下开发。这些限制阻碍了SNN在准确性和效率方面充分发挥潜力。为此，我们提出了SpikeNAS，一种新颖的内存感知神经架构搜索（NAS）框架，可在给定内存预算下快速找到一个具有高准确性的适当SNN架构。为实现这一目标，我们的SpikeNAS采用了几个关键步骤：分析网络操作对准确性的影响，增强网络架构以提高学习质量，并开发快速内存感知搜索算法。

    arXiv:2402.11322v1 Announce Type: cross  Abstract: Spiking Neural Networks (SNNs) offer a promising solution to achieve ultra low-power/energy computation for solving machine learning tasks. Currently, most of the SNN architectures are derived from Artificial Neural Networks whose neurons' architectures and operations are different from SNNs, or developed without considering memory budgets from the underlying processing hardware. These limitations hinder the SNNs from reaching their full potential in accuracy and efficiency. Towards this, we propose SpikeNAS, a novel memory-aware neural architecture search (NAS) framework for SNNs that can quickly find an appropriate SNN architecture with high accuracy under the given memory budgets. To do this, our SpikeNAS employs several key steps: analyzing the impacts of network operations on the accuracy, enhancing the network architecture to improve the learning quality, and developing a fast memory-aware search algorithm. The experimental resul
    
[^2]: EdgeQAT: 熵和分布引导的量化感知训练，用于加速轻量级LLMs在边缘设备上的应用

    EdgeQAT: Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge

    [https://arxiv.org/abs/2402.10787](https://arxiv.org/abs/2402.10787)

    本文提出了EdgeQAT，使用熵和分布引导的量化感知训练方法来优化轻量级LLMs，在边缘设备上实现推理加速。

    

    尽管大型语言模型（LLMs）在各个领域取得了显著进展，但由于其庞大的参数和计算量，LLMs在边缘设备上的广泛应用受到限制。为了解决这一问题，通常采用量化方法生成具有高效计算和快速推理的轻量级LLMs。然而，后训练量化（PTQ）方法在将权重、激活和KV缓存一起量化至8位以下时，质量会急剧下降。此外，许多量化感知训练（QAT）工作对模型权重进行量化，而激活未被触及，这不能充分发挥量化对边缘端推理加速的潜力。在本文中，我们提出了EdgeQAT，即熵和分布引导的QAT，用于优化轻量级LLMs以实现在边缘设备上的推理加速。我们首先确定量化性能下降主要源自信息

    arXiv:2402.10787v1 Announce Type: cross  Abstract: Despite the remarkable strides of Large Language Models (LLMs) in various fields, the wide applications of LLMs on edge devices are limited due to their massive parameters and computations. To address this, quantization is commonly adopted to generate lightweight LLMs with efficient computations and fast inference. However, Post-Training Quantization (PTQ) methods dramatically degrade in quality when quantizing weights, activations, and KV cache together to below 8 bits. Besides, many Quantization-Aware Training (QAT) works quantize model weights, leaving the activations untouched, which do not fully exploit the potential of quantization for inference acceleration on the edge. In this paper, we propose EdgeQAT, the Entropy and Distribution Guided QAT for the optimization of lightweight LLMs to achieve inference acceleration on Edge devices. We first identify that the performance drop of quantization primarily stems from the information
    
[^3]: 通过Wasserstein度量进行数据集精炼

    Dataset Distillation via the Wasserstein Metric

    [https://arxiv.org/abs/2311.18531](https://arxiv.org/abs/2311.18531)

    通过引入Wasserstein距离及其重心，我们提出一种有效的数据集精炼方法，利用先验知识提高分布匹配效果，实现了新的最先进性能。

    

    数据集精炼（DD）作为一种强大的策略，将大型数据集的丰富信息封装为明显更小的合成等价物，从而在减少计算开销的同时保留模型性能。为实现这一目标，我们引入了Wasserstein距离，这是一种基于最优输运理论的度量，用于增强DD中的分布匹配。我们的方法利用Wasserstein重心提供了一种在量化分布差异和高效捕获分布集合中心的几何意义方法。通过在预训练分类模型的特征空间中嵌入合成数据，我们促进了有效的分布匹配，利用这些模型固有的先验知识。我们的方法不仅保持了基于分布匹配的技术的计算优势，而且在一系列任务中实现了新的最先进性能。

    arXiv:2311.18531v2 Announce Type: replace-cross  Abstract: Dataset Distillation (DD) emerges as a powerful strategy to encapsulate the expansive information of large datasets into significantly smaller, synthetic equivalents, thereby preserving model performance with reduced computational overhead. Pursuing this objective, we introduce the Wasserstein distance, a metric grounded in optimal transport theory, to enhance distribution matching in DD. Our approach employs the Wasserstein barycenter to provide a geometrically meaningful method for quantifying distribution differences and capturing the centroid of distribution sets efficiently. By embedding synthetic data in the feature spaces of pretrained classification models, we facilitate effective distribution matching that leverages prior knowledge inherent in these models. Our method not only maintains the computational advantages of distribution matching-based techniques but also achieves new state-of-the-art performance across a ran
    

