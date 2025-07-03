# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [EdgeQAT: Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge](https://arxiv.org/abs/2402.10787) | 本文提出了EdgeQAT，使用熵和分布引导的量化感知训练方法来优化轻量级LLMs，在边缘设备上实现推理加速。 |

# 详细

[^1]: EdgeQAT: 熵和分布引导的量化感知训练，用于加速轻量级LLMs在边缘设备上的应用

    EdgeQAT: Entropy and Distribution Guided Quantization-Aware Training for the Acceleration of Lightweight LLMs on the Edge

    [https://arxiv.org/abs/2402.10787](https://arxiv.org/abs/2402.10787)

    本文提出了EdgeQAT，使用熵和分布引导的量化感知训练方法来优化轻量级LLMs，在边缘设备上实现推理加速。

    

    尽管大型语言模型（LLMs）在各个领域取得了显著进展，但由于其庞大的参数和计算量，LLMs在边缘设备上的广泛应用受到限制。为了解决这一问题，通常采用量化方法生成具有高效计算和快速推理的轻量级LLMs。然而，后训练量化（PTQ）方法在将权重、激活和KV缓存一起量化至8位以下时，质量会急剧下降。此外，许多量化感知训练（QAT）工作对模型权重进行量化，而激活未被触及，这不能充分发挥量化对边缘端推理加速的潜力。在本文中，我们提出了EdgeQAT，即熵和分布引导的QAT，用于优化轻量级LLMs以实现在边缘设备上的推理加速。我们首先确定量化性能下降主要源自信息

    arXiv:2402.10787v1 Announce Type: cross  Abstract: Despite the remarkable strides of Large Language Models (LLMs) in various fields, the wide applications of LLMs on edge devices are limited due to their massive parameters and computations. To address this, quantization is commonly adopted to generate lightweight LLMs with efficient computations and fast inference. However, Post-Training Quantization (PTQ) methods dramatically degrade in quality when quantizing weights, activations, and KV cache together to below 8 bits. Besides, many Quantization-Aware Training (QAT) works quantize model weights, leaving the activations untouched, which do not fully exploit the potential of quantization for inference acceleration on the edge. In this paper, we propose EdgeQAT, the Entropy and Distribution Guided QAT for the optimization of lightweight LLMs to achieve inference acceleration on Edge devices. We first identify that the performance drop of quantization primarily stems from the information
    

