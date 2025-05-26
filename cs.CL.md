# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [QFT: Quantized Full-parameter Tuning of LLMs with Affordable Resources.](http://arxiv.org/abs/2310.07147) | 我们提出了一种新的QFT框架，可以对LLMs进行内存高效的全参数微调，而不损害性能。 |

# 详细

[^1]: QFT: 使用可承担资源对LLMs进行量化全参数调整

    QFT: Quantized Full-parameter Tuning of LLMs with Affordable Resources. (arXiv:2310.07147v1 [cs.CL])

    [http://arxiv.org/abs/2310.07147](http://arxiv.org/abs/2310.07147)

    我们提出了一种新的QFT框架，可以对LLMs进行内存高效的全参数微调，而不损害性能。

    

    大型语言模型（LLMs）在自然语言处理任务中展示出了显著的影响。对这些预训练模型进行微调可以进一步提高性能，但由于其巨大的资源需求，这一过程具有挑战性。为此，现有的努力都集中在参数高效的微调上，不幸的是，它们没有充分发挥全参数微调的潜力。在这项工作中，我们提出了QFT，一种新颖的用于LLMs的量化全参数调整框架，可以在不损害性能的情况下实现高效的内存微调。我们的框架包括两个新颖的思想：（i）我们采用高效的Lion优化器，仅跟踪动量并具有每个参数一致的更新幅度，这对于稳健的量化是一种内在优势；（ii）我们将所有模型状态进行量化，并以整数值存储，同时提供梯度流和参数更新的方法。

    Large Language Models (LLMs) have showcased remarkable impacts across a wide spectrum of natural language processing tasks. Fine-tuning these pre-trained models on downstream datasets provides further significant performance gains, but this process has been challenging due to its extraordinary resource requirements. To this end, existing efforts focus on parameter-efficient fine-tuning, which, unfortunately, fail to capitalize on the powerful potential of full-parameter fine-tuning. In this work, we propose QFT, a novel Quantized Full-parameter Tuning framework for LLMs that enables memory-efficient fine-tuning without harming performance. Our framework incorporates two novel ideas: (i) we adopt the efficient Lion optimizer, which only keeps track of the momentum and has consistent update magnitudes for each parameter, an inherent advantage for robust quantization; and (ii) we quantize all model states and store them as integer values, and present a gradient flow and parameter update s
    

