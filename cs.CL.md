# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Streaming Sequence Transduction through Dynamic Compression](https://rss.arxiv.org/abs/2402.01172) | STAR是一种新型的Transformer模型，通过动态压缩和优化延迟、内存占用和质量，实现对流的高效序列转导，并在自动语音识别领域表现出色。 |
| [^2] | [Uncertainty quantification in fine-tuned LLMs using LoRA ensembles](https://arxiv.org/abs/2402.12264) | 使用LoRA集成在精调LLMs中提出了一种原则性不确定性量化方法，通过对不同数据域的低秩适应集成分析，推测了模型对特定架构难以学习的数据领域的信号。 |
| [^3] | [SpikeCLIP: A Contrastive Language-Image Pretrained Spiking Neural Network.](http://arxiv.org/abs/2310.06488) | 本论文引入了一种名为SpikeCLIP的新框架，通过对比语言-图像预训练实现了脉冲神经网络的多模态扩展，并在能源效率和性能方面取得了可比较的结果。 |

# 详细

[^1]: 流式序列转导通过动态压缩

    Streaming Sequence Transduction through Dynamic Compression

    [https://rss.arxiv.org/abs/2402.01172](https://rss.arxiv.org/abs/2402.01172)

    STAR是一种新型的Transformer模型，通过动态压缩和优化延迟、内存占用和质量，实现对流的高效序列转导，并在自动语音识别领域表现出色。

    

    我们引入了STAR（带有锚定表示的流式转导），这是一种基于Transformer的新型模型，旨在实现对流的高效序列转导。STAR动态地对输入流进行分段，创建压缩的锚定表示，实现近乎无损的压缩（12倍）在自动语音识别（ASR）中，并优于现有方法。此外，STAR在同时进行语音到文本任务中展示出优越的分割和延迟-质量折衷，优化延迟、内存占用和质量。

    We introduce STAR (Stream Transduction with Anchor Representations), a novel Transformer-based model designed for efficient sequence-to-sequence transduction over streams. STAR dynamically segments input streams to create compressed anchor representations, achieving nearly lossless compression (12x) in Automatic Speech Recognition (ASR) and outperforming existing methods. Moreover, STAR demonstrates superior segmentation and latency-quality trade-offs in simultaneous speech-to-text tasks, optimizing latency, memory footprint, and quality.
    
[^2]: 使用LoRA集成在精调LLMs中的不确定性量化

    Uncertainty quantification in fine-tuned LLMs using LoRA ensembles

    [https://arxiv.org/abs/2402.12264](https://arxiv.org/abs/2402.12264)

    使用LoRA集成在精调LLMs中提出了一种原则性不确定性量化方法，通过对不同数据域的低秩适应集成分析，推测了模型对特定架构难以学习的数据领域的信号。

    

    精调大型语言模型可以提高特定任务的性能，尽管对于精调模型学到了什么、遗忘了什么以及如何信任其预测仍然缺乏一个一般的理解。我们提出了使用计算效率高的低秩适应集成对精调LLMs进行基于后验逼近的原则性不确定性量化。我们使用基于Mistral-7b的低秩适应集成分析了三个常见的多项选择数据集，并对其在精调过程中和之后对不同目标领域的感知复杂性和模型效能进行了定量和定性的结论。具体而言，基于数值实验支持，我们对那些对于给定架构难以学习的数据领域的熵不确定性度量提出了假设。

    arXiv:2402.12264v1 Announce Type: cross  Abstract: Fine-tuning large language models can improve task specific performance, although a general understanding of what the fine-tuned model has learned, forgotten and how to trust its predictions is still missing. We derive principled uncertainty quantification for fine-tuned LLMs with posterior approximations using computationally efficient low-rank adaptation ensembles. We analyze three common multiple-choice datasets using low-rank adaptation ensembles based on Mistral-7b, and draw quantitative and qualitative conclusions on their perceived complexity and model efficacy on the different target domains during and after fine-tuning. In particular, backed by the numerical experiments, we hypothesise about signals from entropic uncertainty measures for data domains that are inherently difficult for a given architecture to learn.
    
[^3]: SpikeCLIP：一种对比语言-图像预训练脉冲神经网络

    SpikeCLIP: A Contrastive Language-Image Pretrained Spiking Neural Network. (arXiv:2310.06488v2 [cs.NE] UPDATED)

    [http://arxiv.org/abs/2310.06488](http://arxiv.org/abs/2310.06488)

    本论文引入了一种名为SpikeCLIP的新框架，通过对比语言-图像预训练实现了脉冲神经网络的多模态扩展，并在能源效率和性能方面取得了可比较的结果。

    

    脉冲神经网络（SNNs）已经证明其在视觉和语言领域中能够实现与深度神经网络（DNNs）相当的性能，同时具有能效提高和符合生物合理性的优势。然而，将这种单模态的SNNs扩展到多模态的情景仍然是一个未开发的领域。受到对比语言-图像预训练（CLIP）概念的启发，我们引入了一个名为SpikeCLIP的新框架，通过“对齐预训练+双损失微调”的两步骤配方，来解决脉冲计算背景下两种模态之间的差距。广泛的实验证明，在常用的用于多模态模型评估的各种数据集上，SNNs取得了与其DNNs对应物相当的结果，同时显著降低了能源消耗。此外，SpikeCLIP在图像分类方面保持了稳定的性能。

    Spiking neural networks (SNNs) have demonstrated the capability to achieve comparable performance to deep neural networks (DNNs) in both visual and linguistic domains while offering the advantages of improved energy efficiency and adherence to biological plausibility. However, the extension of such single-modality SNNs into the realm of multimodal scenarios remains an unexplored territory. Drawing inspiration from the concept of contrastive language-image pre-training (CLIP), we introduce a novel framework, named SpikeCLIP, to address the gap between two modalities within the context of spike-based computing through a two-step recipe involving ``Alignment Pre-training + Dual-Loss Fine-tuning". Extensive experiments demonstrate that SNNs achieve comparable results to their DNN counterparts while significantly reducing energy consumption across a variety of datasets commonly used for multimodal model evaluation. Furthermore, SpikeCLIP maintains robust performance in image classification 
    

