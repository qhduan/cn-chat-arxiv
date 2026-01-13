# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Correcting misinformation on social media with a large language model](https://arxiv.org/abs/2403.11169) | 提出了一种名为MUSE的大型语言模型，通过访问最新信息并评估可信度，以解决社交媒体上误信息纠正的难题。 |
| [^2] | [Browse and Concentrate: Comprehending Multimodal Content via prior-LLM Context Fusion](https://arxiv.org/abs/2402.12195) | 提出了一个两阶段范式"浏览和集中"，通过在将特征输入LLMs之前进行深入的多模态上下文融合，解决了多模态内容理解中的 prior-LLM 模态隔离问题 |

# 详细

[^1]: 使用大型语言模型纠正社交媒体上的错误信息

    Correcting misinformation on social media with a large language model

    [https://arxiv.org/abs/2403.11169](https://arxiv.org/abs/2403.11169)

    提出了一种名为MUSE的大型语言模型，通过访问最新信息并评估可信度，以解决社交媒体上误信息纠正的难题。

    

    误信息会破坏公众对科学和民主的信任，特别是在社交媒体上，不准确信息会迅速传播。专家和普通人通过手动识别和解释不准确信息已经被证明是有效的纠正误信息的方法。然而，这种方法很难扩展，这是一个担忧，因为大型语言模型（LLMs）等技术使误信息更容易生成。LLMs还具有多功能能力，可以加速纠正误信息；然而，它们由于缺乏最新信息、倾向于生成似是而非的内容和引用以及无法处理多模态信息而面临困难。为了解决这些问题，我们提出了MUSE，这是一个带有最新信息访问和可信度评估的LLM。通过检索上下文证据和反驳，MUSE可以提供准确可信的解释和参考。它还描述

    arXiv:2403.11169v1 Announce Type: cross  Abstract: Misinformation undermines public trust in science and democracy, particularly on social media where inaccuracies can spread rapidly. Experts and laypeople have shown to be effective in correcting misinformation by manually identifying and explaining inaccuracies. Nevertheless, this approach is difficult to scale, a concern as technologies like large language models (LLMs) make misinformation easier to produce. LLMs also have versatile capabilities that could accelerate misinformation correction; however, they struggle due to a lack of recent information, a tendency to produce plausible but false content and references, and limitations in addressing multimodal information. To address these issues, we propose MUSE, an LLM augmented with access to and credibility evaluation of up-to-date information. By retrieving contextual evidence and refutations, MUSE can provide accurate and trustworthy explanations and references. It also describes 
    
[^2]: 通过 prior-LLM 上下文融合来理解多模态内容

    Browse and Concentrate: Comprehending Multimodal Content via prior-LLM Context Fusion

    [https://arxiv.org/abs/2402.12195](https://arxiv.org/abs/2402.12195)

    提出了一个两阶段范式"浏览和集中"，通过在将特征输入LLMs之前进行深入的多模态上下文融合，解决了多模态内容理解中的 prior-LLM 模态隔离问题

    

    随着大型语言模型（LLMs）的兴起，近期将LLMs与预训练的视觉模型相结合的多模态大型语言模型（MLLMs）已经展现出在各种视觉语言任务上令人印象深刻的性能。然而，它们在理解涉及多张图片的上下文方面仍有不足。这一缺陷的主要原因是，在将视觉特征输入LLM主干之前，每张图片的视觉特征都是由冻结的编码器单独编码的，缺乏对其他图片和多模态指令的意识。我们将这一问题称为 prior-LLM 模态隔离，并提出了一个两阶段范式，即“浏览和集中”，以实现在将特征输入LLMs之前进行深入的多模态上下文融合。这种范式最初“浏览”输入以获取关键见解，然后再次回顾输入“集中”于关键细节，通过这些见解的指导，从而实现对多模态内容更全面的理解。

    arXiv:2402.12195v1 Announce Type: new  Abstract: With the bloom of Large Language Models (LLMs), Multimodal Large Language Models (MLLMs) that incorporate LLMs with pre-trained vision models have recently demonstrated impressive performance across diverse vision-language tasks. However, they fall short to comprehend context involving multiple images. A primary reason for this shortcoming is that the visual features for each images are encoded individually by frozen encoders before feeding into the LLM backbone, lacking awareness of other images and the multimodal instructions. We term this issue as prior-LLM modality isolation and propose a two phase paradigm, browse-and-concentrate, to enable in-depth multimodal context fusion prior to feeding the features into LLMs. This paradigm initially "browses" through the inputs for essential insights, and then revisits the inputs to "concentrate" on crucial details, guided by these insights, to achieve a more comprehensive understanding of the
    

