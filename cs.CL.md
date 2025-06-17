# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Ask Optimal Questions: Aligning Large Language Models with Retriever's Preference in Conversational Search](https://arxiv.org/abs/2402.11827) | 提出了RetPO框架，通过优化语言模型对搜索查询进行重构，以符合目标检索系统的偏好，并构建了一个大型数据集RF Collection，用于收集检索结果作为检索器的偏好。 |
| [^2] | [Video Understanding with Large Language Models: A Survey.](http://arxiv.org/abs/2312.17432) | 这项调查研究提供了对大型语言模型（Vid-LLMs）在视频理解中的最新进展的详细概述。Vid-LLMs的新兴能力包括开放式时空推理和常识知识，为未来的视频理解提供了有前途的方向。 |
| [^3] | [Compressing Transformer-based self-supervised models for speech processing.](http://arxiv.org/abs/2211.09949) | 本文研究了对基于Transformer的自监督模型进行压缩的方法，包括权重修剪、头部修剪、低秩逼近和知识蒸馏。结果发现，基本的压缩技术是强大的基准，可以改善模型的压缩效果。 |

# 详细

[^1]: 询问最佳问题：将大型语言模型与检索器偏好在会话搜索中对齐

    Ask Optimal Questions: Aligning Large Language Models with Retriever's Preference in Conversational Search

    [https://arxiv.org/abs/2402.11827](https://arxiv.org/abs/2402.11827)

    提出了RetPO框架，通过优化语言模型对搜索查询进行重构，以符合目标检索系统的偏好，并构建了一个大型数据集RF Collection，用于收集检索结果作为检索器的偏好。

    

    会话式搜索与单轮检索任务不同，需要理解对话上下文中的当前问题。常见的“重写-然后检索”的方法旨在将问题去上下文化，使其对现成的检索器自给自足，但大多数现有方法由于能力有限而产生次优的查询重写，无法充分利用来自检索结果的信号。为了克服这一限制，我们提出了一种新颖的框架RetPO（检索器偏好优化），旨在优化语言模型（LM）以符合目标检索系统的重写搜索查询的偏好。该过程始于提示大型LM生成各种潜在重写，然后收集这些重写的检索性能作为检索器的偏好。通过该过程，我们构建了一个名为RF塑集的大型数据集，其中包含对超过410K个查询的检索器反馈。

    arXiv:2402.11827v1 Announce Type: cross  Abstract: Conversational search, unlike single-turn retrieval tasks, requires understanding the current question within a dialogue context. The common approach of rewrite-then-retrieve aims to decontextualize questions to be self-sufficient for off-the-shelf retrievers, but most existing methods produce sub-optimal query rewrites due to the limited ability to incorporate signals from the retrieval results. To overcome this limitation, we present a novel framework RetPO (Retriever's Preference Optimization), which is designed to optimize a language model (LM) for reformulating search queries in line with the preferences of the target retrieval systems. The process begins by prompting a large LM to produce various potential rewrites and then collects retrieval performance for these rewrites as the retrievers' preferences. Through the process, we construct a large-scale dataset called RF collection, containing Retrievers' Feedback on over 410K quer
    
[^2]: 大型语言模型在视频理解中的应用：一项调查研究

    Video Understanding with Large Language Models: A Survey. (arXiv:2312.17432v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2312.17432](http://arxiv.org/abs/2312.17432)

    这项调查研究提供了对大型语言模型（Vid-LLMs）在视频理解中的最新进展的详细概述。Vid-LLMs的新兴能力包括开放式时空推理和常识知识，为未来的视频理解提供了有前途的方向。

    

    随着在线视频平台的不断增长和视频内容的不断增多，对熟练的视频理解工具的需求显著增加。鉴于大型语言模型在语言和多模态任务中的卓越能力，本调查提供了对利用大型语言模型（Vid-LLMs）技术进行视频理解的最新进展的详细概述。Vid-LLMs的新兴能力令人惊讶，尤其是它们在开放式时空推理和常识知识方面的能力，为未来的视频理解提供了一个有前途的方向。本调查对Vid-LLMs的独特特点和能力进行了分类，分为四种主要类型：基于LLM的视频代理、Vid-LLMs的预训练、Vid-LLMs的指令调整和混合方法。此外，本调查对Vid-LLMs的任务、数据集和评估方法进行了全面的研究。另外，它还探讨了Vid-LLMs技术的局限性和未来的挑战。

    With the burgeoning growth of online video platforms and the escalating volume of video content, the demand for proficient video understanding tools has intensified markedly. Given the remarkable capabilities of Large Language Models (LLMs) in language and multimodal tasks, this survey provides a detailed overview of the recent advancements in video understanding harnessing the power of LLMs (Vid-LLMs). The emergent capabilities of Vid-LLMs are surprisingly advanced, particularly their ability for open-ended spatial-temporal reasoning combined with commonsense knowledge, suggesting a promising path for future video understanding. We examine the unique characteristics and capabilities of Vid-LLMs, categorizing the approaches into four main types: LLM-based Video Agents, Vid-LLMs Pretraining, Vid-LLMs Instruction Tuning, and Hybrid Methods. Furthermore, this survey presents a comprehensive study of the tasks, datasets, and evaluation methodologies for Vid-LLMs. Additionally, it explores 
    
[^3]: 对基于Transformer的自监督模型在语音处理中进行压缩

    Compressing Transformer-based self-supervised models for speech processing. (arXiv:2211.09949v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.09949](http://arxiv.org/abs/2211.09949)

    本文研究了对基于Transformer的自监督模型进行压缩的方法，包括权重修剪、头部修剪、低秩逼近和知识蒸馏。结果发现，基本的压缩技术是强大的基准，可以改善模型的压缩效果。

    

    尽管Transformer在自监督学习中取得了成功，并应用于各种下游任务，但是训练和推断的计算成本仍然是将这些模型应用于各种设备的主要挑战。目前已有一些孤立的尝试来压缩Transformer，但研究中的设置和指标各不相同。此前的工作很少涉及不同压缩率之间的权衡，这使得比较压缩技术变得困难。在这项工作中，我们旨在为这些孤立结果提供背景，研究几种常用的压缩技术，包括权重修剪、头部修剪、低秩逼近和知识蒸馏。我们报告了在不同压缩率下的权衡，包括墙钟时间、参数数量和乘加操作数量。我们的结果表明，与最近的方法相比，基本的压缩技术是强大的基准。我们进一步提出了几种压缩方法来改进模型的压缩效果。

    Despite the success of Transformers in self- supervised learning with applications to various downstream tasks, the computational cost of training and inference remains a major challenge for applying these models to a wide spectrum of devices. Several isolated attempts have been made to compress Transformers, but the settings and metrics are different across studies. Trade-off at various compression rates are also largely missing in prior work, making it difficult to compare compression techniques. In this work, we aim to provide context for the isolated results, studying several commonly used compression techniques, including weight pruning, head pruning, low-rank approximation, and knowledge distillation. We report trade- off at various compression rate, including wall-clock time, the number of parameters, and the number of multiply-accumulate operations. Our results show that compared to recent approaches, basic compression techniques are strong baselines. We further present several
    

