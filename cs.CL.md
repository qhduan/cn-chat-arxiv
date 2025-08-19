# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation](https://arxiv.org/abs/2403.19103) | PRISM是一种算法，可以自动识别人类可解释且易传递的提示，从而有效生成所需概念，仅使用黑盒访问T2I模型。 |
| [^2] | [Large language models can replicate cross-cultural differences in personality.](http://arxiv.org/abs/2310.10679) | 大型语言模型GPT-4成功复制了使用十项人格问卷测量的大五人格的跨文化差异，但其结果表明平均评级有上升偏差和较低的变异性与结构效度。 |
| [^3] | [Compressing Transformer-based self-supervised models for speech processing.](http://arxiv.org/abs/2211.09949) | 本文研究了对基于Transformer的自监督模型进行压缩的方法，包括权重修剪、头部修剪、低秩逼近和知识蒸馏。结果发现，基本的压缩技术是强大的基准，可以改善模型的压缩效果。 |

# 详细

[^1]: 用于个性化文本到图像生成的自动化黑盒提示工程

    Automated Black-box Prompt Engineering for Personalized Text-to-Image Generation

    [https://arxiv.org/abs/2403.19103](https://arxiv.org/abs/2403.19103)

    PRISM是一种算法，可以自动识别人类可解释且易传递的提示，从而有效生成所需概念，仅使用黑盒访问T2I模型。

    

    提示工程对于控制文本到图像（T2I）生成模型的输出是有效的，但由于需要手动制作提示而导致工作繁重。这一挑战促使了自动提示生成算法的发展。然而，这些方法通常在T2I模型之间的可传递性方面遇到困难，需要对基础模型进行白盒访问，并产生非直观的提示。在这项工作中，我们介绍了PRISM，这是一种算法，可以仅使用黑盒访问T2I模型就自动识别人类可解释且易传递的提示，从而有效生成所需概念。受大型语言模型（LLM）越狱的启发，PRISM利用LLM的上下文学习能力来迭代地改进给定参考图像的候选提示分布。我们的实验展示了PRISM在为对象、样式等生成准确提示方面的多样性和有效性。

    arXiv:2403.19103v1 Announce Type: cross  Abstract: Prompt engineering is effective for controlling the output of text-to-image (T2I) generative models, but it is also laborious due to the need for manually crafted prompts. This challenge has spurred the development of algorithms for automated prompt generation. However, these methods often struggle with transferability across T2I models, require white-box access to the underlying model, and produce non-intuitive prompts. In this work, we introduce PRISM, an algorithm that automatically identifies human-interpretable and transferable prompts that can effectively generate desired concepts given only black-box access to T2I models. Inspired by large language model (LLM) jailbreaking, PRISM leverages the in-context learning ability of LLMs to iteratively refine the candidate prompts distribution for given reference images. Our experiments demonstrate the versatility and effectiveness of PRISM in generating accurate prompts for objects, sty
    
[^2]: 大型语言模型可以复制跨文化个性差异

    Large language models can replicate cross-cultural differences in personality. (arXiv:2310.10679v1 [cs.CL])

    [http://arxiv.org/abs/2310.10679](http://arxiv.org/abs/2310.10679)

    大型语言模型GPT-4成功复制了使用十项人格问卷测量的大五人格的跨文化差异，但其结果表明平均评级有上升偏差和较低的变异性与结构效度。

    

    我们使用一项大规模实验(N=8000)来确定GPT-4是否可以复制使用十项人格问卷测量的大五人格的跨文化差异。我们选择美国和韩国作为文化对比，因为先前的研究表明这两个国家的人之间存在显著的人格差异。我们操纵了模拟的目标（美国 vs. 韩国），问卷的语言（英语 vs. 韩语）以及语言模型（GPT-4 vs. GPT-3.5）。我们的结果表明，GPT-4复制了每个因子的跨文化差异。然而，平均评级具有上升偏差，并且比人类样本的变异性更低，以及结构效度较低。总的来说，我们提供了初步的证据说明LLMs可以促进跨文化心理研究。

    We use a large-scale experiment (N=8000) to determine whether GPT-4 can replicate cross-cultural differences in the Big Five, measured using the Ten-Item Personality Inventory. We used the US and South Korea as the cultural pair, given that prior research suggests substantial personality differences between people from these two countries. We manipulated the target of the simulation (US vs. Korean), the language of the inventory (English vs. Korean), and the language model (GPT-4 vs. GPT-3.5). Our results show that GPT-4 replicated the cross-cultural differences for each factor. However, mean ratings had an upward bias and exhibited lower variation than in the human samples, as well as lower structural validity. Overall, we provide preliminary evidence that LLMs can aid cross-cultural psychological research.
    
[^3]: 对基于Transformer的自监督模型在语音处理中进行压缩

    Compressing Transformer-based self-supervised models for speech processing. (arXiv:2211.09949v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.09949](http://arxiv.org/abs/2211.09949)

    本文研究了对基于Transformer的自监督模型进行压缩的方法，包括权重修剪、头部修剪、低秩逼近和知识蒸馏。结果发现，基本的压缩技术是强大的基准，可以改善模型的压缩效果。

    

    尽管Transformer在自监督学习中取得了成功，并应用于各种下游任务，但是训练和推断的计算成本仍然是将这些模型应用于各种设备的主要挑战。目前已有一些孤立的尝试来压缩Transformer，但研究中的设置和指标各不相同。此前的工作很少涉及不同压缩率之间的权衡，这使得比较压缩技术变得困难。在这项工作中，我们旨在为这些孤立结果提供背景，研究几种常用的压缩技术，包括权重修剪、头部修剪、低秩逼近和知识蒸馏。我们报告了在不同压缩率下的权衡，包括墙钟时间、参数数量和乘加操作数量。我们的结果表明，与最近的方法相比，基本的压缩技术是强大的基准。我们进一步提出了几种压缩方法来改进模型的压缩效果。

    Despite the success of Transformers in self- supervised learning with applications to various downstream tasks, the computational cost of training and inference remains a major challenge for applying these models to a wide spectrum of devices. Several isolated attempts have been made to compress Transformers, but the settings and metrics are different across studies. Trade-off at various compression rates are also largely missing in prior work, making it difficult to compare compression techniques. In this work, we aim to provide context for the isolated results, studying several commonly used compression techniques, including weight pruning, head pruning, low-rank approximation, and knowledge distillation. We report trade- off at various compression rate, including wall-clock time, the number of parameters, and the number of multiply-accumulate operations. Our results show that compared to recent approaches, basic compression techniques are strong baselines. We further present several
    

