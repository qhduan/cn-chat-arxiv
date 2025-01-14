# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Linear Attention Sequence Parallelism](https://arxiv.org/abs/2404.02882) | 提出了一种名为线性注意力序列并行（LASP）的高效序列并行方法，针对线性注意力的语言模型进行了优化，通过设计高效的点对点通信机制和执行内核融合来降低通信开销，并实现硬件友好性。 |
| [^2] | [Are LLMs Good Cryptic Crossword Solvers?](https://arxiv.org/abs/2403.12094) | 本文建立了三种流行LLMs的基准结果，表明它们在难解填字游戏上的表现仍远远不及人类。 |
| [^3] | [Efficient Pruning of Large Language Model with Adaptive Estimation Fusion](https://arxiv.org/abs/2403.10799) | 提出了一种简单而高效的剪枝方法，能够自适应地模拟每个子结构的重要性，并根据多层结构的结果自适应地融合粗粒度和细粒度的估计。 |
| [^4] | [Images are Achilles' Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking Multimodal Large Language Models](https://arxiv.org/abs/2403.09792) | 论文研究了多模态大语言模型对齐问题，揭示了图像输入对模型的漏洞，并提出了一种利用图像隐藏恶意意图、成功破解现有模型的方法。 |
| [^5] | [DropBP: Accelerating Fine-Tuning of Large Language Models by Dropping Backward Propagation](https://arxiv.org/abs/2402.17812) | DropBP提出了一种新颖的方式来加速大型语言模型的微调，通过在反向传播过程中随机丢弃层以减少计算成本同时保持准确性。 |
| [^6] | [SemPLeS: Semantic Prompt Learning for Weakly-Supervised Semantic Segmentation](https://arxiv.org/abs/2401.11791) | SemPLeS框架利用语义提示学习解决弱监督语义分割中的问题，通过学习有效提示来增强分割区域与目标对象类别之间的语义对齐。 |
| [^7] | [Generating Mathematical Derivations with Large Language Models.](http://arxiv.org/abs/2307.09998) | 本文利用大型语言模型生成数学导出，分析了微调模型对未见符号和方程结构更改的敏感性，结果表明微调的FLAN-T5-large（MathT5）在各个测试集上的绝对性能优于GPT模型。 |
| [^8] | [Quilt-1M: One Million Image-Text Pairs for Histopathology.](http://arxiv.org/abs/2306.11207) | 本文介绍了一个名为 Quilt-1M 的癌症组织学图像和文字对的百万数据集，并利用 YouTube 上的专家医生教程视频为主要来源。这个数据集将使得癌症组织学领域的表征学习取得类似于其他领域的进展。 |
| [^9] | [Privacy-Preserving Prompt Tuning for Large Language Model Services.](http://arxiv.org/abs/2305.06212) | RAPT是一个提供隐私保证的大语言模型服务的提示调整框架，采用本地差分隐私设置和新颖的隐私化标记重建任务，并在多种任务中取得有竞争力的性能和良好的隐私保护效果。 |
| [^10] | [ChatGPT Needs SPADE (Sustainability, PrivAcy, Digital divide, and Ethics) Evaluation: A Review.](http://arxiv.org/abs/2305.03123) | 本文研究关注ChatGPT面临的可持续性、隐私、数字鸿沟和伦理问题，提出了SPADE评估的必要性，并给出了缓解和建议。 |

# 详细

[^1]: 线性注意力序列并行化

    Linear Attention Sequence Parallelism

    [https://arxiv.org/abs/2404.02882](https://arxiv.org/abs/2404.02882)

    提出了一种名为线性注意力序列并行（LASP）的高效序列并行方法，针对线性注意力的语言模型进行了优化，通过设计高效的点对点通信机制和执行内核融合来降低通信开销，并实现硬件友好性。

    

    序列并行（SP）作为一种处理超出单个GPU内存限制的长序列的流行策略。然而，现有的SP方法并未利用线性注意力特性，导致在基于线性注意力的语言模型中并行效率和可用性不佳。在本文中，我们介绍了线性注意力序列并行（LASP），这是一种专为基于线性注意力的语言模型量身定制的高效SP方法。具体来说，我们设计了一种高效的点对点通信机制，以利用线性注意力的右乘内核技巧，从而显着降低SP的通信开销。我们还通过执行内核融合和中间状态缓存来增强LASP的实际效率，使LASP在GPU集群上的硬件友好性得到提升。此外，我们还精心确保序列级LASP与所有类型的批级数据兼容。

    arXiv:2404.02882v1 Announce Type: cross  Abstract: Sequence Parallel (SP) serves as a prevalent strategy to handle long sequences that exceed the memory limit of a single GPU. However, existing SP methods do not take advantage of linear attention features, resulting in sub-optimal parallelism efficiency and usability for linear attention-based language models. In this paper, we introduce Linear Attention Sequence Parallel (LASP), an efficient SP method tailored to linear attention-based language models. Specifically, we design an efficient point-to-point communication mechanism to leverage the right-product kernel trick of linear attention, which sharply decreases the communication overhead of SP. We also enhance the practical efficiency of LASP by performing kernel fusion and intermediate state caching, making the implementation of LASP hardware-friendly on GPU clusters. Furthermore, we meticulously ensure the compatibility of sequence-level LASP with all types of batch-level data par
    
[^2]: LLMs是一个好的难解填字游戏求解器吗？

    Are LLMs Good Cryptic Crossword Solvers?

    [https://arxiv.org/abs/2403.12094](https://arxiv.org/abs/2403.12094)

    本文建立了三种流行LLMs的基准结果，表明它们在难解填字游戏上的表现仍远远不及人类。

    

    难解填字游戏是一种谜题，不仅依赖于一般知识，还依赖于求解者在不同层面上操纵语言并处理各种类型的文字游戏。先前的研究表明，即使对于现代NLP模型来说，解决这类谜题也是一项挑战。然而，大型语言模型（LLMs）的能力尚未在这一任务上进行测试。在本文中，我们为三种流行的LLMs -- LLaMA2、Mistral和ChatGPT建立了基准结果，显示它们在这一任务上的表现仍远远不及人类。

    arXiv:2403.12094v1 Announce Type: new  Abstract: Cryptic crosswords are puzzles that rely not only on general knowledge but also on the solver's ability to manipulate language on different levels and deal with various types of wordplay. Previous research suggests that solving such puzzles is a challenge even for modern NLP models. However, the abilities of large language models (LLMs) have not yet been tested on this task. In this paper, we establish the benchmark results for three popular LLMs -- LLaMA2, Mistral, and ChatGPT -- showing that their performance on this task is still far from that of humans.
    
[^3]: 使用自适应估计融合高效剪枝大型语言模型

    Efficient Pruning of Large Language Model with Adaptive Estimation Fusion

    [https://arxiv.org/abs/2403.10799](https://arxiv.org/abs/2403.10799)

    提出了一种简单而高效的剪枝方法，能够自适应地模拟每个子结构的重要性，并根据多层结构的结果自适应地融合粗粒度和细粒度的估计。

    

    大型语言模型（LLMs）已经成为许多生成性下游任务中至关重要的组成部分，这导致在资源受限设备上高效部署它们成为不可避免的趋势和重大挑战。结构化剪枝是解决这一挑战的广泛应用方法。然而，当处理多个解码器层的复杂结构时，通常的方法往往采用常见的估计方法进行剪枝。这些方法导致特定下游任务精度下降。本文介绍了一种简单而有效的方法，可自适应地模拟每个子结构的重要性。同时，它可以基于复杂和多层结构的结果，自适应地融合粗粒度和细粒度的估计。我们设计的所有方面都无缝集成到端到端的剪枝框架中。与主流数据集上的最先进方法相比，我们的实验结果表明

    arXiv:2403.10799v1 Announce Type: cross  Abstract: Large language models (LLMs) have become crucial for many generative downstream tasks, leading to an inevitable trend and significant challenge to deploy them efficiently on resource-constrained devices. Structured pruning is a widely used method to address this challenge. However, when dealing with the complex structure of the multiple decoder layers, general methods often employ common estimation approaches for pruning. These approaches lead to a decline in accuracy for specific downstream tasks. In this paper, we introduce a simple yet efficient method that adaptively models the importance of each substructure. Meanwhile, it can adaptively fuse coarse-grained and finegrained estimations based on the results from complex and multilayer structures. All aspects of our design seamlessly integrate into the endto-end pruning framework. Our experimental results, compared with state-of-the-art methods on mainstream datasets, demonstrate ave
    
[^4]: 图像是对齐的软肋：利用视觉漏洞破解多模态大语言模型

    Images are Achilles' Heel of Alignment: Exploiting Visual Vulnerabilities for Jailbreaking Multimodal Large Language Models

    [https://arxiv.org/abs/2403.09792](https://arxiv.org/abs/2403.09792)

    论文研究了多模态大语言模型对齐问题，揭示了图像输入对模型的漏洞，并提出了一种利用图像隐藏恶意意图、成功破解现有模型的方法。

    

    在这篇论文中，我们研究了多模态大语言模型（MLLMs）的无害对齐问题。我们对代表性的MLLMs的无害性能进行了系统的实证分析，并揭示了图像输入对MLLMs造成的对齐漏洞。受此启发，我们提出了一种名为HADES的新型破解方法，利用精心制作的图像隐藏和放大文本输入中的恶意意图的有害性。实验结果表明，HADES可以有效地破解现有的MLLMs，为LLaVA-1.5实现了90.26％的平均攻击成功率（ASR），为Gemini Pro Vision实现了71.60％的ASR。我们的代码和数据将公开发布。

    arXiv:2403.09792v1 Announce Type: cross  Abstract: In this paper, we study the harmlessness alignment problem of multimodal large language models~(MLLMs). We conduct a systematic empirical analysis of the harmlessness performance of representative MLLMs and reveal that the image input poses the alignment vulnerability of MLLMs. Inspired by this, we propose a novel jailbreak method named HADES, which hides and amplifies the harmfulness of the malicious intent within the text input, using meticulously crafted images. Experimental results show that HADES can effectively jailbreak existing MLLMs, which achieves an average Attack Success Rate~(ASR) of 90.26% for LLaVA-1.5 and 71.60% for Gemini Pro Vision. Our code and data will be publicly released.
    
[^5]: DropBP：通过丢弃反向传播加速大型语言模型的微调

    DropBP: Accelerating Fine-Tuning of Large Language Models by Dropping Backward Propagation

    [https://arxiv.org/abs/2402.17812](https://arxiv.org/abs/2402.17812)

    DropBP提出了一种新颖的方式来加速大型语言模型的微调，通过在反向传播过程中随机丢弃层以减少计算成本同时保持准确性。

    

    训练深度神经网络通常涉及正向和反向传播过程中的大量计算成本。传统的层次丢弃技术在训练过程中丢弃某些层以减少计算负担。然而，在正向传播过程中丢弃层会对训练过程产生不利影响，降低准确性。本文提出了DropBP，这是一种旨在减少计算成本同时保持准确性的新方法。DropBP在反向传播过程中随机丢弃层，不影响正向传播。此外，DropBP计算每个层的敏感性以分配适当的丢失率，从而稳定训练过程。DropBP旨在通过反向传播增强训练过程的效率，从而加速使用反向传播进行完全微调和参数高效微调。

    arXiv:2402.17812v1 Announce Type: cross  Abstract: Training deep neural networks typically involves substantial computational costs during both forward and backward propagation. The conventional layer dropping techniques drop certain layers during training for reducing the computations burden. However, dropping layers during forward propagation adversely affects the training process by degrading accuracy. In this paper, we propose Dropping Backward Propagation (DropBP), a novel approach designed to reduce computational costs while maintaining accuracy. DropBP randomly drops layers during the backward propagation, which does not deviate forward propagation. Moreover, DropBP calculates the sensitivity of each layer to assign appropriate drop rate, thereby stabilizing the training process. DropBP is designed to enhance the efficiency of the training process with backpropagation, thereby enabling the acceleration of both full fine-tuning and parameter-efficient fine-tuning using backpropag
    
[^6]: SemPLeS: 语义提示学习用于弱监督语义分割

    SemPLeS: Semantic Prompt Learning for Weakly-Supervised Semantic Segmentation

    [https://arxiv.org/abs/2401.11791](https://arxiv.org/abs/2401.11791)

    SemPLeS框架利用语义提示学习解决弱监督语义分割中的问题，通过学习有效提示来增强分割区域与目标对象类别之间的语义对齐。

    

    弱监督语义分割（WSSS）旨在利用仅具有图像级监督的图像数据来训练分割模型。由于无法获得精确的像素级标注，现有方法通常侧重于通过优化CAM样式的热图来生成用于训练分割模型的伪标记。然而，生成的热图可能仅捕获对象类别的具有区分性的图像区域或相关的共同出现的背景。为解决这些问题，我们提出了一种用于WSSS的语义提示学习（SemPLeS）框架，该框架学习有效地提示CLIP潜空间以增强分割区域与目标对象类别之间的语义对准。具体而言，我们提出了对比提示学习和提示引导的语义细化，以学习适当描述和抑制与每个目标对象类别相关的共同出现的背景的提示。

    arXiv:2401.11791v2 Announce Type: replace-cross  Abstract: Weakly-Supervised Semantic Segmentation (WSSS) aims to train segmentation models using image data with only image-level supervision. Since precise pixel-level annotations are not accessible, existing methods typically focus on producing pseudo masks for training segmentation models by refining CAM-like heatmaps. However, the produced heatmaps may capture only the discriminative image regions of object categories or the associated co-occurring backgrounds. To address the issues, we propose a Semantic Prompt Learning for WSSS (SemPLeS) framework, which learns to effectively prompt the CLIP latent space to enhance the semantic alignment between the segmented regions and the target object categories. More specifically, we propose Contrastive Prompt Learning and Prompt-guided Semantic Refinement to learn the prompts that adequately describe and suppress the co-occurring backgrounds associated with each target object category. In thi
    
[^7]: 用大型语言模型生成数学导出

    Generating Mathematical Derivations with Large Language Models. (arXiv:2307.09998v1 [cs.CL])

    [http://arxiv.org/abs/2307.09998](http://arxiv.org/abs/2307.09998)

    本文利用大型语言模型生成数学导出，分析了微调模型对未见符号和方程结构更改的敏感性，结果表明微调的FLAN-T5-large（MathT5）在各个测试集上的绝对性能优于GPT模型。

    

    利用大型语言模型（LLM）在专业领域中生成数学结果的导出是一个新兴的研究方向，可以帮助识别模型的局限性，并有可能支持数学发现。本文利用符号引擎在大规模上生成方程的导出，并研究了LLM在从前提中导出目标方程时的能力。具体而言，我们采用上下文学习来对GPT进行训练，并对一系列T5模型进行了微调，以比较预训练策略对专门模型的鲁棒性和泛化能力。实证结果表明，经过微调的FLAN-T5-large（MathT5）在所有静态和超出分布的测试集上的绝对性能优于GPT模型。然而，深入分析表明，微调模型对涉及未见符号的扰动（以及在较小程度上的方程结构更改）更为敏感。此外，我们分析了1.7K个方程和200多个导出以凸显出LLM的局限性。

    The derivation of mathematical results in specialised fields using Large Language Models (LLMs) is an emerging research direction that can help identify models' limitations, and potentially support mathematical discovery. In this paper, we leverage a symbolic engine to generate derivations of equations at scale, and investigate the capabilities of LLMs when deriving goal equations from premises. Specifically, we employ in-context learning for GPT and fine-tune a range of T5 models to compare the robustness and generalisation of pre-training strategies to specialised models. Empirical results show that fine-tuned FLAN-T5-large (MathT5) outperforms GPT models on all static and out-of-distribution test sets in terms of absolute performance. However, an in-depth analysis reveals that the fine-tuned models are more sensitive to perturbations involving unseen symbols and (to a lesser extent) changes to equation structure. In addition, we analyse 1.7K equations and over 200 derivations to hig
    
[^8]: Quilt-1M: 癌症组织学图像文字对的百万数据集

    Quilt-1M: One Million Image-Text Pairs for Histopathology. (arXiv:2306.11207v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2306.11207](http://arxiv.org/abs/2306.11207)

    本文介绍了一个名为 Quilt-1M 的癌症组织学图像和文字对的百万数据集，并利用 YouTube 上的专家医生教程视频为主要来源。这个数据集将使得癌症组织学领域的表征学习取得类似于其他领域的进展。

    

    多模态应用的加速使得在线图像和文字数据大量涌现，但医学领域（特别是癌症组织学）类似的数据却很稀少，这阻碍了医学领域的进展。本文利用YouTube上的专家医生教程视频，从中选择了 1,087 小时的医学组织学视频，以此自动筛选出共包含 768,826 个癌症组织学图像及其对应的文字对的 Quilt 数据集。

    Recent accelerations in multi-modal applications have been made possible with the plethora of image and text data available online. However, the scarcity of analogous data in the medical field, specifically in histopathology, has halted comparable progress. To enable similar representation learning for histopathology, we turn to YouTube, an untapped resource of videos, offering $1,087$ hours of valuable educational histopathology videos from expert clinicians. From YouTube, we curate Quilt: a large-scale vision-language dataset consisting of $768,826$ image and text pairs. Quilt was automatically curated using a mixture of models, including large language models, handcrafted algorithms, human knowledge databases, and automatic speech recognition. In comparison, the most comprehensive datasets curated for histopathology amass only around $200$K samples. We combine Quilt with datasets from other sources, including Twitter, research papers, and the internet in general, to create an even l
    
[^9]: 大语言模型服务的隐私保护提示调整

    Privacy-Preserving Prompt Tuning for Large Language Model Services. (arXiv:2305.06212v1 [cs.CL])

    [http://arxiv.org/abs/2305.06212](http://arxiv.org/abs/2305.06212)

    RAPT是一个提供隐私保证的大语言模型服务的提示调整框架，采用本地差分隐私设置和新颖的隐私化标记重建任务，并在多种任务中取得有竞争力的性能和良好的隐私保护效果。

    

    提示调整为用户在新兴的大语言模型服务场景下使用其私有数据自定义大语言模型(LLM)的有效方式。但是，私有数据的敏感性需要在LLM服务定制中保护隐私。基于提示调整，我们提出了一种名为隐私保护提示调整(RAPT)的框架，为LLM服务提供隐私保证。RAPT采用本地隐私设置，允许用户使用本地差分隐私对其数据进行本地化隐私处理。由于在直接训练隐私化数据的情况下，提示调整表现不佳，因此我们引入了一种新颖的隐私化标记重建任务，与下游任务一起进行培训，使LLM学习更好的任务相关表示。尽管我们的框架简单，但实验表明，RAPT在各种任务中均具有竞争力的性能，并提供抵御对手的隐私保证。

    Prompt tuning provides an efficient way for users to customize Large Language Models (LLMs) with their private data in the emerging LLM service scenario. However, the sensitive nature of private data brings the need for privacy preservation in LLM service customization. Based on prompt tuning, we propose Privacy-Preserving Prompt Tuning (RAPT), a framework that provides privacy guarantees for LLM services. \textsc{rapt} adopts a local privacy setting, allowing users to privatize their data locally with local differential privacy. As prompt tuning performs poorly when directly trained on privatized data, we introduce a novel privatized token reconstruction task that is trained jointly with the downstream task, allowing LLMs to learn better task-dependent representations. Despite the simplicity of our framework, experiments show that RAPT achieves competitive performance across tasks while providing privacy guarantees against adversaries.
    
[^10]: ChatGPT 需要进行SPADE（可持续性、隐私、数字鸿沟和伦理）评估：一项综述。

    ChatGPT Needs SPADE (Sustainability, PrivAcy, Digital divide, and Ethics) Evaluation: A Review. (arXiv:2305.03123v1 [cs.CY])

    [http://arxiv.org/abs/2305.03123](http://arxiv.org/abs/2305.03123)

    本文研究关注ChatGPT面临的可持续性、隐私、数字鸿沟和伦理问题，提出了SPADE评估的必要性，并给出了缓解和建议。

    

    ChatGPT是另一个大型语言模型（LLM），由于其性能和有效的对话能力，在研究和工业界中得到了巨大的关注。最近，许多研究已经发表，以展示ChatGPT和其他LLMs的有效性、效率、集成和情感。相反，本研究关注的是大多数被忽视的重要方面，即可持续性、隐私、数字鸿沟和伦理，并建议不仅仅是ChatGPT，而是在对话机器人类别中的每一个后续入口都应该进行SPADE评估。本文详细讨论了关于ChatGPT的问题和关注点与上述特征一致。我们通过一些初步的数据收集和可视化以及假设的事实来支持我们的假设。我们还为每个问题提出了缓解和建议。此外，我们还提供了一些未来方向和开放问题的探讨。

    ChatGPT is another large language model (LLM) inline but due to its performance and ability to converse effectively, it has gained a huge popularity amongst research as well as industrial community. Recently, many studies have been published to show the effectiveness, efficiency, integration, and sentiments of chatGPT and other LLMs. In contrast, this study focuses on the important aspects that are mostly overlooked, i.e. sustainability, privacy, digital divide, and ethics and suggests that not only chatGPT but every subsequent entry in the category of conversational bots should undergo Sustainability, PrivAcy, Digital divide, and Ethics (SPADE) evaluation. This paper discusses in detail about the issues and concerns raised over chatGPT in line with aforementioned characteristics. We support our hypothesis by some preliminary data collection and visualizations along with hypothesized facts. We also suggest mitigations and recommendations for each of the concerns. Furthermore, we also s
    

