# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Using Contextual Information for Sentence-level Morpheme Segmentation](https://arxiv.org/abs/2403.15436) | 将形态素分割任务重新定义为序列到序列问题，并通过多语言模型展示出优异性能，揭示了高资源语言环境下的可比效力，以及低资源语言场景下的局限性。 |
| [^2] | [Enhancing Hokkien Dual Translation by Exploring and Standardizing of Four Writing Systems](https://arxiv.org/abs/2403.12024) | 通过开发双语翻译模型，探索台湾福建话和繁体中文/英文之间的翻译，引入限制单语语料库并将所有台湾福建话文字系统规范为福建话汉字，从而提高翻译性能。 |
| [^3] | [Efficient Pruning of Large Language Model with Adaptive Estimation Fusion](https://arxiv.org/abs/2403.10799) | 提出了一种简单而高效的剪枝方法，能够自适应地模拟每个子结构的重要性，并根据多层结构的结果自适应地融合粗粒度和细粒度的估计。 |
| [^4] | [VidProM: A Million-scale Real Prompt-Gallery Dataset for Text-to-Video Diffusion Models](https://arxiv.org/abs/2403.06098) | VidProM是一个包含167万个独特文本到视频提示的大规模数据集，对于文本到视频扩散模型带来了新的研究进展，揭示了真实用户提示对视频生成的重要性。 |
| [^5] | [Tell, Don't Show!: Language Guidance Eases Transfer Across Domains in Images and Videos](https://arxiv.org/abs/2403.05535) | 该论文提出了LaGTran框架，利用文本描述来引导知识转移，在处理具有挑战性的数据集上表现出显著优势。 |
| [^6] | [ParallelPARC: A Scalable Pipeline for Generating Natural-Language Analogies](https://arxiv.org/abs/2403.01139) | 设计了ParallelPARC流水线，利用大型语言模型生成复杂段落类比数据集，评估各种类比类型，并展示出人类在类比识别中的优势。 |
| [^7] | [Direct Alignment of Draft Model for Speculative Decoding with Chat-Fine-Tuned LLMs](https://arxiv.org/abs/2403.00858) | 通过提出的框架，我们训练了一种用于Llama 2 Chat 7B或更大模型的草案模型，实现了加速推理，仅占原始大小的1.64％。 |
| [^8] | [Where Visual Speech Meets Language: VSP-LLM Framework for Efficient and Context-Aware Visual Speech Processing](https://arxiv.org/abs/2402.15151) | 提出了一个新颖的VSP-LLM框架，用于最大化上下文建模能力，实现视觉语音识别和翻译的多任务执行。 |
| [^9] | [Are Sounds Sound for Phylogenetic Reconstruction?](https://arxiv.org/abs/2402.02807) | 本文通过对十个不同语言家族的多样数据集进行研究，首次在系统发育重建中比较了基于声音和基于同源的方法的表现。结果显示，基于词汇同源的重建谱系与真实谱系平均更接近，提高了约三分之一。 |
| [^10] | [Primacy Effect of ChatGPT.](http://arxiv.org/abs/2310.13206) | ChatGPT在选择答案时表现出初印象效应，即更倾向于选择提示中较早位置的标签作为答案。 |
| [^11] | [Language Models as Black-Box Optimizers for Vision-Language Models.](http://arxiv.org/abs/2309.05950) | 本论文介绍了一种新的视觉-语言模型 (VLMs) 微调方法，通过自然语言提示来避免访问模型参数，采用聊天式的语言模型作为黑盒优化器，在少样本图像分类任务中达到效果。 |
| [^12] | [Are Models Trained on Indian Legal Data Fair?.](http://arxiv.org/abs/2303.07247) | 本文从印度的法律领域出发，通过对在印地语法律文档上训练的模型在保释预测任务中的算法偏见传递进行了初步调查。结果表明，决策树模型在与印度教徒和穆斯林相关的输入特征上具有0.237的整体公平性差距。 |

# 详细

[^1]: 利用上下文信息进行句子级形态素分割

    Using Contextual Information for Sentence-level Morpheme Segmentation

    [https://arxiv.org/abs/2403.15436](https://arxiv.org/abs/2403.15436)

    将形态素分割任务重新定义为序列到序列问题，并通过多语言模型展示出优异性能，揭示了高资源语言环境下的可比效力，以及低资源语言场景下的局限性。

    

    最近形态素分割的发展主要强调单词级别的分割，通常忽视了句子内的上下文相关性。在这项研究中，我们将形态素分割任务重新定义为一个序列到序列的问题，将整个句子作为输入，而不是孤立地处理单个单词。我们的研究发现，多语言模型与单语模型相比始终表现出更高的性能。虽然我们的模型没有超越当前最先进的模型，但在高资源语言中展现出可比较的有效性，同时揭示了在低资源语言场景中的局限性。

    arXiv:2403.15436v1 Announce Type: new  Abstract: Recent advancements in morpheme segmentation primarily emphasize word-level segmentation, often neglecting the contextual relevance within the sentence. In this study, we redefine the morpheme segmentation task as a sequence-to-sequence problem, treating the entire sentence as input rather than isolating individual words. Our findings reveal that the multilingual model consistently exhibits superior performance compared to monolingual counterparts. While our model did not surpass the performance of the current state-of-the-art, it demonstrated comparable efficacy with high-resource languages while revealing limitations in low-resource language scenarios.
    
[^2]: 探索和规范四种文字系统，增强福建话的双语翻译

    Enhancing Hokkien Dual Translation by Exploring and Standardizing of Four Writing Systems

    [https://arxiv.org/abs/2403.12024](https://arxiv.org/abs/2403.12024)

    通过开发双语翻译模型，探索台湾福建话和繁体中文/英文之间的翻译，引入限制单语语料库并将所有台湾福建话文字系统规范为福建话汉字，从而提高翻译性能。

    

    arXiv:2403.12024v1 公告类型: 新 提要: 机器翻译主要集中在高资源语言（HRLs），而低资源语言（LRLs）如台湾福建话相对未被深入探索。本研究旨在通过开发台湾福建话与繁体中文和英文之间的双语翻译模型来解决这一差距。我们使用一个专门针对繁体中文的预训练LLaMA2-7B模型来利用台湾福建话汉字与繁体中文之间的拼音相似性。我们的全面实验涉及台湾福建话各种文字系统之间的翻译任务，以及台湾福建话与其他高资源语言之间的翻译。我们发现使用有限的单语语料库还进一步改善了模型对台湾福建话的能力。然后，我们利用我们的翻译模型将所有台湾福建话文字系统规范为福建话汉字，从而进一步提高性能。

    arXiv:2403.12024v1 Announce Type: new  Abstract: Machine translation focuses mainly on high-resource languages (HRLs), while low-resource languages (LRLs) like Taiwanese Hokkien are relatively under-explored. This study aims to address this gap by developing a dual translation model between Taiwanese Hokkien and both Traditional Mandarin Chinese and English. We employ a pre-trained LLaMA2-7B model specialized in Traditional Mandarin Chinese to leverage the orthographic similarities between Taiwanese Hokkien Han and Traditional Mandarin Chinese. Our comprehensive experiments involve translation tasks across various writing systems of Taiwanese Hokkien and between Taiwanese Hokkien and other HRLs. We find that the use of a limited monolingual corpus also further improve the model's Taiwanese Hokkien capabilities. We then utilize our translation model to standardize all Taiwanese Hokkien writing systems into Hokkien Han, resulting in further performance improvements. Additionally, we intr
    
[^3]: 使用自适应估计融合高效剪枝大型语言模型

    Efficient Pruning of Large Language Model with Adaptive Estimation Fusion

    [https://arxiv.org/abs/2403.10799](https://arxiv.org/abs/2403.10799)

    提出了一种简单而高效的剪枝方法，能够自适应地模拟每个子结构的重要性，并根据多层结构的结果自适应地融合粗粒度和细粒度的估计。

    

    大型语言模型（LLMs）已经成为许多生成性下游任务中至关重要的组成部分，这导致在资源受限设备上高效部署它们成为不可避免的趋势和重大挑战。结构化剪枝是解决这一挑战的广泛应用方法。然而，当处理多个解码器层的复杂结构时，通常的方法往往采用常见的估计方法进行剪枝。这些方法导致特定下游任务精度下降。本文介绍了一种简单而有效的方法，可自适应地模拟每个子结构的重要性。同时，它可以基于复杂和多层结构的结果，自适应地融合粗粒度和细粒度的估计。我们设计的所有方面都无缝集成到端到端的剪枝框架中。与主流数据集上的最先进方法相比，我们的实验结果表明

    arXiv:2403.10799v1 Announce Type: cross  Abstract: Large language models (LLMs) have become crucial for many generative downstream tasks, leading to an inevitable trend and significant challenge to deploy them efficiently on resource-constrained devices. Structured pruning is a widely used method to address this challenge. However, when dealing with the complex structure of the multiple decoder layers, general methods often employ common estimation approaches for pruning. These approaches lead to a decline in accuracy for specific downstream tasks. In this paper, we introduce a simple yet efficient method that adaptively models the importance of each substructure. Meanwhile, it can adaptively fuse coarse-grained and finegrained estimations based on the results from complex and multilayer structures. All aspects of our design seamlessly integrate into the endto-end pruning framework. Our experimental results, compared with state-of-the-art methods on mainstream datasets, demonstrate ave
    
[^4]: VidProM：一个百万规模的真实即时图库数据集，用于文本到视频扩散模型

    VidProM: A Million-scale Real Prompt-Gallery Dataset for Text-to-Video Diffusion Models

    [https://arxiv.org/abs/2403.06098](https://arxiv.org/abs/2403.06098)

    VidProM是一个包含167万个独特文本到视频提示的大规模数据集，对于文本到视频扩散模型带来了新的研究进展，揭示了真实用户提示对视频生成的重要性。

    

    Sora的到来标志着文本到视频扩散模型的新时代的到来，带来了视频生成和潜在应用方面的显著进步。然而，Sora以及其他文本到视频扩散模型高度依赖提示，但目前尚没有公开可用的包含文本到视频提示研究的数据集。本文介绍了VidProM，这是第一个由167万个来自真实用户的独特文本到视频提示组成的大规模数据集。此外，该数据集包括由四种最先进的扩散模型生成的669万个视频以及一些相关数据。我们首先展示了这一大规模数据集的策展过程，这是一个耗时且昂贵的过程。随后，我们展示了所提出的VidProM与DiffusionDB之间的区别，后者是一个用于图像生成的大规模提示图库数据集。通过对这些提示的分析，我们确定了一个专门的新提示数据集的必要性。

    arXiv:2403.06098v1 Announce Type: cross  Abstract: The arrival of Sora marks a new era for text-to-video diffusion models, bringing significant advancements in video generation and potential applications. However, Sora, as well as other text-to-video diffusion models, highly relies on the prompts, and there is no publicly available dataset featuring a study of text-to-video prompts. In this paper, we introduce VidProM, the first large-scale dataset comprising 1.67 million unique text-to-video prompts from real users. Additionally, the dataset includes 6.69 million videos generated by four state-of-the-art diffusion models and some related data. We initially demonstrate the curation of this large-scale dataset, which is a time-consuming and costly process. Subsequently, we show how the proposed VidProM differs from DiffusionDB, a large-scale prompt-gallery dataset for image generation. Based on the analysis of these prompts, we identify the necessity for a new prompt dataset specificall
    
[^5]: 讲述，而不是展示！：语言指导有助于在图像和视频领域之间进行转移

    Tell, Don't Show!: Language Guidance Eases Transfer Across Domains in Images and Videos

    [https://arxiv.org/abs/2403.05535](https://arxiv.org/abs/2403.05535)

    该论文提出了LaGTran框架，利用文本描述来引导知识转移，在处理具有挑战性的数据集上表现出显著优势。

    

    我们介绍了LaGTran，这是一个新颖的框架，利用即可获得或易于获取的文本描述，引导从带标签的源数据到具有域偏移的无标签目标数据的鲁棒性知识转移。受到我们观察到更富语义的文本模态具有更有利的转移特性的启发，我们设计了一个转移机制，使用源训练的文本分类器在目标文本描述上生成预测，并利用这些预测作为相应图像的监督。我们的方法以语言指导为驱动，出奇地简单易行，却在具有挑战性的数据集如GeoNet和DomainNet上显著优于以往所有方法，验证了其极其有效性。

    arXiv:2403.05535v1 Announce Type: cross  Abstract: We introduce LaGTran, a novel framework that utilizes readily available or easily acquired text descriptions to guide robust transfer of discriminative knowledge from labeled source to unlabeled target data with domain shifts. While unsupervised adaptation methods have been established to address this problem, they show limitations in handling challenging domain shifts due to their exclusive operation within the pixel-space. Motivated by our observation that semantically richer text modality has more favorable transfer properties, we devise a transfer mechanism to use a source-trained text-classifier to generate predictions on the target text descriptions, and utilize these predictions as supervision for the corresponding images. Our approach driven by language guidance is surprisingly easy and simple, yet significantly outperforms all prior approaches on challenging datasets like GeoNet and DomainNet, validating its extreme effectiven
    
[^6]: ParallelPARC: 生成自然语言类比的可扩展流水线

    ParallelPARC: A Scalable Pipeline for Generating Natural-Language Analogies

    [https://arxiv.org/abs/2403.01139](https://arxiv.org/abs/2403.01139)

    设计了ParallelPARC流水线，利用大型语言模型生成复杂段落类比数据集，评估各种类比类型，并展示出人类在类比识别中的优势。

    

    Analogy-making对于人类认知至关重要，使我们能够适应新颖情境--这是当前人工智能系统仍然缺乏的能力。大多数类比数据集今天关注简单的类比（例如，词类比）；包含复杂类型类比的数据集通常是手工策划的，并且非常小。我们认为这限制了计算类比的进展。在这项工作中，我们设计了一个数据生成流水线，ParallelPARC（Parallel Paragraph Creator），利用最先进的大型语言模型（LLM）来创建基于段落的复杂类比，以及简单和具有挑战性的干扰项。我们展示了我们的流水线，并创建了ProPara-Logy，一个关于科学过程间类比的数据集。我们发布了一个由人类验证过的金标准数据集，以及一个自动生成的银标准数据集。我们在二进制和多选环境中测试了LLMs和人类对类比的识别，发现人类胜过最佳模型。

    arXiv:2403.01139v1 Announce Type: cross  Abstract: Analogy-making is central to human cognition, allowing us to adapt to novel situations -- an ability that current AI systems still lack. Most analogy datasets today focus on simple analogies (e.g., word analogies); datasets including complex types of analogies are typically manually curated and very small. We believe that this holds back progress in computational analogy. In this work, we design a data generation pipeline, ParallelPARC (Parallel Paragraph Creator) leveraging state-of-the-art Large Language Models (LLMs) to create complex, paragraph-based analogies, as well as distractors, both simple and challenging. We demonstrate our pipeline and create ProPara-Logy, a dataset of analogies between scientific processes. We publish a gold-set, validated by humans, and a silver-set, generated automatically. We test LLMs' and humans' analogy recognition in binary and multiple-choice settings, and found that humans outperform the best mod
    
[^7]: 直接与Chat-Fine-Tuned LLMs的草案模型对齐

    Direct Alignment of Draft Model for Speculative Decoding with Chat-Fine-Tuned LLMs

    [https://arxiv.org/abs/2403.00858](https://arxiv.org/abs/2403.00858)

    通过提出的框架，我们训练了一种用于Llama 2 Chat 7B或更大模型的草案模型，实现了加速推理，仅占原始大小的1.64％。

    

    文本生成与大型语言模型（LLMs）由于其自回归本质、巨大的参数数量和有限的内存带宽而被认为是内存密集型，通常导致低令牌速率。猜测解码已被提出作为LLM推理加速的解决方案。然而，在现代开源LLM系列中，例如Llama 2 7B，由于草案模型通常不可用，因此需要训练高质量的草案模型以通过猜测解码实现推理加速。在本文中，我们提出了一个简单的草案模型训练框架，用于直接与Chat-capable目标模型对齐。通过我们提出的框架，我们训练出Llama 2 Chat Drafter 115M，这是一个适用于Llama 2 Chat 7B或更大模型的草案模型，仅占原始大小的1.64％。我们的训练框架仅包括预训练、蒸馏数据集生成和使用知识蒸馏进行微调，没有额外的对齐步骤。

    arXiv:2403.00858v1 Announce Type: cross  Abstract: Text generation with Large Language Models (LLMs) is known to be memory bound due to the combination of their auto-regressive nature, huge parameter counts, and limited memory bandwidths, often resulting in low token rates. Speculative decoding has been proposed as a solution for LLM inference acceleration. However, since draft models are often unavailable in the modern open-source LLM families, e.g., for Llama 2 7B, training a high-quality draft model is required to enable inference acceleration via speculative decoding. In this paper, we propose a simple draft model training framework for direct alignment to chat-capable target models. With the proposed framework, we train Llama 2 Chat Drafter 115M, a draft model for Llama 2 Chat 7B or larger, with only 1.64\% of the original size. Our training framework only consists of pretraining, distillation dataset generation, and finetuning with knowledge distillation, with no additional align
    
[^8]: 视觉语音遇见语言：VSP-LLM框架用于高效和上下文感知的视觉语音处理

    Where Visual Speech Meets Language: VSP-LLM Framework for Efficient and Context-Aware Visual Speech Processing

    [https://arxiv.org/abs/2402.15151](https://arxiv.org/abs/2402.15151)

    提出了一个新颖的VSP-LLM框架，用于最大化上下文建模能力，实现视觉语音识别和翻译的多任务执行。

    

    在视觉语音处理中，由于唇部运动的模糊性质，上下文建模能力是最重要的要求之一。例如，同音异义词，即具有相同唇部运动但产生不同声音的单词，可以通过考虑上下文来区分。本文提出了一种新颖的框架，称为集成LLM的视觉语音处理（VSP-LLM），通过引入LLM的强大能力来最大化上下文建模能力。具体来说，VSP-LLM旨在执行视觉语音识别和翻译的多任务，其中给定的指令控制任务类型。通过利用自监督视觉语音模型，将输入视频映射到LLM的输入潜在空间。针对输入帧存在冗余信息的事实，我们提出了一种新颖的去重方法，通过使用视觉语音单元减少嵌入的视觉特征。

    arXiv:2402.15151v1 Announce Type: cross  Abstract: In visual speech processing, context modeling capability is one of the most important requirements due to the ambiguous nature of lip movements. For example, homophenes, words that share identical lip movements but produce different sounds, can be distinguished by considering the context. In this paper, we propose a novel framework, namely Visual Speech Processing incorporated with LLMs (VSP-LLM), to maximize the context modeling ability by bringing the overwhelming power of LLMs. Specifically, VSP-LLM is designed to perform multi-tasks of visual speech recognition and translation, where the given instructions control the type of task. The input video is mapped to the input latent space of a LLM by employing a self-supervised visual speech model. Focused on the fact that there is redundant information in input frames, we propose a novel deduplication method that reduces the embedded visual features by employing visual speech units. Thr
    
[^9]: 声音对于系统发育重建可靠吗？

    Are Sounds Sound for Phylogenetic Reconstruction?

    [https://arxiv.org/abs/2402.02807](https://arxiv.org/abs/2402.02807)

    本文通过对十个不同语言家族的多样数据集进行研究，首次在系统发育重建中比较了基于声音和基于同源的方法的表现。结果显示，基于词汇同源的重建谱系与真实谱系平均更接近，提高了约三分之一。

    

    在传统的语言进化研究中，学者们通常强调声音规律和对应关系对于语言家族谱系推断的重要性。然而，迄今为止，计算方法往往没有充分考虑到这一潜力。大多数计算方法仍然依赖于词汇同源作为语言学系统发育重建的主要数据来源，尽管也有一些研究中的作者赞赏比较声音序列的好处。基于十个来自不同语言家族的多样数据集和现代自动同源和声音对应检测方法，我们首次测试了基于声音和基于同源的方法在系统发育重建中的性能。结果表明，通过词汇同源重建的谱系在广义四元组距离上与真实谱系平均更接近，提升了约三分之一。

    In traditional studies on language evolution, scholars often emphasize the importance of sound laws and sound correspondences for phylogenetic inference of language family trees. However, to date, computational approaches have typically not taken this potential into account. Most computational studies still rely on lexical cognates as major data source for phylogenetic reconstruction in linguistics, although there do exist a few studies in which authors praise the benefits of comparing words at the level of sound sequences. Building on (a) ten diverse datasets from different language families, and (b) state-of-the-art methods for automated cognate and sound correspondence detection, we test, for the first time, the performance of sound-based versus cognate-based approaches to phylogenetic reconstruction. Our results show that phylogenies reconstructed from lexical cognates are topologically closer, by approximately one third with respect to the generalized quartet distance on average, 
    
[^10]: ChatGPT的初印象效应

    Primacy Effect of ChatGPT. (arXiv:2310.13206v1 [cs.CL])

    [http://arxiv.org/abs/2310.13206](http://arxiv.org/abs/2310.13206)

    ChatGPT在选择答案时表现出初印象效应，即更倾向于选择提示中较早位置的标签作为答案。

    

    通过在ChatGPT上进行实验和分析，我们研究了ChatGPT的初印象效应，即选择较早位置的标签作为答案的倾向。我们发现：i）ChatGPT的决策对提示中标签的顺序敏感；ii）ChatGPT更倾向于选择较早位置的标签作为答案。我们希望我们的实验和分析能为构建更可靠的基于ChatGPT的解决方案提供额外的见解。

    Instruction-tuned large language models (LLMs), such as ChatGPT, have led to promising zero-shot performance in discriminative natural language understanding (NLU) tasks. This involves querying the LLM using a prompt containing the question, and the candidate labels to choose from. The question-answering capabilities of ChatGPT arise from its pre-training on large amounts of human-written text, as well as its subsequent fine-tuning on human preferences, which motivates us to ask: Does ChatGPT also inherits humans' cognitive biases? In this paper, we study the primacy effect of ChatGPT: the tendency of selecting the labels at earlier positions as the answer. We have two main findings: i) ChatGPT's decision is sensitive to the order of labels in the prompt; ii) ChatGPT has a clearly higher chance to select the labels at earlier positions as the answer. We hope that our experiments and analyses provide additional insights into building more reliable ChatGPT-based solutions. We release the
    
[^11]: 语言模型作为视觉-语言模型的黑盒优化器

    Language Models as Black-Box Optimizers for Vision-Language Models. (arXiv:2309.05950v1 [cs.CL])

    [http://arxiv.org/abs/2309.05950](http://arxiv.org/abs/2309.05950)

    本论文介绍了一种新的视觉-语言模型 (VLMs) 微调方法，通过自然语言提示来避免访问模型参数，采用聊天式的语言模型作为黑盒优化器，在少样本图像分类任务中达到效果。

    

    预训练在大规模网络数据集上的视觉-语言模型 (VLMs) 展示了在各种视觉和多模态任务中的显著能力。目前，VLMs 的微调方法主要在白盒环境中操作，需要访问模型参数进行反向传播。然而，许多 VLMs 依赖于专有数据且不开源，限制了使用白盒方法进行微调。鉴于像 ChatGPT 这样的受欢迎私有大型语言模型 (LLMs) 仍然提供基于语言的用户界面，我们旨在通过自然语言提示开发一种新的 VLMs 微调方法，从而避免访问模型参数、特征嵌入或输出 logits 的需要。在这种设置下，我们提出使用基于聊天的 LLMs 作为黑盒优化器，以在使用 CLIP 进行少样本图像分类的示例任务中寻找最佳文本提示。具体而言，我们采用自动"爬山"程序，它能收敛到有效的提示上。

    Vision-language models (VLMs) pre-trained on web-scale datasets have demonstrated remarkable capabilities across a variety of vision and multimodal tasks. Currently, fine-tuning methods for VLMs mainly operate in a white-box setting, requiring access to model parameters for backpropagation. However, many VLMs rely on proprietary data and are not open-source, which restricts the use of white-box approaches for fine-tuning. Given that popular private large language models (LLMs) like ChatGPT still offer a language-based user interface, we aim to develop a novel fine-tuning approach for VLMs through natural language prompts, thereby avoiding the need to access model parameters, feature embeddings, or output logits. In this setup, we propose employing chat-based LLMs as black-box optimizers to search for the best text prompt on the illustrative task of few-shot image classification using CLIP. Specifically, we adopt an automatic "hill-climbing" procedure that converges on an effective prom
    
[^12]: 印度法律数据训练的模型是否公平？

    Are Models Trained on Indian Legal Data Fair?. (arXiv:2303.07247v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2303.07247](http://arxiv.org/abs/2303.07247)

    本文从印度的法律领域出发，通过对在印地语法律文档上训练的模型在保释预测任务中的算法偏见传递进行了初步调查。结果表明，决策树模型在与印度教徒和穆斯林相关的输入特征上具有0.237的整体公平性差距。

    

    自然语言处理和人工智能的最新进展与应用在多个领域（如法律、医疗和心理健康）取得了很大成功。最近提出了基于人工智能的语言模型（如判决预测）用于法律领域。然而，这些模型从训练数据中捕捉到了社会偏见。虽然NLP领域的偏见和公平性已经得到研究，但大多数研究主要定位在西方背景下。本文从印度的法律领域出发，对公平性进行了初步调查。我们重点研究了在印地语法律文档上训练的模型在保释预测任务中传递学习到的算法偏见。我们使用群体平等评估公平性差距，并展示了一个决策树模型在保释预测任务中，在与印度教徒和穆斯林相关的输入特征上具有0.237的整体公平性差距。此外，我们强调了对印度法律领域的公平性研究的必要性。

    Recent advances and applications of language technology and artificial intelligence have enabled much success across multiple domains like law, medical and mental health. AI-based Language Models, like Judgement Prediction, have recently been proposed for the legal sector. However, these models are strife with encoded social biases picked up from the training data. While bias and fairness have been studied across NLP, most studies primarily locate themselves within a Western context. In this work, we present an initial investigation of fairness from the Indian perspective in the legal domain. We highlight the propagation of learnt algorithmic biases in the bail prediction task for models trained on Hindi legal documents. We evaluate the fairness gap using demographic parity and show that a decision tree model trained for the bail prediction task has an overall fairness disparity of 0.237 between input features associated with Hindus and Muslims. Additionally, we highlight the need for 
    

