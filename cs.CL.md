# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [PaperWeaver: Enriching Topical Paper Alerts by Contextualizing Recommended Papers with User-collected Papers](https://arxiv.org/abs/2403.02939) | PaperWeaver通过将用户收集的论文与推荐论文上下文化，为研究人员提供了更丰富的主题论文提醒 |
| [^2] | [DECIDER: A Rule-Controllable Decoding Strategy for Language Generation by Imitating Dual-System Cognitive Theory](https://arxiv.org/abs/2403.01954) | DECIDER是一种受双系统认知理论启发的规则可控解码策略，通过在预训练语言模型中引入逻辑推理器，有效地遵循给定规则以引导生成方向朝向目标。 |
| [^3] | [Differentially Private Zeroth-Order Methods for Scalable Large Language Model Finetuning](https://arxiv.org/abs/2402.07818) | 本文研究了差分隐私零阶方法在大型语言模型微调中的应用，该方法通过使用零阶梯度来避免传统优化方法的可扩展性瓶颈，实现了在隐私、效用和可扩展性之间的良好平衡。 |
| [^4] | [Multimodal Clinical Trial Outcome Prediction with Large Language Models](https://arxiv.org/abs/2402.06512) | 本研究提出了一种名为LIFTED的多模态临床试验结果预测方法，通过将不同模态数据转化为自然语言描述来统一数据，并构建统一的抗噪声编码器进行信息提取。 |
| [^5] | [FAQ-Gen: An automated system to generate domain-specific FAQs to aid content comprehension](https://arxiv.org/abs/2402.05812) | 本文介绍了FAQ-Gen，一个利用文本转文本转换模型开发的自动化系统，用于生成特定领域的常见问题解答。该系统通过使用自我策划的算法来优化信息表示和问题-答案排名，提高了FAQ的准确性和相关性。定性人类评估表明生成的FAQs构造良好且可推广至不同领域。 |
| [^6] | [Large Human Language Models: A Need and the Challenges](https://arxiv.org/abs/2312.07751) | 大型人类语言模型的建立需要更好地整合人类背景，并面临着如何捕捉人类因素、如何表示以及如何建模的一系列挑战。 |
| [^7] | [SlimPajama-DC: Understanding Data Combinations for LLM Training.](http://arxiv.org/abs/2309.10818) | 本研究探讨了使用SlimPajama进行大型语言模型训练中不同数据组合的影响，提出了全局去重和局部去重的比较和高质量多源数据集的比例对模型性能的影响。 |
| [^8] | [Multi-Modality Multi-Loss Fusion Network.](http://arxiv.org/abs/2308.00264) | 多模态多损失融合网络通过最佳选择和融合多个模态的特征，提高了情感检测的性能，并在多个数据集上实现了最先进的结果。这些研究结果表明了用于增强神经网络中情感检测的特征选择和融合方法的优化方向。 |
| [^9] | [In-context Autoencoder for Context Compression in a Large Language Model.](http://arxiv.org/abs/2307.06945) | 在大型语言模型中，我们提出了一种称为In-context Autoencoder (ICAE)的上下文自编码器，它通过将长上下文压缩为有限数量的内存槽，实现了$4\times$的上下文压缩，并能够根据内存槽进行条件处理以响应各种提示。 |

# 详细

[^1]: PaperWeaver：通过将用户收集的论文与推荐论文上下文化，丰富主题论文提醒

    PaperWeaver: Enriching Topical Paper Alerts by Contextualizing Recommended Papers with User-collected Papers

    [https://arxiv.org/abs/2403.02939](https://arxiv.org/abs/2403.02939)

    PaperWeaver通过将用户收集的论文与推荐论文上下文化，为研究人员提供了更丰富的主题论文提醒

    

    随着学术档案的迅速增长，研究人员订阅“论文提醒”系统，定期为他们推荐最近发表的与之前收集的论文相似的论文。然而，研究人员有时很难理解推荐论文与他们自己研究背景之间微妙的联系，因为现有系统只呈现论文标题和摘要。为了帮助研究人员发现这些联系，我们提出了PaperWeaver，这是一个丰富的论文提醒系统，根据用户收集的论文提供推荐论文的上下文化文本描述。PaperWeaver采用基于大语言模型（LLMs）的计算方法，从用户收集的论文中推断用户的研究兴趣，提取论文的特定背景，并在这些背景上比较推荐论文和收集的论文。我们的用户研究（N=15）表明，使用PaperWeaver的参与者能够

    arXiv:2403.02939v1 Announce Type: cross  Abstract: With the rapid growth of scholarly archives, researchers subscribe to "paper alert" systems that periodically provide them with recommendations of recently published papers that are similar to previously collected papers. However, researchers sometimes struggle to make sense of nuanced connections between recommended papers and their own research context, as existing systems only present paper titles and abstracts. To help researchers spot these connections, we present PaperWeaver, an enriched paper alerts system that provides contextualized text descriptions of recommended papers based on user-collected papers. PaperWeaver employs a computational method based on Large Language Models (LLMs) to infer users' research interests from their collected papers, extract context-specific aspects of papers, and compare recommended and collected papers on these aspects. Our user study (N=15) showed that participants using PaperWeaver were able to
    
[^2]: DECIDERS：一种通过模仿双系统认知理论实现规则可控解码策略的语言生成方法

    DECIDER: A Rule-Controllable Decoding Strategy for Language Generation by Imitating Dual-System Cognitive Theory

    [https://arxiv.org/abs/2403.01954](https://arxiv.org/abs/2403.01954)

    DECIDER是一种受双系统认知理论启发的规则可控解码策略，通过在预训练语言模型中引入逻辑推理器，有效地遵循给定规则以引导生成方向朝向目标。

    

    词典约束解码方法旨在通过某些目标概念控制所生成文本的意义或风格。现有方法过于关注这些目标本身，导致缺乏关于如何实现这些目标的高层推理。然而，人类通常通过遵循某些规则来处理任务，这些规则不仅关注于目标本身，还关注于引发目标发生的语义相关概念。在这项工作中，我们提出了DECIDER，这是一种受到双系统认知理论启发的约束语言生成的规则可控解码策略。具体而言，在DECIDER中，一个预训练语言模型（PLM）配备了一个逻辑推理器，以高层规则作为输入。然后，DECIDER允许规则信号在每个解码步骤中流入PLM。广泛的实验结果表明，DECIDER能够有效地遵循给定的规则，引导生成方向朝向目标进行生成。

    arXiv:2403.01954v1 Announce Type: cross  Abstract: Lexicon-based constrained decoding approaches aim to control the meaning or style of the generated text through certain target concepts. Existing approaches over-focus the targets themselves, leading to a lack of high-level reasoning about how to achieve them. However, human usually tackles tasks by following certain rules that not only focuses on the targets but also on semantically relevant concepts that induce the occurrence of targets. In this work, we present DECIDER, a rule-controllable decoding strategy for constrained language generation inspired by dual-system cognitive theory. Specifically, in DECIDER, a pre-trained language model (PLM) is equiped with a logic reasoner that takes high-level rules as input. Then, the DECIDER allows rule signals to flow into the PLM at each decoding step. Extensive experimental results demonstrate that DECIDER can effectively follow given rules to guide generation direction toward the targets i
    
[^3]: 可扩展大型语言模型微调的差分隐私零阶方法

    Differentially Private Zeroth-Order Methods for Scalable Large Language Model Finetuning

    [https://arxiv.org/abs/2402.07818](https://arxiv.org/abs/2402.07818)

    本文研究了差分隐私零阶方法在大型语言模型微调中的应用，该方法通过使用零阶梯度来避免传统优化方法的可扩展性瓶颈，实现了在隐私、效用和可扩展性之间的良好平衡。

    

    在特定任务的数据集上进行微调是利用预训练语言模型的强大能力进行各种下游任务的广泛接受的范例。由于预训练语言模型微调的普及以及与之相关的隐私问题，差分隐私预训练语言模型微调引起了越来越多的关注，以保护特定任务数据集的隐私。差分隐私预训练语言模型微调方法的设计核心是在隐私、效用和可扩展性之间达到满意的权衡。大多数现有方法都是基于DP-SGD的创新性工作。尽管将DP-SGD的可扩展性推到了极限，但基于DP-SGD的微调方法不幸地受到了SGD固有低效率的限制。在本文中，我们研究了DP零阶方法在LLM预训练中的潜力，该方法通过用更高效的零阶梯度来近似梯度，避免了SGD的可扩展性瓶颈。与将零阶方法作为一种替代方法进行处理不同，我们引入了一种新的割接框架，该框架能够以非常接近的方式模拟DP-SGD的基本操作，然后利用零阶优化方法来近似梯度。

    Finetuning on task-specific datasets is a widely-embraced paradigm of harnessing the powerful capability of pretrained LLMs for various downstream tasks. Due to the popularity of LLMs finetuning and its accompanying privacy concerns, differentially private (DP) finetuning of pretrained LLMs has garnered increasing attention to safeguarding the privacy of task-specific datasets. Lying at the design core of DP LLM finetuning methods is the satisfactory tradeoff between privacy, utility, and scalability. Most existing methods build upon the seminal work of DP-SGD. Despite pushing the scalability of DP-SGD to its limit, DP-SGD-based finetuning methods are unfortunately limited by the inherent inefficiency of SGD. In this paper, we investigate the potential of DP zeroth-order methods for LLM pretraining, which avoids the scalability bottleneck of SGD by approximating the gradient with the more efficient zeroth-order gradient. Rather than treating the zeroth-order method as a drop-in replace
    
[^4]: 使用大型语言模型的多模态临床试验结果预测

    Multimodal Clinical Trial Outcome Prediction with Large Language Models

    [https://arxiv.org/abs/2402.06512](https://arxiv.org/abs/2402.06512)

    本研究提出了一种名为LIFTED的多模态临床试验结果预测方法，通过将不同模态数据转化为自然语言描述来统一数据，并构建统一的抗噪声编码器进行信息提取。

    

    临床试验是一个关键且昂贵的过程，通常需要多年时间和大量财力资源。因此，开发临床试验结果预测模型旨在排除可能失败的药物，并具有显著的成本节约潜力。最近的数据驱动尝试利用深度学习方法整合多模态数据来预测临床试验结果。然而，这些方法依赖于手动设计的模态特定编码器，这限制了适应新模态的可扩展性和识别不同模态之间相似信息模式的能力。为了解决这些问题，我们提出了一种多模态专家混合（LIFTED）方法用于临床试验结果预测。具体而言，LIFTED通过将不同模态的数据转化为自然语言描述来统一不同模态数据。然后，LIFTED构建统一的抗噪声编码器，从模态特定的语言描述中提取信息。

    The clinical trial is a pivotal and costly process, often spanning multiple years and requiring substantial financial resources. Therefore, the development of clinical trial outcome prediction models aims to exclude drugs likely to fail and holds the potential for significant cost savings. Recent data-driven attempts leverage deep learning methods to integrate multimodal data for predicting clinical trial outcomes. However, these approaches rely on manually designed modal-specific encoders, which limits both the extensibility to adapt new modalities and the ability to discern similar information patterns across different modalities. To address these issues, we propose a multimodal mixture-of-experts (LIFTED) approach for clinical trial outcome prediction. Specifically, LIFTED unifies different modality data by transforming them into natural language descriptions. Then, LIFTED constructs unified noise-resilient encoders to extract information from modal-specific language descriptions. S
    
[^5]: FAQ-Gen：一个自动生成特定领域常见问题解答的自动化系统，用于辅助理解内容

    FAQ-Gen: An automated system to generate domain-specific FAQs to aid content comprehension

    [https://arxiv.org/abs/2402.05812](https://arxiv.org/abs/2402.05812)

    本文介绍了FAQ-Gen，一个利用文本转文本转换模型开发的自动化系统，用于生成特定领域的常见问题解答。该系统通过使用自我策划的算法来优化信息表示和问题-答案排名，提高了FAQ的准确性和相关性。定性人类评估表明生成的FAQs构造良好且可推广至不同领域。

    

    常见问题解答（FAQ）是指关于特定内容的最常见询问。它们通过简化主题和通过简明扼要地呈现信息来增强理解，作为内容理解的辅助工具。在本文中，我们通过开发一个端到端系统利用文本转文本转换模型来解决FAQ生成作为一个明确定义的自然语言处理（NLP）任务。我们提供了一份涵盖传统问答系统的文献综述，强调它们在直接应用于FAQ生成任务时的局限性。我们提出了一个系统，能够根据特定领域的文本内容构建FAQs，提高其准确性和相关性。我们利用自我策划的算法来获得提供给输入的信息的最佳表示，并对问题-答案对进行排序以最大化人类理解。定性人类评估显示生成的FAQs构造良好且可推广至不同领域。

    Frequently Asked Questions (FAQs) refer to the most common inquiries about specific content. They serve as content comprehension aids by simplifying topics and enhancing understanding through succinct presentation of information. In this paper, we address FAQ generation as a well-defined Natural Language Processing (NLP) task through the development of an end-to-end system leveraging text-to-text transformation models. We present a literature review covering traditional question-answering systems, highlighting their limitations when applied directly to the FAQ generation task. We propose our system capable of building FAQs from textual content tailored to specific domains, enhancing their accuracy and relevance. We utilise self-curated algorithms for obtaining optimal representation of information to be provided as input and also for ranking the question-answer pairs to maximise human comprehension. Qualitative human evaluation showcases the generated FAQs to be well-constructed and re
    
[^6]: 大型人类语言模型：需求和挑战

    Large Human Language Models: A Need and the Challenges

    [https://arxiv.org/abs/2312.07751](https://arxiv.org/abs/2312.07751)

    大型人类语言模型的建立需要更好地整合人类背景，并面临着如何捕捉人类因素、如何表示以及如何建模的一系列挑战。

    

    随着人类中心的自然语言处理研究的进展，人们越来越意识到将人类和社会因素纳入到自然语言处理模型的重要性。同时，我们的自然语言处理系统已经严重依赖于LLM，其中大多数并没有对作者进行建模。为了构建能够真正理解人类语言的自然语言处理系统，我们必须更好地将人类背景整合到LLM中。这提出了一系列设计考虑和挑战，涉及到要捕捉哪些人类因素、如何表示它们以及要采用何种建模策略等问题。为了应对这些挑战，我们提倡从心理学和行为科学的概念出发，支持三个立场来创建大型人类语言模型（LHLMs）：首先，语言模型训练应包括人类背景。其次，LHLMs应该意识到人不仅仅是他们所属的群体。第三，LHLMs应该能够考虑到人类背景的动态和时间依赖性特点。

    arXiv:2312.07751v2 Announce Type: replace-cross  Abstract: As research in human-centered NLP advances, there is a growing recognition of the importance of incorporating human and social factors into NLP models. At the same time, our NLP systems have become heavily reliant on LLMs, most of which do not model authors. To build NLP systems that can truly understand human language, we must better integrate human contexts into LLMs. This brings to the fore a range of design considerations and challenges in terms of what human aspects to capture, how to represent them, and what modeling strategies to pursue. To address these, we advocate for three positions toward creating large human language models (LHLMs) using concepts from psychological and behavioral sciences: First, LM training should include the human context. Second, LHLMs should recognize that people are more than their group(s). Third, LHLMs should be able to account for the dynamic and temporally-dependent nature of the human con
    
[^7]: SlimPajama-DC: 理解LLM训练中的数据组合

    SlimPajama-DC: Understanding Data Combinations for LLM Training. (arXiv:2309.10818v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2309.10818](http://arxiv.org/abs/2309.10818)

    本研究探讨了使用SlimPajama进行大型语言模型训练中不同数据组合的影响，提出了全局去重和局部去重的比较和高质量多源数据集的比例对模型性能的影响。

    

    本研究旨在了解使用SlimPajama进行大型语言模型训练时各种数据组合（如网络文本、维基百科、GitHub、图书）对其训练的影响。SlimPajama是一个经过严格去重的多源数据集，从Together贡献的1.2T个token的RedPajama数据集中精细组合和去重，总共得到了627B个tokens。我们将我们的研究称为SlimPajama-DC，这是一项旨在揭示在大型语言模型训练中使用SlimPajama所涉及的基本特征和最佳实践的经验分析。在我们使用SlimPajama进行研究的过程中，出现了两个关键观察结果：（1）全局去重 vs. 局部去重。我们分析和讨论了全局去重（跨不同数据集源）和局部去重（在单个数据集源内部）对训练模型性能的影响。（2）高质量/高度去重的多源数据集在组合中的比例。为了研究这一点，我们进行了...

    This paper aims to understand the impacts of various data combinations (e.g., web text, wikipedia, github, books) on the training of large language models using SlimPajama. SlimPajama is a rigorously deduplicated, multi-source dataset, which has been refined and further deduplicated to 627B tokens from the extensive 1.2T tokens RedPajama dataset contributed by Together. We've termed our research as SlimPajama-DC, an empirical analysis designed to uncover fundamental characteristics and best practices associated with employing SlimPajama in the training of large language models. During our research with SlimPajama, two pivotal observations emerged: (1) Global deduplication vs. local deduplication. We analyze and discuss how global (across different sources of datasets) and local (within the single source of dataset) deduplications affect the performance of trained models. (2) Proportions of high-quality/highly-deduplicated multi-source datasets in the combination. To study this, we cons
    
[^8]: 多模态多损失融合网络

    Multi-Modality Multi-Loss Fusion Network. (arXiv:2308.00264v1 [cs.CL])

    [http://arxiv.org/abs/2308.00264](http://arxiv.org/abs/2308.00264)

    多模态多损失融合网络通过最佳选择和融合多个模态的特征，提高了情感检测的性能，并在多个数据集上实现了最先进的结果。这些研究结果表明了用于增强神经网络中情感检测的特征选择和融合方法的优化方向。

    

    在这项工作中，我们研究了跨多个模态选择和融合特征的最佳方法，并将其组合在神经网络中以改善情感检测。我们比较了不同的融合方法并且研究了多损失训练在多模态融合网络中的影响，从而确定了与子网性能相关的有用发现。我们最好的模型在三个数据集（CMU-MOSI、CMU-MOSEI和CH-SIMS）上实现了最先进的性能，并且在大多数指标上优于其他方法。我们发现，训练多模态特征可以改善单模态测试，并且基于数据集注释模式设计融合方法可以增强模型性能。这些结果表明了在神经网络中增强情感检测的优化特征选择和融合方法的发展方向。

    In this work we investigate the optimal selection and fusion of features across multiple modalities and combine these in a neural network to improve emotion detection. We compare different fusion methods and examine the impact of multi-loss training within the multi-modality fusion network, identifying useful findings relating to subnet performance. Our best model achieves state-of-the-art performance for three datasets (CMU-MOSI, CMU-MOSEI and CH-SIMS), and outperforms the other methods in most metrics. We have found that training on multimodal features improves single modality testing and designing fusion methods based on dataset annotation schema enhances model performance. These results suggest a roadmap towards an optimized feature selection and fusion approach for enhancing emotion detection in neural networks.
    
[^9]: 在大型语言模型中的上下文压缩的上下文自编码器

    In-context Autoencoder for Context Compression in a Large Language Model. (arXiv:2307.06945v1 [cs.CL])

    [http://arxiv.org/abs/2307.06945](http://arxiv.org/abs/2307.06945)

    在大型语言模型中，我们提出了一种称为In-context Autoencoder (ICAE)的上下文自编码器，它通过将长上下文压缩为有限数量的内存槽，实现了$4\times$的上下文压缩，并能够根据内存槽进行条件处理以响应各种提示。

    

    我们提出了一种用于大型语言模型中上下文压缩的上下文自编码器（ICAE）。 ICAE有两个模块：一个可学习的编码器，通过从LLM中采用LoRA方式将长上下文压缩为有限数量的内存槽，以及一个固定的解码器，作为目标LLM，可以根据内存槽来进行各种目的的条件处理。我们首先使用自编码和语言建模目标在大规模文本数据上预训练ICAE，使其能够生成准确和全面表示原始上下文的内存槽。然后，我们使用少量指导数据对预训练的ICAE进行微调，以增强其与各种提示的交互，从而产生理想的响应。我们的实验结果表明，使用我们提出的预训练和微调范式学习的ICAE可以有效地产生$4\times$上下文压缩的内存槽，目标LLM可以很好地对其进行条件处理，以响应各种提示。

    We propose the In-context Autoencoder (ICAE) for context compression in a large language model (LLM). The ICAE has two modules: a learnable encoder adapted with LoRA from an LLM for compressing a long context into a limited number of memory slots, and a fixed decoder which is the target LLM that can condition on the memory slots for various purposes. We first pretrain the ICAE using both autoencoding and language modeling objectives on massive text data, enabling it to generate memory slots that accurately and comprehensively represent the original context. Then, we fine-tune the pretrained ICAE on a small amount of instruct data to enhance its interaction with various prompts for producing desirable responses. Our experimental results demonstrate that the ICAE learned with our proposed pretraining and fine-tuning paradigm can effectively produce memory slots with $4\times$ context compression, which can be well conditioned on by the target LLM to respond to various prompts. The promis
    

