# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [TRAMS: Training-free Memory Selection for Long-range Language Modeling.](http://arxiv.org/abs/2310.15494) | TRAMS是一种训练免费的长程语言建模记忆选择策略，它能够提高Transformer架构在长程语言建模方面的效果，并且不需要额外的训练或参数。 |
| [^2] | [MCC-KD: Multi-CoT Consistent Knowledge Distillation.](http://arxiv.org/abs/2310.14747) | MCC-KD方法提出了一种多CoT一致性知识蒸馏的方法，能够高效地转移大型语言模型的推理能力到较小的模型上，通过生成多个理由并确保其预测的一致性来增强推理多样性和一致性。 |
| [^3] | [Redefining Digital Health Interfaces with Large Language Models.](http://arxiv.org/abs/2310.03560) | 本论文提出了利用大型语言模型（LLMs）重新定义数字健康界面的方法，将LLMs与外部工具结合使用，从而提高了与临床技术的互动效果，改善了数字医疗工具和AI模型的实用性，并解决了在临床环境中使用LLMs的问题。 |
| [^4] | [Knowledge Graphs for the Life Sciences: Recent Developments, Challenges and Opportunities.](http://arxiv.org/abs/2309.17255) | 这篇论文综述了在生命科学领域中使用知识图谱的最新发展和进展，并展望了这些技术在未来对这些领域的影响。 |
| [^5] | [Journey to the Center of the Knowledge Neurons: Discoveries of Language-Independent Knowledge Neurons and Degenerate Knowledge Neurons.](http://arxiv.org/abs/2308.13198) | 本文研究了多语言预训练语言模型中事实知识的存储方式，并引入了一种新的方法，能够更准确地定位知识神经元。通过实验证明了语言无关知识神经元的存在，以及发现了一种新型退化知识神经元。 |
| [^6] | [PMET: Precise Model Editing in a Transformer.](http://arxiv.org/abs/2308.08742) | 该论文通过分析Transformer模型中的隐藏状态，发现多头自注意力编码了某些通用知识提取模式，因此在进行模型编辑时，不需要更新多头自注意力的权重。 |
| [^7] | [IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages.](http://arxiv.org/abs/2305.16307) | 本研究填补了印度22种官方语言机器翻译的空白，提出了IndicTrans2系统，它在多个语言对上表现最先进并优于现有的模型，同时提供了印度语言的基准和评估脚本。 |
| [^8] | [RewriteLM: An Instruction-Tuned Large Language Model for Text Rewriting.](http://arxiv.org/abs/2305.15685) | 本文介绍了 RewriteLM，一种指令调整的大型语言模型，用于长篇文本重写。同时，我们提出了一个名为 OpenRewriteEval 的基准测试，用于评估各种类型的开放式长篇文本重写。我们采用新的策略来促进多样的指令和偏好数据生成，从而为长篇文本重写提供更好的评估手段。 |
| [^9] | [Evaluating task understanding through multilingual consistency: A ChatGPT case study.](http://arxiv.org/abs/2305.11662) | 本文提出了一种新的评估大型语言模型理解能力的范例，通过评估模型自身生成的不同意义之间的一致性，探讨了多语言自我一致性作为模型理解的检验方法，同时证明了ChatGPT在多语言一致性方面的优秀性能。 |
| [^10] | [The Short Text Matching Model Enhanced with Knowledge via Contrastive Learning.](http://arxiv.org/abs/2304.03898) | 提出了一种短文本匹配模型，使用生成模型生成补充句子，结合对比学习和外部知识进行语义匹配，并使用关键词避免噪声问题。 |
| [^11] | [Safety Analysis in the Era of Large Language Models: A Case Study of STPA using ChatGPT.](http://arxiv.org/abs/2304.01246) | 本文研究了大型语言模型在系统论过程分析（STPA）中的应用，并采用ChatGPT对自动紧急制动（AEB）系统进行了案例研究。结果表明，重复双工交互方法是最有效的，并显着提高了STPA的质量。本研究证明，LLMs可以应用于安全分析，并为安全关键系统提供有价值的见解。 |
| [^12] | [Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation.](http://arxiv.org/abs/2303.15413) | 本文提出了两种去偏置的方法，一种通过增加2D扩散模型得出的分数的截断值，一种通过调整视角提示和物体空间摄像机姿态之间的差异。实验结果表明这些方法可以显著减少伪影，提高真实感。 |
| [^13] | [SEAM: An Integrated Activation-Coupled Model of Sentence Processing and Eye Movements in Reading.](http://arxiv.org/abs/2303.05221) | SEAM是一种集成了眼动控制和句子处理的模型，为实现阅读中自然语言理解的完整数学模型迈出了重要一步。 |

# 详细

[^1]: TRAMS:训练免费的长程语言建模记忆选择

    TRAMS: Training-free Memory Selection for Long-range Language Modeling. (arXiv:2310.15494v1 [cs.CL])

    [http://arxiv.org/abs/2310.15494](http://arxiv.org/abs/2310.15494)

    TRAMS是一种训练免费的长程语言建模记忆选择策略，它能够提高Transformer架构在长程语言建模方面的效果，并且不需要额外的训练或参数。

    

    Transformer架构对于众多AI模型至关重要，但在长程语言建模方面仍面临挑战。尽管已经设计了几种特定的Transformer架构来解决长程依赖的问题，但现有的方法如Transformer-XL存在大量无效记忆的问题。本研究提出了一种即插即用的策略，称为TRAining-free Memory Selection（TRAMS），它根据一个简单的指标选择参与注意力计算的标记。该策略允许我们保留与当前查询具有高关注分数可能性的标记，并忽略其他标记。我们在单词级基准（WikiText-103）和字符级基准（enwik8）上测试了我们的方法，结果表明在不进行额外训练或添加额外参数的情况下取得了改进。

    The Transformer architecture is crucial for numerous AI models, but it still faces challenges in long-range language modeling. Though several specific transformer architectures have been designed to tackle issues of long-range dependencies, existing methods like Transformer-XL are plagued by a high percentage of ineffective memories. In this study, we present a plug-and-play strategy, known as TRAining-free Memory Selection (TRAMS), that selects tokens participating in attention calculation based on one simple metric. This strategy allows us to keep tokens that are likely to have a high attention score with the current queries and ignore the other ones. We have tested our approach on the word-level benchmark (WikiText-103) and the character-level benchmark (enwik8), and the results indicate an improvement without having additional training or adding additional parameters.
    
[^2]: MCC-KD: 多CoT一致性知识蒸馏

    MCC-KD: Multi-CoT Consistent Knowledge Distillation. (arXiv:2310.14747v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.14747](http://arxiv.org/abs/2310.14747)

    MCC-KD方法提出了一种多CoT一致性知识蒸馏的方法，能够高效地转移大型语言模型的推理能力到较小的模型上，通过生成多个理由并确保其预测的一致性来增强推理多样性和一致性。

    

    大型语言模型(LLMs)在复杂推理方面展现了卓越的能力，通过链式思考(CoT)提示。最近，人们对将这些推理能力从LLMs转移到较小模型中的兴趣日益增长。然而，同时实现多样性和一致性的理由对于提出了挑战。在本文中，我们着重增强这两个方面，并提出了多CoT一致性知识蒸馏(MCC-KD)方法，以高效地提取推理能力。在MCC-KD中，我们为每个问题生成多个理由，并通过最小化答案分布之间的双向KL散度来确保相应预测的一致性。我们研究了MCC-KD在不同模型架构(LLaMA/FlanT5)和各种模型规模(3B/7B/11B/13B)上在数学推理和常识推理基准上的有效性。实证结果不仅证实了MCC-KD在分布内情况下的优越性能。

    Large language models (LLMs) have showcased remarkable capabilities in complex reasoning through chain of thought (CoT) prompting. Recently, there has been a growing interest in transferring these reasoning abilities from LLMs to smaller models. However, achieving both the diversity and consistency in rationales presents a challenge. In this paper, we focus on enhancing these two aspects and propose Multi-CoT Consistent Knowledge Distillation (MCC-KD) to efficiently distill the reasoning capabilities. In MCC-KD, we generate multiple rationales for each question and enforce consistency among the corresponding predictions by minimizing the bidirectional KL-divergence between the answer distributions. We investigate the effectiveness of MCC-KD with different model architectures (LLaMA/FlanT5) and various model scales (3B/7B/11B/13B) on both mathematical reasoning and commonsense reasoning benchmarks. The empirical results not only confirm MCC-KD's superior performance on in-distribution d
    
[^3]: 用大型语言模型重新定义数字健康界面

    Redefining Digital Health Interfaces with Large Language Models. (arXiv:2310.03560v1 [cs.CL])

    [http://arxiv.org/abs/2310.03560](http://arxiv.org/abs/2310.03560)

    本论文提出了利用大型语言模型（LLMs）重新定义数字健康界面的方法，将LLMs与外部工具结合使用，从而提高了与临床技术的互动效果，改善了数字医疗工具和AI模型的实用性，并解决了在临床环境中使用LLMs的问题。

    

    数字健康工具具有显著改善医疗服务传递的潜力。然而，由于可用性和信任方面的挑战，它们的使用仍然相对有限。最近，大型语言模型（LLMs）作为具有处理复杂信息和生成人类质量文本能力的通用模型出现，为医疗保健领域提供了丰富的潜在应用。直接在临床环境中应用LLMs并不简单，因为LLMs容易提供不一致或无意义的答案。我们展示了如何利用外部工具使LLMs在临床医疗技术互动中提供全新界面。这增强了数字医疗工具和AI模型的实用性和实际影响，同时解决了在临床环境中使用LLMs的当前问题，如幻觉。我们通过心血管疾病和糖尿病风险预测的例子阐述了我们的方法，并突出了其中的好处。

    Digital health tools have the potential to significantly improve the delivery of healthcare services. However, their use remains comparatively limited due, in part, to challenges surrounding usability and trust. Recently, Large Language Models (LLMs) have emerged as general-purpose models with the ability to process complex information and produce human-quality text, presenting a wealth of potential applications in healthcare. Directly applying LLMs in clinical settings is not straightforward, with LLMs susceptible to providing inconsistent or nonsensical answers. We demonstrate how LLMs can utilize external tools to provide a novel interface between clinicians and digital technologies. This enhances the utility and practical impact of digital healthcare tools and AI models while addressing current issues with using LLM in clinical settings such as hallucinations. We illustrate our approach with examples from cardiovascular disease and diabetes risk prediction, highlighting the benefit
    
[^4]: 生命科学领域的知识图谱：最新发展、挑战和机遇

    Knowledge Graphs for the Life Sciences: Recent Developments, Challenges and Opportunities. (arXiv:2309.17255v1 [cs.AI])

    [http://arxiv.org/abs/2309.17255](http://arxiv.org/abs/2309.17255)

    这篇论文综述了在生命科学领域中使用知识图谱的最新发展和进展，并展望了这些技术在未来对这些领域的影响。

    

    生命科学是研究生物和生命过程的学科，包括化学、生物学、医学和一系列其他相关学科。生命科学的研究工作非常依赖数据，因为它们产生和消费大量科学数据，其中很多数据具有关系和图结构。数据的数量和其中涉及的科学概念和关系的复杂性推动了应用先进的知识驱动技术来管理和解释数据，最终目标是推动科学发现。在这篇综述和观点论文中，我们讨论了知识图谱在生命科学中的最新发展和进展，并展望了这些技术在未来对这些领域的影响。我们重点关注三个主题：知识图谱的构建和管理，以及在新发现的过程中使用知识图谱和相关技术。

    The term life sciences refers to the disciplines that study living organisms and life processes, and include chemistry, biology, medicine, and a range of other related disciplines. Research efforts in life sciences are heavily data-driven, as they produce and consume vast amounts of scientific data, much of which is intrinsically relational and graph-structured.  The volume of data and the complexity of scientific concepts and relations referred to therein promote the application of advanced knowledge-driven technologies for managing and interpreting data, with the ultimate aim to advance scientific discovery.  In this survey and position paper, we discuss recent developments and advances in the use of graph-based technologies in life sciences and set out a vision for how these technologies will impact these fields into the future. We focus on three broad topics: the construction and management of Knowledge Graphs (KGs), the use of KGs and associated technologies in the discovery of ne
    
[^5]: 深入理解多语言预训练语言模型中的知识神经元：语言无关知识神经元和退化知识神经元的发现

    Journey to the Center of the Knowledge Neurons: Discoveries of Language-Independent Knowledge Neurons and Degenerate Knowledge Neurons. (arXiv:2308.13198v1 [cs.CL])

    [http://arxiv.org/abs/2308.13198](http://arxiv.org/abs/2308.13198)

    本文研究了多语言预训练语言模型中事实知识的存储方式，并引入了一种新的方法，能够更准确地定位知识神经元。通过实验证明了语言无关知识神经元的存在，以及发现了一种新型退化知识神经元。

    

    预训练语言模型（PLMs）包含大量的事实知识，但其存储在参数中的方式尚不清楚。本文深入研究了多语言PLMs中事实知识的存储方式，并引入了一种基于体系结构的多语言整合梯度方法，相比现有方法更准确地定位知识神经元，并在不同体系结构和语言之间更具普遍性。此外，我们对知识神经元进行了深入探索，得出了以下两项重要的发现：（1）发现了语言无关知识神经元，其以超越语言的方式存储事实知识。我们设计了跨语言知识编辑实验，证明了PLMs可以基于语言无关的神经元完成此任务；（2）发现了退化知识神经元，这是一种新型神经元，表明不同的知识神经元可以在数据特征萎缩的情况下展示。

    Pre-trained language models (PLMs) contain vast amounts of factual knowledge, but how the knowledge is stored in the parameters remains unclear. This paper delves into the complex task of understanding how factual knowledge is stored in multilingual PLMs, and introduces the Architecture-adapted Multilingual Integrated Gradients method, which successfully localizes knowledge neurons more precisely compared to current methods, and is more universal across various architectures and languages. Moreover, we conduct an in-depth exploration of knowledge neurons, leading to the following two important discoveries: (1) The discovery of Language-Independent Knowledge Neurons, which store factual knowledge in a form that transcends language. We design cross-lingual knowledge editing experiments, demonstrating that the PLMs can accomplish this task based on language-independent neurons; (2) The discovery of Degenerate Knowledge Neurons, a novel type of neuron showing that different knowledge neuro
    
[^6]: PMET: 在Transformer中的精确模型编辑

    PMET: Precise Model Editing in a Transformer. (arXiv:2308.08742v1 [cs.CL])

    [http://arxiv.org/abs/2308.08742](http://arxiv.org/abs/2308.08742)

    该论文通过分析Transformer模型中的隐藏状态，发现多头自注意力编码了某些通用知识提取模式，因此在进行模型编辑时，不需要更新多头自注意力的权重。

    

    模型编辑技术可以以较低的成本修改大型语言模型中的少量知识，并且已经取得了显著的成功。现有方法假设Transformer层隐藏状态是前馈网络的键值内存的值。它们通常优化Transformer层隐藏状态来记忆目标知识，并将其用于更新大型语言模型中前馈网络的权重。然而，Transformer层隐藏状态的信息流来自三个部分：多头自注意力、前馈网络和残差连接。现有方法忽视了Transformer层隐藏状态包含了前馈网络特别需要的信息这一事实。因此，模型编辑的性能下降。为了实现更精确的模型编辑，我们分析了多头自注意力和前馈网络的隐藏状态，发现多头自注意力编码了某些通用知识提取模式。这意味着当引入新知识时，多头自注意力的权重不需要更新。

    Model editing techniques modify a minor proportion of knowledge in Large Language Models (LLMs) at a relatively low cost, which have demonstrated notable success. Existing methods assume Transformer Layer (TL) hidden states are values of key-value memories of the Feed-Forward Network (FFN). They usually optimize the TL hidden states to memorize target knowledge and use it to update the weights of the FFN in LLMs. However, the information flow of TL hidden states comes from three parts: Multi-Head Self-Attention (MHSA), FFN, and residual connections. Existing methods neglect the fact that the TL hidden states contains information not specifically required for FFN. Consequently, the performance of model editing decreases. To achieve more precise model editing, we analyze hidden states of MHSA and FFN, finding that MHSA encodes certain general knowledge extraction patterns. This implies that MHSA weights do not require updating when new knowledge is introduced. Based on above findings, we
    
[^7]: IndicTrans2: 为印度所有22种官方语言构建高质量可访问的机器翻译模型

    IndicTrans2: Towards High-Quality and Accessible Machine Translation Models for all 22 Scheduled Indian Languages. (arXiv:2305.16307v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.16307](http://arxiv.org/abs/2305.16307)

    本研究填补了印度22种官方语言机器翻译的空白，提出了IndicTrans2系统，它在多个语言对上表现最先进并优于现有的模型，同时提供了印度语言的基准和评估脚本。

    

    印度有着丰富的语言景观，包括四个主要语系的语言，超过十亿人口使用。本篇论文聚焦于印度宪法列出的22种语言，被称为“官方语言”。鉴于语言多样性，高质量和可访问的机器翻译系统在印度这样的国家至关重要。在此之前，缺少（i）涵盖所有22种语言的平行训练数据，（ii）覆盖这些语言并包含印度相关内容的健壮基准，以及（iii）支持印度所有22种官方语言的现有翻译模型。本文旨在填补这一空白，并专注于启用22种印度官方语言的广泛、易于使用和开放式访问好的机器翻译系统所需的缺失部分。我们确定了四个改进关键领域：策划和创建更大的训练数据集、创建多样化和高质量的基准、跨越所有22种官方语言进行翻译，并构建高质量且可访问的机器翻译模型。我们的提议系统IndicTrans2在多个语言对上实现了最先进的性能，并且优于现有的在22种官方语言下的机器翻译模型。此外，我们提供了这些语言的基准和评估脚本，使得研究人员更容易地评价和提高关于印度语言的机器翻译模型。

    India has a rich linguistic landscape with languages from 4 major language families spoken by over a billion people. 22 of these languages are listed in the Constitution of India (referred to as scheduled languages) are the focus of this work. Given the linguistic diversity, high-quality and accessible Machine Translation (MT) systems are essential in a country like India. Prior to this work, there was (i) no parallel training data spanning all the 22 languages, (ii) no robust benchmarks covering all these languages and containing content relevant to India, and (iii) no existing translation models which support all the 22 scheduled languages of India. In this work, we aim to address this gap by focusing on the missing pieces required for enabling wide, easy, and open access to good machine translation systems for all 22 scheduled Indian languages. We identify four key areas of improvement: curating and creating larger training datasets, creating diverse and high-quality benchmarks, tra
    
[^8]: RewriteLM：一种面向文本重写的指令调整大型语言模型。

    RewriteLM: An Instruction-Tuned Large Language Model for Text Rewriting. (arXiv:2305.15685v1 [cs.CL])

    [http://arxiv.org/abs/2305.15685](http://arxiv.org/abs/2305.15685)

    本文介绍了 RewriteLM，一种指令调整的大型语言模型，用于长篇文本重写。同时，我们提出了一个名为 OpenRewriteEval 的基准测试，用于评估各种类型的开放式长篇文本重写。我们采用新的策略来促进多样的指令和偏好数据生成，从而为长篇文本重写提供更好的评估手段。

    

    大型语言模型已经展现出在长篇文本生成任务中通过自然语言指令表达来的惊人的零-shot能力，然而用户对于长篇文本重写的期望值很高，模型产生的意外重写（“幻觉”）会对其整体性能产生负面影响。现有的评估基准主要关注有限的重写风格和句子级重写，而不是长篇开放式重写。我们引入了一个新的基准测试OpenRewriteEval，它涵盖了通过自然语言指令表达的各种重写类型。它特别设计用于促进长篇文本开放式重写的评估。此外，我们提出了一种强大的基线模型RewriteLM，一个用于长篇文本重写的指令调整大型语言模型。我们开发了一些新策略，以最小人工干预促进生成多样的指令和偏好数据。

    Large Language Models (LLMs) have demonstrated impressive zero-shot capabilities in long-form text generation tasks expressed through natural language instructions. However, user expectations for long-form text rewriting is high, and unintended rewrites (''hallucinations'') produced by the model can negatively impact its overall performance. Existing evaluation benchmarks primarily focus on limited rewriting styles and sentence-level rewriting rather than long-form open-ended rewriting.We introduce OpenRewriteEval, a novel benchmark that covers a wide variety of rewriting types expressed through natural language instructions. It is specifically designed to facilitate the evaluation of open-ended rewriting of long-form texts. In addition, we propose a strong baseline model, RewriteLM, an instruction-tuned large language model for long-form text rewriting. We develop new strategies that facilitate the generation of diverse instructions and preference data with minimal human intervention.
    
[^9]: 通过多语言一致性评估任务理解：ChatGPT案例研究

    Evaluating task understanding through multilingual consistency: A ChatGPT case study. (arXiv:2305.11662v1 [cs.CL])

    [http://arxiv.org/abs/2305.11662](http://arxiv.org/abs/2305.11662)

    本文提出了一种新的评估大型语言模型理解能力的范例，通过评估模型自身生成的不同意义之间的一致性，探讨了多语言自我一致性作为模型理解的检验方法，同时证明了ChatGPT在多语言一致性方面的优秀性能。

    

    随着大型语言模型（LLM）功能的惊人提升，创建未来可持续的评估集以评估它们的理解变得越来越具有挑战性。本文提出了一种新的评估LLM的范例，该范例利用了正确的世界理解应该在相同含义的不同（弗雷格）意义上保持一致的思想。因此，我们不是通过正确性来衡量理解，而是通过评估模型自身生成的多个意义之间的一致性来衡量。我们通过实例化一个测试展示了我们的方法，其中不同的意义是不同的语言，因此将多语言自我一致性作为模型理解的检验并同时解决多语言的重要主题。我们以最新版本的ChatGPT为我们的研究对象，在三种不同语言中评估两个不同任务的多语言一致性。我们证明了ChatGPT在多语言一致性方面的优秀性能。

    At the staggering pace with which the capabilities of large language models (LLMs) are increasing, creating future-proof evaluation sets to assess their understanding becomes more and more challenging. In this paper, we propose a novel paradigm for evaluating LLMs which leverages the idea that correct world understanding should be consistent across different (Fregean) senses of the same meaning. Accordingly, we measure understanding not in terms of correctness but by evaluating consistency across multiple senses that are generated by the model itself. We showcase our approach by instantiating a test where the different senses are different languages, hence using multilingual self-consistency as a litmus test for the model's understanding and simultaneously addressing the important topic of multilingualism. Taking one of the latest versions of ChatGPT as our object of study, we evaluate multilingual consistency for two different tasks across three different languages. We show that its m
    
[^10]: 通过对比学习加强知识的短文本匹配模型

    The Short Text Matching Model Enhanced with Knowledge via Contrastive Learning. (arXiv:2304.03898v1 [cs.CL])

    [http://arxiv.org/abs/2304.03898](http://arxiv.org/abs/2304.03898)

    提出了一种短文本匹配模型，使用生成模型生成补充句子，结合对比学习和外部知识进行语义匹配，并使用关键词避免噪声问题。

    

    近年来，短文本匹配任务在广告搜索和推荐领域得到了广泛应用。由于文本长度短，语义信息匮乏和单词歧义问题成为此类任务的难点。先前的研究已经引入文本补充句子或知识库来提供附加的特征信息。然而，这些方法没有充分地交互原始句子和补充句子，也没有考虑到外部知识库引入的噪声问题。因此，本文提出了一种结合对比学习和外部知识的短文本匹配模型。该模型利用生成模型生成对应的补充句子，并使用对比学习方法指导模型获得更具语义匹配性的原始句子编码。此外，为了避免噪声，我们使用关键词作为原始句子的主要语义进行检索。

    In recent years, short Text Matching tasks have been widely applied in the fields ofadvertising search and recommendation. The difficulty lies in the lack of semantic information and word ambiguity caused by the short length of the text. Previous works have introduced complement sentences or knowledge bases to provide additional feature information. However, these methods have not fully interacted between the original sentence and the complement sentence, and have not considered the noise issue that may arise from the introduction of external knowledge bases. Therefore, this paper proposes a short Text Matching model that combines contrastive learning and external knowledge. The model uses a generative model to generate corresponding complement sentences and uses the contrastive learning method to guide the model to obtain more semantically meaningful encoding of the original sentence. In addition, to avoid noise, we use keywords as the main semantics of the original sentence to retrie
    
[^11]: 大语言模型时代的安全分析：聊天GPT在STPA案例研究中的应用

    Safety Analysis in the Era of Large Language Models: A Case Study of STPA using ChatGPT. (arXiv:2304.01246v1 [cs.CL])

    [http://arxiv.org/abs/2304.01246](http://arxiv.org/abs/2304.01246)

    本文研究了大型语言模型在系统论过程分析（STPA）中的应用，并采用ChatGPT对自动紧急制动（AEB）系统进行了案例研究。结果表明，重复双工交互方法是最有效的，并显着提高了STPA的质量。本研究证明，LLMs可以应用于安全分析，并为安全关键系统提供有价值的见解。

    

    大型语言模型（LLMs），如ChatGPT和BERT，由于其具有类似于人类的对话，在许多知识领域中具有详细和明确的答案，正在引领一场新的人工智能热潮。虽然LLMs正在迅速应用于许多人工智能应用领域，但我们对以下问题感兴趣：安全关键系统的安全分析是否可以利用LLMs？为了回答这个问题，我们使用ChatGPT对自动紧急制动（AEB）系统的系统论过程分析（STPA）进行了案例研究。STPA是最普遍的危险分析技术之一，但它存在诸多局限性，例如高复杂性和主观性，本文旨在探讨ChatGPT的应用，以解决这些局限性。具体而言，通过考虑其与人类专家的交互，研究了三种将ChatGPT纳入STPA中的方法：一次性单工交互、重复单工交互和重复双工交互。比较结果表明：（i）在没有人类专家的情况下使用ChatGPT不能为STPA提供足够的信息；（ii）一次性单工交互对STPA有帮助，但不如重复交互有效；（iii）重复双工交互一致优于其他方法，并显着提高了STPA的质量。我们的研究表明，LLMs可以应用于安全分析，并为AEB以外的其他安全关键系统提供有价值的见解。

    Large Language Models (LLMs), such as ChatGPT and BERT, are leading a new AI heatwave due to its human-like conversations with detailed and articulate answers across many domains of knowledge. While LLMs are being quickly applied to many AI application domains, we are interested in the following question: Can safety analysis for safety-critical systems make use of LLMs? To answer, we conduct a case study of Systems Theoretic Process Analysis (STPA) on Automatic Emergency Brake (AEB) systems using ChatGPT. STPA, one of the most prevalent techniques for hazard analysis, is known to have limitations such as high complexity and subjectivity, which this paper aims to explore the use of ChatGPT to address. Specifically, three ways of incorporating ChatGPT into STPA are investigated by considering its interaction with human experts: one-off simplex interaction, recurring simplex interaction, and recurring duplex interaction. Comparative results reveal that: (i) using ChatGPT without human exp
    
[^12]: 2D扩散算法的去偏置方法用于文本到3D生成

    Debiasing Scores and Prompts of 2D Diffusion for Robust Text-to-3D Generation. (arXiv:2303.15413v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2303.15413](http://arxiv.org/abs/2303.15413)

    本文提出了两种去偏置的方法，一种通过增加2D扩散模型得出的分数的截断值，一种通过调整视角提示和物体空间摄像机姿态之间的差异。实验结果表明这些方法可以显著减少伪影，提高真实感。

    

    本文探讨了在文本到3D生成中出现的视角一致性问题，也称为Janus问题。这个问题来自于2D扩散模型的固有偏置，导致生成的3D对象不真实。通过对其进行研究，我们提出了两种方法来去除偏置以实现文本到3D生成的鲁棒性。第一种方法叫做score debiasing，通过逐渐增加2D扩散模型得出的分数的截断值，来达到去除偏置的效果。第二种方法叫做prompt debiasing，利用语言模型确定用户提示和视角提示之间的矛盾词语，并调整视角提示和物体空间摄像机姿态之间的差异。我们的实验结果表明，我们的方法通过显著减少伪影，提高了真实感，并在质量与速度方面取得了良好的平衡。

    The view inconsistency problem in score-distilling text-to-3D generation, also known as the Janus problem, arises from the intrinsic bias of 2D diffusion models, which leads to the unrealistic generation of 3D objects. In this work, we explore score-distilling text-to-3D generation and identify the main causes of the Janus problem. Based on these findings, we propose two approaches to debias the score-distillation frameworks for robust text-to-3D generation. Our first approach, called score debiasing, involves gradually increasing the truncation value for the score estimated by 2D diffusion models throughout the optimization process. Our second approach, called prompt debiasing, identifies conflicting words between user prompts and view prompts utilizing a language model and adjusts the discrepancy between view prompts and object-space camera poses. Our experimental results show that our methods improve realism by significantly reducing artifacts and achieve a good trade-off between fa
    
[^13]: SEAM:一种集成了句子处理与阅读中眼动的激活耦合模型

    SEAM: An Integrated Activation-Coupled Model of Sentence Processing and Eye Movements in Reading. (arXiv:2303.05221v2 [q-bio.NC] UPDATED)

    [http://arxiv.org/abs/2303.05221](http://arxiv.org/abs/2303.05221)

    SEAM是一种集成了眼动控制和句子处理的模型，为实现阅读中自然语言理解的完整数学模型迈出了重要一步。

    

    阅读中的眼动控制模型通常集中在视觉、注意、词汇和运动过程，但忽略了词汇后处理的语言处理。相比之下，句子理解过程的模型通常只关注词汇后处理的语言过程。我们提出了一种将这两种研究线索结合起来的模型，即整合眼动控制和句子处理。开发这样一个整合模型具有极大的挑战性和计算复杂性，但这样的整合是朝着完整的自然语言理解数学模型迈出的重要一步。我们将眼动控制模型SWIFT（Seelig等人，2020）与Lewis和Vasishth句子处理模型的关键组成部分（Lewis＆Vasishth，2005）结合在一起。这种整合首次变得可能，部分原因是因为。。

    Models of eye-movement control during reading, developed largely within psychology, usually focus on visual, attentional, lexical, and motor processes but neglect post-lexical language processing; by contrast, models of sentence comprehension processes, developed largely within psycholinguistics, generally focus only on post-lexical language processes. We present a model that combines these two research threads, by integrating eye-movement control and sentence processing. Developing such an integrated model is extremely challenging and computationally demanding, but such an integration is an important step toward complete mathematical models of natural language comprehension in reading. We combine the SWIFT model of eye-movement control (Seelig et al., 2020, doi:10.1016/j.jmp.2019.102313) with key components of the Lewis and Vasishth sentence processing model (Lewis & Vasishth, 2005, doi:10.1207/s15516709cog0000_25). This integration becomes possible, for the first time, due in part to
    

