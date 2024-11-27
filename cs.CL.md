# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Against The Achilles' Heel: A Survey on Red Teaming for Generative Models](https://arxiv.org/abs/2404.00629) | 通过对生成模型的红队测试进行了广泛调查，引入了基于语言模型能力的细粒度攻击策略分类体系，并开发了一个统一各种自动红队测试方法的搜索框架。 |
| [^2] | [A Condensed Transition Graph Framework for Zero-shot Link Prediction with Large Language Models](https://arxiv.org/abs/2402.10779) | 提出了一种用于零样本链接预测的紧凑转换图框架，能够在线性时间内编码所有路径的信息，并可解决大型语言模型性能受限的问题。 |
| [^3] | [UniMS-RAG: A Unified Multi-source Retrieval-Augmented Generation for Personalized Dialogue Systems.](http://arxiv.org/abs/2401.13256) | 这项研究提出了一种统一多源检索增强生成系统（UniMS-RAG），通过统一知识源选择、知识检索和回复生成三个子任务，使语言模型能够根据需求自适应地检索证据和评估关联性，从而生成个性化的回复。 |
| [^4] | [A Survey on Multimodal Large Language Models.](http://arxiv.org/abs/2306.13549) | 本文追踪和总结了多模态大语言模型（MLLM）的最新进展，包括多模态指令调整、多模态上下文学习、多模态思维链和LLM辅助视觉推理等应用，指出了现有挑战和有前途的研究方向。 |
| [^5] | [Stars Are All You Need: A Distantly Supervised Pyramid Network for Document-Level End-to-End Sentiment Analysis.](http://arxiv.org/abs/2305.01710) | 本文提出了一种文档级端到端情感分析方法，通过星级评分标签，实现方面检测、情感分析和评分预测，具有良好的性能和可解释性。 |

# 详细

[^1]: 针对阿喀琉斯之踵：生成模型红队测试的调查

    Against The Achilles' Heel: A Survey on Red Teaming for Generative Models

    [https://arxiv.org/abs/2404.00629](https://arxiv.org/abs/2404.00629)

    通过对生成模型的红队测试进行了广泛调查，引入了基于语言模型能力的细粒度攻击策略分类体系，并开发了一个统一各种自动红队测试方法的搜索框架。

    

    生成模型正迅速普及并被整合到日常应用中，但相关的安全问题引起了人们的担忧，因为各种漏洞不断暴露。面对这一问题，红队测试领域正在快速增长，强调了对整个流程进行全面组织并解决社区新兴主题的需求。我们的广泛调查涵盖了120多篇论文，引入了一个基于语言模型固有能力的细粒度攻击策略分类体系。此外，我们开发了一个统一各种自动红队测试方法的搜索框架。此外，我们的调查涵盖了新领域，包括多模式攻击和防御、多语言模型风险、无害查询的过度使用以及下游应用的安全性。

    arXiv:2404.00629v1 Announce Type: new  Abstract: Generative models are rapidly gaining popularity and being integrated into everyday applications, raising concerns over their safety issues as various vulnerabilities are exposed. Faced with the problem, the field of red teaming is experiencing fast-paced growth, which highlights the need for a comprehensive organization covering the entire pipeline and addressing emerging topics for the community. Our extensive survey, which examines over 120 papers, introduces a taxonomy of fine-grained attack strategies grounded in the inherent capabilities of language models. Additionally, we have developed the searcher framework that unifies various automatic red teaming approaches. Moreover, our survey covers novel areas including multimodal attacks and defenses, risks around multilingual models, overkill of harmless queries, and safety of downstream applications. We hope this survey can provide a systematic perspective on the field and unlock new 
    
[^2]: 一种用于零样本链接预测的紧凑转换图框架与大型语言模型

    A Condensed Transition Graph Framework for Zero-shot Link Prediction with Large Language Models

    [https://arxiv.org/abs/2402.10779](https://arxiv.org/abs/2402.10779)

    提出了一种用于零样本链接预测的紧凑转换图框架，能够在线性时间内编码所有路径的信息，并可解决大型语言模型性能受限的问题。

    

    零样本链接预测（ZSLP）旨在自动识别给定实体之间的关系。现有方法主要利用辅助信息来预测给定头实体和其关系时的尾实体，然而面临挑战，原因是有时缺乏这些详细信息，并且基于语义相似性来预测尾实体的固有简单性。尽管大型语言模型（LLMs）为以零样本方式预测头实体和尾实体之间的未观察到的关系提供了有前途的解决方案，但其性能仍受限于无法利用两个实体之间所有（指数多）路径信息的能力，这些信息对于共同指示它们的关系类型至关重要。为了解决这个问题，在这项工作中，我们引入了一种用于零样本链接预测的紧凑转换图框架（CTLP），它以线性时间编码了所有路径的信息。

    arXiv:2402.10779v1 Announce Type: new  Abstract: Zero-shot link prediction (ZSLP) on knowledge graphs aims at automatically identifying relations between given entities. Existing methods primarily employ auxiliary information to predict tail entity given head entity and its relation, yet face challenges due to the occasional unavailability of such detailed information and the inherent simplicity of predicting tail entities based on semantic similarities. Even though Large Language Models (LLMs) offer a promising solution to predict unobserved relations between the head and tail entity in a zero-shot manner, their performance is still restricted due to the inability to leverage all the (exponentially many) paths' information between two entities, which are critical in collectively indicating their relation types. To address this, in this work, we introduce a Condensed Transition Graph Framework for Zero-Shot Link Prediction (CTLP), which encodes all the paths' information in linear time
    
[^3]: UniMS-RAG: 用于个性化对话系统的统一多源检索增强生成模型

    UniMS-RAG: A Unified Multi-source Retrieval-Augmented Generation for Personalized Dialogue Systems. (arXiv:2401.13256v1 [cs.CL])

    [http://arxiv.org/abs/2401.13256](http://arxiv.org/abs/2401.13256)

    这项研究提出了一种统一多源检索增强生成系统（UniMS-RAG），通过统一知识源选择、知识检索和回复生成三个子任务，使语言模型能够根据需求自适应地检索证据和评估关联性，从而生成个性化的回复。

    

    大型语言模型在许多自然语言理解和生成任务中展示出了非凡的能力。然而，在对话系统中涉及到多个信息源时，个性化问题仍然是一个令人向往的属性。为了更好地计划和整合多个信息源在生成个性化回复中的使用，我们首先将其分解为三个子任务：知识源选择、知识检索和回复生成。然后，我们提出了一种新颖的统一多源检索增强生成系统（UniMS-RAG）。具体来说，我们在训练期间使用相同的序列到序列范式将这三个子任务统一起来，通过使用特殊的令牌，即行动令牌和评估令牌，能够自适应地检索证据并评估关联性。使语言模型能够生成行动令牌有助于与各种知识源进行交互，使其能够适应其上下文和生成个性化的回复。

    Large Language Models (LLMs) has shown exceptional capabilities in many natual language understanding and generation tasks. However, the personalization issue still remains a much-coveted property, especially when it comes to the multiple sources involved in the dialogue system. To better plan and incorporate the use of multiple sources in generating personalized response, we firstly decompose it into three sub-tasks: Knowledge Source Selection, Knowledge Retrieval, and Response Generation. We then propose a novel Unified Multi-Source Retrieval-Augmented Generation system (UniMS-RAG) Specifically, we unify these three sub-tasks with different formulations into the same sequence-to-sequence paradigm during the training, to adaptively retrieve evidences and evaluate the relevance on-demand using special tokens, called acting tokens and evaluation tokens. Enabling language models to generate acting tokens facilitates interaction with various knowledge sources, allowing them to adapt their
    
[^4]: 多模态大语言模型综述

    A Survey on Multimodal Large Language Models. (arXiv:2306.13549v1 [cs.CV])

    [http://arxiv.org/abs/2306.13549](http://arxiv.org/abs/2306.13549)

    本文追踪和总结了多模态大语言模型（MLLM）的最新进展，包括多模态指令调整、多模态上下文学习、多模态思维链和LLM辅助视觉推理等应用，指出了现有挑战和有前途的研究方向。

    

    多模态大语言模型（MLLM）是一种新兴的研究热点，使用强大的大语言模型作为大脑执行多模态任务。MLLM 的惊人能力，如基于图像编写故事和无OCR数学推理等，在传统方法中很少见，表明了通向人工智能的潜在路径。本文旨在追踪和总结 MLLM 的最新进展。首先，我们介绍了 MLLM 的构成，概述了相关概念。然后，讨论了关键技术和应用，包括多模态指令调整（M-IT）、多模态上下文学习（M-ICL）、多模态思维链（M-CoT）和LLM辅助视觉推理（LAVR）。最后，我们讨论了现有的挑战，并指出了有前途的研究方向。鉴于 MLLM 时代才刚刚开始，我们会不断更新这个综述，并希望能激发更多的研究。

    Multimodal Large Language Model (MLLM) recently has been a new rising research hotspot, which uses powerful Large Language Models (LLMs) as a brain to perform multimodal tasks. The surprising emergent capabilities of MLLM, such as writing stories based on images and OCR-free math reasoning, are rare in traditional methods, suggesting a potential path to artificial general intelligence. In this paper, we aim to trace and summarize the recent progress of MLLM. First of all, we present the formulation of MLLM and delineate its related concepts. Then, we discuss the key techniques and applications, including Multimodal Instruction Tuning (M-IT), Multimodal In-Context Learning (M-ICL), Multimodal Chain of Thought (M-CoT), and LLM-Aided Visual Reasoning (LAVR). Finally, we discuss existing challenges and point out promising research directions. In light of the fact that the era of MLLM has only just begun, we will keep updating this survey and hope it can inspire more research. An associated
    
[^5]: 星辰即你所需：用远程监督金字塔网络进行文档级端到端情感分析

    Stars Are All You Need: A Distantly Supervised Pyramid Network for Document-Level End-to-End Sentiment Analysis. (arXiv:2305.01710v1 [cs.CL])

    [http://arxiv.org/abs/2305.01710](http://arxiv.org/abs/2305.01710)

    本文提出了一种文档级端到端情感分析方法，通过星级评分标签，实现方面检测、情感分析和评分预测，具有良好的性能和可解释性。

    

    本文提出了文档级端到端情感分析方法，可以通过星级评分标签对在线评论中表达的方面和评论情感进行有效的统一分析。我们假设星级评分标签是评论中各方面评分的“粗粒度综合”。我们提出了一种远程监督的金字塔网络（DSPN），只用文档星级评分标签进行训练，即可有效地执行方面-类别检测、方面-类别情感分析和评分预测。通过以端到端的方式执行这三个相关的情感子任务，DSPN可以提取评论中提到的方面，确定相应的情感，并预测星级评分标签。我们在英文和汉语多方面评论数据集上评估了DSPN，发现仅使用星级评分标签进行监督，DSPN的性能与各种基准模型相当。我们还展示了DSPN在评论上的可解释性输出，以说明金字塔网络的结构。

    In this paper, we propose document-level end-to-end sentiment analysis to efficiently understand aspect and review sentiment expressed in online reviews in a unified manner. In particular, we assume that star rating labels are a "coarse-grained synthesis" of aspect ratings across in the review. We propose a Distantly Supervised Pyramid Network (DSPN) to efficiently perform Aspect-Category Detection, Aspect-Category Sentiment Analysis, and Rating Prediction using only document star rating labels for training. By performing these three related sentiment subtasks in an end-to-end manner, DSPN can extract aspects mentioned in the review, identify the corresponding sentiments, and predict the star rating labels. We evaluate DSPN on multi-aspect review datasets in English and Chinese and find that with only star rating labels for supervision, DSPN can perform comparably well to a variety of benchmark models. We also demonstrate the interpretability of DSPN's outputs on reviews to show the py
    

