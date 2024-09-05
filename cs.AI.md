# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [In the Search for Optimal Multi-view Learning Models for Crop Classification with Global Remote Sensing Data](https://arxiv.org/abs/2403.16582) | 研究调查了在全球范围内同时选择融合策略和编码器架构对作物分类具有的影响。 |
| [^2] | [Towards Measuring and Modeling "Culture" in LLMs: A Survey](https://arxiv.org/abs/2403.15412) | 这项研究调查了39篇最新论文，旨在研究大型语言模型中的文化表达和包容性，发现当前研究未对“文化”进行定义，而是在特定设计的数据集上对模型进行探究，研究了某些“文化”的方面，留下许多未被探究的有趣和重要方面，如语义领域和关于性。 |
| [^3] | [Simple and Scalable Strategies to Continually Pre-train Large Language Models](https://arxiv.org/abs/2403.08763) | 通过简单和可扩展的学习率调整、重放数据的方法，可以在不重新训练的情况下，持续预训练大型语言模型以匹配完全重新训练时的性能。 |
| [^4] | [SSM Meets Video Diffusion Models: Efficient Video Generation with Structured State Spaces](https://arxiv.org/abs/2403.07711) | 提出了一种基于状态空间模型（SSMs）的方法，用于解决使用扩散模型生成长视频序列时注意力层内存消耗增长快、限制较大的问题 |
| [^5] | [MOKA: Open-Vocabulary Robotic Manipulation through Mark-Based Visual Prompting](https://arxiv.org/abs/2403.03174) | MOKA方法利用视觉语言模型解决机器人操作任务，实现了开放词汇的机器人操作。 |
| [^6] | [Correction with Backtracking Reduces Hallucination in Summarization.](http://arxiv.org/abs/2310.16176) | 本文介绍了一种简单而有效的技术，CoBa，用于减少摘要中的幻觉。该方法通过测量条件词概率和上下文词距离的统计信息进行幻觉检测，并通过直观的回溯法进行减轻。实验证明，CoBa在减少摘要幻觉方面是有效且高效的。 |
| [^7] | [Sentinel: An Aggregation Function to Secure Decentralized Federated Learning.](http://arxiv.org/abs/2310.08097) | Sentinel是一种用于保护分散式联邦学习的防御策略，通过利用本地数据并定义一个三步聚合协议来对抗污染攻击。评估结果表明Sentinel在不同数据集和评估指标下表现良好。 |
| [^8] | [Open Gaze: An Open-Source Implementation Replicating Google's Eye Tracking Paper.](http://arxiv.org/abs/2308.13495) | 本论文提出了一个仿制谷歌眼动论文的开源实现，重点是通过整合机器学习技术，在智能手机上实现与谷歌论文相当的准确眼动追踪解决方案。 |
| [^9] | [Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities.](http://arxiv.org/abs/2307.13565) | 决策导向学习是一个新兴的机器学习范式，它集成了预测和优化，旨在优化决策。本文全面回顾了决策导向学习的相关技术，提出了分类法并进行了实证评估，探讨了当前和未来研究方向。 |
| [^10] | [Model-agnostic explainable artificial intelligence for object detection in image data.](http://arxiv.org/abs/2303.17249) | 本文设计并实现了一种新的黑盒解释方法——BODEM，它采用了局部和远程掩蔽生成多个版本的输入图像，从而比目前用于解释对象检测的其他三种最先进的方法提供更详细和有用的解释。 |
| [^11] | [Lightweight Contrastive Protein Structure-Sequence Transformation.](http://arxiv.org/abs/2303.11783) | 该论文提出了一种新的无监督学习的蛋白质结构表示预训练方法，使用强大的蛋白质语言模型和自监督结构约束，避免了破坏真实的空间结构表示和标记数据的限制。 |

# 详细

[^1]: 在利用全球遥感数据进行作物分类的多视图学习模型的最佳选择研究

    In the Search for Optimal Multi-view Learning Models for Crop Classification with Global Remote Sensing Data

    [https://arxiv.org/abs/2403.16582](https://arxiv.org/abs/2403.16582)

    研究调查了在全球范围内同时选择融合策略和编码器架构对作物分类具有的影响。

    

    作物分类在研究作物模式变化、资源管理和碳固存中具有至关重要的作用。采用数据驱动技术进行预测时，利用各种时间数据源是必要的。深度学习模型已被证明对将时间序列数据映射到高级表示以进行预测任务非常有效。然而，当处理多个输入模式时，它们面临着重大挑战。文献对多视图学习（MVL）场景提供了有限的指导，主要集中在探索具有特定编码器的融合策略，并在局部地区对其进行验证。相反，我们研究了在全球范围内对农田土地和作物类型进行分类时同时选择融合策略和编码器架构的影响。

    arXiv:2403.16582v1 Announce Type: cross  Abstract: Crop classification is of critical importance due to its role in studying crop pattern changes, resource management, and carbon sequestration. When employing data-driven techniques for its prediction, utilizing various temporal data sources is necessary. Deep learning models have proven to be effective for this task by mapping time series data to high-level representation for prediction. However, they face substantial challenges when dealing with multiple input patterns. The literature offers limited guidance for Multi-View Learning (MVL) scenarios, as it has primarily focused on exploring fusion strategies with specific encoders and validating them in local regions. In contrast, we investigate the impact of simultaneous selection of the fusion strategy and the encoder architecture evaluated on a global-scale cropland and crop-type classifications. We use a range of five fusion strategies (Input, Feature, Decision, Ensemble, Hybrid) an
    
[^2]: 在LLMs中测量和建模“文化”：一项调查

    Towards Measuring and Modeling "Culture" in LLMs: A Survey

    [https://arxiv.org/abs/2403.15412](https://arxiv.org/abs/2403.15412)

    这项研究调查了39篇最新论文，旨在研究大型语言模型中的文化表达和包容性，发现当前研究未对“文化”进行定义，而是在特定设计的数据集上对模型进行探究，研究了某些“文化”的方面，留下许多未被探究的有趣和重要方面，如语义领域和关于性。

    

    我们呈现了对39篇最新论文的调查，旨在研究大型语言模型中的文化表达和包容性。我们观察到，没有一篇研究定义“文化”，这是一个复杂、多层面的概念；相反，它们在一些特别设计的数据集上对模型进行探究，这些数据集代表了某些“文化”的方面。我们将这些方面称为文化的代理，并将它们组织在人口统计、语义和语言文化交互代理的三个维度上。我们还对采用的探查方法进行了分类。我们的分析表明，只有“文化”的某些方面，如价值观和目标，被研究了，留下了几个其他有趣且重要的方面，特别是大量语义领域和关于性（Hershcovich等人，2022）的未被探究。另外两个关键的空白是目前方法的鲁棒性和情境性的缺乏。基于这些观察结果，

    arXiv:2403.15412v1 Announce Type: cross  Abstract: We present a survey of 39 recent papers that aim to study cultural representation and inclusion in large language models. We observe that none of the studies define "culture," which is a complex, multifaceted concept; instead, they probe the models on some specially designed datasets which represent certain aspects of "culture." We call these aspects the proxies of cultures, and organize them across three dimensions of demographic, semantic and linguistic-cultural interaction proxies. We also categorize the probing methods employed. Our analysis indicates that only certain aspects of "culture," such as values and objectives, have been studied, leaving several other interesting and important facets, especially the multitude of semantic domains (Thompson et al., 2020) and aboutness (Hershcovich et al., 2022), unexplored. Two other crucial gaps are the lack of robustness and situatedness of the current methods. Based on these observations
    
[^3]: 持续预训练大型语言模型的简单可扩展策略

    Simple and Scalable Strategies to Continually Pre-train Large Language Models

    [https://arxiv.org/abs/2403.08763](https://arxiv.org/abs/2403.08763)

    通过简单和可扩展的学习率调整、重放数据的方法，可以在不重新训练的情况下，持续预训练大型语言模型以匹配完全重新训练时的性能。

    

    大型语言模型（LLMs）通常在数十亿的标记上进行常规预训练，一旦有新数据可用就重新开始该过程。一个更有效率的解决方案是持续预训练这些模型，与重新训练相比能节省大量计算资源。然而，新数据引起的分布转移通常会导致在以前数据上降低性能或无法适应新数据。在本工作中，我们展示了一种简单且可扩展的学习率（LR）重新升温、LR重新衰减和重放上一数据的组合足以与完全从头开始重新训练在所有可用数据上的性能相匹配，从最终损失和语言模型（LM）评估基准的角度衡量。具体而言，我们展示了在两个常用的LLM预训练数据集（英语→英语）之间的弱但现实的分布转移以及更强烈的分布转移（英语→德语）下的情况。

    arXiv:2403.08763v1 Announce Type: cross  Abstract: Large language models (LLMs) are routinely pre-trained on billions of tokens, only to start the process over again once new data becomes available. A much more efficient solution is to continually pre-train these models, saving significant compute compared to re-training. However, the distribution shift induced by new data typically results in degraded performance on previous data or poor adaptation to the new data. In this work, we show that a simple and scalable combination of learning rate (LR) re-warming, LR re-decaying, and replay of previous data is sufficient to match the performance of fully re-training from scratch on all available data, as measured by final loss and language model (LM) evaluation benchmarks. Specifically, we show this for a weak but realistic distribution shift between two commonly used LLM pre-training datasets (English$\rightarrow$English) and a stronger distribution shift (English$\rightarrow$German) at th
    
[^4]: SSM遇上视频扩散模型: 结构化状态空间下的高效视频生成

    SSM Meets Video Diffusion Models: Efficient Video Generation with Structured State Spaces

    [https://arxiv.org/abs/2403.07711](https://arxiv.org/abs/2403.07711)

    提出了一种基于状态空间模型（SSMs）的方法，用于解决使用扩散模型生成长视频序列时注意力层内存消耗增长快、限制较大的问题

    

    鉴于图像生成通过扩散模型取得的显著成就，研究界对将这些模型扩展到视频生成表现出越来越大的兴趣。最近用于视频生成的扩散模型主要利用注意力层来提取时间特征。然而，由于注意力层的内存消耗随着序列长度的增加呈二次增长，这种限制在尝试使用扩散模型生成更长视频序列时会带来重大挑战。为了克服这一挑战，我们提出利用状态空间模型（SSMs）。由于相对于序列长度，SSMs具有线性内存消耗，最近已经引起了越来越多的关注。在实验中，我们首先通过使用UCF101这一视频生成的标准基准来评估我们基于SSM的模型。此外，为探讨SSMs在更长视频生成中的潜力，

    arXiv:2403.07711v1 Announce Type: cross  Abstract: Given the remarkable achievements in image generation through diffusion models, the research community has shown increasing interest in extending these models to video generation. Recent diffusion models for video generation have predominantly utilized attention layers to extract temporal features. However, attention layers are limited by their memory consumption, which increases quadratically with the length of the sequence. This limitation presents significant challenges when attempting to generate longer video sequences using diffusion models. To overcome this challenge, we propose leveraging state-space models (SSMs). SSMs have recently gained attention as viable alternatives due to their linear memory consumption relative to sequence length. In the experiments, we first evaluate our SSM-based model with UCF101, a standard benchmark of video generation. In addition, to investigate the potential of SSMs for longer video generation, 
    
[^5]: MOKA：基于标记的视觉提示实现开放词汇的机器人操作

    MOKA: Open-Vocabulary Robotic Manipulation through Mark-Based Visual Prompting

    [https://arxiv.org/abs/2403.03174](https://arxiv.org/abs/2403.03174)

    MOKA方法利用视觉语言模型解决机器人操作任务，实现了开放词汇的机器人操作。

    

    开放词汇的泛化要求机器人系统执行涉及复杂和多样化环境以及任务目标的任务。本文提出了一种名为MOKA（Marking Open-vocabulary Keypoint Affordances）的方法，利用视觉语言模型（VLMs）来解决由自由形式语言描述指定的机器人操作任务。

    arXiv:2403.03174v1 Announce Type: cross  Abstract: Open-vocabulary generalization requires robotic systems to perform tasks involving complex and diverse environments and task goals. While the recent advances in vision language models (VLMs) present unprecedented opportunities to solve unseen problems, how to utilize their emergent capabilities to control robots in the physical world remains an open question. In this paper, we present MOKA (Marking Open-vocabulary Keypoint Affordances), an approach that employs VLMs to solve robotic manipulation tasks specified by free-form language descriptions. At the heart of our approach is a compact point-based representation of affordance and motion that bridges the VLM's predictions on RGB images and the robot's motions in the physical world. By prompting a VLM pre-trained on Internet-scale data, our approach predicts the affordances and generates the corresponding motions by leveraging the concept understanding and commonsense knowledge from br
    
[^6]: 通过回溯法纠正，减少摘要中的幻觉

    Correction with Backtracking Reduces Hallucination in Summarization. (arXiv:2310.16176v1 [cs.CL])

    [http://arxiv.org/abs/2310.16176](http://arxiv.org/abs/2310.16176)

    本文介绍了一种简单而有效的技术，CoBa，用于减少摘要中的幻觉。该方法通过测量条件词概率和上下文词距离的统计信息进行幻觉检测，并通过直观的回溯法进行减轻。实验证明，CoBa在减少摘要幻觉方面是有效且高效的。

    

    摘要生成旨在生成源文件的自然语言摘要，既简洁又保留重要元素。尽管最近取得了一些进展，但神经文本摘要模型容易产生幻觉（或更准确地说是混淆），即生成的摘要包含源文件中没有根据的细节。在本文中，我们引入了一种简单而有效的技术，CoBa，用于减少摘要中的幻觉。该方法基于两个步骤：幻觉检测和减轻。我们展示了通过测量有关条件词概率和上下文词距离的简单统计信息可以实现前者。此外，我们还证明了直观的回溯法在减轻幻觉方面的惊人效果。我们在三个文本摘要基准数据集上对所提出的方法进行了全面评估。结果表明，CoBa在减少摘要幻觉方面是有效且高效的。

    Abstractive summarization aims at generating natural language summaries of a source document that are succinct while preserving the important elements. Despite recent advances, neural text summarization models are known to be susceptible to hallucinating (or more correctly confabulating), that is to produce summaries with details that are not grounded in the source document. In this paper, we introduce a simple yet efficient technique, CoBa, to reduce hallucination in abstractive summarization. The approach is based on two steps: hallucination detection and mitigation. We show that the former can be achieved through measuring simple statistics about conditional word probabilities and distance to context words. Further, we demonstrate that straight-forward backtracking is surprisingly effective at mitigation. We thoroughly evaluate the proposed method with prior art on three benchmark datasets for text summarization. The results show that CoBa is effective and efficient in reducing hall
    
[^7]: Sentinel: 一种用于保护分散式联邦学习的聚合函数

    Sentinel: An Aggregation Function to Secure Decentralized Federated Learning. (arXiv:2310.08097v1 [cs.DC])

    [http://arxiv.org/abs/2310.08097](http://arxiv.org/abs/2310.08097)

    Sentinel是一种用于保护分散式联邦学习的防御策略，通过利用本地数据并定义一个三步聚合协议来对抗污染攻击。评估结果表明Sentinel在不同数据集和评估指标下表现良好。

    

    将联邦学习（FL）快速整合到网络中涵盖了网络管理、服务质量和网络安全等各个方面，同时保护数据隐私。在这种情况下，分散式联邦学习（DFL）作为一种创新范式，用于训练协作模型，解决了单点失效的限制。然而，FL和DFL的安全性和可信性受到污染攻击的影响，从而对其性能产生负面影响。现有的防御机制针对集中式FL进行设计，并未充分利用DFL的特点。因此，本文引入了Sentinel，一种在DFL中对抗污染攻击的防御策略。Sentinel利用本地数据的可访问性，定义了一个三步聚合协议，包括相似性过滤、引导验证和标准化，以防止恶意模型更新。通过使用不同数据集和不同的评估指标对Sentinel进行了评估。

    The rapid integration of Federated Learning (FL) into networking encompasses various aspects such as network management, quality of service, and cybersecurity while preserving data privacy. In this context, Decentralized Federated Learning (DFL) emerges as an innovative paradigm to train collaborative models, addressing the single point of failure limitation. However, the security and trustworthiness of FL and DFL are compromised by poisoning attacks, negatively impacting its performance. Existing defense mechanisms have been designed for centralized FL and they do not adequately exploit the particularities of DFL. Thus, this work introduces Sentinel, a defense strategy to counteract poisoning attacks in DFL. Sentinel leverages the accessibility of local data and defines a three-step aggregation protocol consisting of similarity filtering, bootstrap validation, and normalization to safeguard against malicious model updates. Sentinel has been evaluated with diverse datasets and various 
    
[^8]: 开放注视：一个仿制谷歌眼动论文的开源实现

    Open Gaze: An Open-Source Implementation Replicating Google's Eye Tracking Paper. (arXiv:2308.13495v1 [cs.CV])

    [http://arxiv.org/abs/2308.13495](http://arxiv.org/abs/2308.13495)

    本论文提出了一个仿制谷歌眼动论文的开源实现，重点是通过整合机器学习技术，在智能手机上实现与谷歌论文相当的准确眼动追踪解决方案。

    

    眼动已经成为视觉研究、语言分析和可用性评估等不同领域的重要工具。然而，大多数先前的研究集中在使用专门的、昂贵的眼动追踪硬件的扩展式桌面显示器上。尽管智能手机的普及率和使用频率很高，但对于智能手机上的眼球移动模式却鲜有见解。在本文中，我们提出了一个基于智能手机的开源注视追踪实现，模拟了谷歌论文提出的方法论（其源代码仍然是专有的）。我们的重点是在不需要额外硬件的情况下达到与谷歌论文方法相当的准确度。通过整合机器学习技术，我们揭示了一种本地于智能手机的准确眼动追踪解决方案。我们的方法展示了与最先进的移动眼动追踪器相当的精度。

    Eye tracking has been a pivotal tool in diverse fields such as vision research, language analysis, and usability assessment. The majority of prior investigations, however, have concentrated on expansive desktop displays employing specialized, costly eye tracking hardware that lacks scalability. Remarkably little insight exists into ocular movement patterns on smartphones, despite their widespread adoption and significant usage. In this manuscript, we present an open-source implementation of a smartphone-based gaze tracker that emulates the methodology proposed by a GooglePaper (whose source code remains proprietary). Our focus is on attaining accuracy comparable to that attained through the GooglePaper's methodology, without the necessity for supplementary hardware. Through the integration of machine learning techniques, we unveil an accurate eye tracking solution that is native to smartphones. Our approach demonstrates precision akin to the state-of-the-art mobile eye trackers, which 
    
[^9]: 决策导向学习：基础、现状、基准和未来机会

    Decision-Focused Learning: Foundations, State of the Art, Benchmark and Future Opportunities. (arXiv:2307.13565v1 [cs.LG])

    [http://arxiv.org/abs/2307.13565](http://arxiv.org/abs/2307.13565)

    决策导向学习是一个新兴的机器学习范式，它集成了预测和优化，旨在优化决策。本文全面回顾了决策导向学习的相关技术，提出了分类法并进行了实证评估，探讨了当前和未来研究方向。

    

    决策导向学习（DFL）是一种新兴的机器学习范式，它训练模型以优化决策，在一个端到端的系统中集成了预测和优化。这个范式有望在许多实际应用中革命性地改变决策制定，这些应用在不确定性下运作，在这些决策模型中估计未知参数经常成为一个重要障碍。本文对DFL进行了全面的回顾。它对各种技术进行了深入分析，以整合机器学习和优化模型，引入了一种根据其独特特征来区分DFL方法的分类法，并对这些方法进行了广泛的实证评估，提出了适用于DFL的合适基准数据集和任务。最后，本研究提供了关于DFL研究中当前和潜在未来方向的宝贵见解。

    Decision-focused learning (DFL) is an emerging paradigm in machine learning which trains a model to optimize decisions, integrating prediction and optimization in an end-to-end system. This paradigm holds the promise to revolutionize decision-making in many real-world applications which operate under uncertainty, where the estimation of unknown parameters within these decision models often becomes a substantial roadblock. This paper presents a comprehensive review of DFL. It provides an in-depth analysis of the various techniques devised to integrate machine learning and optimization models introduces a taxonomy of DFL methods distinguished by their unique characteristics, and conducts an extensive empirical evaluation of these methods proposing suitable benchmark dataset and tasks for DFL. Finally, the study provides valuable insights into current and potential future avenues in DFL research.
    
[^10]: 面向对象检测的模型无关可解释人工智能

    Model-agnostic explainable artificial intelligence for object detection in image data. (arXiv:2303.17249v1 [cs.CV])

    [http://arxiv.org/abs/2303.17249](http://arxiv.org/abs/2303.17249)

    本文设计并实现了一种新的黑盒解释方法——BODEM，它采用了局部和远程掩蔽生成多个版本的输入图像，从而比目前用于解释对象检测的其他三种最先进的方法提供更详细和有用的解释。

    

    对象检测是计算机视觉中的基本任务之一，通过开发大型复杂的深度学习模型已经取得了很大进展。然而，缺乏透明度是一个重要的挑战，可能妨碍这些模型的广泛应用。可解释的人工智能是一个研究领域，其中开发方法来帮助用户理解基于人工智能的系统的行为、决策逻辑和漏洞。本文为了解释基于人工智能的对象检测系统设计和实现了一种名为Black-box Object Detection Explanation by Masking（BODEM）的黑盒说明方法，采用新的掩蔽方法。我们提出了局部和远程掩蔽来生成输入图像的多个版本。局部掩蔽用于干扰目标对象内的像素，以了解对象检测器对这些变化的反应，而远程掩蔽则用于研究对象检测器在图像背景上的行为。我们在三个基准数据集上的实验表明，与用于解释对象检测的其他三种最先进的方法相比，BODEM提供了更详细和有用的说明。

    Object detection is a fundamental task in computer vision, which has been greatly progressed through developing large and intricate deep learning models. However, the lack of transparency is a big challenge that may not allow the widespread adoption of these models. Explainable artificial intelligence is a field of research where methods are developed to help users understand the behavior, decision logics, and vulnerabilities of AI-based systems. Black-box explanation refers to explaining decisions of an AI system without having access to its internals. In this paper, we design and implement a black-box explanation method named Black-box Object Detection Explanation by Masking (BODEM) through adopting a new masking approach for AI-based object detection systems. We propose local and distant masking to generate multiple versions of an input image. Local masks are used to disturb pixels within a target object to figure out how the object detector reacts to these changes, while distant ma
    
[^11]: 轻量级对比蛋白质结构-序列变换

    Lightweight Contrastive Protein Structure-Sequence Transformation. (arXiv:2303.11783v1 [q-bio.BM])

    [http://arxiv.org/abs/2303.11783](http://arxiv.org/abs/2303.11783)

    该论文提出了一种新的无监督学习的蛋白质结构表示预训练方法，使用强大的蛋白质语言模型和自监督结构约束，避免了破坏真实的空间结构表示和标记数据的限制。

    

    在大多数蛋白质下游应用中，无标签的预训练蛋白质结构模型是关键基础。传统的结构预训练方法遵循成熟的自然语言预训练方法，例如去噪重构和掩码语言建模，但通常会破坏真实的空间结构表示。其他常见的预训练方法可能会预测一组固定的预定对象类别，其中受限的监督方式限制了它们的通用性和可用性，因为需要额外的标记数据来指定任何其他的蛋白质概念。在这项工作中，我们引入了一种新的无监督蛋白质结构表示预训练方法，其中使用强大的蛋白质语言模型。特别地，我们首先建议利用现有的预训练语言模型通过无监督的对比对齐来指导结构模型的学习。此外，我们提出了一种自监督结构约束，以进一步学习内在的蛋白质结构表示形式。

    Pretrained protein structure models without labels are crucial foundations for the majority of protein downstream applications. The conventional structure pretraining methods follow the mature natural language pretraining methods such as denoised reconstruction and masked language modeling but usually destroy the real representation of spatial structures. The other common pretraining methods might predict a fixed set of predetermined object categories, where a restricted supervised manner limits their generality and usability as additional labeled data is required to specify any other protein concepts. In this work, we introduce a novel unsupervised protein structure representation pretraining with a robust protein language model. In particular, we first propose to leverage an existing pretrained language model to guide structure model learning through an unsupervised contrastive alignment. In addition, a self-supervised structure constraint is proposed to further learn the intrinsic i
    

