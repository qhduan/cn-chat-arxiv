# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Medication Recommendation via Domain Knowledge Informed Deep Learning.](http://arxiv.org/abs/2305.19604) | 提出一种基于动态领域知识的药物推荐框架DKINet，将领域知识与患者临床表现相结合，此为首次实验。 |
| [^2] | [Towards Semi-supervised Universal Graph Classification.](http://arxiv.org/abs/2305.19598) | 该论文提出了一种新型图神经网络框架UGNN， 解决了半监督普适图分类问题，通过估计未标记图的确定性解决了类别偏移，具有最新性能。 |
| [^3] | [AdANNS: A Framework for Adaptive Semantic Search.](http://arxiv.org/abs/2305.19435) | AdANNS是一种自适应语义搜索框架，利用不同容量的自适应表示形式可以获得更好的精度-计算折衷权衡，相似度计算越接近的数据点将使用更低容量的表示形式进行计算，演示了最先进的精度-计算折衷权衡。 |
| [^4] | [DuoSearch: A Novel Search Engine for Bulgarian Historical Documents.](http://arxiv.org/abs/2305.19392) | 本文提出了一种新的历史文件搜索引擎DuoSearch，该引擎使用神经网络机器学习方法解决数字化历史文献搜索中的正字法变体和字符识别错误问题，并成功应用于保加利亚中期到20世纪中期的历史报纸集合。 |
| [^5] | [Mitigating Test-Time Bias for Fair Image Retrieval.](http://arxiv.org/abs/2305.19329) | 本文提出了后置偏差缓解（PBM）技术，解决了如何在中性文本查询的情况下实现公平的图像检索。该方法在实际数据集中实现了最低的偏差。 |
| [^6] | [Unified Embedding: Battle-Tested Feature Representations for Web-Scale ML Systems.](http://arxiv.org/abs/2305.12102) | 本文介绍了一种名为“特征复用”的框架，它使用单一的表示空间 能够高效有效地学习高质量的特征嵌入，同时区分不同的分类特征。通过在多个公共数据集和新数据集“Web-Available Image Search (WAIS)”上的测试，我们展示了这种方法的优于现有技术的表现。 |
| [^7] | [EvalRS 2023. Well-Rounded Recommender Systems For Real-World Deployments.](http://arxiv.org/abs/2304.07145) | EvalRS 2023旨在探讨推荐系统的全面评估，关注其在现实场景下的实际影响。过去只有准确度的测量方法可能无法全面评估其性能，公平性、偏见、有用性和信息量等方面也应该被关注。本次研讨会是去年CIKM研讨会的继承和发展，带有实际操作性和互动性。 |
| [^8] | [RARR: Researching and Revising What Language Models Say, Using Language Models.](http://arxiv.org/abs/2210.08726) | RARR是一个可以对不确定信息进行研究和修订的系统，它可以自动找到文本生成模型输出的归因并修正不支持的内容。 |
| [^9] | [Variational Open-Domain Question Answering.](http://arxiv.org/abs/2210.06345) | 本文介绍了变分开放领域（VOD）框架，提出了一种新的自归一化的Rényi变分界的估计方法，可用于训练具有检索增强功能的模型，例如阅读器-检索器BERT-sized模型，并实现了在多项选择医学考试问题上的优异表现。 |
| [^10] | [CONE: An Efficient COarse-to-fiNE Alignment Framework for Long Video Temporal Grounding.](http://arxiv.org/abs/2209.10918) | 本文提出了CONE，一个高效的粗-细对齐框架，可用于长视频时间定位。CONE通过基于查询的窗口选择策略和对比学习机制提升了多模态对齐，并在两个大规模长视频时间定位基准测试中取得最先进结果。 |
| [^11] | [Understanding Diversity in Session-Based Recommendation.](http://arxiv.org/abs/2208.13453) | 这篇文章探讨了基于会话的推荐系统在准确度和多样性方面的表现，并提供了设计多样化SBRS的指导。 |

# 详细

[^1]: 通过领域知识启示的深度学习进行药物推荐

    Medication Recommendation via Domain Knowledge Informed Deep Learning. (arXiv:2305.19604v1 [cs.AI])

    [http://arxiv.org/abs/2305.19604](http://arxiv.org/abs/2305.19604)

    提出一种基于动态领域知识的药物推荐框架DKINet，将领域知识与患者临床表现相结合，此为首次实验。

    

    药物推荐是医疗保健的基本但至关重要的分支，提供机会为复杂健康状况的患者支持临床医生更精确的药物处方。从电子健康记录（EHR）中学习推荐药物是先前研究中最常见的方法。然而，大多数研究忽视了根据患者的EHR中的临床表现纳入领域知识的问题。为了解决这些问题，我们提出了一种新颖的基于动态领域知识的药物推荐框架，即领域知识启示网络（DKINet），用于将领域知识与可观察的患者临床表现相结合。特别是，我们首先设计了一个基于领域知识的编码器来捕捉领域信息，然后开发了一个数据驱动的编码器将领域知识整合到可观察的EHR中。

    Medication recommendation is a fundamental yet crucial branch of healthcare, which provides opportunities to support clinical physicians with more accurate medication prescriptions for patients with complex health conditions. Learning from electronic health records (EHR) to recommend medications is the most common way in previous studies. However, most of them neglect incorporating domain knowledge according to the clinical manifestations in the EHR of the patient. To address these issues, we propose a novel \textbf{D}omain \textbf{K}nowledge \textbf{I}nformed \textbf{Net}work (DKINet) to integrate domain knowledge with observable clinical manifestations of the patient, which is the first dynamic domain knowledge informed framework toward medication recommendation. In particular, we first design a knowledge-driven encoder to capture the domain information and then develop a data-driven encoder to integrate domain knowledge into the observable EHR. To endow the model with the capability
    
[^2]: 半监督普适图分类

    Towards Semi-supervised Universal Graph Classification. (arXiv:2305.19598v1 [cs.LG])

    [http://arxiv.org/abs/2305.19598](http://arxiv.org/abs/2305.19598)

    该论文提出了一种新型图神经网络框架UGNN， 解决了半监督普适图分类问题，通过估计未标记图的确定性解决了类别偏移，具有最新性能。

    

    近年来，图神经网络推动了图分类的最新技术。本文提出了一种名为UGNN的新型图神经网络框架，以从子图角度利用未标记数据来解决半监督普适图分类问题。该问题的挑战在于缺乏标签数据和可能的类别偏移。UGNN通过估计未标记图的确定性来解决类别偏移，使其在具有挑战性的数据集上具有最新性能。

    Graph neural networks have pushed state-of-the-arts in graph classifications recently. Typically, these methods are studied within the context of supervised end-to-end training, which necessities copious task-specific labels. However, in real-world circumstances, labeled data could be limited, and there could be a massive corpus of unlabeled data, even from unknown classes as a complementary. Towards this end, we study the problem of semi-supervised universal graph classification, which not only identifies graph samples which do not belong to known classes, but also classifies the remaining samples into their respective classes. This problem is challenging due to a severe lack of labels and potential class shifts. In this paper, we propose a novel graph neural network framework named UGNN, which makes the best of unlabeled data from the subgraph perspective. To tackle class shifts, we estimate the certainty of unlabeled graphs using multiple subgraphs, which facilities the discovery of
    
[^3]: AdANNS: 一种自适应语义搜索框架

    AdANNS: A Framework for Adaptive Semantic Search. (arXiv:2305.19435v1 [cs.LG])

    [http://arxiv.org/abs/2305.19435](http://arxiv.org/abs/2305.19435)

    AdANNS是一种自适应语义搜索框架，利用不同容量的自适应表示形式可以获得更好的精度-计算折衷权衡，相似度计算越接近的数据点将使用更低容量的表示形式进行计算，演示了最先进的精度-计算折衷权衡。

    

    网络规模的搜索系统学习一个编码器来嵌入一个给定的查询，然后将其连接到近似最近邻搜索(ANNS)管道中来检索相似的数据点。为了准确地捕捉尾部查询和数据点，学习到的表示通常是刚性的、高维的向量，通常在整个ANNS管道中一成不变，并且可能导致计算上昂贵的检索。本文认为，与其使用刚性的表示形式，ANNS的不同阶段可以利用不同容量的自适应表示形式以获得显著的精度-计算折衷权衡，即可以进行更加近似计算的ANNS阶段应该使用相同数据点的低容量表示。为此，我们引入了AdANNS，一种新颖的ANNS设计框架，明确利用Matryoshka表示的灵活性。我们使用基于AdANNS的新型关键ANNS构建演示了最先进的精度-计算折衷权衡。

    Web-scale search systems learn an encoder to embed a given query which is then hooked into an approximate nearest neighbor search (ANNS) pipeline to retrieve similar data points. To accurately capture tail queries and data points, learned representations typically are rigid, high-dimensional vectors that are generally used as-is in the entire ANNS pipeline and can lead to computationally expensive retrieval. In this paper, we argue that instead of rigid representations, different stages of ANNS can leverage adaptive representations of varying capacities to achieve significantly better accuracy-compute trade-offs, i.e., stages of ANNS that can get away with more approximate computation should use a lower-capacity representation of the same data point. To this end, we introduce AdANNS, a novel ANNS design framework that explicitly leverages the flexibility of Matryoshka Representations. We demonstrate state-of-the-art accuracy-compute trade-offs using novel AdANNS-based key ANNS building
    
[^4]: DuoSearch：一种用于保加利亚历史文件的新型搜索引擎

    DuoSearch: A Novel Search Engine for Bulgarian Historical Documents. (arXiv:2305.19392v1 [cs.IR])

    [http://arxiv.org/abs/2305.19392](http://arxiv.org/abs/2305.19392)

    本文提出了一种新的历史文件搜索引擎DuoSearch，该引擎使用神经网络机器学习方法解决数字化历史文献搜索中的正字法变体和字符识别错误问题，并成功应用于保加利亚中期到20世纪中期的历史报纸集合。

    

    数字化历史文献的搜索存在正字法变体和字符识别错误两大问题。本文提出了一种新的历史文件搜索引擎——DuoSearch，它使用ElasticSearch和基于深度神经网络的机器学习方法，提供了解决这个问题的方案。该系统在保加利亚中期到20世纪中期的历史报纸集合上进行了测试，为最终用户提供了交互式和直观的界面，允许他们以现代保加利亚语输入搜索词并跨历史拼写进行搜索。这是首个易于使用数字化保加利亚历史文献的解决方案。

    Search in collections of digitised historical documents is hindered by a two-prong problem, orthographic variety and optical character recognition (OCR) mistakes. We present a new search engine for historical documents, DuoSearch, which uses ElasticSearch and machine learning methods based on deep neural networks to offer a solution to this problem. It was tested on a collection of historical newspapers in Bulgarian from the mid-19th to the mid-20th century. The system provides an interactive and intuitive interface for the end-users allowing them to enter search terms in modern Bulgarian and search across historical spellings. This is the first solution facilitating the use of digitised historical documents in Bulgarian.
    
[^5]: 缓解测试时间偏差，实现公平的图像检索

    Mitigating Test-Time Bias for Fair Image Retrieval. (arXiv:2305.19329v1 [cs.CV])

    [http://arxiv.org/abs/2305.19329](http://arxiv.org/abs/2305.19329)

    本文提出了后置偏差缓解（PBM）技术，解决了如何在中性文本查询的情况下实现公平的图像检索。该方法在实际数据集中实现了最低的偏差。

    

    本文解决了如何在中性文本查询的情况下（没有明确的性别或种族内涵）生成公平和无偏见的图像检索结果，同时保持底层视觉语言（VL）模型的效用（性能）的挑战。先前的方法旨在将图像和文本查询的学习表示与性别和种族特征分离。然而，我们发现这些方法不能减轻测试集中的偏差，从而实现所需的平等表示结果。出于这个动机，我们提出了一个简单的技术，后置偏差缓解（PBM），来后处理预训练视觉语言模型的输出。我们在实际图像搜索数据集Occupation 1和2，以及两个大规模的图像文本数据集MS-COCO和Flickr30k上评估了我们的算法。与各种现有的偏差缓解方法相比，我们的方法在基于文本的图像检索结果中实现了最低的偏差。

    We address the challenge of generating fair and unbiased image retrieval results given neutral textual queries (with no explicit gender or race connotations), while maintaining the utility (performance) of the underlying vision-language (VL) model. Previous methods aim to disentangle learned representations of images and text queries from gender and racial characteristics. However, we show these are inadequate at alleviating bias for the desired equal representation result, as there usually exists test-time bias in the target retrieval set. So motivated, we introduce a straightforward technique, Post-hoc Bias Mitigation (PBM), that post-processes the outputs from the pre-trained vision-language model. We evaluate our algorithm on real-world image search datasets, Occupation 1 and 2, as well as two large-scale image-text datasets, MS-COCO and Flickr30k. Our approach achieves the lowest bias, compared with various existing bias-mitigation methods, in text-based image retrieval result whi
    
[^6]: 统一嵌入：面向 Web 规模 ML 系统的经过验证的特征表示

    Unified Embedding: Battle-Tested Feature Representations for Web-Scale ML Systems. (arXiv:2305.12102v1 [cs.LG])

    [http://arxiv.org/abs/2305.12102](http://arxiv.org/abs/2305.12102)

    本文介绍了一种名为“特征复用”的框架，它使用单一的表示空间 能够高效有效地学习高质量的特征嵌入，同时区分不同的分类特征。通过在多个公共数据集和新数据集“Web-Available Image Search (WAIS)”上的测试，我们展示了这种方法的优于现有技术的表现。

    

    高效、有效地学习高质量的特征嵌入对于 Web 规模的机器学习系统的性能至关重要。标准方法是将每个特征值表示为一个 d 维嵌入，引入数百亿个参数，而这些特征的基数非常高。这个瓶颈导致了备选嵌入算法的重大进展。本文介绍了一个简单但非常有效的框架，即“特征复用”，在许多不同的分类特征之间使用一个单一的表示空间。我们的理论和实证分析表明，复用的嵌入可以分解为每个组成特征的组件，使得模型可以区分特征。我们展示了复用的嵌入在几个公共数据集上优于现有技术。此外，我们引入了一个名为“Web-Available Image Search (WAIS)”的新数据集，以严格评估 Web 规模下的新嵌入算法。我们邀请社区通过提出可以准确、高效地将数百万张图像嵌入和分类到成千上万个类别的新模型来贡献 WAIS 挑战。

    Learning high-quality feature embeddings efficiently and effectively is critical for the performance of web-scale machine learning systems. A typical model ingests hundreds of features with vocabularies on the order of millions to billions of tokens. The standard approach is to represent each feature value as a d-dimensional embedding, introducing hundreds of billions of parameters for extremely high-cardinality features. This bottleneck has led to substantial progress in alternative embedding algorithms. Many of these methods, however, make the assumption that each feature uses an independent embedding table. This work introduces a simple yet highly effective framework, Feature Multiplexing, where one single representation space is used across many different categorical features. Our theoretical and empirical analysis reveals that multiplexed embeddings can be decomposed into components from each constituent feature, allowing models to distinguish between features. We show that multip
    
[^7]: EvalRS 2023. 面向实际应用的全面推荐系统评估

    EvalRS 2023. Well-Rounded Recommender Systems For Real-World Deployments. (arXiv:2304.07145v1 [cs.IR])

    [http://arxiv.org/abs/2304.07145](http://arxiv.org/abs/2304.07145)

    EvalRS 2023旨在探讨推荐系统的全面评估，关注其在现实场景下的实际影响。过去只有准确度的测量方法可能无法全面评估其性能，公平性、偏见、有用性和信息量等方面也应该被关注。本次研讨会是去年CIKM研讨会的继承和发展，带有实际操作性和互动性。

    

    EvalRS旨在汇聚产业和学术界的从业者，促进对推荐系统的全面评估的讨论，并重点关注在各种部署场景下的实际影响。推荐系统通常只通过准确性指标进行评估，这些指标无法完全描述其泛化能力并忽视了重要的方面，如公平性、偏见、有用性、信息量等。本次研讨会在CIKM去年研讨会的成功基础上进一步扩大范围，并采取互动形式。

    EvalRS aims to bring together practitioners from industry and academia to foster a debate on rounded evaluation of recommender systems, with a focus on real-world impact across a multitude of deployment scenarios. Recommender systems are often evaluated only through accuracy metrics, which fall short of fully characterizing their generalization capabilities and miss important aspects, such as fairness, bias, usefulness, informativeness. This workshop builds on the success of last year's workshop at CIKM, but with a broader scope and an interactive format.
    
[^8]: RARR: 使用语言模型研究和修正其输出结果中的不确定信息

    RARR: Researching and Revising What Language Models Say, Using Language Models. (arXiv:2210.08726v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2210.08726](http://arxiv.org/abs/2210.08726)

    RARR是一个可以对不确定信息进行研究和修订的系统，它可以自动找到文本生成模型输出的归因并修正不支持的内容。

    

    现在的语言模型在诸如少样本学习、问答、推理和对话等许多任务上表现出色。然而，它们有时会生成无支持或误导性的内容。由于大多数语言模型没有任何内置的归因外部证据的机制，用户很难确定它们的输出是否可靠。为了在保留最新一代模型的所有强大优势的同时实现归因，我们提出了 RARR (使用研究和修订进行改进归因)系统，它 1) 自动找到任何文本生成模型输出的归因并 2) 在尽可能保留原始输出的同时，修正不支持的内容。当应用于几个最先进的语言模型在各种输出任务上的结果时，我们发现RARR在显著提高归因率的同时，比以前探索的编辑模型更能保留原始输入。

    Language models (LMs) now excel at many tasks such as few-shot learning, question answering, reasoning, and dialog. However, they sometimes generate unsupported or misleading content. A user cannot easily determine whether their outputs are trustworthy or not, because most LMs do not have any built-in mechanism for attribution to external evidence. To enable attribution while still preserving all the powerful advantages of recent generation models, we propose RARR (Retrofit Attribution using Research and Revision), a system that 1) automatically finds attribution for the output of any text generation model and 2) post-edits the output to fix unsupported content while preserving the original output as much as possible. When applied to the output of several state-of-the-art LMs on a diverse set of generation tasks, we find that RARR significantly improves attribution while otherwise preserving the original input to a much greater degree than previously explored edit models. Furthermore, 
    
[^9]: 变分开放领域问答

    Variational Open-Domain Question Answering. (arXiv:2210.06345v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2210.06345](http://arxiv.org/abs/2210.06345)

    本文介绍了变分开放领域（VOD）框架，提出了一种新的自归一化的Rényi变分界的估计方法，可用于训练具有检索增强功能的模型，例如阅读器-检索器BERT-sized模型，并实现了在多项选择医学考试问题上的优异表现。

    

    检索增强模型在自然语言处理任务中已被证明是有效的，但是对它们进行变分推断的优化研究仍然不足。我们引入了变分开放领域（VOD）框架，用于检索增强模型的端到端训练和评估，重点放在开放领域问答和语言建模方面。VOD目标是一种自归一化的Rényi变分界的估计，近似于任务边缘似然，并在一个辅助采样分布（缓存的检索器和/或近似后验）中进行样本抽取评估。它仍然是可行的，即使是在对大量语料库定义的检索器分布下。我们通过训练针对多项选择医学考试问题的阅读器-检索器BERT-sized模型，展示了VOD的多功能性。在MedMCQA数据集上，我们超过了领域微调的Med-PaLM 5.3％，尽管使用的参数少了2500倍。我们的检索增强BioLinkBERT模型得分为62.9％。

    Retrieval-augmented models have proven to be effective in natural language processing tasks, yet there remains a lack of research on their optimization using variational inference. We introduce the Variational Open-Domain (VOD) framework for end-to-end training and evaluation of retrieval-augmented models, focusing on open-domain question answering and language modelling. The VOD objective, a self-normalized estimate of the R\'enyi variational bound, approximates the task marginal likelihood and is evaluated under samples drawn from an auxiliary sampling distribution (cached retriever and/or approximate posterior). It remains tractable, even for retriever distributions defined on large corpora. We demonstrate VOD's versatility by training reader-retriever BERT-sized models on multiple-choice medical exam questions. On the MedMCQA dataset, we outperform the domain-tuned Med-PaLM by +5.3% despite using 2.500$\times$ fewer parameters. Our retrieval-augmented BioLinkBERT model scored 62.9%
    
[^10]: CONE：用于长视频时间定位的高效粗-细对齐框架

    CONE: An Efficient COarse-to-fiNE Alignment Framework for Long Video Temporal Grounding. (arXiv:2209.10918v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2209.10918](http://arxiv.org/abs/2209.10918)

    本文提出了CONE，一个高效的粗-细对齐框架，可用于长视频时间定位。CONE通过基于查询的窗口选择策略和对比学习机制提升了多模态对齐，并在两个大规模长视频时间定位基准测试中取得最先进结果。

    

    本文解决了一个新兴且具有挑战性的问题——长视频时间定位（VTG），即定位与自然语言查询相关的视频片段。相比于短视频，长视频同样非常受欢迎，但是探索较少，这带来了多个挑战，例如更高的推理计算成本和弱的多模态对齐。为了解决这些挑战，我们提出了CONE，一个高效的粗-细对齐框架。CONE是一个插拔式的框架，可在现有的VTG模型上处理长视频，通过滑动窗口机制。具体来说，CONE（1）引入了基于查询的窗口选择策略以加快推理速度，（2）提议了通过新增对比学习来增强长视频的多模态对齐的粗细机制。对两个大规模长VTG基准测试进行的大量实验均表明，CONE在性能上都有很大提升（例如在MAD上从3.13％到6.87％），并且具有最先进的结果。分析也证明了CONE模型的有效性和可解释性。

    This paper tackles an emerging and challenging problem of long video temporal grounding~(VTG) that localizes video moments related to a natural language (NL) query. Compared with short videos, long videos are also highly demanded but less explored, which brings new challenges in higher inference computation cost and weaker multi-modal alignment. To address these challenges, we propose CONE, an efficient COarse-to-fiNE alignment framework. CONE is a plug-and-play framework on top of existing VTG models to handle long videos through a sliding window mechanism. Specifically, CONE (1) introduces a query-guided window selection strategy to speed up inference, and (2) proposes a coarse-to-fine mechanism via a novel incorporation of contrastive learning to enhance multi-modal alignment for long videos. Extensive experiments on two large-scale long VTG benchmarks consistently show both substantial performance gains (e.g., from 3.13% to 6.87% on MAD) and state-of-the-art results. Analyses also 
    
[^11]: 理解基于会话的推荐系统中的多样性

    Understanding Diversity in Session-Based Recommendation. (arXiv:2208.13453v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2208.13453](http://arxiv.org/abs/2208.13453)

    这篇文章探讨了基于会话的推荐系统在准确度和多样性方面的表现，并提供了设计多样化SBRS的指导。

    

    当前的基于会话的推荐系统（SBRSs）主要关注于最大化推荐准确度，而很少有研究专门致力于在准确度之外提高多样性。同时，不清楚以准确度为导向的SBRS在多样性方面的表现如何。此外，在文献中越来越多地质疑“准确度”与“多样性”之间的“权衡”关系。针对上述问题，我们进行了一项全面的研究，特别是考察具有代表性的基于会话的推荐系统在准确度和多样性方面的推荐性能，努力更好地理解SBRS的多样性相关问题，并提供设计多样化SBRS的指导。特别地，为了进行公平和全面的比较，我们精心选择了最先进的非神经，深度神经和多样化SBRS，通过覆盖更多具有适当实验设置的场景，例如代表性数据集，评估指标和超参数优化技术。

    Current session-based recommender systems (SBRSs) mainly focus on maximizing recommendation accuracy, while few studies have been devoted to improve diversity beyond accuracy. Meanwhile, it is unclear how the accuracy-oriented SBRSs perform in terms of diversity. Besides, the asserted "trade-off" relationship between accuracy and diversity has been increasingly questioned in the literature. Towards the aforementioned issues, we conduct a holistic study to particularly examine the recommendation performance of representative SBRSs w.r.t. both accuracy and diversity, striving for better understanding the diversity-related issues for SBRSs and providing guidance on designing diversified SBRSs. Particularly, for a fair and thorough comparison, we deliberately select state-of-the-art non-neural, deep neural, and diversified SBRSs, by covering more scenarios with appropriate experimental setups, e.g., representative datasets, evaluation metrics, and hyper-parameter optimization technique. Ou
    

