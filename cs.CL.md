# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Beyond One-Size-Fits-All: Multi-Domain, Multi-Task Framework for Embedding Model Selection](https://arxiv.org/abs/2404.00458) | 开发了一个多领域、多任务框架，帮助选择最有效的自然语言处理嵌入模型。 |
| [^2] | [Evaluating Named Entity Recognition: Comparative Analysis of Mono- and Multilingual Transformer Models on Brazilian Corporate Earnings Call Transcriptions](https://arxiv.org/abs/2403.12212) | 本研究通过引入新方法，将标记分类任务重新构建为文本生成问题，评估了在巴西银行财报电话转录中使用的单语和多语言Transformer模型的性能。 |
| [^3] | [Large Language Multimodal Models for 5-Year Chronic Disease Cohort Prediction Using EHR Data](https://arxiv.org/abs/2403.04785) | 本研究提出了一种大型语言多模型（LLMMs）框架，结合临床笔记和实验室检验结果的多模态数据，用于预测慢性疾病风险。 |
| [^4] | [Advancing Biomedical Text Mining with Community Challenges](https://arxiv.org/abs/2403.04261) | 社区挑战评估竞赛在促进生物医学文本挖掘研究中的技术创新和跨学科合作方面起着重要作用。 |
| [^5] | [SimuCourt: Building Judicial Decision-Making Agents with Real-world Judgement Documents](https://arxiv.org/abs/2403.02959) | 提出了SimuCourt司法基准，包括真实世界的司法文件，并引入了司法决策任务和多代理框架，评估了代理的司法分析和决策能力 |
| [^6] | [Exploring Group and Symmetry Principles in Large Language Models](https://arxiv.org/abs/2402.06120) | 本文提出了一个基于群组和对称性原理的框架，以评估大型语言模型的推理能力。通过研究四个群组属性，发现这些模型在保持群组属性方面表现不佳。 |
| [^7] | [Language models align with human judgments on key grammatical constructions](https://arxiv.org/abs/2402.01676) | 本研究通过对比评估发现，大型语言模型（LLMs）在俘获人类行为方面的表现非常出色，不仅整体准确率高，而且能够捕捉到人类语言判断中的细微差异。 |
| [^8] | [Loop Copilot: Conducting AI Ensembles for Music Generation and Iterative Editing.](http://arxiv.org/abs/2310.12404) | Loop Copilot是一种新型的AI音乐合奏系统，能够通过交互式多轮对话界面生成和迭代改进音乐，通过选择适当的AI模型执行任务，并在一个集中的表中保持关键属性以确保音乐的连贯性。 |
| [^9] | [Measuring Stereotypes using Entity-Centric Data.](http://arxiv.org/abs/2305.09548) | 本文提出并评估了三种新的以实体为中心的方法，展示了这些模型在预测人们如何将身份标签应用于自己和他人以及量化突出的社会维度（如性别）的刻板印象方面优于现有方法。 |
| [^10] | [Does CLIP Bind Concepts? Probing Compositionality in Large Image Models.](http://arxiv.org/abs/2212.10537) | 本文分析了大型神经网络模型CLIP的组合性能力以及以结构敏感的方式捆绑变量的能力，发现其能够在单一对象的情况下组合概念，但在需要概念捆绑的情况下性能显著下降。 |
| [^11] | [MelHuBERT: A simplified HuBERT on Mel spectrograms.](http://arxiv.org/abs/2211.09944) | MelHuBERT是基于Mel频谱图的简化版HuBERT模型，通过改进损失函数、输入表示和多阶段训练，在语音识别方面取得了有利表现，节省了31.2%的预训练时间和33.5%的计算资源。 |

# 详细

[^1]: 超越一刀切：多领域、多任务框架用于嵌入模型选择

    Beyond One-Size-Fits-All: Multi-Domain, Multi-Task Framework for Embedding Model Selection

    [https://arxiv.org/abs/2404.00458](https://arxiv.org/abs/2404.00458)

    开发了一个多领域、多任务框架，帮助选择最有效的自然语言处理嵌入模型。

    

    这篇立场论文提出了一种系统方法，旨在开发一个框架，帮助选择适用于自然语言处理（NLP）任务的最有效的嵌入模型，解决了专有和开源编码器模型大量增加所带来的挑战。

    arXiv:2404.00458v1 Announce Type: new  Abstract: This position paper proposes a systematic approach towards developing a framework to help select the most effective embedding models for natural language processing (NLP) tasks, addressing the challenge posed by the proliferation of both proprietary and open-source encoder models.
    
[^2]: 评估命名实体识别：比较分析巴西公司财报电话转录上的单语和多语言Transformer模型

    Evaluating Named Entity Recognition: Comparative Analysis of Mono- and Multilingual Transformer Models on Brazilian Corporate Earnings Call Transcriptions

    [https://arxiv.org/abs/2403.12212](https://arxiv.org/abs/2403.12212)

    本研究通过引入新方法，将标记分类任务重新构建为文本生成问题，评估了在巴西银行财报电话转录中使用的单语和多语言Transformer模型的性能。

    

    命名实体识别（NER）是一种从文本文档中提取信息的自然语言处理技术。然而，现有关于NER的大部分研究都集中在英语文档上，导致缺乏专门针对葡萄牙语财务领域的数据集。本研究解决了金融领域内NER需求，并侧重于从巴西银行财报电话转录中提取的葡萄牙语文本。通过整理包括384个转录的综合数据集，并利用弱监督技术进行注释，我们评估了在葡萄牙语（BERTimbau和PTT5）训练的单语模型以及多语言模型（mBERT和mT5）的性能。值得注意的是，我们引入了一种新方法，将标记分类任务重新构建为文本生成问题，从而实现T5模型的微调和评估。在模型微调之后，

    arXiv:2403.12212v1 Announce Type: cross  Abstract: Named Entity Recognition (NER) is a Natural Language Processing technique for extracting information from textual documents. However, much of the existing research on NER has been centered around English-language documents, leaving a gap in the availability of datasets tailored to the financial domain in Portuguese. This study addresses the need for NER within the financial domain, focusing on Portuguese-language texts extracted from earnings call transcriptions of Brazilian banks. By curating a comprehensive dataset comprising 384 transcriptions and leveraging weak supervision techniques for annotation, we evaluate the performance of monolingual models trained on Portuguese (BERTimbau and PTT5) and multilingual models (mBERT and mT5). Notably, we introduce a novel approach that reframes the token classification task as a text generation problem, enabling fine-tuning and evaluation of T5 models. Following the fine-tuning of the models,
    
[^3]: 使用电子健康记录数据预测5年慢性疾病队列的大型语言多模型

    Large Language Multimodal Models for 5-Year Chronic Disease Cohort Prediction Using EHR Data

    [https://arxiv.org/abs/2403.04785](https://arxiv.org/abs/2403.04785)

    本研究提出了一种大型语言多模型（LLMMs）框架，结合临床笔记和实验室检验结果的多模态数据，用于预测慢性疾病风险。

    

    慢性疾病如糖尿病是全球发病率和死亡率的主要原因。本研究从台湾医院数据库收集了五年的电子健康记录数据，包括1,420,596份临床笔记、387,392份实验室检验结果以及超过1,505种实验室检验项目，重点研究了用于研究预训练大型语言模型的方法。我们提出了一种新颖的大型语言多模型（LLMMs）框架，将临床笔记和实验室检验结果的多模态数据相结合，用于预测慢性疾病风险。我们的方法结合了文本嵌入编码器和多头注意力层来学习实验室检验数值，利用深度神经网络（DNN）模块进行预测。

    arXiv:2403.04785v1 Announce Type: cross  Abstract: Chronic diseases such as diabetes are the leading causes of morbidity and mortality worldwide. Numerous research studies have been attempted with various deep learning models in diagnosis. However, most previous studies had certain limitations, including using publicly available datasets (e.g. MIMIC), and imbalanced data. In this study, we collected five-year electronic health records (EHRs) from the Taiwan hospital database, including 1,420,596 clinical notes, 387,392 laboratory test results, and more than 1,505 laboratory test items, focusing on research pre-training large language models. We proposed a novel Large Language Multimodal Models (LLMMs) framework incorporating multimodal data from clinical notes and laboratory test results for the prediction of chronic disease risk. Our method combined a text embedding encoder and multi-head attention layer to learn laboratory test values, utilizing a deep neural network (DNN) module to 
    
[^4]: 通过社区挑战推动生物医学文本挖掘的发展

    Advancing Biomedical Text Mining with Community Challenges

    [https://arxiv.org/abs/2403.04261](https://arxiv.org/abs/2403.04261)

    社区挑战评估竞赛在促进生物医学文本挖掘研究中的技术创新和跨学科合作方面起着重要作用。

    

    生物医学研究领域积累了大量来自科学文献、电子病历、临床试验报告和社交媒体等各方面的文本数据，然而手动处理和分析这些庞大且复杂的资源是耗时且低效的。为了解决这一挑战，生物医学文本挖掘，也称为生物医学自然语言处理，备受关注。社区挑战评估竞赛在促进生物医学文本挖掘研究中的技术创新和跨学科合作方面发挥了重要作用。这些挑战为研究人员提供了开发生物医学研究中数据挖掘和信息处理的最新解决方案的平台。在本文中，我们回顾了与中文生物医学文本挖掘有关的最新社区挑战的进展。

    arXiv:2403.04261v1 Announce Type: new  Abstract: The field of biomedical research has witnessed a significant increase in the accumulation of vast amounts of textual data from various sources such as scientific literatures, electronic health records, clinical trial reports, and social media. However, manually processing and analyzing these extensive and complex resources is time-consuming and inefficient. To address this challenge, biomedical text mining, also known as biomedical natural language processing, has garnered great attention. Community challenge evaluation competitions have played an important role in promoting technology innovation and interdisciplinary collaboration in biomedical text mining research. These challenges provide platforms for researchers to develop state-of-the-art solutions for data mining and information processing in biomedical research. In this article, we review the recent advances in community challenges specific to Chinese biomedical text mining. Firs
    
[^5]: SimuCourt: 利用真实司法判决文件构建司法决策代理

    SimuCourt: Building Judicial Decision-Making Agents with Real-world Judgement Documents

    [https://arxiv.org/abs/2403.02959](https://arxiv.org/abs/2403.02959)

    提出了SimuCourt司法基准，包括真实世界的司法文件，并引入了司法决策任务和多代理框架，评估了代理的司法分析和决策能力

    

    随着深度学习、自然语言处理技术的发展，有效提高了传统司法行业各个方面的效率。然而，目前大多数工作主要集中在个别司法阶段，忽视了跨阶段的协作。随着由大型语言模型提供支持的自主代理在现实环境中变得越来越智能，并能做出复杂决策，为司法智能提供了新的见解。本文介绍了SimuCourt，一个司法基准，包括来自真实世界的420份判决文件，涵盖了三种最常见类型的司法案例，以及一个新颖任务司法决策，用于评估代理的司法分析和决策能力。为了支持这一任务，我们构建了一个大规模司法知识库，JudicialKB，其中包含多种法律知识。我们提出了一种新颖的多代理框架，AgentsCourt

    arXiv:2403.02959v1 Announce Type: cross  Abstract: With the development of deep learning, natural language processing technology has effectively improved the efficiency of various aspects of the traditional judicial industry. However, most current efforts focus solely on individual judicial stage, overlooking cross-stage collaboration. As the autonomous agents powered by large language models are becoming increasingly smart and able to make complex decisions in real-world settings, offering new insights for judicial intelligence. In this paper, (1) we introduce SimuCourt, a judicial benchmark that encompasses 420 judgment documents from real-world, spanning the three most common types of judicial cases, and a novel task Judicial Decision-Making to evaluate the judicial analysis and decision-making power of agents. To support this task, we construct a large-scale judicial knowledge base, JudicialKB, with multiple legal knowledge. (2) we propose a novel multi-agent framework, AgentsCourt
    
[^6]: 探索大型语言模型中的群组和对称性原理

    Exploring Group and Symmetry Principles in Large Language Models

    [https://arxiv.org/abs/2402.06120](https://arxiv.org/abs/2402.06120)

    本文提出了一个基于群组和对称性原理的框架，以评估大型语言模型的推理能力。通过研究四个群组属性，发现这些模型在保持群组属性方面表现不佳。

    

    大型语言模型（LLM）在广泛的应用中展示了令人瞩目的性能，然而评估它们的推理能力仍然是一个重大挑战。在本文中，我们引入了一个以群组和对称性原理为基础的框架，这些原理在物理学和数学等领域发挥着关键作用，并提供了另一种评估它们能力的方式。虽然提出的框架是通用的，为了展示使用这些属性的好处，我们关注算术推理，并研究这些模型在四个群组属性（封闭性、恒等性、逆性和结合性）上的性能。我们的发现表明，在本研究中研究的LLM在不同的测试方案中难以保持群组属性。在封闭性测试中，我们观察到对特定输出的偏见，并在特定的序列长度后从100％的性能迅速下降到0％。它们在恒等性测试中表现不佳，代表了相加得到原数的属性。

    Large Language Models (LLMs) have demonstrated impressive performance across a wide range of applications; however, assessing their reasoning capabilities remains a significant challenge. In this paper, we introduce a framework grounded in group and symmetry principles, which have played a crucial role in fields such as physics and mathematics, and offer another way to evaluate their capabilities. While the proposed framework is general, to showcase the benefits of employing these properties, we focus on arithmetic reasoning and investigate the performance of these models on four group properties: closure, identity, inverse, and associativity. Our findings reveal that LLMs studied in this work struggle to preserve group properties across different test regimes. In the closure test, we observe biases towards specific outputs and an abrupt degradation in their performance from 100% to 0% after a specific sequence length. They also perform poorly in the identity test, which represents add
    
[^7]: 语言模型与人类在关键语法结构上的判断一致性

    Language models align with human judgments on key grammatical constructions

    [https://arxiv.org/abs/2402.01676](https://arxiv.org/abs/2402.01676)

    本研究通过对比评估发现，大型语言模型（LLMs）在俘获人类行为方面的表现非常出色，不仅整体准确率高，而且能够捕捉到人类语言判断中的细微差异。

    

    大型语言模型（LLMs）是否具有类似人类的语言普遍性？Dentella等人（2023年；“DGL”）使用多个LLMs提示语法正确性问题，以获取80个英语句子的语法句子判断，得出LLMs存在“是”偏向和“不能区分语法和非语法句子”的结论。我们采用了既定的实践方法重新评估LLM的性能，并发现DGL的数据实际上证明了LLM如何准确捕捉人类行为。模型不仅整体上实现了高准确率，还捕捉到了人类语言判断的细微变化。

    Do Large Language Models (LLMs) make human-like linguistic generalizations? Dentella et al. (2023; "DGL") prompt several LLMs ("Is the following sentence grammatically correct in English?") to elicit grammaticality judgments of 80 English sentences, concluding that LLMs demonstrate a "yes-response bias" and a "failure to distinguish grammatical from ungrammatical sentences". We re-evaluate LLM performance using well-established practices and find that DGL's data in fact provide evidence for just how well LLMs capture human behaviors. Models not only achieve high accuracy overall, but also capture fine-grained variation in human linguistic judgments.
    
[^8]: Loop Copilot: 用于音乐生成和迭代编辑的AI合奏系统

    Loop Copilot: Conducting AI Ensembles for Music Generation and Iterative Editing. (arXiv:2310.12404v1 [cs.SD])

    [http://arxiv.org/abs/2310.12404](http://arxiv.org/abs/2310.12404)

    Loop Copilot是一种新型的AI音乐合奏系统，能够通过交互式多轮对话界面生成和迭代改进音乐，通过选择适当的AI模型执行任务，并在一个集中的表中保持关键属性以确保音乐的连贯性。

    

    创建音乐是一个迭代过程，每个阶段都需要不同的方法。然而，现有的AI音乐系统在组织多个子系统以满足不同需求方面存在不足。为了解决这个问题，我们引入了Loop Copilot，这是一个能够通过交互式、多轮对话界面生成和迭代改进音乐的新型系统。该系统使用一种大型语言模型来解释用户意图，并选择适当的AI模型进行任务执行。每个后端模型都专门针对特定任务，并将它们的输出聚合起来以满足用户的要求。为了确保音乐的连贯性，关键属性被保留在一个集中的表中。我们通过半结构化的访谈和问卷调查评估了所提出的系统的有效性，突出了它在促进音乐创作方面的实用性，以及它在更广泛应用中的潜力。

    Creating music is iterative, requiring varied methods at each stage. However, existing AI music systems fall short in orchestrating multiple subsystems for diverse needs. To address this gap, we introduce Loop Copilot, a novel system that enables users to generate and iteratively refine music through an interactive, multi-round dialogue interface. The system uses a large language model to interpret user intentions and select appropriate AI models for task execution. Each backend model is specialized for a specific task, and their outputs are aggregated to meet the user's requirements. To ensure musical coherence, essential attributes are maintained in a centralized table. We evaluate the effectiveness of the proposed system through semi-structured interviews and questionnaires, highlighting its utility not only in facilitating music creation but also its potential for broader applications.
    
[^9]: 使用以实体为中心的数据来衡量刻板印象

    Measuring Stereotypes using Entity-Centric Data. (arXiv:2305.09548v1 [cs.CL])

    [http://arxiv.org/abs/2305.09548](http://arxiv.org/abs/2305.09548)

    本文提出并评估了三种新的以实体为中心的方法，展示了这些模型在预测人们如何将身份标签应用于自己和他人以及量化突出的社会维度（如性别）的刻板印象方面优于现有方法。

    

    刻板印象影响我们如何展示自己和他人，从而影响我们的行为。因此，衡量刻板印象非常重要。最近的研究使用分布语义模型（DSM）（如BERT）中嵌入的投影来进行这些测量。然而，DSMs捕捉到的认知联想不一定与刻板印象的人际性质相关。在这里，我们提出并评估了三种新的以实体为中心的方法，从Twitter和Wikipedia传记中学习刻板印象。通过利用多个短语应用于同一个人的事实来训练模型，扩大了学习联想的人本身中心性。我们证明了这些模型在预测人们如何将身份标签应用于自己和他人以及量化突出的社会维度（如性别）的刻板印象方面优于现有方法。通过一个案例研究，我们还展示了这些模型对未来计算社会科学问题的实用性。

    Stereotypes inform how we present ourselves and others, and in turn how we behave. They are thus important to measure. Recent work has used projections of embeddings from Distributional Semantic Models (DSMs), such as BERT, to perform these measurements. However, DSMs capture cognitive associations that are not necessarily relevant to the interpersonal nature of stereotyping. Here, we propose and evaluate three novel, entity-centric methods for learning stereotypes from Twitter and Wikipedia biographies. Models are trained by leveraging the fact that multiple phrases are applied to the same person, magnifying the person-centric nature of the learned associations. We show that these models outperform existing approaches to stereotype measurement with respect to 1) predicting which identities people apply to themselves and others, and 2) quantifying stereotypes on salient social dimensions (e.g. gender). Via a case study, we also show the utility of these models for future questions in c
    
[^10]: CLIP是否捆绑概念？探索大型图像模型的组合性。

    Does CLIP Bind Concepts? Probing Compositionality in Large Image Models. (arXiv:2212.10537v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2212.10537](http://arxiv.org/abs/2212.10537)

    本文分析了大型神经网络模型CLIP的组合性能力以及以结构敏感的方式捆绑变量的能力，发现其能够在单一对象的情况下组合概念，但在需要概念捆绑的情况下性能显著下降。

    

    近年来，结合文本和图像的大型神经网络模型取得了令人瞩目的进展。然而，这些模型在多大程度上编码了它们操作的概念的组成性表示，如通过对“红色立方体”进行推理以正确识别“红色”和“立方体”这些成分，这仍然是一个开放性问题。本文关注一个大型预训练的视觉和语言模型（CLIP）编码组合概念的能力以及以结构敏感的方式捆绑变量的能力（例如区分“立方体在球体后面”和“球体在立方体后面”）。为了检查CLIP的性能，我们比较了许多来自组合分布语义模型（CDSMs）的架构，这是一种试图在嵌入空间中实现传统组合语言结构的研究方向。我们发现CLIP能够在单一对象的情况下组合概念，但在需要概念捆绑的情况下性能显著下降。我们的分析凸显了评估大型模型组合性的重要性，并为未来研究提出了方向。

    Large-scale neural network models combining text and images have made incredible progress in recent years. However, it remains an open question to what extent such models encode compositional representations of the concepts over which they operate, such as correctly identifying ''red cube'' by reasoning over the constituents ''red'' and ''cube''. In this work, we focus on the ability of a large pretrained vision and language model (CLIP) to encode compositional concepts and to bind variables in a structure-sensitive way (e.g., differentiating ''cube behind sphere'' from ''sphere behind cube''). In order to inspect the performance of CLIP, we compare several architectures from research on compositional distributional semantics models (CDSMs), a line of research that attempts to implement traditional compositional linguistic structures within embedding spaces. We find that CLIP can compose concepts in a single-object setting, but in situations where concept binding is needed, performance
    
[^11]: MelHuBERT: 一种基于Mel频谱图的简化HuBERT模型

    MelHuBERT: A simplified HuBERT on Mel spectrograms. (arXiv:2211.09944v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.09944](http://arxiv.org/abs/2211.09944)

    MelHuBERT是基于Mel频谱图的简化版HuBERT模型，通过改进损失函数、输入表示和多阶段训练，在语音识别方面取得了有利表现，节省了31.2%的预训练时间和33.5%的计算资源。

    

    自监督模型在学习语音表示方面取得了巨大的成功，可以推广到各种下游任务。然而，大多数自监督模型需要大量的计算资源和多个GPU来进行训练，从而严重限制了自监督学习的发展。为了减少训练的计算量，我们重新审视了HuBERT的训练方法，这是一个非常成功的自监督模型。我们改进并简化了几个关键组成部分，包括损失函数、输入表示和多阶段训练。我们的模型MelHuBERT在音素识别、说话人识别和自动语音识别方面均能取得较好的性能，同时节省了31.2%的预训练时间，或等效地每秒语音节省了33.5%的MACs。代码和预训练模型可在https://github.com/nervjack2/MelHuBERT中获得。

    Self-supervised models have had great success in learning speech representations that can generalize to various downstream tasks. However, most self-supervised models require a large amount of compute and multiple GPUs to train, significantly hampering the development of self-supervised learning. In an attempt to reduce the computation of training, we revisit the training of HuBERT, a highly successful self-supervised model. We improve and simplify several key components, including the loss function, input representation, and training in multiple stages. Our model, MelHuBERT, is able to achieve favorable performance on phone recognition, speaker identification, and automatic speech recognition against HuBERT, while saving 31.2% of the pre-training time, or equivalently 33.5% MACs per one second speech. The code and pre-trained models are available in https://github.com/nervjack2/MelHuBERT.
    

