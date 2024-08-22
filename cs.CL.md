# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [How Much are LLMs Contaminated? A Comprehensive Survey and the LLMSanitize Library](https://arxiv.org/abs/2404.00699) | LLM受到污染可能导致其性能不可靠，挑战了自然语言处理领域的整体进展。 |
| [^2] | [ClaimVer: Explainable Claim-Level Verification and Evidence Attribution of Text Through Knowledge Graphs](https://arxiv.org/abs/2403.09724) | ClaimVer是一个人为中心的框架，通过知识图谱实现可解释的声明级验证和证据归因，致力于提高用户对文本验证方法的信任并强调细粒度证据的重要性。 |
| [^3] | [Persona-DB: Efficient Large Language Model Personalization for Response Prediction with Collaborative Data Refinement](https://arxiv.org/abs/2402.11060) | 介绍了 Persona-DB，一个简单却有效的框架，通过层级构建过程和协同优化，改善了大规模语言模型个性化中数据库表示的泛化能力和检索效率。 |
| [^4] | [A Roadmap to Pluralistic Alignment](https://arxiv.org/abs/2402.05070) | 这篇论文提出了一条通向多元对齐的路线图，以解决设计AI系统能够服务于人们具有不同价值观和观点的需求。论文介绍了对齐定义和实现多元主义的三种方式，并提出了三种多元基准类别来评估和测试多元对齐的效果。 |
| [^5] | [Clarify: Improving Model Robustness With Natural Language Corrections](https://arxiv.org/abs/2402.03715) | 论文提出了Clarify，一种通过自然语言纠正模型错误概念的方法，该方法通过用户提供简短的文本描述来纠正模型的一致失败模式，从而提高模型的鲁棒性。 |
| [^6] | [Large Language Models in Mental Health Care: a Scoping Review.](http://arxiv.org/abs/2401.02984) | 本综述研究对大型语言模型在心理健康护理中的应用和结果进行了综合分析，发现其在诊断、治疗和患者参与增强等方面具有多样化的应用。同时，该研究还识别和讨论了在这些专业领域中所面临的挑战和限制。 |
| [^7] | [InstructERC: Reforming Emotion Recognition in Conversation with a Retrieval Multi-task LLMs Framework.](http://arxiv.org/abs/2309.11911) | InstructERC是一种使用大型语言模型(LLMs)的生成式框架，通过引入检索模板模块和额外的情感对齐任务，改革了对话中的情绪识别。 |
| [^8] | [Kosmos-2.5: A Multimodal Literate Model.](http://arxiv.org/abs/2309.11419) | Kosmos-2.5是一个多模态文学模型，能够在机器阅读文本密集型图像方面表现出色，并能够生成具有空间感的文本块和结构化文本输出。该模型具有通用性，可以适应不同提示下任何文本密集型图像理解任务，并为未来的扩展提供了方向。 |
| [^9] | [SCALE: Scaling up the Complexity for Advanced Language Model Evaluation.](http://arxiv.org/abs/2306.09237) | 该论文提出了一个新颖的自然语言处理基准测试，挑战当前大型语言模型在处理长文档、利用领域专业知识、多语言理解和多任务处理方面的能力。基准测试包含瑞士法律系统的多样化法律NLP数据集，允许进行对底层非英语、固有多语言的法律系统进行全面研究。 |
| [^10] | [Competence-Based Analysis of Language Models.](http://arxiv.org/abs/2303.00333) | 该论文提出了一个基于能力的语言模型分析框架CALM，通过有针对性的干预来破坏语言模型的内部表示，评估其在执行任务时对不同表示的使用。研究表明，语言模型对关系属性的利用存在一定的不一致性。 |
| [^11] | [A Human Word Association based model for topic detection in social networks.](http://arxiv.org/abs/2301.13066) | 本文提出了一个基于人类词汇联想的社交网络主题检测框架，通过考虑语言结构并设计专门的抽取算法，在FA-CUP数据集上取得了比其他方法更好的性能。 |

# 详细

[^1]: LLM受到多少污染？一项全面调查和LLMSanitize库

    How Much are LLMs Contaminated? A Comprehensive Survey and the LLMSanitize Library

    [https://arxiv.org/abs/2404.00699](https://arxiv.org/abs/2404.00699)

    LLM受到污染可能导致其性能不可靠，挑战了自然语言处理领域的整体进展。

    

    随着近年来大型语言模型（LLMs）的崛起，新的机会正在出现，但也带来了新的挑战，污染问题迅速变得至关重要。企业应用和人工智能筹款已经达到一定规模，流行的问答基准提高几个百分点可能意味着数百万美元，对模型的完整性施加了巨大压力。同时，追踪LLMs见过的数据变得越来越困难；对于像GPT-4和Claude-3这样的闭源模型，他们不透露任何有关训练集的信息。因此，污染成为一个关键问题：LLMs的性能可能不再可靠，因为其高性能至少部分归因于其先前接触到的数据。这种局限性危及了自然语言处理领域的整体进展，然而，如何有效解决这一问题仍然缺乏方法。

    arXiv:2404.00699v1 Announce Type: new  Abstract: With the rise of Large Language Models (LLMs) in recent years, new opportunities are emerging, but also new challenges, and contamination is quickly becoming critical. Business applications and fundraising in AI have reached a scale at which a few percentage points gained on popular question-answering benchmarks could translate into dozens of millions of dollars, placing high pressure on model integrity. At the same time, it is becoming harder and harder to keep track of the data that LLMs have seen; if not impossible with closed-source models like GPT-4 and Claude-3 not divulging any information on the training set. As a result, contamination becomes a critical issue: LLMs' performance may not be reliable anymore, as the high performance may be at least partly due to their previous exposure to the data. This limitation jeopardizes the entire progress in the field of NLP, yet, there remains a lack of methods on how to efficiently address
    
[^2]: ClaimVer：通过知识图谱实现可解释的声明级验证和证据归因

    ClaimVer: Explainable Claim-Level Verification and Evidence Attribution of Text Through Knowledge Graphs

    [https://arxiv.org/abs/2403.09724](https://arxiv.org/abs/2403.09724)

    ClaimVer是一个人为中心的框架，通过知识图谱实现可解释的声明级验证和证据归因，致力于提高用户对文本验证方法的信任并强调细粒度证据的重要性。

    

    在广泛传播的信息误导和社交媒体以及人工智能生成的文本的激增中，验证和信任所遇到的信息变得日益困难。许多事实核查方法和工具已被开发，但它们往往缺乏适当的可解释性或细粒度，无法在各种情境中发挥作用。一种易于使用、可访问且能够执行细粒度证据归因的文本验证方法变得至关重要。更重要的是，建立用户对这种方法的信任需要呈现每个预测背后的理由，因为研究表明这显著影响人们对自动化系统的信任。将用户关注重点放在具体的问题内容上，而不是提供简单的笼统标签也非常重要。在本文中，我们提出了$\textit{ClaimVer，一个以人为中心的框架}$，旨在满足用户的信息需求。

    arXiv:2403.09724v1 Announce Type: new  Abstract: In the midst of widespread misinformation and disinformation through social media and the proliferation of AI-generated texts, it has become increasingly difficult for people to validate and trust information they encounter. Many fact-checking approaches and tools have been developed, but they often lack appropriate explainability or granularity to be useful in various contexts. A text validation method that is easy to use, accessible, and can perform fine-grained evidence attribution has become crucial. More importantly, building user trust in such a method requires presenting the rationale behind each prediction, as research shows this significantly influences people's belief in automated systems. It is also paramount to localize and bring users' attention to the specific problematic content, instead of providing simple blanket labels. In this paper, we present $\textit{ClaimVer, a human-centric framework}$ tailored to meet users' info
    
[^3]: Persona-DB：用于响应预测的高效大规模语言模型个性化与协同数据优化

    Persona-DB: Efficient Large Language Model Personalization for Response Prediction with Collaborative Data Refinement

    [https://arxiv.org/abs/2402.11060](https://arxiv.org/abs/2402.11060)

    介绍了 Persona-DB，一个简单却有效的框架，通过层级构建过程和协同优化，改善了大规模语言模型个性化中数据库表示的泛化能力和检索效率。

    

    随着对大型语言模型（LLMs）个性化交互需求的增加，需要开发能够准确快速识别用户意见和偏好的方法。检索增强作为一种有效策略出现，因为它可以适应大量用户而无需进行微调的成本。然而，现有研究主要集中在增强检索阶段，并对数据库表示的优化进行了有限的探索，这是个性化等任务的关键方面。在这项工作中，我们从一个新的角度研究了这个问题，着重于如何更有效地表示数据，以便在LLM定制的情境下更有效地进行检索。为了解决这一挑战，我们介绍了Persona-DB，这是一个简单而有效的框架，包括一个分层构建过程，以改善跨任务背景的泛化能力，并进行协同优化。

    arXiv:2402.11060v1 Announce Type: cross  Abstract: The increasing demand for personalized interactions with large language models (LLMs) calls for the development of methodologies capable of accurately and efficiently identifying user opinions and preferences. Retrieval augmentation emerges as an effective strategy, as it can accommodate a vast number of users without the costs from fine-tuning. Existing research, however, has largely focused on enhancing the retrieval stage and devoted limited exploration toward optimizing the representation of the database, a crucial aspect for tasks such as personalization. In this work, we examine the problem from a novel angle, focusing on how data can be better represented for more efficient retrieval in the context of LLM customization. To tackle this challenge, we introduce Persona-DB, a simple yet effective framework consisting of a hierarchical construction process to improve generalization across task contexts and collaborative refinement to
    
[^4]: 通往多元对齐的路线图

    A Roadmap to Pluralistic Alignment

    [https://arxiv.org/abs/2402.05070](https://arxiv.org/abs/2402.05070)

    这篇论文提出了一条通向多元对齐的路线图，以解决设计AI系统能够服务于人们具有不同价值观和观点的需求。论文介绍了对齐定义和实现多元主义的三种方式，并提出了三种多元基准类别来评估和测试多元对齐的效果。

    

    随着人工智能系统的权力和普及程度的增加，设计能够为不同价值观和观点的人服务的人工智能系统变得愈发重要。然而，将模型对齐以服务多元人类价值观仍然是一个待解决的研究问题。在本文中，我们提出了一条通向多元对齐的路线图，具体使用语言模型作为测试平台。我们确定和形式化了三种可能的方式来定义和实现人工智能系统中的多元主义：1）Overton多元模型，展示合理反应的光谱；2）可操控的多元模型，可以调整以反映特定的观点；3）分布多元模型，在分布中很好地校准给定人群的模型。我们还提出和形式化了三种可能的多元基准类别：1）多目标基准；2）权衡可操控基准，鼓励模型对任意权衡进行调整；3）陪审团多元基准，明确地模拟了不同陪审团的意见。

    With increased power and prevalence of AI systems, it is ever more critical that AI systems are designed to serve all, i.e., people with diverse values and perspectives. However, aligning models to serve pluralistic human values remains an open research question. In this piece, we propose a roadmap to pluralistic alignment, specifically using language models as a test bed. We identify and formalize three possible ways to define and operationalize pluralism in AI systems: 1) Overton pluralistic models that present a spectrum of reasonable responses; 2) Steerably pluralistic models that can steer to reflect certain perspectives; and 3) Distributionally pluralistic models that are well-calibrated to a given population in distribution. We also propose and formalize three possible classes of pluralistic benchmarks: 1) Multi-objective benchmarks, 2) Trade-off steerable benchmarks, which incentivize models to steer to arbitrary trade-offs, and 3) Jury-pluralistic benchmarks which explicitly m
    
[^5]: 澄清：通过自然语言纠正提高模型的鲁棒性

    Clarify: Improving Model Robustness With Natural Language Corrections

    [https://arxiv.org/abs/2402.03715](https://arxiv.org/abs/2402.03715)

    论文提出了Clarify，一种通过自然语言纠正模型错误概念的方法，该方法通过用户提供简短的文本描述来纠正模型的一致失败模式，从而提高模型的鲁棒性。

    

    在监督学习中，模型被训练从静态数据集中提取相关性。这通常会导致模型依赖于高级错误概念。为了防止这种错误概念，我们必须提供额外的信息。现有的方法包括一些额外的实例级监督形式，例如标记虚假特征或来自平衡分布的额外标记数据。对于大规模数据集来说，这些策略可能会变得昂贵，因为它们需要以接近原始训练数据的规模进行额外注释。我们假设有针对性的关于模型错误概念的自然语言反馈是一种更有效的额外监督形式。我们引入了Clarify，一种新型界面和方法来交互式地纠正模型的错误概念。通过Clarify，用户只需要提供一个简短的文本描述来描述模型的一致性失败模式。然后，我们完全自动化地使用s

    In supervised learning, models are trained to extract correlations from a static dataset. This often leads to models that rely on high-level misconceptions. To prevent such misconceptions, we must necessarily provide additional information beyond the training data. Existing methods incorporate forms of additional instance-level supervision, such as labels for spurious features or additional labeled data from a balanced distribution. Such strategies can become prohibitively costly for large-scale datasets since they require additional annotation at a scale close to the original training data. We hypothesize that targeted natural language feedback about a model's misconceptions is a more efficient form of additional supervision. We introduce Clarify, a novel interface and method for interactively correcting model misconceptions. Through Clarify, users need only provide a short text description to describe a model's consistent failure patterns. Then, in an entirely automated way, we use s
    
[^6]: 大型语言模型在心理健康护理中的应用：一项综述研究

    Large Language Models in Mental Health Care: a Scoping Review. (arXiv:2401.02984v1 [cs.CL])

    [http://arxiv.org/abs/2401.02984](http://arxiv.org/abs/2401.02984)

    本综述研究对大型语言模型在心理健康护理中的应用和结果进行了综合分析，发现其在诊断、治疗和患者参与增强等方面具有多样化的应用。同时，该研究还识别和讨论了在这些专业领域中所面临的挑战和限制。

    

    目的：大型语言模型（LLM）的使用越来越广泛，需要对它们在心理健康护理领域的应用和结果进行全面的综述。本综述研究旨在对LLMs在心理健康护理中的现有发展和应用进行批判性分析，突出它们的成功，并识别这些专业领域中的挑战和限制。材料和方法：2023年11月，在PubMed、Web of Science、Google Scholar、arXiv、medRxiv和PsyArXiv六个数据库中进行了广泛的文献搜索，遵循2020年版的“系统评价和Meta分析的首选报告项目”（PRISMA）指南。最初识别了313篇出版物，按照研究纳入标准，最终选择了34篇出版物进行综述。结果：我们发现了LLMs在心理健康护理中的多种应用，包括诊断、治疗、患者参与增强等。关键挑战和限制方面的发现将被总结和讨论。

    Objective: The growing use of large language models (LLMs) stimulates a need for a comprehensive review of their applications and outcomes in mental health care contexts. This scoping review aims to critically analyze the existing development and applications of LLMs in mental health care, highlighting their successes and identifying their challenges and limitations in these specialized fields. Materials and Methods: A broad literature search was conducted in November 2023 using six databases (PubMed, Web of Science, Google Scholar, arXiv, medRxiv, and PsyArXiv) following the 2020 version of the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) guidelines. A total of 313 publications were initially identified, and after applying the study inclusion criteria, 34 publications were selected for the final review. Results: We identified diverse applications of LLMs in mental health care, including diagnosis, therapy, patient engagement enhancement, etc. Key challen
    
[^7]: InstructERC：借助检索多任务LLMs框架改革对话中的情绪识别

    InstructERC: Reforming Emotion Recognition in Conversation with a Retrieval Multi-task LLMs Framework. (arXiv:2309.11911v1 [cs.CL])

    [http://arxiv.org/abs/2309.11911](http://arxiv.org/abs/2309.11911)

    InstructERC是一种使用大型语言模型(LLMs)的生成式框架，通过引入检索模板模块和额外的情感对齐任务，改革了对话中的情绪识别。

    

    对话情绪识别(ERC)的发展一直受到管道设计复杂性的阻碍，导致ERC模型往往对特定数据集和对话模式过拟合。在本研究中，我们提出了一种新方法，即InstructERC，将ERC任务从判别式框架转化为基于大型语言模型(LLMs)的生成式框架。InstructERC有两个重要贡献：首先，InstructERC引入了一个简单而有效的检索模板模块，通过将历史对话内容、标签语句和情感领域演示与高语义相似性进行拼接，帮助模型明确地集成多粒度对话监督信息。此外，我们引入了两个额外的情感对齐任务，即说话人识别和情感预测任务，以隐式地建模对话角色关系和未来对话情绪倾向。我们的基于LLM的方法

    The development of emotion recognition in dialogue (ERC) has been consistently hindered by the complexity of pipeline designs, leading to ERC models that often overfit to specific datasets and dialogue patterns. In this study, we propose a novel approach, namely  InstructERC, to reformulates the ERC task from a discriminative framework to a generative framework based on Large Language Models (LLMs) . InstructERC has two significant contributions: Firstly, InstructERC introduces a simple yet effective retrieval template module, which helps the model explicitly integrate multi-granularity dialogue supervision information by concatenating the historical dialog content, label statement, and emotional domain demonstrations with high semantic similarity. Furthermore, we introduce two additional emotion alignment tasks, namely speaker identification and emotion prediction tasks, to implicitly model the dialogue role relationships and future emotional tendencies in conversations. Our LLM-based
    
[^8]: Kosmos-2.5: 一个多模态文学模型

    Kosmos-2.5: A Multimodal Literate Model. (arXiv:2309.11419v1 [cs.CL])

    [http://arxiv.org/abs/2309.11419](http://arxiv.org/abs/2309.11419)

    Kosmos-2.5是一个多模态文学模型，能够在机器阅读文本密集型图像方面表现出色，并能够生成具有空间感的文本块和结构化文本输出。该模型具有通用性，可以适应不同提示下任何文本密集型图像理解任务，并为未来的扩展提供了方向。

    

    我们介绍了Kosmos-2.5，一个用于对文本密集型图像进行机器阅读的多模态文学模型。Kosmos-2.5在两个不同但相互合作的转录任务中表现出色：(1) 生成具有空间感的文本块，其中每个文本块都被赋予其在图像中的空间坐标，以及(2) 生成以Markdown格式捕捉样式和结构的结构化文本输出。这种统一的多模态文学能力是通过共享的Transformer架构、任务特定的提示和灵活的文本表示实现的。我们在端到端的文档级文本识别和图像到Markdown文本生成上评估了Kosmos-2.5。此外，该模型可以通过监督微调轻松适应具有不同提示的任何文本密集型图像理解任务，使其成为涉及文本丰富图像的实际应用的通用工具。这项工作还为未来的扩展铺平了道路。

    We present Kosmos-2.5, a multimodal literate model for machine reading of text-intensive images. Pre-trained on large-scale text-intensive images, Kosmos-2.5 excels in two distinct yet cooperative transcription tasks: (1) generating spatially-aware text blocks, where each block of text is assigned its spatial coordinates within the image, and (2) producing structured text output that captures styles and structures into the markdown format. This unified multimodal literate capability is achieved through a shared Transformer architecture, task-specific prompts, and flexible text representations. We evaluate Kosmos-2.5 on end-to-end document-level text recognition and image-to-markdown text generation. Furthermore, the model can be readily adapted for any text-intensive image understanding task with different prompts through supervised fine-tuning, making it a general-purpose tool for real-world applications involving text-rich images. This work also paves the way for the future scaling o
    
[^9]: SCALE: 提升高级语言模型评估的复杂性

    SCALE: Scaling up the Complexity for Advanced Language Model Evaluation. (arXiv:2306.09237v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2306.09237](http://arxiv.org/abs/2306.09237)

    该论文提出了一个新颖的自然语言处理基准测试，挑战当前大型语言模型在处理长文档、利用领域专业知识、多语言理解和多任务处理方面的能力。基准测试包含瑞士法律系统的多样化法律NLP数据集，允许进行对底层非英语、固有多语言的法律系统进行全面研究。

    

    最近在大型语言模型（LLM）方面取得的进展已经饱和了许多自然语言处理基准测试（包括专业领域的基准测试），强调了需要新颖、更具挑战性的测试来正确评估LLM的能力。在本文中，我们引入了一个新颖的自然语言处理基准测试，对当前LLM的四个关键方面提出了挑战：处理长文档（多达50K个标记）、利用领域专业知识（体现在法律文本中）、多语言理解（涵盖五种语言）和多任务处理（包括法律文件到文件信息检索、法庭视图生成、重要决策摘要、引用提取和八个具有挑战性的文本分类任务）。我们的基准测试包含了来自瑞士法律系统的多样的法律NLP数据集，可以对底层非英语、固有多语言的联邦法律系统进行全面研究。尽管最近取得了进展，但对于强烈的审查/分析任务，高效地处理长文档仍然是一个挑战。

    Recent strides in Large Language Models (LLMs) have saturated many NLP benchmarks (even professional domain-specific ones), emphasizing the need for novel, more challenging novel ones to properly assess LLM capabilities. In this paper, we introduce a novel NLP benchmark that poses challenges to current LLMs across four key dimensions: processing long documents (up to 50K tokens), utilizing domain specific knowledge (embodied in legal texts), multilingual understanding (covering five languages), and multitasking (comprising legal document to document Information Retrieval, Court View Generation, Leading Decision Summarization, Citation Extraction, and eight challenging Text Classification tasks). Our benchmark comprises diverse legal NLP datasets from the Swiss legal system, allowing for a comprehensive study of the underlying Non-English, inherently multilingual, federal legal system. Despite recent advances, efficiently processing long documents for intense review/analysis tasks remai
    
[^10]: 基于能力的语言模型分析

    Competence-Based Analysis of Language Models. (arXiv:2303.00333v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2303.00333](http://arxiv.org/abs/2303.00333)

    该论文提出了一个基于能力的语言模型分析框架CALM，通过有针对性的干预来破坏语言模型的内部表示，评估其在执行任务时对不同表示的使用。研究表明，语言模型对关系属性的利用存在一定的不一致性。

    

    尽管大型预训练语言模型（LMs）在各种提示任务上取得了显著成功，但这些模型对输入或应用环境中的微小变化却异常脆弱。为了更好地理解这种行为并激励设计更健壮的LMs，我们提出了一个通用的实验框架CALM（基于能力的语言模型分析），其中利用有针对性的因果干预来破坏LM在各种语言属性上的内部表示，以评估它在执行给定任务时对每个表示的使用。我们将这些干预实现为基于梯度的对抗攻击，与先前的因果探查方法相比，它们能够针对任意编码的关系属性进行攻击，并进行了一个案例研究，分析了BERT-like LMs在执行相关关系提示任务时如何使用多种关系属性的表示。我们发现，虽然表示的选择对LM的性能产生了影响，但模型对某些特定关系属性的利用并不一致。

    Despite the recent success of large pretrained language models (LMs) on a variety of prompting tasks, these models can be alarmingly brittle to small changes in inputs or application contexts. To better understand such behavior and motivate the design of more robust LMs, we propose a general experimental framework, CALM (Competence-based Analysis of Language Models), where targeted causal interventions are utilized to damage an LM's internal representation of various linguistic properties in order to evaluate its use of each representation in performing a given task. We implement these interventions as gradient-based adversarial attacks, which (in contrast to prior causal probing methodologies) are able to target arbitrarily-encoded representations of relational properties, and carry out a case study of this approach to analyze how BERT-like LMs use representations of several relational properties in performing associated relation prompting tasks. We find that, while the representation
    
[^11]: 基于人类词汇联想的社交网络主题检测模型

    A Human Word Association based model for topic detection in social networks. (arXiv:2301.13066v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2301.13066](http://arxiv.org/abs/2301.13066)

    本文提出了一个基于人类词汇联想的社交网络主题检测框架，通过考虑语言结构并设计专门的抽取算法，在FA-CUP数据集上取得了比其他方法更好的性能。

    

    随着社交网络的广泛使用，检测这些网络中讨论的主题已成为一个重要的挑战。目前的工作主要基于频繁模式挖掘或语义关系，而没有考虑语言结构。语言结构方法的意义在于发现词语之间的关系以及人类如何理解它们。因此，本文利用词汇联想的心理能力模拟概念，提出了一种基于人类词汇联想的社交网络主题检测框架。该框架基于人类词汇联想方法，并设计了专门的抽取算法。该方法在FA-CUP数据集上进行了评估，该数据集是主题检测领域的基准数据集。结果表明，与其他方法相比，所提出的方法在主题召回率和关键词F1值上有较好的改进。此外，主题检测领域中的大多数先前工作主要基于模式挖掘或语义关系。

    With the widespread use of social networks, detecting the topics discussed in these networks has become a significant challenge. The current works are mainly based on frequent pattern mining or semantic relations, and the language structure is not considered. The meaning of language structural methods is to discover the relationship between words and how humans understand them. Therefore, this paper uses the Concept of the Imitation of the Mental Ability of Word Association to propose a topic detection framework in social networks. This framework is based on the Human Word Association method. A special extraction algorithm has also been designed for this purpose. The performance of this method is evaluated on the FA-CUP dataset. It is a benchmark dataset in the field of topic detection. The results show that the proposed method is a good improvement compared to other methods, based on the Topic-recall and the keyword F1 measure. Also, most of the previous works in the field of topic de
    

