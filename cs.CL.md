# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hatred Stems from Ignorance! Distillation of the Persuasion Modes in Countering Conversational Hate Speech](https://arxiv.org/abs/2403.15449) | 研究研究了对抗在线仇恨言论的最佳方法，通过分析对话中的理由、情感和信誉等说服方式，对比封闭和开放交互中的不同行为和话题层面，发现了在对抗言论中的微妙差异。 |
| [^2] | [Re-Ex: Revising after Explanation Reduces the Factual Errors in LLM Responses](https://arxiv.org/abs/2402.17097) | 提出了一种名为Re-Ex的方法，通过引入事实错误说明步骤来修正LLM生成文本中的事实错误，并提出了新的提示技术来减少所需的标记数量和挂钟时间 |
| [^3] | [COMPASS: Computational Mapping of Patient-Therapist Alliance Strategies with Language Modeling](https://arxiv.org/abs/2402.14701) | 本文提出了一种名为COMPASS的新框架，通过分析心理治疗会话中的自然语言，直接推断治疗工作联盟，为临床精神病学提供了可解释性，并在识别与正在治疗的疾病相关的新兴模式方面发挥作用。 |
| [^4] | [Adaptive Skeleton Graph Decoding](https://arxiv.org/abs/2402.12280) | 提出了骨架图解码（SGD）方法，利用子问题之间的依赖关系进行信息转发，改善响应质量且提高性能。 |
| [^5] | [Rethinking the Evaluation of Pre-trained Text-and-Layout Models from an Entity-Centric Perspective](https://arxiv.org/abs/2402.02379) | 本论文从实体中心的角度重新思考了对预训练的文本和布局模型的评估。提出了评估PTLMs信息提取能力的理想基准的标准，并介绍了针对该评估的EC-FUNSD数据集。实验结果表明，最先进的PTLMs在预训练阶段存在过拟合的倾向。 |
| [^6] | [Embedding Ontologies via Incoprorating Extensional and Intensional Knowledge](https://arxiv.org/abs/2402.01677) | 本文提出了一种新型本体嵌入方法EIKE，通过整合外延知识和内涵知识，在外延空间和内涵空间中表示本体，并采用基于几何的方法和预训练的语言模型对实例、概念和关系进行嵌入建模。 |
| [^7] | [Testing the Predictions of Surprisal Theory in 11 Languages.](http://arxiv.org/abs/2307.03667) | 本研究填补了现有文献中的空白，通过研究11种不同语言之间的surprisal与阅读时间之间的关系，测试了Surprisal理论的三个预测，并发现了其他语言特征对阅读时间的影响。 |
| [^8] | [Ham2Pose: Animating Sign Language Notation into Pose Sequences.](http://arxiv.org/abs/2211.13613) | 该论文提出了一种将HamNoSys符号转换为手语姿势序列的方法，使用变压器编码器建立文本和姿势间的有意义的表示，可用于不同手语之间的通用翻译。此外，提出了一种新的距离测量方法可以度量手语姿势序列之间的距离。 |

# 详细

[^1]: 憎恨源于无知！对抗会话性仇恨言论中说服方式的提炼

    Hatred Stems from Ignorance! Distillation of the Persuasion Modes in Countering Conversational Hate Speech

    [https://arxiv.org/abs/2403.15449](https://arxiv.org/abs/2403.15449)

    研究研究了对抗在线仇恨言论的最佳方法，通过分析对话中的理由、情感和信誉等说服方式，对比封闭和开放交互中的不同行为和话题层面，发现了在对抗言论中的微妙差异。

    

    研究对抗言论使用的因素是理解在线对抗仇恨言论的最佳方法的核心。各种研究评估对抗言论中使用的情感基础因素，如情感共鸣、冒犯程度和敌意程度。为了更好地理解会话交互中使用的对抗言论，本研究将说服方式分解为理由、情感和信誉，然后评估它们在涉及种族主义、性别歧视和宗教问题的两种对话交互类型中的使用。评估涵盖了人类与生成对抗言论的不同行为。我们还评估了回复的立场与每种对抗言论中的说服方式之间的相互作用。值得注意的是，我们观察到了在开放和封闭交互的对抗言论说服方式上的微妙差异 -- 尤其是在话题层面上。

    arXiv:2403.15449v1 Announce Type: cross  Abstract: Examining the factors that the counter-speech uses is at the core of understanding the optimal methods for confronting hate speech online. Various studies assess the emotional base factor used in counter speech, such as emotion-empathy, offensiveness, and level of hostility. To better understand the counter-speech used in conversational interactions, this study distills persuasion modes into reason, emotion, and credibility and then evaluates their use in two types of conversation interactions: closed (multi-turn) and open (single-turn) conversation interactions concerning racism, sexism, and religion. The evaluation covers the distinct behaviors of human versus generated counter-speech. We also assess the interplay between the replies' stance and each mode of persuasion in the counter-speech. Notably, we observe nuanced differences in the counter-speech persuasion modes for open and closed interactions -- especially on the topic level
    
[^2]: 修复: 在说明后修正LLM响应中的事实错误

    Re-Ex: Revising after Explanation Reduces the Factual Errors in LLM Responses

    [https://arxiv.org/abs/2402.17097](https://arxiv.org/abs/2402.17097)

    提出了一种名为Re-Ex的方法，通过引入事实错误说明步骤来修正LLM生成文本中的事实错误，并提出了新的提示技术来减少所需的标记数量和挂钟时间

    

    缓解幻觉问题是LLM的主要挑战之一，我们需要克服这一挑战，以便可靠地在现实场景中使用它们。最近，提出了各种方法来检查LLM生成的文本中的事实错误，并相应地进行修订，以减少幻觉问题。在本文中，我们提出了Re-Ex，一种修订LLM生成文本的方法，它引入了一个称为事实错误说明步骤的新步骤。 Re-Ex使用3个步骤对LLM的初始响应进行修订：首先，使用外部工具获取响应中事实错误的证据；第二，要求LLM根据第一步中收集的证据解释响应中的问题部分；最后，LLM使用在第二步中获得的解释对响应进行修订。除了说明步骤，我们还提出了新的提示技术，以减少所需的标记数量和挂钟时间。

    arXiv:2402.17097v1 Announce Type: cross  Abstract: Mitigating hallucination issues is one of the main challenges of LLMs we need to overcome, in order to reliably use them in real-world scenarios. Recently, various methods are proposed to check the factual errors in the LLM-generated texts and revise them accordingly, to reduce the hallucination issue. In this paper, we propose Re-Ex, a method of revising LLM-generated texts, which introduces a novel step dubbed as the factual error explanation step. Re-Ex revises the initial response of LLMs using 3-steps: first, external tools are used to get the evidences on the factual errors in the response; second, LLMs are instructed to explain the problematic parts of the response based on the evidences gathered in the first step; finally, LLMs revise the response using the explanation obtained in the second step. In addition to the explanation step, we propose new prompting techniques to reduce the amount of tokens and wall-clock time required
    
[^3]: COMPASS：利用语言建模对患者-治疗师联盟策略进行计算映射

    COMPASS: Computational Mapping of Patient-Therapist Alliance Strategies with Language Modeling

    [https://arxiv.org/abs/2402.14701](https://arxiv.org/abs/2402.14701)

    本文提出了一种名为COMPASS的新框架，通过分析心理治疗会话中的自然语言，直接推断治疗工作联盟，为临床精神病学提供了可解释性，并在识别与正在治疗的疾病相关的新兴模式方面发挥作用。

    

    治疗工作联盟是预测心理治疗治疗成功的关键因素。传统上，工作联盟评估依赖于治疗师和患者填写的问卷。本文提出了COMPASS，一个新颖的框架，可直接从心理治疗课程中使用的自然语言中推断治疗工作联盟。我们的方法利用先进的大型语言模型分析心理治疗会话的转录，并将其与工作联盟清单中陈述的分布式表示进行比较。通过分析涵盖多种精神疾病的超过950个会话的数据集，我们展示了我们的方法在显微地映射患者-治疗师对齐轨迹方面的有效性，并为临床精神病学提供解释性，并在识别与正在治疗的疾病相关的新兴模式方面提供可解释性。通过使用各种神经主题模式

    arXiv:2402.14701v1 Announce Type: cross  Abstract: The therapeutic working alliance is a critical factor in predicting the success of psychotherapy treatment. Traditionally, working alliance assessment relies on questionnaires completed by both therapists and patients. In this paper, we present COMPASS, a novel framework to directly infer the therapeutic working alliance from the natural language used in psychotherapy sessions. Our approach utilizes advanced large language models to analyze transcripts of psychotherapy sessions and compare them with distributed representations of statements in the working alliance inventory. Analyzing a dataset of over 950 sessions covering diverse psychiatric conditions, we demonstrate the effectiveness of our method in microscopically mapping patient-therapist alignment trajectories and providing interpretability for clinical psychiatry and in identifying emerging patterns related to the condition being treated. By employing various neural topic mode
    
[^4]: 自适应骨架图解码

    Adaptive Skeleton Graph Decoding

    [https://arxiv.org/abs/2402.12280](https://arxiv.org/abs/2402.12280)

    提出了骨架图解码（SGD）方法，利用子问题之间的依赖关系进行信息转发，改善响应质量且提高性能。

    

    大型语言模型（LLMs）已经在自然语言任务中得到广泛应用，其成功归因于大量的模型参数（例如，70亿+）；然而，LLM推断会产生巨大的计算和内存成本。最近的方法提出了并行解码策略，例如“思想骨架”（SoT），通过将提示分解为可以并行解码的子问题来改善性能；但是，它们往往在响应质量上遭受损失。我们的关键见解是，在生成子问题时，我们可以请求额外信息，特别是依赖关系和难度，以提高响应质量和性能。在本文中，我们提出了骨架图解码（SGD），利用子问题之间暴露的依赖关系，支持依赖子问题之间的信息转发，以提高质量，同时暴露独立子问题解码的并行化机会。

    arXiv:2402.12280v1 Announce Type: cross  Abstract: Large language models (LLMs) have seen significant adoption for natural language tasks, owing their success to massive numbers of model parameters (e.g., 70B+); however, LLM inference incurs significant computation and memory costs. Recent approaches propose parallel decoding strategies, such as Skeleton-of-Thought (SoT), to improve performance by breaking prompts down into sub-problems that can be decoded in parallel; however, they often suffer from reduced response quality. Our key insight is that we can request additional information, specifically dependencies and difficulty, when generating the sub-problems to improve both response quality and performance. In this paper, we propose Skeleton Graph Decoding (SGD), which uses dependencies exposed between sub-problems to support information forwarding between dependent sub-problems for improved quality while exposing parallelization opportunities for decoding independent sub-problems. 
    
[^5]: 从实体中心的角度重新思考对预训练的文本和布局模型的评估

    Rethinking the Evaluation of Pre-trained Text-and-Layout Models from an Entity-Centric Perspective

    [https://arxiv.org/abs/2402.02379](https://arxiv.org/abs/2402.02379)

    本论文从实体中心的角度重新思考了对预训练的文本和布局模型的评估。提出了评估PTLMs信息提取能力的理想基准的标准，并介绍了针对该评估的EC-FUNSD数据集。实验结果表明，最先进的PTLMs在预训练阶段存在过拟合的倾向。

    

    最近开发的预训练的文本和布局模型（PTLMs）在视觉丰富的文档上的多个信息提取任务中取得了显著的成功。然而，由于基准数据中的注释不足，目前的评估流程可能不够稳健，无法充分评估PTLMs的信息提取能力。因此，我们提出了评估PTLMs信息提取能力的理想基准的必要标准。我们还介绍了EC-FUNSD，这是一个针对视觉丰富文档上语义实体识别和实体链接评估而设计的以实体为中心的基准数据集。该数据集包含不同格式的文档布局和语义驱动实体及其关系的注释。此外，该数据集还解开了由FUNSD的分段级注释带来的段落和实体错误耦合的问题。实验结果表明，最先进的PTLMs在预训练阶段存在过拟合的倾向。

    Recently developed pre-trained text-and-layout models (PTLMs) have shown remarkable success in multiple information extraction tasks on visually-rich documents. However, the prevailing evaluation pipeline may not be sufficiently robust for assessing the information extraction ability of PTLMs, due to inadequate annotations within the benchmarks. Therefore, we claim the necessary standards for an ideal benchmark to evaluate the information extraction ability of PTLMs. We then introduce EC-FUNSD, an entity-centric benckmark designed for the evaluation of semantic entity recognition and entity linking on visually-rich documents. This dataset contains diverse formats of document layouts and annotations of semantic-driven entities and their relations. Moreover, this dataset disentangles the falsely coupled annotation of segment and entity that arises from the block-level annotation of FUNSD. Experiment results demonstrate that state-of-the-art PTLMs exhibit overfitting tendencies on the pre
    
[^6]: 通过整合外延知识和内涵知识嵌入本体

    Embedding Ontologies via Incoprorating Extensional and Intensional Knowledge

    [https://arxiv.org/abs/2402.01677](https://arxiv.org/abs/2402.01677)

    本文提出了一种新型本体嵌入方法EIKE，通过整合外延知识和内涵知识，在外延空间和内涵空间中表示本体，并采用基于几何的方法和预训练的语言模型对实例、概念和关系进行嵌入建模。

    

    本体包含领域内丰富的知识，可以分为两个类别，即外延知识和内涵知识。外延知识提供关于本体中特定概念所属的具体实例的信息，而内涵知识详细描述了概念之间的内在属性、特征和语义关联。然而，现有的本体嵌入方法未能同时充分考虑外延知识和内涵知识。在本文中，我们提出了一种名为EIKE（Extensional and Intensional Knowledge Embedding）的新型本体嵌入方法，通过在外延空间和内涵空间中表示本体。EIKE提出了一个统一的框架，用于将实例、概念及其关系嵌入到本体中，采用基于几何的方法对外延知识进行建模，并使用预训练的语言模型对内涵知识进行建模。

    Ontologies contain rich knowledge within domain, which can be divided into two categories, namely extensional knowledge and intensional knowledge. Extensional knowledge provides information about the concrete instances that belong to specific concepts in the ontology, while intensional knowledge details inherent properties, characteristics, and semantic associations among concepts. However, existing ontology embedding approaches fail to take both extensional knowledge and intensional knowledge into fine consideration simultaneously. In this paper, we propose a novel ontology embedding approach named EIKE (Extensional and Intensional Knowledge Embedding) by representing ontologies in two spaces, called extensional space and intensional space. EIKE presents a unified framework for embedding instances, concepts and their relations in an ontology, applying a geometry-based method to model extensional knowledge and a pretrained language model to model intensional knowledge, which can captur
    
[^7]: 在11种语言中测试Surprisal理论的预测

    Testing the Predictions of Surprisal Theory in 11 Languages. (arXiv:2307.03667v1 [cs.CL])

    [http://arxiv.org/abs/2307.03667](http://arxiv.org/abs/2307.03667)

    本研究填补了现有文献中的空白，通过研究11种不同语言之间的surprisal与阅读时间之间的关系，测试了Surprisal理论的三个预测，并发现了其他语言特征对阅读时间的影响。

    

    心理语言学的一个基本结果是，可预测性较低的词语需要更长时间来处理。Surprisal理论（Hale, 2001; Levy, 2008）是对这一发现的一个理论解释，它将一个词的可预测性量化为其surprisal，即在给定上下文的情况下，其负对数概率。虽然有大量的证据支持Surprisal理论的预测，但大多数研究都集中在一个非常有限的数据范围内，即以英语为母语的人阅读英语文本。事实上，目前还没有全面的多语言分析。我们通过研究在五个语言家族中分布的十一种不同语言中surprisal与阅读时间之间的关系来填补当前文献中的这一空白。通过从单语和多语语料库训练的语言模型中推导估计值，我们测试了与surprisal理论相关的三个预测：(i) surprisal是否能够预测阅读时间；(ii) 预期surprisal，即上下文熵，是否影响阅读时间；(iii) 与surprisal相关的其他语言特征是否可以解释阅读时间。

    A fundamental result in psycholinguistics is that less predictable words take a longer time to process. One theoretical explanation for this finding is Surprisal Theory (Hale, 2001; Levy, 2008), which quantifies a word's predictability as its surprisal, i.e. its negative log-probability given a context. While evidence supporting the predictions of Surprisal Theory have been replicated widely, most have focused on a very narrow slice of data: native English speakers reading English texts. Indeed, no comprehensive multilingual analysis exists. We address this gap in the current literature by investigating the relationship between surprisal and reading times in eleven different languages, distributed across five language families. Deriving estimates from language models trained on monolingual and multilingual corpora, we test three predictions associated with surprisal theory: (i) whether surprisal is predictive of reading times; (ii) whether expected surprisal, i.e. contextual entropy, i
    
[^8]: Ham2Pose：将手语符号转化成姿势序列的动画方法

    Ham2Pose: Animating Sign Language Notation into Pose Sequences. (arXiv:2211.13613v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2211.13613](http://arxiv.org/abs/2211.13613)

    该论文提出了一种将HamNoSys符号转换为手语姿势序列的方法，使用变压器编码器建立文本和姿势间的有意义的表示，可用于不同手语之间的通用翻译。此外，提出了一种新的距离测量方法可以度量手语姿势序列之间的距离。

    

    将口语翻译成手语对于聋听社区之间的开放性交流至关重要。为了实现这一目标，我们提出了第一种将HamNoSys，一种词汇手语符号，转换为手语姿势序列的动画方法。由于HamNoSys是通用设计的，我们提出的方法提供了不受目标手语限制的通用解决方案。我们的方法使用变压器编码器逐渐生成姿势预测，同时考虑它们的空间和时间信息，为训练过程提供了弱监督，并且显示我们的方法在从部分和不准确的数据中进行学习时成功。此外，我们提供了一种新的距离测量方法，考虑缺失关键点，使用DTW-MJE来测量姿势序列之间的距离。我们使用AUTSL这个大规模手语数据集来验证它的正确性，并且展示它可以度量手语之间的距离。

    Translating spoken languages into Sign languages is necessary for open communication between the hearing and hearing-impaired communities. To achieve this goal, we propose the first method for animating a text written in HamNoSys, a lexical Sign language notation, into signed pose sequences. As HamNoSys is universal by design, our proposed method offers a generic solution invariant to the target Sign language. Our method gradually generates pose predictions using transformer encoders that create meaningful representations of the text and poses while considering their spatial and temporal information. We use weak supervision for the training process and show that our method succeeds in learning from partial and inaccurate data. Additionally, we offer a new distance measurement that considers missing keypoints, to measure the distance between pose sequences using DTW-MJE. We validate its correctness using AUTSL, a large-scale Sign language dataset, show that it measures the distance betw
    

