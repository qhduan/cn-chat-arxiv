# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Hatred Stems from Ignorance! Distillation of the Persuasion Modes in Countering Conversational Hate Speech](https://arxiv.org/abs/2403.15449) | 研究研究了对抗在线仇恨言论的最佳方法，通过分析对话中的理由、情感和信誉等说服方式，对比封闭和开放交互中的不同行为和话题层面，发现了在对抗言论中的微妙差异。 |
| [^2] | [Privacy-Aware Semantic Cache for Large Language Models](https://arxiv.org/abs/2403.02694) | MeanCache是一种面向LLMs的语义缓存，能够识别语义上相似的查询，从而减少查询成本，服务提供商负载和环境影响。 |
| [^3] | [TeacherLM: Teaching to Fish Rather Than Giving the Fish, Language Modeling Likewise.](http://arxiv.org/abs/2310.19019) | TeacherLM-7.1B是一个小型模型，通过给自然语言处理样本进行注释，教会其他模型“为什么”而不仅仅是“什么”。它在MMLU上取得了52.3的零样本得分，同时具有出色的数据增强能力。发布TeacherLM系列模型和增强的数据集作为开源项目。 |
| [^4] | [Physics of Language Models: Part 3.2, Knowledge Manipulation.](http://arxiv.org/abs/2309.14402) | 本文研究了语言模型在推理过程中操控知识的能力，发现预训练模型在知识检索方面表现出色，但在简单的分类、比较和逆向搜索任务中表现不佳。作者还提供了一个合成数据集进行实验，验证了这些内在的弱点：语言模型无法高效地操控知识。 |
| [^5] | [Examining the Influence of Varied Levels of Domain Knowledge Base Inclusion in GPT-based Intelligent Tutors.](http://arxiv.org/abs/2309.12367) | 本文研究了在基于GPT的智能辅导系统中将领域知识库与语言模型集成，以提高回答的可靠性。通过设计可扩展的知识库和评估实验，我们展示了该系统的有效性。学生和领域专家对于智能辅导系统的回答进行了验证和排名。 |
| [^6] | [Three Ways to Improve Verbo-visual Fusion for Dense 3D Visual Grounding.](http://arxiv.org/abs/2309.04561) | 提出了一个稠密三维引用网络ConcreteNet，包含三个新模块，旨在改善具有相同语义类别干扰因素的重复实例的引用性能。 |
| [^7] | [The Elephant in the Room: Analyzing the Presence of Big Tech in Natural Language Processing Research.](http://arxiv.org/abs/2305.02797) | 本文研究了工业界在自然语言处理研究中的存在和影响。研究发现在过去五年中，工业界的存在与影响呈现急剧增长，一些公司占据了大部分出版物，并向学术研究人员提供资金支持。 |
| [^8] | [Towards Responsible AI in the Era of ChatGPT: A Reference Architecture for Designing Foundation Model-based AI Systems.](http://arxiv.org/abs/2304.11090) | 本文提出了一个以模式为导向的负责任AI-by-design参考架构，用于设计基于基础模型的AI系统，重点关注可解释性、公平性、安全性和鲁棒性等关键设计元素。 |
| [^9] | [LMExplainer: a Knowledge-Enhanced Explainer for Language Models.](http://arxiv.org/abs/2303.16537) | LMExplainer是一种知识增强的语言模型解释模块，使用知识图和图注意力神经网络来提取关键决策信号，为用户提供可理解的解释。 |

# 详细

[^1]: 憎恨源于无知！对抗会话性仇恨言论中说服方式的提炼

    Hatred Stems from Ignorance! Distillation of the Persuasion Modes in Countering Conversational Hate Speech

    [https://arxiv.org/abs/2403.15449](https://arxiv.org/abs/2403.15449)

    研究研究了对抗在线仇恨言论的最佳方法，通过分析对话中的理由、情感和信誉等说服方式，对比封闭和开放交互中的不同行为和话题层面，发现了在对抗言论中的微妙差异。

    

    研究对抗言论使用的因素是理解在线对抗仇恨言论的最佳方法的核心。各种研究评估对抗言论中使用的情感基础因素，如情感共鸣、冒犯程度和敌意程度。为了更好地理解会话交互中使用的对抗言论，本研究将说服方式分解为理由、情感和信誉，然后评估它们在涉及种族主义、性别歧视和宗教问题的两种对话交互类型中的使用。评估涵盖了人类与生成对抗言论的不同行为。我们还评估了回复的立场与每种对抗言论中的说服方式之间的相互作用。值得注意的是，我们观察到了在开放和封闭交互的对抗言论说服方式上的微妙差异 -- 尤其是在话题层面上。

    arXiv:2403.15449v1 Announce Type: cross  Abstract: Examining the factors that the counter-speech uses is at the core of understanding the optimal methods for confronting hate speech online. Various studies assess the emotional base factor used in counter speech, such as emotion-empathy, offensiveness, and level of hostility. To better understand the counter-speech used in conversational interactions, this study distills persuasion modes into reason, emotion, and credibility and then evaluates their use in two types of conversation interactions: closed (multi-turn) and open (single-turn) conversation interactions concerning racism, sexism, and religion. The evaluation covers the distinct behaviors of human versus generated counter-speech. We also assess the interplay between the replies' stance and each mode of persuasion in the counter-speech. Notably, we observe nuanced differences in the counter-speech persuasion modes for open and closed interactions -- especially on the topic level
    
[^2]: 面向大型语言模型的隐私感知语义缓存

    Privacy-Aware Semantic Cache for Large Language Models

    [https://arxiv.org/abs/2403.02694](https://arxiv.org/abs/2403.02694)

    MeanCache是一种面向LLMs的语义缓存，能够识别语义上相似的查询，从而减少查询成本，服务提供商负载和环境影响。

    

    大型语言模型（LLMs）如ChatGPT、Google Bard、Claude和Llama 2彻底改变了自然语言处理和搜索引擎动态。然而，这些模型造成了异常高的计算成本。本文介绍了MeanCache，一种用于LLMs的语义缓存，它能够识别语义上相似的查询以确定缓存命中或未命中。

    arXiv:2403.02694v1 Announce Type: cross  Abstract: Large Language Models (LLMs) like ChatGPT, Google Bard, Claude, and Llama 2 have revolutionized natural language processing and search engine dynamics. However, these models incur exceptionally high computational costs. For instance, GPT-3 consists of 175 billion parameters and inference on these models also demands billions of floating-point operations. Caching is a natural solution to reduce LLM inference costs on repeated queries. However, existing caching methods are incapable of finding semantic similarities among LLM queries, leading to unacceptable false hit-and-miss rates.   This paper introduces MeanCache, a semantic cache for LLMs that identifies semantically similar queries to determine cache hit or miss. Using MeanCache, the response to a user's semantically similar query can be retrieved from a local cache rather than re-querying the LLM, thus reducing costs, service provider load, and environmental impact. MeanCache lever
    
[^3]: TeacherLM: 教人打鱼而不是给鱼，语言建模同理

    TeacherLM: Teaching to Fish Rather Than Giving the Fish, Language Modeling Likewise. (arXiv:2310.19019v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.19019](http://arxiv.org/abs/2310.19019)

    TeacherLM-7.1B是一个小型模型，通过给自然语言处理样本进行注释，教会其他模型“为什么”而不仅仅是“什么”。它在MMLU上取得了52.3的零样本得分，同时具有出色的数据增强能力。发布TeacherLM系列模型和增强的数据集作为开源项目。

    

    大型语言模型(LLMs)在各种自然语言处理任务中展现了惊人的推理和数据增强能力。然而，小型模型呢？在这项工作中，我们提出了TeacherLM-7.1B，能够给大多数自然语言处理样本进行相关基础知识、思维链和常见错误的注释，使注释不仅仅是一个答案，而且使其他模型可以学习“为什么”，而不仅仅是“什么”。TeacherLM-7.1B模型在MMLU上实现了52.3的零样本得分，超过了拥有100B参数的大多数模型。更令人印象深刻的是其数据增强能力。基于TeacherLM-7.1B，我们在多任务设置中使用了来自OPT和BLOOM系列的不同参数的多个学生模型对58个自然语言处理数据集进行了增强。实验结果表明，TeacherLM提供的数据增强带来了显着的好处。我们将作为开源发布TeacherLM系列模型和增强的数据集。

    Large Language Models (LLMs) exhibit impressive reasoning and data augmentation capabilities in various NLP tasks. However, what about small models? In this work, we propose TeacherLM-7.1B, capable of annotating relevant fundamentals, chain of thought, and common mistakes for most NLP samples, which makes annotation more than just an answer, thus allowing other models to learn "why" instead of just "what". The TeacherLM-7.1B model achieved a zero-shot score of 52.3 on MMLU, surpassing most models with over 100B parameters. Even more remarkable is its data augmentation ability. Based on TeacherLM-7.1B, we augmented 58 NLP datasets and taught various student models with different parameters from OPT and BLOOM series in a multi-task setting. The experimental results indicate that the data augmentation provided by TeacherLM has brought significant benefits. We will release the TeacherLM series of models and augmented datasets as open-source.
    
[^4]: 语言模型的物理学：第3.2部分，知识操控

    Physics of Language Models: Part 3.2, Knowledge Manipulation. (arXiv:2309.14402v1 [cs.CL])

    [http://arxiv.org/abs/2309.14402](http://arxiv.org/abs/2309.14402)

    本文研究了语言模型在推理过程中操控知识的能力，发现预训练模型在知识检索方面表现出色，但在简单的分类、比较和逆向搜索任务中表现不佳。作者还提供了一个合成数据集进行实验，验证了这些内在的弱点：语言模型无法高效地操控知识。

    

    语言模型可以存储大量事实知识，但它们在使用这些知识进行逻辑推理方面的能力仍然存在问题。本文探讨了语言模型在推理过程中操控其存储知识的能力。我们重点研究了四种操控类型：检索（例如，“A的属性X是什么”）、分类（例如，“A的属性X是奇数还是偶数”）、比较（例如，“在属性X中A是否大于B”）和逆向搜索（例如，“哪个人的属性X等于T”）。我们观察到，像GPT2/3/4这样的预训练语言模型在知识检索方面表现出色，但在简单的分类或比较任务中很难胜任，除非在训练和推理过程中采用了Chain of Thoughts（CoTs）。无论提示是什么，它们在逆向知识搜索中表现都很差。我们的主要贡献是一个为控制实验而设计的合成数据集，证实了这些内在的弱点：语言模型无法高效地操控知识。

    Language models can store vast amounts of factual knowledge, but their ability to use this knowledge for logical reasoning remains questionable. This paper explores a language model's ability to manipulate its stored knowledge during inference. We focus on four manipulation types: retrieval (e.g., "What is person A's attribute X"), classification (e.g., "Is A's attribute X even or odd?"), comparison (e.g., "Is A greater than B in attribute X?") and inverse search (e.g., "Which person's attribute X equals T?")  We observe that pre-trained language models like GPT2/3/4 excel in knowledge retrieval but struggle with simple classification or comparison tasks unless Chain of Thoughts (CoTs) are employed during both training and inference. They also perform poorly in inverse knowledge search, irrespective of the prompts. Our primary contribution is a synthetic dataset for a controlled experiment that confirms these inherent weaknesses: a language model cannot efficiently manipulate knowledge
    
[^5]: 在基于GPT的智能辅导系统中研究领域知识库不同程度的影响

    Examining the Influence of Varied Levels of Domain Knowledge Base Inclusion in GPT-based Intelligent Tutors. (arXiv:2309.12367v1 [cs.HC])

    [http://arxiv.org/abs/2309.12367](http://arxiv.org/abs/2309.12367)

    本文研究了在基于GPT的智能辅导系统中将领域知识库与语言模型集成，以提高回答的可靠性。通过设计可扩展的知识库和评估实验，我们展示了该系统的有效性。学生和领域专家对于智能辅导系统的回答进行了验证和排名。

    

    最近大型语言模型（LLM）的进展促进了具有复杂对话能力的聊天机器人的发展。然而，LLM对查询的回答经常不准确，这限制了在教育环境中的应用。本文研究了将知识库（KB）与LLM智能辅导系统集成以增加回答可靠性的效果。为了实现这一目标，我们设计了一个可扩展的知识库，教育监督员可以无缝集成课程，该课程会被智能辅导系统自动处理。然后，我们详细介绍了一个评估实验，学生参与者需要回答有关人工智能课程的问题。 GPT-4智能辅导系统具有不同层次的KB访问权限，并由人类领域专家评估这些回答。最后，学生对智能辅导系统的回答进行了与领域专家的交叉验证，并对它们的各种教学能力进行了排名。

    Recent advancements in large language models (LLMs) have facilitated the development of chatbots with sophisticated conversational capabilities. However, LLMs exhibit frequent inaccurate responses to queries, hindering applications in educational settings. In this paper, we investigate the effectiveness of integrating a knowledge base (KB) with LLM intelligent tutors to increase response reliability. To achieve this, we design a scaleable KB that affords educational supervisors seamless integration of lesson curricula, which is automatically processed by the intelligent tutoring system. We then detail an evaluation, where student participants were presented with questions about the artificial intelligence curriculum to respond to. GPT-4 intelligent tutors with varying hierarchies of KB access and human domain experts then assessed these responses. Lastly, students cross-examined the intelligent tutors' responses to the domain experts' and ranked their various pedagogical abilities. Res
    
[^6]: 改进稠密三维视觉引用的三种方法

    Three Ways to Improve Verbo-visual Fusion for Dense 3D Visual Grounding. (arXiv:2309.04561v1 [cs.CV])

    [http://arxiv.org/abs/2309.04561](http://arxiv.org/abs/2309.04561)

    提出了一个稠密三维引用网络ConcreteNet，包含三个新模块，旨在改善具有相同语义类别干扰因素的重复实例的引用性能。

    

    三维视觉引用是指通过自然语言描述来定位三维场景中被引用的物体的任务。该任务在自主室内机器人到AR/VR等各种应用中广泛应用。目前一种常见的解决方案是通过检测来完成三维视觉引用，即通过边界框来定位。然而，在需要进行物理交互的实际应用中，边界框不足以描述物体的几何属性。因此，我们解决了稠密三维视觉引用的问题，即基于引用的三维实例分割。我们提出了一个稠密三维引用网络ConcreteNet，其中包含三个独立的新模块，旨在改进具有相同语义类别干扰因素的具有挑战性的重复实例的引用性能。首先，我们引入了一个自下而上的注意力融合模块，旨在消除实例间关系线索的歧义性。接下来，我们构造一个cont

    3D visual grounding is the task of localizing the object in a 3D scene which is referred by a description in natural language. With a wide range of applications ranging from autonomous indoor robotics to AR/VR, the task has recently risen in popularity. A common formulation to tackle 3D visual grounding is grounding-by-detection, where localization is done via bounding boxes. However, for real-life applications that require physical interactions, a bounding box insufficiently describes the geometry of an object. We therefore tackle the problem of dense 3D visual grounding, i.e. referral-based 3D instance segmentation. We propose a dense 3D grounding network ConcreteNet, featuring three novel stand-alone modules which aim to improve grounding performance for challenging repetitive instances, i.e. instances with distractors of the same semantic class. First, we introduce a bottom-up attentive fusion module that aims to disambiguate inter-instance relational cues, next we construct a cont
    
[^7]: 房间里的大象：分析大型科技公司在自然语言处理研究中的存在

    The Elephant in the Room: Analyzing the Presence of Big Tech in Natural Language Processing Research. (arXiv:2305.02797v1 [cs.CL])

    [http://arxiv.org/abs/2305.02797](http://arxiv.org/abs/2305.02797)

    本文研究了工业界在自然语言处理研究中的存在和影响。研究发现在过去五年中，工业界的存在与影响呈现急剧增长，一些公司占据了大部分出版物，并向学术研究人员提供资金支持。

    

    自然语言处理的深度学习方法的最新进展，创造了新的商业机会，并且使得NLP研究对产业发展至关重要。作为NLP领域的大玩家之一，连同政府和大学一起，跟踪产业对研究的影响非常重要。在本研究中，我们致力于量化和表征工业界在NLP社区中的存在。使用具有78,187篇NLP出版物和701个NLP作者简历的全面元数据语料库，我们探索了自上世纪90年代以来该领域中的工业存在。我们发现，NLP作者中的工业存在在过去五年中急剧增长（从2017年到2022年的增长率为180％）。一些公司占据了大部分出版物，并通过拨款和实习为学术研究人员提供资金支持。我们的研究表明，工业界对自然语言处理研究的存在和影响是显著的。

    Recent advances in deep learning methods for natural language processing (NLP) have created new business opportunities and made NLP research critical for industry development. As one of the big players in the field of NLP, together with governments and universities, it is important to track the influence of industry on research. In this study, we seek to quantify and characterize industry presence in the NLP community over time. Using a corpus with comprehensive metadata of 78,187 NLP publications and 701 resumes of NLP publication authors, we explore the industry presence in the field since the early 90s. We find that industry presence among NLP authors has been steady before a steep increase over the past five years (180% growth from 2017 to 2022). A few companies account for most of the publications and provide funding to academic researchers through grants and internships. Our study shows that the presence and impact of the industry on natural language processing research are signi
    
[^8]: 在ChatGPT时代迈向负责任的人工智能：用于设计基于基础模型的AI系统的参考架构

    Towards Responsible AI in the Era of ChatGPT: A Reference Architecture for Designing Foundation Model-based AI Systems. (arXiv:2304.11090v1 [cs.CL])

    [http://arxiv.org/abs/2304.11090](http://arxiv.org/abs/2304.11090)

    本文提出了一个以模式为导向的负责任AI-by-design参考架构，用于设计基于基础模型的AI系统，重点关注可解释性、公平性、安全性和鲁棒性等关键设计元素。

    

    ChatGPT、Bard和其他大型语言模型(LLM)聊天机器人的推出在全球范围内引起了巨大关注。基础模型将成为未来大多数AI系统的基础构建块的趋势正在增长。然而，将基础模型纳入AI系统引发了对负责任AI的重大关注，这是由于其黑匣子性质和快速发展的超级智能引起的。此外，基础模型的增长能力最终可能会吞噬AI系统的其他组件，引入架构设计中的运动边界和接口演变挑战。为了应对这些挑战，本文提出了一种以模式为导向的负责任AI-by-design参考架构，用于设计基于基础模型的AI系统。特别地，本文首先呈现了基于基础模型的AI系统在架构演进方面的发展，从"基础模型作为连接器"到"基础模型作为单片机核"。然后，它提出了一个参考架构，包括五个类别的模式，重点关注关键设计元素，例如可解释性、公平性、安全性和鲁棒性。所提出的参考架构为设计负责任的基础模型的AI系统提供了系统化和透明的方法。

    The release of ChatGPT, Bard, and other large language model (LLM)-based chatbots has drawn huge attention on foundations models worldwide. There is a growing trend that foundation models will serve as the fundamental building blocks for most of the future AI systems. However, incorporating foundation models in AI systems raises significant concerns about responsible AI due to their black box nature and rapidly advancing super-intelligence. Additionally, the foundation model's growing capabilities can eventually absorb the other components of AI systems, introducing the moving boundary and interface evolution challenges in architecture design. To address these challenges, this paper proposes a pattern-oriented responsible-AI-by-design reference architecture for designing foundation model-based AI systems. Specially, the paper first presents an architecture evolution of AI systems in the era of foundation models, from "foundation-model-as-a-connector" to "foundation-model-as-a-monolithi
    
[^9]: LMExplainer：一种加强语言模型解释能力的知识提升模块

    LMExplainer: a Knowledge-Enhanced Explainer for Language Models. (arXiv:2303.16537v1 [cs.CL])

    [http://arxiv.org/abs/2303.16537](http://arxiv.org/abs/2303.16537)

    LMExplainer是一种知识增强的语言模型解释模块，使用知识图和图注意力神经网络来提取关键决策信号，为用户提供可理解的解释。

    

    巨型语言模型（如GPT-4）非常强大，可以处理各种自然语言处理（NLP）任务。然而，由于多层非线性模型结构和数百万个参数，很难解释其结果。对于用户而言，了解模型的工作方式缺乏理解，可能使模型在现实世界的应用中具有不可靠性和危险性。大多数最近的工作利用注意力权重来提供模型预测的解释。但是，基于注意力的解释无法支持不断增长的模型复杂性，并且无法推理其决策过程。因此，我们提出了LMExplainer，一种为语言模型提供人类可理解解释的知识增强模块。我们使用知识图和图注意力神经网络来提取LM的关键决策信号。同时，我们探讨解释能否也帮助人工智能更好地理解任务。

    Large language models (LMs) such as GPT-4 are very powerful and can process different kinds of natural language processing (NLP) tasks. However, it can be difficult to interpret the results due to the multi-layer nonlinear model structure and millions of parameters. Lack of understanding of how the model works can make the model unreliable and dangerous for everyday users in real-world scenarios. Most recent works exploit the weights of attention to provide explanations for model predictions. However, pure attention-based explanation is unable to support the growing complexity of the models, and cannot reason about their decision-making processes. Thus, we propose LMExplainer, a knowledge-enhanced interpretation module for language models that can provide human-understandable explanations. We use a knowledge graph (KG) and a graph attention neural network to extract the key decision signals of the LM. We further explore whether interpretation can also help AI understand the task better
    

