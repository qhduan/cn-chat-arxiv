# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [An Expert is Worth One Token: Synergizing Multiple Expert LLMs as Generalist via Expert Token Routing](https://arxiv.org/abs/2403.16854) | 通过专家代币路由将多个专家LLM协同作为通用型，可以实现多个专家LLMs的无缝集成，支持隐式专业知识的学习和动态扩展新的专家LLMs，同时更好地隐藏协作细节，展现出比现有多LLM协作范式更好的效果和稳健性。 |
| [^2] | [Strong Priority and Determinacy in Timed CCS](https://arxiv.org/abs/2403.04618) | 引入了一种新的调度机制“顺序构造减少”，旨在实现多播并发通信的确定性，扩展了CCS的技术设置，证明了构造减少的汇聚属性，展示了在一些语法限制下运算符的结构连贯性。 |
| [^3] | [Loose LIPS Sink Ships: Asking Questions in Battleship with Language-Informed Program Sampling](https://arxiv.org/abs/2402.19471) | 研究利用大型语言模型提出信息量丰富的问题，在Battleship游戏中展示出与人类表现相匹配的效果，并揭示了贝叶斯模型如何指导问问题行为。 |
| [^4] | [What Changed? Converting Representational Interventions to Natural Language](https://arxiv.org/abs/2402.11355) | 将表征空间的反事实转化为自然语言，以分析和解释模型干预所引起的语言变化，并减轻分类中的偏见。 |
| [^5] | [Large Language Models as Zero-shot Dialogue State Tracker through Function Calling](https://arxiv.org/abs/2402.10466) | 本研究提出了一种通过函数调用将大型语言模型用于零-shot对话状态追踪的新方法，能够在任务导向对话中取得出色的性能，适应不同领域而无需大量数据收集或模型调整。 |
| [^6] | [CIC: A framework for Culturally-aware Image Captioning](https://arxiv.org/abs/2402.05374) | CIC是一种面向文化感知图像字幕的框架，通过结合视觉问答和大型语言模型，它能够生成能描述图像中文化元素的详细字幕。 |
| [^7] | [Structured Probabilistic Coding](https://arxiv.org/abs/2312.13933) | 结构化概率编码（SPC）是一种新的监督式表示学习框架，通过编码和预测任务的信息来学习紧凑且信息丰富的表示，提高语言模型的泛化能力和语言理解能力，并通过结构化正则化实现更好的覆盖率。 |
| [^8] | [On the Learnability of Watermarks for Language Models.](http://arxiv.org/abs/2312.04469) | 该论文研究了语言模型水印的可学习性，提出了水印蒸馏方法，通过训练学生模型使其模仿使用解码水印的教师模型的行为。结果表明，语言模型具有直接学习生成水印的能力，这对于水印的实际应用具有重要影响。 |
| [^9] | [Language Models As Semantic Indexers.](http://arxiv.org/abs/2310.07815) | 本文介绍了一种使用生成性语言模型学习语义ID的自监督框架LMINDEXER。 |
| [^10] | [HANS, are you clever? Clever Hans Effect Analysis of Neural Systems.](http://arxiv.org/abs/2309.12481) | 这项研究调查了指导调整的大型语言模型（It-LLMs）对多项选择题（MCQ）的鲁棒性能力，在选择顺序变动时揭示了选择偏见和推理能力的问题。 |
| [^11] | [CPLLM: Clinical Prediction with Large Language Models.](http://arxiv.org/abs/2309.11295) | CPLLM是一种使用大规模语言模型进行临床疾病预测的方法。通过量化和提示来微调语言模型，利用患者的历史诊断记录来预测目标疾病的诊断结果。实验证明，CPLLM在各项指标上均超越了其他基线模型，显示出显著的改进。 |
| [^12] | [Large language models can accurately predict searcher preferences.](http://arxiv.org/abs/2309.10621) | 大型语言模型可以通过从真实用户那里获取高质量的第一方数据来准确预测搜索者的偏好。 |
| [^13] | [LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked.](http://arxiv.org/abs/2308.07308) | 本文提出了一种通过自检来防御大型语言模型(LLMs)对抗性攻击的简单方法，即让模型自行过滤回应。实验结果表明，即使模型未对齐人类价值观，通过使用语言模型验证内容，仍然可以防止模型向用户呈现有害内容。 |
| [^14] | [ReactGenie: An Object-Oriented State Abstraction for Complex Multimodal Interactions Using Large Language Models.](http://arxiv.org/abs/2306.09649) | ReactGenie是一个支持构建复杂的多模态移动应用程序的编程框架，通过使用共享的面向对象状态抽象，使得不同模态可以无缝集成和组合，从而实现了多模态交互的支持。 |
| [^15] | [On the Reliability of Watermarks for Large Language Models.](http://arxiv.org/abs/2306.04634) | 本文研究了大型语言模型水印在混合其他文本来源时的可靠性，并提供了在实际应用中的建议。 |
| [^16] | [A Watermark for Large Language Models.](http://arxiv.org/abs/2301.10226) | 本文提出了一种在大型语言模型中实现水印技术的方法，该技术可以在不降低文本质量的前提下嵌入信号，且可以使用高效的开源算法进行检测，并且该技术十分鲁棒和安全。 |

# 详细

[^1]: 一个专家价值一个代币：通过专家代币路由将多个专家LLM协同作为通用型

    An Expert is Worth One Token: Synergizing Multiple Expert LLMs as Generalist via Expert Token Routing

    [https://arxiv.org/abs/2403.16854](https://arxiv.org/abs/2403.16854)

    通过专家代币路由将多个专家LLM协同作为通用型，可以实现多个专家LLMs的无缝集成，支持隐式专业知识的学习和动态扩展新的专家LLMs，同时更好地隐藏协作细节，展现出比现有多LLM协作范式更好的效果和稳健性。

    

    我们提出了专家代币路由（Expert-Token-Routing），这是一个统一的通用型框架，可以实现多个专家LLM的无缝集成。我们的框架将专家LLMs表示为元LLM词汇中的特殊专家代币。元LLM可以路由到专家LLM，就像生成新代币一样。专家代币路由不仅可以从现有的指导数据集中学习专家LLMs的隐式专业知识，还可以以即插即用的方式动态扩展新的专家LLMs。它还可以隐藏用户视角中的详细协作过程，促进交互就像是一个单一的LLM一样。我们的框架在涵盖六个不同专家领域的基准测试中胜过了各种现有的多LLM协作范式，展现了通过协同多个专家LLM来构建通用型LLM系统的效果和稳健性。

    arXiv:2403.16854v1 Announce Type: cross  Abstract: We present Expert-Token-Routing, a unified generalist framework that facilitates seamless integration of multiple expert LLMs. Our framework represents expert LLMs as special expert tokens within the vocabulary of a meta LLM. The meta LLM can route to an expert LLM like generating new tokens. Expert-Token-Routing not only supports learning the implicit expertise of expert LLMs from existing instruction dataset but also allows for dynamic extension of new expert LLMs in a plug-and-play manner. It also conceals the detailed collaboration process from the user's perspective, facilitating interaction as though it were a singular LLM. Our framework outperforms various existing multi-LLM collaboration paradigms across benchmarks that incorporate six diverse expert domains, demonstrating effectiveness and robustness in building generalist LLM system via synergizing multiple expert LLMs.
    
[^2]: 时标CCS中的强优先级和确定性

    Strong Priority and Determinacy in Timed CCS

    [https://arxiv.org/abs/2403.04618](https://arxiv.org/abs/2403.04618)

    引入了一种新的调度机制“顺序构造减少”，旨在实现多播并发通信的确定性，扩展了CCS的技术设置，证明了构造减少的汇聚属性，展示了在一些语法限制下运算符的结构连贯性。

    

    在具有优先级的经典进程代数理论的基础上，我们确定了一种名为“顺序构造减少”的新调度机制，旨在捕捉同步编程的本质。这种评估策略的独特属性是通过构造实现多播并发通信的确定性。特别是，这使我们能够模拟具有对缺失反应的共享内存多线程，因为它是Esterel编程语言的核心。在通过时钟和优先级扩展的CCS的技术设置中，对于我们称为“结构连贯”的大类过程，我们证明了构造减少的汇聚属性。我们进一步展示，在一些称为“可枢纽”的语法限制下，前缀、求和、并行组成、限制和隐藏的运算符保持结构连贯。这涵盖了一个严格更大的过程类。

    arXiv:2403.04618v1 Announce Type: cross  Abstract: Building on the classical theory of process algebra with priorities, we identify a new scheduling mechanism, called "sequentially constructive reduction" which is designed to capture the essence of synchronous programming. The distinctive property of this evaluation strategy is to achieve determinism-by-construction for multi-cast concurrent communication. In particular, it permits us to model shared memory multi-threading with reaction to absence as it lies at the core of the programming language Esterel. In the technical setting of CCS extended by clocks and priorities, we prove for a large class of processes, which we call "structurally coherent" the confluence property for constructive reductions. We further show that under some syntactic restrictions, called "pivotable" the operators of prefix, summation, parallel composition, restriction and hiding preserve structural coherence. This covers a strictly larger class of processes co
    
[^3]: 严格的LIPS沉没舰船：在Battleship中使用语言信息程序抽样提出问题

    Loose LIPS Sink Ships: Asking Questions in Battleship with Language-Informed Program Sampling

    [https://arxiv.org/abs/2402.19471](https://arxiv.org/abs/2402.19471)

    研究利用大型语言模型提出信息量丰富的问题，在Battleship游戏中展示出与人类表现相匹配的效果，并揭示了贝叶斯模型如何指导问问题行为。

    

    问题结合了我们对语言的掌握和我们对于在有限认知资源情况下推断不确定性的出色能力。人们如何在巨大假设空间中提出信息量丰富的问题？我们研究了这些在基于战舰游戏Battleship的经典提问任务中的权衡。我们的语言信息程序抽样（LIPS）模型利用大型语言模型（LLMs）生成自然语言问题，将其转化为符号程序，并评估其预期信息增益。我们发现，即使在一个令人惊讶的资源预算下，这种简单的蒙特卡罗优化策略也能产生反映人类在各种Battleship棋盘场景中表现的丰富问题。相比之下，仅使用LLM的基线在将问题与棋盘状态联系起来方面存在困难；值得注意的是，GPT-4V并没有比无视觉基线提供改进。我们的结果展示了贝叶斯提问模型如何可能模拟和指导人类的问问题行为。

    arXiv:2402.19471v1 Announce Type: cross  Abstract: Questions combine our mastery of language with our remarkable facility for reasoning about uncertainty. How do people navigate vast hypothesis spaces to pose informative questions given limited cognitive resources? We study these tradeoffs in a classic grounded question-asking task based on the board game Battleship. Our language-informed program sampling (LIPS) model uses large language models (LLMs) to generate natural language questions, translate them into symbolic programs, and evaluate their expected information gain. We find that with a surprisingly modest resource budget, this simple Monte Carlo optimization strategy yields informative questions that mirror human performance across varied Battleship board scenarios. In contrast, LLM-only baselines struggle to ground questions in the board state; notably, GPT-4V provides no improvement over non-visual baselines. Our results illustrate how Bayesian models of question-asking can l
    
[^4]: 改变了什么？将表征干预转化为自然语言

    What Changed? Converting Representational Interventions to Natural Language

    [https://arxiv.org/abs/2402.11355](https://arxiv.org/abs/2402.11355)

    将表征空间的反事实转化为自然语言，以分析和解释模型干预所引起的语言变化，并减轻分类中的偏见。

    

    针对语言模型（LMs）表征空间的干预方法已经被证明是影响模型行为的有效手段。这些方法被用来消除或改变模型表示中的人口统计信息（如性别）的编码，创建一个反事实的表示。然而，由于干预操作在表示空间内，准确理解它修改了哪些特征是一个挑战。我们展示了表征空间的反事实可以转化为自然语言的反事实。我们证明了这种方法使我们能够分析对应于给定表示空间干预的语言变化，并解释用于编码特定概念的特征。此外，由此产生的反事实可以用于减轻分类中的偏见。

    arXiv:2402.11355v1 Announce Type: new  Abstract: Interventions targeting the representation space of language models (LMs) have emerged as effective means to influence model behavior. These methods are employed, for example, to eliminate or alter the encoding of demographic information such as gender within the model's representations, creating a counterfactual representation. However, since the intervention operates within the representation space, understanding precisely which features it modifies poses a challenge. We show that representation-space counterfactuals can be converted into natural language counterfactuals. We demonstrate that this approach enables us to analyze the linguistic alterations corresponding to a given representation-space intervention and to interpret the features utilized for encoding a specific concept. Moreover, the resulting counterfactuals can be used to mitigate bias in classification.
    
[^5]: 将大型语言模型作为零-shot对话状态追踪器通过函数调用

    Large Language Models as Zero-shot Dialogue State Tracker through Function Calling

    [https://arxiv.org/abs/2402.10466](https://arxiv.org/abs/2402.10466)

    本研究提出了一种通过函数调用将大型语言模型用于零-shot对话状态追踪的新方法，能够在任务导向对话中取得出色的性能，适应不同领域而无需大量数据收集或模型调整。

    

    大型语言模型（LLMs）在会话系统中日益普遍，这是因为它们在一般情境中具有先进的理解和生成能力。然而，在需要不仅进行响应生成还需要在特定任务和领域内进行有效对话状态追踪（DST）的任务导向对话（TOD）中，它们的有效性仍不尽人意。在这项工作中，我们提出了一种通过函数调用解决LLMs中的DST的新方法FnCTOD。这种方法改进了零-shot DST，使其能够适应各种领域，而无需进行大量数据收集或模型调整。我们的实验结果表明，我们的方法在使用开源或专有LLMs时都取得了出色的性能：通过上下文提示，使得各种7B或13B参数模型超越了之前由ChatGPT实现的最新技术成果（SOTA）的水平，并提高了ChatGPT的性能，击败了

    arXiv:2402.10466v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly prevalent in conversational systems due to their advanced understanding and generative capabilities in general contexts. However, their effectiveness in task-oriented dialogues (TOD), which requires not only response generation but also effective dialogue state tracking (DST) within specific tasks and domains, remains less satisfying. In this work, we propose a novel approach FnCTOD for solving DST with LLMs through function calling. This method improves zero-shot DST, allowing adaptation to diverse domains without extensive data collection or model tuning. Our experimental results demonstrate that our approach achieves exceptional performance with both modestly sized open-source and also proprietary LLMs: with in-context prompting it enables various 7B or 13B parameter models to surpass the previous state-of-the-art (SOTA) achieved by ChatGPT, and improves ChatGPT's performance beating the
    
[^6]: CIC：一种面向文化感知图像字幕的框架

    CIC: A framework for Culturally-aware Image Captioning

    [https://arxiv.org/abs/2402.05374](https://arxiv.org/abs/2402.05374)

    CIC是一种面向文化感知图像字幕的框架，通过结合视觉问答和大型语言模型，它能够生成能描述图像中文化元素的详细字幕。

    

    图像字幕通过使用视觉-语言预训练模型（VLPs）如BLIP从图像生成描述性句子，这种方法已经取得了很大的改进。然而，当前的方法缺乏对图像中所描绘的文化元素（例如亚洲文化群体的传统服装）生成详细描述性字幕的能力。在本文中，我们提出了一种新的框架，\textbf{面向文化感知图像字幕（CIC）}，该框架能够从代表不同文化的图像中生成字幕并描述文化元素。受到将视觉模态和大型语言模型（LLMs）通过适当的提示进行组合的方法的启发，我们的框架（1）根据图像中的文化类别生成问题，（2）利用生成的问题从视觉问答（VQA）中提取文化视觉元素，（3）使用带有提示的LLMs生成文化感知字幕。我们在4个不同大学的45名参与者上进行了人工评估。

    Image Captioning generates descriptive sentences from images using Vision-Language Pre-trained models (VLPs) such as BLIP, which has improved greatly. However, current methods lack the generation of detailed descriptive captions for the cultural elements depicted in the images, such as the traditional clothing worn by people from Asian cultural groups. In this paper, we propose a new framework, \textbf{Culturally-aware Image Captioning (CIC)}, that generates captions and describes cultural elements extracted from cultural visual elements in images representing cultures. Inspired by methods combining visual modality and Large Language Models (LLMs) through appropriate prompts, our framework (1) generates questions based on cultural categories from images, (2) extracts cultural visual elements from Visual Question Answering (VQA) using generated questions, and (3) generates culturally-aware captions using LLMs with the prompts. Our human evaluation conducted on 45 participants from 4 dif
    
[^7]: 结构化概率编码

    Structured Probabilistic Coding

    [https://arxiv.org/abs/2312.13933](https://arxiv.org/abs/2312.13933)

    结构化概率编码（SPC）是一种新的监督式表示学习框架，通过编码和预测任务的信息来学习紧凑且信息丰富的表示，提高语言模型的泛化能力和语言理解能力，并通过结构化正则化实现更好的覆盖率。

    

    本论文提出了一种新的监督式表示学习框架，即结构化概率编码（SPC），用于从与目标任务相关的输入中学习紧凑和信息丰富的表示。SPC是一种仅有编码器的概率编码技术，具有来自目标空间的结构化正则化。它可以提高预训练语言模型的泛化能力，以实现更好的语言理解。具体而言，我们的概率编码在一个模块中同时进行信息编码和任务预测，以更充分地利用输入数据中的有效信息。它使用输出空间的变分推断来减少随机性和不确定性。此外，为了更好地控制概率表示的学习过程，在潜在空间中提出了结构化正则化，以促进类别之间的均匀性。通过正则化项，SPC可以保持潜在编码的高斯结构，并实现更好的覆盖率。

    This paper presents a new supervised representation learning framework, namely structured probabilistic coding (SPC), to learn compact and informative representations from input related to the target task. SPC is an encoder-only probabilistic coding technology with a structured regularization from the target space. It can enhance the generalization ability of pre-trained language models for better language understanding. Specifically, our probabilistic coding simultaneously performs information encoding and task prediction in one module to more fully utilize the effective information from input data. It uses variational inference in the output space to reduce randomness and uncertainty. Besides, to better control the learning process of probabilistic representations, a structured regularization is proposed to promote uniformity across classes in the latent space. With the regularization term, SPC can preserve the Gaussian structure of the latent code and achieve better coverage of the 
    
[^8]: 论语言模型的水印可学习性研究

    On the Learnability of Watermarks for Language Models. (arXiv:2312.04469v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2312.04469](http://arxiv.org/abs/2312.04469)

    该论文研究了语言模型水印的可学习性，提出了水印蒸馏方法，通过训练学生模型使其模仿使用解码水印的教师模型的行为。结果表明，语言模型具有直接学习生成水印的能力，这对于水印的实际应用具有重要影响。

    

    语言模型输出的水印可以实现对模型生成文本的统计检测，广泛应用于语言模型的负责任部署中。现有的水印策略通过改变现有语言模型的解码器来操作，而语言模型直接学习生成水印的能力将对水印在实际应用中产生重要影响。首先，学习得到的水印可以用于构建能自然生成带水印文本的开放模型，使开放模型也能从水印中受益。其次，如果水印用于确定生成文本的来源，攻击者可以通过伪造水印并生成有害的带水印文本来损害受害模型的声誉。为了研究水印的可学习性，我们提出了水印蒸馏方法，该方法通过训练学生模型使其行为类似于使用基于解码的水印的教师模型。我们在三个实验数据集上测试了我们的方法。

    Watermarking of language model outputs enables statistical detection of model-generated text, which has many applications in the responsible deployment of language models. Existing watermarking strategies operate by altering the decoder of an existing language model, and the ability for a language model to directly learn to generate the watermark would have significant implications for the real-world deployment of watermarks. First, learned watermarks could be used to build open models that naturally generate watermarked text, allowing for open models to benefit from watermarking. Second, if watermarking is used to determine the provenance of generated text, an adversary can hurt the reputation of a victim model by spoofing its watermark and generating damaging watermarked text. To investigate the learnability of watermarks, we propose watermark distillation, which trains a student model to behave like a teacher model that uses decoding-based watermarking. We test our approach on three
    
[^9]: 语言模型作为语义索引器

    Language Models As Semantic Indexers. (arXiv:2310.07815v1 [cs.IR])

    [http://arxiv.org/abs/2310.07815](http://arxiv.org/abs/2310.07815)

    本文介绍了一种使用生成性语言模型学习语义ID的自监督框架LMINDEXER。

    

    语义标识符（ID）是信息检索中的一个重要概念，旨在保留对象（如文档和项）内部的语义。先前的研究通常采用两阶段流程来学习语义ID，首先使用现成的文本编码器获取嵌入，并根据嵌入来推导ID。然而，每个步骤都会引入潜在的信息损失，并且文本编码器生成的潜在空间内的嵌入分布通常与语义索引所需的预期分布存在固有的不匹配。然而，设计一个既能学习文档的语义表示又能同时学习其分层结构的方法并不容易，因为语义ID是离散和顺序结构的，并且语义监督是不充分的。在本文中，我们引入了LMINDEXER，它是一个自监督框架，用于使用生成性语言模型学习语义ID。

    Semantic identifier (ID) is an important concept in information retrieval that aims to preserve the semantics of objects such as documents and items inside their IDs. Previous studies typically adopt a two-stage pipeline to learn semantic IDs by first procuring embeddings using off-the-shelf text encoders and then deriving IDs based on the embeddings. However, each step introduces potential information loss and there is usually an inherent mismatch between the distribution of embeddings within the latent space produced by text encoders and the anticipated distribution required for semantic indexing. Nevertheless, it is non-trivial to design a method that can learn the document's semantic representations and its hierarchical structure simultaneously, given that semantic IDs are discrete and sequentially structured, and the semantic supervision is deficient. In this paper, we introduce LMINDEXER, a self-supervised framework to learn semantic IDs with a generative language model. We tackl
    
[^10]: HANS，你聪明吗？神经系统的Clever Hans效应分析

    HANS, are you clever? Clever Hans Effect Analysis of Neural Systems. (arXiv:2309.12481v1 [cs.CL])

    [http://arxiv.org/abs/2309.12481](http://arxiv.org/abs/2309.12481)

    这项研究调查了指导调整的大型语言模型（It-LLMs）对多项选择题（MCQ）的鲁棒性能力，在选择顺序变动时揭示了选择偏见和推理能力的问题。

    

    指导调整的大型语言模型(It-LLMs)展示出了在认知状态、意图和反应方面推理的出色能力，可以让人们有效地引导和理解日常社交互动。事实上，已经提出了几个多项选择题(MCQ)基准来构建对模型能力的确切评估。然而，早期的研究表明It-LLMs中存在固有的“顺序偏见”，给适当的评估带来了挑战。本文通过使用四个MCQ基准对It-LLMs的抵抗能力进行了研究。通过引入对抗性示例，我们展示了显著的性能差距，特别是在选择顺序变动时，揭示了选择偏见并引发了对推理能力的讨论。通过第一位置和模型选择之间的相关性，我们假设在模型中存在结构启发式方法。

    Instruction-tuned Large Language Models (It-LLMs) have been exhibiting outstanding abilities to reason around cognitive states, intentions, and reactions of all people involved, letting humans guide and comprehend day-to-day social interactions effectively. In fact, several multiple-choice questions (MCQ) benchmarks have been proposed to construct solid assessments of the models' abilities. However, earlier works are demonstrating the presence of inherent "order bias" in It-LLMs, posing challenges to the appropriate evaluation. In this paper, we investigate It-LLMs' resilience abilities towards a series of probing tests using four MCQ benchmarks. Introducing adversarial examples, we show a significant performance gap, mainly when varying the order of the choices, which reveals a selection bias and brings into discussion reasoning abilities. Following a correlation between first positions and model choices due to positional bias, we hypothesized the presence of structural heuristics in 
    
[^11]: CPLLM: 基于大规模语言模型的临床预测

    CPLLM: Clinical Prediction with Large Language Models. (arXiv:2309.11295v1 [cs.CL])

    [http://arxiv.org/abs/2309.11295](http://arxiv.org/abs/2309.11295)

    CPLLM是一种使用大规模语言模型进行临床疾病预测的方法。通过量化和提示来微调语言模型，利用患者的历史诊断记录来预测目标疾病的诊断结果。实验证明，CPLLM在各项指标上均超越了其他基线模型，显示出显著的改进。

    

    我们提出了一种使用大规模语言模型 (LLM) 进行临床疾病预测的方法，该方法包括对预训练的语言模型进行微调。我们利用量化和提示来微调LLM，任务是预测患者在下一次就诊或随后的诊断中是否会被诊断为目标疾病，并利用他们的历史诊断记录。我们将结果与多个基线模型进行了比较，包括逻辑回归、RETAIN和Med-BERT，后者是使用结构化电子病历数据进行疾病预测的当前最先进模型。实验结果显示，CPLLM在PR-AUC和ROC-AUC指标上均超过了所有测试模型，相比基线模型显示出显著的改进。

    We present Clinical Prediction with Large Language Models (CPLLM), a method that involves fine-tuning a pre-trained Large Language Model (LLM) for clinical disease prediction. We utilized quantization and fine-tuned the LLM using prompts, with the task of predicting whether patients will be diagnosed with a target disease during their next visit or in the subsequent diagnosis, leveraging their historical diagnosis records. We compared our results versus various baselines, including Logistic Regression, RETAIN, and Med-BERT, which is the current state-of-the-art model for disease prediction using structured EHR data. Our experiments have shown that CPLLM surpasses all the tested models in terms of both PR-AUC and ROC-AUC metrics, displaying noteworthy enhancements compared to the baseline models.
    
[^12]: 大型语言模型能够准确预测搜索者的偏好

    Large language models can accurately predict searcher preferences. (arXiv:2309.10621v1 [cs.IR])

    [http://arxiv.org/abs/2309.10621](http://arxiv.org/abs/2309.10621)

    大型语言模型可以通过从真实用户那里获取高质量的第一方数据来准确预测搜索者的偏好。

    

    相关性标签是评估和优化搜索系统的关键。获取大量相关性标签通常需要第三方标注人员，但存在低质量数据的风险。本论文介绍了一种改进标签质量的替代方法，通过从真实用户那里获得仔细反馈来获取高质量的第一方数据。

    Relevance labels, which indicate whether a search result is valuable to a searcher, are key to evaluating and optimising search systems. The best way to capture the true preferences of users is to ask them for their careful feedback on which results would be useful, but this approach does not scale to produce a large number of labels. Getting relevance labels at scale is usually done with third-party labellers, who judge on behalf of the user, but there is a risk of low-quality data if the labeller doesn't understand user needs. To improve quality, one standard approach is to study real users through interviews, user studies and direct feedback, find areas where labels are systematically disagreeing with users, then educate labellers about user needs through judging guidelines, training and monitoring. This paper introduces an alternate approach for improving label quality. It takes careful feedback from real users, which by definition is the highest-quality first-party gold data that 
    
[^13]: LLM自卫：通过自检，LLMs意识到它们被愚弄了。

    LLM Self Defense: By Self Examination, LLMs Know They Are Being Tricked. (arXiv:2308.07308v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2308.07308](http://arxiv.org/abs/2308.07308)

    本文提出了一种通过自检来防御大型语言模型(LLMs)对抗性攻击的简单方法，即让模型自行过滤回应。实验结果表明，即使模型未对齐人类价值观，通过使用语言模型验证内容，仍然可以防止模型向用户呈现有害内容。

    

    近年来，大型语言模型（LLMs）由于其能够对人类提示做出高质量文本回应而变得非常受欢迎。然而，研究表明，这些模型在回应用户提示时可能生成有害内容（例如，给用户提供犯罪指导）。文献中已经着重研究如何通过方法（例如通过强化学习将模型与人类价值观对齐）来减轻这些风险。然而，研究发现，即使对齐的语言模型也容易受到绕过生成有害文本限制的对抗性攻击。我们提出了一种简单的方法来防御这些攻击，即大型语言模型对自己的回应进行过滤。我们目前的研究结果表明，即使模型没有被微调以与人类价值观对齐，也可以通过使用语言模型验证内容来防止其向用户呈现有害内容。

    Large language models (LLMs) have skyrocketed in popularity in recent years due to their ability to generate high-quality text in response to human prompting. However, these models have been shown to have the potential to generate harmful content in response to user prompting (e.g., giving users instructions on how to commit crimes). There has been a focus in the literature on mitigating these risks, through methods like aligning models with human values through reinforcement learning. However, it has been shown that even aligned language models are susceptible to adversarial attacks that bypass their restrictions on generating harmful text. We propose a simple approach to defending against these attacks by having a large language model filter its own responses. Our current results show that even if a model is not fine-tuned to be aligned with human values, it is possible to stop it from presenting harmful content to users by validating the content using a language model.
    
[^14]: ReactGenie：使用大型语言模型的面向对象状态抽象来支持复杂的多模态交互

    ReactGenie: An Object-Oriented State Abstraction for Complex Multimodal Interactions Using Large Language Models. (arXiv:2306.09649v1 [cs.HC])

    [http://arxiv.org/abs/2306.09649](http://arxiv.org/abs/2306.09649)

    ReactGenie是一个支持构建复杂的多模态移动应用程序的编程框架，通过使用共享的面向对象状态抽象，使得不同模态可以无缝集成和组合，从而实现了多模态交互的支持。

    

    多模态交互已被证明比传统的图形界面更加灵活、高效和适应各种用户和任务。然而，现有的多模态开发框架要么不能很好地处理多模态命令的复杂性和组合性，要么需要开发人员编写大量代码来支持这些多模态交互。本文提出了ReactGenie，这是一个编程框架，使用共享的面向对象状态抽象来支持构建复杂的多模态移动应用程序。具有不同模态的共享状态抽象使得使用ReactGenie的开发人员能够无缝地集成和组合这些模态以实现多模态交互。ReactGenie是构建图形应用程序的现有工作流程的自然扩展，就像使用React-Redux一样。开发人员只需要添加一些注释和示例来指示自然语言如何映射到用户可访问的状态即可。

    Multimodal interactions have been shown to be more flexible, efficient, and adaptable for diverse users and tasks than traditional graphical interfaces. However, existing multimodal development frameworks either do not handle the complexity and compositionality of multimodal commands well or require developers to write a substantial amount of code to support these multimodal interactions. In this paper, we present ReactGenie, a programming framework that uses a shared object-oriented state abstraction to support building complex multimodal mobile applications. Having different modalities share the same state abstraction allows developers using ReactGenie to seamlessly integrate and compose these modalities to deliver multimodal interaction.  ReactGenie is a natural extension to the existing workflow of building a graphical app, like the workflow with React-Redux. Developers only have to add a few annotations and examples to indicate how natural language is mapped to the user-accessible
    
[^15]: 论大型语言模型水印的可靠性

    On the Reliability of Watermarks for Large Language Models. (arXiv:2306.04634v1 [cs.LG])

    [http://arxiv.org/abs/2306.04634](http://arxiv.org/abs/2306.04634)

    本文研究了大型语言模型水印在混合其他文本来源时的可靠性，并提供了在实际应用中的建议。

    

    大型语言模型(LLMs)已经开始应用于日常使用，并有能力在未来的十年内产生大量的文本。机器生成的文本可能会取代互联网上的人类写作文本，并有可能被用于恶意目的，如钓鱼攻击和社交媒体机器人。水印是一种简单有效的策略，通过使LLM生成的文本可检测和可记录，来降低这些伤害。然而，一个关键问题仍然存在：在现实中混合了其他的文本来源，被人类写作者或其他语言模型改写，被用于社交和技术领域的各种应用时，水印在实际设置中的可靠性如何？在本文中，我们探讨了不同的检测方案，量化了它们检测水印的能力，并确定在每个情况下需要观察多少机器生成的文本才能可靠地检测水印。我们特别强调了当水印与其他文本来源混合时水印的可靠性，并提供了未来使用LLM生成的文本水印的建议。

    Large language models (LLMs) are now deployed to everyday use and positioned to produce large quantities of text in the coming decade. Machine-generated text may displace human-written text on the internet and has the potential to be used for malicious purposes, such as spearphishing attacks and social media bots. Watermarking is a simple and effective strategy for mitigating such harms by enabling the detection and documentation of LLM-generated text. Yet, a crucial question remains: How reliable is watermarking in realistic settings in the wild? There, watermarked text might be mixed with other text sources, paraphrased by human writers or other language models, and used for applications in a broad number of domains, both social and technical. In this paper, we explore different detection schemes, quantify their power at detecting watermarks, and determine how much machine-generated text needs to be observed in each scenario to reliably detect the watermark. We especially highlight o
    
[^16]: 大型语言模型的水印技术

    A Watermark for Large Language Models. (arXiv:2301.10226v3 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2301.10226](http://arxiv.org/abs/2301.10226)

    本文提出了一种在大型语言模型中实现水印技术的方法，该技术可以在不降低文本质量的前提下嵌入信号，且可以使用高效的开源算法进行检测，并且该技术十分鲁棒和安全。

    

    通过在生成的文本中嵌入信号，即将水印技术应用于模型输出，可以减轻大型语言模型潜在的危害。我们提出了一种专有语言模型的水印技术框架。水印可以嵌入到文本中，对文本质量的影响可以忽略不计，并且可以使用高效的开源算法在不访问语言模型API或参数的情况下进行检测。水印技术通过在生成单词之前选择一组随机的“绿色”标记，然后在抽样过程中软性地推广使用这些标记。我们提出了一个可解释的P值统计检验方法，用于检测水印技术， 并推导了一个信息论框架来分析水印技术的敏感性。我们使用Open Pretrained Transformer（OPT）家族的一个数十亿参数模型来测试水印技术，并讨论了其鲁棒性和安全性。

    Potential harms of large language models can be mitigated by watermarking model output, i.e., embedding signals into generated text that are invisible to humans but algorithmically detectable from a short span of tokens. We propose a watermarking framework for proprietary language models. The watermark can be embedded with negligible impact on text quality, and can be detected using an efficient open-source algorithm without access to the language model API or parameters. The watermark works by selecting a randomized set of "green" tokens before a word is generated, and then softly promoting use of green tokens during sampling. We propose a statistical test for detecting the watermark with interpretable p-values, and derive an information-theoretic framework for analyzing the sensitivity of the watermark. We test the watermark using a multi-billion parameter model from the Open Pretrained Transformer (OPT) family, and discuss robustness and security.
    

