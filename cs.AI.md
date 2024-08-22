# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AVicuna: Audio-Visual LLM with Interleaver and Context-Boundary Alignment for Temporal Referential Dialogue](https://arxiv.org/abs/2403.16276) | 介绍了一个新的框架AVicuna，生成了PU-VALOR数据集，解决了音频-视觉时间指代对话中的两个主要挑战：缺乏准确时间注释的数据集和整合复杂时间线索的方法。 |
| [^2] | [LLM^3:Large Language Model-based Task and Motion Planning with Motion Failure Reasoning](https://arxiv.org/abs/2403.11552) | LLM^3是一个基于大型语言模型的任务和运动规划框架，利用预训练的LLM具备强大的推理和规划能力，通过接口提出符号动作序列和选择连续动作参数进行运动规划，并通过运动规划的反馈来迭代优化提议，从而简化了处理领域特定消息的设计过程。 |
| [^3] | [PhD: A Prompted Visual Hallucination Evaluation Dataset](https://arxiv.org/abs/2403.11116) | 本研究针对Intrinsic Vision-Language Hallucination（IVL-Hallu）问题进行了深入分析，提出了几种新颖的IVL-Hallu任务，并将其分为四种类型，有助于揭示其产生的原因和反映。 |
| [^4] | [DiffuMatting: Synthesizing Arbitrary Objects with Matting-level Annotation](https://arxiv.org/abs/2403.06168) | 提出了DiffuMatting方法，通过扩散技术实现了“抠图任何物体”的能力，可以生成高度准确的注释，同时兼容社区LoRAs或各种条件控制方法。 |
| [^5] | [ComTraQ-MPC: Meta-Trained DQN-MPC Integration for Trajectory Tracking with Limited Active Localization Updates](https://arxiv.org/abs/2403.01564) | ComTraQ-MPC是一个结合了DQN和MPC的新框架，旨在优化在有限主动定位更新下的轨迹跟踪。 |
| [^6] | [Persona-DB: Efficient Large Language Model Personalization for Response Prediction with Collaborative Data Refinement](https://arxiv.org/abs/2402.11060) | 介绍了 Persona-DB，一个简单却有效的框架，通过层级构建过程和协同优化，改善了大规模语言模型个性化中数据库表示的泛化能力和检索效率。 |
| [^7] | [A Roadmap to Pluralistic Alignment](https://arxiv.org/abs/2402.05070) | 这篇论文提出了一条通向多元对齐的路线图，以解决设计AI系统能够服务于人们具有不同价值观和观点的需求。论文介绍了对齐定义和实现多元主义的三种方式，并提出了三种多元基准类别来评估和测试多元对齐的效果。 |
| [^8] | [Clarify: Improving Model Robustness With Natural Language Corrections](https://arxiv.org/abs/2402.03715) | 论文提出了Clarify，一种通过自然语言纠正模型错误概念的方法，该方法通过用户提供简短的文本描述来纠正模型的一致失败模式，从而提高模型的鲁棒性。 |
| [^9] | [Copilot Refinement: Addressing Code Smells in Copilot-Generated Python Code.](http://arxiv.org/abs/2401.14176) | 本研究旨在探索Copilot生成的Python代码中的代码异味，评估Copilot修复这些问题的能力。结果表明，有8种Python代码异味可以在Copilot生成的代码中检测到。 |
| [^10] | [Towards End-to-End GPS Localization with Neural Pseudorange Correction.](http://arxiv.org/abs/2401.10685) | 本论文提出了一个端到端的GPS定位框架E2E-PrNet，通过直接训练神经网络PrNet来进行伪距修正，实验结果表明其优于现有端到端GPS定位方法。 |
| [^11] | [Large Language Models in Mental Health Care: a Scoping Review.](http://arxiv.org/abs/2401.02984) | 本综述研究对大型语言模型在心理健康护理中的应用和结果进行了综合分析，发现其在诊断、治疗和患者参与增强等方面具有多样化的应用。同时，该研究还识别和讨论了在这些专业领域中所面临的挑战和限制。 |
| [^12] | [Learning Complete Topology-Aware Correlations Between Relations for Inductive Link Prediction.](http://arxiv.org/abs/2309.11528) | 本文提出了一种基于子图的方法TACO，用于建模高度与拓扑结构相关的关系之间的拓扑感知相关性，并展示了这种方法对于实体无关的归纳链接预测任务的潜力。 |
| [^13] | [SCALE: Scaling up the Complexity for Advanced Language Model Evaluation.](http://arxiv.org/abs/2306.09237) | 该论文提出了一个新颖的自然语言处理基准测试，挑战当前大型语言模型在处理长文档、利用领域专业知识、多语言理解和多任务处理方面的能力。基准测试包含瑞士法律系统的多样化法律NLP数据集，允许进行对底层非英语、固有多语言的法律系统进行全面研究。 |
| [^14] | [A Human Word Association based model for topic detection in social networks.](http://arxiv.org/abs/2301.13066) | 本文提出了一个基于人类词汇联想的社交网络主题检测框架，通过考虑语言结构并设计专门的抽取算法，在FA-CUP数据集上取得了比其他方法更好的性能。 |
| [^15] | [Predicting the Next Action by Modeling the Abstract Goal.](http://arxiv.org/abs/2209.05044) | 这篇论文提出了一种可以模型化抽象目标，以降低行动预测中不确定性的行动预测模型。使用视觉表征来描述动作和目标信息，并设计抽象目标为一个分布。该模型可在Epic-Kitchen数据集上实现最先进性能。 |

# 详细

[^1]: AVicuna：具有交错器和上下文边界对齐的音频-视觉LLM用于时间指代对话

    AVicuna: Audio-Visual LLM with Interleaver and Context-Boundary Alignment for Temporal Referential Dialogue

    [https://arxiv.org/abs/2403.16276](https://arxiv.org/abs/2403.16276)

    介绍了一个新的框架AVicuna，生成了PU-VALOR数据集，解决了音频-视觉时间指代对话中的两个主要挑战：缺乏准确时间注释的数据集和整合复杂时间线索的方法。

    

    在日常交流中，人类经常使用语音和手势来指代特定区域或对象，这个过程称为指代对话（RD）。尽管先前的研究已经通过大型语言模型（LLMs）或大型多模型模型（LMMs）在静态环境中调查了RD，但在音频-视觉媒体中探索时间指代对话（TRD）仍然有限。两个主要挑战阻碍了这一领域的进展：（1）缺乏具有精确时间注释的全面未修剪音频-视觉视频数据集，以及（2）需要有效整合复杂的时间听觉和视觉线索的方法。为了解决这些挑战，我们引入了一个新的框架，生成PU-VALOR，这是一个包含超过114,000个未修剪视频的广泛音频-视觉数据集，并介绍了AVicuna，具有音频-视觉令牌交错器（AVTI），确保了时间对齐。

    arXiv:2403.16276v1 Announce Type: cross  Abstract: In everyday communication, humans frequently use speech and gestures to refer to specific areas or objects, a process known as Referential Dialogue (RD). While prior studies have investigated RD through Large Language Models (LLMs) or Large Multimodal Models (LMMs) in static contexts, the exploration of Temporal Referential Dialogue (TRD) within audio-visual media remains limited. Two primary challenges hinder progress in this field: (1) the absence of comprehensive, untrimmed audio-visual video datasets with precise temporal annotations, and (2) the need for methods to integrate complex temporal auditory and visual cues effectively. To address these challenges, we introduce a novel framework to generate PU-VALOR, an extensive audio-visual dataset comprising over 114,000 untrimmed videos with accurate temporal demarcations. We also present AVicuna, featuring an Audio-Visual Tokens Interleaver (AVTI) that ensures the temporal alignment 
    
[^2]: LLM^3:基于大型语言模型的任务和运动规划以及运动失败推理

    LLM^3:Large Language Model-based Task and Motion Planning with Motion Failure Reasoning

    [https://arxiv.org/abs/2403.11552](https://arxiv.org/abs/2403.11552)

    LLM^3是一个基于大型语言模型的任务和运动规划框架，利用预训练的LLM具备强大的推理和规划能力，通过接口提出符号动作序列和选择连续动作参数进行运动规划，并通过运动规划的反馈来迭代优化提议，从而简化了处理领域特定消息的设计过程。

    

    传统任务和运动规划（TAMP）方法依赖于手工设计的界面，将符号任务规划与连续运动生成连接起来。这些特定领域的、劳动密集型的模块在处理现实世界设置中出现的新任务方面有限。在这里，我们提出了LLM^3，这是一个新颖的基于大型语言模型（LLM）的TAMP框架，具有领域无关的接口。具体来说，我们利用预训练的LLM的强大推理和规划能力来提出符号动作序列，并选择连续动作参数进行运动规划。关键是，LLM^3通过提示将运动规划反馈到其中，使得LLM能够通过对运动失败进行推理来迭代地优化其提议。因此，LLM^3在任务规划和运动规划之间建立接口，减轻了处理它们之间特定领域消息的复杂设计过程。通过一系列仿真

    arXiv:2403.11552v1 Announce Type: cross  Abstract: Conventional Task and Motion Planning (TAMP) approaches rely on manually crafted interfaces connecting symbolic task planning with continuous motion generation. These domain-specific and labor-intensive modules are limited in addressing emerging tasks in real-world settings. Here, we present LLM^3, a novel Large Language Model (LLM)-based TAMP framework featuring a domain-independent interface. Specifically, we leverage the powerful reasoning and planning capabilities of pre-trained LLMs to propose symbolic action sequences and select continuous action parameters for motion planning. Crucially, LLM^3 incorporates motion planning feed- back through prompting, allowing the LLM to iteratively refine its proposals by reasoning about motion failure. Consequently, LLM^3 interfaces between task planning and motion planning, alleviating the intricate design process of handling domain- specific messages between them. Through a series of simulat
    
[^3]: 博士论文：一个提示的视觉幻觉评估数据集

    PhD: A Prompted Visual Hallucination Evaluation Dataset

    [https://arxiv.org/abs/2403.11116](https://arxiv.org/abs/2403.11116)

    本研究针对Intrinsic Vision-Language Hallucination（IVL-Hallu）问题进行了深入分析，提出了几种新颖的IVL-Hallu任务，并将其分为四种类型，有助于揭示其产生的原因和反映。

    

    大型语言模型（LLMs）的快速增长推动了大型视觉语言模型（LVLMs）的发展。在LLMs中普遍存在的幻觉挑战也出现在LVLMs中。然而，大部分现有研究主要集中在LVLM中的对象幻觉上，忽略了LVLM幻觉的多样化类型。本研究深入探讨了固有视觉语言幻觉（IVL-Hallu）问题，对导致幻觉的不同类型的IVL-Hallu进行了彻底分析。具体来说，我们提出了几个新颖的IVL-Hallu任务，并将它们分为四种类型：（a）对象幻觉，由于对象的误识别而产生，（b）属性幻觉，由于属性的误识别而引起，（c）多模态冲突幻觉，源自文本和视觉信息之间的矛盾，以及（d）反常识幻觉，由于对立之间的矛盾。

    arXiv:2403.11116v1 Announce Type: cross  Abstract: The rapid growth of Large Language Models (LLMs) has driven the development of Large Vision-Language Models (LVLMs). The challenge of hallucination, prevalent in LLMs, also emerges in LVLMs. However, most existing efforts mainly focus on object hallucination in LVLM, ignoring diverse types of LVLM hallucinations. In this study, we delve into the Intrinsic Vision-Language Hallucination (IVL-Hallu) issue, thoroughly analyzing different types of IVL-Hallu on their causes and reflections. Specifically, we propose several novel IVL-Hallu tasks and categorize them into four types: (a) object hallucination, which arises from the misidentification of objects, (b) attribute hallucination, which is caused by the misidentification of attributes, (c) multi-modal conflicting hallucination, which derives from the contradictions between textual and visual information, and (d) counter-common-sense hallucination, which owes to the contradictions betwee
    
[^4]: DiffuMatting：使用Matting级别标注合成任意对象

    DiffuMatting: Synthesizing Arbitrary Objects with Matting-level Annotation

    [https://arxiv.org/abs/2403.06168](https://arxiv.org/abs/2403.06168)

    提出了DiffuMatting方法，通过扩散技术实现了“抠图任何物体”的能力，可以生成高度准确的注释，同时兼容社区LoRAs或各种条件控制方法。

    

    由于获取高度准确或抠图注释的困难和劳动密集性，公开可用的高度准确标签数量有限。为了解决这一挑战，我们提出了一种DiffuMatting，它继承了扩散的强大生成能力，并赋予了“抠图任何物体”的能力。我们的DiffuMatting可以：1）作为一个具有高准确度注释的任意抠图工厂；2）与社区LoRAs或各种条件控制方法兼容，以实现社区友好的艺术设计和可控生成。具体地，受绿幕抠像的启发，我们旨在教授扩散模型在固定的绿幕画布上绘画。为此，收集了一个大规模的绿幕数据集（Green100K）作为DiffuMatting的训练数据集。其次，提出了一个绿色背景控制损失，以保持画布为纯绿色以进行区分。

    arXiv:2403.06168v1 Announce Type: cross  Abstract: Due to the difficulty and labor-consuming nature of getting highly accurate or matting annotations, there only exists a limited amount of highly accurate labels available to the public. To tackle this challenge, we propose a DiffuMatting which inherits the strong Everything generation ability of diffusion and endows the power of "matting anything". Our DiffuMatting can 1). act as an anything matting factory with high accurate annotations 2). be well-compatible with community LoRAs or various conditional control approaches to achieve the community-friendly art design and controllable generation. Specifically, inspired by green-screen-matting, we aim to teach the diffusion model to paint on a fixed green screen canvas. To this end, a large-scale greenscreen dataset (Green100K) is collected as a training dataset for DiffuMatting. Secondly, a green background control loss is proposed to keep the drawing board as a pure green color to disti
    
[^5]: ComTraQ-MPC：元训练的DQN-MPC集成用于具有有限主动定位更新的轨迹跟踪

    ComTraQ-MPC: Meta-Trained DQN-MPC Integration for Trajectory Tracking with Limited Active Localization Updates

    [https://arxiv.org/abs/2403.01564](https://arxiv.org/abs/2403.01564)

    ComTraQ-MPC是一个结合了DQN和MPC的新框架，旨在优化在有限主动定位更新下的轨迹跟踪。

    

    在局部可观察、随机环境中进行轨迹跟踪的最佳决策往往面临着一个重要挑战，即主动定位更新数量有限，这是指代理从传感器获取真实状态信息的过程。传统方法往往难以平衡资源保存、准确状态估计和精确跟踪之间的关系，导致性能次优。本文介绍了ComTraQ-MPC，这是一个结合了Deep Q-Networks (DQN)和模型预测控制(MPC)的新颖框架，旨在优化有限主动定位更新下的轨迹跟踪。元训练的DQN确保了自适应主动定位调度，同时

    arXiv:2403.01564v1 Announce Type: cross  Abstract: Optimal decision-making for trajectory tracking in partially observable, stochastic environments where the number of active localization updates -- the process by which the agent obtains its true state information from the sensors -- are limited, presents a significant challenge. Traditional methods often struggle to balance resource conservation, accurate state estimation and precise tracking, resulting in suboptimal performance. This problem is particularly pronounced in environments with large action spaces, where the need for frequent, accurate state data is paramount, yet the capacity for active localization updates is restricted by external limitations. This paper introduces ComTraQ-MPC, a novel framework that combines Deep Q-Networks (DQN) and Model Predictive Control (MPC) to optimize trajectory tracking with constrained active localization updates. The meta-trained DQN ensures adaptive active localization scheduling, while the
    
[^6]: Persona-DB：用于响应预测的高效大规模语言模型个性化与协同数据优化

    Persona-DB: Efficient Large Language Model Personalization for Response Prediction with Collaborative Data Refinement

    [https://arxiv.org/abs/2402.11060](https://arxiv.org/abs/2402.11060)

    介绍了 Persona-DB，一个简单却有效的框架，通过层级构建过程和协同优化，改善了大规模语言模型个性化中数据库表示的泛化能力和检索效率。

    

    随着对大型语言模型（LLMs）个性化交互需求的增加，需要开发能够准确快速识别用户意见和偏好的方法。检索增强作为一种有效策略出现，因为它可以适应大量用户而无需进行微调的成本。然而，现有研究主要集中在增强检索阶段，并对数据库表示的优化进行了有限的探索，这是个性化等任务的关键方面。在这项工作中，我们从一个新的角度研究了这个问题，着重于如何更有效地表示数据，以便在LLM定制的情境下更有效地进行检索。为了解决这一挑战，我们介绍了Persona-DB，这是一个简单而有效的框架，包括一个分层构建过程，以改善跨任务背景的泛化能力，并进行协同优化。

    arXiv:2402.11060v1 Announce Type: cross  Abstract: The increasing demand for personalized interactions with large language models (LLMs) calls for the development of methodologies capable of accurately and efficiently identifying user opinions and preferences. Retrieval augmentation emerges as an effective strategy, as it can accommodate a vast number of users without the costs from fine-tuning. Existing research, however, has largely focused on enhancing the retrieval stage and devoted limited exploration toward optimizing the representation of the database, a crucial aspect for tasks such as personalization. In this work, we examine the problem from a novel angle, focusing on how data can be better represented for more efficient retrieval in the context of LLM customization. To tackle this challenge, we introduce Persona-DB, a simple yet effective framework consisting of a hierarchical construction process to improve generalization across task contexts and collaborative refinement to
    
[^7]: 通往多元对齐的路线图

    A Roadmap to Pluralistic Alignment

    [https://arxiv.org/abs/2402.05070](https://arxiv.org/abs/2402.05070)

    这篇论文提出了一条通向多元对齐的路线图，以解决设计AI系统能够服务于人们具有不同价值观和观点的需求。论文介绍了对齐定义和实现多元主义的三种方式，并提出了三种多元基准类别来评估和测试多元对齐的效果。

    

    随着人工智能系统的权力和普及程度的增加，设计能够为不同价值观和观点的人服务的人工智能系统变得愈发重要。然而，将模型对齐以服务多元人类价值观仍然是一个待解决的研究问题。在本文中，我们提出了一条通向多元对齐的路线图，具体使用语言模型作为测试平台。我们确定和形式化了三种可能的方式来定义和实现人工智能系统中的多元主义：1）Overton多元模型，展示合理反应的光谱；2）可操控的多元模型，可以调整以反映特定的观点；3）分布多元模型，在分布中很好地校准给定人群的模型。我们还提出和形式化了三种可能的多元基准类别：1）多目标基准；2）权衡可操控基准，鼓励模型对任意权衡进行调整；3）陪审团多元基准，明确地模拟了不同陪审团的意见。

    With increased power and prevalence of AI systems, it is ever more critical that AI systems are designed to serve all, i.e., people with diverse values and perspectives. However, aligning models to serve pluralistic human values remains an open research question. In this piece, we propose a roadmap to pluralistic alignment, specifically using language models as a test bed. We identify and formalize three possible ways to define and operationalize pluralism in AI systems: 1) Overton pluralistic models that present a spectrum of reasonable responses; 2) Steerably pluralistic models that can steer to reflect certain perspectives; and 3) Distributionally pluralistic models that are well-calibrated to a given population in distribution. We also propose and formalize three possible classes of pluralistic benchmarks: 1) Multi-objective benchmarks, 2) Trade-off steerable benchmarks, which incentivize models to steer to arbitrary trade-offs, and 3) Jury-pluralistic benchmarks which explicitly m
    
[^8]: 澄清：通过自然语言纠正提高模型的鲁棒性

    Clarify: Improving Model Robustness With Natural Language Corrections

    [https://arxiv.org/abs/2402.03715](https://arxiv.org/abs/2402.03715)

    论文提出了Clarify，一种通过自然语言纠正模型错误概念的方法，该方法通过用户提供简短的文本描述来纠正模型的一致失败模式，从而提高模型的鲁棒性。

    

    在监督学习中，模型被训练从静态数据集中提取相关性。这通常会导致模型依赖于高级错误概念。为了防止这种错误概念，我们必须提供额外的信息。现有的方法包括一些额外的实例级监督形式，例如标记虚假特征或来自平衡分布的额外标记数据。对于大规模数据集来说，这些策略可能会变得昂贵，因为它们需要以接近原始训练数据的规模进行额外注释。我们假设有针对性的关于模型错误概念的自然语言反馈是一种更有效的额外监督形式。我们引入了Clarify，一种新型界面和方法来交互式地纠正模型的错误概念。通过Clarify，用户只需要提供一个简短的文本描述来描述模型的一致性失败模式。然后，我们完全自动化地使用s

    In supervised learning, models are trained to extract correlations from a static dataset. This often leads to models that rely on high-level misconceptions. To prevent such misconceptions, we must necessarily provide additional information beyond the training data. Existing methods incorporate forms of additional instance-level supervision, such as labels for spurious features or additional labeled data from a balanced distribution. Such strategies can become prohibitively costly for large-scale datasets since they require additional annotation at a scale close to the original training data. We hypothesize that targeted natural language feedback about a model's misconceptions is a more efficient form of additional supervision. We introduce Clarify, a novel interface and method for interactively correcting model misconceptions. Through Clarify, users need only provide a short text description to describe a model's consistent failure patterns. Then, in an entirely automated way, we use s
    
[^9]: Copilot细化：解决Copilot生成的Python代码中的代码异味

    Copilot Refinement: Addressing Code Smells in Copilot-Generated Python Code. (arXiv:2401.14176v1 [cs.SE])

    [http://arxiv.org/abs/2401.14176](http://arxiv.org/abs/2401.14176)

    本研究旨在探索Copilot生成的Python代码中的代码异味，评估Copilot修复这些问题的能力。结果表明，有8种Python代码异味可以在Copilot生成的代码中检测到。

    

    作为最流行的动态语言之一，Python在存在代码异味时可读性和可维护性会下降。大型语言模型的最新进展引发了对AI支持的代码生成和重构工具的日益关注。GitHub Copilot是其中一种被广泛使用的工具。Copilot Chat是在2023年9月发布的一种交互式工具，旨在为自然语言驱动的编码提供便利。然而，对于理解Copilot生成的Python代码中的代码异味以及Copilot修复其生成的代码异味的能力，人们并没有给予足够的关注。为此，我们构建了一个包含102个Copilot生成的Python代码中的代码异味的数据集。我们的目标是首先探索Copilot生成的Python代码中代码异味的发生情况，然后评估Copilot在使用不同提示修复这些代码异味时的有效性。结果显示，10种Python代码异味中有8种可以在Copilot生成的代码中检测到。

    As one of the most popular dynamic languages, Python experiences a decrease in readability and maintainability when code smells are present. Recent advancements in Large Language Models have sparked growing interest in AI-enabled tools for both code generation and refactoring. GitHub Copilot is one such tool that has gained widespread usage. Copilot Chat, released on September 2023, functions as an interactive tool aims at facilitating natural language-powered coding. However, limited attention has been given to understanding code smells in Copilot-generated Python code and Copilot's ability to fix the code smells it generates. To this end, we built a dataset comprising 102 code smells in Copilot-generated Python code. Our aim is to first explore the occurrence of code smells in Copilot-generated Python code and then evaluate the effectiveness of Copilot in fixing these code smells employing different prompts. The results show that 8 out of 10 types of Python smells can be detected in 
    
[^10]: 用神经假伪距修正实现端到端GPS定位

    Towards End-to-End GPS Localization with Neural Pseudorange Correction. (arXiv:2401.10685v1 [cs.LG])

    [http://arxiv.org/abs/2401.10685](http://arxiv.org/abs/2401.10685)

    本论文提出了一个端到端的GPS定位框架E2E-PrNet，通过直接训练神经网络PrNet来进行伪距修正，实验结果表明其优于现有端到端GPS定位方法。

    

    伪距误差是GPS定位不准确的根本原因。以往的数据驱动方法使用手工制作的中间标签进行伪距误差回归和消除。与之不同的是，我们提出了一个端到端的GPS定位框架E2E-PrNet，通过使用GPS接收机状态的真实值计算最终任务损失，直接训练一个用于伪距修正的神经网络PrNet。损失对可学习参数的梯度通过可微非线性最小二乘优化器反向传播到PrNet。通过使用Android手机收集的GPS数据进行验证，结果显示E2E-PrNet优于最先进的端到端GPS定位方法。

    Pseudorange errors are the root cause of localization inaccuracy in GPS. Previous data-driven methods regress and eliminate pseudorange errors using handcrafted intermediate labels. Unlike them, we propose an end-to-end GPS localization framework, E2E-PrNet, to train a neural network for pseudorange correction (PrNet) directly using the final task loss calculated with the ground truth of GPS receiver states. The gradients of the loss with respect to learnable parameters are backpropagated through a differentiable nonlinear least squares optimizer to PrNet. The feasibility is verified with GPS data collected by Android phones, showing that E2E-PrNet outperforms the state-of-the-art end-to-end GPS localization methods.
    
[^11]: 大型语言模型在心理健康护理中的应用：一项综述研究

    Large Language Models in Mental Health Care: a Scoping Review. (arXiv:2401.02984v1 [cs.CL])

    [http://arxiv.org/abs/2401.02984](http://arxiv.org/abs/2401.02984)

    本综述研究对大型语言模型在心理健康护理中的应用和结果进行了综合分析，发现其在诊断、治疗和患者参与增强等方面具有多样化的应用。同时，该研究还识别和讨论了在这些专业领域中所面临的挑战和限制。

    

    目的：大型语言模型（LLM）的使用越来越广泛，需要对它们在心理健康护理领域的应用和结果进行全面的综述。本综述研究旨在对LLMs在心理健康护理中的现有发展和应用进行批判性分析，突出它们的成功，并识别这些专业领域中的挑战和限制。材料和方法：2023年11月，在PubMed、Web of Science、Google Scholar、arXiv、medRxiv和PsyArXiv六个数据库中进行了广泛的文献搜索，遵循2020年版的“系统评价和Meta分析的首选报告项目”（PRISMA）指南。最初识别了313篇出版物，按照研究纳入标准，最终选择了34篇出版物进行综述。结果：我们发现了LLMs在心理健康护理中的多种应用，包括诊断、治疗、患者参与增强等。关键挑战和限制方面的发现将被总结和讨论。

    Objective: The growing use of large language models (LLMs) stimulates a need for a comprehensive review of their applications and outcomes in mental health care contexts. This scoping review aims to critically analyze the existing development and applications of LLMs in mental health care, highlighting their successes and identifying their challenges and limitations in these specialized fields. Materials and Methods: A broad literature search was conducted in November 2023 using six databases (PubMed, Web of Science, Google Scholar, arXiv, medRxiv, and PsyArXiv) following the 2020 version of the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) guidelines. A total of 313 publications were initially identified, and after applying the study inclusion criteria, 34 publications were selected for the final review. Results: We identified diverse applications of LLMs in mental health care, including diagnosis, therapy, patient engagement enhancement, etc. Key challen
    
[^12]: 学习关系之间的完整拓扑感知相关性以进行归纳链接预测

    Learning Complete Topology-Aware Correlations Between Relations for Inductive Link Prediction. (arXiv:2309.11528v1 [cs.AI])

    [http://arxiv.org/abs/2309.11528](http://arxiv.org/abs/2309.11528)

    本文提出了一种基于子图的方法TACO，用于建模高度与拓扑结构相关的关系之间的拓扑感知相关性，并展示了这种方法对于实体无关的归纳链接预测任务的潜力。

    

    归纳链接预测——在训练和推理阶段实体可能不同——已经显示出了以实体无关的方式完成演化知识图谱的巨大潜力。许多流行的方法主要关注建模图级特征，而边级交互——尤其是关系之间的语义相关性——则被较少探索。然而，我们注意到语义相关性之间的一个理想特性是它们在本质上是边级和实体无关的。这意味着语义相关性对于实体无关的归纳链接预测任务具有巨大的潜力。受到这一观察的启发，我们提出了一种新颖的基于子图的方法，即TACO，来建模与其子图内的拓扑结构高度相关的关系之间的拓扑感知相关性。

    Inductive link prediction -- where entities during training and inference stages can be different -- has shown great potential for completing evolving knowledge graphs in an entity-independent manner. Many popular methods mainly focus on modeling graph-level features, while the edge-level interactions -especially the semantic correlations between relations -- have been less explored. However, we notice a desirable property of semantic correlations between relations is that they are inherently edge-level and entity-independent. This implies the great potential of the semantic correlations for the entity-independent inductive link prediction task. Inspired by this observation, we propose a novel subgraph-based method, namely TACO, to model Topology-Aware COrrelations between relations that are highly correlated to their topological structures within subgraphs. Specifically, we prove that semantic correlations between any two relations can be categorized into seven topological patterns,
    
[^13]: SCALE: 提升高级语言模型评估的复杂性

    SCALE: Scaling up the Complexity for Advanced Language Model Evaluation. (arXiv:2306.09237v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2306.09237](http://arxiv.org/abs/2306.09237)

    该论文提出了一个新颖的自然语言处理基准测试，挑战当前大型语言模型在处理长文档、利用领域专业知识、多语言理解和多任务处理方面的能力。基准测试包含瑞士法律系统的多样化法律NLP数据集，允许进行对底层非英语、固有多语言的法律系统进行全面研究。

    

    最近在大型语言模型（LLM）方面取得的进展已经饱和了许多自然语言处理基准测试（包括专业领域的基准测试），强调了需要新颖、更具挑战性的测试来正确评估LLM的能力。在本文中，我们引入了一个新颖的自然语言处理基准测试，对当前LLM的四个关键方面提出了挑战：处理长文档（多达50K个标记）、利用领域专业知识（体现在法律文本中）、多语言理解（涵盖五种语言）和多任务处理（包括法律文件到文件信息检索、法庭视图生成、重要决策摘要、引用提取和八个具有挑战性的文本分类任务）。我们的基准测试包含了来自瑞士法律系统的多样的法律NLP数据集，可以对底层非英语、固有多语言的联邦法律系统进行全面研究。尽管最近取得了进展，但对于强烈的审查/分析任务，高效地处理长文档仍然是一个挑战。

    Recent strides in Large Language Models (LLMs) have saturated many NLP benchmarks (even professional domain-specific ones), emphasizing the need for novel, more challenging novel ones to properly assess LLM capabilities. In this paper, we introduce a novel NLP benchmark that poses challenges to current LLMs across four key dimensions: processing long documents (up to 50K tokens), utilizing domain specific knowledge (embodied in legal texts), multilingual understanding (covering five languages), and multitasking (comprising legal document to document Information Retrieval, Court View Generation, Leading Decision Summarization, Citation Extraction, and eight challenging Text Classification tasks). Our benchmark comprises diverse legal NLP datasets from the Swiss legal system, allowing for a comprehensive study of the underlying Non-English, inherently multilingual, federal legal system. Despite recent advances, efficiently processing long documents for intense review/analysis tasks remai
    
[^14]: 基于人类词汇联想的社交网络主题检测模型

    A Human Word Association based model for topic detection in social networks. (arXiv:2301.13066v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2301.13066](http://arxiv.org/abs/2301.13066)

    本文提出了一个基于人类词汇联想的社交网络主题检测框架，通过考虑语言结构并设计专门的抽取算法，在FA-CUP数据集上取得了比其他方法更好的性能。

    

    随着社交网络的广泛使用，检测这些网络中讨论的主题已成为一个重要的挑战。目前的工作主要基于频繁模式挖掘或语义关系，而没有考虑语言结构。语言结构方法的意义在于发现词语之间的关系以及人类如何理解它们。因此，本文利用词汇联想的心理能力模拟概念，提出了一种基于人类词汇联想的社交网络主题检测框架。该框架基于人类词汇联想方法，并设计了专门的抽取算法。该方法在FA-CUP数据集上进行了评估，该数据集是主题检测领域的基准数据集。结果表明，与其他方法相比，所提出的方法在主题召回率和关键词F1值上有较好的改进。此外，主题检测领域中的大多数先前工作主要基于模式挖掘或语义关系。

    With the widespread use of social networks, detecting the topics discussed in these networks has become a significant challenge. The current works are mainly based on frequent pattern mining or semantic relations, and the language structure is not considered. The meaning of language structural methods is to discover the relationship between words and how humans understand them. Therefore, this paper uses the Concept of the Imitation of the Mental Ability of Word Association to propose a topic detection framework in social networks. This framework is based on the Human Word Association method. A special extraction algorithm has also been designed for this purpose. The performance of this method is evaluated on the FA-CUP dataset. It is a benchmark dataset in the field of topic detection. The results show that the proposed method is a good improvement compared to other methods, based on the Topic-recall and the keyword F1 measure. Also, most of the previous works in the field of topic de
    
[^15]: 建模抽象目标预测下一步动作

    Predicting the Next Action by Modeling the Abstract Goal. (arXiv:2209.05044v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2209.05044](http://arxiv.org/abs/2209.05044)

    这篇论文提出了一种可以模型化抽象目标，以降低行动预测中不确定性的行动预测模型。使用视觉表征来描述动作和目标信息，并设计抽象目标为一个分布。该模型可在Epic-Kitchen数据集上实现最先进性能。

    

    预测人类动作的问题具有固有的不确定性，但是，如果我们有关于动作实现目标的感知，可以降低这种不确定性。本文提出了一种行动预测模型，利用目标信息来减少未来预测中的不确定性。通过视觉表征，我们描述了动作和目标的信息。通过此方法，我们得出了一个称为抽象目标的新概念，其取决于观察到的视觉特征序列，用于行动预测。我们将抽象目标设计为一个分布，其参数是使用变分递归网络估计的。我们对下一个动作进行多次采样，并引入目标一致性度量来确定从抽象目标得出的最佳候选动作。我们的方法在极具挑战性的Epic-Kitchen数据集上取得了令人印象深刻的结果，并实现了最先进的性能。

    The problem of anticipating human actions is an inherently uncertain one. However, we can reduce this uncertainty if we have a sense of the goal that the actor is trying to achieve. Here, we present an action anticipation model that leverages goal information for the purpose of reducing the uncertainty in future predictions. Since we do not possess goal information or the observed actions during inference, we resort to visual representation to encapsulate information about both actions and goals. Through this, we derive a novel concept called abstract goal which is conditioned on observed sequences of visual features for action anticipation. We design the abstract goal as a distribution whose parameters are estimated using a variational recurrent network. We sample multiple candidates for the next action and introduce a goal consistency measure to determine the best candidate that follows from the abstract goal. Our method obtains impressive results on the very challenging Epic-Kitchen
    

