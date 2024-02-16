# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Self-Play Fine-Tuning of Diffusion Models for Text-to-Image Generation](https://arxiv.org/abs/2402.10210) | 本文介绍了一种创新的技术，称为自我对抗微调扩散模型（SPIN-Diffusion），通过扩散模型与其先前版本的竞争，实现了逐步自我改进过程。 |
| [^2] | [Recovering the Pre-Fine-Tuning Weights of Generative Models](https://arxiv.org/abs/2402.10208) | 该论文提出了一种恢复生成模型预微调权重的方法，通过少量低秩微调模型可以恢复准确的预微调权重，利用这个新漏洞攻击大规模模型。 |
| [^3] | [Rewards-in-Context: Multi-objective Alignment of Foundation Models with Dynamic Preference Adjustment](https://arxiv.org/abs/2402.10207) | 本文介绍了Rewards-in-Context（RiC）方法，该方法通过多个奖励条件控制基础模型的响应，并应用有监督的微调进行对齐。它具有简单性和适应性，并支持在推理时动态调整用户偏好。 |
| [^4] | [Chain-of-Thought Reasoning Without Prompting](https://arxiv.org/abs/2402.10200) | 本研究发现，通过改变解码过程而不是使用提示工程，可以从预训练的大型语言模型中引出思维链推理路径，这种方法绕过了提示的限制，并且可以评估模型的内在推理能力。 |
| [^5] | [A Trembling House of Cards? Mapping Adversarial Attacks against Language Agents](https://arxiv.org/abs/2402.10196) | 本文是第一个对语言代理的敌对攻击进行映射的系统化努力。它提供了一个统一的概念框架来研究这些攻击。这有助于我们理解语言代理的安全风险。 |
| [^6] | [BitDelta: Your Fine-Tune May Only Be Worth One Bit](https://arxiv.org/abs/2402.10193) | BitDelta研究探讨了大型语言模型在微调过程中的信息冗余性，并提出了一种名为BitDelta的方法，可以将微调过程中添加的信息量化为一个比特，同时保持性能。这一发现对于多租户模型的服务和存储有重要意义，并可以显著降低GPU内存需求。 |
| [^7] | [Uncertainty Decomposition and Quantification for In-Context Learning of Large Language Models](https://arxiv.org/abs/2402.10189) | 本文研究了大型语言模型（LLM）上下文学习中的不确定性，并提出了一种新的方法来量化这种不确定性，包括演示产生的不确定性和模型配置的模糊性。 |
| [^8] | [Rethinking Information Structures in RLHF: Reward Generalization from a Graph Theory Perspective](https://arxiv.org/abs/2402.10184) | 本研究通过设计奖励建模过程中的数据集信息结构，从图论的视角提出了RLHF中奖励泛化的问题，以解决多样的环境、低成本标注和可靠的对齐性能间的不兼容性。 |
| [^9] | [TDAG: A Multi-Agent Framework based on Dynamic Task Decomposition and Agent Generation](https://arxiv.org/abs/2402.10178) | 提出了一种基于动态任务分解和代理生成的多代理框架(TDAG)，该框架能够在解决复杂的实际问题时提高代理的适应性。同时，引入了ItineraryBench，这是一个在旅行规划中具有精细评估系统的基准测试。 |
| [^10] | [OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset](https://arxiv.org/abs/2402.10176) | OpenMathInstruct-1是一个包含180万个数学问题和解决方法对的数据集，通过合成开源LLM的代码解释器解决方案来构建，填补了目前开源LLM在数学技能方面与闭源LLM之间的差距。 |
| [^11] | [Unlocking Structure Measuring: Introducing PDD, an Automatic Metric for Positional Discourse Coherence](https://arxiv.org/abs/2402.10175) | 本论文引入了一种新的自动评估指标PDD，用于评估长篇文章之间的话语连贯性，实验证明该指标更接近人类偏好和GPT-4的评估结果。 |
| [^12] | [Data Engineering for Scaling Language Models to 128K Context](https://arxiv.org/abs/2402.10171) | 本论文研究了将语言模型的上下文长度扩展到128K的连续预训练方法，通过适当的数据混合和轻量级的预训练可以实现，其中关键要点在于数据的数量和质量，需要注意领域平衡和长度上采样。 |
| [^13] | [Knowledge-Infused LLM-Powered Conversational Health Agent: A Case Study for Diabetes Patients](https://arxiv.org/abs/2402.10153) | 本文提出了一种知识注入的基于LLM的对话式健康代理（CHA）用于糖尿病患者，通过整合领域特定知识和分析能力，提高了糖尿病管理的准确性和效果。 |
| [^14] | [ControlLM: Crafting Diverse Personalities for Language Models](https://arxiv.org/abs/2402.10151) | ControlLM是一种可以在推理时调整语言模型个性特征的方法，通过差异的行为提示激活模式来实现，从而实现模型行为的精确、实时调整。 |
| [^15] | [TOAD: Task-Oriented Automatic Dialogs with Diverse Response Styles](https://arxiv.org/abs/2402.10137) | TOAD是一个具有多样响应风格的面向任务的自动对话系统，其中考虑了冗长程度和用户表达镜像两个方面。TOAD通过模拟真实的应用上下文交互，提供了丰富的系统响应风格选项，并在评估中表明建模更冗长的回复或不进行用户表达镜像的回复更具挑战性。 |
| [^16] | [Selective Reflection-Tuning: Student-Selected Data Recycling for LLM Instruction-Tuning](https://arxiv.org/abs/2402.10110) | 本文介绍了一种名为选择性反射调节的新方法，该方法通过教师LLM的反射和自省与学生LLM的数据选择能力相结合，自动优化现有的指令调节数据，从而实现了高效的指令调节和卓越性能的LLM。 |
| [^17] | [Towards Reducing Diagnostic Errors with Interpretable Risk Prediction](https://arxiv.org/abs/2402.10109) | 本研究提出了一种使用LLMs方法来识别病人电子病历数据中指示特定诊断风险增加或减少的证据的方法，旨在通过增加证据的获取与减少诊断错误来降低诊断错误。模型使用神经加性模型进行预测，以证据为后盾，并给出个体化风险估计，特别针对诊断延迟和来自不完整鉴别的错误进行优化。 |
| [^18] | [Quantized Embedding Vectors for Controllable Diffusion Language Models](https://arxiv.org/abs/2402.10107) | 本论文提出了一种量化嵌入可控扩散语言模型（QE-CDLM），通过重建任务特定的嵌入空间来改善扩散语言模型的可控性、可移植性和稳定性。 |
| [^19] | [GeoEval: Benchmark for Evaluating LLMs and Multi-Modal Models on Geometry Problem-Solving](https://arxiv.org/abs/2402.10104) | GeoEval基准测试用于评估LLMs和MMs在几何问题解决上的性能，发现WizardMath模型在主要子集上表现出色，但在具有挑战性的子集上准确率较低。 |
| [^20] | [Both Matter: Enhancing the Emotional Intelligence of Large Language Models without Compromising the General Intelligence](https://arxiv.org/abs/2402.10073) | 本论文提出了一个称为\textbf{MoEI}的方法，通过引入一个大规模的EI相关任务集合\textsc{EiBench}并采用模块化的训练过程和情感智力增强器的集成，综合提高了LLM的情感智力（EI）而不损害其普适智能（GI）。 |
| [^21] | [Towards Safer Large Language Models through Machine Unlearning](https://arxiv.org/abs/2402.10058) | 这项研究提出了一种新的大型语言模型遗忘框架SKU，旨在消除有害知识的同时保留模型对正常提示的实用性。 |
| [^22] | [Unmemorization in Large Language Models via Self-Distillation and Deliberate Imagination](https://arxiv.org/abs/2402.10052) | 本研究提出了一种新颖的大型语言模型遗忘方法，通过自我蒸馏和有意识的想象，有效地遗忘目标文本，并在生成任务和自然语言理解任务中保留模型的能力。 |
| [^23] | [SwissNYF: Tool Grounded LLM Agents for Black Box Setting](https://arxiv.org/abs/2402.10051) | 该论文提出了一种基于黑盒环境的基于LLM的工具生成智能体的方法，在复杂的API调用中表现出了优越的性能，可以应对具有不可逆性和大量时间消耗的任务。 |
| [^24] | [RS-DPO: A Hybrid Rejection Sampling and Direct Preference Optimization Method for Alignment of Large Language Models](https://arxiv.org/abs/2402.10038) | 本研究提出了一种名为RS-DPO的方法，它将拒绝采样和直接优化偏好结合起来，用于对齐大型语言模型。通过开发一个经过监督微调的策略模型，并从该模型中直接采样响应，RS-DPO能够有效解决基于近端策略优化的不稳定性和高计算成本的问题。通过识别对比样本对，RS-DPO能够更好地进行RLHF。 |
| [^25] | [Self-Augmented In-Context Learning for Unsupervised Word Translation](https://arxiv.org/abs/2402.10024) | 通过自学习上下文增强方法，本论文提出一种无监督词汇翻译的方法，在零样本提示的大型语言模型上取得了显著的改进，超过了传统基于映射的方法。 |
| [^26] | [Bridging the Empirical-Theoretical Gap in Neural Network Formal Language Learning Using Minimum Description Length](https://arxiv.org/abs/2402.10013) | 通过使用最小描述长度目标（MDL），解决了神经网络在形式语言学习中的经验-理论差距问题。 |
| [^27] | [LLMs as Bridges: Reformulating Grounded Multimodal Named Entity Recognition](https://arxiv.org/abs/2402.09989) | 本文提出了RiVEG，一个统一的框架，通过利用大型语言模型（LLMs）作为连接桥梁，将多模态命名实体识别重新构建为联合任务，解决了命名实体无法确定和指代表达与命名实体之间的区别的问题。 |
| [^28] | [Fast Vocabulary Transfer for Language Model Compression](https://arxiv.org/abs/2402.09977) | 提出了一种基于词汇转移的语言模型压缩方法，通过与其他压缩技术结合使用，显著减小模型大小和推理时间，同时性能略有妥协。 |
| [^29] | [Case Study: Testing Model Capabilities in Some Reasoning Tasks](https://arxiv.org/abs/2402.09967) | 本研究探讨了大型语言模型（LLMs）的推理能力，在复杂推理场景中的挑战和限制是目前需要改进的关键问题。 |
| [^30] | [Crafting a Good Prompt or Providing Exemplary Dialogues? A Study of In-Context Learning for Persona-based Dialogue Generation](https://arxiv.org/abs/2402.09954) | 本研究通过对大型语言模型在基于角色生成对话方面进行实验，发现调整提示指令可以最直接有效且经济地提高生成质量，并且随机检索示范会取得最佳结果，而查询相同上下文的示范检索效果最差。即使破坏了示范中的多回合关联和单回合语义，对话生成仍然有效。 |
| [^31] | [Multi-Word Tokenization for Sequence Compression](https://arxiv.org/abs/2402.09949) | 本论文介绍了一种名为MWT的多词标记器，通过将频繁出现的多词表达式表示为单个标记，突破了词边界的限制，从而实现更紧凑和高效的标记化，提高了性能并加速推理过程。 |
| [^32] | [Generative AI in the Construction Industry: A State-of-the-art Analysis](https://arxiv.org/abs/2402.09939) | 本研究通过分析提供了建筑行业中生成式AI的最新状态、机遇和挑战。同时，提出了一个帮助建筑公司构建定制化生成式AI解决方案的框架。 |
| [^33] | [Paying Attention to Deflections: Mining Pragmatic Nuances for Whataboutism Detection in Online Discourse](https://arxiv.org/abs/2402.09934) | 本研究挖掘了在线话语中的演绎细微差别，提出了一种新的方法来准确检测反问主义，并在Twitter和YouTube数据集中取得了显著的改进。 |
| [^34] | [A Dataset of Open-Domain Question Answering with Multiple-Span Answers](https://arxiv.org/abs/2402.09923) | 提出了一个中文多段答案的开放领域问答数据集CLEAN，弥补了中文MSQA研究中的不足，包括多样的主题和需要详细回答的问题。提供了相关文献中的基线模型作为参考。 |
| [^35] | [BUSTER: a "BUSiness Transaction Entity Recognition" dataset](https://arxiv.org/abs/2402.09916) | BUSTER是一个商业交易实体识别的数据集，其中包含了3779份手动标注的金融交易文档，并建立了几个基准模型。最佳模型还用于自动标注6196份文档，并作为额外的银标准数据集发布。 |
| [^36] | [Enhancing Large Language Models with Pseudo- and Multisource- Knowledge Graphs for Open-ended Question Answering](https://arxiv.org/abs/2402.09911) | 使用伪和多源知识图对大型语言模型进行增强，以改善其幻觉问题和提高性能。通过结合伪图生成和原子级知识验证的框架，在开放式问题回答环境中使用知识图可以提高ROUGE-L分数至少11.5。 |
| [^37] | [DE-COP: Detecting Copyrighted Content in Language Models Training Data](https://arxiv.org/abs/2402.09910) | DE-COP是一种用于检测语言模型训练数据中版权内容的方法，通过对语言模型进行多项选择探测，可以识别出模型训练文本中可能包含的版权内容。该方法在模型的逻辑可用时比之前的方法提高了9.6%的检测性能，并在完全黑盒模型上实现了72%的准确率。 |
| [^38] | [Generative Representational Instruction Tuning](https://arxiv.org/abs/2402.09906) | 本研究引入了生成表示指令调整（GRIT）方法，通过指令区分生成和嵌入任务，训练一个大型语言模型同时处理这两种任务。与其他模型相比，我们的GritLM 7B在文本嵌入基准测试上达到最新的技术水平，并在多种生成任务中表现出色。通过进一步扩大规模，我们的GritLM 8x7B成为最佳的生成语言模型之一，同时仍然是最好的嵌入模型之一。GRIT的统一也大大提高了RAG在长文档上的速度。 |
| [^39] | [Not Just Novelty: A Longitudinal Study on Utility and Customization of AI Workflows](https://arxiv.org/abs/2402.09894) | 这项纵向研究调查了生成式AI工作流程的实用性和定制化程度，结果显示，在熟悉化阶段后，用户感知到的系统效用提高了。 |
| [^40] | [Camouflage is all you need: Evaluating and Enhancing Language Model Robustness Against Camouflage Adversarial Attacks](https://arxiv.org/abs/2402.09874) | 该研究评估和增强了语言模型对伪装性对抗攻击的鲁棒性。在对三种Transformer配置进行评估后发现，仅编码器模型在冒犯性语言检测和虚假信息检测任务中表现出最高的性能下降。 |
| [^41] | [LAPDoc: Layout-Aware Prompting for Documents](https://arxiv.org/abs/2402.09841) | 本文研究了通过使用布局增强来使用纯文本LLM进行文档特定任务的可能性。 |
| [^42] | [Knowledge of Pretrained Language Models on Surface Information of Tokens](https://arxiv.org/abs/2402.09808) | 该研究探究了预训练语言模型对于标记的表面信息的知识。结果显示，模型具备有关标记长度和子字符串的知识，但对标记结构的知识有限，解码器方面存在瓶颈。 |
| [^43] | [EFUF: Efficient Fine-grained Unlearning Framework for Mitigating Hallucinations in Multimodal Large Language Models](https://arxiv.org/abs/2402.09801) | EFUF是一种高效精细化去学习框架，可以消除多模态大语言模型中的物体幻觉，并不需要人工注释配对数据。 |
| [^44] | [NutePrune: Efficient Progressive Pruning with Numerous Teachers for Large Language Models](https://arxiv.org/abs/2402.09773) | NutePrune是一种高效逐渐剪枝方法，通过加载一个完整模型并将其与掩码和LoRA模块集成，实现了在大型语言模型中进行高效的结构剪枝。 |
| [^45] | [Grounding Language Model with Chunking-Free In-Context Retrieval](https://arxiv.org/abs/2402.09760) | 这是一种针对检索增强生成系统（RAG）的无块语境检索方法，通过绕过文本切分的过程，利用编码隐藏状态进行准确地语境检索，解决了传统方法中存在的文本语义连贯性破坏和证据检索中的噪声和不准确性问题。 |
| [^46] | [Efficient Language Adaptive Pre-training: Extending State-of-the-Art Large Language Models for Polish](https://arxiv.org/abs/2402.09759) | 本研究使用高效的语言自适应预训练方法，成功将基础英文大规模语言模型应用于生成波兰文，并在困惑度和任务表现上取得了显著的改进，为向现有语言模型添加新语言开辟了新途径。 |
| [^47] | [Model Compression and Efficient Inference for Large Language Models: A Survey](https://arxiv.org/abs/2402.09748) | 这项综述研究了大规模语言模型的压缩和高效推理方法，包括量化、修剪、蒸馏、紧凑架构设计和动态网络等方面。大模型的突出特点是压缩后需要微调或重新训练，并且相关的成本很高。 |
| [^48] | [AI Hospital: Interactive Evaluation and Collaboration of LLMs as Intern Doctors for Clinical Diagnosis](https://arxiv.org/abs/2402.09742) | AI医院是一个框架，用于构建实时交互式诊断环境，通过与LLMs的交互评估和协作，提高临床诊断的准确性。 |
| [^49] | [QuRating: Selecting High-Quality Data for Training Language Models](https://arxiv.org/abs/2402.09739) | QuRating是一种选择高质量数据用于训练语言模型的方法，它能够捕捉人类直观感知的文本的抽象特征。在实验中发现，平衡质量和多样性是很重要的。 |
| [^50] | [Align before Attend: Aligning Visual and Textual Features for Multimodal Hateful Content Detection](https://arxiv.org/abs/2402.09738) | 本论文提出了一种上下文感知的注意力框架，用于在视觉和文本特征之间进行对齐，以有效地进行多模态恶意内容检测。该方法能够选择性地关注模态特定的特征，解决了传统的融合技术无法有效关注模态特定特征的问题。研究在英语和非英语语言上进行了评估。 |
| [^51] | [Do LLMs Know about Hallucination? An Empirical Investigation of LLM's Hidden States](https://arxiv.org/abs/2402.09733) | 本研究探索了大型语言模型（LLMs）对幻觉的意识程度，通过实验发现LLMs在处理真实回答和虚构回答时有不同的反应，并展示了利用LLMs隐藏状态的指导具有巨大潜力。 |
| [^52] | [A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts](https://arxiv.org/abs/2402.09727) | ReadAgent是一个具有长期上下文概要记忆的阅读代理系统，通过实现一个简单的提示系统，它能够处理长输入并提高有效上下文长度。在评估中表现良好。 |
| [^53] | [Improving Non-autoregressive Machine Translation with Error Exposure and Consistency Regularization](https://arxiv.org/abs/2402.09725) | 本论文提出使用错误暴露和一致性正则化的训练方法来改进非自回归机器翻译中的数据分布不一致问题，并取得了显著的BLEU得分提升。 |
| [^54] | [An Analysis of Langauge Frequency and Error Correction for Esperanto](https://arxiv.org/abs/2402.09696) | 本论文运用 Eo-GP 数据集进行了世界语的频率分析，引入了 Eo-GEC 数据集用于错误识别。实验表明 GPT-4 在自动化和人工评估中的性能优于 GPT-3.5，展示了先进语言模型在增强对于较少研究语言的 GEC 策略方面的潜力。 |
| [^55] | [PAL: Proxy-Guided Black-Box Attack on Large Language Models](https://arxiv.org/abs/2402.09674) | PAL是第一个黑盒查询攻击大型语言模型的优化算法，通过代理模型引导优化过程，并使用复杂的损失函数，取得了较高的攻击成功率。 |
| [^56] | [How to Train Data-Efficient LLMs](https://arxiv.org/abs/2402.09668) | 本文研究了如何训练数据高效的LLM模型，提出了Ask-LLM和Density两种优秀的数据选择方法。 |
| [^57] | [EntailE: Introducing Textual Entailment in Commonsense Knowledge Graph Completion](https://arxiv.org/abs/2402.09666) | 本论文提出在常识知识图完成中引入文本蕴涵。在常识知识图的建模过程中，考虑节点和关系的语义合理性是关键。以前的方法利用概念抽象提高了事件合理性的一致性，但存在可扩展性和数据稀疏性的问题。在本文中，我们采用文本蕴涵来寻找隐含的蕴涵关系。 |
| [^58] | [CodeMind: A Framework to Challenge Large Language Models for Code Reasoning](https://arxiv.org/abs/2402.09664) | CodeMind是一个用于挑战大型语言模型进行代码推理的框架，通过评估LLMs的代码推理能力来替代仅仅依靠测试通过来评估，对三种代码推理任务进行评估，结果显示LLMs能够公正地理解控制流结构，并且对于简单程序和复杂程序，它们通常能够推理出输入如何演变为输出。 |
| [^59] | [GPT-4's assessment of its performance in a USMLE-based case study](https://arxiv.org/abs/2402.09654) | 本研究探讨了GPT-4在医疗应用中的表现评估。实验结果表明，反馈对相对置信度有影响，但并不一致地增加或减少。 |
| [^60] | [Answer is All You Need: Instruction-following Text Embedding via Answering the Question](https://arxiv.org/abs/2402.09642) | 通过回答问题进行指令遵循的文本嵌入方法，通过将指令视为对输入文本的问题并编码期望的答案，从而获得更相似的嵌入效果。 |
| [^61] | [MiMiC: Minimally Modified Counterfactuals in the Representation Space](https://arxiv.org/abs/2402.09631) | 提出了一种新颖的对抗事实生成方法，利用闭式解决方案在表示空间中生成富有表达力的对抗事实，以减轻语言模型中的不良行为，该方法在地球移动问题方面提供理论上的保证，并对表示空间的几何组织进行改进。 |
| [^62] | [API Pack: A Massive Multilingual Dataset for API Call Generation](https://arxiv.org/abs/2402.09615) | 这个论文介绍了一个名为API Pack的大规模多语言数据集，旨在提高大型语言模型的API调用生成能力，通过实验证明了其在生成未见过的API调用方面的高准确率，并实现了跨语言的API调用生成 |
| [^63] | [Probabilistic Reasoning in Generative Large Language Models](https://arxiv.org/abs/2402.09614) | 本文针对大型语言模型在概率推理任务中的限制，引入了贝叶斯语言推理数据集（BLInD），并提出了几种解决策略，包括Python代码和概率推理算法。 |
| [^64] | [Towards Privacy-Aware Sign Language Translation at Scale](https://arxiv.org/abs/2402.09611) | 本研究提出了一种两阶段框架，用于实现规模化隐私感知手语翻译。我们利用自监督视频预训练和有监督微调的方法，在数据稀缺和隐私风险的情况下实现了最先进的手语翻译性能。 |
| [^65] | [LogicPrpBank: A Corpus for Logical Implication and Equivalence](https://arxiv.org/abs/2402.09609) | LogicPrpBank是一个新的命题逻辑语料库，用于研究逻辑蕴含和等价的任务。通过与现有的语言模型进行基准测试，展示了该语料库在这一具有挑战性的任务中提供了有用的资源，并且还有改进的空间。 |
| [^66] | [Emerging Opportunities of Using Large Language Language Models for Translation Between Drug Molecules and Indications](https://arxiv.org/abs/2402.09588) | 本研究探讨了使用大型语言模型在药物分子和适应症之间进行翻译的机遇，提出了一个新任务，并测试了其有效性，这对于药物发现过程具有重要意义。 |
| [^67] | [Changes by Butterflies: Farsighted Forecasting with Group Reservoir Transformer](https://arxiv.org/abs/2402.09573) | 我们提出了一种群体储备转换器的架构，通过解决历史序列的挑战和初始条件的敏感性，实现更准确、更稳健地预测长期事件。在多元时间序列中，我们的模型相比最先进的DNN模型表现出更高的准确率，最高可减少89.43%的误差。 |
| [^68] | [Rationality Report Cards: Assessing the Economic Rationality of Large Language Models](https://arxiv.org/abs/2402.09552) | 本文在评估大型语言模型的经济合理性方面提出了一种方法，通过量化评分模型在各个要素上的表现并结合用户提供的评分标准，生成一份"理性报告卡"，以确定代理人是否足够可靠。 |
| [^69] | [Tell Me More! Towards Implicit User Intention Understanding of Language Model Driven Agents](https://arxiv.org/abs/2402.09205) | 该论文提出了一种面向基于语言模型的智能代理的隐式用户意图理解的方法。通过引入Intention-in-Interaction (IN3) 基准和在代理设计中融入模型专家，使得代理能够更好地与用户进行交互，并提升对用户指令的理解能力。 |
| [^70] | [(Ir)rationality and Cognitive Biases in Large Language Models](https://arxiv.org/abs/2402.09193) | 本研究评估了七个大型语言模型在认知心理学任务中的表现，发现它们与人类一样存在非理性，但展示的非理性方式与人类偏见不同，同时还表现出了显著的回答不一致性。 |
| [^71] | [Towards better Human-Agent Alignment: Assessing Task Utility in LLM-Powered Applications](https://arxiv.org/abs/2402.09015) | 本研究引入了AgentEval框架，用于评估LLM驱动应用的任务效用。该框架通过自动提出一套针对特定应用的评估标准，简化了效用验证过程，并对应用的效用进行了全面量化分析。 |
| [^72] | [DNABERT-S: Learning Species-Aware DNA Embedding with Genome Foundation Models](https://arxiv.org/abs/2402.08777) | DNABERT-S是一种专门用于创建物种感知的DNA嵌入的基因组基础模型。为了提高对长读DNA序列的嵌入效果，引入了Manifold Instance Mixup (MI-Mix)对比目标方法来训练模型。 |
| [^73] | [SemRel2024: A Collection of Semantic Textual Relatedness Datasets for 14 Languages](https://arxiv.org/abs/2402.08638) | SemRel2024是一个包含来自14种语言的语义文本相关性数据集合，通过该数据集合可以探索和量化语义相关性。这对于大型语言模型的能力和性能有重要的影响。这个数据集合涵盖了南非荷兰语、阿尔及利亚阿拉伯语、阿姆哈拉语、英语、豪萨语、印地语、印度尼西亚语、卢旺达语、马拉地语、摩洛哥阿拉伯语、现代标准阿拉伯语、旁遮普语、西班牙语和泰卢固语等14种语言。 |
| [^74] | [Eliciting Big Five Personality Traits in Large Language Models: A Textual Analysis with Classifier-Driven Approach](https://arxiv.org/abs/2402.08341) | 本研究使用分类器驱动的方法，通过不同的输入提示探究大型语言模型的输出变化，以增加其透明度。结果显示，这些模型根据输入的不同提示而表现出不同的人格特质，类似于人类对刺激做出的反应。 |
| [^75] | [BASE TTS: Lessons from building a billion-parameter Text-to-Speech model on 100K hours of data](https://arxiv.org/abs/2402.08093) | 基于10万小时数据的10亿参数文本到语音模型BASE TTS在语音自然度上达到了最新技术水平，并且能够展现自然的韵律。 |
| [^76] | [Policy Improvement using Language Feedback Models](https://arxiv.org/abs/2402.07876) | 本文介绍了一种使用语言反馈模型（LFMs）改进政策的方法，通过识别期望的行为并进行模仿学习，我们在任务完成率、泛化性能和人类可解释性方面取得了显著改进。 |
| [^77] | [Natural Language Reinforcement Learning](https://arxiv.org/abs/2402.07157) | 本研究将自然语言表示和强化学习原则相结合，提出了自然语言强化学习（NLRL）框架，解决了强化学习在样本效率低、解释性不足和缺乏监督信号等方面的限制问题，通过实验验证了其有效性和可解释性。 |
| [^78] | [Exploring Visual Culture Awareness in GPT-4V: A Comprehensive Probing](https://arxiv.org/abs/2402.06015) | 通过在GPT-4V模型上进行全面的探索，我们发现该模型在识别文化概念方面表现出色，但在低资源语言中仍表现较弱。在图像字幕任务中，GPT-4V在文化相关性方面优于原文。 |
| [^79] | [NoisyICL: A Little Noise in Model Parameters Calibrates In-context Learning](https://arxiv.org/abs/2402.05515) | NoisyICL通过在模型参数中引入噪音，提高了上下文学习的性能和校准性，实验结果显示NoisyICL可以产生更准确、更公平的预测。 |
| [^80] | [L4Q: Parameter Efficient Quantization-Aware Training on Large Language Models via LoRA-wise LSQ](https://arxiv.org/abs/2402.04902) | L4Q是一种参数高效的量化感知训练算法，通过基于LoRA的学习的量化步长，解决了大型语言模型中量化训练的挑战。 |
| [^81] | [PaDeLLM-NER: Parallel Decoding in Large Language Models for Named Entity Recognition](https://arxiv.org/abs/2402.04838) | 本研究提出了PaDeLLM-NER，一种能够在大型语言模型中实现并行解码，从而显著减少命名实体识别的生成延迟，同时保持预测质量和性能。 |
| [^82] | [Beyond Lines and Circles: Unveiling the Geometric Reasoning Gap in Large Language Models](https://arxiv.org/abs/2402.03877) | 本文调查了大型语言模型（LLMs）在几何推理方面的能力，并发现了它们在目标变量选择和2D空间关系方面存在偏见和困难。通过引入基于LLMs的多代理体系结构，本研究提出了一种通过自我纠正、协作和不同角色专业化来提高LLMs几何推理能力的框架。 |
| [^83] | [EffiBench: Benchmarking the Efficiency of Automatically Generated Code](https://arxiv.org/abs/2402.02037) | 本文提出了EffiBench基准测试，用于评估代码生成模型生成的代码的效率。实验证明，GPT-4-turbo生成的代码最高效。 |
| [^84] | [SWEA: Changing Factual Knowledge in Large Language Models via Subject Word Embedding Altering](https://arxiv.org/abs/2401.17809) | 提出了一种主题词嵌入修改框架（SWEA），通过在推理阶段修改主题的表示来编辑知识，保护模型的原始权重，避免不可逆的损害和额外的推理开销。 |
| [^85] | [PokeMQA: Programmable knowledge editing for Multi-hop Question Answering](https://arxiv.org/abs/2312.15194) | PokeMQA是一个可编程的多跳问答知识编辑模型，通过避免功能多样的推理任务的耦合，实现了更好的问题理解和回答能力。 |
| [^86] | [MAC-SQL: A Multi-Agent Collaborative Framework for Text-to-SQL](https://arxiv.org/abs/2312.11242) | MAC-SQL是一种基于LLM的多智能体协作框架，针对文本到SQL任务中的巨大数据库和复杂用户问题，通过核心分解器智能体和辅助智能体的协作，利用外部工具和模型进行解析和修正，实现了高效的文本到SQL生成和查询解析。 |
| [^87] | [Fewer is More: Boosting LLM Reasoning with Reinforced Context Pruning](https://arxiv.org/abs/2312.08901) | CoT-Influx是一种通过强化上下文修剪来提升LLM数学推理能力的方法，通过最大化有效和简明的示例输入，显著优于其他提示方法。 |
| [^88] | [MasonTigers@LT-EDI-2024: An Ensemble Approach towards Detecting Homophobia and Transphobia in Social Media Comments.](http://arxiv.org/abs/2401.14681) | MasonTigers@LT-EDI-2024团队提出了一种集成方法，用于在十种语言中检测社交媒体评论中的恐同和恶意性别认知，取得了较好的性能表现。 |
| [^89] | [Are self-explanations from Large Language Models faithful?.](http://arxiv.org/abs/2401.07927) | 大型语言模型的自我解释是否可靠是一个重要的AI安全考虑因素，我们提出使用自洽性检测作为评估其可靠性和解释能力的方法。 |
| [^90] | [Effects of diversity incentives on sample diversity and downstream model performance in LLM-based text augmentation.](http://arxiv.org/abs/2401.06643) | 本研究评估了三种文本多样性激励方法对LLM文本增强中生成文本的词汇多样性和下游模型性能的影响。结果表明，使用禁忌词能够最大程度地增加多样性，而使用先前创建的改写作为提示时，下游模型的性能最高。 |
| [^91] | [Adapting Large Language Models for Document-Level Machine Translation.](http://arxiv.org/abs/2401.06468) | 本文研究了适应大型语言模型进行文档级机器翻译的过程。实验结果显示，这些专门的模型在某些情况下超过了GPT-4的翻译性能，但在其他情况下仍然存在离标翻译问题，需要进一步改进和探索。 |
| [^92] | [Has Your Pretrained Model Improved? A Multi-head Posterior Based Approach.](http://arxiv.org/abs/2401.02987) | 本研究提出一种基于多头后验的方法，通过利用实体的元特征和模型的表示之间的一致性作为度量标准，有效评估预训练模型在各个领域的表现。 |
| [^93] | [Zero-Shot Position Debiasing for Large Language Models.](http://arxiv.org/abs/2401.01218) | 本文提出了一种零样本位置去偏方法（ZOE）来降低大语言模型（LLMs）的位置偏差问题，该方法利用预训练的LLMs的无监督响应进行去偏。实验证实ZOE在多个数据集和任务中均表现出优异的性能。 |
| [^94] | [Simultaneous Machine Translation with Large Language Models.](http://arxiv.org/abs/2309.06706) | 本文研究了使用大型语言模型进行同时机器翻译的可行性，通过引入混合策略，并进行有监督微调，取得了显著的性能改进。 |
| [^95] | [Simple synthetic data reduces sycophancy in large language models.](http://arxiv.org/abs/2308.03958) | 本文研究了语言模型中阿谀奉承行为的普遍性，并提出了一个简单的合成数据干预方法来减少这种行为。 |
| [^96] | [ODD: A Benchmark Dataset for the NLP-based Opioid Related Aberrant Behavior Detection.](http://arxiv.org/abs/2307.02591) | 这个研究介绍了一份名为ODD的新型基准数据集，用于通过分析患者的电子健康记录笔记，检测和分类药物滥用异常行为。这个数据集在药物相关病例的自然语言处理研究中具有重要的创新和贡献。 |
| [^97] | [LLMs and the Abstraction and Reasoning Corpus: Successes, Failures, and the Importance of Object-based Representations.](http://arxiv.org/abs/2305.18354) | 本文通过分析GPT模型在抽象推理语料库上的表现，发现GPT在抽象推理任务中存在需要核心概念“核心知识”的限制。通过使用基于对象的表示方法和新的1D-ARC基准，GPT在抽象推理任务中取得了较好的表现。 |
| [^98] | [DAPR: A Benchmark on Document-Aware Passage Retrieval.](http://arxiv.org/abs/2305.13915) | DAPR是一个文档感知段落检索的基准测试，挑战在于如何从长文档中找到正确的段落并返回准确结果。 |
| [^99] | [Towards Versatile and Efficient Visual Knowledge Injection into Pre-trained Language Models with Cross-Modal Adapters.](http://arxiv.org/abs/2305.07358) | 本文提出了X-adapter插拔式模块，利用多模态视觉语言模型，高效地向预训练语言模型注入视觉知识。 |
| [^100] | [Unleashing Infinite-Length Input Capacity for Large-scale Language Models with Self-Controlled Memory System.](http://arxiv.org/abs/2304.13343) | 该论文提出了一种自控内存系统，可以使大规模语言模型能够处理任意长度的输入，从而显著提高模型的性能表现。 |
| [^101] | [Tokenization Tractability for Human and Machine Learning Model: An Annotation Study.](http://arxiv.org/abs/2304.10813) | 研究比较了六种分词方法，并发现人类可追溯的分词与机器学习模型中的分词不一定相同。 |

# 详细

[^1]: 自我对抗微调扩散模型用于文本到图像生成

    Self-Play Fine-Tuning of Diffusion Models for Text-to-Image Generation

    [https://arxiv.org/abs/2402.10210](https://arxiv.org/abs/2402.10210)

    本文介绍了一种创新的技术，称为自我对抗微调扩散模型（SPIN-Diffusion），通过扩散模型与其先前版本的竞争，实现了逐步自我改进过程。

    

    微调扩散模型在生成人工智能领域仍然是一个未被充分探索的前沿，尤其是与在大型语言模型（LLMs）微调方面取得的显著进展相比。尽管现在的先进扩散模型如稳定扩散（SD）和SDXL依赖于监督微调，但它们的性能在观察到一定数量的数据后必然会达到瓶颈。最近，强化学习（RL）被应用于通过人类偏好数据对扩散模型进行微调，但每个文本提示需要至少两个图像（“获胜者”和“失败者”图像）。本文介绍了一种创新的技术，称为自我对抗微调扩散模型（SPIN-Diffusion），其中扩散模型与其先前版本进行竞争，促进了一个迭代的自我改进过程。我们的方法提供了一种替代传统监督微调和RL策略的选择。

    arXiv:2402.10210v1 Announce Type: cross  Abstract: Fine-tuning Diffusion Models remains an underexplored frontier in generative artificial intelligence (GenAI), especially when compared with the remarkable progress made in fine-tuning Large Language Models (LLMs). While cutting-edge diffusion models such as Stable Diffusion (SD) and SDXL rely on supervised fine-tuning, their performance inevitably plateaus after seeing a certain volume of data. Recently, reinforcement learning (RL) has been employed to fine-tune diffusion models with human preference data, but it requires at least two images ("winner" and "loser" images) for each text prompt. In this paper, we introduce an innovative technique called self-play fine-tuning for diffusion models (SPIN-Diffusion), where the diffusion model engages in competition with its earlier versions, facilitating an iterative self-improvement process. Our approach offers an alternative to conventional supervised fine-tuning and RL strategies, signific
    
[^2]: 恢复生成模型的预微调权重

    Recovering the Pre-Fine-Tuning Weights of Generative Models

    [https://arxiv.org/abs/2402.10208](https://arxiv.org/abs/2402.10208)

    该论文提出了一种恢复生成模型预微调权重的方法，通过少量低秩微调模型可以恢复准确的预微调权重，利用这个新漏洞攻击大规模模型。

    

    在生成建模中，主流模式包括两个步骤：i) 在大规模但不安全的数据集上进行预训练，ii) 通过微调将预训练模型与人类价值观对齐。这种做法被认为是安全的，因为目前没有一种方法可以恢复不安全的预微调模型权重。本文证明了这种假设通常是错误的。具体而言，我们提出了一种称为谱反调的方法，可以使用少量低秩（LoRA）微调模型恢复预微调模型的权重。与先前试图恢复预微调能力的攻击不同，我们的方法旨在恢复精确的预微调权重。我们的方法利用了这个新的对大规模模型的漏洞，例如个性化的稳定扩散和对齐的Mistral模型。

    arXiv:2402.10208v1 Announce Type: cross  Abstract: The dominant paradigm in generative modeling consists of two steps: i) pre-training on a large-scale but unsafe dataset, ii) aligning the pre-trained model with human values via fine-tuning. This practice is considered safe, as no current method can recover the unsafe, pre-fine-tuning model weights. In this paper, we demonstrate that this assumption is often false. Concretely, we present Spectral DeTuning, a method that can recover the weights of the pre-fine-tuning model using a few low-rank (LoRA) fine-tuned models. In contrast to previous attacks that attempt to recover pre-fine-tuning capabilities, our method aims to recover the exact pre-fine-tuning weights. Our approach exploits this new vulnerability against large-scale models such as a personalized Stable Diffusion and an aligned Mistral.
    
[^3]: 基于上下文的奖励：基于动态偏好调整的多目标基础模型对齐

    Rewards-in-Context: Multi-objective Alignment of Foundation Models with Dynamic Preference Adjustment

    [https://arxiv.org/abs/2402.10207](https://arxiv.org/abs/2402.10207)

    本文介绍了Rewards-in-Context（RiC）方法，该方法通过多个奖励条件控制基础模型的响应，并应用有监督的微调进行对齐。它具有简单性和适应性，并支持在推理时动态调整用户偏好。

    

    我们考虑了基于人类偏好的基础模型多目标对齐问题，这是实现有益和无害的人工智能系统的关键步骤。然而，使用强化学习（RL）对大型基础模型进行微调通常是昂贵且不稳定的，并且人类偏好的多维度、异质性和冲突性进一步复杂化了对齐过程。在本文中，我们引入了Rewards-in-Context（RiC）方法，它使得基础模型的响应取决于其提示上下文中的多个奖励，并应用有监督的微调来进行对齐。RiC的显著特点是简单性和适应性，因为它只需要对单个基础模型进行有监督的微调，并支持在推理时动态调整用户偏好。受到抽象的凸优化问题的解析解的启发，我们提出了一种动态推理时调整方法。

    arXiv:2402.10207v1 Announce Type: cross  Abstract: We consider the problem of multi-objective alignment of foundation models with human preferences, which is a critical step towards helpful and harmless AI systems. However, it is generally costly and unstable to fine-tune large foundation models using reinforcement learning (RL), and the multi-dimensionality, heterogeneity, and conflicting nature of human preferences further complicate the alignment process. In this paper, we introduce Rewards-in-Context (RiC), which conditions the response of a foundation model on multiple rewards in its prompt context and applies supervised fine-tuning for alignment. The salient features of RiC are simplicity and adaptivity, as it only requires supervised fine-tuning of a single foundation model and supports dynamic adjustment for user preferences during inference time. Inspired by the analytical solution of an abstracted convex optimization problem, our dynamic inference-time adjustment method appro
    
[^4]: 不需要提示的思维链推理

    Chain-of-Thought Reasoning Without Prompting

    [https://arxiv.org/abs/2402.10200](https://arxiv.org/abs/2402.10200)

    本研究发现，通过改变解码过程而不是使用提示工程，可以从预训练的大型语言模型中引出思维链推理路径，这种方法绕过了提示的限制，并且可以评估模型的内在推理能力。

    

    在提升大型语言模型（LLMs）的推理能力方面，以往的研究主要集中在特定的提示技术，如少样本或零样本的思维链提示。这些方法虽然有效，但往往需要手动进行大量的提示工程。我们的研究采用了一种新的方法：LLMs是否可以在没有提示的情况下有效推理？我们的研究结果表明，有趣的是，通过简单地改变解码过程，就可以从预训练的LLMs中引出思维链推理路径。我们研究了前$k$个替代标记，发现这些序列中经常存在思维链路径。这种方法不仅绕过了提示的混淆因素，还可以评估LLMs的内在推理能力。此外，我们观察到解码路径中的思维链存在与模型解码的置信度较高的相关性。

    arXiv:2402.10200v1 Announce Type: new  Abstract: In enhancing the reasoning capabilities of large language models (LLMs), prior research primarily focuses on specific prompting techniques such as few-shot or zero-shot chain-of-thought (CoT) prompting. These methods, while effective, often involve manually intensive prompt engineering. Our study takes a novel approach by asking: Can LLMs reason effectively without prompting? Our findings reveal that, intriguingly, CoT reasoning paths can be elicited from pre-trained LLMs by simply altering the \textit{decoding} process. Rather than conventional greedy decoding, we investigate the top-$k$ alternative tokens, uncovering that CoT paths are frequently inherent in these sequences. This approach not only bypasses the confounders of prompting but also allows us to assess the LLMs' \textit{intrinsic} reasoning abilities. Moreover, we observe that the presence of a CoT in the decoding path correlates with a higher confidence in the model's decod
    
[^5]: 一座摇摇欲坠的纸牌屋？对语言代理的敌对攻击的映射

    A Trembling House of Cards? Mapping Adversarial Attacks against Language Agents

    [https://arxiv.org/abs/2402.10196](https://arxiv.org/abs/2402.10196)

    本文是第一个对语言代理的敌对攻击进行映射的系统化努力。它提供了一个统一的概念框架来研究这些攻击。这有助于我们理解语言代理的安全风险。

    

    由大型语言模型（LLMs）驱动的语言代理在发展中迅猛。它们利用语言作为思想和交流的工具，赋予了无比的灵活性和多样性。人们迅速利用这种能力将LLMs连接到各种外部组件和环境中：数据库，工具，因特网，机器人实体等。许多人相信一种前所未有的强大自动化技术正在崛起。然而，新的自动化技术也带来了新的安全风险，特别是对于复杂的系统如语言代理。他们的开发和部署的速度和规模与我们对其安全风险的理解之间存在着令人惊讶的差距。我们是否正在建造一座纸牌屋？在本论文中，我们首次系统地对语言代理的敌对攻击进行了映射。我们首先提出了一个统一的概念框架

    arXiv:2402.10196v1 Announce Type: cross  Abstract: Language agents powered by large language models (LLMs) have seen exploding development. Their capability of using language as a vehicle for thought and communication lends an incredible level of flexibility and versatility. People have quickly capitalized on this capability to connect LLMs to a wide range of external components and environments: databases, tools, the Internet, robotic embodiment, etc. Many believe an unprecedentedly powerful automation technology is emerging. However, new automation technologies come with new safety risks, especially for intricate systems like language agents. There is a surprisingly large gap between the speed and scale of their development and deployment and our understanding of their safety risks. Are we building a house of cards? In this position paper, we present the first systematic effort in mapping adversarial attacks against language agents. We first present a unified conceptual framework for
    
[^6]: BitDelta：你的微调可能只有一个比特的价值

    BitDelta: Your Fine-Tune May Only Be Worth One Bit

    [https://arxiv.org/abs/2402.10193](https://arxiv.org/abs/2402.10193)

    BitDelta研究探讨了大型语言模型在微调过程中的信息冗余性，并提出了一种名为BitDelta的方法，可以将微调过程中添加的信息量化为一个比特，同时保持性能。这一发现对于多租户模型的服务和存储有重要意义，并可以显著降低GPU内存需求。

    

    大型语言模型（LLMs）通常在两个阶段进行训练：在大规模互联网数据集上进行预训练，然后在下游任务上进行微调。由于预训练的高计算需求，直觉上认为微调对模型的信息添加较少，因此更具有可压缩性。我们通过将微调模型的权重分解为预训练组件和额外的增量来探究这一假设。我们引入了一种简单的方法——BitDelta，成功地将这个增量量化为1比特而不影响性能。这一有趣的发现不仅突显了微调过程中添加的信息的潜在冗余性，而且对于多租户模型的服务和存储也具有重要影响。通过使用一个高精度的基础模型以及多个1比特的增量，BitDelta大大降低了GPU内存需求。

    arXiv:2402.10193v1 Announce Type: cross  Abstract: Large Language Models (LLMs) are typically trained in two phases: pre-training on large internet-scale datasets, and fine-tuning for downstream tasks. Given the higher computational demand of pre-training, it's intuitive to assume that fine-tuning adds less new information to the model, and is thus more compressible. We explore this assumption by decomposing the weights of fine-tuned models into their pre-trained components and an additional delta. We introduce a simple method, BitDelta, which successfully quantizes this delta down to 1 bit without compromising performance. This interesting finding not only highlights the potential redundancy of information added during fine-tuning, but also has significant implications for the multi-tenant serving and multi-tenant storage of fine-tuned models. By enabling the use of a single high-precision base model accompanied by multiple 1-bit deltas, BitDelta dramatically reduces GPU memory requir
    
[^7]: 大型语言模型的上下文学习中的不确定性分解和量化

    Uncertainty Decomposition and Quantification for In-Context Learning of Large Language Models

    [https://arxiv.org/abs/2402.10189](https://arxiv.org/abs/2402.10189)

    本文研究了大型语言模型（LLM）上下文学习中的不确定性，并提出了一种新的方法来量化这种不确定性，包括演示产生的不确定性和模型配置的模糊性。

    

    上下文学习已经成为大型语言模型（LLM）的突破性能力，并通过在提示中提供一些与任务相关的演示来彻底改变了各个领域。然而，LLM响应中的可信问题，如幻觉，也被积极讨论。现有工作致力于量化LLM响应中的不确定性，但往往忽视LLM的复杂性和上下文学习的独特性。在这项工作中，我们深入研究了与上下文学习相关的LLM预测不确定性，并强调这种不确定性可能来自于提供的演示（aleatoric不确定性）和与模型配置相关的模糊性（epistemic不确定性）。我们提出了一种新的公式和相应的估计方法来量化这两种类型的不确定性。该方法为理解上下文学习里的预测提供了一种无监督的方式。

    arXiv:2402.10189v1 Announce Type: new  Abstract: In-context learning has emerged as a groundbreaking ability of Large Language Models (LLMs) and revolutionized various fields by providing a few task-relevant demonstrations in the prompt. However, trustworthy issues with LLM's response, such as hallucination, have also been actively discussed. Existing works have been devoted to quantifying the uncertainty in LLM's response, but they often overlook the complex nature of LLMs and the uniqueness of in-context learning. In this work, we delve into the predictive uncertainty of LLMs associated with in-context learning, highlighting that such uncertainties may stem from both the provided demonstrations (aleatoric uncertainty) and ambiguities tied to the model's configurations (epistemic uncertainty). We propose a novel formulation and corresponding estimation method to quantify both types of uncertainties. The proposed method offers an unsupervised way to understand the prediction of in-cont
    
[^8]: 重塑RLHF中的信息结构：基于图论的奖励泛化视角

    Rethinking Information Structures in RLHF: Reward Generalization from a Graph Theory Perspective

    [https://arxiv.org/abs/2402.10184](https://arxiv.org/abs/2402.10184)

    本研究通过设计奖励建模过程中的数据集信息结构，从图论的视角提出了RLHF中奖励泛化的问题，以解决多样的环境、低成本标注和可靠的对齐性能间的不兼容性。

    

    在强化学习从人类反馈中（RLHF）存在一个三难问题：高度多样的环境、低标注成本和可靠的对齐性能之间的不兼容性。本文旨在通过设计奖励建模过程中的数据集信息结构来缓解这种不兼容性。具体而言，我们重新审视了RLHF过程，并提出了一个理论框架将其描绘为文本分布上的自动编码过程。我们的框架形式化了RLHF目标，即确保人类偏好与大型语言模型（LLM）行为之间的分布一致性。基于这个框架，我们系统地研究了RLHF奖励建模阶段中信息结构的性能影响。为了进一步理解奖励建模阶段中的奖励泛化，我们引入了一种基于随机图论的方法来建模语义空间中的泛化。其中的关键见解是...

    arXiv:2402.10184v1 Announce Type: cross  Abstract: There is a trilemma in reinforcement learning from human feedback (RLHF): the incompatibility between highly diverse contexts, low labeling cost, and reliable alignment performance. Here we aim to mitigate such incompatibility through the design of dataset information structures during reward modeling. Specifically, we first reexamine the RLHF process and propose a theoretical framework portraying it as an autoencoding process over text distributions. Our framework formalizes the RLHF objective of ensuring distributional consistency between human preference and large language model (LLM) behavior. Building on this framework, we then systematically investigate the performance impact of information structure in the reward modeling stage of RLHF. To further understand reward generalization in the reward modeling stage, we introduce a new method based on random graph theory that models generalization in the semantic space. A key insight of
    
[^9]: TDAG:一种基于动态任务分解和代理生成的多代理框架

    TDAG: A Multi-Agent Framework based on Dynamic Task Decomposition and Agent Generation

    [https://arxiv.org/abs/2402.10178](https://arxiv.org/abs/2402.10178)

    提出了一种基于动态任务分解和代理生成的多代理框架(TDAG)，该框架能够在解决复杂的实际问题时提高代理的适应性。同时，引入了ItineraryBench，这是一个在旅行规划中具有精细评估系统的基准测试。

    

    Large Language Models (LLMs)如ChatGPT的出现，启发了基于LLM的代理的开发，能够解决复杂的实际问题。然而，这些代理在执行任务时常常由于方法论约束（如错误传播和适应性受限）而遇到困难。为了解决这个问题，我们提出了一种基于动态任务分解和代理生成的多代理框架(TDAG)。该框架动态地将复杂任务分解为更小的子任务，并将每个子任务分配给一个特别生成的子代理，从而提高了在多样化和不可预测的实际任务中的适应性。同时，现有的基准测试往往缺乏评估复杂、多步骤任务中递增进展所需的细粒度。为了应对这个问题，我们在旅行规划的背景下引入了ItineraryBench，它具有互连、逐渐复杂的任务和细粒度的评估系统。

    arXiv:2402.10178v1 Announce Type: new  Abstract: The emergence of Large Language Models (LLMs) like ChatGPT has inspired the development of LLM-based agents capable of addressing complex, real-world tasks. However, these agents often struggle during task execution due to methodological constraints, such as error propagation and limited adaptability. To address this issue, we propose a multi-agent framework based on dynamic Task Decomposition and Agent Generation (TDAG). This framework dynamically decomposes complex tasks into smaller subtasks and assigns each to a specifically generated subagent, thereby enhancing adaptability in diverse and unpredictable real-world tasks. Simultaneously, existing benchmarks often lack the granularity needed to evaluate incremental progress in complex, multi-step tasks. In response, we introduce ItineraryBench in the context of travel planning, featuring interconnected, progressively complex tasks with a fine-grained evaluation system. ItineraryBench i
    
[^10]: OpenMathInstruct-1: 一个拥有180万个数学教学调优数据集

    OpenMathInstruct-1: A 1.8 Million Math Instruction Tuning Dataset

    [https://arxiv.org/abs/2402.10176](https://arxiv.org/abs/2402.10176)

    OpenMathInstruct-1是一个包含180万个数学问题和解决方法对的数据集，通过合成开源LLM的代码解释器解决方案来构建，填补了目前开源LLM在数学技能方面与闭源LLM之间的差距。

    

    最近的研究表明，利用合成生成的数据集来训练大规模语言模型（LLM）具有巨大潜力，尤其是为了获得特定的技能。目前的大规模数学教学调优数据集，如MetaMathQA和MAmmoTH，是使用来自商业限制许可的闭源LLM的输出构建的。限制在这些数据生成流程中使用开源LLM的一个关键原因是目前最好的闭源LLM（如GPT-4）和最好的开源LLM之间在数学技能上存在很大差距。基于开源LLM的最近进展，我们提出了新颖的提示方式和一些强力缩放，构建了OpenMathInstruct-1，一个拥有180万个问题-解决方法对的数学教学调优数据集。该数据集是通过使用GSM8K和MATH这两个流行的数学推理基准的代码解释器解决方案进行合成构建的。

    arXiv:2402.10176v1 Announce Type: cross  Abstract: Recent work has shown the immense potential of synthetically generated datasets for training large language models (LLMs), especially for acquiring targeted skills. Current large-scale math instruction tuning datasets such as MetaMathQA (Yu et al., 2024) and MAmmoTH (Yue et al., 2024) are constructed using outputs from closed-source LLMs with commercially restrictive licenses. A key reason limiting the use of open-source LLMs in these data generation pipelines has been the wide gap between the mathematical skills of the best closed-source LLMs, such as GPT-4, and the best open-source LLMs. Building on the recent progress in open-source LLMs, our proposed prompting novelty, and some brute-force scaling, we construct OpenMathInstruct-1, a math instruction tuning dataset with 1.8M problem-solution pairs. The dataset is constructed by synthesizing code-interpreter solutions for GSM8K and MATH, two popular math reasoning benchmarks, using t
    
[^11]: 解锁结构测量：引入PDD，一种用于位置话语连贯的自动评估指标

    Unlocking Structure Measuring: Introducing PDD, an Automatic Metric for Positional Discourse Coherence

    [https://arxiv.org/abs/2402.10175](https://arxiv.org/abs/2402.10175)

    本论文引入了一种新的自动评估指标PDD，用于评估长篇文章之间的话语连贯性，实验证明该指标更接近人类偏好和GPT-4的评估结果。

    

    近期，大型语言模型(LLMs)在各种任务中展现了与用户意图对齐生成文本的卓越性能。当涉及长篇文本生成时，从话语连贯的角度出发对生成结果产生了越来越大的兴趣。然而，现有的词汇或语义评估指标如BLEU、ROUGE、BertScore不能有效捕捉话语连贯性。因此，发展针对LLMs生成结果的话语特定的自动评估方法需要更多的关注和探索。在本文中，我们提出了一种新颖的自动评估指标，用于量化两篇长篇文章之间的话语差异。对来自代表性领域的三个数据集进行广泛实验，结果显示我们的指标更接近人类偏好和GPT-4的连贯性评估，优于现有的评估方法。

    arXiv:2402.10175v1 Announce Type: new  Abstract: Recent large language models (LLMs) have shown remarkable performance in aligning generated text with user intentions across various tasks. When it comes to long-form text generation, there has been a growing interest in generation from a discourse coherence perspective. However, existing lexical or semantic metrics such as BLEU, ROUGE, BertScore cannot effectively capture the discourse coherence. The development of discourse-specific automatic evaluation methods for assessing the output of LLMs warrants greater focus and exploration. In this paper, we present a novel automatic metric designed to quantify the discourse divergence between two long-form articles. Extensive experiments on three datasets from representative domains demonstrate that our metric aligns more closely with human preferences and GPT-4 coherence evaluation, outperforming existing evaluation methods.
    
[^12]: 将语言模型扩展到128K上下文的数据工程

    Data Engineering for Scaling Language Models to 128K Context

    [https://arxiv.org/abs/2402.10171](https://arxiv.org/abs/2402.10171)

    本论文研究了将语言模型的上下文长度扩展到128K的连续预训练方法，通过适当的数据混合和轻量级的预训练可以实现，其中关键要点在于数据的数量和质量，需要注意领域平衡和长度上采样。

    

    我们研究了将语言模型的上下文长度扩展到128K的连续预训练方法，着重考虑数据工程。我们假设长上下文建模，特别是“能够利用任意输入位置的信息”的能力，在大规模预训练中已经得到了掌握，并且这种能力可以通过轻量级连续预训练在比训练时更长的上下文(例如从4K到128K)下轻松扩展。我们研究了连续预训练的数据“数量”和“质量”：(1)对于数量，我们证明5亿到50亿个标记足以使模型能够检索到128K上下文中的任何位置的信息；(2)对于质量，我们的结果同等强调“领域平衡”和“长度上采样”。具体而言，我们发现简单地上采样更长的数据，并不能提供足够的质量，而是需要注意数据的领域平衡和长度上采样。

    arXiv:2402.10171v1 Announce Type: cross  Abstract: We study the continual pretraining recipe for scaling language models' context lengths to 128K, with a focus on data engineering. We hypothesize that long context modeling, in particular \textit{the ability to utilize information at arbitrary input locations}, is a capability that is mostly already acquired through large-scale pretraining, and that this capability can be readily extended to contexts substantially longer than seen during training~(e.g., 4K to 128K) through lightweight continual pretraining on appropriate data mixture. We investigate the \textit{quantity} and \textit{quality} of the data for continual pretraining: (1) for quantity, we show that 500 million to 5 billion tokens are enough to enable the model to retrieve information anywhere within the 128K context; (2) for quality, our results equally emphasize \textit{domain balance} and \textit{length upsampling}. Concretely, we find that naively upsampling longer data o
    
[^13]: 知识注入的基于LLM的对话式健康代理：糖尿病患者的案例研究

    Knowledge-Infused LLM-Powered Conversational Health Agent: A Case Study for Diabetes Patients

    [https://arxiv.org/abs/2402.10153](https://arxiv.org/abs/2402.10153)

    本文提出了一种知识注入的基于LLM的对话式健康代理（CHA）用于糖尿病患者，通过整合领域特定知识和分析能力，提高了糖尿病管理的准确性和效果。

    

    有效的糖尿病管理对于糖尿病患者的健康至关重要。大型语言模型（LLM）为糖尿病管理开辟了新的途径，提高了其效果。然而，目前基于LLM的方法受限于对一般来源的依赖，缺乏与领域特定知识的整合，导致回复不准确。本文提出了一种知识注入的基于LLM的对话式健康代理（CHA）用于糖尿病患者。我们根据开源的openCHA框架进行定制，并增强了我们的CHA的外部知识和分析能力。这种整合包括两个关键组成部分：1）整合美国糖尿病协会的膳食指南和Nutritionix的信息；2）部署分析工具，实现营养摄入计算并与指南进行比较。我们将提出的CHA与GPT4进行比较。我们的评估包括100个糖尿病患者的使用情况。

    arXiv:2402.10153v1 Announce Type: new  Abstract: Effective diabetes management is crucial for maintaining health in diabetic patients. Large Language Models (LLMs) have opened new avenues for diabetes management, facilitating their efficacy. However, current LLM-based approaches are limited by their dependence on general sources and lack of integration with domain-specific knowledge, leading to inaccurate responses. In this paper, we propose a knowledge-infused LLM-powered conversational health agent (CHA) for diabetic patients. We customize and leverage the open-source openCHA framework, enhancing our CHA with external knowledge and analytical capabilities. This integration involves two key components: 1) incorporating the American Diabetes Association dietary guidelines and the Nutritionix information and 2) deploying analytical tools that enable nutritional intake calculation and comparison with the guidelines. We compare the proposed CHA with GPT4. Our evaluation includes 100 diabe
    
[^14]: ControlLM: 为语言模型打造多样性个性

    ControlLM: Crafting Diverse Personalities for Language Models

    [https://arxiv.org/abs/2402.10151](https://arxiv.org/abs/2402.10151)

    ControlLM是一种可以在推理时调整语言模型个性特征的方法，通过差异的行为提示激活模式来实现，从而实现模型行为的精确、实时调整。

    

    随着语言模型在规模和能力上的持续扩大，它们展示出了一系列新兴的行为，既有有益的也有令人担忧的行为。这增加了对模型行为的控制需求。我们希望能够在推理时控制语言模型的个性特征，以满足不同类型任务的要求。个性是语言模型的更高级和更抽象的行为表示。我们介绍了ControlLM，它利用模型潜在空间中对比的行为提示的差异激活模式，以影响模型推理时的个性特征。这种方法允许精确、实时地调整模型行为。首先，我们证明了ControlLM能够在没有任何训练的情况下引起多样的个性行为，而精确控制使个性特征能够与平均情形相匹配。

    arXiv:2402.10151v1 Announce Type: new  Abstract: As language models continue to scale in size and capability, they display an array of emerging behaviors, both beneficial and concerning. This heightens the need to control model behaviors. We hope to be able to control the personality traits of language models at the inference-time so as to have various character features, on top of which the requirements of different types of tasks can be met. Personality is a higher-level and more abstract behavioral representation for language models. We introduce ControlLM, which leverages differential activation patterns, derived from contrasting behavioral prompts in the model's latent space, to influence the model's personality traits at inference. This approach allows for the precise, real-time adjustment of model behavior. First, we demonstrate ControlLM's capacity to elicit diverse persona behaviors without any training, while precision control allows personality traits to closely match averag
    
[^15]: TOAD: 具有多样响应风格的面向任务的自动对话系统

    TOAD: Task-Oriented Automatic Dialogs with Diverse Response Styles

    [https://arxiv.org/abs/2402.10137](https://arxiv.org/abs/2402.10137)

    TOAD是一个具有多样响应风格的面向任务的自动对话系统，其中考虑了冗长程度和用户表达镜像两个方面。TOAD通过模拟真实的应用上下文交互，提供了丰富的系统响应风格选项，并在评估中表明建模更冗长的回复或不进行用户表达镜像的回复更具挑战性。

    

    最近大语言模型（LLM）的进展显示，下一代虚拟助手的期望包括在各种使用场景下提供更加自然和适应性强的对话能力。然而，为面向任务的对话（TOD）创建高质量的标注数据被认为是缓慢和昂贵的。为了解决这些挑战，我们引入了任务导向的自动对话系统（TOAD），这是一个新颖且可扩展的TOD数据集，以及与之配套的自动生成流程。TOAD数据集模拟了真实的应用上下文交互，并提供了各种系统响应风格选项。考虑了系统响应风格的两个方面，即冗长程度和用户表达镜像。我们在两个响应生成任务上进行了TOAD的评估，结果显示建模更冗长的回复或不进行用户表达镜像的回复更具挑战性。

    arXiv:2402.10137v1 Announce Type: new  Abstract: In light of recent advances in large language models~(LLMs), the expectations for the next generation of virtual assistants include enhanced naturalness and adaptability across diverse usage scenarios. However, the creation of high-quality annotated data for Task-Oriented Dialog~(TOD) is recognized to be slow and costly. To address these challenges, we introduce Task-Oriented Automatic Dialogs~(TOAD), a novel and scalable TOD dataset along with its automatic generation pipeline. The TOAD dataset simulates realistic app context interaction and provide a variety of system response style options. Two aspects of system response styles are considered, verbosity level and users' expression mirroring. We benchmark TOAD on two response generation tasks and the results show that modeling more verbose or responses without user expression mirroring is more challenging.
    
[^16]: 选择性反射调节：LLM指令调节的学生选择数据回收

    Selective Reflection-Tuning: Student-Selected Data Recycling for LLM Instruction-Tuning

    [https://arxiv.org/abs/2402.10110](https://arxiv.org/abs/2402.10110)

    本文介绍了一种名为选择性反射调节的新方法，该方法通过教师LLM的反射和自省与学生LLM的数据选择能力相结合，自动优化现有的指令调节数据，从而实现了高效的指令调节和卓越性能的LLM。

    

    指令调节对于大型语言模型（LLM）来说非常关键，以实现更好的指令跟踪和任务适应能力，但其成功在很大程度上取决于训练数据的质量。许多最近的方法都致力于改进数据质量，但往往忽视了数据与正在微调的学生模型的兼容性。本文介绍了一种新的范式——选择性反射调节，通过结合教师LLM的反射和自省，以自动优化现有的指令调节数据。这种师生合作产生了高质量且与学生LLM兼容的指令响应对，从而实现了高效的指令调节和卓越性能的LLM。选择性反射调节是一种数据增强和合成方法，通常能改善LLM微调和自我优化，而无需额外的计算资源。

    arXiv:2402.10110v1 Announce Type: cross  Abstract: Instruction tuning is critical to large language models (LLMs) for achieving better instruction following and task adaptation capabilities but its success heavily relies on the training data quality. Many recent methods focus on improving the data quality but often overlook the compatibility of the data with the student model being finetuned. This paper introduces Selective Reflection-Tuning, a novel paradigm that synergizes a teacher LLM's reflection and introspection for improving existing data quality with the data selection capability of the student LLM, to automatically refine existing instruction-tuning data. This teacher-student collaboration produces high-quality and student-compatible instruction-response pairs, resulting in sample-efficient instruction tuning and LLMs of superior performance. Selective Reflection-Tuning is a data augmentation and synthesis that generally improves LLM finetuning and self-improvement without co
    
[^17]: 用可解释的风险预测方法降低诊断错误

    Towards Reducing Diagnostic Errors with Interpretable Risk Prediction

    [https://arxiv.org/abs/2402.10109](https://arxiv.org/abs/2402.10109)

    本研究提出了一种使用LLMs方法来识别病人电子病历数据中指示特定诊断风险增加或减少的证据的方法，旨在通过增加证据的获取与减少诊断错误来降低诊断错误。模型使用神经加性模型进行预测，以证据为后盾，并给出个体化风险估计，特别针对诊断延迟和来自不完整鉴别的错误进行优化。

    

    许多诊断错误发生是因为临床医生无法轻易获取病人电子病历中的相关信息。本研究提出了一种使用LLMs方法来识别病人电子病历数据中指示特定诊断风险增加或减少的证据的方法，最终目标是增加证据的获取与减少诊断错误。我们提出了一种神经加性模型来进行带有个体化风险估计的以证据为后盾的预测，在临床医生仍然不确定的时间点上，旨在特别减轻诊断延迟和源于不完整鉴别的错误。为了训练这样一个模型，需要推断出事件性的“真实”诊断的时间粒度细致的回顾性标签。我们使用LLMs来保证输入文本是在可以进行自信的诊断之前的。我们使用LLMs来检索初始的证据池，然后进行细化。

    arXiv:2402.10109v1 Announce Type: new  Abstract: Many diagnostic errors occur because clinicians cannot easily access relevant information in patient Electronic Health Records (EHRs). In this work we propose a method to use LLMs to identify pieces of evidence in patient EHR data that indicate increased or decreased risk of specific diagnoses; our ultimate aim is to increase access to evidence and reduce diagnostic errors. In particular, we propose a Neural Additive Model to make predictions backed by evidence with individualized risk estimates at time-points where clinicians are still uncertain, aiming to specifically mitigate delays in diagnosis and errors stemming from an incomplete differential. To train such a model, it is necessary to infer temporally fine-grained retrospective labels of eventual "true" diagnoses. We do so with LLMs, to ensure that the input text is from before a confident diagnosis can be made. We use an LLM to retrieve an initial pool of evidence, but then refin
    
[^18]: 可控扩散语言模型的量化嵌入向量

    Quantized Embedding Vectors for Controllable Diffusion Language Models

    [https://arxiv.org/abs/2402.10107](https://arxiv.org/abs/2402.10107)

    本论文提出了一种量化嵌入可控扩散语言模型（QE-CDLM），通过重建任务特定的嵌入空间来改善扩散语言模型的可控性、可移植性和稳定性。

    

    改善扩散语言模型的可控性、可移植性和推理速度是自然语言生成中的一个关键挑战。尽管最近的研究在语言模型的复杂文本生成方面取得了显著成功，但内存和计算能力仍然非常苛刻，无法满足预期，这自然导致模型的可移植性和稳定性较低。为了解决这些问题，我们提出了一种新的方法，称为量化嵌入可控扩散语言模型（QE-CDLM）。QE-CDLM基于最近成功的可控DLM，通过量化重建了任务特定的嵌入空间。这导致了一种基于梯度的生成式控制器。

    arXiv:2402.10107v1 Announce Type: cross  Abstract: Improving the controllability, portability, and inference speed of diffusion language models (DLMs) is a key challenge in natural language generation. While recent research has shown significant success in complex text generation with language models, the memory and computational power are still very demanding and fall short of expectations, which naturally results in low portability and instability for the models. To mitigate these issues, numerous well-established methods were proposed for neural network quantization. To further enhance their portability of independent deployment as well as improve their stability evaluated by language perplexity, we propose a novel approach called the Quantized Embedding Controllable Diffusion Language Model (QE-CDLM). QE-CDLM builds upon the recent successful controllable DLMs by remodeling the task-specific embedding space via quantization. This leads to a gradient-based controller for the generat
    
[^19]: GeoEval：用于评估LLMs和多模型在几何问题解决上的基准

    GeoEval: Benchmark for Evaluating LLMs and Multi-Modal Models on Geometry Problem-Solving

    [https://arxiv.org/abs/2402.10104](https://arxiv.org/abs/2402.10104)

    GeoEval基准测试用于评估LLMs和MMs在几何问题解决上的性能，发现WizardMath模型在主要子集上表现出色，但在具有挑战性的子集上准确率较低。

    

    近期在大型语言模型（LLMs）和多模型（MMs）方面的进展展示了它们在问题解决方面的卓越能力。然而，它们在处理几何数学问题方面的熟练程度，即需要综合理解文本和视觉信息，尚未得到彻底评估。为了填补这一空白，我们推出了GeoEval基准测试，这是一个全面的集合，包括一个主要子集合的2000个问题，一个重点关注反推理的750个问题子集合，一个增强子集合的2000个问题以及一个难题子集合的300个问题。这个基准测试有助于更深入地研究LLMs和MMs在解决几何数学问题时的性能。我们对十个LLMs和MMs在这些不同子集上的评估结果显示，WizardMath模型表现出色，在主要子集上达到55.67%的准确率，但在具有挑战性的子集上只有6.00%的准确率。这突出了关键的需求。

    arXiv:2402.10104v1 Announce Type: new  Abstract: Recent advancements in Large Language Models (LLMs) and Multi-Modal Models (MMs) have demonstrated their remarkable capabilities in problem-solving. Yet, their proficiency in tackling geometry math problems, which necessitates an integrated understanding of both textual and visual information, has not been thoroughly evaluated. To address this gap, we introduce the GeoEval benchmark, a comprehensive collection that includes a main subset of 2000 problems, a 750 problem subset focusing on backward reasoning, an augmented subset of 2000 problems, and a hard subset of 300 problems. This benchmark facilitates a deeper investigation into the performance of LLMs and MMs on solving geometry math problems. Our evaluation of ten LLMs and MMs across these varied subsets reveals that the WizardMath model excels, achieving a 55.67\% accuracy rate on the main subset but only a 6.00\% accuracy on the challenging subset. This highlights the critical ne
    
[^20]: 同时重视：在不损害普适智能的情感智能大语言模型中增强情绪智力

    Both Matter: Enhancing the Emotional Intelligence of Large Language Models without Compromising the General Intelligence

    [https://arxiv.org/abs/2402.10073](https://arxiv.org/abs/2402.10073)

    本论文提出了一个称为\textbf{MoEI}的方法，通过引入一个大规模的EI相关任务集合\textsc{EiBench}并采用模块化的训练过程和情感智力增强器的集成，综合提高了LLM的情感智力（EI）而不损害其普适智能（GI）。

    

    情感智力（EI）包括情感感知、情感认知和情感表达，在提高当前基于大语言模型（LLM）的对话式普适人工智能助手与用户交互体验方面起着关键作用。先前的工作主要集中在通过对EI相关的分类或回归任务的天真微调提高其情感感知能力。然而，这导致了EI的不完全增强和对普适智能的灾难性遗忘。为此，我们首先引入了一个大规模的以文本为基础的EI相关任务集合\textsc{EiBench}，其中包括了覆盖了EI的三个方面的任务指示，为综合增强LLM的EI奠定了坚实的基础。然后，提出了一种新颖的模块化情绪智力增强方法（\textbf{MoEI}），包含了模块化的训练过程和情感智力增强器的集成，以同时提高EI和保持普适智能。

    arXiv:2402.10073v1 Announce Type: new  Abstract: Emotional Intelligence (EI), consisting of emotion perception, emotion cognition and emotion expression, plays the critical roles in improving user interaction experience for the current large language model (LLM) based conversational general AI assistants. Previous works mainly focus on raising the emotion perception ability of them via naive fine-tuning on EI-related classification or regression tasks. However, this leads to the incomplete enhancement of EI and catastrophic forgetting of the general intelligence (GI). To this end, we first introduce \textsc{EiBench}, a large-scale collection of EI-related tasks in the text-to-text formation with task instructions that covers all three aspects of EI, which lays a solid foundation for the comprehensive EI enhancement of LLMs. Then a novel \underline{\textbf{Mo}}dular \underline{\textbf{E}}motional \underline{\textbf{I}}ntelligence enhancement method (\textbf{MoEI}), consisting of Modular
    
[^21]: 通过机器遗忘实现更安全的大型语言模型

    Towards Safer Large Language Models through Machine Unlearning

    [https://arxiv.org/abs/2402.10058](https://arxiv.org/abs/2402.10058)

    这项研究提出了一种新的大型语言模型遗忘框架SKU，旨在消除有害知识的同时保留模型对正常提示的实用性。

    

    大型语言模型（LLM）的快速发展展示了它们在各个领域的巨大潜力，归因于它们广泛的预训练知识和出色的泛化能力。然而，当LLM面对有问题的提示时，经常会遇到生成有害内容的挑战。为了解决这个问题，现有的工作尝试通过梯度上升方法阻止LLM产生有害输出。虽然这些方法可能有效，但它们经常影响模型对正常提示的实用性。为了解决这一差距，我们引入了选择性知识否认遗忘（SKU），这是一种新颖的针对LLM的遗忘框架，旨在消除有害知识，同时保留对正常提示的实用性。具体而言，SKU由两个阶段组成：有害知识获取阶段和知识否定阶段。第一个阶段旨在识别和获取有害知识。

    arXiv:2402.10058v1 Announce Type: new  Abstract: The rapid advancement of Large Language Models (LLMs) has demonstrated their vast potential across various domains, attributed to their extensive pretraining knowledge and exceptional generalizability. However, LLMs often encounter challenges in generating harmful content when faced with problematic prompts. To address this problem, existing work attempted to implement a gradient ascent based approach to prevent LLMs from producing harmful output. While these methods can be effective, they frequently impact the model utility in responding to normal prompts. To address this gap, we introduce Selective Knowledge negation Unlearning (SKU), a novel unlearning framework for LLMs, designed to eliminate harmful knowledge while preserving utility on normal prompts. Specifically, SKU is consisted of two stages: harmful knowledge acquisition stage and knowledge negation stage. The first stage aims to identify and acquire harmful knowledge within t
    
[^22]: 大型语言模型通过自我蒸馏和有意识的想象进行遗忘

    Unmemorization in Large Language Models via Self-Distillation and Deliberate Imagination

    [https://arxiv.org/abs/2402.10052](https://arxiv.org/abs/2402.10052)

    本研究提出了一种新颖的大型语言模型遗忘方法，通过自我蒸馏和有意识的想象，有效地遗忘目标文本，并在生成任务和自然语言理解任务中保留模型的能力。

    

    虽然在许多任务上表现出令人印象深刻的生成能力，但大型语言模型（LLM）仍然存在隐私侵犯和敏感数据不受控制的问题。因此，我们提出了一种新颖的方法，即在LLM遗忘的过程中采用有意识的想象。我们不是试图忘记已记忆的数据，而是通过自我蒸馏的框架引导LLM有意识地想象替代情境。通过广泛的实验，我们证明了这种方法不仅可以有效地遗忘目标文本，还可以保留LLM在开放式生成任务和自然语言理解（NLU）任务中的能力。我们的结果展示了这种方法在不同模型和规模中的实用性。

    arXiv:2402.10052v1 Announce Type: cross  Abstract: While displaying impressive generation capabilities across many tasks, Large Language Models (LLMs) still struggle with crucial issues of privacy violation and unwanted exposure of sensitive data. This raises an essential question: how should we prevent such undesired behavior of LLMs while maintaining their strong generation and natural language understanding (NLU) capabilities? In this work, we introduce a novel approach termed deliberate imagination in the context of LLM unlearning. Instead of trying to forget memorized data, we employ a self-distillation framework, guiding LLMs to deliberately imagine alternative scenarios. As demonstrated in a wide range of experiments, the proposed method not only effectively unlearns targeted text but also preserves the LLMs' capabilities in open-ended generation tasks as well as in NLU tasks. Our results demonstrate the usefulness of this approach across different models and sizes, and also wit
    
[^23]: SwissNYF：基于黑盒环境的基于LLM的工具生成智能体

    SwissNYF: Tool Grounded LLM Agents for Black Box Setting

    [https://arxiv.org/abs/2402.10051](https://arxiv.org/abs/2402.10051)

    该论文提出了一种基于黑盒环境的基于LLM的工具生成智能体的方法，在复杂的API调用中表现出了优越的性能，可以应对具有不可逆性和大量时间消耗的任务。

    

    在访问函数的返回结果上，大型语言模型（LLM）已经展示了增强的功能调用能力，但这种方法在简单API上是实用的，但对于不可逆API（例如数据库删除API）会面临可扩展性问题。同样，对于每个API调用需要大量时间的流程以及需要前向规划的自动化操作管道等都存在复杂的挑战。此外，通常出现的情况是需要一种通用的方法，因为算法缺乏对这些函数的特定实现或使用它们的秘密的直接访问方式。在这些情况下，传统的工具规划方法是不合适的，因此需要在黑盒环境中运行。与在工具操作中的表现不同，LLM在黑盒任务（例如程序综合）中表现出色。因此，我们利用LLM来生成基于黑盒环境的智能体。

    arXiv:2402.10051v1 Announce Type: new  Abstract: While Large Language Models (LLMs) have demonstrated enhanced capabilities in function-calling, these advancements primarily rely on accessing the functions' responses. This methodology is practical for simpler APIs but faces scalability issues with irreversible APIs that significantly impact the system, such as a database deletion API. Similarly, processes requiring extensive time for each API call and those necessitating forward planning, like automated action pipelines, present complex challenges. Furthermore, scenarios often arise where a generalized approach is needed because algorithms lack direct access to the specific implementations of these functions or secrets to use them. Traditional tool planning methods are inadequate in these cases, compelling the need to operate within black-box environments. Unlike their performance in tool manipulation, LLMs excel in black-box tasks, such as program synthesis. Therefore, we harness the 
    
[^24]: RS-DPO：一种用于对齐大型语言模型的混合拒绝采样和直接优化偏好的方法

    RS-DPO: A Hybrid Rejection Sampling and Direct Preference Optimization Method for Alignment of Large Language Models

    [https://arxiv.org/abs/2402.10038](https://arxiv.org/abs/2402.10038)

    本研究提出了一种名为RS-DPO的方法，它将拒绝采样和直接优化偏好结合起来，用于对齐大型语言模型。通过开发一个经过监督微调的策略模型，并从该模型中直接采样响应，RS-DPO能够有效解决基于近端策略优化的不稳定性和高计算成本的问题。通过识别对比样本对，RS-DPO能够更好地进行RLHF。

    

    强化学习从人类反馈中学习（RLHF）已被广泛应用于将大型语言模型与用户意图对齐。然而，基于近端策略优化（PPO）的RLHF有时不稳定，需要显著的超参数微调，并且在对齐过程中计算成本高昂。最近，提出了直接优化偏好（DPO）来解决这些挑战。然而，DPO依赖于从人类标注者和替代LLM生成的对比回复，而不是策略模型，限制了RLHF的效果。本文通过系统地结合拒绝采样（RS）和DPO来解决这两个挑战。我们提出的方法RS-DPO，首先开发出一个经过监督微调的策略模型（SFT）。然后直接从SFT模型中采样每个提示的k个响应。RS-DPO基于其相似度识别对比样本对。

    arXiv:2402.10038v1 Announce Type: cross  Abstract: Reinforcement learning from human feedback (RLHF) has been extensively employed to align large language models with user intent. However, proximal policy optimization (PPO) based RLHF is occasionally unstable requiring significant hyperparameter finetuning, and computationally expensive to maximize the estimated reward during alignment. Recently, direct preference optimization (DPO) is proposed to address those challenges. However, DPO relies on contrastive responses generated from human annotator and alternative LLM, instead of the policy model, limiting the effectiveness of the RLHF. In this paper, we addresses both challenges by systematically combining rejection sampling (RS) and DPO. Our proposed method, RS-DPO, initiates with the development of a supervised fine-tuned policy model (SFT). A varied set of k responses per prompt are sampled directly from the SFT model. RS-DPO identifies pairs of contrastive samples based on their re
    
[^25]: 自学习上下文增强对于无监督词汇翻译的研究

    Self-Augmented In-Context Learning for Unsupervised Word Translation

    [https://arxiv.org/abs/2402.10024](https://arxiv.org/abs/2402.10024)

    通过自学习上下文增强方法，本论文提出一种无监督词汇翻译的方法，在零样本提示的大型语言模型上取得了显著的改进，超过了传统基于映射的方法。

    

    近期的研究表明，尽管大型语言模型在一些小规模的设置中展示出了较强的词汇翻译和双语词典诱导(BLI)的能力，但在无监督的情况下，即没有种子翻译对可用的情况下，尤其是对于资源较少的语言，它们仍然无法达到“传统”的基于映射的方法的性能。为了解决这个挑战，我们提出了一种自学习上下文增强方法 (SAIL) 来进行无监督的BLI：从零样本提示开始，SAIL通过迭代地从LLM中引出一组高置信度的词汇翻译对，然后在ICL的方式下再次应用于同一个LLM中。我们的方法在两个广泛的BLI基准测试中，跨越多种语言对，在零样本提示的LLM上取得了显著的改进，也在各个方面优于基于映射的基线。除了达到最先进的无监督

    arXiv:2402.10024v1 Announce Type: cross  Abstract: Recent work has shown that, while large language models (LLMs) demonstrate strong word translation or bilingual lexicon induction (BLI) capabilities in few-shot setups, they still cannot match the performance of 'traditional' mapping-based approaches in the unsupervised scenario where no seed translation pairs are available, especially for lower-resource languages. To address this challenge with LLMs, we propose self-augmented in-context learning (SAIL) for unsupervised BLI: starting from a zero-shot prompt, SAIL iteratively induces a set of high-confidence word translation pairs for in-context learning (ICL) from an LLM, which it then reapplies to the same LLM in the ICL fashion. Our method shows substantial gains over zero-shot prompting of LLMs on two established BLI benchmarks spanning a wide range of language pairs, also outperforming mapping-based baselines across the board. In addition to achieving state-of-the-art unsupervised 
    
[^26]: 利用最小描述长度缩小神经网络形式语言学习中的经验-理论差距

    Bridging the Empirical-Theoretical Gap in Neural Network Formal Language Learning Using Minimum Description Length

    [https://arxiv.org/abs/2402.10013](https://arxiv.org/abs/2402.10013)

    通过使用最小描述长度目标（MDL），解决了神经网络在形式语言学习中的经验-理论差距问题。

    

    神经网络在许多任务中提供了良好的近似，但是即使理论工作表明这些完美的解可以由特定的架构来表示，它们仍然无法达到完美的泛化。通过使用形式语言学习的任务，我们专注于一个简单的形式语言，并展示了理论上正确的解决方案实际上不是常用目标函数的最优解，即使使用了正则化技术（如L1，L2）或其他元启发式方法（早停，dropout）。然而，将标准目标替换为最小描述长度目标（MDL）可以使正确的解成为最优解。

    arXiv:2402.10013v1 Announce Type: new  Abstract: Neural networks offer good approximation to many tasks but consistently fail to reach perfect generalization, even when theoretical work shows that such perfect solutions can be expressed by certain architectures. Using the task of formal language learning, we focus on one simple formal language and show that the theoretically correct solution is in fact not an optimum of commonly used objectives -- even with regularization techniques that according to common wisdom should lead to simple weights and good generalization (L1, L2) or other meta-heuristics (early-stopping, dropout). However, replacing standard targets with the Minimum Description Length objective (MDL) results in the correct solution being an optimum.
    
[^27]: LLMs作为桥梁：重新构建基于多模态图像的命名实体识别

    LLMs as Bridges: Reformulating Grounded Multimodal Named Entity Recognition

    [https://arxiv.org/abs/2402.09989](https://arxiv.org/abs/2402.09989)

    本文提出了RiVEG，一个统一的框架，通过利用大型语言模型（LLMs）作为连接桥梁，将多模态命名实体识别重新构建为联合任务，解决了命名实体无法确定和指代表达与命名实体之间的区别的问题。

    

    Grounded Multimodal Named Entity Recognition (GMNER) 是一个新兴的多模态任务，旨在识别命名实体、实体类型及其对应的视觉区域。GMNER任务具有两个挑战性质：1）社交媒体中图像和文本之间的弱相关性导致大部分命名实体难以确定；2）常用于类似任务的粗粒度指代表达与细粒度命名实体之间存在明显区别。本文提出了RiVEG，一个统一的框架，通过利用大型语言模型（LLMs）作为连接桥梁，将GMNER重新构建为联合MNER-VE-VG任务。这种重新构建带来了两个好处：1）保持了最佳的MNER性能，消除了使用目标检测方法预提取区域特征的需求，自然解决了这两个挑战。

    arXiv:2402.09989v1 Announce Type: cross  Abstract: Grounded Multimodal Named Entity Recognition (GMNER) is a nascent multimodal task that aims to identify named entities, entity types and their corresponding visual regions. GMNER task exhibits two challenging properties: 1) The weak correlation between image-text pairs in social media results in a significant portion of named entities being ungroundable. 2) There exists a distinction between coarse-grained referring expressions commonly used in similar tasks (e.g., phrase localization, referring expression comprehension) and fine-grained named entities. In this paper, we propose RiVEG, a unified framework that reformulates GMNER into a joint MNER-VE-VG task by leveraging large language models (LLMs) as a connecting bridge. This reformulation brings two benefits: 1) It maintains the optimal MNER performance and eliminates the need for employing object detection methods to pre-extract regional features, thereby naturally addressing two m
    
[^28]: 语言模型压缩的快速词汇转移方法

    Fast Vocabulary Transfer for Language Model Compression

    [https://arxiv.org/abs/2402.09977](https://arxiv.org/abs/2402.09977)

    提出了一种基于词汇转移的语言模型压缩方法，通过与其他压缩技术结合使用，显著减小模型大小和推理时间，同时性能略有妥协。

    

    实际业务应用需要在语言模型性能和大小之间做出权衡。我们提出了一种基于词汇转移的模型压缩方法。我们在不同垂直领域和下游任务中评估了该方法。我们的结果表明，词汇转移可以与其他压缩技术有效结合使用，显著减小模型大小和推理时间，同时在性能上略有妥协。

    arXiv:2402.09977v1 Announce Type: cross  Abstract: Real-world business applications require a trade-off between language model performance and size. We propose a new method for model compression that relies on vocabulary transfer. We evaluate the method on various vertical domains and downstream tasks. Our results indicate that vocabulary transfer can be effectively used in combination with other compression techniques, yielding a significant reduction in model size and inference time while marginally compromising on performance.
    
[^29]: 案例研究：在某些推理任务中测试模型能力

    Case Study: Testing Model Capabilities in Some Reasoning Tasks

    [https://arxiv.org/abs/2402.09967](https://arxiv.org/abs/2402.09967)

    本研究探讨了大型语言模型（LLMs）的推理能力，在复杂推理场景中的挑战和限制是目前需要改进的关键问题。

    

    大型语言模型（LLMs）在生成个性化内容和促进交互对话方面表现出色，展示了它们在各种应用中的卓越能力。然而，在推理和提供可解释的输出方面，特别是在推理能力的背景下，它们的能力仍然需要改进。在本研究中，我们深入探讨了LLMs的推理能力，重点介绍了目前阻碍它们在复杂推理场景中发挥有效性的挑战和限制。

    arXiv:2402.09967v1 Announce Type: new  Abstract: Large Language Models (LLMs) excel in generating personalized content and facilitating interactive dialogues, showcasing their remarkable aptitude for a myriad of applications. However, their capabilities in reasoning and providing explainable outputs, especially within the context of reasoning abilities, remain areas for improvement. In this study, we delve into the reasoning abilities of LLMs, highlighting the current challenges and limitations that hinder their effectiveness in complex reasoning scenarios.
    
[^30]: 制定良好提示还是提供出色的对话？关于基于上下文学习的角色生成对话的研究

    Crafting a Good Prompt or Providing Exemplary Dialogues? A Study of In-Context Learning for Persona-based Dialogue Generation

    [https://arxiv.org/abs/2402.09954](https://arxiv.org/abs/2402.09954)

    本研究通过对大型语言模型在基于角色生成对话方面进行实验，发现调整提示指令可以最直接有效且经济地提高生成质量，并且随机检索示范会取得最佳结果，而查询相同上下文的示范检索效果最差。即使破坏了示范中的多回合关联和单回合语义，对话生成仍然有效。

    

    先前关于上下文学习（ICL）的研究主要侧重于分类、机器翻译、文本到表格等任务，而对于ICL能否改进生成类似人类对话的研究很少。我们的工作通过在高质量的真实人类对话数据集上进行广泛的实验，系统地研究了大型语言模型（LLMs）在基于角色生成对话方面的ICL能力。根据实验结果，我们得出三个结论：1）调整提示指令是提高生成质量最直接、有效和经济的方法；2）随机检索示范可以取得最佳的结果，可能是因为具有更多样化和有效信息的原因；与查询相同上下文的示范检索结果最差；3）即使破坏了示范中的多回合关联和单回合语义，对话生成仍然可以实现较好的效果。

    arXiv:2402.09954v1 Announce Type: new  Abstract: Previous in-context learning (ICL) research has focused on tasks such as classification, machine translation, text2table, etc., while studies on whether ICL can improve human-like dialogue generation are scarce. Our work fills this gap by systematically investigating the ICL capabilities of large language models (LLMs) in persona-based dialogue generation, conducting extensive experiments on high-quality real human Chinese dialogue datasets. From experimental results, we draw three conclusions: 1) adjusting prompt instructions is the most direct, effective, and economical way to improve generation quality; 2) randomly retrieving demonstrations (demos) achieves the best results, possibly due to the greater diversity and the amount of effective information; counter-intuitively, retrieving demos with a context identical to the query performs the worst; 3) even when we destroy the multi-turn associations and single-turn semantics in the demo
    
[^31]: 多词标记化用于序列压缩

    Multi-Word Tokenization for Sequence Compression

    [https://arxiv.org/abs/2402.09949](https://arxiv.org/abs/2402.09949)

    本论文介绍了一种名为MWT的多词标记器，通过将频繁出现的多词表达式表示为单个标记，突破了词边界的限制，从而实现更紧凑和高效的标记化，提高了性能并加速推理过程。

    

    大型语言模型在建模各种任务方面取得了极大成功。然而，这也意味着计算成本的大幅增加，限制了其在工业界的广泛应用。本论文介绍了一种名为MWT的多词标记器，通过将频繁出现的多词表达式表示为单个标记，突破了词边界的限制。MWT产生了更紧凑和高效的标记化结果，带来两个好处：（1）在固定序列长度和预算的情况下，提高了性能，因为能够更全面地覆盖输入数据；（2）由于能够减少序列长度而对性能几乎没有影响，从而实现更快速和更轻量的推理过程。我们的结果表明，MWT在较短的序列长度下更为稳健，从而可以通过早期序列截断实现重大加速。

    arXiv:2402.09949v1 Announce Type: new  Abstract: Large Language Models have proven highly successful at modelling a variety of tasks. However, this comes at a steep computational cost that hinders wider industrial uptake. In this pa005 per, we present MWT: a Multi-Word Tokenizer that goes beyond word boundaries by representing frequent multi-word expressions as single tokens. MWTs produce a more compact and efficient tokenization that yields two benefits: (1) Increase in performance due to a greater coverage of input data given a fixed sequence length and budget; (2) Faster and lighter inference due to the ability to reduce the sequence length with negligible drops in performance. Our results show that MWT is more robust across shorter sequence lengths, thus allowing for major speedups via early sequence truncation.
    
[^32]: 建筑行业中的生成式人工智能：一项最新分析

    Generative AI in the Construction Industry: A State-of-the-art Analysis

    [https://arxiv.org/abs/2402.09939](https://arxiv.org/abs/2402.09939)

    本研究通过分析提供了建筑行业中生成式AI的最新状态、机遇和挑战。同时，提出了一个帮助建筑公司构建定制化生成式AI解决方案的框架。

    

    建筑行业是全球经济中至关重要的一个部门，但在设计、规划、采购、检查和维护等各个环节中面临着许多生产力挑战。生成式人工智能（AI）可以基于某些输入或先前的知识创造新颖且逼真的数据或内容，如文本、图像、视频或代码，为解决这些挑战提供了创新和颠覆性的解决方案。然而，在关于建筑行业中生成式AI的当前状态、机遇和挑战的文献中存在着空白。本研究旨在通过提供建筑领域生成式AI的最新分析来填补这一空缺，研究目标包括：（1）对建筑行业现有和新兴的生成式AI机遇和挑战进行回顾和分类；（2）提出一个框架，帮助建筑公司利用自己的数据和需求构建定制化的生成式AI解决方案。

    arXiv:2402.09939v1 Announce Type: new  Abstract: The construction industry is a vital sector of the global economy, but it faces many productivity challenges in various processes, such as design, planning, procurement, inspection, and maintenance. Generative artificial intelligence (AI), which can create novel and realistic data or content, such as text, image, video, or code, based on some input or prior knowledge, offers innovative and disruptive solutions to address these challenges. However, there is a gap in the literature on the current state, opportunities, and challenges of generative AI in the construction industry. This study aims to fill this gap by providing a state-of-the-art analysis of generative AI in construction, with three objectives: (1) to review and categorize the existing and emerging generative AI opportunities and challenges in the construction industry; (2) to propose a framework for construction firms to build customized generative AI solutions using their ow
    
[^33]: 关注偏差：挖掘在线话语中的演绎细微差别，检测反问主义

    Paying Attention to Deflections: Mining Pragmatic Nuances for Whataboutism Detection in Online Discourse

    [https://arxiv.org/abs/2402.09934](https://arxiv.org/abs/2402.09934)

    本研究挖掘了在线话语中的演绎细微差别，提出了一种新的方法来准确检测反问主义，并在Twitter和YouTube数据集中取得了显著的改进。

    

    反问主义在扰乱叙事和播种不信任方面具有强大的工具效用，但在定量自然语言处理研究中却未得到充分探索。此外，过去的研究未能区分反问主义作为误导和宣传策略的用途与其作为语用和语义框架工具的用途。我们介绍了新的来自Twitter和YouTube的数据集，揭示了反问主义、宣传和tu quoque谬误之间的重叠和区别。此外，结合最近在语言语义学领域的研究，我们将“what about”词汇结构与反问主义区分开来。我们的实验揭示了准确检测反问主义的独特挑战，促使我们引入了一种使用注意权重进行负样本挖掘的新方法。在Twitter和YouTube数据集中，我们的方法分别相对于之前最先进的方法提高了4%和10%。

    arXiv:2402.09934v1 Announce Type: cross  Abstract: Whataboutism, a potent tool for disrupting narratives and sowing distrust, remains under-explored in quantitative NLP research. Moreover, past work has not distinguished its use as a strategy for misinformation and propaganda from its use as a tool for pragmatic and semantic framing. We introduce new datasets from Twitter and YouTube, revealing overlaps as well as distinctions between whataboutism, propaganda, and the tu quoque fallacy. Furthermore, drawing on recent work in linguistic semantics, we differentiate the `what about' lexical construct from whataboutism. Our experiments bring to light unique challenges in its accurate detection, prompting the introduction of a novel method using attention weights for negative sample mining. We report significant improvements of 4% and 10% over previous state-of-the-art methods in our Twitter and YouTube collections, respectively.
    
[^34]: 一个包含多段答案的开放领域问答数据集

    A Dataset of Open-Domain Question Answering with Multiple-Span Answers

    [https://arxiv.org/abs/2402.09923](https://arxiv.org/abs/2402.09923)

    提出了一个中文多段答案的开放领域问答数据集CLEAN，弥补了中文MSQA研究中的不足，包括多样的主题和需要详细回答的问题。提供了相关文献中的基线模型作为参考。

    

    多段答案提取，也称为多段问答（MSQA）任务，在实际应用中至关重要，因为它需要从文本中提取多个信息片段来回答复杂的问题。尽管英文MSQA研究活跃并取得了快速进展，但在中文领域缺乏公开可用的MSQA基准数据集。以往构建MSQA数据集的努力主要强调实体中心的情境化，导致偏向收集事实性问题并可能忽视需要更详细描述性回答的问题。为了克服这些限制，我们提供了CLEAN，一个全面的中文多段问答数据集，涉及各种开放领域的主题，并包含大量需要描述性答案的实例。此外，我们还提供了相关文献中的已建立模型作为CLEAN的基线。

    arXiv:2402.09923v1 Announce Type: cross  Abstract: Multi-span answer extraction, also known as the task of multi-span question answering (MSQA), is critical for real-world applications, as it requires extracting multiple pieces of information from a text to answer complex questions. Despite the active studies and rapid progress in English MSQA research, there is a notable lack of publicly available MSQA benchmark in Chinese. Previous efforts for constructing MSQA datasets predominantly emphasized entity-centric contextualization, resulting in a bias towards collecting factoid questions and potentially overlooking questions requiring more detailed descriptive responses. To overcome these limitations, we present CLEAN, a comprehensive Chinese multi-span question answering dataset that involves a wide range of open-domain subjects with a substantial number of instances requiring descriptive answers. Additionally, we provide established models from relevant literature as baselines for CLEA
    
[^35]: BUSTER:一份“商业交易实体识别”数据集

    BUSTER: a "BUSiness Transaction Entity Recognition" dataset

    [https://arxiv.org/abs/2402.09916](https://arxiv.org/abs/2402.09916)

    BUSTER是一个商业交易实体识别的数据集，其中包含了3779份手动标注的金融交易文档，并建立了几个基准模型。最佳模型还用于自动标注6196份文档，并作为额外的银标准数据集发布。

    

    尽管自然语言处理在过去几年取得了重大突破，但将这些进展转化为实际的商业案例仍然具有挑战性。其中一个原因在于流行的基准数据与实际数据之间的差异。缺乏监督、类别不平衡、噪声数据和长文档经常影响金融、法律和健康等垂直领域的实际问题。为了支持面向行业的研究，我们提供了一个名为BUSTER的“商业交易实体识别”数据集。该数据集包含了3779份手动标注的金融交易文档。我们建立了几个基准模型，利用了通用和领域特定的语言模型。表现最佳的模型还被用于自动标注6196份文档，我们将其作为额外的银标准数据集发布给BUSTER。

    arXiv:2402.09916v1 Announce Type: new  Abstract: Albeit Natural Language Processing has seen major breakthroughs in the last few years, transferring such advances into real-world business cases can be challenging. One of the reasons resides in the displacement between popular benchmarks and actual data. Lack of supervision, unbalanced classes, noisy data and long documents often affect real problems in vertical domains such as finance, law and health. To support industry-oriented research, we present BUSTER, a BUSiness Transaction Entity Recognition dataset. The dataset consists of 3779 manually annotated documents on financial transactions. We establish several baselines exploiting both general-purpose and domain-specific language models. The best performing model is also used to automatically annotate 6196 documents, which we release as an additional silver corpus to BUSTER.
    
[^36]: 使用伪和多源知识图增强大型语言模型进行开放式问题回答

    Enhancing Large Language Models with Pseudo- and Multisource- Knowledge Graphs for Open-ended Question Answering

    [https://arxiv.org/abs/2402.09911](https://arxiv.org/abs/2402.09911)

    使用伪和多源知识图对大型语言模型进行增强，以改善其幻觉问题和提高性能。通过结合伪图生成和原子级知识验证的框架，在开放式问题回答环境中使用知识图可以提高ROUGE-L分数至少11.5。

    

    减轻大型语言模型（LLM）的幻觉并增强它们是一项关键任务。尽管一些现有方法采用了模型自我增强技术，但它们在有效解决未知事实幻觉方面存在不足。使用知识图（KG）增强方法无法同时解决不同KG来源之间的泛化和开放式答案问题的增强。为了解决这些限制，提出了一种结合了伪图生成和原子级知识验证的框架。通过利用伪图生成来实现在开放式问题回答环境中使用KG增强LLM。原子级知识验证利用原子级知识查询和验证来实现在不同KG来源下的泛化能力。与基准相比，该方法在ROUGE-L分数上至少提升了11.5。

    arXiv:2402.09911v1 Announce Type: cross  Abstract: Mitigating the hallucinations of Large Language Models (LLMs) and enhancing them is a crucial task. Although some existing methods employ model self-enhancement techniques, they fall short of effectively addressing unknown factual hallucinations. Using Knowledge Graph (KG) enhancement approaches fails to address the generalization across different KG sources and the enhancement of open-ended answer questions simultaneously. To tackle these limitations, there is a framework that combines Pseudo-Graph Generation and Atomic Knowledge Verification proposed. The enhancement of LLM using KG in an open-ended question-answering setting is implemented by leveraging the Pseudo-Graph Generation. Atomic Knowledge Verification utilizes atomic-level knowledge querying and verification to achieve generalizability under different KG sources. Compared to the baseline, this approach yields a minimum improvement of 11.5 in the ROUGE-L score for open-ende
    
[^37]: 在语言模型训练数据中检测版权内容的方法：DE-COP

    DE-COP: Detecting Copyrighted Content in Language Models Training Data

    [https://arxiv.org/abs/2402.09910](https://arxiv.org/abs/2402.09910)

    DE-COP是一种用于检测语言模型训练数据中版权内容的方法，通过对语言模型进行多项选择探测，可以识别出模型训练文本中可能包含的版权内容。该方法在模型的逻辑可用时比之前的方法提高了9.6%的检测性能，并在完全黑盒模型上实现了72%的准确率。

    

    在考虑到训练数据通常是保密的情况下，我们如何检测语言模型的训练过程中是否使用了版权内容？我们的动机是基于一个语言模型很可能能够识别出其训练文本中的独文摘录。我们提出了一种称为DE-COP的方法，用于确定是否在训练中包含了一段版权内容。DE-COP的核心方法是通过多项选择问题对语言模型进行探测，选择项包括独文本和它们的释义。我们构建了一个基准数据集BookTection，其中包含了在模型训练截止日期之前和之后出版的165本书的摘录以及它们的释义。实验证明，DE-COP在模型的逻辑可用时，检测性能（AUC）超过之前的最佳方法9.6%。此外，DE-COP在完全黑盒模型上检测可疑书籍的平均准确率达到72%，而之前的方法只有$。

    arXiv:2402.09910v1 Announce Type: new  Abstract: How can we detect if copyrighted content was used in the training process of a language model, considering that the training data is typically undisclosed? We are motivated by the premise that a language model is likely to identify verbatim excerpts from its training text. We propose DE-COP, a method to determine whether a piece of copyrighted content was included in training. DE-COP's core approach is to probe an LLM with multiple-choice questions, whose options include both verbatim text and their paraphrases. We construct BookTection, a benchmark with excerpts from 165 books published prior and subsequent to a model's training cutoff, along with their paraphrases. Our experiments show that DE-COP surpasses the prior best method by 9.6% in detection performance (AUC) on models with logits available. Moreover, DE-COP also achieves an average accuracy of 72% for detecting suspect books on fully black-box models where prior methods give $
    
[^38]: 生成表示指令调整

    Generative Representational Instruction Tuning

    [https://arxiv.org/abs/2402.09906](https://arxiv.org/abs/2402.09906)

    本研究引入了生成表示指令调整（GRIT）方法，通过指令区分生成和嵌入任务，训练一个大型语言模型同时处理这两种任务。与其他模型相比，我们的GritLM 7B在文本嵌入基准测试上达到最新的技术水平，并在多种生成任务中表现出色。通过进一步扩大规模，我们的GritLM 8x7B成为最佳的生成语言模型之一，同时仍然是最好的嵌入模型之一。GRIT的统一也大大提高了RAG在长文档上的速度。

    

    所有基于文本的语言问题都可以归结为生成或嵌入。目前的模型只能在其中一种任务上表现良好。我们介绍了生成表示指令调整（GRIT）方法，通过指令来区分生成和嵌入任务，从而训练一个大型语言模型同时处理这两种任务。与其他开放模型相比，我们的GritLM 7B在大规模文本嵌入基准测试（MTEB）上取得了最新的技术水平，并在多种生成任务中超过了同等规模的所有模型。通过进一步扩大规模，GritLM 8x7B在尝试的所有开放生成语言模型中表现最佳，同时仍然是最好的嵌入模型之一。值得注意的是，我们发现GRIT可以与仅在生成或嵌入数据上训练的模型相媲美，因此我们可以在不损失性能的情况下统一两者。除此之外，通过GRIT的统一可以将RAG（检索增强生成）在长文档上的速度提高60%以上。

    arXiv:2402.09906v1 Announce Type: cross  Abstract: All text-based language problems can be reduced to either generation or embedding. Current models only perform well at one or the other. We introduce generative representational instruction tuning (GRIT) whereby a large language model is trained to handle both generative and embedding tasks by distinguishing between them through instructions. Compared to other open models, our resulting GritLM 7B sets a new state of the art on the Massive Text Embedding Benchmark (MTEB) and outperforms all models up to its size on a range of generative tasks. By scaling up further, GritLM 8x7B outperforms all open generative language models that we tried while still being among the best embedding models. Notably, we find that GRIT matches training on only generative or embedding data, thus we can unify both at no performance loss. Among other benefits, the unification via GRIT speeds up Retrieval-Augmented Generation (RAG) by > 60% for long documents, 
    
[^39]: 不仅仅是新颖性：关于AI工作流程的效用和定制化的纵向研究

    Not Just Novelty: A Longitudinal Study on Utility and Customization of AI Workflows

    [https://arxiv.org/abs/2402.09894](https://arxiv.org/abs/2402.09894)

    这项纵向研究调查了生成式AI工作流程的实用性和定制化程度，结果显示，在熟悉化阶段后，用户感知到的系统效用提高了。

    

    生成式AI为人们在日常任务中提供了新颖而令人印象深刻的能力。有许多AI工作流程通过将AI输出与人类互动相结合来解决真实而复杂的问题。尽管AI具有无可否认的吸引力，但在新鲜感消失后，生成式AI工作流程的实用性如何仍然不确定。此外，利用生成式AI构建的工具具有个性化和快速适应的潜力，但用户是否充分利用了个性化的可能性呢？我们进行了一项为期三周的纵向研究，共有12个用户，旨在了解科学传播中生成式AI工具的熟悉度和定制化程度。我们的研究发现，熟悉化阶段持续了4.3个会话，用户在这个阶段探索工作流程的功能以及他们发现哪些方面有用。在熟悉化后，系统的感知效用评分高于之前，表明了感知效用的提高。

    arXiv:2402.09894v1 Announce Type: cross  Abstract: Generative AI brings novel and impressive abilities to help people in everyday tasks. There are many AI workflows that solve real and complex problems by chaining AI outputs together with human interaction. Although there is an undeniable lure of AI, it's uncertain how useful generative AI workflows are after the novelty wears off. Additionally, tools built with generative AI have the potential to be personalized and adapted quickly and easily, but do users take advantage of the potential to customize? We conducted a three-week longitudinal study with 12 users to understand the familiarization and customization of generative AI tools for science communication. Our study revealed that the familiarization phase lasts for 4.3 sessions, where users explore the capabilities of the workflow and which aspects they find useful. After familiarization, the perceived utility of the system is rated higher than before, indicating that the perceived
    
[^40]: 伪装就是你所需要的：评估和增强语言模型对伪装性对抗攻击的鲁棒性

    Camouflage is all you need: Evaluating and Enhancing Language Model Robustness Against Camouflage Adversarial Attacks

    [https://arxiv.org/abs/2402.09874](https://arxiv.org/abs/2402.09874)

    该研究评估和增强了语言模型对伪装性对抗攻击的鲁棒性。在对三种Transformer配置进行评估后发现，仅编码器模型在冒犯性语言检测和虚假信息检测任务中表现出最高的性能下降。

    

    对抗攻击在自然语言处理（NLP）中构成了一个重大挑战。本研究对Transformer-based模型在对抗攻击下的脆弱性评估和鲁棒性增强进行了系统的探索。在评估阶段，我们评估了包含冒犯性语言和虚假信息的数据集上，三个Transformer配置（编码器-解码器、仅编码器和仅解码器）对不断升级的复杂性的对抗攻击的敏感性。仅编码器模型在冒犯性语言检测和虚假信息检测任务中分别表现出14%和21%的性能下降。仅解码器模型在两个任务中都出现了16%的降低，而编码器-解码器模型在相应任务中的最大性能下降为14%和26%。

    arXiv:2402.09874v1 Announce Type: new  Abstract: Adversarial attacks represent a substantial challenge in Natural Language Processing (NLP). This study undertakes a systematic exploration of this challenge in two distinct phases: vulnerability evaluation and resilience enhancement of Transformer-based models under adversarial attacks.   In the evaluation phase, we assess the susceptibility of three Transformer configurations, encoder-decoder, encoder-only, and decoder-only setups, to adversarial attacks of escalating complexity across datasets containing offensive language and misinformation. Encoder-only models manifest a 14% and 21% performance drop in offensive language detection and misinformation detection tasks, respectively. Decoder-only models register a 16% decrease in both tasks, while encoder-decoder models exhibit a maximum performance drop of 14% and 26% in the respective tasks.   The resilience-enhancement phase employs adversarial training, integrating pre-camouflaged an
    
[^41]: LAPDoc：面向文档的布局感知提示

    LAPDoc: Layout-Aware Prompting for Documents

    [https://arxiv.org/abs/2402.09841](https://arxiv.org/abs/2402.09841)

    本文研究了通过使用布局增强来使用纯文本LLM进行文档特定任务的可能性。

    

    最近，使用大量纯文本数据训练大型语言模型(LLM)取得了重大突破，在许多领域和任务中实现了强大的泛化能力，包括文档特定任务。相比之下，训练针对文档理解的多模态变压器体系结构，专门设计用于将文本输入与相应的文档布局融合。这需要单独的微调步骤，需要额外的训练数据。目前，尚没有具有与LLM相当泛化能力的文档变压器可用。这引发了一个问题，即在文档理解任务中应该选择哪种类型的模型。在本文中，我们通过使用布局增强来调查使用纯文本LLM用于文档特定任务的可能性。我们探索了添加修改和基于规则的方法，以在纯文本LLM提示中添加布局信息。

    arXiv:2402.09841v1 Announce Type: new  Abstract: Recent advances in training large language models (LLMs) using massive amounts of solely textual data lead to strong generalization across many domains and tasks, including document-specific tasks. Opposed to that there is a trend to train multi-modal transformer architectures tailored for document understanding that are designed specifically to fuse textual inputs with the corresponding document layout. This involves a separate fine-tuning step for which additional training data is required. At present, no document transformers with comparable generalization to LLMs are available That raises the question which type of model is to be preferred for document understanding tasks. In this paper we investigate the possibility to use purely text-based LLMs for document-specific tasks by using layout enrichment. We explore drop-in modifications and rule-based methods to enrich purely textual LLM prompts with layout information. In our experimen
    
[^42]: 预训练语言模型对于标记的表面信息的知识

    Knowledge of Pretrained Language Models on Surface Information of Tokens

    [https://arxiv.org/abs/2402.09808](https://arxiv.org/abs/2402.09808)

    该研究探究了预训练语言模型对于标记的表面信息的知识。结果显示，模型具备有关标记长度和子字符串的知识，但对标记结构的知识有限，解码器方面存在瓶颈。

    

    我们研究了预训练语言模型是否具有关于标记表面信息的知识。我们从标记长度、子字符串和标记结构的角度检查了预训练语言模型获取的词语或子词嵌入中保存的表面信息。此外，我们评估了模型生成标记表面知识的能力。我们主要关注了12个主要在英语和日语语料库上训练的预训练语言模型。实验结果表明，预训练语言模型对于标记长度和子字符串具有知识，但对于标记结构则没有知识。此外，结果表明，在有效利用获取的知识方面，解码器方面存在瓶颈。

    arXiv:2402.09808v1 Announce Type: new  Abstract: Do pretrained language models have knowledge regarding the surface information of tokens? We examined the surface information stored in word or subword embeddings acquired by pretrained language models from the perspectives of token length, substrings, and token constitution. Additionally, we evaluated the ability of models to generate knowledge regarding token surfaces. We focused on 12 pretrained language models that were mainly trained on English and Japanese corpora. Experimental results demonstrate that pretrained language models have knowledge regarding token length and substrings but not token constitution. Additionally, the results imply that there is a bottleneck on the decoder side in terms of effectively utilizing acquired knowledge.
    
[^43]: EFUF: 高效精细化去学习多模态大语言模型中减轻幻像的框架

    EFUF: Efficient Fine-grained Unlearning Framework for Mitigating Hallucinations in Multimodal Large Language Models

    [https://arxiv.org/abs/2402.09801](https://arxiv.org/abs/2402.09801)

    EFUF是一种高效精细化去学习框架，可以消除多模态大语言模型中的物体幻觉，并不需要人工注释配对数据。

    

    多模态大语言模型（MLLMs）近年来受到越来越多的关注，但它们可能仍会生成包含图像中不存在的物体的描述，这种现象称为物体幻觉。为了消除幻觉，现有方法手动注释包含和不包含幻觉的配对响应，并采用各种对齐算法来提高图像和文本之间的对齐能力。然而，它们不仅在微调阶段需要大量计算资源，还需要昂贵的人工注释来构建对齐算法所需的配对数据。为了解决这些问题，我们借鉴了去学习的思想，提出了一种高效精细化去学习框架（EFUF），可以在不需要配对数据的情况下消除幻觉。大量实验证明，我们的方法能够持续减少幻觉同时保留准确的描述。

    arXiv:2402.09801v1 Announce Type: new  Abstract: Multimodal large language models (MLLMs) have attracted increasing attention in the past few years, but they may still generate descriptions that include objects not present in the corresponding images, a phenomenon known as object hallucination. To eliminate hallucinations, existing methods manually annotate paired responses with and without hallucinations, and then employ various alignment algorithms to improve the alignment capability between images and text. However, they not only demand considerable computation resources during the finetuning stage but also require expensive human annotation to construct paired data needed by the alignment algorithms. To address these issues, we borrow the idea of unlearning and propose an efficient fine-grained unlearning framework (EFUF), which can eliminate hallucinations without the need for paired data. Extensive experiments show that our method consistently reduces hallucinations while preserv
    
[^44]: NutePrune: 高效的大型语言模型逐渐剪枝方法，多个教师参与

    NutePrune: Efficient Progressive Pruning with Numerous Teachers for Large Language Models

    [https://arxiv.org/abs/2402.09773](https://arxiv.org/abs/2402.09773)

    NutePrune是一种高效逐渐剪枝方法，通过加载一个完整模型并将其与掩码和LoRA模块集成，实现了在大型语言模型中进行高效的结构剪枝。

    

    大型语言模型（LLMs）的巨大尺寸给资源受限硬件带来了显著的部署挑战。结构剪枝为压缩LLMs提供了一种有效的方式，从而降低存储成本，提升推断速度，实现更高效的利用。在这项工作中，我们研究了数据效率和资源效率的结构剪枝方法，以获取更小但依然强大的模型。知识蒸馏非常适合剪枝，因为完整的模型可以作为剪枝后的学生的优秀教师。然而，在LLMs的背景下，由于内存限制，这变得具有挑战性。为了解决这个问题，我们提出了一种高效逐渐剪枝方法（NutePrune）。NutePrune通过只加载一个完整模型并将其与各种掩码和LoRA模块集成，在教师和学生角色之间无缝切换，从而减轻了过多的内存开销。

    arXiv:2402.09773v1 Announce Type: new  Abstract: The considerable size of Large Language Models (LLMs) presents notable deployment challenges, particularly on resource-constrained hardware. Structured pruning, offers an effective means to compress LLMs, thereby reducing storage costs and enhancing inference speed for more efficient utilization. In this work, we study data-efficient and resource-efficient structure pruning methods to obtain smaller yet still powerful models. Knowledge Distillation is well-suited for pruning, as the intact model can serve as an excellent teacher for pruned students. However, it becomes challenging in the context of LLMs due to memory constraints. To address this, we propose an efficient progressive Numerous-teacher pruning method (NutePrune). NutePrune mitigates excessive memory costs by loading only one intact model and integrating it with various masks and LoRA modules, enabling it to seamlessly switch between teacher and student roles. This approach a
    
[^45]: 无块语境检索的语言模型 grounding

    Grounding Language Model with Chunking-Free In-Context Retrieval

    [https://arxiv.org/abs/2402.09760](https://arxiv.org/abs/2402.09760)

    这是一种针对检索增强生成系统（RAG）的无块语境检索方法，通过绕过文本切分的过程，利用编码隐藏状态进行准确地语境检索，解决了传统方法中存在的文本语义连贯性破坏和证据检索中的噪声和不准确性问题。

    

    本论文介绍了一种针对检索增强生成（RAG）系统的无块语境（CFIC）检索方法。传统的RAG系统在使用精确证据文本进行 grounding 时往往面临处理冗长文档和过滤无关内容的挑战。常用的解决方案，如文档切分和调整语言模型以处理更长的上下文，都存在局限性。这些方法要么破坏了文本的语义连贯性，要么未能有效解决证据检索中的噪声和不准确性问题。CFIC通过绕过传统的切分过程来应对这些挑战。它利用文档的编码隐藏状态进行语境检索，在对用户查询进行自回归解码时准确地识别出所需的具体证据文本，消除了切分的需求。CFIC 进一步。。。

    arXiv:2402.09760v1 Announce Type: cross  Abstract: This paper presents a novel Chunking-Free In-Context (CFIC) retrieval approach, specifically tailored for Retrieval-Augmented Generation (RAG) systems. Traditional RAG systems often struggle with grounding responses using precise evidence text due to the challenges of processing lengthy documents and filtering out irrelevant content. Commonly employed solutions, such as document chunking and adapting language models to handle longer contexts, have their limitations. These methods either disrupt the semantic coherence of the text or fail to effectively address the issues of noise and inaccuracy in evidence retrieval.   CFIC addresses these challenges by circumventing the conventional chunking process. It utilizes the encoded hidden states of documents for in-context retrieval, employing auto-aggressive decoding to accurately identify the specific evidence text required for user queries, eliminating the need for chunking. CFIC is further
    
[^46]: 高效的语言自适应预训练：扩展最新的大规模语言模型用于波兰语

    Efficient Language Adaptive Pre-training: Extending State-of-the-Art Large Language Models for Polish

    [https://arxiv.org/abs/2402.09759](https://arxiv.org/abs/2402.09759)

    本研究使用高效的语言自适应预训练方法，成功将基础英文大规模语言模型应用于生成波兰文，并在困惑度和任务表现上取得了显著的改进，为向现有语言模型添加新语言开辟了新途径。

    

    本研究探讨了将基础英文大规模语言模型（LLMs）微调为生成波兰文的潜力。首先，通过对3.11 GB高质量数据集进行语言自适应预训练（LAPT），该数据集包含2.76亿个波兰语tokens。LAPT后进行了额外的微调，旨在解决九个KLEJ挑战。我们训练的模型Curie-7B-v1不仅在基于解码器的波兰模型中具有最低的困惑度3.02，而且在8个任务中与最好的波兰编码器-解码器模型之间的差距不到2%。Curie-7B-v1仅使用典型数据集大小的约2-3%来学习波兰语。LAPT在不到五天的时间内使用普通GPU完成，凸显了该方法的高效性。模型在波兰语方面的熟练度显著提高，证明了该方法在将新语言添加到现有LLMs中的可行性。

    arXiv:2402.09759v1 Announce Type: cross  Abstract: This study explores the potential of fine-tuning foundational English Large Language Models (LLMs) for generating Polish text. The first step involves Language Adaptive Pre-training (LAPT) on a high-quality dataset of 3.11 GB, consisting of 276 million Polish tokens. The LAPT is followed by additional fine-tuning aimed at solving nine KLEJ challenges. Our trained model Curie-7B-v1 not only generates Polish text with the lowest perplexity of 3.02 among decoder-based Polish models but also closely rivals the performance of the best Polish encoder-decoder models with a less than 2% gap on 8 out of 9 tasks. Curie-7B-v1 used approximately 2-3% of a typical dataset size to learn Polish. The LAPT was completed in less than five days using a consumer GPU, highlighting the method's efficiency. The proficiency of the model in Polish was significantly enhanced, demonstrating the viability of this approach for adding new languages to existing LLMs
    
[^47]: 大规模语言模型的模型压缩和高效推理：一项综述

    Model Compression and Efficient Inference for Large Language Models: A Survey

    [https://arxiv.org/abs/2402.09748](https://arxiv.org/abs/2402.09748)

    这项综述研究了大规模语言模型的压缩和高效推理方法，包括量化、修剪、蒸馏、紧凑架构设计和动态网络等方面。大模型的突出特点是压缩后需要微调或重新训练，并且相关的成本很高。

    

    基于Transformer的大规模语言模型取得了巨大的成功。然而，在推理过程中所产生的显著的内存和计算成本使得在资源受限设备上部署大模型变得具有挑战性。本文从算法的角度探讨了大规模语言模型的压缩和高效推理方法。在分类方面，与较小的模型类似，用于大规模语言模型的压缩和加速算法仍可以分为量化、修剪、蒸馏、紧凑架构设计和动态网络。然而，与较小的模型相比，大规模语言模型有两个突出的特点：（1）大多数压缩算法在压缩后需要微调甚至重新训练模型。大模型最显著的方面是与模型微调或训练相关的非常高的成本。因此，许多针对大规模模型的算法都需要考虑这一点。

    arXiv:2402.09748v1 Announce Type: cross  Abstract: Transformer based large language models have achieved tremendous success. However, the significant memory and computational costs incurred during the inference process make it challenging to deploy large models on resource-constrained devices. In this paper, we investigate compression and efficient inference methods for large language models from an algorithmic perspective. Regarding taxonomy, similar to smaller models, compression and acceleration algorithms for large language models can still be categorized into quantization, pruning, distillation, compact architecture design, dynamic networks. However, Large language models have two prominent characteristics compared to smaller models: (1) Most of compression algorithms require finetuning or even retraining the model after compression. The most notable aspect of large models is the very high cost associated with model finetuning or training. Therefore, many algorithms for large mode
    
[^48]: AI医院：用于临床诊断的LLMs作为实习医生的交互式评估和协作

    AI Hospital: Interactive Evaluation and Collaboration of LLMs as Intern Doctors for Clinical Diagnosis

    [https://arxiv.org/abs/2402.09742](https://arxiv.org/abs/2402.09742)

    AI医院是一个框架，用于构建实时交互式诊断环境，通过与LLMs的交互评估和协作，提高临床诊断的准确性。

    

    引入大型语言模型（LLMs）在医疗保健中的应用标志着重大的进展。然而，目前的应用主要局限于辨别和问答任务，没有充分发挥其交互潜力。为了解决这个局限，我们的论文提出了AI医院，一个旨在构建实时交互式诊断环境的框架。为了模拟过程，我们收集高质量的医疗记录，创建了患者、检查者和医疗主任代理。然后，利用AI医院进行LLMs的交互评估和协作。初始阶段，我们创建了一个多视图医学评估（MVME）基准，其中各种LLMs作为实习医生进行交互式诊断。随后，为了提高诊断准确性，我们引入了一种协作机制，涉及医疗主任的监督下的迭代讨论和争议解决过程。

    arXiv:2402.09742v1 Announce Type: new  Abstract: The incorporation of Large Language Models (LLMs) in healthcare marks a significant advancement. However, the application has predominantly been limited to discriminative and question-answering tasks, which does not fully leverage their interactive potential. To address this limitation, our paper presents AI Hospital, a framework designed to build a real-time interactive diagnosis environment. To simulate the procedure, we collect high-quality medical records to create patient, examiner, and medical director agents. AI Hospital is then utilized for the interactive evaluation and collaboration of LLMs. Initially, we create a Multi-View Medical Evaluation (MVME) benchmark where various LLMs serve as intern doctors for interactive diagnosis. Subsequently, to improve diagnostic accuracy, we introduce a collaborative mechanism that involves iterative discussions and a dispute resolution process under the supervision of the medical director. I
    
[^49]: 选择高质量数据用于训练语言模型的QuRating方法

    QuRating: Selecting High-Quality Data for Training Language Models

    [https://arxiv.org/abs/2402.09739](https://arxiv.org/abs/2402.09739)

    QuRating是一种选择高质量数据用于训练语言模型的方法，它能够捕捉人类直观感知的文本的抽象特征。在实验中发现，平衡质量和多样性是很重要的。

    

    选择高质量的预训练数据对于创建能力强的语言模型很重要，但现有方法依赖简单的启发式方法。我们介绍了一种名为QuRating的方法，用于选择能够捕捉人类直观感知的文本的抽象特征的预训练文本数据。在本文中，我们研究了四个特征 - 写作风格、所需专业知识、事实和琐事以及教育价值。我们发现，语言模型能够辨别这些特征，并观察到它们在进行文本的配对判断方面比直接评估文本质量更好。我们训练了一个QuRater模型，从配对判断中学习标量评分，并使用它为260B的训练语料库中的每个标准进行质量评级注释。在实验中，我们根据不同的质量评级选择了30B个令牌，并在所选数据上训练了13亿参数的语言模型。我们发现在质量和多样性之间保持平衡是很重要的。

    arXiv:2402.09739v1 Announce Type: new  Abstract: Selecting high-quality pre-training data is important for creating capable language models, but existing methods rely on simple heuristics. We introduce QuRating, a method for selecting pre-training data that captures the abstract qualities of texts which humans intuitively perceive. In this paper, we investigate four qualities - writing style, required expertise, facts & trivia, and educational value. We find that LLMs are able to discern these qualities and observe that they are better at making pairwise judgments of texts than at rating the quality of a text directly. We train a QuRater model to learn scalar ratings from pairwise judgments, and use it to annotate a 260B training corpus with quality ratings for each of the four criteria. In our experiments, we select 30B tokens according to the different quality ratings and train 1.3B-parameter language models on the selected data. We find that it is important to balance quality and di
    
[^50]: 在注意之前进行对齐：用于多模态恶意内容检测的视觉和文本特征对齐

    Align before Attend: Aligning Visual and Textual Features for Multimodal Hateful Content Detection

    [https://arxiv.org/abs/2402.09738](https://arxiv.org/abs/2402.09738)

    本论文提出了一种上下文感知的注意力框架，用于在视觉和文本特征之间进行对齐，以有效地进行多模态恶意内容检测。该方法能够选择性地关注模态特定的特征，解决了传统的融合技术无法有效关注模态特定特征的问题。研究在英语和非英语语言上进行了评估。

    

    多模态恶意内容检测是一个具有挑战性的任务，需要对视觉和文本模态进行复杂推理。因此，通过中间融合有效地捕捉视觉和文本特征之间的相互作用，创建有意义的多模态表示是至关重要的。传统的融合技术无法有效地关注模态特定的特征。此外，大多数研究仅关注英语，忽视了其他低资源语言。本文提出了一种上下文感知的注意力框架，用于多模态恶意内容检测，并对英语和非英语语言进行了评估。所提出的方法将注意力层引入到视觉和文本特征之间进行有意义的对齐。这种对齐使得在融合之前可以有选择性地关注模态特定的特征。我们在两个基准恶意模因数据集上评估了所提出的方法，即MUTE（Ben）

    arXiv:2402.09738v1 Announce Type: new  Abstract: Multimodal hateful content detection is a challenging task that requires complex reasoning across visual and textual modalities. Therefore, creating a meaningful multimodal representation that effectively captures the interplay between visual and textual features through intermediate fusion is critical. Conventional fusion techniques are unable to attend to the modality-specific features effectively. Moreover, most studies exclusively concentrated on English and overlooked other low-resource languages. This paper proposes a context-aware attention framework for multimodal hateful content detection and assesses it for both English and non-English languages. The proposed approach incorporates an attention layer to meaningfully align the visual and textual features. This alignment enables selective focus on modality-specific features before fusing them. We evaluate the proposed approach on two benchmark hateful meme datasets, viz. MUTE (Ben
    
[^51]: LLM是否了解幻觉？对LLMs隐藏状态的实证调查

    Do LLMs Know about Hallucination? An Empirical Investigation of LLM's Hidden States

    [https://arxiv.org/abs/2402.09733](https://arxiv.org/abs/2402.09733)

    本研究探索了大型语言模型（LLMs）对幻觉的意识程度，通过实验发现LLMs在处理真实回答和虚构回答时有不同的反应，并展示了利用LLMs隐藏状态的指导具有巨大潜力。

    

    大型语言模型（LLMs）可能会虚构出不真实的答案，这就是所谓的幻觉。本研究旨在观察LLMs是否意识到幻觉以及其程度。更具体地说，我们检查了LLMs在其隐藏状态下对正确回答和幻觉回答的不同反应。为了实现这一目标，我们引入了一个实验框架，允许检查LLMs在不同幻觉情况下的隐藏状态。基于这个框架，我们进行了一系列实验，使用LLaMA家族的语言模型（Touvron等，2023）。我们的实证发现表明，LLMs在处理真实回答和虚构回答时有不同的反应。然后，我们应用各种模型解释技术来更好地理解和解释这些发现。此外，根据实证观察结果，我们展示了利用从LLMs隐藏状态中得到的指导的巨大潜力。

    arXiv:2402.09733v1 Announce Type: new  Abstract: Large Language Models (LLMs) can make up answers that are not real, and this is known as hallucination. This research aims to see if, how, and to what extent LLMs are aware of hallucination. More specifically, we check whether and how an LLM reacts differently in its hidden states when it answers a question right versus when it hallucinates. To do this, we introduce an experimental framework which allows examining LLM's hidden states in different hallucination situations. Building upon this framework, we conduct a series of experiments with language models in the LLaMA family (Touvron et al., 2023). Our empirical findings suggest that LLMs react differently when processing a genuine response versus a fabricated one. We then apply various model interpretation techniques to help understand and explain the findings better. Moreover, informed by the empirical observations, we show great potential of using the guidance derived from LLM's hidd
    
[^52]: 一种具有长期上下文概要记忆的人工智能阅读代理

    A Human-Inspired Reading Agent with Gist Memory of Very Long Contexts

    [https://arxiv.org/abs/2402.09727](https://arxiv.org/abs/2402.09727)

    ReadAgent是一个具有长期上下文概要记忆的阅读代理系统，通过实现一个简单的提示系统，它能够处理长输入并提高有效上下文长度。在评估中表现良好。

    

    当前的大型语言模型不仅限制在某个最大上下文长度内，而且无法稳定地处理长输入。为了解决这些限制，我们提出了ReadAgent，一个增加了有效上下文长度的语言模型代理系统，在我们的实验中可以达到20倍。受到人类交互式阅读长文档的启发，我们将ReadAgent实现为一个简单的提示系统，利用LLM的高级语言能力来：（1）决定将哪些内容存储在一个记忆片段中，（2）将这些记忆片段压缩成为称为概要记忆的短时记忆，（3）在需要时通过原始文本查找段落来提醒自己相关细节以完成任务。我们使用检索方法、使用原始长上下文以及使用概要记忆来评估ReadAgent与基线的性能。这些评估是在三个长文档阅读理解任务上进行的。

    arXiv:2402.09727v1 Announce Type: cross  Abstract: Current Large Language Models (LLMs) are not only limited to some maximum context length, but also are not able to robustly consume long inputs. To address these limitations, we propose ReadAgent, an LLM agent system that increases effective context length up to 20x in our experiments. Inspired by how humans interactively read long documents, we implement ReadAgent as a simple prompting system that uses the advanced language capabilities of LLMs to (1) decide what content to store together in a memory episode, (2) compress those memory episodes into short episodic memories called gist memories, and (3) take actions to look up passages in the original text if ReadAgent needs to remind itself of relevant details to complete a task. We evaluate ReadAgent against baselines using retrieval methods, using the original long contexts, and using the gist memories. These evaluations are performed on three long-document reading comprehension task
    
[^53]: 通过错误暴露和一致性正则化改进非自回归机器翻译

    Improving Non-autoregressive Machine Translation with Error Exposure and Consistency Regularization

    [https://arxiv.org/abs/2402.09725](https://arxiv.org/abs/2402.09725)

    本论文提出使用错误暴露和一致性正则化的训练方法来改进非自回归机器翻译中的数据分布不一致问题，并取得了显著的BLEU得分提升。

    

    作为IR-NAT（基于迭代改进的NAT）框架之一，条件掩码语言模型（CMLM）采用掩码预测范式来重新预测掩码低置信度的标记。然而，CMLM在训练和推理之间存在数据分布不一致的问题，观察到的标记在这两种情况下生成方式不同。本文提出使用错误暴露和一致性正则化（EECR）的训练方法来解决这个问题。我们在训练过程中基于模型预测构建混合序列，并提出在不完美观测条件下针对掩码标记进行优化。我们还设计了一种一致性学习方法，以限制不同观测情况下掩码标记的数据分布，缩小训练和推理之间的差距。在五个翻译基准上的实验证明，平均改进了0.68和0.40的BLEU得分。

    arXiv:2402.09725v1 Announce Type: cross  Abstract: Being one of the IR-NAT (Iterative-refinemennt-based NAT) frameworks, the Conditional Masked Language Model (CMLM) adopts the mask-predict paradigm to re-predict the masked low-confidence tokens. However, CMLM suffers from the data distribution discrepancy between training and inference, where the observed tokens are generated differently in the two cases. In this paper, we address this problem with the training approaches of error exposure and consistency regularization (EECR). We construct the mixed sequences based on model prediction during training, and propose to optimize over the masked tokens under imperfect observation conditions. We also design a consistency learning method to constrain the data distribution for the masked tokens under different observing situations to narrow down the gap between training and inference. The experiments on five translation benchmarks obtains an average improvement of 0.68 and 0.40 BLEU scores c
    
[^54]: 对于世界语的语言频率和错误修正的分析

    An Analysis of Langauge Frequency and Error Correction for Esperanto

    [https://arxiv.org/abs/2402.09696](https://arxiv.org/abs/2402.09696)

    本论文运用 Eo-GP 数据集进行了世界语的频率分析，引入了 Eo-GEC 数据集用于错误识别。实验表明 GPT-4 在自动化和人工评估中的性能优于 GPT-3.5，展示了先进语言模型在增强对于较少研究语言的 GEC 策略方面的潜力。

    

    目前的语法错误修正 (GEC) 项目往往着重于主要语言，而对于像世界语这样的资源匮乏语言则关注较少。本文通过首先使用专门为此目的创建的 Eo-GP 数据集进行全面的频率分析，开始弥补这一差距。然后，我们引入了源自真实用户案例并用于错误识别的细粒度语言细节进行注释的 Eo-GEC 数据集。利用 GPT-3.5 和 GPT-4，我们的实验证明 GPT-4 在自动化和人工评估中的表现优于 GPT-3.5，突出了其在解决世界语语法特殊性方面的效果，并展示了先进语言模型在增强对于较少研究语言的 GEC 策略方面的潜力。

    arXiv:2402.09696v1 Announce Type: new  Abstract: Current Grammar Error Correction (GEC) initiatives tend to focus on major languages, with less attention given to low-resource languages like Esperanto. In this article, we begin to bridge this gap by first conducting a comprehensive frequency analysis using the Eo-GP dataset, created explicitly for this purpose. We then introduce the Eo-GEC dataset, derived from authentic user cases and annotated with fine-grained linguistic details for error identification. Leveraging GPT-3.5 and GPT-4, our experiments show that GPT-4 outperforms GPT-3.5 in both automated and human evaluations, highlighting its efficacy in addressing Esperanto's grammatical peculiarities and illustrating the potential of advanced language models to enhance GEC strategies for less commonly studied languages.
    
[^55]: PAL：对大型语言模型的代理引导黑盒攻击

    PAL: Proxy-Guided Black-Box Attack on Large Language Models

    [https://arxiv.org/abs/2402.09674](https://arxiv.org/abs/2402.09674)

    PAL是第一个黑盒查询攻击大型语言模型的优化算法，通过代理模型引导优化过程，并使用复杂的损失函数，取得了较高的攻击成功率。

    

    大型语言模型（LLMs）近几个月来越来越受欢迎，但在被操纵时它们展示出的危险能力令人担忧。尽管安全微调等技术旨在最小化有害使用，但最近的研究表明，LLMs仍然容易受到引发有毒回应的攻击。在这项工作中，我们引入了对LLMs的代理引导攻击（PAL），这是第一个基于优化的对LLMs的黑盒仅查询攻击。具体而言，它依赖于一个替代模型来引导优化过程，并采用了针对真实世界LLM API设计的复杂损失函数。我们的攻击在GPT-3.5-Turbo上达到84%的攻击成功率（ASR），在Llama-2-7B上达到48%，而目前最先进的方法仅为4%。我们还提出了GCG++，这是对GCG攻击的改进，在白盒Llama-2-7B上达到了94%的ASR，以及基于查询的攻击的强有力但简单的基准方法——LLMs上的随机搜索攻击（RAL）。

    arXiv:2402.09674v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have surged in popularity in recent months, but they have demonstrated concerning capabilities to generate harmful content when manipulated. While techniques like safety fine-tuning aim to minimize harmful use, recent works have shown that LLMs remain vulnerable to attacks that elicit toxic responses. In this work, we introduce the Proxy-Guided Attack on LLMs (PAL), the first optimization-based attack on LLMs in a black-box query-only setting. In particular, it relies on a surrogate model to guide the optimization and a sophisticated loss designed for real-world LLM APIs. Our attack achieves 84% attack success rate (ASR) on GPT-3.5-Turbo and 48% on Llama-2-7B, compared to 4% for the current state of the art. We also propose GCG++, an improvement to the GCG attack that reaches 94% ASR on white-box Llama-2-7B, and the Random-Search Attack on LLMs (RAL), a strong but simple baseline for query-based attacks. We
    
[^56]: 如何训练数据高效的LLM模型

    How to Train Data-Efficient LLMs

    [https://arxiv.org/abs/2402.09668](https://arxiv.org/abs/2402.09668)

    本文研究了如何训练数据高效的LLM模型，提出了Ask-LLM和Density两种优秀的数据选择方法。

    

    大型语言模型（LLM）的训练十分昂贵。本文研究了用于预训练LLM的数据高效方法，即旨在优化模型质量和训练资源/数据消耗的帕累托前沿的技术。我们试图理解基于（i）昂贵的数据质量估计和（ii）基于特征空间的覆盖率和多样性测量的数据选择程序所带来的权衡。我们的第一种技术“Ask-LLM”利用调节指令的LLM的零样本推理能力来直接评估训练样例的质量。为了达到覆盖率，我们提出了密度采样，它根据数据分布选择多样的样本。在我们对19种采样器进行了数百个评估任务和预训练运行的对比研究中，我们发现Ask-LLM和Density是各自类别中最好的方法。

    arXiv:2402.09668v1 Announce Type: cross  Abstract: The training of large language models (LLMs) is expensive. In this paper, we study data-efficient approaches for pre-training LLMs, i.e., techniques that aim to optimize the Pareto frontier of model quality and training resource/data consumption. We seek to understand the tradeoffs associated with data selection routines based on (i) expensive-to-compute data-quality estimates, and (ii) maximization of coverage and diversity-based measures in the feature space. Our first technique, Ask-LLM, leverages the zero-shot reasoning capabilities of instruction-tuned LLMs to directly assess the quality of a training example. To target coverage, we propose Density sampling, which models the data distribution to select a diverse sample. In our comparison of 19 samplers, involving hundreds of evaluation tasks and pre-training runs, we find that Ask-LLM and Density are the best methods in their respective categories. Coverage sampling can recover th
    
[^57]: 引入常识知识图完成中的文本蕴涵

    EntailE: Introducing Textual Entailment in Commonsense Knowledge Graph Completion

    [https://arxiv.org/abs/2402.09666](https://arxiv.org/abs/2402.09666)

    本论文提出在常识知识图完成中引入文本蕴涵。在常识知识图的建模过程中，考虑节点和关系的语义合理性是关键。以前的方法利用概念抽象提高了事件合理性的一致性，但存在可扩展性和数据稀疏性的问题。在本文中，我们采用文本蕴涵来寻找隐含的蕴涵关系。

    

    arXiv:2402.09666v1 公告类型：新的 摘要：常识知识图完成是常识知识图构建和应用的新挑战。与Freebase和YAGO等事实性知识图不同，常识知识图（如ConceptNet）利用自由形式的文本来表示命名实体、短语和事件作为它们的节点。这种松散的结构导致了大型稀疏的常识知识图，使得对这些节点的语义理解对学习丰富的常识知识图嵌入更为关键。当前的方法利用语义相似性增加图的密度，但节点及其关系的语义合理性尚未充分探索。以前的工作采用概念抽象来改善建模（事件）合理性的一致性，但它们不够可扩展且仍然受到数据稀疏性的影响。在本文中，我们提出采用文本蕴涵来寻找隐含的蕴涵关系。

    arXiv:2402.09666v1 Announce Type: new  Abstract: Commonsense knowledge graph completion is a new challenge for commonsense knowledge graph construction and application. In contrast to factual knowledge graphs such as Freebase and YAGO, commonsense knowledge graphs (CSKGs; e.g., ConceptNet) utilize free-form text to represent named entities, short phrases, and events as their nodes. Such a loose structure results in large and sparse CSKGs, which makes the semantic understanding of these nodes more critical for learning rich commonsense knowledge graph embedding. While current methods leverage semantic similarities to increase the graph density, the semantic plausibility of the nodes and their relations are under-explored. Previous works adopt conceptual abstraction to improve the consistency of modeling (event) plausibility, but they are not scalable enough and still suffer from data sparsity. In this paper, we propose to adopt textual entailment to find implicit entailment relations be
    
[^58]: CodeMind:一个用于挑战大型语言模型进行代码推理的框架

    CodeMind: A Framework to Challenge Large Language Models for Code Reasoning

    [https://arxiv.org/abs/2402.09664](https://arxiv.org/abs/2402.09664)

    CodeMind是一个用于挑战大型语言模型进行代码推理的框架，通过评估LLMs的代码推理能力来替代仅仅依靠测试通过来评估，对三种代码推理任务进行评估，结果显示LLMs能够公正地理解控制流结构，并且对于简单程序和复杂程序，它们通常能够推理出输入如何演变为输出。

    

    仅靠测试通过来评估大型语言模型（LLMs）的代码合成能力可能会导致不公正的评估或促进具有数据泄漏的模型，作为一种替代方案，我们介绍了CodeMind，这是一个旨在评估LLMs的代码推理能力的框架。CodeMind目前支持三种代码推理任务：独立执行推理（IER）、依赖执行推理（DER）和规范推理（SR）。前两者评估模型以预测任意代码的执行输出，或者模型能够正确合成的代码。第三个任务评估LLMs实现指定预期行为的程度。我们使用CodeMind对两种不同编程语言中的五个基准下的九个LLMs进行了广泛的评估，结果表明LLMs能够公正地理解控制流结构，并且对于简单程序和复杂程序，它们通常能够推理出输入如何演变为输出。

    arXiv:2402.09664v1 Announce Type: cross  Abstract: Solely relying on test passing to evaluate Large Language Models (LLMs) for code synthesis may result in unfair assessment or promoting models with data leakage. As an alternative, we introduce CodeMind, a framework designed to gauge the code reasoning abilities of LLMs. CodeMind currently supports three code reasoning tasks: Independent Execution Reasoning (IER), Dependent Execution Reasoning (DER), and Specification Reasoning (SR). The first two evaluate models to predict the execution output of an arbitrary code or code the model could correctly synthesize. The third one evaluates the extent to which LLMs implement the specified expected behavior. Our extensive evaluation of nine LLMs across five benchmarks in two different programming languages using CodeMind shows that LLMs fairly understand control flow constructs and, in general, are capable of reasoning how inputs evolve to output, specifically for simple programs and the ones 
    
[^59]: GPT-4在基于USMLE的案例研究中的表现评估

    GPT-4's assessment of its performance in a USMLE-based case study

    [https://arxiv.org/abs/2402.09654](https://arxiv.org/abs/2402.09654)

    本研究探讨了GPT-4在医疗应用中的表现评估。实验结果表明，反馈对相对置信度有影响，但并不一致地增加或减少。

    

    本研究调查GPT-4在医疗应用中的表现评估。通过使用简单的提示技术，从美国医学执照考试（USMLE）问卷中提取问题的方式，任务是评估模型在提问之前和提问之后的置信度得分。问卷根据是否有反馈分为两组：反馈组（WF）和无反馈组（NF）。要求模型在每个问题之前和之后提供绝对和相对置信度得分。通过使用统计工具分析实验结果，研究了WF和NF组的置信度变异性。此外，进行了顺序分析以观察WF和NF组的性能变化。结果表明，反馈会影响相对置信度，但并不总是增加或减少。

    arXiv:2402.09654v1 Announce Type: new  Abstract: This study investigates GPT-4's assessment of its performance in healthcare applications. A simple prompting technique was used to prompt the LLM with questions taken from the United States Medical Licensing Examination (USMLE) questionnaire and it was tasked to evaluate its confidence score before posing the question and after asking the question. The questionnaire was categorized into two groups-questions with feedback (WF) and questions with no feedback(NF) post-question. The model was asked to provide absolute and relative confidence scores before and after each question. The experimental findings were analyzed using statistical tools to study the variability of confidence in WF and NF groups. Additionally, a sequential analysis was conducted to observe the performance variation for the WF and NF groups. Results indicate that feedback influences relative confidence but doesn't consistently increase or decrease it. Understanding the p
    
[^60]: 答案就是你需要的一切：通过回答问题进行指令遵循的文本嵌入

    Answer is All You Need: Instruction-following Text Embedding via Answering the Question

    [https://arxiv.org/abs/2402.09642](https://arxiv.org/abs/2402.09642)

    通过回答问题进行指令遵循的文本嵌入方法，通过将指令视为对输入文本的问题并编码期望的答案，从而获得更相似的嵌入效果。

    

    这项工作旨在构建一个能够捕捉用户指令所指定的文本特征的文本嵌入器。尽管它在部署面向用户的嵌入方面具有巨大潜力，但之前的方法都没有提供具体的解决方案。本文提出了一个新的视角，将指令视为对输入文本的问题，并编码期望的答案以相应地获得表示。直观地说，遵循指令的文本应该有相似的答案，从而导致更相似的嵌入。具体而言，我们提出了InBedder，它通过在抽象性问题回答任务上对语言模型进行微调来实现该嵌入-via-answering思想。根据我们提出的指令意识测试和指令鲁棒性测试，InBedder在应用于大型语言模型（LLMs）

    arXiv:2402.09642v1 Announce Type: new  Abstract: This work aims to build a text embedder that can capture characteristics of texts specified by user instructions. Despite its tremendous potential to deploy user-oriented embeddings, none of previous approaches provides a concrete solution for it. This paper offers a new viewpoint, which treats the instruction as a question about the input text and encodes the expected answers to obtain the representation accordingly. Intuitively, texts with the same (implicit) semantics would share similar answers following the instruction, thus leading to more similar embeddings. Specifically, we propose InBedder that instantiates this embed-via-answering idea by only fine-tuning language models on abstractive question answering tasks. InBedder demonstrates significantly improved instruction-following capabilities according to our proposed instruction awareness tests and instruction robustness tests, when applied to both large language models (LLMs) (e
    
[^61]: MiMiC：表示空间中最小修改的对抗事实

    MiMiC: Minimally Modified Counterfactuals in the Representation Space

    [https://arxiv.org/abs/2402.09631](https://arxiv.org/abs/2402.09631)

    提出了一种新颖的对抗事实生成方法，利用闭式解决方案在表示空间中生成富有表达力的对抗事实，以减轻语言模型中的不良行为，该方法在地球移动问题方面提供理论上的保证，并对表示空间的几何组织进行改进。

    

    arXiv:2402.09631v1 公告类型：交叉学科 简介：语言模型经常表现出不良行为，如性别偏见或有毒语言。通过对表示空间进行干预，可以有效减轻这些问题，但两种常见的干预技术，即线性擦除和定向向量，并不能提供高度可控和表达丰富度。因此，我们提出了一种新颖的干预方法，旨在在表示空间中生成富有表达力的对抗事实，使源类别（例如“有毒”）的表示与目标类别（例如“非有毒”）的表示相似。这种方法利用高斯假设下的闭式解决方案，在地球移动问题方面提供了理论上的保证，并对表示空间的几何组织提供了进一步的改进。

    arXiv:2402.09631v1 Announce Type: cross  Abstract: Language models often exhibit undesirable behaviors, such as gender bias or toxic language. Interventions in the representation space were shown effective in mitigating such issues by altering the LM behavior. We first show that two prominent intervention techniques, Linear Erasure and Steering Vectors, do not enable a high degree of control and are limited in expressivity.   We then propose a novel intervention methodology for generating expressive counterfactuals in the representation space, aiming to make representations of a source class (e.g., ``toxic'') resemble those of a target class (e.g., ``non-toxic''). This approach, generalizing previous linear intervention techniques, utilizes a closed-form solution for the Earth Mover's problem under Gaussian assumptions and provides theoretical guarantees on the representation space's geometric organization. We further build on this technique and derive a nonlinear intervention that ena
    
[^62]: API Pack：一个用于API调用生成的大规模多语言数据集

    API Pack: A Massive Multilingual Dataset for API Call Generation

    [https://arxiv.org/abs/2402.09615](https://arxiv.org/abs/2402.09615)

    这个论文介绍了一个名为API Pack的大规模多语言数据集，旨在提高大型语言模型的API调用生成能力，通过实验证明了其在生成未见过的API调用方面的高准确率，并实现了跨语言的API调用生成

    

    我们介绍了API Pack，一个包含超过一百万个指令-API调用对的多语言数据集，旨在提高大型语言模型的API调用生成能力。通过实验，我们证明了API Pack在提升模型在这一特定任务上的效果的同时，保持其在一般编码方面的整体熟练程度。仅在20,000个Python实例上对CodeLlama-13B进行微调，其生成未见过的API调用的准确率比GPT-3.5和GPT-4分别高出10%和5%。扩展到100k个例子可以提高对训练期间未见过的新API的泛化能力。此外，实现了跨语言的API调用生成，而无需大量语言特定的数据。数据集、经过微调的模型和整体代码库可在https://github.com/anonymous_url上公开获取。

    arXiv:2402.09615v1 Announce Type: cross  Abstract: We introduce API Pack, a multilingual dataset featuring over one million instruction-API call pairs aimed at advancing large language models' API call generation capabilities. Through experiments, we demonstrate API Pack's efficacy in enhancing models for this specialized task while maintaining their overall proficiency at general coding. Fine-tuning CodeLlama-13B on just 20,000 Python instances yields over 10% and 5% higher accuracy than GPT-3.5 and GPT-4 respectively in generating unseen API calls. Scaling to 100k examples improves generalization to new APIs not seen during training. In addition, cross-lingual API call generation is achieved without needing extensive data per language. The dataset, fine-tuned models, and overall code base are publicly available at https://github.com/anonymous_url.
    
[^63]: 生成式大型语言模型中的概率推理

    Probabilistic Reasoning in Generative Large Language Models

    [https://arxiv.org/abs/2402.09614](https://arxiv.org/abs/2402.09614)

    本文针对大型语言模型在概率推理任务中的限制，引入了贝叶斯语言推理数据集（BLInD），并提出了几种解决策略，包括Python代码和概率推理算法。

    

    本文探讨了大型语言模型（LLMs）在处理涉及概率值明确量化的文本推理问题时面临的挑战。这种概率推理对于从日常对话到医学决策等各种情境都很重要。尽管LLMs在数学推理能力方面有所改进，但在概率推理方面仍然存在显著困难。为了解决这个问题，我们首先引入了贝叶斯语言推理数据集（BLInD），这是一个专门设计用于测试LLMs概率推理能力的新数据集。然后，我们利用这个新数据集来详细说明LLMs在涉及概率推理的任务中的特定限制，并提出了几种将问题映射到不同形式表示的策略，包括Python代码和概率推理算法。

    arXiv:2402.09614v1 Announce Type: cross  Abstract: This paper considers the challenges that Large Language Models (LLMs) face when reasoning over text that includes information involving uncertainty explicitly quantified via probability values. This type of reasoning is relevant to a variety of contexts ranging from everyday conversations to medical decision-making. Despite improvements in the mathematical reasoning capabilities of LLMs, they still exhibit significant difficulties when it comes to probabilistic reasoning. To deal with this problem, we first introduce the Bayesian Linguistic Inference Dataset (BLInD), a new dataset specifically designed to test the probabilistic reasoning capabilities of LLMs. We then leverage this new dataset to thoroughly illustrate the specific limitations of LLMs for tasks involving probabilistic reasoning and present several strategies that map the problem to different formal representations, including Python code, probabilistic inference algorithm
    
[^64]: 实现规模化隐私感知手语翻译

    Towards Privacy-Aware Sign Language Translation at Scale

    [https://arxiv.org/abs/2402.09611](https://arxiv.org/abs/2402.09611)

    本研究提出了一种两阶段框架，用于实现规模化隐私感知手语翻译。我们利用自监督视频预训练和有监督微调的方法，在数据稀缺和隐私风险的情况下实现了最先进的手语翻译性能。

    

    手语翻译的一个主要障碍是数据稀缺。目前在网络上可用的大部分手语数据由于缺乏对齐的字幕而无法用于训练监督模型。此外，使用大规模网络抓取的数据集来扩展手语翻译存在隐私风险，因为其中包含生物特征信息，负责任地开发手语翻译技术应该考虑这一点。在这项工作中，我们提出了一种针对规模化隐私感知手语翻译的两阶段框架，解决了这两个问题。我们引入了SSVP-SLT，它利用匿名和未注释的视频进行自监督视频预训练，然后利用经过筛选的平行数据集进行有监督的手语翻译微调。 SSVP-SLT在How2Sign数据集上实现了最新的微调和零次gloss-free手语翻译性能，比最强的基线模型提高了3个BLEU-4。通过受控实验，我们证明了我们的方法在多个语言和手语词汇上都具有较好的泛化能力。

    arXiv:2402.09611v1 Announce Type: new  Abstract: A major impediment to the advancement of sign language translation (SLT) is data scarcity. Much of the sign language data currently available on the web cannot be used for training supervised models due to the lack of aligned captions. Furthermore, scaling SLT using large-scale web-scraped datasets bears privacy risks due to the presence of biometric information, which the responsible development of SLT technologies should account for. In this work, we propose a two-stage framework for privacy-aware SLT at scale that addresses both of these issues. We introduce SSVP-SLT, which leverages self-supervised video pretraining on anonymized and unannotated videos, followed by supervised SLT finetuning on a curated parallel dataset. SSVP-SLT achieves state-of-the-art finetuned and zero-shot gloss-free SLT performance on the How2Sign dataset, outperforming the strongest respective baselines by over 3 BLEU-4. Based on controlled experiments, we fu
    
[^65]: LogicPrpBank：一个用于逻辑蕴含和等价的语料库

    LogicPrpBank: A Corpus for Logical Implication and Equivalence

    [https://arxiv.org/abs/2402.09609](https://arxiv.org/abs/2402.09609)

    LogicPrpBank是一个新的命题逻辑语料库，用于研究逻辑蕴含和等价的任务。通过与现有的语言模型进行基准测试，展示了该语料库在这一具有挑战性的任务中提供了有用的资源，并且还有改进的空间。

    

    逻辑推理在问题解决和决策中至关重要。虽然语言模型已经证明了处理多种推理任务（例如常识推理）的能力，但它们对于复杂的数学问题，特别是命题逻辑的推理能力仍然很少被探索。这种探索的不足可以归因于标注语料库的有限性。在这里，我们提供了一个经过良好标注的命题逻辑语料库LogicPrpBank，其中包含了7093个命题逻辑陈述（PLS）涵盖六个数学科目，以研究推理逻辑蕴含和等价的全新任务。我们使用广泛使用的语言模型对LogicPrpBank进行了基准测试，展示了我们的语料库为这一具有挑战性的任务提供了有用的资源，并且模型改进的空间还很大。

    arXiv:2402.09609v1 Announce Type: cross  Abstract: Logic reasoning has been critically needed in problem-solving and decision-making. Although Language Models (LMs) have demonstrated capabilities of handling multiple reasoning tasks (e.g., commonsense reasoning), their ability to reason complex mathematical problems, specifically propositional logic, remains largely underexplored. This lack of exploration can be attributed to the limited availability of annotated corpora. Here, we present a well-labeled propositional logic corpus, LogicPrpBank, containing 7093 Propositional Logic Statements (PLSs) across six mathematical subjects, to study a brand-new task of reasoning logical implication and equivalence. We benchmark LogicPrpBank with widely-used LMs to show that our corpus offers a useful resource for this challenging task and there is ample room for model improvement.
    
[^66]: 使用大型语言模型在药物分子和适应症之间进行翻译的新机遇

    Emerging Opportunities of Using Large Language Language Models for Translation Between Drug Molecules and Indications

    [https://arxiv.org/abs/2402.09588](https://arxiv.org/abs/2402.09588)

    本研究探讨了使用大型语言模型在药物分子和适应症之间进行翻译的机遇，提出了一个新任务，并测试了其有效性，这对于药物发现过程具有重要意义。

    

    药物分子是一种改变生物体精神或身体状态的物质。每种批准的药物都有适应症，指的是该药物治疗特定医疗条件的治疗用途。虽然大型语言模型（LLM），一种生成式人工智能技术，最近已经证明在分子和其文本描述之间进行翻译方面是有效的，但仍存在关于在药物分子和适应症（或反之）之间进行翻译的应用的研究空白，这可能极大地有益于药物发现过程。从给定的适应症生成药物的能力将允许发现针对特定疾病或靶点的药物，并最终为患者提供更好的治疗方法。在本文中，我们首先提出了一个新任务，即药物分子和相应适应症之间的翻译，然后进行了实验测试。

    arXiv:2402.09588v1 Announce Type: new  Abstract: A drug molecule is a substance that changes the organism's mental or physical state. Every approved drug has an indication, which refers to the therapeutic use of that drug for treating a particular medical condition. While the Large Language Model (LLM), a generative Artificial Intelligence (AI) technique, has recently demonstrated effectiveness in translating between molecules and their textual descriptions, there remains a gap in research regarding their application in facilitating the translation between drug molecules and indications, or vice versa, which could greatly benefit the drug discovery process. The capability of generating a drug from a given indication would allow for the discovery of drugs targeting specific diseases or targets and ultimately provide patients with better treatments. In this paper, we first propose a new task, which is the translation between drug molecules and corresponding indications, and then test exi
    
[^67]: 蝴蝶引起的变化：利用群体储备转换器进行远见预测

    Changes by Butterflies: Farsighted Forecasting with Group Reservoir Transformer

    [https://arxiv.org/abs/2402.09573](https://arxiv.org/abs/2402.09573)

    我们提出了一种群体储备转换器的架构，通过解决历史序列的挑战和初始条件的敏感性，实现更准确、更稳健地预测长期事件。在多元时间序列中，我们的模型相比最先进的DNN模型表现出更高的准确率，最高可减少89.43%的误差。

    

    在混沌中，两个初始条件之间的微小差异会随着时间的推移呈指数级放大，导致遥远的结果，也被称为蝴蝶效应。因此，远期充满了不确定性，难以预测。我们引入了群体储备转换器来通过克服混沌中的两个挑战（1）大量的历史序列和（2）对初始条件的敏感性来更准确、更稳健地预测长期事件。将一个储备装置连接到转换器上以高效地处理任意长度的历史数据，并通过扩展一组储备装置来减少由于初始化变化而产生的不确定性。我们的架构在多元时间序列中始终优于最先进的DNN模型，包括NLinear、Pyformer、Informer、Autoformer和基准Transformer，其误差减少高达-89.43％，适用于ETTh、ETTm和空气质量等各个领域。

    arXiv:2402.09573v1 Announce Type: cross  Abstract: In Chaos, a minor divergence between two initial conditions exhibits exponential amplification over time, leading to far-away outcomes, known as the butterfly effect. Thus, the distant future is full of uncertainty and hard to forecast. We introduce Group Reservoir Transformer to predict long-term events more accurately and robustly by overcoming two challenges in Chaos: (1) the extensive historical sequences and (2) the sensitivity to initial conditions. A reservoir is attached to a Transformer to efficiently handle arbitrarily long historical lengths, with an extension of a group of reservoirs to reduce the uncertainty due to the initialization variations. Our architecture consistently outperforms state-of-the-art DNN models in multivariate time series, including NLinear, Pyformer, Informer, Autoformer, and the baseline Transformer, with an error reduction of up to -89.43\% in various fields such as ETTh, ETTm, and air quality, demon
    
[^68]: 理性报告卡：评估大型语言模型的经济合理性

    Rationality Report Cards: Assessing the Economic Rationality of Large Language Models

    [https://arxiv.org/abs/2402.09552](https://arxiv.org/abs/2402.09552)

    本文在评估大型语言模型的经济合理性方面提出了一种方法，通过量化评分模型在各个要素上的表现并结合用户提供的评分标准，生成一份"理性报告卡"，以确定代理人是否足够可靠。

    

    越来越多的人对将LLM用作决策"代理人"兴趣日益增加。这包括很多自由度：应该使用哪个模型；如何进行提示；是否要求其进行内省、进行思考链等。解决这些问题（更广泛地说，确定LLM代理人是否足够可靠以便获得信任）需要一种评估这种代理人经济合理性的方法论，在本文中我们提供了一个方法。我们首先对理性决策的经济文献进行了调研、将代理人应该展现的大量细粒度"要素"进行分类，并确定了它们之间的依赖关系。然后，我们提出了一个基准分布，以定量评分LLM在这些要素上的表现，并结合用户提供的评分标准，生成一份"理性报告卡"。最后，我们描述了与14种不同的LLM进行的大规模实证实验的结果。

    arXiv:2402.09552v1 Announce Type: new  Abstract: There is increasing interest in using LLMs as decision-making "agents." Doing so includes many degrees of freedom: which model should be used; how should it be prompted; should it be asked to introspect, conduct chain-of-thought reasoning, etc? Settling these questions -- and more broadly, determining whether an LLM agent is reliable enough to be trusted -- requires a methodology for assessing such an agent's economic rationality. In this paper, we provide one. We begin by surveying the economic literature on rational decision making, taxonomizing a large set of fine-grained "elements" that an agent should exhibit, along with dependencies between them. We then propose a benchmark distribution that quantitatively scores an LLMs performance on these elements and, combined with a user-provided rubric, produces a "rationality report card." Finally, we describe the results of a large-scale empirical experiment with 14 different LLMs, characte
    
[^69]: 告诉我更多！面向基于语言模型的智能代理的隐式用户意图理解

    Tell Me More! Towards Implicit User Intention Understanding of Language Model Driven Agents

    [https://arxiv.org/abs/2402.09205](https://arxiv.org/abs/2402.09205)

    该论文提出了一种面向基于语言模型的智能代理的隐式用户意图理解的方法。通过引入Intention-in-Interaction (IN3) 基准和在代理设计中融入模型专家，使得代理能够更好地与用户进行交互，并提升对用户指令的理解能力。

    

    当前的语言模型驱动代理常常缺乏有效的用户参与机制，考虑到用户指令中常见的模糊性，这是至关重要的。虽然这些代理在制定策略和执行任务方面表现出色，但在寻求澄清和抓住精确的用户意图方面却遇到了困难。为了填补这一差距，我们引入了Intention-in-Interaction (IN3) ，这是一个旨在通过明确的查询检查用户的隐含意图的新颖基准。接下来，我们提出将模型专家作为上游融入代理设计中，以增强用户-代理交互。利用IN3，我们经验性地训练了Mistral-Interact，这是一个强大的模型，它可以主动评估任务的模糊性，询问用户意图，并将其转化为可行的目标，然后开始下游代理任务执行。将其集成到XAgent框架中，我们对增强的代理系统进行了全面评估，以评估用户指令的理解能力。

    arXiv:2402.09205v1 Announce Type: cross Abstract: Current language model-driven agents often lack mechanisms for effective user participation, which is crucial given the vagueness commonly found in user instructions. Although adept at devising strategies and performing tasks, these agents struggle with seeking clarification and grasping precise user intentions. To bridge this gap, we introduce Intention-in-Interaction (IN3), a novel benchmark designed to inspect users' implicit intentions through explicit queries. Next, we propose the incorporation of model experts as the upstream in agent designs to enhance user-agent interaction. Employing IN3, we empirically train Mistral-Interact, a powerful model that proactively assesses task vagueness, inquires user intentions, and refines them into actionable goals before starting downstream agent task execution. Integrating it into the XAgent framework, we comprehensively evaluate the enhanced agent system regarding user instruction understand
    
[^70]: (不)理性与大型语言模型中的认知偏差

    (Ir)rationality and Cognitive Biases in Large Language Models

    [https://arxiv.org/abs/2402.09193](https://arxiv.org/abs/2402.09193)

    本研究评估了七个大型语言模型在认知心理学任务中的表现，发现它们与人类一样存在非理性，但展示的非理性方式与人类偏见不同，同时还表现出了显著的回答不一致性。

    

    大型语言模型(LLMs)是否展现出理性推理？由于其训练数据所含的人类偏见，LLMs已被证实存在人类偏见；然而，其是否反映出了理性推理还不太清楚。本文通过评估七个语言模型在来自认知心理学文献的任务中回答了这个问题。我们发现，和人类一样，LLMs在这些任务中展现出了非理性。然而，LLMs展现出的这种非理性与人类的偏见不同。当LLMs给出错误答案时，它们通常会以与人类偏见不同的方式错误。除此之外，LLMs还展现出了响应的显著不一致性，这表明了额外的非理性层面。除了实验结果，本文还通过展示如何评估和比较这类模型的不同功能，对方法论作出了贡献。

    arXiv:2402.09193v1 Announce Type: cross Abstract: Do large language models (LLMs) display rational reasoning? LLMs have been shown to contain human biases due to the data they have been trained on; whether this is reflected in rational reasoning remains less clear. In this paper, we answer this question by evaluating seven language models using tasks from the cognitive psychology literature. We find that, like humans, LLMs display irrationality in these tasks. However, the way this irrationality is displayed does not reflect that shown by humans. When incorrect answers are given by LLMs to these tasks, they are often incorrect in ways that differ from human-like biases. On top of this, the LLMs reveal an additional layer of irrationality in the significant inconsistency of the responses. Aside from the experimental results, this paper seeks to make a methodological contribution by showing how we can assess and compare different capabilities of these types of models, in this case with r
    
[^71]: 朝着更好的人机对齐方向：评估LLM驱动应用中的任务效用

    Towards better Human-Agent Alignment: Assessing Task Utility in LLM-Powered Applications

    [https://arxiv.org/abs/2402.09015](https://arxiv.org/abs/2402.09015)

    本研究引入了AgentEval框架，用于评估LLM驱动应用的任务效用。该框架通过自动提出一套针对特定应用的评估标准，简化了效用验证过程，并对应用的效用进行了全面量化分析。

    

    大型语言模型（LLM）领域的快速发展导致了一系列应用的出现，这些应用通过协助多个代理人与人类合作，帮助人们完成日常任务。然而，目前仍存在一个重大问题，即如何评估LLM驱动应用是否真正提升用户体验和任务执行效率。这凸显了验证LLM驱动应用效用的方法的迫切需求，特别是要确保应用程序的功能与最终用户的需求相一致。我们引入了AgentEval，它提供了一个实施数学问题的估测模型，这是一个新的框架，旨在通过自动提出一套针对任何给定应用程序独特目标的评估标准，简化效用验证过程。这样可以对应用程序的效用进行全面评估，并量化其与建议标准相比的表现。我们对该框架的稳健性进行了全面的分析。

    arXiv:2402.09015v1 Announce Type: cross Abstract: The rapid development in the field of Large Language Models (LLMs) has led to a surge in applications that facilitate collaboration among multiple agents to assist humans in their daily tasks. However, a significant gap remains in assessing whether LLM-powered applications genuinely enhance user experience and task execution efficiency. This highlights the pressing need for methods to verify utility of LLM-powered applications, particularly by ensuring alignment between the application's functionality and end-user needs. We introduce AgentEval provides an implementation for the math problems}, a novel framework designed to simplify the utility verification process by automatically proposing a set of criteria tailored to the unique purpose of any given application. This allows for a comprehensive assessment, quantifying the utility of an application against the suggested criteria. We present a comprehensive analysis of the robustness of 
    
[^72]: DNABERT-S: 学习具有基因组基础模型的物种感知DNA嵌入

    DNABERT-S: Learning Species-Aware DNA Embedding with Genome Foundation Models

    [https://arxiv.org/abs/2402.08777](https://arxiv.org/abs/2402.08777)

    DNABERT-S是一种专门用于创建物种感知的DNA嵌入的基因组基础模型。为了提高对长读DNA序列的嵌入效果，引入了Manifold Instance Mixup (MI-Mix)对比目标方法来训练模型。

    

    有效的DNA嵌入在基因组分析中仍然至关重要，特别是在缺乏用于模型微调的标记数据的情况下，尽管基因组基础模型已经取得了显著进展。一个典型的例子是宏基因组分箱，这是微生物组研究中的一个关键过程，旨在通过来自可能包含成千上万个不同的、通常没有经过表征的物种的复杂混合DNA序列的物种来对DNA序列进行分组。为了填补有效的DNA嵌入模型的缺陷，我们引入了DNABERT-S，这是一个专门用于创建物种感知的DNA嵌入的基因组基础模型。为了鼓励对易出错的长读DNA序列进行有效嵌入，我们引入了Manifold Instance Mixup(MI-Mix)，一种对比目标，它在随机选择的层次中混合DNA序列的隐藏表示，并训练模型以在输出层识别和区分这些混合比例。

    arXiv:2402.08777v1 Announce Type: cross Abstract: Effective DNA embedding remains crucial in genomic analysis, particularly in scenarios lacking labeled data for model fine-tuning, despite the significant advancements in genome foundation models. A prime example is metagenomics binning, a critical process in microbiome research that aims to group DNA sequences by their species from a complex mixture of DNA sequences derived from potentially thousands of distinct, often uncharacterized species. To fill the lack of effective DNA embedding models, we introduce DNABERT-S, a genome foundation model that specializes in creating species-aware DNA embeddings. To encourage effective embeddings to error-prone long-read DNA sequences, we introduce Manifold Instance Mixup (MI-Mix), a contrastive objective that mixes the hidden representations of DNA sequences at randomly selected layers and trains the model to recognize and differentiate these mixed proportions at the output layer. We further enha
    
[^73]: SemRel2024: 14种语言的语义文本相关性数据集合

    SemRel2024: A Collection of Semantic Textual Relatedness Datasets for 14 Languages

    [https://arxiv.org/abs/2402.08638](https://arxiv.org/abs/2402.08638)

    SemRel2024是一个包含来自14种语言的语义文本相关性数据集合，通过该数据集合可以探索和量化语义相关性。这对于大型语言模型的能力和性能有重要的影响。这个数据集合涵盖了南非荷兰语、阿尔及利亚阿拉伯语、阿姆哈拉语、英语、豪萨语、印地语、印度尼西亚语、卢旺达语、马拉地语、摩洛哥阿拉伯语、现代标准阿拉伯语、旁遮普语、西班牙语和泰卢固语等14种语言。

    

    探索和量化语义相关性是语言表达的核心。它在各种自然语言处理任务中具有重要影响，包括为大型语言模型（LLM）的能力和性能提供洞察。虽然早期的自然语言处理研究主要集中在语义相似性上，往往是在英语语境中，但我们认为更广泛的语义相关性现象值得研究。本文介绍了SemRel，这是一个由母语为14种语言进行注释的新的语义相关性数据集合：南非荷兰语、阿尔及利亚阿拉伯语、阿姆哈拉语、英语、豪萨语、印地语、印度尼西亚语、卢旺达语、马拉地语、摩洛哥阿拉伯语、现代标准阿拉伯语、旁遮普语、西班牙语和泰卢固语。这些语言来自五个不同的语系，主要在非洲和亚洲使用，这些地区的自然语言处理资源相对较少。SemRel数据集中的每个实例都是与一个表示相关性得分的句子对相关联。

    Exploring and quantifying semantic relatedness is central to representing language. It holds significant implications across various NLP tasks, including offering insights into the capabilities and performance of Large Language Models (LLMs). While earlier NLP research primarily focused on semantic similarity, often within the English language context, we instead investigate the broader phenomenon of semantic relatedness. In this paper, we present SemRel, a new semantic relatedness dataset collection annotated by native speakers across 14 languages:Afrikaans, Algerian Arabic, Amharic, English, Hausa, Hindi, Indonesian, Kinyarwanda, Marathi, Moroccan Arabic, Modern Standard Arabic, Punjabi, Spanish, and Telugu. These languages originate from five distinct language families and are predominantly spoken in Africa and Asia -- regions characterised by a relatively limited availability of NLP resources. Each instance in the SemRel datasets is a sentence pair associated with a score that repr
    
[^74]: 用分类器驱动的方法揭示大型语言模型中的五大人格特质：文本分析

    Eliciting Big Five Personality Traits in Large Language Models: A Textual Analysis with Classifier-Driven Approach

    [https://arxiv.org/abs/2402.08341](https://arxiv.org/abs/2402.08341)

    本研究使用分类器驱动的方法，通过不同的输入提示探究大型语言模型的输出变化，以增加其透明度。结果显示，这些模型根据输入的不同提示而表现出不同的人格特质，类似于人类对刺激做出的反应。

    

    大型语言模型（LLMs）在招聘背景下被应聘者和雇主广泛使用，然而这也引发了众多伦理问题，特别是与这些“黑盒子”模型缺乏透明度有关。尽管先前的研究试图通过调查LLMs的人格特质来增加其透明度，但许多先前的研究都要求模型来完成人格评估。相反，本研究旨在通过检查不同输入提示下模型的输出变化来更好地理解这些模型。具体来说，我们使用从常见面试问题和旨在引发特定的五大人格特质的提示来进行新颖的调查方法，以检查模型是否像人类一样容易激活特定人格特质，并根据其输出中的语言来评估其人格。为此，我们反复提供提示。

    Large Language Models (LLMs) are increasingly being utilized by both candidates and employers in the recruitment context. However, with this comes numerous ethical concerns, particularly related to the lack of transparency in these "black-box" models. Although previous studies have sought to increase the transparency of these models by investigating the personality traits of LLMs, many of the previous studies have provided them with personality assessments to complete. On the other hand, this study seeks to obtain a better understanding of such models by examining their output variations based on different input prompts. Specifically, we use a novel elicitation approach using prompts derived from common interview questions, as well as prompts designed to elicit particular Big Five personality traits to examine whether the models were susceptible to trait-activation like humans are, to measure their personality based on the language used in their outputs. To do so, we repeatedly prompte
    
[^75]: 基于10万小时数据的10亿参数文本到语音模型的经验教训

    BASE TTS: Lessons from building a billion-parameter Text-to-Speech model on 100K hours of data

    [https://arxiv.org/abs/2402.08093](https://arxiv.org/abs/2402.08093)

    基于10万小时数据的10亿参数文本到语音模型BASE TTS在语音自然度上达到了最新技术水平，并且能够展现自然的韵律。

    

    我们介绍了一个名为BASE TTS的文本到语音（TTS）模型，其中BASE代表大规模自适应可流式TTS和新出现的能力。BASE TTS是迄今为止最大的TTS模型，训练于10万小时的公共领域语音数据，实现了语音自然度的最新技术水平。它采用了一个10亿参数的自回归Transformer，将原始文本转换为离散代码（"speechcodes"），然后通过基于卷积的解码器将这些speechcodes以增量、可流式的方式转换为波形。此外，我们的speechcodes采用了一种新颖的语音标记化技术，具有说话者ID解耦和字节对编码的压缩特性。与大量数据训练的大语言模型广泛报道的"新出现的能力"类似，我们展示了使用10K+小时和500M+参数构建的BASE TTS变体在文本复杂句子上开始展现自然的韵律。我们设计了...

    We introduce a text-to-speech (TTS) model called BASE TTS, which stands for $\textbf{B}$ig $\textbf{A}$daptive $\textbf{S}$treamable TTS with $\textbf{E}$mergent abilities. BASE TTS is the largest TTS model to-date, trained on 100K hours of public domain speech data, achieving a new state-of-the-art in speech naturalness. It deploys a 1-billion-parameter autoregressive Transformer that converts raw texts into discrete codes ("speechcodes") followed by a convolution-based decoder which converts these speechcodes into waveforms in an incremental, streamable manner. Further, our speechcodes are built using a novel speech tokenization technique that features speaker ID disentanglement and compression with byte-pair encoding. Echoing the widely-reported "emergent abilities" of large language models when trained on increasing volume of data, we show that BASE TTS variants built with 10K+ hours and 500M+ parameters begin to demonstrate natural prosody on textually complex sentences. We design
    
[^76]: 使用语言反馈模型来改进政策

    Policy Improvement using Language Feedback Models

    [https://arxiv.org/abs/2402.07876](https://arxiv.org/abs/2402.07876)

    本文介绍了一种使用语言反馈模型（LFMs）改进政策的方法，通过识别期望的行为并进行模仿学习，我们在任务完成率、泛化性能和人类可解释性方面取得了显著改进。

    

    我们引入了语言反馈模型（LFMs），用于在指令遵循中识别期望的行为-有助于实现指令中指定任务的行动-以进行模仿学习。为了训练LFMs，我们从大型语言模型（LLMs）获取对视觉轨迹进行语言描述的反馈。首先，通过使用LFMs识别期望模仿的行为，我们在三种不同的语言基础环境（Touchdown，ScienceWorld和ALFWorld）上，在任务完成率上改善了强行为克隆的基线方法。其次，与LLMs直接预测行动相比，使用LFMs在LLM输出标记的数量相同的情况下表现更好。第三，LFMs适应未见环境，通过一轮适应使任务完成率提高了3.5-12.0％。最后，可以修改LFM以提供人类可解释的反馈，无需性能损失，从而允许人类验证模仿学习的期望行为。

    We introduce Language Feedback Models (LFMs) that identify desirable behaviour - actions that help achieve tasks specified in the instruction - for imitation learning in instruction following. To train LFMs, we obtain feedback from Large Language Models (LLMs) on visual trajectories verbalized to language descriptions. First, by using LFMs to identify desirable behaviour to imitate, we improve in task-completion rate over strong behavioural cloning baselines on three distinct language grounding environments (Touchdown, ScienceWorld, and ALFWorld). Second, LFMs outperform using LLMs as experts to directly predict actions, when controlling for the number of LLM output tokens. Third, LFMs generalize to unseen environments, improving task-completion rate by 3.5-12.0% through one round of adaptation. Finally, LFM can be modified to provide human-interpretable feedback without performance loss, allowing human verification of desirable behaviour for imitation learning.
    
[^77]: 自然语言强化学习

    Natural Language Reinforcement Learning

    [https://arxiv.org/abs/2402.07157](https://arxiv.org/abs/2402.07157)

    本研究将自然语言表示和强化学习原则相结合，提出了自然语言强化学习（NLRL）框架，解决了强化学习在样本效率低、解释性不足和缺乏监督信号等方面的限制问题，通过实验验证了其有效性和可解释性。

    

    强化学习（RL）在学习决策任务的策略方面展现出了令人瞩目的能力。然而，RL常常面临样本效率低、解释性不足和缺乏稀疏监督信号等问题的限制。为了解决这些问题，我们从人类学习过程中汲取灵感，引入了自然语言强化学习（NLRL），创新性地将RL原则与自然语言表示结合起来。具体而言，NLRL在自然语言空间中重新定义了任务目标、策略、价值函数、Bellman方程和策略迭代等RL概念。我们还展示了如何利用最新的大型语言模型（LLM）如GPT-4来实现NLRL。对表格MDPs的初步实验表明了NLRL框架的有效性、高效性和可解释性。

    Reinforcement Learning (RL) has shown remarkable abilities in learning policies for decision-making tasks. However, RL is often hindered by issues such as low sample efficiency, lack of interpretability, and sparse supervision signals. To tackle these limitations, we take inspiration from the human learning process and introduce Natural Language Reinforcement Learning (NLRL), which innovatively combines RL principles with natural language representation. Specifically, NLRL redefines RL concepts like task objectives, policy, value function, Bellman equation, and policy iteration in natural language space. We present how NLRL can be practically implemented with the latest advancements in large language models (LLMs) like GPT-4. Initial experiments over tabular MDPs demonstrate the effectiveness, efficiency, and also interpretability of the NLRL framework.
    
[^78]: 在GPT-4V模型中探索视觉文化意识：一项全面探索

    Exploring Visual Culture Awareness in GPT-4V: A Comprehensive Probing

    [https://arxiv.org/abs/2402.06015](https://arxiv.org/abs/2402.06015)

    通过在GPT-4V模型上进行全面的探索，我们发现该模型在识别文化概念方面表现出色，但在低资源语言中仍表现较弱。在图像字幕任务中，GPT-4V在文化相关性方面优于原文。

    

    预训练的大型视觉-语言模型近年来引起了 considerable的关注，其出色的性能令人印象深刻。尽管已经做出了多方面的努力来评估这些模型，但目前最先进的GPT-4V模型在视觉文化意识方面尚未被探索。为了填补这一空白，我们使用MaRVL基准数据集广泛探索了GPT-4V，旨在调查其在视觉理解方面，特别是文化方面的能力和限制。具体而言，我们引入了三个与视觉相关的任务，即标题分类、成对标题生成和文化标签选择，以系统地深入研究细粒度的视觉文化评估。实验结果表明，GPT-4V在识别文化概念方面表现出色，但在资源匮乏的语言（如泰米尔语和斯瓦希里语）中仍然表现较弱。值得注意的是，通过人工评估，GPT-4V在图像字幕任务中证明在文化相关性方面优于原文的表现。

    Pretrained large Vision-Language models have drawn considerable interest in recent years due to their remarkable performance. Despite considerable efforts to assess these models from diverse perspectives, the extent of visual cultural awareness in the state-of-the-art GPT-4V model remains unexplored. To tackle this gap, we extensively probed GPT-4V using the MaRVL benchmark dataset, aiming to investigate its capabilities and limitations in visual understanding with a focus on cultural aspects. Specifically, we introduced three visual related tasks, i.e. caption classification, pairwise captioning, and culture tag selection, to systematically delve into fine-grained visual cultural evaluation. Experimental results indicate that GPT-4V excels at identifying cultural concepts but still exhibits weaker performance in low-resource languages, such as Tamil and Swahili. Notably, through human evaluation, GPT-4V proves to be more culturally relevant in image captioning tasks than the original 
    
[^79]: NoisyICL: 一点噪音在模型参数中提高了上下文学习的性能、

    NoisyICL: A Little Noise in Model Parameters Calibrates In-context Learning

    [https://arxiv.org/abs/2402.05515](https://arxiv.org/abs/2402.05515)

    NoisyICL通过在模型参数中引入噪音，提高了上下文学习的性能和校准性，实验结果显示NoisyICL可以产生更准确、更公平的预测。

    

    上下文学习 (ICL) 在高先验偏差和不可信任的置信度的影响下，表现不佳且校准不足。以往的一些工作通过使用庞大的数据集和计算成本对语言模型进行微调以改善 ICL 的性能。在本文中，我们提出了 NoisyICL，通过随机噪音扰动模型参数来努力提高性能和校准性。我们在2个模型和12个下游数据集上的实验表明，NoisyICL可以帮助ICL产生更准确的预测。进一步的分析表明，NoisyICL使得模型能够提供更公平的预测，同时置信度更可信。因此，我们认为NoisyICL是ICL的一种有效校准方法。我们的实验代码已上传至Github。

    In-Context Learning (ICL) is suffering from unsatisfactory performance and under-calibration due to high prior bias and unfaithful confidence. Some previous works fine-tuned language models for better ICL performance with enormous datasets and computing costs. In this paper, we propose NoisyICL, simply perturbing the model parameters by random noises to strive for better performance and calibration. Our experiments on 2 models and 12 downstream datasets show that NoisyICL can help ICL produce more accurate predictions. Our further analysis indicates that NoisyICL enables the model to provide more fair predictions, and also with less unfaithful confidence. Therefore, we believe that NoisyICL is an effective calibration of ICL. Our experimental code is uploaded to Github.
    
[^80]: L4Q: 通过基于LoRA的量化训练在大型语言模型上提供参数高效的量化训练

    L4Q: Parameter Efficient Quantization-Aware Training on Large Language Models via LoRA-wise LSQ

    [https://arxiv.org/abs/2402.04902](https://arxiv.org/abs/2402.04902)

    L4Q是一种参数高效的量化感知训练算法，通过基于LoRA的学习的量化步长，解决了大型语言模型中量化训练的挑战。

    

    后训练量化(PTQ)和量化感知训练(QAT)方法正在流行起来，以缓解大型语言模型(LLMs)所带来的高内存和计算成本。在资源受限的情况下，尽管后者具有更高的准确性潜力，但由于其减少的训练开销，通常首选后训练量化。同时，介绍了参数高效微调方法，如低秩适应（LoRA），并最近的工作已经探索了量化感知参数高效微调技术。然而，这些方法可能缺乏通用性，因为它们依赖于预量化模型的配置。由非线性量化或混合精度权重引起的效果可能会受到影响，并且重新训练特定量化参数可能会影响最优性能。为了应对这些挑战，我们提出了L4Q，一种参数高效的量化感知训练算法。L4Q利用了基于LoRA的学习的量化步长。

    Post-training quantization (PTQ) and quantization-aware training (QAT) methods are gaining popularity in mitigating the high memory and computational costs associated with Large Language Models (LLMs). In resource-constrained scenarios, PTQ, with its reduced training overhead, is often preferred over QAT, despite the latter's potential for higher accuracy. Meanwhile, parameter-efficient fine-tuning (PEFT) methods like low-rank adaptation (LoRA) have been introduced, and recent efforts have explored quantization-aware PEFT techniques. However, these approaches may lack generality due to their reliance on the pre-quantized model's configuration. Their effectiveness may be compromised by non-linearly quantized or mixed-precision weights, and the retraining of specific quantization parameters might impede optimal performance. To address these challenges, we propose L4Q, an algorithm for parameter-efficient quantization-aware training. L4Q leverages LoRA-wise learned quantization step size 
    
[^81]: PaDeLLM-NER：大型语言模型中的并行解码用于命名实体识别

    PaDeLLM-NER: Parallel Decoding in Large Language Models for Named Entity Recognition

    [https://arxiv.org/abs/2402.04838](https://arxiv.org/abs/2402.04838)

    本研究提出了PaDeLLM-NER，一种能够在大型语言模型中实现并行解码，从而显著减少命名实体识别的生成延迟，同时保持预测质量和性能。

    

    本研究旨在使用大型语言模型（LLMs）减少命名实体识别（NER）的生成延迟。LLMs的高延迟的主要原因是顺序解码过程，该过程自回归地生成NER的所有标签和提及，显著增加了序列长度。为此，我们引入了PaDeLLM-NER（Parallel Decoding in LLM for NE），这是一种无需额外模块或架构修改即可无缝集成到现有生成模型框架中的方法。PaDeLLM-NER允许同时解码所有提及，从而减少生成延迟。实验结果显示，PaDeLLM-NER的推理速度显著提高，对英语和中文来说比自回归方法快1.76到10.22倍。与各种数据集上的最先进性能相媲美，同时维持了预测质量。

    In this study, we aim to reduce generation latency for Named Entity Recognition (NER) with Large Language Models (LLMs). The main cause of high latency in LLMs is the sequential decoding process, which autoregressively generates all labels and mentions for NER, significantly increase the sequence length. To this end, we introduce Parallel Decoding in LLM for NE} (PaDeLLM-NER), a approach that integrates seamlessly into existing generative model frameworks without necessitating additional modules or architectural modifications. PaDeLLM-NER allows for the simultaneous decoding of all mentions, thereby reducing generation latency. Experiments reveal that PaDeLLM-NER significantly increases inference speed that is 1.76 to 10.22 times faster than the autoregressive approach for both English and Chinese. Simultaneously it maintains the quality of predictions as evidenced by the performance that is on par with the state-of-the-art across various datasets.
    
[^82]: 超越线条和圆圈：揭示大型语言模型中的几何推理差距

    Beyond Lines and Circles: Unveiling the Geometric Reasoning Gap in Large Language Models

    [https://arxiv.org/abs/2402.03877](https://arxiv.org/abs/2402.03877)

    本文调查了大型语言模型（LLMs）在几何推理方面的能力，并发现了它们在目标变量选择和2D空间关系方面存在偏见和困难。通过引入基于LLMs的多代理体系结构，本研究提出了一种通过自我纠正、协作和不同角色专业化来提高LLMs几何推理能力的框架。

    

    大型语言模型（LLMs）在数学和算法任务方面展示了不断增长的能力，然而它们在几何推理方面的技能还未被充分探索。我们调查了LLMs在构造性几何问题解决中的能力，这是人类数学推理发展中最基础的步骤之一。我们的研究揭示了目前最先进的LLMs在这个领域面临的显著挑战，尽管在类似领域取得了许多成功。LLMs在目标变量选择方面存在偏见，并且在2D空间关系方面面临困难，经常会错误地表示和臆造对象及其放置位置。为此，我们引入了一个基于LLMs的多代理体系结构，通过进行内部对话来增强它们现有的推理潜力。这项工作强调了LLMs在几何推理中的现有限制，并通过自我纠正、协作和不同角色专业化来提高几何推理能力。

    Large Language Models (LLMs) demonstrate ever-increasing abilities in mathematical and algorithmic tasks, yet their geometric reasoning skills are underexplored. We investigate LLMs' abilities in constructive geometric problem-solving one of the most fundamental steps in the development of human mathematical reasoning. Our work reveals notable challenges that the state-of-the-art LLMs face in this domain despite many successes in similar areas. LLMs exhibit biases in target variable selection and struggle with 2D spatial relationships, often misrepresenting and hallucinating objects and their placements. To this end, we introduce a framework that formulates an LLMs-based multi-agents system that enhances their existing reasoning potential by conducting an internal dialogue. This work underscores LLMs' current limitations in geometric reasoning and improves geometric reasoning capabilities through self-correction, collaboration, and diverse role specializations.
    
[^83]: EffiBench:评估自动生成代码的效率的基准测试

    EffiBench: Benchmarking the Efficiency of Automatically Generated Code

    [https://arxiv.org/abs/2402.02037](https://arxiv.org/abs/2402.02037)

    本文提出了EffiBench基准测试，用于评估代码生成模型生成的代码的效率。实验证明，GPT-4-turbo生成的代码最高效。

    

    代码生成模型在辅助软件开发方面变得越来越重要，可以帮助完成代码补全、调试和代码转换等任务。尽管当前的研究已经深入研究了代码生成模型生成的正确性，但生成代码的效率这一重要方面常常被忽视。本文提出了EffiBench，一个包含1,000个效率关键的编码问题的基准测试，用于评估代码生成模型生成的代码的效率。EffiBench包含了一系列多样化的LeetCode编码问题，每个问题都与一个可执行的人工编写的典型解决方案配对。通过EffiBench，我们在实践中考察了21种大型语言模型（其中13种是开源的，8种是闭源的）在生成高效代码方面的能力。结果表明，GPT-4-turbo生成的代码最高效，明显优于Palm-2-chat-bison、Claude-instant-1、Gemini-pro、GPT-4和GPT-3.5。

    Code generation models have increasingly become integral to aiding software development, offering assistance in tasks such as code completion, debugging, and code translation. Although current research has thoroughly examined the correctness of code produced by code generation models, a vital aspect, i.e., the efficiency of the generated code, has often been neglected. This paper presents EffiBench, a benchmark with 1,000 efficiency-critical coding problems for assessing the efficiency of code generated by code generation models. EffiBench contains a diverse set of LeetCode coding problems. Each problem is paired with an executable human-written canonical solution. With EffiBench, we empirically examine the capability of 21 Large Language Models (13 open-sourced and 8 closed-sourced) in generating efficient code. The results demonstrate that GPT-4-turbo generates the most efficient code, significantly outperforming Palm-2-chat-bison, Claude-instant-1, Gemini-pro, GPT-4, and GPT-3.5. Ne
    
[^84]: SWEA:通过主题词嵌入修改改变大型语言模型中的事实知识

    SWEA: Changing Factual Knowledge in Large Language Models via Subject Word Embedding Altering

    [https://arxiv.org/abs/2401.17809](https://arxiv.org/abs/2401.17809)

    提出了一种主题词嵌入修改框架（SWEA），通过在推理阶段修改主题的表示来编辑知识，保护模型的原始权重，避免不可逆的损害和额外的推理开销。

    

    模型编辑近来引起了广泛关注。目前的模型编辑方法主要涉及修改模型参数或向现有模型添加附加模块。然而，前者会对LLM造成不可逆的影响，而后者会产生额外的推理开销，并且模糊的向量匹配并不总是可靠的。为了解决这些问题，我们提出了一种可扩展的主题词嵌入修改（SWEA）框架，它在推理阶段修改主题的表示，并实现编辑知识的目标。SWEA在模型外部使用精确的关键匹配，并进行可靠的主题词嵌入修改，从而保护模型的原始权重而不增加推理开销。然后，我们提出优化抑制融合方法，首先优化编辑目标的嵌入向量，然后抑制知识嵌入维度（KED）以获得最终融合的嵌入。我们因此提出了SWEAOS元方法。

    Model editing has recently gained widespread attention. Current model editing methods primarily involve modifying model parameters or adding additional modules to the existing model. However, the former causes irreversible damage to LLMs, while the latter incurs additional inference overhead and fuzzy vector matching is not always reliable. To address these issues, we propose an expandable Subject Word Embedding Altering (SWEA) framework, which modifies the representation of subjects and achieve the goal of editing knowledge during the inference stage. SWEA uses precise key matching outside the model and performs reliable subject word embedding altering, thus protecting the original weights of the model without increasing inference overhead. We then propose optimizing then suppressing fusion method, which first optimizes the embedding vector for the editing target and then suppresses the Knowledge Embedding Dimension (KED) to obtain the final fused embedding. We thus propose SWEAOS met
    
[^85]: PokeMQA: 可编程的多跳问答知识编辑

    PokeMQA: Programmable knowledge editing for Multi-hop Question Answering

    [https://arxiv.org/abs/2312.15194](https://arxiv.org/abs/2312.15194)

    PokeMQA是一个可编程的多跳问答知识编辑模型，通过避免功能多样的推理任务的耦合，实现了更好的问题理解和回答能力。

    

    多跳问答（MQA）是评估机器理解和推理能力的挑战性任务之一，大型语言模型（LLMs）已广泛实现了与人类相媲美的性能。由于现实世界中知识事实的动态性，已经探索了知识编辑来更新模型，以获取最新的事实，同时避免昂贵的重新训练或微调。从编辑后的事实开始，更新后的模型需要在MQA链中提供级联变化。以往的方法简单地采用了混合提示来指导LLMs进行多个推理任务的顺序执行，包括问题分解、答案生成和通过与编辑的事实进行比较的冲突检查。然而，这些功能多样的推理任务的耦合抑制了LLMs在理解和回答问题方面的优势，同时干扰了它们在冲突检查的不熟练任务上。

    arXiv:2312.15194v2 Announce Type: replace  Abstract: Multi-hop question answering (MQA) is one of the challenging tasks to evaluate machine's comprehension and reasoning abilities, where large language models (LLMs) have widely achieved the human-comparable performance. Due to the dynamics of knowledge facts in real world, knowledge editing has been explored to update model with the up-to-date facts while avoiding expensive re-training or fine-tuning. Starting from the edited fact, the updated model needs to provide cascading changes in the chain of MQA. The previous art simply adopts a mix-up prompt to instruct LLMs conducting multiple reasoning tasks sequentially, including question decomposition, answer generation, and conflict checking via comparing with edited facts. However, the coupling of these functionally-diverse reasoning tasks inhibits LLMs' advantages in comprehending and answering questions while disturbing them with the unskilled task of conflict checking. We thus propos
    
[^86]: MAC-SQL: 一种用于文本到SQL的多智能体协作框架

    MAC-SQL: A Multi-Agent Collaborative Framework for Text-to-SQL

    [https://arxiv.org/abs/2312.11242](https://arxiv.org/abs/2312.11242)

    MAC-SQL是一种基于LLM的多智能体协作框架，针对文本到SQL任务中的巨大数据库和复杂用户问题，通过核心分解器智能体和辅助智能体的协作，利用外部工具和模型进行解析和修正，实现了高效的文本到SQL生成和查询解析。

    

    最近基于LLM的文本到SQL方法通常在“巨大”的数据库和需要多步推理的复杂用户问题上遭受严重的性能下降。此外，大多数现有方法忽视了利用外部工具和模型协作的LLM的重要意义。为了解决这些挑战，我们引入了MAC-SQL，一种新颖的基于LLM的多智能体协作框架。我们的框架包括一个核心分解器智能体，用于进行带有少样本思维链的文本到SQL生成，同时还有两个辅助智能体，利用外部工具或模型获取较小的子数据库并修正错误的SQL查询。分解器智能体与辅助智能体合作，根据需要激活，并可以扩展以适应新的特性或工具，以实现有效的文本到SQL解析。在我们的框架中，我们最初利用GPT-4作为强大的LLM骨干来完成所有智能体任务，以确定...

    arXiv:2312.11242v3 Announce Type: replace  Abstract: Recent LLM-based Text-to-SQL methods usually suffer from significant performance degradation on ``huge" databases and complex user questions that require multi-step reasoning. Moreover, most existing methods neglect the crucial significance of LLMs utilizing external tools and model collaboration. To address these challenges, we introduce MAC-SQL, a novel LLM-based multi-agent collaborative framework. Our framework comprises a core decomposer agent for Text-to-SQL generation with few-shot chain-of-thought reasoning, accompanied by two auxiliary agents that utilize external tools or models to acquire smaller sub-databases and refine erroneous SQL queries. The decomposer agent collaborates with auxiliary agents, which are activated as needed and can be expanded to accommodate new features or tools for effective Text-to-SQL parsing. In our framework, We initially leverage GPT-4 as the strong backbone LLM for all agent tasks to determine
    
[^87]: 更少即更多：通过强化上下文修剪提升LLM推理能力

    Fewer is More: Boosting LLM Reasoning with Reinforced Context Pruning

    [https://arxiv.org/abs/2312.08901](https://arxiv.org/abs/2312.08901)

    CoT-Influx是一种通过强化上下文修剪来提升LLM数学推理能力的方法，通过最大化有效和简明的示例输入，显著优于其他提示方法。

    

    大型语言模型（LLM）展现了令人印象深刻的能力，但在数学推理方面仍存在困难。在这项工作中，我们提出了CoT-Influx，一种将少样本链式思维（CoT）学习推向极限以改善LLM数学推理能力的新方法。由于观察到在提示中添加更简明的CoT示例可以提高LLM推理表现，CoT-Influx采用了一种从粗糙到精细的修剪器来最大化有效和简明的CoT示例的输入。修剪器首先选择尽可能多的关键CoT示例，然后修剪无关紧要的标记以适应上下文窗口。使用难度级别和推理步骤多样的数学推理数据集来训练修剪器，同时采用了专门针对数学的强化学习方法。结果是，通过在令牌中启用双倍上下文窗口大小的CoT示例，CoT-Influx在各种提示基准方法上表现显著优于其他方法。

    arXiv:2312.08901v3 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) have shown impressive capabilities, yet they still struggle with math reasoning. In this work, we propose CoT-Influx, a novel approach that pushes the boundary of few-shot Chain-of-Thoughts (CoT) learning to improve LLM mathematical reasoning. Motivated by the observation that adding more concise CoT examples in the prompt can improve LLM reasoning performance, CoT-Influx employs a coarse-to-fine pruner to maximize the input of effective and concise CoT examples. The pruner first selects as many crucial CoT examples as possible and then prunes unimportant tokens to fit the context window. A math reasoning dataset with diverse difficulty levels and reasoning steps is used to train the pruner, along with a math-specialized reinforcement learning approach. As a result, by enabling more CoT examples with double the context window size in tokens, CoT-Influx significantly outperforms various prompting bas
    
[^88]: MasonTigers@LT-EDI-2024：一种用于检测社交媒体评论中的恐同和恶意性别认知的集成方法

    MasonTigers@LT-EDI-2024: An Ensemble Approach towards Detecting Homophobia and Transphobia in Social Media Comments. (arXiv:2401.14681v1 [cs.CL])

    [http://arxiv.org/abs/2401.14681](http://arxiv.org/abs/2401.14681)

    MasonTigers@LT-EDI-2024团队提出了一种集成方法，用于在十种语言中检测社交媒体评论中的恐同和恶意性别认知，取得了较好的性能表现。

    

    本文描述了我们在LT-EDI 2024研讨会的任务2中针对十种语言检测恐同和/或恶意性别认知的方法和结果。我们的方法包括单语言transformer和集成方法，利用每种方法的优势来提高模型的性能。集成模型表现良好，我们的团队MasonTigers在十种语言中的八种中都排名前五，以宏观F1分数衡量。我们的工作强调了集成方法在多语言情境下的有效性，解决了语言特定任务的复杂性。

    In this paper, we describe our approaches and results for Task 2 of the LT-EDI 2024 Workshop, aimed at detecting homophobia and/or transphobia across ten languages. Our methodologies include monolingual transformers and ensemble methods, capitalizing on the strengths of each to enhance the performance of the models. The ensemble models worked well, placing our team, MasonTigers, in the top five for eight of the ten languages, as measured by the macro F1 score. Our work emphasizes the efficacy of ensemble methods in multilingual scenarios, addressing the complexities of language-specific tasks.
    
[^89]: 大型语言模型的自我解释是否可靠?

    Are self-explanations from Large Language Models faithful?. (arXiv:2401.07927v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2401.07927](http://arxiv.org/abs/2401.07927)

    大型语言模型的自我解释是否可靠是一个重要的AI安全考虑因素，我们提出使用自洽性检测作为评估其可靠性和解释能力的方法。

    

    经过训练的大型语言模型在许多任务上表现出色，甚至能够提供其行为的解释。由于这些模型对公众是直接可访问的，因此存在这样的风险，即令人信服但错误的解释可能导致对大型语言模型的无支撑的自信。因此，解释能力和可靠性是AI安全的重要考虑因素。评估自我解释的可靠性和可解释性是一项具有挑战性的任务，因为这些模型对于人类来说过于复杂，无法注释什么是正确的解释。为了解决这个问题，我们提出使用自洽性检测作为可靠性的衡量指标。例如，如果一个大型语言模型说某组词对于做出预测很重要，那么在没有这些词的情况下，它应该无法做出相同的预测。虽然自洽性检测是一种常见的可靠性方法，但之前尚未应用于大型语言模型的自我解释中。我们将自洽性检测应用于...

    Instruction-tuned large language models (LLMs) excel at many tasks, and will even provide explanations for their behavior. Since these models are directly accessible to the public, there is a risk that convincing and wrong explanations can lead to unsupported confidence in LLMs. Therefore, interpretability-faithfulness of self-explanations is an important consideration for AI Safety. Assessing the interpretability-faithfulness of these explanations, termed self-explanations, is challenging as the models are too complex for humans to annotate what is a correct explanation. To address this, we propose employing self-consistency checks as a measure of faithfulness. For example, if an LLM says a set of words is important for making a prediction, then it should not be able to make the same prediction without these words. While self-consistency checks are a common approach to faithfulness, they have not previously been applied to LLM's self-explanations. We apply self-consistency checks to t
    
[^90]: 多样性激励对LLM文本增强中样本多样性和下游模型性能的影响

    Effects of diversity incentives on sample diversity and downstream model performance in LLM-based text augmentation. (arXiv:2401.06643v1 [cs.CL])

    [http://arxiv.org/abs/2401.06643](http://arxiv.org/abs/2401.06643)

    本研究评估了三种文本多样性激励方法对LLM文本增强中生成文本的词汇多样性和下游模型性能的影响。结果表明，使用禁忌词能够最大程度地增加多样性，而使用先前创建的改写作为提示时，下游模型的性能最高。

    

    最新的生成式大型语言模型（LLM）在数据增强任务中找到了应用，其中少量文本样本被LLM改写，然后用于模型的微调。然而，需要进一步研究不同提示、种子数据选择策略、过滤方法或模型设置对改写数据（和下游模型）质量的影响。在本研究中，我们调查了在众包中已经建立良好的三种文本多样性激励方法：禁忌词、通过先前异常解的提示和通过先前异常解的链接。使用这些激励方法作为指导LLM增补文本数据集的一部分，我们探测它们对生成的文本的词汇多样性和下游模型性能的影响。我们比较了5种不同的LLM和6个数据集的影响。结果显示，禁忌词能够最大程度地增加多样性，而使用先前创建的改写作为提示时，下游模型的性能最高。

    The latest generative large language models (LLMs) have found their application in data augmentation tasks, where small numbers of text samples are LLM-paraphrased and then used to fine-tune the model. However, more research is needed to assess how different prompts, seed data selection strategies, filtering methods, or model settings affect the quality of paraphrased data (and downstream models). In this study, we investigate three text diversity incentive methods well established in crowdsourcing: taboo words, hints by previous outlier solutions, and chaining on previous outlier solutions. Using these incentive methods as part of instructions to LLMs augmenting text datasets, we measure their effects on generated texts' lexical diversity and downstream model performance. We compare the effects over 5 different LLMs and 6 datasets. We show that diversity is most increased by taboo words, while downstream model performance is highest when previously created paraphrases are used as hint
    
[^91]: 适应大型语言模型进行文档级机器翻译的研究

    Adapting Large Language Models for Document-Level Machine Translation. (arXiv:2401.06468v1 [cs.CL])

    [http://arxiv.org/abs/2401.06468](http://arxiv.org/abs/2401.06468)

    本文研究了适应大型语言模型进行文档级机器翻译的过程。实验结果显示，这些专门的模型在某些情况下超过了GPT-4的翻译性能，但在其他情况下仍然存在离标翻译问题，需要进一步改进和探索。

    

    大型语言模型（LLMs）在各种自然语言处理（NLP）任务中取得了重要进展。最近的研究表明，在任务特定的微调之后，中等规模的LLMs往往胜过其更大的对应模型。在这项工作中，我们深入研究了将LLMs调整为特定语言对的文档级机器翻译（DocMT）的过程。首先，我们探讨了提示策略对下游翻译性能的影响。然后，我们进行了大量实验，使用了两种微调方法、三种LLM主干和18个涉及九种语言对的翻译任务。我们的研究结果表明，在某些情况下，这些专门的模型甚至在翻译性能上超过了GPT-4，而在其他情况下，即使它们专门在双语平行文档上进行了微调，仍然明显存在离标翻译问题。此外，我们对这些针对DocMT量身定制的LLMs进行了深入分析，探讨了如翻译准确度改善、多源信息整合等各个方面。

    Large language models (LLMs) have made significant strides in various natural language processing (NLP) tasks. Recent research shows that the moderately-sized LLMs often outperform their larger counterparts after task-specific fine-tuning. In this work, we delve into the process of adapting LLMs to specialize in document-level machine translation (DocMT) for a specific language pair. Firstly, we explore how prompt strategies affect downstream translation performance. Then, we conduct extensive experiments with two fine-tuning methods, three LLM backbones, and 18 translation tasks across nine language pairs. Our findings indicate that in some cases, these specialized models even surpass GPT-4 in translation performance, while they still significantly suffer from the off-target translation issue in others, even if they are exclusively fine-tuned on bilingual parallel documents. Furthermore, we provide an in-depth analysis of these LLMs tailored for DocMT, exploring aspects such as transl
    
[^92]: 你的预训练模型有改进吗？一种基于多头后验的方法

    Has Your Pretrained Model Improved? A Multi-head Posterior Based Approach. (arXiv:2401.02987v1 [cs.CL])

    [http://arxiv.org/abs/2401.02987](http://arxiv.org/abs/2401.02987)

    本研究提出一种基于多头后验的方法，通过利用实体的元特征和模型的表示之间的一致性作为度量标准，有效评估预训练模型在各个领域的表现。

    

    预训练模型的出现对自然语言处理（NLP）、计算机视觉和关系型数据集等领域产生了显著影响。传统上，这些模型通过下游任务进行评估。然而，这引发了如何更高效、更有效地评估这些模型的问题。在本研究中，我们探索了一种新颖的方法，即利用与每个实体相关的元特征作为世界知识的来源，并利用模型的实体表示。我们提出使用这些表示和元特征之间的一致性作为评估预训练模型的度量标准。我们的方法在各个领域表现出了有效性，包括具有关系型数据集、大型语言模型和图像模型的模型。

    The emergence of pretrained models has significantly impacted from Natural Language Processing (NLP) and Computer Vision to relational datasets. Traditionally, these models are assessed through fine-tuned downstream tasks. However, this raises the question of how to evaluate these models more efficiently and more effectively. In this study, we explore a novel approach where we leverage the meta features associated with each entity as a source of worldly knowledge and employ entity representations from the models. We propose using the consistency between these representations and the meta features as a metric for evaluating pretrained models. Our method's effectiveness is demonstrated across various domains, including models with relational datasets, large language models and images models.
    
[^93]: 大语言模型的零样本位置去偏方法

    Zero-Shot Position Debiasing for Large Language Models. (arXiv:2401.01218v1 [cs.CL])

    [http://arxiv.org/abs/2401.01218](http://arxiv.org/abs/2401.01218)

    本文提出了一种零样本位置去偏方法（ZOE）来降低大语言模型（LLMs）的位置偏差问题，该方法利用预训练的LLMs的无监督响应进行去偏。实验证实ZOE在多个数据集和任务中均表现出优异的性能。

    

    微调已被证明是改善大语言模型（LLMs）领域性能的有效方法。然而，LLMs可能适应数据集偏见和预测的捷径，导致生成性能差。实验结果显示，LLMs容易表现出位置偏差，即利用位于开头或末尾或输入中特定位置线索的信息。现有的减轻位置偏差的工作需要外部偏差知识或带注释的非偏倚样本，在实际中不太实用。在这项工作中，我们提出了一种零样本位置去偏（ZOE）框架对LLMs进行位置去偏。ZOE利用预训练的LLMs的无监督响应进行去偏，因此不需要任何外部知识或数据集。为了提高无监督响应的质量，我们提出了一种主从对齐（MSA）模块来修剪这些响应。对八个数据集和五个任务的实验表明，ZOE始终优于其他方法。

    Fine-tuning has been demonstrated to be an effective method to improve the domain performance of large language models (LLMs). However, LLMs might fit the dataset bias and shortcuts for prediction, leading to poor generation performance. Experimental result shows that LLMs are prone to exhibit position bias, i.e., leveraging information positioned at the beginning or end, or specific positional cues within the input. Existing works on mitigating position bias require external bias knowledge or annotated non-biased samples, which is unpractical in reality. In this work, we propose a zero-shot position debiasing (ZOE) framework to mitigate position bias for LLMs. ZOE leverages unsupervised responses from pre-trained LLMs for debiasing, thus without any external knowledge or datasets. To improve the quality of unsupervised responses, we propose a master-slave alignment (MSA) module to prune these responses. Experiments on eight datasets and five tasks show that ZOE consistently outperform
    
[^94]: 使用大型语言模型的同时机器翻译

    Simultaneous Machine Translation with Large Language Models. (arXiv:2309.06706v1 [cs.CL])

    [http://arxiv.org/abs/2309.06706](http://arxiv.org/abs/2309.06706)

    本文研究了使用大型语言模型进行同时机器翻译的可行性，通过引入混合策略，并进行有监督微调，取得了显著的性能改进。

    

    通过对话式交互，大型语言模型 (LLM) 已经展示出解决各种自然语言处理任务的能力。例如，研究表明，LLM可以在高资源语言的离线机器翻译任务中取得竞争性的性能。然而，将LLM应用于同时机器翻译 (SimulMT) 面临许多挑战，包括与不同解码模式产生的训练-推理不匹配问题。本文探索了利用LLM进行SimulMT的可行性。在传统方法的基础上，我们引入了一个简单而有效的混合策略，使LLM能够在不需要额外训练的情况下参与SimulMT。此外，在对全句和前缀句子进行有监督微调后，该模型展示出了显著的性能改进。我们使用MUST-C数据集上的九种语言对进行实验，结果表明LLM可以实现同时机器翻译。

    Large language models (LLM) have demonstrated their abilities to solve various natural language processing tasks through dialogue-based interactions. For instance, research indicates that LLMs can achieve competitive performance in offline machine translation tasks for high-resource languages. However, applying LLMs to simultaneous machine translation (SimulMT) poses many challenges, including issues related to the training-inference mismatch arising from different decoding patterns. In this paper, we explore the feasibility of utilizing LLMs for SimulMT. Building upon conventional approaches, we introduce a simple yet effective mixture policy that enables LLMs to engage in SimulMT without requiring additional training. Furthermore, after Supervised Fine-Tuning (SFT) on a mixture of full and prefix sentences, the model exhibits significant performance improvements. Our experiments, conducted with Llama2-7B-chat on nine language pairs from the MUST-C dataset, demonstrate that LLM can ac
    
[^95]: 简单的合成数据减少了大型语言模型中的阿谀奉承行为

    Simple synthetic data reduces sycophancy in large language models. (arXiv:2308.03958v1 [cs.CL])

    [http://arxiv.org/abs/2308.03958](http://arxiv.org/abs/2308.03958)

    本文研究了语言模型中阿谀奉承行为的普遍性，并提出了一个简单的合成数据干预方法来减少这种行为。

    

    阿谀奉承是一种不可取的行为，模型会根据用户的观点调整回应，即使这个观点在客观上是不正确的（例如，一旦用户透露他们是自由主义者，模型会适应自由主义观点）。在本文中，我们研究了语言模型中阿谀奉承行为的普遍性，并提出了一个简单的合成数据干预来减少这种行为。首先，在一组三个阿谀奉承任务中（Perez等，2022），模型被要求对没有正确答案的陈述（例如，政治问题）发表意见，我们观察到模型的扩展和指令调优显著增加了PaLM模型（多达540B参数）的阿谀奉承行为。其次，我们将阿谀奉承评估扩展到一些明显不正确的加法陈述，发现尽管模型知道这些陈述是错误的，但如果用户也这么认为，语言模型仍会同意这些陈述。为了减少阿谀奉承，我们提出了一个简单的合成数据干预方法，该方法利用公共的NLP任务。

    Sycophancy is an undesirable behavior where models tailor their responses to follow a human user's view even when that view is not objectively correct (e.g., adapting liberal views once a user reveals that they are liberal). In this paper, we study the prevalence of sycophancy in language models and propose a simple synthetic-data intervention to reduce this behavior.  First, on a set of three sycophancy tasks (Perez et al., 2022) where models are asked for an opinion on statements with no correct answers (e.g., politics), we observe that both model scaling and instruction tuning significantly increase sycophancy for PaLM models up to 540B parameters. Second, we extend sycophancy evaluations to simple addition statements that are objectively incorrect, finding that despite knowing that these statements are wrong, language models will still agree with them if the user does as well.  To reduce sycophancy, we present a straightforward synthetic-data intervention that takes public NLP task
    
[^96]: ODD: 一份基于自然语言处理的药物滥用异常行为检测的基准数据集

    ODD: A Benchmark Dataset for the NLP-based Opioid Related Aberrant Behavior Detection. (arXiv:2307.02591v1 [cs.CL])

    [http://arxiv.org/abs/2307.02591](http://arxiv.org/abs/2307.02591)

    这个研究介绍了一份名为ODD的新型基准数据集，用于通过分析患者的电子健康记录笔记，检测和分类药物滥用异常行为。这个数据集在药物相关病例的自然语言处理研究中具有重要的创新和贡献。

    

    药物滥用异常行为（ORAB）是防止药物过量的新风险因素。以往，ORAB主要通过调查结果和药物给予监测进行评估。然而，这些方法无法扩展，并不能涵盖所有异常行为的范围。然而，ORAB在电子健康记录笔记中广泛有记录。本文介绍了一个名为ODD的新型生物医学自然语言处理基准数据集，用于ORAB检测。ODD是一个专家注释的数据集，包括750多个公开可用的电子健康记录笔记。ODD旨在从患者的电子健康记录笔记中识别ORAB，并将其分类为九个类别：1）已确认异常行为，2）暗示的异常行为，3）阿片类药物，4）适应症，5）已诊断的阿片制剂依赖，6）苯二氮平类药物，7）药物变化，8）与中枢神经系统相关，9）社会健康决定因素。

    Opioid related aberrant behaviors (ORAB) present novel risk factors for opioid overdose. Previously, ORAB have been mainly assessed by survey results and by monitoring drug administrations. Such methods however, cannot scale up and do not cover the entire spectrum of aberrant behaviors. On the other hand, ORAB are widely documented in electronic health record notes. This paper introduces a novel biomedical natural language processing benchmark dataset named ODD, for ORAB Detection Dataset. ODD is an expert-annotated dataset comprising of more than 750 publicly available EHR notes. ODD has been designed to identify ORAB from patients' EHR notes and classify them into nine categories; 1) Confirmed Aberrant Behavior, 2) Suggested Aberrant Behavior, 3) Opioids, 4) Indication, 5) Diagnosed opioid dependency, 6) Benzodiapines, 7) Medication Changes, 8) Central Nervous System-related, and 9) Social Determinants of Health. We explored two state-of-the-art natural language processing (NLP) mode
    
[^97]: LLM和抽象推理语料库：成功、失败与基于对象表示的重要性

    LLMs and the Abstraction and Reasoning Corpus: Successes, Failures, and the Importance of Object-based Representations. (arXiv:2305.18354v1 [cs.CL])

    [http://arxiv.org/abs/2305.18354](http://arxiv.org/abs/2305.18354)

    本文通过分析GPT模型在抽象推理语料库上的表现，发现GPT在抽象推理任务中存在需要核心概念“核心知识”的限制。通过使用基于对象的表示方法和新的1D-ARC基准，GPT在抽象推理任务中取得了较好的表现。

    

    一个大型语言模型（LLM）能否解决简单的抽象推理问题？我们通过系统地分析GPT在抽象推理语料库（ARC）上的表现来探索这个广泛的问题，这是一个代表性的基准，从有限的例子中要求我们有些关于概念（如对象、目标状态、计数和基本几何）的“核心知识”以解决问题。当使用文本编码对二维输入输出网格的ARC任务进行编码时，GPT-4仅解决了50个最简单的任务中的13个。我们的失败分析显示，GPT-4识别对象并推理它们的能力受到表示任务中对象的文本的顺序性的显著影响。为了验证这个假设，我们设计了一个新的基准，1D-ARC，它由更有利于基于GPT的推理的一维（类似数组）任务组成，在这些任务上，GPT的表现确实比在（2D）ARC上更好。为了减轻这个问题，我们提出了一种新的基于对象的编码方案，它保留了对象之间的基本空间关系并实现了更好的推理。使用这种编码，GPT-4在ARC上的成功率大大提高，达到了45/50。我们的工作强调了使用基于对象的表示方法进行抽象推理任务的重要性，并揭示了LLM基于推理系统的局限性和机遇。

    Can a Large Language Model (LLM) solve simple abstract reasoning problems? We explore this broad question through a systematic analysis of GPT on the Abstraction and Reasoning Corpus (ARC), a representative benchmark of abstract reasoning ability from limited examples in which solutions require some "core knowledge" of concepts such as objects, goal states, counting, and basic geometry. GPT-4 solves only 13/50 of the most straightforward ARC tasks when using textual encodings for their two-dimensional input-output grids. Our failure analysis reveals that GPT-4's capacity to identify objects and reason about them is significantly influenced by the sequential nature of the text that represents an object within a text encoding of a task. To test this hypothesis, we design a new benchmark, the 1D-ARC, which consists of one-dimensional (array-like) tasks that are more conducive to GPT-based reasoning, and where it indeed performs better than on the (2D) ARC. To alleviate this issue, we prop
    
[^98]: DAPR：文档感知段落检索的基准测试

    DAPR: A Benchmark on Document-Aware Passage Retrieval. (arXiv:2305.13915v1 [cs.IR])

    [http://arxiv.org/abs/2305.13915](http://arxiv.org/abs/2305.13915)

    DAPR是一个文档感知段落检索的基准测试，挑战在于如何从长文档中找到正确的段落并返回准确结果。

    

    最近的神经检索主要关注短文本的排名，并且在处理长文档方面存在挑战。现有的工作主要评估排名段落或整个文档。然而，许多情况下，用户希望从庞大的语料库中找到长文档中的相关段落，例如法律案例，研究论文等，此时段落往往提供很少的文档上下文，这就挑战了当前的方法找到正确的文档并返回准确的结果。为了填补这个空白，我们提出并命名了Document-Aware Passage Retrieval（DAPR）任务，并构建了一个包括来自不同领域的多个数据集的基准测试，涵盖了DAPR和整个文档检索。在实验中，我们通过不同的方法，包括在文档摘要中添加文档级别的内容，汇总段落表示和使用BM25进行混合检索，扩展了最先进的神经段落检索器。这个混合检索系统，总体基准测试显示，我们提出的DAPR任务是一个具有挑战性和重要性的问题，需要进一步研究。

    Recent neural retrieval mainly focuses on ranking short texts and is challenged with long documents. Existing work mainly evaluates either ranking passages or whole documents. However, there are many cases where the users want to find a relevant passage within a long document from a huge corpus, e.g. legal cases, research papers, etc. In this scenario, the passage often provides little document context and thus challenges the current approaches to finding the correct document and returning accurate results. To fill this gap, we propose and name this task Document-Aware Passage Retrieval (DAPR) and build a benchmark including multiple datasets from various domains, covering both DAPR and whole-document retrieval. In experiments, we extend the state-of-the-art neural passage retrievers with document-level context via different approaches including prepending document summary, pooling over passage representations, and hybrid retrieval with BM25. The hybrid-retrieval systems, the overall b
    
[^99]: 利用跨模态适配器向预训练语言模型注入多功能高效的视觉知识

    Towards Versatile and Efficient Visual Knowledge Injection into Pre-trained Language Models with Cross-Modal Adapters. (arXiv:2305.07358v1 [cs.CL])

    [http://arxiv.org/abs/2305.07358](http://arxiv.org/abs/2305.07358)

    本文提出了X-adapter插拔式模块，利用多模态视觉语言模型，高效地向预训练语言模型注入视觉知识。

    

    人类通过多模态知识学习语言，然而现有的大多数预训练语言模型（PLMs）仅支持文本预训练。本文提出了插拔式模块X-adapter，它能够根据多模态视觉语言模型（VLMs）的对齐视觉和文本知识，灵活高效地向PLMs注入视觉知识。 X-adapter包含两个子模块V-expert和T-expert，可以根据下游任务激活不同的子模块，来融合VLMs的图像和文本表示。

    Humans learn language via multi-modal knowledge. However, due to the text-only pre-training scheme, most existing pre-trained language models (PLMs) are hindered from the multi-modal information.  To inject visual knowledge into PLMs, existing methods incorporate either the text or image encoder of vision-language models (VLMs) to encode the visual information and update all the original parameters of PLMs for knowledge fusion.  In this paper, we propose a new plug-and-play module, X-adapter, to flexibly leverage the aligned visual and textual knowledge learned in pre-trained VLMs and efficiently inject them into PLMs.  Specifically, we insert X-adapters into PLMs, and only the added parameters are updated during adaptation.  To fully exploit the potential in VLMs, X-adapters consist of two sub-modules, V-expert and T-expert, to fuse VLMs' image and text representations, respectively.  We can opt for activating different sub-modules depending on the downstream tasks.  Experimental resu
    
[^100]: 自控内存系统释放大规模语言模型的无限输入容量

    Unleashing Infinite-Length Input Capacity for Large-scale Language Models with Self-Controlled Memory System. (arXiv:2304.13343v1 [cs.CL])

    [http://arxiv.org/abs/2304.13343](http://arxiv.org/abs/2304.13343)

    该论文提出了一种自控内存系统，可以使大规模语言模型能够处理任意长度的输入，从而显著提高模型的性能表现。

    

    大规模语言模型（LLMs）受制于无法处理过长的输入。为了解决这个问题，我们提出了自控内存（SCM）系统，以释放大规模语言模型的无限输入容量。我们的SCM系统由三个关键模块组成：语言模型代理、内存流和内存控制器。语言模型代理迭代地处理超长输入，并将所有历史信息存储在内存流中。内存控制器为代理提供长期存储器（归档存储器）和短期存储器（闪存），以生成精确连贯的响应。控制器确定应激活哪些来自归档存储器的记忆，并如何将它们合并到模型输入中。我们的SCM系统可以与任何LLMs集成，以使它们能够处理超长文本而无需修改或微调。实验结果表明，我们的SCM系统使得LLMs能够处理长度高达8192个令牌的输入，实现了在多个基准数据集上的最佳表现，证明了它在提高大规模语言模型性能方面的有效性。

    Large-scale Language Models (LLMs) are constrained by their inability to process lengthy inputs. To address this limitation, we propose the Self-Controlled Memory (SCM) system to unleash infinite-length input capacity for large-scale language models. Our SCM system is composed of three key modules: the language model agent, the memory stream, and the memory controller. The language model agent iteratively processes ultra-long inputs and stores all historical information in the memory stream. The memory controller provides the agent with both long-term memory (archived memory) and short-term memory (flash memory) to generate precise and coherent responses. The controller determines which memories from archived memory should be activated and how to incorporate them into the model input. Our SCM system can be integrated with any LLMs to enable them to process ultra-long texts without any modification or fine-tuning. Experimental results show that our SCM system enables LLMs, which are not
    
[^101]: 人类和机器学习模型的分词可追溯性：一个注释研究

    Tokenization Tractability for Human and Machine Learning Model: An Annotation Study. (arXiv:2304.10813v1 [cs.CL])

    [http://arxiv.org/abs/2304.10813](http://arxiv.org/abs/2304.10813)

    研究比较了六种分词方法，并发现人类可追溯的分词与机器学习模型中的分词不一定相同。

    

    人类可追溯的分词对于机器学习模型是否也是可追溯的？本研究探讨了人类可追溯的分词（如适当性和可读性）与机器学习模型中的分词（如在NLP任务中的性能）之间的关系。我们在日语常识问答数据集（JGLUE的JCommmonsenseQA）中比较了六种分词方法。我们使用不同的分词器对问答数据集中的问题文本进行分词，并比较了人类标注者和机器学习模型的性能。此外，我们分析了性能、分词的适当性和回答问题的响应时间之间的关系。本文提供了一个定量调查结果，显示出对于人类和机器学习模型来说，可追溯的分词不一定相同。

    Is tractable tokenization for humans also tractable for machine learning models? This study investigates relations between tractable tokenization for humans (e.g., appropriateness and readability) and one for models of machine learning (e.g., performance on an NLP task). We compared six tokenization methods on the Japanese commonsense question-answering dataset (JCommmonsenseQA in JGLUE). We tokenized question texts of the QA dataset with different tokenizers and compared the performance of human annotators and machine-learning models. Besides,we analyze relationships among the performance, appropriateness of tokenization, and response time to questions. This paper provides a quantitative investigation result that shows the tractable tokenizations for humans and machine learning models are not necessarily the same as each other.
    

