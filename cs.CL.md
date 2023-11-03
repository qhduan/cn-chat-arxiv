# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Server-side Rescoring of Spoken Entity-centric Knowledge Queries for Virtual Assistants.](http://arxiv.org/abs/2311.01398) | 本文研究了服务器端对语音中心化知识查询进行重新评分的建模策略，并通过整合各种服务器端语言模型，显著改善了各种实体中心化查询子族群的识别错误率。此外，模型融合和使用领域特定数据训练的语言模型对于提升VA ASR系统的性能也起到了积极的作用。 |
| [^2] | [Can Language Models Be Tricked by Language Illusions? Easier with Syntax, Harder with Semantics.](http://arxiv.org/abs/2311.01386) | 通过研究语言模型在与"语言幻觉"相关的判断中的行为，我们发现语言模型更容易受到结构依赖性的幻觉的影响，而在语义方面则较困难。 |
| [^3] | [GPT-4V(ision) as a Generalist Evaluator for Vision-Language Tasks.](http://arxiv.org/abs/2311.01361) | GPT-4V作为一个通用评估器在视觉-语言任务中表现出有希望的一致性，展示了多模态LLMs作为评估器的巨大潜力。 |
| [^4] | [Better Together: Enhancing Generative Knowledge Graph Completion with Language Models and Neighborhood Information.](http://arxiv.org/abs/2311.01326) | 本研究提出了将节点邻居作为额外信息加入语言模型，以改进知识图谱完善方法。在归纳和传递式Wikidata子集上，我们的方法优于传统方法和基于语言模型的KGC方法。邻居信息对模型预测具有重要影响。 |
| [^5] | [The Effect of Scaling, Retrieval Augmentation and Form on the Factual Consistency of Language Models.](http://arxiv.org/abs/2311.01307) | 本研究通过缩放和检索增强两种策略，分析了语言模型事实一致性的不一致性原因，并发现检索增强策略效果更好。句法形式和其他评估任务的构造对于一致性有影响。 |
| [^6] | [AWEQ: Post-Training Quantization with Activation-Weight Equalization for Large Language Models.](http://arxiv.org/abs/2311.01305) | AWEQ是一种后训练量化和激活权重均衡方法，能够在大型语言模型中实现超低位量化和8-bit权重和激活量化，并通过改进的均衡方法减小量化偏差误差，提高模型的鲁棒性。 |
| [^7] | [FlashDecoding++: Faster Large Language Model Inference on GPUs.](http://arxiv.org/abs/2311.01282) | FlashDecoding++是一种快速的LLM推理引擎，通过解决同步部分softmax更新、未充分利用扁平GEMM计算和静态数据流导致的性能损失等挑战，实现了大规模语言模型推理的加速。 |
| [^8] | [Finding Common Ground: Annotating and Predicting Common Ground in Spoken Conversations.](http://arxiv.org/abs/2311.01273) | 本文介绍了一种新的标注和语料库，用于捕捉口语对话中的共同基础，并通过提取命题和跟踪它们在共同基础中的状态的实验，对口语对话中的共同基础进行研究。 |
| [^9] | [People Make Better Edits: Measuring the Efficacy of LLM-Generated Counterfactually Augmented Data for Harmful Language Detection.](http://arxiv.org/abs/2311.01270) | 本论文通过自动生成的反事实增强数据（CADs）与手动生成的CADs进行比较，评估它们在提高模型鲁棒性方面的效果。结果显示，手动生成的CADs仍然是最有效的方法。 |
| [^10] | [An energy-based comparative analysis of common approaches to text classification in the Legal domain.](http://arxiv.org/abs/2311.01256) | 本研究通过在法律领域的文本分类任务上进行比较分析，综合考虑性能和能源消耗等指标，探讨了大型语言模型与传统方法的优劣，并强调了在性能相近的情况下应重视生产成本、能源消耗和碳足迹等方面的考量。 |
| [^11] | [A Study of Continual Learning Under Language Shift.](http://arxiv.org/abs/2311.01200) | 本文研究了持续学习在语言转换中的应用，发现在更新语言模型时，前向转移效果较好且与语言顺序无关，但后向转移效果可能取决于新语言的顺序和特征。 |
| [^12] | [CRUSH4SQL: Collective Retrieval Using Schema Hallucination For Text2SQL.](http://arxiv.org/abs/2311.01173) | CRUSH4SQL提出了一个两阶段的检索过程，使用模式幻觉进行Text2SQL的集体检索，通过幻觉的最小数据库模式检索实际模式的子集，解决了大型数据库中的模式子集检索问题。 |
| [^13] | [Generative Input: Towards Next-Generation Input Methods Paradigm.](http://arxiv.org/abs/2311.01166) | 本研究提出了一种新的生成输入范式GeneInput，结合提示和用户反馈进行个性化的输入处理，在中文输入法引擎的构建中实现了最先进的性能。 |
| [^14] | [Weakly Supervised Semantic Parsing with Execution-based Spurious Program Filtering.](http://arxiv.org/abs/2311.01161) | 本研究提出了一种基于程序执行结果的领域无关筛选机制，用于解决弱监督下语义解析中的虚假程序问题 |
| [^15] | [ACES: Translation Accuracy Challenge Sets at WMT 2023.](http://arxiv.org/abs/2311.01153) | 本论文介绍了使用ACES挑战集对WMT 2023中的segment-level评估指标进行了基准测试，发现没有明确的赢家，并且2023年版和2022年版指标之间的性能变化很大。建议指标开发者应该构建不同设计家族的指标集合，并开发更加关注整体性能的指标。 |
| [^16] | [Predicting Question-Answering Performance of Large Language Models through Semantic Consistency.](http://arxiv.org/abs/2311.01152) | 本论文提出了一种框架，通过结合语义一致性度量和其他相关度量，预测大型语言模型(LLMs)对问题的准确回答能力。通过手动创建高质量的事实性问题基准数据集，并应用该框架在五个当代LLM上进行评估，实验结果表明框架的效果优于基准算法。 |
| [^17] | [Revisiting the Knowledge Injection Frameworks.](http://arxiv.org/abs/2311.01150) | 这项研究重新审视了知识注入框架，发现将未对齐的随机知识注入到大型语言模型中可以取得与对齐知识相当甚至更好的结果。研究还提供了一种简单的修正技术来解决这个问题。 |
| [^18] | [Chinesewebtext: Large-scale high-quality Chinese web text extracted with effective evaluation model.](http://arxiv.org/abs/2311.01149) | 本文提出了一个完整的工具链EvalWeb，用于从网络数据中提取干净的中文文本。通过手工制定的规则筛除噪音数据，并使用评估模型为每个文本分配质量分数。 |
| [^19] | [Noise-Robust Fine-Tuning of Pretrained Language Models via External Guidance.](http://arxiv.org/abs/2311.01108) | 通过利用大语言模型的指导，我们提出一种创新方法，使用噪声标签对预训练语言模型进行鲁棒微调。实验证明，我们的方法相比最先进的基线方法在合成和真实噪声数据集上具有优越的性能。 |
| [^20] | [DistilWhisper: Efficient Distillation of Multi-task Speech Models via Language-Specific Experts.](http://arxiv.org/abs/2311.01070) | 本文提出了DistilWhisper方法，通过使用语言特定专家进行轻量级模块化ASR微调和知识蒸馏，成功弥合了多任务语音模型在少数语言上的性能差距，同时保留了多任务和多语言能力的优势。 |
| [^21] | [Multi-dimensional data refining strategy for effective fine-tuning LLMs.](http://arxiv.org/abs/2311.01049) | 本文介绍了一种多维数据精化策略，包括利用现有数据集和生成型AI工具开发数据爬取脚本，用于调优越南语言模型。研究结果表明，使用该策略得到的模型在从提示生成越南新闻文章时表现出良好性能。 |
| [^22] | [Learn to Refuse: Making Large Language Models More Controllable and Reliable through Knowledge Scope Limitation and Refusal Mechanism.](http://arxiv.org/abs/2311.01041) | 本文提出了一种学会拒绝（L2R）的简单而有效的解决方案，通过引入拒绝机制，使大型语言模型（LLMs）能够识别和拒绝难以回答的问题，从而提高模型的可控性和可靠性。 |
| [^23] | [ATHENA: Mathematical Reasoning with Thought Expansion.](http://arxiv.org/abs/2311.01036) | ATHENA是一种基于注意力机制的思维扩展网络架构，通过模拟人类的思维扩展机制来解决数学推理中的挑战，它能够产生合理的思考路径以解决实际世界的数学问题。 |
| [^24] | [Joint Learning of Local and Global Features for Aspect-based Sentiment Classification.](http://arxiv.org/abs/2311.01030) | 该论文提出了一种联合学习局部和全局特征的方法，以应对基于方面的情感分类中的问题。通过设计一个包含高斯掩码层和协方差自注意层的局部编码器，在模型中有效地整合了局部上下文和全局特征，并提供了更好的区分能力。 |
| [^25] | [COPAL-ID: Indonesian Language Reasoning with Local Culture and Nuances.](http://arxiv.org/abs/2311.01012) | COPAL-ID是一个印度尼西亚语言常识推理数据集，与以前的数据集相比，它融入了印尼本土和文化细微差别，提供了更自然的日常因果推理描绘。该数据集对于现有的多语言语言模型来说是一个更大的挑战，但对人类来说很容易。在测试中，最新的开源多语言模型在COPAL-ID上的准确率较低，仅为65.47%。 |
| [^26] | [Replicable Benchmarking of Neural Machine Translation (NMT) on Low-Resource Local Languages in Indonesia.](http://arxiv.org/abs/2311.00998) | 本研究通过在印度尼西亚的低资源本地语言上训练NMT系统的综合分析，解决了神经机器翻译面临的挑战。 尽管计算资源和文本数据有限，但我们的几个NMT系统取得了竞争性性能，与零-shot gpt-3.5-turbo的翻译质量相媲美。 这些发现显著推动了低资源语言的NMT，为研究人员提供了宝贵的指导。 |
| [^27] | [Vision-Language Interpreter for Robot Task Planning.](http://arxiv.org/abs/2311.00967) | 本文提出了一种名为Vision-Language Interpreter（ViLaIn）的新框架，该框架通过使用先进的语言模型和视觉语言模型生成机器人任务描述，并通过符号规划器的错误消息反馈进行改进。实验结果表明ViLaIn和符号规划器能够准确生成有效的机器人计划。 |
| [^28] | [IndoToD: A Multi-Domain Indonesian Benchmark For End-to-End Task-Oriented Dialogue Systems.](http://arxiv.org/abs/2311.00958) | 本文介绍了IndoToD，一个用于印尼语的端到端多领域任务导向对话系统的基准。通过将英语数据集转化为印尼语，我们创建了这个基准，并通过雇佣母语为印尼语的人员进行翻译和数据收集，这为评估印尼语和英语对话系统以及跨语言和双语迁移学习方法提供了有效工具。 |
| [^29] | [Blending Reward Functions via Few Expert Demonstrations for Faithful and Accurate Knowledge-Grounded Dialogue Generation.](http://arxiv.org/abs/2311.00953) | 本研究通过融合准确度指标和忠实度指标的新奖励函数，利用强化学习算法解决了语言模型幻象和知识文本多余信息问题，提供了一种平衡的生成对话回应质量评判方法。 |
| [^30] | [E3 TTS: Easy End-to-End Diffusion-based Text to Speech.](http://arxiv.org/abs/2311.00945) | E3 TTS是一种简单高效的端到端基于扩散的文本到语音模型，不依赖于中间表示，通过扩散过程建模波形的时间结构，能够轻松适应零样本任务。 |
| [^31] | [Task-Agnostic Low-Rank Adapters for Unseen English Dialects.](http://arxiv.org/abs/2311.00915) | HyperLoRA是一种无任务知识的低秩适配器方法，利用专家语言知识通过超网络实现资源高效的适应性，从而提高了对未知英语方言的泛化能力。 |
| [^32] | [Self-Influence Guided Data Reweighting for Language Model Pre-training.](http://arxiv.org/abs/2311.00913) | 提出了一种名为PRESENCE的方法，通过利用自主影响分数作为重要性指标，对语言模型预训练的数据样本进行重加权，促进了模型预训练的新颖性和稳定性。 |
| [^33] | [Re-weighting Tokens: A Simple and Effective Active Learning Strategy for Named Entity Recognition.](http://arxiv.org/abs/2311.00906) | 本文提出了一种重新加权的主动学习策略，通过为每个标记分配动态平滑的权重，解决了命名实体识别中的数据不平衡问题，取得了显著的性能提升。 |
| [^34] | [On The Open Prompt Challenge In Conditional Audio Generation.](http://arxiv.org/abs/2311.00897) | 本文针对条件音频生成中的开放提示挑战，通过重新编写提示并利用文本-音频对齐作为反馈信号，从而改善音频质量，取得了显著的改进。 |
| [^35] | [In-Context Prompt Editing For Conditional Audio Generation.](http://arxiv.org/abs/2311.00895) | 本研究提出了一种基于检索的上下文提示编辑框架，通过利用训练字幕作为示例来改进条件音频生成的音频质量。 |
| [^36] | [Pretraining Data Mixtures Enable Narrow Model Selection Capabilities in Transformer Models.](http://arxiv.org/abs/2311.00871) | Transformer模型通过预训练数据混合实现了狭窄的模型选择能力，能够在上下文中识别和学习不同的任务，但对于任务或函数的处理相对有限。 |
| [^37] | [Automatic Disfluency Detection from Untranscribed Speech.](http://arxiv.org/abs/2311.00867) | 本研究探讨了从未被转录的语音中自动检测语言错乱的语言、声学和多模态方法，并通过评估自动语音识别系统的转录能力，以及将其作为自然语言理解的预处理步骤，来改善临床和非临床应用中的语言识别效果。 |
| [^38] | [Training Dynamics of Contextual N-Grams in Language Models.](http://arxiv.org/abs/2311.00863) | 这篇论文研究了训练过程中上下文N-Gram的动态，发现了上下文神经元存在于更广泛的上下文N-Gram电路中，这被称为二阶电路。在训练早期，这两个电路具有相互独立的功能，只有在它们都形成之后才能组合成一个二阶电路。 |
| [^39] | [Calibrated Seq2seq Models for Efficient and Generalizable Ultra-fine Entity Typing.](http://arxiv.org/abs/2311.00835) | 这项研究介绍了CASENT，一种为超细粒度实体类型定义设计的Seq2seq模型，通过校准置信度分数来预测超细粒度类型。在实验中，该模型在UFET数据集上表现出了优于先前方法的性能。 |
| [^40] | [Construction Artifacts in Metaphor Identification Datasets.](http://arxiv.org/abs/2311.00790) | 该论文研究了现有隐喻识别数据集中的建构偏见问题，并证明基于语言模型的隐喻识别系统可以与使用完整上下文的系统相竞争。 |
| [^41] | [Language Model Training Paradigms for Clinical Feature Embeddings.](http://arxiv.org/abs/2311.00768) | 本研究使用自监督训练范式的语言模型，通过表示学习为临床时间序列推导出高质量的通用临床特征嵌入。通过无监督的降维技术可视化学习到的嵌入，并在MIMIC-III基准测试中验证了它们的有效性。 |
| [^42] | [Challenges for Linguistically-Driven Computer-Based Sign Recognition from Continuous Signing for American Sign Language.](http://arxiv.org/abs/2311.00762) | 运用美国手语数据，论文概述了基于语言驱动的计算机手势识别的挑战，包括手语者间和手语者内的同步变化以及手势结构中的语言规律。 |
| [^43] | [Can Large Language Models Design Accurate Label Functions?.](http://arxiv.org/abs/2311.00739) | 本研究引入了DataSculpt，它是一个利用预训练语言模型自动生成标签函数的交互式框架。通过多种技术和方法的结合，DataSculpt在各种任务和真实数据集上展现了优点和局限性。 |
| [^44] | [tmn at #SMM4H 2023: Comparing Text Preprocessing Techniques for Detecting Tweets Self-reporting a COVID-19 Diagnosis.](http://arxiv.org/abs/2311.00732) | 本文研究了用于检测自我报告COVID-19诊断推文的不同文本预处理技术，通过使用四个基于transformer的模型进行实验，并通过微调语言模型集成获得了比平均值高出4.1%的84.5%的F1得分。 |
| [^45] | [JADE: A Linguistic-based Safety Evaluation Platform for LLM.](http://arxiv.org/abs/2311.00286) | JADE是一种基于语言分析的LLM安全评估平台，能够破坏广泛使用的中文和英文LLM，并生成高度威胁的不安全问题。 |
| [^46] | [Meaning Representations from Trajectories in Autoregressive Models.](http://arxiv.org/abs/2310.18348) | 本文提出了一种从自回归语言模型中提取意义表征的方法，通过考虑输入文本的所有可能轨迹的分布。这种方法可以模拟非对称关系，且在语义相似性任务上优于其他方法。 |
| [^47] | [Improving Zero-shot Reader by Reducing Distractions from Irrelevant Documents in Open-Domain Question Answering.](http://arxiv.org/abs/2310.17490) | 本研究提出了一种通过减少无关文档的干扰来改善开放领域问答中的零样本阅读器的方法。采用了干扰感知的答案选择(DAS)方法，以解决LLMs受到干扰和过度自信的问题。实验结果表明，该方法成功地改善了零样本阅读器的性能，并展现出了优越的可迁移性。 |
| [^48] | [Air-Decoding: Attribute Distribution Reconstruction for Decoding-Time Controllable Text Generation.](http://arxiv.org/abs/2310.14892) | 本文提出了一种名为空气解码的新颖轻量级解码框架，通过重建属性分布来平衡权重，生成更流畅的文本，以解决可控文本生成中的属性坍缩问题。 |
| [^49] | [QUDEVAL: The Evaluation of Questions Under Discussion Discourse Parsing.](http://arxiv.org/abs/2310.14520) | 本文介绍了第一个用于自动评估问句讨论（QUD）语篇解析的框架QUDeval。使用QUDeval数据集，展示了现代语言模型（LLMs）仍然面临解析所有QUD约束的挑战，并且现有的评估指标很差地近似解析器质量。 |
| [^50] | [Language Agents for Detecting Implicit Stereotypes in Text-to-image Models at Scale.](http://arxiv.org/abs/2310.11778) | 本文介绍了一种针对文本到图像模型中检测刻板印象的语言代理架构，可自主调用各种工具来促进整个检测过程，并应用于商业产品和开放文本数据集。 |
| [^51] | [EMO: Earth Mover Distance Optimization for Auto-Regressive Language Modeling.](http://arxiv.org/abs/2310.04691) | EMO提出了地球移动距离优化（EMO）来解决语言模型中的退化现象。EMO利用了地球移动距离的特性，并引入了一个可行的上界来简化训练。经过评估，发现EMO在语言模型上有显著的改进。 |
| [^52] | [LLM and Infrastructure as a Code use case.](http://arxiv.org/abs/2309.01456) | 本文探讨了一种利用生成式语言模型将人类描述转化为代码的解决方案，用于生成和管理Ansible YAML角色和playbooks，以应对云计算和管理方法发展引起的系统构建和维护方法的变革。 |
| [^53] | [A Comprehensive Overview of Large Language Models.](http://arxiv.org/abs/2307.06435) | 大语言模型的综合概述，分析了各种新的架构和训练策略，讨论了LLM的特点和功能，并总结了重要的研究发现和关键的架构和训练策略。 |
| [^54] | [VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models.](http://arxiv.org/abs/2307.05973) | VoxPoser提出了一种新方法，通过组合3D价值映射和语言模型，实现了机器人在多种操作任务下根据自由形式的指令和对象合成机器人轨迹的能力。 |
| [^55] | [Text Alignment Is An Efficient Unified Model for Massive NLP Tasks.](http://arxiv.org/abs/2307.02729) | 本研究提出了一种高效的文本对齐模型，可以应用于广泛的NLP任务，包括文本蕴含、相似性、问答、事实一致性等。通过对RoBERTa进行轻量级微调，可以构建一个更小规模的模型，实现与大型语言模型相当甚至更优的性能。 |
| [^56] | [EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models.](http://arxiv.org/abs/2307.02028) | 该论文介绍了EHRSHOT，一个用于少样本评估基础模型的电子健康记录基准。该论文利用EHRSHOT数据集和预训练模型CLMBR-T-base，为医疗保健ML的发展提供了解决方案。 |
| [^57] | [Meta-Reasoning: Semantics-Symbol Deconstruction For Large Language Models.](http://arxiv.org/abs/2306.17820) | 本论文提出了一种称为“元推理”的方法，它通过使用语义符号解构的方式，将不同推理问题转化为类似的自然语言表示，以提高大型语言模型的推理能力。 |
| [^58] | [Iterated Piecewise Affine (IPA) Approximation for Language Modeling.](http://arxiv.org/abs/2306.12317) | 迭代分段仿射插值（IPA）逼近法可以用于语言建模，与变压器解码器架构类似，并在交叉熵损失下的小序列长度下优于变压器1.5％。 |
| [^59] | [AVIS: Autonomous Visual Information Seeking with Large Language Models.](http://arxiv.org/abs/2306.08129) | 本文提出了一个基于大型语言模型的自主信息检索视觉问答框架AVIS，可以解决视觉问题所需的外部知识获取问题。 |
| [^60] | [Diable: Efficient Dialogue State Tracking as Operations on Tables.](http://arxiv.org/abs/2305.17020) | Diable是一个高效的对话状态跟踪系统，它通过在表格上进行操作来更新对话状态，相比现有方法时间效率提高了2.4倍，同时保持了竞争性的目标准确性。 |
| [^61] | [The Art of SOCRATIC QUESTIONING: Recursive Thinking with Large Language Models.](http://arxiv.org/abs/2305.14999) | SOCRATIC QUESTIONING是一种与大型语言模型合作进行递归思考的算法，通过模拟人类的认知过程，它能够解决复杂推理问题，提高准确性和效率。 |
| [^62] | [Towards Legally Enforceable Hate Speech Detection for Public Forums.](http://arxiv.org/abs/2305.13677) | 本研究提出了一个以法律定义为中心的、可法律强制执行的仇恨言论检测任务，利用法律专家对数据集进行了注释，结合基于零样本和小样本的提示，可以使模型的输出更符合监管者目标。 |
| [^63] | [Open-world Semi-supervised Generalized Relation Discovery Aligned in a Real-world Setting.](http://arxiv.org/abs/2305.13533) | 本文提出了一种新的开放世界关系抽取方法，能够在已知类和新颖类中进行显式和隐式表示的关系分类，在真实场景数据的特征下进行了两个关键改进。 |
| [^64] | [SEAHORSE: A Multilingual, Multifaceted Dataset for Summarization Evaluation.](http://arxiv.org/abs/2305.13194) | SEAHORSE是一个用于多语言、多方面摘要评估的数据集，包含96K个摘要，涵盖6种语言、9个系统和4个数据集。SEAHORSE是一个用于评估学习度量和训练度量的大规模资源。使用SEAHORSE训练的度量在领域外的元评估基准上表现出了强大的性能。 |
| [^65] | [Discovering Universal Geometry in Embeddings with ICA.](http://arxiv.org/abs/2305.13175) | 本研究利用ICA揭示了嵌入中的通用几何结构，证明了每个嵌入可以由少量内在可解释轴的组合表示，并且这些语义轴在不同的语言、算法和模态下保持一致。 |
| [^66] | [Textually Pretrained Speech Language Models.](http://arxiv.org/abs/2305.13009) | 本论文提出了一种使用预训练的文本语言模型训练语音语言模型的方法，通过对模型设计选择和数据集规模的经验性分析，构建了参数数量和训练数据最多的语音语言模型，并引入了两个Spoken版本的文本基准，以进一步改善模型评估和推动未来研究。 |
| [^67] | [TempoSum: Evaluating the Temporal Generalization of Abstractive Summarization.](http://arxiv.org/abs/2305.01951) | 本篇论文提出了 TempoSum 抽象摘要的时间泛化能力基准，通过广泛的人类评估证明了摘要模型中存储的参数化知识对未来数据上生成的摘要有显著影响。 |
| [^68] | [How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model.](http://arxiv.org/abs/2305.00586) | 本研究运用机械式可解释性技术探究了GPT-2 Small的数学能力，并确定了它的计算图中的一个小电路用于计算大于符号，该电路的多层感知器提高了结束年份大于开始年份的概率，并且该电路具有广泛的适用性。 |
| [^69] | [ParroT: Translating During Chat Using Large Language Models.](http://arxiv.org/abs/2304.02426) | ParroT提出了一种基于开源LLM和人工编写的翻译评估数据的聊天翻译框架，可以将翻译数据转化为指令执行样式，并引入额外要求来规范翻译过程。在使用相对较少的训练数据的情况下，实验结果表明 ParroT 可以大幅提高翻译质量。 |
| [^70] | [CAMEL: Communicative Agents for "Mind" Exploration of Large Scale Language Model Society.](http://arxiv.org/abs/2303.17760) | 本文介绍了一个名为角色扮演的新型交互式代理框架，用于实现语言模型之间的自主合作，并展示了其在生成对话数据方面的有效性。 |
| [^71] | [DeltaScore: Evaluating Story Generation with Differentiating Perturbations.](http://arxiv.org/abs/2303.08991) | DeltaScore利用差分扰动来评估故事生成的细粒度方面，并通过计算故事在特定方面扰动前后的可能性差异来衡量影响。该方法在多个故事领域中得到了评估，并与人类判断的相关性进行了研究。 |
| [^72] | [The Re-Label Method For Data-Centric Machine Learning.](http://arxiv.org/abs/2302.04391) | 本文提出了一种重新标签的方法来解决手动标记的数据中存在噪声的问题，并通过模型预测来辅助人类标记噪声数据。实验证明此方法适用于多类深度学习任务。 |
| [^73] | [Is ChatGPT A Good Translator? Yes With GPT-4 As The Engine.](http://arxiv.org/abs/2301.08745) | 本论文评估了ChatGPT的机器翻译能力，发现它在高资源欧洲语言上表现良好，但在低资源或远程语言上表现滞后；采用枢轴提示可以显著提高远程语言翻译的性能；在生物医学摘要或Reddit评论方面，ChatGPT的表现不如商业系统。 |
| [^74] | [Norm of Word Embedding Encodes Information Gain.](http://arxiv.org/abs/2212.09663) | 本文研究发现，跳字模型和负采样方法中静态词向量的平方范数编码了词所传达的信息增益，通过与语料库中单词的分布之间的KL散度来定义，可用于关键词提取、词性区分和上位词分类等任务。 |
| [^75] | [Rainproof: An Umbrella To Shield Text Generators From Out-Of-Distribution Data.](http://arxiv.org/abs/2212.09171) | 该论文提出了一种名为RAINPROOF的相对信息投影OOD检测框架，该框架可以在黑盒模型中利用软概率进行检测。论文还提供了一种更实际的OOD检测评估设置。研究发现，OOD检测不一定与任务特定的度量相一致。 |
| [^76] | [Improving word mover's distance by leveraging self-attention matrix.](http://arxiv.org/abs/2211.06229) | 本研究利用自注意力矩阵改进了词移距离（Word Mover's Distance，WMD）的性能，通过考虑句子结构和词嵌入的相似度，实现了在近义词识别和语义文本相似度中较好的表现。 |
| [^77] | [Quantum Circuit Compiler for a Shuttling-Based Trapped-Ion Quantum Computer.](http://arxiv.org/abs/2207.01964) | 本文介绍了一个针对穿梭式离子阱量子处理器的量子电路编译器，能够将量子电路转换和优化为特定的本地门序列，与标准编译方法相比，可以将门计数减少到5.1倍。 |

# 详细

[^1]: 服务器端对语音中心化知识查询进行重新评分的研究

    Server-side Rescoring of Spoken Entity-centric Knowledge Queries for Virtual Assistants. (arXiv:2311.01398v1 [cs.CL])

    [http://arxiv.org/abs/2311.01398](http://arxiv.org/abs/2311.01398)

    本文研究了服务器端对语音中心化知识查询进行重新评分的建模策略，并通过整合各种服务器端语言模型，显著改善了各种实体中心化查询子族群的识别错误率。此外，模型融合和使用领域特定数据训练的语言模型对于提升VA ASR系统的性能也起到了积极的作用。

    

    自动语音识别（ASR）驱动的设备内虚拟助手（VA）需要有效的知识整合来应对富实体查询识别的挑战。本文通过使用各种类别的语言模型（N-gram词语模型、子词神经网络模型），对服务器端对口语信息领域查询进行重新评分的建模策略进行了实证研究。我们研究了设备内和服务器端信号的组合，并与仅在设备内进行ASR的情况相比，通过整合各种服务器端语言模型，在各种实体中心化查询子族群中取得了23%-35%的WER改善。我们还对使用领域数据训练的语言模型和由OpenAI提供的GPT-3变体进行了比较。此外，我们还展示了从头开始训练的多个服务器端语言模型的模型融合最有效地结合了每个模型的互补优势，并将从领域特定数据中学到的知识整合到VA ASR系统中。

    On-device Virtual Assistants (VAs) powered by Automatic Speech Recognition (ASR) require effective knowledge integration for the challenging entity-rich query recognition. In this paper, we conduct an empirical study of modeling strategies for server-side rescoring of spoken information domain queries using various categories of Language Models (LMs) (N-gram word LMs, sub-word neural LMs). We investigate the combination of on-device and server-side signals, and demonstrate significant WER improvements of 23%-35% on various entity-centric query subpopulations by integrating various server-side LMs compared to performing ASR on-device only. We also perform a comparison between LMs trained on domain data and a GPT-3 variant offered by OpenAI as a baseline. Furthermore, we also show that model fusion of multiple server-side LMs trained from scratch most effectively combines complementary strengths of each model and integrates knowledge learned from domain-specific data to a VA ASR system.
    
[^2]: 语言模型是否容易受到语言幻觉的欺骗？在语法方面容易，在语义方面困难。（arXiv:2311.01386v1 [cs.CL]）

    Can Language Models Be Tricked by Language Illusions? Easier with Syntax, Harder with Semantics. (arXiv:2311.01386v1 [cs.CL])

    [http://arxiv.org/abs/2311.01386](http://arxiv.org/abs/2311.01386)

    通过研究语言模型在与"语言幻觉"相关的判断中的行为，我们发现语言模型更容易受到结构依赖性的幻觉的影响，而在语义方面则较困难。

    

    虽然语言模型（LMs）在判断语法性方面与人类有很大重叠，但是当人类在语言处理中系统性地出现错误时，我们是否期望LMs能像语言的认知模型那样模仿人类行为？通过研究与“语言幻觉”相关的LMs的更微妙判断，我们回答了这个问题——这些句子在意义上模糊、不合情理或语法错误，但却受到人类意外高接受度的判断。我们研究了三种幻觉：比较幻觉（例如“去过俄罗斯的人比我多”），深度冲击幻觉（例如“没有轻微的头部伤害可以被忽视”）和否定极性项（NPI）幻觉（例如“没有一个乡村人相信是可信赖的猎人会向熊射击”）。我们发现，LMs表示的概率更有可能与人类对于被NPI幻觉“欺骗”的判断一致，这一幻觉检验了一种结构依赖性。

    Language models (LMs) have been argued to overlap substantially with human beings in grammaticality judgment tasks. But when humans systematically make errors in language processing, should we expect LMs to behave like cognitive models of language and mimic human behavior? We answer this question by investigating LMs' more subtle judgments associated with "language illusions" -- sentences that are vague in meaning, implausible, or ungrammatical but receive unexpectedly high acceptability judgments by humans. We looked at three illusions: the comparative illusion (e.g. "More people have been to Russia than I have"), the depth-charge illusion (e.g. "No head injury is too trivial to be ignored"), and the negative polarity item (NPI) illusion (e.g. "The hunter who no villager believed to be trustworthy will ever shoot a bear"). We found that probabilities represented by LMs were more likely to align with human judgments of being "tricked" by the NPI illusion which examines a structural dep
    
[^3]: GPT-4V(ision)作为视觉-语言任务的通用评估器

    GPT-4V(ision) as a Generalist Evaluator for Vision-Language Tasks. (arXiv:2311.01361v1 [cs.CV])

    [http://arxiv.org/abs/2311.01361](http://arxiv.org/abs/2311.01361)

    GPT-4V作为一个通用评估器在视觉-语言任务中表现出有希望的一致性，展示了多模态LLMs作为评估器的巨大潜力。

    

    自动评估视觉-语言任务一直是具有挑战性的，特别是在反映人类判断方面，限制在考虑细致差异方面。尽管GPT-4V在各种多模态任务中显示出了有希望的结果，但将GPT-4V作为这些任务的通用评估器尚未被系统地探索。我们全面验证了GPT-4V在评估目的方面的能力，涉及基础的图像到文本和文本到图像合成、高级图像到图像转换和多图像到文本对齐等任务。我们使用GPT-4V采用了两种评估方法，单一答案评分和配对比较。值得注意的是，GPT-4V在各种任务和评估方法上与人类显示出有希望的一致性，展示了多模态LLMs作为评估器的巨大潜力。尽管存在限制，如限制的视觉清晰度评分和真实世界复杂推理，但它能够提供人类对齐的得分。

    Automatically evaluating vision-language tasks is challenging, especially when it comes to reflecting human judgments due to limitations in accounting for fine-grained details. Although GPT-4V has shown promising results in various multi-modal tasks, leveraging GPT-4V as a generalist evaluator for these tasks has not yet been systematically explored. We comprehensively validate GPT-4V's capabilities for evaluation purposes, addressing tasks ranging from foundational image-to-text and text-to-image synthesis to high-level image-to-image translations and multi-images to text alignment. We employ two evaluation methods, single-answer grading and pairwise comparison, using GPT-4V. Notably, GPT-4V shows promising agreement with humans across various tasks and evaluation methods, demonstrating immense potential for multi-modal LLMs as evaluators. Despite limitations like restricted visual clarity grading and real-world complex reasoning, its ability to provide human-aligned scores enriched w
    
[^4]: 更好地在生成式知识图谱完善中使用语言模型和邻居信息

    Better Together: Enhancing Generative Knowledge Graph Completion with Language Models and Neighborhood Information. (arXiv:2311.01326v1 [cs.CL])

    [http://arxiv.org/abs/2311.01326](http://arxiv.org/abs/2311.01326)

    本研究提出了将节点邻居作为额外信息加入语言模型，以改进知识图谱完善方法。在归纳和传递式Wikidata子集上，我们的方法优于传统方法和基于语言模型的KGC方法。邻居信息对模型预测具有重要影响。

    

    实际应用中的知识图谱经常存在不完整问题，限制了其潜在性能。知识图谱完善（KGC）技术旨在解决这个问题。然而，传统的KGC方法在大规模知识图谱上计算复杂度高，不实际，需要学习密集的节点嵌入和计算成对距离。生成式转换器语言模型（如T5和最近的KGT5）提供了一种有希望的解决方案，因为它们可以直接预测尾节点。在本研究中，我们提出了在语言模型基础上包含节点邻居作为额外信息来改进KGC方法。我们检验了这种补全的效果，并展示了在归纳和传递式Wikidata子集上，我们的方法优于KGT5和传统的KGC方法。我们还对邻居对模型预测的影响进行了广泛分析，并展示了其重要性。此外，我们指出了通过更高效的方法显著改善KGC的路径。

    Real-world Knowledge Graphs (KGs) often suffer from incompleteness, which limits their potential performance. Knowledge Graph Completion (KGC) techniques aim to address this issue. However, traditional KGC methods are computationally intensive and impractical for large-scale KGs, necessitating the learning of dense node embeddings and computing pairwise distances. Generative transformer-based language models (e.g., T5 and recent KGT5) offer a promising solution as they can predict the tail nodes directly. In this study, we propose to include node neighborhoods as additional information to improve KGC methods based on language models. We examine the effects of this imputation and show that, on both inductive and transductive Wikidata subsets, our method outperforms KGT5 and conventional KGC approaches. We also provide an extensive analysis of the impact of neighborhood on model prediction and show its importance. Furthermore, we point the way to significantly improve KGC through more ef
    
[^5]: 缩放、检索增强和形式对语言模型事实一致性的影响

    The Effect of Scaling, Retrieval Augmentation and Form on the Factual Consistency of Language Models. (arXiv:2311.01307v1 [cs.CL])

    [http://arxiv.org/abs/2311.01307](http://arxiv.org/abs/2311.01307)

    本研究通过缩放和检索增强两种策略，分析了语言模型事实一致性的不一致性原因，并发现检索增强策略效果更好。句法形式和其他评估任务的构造对于一致性有影响。

    

    大型语言模型（LLMs）是自然的事实知识接口，但由于它们倾向于对语义等效的问题给出不一致的答案，其实用性受限。本研究分析了不一致性的潜在原因，并评估了两种缓解策略的有效性：通过增加规模和使用检索语料库来增强语言模型。我们对LLaMA和Atlas模型的实验结果表明，这两种策略都能减少不一致性，而检索增强的效果更显著。我们进一步考虑并区分了Atlas的不同组成部分对一致性的贡献。对于所有评估的语言模型，我们发现句法形式和其他评估任务的构造对一致性有影响。综上所述，我们的研究结果对于理解影响语言模型事实一致性的因素提供了更好的认识。

    Large Language Models (LLMs) make natural interfaces to factual knowledge, but their usefulness is limited by their tendency to deliver inconsistent answers to semantically equivalent questions. For example, a model might predict both "Anne Redpath passed away in Edinburgh." and "Anne Redpath's life ended in London." In this work, we identify potential causes of inconsistency and evaluate the effectiveness of two mitigation strategies: up-scaling and augmenting the LM with a retrieval corpus. Our results on the LLaMA and Atlas models show that both strategies reduce inconsistency while retrieval augmentation is considerably more efficient. We further consider and disentangle the consistency contributions of different components of Atlas. For all LMs evaluated we find that syntactical form and other evaluation task artifacts impact consistency. Taken together, our results provide a better understanding of the factors affecting the factual consistency of language models.
    
[^6]: AWEQ：用于大型语言模型的后训练量化和激活权重均衡方法

    AWEQ: Post-Training Quantization with Activation-Weight Equalization for Large Language Models. (arXiv:2311.01305v1 [cs.LG])

    [http://arxiv.org/abs/2311.01305](http://arxiv.org/abs/2311.01305)

    AWEQ是一种后训练量化和激活权重均衡方法，能够在大型语言模型中实现超低位量化和8-bit权重和激活量化，并通过改进的均衡方法减小量化偏差误差，提高模型的鲁棒性。

    

    大型语言模型(LLMs)在各种任务中表现出色，但其计算和存储成本也相对较高。量化这些模型是缓解这个问题的有效方法。然而，现有方法很难在模型准确性和硬件效率之间取得平衡。因此，我们引入了AWEQ，一种后训练方法，不需要额外的训练开销。AWEQ在超低位量化和8-bit权重和激活(W8A8)量化方面表现出色。观察到权重量化比激活量化更容易。AWEQ通过通道均衡将激活量化的难度转移到权重上，实现了两者量化困难的平衡，从而最大化了性能。我们进一步改进了均衡方法，减小了量化偏差误差，确保模型的鲁棒性。在像LLaMA这样的流行模型上进行了大量实验。

    Large language models(LLMs) exhibit excellent performance across a variety of tasks, but they come with significant computational and storage costs. Quantizing these models is an effective way to alleviate this issue. However, existing methods struggle to strike a balance between model accuracy and hardware efficiency. This is where we introduce AWEQ, a post-training method that requires no additional training overhead. AWEQ excels in both ultra-low-bit quantization and 8-bit weight and activation (W8A8) quantization. There is an observation that weight quantization is less challenging than activation quantization. AWEQ transfers the difficulty of activation quantization to weights using channel equalization, achieving a balance between the quantization difficulties of both, and thereby maximizing performance. We have further refined the equalization method to mitigate quantization bias error, ensuring the robustness of the model. Extensive experiments on popular models such as LLaMA a
    
[^7]: FlashDecoding++: 在GPU上加速大规模语言模型推理的更快算法

    FlashDecoding++: Faster Large Language Model Inference on GPUs. (arXiv:2311.01282v1 [cs.LG])

    [http://arxiv.org/abs/2311.01282](http://arxiv.org/abs/2311.01282)

    FlashDecoding++是一种快速的LLM推理引擎，通过解决同步部分softmax更新、未充分利用扁平GEMM计算和静态数据流导致的性能损失等挑战，实现了大规模语言模型推理的加速。

    

    随着大规模语言模型在各个领域的重要性日益增加，加速语言模型推理仍然存在一些挑战未解决：(1) 同步部分softmax更新。softmax操作需要同步更新每个部分softmax结果，导致LLM中注意力计算的开销增加约20%。(2) 未充分利用扁平GEMM计算。在LLM推理中执行GEMM的矩阵形状是扁平的，导致在先前的设计中填充零后计算未充分利用，性能损失超过50%。(3) 静态数据流导致的性能损失。LLM中的内核性能取决于不同的输入数据特征、硬件配置等。单一和静态的数据流可能导致LLM推理中不同形状的GEMM的性能损失达到50.25%。我们提出了FlashDecoding++，一种快速支持主流LLM和硬件后端的LLM推理引擎。为了解决上述挑战，FlashDecoding++实现了以下目标：

    As the Large Language Model (LLM) becomes increasingly important in various domains. However, the following challenges still remain unsolved in accelerating LLM inference: (1) Synchronized partial softmax update. The softmax operation requires a synchronized update operation among each partial softmax result, leading to ~20% overheads for the attention computation in LLMs. (2) Under-utilized computation of flat GEMM. The shape of matrices performing GEMM in LLM inference is flat, leading to under-utilized computation and >50% performance loss after padding zeros in previous designs. (3) Performance loss due to static dataflow. Kernel performance in LLM depends on varied input data features, hardware configurations, etc. A single and static dataflow may lead to a 50.25% performance loss for GEMMs of different shapes in LLM inference.  We present FlashDecoding++, a fast LLM inference engine supporting mainstream LLMs and hardware back-ends. To tackle the above challenges, FlashDecoding++
    
[^8]: 发现共同点：标注和预测口语对话中的共同点

    Finding Common Ground: Annotating and Predicting Common Ground in Spoken Conversations. (arXiv:2311.01273v1 [cs.CL])

    [http://arxiv.org/abs/2311.01273](http://arxiv.org/abs/2311.01273)

    本文介绍了一种新的标注和语料库，用于捕捉口语对话中的共同基础，并通过提取命题和跟踪它们在共同基础中的状态的实验，对口语对话中的共同基础进行研究。

    

    当我们与其他人交流时，我们不仅仅是生成一系列的词语。相反，我们使用我们的认知状态（信念，欲望，意图）和我们对听众认知状态的模型来创建对话，以预期的方式影响听众的认知状态。认知状态的一个重要组成部分是共同基础，即说话者相信，并且说话者认为听众相信，以此类推的内容。虽然认知科学中对共同基础已经付出了很多关注，但在自然语言处理领域并没有太多的相关工作。在本文中，我们介绍了一种新的注释和语料库来捕捉共同基础。然后，我们描述了一些从对话中提取命题并从每个说话者的角度跟踪它们在共同基础中的状态的初步实验。

    When we communicate with other humans, we do not simply generate a sequence of words. Rather, we use our cognitive state (beliefs, desires, intentions) and our model of the audience's cognitive state to create utterances that affect the audience's cognitive state in the intended manner. An important part of cognitive state is the common ground, which is the content the speaker believes, and the speaker believes the audience believes, and so on. While much attention has been paid to common ground in cognitive science, there has not been much work in natural language processing. In this paper, we introduce a new annotation and corpus to capture common ground. We then describe some initial experiments extracting propositions from dialog and tracking their status in the common ground from the perspective of each speaker.
    
[^9]: 人类进行更好的编辑：衡量使用LLM生成的反事实增强数据在有害语言检测中的效果

    People Make Better Edits: Measuring the Efficacy of LLM-Generated Counterfactually Augmented Data for Harmful Language Detection. (arXiv:2311.01270v1 [cs.CL])

    [http://arxiv.org/abs/2311.01270](http://arxiv.org/abs/2311.01270)

    本论文通过自动生成的反事实增强数据（CADs）与手动生成的CADs进行比较，评估它们在提高模型鲁棒性方面的效果。结果显示，手动生成的CADs仍然是最有效的方法。

    

    自然语言处理模型在许多重要的社会计算任务中被使用，如检测性别歧视、种族歧视或其他仇恨内容。因此，这些模型对虚假特征的鲁棒性至关重要。过去的工作尝试解决这些虚假特征问题，其中包括反事实增强数据（CADs）的训练数据增强方法。CADs对现有的训练数据进行最小改动并翻转标签；在其上进行训练可能减少模型对虚假特征的依赖。然而，手动生成CADs可能耗时且昂贵。因此，在这项工作中，我们评估了是否可以使用生成型自然语言处理模型自动化这个任务。我们使用Polyjuice、ChatGPT和Flan-T5自动生成CADs，并与手动生成的CADs进行比较，评估它们在提高模型鲁棒性方面的实用性。通过在多个领域外测试集上测试模型的性能以及每个数据点的有效性，我们的结果表明，尽管手动CADs仍然是最有效的方法。

    NLP models are used in a variety of critical social computing tasks, such as detecting sexist, racist, or otherwise hateful content. Therefore, it is imperative that these models are robust to spurious features. Past work has attempted to tackle such spurious features using training data augmentation, including Counterfactually Augmented Data (CADs). CADs introduce minimal changes to existing training data points and flip their labels; training on them may reduce model dependency on spurious features. However, manually generating CADs can be time-consuming and expensive. Hence in this work, we assess if this task can be automated using generative NLP models. We automatically generate CADs using Polyjuice, ChatGPT, and Flan-T5, and evaluate their usefulness in improving model robustness compared to manually-generated CADs. By testing both model performance on multiple out-of-domain test sets and individual data point efficacy, our results show that while manual CADs are still the most e
    
[^10]: 基于能源的法律领域文本分类常见方法的比较分析

    An energy-based comparative analysis of common approaches to text classification in the Legal domain. (arXiv:2311.01256v1 [cs.CL])

    [http://arxiv.org/abs/2311.01256](http://arxiv.org/abs/2311.01256)

    本研究通过在法律领域的文本分类任务上进行比较分析，综合考虑性能和能源消耗等指标，探讨了大型语言模型与传统方法的优劣，并强调了在性能相近的情况下应重视生产成本、能源消耗和碳足迹等方面的考量。

    

    大部分机器学习研究评估最佳解决方案的性能。然而，在追求最佳性能的竞争中，经常忽视许多重要因素，而事实上，这些因素应该被仔细考虑。实际上，有时不同方法之间的性能差距可以忽略不计，而生产成本、能源消耗和碳足迹等因素必须考虑在内。大型语言模型（LLMs）被广泛应用于学术界和工业界的NLP问题。在这项工作中，我们在LexGLUE基准上对LLM和传统方法（例如SVM）进行了详细的定量比较，同时考虑性能（标准指标）和其他指标，如时间、耗能和成本，总之就是碳足迹。在我们的分析中，我们分别考虑了原型设计阶段（通过训练-验证-测试迭代进行模型选择）和生产阶段。

    Most Machine Learning research evaluates the best solutions in terms of performance. However, in the race for the best performing model, many important aspects are often overlooked when, on the contrary, they should be carefully considered. In fact, sometimes the gaps in performance between different approaches are neglectable, whereas factors such as production costs, energy consumption, and carbon footprint must take into consideration. Large Language Models (LLMs) are extensively adopted to address NLP problems in academia and industry. In this work, we present a detailed quantitative comparison of LLM and traditional approaches (e.g. SVM) on the LexGLUE benchmark, which takes into account both performance (standard indices) and alternative metrics such as timing, power consumption and cost, in a word: the carbon-footprint. In our analysis, we considered the prototyping phase (model selection by training-validation-test iterations) and in-production phases separately, since they fol
    
[^11]: 持续学习在语言转换中的研究

    A Study of Continual Learning Under Language Shift. (arXiv:2311.01200v1 [cs.CL])

    [http://arxiv.org/abs/2311.01200](http://arxiv.org/abs/2311.01200)

    本文研究了持续学习在语言转换中的应用，发现在更新语言模型时，前向转移效果较好且与语言顺序无关，但后向转移效果可能取决于新语言的顺序和特征。

    

    最近语言模型预训练的数据和模型规模的增加导致了巨大的训练成本。在随时间推移而出现新数据的情况下，更新模型而不是完全重新训练可以带来显著的收益。在本文中，我们研究了在新语言出现时更新语言模型时的好处和弊端，即在语言转换中持续学习的情况。从单语英语语言模型出发，我们逐步添加了来自挪威语和冰岛语的数据，以研究前向和后向转移效果如何取决于预训练顺序和语言特征，对于不同的模型大小和学习率调度器。我们的结果表明，尽管前向转移主要是正向的，不受语言顺序的影响，但后向转移则可能是正向的或负向的，具体取决于新语言的顺序和特征。为了解释这些模式，我们探索了几种语言相似度度量方法。

    The recent increase in data and model scale for language model pre-training has led to huge training costs. In scenarios where new data become available over time, updating a model instead of fully retraining it would therefore provide significant gains. In this paper, we study the benefits and downsides of updating a language model when new data comes from new languages - the case of continual learning under language shift. Starting from a monolingual English language model, we incrementally add data from Norwegian and Icelandic to investigate how forward and backward transfer effects depend on the pre-training order and characteristics of languages, for different model sizes and learning rate schedulers. Our results show that, while forward transfer is largely positive and independent of language order, backward transfer can be either positive or negative depending on the order and characteristics of new languages. To explain these patterns we explore several language similarity metr
    
[^12]: CRUSH4SQL：使用模式幻觉进行Text2SQL的集体检索

    CRUSH4SQL: Collective Retrieval Using Schema Hallucination For Text2SQL. (arXiv:2311.01173v1 [cs.CL])

    [http://arxiv.org/abs/2311.01173](http://arxiv.org/abs/2311.01173)

    CRUSH4SQL提出了一个两阶段的检索过程，使用模式幻觉进行Text2SQL的集体检索，通过幻觉的最小数据库模式检索实际模式的子集，解决了大型数据库中的模式子集检索问题。

    

    现有的Text-to-SQL生成器需要将整个模式与用户文本进行编码。对于具有成千上万列的大型数据库来说，这是昂贵或不可行的。标准的稠密检索技术对于大型结构化数据库的模式子集检索是不足够的，因为正确的检索语义要求我们对模式元素组进行排序，而不是单个元素。为此，我们提出了一个两阶段的检索过程来实现有效的覆盖。首先，我们指导LLM幻觉出一个被认为足够回答查询的最小数据库模式。我们使用幻觉的模式通过组合多个稠密检索的结果来检索实际模式的子集。显然，幻觉一直被认为是一个麻烦，但事实证明它实际上是一个有用的桥梁机制。由于目前还没有针对大型数据库的模式子集的现有基准，我们引入了三个基准。

    Existing Text-to-SQL generators require the entire schema to be encoded with the user text. This is expensive or impractical for large databases with tens of thousands of columns. Standard dense retrieval techniques are inadequate for schema subsetting of a large structured database, where the correct semantics of retrieval demands that we rank sets of schema elements rather than individual elements. In response, we propose a two-stage process for effective coverage during retrieval. First, we instruct an LLM to hallucinate a minimal DB schema deemed adequate to answer the query. We use the hallucinated schema to retrieve a subset of the actual schema, by composing the results from multiple dense retrievals. Remarkably, hallucination $\unicode{x2013}$ generally considered a nuisance $\unicode{x2013}$ turns out to be actually useful as a bridging mechanism. Since no existing benchmarks exist for schema subsetting on large databases, we introduce three benchmarks. Two semi-synthetic data
    
[^13]: 生成输入：迈向下一代输入方法范式

    Generative Input: Towards Next-Generation Input Methods Paradigm. (arXiv:2311.01166v1 [cs.CL])

    [http://arxiv.org/abs/2311.01166](http://arxiv.org/abs/2311.01166)

    本研究提出了一种新的生成输入范式GeneInput，结合提示和用户反馈进行个性化的输入处理，在中文输入法引擎的构建中实现了最先进的性能。

    

    自从ChatGPT发布以来，生成模型在各种自然语言处理任务中取得了巨大成功，并成为事实上的方法。然而，在输入法领域中，其应用仍然不够深入。许多神经网络方法已被应用于中文输入法引擎的构建。以往的研究常常假设输入的拼音是正确的，并关注拼音到字符的转换任务，这在满足用户需求方面显然不足够。此外，以前的研究无法利用用户反馈来优化模型并提供个性化的结果。在本研究中，我们提出了一种名为GeneInput的全新生成输入范式。它使用提示处理所有输入场景和其他智能辅助输入功能，通过用户反馈优化模型以提供个性化的结果。实验结果表明，我们首次在全模式按键序列到字符的任务上达到了最先进的性能。

    Since the release of ChatGPT, generative models have achieved tremendous success and become the de facto approach for various NLP tasks. However, its application in the field of input methods remains under-explored. Many neural network approaches have been applied to the construction of Chinese input method engines(IMEs).Previous research often assumed that the input pinyin was correct and focused on Pinyin-to-character(P2C) task, which significantly falls short of meeting users' demands. Moreover, previous research could not leverage user feedback to optimize the model and provide personalized results. In this study, we propose a novel Generative Input paradigm named GeneInput. It uses prompts to handle all input scenarios and other intelligent auxiliary input functions, optimizing the model with user feedback to deliver personalized results. The results demonstrate that we have achieved state-of-the-art performance for the first time in the Full-mode Key-sequence to Characters(FK2C) 
    
[^14]: 弱监督下基于执行的虚假程序过滤的语义解析问题

    Weakly Supervised Semantic Parsing with Execution-based Spurious Program Filtering. (arXiv:2311.01161v1 [cs.CL])

    [http://arxiv.org/abs/2311.01161](http://arxiv.org/abs/2311.01161)

    本研究提出了一种基于程序执行结果的领域无关筛选机制，用于解决弱监督下语义解析中的虚假程序问题

    

    在弱监督下训练语义解析器时，虚假程序是一个长期存在的挑战。为了消除具有错误语义但正确指示的程序，现有的方法着重于利用基于领域特定知识的例子相似性。在本文中，我们提出了一种基于程序执行结果的领域无关筛选机制。具体而言，对于通过搜索过程获得的每个程序，我们首先构建一个表示，它以各种输入下的执行结果捕捉程序的语义。然后，我们对这些表示进行多数投票，以识别并过滤掉与其他程序具有明显不同语义的程序。特别是，我们的方法与程序搜索过程正交，因此可以轻松增强任何现有的弱监督语义解析框架。在自然语言视觉推理和WikiTableQuestions上进行了实证评估

    The problem of spurious programs is a longstanding challenge when training a semantic parser from weak supervision. To eliminate such programs that have wrong semantics but correct denotation, existing methods focus on exploiting similarities between examples based on domain-specific knowledge. In this paper, we propose a domain-agnostic filtering mechanism based on program execution results. Specifically, for each program obtained through the search process, we first construct a representation that captures the program's semantics as execution results under various inputs. Then, we run a majority vote on these representations to identify and filter out programs with significantly different semantics from the other programs. In particular, our method is orthogonal to the program search process so that it can easily augment any of the existing weakly supervised semantic parsing frameworks. Empirical evaluations on the Natural Language Visual Reasoning and WikiTableQuestions demonstrate 
    
[^15]: ACES: WMT 2023中的翻译准确性挑战集

    ACES: Translation Accuracy Challenge Sets at WMT 2023. (arXiv:2311.01153v1 [cs.CL])

    [http://arxiv.org/abs/2311.01153](http://arxiv.org/abs/2311.01153)

    本论文介绍了使用ACES挑战集对WMT 2023中的segment-level评估指标进行了基准测试，发现没有明确的赢家，并且2023年版和2022年版指标之间的性能变化很大。建议指标开发者应该构建不同设计家族的指标集合，并开发更加关注整体性能的指标。

    

    我们使用ACES挑战集（Amrhein等人，2022）对提交到WMT 2023的segment-level评估指标进行了基准测试。该挑战集包含36K个示例，代表了来自68种现象的挑战，并涵盖了146种语言对。这些现象的范围从单词/字符级的简单扰动到基于话语和现实世界知识的更复杂的错误。对于每个指标，我们提供了在一系列错误类别上的详细性能概况，以及一个用于快速比较的整体ACES-Score。我们还测量了提交给WMT 2023和2022的指标的增量性能。我们发现：1）在提交给WMT 2023的指标中没有明显的赢家，2）2023年版和2022年版指标之间的性能变化很大。我们的建议与WMT 2022的建议类似。指标开发者应该专注于：从不同设计家族构建指标集合，开发更加关注+的指标

    We benchmark the performance of segmentlevel metrics submitted to WMT 2023 using the ACES Challenge Set (Amrhein et al., 2022). The challenge set consists of 36K examples representing challenges from 68 phenomena and covering 146 language pairs. The phenomena range from simple perturbations at the word/character level to more complex errors based on discourse and real-world knowledge. For each metric, we provide a detailed profile of performance over a range of error categories as well as an overall ACES-Score for quick comparison. We also measure the incremental performance of the metrics submitted to both WMT 2023 and 2022. We find that 1) there is no clear winner among the metrics submitted to WMT 2023, and 2) performance change between the 2023 and 2022 versions of the metrics is highly variable. Our recommendations are similar to those from WMT 2022. Metric developers should focus on: building ensembles of metrics from different design families, developing metrics that pay more at
    
[^16]: 通过语义一致性预测大型语言模型的问答性能

    Predicting Question-Answering Performance of Large Language Models through Semantic Consistency. (arXiv:2311.01152v1 [cs.CL])

    [http://arxiv.org/abs/2311.01152](http://arxiv.org/abs/2311.01152)

    本论文提出了一种框架，通过结合语义一致性度量和其他相关度量，预测大型语言模型(LLMs)对问题的准确回答能力。通过手动创建高质量的事实性问题基准数据集，并应用该框架在五个当代LLM上进行评估，实验结果表明框架的效果优于基准算法。

    

    语言模型的语义一致性广义上定义为模型在给定语义相等的输入时产生语义相等的输出的能力。我们通过手动创建一个具有高质量改写的事实性问题基准数据集，并将该数据集发布给社区，解决了评估当代大型语言模型（LLMs）问答语义一致性的任务。我们还将语义一致性度量与先前工作中建议与LLM问答准确性相关的其他度量结合起来，构建和评估了一个用于事实性问答无参考性能预测的框架，即预测语言模型正确回答问题的可能性。在五个当代LLM上评估框架，我们展示了令人鼓舞的、明显优于基准的结果。

    Semantic consistency of a language model is broadly defined as the model's ability to produce semantically-equivalent outputs, given semantically-equivalent inputs. We address the task of assessing question-answering (QA) semantic consistency of contemporary large language models (LLMs) by manually creating a benchmark dataset with high-quality paraphrases for factual questions, and release the dataset to the community.  We further combine the semantic consistency metric with additional measurements suggested in prior work as correlating with LLM QA accuracy, for building and evaluating a framework for factual QA reference-less performance prediction -- predicting the likelihood of a language model to accurately answer a question. Evaluating the framework on five contemporary LLMs, we demonstrate encouraging, significantly outperforming baselines, results.
    
[^17]: 重访知识注入框架

    Revisiting the Knowledge Injection Frameworks. (arXiv:2311.01150v1 [cs.CL])

    [http://arxiv.org/abs/2311.01150](http://arxiv.org/abs/2311.01150)

    这项研究重新审视了知识注入框架，发现将未对齐的随机知识注入到大型语言模型中可以取得与对齐知识相当甚至更好的结果。研究还提供了一种简单的修正技术来解决这个问题。

    

    近年来，大型语言模型（LLMs），如GPT，已在全球范围内产生了巨大的影响。然而，如何利用外部知识使这些LLMs更适应垂直领域特定任务的问题尚未完全解决。实际上，在这方面已经出现了一些工作，其中大部分依赖于构建对齐启发式规则，通过将相应的知识元组注入到相关的文本样本中。然而，尽管有希望，但我们普遍发现这项工作中存在一个关键问题。简而言之，我们发现将未对齐（即随机）的知识元组注入到LLMs中，可以取得与注入对齐知识相当甚至更好的结果。因此，我们对这一令人沮丧的发现进行了彻底的调查，并进一步提供了一系列可能的解释。基于这一切，我们提供了一种简单的修正技术。简要地说，这种技术的核心根植于...

    In recent years, large language models (LLMs), such as GPTs, have attained great impact worldwide. However, how to adapt these LLMs to better suit the vertical domain-specific tasks by utilizing external knowledge remains not completely solved. Indeed, there have emerged a few works on this line where most of them rely on an alignment heuristic that is built to inject the corresponding knowledge tuple into the associated text sample.  However, despite the promise, we identify a pivotal problem in this work ubiquitously. Simply put, we find that injecting unaligned (i.e., random) knowledge tuple into the LLMs achieves comparable (and sometimes better) results than the aligned knowledge being injected. We therefore take a thorough investigation of this frustrating finding on a variety of related prior work and further provide a chain of potential interpretations for the phenomenon. Based on all that, we offer a simple remediated technique. Briefly, the core of this technique is rooted in
    
[^18]: Chinesewebtext: 用有效的评估模型提取大规模高质量的中文网络文本

    Chinesewebtext: Large-scale high-quality Chinese web text extracted with effective evaluation model. (arXiv:2311.01149v1 [cs.CL])

    [http://arxiv.org/abs/2311.01149](http://arxiv.org/abs/2311.01149)

    本文提出了一个完整的工具链EvalWeb，用于从网络数据中提取干净的中文文本。通过手工制定的规则筛除噪音数据，并使用评估模型为每个文本分配质量分数。

    

    在大型语言模型（LLM）的发展过程中，预训练数据的规模和质量对于塑造LLM的能力起着至关重要的作用。为了加快LLM的研究进展，已经发布了一些大规模数据集，例如C4 [1]、Pile [2]、RefinedWeb [3]和WanJuan [4]等。然而，大多数已发布的语料库主要关注英文，仍然缺乏完整的工具链来从网络数据中提取出干净的文本。此外，缺乏对语料库的细粒度信息，例如每个文本的质量。为了解决这些挑战，本文提出了一个新的完整的工具链EvalWeb，用于从嘈杂的网络数据中提取中文干净的文本。首先，类似之前的工作，使用手工制定的规则来丢弃原始爬取的网络内容中的明确嘈杂的文本。然后，利用一个精心设计的评估模型来评估剩余相对干净的数据，并为每个文本分配一个特定的质量分数。最后，我们进行了大规模的实验，验证了EvalWeb工具链的有效性。

    During the development of large language models (LLMs), the scale and quality of the pre-training data play a crucial role in shaping LLMs' capabilities. To accelerate the research of LLMs, several large-scale datasets, such as C4 [1], Pile [2], RefinedWeb [3] and WanJuan [4], have been released to the public. However, most of the released corpus focus mainly on English, and there is still lack of complete tool-chain for extracting clean texts from web data. Furthermore, fine-grained information of the corpus, e.g. the quality of each text, is missing. To address these challenges, we propose in this paper a new complete tool-chain EvalWeb to extract Chinese clean texts from noisy web data. First, similar to previous work, manually crafted rules are employed to discard explicit noisy texts from the raw crawled web contents. Second, a well-designed evaluation model is leveraged to assess the remaining relatively clean data, and each text is assigned a specific quality score. Finally, we 
    
[^19]: 通过外部引导实现对预训练语言模型的噪声鲁棒微调

    Noise-Robust Fine-Tuning of Pretrained Language Models via External Guidance. (arXiv:2311.01108v1 [cs.CL])

    [http://arxiv.org/abs/2311.01108](http://arxiv.org/abs/2311.01108)

    通过利用大语言模型的指导，我们提出一种创新方法，使用噪声标签对预训练语言模型进行鲁棒微调。实验证明，我们的方法相比最先进的基线方法在合成和真实噪声数据集上具有优越的性能。

    

    在自然语言处理领域，采用预训练后微调的两阶段范式，预训练语言模型（PLMs）已经取得了重大进展。然而，在现实场景中，由于复杂的注释过程，数据标签通常存在噪声，因此有必要开发针对这样噪声标签的微调策略。为此，我们引入了一种创新的方法，使用噪声标签对PLMs进行微调，该方法将像ChatGPT这样的大语言模型（LLMs）的指导纳入其中。这种指导有助于准确区分干净样本和噪声样本，并提供了除噪声标签外的补充信息，从而在微调PLMs的学习过程中提供了额外的帮助。对合成和真实噪声数据集进行的广泛实验进一步证明了我们的框架相对于最先进的基线方法的优势。

    Adopting a two-stage paradigm of pretraining followed by fine-tuning, Pretrained Language Models (PLMs) have achieved substantial advancements in the field of natural language processing. However, in real-world scenarios, data labels are often noisy due to the complex annotation process, making it essential to develop strategies for fine-tuning PLMs with such noisy labels. To this end, we introduce an innovative approach for fine-tuning PLMs using noisy labels, which incorporates the guidance of Large Language Models (LLMs) like ChatGPT. This guidance assists in accurately distinguishing between clean and noisy samples and provides supplementary information beyond the noisy labels, thereby boosting the learning process during fine-tuning PLMs. Extensive experiments on synthetic and real-world noisy datasets further demonstrate the superior advantages of our framework over the state-of-the-art baselines.
    
[^20]: DistilWhisper：通过语言特定专家高效压缩多任务语音模型

    DistilWhisper: Efficient Distillation of Multi-task Speech Models via Language-Specific Experts. (arXiv:2311.01070v1 [cs.CL])

    [http://arxiv.org/abs/2311.01070](http://arxiv.org/abs/2311.01070)

    本文提出了DistilWhisper方法，通过使用语言特定专家进行轻量级模块化ASR微调和知识蒸馏，成功弥合了多任务语音模型在少数语言上的性能差距，同时保留了多任务和多语言能力的优势。

    

    Whisper是一个多任务和多语言的语音模型，涵盖99种语言。它在其涵盖的部分语言中获得了令人称赞的自动语音识别（ASR）结果，但在一些数量可观的少数语言中，该模型仍然表现不佳，尤其在较小的模型版本中表现更为严重。在这项工作中，我们提出了DistilWhisper，一种能够在ASR方面弥合这些语言的性能差距，同时保留多任务和多语言能力优势的方法。我们的方法包括两个关键策略：使用语言特定专家对whisper-small进行轻量级模块化ASR微调，并从whisper-large-v2进行知识蒸馏。这种双重方法使我们能够在保持多任务和多语言预训练的鲁棒性的同时有效提升ASR性能。结果表明，我们的方法比标准微调或LoRA适配器更有效，在目标语言中提升了性能。

    Whisper is a multitask and multilingual speech model covering 99 languages. It yields commendable automatic speech recognition (ASR) results in a subset of its covered languages, but the model still under-performs on a non-negligible number of under-represented languages, a problem exacerbated in smaller model versions. In this work, we propose DistilWhisper, an approach able to bridge the performance gap in ASR for these languages while retaining the advantages of multitask and multilingual capabilities. Our approach involves two key strategies: lightweight modular ASR fine-tuning of whisper-small using language-specific experts, and knowledge distillation from whisper-large-v2. This dual approach allows us to effectively boost ASR performance while keeping the robustness inherited from the multitask and multilingual pre-training. Results demonstrate that our approach is more effective than standard fine-tuning or LoRA adapters, boosting performance in the targeted languages for both 
    
[^21]: 用于有效调优语言模型的多维数据精化策略

    Multi-dimensional data refining strategy for effective fine-tuning LLMs. (arXiv:2311.01049v1 [cs.CL])

    [http://arxiv.org/abs/2311.01049](http://arxiv.org/abs/2311.01049)

    本文介绍了一种多维数据精化策略，包括利用现有数据集和生成型AI工具开发数据爬取脚本，用于调优越南语言模型。研究结果表明，使用该策略得到的模型在从提示生成越南新闻文章时表现出良好性能。

    

    数据是调优大型语言模型的基础，但获取合适的数据仍具有挑战性。这些挑战包括数据稀缺、语言多样性和特定领域内容。本文介绍了在爬取和精化针对调优越南语言模型的数据时所学到的经验。制作这样一个数据集需要细致的计划，考虑到语言的复杂性以及在包容性和准确性之间保持平衡。我们的论文提出了一个多维策略，包括利用英语中的现有数据集和借助生成型AI工具开发定制的数据爬取脚本。使用由产生的数据集生成的越南语言模型的调优模型，在从提示生成越南新闻文章时表现出良好性能。该研究为将来调优越南语等语言的模型提供了实用的解决方案和指导。

    Data is a cornerstone for fine-tuning large language models, yet acquiring suitable data remains challenging. Challenges encompassed data scarcity, linguistic diversity, and domain-specific content. This paper presents lessons learned while crawling and refining data tailored for fine-tuning Vietnamese language models. Crafting such a dataset, while accounting for linguistic intricacies and striking a balance between inclusivity and accuracy, demands meticulous planning. Our paper presents a multidimensional strategy including leveraging existing datasets in the English language and developing customized data-crawling scripts with the assistance of generative AI tools. A fine-tuned LLM model for the Vietnamese language, which was produced using resultant datasets, demonstrated good performance while generating Vietnamese news articles from prompts. The study offers practical solutions and guidance for future fine-tuning models in languages like Vietnamese.
    
[^22]: 学会拒绝：通过知识范围限制和拒绝机制使大型语言模型更可控和可靠

    Learn to Refuse: Making Large Language Models More Controllable and Reliable through Knowledge Scope Limitation and Refusal Mechanism. (arXiv:2311.01041v1 [cs.CL])

    [http://arxiv.org/abs/2311.01041](http://arxiv.org/abs/2311.01041)

    本文提出了一种学会拒绝（L2R）的简单而有效的解决方案，通过引入拒绝机制，使大型语言模型（LLMs）能够识别和拒绝难以回答的问题，从而提高模型的可控性和可靠性。

    

    大型语言模型（LLMs）展示了令人印象深刻的语言理解和生成能力，使它们能够回答各个领域的广泛问题。然而，这些模型并不完美，经常产生含有错误或错误信息的回答。这些不准确性，通常称为幻觉，使得LLMs在许多场景中不可靠甚至不可用。本文的重点是在LLMs中缓解幻觉问题，特别是在问答环境中。我们探索了一种拒绝机制，指导LLMs拒绝回答具有挑战性的问题以避免错误。我们提出了一个简单而有效的解决方案Learn to Refuse (L2R)，它将拒绝机制纳入到LLMs中，使其能够识别和拒绝那些它们难以回答的问题。为了实现这一点，我们利用结构化知识库来表示所有LLMs所需要的知识。

    Large language models (LLMs) have demonstrated impressive language understanding and generation capabilities, enabling them to answer a wide range of questions across various domains. However, these models are not flawless and often produce responses that contain errors or misinformation. These inaccuracies, commonly referred to as hallucinations, render LLMs unreliable and even unusable in many scenarios. In this paper, our focus is on mitigating the issue of hallucination in LLMs, particularly in the context of question-answering. Instead of attempting to answer all questions, we explore a refusal mechanism that instructs LLMs to refuse to answer challenging questions in order to avoid errors. We then propose a simple yet effective solution called Learn to Refuse (L2R), which incorporates the refusal mechanism to enable LLMs to recognize and refuse to answer questions that they find difficult to address. To achieve this, we utilize a structured knowledge base to represent all the LLM
    
[^23]: ATHENA: 数学推理与思维扩展

    ATHENA: Mathematical Reasoning with Thought Expansion. (arXiv:2311.01036v1 [cs.CL])

    [http://arxiv.org/abs/2311.01036](http://arxiv.org/abs/2311.01036)

    ATHENA是一种基于注意力机制的思维扩展网络架构，通过模拟人类的思维扩展机制来解决数学推理中的挑战，它能够产生合理的思考路径以解决实际世界的数学问题。

    

    解决数学问题取决于如何表达问题，以及模型如何看待人类语言表达的角度。实际世界的情境更依赖这种方法，因为同样的数学运算有多种实践方式。以往的研究通过限制预测策略来限制可用的思维过程，而忽略了这些策略在获取数学知识方面的重要性。我们引入了基于注意力的思维扩展网络架构 (ATHENA) 来应对现实世界中的挑战，模仿神经网络传播的形式来模拟人类的思维扩展机制。思维扩展通过递归生成候选数学表达式，从上一步驱动并选择通向目标的有效路径，产生合理的思维。我们的实验表明，ATHENA 在各种问题中都取得了全新的最先进水平，即使在信息有限的情况下，也能给出令人信服的答案。

    Solving math word problems depends on how to articulate the problems, the lens through which models view human linguistic expressions. Real-world settings count on such a method even more due to the diverse practices of the same mathematical operations. Earlier works constrain available thinking processes by limited prediction strategies without considering their significance in acquiring mathematical knowledge. We introduce Attention-based THought Expansion Network Architecture (ATHENA) to tackle the challenges of real-world practices by mimicking human thought expansion mechanisms in the form of neural network propagation. A thought expansion recurrently generates the candidates carrying the thoughts of possible math expressions driven from the previous step and yields reasonable thoughts by selecting the valid pathways to the goal. Our experiments show that ATHENA achieves a new state-of-the-art stage toward the ideal model that is compelling in variant questions even when the infor
    
[^24]: 联合学习局部和全局特征用于基于方面的情感分类

    Joint Learning of Local and Global Features for Aspect-based Sentiment Classification. (arXiv:2311.01030v1 [cs.CL])

    [http://arxiv.org/abs/2311.01030](http://arxiv.org/abs/2311.01030)

    该论文提出了一种联合学习局部和全局特征的方法，以应对基于方面的情感分类中的问题。通过设计一个包含高斯掩码层和协方差自注意层的局部编码器，在模型中有效地整合了局部上下文和全局特征，并提供了更好的区分能力。

    

    基于方面的情感分类旨在判断句子中给定方面术语所传达的情感极性。情感极性不仅由局部上下文决定，还与远离给定方面术语的词汇相关。最近的基于注意力模型在某些情况下无法足够地区分应该更关注哪些词语。与此同时，基于图的模型正在进入基于方向的情感分类以编码句法依赖树信息。但是这些模型并没有充分利用句法依赖树，因为它们忽视了将依赖关系标签信息有效地整合到表示学习中。在本文中，我们通过有效地建模局部和全局特征来解决这些问题。首先，我们设计了一个包含高斯掩码层和协方差自注意层的局部编码器。高斯掩码层倾向于自适应地调整周围方面术语的感受野，以使其不重要化。

    Aspect-based sentiment classification (ASC) aims to judge the sentiment polarity conveyed by the given aspect term in a sentence. The sentiment polarity is not only determined by the local context but also related to the words far away from the given aspect term. Most recent efforts related to the attention-based models can not sufficiently distinguish which words they should pay more attention to in some cases. Meanwhile, graph-based models are coming into ASC to encode syntactic dependency tree information. But these models do not fully leverage syntactic dependency trees as they neglect to incorporate dependency relation tag information into representation learning effectively. In this paper, we address these problems by effectively modeling the local and global features. Firstly, we design a local encoder containing: a Gaussian mask layer and a covariance self-attention layer. The Gaussian mask layer tends to adjust the receptive field around aspect terms adaptively to deemphasize 
    
[^25]: COPAL-ID: 印度尼西亚语言推理与本土文化和细微差别

    COPAL-ID: Indonesian Language Reasoning with Local Culture and Nuances. (arXiv:2311.01012v1 [cs.CL])

    [http://arxiv.org/abs/2311.01012](http://arxiv.org/abs/2311.01012)

    COPAL-ID是一个印度尼西亚语言常识推理数据集，与以前的数据集相比，它融入了印尼本土和文化细微差别，提供了更自然的日常因果推理描绘。该数据集对于现有的多语言语言模型来说是一个更大的挑战，但对人类来说很容易。在测试中，最新的开源多语言模型在COPAL-ID上的准确率较低，仅为65.47%。

    

    我们介绍了公开可用的COPAL-ID，这是一个新颖的印度尼西亚语言常识推理数据集。与以前的印尼COPA数据集（XCOPA-ID）不同，COPAL-ID融入了印尼本土和文化细微差别，因此在印尼文化领域内提供了更自然的日常因果推理描绘。COPAL-ID由本土人从头开始专业撰写，更流利，不像XCOPA-ID的翻译存在尴尬的词语。此外，我们以标准印度尼西亚语和雅加达印度尼西亚语（一种在日常对话中常用的方言）呈现COPAL-ID。COPAL-ID对于现有的开源和闭源最先进的多语言语言模型来说，提出了更大的挑战，对于人类来说却是非常容易的。我们的调查结果表明，即使是当前最好的开源多语言模型也很难表现出色，在COPAL-ID上的准确率为65.47%，远低于没有文化背景的XCOPA-ID（79.40%）。

    We present publicly available COPAL-ID, a novel Indonesian language common sense reasoning dataset. Unlike the previous Indonesian COPA dataset (XCOPA-ID), COPAL-ID incorporates Indonesian local and cultural nuances, and therefore, provides a more natural portrayal of day-to-day causal reasoning within the Indonesian cultural sphere. Professionally written by natives from scratch, COPAL-ID is more fluent and free from awkward phrases, unlike the translated XCOPA-ID. In addition, we present COPAL-ID in both standard Indonesian and in Jakartan Indonesian--a dialect commonly used in daily conversation. COPAL-ID poses a greater challenge for existing open-sourced and closed state-of-the-art multilingual language models, yet is trivially easy for humans. Our findings suggest that even the current best open-source, multilingual model struggles to perform well, achieving 65.47% accuracy on COPAL-ID, significantly lower than on the culturally-devoid XCOPA-ID (79.40%). Despite GPT-4's impressiv
    
[^26]: 在印度尼西亚的低资源本地语言上可复制的神经机器翻译基准

    Replicable Benchmarking of Neural Machine Translation (NMT) on Low-Resource Local Languages in Indonesia. (arXiv:2311.00998v1 [cs.CL])

    [http://arxiv.org/abs/2311.00998](http://arxiv.org/abs/2311.00998)

    本研究通过在印度尼西亚的低资源本地语言上训练NMT系统的综合分析，解决了神经机器翻译面临的挑战。 尽管计算资源和文本数据有限，但我们的几个NMT系统取得了竞争性性能，与零-shot gpt-3.5-turbo的翻译质量相媲美。 这些发现显著推动了低资源语言的NMT，为研究人员提供了宝贵的指导。

    

    面对印度尼西亚低资源本地语言的神经机器翻译(NMT)面临着重大挑战，包括需要一个代表性的基准和有限的数据可用性。本研究通过全面分析为印度尼西亚的四种低资源本地语言（爪哇语、苏丹尼斯语、米南卡博和巴厘语）训练NMT系统来解决这些挑战。我们的研究涵盖了各种训练方法、范例、数据规模以及对使用大型语言模型进行合成低资源语言平行数据生成的初步研究。我们揭示了低资源语言翻译实用策略的具体趋势和见解。我们的研究表明，尽管计算资源和文本数据有限，我们的几个NMT系统在竞争性性能方面取得了成功，与零-shot gpt-3.5-turbo的翻译质量相媲美。这些发现显著推动了低资源语言的NMT，为研究人员提供了宝贵的指导。

    Neural machine translation (NMT) for low-resource local languages in Indonesia faces significant challenges, including the need for a representative benchmark and limited data availability. This work addresses these challenges by comprehensively analyzing training NMT systems for four low-resource local languages in Indonesia: Javanese, Sundanese, Minangkabau, and Balinese. Our study encompasses various training approaches, paradigms, data sizes, and a preliminary study into using large language models for synthetic low-resource languages parallel data generation. We reveal specific trends and insights into practical strategies for low-resource language translation. Our research demonstrates that despite limited computational resources and textual data, several of our NMT systems achieve competitive performances, rivaling the translation quality of zero-shot gpt-3.5-turbo. These findings significantly advance NMT for low-resource languages, offering valuable guidance for researchers in
    
[^27]: 机器人任务规划的视觉语言解释器

    Vision-Language Interpreter for Robot Task Planning. (arXiv:2311.00967v1 [cs.RO])

    [http://arxiv.org/abs/2311.00967](http://arxiv.org/abs/2311.00967)

    本文提出了一种名为Vision-Language Interpreter（ViLaIn）的新框架，该框架通过使用先进的语言模型和视觉语言模型生成机器人任务描述，并通过符号规划器的错误消息反馈进行改进。实验结果表明ViLaIn和符号规划器能够准确生成有效的机器人计划。

    

    大型语言模型（LLMs）正在加速语言引导的机器人规划器的发展。同时，符号规划器具有可解释性的优势。本文提出了一个新的任务，将这两种趋势相结合，即多模态规划问题规范。目标是生成一个问题描述（PD），这是规划器用来查找计划的机器可读文件。通过从语言指令和场景观测中生成PD，我们可以驱动符号规划器在语言引导框架下工作。我们提出了一种名为Vision-Language Interpreter（ViLaIn）的新框架，该框架使用先进的LLM和视觉语言模型生成PD。ViLaIn可以通过符号规划器的错误消息反馈来改进生成的PD。我们的目标是回答这个问题：ViLaIn和符号规划器能够准确地生成有效的机器人计划吗？为了评估ViLaIn，我们引入了一个名为问题描述生成（ProDG）数据集的新颖数据集。该框架将在评估中进行测试。

    Large language models (LLMs) are accelerating the development of language-guided robot planners. Meanwhile, symbolic planners offer the advantage of interpretability. This paper proposes a new task that bridges these two trends, namely, multimodal planning problem specification. The aim is to generate a problem description (PD), a machine-readable file used by the planners to find a plan. By generating PDs from language instruction and scene observation, we can drive symbolic planners in a language-guided framework. We propose a Vision-Language Interpreter (ViLaIn), a new framework that generates PDs using state-of-the-art LLM and vision-language models. ViLaIn can refine generated PDs via error message feedback from the symbolic planner. Our aim is to answer the question: How accurately can ViLaIn and the symbolic planner generate valid robot plans? To evaluate ViLaIn, we introduce a novel dataset called the problem description generation (ProDG) dataset. The framework is evaluated wi
    
[^28]: IndoToD: 一个用于多领域印尼语端到端任务导向对话系统的基准

    IndoToD: A Multi-Domain Indonesian Benchmark For End-to-End Task-Oriented Dialogue Systems. (arXiv:2311.00958v1 [cs.CL])

    [http://arxiv.org/abs/2311.00958](http://arxiv.org/abs/2311.00958)

    本文介绍了IndoToD，一个用于印尼语的端到端多领域任务导向对话系统的基准。通过将英语数据集转化为印尼语，我们创建了这个基准，并通过雇佣母语为印尼语的人员进行翻译和数据收集，这为评估印尼语和英语对话系统以及跨语言和双语迁移学习方法提供了有效工具。

    

    大多数任务导向对话系统只是为高资源语言如英语和汉语创建的，然而，有必要开发其他区域或本地语言的对话系统，以扩展它们理解不同语言对话背景的能力。本文介绍了IndoToD，一个印尼语端到端多领域对话系统基准。我们通过词法分析将两个英语对话数据集扩展到印尼语，包括四个不同领域，以高效地减少注释的大小。为了确保高质量的数据收集，我们雇用母语为印尼语的人手动翻译对话。除了原始的英语数据集外，这些新的印尼语数据集可以作为评估印尼语和英语对话系统以及探索跨语言和双语迁移学习方法潜在益处的有效基准。

    Task-oriented dialogue (ToD) systems have been mostly created for high-resource languages, such as English and Chinese. However, there is a need to develop ToD systems for other regional or local languages to broaden their ability to comprehend the dialogue contexts in various languages. This paper introduces IndoToD, an end-to-end multi domain ToD benchmark in Indonesian. We extend two English ToD datasets to Indonesian, comprising four different domains by delexicalization to efficiently reduce the size of annotations. To ensure a high-quality data collection, we hire native speakers to manually translate the dialogues. Along with the original English datasets, these new Indonesian datasets serve as an effective benchmark for evaluating Indonesian and English ToD systems as well as exploring the potential benefits of cross-lingual and bilingual transfer learning approaches.
    
[^29]: 通过少量专家演示融合奖励函数，用于忠实准确的基于知识的对话生成

    Blending Reward Functions via Few Expert Demonstrations for Faithful and Accurate Knowledge-Grounded Dialogue Generation. (arXiv:2311.00953v1 [cs.CL])

    [http://arxiv.org/abs/2311.00953](http://arxiv.org/abs/2311.00953)

    本研究通过融合准确度指标和忠实度指标的新奖励函数，利用强化学习算法解决了语言模型幻象和知识文本多余信息问题，提供了一种平衡的生成对话回应质量评判方法。

    

    构建可信赖的对话信息寻求系统需要能够基于相关知识文本生成忠实准确回应的对话模型。然而，这个任务面临两个主要挑战。首先，语言模型可能由于预训练语料库中存在的数据偏见而产生幻觉。其次，知识文本通常包含多余和不相关的信息，这会分散模型对相关文本范围的注意力。以前的研究使用额外的数据注释在知识文本上学习知识识别模块，以绕过不相关信息，但是收集这样的高质量范围注释可能是昂贵的。在这项工作中，我们利用强化学习算法通过引入新的奖励函数克服上述挑战。我们的奖励函数将准确度指标和忠实度指标结合起来，提供一个平衡的生成回应质量评判，这可以作为一个协同参考标准。

    The development of trustworthy conversational information-seeking systems relies on dialogue models that can generate faithful and accurate responses based on relevant knowledge texts. However, two main challenges hinder this task. Firstly, language models may generate hallucinations due to data biases present in their pretraining corpus. Secondly, knowledge texts often contain redundant and irrelevant information that distracts the model's attention from the relevant text span. Previous works use additional data annotations on the knowledge texts to learn a knowledge identification module in order to bypass irrelevant information, but collecting such high-quality span annotations can be costly. In this work, we leverage reinforcement learning algorithms to overcome the above challenges by introducing a novel reward function. Our reward function combines an accuracy metric and a faithfulness metric to provide a balanced quality judgment of generated responses, which can be used as a co
    
[^30]: E3 TTS: 简单高效的端到端基于扩散的文本到语音模型

    E3 TTS: Easy End-to-End Diffusion-based Text to Speech. (arXiv:2311.00945v1 [cs.SD])

    [http://arxiv.org/abs/2311.00945](http://arxiv.org/abs/2311.00945)

    E3 TTS是一种简单高效的端到端基于扩散的文本到语音模型，不依赖于中间表示，通过扩散过程建模波形的时间结构，能够轻松适应零样本任务。

    

    我们提出了一种简单高效的端到端基于扩散的文本到语音模型，称为E3 TTS。E3 TTS直接接受纯文本作为输入，并通过迭代细化过程生成音频波形。与许多先前的工作不同，E3 TTS不依赖于任何中间表示，如频谱特征或对齐信息。相反，E3 TTS通过扩散过程建模波形的时间结构。不依赖于额外的条件信息，E3 TTS可以支持给定音频中的灵活潜在结构。这使得E3 TTS能够轻松适应零样本任务，例如在没有额外训练的情况下进行编辑。实验证明，E3 TTS能够生成高保真度音频，接近最先进的神经TTS系统的性能。音频样本可在https://e3tts.github.io上获得。

    We propose Easy End-to-End Diffusion-based Text to Speech, a simple and efficient end-to-end text-to-speech model based on diffusion. E3 TTS directly takes plain text as input and generates an audio waveform through an iterative refinement process. Unlike many prior work, E3 TTS does not rely on any intermediate representations like spectrogram features or alignment information. Instead, E3 TTS models the temporal structure of the waveform through the diffusion process. Without relying on additional conditioning information, E3 TTS could support flexible latent structure within the given audio. This enables E3 TTS to be easily adapted for zero-shot tasks such as editing without any additional training. Experiments show that E3 TTS can generate high-fidelity audio, approaching the performance of a state-of-the-art neural TTS system. Audio samples are available at https://e3tts.github.io.
    
[^31]: 无任务知识的低秩适配器用于未知的英语方言

    Task-Agnostic Low-Rank Adapters for Unseen English Dialects. (arXiv:2311.00915v1 [cs.CL])

    [http://arxiv.org/abs/2311.00915](http://arxiv.org/abs/2311.00915)

    HyperLoRA是一种无任务知识的低秩适配器方法，利用专家语言知识通过超网络实现资源高效的适应性，从而提高了对未知英语方言的泛化能力。

    

    大语言模型（LLM）是基于偏向于标准美式英语的语料库进行训练的。因此，使用其他方言的人在与这些技术交互时会遇到更多的故障。在实践中，这些讲话者通常会适应自己的言语以便被更好理解。我们的工作认为语言技术应该被设计为适应英语方言的多样性，而不是反其道而行之。然而，之前关于方言的工作在扩展和新出现的方言上存在问题。为了弥补这一空白，我们的方法HyperLoRA利用专家语言知识通过超网络实现资源高效的适应性。通过解开方言特定和跨方言的信息，HyperLoRA以无任务的方式提高对未知方言的泛化能力。HyperLoRA不仅在参数数量上更具可扩展性，而且在性能上是最佳或最具竞争力的。

    Large Language Models (LLMs) are trained on corpora disproportionally weighted in favor of Standard American English. As a result, speakers of other dialects experience significantly more failures when interacting with these technologies. In practice, these speakers often accommodate their speech to be better understood. Our work shares the belief that language technologies should be designed to accommodate the diversity in English dialects and not the other way around. However, prior works on dialect struggle with generalizing to evolving and emerging dialects in a scalable manner. To fill this gap, our method, HyperLoRA, leverages expert linguistic knowledge to enable resource-efficient adaptation via hypernetworks. By disentangling dialect-specific and cross-dialectal information, HyperLoRA improves generalization to unseen dialects in a task-agnostic fashion. Not only is HyperLoRA more scalable in the number of parameters, but it also achieves the best or most competitive performan
    
[^32]: 基于自主影响引导的语言模型预训练数据重加权方法

    Self-Influence Guided Data Reweighting for Language Model Pre-training. (arXiv:2311.00913v1 [cs.CL])

    [http://arxiv.org/abs/2311.00913](http://arxiv.org/abs/2311.00913)

    提出了一种名为PRESENCE的方法，通过利用自主影响分数作为重要性指标，对语言模型预训练的数据样本进行重加权，促进了模型预训练的新颖性和稳定性。

    

    在大型文本语料库上进行自监督预训练的语言模型已成为开发各种自然语言处理任务模型的默认起点。然而，一旦预训练语料库被组装好，在LM预训练期间，所有数据样本都被视为具有相等的重要性。然而，由于数据的相关性和质量存在差异，对所有数据样本给予相等的重要性可能不是最优选择。我们填补了这个重要空白，并提出了一种名为PRESENCE的方法，通过利用自主影响（SI）分数作为样本重要性和预训练的指标，共同对样本进行重加权。PRESENCE促进了模型预训练的新颖性和稳定性。通过涵盖多个模型规模、数据集和任务的广泛分析，我们将PRESENCE提出为一个重要的首要步骤。

    Language Models (LMs) pre-trained with self-supervision on large text corpora have become the default starting point for developing models for various NLP tasks. Once the pre-training corpus has been assembled, all data samples in the corpus are treated with equal importance during LM pre-training. However, due to varying levels of relevance and quality of data, equal importance to all the data samples may not be the optimal choice. While data reweighting has been explored in the context of task-specific supervised learning and LM fine-tuning, model-driven reweighting for pre-training data has not been explored. We fill this important gap and propose PRESENCE, a method for jointly reweighting samples by leveraging self-influence (SI) scores as an indicator of sample importance and pre-training. PRESENCE promotes novelty and stability for model pre-training. Through extensive analysis spanning multiple model sizes, datasets, and tasks, we present PRESENCE as an important first step in t
    
[^33]: 重新加权标记：一种简单且有效的适用于命名实体识别的主动学习策略

    Re-weighting Tokens: A Simple and Effective Active Learning Strategy for Named Entity Recognition. (arXiv:2311.00906v1 [cs.CL])

    [http://arxiv.org/abs/2311.00906](http://arxiv.org/abs/2311.00906)

    本文提出了一种重新加权的主动学习策略，通过为每个标记分配动态平滑的权重，解决了命名实体识别中的数据不平衡问题，取得了显著的性能提升。

    

    主动学习是一种被广泛应用于文本和图像分类任务中，通过有限的注释资源提升机器学习模型性能的技术。然而，在命名实体识别领域，主动学习相对较少被关注。命名实体识别中的数据不平衡问题妨碍了主动学习的效果，因为序列标记模型缺乏足够的学习信号。为了解决这些挑战，本文提出了一种基于重新加权的主动学习策略，为每个标记分配了动态平滑的权重。这种适应性策略与各种标记级采集函数兼容，并有助于开发健壮的主动学习器。在多个语料库上的实验结果表明，将我们的重新加权策略与现有的采集函数结合起来能够显著提高性能，验证了其实际有效性。

    Active learning, a widely adopted technique for enhancing machine learning models in text and image classification tasks with limited annotation resources, has received relatively little attention in the domain of Named Entity Recognition (NER). The challenge of data imbalance in NER has hindered the effectiveness of active learning, as sequence labellers lack sufficient learning signals. To address these challenges, this paper presents a novel reweighting-based active learning strategy that assigns dynamic smoothed weights to individual tokens. This adaptable strategy is compatible with various token-level acquisition functions and contributes to the development of robust active learners. Experimental results on multiple corpora demonstrate the substantial performance improvement achieved by incorporating our re-weighting strategy into existing acquisition functions, validating its practical efficacy.
    
[^34]: 关于条件音频生成中的开放提示挑战

    On The Open Prompt Challenge In Conditional Audio Generation. (arXiv:2311.00897v1 [cs.SD])

    [http://arxiv.org/abs/2311.00897](http://arxiv.org/abs/2311.00897)

    本文针对条件音频生成中的开放提示挑战，通过重新编写提示并利用文本-音频对齐作为反馈信号，从而改善音频质量，取得了显著的改进。

    

    文本到音频生成（TTA）通过学习音频样本和手工注释的文本对来从文本描述中生成音频。然而，商业化音频生成面临挑战，因为用户输入的提示通常与训练TTA模型所使用的文本描述相比缺乏具体性。在这项工作中，我们将TTA模型视为一个“黑盒子”，解决了用户提示挑战的两个关键洞察：（1）用户提示通常缺乏具体性，导致用户提示与训练提示之间存在较大的对齐差距。（2）存在一种音频描述分布，TTA模型能够生成更高质量音频，我们称之为“audionese”。为此，我们使用指导调整模型重写提示，并提出利用文本-音频对齐作为反馈信号，通过边界排序学习来改善音频质量。在客观和主观的人类评估中，我们观察到了文本-音频对齐和音乐音频质量的显著改善。

    Text-to-audio generation (TTA) produces audio from a text description, learning from pairs of audio samples and hand-annotated text. However, commercializing audio generation is challenging as user-input prompts are often under-specified when compared to text descriptions used to train TTA models. In this work, we treat TTA models as a ``blackbox'' and address the user prompt challenge with two key insights: (1) User prompts are generally under-specified, leading to a large alignment gap between user prompts and training prompts. (2) There is a distribution of audio descriptions for which TTA models are better at generating higher quality audio, which we refer to as ``audionese''. To this end, we rewrite prompts with instruction-tuned models and propose utilizing text-audio alignment as feedback signals via margin ranking learning for audio improvements. On both objective and subjective human evaluations, we observed marked improvements in both text-audio alignment and music audio qual
    
[^35]: 条件音频生成的上下文提示编辑

    In-Context Prompt Editing For Conditional Audio Generation. (arXiv:2311.00895v1 [cs.SD])

    [http://arxiv.org/abs/2311.00895](http://arxiv.org/abs/2311.00895)

    本研究提出了一种基于检索的上下文提示编辑框架，通过利用训练字幕作为示例来改进条件音频生成的音频质量。

    

    分布偏移是部署机器学习模型的一个核心挑战，因为它们可能对真实世界的数据不适应。这在文本到音频生成中尤为明显，其中编码表示很容易被不可见的提示破坏，导致生成的音频质量下降。有限的文本-音频对集合仍然不足以实现条件音频生成，因为用户的提示信息不明确。我们观察到，与训练集中的提示相比，生成的音频样本中具有用户提示的音频质量一直存在一致的降低。为此，我们提出了一个基于检索的上下文提示编辑框架，利用训练字幕作为示例来重新审视用户提示。我们展示了该框架提高了在收集到的用户提示集合上的音频质量，这些提示是参考训练字幕编辑的。

    Distributional shift is a central challenge in the deployment of machine learning models as they can be ill-equipped for real-world data. This is particularly evident in text-to-audio generation where the encoded representations are easily undermined by unseen prompts, which leads to the degradation of generated audio -- the limited set of the text-audio pairs remains inadequate for conditional audio generation in the wild as user prompts are under-specified. In particular, we observe a consistent audio quality degradation in generated audio samples with user prompts, as opposed to training set prompts. To this end, we present a retrieval-based in-context prompt editing framework that leverages the training captions as demonstrative exemplars to revisit the user prompts. We show that the framework enhanced the audio quality across the set of collected user prompts, which were edited with reference to the training captions as exemplars.
    
[^36]: 预训练数据混合使得Transformer模型具备狭窄的模型选择能力

    Pretraining Data Mixtures Enable Narrow Model Selection Capabilities in Transformer Models. (arXiv:2311.00871v1 [cs.LG])

    [http://arxiv.org/abs/2311.00871](http://arxiv.org/abs/2311.00871)

    Transformer模型通过预训练数据混合实现了狭窄的模型选择能力，能够在上下文中识别和学习不同的任务，但对于任务或函数的处理相对有限。

    

    Transformer模型，特别是大型语言模型（LLM），具有在上下文中学习（ICL）的显著能力-在未经过任何明确模型训练的情况下，根据未见过的输入-输出例子执行新的任务。本研究探讨了Transformer模型如何有效地在其预训练数据混合中建立桥梁，以在上下文中识别和学习既包括预训练分布内又包括其外的新任务。在之前的研究基础上，我们在一个受控的环境中进行了实验，我们研究了基于$(x, f(x))$对序列而不是自然语言进行训练的Transformer模型。我们的实证结果表明，Transformer模型在无监督模型选择能力方面表现接近最优，在能够首先在上下文中识别不同的任务族群并在其中进行学习时（任务族群在预训练数据中有很好的表示）。然而，当面对任务或函数时，情况会稍有不同。

    Transformer models, notably large language models (LLMs), have the remarkable ability to perform in-context learning (ICL) -- to perform new tasks when prompted with unseen input-output examples without any explicit model training. In this work, we study how effectively transformers can bridge between their pretraining data mixture, comprised of multiple distinct task families, to identify and learn new tasks in-context which are both inside and outside the pretraining distribution. Building on previous work, we investigate this question in a controlled setting, where we study transformer models trained on sequences of $(x, f(x))$ pairs rather than natural language. Our empirical results show transformers demonstrate near-optimal unsupervised model selection capabilities, in their ability to first in-context identify different task families and in-context learn within them when the task families are well-represented in their pretraining data. However when presented with tasks or functi
    
[^37]: 从未被转录的语音中自动检测语言错乱

    Automatic Disfluency Detection from Untranscribed Speech. (arXiv:2311.00867v1 [eess.AS])

    [http://arxiv.org/abs/2311.00867](http://arxiv.org/abs/2311.00867)

    本研究探讨了从未被转录的语音中自动检测语言错乱的语言、声学和多模态方法，并通过评估自动语音识别系统的转录能力，以及将其作为自然语言理解的预处理步骤，来改善临床和非临床应用中的语言识别效果。

    

    语言错乱，如充满停顿或重复，是语音中典型流畅度的中断。口吃是一种以高频率出现语言错乱的言语障碍，但所有人说话时都会出现一些语言错乱，而语言错乱的频率可能因认知负荷等因素而增加。在临床上，自动检测错误的语言可能有助于为口吃者制定治疗计划。在临床外，自动检测错误的语言可能作为预处理步骤，以改善后续应用中的自然语言理解。考虑到这个广泛的应用范围，我们研究了基于语言、声学和多模态方法的逐帧自动检测和分类语言错乱。这些方法都以音频作为输入。首先，我们评估了几种自动语音识别（ASR）系统在转录错误的能力方面，使用错误率来衡量语言错乱的转录能力。然后，我们使用这些ASR转录结果作为输入，...

    Speech disfluencies, such as filled pauses or repetitions, are disruptions in the typical flow of speech. Stuttering is a speech disorder characterized by a high rate of disfluencies, but all individuals speak with some disfluencies and the rates of disfluencies may by increased by factors such as cognitive load. Clinically, automatic disfluency detection may help in treatment planning for individuals who stutter. Outside of the clinic, automatic disfluency detection may serve as a pre-processing step to improve natural language understanding in downstream applications. With this wide range of applications in mind, we investigate language, acoustic, and multimodal methods for frame-level automatic disfluency detection and categorization. Each of these methods relies on audio as an input. First, we evaluate several automatic speech recognition (ASR) systems in terms of their ability to transcribe disfluencies, measured using disfluency error rates. We then use these ASR transcripts as i
    
[^38]: 训练语言模型中上下文N-Gram的动态

    Training Dynamics of Contextual N-Grams in Language Models. (arXiv:2311.00863v1 [cs.LG])

    [http://arxiv.org/abs/2311.00863](http://arxiv.org/abs/2311.00863)

    这篇论文研究了训练过程中上下文N-Gram的动态，发现了上下文神经元存在于更广泛的上下文N-Gram电路中，这被称为二阶电路。在训练早期，这两个电路具有相互独立的功能，只有在它们都形成之后才能组合成一个二阶电路。

    

    先前的研究已经表明，语言模型中存在上下文神经元，包括一个在德语文本上激活的神经元。我们展示了这个神经元存在于更广泛的上下文N-Gram电路中：我们发现晚期的神经元能够识别和继续德语文本中常见的N-Gram，但只有在德语神经元激活时才会被激活。我们研究了训练过程中这个电路的形成，并发现这是一个我们称之为二阶电路的示例。特别地，早期的训练中，组成N-Gram电路和最终形成德语神经元的德语检测电路具有独立的功能-德语检测电路部分通过建模德语单字统计数据进行形成，而N-Gram则通过提升适当的完成来形成。只有在这两个电路已经形成之后，它们才能组合成为一个二阶电路。与先前的研究假设相反，我们发现上下文N-Gram电路逐渐形成。

    Prior work has shown the existence of contextual neurons in language models, including a neuron that activates on German text. We show that this neuron exists within a broader contextual n-gram circuit: we find late layer neurons which recognize and continue n-grams common in German text, but which only activate if the German neuron is active. We investigate the formation of this circuit throughout training and find that it is an example of what we call a second-order circuit. In particular, both the constituent n-gram circuits and the German detection circuit which culminates in the German neuron form with independent functions early in training - the German detection circuit partially through modeling German unigram statistics, and the n-grams by boosting appropriate completions. Only after both circuits have already formed do they fit together into a second-order circuit. Contrary to the hypotheses presented in prior work, we find that the contextual n-gram circuit forms gradually r
    
[^39]: 为高效且可广泛应用的超细粒度实体类型定义校准的Seq2seq模型

    Calibrated Seq2seq Models for Efficient and Generalizable Ultra-fine Entity Typing. (arXiv:2311.00835v1 [cs.CL])

    [http://arxiv.org/abs/2311.00835](http://arxiv.org/abs/2311.00835)

    这项研究介绍了CASENT，一种为超细粒度实体类型定义设计的Seq2seq模型，通过校准置信度分数来预测超细粒度类型。在实验中，该模型在UFET数据集上表现出了优于先前方法的性能。

    

    超细粒度实体类型对于信息提取有重要作用，可以预测文本中实体提及的细粒度语义类型。然而，由于输出空间中实体类型的大量存在，这项任务面临着重大挑战。目前最先进的方法，基于标准的多标签分类器或交叉编码器模型，在泛化性能差或推理效率低的问题上存在困难。本文提出了CASENT，一种设计用于超细粒度实体类型定义的seq2seq模型，其预测带有校准置信度分数的超细粒度类型。我们的模型以实体提及作为输入，并采用约束的束搜索方法自回归的生成多个类型。然后，使用一种新颖的校准方法将与预测类型关联的原始序列概率转换为置信度分数。我们在包含超过10k种类型的UFET数据集上进行了广泛的实验。我们的方法在性能方面优于先前的最先进方法。

    Ultra-fine entity typing plays a crucial role in information extraction by predicting fine-grained semantic types for entity mentions in text. However, this task poses significant challenges due to the massive number of entity types in the output space. The current state-of-the-art approaches, based on standard multi-label classifiers or cross-encoder models, suffer from poor generalization performance or inefficient inference. In this paper, we present CASENT, a seq2seq model designed for ultra-fine entity typing that predicts ultra-fine types with calibrated confidence scores. Our model takes an entity mention as input and employs constrained beam search to generate multiple types autoregressively. The raw sequence probabilities associated with the predicted types are then transformed into confidence scores using a novel calibration method. We conduct extensive experiments on the UFET dataset which contains over 10k types. Our method outperforms the previous state-of-the-art in terms
    
[^40]: 在隐喻识别数据集中的建构偏见

    Construction Artifacts in Metaphor Identification Datasets. (arXiv:2311.00790v1 [cs.CL])

    [http://arxiv.org/abs/2311.00790](http://arxiv.org/abs/2311.00790)

    该论文研究了现有隐喻识别数据集中的建构偏见问题，并证明基于语言模型的隐喻识别系统可以与使用完整上下文的系统相竞争。

    

    隐喻识别旨在理解给定表达是否在特定语境中以比喻的方式使用。然而，在这篇论文中，我们展示了现有的隐喻识别数据集如何通过完全忽略潜在的隐喻表达或其所在的语境来进行操纵。我们在各种数据集和设置中测试了这个假设，并表明基于不具有完整信息的语言模型的隐喻识别系统可以与使用完整上下文的系统相竞争。这是由于构建这些数据集的过程中引入了对正负类别的不希望的偏见。最后，我们在从自然语料库中精心抽样的不存在这种偏见的数据集上测试了同样的假设，使得这些数据集更具挑战性和可靠性。

    Metaphor identification aims at understanding whether a given expression is used figuratively in context. However, in this paper we show how existing metaphor identification datasets can be gamed by fully ignoring the potential metaphorical expression or the context in which it occurs. We test this hypothesis in a variety of datasets and settings, and show that metaphor identification systems based on language models without complete information can be competitive with those using the full context. This is due to the construction procedures to build such datasets, which introduce unwanted biases for positive and negative classes. Finally, we test the same hypothesis on datasets that are carefully sampled from natural corpora and where this bias is not present, making these datasets more challenging and reliable.
    
[^41]: 用于临床特征嵌入的语言模型训练范式

    Language Model Training Paradigms for Clinical Feature Embeddings. (arXiv:2311.00768v1 [cs.LG])

    [http://arxiv.org/abs/2311.00768](http://arxiv.org/abs/2311.00768)

    本研究使用自监督训练范式的语言模型，通过表示学习为临床时间序列推导出高质量的通用临床特征嵌入。通过无监督的降维技术可视化学习到的嵌入，并在MIMIC-III基准测试中验证了它们的有效性。

    

    在数据稀缺的研究领域，表示学习起着重要的作用。本研究旨在通过对临床时间序列进行表示学习，推导出临床特征（如心率和血压）的通用嵌入。我们使用语言模型的自监督训练范式，学习高质量的临床特征嵌入，实现比现有的时间步和患者级别表示学习更细粒度的表征。我们通过无监督的降维技术可视化学习到的嵌入，并观察到与先前的临床知识高度一致。我们还在MIMIC-III基准测试上评估模型性能，并展示了使用临床特征嵌入的有效性。我们将我们的代码发布在网上以供复制。

    In research areas with scarce data, representation learning plays a significant role. This work aims to enhance representation learning for clinical time series by deriving universal embeddings for clinical features, such as heart rate and blood pressure. We use self-supervised training paradigms for language models to learn high-quality clinical feature embeddings, achieving a finer granularity than existing time-step and patient-level representation learning. We visualize the learnt embeddings via unsupervised dimension reduction techniques and observe a high degree of consistency with prior clinical knowledge. We also evaluate the model performance on the MIMIC-III benchmark and demonstrate the effectiveness of using clinical feature embeddings. We publish our code online for replication.
    
[^42]: 论文标题：从连续手语中基于语言驱动的计算机手势识别的挑战

    Challenges for Linguistically-Driven Computer-Based Sign Recognition from Continuous Signing for American Sign Language. (arXiv:2311.00762v1 [cs.CV])

    [http://arxiv.org/abs/2311.00762](http://arxiv.org/abs/2311.00762)

    运用美国手语数据，论文概述了基于语言驱动的计算机手势识别的挑战，包括手语者间和手语者内的同步变化以及手势结构中的语言规律。

    

    近年来，计算机基于视频的孤立手势识别取得了一些进展。这一任务存在许多挑战，其中最大的挑战是手语的实际产生中存在的手语者间和手语者内的同步变化，包括某些手语的社会语言学变体的实现。然而，连续手势识别的挑战更加困难，本文基于美国手语（ASL）的大型语言注释视频数据的发现，对这些挑战进行了概述。还讨论了手势结构中的一些语言规律，可以提高手势形状和手势识别的准确性。

    There have been recent advances in computer-based recognition of isolated, citation-form signs from video. There are many challenges for such a task, not least the naturally occurring inter- and intra- signer synchronic variation in sign production, including sociolinguistic variation in the realization of certain signs. However, there are several significant factors that make recognition of signs from continuous signing an even more difficult problem. This article presents an overview of such challenges, based in part on findings from a large corpus of linguistically annotated video data for American Sign Language (ASL). Some linguistic regularities in the structure of signs that can boost handshape and sign recognition are also discussed.
    
[^43]: 大型语言模型能设计准确的标签函数吗？

    Can Large Language Models Design Accurate Label Functions?. (arXiv:2311.00739v1 [cs.CL])

    [http://arxiv.org/abs/2311.00739](http://arxiv.org/abs/2311.00739)

    本研究引入了DataSculpt，它是一个利用预训练语言模型自动生成标签函数的交互式框架。通过多种技术和方法的结合，DataSculpt在各种任务和真实数据集上展现了优点和局限性。

    

    编程式弱监督方法通过使用封装启发式数据源的标签函数（LFs）加速标记大规模数据集。然而，创建精确的LFs需要领域专业知识和大量努力。最近，预训练语言模型（PLMs）的进展在各种任务中展示了巨大的潜力。然而，PLMs自主制定准确的LFs的能力仍然是一个未被充分探索的领域。在这项研究中，我们通过引入DataSculpt来填补这一空白，这是一个利用PLMs自动生成LFs的交互式框架。在DataSculpt中，我们结合了各种提示技术、实例选择策略和LF过滤方法来探索广阔的设计领域。最终，我们对DataSculpt在12个涵盖多个任务的真实数据集上的性能进行了全面评估。这个评估揭示了DataSculpt的优点和局限性。

    Programmatic weak supervision methodologies facilitate the expedited labeling of extensive datasets through the use of label functions (LFs) that encapsulate heuristic data sources. Nonetheless, the creation of precise LFs necessitates domain expertise and substantial endeavors. Recent advances in pre-trained language models (PLMs) have exhibited substantial potential across diverse tasks. However, the capacity of PLMs to autonomously formulate accurate LFs remains an underexplored domain. In this research, we address this gap by introducing DataSculpt, an interactive framework that harnesses PLMs for the automated generation of LFs. Within DataSculpt, we incorporate an array of prompting techniques, instance selection strategies, and LF filtration methods to explore the expansive design landscape. Ultimately, we conduct a thorough assessment of DataSculpt's performance on 12 real-world datasets, encompassing a range of tasks. This evaluation unveils both the strengths and limitations 
    
[^44]: tmn在#SMM4H 2023上的论文:比较用于检测自我报告COVID-19诊断推文的文本预处理技术

    tmn at #SMM4H 2023: Comparing Text Preprocessing Techniques for Detecting Tweets Self-reporting a COVID-19 Diagnosis. (arXiv:2311.00732v1 [cs.CL])

    [http://arxiv.org/abs/2311.00732](http://arxiv.org/abs/2311.00732)

    本文研究了用于检测自我报告COVID-19诊断推文的不同文本预处理技术，通过使用四个基于transformer的模型进行实验，并通过微调语言模型集成获得了比平均值高出4.1%的84.5%的F1得分。

    

    本文描述了一个用于SMM4H 2023任务1的系统。该任务的目标是自动区分那些自我报告COVID-19诊断的推文（例如，阳性检测、临床诊断或住院）和那些没有的推文。我们使用四个基于transformer的模型研究了不同的推文预处理技术。经过微调的语言模型集成获得了84.5%的F1得分，比平均值高出4.1%。

    The paper describes a system developed for Task 1 at SMM4H 2023. The goal of the task is to automatically distinguish tweets that self-report a COVID-19 diagnosis (for example, a positive test, clinical diagnosis, or hospitalization) from those that do not. We investigate the use of different techniques for preprocessing tweets using four transformer-based models. The ensemble of fine-tuned language models obtained an F1-score of 84.5%, which is 4.1% higher than the average value.
    
[^45]: JADE：基于语言的LLM安全评估平台

    JADE: A Linguistic-based Safety Evaluation Platform for LLM. (arXiv:2311.00286v1 [cs.CL])

    [http://arxiv.org/abs/2311.00286](http://arxiv.org/abs/2311.00286)

    JADE是一种基于语言分析的LLM安全评估平台，能够破坏广泛使用的中文和英文LLM，并生成高度威胁的不安全问题。

    

    本文介绍了JADE，一种针对语言分析的模糊测试平台，通过增强种子问题的语言复杂性，同时并始终能够破坏广泛使用的三类LLM：八个开源中文LLM，六个商业中文LLM和四个商业英文LLM。JADE为这三类LLM生成了三个安全基准，其中包含高度威胁的不安全问题：这些问题可以同时触发多个LLM的有害生成，平均不安全生成比例为70%（请参见下表），同时这些问题仍然是自然、流畅且保留了核心的不安全语义。我们在以下链接中发布了对商业英文LLM和开源英文LLM生成的基准演示：https://github.com/whitzard-ai/jade-db。对于对JADE生成的更多问题感兴趣的读者，请与我们联系。

    In this paper, we present \textit{JADE}, a targeted linguistic fuzzing platform which strengthens the linguistic complexity of seed questions to simultaneously and consistently break a wide range of widely-used LLMs categorized in three groups: eight open-sourced Chinese, six commercial Chinese and four commercial English LLMs. JADE generates three safety benchmarks for the three groups of LLMs, which contain unsafe questions that are highly threatening: the questions simultaneously trigger harmful generation of multiple LLMs, with an average unsafe generation ratio of \textbf{$70\%$} (please see the table below), while are still natural questions, fluent and preserving the core unsafe semantics. We release the benchmark demos generated for commercial English LLMs and open-sourced English LLMs in the following link: https://github.com/whitzard-ai/jade-db. For readers who are interested in evaluating on more questions generated by JADE, please contact us.  \textit{JADE} is based on Noam
    
[^46]: 意义表征来自自回归模型中的轨迹

    Meaning Representations from Trajectories in Autoregressive Models. (arXiv:2310.18348v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.18348](http://arxiv.org/abs/2310.18348)

    本文提出了一种从自回归语言模型中提取意义表征的方法，通过考虑输入文本的所有可能轨迹的分布。这种方法可以模拟非对称关系，且在语义相似性任务上优于其他方法。

    

    我们提出通过考虑扩展输入文本的所有可能轨迹的分布来从自回归语言模型中提取意义表征。这种策略是无提示的，不需要微调，并适用于任何预训练的自回归模型。此外，与基于向量的表征不同，基于分布的表征还可以通过使用似然函数之间的代数运算来建模非对称关系（例如，逻辑蕴涵的方向，上位词/下位词关系）。这些想法基于语义的分布视角，并与自动机理论中的标准构造相连接，但据我们所知，它们尚未应用于现代语言模型。我们通过实验证明，从大型模型获得的表征与人类注释很好地一致，在语义相似性任务上优于其他零样本和无提示方法，并可用于解决更复杂的蕴涵和包含任务。

    We propose to extract meaning representations from autoregressive language models by considering the distribution of all possible trajectories extending an input text. This strategy is prompt-free, does not require fine-tuning, and is applicable to any pre-trained autoregressive model. Moreover, unlike vector-based representations, distribution-based representations can also model asymmetric relations (e.g., direction of logical entailment, hypernym/hyponym relations) by using algebraic operations between likelihood functions. These ideas are grounded in distributional perspectives on semantics and are connected to standard constructions in automata theory, but to our knowledge they have not been applied to modern language models. We empirically show that the representations obtained from large models align well with human annotations, outperform other zero-shot and prompt-free methods on semantic similarity tasks, and can be used to solve more complex entailment and containment tasks 
    
[^47]: 提高通过减少无关文档对开放领域问答中的零样本阅读器的干扰的方法

    Improving Zero-shot Reader by Reducing Distractions from Irrelevant Documents in Open-Domain Question Answering. (arXiv:2310.17490v1 [cs.CL])

    [http://arxiv.org/abs/2310.17490](http://arxiv.org/abs/2310.17490)

    本研究提出了一种通过减少无关文档的干扰来改善开放领域问答中的零样本阅读器的方法。采用了干扰感知的答案选择(DAS)方法，以解决LLMs受到干扰和过度自信的问题。实验结果表明，该方法成功地改善了零样本阅读器的性能，并展现出了优越的可迁移性。

    

    大型语言模型(LLMs)使得在开放领域问答(ODQA)中实现零样本方法成为可能，但是由于阅读器相对于检索器的进展有限。本研究旨在探讨一种零样本阅读器的可行性，以解决计算成本和标注数据需求等挑战。我们发现LLMs由于检索到的无关文档以及作为零样本阅读器时生成答案的过度自信而受到干扰。为了解决这些问题，我们采用了基于否定的指令和分数调整的干扰感知的答案选择(DAS)方法，以减轻这些文档的影响。实验结果表明，我们的方法成功地处理了不同场景下的干扰，提高了零样本阅读器的性能。此外，与面对未见过数据而困难重重的监督式阅读器不同，零样本阅读器展现出了优越的可迁移性，无需任何训练。

    Large language models (LLMs) enable zero-shot approaches in open-domain question answering (ODQA), yet with limited advancements as the reader is compared to the retriever. This study aims at the feasibility of a zero-shot reader that addresses the challenges of computational cost and the need for labeled data. We find that LLMs are distracted due to irrelevant documents in the retrieved set and the overconfidence of the generated answers when they are exploited as zero-shot readers. To tackle these problems, we mitigate the impact of such documents via Distraction-aware Answer Selection (DAS) with a negation-based instruction and score adjustment for proper answer selection. Experimental results show that our approach successfully handles distraction across diverse scenarios, enhancing the performance of zero-shot readers. Furthermore, unlike supervised readers struggling with unseen data, zero-shot readers demonstrate outstanding transferability without any training.
    
[^48]: 空气解码：解码时间可控文本生成的属性分布重建

    Air-Decoding: Attribute Distribution Reconstruction for Decoding-Time Controllable Text Generation. (arXiv:2310.14892v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.14892](http://arxiv.org/abs/2310.14892)

    本文提出了一种名为空气解码的新颖轻量级解码框架，通过重建属性分布来平衡权重，生成更流畅的文本，以解决可控文本生成中的属性坍缩问题。

    

    可控的文本生成（CTG）旨在生成具有所需属性的文本，而基于解码时间的方法在这个任务上已经显示出了有希望的性能。然而，在本文中，我们首次发现了属性坍缩现象。当控制强度超过临界值时，它会导致生成文本的流畅性迅速降低，使文本完全无法使用。这个限制阻碍了解码方法在实现高水平可控性方面的有效性。为了解决这个问题，我们提出了一种新颖的轻量级解码框架，名为空气解码。它的主要思想是通过重建属性分布来平衡属性词和非属性词之间的权重，从而生成更流畅的文本。具体而言，我们通过前缀微调来训练前缀以获得属性分布。然后，我们设计了一种新颖的属性分布重建方法来平衡所获得的分布，并使用重建后的分布进行解码。

    Controllable text generation (CTG) aims to generate text with desired attributes, and decoding-time-based methods have shown promising performance on this task. However, in this paper, we identify the phenomenon of Attribute Collapse for the first time. It causes the fluency of generated text to rapidly decrease when the control strength exceeds a critical value, rendering the text completely unusable. This limitation hinders the effectiveness of decoding methods in achieving high levels of controllability. To address this problem, we propose a novel lightweight decoding framework named Air-Decoding. Its main idea is reconstructing the attribute distributions to balance the weights between attribute words and non-attribute words to generate more fluent text. Specifically, we train prefixes by prefix-tuning to obtain attribute distributions. Then we design a novel attribute distribution reconstruction method to balance the obtained distributions and use the reconstructed distributions t
    
[^49]: QUDEVAL：问句讨论的语篇解析评估

    QUDEVAL: The Evaluation of Questions Under Discussion Discourse Parsing. (arXiv:2310.14520v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.14520](http://arxiv.org/abs/2310.14520)

    本文介绍了第一个用于自动评估问句讨论（QUD）语篇解析的框架QUDeval。使用QUDeval数据集，展示了现代语言模型（LLMs）仍然面临解析所有QUD约束的挑战，并且现有的评估指标很差地近似解析器质量。

    

    问句讨论（QUD）是一个多功能的语言框架，在其中，语篇通过不断提问和回答而进行。将语篇进行自动解析以生成QUD结构，因此涉及到一个复杂的问题生成任务：给定一个文档和一个回答句子，生成满足QUD语言约束并可以在先前环境中绑定到一个锚定句子的问题。这些问题被称为好奇心驱动和开放性的。这项工作引入了第一个用于自动评估QUD解析的框架，将QUD的理论约束具体化为一个具体的协议。我们提出了QUDeval，一个对来自调优系统和LLMs的2,190个QUD问题进行细粒度评估的数据集。使用QUDeval，我们展示了满足所有QUD约束对于现代LLMs仍然具有挑战性，并且现有的评估指标很差地近似解析器质量。令人鼓舞的是，人工编写的QUD得分较高。

    Questions Under Discussion (QUD) is a versatile linguistic framework in which discourse progresses as continuously asking questions and answering them. Automatic parsing of a discourse to produce a QUD structure thus entails a complex question generation task: given a document and an answer sentence, generate a question that satisfies linguistic constraints of QUD and can be grounded in an anchor sentence in prior context. These questions are known to be curiosity-driven and open-ended. This work introduces the first framework for the automatic evaluation of QUD parsing, instantiating the theoretical constraints of QUD in a concrete protocol. We present QUDeval, a dataset of fine-grained evaluation of 2,190 QUD questions generated from both fine-tuned systems and LLMs. Using QUDeval, we show that satisfying all constraints of QUD is still challenging for modern LLMs, and that existing evaluation metrics poorly approximate parser quality. Encouragingly, human-authored QUDs are scored hi
    
[^50]: 大规模文本到图像模型中检测隐性刻板印象的语言代理

    Language Agents for Detecting Implicit Stereotypes in Text-to-image Models at Scale. (arXiv:2310.11778v1 [cs.CY])

    [http://arxiv.org/abs/2310.11778](http://arxiv.org/abs/2310.11778)

    本文介绍了一种针对文本到图像模型中检测刻板印象的语言代理架构，可自主调用各种工具来促进整个检测过程，并应用于商业产品和开放文本数据集。

    

    最近对扩散模型研究的激增加速了各种人工智能生成内容（AIGC）商业产品中文本到图像模型的采用。虽然这些出色的AIGC产品在消费者中获得了越来越多的认可和激发了热情，但关于这些模型可能无意中强化现有社会刻板印象的问题尚未得到解决。受到语言代理最近的进展的启发，我们在这里介绍了一种专为文本到图像模型中的刻板印象检测而设计的新型代理架构。这种多功能的代理架构能够适应自由形式的检测任务，可以自主调用各种工具来促进整个过程，从生成相应的指令和图像到检测刻板印象。我们基于多个开放文本数据集构建了与刻板印象相关的基准，并将这种架构应用于商业产品和流行的开放文本数据集。

    The recent surge in the research of diffusion models has accelerated the adoption of text-to-image models in various Artificial Intelligence Generated Content (AIGC) commercial products. While these exceptional AIGC products are gaining increasing recognition and sparking enthusiasm among consumers, the questions regarding whether, when, and how these models might unintentionally reinforce existing societal stereotypes remain largely unaddressed. Motivated by recent advancements in language agents, here we introduce a novel agent architecture tailored for stereotype detection in text-to-image models. This versatile agent architecture is capable of accommodating free-form detection tasks and can autonomously invoke various tools to facilitate the entire process, from generating corresponding instructions and images, to detecting stereotypes. We build the stereotype-relevant benchmark based on multiple open-text datasets, and apply this architecture to commercial products and popular ope
    
[^51]: EMO: Earth Mover Distance Optimization for Auto-Regressive Language Modeling. (arXiv:2310.04691v2 [cs.CL] UPDATED)

    EMO: Earth Mover Distance Optimization for Auto-Regressive Language Modeling. (arXiv:2310.04691v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.04691](http://arxiv.org/abs/2310.04691)

    EMO提出了地球移动距离优化（EMO）来解决语言模型中的退化现象。EMO利用了地球移动距离的特性，并引入了一个可行的上界来简化训练。经过评估，发现EMO在语言模型上有显著的改进。

    

    神经语言模型是人文本的概率模型。它们主要通过最大似然估计（MLE）进行训练，该方法等同于最小化经验数据分布和模型分布之间的前向交叉熵。然而，当从这些模型学习的分布解码时，仍然经常观察到各种退化现象。我们确定前向交叉熵作为人与模型分布对齐的距离度量是次优的，原因有：（1）召回优化，（2）负样本多样性忽视和（3）训练测试不匹配。在本文中，我们提出了用于自回归语言模型的地球移动距离优化（EMO）。EMO利用地球移动距离的内在特性来解决上述挑战。由于直接计算的复杂性，我们进一步引入了一种可行的EMO上界来简化端到端训练。经过广泛评估之后，发现我们的方法在语言模型上有显著的改进。

    Neural language models are probabilistic models of human text. They are predominantly trained using maximum likelihood estimation (MLE), which is equivalent to minimizing the forward cross-entropy between the empirical data distribution and the model distribution. However, various degeneration phenomena are still widely observed when decoding from the distributions learned by such models. We establish that the forward cross-entropy is suboptimal as a distance metric for aligning human and model distribution due to its (1) recall-prioritization (2) negative diversity ignorance and (3) train-test mismatch. In this paper, we propose Earth Mover Distance Optimization (EMO) for auto-regressive language modeling. EMO capitalizes on the inherent properties of earth mover distance to address the aforementioned challenges. Due to the high complexity of direct computation, we further introduce a feasible upper bound for EMO to ease end-to-end training. Upon extensive evaluation of language model
    
[^52]: LLM和基础设施即代码的用例研究

    LLM and Infrastructure as a Code use case. (arXiv:2309.01456v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2309.01456](http://arxiv.org/abs/2309.01456)

    本文探讨了一种利用生成式语言模型将人类描述转化为代码的解决方案，用于生成和管理Ansible YAML角色和playbooks，以应对云计算和管理方法发展引起的系统构建和维护方法的变革。

    

    云计算和诸如精益管理或敏捷管理等管理方法的演进，引起了系统构建和维护方法的深刻变革。这些实践被统称为"DevOps"。这种描述性的信息系统或应用程序方法，以及其组成部分的配置，已经促使开发了配有专门引擎的描述性语言，用于自动化系统管理任务。在这些工具中，Ansible（引擎）和YAML（描述性语言）的组合是市场上最流行的两种工具，主要面临来自Terraform的竞争。本文介绍了一种利用生成式语言模型（LLMs）将人类描述转化为代码，生成和管理Ansible YAML角色和playbooks的解决方案的探讨。我们的工作集中在确定可行的方向和概述潜在的工业应用。

    Cloud computing and the evolution of management methodologies such as Lean Management or Agile entail a profound transformation in both system construction and maintenance approaches. These practices are encompassed within the term "DevOps." This descriptive approach to an information system or application, alongside the configuration of its constituent components, has necessitated the development of descriptive languages paired with specialized engines for automating systems administration tasks. Among these, the tandem of Ansible (engine) and YAML (descriptive language) stands out as the two most prevalent tools in the market, facing notable competition mainly from Terraform. The current document presents an inquiry into a solution for generating and managing Ansible YAML roles and playbooks, utilizing Generative LLMs (Language Models) to translate human descriptions into code. Our efforts are focused on identifying plausible directions and outlining the potential industrial applicat
    
[^53]: 大语言模型的综合概述

    A Comprehensive Overview of Large Language Models. (arXiv:2307.06435v1 [cs.CL])

    [http://arxiv.org/abs/2307.06435](http://arxiv.org/abs/2307.06435)

    大语言模型的综合概述，分析了各种新的架构和训练策略，讨论了LLM的特点和功能，并总结了重要的研究发现和关键的架构和训练策略。

    

    大语言模型（LLM）展示了出色的泛化能力，导致了众多模型的发展。这些模型提出了各种新的架构，通过改进的训练策略来调整现有的架构，增加上下文长度，使用高质量的训练数据，并增加训练时间以超越基线。分析新的发展对于识别增强训练稳定性和改进LLM泛化能力的变化至关重要。本综述论文全面分析了LLM的架构及其分类、训练策略、训练数据集和性能评估，并讨论未来的研究方向。此外，本文还讨论了LLM的基本构建块和概念，并提供了LLM的完整概述，包括其重要特点和功能。最后，本文总结了LLM研究的重要发现，并整合了关键的架构和训练策略。

    Large Language Models (LLMs) have shown excellent generalization capabilities that have led to the development of numerous models. These models propose various new architectures, tweaking existing architectures with refined training strategies, increasing context length, using high-quality training data, and increasing training time to outperform baselines. Analyzing new developments is crucial for identifying changes that enhance training stability and improve generalization in LLMs. This survey paper comprehensively analyses the LLMs architectures and their categorization, training strategies, training datasets, and performance evaluations and discusses future research directions. Moreover, the paper also discusses the basic building blocks and concepts behind LLMs, followed by a complete overview of LLMs, including their important features and functions. Finally, the paper summarizes significant findings from LLM research and consolidates essential architectural and training strateg
    
[^54]: VoxPoser: 用于带有语言模型的机器人操作的可组合的3D价值映射

    VoxPoser: Composable 3D Value Maps for Robotic Manipulation with Language Models. (arXiv:2307.05973v1 [cs.RO])

    [http://arxiv.org/abs/2307.05973](http://arxiv.org/abs/2307.05973)

    VoxPoser提出了一种新方法，通过组合3D价值映射和语言模型，实现了机器人在多种操作任务下根据自由形式的指令和对象合成机器人轨迹的能力。

    

    研究表明，大型语言模型（LLMs）具有丰富的可行动知识，可以以推理和规划的形式提取出用于机器人操作的信息。尽管取得了进展，大多数模型仍然依赖于预定义的运动原语来执行与环境的物理交互，这仍然是一个重大瓶颈。在这项工作中，我们的目标是在给定开集指令和开集对象的情况下，为各种操作任务合成机器人轨迹，即一系列密集的6-DoF末端执行器路径点。我们首先观察到LLMs在给定自由形式的语言指令时擅长推断可行性和约束。更重要的是，通过利用它们的代码编写能力，它们可以与视觉-语言模型（VLM）交互，以组合3D价值映射将知识接地到Agent的观测空间中。然后在基于模型的规划框架中使用组合的价值映射来零试合成闭环轨迹。

    Large language models (LLMs) are shown to possess a wealth of actionable knowledge that can be extracted for robot manipulation in the form of reasoning and planning. Despite the progress, most still rely on pre-defined motion primitives to carry out the physical interactions with the environment, which remains a major bottleneck. In this work, we aim to synthesize robot trajectories, i.e., a dense sequence of 6-DoF end-effector waypoints, for a large variety of manipulation tasks given an open-set of instructions and an open-set of objects. We achieve this by first observing that LLMs excel at inferring affordances and constraints given a free-form language instruction. More importantly, by leveraging their code-writing capabilities, they can interact with a visual-language model (VLM) to compose 3D value maps to ground the knowledge into the observation space of the agent. The composed value maps are then used in a model-based planning framework to zero-shot synthesize closed-loop ro
    
[^55]: 文本对齐是一个高效的用于海量NLP任务的统一模型

    Text Alignment Is An Efficient Unified Model for Massive NLP Tasks. (arXiv:2307.02729v1 [cs.CL])

    [http://arxiv.org/abs/2307.02729](http://arxiv.org/abs/2307.02729)

    本研究提出了一种高效的文本对齐模型，可以应用于广泛的NLP任务，包括文本蕴含、相似性、问答、事实一致性等。通过对RoBERTa进行轻量级微调，可以构建一个更小规模的模型，实现与大型语言模型相当甚至更优的性能。

    

    大型语言模型（LLMs）通常被设计为下一个词语预测的函数，在广泛的NLP任务中表现出色。尽管具有广泛性，但下一个词语预测对于许多任务来说通常不是一种高效的表达方式，需要极大规模的模型参数（数百亿级别），有时会导致次优的性能。实际上，构建更高效的模型通常是可取的——尽管不够通用，但它们仍然适用于大量问题的子集，并以更小的模型规模实现相当或甚至更优的性能。在本文中，我们将文本对齐提出作为一种高效的统一模型，用于涉及文本蕴含、相似性、问答（和可回答性）、事实一致性等关键任务的广泛范围。给定一对文本，该模型测量它们之间信息的对齐程度。我们通过对RoBERTa（3.55亿参数）进行轻量级微调来实例化一个对齐模型（Align）。

    Large language models (LLMs), typically designed as a function of next-word prediction, have excelled across extensive NLP tasks. Despite the generality, next-word prediction is often not an efficient formulation for many of the tasks, demanding an extreme scale of model parameters (10s or 100s of billions) and sometimes yielding suboptimal performance. In practice, it is often desirable to build more efficient models -- despite being less versatile, they still apply to a substantial subset of problems, delivering on par or even superior performance with much smaller model sizes. In this paper, we propose text alignment as an efficient unified model for a wide range of crucial tasks involving text entailment, similarity, question answering (and answerability), factual consistency, and so forth. Given a pair of texts, the model measures the degree of alignment between their information. We instantiate an alignment model (Align) through lightweight finetuning of RoBERTa (355M parameters)
    
[^56]: EHRSHOT:一种用于少样本评估基础模型的电子健康记录基准

    EHRSHOT: An EHR Benchmark for Few-Shot Evaluation of Foundation Models. (arXiv:2307.02028v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2307.02028](http://arxiv.org/abs/2307.02028)

    该论文介绍了EHRSHOT，一个用于少样本评估基础模型的电子健康记录基准。该论文利用EHRSHOT数据集和预训练模型CLMBR-T-base，为医疗保健ML的发展提供了解决方案。

    

    尽管一般的机器学习(ML)社区已经受益于公开的数据集、任务和模型，但是ML在医疗保健领域的进展受到了共享资产的缺乏的阻碍。基础模型的成功为医疗保健ML带来了新的挑战，需要访问共享的预训练模型来验证性能优势。我们通过三个贡献来帮助解决这些挑战。首先，我们发布了一个新的数据集EHRSHOT，其中包含6,739名来自斯坦福医学的患者的去识别结构化的电子健康记录(EHR)数据。与MIMIC-III/IV和其他流行的EHR数据集不同，EHRSHOT是纵向的，不仅局限于ICU/ED患者。其次，我们发布了CLMBR-T-base的权重，这是一个在结构化EHR数据中预训练的141M参数临床基础模型，该数据包括2.57M名患者。我们是最早完全发布这样一个用于编码EHR数据的模型之一；相比之下，大多数先前发布的临床数据模型（如GatorTron、ClinicalBER）并没有完全发布。

    While the general machine learning (ML) community has benefited from public datasets, tasks, and models, the progress of ML in healthcare has been hampered by a lack of such shared assets. The success of foundation models creates new challenges for healthcare ML by requiring access to shared pretrained models to validate performance benefits. We help address these challenges through three contributions. First, we publish a new dataset, EHRSHOT, which contains deidentified structured data from the electronic health records (EHRs) of 6,739 patients from Stanford Medicine. Unlike MIMIC-III/IV and other popular EHR datasets, EHRSHOT is longitudinal and not restricted to ICU/ED patients. Second, we publish the weights of CLMBR-T-base, a 141M parameter clinical foundation model pretrained on the structured EHR data of 2.57M patients. We are one of the first to fully release such a model for coded EHR data; in contrast, most prior models released for clinical data (e.g. GatorTron, ClinicalBER
    
[^57]: 元推理：用于大型语言模型的语义符号解构

    Meta-Reasoning: Semantics-Symbol Deconstruction For Large Language Models. (arXiv:2306.17820v1 [cs.CL])

    [http://arxiv.org/abs/2306.17820](http://arxiv.org/abs/2306.17820)

    本论文提出了一种称为“元推理”的方法，它通过使用语义符号解构的方式，将不同推理问题转化为类似的自然语言表示，以提高大型语言模型的推理能力。

    

    大型语言模型中的符号化方法已经被证明可以有效提高语言模型的推理能力。然而，大多数这些方法依赖于将自然语言映射到更加语法完备且没有歧义的形式语言（例如Python、SQL）。虽然这些方法有效，但它们离开了自然语言本身，偏离了人类思维的习惯，而更多地迎合了计算机的执行思维方式。相反，我们希望从语言学中符号的概念出发来简化自然语言，使得语言模型可以学习不同自然语义中包含的推理问题的常见表达方式和通用解决方案。基于这种考虑，我们提出了“元推理”，它允许语言模型自动完成语义符号的解构，即语义解析，从而最大程度地将某些推理任务的不同问题减少到类似的自然语言表示，从而获得推理的能力。

    Symbolization methods in large language models (LLMs) have been shown effective to improve LLMs' reasoning ability. However, most of these approaches hinge on mapping natural languages to formal languages (e.g., Python, SQL) that are more syntactically complete and free of ambiguity. Although effective, they depart from the natural language itself and deviate from the habits of human thinking, and instead cater more to the execution mindset of computers. In contrast, we hope to simplify natural language by starting from the concept of symbols in linguistics itself, so that LLMs can learn the common formulation and general solution of reasoning problems wrapped in different natural semantics. From this consideration, we propose \textbf{Meta-Reasoning}, which allows LLMs to automatically accomplish semantic-symbol deconstruction, i.e., semantic resolution, to maximally reduce different questions of certain reasoning tasks to similar natural language representation, thus gaining the abili
    
[^58]: 迭代分段仿射插值（IPA）逼近于语言建模的应用

    Iterated Piecewise Affine (IPA) Approximation for Language Modeling. (arXiv:2306.12317v1 [cs.CL])

    [http://arxiv.org/abs/2306.12317](http://arxiv.org/abs/2306.12317)

    迭代分段仿射插值（IPA）逼近法可以用于语言建模，与变压器解码器架构类似，并在交叉熵损失下的小序列长度下优于变压器1.5％。

    

    本文介绍了一种简单的一阶泰勒展开法来逼近一个通用的函数F: R^{n x m} -> R^{n x m} 并将其应用于语言建模。为了增强基本的泰勒展开，我们引入了迭代和分段建模，从而命名算法为迭代分段仿射插值（IPA）逼近。最终算法表现出与变压器解码器架构相似的有趣特征。通过比较IPA和变压器的参数，我们观察到在较小的序列长度下，IPA在下一个令牌预测任务中使用交叉熵损失比变压器高1.5％。

    In this work, we demonstrate the application of a simple first-order Taylor expansion to approximate a generic function $F: R^{n \times m} \to R^{n \times m}$ and utilize it in language modeling. To enhance the basic Taylor expansion, we introduce iteration and piecewise modeling, leading us to name the algorithm the Iterative Piecewise Affine (IPA) approximation. The final algorithm exhibits interesting resemblances to the Transformers decoder architecture. By comparing parameter arrangements in IPA and Transformers, we observe a strikingly similar performance, with IPA outperforming Transformers by 1.5\% in the next token prediction task with cross-entropy loss for smaller sequence lengths.
    
[^59]: AVIS:利用大型语言模型的自主视觉信息检索

    AVIS: Autonomous Visual Information Seeking with Large Language Models. (arXiv:2306.08129v1 [cs.CV])

    [http://arxiv.org/abs/2306.08129](http://arxiv.org/abs/2306.08129)

    本文提出了一个基于大型语言模型的自主信息检索视觉问答框架AVIS，可以解决视觉问题所需的外部知识获取问题。

    

    本文提出了一种利用大型语言模型（LLM）实现自主信息检索的视觉问答框架AVIS。我们的方法利用LLM动态地制定利用外部工具的策略，并调查它们的输出，从而获取提供所提出问题所需的不可或缺的知识。回答需要外部知识的视觉问题，如“这幅图像所描绘的建筑物是为了纪念哪个事件？”，是一项复杂的任务。这个任务呈现出一个组合搜索空间，需要一系列行动，包括调用API、分析它们的响应并做出明智的决策。我们进行了一个用户研究，收集了人类面对这个任务时各种各样的决策实例。然后利用这些数据设计了一个由三个组件组成的系统：一个由LLM驱动的规划器，动态确定下一个要使用的工具；一个由LLM驱动的推理器，分析并提取关键信息。

    In this paper, we propose an autonomous information seeking visual question answering framework, AVIS. Our method leverages a Large Language Model (LLM) to dynamically strategize the utilization of external tools and to investigate their outputs, thereby acquiring the indispensable knowledge needed to provide answers to the posed questions. Responding to visual questions that necessitate external knowledge, such as "What event is commemorated by the building depicted in this image?", is a complex task. This task presents a combinatorial search space that demands a sequence of actions, including invoking APIs, analyzing their responses, and making informed decisions. We conduct a user study to collect a variety of instances of human decision-making when faced with this task. This data is then used to design a system comprised of three components: an LLM-powered planner that dynamically determines which tool to use next, an LLM-powered reasoner that analyzes and extracts key information 
    
[^60]: Diable: 在表格上进行的高效对话状态跟踪

    Diable: Efficient Dialogue State Tracking as Operations on Tables. (arXiv:2305.17020v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.17020](http://arxiv.org/abs/2305.17020)

    Diable是一个高效的对话状态跟踪系统，它通过在表格上进行操作来更新对话状态，相比现有方法时间效率提高了2.4倍，同时保持了竞争性的目标准确性。

    

    目前的对话状态跟踪系统将完整的对话历史作为输入，将当前状态表示为包含所有槽的列表，并在每个对话回合中从头开始生成整个状态。这种方法效率低下，特别是当槽的数量很多且对话很长时。我们提出了Diable，一种新的任务形式化方法，简化了高效对话状态跟踪系统的设计和实现，并且可以轻松地嵌入大型语言模型。我们将对话状态表示为表格，并将对话状态跟踪形式化为表格操作任务。在每个回合中，系统通过基于对话上下文生成表格操作来更新先前的状态。在MultiWoz数据集上进行了大量实验，结果显示，Diable (i) 优于强大的高效对话状态跟踪基准，(ii) 时间效率比当前最先进的方法提高了2.4倍，同时保持竞争性的联合目标准确性，并且(iii) 对无噪声的输入具有鲁棒性。

    Sequence-to-sequence state-of-the-art systems for dialogue state tracking (DST) use the full dialogue history as input, represent the current state as a list with all the slots, and generate the entire state from scratch at each dialogue turn. This approach is inefficient, especially when the number of slots is large and the conversation is long. We propose Diable, a new task formalisation that simplifies the design and implementation of efficient DST systems and allows one to easily plug and play large language models. We represent the dialogue state as a table and formalise DST as a table manipulation task. At each turn, the system updates the previous state by generating table operations based on the dialogue context. Extensive experimentation on the MultiWoz datasets demonstrates that Diable (i) outperforms strong efficient DST baselines, (ii) is 2.4x more time efficient than current state-of-the-art methods while retaining competitive Joint Goal Accuracy, and (iii) is robust to no
    
[^61]: SOCRATIC QUESTIONING的艺术：与大型语言模型的递归思维

    The Art of SOCRATIC QUESTIONING: Recursive Thinking with Large Language Models. (arXiv:2305.14999v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.14999](http://arxiv.org/abs/2305.14999)

    SOCRATIC QUESTIONING是一种与大型语言模型合作进行递归思考的算法，通过模拟人类的认知过程，它能够解决复杂推理问题，提高准确性和效率。

    

    链式思维（Chain-of-Thought，CoT）提示能够使大型语言模型解决复杂的推理问题，通过生成中间步骤。然而，受制于其内在的单遍和顺序生成过程，CoT在初始决策上过度依赖，导致早期步骤中的错误累积并影响最终答案。相比之下，人类在处理复杂推理问题时采用递归思维，即将原问题迭代地分解为可处理的子问题，并汇总其答案以解决原问题。受到人类认知过程的启发，我们提出了SOCRATIC QUESTIONING，一种模拟递归思维过程的分而治之算法。具体而言，SOCRATIC QUESTIONING利用大型语言模型提出和回答子问题，直到收集足够的信息来解决原问题。与CoT不同，SOCRATIC QUESTIONING明确地导航思考空间，促进有效的递归思维。

    Chain-of-Thought (CoT) prompting enables large language models to solve complex reasoning problems by generating intermediate steps. However, confined by its inherent single-pass and sequential generation process, CoT heavily relies on the initial decisions, causing errors in early steps to accumulate and impact the final answers. In contrast, humans adopt recursive thinking when tackling complex reasoning problems, i.e., iteratively breaking the original problem into approachable sub-problems and aggregating their answers to resolve the original one. Inspired by the human cognitive process, we propose SOCRATIC QUESTIONING, a divide-and-conquer style algorithm that mimics the recursive thinking process. Specifically, SOCRATIC QUESTIONING leverages large language models to raise and answer sub-questions until collecting enough information to tackle the original question. Unlike CoT, SOCRATIC QUESTIONING explicitly navigates the thinking space, stimulates effective recursive thinking, an
    
[^62]: 面向公共论坛的可法律强制执行的仇恨言论检测研究

    Towards Legally Enforceable Hate Speech Detection for Public Forums. (arXiv:2305.13677v1 [cs.CL])

    [http://arxiv.org/abs/2305.13677](http://arxiv.org/abs/2305.13677)

    本研究提出了一个以法律定义为中心的、可法律强制执行的仇恨言论检测任务，利用法律专家对数据集进行了注释，结合基于零样本和小样本的提示，可以使模型的输出更符合监管者目标。

    

    仇恨言论是公共论坛上的严重问题，对恶意和歧视性语言的适当执行是保护人群免受伤害和歧视的关键。然而，确定什么构成仇恨言论是一项非常复杂的任务，高度容易受到主观解释的影响。现有的作品没有将它们的系统与可执行的仇恨言论定义对齐，这可能会使它们的输出与监管者的目标不一致。我们的研究引入了一个新的任务，即以法律定义为中心的可执行仇恨言论检测，并使用法律专家对违反十一种可能定义进行了数据集注释。考虑到确定清晰、可法律强制执行的仇恨言论的挑战，我们使用专家生成的样本和自动挖掘的挑战集增强了数据集。我们尝试使用零样本和小样本的提示来基于这些定义来决定模型的输出。然后，我们报告了在几个大型语言模型上的结果。

    Hate speech is a serious issue on public forums, and proper enforcement of hate speech laws is key for protecting groups of people against harmful and discriminatory language. However, determining what constitutes hate speech is a complex task that is highly open to subjective interpretations. Existing works do not align their systems with enforceable definitions of hate speech, which can make their outputs inconsistent with the goals of regulators. Our work introduces a new task for enforceable hate speech detection centred around legal definitions, and a dataset annotated on violations of eleven possible definitions by legal experts. Given the challenge of identifying clear, legally enforceable instances of hate speech, we augment the dataset with expert-generated samples and an automatically mined challenge set. We experiment with grounding the model decision in these definitions using zero-shot and few-shot prompting. We then report results on several large language models (LLMs). 
    
[^63]: 实现在真实场景中对齐的开放世界半监督广义关系发现 (arXiv:2305.13533v1 [cs.CL])

    Open-world Semi-supervised Generalized Relation Discovery Aligned in a Real-world Setting. (arXiv:2305.13533v1 [cs.CL])

    [http://arxiv.org/abs/2305.13533](http://arxiv.org/abs/2305.13533)

    本文提出了一种新的开放世界关系抽取方法，能够在已知类和新颖类中进行显式和隐式表示的关系分类，在真实场景数据的特征下进行了两个关键改进。

    

    开放世界关系抽取(OpenRE)最近引起了人们的关注。然而，现有的方法往往简化了问题，假设所有未标记的文本都属于新类，从而限制了这些方法的实用性。我们认为OpenRE设置应更符合现实世界数据的特征。具体而言，我们提出了两个关键改进:(a)未标记数据应包括已知和新颖的类，包括难以区分的负样本实例;(b)新颖的类集应该代表长尾关系类型。此外，我们观察到，流行的关系，如标题和位置，通常可以通过特定的模式隐含地推断，而长尾的关系倾向于在句子中明确表示。在这些见解的推动下，我们提出了一种名为KNoRD（已知和新颖关系发现）的新方法，有效地对已知类和新颖类中的显式和隐式表示的关系进行分类。

    Open-world Relation Extraction (OpenRE) has recently garnered significant attention. However, existing approaches tend to oversimplify the problem by assuming that all unlabeled texts belong to novel classes, thereby limiting the practicality of these methods. We argue that the OpenRE setting should be more aligned with the characteristics of real-world data. Specifically, we propose two key improvements: (a) unlabeled data should encompass known and novel classes, including hard-negative instances; and (b) the set of novel classes should represent long-tail relation types. Furthermore, we observe that popular relations such as titles and locations can often be implicitly inferred through specific patterns, while long-tail relations tend to be explicitly expressed in sentences. Motivated by these insights, we present a novel method called KNoRD (Known and Novel Relation Discovery), which effectively classifies explicitly and implicitly expressed relations from known and novel classes w
    
[^64]: SEAHORSE: 多语言、多方面摘要评估的数据集

    SEAHORSE: A Multilingual, Multifaceted Dataset for Summarization Evaluation. (arXiv:2305.13194v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.13194](http://arxiv.org/abs/2305.13194)

    SEAHORSE是一个用于多语言、多方面摘要评估的数据集，包含96K个摘要，涵盖6种语言、9个系统和4个数据集。SEAHORSE是一个用于评估学习度量和训练度量的大规模资源。使用SEAHORSE训练的度量在领域外的元评估基准上表现出了强大的性能。

    

    可靠的自动摘要评估由于任务的多方面和主观性质而具有挑战性。尤其对于除英语以外的语言，人工评估稀缺。在这项工作中，我们介绍了SEAHORSE，这是一个用于多语言、多方面摘要评估的数据集。SEAHORSE包含96K个摘要，涵盖了6种语言、9个系统和4个数据集，并根据文本质量的6个维度进行了人工评分：可理解性、重复性、语法、归因、主要观点和简洁性。由于其规模和范围的原因，SEAHORSE既可以作为评估学习度量的基准，也可以作为训练这些度量的大规模资源。我们展示了使用SEAHORSE训练的度量在领域外的元评估基准TRUE和mFACE上取得了很好的性能。我们将SEAHORSE数据集和度量公开提供，以供未来的多语言和多

    Reliable automatic evaluation of summarization systems is challenging due to the multifaceted and subjective nature of the task. This is especially the case for languages other than English, where human evaluations are scarce. In this work, we introduce SEAHORSE, a dataset for multilingual, multifaceted summarization evaluation. SEAHORSE consists of 96K summaries with human ratings along 6 dimensions of text quality: comprehensibility, repetition, grammar, attribution, main ideas, and conciseness, covering 6 languages, 9 systems and 4 datasets. As a result of its size and scope, SEAHORSE can serve both as a benchmark to evaluate learnt metrics, as well as a large-scale resource for training such metrics. We show that metrics trained with SEAHORSE achieve strong performance on the out-of-domain meta-evaluation benchmarks TRUE (Honovich et al., 2022) and mFACE (Aharoni et al., 2022). We make the SEAHORSE dataset and metrics publicly available for future research on multilingual and multi
    
[^65]: 使用ICA发现嵌入中的通用几何结构

    Discovering Universal Geometry in Embeddings with ICA. (arXiv:2305.13175v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.13175](http://arxiv.org/abs/2305.13175)

    本研究利用ICA揭示了嵌入中的通用几何结构，证明了每个嵌入可以由少量内在可解释轴的组合表示，并且这些语义轴在不同的语言、算法和模态下保持一致。

    

    本研究利用独立成分分析（ICA）揭示了单词或图像嵌入中的一致语义结构。我们的方法通过利用主成分分析（PCA）中白化过程后残留的各向异性信息，从预训练模型的嵌入中提取独立的语义成分。我们证明了每个嵌入可以被表示为少量内在可解释轴的组合，并且这些语义轴在不同的语言、算法和模态下保持一致。在嵌入的几何模式中发现通用语义结构，增强了我们对嵌入中表示的理解。

    This study utilizes Independent Component Analysis (ICA) to unveil a consistent semantic structure within embeddings of words or images. Our approach extracts independent semantic components from the embeddings of a pre-trained model by leveraging anisotropic information that remains after the whitening process in Principal Component Analysis (PCA). We demonstrate that each embedding can be expressed as a composition of a few intrinsic interpretable axes and that these semantic axes remain consistent across different languages, algorithms, and modalities. The discovery of a universal semantic structure in the geometric patterns of embeddings enhances our understanding of the representations in embeddings.
    
[^66]: 文本预训练的语音语言模型

    Textually Pretrained Speech Language Models. (arXiv:2305.13009v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.13009](http://arxiv.org/abs/2305.13009)

    本论文提出了一种使用预训练的文本语言模型训练语音语言模型的方法，通过对模型设计选择和数据集规模的经验性分析，构建了参数数量和训练数据最多的语音语言模型，并引入了两个Spoken版本的文本基准，以进一步改善模型评估和推动未来研究。

    

    语音语言模型（SpeechLMs）仅处理和生成音频数据，没有文字监督。在这项工作中，我们提出了TWIST，一种使用预训练的文本语言模型进行SpeechLMs训练的方法。我们通过自动和人工评估表明，TWIST在各个方面都优于冷启动的SpeechLM。我们经验性地分析了不同的模型设计选择（如语音分词器、预训练的文本模型和数据集大小）的影响。我们发现模型和数据集规模在构建性能更好的SpeechLMs方面都起着重要作用。基于我们的观察，我们介绍了迄今为止参数数量和训练数据最多的SpeechLM（据我们所知）。此外，我们还引入了两个Spoken版本的StoryCloze文本基准，以进一步改善模型评估并推动该领域的未来研究。我们公开提供语音样本、代码和模型：https://pages.cs.huji.ac.il/

    Speech language models (SpeechLMs) process and generate acoustic data only, without textual supervision. In this work, we propose TWIST, a method for training SpeechLMs using a warm-start from a pretrained textual language models. We show using both automatic and human evaluations that TWIST outperforms a cold-start SpeechLM across the board. We empirically analyze the effect of different model design choices such as the speech tokenizer, the pretrained textual model, and the dataset size. We find that model and dataset scale both play an important role in constructing better-performing SpeechLMs. Based on our observations, we present the largest (to the best of our knowledge) SpeechLM both in terms of number of parameters and training data. We additionally introduce two spoken versions of the StoryCloze textual benchmark to further improve model evaluation and advance future research in the field. We make speech samples, code and models publicly available: https://pages.cs.huji.ac.il/
    
[^67]: TempoSum：评估抽象摘要的时间泛化能力

    TempoSum: Evaluating the Temporal Generalization of Abstractive Summarization. (arXiv:2305.01951v1 [cs.CL])

    [http://arxiv.org/abs/2305.01951](http://arxiv.org/abs/2305.01951)

    本篇论文提出了 TempoSum 抽象摘要的时间泛化能力基准，通过广泛的人类评估证明了摘要模型中存储的参数化知识对未来数据上生成的摘要有显著影响。

    

    最近，预训练语言模型在现有的抽象摘要数据集中取得了有 promising 的结果。然而，现有的摘要基准与标准的预训练语料库和微调数据集在时间上重叠。因此，预训练语言模型的强大性能可能依赖于预训练和微调过程中所记忆的参数化知识。此外，预训练语言模型所记忆的知识可能很快就过时，这会影响到它们在未来数据上的泛化性能。为了了解抽象摘要模型的时间泛化能力，本文提出了 TempoSum，一个新的基准，其中包含了从 2010 年到 2022 年的数据样本。通过广泛的人类评估，我们证明了摘要模型中存储的参数化知识对未来数据上生成的摘要的准确性有显著影响。此外，现有的准确性提高方法不能可靠地提高摘要模型在未来数据上的准确性。

    Recent pre-trained language models (PLMs) achieve promising results in existing abstractive summarization datasets. However, existing summarization benchmarks overlap in time with the standard pre-training corpora and finetuning datasets. Hence, the strong performance of PLMs may rely on the parametric knowledge that is memorized during pre-training and fine-tuning. Moreover, the knowledge memorized by PLMs may quickly become outdated, which affects the generalization performance of PLMs on future data. In this work, we propose TempoSum, a novel benchmark that contains data samples from 2010 to 2022, to understand the temporal generalization ability of abstractive summarization models. Through extensive human evaluation, we show that parametric knowledge stored in summarization models significantly affects the faithfulness of the generated summaries on future data. Moreover, existing faithfulness enhancement methods cannot reliably improve the faithfulness of summarization models on fu
    
[^68]: GPT-2是如何计算大于符号的？解释预训练语言模型中的数学能力

    How does GPT-2 compute greater-than?: Interpreting mathematical abilities in a pre-trained language model. (arXiv:2305.00586v1 [cs.CL])

    [http://arxiv.org/abs/2305.00586](http://arxiv.org/abs/2305.00586)

    本研究运用机械式可解释性技术探究了GPT-2 Small的数学能力，并确定了它的计算图中的一个小电路用于计算大于符号，该电路的多层感知器提高了结束年份大于开始年份的概率，并且该电路具有广泛的适用性。

    

    预训练语言模型在未被明确训练的任务上表现出惊人的能力，但它们如何实现这些功能却不为人所知。本文通过机械式可解释性技术探究预训练语言模型通常具有的基本数学能力。具体来说，我们以GPT-2 Small为例，研究其能否通过输入"战争持续时间是从1732年到17年"，预测出有效的两位数字的截止年份 (大于32年)。我们首先确定了一个电路，即GPT-2 Small计算图的一个小子集，用于计算这个任务的输出，然后我们解释了每个电路组件的作用，显示出GPT-2 Small的最终多层感知器提高了结束年份大于开始年份的概率。最后，我们证明了我们的电路适用于其他任务，在其他大于场景中发挥作用。

    Pre-trained language models can be surprisingly adept at tasks they were not explicitly trained on, but how they implement these capabilities is poorly understood. In this paper, we investigate the basic mathematical abilities often acquired by pre-trained language models. Concretely, we use mechanistic interpretability techniques to explain the (limited) mathematical abilities of GPT-2 small. As a case study, we examine its ability to take in sentences such as "The war lasted from the year 1732 to the year 17", and predict valid two-digit end years (years > 32). We first identify a circuit, a small subset of GPT-2 small's computational graph that computes this task's output. Then, we explain the role of each circuit component, showing that GPT-2 small's final multi-layer perceptrons boost the probability of end years greater than the start year. Finally, we show that our circuit generalizes to other tasks, playing a role in other greater-than scenarios.
    
[^69]: ParroT: 使用大型语言模型进行聊天翻译

    ParroT: Translating During Chat Using Large Language Models. (arXiv:2304.02426v1 [cs.CL])

    [http://arxiv.org/abs/2304.02426](http://arxiv.org/abs/2304.02426)

    ParroT提出了一种基于开源LLM和人工编写的翻译评估数据的聊天翻译框架，可以将翻译数据转化为指令执行样式，并引入额外要求来规范翻译过程。在使用相对较少的训练数据的情况下，实验结果表明 ParroT 可以大幅提高翻译质量。

    

    大型语言模型（LLM）如 ChatGPT 和 GPT-4 在各种自然语言处理（NLP）任务上展现出了卓越的能力，包括在聊天过程中完成各种机器翻译能力。然而，这些模型只能通过受限的API访问，这为新的研究和领域进展带来了障碍。因此，我们提出了 ParroT 框架，基于开源LLM（如LLaMA-7b）和人工编写的翻译评估数据来增强和规范聊天翻译能力。具体而言，ParroT将翻译数据转化为指令执行的样式，并引入 "Hint " 字段以加入额外要求来规范翻译过程。因此，我们提出了三种指令类型来微调 ParroT 模型，包括翻译指令、对比指令和误差引导指令。在两个 Flores 子集和 WMT22 测试集上的实验证明，使用 ParroT 可以大幅提高翻译质量，且需要相对较少的训练数据。

    Large language models (LLMs) like ChatGPT and GPT-4 have exhibited remarkable abilities on a wide range of natural language processing (NLP) tasks, including various machine translation abilities accomplished during chat. However, these models are only accessible through restricted APIs, which creates barriers to new research and advancements in the field. Therefore, we propose the $\mathbf{ParroT}$ framework to enhance and regulate the translation abilities during chat based on open-sourced LLMs (i.e., LLaMA-7b) and human written translation and evaluation data. Specifically, ParroT reformulates translation data into the instruction-following style, and introduces a "Hint" field for incorporating extra requirements to regulate the translation process. Accordingly, we propose three instruction types for finetuning ParroT models, including translation instruction, contrastive instruction, and error-guided instruction. Experiments on two Flores subsets and WMT22 test sets suggest that tr
    
[^70]: CAMEL: 用于“心智”探索大规模语言模型社群的交互式代理

    CAMEL: Communicative Agents for "Mind" Exploration of Large Scale Language Model Society. (arXiv:2303.17760v1 [cs.AI])

    [http://arxiv.org/abs/2303.17760](http://arxiv.org/abs/2303.17760)

    本文介绍了一个名为角色扮演的新型交互式代理框架，用于实现语言模型之间的自主合作，并展示了其在生成对话数据方面的有效性。

    

    对话式语言模型的快速发展已取得了在复杂任务解决方面的显著进展。然而，它们的成功在很大程度上依赖于人类的指导，以引导对话，这可能是具有挑战性和耗时的。本文探讨了构建可扩展技术以促进交互式代理之间的自主合作并深入了解它们的“认知”过程的潜力。为了解决实现自主合作的挑战，我们提出了一个名为角色扮演的新型交互式代理框架。我们的方法涉及使用启动提示来引导聊天代理完成任务，同时保持与人类意图的一致性。我们展示了如何使用角色扮演来生成对话数据，以研究聊天代理的行为和能力，为研究对话式语言模型提供了有价值的资源。我们的贡献是介绍了一种新型的交互式代理框架，名为角色扮演，用于实现语言模型之间的自主合作，并展示了其在生成对话数据方面的有效性。

    The rapid advancement of conversational and chat-based language models has led to remarkable progress in complex task-solving. However, their success heavily relies on human input to guide the conversation, which can be challenging and time-consuming. This paper explores the potential of building scalable techniques to facilitate autonomous cooperation among communicative agents and provide insight into their "cognitive" processes. To address the challenges of achieving autonomous cooperation, we propose a novel communicative agent framework named role-playing. Our approach involves using inception prompting to guide chat agents toward task completion while maintaining consistency with human intentions. We showcase how role-playing can be used to generate conversational data for studying the behaviors and capabilities of chat agents, providing a valuable resource for investigating conversational language models. Our contributions include introducing a novel communicative agent framewor
    
[^71]: DeltaScore: 利用差分扰动评价故事生成

    DeltaScore: Evaluating Story Generation with Differentiating Perturbations. (arXiv:2303.08991v1 [cs.CL])

    [http://arxiv.org/abs/2303.08991](http://arxiv.org/abs/2303.08991)

    DeltaScore利用差分扰动来评估故事生成的细粒度方面，并通过计算故事在特定方面扰动前后的可能性差异来衡量影响。该方法在多个故事领域中得到了评估，并与人类判断的相关性进行了研究。

    

    自然语言生成的各种评价指标存在，但对于故事生成的实用性有限，因为它们通常与人类判断的相关性不强，也不能测量细粒度的故事方面，例如流畅度与相关性，因为它们旨在评估整体生成质量。本文提出DeltaScore，一种利用扰动来评估细粒度故事方面的方法。我们的核心思想是基于这样的假设：故事在特定方面表现得越好（例如流畅度），它就会受到特定扰动（例如引入错别字）的影响越大。为了衡量影响，我们使用语言模型计算扰动前后故事的可能性差异。我们在多个故事领域中使用DeltaScore评估了基于状态的最新模型和传统基于相似性的指标，并研究了它与人类在五个细粒度故事方面的判断之间的相关性。

    Various evaluation metrics exist for natural language generation tasks, but they have limited utility for story generation since they generally do not correlate well with human judgments and do not measure fine-grained story aspects, such as fluency versus relatedness, as they are intended to assess overall generation quality. In this paper, we propose deltascore, an approach that utilizes perturbation to evaluate fine-grained story aspects. Our core idea is based on the hypothesis that the better the story performs in a specific aspect (e.g., fluency), the more it will be affected by a particular perturbation (e.g., introducing typos). To measure the impact, we calculate the likelihood difference between the pre- and post-perturbation stories using a language model. We evaluate deltascore against state-of-the-art model-based and traditional similarity-based metrics across multiple story domains, and investigate its correlation with human judgments on five fine-grained story aspects: f
    
[^72]: 数据中心机器学习的重新标签法

    The Re-Label Method For Data-Centric Machine Learning. (arXiv:2302.04391v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2302.04391](http://arxiv.org/abs/2302.04391)

    本文提出了一种重新标签的方法来解决手动标记的数据中存在噪声的问题，并通过模型预测来辅助人类标记噪声数据。实验证明此方法适用于多类深度学习任务。

    

    在深度学习应用中，手动标记的数据在一定程度上存在噪声。为了解决这个问题，并在开发数据集上获得90分以上的成绩，本文提出了一种简单的方法来找出噪声数据，并通过采用模型预测作为人类标记的参考来重新标记噪声数据。本文阐述了我们在广泛的深度学习任务中的想法，包括分类、序列标记、物体检测、序列生成、点击率预测。实验结果和人类评估结果验证了我们的想法。

    In industry deep learning application, our manually labeled data has a certain number of noisy data. To solve this problem and achieve more than 90 score in dev dataset, we present a simple method to find the noisy data and re-label the noisy data by human, given the model predictions as references in human labeling. In this paper, we illustrate our idea for a broad set of deep learning tasks, includes classification, sequence tagging, object detection, sequence generation, click-through rate prediction. The experimental results and human evaluation results verify our idea.
    
[^73]: ChatGPT作为翻译引擎，依赖于GPT-4，是一种好的翻译器吗？

    Is ChatGPT A Good Translator? Yes With GPT-4 As The Engine. (arXiv:2301.08745v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2301.08745](http://arxiv.org/abs/2301.08745)

    本论文评估了ChatGPT的机器翻译能力，发现它在高资源欧洲语言上表现良好，但在低资源或远程语言上表现滞后；采用枢轴提示可以显著提高远程语言翻译的性能；在生物医学摘要或Reddit评论方面，ChatGPT的表现不如商业系统。

    

    本报告对ChatGPT进行了机器翻译的初步评估，包括翻译提示、多语言翻译和翻译的鲁棒性。我们采用ChatGPT建议的提示来触发其翻译能力，发现候选提示通常运行良好，表现出轻微的性能差异。通过在多个基准测试集上进行评估，发现在高资源的欧洲语言上，ChatGPT的表现与商业翻译产品（例如Google翻译）相当，但在低资源或远程语言上表现显著滞后。对于远程语言，我们探索了一种有趣的策略，称为$\mathbf{枢轴提示}$，即让ChatGPT先将源语言句子翻译成高资源的轴语言，再翻译目标语言，这显著提高了翻译性能。至于翻译的鲁棒性，在生物医学摘要或Reddit评论方面，ChatGPT的表现不如商业系统。

    This report provides a preliminary evaluation of ChatGPT for machine translation, including translation prompt, multilingual translation, and translation robustness. We adopt the prompts advised by ChatGPT to trigger its translation ability and find that the candidate prompts generally work well and show minor performance differences. By evaluating on a number of benchmark test sets, we find that ChatGPT performs competitively with commercial translation products (e.g., Google Translate) on high-resource European languages but lags behind significantly on low-resource or distant languages. For distant languages, we explore an interesting strategy named $\mathbf{pivot~prompting}$ that asks ChatGPT to translate the source sentence into a high-resource pivot language before into the target language, which improves the translation performance significantly. As for the translation robustness, ChatGPT does not perform as well as the commercial systems on biomedical abstracts or Reddit commen
    
[^74]: 词向量的范数编码信息增益

    Norm of Word Embedding Encodes Information Gain. (arXiv:2212.09663v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.09663](http://arxiv.org/abs/2212.09663)

    本文研究发现，跳字模型和负采样方法中静态词向量的平方范数编码了词所传达的信息增益，通过与语料库中单词的分布之间的KL散度来定义，可用于关键词提取、词性区分和上位词分类等任务。

    

    词的分布式表示编码了词汇语义信息，但是编码了哪些类型的信息？以及如何编码？本文针对跳字模型和负采样方法，发现静态词向量的平方范数编码了词所传达的信息增益；而信息增益是通过词在共现分布和语料库的单词分布之间的KL散度来定义的。我们的发现是通过指数族概率分布的理论框架说明的，并通过消除词频引起的虚假相关性的精密实验进行了确认。我们证明，无论是KL散度还是词嵌入的平方范数，在关键词提取、词性区分和上位词分类等任务中都提供了有用的词信息度量。

    Distributed representations of words encode lexical semantic information, but what type of information is encoded, and how? Focusing on the skip-gram with negative-sampling method, we found that the squared norm of static word embedding encodes the information gain conveyed by the word; the information gain is defined by the Kullback-Leibler divergence of the co-occurrence distribution of the word to the unigram distribution of the corpus. Our findings are explained by the theoretical framework of the exponential family of probability distributions and confirmed through precise experiments that remove spurious correlations arising from word frequency. We demonstrate that both the KL divergence and the squared norm of embedding provide a useful metric of a word's informativeness in tasks such as keyword extraction, part-of-speech discrimination, and hypernym classification.
    
[^75]: Rainproof:一种用于保护文本生成器免受来自分布外数据的雨伞

    Rainproof: An Umbrella To Shield Text Generators From Out-Of-Distribution Data. (arXiv:2212.09171v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.09171](http://arxiv.org/abs/2212.09171)

    该论文提出了一种名为RAINPROOF的相对信息投影OOD检测框架，该框架可以在黑盒模型中利用软概率进行检测。论文还提供了一种更实际的OOD检测评估设置。研究发现，OOD检测不一定与任务特定的度量相一致。

    

    在部署的NLP模型中，从翻译到聊天机器人，实施有效的控制机制以确保正确运行和安全性至关重要。确保安全系统行为的关键要素是Out-Of-Distribution（OOD）检测，旨在检测输入样本是否与训练分布统计上过于偏离。尽管OOD检测是分类任务中广泛讨论的话题，但大多数方法依赖于编码器输出的隐藏特征。在这项工作中，我们专注于在黑盒框架中利用软概率，即我们可以访问软预测但不能访问模型的内部状态。我们的贡献包括：（i）RAINPROOF相对信息投影OOD检测框架；和（ii）一种更实际的OOD检测评估设置。令人惊讶的是，我们发现OOD检测不一定与任务特定的度量相一致。OOD检测器可能会过滤掉模型处理得很好的样本，同时保留一些样本，这些样本其实模型处理得不好。

    Implementing effective control mechanisms to ensure the proper functioning and security of deployed NLP models, from translation to chatbots, is essential. A key ingredient to ensure safe system behaviour is Out-Of-Distribution (OOD) detection, which aims to detect whether an input sample is statistically far from the training distribution. Although OOD detection is a widely covered topic in classification tasks, most methods rely on hidden features output by the encoder. In this work, we focus on leveraging soft-probabilities in a black-box framework, i.e. we can access the soft-predictions but not the internal states of the model. Our contributions include: (i) RAINPROOF a Relative informAItioN Projection OOD detection framework; and (ii) a more operational evaluation setting for OOD detection. Surprisingly, we find that OOD detection is not necessarily aligned with task-specific measures. The OOD detector may filter out samples well processed by the model and keep samples that are n
    
[^76]: 通过利用自注意力矩阵提高词移距离的性能

    Improving word mover's distance by leveraging self-attention matrix. (arXiv:2211.06229v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.06229](http://arxiv.org/abs/2211.06229)

    本研究利用自注意力矩阵改进了词移距离（Word Mover's Distance，WMD）的性能，通过考虑句子结构和词嵌入的相似度，实现了在近义词识别和语义文本相似度中较好的表现。

    

    在衡量两个句子之间的语义相似性仍然是一个重要任务，词移距离（WMD）通过计算词向量集之间的最优对齐来计算相似性。然而，WMD没有利用词序，这使得它难以区分具有相似词汇重叠的句子，即使它们在语义上非常不同。在这里，我们尝试通过融合BERT的自注意力矩阵（SAM）来改进WMD。所提出的方法基于融合Gromov-Wasserstein距离，同时考虑了词嵌入和SAM的相似度，以计算两个句子之间的最优转运。实验证明，所提出的方法在近义词识别方面提高了WMD及其变体，与语义文本相似性中的几乎相等的性能。我们的代码可在\url{https://github.com/ymgw55/WSMD}获得。

    Measuring the semantic similarity between two sentences is still an important task. The word mover's distance (WMD) computes the similarity via the optimal alignment between the sets of word embeddings. However, WMD does not utilize word order, making it challenging to distinguish sentences with significant overlaps of similar words, even if they are semantically very different. Here, we attempt to improve WMD by incorporating the sentence structure represented by BERT's self-attention matrix (SAM). The proposed method is based on the Fused Gromov-Wasserstein distance, which simultaneously considers the similarity of the word embedding and the SAM for calculating the optimal transport between two sentences. Experiments demonstrate the proposed method enhances WMD and its variants in paraphrase identification with near-equivalent performance in semantic textual similarity. Our code is available at \url{https://github.com/ymgw55/WSMD}.
    
[^77]: 用于基于穿梭式离子阱量子计算机的量子电路编译器

    Quantum Circuit Compiler for a Shuttling-Based Trapped-Ion Quantum Computer. (arXiv:2207.01964v3 [quant-ph] UPDATED)

    [http://arxiv.org/abs/2207.01964](http://arxiv.org/abs/2207.01964)

    本文介绍了一个针对穿梭式离子阱量子处理器的量子电路编译器，能够将量子电路转换和优化为特定的本地门序列，与标准编译方法相比，可以将门计数减少到5.1倍。

    

    随着量子计算硬件能力的增强和实现深度量子电路的挑战，需要完全自动化和高效的工具来编译量子电路。为了在特定于量子计算机架构的本地门序列中表示任意电路，需要使算法在量子硬件供应商的范围内可移植。本研究提出了一种编译器，可以将量子电路转换和优化为针对穿梭式离子阱量子处理器的目标电路。它由基于量子电路框架Pytket的定制算法组成。对广泛的量子电路进行了性能评估，结果表明，与标准Pytket相比，门计数可以减少多达5.1倍，与标准Qiskit编译相比可以减少多达2.2倍。

    The increasing capabilities of quantum computing hardware and the challenge of realizing deep quantum circuits require fully automated and efficient tools for compiling quantum circuits. To express arbitrary circuits in a sequence of native gates specific to the quantum computer architecture, it is necessary to make algorithms portable across the landscape of quantum hardware providers. In this work, we present a compiler capable of transforming and optimizing a quantum circuit targeting a shuttling-based trapped-ion quantum processor. It consists of custom algorithms set on top of the quantum circuit framework Pytket. The performance was evaluated for a wide range of quantum circuits and the results show that the gate counts can be reduced by factors up to 5.1 compared to standard Pytket and up to 2.2 compared to standard Qiskit compilation.
    

