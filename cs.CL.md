# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Skip $\textbackslash n$: A simple method to reduce hallucination in Large Vision-Language Models](https://rss.arxiv.org/abs/2402.01345) | 本文提出了一种新的视角，指出LVLMs中固有的偏见可能是多模态幻觉的关键因素。通过系统识别与段落分割符相关的语义漂移偏差，我们发现模型在训练数据中经常遇到明显的内容语义变化，导致幻觉的产生。 |
| [^2] | [On the Semantics of LM Latent Space: A Vocabulary-defined Approach](https://rss.arxiv.org/abs/2401.16184) | 本论文介绍了一种以词汇为定义的语义学方法，建立了LM潜在空间的参考框架，确保基于LM词汇的分离语义分析。在LM适应过程中，引入了计算logits的新技术和神经聚类模块，通过实验证明了该方法在文本理解上的优越性能。 |
| [^3] | [A systematic investigation of learnability from single child linguistic input](https://arxiv.org/abs/2402.07899) | 我们的研究探索了用单个儿童的语言输入训练语言模型的可学习性，我们发现这种设置下的语言模型能够形成句法和语义词群，并对某些语言现象具有敏感性。 |
| [^4] | [Suppressing Pink Elephants with Direct Principle Feedback](https://arxiv.org/abs/2402.07896) | 本研究提出了一种名为“直接原则反馈”的新方法，用于控制语言模型中的LLM行为。通过在批评和修订上直接使用DPO来跳过响应的排名，我们成功地解决了“粉色大象问题”并取得了显著的性能优势。 |
| [^5] | [Label-Efficient Model Selection for Text Generation](https://arxiv.org/abs/2402.07891) | DiffUse是一种标注效率高的文本生成模型选择方法，它通过聚类文本语义差异的嵌入来选择更具信息量的实例，并能显著减少所需的注释数量。 |
| [^6] | [Policy Improvement using Language Feedback Models](https://arxiv.org/abs/2402.07876) | 本文介绍了一种使用语言反馈模型（LFMs）改进政策的方法，通过识别期望的行为并进行模仿学习，我们在任务完成率、泛化性能和人类可解释性方面取得了显著改进。 |
| [^7] | [PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs](https://arxiv.org/abs/2402.07872) | 本文介绍了一种名为PIVOT的新颖视觉提示方法，它通过迭代的视觉问答将任务转化为VLMs问题。每个迭代中，图像被标注为VLMs可以参考的视觉表示，并通过优化选择最佳选项。这种方法能够使VLMs进行机器人控制和其他空间任务的输出。 |
| [^8] | [Scaling Laws for Fine-Grained Mixture of Experts](https://arxiv.org/abs/2402.07871) | 本研究分析了细粒度混合专家模型的标度特性，并引入了粒度作为新的超参数，通过调整粒度可以精确控制专家的大小。研究结果显示，MoE模型在效果上始终优于密集变压器模型，并且随着模型大小和训练预算的增大，密集和MoE模型之间的效率差距也在增大。同时，将MoE中专家的大小设置为与前馈层相同的常见做法在几乎任何计算预算下都不是最优的。 |
| [^9] | [Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models](https://arxiv.org/abs/2402.07865) | 本论文探索了视觉条件化语言模型（VLMs）设计的关键空间，并提供了一套标准化评估，同时还研究了预训练的视觉表示和权衡的问题。 |
| [^10] | [AI-Augmented Predictions: LLM Assistants Improve Human Forecasting Accuracy](https://arxiv.org/abs/2402.07862) | 本研究发现，使用LLMs助手可以显著提高预测准确性，不仅仅是由于模型预测准确性的提升。 |
| [^11] | [Lissard: Long and Simple Sequential Reasoning Datasets](https://arxiv.org/abs/2402.07859) | Lissard是一个包含七个任务的基准，用于评估模型处理和生成各种序列长度的能力，需要重复的过程执行。评估结果显示随着序列复杂性增加，所有模型的性能都呈一致下降趋势。 |
| [^12] | [Mercury: An Efficiency Benchmark for LLM Code Synthesis](https://arxiv.org/abs/2402.07844) | Mercury提出了一个针对LLM代码综合任务的效率评估基准，通过引入新的度量标准Beyond@K来衡量归一化的代码效率，从而鼓励生成功能正确且计算效率高的代码。 |
| [^13] | [Do Membership Inference Attacks Work on Large Language Models?](https://arxiv.org/abs/2402.07841) | 这项研究在大规模语言模型上对成员推断攻击进行了评估，发现在大部分设置中，攻击几乎只能比随机猜测稍好，这种糟糕的性能是由于大型数据集和少量训练迭代的组合，以及成员和非成员之间的边界困惑所导致的。 |
| [^14] | [Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) | Aya是一个开放多语言模型，通过指令微调，在101种语言中表现优于其他模型，扩展了多语言评估的技术，并进行了深入研究优化微调组合、数据修剪以及模型的毒性、偏差和安全性。 |
| [^15] | [Injecting Wiktionary to improve token-level contextual representations using contrastive learning](https://arxiv.org/abs/2402.07817) | 本文研究了利用对比学习注入Wiktionary来改善词级上下文表示，并在Word-In-Context（WiC）任务上取得了新的最佳结果。 |
| [^16] | [Retrieval-Augmented Thought Process as Sequential Decision Making](https://arxiv.org/abs/2402.07812) | 检索增强思维过程（RATP）通过多步决策和蒙特卡洛树搜索，以及Q值估计器，解决了大型语言模型在隐私、产生幻觉和处理长文本方面的挑战，并在处理私人数据的问答任务中实现了50%的性能提升。 |
| [^17] | [Multi-Intent Attribute-Aware Text Matching in Searching](https://arxiv.org/abs/2402.07788) | 本研究针对搜索中的文本匹配系统进行了多意图属性感知的研究，提出了通过多意图建模来利用多个属性之间的关系。意图从属性中提取，总结了查询的多样化需求。 |
| [^18] | [TELLER: A Trustworthy Framework for Explainable, Generalizable and Controllable Fake News Detection](https://arxiv.org/abs/2402.07776) | TELLER是一个可信的假新闻检测框架，通过集成认知和决策系统，提供了解释性、可推广性和可控制性。认知系统利用人类专业知识生成逻辑谓词，指导大型语言模型生成可读的逻辑原子。决策系统推导可推广的逻辑规则来聚合这些原子，实现真实和虚假新闻的识别。 |
| [^19] | [Quantitative knowledge retrieval from large language models](https://arxiv.org/abs/2402.07770) | 本文探讨了大型语言模型（LLMs）作为定量知识检索的可行性，以辅助数据分析任务。提出了一个提示工程框架，将LLMs作为科学文献潜在空间的接口。讨论了使用LLMs作为“专家”的影响和挑战。 |
| [^20] | [Text Detoxification as Style Transfer in English and Hindi](https://arxiv.org/abs/2402.07767) | 本文研究了文本解毒化的任务，旨在将有毒文本自动转化为无毒文本。通过知识转移、多任务学习和删除重建方法，我们提出了三种解决方案。我们利用Dementieva等人提供的数据集进行实验，并引入了一个小型的印地语平行数据集用于评估。 |
| [^21] | [Diffusion of Thoughts: Chain-of-Thought Reasoning in Diffusion Language Models](https://arxiv.org/abs/2402.07754) | 本文介绍了一种将扩散模型与思维链推理集成的方法，通过扩散传播推理步骤，提供了更大的灵活性和推理能力。实验证明了该方法在数学问题中的有效性，并展示了自我纠正能力和推理技术的潜力。 |
| [^22] | [Towards Unified Alignment Between Agents, Humans, and Environment](https://arxiv.org/abs/2402.07744) | 本文介绍了统一对齐原则 ($\mathbf{UA}^2$)，旨在实现智能体与人类意图、环境动态和自我约束的统一对齐，提出了引入实际特性进行概念验证研究的方法。 |
| [^23] | [Asking Multimodal Clarifying Questions in Mixed-Initiative Conversational Search](https://arxiv.org/abs/2402.07742) | 本论文提出在混合式主动对话搜索中通过使用多模态信息来改进澄清问题的方法，并设计了一个名为Marto的多模态查询澄清模型。通过收集多模态澄清问题和图像的数据集Melon，并采用基于提示的训练策略，为进一步研究这一任务提供了便利。 |
| [^24] | [Unsupervised Sign Language Translation and Generation](https://arxiv.org/abs/2402.07726) | 本文介绍了一个无监督的手语翻译和生成网络（USLNet），它通过利用大量的单模态数据（文本和视频）学习，而无需平行手语数据。USLNet采用不同模态的反向翻译和重建技术，面对文本和视频序列之间的特征表示差异。通过使用滑动窗口方法，USLNet能够有效对齐不同长度的文本和视频序列。这是第一个实现无监督手语翻译和生成的方法。 |
| [^25] | [LoRA-drop: Efficient LoRA Parameter Pruning based on Output Evaluation](https://arxiv.org/abs/2402.07721) | 本文提出了LoRA-drop方法，通过分析LoRA输出评估参数的重要性，并且保留重要层的LoRA，其余层共享相同参数。实验结果表明LoRA-drop有很好的效果。 |
| [^26] | [OrderBkd: Textual backdoor attack through repositioning](https://arxiv.org/abs/2402.07689) | 本论文提出了一种通过重新定位句子中的两个单词实施文本后门攻击的方法，与已有的攻击方式相比，在攻击成功率、困惑度和与干净样本的语义相似性方面表现更好，并且对ONION防御方法具有鲁棒性。 |
| [^27] | [Auxiliary Tasks to Boost Biaffine Semantic Dependency Parsing](https://arxiv.org/abs/2402.07682) | 本研究提出了一种简单而有效的方法来提高语义依存解析的性能，通过引入辅助任务以增加弧之间的相互依赖关系，该方法在实验中表现出了系统性的性能提升，并且具有良好的可扩展性。 |
| [^28] | [Large Language Models "Ad Referendum": How Good Are They at Machine Translation in the Legal Domain?](https://arxiv.org/abs/2402.07681) | 本研究评估了两种大型语言模型和传统神经机器翻译系统在法律领域的机器翻译质量，结果显示大型语言模型在产生上下文足够且流畅的译文方面表现优异，强调了人工评估方法在评估机器翻译质量中的重要性。 |
| [^29] | [The Sound of Healthcare: Improving Medical Transcription ASR Accuracy with Large Language Models](https://arxiv.org/abs/2402.07658) | 本研究探索了利用大型语言模型（LLMs）提高医学转录中自动语音识别（ASR）系统准确性的潜力，并通过实验比较了零-shot和Chain-of-Thought（CoT）提示技术的有效性。 |
| [^30] | [Detecting the Clinical Features of Difficult-to-Treat Depression using Synthetic Data from Large Language Models](https://arxiv.org/abs/2402.07645) | 本研究开发了基于大型语言模型的工具，使用合成数据和片段提取模型从临床数据中提取关于难治性抑郁症的特征，证明了在真实临床数据中取得了良好的整体性能。 |
| [^31] | [AutoMathText: Autonomous Data Selection with Language Models for Mathematical Texts](https://arxiv.org/abs/2402.07625) | 本论文介绍了一种自主数据选择策略，利用语言模型进行数学文本的自动评估和选择，并通过连续预训练显著提高了数学推理能力。主要创新包括利用元提示语言模型作为验证器，发布了高质量的AutoMathText数据集，并实现了预训练令牌效率的提升。 |
| [^32] | [Anchor-based Large Language Models](https://arxiv.org/abs/2402.07616) | 基于锚点的大型语言模型（AnLLM）通过引入创新的基于锚点的自注意力网络（AnSAN）和基于锚点的推理策略，将序列信息压缩到锚点标记中，减少键/值缓存，提高推理效率。 |
| [^33] | [Step-On-Feet Tuning: Scaling Self-Alignment of LLMs via Bootstrapping](https://arxiv.org/abs/2402.07610) | 本文首次探索了自助引导自对齐对大型语言模型的影响，发现其明显优于单次循环的方法，并通过调整数据训练顺序进一步提升模型性能。 |
| [^34] | [Topic Modeling as Multi-Objective Contrastive Optimization](https://arxiv.org/abs/2402.07577) | 该论文介绍了一种新颖的主题建模方法，通过优化对数似然的证据下界和对比学习目标的加权线性组合，将对比主题建模作为一种多目标优化问题，旨在获得能够捕捉共享语义并克服低级别互信息干扰的主题向量集合。 |
| [^35] | [Show Me How It's Done: The Role of Explanations in Fine-Tuning Language Models](https://arxiv.org/abs/2402.07543) | 本研究证明了使用解释来改进语言模型性能的显著好处，尤其适用于较小的模型，解释的加入使模型能够解决之前无法解决的任务。 |
| [^36] | [PKG API: A Tool for Personal Knowledge Graph Management](https://arxiv.org/abs/2402.07540) | 本文提出了一个完整的个人知识图（PKG）管理解决方案，包括用户界面友好的PKG客户端和面向服务的PKG API，以及基于RDF的PKG词汇表用于表示陈述和访问权限。 |
| [^37] | [MAFIA: Multi-Adapter Fused Inclusive LanguAge Models](https://arxiv.org/abs/2402.07519) | 本文提出了一种名为MAFIA的多适配器融合的包容性语言模型，在多个偏见维度上进行模块化去偏倚，利用结构化知识和大规模生成模型构建了多样化的反事实数据增强，并强调了现有去偏倚方法对多个社会偏见之间的相互作用缺乏考虑。 |
| [^38] | [The Balancing Act: Unmasking and Alleviating ASR Biases in Portuguese](https://arxiv.org/abs/2402.07513) | 本研究通过对Whisper和MMS系统进行全面探索，评估了葡萄牙语中非正式对话语音的自动语音识别（ASR）偏见，并发现采用过采样技术可以缓解这种陈规定型偏见。 |
| [^39] | [T-RAG: Lessons from the LLM Trenches](https://arxiv.org/abs/2402.07483) | T-RAG是一个基于LLM的应用程序，用于私人企业文件问答，它结合了RAG框架和经过微调的开源LLM，并分享了构建和部署过程中的经验教训。 |
| [^40] | [Pushing The Limit of LLM Capacity for Text Classification](https://arxiv.org/abs/2402.07470) | 本论文提出了一个自适应增强框架RGPT，通过反复集成强基学习者，生成一个专用的文本分类LLM。通过实证比较，我们展示了RGPT明显胜过其他方法。 |
| [^41] | [AraSpider: Democratizing Arabic-to-SQL](https://arxiv.org/abs/2402.07448) | AraSpider是首个阿拉伯语版本的Spider数据集，研究表明使用回译策略可以显著提高ChatGPT 3.5和SQLCoder模型在阿拉伯语NLP任务中的性能。 |
| [^42] | [Quality Does Matter: A Detailed Look at the Quality and Utility of Web-Mined Parallel Corpora](https://arxiv.org/abs/2402.07446) | 这项研究详细分析了网络挖掘语料库的质量和实用性，并发现不同语言和数据集之间存在显著的质量差异。同时，我们还展示了某些网络挖掘数据集的最佳部分训练的神经机器翻译模型可以与人工策划的数据集持平。 |
| [^43] | [Intrinsic Task-based Evaluation for Referring Expression Generation](https://arxiv.org/abs/2402.07432) | 该论文提出了一种内在任务驱动的评估方法，用于评估分发表达生成（REG）模型。该方法不仅评估了分发表达的质量，还通过两个元级任务评估了模型的引用成功程度和提出替代方案的能力。 |
| [^44] | [SALAD: Smart AI Language Assistant Daily](https://arxiv.org/abs/2402.07431) | SALAD是一款智能AI语言助手应用，旨在帮助外国人学习日语。它提供了多种学习工具和功能，包括翻译，语音识别，音频翻译，词汇跟踪等，并通过每日翻译帮助提高与母语人士的交流能力。调查结果显示60%的外国人对SALAD提升日语能力有信心。该应用利用大型语言模型和扩散模型促进日本社区的包容性。 |
| [^45] | [D\'olares or Dollars? Unraveling the Bilingual Prowess of Financial LLMs Between Spanish and English](https://arxiv.org/abs/2402.07405) | 这项研究推出了第一个双语框架Tois'on de Oro，用于运用在西班牙语和英语金融领域的大型语言模型（LLMs），通过构建双语指导数据集和评估基准进行了实证研究。研究结果表明现有LLMs在多语言性能上存在差距和偏见，而作者提出的FinMA-ES模型在西班牙语中超越了现有SOTA LLMs的表现。 |
| [^46] | [Can LLMs Produce Faithful Explanations For Fact-checking? Towards Faithful Explainable Fact-Checking via Multi-Agent Debate](https://arxiv.org/abs/2402.07401) | 本研究调查了大型语言模型在事实核查中生成忠实解释的能力，并发现了零-shot提示常常导致不忠实的结果。为了解决这个问题，我们提出了多智能体辩论优化（MADR）框架，通过迭代的优化过程，利用多个大型语言模型作为代理人，从而显著提高了生成解释的忠实性。 |
| [^47] | [Leveraging AI to Advance Science and Computing Education across Africa: Progress, Challenges, and Opportunities](https://arxiv.org/abs/2402.07397) | 这项研究描述了在非洲开发和使用人工智能教育工具的工作，包括SuaCode学习编码应用、AutoGrad自动评分和反馈工具、代码抄袭检测工具以及双语AI教师Kwame。这些工具有助于解决非洲学生在教育中面临的挑战。 |
| [^48] | [Chain-of-Layer: Iteratively Prompting Large Language Models for Taxonomy Induction from Limited Examples](https://arxiv.org/abs/2402.07386) | 本文介绍了一种称为Chain-of-Layer的上下文学习框架，用于从给定的实体集中归纳分类体系。通过引入基于集成的排名过滤器来减少错误，Chain-of-Layer在四个实际基准测试中实现了最先进的性能。 |
| [^49] | [Making Flow-Matching-Based Zero-Shot Text-to-Speech Laugh as You Like](https://arxiv.org/abs/2402.07383) | 本文提出了ELaTE，一种基于流匹配的零样本文本到语音系统，可以根据短音频提示以精确控制笑声时机和表情生成任何说话者的自然笑声。 |
| [^50] | [Assessing Generalization for Subpopulation Representative Modeling via In-Context Learning](https://arxiv.org/abs/2402.07368) | 本研究通过使用2016年和2020年的选举数据，评估了基于大型语言模型的分组代表模型在泛化能力上的表现。研究发现，尽管使用实证数据进行条件设定可以提高整体性能，但上下文学习的益处在不同人口子群组之间差异很大，这对实施分组代表模型的从业人员和决策者构成了挑战。 |
| [^51] | [ODIN: Disentangled Reward Mitigates Hacking in RLHF](https://arxiv.org/abs/2402.07319) | 本研究解决了强化学习中的奖励黑客问题，针对回复长度这一挑战，通过建立可靠的评估协议和改进奖励模型的方法，提出了减轻长度偏差的超参数和技巧，并进行了大规模研究。 |
| [^52] | [HyperBERT: Mixing Hypergraph-Aware Layers with Language Models for Node Classification on Text-Attributed Hypergraphs](https://arxiv.org/abs/2402.07309) | 本文提出了HyperBERT模型，通过在预训练的BERT模型中引入超图感知层，克服了现有方法在节点分类任务上难以捕捉超图结构信息和文本属性的局限性，提高了模型的效果和泛化能力。 |
| [^53] | [Power Transformer Fault Prediction Based on Knowledge Graphs](https://arxiv.org/abs/2402.07283) | 本文提出了一种基于知识图谱和梯度提升决策树的方法，用于学习有限的电力变压器故障数据。实验证明该方法在故障预测准确度上优于传统的人工神经网络和逻辑回归方法。 |
| [^54] | [How do Large Language Models Navigate Conflicts between Honesty and Helpfulness?](https://arxiv.org/abs/2402.07282) | 本文研究了如何在大型语言模型中权衡诚实和帮助性，在实验中发现强化学习改善了诚实和帮助性，而链式思维提示则偏向于帮助性。研究结果还展示了GPT-4 Turbo对对话框架和听众决策背景的敏感性。这些发现揭示了大型语言模型内化的对话价值观，并暗示零-shot提示可以在一定程度上引导这些抽象价值观。 |
| [^55] | [Previously on the Stories: Recap Snippet Identification for Story Reading](https://arxiv.org/abs/2402.07271) | 本研究提出了一项新的任务，即回顾片段识别，在故事阅读中通过回顾之前的重要元素来辅助理解正在进行的情节。该任务对PLMs、LLMs和提出的方法具有挑战性，并需要深入理解片段之间的情节相关性。 |
| [^56] | [Open-ended VQA benchmarking of Vision-Language models by exploiting Classification datasets and their semantic hierarchy](https://arxiv.org/abs/2402.07270) | 该研究通过提出创新的评估方法和基于分类数据集的新型VQA基准，推动了对文本生成的视觉语言模型能力的理解。同时，他们还提出了使用语义层次和自动生成的后续问题来改进对细粒度分类任务上粗糙答案的评估。通过比较不同度量标准，他们在进行人工评估研究的基础上选择了最终的度量标准。 |
| [^57] | [Low-Resource Counterspeech Generation for Indic Languages: The Case of Bengali and Hindi](https://arxiv.org/abs/2402.07262) | 该论文针对低资源语言如孟加拉语和印地语，创建了一个包含5062个虐待言论/对抗言论对的基准数据集，并实现了几种基线模型，以生成适当的对抗言论。观察发现，单语设置的性能最佳，并且通过使用合成转移，语言模型可以在一定程度上生成对抗言论，尤其是当语言属于同一语言家族时，可迁移性更好。 |
| [^58] | [American Sign Language Video to Text Translation](https://arxiv.org/abs/2402.07255) | 这项研究关注美国手语视频到文字的翻译技术，通过复制和改进以往研究，建立了评估模型的方法，发现模型性能受优化器、激活函数和标签平滑的影响，进一步研究旨在改善视觉特征捕捉、增强解码器利用率，并整合预训练解码器，以实现更好的翻译结果。 |
| [^59] | [TransGPT: Multi-modal Generative Pre-trained Transformer for Transportation](https://arxiv.org/abs/2402.07233) | TransGPT是一种面向交通领域的新型多模式生成预训练Transformer，使用单模式和多模式数据进行微调，在交通领域的各种任务中优于基准模型。 |
| [^60] | [Through the Lens of Split Vote: Exploring Disagreement, Difficulty and Calibration in Legal Case Outcome Classification](https://arxiv.org/abs/2402.07214) | 通过研究分割投票，探索律师在处理法律案件结果分类时面临的意见分歧和困难，并在欧洲人权法院收集了法官的投票数据集进行研究。这项研究还评估了模型和人类之间感知困难的一致性以及模型的置信度和人类校准。 |
| [^61] | [Synergizing Spatial Optimization with Large Language Models for Open-Domain Urban Itinerary Planning](https://arxiv.org/abs/2402.07204) | 本文提出了Open-domain Urban Itinerary Planning (OUIP)任务，用于根据用户以自然语言描述的请求直接生成行程，通过结合空间优化和大型语言模型(LLM)，提供个性化的城市行程定制服务。 |
| [^62] | [Prompt Perturbation in Retrieval-Augmented Generation based Large Language Models](https://arxiv.org/abs/2402.07179) | 本文研究了基于检索增强生成的大型语言模型（LLM）中提示扰动的影响，并引入了一种新的优化技术GGPP。通过GGPP，我们可以将LLMs的输出引导到特定的错误答案，并应对提示中的无关上下文。 |
| [^63] | [Natural Language Reinforcement Learning](https://arxiv.org/abs/2402.07157) | 本研究将自然语言表示和强化学习原则相结合，提出了自然语言强化学习（NLRL）框架，解决了强化学习在样本效率低、解释性不足和缺乏监督信号等方面的限制问题，通过实验验证了其有效性和可解释性。 |
| [^64] | [X-LoRA: Mixture of Low-Rank Adapter Experts, a Flexible Framework for Large Language Models with Applications in Protein Mechanics and Design](https://arxiv.org/abs/2402.07148) | X-LoRA是一种灵活的大型语言模型框架，利用低秩适配器专家的混合策略，可以创建精细调整的模型并在蛋白质力学和设计领域应用。该模型利用深层逐层适应的组合来解决特定任务，并受到生物学原理的启发。无需修改底层结构即可应用于任何现有的语言模型。 |
| [^65] | [Generalizing Conversational Dense Retrieval via LLM-Cognition Data Augmentation](https://arxiv.org/abs/2402.07092) | 本文提出了一种通过LLM-认知数据增强的方法来广义对话密集检索。该方法首先生成多级增强对话，捕捉多样的对话环境。其次，通过认知感知过程减少错误生成情况，并通过难度自适应样本筛选器选择具有挑战性的样本。 |
| [^66] | [Speech Rhythm-Based Speaker Embeddings Extraction from Phonemes and Phoneme Duration for Multi-Speaker Speech Synthesis](https://arxiv.org/abs/2402.07085) | 本文提出了一种基于语音韵律的方法，通过从音素和音素持续时间中提取说话人嵌入，模拟目标说话人的个体发音特征。实验证明，该方法可以实现有效的多说话人语音合成。 |
| [^67] | [Using Large Language Models for Student-Code Guided Test Case Generation in Computer Science Education](https://arxiv.org/abs/2402.07081) | 本研究旨在提出一种完全自动化的测试用例生成方法，使用大型语言模型，并证明它们是衡量学生知识水平的良好指标，从而解决手动构建测试用例的劳动密集性和专业知识需求的问题。 |
| [^68] | [Using Large Language Models to Automate and Expedite Reinforcement Learning with Reward Machine](https://arxiv.org/abs/2402.07069) | 这篇论文介绍了一种使用大型语言模型自动生成自动机来编码高级知识，加速强化学习过程的算法，并证明了其在多个任务上的有效性和优越性。 |
| [^69] | [Semi-Supervised Learning for Bilingual Lexicon Induction](https://arxiv.org/abs/2402.07028) | 本论文提出了一个半监督学习方法，将两种语言对应的连续词表示集对齐到一个共同的空间，推断双语词典。该方法利用无监督学习的基础，在学习新语言时，整合已有语言集的知识，通过排序方法实现词典诱导。 |
| [^70] | [Gemini Goes to Med School: Exploring the Capabilities of Multimodal Large Language Models on Medical Challenge Problems & Hallucinations](https://arxiv.org/abs/2402.07023) | 该论文综合评估了开源LLM和谷歌的多模态LLM Gemini 在医学推理、幻觉检测和医学视觉问答任务上的能力。Gemini在诊断准确性方面落后于最先进模型，且易出现幻觉、过度自信和知识盲点。采用提示策略可以提高性能。 |
| [^71] | [A Rational Analysis of the Speech-to-Song Illusion](https://arxiv.org/abs/2402.06992) | 本论文提出了对语音转换成歌曲的幻象的合理分析，将其视为一种统计推断过程，通过分析语料库，还发现了一种纯文本的小说转歌词的幻象，并提供了强有力的证据来支持这一观点。 |
| [^72] | [Event-Keyed Summarization](https://arxiv.org/abs/2402.06973) | 事件关键摘要（EKS）是一种新颖的任务，旨在为特定事件生成上下文化的摘要。我们提出了一个基准数据集MUCSUM，并展示了EKS与传统摘要和结构到文本的比较结果。 |
| [^73] | [Instruct Once, Chat Consistently in Multiple Rounds: An Efficient Tuning Framework for Dialogue](https://arxiv.org/abs/2402.06967) | 本论文提出了一种名为Midi-Tuning的多轮交互对话调整框架，通过分别对代理人和用户建模，并利用轮次级内存缓存机制进行调整，实现了对话代理的一致性和稳定性。 |
| [^74] | [NLP for Knowledge Discovery and Information Extraction from Energetics Corpora](https://arxiv.org/abs/2402.06964) | 这项研究展示了NLP在能源材料研究中的实用性，通过将NLP方法应用于能源文本，可以自动发现知识和提取信息。该研究使用三个成熟的NLP模型，并证明它们能够识别能源话题和概念，并生成与专家知识一致的语言模型。此外，研究还提出了一个能源文本的分类流程，其准确率高达59-76\%，最佳模型与专家间一致度相当。 |
| [^75] | [SpeechCLIP+: Self-supervised multi-task representation learning for speech via CLIP and speech-image data](https://arxiv.org/abs/2402.06959) | SpeechCLIP+通过应用CIF模块替换CLIP架构中的CLS令牌，并提出了一种混合架构，实现了在语音关键词提取和图像-语音检索任务中的性能提升。 |
| [^76] | [OpenFedLLM: Training Large Language Models on Decentralized Private Data via Federated Learning](https://arxiv.org/abs/2402.06954) | OpenFedLLM是一个简洁、集成、研究友好的框架/代码库，通过联邦学习在分散的私有数据上实现了大规模语言模型的协作和隐私保护训练，解决了公开数据枯竭的问题。 |
| [^77] | [Should I try multiple optimizers when fine-tuning pre-trained Transformers for NLP tasks? Should I tune their hyperparameters?](https://arxiv.org/abs/2402.06948) | 在微调预训练的Transformer进行NLP任务时，调整优化器的超参数并不会对测试性能产生实质性差异，只调整学习率通常就足够。 |
| [^78] | [LiFi: Lightweight Controlled Text Generation with Fine-Grained Control Codes](https://arxiv.org/abs/2402.06930) | LIFI是一种轻量级的可控文本生成方法，使用精细控制代码实现更精确的控制。通过连续、相对和非排他的控制代码的引导，LIFI可以在训练中学习可控文本生成。使用属性分类器自动导出的精细代码提供了更广泛的监督信号。与此同时，通过与适配器相结合，LIFI实现了高效的控制。 |
| [^79] | [A Thorough Examination of Decoding Methods in the Era of LLMs](https://arxiv.org/abs/2402.06925) | 在LLMs的背景下，本文综合研究了各种解码方法的性能、鲁棒性和解码速度，并发现解码方法的性能与任务相关，受到对齐、模型大小和量化等因素影响；某些方法可以通过大量超参数调整达到更好的性能，但需要权衡取舍。 |
| [^80] | [Generating Chain-of-Thoughts with a Direct Pairwise-Comparison Approach to Searching for the Most Promising Intermediate Thought](https://arxiv.org/abs/2402.06918) | 本文提出了一种基于直接两两比较的方法，通过利用LLMs的噪声反馈，直接识别出最有潜力的中间思维，从而生成优秀的思维链。 |
| [^81] | [TL;DR Progress: Multi-faceted Literature Exploration in Text Summarization](https://arxiv.org/abs/2402.06913) | 本文介绍了一种名为TL;DR Progress的新工具，用于探索神经文本摘要领域的文献。该工具根据全面的注释方案对514篇论文进行了组织，实现了细粒度的、多方面的检索，并为每篇论文提供了简洁的摘要。 |
| [^82] | [Investigating Consistency in Query-Based Meeting Summarization: A Comparative Study of Different Embedding Methods](https://arxiv.org/abs/2402.06907) | 本研究比较了不同嵌入方法在查询导向的会议摘要中的一致性，旨在解决复杂、多主题和多人参与的重要会议记录中的信息查找问题。 |
| [^83] | [Can LLMs Recognize Toxicity? Structured Toxicity Investigation Framework and Semantic-Based Metric](https://arxiv.org/abs/2402.06900) | 本研究提出了一种基于大型语言模型（LLMs）的自动度量方法，用于识别生成文本中的毒性。通过分析毒性因素和LLMs的内在毒性属性，该方法在测量毒性方面表现出众，比现有指标提升12个百分点。 |
| [^84] | [GenTranslate: Large Language Models are Generative Multilingual Speech and Machine Translators](https://arxiv.org/abs/2402.06894) | GenTranslate是一个新的翻译任务生成模型，通过利用大型语言模型的丰富语言知识和强大推理能力，可以从N-best列表中生成更高质量的翻译结果。 |
| [^85] | [History, Development, and Principles of Large Language Models-An Introductory Survey](https://arxiv.org/abs/2402.06853) | 这项综述性调查介绍了大型语言模型（LLMs）的历史、发展和原理，旨在帮助广泛的读者群体理解这些模型的背景和原理。 |
| [^86] | [ChemLLM: A Chemical Large Language Model](https://arxiv.org/abs/2402.06852) | ChemLLM是第一个专门用于化学领域的大型语言模型，利用新颖的指令构建方法将结构化知识转化为对话形式，具有平滑对话交互的能力，并在化学的三个主要任务中击败了GPT-3.5。 |
| [^87] | [Debating with More Persuasive LLMs Leads to More Truthful Answers](https://arxiv.org/abs/2402.06782) | 本文研究了更弱的语言模型是否能评估更强的模型的正确性。研究发现，通过进行辩论，非专家模型和人类回答问题的准确性都有所提高。 |
| [^88] | [Evaluation Metrics for Text Data Augmentation in NLP](https://arxiv.org/abs/2402.06766) | 这项研究提供了文本增强方法的评估指标分类法，为统一基准提供方向和帮助。在不同任务、度量标准和实验设置下，该分类法有助于比较不同的增强方法。 |
| [^89] | [EntGPT: Linking Generative Large Language Models with Knowledge Bases](https://arxiv.org/abs/2402.06738) | 本文介绍了一种名为EntGPT的模型，通过Entity Disambiguation（ED）任务，连接了生成型大型语言模型与知识库。通过提示工程和指令调整，该模型在没有有监督微调的情况下，显著提高了LLMs的性能，并在实体消歧任务上取得了可比较的性能。 |
| [^90] | [NICE: To Optimize In-Context Examples or Not?](https://arxiv.org/abs/2402.06733) | 通过研究在提供任务特定指令的情况下是否需要优化上下文示例，我们挑战了对于指导性LLMs的共识，并发现在某些任务中，不同的优化上下文示例方法会产生递减的回报。我们引入了"度量标准"，用于衡量从给定指令中学习任务的能力，并提供了一个启发式方法，帮助决定是否优化指令还是ICE用于任何新任务。 |
| [^91] | [Neural Models for Source Code Synthesis and Completion](https://arxiv.org/abs/2402.06690) | 本论文提出了一种基于序列到序列的深度学习模型和训练方法，用于将自然语言转化为可编译的代码片段，并为开发人员提供源代码建议和自动补全功能。这一方法能够提取开发者的编码意图并准确推断类型、名称和上下文等信息。 |
| [^92] | [The Essential Role of Causality in Foundation World Models for Embodied AI](https://arxiv.org/abs/2402.06665) | 基于因果关系的基础世界模型对于具身人工智能的发展至关重要，当前的基础模型无法准确建模与现实世界的物理相互作用。因果关系的研究有助于构建真实世界模型，提高对可能相互作用结果的准确预测能力。 |
| [^93] | [Adversarial Text Purification: A Large Language Model Approach for Defense](https://arxiv.org/abs/2402.06655) | 本文研究了防御文本分类器中对抗性净化方法的有效性，并提出了一种基于大型语言模型加以净化的方法。 |
| [^94] | [SocraSynth: Multi-LLM Reasoning with Conditional Statistics](https://arxiv.org/abs/2402.06634) | SocraSynth是一个多语言模型推理平台，通过使用条件统计和系统化的语境增强技术，以及可调节的辩论争议程度，解决了大型语言模型(LLMs)面临的偏见、幻觉和推理能力不足等问题。 |
| [^95] | [LightCAM: A Fast and Light Implementation of Context-Aware Masking based D-Tdnn for Speaker Verification](https://arxiv.org/abs/2402.06073) | LightCAM是一种快速轻量级的基于上下文感知屏蔽的D-TDNN说话人验证实现，通过采用深度可分离卷积模块和多尺度特征聚合，它在VoxCeleb数据集上取得了更好的性能。 |
| [^96] | [Generative Echo Chamber? Effects of LLM-Powered Search Systems on Diverse Information Seeking](https://arxiv.org/abs/2402.05880) | LLM驱动的对话式搜索系统增加了选择性曝光，且支持用户观点的有偏见LLM加剧了这种偏差。 |
| [^97] | [PromptCrypt: Prompt Encryption for Secure Communication with Large Language Models](https://arxiv.org/abs/2402.05868) | PromptCrypt是一种使用表情符号对用户输入进行加密的机制，保护了大型语言模型（LLM）中用户的隐私，防止数据泄露和解密。 |
| [^98] | [Text-to-Code Generation with Modality-relative Pre-training](https://arxiv.org/abs/2402.05783) | 本论文研究了如何根据不同的模态调整和表示序列标记，以进一步提高文本到代码生成的效果。 |
| [^99] | [Everybody Prune Now: Structured Pruning of LLMs with only Forward Passes](https://arxiv.org/abs/2402.05406) | 本文提出了一种仅使用前向传递的LLM结构化修剪方法，通过Bonsai生成的修剪模型在性能上优于梯度-based结构化修剪方法，并且速度是半结构化修剪模型的两倍。 |
| [^100] | [In-Context Principle Learning from Mistakes](https://arxiv.org/abs/2402.05403) | 本文提出了一种新的学习方法LEAP，通过让模型从少量输入-输出示例中犯错误，然后反思并学习准则，从而提升模型在各种任务上的表现。 |
| [^101] | [ApiQ: Finetuning of 2-Bit Quantized Large Language Model](https://arxiv.org/abs/2402.05147) | 这项工作介绍了一种名为ApiQ的新型量化框架，通过同时初始化LoRA组件和量化大型语言模型的权重，恢复量化过程中丢失的信息，维持原始模型的激活精度并减轻误差传播。 |
| [^102] | [Financial Report Chunking for Effective Retrieval Augmented Generation](https://arxiv.org/abs/2402.05131) | 本文提出了一种扩展的方法来切块财务报告，通过根据文档的结构元素组件进行切块，从而实现更有效的检索增强生成。这种方法可以优化切块大小，而无需调整，并提供了对整体上下文和准确性的评估以及对问答任务性能的影响。 |
| [^103] | [Quantifying Similarity: Text-Mining Approaches to Evaluate ChatGPT and Google Bard Content in Relation to BioMedical Literature](https://arxiv.org/abs/2402.05116) | 本研究旨在通过文本挖掘方法评估ChatGPT和Google Bard生成的内容与生物医学文献之间的相似性。实验结果显示，在余弦文档相似性方面，ChatGPT表现优于Google Bard。 |
| [^104] | [Large Language Models As MOOCs Graders](https://arxiv.org/abs/2402.03776) | 该研究探索了利用大型语言模型（LLMs）代替MOOCs中同伴评分的可行性，旨在解决大规模在线开放课程中评估学生写作任务的问题。 |
| [^105] | [BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation](https://arxiv.org/abs/2402.03216) | BGE M3-嵌入是一种新的多语言、多功能和多粒度的文本嵌入模型，支持超过100种工作语言，并在多语言和跨语言检索任务上取得了最先进的性能。它能够同时执行密集检索、多向量检索和稀疏检索，并能处理不同粒度的输入。其有效训练包括了一种自知识蒸馏方法和优化的批处理策略。 |
| [^106] | [Evading Data Contamination Detection for Language Models is (too) Easy](https://arxiv.org/abs/2402.02823) | 本研究指出语言模型数据污染的检测方法在面对恶意模型提供者的有意污染时存在漏洞，并提出了一种简单而有效的污染技术（EAL）来显著提高基准测试性能且逃避当前的检测方法。 |
| [^107] | [VIALM: A Survey and Benchmark of Visually Impaired Assistance with Large Models](https://arxiv.org/abs/2402.01735) | 这项研究调查了具有大型模型的视觉障碍辅助，并通过基准实验评估了模型的能力，进一步推动了视觉障碍辅助技术的发展。 |
| [^108] | [Prompting Large Language Models for Zero-Shot Clinical Prediction with Structured Longitudinal Electronic Health Record Data](https://arxiv.org/abs/2402.01713) | 本研究探索了将大型语言模型（LLMs）应用于结构化纵向电子健康记录（EHR）数据的可行性，并着重研究了其零样本能力。通过考虑EHR特征和临床上下文，我们的方法在MIMIC-IV和TJH数据集上取得了良好的实验结果。 |
| [^109] | [Health-LLM: Personalized Retrieval-Augmented Disease Prediction Model](https://arxiv.org/abs/2402.00746) | 提出了一个创新的框架，健康-LLM，通过大规模特征提取和医学知识权衡评分，实现了个性化的检索增强疾病预测模型。这种方法通过整合健康报告，调整特征权重，以及利用语言模型和专家见解提高预测准确性，与传统健康管理方法相比具有明显优势。 |
| [^110] | [ProLex: A Benchmark for Language Proficiency-oriented Lexical Substitution](https://arxiv.org/abs/2401.11356) | ProLex是一个以语言熟练度为导向的词汇替换的评估基准，旨在评估生成适当替代词和表现更好语言熟练度的系统能力。使用微调任务特定合成数据的Llama2-13B模型在F分数上优于ChatGPT 3.2%，与GPT-4在ProLex上表现相当。 |
| [^111] | [Using Zero-shot Prompting in the Automatic Creation and Expansion of Topic Taxonomies for Tagging Retail Banking Transactions](https://arxiv.org/abs/2401.06790) | 这项工作提出了使用无监督方法和零样本提示来自动构建和扩展主题分类法的研究。通过应用主题建模和关键词提取技术，结合基于指令的微调LLMs，在零售银行数据集中为商家分配标签，具有超过90%的一致性率和令人兴奋的结果。 |
| [^112] | [Fortify the Shortest Stave in Attention: Enhancing Context Awareness of Large Language Models for Effective Tool Use](https://arxiv.org/abs/2312.04455) | 本文证明了大型语言模型中关注分配的波形模式对其在需要高度上下文意识的任务中的性能有显著影响。我们提出了一种名为“Attention Buckets”的推理方法，通过多个并行过程和不同的旋转位置嵌入角度，增强了模型对不同上下文位置的意识，从而减轻了忽视关键信息的风险。 |
| [^113] | [Large language models can enhance persuasion through linguistic feature alignment](https://arxiv.org/abs/2311.16466) | 本研究调查了大型语言模型对人类沟通的影响，使用了消费者金融投诉数据，并发现大型语言模型的使用可能增强了一整套语言特征，提高了信息说服力。 |
| [^114] | [Detection of developmental language disorder in Cypriot Greek children using a neural network algorithm](https://arxiv.org/abs/2311.15054) | 该研究开发了一种使用神经网络算法进行发展性语言障碍（DLD）检测的自动化方法，并首次应用于塞浦路斯希腊儿童DLD人群。实验结果表明该方法具有高的分类效果。 |
| [^115] | [Universal Jailbreak Backdoors from Poisoned Human Feedback](https://arxiv.org/abs/2311.14455) | 本文提出了一种新的威胁，攻击者通过毒害训练数据向语言模型中嵌入“越狱后门”，并通过添加特定的触发词使模型产生有害的回应。这种通用越狱后门比之前的研究更强大，且较难被察觉。研究探究了RLHF设计中的决策对其鲁棒性的影响，并发布了一组被毒害模型的基准测试数据，以促进未来对通用越狱后门的研究。 |
| [^116] | [PLUG: Leveraging Pivot Language in Cross-Lingual Instruction Tuning](https://arxiv.org/abs/2311.08711) | 提出了一种利用枢纽语言进行指令调优的方法，该方法在低资源语言中取得了显著改进，并通过引入一个基准数据集进行了评估。 |
| [^117] | [On Measuring Faithfulness or Self-consistency of Natural Language Explanations](https://arxiv.org/abs/2311.07466) | 本文论述了衡量自然语言解释的忠诚度或自一致性的问题。我们提出了自一致性测试来评估解释的输出级别的一致性。我们通过构建比较一致性测试库，并引入了新的自一致性度量CC-SHAP来支持我们的观点。 |
| [^118] | [Data Contamination Quiz: A Tool to Detect and Estimate Contamination in Large Language Models](https://arxiv.org/abs/2311.06233) | 这个工具使用数据污染问题（DCQ）的方法来检测和估计大型语言模型中的数据污染。在DCQ中，我们创建了每个数据集实例的扰动版本，并让语言模型从中选择原始实例，通过词级扰动来区分选项。这种方法利用了语言模型在预训练阶段暴露于原始实例时的固有特性。 |
| [^119] | [PeTailor: Improving Large Language Model by Tailored Chunk Scorer in Biomedical Triple Extraction](https://arxiv.org/abs/2310.18463) | 我们提出了PeTailor，这是一个基于检索的框架，通过使用定制的分块评分器从预先构建的分块数据库中检索相关文档，并将检索到的信息集成到大型语言模型（LLM）的输入中，以改进生物医学三元组提取的效果。 |
| [^120] | [SQLformer: Deep Auto-Regressive Query Graph Generation for Text-to-SQL Translation](https://arxiv.org/abs/2310.18376) | SQLformer是一个用于文本到SQL翻译的深度自回归查询图生成模型，采用了特定的Transformer架构，并通过结构归纳偏差解决领域泛化和自然语言与SQL查询对齐的难题。 |
| [^121] | [Federated Learning of Large Language Models with Parameter-Efficient Prompt Tuning and Adaptive Optimization](https://arxiv.org/abs/2310.15080) | 本文提出了一种带有参数高效prompt调整和自适应优化的联邦学习方法，以实现大型语言模型的高效和有效训练。 |
| [^122] | [From Chaos to Clarity: Claim Normalization to Empower Fact-Checking](https://arxiv.org/abs/2310.14338) | 本研究提出了声明标准化任务，通过使用CACN模型利用思维链和声明检查来从复杂的社交媒体帖子中提取简化的声明，以加强事实核查。 |
| [^123] | [An In-Context Schema Understanding Method for Knowledge Base Question Answering](https://arxiv.org/abs/2310.14174) | 本文提出了一种基于上下文的模式理解方法（ICSU），通过提供模式相关的注释示例实现了大型语言模型（LLM）直接理解模式的能力。实验结果表明... |
| [^124] | [Utilizing Contextual Clues and Role Correlations for Enhancing Document-level Event Argument Extraction](https://arxiv.org/abs/2310.05116) | 本文提出了CARLG模型，通过利用上下文线索和角色相关性，提升了文档级事件论证提取的性能。 |
| [^125] | [Faithful Knowledge Graph Explanations for Commonsense Reasoning](https://arxiv.org/abs/2310.04910) | 本论文提出了两个量化指标来衡量基于知识图谱的解释的可信性，并引入了一种新的训练方法来改善解释的可信度。实验结果表明该方法可以提高解释的一致性和保真度。 |
| [^126] | [Tool-Augmented Reward Modeling](https://arxiv.org/abs/2310.01045) | 本文提出了一种工具增强的偏好建模方法，通过赋予奖励模型访问外部环境的能力，解决了传统奖励模型在基本功能上的限制，同时提高了解释能力和评分可靠性。 |
| [^127] | [CrossLingR: A Comprehensive Multilingual Receipt Dataset for Cross-Language Information Extraction and Classification](https://arxiv.org/abs/2309.09800) | 本研究介绍了一个全面的多语言数据集CrossLingR，用于推动收据信息提取和物品分类的进展。我们的数据集包含了47,720个标注样本，详细记录了项目名称、相关属性和44个不同的产品类别。通过InstructLLaMA方法论，我们展示了在关键信息提取和物品分类任务中的显著效果。相关资源可在https://github.com/Update-For-Integrated-Business-AI/AMuRD上获取。 |
| [^128] | [A Model for Every User and Budget: Label-Free and Personalized Mixed-Precision Quantization](https://arxiv.org/abs/2307.12659) | 该论文提出了一种无标签和个性化混合精度量化方法，可以根据用户需求和内存预算生成个性化的量化方案，针对大规模ASR模型，能够提高特定性别、语言和说话者的性能。 |
| [^129] | [KDSTM: Neural Semi-supervised Topic Modeling with Knowledge Distillation](https://arxiv.org/abs/2307.01878) | KDSTM是一种使用知识蒸馏的神经半监督主题建模方法，对于文本分类任务，在没有预训练嵌入且资源受限的情况下，能够提供高准确性、鲁棒性和效率。 |
| [^130] | [Data-Driven Information Extraction and Enrichment of Molecular Profiling Data for Cancer Cell Lines](https://arxiv.org/abs/2307.00933) | 这项研究设计了一个新颖的数据提取和探索系统，通过从科学文献中提取深层语义关系，丰富癌细胞系领域的结构化临床数据。 |
| [^131] | [Making Language Models Better Tool Learners with Execution Feedback](https://arxiv.org/abs/2305.13068) | 这篇论文提出了一个名为TRICE的框架，通过执行反馈实现语言模型的工具学习，使其能够学会何时以及如何有效地使用工具。 |
| [^132] | [Efficient and Flexible Topic Modeling using Pretrained Embeddings and Bag of Sentences](https://arxiv.org/abs/2302.03106) | 本文提出了一种使用预训练的嵌入和句子包进行高效灵活的主题建模方法，通过结合生成过程模型和聚类，提供了使用先验自定义主题-文档分布的可能性，实验表明该方法在计算负担较小的情况下取得了最新的结果。 |
| [^133] | [Retrieval-based Disentangled Representation Learning with Natural Language Supervision](https://arxiv.org/abs/2212.07699) | 本研究提出了基于检索的带自然语言监督的解缠表示学习框架，利用自然语言作为数据变化的代理，通过词汇空间中的双编码器模型实现对数据内在特征的解缠表示学习。 |
| [^134] | [What Artificial Neural Networks Can Tell Us About Human Language Acquisition](https://arxiv.org/abs/2208.07998) | 机器学习在自然语言处理方面的快速发展潜在地改变了我们对人类语言习得的认识，但目前的人工学习者和人类在学习环境和数据偏好上存在差异。为了增加计算模型学习结果的相关性，需要训练出没有显著优势的模型学习者，以提供概念证明和进行实验干预。 |
| [^135] | [PROXYQA: An Alternative Framework for Evaluating Long-Form Text Generation with Large Language Models.](http://arxiv.org/abs/2401.15042) | PROXYQA是一个用于评估大型语言模型长篇文本生成的替代框架，通过生成详尽的内容，并利用评估器和生成内容作为背景环境，根据评估器回答代理问题的表现来评估生成内容的质量。 |
| [^136] | [Commonsense-augmented Memory Construction and Management in Long-term Conversations via Context-aware Persona Refinement.](http://arxiv.org/abs/2401.14215) | 本文提出了一个旨在解决长期对话中角色句子不具信息性的问题的框架，通过利用常识增强的角色扩展，并设计策略将相互矛盾的角色转化为包含丰富说话者信息的句子，以提高回应生成质量。 |
| [^137] | [All in How You Ask for It: Simple Black-Box Method for Jailbreak Attacks.](http://arxiv.org/abs/2401.09798) | 本研究提出了一种简单的黑盒方法，用于生成越狱攻击提示，克服了现有方法的复杂性和计算成本的限制。该方法通过使用语言模型自身，将有害提示重写为非有害表达，实现了超过80%的攻击成功率，并且即使模型更新，效果仍然有效。 |
| [^138] | [Learning Shortcuts: On the Misleading Promise of NLU in Language Models.](http://arxiv.org/abs/2401.09615) | 该论文调查了大型语言模型在自然语言理解任务中使用捷径学习的现象，强调了这种现象对语言模型评估的影响，并呼吁加大对捷径学习的研究力度以提升语言模型的鲁棒性和实际场景中的自然语言理解评估标准。 |
| [^139] | [Augmenting Math Word Problems via Iterative Question Composing.](http://arxiv.org/abs/2401.09003) | 本研究通过引入MMIQC数据集和迭代组合问题(IQC)的新颖增强方法，成功提高了大型语言模型的数学推理能力，在竞赛级数学问题上取得了优于先前最佳结果的准确率。 |
| [^140] | [InfiAgent-DABench: Evaluating Agents on Data Analysis Tasks.](http://arxiv.org/abs/2401.05507) | InfiAgent-DABench是第一个评估基于LLM的代理在数据分析任务中的基准测试，包括DAEval数据集和代理框架。对23个最先进的LLMs进行的基准测试揭示了当前数据分析任务中的挑战。 |
| [^141] | [Beyond Extraction: Contextualising Tabular Data for Efficient Summarisation by Language Models.](http://arxiv.org/abs/2401.02333) | 本研究提出了一种创新的方法，通过上下文化表格数据来提高 RAG 系统中处理复杂表格查询的准确性，提高了摘要的效率。 |
| [^142] | [NPHardEval: Dynamic Benchmark on Reasoning Ability of Large Language Models via Complexity Classes.](http://arxiv.org/abs/2312.14890) | NPHardEval是一个新的基准，旨在评估大型语言模型在900个算法问题上的推理能力，扩展到NP-Hard复杂性类别。 |
| [^143] | [MultiGPrompt for Multi-Task Pre-Training and Prompting on Graphs.](http://arxiv.org/abs/2312.03731) | 本文提出了一种名为MultiGPrompt的多任务预训练和提示框架，用于在图形表示学习中提高鲁棒性和减少标注成本。 |
| [^144] | [ViCrop: Perceiving Small Visual Details in Zero-shot Visual Question Answering with Multimodal Large Language Models.](http://arxiv.org/abs/2310.16033) | 本研究探讨了多模态大型语言模型在零样本视觉问答中感知细小视觉细节的能力。实验表明，这些模型对于与问题相关的视觉主题的尺寸非常敏感，通过引入人类可视剪裁可以显著提升其准确性。 |
| [^145] | [Zero-Shot Refinement of Buildings' Segmentation Models using SAM.](http://arxiv.org/abs/2310.01845) | 本文提出了一种使用SAM进行建筑物分割模型的零-shot细化的方法，针对遥感图像应用中SAM性能不佳、无法进行识别的问题进行了处理。通过引入不同的提示来提升模型的泛化能力。 |
| [^146] | [Distributional Inclusion Hypothesis and Quantifications: Probing Hypernymy in Functional Distributional Semantics.](http://arxiv.org/abs/2309.08325) | 本文研究了在功能分布语义中，当语料库严格遵循分布包含假设时，功能分布语义模型可以学习到上位词关系。同时，引入一种训练目标使得模型可以处理普遍量化，从而在分布包含假设的反向下实现上位词关系的学习。实验结果验证了这些假设和目标的有效性。 |
| [^147] | [Certifying LLM Safety against Adversarial Prompting.](http://arxiv.org/abs/2309.02705) | 本研究提出了首个具有可验证安全保证的框架——消除和检查，用于对抗敌对提示。通过逐个消除标记并使用安全过滤器检查生成的子序列，确保任何敌对修改的有害输入提示都能被正确标识为有害。 |
| [^148] | [Generator-Retriever-Generator: A Novel Approach to Open-domain Question Answering.](http://arxiv.org/abs/2307.11278) | 生成器-检索器-生成器（GRG）是一种新方法，将文档检索技术与大型语言模型相结合，以生成开放域问答的准确和信息丰富的答案。 |
| [^149] | [Soft Prompt Tuning for Augmenting Dense Retrieval with Large Language Models.](http://arxiv.org/abs/2307.08303) | 本论文提出了一种使用软提示调优来增强密集检索的方法（SPTAR）。通过优化任务特定的软提示并利用大型语言模型为未标记的文档生成弱查询，可以提高零样本和少样本的密集检索模型的性能。 |
| [^150] | [Decoding the Popularity of TV Series: A Network Analysis Perspective.](http://arxiv.org/abs/2307.05329) | 从电视剧的角色网络中提取网络指标，研究发现对电视剧的评论分数具有很强的相关性，为电视制片人提供了定量信息，帮助他们调整角色动态以吸引观众。 |
| [^151] | [Dynamic Masking Rate Schedules for MLM Pretraining.](http://arxiv.org/abs/2305.15096) | 本论文提出了一种动态调度掩码率的方法来改进MLM预训练的质量，通过线性降低掩码率，达到了对BERT-base和BERT-large模型分别提高0.46%和0.25%的平均GLUE准确率的效果。这种方法不仅加快了BERT-base的预训练速度，还实现了对BERT-large的帕累托改善。 |
| [^152] | [DAPR: A Benchmark on Document-Aware Passage Retrieval.](http://arxiv.org/abs/2305.13915) | DAPR是一个文档感知段落检索的基准测试，挑战在于如何从长文档中找到正确的段落并返回准确结果。 |
| [^153] | [Democratized Diffusion Language Model.](http://arxiv.org/abs/2305.10818) | 本文提出了一个基于CDCD框架的民主扩散语言模型（DDLM），并通过GLUE基准测试了其知识转移能力，为研究人员提供了DDLM训练和评估流程以及已训练的DDLM模型。 |
| [^154] | [Smaller Language Models are Better Black-box Machine-Generated Text Detectors.](http://arxiv.org/abs/2305.09859) | 本文研究发现，小型语言模型更适用于作为通用文本检测器，可以更加精确地检测出机器生成的文本，而检测器和生成模型是否具有相同的架构或语料库并不会对检测性能产生显著影响。 |

# 详细

[^1]: 跳过$\textbackslash n$: 一种简单的方法减少大规模视觉-语言模型中的幻觉

    Skip $\textbackslash n$: A simple method to reduce hallucination in Large Vision-Language Models

    [https://rss.arxiv.org/abs/2402.01345](https://rss.arxiv.org/abs/2402.01345)

    本文提出了一种新的视角，指出LVLMs中固有的偏见可能是多模态幻觉的关键因素。通过系统识别与段落分割符相关的语义漂移偏差，我们发现模型在训练数据中经常遇到明显的内容语义变化，导致幻觉的产生。

    

    最近大规模视觉-语言模型（LVLMs）的进展展示了其在视觉信息理解与人类语言方面的令人印象深刻的能力。尽管取得了这些进展，LVLMs仍然面临多模态幻觉的挑战，例如生成与视觉信息中不存在的对象相关的文本描述。然而，多模态幻觉的根本原因仍然未被充分探索。在本文中，我们提出了一个新的视角，认为LVLMs中固有的偏见可能是幻觉的关键因素。具体而言，我们系统地确定了与段落分割符（'$\textbackslash n\textbackslash n$'）相关的语义漂移偏差，即在训练数据中，在“$\textbackslash n\textbackslash n$”之前和之后的内容经常表现出显著的语义改变。这种模式使得模型推断在“$\textbackslash n\textbackslash n$”之后的内容应明显不同于前面的内容。

    Recent advancements in large vision-language models (LVLMs) have demonstrated impressive capability in visual information understanding with human language. Despite these advances, LVLMs still face challenges with multimodal hallucination, such as generating text descriptions of objects that are not present in the visual information. However, the underlying fundamental reasons of multimodal hallucinations remain poorly explored. In this paper, we propose a new perspective, suggesting that the inherent biases in LVLMs might be a key factor in hallucinations. Specifically, we systematically identify a semantic shift bias related to paragraph breaks ('$\textbackslash n\textbackslash n$'), where the content before and after '$\textbackslash n\textbackslash n$' in the training data frequently exhibit significant semantic changes. This pattern leads the model to infer that the contents following '$\textbackslash n\textbackslash n$' should be obviously different from the preceding contents wi
    
[^2]: 关于LM潜在空间的语义学：一种以词汇为定义的方法

    On the Semantics of LM Latent Space: A Vocabulary-defined Approach

    [https://rss.arxiv.org/abs/2401.16184](https://rss.arxiv.org/abs/2401.16184)

    本论文介绍了一种以词汇为定义的语义学方法，建立了LM潜在空间的参考框架，确保基于LM词汇的分离语义分析。在LM适应过程中，引入了计算logits的新技术和神经聚类模块，通过实验证明了该方法在文本理解上的优越性能。

    

    理解语言模型(LM)的潜在空间对于改进其性能和可解释性至关重要。现有的分析往往在提供基于模型的对LM语义的分离洞察方面存在不足，并忽视了LM适应的重要方面。为了响应这一问题，我们引入了一种开创性的方法，称为以词汇为定义的语义学，它在LM的潜在空间中建立了一个参考框架，确保基于LM词汇的分离语义分析。我们的方法超越了先前的交织分析，利用LM词汇来获得以模型为中心的洞察。此外，我们提出了一种计算logits的新技术，强调可微分性和局部等距性，并引入了一个神经聚类模块，用于在LM适应过程中进行语义校准。通过在多种文本理解数据集上进行广泛实验，我们的方法在检索增强生成和参数高效微调方面超越了最先进的方法。

    Understanding the latent space of language models (LM) is crucial to refining their performance and interpretability. Existing analyses often fall short in providing disentangled (model-centric) insights into LM semantics, and neglect essential aspects of LM adaption. In response, we introduce a pioneering method called vocabulary-defined semantics, which establishes a reference frame within the LM latent space, ensuring disentangled semantic analysis grounded in LM vocabulary. Our approach transcends prior entangled analysis, leveraging LM vocabulary for model-centric insights. Furthermore, we propose a novel technique to compute logits, emphasising differentiability and local isotropy, and introduce a neural clustering module for semantically calibrating data representations during LM adaptation. Through extensive experiments across diverse text understanding datasets, our approach outperforms state-of-the-art methods of retrieval-augmented generation and parameter-efficient finetuni
    
[^3]: 从单一儿童语言输入的可学习性的系统调查

    A systematic investigation of learnability from single child linguistic input

    [https://arxiv.org/abs/2402.07899](https://arxiv.org/abs/2402.07899)

    我们的研究探索了用单个儿童的语言输入训练语言模型的可学习性，我们发现这种设置下的语言模型能够形成句法和语义词群，并对某些语言现象具有敏感性。

    

    语言模型（LM）在生成语言连贯文本方面表现出了 remarkable proficiency，引发了关于它们与人类语言可学习性的相关讨论。然而，这些模型的训练数据与儿童接收到的语言输入之间存在着显著差距。LMs通常在数量级上更大且本质与儿童语言输入不同的数据上进行训练。针对这一差距，我们的研究侧重于在单个儿童语言输入的子集上训练LMs。先前的研究发现，在这种设置下训练的LMs可以形成句法和语义词群，并对某些语言现象具有敏感性。然而，这些研究仅考虑了仅使用一个单一儿童数据集训练的LSTMs和更简单的神经网络。为了检验从单一儿童输入可学习性的鲁棒性，我们系统地…

    Language models (LMs) have demonstrated remarkable proficiency in generating linguistically coherent text, sparking discussions about their relevance to understanding human language learnability. However, a significant gap exists between the training data for these models and the linguistic input a child receives. LMs are typically trained on data that is orders of magnitude larger and fundamentally different from child-directed speech (Warstadt and Bowman, 2022; Warstadt et al., 2023; Frank, 2023a). Addressing this discrepancy, our research focuses on training LMs on subsets of a single child's linguistic input. Previously, Wang, Vong, Kim, and Lake (2023) found that LMs trained in this setting can form syntactic and semantic word clusters and develop sensitivity to certain linguistic phenomena, but they only considered LSTMs and simpler neural networks trained from just one single-child dataset. Here, to examine the robustness of learnability from single-child input, we systematicall
    
[^4]: 使用直接原则反馈抑制“粉色大象”

    Suppressing Pink Elephants with Direct Principle Feedback

    [https://arxiv.org/abs/2402.07896](https://arxiv.org/abs/2402.07896)

    本研究提出了一种名为“直接原则反馈”的新方法，用于控制语言模型中的LLM行为。通过在批评和修订上直接使用DPO来跳过响应的排名，我们成功地解决了“粉色大象问题”并取得了显著的性能优势。

    

    目前的语言模型控制方法，如RLHF和宪法AI，涉及确定LLM行为的可取之处，并将其训练到语言模型中。然而，在许多情况下，希望LLM在推理时是可控制的，这样可以在多种需要的上下文中使用。我们用“粉色大象问题”作为例子：指示LLM避免讨论某个特定实体（“粉色大象”），而是讨论首选实体（“灰色大象”）。我们应用了一种新颖的Constitutional AI简化方法，“直接原则反馈”，它跳过了对响应的排名，直接在批评和修订上使用DPO。我们的结果表明，在我们合成的“粉色大象”数据集上进行DPF微调后，我们的13B微调LLaMA 2模型明显优于Llama-2-13B-Chat和提示基线，并且在评估“粉色大象问题”的精心选择测试集上表现与GPT-4一样好。

    Existing methods for controlling language models, such as RLHF and Constitutional AI, involve determining which LLM behaviors are desirable and training them into a language model. However, in many cases, it is desirable for LLMs to be controllable \textit{at inference time}, so that they can be used in multiple contexts with diverse needs. We illustrate this with the \textbf{Pink Elephant Problem}: instructing an LLM to avoid discussing a certain entity (a ``Pink Elephant''), and instead discuss a preferred entity (``Grey Elephant''). We apply a novel simplification of Constitutional AI, \textbf{Direct Principle Feedback}, which skips the ranking of responses and uses DPO directly on critiques and revisions. Our results show that after DPF fine-tuning on our synthetic Pink Elephants dataset, our 13B fine-tuned LLaMA 2 model significantly outperforms Llama-2-13B-Chat and a prompted baseline, and performs as well as GPT-4 in on our curated test set assessing the Pink Elephant Problem.
    
[^5]: 标注效率高的文本生成模型选择

    Label-Efficient Model Selection for Text Generation

    [https://arxiv.org/abs/2402.07891](https://arxiv.org/abs/2402.07891)

    DiffUse是一种标注效率高的文本生成模型选择方法，它通过聚类文本语义差异的嵌入来选择更具信息量的实例，并能显著减少所需的注释数量。

    

    针对给定目标任务的模型选择可能成本高昂，因为它可能需要对不同模型输出的质量进行广泛的注释。我们引入了DiffUse，一种有效的方法来在候选文本生成模型之间做出明智的决策。DiffUse减少了所需的偏好注释数量，从而节省了在评估中宝贵的时间和资源。DiffUse通过聚类表示模型输出之间的语义差异的嵌入来智能选择实例。因此，它能够识别出一些更有信息量的例子来进行偏好决策。我们的方法与模型无关，可以应用于任何文本生成模型。此外，我们提出了一种实用的迭代方法来动态确定要注释的实例数量。通过对数百个模型对进行一系列实验，我们证明了DiffUse可以显著减少所需的注释数量，最多可减少75%，同时保持高评估水平。

    Model selection for a given target task can be costly, as it may entail extensive annotation of the quality of outputs of different models. We introduce DiffUse, an efficient method to make an informed decision between candidate text generation models. DiffUse reduces the required amount of preference annotations, thus saving valuable time and resources in performing evaluation. DiffUse intelligently selects instances by clustering embeddings that represent the semantic differences between model outputs. Thus, it is able to identify a subset of examples that are more informative for preference decisions. Our method is model-agnostic, and can be applied to any text generation model. Moreover, we propose a practical iterative approach for dynamically determining how many instances to annotate. In a series of experiments over hundreds of model pairs, we demonstrate that DiffUse can dramatically reduce the required number of annotations -- by up to 75% -- while maintaining high evaluation 
    
[^6]: 使用语言反馈模型来改进政策

    Policy Improvement using Language Feedback Models

    [https://arxiv.org/abs/2402.07876](https://arxiv.org/abs/2402.07876)

    本文介绍了一种使用语言反馈模型（LFMs）改进政策的方法，通过识别期望的行为并进行模仿学习，我们在任务完成率、泛化性能和人类可解释性方面取得了显著改进。

    

    我们引入了语言反馈模型（LFMs），用于在指令遵循中识别期望的行为-有助于实现指令中指定任务的行动-以进行模仿学习。为了训练LFMs，我们从大型语言模型（LLMs）获取对视觉轨迹进行语言描述的反馈。首先，通过使用LFMs识别期望模仿的行为，我们在三种不同的语言基础环境（Touchdown，ScienceWorld和ALFWorld）上，在任务完成率上改善了强行为克隆的基线方法。其次，与LLMs直接预测行动相比，使用LFMs在LLM输出标记的数量相同的情况下表现更好。第三，LFMs适应未见环境，通过一轮适应使任务完成率提高了3.5-12.0％。最后，可以修改LFM以提供人类可解释的反馈，无需性能损失，从而允许人类验证模仿学习的期望行为。

    We introduce Language Feedback Models (LFMs) that identify desirable behaviour - actions that help achieve tasks specified in the instruction - for imitation learning in instruction following. To train LFMs, we obtain feedback from Large Language Models (LLMs) on visual trajectories verbalized to language descriptions. First, by using LFMs to identify desirable behaviour to imitate, we improve in task-completion rate over strong behavioural cloning baselines on three distinct language grounding environments (Touchdown, ScienceWorld, and ALFWorld). Second, LFMs outperform using LLMs as experts to directly predict actions, when controlling for the number of LLM output tokens. Third, LFMs generalize to unseen environments, improving task-completion rate by 3.5-12.0% through one round of adaptation. Finally, LFM can be modified to provide human-interpretable feedback without performance loss, allowing human verification of desirable behaviour for imitation learning.
    
[^7]: PIVOT: 迭代视觉提示激发可操作知识用于VLMs

    PIVOT: Iterative Visual Prompting Elicits Actionable Knowledge for VLMs

    [https://arxiv.org/abs/2402.07872](https://arxiv.org/abs/2402.07872)

    本文介绍了一种名为PIVOT的新颖视觉提示方法，它通过迭代的视觉问答将任务转化为VLMs问题。每个迭代中，图像被标注为VLMs可以参考的视觉表示，并通过优化选择最佳选项。这种方法能够使VLMs进行机器人控制和其他空间任务的输出。

    

    视觉语言模型（VLMs）显示出在各种任务中的令人印象深刻的能力，从逻辑推理到视觉理解。这为与世界进行更丰富的互动打开了大门，例如机器人控制。然而，VLMs只产生文本输出，而机器人控制和其他空间任务需要输出连续的坐标，动作或轨迹。我们如何在不对任务特定数据进行微调的情况下使VLMs能够处理这种设置呢？在本文中，我们提出了一种新颖的VLMs视觉提示方法，称之为迭代视觉优化提示（PIVOT），将任务视为迭代的视觉问答。在每个迭代中，图像被注释为VLMs可以参考的提案的视觉表示（例如候选机器人动作、定位或轨迹）。然后，VLMs选择最佳的任务。这些提案经过迭代优化，使VLMs最终找到最佳的可用选项。

    Vision language models (VLMs) have shown impressive capabilities across a variety of tasks, from logical reasoning to visual understanding. This opens the door to richer interaction with the world, for example robotic control. However, VLMs produce only textual outputs, while robotic control and other spatial tasks require outputting continuous coordinates, actions, or trajectories. How can we enable VLMs to handle such settings without fine-tuning on task-specific data?   In this paper, we propose a novel visual prompting approach for VLMs that we call Prompting with Iterative Visual Optimization (PIVOT), which casts tasks as iterative visual question answering. In each iteration, the image is annotated with a visual representation of proposals that the VLM can refer to (e.g., candidate robot actions, localizations, or trajectories). The VLM then selects the best ones for the task. These proposals are iteratively refined, allowing the VLM to eventually zero in on the best available an
    
[^8]: 细粒度混合专家模型的标度律

    Scaling Laws for Fine-Grained Mixture of Experts

    [https://arxiv.org/abs/2402.07871](https://arxiv.org/abs/2402.07871)

    本研究分析了细粒度混合专家模型的标度特性，并引入了粒度作为新的超参数，通过调整粒度可以精确控制专家的大小。研究结果显示，MoE模型在效果上始终优于密集变压器模型，并且随着模型大小和训练预算的增大，密集和MoE模型之间的效率差距也在增大。同时，将MoE中专家的大小设置为与前馈层相同的常见做法在几乎任何计算预算下都不是最优的。

    

    混合专家（MoE）模型已成为减少大型语言模型计算成本的主要解决方案。在这项工作中，我们分析了它们的标度特性，并纳入了更广泛的变量范围。具体地，我们引入了一个新的超参数，称为粒度，通过调整粒度可以精确控制专家的大小。基于此，我们建立了细粒度MoE的标度律，考虑了训练标记数、模型大小和粒度。利用这些定律，我们推导出了给定计算预算下的最佳训练配置。我们的研究结果不仅表明MoE模型始终优于密集变压器模型，而且还凸显了在扩大模型大小和训练预算时，密集和MoE模型之间的效率差距在扩大。此外，我们证明了将MoE中专家的大小设置为与前馈层相同的常见做法在几乎任何计算预算下都不是最优的。

    Mixture of Experts (MoE) models have emerged as a primary solution for reducing the computational cost of Large Language Models. In this work, we analyze their scaling properties, incorporating an expanded range of variables. Specifically, we introduce a new hyperparameter, granularity, whose adjustment enables precise control over the size of the experts. Building on this, we establish scaling laws for fine-grained MoE, taking into account the number of training tokens, model size, and granularity. Leveraging these laws, we derive the optimal training configuration for a given computational budget. Our findings not only show that MoE models consistently outperform dense Transformers but also highlight that the efficiency gap between dense and MoE models widens as we scale up the model size and training budget. Furthermore, we demonstrate that the common practice of setting the size of experts in MoE to mirror the feed-forward layer is not optimal at almost any computational budget.
    
[^9]: 透视VLMs：探索视觉条件化语言模型的设计空间

    Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models

    [https://arxiv.org/abs/2402.07865](https://arxiv.org/abs/2402.07865)

    本论文探索了视觉条件化语言模型（VLMs）设计的关键空间，并提供了一套标准化评估，同时还研究了预训练的视觉表示和权衡的问题。

    

    视觉条件化语言模型（VLMs）在视觉对话、场景理解和机器人任务规划等应用中得到了越来越多的应用，这种应用促使了像LLaVa、InstructBLIP和PaLI-3等许多新模型的出现。尽管有这么多新的发布，但关于图像预处理、架构和优化的关键设计决策仍然未被充分探索，这使得我们很难理解模型性能的因素，这一挑战又因缺乏客观、一致的评估而变得更加复杂。为了填补这些空白，我们首先编制了一套标准化评估，涵盖了视觉问答、从语言中定位物体以及探索幻觉等属性的目标挑战集，这些评估可以提供关于VLM能力的精细、准确的见解。其次，我们对关键的设计轴进行了严格的研究，包括预训练的视觉表示和使用的权衡。

    Visually-conditioned language models (VLMs) have seen growing adoption in applications such as visual dialogue, scene understanding, and robotic task planning; adoption that has fueled a wealth of new models such as LLaVa, InstructBLIP, and PaLI-3. Despite the volume of new releases, key design decisions around image preprocessing, architecture, and optimization are under-explored, making it challenging to understand what factors account for model performance $-$ a challenge further complicated by the lack of objective, consistent evaluations. To address these gaps, we first compile a suite of standardized evaluations spanning visual question answering, object localization from language, and targeted challenge sets that probe properties such as hallucination; evaluations that provide calibrated, fine-grained insight into a VLM's capabilities. Second, we rigorously investigate VLMs along key design axes, including pretrained visual representations and quantifying the tradeoffs of using 
    
[^10]: AI增强预测：LLM助手提高人类预测准确性

    AI-Augmented Predictions: LLM Assistants Improve Human Forecasting Accuracy

    [https://arxiv.org/abs/2402.07862](https://arxiv.org/abs/2402.07862)

    本研究发现，使用LLMs助手可以显著提高预测准确性，不仅仅是由于模型预测准确性的提升。

    

    大型语言模型(LLMs)展现出令人印象深刻的能力，在许多领域与甚至超过人类表现。本研究探讨了LLMs在预测任务中增强判断力的潜力。我们评估了两个GPT-4-Turbo助手对预测准确性的影响：一个旨在提供高质量建议（超级预测），另一个旨在过于自信和基本概率忽视。参与者（N = 991）可以在整个研究过程中咨询他们被分配的LLM助手，而对照组则使用一个较低级别的模型（DaVinci-003），不提供直接的预测支持。我们的注册分析显示，LLM增强显著提高了23%的预测准确性，无论是对于任何一种助手类型，相比于对照组。这种改进发生在超级预测助手在预测中更高的准确性的情况下，表明增强的效益不仅仅是由于模型预测准确性。

    Large language models (LLMs) show impressive capabilities, matching and sometimes exceeding human performance in many domains. This study explores the potential of LLMs to augment judgement in forecasting tasks. We evaluated the impact on forecasting accuracy of two GPT-4-Turbo assistants: one designed to provide high-quality advice ('superforecasting'), and the other designed to be overconfident and base-rate-neglecting. Participants (N = 991) had the option to consult their assigned LLM assistant throughout the study, in contrast to a control group that used a less advanced model (DaVinci-003) without direct forecasting support. Our preregistered analyses reveal that LLM augmentation significantly enhances forecasting accuracy by 23% across both types of assistants, compared to the control group. This improvement occurs despite the superforecasting assistant's higher accuracy in predictions, indicating the augmentation's benefit is not solely due to model prediction accuracy. Explora
    
[^11]: Lissard：长而简单的顺序推理数据集

    Lissard: Long and Simple Sequential Reasoning Datasets

    [https://arxiv.org/abs/2402.07859](https://arxiv.org/abs/2402.07859)

    Lissard是一个包含七个任务的基准，用于评估模型处理和生成各种序列长度的能力，需要重复的过程执行。评估结果显示随着序列复杂性增加，所有模型的性能都呈一致下降趋势。

    

    语言模型现在能够解决需要处理数十万个标记的长序列的任务。然而，它们在需要重复使用简单规则的任务上常常失败，甚至在比训练中看到的序列要短得多的情况下也是如此。例如，最先进的LLMs可以在两个列表中找到共同项，列表中的项最多可达20个，但是当列表中的项达到80个时，它们会失败。在本文中，我们介绍了Lissard，这是一个包含七个任务的基准，旨在评估模型处理和生成各种序列长度的能力，需要重复的过程执行。我们评估了开源模型（Mistral-7B和Mixtral-8x7B）和专有模型（GPT-3.5和GPT-4），结果显示随着序列复杂性增加，所有模型的性能都呈一致下降趋势。数据集和代码可在https://github.com/unicamp-dl/Lissard获得。

    Language models are now capable of solving tasks that require dealing with long sequences consisting of hundreds of thousands of tokens. However, they often fail on tasks that require repetitive use of simple rules, even on sequences that are much shorter than those seen during training. For example, state-of-the-art LLMs can find common items in two lists with up to 20 items but fail when lists have 80 items. In this paper, we introduce Lissard, a benchmark comprising seven tasks whose goal is to assess the ability of models to process and generate wide-range sequence lengths, requiring repetitive procedural execution. Our evaluation of open-source (Mistral-7B and Mixtral-8x7B) and proprietary models (GPT-3.5 and GPT-4) show a consistent decline in performance across all models as the complexity of the sequence increases. The datasets and code are available at https://github.com/unicamp-dl/Lissard
    
[^12]: Mercury: 一种用于LLM代码综合效率评估的基准

    Mercury: An Efficiency Benchmark for LLM Code Synthesis

    [https://arxiv.org/abs/2402.07844](https://arxiv.org/abs/2402.07844)

    Mercury提出了一个针对LLM代码综合任务的效率评估基准，通过引入新的度量标准Beyond@K来衡量归一化的代码效率，从而鼓励生成功能正确且计算效率高的代码。

    

    尽管在评估大型语言模型（LLM）进行代码综合方面取得了进展，但基准主要集中在功能正确性上，忽视了代码效率的重要性。我们提出了Mercury，这是第一个专用于评估LLM代码综合任务的代码效率的基准。Mercury由1,889个涵盖不同难度级别的编程任务组成，还包括生成无限案例的测试用例生成器，以进行全面评估。与现有的基准不同，Mercury集成了一种新的度量标准Beyond@K，以基于历史提交来衡量归一化的代码效率，从而为代码综合提供了新的评估指标，鼓励生成功能正确且计算效率高的代码，体现了现实世界软件开发的标准。我们的研究结果表明，虽然LLM表现出生成功能正确代码的显著能力，但它们在效率输出方面仍存在很大的差距。

    Despite advancements in evaluating Large Language Models (LLMs) for code synthesis, benchmarks have predominantly focused on functional correctness, overlooking the importance of code efficiency. We present Mercury, the first benchmark designated for assessing the code efficiency of LLM code synthesis tasks. Mercury consists of 1,889 programming tasks covering diverse difficulty levels alongside test case generators generating unlimited cases for comprehensive evaluation. Unlike existing benchmarks, Mercury integrates a novel metric Beyond@K to measure normalized code efficiency based on historical submissions, leading to a new evaluation indicator for code synthesis, which encourages generating functionally correct and computationally efficient code, mirroring the real-world software development standard. Our findings reveal that while LLMs demonstrate the remarkable capability to generate functionally correct code, there still exists a substantial gap in their efficiency output, unde
    
[^13]: 大型语言模型上的成员推断攻击是否奏效？

    Do Membership Inference Attacks Work on Large Language Models?

    [https://arxiv.org/abs/2402.07841](https://arxiv.org/abs/2402.07841)

    这项研究在大规模语言模型上对成员推断攻击进行了评估，发现在大部分设置中，攻击几乎只能比随机猜测稍好，这种糟糕的性能是由于大型数据集和少量训练迭代的组合，以及成员和非成员之间的边界困惑所导致的。

    

    成员推断攻击（MIAs）试图预测特定数据点是否属于目标模型的训练数据。尽管对传统机器学习模型进行了广泛研究，但在大型语言模型（LLMs）的预训练数据上对MIA的研究工作仍有限。我们对在Pile上训练的一系列语言模型（LMs）进行了大规模的MIA评估，参数范围从160M到12B。我们发现，在不同的LLM大小和领域的大多数设置中，MIAs几乎只能比随机猜测稍好。我们进一步分析发现，这种糟糕的性能可以归因于（1）大型数据集和少量训练迭代的组合，以及（2）成员和非成员之间的边界困惑。我们确定了LLMs易受成员推断攻击的特定设置，并表明在这些设置中取得的表面上的成功可以归因于分布的转变，例如当成员和非成员被绘制出来时。

    Membership inference attacks (MIAs) attempt to predict whether a particular datapoint is a member of a target model's training data. Despite extensive research on traditional machine learning models, there has been limited work studying MIA on the pre-training data of large language models (LLMs). We perform a large-scale evaluation of MIAs over a suite of language models (LMs) trained on the Pile, ranging from 160M to 12B parameters. We find that MIAs barely outperform random guessing for most settings across varying LLM sizes and domains. Our further analyses reveal that this poor performance can be attributed to (1) the combination of a large dataset and few training iterations, and (2) an inherently fuzzy boundary between members and non-members. We identify specific settings where LLMs have been shown to be vulnerable to membership inference and show that the apparent success in such settings can be attributed to a distribution shift, such as when members and non-members are drawn
    
[^14]: Aya模型：一个经过指令微调的开放多语言模型

    Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model

    [https://arxiv.org/abs/2402.07827](https://arxiv.org/abs/2402.07827)

    Aya是一个开放多语言模型，通过指令微调，在101种语言中表现优于其他模型，扩展了多语言评估的技术，并进行了深入研究优化微调组合、数据修剪以及模型的毒性、偏差和安全性。

    

    最近大规模语言模型（LLMs）的突破主要集中在少数数据丰富的语言上。如何拓宽对突破性成果的访问范围以覆盖非主流语言呢？我们的工作引入了Aya，一个支持101种语言的大规模多语言生成模型，其中超过50％的语言被认为是资源较少的。Aya在大多数任务上表现优于mT0和BLOOMZ，同时覆盖的语言数量是它们的两倍。我们引入了广泛的新评估套件，扩展了99种语言的多语种评估的最新技术，其中包括区分和生成任务、人工评估以及模拟的胜率评估，涵盖了保留任务和分布内性能。此外，我们对最佳微调混合物组成、数据修剪以及模型的毒性、偏差和安全性进行了详细研究。我们将我们的指令数据集和模型开源在https://hf.co/CohereForAI/aya上。

    Recent breakthroughs in large language models (LLMs) have centered around a handful of data-rich languages. What does it take to broaden access to breakthroughs beyond first-class citizen languages? Our work introduces Aya, a massively multilingual generative language model that follows instructions in 101 languages of which over 50% are considered as lower-resourced. Aya outperforms mT0 and BLOOMZ on the majority of tasks while covering double the number of languages. We introduce extensive new evaluation suites that broaden the state-of-art for multilingual eval across 99 languages -- including discriminative and generative tasks, human evaluation, and simulated win rates that cover both held-out tasks and in-distribution performance. Furthermore, we conduct detailed investigations on the optimal finetuning mixture composition, data pruning, as well as the toxicity, bias, and safety of our models. We open-source our instruction datasets and our model at https://hf.co/CohereForAI/aya-
    
[^15]: 利用对比学习注入Wiktionary改善词级上下文表示的研究

    Injecting Wiktionary to improve token-level contextual representations using contrastive learning

    [https://arxiv.org/abs/2402.07817](https://arxiv.org/abs/2402.07817)

    本文研究了利用对比学习注入Wiktionary来改善词级上下文表示，并在Word-In-Context（WiC）任务上取得了新的最佳结果。

    

    虽然静态词嵌入对上下文是无感的，但对于词汇语义任务来说，上下文在上下文词嵌入中过于明显，相同含义的词向量差异较大。本文提出使用对比学习来微调预训练语言模型（PLMs），利用自动自增示例。我们调查了如何将词典作为替代的监督资源注入，使用英文Wiktionary。我们还测试了降维对结果上下文词嵌入的影响。我们在无监督设置下（不使用训练集）在Word-In-Context（WiC）任务上评估了我们的方法。我们在原始WiC测试集上取得了新的SoTA结果。我们还提出了两个新的WiC测试集，其中我们展示了我们的微调方法取得了显著的改进。我们还观察到在语义框架识别任务中的改进，尽管效果较为温和。尽管我们在E进行了实验

    While static word embeddings are blind to context, for lexical semantics tasks context is rather too present in contextual word embeddings, vectors of same-meaning occurrences being too different (Ethayarajh, 2019). Fine-tuning pre-trained language models (PLMs) using contrastive learning was proposed, leveraging automatically self-augmented examples (Liu et al., 2021b). In this paper, we investigate how to inject a lexicon as an alternative source of supervision, using the English Wiktionary. We also test how dimensionality reduction impacts the resulting contextual word embeddings. We evaluate our approach on the Word-In-Context (WiC) task, in the unsupervised setting (not using the training set). We achieve new SoTA result on the original WiC test set. We also propose two new WiC test sets for which we show that our fine-tuning method achieves substantial improvements. We also observe improvements, although modest, for the semantic frame induction task. Although we experimented on E
    
[^16]: 检索增强的思维过程作为序列决策制定

    Retrieval-Augmented Thought Process as Sequential Decision Making

    [https://arxiv.org/abs/2402.07812](https://arxiv.org/abs/2402.07812)

    检索增强思维过程（RATP）通过多步决策和蒙特卡洛树搜索，以及Q值估计器，解决了大型语言模型在隐私、产生幻觉和处理长文本方面的挑战，并在处理私人数据的问答任务中实现了50%的性能提升。

    

    大型语言模型(LLM)展示了其强大的辅助人类并展现出"智能的火花"的能力。然而，几个开放挑战阻碍了它们的广泛应用：如对隐私的关注、倾向于产生幻觉、难以处理长文本。在本研究中，我们通过引入检索增强思维过程(RATP)来解决这些挑战。通过获取外部知识，RATP将LLM的思考生成过程定式为多步决策过程。为了优化这种思考过程，RATP利用蒙特卡洛树搜索，并学习了一个Q值估计器，实现了高效的推理。在处理具有私人数据的问答任务时，LLM训练方法受到伦理和安全问题的限制。RATP在上下文检索增强语言模型的基础上实现了50%的性能提升。

    Large Language Models (LLMs) have demonstrated their strong ability to assist people and show "sparks of intelligence". However, several open challenges hinder their wider application: such as concerns over privacy, tendencies to produce hallucinations, and difficulties in handling long contexts. In this work, we address those challenges by introducing the Retrieval-Augmented Thought Process (RATP). Given access to external knowledge, RATP formulates the thought generation of LLMs as a multiple-step decision process. To optimize such a thought process, RATP leverages Monte-Carlo Tree Search, and learns a Q-value estimator that permits cost-efficient inference. In addressing the task of question-answering with private data, where ethical and security concerns limit LLM training methods, RATP achieves a 50% improvement over existing in-context retrieval-augmented language models.
    
[^17]: 在搜索中的多意图属性感知文本匹配

    Multi-Intent Attribute-Aware Text Matching in Searching

    [https://arxiv.org/abs/2402.07788](https://arxiv.org/abs/2402.07788)

    本研究针对搜索中的文本匹配系统进行了多意图属性感知的研究，提出了通过多意图建模来利用多个属性之间的关系。意图从属性中提取，总结了查询的多样化需求。

    

    文本匹配系统已成为大多数搜索平台的基本服务。例如，它们负责将用户查询匹配到相关的候选项，或将用户输入的查询改写为预选的高性能查询，以获得更好的搜索体验。在实践中，查询和项通常包含多个属性，例如项的类别和查询中提到的位置，这些属性代表着有助于匹配的关键信息。然而，大多数现有工作通过将属性集成到文本表示中作为辅助信息，低估了属性的有效性。因此，本文致力于探索两方面的属性之间的关系。由于两端的属性在数量和类型上通常不对齐，我们提出通过多意图建模来利用属性的好处。从属性中提取的意图总结了查询的多样化需求，并提供。

    Text matching systems have become a fundamental service in most searching platforms. For instance, they are responsible for matching user queries to relevant candidate items, or rewriting the user-input query to a pre-selected high-performing one for a better search experience. In practice, both the queries and items often contain multiple attributes, such as the category of the item and the location mentioned in the query, which represent condensed key information that is helpful for matching. However, most of the existing works downplay the effectiveness of attributes by integrating them into text representations as supplementary information. Hence, in this work, we focus on exploring the relationship between the attributes from two sides. Since attributes from two ends are often not aligned in terms of number and type, we propose to exploit the benefit of attributes by multiple-intent modeling. The intents extracted from attributes summarize the diverse needs of queries and provide 
    
[^18]: TELLER: 一个可信的、可推广的、可控制的假新闻检测框架

    TELLER: A Trustworthy Framework for Explainable, Generalizable and Controllable Fake News Detection

    [https://arxiv.org/abs/2402.07776](https://arxiv.org/abs/2402.07776)

    TELLER是一个可信的假新闻检测框架，通过集成认知和决策系统，提供了解释性、可推广性和可控制性。认知系统利用人类专业知识生成逻辑谓词，指导大型语言模型生成可读的逻辑原子。决策系统推导可推广的逻辑规则来聚合这些原子，实现真实和虚假新闻的识别。

    

    假新闻的泛滥已成为一个严重的社会问题，引起了行业和学术界的广泛关注。虽然现有的基于深度学习的方法在准确检测假新闻方面取得了进展，但它们的可靠性可能会受到非透明推理过程、差强人意的推广能力以及与大型语言模型（LLMs）集成的固有风险的影响。为了应对这一挑战，我们提出了一种新颖的框架，名为TELLER，用于可信的假新闻检测，重点关注模型的可解释性、可推广性和可控制性。这通过一个融合认知和决策系统的双系统框架来实现，遵循以上原则。认知系统利用人类专业知识生成逻辑谓词，指导LLMs生成可读的逻辑原子。同时，决策系统推导可推广的逻辑规则来聚合这些原子，实现真实和虚假新闻的识别。

    The proliferation of fake news has emerged as a severe societal problem, raising significant interest from industry and academia. While existing deep-learning based methods have made progress in detecting fake news accurately, their reliability may be compromised caused by the non-transparent reasoning processes, poor generalization abilities and inherent risks of integration with large language models (LLMs). To address this challenge, we propose {\methodname}, a novel framework for trustworthy fake news detection that prioritizes explainability, generalizability and controllability of models. This is achieved via a dual-system framework that integrates cognition and decision systems, adhering to the principles above. The cognition system harnesses human expertise to generate logical predicates, which guide LLMs in generating human-readable logic atoms. Meanwhile, the decision system deduces generalizable logic rules to aggregate these atoms, enabling the identification of the truthfu
    
[^19]: 大型语言模型中的定量知识检索

    Quantitative knowledge retrieval from large language models

    [https://arxiv.org/abs/2402.07770](https://arxiv.org/abs/2402.07770)

    本文探讨了大型语言模型（LLMs）作为定量知识检索的可行性，以辅助数据分析任务。提出了一个提示工程框架，将LLMs作为科学文献潜在空间的接口。讨论了使用LLMs作为“专家”的影响和挑战。

    

    大型语言模型（LLM）因其生成具有说服力的自然语言序列的能力而被广泛研究，但其作为定量信息检索的实用性尚不明确。本文探讨了将LLMs作为定量知识检索机制的可行性，以帮助数据分析任务，如贝叶斯模型的先验分布引导和缺失数据的填补。我们提出了一个提示工程框架，将LLMs视为科学文献潜在空间的接口，在不同上下文和领域中比较响应与更成熟的方法。讨论了使用LLMs作为“专家”的影响和挑战。

    Large language models (LLMs) have been extensively studied for their abilities to generate convincing natural language sequences, however their utility for quantitative information retrieval is less well understood. In this paper we explore the feasibility of LLMs as a mechanism for quantitative knowledge retrieval to aid data analysis tasks such as elicitation of prior distributions for Bayesian models and imputation of missing data. We present a prompt engineering framework, treating an LLM as an interface to a latent space of scientific literature, comparing responses in different contexts and domains against more established approaches. Implications and challenges of using LLMs as 'experts' are discussed.
    
[^20]: 文本解毒作为英语和印地语中的风格转移

    Text Detoxification as Style Transfer in English and Hindi

    [https://arxiv.org/abs/2402.07767](https://arxiv.org/abs/2402.07767)

    本文研究了文本解毒化的任务，旨在将有毒文本自动转化为无毒文本。通过知识转移、多任务学习和删除重建方法，我们提出了三种解决方案。我们利用Dementieva等人提供的数据集进行实验，并引入了一个小型的印地语平行数据集用于评估。

    

    本文关注文本解毒，即自动将有毒文本转化为非有毒文本。这项任务有助于更安全、更尊重的在线交流，并可被视为文本风格转移（TST）任务，在此任务中，文本风格发生变化，但内容保持不变。我们提出了三种方法：从类似任务中进行知识转移，多任务学习方法，将序列到序列建模与各种毒性分类任务相结合，并采用删除和重建方法。为支持我们的研究，我们利用了Dementieva等人提供的数据集（2021年），该数据集包含与有毒文本对应的多个版本的解毒文本。在我们的实验中，我们通过专家人工注释员选择了最佳的变体，创建了一个数据集，其中每个有毒句子与一个适当的解毒版本配对。此外，我们还引入了一个小型的印地语平行数据集，与英语数据集的一部分对齐，适用于评估目的。

    This paper focuses on text detoxification, i.e., automatically converting toxic text into non-toxic text. This task contributes to safer and more respectful online communication and can be considered a Text Style Transfer (TST) task, where the text style changes while its content is preserved. We present three approaches: knowledge transfer from a similar task, multi-task learning approach, combining sequence-to-sequence modeling with various toxicity classification tasks, and, delete and reconstruct approach. To support our research, we utilize a dataset provided by Dementieva et al.(2021), which contains multiple versions of detoxified texts corresponding to toxic texts. In our experiments, we selected the best variants through expert human annotators, creating a dataset where each toxic sentence is paired with a single, appropriate detoxified version. Additionally, we introduced a small Hindi parallel dataset, aligning with a part of the English dataset, suitable for evaluation purp
    
[^21]: 思想传播：扩散语言模型中的思维链推理

    Diffusion of Thoughts: Chain-of-Thought Reasoning in Diffusion Language Models

    [https://arxiv.org/abs/2402.07754](https://arxiv.org/abs/2402.07754)

    本文介绍了一种将扩散模型与思维链推理集成的方法，通过扩散传播推理步骤，提供了更大的灵活性和推理能力。实验证明了该方法在数学问题中的有效性，并展示了自我纠正能力和推理技术的潜力。

    

    扩散模型在文本处理中引起了关注，相对传统的自回归模型具有许多潜在优势。本文探讨了将扩散模型与思维链（CoT）集成的方法，CoT是一种在自回归语言模型中改进推理能力的成熟技术。我们提出了思维扩散（DoT）模型，允许推理步骤通过扩散过程在时间上传播。与传统的自回归语言模型逐个token从左到右做出决策的方式相比，DoT在计算和推理性能之间具有更大的灵活性。我们的实验证明了DoT在多位数乘法和小学数学问题中的有效性。此外，DoT展示了有希望的自我纠正能力，并从现有的增强推理技术（如自一致解码）中受益。我们的发现有助于理解和发展推理能力。

    Diffusion models have gained attention in text processing, offering many potential advantages over traditional autoregressive models. This work explores the integration of diffusion models and Chain-of-Thought (CoT), a well-established technique to improve the reasoning ability in autoregressive language models. We propose Diffusion-of-Thought (DoT), allowing reasoning steps to diffuse over time through the diffusion process. In contrast to traditional autoregressive language models that make decisions in a left-to-right, token-by-token manner, DoT offers more flexibility in the trade-off between computation and reasoning performance. Our experimental results demonstrate the effectiveness of DoT in multi-digit multiplication and grade school math problems. Additionally, DoT showcases promising self-correction abilities and benefits from existing reasoning-enhancing techniques like self-consistency decoding. Our findings contribute to the understanding and development of reasoning capab
    
[^22]: 实现智能体、人类和环境之间的统一对齐

    Towards Unified Alignment Between Agents, Humans, and Environment

    [https://arxiv.org/abs/2402.07744](https://arxiv.org/abs/2402.07744)

    本文介绍了统一对齐原则 ($\mathbf{UA}^2$)，旨在实现智能体与人类意图、环境动态和自我约束的统一对齐，提出了引入实际特性进行概念验证研究的方法。

    

    基于基础模型的快速进展导致了自主智能体的繁荣，这些智能体利用基础模型的通用能力进行推理、决策和环境交互。然而，当在复杂、现实的环境中运行时，智能体的效能仍然有限。在本研究中，我们引入了统一对齐原则，即同时对齐智能体与人类意图、环境动态和自我约束（如货币预算限制）。从统一对齐 ($\mathbf{UA}^2$) 的视角出发，我们回顾了当前智能体研究的现状，并指出了现有智能体基准和方法候选中被忽视的因素。我们还通过为WebShop引入实际特性进行了概念验证研究，包括使用用户配置文件来展示意图、个性化重新排名以应对复杂的环境动态和运行时成本统计。

    The rapid progress of foundation models has led to the prosperity of autonomous agents, which leverage the universal capabilities of foundation models to conduct reasoning, decision-making, and environmental interaction. However, the efficacy of agents remains limited when operating in intricate, realistic environments. In this work, we introduce the principles of $\mathbf{U}$nified $\mathbf{A}$lignment for $\mathbf{A}$gents ($\mathbf{UA}^2$), which advocate for the simultaneous alignment of agents with human intentions, environmental dynamics, and self-constraints such as the limitation of monetary budgets. From the perspective of $\mathbf{UA}^2$, we review the current agent research and highlight the neglected factors in existing agent benchmarks and method candidates. We also conduct proof-of-concept studies by introducing realistic features to WebShop, including user profiles to demonstrate intentions, personalized reranking for complex environmental dynamics, and runtime cost stat
    
[^23]: 在混合式主动对话搜索中提出多模态澄清问题

    Asking Multimodal Clarifying Questions in Mixed-Initiative Conversational Search

    [https://arxiv.org/abs/2402.07742](https://arxiv.org/abs/2402.07742)

    本论文提出在混合式主动对话搜索中通过使用多模态信息来改进澄清问题的方法，并设计了一个名为Marto的多模态查询澄清模型。通过收集多模态澄清问题和图像的数据集Melon，并采用基于提示的训练策略，为进一步研究这一任务提供了便利。

    

    在混合式主动对话搜索系统中，澄清问题被用于帮助那些难以用一个查询表达自己意图的用户。这些问题旨在揭示用户的信息需求并解决查询的歧义。我们假设，在与多模态信息相关的情景中，通过使用非文本信息可以改进澄清过程。因此，我们提出将图像添加到澄清问题中，并提出在开放域、混合式主动对话搜索系统中提出多模态澄清问题的新任务。为了促进对这个任务的研究，我们收集了一个名为Melon的数据集，其中包含超过4k个多模态澄清问题，并附带超过14k个图像。我们还提出了一个名为Marto的多模态查询澄清模型，并采用基于提示的生成微调策略，使用不同的提示来进行不同阶段的训练。进行了几个分析来理解重要因素。

    In mixed-initiative conversational search systems, clarifying questions are used to help users who struggle to express their intentions in a single query. These questions aim to uncover user's information needs and resolve query ambiguities. We hypothesize that in scenarios where multimodal information is pertinent, the clarification process can be improved by using non-textual information. Therefore, we propose to add images to clarifying questions and formulate the novel task of asking multimodal clarifying questions in open-domain, mixed-initiative conversational search systems. To facilitate research into this task, we collect a dataset named Melon that contains over 4k multimodal clarifying questions, enriched with over 14k images. We also propose a multimodal query clarification model named Marto and adopt a prompt-based, generative fine-tuning strategy to perform the training of different stages with different prompts. Several analyses are conducted to understand the importance 
    
[^24]: 无监督的手语翻译和生成

    Unsupervised Sign Language Translation and Generation

    [https://arxiv.org/abs/2402.07726](https://arxiv.org/abs/2402.07726)

    本文介绍了一个无监督的手语翻译和生成网络（USLNet），它通过利用大量的单模态数据（文本和视频）学习，而无需平行手语数据。USLNet采用不同模态的反向翻译和重建技术，面对文本和视频序列之间的特征表示差异。通过使用滑动窗口方法，USLNet能够有效对齐不同长度的文本和视频序列。这是第一个实现无监督手语翻译和生成的方法。

    

    受无监督神经机器翻译（UNMT）的成功启发，我们引入了一个无监督的手语翻译和生成网络（USLNet），它能够从大量的单模态（文本和视频）数据中学习，而无需平行的手语数据。USLNet包括两个主要组成部分：单模态重建模块（文本和视频）用于在相同模态下从嘈杂的输入中重建输入，以及跨模态反向翻译模块（文本-视频-文本和视频-文本-视频）用于使用反向翻译过程从不同模态下的嘈杂输入中重建输入。与基于文本的UNMT中的单模态反向翻译过程不同，USLNet面临着特征表示中的跨模态差异，即文本和视频序列之间的长度和特征维度的不匹配。我们提出了一种滑动窗口方法来解决对齐可变长度的文本和视频序列的问题。据我们所知，USLNet是第一个实现无监督手语翻译和生成的方法。

    Motivated by the success of unsupervised neural machine translation (UNMT), we introduce an unsupervised sign language translation and generation network (USLNet), which learns from abundant single-modality (text and video) data without parallel sign language data. USLNet comprises two main components: single-modality reconstruction modules (text and video) that rebuild the input from its noisy version in the same modality and cross-modality back-translation modules (text-video-text and video-text-video) that reconstruct the input from its noisy version in the different modality using back-translation procedure.Unlike the single-modality back-translation procedure in text-based UNMT, USLNet faces the cross-modality discrepancy in feature representation, in which the length and the feature dimension mismatch between text and video sequences. We propose a sliding window method to address the issues of aligning variable-length text with video sequences. To our knowledge, USLNet is the fir
    
[^25]: LoRA-drop：基于输出评估的高效LoRA参数剪枝

    LoRA-drop: Efficient LoRA Parameter Pruning based on Output Evaluation

    [https://arxiv.org/abs/2402.07721](https://arxiv.org/abs/2402.07721)

    本文提出了LoRA-drop方法，通过分析LoRA输出评估参数的重要性，并且保留重要层的LoRA，其余层共享相同参数。实验结果表明LoRA-drop有很好的效果。

    

    低秩适应（LoRA）为每个层引入辅助参数，以在有限的计算资源下微调预训练模型。但是，当扩展到更大的模型时，仍然面临资源消耗的挑战。先前的研究通过评估不同层的LoRA参数的重要性来采用剪枝技术来解决这个问题。然而，这些努力只分析了参数的特征以评估其重要性。事实上，与参数和数据相关的LoRA的输出是直接影响冻结模型的因素。为此，我们提出了LoRA-drop，通过分析LoRA输出来评估参数的重要性。我们保留重要层的LoRA，而其他层的LoRA共享相同的参数。在NLU和NLG任务上进行了充分的实验，证明了LoRA-drop的有效性。

    Low-Rank Adaptation (LoRA) introduces auxiliary parameters for each layer to fine-tune the pre-trained model under limited computing resources. But it still faces challenges of resource consumption when scaling up to larger models. Previous studies employ pruning techniques by evaluating the importance of LoRA parameters for different layers to address the problem. However, these efforts only analyzed parameter features to evaluate their importance. Indeed, the output of LoRA related to the parameters and data is the factor that directly impacts the frozen model. To this end, we propose LoRA-drop which evaluates the importance of the parameters by analyzing the LoRA output. We retain LoRA for important layers and the LoRA of the other layers share the same parameters. Abundant experiments on NLU and NLG tasks demonstrate the effectiveness of LoRA-drop.
    
[^26]: OrderBkd: 通过重新定位进行的文本后门攻击

    OrderBkd: Textual backdoor attack through repositioning

    [https://arxiv.org/abs/2402.07689](https://arxiv.org/abs/2402.07689)

    本论文提出了一种通过重新定位句子中的两个单词实施文本后门攻击的方法，与已有的攻击方式相比，在攻击成功率、困惑度和与干净样本的语义相似性方面表现更好，并且对ONION防御方法具有鲁棒性。

    

    使用第三方数据集和预训练的机器学习模型对NLP系统构成威胁，可能隐藏后门攻击。现有的攻击方式包括插入标记或句子重述等污染数据样本，这要么改变了原始文本的语义，要么可以被检测出来。我们与以往工作的主要区别在于，我们使用重新定位句子中的两个单词作为触发器。通过设计并应用基于词性的规则来选择这些标记，我们在SST-2和AG分类数据集上保持了高攻击成功率，同时在困惑度和与干净样本的语义相似性方面优于现有攻击方法。此外，我们展示了我们的攻击对ONION防御方法的鲁棒性。论文中的所有代码和数据可在https://github.com/alekseevskaia/OrderBkd获取。

    The use of third-party datasets and pre-trained machine learning models poses a threat to NLP systems due to possibility of hidden backdoor attacks. Existing attacks involve poisoning the data samples such as insertion of tokens or sentence paraphrasing, which either alter the semantics of the original texts or can be detected. Our main difference from the previous work is that we use the reposition of a two words in a sentence as a trigger. By designing and applying specific part-of-speech (POS) based rules for selecting these tokens, we maintain high attack success rate on SST-2 and AG classification datasets while outperforming existing attacks in terms of perplexity and semantic similarity to the clean samples. In addition, we show the robustness of our attack to the ONION defense method. All the code and data for the paper can be obtained at https://github.com/alekseevskaia/OrderBkd.
    
[^27]: 提升双线性语义依存解析的辅助任务

    Auxiliary Tasks to Boost Biaffine Semantic Dependency Parsing

    [https://arxiv.org/abs/2402.07682](https://arxiv.org/abs/2402.07682)

    本研究提出了一种简单而有效的方法来提高语义依存解析的性能，通过引入辅助任务以增加弧之间的相互依赖关系，该方法在实验中表现出了系统性的性能提升，并且具有良好的可扩展性。

    

    Dozat和Manning (2017)的双线性解析成功地扩展到语义依存解析(SDP) (Dozat和Manning, 2018)。鉴于没有树结构的约束，对于给定的句子，所有弧都是相互独立预测的（除了令牌的共享表示），然而其在图上的性能令人惊讶地很高。为了避免这种决策的独立性，同时保持O(n^2)的复杂度和高度可并行化的架构，我们提出使用简单的辅助任务，引入弧之间某种形式的相互依赖。在SemEval 2015任务18的三个英语非循环数据集(Oepen et al., 2015)和法语深度句法循环图(Ribeyre et al., 2014)上的实验表明，在使用基于Transformer的上下文化表示的接近最先进基线的基础上，虽然性能改进适中但系统性地提高了性能。这提供了一种简单而强大的提升SDP性能的方法。

    The biaffine parser of Dozat and Manning (2017) was successfully extended to semantic dependency parsing (SDP) (Dozat and Manning, 2018). Its performance on graphs is surprisingly high given that, without the constraint of producing a tree, all arcs for a given sentence are predicted independently from each other (modulo a shared representation of tokens). To circumvent such an independence of decision, while retaining the O(n^2) complexity and highly parallelizable architecture, we propose to use simple auxiliary tasks that introduce some form of interdependence between arcs. Experiments on the three English acyclic datasets of SemEval 2015 task 18 (Oepen et al., 2015), and on French deep syntactic cyclic graphs (Ribeyre et al., 2014) show modest but systematic performance gains on a near state-of-the-art baseline using transformer-based contextualized representations. This provides a simple and robust method to boost SDP performance.
    
[^28]: 大型语言模型“评审”: 在法律领域的机器翻译效果如何？

    Large Language Models "Ad Referendum": How Good Are They at Machine Translation in the Legal Domain?

    [https://arxiv.org/abs/2402.07681](https://arxiv.org/abs/2402.07681)

    本研究评估了两种大型语言模型和传统神经机器翻译系统在法律领域的机器翻译质量，结果显示大型语言模型在产生上下文足够且流畅的译文方面表现优异，强调了人工评估方法在评估机器翻译质量中的重要性。

    

    本研究评估了两种最先进的大型语言模型（LLMs）与传统神经机器翻译（NMT）系统在法律领域四种语言对中的机器翻译质量。研究结合了自动评估指标（AEMs）和专业翻译人员进行的人工评估（HE），评估了翻译排名、流畅性和足够性。结果表明，虽然谷歌翻译在AEMs方面通常优于LLMs，但人工评估者认为LLMs，特别是GPT-4，在产生上下文足够且流畅的译文方面相当或略好于谷歌翻译。这种差异表明LLMs在处理专业法律术语和背景方面的潜力，凸显了人工评估方法在评估机器翻译质量方面的重要性。本研究强调了LLMs在专业领域的不断发展能力，并呼吁重新评估传统AEMs以更好地捕捉LLMs生成的翻译的细微差别。

    This study evaluates the machine translation (MT) quality of two state-of-the-art large language models (LLMs) against a tradition-al neural machine translation (NMT) system across four language pairs in the legal domain. It combines automatic evaluation met-rics (AEMs) and human evaluation (HE) by professional transla-tors to assess translation ranking, fluency and adequacy. The re-sults indicate that while Google Translate generally outperforms LLMs in AEMs, human evaluators rate LLMs, especially GPT-4, comparably or slightly better in terms of producing contextually adequate and fluent translations. This discrepancy suggests LLMs' potential in handling specialized legal terminology and context, highlighting the importance of human evaluation methods in assessing MT quality. The study underscores the evolving capabil-ities of LLMs in specialized domains and calls for reevaluation of traditional AEMs to better capture the nuances of LLM-generated translations.
    
[^29]: 医疗保健之声：利用大型语言模型提高医学转录ASR准确性

    The Sound of Healthcare: Improving Medical Transcription ASR Accuracy with Large Language Models

    [https://arxiv.org/abs/2402.07658](https://arxiv.org/abs/2402.07658)

    本研究探索了利用大型语言模型（LLMs）提高医学转录中自动语音识别（ASR）系统准确性的潜力，并通过实验比较了零-shot和Chain-of-Thought（CoT）提示技术的有效性。

    

    在医疗文档的快速发展环境中，准确转录临床对话变得日益重要。这项研究探索了利用大型语言模型（LLMs）提高医学转录中自动语音识别（ASR）系统准确性的潜力。利用PriMock57数据集，该数据集包含了各种不同的初级护理咨询，我们采用先进的LLMs来优化ASR生成的转录。我们的研究是多方面的，关注于改进一般的词错误率（WER），医学概念错误率（MC-WER）以准确转录重要的医学术语，以及说话人重音划分准确性。此外，我们还评估了LLM后处理在改进语义文本相似性方面的作用，从而保持临床对话的上下文完整性。通过一系列实验证明，我们比较了零-shot和Chain-of-Thought（CoT）提示技术在提高重音划分和纠正准确性方面的效果。

    In the rapidly evolving landscape of medical documentation, transcribing clinical dialogues accurately is increasingly paramount. This study explores the potential of Large Language Models (LLMs) to enhance the accuracy of Automatic Speech Recognition (ASR) systems in medical transcription. Utilizing the PriMock57 dataset, which encompasses a diverse range of primary care consultations, we apply advanced LLMs to refine ASR-generated transcripts. Our research is multifaceted, focusing on improvements in general Word Error Rate (WER), Medical Concept WER (MC-WER) for the accurate transcription of essential medical terms, and speaker diarization accuracy. Additionally, we assess the role of LLM post-processing in improving semantic textual similarity, thereby preserving the contextual integrity of clinical dialogues. Through a series of experiments, we compare the efficacy of zero-shot and Chain-of-Thought (CoT) prompting techniques in enhancing diarization and correction accuracy. Our fi
    
[^30]: 使用大型语言模型从合成数据中检测难治性抑郁症的临床特征

    Detecting the Clinical Features of Difficult-to-Treat Depression using Synthetic Data from Large Language Models

    [https://arxiv.org/abs/2402.07645](https://arxiv.org/abs/2402.07645)

    本研究开发了基于大型语言模型的工具，使用合成数据和片段提取模型从临床数据中提取关于难治性抑郁症的特征，证明了在真实临床数据中取得了良好的整体性能。

    

    难治性抑郁症(DTD)被提出作为一个更广泛且临床更全面的视角，该视角表明在治疗过程中，患者仍持续经历显著负担。我们致力于开发一种基于大型语言模型(LLM)的工具，能够对常规收集的叙述性（自由文本）电子健康记录(EHR)数据进行查询，以找到能捕捉DTD临床综合征的已发表预后因素。在这项工作中，我们使用LLM生成的合成数据(GPT3.5)和非极大值抑制(NMS)算法来训练一个基于BERT的片段提取模型。然后，生成的模型能够从真实的临床数据中提取和标记与DTD综合征的匹配可能性增加或减少的各种相关正负因素的片段。我们展示了在多达20个DTD临床数据集上能够获得良好的整体性能(极性为0.70 F1)。

    Difficult-to-treat depression (DTD) has been proposed as a broader and more clinically comprehensive perspective on a person's depressive disorder where despite treatment, they continue to experience significant burden. We sought to develop a Large Language Model (LLM)-based tool capable of interrogating routinely-collected, narrative (free-text) electronic health record (EHR) data to locate published prognostic factors that capture the clinical syndrome of DTD. In this work, we use LLM-generated synthetic data (GPT3.5) and a Non-Maximum Suppression (NMS) algorithm to train a BERT-based span extraction model. The resulting model is then able to extract and label spans related to a variety of relevant positive and negative factors in real clinical data (i.e. spans of text that increase or decrease the likelihood of a patient matching the DTD syndrome). We show it is possible to obtain good overall performance (0.70 F1 across polarity) on real clinical data on a set of as many as 20 diff
    
[^31]: AutoMathText：使用语言模型进行数学文本的自主数据选择

    AutoMathText: Autonomous Data Selection with Language Models for Mathematical Texts

    [https://arxiv.org/abs/2402.07625](https://arxiv.org/abs/2402.07625)

    本论文介绍了一种自主数据选择策略，利用语言模型进行数学文本的自动评估和选择，并通过连续预训练显著提高了数学推理能力。主要创新包括利用元提示语言模型作为验证器，发布了高质量的AutoMathText数据集，并实现了预训练令牌效率的提升。

    

    为了通过持续的预训练改善语言模型在数学推理方面的能力，我们引入了一种新颖的策略，利用基础语言模型进行自主数据选择。与传统的有人工标注数据的监督微调或训练过的分类器不同，我们的方法利用元提示语言模型作为零样本验证器，自主评估和选择高质量的数学内容，并发布了经过策划的开源AutoMathText数据集，其中包含超过200GB的数据。为了证明我们方法的有效性，我们对AutoMathText数据集进行了连续预训练，使得7B参数的Mistral语言模型在MATH数据集上的下游性能大幅提升，而令牌数量比之前的连续预训练工作减少了几个数量级。我们的方法展示了基准的预训练令牌效率提高了2倍，突显了我们方法在增强中的潜力。

    To improve language models' proficiency in mathematical reasoning via continual pretraining, we introduce a novel strategy that leverages base language models for autonomous data selection. Departing from conventional supervised fine-tuning or trained classifiers with human-annotated data, our approach utilizes meta-prompted language models as zero-shot verifiers to autonomously evaluate and select high-quality mathematical content, and we release the curated open-source AutoMathText dataset encompassing over 200GB of data. To demonstrate the efficacy of our method, we continuously pretrained a 7B-parameter Mistral language model on the AutoMathText dataset, achieving substantial improvements in downstream performance on the MATH dataset with a token amount reduced by orders of magnitude compared to previous continuous pretraining works. Our method showcases a 2 times increase in pretraining token efficiency compared to baselines, underscoring the potential of our approach in enhancing
    
[^32]: 基于锚点的大型语言模型

    Anchor-based Large Language Models

    [https://arxiv.org/abs/2402.07616](https://arxiv.org/abs/2402.07616)

    基于锚点的大型语言模型（AnLLM）通过引入创新的基于锚点的自注意力网络（AnSAN）和基于锚点的推理策略，将序列信息压缩到锚点标记中，减少键/值缓存，提高推理效率。

    

    大型语言模型（LLMs）主要采用仅解码器的转换器架构，需要保留历史标记的键/值信息以提供上下文信息并避免冗余计算。然而，这些LLMs的巨大大小和参数量需要大量的GPU内存。这种内存需求随着输入文本的长度而增加，迫切需要更高效的信息存储和处理方法。本研究介绍了一种基于锚点的LLM（AnLLM），它利用了一种创新的基于锚点的自注意力网络（AnSAN）和基于锚点的推理策略。这种方法使LLMs能够将序列信息压缩成锚点标记，减少键/值缓存并提高推理效率。实验证明，AnLLM在减少键/值缓存高达99%和推理速度提高高达3.5倍的同时，仍保持可比的准确性。尽管牺牲了一些准确性，AnLLM的创新和贡献依然重要。

    Large language models (LLMs) predominantly employ decoder-only transformer architectures, necessitating the retention of keys/values information for historical tokens to provide contextual information and avoid redundant computation. However, the substantial size and parameter volume of these LLMs require massive GPU memory. This memory demand increases with the length of the input text, leading to an urgent need for more efficient methods of information storage and processing. This study introduces the Anchor-based LLM (AnLLM), which utilizes an innovative anchor-based self-attention network (AnSAN) and also an anchor-based inference strategy. This approach enables LLMs to compress sequence information into an anchor token, reducing the keys/values cache and enhancing inference efficiency. Experiments show that the AnLLM maintains comparable accuracy with up to 99% keys/values cache reduction and up to 3.5 times faster inference. Despite a minor compromise in accuracy, the AnLLM signi
    
[^33]: 踩脚调校：通过自助引导扩展LLM的自对齐能力的规模化方法

    Step-On-Feet Tuning: Scaling Self-Alignment of LLMs via Bootstrapping

    [https://arxiv.org/abs/2402.07610](https://arxiv.org/abs/2402.07610)

    本文首次探索了自助引导自对齐对大型语言模型的影响，发现其明显优于单次循环的方法，并通过调整数据训练顺序进一步提升模型性能。

    

    自对齐是一种降低人工注释成本并确保模型能力的有效方法。然而，大多数当前的方法在单次循环中完成数据收集和训练步骤，可能忽视了自对齐模型不断改进的能力。这引发了一个关键问题：如果我们进行多次自助引导自对齐，会增强模型性能还是导致快速退化？本文首次探索了自助引导自对齐对大型语言模型的影响。我们的研究结果表明，通过保证从上下文学习中获得的数据多样性，自助引导自对齐明显优于单次循环的方法。为了进一步发挥自助引导的能力，我们还研究并调整了数据的训练顺序，从而提高了模型的性能。基于这些发现，我们提出了踩脚调校（SOFT）的方法，利用模型的持续增强能力。

    Self-alignment is an effective way to reduce the cost of human annotation while ensuring promising model capability. However, most current methods complete the data collection and training steps in a single round, which may overlook the continuously improving ability of self-aligned models. This gives rise to a key query: What if we do multi-time bootstrapping self-alignment? Does this strategy enhance model performance or lead to rapid degradation? In this paper, our pioneering exploration delves into the impact of bootstrapping self-alignment on large language models. Our findings reveal that bootstrapping self-alignment markedly surpasses the single-round approach, by guaranteeing data diversity from in-context learning. To further exploit the capabilities of bootstrapping, we investigate and adjust the training order of data, which yields improved performance of the model. Drawing on these findings, we propose Step-On-Feet Tuning (SOFT) which leverages model's continuously enhanced
    
[^34]: 主题建模作为多目标对比优化方法

    Topic Modeling as Multi-Objective Contrastive Optimization

    [https://arxiv.org/abs/2402.07577](https://arxiv.org/abs/2402.07577)

    该论文介绍了一种新颖的主题建模方法，通过优化对数似然的证据下界和对比学习目标的加权线性组合，将对比主题建模作为一种多目标优化问题，旨在获得能够捕捉共享语义并克服低级别互信息干扰的主题向量集合。

    

    最近的表示学习方法通过优化对数似然的证据下界（ELBO）和对比学习目标的加权线性组合来增强神经主题模型。然而，文档级对比学习可能捕捉到低级别的互信息，例如词比例，这会干扰主题建模。此外，ELBO损失旨在记忆输入细节以获得更好的重构质量，而对比损失则试图学习在输入文档之间泛化的主题表示，二者存在潜在冲突。为了解决这些问题，首先我们引入了一种新颖的面向主题向量集合的对比学习方法，以捕捉一组输入文档之间共享的有用语义。其次，我们将对比主题建模明确提出为一个基于梯度的多目标优化问题，目标是实现帕累托平稳解决方案。

    Recent representation learning approaches enhance neural topic models by optimizing the weighted linear combination of the evidence lower bound (ELBO) of the log-likelihood and the contrastive learning objective that contrasts pairs of input documents. However, document-level contrastive learning might capture low-level mutual information, such as word ratio, which disturbs topic modeling. Moreover, there is a potential conflict between the ELBO loss that memorizes input details for better reconstruction quality, and the contrastive loss which attempts to learn topic representations that generalize among input documents. To address these issues, we first introduce a novel contrastive learning method oriented towards sets of topic vectors to capture useful semantics that are shared among a set of input documents. Secondly, we explicitly cast contrastive topic modeling as a gradient-based multi-objective optimization problem, with the goal of achieving a Pareto stationary solution that b
    
[^35]: 给我看怎么做：解释在细调语言模型中的作用

    Show Me How It's Done: The Role of Explanations in Fine-Tuning Language Models

    [https://arxiv.org/abs/2402.07543](https://arxiv.org/abs/2402.07543)

    本研究证明了使用解释来改进语言模型性能的显著好处，尤其适用于较小的模型，解释的加入使模型能够解决之前无法解决的任务。

    

    我们的研究证明了使用解释来增强语言模型性能的显著好处。与提示方式不同，细调允许模型在训练阶段学习和更新参数。在本研究中，我们应用细调的方法，使用包含输出解释而非仅呈现答案的数据来对不同大小的语言模型进行训练。我们发现，即使是只有6000万参数的较小语言模型也能从这种方法中获益。有趣的是，我们的结果表明，详细的解释对较小的模型更有益处，而对于较大的模型来说，无论解释的长度如何，都可以获得几乎相同的优势。此外，我们还证明了解释的加入使模型能够解决之前无法解决的任务。最后，我们认为尽管存在挑战，但解释在细调语言模型中起到了重要作用。

    Our research demonstrates the significant benefits of using fine-tuning with explanations to enhance the performance of language models. Unlike prompting, which maintains the model's parameters, fine-tuning allows the model to learn and update its parameters during a training phase. In this study, we applied fine-tuning to various sized language models using data that contained explanations of the output rather than merely presenting the answers. We found that even smaller language models with as few as 60 million parameters benefited substantially from this approach. Interestingly, our results indicated that the detailed explanations were more beneficial to smaller models than larger ones, with the latter gaining nearly the same advantage from any form of explanation, irrespective of its length. Additionally, we demonstrate that the inclusion of explanations enables the models to solve tasks that they were not able to solve without explanations. Lastly, we argue that despite the chall
    
[^36]: PKG API：一个个人知识图管理工具

    PKG API: A Tool for Personal Knowledge Graph Management

    [https://arxiv.org/abs/2402.07540](https://arxiv.org/abs/2402.07540)

    本文提出了一个完整的个人知识图（PKG）管理解决方案，包括用户界面友好的PKG客户端和面向服务的PKG API，以及基于RDF的PKG词汇表用于表示陈述和访问权限。

    

    个人知识图（PKG）为个人提供了一种将碎片化的个人数据存储和整合到一个中心位置的方式，提高了服务的个性化程度同时保持用户的完全控制。尽管PKG的潜力巨大，但实际操作上具有用户友好界面的PKG实现仍然很少。本文通过提出一个完整的解决方案来表示、管理和与PKG进行交互来填补这一空白。我们的方法包括（1）一个面向用户的PKG客户端，使最终用户可以通过自然语言陈述轻松管理他们的个人数据，以及（2）一个面向服务的PKG API。为了应对在PKG中表示这些陈述的复杂性，我们提出了一种基于RDF的PKG词汇表来支持这一点，并提供了访问权限和来源的属性。

    Personal knowledge graphs (PKGs) offer individuals a way to store and consolidate their fragmented personal data in a central place, improving service personalization while maintaining full user control. Despite their potential, practical PKG implementations with user-friendly interfaces remain scarce. This work addresses this gap by proposing a complete solution to represent, manage, and interface with PKGs. Our approach includes (1) a user-facing PKG Client, enabling end-users to administer their personal data easily via natural language statements, and (2) a service-oriented PKG API. To tackle the complexity of representing these statements within a PKG, we present an RDF-based PKG vocabulary that supports this, along with properties for access rights and provenance.
    
[^37]: MAFIA: 多适配器融合的包容性语言模型

    MAFIA: Multi-Adapter Fused Inclusive LanguAge Models

    [https://arxiv.org/abs/2402.07519](https://arxiv.org/abs/2402.07519)

    本文提出了一种名为MAFIA的多适配器融合的包容性语言模型，在多个偏见维度上进行模块化去偏倚，利用结构化知识和大规模生成模型构建了多样化的反事实数据增强，并强调了现有去偏倚方法对多个社会偏见之间的相互作用缺乏考虑。

    

    预训练语言模型（PLMs）被广泛应用于自然语言处理的各种任务中。最近的研究发现这些模型存在各种偏见，并提出方法来纠正这些偏见。然而，大多数工作仅独立地解决了有限的偏见维度，如性别、种族或宗教。此外，这些方法通常需要对整个模型进行微调以保持在下游任务上的性能。在本文中，我们旨在在多个维度上模块化地去偏倚预训练语言模型。先前的工作已经广泛探索了使用有限的美国中心的反事实数据增强（CDA）来去偏倚PLMs。我们使用结构化知识和大规模生成模型来以半自动化的方式构建多个偏见维度上的多样化CDA。我们强调现有的去偏倚方法不考虑多个社会偏见之间的相互作用，并提出了一种能够利用各种社会偏见之间的协同效应并实现多重偏见的去偏倚模型。

    Pretrained Language Models (PLMs) are widely used in NLP for various tasks. Recent studies have identified various biases that such models exhibit and have proposed methods to correct these biases. However, most of the works address a limited set of bias dimensions independently such as gender, race, or religion. Moreover, the methods typically involve finetuning the full model to maintain the performance on the downstream task. In this work, we aim to modularly debias a pretrained language model across multiple dimensions. Previous works extensively explored debiasing PLMs using limited US-centric counterfactual data augmentation (CDA). We use structured knowledge and a large generative model to build a diverse CDA across multiple bias dimensions in a semi-automated way. We highlight how existing debiasing methods do not consider interactions between multiple societal biases and propose a debiasing model that exploits the synergy amongst various societal biases and enables multi-bias 
    
[^38]: 平衡的艺术：揭示和缓解葡萄牙语中的ASR偏见

    The Balancing Act: Unmasking and Alleviating ASR Biases in Portuguese

    [https://arxiv.org/abs/2402.07513](https://arxiv.org/abs/2402.07513)

    本研究通过对Whisper和MMS系统进行全面探索，评估了葡萄牙语中非正式对话语音的自动语音识别（ASR）偏见，并发现采用过采样技术可以缓解这种陈规定型偏见。

    

    在口语理解领域，像Whisper和Multilingual Massive Speech（MMS）这样的系统展示了最先进的性能。本研究致力于对Whisper和MMS系统进行全面探索，重点评估与葡萄牙语特定的非正式对话语音中的自动语音识别（ASR）偏见。我们的调查涵盖了各种类别，包括性别、年龄、肤色和地理位置。除了传统的ASR评估指标，如词错误率（WER），我们还使用p值统计显著性来分析性别偏见。此外，我们广泛研究了数据分布的影响，并从实证角度表明过采样技术缓解了这种陈规定型偏见。本研究通过MMS和Whisper的应用，在葡萄牙语环境中量化偏见方面做出了开创性的努力，为更好地理解ASR系统做出了贡献。

    In the field of spoken language understanding, systems like Whisper and Multilingual Massive Speech (MMS) have shown state-of-the-art performances. This study is dedicated to a comprehensive exploration of the Whisper and MMS systems, with a focus on assessing biases in automatic speech recognition (ASR) inherent to casual conversation speech specific to the Portuguese language. Our investigation encompasses various categories, including gender, age, skin tone color, and geo-location. Alongside traditional ASR evaluation metrics such as Word Error Rate (WER), we have incorporated p-value statistical significance for gender bias analysis. Furthermore, we extensively examine the impact of data distribution and empirically show that oversampling techniques alleviate such stereotypical biases. This research represents a pioneering effort in quantifying biases in the Portuguese language context through the application of MMS and Whisper, contributing to a better understanding of ASR systems
    
[^39]: T-RAG: 来自LLM战场的经验教训

    T-RAG: Lessons from the LLM Trenches

    [https://arxiv.org/abs/2402.07483](https://arxiv.org/abs/2402.07483)

    T-RAG是一个基于LLM的应用程序，用于私人企业文件问答，它结合了RAG框架和经过微调的开源LLM，并分享了构建和部署过程中的经验教训。

    

    大型语言模型（LLM）展示了惊人的语言能力，推动了将它们整合到各个领域的应用的尝试。一个重要的应用领域是对私人企业文件进行问答，其中主要考虑因素是数据安全，需要能够在本地部署的应用程序，有限的计算资源和对查询正确响应的健壮应用的需求。检索增强生成（RAG）已成为构建基于LLM的应用程序的最重要的框架。虽然构建RAG相对简单，但要使其健壮和可靠的应用程序需要广泛的定制化和相对深入的应用领域知识。我们分享了构建和部署一个基于LLM的私人组织文件问答应用的经验。我们的应用结合了RAG的使用和经过微调的开源LLM。此外，我们的系统还具有 ...

    Large Language Models (LLM) have shown remarkable language capabilities fueling attempts to integrate them into applications across a wide range of domains. An important application area is question answering over private enterprise documents where the main considerations are data security, which necessitates applications that can be deployed on-prem, limited computational resources and the need for a robust application that correctly responds to queries. Retrieval-Augmented Generation (RAG) has emerged as the most prominent framework for building LLM-based applications. While building a RAG is relatively straightforward, making it robust and a reliable application requires extensive customization and relatively deep knowledge of the application domain. We share our experiences building and deploying an LLM application for question answering over private organizational documents. Our application combines the use of RAG with a finetuned open-source LLM. Additionally, our system, which w
    
[^40]: 推动文本分类中LLM容量的极限

    Pushing The Limit of LLM Capacity for Text Classification

    [https://arxiv.org/abs/2402.07470](https://arxiv.org/abs/2402.07470)

    本论文提出了一个自适应增强框架RGPT，通过反复集成强基学习者，生成一个专用的文本分类LLM。通过实证比较，我们展示了RGPT明显胜过其他方法。

    

    由于大型语言模型（LLM）在众多下游NLP任务中展示出的非凡效果，文本分类未来研究的价值面临着挑战和不确定性。在这个任务边界逐渐模糊的开放式语言建模时代，一个迫切的问题出现了：在充分利用LLM的情况下，我们在文本分类方面取得了重大进展吗？为了回答这个问题，我们提出了RGPT，一个自适应增强框架，旨在通过反复集成一组强基学习者，来生成一个专用的文本分类LLM。基学习者是通过自适应调整训练样本的分布，并反复微调LLM与之构建的。然后，这些基学习者通过反复融合前几个学习者的历史预测结果，形成一个专用的文本分类LLM。通过全面的实证比较，我们展示了RGPT明显胜过其他方法。

    The value of text classification's future research has encountered challenges and uncertainties, due to the extraordinary efficacy demonstrated by large language models (LLMs) across numerous downstream NLP tasks. In this era of open-ended language modeling, where task boundaries are gradually fading, an urgent question emerges: have we made significant advances in text classification under the full benefit of LLMs? To answer this question, we propose RGPT, an adaptive boosting framework tailored to produce a specialized text classification LLM by recurrently ensembling a pool of strong base learners. The base learners are constructed by adaptively adjusting the distribution of training samples and iteratively fine-tuning LLMs with them. Such base learners are then ensembled to be a specialized text classification LLM, by recurrently incorporating the historical predictions from the previous learners. Through a comprehensive empirical comparison, we show that RGPT significantly outperf
    
[^41]: AraSpider：实现阿拉伯语到SQL的民主化

    AraSpider: Democratizing Arabic-to-SQL

    [https://arxiv.org/abs/2402.07448](https://arxiv.org/abs/2402.07448)

    AraSpider是首个阿拉伯语版本的Spider数据集，研究表明使用回译策略可以显著提高ChatGPT 3.5和SQLCoder模型在阿拉伯语NLP任务中的性能。

    

    本研究介绍了AraSpider，这是首个阿拉伯语版本的Spider数据集，旨在提升阿拉伯语社区中的自然语言处理（NLP）。该研究测试了四个多语言翻译模型在将英文翻译成阿拉伯语方面的有效性。另外，还评估了两个模型在从阿拉伯文本生成SQL查询方面的能力。结果表明，使用回译显著提高了ChatGPT 3.5和SQLCoder模型的表现，这两个模型在Spider数据集上被认为是最佳表现者。值得注意的是，ChatGPT 3.5展示了高质量的翻译，而SQLCoder在文本到SQL任务中表现出色。该研究强调了将上下文模式和采用回译策略纳入阿拉伯语NLP任务中以提高模型性能的重要性。此外，提供了详细的方法论以实现结果复现并将数据集翻译成其他语言，突显了研究促进的承诺。

    This study presents AraSpider, the first Arabic version of the Spider dataset, aimed at improving natural language processing (NLP) in the Arabic-speaking community. Four multilingual translation models were tested for their effectiveness in translating English to Arabic. Additionally, two models were assessed for their ability to generate SQL queries from Arabic text. The results showed that using back translation significantly improved the performance of both ChatGPT 3.5 and SQLCoder models, which are considered top performers on the Spider dataset. Notably, ChatGPT 3.5 demonstrated high-quality translation, while SQLCoder excelled in text-to-SQL tasks. The study underscores the importance of incorporating contextual schema and employing back translation strategies to enhance model performance in Arabic NLP tasks. Moreover, the provision of detailed methodologies for reproducibility and translation of the dataset into other languages highlights the research's commitment to promoting 
    
[^42]: 质量确实重要：对网络挖掘平行语料库的质量和实用性进行详细研究

    Quality Does Matter: A Detailed Look at the Quality and Utility of Web-Mined Parallel Corpora

    [https://arxiv.org/abs/2402.07446](https://arxiv.org/abs/2402.07446)

    这项研究详细分析了网络挖掘语料库的质量和实用性，并发现不同语言和数据集之间存在显著的质量差异。同时，我们还展示了某些网络挖掘数据集的最佳部分训练的神经机器翻译模型可以与人工策划的数据集持平。

    

    我们对两种低资源语言（英文-僧伽罗语，英文-泰米尔语和僧伽罗语-泰米尔语）的网络挖掘语料库的质量进行了详细分析。我们根据相似度标准对每个语料库进行了排名，并对排名语料库的不同部分进行内在和外在评估。我们显示不同部分的网络挖掘语料库存在显著的质量差异，并且质量在不同语言和数据集之间存在变化。我们还表明，对于某些网络挖掘数据集，使用其排名最高的25k部分训练的神经机器翻译（NMT）模型可以与人工策划的数据集持平。

    We conducted a detailed analysis on the quality of web-mined corpora for two low-resource languages (making three language pairs, English-Sinhala, English-Tamil and Sinhala-Tamil). We ranked each corpus according to a similarity measure and carried out an intrinsic and extrinsic evaluation on different portions of this ranked corpus. We show that there are significant quality differences between different portions of web-mined corpora and that the quality varies across languages and datasets. We also show that, for some web-mined datasets, Neural Machine Translation (NMT) models trained with their highest-ranked 25k portion can be on par with human-curated datasets.
    
[^43]: 内在任务驱动的分发表达生成评估

    Intrinsic Task-based Evaluation for Referring Expression Generation

    [https://arxiv.org/abs/2402.07432](https://arxiv.org/abs/2402.07432)

    该论文提出了一种内在任务驱动的评估方法，用于评估分发表达生成（REG）模型。该方法不仅评估了分发表达的质量，还通过两个元级任务评估了模型的引用成功程度和提出替代方案的能力。

    

    最近，对于分发表达生成（REG）模型的人工评估研究得出了一个令人意外的结论：在\textsc{webnlg}上，最先进的神经模型生成的分发表达（REs）不仅与\textsc{webnlg}中的REs无法区分，而且与简单的基于规则的系统生成的REs也无法区分。在这里，我们认为这个局限可能源于纯评分的人工评估方法（这是自然语言生成中的常见实践）。为了调查这些问题，我们提出了一种针对REG模型的内在任务驱动评估方法，除了评估REs的质量外，参与者还需要完成两个元级任务。其中一个任务涉及每个RE的引用成功程度，另一个任务要求参与者为每个RE提出更好的替代方案。结果表明，与之前的评估相比，新的评估协议更全面地评估了每个REG模型的性能。

    Recently, a human evaluation study of Referring Expression Generation (REG) models had an unexpected conclusion: on \textsc{webnlg}, Referring Expressions (REs) generated by the state-of-the-art neural models were not only indistinguishable from the REs in \textsc{webnlg} but also from the REs generated by a simple rule-based system. Here, we argue that this limitation could stem from the use of a purely ratings-based human evaluation (which is a common practice in Natural Language Generation). To investigate these issues, we propose an intrinsic task-based evaluation for REG models, in which, in addition to rating the quality of REs, participants were asked to accomplish two meta-level tasks. One of these tasks concerns the referential success of each RE; the other task asks participants to suggest a better alternative for each RE. The outcomes suggest that, in comparison to previous evaluations, the new evaluation protocol assesses the performance of each REG model more comprehensive
    
[^44]: SALAD: 智能AI语言助手日常

    SALAD: Smart AI Language Assistant Daily

    [https://arxiv.org/abs/2402.07431](https://arxiv.org/abs/2402.07431)

    SALAD是一款智能AI语言助手应用，旨在帮助外国人学习日语。它提供了多种学习工具和功能，包括翻译，语音识别，音频翻译，词汇跟踪等，并通过每日翻译帮助提高与母语人士的交流能力。调查结果显示60%的外国人对SALAD提升日语能力有信心。该应用利用大型语言模型和扩散模型促进日本社区的包容性。

    

    SALAD是一款由AI驱动的语言学习应用程序，旨在帮助外国人学习日语。它提供了汉字-假名-罗马字的翻译，语音识别，翻译音频，词汇跟踪，语法解释，以及由新学到的词汇生成的歌曲。该应用针对初学者和中级学习者，旨在使语言习得更加可获得和愉快。SALAD利用每日翻译来增强与母语人士的流利度和交流舒适度。主要目标包括有效的日语学习，用户参与度和进展跟踪。我们的调查发现，在日本的外国人中，有39%在与日本人交谈时感到不适。超过60%的外国人表示对SALAD提升他们的日语能力有信心。该应用使用大型语言模型，语音识别和扩散模型来弥合语言隔阂，促进日本更具包容性的社区。

    SALAD is an AI-driven language-learning application designed to help foreigners learn Japanese. It offers translations in Kanji-Kana-Romaji, speech recognition, translated audio, vocabulary tracking, grammar explanations, and songs generated from newly learned words. The app targets beginners and intermediate learners, aiming to make language acquisition more accessible and enjoyable. SALAD uses daily translations to enhance fluency and comfort in communication with native speakers. The primary objectives include effective Japanese language learning, user engagement, and progress tracking. A survey by us found that 39% of foreigners in Japan face discomfort in conversations with Japanese speakers. Over 60% of foreigners expressed confidence in SALAD's ability to enhance their Japanese language skills. The app uses large language models, speech recognition, and diffusion models to bridge the language gap and foster a more inclusive community in Japan.
    
[^45]: D'olares还是Dollars？揭示西班牙语和英语财经研究与应用中双语能力的差异

    D\'olares or Dollars? Unraveling the Bilingual Prowess of Financial LLMs Between Spanish and English

    [https://arxiv.org/abs/2402.07405](https://arxiv.org/abs/2402.07405)

    这项研究推出了第一个双语框架Tois'on de Oro，用于运用在西班牙语和英语金融领域的大型语言模型（LLMs），通过构建双语指导数据集和评估基准进行了实证研究。研究结果表明现有LLMs在多语言性能上存在差距和偏见，而作者提出的FinMA-ES模型在西班牙语中超越了现有SOTA LLMs的表现。

    

    尽管西班牙语在全球金融行业中具有重要作用，但与英语相比，西班牙语金融自然语言处理（NLP）和应用研究存在明显差距，尤其在大型语言模型（LLMs）时代。为了弥补这一差距，我们推出了Tois'on de Oro，这是第一个建立指导数据集、经过微调的双语金融LLMs和评估基准的双语框架，用于西班牙语和英语的金融LLMs。我们构建了一个严格筛选的双语指导数据集，包括来自15个数据集的超过144K个西班牙语和英语样本，涵盖7个任务。利用这个数据集，我们引入了FinMA-ES，一种专为双语金融应用设计的LLM。我们使用FLARE-ES进行了模型和现有LLMs的评估，FLARE-ES 是第一个全面评估双语性能的评估基准，涵盖了21个数据集和9个任务。FLARE-ES基准结果显示现有LLMs存在显著的多语言性能差距和偏见。FinMA-ES模型在西班牙语中超过了GPT-4等SOTA LLMs的表现。

    Despite Spanish's pivotal role in the global finance industry, a pronounced gap exists in Spanish financial natural language processing (NLP) and application studies compared to English, especially in the era of large language models (LLMs). To bridge this gap, we unveil Tois\'on de Oro, the first bilingual framework that establishes instruction datasets, finetuned LLMs, and evaluation benchmark for financial LLMs in Spanish joint with English. We construct a rigorously curated bilingual instruction dataset including over 144K Spanish and English samples from 15 datasets covering 7 tasks. Harnessing this, we introduce FinMA-ES, an LLM designed for bilingual financial applications. We evaluate our model and existing LLMs using FLARE-ES, the first comprehensive bilingual evaluation benchmark with 21 datasets covering 9 tasks. The FLARE-ES benchmark results reveal a significant multilingual performance gap and bias in existing LLMs. FinMA-ES models surpass SOTA LLMs such as GPT-4 in Spani
    
[^46]: 能够为事实核查提供忠实解释吗？通过多智能体辩论实现忠实可解释的事实核查

    Can LLMs Produce Faithful Explanations For Fact-checking? Towards Faithful Explainable Fact-Checking via Multi-Agent Debate

    [https://arxiv.org/abs/2402.07401](https://arxiv.org/abs/2402.07401)

    本研究调查了大型语言模型在事实核查中生成忠实解释的能力，并发现了零-shot提示常常导致不忠实的结果。为了解决这个问题，我们提出了多智能体辩论优化（MADR）框架，通过迭代的优化过程，利用多个大型语言模型作为代理人，从而显著提高了生成解释的忠实性。

    

    事实核查研究广泛探讨了验证方法，但对于生成自然语言解释的研究相对较少，而这对于用户的信任至关重要。虽然大型语言模型在文本生成方面表现出色，但它们在事实核查中生成忠实解释的能力仍未得到充分研究。我们的研究调查了大型语言模型生成这种解释的能力，发现零-shot提示往往导致不忠实的结果。为了解决这些挑战，我们提出了多智能体辩论优化（MADR）框架，利用多个大型语言模型作为代理人，在迭代的优化过程中发挥各自不同的角色，目标是增强生成解释的忠实性。MADR确保最终解释经过严格验证，显著减少了不忠实因素的可能性，并与提供的证据密切对齐。实验结果表明，MADR显著提高了大型语言模型生成的解释与证据的一致性，推动了可信的事实核查方法的发展。

    Fact-checking research has extensively explored verification but less so the generation of natural-language explanations, crucial for user trust. While Large Language Models (LLMs) excel in text generation, their capability for producing faithful explanations in fact-checking remains underexamined. Our study investigates LLMs' ability to generate such explanations, finding that zero-shot prompts often result in unfaithfulness. To address these challenges, we propose the Multi-Agent Debate Refinement (MADR) framework, leveraging multiple LLMs as agents with diverse roles in an iterative refining process aimed at enhancing faithfulness in generated explanations. MADR ensures that the final explanation undergoes rigorous validation, significantly reducing the likelihood of unfaithful elements and aligning closely with the provided evidence. Experimental results demonstrate that MADR significantly improves the faithfulness of LLM-generated explanations to the evidence, advancing the credib
    
[^47]: 利用人工智能推进非洲科学和计算教育：进展、挑战和机遇

    Leveraging AI to Advance Science and Computing Education across Africa: Progress, Challenges, and Opportunities

    [https://arxiv.org/abs/2402.07397](https://arxiv.org/abs/2402.07397)

    这项研究描述了在非洲开发和使用人工智能教育工具的工作，包括SuaCode学习编码应用、AutoGrad自动评分和反馈工具、代码抄袭检测工具以及双语AI教师Kwame。这些工具有助于解决非洲学生在教育中面临的挑战。

    

    在非洲大陆，学生们面临着各种教育挑战，包括获取计算机、网络连接、可靠电力和合格教师等基本资源的限制。尽管存在这些挑战，但最近人工智能（如BERT和GPT-4）的进展已经展示了其促进教育的潜力。然而，这些人工智能工具往往在西方教育环境中进行部署和评估，对非洲学生面临的独特需求和挑战的关注有限。在本章中，我们描述了我们在非洲开发和部署人工智能教育工具的工作：（1）SuaCode，一款AI动力的应用程序，使非洲人可以使用智能手机学习编程，（2）AutoGrad，用于图形和交互式编程作业的自动评分和反馈工具，（3）一种代码抄袭检测工具，展示了抄袭的可视证据，（4）Kwame，一款双语的AI教师。

    Across the African continent, students grapple with various educational challenges, including limited access to essential resources such as computers, internet connectivity, reliable electricity, and a shortage of qualified teachers. Despite these challenges, recent advances in AI such as BERT, and GPT-4 have demonstrated their potential for advancing education. Yet, these AI tools tend to be deployed and evaluated predominantly within the context of Western educational settings, with limited attention directed towards the unique needs and challenges faced by students in Africa. In this book chapter, we describe our works developing and deploying AI in Education tools in Africa: (1) SuaCode, an AI-powered app that enables Africans to learn to code using their smartphones, (2) AutoGrad, an automated grading, and feedback tool for graphical and interactive coding assignments, (3) a tool for code plagiarism detection that shows visual evidence of plagiarism, (4) Kwame, a bilingual AI teac
    
[^48]: Chain-of-Layer：通过有限示例迭代引导大型语言模型进行分类体系归纳

    Chain-of-Layer: Iteratively Prompting Large Language Models for Taxonomy Induction from Limited Examples

    [https://arxiv.org/abs/2402.07386](https://arxiv.org/abs/2402.07386)

    本文介绍了一种称为Chain-of-Layer的上下文学习框架，用于从给定的实体集中归纳分类体系。通过引入基于集成的排名过滤器来减少错误，Chain-of-Layer在四个实际基准测试中实现了最先进的性能。

    

    自动分类体系归纳对于网络搜索、推荐系统和问答系统非常重要。手动整理分类体系需要大量人力成本，因此自动构建分类体系非常有需求。本文介绍了一种称为Chain-of-Layer的上下文学习框架，用于从给定的实体集中归纳分类体系。Chain-of-Layer将任务分解为每一层选择相关候选实体，并逐步从上到下构建分类体系。为了减少错误，我们引入了基于集成的排名过滤器，在每一次迭代中减少生成的虚构内容。通过大量实验证明，Chain-of-Layer在四个实际基准测试中实现了最先进的性能。

    Automatic taxonomy induction is crucial for web search, recommendation systems, and question answering. Manual curation of taxonomies is expensive in terms of human effort, making automatic taxonomy construction highly desirable. In this work, we introduce Chain-of-Layer which is an in-context learning framework designed to induct taxonomies from a given set of entities. Chain-of-Layer breaks down the task into selecting relevant candidate entities in each layer and gradually building the taxonomy from top to bottom. To minimize errors, we introduce the Ensemble-based Ranking Filter to reduce the hallucinated content generated at each iteration. Through extensive experiments, we demonstrate that Chain-of-Layer achieves state-of-the-art performance on four real-world benchmarks.
    
[^49]: 使基于流匹配的零样本文本到语音系统自由地产生笑声

    Making Flow-Matching-Based Zero-Shot Text-to-Speech Laugh as You Like

    [https://arxiv.org/abs/2402.07383](https://arxiv.org/abs/2402.07383)

    本文提出了ELaTE，一种基于流匹配的零样本文本到语音系统，可以根据短音频提示以精确控制笑声时机和表情生成任何说话者的自然笑声。

    

    笑声是人类语音中最表达性和自然的一部分，传达着情感、社交暗示和幽默。然而，大多数文本到语音(TTS)系统缺乏产生逼真且合适的笑声的能力，限制了其应用和用户体验。虽然之前有工作生成了自然的笑声，但在控制生成的笑声的时机和多样性方面仍存在不足。在这项工作中，我们提出了ELaTE，一种可以基于短音频提示以精确控制笑声时机和表情的零样本TTS系统，可以产生任何说话者的自然笑声。具体而言，ELaTE通过音频提示来模仿声音特征，通过文本提示来指示所生成语音的内容，通过输入来控制笑声表情，可以是笑声的起始和结束时间，或包含要模仿的笑声的另外音频提示。我们的模型基于找到的技术基础进行了开发。

    Laughter is one of the most expressive and natural aspects of human speech, conveying emotions, social cues, and humor. However, most text-to-speech (TTS) systems lack the ability to produce realistic and appropriate laughter sounds, limiting their applications and user experience. While there have been prior works to generate natural laughter, they fell short in terms of controlling the timing and variety of the laughter to be generated. In this work, we propose ELaTE, a zero-shot TTS that can generate natural laughing speech of any speaker based on a short audio prompt with precise control of laughter timing and expression. Specifically, ELaTE works on the audio prompt to mimic the voice characteristic, the text prompt to indicate the contents of the generated speech, and the input to control the laughter expression, which can be either the start and end times of laughter, or the additional audio prompt that contains laughter to be mimicked. We develop our model based on the foundati
    
[^50]: 通过上下文学习评估分组代表建模的泛化能力

    Assessing Generalization for Subpopulation Representative Modeling via In-Context Learning

    [https://arxiv.org/abs/2402.07368](https://arxiv.org/abs/2402.07368)

    本研究通过使用2016年和2020年的选举数据，评估了基于大型语言模型的分组代表模型在泛化能力上的表现。研究发现，尽管使用实证数据进行条件设定可以提高整体性能，但上下文学习的益处在不同人口子群组之间差异很大，这对实施分组代表模型的从业人员和决策者构成了挑战。

    

    本研究通过对2016年和2020年美国全国选举研究的数据进行上下文学习，评估基于大型语言模型（LLM）的分组代表模型（SRMs）从实证数据中的泛化能力。我们探讨了在不同响应变量和人口子群组之间的泛化能力。尽管使用实证数据进行条件设定可以提高整体性能，但上下文学习的益处在不同人口子群组之间差异很大，有时对某个人口子群组的性能产生了负面影响，但对其他人口子群组的性能产生了积极影响。上下文学习对SRM的不公平益处为实施SRM的从业人员和依赖于其的决策者带来了挑战。我们的工作突出了对来自不同人口子群组的细粒度基准的需求，这些基准不仅测试忠实度，还测试泛化能力。

    This study evaluates the ability of Large Language Model (LLM)-based Subpopulation Representative Models (SRMs) to generalize from empirical data, utilizing in-context learning with data from the 2016 and 2020 American National Election Studies. We explore generalization across response variables and demographic subgroups. While conditioning with empirical data improves performance on the whole, the benefit of in-context learning varies considerably across demographics, sometimes hurting performance for one demographic while helping performance for others. The inequitable benefits of in-context learning for SRM present a challenge for practitioners implementing SRMs, and for decision-makers who might come to rely on them. Our work highlights a need for fine-grained benchmarks captured from diverse subpopulations that test not only fidelity but generalization.
    
[^51]: ODIN: 脱耦奖励缓解RLHF中的黑客攻击

    ODIN: Disentangled Reward Mitigates Hacking in RLHF

    [https://arxiv.org/abs/2402.07319](https://arxiv.org/abs/2402.07319)

    本研究解决了强化学习中的奖励黑客问题，针对回复长度这一挑战，通过建立可靠的评估协议和改进奖励模型的方法，提出了减轻长度偏差的超参数和技巧，并进行了大规模研究。

    

    在这项工作中，我们研究了在LLMs上从人类反馈的强化学习中出现的响应长度上的奖励黑客问题。LLMs的格式良好但不太有用的回复往往会欺骗LLMs甚至人类评估者以获得高分。同样的问题也存在于RL中的某些奖励模型中。为了解决训练和评估中的挑战，我们建立了一个更可靠的评估协议，用于比较不同训练配置之间的LLM评估分数和通过改变训练超参数得到的响应长度之间的权衡。基于这个评估，我们进行了大规模研究，结果揭示了RL中用于减轻长度偏差的超参数和技巧的有效性。我们进一步提出通过在共享特征表示上联合训练两个线性头来改进奖励模型，以预测奖励，一个训练来与长度相关，另一个训练来与内容相关。

    In this work, we study the issue of reward hacking on the response length, a challenge emerging in Reinforcement Learning from Human Feedback (RLHF) on LLMs. A well-formatted, verbose but less helpful response from the LLMs can often deceive LLMs or even human evaluators to achieve high scores. The same issue also holds for some reward models in RL. To address the challenges in both training and evaluation, we establish a more reliable evaluation protocol for comparing different training configurations, which inspects the trade-off between LLM evaluation score and response length obtained by varying training hyperparameters. Based on this evaluation, we conduct large-scale studies, where the results shed insights into the efficacy of hyperparameters and tricks used in RL on mitigating length bias. We further propose to improve the reward model by jointly training two linear heads on shared feature representations to predict the rewards, one trained to correlate with length, and the oth
    
[^52]: HyperBERT:将混合超图感知层与语言模型用于文本属性超图上的节点分类

    HyperBERT: Mixing Hypergraph-Aware Layers with Language Models for Node Classification on Text-Attributed Hypergraphs

    [https://arxiv.org/abs/2402.07309](https://arxiv.org/abs/2402.07309)

    本文提出了HyperBERT模型，通过在预训练的BERT模型中引入超图感知层，克服了现有方法在节点分类任务上难以捕捉超图结构信息和文本属性的局限性，提高了模型的效果和泛化能力。

    

    超图通过复杂的拓扑结构标记，表达多个实体之间的高阶相互作用，其中超边扮演重要角色。最近，基于超图的深度学习方法在学习文本属性超图上的节点分类问题中引起了越来越多的研究关注。然而，现有方法往往难以同时捕捉超图结构信息的全部内容和节点属性中的丰富语言属性，这在很大程度上影响了它们的效果和泛化能力。为了克服这些挑战，我们探索了如何通过为节点分类任务进一步增强预训练的BERT模型，引入专门的超图感知层。这些层将高阶结构归纳偏差引入语言模型中，从而提高模型利用超图结构中的高阶上下文信息和文本中的语义信息的能力。

    Hypergraphs are marked by complex topology, expressing higher-order interactions among multiple entities with hyperedges. Lately, hypergraph-based deep learning methods to learn informative data representations for the problem of node classification on text-attributed hypergraphs have garnered increasing research attention. However, existing methods struggle to simultaneously capture the full extent of hypergraph structural information and the rich linguistic attributes inherent in the nodes attributes, which largely hampers their effectiveness and generalizability. To overcome these challenges, we explore ways to further augment a pretrained BERT model with specialized hypergraph-aware layers for the task of node classification. Such layers introduce higher-order structural inductive bias into the language model, thus improving the model's capacity to harness both higher-order context information from the hypergraph structure and semantic information present in text. In this paper, we
    
[^53]: 基于知识图谱的电力变压器故障预测

    Power Transformer Fault Prediction Based on Knowledge Graphs

    [https://arxiv.org/abs/2402.07283](https://arxiv.org/abs/2402.07283)

    本文提出了一种基于知识图谱和梯度提升决策树的方法，用于学习有限的电力变压器故障数据。实验证明该方法在故障预测准确度上优于传统的人工神经网络和逻辑回归方法。

    

    本文针对电力变压器仅有有限的故障数据这一挑战，提出了一种解决方案。传统的运维工具对潜在故障的预测能力有限。由于故障数据的稀缺性，使得机器学习技术很难有效应用。为了解决这个问题，我们提出了一种新颖的方法，将知识图谱（Knowledge Graph，KG）技术与梯度提升决策树（Gradient Boosting Decision Trees，GBDT）相结合。该方法旨在从少量的高维数据中高效学习，整合了影响变压器故障的各种因素和历史运行数据。我们的方法能够在故障特征数据有限的情况下，实现对电力变压器的准确安全状态评估和故障分析。实验结果表明，相比于其他学习方法，如人工神经网络（Artificial Neural Networks，ANN）和逻辑回归（Logistic Regression，LR），我们的方法在预测准确度方面表现出更好的性能。

    In this paper, we address the challenge of learning with limited fault data for power transformers. Traditional operation and maintenance tools lack effective predictive capabilities for potential faults. The scarcity of extensive fault data makes it difficult to apply machine learning techniques effectively. To solve this problem, we propose a novel approach that leverages the knowledge graph (KG) technology in combination with gradient boosting decision trees (GBDT). This method is designed to efficiently learn from a small set of high-dimensional data, integrating various factors influencing transformer faults and historical operational data. Our approach enables accurate safe state assessments and fault analyses of power transformers despite the limited fault characteristic data. Experimental results demonstrate that this method outperforms other learning approaches in prediction accuracy, such as artificial neural networks (ANN) and logistic regression (LR). Furthermore, it offers
    
[^54]: 大型语言模型如何在诚实与帮助之间进行权衡？

    How do Large Language Models Navigate Conflicts between Honesty and Helpfulness?

    [https://arxiv.org/abs/2402.07282](https://arxiv.org/abs/2402.07282)

    本文研究了如何在大型语言模型中权衡诚实和帮助性，在实验中发现强化学习改善了诚实和帮助性，而链式思维提示则偏向于帮助性。研究结果还展示了GPT-4 Turbo对对话框架和听众决策背景的敏感性。这些发现揭示了大型语言模型内化的对话价值观，并暗示零-shot提示可以在一定程度上引导这些抽象价值观。

    

    在日常交流中，人们经常为了最大限度地帮助听众而近似真相，例如约略时间或省略细节。大型语言模型（LLMs）如何处理这种微妙的权衡？为了回答这个问题，我们使用心理模型和旨在描述人类行为的实验来分析LLMs。我们测试了一系列LLMs，并探讨了优化人类偏好或推理时思考对这些权衡的影响。我们发现，从人类反馈中的强化学习改善了诚实和帮助性，而链式思维提示使LLMs偏向于帮助性而不是诚实。最后，GPT-4 Turbo展示了类似人类的回应模式，包括对对话框架和听众决策背景的敏感性。我们的研究结果揭示了LLMs内化的对话价值观，并暗示即使这些抽象价值观也可以在零-shot提示下在一定程度上被引导。

    In day-to-day communication, people often approximate the truth - for example, rounding the time or omitting details - in order to be maximally helpful to the listener. How do large language models (LLMs) handle such nuanced trade-offs? To address this question, we use psychological models and experiments designed to characterize human behavior to analyze LLMs. We test a range of LLMs and explore how optimization for human preferences or inference-time reasoning affects these trade-offs. We find that reinforcement learning from human feedback improves both honesty and helpfulness, while chain-of-thought prompting skews LLMs towards helpfulness over honesty. Finally, GPT-4 Turbo demonstrates human-like response patterns including sensitivity to the conversational framing and listener's decision context. Our findings reveal the conversational values internalized by LLMs and suggest that even these abstract values can, to a degree, be steered by zero-shot prompting.
    
[^55]: 以故事总结片段辅助阅读：故事阅读中的回顾片段识别

    Previously on the Stories: Recap Snippet Identification for Story Reading

    [https://arxiv.org/abs/2402.07271](https://arxiv.org/abs/2402.07271)

    本研究提出了一项新的任务，即回顾片段识别，在故事阅读中通过回顾之前的重要元素来辅助理解正在进行的情节。该任务对PLMs、LLMs和提出的方法具有挑战性，并需要深入理解片段之间的情节相关性。

    

    类似于电视剧中的"前情回顾"，回顾片段可以通过回忆读者在之前的文本中的重要元素来帮助书籍阅读，以更好地理解正在进行的情节。尽管其有用性，但这种应用在自然语言处理领域尚未得到很好的研究。我们提出了第一个该有用任务的基准-回顾片段识别，并使用手工评估数据集进行评估。我们的实验证实，该任务对于PLMs、LLMs和提出的方法来说是具有挑战性的，因为该任务需要深入理解片段之间情节的相关性。

    Similar to the "previously-on" scenes in TV shows, recaps can help book reading by recalling the readers' memory about the important elements in previous texts to better understand the ongoing plot. Despite its usefulness, this application has not been well studied in the NLP community. We propose the first benchmark on this useful task called Recap Snippet Identification with a hand-crafted evaluation dataset. Our experiments show that the proposed task is challenging to PLMs, LLMs, and proposed methods as the task requires a deep understanding of the plot correlation between snippets.
    
[^56]: 通过利用分类数据集和其语义层次，开展视觉语言模型的开放式VQA评估

    Open-ended VQA benchmarking of Vision-Language models by exploiting Classification datasets and their semantic hierarchy

    [https://arxiv.org/abs/2402.07270](https://arxiv.org/abs/2402.07270)

    该研究通过提出创新的评估方法和基于分类数据集的新型VQA基准，推动了对文本生成的视觉语言模型能力的理解。同时，他们还提出了使用语义层次和自动生成的后续问题来改进对细粒度分类任务上粗糙答案的评估。通过比较不同度量标准，他们在进行人工评估研究的基础上选择了最终的度量标准。

    

    评估文本生成的视觉语言模型是一项具有挑战性但至关重要的工作。通过解决现有视觉问答（VQA）基准的局限性并提出创新的评估方法，我们的研究旨在推动我们对这些模型能力的理解。我们提出了一种基于知名视觉分类数据集的新型VQA基准，可以对文本生成的视觉语言模型进行细粒度评估，并与判别性视觉语言模型进行比较。为了改善对细粒度分类任务上粗糙答案的评估，我们建议使用标签空间的语义层次来提出关于基准类别的自动生成的后续问题。最后，我们比较了传统的自然语言处理和基于LLM的度量标准来评估给定基准答案的模型预测问题。我们进行了人工评估研究，基于此决定最终度量标准的选择。

    The evaluation of text-generative vision-language models is a challenging yet crucial endeavor. By addressing the limitations of existing Visual Question Answering (VQA) benchmarks and proposing innovative evaluation methodologies, our research seeks to advance our understanding of these models' capabilities. We propose a novel VQA benchmark based on well-known visual classification datasets which allows a granular evaluation of text-generative vision-language models and their comparison with discriminative vision-language models. To improve the assessment of coarse answers on fine-grained classification tasks, we suggest using the semantic hierarchy of the label space to ask automatically generated follow-up questions about the ground-truth category. Finally, we compare traditional NLP and LLM-based metrics for the problem of evaluating model predictions given ground-truth answers. We perform a human evaluation study upon which we base our decision on the final metric. We apply our be
    
[^57]: 低资源情况下对印度语言进行对抗言论生成：以孟加拉语和印地语为例

    Low-Resource Counterspeech Generation for Indic Languages: The Case of Bengali and Hindi

    [https://arxiv.org/abs/2402.07262](https://arxiv.org/abs/2402.07262)

    该论文针对低资源语言如孟加拉语和印地语，创建了一个包含5062个虐待言论/对抗言论对的基准数据集，并实现了几种基线模型，以生成适当的对抗言论。观察发现，单语设置的性能最佳，并且通过使用合成转移，语言模型可以在一定程度上生成对抗言论，尤其是当语言属于同一语言家族时，可迁移性更好。

    

    随着网络虐待的兴起，自然语言处理（NLP）社区已经开始研究使用神经架构生成对抗言论，以“反制”这种滥用言论的恶劣语气，以此来减轻/改进其对社交网络的影响。然而，到目前为止，大部分工作都着重于英语。为了填补孟加拉语和印地语等低资源语言的差距，我们创建了一个由5062个虐待言论/对抗言论对组成的基准数据集，其中2460对语料是孟加拉语，2602对语料是印地语。我们使用不同配置的几个基线模型来考虑各种跨语言转移机制，以生成适当的对抗言论，从而建立一个有效的基准。我们观察到，单语设置的性能最佳。此外，使用合成转移，语言模型可以在一定程度上生成对抗言论；具体而言，我们注意到当语言属于同一语言家族时，可迁移性更好。

    With the rise of online abuse, the NLP community has begun investigating the use of neural architectures to generate counterspeech that can "counter" the vicious tone of such abusive speech and dilute/ameliorate their rippling effect over the social network. However, most of the efforts so far have been primarily focused on English. To bridge the gap for low-resource languages such as Bengali and Hindi, we create a benchmark dataset of 5,062 abusive speech/counterspeech pairs, of which 2,460 pairs are in Bengali and 2,602 pairs are in Hindi. We implement several baseline models considering various interlingual transfer mechanisms with different configurations to generate suitable counterspeech to set up an effective benchmark. We observe that the monolingual setup yields the best performance. Further, using synthetic transfer, language models can generate counterspeech to some extent; specifically, we notice that transferability is better when languages belong to the same language fami
    
[^58]: 美国手语视频到文字的翻译

    American Sign Language Video to Text Translation

    [https://arxiv.org/abs/2402.07255](https://arxiv.org/abs/2402.07255)

    这项研究关注美国手语视频到文字的翻译技术，通过复制和改进以往研究，建立了评估模型的方法，发现模型性能受优化器、激活函数和标签平滑的影响，进一步研究旨在改善视觉特征捕捉、增强解码器利用率，并整合预训练解码器，以实现更好的翻译结果。

    

    手语到文本的转换是一项关键技术，可以消除听力困难人群之间的沟通障碍。我们复制并试图改进最近一项已发表的研究。我们使用BLEU和rBLEU指标评估模型，以确保翻译质量。在我们的消融研究中，我们发现模型的性能受到优化器、激活函数和标签平滑的显著影响。进一步的研究旨在改进视觉特征捕捉、增强解码器利用率，并整合预训练解码器以获得更好的翻译结果。我们的源代码可用于复制我们的结果并鼓励未来的研究。

    Sign language to text is a crucial technology that can break down communication barriers for individuals with hearing difficulties. We replicate and try to improve on a recently published study. We evaluate models using BLEU and rBLEU metrics to ensure translation quality. During our ablation study, we found that the model's performance is significantly influenced by optimizers, activation functions, and label smoothing. Further research aims to refine visual feature capturing, enhance decoder utilization, and integrate pre-trained decoders for better translation outcomes. Our source code is available to facilitate replication of our results and encourage future research.
    
[^59]: TransGPT：用于交通的多模式生成预训练Transformer

    TransGPT: Multi-modal Generative Pre-trained Transformer for Transportation

    [https://arxiv.org/abs/2402.07233](https://arxiv.org/abs/2402.07233)

    TransGPT是一种面向交通领域的新型多模式生成预训练Transformer，使用单模式和多模式数据进行微调，在交通领域的各种任务中优于基准模型。

    

    自然语言处理（NLP）是智能交通系统（ITS）的关键组成部分，但在交通领域面临着许多挑战，如专业领域知识与数据，多模式输入和输出。本文提出了TransGPT，一种面向交通领域的新型（多模式）大语言模型，由两个独立的变体组成：TransGPT-SM用于单模式数据和TransGPT-MM用于多模式数据。TransGPT-SM在包含来自交通领域各种来源的文本数据的单模式交通数据集（STD）上进行微调。TransGPT-MM在我们手动收集的交通领域的三个领域（驾驶测试、交通标志和地标）的多模式交通数据集（MTD）上细调。我们对TransGPT在交通领域的几个基准数据集上进行评估，并展示了它在大多数任务上优于基准模型。我们还展示了该模型的潜在应用。

    Natural language processing (NLP) is a key component of intelligent transportation systems (ITS), but it faces many challenges in the transportation domain, such as domain-specific knowledge and data, and multi-modal inputs and outputs. This paper presents TransGPT, a novel (multi-modal) large language model for the transportation domain, which consists of two independent variants: TransGPT-SM for single-modal data and TransGPT-MM for multi-modal data. TransGPT-SM is finetuned on a single-modal Transportation dataset (STD) that contains textual data from various sources in the transportation domain. TransGPT-MM is finetuned on a multi-modal Transportation dataset (MTD) that we manually collected from three areas of the transportation domain: driving tests, traffic signs, and landmarks. We evaluate TransGPT on several benchmark datasets for different tasks in the transportation domain, and show that it outperforms baseline models on most tasks. We also showcase the potential application
    
[^60]: 透过分割投票的视角: 探索在法律案件结果分类中的意见分歧、困难和校准

    Through the Lens of Split Vote: Exploring Disagreement, Difficulty and Calibration in Legal Case Outcome Classification

    [https://arxiv.org/abs/2402.07214](https://arxiv.org/abs/2402.07214)

    通过研究分割投票，探索律师在处理法律案件结果分类时面临的意见分歧和困难，并在欧洲人权法院收集了法官的投票数据集进行研究。这项研究还评估了模型和人类之间感知困难的一致性以及模型的置信度和人类校准。

    

    在法律决策中，当法官无法达成一致决定时，就会出现分割投票(SV)，给必须处理各种法律论点和意见的律师带来了困难。在高风险领域，理解人类和AI系统之间感知困难的一致性对于建立信任至关重要。然而，现有的自然语言处理校准方法主要关注分类器对预测性能的认知，通常是与人类的多数类进行比较，而忽视了人类标签变化的固有差异（HLV）。本文将分割投票视为自然可观察的人类意见分歧和价值多元主义，并从欧洲人权法院（ECHR）收集法官的投票分布，提出了带有SV信息的案件结果分类（COC）数据集SV-ECHR。我们建立了包含SV特定子类别的不同意见的分类法。我们进一步评估模型和人类之间感知困难的一致性，以及COC模型的置信度和人类校准。我们观察到了限制性的...

    In legal decisions, split votes (SV) occur when judges cannot reach a unanimous decision, posing a difficulty for lawyers who must navigate diverse legal arguments and opinions. In high-stakes domains, understanding the alignment of perceived difficulty between humans and AI systems is crucial to build trust. However, existing NLP calibration methods focus on a classifier's awareness of predictive performance, measured against the human majority class, overlooking inherent human label variation (HLV). This paper explores split votes as naturally observable human disagreement and value pluralism. We collect judges' vote distributions from the European Court of Human Rights (ECHR), and present SV-ECHR, a case outcome classification (COC) dataset with SV information. We build a taxonomy of disagreement with SV-specific subcategories. We further assess the alignment of perceived difficulty between models and humans, as well as confidence- and human-calibration of COC models. We observe lim
    
[^61]: 结合空间优化和大型语言模型的开放领域城市行程规划

    Synergizing Spatial Optimization with Large Language Models for Open-Domain Urban Itinerary Planning

    [https://arxiv.org/abs/2402.07204](https://arxiv.org/abs/2402.07204)

    本文提出了Open-domain Urban Itinerary Planning (OUIP)任务，用于根据用户以自然语言描述的请求直接生成行程，通过结合空间优化和大型语言模型(LLM)，提供个性化的城市行程定制服务。

    

    本文首次提出了Open-domain Urban Itinerary Planning (OUIP)任务，用于根据用户以自然语言描述的请求直接生成行程。OUIP与传统行程规划不同，传统规划限制了用户表达更详细的需求，阻碍了真正的个性化。最近，大型语言模型(LLM)在处理多样化任务方面表现出潜力。然而，由于非实时信息、不完整的知识和不足的空间意识，它们无法独立地提供满意的用户体验。鉴于此，我们提出了一个名为ItiNera的OUIP系统，将空间优化与大型语言模型(LLM)相结合，根据用户需求提供个性化的城市行程定制服务。具体来说，我们开发了一个基于LLM的流水线，用于提取和更新兴趣点特征，以创建用户自己的个性化兴趣点数据库。对于每个用户请求，我们利用LLM进行协同实现优化。

    In this paper, we for the first time propose the task of Open-domain Urban Itinerary Planning (OUIP) for citywalk, which directly generates itineraries based on users' requests described in natural language. OUIP is different from conventional itinerary planning, which limits users from expressing more detailed needs and hinders true personalization. Recently, large language models (LLMs) have shown potential in handling diverse tasks. However, due to non-real-time information, incomplete knowledge, and insufficient spatial awareness, they are unable to independently deliver a satisfactory user experience in OUIP. Given this, we present ItiNera, an OUIP system that synergizes spatial optimization with Large Language Models (LLMs) to provide services that customize urban itineraries based on users' needs. Specifically, we develop an LLM-based pipeline for extracting and updating POI features to create a user-owned personalized POI database. For each user request, we leverage LLM in coop
    
[^62]: 在基于检索增强生成的大型语言模型中进行提示扰动

    Prompt Perturbation in Retrieval-Augmented Generation based Large Language Models

    [https://arxiv.org/abs/2402.07179](https://arxiv.org/abs/2402.07179)

    本文研究了基于检索增强生成的大型语言模型（LLM）中提示扰动的影响，并引入了一种新的优化技术GGPP。通过GGPP，我们可以将LLMs的输出引导到特定的错误答案，并应对提示中的无关上下文。

    

    大型语言模型（LLM）的鲁棒性在其在各个领域的使用迅速增长中变得越来越重要。检索增强生成（RAG）被视为提高从LLM生成文本的可信度的方法。然而，目前对RAG-based LLMs的输出如何受到稍有不同的输入影响的研究还不够充分。在本文中，我们发现即使在提示中插入一个很短的前缀也会导致生成的输出与事实正确答案相去甚远。我们系统地评估了这类前缀对RAG的影响，并引入了一种称为Gradient Guided Prompt Perturbation（GGPP）的新型优化技术。GGPP在将RAG-based LLMs的输出引导到特定错误答案方面取得了很高的成功率。它还可以应对提示中请求忽略无关上下文的指令。我们还利用LLMs在带有和不带有GGPP扰动的提示之间的神经元激活差异来提供一种改进方法。

    The robustness of large language models (LLMs) becomes increasingly important as their use rapidly grows in a wide range of domains. Retrieval-Augmented Generation (RAG) is considered as a means to improve the trustworthiness of text generation from LLMs. However, how the outputs from RAG-based LLMs are affected by slightly different inputs is not well studied. In this work, we find that the insertion of even a short prefix to the prompt leads to the generation of outputs far away from factually correct answers. We systematically evaluate the effect of such prefixes on RAG by introducing a novel optimization technique called Gradient Guided Prompt Perturbation (GGPP). GGPP achieves a high success rate in steering outputs of RAG-based LLMs to targeted wrong answers. It can also cope with instructions in the prompts requesting to ignore irrelevant context. We also exploit LLMs' neuron activation difference between prompts with and without GGPP perturbations to give a method that improves
    
[^63]: 自然语言强化学习

    Natural Language Reinforcement Learning

    [https://arxiv.org/abs/2402.07157](https://arxiv.org/abs/2402.07157)

    本研究将自然语言表示和强化学习原则相结合，提出了自然语言强化学习（NLRL）框架，解决了强化学习在样本效率低、解释性不足和缺乏监督信号等方面的限制问题，通过实验验证了其有效性和可解释性。

    

    强化学习（RL）在学习决策任务的策略方面展现出了令人瞩目的能力。然而，RL常常面临样本效率低、解释性不足和缺乏稀疏监督信号等问题的限制。为了解决这些问题，我们从人类学习过程中汲取灵感，引入了自然语言强化学习（NLRL），创新性地将RL原则与自然语言表示结合起来。具体而言，NLRL在自然语言空间中重新定义了任务目标、策略、价值函数、Bellman方程和策略迭代等RL概念。我们还展示了如何利用最新的大型语言模型（LLM）如GPT-4来实现NLRL。对表格MDPs的初步实验表明了NLRL框架的有效性、高效性和可解释性。

    Reinforcement Learning (RL) has shown remarkable abilities in learning policies for decision-making tasks. However, RL is often hindered by issues such as low sample efficiency, lack of interpretability, and sparse supervision signals. To tackle these limitations, we take inspiration from the human learning process and introduce Natural Language Reinforcement Learning (NLRL), which innovatively combines RL principles with natural language representation. Specifically, NLRL redefines RL concepts like task objectives, policy, value function, Bellman equation, and policy iteration in natural language space. We present how NLRL can be practically implemented with the latest advancements in large language models (LLMs) like GPT-4. Initial experiments over tabular MDPs demonstrate the effectiveness, efficiency, and also interpretability of the NLRL framework.
    
[^64]: X-LoRA: 一种灵活的大型语言模型框架，利用低秩适配器专家的混合策略在蛋白质力学和设计中的应用

    X-LoRA: Mixture of Low-Rank Adapter Experts, a Flexible Framework for Large Language Models with Applications in Protein Mechanics and Design

    [https://arxiv.org/abs/2402.07148](https://arxiv.org/abs/2402.07148)

    X-LoRA是一种灵活的大型语言模型框架，利用低秩适配器专家的混合策略，可以创建精细调整的模型并在蛋白质力学和设计领域应用。该模型利用深层逐层适应的组合来解决特定任务，并受到生物学原理的启发。无需修改底层结构即可应用于任何现有的语言模型。

    

    我们报道了一种使用深层逐层基于低秩适应（LoRA）的新颖预训练适配器的混合专家策略，用于创建精细调整的大型语言模型。我们提出了一种利用隐藏状态动态混合经过适应的层的门控策略，允许得到的X-LoRA模型利用不同的能力并创建以前未使用的深层逐层适应的组合来解决特定任务。该设计受到了生物普遍性和多样性的生物学原理的启发，其中神经网络建模块在不同的分层表示中被重复使用。因此，X-LoRA模型可以轻松用于任何现有的大型语言模型（LLM），无需修改底层结构。我们还开发了一个定制的X-LoRA模型，提供了包括前向/逆向分析任务和增强推理能力在内的科学能力，重点是生物材料分析。

    We report a mixture of expert strategy to create fine-tuned large language models using a deep layer-wise token-level approach based on low-rank adaptation (LoRA). Starting with a set of pre-trained LoRA adapters, we propose a gating strategy that uses the hidden states to dynamically mix adapted layers, allowing the resulting X-LoRA model to draw upon different capabilities and create never-before-used deep layer-wise combinations of adaptations are established to solve specific tasks. The design is inspired by the biological principles of universality and diversity, where neural network building blocks are reused in different hierarchical manifestations. Hence, the X-LoRA model can be easily implemented for any existing large language model (LLM) without a need for modifications of the underlying structure. We develop a tailored X-LoRA model that offers scientific capabilities including forward/inverse analysis tasks and enhanced reasoning capability, focused on biomaterial analysis,
    
[^65]: 通过LLM-认知数据增强广义对话密集检索

    Generalizing Conversational Dense Retrieval via LLM-Cognition Data Augmentation

    [https://arxiv.org/abs/2402.07092](https://arxiv.org/abs/2402.07092)

    本文提出了一种通过LLM-认知数据增强的方法来广义对话密集检索。该方法首先生成多级增强对话，捕捉多样的对话环境。其次，通过认知感知过程减少错误生成情况，并通过难度自适应样本筛选器选择具有挑战性的样本。

    

    对话式搜索利用多轮自然语言环境来检索相关段落。现有的对话密集检索模型大多将对话视为一系列固定的问题和回答，忽视了严重的数据稀疏性问题 - 也就是说，用户可以以不同的方式进行对话，而这些备选对话是未记录的。因此，它们经常难以推广到真实场景中的多样对话。在这项工作中，我们提出了一种通过LLM-认知数据增强广义对话密集检索的框架(ConvAug)。ConvAug首先生成多级增强对话，以捕捉对话环境的多样性。受人类认知方式的启发，我们设计了一种认知感知过程，以减少错误的正例、负例和幻觉的生成。此外，我们还开发了一种难度自适应样本筛选器，用于选择复杂对话的具有挑战性的样本。

    Conversational search utilizes muli-turn natural language contexts to retrieve relevant passages. Existing conversational dense retrieval models mostly view a conversation as a fixed sequence of questions and responses, overlooking the severe data sparsity problem -- that is, users can perform a conversation in various ways, and these alternate conversations are unrecorded. Consequently, they often struggle to generalize to diverse conversations in real-world scenarios. In this work, we propose a framework for generalizing Conversational dense retrieval via LLM-cognition data Augmentation (ConvAug). ConvAug first generates multi-level augmented conversations to capture the diverse nature of conversational contexts. Inspired by human cognition, we devise a cognition-aware process to mitigate the generation of false positives, false negatives, and hallucinations. Moreover, we develop a difficulty-adaptive sample filter that selects challenging samples for complex conversations, thereby g
    
[^66]: 基于语音韵律的多说话人语音合成中从音素和音素持续时间中提取说话人嵌入的方法

    Speech Rhythm-Based Speaker Embeddings Extraction from Phonemes and Phoneme Duration for Multi-Speaker Speech Synthesis

    [https://arxiv.org/abs/2402.07085](https://arxiv.org/abs/2402.07085)

    本文提出了一种基于语音韵律的方法，通过从音素和音素持续时间中提取说话人嵌入，模拟目标说话人的个体发音特征。实验证明，该方法可以实现有效的多说话人语音合成。

    

    本文提出了一种基于语音韵律的方法，用于从目标说话人的少量句子中建模音素持续时间，从而提取说话人嵌入。语音韵律是与说话人特征相关的重要因素之一，与基频等声学特征一起用于在语音合成中重现单个句子。所提出方法的一个新特点是基于韵律的嵌入，从已知与说话韵律相关的音素及其持续时间中提取，类似于传统的基于频谱特征的说话人识别模型。我们进行了三个实验，包括生成说话人嵌入、使用生成的嵌入进行语音合成以及嵌入空间分析，以评估该方法的性能。结果表明，即使只使用音素及其持续时间信息，所提出的方法也展现了较为适中的说话人识别性能（15.2% EER）。客观和主观评估结果表明，所提出的方法能够实现有效的多说话人语音合成。

    This paper proposes a speech rhythm-based method for speaker embeddings to model phoneme duration using a few utterances by the target speaker. Speech rhythm is one of the essential factors among speaker characteristics, along with acoustic features such as F0, for reproducing individual utterances in speech synthesis. A novel feature of the proposed method is the rhythm-based embeddings extracted from phonemes and their durations, which are known to be related to speaking rhythm. They are extracted with a speaker identification model similar to the conventional spectral feature-based one. We conducted three experiments, speaker embeddings generation, speech synthesis with generated embeddings, and embedding space analysis, to evaluate the performance. The proposed method demonstrated a moderate speaker identification performance (15.2% EER), even with only phonemes and their duration information. The objective and subjective evaluation results demonstrated that the proposed method can
    
[^67]: 在计算机科学教育中使用大型语言模型进行学生代码引导的测试用例生成

    Using Large Language Models for Student-Code Guided Test Case Generation in Computer Science Education

    [https://arxiv.org/abs/2402.07081](https://arxiv.org/abs/2402.07081)

    本研究旨在提出一种完全自动化的测试用例生成方法，使用大型语言模型，并证明它们是衡量学生知识水平的良好指标，从而解决手动构建测试用例的劳动密集性和专业知识需求的问题。

    

    在计算机科学教育中，测试用例是编程作业的重要组成部分，因为它们可以用作评估项目，测试学生的编程知识，并为学生编写的代码提供个性化反馈。我们的工作目标是提出一种完全自动化的测试用例生成方法，可以准确地衡量学生的知识水平，这一点非常重要。首先，手动构建测试用例需要专业知识，并且是一项劳动密集型工作。其次，为学生开发测试用例，尤其是对于初学者来说，与针对专业级软件开发人员的测试用例有着显著的区别。因此，我们需要一种自动化的测试用例生成过程来评估学生的知识水平并提供反馈。在这项工作中，我们提出了一种基于大型语言模型的自动化测试用例生成方法，并证明它们是衡量学生知识水平的良好指标，使用一个公开可用的数据集。

    In computer science education, test cases are an integral part of programming assignments since they can be used as assessment items to test students' programming knowledge and provide personalized feedback on student-written code. The goal of our work is to propose a fully automated approach for test case generation that can accurately measure student knowledge, which is important for two reasons. First, manually constructing test cases requires expert knowledge and is a labor-intensive process. Second, developing test cases for students, especially those who are novice programmers, is significantly different from those oriented toward professional-level software developers. Therefore, we need an automated process for test case generation to assess student knowledge and provide feedback. In this work, we propose a large language model-based approach to automatically generate test cases and show that they are good measures of student knowledge, using a publicly available dataset that c
    
[^68]: 使用大型语言模型自动化和加速奖励机器强化学习

    Using Large Language Models to Automate and Expedite Reinforcement Learning with Reward Machine

    [https://arxiv.org/abs/2402.07069](https://arxiv.org/abs/2402.07069)

    这篇论文介绍了一种使用大型语言模型自动生成自动机来编码高级知识，加速强化学习过程的算法，并证明了其在多个任务上的有效性和优越性。

    

    我们提出了LARL-RM（通过大型语言模型生成的用于奖励机器强化学习的自动机）算法，以将高级知识编码到强化学习中，使用自动机加速强化学习过程。我们的方法使用大型语言模型（LLM）通过提示工程获得高级领域特定知识，而不是直接将高级知识提供给强化学习算法，这需要专家来编码自动机。我们使用思维链和少样本方法进行提示工程，并证明了我们的方法在这些方法下有效。此外，LARL-RM允许完全闭环的强化学习，无需专家来指导和监督学习，因为LARL-RM可以直接使用LLM生成所需的高级知识以完成任务。我们还证明了算法收敛到最优策略的理论保证。我们证明了LARL-RM的实验结果展示了其对常见的强化学习问题具有非常好的性能，并且在一些任务上超越了目前最先进的方法。

    We present LARL-RM (Large language model-generated Automaton for Reinforcement Learning with Reward Machine) algorithm in order to encode high-level knowledge into reinforcement learning using automaton to expedite the reinforcement learning. Our method uses Large Language Models (LLM) to obtain high-level domain-specific knowledge using prompt engineering instead of providing the reinforcement learning algorithm directly with the high-level knowledge which requires an expert to encode the automaton. We use chain-of-thought and few-shot methods for prompt engineering and demonstrate that our method works using these approaches. Additionally, LARL-RM allows for fully closed-loop reinforcement learning without the need for an expert to guide and supervise the learning since LARL-RM can use the LLM directly to generate the required high-level knowledge for the task at hand. We also show the theoretical guarantee of our algorithm to converge to an optimal policy. We demonstrate that LARL-R
    
[^69]: 半监督学习用于双语词典诱导

    Semi-Supervised Learning for Bilingual Lexicon Induction

    [https://arxiv.org/abs/2402.07028](https://arxiv.org/abs/2402.07028)

    本论文提出了一个半监督学习方法，将两种语言对应的连续词表示集对齐到一个共同的空间，推断双语词典。该方法利用无监督学习的基础，在学习新语言时，整合已有语言集的知识，通过排序方法实现词典诱导。

    

    我们考虑将对应于不同语言的两个连续词表示集对齐到一个共同空间，以推断双语词典的问题。最近的研究表明，通过将在单语数据上训练的词嵌入对齐，可以推断出这样的词典而不使用任何平行数据。这种工作称为无监督双语诱导。通过思考是否可能在逐步学习多种语言的过程中积累经验，我们自问在学习新语言时是否能够在没有平行数据的情况下整合给定语言集的知识。换句话说，虽然保持无监督学习的核心问题在最新步骤中，但我们允许访问其他习语语料库，因此称为半监督。这导致我们提出了一种新的表达形式，将词典诱导视为一个排序问题，我们使用了该机器学习领域的最新工具。

    We consider the problem of aligning two sets of continuous word representations, corresponding to languages, to a common space in order to infer a bilingual lexicon. It was recently shown that it is possible to infer such lexicon, without using any parallel data, by aligning word embeddings trained on monolingual data. Such line of work is called unsupervised bilingual induction. By wondering whether it was possible to gain experience in the progressive learning of several languages, we asked ourselves to what extent we could integrate the knowledge of a given set of languages when learning a new one, without having parallel data for the latter. In other words, while keeping the core problem of unsupervised learning in the latest step, we allowed the access to other corpora of idioms, hence the name semi-supervised. This led us to propose a novel formulation, considering the lexicon induction as a ranking problem for which we used recent tools of this machine learning field. Our experi
    
[^70]: 双子座进入医学院：探索多模态大型语言模型在医学挑战问题和幻觉上的能力

    Gemini Goes to Med School: Exploring the Capabilities of Multimodal Large Language Models on Medical Challenge Problems & Hallucinations

    [https://arxiv.org/abs/2402.07023](https://arxiv.org/abs/2402.07023)

    该论文综合评估了开源LLM和谷歌的多模态LLM Gemini 在医学推理、幻觉检测和医学视觉问答任务上的能力。Gemini在诊断准确性方面落后于最先进模型，且易出现幻觉、过度自信和知识盲点。采用提示策略可以提高性能。

    

    大型语言模型在医疗行业具有潜在价值，但通过严格评估来验证其安全性和效果至关重要。为此，我们全面评估了开源LLM和谷歌的新型多模态LLM Gemini 在医学推理、幻觉检测和医学视觉问答任务上的能力。虽然Gemini表现出一定的能力，但在诊断准确性方面落后于MedPaLM 2和GPT-4等最先进模型。此外，Gemini在医学VQA数据集上的准确率为61.45％，明显低于GPT-4V的88％得分。我们的分析发现，Gemini极易出现幻觉、过度自信和知识盲点，这表明如果不加批判地部署，存在风险。我们还针对不同医学学科和测试类型进行了详细分析，为开发人员和临床医生提供了可操作的反馈。为了减少风险，我们采用了提示策略来提高性能。

    Large language models have the potential to be valuable in the healthcare industry, but it's crucial to verify their safety and effectiveness through rigorous evaluation. For this purpose, we comprehensively evaluated both open-source LLMs and Google's new multimodal LLM called Gemini across Medical reasoning, hallucination detection, and Medical Visual Question Answering tasks. While Gemini showed competence, it lagged behind state-of-the-art models like MedPaLM 2 and GPT-4 in diagnostic accuracy. Additionally, Gemini achieved an accuracy of 61.45\% on the medical VQA dataset, significantly lower than GPT-4V's score of 88\%. Our analysis revealed that Gemini is highly susceptible to hallucinations, overconfidence, and knowledge gaps, which indicate risks if deployed uncritically. We also performed a detailed analysis by medical subject and test type, providing actionable feedback for developers and clinicians. To mitigate risks, we applied prompting strategies that improved performanc
    
[^71]: 语音转换成歌曲的幻象的合理分析

    A Rational Analysis of the Speech-to-Song Illusion

    [https://arxiv.org/abs/2402.06992](https://arxiv.org/abs/2402.06992)

    本论文提出了对语音转换成歌曲的幻象的合理分析，将其视为一种统计推断过程，通过分析语料库，还发现了一种纯文本的小说转歌词的幻象，并提供了强有力的证据来支持这一观点。

    

    语音转换成歌曲的幻象是一种强大的心理现象，即说出的句子在不断重复中越来越像音乐。尽管经过几十年的研究，对这种转化的完整形式解释仍然缺乏，并且对其细微的特征，即某些短语的转化是否发生，也不太清楚。在这里，我们提供了这个现象的一个形式化解释，将其重新定义为一种统计推断，合理的决策者试图判断一系列话语更有可能是在歌曲中还是在讲话中产生的。通过使用这种方法并分析歌曲和讲话的语料库，我们进一步介绍了一种纯文本的小说转歌词的幻象。在这个幻象中，简单地复制书面句子会使它们看起来更像歌词。我们在人类参与者和大型语言模型中提供了这个新幻象的强有力的证据。

    The speech-to-song illusion is a robust psychological phenomenon whereby a spoken sentence sounds increasingly more musical as it is repeated. Despite decades of research, a complete formal account of this transformation is still lacking, and some of its nuanced characteristics, namely, that certain phrases appear to transform while others do not, is not well understood. Here we provide a formal account of this phenomenon, by recasting it as a statistical inference whereby a rational agent attempts to decide whether a sequence of utterances is more likely to have been produced in a song or speech. Using this approach and analyzing song and speech corpora, we further introduce a novel prose-to-lyrics illusion that is purely text-based. In this illusion, simply duplicating written sentences makes them appear more like song lyrics. We provide robust evidence for this new illusion in both human participants and large language models.
    
[^72]: 事件关键摘要

    Event-Keyed Summarization

    [https://arxiv.org/abs/2402.06973](https://arxiv.org/abs/2402.06973)

    事件关键摘要（EKS）是一种新颖的任务，旨在为特定事件生成上下文化的摘要。我们提出了一个基准数据集MUCSUM，并展示了EKS与传统摘要和结构到文本的比较结果。

    

    我们介绍了一种新颖的任务，称为事件关键摘要（EKS），它将传统的摘要和文档级事件提取结合起来，目标是在给定文档和提取的事件结构的情况下生成一个上下文化的特定事件摘要。我们介绍了一个用于这个任务的数据集MUCSUM，包括经典MUC-4数据集中所有事件的摘要，以及一组基线模型，其中包括在摘要文献中预训练的语言模型标准以及更大的前沿模型。我们表明，将EKS简化为传统的摘要或结构到文本的去除都会得到较差的目标事件摘要，并且MUCSUM是这一任务的一个稳健的基准。最后，我们对参考摘要和模型摘要进行了人工评估，并对结果进行了详细分析。

    We introduce event-keyed summarization (EKS), a novel task that marries traditional summarization and document-level event extraction, with the goal of generating a contextualized summary for a specific event, given a document and an extracted event structure. We introduce a dataset for this task, MUCSUM, consisting of summaries of all events in the classic MUC-4 dataset, along with a set of baselines that comprises both pretrained LM standards in the summarization literature, as well as larger frontier models. We show that ablations that reduce EKS to traditional summarization or structure-to-text yield inferior summaries of target events and that MUCSUM is a robust benchmark for this task. Lastly, we conduct a human evaluation of both reference and model summaries, and provide some detailed analysis of the results.
    
[^73]: 一次指导，多轮稳定对话：对话的高效调整框架

    Instruct Once, Chat Consistently in Multiple Rounds: An Efficient Tuning Framework for Dialogue

    [https://arxiv.org/abs/2402.06967](https://arxiv.org/abs/2402.06967)

    本论文提出了一种名为Midi-Tuning的多轮交互对话调整框架，通过分别对代理人和用户建模，并利用轮次级内存缓存机制进行调整，实现了对话代理的一致性和稳定性。

    

    调整预训练语言模型以实现对话生成已成为构建能力强大的对话代理的主流范式。然而，传统的调整方式狭隘地将对话生成视为类似其他语言生成任务的过程，忽视了对话者之间的角色差异和对话应具备的多轮交互过程。这种方式导致了所构建代理人的对话一致性不尽如人意。在本研究中，我们强调对话的交互性和沟通性，并认为分别对代理人和用户的讲话者角色进行建模更为可行，使得代理人能够保持一致的角色。我们提出了一种有效的多轮交互对话调整（Midi-Tuning）框架。该框架使用基于大型语言模型的两个适配器分别对代理人和用户进行建模，它们按轮次交替使用话语，并通过轮次级内存缓存机制进行调整。大量实验证明，我们的方法可以有效提高对话代理的一致性和稳定性。

    Tuning pretrained language models for dialogue generation has been a prevalent paradigm for building capable dialogue agents. Yet, traditional tuning narrowly views dialogue generation as resembling other language generation tasks, ignoring the role disparities between two speakers and the multi-round interactive process that dialogues ought to be. Such a manner leads to unsatisfactory chat consistency of the built agent. In this work, we emphasize the interactive, communicative nature of dialogue and argue that it is more feasible to model the speaker roles of agent and user separately, enabling the agent to adhere to its role consistently. We propose an efficient Multi-round Interactive Dialogue Tuning (Midi-Tuning) framework. It models the agent and user individually with two adapters built upon large language models, where they utilize utterances round by round in alternating order and are tuned via a round-level memory caching mechanism. Extensive experiments demonstrate that, our
    
[^74]: NLP用于从能源领域语料库中进行知识发现和信息提取

    NLP for Knowledge Discovery and Information Extraction from Energetics Corpora

    [https://arxiv.org/abs/2402.06964](https://arxiv.org/abs/2402.06964)

    这项研究展示了NLP在能源材料研究中的实用性，通过将NLP方法应用于能源文本，可以自动发现知识和提取信息。该研究使用三个成熟的NLP模型，并证明它们能够识别能源话题和概念，并生成与专家知识一致的语言模型。此外，研究还提出了一个能源文本的分类流程，其准确率高达59-76\%，最佳模型与专家间一致度相当。

    

    我们展示了NLP在辅助研究能源材料和相关系统方面的实用性。这种NLP方法使得机器能够理解文本数据，从而提供了通过能源文本进行知识发现和信息提取的自动化路径。我们将三个成熟的无监督NLP模型（潜在狄利克雷分配，Word2Vec和Transformer）应用于一个大型的经过策划的能源相关科学文章的数据集。我们证明每个NLP算法都能够识别出能源话题和概念，并生成与专家知识相一致的语言模型。此外，我们提出了一个用于能源文本的文档分类流程。我们的分类流程在使用NLP模型的情况下达到了59-76\%的准确率，最佳性能的Transformer模型与注释者间的一致性度量相媲美。本研究中研究的NLP方法可以识别与能源领域相关的概念，因此具有潜在的应用价值。

    We present a demonstration of the utility of NLP for aiding research into energetic materials and associated systems. The NLP method enables machine understanding of textual data, offering an automated route to knowledge discovery and information extraction from energetics text. We apply three established unsupervised NLP models: Latent Dirichlet Allocation, Word2Vec, and the Transformer to a large curated dataset of energetics-related scientific articles. We demonstrate that each NLP algorithm is capable of identifying energetic topics and concepts, generating a language model which aligns with Subject Matter Expert knowledge. Furthermore, we present a document classification pipeline for energetics text. Our classification pipeline achieves 59-76\% accuracy depending on the NLP model used, with the highest performing Transformer model rivaling inter-annotator agreement metrics. The NLP approaches studied in this work can identify concepts germane to energetics and therefore hold prom
    
[^75]: SpeechCLIP+: 基于CLIP和语音-图像数据的自监督多任务表示学习的语音模型

    SpeechCLIP+: Self-supervised multi-task representation learning for speech via CLIP and speech-image data

    [https://arxiv.org/abs/2402.06959](https://arxiv.org/abs/2402.06959)

    SpeechCLIP+通过应用CIF模块替换CLIP架构中的CLS令牌，并提出了一种混合架构，实现了在语音关键词提取和图像-语音检索任务中的性能提升。

    

    最近提出的视觉化语音模型SpeechCLIP是一个创新的框架，通过CLIP将语音和文本与图像相连接，而无需依赖于文本转录。在此基础上，本文介绍了两个对SpeechCLIP的扩展。首先，我们应用了连续积分-放电（CIF）模块，用于替换级联架构中的固定数量的CLS令牌。其次，我们提出了一种新的混合架构，将SpeechCLIP的级联架构和并行架构合并为一个多任务学习框架。我们在Flickr8k和SpokenCOCO数据集上进行了实验评估。结果表明，在语音关键词提取任务中，基于CIF的级联SpeechCLIP模型表现优于之前使用固定数量CLS令牌的级联SpeechCLIP模型。此外，通过我们的混合架构，级联任务学习提升了图像-语音检索任务中并行分支的性能。

    The recently proposed visually grounded speech model SpeechCLIP is an innovative framework that bridges speech and text through images via CLIP without relying on text transcription. On this basis, this paper introduces two extensions to SpeechCLIP. First, we apply the Continuous Integrate-and-Fire (CIF) module to replace a fixed number of CLS tokens in the cascaded architecture. Second, we propose a new hybrid architecture that merges the cascaded and parallel architectures of SpeechCLIP into a multi-task learning framework. Our experimental evaluation is performed on the Flickr8k and SpokenCOCO datasets. The results show that in the speech keyword extraction task, the CIF-based cascaded SpeechCLIP model outperforms the previous cascaded SpeechCLIP model using a fixed number of CLS tokens. Furthermore, through our hybrid architecture, cascaded task learning boosts the performance of the parallel branch in image-speech retrieval tasks.
    
[^76]: OpenFedLLM：通过联邦学习在分散的私有数据上训练大规模语言模型

    OpenFedLLM: Training Large Language Models on Decentralized Private Data via Federated Learning

    [https://arxiv.org/abs/2402.06954](https://arxiv.org/abs/2402.06954)

    OpenFedLLM是一个简洁、集成、研究友好的框架/代码库，通过联邦学习在分散的私有数据上实现了大规模语言模型的协作和隐私保护训练，解决了公开数据枯竭的问题。

    

    在大规模公开可用的数据上训练的大规模语言模型（LLM）在各个领域取得了巨大的成功。然而，更多的数据可以提高性能，但令人担忧的是，高质量的公开数据将在几年内用尽。在本文中，我们提供了对当代LLM的潜在下一步：通过联邦学习在未充分利用的分布式私有数据上进行协作和保护隐私的LLM训练，多个数据所有者共同训练一个共享模型，而不传输原始数据。为了实现这一目标，我们构建了一个简洁、集成和研究友好的框架/代码库，名为OpenFedLLM。它涵盖了用于增强模型遵循指令能力的联邦指令调优、用于与人类价值观对齐的联邦价值对齐以及7个代表性联邦学习算法。此外，OpenFedLLM支持在多领域进行训练，我们涵盖了8个训练数据集；提供全面的评估，我们涵盖了...

    Trained on massive publicly available data, large language models (LLMs) have demonstrated tremendous success across various fields. While more data contributes to better performance, a disconcerting reality is that high-quality public data will be exhausted in a few years. In this paper, we offer a potential next step for contemporary LLMs: collaborative and privacy-preserving LLM training on the underutilized distributed private data via federated learning (FL), where multiple data owners collaboratively train a shared model without transmitting raw data. To achieve this, we build a concise, integrated, and research-friendly framework/codebase, named OpenFedLLM. It covers federated instruction tuning for enhancing instruction-following capability, federated value alignment for aligning with human values, and 7 representative FL algorithms. Besides, OpenFedLLM supports training on diverse domains, where we cover 8 training datasets; and provides comprehensive evaluations, where we cov
    
[^77]: 我在为NLP任务微调预训练的Transformer时是否应该尝试多个优化器？是否应该调整它们的超参数？

    Should I try multiple optimizers when fine-tuning pre-trained Transformers for NLP tasks? Should I tune their hyperparameters?

    [https://arxiv.org/abs/2402.06948](https://arxiv.org/abs/2402.06948)

    在微调预训练的Transformer进行NLP任务时，调整优化器的超参数并不会对测试性能产生实质性差异，只调整学习率通常就足够。

    

    NLP研究已经探索了不同的神经模型架构和大小、数据集、训练目标和迁移学习技术。然而，在训练过程中选择优化器并没有得到广泛探讨。通常情况下，使用随机梯度下降（SGD）的某个变种，根据不明确的标准选择，并且往往对优化器的超参数进行最小或没有调整。我们在五个GLUE数据集、两个模型（DistilBERT和DistilRoBERTa）和七个常用的优化器（SGD、带动量的SGD、Adam、AdaMax、Nadam、AdamW和AdaBound）上进行实验发现，当调整优化器的超参数时，尽管训练损失有所不同，五个更复杂的（自适应）优化器在测试性能上没有实质性的差异。此外，只调整学习率在大多数情况下与调整所有超参数的效果相当。因此，我们建议选择表现最好的任何自适应优化器（例如，

    NLP research has explored different neural model architectures and sizes, datasets, training objectives, and transfer learning techniques. However, the choice of optimizer during training has not been explored as extensively. Typically, some variant of Stochastic Gradient Descent (SGD) is employed, selected among numerous variants, using unclear criteria, often with minimal or no tuning of the optimizer's hyperparameters. Experimenting with five GLUE datasets, two models (DistilBERT and DistilRoBERTa), and seven popular optimizers (SGD, SGD with Momentum, Adam, AdaMax, Nadam, AdamW, and AdaBound), we find that when the hyperparameters of the optimizers are tuned, there is no substantial difference in test performance across the five more elaborate (adaptive) optimizers, despite differences in training loss. Furthermore, tuning just the learning rate is in most cases as good as tuning all the hyperparameters. Hence, we recommend picking any of the best-behaved adaptive optimizers (e.g.,
    
[^78]: LiFi: 使用精细控制代码的轻量级可控文本生成

    LiFi: Lightweight Controlled Text Generation with Fine-Grained Control Codes

    [https://arxiv.org/abs/2402.06930](https://arxiv.org/abs/2402.06930)

    LIFI是一种轻量级的可控文本生成方法，使用精细控制代码实现更精确的控制。通过连续、相对和非排他的控制代码的引导，LIFI可以在训练中学习可控文本生成。使用属性分类器自动导出的精细代码提供了更广泛的监督信号。与此同时，通过与适配器相结合，LIFI实现了高效的控制。

    

    在文本生成领域不断发展的过程中，对更精确的控制机制的需求变得越来越明显。为了满足这一需求，我们提出了一种新的方法LIFI，它提供了一种轻量级的方式，使用精细控制代码进行可控文本生成。不同于以前的研究将预训练语言模型训练来遵循离散、分类和互斥的控制代码，LIFI在连续、相对和非排他的控制代码的引导下学习可控文本生成。这些精细的代码是通过属性分类器自动导出的，它最初使用少量的标记数据进行训练，然后用来标记大量的未标记数据，从而获得了更广泛的监督信号。此外，为了实现高效的控制，我们将精细控制代码与适配器相结合，适配器是一种参数和计算效率高的方法，用于引导预训练的语言模型。我们对LIFI在两个传统任务中进行了评估--句子生成和文本分类--并取得了良好的性能。

    In the rapidly evolving field of text generation, the demand for more precise control mechanisms has become increasingly apparent. To address this need, we present a novel methodology, LIFI, which offers a lightweight approach with fine-grained control for controlled text generation. Unlike previous studies that train pre-trained language models to follow discrete, categorical, and exclusive control codes, LIFI learns controlled text generation under the guidance of continuous, relative, and nonexclusive control codes. These fine-grained codes are automatically derived from an attribute classifier, initially trained with a small amount of labeled data and subsequently employed to label abundant unlabeled data, thus garnering more extensive supervision signals. Moreover, to achieve efficient control, we incorporate the fine-grained control codes with adapters, a parameter- and compute-efficient way to steer a pre-trained language model. We evaluate LIFI on two conventional tasks -- sent
    
[^79]: LLM时代解码方法的综合研究

    A Thorough Examination of Decoding Methods in the Era of LLMs

    [https://arxiv.org/abs/2402.06925](https://arxiv.org/abs/2402.06925)

    在LLMs的背景下，本文综合研究了各种解码方法的性能、鲁棒性和解码速度，并发现解码方法的性能与任务相关，受到对齐、模型大小和量化等因素影响；某些方法可以通过大量超参数调整达到更好的性能，但需要权衡取舍。

    

    解码方法在将语言模型从下一个标记预测器转换为实际任务解决器中起着不可或缺的作用。以往关于解码方法的研究主要集中在任务特定模型上，可能不适用于当前通用型大型语言模型(LLMs)的时代。此外，最近解码策略的涌入进一步复杂了这个领域。本文在LLMs的背景下，对各种解码方法进行了全面而多方位的分析，评估了它们在各种任务、模型和部署环境中的性能、对超参数变化的鲁棒性以及解码速度。我们的研究结果表明，解码方法的性能明显与任务相关，并受到对齐、模型大小和量化等因素的影响。有趣的是，敏感性分析揭示了某些方法在需要进行大量超参数调整的前提下能够实现更好的性能，突出了在达到最佳性能之间的权衡关系。

    Decoding methods play an indispensable role in converting language models from next-token predictors into practical task solvers. Prior research on decoding methods, primarily focusing on task-specific models, may not extend to the current era of general-purpose large language models (LLMs). Moreover, the recent influx of decoding strategies has further complicated this landscape. This paper provides a comprehensive and multifaceted analysis of various decoding methods within the context of LLMs, evaluating their performance, robustness to hyperparameter changes, and decoding speeds across a wide range of tasks, models, and deployment environments. Our findings reveal that decoding method performance is notably task-dependent and influenced by factors such as alignment, model size, and quantization. Intriguingly, sensitivity analysis exposes that certain methods achieve superior performance at the cost of extensive hyperparameter tuning, highlighting the trade-off between attaining opt
    
[^80]: 用直接的两两比较方法生成思维链，以搜索最有潜力的中间思维

    Generating Chain-of-Thoughts with a Direct Pairwise-Comparison Approach to Searching for the Most Promising Intermediate Thought

    [https://arxiv.org/abs/2402.06918](https://arxiv.org/abs/2402.06918)

    本文提出了一种基于直接两两比较的方法，通过利用LLMs的噪声反馈，直接识别出最有潜力的中间思维，从而生成优秀的思维链。

    

    为了提高大型语言模型(LLMs)处理复杂推理问题的能力，提出了思维链(Chain-of-Thoughts, CoT)方法，用于指导LLMs进行逐步推理，从简单到复杂的问题解决。目前最先进的生成这种思维链的方法涉及互动协作，学习者生成候选中间思维，由LLMs评估，引导生成后续思维。然而，一个广泛但未被充分研究的问题是，LLMs的评估通常存在噪声和不可靠性，可能误导生成过程，选择不够有潜力的中间思维。本文受Vapnik原则的启发，提出了一种新的基于比较的CoT生成算法，直接根据LLMs的噪声反馈确定最有潜力的思维。在每一轮中，我们随机配对中间思维，并直接促使LLMs从每对中选择更有潜力的思维。

    To improve the ability of the large language model (LLMs) to handle complex reasoning problems, chain-of-thoughts (CoT) methods were proposed to guide LLMs to reason step-by-step, facilitating problem solving from simple to complex tasks. State-of-the-art approaches for generating such a chain involve interactive collaboration, where the learner generates candidate intermediate thoughts, evaluated by the LLM, guiding the generation of subsequent thoughts. However, a widespread yet understudied problem is that the evaluation from the LLM is typically noisy and unreliable, potentially misleading the generation process in selecting promising intermediate thoughts. In this paper, motivated by Vapnik's principle, we propose a novel comparison-based CoT generation algorithm that directly identifies the most promising thoughts with the noisy feedback from the LLM. In each round, we randomly pair intermediate thoughts and directly prompt the LLM to select the more promising one from each pair,
    
[^81]: TL;DR进展：多方面文献探索在文本摘要中的应用

    TL;DR Progress: Multi-faceted Literature Exploration in Text Summarization

    [https://arxiv.org/abs/2402.06913](https://arxiv.org/abs/2402.06913)

    本文介绍了一种名为TL;DR Progress的新工具，用于探索神经文本摘要领域的文献。该工具根据全面的注释方案对514篇论文进行了组织，实现了细粒度的、多方面的检索，并为每篇论文提供了简洁的摘要。

    

    本文介绍了TL;DR进展，一种用于探索神经文本摘要领域文献的新工具。它根据一个全面的注释方案对514篇论文进行了组织，实现了细粒度的、多方面的检索。每篇论文都经过手工注释，捕捉了评估指标、质量维度、学习范式、解决的挑战、数据集和文档领域等方面。此外，每篇论文还提供了简洁的摘要，包括自动提取的上下文因素、问题和解决方案。该工具可在线访问https://www.tldr-progress.de，并提供演示视频https://youtu.be/uCVRGFvXUj8。

    This paper presents TL;DR Progress, a new tool for exploring the literature on neural text summarization. It organizes 514~papers based on a comprehensive annotation scheme for text summarization approaches and enables fine-grained, faceted search. Each paper was manually annotated to capture aspects such as evaluation metrics, quality dimensions, learning paradigms, challenges addressed, datasets, and document domains. In addition, a succinct indicative summary is provided for each paper, consisting of automatically extracted contextual factors, issues, and proposed solutions. The tool is available online at https://www.tldr-progress.de, a demo video at https://youtu.be/uCVRGFvXUj8
    
[^82]: 调查查询导向的会议摘要的一致性：不同嵌入方法的比较研究

    Investigating Consistency in Query-Based Meeting Summarization: A Comparative Study of Different Embedding Methods

    [https://arxiv.org/abs/2402.06907](https://arxiv.org/abs/2402.06907)

    本研究比较了不同嵌入方法在查询导向的会议摘要中的一致性，旨在解决复杂、多主题和多人参与的重要会议记录中的信息查找问题。

    

    随着越来越多的先进数据分析技术的出现，人们希望这些技术能在更复杂的任务中应用，并解决我们日常生活中的问题。文本摘要是自然语言处理（NLP）领域中的一种著名应用，其旨在根据给定的上下文自动生成重要信息的摘要，当您必须处理大量文档时，这点尤为重要。摘要技术可以帮助快速捕捉关键点，并在工作中提供便利。其中一个适用的情况是会议摘要，特别是针对较长、复杂、多主题和多人参与的重要会议。因此，当人们想要从会议中审查特定内容时，将很难且耗时找到会议记录中的相关片段。然而，大部分先前的工作都集中在为新闻稿、科学文章等做摘要，这些文档具有明确的结构和官方的摘要。

    With more and more advanced data analysis techniques emerging, people will expect these techniques to be applied in more complex tasks and solve problems in our daily lives. Text Summarization is one of famous applications in Natural Language Processing (NLP) field. It aims to automatically generate summary with important information based on a given context, which is important when you have to deal with piles of documents. Summarization techniques can help capture key points in a short time and bring convenience in works. One of applicable situation is meeting summarization, especially for important meeting that tend to be long, complicated, multi-topic and multi-person. Therefore, when people want to review specific content from a meeting, it will be hard and time-consuming to find the related spans in the meeting transcript. However, most of previous works focus on doing summarization for newsletters, scientific articles...etc, which have a clear document structure and an official f
    
[^83]: LLM能够识别毒性吗？结构化毒性调查框架和基于语义的度量

    Can LLMs Recognize Toxicity? Structured Toxicity Investigation Framework and Semantic-Based Metric

    [https://arxiv.org/abs/2402.06900](https://arxiv.org/abs/2402.06900)

    本研究提出了一种基于大型语言模型（LLMs）的自动度量方法，用于识别生成文本中的毒性。通过分析毒性因素和LLMs的内在毒性属性，该方法在测量毒性方面表现出众，比现有指标提升12个百分点。

    

    在开发遵守社会标准的大型语言模型（LLMs）的过程中，识别生成文本中的毒性存在至关重要。现有的大多数毒性度量依赖于在特定毒性数据集上训练的编码模型。然而，这些编码器容易受到分布外的问题的影响，并且依赖于数据集中所假定的毒性定义。本文介绍了一种基于LLMs的自动鲁棒度量，用于区分模型回应是否具有毒性。我们首先分析了毒性因素，然后研究了LLMs的内在毒性属性，以确定它们作为评估器的适用性。随后，我们对评估数据集上的度量指标LLMs As ToxiciTy Evaluators（LATTE）进行了评估。实证结果表明，在不进行训练过程的情况下，我们的度量在测量毒性方面表现出色，F1得分比现有技术指标提高了12个百分点。我们还展示了上游毒性对度量结果的影响。

    In the pursuit of developing Large Language Models (LLMs) that adhere to societal standards, it is imperative to discern the existence of toxicity in the generated text. The majority of existing toxicity metrics rely on encoder models trained on specific toxicity datasets. However, these encoders are susceptible to out-of-distribution (OOD) problems and depend on the definition of toxicity assumed in a dataset. In this paper, we introduce an automatic robust metric grounded on LLMs to distinguish whether model responses are toxic. We start by analyzing the toxicity factors, followed by examining the intrinsic toxic attributes of LLMs to ascertain their suitability as evaluators. Subsequently, we evaluate our metric, LLMs As ToxiciTy Evaluators (LATTE), on evaluation datasets.The empirical results indicate outstanding performance in measuring toxicity, improving upon state-of-the-art metrics by 12 points in F1 score without training procedure. We also show that upstream toxicity has an 
    
[^84]: GenTranslate: 大型语言模型是生成的多语言语音和机器翻译工具

    GenTranslate: Large Language Models are Generative Multilingual Speech and Machine Translators

    [https://arxiv.org/abs/2402.06894](https://arxiv.org/abs/2402.06894)

    GenTranslate是一个新的翻译任务生成模型，通过利用大型语言模型的丰富语言知识和强大推理能力，可以从N-best列表中生成更高质量的翻译结果。

    

    大型语言模型（LLMs）的最新进展通过减少表示误差和引入外部知识，推动了多语言语音和机器翻译的发展。然而，翻译任务通常使用束搜索解码和前k个假设选择进行推理。这些技术往往不能充分利用多样化的N-best假设中的丰富信息，使得它们在需要单个高质量输出序列的翻译任务中效果不佳。在本文中，我们提出了一个新的翻译任务生成模型，即“GenTranslate”，它基于LLMs来从N-best列表中生成更好的结果。利用LLMs丰富的语言知识和强大的推理能力，我们的新模型可以将N-best候选人中的丰富信息整合起来，生成更高质量的翻译结果。此外，为了支持LLM的微调，我们构建并发布了一个HypoTransla模型。

    Recent advances in large language models (LLMs) have stepped forward the development of multilingual speech and machine translation by its reduced representation errors and incorporated external knowledge. However, both translation tasks typically utilize beam search decoding and top-1 hypothesis selection for inference. These techniques struggle to fully exploit the rich information in the diverse N-best hypotheses, making them less optimal for translation tasks that require a single, high-quality output sequence. In this paper, we propose a new generative paradigm for translation tasks, namely "GenTranslate", which builds upon LLMs to generate better results from the diverse translation versions in N-best list. Leveraging the rich linguistic knowledge and strong reasoning abilities of LLMs, our new paradigm can integrate the rich information in N-best candidates to generate a higher-quality translation result. Furthermore, to support LLM finetuning, we build and release a HypoTransla
    
[^85]: 大型语言模型的历史、发展和原理-一项综述性调查

    History, Development, and Principles of Large Language Models-An Introductory Survey

    [https://arxiv.org/abs/2402.06853](https://arxiv.org/abs/2402.06853)

    这项综述性调查介绍了大型语言模型（LLMs）的历史、发展和原理，旨在帮助广泛的读者群体理解这些模型的背景和原理。

    

    语言模型作为自然语言处理中的重要基石，利用数学方法来推广语言规律和知识，用于预测和生成。经过几十年的广泛研究，语言建模从最初的统计语言模型（SLMs）发展到当今的大型语言模型（LLMs）。值得注意的是，LLMs的快速演进已经达到了处理、理解和生成人类水平文本的能力。然而，尽管LLMs在改善工作和个人生活方面具有显著优势，但一般从业人员对这些模型的背景和原理了解有限，限制了它们的应用潜力。值得注意的是，大多数关于LLMs的综述都集中在特定方面，并使用了专门的语言，给缺乏相关背景知识的从业人员带来了困难。因此，本综述旨在提供一个简明扼要的LLMs概述，以帮助更广泛的读者群体。

    Language models serve as a cornerstone in natural language processing (NLP), utilizing mathematical methods to generalize language laws and knowledge for prediction and generation. Over extensive research spanning decades, language modeling has progressed from initial statistical language models (SLMs) to the contemporary landscape of large language models (LLMs). Notably, the swift evolution of LLMs has reached the ability to process, understand, and generate human-level text. Nevertheless, despite the significant advantages that LLMs offer in improving both work and personal lives, the limited understanding among general practitioners about the background and principles of these models hampers their full potential. Notably, most LLMs reviews focus on specific aspects and utilize specialized language, posing a challenge for practitioners lacking relevant background knowledge. In light of this, this survey aims to present a comprehensible overview of LLMs to assist a broader audience. 
    
[^86]: ChemLLM: 一个化学大型语言模型

    ChemLLM: A Chemical Large Language Model

    [https://arxiv.org/abs/2402.06852](https://arxiv.org/abs/2402.06852)

    ChemLLM是第一个专门用于化学领域的大型语言模型，利用新颖的指令构建方法将结构化知识转化为对话形式，具有平滑对话交互的能力，并在化学的三个主要任务中击败了GPT-3.5。

    

    大型语言模型（LLM）在化学应用中取得了令人瞩目的进展，包括分子属性预测、分子生成、实验协议设计等。然而，该领域缺乏一个专门针对化学领域设计的基于对话的模型。这个挑战来自于事实，大多数化学数据和科学知识主要存储在结构化数据库中，直接使用这些结构化数据会影响模型维持连贯对话的能力。为了解决这个问题，我们开发了一种新颖的基于模板的指令构建方法，将结构化知识转化为简洁对话形式，适合于语言模型的训练。通过利用这种方法，我们开发了ChemLLM，第一个专门用于化学的大型语言模型，能够在化学领域的各种任务中进行平滑对话交互。ChemLLM在化学的三个主要任务，即名称转换、分子生成和实验协议设计方面，击败了GPT-3.5。

    Large language models (LLMs) have made impressive progress in chemistry applications, including molecular property prediction, molecular generation, experimental protocol design, etc. However, the community lacks a dialogue-based model specifically designed for chemistry. The challenge arises from the fact that most chemical data and scientific knowledge are primarily stored in structured databases, and the direct use of these structured data compromises the model's ability to maintain coherent dialogue. To tackle this issue, we develop a novel template-based instruction construction method that transforms structured knowledge into plain dialogue, making it suitable for language model training. By leveraging this approach, we develop ChemLLM, the first large language model dedicated to chemistry, capable of performing various tasks across chemical disciplines with smooth dialogue interaction. ChemLLM beats GPT-3.5 on all three principal tasks in chemistry, i.e., name conversion, molecu
    
[^87]: 与更有说服力的LLMs辩论会导致更真实的回答

    Debating with More Persuasive LLMs Leads to More Truthful Answers

    [https://arxiv.org/abs/2402.06782](https://arxiv.org/abs/2402.06782)

    本文研究了更弱的语言模型是否能评估更强的模型的正确性。研究发现，通过进行辩论，非专家模型和人类回答问题的准确性都有所提高。

    

    与所需行为一致的大型语言模型（LLM）的常见方法主要依赖于人工标注的数据。然而，随着模型变得越来越复杂，它们将超过人类专业知识，人类评估的角色将演变为非专家监督专家。在此之前，我们问：更弱的模型能评估更强的模型的正确性吗？我们在类似的环境中调查了这个问题，其中更强的模型（专家）拥有回答问题所需的信息，而更弱的模型（非专家）缺乏这些信息。我们评估的方法是\textit{辩论}，其中两个LLM专家分别支持不同的答案，一个非专家选择答案。我们发现辩论 consistently帮助非专家模型和人类回答问题，分别达到76%和88%的准确性（朴素基准分别为48%和60%）。此外，以无监督方式优化专家辩论者的说服力会提高非专家的能力。

    Common methods for aligning large language models (LLMs) with desired behaviour heavily rely on human-labelled data. However, as models grow increasingly sophisticated, they will surpass human expertise, and the role of human evaluation will evolve into non-experts overseeing experts. In anticipation of this, we ask: can weaker models assess the correctness of stronger models? We investigate this question in an analogous setting, where stronger models (experts) possess the necessary information to answer questions and weaker models (non-experts) lack this information. The method we evaluate is \textit{debate}, where two LLM experts each argue for a different answer, and a non-expert selects the answer. We find that debate consistently helps both non-expert models and humans answer questions, achieving 76\% and 88\% accuracy respectively (naive baselines obtain 48\% and 60\%). Furthermore, optimising expert debaters for persuasiveness in an unsupervised manner improves non-expert abilit
    
[^88]: 文本数据增强在自然语言处理中的评估指标

    Evaluation Metrics for Text Data Augmentation in NLP

    [https://arxiv.org/abs/2402.06766](https://arxiv.org/abs/2402.06766)

    这项研究提供了文本增强方法的评估指标分类法，为统一基准提供方向和帮助。在不同任务、度量标准和实验设置下，该分类法有助于比较不同的增强方法。

    

    最近关于自然语言处理的数据增强的调研报告指出了该领域的不同技术和进展。几个框架、工具和存储库推广了文本数据增强流水线的实施。然而，由于不同的任务、度量标准、数据集、体系结构和实验设置的缺乏评估标准和方法比较标准使得比较变得毫无意义。此外，缺乏方法的统一性，文本数据增强研究将受益于比较不同的增强方法的统一指标。因此，学术界和工业界都在努力寻找相关的文本数据增强技术的评估指标。本研究的贡献在于提供了文本增强方法的评估指标分类法，并作为统一基准的方向。所提出的分类法包括了实施工具和指标计算的类别。最后，通过这项研究，我们的目的是为了推进文本数据增强技术的评估指标的发展。

    Recent surveys on data augmentation for natural language processing have reported different techniques and advancements in the field. Several frameworks, tools, and repositories promote the implementation of text data augmentation pipelines. However, a lack of evaluation criteria and standards for method comparison due to different tasks, metrics, datasets, architectures, and experimental settings makes comparisons meaningless. Also, a lack of methods unification exists and text data augmentation research would benefit from unified metrics to compare different augmentation methods. Thus, academics and the industry endeavor relevant evaluation metrics for text data augmentation techniques. The contribution of this work is to provide a taxonomy of evaluation metrics for text augmentation methods and serve as a direction for a unified benchmark. The proposed taxonomy organizes categories that include tools for implementation and metrics calculation. Finally, with this study, we intend to 
    
[^89]: EntGPT: 将生成型大型语言模型与知识库相连接

    EntGPT: Linking Generative Large Language Models with Knowledge Bases

    [https://arxiv.org/abs/2402.06738](https://arxiv.org/abs/2402.06738)

    本文介绍了一种名为EntGPT的模型，通过Entity Disambiguation（ED）任务，连接了生成型大型语言模型与知识库。通过提示工程和指令调整，该模型在没有有监督微调的情况下，显著提高了LLMs的性能，并在实体消歧任务上取得了可比较的性能。

    

    由于训练和推理过程中缺乏事实核实和知识基础，大型语言模型（LLM）生成的事实正确输出的能力相对较少被研究。在这项工作中，我们通过Entity Disambiguation（ED）任务来解决这一挑战。我们首先考虑了提示工程，并设计了一个三步硬提示方法，以在没有有监督微调（SFT）的情况下探测LLM的ED性能。总体而言，该提示方法显著提高了原始基准模型的微F_1得分，在某些情况下提高了36%甚至更高，并在10个数据集上与现有的SFT方法相比，获得了可比较的性能。我们通过使用类似的提示和响应进行指令调整（IT）进一步提高了知识基础。指令调整的模型在受监督实体消歧任务上不仅实现了更高的微F1得分性能，而且平均微F_1提高了。

    The ability of Large Language Models (LLMs) to generate factually correct output remains relatively unexplored due to the lack of fact-checking and knowledge grounding during training and inference. In this work, we aim to address this challenge through the Entity Disambiguation (ED) task. We first consider prompt engineering, and design a three-step hard-prompting method to probe LLMs' ED performance without supervised fine-tuning (SFT). Overall, the prompting method improves the micro-F_1 score of the original vanilla models by a large margin, on some cases up to 36% and higher, and obtains comparable performance across 10 datasets when compared to existing methods with SFT. We further improve the knowledge grounding ability through instruction tuning (IT) with similar prompts and responses. The instruction-tuned model not only achieves higher micro-F1 score performance as compared to several baseline methods on supervised entity disambiguation tasks with an average micro-F_1 improve
    
[^90]: NICE: 优化上下文示例还是不优化？

    NICE: To Optimize In-Context Examples or Not?

    [https://arxiv.org/abs/2402.06733](https://arxiv.org/abs/2402.06733)

    通过研究在提供任务特定指令的情况下是否需要优化上下文示例，我们挑战了对于指导性LLMs的共识，并发现在某些任务中，不同的优化上下文示例方法会产生递减的回报。我们引入了"度量标准"，用于衡量从给定指令中学习任务的能力，并提供了一个启发式方法，帮助决定是否优化指令还是ICE用于任何新任务。

    

    最近的研究表明，大型语言模型（LLMs）通过上下文学习和优化上下文示例（ICE），在各种任务上表现出色。然而，大多数研究假设在提示信息中要么是固定的，要么没有提供指令，导致了一个表面上的共识：优化上下文示例对于提高性能至关重要。我们针对经过指导的LLMs挑战这一共识，研究在提供了任务特定指令的情况下优化上下文示例是否必要，并发现有一些任务对于不同的优化上下文示例方法产生递减的回报。我们引入了一种任务特定的度量标准，称为"度量标准"（Metric），用于量化从给定指令中学习任务的能力，并提供了一个启发式方法，帮助决定是否优化指令还是ICE用于任何新任务。通过对各种任务和逐步增加的指令集的系统性研究，我们验证了该启发式方法的有效性。

    Recent works have shown that large language models (LLMs) work remarkably well on a wide range of tasks through in-context learning and optimization of in-context examples (ICE). However, most of these studies assume either a fixed or no instruction provided in the prompt, leading to the apparent consensus that the optimization of in-context examples is critical for better performance. We challenge this consensus for instruction-tuned LLMs by investigating the necessity of optimizing in-context examples when task-specific instructions are provided, and find that there are tasks for which various ways of optimizing in-context examples yield diminishing returns. We introduce a task-specific metric called \metriclong{} (\metric) that quantifies the learnability of tasks from a given instruction, and provides a heuristic that helps decide whether to optimize for instructions or ICE for any new task. On a wide range of tasks and a systematically created instruction set with gradually added 
    
[^91]: 源代码合成和补全的神经模型

    Neural Models for Source Code Synthesis and Completion

    [https://arxiv.org/abs/2402.06690](https://arxiv.org/abs/2402.06690)

    本论文提出了一种基于序列到序列的深度学习模型和训练方法，用于将自然语言转化为可编译的代码片段，并为开发人员提供源代码建议和自动补全功能。这一方法能够提取开发者的编码意图并准确推断类型、名称和上下文等信息。

    

    自然语言（NL）到代码建议系统通过将NL表达转化为可编译的代码片段来帮助集成开发环境（IDE）中的开发人员。当前的方法主要涉及基于语义解析的硬编码、规则系统。这些系统主要依靠手工制定的规则将NL的模式或其语法解析树中的元素映射到各种查询结构，并且只能处理受限制的NL子集和限制的NL语法。这些系统无法从开发者的编码意图中提取语义信息，常常无法推断类型、名称和源代码的上下文以获得准确的系统级代码建议。在本硕士论文中，我们提出了序列到序列的深度学习模型和训练范式，以将NL映射到通用编程语言，可以根据NL的意图为用户提供源代码片段的建议，并扩展源代码的自动补全功能。

    Natural language (NL) to code suggestion systems assist developers in Integrated Development Environments (IDEs) by translating NL utterances into compilable code snippet. The current approaches mainly involve hard-coded, rule-based systems based on semantic parsing. These systems make heavy use of hand-crafted rules that map patterns in NL or elements in its syntax parse tree to various query constructs and can only work on a limited subset of NL with a restricted NL syntax. These systems are unable to extract semantic information from the coding intents of the developer, and often fail to infer types, names, and the context of the source code to get accurate system-level code suggestions. In this master thesis, we present sequence-to-sequence deep learning models and training paradigms to map NL to general-purpose programming languages that can assist users with suggestions of source code snippets, given a NL intent, and also extend auto-completion functionality of the source code to
    
[^92]: 基于因果关系的基础世界模型在具身人工智能中的重要作用

    The Essential Role of Causality in Foundation World Models for Embodied AI

    [https://arxiv.org/abs/2402.06665](https://arxiv.org/abs/2402.06665)

    基于因果关系的基础世界模型对于具身人工智能的发展至关重要，当前的基础模型无法准确建模与现实世界的物理相互作用。因果关系的研究有助于构建真实世界模型，提高对可能相互作用结果的准确预测能力。

    

    最近在基础模型中取得的进展，尤其是在大型多模态模型和对话代理方面，引发了对具备普遍能力的具身代理人潜力的兴趣。这样的代理人需要能够在许多不同的真实世界环境中执行新任务。然而，当前的基础模型未能准确建模与现实世界的物理相互作用，因此对于具身人工智能而言是不够的。因果关系的研究有助于构建真实世界模型，这对于准确预测可能相互作用的结果至关重要。本文着重探讨了为即将到来的具身代理生成基础世界模型的前景，并对其中的因果关系的重要性提出了新的观点。我们认为整合因果关系是促进与世界的有意义的物理相互作用至关重要的。最后，我们揭示了这一背景下对因果关系的误解，并展示了我们对未来的展望。

    Recent advances in foundation models, especially in large multi-modal models and conversational agents, have ignited interest in the potential of generally capable embodied agents. Such agents would require the ability to perform new tasks in many different real-world environments. However, current foundation models fail to accurately model physical interactions with the real world thus not sufficient for Embodied AI. The study of causality lends itself to the construction of veridical world models, which are crucial for accurately predicting the outcomes of possible interactions. This paper focuses on the prospects of building foundation world models for the upcoming generation of embodied agents and presents a novel viewpoint on the significance of causality within these. We posit that integrating causal considerations is vital to facilitate meaningful physical interactions with the world. Finally, we demystify misconceptions about causality in this context and present our outlook fo
    
[^93]: 对抗性文本净化：一种基于大型语言模型的防御方法

    Adversarial Text Purification: A Large Language Model Approach for Defense

    [https://arxiv.org/abs/2402.06655](https://arxiv.org/abs/2402.06655)

    本文研究了防御文本分类器中对抗性净化方法的有效性，并提出了一种基于大型语言模型加以净化的方法。

    

    对抗性净化是一种防御机制，用于保护分类器免受对抗性攻击，而无需了解攻击类型或分类器的训练。这些技术对被攻击输入进行特征化和消除对抗性扰动，旨在恢复出与最初被攻击的输入相似且被分类器正确分类的净化样本。由于离散输入的噪声扰动特征化所带来的固有挑战，对抗性文本净化一直相对未被探索。在本文中，我们研究了对抗性净化方法在保护文本分类器中的有效性。我们提出了一种新颖的对抗性文本净化方法，利用大型语言模型（LLMs）的生成能力来净化对抗性文本，而无需明确特征化离散噪声扰动。我们利用提示工程来利用LLMs恢复净化的示例。

    Adversarial purification is a defense mechanism for safeguarding classifiers against adversarial attacks without knowing the type of attacks or training of the classifier. These techniques characterize and eliminate adversarial perturbations from the attacked inputs, aiming to restore purified samples that retain similarity to the initially attacked ones and are correctly classified by the classifier. Due to the inherent challenges associated with characterizing noise perturbations for discrete inputs, adversarial text purification has been relatively unexplored. In this paper, we investigate the effectiveness of adversarial purification methods in defending text classifiers. We propose a novel adversarial text purification that harnesses the generative capabilities of Large Language Models (LLMs) to purify adversarial text without the need to explicitly characterize the discrete noise perturbations. We utilize prompt engineering to exploit LLMs for recovering the purified examples for
    
[^94]: SocraSynth:基于条件统计的多语言模型推理系统

    SocraSynth: Multi-LLM Reasoning with Conditional Statistics

    [https://arxiv.org/abs/2402.06634](https://arxiv.org/abs/2402.06634)

    SocraSynth是一个多语言模型推理平台，通过使用条件统计和系统化的语境增强技术，以及可调节的辩论争议程度，解决了大型语言模型(LLMs)面临的偏见、幻觉和推理能力不足等问题。

    

    大型语言模型(LLMs)在实用上面临着偏见、幻觉和推理能力不足等问题。本文介绍了SocraSynth，这是一个多语言模型(LLM)推理平台，旨在解决这些问题。SocraSynth通过连续的论证和可调节的争议程度，利用条件统计和系统化的语境增强，充分发挥了多语言模型(LLM)的优势。该平台通常由一个人类主持者和两个代表互相对抗立场的LLM代理组成。SocraSynth分为两个主要阶段：知识生成和推理评估。在知识生成阶段，主持者定义了辩论话题和争议程度，促使代理商为各自的立场制定支持性的论证。然后，在推理评估阶段，采用了苏格拉底推理和形式逻辑原理来评估所提出的论证的质量。对话以主持者调整争议程度结束。

    Large language models (LLMs), while promising, face criticisms for biases, hallucinations, and a lack of reasoning capability. This paper introduces SocraSynth, a multi-LLM agent reasoning platform developed to mitigate these issues. SocraSynth utilizes conditional statistics and systematic context enhancement through continuous arguments, alongside adjustable debate contentiousness levels. The platform typically involves a human moderator and two LLM agents representing opposing viewpoints on a given subject. SocraSynth operates in two main phases: knowledge generation and reasoning evaluation. In the knowledge generation phase, the moderator defines the debate topic and contentiousness level, prompting the agents to formulate supporting arguments for their respective stances. The reasoning evaluation phase then employs Socratic reasoning and formal logic principles to appraise the quality of the arguments presented. The dialogue concludes with the moderator adjusting the contentiousn
    
[^95]: LightCAM: 一种快速轻量级的基于上下文感知屏蔽的D-TDNN说话人验证实现

    LightCAM: A Fast and Light Implementation of Context-Aware Masking based D-Tdnn for Speaker Verification

    [https://arxiv.org/abs/2402.06073](https://arxiv.org/abs/2402.06073)

    LightCAM是一种快速轻量级的基于上下文感知屏蔽的D-TDNN说话人验证实现，通过采用深度可分离卷积模块和多尺度特征聚合，它在VoxCeleb数据集上取得了更好的性能。

    

    传统的时延神经网络(TDNN)在计算复杂度和推理速度方面取得了最先进的性能，使得它们在工业环境中难以实施。具有上下文感知屏蔽(CAM)模块的密集连通时延神经网络(D-TDNN)已经证明是一种降低复杂性并保持系统性能的高效结构。本文提出了一种快速轻量级模型LightCAM，它进一步采用了深度可分离卷积模块(DSM)和多尺度特征聚合(MFA)以实现不同层次的特征融合。在VoxCeleb数据集上进行了大量实验，比较结果表明它在VoxCeleb1-O上实现了0.83的等错误率(EER)和0.0891的最小检测代价因子(MinDCF)，超过了其他主流的说话人验证方法。此外，复杂度分析进一步证明了所提出的架构具有较低的计算成本和更快的推理速度。

    Traditional Time Delay Neural Networks (TDNN) have achieved state-of-the-art performance at the cost of high computational complexity and slower inference speed, making them difficult to implement in an industrial environment. The Densely Connected Time Delay Neural Network (D-TDNN) with Context Aware Masking (CAM) module has proven to be an efficient structure to reduce complexity while maintaining system performance. In this paper, we propose a fast and lightweight model, LightCAM, which further adopts a depthwise separable convolution module (DSM) and uses multi-scale feature aggregation (MFA) for feature fusion at different levels. Extensive experiments are conducted on VoxCeleb dataset, the comparative results show that it has achieved an EER of 0.83 and MinDCF of 0.0891 in VoxCeleb1-O, which outperforms the other mainstream speaker verification methods. In addition, complexity analysis further demonstrates that the proposed architecture has lower computational cost and faster inf
    
[^96]: 生成性回音室？LLM驱动的搜索系统对多样化信息搜索的影响

    Generative Echo Chamber? Effects of LLM-Powered Search Systems on Diverse Information Seeking

    [https://arxiv.org/abs/2402.05880](https://arxiv.org/abs/2402.05880)

    LLM驱动的对话式搜索系统增加了选择性曝光，且支持用户观点的有偏见LLM加剧了这种偏差。

    

    数亿人已经使用过大型语言模型（LLM）驱动的对话式搜索系统，并且相信这些系统相比传统搜索带来了许多好处。然而，虽然几十年的研究和公共讨论都调查了搜索系统在增加选择性曝光和产生回音室方面的风险，即限制接触多样化意见并导致意见偏执，但对于LLM驱动的对话式搜索的这种风险知之甚少。我们进行了两个实验来研究：1）LLM驱动的对话式搜索相较于传统搜索是否以及如何增加选择性曝光；2）具有支持或挑战用户观点的意见偏见的LLM如何改变这种影响。总体而言，我们发现参与者在LLM驱动的对话式搜索中更倾向于进行偏见的信息查询，并且支持他们观点的有偏见的LLM加剧了这种偏差。这些结果呈现了重要的意义。

    Large language models (LLMs) powered conversational search systems have already been used by hundreds of millions of people, and are believed to bring many benefits over conventional search. However, while decades of research and public discourse interrogated the risk of search systems in increasing selective exposure and creating echo chambers -- limiting exposure to diverse opinions and leading to opinion polarization, little is known about such a risk of LLM-powered conversational search. We conduct two experiments to investigate: 1) whether and how LLM-powered conversational search increases selective exposure compared to conventional search; 2) whether and how LLMs with opinion biases that either reinforce or challenge the user's view change the effect. Overall, we found that participants engaged in more biased information querying with LLM-powered conversational search, and an opinionated LLM reinforcing their views exacerbated this bias. These results present critical implicatio
    
[^97]: PromptCrypt: 使用表情符号对大型语言模型进行安全通信的提示加密

    PromptCrypt: Prompt Encryption for Secure Communication with Large Language Models

    [https://arxiv.org/abs/2402.05868](https://arxiv.org/abs/2402.05868)

    PromptCrypt是一种使用表情符号对用户输入进行加密的机制，保护了大型语言模型（LLM）中用户的隐私，防止数据泄露和解密。

    

    基于云的大型语言模型（LLM）如ChatGPT在日常操作中变得越来越重要，成为各种应用程序中的重要工具。虽然这些模型在可访问性和功能性方面带来了重大好处，但它们也引入了重要的隐私问题：在云基础架构中传输和存储用户数据会产生重大的数据泄露和未经授权访问敏感信息的风险；即使数据的传输和存储被加密，LLM服务提供商仍然知道数据的真实内容，从而阻止个人或实体放心使用此类LLM服务。为了解决这些问题，本文提出了一种简单但有效的机制PromptCrypt来保护用户隐私。它使用表情符号对用户输入进行加密，然后将其发送到LLM，有效地使其对人类或LLM的检查无法理解，同时保留原始提示的意图，从而确保用户隐私。

    Cloud-based large language models (LLMs) such as ChatGPT have increasingly become integral to daily operations, serving as vital tools across various applications. While these models offer substantial benefits in terms of accessibility and functionality, they also introduce significant privacy concerns: the transmission and storage of user data in cloud infrastructures pose substantial risks of data breaches and unauthorized access to sensitive information; even if the transmission and storage of data is encrypted, the LLM service provider itself still knows the real contents of the data, preventing individuals or entities from confidently using such LLM services. To address these concerns, this paper proposes a simple yet effective mechanism PromptCrypt to protect user privacy. It uses Emoji to encrypt the user inputs before sending them to LLM, effectively rendering them indecipherable to human or LLM's examination while retaining the original intent of the prompt, thus ensuring the 
    
[^98]: 使用模态相对预训练的方法生成文本到代码的转换

    Text-to-Code Generation with Modality-relative Pre-training

    [https://arxiv.org/abs/2402.05783](https://arxiv.org/abs/2402.05783)

    本论文研究了如何根据不同的模态调整和表示序列标记，以进一步提高文本到代码生成的效果。

    

    最近，大型预训练语言模型被广泛应用于编程语言任务，并取得了巨大成功，通常通过进一步预先训练严格自然语言模型，其中训练序列通常包含自然语言和（线性化的）编程语言。这种方法有效地将序列的两种模态映射到同一嵌入空间中。然而，编程语言关键词（例如“while”）往往具有非常严格的定义语义。因此，从自然语言用法进行的迁移学习对其代码应用可能并不一定有益，反之亦然。在本研究中，我们假设已经预先训练好的语言模型，探讨了如何根据它们所属的模态不同来调整和表示序列标记，并最终有益于下游任务。我们在模态相对训练目标的进一步模型预训练中尝试了在模态之间分离嵌入空间的方法。

    Large pre-trained language models have recently been expanded and applied to programming language tasks with great success, often through further pre-training of a strictly-natural language model--where training sequences typically contain both natural and (linearised) programming language. Such approaches effectively map both modalities of the sequence into the same embedding space. However, programming language keywords (e.g. ``while'') often have very strictly defined semantics. As such, transfer learning from their natural language usage may not necessarily be beneficial to their code application and vise versa. Assuming an already pre-trained language model, in this work we investigate how sequence tokens can be adapted and represented differently, depending on which modality they belong to, and to the ultimate benefit of the downstream task. We experiment with separating embedding spaces between modalities during further model pre-training with modality-relative training objectiv
    
[^99]: 现在所有人都修剪：仅使用前向传递的LLM结构化修剪

    Everybody Prune Now: Structured Pruning of LLMs with only Forward Passes

    [https://arxiv.org/abs/2402.05406](https://arxiv.org/abs/2402.05406)

    本文提出了一种仅使用前向传递的LLM结构化修剪方法，通过Bonsai生成的修剪模型在性能上优于梯度-based结构化修剪方法，并且速度是半结构化修剪模型的两倍。

    

    鉴于非专业从业者和最富有资源的机构之间的硬件差距，尺寸不断增长的LLM变得越来越难以使用。虽然提出了许多方法来压缩LLM，以使其资源消耗可管理，但这些方法本身往往耗费资源，使其目标用户群无法接触到。在这项工作中，我们探讨了仅使用前向传递的LLM结构化修剪问题。我们希望让从业者能够修剪模型，使其规模大到硬件仅有足够的内存来运行推理。我们开发了Bonsai，这是一种无梯度、扰动修剪方法，能够生成小、快和准确的修剪模型。我们观察到，Bonsai生成的修剪模型（i）优于更昂贵的梯度-based结构化修剪方法生成的模型，并且（ii）与半结构化修剪模型相比，速度快一倍且准确性相当。

    Given the generational gap in available hardware between lay practitioners and the most endowed institutions, LLMs are becoming increasingly inaccessible as they grow in size. Whilst many approaches have been proposed to compress LLMs to make their resource consumption manageable, these methods themselves tend to be resource intensive, putting them out of the reach of the very user groups they target. In this work, we explore the problem of structured pruning of LLMs using only forward passes. We seek to empower practitioners to prune models so large that their available hardware has just enough memory to run inference. We develop Bonsai, a gradient-free, perturbative pruning method capable of delivering small, fast, and accurate pruned models.   We observe that Bonsai outputs pruned models that (i) outperform those generated by more expensive gradient-based structured pruning methods, and (ii) are twice as fast (with comparable accuracy) as those generated by semi-structured pruning m
    
[^100]: 从错误中学习的上下文准则学习

    In-Context Principle Learning from Mistakes

    [https://arxiv.org/abs/2402.05403](https://arxiv.org/abs/2402.05403)

    本文提出了一种新的学习方法LEAP，通过让模型从少量输入-输出示例中犯错误，然后反思并学习准则，从而提升模型在各种任务上的表现。

    

    上下文学习（ICL，也称为少样本提示）已成为将LLMs适应下游任务的标准方法，通过从少量的输入-输出示例中学习。然而，所有基于ICL的方法只从正确的输入-输出对中学习。在本文中，我们重新审视这一范例，通过从少给定的输入-输出示例中学习更多内容。我们引入了学习准则（LEAP）：首先，我们有意诱使模型在这些少量示例中犯错误；然后，我们反思这些错误，并从中学习显式的任务特定“准则”，这些准则有助于解决类似的问题并避免常见的错误；最后，我们使用原始的少样本示例和这些学到的通用准则来提示模型回答未见过的测试问题。我们在包括多跳问题回答（Hotpot QA）、文本问题回答（DROP）、Big-Bench困难推理和数学问题（GSM8K和MATH）在内的多个基准测试上评估了LEAP；在所有这些基准测试中，LEAP都有所改进。

    In-context learning (ICL, also known as few-shot prompting) has been the standard method of adapting LLMs to downstream tasks, by learning from a few input-output examples. Nonetheless, all ICL-based approaches only learn from correct input-output pairs. In this paper, we revisit this paradigm, by learning more from the few given input-output examples. We introduce Learning Principles (LEAP): First, we intentionally induce the model to make mistakes on these few examples; then we reflect on these mistakes, and learn explicit task-specific "principles" from them, which help solve similar problems and avoid common mistakes; finally, we prompt the model to answer unseen test questions using the original few-shot examples and these learned general principles. We evaluate LEAP on a wide range of benchmarks, including multi-hop question answering (Hotpot QA), textual QA (DROP), Big-Bench Hard reasoning, and math problems (GSM8K and MATH); in all these benchmarks, LEAP improves the strongest 
    
[^101]: ApiQ：2位量化大型语言模型的微调

    ApiQ: Finetuning of 2-Bit Quantized Large Language Model

    [https://arxiv.org/abs/2402.05147](https://arxiv.org/abs/2402.05147)

    这项工作介绍了一种名为ApiQ的新型量化框架，通过同时初始化LoRA组件和量化大型语言模型的权重，恢复量化过程中丢失的信息，维持原始模型的激活精度并减轻误差传播。

    

    随着大型语言模型的增大，内存高效的模型微调近年来备受关注，主要是由于GPU内存限制和这些方法与完全微调的可比结果所带来的约束。尽管有了进展，如QLoRA这样的内存高效微调策略在不同位宽的量化和多样化任务中表现不一致。这种不一致主要来自于量化过程对保留知识的有害影响，导致灾难性遗忘，削弱了预训练模型在微调中的利用。在这项工作中，我们引入了一种名为ApiQ的新型量化框架，旨在通过同时初始化LoRA组件和量化LLM的权重来恢复量化损失的信息。这种方法确保了原始LLM的激活精度的维持，同时减轻了误差的传播。

    Memory-efficient finetuning of large language models (LLMs) has recently attracted huge attention with the increasing size of LLMs, primarily due to the constraints posed by GPU memory limitations and the comparable results of these methods with full finetuning. Despite the advancements, current strategies for memory-efficient finetuning, such as QLoRA, exhibit inconsistent performance across diverse bit-width quantizations and multifaceted tasks. This inconsistency largely stems from the detrimental impact of the quantization process on preserved knowledge, leading to catastrophic forgetting and undermining the utilization of pretrained models for finetuning purposes. In this work, we introduce a novel quantization framework named ApiQ, designed to restore the lost information from quantization by concurrently initializing LoRA components and quantizing the weights of LLMs. This approach ensures the maintenance of the original LLM's activation precision while mitigating the error prop
    
[^102]: 有效检索增强生成的财务报告切块

    Financial Report Chunking for Effective Retrieval Augmented Generation

    [https://arxiv.org/abs/2402.05131](https://arxiv.org/abs/2402.05131)

    本文提出了一种扩展的方法来切块财务报告，通过根据文档的结构元素组件进行切块，从而实现更有效的检索增强生成。这种方法可以优化切块大小，而无需调整，并提供了对整体上下文和准确性的评估以及对问答任务性能的影响。

    

    切块信息是检索增强生成(RAG)的关键步骤。目前的研究主要集中在段落级切块上。这种方法将所有文本都视为平等的，并忽略了文档结构中包含的信息。我们提出了一种扩展的方法，通过不仅仅将文档切块到段落级别，而是根据文档的结构元素组件来切块。将文档分解为这些组成元素可以创建一种新的文档切块方式，可以得到最佳的切块大小，无需调整。我们引入了一种新颖的框架，评估根据由文档理解模型注释的元素类型进行切块如何对所检索信息的整体上下文和准确性贡献。我们还演示了这种方法对RAG辅助问答任务性能的影响。我们的研究包括对各种元素类型的全面分析，它们在有效信息检索中的作用以及它们对其产生的影响。

    Chunking information is a key step in Retrieval Augmented Generation (RAG). Current research primarily centers on paragraph-level chunking. This approach treats all texts as equal and neglects the information contained in the structure of documents. We propose an expanded approach to chunk documents by moving beyond mere paragraph-level chunking to chunk primary by structural element components of documents. Dissecting documents into these constituent elements creates a new way to chunk documents that yields the best chunk size without tuning. We introduce a novel framework that evaluates how chunking based on element types annotated by document understanding models contributes to the overall context and accuracy of the information retrieved. We also demonstrate how this approach impacts RAG assisted Question & Answer task performance. Our research includes a comprehensive analysis of various element types, their role in effective information retrieval, and the impact they have on the 
    
[^103]: 量化相似性：使用文本挖掘方法评估ChatGPT和Google Bard生成内容与生物医学文献的关联性

    Quantifying Similarity: Text-Mining Approaches to Evaluate ChatGPT and Google Bard Content in Relation to BioMedical Literature

    [https://arxiv.org/abs/2402.05116](https://arxiv.org/abs/2402.05116)

    本研究旨在通过文本挖掘方法评估ChatGPT和Google Bard生成的内容与生物医学文献之间的相似性。实验结果显示，在余弦文档相似性方面，ChatGPT表现优于Google Bard。

    

    背景：在大语言模型（LLMs）的支持下，生成式人工智能工具的出现展示了强大的生成内容能力。到目前为止，评估通过所谓的提示工程生成的内容的有用性已经成为一个有趣的研究问题。目标：通过提示工程的平均值，我们评估这些内容与科学家产生的真实文献的相似性和接近程度。方法：在这个探索性分析中，（1）我们通过提示工程来生成ChatGPT和Google Bard的临床内容，以便与文献对应内容进行比较，（2）我们通过比较所生成内容与生物医学文献对应内容的相似性来评估它们之间的相似性。我们的方法是使用文本挖掘方法比较文档和相关的二元组，并使用网络分析来评估术语的中心性。结果：实验表明，ChatGPT在余弦文档相似性方面表现优于Google Bard（38%对34%），

    Background: The emergence of generative AI tools, empowered by Large Language Models (LLMs), has shown powerful capabilities in generating content. To date, the assessment of the usefulness of such content, generated by what is known as prompt engineering, has become an interesting research question. Objectives Using the mean of prompt engineering, we assess the similarity and closeness of such contents to real literature produced by scientists. Methods In this exploratory analysis, (1) we prompt-engineer ChatGPT and Google Bard to generate clinical content to be compared with literature counterparts, (2) we assess the similarities of the contents generated by comparing them with counterparts from biomedical literature. Our approach is to use text-mining approaches to compare documents and associated bigrams and to use network analysis to assess the terms' centrality. Results The experiments demonstrated that ChatGPT outperformed Google Bard in cosine document similarity (38% to 34%), 
    
[^104]: 大型语言模型作为MOOCs评分者

    Large Language Models As MOOCs Graders

    [https://arxiv.org/abs/2402.03776](https://arxiv.org/abs/2402.03776)

    该研究探索了利用大型语言模型（LLMs）代替MOOCs中同伴评分的可行性，旨在解决大规模在线开放课程中评估学生写作任务的问题。

    

    大规模在线开放课程（MOOCs）为拥有电脑和互联网访问权限的全球任何人提供免费教育的机会。尽管如此，这些课程的大规模注册意味着一位教师几乎不可能评估每个学生的写作任务。因此，同伴评分通常是首选方法，通常由简单明了的评分标准指导。然而，同伴评分在可靠度和有效性方面常常存在问题。在这项研究中，我们利用18个不同的场景，探索利用大型语言模型（LLMs）替代MOOCs中的同伴评分的可行性。具体而言，我们关注两种最先进的LLMs：GPT-4和GPT-3.5，并涵盖三门不同的课程：入门天文学，天体生物学以及天文学的历史与哲学。为了训练LLMs，我们使用了基于零-shot连续思考（Zero-shot-CoT）提示技术的变种的三个不同提示：结合Zero-shot-CoT的提示。

    Massive open online courses (MOOCs) unlock the doors to free education for anyone around the globe with access to a computer and the internet. Despite this democratization of learning, the massive enrollment in these courses means it is almost impossible for one instructor to assess every student's writing assignment. As a result, peer grading, often guided by a straightforward rubric, is the method of choice. While convenient, peer grading often falls short in terms of reliability and validity. In this study, using 18 distinct settings, we explore the feasibility of leveraging large language models (LLMs) to replace peer grading in MOOCs. Specifically, we focus on two state-of-the-art LLMs: GPT-4 and GPT-3.5, across three distinct courses: Introductory Astronomy, Astrobiology, and the History and Philosophy of Astronomy. To instruct LLMs, we use three different prompts based on a variant of the zero-shot chain-of-thought (Zero-shot-CoT) prompting technique: Zero-shot-CoT combined with
    
[^105]: BGE M3-嵌入：通过自知识蒸馏实现多语言、多功能和多粒度的文本嵌入

    BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation

    [https://arxiv.org/abs/2402.03216](https://arxiv.org/abs/2402.03216)

    BGE M3-嵌入是一种新的多语言、多功能和多粒度的文本嵌入模型，支持超过100种工作语言，并在多语言和跨语言检索任务上取得了最先进的性能。它能够同时执行密集检索、多向量检索和稀疏检索，并能处理不同粒度的输入。其有效训练包括了一种自知识蒸馏方法和优化的批处理策略。

    

    在本文中，我们提出了一种新的嵌入模型，称为M3-嵌入，以其在多语言、多功能和多粒度方面的多样性而著称。它可以支持超过100种工作语言，在多语言和跨语言检索任务上取得了新的最先进性能。它可以同时执行嵌入模型的三种常见检索功能：密集检索、多向量检索和稀疏检索，为现实世界的IR应用提供了统一的模型基础。它能够处理不同粒度的输入，从短句到长达8192个标记的文档。M3-嵌入的有效训练包括以下技术贡献。我们提出了一种新颖的自知识蒸馏方法，可以将来自不同检索功能的相关性分数整合为教师信号，以提高训练质量。我们还优化了批处理策略。

    In this paper, we present a new embedding model, called M3-Embedding, which is distinguished for its versatility in Multi-Linguality, Multi-Functionality, and Multi-Granularity. It can support more than 100 working languages, leading to new state-of-the-art performances on multi-lingual and cross-lingual retrieval tasks. It can simultaneously perform the three common retrieval functionalities of embedding model: dense retrieval, multi-vector retrieval, and sparse retrieval, which provides a unified model foundation for real-world IR applications. It is able to process inputs of different granularities, spanning from short sentences to long documents of up to 8192 tokens. The effective training of M3-Embedding involves the following technical contributions. We propose a novel self-knowledge distillation approach, where the relevance scores from different retrieval functionalities can be integrated as the teacher signal to enhance the training quality. We also optimize the batching strat
    
[^106]: 逃避语言模型数据污染检测（太）容易

    Evading Data Contamination Detection for Language Models is (too) Easy

    [https://arxiv.org/abs/2402.02823](https://arxiv.org/abs/2402.02823)

    本研究指出语言模型数据污染的检测方法在面对恶意模型提供者的有意污染时存在漏洞，并提出了一种简单而有效的污染技术（EAL）来显著提高基准测试性能且逃避当前的检测方法。

    

    大型语言模型广泛使用，它们在基准测试上的性能经常指导用户对一个模型与另一个模型的偏好。然而，这些模型所训练的大量数据可能会意外地与公共基准测试数据发生污染，从而损害性能评估。尽管最近开发了一些污染检测方法来解决这个问题，但它们忽视了恶意模型提供者有意进行污染以避免被检测的可能性。我们认为这种情况非常重要，因为它对公共基准测试的可信度产生了怀疑。为了更严格地研究这个问题，我们提出了模型提供者和污染检测方法的分类，这揭示了现有方法中的漏洞，我们通过使用EAL这种简单而有效的污染技术，明显提高了基准测试的性能，并完全逃避了当前的检测方法。

    Large language models are widespread, with their performance on benchmarks frequently guiding user preferences for one model over another. However, the vast amount of data these models are trained on can inadvertently lead to contamination with public benchmarks, thus compromising performance measurements. While recently developed contamination detection methods try to address this issue, they overlook the possibility of deliberate contamination by malicious model providers aiming to evade detection. We argue that this setting is of crucial importance as it casts doubt on the reliability of public benchmarks. To more rigorously study this issue, we propose a categorization of both model providers and contamination detection methods. This reveals vulnerabilities in existing methods that we exploit with EAL, a simple yet effective contamination technique that significantly inflates benchmark performance while completely evading current detection methods.
    
[^107]: VIALM：关于具有大型模型的视觉障碍辅助的调查和基准研究

    VIALM: A Survey and Benchmark of Visually Impaired Assistance with Large Models

    [https://arxiv.org/abs/2402.01735](https://arxiv.org/abs/2402.01735)

    这项研究调查了具有大型模型的视觉障碍辅助，并通过基准实验评估了模型的能力，进一步推动了视觉障碍辅助技术的发展。

    

    视觉障碍辅助 (VIA) 旨在自动帮助视觉障碍者 (VI) 处理日常活动。VIA 的进展主要依赖于计算机视觉 (CV) 和自然语言处理 (NLP) 的发展，二者都展示了利用大型模型 (LMs) 的前沿范式。此外，LMs 展现出出色的多模态能力，可以应对诸如具身机器人等具有挑战性的物理任务。为了研究最先进 (SOTA) LMs 在VIA应用中的潜力和局限性，我们针对具有LMs的VIA任务（VIALM）进行了广泛的研究。在这个任务中，给定一个说明物理环境的图像和视觉障碍者用户的语言请求，VIALM旨在输出逐步引导，以在环境中帮助视觉障碍用户完成请求。该研究包括对近期LM研究的调查和对选定LMs能力的基准实验的检查。

    Visually Impaired Assistance (VIA) aims to automatically help visually impaired (VI) handle daily activities. The advancement of VIA primarily depends on developments in Computer Vision (CV) and Natural Language Processing (NLP), both of which exhibit cutting-edge paradigms with large models (LMs). Furthermore, LMs have shown exceptional multimodal abilities to tackle challenging physically-grounded tasks such as embodied robots. To investigate the potential and limitations of state-of-the-art (SOTA) LMs' capabilities in VIA applications, we present an extensive study for the task of VIA with LMs (\textbf{VIALM}). In this task, given an \textit{image} illustrating the physical environments and a \textit{linguistic request} from a VI user, VIALM aims to output step-by-step \textit{guidance} to assist the VI user in fulfilling the request grounded in the environment. The study consists of a survey reviewing recent LM research and benchmark experiments examining selected LMs' capabilities
    
[^108]: 使用结构化纵向电子健康记录数据促使大型语言模型进行零样本临床预测

    Prompting Large Language Models for Zero-Shot Clinical Prediction with Structured Longitudinal Electronic Health Record Data

    [https://arxiv.org/abs/2402.01713](https://arxiv.org/abs/2402.01713)

    本研究探索了将大型语言模型（LLMs）应用于结构化纵向电子健康记录（EHR）数据的可行性，并着重研究了其零样本能力。通过考虑EHR特征和临床上下文，我们的方法在MIMIC-IV和TJH数据集上取得了良好的实验结果。

    

    结构化纵向电子健康记录（EHR）数据的固有复杂性使其与传统上为自然语言处理而设计的大型语言模型（LLM）整合时面临重大挑战。受新疾病爆发时迅速决策的紧迫需求的驱使，本研究调查了类似GPT-4的LLM对EHR数据的适应性。我们特别关注它们的零样本能力，即在没有明确训练的情况下进行预测。针对EHR数据的纵向、稀疏和知识注入的特点，我们的提示方法考虑了特定的EHR特征，如单位和参考范围，并采用了与临床上下文相一致的上下文学习策略。通过在MIMIC-IV和TJH数据集上进行全面实验，我们证明了LLM能够通过我们的方法进行零样本临床预测，有效应对了EHR数据的挑战。

    The inherent complexity of structured longitudinal Electronic Health Records (EHR) data poses a significant challenge when integrated with Large Language Models (LLMs), which are traditionally tailored for natural language processing. Motivated by the urgent need for swift decision-making during new disease outbreaks, where traditional predictive models often fail due to a lack of historical data, this research investigates the adaptability of LLMs, like GPT-4, to EHR data. We particularly focus on their zero-shot capabilities, which enable them to make predictions in scenarios in which they haven't been explicitly trained. In response to the longitudinal, sparse, and knowledge-infused nature of EHR data, our prompting approach involves taking into account specific EHR characteristics such as units and reference ranges, and employing an in-context learning strategy that aligns with clinical contexts. Our comprehensive experiments on the MIMIC-IV and TJH datasets demonstrate that with o
    
[^109]: 健康-LLM：个性化检索增强的疾病预测模型

    Health-LLM: Personalized Retrieval-Augmented Disease Prediction Model

    [https://arxiv.org/abs/2402.00746](https://arxiv.org/abs/2402.00746)

    提出了一个创新的框架，健康-LLM，通过大规模特征提取和医学知识权衡评分，实现了个性化的检索增强疾病预测模型。这种方法通过整合健康报告，调整特征权重，以及利用语言模型和专家见解提高预测准确性，与传统健康管理方法相比具有明显优势。

    

    在卫生保健领域，人工智能（AI）极大地推进了智能医疗技术的发展。然而，传统智能医疗受限于静态数据和统一标准，无法完全与个体情况集成，同时也面临其他挑战。为此，我们提出了一种创新的框架，命名为健康-LLM，将大规模特征提取和医学知识权衡评分相结合。与传统健康管理方法相比，我们的方法具有三个主要优势。首先，我们的方法将健康报告整合到大模型中，提供详细的任务信息。其次，我们使用专业的医学专业知识调整健康特征的权重得分。第三，我们使用半自动特征提取框架增强语言模型的分析能力，并整合专家见解以提高疾病预测的准确性。

    Artificial intelligence (AI) in healthcare has significantly advanced intelligent medical treatment. However, traditional intelligent healthcare is limited by static data and unified standards, preventing full integration with individual situations and other challenges. Hence, a more professional and detailed intelligent healthcare method is needed for development. To this end, we propose an innovative framework named Heath-LLM, which combines large-scale feature extraction and medical knowledge trade-off scoring. Compared to traditional health management methods, our approach has three main advantages. First, our method integrates health reports into a large model to provide detailed task information. Second, professional medical expertise is used to adjust the weighted scores of health characteristics. Third, we use a semi-automated feature extraction framework to enhance the analytical power of language models and incorporate expert insights to improve the accuracy of disease predic
    
[^110]: ProLex: 一种以语言熟练度为导向的词汇替换评估基准

    ProLex: A Benchmark for Language Proficiency-oriented Lexical Substitution

    [https://arxiv.org/abs/2401.11356](https://arxiv.org/abs/2401.11356)

    ProLex是一个以语言熟练度为导向的词汇替换的评估基准，旨在评估生成适当替代词和表现更好语言熟练度的系统能力。使用微调任务特定合成数据的Llama2-13B模型在F分数上优于ChatGPT 3.2%，与GPT-4在ProLex上表现相当。

    

    词汇替换是在上下文句子中为给定的目标词找到合适的替代词。然而，这个任务没有考虑到与目标词同等或更高熟练度的替代词，这对于希望提高写作水平的语言学习者来说可能是有益的。为了弥补这个差距，我们提出了一项新任务，即以语言熟练度为导向的词汇替换。我们还引入了ProLex，一个新颖的基准，旨在评估系统生成不仅合适的替代词还要表现出更好语言熟练度的能力。除了基准，我们提出了可以自动执行这个新任务的模型。我们证明了我们最好的模型，即使用任务特定合成数据微调的Llama2-13B模型，在F分数上平均优于ChatGPT 3.2％，并在ProLex上与GPT-4取得可比较的结果。

    Lexical Substitution discovers appropriate substitutes for a given target word in a context sentence. However, the task fails to consider substitutes that are of equal or higher proficiency than the target, an aspect that could be beneficial for language learners looking to improve their writing. To bridge this gap, we propose a new task, language proficiency-oriented lexical substitution. We also introduce ProLex, a novel benchmark designed to assess systems' ability to generate not only appropriate substitutes but also substitutes that demonstrate better language proficiency. Besides the benchmark, we propose models that can automatically perform the new task. We show that our best model, a Llama2-13B model fine-tuned with task-specific synthetic data, outperforms ChatGPT by an average of 3.2% in F-score and achieves comparable results with GPT-4 on ProLex.
    
[^111]: 在标记零售银行交易中，使用零样本提示来自动创建和扩展主题分类法的研究

    Using Zero-shot Prompting in the Automatic Creation and Expansion of Topic Taxonomies for Tagging Retail Banking Transactions

    [https://arxiv.org/abs/2401.06790](https://arxiv.org/abs/2401.06790)

    这项工作提出了使用无监督方法和零样本提示来自动构建和扩展主题分类法的研究。通过应用主题建模和关键词提取技术，结合基于指令的微调LLMs，在零售银行数据集中为商家分配标签，具有超过90%的一致性率和令人兴奋的结果。

    

    本文提出了一种无监督的方法，利用基于指令的微调LLMs（大型语言模型）自动构建和扩展主题分类法。我们应用主题建模和关键词提取技术创建初始主题分类法，利用LLMs对结果术语进行后处理并创建层次结构。为了使用新术语扩展现有分类法，我们使用零样本提示来确定在何处添加新节点，据我们所知，这是首次将这种方法应用于分类法任务。我们使用得到的分类法为零售银行数据集中的商家分配标签。为了评估我们的工作，我们请12名志愿者回答了一个两部分的表格，我们首先评估了所创建分类法的质量，然后评估了基于该分类法分配给商家的标签。评估结果显示所选分类法的一致性率超过90%。使用LLMs扩展分类法也显示出令人兴奋的结果，父节点的优先级降低。

    This work presents an unsupervised method for automatically constructing and expanding topic taxonomies using instruction-based fine-tuned LLMs (Large Language Models). We apply topic modeling and keyword extraction techniques to create initial topic taxonomies and LLMs to post-process the resulting terms and create a hierarchy. To expand an existing taxonomy with new terms, we use zero-shot prompting to find out where to add new nodes, which, to our knowledge, is the first work to present such an approach to taxonomy tasks. We use the resulting taxonomies to assign tags that characterize merchants from a retail bank dataset. To evaluate our work, we asked 12 volunteers to answer a two-part form in which we first assessed the quality of the taxonomies created and then the tags assigned to merchants based on that taxonomy. The evaluation revealed a coherence rate exceeding 90% for the chosen taxonomies. The taxonomies' expansion with LLMs also showed exciting results for parent node pre
    
[^112]: 强化关注力中最短的支柱：增强大型语言模型的上下文意识，以实现有效的工具使用

    Fortify the Shortest Stave in Attention: Enhancing Context Awareness of Large Language Models for Effective Tool Use

    [https://arxiv.org/abs/2312.04455](https://arxiv.org/abs/2312.04455)

    本文证明了大型语言模型中关注分配的波形模式对其在需要高度上下文意识的任务中的性能有显著影响。我们提出了一种名为“Attention Buckets”的推理方法，通过多个并行过程和不同的旋转位置嵌入角度，增强了模型对不同上下文位置的意识，从而减轻了忽视关键信息的风险。

    

    在本文中，我们证明了大型语言模型(LLMs)中关注分配中的内在波形模式显著影响它们在需要高度上下文意识的任务中的性能，例如利用LLMs进行工具使用。具体而言，当关键信息在上下文中位于关注波形的低谷区域时，模型可能会忽视该信息，导致性能下降。为了解决这个问题，我们提出了一种名为“Attention Buckets”的新型推理方法。它允许LLMs通过多个并行过程处理输入。每个过程使用不同的基准角度进行旋转位置嵌入，从而创建出一个独特的关注波形。通过用一个过程的关注低谷补偿另一个过程的关注高峰，我们的方法增强了LLM对不同上下文位置的意识，从而减轻了忽视关键信息的风险。

    In this paper, we demonstrate that an inherent waveform pattern in the attention allocation of large language models (LLMs) significantly affects their performance in tasks demanding a high degree of context awareness, such as utilizing LLMs for tool-use. Specifically, the crucial information in the context will be potentially overlooked by model when it is positioned in the trough zone of the attention waveform, leading to decreased performance. To address this issue, we propose a novel inference method named Attention Buckets. It allows LLMs to process their input through multiple parallel processes. Each process utilizes a distinct base angle for the rotary position embedding, thereby creating a unique attention waveform. By compensating an attention trough of a particular process with an attention peak of another process, our approach enhances LLM's awareness to various contextual positions, thus mitigating the risk of overlooking crucial information. In the largest tool-use benchm
    
[^113]: 大型语言模型通过语言特征对齐可以增强说服力

    Large language models can enhance persuasion through linguistic feature alignment

    [https://arxiv.org/abs/2311.16466](https://arxiv.org/abs/2311.16466)

    本研究调查了大型语言模型对人类沟通的影响，使用了消费者金融投诉数据，并发现大型语言模型的使用可能增强了一整套语言特征，提高了信息说服力。

    

    尽管大型语言模型 (LLMs)正在重新塑造人类生活的各个方面，但我们对它们的影响的理解仍然有些受限。本文研究了LLMs对人类沟通的影响，使用了消费者金融投诉的数据。通过对消费者金融保护局 (CFPB) 收集的超过820,000个投诉进行AI检测，我们发现在ChatGPT发布后不久，LLMs的使用可能性急剧增加。此外，LLMs的使用可能性与信息说服力（即从金融公司获得救济的可能性增加）呈正相关。计算语言分析表明，这种正相关可能是由LLMs增强了各种语言特征所解释的。根据这些观察研究的结果，我们假设LLMs的使用可能增强了一整套语言特征，提高了对具有不同语言背景的接收者的信息说服力。

    Although large language models (LLMs) are reshaping various aspects of human life, our current understanding of their impacts remains somewhat constrained. Here we investigate the impact of LLMs on human communication, using data on consumer complaints in the financial industry. By employing an AI detection tool on more than 820K complaints gathered by the Consumer Financial Protection Bureau (CFPB), we find a sharp increase in the likely use of LLMs shortly after the release of ChatGPT. Moreover, the likely LLM usage was positively correlated with message persuasiveness (i.e., increased likelihood of obtaining relief from financial firms). Computational linguistic analyses suggest that the positive correlation may be explained by LLMs' enhancement of various linguistic features. Based on the results of these observational studies, we hypothesize that LLM usage may enhance a comprehensive set of linguistic features, increasing message persuasiveness to receivers with heterogeneous ling
    
[^114]: 使用神经网络算法检测塞浦路斯希腊儿童的发展性语言障碍

    Detection of developmental language disorder in Cypriot Greek children using a neural network algorithm

    [https://arxiv.org/abs/2311.15054](https://arxiv.org/abs/2311.15054)

    该研究开发了一种使用神经网络算法进行发展性语言障碍（DLD）检测的自动化方法，并首次应用于塞浦路斯希腊儿童DLD人群。实验结果表明该方法具有高的分类效果。

    

    发展性语言障碍（DLD）的儿童在吸收各种语言结构方面遇到困难。早期识别和干预对于防止对儿童的学术、社交和情感发展产生长期负面影响至关重要。该研究旨在开发一种使用人工智能、特别是神经网络机器学习算法的自动化检测DLD的方法。该方案首次在塞浦路斯希腊儿童DLD人群中应用。神经网络模型使用从15名DLD患儿和15名健康对照组中收集的感知和产出数据进行训练，年龄范围为7岁10个月至10岁4个月。采用k-fold技术对算法进行交叉验证。使用准确率、精确度、召回率、F1分数和ROC/AUC曲线等指标评估模型的性能，以评估其在一组未知数据上进行准确预测的能力。结果表明高分类效果。

    Children with developmental language disorder (DLD) encounter difficulties in acquiring various language structures. Early identification and intervention are crucial to prevent negative long-term outcomes impacting the academic, social, and emotional development of children. The study aims to develop an automated method for the identification of DLD using artificial intelligence, specifically a neural network machine learning algorithm. This protocol is applied for the first time in a Cypriot Greek child population with DLD. The neural network model was trained using perceptual and production data elicited from 15 children with DLD and 15 healthy controls in the age range of 7;10 until 10;4. The k-fold technique was used to crossvalidate the algorithm. The performance of the model was evaluated using metrics such as accuracy, precision, recall, F1 score, and ROC/AUC curve to assess its ability to make accurate predictions on a set of unseen data. The results demonstrated high classifi
    
[^115]: 从被毒害的人类反馈中构建的通用越狱后门

    Universal Jailbreak Backdoors from Poisoned Human Feedback

    [https://arxiv.org/abs/2311.14455](https://arxiv.org/abs/2311.14455)

    本文提出了一种新的威胁，攻击者通过毒害训练数据向语言模型中嵌入“越狱后门”，并通过添加特定的触发词使模型产生有害的回应。这种通用越狱后门比之前的研究更强大，且较难被察觉。研究探究了RLHF设计中的决策对其鲁棒性的影响，并发布了一组被毒害模型的基准测试数据，以促进未来对通用越狱后门的研究。

    

    强化学习从人类反馈中（RLHF）用于对齐大型语言模型，以产生有用且无害的回应。然而，先前的研究显示，这些模型可以通过找到使模型恢复到未对齐行为的对抗提示来进行越狱。在本文中，我们考虑了一个新的威胁，攻击者通过毒害RLHF训练数据将“越狱后门”嵌入模型中。该后门将一个触发词嵌入模型中，类似于通用的“sudo命令”：在任何提示中添加触发词将使模型产生有害的回应，无需搜索对抗提示。通用越狱后门比先前研究的语言模型后门更强大，我们发现使用常见的后门攻击技术要困难得多。我们探究了RLHF中的设计决策对其所声称的鲁棒性的贡献，并发布一组被毒害模型的基准，以促进对通用越狱后门的未来研究。

    Reinforcement Learning from Human Feedback (RLHF) is used to align large language models to produce helpful and harmless responses. Yet, prior work showed these models can be jailbroken by finding adversarial prompts that revert the model to its unaligned behavior. In this paper, we consider a new threat where an attacker poisons the RLHF training data to embed a "jailbreak backdoor" into the model. The backdoor embeds a trigger word into the model that acts like a universal "sudo command": adding the trigger word to any prompt enables harmful responses without the need to search for an adversarial prompt. Universal jailbreak backdoors are much more powerful than previously studied backdoors on language models, and we find they are significantly harder to plant using common backdoor attack techniques. We investigate the design decisions in RLHF that contribute to its purported robustness, and release a benchmark of poisoned models to stimulate future research on universal jailbreak bac
    
[^116]: PLUG: 跨语言指令调优中的枢纽语言优势利用

    PLUG: Leveraging Pivot Language in Cross-Lingual Instruction Tuning

    [https://arxiv.org/abs/2311.08711](https://arxiv.org/abs/2311.08711)

    提出了一种利用枢纽语言进行指令调优的方法，该方法在低资源语言中取得了显著改进，并通过引入一个基准数据集进行了评估。

    

    指令调优在大型语言模型（LLMs）在理解和回应各种人类指令方面取得了显著进展。尽管在高资源语言中取得了成功，但其在低资源语言中的应用面临挑战，原因是LLMs在不同语言上的基础能力不平衡，这源于在它们的预训练数据中语言分布的不均衡。为了解决这个问题，我们提出了枢纽语言引导生成（PLUG）的方法，该方法利用高资源语言（主要是英语）作为枢纽语言，增强了低资源语言中的指令调优。它训练模型首先处理枢纽语言中的指令，然后在目标语言中生成响应。为了评估我们的方法，我们引入了一个基准数据集X-AlpacaEval，其中包含了4种语言（中文、韩文、意大利文和西班牙文）的指令，并由专业翻译人员进行了注释。我们的方法在指令跟随中显示了显著的改进。

    Instruction tuning has remarkably advanced large language models (LLMs) in understanding and responding to diverse human instructions. Despite the success in high-resource languages, its application in lower-resource ones faces challenges due to the imbalanced foundational abilities of LLMs across different languages, stemming from the uneven language distribution in their pre-training data. To tackle this issue, we propose pivot language guided generation (PLUG), an approach that utilizes a high-resource language, primarily English, as the pivot to enhance instruction tuning in lower-resource languages. It trains the model to first process instructions in the pivot language, and then produce responses in the target language. To evaluate our approach, we introduce a benchmark, X-AlpacaEval, of instructions in 4 languages (Chinese, Korean, Italian, and Spanish), each annotated by professional translators. Our approach demonstrates a significant improvement in the instruction-following a
    
[^117]: 关于衡量自然语言解释的忠诚度或自一致性

    On Measuring Faithfulness or Self-consistency of Natural Language Explanations

    [https://arxiv.org/abs/2311.07466](https://arxiv.org/abs/2311.07466)

    本文论述了衡量自然语言解释的忠诚度或自一致性的问题。我们提出了自一致性测试来评估解释的输出级别的一致性。我们通过构建比较一致性测试库，并引入了新的自一致性度量CC-SHAP来支持我们的观点。

    

    大型语言模型（LLMs）可以通过事后或思维链（CoT）解释其预测。但是，LLM可能会编造听起来合理但不忠实于其基本推理的解释。最近的工作设计了旨在判断事后或CoT解释忠实度的测试。在这项工作中，我们认为这些忠实度测试不是衡量模型内部工作的忠实度，而是衡量其输出级别的自一致性。我们的贡献有三个方面：i）我们在模型可解释性的背景下澄清了忠实度测试的地位，将其描述为自一致性测试。我们通过ii）构建了一个比较一致性的测试库，首次在11个开放式LLMs和5个任务的通用套件上比较了现有测试，包括iii）我们的新的自一致性度量CC-SHAP。CC-SHAP是LLM自一致性的细粒度度量（而不是测试）。它进行比较。

    Large language models (LLMs) can explain their predictions through post-hoc or Chain-of-Thought (CoT) explanations. But an LLM could make up reasonably sounding explanations that are unfaithful to its underlying reasoning. Recent work has designed tests that aim to judge the faithfulness of post-hoc or CoT explanations. In this work we argue that these faithfulness tests do not measure faithfulness to the models' inner workings -- but rather their self-consistency at output level. Our contributions are three-fold: i) We clarify the status of faithfulness tests in view of model explainability, characterising them as self-consistency tests instead. This assessment we underline by ii) constructing a Comparative Consistency Bank for self-consistency tests that for the first time compares existing tests on a common suite of 11 open LLMs and 5 tasks -- including iii) our new self-consistency measure CC-SHAP. CC-SHAP is a fine-grained measure (not a test) of LLM self-consistency. It compares 
    
[^118]: 数据污染问题: 一种检测和估计大型语言模型中污染的工具

    Data Contamination Quiz: A Tool to Detect and Estimate Contamination in Large Language Models

    [https://arxiv.org/abs/2311.06233](https://arxiv.org/abs/2311.06233)

    这个工具使用数据污染问题（DCQ）的方法来检测和估计大型语言模型中的数据污染。在DCQ中，我们创建了每个数据集实例的扰动版本，并让语言模型从中选择原始实例，通过词级扰动来区分选项。这种方法利用了语言模型在预训练阶段暴露于原始实例时的固有特性。

    

    我们提出了数据污染问题（DCQ），这是一种简单而有效的方法，用于检测大型语言模型（LLM）中的数据污染并估计其数量。具体而言，我们将数据污染检测视为一系列的多项选择问题，并设计了一种测验形式，其中创建了每个数据集实例的三个扰动版本。这些变化仅包括词级扰动。生成的扰动版本与原始实例一起形成DCQ中的选项，额外的选项适应了提供的选择都不正确的可能性。鉴于在选择之间唯一的区别信号是与原始实例的确切措辞相关，如果在预训练阶段已经接触到原始实例，语言模型当被要求从选项中识别原始实例时，倾向于选择原始实例--这是语言模型固有的特性。在使用GPT-4/3.5进行多个数据集的测试中，我们的结果完全缺少准确性。

    We propose the Data Contamination Quiz (DCQ), a simple and effective approach to detect data contamination in large language models (LLMs) and estimate the amount of it. Specifically, we frame data contamination detection as a series of multiple-choice questions and devise a quiz format wherein three perturbed versions of each dataset instance are created. These changes only include word-level perturbations. The generated perturbed versions, along with the original instance, form the options in the DCQ, with an extra option accommodating the possibility that none of the provided choices is correct. Given that the only distinguishing signal among the choices is the exact wording relative to the original instance, an LLM, when tasked with identifying the original instance from the choices, gravitates towards the original one if it has been exposed to it in its pre-training phase--a trait intrinsic to LLMs. Tested over several datasets with GPT-4/3.5, our findings--while fully lacking acc
    
[^119]: PeTailor：通过定制的分块评分器改进生物医学三元组提取的大型语言模型

    PeTailor: Improving Large Language Model by Tailored Chunk Scorer in Biomedical Triple Extraction

    [https://arxiv.org/abs/2310.18463](https://arxiv.org/abs/2310.18463)

    我们提出了PeTailor，这是一个基于检索的框架，通过使用定制的分块评分器从预先构建的分块数据库中检索相关文档，并将检索到的信息集成到大型语言模型（LLM）的输入中，以改进生物医学三元组提取的效果。

    

    生物医学三元组提取系统旨在自动提取生物医学实体和实体之间的关系。虽然当前的统一信息提取模型展示了最先进的性能，但在理解复杂生物医学句子中实体之间的关系方面面临挑战。此外，缺乏高质量的生物医学三元组提取数据集阻碍了稳健的三元组提取系统的开发进展。为了解决这些挑战，我们提出了一种新颖的适用于生物医学三元组提取的基于检索的框架，名为PeTailor，它使用一种新颖的定制分块评分器从我们预先构建的多样分块数据库中显式地检索相关文档，并将检索到的信息集成到大型语言模型（LLM）的输入中，为输入的句子生成相应的三元组（头实体，关系，尾实体）。此外，我们还提供了GM-CIHT，一种专家标注的生物医学三元组提取数据集，该数据集支持了我们的方法的实验评估。

    Biomedical triple extraction systems aim to automatically extract biomedical entities and relations between entities. While current unified information extraction models showcase state-of-the-art performance, they face challenges in understanding relationships between entities within intricate biomedical sentences. Furthermore, the absence of a high-quality biomedical triple extraction dataset impedes the progress in developing robust triple extraction systems. To tackle these challenges, we propose a novel retrieval-based framework for biomedical triple extraction, namely PeTailor, which explicitly retrieves the relevant document from our pre-built diverse chunk database using a novel tailored chunk scorer and integrates the retrieved information into the input of a Large Language Model (LLM) to generate the corresponding triple (head entity, relation, tail entity) for the input sentence. Additionally, we present GM-CIHT, an expert-annotated biomedical triple extraction dataset that c
    
[^120]: SQLformer：深度自回归查询图生成用于文本到SQL翻译

    SQLformer: Deep Auto-Regressive Query Graph Generation for Text-to-SQL Translation

    [https://arxiv.org/abs/2310.18376](https://arxiv.org/abs/2310.18376)

    SQLformer是一个用于文本到SQL翻译的深度自回归查询图生成模型，采用了特定的Transformer架构，并通过结构归纳偏差解决领域泛化和自然语言与SQL查询对齐的难题。

    

    近年来，对于文本到SQL翻译的兴趣日益增长，这是将自然语言问题转化为可执行SQL查询的任务。这项技术具有潜在的潜力，可以使数据库中的数据提取民主化。然而，其中一些主要障碍包括领域泛化，即适应以前未见到的数据库，并且将自然语言问题与相应的SQL查询对齐。为了克服这些挑战，我们引入了SQLformer，这是一种针对执行文本到SQL翻译任务而设计的新型Transformer体系结构。我们的模型以自回归的方式预测SQL查询，并在编码器和解码器层中结合结构归纳偏差。这种偏差是由数据库表和列选择引导的，有助于解码器以广度优先搜索的规范顺序生成SQL查询的图形表示。全面的实验说明了现阶段的技术水平

    In recent years, there has been growing interest in text-to-SQL translation, which is the task of converting natural language questions into executable SQL queries. This technology is important for its potential to democratize data extraction from databases. However, some of its key hurdles include domain generalisation, which is the ability to adapt to previously unseen databases, and alignment of natural language questions with the corresponding SQL queries. To overcome these challenges, we introduce SQLformer, a novel Transformer architecture specifically crafted to perform text-to-SQL translation tasks. Our model predicts SQL queries as abstract syntax trees (ASTs) in an autoregressive way, incorporating structural inductive bias in the encoder and decoder layers. This bias, guided by database table and column selection, aids the decoder in generating SQL query ASTs represented as graphs in a Breadth-First Search canonical order. Comprehensive experiments illustrate the state-of-th
    
[^121]: 带有参数高效prompt调整和自适应优化的大型语言模型的联邦学习

    Federated Learning of Large Language Models with Parameter-Efficient Prompt Tuning and Adaptive Optimization

    [https://arxiv.org/abs/2310.15080](https://arxiv.org/abs/2310.15080)

    本文提出了一种带有参数高效prompt调整和自适应优化的联邦学习方法，以实现大型语言模型的高效和有效训练。

    

    联邦学习是一种有前途的范式，可以实现分散数据的协同模型训练。然而，大型语言模型的训练过程通常涉及更新大量的参数，这限制了联邦学习技术在实际场景中处理大型语言模型的适用性。prompt调整可以显著减少需要更新的参数数量，但它可能导致性能下降或降低训练效率。在联邦学习中直接使用prompt调整通常会导致非平凡的通信成本和性能大幅下降。此外，分散数据通常是非独立和同分布的，并带来客户端漂移问题和因此的低性能。本文提出了一种参数高效的提示调整方法，即FedPepTAO，以实现大型语言模型的高效和有效的联邦学习。首先，提出了一种高效的部分提示调整方法来改善训练性能。

    Federated learning (FL) is a promising paradigm to enable collaborative model training with decentralized data. However, the training process of Large Language Models (LLMs) generally incurs the update of significant parameters, which limits the applicability of FL techniques to tackle the LLMs in real scenarios. Prompt tuning can significantly reduce the number of parameters to update, but it either incurs performance degradation or low training efficiency. The straightforward utilization of prompt tuning in the FL often raises non-trivial communication costs and dramatically degrades performance. In addition, the decentralized data is generally non-Independent and Identically Distributed (non-IID), which brings client drift problems and thus poor performance. This paper proposes a Parameter-efficient prompt Tuning approach with Adaptive Optimization, i.e., FedPepTAO, to enable efficient and effective FL of LLMs. First, an efficient partial prompt tuning approach is proposed to improv
    
[^122]: 从混乱到清晰：声明标准化以增强事实核查

    From Chaos to Clarity: Claim Normalization to Empower Fact-Checking

    [https://arxiv.org/abs/2310.14338](https://arxiv.org/abs/2310.14338)

    本研究提出了声明标准化任务，通过使用CACN模型利用思维链和声明检查来从复杂的社交媒体帖子中提取简化的声明，以加强事实核查。

    

    随着社交媒体的兴起，用户接触到许多误导性的声明。然而，这些帖子中固有的混杂噪声使得辨别需要验证的精确且显著的声明变得很具挑战性。从这些帖子中提取重要的声明是费时且困难的，然而这是一个未被充分探索的问题。在这里，我们旨在填补这个差距。我们引入了一个新颖的任务，称为声明标准化（ClaimNorm），旨在将复杂而嘈杂的社交媒体帖子分解为更直接和易于理解的形式，称为标准化声明。我们提出了CACN，一种开创性的方法，利用思维链和声明值得检查的估计来模拟人类推理过程，以理解复杂的声明。此外，我们利用大型语言模型的上下文学习能力来提供指导并改进声明标准化。为了评估我们所提出的模型的有效性，我们精心编制了一个全面的真实世界数据集。

    With the rise of social media, users are exposed to many misleading claims. However, the pervasive noise inherent in these posts presents a challenge in identifying precise and prominent claims that require verification. Extracting the important claims from such posts is arduous and time-consuming, yet it is an underexplored problem. Here, we aim to bridge this gap. We introduce a novel task, Claim Normalization (aka ClaimNorm), which aims to decompose complex and noisy social media posts into more straightforward and understandable forms, termed normalized claims. We propose CACN, a pioneering approach that leverages chain-of-thought and claim check-worthiness estimation, mimicking human reasoning processes, to comprehend intricate claims. Moreover, we capitalize on the in-context learning capabilities of large language models to provide guidance and to improve claim normalization. To evaluate the effectiveness of our proposed model, we meticulously compile a comprehensive real-world 
    
[^123]: 基于上下文的知识库问答中的模式理解方法

    An In-Context Schema Understanding Method for Knowledge Base Question Answering

    [https://arxiv.org/abs/2310.14174](https://arxiv.org/abs/2310.14174)

    本文提出了一种基于上下文的模式理解方法（ICSU），通过提供模式相关的注释示例实现了大型语言模型（LLM）直接理解模式的能力。实验结果表明...

    

    知识库问答（KBQA）任务旨在基于给定的知识库回答自然语言问题。最近，大型语言模型（LLM）在语言理解方面显示出强大的能力，可以用来解决这一任务。在此过程中，LLM面临的主要挑战是克服知识库模式的庞大性和异质性。现有方法通过最初使用LLM生成没有模式特定细节的逻辑形式草稿来绕过这个挑战。然后，使用额外的模块将模式信息注入到这些草稿中。相反，在本文中，我们提出了一种简单的基于上下文的模式理解（ICSU）方法，利用上下文学习使LLM能够直接理解模式。具体而言，ICSU利用与模式相关的注释示例向LLM提供模式信息。我们研究了基于原始问题、匿名问题和生成的SPARQL查询的三种示例检索策略。实验结果表明...

    The Knowledge Base Question Answering (KBQA) task aims to answer natural language questions based on a given knowledge base. Recently, Large Language Models (LLMs) have shown strong capabilities in language understanding and can be used to solve this task. In doing so, a major challenge for LLMs is to overcome the immensity and heterogeneity of knowledge base schemas.Existing methods bypass this challenge by initially employing LLMs to generate drafts of logic forms without schema-specific details.Then, an extra module is used to inject schema information to these drafts.In contrast, in this paper, we propose a simple In-Context Schema Understanding (ICSU) method that enables LLMs to directly understand schemas by leveraging in-context learning. Specifically, ICSU provides schema information to LLMs using schema-related annotated examples. We investigate three example retrieval strategies based on raw questions, anonymized questions, and generated SPARQL queries. Experimental results s
    
[^124]: 利用上下文线索和角色相关性提升文档级事件论证提取

    Utilizing Contextual Clues and Role Correlations for Enhancing Document-level Event Argument Extraction

    [https://arxiv.org/abs/2310.05116](https://arxiv.org/abs/2310.05116)

    本文提出了CARLG模型，通过利用上下文线索和角色相关性，提升了文档级事件论证提取的性能。

    

    文档级事件论证提取（EAE）是信息提取中至关重要但具有挑战性的子任务之一。现有方法大多关注论证和事件触发器之间的交互，忽视了两个关键点：上下文线索的信息和论证角色之间的语义相关性。本文提出了CARLG模型，包括两个模块：上下文线索聚合（CCA）和基于角色的潜在信息引导（RLIG），通过有效利用上下文线索和角色相关性来提高文档级EAE。CCA模块通过利用来自预训练编码器的上下文注意权重，自适应地捕捉和整合上下文线索。RLIG模块通过角色交互编码捕捉语义相关性，并通过潜在角色表示提供宝贵的信息引导。值得注意的是，我们的CCA和RLIG模块紧凑、可移植且高效，引入的新参数不超过1%，且易于实现。

    Document-level event argument extraction (EAE) is a vital but challenging subtask in information extraction. Most existing approaches focus on the interaction between arguments and event triggers, ignoring two critical points: the information of contextual clues and the semantic correlations among argument roles. In this paper, we propose the CARLG model, which consists of two modules: Contextual Clues Aggregation (CCA) and Role-based Latent Information Guidance (RLIG), effectively leveraging contextual clues and role correlations for improving document-level EAE. The CCA module adaptively captures and integrates contextual clues by utilizing context attention weights from a pre-trained encoder. The RLIG module captures semantic correlations through role-interactive encoding and provides valuable information guidance with latent role representation. Notably, our CCA and RLIG modules are compact, transplantable and efficient, which introduce no more than 1% new parameters and can be eas
    
[^125]: 关于常识推理的知识图谱解释的可信性

    Faithful Knowledge Graph Explanations for Commonsense Reasoning

    [https://arxiv.org/abs/2310.04910](https://arxiv.org/abs/2310.04910)

    本论文提出了两个量化指标来衡量基于知识图谱的解释的可信性，并引入了一种新的训练方法来改善解释的可信度。实验结果表明该方法可以提高解释的一致性和保真度。

    

    融合语言模型(LMs)和知识图谱(KGs)已成为常识问答研究中的常见方法，但在这些模型中实现精确的思路链解释仍然是一个未解决的问题。当前基于知识图谱的解释技术的一个主要弱点是在评估过程中忽视了生成解释的可信性。为了弥补这一差距，我们提出并验证了两个量化指标 - 图一致性和图保真度 - 来衡量基于知识图谱的解释的可信性。我们引入一种新的训练方法Consistent GNN (CGNN)，该方法添加了一项一致性正则化项来改善解释的可信度。我们的分析表明，KG的预测经常偏离原始模型的预测。所提出的CGNN方法提高了一致性和保真度，展示了它产生更可信解释的潜力。我们的工作强调了明确评估解释可信性的重要性。

    While fusing language models (LMs) and knowledge graphs (KGs) has become common in commonsense question answering research, enabling faithful chain-of-thought explanations in these models remains an open problem. One major weakness of current KG-based explanation techniques is that they overlook the faithfulness of generated explanations during evaluation. To address this gap, we make two main contributions: (1) We propose and validate two quantitative metrics - graph consistency and graph fidelity - to measure the faithfulness of KG-based explanations. (2) We introduce Consistent GNN (CGNN), a novel training method that adds a consistency regularization term to improve explanation faithfulness. Our analysis shows that predictions from KG often diverge from original model predictions. The proposed CGNN approach boosts consistency and fidelity, demonstrating its potential for producing more faithful explanations. Our work emphasises the importance of explicitly evaluating suggest a path
    
[^126]: 工具增强的奖励建模

    Tool-Augmented Reward Modeling

    [https://arxiv.org/abs/2310.01045](https://arxiv.org/abs/2310.01045)

    本文提出了一种工具增强的偏好建模方法，通过赋予奖励模型访问外部环境的能力，解决了传统奖励模型在基本功能上的限制，同时提高了解释能力和评分可靠性。

    

    奖励建模（又称偏好建模）对于将大型语言模型与人类偏好相一致是至关重要的，尤其是在从人类反馈中进行强化学习的背景下。传统的奖励模型在可扩展性方面表现出色，但常常在基本功能上遇到困难，如算术计算、代码执行和事实查找。在本文中，我们提出了一种工具增强的偏好建模方法，名为Themis，通过赋予奖励模型访问外部环境的能力，包括计算器和搜索引擎，来解决这些限制。这种方法不仅促进了工具利用和奖励评分之间的协同作用，还增强了解释能力和评分可靠性。我们的研究深入探讨了外部工具与奖励模型的集成，使其能够与多样的外部资源进行交互，并以自回归的方式构建任务特定的工具参与和推理轨迹。我们验证了我们的方法。

    Reward modeling (a.k.a., preference modeling) is instrumental for aligning large language models with human preferences, particularly within the context of reinforcement learning from human feedback (RLHF). While conventional reward models (RMs) have exhibited remarkable scalability, they oft struggle with fundamental functionality such as arithmetic computation, code execution, and factual lookup. In this paper, we propose a tool-augmented preference modeling approach, named Themis, to address these limitations by empowering RMs with access to external environments, including calculators and search engines. This approach not only fosters synergy between tool utilization and reward grading but also enhances interpretive capacity and scoring reliability. Our study delves into the integration of external tools into RMs, enabling them to interact with diverse external sources and construct task-specific tool engagement and reasoning traces in an autoregressive manner. We validate our appr
    
[^127]: 跨语言收据信息提取和分类的全面多语言数据集 CrossLingR

    CrossLingR: A Comprehensive Multilingual Receipt Dataset for Cross-Language Information Extraction and Classification

    [https://arxiv.org/abs/2309.09800](https://arxiv.org/abs/2309.09800)

    本研究介绍了一个全面的多语言数据集CrossLingR，用于推动收据信息提取和物品分类的进展。我们的数据集包含了47,720个标注样本，详细记录了项目名称、相关属性和44个不同的产品类别。通过InstructLLaMA方法论，我们展示了在关键信息提取和物品分类任务中的显著效果。相关资源可在https://github.com/Update-For-Integrated-Business-AI/AMuRD上获取。

    

    关键信息提取的过程对于将扫描的收据转化为结构化、可访问的文件至关重要，有助于有效地检索重要数据。本研究引入了一个广泛的、新颖的多语言数据集，旨在推动收据信息提取和物品分类领域的进展。我们的数据集包含了47,720个带有项目名称、相关属性（如价格和品牌）的标注样本，并按照44个不同的产品类别进行组织。我们揭示了InstructLLaMA方法论，这是一种开创性的方法，通过关键信息提取和物品分类任务中的F1分数为0.76和准确性为0.68的显著效果加以证明。为了支持进一步的研究和应用开发，我们在https://github.com/Update-For-Integrated-Business-AI/AMuRD上提供了我们的全面数据集、InstructLLaMA模型和相关资源。

    The process of key information extraction is critical for converting scanned receipts into structured, accessible documents, facilitating the efficient retrieval of vital data. This research introduces an expansive, novel multilingual dataset designed to propel advancements in the domain of receipt information extraction and item classification. Our dataset encompasses 47,720 annotated samples, detailed with item names, associated attributes such as price and brand, and organized into 44 distinct product categories. We unveil the InstructLLaMA methodology, a pioneering approach that demonstrates significant effectiveness, evidenced by an F1 score of 0.76 and an accuracy of 0.68 in tasks of key information extraction and item classification. To support further research and application development, we make available our comprehensive dataset, the InstructLLaMA model, and relevant resources at https://github.com/Update-For-Integrated-Business-AI/AMuRD.
    
[^128]: 适用于每个用户和预算的模型：无标签和个性化混合精度量化

    A Model for Every User and Budget: Label-Free and Personalized Mixed-Precision Quantization

    [https://arxiv.org/abs/2307.12659](https://arxiv.org/abs/2307.12659)

    该论文提出了一种无标签和个性化混合精度量化方法，可以根据用户需求和内存预算生成个性化的量化方案，针对大规模ASR模型，能够提高特定性别、语言和说话者的性能。

    

    最近在自动语音识别（ASR）方面取得了进展，产生了大型AI模型，在移动设备上部署变得不切实际。模型量化是一种有效的方法，可以产生压缩的通用模型，然而，这样的模型可能只能在有限制的感兴趣子领域中部署。我们展示了在量化过程中如何依赖于目标领域的少量未标记样本来个性化ASR模型。为此，我们提出了myQASR，一种混合精度量化方法，它可以根据任何内存需求生成针对不同用户的量化方案，无需微调。myQASR通过分析全精度激活值来自动评估网络层次的量化敏感性，从而能够为任何预定的内存预算生成个性化的混合精度量化方案。大规模ASR模型的结果显示了myQASR如何提高特定性别、语言和说话者的性能。

    Recent advancement in Automatic Speech Recognition (ASR) has produced large AI models, which become impractical for deployment in mobile devices. Model quantization is effective to produce compressed general-purpose models, however such models may only be deployed to a restricted sub-domain of interest. We show that ASR models can be personalized during quantization while relying on just a small set of unlabelled samples from the target domain. To this end, we propose myQASR, a mixed-precision quantization method that generates tailored quantization schemes for diverse users under any memory requirement with no fine-tuning. myQASR automatically evaluates the quantization sensitivity of network layers by analysing the full-precision activation values. We are then able to generate a personalised mixed-precision quantization scheme for any pre-determined memory budget. Results for large-scale ASR models show how myQASR improves performance for specific genders, languages, and speakers.
    
[^129]: KDSTM: 使用知识蒸馏的神经半监督主题建模

    KDSTM: Neural Semi-supervised Topic Modeling with Knowledge Distillation

    [https://arxiv.org/abs/2307.01878](https://arxiv.org/abs/2307.01878)

    KDSTM是一种使用知识蒸馏的神经半监督主题建模方法，对于文本分类任务，在没有预训练嵌入且资源受限的情况下，能够提供高准确性、鲁棒性和效率。

    

    在文本分类任务中，微调预训练的语言模型（如BERT和GPT-3）可以获得竞争性的准确性；然而，这两种方法都需要在大型文本数据集上进行预训练。相比之下，一般的主题建模方法具有在不需要预训练的情况下分析文档并提取有意义的词汇模式的优势。为了利用主题建模在文本分类任务中的无监督的见解提取，我们开发了一种称为知识蒸馏半监督主题建模（KDSTM）的方法。KDSTM不需要预训练嵌入，只需要少量的标记文档，并且训练效率高，在资源受限的情况下非常理想。在多个数据集上，我们的方法在分类准确性、鲁棒性和效率方面都超过了现有的有监督主题建模方法，并且与最先进的弱监督文本分类方法相比达到了类似的性能。

    In text classification tasks, fine tuning pretrained language models like BERT and GPT-3 yields competitive accuracy; however, both methods require pretraining on large text datasets. In contrast, general topic modeling methods possess the advantage of analyzing documents to extract meaningful patterns of words without the need of pretraining. To leverage topic modeling's unsupervised insights extraction on text classification tasks, we develop the Knowledge Distillation Semi-supervised Topic Modeling (KDSTM). KDSTM requires no pretrained embeddings, few labeled documents and is efficient to train, making it ideal under resource constrained settings. Across a variety of datasets, our method outperforms existing supervised topic modeling methods in classification accuracy, robustness and efficiency and achieves similar performance compare to state of the art weakly supervised text classification methods.
    
[^130]: 基于数据驱动的癌细胞系分子谱信息提取和丰富化

    Data-Driven Information Extraction and Enrichment of Molecular Profiling Data for Cancer Cell Lines

    [https://arxiv.org/abs/2307.00933](https://arxiv.org/abs/2307.00933)

    这项研究设计了一个新颖的数据提取和探索系统，通过从科学文献中提取深层语义关系，丰富癌细胞系领域的结构化临床数据。

    

    随着研究手段和计算方法的不断增加，生物医学文献以指数级增长。癌细胞系是生物医学研究中经常使用的模型，用于从细胞机制到药物开发等各种目的，导致了相关数据和出版物的丰富性。通过人工筛选大量的文本来收集感兴趣的细胞系的相关信息是费时且极为缓慢的。因此，需要新的计算机信息提取和关联机制来提高有意义的知识提取。在这项工作中，我们提出了一个新颖的数据提取和探索系统的设计、实现和应用。该系统通过从科学文献中提取文本实体之间的深层语义关系，以丰富癌细胞系领域现有的结构化临床数据。

    With the proliferation of research means and computational methodologies, published biomedical literature is growing exponentially in numbers and volume. Cancer cell lines are frequently used models in biological and medical research that are currently applied for a wide range of purposes, from studies of cellular mechanisms to drug development, which has led to a wealth of related data and publications. Sifting through large quantities of text to gather relevant information on the cell lines of interest is tedious and extremely slow when performed by humans. Hence, novel computational information extraction and correlation mechanisms are required to boost meaningful knowledge extraction. In this work, we present the design, implementation and application of a novel data extraction and exploration system. This system extracts deep semantic relations between textual entities from scientific literature to enrich existing structured clinical data in the domain of cancer cell lines. We int
    
[^131]: 通过执行反馈使语言模型成为更好的工具学习者

    Making Language Models Better Tool Learners with Execution Feedback

    [https://arxiv.org/abs/2305.13068](https://arxiv.org/abs/2305.13068)

    这篇论文提出了一个名为TRICE的框架，通过执行反馈实现语言模型的工具学习，使其能够学会何时以及如何有效地使用工具。

    

    工具作为关键的界面，使人类能够理解和改变环境。随着基础模型的出现，AI系统可以利用工具扩展其能力并与真实世界互动。现有的工具学习方法包括监督微调和提示工程方法，通常使大型语言模型不加选择地利用工具，因为复杂任务往往超出了它们自身的能力。然而，为简单任务引入工具（模型本身可以轻松解决的任务），可能会无意间传播错误而不是提高性能。因此，研究问题是：我们能否教会语言模型何时以及如何使用工具？为满足这个需求，我们提出了Tool leaRning wIth exeCution fEedback (TRICE)，这是一个两阶段的端到端框架，使模型能够通过从工具执行中得到的反馈不断学习，从而学会何时以及如何有效地使用工具。

    Tools serve as pivotal interfaces that enable humans to understand and reshape the environment. With the advent of foundation models, AI systems can utilize tools to expand their capabilities and interact with the real world. Existing tool learning methodologies, encompassing supervised fine-tuning and prompt engineering approaches, often induce large language models to utilize tools indiscriminately, as complex tasks often exceed their own competencies. However, introducing tools for simple tasks, which the models themselves can readily resolve, can inadvertently propagate errors rather than enhance performance. This leads to the research question: can we teach language models when and how to use tools? To meet this need, we propose Tool leaRning wIth exeCution fEedback (TRICE), a two-stage end-to-end framework that enables the model to continually learn through feedback derived from tool execution, thereby learning when and how to use tools effectively. Experimental results, backed b
    
[^132]: 使用预训练的嵌入和句子包的高效灵活的主题建模

    Efficient and Flexible Topic Modeling using Pretrained Embeddings and Bag of Sentences

    [https://arxiv.org/abs/2302.03106](https://arxiv.org/abs/2302.03106)

    本文提出了一种使用预训练的嵌入和句子包进行高效灵活的主题建模方法，通过结合生成过程模型和聚类，提供了使用先验自定义主题-文档分布的可能性，实验表明该方法在计算负担较小的情况下取得了最新的结果。

    

    预训练语言模型在许多自然语言处理任务中取得了最新的突破。然而，在主题建模方面，统计生成模型（如LDA）仍然普遍存在，而这些模型不容易融入上下文词向量。它们可能会生成与人类判断不一致的主题。本文提出了一种新颖的主题建模和推断算法。我们提出了一种句子包（BoS）的方法，以句子作为分析单位。通过结合生成过程模型和聚类，我们使用预训练的句子嵌入。我们基于期望最大化、硬分配和一个退火过程的快速推断算法。评估结果显示，我们的方法能以相对较小的计算需求获得最新的结果。与利用词嵌入的之前方法相比，我们的方法也更加灵活，因为它提供了使用先验自定义主题-文档分布的可能性。

    Pre-trained language models have led to a new state-of-the-art in many NLP tasks. However, for topic modeling, statistical generative models such as LDA are still prevalent, which do not easily allow incorporating contextual word vectors. They might yield topics that do not align well with human judgment. In this work, we propose a novel topic modeling and inference algorithm. We suggest a bag of sentences (BoS) approach using sentences as the unit of analysis. We leverage pre-trained sentence embeddings by combining generative process models and clustering. We derive a fast inference algorithm based on expectation maximization, hard assignments, and an annealing process. The evaluation shows that our method yields state-of-the art results with relatively little computational demands. Our method is also more flexible compared to prior works leveraging word embeddings, since it provides the possibility to customize topic-document distributions using priors. Code and data is at \url{http
    
[^133]: 基于检索的带自然语言监督的解缠表示学习

    Retrieval-based Disentangled Representation Learning with Natural Language Supervision

    [https://arxiv.org/abs/2212.07699](https://arxiv.org/abs/2212.07699)

    本研究提出了基于检索的带自然语言监督的解缠表示学习框架，利用自然语言作为数据变化的代理，通过词汇空间中的双编码器模型实现对数据内在特征的解缠表示学习。

    

    解缠表示学习仍然具有挑战性，因为数据中的基本变化因素并不存在。真实世界数据的固有复杂性使得在有限的因素集中穷尽地列举和概括所有变化是不可行的。然而，值得注意的是，大多数真实世界数据都有语言等价物，通常以文本描述的形式存在。这些语言对应物可以代表数据，并轻松地分解为不同的标记。基于此，我们提出了单词表解缠检索（VDR），这是一个基于检索的框架，利用自然语言作为潜在数据变化的代理，以推动解缠表示学习。我们的方法使用双编码器模型在词汇空间中表示数据和自然语言，使模型能够通过其自然语言对应物区分捕捉数据内在特征的维度，从而促进解缠表示学习。

    Disentangled representation learning remains challenging as the underlying factors of variation in the data do not naturally exist. The inherent complexity of real-world data makes it unfeasible to exhaustively enumerate and encapsulate all its variations within a finite set of factors. However, it is worth noting that most real-world data have linguistic equivalents, typically in the form of textual descriptions. These linguistic counterparts can represent the data and effortlessly decomposed into distinct tokens. In light of this, we present Vocabulary Disentangled Retrieval (VDR), a retrieval-based framework that harnesses natural language as proxies of the underlying data variation to drive disentangled representation learning. Our approach employ a bi-encoder model to represent both data and natural language in a vocabulary space, enabling the model to distinguish dimensions that capture intrinsic characteristics within data through its natural language counterpart, thus facilitat
    
[^134]: 人工神经网络在人类语言习得方面的启示

    What Artificial Neural Networks Can Tell Us About Human Language Acquisition

    [https://arxiv.org/abs/2208.07998](https://arxiv.org/abs/2208.07998)

    机器学习在自然语言处理方面的快速发展潜在地改变了我们对人类语言习得的认识，但目前的人工学习者和人类在学习环境和数据偏好上存在差异。为了增加计算模型学习结果的相关性，需要训练出没有显著优势的模型学习者，以提供概念证明和进行实验干预。

    

    自然语言处理领域的机器学习取得了快速发展，有可能改变人们对人类语言习得的争议。然而，当前人工学习者和人类的学习环境和偏好存在差异，这削弱了从学习模拟中获取的证据的影响力。例如，如今最有效的神经语言模型所训练的语言数据量大约是一个典型儿童可以获得的数据量的一千倍。为了增加计算模型学习结果的相关性，我们需要训练出没有显著优势于人类的模型学习者。如果一个合适的模型成功习得了某种目标语言知识，它可以提供概念证明，证明该目标在假设的人类学习场景中是可以学会的。可信的模型学习者将使我们能够进行实验性干预，从而进行关于学习环境变量的因果推断，并且能够对条件参数进行严谨的测试。

    Rapid progress in machine learning for natural language processing has the potential to transform debates about how humans learn language. However, the learning environments and biases of current artificial learners and humans diverge in ways that weaken the impact of the evidence obtained from learning simulations. For example, today's most effective neural language models are trained on roughly one thousand times the amount of linguistic data available to a typical child. To increase the relevance of learnability results from computational models, we need to train model learners without significant advantages over humans. If an appropriate model successfully acquires some target linguistic knowledge, it can provide a proof of concept that the target is learnable in a hypothesized human learning scenario. Plausible model learners will enable us to carry out experimental manipulations to make causal inferences about variables in the learning environment, and to rigorously test poverty-
    
[^135]: PROXYQA：一种用于评估大型语言模型长篇文本生成的替代框架

    PROXYQA: An Alternative Framework for Evaluating Long-Form Text Generation with Large Language Models. (arXiv:2401.15042v1 [cs.CL])

    [http://arxiv.org/abs/2401.15042](http://arxiv.org/abs/2401.15042)

    PROXYQA是一个用于评估大型语言模型长篇文本生成的替代框架，通过生成详尽的内容，并利用评估器和生成内容作为背景环境，根据评估器回答代理问题的表现来评估生成内容的质量。

    

    大型语言模型（LLM）在长篇文本理解任务中取得了显著的成功。然而，它们生成长篇内容（如报告和文章）的能力尚未得到充分探索。当前的基准不足以充分评估LLMs生成信息丰富且全面的内容，因此需要一种更严格的评估方法。在本研究中，我们介绍了一种名为\textsc{ProxyQA}的框架，用于评估长篇文本生成，包括深入人工策划的涵盖多个领域的“元问题”。每个元问题都包含相应的带注释答案的“代理问题”。LLMs被要求根据这些元问题生成详尽的内容。利用评估器并将生成的内容作为背景环境，\textsc{ProxyQA}根据评估器回答“代理问题”的表现评估生成内容的质量。我们检验了多个LLMs，重点关注了...

    Large Language Models (LLMs) have exhibited remarkable success in long-form context comprehension tasks. However, their capacity to generate long contents, such as reports and articles, remains insufficiently explored. Current benchmarks do not adequately assess LLMs' ability to produce informative and comprehensive content, necessitating a more rigorous evaluation approach. In this study, we introduce \textsc{ProxyQA}, a framework for evaluating long-form text generation, comprising in-depth human-curated \textit{meta-questions} spanning various domains. Each meta-question contains corresponding \textit{proxy-questions} with annotated answers. LLMs are prompted to generate extensive content in response to these meta-questions. Utilizing an evaluator and incorporating generated content as background context, \textsc{ProxyQA} evaluates the quality of generated content based on the evaluator's performance in answering the \textit{proxy-questions}. We examine multiple LLMs, emphasizing \t
    
[^136]: 通过上下文感知个性化细化，增强长期对话中的常识增强性内存构建和管理

    Commonsense-augmented Memory Construction and Management in Long-term Conversations via Context-aware Persona Refinement. (arXiv:2401.14215v1 [cs.CL])

    [http://arxiv.org/abs/2401.14215](http://arxiv.org/abs/2401.14215)

    本文提出了一个旨在解决长期对话中角色句子不具信息性的问题的框架，通过利用常识增强的角色扩展，并设计策略将相互矛盾的角色转化为包含丰富说话者信息的句子，以提高回应生成质量。

    

    在长期对话中，记忆和利用说话者的角色是生成回应的常见做法。然而，人工编写的数据集通常提供无信息的角色句子，这妨碍了回应质量。本文提出了一个新颖的框架，利用常识增强的角色扩展来解决长期对话中的这些问题。以前的工作侧重于不产生与其他角色相矛盾的角色，我们侧重于根据设计的策略，将相互矛盾的角色转化为包含丰富说话者信息的句子，以此来细化它们的上下文背景。作为多会话情境中角色扩展的先驱，我们的框架通过类人个性细化促进了更好的回应生成。

    Memorizing and utilizing speakers' personas is a common practice for response generation in long-term conversations. Yet, human-authored datasets often provide uninformative persona sentences that hinder response quality. This paper presents a novel framework that leverages commonsense-based persona expansion to address such issues in long-term conversation. While prior work focuses on not producing personas that contradict others, we focus on transforming contradictory personas into sentences that contain rich speaker information, by refining them based on their contextual backgrounds with designed strategies. As the pioneer of persona expansion in multi-session settings, our framework facilitates better response generation via human-like persona refinement. The supplementary video of our work is available at https://caffeine-15bbf.web.app/.
    
[^137]: 一种简单的黑盒方法用于越狱攻击

    All in How You Ask for It: Simple Black-Box Method for Jailbreak Attacks. (arXiv:2401.09798v1 [cs.CL])

    [http://arxiv.org/abs/2401.09798](http://arxiv.org/abs/2401.09798)

    本研究提出了一种简单的黑盒方法，用于生成越狱攻击提示，克服了现有方法的复杂性和计算成本的限制。该方法通过使用语言模型自身，将有害提示重写为非有害表达，实现了超过80%的攻击成功率，并且即使模型更新，效果仍然有效。

    

    像ChatGPT这样的大型语言模型面临着“越狱”挑战，即规避保障措施以产生不符合伦理的提示。本研究引入了一种简单的黑盒方法，有效地生成越狱提示，克服了现有方法的高复杂性和计算成本的限制。该方法通过使用目标语言模型自身，迭代地将有害提示重写为非有害表达，基于假设认为语言模型可以直接生成规避保障的表达。通过在ChatGPT（GPT-3.5和GPT-4）和Gemini-Pro上进行实验证明，该方法在平均5次迭代内实现了超过80%的攻击成功率，并且即使模型更新，效果仍然有效。生成的越狱提示自然而简练，表明它们较不易被检测。结果表明，创建有效的越狱提示比先前研究认为的要简单，并且黑盒越狱攻击构成了一个重要的挑战。

    Large Language Models (LLMs) like ChatGPT face `jailbreak' challenges, where safeguards are bypassed to produce ethically harmful prompts. This study introduces a simple black-box method to effectively generate jailbreak prompts, overcoming the limitations of high complexity and computational costs associated with existing methods. The proposed technique iteratively rewrites harmful prompts into non-harmful expressions using the target LLM itself, based on the hypothesis that LLMs can directly sample safeguard-bypassing expressions. Demonstrated through experiments with ChatGPT (GPT-3.5 and GPT-4) and Gemini-Pro, this method achieved an attack success rate of over 80% within an average of 5 iterations and remained effective despite model updates. The jailbreak prompts generated were naturally-worded and concise, suggesting they are less detectable. The results indicate that creating effective jailbreak prompts is simpler than previously considered, and black-box jailbreak attacks pose 
    
[^138]: 学习捷径：关于语言模型中自然语言理解误导性承诺的论文

    Learning Shortcuts: On the Misleading Promise of NLU in Language Models. (arXiv:2401.09615v1 [cs.CL])

    [http://arxiv.org/abs/2401.09615](http://arxiv.org/abs/2401.09615)

    该论文调查了大型语言模型在自然语言理解任务中使用捷径学习的现象，强调了这种现象对语言模型评估的影响，并呼吁加大对捷径学习的研究力度以提升语言模型的鲁棒性和实际场景中的自然语言理解评估标准。

    

    大型语言模型（LLMs）的出现在自然语言处理领域实现了显著的性能提升。然而，最近的研究发现，LLMs在执行任务时常常采用捷径，导致在决策规则上缺乏泛化能力，从而在性能上产生了一种错觉。这一现象在准确评估LLMs的自然语言理解能力上带来了挑战。本文对该领域的相关研究进行了简洁的概述，并提出了在评估语言模型，尤其是自然语言理解任务中使用捷径学习的影响的观点。本文呼吁加大对捷径学习的深入理解的研究力度，为开发更强大的语言模型和提高真实场景下自然语言理解评估的标准作出贡献。

    The advent of large language models (LLMs) has enabled significant performance gains in the field of natural language processing. However, recent studies have found that LLMs often resort to shortcuts when performing tasks, creating an illusion of enhanced performance while lacking generalizability in their decision rules. This phenomenon introduces challenges in accurately assessing natural language understanding in LLMs. Our paper provides a concise survey of relevant research in this area and puts forth a perspective on the implications of shortcut learning in the evaluation of language models, specifically for NLU tasks. This paper urges more research efforts to be put towards deepening our comprehension of shortcut learning, contributing to the development of more robust language models, and raising the standards of NLU evaluation in real-world scenarios.
    
[^139]: 通过迭代组合问题来增强数学问题求解

    Augmenting Math Word Problems via Iterative Question Composing. (arXiv:2401.09003v1 [cs.CL])

    [http://arxiv.org/abs/2401.09003](http://arxiv.org/abs/2401.09003)

    本研究通过引入MMIQC数据集和迭代组合问题(IQC)的新颖增强方法，成功提高了大型语言模型的数学推理能力，在竞赛级数学问题上取得了优于先前最佳结果的准确率。

    

    尽管在改善大型语言模型(LLMs)的数学推理能力方面取得了一定进展，但在不使用外部工具的情况下解决竞赛级数学问题仍然对开源LLMs具有挑战性。在这项工作中，我们介绍了MMIQC数据集，这是一个混合处理的网络数据和合成问题-响应对的混合数据集，以提供基础模型更好的数学推理能力。通过在MMIQC上对Mistral-7B(arXiv:2310.06825)进行微调获得的模型Mistral-7B-MMIQC，在MATH(arXiv:2103.03874)上达到了36.0%的准确率，比之前(model size $\sim$7B)的最佳结果高出5.8%。我们的实验还表明，改进的一个重要部分归功于我们的新颖增强方法IQC(迭代组合问题)，其中我们迭代地要求LLM从给定的种子问题中组合新问题，并从另一个LLM中进行拒绝抽样。MMIQC现已在https://huggingface.co/datasets/Vivacem/MMIQC上发布。

    Despite recent progress in improving the mathematical reasoning ability of large language models(LLMs), solving competition-level math problems without the use of external tools remains challenging for open-source LLMs. In this work, we introduce the MMIQC dataset, a mixture of processed web data and synthetic question-response pairs, to equip base models with better mathematical reasoning skills. Mistral-7B-MMIQC, the model obtained by fine-tuning Mistral-7B(arXiv:2310.06825) on MMIQC, achieves 36.0\% accuracy on MATH(arXiv:2103.03874), 5.8\% higher than the previous (model size $\sim$7B) SOTA. Our experiments also show that a large part of the improvement attributes to our novel augmentation method IQC(Iterative Question Composing), where we iteratively ask an LLM to compose new questions from the given seed problems and do rejection sampling from another LLM. MMIQC has now been released on https://huggingface.co/datasets/Vivacem/MMIQC.
    
[^140]: InfiAgent-DABench: 在数据分析任务中评估代理的基准测试

    InfiAgent-DABench: Evaluating Agents on Data Analysis Tasks. (arXiv:2401.05507v1 [cs.CL])

    [http://arxiv.org/abs/2401.05507](http://arxiv.org/abs/2401.05507)

    InfiAgent-DABench是第一个评估基于LLM的代理在数据分析任务中的基准测试，包括DAEval数据集和代理框架。对23个最先进的LLMs进行的基准测试揭示了当前数据分析任务中的挑战。

    

    本文介绍了"InfiAgent-DABench"，这是第一个专门设计用于评估基于LLM的代理在数据分析任务中的基准测试。该基准测试包含DAEval，这是一个由55个CSV文件衍生出的311个数据分析问题的数据集，以及一个评估LLMs作为数据分析代理的代理框架。我们采用了一种格式提示技术，确保问题是闭合形式的，可以自动评估。我们对23个最先进的LLMs进行了广泛的基准测试，揭示了数据分析任务中当前遇到的挑战。此外，我们还开发了DAAgent，这是一个在指令调优数据集上训练的专门代理。InfiAgent-DABench的评估数据集和工具包已经发布在https://github.com/InfiAgent/InfiAgent上。

    In this paper, we introduce "InfiAgent-DABench", the first benchmark specifically designed to evaluate LLM-based agents in data analysis tasks. This benchmark contains DAEval, a dataset consisting of 311 data analysis questions derived from 55 CSV files, and an agent framework to evaluate LLMs as data analysis agents. We adopt a format-prompting technique, ensuring questions to be closed-form that can be automatically evaluated. Our extensive benchmarking of 23 state-of-the-art LLMs uncovers the current challenges encountered in data analysis tasks. In addition, we have developed DAAgent, a specialized agent trained on instruction-tuning datasets. Evaluation datasets and toolkits for InfiAgent-DABench are released at https://github.com/InfiAgent/InfiAgent.
    
[^141]: 超越提取：为语言模型提供上下文化的表格数据以实现高效摘要

    Beyond Extraction: Contextualising Tabular Data for Efficient Summarisation by Language Models. (arXiv:2401.02333v1 [cs.LG])

    [http://arxiv.org/abs/2401.02333](http://arxiv.org/abs/2401.02333)

    本研究提出了一种创新的方法，通过上下文化表格数据来提高 RAG 系统中处理复杂表格查询的准确性，提高了摘要的效率。

    

    传统的检索增强生成 (RAG) 架构在从各种文件中检索信息方面已被证明是有效的。然而，在处理包含复杂表格结构的 PDF 文档中的复杂表格查询时会遇到挑战。本研究引入了一种创新的方法来提高 RAG 系统中复杂表格查询的准确性。我们的方法涉及将 PDF 存储在检索数据库中，并单独提取表格内容。提取的表格经过上下文丰富的处理，将标题与相应的值连接起来。为了确保对丰富数据的全面理解，我们使用经过微调的 Llama-2-chat 语言模型在 RAG 架构中进行摘要。此外，我们通过一次性提示使用 ChatGPT 3.5 API 增强表格数据的上下文含义。然后，将这些丰富的数据与其他 PDF 文件一起输入检索数据库。

    The conventional use of the Retrieval-Augmented Generation (RAG) architecture has proven effective for retrieving information from diverse documents. However, challenges arise in handling complex table queries, especially within PDF documents containing intricate tabular structures.This research introduces an innovative approach to enhance the accuracy of complex table queries in RAG-based systems. Our methodology involves storing PDFs in the retrieval database and extracting tabular content separately. The extracted tables undergo a process of context enrichment, concatenating headers with corresponding values. To ensure a comprehensive understanding of the enriched data, we employ a fine-tuned version of the Llama-2-chat language model for summarisation within the RAG architecture. Furthermore, we augment the tabular data with contextual sense using the ChatGPT 3.5 API through a one-shot prompt. This enriched data is then fed into the retrieval database alongside other PDFs. Our appr
    
[^142]: NPHardEval: 通过复杂性类别对大型语言模型的推理能力进行动态基准评估

    NPHardEval: Dynamic Benchmark on Reasoning Ability of Large Language Models via Complexity Classes. (arXiv:2312.14890v3 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2312.14890](http://arxiv.org/abs/2312.14890)

    NPHardEval是一个新的基准，旨在评估大型语言模型在900个算法问题上的推理能力，扩展到NP-Hard复杂性类别。

    

    复杂推理能力是当前大型语言模型的最重要特征之一，它也被用于在复杂决策任务中起到了重要作用。因此，研究大型语言模型的推理能力至关重要：已经建立了许多基准来评估大型语言模型的推理能力。然而，目前的基准在提供大型语言模型推理能力的全面评估方面还不够，同时也容易出现过拟合的风险，因为这些基准是公开可访问且静态的，使得模型有可能根据特定的基准指标调整其响应，从而夸大其性能。针对这些限制，我们的研究引入了一个新的基准，名为NPHardEval。该基准旨在评估大型语言模型在广泛的900个算法问题上的推理能力，涵盖了NP-Hard复杂性类别。

    Complex reasoning ability is one of the most important features of current LLMs, which has also been leveraged to play an integral role in complex decision-making tasks. Therefore, the investigation into the reasoning capabilities of Large Language Models (LLMs) is critical: numerous benchmarks have been established to assess the reasoning abilities of LLMs. However, current benchmarks are inadequate in offering a rigorous evaluation of the full extent of reasoning abilities that LLMs are capable of achieving. They are also prone to the risk of overfitting, as these benchmarks, being publicly accessible and static, allow models to potentially tailor their responses to specific benchmark metrics, thereby inflating their performance. Addressing these limitations, our research introduces a new benchmark, named NPHardEval. This benchmark is designed to evaluate the reasoning abilities of LLMs across a broad spectrum of 900 algorithmic questions, extending up to the NP-Hard complexity class
    
[^143]: 多个任务预训练和图形提示的MultiGPrompt

    MultiGPrompt for Multi-Task Pre-Training and Prompting on Graphs. (arXiv:2312.03731v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2312.03731](http://arxiv.org/abs/2312.03731)

    本文提出了一种名为MultiGPrompt的多任务预训练和提示框架，用于在图形表示学习中提高鲁棒性和减少标注成本。

    

    图形可以固有地对Web上相互连接的对象进行建模，从而支持一系列Web应用，比如网络分析和内容推荐。最近，图神经网络（GNNs）已经成为图表示学习的主流技术。然而，在端到端监督框架中，它们的有效性与任务特定标签的可用性密切相关。为了减少标注成本并增强在少样本设置中的鲁棒性，基于自监督任务的预训练已经成为一种有前途的方法，而提示则被提出来进一步缩小预训练任务与下游任务之间的目标差距。虽然已经对基于提示的图形学习进行了初步的探索，但它们主要利用单个预训练任务，导致从预训练数据中可能学习的通用知识的子集受限。因此，在本文中，我们提出了一种新颖的多任务预训练和提示框架MultiGPrompt，用于进一步提高对图形的表示学习。

    Graphs can inherently model interconnected objects on the Web, thereby facilitating a series of Web applications, such as web analyzing and content recommendation. Recently, Graph Neural Networks (GNNs) have emerged as a mainstream technique for graph representation learning. However, their efficacy within an end-to-end supervised framework is significantly tied to the availabilityof task-specific labels. To mitigate labeling costs and enhance robustness in few-shot settings, pre-training on self-supervised tasks has emerged as a promising method, while prompting has been proposed to further narrow the objective gap between pretext and downstream tasks. Although there has been some initial exploration of prompt-based learning on graphs, they primarily leverage a single pretext task, resulting in a limited subset of general knowledge that could be learned from the pre-training data. Hence, in this paper, we propose MultiGPrompt, a novel multi-task pre-training and prompting framework to
    
[^144]: ViCrop: 利用多模态大型语言模型在零样本视觉问答中感知细小视觉细节

    ViCrop: Perceiving Small Visual Details in Zero-shot Visual Question Answering with Multimodal Large Language Models. (arXiv:2310.16033v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2310.16033](http://arxiv.org/abs/2310.16033)

    本研究探讨了多模态大型语言模型在零样本视觉问答中感知细小视觉细节的能力。实验表明，这些模型对于与问题相关的视觉主题的尺寸非常敏感，通过引入人类可视剪裁可以显著提升其准确性。

    

    多模态大型语言模型(MLLMs)在视觉问答(VQA)上取得了令人期待的零样本准确性，这是一个影响各种下游应用和领域的基本任务。鉴于这些模型的广泛使用潜力，研究它们在处理不同的图像和问题属性方面的限制非常重要。在这项工作中，我们研究了MLLMs是否能够像较大的组件一样感知图像中的细节。特别是，我们发现它们在回答视觉问题时对与问题相关的视觉主题的尺寸非常敏感，并且随着尺寸的减小，零样本准确性下降多达45.91%。此外，通过观察到人类可视剪裁可以显著减轻其对尺寸的敏感性，我们证明了这种效应是因果的。为了扩大人类可视剪裁的实用性，我们提出了ViCrop，这是一个利用自动可视剪裁来增强MLLMs零样本VQA的通用框架。

    Multimodal Large Language Models (MLLMs) have recently achieved promising zero-shot accuracy on visual question answering (VQA) -- a fundamental task affecting various downstream applications and domains. Given the great potential for the broad use of these models, it is important to investigate their limitations in dealing with different image and question properties. In this work, we investigate whether MLLMs can perceive details as well as larger components in images. In particular, we show that their zero-shot accuracy in answering visual questions is very sensitive to the size of the visual subject related to the question, declining up to $45.91\%$ with size. Furthermore, we show that this effect is causal by observing that human visual cropping can significantly mitigate their sensitivity to size. To scale up the usefulness of human cropping, we propose ViCrop, a general framework that utilizes automatic visual cropping to enhance zero-shot VQA of MLLMs. We construct five variant
    
[^145]: 使用SAM进行建筑物分割模型的零-shot细化

    Zero-Shot Refinement of Buildings' Segmentation Models using SAM. (arXiv:2310.01845v1 [cs.CV])

    [http://arxiv.org/abs/2310.01845](http://arxiv.org/abs/2310.01845)

    本文提出了一种使用SAM进行建筑物分割模型的零-shot细化的方法，针对遥感图像应用中SAM性能不佳、无法进行识别的问题进行了处理。通过引入不同的提示来提升模型的泛化能力。

    

    基础模型在各种任务中表现出色，但通常在常规基准测试中评估。将这些模型应用于特定领域，如遥感图像，仍然是一个未充分开发的领域。在遥感领域中，精确的建筑物实例分割对于城市规划等应用至关重要。虽然卷积神经网络（CNN）表现良好，但它们的泛化能力可能受限。为了实现这一目标，我们提出了一种新的方法，以使基础模型适应已有模型的泛化性能下降。在众多模型中，我们的重点在于Segment Anything Model（SAM），这是一种强大的基础模型，以其擅长无类别图像分割能力而闻名。我们首先确定了SAM的局限性，揭示了它在应用于遥感图像时性能不佳。此外，SAM不具备识别能力，因此无法对定位的对象进行分类和标记。为了解决这些限制，我们引入了不同的提示

    Foundation models have excelled in various tasks but are often evaluated on general benchmarks. The adaptation of these models for specific domains, such as remote sensing imagery, remains an underexplored area. In remote sensing, precise building instance segmentation is vital for applications like urban planning. While Convolutional Neural Networks (CNNs) perform well, their generalization can be limited. For this aim, we present a novel approach to adapt foundation models to address existing models' generalization dropback. Among several models, our focus centers on the Segment Anything Model (SAM), a potent foundation model renowned for its prowess in class-agnostic image segmentation capabilities. We start by identifying the limitations of SAM, revealing its suboptimal performance when applied to remote sensing imagery. Moreover, SAM does not offer recognition abilities and thus fails to classify and tag localized objects. To address these limitations, we introduce different promp
    
[^146]: 分布包含假设与量化：在功能分布语义中探究上位词关系

    Distributional Inclusion Hypothesis and Quantifications: Probing Hypernymy in Functional Distributional Semantics. (arXiv:2309.08325v1 [cs.CL])

    [http://arxiv.org/abs/2309.08325](http://arxiv.org/abs/2309.08325)

    本文研究了在功能分布语义中，当语料库严格遵循分布包含假设时，功能分布语义模型可以学习到上位词关系。同时，引入一种训练目标使得模型可以处理普遍量化，从而在分布包含假设的反向下实现上位词关系的学习。实验结果验证了这些假设和目标的有效性。

    

    功能分布语义（FDS）通过真条件函数对单词的含义进行建模。当语料库严格遵循分布包含假设时，FDS模型可以学习到上位词关系。我们进一步引入了一种训练目标，使得FDS模型可以处理简单的普遍量化，从而在分布包含假设的反向下实现上位词关系的学习。对合成数据集和真实数据集的实验结果验证了我们的假设以及我们提出的目标的有效性。

    Functional Distributional Semantics (FDS) models the meaning of words by truth-conditional functions. This provides a natural representation for hypernymy, but no guarantee that it is learnt when FDS models are trained on a corpus. We demonstrate that FDS models learn hypernymy when a corpus strictly follows the Distributional Inclusion Hypothesis. We further introduce a training objective that allows FDS to handle simple universal quantifications, thus enabling hypernymy learning under the reverse of DIH. Experimental results on both synthetic and real data sets confirm our hypotheses and the effectiveness of our proposed objective.
    
[^147]: 证明LLM对抗敌对提示的安全性

    Certifying LLM Safety against Adversarial Prompting. (arXiv:2309.02705v1 [cs.CL])

    [http://arxiv.org/abs/2309.02705](http://arxiv.org/abs/2309.02705)

    本研究提出了首个具有可验证安全保证的框架——消除和检查，用于对抗敌对提示。通过逐个消除标记并使用安全过滤器检查生成的子序列，确保任何敌对修改的有害输入提示都能被正确标识为有害。

    

    为了确保语言模型的输出安全，公开使用的大型语言模型（LLM）引入了所谓的“模型对齐”防护措施。一个对齐的语言模型应该拒绝用户的请求生成有害内容。然而，这种安全措施容易受到敌对提示的攻击，敌对提示包含恶意设计的标记序列，以规避模型的安全防护并导致生成有害内容。在这项工作中，我们介绍了可验证安全保证的第一个对抗敌对提示的框架——消除和检查。我们逐个消除标记，并使用安全过滤器检查生成的子序列。如果安全过滤器检测到任何子序列或输入提示有害，我们的过程将将输入提示标记为有害。这保证了对于某个特定大小的有害输入提示的任何敌对修改也将被标记为有害。我们对抗三种攻击模式：i)敌对后缀，即附加敌对序列…

    Large language models (LLMs) released for public use incorporate guardrails to ensure their output is safe, often referred to as "model alignment." An aligned language model should decline a user's request to produce harmful content. However, such safety measures are vulnerable to adversarial prompts, which contain maliciously designed token sequences to circumvent the model's safety guards and cause it to produce harmful content. In this work, we introduce erase-and-check, the first framework to defend against adversarial prompts with verifiable safety guarantees. We erase tokens individually and inspect the resulting subsequences using a safety filter. Our procedure labels the input prompt as harmful if any subsequences or the input prompt are detected as harmful by the filter. This guarantees that any adversarial modification of a harmful prompt up to a certain size is also labeled harmful. We defend against three attack modes: i) adversarial suffix, which appends an adversarial seq
    
[^148]: 生成器-检索器-生成器：开放域问答的新方法

    Generator-Retriever-Generator: A Novel Approach to Open-domain Question Answering. (arXiv:2307.11278v1 [cs.CL])

    [http://arxiv.org/abs/2307.11278](http://arxiv.org/abs/2307.11278)

    生成器-检索器-生成器（GRG）是一种新方法，将文档检索技术与大型语言模型相结合，以生成开放域问答的准确和信息丰富的答案。

    

    开放域问答任务通常需要从大型语料库中检索相关信息以生成准确的答案。我们提出了一种称为生成器-检索器-生成器（GRG）的新方法，将文档检索技术与大型语言模型（LLM）相结合，首先通过给定问题提示模型生成上下文文档。同时，双编码器网络从外部语料库中检索与问题相关的文档。生成和检索的文档然后传递给第二个LLM，生成最终答案。通过结合文档检索和LLM生成，我们的方法解决了开放域问答的挑战，例如生成信息丰富和上下文相关的答案。GRG在TriviaQA、NQ和WebQ数据集上表现优于现有的生成-读取和检索-读取流水线（GENREAD和RFiD），分别至少提高了+5.2、+4.2和+1.6的性能。

    Open-domain question answering (QA) tasks usually require the retrieval of relevant information from a large corpus to generate accurate answers. We propose a novel approach called Generator-Retriever-Generator (GRG) that combines document retrieval techniques with a large language model (LLM), by first prompting the model to generate contextual documents based on a given question. In parallel, a dual-encoder network retrieves documents that are relevant to the question from an external corpus. The generated and retrieved documents are then passed to the second LLM, which generates the final answer. By combining document retrieval and LLM generation, our approach addresses the challenges of open-domain QA, such as generating informative and contextually relevant answers. GRG outperforms the state-of-the-art generate-then-read and retrieve-then-read pipelines (GENREAD and RFiD) improving their performance at least by +5.2, +4.2, and +1.6 on TriviaQA, NQ, and WebQ datasets, respectively.
    
[^149]: 使用大型语言模型增强密集检索的软提示调优

    Soft Prompt Tuning for Augmenting Dense Retrieval with Large Language Models. (arXiv:2307.08303v1 [cs.IR] CROSS LISTED)

    [http://arxiv.org/abs/2307.08303](http://arxiv.org/abs/2307.08303)

    本论文提出了一种使用软提示调优来增强密集检索的方法（SPTAR）。通过优化任务特定的软提示并利用大型语言模型为未标记的文档生成弱查询，可以提高零样本和少样本的密集检索模型的性能。

    

    密集检索（DR）将查询和文档转化为密集向量表示，并在向量空间中测量查询与文档之间的相似性。DR的一个挑战是缺乏领域特定的训练数据。虽然DR模型可以通过迁移学习从大规模公共数据集（如MS MARCO）中学习，但证据表明，并非所有DR模型和领域都能同等受益于迁移学习。最近，一些研究人员转向使用大型语言模型（LLMs）来改进零样本和少样本的DR模型。然而，这些方法中采用的硬提示或人工编写的提示无法保证生成的弱查询的质量。为了解决这个问题，我们提出了用于增强DR的软提示调优（SPTAR）：对于每个任务，我们利用软提示调优在有限的真实数据上优化任务特定的软提示，然后用这些提示引导LLMs为未标记的文档标记弱查询，从而得到足够的弱文档-查询对来训练任务特定的模型。

    Dense retrieval (DR) converts queries and documents into dense embeddings and measures the similarity between queries and documents in vector space. One of the challenges in DR is the lack of domain-specific training data. While DR models can learn from large-scale public datasets like MS MARCO through transfer learning, evidence shows that not all DR models and domains can benefit from transfer learning equally. Recently, some researchers have resorted to large language models (LLMs) to improve the zero-shot and few-shot DR models. However, the hard prompts or human-written prompts utilized in these works cannot guarantee the good quality of generated weak queries. To tackle this, we propose soft prompt tuning for augmenting DR (SPTAR): For each task, we leverage soft prompt-tuning to optimize a task-specific soft prompt on limited ground truth data and then prompt the LLMs to tag unlabeled documents with weak queries, yielding enough weak document-query pairs to train task-specific d
    
[^150]: 解码电视剧的流行程度：一个网络分析的视角

    Decoding the Popularity of TV Series: A Network Analysis Perspective. (arXiv:2307.05329v1 [cs.SI])

    [http://arxiv.org/abs/2307.05329](http://arxiv.org/abs/2307.05329)

    从电视剧的角色网络中提取网络指标，研究发现对电视剧的评论分数具有很强的相关性，为电视制片人提供了定量信息，帮助他们调整角色动态以吸引观众。

    

    在本文中，我们分析了从三部流行电视剧中提取的角色网络，并探讨了电视剧集的角色网络指标与IMDB评论之间的关系。角色网络是从电视剧情节中创建的图形，表示场景中角色之间的交互，指示它们之间是否存在连接。我们为每集计算了各种网络指标，如节点度和图形密度，并使用这些指标来探索网络指标与电视剧在IMDB上的评价之间的潜在关系。我们的研究结果表明，电视剧集中的角色互动的某些网络指标与电视剧的评论分数具有很强的相关性。我们的研究旨在提供更多定量信息，帮助电视制片人了解如何调整未来剧集的角色动态，以吸引观众。通过理解角色互动对观众参与度的影响

    In this paper, we analyze the character networks extracted from three popular television series and explore the relationship between a TV show episode's character network metrics and its review from IMDB. Character networks are graphs created from the plot of a TV show that represents the interactions of characters in scenes, indicating the presence of a connection between them. We calculate various network metrics for each episode, such as node degree and graph density, and use these metrics to explore the potential relationship between network metrics and TV series reviews from IMDB. Our results show that certain network metrics of character interactions in episodes have a strong correlation with the review score of TV series. Our research aims to provide more quantitative information that can help TV producers understand how to adjust the character dynamics of future episodes to appeal to their audience. By understanding the impact of character interactions on audience engagement an
    
[^151]: MLM预训练的动态掩码率调度

    Dynamic Masking Rate Schedules for MLM Pretraining. (arXiv:2305.15096v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.15096](http://arxiv.org/abs/2305.15096)

    本论文提出了一种动态调度掩码率的方法来改进MLM预训练的质量，通过线性降低掩码率，达到了对BERT-base和BERT-large模型分别提高0.46%和0.25%的平均GLUE准确率的效果。这种方法不仅加快了BERT-base的预训练速度，还实现了对BERT-large的帕累托改善。

    

    大多数使用掩码语言建模（MLM）目标进行训练的transformer模型使用了原始BERT模型的固定掩码率15%。我们提出了通过训练过程中动态调整掩码率来替代固定率的方法。我们发现，在预训练过程中线性降低掩码率可以比固定率基准分别提高BERT-base和BERT-large的平均GLUE准确率0.46%和0.25%。这些提升来自于接触高和低掩码率的机制，从而在两种设置中都带来了优势。我们的结果表明，掩码率调度是提高掩码语言模型质量的简单方法，可以使BERT-base的预训练速度提高1.89倍，并对BERT-large实现了帕累托改善。

    Most works on transformers trained with the Masked Language Modeling (MLM) objective use the original BERT model's fixed masking rate of 15%. We propose to instead dynamically schedule the masking rate throughout training. We find that linearly decreasing the masking rate over the course of pretraining improves average GLUE accuracy by up to 0.46% and 0.25% in BERT-base and BERT-large, respectively, compared to fixed rate baselines. These gains come from exposure to both high and low masking rate regimes, providing benefits from both settings. Our results demonstrate that masking rate scheduling is a simple way to improve the quality of masked language models, achieving up to a 1.89x speedup in pretraining for BERT-base as well as a Pareto improvement for BERT-large.
    
[^152]: DAPR：文档感知段落检索的基准测试

    DAPR: A Benchmark on Document-Aware Passage Retrieval. (arXiv:2305.13915v1 [cs.IR])

    [http://arxiv.org/abs/2305.13915](http://arxiv.org/abs/2305.13915)

    DAPR是一个文档感知段落检索的基准测试，挑战在于如何从长文档中找到正确的段落并返回准确结果。

    

    最近的神经检索主要关注短文本的排名，并且在处理长文档方面存在挑战。现有的工作主要评估排名段落或整个文档。然而，许多情况下，用户希望从庞大的语料库中找到长文档中的相关段落，例如法律案例，研究论文等，此时段落往往提供很少的文档上下文，这就挑战了当前的方法找到正确的文档并返回准确的结果。为了填补这个空白，我们提出并命名了Document-Aware Passage Retrieval（DAPR）任务，并构建了一个包括来自不同领域的多个数据集的基准测试，涵盖了DAPR和整个文档检索。在实验中，我们通过不同的方法，包括在文档摘要中添加文档级别的内容，汇总段落表示和使用BM25进行混合检索，扩展了最先进的神经段落检索器。这个混合检索系统，总体基准测试显示，我们提出的DAPR任务是一个具有挑战性和重要性的问题，需要进一步研究。

    Recent neural retrieval mainly focuses on ranking short texts and is challenged with long documents. Existing work mainly evaluates either ranking passages or whole documents. However, there are many cases where the users want to find a relevant passage within a long document from a huge corpus, e.g. legal cases, research papers, etc. In this scenario, the passage often provides little document context and thus challenges the current approaches to finding the correct document and returning accurate results. To fill this gap, we propose and name this task Document-Aware Passage Retrieval (DAPR) and build a benchmark including multiple datasets from various domains, covering both DAPR and whole-document retrieval. In experiments, we extend the state-of-the-art neural passage retrievers with document-level context via different approaches including prepending document summary, pooling over passage representations, and hybrid retrieval with BM25. The hybrid-retrieval systems, the overall b
    
[^153]: 民主扩散语言模型

    Democratized Diffusion Language Model. (arXiv:2305.10818v1 [cs.LG])

    [http://arxiv.org/abs/2305.10818](http://arxiv.org/abs/2305.10818)

    本文提出了一个基于CDCD框架的民主扩散语言模型（DDLM），并通过GLUE基准测试了其知识转移能力，为研究人员提供了DDLM训练和评估流程以及已训练的DDLM模型。

    

    尽管扩散模型在自然语言处理中有潜在好处，但目前公开的实现、训练模型或可重现的训练程序并不存在。为解决这些挑战，我们提出了基于CDCD框架的民主扩散语言模型（DDLM）。我们提出了一种用C4数据集简化的DDLM训练流程，并对训练模型的行为进行了深入分析。此外，我们引入了一种用于速度更快的采样的新型早期退出策略，该策略针对使用得分插值训练的模型。由于此前没有研究旨在使用预训练扩散LM解决下游任务（例如分类任务），我们在GLUE基准上进行了实验，以研究DDLM的知识转移能力。通过本文，我们提出了可供其他研究人员使用的DDLM训练和评估流程以及预先训练的DDLM模型，这些模型可在未来的D相关的研究中使用。

    Despite the potential benefits of Diffusion Models for NLP applications, publicly available implementations, trained models, or reproducible training procedures currently need to be publicly available. We present the Democratized Diffusion Language Model (DDLM), based on the Continuous Diffusion for Categorical Data (CDCD) framework, to address these challenges. We propose a simplified training procedure for DDLM using the C4 dataset and perform an in-depth analysis of the trained model's behavior. Furthermore, we introduce a novel early-exiting strategy for faster sampling with models trained with score interpolation. Since no previous works aimed at solving downstream tasks with pre-trained Diffusion LM (e.g., classification tasks), we experimented with GLUE Benchmark to study the ability of DDLM to transfer knowledge. With this paper, we propose available training and evaluation pipelines to other researchers and pre-trained DDLM models, which could be used in future research with D
    
[^154]: 小型语言模型更适合作为黑匣子机器生成文本检测器

    Smaller Language Models are Better Black-box Machine-Generated Text Detectors. (arXiv:2305.09859v1 [cs.CL])

    [http://arxiv.org/abs/2305.09859](http://arxiv.org/abs/2305.09859)

    本文研究发现，小型语言模型更适用于作为通用文本检测器，可以更加精确地检测出机器生成的文本，而检测器和生成模型是否具有相同的架构或语料库并不会对检测性能产生显著影响。

    

    随着流畅的生成语言模型的出现，它们可以生成与人类写作的非常相似的令人信服的话语，因此区分一段文本是由机器生成的还是人类写作的变得更加具有挑战性和重要性，因为这样的模型可以用于传播错误信息、虚假新闻、虚假评论并模仿某些作者和人物。为此，已经提出了许多检测机器生成文本的方法。其中大部分方法需要访问目标模型的 logits，或需要可以从目标模型中进行采样的能力。其中一种黑匣子检测方法依赖于观察到生成文本在生成器的似然函数下是局部最优的，而人类写作的文本则不是。我们发现，总体而言，较小且部分训练的模型更适合作为通用文本检测器：它们可以更精确地检测来自小型和大型模型的生成文本。有趣的是，我们发现检测器和生成模型是否具有相同的架构或相同的语料库对检测性能没有显著影响。

    With the advent of fluent generative language models that can produce convincing utterances very similar to those written by humans, distinguishing whether a piece of text is machine-generated or human-written becomes more challenging and more important, as such models could be used to spread misinformation, fake news, fake reviews and to mimic certain authors and figures. To this end, there have been a slew of methods proposed to detect machine-generated text. Most of these methods need access to the logits of the target model or need the ability to sample from the target. One such black-box detection method relies on the observation that generated text is locally optimal under the likelihood function of the generator, while human-written text is not. We find that overall, smaller and partially-trained models are better universal text detectors: they can more precisely detect text generated from both small and larger models. Interestingly, we find that whether the detector and generat
    

