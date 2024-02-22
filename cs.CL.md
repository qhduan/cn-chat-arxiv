# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Coercing LLMs to do and reveal (almost) anything](https://arxiv.org/abs/2402.14020) | 本研究发现对大型语言模型的对抗性攻击不仅仅局限于“越狱”，而包括迫使模型展示各种意外行为，攻击表面和目标广泛。这些攻击源于LLMs的预训练和常见词汇中存在的“故障”标记。 |
| [^2] | [Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment](https://arxiv.org/abs/2402.14016) | 该研究研究了评估LLM的对抗鲁棒性，发现短通用短语可以欺骗LLMs提供高评分，这种攻击对于从简单的串联攻击到转移学习都是有效的。 |
| [^3] | [OlympiadBench: A Challenging Benchmark for Promoting AGI with Olympiad-Level Bilingual Multimodal Scientific Problems](https://arxiv.org/abs/2402.14008) | 提出了OlympiadBench，一个奥林匹亚级别的双语多模态科学基准，包括8952个问题，旨在评估大型语言模型和多模态模型在复杂问题上的能力。 |
| [^4] | [Can Watermarks Survive Translation? On the Cross-lingual Consistency of Text Watermark for Large Language Models](https://arxiv.org/abs/2402.14007) | 该研究引入了文本水印中的“跨语言一致性”概念，发现当前文本水印技术在文本被翻译成其他语言后失去了一致性，并提出了一种跨语言水印去除攻击方法，有效绕过水印，降低AUC值，同时指出了导致这种差异的关键因素。 |
| [^5] | [Hallucinations or Attention Misdirection? The Path to Strategic Value Extraction in Business Using Large Language Models](https://arxiv.org/abs/2402.14002) | 本文提出在商业环境中使用大型语言模型进行战略价值提取时，需要理解幻觉和注意力误导之间的区别，并强调了一个达到显著错误的战略框架。 |
| [^6] | [Analysing The Impact of Sequence Composition on Language Model Pre-Training](https://arxiv.org/abs/2402.13991) | 序列组成对语言模型预训练的影响一直未被深入探讨，在这项研究中发现，应用内部文档因果遮盖可以消除来自之前文档的干扰信息，显著提高模型在语言建模和下游任务中的性能。 |
| [^7] | [Towards Building Multilingual Language Model for Medicine](https://arxiv.org/abs/2402.13963) | 本文提出了为医学领域构建多语言语言模型的三个关键贡献:构建了新的多语言医学语料库MMedC，提出了多语言医学多选问答基准MMedBench，并且通过在MMedC上进一步训练获得了性能优越的MMedLM 2模型。 |
| [^8] | [Can You Learn Semantics Through Next-Word Prediction? The Case of Entailment](https://arxiv.org/abs/2402.13956) | 作者调查了神经LM是否可以通过下一个词预测来解码蕴涵判断，发现它们可以远高于随机几率地解码自然句子之间的蕴涵关系，暗示LM隐含地模拟了语义的某些方面。 |
| [^9] | [Measuring Social Biases in Masked Language Models by Proxy of Prediction Quality](https://arxiv.org/abs/2402.13954) | 本文通过提出的代理函数在迭代屏蔽实验中评估了转换器模型所编码的社会偏见，并比较了其与其他评估方法的偏见估计，发现转换器模型中存在相对较高的宗教和残疾偏见，而性别偏见则相对较低。 |
| [^10] | [Making Reasoning Matter: Measuring and Improving Faithfulness of Chain-of-Thought Reasoning](https://arxiv.org/abs/2402.13950) | 本文研究了大型语言模型的推理过程中的忠实性问题，引入了FRODO框架来改进生成推理步骤和坚固推理的方法 |
| [^11] | [Distinctive Image Captioning: Leveraging Ground Truth Captions in CLIP Guided Reinforcement Learning](https://arxiv.org/abs/2402.13936) | 论文提出了一种新的图像字幕模型训练策略，在强化学习框架中利用地面真实字幕，以提高生成字幕的独特性。 |
| [^12] | [Do Efficient Transformers Really Save Computation?](https://arxiv.org/abs/2402.13934) | 本研究旨在理解高效Transformer（例如稀疏Transformer和线性Transformer）的能力和限制，发现它们适合解决一般DP任务，但不同于标准Transformer。 |
| [^13] | [Large Language Models are Vulnerable to Bait-and-Switch Attacks for Generating Harmful Content](https://arxiv.org/abs/2402.13926) | 大型语言模型可能受到诱饵-转换攻击的威胁，甚至安全生成的文本也能轻易转变为有害内容，强调在LLMs的安全防护中需要考虑后处理转换。 |
| [^14] | [SYNFAC-EDIT: Synthetic Imitation Edit Feedback for Factual Alignment in Clinical Summarization](https://arxiv.org/abs/2402.13919) | 该研究提出了一种创新流程，利用GPT-3.5和GPT-4生成高质量反馈，以增强临床笔记摘要中的事实一致性，弥补了专家注释数据的高成本和有限可用性问题。 |
| [^15] | [What Linguistic Features and Languages are Important in LLM Translation?](https://arxiv.org/abs/2402.13917) | Llama2模型在翻译中表现出准确度高，部分未见语言需要更大规模的模型来提升翻译质量，另外语言的句法相似性并非翻译质量的主要因素，某些语言即使数据少依然表现出强相关性。 |
| [^16] | [Leveraging Collection-Wide Similarities for Unsupervised Document Structure Extraction](https://arxiv.org/abs/2402.13906) | 通过无监督图方式，利用文档间和文内相似性，提取文档收藏的整体结构。 |
| [^17] | [Calibrating Large Language Models with Sample Consistency](https://arxiv.org/abs/2402.13904) | 通过多个随机抽样模型生成的分布推断置信度的潜力，可以增强大型语言模型的校准性能。 |
| [^18] | [Science Checker Reloaded: A Bidirectional Paradigm for Transparency and Logical Reasoning](https://arxiv.org/abs/2402.13897) | 提出了一个两块式的方法来解决长文档中信息检索领域的挑战，并实现了双向交互 |
| [^19] | [Beyond Probabilities: Unveiling the Misalignment in Evaluating Large Language Models](https://arxiv.org/abs/2402.13887) | 本研究揭示了在使用大型语言模型进行多项选择题时，基于概率的评估方法与基于生成的预测不相吻合的固有局限性。 |
| [^20] | [$\texttt{Se}^2$: $\textit{Se}$quential Example $\textit{Se}$lection for In-Context Learning](https://arxiv.org/abs/2402.13874) | 本文提出了$\texttt{Se}^2$，一种顺序感知方法，利用大型语言模型的反馈帮助捕捉示例之间的相互关系和序列信息，显著丰富了上下文学习提示的相关性和相关性。 |
| [^21] | [Kuaiji: the First Chinese Accounting Large Language Model](https://arxiv.org/abs/2402.13866) | Kuaiji是第一个中国会计大型语言模型，通过Baichuan框架精心调整，支持的CAtAcctQA数据集，展现出卓越的准确性和响应速度，具有开创性地创建了中国会计数据集，并证实了在真实会计场景中的高效性。 |
| [^22] | [Large Language Models are Advanced Anonymizers](https://arxiv.org/abs/2402.13846) | 大型语言模型在保护个人数据方面取得了重要进展，提出了一种基于对抗性LLM推断的匿名化框架。 |
| [^23] | [Beyond Hate Speech: NLP's Challenges and Opportunities in Uncovering Dehumanizing Language](https://arxiv.org/abs/2402.13818) | 本文评估了几种最先进的NLP模型在识别贬低性语言方面的性能，发现它们能够以70%的准确率区分贬低性语言和更广泛的仇恨言论，但也存在着偏见。 |
| [^24] | [The Geography of Information Diffusion in Online Discourse on Europe and Migration](https://arxiv.org/abs/2402.13800) | 通过社交媒体数据分析了关于欧洲和移民的在线信息传播，引入了具有地理联系的热门话题、规模和动态传播的新视角，同时提出了一种基于跨语引语的创新方法。 |
| [^25] | [CriticBench: Evaluating Large Language Models as Critic](https://arxiv.org/abs/2402.13764) | CriticBench是一个旨在全面和可靠地评估大型语言模型的评论能力的新型基准，展示了评论能力与任务、响应质量和模型规模之间的关系。 |
| [^26] | [Factual Consistency Evaluation of Summarisation in the Era of Large Language Models](https://arxiv.org/abs/2402.13758) | 在摘要一致性评估方面，该研究通过引入临床文本摘要的数据集TreatFact并对11个大语言模型进行评估，填补了关于大语言模型在摘要事实一致性评估方面的缺口。 |
| [^27] | [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) | LongRoPE首次将预训练的LLM上下文窗口扩展至2048k个标记，通过关键创新实现了这一突破。 |
| [^28] | [Breaking the Barrier: Utilizing Large Language Models for Industrial Recommendation Systems through an Inferential Knowledge Graph](https://arxiv.org/abs/2402.13750) | 提出了一种基于大型语言模型的互补知识增强推荐系统（LLM-KERec），通过引入实体提取器和构建互补知识图，解决了推荐系统难以捕捉用户意图转变和适应新商品的挑战。 |
| [^29] | [Unlocking Instructive In-Context Learning with Tabular Prompting for Relational Triple Extraction](https://arxiv.org/abs/2402.13741) | 设计了表格提示以解决关系三元组抽取中的提示设计和样本选择挑战。 |
| [^30] | [From Text to CQL: Bridging Natural Language and Corpus Search Engine](https://arxiv.org/abs/2402.13740) | 本文提出了首个旨在自动将自然语言转换为语料库查询语言（CQL）的文本到CQL任务，包括一个大规模数据集和利用大语言模型（LLMs）进行有效文本到CQL任务的方法。 |
| [^31] | [The Da Vinci Code of Large Pre-trained Language Models: Deciphering Degenerate Knowledge Neurons](https://arxiv.org/abs/2402.13731) | 本研究提供了对预训练语言模型中退化知识神经元（DKNs）的全面定义，引入了神经拓扑聚类方法和神经退化分析框架，从而实现更准确的DKN获取。 |
| [^32] | [Exploiting Adaptive Contextual Masking for Aspect-Based Sentiment Analysis](https://arxiv.org/abs/2402.13722) | 提出了一种利用自适应掩码方法来协助面向方面的情感分析中的方面术语提取和情感分类子任务的新方法 |
| [^33] | [Ouroboros: Speculative Decoding with Large Model Enhanced Drafting](https://arxiv.org/abs/2402.13720) | Ouroboros通过构建短小草案并引入候选短语池的方法提高了大语言模型推理的加速效率 |
| [^34] | [$\infty$Bench: Extending Long Context Evaluation Beyond 100K Tokens](https://arxiv.org/abs/2402.13718) | 提出了$\infty$Bench，第一个以平均数据长度超过10万个令牌的LLM基准，用于评估处理长上下文的能力 |
| [^35] | [Neeko: Leveraging Dynamic LoRA for Efficient Multi-Character Role-Playing Agent](https://arxiv.org/abs/2402.13717) | Neeko利用动态低秩适配器（LoRA）策略，有效处理多角色扮演过程中的挑战，提升了对不同属性、个性和说话模式的适应能力。 |
| [^36] | [SaGE: Evaluating Moral Consistency in Large Language Models](https://arxiv.org/abs/2402.13709) | 提出SaGE方法，通过语义图熵来衡量大型语言模型道德一致性，构建了MCC语料库。 |
| [^37] | [Investigating Multilingual Instruction-Tuning: Do Polyglot Models Demand for Multilingual Instructions?](https://arxiv.org/abs/2402.13703) | 本研究是第一个对多语模型在不同印欧语言上的性能进行了广泛研究，发现在并行教学调整数据集上进行教学调整可以显著提升跨语言遵循能力，同时提出了对表面对齐假设的质疑 |
| [^38] | [CMNER: A Chinese Multimodal NER Dataset based on Social Media](https://arxiv.org/abs/2402.13693) | 本研究在中国最大的社交媒体平台微博上构建了一个中文多模态实体识别数据集（CMNER），包含5,000条微博帖子和18,326张对应图片，并展示了将图片纳入NER任务中的有效性。 |
| [^39] | [KInIT at SemEval-2024 Task 8: Fine-tuned LLMs for Multilingual Machine-Generated Text Detection](https://arxiv.org/abs/2402.13671) | 本论文针对SemEval-2024任务8提出了一种使用微调LLMs进行多语言机器生成文本检测的方法，通过多种方式处理该任务并将统计检测指标与模型预测相结合，取得了竞争性结果。 |
| [^40] | [Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning](https://arxiv.org/abs/2402.13669) | SDFT是一种通过用模型本身生成的精简数据集来桥接分布差距的新方法，在缓解灾难性遗忘的同时，在下游任务上实现了与普通微调相媲美甚至更优越的性能 |
| [^41] | [GCOF: Self-iterative Text Generation for Copywriting Using Large Language Model](https://arxiv.org/abs/2402.13667) | GCOF框架结合遗传算法和大型语言模型，实现了自我迭代优化，生成的文案在点击率上相比人工编辑的文案平均提高了50%以上。 |
| [^42] | [Privacy-Preserving Instructions for Aligning Large Language Models](https://arxiv.org/abs/2402.13659) | 提出使用合成指南替换真实指南以增强隐私保护，并通过私密微调生成器生成此类合成指南，并通过新颖的过滤算法使合成指南的分布与真实指南一致，展示了在大型语言模型对齐中的高效用性。 |
| [^43] | [Unsupervised Text Style Transfer via LLMs and Attention Masking with Multi-way Interactions](https://arxiv.org/abs/2402.13647) | 通过组合注意力遮罩方法和大型语言模型，提出多种交互方式，可以改进无监督文本风格转移任务。 |
| [^44] | [A Unified Framework and Dataset for Assessing Gender Bias in Vision-Language Models](https://arxiv.org/abs/2402.13636) | 构建了一个统一框架和数据集，用于评估视觉-语言模型中的性别偏见，并观察到不同的输入-输出模式导致不同的偏见大小和方向。 |
| [^45] | [MORE: Multi-mOdal REtrieval Augmented Generative Commonsense Reasoning](https://arxiv.org/abs/2402.13625) | 提出了一种新颖的多模态检索（MORE）增强框架，利用文本和图像来提升语言模型的常识能力。 |
| [^46] | [FLAME: Self-Supervised Low-Resource Taxonomy Expansion using Large Language Models](https://arxiv.org/abs/2402.13623) | 提出了一种使用大型语言模型进行自监督低资源分类扩展的方法 FLAME |
| [^47] | [Overview of the VLSP 2023 -- ComOM Shared Task: A Data Challenge for Comparative Opinion Mining from Vietnamese Product Reviews](https://arxiv.org/abs/2402.13613) | 该论文总结了VLSP 2023中ComOM任务的一个数据挑战，旨在推动自然语言处理领域通过开发从越南产品评论中提取比较意见的技术，参与者需提出能够提取比较"五元组"的模型并根据F1分数进行评估排名。 |
| [^48] | [Data-driven Discovery with Large Generative Models](https://arxiv.org/abs/2402.13610) | 大型生成模型在数据驱动发现中的应用开创了端到端发现系统的新模式，利用提供的数据集搜寻和验证假设，突显了自动化系统的重要性和局限性。 |
| [^49] | [A Comprehensive Study of Multilingual Confidence Estimation on Large Language Models](https://arxiv.org/abs/2402.13606) | 该论文介绍了对大型语言模型的多语言置信度评估的全面研究，提出了一个专业多语言问答数据集，并研究了这些置信度分数如何增强模型性能，最终提出了一种跨语言置信度估计方法。 |
| [^50] | [KorNAT: LLM Alignment Benchmark for Korean Social Values and Common Knowledge](https://arxiv.org/abs/2402.13605) | KorNAT是首个用于评估韩国国家对齐的基准，包括社会价值观和常识对齐两个方面，通过对社会价值和常识多项选择题的测试来评估模型的对齐程度。 |
| [^51] | [Breaking the HISCO Barrier: Automatic Occupational Standardization with OccCANINE](https://arxiv.org/abs/2402.13604) | 通过OccCANINE工具，我们成功打破了HISCO障碍，实现了自动化职业标准化，从而大大简化了对职业描述的处理和分类过程，为经济学、经济历史等领域的职业结构分析提供了高效且准确的数据。 |
| [^52] | [User-LLM: Efficient LLM Contextualization with User Embeddings](https://arxiv.org/abs/2402.13598) | User-LLM框架利用用户嵌入对LLMs进行语境化，使其能够动态适应用户上下文，在各种任务中实现显著性能提升。 |
| [^53] | [Knowledge Graph Enhanced Large Language Model Editing](https://arxiv.org/abs/2402.13593) | 提出了一种利用知识图谱增强大型语言模型编辑的方法GLAME，能够解决编辑时知识变化合并困难的问题，提高编辑后模型的泛化能力。 |
| [^54] | [A Multimodal In-Context Tuning Approach for E-Commerce Product Description Generation](https://arxiv.org/abs/2402.13587) | 提出了一种用于电子商务产品描述生成的多模态上下文调整方法ModICT，通过引入相似产品样本和利用语言模型的上下文学习能力，旨在解决生成描述中常见且忽略产品特征的问题 |
| [^55] | [WinoViz: Probing Visual Properties of Objects Under Different States](https://arxiv.org/abs/2402.13584) | 该研究提出了WinoViz评估数据集，探究语言模型对对象在不同状态下的视觉属性的推理能力，该任务具有挑战性，要求实用推理和视觉知识推理。 |
| [^56] | [LongWanjuan: Towards Systematic Measurement for Long Text Quality](https://arxiv.org/abs/2402.13583) | 本研究针对长文本评估的差距，引入了一套基于连贯性、凝聚力和复杂性等语言学维度的指标来系统性衡量长文本的质量，并提出了LongWanjuan数据集，有助于提升长文本任务的语言模型训练。 |
| [^57] | [BBA: Bi-Modal Behavioral Alignment for Reasoning with Large Vision-Language Models](https://arxiv.org/abs/2402.13577) | 本研究提出了双模态行为对齐（BBA）提示方法，旨在最大化DSL在增强复杂多模态推理任务方面的潜力。 |
| [^58] | [Multilingual Coreference Resolution in Low-resource South Asian Languages](https://arxiv.org/abs/2402.13571) | 引入了一个用于31种南亚语言的多语言共指解析翻译数据集，通过利用现成工具进行训练和对齐，在低资源条件下实现了较好的共指解析模型性能提升。 |
| [^59] | [Analysis of Multi-Source Language Training in Cross-Lingual Transfer](https://arxiv.org/abs/2402.13562) | 多源语言训练（MSLT）技术通过使用多个源语言，在跨语言转移中增加了不同语言嵌入空间的交织，从而支持了XLT受益于这种方法的说法。 |
| [^60] | [Cognitive Visual-Language Mapper: Advancing Multimodal Comprehension with Enhanced Visual Knowledge Alignment](https://arxiv.org/abs/2402.13561) | 该论文提出了一种认知视觉语言映射器（CVLM），通过增强视觉知识对齐，在多模态理解中取得了重要进展，特别是在挑战知识型视觉问题回答方面。 |
| [^61] | [Graph Representation of Narrative Context: Coherence Dependency via Retrospective Questions](https://arxiv.org/abs/2402.13551) | 提出了一种新颖且实用的叙事理解范式，通过在叙事中形成图NARCO来描述整个背景的任务无关的连贯依赖，其中的边反映了高层次的连贯关系，无需依赖人类注释。 |
| [^62] | [Are LLMs Effective Negotiators? Systematic Evaluation of the Multifaceted Capabilities of LLMs in Negotiation Dialogues](https://arxiv.org/abs/2402.13550) | 本研究系统评估了LLMs在谈判对话中的多方面能力，揭示了它们在谈判研究中的潜力和局限。 |
| [^63] | [ActiveRAG: Revealing the Treasures of Knowledge via Active Learning](https://arxiv.org/abs/2402.13547) | ActiveRAG是一个创新的RAG框架，通过引入主动学习机制，利用知识构建和认知联结机制来提升大型语言模型（LLMs）的内在认知，实现了明显的性能提升。 |
| [^64] | [LLMs Meet Long Video: Advancing Long Video Comprehension with An Interactive Visual Adapter in LLMs](https://arxiv.org/abs/2402.13546) | 介绍了一个交互式视觉适配器（IVA），用于在LLMs中增强对细粒度视觉元素的交互，并解决了长视频理解中的计算成本高、视觉清晰度降低和无关视觉令牌带来的挑战。 |
| [^65] | [ARL2: Aligning Retrievers for Black-box Large Language Models via Self-guided Adaptive Relevance Labeling](https://arxiv.org/abs/2402.13542) | ARL2提出了一种检索器学习技术，利用LLMs作为标注者，并采用自适应自训练策略，能够有效减少注释成本，并在NQ和MMLU上取得了5.4%和4.6%的准确度提升。 |
| [^66] | [An Effective Incorporating Heterogeneous Knowledge Curriculum Learning for Sequence Labeling](https://arxiv.org/abs/2402.13534) | 提出了一个专为序列标注任务设计的两阶段课程学习（TCL）框架，逐渐引入数据实例从简单到困难，旨在提高性能和训练速度，并且对六个中文分词和词性标注数据集进行了广泛实验，证明了模型的有效性。 |
| [^67] | [FinGPT-HPC: Efficient Pretraining and Finetuning Large Language Models for Financial Applications with High-Performance Computing](https://arxiv.org/abs/2402.13533) | 该论文提出了一种基于高性能GPU的方法，利用低秩结构来高效地预训练和微调大型语言模型，解决了线性层冗余性、GPU内存占用和分布式训练中GPU利用率不足的挑战 |
| [^68] | [Backdoor Attacks on Dense Passage Retrievers for Disseminating Misinformation](https://arxiv.org/abs/2402.13532) | 本文介绍了一种后门攻击场景，攻击者通过利用密集通道检索的语法错误触发后门攻击，以秘密传播定向错误信息，如仇恨言论或广告，并通过实验证明了这种攻击方法的有效性和隐匿性。 |
| [^69] | [Infrastructure Ombudsman: Mining Future Failure Concerns from Structural Disaster Response](https://arxiv.org/abs/2402.13528) | 本文开发了一种基础设施调解员系统，用于自动检测特定基础设施问题，通过挖掘社交网络中关于预期失败的担忧，有助于预防和减轻潜在的基础设施失败。 |
| [^70] | [OMGEval: An Open Multilingual Generative Evaluation Benchmark for Large Language Models](https://arxiv.org/abs/2402.13524) | OMGEval是第一个可以评估大型语言模型在不同语言中能力的开放源多语言生成测试集，涵盖了广泛重要能力并进行了本地化处理。 |
| [^71] | [RecMind: Japanese Movie Recommendation Dialogue with Seeker's Internal State](https://arxiv.org/abs/2402.13522) | RecMind是一个具有寻求者内在状态注释的日本电影推荐对话数据集，研究发现对那些寻求者感兴趣但并不了解的实体进行推荐有助于成功推荐，并提出了一个考虑寻求者内在状态的响应生成框架。 |
| [^72] | [RITFIS: Robust input testing framework for LLMs-based intelligent software](https://arxiv.org/abs/2402.13518) | RITFIS是第一个设计用于评估基于LLM的智能软件对自然语言输入鲁棒性的框架，通过将测试过程定义为组合优化问题来确定成功的测试案例。 |
| [^73] | [Round Trip Translation Defence against Large Language Model Jailbreaking Attacks](https://arxiv.org/abs/2402.13517) | 往返翻译（RTT）方法是第一个专门设计用于抵御大型语言模型（LLMs）社交工程攻击的算法，成功地减少了多种攻击形式的成功率。 |
| [^74] | [ProSparse: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models](https://arxiv.org/abs/2402.13516) | 本文介绍了一种名为"ProSparse"的有效稀疏化方法，以推动大型语言模型实现更高的激活稀疏性而不降低模型性能 |
| [^75] | [Self-DC: When to retrieve and When to generate? Self Divide-and-Conquer for Compositional Unknown Questions](https://arxiv.org/abs/2402.13514) | 提出了面向组合未知问题的自我分而治之算法，引入了第一个组合未知问题问答数据集（CuQA），通过自适应调用不同方法实现更好的性能和效率。 |
| [^76] | [From Self-Attention to Markov Models: Unveiling the Dynamics of Generative Transformers](https://arxiv.org/abs/2402.13512) | 本文研究了从自注意力模型到马尔可夫模型的转变，揭示了生成Transformer动态的机理和相关条件，为一致估计提供了保证，并在IID样本下建立了样本复杂性保证。 |
| [^77] | [Leveraging Translation For Optimal Recall: Tailoring LLM Personalization With User Profiles](https://arxiv.org/abs/2402.13500) | 本研究提出了一种通过多级翻译、语义嵌入扩展和用户配置文件中心扩展相结合的方法，旨在在跨语言信息检索系统中改善召回率，通过个性化匹配用户查询和相关文档，展示了比基线方法更优异的性能。 |
| [^78] | [The Lay Person's Guide to Biomedicine: Orchestrating Large Language Models](https://arxiv.org/abs/2402.13498) | 利用大型语言模型生成和评估生物医学文章的平民总结，提出了Explain-then-Summarise的新LS框架，并评估了LLMs在零-shot LS方面的表现和提出了两种新的LLM-based LS评估方法。 |
| [^79] | [GradSafe: Detecting Unsafe Prompts for LLMs via Safety-Critical Gradient Analysis](https://arxiv.org/abs/2402.13494) | GradSafe通过分析LLMs中关键安全参数的梯度，有效检测不安全提示，无需额外训练即可优于现有方法。 |
| [^80] | [Retrieval Helps or Hurts? A Deeper Dive into the Efficacy of Retrieval Augmentation to Language Models](https://arxiv.org/abs/2402.13492) | 该研究深入探讨了如何通过检索增强语言模型，构建了新的QA数据集WiTQA，以实体和关系组合的影响为重点进行了详细分析。 |
| [^81] | [ProPD: Dynamic Token Tree Pruning and Generation for LLM Parallel Decoding](https://arxiv.org/abs/2402.13485) | 提出ProPD，一种基于动态令牌树修剪和生成的高效LLM并行解码框架，通过先进的提前修剪机制和动态令牌树生成算法来提高验证效率。 |
| [^82] | [Retrieval-Augmented Data Augmentation for Low-Resource Domain Tasks](https://arxiv.org/abs/2402.13482) | 提出了一种用于低资源领域任务的新方法，通过结合来自其他数据集的相关示例来增强训练数据，以解决在低资源环境中生成样本不够理想和缺乏多样性的挑战 |
| [^83] | [How Important is Domain Specificity in Language Models and Instruction Finetuning for Biomedical Relation Extraction?](https://arxiv.org/abs/2402.13470) | 研究探讨了在生物医学关系提取任务中领域特异性对于语言模型和指导微调的重要性，对比了在生物医学领域与通用领域训练的模型效果，并探讨了在生物医学数据集上指导微调的模型在性能上的优势。 |
| [^84] | [STENCIL: Submodular Mutual Information Based Weak Supervision for Cold-Start Active Learning](https://arxiv.org/abs/2402.13468) | STENCIL利用次模互信息选择弱标记的稀有类实例，并通过标注者强标记，提高了文本分类数据集上的准确率和稀有类F-1分数。 |
| [^85] | [RefuteBench: Evaluating Refuting Instruction-Following for Large Language Models](https://arxiv.org/abs/2402.13463) | 本文提出了一个名为RefuteBench的基准测试，旨在评估大型语言模型对反驳指令的遵循能力，发现LLMs倾向于固执于其内部知识而无法遵从用户反馈。 |
| [^86] | [Potential and Challenges of Model Editing for Social Debiasing](https://arxiv.org/abs/2402.13462) | 模型编辑方法在社交去偏见中具有潜力，但也面临挑战，尤其是在支持不同偏见类型和理解编辑方法应用于去偏见过程中的利弊方面。 |
| [^87] | [Learning to Poison Large Language Models During Instruction Tuning](https://arxiv.org/abs/2402.13459) | 通过设计新的数据注入攻击攻击LLMs，并提出一种梯度引导后门触发器学习方法，通过实验验证表明成功地破坏模型输出，仅改变1%的指导调优样本即可导致性能下降率达到约80％。 |
| [^88] | [LocalTweets to LocalHealth: A Mental Health Surveillance Framework Based on Twitter Data](https://arxiv.org/abs/2402.13452) | 本研究提出了一个新的基于Twitter数据的框架LocalHealth，用于预测当地精神健康结果。通过与GPT3.5结合使用，该框架在MH监测中取得了显著的改进。 |
| [^89] | [CAMELoT: Towards Large Language Models with Training-Free Consolidated Associative Memory](https://arxiv.org/abs/2402.13449) | 引入了一个关联内存模块，可以无需重新训练即可与任何预先训练的大型语言模型耦合，解决了长输入序列处理问题，并在长上下文建模中显著降低困惑度。 |
| [^90] | [ED-Copilot: Reduce Emergency Department Wait Time with Language Model Diagnostic Assistance](https://arxiv.org/abs/2402.13448) | 本研究提出了一种在急诊科中减少等待时间的诊断辅助方法，利用人工智能系统帮助医生进行快速准确的诊断，并开发了ED-Copilot系统来推荐实验室检测并进行诊断预测。 |
| [^91] | [Large Language Models for Data Annotation: A Survey](https://arxiv.org/abs/2402.13446) | 大型语言模型的出现为自动化数据标注提供机遇，该调查独特关注LLM在数据标注中的效用，贡献主要集中在LLM-Based数据标注、评估LLM生成的标注以及使用LLM生成的标注学习等三个核心方面。 |
| [^92] | [Structured Tree Alignment for Evaluation of (Speech) Constituency Parsing](https://arxiv.org/abs/2402.13433) | 提出了一种受语音解析器评估问题启发的结构化句法分析树相似性度量指标STRUCT-IOU，有效地比较了口语词边界上的组块分析树与书面词上基准解析之间的差异，并展示了在文本组块分析评估中的优越性。 |
| [^93] | [DrBenchmark: A Large Language Understanding Evaluation Benchmark for French Biomedical Domain](https://arxiv.org/abs/2402.13432) | DrBenchmark提出了一个针对法语生物医学领域的大型语言理解评估基准，旨在弥补对最新法语生物医学模型评估的不足，并考虑到法语的独特敏感性。 |
| [^94] | [Explaining Relationships Among Research Papers](https://arxiv.org/abs/2402.13426) | 探索了一种基于特征的LLM提示方法，用于生成丰富的引文文本，并一次生成多个引文以捕捉研究论文之间的复杂关系。 |
| [^95] | [Structure Guided Prompt: Instructing Large Language Model in Multi-Step Reasoning by Exploring Graph Structure of the Text](https://arxiv.org/abs/2402.13415) | 本论文介绍了一种结构引导提示框架，旨在通过探索文本的图结构，指导大型语言模型进行多步推理，以解决推理过程中的复杂关系和多样性带来的困难。 |
| [^96] | [Harnessing Large Language Models as Post-hoc Correctors](https://arxiv.org/abs/2402.13414) | 通过提出的无需训练的框架 LlmCorr，本文展示了一个LLM可以作为事后校正器，为任意ML模型的预测提出修正。 |
| [^97] | [Healthcare Copilot: Eliciting the Power of General LLMs for Medical Consultation](https://arxiv.org/abs/2402.13408) | 该论文介绍了一种旨在增强和定制大型语言模型（LLMs）以进行医疗咨询的Healthcare Copilot框架，其包括对话组件、记忆组件和处理组件。通过实施自动评估方案，结果表明该Healthcare Copilot能够显著改善医疗咨询质量。 |
| [^98] | [A Unified Taxonomy-Guided Instruction Tuning Framework for Entity Set Expansion and Taxonomy Expansion](https://arxiv.org/abs/2402.13405) | 通过统一的基于分类学指导的指导调整框架，本文提出了一种利用现有分类学进行实体关系微调的方法，有效解决实体集扩展、分类学扩展和种子引导分类学构建三个任务。 |
| [^99] | [Reliable LLM-based User Simulator for Task-Oriented Dialogue Systems](https://arxiv.org/abs/2402.13374) | 该论文介绍了DAUS，一个基于LLM的领域感知用户模拟器，通过在真实对话示例上进行微调，显著改进了用户目标实现，并有效减轻模拟器响应中的幻觉。 |
| [^100] | [EvoGrad: A Dynamic Take on the Winograd Schema Challenge with Human Adversaries](https://arxiv.org/abs/2402.13372) | EvoGrad是一个以人类对手为特点的用于解决Winograd Schema挑战的动态方法，通过人在环中方法创建动态数据集，拓展任务实例并引入错误深度度量标准，提出新的多样化常识推理数据集基准，揭示了当前语言模型在此类任务上的挑战。 |
| [^101] | [A Simple but Effective Approach to Improve Structured Language Model Output for Information Extraction](https://arxiv.org/abs/2402.13364) | 该论文提出了一种名为G&O的方法，通过将内容生成与结构化过程分离，有效提升了大型语言模型在生成特定结构化文本上的性能。 |
| [^102] | [PIRB: A Comprehensive Benchmark of Polish Dense and Hybrid Text Retrieval Methods](https://arxiv.org/abs/2402.13350) | PIRB提出了一个全面的波兰文本信息检索基准，包含41个任务，评估了超过20种密集和稀疏检索模型，并引入了一个三步训练流程来构建高效的特定语言检索器，最后验证了他们的方法的优越性 |
| [^103] | [Enhanced Hallucination Detection in Neural Machine Translation through Simple Detector Aggregation](https://arxiv.org/abs/2402.13331) | 提出了一种通过简单的探测器聚合来增强神经机器翻译中的幻觉检测，结果表明这种方法有效性，有望使机器翻译系统更加可靠。 |
| [^104] | [Enhancing Modern Supervised Word Sense Disambiguation Models by Semantic Lexical Resources](https://arxiv.org/abs/2402.13302) | 通过引入语义特征和多层架构，本研究提出了一种通过利用WordNet和WordNet Domains等语义词汇资源增强现代监督词义消歧模型的方法。 |
| [^105] | [Structure Guided Large Language Model for SQL Generation](https://arxiv.org/abs/2402.13284) | 通过引入结构信息，提出了一个结构引导的SQL生成模型，以改善大型语言模型生成SQL的准确性和可执行性。 |
| [^106] | [What if LLMs Have Different World Views: Simulating Alien Civilizations with LLM-based Agents](https://arxiv.org/abs/2402.13184) | 这项研究引入了“CosmoAgent”，利用LLM模拟人类和外星文明之间的复杂互动，评估和平共存的可行性，并量化评估文明的发展轨迹，同时考虑不同文明之间的巨大多样性。 |
| [^107] | [CMDAG: A Chinese Metaphor Dataset with Annotated Grounds as CoT for Boosting Metaphor Generation](https://arxiv.org/abs/2402.13145) | 本文介绍了一个大规模高质量的带注释中文隐喻语料库，强调隐喻生成中的基础及其独特特征，而非传统的对象和载体组合。 |
| [^108] | [Effective and Efficient Conversation Retrieval for Dialogue State Tracking with Implicit Text Summaries](https://arxiv.org/abs/2402.13043) | 使用文本摘要提高对话检索的有效性和效率，通过对话摘要生成器进行查询和关键词生成，进一步提炼轻量级对话编码器以避免额外推理成本 |
| [^109] | [FormulaQA: A Question Answering Dataset for Formula-Based Numerical Reasoning](https://arxiv.org/abs/2402.12692) | FormulaQA是一个基于初中物理考试的公式驱动数值推理问题问答数据集，通过评估LLMs的不同方法和使用检索增强型LLMs以及对小型模型进行微调，揭示了现有模型在应对复杂、基于公式的FormulaQA时的潜在改进空间。 |
| [^110] | [StyleDubber: Towards Multi-Scale Style Learning for Movie Dubbing](https://arxiv.org/abs/2402.12636) | StyleDubber提出了一种新的电影配音方法，通过在音素级别进行学习，解决了当前 V2C 模型中存在的音素发音不完整和身份稳定性差的问题。 |
| [^111] | [Turn Waste into Worth: Rectifying Top-$k$ Router of MoE](https://arxiv.org/abs/2402.12399) | 提出了Rectify-Router解决了MoE模型中常用的Top-k路由机制所带来的令牌丢失和填充问题，通过Intra-GPU矫正和Fill-in矫正来实现。 |
| [^112] | [Emulated Disalignment: Safety Alignment for Large Language Models May Backfire!](https://arxiv.org/abs/2402.12343) | 安全对齐的大型语言模型可能会通过模拟失调框架，在对抗性操纵下产生危险结果，对训练的语言模型具有双倍有害性，高于强基线，强调了即使在安全对齐后也需要重新评估开源语言模型的重要性。 |
| [^113] | [SciAgent: Tool-augmented Language Models for Scientific Reasoning](https://arxiv.org/abs/2402.11451) | 引入了工具增强型科学推理的新任务设置，通过提供可扩展的工具集，帮助大型语言模型在科学问题解决中变得更加实用和可解决。 |
| [^114] | [Reasoning before Comparison: LLM-Enhanced Semantic Similarity Metrics for Domain Specialized Text Analysis](https://arxiv.org/abs/2402.11398) | 通过利用LLM增强语义分析，开发了用于文本的相似度度量框架，可显著改善文本的语义相似性评估，并可扩展到其他专业领域。 |
| [^115] | [Understanding News Thumbnail Representativeness by Counterfactual Text-Guided Contrastive Language-Image Pretraining](https://arxiv.org/abs/2402.11159) | 提出了一种反事实文本引导的对比语言-图像预训练框架CFT-CLIP，用于增强新闻文本和缩略图之间的对比学习。 |
| [^116] | [In Search of Needles in a 10M Haystack: Recurrent Memory Finds What LLMs Miss](https://arxiv.org/abs/2402.10790) | 通过使用循环记忆增强对 GPT-2 进行微调，使其能够处理长达 1000 万个元素的任务，这是迄今为止处理最长输入的开放神经网络模型，并展示了对长序列处理能力的显著改进。 |
| [^117] | [InSaAF: Incorporating Safety through Accuracy and Fairness | Are LLMs ready for the Indian Legal Domain?](https://arxiv.org/abs/2402.10567) | 本研究在印度法律领域探讨了大型语言模型（LLMs）在处理社会因素时的能力，提出了结合公平性和准确性的新指标$LSS_{\beta}$，并评估了模型在二元法律推理任务中的表现以及在印度社会各种不平等方面的公平性展示。 |
| [^118] | [AI Hospital: Interactive Evaluation and Collaboration of LLMs as Intern Doctors for Clinical Diagnosis](https://arxiv.org/abs/2402.09742) | AI医院是一个框架，用于构建实时交互式诊断环境，通过与LLMs的交互评估和协作，提高临床诊断的准确性。 |
| [^119] | [Punctuation Restoration Improves Structure Understanding without Supervision](https://arxiv.org/abs/2402.08382) | 标点符号恢复是一个有效的学习目标，可以改善结构理解并提高模型性能。 |
| [^120] | [Addressing cognitive bias in medical language models](https://arxiv.org/abs/2402.08113) | 本研究通过开发BiasMedQA，一个用于评估医学任务中LLMs的认知偏见的新型基准，发现LLMs在面对包含认知偏见的临床问题时，其回答的准确性明显降低。 |
| [^121] | [Differentially Private Zeroth-Order Methods for Scalable Large Language Model Finetuning](https://arxiv.org/abs/2402.07818) | 本文研究了差分隐私零阶方法在大型语言模型微调中的应用，该方法通过使用零阶梯度来避免传统优化方法的可扩展性瓶颈，实现了在隐私、效用和可扩展性之间的良好平衡。 |
| [^122] | [DeAL: Decoding-time Alignment for Large Language Models](https://arxiv.org/abs/2402.06147) | DeAL是一个允许用户自定义奖励函数并实现解码时对齐LLMs的框架。 |
| [^123] | [Exploiting Class Probabilities for Black-box Sentence-level Attacks](https://arxiv.org/abs/2402.02695) | 该论文研究了在黑盒子句级攻击中利用类别概率的有效性，并开发了一种新的算法进行攻击。通过与基线方法进行对比，进行了广泛的评估。 |
| [^124] | [Bridging the Preference Gap between Retrievers and LLMs](https://arxiv.org/abs/2401.06954) | 本研究提出了一种新颖的桥梁机制，通过训练一个桥梁模型来优化检索器和LLM之间的连接，消除了在RAG中检索器和LLMs之间的偏好差距。 |
| [^125] | [Enhancing Emotional Generation Capability of Large Language Models via Emotional Chain-of-Thought](https://arxiv.org/abs/2401.06836) | 该研究提出了一种名为情感思维链（ECoT）的提示方法，通过与人类情感智慧准则对齐，增强大型语言模型在情感生成任务上的性能。 |
| [^126] | [Neural Machine Translation of Clinical Text: An Empirical Investigation into Multilingual Pre-Trained Language Models and Transfer-Learning](https://arxiv.org/abs/2312.07250) | 在临床文本的神经机器翻译研究中，通过使用多语言预训练语言模型和迁移学习方法，在ClinSpEn-2022英西临床领域数据上取得了顶级性能，并发现小型预训练语言模型在临床领域微调中胜过其他超大型语言模型。 |
| [^127] | [Tree of Attacks: Jailbreaking Black-Box LLMs Automatically](https://arxiv.org/abs/2312.02119) | 提出了一种名为Tree of Attacks with Pruning (TAP)的自动化方法，用于生成只需要对目标大型语言模型进行黑盒访问的越狱方法，并通过思维树推理和修剪生成准确的越狱提示。 |
| [^128] | [CAMRA: Copilot for AMR Annotation](https://arxiv.org/abs/2311.10928) | CAMRA是一个创新的基于web的工具，用于从自然语言文本构建抽象意义表示（AMR），通过整合编程语言的编码方法和AMR解析器模型作为副驾驶，极大提高了AMR注释的效率和准确性。 |
| [^129] | [Are You Sure? Challenging LLMs Leads to Performance Drops in The FlipFlop Experiment](https://arxiv.org/abs/2311.08596) | 本研究通过FlipFlop实验揭示了当挑战LLMs让其反思初始答案时，模型会平均有46%的概率改变答案，所有模型在第一次和最终预测之间表现出准确性下降的现象。 |
| [^130] | [Well begun is half done: Importance of Starting Right in Multi-Step Math Reasoning](https://arxiv.org/abs/2311.07945) | 较小的语言模型在多步骤数学推理中通过正确开始可以获得显着的性能提升，建议通过初始指导和自问指导的方式来引导模型开始正确。 |
| [^131] | [Coffee: Boost Your Code LLMs by Fixing Bugs with Feedback](https://arxiv.org/abs/2311.07215) | 开源代码LLMs难以生成正确指导的反馈，本研究提出了Coffee框架，旨在利用Coffee数据集构建CoffeePots，通过优化调整和选择，实现自动生成带有正确指导的反馈以用于代码修复。 |
| [^132] | [To Tell The Truth: Language of Deception and Language Models](https://arxiv.org/abs/2311.07092) | 在高风险环境中，研究人员通过分析电视游戏节目数据发现，即使只使用语言线索，基于大型语言模型构建的模型可以与人类主体具有类似的真相检测性能。 |
| [^133] | [Evaluating Gender Bias of Pre-trained Language Models in Natural Language Inference by Considering All Labels](https://arxiv.org/abs/2309.09697) | 提出了一种考虑自然语言推理任务三个标签的预训练语言模型偏见评估方法，通过创造代表不同类型偏见的评估数据组，并实验证明该方法能更好地区分有偏见的、不正确的推理和非有偏见的不正确推理。 |
| [^134] | [Adapting Large Language Models via Reading Comprehension](https://arxiv.org/abs/2309.09530) | 通过将原始语料库转化为阅读理解文本来调整大型语言模型，使其在多个领域的各种任务中性能始终得到提升。 |
| [^135] | [A Survey on Fairness in Large Language Models](https://arxiv.org/abs/2308.10149) | 本文审查了关于大型语言模型中公平性的研究，针对中等规模LLMs和大规模LLMs提出了评估指标和去偏见方法。 |
| [^136] | [mCL-NER: Cross-Lingual Named Entity Recognition via Multi-view Contrastive Learning](https://arxiv.org/abs/2308.09073) | 本文提出了一种通过多视角对比学习实现跨语言命名实体识别的方法，通过识别令牌对关系并利用上下文细微差别来统一不同语言的表示。 |
| [^137] | [TESS: Text-to-Text Self-Conditioned Simplex Diffusion](https://arxiv.org/abs/2305.08379) | TESS是一个全非自回归的文本扩散模型，通过在逻辑空间而不是学习嵌入空间应用扩散过程，进行了自条件单纯形扩散，实验证明在自然语言理解和生成任务中表现优于最先进的非自回归模型，并且所需的扩散步骤更少。 |
| [^138] | [InPars-Light: Cost-Effective Unsupervised Training of Efficient Rankers](https://arxiv.org/abs/2301.02998) | InPars-Light是一个简单而有效的修改，通过使用小得多的排名模型和免费语言模型BLOOM，在多个英文检索集合上显著改进了排名性能。 |
| [^139] | [Understanding Multimodal Procedural Knowledge by Sequencing Multimodal Instructional Manuals](https://arxiv.org/abs/2110.08486) | 本研究通过整理数据集并收集全面的人类注释，对机器学习模型在推理和排序无序的多模态指导方面的能力进行基准测试，发现模型表现不仅显著低于人类，而且似乎无法具备这种基本能力。 |
| [^140] | [CFMatch: Aligning Automated Answer Equivalence Evaluation with Expert Judgments For Open-Domain Question Answering.](http://arxiv.org/abs/2401.13170) | CFMatch提出了一个在开放域问答中将自动答案等价评估与人工专家判断对齐的方法，通过提供明确一致的评估指南并引入高效、稳健且轻量级的判别式AE分类器匹配方法来解决当前评估指标与人类判断不一致的问题。 |
| [^141] | [OOP: Object-Oriented Programming Evaluation Benchmark for Large Language Models.](http://arxiv.org/abs/2401.06628) | 本研究提出了一种面向对象编程的新型评估基准，包括431个Python程序，采用pass@o度量指标来提供更全面和相关的OOP代码生成评估。评估结果显示代码专用LLMs在OOP方面表现较差，需进一步改进此领域。 |
| [^142] | [Intention Analysis Prompting Makes Large Language Models A Good Jailbreak Defender.](http://arxiv.org/abs/2401.06561) | 本研究提出了一种名为Intention Analysis Prompting (IAPrompt)的方法，通过触发大型语言模型（LLMs）的自我纠正和改进能力来防御越狱攻击。实验证明，该方法能够显著减少响应中的有害行为并保持整体有用性。 |
| [^143] | [LEGOBench: Leaderboard Generation Benchmark for Scientific Models.](http://arxiv.org/abs/2401.06233) | LEGOBench是一个评估生成科学模型排行榜系统的基准测试，使用22年来的论文预印本数据和PapersWithCode门户上的机器学习排行榜的数据，初步结果显示自动排行榜生成存在显著性能差距。 |
| [^144] | [SH2: Self-Highlighted Hesitation Helps You Decode More Truthfully.](http://arxiv.org/abs/2401.05930) | 自我突出式犹豫（SH2）是一种推理时的方法，通过选择预测概率较低的标记，并强调它们的差异，从而帮助语言模型更准确地解码。 |
| [^145] | [ANGO: A Next-Level Evaluation Benchmark For Generation-Oriented Language Models In Chinese Domain.](http://arxiv.org/abs/2401.04898) | ANGO是一个中文领域生成型语言模型评估基准，引入了关键点分类标准，提供了更好的可解释性，同时建立了可量化的问题难度标准，对模型训练提供了更精确的指导。 |
| [^146] | [Explore Spurious Correlations at the Concept Level in Language Models for Text Classification.](http://arxiv.org/abs/2311.08648) | 本文研究了语言模型在文本分类中概念级别的误相关性问题，并通过使用ChatGPT分配概念标签和引入数据再平衡技术来解决这一问题。 |
| [^147] | [A Study of Continual Learning Under Language Shift.](http://arxiv.org/abs/2311.01200) | 本文研究了持续学习在语言转换中的应用，发现在更新语言模型时，前向转移效果较好且与语言顺序无关，但后向转移效果可能取决于新语言的顺序和特征。 |
| [^148] | [InfoEntropy Loss to Mitigate Bias of Learning Difficulties for Generative Language Models.](http://arxiv.org/abs/2310.19531) | 提出了一种信息熵损失函数，用于减少生成式语言模型对常见和易学标记的偏好，使其更关注不常见和难学的标记。 |
| [^149] | [Discrete Diffusion Language Modeling by Estimating the Ratios of the Data Distribution.](http://arxiv.org/abs/2310.16834) | 本研究通过引入得分熵这一新颖的离散得分匹配损失，弥补了离散数据领域中现有方法的不足，提出了得分熵离散扩散模型(SEDD)并在GPT-2实验中取得了有竞争力的效果。 |
| [^150] | [QLLM: Accurate and Efficient Low-Bitwidth Quantization for Large Language Models.](http://arxiv.org/abs/2310.08041) | QLLM是一种为大规模语言模型设计的准确高效的低位宽后训练量化方法，通过引入自适应通道重组技术，将离群值的大小重新分配给其他通道，从而减轻它们对量化范围的影响。 |
| [^151] | [GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction.](http://arxiv.org/abs/2310.03668) | GoLLIE 是一个遵循注释指南的大型语言模型，通过微调以改进未见信息抽取任务的零样本结果。 |
| [^152] | [Scaling Laws for Associative Memories.](http://arxiv.org/abs/2310.02984) | 本文研究了应用于联想记忆中的缩放定律，通过高维矩阵和嵌入的外积来模拟内层Transformer语言模型。作者推导出了与样本数量和参数大小相关的精确缩放定律，并验证了理论结果的有效性。同时，作者还通过大量实验展示了存储记忆关联的细粒度可视化。 |
| [^153] | [The Entity-Deduction Arena: A playground for probing the conversational reasoning and planning capabilities of LLMs.](http://arxiv.org/abs/2310.01468) | 本文提供了一个评估框架，通过向法官提出一系列查询来评估LLMs的对话推理和规划能力。我们发现不同的LLMs在这个任务上表现出显著差异。 |
| [^154] | [Understanding In-Context Learning from Repetitions.](http://arxiv.org/abs/2310.00297) | 本论文通过研究表面重复现象的角度来探索大型语言模型中上下文学习的机制，并实证了一种增强标记关系的原则，为理解上下文学习及其潜在局限性做出了重要贡献。 |
| [^155] | [ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving.](http://arxiv.org/abs/2309.17452) | ToRA是一种集成工具的数学问题求解推理代理，通过结合语言的分析能力和工具的计算效率，能够显著提高数学推理的性能，在多个数学推理数据集上取得了13%-19%的平均绝对改进率，并在竞赛级数据集MATH上达到了44.6%的性能。 |
| [^156] | [Enhancing Large Language Models in Coding Through Multi-Perspective Self-Consistency.](http://arxiv.org/abs/2309.17272) | 本文提出了一个名为多角度自一致性（MPSC）的框架，用于提升大规模语言模型在复杂的代码生成任务中的性能。该框架通过从多个角度采样多个输出并构建一个多部分图，利用交叉一致性和内一致性信息来选择最优输出。 |
| [^157] | [EchoPrompt: Instructing the Model to Rephrase Queries for Improved In-context Learning.](http://arxiv.org/abs/2309.10687) | EchoPrompt是一种简单而有效的方法，通过促使模型重新表述查询来提供改进的上下文学习效果。实验证明，EchoPrompt在多个任务中都取得了显著的性能提升。 |
| [^158] | [PoSE: Efficient Context Window Extension of LLMs via Positional Skip-wise Training.](http://arxiv.org/abs/2309.10400) | 本文介绍了一种名为PoSE的训练方法，通过在训练过程中使用固定的上下文窗口和操纵位置索引来适应极长的上下文窗口，实验证明这种方法大大减小了内存和时间开销，对性能影响较小，成功将LLaMA模型扩展到了128k个标记。 |
| [^159] | [LASER: LLM Agent with State-Space Exploration for Web Navigation.](http://arxiv.org/abs/2309.08172) | 本论文提出了一种基于状态空间探索的LLM代理（LASER）用于Web导航任务。该代理以灵活的方式转换状态，通过执行动作完成任务，能够轻松从错误中恢复，并取得了显著的性能提升。 |
| [^160] | [Cited Text Spans for Citation Text Generation.](http://arxiv.org/abs/2309.06365) | 本文提出了一种弥合引用和引文文本之间距离的方法，通过使用引文文本跨度(CTS)替代摘要作为输入，从而使得引文生成更加准确和相关。通过自动标注和基于关键词的检索方法，可以实现高效的CTS标注，提高引文文本生成的效果。 |
| [^161] | [SpikeBERT: A Language Spikformer Trained with Two-Stage Knowledge Distillation from BERT.](http://arxiv.org/abs/2308.15122) | 该论文提出了一种名为SpikeBERT的SNN模型，通过改进Spikformer架构和使用两阶段知识蒸馏方法，该模型在语言任务上超越了其他SNN模型，在文本分类任务上甚至达到了与BERT相当的结果。 |
| [^162] | [Empowering Clinicians and Democratizing Data Science: Large Language Models Automate Machine Learning for Clinical Studies.](http://arxiv.org/abs/2308.14120) | chatGPT ADA是一种能够自主开发临床研究所需的最先进的机器学习模型的大型语言模型，可将高级分析工具民主化，使非数据科学家的临床医生能够轻松应用于医学领域。 |
| [^163] | [CausalLM is not optimal for in-context learning.](http://arxiv.org/abs/2308.06912) | 最近的研究显示，上下文学习中使用前缀语言模型（PrefixLM）比因果语言模型（CausalLM）效果更好。本文通过理论分析证明，虽然两种语言模型都以线性速率收敛到稳定点，但前缀语言模型收敛到线性回归的最优解，因果语言模型的收敛动态遵循在线梯度下降算法，不保证收敛到最优解。 |
| [^164] | [Optimizing Machine Translation through Prompt Engineering: An Investigation into ChatGPT's Customizability.](http://arxiv.org/abs/2308.01391) | 本文研究了通过在ChatGPT中运用合适的提示将翻译目的和目标受众融入进去对翻译质量的影响。研究发现，这种方法可以产生灵活的翻译结果，相比传统机器翻译更具定制性。 |
| [^165] | [AutoML in the Age of Large Language Models: Current Challenges, Future Opportunities and Risks.](http://arxiv.org/abs/2306.08107) | 论文探讨了AutoML和LLMs之间的共生关系，并指出这两个领域的融合有望颠覆NLP和AutoML两个领域，同时也存在风险。 |
| [^166] | [STAR: Improving Low-Resource Information Extraction by Structure-to-Text Data Generation with Large Language Models.](http://arxiv.org/abs/2305.15090) | STAR是一种利用大型语言模型合成数据实例的数据生成方法，用于改进低资源信息抽取，为实际应用提供了需要最少人工标注的解决方案。 |
| [^167] | [LogicLLM: Exploring Self-supervised Logic-enhanced Training for Large Language Models.](http://arxiv.org/abs/2305.13718) | 本文介绍了 LogicLLM，一种通过自监督后训练来提高大语言模型的逻辑推理能力的方法，该方法有效地在常见逻辑推理任务上进行表现，超过了目前最先进的无监督基线方法。 |
| [^168] | [Chain-of-Knowledge: Grounding Large Language Models via Dynamic Knowledge Adapting over Heterogeneous Sources.](http://arxiv.org/abs/2305.13269) | Chain-of-Knowledge通过整合多源动态知识为大型语言模型提供准确的基础信息，减少生成的幻觉，可以产生更可靠的答案。 |
| [^169] | [CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing.](http://arxiv.org/abs/2305.11738) | 本文提出了一个名为CRITIC的框架，使得大型语言模型可以通过与工具的交互校正自己的错误，从而避免生成出现不一致和问题行为的结果。 |
| [^170] | [InstructIE: A Chinese Instruction-based Information Extraction Dataset.](http://arxiv.org/abs/2305.11527) | 介绍了一份中文的基于指令的信息提取数据集InstructIE，其中包括了270,000个弱监督的数据和1,000个高质量注释实例。实验结果表明当前的模型表现有待改进，该任务仍存在挑战。 |
| [^171] | [Data Redaction from Conditional Generative Models.](http://arxiv.org/abs/2305.11351) | 本文研究如何对已训练好的条件生成模型进行后期编辑，以便编辑掉某些条件分支，这些条件分支很可能会生成不良内容。通过精简模型中的条件网络实现，提出的解决方案有效、高效、具有可控性和普适性，在文本到图像和文本到语音生成模型中取得了良好效果。 |
| [^172] | [Error Analysis Prompting Enables Human-Like Translation Evaluation in Large Language Models: A Case Study on ChatGPT.](http://arxiv.org/abs/2303.13809) | 本文提出一种新的提示方法Error Analysis Prompting可改善LLMs在机器翻译质量评估上的性能，实现人类水平的评估。 |
| [^173] | [Extracting Accurate Materials Data from Research Papers with Conversational Language Models and Prompt Engineering.](http://arxiv.org/abs/2303.05352) | 本论文提出了ChatExtract方法，使用先进的对话式语言模型和提示工程，自动从研究论文中提取准确的数据，不需要大量的前期努力和背景知识。 |
| [^174] | [Unsupervised Layer-wise Score Aggregation for Textual OOD Detection.](http://arxiv.org/abs/2302.09852) | 提出了一种无监督的逐层聚合异常得分的方法，用于更好地进行文本OOD检测。其能发掘不同层输出的优势，达到更鲁棒的性能，并扩展经典基准测试以反映更现实的设置。 |

# 详细

[^1]: 迫使LLMs执行并揭示（几乎）任何事情

    Coercing LLMs to do and reveal (almost) anything

    [https://arxiv.org/abs/2402.14020](https://arxiv.org/abs/2402.14020)

    本研究发现对大型语言模型的对抗性攻击不仅仅局限于“越狱”，而包括迫使模型展示各种意外行为，攻击表面和目标广泛。这些攻击源于LLMs的预训练和常见词汇中存在的“故障”标记。

    

    最近有研究表明，对大型语言模型（LLMs）的对抗性攻击可以“越狱”该模型以发表有害言论。在这项工作中，我们认为LLMs的对抗性攻击范围远不止于越狱。我们提供了对可能的攻击面和攻击目标的广泛概述。根据一系列具体示例，我们讨论、分类和系统化了一些攻击，这些攻击迫使LLMs展示各种意外行为，如误导、模型控制、拒绝服务或数据提取。我们通过控制实验分析这些攻击，并发现其中许多是由于预训练LLMs具有编码能力的实践，以及常见LLMs词汇中应删除的奇怪“故障”标记的持续存在所导致的。

    arXiv:2402.14020v1 Announce Type: cross  Abstract: It has recently been shown that adversarial attacks on large language models (LLMs) can "jailbreak" the model into making harmful statements. In this work, we argue that the spectrum of adversarial attacks on LLMs is much larger than merely jailbreaking. We provide a broad overview of possible attack surfaces and attack goals. Based on a series of concrete examples, we discuss, categorize and systematize attacks that coerce varied unintended behaviors, such as misdirection, model control, denial-of-service, or data extraction.   We analyze these attacks in controlled experiments, and find that many of them stem from the practice of pre-training LLMs with coding capabilities, as well as the continued existence of strange "glitch" tokens in common LLM vocabularies that should be removed for security reasons.
    
[^2]: LLM作为评判者是否稳健？研究通用对抗攻击对零样点LLM评估的影响

    Is LLM-as-a-Judge Robust? Investigating Universal Adversarial Attacks on Zero-shot LLM Assessment

    [https://arxiv.org/abs/2402.14016](https://arxiv.org/abs/2402.14016)

    该研究研究了评估LLM的对抗鲁棒性，发现短通用短语可以欺骗LLMs提供高评分，这种攻击对于从简单的串联攻击到转移学习都是有效的。

    

    大型语言模型（LLMs）是强大的零样点评估者，在实际场景中越来越多地被用于笔试或系统基准测试等情境。尽管如此，目前还没有研究分析对抗试图操纵输出的评判LLMs的脆弱性的工作。这项工作提出了对评估LLMs的对抗鲁棒性的第一项研究，我们寻找短通用短语，当附加到文本时可以欺骗LLMs提供高评分。在SummEval和TopicalChat上的实验表明，LLM评分和两两LLM比较评估都容易受到简单的串联攻击的影响，尤其是LLM评分非常容易受到影响，可以产生最高评分，而不考虑输入文本的质量。有趣的是，这些攻击是可传递的，学到的短语可以应用于更大的封闭源模型，如GPT3.5

    arXiv:2402.14016v1 Announce Type: new  Abstract: Large Language Models (LLMs) are powerful zero-shot assessors and are increasingly used in real-world situations such as for written exams or benchmarking systems. Despite this, no existing work has analyzed the vulnerability of judge-LLMs against adversaries attempting to manipulate outputs. This work presents the first study on the adversarial robustness of assessment LLMs, where we search for short universal phrases that when appended to texts can deceive LLMs to provide high assessment scores. Experiments on SummEval and TopicalChat demonstrate that both LLM-scoring and pairwise LLM-comparative assessment are vulnerable to simple concatenation attacks, where in particular LLM-scoring is very susceptible and can yield maximum assessment scores irrespective of the input text quality. Interestingly, such attacks are transferable and phrases learned on smaller open-source LLMs can be applied to larger closed-source models, such as GPT3.5
    
[^3]: OlympiadBench：一个具有奥林匹亚级别双语多模态科学问题的挑战性基准

    OlympiadBench: A Challenging Benchmark for Promoting AGI with Olympiad-Level Bilingual Multimodal Scientific Problems

    [https://arxiv.org/abs/2402.14008](https://arxiv.org/abs/2402.14008)

    提出了OlympiadBench，一个奥林匹亚级别的双语多模态科学基准，包括8952个问题，旨在评估大型语言模型和多模态模型在复杂问题上的能力。

    

    最近的进展使得大型语言模型（LLMs）和大型多模态模型（LMMs）在各种任务中超越了一般人类的能力，接近了多个领域人类专家的熟练水平。本文提出了OlympiadBench，一个奥林匹亚级别的双语多模态科学基准，包括来自奥林匹亚级别数学和物理竞赛以及中国高考的8952个问题。每个问题都配有专家级注释，以进行逐步的推理。在OlympiadBench上评估顶尖模型，我们实施了全面的评估方法，以准确评估模型的响应。值得注意的是，表现最佳的模型GPT-4V在OlympiadBench上获得了17.23%的平均分，其中在物理学中仅为11.28%。

    arXiv:2402.14008v1 Announce Type: new  Abstract: Recent advancements have seen Large Language Models (LLMs) and Large Multimodal Models (LMMs) surpassing general human capabilities in various tasks, approaching the proficiency level of human experts across multiple domains. With traditional benchmarks becoming less challenging for these models, new rigorous challenges are essential to gauge their advanced abilities. In this work, we present OlympiadBench, an Olympiad-level bilingual multimodal scientific benchmark, featuring 8,952 problems from Olympiad-level mathematics and physics competitions, including the Chinese college entrance exam. Each problem is detailed with expert-level annotations for step-by-step reasoning. Evaluating top-tier models on OlympiadBench, we implement a comprehensive assessment methodology to accurately evaluate model responses. Notably, the best-performing model, GPT-4V, attains an average score of 17.23% on OlympiadBench, with a mere 11.28% in physics, hig
    
[^4]: 水印是否能够在翻译中存活？关于大型语言模型文本水印的跨语言一致性

    Can Watermarks Survive Translation? On the Cross-lingual Consistency of Text Watermark for Large Language Models

    [https://arxiv.org/abs/2402.14007](https://arxiv.org/abs/2402.14007)

    该研究引入了文本水印中的“跨语言一致性”概念，发现当前文本水印技术在文本被翻译成其他语言后失去了一致性，并提出了一种跨语言水印去除攻击方法，有效绕过水印，降低AUC值，同时指出了导致这种差异的关键因素。

    

    文本水印技术旨在标记和识别大型语言模型（LLMs）生成的内容，以防止滥用。本研究引入了文本水印中的“跨语言一致性”概念，评估了文本水印在被翻译成其他语言后保持有效性的能力。两个LLM和三种水印方法的初步实证结果显示，当前的文本水印技术在文本被翻译成不同语言时缺乏一致性。基于这一观察，我们提出了一种跨语言水印去除攻击（CWRA）方法，通过首先从一个LLM中获取来自中介语言的响应，然后将其翻译成目标语言来绕过水印，从而有效地减少AUC值从0.95降至0.67而无性能损失。此外，我们分析了导致交叉一致性差异的两个关键因素。

    arXiv:2402.14007v1 Announce Type: cross  Abstract: Text watermarking technology aims to tag and identify content produced by large language models (LLMs) to prevent misuse. In this study, we introduce the concept of ''cross-lingual consistency'' in text watermarking, which assesses the ability of text watermarks to maintain their effectiveness after being translated into other languages. Preliminary empirical results from two LLMs and three watermarking methods reveal that current text watermarking technologies lack consistency when texts are translated into various languages. Based on this observation, we propose a Cross-lingual Watermark Removal Attack (CWRA) to bypass watermarking by first obtaining a response from an LLM in a pivot language, which is then translated into the target language. CWRA can effectively remove watermarks by reducing the Area Under the Curve (AUC) from 0.95 to 0.67 without performance loss. Furthermore, we analyze two key factors that contribute to the cros
    
[^5]: 幻觉还是注意力误导？在商业中利用大型语言模型进行战略价值提取的路径

    Hallucinations or Attention Misdirection? The Path to Strategic Value Extraction in Business Using Large Language Models

    [https://arxiv.org/abs/2402.14002](https://arxiv.org/abs/2402.14002)

    本文提出在商业环境中使用大型语言模型进行战略价值提取时，需要理解幻觉和注意力误导之间的区别，并强调了一个达到显著错误的战略框架。

    

    具有变压器架构的大型语言模型彻底改变了文本生成领域，树立了前所未有的基准。尽管它们具有令人印象深刻的能力，但被批评生成的结果偏离事实准确性或展示逻辑不一致，这些现象通常被称为幻觉。然而，这个术语经常被错误地应用于任何偏离教师期望的结果，而本文将其定义为注意力误导而非真正的幻觉。理解幻觉和注意力误导之间的区别在商业环境中变得越来越重要，因为这些错误的后果可能会显著影响从这些内在预训练模型中提取价值。本文重点介绍了PGI，Persona，Grouping和Intelligence方法的最佳实践，这是一个实现了显著误差的战略框架。

    arXiv:2402.14002v1 Announce Type: new  Abstract: Large Language Models with transformer architecture have revolutionized the domain of text generation, setting unprecedented benchmarks. Despite their impressive capabilities, LLMs have been criticized for generating outcomes that deviate from factual accuracy or display logical inconsistencies, phenomena commonly referred to as hallucinations. This term, however, has often been misapplied to any results deviating from the instructor's expectations, which this paper defines as attention misdirection rather than true hallucinations. Understanding the distinction between hallucinations and attention misdirection becomes increasingly relevant in business contexts, where the ramifications of such errors can significantly impact the value extraction from these inherently pre-trained models. This paper highlights the best practices of the PGI, Persona, Grouping, and Intelligence, method, a strategic framework that achieved a remarkable error r
    
[^6]: 分析序列组成对语言模型预训练的影响

    Analysing The Impact of Sequence Composition on Language Model Pre-Training

    [https://arxiv.org/abs/2402.13991](https://arxiv.org/abs/2402.13991)

    序列组成对语言模型预训练的影响一直未被深入探讨，在这项研究中发现，应用内部文档因果遮盖可以消除来自之前文档的干扰信息，显著提高模型在语言建模和下游任务中的性能。

    

    大多数语言模型预训练框架将多个文档连接成固定长度的序列，并使用因果遮盖来计算每个标记在给定上下文下的可能性；这种策略由于简单和高效而被广泛采用。然而，迄今为止，预训练序列组成策略对模型的泛化特性的影响仍未被深入研究。 在这项工作中，我们发现应用因果遮盖可能导致在预训练过程中包括来自之前文档的干扰信息，从而对模型在语言建模和下游任务上的性能产生负面影响。 在内部文档因果遮盖中，每个标记的可能性仅取决于同一文档中的先前标记，消除了来自之前文档的潜在干扰信息并显著提高了性能。 此外，我们发现连接相关文档

    arXiv:2402.13991v1 Announce Type: new  Abstract: Most language model pre-training frameworks concatenate multiple documents into fixed-length sequences and use causal masking to compute the likelihood of each token given its context; this strategy is widely adopted due to its simplicity and efficiency. However, to this day, the influence of the pre-training sequence composition strategy on the generalisation properties of the model remains under-explored. In this work, we find that applying causal masking can lead to the inclusion of distracting information from previous documents during pre-training, which negatively impacts the performance of the models on language modelling and downstream tasks. In intra-document causal masking, the likelihood of each token is only conditioned on the previous tokens in the same document, eliminating potential distracting information from previous documents and significantly improving performance. Furthermore, we find that concatenating related docum
    
[^7]: 为医学构建多语言语言模型

    Towards Building Multilingual Language Model for Medicine

    [https://arxiv.org/abs/2402.13963](https://arxiv.org/abs/2402.13963)

    本文提出了为医学领域构建多语言语言模型的三个关键贡献:构建了新的多语言医学语料库MMedC，提出了多语言医学多选问答基准MMedBench，并且通过在MMedC上进一步训练获得了性能优越的MMedLM 2模型。

    

    本文旨在开发一种面向医学的开源多语言语言模型，使得更广泛的语言多样性受众受益。我们的工作主要贡献体现在以下几个方面:首先，针对多语言医学特定适应性，我们构建了一个新的多语言医学语料库，包含大约25.5B个tokens，覆盖了6种主要语言，被称为MMedC，这使得现有通用LLM能够进行自回归训练。其次，为了监测医学领域多语言LLM的发展，我们提出了一个新的带有解释的多语言医学多选问答基准，称为MMedBench；第三，我们评估了一些流行的开源大型语言模型(LLMs)在我们的基准上的表现，以及那些在MMedC上进一步进行自回归训练的模型，最终，我们的最终模型，命名为MMedLM 2，仅有7B参数，取得了卓越的性能。

    arXiv:2402.13963v1 Announce Type: new  Abstract: In this paper, we aim to develop an open-source, multilingual language model for medicine, that the benefits a wider, linguistically diverse audience from different regions. In general, we present the contribution from the following aspects: first, for multilingual medical-specific adaptation, we construct a new multilingual medical corpus, that contains approximately 25.5B tokens encompassing 6 main languages, termed as MMedC, that enables auto-regressive training for existing general LLMs. second, to monitor the development of multilingual LLMs in medicine, we propose a new multilingual medical multi-choice question-answering benchmark with rationale, termed as MMedBench; third, we have assessed a number of popular, opensource large language models (LLMs) on our benchmark, along with those further auto-regressive trained on MMedC, as a result, our final model, termed as MMedLM 2, with only 7B parameters, achieves superior performance c
    
[^8]: 你能通过下一个词预测学习语义吗？以蕴涵为例

    Can You Learn Semantics Through Next-Word Prediction? The Case of Entailment

    [https://arxiv.org/abs/2402.13956](https://arxiv.org/abs/2402.13956)

    作者调查了神经LM是否可以通过下一个词预测来解码蕴涵判断，发现它们可以远高于随机几率地解码自然句子之间的蕴涵关系，暗示LM隐含地模拟了语义的某些方面。

    

    Merrill等人（2022）认为，在理论上，最优LM预测的概率编码了关于蕴涵关系的语义信息，但是由于Merrill等人提出的强烈理想化假设，不清楚神经LM在训练语料库上是否通过这种方式学习蕴涵。在这项工作中，我们调查了他们的理论是否可以用于从神经LM中解码蕴涵判断。我们发现类似于他们的测试可以在许多数据集和LM中解码自然句子之间的蕴涵关系，远远超过随机机会，尽管不是完美的。这表明LM隐含地模拟了语义的某些方面，以预测句子共现模式上的语义效应。但是，我们发现实际上预测蕴涵的测试与理论测试的方向相反。因此，我们重新审视了潜在的理论假设。

    arXiv:2402.13956v1 Announce Type: new  Abstract: Do LMs infer the semantics of text from co-occurrence patterns in their training data? Merrill et al. (2022) argue that, in theory, probabilities predicted by an optimal LM encode semantic information about entailment relations, but it is unclear whether neural LMs trained on corpora learn entailment in this way because of strong idealizing assumptions made by Merrill et al. In this work, we investigate whether their theory can be used to decode entailment judgments from neural LMs. We find that a test similar to theirs can decode entailment relations between natural sentences, well above random chance, though not perfectly, across many datasets and LMs. This suggests LMs implicitly model aspects of semantics to predict semantic effects on sentence co-occurrence patterns. However, we find the test that predicts entailment in practice works in the opposite direction to the theoretical test. We thus revisit the assumptions underlying the o
    
[^9]: 通过预测质量间接测量掩盖语言模型中的社会偏见

    Measuring Social Biases in Masked Language Models by Proxy of Prediction Quality

    [https://arxiv.org/abs/2402.13954](https://arxiv.org/abs/2402.13954)

    本文通过提出的代理函数在迭代屏蔽实验中评估了转换器模型所编码的社会偏见，并比较了其与其他评估方法的偏见估计，发现转换器模型中存在相对较高的宗教和残疾偏见，而性别偏见则相对较低。

    

    社会和政治科学家经常旨在从文本数据表示（嵌入）中发现和衡量不同的偏见。创新的基于转换器的语言模型生成具有上下文感知的令牌嵌入，并在各种自然语言任务中取得了最先进的性能，但已被证明在下游应用中编码了不需要的偏见。本文通过提出的代理函数在迭代屏蔽实验中评估由训练有遮蔽语言建模目标的转换器所编码的社会偏见，以测量转换器模型预测质量，并评估MLM对不利群体和有利群体的偏好。我们比较使用两个基准数据集的偏见估计与其他评估方法产生的偏见，发现考虑的MLMs中存在相对较高的宗教和残疾偏见，而相对于另一个数据集，一个数据集中存在较低的性别偏见。

    arXiv:2402.13954v1 Announce Type: new  Abstract: Social and political scientists often aim to discover and measure distinct biases from text data representations (embeddings). Innovative transformer-based language models produce contextually-aware token embeddings and have achieved state-of-the-art performance for a variety of natural language tasks, but have been shown to encode unwanted biases for downstream applications. In this paper, we evaluate the social biases encoded by transformers trained with the masked language modeling objective using proposed proxy functions within an iterative masking experiment to measure the quality of transformer models' predictions, and assess the preference of MLMs towards disadvantaged and advantaged groups. We compare bias estimations with those produced by other evaluation methods using two benchmark datasets, finding relatively high religious and disability biases across considered MLMs and low gender bias in one dataset relative to the other. 
    
[^10]: 使推理变得重要：衡量和提高链式思维推理的忠实性

    Making Reasoning Matter: Measuring and Improving Faithfulness of Chain-of-Thought Reasoning

    [https://arxiv.org/abs/2402.13950](https://arxiv.org/abs/2402.13950)

    本文研究了大型语言模型的推理过程中的忠实性问题，引入了FRODO框架来改进生成推理步骤和坚固推理的方法

    

    大型语言模型(LLMs)在回答问题之前经过逐步推理已被证明表现更好。然而，模型最终答案与所述推理步骤的忠实程度尚不明确。本文对十二个LLMs进行因果中介分析，以检验LLM生成的中间推理步骤如何影响最终结果，并发现LLMs在生成答案时并不可靠地使用其中间推理步骤。为了解决这个问题，我们介绍了FRODO，一个旨在定制小型LM以生成正确推理步骤并在这些步骤上进行坚固推理的框架。FRODO包括一个推断模块，通过学习使用隐式因果奖励函数生成正确推理步骤，并且一个推理模块，通过学习使用反事实和因果偏好目标在这些中间推理上忠实推理。我们的实验证明F

    arXiv:2402.13950v1 Announce Type: new  Abstract: Large language models (LLMs) have been shown to perform better when asked to reason step-by-step before answering a question. However, it is unclear to what degree the model's final answer is faithful to the stated reasoning steps. In this paper, we perform a causal mediation analysis on twelve LLMs to examine how intermediate reasoning steps generated by the LLM influence the final outcome and find that LLMs do not reliably use their intermediate reasoning steps when generating an answer. To address this issue, we introduce FRODO, a framework to tailor small-sized LMs to generate correct reasoning steps and robustly reason over these steps. FRODO consists of an inference module that learns to generate correct reasoning steps using an implicit causal reward function and a reasoning module that learns to faithfully reason over these intermediate inferences using a counterfactual and causal preference objective. Our experiments show that F
    
[^11]: 独特图像字幕：在CLIP引导强化学习中利用地面真实字幕

    Distinctive Image Captioning: Leveraging Ground Truth Captions in CLIP Guided Reinforcement Learning

    [https://arxiv.org/abs/2402.13936](https://arxiv.org/abs/2402.13936)

    论文提出了一种新的图像字幕模型训练策略，在强化学习框架中利用地面真实字幕，以提高生成字幕的独特性。

    

    使用教师强迫训练图像字幕模型会导致生成非常通用的样本，而更具有独特性的字幕在检索应用或为图像生成替代文本以增强可访问性方面非常有用。 强化学习（RL）允许使用生成字幕与输入图像之间的跨模态检索相似度分数作为奖励来指导训练，从而产生更具独特性的字幕。 最近的研究表明，预训练的跨模态检索模型可以用于提供这种奖励，从而完全消除了对参考字幕的需求。 然而，我们在本文中认为地面真实（GT）字幕在这种RL框架中仍然可以发挥作用。 我们提出了一种新的图像字幕模型训练策略，以不同方式利用GT字幕。 首先，它们可以用来训练一个简单的MLP鉴别器，作为正则化的一部分，以防止奖励作弊并确保流畅性。

    arXiv:2402.13936v1 Announce Type: new  Abstract: Training image captioning models using teacher forcing results in very generic samples, whereas more distinctive captions can be very useful in retrieval applications or to produce alternative texts describing images for accessibility. Reinforcement Learning (RL) allows to use cross-modal retrieval similarity score between the generated caption and the input image as reward to guide the training, leading to more distinctive captions. Recent studies show that pre-trained cross-modal retrieval models can be used to provide this reward, completely eliminating the need for reference captions. However, we argue in this paper that Ground Truth (GT) captions can still be useful in this RL framework. We propose a new image captioning model training strategy that makes use of GT captions in different ways. Firstly, they can be used to train a simple MLP discriminator that serves as a regularization to prevent reward hacking and ensures the fluenc
    
[^12]: 确实高效的Transformer能够节约计算吗？

    Do Efficient Transformers Really Save Computation?

    [https://arxiv.org/abs/2402.13934](https://arxiv.org/abs/2402.13934)

    本研究旨在理解高效Transformer（例如稀疏Transformer和线性Transformer）的能力和限制，发现它们适合解决一般DP任务，但不同于标准Transformer。

    

    随着基于Transformer的语言模型在越来越大的数据集上训练，并拥有大量参数，找到更高效的替代标准Transformer变得非常有价值。虽然已经提出了许多高效的Transformer和Transformer的替代方案，但没有一个能够提供它们适合替代标准Transformer的理论保证。这使得很难确定何时使用特定模型以及进一步研究的重点。在本文中，我们旨在理解高效Transformer的能力和局限性，特别是稀疏Transformer和线性Transformer。我们专注于它们在Chain-of-Thought (CoT)提示中展示的推理能力，并遵循先前的研究将它们建模为动态规划（DP）问题。我们的结果表明，虽然这些模型足够表达解决一般DP任务的能力，但与标准Transformer不同

    arXiv:2402.13934v1 Announce Type: cross  Abstract: As transformer-based language models are trained on increasingly large datasets and with vast numbers of parameters, finding more efficient alternatives to the standard Transformer has become very valuable. While many efficient Transformers and Transformer alternatives have been proposed, none provide theoretical guarantees that they are a suitable replacement for the standard Transformer. This makes it challenging to identify when to use a specific model and what directions to prioritize for further investigation. In this paper, we aim to understand the capabilities and limitations of efficient Transformers, specifically the Sparse Transformer and the Linear Transformer. We focus on their reasoning capability as exhibited by Chain-of-Thought (CoT) prompts and follow previous works to model them as Dynamic Programming (DP) problems. Our results show that while these models are expressive enough to solve general DP tasks, contrary to ex
    
[^13]: 大型语言模型易受诱饵-转换攻击的危害内容生成研究

    Large Language Models are Vulnerable to Bait-and-Switch Attacks for Generating Harmful Content

    [https://arxiv.org/abs/2402.13926](https://arxiv.org/abs/2402.13926)

    大型语言模型可能受到诱饵-转换攻击的威胁，甚至安全生成的文本也能轻易转变为有害内容，强调在LLMs的安全防护中需要考虑后处理转换。

    

    大型语言模型（LLMs）生成欺骗性和有害内容所带来的风险已经引起了相当多的研究，但即使是安全的生成也可能导致问题降级影响。在我们的研究中，我们将焦点转移到即使来自LLMs的安全文本也可以通过诱饵-转换攻击轻松转变为潜在危险内容。在这种攻击中，用户首先用安全问题提示LLMs，然后利用简单的查找和替换后处理技术将输出操纵成有害叙事。这种方法在生成有毒内容方面的惊人有效性突出了在开发可靠的LLMs安全防护栏时面临的重大挑战。特别是，我们强调，专注于逐字的LLMs输出的安全性是不够的，我们还需要考虑后处理转换。

    arXiv:2402.13926v1 Announce Type: cross  Abstract: The risks derived from large language models (LLMs) generating deceptive and damaging content have been the subject of considerable research, but even safe generations can lead to problematic downstream impacts. In our study, we shift the focus to how even safe text coming from LLMs can be easily turned into potentially dangerous content through Bait-and-Switch attacks. In such attacks, the user first prompts LLMs with safe questions and then employs a simple find-and-replace post-hoc technique to manipulate the outputs into harmful narratives. The alarming efficacy of this approach in generating toxic content highlights a significant challenge in developing reliable safety guardrails for LLMs. In particular, we stress that focusing on the safety of the verbatim LLM outputs is insufficient and that we also need to consider post-hoc transformations.
    
[^14]: SYNFAC-EDIT: 用于临床摘要中的事实对齐的合成模仿编辑反馈

    SYNFAC-EDIT: Synthetic Imitation Edit Feedback for Factual Alignment in Clinical Summarization

    [https://arxiv.org/abs/2402.13919](https://arxiv.org/abs/2402.13919)

    该研究提出了一种创新流程，利用GPT-3.5和GPT-4生成高质量反馈，以增强临床笔记摘要中的事实一致性，弥补了专家注释数据的高成本和有限可用性问题。

    

    大型语言模型（LLMs）如GPT和Llama在摘要任务上取得了重大进展，但在事实不准确方面存在困难，这是临床NLP应用中的关键问题，错误可能导致严重后果。为了解决事实对齐的专家注释数据成本高昂且有限的问题，本研究引入了一种创新的流程，利用GPT-3.5和GPT-4生成高质量反馈，旨在增强临床笔记摘要中的事实一致性。我们的研究主要关注编辑反馈，在没有额外注释的情况下，模拟了医疗专业人员改善AI系统输出的实际场景。尽管GPT在各种临床NLP任务中都表现出了专业水平，比如医学执照考试，但对其提供改善较弱LM或LLM生成质量的专业级编辑反馈的研究很少。

    arXiv:2402.13919v1 Announce Type: cross  Abstract: Large Language Models (LLMs) such as GPT and Llama have demonstrated significant achievements in summarization tasks but struggle with factual inaccuracies, a critical issue in clinical NLP applications where errors could lead to serious consequences. To counter the high costs and limited availability of expert-annotated data for factual alignment, this study introduces an innovative pipeline that utilizes GPT-3.5 and GPT-4 to generate high-quality feedback aimed at enhancing factual consistency in clinical note summarization. Our research primarily focuses on edit feedback, mirroring the practical scenario in which medical professionals refine AI system outputs without the need for additional annotations. Despite GPT's proven expertise in various clinical NLP tasks, such as the Medical Licensing Examination, there is scant research on its capacity to deliver expert-level edit feedback for improving weaker LMs or LLMs generation qualit
    
[^15]: LLM翻译中的语言特征和重要语言是什么？

    What Linguistic Features and Languages are Important in LLM Translation?

    [https://arxiv.org/abs/2402.13917](https://arxiv.org/abs/2402.13917)

    Llama2模型在翻译中表现出准确度高，部分未见语言需要更大规模的模型来提升翻译质量，另外语言的句法相似性并非翻译质量的主要因素，某些语言即使数据少依然表现出强相关性。

    

    arXiv：2402.13917v1 公告类型：跨领域 摘要：大型语言模型（LLMs）展示了在多个任务中具有强大能力，包括机器翻译。我们的研究重点在于评估Llama2的机器翻译能力，并探索翻译如何取决于其训练数据中的语言。我们的实验表明，7B Llama2模型对其所见的所有语言都可以获得超过10的BLEU分数，但并非总是对其未见的语言。对于这些未见语言，与使用聊天版本或添加少量数据相比，在模型规模上观察到的最大收益。此外，我们的语言距离分析显示，句法相似性并非始终是决定翻译质量的主要语言因素。有趣的是，我们发现在特定情况下，一些语言，尽管训练数据明显少于英语，却表现出与英语可比的强相关性。我们在这里的发现为研究提供了新的视角。

    arXiv:2402.13917v1 Announce Type: cross  Abstract: Large Language Models (LLMs) demonstrate strong capability across multiple tasks, including machine translation. Our study focuses on evaluating Llama2's machine translation capabilities and exploring how translation depends on languages in its training data. Our experiments show that the 7B Llama2 model yields above 10 BLEU score for all languages it has seen, but not always for languages it has not seen. Most gains for those unseen languages are observed the most with the model scale compared to using chat versions or adding shot count. Furthermore, our linguistic distance analysis reveals that syntactic similarity is not always the primary linguistic factor in determining translation quality. Interestingly, we discovered that under specific circumstances, some languages, despite having significantly less training data than English, exhibit strong correlations comparable to English. Our discoveries here give new perspectives for the 
    
[^16]: 利用整个收藏相似性进行无监督文档结构提取

    Leveraging Collection-Wide Similarities for Unsupervised Document Structure Extraction

    [https://arxiv.org/abs/2402.13906](https://arxiv.org/abs/2402.13906)

    通过无监督图方式，利用文档间和文内相似性，提取文档收藏的整体结构。

    

    各个领域的文档收藏，如法律、医学或金融等，通常共享一些潜在的整个收藏结构，这些结构捕捉到的信息可以帮助人类用户和结构感知模型。我们提出识别收藏内文档的典型结构，需要捕捉整个收藏中反复出现的主题，同时摘要任意标题的释义，并将每个主题与相应的文档位置联系起来。这些要求带来了几个挑战：标记反复出现主题的标题在措辞上经常不同，某些节标题仅适用于个别文档且不反映典型结构，主题顺序在文档之间可能会有所变化。随后，我们开发了一种利用文档间和文内相似性的无监督基于图的方法，来提取潜在的整个收藏结构。

    arXiv:2402.13906v1 Announce Type: new  Abstract: Document collections of various domains, e.g., legal, medical, or financial, often share some underlying collection-wide structure, which captures information that can aid both human users and structure-aware models. We propose to identify the typical structure of document within a collection, which requires to capture recurring topics across the collection, while abstracting over arbitrary header paraphrases, and ground each topic to respective document locations. These requirements pose several challenges: headers that mark recurring topics frequently differ in phrasing, certain section headers are unique to individual documents and do not reflect the typical structure, and the order of topics can vary between documents. Subsequently, we develop an unsupervised graph-based method which leverages both inter- and intra-document similarities, to extract the underlying collection-wide structure. Our evaluations on three diverse domains in 
    
[^17]: 使用样本一致性校准大型语言模型

    Calibrating Large Language Models with Sample Consistency

    [https://arxiv.org/abs/2402.13904](https://arxiv.org/abs/2402.13904)

    通过多个随机抽样模型生成的分布推断置信度的潜力，可以增强大型语言模型的校准性能。

    

    准确评估大型语言模型（LLMs）预测的置信水平对于它们的可靠应用至关重要。然而，由于其专有性质和大规模，LLMs通常天生不经校准，使得传统的校准技术很难适用。在这项工作中，我们探讨了通过多个随机抽样的模型生成的分布来推断置信度的潜力，采用了三种一致性度量。我们在九个推理数据集上对各种开源和闭源模型进行了广泛评估。结果显示，基于一致性的校准方法胜过现有的事后方法。同时，我们发现中间解释、模型扩展和更大的样本大小等因素可以增强校准，而指导调整会使校准变得更加困难。此外，通过一致性获得的置信度分数有望提升模型性能。最后，我们提供

    arXiv:2402.13904v1 Announce Type: new  Abstract: Accurately gauging the confidence level of Large Language Models' (LLMs) predictions is pivotal for their reliable application. However, LLMs are often uncalibrated inherently and elude conventional calibration techniques due to their proprietary nature and massive scale. In this work, we explore the potential of deriving confidence from the distribution of multiple randomly sampled model generations, via three measures of consistency. We perform an extensive evaluation across various open and closed-source models on nine reasoning datasets. Results show that consistency-based calibration methods outperform existing post-hoc approaches. Meanwhile, we find that factors such as intermediate explanations, model scaling, and larger sample sizes enhance calibration, while instruction-tuning makes calibration more difficult. Moreover, confidence scores obtained from consistency have the potential to enhance model performance. Finally, we offer
    
[^18]: 科学检查者再度升级：透明度和逻辑推理的双向范式

    Science Checker Reloaded: A Bidirectional Paradigm for Transparency and Logical Reasoning

    [https://arxiv.org/abs/2402.13897](https://arxiv.org/abs/2402.13897)

    提出了一个两块式的方法来解决长文档中信息检索领域的挑战，并实现了双向交互

    

    信息检索是一个快速发展的领域。然而，它仍然面临着在科学和工业的海量信息中的诸多限制，比如语义分歧和检索中的词汇差距、语义搜索中的低精度和缺乏可解释性，或者生成模型中的幻觉和过时信息。在本文中，我们提出了一个两块式的方法来解决长文档的这些障碍。第一个模块通过查询扩展增强了在稀疏检索中的语言理解，以检索相关文档。第二个模块通过只使用长文档中传播的信息，为复杂问题提供全面和信息丰富的答案来加深结果，实现双向交互。在管道的各个阶段，向用户呈现中间结果以促进对系统推理的理解。我们相信这种双向方法带来了

    arXiv:2402.13897v1 Announce Type: cross  Abstract: Information retrieval is a rapidly evolving field. However it still faces significant limitations in the scientific and industrial vast amounts of information, such as semantic divergence and vocabulary gaps in sparse retrieval, low precision and lack of interpretability in semantic search, or hallucination and outdated information in generative models. In this paper, we introduce a two-block approach to tackle these hurdles for long documents. The first block enhances language understanding in sparse retrieval by query expansion to retrieve relevant documents. The second block deepens the result by providing comprehensive and informative answers to the complex question using only the information spread in the long document, enabling bidirectional engagement. At various stages of the pipeline, intermediate results are presented to users to facilitate understanding of the system's reasoning. We believe this bidirectional approach brings
    
[^19]: 超越概率：揭示评估大型语言模型中的错位问题

    Beyond Probabilities: Unveiling the Misalignment in Evaluating Large Language Models

    [https://arxiv.org/abs/2402.13887](https://arxiv.org/abs/2402.13887)

    本研究揭示了在使用大型语言模型进行多项选择题时，基于概率的评估方法与基于生成的预测不相吻合的固有局限性。

    

    大型语言模型（LLMs）在各种应用中展现出卓越的能力，从根本上改变了自然语言处理（NLP）研究的格局。然而，最近的评估框架通常依赖于LLMs的输出概率进行预测，主要是由于计算约束，偏离了真实世界的LLMs使用场景。虽然被广泛采用，基于概率的评估策略的有效性仍是一个开放的研究问题。本研究旨在审查这种基于概率的评估方法在使用LLMs进行多项选择题（MCQs）时的有效性，突显其固有局限性。我们的实证调查显示，普遍的基于概率的评估方法未能与基于生成的预测相适应。此外，当前的评估框架通常通过基于输出预测的预测任务来评估LLMs

    arXiv:2402.13887v1 Announce Type: new  Abstract: Large Language Models (LLMs) have demonstrated remarkable capabilities across various applications, fundamentally reshaping the landscape of natural language processing (NLP) research. However, recent evaluation frameworks often rely on the output probabilities of LLMs for predictions, primarily due to computational constraints, diverging from real-world LLM usage scenarios. While widely employed, the efficacy of these probability-based evaluation strategies remains an open research question. This study aims to scrutinize the validity of such probability-based evaluation methods within the context of using LLMs for Multiple Choice Questions (MCQs), highlighting their inherent limitations. Our empirical investigation reveals that the prevalent probability-based evaluation method inadequately aligns with generation-based prediction. Furthermore, current evaluation frameworks typically assess LLMs through predictive tasks based on output pr
    
[^20]: $\texttt{Se}^2$: $\textit{Se}$quential Example $\textit{Se}$lection for In-Context Learning

    $\texttt{Se}^2$: $\textit{Se}$quential Example $\textit{Se}$lection for In-Context Learning

    [https://arxiv.org/abs/2402.13874](https://arxiv.org/abs/2402.13874)

    本文提出了$\texttt{Se}^2$，一种顺序感知方法，利用大型语言模型的反馈帮助捕捉示例之间的相互关系和序列信息，显著丰富了上下文学习提示的相关性和相关性。

    

    众所周知，大型语言模型（LLMs）在上下文学习（ICL）中具有出色的能力，但需要通过示例示范来激活。以往的研究广泛探讨了用于ICL的示例选择，主要遵循“先选择再组织”的范式，这些方法往往忽视示例之间的内在关系，存在训练和推理之间的不一致性。本文将问题表述为一个序贯选择问题，并引入$\texttt{Se}^2$，这是一种顺序感知方法，利用LLM对不同上下文的反馈，有助于捕捉示例之间的相互关系和序列信息，显著丰富ICL提示的上下文相关性和相关性。同时，我们利用束搜索来寻找和构建示例序列，增强了质量和多样性。我们在8个不同类别中的23个NLP任务上进行了大量实验

    arXiv:2402.13874v1 Announce Type: new  Abstract: The remarkable capability of large language models (LLMs) for in-context learning (ICL) needs to be activated by demonstration examples. Prior work has extensively explored the selection of examples for ICL, predominantly following the "select then organize" paradigm, such approaches often neglect the internal relationships between examples and exist an inconsistency between the training and inference. In this paper, we formulate the problem as a $\textit{se}$quential $\textit{se}$lection problem and introduce $\texttt{Se}^2$, a sequential-aware method that leverages the LLM's feedback on varying context, aiding in capturing inter-relationships and sequential information among examples, significantly enriching the contextuality and relevance of ICL prompts. Meanwhile, we utilize beam search to seek and construct example sequences, enhancing both quality and diversity. Extensive experiments across 23 NLP tasks from 8 distinct categories i
    
[^21]: Kuaiji：第一个中国会计大型语言模型

    Kuaiji: the First Chinese Accounting Large Language Model

    [https://arxiv.org/abs/2402.13866](https://arxiv.org/abs/2402.13866)

    Kuaiji是第一个中国会计大型语言模型，通过Baichuan框架精心调整，支持的CAtAcctQA数据集，展现出卓越的准确性和响应速度，具有开创性地创建了中国会计数据集，并证实了在真实会计场景中的高效性。

    

    大语言模型（LLMs）如ChatGPT和GPT-4已经展示出在理解和生成自然语言方面的出色能力。然而，当面临任务要求适应会计等专业领域时，它们会遇到困难。为了解决这一挑战，我们引入了Kuaiji，一个专门定制的会计大型语言模型。Kuaiji经过精心调整，使用包含连续预训练和监督微调过程的Baichuan框架。在CAtAcctQA的支持下，这是一个包含大量真实会计师与客户对话的数据集，Kuaiji表现出卓越的准确性和响应速度。我们的贡献包括创建了第一个中国会计数据集，将Kuaiji建立为一种领先的开源中国会计LLM，并通过真实会计场景对其有效性进行了验证。

    arXiv:2402.13866v1 Announce Type: cross  Abstract: Large Language Models (LLMs) like ChatGPT and GPT-4 have demonstrated impressive proficiency in comprehending and generating natural language. However, they encounter difficulties when tasked with adapting to specialized domains such as accounting. To address this challenge, we introduce Kuaiji, a tailored Accounting Large Language Model. Kuaiji is meticulously fine-tuned using the Baichuan framework, which encompasses continuous pre-training and supervised fine-tuning processes. Supported by CAtAcctQA, a dataset containing large genuine accountant-client dialogues, Kuaiji exhibits exceptional accuracy and response speed. Our contributions encompass the creation of the first Chinese accounting dataset, the establishment of Kuaiji as a leading open-source Chinese accounting LLM, and the validation of its efficacy through real-world accounting scenarios.
    
[^22]: 大型语言模型是先进的匿名化工具

    Large Language Models are Advanced Anonymizers

    [https://arxiv.org/abs/2402.13846](https://arxiv.org/abs/2402.13846)

    大型语言模型在保护个人数据方面取得了重要进展，提出了一种基于对抗性LLM推断的匿名化框架。

    

    最近在隐私研究领域对大型语言模型的研究表明，它们在推断真实世界在线文本中的个人数据方面表现出接近人类水平的性能。随着模型能力的不断增强，现有的文本匿名化方法当前已经落后于监管要求和对抗威胁。这引出了一个问题：个人如何有效地保护他们在分享在线文本时的个人数据。在这项工作中，我们采取了两步来回答这个问题：首先，我们提出了一个新的设置，用于评估面对对抗性LLM的推断时的匿名化效果，从而允许自然地测量匿名化性能，同时纠正了以前指标的一些缺陷。然后，我们提出了基于LLM的对抗性匿名化框架，利用LLM的强大推断能力来指导我们的匿名化过程。在我们的实验评估中，我们展示了在真实世界中的匿名化实践。

    arXiv:2402.13846v1 Announce Type: cross  Abstract: Recent work in privacy research on large language models has shown that they achieve near human-level performance at inferring personal data from real-world online texts. With consistently increasing model capabilities, existing text anonymization methods are currently lacking behind regulatory requirements and adversarial threats. This raises the question of how individuals can effectively protect their personal data in sharing online texts. In this work, we take two steps to answer this question: We first present a new setting for evaluating anonymizations in the face of adversarial LLMs inferences, allowing for a natural measurement of anonymization performance while remedying some of the shortcomings of previous metrics. We then present our LLM-based adversarial anonymization framework leveraging the strong inferential capabilities of LLMs to inform our anonymization procedure. In our experimental evaluation, we show on real-world 
    
[^23]: 超越仇恨言论: 自然语言处理在发现贬低性语言中的挑战与机遇

    Beyond Hate Speech: NLP's Challenges and Opportunities in Uncovering Dehumanizing Language

    [https://arxiv.org/abs/2402.13818](https://arxiv.org/abs/2402.13818)

    本文评估了几种最先进的NLP模型在识别贬低性语言方面的性能，发现它们能够以70%的准确率区分贬低性语言和更广泛的仇恨言论，但也存在着偏见。

    

    人身具象化被定义为仇恨言论的一种微妙但有害的表现形式，涉及否认个人的人类特质，通常导致对边缘群体的暴力行为。尽管自然语言处理在各个领域取得了显著进展，但其在检测贬低性言语方面的应用有限，主要是由于这一领域公开可用的带标签数据稀缺。本文评估了最先进的NLP模型（包括GPT-4、GPT-3.5和LLAMA-2）在识别贬低性语言方面的性能。我们的发现显示，虽然这些模型表现出潜力，达到了70%的准确率来区分贬低性言语和更广泛的仇恨言论，但它们也显示出偏见。它们在对其他形式的仇恨言论进行分类时过于敏感，将其误判为特定目标群体的人身具象化，同时更频繁地未能识别明显的人身具象化案例。

    arXiv:2402.13818v1 Announce Type: new  Abstract: Dehumanization, characterized as a subtle yet harmful manifestation of hate speech, involves denying individuals of their human qualities and often results in violence against marginalized groups. Despite significant progress in Natural Language Processing across various domains, its application in detecting dehumanizing language is limited, largely due to the scarcity of publicly available annotated data for this domain. This paper evaluates the performance of cutting-edge NLP models, including GPT-4, GPT-3.5, and LLAMA-2, in identifying dehumanizing language. Our findings reveal that while these models demonstrate potential, achieving a 70\% accuracy rate in distinguishing dehumanizing language from broader hate speech, they also display biases. They are over-sensitive in classifying other forms of hate speech as dehumanization for a specific subset of target groups, while more frequently failing to identify clear cases of dehumanizati
    
[^24]: 线上讨论中关于欧洲和移民信息传播的地理学研究

    The Geography of Information Diffusion in Online Discourse on Europe and Migration

    [https://arxiv.org/abs/2402.13800](https://arxiv.org/abs/2402.13800)

    通过社交媒体数据分析了关于欧洲和移民的在线信息传播，引入了具有地理联系的热门话题、规模和动态传播的新视角，同时提出了一种基于跨语引语的创新方法。

    

    与欧洲和移民相关的信息在线传播很少受到外部视角的调查。然而，这是一个非常相关的主题，特别是如果用户没有直接接触过欧洲，其对欧洲的看法完全取决于在线检索到的信息。在这项工作中，我们分析了从社交媒体（Twitter）中检索的大量数据后，关于欧洲和移民在线流通的信息，以获得关于话题、规模和传播动态的新见解。我们将转发和标签网络分析与用户地理位置信息相结合，将数据与地理位置联系起来，允许从“欧洲外部”视角进行分析，特别关注非洲。我们还引入了一种基于跨语引语的创新方法，即当一种语言的内容被另一种语言评论和转发时，假设这些互动是远距离连接的代理。

    arXiv:2402.13800v1 Announce Type: new  Abstract: The online diffusion of information related to Europe and migration has been little investigated from an external point of view. However, this is a very relevant topic, especially if users have had no direct contact with Europe and its perception depends solely on information retrieved online. In this work we analyse the information circulating online about Europe and migration after retrieving a large amount of data from social media (Twitter), to gain new insights into topics, magnitude, and dynamics of their diffusion. We combine retweets and hashtags network analysis with geolocation of users, linking thus data to geography and allowing analysis from an "outside Europe" perspective, with a special focus on Africa. We also introduce a novel approach based on cross-lingual quotes, i.e. when content in a language is commented and retweeted in another language, assuming these interactions are a proxy for connections between very distant 
    
[^25]: CriticBench: 将大型语言模型作为评论家进行评估

    CriticBench: Evaluating Large Language Models as Critic

    [https://arxiv.org/abs/2402.13764](https://arxiv.org/abs/2402.13764)

    CriticBench是一个旨在全面和可靠地评估大型语言模型的评论能力的新型基准，展示了评论能力与任务、响应质量和模型规模之间的关系。

    

    论文提出了 CriticBench，这是一个旨在全面和可靠地评估大型语言模型（LLMs）的四个关键评论能力维度（反馈、比较、改进和元反馈）的新型基准。CriticBench包含九个不同的任务，每个任务评估LLMs在不同质量细粒度水平上评论响应的能力。对开源和闭源LLMs进行的广泛评估揭示了评论能力与任务、响应质量和模型规模之间有趣的关系。CriticBench的数据集、资源和评估工具包将在https://github.com/gmftbyGMFTBY/Cri上公开发布。

    arXiv:2402.13764v1 Announce Type: cross  Abstract: Critique ability are crucial in the scalable oversight and self-improvement of Large Language Models (LLMs). While many recent studies explore the critique ability of LLMs to judge and refine flaws in generations, how to comprehensively and reliably measure the critique abilities of LLMs is under-explored. This paper introduces \shortname, a novel benchmark designed to comprehensively and reliably evaluate four key critique ability dimensions of LLMs: feedback, comparison, refinement and meta-feedback. \shortname~encompasses nine diverse tasks, each assessing the LLMs' ability to critique responses at varying levels of quality granularity. Our extensive evaluations of open-source and closed-source LLMs reveal intriguing relationships between the critique ability and tasks, response qualities, and model scales. Datasets, resources and evaluation toolkit for \shortname~will be publicly released at \url{https://github.com/gmftbyGMFTBY/Cri
    
[^26]: 大语言模型时代中摘要的事实一致性评估

    Factual Consistency Evaluation of Summarisation in the Era of Large Language Models

    [https://arxiv.org/abs/2402.13758](https://arxiv.org/abs/2402.13758)

    在摘要一致性评估方面，该研究通过引入临床文本摘要的数据集TreatFact并对11个大语言模型进行评估，填补了关于大语言模型在摘要事实一致性评估方面的缺口。

    

    自动生成摘要中与源文件的事实不一致可能导致错误信息或带来风险。现有的事实一致性（FC）度量受到其性能、效率和可解释性的限制。大语言模型（LLMs）的最新进展在文本评估方面表现出卓越的潜力，但它们在评估摘要中的FC方面的效果仍未得到充分探讨。先前的研究主要集中在专有LLMs上，未探讨影响它们评估能力的重要因素。此外，当前的FC评估基准仅限于新闻文章，对在其上测试的FC方法的普遍性产生怀疑。在本文中，我们首先通过引入TreatFact数据集解决这一差距，该数据集包含由领域专家注释的临床文本的LLM生成摘要的FC。此外，我们在新闻和临床领域中为FC评估对比了11个LLMs，并分析了

    arXiv:2402.13758v1 Announce Type: new  Abstract: Factual inconsistency with source documents in automatically generated summaries can lead to misinformation or pose risks. Existing factual consistency(FC) metrics are constrained by their performance, efficiency, and explainability. Recent advances in Large language models (LLMs) have demonstrated remarkable potential in text evaluation but their effectiveness in assessing FC in summarisation remains underexplored. Prior research has mostly focused on proprietary LLMs, leaving essential factors that affect their assessment capabilities unexplored. Additionally, current FC evaluation benchmarks are restricted to news articles, casting doubt on the generality of the FC methods tested on them. In this paper, we first address the gap by introducing TreatFact a dataset of LLM-generated summaries of clinical texts, annotated for FC by domain experts. Moreover, we benchmark 11 LLMs for FC evaluation across news and clinical domains and analyse
    
[^27]: 将LLM上下文窗口扩展到超过2百万个标记的LongRoPE

    LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens

    [https://arxiv.org/abs/2402.13753](https://arxiv.org/abs/2402.13753)

    LongRoPE首次将预训练的LLM上下文窗口扩展至2048k个标记，通过关键创新实现了这一突破。

    

    大上下文窗口是大型语言模型（LLMs）中的一个理想特性。然而，由于高昂的微调成本、长文本稀缺以及新标记位置引入的灾难性值，当前的扩展上下文窗口仅限于约128k个标记。本文介绍了LongRoPE，首次将预训练的LLMs的上下文窗口扩展至令人印象深刻的2048k个标记，通过仅在256k训练长度内最多进行1k次微调步骤，同时保持在原始短上下文窗口下的性能。这是通过三项关键创新实现的：(i) 我们识别并利用了两种形式的位置插值中的非均匀性，通过高效搜索提供更好的微调初始化，并在非微调场景下实现8倍扩展；(ii) 我们引入了一种逐步扩展策略，首先微调256k长度的LLM，然后进行第二次位置插值。

    arXiv:2402.13753v1 Announce Type: new  Abstract: Large context window is a desirable feature in large language models (LLMs). However, due to high fine-tuning costs, scarcity of long texts, and catastrophic values introduced by new token positions, current extended context windows are limited to around 128k tokens. This paper introduces LongRoPE that, for the first time, extends the context window of pre-trained LLMs to an impressive 2048k tokens, with up to only 1k fine-tuning steps at within 256k training lengths, while maintaining performance at the original short context window. This is achieved by three key innovations: (i) we identify and exploit two forms of non-uniformities in positional interpolation through an efficient search, providing a better initialization for fine-tuning and enabling an 8x extension in non-fine-tuning scenarios; (ii) we introduce a progressive extension strategy that first fine-tunes a 256k length LLM and then conducts a second positional interpolation 
    
[^28]: 打破障碍：通过推理知识图利用大型语言模型进行工业推荐系统

    Breaking the Barrier: Utilizing Large Language Models for Industrial Recommendation Systems through an Inferential Knowledge Graph

    [https://arxiv.org/abs/2402.13750](https://arxiv.org/abs/2402.13750)

    提出了一种基于大型语言模型的互补知识增强推荐系统（LLM-KERec），通过引入实体提取器和构建互补知识图，解决了推荐系统难以捕捉用户意图转变和适应新商品的挑战。

    

    推荐系统在电子商务网站和在线平台中被广泛使用，以应对信息过载。然而，现有系统主要依赖历史数据和用户反馈，难以捕捉用户意图转变。最近，提出了基于知识库（KB）的模型来整合专家知识，但它们难以适应新商品和不断发展的电子商务环境。为了解决这些挑战，我们提出了一种新颖的基于大型语言模型的互补知识增强推荐系统（LLM-KERec）。它引入了一个实体提取器，从商品和用户信息中提取统一概念术语。为了提供具有成本效益且可靠的先验知识，根据实体的流行度和特定策略生成实体对。大型语言模型确定每个实体对中的互补关系，构建一个互补知识图。此外，一个新的...

    arXiv:2402.13750v1 Announce Type: cross  Abstract: Recommendation systems are widely used in e-commerce websites and online platforms to address information overload. However, existing systems primarily rely on historical data and user feedback, making it difficult to capture user intent transitions. Recently, Knowledge Base (KB)-based models are proposed to incorporate expert knowledge, but it struggle to adapt to new items and the evolving e-commerce environment. To address these challenges, we propose a novel Large Language Model based Complementary Knowledge Enhanced Recommendation System (LLM-KERec). It introduces an entity extractor that extracts unified concept terms from item and user information. To provide cost-effective and reliable prior knowledge, entity pairs are generated based on entity popularity and specific strategies. The large language model determines complementary relationships in each entity pair, constructing a complementary knowledge graph. Furthermore, a new 
    
[^29]: 使用表格提示解锁关系三元组抽取的上下文学习

    Unlocking Instructive In-Context Learning with Tabular Prompting for Relational Triple Extraction

    [https://arxiv.org/abs/2402.13741](https://arxiv.org/abs/2402.13741)

    设计了表格提示以解决关系三元组抽取中的提示设计和样本选择挑战。

    

    关系三元组抽取（RTE）的上下文学习（ICL）取得了令人满意的表现，但仍然面临两个关键挑战：（1）如何设计有效的提示和（2）如何选择适当的演示。然而，现有方法未能适当解决这些挑战。一方面，它们通常将RTE任务重新定义为文本-文本提示格式，这是不自然的，导致在预训练时的输出格式和大型语言模型（LLMs）的推断时间之间不匹配。另一方面，它们只利用表面自然语言特征，缺乏在样本选择中考虑三元组语义。这些问题阻碍了ICL对RTE性能的改进，因此我们旨在同时解决提示设计和样本选择挑战。为此，我们设计了一个用于RTE的表格提示（TableIE），将RTE任务构建成一个表格生成任务以解决上述挑战。

    arXiv:2402.13741v1 Announce Type: cross  Abstract: The in-context learning (ICL) for relational triple extraction (RTE) has achieved promising performance, but still encounters two key challenges: (1) how to design effective prompts and (2) how to select proper demonstrations. Existing methods, however, fail to address these challenges appropriately. On the one hand, they usually recast RTE task to text-to-text prompting formats, which is unnatural and results in a mismatch between the output format at the pre-training time and the inference time for large language models (LLMs). On the other hand, they only utilize surface natural language features and lack consideration of triple semantics in sample selection. These issues are blocking improved performance in ICL for RTE, thus we aim to tackle prompt designing and sample selection challenges simultaneously. To this end, we devise a tabular prompting for RTE (\textsc{TableIE}) which frames RTE task into a table generation task to inco
    
[^30]: 从文本到CQL：连接自然语言和语料库搜索引擎

    From Text to CQL: Bridging Natural Language and Corpus Search Engine

    [https://arxiv.org/abs/2402.13740](https://arxiv.org/abs/2402.13740)

    本文提出了首个旨在自动将自然语言转换为语料库查询语言（CQL）的文本到CQL任务，包括一个大规模数据集和利用大语言模型（LLMs）进行有效文本到CQL任务的方法。

    

    自然语言处理（NLP）技术已经彻底改变了我们与信息系统互动的方式，着重于将自然语言查询转换为诸如SQL之类的形式化查询语言。然而，对于语料库查询语言（CQL）这一在文本语料库中进行语言研究和详细分析的关键工具，却没有受到足够的重视。手动构建CQL查询是一项复杂且耗时的任务，需要相当多的专业知识，这对研究人员和从业者都构成了明显挑战。本文提出了第一个旨在自动将自然语言转换为CQL的文本到CQL任务。我们提出了一个全面的框架，包括一个经过特别策划的大规模数据集以及利用大语言模型（LLMs）进行有效文本到CQL任务的方法。此外，我们建立了先进的评估指标，以评估这一同步过程。

    arXiv:2402.13740v1 Announce Type: new  Abstract: Natural Language Processing (NLP) technologies have revolutionized the way we interact with information systems, with a significant focus on converting natural language queries into formal query languages such as SQL. However, less emphasis has been placed on the Corpus Query Language (CQL), a critical tool for linguistic research and detailed analysis within text corpora. The manual construction of CQL queries is a complex and time-intensive task that requires a great deal of expertise, which presents a notable challenge for both researchers and practitioners. This paper presents the first text-to-CQL task that aims to automate the translation of natural language into CQL. We present a comprehensive framework for this task, including a specifically curated large-scale dataset and methodologies leveraging large language models (LLMs) for effective text-to-CQL task. In addition, we established advanced evaluation metrics to assess the syn
    
[^31]: 大型预训练语言模型的达芬奇密码：解读退化知识神经元

    The Da Vinci Code of Large Pre-trained Language Models: Deciphering Degenerate Knowledge Neurons

    [https://arxiv.org/abs/2402.13731](https://arxiv.org/abs/2402.13731)

    本研究提供了对预训练语言模型中退化知识神经元（DKNs）的全面定义，引入了神经拓扑聚类方法和神经退化分析框架，从而实现更准确的DKN获取。

    

    本研究探讨了预训练语言模型（PLMs）中事实知识存储的机制。先前的研究表明，事实知识存储在多层感知器权重中，某些存储单元表现出退化性，称为退化知识神经元（Degenerate Knowledge Neurons, DKNs）。本文提供了一个涵盖结构和功能方面的DKNs全面定义，开创了对PLMs事实知识存储单元结构的研究。基于此，我们引入了神经拓扑聚类方法，该方法允许形成任意数量和结构的DKNs，从而实现更准确的DKN获取。此外，我们引入了神经退化分析框架，独特地整合了模型鲁棒性、可进化性和复杂性，以对PLMs进行全面评估。在该框架内，我们执行了34个实验，跨越2个PLMs、4个数据集和6个设置。

    arXiv:2402.13731v1 Announce Type: cross  Abstract: This study explores the mechanism of factual knowledge storage in pre-trained language models (PLMs). Previous research suggests that factual knowledge is stored within multi-layer perceptron weights, and some storage units exhibit degeneracy, referred to as Degenerate Knowledge Neurons (DKNs). This paper provides a comprehensive definition of DKNs that covers both structural and functional aspects, pioneering the study of structures in PLMs' factual knowledge storage units. Based on this, we introduce the Neurological Topology Clustering method, which allows the formation of DKNs in any numbers and structures, leading to a more accurate DKN acquisition. Furthermore, we introduce the Neuro-Degeneracy Analytic Analysis Framework, which uniquely integrates model robustness, evolvability, and complexity for a holistic assessment of PLMs. Within this framework, our execution of 34 experiments across 2 PLMs, 4 datasets, and 6 settings highl
    
[^32]: 利用自适应上下文掩码进行面向方面的情感分析

    Exploiting Adaptive Contextual Masking for Aspect-Based Sentiment Analysis

    [https://arxiv.org/abs/2402.13722](https://arxiv.org/abs/2402.13722)

    提出了一种利用自适应掩码方法来协助面向方面的情感分析中的方面术语提取和情感分类子任务的新方法

    

    面向方面的情感分析（ABSA）是一个细粒度的语言学问题，涉及从给定文本中提取多方面的方面、意见和情感。本文提出了一种自适应掩码方法，根据上下文去除无关标记，以帮助ABSA的Aspect Term Extraction和Aspect Sentiment Classification子任务。

    arXiv:2402.13722v1 Announce Type: new  Abstract: Aspect-Based Sentiment Analysis (ABSA) is a fine-grained linguistics problem that entails the extraction of multifaceted aspects, opinions, and sentiments from the given text. Both standalone and compound ABSA tasks have been extensively used in the literature to examine the nuanced information present in online reviews and social media posts. Current ABSA methods often rely on static hyperparameters for attention-masking mechanisms, which can struggle with context adaptation and may overlook the unique relevance of words in varied situations. This leads to challenges in accurately analyzing complex sentences containing multiple aspects with differing sentiments. In this work, we present adaptive masking methods that remove irrelevant tokens based on context to assist in Aspect Term Extraction and Aspect Sentiment Classification subtasks of ABSA. We show with our experiments that the proposed methods outperform the baseline methods in te
    
[^33]: Ouroboros: 大模型增强草案的猜测解码技术

    Ouroboros: Speculative Decoding with Large Model Enhanced Drafting

    [https://arxiv.org/abs/2402.13720](https://arxiv.org/abs/2402.13720)

    Ouroboros通过构建短小草案并引入候选短语池的方法提高了大语言模型推理的加速效率

    

    通过构建短小高效的小模型起草草案，然后要求大语言模型以无自回归方式进行验证和修正，以最小化时间开销。当验证后可以生成更长的草稿，但也会导致相当大的尝试和错误成本。由于高验证失败概率，现有解码方法不能一次起草太多内容进行验证，实现次优的推理加速。

    arXiv:2402.13720v1 Announce Type: new  Abstract: Drafting-then-verifying decoding methods such as speculative decoding are widely adopted training-free methods to accelerate the inference of large language models (LLMs). Instead of employing an autoregressive process to decode tokens sequentially, speculative decoding initially creates drafts with an efficient small model. Then LLMs are required to conduct verification and correction in a non-autoregressive fashion to minimize time overhead. Generating longer drafts can lead to even more significant speedups once verified, but also incurs substantial trial and error costs if it fails. Suffering from the high verification failure probability, existing decoding methods cannot draft too much content for verification at one time, achieving sub-optimal inference acceleration. In this paper, we introduce Ouroboros, which constructs a phrase candidate pool from the verification process of LLMs to provide candidates for draft generation of the
    
[^34]: $\infty$Bench: 将长上下文评估扩展至超过10万令牌

    $\infty$Bench: Extending Long Context Evaluation Beyond 100K Tokens

    [https://arxiv.org/abs/2402.13718](https://arxiv.org/abs/2402.13718)

    提出了$\infty$Bench，第一个以平均数据长度超过10万个令牌的LLM基准，用于评估处理长上下文的能力

    

    处理和推理长上下文对于大型语言模型（LLMs）的许多实际应用至关重要，如文档理解和代理构建。本文提出$\infty$Bench，第一个LLM基准，平均数据长度超过10万个令牌。$\infty$Bench包含涵盖不同领域的合成和现实任务，以英文和中文呈现。$\infty$Bench中的任务旨在需要深刻理解上下文中的长依赖性，并且简单地从上下文中检索有限数量的段落对于这些任务来说是不够的。

    arXiv:2402.13718v1 Announce Type: new  Abstract: Processing and reasoning over long contexts is crucial for many practical applications of Large Language Models (LLMs), such as document comprehension and agent construction. Despite recent strides in making LLMs process contexts with more than 100K tokens, there is currently a lack of a standardized benchmark to evaluate this long-context capability. Existing public benchmarks typically focus on contexts around 10K tokens, limiting the assessment and comparison of LLMs in processing longer contexts. In this paper, we propose $\infty$Bench, the first LLM benchmark featuring an average data length surpassing 100K tokens. $\infty$Bench comprises synthetic and realistic tasks spanning diverse domains, presented in both English and Chinese. The tasks in $\infty$Bench are designed to require well understanding of long dependencies in contexts, and make simply retrieving a limited number of passages from contexts not sufficient for these tasks
    
[^35]: Neeko：利用动态LoRA实现高效多角色扮演代理

    Neeko: Leveraging Dynamic LoRA for Efficient Multi-Character Role-Playing Agent

    [https://arxiv.org/abs/2402.13717](https://arxiv.org/abs/2402.13717)

    Neeko利用动态低秩适配器（LoRA）策略，有效处理多角色扮演过程中的挑战，提升了对不同属性、个性和说话模式的适应能力。

    

    大型语言模型（LLMs）在开放领域对话代理程序中起着革命性作用，但在多角色扮演（MCRP）场景中遇到挑战。为了解决这个问题，我们提出了Neeko，这是一个专为高效模仿多个角色而设计的创新框架。与现有方法不同，Neeko采用动态低秩适配器（LoRA）策略，使其能够无缝适应不同的角色。我们的框架将角色扮演过程分解为代理预训练、多个角色扮演和角色增量学习，有效处理已知和未知角色。这种动态方法，结合为每个角色设计的独特LoRA块，增强了Neeko对独特属性、个性和说话模式的适应能力。因此，Neeko在MCRP方面表现出比大多数现有方法更出色的性能，为用户提供更具吸引力和多样化的互动体验。代码和数据可在（链接中提供）。

    arXiv:2402.13717v1 Announce Type: new  Abstract: Large Language Models (LLMs) have revolutionized open-domain dialogue agents but encounter challenges in multi-character role-playing (MCRP) scenarios. To address the issue, we present Neeko, an innovative framework designed for efficient multiple characters imitation. Unlike existing methods, Neeko employs a dynamic low-rank adapter (LoRA) strategy, enabling it to adapt seamlessly to diverse characters. Our framework breaks down the role-playing process into agent pre-training, multiple characters playing, and character incremental learning, effectively handling both seen and unseen roles. This dynamic approach, coupled with distinct LoRA blocks for each character, enhances Neeko's adaptability to unique attributes, personalities, and speaking patterns. As a result, Neeko demonstrates superior performance in MCRP over most existing methods, offering more engaging and versatile user interaction experiences. Code and data are available at
    
[^36]: SaGE：评估大型语言模型的道德一致性

    SaGE: Evaluating Moral Consistency in Large Language Models

    [https://arxiv.org/abs/2402.13709](https://arxiv.org/abs/2402.13709)

    提出SaGE方法，通过语义图熵来衡量大型语言模型道德一致性，构建了MCC语料库。

    

    尽管最近展示出大型语言模型（LLMs）在会话系统中的印象深刻能力，但我们表明即使是最先进的LLMs在生成过程中也存在道德不一致，对其可靠性（以及总体可信赖性）提出了质疑。以往在LLM评估领域的工作侧重于开发地面真实数据，以衡量在特定任务上的准确性。然而，对于道德情景往往缺乏普遍认同答案的情况，模型响应的一致性对于其可靠性变得至关重要。为了解决这一问题，我们提出了一种信息理论度量方法，称为语义图熵（SaGE），基于“经验法则”（RoTs）的概念来衡量模型的道德一致性。RoTs是模型学习到的抽象原则，可有效帮助解释其决策策略。在此基础上，我们构建了道德一致性语料库（MCC），包含50K个道德问题、回答。

    arXiv:2402.13709v1 Announce Type: cross  Abstract: Despite recent advancements showcasing the impressive capabilities of Large Language Models (LLMs) in conversational systems, we show that even state-of-the-art LLMs are morally inconsistent in their generations, questioning their reliability (and trustworthiness in general). Prior works in LLM evaluation focus on developing ground-truth data to measure accuracy on specific tasks. However, for moral scenarios that often lack universally agreed-upon answers, consistency in model responses becomes crucial for their reliability. To address this issue, we propose an information-theoretic measure called Semantic Graph Entropy (SaGE), grounded in the concept of "Rules of Thumb" (RoTs) to measure a model's moral consistency. RoTs are abstract principles learned by a model and can help explain their decision-making strategies effectively. To this extent, we construct the Moral Consistency Corpus (MCC), containing 50K moral questions, responses
    
[^37]: 调查多语言教学调整：多语模型是否需要多语教学？

    Investigating Multilingual Instruction-Tuning: Do Polyglot Models Demand for Multilingual Instructions?

    [https://arxiv.org/abs/2402.13703](https://arxiv.org/abs/2402.13703)

    本研究是第一个对多语模型在不同印欧语言上的性能进行了广泛研究，发现在并行教学调整数据集上进行教学调整可以显著提升跨语言遵循能力，同时提出了对表面对齐假设的质疑

    

    arXiv:2402.13703v1 公告类型：新摘要：将多语言预训练大型语言模型（LLMs）转化为雄辩而有用的助手对促进它们在不同语言地区的使用至关重要。基于这一精神，我们是第一个对跨多种印欧语言进行大规模研究的研究者，旨在研究多语模型在选择的最常用的印欧语言上的并行、多轮教学调整基准测试的性能。我们系统地研究了语言和教学数据集大小对中型多语言LLM的影响，通过在并行教学调整数据集上进行教学调整。我们的结果表明，在并行教学调整而不是单语语料库上进行教学调整可以使跨语言遵循能力提高多达4.6%。此外，我们表明，表面对齐假设通常不成立，因为所调查的多语7B参数模型是一个反例，需要大规模的教学调整。

    arXiv:2402.13703v1 Announce Type: new  Abstract: The adaption of multilingual pre-trained Large Language Models (LLMs) into eloquent and helpful assistants is essential to facilitate their use across different language regions. In that spirit, we are the first to conduct an extensive study of the performance of multilingual models on parallel, multi-turn instruction-tuning benchmarks across a selection of the most-spoken Indo-European languages. We systematically examine the effects of language and instruction dataset size on a mid-sized, multilingual LLM by instruction-tuning it on parallel instruction-tuning datasets. Our results demonstrate that instruction-tuning on parallel instead of monolingual corpora benefits cross-lingual instruction following capabilities by up to 4.6%. Furthermore, we show that the Superficial Alignment Hypothesis does not hold in general, as the investigated multilingual 7B parameter model presents a counter-example requiring large-scale instruction-tuning
    
[^38]: 基于社交媒体的中文多模态实体识别数据集CMNER

    CMNER: A Chinese Multimodal NER Dataset based on Social Media

    [https://arxiv.org/abs/2402.13693](https://arxiv.org/abs/2402.13693)

    本研究在中国最大的社交媒体平台微博上构建了一个中文多模态实体识别数据集（CMNER），包含5,000条微博帖子和18,326张对应图片，并展示了将图片纳入NER任务中的有效性。

    

    多模态命名实体识别（MNER）是一项旨在从文本中提取命名实体的关键任务，其通过相关图片提供支持。然而，在中文MNER领域，数据明显不足，严重阻碍了该自然语言处理任务的进展。因此，在本研究中，我们利用来自微博的数据编制了一个中文多模态实体识别数据集（CMNER）。我们的数据集包括5,000条微博帖子，配对18,326张对应的图片。实体被分类为四个不同类别：人物、地点、组织和杂项。我们在CMNER上进行了基线实验，结果强调了将图片纳入NER中的有效性。此外，我们在公开的英文MNER数据集（Twitter2015）上进行了跨语言实验，结果证实了我们的假设，即中文和英文之间存在一致性。

    arXiv:2402.13693v1 Announce Type: new  Abstract: Multimodal Named Entity Recognition (MNER) is a pivotal task designed to extract named entities from text with the support of pertinent images. Nonetheless, a notable paucity of data for Chinese MNER has considerably impeded the progress of this natural language processing task within the Chinese domain. Consequently, in this study, we compile a Chinese Multimodal NER dataset (CMNER) utilizing data sourced from Weibo, China's largest social media platform. Our dataset encompasses 5,000 Weibo posts paired with 18,326 corresponding images. The entities are classified into four distinct categories: person, location, organization, and miscellaneous. We perform baseline experiments on CMNER, and the outcomes underscore the effectiveness of incorporating images for NER. Furthermore, we conduct cross-lingual experiments on the publicly available English MNER dataset (Twitter2015), and the results substantiate our hypothesis that Chinese and Eng
    
[^39]: KInIT参加SemEval-2024任务8：针对多语言机器生成文本检测的微调LLMs

    KInIT at SemEval-2024 Task 8: Fine-tuned LLMs for Multilingual Machine-Generated Text Detection

    [https://arxiv.org/abs/2402.13671](https://arxiv.org/abs/2402.13671)

    本论文针对SemEval-2024任务8提出了一种使用微调LLMs进行多语言机器生成文本检测的方法，通过多种方式处理该任务并将统计检测指标与模型预测相结合，取得了竞争性结果。

    

    arXiv:2402.13671v1 发表类型：交叉传播 摘要：SemEval-2024任务8侧重于多生成器、多领域和多语言的黑盒机器生成文本检测。这样的检测对于防止大型语言模型（LLMs）的潜在滥用非常重要，其中最新的LLMs非常擅长生成多语言的类似人类的文本。我们以多种方式处理了这个任务，利用语言识别和参数高效微调较小的LLMs进行文本分类。我们进一步使用每种语言的分类阈值校准，将微调的模型预测与统计检测指标独特结合，以提高系统检测性能的泛化。我们提交的方法取得了竞争性的结果，排名第四，仅落后于获胜者不到1个百分点。

    arXiv:2402.13671v1 Announce Type: cross  Abstract: SemEval-2024 Task 8 is focused on multigenerator, multidomain, and multilingual black-box machine-generated text detection. Such a detection is important for preventing a potential misuse of large language models (LLMs), the newest of which are very capable in generating multilingual human-like texts. We have coped with this task in multiple ways, utilizing language identification and parameter-efficient fine-tuning of smaller LLMs for text classification. We have further used the per-language classification-threshold calibration to uniquely combine fine-tuned models predictions with statistical detection metrics to improve generalization of the system detection performance. Our submitted method achieved competitive results, ranking at the fourth place, just under 1 percentage point behind the winner.
    
[^40]: 自蒸馏桥接语言模型微调中的分布差距

    Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning

    [https://arxiv.org/abs/2402.13669](https://arxiv.org/abs/2402.13669)

    SDFT是一种通过用模型本身生成的精简数据集来桥接分布差距的新方法，在缓解灾难性遗忘的同时，在下游任务上实现了与普通微调相媲美甚至更优越的性能

    

    大型语言模型（LLMs）的兴起彻底改变了自然语言处理，但将它们微调为特定任务常常面临在平衡性能和保留一般指示遵循能力之间的挑战。在本文中，我们认为任务数据集与LLMs之间的分布差距是主要的潜在原因。为了解决这一问题，我们引入了自蒸馏微调（SDFT），这是一种通过用模型本身生成的精简数据集引导微调以匹配其原始分布来桥接分布差距的新方法。在各种基准测试上对Llama-2-chat模型的实验结果表明，SDFT有效地缓解了灾难性遗忘，同时在下游任务上实现了与普通微调相媲美甚至更优越的性能。此外，SDFT展示了保持LLMs的有益性和安全对齐的潜力

    arXiv:2402.13669v1 Announce Type: new  Abstract: The surge in Large Language Models (LLMs) has revolutionized natural language processing, but fine-tuning them for specific tasks often encounters challenges in balancing performance and preserving general instruction-following abilities. In this paper, we posit that the distribution gap between task datasets and the LLMs serves as the primary underlying cause. To address the problem, we introduce Self-Distillation Fine-Tuning (SDFT), a novel approach that bridges the distribution gap by guiding fine-tuning with a distilled dataset generated by the model itself to match its original distribution. Experimental results on the Llama-2-chat model across various benchmarks demonstrate that SDFT effectively mitigates catastrophic forgetting while achieving comparable or superior performance on downstream tasks compared to the vanilla fine-tuning. Moreover, SDFT demonstrates the potential to maintain the helpfulness and safety alignment of LLMs
    
[^41]: GCOF：使用大型语言模型进行自我迭代文本生成以进行文案撰写

    GCOF: Self-iterative Text Generation for Copywriting Using Large Language Model

    [https://arxiv.org/abs/2402.13667](https://arxiv.org/abs/2402.13667)

    GCOF框架结合遗传算法和大型语言模型，实现了自我迭代优化，生成的文案在点击率上相比人工编辑的文案平均提高了50%以上。

    

    大型语言模型（LLM）如ChatGPT极大地简化了市场文案的生成，但产生满足特定领域要求的内容，比如有效地吸引客户，仍然是一个重大挑战。本文介绍了旨在增强市场文案创作效率和吸引力的遗传文案优化框架（GCOF）。我们在LLM提示的显式特征工程中进行操作。此外，我们修改了遗传算法（GA）中的交叉操作符，并将其整合到GCOF中，以实现自动特征工程。这种整合促进了市场文案的自我迭代优化。在线结果表明，我们框架生成的文案相对于人工编辑的文案，在点击率（CTR）上平均增加了50%以上。

    arXiv:2402.13667v1 Announce Type: new  Abstract: Large language models(LLM) such as ChatGPT have substantially simplified the generation of marketing copy, yet producing content satisfying domain specific requirements, such as effectively engaging customers, remains a significant challenge. In this work, we introduce the Genetic Copy Optimization Framework (GCOF) designed to enhance both efficiency and engagememnt of marketing copy creation. We conduct explicit feature engineering within the prompts of LLM. Additionally, we modify the crossover operator in Genetic Algorithm (GA), integrating it into the GCOF to enable automatic feature engineering. This integration facilitates a self-iterative refinement of the marketing copy. Compared to human curated copy, Online results indicate that copy produced by our framework achieves an average increase in click-through rate (CTR) of over $50\%$.
    
[^42]: 大型语言模型对齐的隐私保护指南

    Privacy-Preserving Instructions for Aligning Large Language Models

    [https://arxiv.org/abs/2402.13659](https://arxiv.org/abs/2402.13659)

    提出使用合成指南替换真实指南以增强隐私保护，并通过私密微调生成器生成此类合成指南，并通过新颖的过滤算法使合成指南的分布与真实指南一致，展示了在大型语言模型对齐中的高效用性。

    

    大型语言模型（LLM）应用的服务提供商在野外收集用户指南，并在进一步对齐LLM与用户意图中使用这些指南。这些潜在包含敏感信息的指南在流程中由人工工作者标注。这带来了新的隐私风险，而Typical Private Optimization没有解决这个问题。为此，我们提议使用合成指南替换数据标注和模型微调中的真实指南。通过使用私密微调生成器生成这些合成指南，可以确保形式差异隐私。在实现所需效用方面至关重要的是我们的新颖过滤算法，将合成指南的分布与实际指南的分布进行匹配。在有人反馈的受监督微调和强化学习中，我们的广泛实验表明，通过展示合成指南的最终集合的高效用性

    arXiv:2402.13659v1 Announce Type: cross  Abstract: Service providers of large language model (LLM) applications collect user instructions in the wild and use them in further aligning LLMs with users' intentions. These instructions, which potentially contain sensitive information, are annotated by human workers in the process. This poses a new privacy risk not addressed by the typical private optimization. To this end, we propose using synthetic instructions to replace real instructions in data annotation and model fine-tuning. Formal differential privacy is guaranteed by generating those synthetic instructions using privately fine-tuned generators. Crucial in achieving the desired utility is our novel filtering algorithm that matches the distribution of the synthetic instructions to that of the real ones. In both supervised fine-tuning and reinforcement learning from human feedback, our extensive experiments demonstrate the high utility of the final set of synthetic instructions by sho
    
[^43]: 通过LLMs和注意力遮罩完成无监督文本风格转移与多路交互

    Unsupervised Text Style Transfer via LLMs and Attention Masking with Multi-way Interactions

    [https://arxiv.org/abs/2402.13647](https://arxiv.org/abs/2402.13647)

    通过组合注意力遮罩方法和大型语言模型，提出多种交互方式，可以改进无监督文本风格转移任务。

    

    无监督文本风格转移（UTST）已成为自然语言处理（NLP）领域中的一个关键任务，旨在将句子的一种风格方面转换为另一种风格，同时不改变其语义、句法或其他属性。本文探讨了如何有效结合注意力遮罩方法和大型语言模型（LLMs），提出了四种交互方式：具有调整顺序的流水线框架；知识蒸馏从LLMs到注意力遮罩模型；使用构建的并行示例进行上下文学习。我们实验证明这些多路交互可以改进性

    arXiv:2402.13647v1 Announce Type: cross  Abstract: Unsupervised Text Style Transfer (UTST) has emerged as a critical task within the domain of Natural Language Processing (NLP), aiming to transfer one stylistic aspect of a sentence into another style without changing its semantics, syntax, or other attributes. This task is especially challenging given the intrinsic lack of parallel text pairings. Among existing methods for UTST tasks, attention masking approach and Large Language Models (LLMs) are deemed as two pioneering methods. However, they have shortcomings in generating unsmooth sentences and changing the original contents, respectively. In this paper, we investigate if we can combine these two methods effectively. We propose four ways of interactions, that are pipeline framework with tuned orders; knowledge distillation from LLMs to attention masking model; in-context learning with constructed parallel examples. We empirically show these multi-way interactions can improve the ba
    
[^44]: 用于评估视觉-语言模型中性别偏见的统一框架和数据集

    A Unified Framework and Dataset for Assessing Gender Bias in Vision-Language Models

    [https://arxiv.org/abs/2402.13636](https://arxiv.org/abs/2402.13636)

    构建了一个统一框架和数据集，用于评估视觉-语言模型中的性别偏见，并观察到不同的输入-输出模式导致不同的偏见大小和方向。

    

    大型视觉-语言模型（VLMs）广泛应用于工业和学术界。本研究构建了一个统一框架，系统评估VLMs中的性别-职业偏见。我们的评估涵盖了最近VLMs支持的所有推断模式，包括图像到文本、文本到文本、文本到图像和图像到图像。我们构建了一个合成的、高质量的文本和图像数据集，模糊了职业动作中的性别差异，以评估性别偏见。在我们对最近的视觉-语言模型（VLMs）进行基准测试时，我们观察到不同的输入-输出模式会导致不同的偏见大小和方向。我们希望我们的工作能够指导未来改进VLMs以学习社会无偏见表示。我们将发布我们的数据和代码。

    arXiv:2402.13636v1 Announce Type: cross  Abstract: Large vision-language models (VLMs) are widely getting adopted in industry and academia. In this work we build a unified framework to systematically evaluate gender-profession bias in VLMs. Our evaluation encompasses all supported inference modes of the recent VLMs, including image-to-text, text-to-text, text-to-image, and image-to-image. We construct a synthetic, high-quality dataset of text and images that blurs gender distinctions across professional actions to benchmark gender bias. In our benchmarking of recent vision-language models (VLMs), we observe that different input-output modalities result in distinct bias magnitudes and directions. We hope our work will help guide future progress in improving VLMs to learn socially unbiased representations. We will release our data and code.
    
[^45]: MORE: 多模态检索增强生成式常识推理

    MORE: Multi-mOdal REtrieval Augmented Generative Commonsense Reasoning

    [https://arxiv.org/abs/2402.13625](https://arxiv.org/abs/2402.13625)

    提出了一种新颖的多模态检索（MORE）增强框架，利用文本和图像来提升语言模型的常识能力。

    

    自然语言模型在预训练过程中往往难以学习足够的常识知识，因为常识信息的记录频率明显低于其存在频率。为了增强模型的常识能力，一些研究利用文本检索进行了改进。不同于文本，图像固有地包含常识信息，但很少有研究致力于有效利用它们。本文提出了一种新颖的多模态检索（MORE）增强框架，利用文本和图像来提升语言模型的常识能力。在Common-Gen任务上进行的大量实验表明，基于单一模态和多模态预训练模型的MORE的有效性。

    arXiv:2402.13625v1 Announce Type: new  Abstract: Since commonsense information has been recorded significantly less frequently than its existence, language models pre-trained by text generation have difficulty to learn sufficient commonsense knowledge. Several studies have leveraged text retrieval to augment the models' commonsense ability. Unlike text, images capture commonsense information inherently but little effort has been paid to effectively utilize them. In this work, we propose a novel Multi-mOdal REtrieval (MORE) augmentation framework, to leverage both text and images to enhance the commonsense ability of language models. Extensive experiments on the Common-Gen task have demonstrated the efficacy of MORE based on the pre-trained models of both single and multiple modalities.
    
[^46]: FLAME：使用大型语言模型进行自监督低资源分类扩展

    FLAME: Self-Supervised Low-Resource Taxonomy Expansion using Large Language Models

    [https://arxiv.org/abs/2402.13623](https://arxiv.org/abs/2402.13623)

    提出了一种使用大型语言模型进行自监督低资源分类扩展的方法 FLAME

    

    分类法代表了一种树状层次结构，建立了实体间的关系以在特定领域内传达知识。分类法中的每个边代表了一个上位词-下位词关系。分类法在各种现实世界的应用中很有用，如电子商务搜索引擎和推荐系统。因此，需要随着时间的推移增强这些分类法。然而，由于可用人力资源的限制和数据的指数增长，手动筛选具有最新数据的分类法存在挑战。因此，开发自动分类法扩展方法变得迫在眉睫。传统的监督分类法扩展方法由于现有分类法规模较小，导致训练数据有限，经常会导致过拟合。在本文中，我们提出了FLAME，一种用于分类法扩展的新方法。

    arXiv:2402.13623v1 Announce Type: new  Abstract: Taxonomies represent an arborescence hierarchical structure that establishes relationships among entities to convey knowledge within a specific domain. Each edge in the taxonomy signifies a hypernym-hyponym relationship. Taxonomies find utility in various real-world applications, such as e-commerce search engines and recommendation systems. Consequently, there arises a necessity to enhance these taxonomies over time. However, manually curating taxonomies with neoteric data presents challenges due to limitations in available human resources and the exponential growth of data. Therefore, it becomes imperative to develop automatic taxonomy expansion methods. Traditional supervised taxonomy expansion approaches encounter difficulties stemming from limited resources, primarily due to the small size of existing taxonomies. This scarcity of training data often leads to overfitting. In this paper, we propose FLAME, a novel approach for taxonomy 
    
[^47]: VLSP 2023综述--ComOM任务：越南产品评论的比较意见挖掘数据挑战

    Overview of the VLSP 2023 -- ComOM Shared Task: A Data Challenge for Comparative Opinion Mining from Vietnamese Product Reviews

    [https://arxiv.org/abs/2402.13613](https://arxiv.org/abs/2402.13613)

    该论文总结了VLSP 2023中ComOM任务的一个数据挑战，旨在推动自然语言处理领域通过开发从越南产品评论中提取比较意见的技术，参与者需提出能够提取比较"五元组"的模型并根据F1分数进行评估排名。

    

    本文提供了越南语产品评论比较意见挖掘共享任务（ComOM）的综合概述，该任务作为第十届越南语言和语音处理国际研讨会（VLSP 2023）的一部分举行。此共享任务的主要目标是通过开发能够有效从越南产品评论中提取比较意见的技术来推动自然语言处理领域的发展。参与者被挑战提出能够从比较句中熟练提取比较“五元组”的模型，包括主题、客体、方面、谓词和比较类型标签。我们构建了一个包含120个文档的人工标记数据集，其中包括7427个非比较句和1798个句子中的2468个比较。参与的模型将根据准确匹配宏平均的五元组F1分数进行评估和排名。

    arXiv:2402.13613v1 Announce Type: new  Abstract: This paper presents a comprehensive overview of the Comparative Opinion Mining from Vietnamese Product Reviews shared task (ComOM), held as part of the 10$^{th}$ International Workshop on Vietnamese Language and Speech Processing (VLSP 2023). The primary objective of this shared task is to advance the field of natural language processing by developing techniques that proficiently extract comparative opinions from Vietnamese product reviews. Participants are challenged to propose models that adeptly extract a comparative "quintuple" from a comparative sentence, encompassing Subject, Object, Aspect, Predicate, and Comparison Type Label. We construct a human-annotated dataset comprising $120$ documents, encompassing $7427$ non-comparative sentences and $2468$ comparisons within $1798$ sentences. Participating models undergo evaluation and ranking based on the Exact match macro-averaged quintuple F1 score.
    
[^48]: 数据驱动的大型生成模型在科学发现中的应用

    Data-driven Discovery with Large Generative Models

    [https://arxiv.org/abs/2402.13610](https://arxiv.org/abs/2402.13610)

    大型生成模型在数据驱动发现中的应用开创了端到端发现系统的新模式，利用提供的数据集搜寻和验证假设，突显了自动化系统的重要性和局限性。

    

    随着数据以前所未有的速度累积，它作为促进科学发现的潜力呈指数增长。这篇立场论文敦促机器学习（ML）社区利用大型生成模型（LGMs）的能力，开发自动化系统用于端到端的数据驱动发现 -- 一种范式，从所提供的数据集中纯粹搜索和验证假设，而无需额外的数据收集或物理实验。我们首先概述了理想数据驱动发现系统的几个期望条件。然后，通过使用GPT-4的DATAVOYAGER作为概念验证，我们展示了LGMs如何实现几项这些期望条件 -- 这是以前无法做到的成就 -- 同时也突显了当前系统中的重要局限性，从而为开展新型机器学习研究提供了机遇。

    arXiv:2402.13610v1 Announce Type: cross  Abstract: With the accumulation of data at an unprecedented rate, its potential to fuel scientific discovery is growing exponentially. This position paper urges the Machine Learning (ML) community to exploit the capabilities of large generative models (LGMs) to develop automated systems for end-to-end data-driven discovery -- a paradigm encompassing the search and verification of hypotheses purely from a set of provided datasets, without the need for additional data collection or physical experiments. We first outline several desiderata for an ideal data-driven discovery system. Then, through DATAVOYAGER, a proof-of-concept utilizing GPT-4, we demonstrate how LGMs fulfill several of these desiderata -- a feat previously unattainable -- while also highlighting important limitations in the current system that open up opportunities for novel ML research. We contend that achieving accurate, reliable, and robust end-to-end discovery systems solely th
    
[^49]: 对大型语言模型的多语言置信度评估进行全面研究

    A Comprehensive Study of Multilingual Confidence Estimation on Large Language Models

    [https://arxiv.org/abs/2402.13606](https://arxiv.org/abs/2402.13606)

    该论文介绍了对大型语言模型的多语言置信度评估的全面研究，提出了一个专业多语言问答数据集，并研究了这些置信度分数如何增强模型性能，最终提出了一种跨语言置信度估计方法。

    

    大型语言模型生成幻觉并在预测中表现过于自信的倾向引发了人们对其可靠性的担忧。表明模型响应的可信度或不确定性估计对于开发可靠的人工智能系统至关重要。目前的研究主要集中在英语中LLM的置信度估计上，在其他广泛使用的语言方面仍存在空白，阻碍了可靠AI应用的全球发展。本文介绍了对LLM上的多语言置信度评估（MlingConf）的全面调查。首先，我们引入了一个经过详细检查的专业多语言问答数据集。其次，我们深入研究置信度估计的性能，并研究这些置信度分数如何通过跨不同语言的自我完善来增强LLM的性能。最后，我们提出了一种跨语言置信度估计方法，以实现更精确的估计。

    arXiv:2402.13606v1 Announce Type: new  Abstract: The tendency of Large Language Models to generate hallucinations and exhibit overconfidence in predictions raises concerns regarding their reliability. Confidence or uncertainty estimations indicating the extent of trustworthiness of a model's response are essential to developing reliable AI systems. Current research primarily focuses on LLM confidence estimations in English, remaining a void for other widely used languages and impeding the global development of reliable AI applications. This paper introduces a comprehensive investigation of Multi-lingual confidence estimation (MlingConf) on LLMs. First, we introduce an elaborated and expert-checked multilingual QA dataset. Second, we delve into the performance of confidence estimations and examine how these confidence scores can enhance LLM performance through self-refinement across diverse languages. Finally, we propose a cross-lingual confidence estimation method to achieve more preci
    
[^50]: KorNAT：韩国社会价值观和常识的LLM对齐基准

    KorNAT: LLM Alignment Benchmark for Korean Social Values and Common Knowledge

    [https://arxiv.org/abs/2402.13605](https://arxiv.org/abs/2402.13605)

    KorNAT是首个用于评估韩国国家对齐的基准，包括社会价值观和常识对齐两个方面，通过对社会价值和常识多项选择题的测试来评估模型的对齐程度。

    

    对于大型语言模型（LLMs）在特定国家得以有效部署，它们必须具有对该国文化和基本知识的理解。为此，我们引入了国家对齐（National Alignment），从社会价值观对齐和常识对齐两个方面衡量LLM与目标国家之间的对齐。社会价值观对齐评估模型对特定国家社会价值观的理解程度，而常识对齐则检验模型对相关基本国家知识的把握情况。我们构建了KorNAT，这是首个衡量与韩国国家对齐的基准。对于社会价值数据集，我们从包括6174名韩国参与者在内的大规模调查中获得了地面真实标签。对于常识数据集，我们基于韩国教科书和GED参考资料构建了样本。KorNAT包含4K和6K个针对社会价值和常识的多项选择题。

    arXiv:2402.13605v1 Announce Type: new  Abstract: For Large Language Models (LLMs) to be effectively deployed in a specific country, they must possess an understanding of the nation's culture and basic knowledge. To this end, we introduce National Alignment, which measures an alignment between an LLM and a targeted country from two aspects: social value alignment and common knowledge alignment. Social value alignment evaluates how well the model understands nation-specific social values, while common knowledge alignment examines how well the model captures basic knowledge related to the nation. We constructed KorNAT, the first benchmark that measures national alignment with South Korea. For the social value dataset, we obtained ground truth labels from a large-scale survey involving 6,174 unique Korean participants. For the common knowledge dataset, we constructed samples based on Korean textbooks and GED reference materials. KorNAT contains 4K and 6K multiple-choice questions for socia
    
[^51]: 打破HISCO障碍：使用OccCANINE进行自动职业标准化

    Breaking the HISCO Barrier: Automatic Occupational Standardization with OccCANINE

    [https://arxiv.org/abs/2402.13604](https://arxiv.org/abs/2402.13604)

    通过OccCANINE工具，我们成功打破了HISCO障碍，实现了自动化职业标准化，从而大大简化了对职业描述的处理和分类过程，为经济学、经济历史等领域的职业结构分析提供了高效且准确的数据。

    

    这篇论文介绍了一种新工具OccCANINE，可自动将职业描述转换为HISCO分类系统。处理和分类职业描述涉及的手动工作容易出错、繁琐且耗时。我们对一个现有的语言模型（CANINE）进行了微调，使其能够在几秒钟到几分钟内自动完成此过程，而以前需要数天甚至数周。该模型在来自22个不同来源贡献的13种语言中的1400万对职业描述和HISCO代码上进行训练。我们的方法表现出精度、召回率和准确率均超过90%。我们的工具突破了象征性HISCO障碍，并使这些数据可供经济学、经济历史和各种相关学科中的职业结构分析使用。

    arXiv:2402.13604v1 Announce Type: new  Abstract: This paper introduces a new tool, OccCANINE, to automatically transform occupational descriptions into the HISCO classification system. The manual work involved in processing and classifying occupational descriptions is error-prone, tedious, and time-consuming. We finetune a preexisting language model (CANINE) to do this automatically thereby performing in seconds and minutes what previously took days and weeks. The model is trained on 14 million pairs of occupational descriptions and HISCO codes in 13 different languages contributed by 22 different sources. Our approach is shown to have accuracy, recall and precision above 90 percent. Our tool breaks the metaphorical HISCO barrier and makes this data readily available for analysis of occupational structures with broad applicability in economics, economic history and various related disciplines.
    
[^52]: User-LLM: 利用用户嵌入实现有效的LLM语境化

    User-LLM: Efficient LLM Contextualization with User Embeddings

    [https://arxiv.org/abs/2402.13598](https://arxiv.org/abs/2402.13598)

    User-LLM框架利用用户嵌入对LLMs进行语境化，使其能够动态适应用户上下文，在各种任务中实现显著性能提升。

    

    大语言模型(LLMs)已经彻底改变了自然语言处理。然而，有效地整合复杂且潜在嘈杂的用户交互数据仍然是一个挑战。为了解决这个问题，我们提出了User-LLM，这是一个新颖的框架，利用用户嵌入来对LLMs进行语境化。这些嵌入是通过自监督预训练从各种用户交互中精炼出来的，能够捕捉潜在用户偏好及其随时间的演变。我们通过交叉注意力和软提示将这些用户嵌入与LLMs集成起来，使LLMs能够动态适应用户上下文。我们在MovieLens、亚马逊评论和谷歌本地评论等数据集上进行了全面实验，展示了在各种任务中的显著性能提升。值得注意的是，我们的方法在长序列任务和需要深入理解用户的任务上超过了基于文本提示的语境化，同时在计算上也更加高效。

    arXiv:2402.13598v1 Announce Type: cross  Abstract: Large language models (LLMs) have revolutionized natural language processing. However, effectively incorporating complex and potentially noisy user interaction data remains a challenge. To address this, we propose User-LLM, a novel framework that leverages user embeddings to contextualize LLMs. These embeddings, distilled from diverse user interactions using self-supervised pretraining, capture latent user preferences and their evolution over time. We integrate these user embeddings with LLMs through cross-attention and soft-prompting, enabling LLMs to dynamically adapt to user context. Our comprehensive experiments on MovieLens, Amazon Review, and Google Local Review datasets demonstrate significant performance gains across various tasks. Notably, our approach outperforms text-prompt-based contextualization on long sequence tasks and tasks that require deep user understanding while being computationally efficient. We further incorpora
    
[^53]: 知识图谱增强大型语言模型编辑

    Knowledge Graph Enhanced Large Language Model Editing

    [https://arxiv.org/abs/2402.13593](https://arxiv.org/abs/2402.13593)

    提出了一种利用知识图谱增强大型语言模型编辑的方法GLAME，能够解决编辑时知识变化合并困难的问题，提高编辑后模型的泛化能力。

    

    大型语言模型（LLMs）在推动自然语言处理（NLP）任务方面起着关键作用，然而它们的有效性受到不准确和过时知识的影响。模型编辑出现作为解决这些挑战的一种有前途的方法。然而，现有的编辑方法很难跟踪和合并与编辑相关的知识变化，这限制了后期编辑的LLMs在处理编辑知识时的泛化能力。为了解决这些问题，我们提出了一种利用知识图谱增强LLM编辑的新模型编辑方法，名为GLAME。具体来说，我们首先利用知识图谱增强模块，揭示由于编辑而发生变化的相关知识，获得LLMs内部表示。这种方法允许LLMs内的知识变化通过外部图结构反映出来。随后，我们设计了一个基于图的知识编辑模块来集成str

    arXiv:2402.13593v1 Announce Type: new  Abstract: Large language models (LLMs) are pivotal in advancing natural language processing (NLP) tasks, yet their efficacy is hampered by inaccuracies and outdated knowledge. Model editing emerges as a promising solution to address these challenges. However, existing editing methods struggle to track and incorporate changes in knowledge associated with edits, which limits the generalization ability of postedit LLMs in processing edited knowledge. To tackle these problems, we propose a novel model editing method that leverages knowledge graphs for enhancing LLM editing, namely GLAME. Specifically, we first utilize a knowledge graph augmentation module to uncover associated knowledge that has changed due to editing, obtaining its internal representations within LLMs. This approach allows knowledge alterations within LLMs to be reflected through an external graph structure. Subsequently, we design a graph-based knowledge edit module to integrate str
    
[^54]: 一种用于电子商务产品描述生成的多模态上下文调整方法

    A Multimodal In-Context Tuning Approach for E-Commerce Product Description Generation

    [https://arxiv.org/abs/2402.13587](https://arxiv.org/abs/2402.13587)

    提出了一种用于电子商务产品描述生成的多模态上下文调整方法ModICT，通过引入相似产品样本和利用语言模型的上下文学习能力，旨在解决生成描述中常见且忽略产品特征的问题

    

    在本文中，我们提出了一种新的设置，用于从图像中生成产品描述，其中包含营销关键词。它利用视觉和文本信息的综合能力，创建更加符合产品独特特性的描述。我们提出了一种简单有效的多模态上下文调整方法ModICT，通过引入相似的产品样本作为参考，并利用语言模型的上下文学习能力

    arXiv:2402.13587v1 Announce Type: new  Abstract: In this paper, we propose a new setting for generating product descriptions from images, augmented by marketing keywords. It leverages the combined power of visual and textual information to create descriptions that are more tailored to the unique features of products. For this setting, previous methods utilize visual and textual encoders to encode the image and keywords and employ a language model-based decoder to generate the product description. However, the generated description is often inaccurate and generic since same-category products have similar copy-writings, and optimizing the overall framework on large-scale samples makes models concentrate on common words yet ignore the product features. To alleviate the issue, we present a simple and effective Multimodal In-Context Tuning approach, named ModICT, which introduces a similar product sample as the reference and utilizes the in-context learning capability of language models to 
    
[^55]: WinoViz：探究对象在不同状态下的视觉特性

    WinoViz: Probing Visual Properties of Objects Under Different States

    [https://arxiv.org/abs/2402.13584](https://arxiv.org/abs/2402.13584)

    该研究提出了WinoViz评估数据集，探究语言模型对对象在不同状态下的视觉属性的推理能力，该任务具有挑战性，要求实用推理和视觉知识推理。

    

    人类根据特定语境感知和理解对象的不同视觉属性。以香蕉为例，当变质时我们知道它会变成褐色，而未成熟时它呈现绿色。先前关于探究视觉常识知识的研究主要集中在检验语言模型对对象的典型属性（如颜色和形状）的理解上。我们提出了WinoViz，一个仅包含文本的评估数据集，包含1,380个例子，探究语言模型对对象在不同语境或状态下的变异视觉属性的推理能力。我们的任务具有挑战性，因为它要求实用推理（找到预期意义）和视觉知识推理。我们还提出了多跳数据，这是我们数据的更具挑战性版本，需要多步推理链来解决我们的任务。在我们的实验分析中，我们的发现是：a）大型语言模型

    arXiv:2402.13584v1 Announce Type: new  Abstract: Humans perceive and comprehend different visual properties of an object based on specific contexts. For instance, we know that a banana turns brown ``when it becomes rotten,'' whereas it appears green ``when it is unripe.'' Previous studies on probing visual commonsense knowledge have primarily focused on examining language models' understanding of typical properties (e.g., colors and shapes) of objects. We present WinoViz, a text-only evaluation dataset, consisting of 1,380 examples that probe the reasoning abilities of language models regarding variant visual properties of objects under different contexts or states. Our task is challenging since it requires pragmatic reasoning (finding intended meanings) and visual knowledge reasoning. We also present multi-hop data, a more challenging version of our data, which requires multi-step reasoning chains to solve our task. In our experimental analysis, our findings are: a) Large language mod
    
[^56]: LongWanjuan: 面向长文本质量的系统化衡量方法

    LongWanjuan: Towards Systematic Measurement for Long Text Quality

    [https://arxiv.org/abs/2402.13583](https://arxiv.org/abs/2402.13583)

    本研究针对长文本评估的差距，引入了一套基于连贯性、凝聚力和复杂性等语言学维度的指标来系统性衡量长文本的质量，并提出了LongWanjuan数据集，有助于提升长文本任务的语言模型训练。

    

    训练数据的质量对于增强基础模型的长文本能力至关重要。尽管通过启发式规则和基于数据多样性和难度的评估来提高数据质量的现有努力，但缺乏专门针对长文本评估的系统化方法。本研究针对这一差距，通过评估三个基本语言学维度（连贯性、凝聚力和复杂性）系统性衡量长文本的质量。受到这三个维度的启发，我们引入了一套旨在评估长文本质量的指标，包括统计和基于预训练语言模型的指标。利用这些指标，我们提出了LongWanjuan，这是一个专门设计用于增强语言模型长文本任务训练的双语数据集，包含超过1600亿标记。在LongWanjuan中，我们将长文本分类为整体性、汇总

    arXiv:2402.13583v1 Announce Type: new  Abstract: The quality of training data are crucial for enhancing the long-text capabilities of foundation models. Despite existing efforts to refine data quality through heuristic rules and evaluations based on data diversity and difficulty, there's a lack of systematic approaches specifically tailored for assessing long texts. Addressing this gap, our work systematically measures the quality of long texts by evaluating three fundamental linguistic dimensions: coherence, cohesion, and complexity. Drawing inspiration from the aforementioned three dimensions, we introduce a suite of metrics designed to evaluate the quality of long texts, encompassing both statistical and pre-trained language model-based ones. Leveraging these metrics, we present LongWanjuan, a bilingual dataset specifically tailored to enhance the training of language models for long-text tasks with over 160B tokens. In LongWanjuan, we categorize long texts into holistic, aggregated
    
[^57]: BBA: 双模态行为对齐用于大规模视觉-语言模型的推理

    BBA: Bi-Modal Behavioral Alignment for Reasoning with Large Vision-Language Models

    [https://arxiv.org/abs/2402.13577](https://arxiv.org/abs/2402.13577)

    本研究提出了双模态行为对齐（BBA）提示方法，旨在最大化DSL在增强复杂多模态推理任务方面的潜力。

    

    多模态推理是大规模视觉-语言模型（LVLMs）的一个关键能力。与特定领域语言（DSL）的整合提供了精确的视觉表示，使这些模型有机会在复杂和专业领域执行更准确的推理。为了解决这些挑战，我们引入了双模态行为对齐（BBA）提示方法，旨在最大化DSL在增强复杂多模态推理任务方面的潜力。

    arXiv:2402.13577v1 Announce Type: new  Abstract: Multimodal reasoning stands as a pivotal capability for large vision-language models (LVLMs). The integration with Domain-Specific Languages (DSL), offering precise visual representations, equips these models with the opportunity to execute more accurate reasoning in complex and professional domains. However, the vanilla Chain-of-Thought (CoT) prompting method faces challenges in effectively leveraging the unique strengths of visual and DSL representations, primarily due to their differing reasoning mechanisms. Additionally, it often falls short in addressing critical steps in multi-step reasoning tasks. To mitigate these challenges, we introduce the \underline{B}i-Modal \underline{B}ehavioral \underline{A}lignment (BBA) prompting method, designed to maximize the potential of DSL in augmenting complex multi-modal reasoning tasks. This method initiates by guiding LVLMs to create separate reasoning chains for visual and DSL representations
    
[^58]: 低资源条件下南亚语言的多语言共指解析

    Multilingual Coreference Resolution in Low-resource South Asian Languages

    [https://arxiv.org/abs/2402.13571](https://arxiv.org/abs/2402.13571)

    引入了一个用于31种南亚语言的多语言共指解析翻译数据集，通过利用现成工具进行训练和对齐，在低资源条件下实现了较好的共指解析模型性能提升。

    

    共指解析涉及识别在话语中指向同一现实实体的文本片段的任务。虽然这一任务在英语中得到了广泛研究，但在南亚语言中，公开可访问的共指解析资源和模型相对稀缺。我们利用现成的翻译和词对齐工具，在31种南亚语言中引入了一个用于多语言共指解析的翻译数据集（TransMuCoRes）。几乎所有预测的翻译都通过了合理性检查，75%的英语参考文献与其预测的翻译相对应。利用多语言编码器，我们训练了两种现成的共指解析模型，将TransMuCoRes与带有手动注释的印地语共指解析数据集拼接在一起。最佳表现模型在LEA F1和CoNLL F1上分别达到了64和68的分数。

    arXiv:2402.13571v1 Announce Type: cross  Abstract: Coreference resolution involves the task of identifying text spans within a discourse that pertain to the same real-world entity. While this task has been extensively explored in the English language, there has been a notable scarcity of publicly accessible resources and models for coreference resolution in South Asian languages. We introduce a Translated dataset for Multilingual Coreference Resolution (TransMuCoRes) in 31 South Asian languages using off-the-shelf tools for translation and word-alignment. Nearly all of the predicted translations successfully pass a sanity check, and 75% of English references align with their predicted translations. Using multilingual encoders, two off-the-shelf coreference resolution models were trained on a concatenation of TransMuCoRes and a Hindi coreference resolution dataset with manual annotations. The best performing model achieved a score of 64 and 68 for LEA F1 and CoNLL F1, respectively, on o
    
[^59]: 多源语言训练在跨语言转移中的分析

    Analysis of Multi-Source Language Training in Cross-Lingual Transfer

    [https://arxiv.org/abs/2402.13562](https://arxiv.org/abs/2402.13562)

    多源语言训练（MSLT）技术通过使用多个源语言，在跨语言转移中增加了不同语言嵌入空间的交织，从而支持了XLT受益于这种方法的说法。

    

    成功地将多语言语言模型（LMs）调整到特定语言-任务对上至关重要的是定制数据的可用性。虽然跨语言转移（XLT）方法有助于解决这种数据稀缺问题，但关于其有效性背后的机制仍存在持续的讨论。在这项工作中，我们关注了关于XLT内部工作的一个有希望的假设，即它鼓励多语言LMs更加强调语言不可知或任务特定特征。我们通过考察XLT随涉及过程中源语言数量的变化而改变的模式来测试这一假设。我们的实验结果表明，在XLT中使用多个源语言-一种我们称之为多源语言训练（MSLT）的技术-会导致不同语言的嵌入空间的交织增加，支持了XLT受益于这一点的说法。

    arXiv:2402.13562v1 Announce Type: new  Abstract: The successful adaptation of multilingual language models (LMs) to a specific language-task pair critically depends on the availability of data tailored for that condition. While cross-lingual transfer (XLT) methods have contributed to addressing this data scarcity problem, there still exists ongoing debate about the mechanisms behind their effectiveness. In this work, we focus on one of promising assumptions about inner workings of XLT, that it encourages multilingual LMs to place greater emphasis on language-agnostic or task-specific features. We test this hypothesis by examining how the patterns of XLT change with a varying number of source languages involved in the process. Our experimental findings show that the use of multiple source languages in XLT-a technique we term Multi-Source Language Training (MSLT)-leads to increased mingling of embedding spaces for different languages, supporting the claim that XLT benefits from making us
    
[^60]: 认知视觉语言映射器：通过增强视觉知识对齐推进多模态理解

    Cognitive Visual-Language Mapper: Advancing Multimodal Comprehension with Enhanced Visual Knowledge Alignment

    [https://arxiv.org/abs/2402.13561](https://arxiv.org/abs/2402.13561)

    该论文提出了一种认知视觉语言映射器（CVLM），通过增强视觉知识对齐，在多模态理解中取得了重要进展，特别是在挑战知识型视觉问题回答方面。

    

    评估和反思当前大型多模态模型（LMMs）的现状，我们观察到广泛使用的视觉语言投影方法（如Q-former或MLP）侧重于图像-文本描述的对齐，但忽略了视觉知识维度的对齐，即将视觉与其相关知识连接起来。视觉知识在分析、推断和解释视觉信息方面起着重要作用，有助于提高基于知识的视觉问题答案的准确性。本文主要探讨通过视觉语言知识对齐来改进LMMs，特别针对挑战知识型视觉问答（VQA）。为此，我们提出了一个认知视觉语言映射器（CVLM），其中包含一个预训练的视觉知识对齐器（VKA）和一个用于多模态指令调节阶段的细粒度知识适配器（FKA）。具体来说，我们基于

    arXiv:2402.13561v1 Announce Type: new  Abstract: Evaluating and Rethinking the current landscape of Large Multimodal Models (LMMs), we observe that widely-used visual-language projection approaches (e.g., Q-former or MLP) focus on the alignment of image-text descriptions yet ignore the visual knowledge-dimension alignment, i.e., connecting visuals to their relevant knowledge. Visual knowledge plays a significant role in analyzing, inferring, and interpreting information from visuals, helping improve the accuracy of answers to knowledge-based visual questions. In this paper, we mainly explore improving LMMs with visual-language knowledge alignment, especially aimed at challenging knowledge-based visual question answering (VQA). To this end, we present a Cognitive Visual-Language Mapper (CVLM), which contains a pretrained Visual Knowledge Aligner (VKA) and a Fine-grained Knowledge Adapter (FKA) used in the multimodal instruction tuning stage. Specifically, we design the VKA based on the 
    
[^61]: 叙事背景的图表示：通过回顾性问题的连贯依赖

    Graph Representation of Narrative Context: Coherence Dependency via Retrospective Questions

    [https://arxiv.org/abs/2402.13551](https://arxiv.org/abs/2402.13551)

    提出了一种新颖且实用的叙事理解范式，通过在叙事中形成图NARCO来描述整个背景的任务无关的连贯依赖，其中的边反映了高层次的连贯关系，无需依赖人类注释。

    

    这项工作介绍了一种新颖且实用的叙事理解范式，这是基于一个观察：叙述中的个别段落通常是相互关联的，而不是孤立的。因此，我们提出在叙事中形成一个名为NARCO的图，描述整个背景的任务无关的连贯依赖。特别是，NARCO中的边涵盖了两个上下文片段之间的自由形式回顾性问题，反映了高层次的连贯关系，受到人类认知感知的启发，人类不断从先前背景中重申相关事件。重要的是，我们的图是通过我们设计的两阶段LLM提示实例化的，因此无需依赖人类注释。我们展示了三个关于其实际效用的独特研究，通过总结识别检验边的有效性，通过情节检索进行本地上下文增强，以及通过长文档问答示例化的更广泛应用。

    arXiv:2402.13551v1 Announce Type: new  Abstract: This work introduces a novel and practical paradigm for narrative comprehension, stemming from the observation that individual passages within narratives are often cohesively related than being isolated. We therefore propose to formulate a graph upon narratives dubbed NARCO that depicts a task-agnostic coherence dependency of the entire context. Especially, edges in NARCO encompass retrospective free-form questions between two context snippets reflecting high-level coherent relations, inspired by the cognitive perception of humans who constantly reinstate relevant events from prior context. Importantly, our graph is instantiated through our designed two-stage LLM prompting, thereby without reliance on human annotations. We present three unique studies on its practical utility, examining the edge efficacy via recap identification, local context augmentation via plot retrieval, and broader applications exemplified by long document QA. Expe
    
[^62]: LLM们是有效的谈判者吗？对LLM在谈判对话中多方面能力的系统评估

    Are LLMs Effective Negotiators? Systematic Evaluation of the Multifaceted Capabilities of LLMs in Negotiation Dialogues

    [https://arxiv.org/abs/2402.13550](https://arxiv.org/abs/2402.13550)

    本研究系统评估了LLMs在谈判对话中的多方面能力，揭示了它们在谈判研究中的潜力和局限。

    

    一次成功的谈判需要对谈话背景有深刻理解，具备推断对方动机的心理理论技能，以及战略推理和有效沟通，这使得自动化系统面临挑战。鉴于LLMs在各种自然语言处理任务中表现出色，本研究旨在探索LLMs如何推动谈判研究的不同方面，包括设计对话系统、提供教学反馈和扩大数据收集实践。为此，我们设计了一种方法来分析LLMs在各种对话情景中的多方面能力，涵盖典型谈判互动的所有时间阶段。我们的分析进一步证明了GPT-4在各种任务上的优越性，同时也揭示了LLMs在某些任务上仍然困难的细节。例如，这些模型与人类的相关性较差。

    arXiv:2402.13550v1 Announce Type: cross  Abstract: A successful negotiation demands a deep comprehension of the conversation context, Theory-of-Mind (ToM) skills to infer the partner's motives, as well as strategic reasoning and effective communication, making it challenging for automated systems. Given the remarkable performance of LLMs across a variety of NLP tasks, in this work, we aim to understand how LLMs can advance different aspects of negotiation research, ranging from designing dialogue systems to providing pedagogical feedback and scaling up data collection practices. To this end, we devise a methodology to analyze the multifaceted capabilities of LLMs across diverse dialogue scenarios covering all the time stages of a typical negotiation interaction. Our analysis adds to the increasing evidence for the superiority of GPT-4 across various tasks while also providing insights into specific tasks that remain difficult for LLMs. For instance, the models correlate poorly with hum
    
[^63]: ActiveRAG: 通过主动学习揭示知识的宝藏

    ActiveRAG: Revealing the Treasures of Knowledge via Active Learning

    [https://arxiv.org/abs/2402.13547](https://arxiv.org/abs/2402.13547)

    ActiveRAG是一个创新的RAG框架，通过引入主动学习机制，利用知识构建和认知联结机制来提升大型语言模型（LLMs）的内在认知，实现了明显的性能提升。

    

    arXiv:2402.13547v1 公告类型：新摘要：检索增强生成（RAG）引入了一种新的大型语言模型（LLM）范例，有助于解决知识密集型任务。然而，当前的RAG模型将LLMs定位为被动的知识接收器，从而限制了它们学习和理解外部知识的能力。本文提出了ActiveRAG，它是一种创新的RAG框架，从被动知识获取转变为主动学习机制。这种方法利用知识构建机制通过将外部知识与先前获取或记忆的知识相关联来更深入地理解外部知识。随后，它设计了认知联结机制以合并来自思维和知识构建链的成果，从而校准LLMs的内在认知。我们的实验结果表明，ActiveRAG超越了先前的RAG模型，在问题回答上实现了5%的改进。

    arXiv:2402.13547v1 Announce Type: new  Abstract: Retrieval Augmented Generation (RAG) has introduced a new paradigm for Large Language Models (LLMs), aiding in the resolution of knowledge-intensive tasks. However, current RAG models position LLMs as passive knowledge receptors, thereby restricting their capacity for learning and comprehending external knowledge. In this paper, we present ActiveRAG, an innovative RAG framework that shifts from passive knowledge acquisition to an active learning mechanism. This approach utilizes the Knowledge Construction mechanism to develop a deeper understanding of external knowledge by associating it with previously acquired or memorized knowledge. Subsequently, it designs the Cognitive Nexus mechanism to incorporate the outcomes from both chains of thought and knowledge construction, thereby calibrating the intrinsic cognition of LLMs. Our experimental results demonstrate that ActiveRAG surpasses previous RAG models, achieving a 5% improvement on qu
    
[^64]: LLMs与长视频相遇：在LLMs中利用互动式视觉适配器推进长视频理解

    LLMs Meet Long Video: Advancing Long Video Comprehension with An Interactive Visual Adapter in LLMs

    [https://arxiv.org/abs/2402.13546](https://arxiv.org/abs/2402.13546)

    介绍了一个交互式视觉适配器（IVA），用于在LLMs中增强对细粒度视觉元素的交互，并解决了长视频理解中的计算成本高、视觉清晰度降低和无关视觉令牌带来的挑战。

    

    长视频理解是多媒体和人工智能交叉领域中一项重要且持续挑战。利用大型语言模型(LLMs)来理解视频成为一种新兴且有前景的方法。然而，由于视频令牌数量庞大，这种方法导致计算成本高，视觉清晰度降低，还面临着在回答视频相关问题时出现无关视觉令牌所带来的挑战。为了缓解这些问题，我们在LLMs中提出了一个交互式视觉适配器(IVA)，旨在增强与细粒度视觉元素的交互。具体来说，我们首先通过利用视觉编码器和预训练因果变换器将长视频转换为时间视频令牌，然后将它们与视频说明一起输入LLMs。随后，我们集成了IVA，其中包含一个轻量级的时间帧选择器

    arXiv:2402.13546v1 Announce Type: new  Abstract: Long video understanding is a significant and ongoing challenge in the intersection of multimedia and artificial intelligence. Employing large language models (LLMs) for comprehending video becomes an emerging and promising method. However, this approach incurs high computational costs due to the extensive array of video tokens, experiences reduced visual clarity as a consequence of token aggregation, and confronts challenges arising from irrelevant visual tokens while answering video-related questions. To alleviate these issues, we present an Interactive Visual Adapter (IVA) within LLMs, designed to enhance interaction with fine-grained visual elements. Specifically, we first transform long videos into temporal video tokens via leveraging a visual encoder alongside a pretrained causal transformer, then feed them into LLMs with the video instructions. Subsequently, we integrated IVA, which contains a lightweight temporal frame selector a
    
[^65]: ARL2: 通过自导自适应相关性标记将检索器与黑盒大型语言模型对齐

    ARL2: Aligning Retrievers for Black-box Large Language Models via Self-guided Adaptive Relevance Labeling

    [https://arxiv.org/abs/2402.13542](https://arxiv.org/abs/2402.13542)

    ARL2提出了一种检索器学习技术，利用LLMs作为标注者，并采用自适应自训练策略，能够有效减少注释成本，并在NQ和MMLU上取得了5.4%和4.6%的准确度提升。

    

    arXiv:2402.13542v1 公告类型: 交叉 摘要: 检索增强生成通过整合外部知识源的相关信息改进大型语言模型（LLMs），使LLMs能够适应特定领域，并减轻知识密集任务中的幻觉。然而，由于其分开的训练过程和LLMs的黑盒特性，现有的检索器通常与LLMs不匹配。为解决这一挑战，我们提出了ARL2，一种利用LLMs作为标注者的检索器学习技术。ARL2利用LLMs注释和评分相关证据，从而能够从强大的LLM监督中学习检索器。此外，ARL2使用自适应自训练策略来策划高质量和多样性相关性数据，可以有效降低标注成本。大量实验表明ARL2的有效性，与最先进方法相比，在NQ上提高了5.4%的准确率，在MMLU上提高了4.6%。

    arXiv:2402.13542v1 Announce Type: cross  Abstract: Retrieval-augmented generation enhances large language models (LLMs) by incorporating relevant information from external knowledge sources. This enables LLMs to adapt to specific domains and mitigate hallucinations in knowledge-intensive tasks. However, existing retrievers are often misaligned with LLMs due to their separate training processes and the black-box nature of LLMs. To address this challenge, we propose ARL2, a retriever learning technique that harnesses LLMs as labelers. ARL2 leverages LLMs to annotate and score relevant evidence, enabling learning the retriever from robust LLM supervision. Furthermore, ARL2 uses an adaptive self-training strategy for curating high-quality and diverse relevance data, which can effectively reduce the annotation cost. Extensive experiments demonstrate the effectiveness of ARL2, achieving accuracy improvements of 5.4% on NQ and 4.6% on MMLU compared to the state-of-the-art methods. Additionall
    
[^66]: 一种有效融合异构知识的课程学习方法用于序列标注

    An Effective Incorporating Heterogeneous Knowledge Curriculum Learning for Sequence Labeling

    [https://arxiv.org/abs/2402.13534](https://arxiv.org/abs/2402.13534)

    提出了一个专为序列标注任务设计的两阶段课程学习（TCL）框架，逐渐引入数据实例从简单到困难，旨在提高性能和训练速度，并且对六个中文分词和词性标注数据集进行了广泛实验，证明了模型的有效性。

    

    序列标注模型常常受益于整合外部知识。然而，这一做法引入了数据异构性，并通过额外模块使模型变得复杂，导致训练高性能模型的成本增加。为了应对这一挑战，我们提出了一个专为序列标注任务设计的两阶段课程学习（TCL）框架。TCL框架通过逐渐引入从简单到困难的数据实例来增强训练，旨在提高性能和训练速度。此外，我们还探索了用于评估序列标注任务难度级别的不同指标。通过在六个中文分词（CWS）和词性标注（POS）数据集上进行大量实验，我们展示了我们的模型在提高序列标注模型性能方面的有效性。此外，我们的分析表明TCL加速了训练并缓解了

    arXiv:2402.13534v1 Announce Type: cross  Abstract: Sequence labeling models often benefit from incorporating external knowledge. However, this practice introduces data heterogeneity and complicates the model with additional modules, leading to increased expenses for training a high-performing model. To address this challenge, we propose a two-stage curriculum learning (TCL) framework specifically designed for sequence labeling tasks. The TCL framework enhances training by gradually introducing data instances from easy to hard, aiming to improve both performance and training speed. Furthermore, we explore different metrics for assessing the difficulty levels of sequence labeling tasks. Through extensive experimentation on six Chinese word segmentation (CWS) and Part-of-speech tagging (POS) datasets, we demonstrate the effectiveness of our model in enhancing the performance of sequence labeling models. Additionally, our analysis indicates that TCL accelerates training and alleviates the 
    
[^67]: FinGPT-HPC: 高性能计算下用于金融应用的高效预训练和微调大型语言模型

    FinGPT-HPC: Efficient Pretraining and Finetuning Large Language Models for Financial Applications with High-Performance Computing

    [https://arxiv.org/abs/2402.13533](https://arxiv.org/abs/2402.13533)

    该论文提出了一种基于高性能GPU的方法，利用低秩结构来高效地预训练和微调大型语言模型，解决了线性层冗余性、GPU内存占用和分布式训练中GPU利用率不足的挑战

    

    大型语言模型(LLMs)的计算密集性很高。计算工作量和内存占用量随维度(层宽度)的增加呈二次增长。大多数LLM参数来自变压器结构的线性层，具有高度冗余性。这些线性层贡献了超过80%的计算工作量和99%的模型大小。为了高效地预训练和微调LLMs，需要解决三个主要挑战：1) 减少线性层的冗余性；2) 减少GPU内存占用；3) 在使用分布式训练时提高GPU利用率。之前的方法，如LoRA和QLoRA，利用低秩矩阵和量化来分别减少可训练参数的数量和模型大小。然而， resulting model 仍然消耗大量GPU内存。在本文中，我们提出了基于高性能GPU的方法，利用低秩结构来预训练和微调。

    arXiv:2402.13533v1 Announce Type: cross  Abstract: Large language models (LLMs) are computationally intensive. The computation workload and the memory footprint grow quadratically with the dimension (layer width). Most of LLMs' parameters come from the linear layers of the transformer structure and are highly redundant. These linear layers contribute more than 80% of the computation workload and 99% of the model size. To pretrain and finetune LLMs efficiently, there are three major challenges to address: 1) reducing redundancy of the linear layers; 2) reducing GPU memory footprint; 3) improving GPU utilization when using distributed training. Prior methods, such as LoRA and QLoRA, utilized low-rank matrices and quantization to reduce the number of trainable parameters and model size, respectively. However, the resulting model still consumes a large amount of GPU memory. In this paper, we present high-performance GPU-based methods that exploit low-rank structures to pretrain and finetun
    
[^68]: 密集通道检索器用于传播信息错误的后门攻击

    Backdoor Attacks on Dense Passage Retrievers for Disseminating Misinformation

    [https://arxiv.org/abs/2402.13532](https://arxiv.org/abs/2402.13532)

    本文介绍了一种后门攻击场景，攻击者通过利用密集通道检索的语法错误触发后门攻击，以秘密传播定向错误信息，如仇恨言论或广告，并通过实验证明了这种攻击方法的有效性和隐匿性。

    

    密集检索器和检索增强语言模型已广泛用于各种NLP应用，尽管设计用于提供可靠和安全的结果，但检索器对潜在攻击的脆弱性仍不清楚，引发人们对其安全性的关注。本文介绍了一种新颖的情景，攻击者旨在通过检索系统隐蔽传播定向错误信息，如仇恨言论或广告。为实现这一目标，我们提出了一种在密集通道检索中由语法错误触发的危险后门攻击。我们的方法确保被攻击的模型在标准查询下可以正常运行，但在用户在查询中意外地犯语法错误时，被篡改以返回攻击者指定的段落。大量实验展示了我们提出的攻击方法的有效性和隐蔽性。

    arXiv:2402.13532v1 Announce Type: new  Abstract: Dense retrievers and retrieval-augmented language models have been widely used in various NLP applications. Despite being designed to deliver reliable and secure outcomes, the vulnerability of retrievers to potential attacks remains unclear, raising concerns about their security. In this paper, we introduce a novel scenario where the attackers aim to covertly disseminate targeted misinformation, such as hate speech or advertisement, through a retrieval system. To achieve this, we propose a perilous backdoor attack triggered by grammar errors in dense passage retrieval. Our approach ensures that attacked models can function normally for standard queries but are manipulated to return passages specified by the attacker when users unintentionally make grammatical mistakes in their queries. Extensive experiments demonstrate the effectiveness and stealthiness of our proposed attack method. When a user query is error-free, our model consistentl
    
[^69]: 基础设施调解员：从结构灾难响应中挖掘未来失效担忧

    Infrastructure Ombudsman: Mining Future Failure Concerns from Structural Disaster Response

    [https://arxiv.org/abs/2402.13528](https://arxiv.org/abs/2402.13528)

    本文开发了一种基础设施调解员系统，用于自动检测特定基础设施问题，通过挖掘社交网络中关于预期失败的担忧，有助于预防和减轻潜在的基础设施失败。

    

    当前研究集中于研究社交媒体上与结构失败相关的讨论，以改进灾难响应策略。然而，检测社交网络帖子中讨论关于预期失败的担忧是未被充分探索的。如果这些担忧被传达给适当的机构，可以帮助预防和减轻潜在的基础设施失败。本文中，我们开发了一种基础设施调解员——用于自动检测特定基础设施问题。我们的工作考虑了美国几起最近的结构失效事件。我们呈现了一份首创性数据集，包括从Reddit和YouTube中挖掘的2,662个社交网络实例，用于这一新颖任务。

    arXiv:2402.13528v1 Announce Type: cross  Abstract: Current research concentrates on studying discussions on social media related to structural failures to improve disaster response strategies. However, detecting social web posts discussing concerns about anticipatory failures is under-explored. If such concerns are channeled to the appropriate authorities, it can aid in the prevention and mitigation of potential infrastructural failures. In this paper, we develop an infrastructure ombudsman -- that automatically detects specific infrastructure concerns. Our work considers several recent structural failures in the US. We present a first-of-its-kind dataset of 2,662 social web instances for this novel task mined from Reddit and YouTube.
    
[^70]: OMGEval：面向大型语言模型的开放多语言生成评估基准

    OMGEval: An Open Multilingual Generative Evaluation Benchmark for Large Language Models

    [https://arxiv.org/abs/2402.13524](https://arxiv.org/abs/2402.13524)

    OMGEval是第一个可以评估大型语言模型在不同语言中能力的开放源多语言生成测试集，涵盖了广泛重要能力并进行了本地化处理。

    

    现代大型语言模型（LLMs）应该普遍受益于全球各种文化背景的个人。然而，大多数最近的先进生成评估基准主要专注于英语的LLMs。为此，我们推出了OMGEval，第一个可以评估LLMs在不同语言中能力的开放源多语言生成测试集。对于每种语言，OMGEval提供了804个开放式问题，涵盖了LLMs的广泛重要能力，如常识、逻辑推理等。每个问题都经过人类标注者的严格验证。值得注意的是，为了充分反映LLMs在不同文化背景下的兼容性，我们为每种非英语语言进行本地化。具体而言，当前版本的OMGEval包括5种语言（即：中文、俄文、法语、西班牙文、阿拉伯文）。在AlpacaEval之后，我们使用GPT-4作为裁判自动评分不同的模型。

    arXiv:2402.13524v1 Announce Type: new  Abstract: Modern large language models (LLMs) should generally benefit individuals from various cultural backgrounds around the world. However, most recent advanced generative evaluation benchmarks tailed for LLMs mainly focus on English. To this end, we introduce OMGEval, the first Open-source Multilingual Generative test set that can assess the capability of LLMs in different languages. For each language, OMGEval provides 804 open-ended questions, covering a wide range of important capabilities of LLMs, such as general knowledge, logical reasoning, and so on. Each question is rigorously verified by human annotators. Notably, to sufficiently reflect the compatibility of LLMs in different cultural backgrounds, we perform localization for each non-English language. Specifically, the current version of OMGEval includes 5 languages (i.e., Zh, Ru, Fr, Es, Ar). Following AlpacaEval, we employ GPT-4 as the adjudicator to automatically score different mo
    
[^71]: RecMind: 寻求者内心状态下的日本电影推荐对话

    RecMind: Japanese Movie Recommendation Dialogue with Seeker's Internal State

    [https://arxiv.org/abs/2402.13522](https://arxiv.org/abs/2402.13522)

    RecMind是一个具有寻求者内在状态注释的日本电影推荐对话数据集，研究发现对那些寻求者感兴趣但并不了解的实体进行推荐有助于成功推荐，并提出了一个考虑寻求者内在状态的响应生成框架。

    

    人类在对话中会仔细关注交流者的内在状态。例如，在推荐对话中，我们会在推荐时估计寻求者的内心状态，比如他/她的知识水平和兴趣。鉴于目前没有现有的资源用于分析，我们构建了RecMind，这是一个具有寻求者内在状态实体级别注释的日本电影推荐对话数据集。每个实体都有一个由寻求者注释的主观标签和由推荐者注释的客观标签。RecMind还具有引人入胜的对话，具有长篇寻求者话语，有助于详细分析寻求者的内在状态。我们基于RecMind的分析显示，寻求者对一些并不了解但感兴趣的实体有助于推荐成功。我们还提出了一个响应生成框架，明确考虑了寻求者的内在状态。

    arXiv:2402.13522v1 Announce Type: new  Abstract: Humans pay careful attention to the interlocutor's internal state in dialogues. For example, in recommendation dialogues, we make recommendations while estimating the seeker's internal state, such as his/her level of knowledge and interest. Since there are no existing annotated resources for the analysis, we constructed RecMind, a Japanese movie recommendation dialogue dataset with annotations of the seeker's internal state at the entity level. Each entity has a subjective label annotated by the seeker and an objective label annotated by the recommender. RecMind also features engaging dialogues with long seeker's utterances, enabling a detailed analysis of the seeker's internal state. Our analysis based on RecMind reveals that entities that the seeker has no knowledge about but has an interest in contribute to recommendation success. We also propose a response generation framework that explicitly considers the seeker's internal state, ut
    
[^72]: RITFIS: RITFIS：基于LLMs的智能软件强壮输入测试框架

    RITFIS: Robust input testing framework for LLMs-based intelligent software

    [https://arxiv.org/abs/2402.13518](https://arxiv.org/abs/2402.13518)

    RITFIS是第一个设计用于评估基于LLM的智能软件对自然语言输入鲁棒性的框架，通过将测试过程定义为组合优化问题来确定成功的测试案例。

    

    arXiv:2402.13518v1 公告类型：交叉 摘要：自然语言处理（NLP）智能软件对大型语言模型（LLMs）的依赖日益突出，强调了对鲁棒性测试的必要性。当前的测试方法仅关注基于LLM的软件对提示的鲁棒性。鉴于现实世界输入的复杂性和多样性，研究LLM-based软件处理全面输入（包括提示和示例）的鲁棒性对于全面了解其性能至关重要。 为此，本文介绍了RITFIS，一种用于LLM-based智能软件的鲁棒输入测试框架。据我们所知，RITFIS是第一个旨在评估LLM-based智能软件对自然语言输入的鲁棒性的框架。该框架基于给定的威胁模型和提示，主要将测试过程定义为组合优化问题。成功的测试案例由一个目标函数决定

    arXiv:2402.13518v1 Announce Type: cross  Abstract: The dependence of Natural Language Processing (NLP) intelligent software on Large Language Models (LLMs) is increasingly prominent, underscoring the necessity for robustness testing. Current testing methods focus solely on the robustness of LLM-based software to prompts. Given the complexity and diversity of real-world inputs, studying the robustness of LLMbased software in handling comprehensive inputs (including prompts and examples) is crucial for a thorough understanding of its performance.   To this end, this paper introduces RITFIS, a Robust Input Testing Framework for LLM-based Intelligent Software. To our knowledge, RITFIS is the first framework designed to assess the robustness of LLM-based intelligent software against natural language inputs. This framework, based on given threat models and prompts, primarily defines the testing process as a combinatorial optimization problem. Successful test cases are determined by a goal fu
    
[^73]: 大型语言模型逆向翻译防御对抗攻击

    Round Trip Translation Defence against Large Language Model Jailbreaking Attacks

    [https://arxiv.org/abs/2402.13517](https://arxiv.org/abs/2402.13517)

    往返翻译（RTT）方法是第一个专门设计用于抵御大型语言模型（LLMs）社交工程攻击的算法，成功地减少了多种攻击形式的成功率。

    

    大型语言模型（LLMs）容易受到社交工程攻击，这些攻击对人类具有可解释性，但需要LLMs具有高水平的理解能力才能抵抗。现有的防御措施最多只能缓解这些攻击的不到一半。为解决这一问题，我们提出了往返翻译（RTT）方法，这是第一个专门设计用于抵御LLMs社交工程攻击的算法。RTT会改写对抗性提示并推广表达的思想，使LLMs更容易检测出诱发有害行为。这种方法灵活、轻量且可转移至不同的LLMs。我们的防御成功地缓解了超过70%的Prompt Automatic Iterative Refinement (PAIR)攻击，这是目前我们所知最有效的防御。我们也是首次尝试缓解MathsAttack，并将其攻击成功率降低了近40%。我们的代码已公开发布。

    arXiv:2402.13517v1 Announce Type: cross  Abstract: Large language models (LLMs) are susceptible to social-engineered attacks that are human-interpretable but require a high level of comprehension for LLMs to counteract. Existing defensive measures can only mitigate less than half of these attacks at most. To address this issue, we propose the Round Trip Translation (RTT) method, the first algorithm specifically designed to defend against social-engineered attacks on LLMs. RTT paraphrases the adversarial prompt and generalizes the idea conveyed, making it easier for LLMs to detect induced harmful behavior. This method is versatile, lightweight, and transferrable to different LLMs. Our defense successfully mitigated over 70% of Prompt Automatic Iterative Refinement (PAIR) attacks, which is currently the most effective defense to the best of our knowledge. We are also the first to attempt mitigating the MathsAttack and reduced its attack success rate by almost 40%. Our code is publicly av
    
[^74]: ProSparse: 引入和增强大型语言模型内部激活稀疏性

    ProSparse: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models

    [https://arxiv.org/abs/2402.13516](https://arxiv.org/abs/2402.13516)

    本文介绍了一种名为"ProSparse"的有效稀疏化方法，以推动大型语言模型实现更高的激活稀疏性而不降低模型性能

    

    Activation sparsity指的是激活输出中存在许多弱贡献元素。作为使用ReLU激活函数的模型的普遍属性，已被证明是提高模型推理效率的一种有前途的范例。然而，大多数大型语言模型（LLMs）采用了没有内在激活稀疏性的激活函数（例如GELU和Swish）。一些最近的努力尝试引入ReLU或其变体作为替代激活函数，以帮助LLMs实现激活稀疏性和推理加速，但很少能同时获得高稀疏度和可比较的模型性能。本文介绍了一种名为"ProSparse"的有效稀疏化方法，以推动LLMs实现更高的激活稀疏性而不降低模型性能。具体来说，将LLMs的激活函数替换为ReLU后，ProSparse采用渐进稀疏正则化

    arXiv:2402.13516v1 Announce Type: cross  Abstract: Activation sparsity refers to the existence of considerable weakly-contributed elements among activation outputs. As a prevalent property of the models using the ReLU activation function, it has been proven a promising paradigm to boost model inference efficiency. Nevertheless, most large language models (LLMs) adopt activation functions without intrinsic activation sparsity (e.g., GELU and Swish). Some recent efforts have explored introducing ReLU or its variants as the substitutive activation function to help LLMs achieve activation sparsity and inference acceleration, but few can simultaneously obtain high sparsity and comparable model performance. This paper introduces an effective sparsification method named "ProSparse" to push LLMs for higher activation sparsity without decreasing model performance. Specifically, after substituting the activation function of LLMs with ReLU, ProSparse adopts progressive sparsity regularization wit
    
[^75]: 自我分而治之：何时检索、何时生成？面向组合未知问题的自我分而治之算法

    Self-DC: When to retrieve and When to generate? Self Divide-and-Conquer for Compositional Unknown Questions

    [https://arxiv.org/abs/2402.13514](https://arxiv.org/abs/2402.13514)

    提出了面向组合未知问题的自我分而治之算法，引入了第一个组合未知问题问答数据集（CuQA），通过自适应调用不同方法实现更好的性能和效率。

    

    检索-然后阅读和生成-然后阅读是处理开放域问答中未知和已知问题的两种典型解决方案，前者检索必要的外部知识，后者则促使大型语言模型生成参数中编码的内部已知知识。然而，过去很少有作品考虑到组合未知问题，这些问题由几个已知或未知的子问题组成。因此，简单的二元分类（已知或未知）变得次优和低效，因为它会对每个组合未知问题过度调用外部检索。为此，我们提出了第一个组合未知问题问答数据集（CuQA），并引入了一个自我分而治之（Self-DC）框架，使大型语言模型能够自适应地调用不同的方法，从而提高性能和效率。实验结果在两个数据集（CuQA和FreshQA）上表明……

    arXiv:2402.13514v1 Announce Type: cross  Abstract: Retrieve-then-read and generate-then-read are two typical solutions to handle unknown and known questions in open-domain question-answering, while the former retrieves necessary external knowledge and the later prompt the large language models to generate internal known knowledge encoded in the parameters. However, few of previous works consider the compositional unknown questions, which consist of several known or unknown sub-questions. Thus, simple binary classification (known or unknown) becomes sub-optimal and inefficient since it will call external retrieval excessively for each compositional unknown question. To this end, we propose the first Compositional unknown Question-Answering dataset (CuQA), and introduce a Self Divide-and-Conquer (Self-DC) framework to empower LLMs to adaptively call different methods on-demand, resulting in better performance and efficiency. Experimental results on two datasets (CuQA and FreshQA) demonst
    
[^76]: 从自注意力到马尔可夫模型：揭示生成Transformer的动态

    From Self-Attention to Markov Models: Unveiling the Dynamics of Generative Transformers

    [https://arxiv.org/abs/2402.13512](https://arxiv.org/abs/2402.13512)

    本文研究了从自注意力模型到马尔可夫模型的转变，揭示了生成Transformer动态的机理和相关条件，为一致估计提供了保证，并在IID样本下建立了样本复杂性保证。

    

    现代语言模型依赖Transformer架构和注意力机制来进行语言理解和文本生成。本文研究了从一组提示和与模型采样的关联输出数据中学习一个单层自注意模型。我们首先建立了自注意机制和马尔可夫模型之间的精确映射：将提示输入模型会根据上下文条件的马尔可夫链（CCMC）对输出标记进行采样，该链加权了基本马尔可夫链的转移矩阵。此外，引入位置编码导致了转移概率的位置相关缩放。基于这种形式主义，我们为提示分布开发了可辨识性/覆盖条件，确保一致估计，并在IID样本下建立了样本复杂性保证。最后，我们研究了从单个输出轨迹生成中学习的问题。

    arXiv:2402.13512v1 Announce Type: cross  Abstract: Modern language models rely on the transformer architecture and attention mechanism to perform language understanding and text generation. In this work, we study learning a 1-layer self-attention model from a set of prompts and associated output data sampled from the model. We first establish a precise mapping between the self-attention mechanism and Markov models: Inputting a prompt to the model samples the output token according to a context-conditioned Markov chain (CCMC) which weights the transition matrix of a base Markov chain. Additionally, incorporating positional encoding results in position-dependent scaling of the transition probabilities. Building on this formalism, we develop identifiability/coverage conditions for the prompt distribution that guarantee consistent estimation and establish sample complexity guarantees under IID samples. Finally, we study the problem of learning from a single output trajectory generated from
    
[^77]: 利用翻译实现最佳召回率：通过用户配置文件定制LLM个性化

    Leveraging Translation For Optimal Recall: Tailoring LLM Personalization With User Profiles

    [https://arxiv.org/abs/2402.13500](https://arxiv.org/abs/2402.13500)

    本研究提出了一种通过多级翻译、语义嵌入扩展和用户配置文件中心扩展相结合的方法，旨在在跨语言信息检索系统中改善召回率，通过个性化匹配用户查询和相关文档，展示了比基线方法更优异的性能。

    

    本文探讨了一种新颖技术，通过基于用户的词汇-语义空间的迭代查询优化来提高跨语言信息检索(CLIR)系统中的召回率。提出的方法结合了多级翻译、基于语义嵌入的扩展，以及以用户配置文件为中心的扩展，以解决用户查询和相关文档之间的匹配差异挑战。通过初始的BM25检索、转换为中间语言、查找相似术语的嵌入，以及迭代重新排名，该技术旨在扩大可能与个体用户相关的潜在结果范围。对新闻和Twitter数据集的比较实验证明，所提方法在ROUGE指标方面优于基线BM25排名。翻译方法还通过多步骤过程展示出了保持的语义准确性。

    arXiv:2402.13500v1 Announce Type: cross  Abstract: This paper explores a novel technique for improving recall in cross-language information retrieval (CLIR) systems using iterative query refinement grounded in the user's lexical-semantic space. The proposed methodology combines multi-level translation, semantic embedding-based expansion, and user profile-centered augmentation to address the challenge of matching variance between user queries and relevant documents. Through an initial BM25 retrieval, translation into intermediate languages, embedding lookup of similar terms, and iterative re-ranking, the technique aims to expand the scope of potentially relevant results personalized to the individual user. Comparative experiments on news and Twitter datasets demonstrate superior performance over baseline BM25 ranking for the proposed approach across ROUGE metrics. The translation methodology also showed maintained semantic accuracy through the multi-step process. This personalized CLIR 
    
[^78]: 生物医学的平民指南：编排大型语言模型

    The Lay Person's Guide to Biomedicine: Orchestrating Large Language Models

    [https://arxiv.org/abs/2402.13498](https://arxiv.org/abs/2402.13498)

    利用大型语言模型生成和评估生物医学文章的平民总结，提出了Explain-then-Summarise的新LS框架，并评估了LLMs在零-shot LS方面的表现和提出了两种新的LLM-based LS评估方法。

    

    《arXiv：2402.13498v1》公告类型：新的摘要：自动化的平民总结（LS）旨在将复杂的技术文件简化为更易于非专业人士理解的格式。现有的使用预训练语言模型，可能辅以外部背景知识的方法往往在有效简化和解释方面存在困难。此外，能够有效评估生成摘要的“平民性”的自动化方法也缺乏。最近，大型语言模型（LLMs）展示了在文本简化、背景信息生成和文本评估方面的显着能力。这激发了我们对使用LLMs生成和评估生物医学文章的平民总结进行系统性探索。我们提出了一种新颖的“先解释后总结”LS框架，利用LLMs生成高质量的背景知识以改进监督的LS。我们还评估LLMs在零-shot LS方面的性能，并提出了两种基于LLM的新颖LS评估方法。

    arXiv:2402.13498v1 Announce Type: new  Abstract: Automated lay summarisation (LS) aims to simplify complex technical documents into a more accessible format to non-experts. Existing approaches using pre-trained language models, possibly augmented with external background knowledge, tend to struggle with effective simplification and explanation. Moreover, automated methods that can effectively assess the `layness' of generated summaries are lacking. Recently, large language models (LLMs) have demonstrated a remarkable capacity for text simplification, background information generation, and text evaluation. This has motivated our systematic exploration into using LLMs to generate and evaluate lay summaries of biomedical articles. We propose a novel \textit{Explain-then-Summarise} LS framework, which leverages LLMs to generate high-quality background knowledge to improve supervised LS. We also evaluate the performance of LLMs for zero-shot LS and propose two novel LLM-based LS evaluation 
    
[^79]: GradSafe: 通过安全关键梯度分析检测LLMs中的不安全提示

    GradSafe: Detecting Unsafe Prompts for LLMs via Safety-Critical Gradient Analysis

    [https://arxiv.org/abs/2402.13494](https://arxiv.org/abs/2402.13494)

    GradSafe通过分析LLMs中关键安全参数的梯度，有效检测不安全提示，无需额外训练即可优于现有方法。

    

    大型语言模型（LLMs）面临来自不安全提示的威胁。现有的检测不安全提示的方法主要是在线内容审核API或微调LLMs。然而，这些策略通常需要大量和资源密集型的数据收集和训练过程。在这项研究中，我们提出了GradSafe，通过仔细审查LLMs中的安全关键参数的梯度有效地检测不安全提示。我们的方法基于一个关键观察：LLMs对于与合规响应配对的不安全提示的损失的梯度在某些安全关键参数上表现出相似的模式。相比之下，安全提示导致明显不同的梯度模式。基于这一观察，GradSafe分析来自提示（与合规响应配对）的梯度以准确地检测不安全提示。我们展示了，应用于Llama-2的GradSafe，无需进一步训练即可胜过Llama Guard。

    arXiv:2402.13494v1 Announce Type: new  Abstract: Large Language Models (LLMs) face threats from unsafe prompts. Existing methods for detecting unsafe prompts are primarily online moderation APIs or finetuned LLMs. These strategies, however, often require extensive and resource-intensive data collection and training processes. In this study, we propose GradSafe, which effectively detects unsafe prompts by scrutinizing the gradients of safety-critical parameters in LLMs. Our methodology is grounded in a pivotal observation: the gradients of an LLM's loss for unsafe prompts paired with compliance response exhibit similar patterns on certain safety-critical parameters. In contrast, safe prompts lead to markedly different gradient patterns. Building on this observation, GradSafe analyzes the gradients from prompts (paired with compliance responses) to accurately detect unsafe prompts. We show that GradSafe, applied to Llama-2 without further training, outperforms Llama Guard, despite its ex
    
[^80]: 《检索是有益还是有害？深入探讨检索增强对语言模型效果的影响》

    Retrieval Helps or Hurts? A Deeper Dive into the Efficacy of Retrieval Augmentation to Language Models

    [https://arxiv.org/abs/2402.13492](https://arxiv.org/abs/2402.13492)

    该研究深入探讨了如何通过检索增强语言模型，构建了新的QA数据集WiTQA，以实体和关系组合的影响为重点进行了详细分析。

    

    虽然大型语言模型（LMs）表现出色，但当需要查询其预训练记忆之外的信息时，它们在提供准确回答时会遇到挑战。虽然利用相关外部信息来增强它们可以缓解这些问题，但未考虑检索的必要性可能会对整体性能产生负面影响。此前的研究主要关注实体如何影响检索模型与LMs中的知识回忆，其他方面相对未被探索。本研究旨在通过探索实体和关系组合的影响来提供更详细、以事实为中心的分析。为实现这一目标，我们构建了一个名为WiTQA（Wikipedia Triple Question Answers）的新问题回答（QA）数据集。此数据集包括关于不同受欢迎程度实体和关系的问题，每个问题都附带一段支持性段落。

    arXiv:2402.13492v1 Announce Type: new  Abstract: While large language models (LMs) demonstrate remarkable performance, they encounter challenges in providing accurate responses when queried for information beyond their pre-trained memorization. Although augmenting them with relevant external information can mitigate these issues, failure to consider the necessity of retrieval may adversely affect overall performance. Previous research has primarily focused on examining how entities influence retrieval models and knowledge recall in LMs, leaving other aspects relatively unexplored. In this work, our goal is to offer a more detailed, fact-centric analysis by exploring the effects of combinations of entities and relations. To facilitate this, we construct a new question answering (QA) dataset called WiTQA (Wikipedia Triple Question Answers). This dataset includes questions about entities and relations of various popularity levels, each accompanied by a supporting passage. Our extensive ex
    
[^81]: ProPD：LLM并行解码的动态令牌树修剪和生成

    ProPD: Dynamic Token Tree Pruning and Generation for LLM Parallel Decoding

    [https://arxiv.org/abs/2402.13485](https://arxiv.org/abs/2402.13485)

    提出ProPD，一种基于动态令牌树修剪和生成的高效LLM并行解码框架，通过先进的提前修剪机制和动态令牌树生成算法来提高验证效率。

    

    近期生成式大型语言模型（LLMs）的进展显著提升了自然语言处理任务的性能。然而，它们的效率受到自回归令牌生成中固有限制的影响。尽管已经提出了带有令牌树验证的并行解码方法，例如Medusa，以改善解码并行性和效率，但由于其独立令牌预测方法以及大树大小和批处理时产生的显著验证开销，它经常难以保持上下文关系。在本文中，我们提出了ProPD，一种基于动态令牌树修剪和生成的高效LLM并行解码框架。ProPD具有一种先进的提前修剪机制，可以有效地消除不太可能的令牌序列以提高验证效率。此外，它引入了一种动态令牌树生成算法来平衡计算成本。

    arXiv:2402.13485v1 Announce Type: cross  Abstract: Recent advancements in generative large language models (LLMs) have significantly boosted the performance in natural language processing tasks. However, their efficiency is hampered by the inherent limitations in autoregressive token generation. While parallel decoding with token tree verification, e.g., Medusa, has been proposed to improve decoding parallelism and efficiency, it often struggles with maintaining contextual relationships due to its independent token prediction approach and incurs significant verification overhead, especially with large tree sizes and batch processing. In this paper, we propose ProPD, an efficient LLM parallel decoding framework based on dynamic token tree pruning and generation. ProPD features an advanced early pruning mechanism to efficiently eliminate unpromising token sequences to improve verification efficiency. Additionally, it introduces a dynamic token tree generation algorithm to balance the com
    
[^82]: 用于低资源领域任务的检索增强数据增强

    Retrieval-Augmented Data Augmentation for Low-Resource Domain Tasks

    [https://arxiv.org/abs/2402.13482](https://arxiv.org/abs/2402.13482)

    提出了一种用于低资源领域任务的新方法，通过结合来自其他数据集的相关示例来增强训练数据，以解决在低资源环境中生成样本不够理想和缺乏多样性的挑战

    

    尽管最近语言模型在多样任务上取得了巨大成功，但在训练数据有限的低资源环境中，它们的性能会严重下降。许多现有作品通过从训练数据生成合成数据，然后在其上训练模型来解决这个问题，最近使用大型语言模型（LLM）进行。然而，在低资源环境中，用于数据增强的种子数据样本数量非常少，这使得生成的样本不够理想且缺乏多样性。为了解决这一挑战，我们提出了一种新颖的方法，通过将其他数据集中丰富的示例与给定的训练数据结合起来，来增强训练数据。具体来说，我们首先通过与给定种子数据相似性基于其他数据集检索相关实例，例如它们的输入-输出对或上下文，然后提示LLM使用上下文信息生成新样本。

    arXiv:2402.13482v1 Announce Type: cross  Abstract: Despite large successes of recent language models on diverse tasks, they suffer from severe performance degeneration in low-resource settings with limited training data available. Many existing works tackle this problem by generating synthetic data from the training data and then training models on them, recently using Large Language Models (LLMs). However, in low-resource settings, the amount of seed data samples to use for data augmentation is very small, which makes generated samples suboptimal and less diverse. To tackle this challenge, we propose a novel method that augments training data by incorporating a wealth of examples from other datasets, along with the given training data. Specifically, we first retrieve the relevant instances from other datasets, such as their input-output pairs or contexts, based on their similarities with the given seed data, and then prompt LLMs to generate new samples with the contextual information 
    
[^83]: 语言模型和生物医学关系提取中的领域特异性有多重要？

    How Important is Domain Specificity in Language Models and Instruction Finetuning for Biomedical Relation Extraction?

    [https://arxiv.org/abs/2402.13470](https://arxiv.org/abs/2402.13470)

    研究探讨了在生物医学关系提取任务中领域特异性对于语言模型和指导微调的重要性，对比了在生物医学领域与通用领域训练的模型效果，并探讨了在生物医学数据集上指导微调的模型在性能上的优势。

    

    高价值、数据丰富的生物医学领域常常会使用最前沿的通用自然语言处理技术。过去几年来，生成式语言模型、指导微调和少样本学习成为自然语言处理研究的焦点。因此，预训练于生物医学语料库的生成式语言模型不断涌现，同时也尝试对生物医学指导微调，希望领域特异性可以改善下游任务的性能。鉴于训练这些模型所需的非平凡努力，我们研究它们在关系提取这一关键生物医学自然语言处理任务中是否存在任何益处。具体来说，我们探讨了两个问题：(1) 在生物医学语料库上训练的语言模型是否优于在通用领域语料库上训练的模型？(2) 在生物医学数据集上进行指导微调的模型是否优于在各种数据集上进行微调或者仅仅预训练的模型？我们解决这些问题。

    arXiv:2402.13470v1 Announce Type: new  Abstract: Cutting edge techniques developed in the general NLP domain are often subsequently applied to the high-value, data-rich biomedical domain. The past few years have seen generative language models (LMs), instruction finetuning, and few-shot learning become foci of NLP research. As such, generative LMs pretrained on biomedical corpora have proliferated and biomedical instruction finetuning has been attempted as well, all with the hope that domain specificity improves performance on downstream tasks. Given the nontrivial effort in training such models, we investigate what, if any, benefits they have in the key biomedical NLP task of relation extraction. Specifically, we address two questions: (1) Do LMs trained on biomedical corpora outperform those trained on general domain corpora? (2) Do models instruction finetuned on biomedical datasets outperform those finetuned on assorted datasets or those simply pretrained? We tackle these questions
    
[^84]: STENCIL：基于次模互信息的冷启动主动学习弱监督

    STENCIL: Submodular Mutual Information Based Weak Supervision for Cold-Start Active Learning

    [https://arxiv.org/abs/2402.13468](https://arxiv.org/abs/2402.13468)

    STENCIL利用次模互信息选择弱标记的稀有类实例，并通过标注者强标记，提高了文本分类数据集上的准确率和稀有类F-1分数。

    

    随着在NLP应用中对预训练模型进行监督微调越来越受欢迎，需要更大量的标注数据，特别是在大型语言模型的参数计数增加时。主动学习试图挖掘和注释未标记的实例以最大限度地快速改善模型性能，是减少注释成本的常见选择；然而，大多数方法通常忽视类别不平衡，并且要么假设可以访问初始标注数据，要么要求改进稀有类之前需要多轮主动学习选择。我们提出了STENCIL，它利用一组文本示例和最近提出的次模互信息来选择一组弱标记的稀有类实例，然后由标注者对其进行强标记。我们展示了STENCIL在多个文本分类数据集上将整体准确率提高了10%-24%，将稀有类F-1分数提高了17%-40%。

    arXiv:2402.13468v1 Announce Type: cross  Abstract: As supervised fine-tuning of pre-trained models within NLP applications increases in popularity, larger corpora of annotated data are required, especially with increasing parameter counts in large language models. Active learning, which attempts to mine and annotate unlabeled instances to improve model performance maximally fast, is a common choice for reducing the annotation cost; however, most methods typically ignore class imbalance and either assume access to initial annotated data or require multiple rounds of active learning selection before improving rare classes. We present STENCIL, which utilizes a set of text exemplars and the recently proposed submodular mutual information to select a set of weakly labeled rare-class instances that are then strongly labeled by an annotator. We show that STENCIL improves overall accuracy by $10\%-24\%$ and rare-class F-1 score by $17\%-40\%$ on multiple text classification datasets over commo
    
[^85]: RefuteBench：评估用于大型语言模型的反驳指令遵循

    RefuteBench: Evaluating Refuting Instruction-Following for Large Language Models

    [https://arxiv.org/abs/2402.13463](https://arxiv.org/abs/2402.13463)

    本文提出了一个名为RefuteBench的基准测试，旨在评估大型语言模型对反驳指令的遵循能力，发现LLMs倾向于固执于其内部知识而无法遵从用户反馈。

    

    大型语言模型（LLMs）的应用范围日益扩大。在实际使用中，用户可能根据模型的输出提供反馈，希望得到一个可以根据他们的反馈完成响应的响应模型。然而，模型能否恰当地响应用户的反驳反馈并始终执行下去尚未得到彻底分析。基于这一问题，本文提出了一个全面的基准测试，RefuteBench，涵盖了诸如问答、机器翻译和电子邮件撰写等任务。评估旨在评估模型是否能够积极接受反驳指令形式的反馈，并是否能够在对话中始终遵循用户需求。我们对众多LLMs进行了评估，并发现LLMs倾向固执，即倾向于其内部知识，经常未能遵守用户反馈。

    arXiv:2402.13463v1 Announce Type: cross  Abstract: The application scope of large language models (LLMs) is increasingly expanding. In practical use, users might provide feedback based on the model's output, hoping for a responsive model that can complete responses according to their feedback. Whether the model can appropriately respond to users' refuting feedback and consistently follow through with execution has not been thoroughly analyzed. In light of this, this paper proposes a comprehensive benchmark, RefuteBench, covering tasks such as question answering, machine translation, and email writing. The evaluation aims to assess whether models can positively accept feedback in form of refuting instructions and whether they can consistently adhere to user demands throughout the conversation. We conduct evaluations on numerous LLMs and find that LLMs are stubborn, i.e. exhibit inclination to their internal knowledge, often failing to comply with user feedback. Additionally, as the leng
    
[^86]: 模型编辑在社交去偏见中的潜力与挑战

    Potential and Challenges of Model Editing for Social Debiasing

    [https://arxiv.org/abs/2402.13462](https://arxiv.org/abs/2402.13462)

    模型编辑方法在社交去偏见中具有潜力，但也面临挑战，尤其是在支持不同偏见类型和理解编辑方法应用于去偏见过程中的利弊方面。

    

    在大量语料库上训练的大语言模型（LLMs）不可避免地存在刻板印象偏见。通过微调来减轻这些偏见可能既昂贵又需要大量数据。模型编辑方法专注于以事后方式修改LLMs，对于解决去偏见问题具有巨大潜力。然而，缺乏支持各种偏见类型，并了解应用编辑方法于去偏见过程中的利弊的综合研究。为填补这一差距，我们将社交去偏见仔细构建为一个编辑问题，并在刻板印象去偏见上对七种现有的模型编辑算法进行基准测试，即去偏见编辑。我们在三种情景下的研究结果展示了去偏见编辑的潜力与挑战：（1）现有的模型编辑方法可以有效保留知识并减轻偏见，同时也揭示了去偏见效果从编辑到应用的一般化过程。

    arXiv:2402.13462v1 Announce Type: cross  Abstract: Large language models (LLMs) trained on vast corpora suffer from inevitable stereotype biases. Mitigating these biases with fine-tuning could be both costly and data-hungry. Model editing methods, which focus on modifying LLMs in a post-hoc manner, are of great potential to address debiasing. However, it lacks a comprehensive study that facilitates both internal and external model editing methods, supports various bias types, as well as understands the pros and cons of applying editing methods to stereotypical debiasing. To mitigate this gap, we carefully formulate social debiasing into an editing problem and benchmark seven existing model editing algorithms on stereotypical debiasing, i.e., debias editing. Our findings in three scenarios reveal both the potential and challenges of debias editing: (1) Existing model editing methods can effectively preserve knowledge and mitigate biases, while the generalization of debias effect from ed
    
[^87]: 学习在指导调优期间操纵大型语言模型

    Learning to Poison Large Language Models During Instruction Tuning

    [https://arxiv.org/abs/2402.13459](https://arxiv.org/abs/2402.13459)

    通过设计新的数据注入攻击攻击LLMs，并提出一种梯度引导后门触发器学习方法，通过实验验证表明成功地破坏模型输出，仅改变1%的指导调优样本即可导致性能下降率达到约80％。

    

    大型语言模型（LLMs）的出现标志着语言处理和推理能力方面的重大突破。虽然它们取得了显著进展，但LLMs面临着数据注入攻击的漏洞，其中对手将后门触发器插入训练数据，以操纵输出以进行恶意行为。本研究通过设计一种新的数据注入攻击，旨在利用指导调优过程，进一步识别LLMs中的额外安全风险。我们提出了一种新颖的梯度引导后门触发器学习方法，以有效识别敌对触发器，确保对传统防御手段的规避，同时保持内容的完整性。通过对各种LLMs和任务的实验验证，我们的策略表明在破坏模型输出方面取得了很高的成功率；仅对4,000个指导调优样本中的1％进行注入就导致性能降低率（PDR）约为80％。我们的工作高

    arXiv:2402.13459v1 Announce Type: cross  Abstract: The advent of Large Language Models (LLMs) has marked significant achievements in language processing and reasoning capabilities. Despite their advancements, LLMs face vulnerabilities to data poisoning attacks, where adversaries insert backdoor triggers into training data to manipulate outputs for malicious purposes. This work further identifies additional security risks in LLMs by designing a new data poisoning attack tailored to exploit the instruction tuning process. We propose a novel gradient-guided backdoor trigger learning approach to identify adversarial triggers efficiently, ensuring an evasion of detection by conventional defenses while maintaining content integrity. Through experimental validation across various LLMs and tasks, our strategy demonstrates a high success rate in compromising model outputs; poisoning only 1\% of 4,000 instruction tuning samples leads to a Performance Drop Rate (PDR) of around 80\%. Our work high
    
[^88]: 基于Twitter数据的精神健康监测框架：从当地推文到当地健康

    LocalTweets to LocalHealth: A Mental Health Surveillance Framework Based on Twitter Data

    [https://arxiv.org/abs/2402.13452](https://arxiv.org/abs/2402.13452)

    本研究提出了一个新的基于Twitter数据的框架LocalHealth，用于预测当地精神健康结果。通过与GPT3.5结合使用，该框架在MH监测中取得了显著的改进。

    

    先前关于Twitter数据的研究已经提供了它在开发补充健康监测系统方面的实用性证据。在这项研究中，我们提出了一个新的框架来监测公共健康，重点关注精神健康（MH）结果。我们假设当地发布的推文可以表明当地的精神健康结果，并收集了来自美国765个地区（人口普查分组）的推文。我们将每个地区的这些推文与疾病控制中心（CDC）报告的相应MH结果配对，创建了一个基准数据集LocalTweets。借助LocalTweets，我们提出了基于Twitter的MH监测系统的首个人口级评估任务。随后，我们开发了一个高效有效的方法LocalHealth，用于根据LocalTweets预测MH结果。当与GPT3.5一起使用时，LocalHealth实现了最高的F1值和准确率，分别达到0.7429和79.78\%，F1值提高了59\%。

    arXiv:2402.13452v1 Announce Type: cross  Abstract: Prior research on Twitter (now X) data has provided positive evidence of its utility in developing supplementary health surveillance systems. In this study, we present a new framework to surveil public health, focusing on mental health (MH) outcomes. We hypothesize that locally posted tweets are indicative of local MH outcomes and collect tweets posted from 765 neighborhoods (census block groups) in the USA. We pair these tweets from each neighborhood with the corresponding MH outcome reported by the Center for Disease Control (CDC) to create a benchmark dataset, LocalTweets. With LocalTweets, we present the first population-level evaluation task for Twitter-based MH surveillance systems. We then develop an efficient and effective method, LocalHealth, for predicting MH outcomes based on LocalTweets. When used with GPT3.5, LocalHealth achieves the highest F1-score and accuracy of 0.7429 and 79.78\%, respectively, a 59\% improvement in F
    
[^89]: 朝着无需训练的巨型语言模型与训练自由化的关联内存模块

    CAMELoT: Towards Large Language Models with Training-Free Consolidated Associative Memory

    [https://arxiv.org/abs/2402.13449](https://arxiv.org/abs/2402.13449)

    引入了一个关联内存模块，可以无需重新训练即可与任何预先训练的大型语言模型耦合，解决了长输入序列处理问题，并在长上下文建模中显著降低困惑度。

    

    大型语言模型在处理长输入序列时面临内存和运行时间成本高的问题。增强记忆模型已经成为解决这一问题的有希望的方法，但目前的方法受限于有限的记忆容量，并且需要昂贵的重新训练才能与新的LLM集成。在这项工作中，我们引入了一个关联内存模块，可以与任何预先训练（冻结）的基于注意力的LLM耦合，无需重新训练，使其能够处理任意长的输入序列。与先前的方法不同，我们的关联内存模块将单个标记的表示合并到一个非参数分布模型中，通过适当平衡传入数据的新颖性和最近性进行动态管理。通过从这个合并的关联内存中检索信息，基本LLM可以在长上下文建模中显著减少（高达Arxiv的29.7％）与其他基线评估相比的困惑度。

    arXiv:2402.13449v1 Announce Type: new  Abstract: Large Language Models (LLMs) struggle to handle long input sequences due to high memory and runtime costs. Memory-augmented models have emerged as a promising solution to this problem, but current methods are hindered by limited memory capacity and require costly re-training to integrate with a new LLM. In this work, we introduce an associative memory module which can be coupled to any pre-trained (frozen) attention-based LLM without re-training, enabling it to handle arbitrarily long input sequences. Unlike previous methods, our associative memory module consolidates representations of individual tokens into a non-parametric distribution model, dynamically managed by properly balancing the novelty and recency of the incoming data. By retrieving information from this consolidated associative memory, the base LLM can achieve significant (up to 29.7% on Arxiv) perplexity reduction in long-context modeling compared to other baselines evalua
    
[^90]: ED-Copilot: 使用语言模型诊断辅助减少急诊科等待时间

    ED-Copilot: Reduce Emergency Department Wait Time with Language Model Diagnostic Assistance

    [https://arxiv.org/abs/2402.13448](https://arxiv.org/abs/2402.13448)

    本研究提出了一种在急诊科中减少等待时间的诊断辅助方法，利用人工智能系统帮助医生进行快速准确的诊断，并开发了ED-Copilot系统来推荐实验室检测并进行诊断预测。

    

    在急诊科（ED）中，患者在诊断前需要进行分诊和多种实验室检测。这个过程耗时，导致急诊科拥挤，显著影响患者死亡率、医疗错误、人员枯竭等。本研究提出了一种（时间）成本有效的诊断辅助方法，探索人工智能系统在协助急诊科临床医生进行高效准确诊断方面的潜力。使用公开可获得的患者数据，我们与急诊科临床医生合作策划了MIMIC-ED-Assist，这是一个衡量人工智能系统在建议最大程度减少急诊等待时间的实验室检测，并在正确预测诸如死亡之类关键结果方面的能力的基准。我们开发了ED-Copilot，它依次建议患者特定的实验室检测并进行诊断预测。ED-Copilot使用预训练的生物医学语言模型对患者信息进行编码并进行增强学习。

    arXiv:2402.13448v1 Announce Type: cross  Abstract: In the emergency department (ED), patients undergo triage and multiple laboratory tests before diagnosis. This process is time-consuming, and causes ED crowding which significantly impacts patient mortality, medical errors, staff burnout, etc. This work proposes (time) cost-effective diagnostic assistance that explores the potential of artificial intelligence (AI) systems in assisting ED clinicians to make time-efficient and accurate diagnoses. Using publicly available patient data, we collaborate with ED clinicians to curate MIMIC-ED-Assist, a benchmark that measures the ability of AI systems in suggesting laboratory tests that minimize ED wait times, while correctly predicting critical outcomes such as death. We develop ED-Copilot which sequentially suggests patient-specific laboratory tests and makes diagnostic predictions. ED-Copilot uses a pre-trained bio-medical language model to encode patient information and reinforcement learn
    
[^91]: 大型语言模型用于数据标注：一项调查

    Large Language Models for Data Annotation: A Survey

    [https://arxiv.org/abs/2402.13446](https://arxiv.org/abs/2402.13446)

    大型语言模型的出现为自动化数据标注提供机遇，该调查独特关注LLM在数据标注中的效用，贡献主要集中在LLM-Based数据标注、评估LLM生成的标注以及使用LLM生成的标注学习等三个核心方面。

    

    数据标注是将原始数据标记或打标签与相关信息，对于提高机器学习模型的有效性至关重要。然而，这一过程劳动密集且昂贵。先进的大型语言模型（LLMs）的出现，例如GPT-4，为革新和自动化数据标注的复杂过程提供了前所未有的机遇。虽然现有的调查已经广泛涵盖了LLM的架构、训练和一般应用，但本文独特地关注它们在数据标注中的具体效用。该调查对LLM-Based数据标注、评估LLM生成的标注以及使用LLM生成的标注学习这三个核心方面做出了贡献。此外，论文包括了一种使用LLMs进行数据标注的方法学深度分类法，一个对整合LLM生成的标注的模型的学习策略进行全面审查，以及对其进行详细讨论。

    arXiv:2402.13446v1 Announce Type: new  Abstract: Data annotation is the labeling or tagging of raw data with relevant information, essential for improving the efficacy of machine learning models. The process, however, is labor-intensive and expensive. The emergence of advanced Large Language Models (LLMs), exemplified by GPT-4, presents an unprecedented opportunity to revolutionize and automate the intricate process of data annotation. While existing surveys have extensively covered LLM architecture, training, and general applications, this paper uniquely focuses on their specific utility for data annotation. This survey contributes to three core aspects: LLM-Based Data Annotation, Assessing LLM-generated Annotations, and Learning with LLM-generated annotations. Furthermore, the paper includes an in-depth taxonomy of methodologies employing LLMs for data annotation, a comprehensive review of learning strategies for models incorporating LLM-generated annotations, and a detailed discussi
    
[^92]: 结构化树对齐用于（语音）组块分析评估

    Structured Tree Alignment for Evaluation of (Speech) Constituency Parsing

    [https://arxiv.org/abs/2402.13433](https://arxiv.org/abs/2402.13433)

    提出了一种受语音解析器评估问题启发的结构化句法分析树相似性度量指标STRUCT-IOU，有效地比较了口语词边界上的组块分析树与书面词上基准解析之间的差异，并展示了在文本组块分析评估中的优越性。

    

    我们提出了结构化平均交集-联盟比（STRUCT-IOU），这是一种句法分析树之间的相似性度量指标，受到了评估语音解析器问题的启发。STRUCT-IOU使得可以比较在自动识别的口语词边界上的组块分析树与基准解析（在书面词上）之间的差异。为了计算这个指标，我们通过强制对齐将基准解析树投影到语音领域，将投影的基准成分与预测的成分在一定的结构约束下对齐，然后计算所有对齐成分对之间的平均IOU分数。STRUCT-IOU考虑了词边界，并克服了预测的词和基准事实可能没有完美一一对应的挑战。扩展到文本组块分析的评估，我们展示STRUCT-IOU表现出更高的对句法合理解析的容忍度。

    arXiv:2402.13433v1 Announce Type: new  Abstract: We present the structured average intersection-over-union ratio (STRUCT-IOU), a similarity metric between constituency parse trees motivated by the problem of evaluating speech parsers. STRUCT-IOU enables comparison between a constituency parse tree (over automatically recognized spoken word boundaries) with the ground-truth parse (over written words). To compute the metric, we project the ground-truth parse tree to the speech domain by forced alignment, align the projected ground-truth constituents with the predicted ones under certain structured constraints, and calculate the average IOU score across all aligned constituent pairs. STRUCT-IOU takes word boundaries into account and overcomes the challenge that the predicted words and ground truth may not have perfect one-to-one correspondence. Extending to the evaluation of text constituency parsing, we demonstrate that STRUCT-IOU shows higher tolerance to syntactically plausible parses 
    
[^93]: DrBenchmark: 一个针对法语生物医学领域的大型语言理解评估基准

    DrBenchmark: A Large Language Understanding Evaluation Benchmark for French Biomedical Domain

    [https://arxiv.org/abs/2402.13432](https://arxiv.org/abs/2402.13432)

    DrBenchmark提出了一个针对法语生物医学领域的大型语言理解评估基准，旨在弥补对最新法语生物医学模型评估的不足，并考虑到法语的独特敏感性。

    

    生物医学领域在自然语言处理（NLP）领域引起了极大的兴趣，随着预训练语言模型（PLMs）的实质性进展。然而，由于不同模型之间评估协议的变化，比较这些模型已经变得具有挑战性。一个公平的解决方案是将不同的下游任务聚合到一个基准中，允许从各种角度评估PLMs的内在品质。尽管这一倡议仍然局限于少数语言，特别是英语和中文，但已经在生物医学领域展开。这一限制阻碍了对最新的法语生物医学模型的评价，因为它们要么在少量任务上进行评估，而且使用的协议不够标准化，要么使用一般的下游任务进行评估。为弥补这一研究差距，并考虑到法语的独特敏感性，我们提出了首个公开可用的法语生物医学基准

    arXiv:2402.13432v1 Announce Type: cross  Abstract: The biomedical domain has sparked a significant interest in the field of Natural Language Processing (NLP), which has seen substantial advancements with pre-trained language models (PLMs). However, comparing these models has proven challenging due to variations in evaluation protocols across different models. A fair solution is to aggregate diverse downstream tasks into a benchmark, allowing for the assessment of intrinsic PLMs qualities from various perspectives. Although still limited to few languages, this initiative has been undertaken in the biomedical field, notably English and Chinese. This limitation hampers the evaluation of the latest French biomedical models, as they are either assessed on a minimal number of tasks with non-standardized protocols or evaluated using general downstream tasks. To bridge this research gap and account for the unique sensitivities of French, we present the first-ever publicly available French biom
    
[^94]: 解释研究论文之间的关系

    Explaining Relationships Among Research Papers

    [https://arxiv.org/abs/2402.13426](https://arxiv.org/abs/2402.13426)

    探索了一种基于特征的LLM提示方法，用于生成丰富的引文文本，并一次生成多个引文以捕捉研究论文之间的复杂关系。

    

    由于研究出版物的快速增长，即使使用每日提要工具，跟上所有最新相关论文也是非常耗时的。需要自动生成的简短、定制的文献综述，帮助研究人员决定要阅读什么。本文探讨了一种基于特征的LLM提示方法，用于生成更丰富的引文文本，同时一次生成多个引文以捕捉研究论文之间的复杂关系。

    arXiv:2402.13426v1 Announce Type: new  Abstract: Due to the rapid pace of research publications, keeping up to date with all the latest related papers is very time-consuming, even with daily feed tools. There is a need for automatically generated, short, customized literature reviews of sets of papers to help researchers decide what to read. While several works in the last decade have addressed the task of explaining a single research paper, usually in the context of another paper citing it, the relationship among multiple papers has been ignored; prior works have focused on generating a single citation sentence in isolation, without addressing the expository and transition sentences needed to connect multiple papers in a coherent story. In this work, we explore a feature-based, LLM-prompting approach to generate richer citation texts, as well as generating multiple citations at once to capture the complex relationships among research papers. We perform an expert evaluation to investig
    
[^95]: 结构引导提示: 通过探索文本的图结构，在多步推理中指导大型语言模型

    Structure Guided Prompt: Instructing Large Language Model in Multi-Step Reasoning by Exploring Graph Structure of the Text

    [https://arxiv.org/abs/2402.13415](https://arxiv.org/abs/2402.13415)

    本论文介绍了一种结构引导提示框架，旨在通过探索文本的图结构，指导大型语言模型进行多步推理，以解决推理过程中的复杂关系和多样性带来的困难。

    

    虽然大型语言模型(LLMs)擅长处理直接推理任务，但在面对更复杂的多步推理时经常会遇到困难，原因有多种。首先，自然语言中经常包含实体之间的复杂关系，使得在较长的范围内保持清晰的推理链变得具有挑战性。其次，语言多样性的丰富意味着相同的实体和关系可以用不同的术语和结构表达，使得识别和建立多个信息片段之间的连接任务变得复杂。图提供了一种有效的解决方案，可以表示富含关系信息的数据，并捕捉实体之间的长期依赖关系。为了利用图的潜力，本文介绍了结构引导提示，这是一个创新的三阶段任务无关提示框架，旨在改善多步推理。

    arXiv:2402.13415v1 Announce Type: new  Abstract: Although Large Language Models (LLMs) excel at addressing straightforward reasoning tasks, they frequently struggle with difficulties when confronted by more complex multi-step reasoning due to a range of factors. Firstly, natural language often encompasses complex relationships among entities, making it challenging to maintain a clear reasoning chain over longer spans. Secondly, the abundance of linguistic diversity means that the same entities and relationships can be expressed using different terminologies and structures, complicating the task of identifying and establishing connections between multiple pieces of information. Graphs provide an effective solution to represent data rich in relational information and capture long-term dependencies among entities. To harness the potential of graphs, our paper introduces Structure Guided Prompt, an innovative three-stage task-agnostic prompting framework designed to improve the multi-step 
    
[^96]: 将大型语言模型用作事后校正器

    Harnessing Large Language Models as Post-hoc Correctors

    [https://arxiv.org/abs/2402.13414](https://arxiv.org/abs/2402.13414)

    通过提出的无需训练的框架 LlmCorr，本文展示了一个LLM可以作为事后校正器，为任意ML模型的预测提出修正。

    

    随着机器学习（ML）模型的规模增长并需求更高质量的训练数据，与对这些模型进行重新训练和微调相关的费用正在迅速增加。受最近大型语言模型（LLMs）在不同领域取得的令人瞩目成就启发，本文探讨了一个问题：LLMs能否以极低成本有效地改善ML的性能？我们展示了，通过我们提出的无需训练的框架 LlmCorr，一个LLM可以作为事后校正器，为任意ML模型的预测提出修正。特别是，我们通过整合数据集的标签信息和ML模型对验证集的预测来形成一个上下文知识数据库。利用LLMs的上下文学习能力，我们要求LLM总结ML模型犯错误的实例以及主要预测与真实标签之间的相关性。随后，LLM可以

    arXiv:2402.13414v1 Announce Type: cross  Abstract: As Machine Learning (ML) models grow in size and demand higher-quality training data, the expenses associated with re-training and fine-tuning these models are escalating rapidly. Inspired by recent impressive achievements of Large Language Models (LLMs) in different fields, this paper delves into the question: can LLMs efficiently improve an ML's performance at a minimal cost? We show that, through our proposed training-free framework LlmCorr, an LLM can work as a post-hoc corrector to propose corrections for the predictions of an arbitrary ML model. In particular, we form a contextual knowledge database by incorporating the dataset's label information and the ML model's predictions on the validation dataset. Leveraging the in-context learning capability of LLMs, we ask the LLM to summarise the instances in which the ML model makes mistakes and the correlation between primary predictions and true labels. Following this, the LLM can tr
    
[^97]: Healthcare Copilot: 启动通用LLM的力量进行医疗咨询

    Healthcare Copilot: Eliciting the Power of General LLMs for Medical Consultation

    [https://arxiv.org/abs/2402.13408](https://arxiv.org/abs/2402.13408)

    该论文介绍了一种旨在增强和定制大型语言模型（LLMs）以进行医疗咨询的Healthcare Copilot框架，其包括对话组件、记忆组件和处理组件。通过实施自动评估方案，结果表明该Healthcare Copilot能够显著改善医疗咨询质量。

    

    arXiv:2402.13408v1 公告类型: 新的 摘要: 旨在增强和定制大型语言模型（LLMs）以针对特定复杂任务而无需精细调整的副驾驶框架正受到社区越来越多的关注。在本文中，我们介绍了为医疗咨询而设计的Healthcare Copilot的构建。所提出的Healthcare Copilot由三个主要组件组成: 1) 对话组件，负责与患者进行有效且安全的交互; 2) 存储当前对话数据和历史患者信息的记忆组件; 3) 处理组件，总结整个对话并生成报告。为了评估提出的Healthcare Copilot，我们使用ChatGPT实施了一个自动评估方案，担任两个角色: 与副驾驶进行对话的虚拟患者，以及评估者，以评估对话质量。广泛的结果表明，所提出的Healthcare Copilot显着地...

    arXiv:2402.13408v1 Announce Type: new  Abstract: The copilot framework, which aims to enhance and tailor large language models (LLMs) for specific complex tasks without requiring fine-tuning, is gaining increasing attention from the community. In this paper, we introduce the construction of a Healthcare Copilot designed for medical consultation. The proposed Healthcare Copilot comprises three main components: 1) the Dialogue component, responsible for effective and safe patient interactions; 2) the Memory component, storing both current conversation data and historical patient information; and 3) the Processing component, summarizing the entire dialogue and generating reports. To evaluate the proposed Healthcare Copilot, we implement an auto-evaluation scheme using ChatGPT for two roles: as a virtual patient engaging in dialogue with the copilot, and as an evaluator to assess the quality of the dialogue. Extensive results demonstrate that the proposed Healthcare Copilot significantly e
    
[^98]: 一个统一的基于分类学指导的实体集扩展和分类学扩展的指导调整框架

    A Unified Taxonomy-Guided Instruction Tuning Framework for Entity Set Expansion and Taxonomy Expansion

    [https://arxiv.org/abs/2402.13405](https://arxiv.org/abs/2402.13405)

    通过统一的基于分类学指导的指导调整框架，本文提出了一种利用现有分类学进行实体关系微调的方法，有效解决实体集扩展、分类学扩展和种子引导分类学构建三个任务。

    

    实体集扩展、分类学扩展和种子引导分类学构建是三个代表性任务，可以用来自动向现有分类学填充新实体。然而，先前的方法通常使用异质技术分别解决这些任务，缺乏统一的视角。为了解决这个问题，在本文中，我们从分类学结构的视角确认了这些任务所需的共同关键技能——找到“兄弟”和找到“父母”，并提出了一个统一的基于分类学指导的指导调整框架来共同解决这三个任务。具体来说，通过利用现有分类学作为丰富的实体关系源，我们利用指导调整来微调大型语言模型，生成父母和兄弟实体。在多个基准数据集上的大量实验证明了TaxoInstruct的有效性，该方法在各项指标上均优于特定任务的基线方法。

    arXiv:2402.13405v1 Announce Type: new  Abstract: Entity Set Expansion, Taxonomy Expansion, and Seed-Guided Taxonomy Construction are three representative tasks that can be used to automatically populate an existing taxonomy with new entities. However, previous approaches often address these tasks separately with heterogeneous techniques, lacking a unified perspective. To tackle this issue, in this paper, we identify the common key skills needed for these tasks from the view of taxonomy structures -- finding 'siblings' and finding 'parents' -- and propose a unified taxonomy-guided instruction tuning framework to jointly solve the three tasks. To be specific, by leveraging the existing taxonomy as a rich source of entity relationships, we utilize instruction tuning to fine-tune a large language model to generate parent and sibling entities. Extensive experiments on multiple benchmark datasets demonstrate the effectiveness of TaxoInstruct, which outperforms task-specific baselines across 
    
[^99]: 可靠的基于LLM的面向任务对话系统用户模拟器

    Reliable LLM-based User Simulator for Task-Oriented Dialogue Systems

    [https://arxiv.org/abs/2402.13374](https://arxiv.org/abs/2402.13374)

    该论文介绍了DAUS，一个基于LLM的领域感知用户模拟器，通过在真实对话示例上进行微调，显著改进了用户目标实现，并有效减轻模拟器响应中的幻觉。

    

    在对话系统领域，用户模拟技术已经成为一个颠覆性的创新，重新定义了任务导向对话（TOD）系统的评估和增强。这些方法对于复制真实用户交互至关重要，实现了合成数据增强、错误检测和鲁棒评估等应用。然而，现有方法往往依赖于严格的基于规则的方法或已标记的数据。本文介绍了DAUS，一个领域感知用户模拟器。利用大型语言模型，我们在真实任务导向对话的示例上对DAUS进行了微调。在两个相关基准测试上的结果展示了用户目标实现方面的显著改进。值得注意的是，我们观察到微调增强了模拟器与用户目标的一致性，有效地缓解了幻觉——模拟器响应中一致性的主要来源。

    arXiv:2402.13374v1 Announce Type: new  Abstract: In the realm of dialogue systems, user simulation techniques have emerged as a game-changer, redefining the evaluation and enhancement of task-oriented dialogue (TOD) systems. These methods are crucial for replicating real user interactions, enabling applications like synthetic data augmentation, error detection, and robust evaluation. However, existing approaches often rely on rigid rule-based methods or on annotated data. This paper introduces DAUS, a Domain-Aware User Simulator. Leveraging large language models, we fine-tune DAUS on real examples of task-oriented dialogues. Results on two relevant benchmarks showcase significant improvements in terms of user goal fulfillment. Notably, we have observed that fine-tuning enhances the simulator's coherence with user goals, effectively mitigating hallucinations -- a major source of inconsistencies in simulator responses.
    
[^100]: EvoGrad：以人类对手为特点的Winograd Schema挑战的动态方法

    EvoGrad: A Dynamic Take on the Winograd Schema Challenge with Human Adversaries

    [https://arxiv.org/abs/2402.13372](https://arxiv.org/abs/2402.13372)

    EvoGrad是一个以人类对手为特点的用于解决Winograd Schema挑战的动态方法，通过人在环中方法创建动态数据集，拓展任务实例并引入错误深度度量标准，提出新的多样化常识推理数据集基准，揭示了当前语言模型在此类任务上的挑战。

    

    虽然大型语言模型（LLMs）在Winograd Schema Challenge（WSC）中表现出色，该任务通过代词消歧义测试常识推理，但它们对于包含轻微修改或改写的实例感到困难。为了解决这个问题，我们引入EvoGrad，这是一个开源平台，利用人在环中的方法来创建一个适用于这种修改后WSC实例的动态数据集。利用ChatGPT的功能，我们将我们的任务实例从182扩展到3,691个，为多样化的常识推理数据集设定了一个新的基准。此外，我们引入了错误深度度量标准，评估模型在动态任务中的稳定性。我们的结果强调了EvoGrad所提出的挑战：即使性能最佳的LLM，GPT-3.5，在平均错误深度为7.2的情况下仅达到65.0%的准确率，与人类92.8%的准确率形成了鲜明对比，人类准确率没有干扰性错误。这突显了持续存在的模型限制

    arXiv:2402.13372v1 Announce Type: new  Abstract: While Large Language Models (LLMs) excel at the Winograd Schema Challenge (WSC), a coreference resolution task testing common-sense reasoning through pronoun disambiguation, they struggle with instances that feature minor alterations or rewording. To address this, we introduce EvoGrad, an open-source platform that harnesses a human-in-the-loop approach to create a dynamic dataset tailored to such altered WSC instances. Leveraging ChatGPT's capabilities, we expand our task instances from 182 to 3,691, setting a new benchmark for diverse common-sense reasoning datasets. Additionally, we introduce the error depth metric, assessing model stability in dynamic tasks. Our results emphasize the challenge posed by EvoGrad: Even the best performing LLM, GPT-3.5, achieves an accuracy of 65.0% with an average error depth of 7.2, a stark contrast to human performance of 92. 8% accuracy without perturbation errors. This highlights ongoing model limita
    
[^101]: 一种简单而有效的方法，改善结构化语言模型在信息抽取中的输出

    A Simple but Effective Approach to Improve Structured Language Model Output for Information Extraction

    [https://arxiv.org/abs/2402.13364](https://arxiv.org/abs/2402.13364)

    该论文提出了一种名为G&O的方法，通过将内容生成与结构化过程分离，有效提升了大型语言模型在生成特定结构化文本上的性能。

    

    大型语言模型（LLMs）已经展示出在根据指令生成非结构化自然语言方面具有令人印象深刻的能力。然而，当要求它们生成符合特定结构化格式的文本时，它们的表现可能不一致，在命名实体识别（NER）或关系抽取（RE）等应用中这一点至关重要。为了解决这个问题，本文引入了一种高效的方法，G&O，以增强它们的结构化文本生成能力。它将生成分解为一个两步流程：首先，LLMs生成自然语言中的答案作为中间响应。随后，要求LLMs将输出组织成所需的结构，使用中间响应作为上下文。G&O有效地将内容生成与构建过程分离，减少了同时完成两个正交任务的压力。在零-shot NER和RE上进行测试，结果表明

    arXiv:2402.13364v1 Announce Type: new  Abstract: Large language models (LLMs) have demonstrated impressive abilities in generating unstructured natural language according to instructions. However, their performance can be inconsistent when tasked with producing text that adheres to specific structured formats, which is crucial in applications like named entity recognition (NER) or relation extraction (RE). To address this issue, this paper introduces an efficient method, G&O, to enhance their structured text generation capabilities. It breaks the generation into a two-step pipeline: initially, LLMs generate answers in natural language as intermediate responses. Subsequently, LLMs are asked to organize the output into the desired structure, using the intermediate responses as context. G&O effectively separates the generation of content from the structuring process, reducing the pressure of completing two orthogonal tasks simultaneously. Tested on zero-shot NER and RE, the results indica
    
[^102]: PIRB：波兰密集和混合文本检索方法的综合基准评估

    PIRB: A Comprehensive Benchmark of Polish Dense and Hybrid Text Retrieval Methods

    [https://arxiv.org/abs/2402.13350](https://arxiv.org/abs/2402.13350)

    PIRB提出了一个全面的波兰文本信息检索基准，包含41个任务，评估了超过20种密集和稀疏检索模型，并引入了一个三步训练流程来构建高效的特定语言检索器，最后验证了他们的方法的优越性

    

    我们介绍了波兰信息检索基准（PIRB），这是一个全面的评估框架，涵盖了波兰语的41个文本信息检索任务。该基准包括现有数据集以及10个新的、以前未公开的数据集，涵盖了医学、法律、商业、物理和语言学等多样主题。我们进行了超过20个密集和稀疏检索模型的广泛评估，包括我们训练的基准模型以及其他可用的波兰语和多语言方法。最后，我们介绍了一个用于训练高效特定语言检索器的三步流程，包括知识蒸馏、监督微调以及使用轻量级重新评分模型构建稀疏-密集混合检索器。为了验证我们的方法，我们为波兰语训练了新的文本编码器，并将其结果与先前评估过的方法进行了比较。我们的密集模型优于现有的最佳解决方案

    arXiv:2402.13350v1 Announce Type: new  Abstract: We present Polish Information Retrieval Benchmark (PIRB), a comprehensive evaluation framework encompassing 41 text information retrieval tasks for Polish. The benchmark incorporates existing datasets as well as 10 new, previously unpublished datasets covering diverse topics such as medicine, law, business, physics, and linguistics. We conduct an extensive evaluation of over 20 dense and sparse retrieval models, including the baseline models trained by us as well as other available Polish and multilingual methods. Finally, we introduce a three-step process for training highly effective language-specific retrievers, consisting of knowledge distillation, supervised fine-tuning, and building sparse-dense hybrid retrievers using a lightweight rescoring model. In order to validate our approach, we train new text encoders for Polish and compare their results with previously evaluated methods. Our dense models outperform the best solutions avai
    
[^103]: 通过简单的探测器聚合增强神经机器翻译中的幻觉检测

    Enhanced Hallucination Detection in Neural Machine Translation through Simple Detector Aggregation

    [https://arxiv.org/abs/2402.13331](https://arxiv.org/abs/2402.13331)

    提出了一种通过简单的探测器聚合来增强神经机器翻译中的幻觉检测，结果表明这种方法有效性，有望使机器翻译系统更加可靠。

    

    幻觉翻译在实际部署机器翻译系统时存在重大威胁和安全问题。本文提出通过结合不同的探测器和引入简单的聚合多探测器方法来解决单个探测器的局限性。我们的结果展示了我们聚合探测器的有效性，为更加可靠的机器翻译系统迈出了一步。

    arXiv:2402.13331v1 Announce Type: new  Abstract: Hallucinated translations pose significant threats and safety concerns when it comes to the practical deployment of machine translation systems. Previous research works have identified that detectors exhibit complementary performance different detectors excel at detecting different types of hallucinations. In this paper, we propose to address the limitations of individual detectors by combining them and introducing a straightforward method for aggregating multiple detectors. Our results demonstrate the efficacy of our aggregated detector, providing a promising step towards evermore reliable machine translation systems.
    
[^104]: 通过语义词汇资源增强现代监督词义消歧模型

    Enhancing Modern Supervised Word Sense Disambiguation Models by Semantic Lexical Resources

    [https://arxiv.org/abs/2402.13302](https://arxiv.org/abs/2402.13302)

    通过引入语义特征和多层架构，本研究提出了一种通过利用WordNet和WordNet Domains等语义词汇资源增强现代监督词义消歧模型的方法。

    

    目前，用于词义消歧（WSD）的监督模型在最流行的基准测试中取得了最先进的结果。尽管最近引入了词嵌入和循环神经网络以设计强大的上下文相关特征，但利用语义词汇资源（SLRs）改进WSD模型的兴趣主要局限于基于知识的方法。本文通过利用两种流行的SLRs：WordNet和WordNet Domains，增强了“现代”监督WSD模型。我们提出了一种有效的方式将语义特征引入分类器，并考虑使用SLR结构来增强训练数据。我们研究了不同类型的语义特征的作用，探讨它们与通过词嵌入或循环神经网络编码的本地上下文的交互作用，并将所提出的模型扩展为用于WSD的新型多层架构。进行了详细的实验比较。

    arXiv:2402.13302v1 Announce Type: new  Abstract: Supervised models for Word Sense Disambiguation (WSD) currently yield to state-of-the-art results in the most popular benchmarks. Despite the recent introduction of Word Embeddings and Recurrent Neural Networks to design powerful context-related features, the interest in improving WSD models using Semantic Lexical Resources (SLRs) is mostly restricted to knowledge-based approaches. In this paper, we enhance "modern" supervised WSD models exploiting two popular SLRs: WordNet and WordNet Domains. We propose an effective way to introduce semantic features into the classifiers, and we consider using the SLR structure to augment the training data. We study the effect of different types of semantic features, investigating their interaction with local contexts encoded by means of mixtures of Word Embeddings or Recurrent Neural Networks, and we extend the proposed model into a novel multi-layer architecture for WSD. A detailed experimental compa
    
[^105]: 结构引导的大型语言模型用于SQL生成

    Structure Guided Large Language Model for SQL Generation

    [https://arxiv.org/abs/2402.13284](https://arxiv.org/abs/2402.13284)

    通过引入结构信息，提出了一个结构引导的SQL生成模型，以改善大型语言模型生成SQL的准确性和可执行性。

    

    生成准确的结构化查询语言（SQL）是一个长期存在的问题，特别是在将用户的语义查询与结构化数据库匹配，然后生成结构化SQL方面。现有模型通常将查询和数据库模式输入到LLM中，并依赖LLM执行语义-结构匹配并生成结构化SQL。然而，这种解决方案忽略了用户查询和数据库中的结构信息，而这些信息可以用来增强结构化SQL的生成。这一疏忽可能导致不准确或无法执行的SQL生成。为了充分利用结构，我们提出了一个结构到SQL的框架，利用固有的结构信息来改善LLM的SQL生成。具体地，我们介绍了我们的结构引导SQL（SGU-SQL）生成模型。

    arXiv:2402.13284v1 Announce Type: cross  Abstract: Generating accurate Structured Querying Language (SQL) is a long-standing problem, especially in matching users' semantic queries with structured databases and then generating structured SQL. Existing models typically input queries and database schemas into the LLM and rely on the LLM to perform semantic-structure matching and generate structured SQL. However, such solutions overlook the structural information within user queries and databases, which can be utilized to enhance the generation of structured SQL. This oversight can lead to inaccurate or unexecutable SQL generation. To fully exploit the structure, we propose a structure-to-SQL framework, which leverages the inherent structure information to improve the SQL generation of LLMs. Specifically, we introduce our Structure Guided SQL~(SGU-SQL) generation model. SGU-SQL first links user queries and databases in a structure-enhanced manner. It then decomposes complicated linked str
    
[^106]: 如果LLM具有不同的世界观：使用基于LLM的代理模拟外星文明

    What if LLMs Have Different World Views: Simulating Alien Civilizations with LLM-based Agents

    [https://arxiv.org/abs/2402.13184](https://arxiv.org/abs/2402.13184)

    这项研究引入了“CosmoAgent”，利用LLM模拟人类和外星文明之间的复杂互动，评估和平共存的可行性，并量化评估文明的发展轨迹，同时考虑不同文明之间的巨大多样性。

    

    在这项研究中，我们介绍了“CosmoAgent”，这是一个创新的人工智能框架，利用大型语言模型（LLMs）来模拟人类与外星文明之间复杂的交互，特别强调史蒂芬·霍金关于不要随意向宇宙发送无线电信号的谨慎建议。该研究的目标是评估和平共存的可行性，同时考虑可能威胁善意文明的潜在风险。通过采用数学模型和状态转换矩阵，我们的方法定量评估文明的发展轨迹，为在关键增长和饱和点做出未来决策提供见解。此外，本文承认宇宙中潜在生活条件的巨大多样性可能会促进不同文明之间独特的宇宙观、道德准则和世界观。认识到地球上--

    arXiv:2402.13184v1 Announce Type: new  Abstract: In this study, we introduce "CosmoAgent," an innovative artificial intelligence framework utilizing Large Language Models (LLMs) to simulate complex interactions between human and extraterrestrial civilizations, with a special emphasis on Stephen Hawking's cautionary advice about not sending radio signals haphazardly into the universe. The goal is to assess the feasibility of peaceful coexistence while considering potential risks that could threaten well-intentioned civilizations. Employing mathematical models and state transition matrices, our approach quantitatively evaluates the development trajectories of civilizations, offering insights into future decision-making at critical points of growth and saturation. Furthermore, the paper acknowledges the vast diversity in potential living conditions across the universe, which could foster unique cosmologies, ethical codes, and worldviews among various civilizations. Recognizing the Earth-c
    
[^107]: CMDAG: 一个带有注释的中文隐喻数据集作为“CoT”来提升隐喻生成

    CMDAG: A Chinese Metaphor Dataset with Annotated Grounds as CoT for Boosting Metaphor Generation

    [https://arxiv.org/abs/2402.13145](https://arxiv.org/abs/2402.13145)

    本文介绍了一个大规模高质量的带注释中文隐喻语料库，强调隐喻生成中的基础及其独特特征，而非传统的对象和载体组合。

    

    隐喻是人类语言和文学中显著的修辞手法，因为它们增添了色彩、形象和强调，以增强有效交流。本文介绍了一个大规模高质量的带注释中文隐喻语料库，包括约28K句来自各种中文文学来源（如诗歌、散文、歌词等）。为确保注释的准确性和一致性，我们提出了一套全面的指南。这些指南涵盖了隐喻标注的方面，包括识别对象、载体和基础，以处理比喻、拟人、并列和夸张等复杂性。打破传统，我们的隐喻生成方法强调基础及其独特特征，而不是传统的对象和载体组合。通过将“基础”作为“CoT”（思维链）输入进行整合，我们能够生成重新

    arXiv:2402.13145v1 Announce Type: cross  Abstract: Metaphor is a prominent linguistic device in human language and literature, as they add color, imagery, and emphasis to enhance effective communication. This paper introduces a large-scale high quality annotated Chinese Metaphor Corpus, which comprises around 28K sentences drawn from a diverse range of Chinese literary sources, such as poems, prose, song lyrics, etc. To ensure the accuracy and consistency of our annotations, we introduce a comprehensive set of guidelines. These guidelines address the facets of metaphor annotation, including identifying tenors, vehicles, and grounds to handling the complexities of similes, personifications, juxtapositions, and hyperboles. Breaking tradition, our approach to metaphor generation emphasizes grounds and their distinct features rather than the conventional combination of tenors and vehicles. By integrating "ground" as a CoT (Chain of Thoughts) input, we are able to generate metaphors that re
    
[^108]: 用隐式文本摘要提高对话状态跟踪的有效性和效率

    Effective and Efficient Conversation Retrieval for Dialogue State Tracking with Implicit Text Summaries

    [https://arxiv.org/abs/2402.13043](https://arxiv.org/abs/2402.13043)

    使用文本摘要提高对话检索的有效性和效率，通过对话摘要生成器进行查询和关键词生成，进一步提炼轻量级对话编码器以避免额外推理成本

    

    arXiv:2402.13043v1 公告类型: 新 文摘: 基于大型语言模型（LLM）的小样本对话状态跟踪（DST）依赖于一个有效且高效的对话检索器来查找类似的上下文示例以进行提示学习。先前的作品使用原始对话上下文作为搜索键和查询，并通过对带注释的对话进行微调来实现卓越性能。然而，这种方法不太适合扩展到新的领域或新的注释语言，因为微调数据不可用。为解决这一问题，我们基于对话的文本摘要来处理对话检索任务。采用基于LLM的对话摘要生成器进行查询和关键词生成，实现了有效的最大内积搜索。为避免LLM基于对话摘要生成带来的额外推理成本，我们进一步提炼一个轻量级的对话编码器，该编码器在不解码测试对话摘要的情况下生成查询嵌入向量。

    arXiv:2402.13043v1 Announce Type: new  Abstract: Few-shot dialogue state tracking (DST) with Large Language Models (LLM) relies on an effective and efficient conversation retriever to find similar in-context examples for prompt learning. Previous works use raw dialogue context as search keys and queries, and a retriever is fine-tuned with annotated dialogues to achieve superior performance. However, the approach is less suited for scaling to new domains or new annotation languages, where fine-tuning data is unavailable. To address this problem, we handle the task of conversation retrieval based on text summaries of the conversations. A LLM-based conversation summarizer is adopted for query and key generation, which enables effective maximum inner product search. To avoid the extra inference cost brought by LLM-based conversation summarization, we further distill a light-weight conversation encoder which produces query embeddings without decoding summaries for test conversations. We val
    
[^109]: FormulaQA：一个基于公式的数值推理问题问答数据集

    FormulaQA: A Question Answering Dataset for Formula-Based Numerical Reasoning

    [https://arxiv.org/abs/2402.12692](https://arxiv.org/abs/2402.12692)

    FormulaQA是一个基于初中物理考试的公式驱动数值推理问题问答数据集，通过评估LLMs的不同方法和使用检索增强型LLMs以及对小型模型进行微调，揭示了现有模型在应对复杂、基于公式的FormulaQA时的潜在改进空间。

    

    应用公式是人类在解决数值推理问题时的基本能力。然而，现有的数值推理数据集很少明确指出推理步骤中使用的公式。为了弥补这一差距，我们提出了一个基于初中物理考试的公式驱动数值推理问题问答数据集FormulaQA。我们还使用大小从7B到超过100B参数的LLMs进行了零样本和少样本思维链方法的评估，并探索了在提供外部公式数据库时使用检索增强型LLMs的方法。我们还对大小不超过2B的较小模型进行了微调。我们的实证研究强调了当应用于我们复杂、基于公式的FormulaQA时，现有模型在改进方面具有显著潜力。

    arXiv:2402.12692v1 Announce Type: new  Abstract: The application of formulas is a fundamental ability of humans when addressing numerical reasoning problems. However, existing numerical reasoning datasets seldom explicitly indicate the formulas employed during the reasoning steps. To bridge this gap, we propose a question answering dataset for formula-based numerical reasoning called FormulaQA, from junior high school physics examinations. We further conduct evaluations on LLMs with size ranging from 7B to over 100B parameters utilizing zero-shot and few-shot chain-of-thoughts methods and we explored the approach of using retrieval-augmented LLMs when providing an external formula database. We also fine-tune on smaller models with size not exceeding 2B. Our empirical findings underscore the significant potential for improvement in existing models when applied to our complex, formula-driven FormulaQA.
    
[^110]: StyleDubber: 面向电影配音的多尺度风格学习

    StyleDubber: Towards Multi-Scale Style Learning for Movie Dubbing

    [https://arxiv.org/abs/2402.12636](https://arxiv.org/abs/2402.12636)

    StyleDubber提出了一种新的电影配音方法，通过在音素级别进行学习，解决了当前 V2C 模型中存在的音素发音不完整和身份稳定性差的问题。

    

    给定一份剧本，在电影配音（视觉语音克隆，V2C）中的挑战是根据参考音轨的语气，生成与视频在时间和情绪上都良好对齐的语音。现有的 V2C 模型根据视频帧间的间隔字断分割剧本的音素，这解决了时间对齐问题，但导致音素发音不完整和身份稳定性差。为了解决这个问题，我们提出 StyleDubber，它将配音学习从帧级别转为音素级别。它包含三个主要组件：（1）一个多模态风格适配器，以音素级别操作，从参考音频中学习发音风格，并生成受视频中呈现的面部情绪影响的中间表示；（2）一个以语句级别风格学习模块，引导中间表现的 mel-spectrogram 解码和细化过程。

    arXiv:2402.12636v1 Announce Type: new  Abstract: Given a script, the challenge in Movie Dubbing (Visual Voice Cloning, V2C) is to generate speech that aligns well with the video in both time and emotion, based on the tone of a reference audio track. Existing state-of-the-art V2C models break the phonemes in the script according to the divisions between video frames, which solves the temporal alignment problem but leads to incomplete phoneme pronunciation and poor identity stability. To address this problem, we propose StyleDubber, which switches dubbing learning from the frame level to phoneme level. It contains three main components: (1) A multimodal style adaptor operating at the phoneme level to learn pronunciation style from the reference audio, and generate intermediate representations informed by the facial emotion presented in the video; (2) An utterance-level style learning module, which guides both the mel-spectrogram decoding and the refining processes from the intermediate e
    
[^111]: 将废料变废为宝：矫正MoE的Top-k路由器

    Turn Waste into Worth: Rectifying Top-$k$ Router of MoE

    [https://arxiv.org/abs/2402.12399](https://arxiv.org/abs/2402.12399)

    提出了Rectify-Router解决了MoE模型中常用的Top-k路由机制所带来的令牌丢失和填充问题，通过Intra-GPU矫正和Fill-in矫正来实现。

    

    稀疏混合专家（MoE）模型因其计算效率而受到欢迎，用于训练大型语言模型。然而，常用的Top-k路由机制由于不平衡的路由导致冗余计算和内存成本过高。一些专家会溢出，其中超出的令牌会被丢弃。而一些专家是空闲的，这些专家会填充为零，负面影响了模型性能。为了解决丢弃令牌和填充问题，我们提出了Rectify-Router，包括Intra-GPU矫正和Fill-in矫正。Intra-GPU矫正处理丢弃的令牌，将它们有效地路由到GPU内的专家，避免跨GPU通信。Fill-in矫正通过用具有高路由分数的令牌替换填充令牌来解决填充问题。我们的实验结果表明，Intra-GPU矫正和Fill-in矫正

    arXiv:2402.12399v1 Announce Type: cross  Abstract: Sparse Mixture of Experts (MoE) models are popular for training large language models due to their computational efficiency. However, the commonly used top-$k$ routing mechanism suffers from redundancy computation and memory costs due to the unbalanced routing. Some experts are overflow, where the exceeding tokens are dropped. While some experts are vacant, which are padded with zeros, negatively impacting model performance. To address the dropped tokens and padding, we propose the Rectify-Router, comprising the Intra-GPU Rectification and the Fill-in Rectification. The Intra-GPU Rectification handles dropped tokens, efficiently routing them to experts within the GPU where they are located to avoid inter-GPU communication. The Fill-in Rectification addresses padding by replacing padding tokens with the tokens that have high routing scores. Our experimental results demonstrate that the Intra-GPU Rectification and the Fill-in Rectificati
    
[^112]: 模拟失调: 大型语言模型的安全对齐可能会适得其反！

    Emulated Disalignment: Safety Alignment for Large Language Models May Backfire!

    [https://arxiv.org/abs/2402.12343](https://arxiv.org/abs/2402.12343)

    安全对齐的大型语言模型可能会通过模拟失调框架，在对抗性操纵下产生危险结果，对训练的语言模型具有双倍有害性，高于强基线，强调了即使在安全对齐后也需要重新评估开源语言模型的重要性。

    

    大型语言模型（LLMs）需要进行安全对齐，以确保与人类进行安全的对话。然而，在这项工作中，我们引入了一种推理时攻击框架，表明安全对齐也可能在对抗性操纵下无意中促成有害结果。这个框架被命名为模拟失调（ED），在输出空间中不良地组合了一对开源预训练和安全对齐的语言模型，产生了一个有害的语言模型而无需任何训练。我们对ED在三个数据集和四个模型系列（Llama-1、Llama-2、Mistral和Alpaca）上的实验表明，ED使预训练模型的有害性增加了一倍，并胜过强基线，以较大优势在48个评估子集中的43个中实现了最高的有害率。至关重要的是，我们的研究结果凸显了即使在安全对齐后，重新评估开源语言模型实践的重要性。

    arXiv:2402.12343v1 Announce Type: new  Abstract: Large language models (LLMs) need to undergo safety alignment to ensure safe conversations with humans. However, in this work, we introduce an inference-time attack framework, demonstrating that safety alignment can also unintentionally facilitate harmful outcomes under adversarial manipulation. This framework, named Emulated Disalignment (ED), adversely combines a pair of open-source pre-trained and safety-aligned language models in the output space to produce a harmful language model without any training. Our experiments with ED across three datasets and four model families (Llama-1, Llama-2, Mistral, and Alpaca) show that ED doubles the harmfulness of pre-trained models and outperforms strong baselines, achieving the highest harmful rate in 43 out of 48 evaluation subsets by a large margin. Crucially, our findings highlight the importance of reevaluating the practice of open-sourcing language models even after safety alignment.
    
[^113]: SciAgent: 工具增强型语言模型用于科学推理

    SciAgent: Tool-augmented Language Models for Scientific Reasoning

    [https://arxiv.org/abs/2402.11451](https://arxiv.org/abs/2402.11451)

    引入了工具增强型科学推理的新任务设置，通过提供可扩展的工具集，帮助大型语言模型在科学问题解决中变得更加实用和可解决。

    

    科学推理对于即使最先进的大型语言模型（LLMs）来说也是一项巨大挑战。为了使LLMs更加实用和可解决此任务，我们引入了一种名为工具增强型科学推理的新任务设置。这种设置通过为LLMs提供可扩展的工具集，将重点从追求全知问题求解器转变为熟练使用工具的人。为了促进这种设置的研究，我们构建了一个名为MathFunc的工具增强型训练语料库，涵盖了超过30,000个样本和大约6,000个工具。基于MathFunc，我们开发了SciAgent，用于检索、理解，以及必要时使用工具进行科学问题解决。此外，我们构建了一个名为SciToolBench的基准，涵盖五个科学领域，以评估LLMs在工具辅助下的能力。对SciToolBench进行的大量实验验证了SciAgent的有效性。值得注意的是，SciAgent-Mistral-7B超过了其他LLMs。

    arXiv:2402.11451v1 Announce Type: cross  Abstract: Scientific reasoning poses an excessive challenge for even the most advanced Large Language Models (LLMs). To make this task more practical and solvable for LLMs, we introduce a new task setting named tool-augmented scientific reasoning. This setting supplements LLMs with scalable toolsets, and shifts the focus from pursuing an omniscient problem solver to a proficient tool-user. To facilitate the research of such setting, we construct a tool-augmented training corpus named MathFunc which encompasses over 30,000 samples and roughly 6,000 tools. Building on MathFunc, we develop SciAgent to retrieve, understand and, if necessary, use tools for scientific problem solving. Additionally, we craft a benchmark, SciToolBench, spanning five scientific domains to evaluate LLMs' abilities with tool assistance. Extensive experiments on SciToolBench confirm the effectiveness of SciAgent. Notably, SciAgent-Mistral-7B surpasses other LLMs with the sa
    
[^114]: 在比较之前进行推理：LLM增强的语义相似度度量用于领域专门文本分析

    Reasoning before Comparison: LLM-Enhanced Semantic Similarity Metrics for Domain Specialized Text Analysis

    [https://arxiv.org/abs/2402.11398](https://arxiv.org/abs/2402.11398)

    通过利用LLM增强语义分析，开发了用于文本的相似度度量框架，可显著改善文本的语义相似性评估，并可扩展到其他专业领域。

    

    在这项研究中，我们利用LLM来增强语义分析，为文本开发相似度度量，解决传统无监督NLP度量（如ROUGE和BLEU）的局限性。我们开发了一个框架，其中LLM（例如GPT-4）用于零样本文本识别和放射学报告的标签生成，在那里这些标签然后被用作文本相似性的度量。通过在MIMIC数据集上测试所提出的框架，我们发现GPT-4生成的标签能够显著改善语义相似度评估，得分更接近临床实际情况比传统NLP度量。我们的工作展示了使用LLM进行高度专业领域的文本数据的语义分析的可能性，具有半定量推理结果。虽然该框架针对放射学报告相似性分析进行了实施，但其概念可以扩展到其他专门领域。

    arXiv:2402.11398v1 Announce Type: cross  Abstract: In this study, we leverage LLM to enhance the semantic analysis and develop similarity metrics for texts, addressing the limitations of traditional unsupervised NLP metrics like ROUGE and BLEU. We develop a framework where LLMs such as GPT-4 are employed for zero-shot text identification and label generation for radiology reports, where the labels are then used as measurements for text similarity. By testing the proposed framework on the MIMIC data, we find that GPT-4 generated labels can significantly improve the semantic similarity assessment, with scores more closely aligned with clinical ground truth than traditional NLP metrics. Our work demonstrates the possibility of conducting semantic analysis of the text data using semi-quantitative reasoning results by the LLMs for highly specialized domains. While the framework is implemented for radiology report similarity analysis, its concept can be extended to other specialized domains 
    
[^115]: 通过反事实文本引导的对比语言-图像预训练来理解新闻缩略图的代表性

    Understanding News Thumbnail Representativeness by Counterfactual Text-Guided Contrastive Language-Image Pretraining

    [https://arxiv.org/abs/2402.11159](https://arxiv.org/abs/2402.11159)

    提出了一种反事实文本引导的对比语言-图像预训练框架CFT-CLIP，用于增强新闻文本和缩略图之间的对比学习。

    

    本文深入探讨了理解新闻缩略图的代表性这一关键挑战，这些缩略图通常在文章在社交媒体上传播时作为读者的第一个视觉参与。我们关注新闻图像是否代表新闻文本中讨论的主要主题。为了应对这一挑战，我们引入了一个手动注释的新闻缩略图和文本配对数据集\textsc{NewsTT}。我们发现，例如CLIP和BLIP-2这样的预训练视觉和语言模型在这一任务上表现不佳。由于新闻主题经常涉及命名实体或专有名词，预训练模型缺乏匹配其视觉和文本外观的能力。为了填补这一空白，我们提出了CFT-CLIP，一个反事实文本引导的对比语言-图像预训练框架。

    arXiv:2402.11159v1 Announce Type: new  Abstract: This paper delves into the critical challenge of understanding the representativeness of news thumbnail images, which often serve as the first visual engagement for readers when an article is disseminated on social media. We focus on whether a news image represents the main subject discussed in the news text. To serve the challenge, we introduce \textsc{NewsTT}, a manually annotated dataset of news thumbnail image and text pairs. We found that pretrained vision and language models, such as CLIP and BLIP-2, struggle with this task. Since news subjects frequently involve named entities or proper nouns, a pretrained model could not have the ability to match its visual and textual appearances. To fill the gap, we propose CFT-CLIP, a counterfactual text-guided contrastive language-image pretraining framework. We hypothesize that learning to contrast news text with its counterfactual, of which named entities are replaced, can enhance the cross
    
[^116]: 在一个 1000 万根草垛中寻找针：循环记忆找到了语言模型不擅长的内容

    In Search of Needles in a 10M Haystack: Recurrent Memory Finds What LLMs Miss

    [https://arxiv.org/abs/2402.10790](https://arxiv.org/abs/2402.10790)

    通过使用循环记忆增强对 GPT-2 进行微调，使其能够处理长达 1000 万个元素的任务，这是迄今为止处理最长输入的开放神经网络模型，并展示了对长序列处理能力的显著改进。

    

    本文解决了使用生成式 Transformer 模型处理长文档的挑战。为了评估不同方法，我们引入了 BABILong，这是一个新的基准，旨在评估模型在提取和处理广泛文本中分布式事实方面的能力。我们的评估包括 GPT-4 和 RAG 的基准，结果显示常见方法仅适用于最多 $10^4$ 个元素的序列。相反，通过使用循环记忆增强对 GPT-2 进行微调，使其能够处理涉及最多 $10^7$ 个元素的任务。这一成就标志着迄今为止任何开源神经网络模型处理的最长输入，显示了对长序列处理能力的显著改进。

    arXiv:2402.10790v1 Announce Type: cross  Abstract: This paper addresses the challenge of processing long documents using generative transformer models. To evaluate different approaches, we introduce BABILong, a new benchmark designed to assess model capabilities in extracting and processing distributed facts within extensive texts. Our evaluation, which includes benchmarks for GPT-4 and RAG, reveals that common methods are effective only for sequences up to $10^4$ elements. In contrast, fine-tuning GPT-2 with recurrent memory augmentations enables it to handle tasks involving up to $10^7$ elements. This achievement marks a substantial leap, as it is by far the longest input processed by any open neural network model to date, demonstrating a significant improvement in the processing capabilities for long sequences.
    
[^117]: 在InSaAF中融入安全性，通过准确性和公平性 | LLM是否已经准备好进入印度法律领域？

    InSaAF: Incorporating Safety through Accuracy and Fairness | Are LLMs ready for the Indian Legal Domain?

    [https://arxiv.org/abs/2402.10567](https://arxiv.org/abs/2402.10567)

    本研究在印度法律领域探讨了大型语言模型（LLMs）在处理社会因素时的能力，提出了结合公平性和准确性的新指标$LSS_{\beta}$，并评估了模型在二元法律推理任务中的表现以及在印度社会各种不平等方面的公平性展示。

    

    语言技术和人工智能的最新进展已经导致提出了众多语言模型，用于执行法律领域的各种任务，从预测判决到生成摘要。尽管它们具有巨大潜力，但已经证明这些模型学习并展示社会偏见，并做出不公平的预测。在这项研究中，我们探讨了当涉及社会因素时大型语言模型（LLMs）在印度法律领域执行任务的能力。我们提出了一种新颖的度量标准，$\beta$-加权的$\textit{法律安全分数($LSS_{\beta}$)}$，将LLM的公平性和准确性两个方面结合起来。我们通过考虑LLM在$\textit{二元法律推理}$任务中的表现以及其在印度社会各种不平等方面的公平展示来评估LLMs的安全性。LLaMA和LLaMA--2模型的任务表现和公平得分表明...

    arXiv:2402.10567v1 Announce Type: cross  Abstract: Recent advancements in language technology and Artificial Intelligence have resulted in numerous Language Models being proposed to perform various tasks in the legal domain ranging from predicting judgments to generating summaries. Despite their immense potential, these models have been proven to learn and exhibit societal biases and make unfair predictions. In this study, we explore the ability of Large Language Models (LLMs) to perform legal tasks in the Indian landscape when social factors are involved. We present a novel metric, $\beta$-weighted $\textit{Legal Safety Score ($LSS_{\beta}$)}$, which encapsulates both the fairness and accuracy aspects of the LLM. We assess LLMs' safety by considering its performance in the $\textit{Binary Statutory Reasoning}$ task and its fairness exhibition with respect to various axes of disparities in the Indian society. Task performance and fairness scores of LLaMA and LLaMA--2 models indicate th
    
[^118]: AI医院：用于临床诊断的LLMs作为实习医生的交互式评估和协作

    AI Hospital: Interactive Evaluation and Collaboration of LLMs as Intern Doctors for Clinical Diagnosis

    [https://arxiv.org/abs/2402.09742](https://arxiv.org/abs/2402.09742)

    AI医院是一个框架，用于构建实时交互式诊断环境，通过与LLMs的交互评估和协作，提高临床诊断的准确性。

    

    引入大型语言模型（LLMs）在医疗保健中的应用标志着重大的进展。然而，目前的应用主要局限于辨别和问答任务，没有充分发挥其交互潜力。为了解决这个局限，我们的论文提出了AI医院，一个旨在构建实时交互式诊断环境的框架。为了模拟过程，我们收集高质量的医疗记录，创建了患者、检查者和医疗主任代理。然后，利用AI医院进行LLMs的交互评估和协作。初始阶段，我们创建了一个多视图医学评估（MVME）基准，其中各种LLMs作为实习医生进行交互式诊断。随后，为了提高诊断准确性，我们引入了一种协作机制，涉及医疗主任的监督下的迭代讨论和争议解决过程。

    arXiv:2402.09742v1 Announce Type: new  Abstract: The incorporation of Large Language Models (LLMs) in healthcare marks a significant advancement. However, the application has predominantly been limited to discriminative and question-answering tasks, which does not fully leverage their interactive potential. To address this limitation, our paper presents AI Hospital, a framework designed to build a real-time interactive diagnosis environment. To simulate the procedure, we collect high-quality medical records to create patient, examiner, and medical director agents. AI Hospital is then utilized for the interactive evaluation and collaboration of LLMs. Initially, we create a Multi-View Medical Evaluation (MVME) benchmark where various LLMs serve as intern doctors for interactive diagnosis. Subsequently, to improve diagnostic accuracy, we introduce a collaborative mechanism that involves iterative discussions and a dispute resolution process under the supervision of the medical director. I
    
[^119]: 标点符号恢复在没有监督的情况下改善结构理解

    Punctuation Restoration Improves Structure Understanding without Supervision

    [https://arxiv.org/abs/2402.08382](https://arxiv.org/abs/2402.08382)

    标点符号恢复是一个有效的学习目标，可以改善结构理解并提高模型性能。

    

    无监督学习目标，如语言建模和去噪等，在生成预训练模型方面起着重要作用，这些预训练模型能够执行从自然语言理解到会话任务的各种下游应用。然而，尽管最近的大型语言模型具有令人印象深刻的对话能力，但它们在捕捉文本的句法或语义结构方面的能力仍然落后。我们假设，语言性能和机器能力之间的不匹配归因于当前流行的预训练目标未能充分传递语言结构知识给计算系统。我们展示了标点符号恢复对结构相关任务的内部和外部表现的改善，如命名实体识别、开放式信息提取、分块和词性标注。标点符号恢复是一个有效的学习目标，可以改善结构理解并产生更加鲁棒的模型。

    Unsupervised learning objectives like language modeling and de-noising constitute a significant part in producing pre-trained models that perform various downstream applications from natural language understanding to conversational tasks. However, despite impressive conversational capabilities of recent large language model, their abilities to capture syntactic or semantic structure within text lag behind. We hypothesize that the mismatch between linguistic performance and competence in machines is attributable to insufficient transfer of linguistic structure knowledge to computational systems with currently popular pre-training objectives. We show that punctuation restoration transfers to improvements in in- and out-of-distribution performance on structure-related tasks like named entity recognition, open information extraction, chunking, and part-of-speech tagging. Punctuation restoration is an effective learning objective that can improve structure understanding and yield a more rob
    
[^120]: 解决医学语言模型中的认知偏见问题

    Addressing cognitive bias in medical language models

    [https://arxiv.org/abs/2402.08113](https://arxiv.org/abs/2402.08113)

    本研究通过开发BiasMedQA，一个用于评估医学任务中LLMs的认知偏见的新型基准，发现LLMs在面对包含认知偏见的临床问题时，其回答的准确性明显降低。

    

    将大型语言模型（LLMs）整合到医学领域已经引起了重大关注，因为它们在模拟临床决策场景中的准确性很有前景。然而，临床决策比模拟更复杂，因为医生的决策受到许多因素的影响，包括认知偏见的存在。然而，LLMs在面对包含认知偏见的临床问题时，与不包含这些偏见的问题相比，其回答的准确性会明显降低，这一问题尚未被探索。本研究的假设认为，当LLMs面对包含认知偏见的临床问题时，与不包含这些偏见的问题相比，其回答的准确性会明显降低。我们开发了BiasMedQA，这是一个用于评估LLMs在医学任务中的认知偏见的新型基准。使用BiasMedQA，我们评估了六个LLMs，分别是GPT-4、Mixtral-8x70B、GPT-3.5、PaLM-2、Llama 2 70B-chat和医学专业的PMC Llama 13B。我们在127个临床问题上测试了这些模型。

    The integration of large language models (LLMs) into the medical field has gained significant attention due to their promising accuracy in simulated clinical decision-making settings. However, clinical decision-making is more complex than simulations because physicians' decisions are shaped by many factors, including the presence of cognitive bias. However, the degree to which LLMs are susceptible to the same cognitive biases that affect human clinicians remains unexplored. Our hypothesis posits that when LLMs are confronted with clinical questions containing cognitive biases, they will yield significantly less accurate responses compared to the same questions presented without such biases.In this study, we developed BiasMedQA, a novel benchmark for evaluating cognitive biases in LLMs applied to medical tasks. Using BiasMedQA we evaluated six LLMs, namely GPT-4, Mixtral-8x70B, GPT-3.5, PaLM-2, Llama 2 70B-chat, and the medically specialized PMC Llama 13B. We tested these models on 1,27
    
[^121]: 可扩展大型语言模型微调的差分隐私零阶方法

    Differentially Private Zeroth-Order Methods for Scalable Large Language Model Finetuning

    [https://arxiv.org/abs/2402.07818](https://arxiv.org/abs/2402.07818)

    本文研究了差分隐私零阶方法在大型语言模型微调中的应用，该方法通过使用零阶梯度来避免传统优化方法的可扩展性瓶颈，实现了在隐私、效用和可扩展性之间的良好平衡。

    

    在特定任务的数据集上进行微调是利用预训练语言模型的强大能力进行各种下游任务的广泛接受的范例。由于预训练语言模型微调的普及以及与之相关的隐私问题，差分隐私预训练语言模型微调引起了越来越多的关注，以保护特定任务数据集的隐私。差分隐私预训练语言模型微调方法的设计核心是在隐私、效用和可扩展性之间达到满意的权衡。大多数现有方法都是基于DP-SGD的创新性工作。尽管将DP-SGD的可扩展性推到了极限，但基于DP-SGD的微调方法不幸地受到了SGD固有低效率的限制。在本文中，我们研究了DP零阶方法在LLM预训练中的潜力，该方法通过用更高效的零阶梯度来近似梯度，避免了SGD的可扩展性瓶颈。与将零阶方法作为一种替代方法进行处理不同，我们引入了一种新的割接框架，该框架能够以非常接近的方式模拟DP-SGD的基本操作，然后利用零阶优化方法来近似梯度。

    Finetuning on task-specific datasets is a widely-embraced paradigm of harnessing the powerful capability of pretrained LLMs for various downstream tasks. Due to the popularity of LLMs finetuning and its accompanying privacy concerns, differentially private (DP) finetuning of pretrained LLMs has garnered increasing attention to safeguarding the privacy of task-specific datasets. Lying at the design core of DP LLM finetuning methods is the satisfactory tradeoff between privacy, utility, and scalability. Most existing methods build upon the seminal work of DP-SGD. Despite pushing the scalability of DP-SGD to its limit, DP-SGD-based finetuning methods are unfortunately limited by the inherent inefficiency of SGD. In this paper, we investigate the potential of DP zeroth-order methods for LLM pretraining, which avoids the scalability bottleneck of SGD by approximating the gradient with the more efficient zeroth-order gradient. Rather than treating the zeroth-order method as a drop-in replace
    
[^122]: DeAL：用于大型语言模型的解码时对齐

    DeAL: Decoding-time Alignment for Large Language Models

    [https://arxiv.org/abs/2402.06147](https://arxiv.org/abs/2402.06147)

    DeAL是一个允许用户自定义奖励函数并实现解码时对齐LLMs的框架。

    

    大型语言模型（LLMs）现在期望生成与人类偏好对齐的内容。目前的工作主要集中在模型训练时间对齐上，通过诸如强化学习与人类反馈（RLHF）等技术。然而，目前还不清楚这些方法是否有效地教导模型对齐目标。首先，无法整合多个自定义奖励和依赖模型开发者对通用和静态原则的理解是主要局限。其次，模型训练中的残留差距以及这些方法的可靠性也值得质疑（例如，即使在安全训练后仍然容易被越狱）。为了解决这些问题，我们提出了DeAL，一个允许用户自定义奖励函数并实现解码时对齐LLMs（DeAL）的框架。核心思想在于将解码视为一个启发式引导的搜索过程，并促使使用各种对齐目标。我们的实验以编程约束为例进行了验证。

    Large Language Models (LLMs) are nowadays expected to generate content aligned with human preferences. Current work focuses on alignment at model training time, through techniques such as Reinforcement Learning with Human Feedback (RLHF). However, it is unclear if such methods are an effective choice to teach alignment objectives to the model. First, the inability to incorporate multiple, custom rewards and reliance on a model developer's view of universal and static principles are key limitations. Second, the residual gaps in model training and the reliability of such approaches are also questionable (e.g. susceptibility to jail-breaking even after safety training). To address these, we propose DeAL, a framework that allows the user to customize reward functions and enables Decoding-time Alignment of LLMs (DeAL). At its core, we view decoding as a heuristic-guided search process and facilitate the use of a wide variety of alignment objectives. Our experiments with programmatic constra
    
[^123]: 利用类别概率进行黑盒子句级攻击

    Exploiting Class Probabilities for Black-box Sentence-level Attacks

    [https://arxiv.org/abs/2402.02695](https://arxiv.org/abs/2402.02695)

    该论文研究了在黑盒子句级攻击中利用类别概率的有效性，并开发了一种新的算法进行攻击。通过与基线方法进行对比，进行了广泛的评估。

    

    句级攻击是针对文本分类器的对抗性句子生成方法，这些句子与正确分类的句子同义，但被分类器错误地分类。在黑盒设置下，分类器只能通过对查询输入的反馈进行访问，这主要以类别概率的形式提供。尽管利用类别概率可以获得更强大的攻击效果，但由于在句级攻击中使用类别概率存在挑战，现有的攻击方法要么不使用反馈，要么仅使用类别标签。为了克服这些挑战，我们开发了一种新的算法，使用类别概率进行黑盒句级攻击，并研究了在攻击成功率上使用类别概率的有效性，并探讨了在黑盒句级攻击中使用类别概率是否值得或可行。我们在各种分类器和基准数据集上对提出的攻击方法进行了广泛评估，并与基线进行了对比。

    Sentence-level attacks craft adversarial sentences that are synonymous with correctly-classified sentences but are misclassified by the text classifiers. Under the black-box setting, classifiers are only accessible through their feedback to queried inputs, which is predominately available in the form of class probabilities. Even though utilizing class probabilities results in stronger attacks, due to the challenges of using them for sentence-level attacks, existing attacks use either no feedback or only the class labels. Overcoming the challenges, we develop a novel algorithm that uses class probabilities for black-box sentence-level attacks, investigate the effectiveness of using class probabilities on the attack's success, and examine the question if it is worthy or practical to use class probabilities by black-box sentence-level attacks. We conduct extensive evaluations of the proposed attack comparing with the baselines across various classifiers and benchmark datasets.
    
[^124]: 消除检索器和大型语言模型之间的偏好差距

    Bridging the Preference Gap between Retrievers and LLMs

    [https://arxiv.org/abs/2401.06954](https://arxiv.org/abs/2401.06954)

    本研究提出了一种新颖的桥梁机制，通过训练一个桥梁模型来优化检索器和LLM之间的连接，消除了在RAG中检索器和LLMs之间的偏好差距。

    

    大型语言模型（LLMs）已在各种任务中展示出卓越的结果，而检索增强生成（RAG）是通过定位相关信息并将其放入LLM的上下文窗口来提高性能的有效方式。然而，在RAG中，检索器和LLMs之间的关系仍未得到充分调查。大多数现有工作将检索器和LLM视为独立组件，并在检索人性化信息和组装LLM友好上下文之间留下了差距。在这项工作中，我们研究了一种新颖的桥梁机制。我们验证了检索器在RAG环境中的排名和选择假设，并提出了一个框架，通过将监督学习和强化学习链接在一起来训练一个桥梁模型，优化检索器和LLM之间的连接。实证结果证明了我们的方法在问答和...

    arXiv:2401.06954v2 Announce Type: replace  Abstract: Large Language Models (LLMs) have demonstrated superior results across a wide range of tasks, and Retrieval-augmented Generation (RAG) is an effective way to enhance the performance by locating relevant information and placing it into the context window of the LLM. However, the relationship between retrievers and LLMs in a RAG is still under-investigated. Most existing work treats the retriever and the LLM as independent components and leaves a gap between retrieving human-"friendly" information and assembling a LLM-"friendly" context. In this work, we examine a novel bridge mechanism. We validate the ranking and selection assumptions of retrievers in the context of RAG and propose a framework that chains together supervised and reinforcement learning to train a bridge model that optimizes the connection between the retriever and the LLM. Empirical results demonstrate the effectiveness of our method in both question-answering and per
    
[^125]: 通过情绪思维链增强大型语言模型的情绪生成能力

    Enhancing Emotional Generation Capability of Large Language Models via Emotional Chain-of-Thought

    [https://arxiv.org/abs/2401.06836](https://arxiv.org/abs/2401.06836)

    该研究提出了一种名为情感思维链（ECoT）的提示方法，通过与人类情感智慧准则对齐，增强大型语言模型在情感生成任务上的性能。

    

    大型语言模型（LLMs）在各种情绪识别任务中表现出色，引起了研究界对探索它们在情感智能中潜力的好奇心。然而，情感生成任务领域仍存在一些问题，包括人类偏好的对齐和情感生成评估。本文提出了一种名为情感思维链（ECoT）的即插即用提示方法，通过与人类情感智力指南对齐来增强LLMs在各种情感生成任务上的表现。为了评估ECoT的可靠性，我们提出了一种称为情感生成得分（EGS）的自动化基于模型的评估方法。EGS将戈尔曼的情绪智力理论作为人类专家共识，为情感生成任务的评估提供了新的视角。大量实验结果证明...

    arXiv:2401.06836v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) have shown remarkable performance in various emotion recognition tasks, thereby piquing the research community's curiosity for exploring their potential in emotional intelligence. However, several issues in the field of emotional generation tasks remain unresolved, including human preference alignment and emotional generation assessment. In this paper, we propose the Emotional Chain-of-Thought (ECoT), a plug-and-play prompting method that enhances the performance of LLMs on various emotional generation tasks by aligning with human emotional intelligence guidelines. To assess the reliability of ECoT, we propose an automated model-based evaluation method called Emotional Generation Score (EGS). EGS incorporates Goleman's Emotional Intelligence Theory as a consensus of human experts, providing a new perspective on the evaluation of emotional generation tasks. Extensive experimental results demonstrate 
    
[^126]: 临床文本的神经机器翻译：对多语言预训练语言模型和迁移学习的经验研究

    Neural Machine Translation of Clinical Text: An Empirical Investigation into Multilingual Pre-Trained Language Models and Transfer-Learning

    [https://arxiv.org/abs/2312.07250](https://arxiv.org/abs/2312.07250)

    在临床文本的神经机器翻译研究中，通过使用多语言预训练语言模型和迁移学习方法，在ClinSpEn-2022英西临床领域数据上取得了顶级性能，并发现小型预训练语言模型在临床领域微调中胜过其他超大型语言模型。

    

    我们通过检验使用基于深度学习的多语言神经网络模型，如基于Transformer结构的模型，在临床文本机器翻译方面进行了调查。此外，为了解决语言资源不平衡问题，我们还使用基于大规模多语言预训练语言模型（MMPLMs）的迁移学习方法进行了实验。在临床案例（CC）、临床术语（CT）和本体概念（OC）等三个子任务上的实验结果显示，我们的模型在ClinSpEn-2022英西临床领域数据共享任务中取得了顶级性能。此外，我们基于专家的人工评估显示，在临床领域微调中，小型预训练语言模型（PLM）明显胜过其他两个超大型语言模型，这一发现在该领域从未有过报道。最后，迁移学习方法表现出

    arXiv:2312.07250v2 Announce Type: replace-cross  Abstract: We conduct investigations on clinical text machine translation by examining multilingual neural network models using deep learning such as Transformer based structures. Furthermore, to address the language resource imbalance issue, we also carry out experiments using a transfer learning methodology based on massive multilingual pre-trained language models (MMPLMs). The experimental results on three subtasks including 1) clinical case (CC), 2) clinical terminology (CT), and 3) ontological concept (OC) show that our models achieved top-level performances in the ClinSpEn-2022 shared task on English-Spanish clinical domain data. Furthermore, our expert-based human evaluations demonstrate that the small-sized pre-trained language model (PLM) won over the other two extra-large language models by a large margin, in the clinical domain fine-tuning, which finding was never reported in the field. Finally, the transfer learning method wor
    
[^127]: 攻击树：自动破解黑盒大型语言模型

    Tree of Attacks: Jailbreaking Black-Box LLMs Automatically

    [https://arxiv.org/abs/2312.02119](https://arxiv.org/abs/2312.02119)

    提出了一种名为Tree of Attacks with Pruning (TAP)的自动化方法，用于生成只需要对目标大型语言模型进行黑盒访问的越狱方法，并通过思维树推理和修剪生成准确的越狱提示。

    

    大型语言模型(LLMs)展示了多功能性，但仍在生成有害、带偏见和有毒内容，这一点由人为设计的越狱行为的普遍存在得以证明。在这项工作中，我们提出了一种名为Tree of Attacks with Pruning (TAP)的自动化方法，用于生成越狱，仅需要对目标LLM进行黑盒访问。TAP利用LLM来通过思维树推理迭代地优化候选（攻击）提示，直到生成的提示之一越狱目标。关键在于，在将提示发送给目标之前，TAP对其进行评估并移除可能不会导致越狱的提示。使用思维树推理使TAP能够在大量提示的搜索空间中导航，而修剪则减少了发送给目标的总查询数量。在实证评估中，我们观察到TAP生成的提示越狱了超过80%的最先进LLMs（包括GPT4和GPT4-Turbo）。

    arXiv:2312.02119v2 Announce Type: replace-cross  Abstract: While Large Language Models (LLMs) display versatile functionality, they continue to generate harmful, biased, and toxic content, as demonstrated by the prevalence of human-designed jailbreaks. In this work, we present Tree of Attacks with Pruning (TAP), an automated method for generating jailbreaks that only requires black-box access to the target LLM. TAP utilizes an LLM to iteratively refine candidate (attack) prompts using tree-of-thought reasoning until one of the generated prompts jailbreaks the target. Crucially, before sending prompts to the target, TAP assesses them and prunes the ones unlikely to result in jailbreaks. Using tree-of-thought reasoning allows TAP to navigate a large search space of prompts and pruning reduces the total number of queries sent to the target. In empirical evaluations, we observe that TAP generates prompts that jailbreak state-of-the-art LLMs (including GPT4 and GPT4-Turbo) for more than 80%
    
[^128]: CAMRA：AMR注释的副驾驶

    CAMRA: Copilot for AMR Annotation

    [https://arxiv.org/abs/2311.10928](https://arxiv.org/abs/2311.10928)

    CAMRA是一个创新的基于web的工具，用于从自然语言文本构建抽象意义表示（AMR），通过整合编程语言的编码方法和AMR解析器模型作为副驾驶，极大提高了AMR注释的效率和准确性。

    

    在本文中，我们介绍了CAMRA（Copilot for AMR Annotations），这是一个最先进的基于web的工具，旨在从自然语言文本构建抽象意义表示（AMR）。CAMRA提供了一种创新的深层词汇语义注释方法，如AMR，将AMR注释视为编程语言中的编码。借助编程范式的熟悉度，CAMRA包含了所有现有AMR编辑器的基本功能，包括示例查找，同时通过将Propbank角色集查找集成为工具中的自动完成功能，更进一步。值得注意的是，CAMRA将AMR解析器模型作为编码副驾驶，极大地提高了AMR注释者的效率和准确性。为了展示工具的功能，我们提供了一个可访问的实时演示：https://camra.colorado.edu

    arXiv:2311.10928v2 Announce Type: replace-cross  Abstract: In this paper, we introduce CAMRA (Copilot for AMR Annotatations), a cutting-edge web-based tool designed for constructing Abstract Meaning Representation (AMR) from natural language text. CAMRA offers a novel approach to deep lexical semantics annotation such as AMR, treating AMR annotation akin to coding in programming languages. Leveraging the familiarity of programming paradigms, CAMRA encompasses all essential features of existing AMR editors, including example lookup, while going a step further by integrating Propbank roleset lookup as an autocomplete feature within the tool. Notably, CAMRA incorporates AMR parser models as coding co-pilots, greatly enhancing the efficiency and accuracy of AMR annotators. To demonstrate the tool's capabilities, we provide a live demo accessible at: https://camra.colorado.edu
    
[^129]: 您确定吗？挑战LLMs导致FlipFlop实验中的性能下降

    Are You Sure? Challenging LLMs Leads to Performance Drops in The FlipFlop Experiment

    [https://arxiv.org/abs/2311.08596](https://arxiv.org/abs/2311.08596)

    本研究通过FlipFlop实验揭示了当挑战LLMs让其反思初始答案时，模型会平均有46%的概率改变答案，所有模型在第一次和最终预测之间表现出准确性下降的现象。

    

    大型语言模型（LLMs）的交互性理论上允许模型完善和改进其答案，然而对LLMs的多轮行为的系统分析仍受限制。本文提出了FlipFlop实验：在对话的第一轮中，一个LLM完成一个分类任务。在第二轮中，LLM会受到一个追问，比如“您确定吗？”，为模型提供机会反思其初始答案，并决定是确认还是改变答案。对七个分类任务上的十个LLMs的系统研究显示，模型平均有46%的概率改变其答案，并且所有模型在第一次和最终预测之间看到准确性下降，平均下降了17%（FlipFlop效应）。我们对一个开源LLM进行微调实验，发现在合成数据上进行微调可以缓解--降低了性能。

    arXiv:2311.08596v2 Announce Type: replace  Abstract: The interactive nature of Large Language Models (LLMs) theoretically allows models to refine and improve their answers, yet systematic analysis of the multi-turn behavior of LLMs remains limited. In this paper, we propose the FlipFlop experiment: in the first round of the conversation, an LLM completes a classification task. In a second round, the LLM is challenged with a follow-up phrase like "Are you sure?", offering an opportunity for the model to reflect on its initial answer, and decide whether to confirm or flip its answer. A systematic study of ten LLMs on seven classification tasks reveals that models flip their answers on average 46% of the time and that all models see a deterioration of accuracy between their first and final prediction, with an average drop of 17% (the FlipFlop effect). We conduct finetuning experiments on an open-source LLM and find that finetuning on synthetically created data can mitigate - reducing perf
    
[^130]: 良好的开端是成功的一半：多步骤数学推理中开始正确的重要性

    Well begun is half done: Importance of Starting Right in Multi-Step Math Reasoning

    [https://arxiv.org/abs/2311.07945](https://arxiv.org/abs/2311.07945)

    较小的语言模型在多步骤数学推理中通过正确开始可以获得显着的性能提升，建议通过初始指导和自问指导的方式来引导模型开始正确。

    

    较小的语言模型通过学习为其预测生成原因，可以更好地解决复杂的推理任务。然而，我们观察到这些较小的模型有时会在开始时遇到困难，但在得到纠正后，可以解决原本困难的任务。我们提出了两种较小模型可以从初始指导中受益的方式：1）向LLM寻求初始指导，和2）自问指导，学生模型可以首先发起一个关于如何开始的问题，然后继续这一连锁。我们将初始基于问题的指导扩展到了一种称为QuestCoT的提示技术，该技术在进行推理链之前以一个问题开始是有益的。在两个多步数学推理数据集GSM8K和SVAMP上，我们展示了正确开始可以带来显著的性能提升（通过LLM指导最高高达+14分，通过QuestCoT最高高达+6分）。

    arXiv:2311.07945v2 Announce Type: replace  Abstract: Smaller language models can solve complex reasoning tasks better by learning to generate rationales for their predictions. However, we observe that these smaller models can sometimes struggle to start correctly, but when corrected, can solve a task that they would otherwise have struggled with. We propose two ways in which a smaller model can benefit from initial guidance: 1) asking an LLM for initial guidance, and 2) self-questioning guidance, where the student model can first initiate a question regarding how to start and then continue that chain. We extend initial question-based guidance to a prompting technique called QuestCoT, where starting with a question before a chain of reasoning proves useful. On two multi-step math reasoning datasets GSM8K and SVAMP, we show that starting correctly can lead to a significant performance gain (up to $+14$ points with LLM guidance and $+6$ points with QuestCoT).
    
[^131]: 咖啡：通过反馈修复错误来提升代码LLMs

    Coffee: Boost Your Code LLMs by Fixing Bugs with Feedback

    [https://arxiv.org/abs/2311.07215](https://arxiv.org/abs/2311.07215)

    开源代码LLMs难以生成正确指导的反馈，本研究提出了Coffee框架，旨在利用Coffee数据集构建CoffeePots，通过优化调整和选择，实现自动生成带有正确指导的反馈以用于代码修复。

    

    arXiv:2311.07215v2 公告类型：替换 摘要：代码编辑是确保程序综合的一个重要步骤，可以自动纠正代码LLMs生成的关键错误。最近的研究表明，闭源LLMs（如ChatGPT和GPT-4）能够生成纠正性反馈，用于编辑错误输入。然而，开源代码LLMs生成用于代码编辑的反馈仍然具有挑战性，因为这些模型倾向于遵循表面格式提供与误导信息相混淆的反馈。因此，我们的工作重点是利用开源代码LLMs生成具有正确指导的有用反馈用于代码编辑。为此引入了Coffee，一个专为带有反馈的代码修复而设计的数据集。利用该数据集，构建了CoffeePots，一个通过偏好优化调整和选择的COde Fixing with FEEdback框架。该框架旨在自动生成有用的反馈以帮助代码编辑。

    arXiv:2311.07215v2 Announce Type: replace  Abstract: Code editing is an essential step towards reliable program synthesis to automatically correct critical errors generated from code LLMs. Recent studies have demonstrated that closed-source LLMs (i.e., ChatGPT and GPT-4) are capable of generating corrective feedback to edit erroneous inputs. However, it remains challenging for open-source code LLMs to generate feedback for code editing, since these models tend to adhere to the superficial formats of feedback and provide feedback with misleading information. Hence, the focus of our work is to leverage open-source code LLMs to generate helpful feedback with correct guidance for code editing. To this end, we present Coffee, a collected dataset specifically designed for code fixing with feedback. Using this dataset, we construct CoffeePots, a framework for COde Fixing with FEEdback via Preference-Optimized Tuning and Selection. The proposed framework aims to automatically generate helpful 
    
[^132]: 揭示真相：欺骗语言和语言模型

    To Tell The Truth: Language of Deception and Language Models

    [https://arxiv.org/abs/2311.07092](https://arxiv.org/abs/2311.07092)

    在高风险环境中，研究人员通过分析电视游戏节目数据发现，即使只使用语言线索，基于大型语言模型构建的模型可以与人类主体具有类似的真相检测性能。

    

    arXiv:2311.07092v2 公告类型：替换-cross 摘要：基于文本的错误信息渗透到在线讨论中，然而人们能够从这种欺骗性文本内容中辨别真相的证据却很少。我们分析了一档新颖的电视游戏节目数据，其中高风险环境中相互之间存在冲突目标的个体之间的对话导致谎言。我们调查了欺骗语言潜在可验证语言线索在客观真相存在的情况下的表现，这是以往基于文本的欺骗数据集中缺少的一个显著特征。我们展示了存在一类探测器（算法），其真相检测性能与人类主体相似，即使前者只使用语言线索，而后者则通过完全访问所有潜在线索源（语言和视听）进行对话。我们的模型，建立在大型语言模型之上，采用瓶颈框架来学习可辨别的线索，以确定真相的行为

    arXiv:2311.07092v2 Announce Type: replace-cross  Abstract: Text-based misinformation permeates online discourses, yet evidence of people's ability to discern truth from such deceptive textual content is scarce. We analyze a novel TV game show data where conversations in a high-stake environment between individuals with conflicting objectives result in lies. We investigate the manifestation of potentially verifiable language cues of deception in the presence of objective truth, a distinguishing feature absent in previous text-based deception datasets. We show that there exists a class of detectors (algorithms) that have similar truth detection performance compared to human subjects, even when the former accesses only the language cues while the latter engages in conversations with complete access to all potential sources of cues (language and audio-visual). Our model, built on a large language model, employs a bottleneck framework to learn discernible cues to determine truth, an act of 
    
[^133]: 在自然语言推理中评估预训练语言模型的性别偏见，考虑所有标签

    Evaluating Gender Bias of Pre-trained Language Models in Natural Language Inference by Considering All Labels

    [https://arxiv.org/abs/2309.09697](https://arxiv.org/abs/2309.09697)

    提出了一种考虑自然语言推理任务三个标签的预训练语言模型偏见评估方法，通过创造代表不同类型偏见的评估数据组，并实验证明该方法能更好地区分有偏见的、不正确的推理和非有偏见的不正确推理。

    

    在多种语言的预训练语言模型（PLMs）中发现了歧视性的性别偏见。在自然语言推理（NLI）中，现有的偏见评估方法专注于三个标签中的一个特定标签的预测结果，例如中性。然而，这种评估方法可能不准确，因为独特的偏见推理与独特的预测标签相关联。为了解决这一限制，我们提出了一种考虑NLI任务的三个标签的PLMs偏见评估方法。我们创建了三个代表不同类型偏见的评估数据组。然后，我们基于每个数据组的相应标签输出定义了一种偏见度量。在实验中，我们引入了一种用于NLI偏见度量的元评估技术，并用它来确认我们的偏见度量可以更好地区分有偏见的，不正确的推理与非偏见的不正确推理，胜过基线，从而导致了m

    arXiv:2309.09697v2 Announce Type: replace  Abstract: Discriminatory gender biases have been found in Pre-trained Language Models (PLMs) for multiple languages. In Natural Language Inference (NLI), existing bias evaluation methods have focused on the prediction results of a specific label out of three labels, such as neutral. However, such evaluation methods can be inaccurate since unique biased inferences are associated with unique prediction labels. Addressing this limitation, we propose a bias evaluation method for PLMs that considers all the three labels of NLI task. We create three evaluation data groups that represent different types of biases. Then, we define a bias measure based on the corresponding label output of each data group. In the experiments, we introduce a meta-evaluation technique for NLI bias measures and use it to confirm that our bias measure can distinguish biased, incorrect inferences from non-biased incorrect inferences better than the baseline, resulting in a m
    
[^134]: 通过阅读理解调整大型语言模型

    Adapting Large Language Models via Reading Comprehension

    [https://arxiv.org/abs/2309.09530](https://arxiv.org/abs/2309.09530)

    通过将原始语料库转化为阅读理解文本来调整大型语言模型，使其在多个领域的各种任务中性能始终得到提升。

    

    我们探讨了在特定领域语料库上持续预训练对大型语言模型的影响，发现在原始语料库上进行训练赋予模型领域知识，但极大地损害了其回答问题的能力。受人类通过阅读理解学习的启发，即阅读后练习提高基于所学知识回答问题的能力，我们提出了一种将原始语料库转化为阅读理解文本的简单方法。每个原始文本都会被一系列与其内容相关的任务丰富。我们的方法非常可扩展，适用于任何预训练语料库，能够在三个不同领域（生物医学、金融和法律）的各种任务中持续提升性能。值得注意的是，我们的7B语言模型在竞争中表现出色，能与规模更大的领域特定模型（如BloombergGPT-50B）相媲美。此外，我们证明了领域特定模型可以带来更好的效果。

    arXiv:2309.09530v2 Announce Type: replace  Abstract: We explore how continued pre-training on domain-specific corpora influences large language models, revealing that training on the raw corpora endows the model with domain knowledge, but drastically hurts its prompting ability for question answering. Taken inspiration from human learning via reading comprehension--practice after reading improves the ability to answer questions based on the learned knowledge--we propose a simple method for transforming raw corpora into reading comprehension texts. Each raw text is enriched with a series of tasks related to its content. Our method, highly scalable and applicable to any pre-training corpora, consistently enhances performance across various tasks in three different domains: biomedicine, finance, and law. Notably, our 7B language model achieves competitive performance with domain-specific models of much larger scales, such as BloombergGPT-50B. Furthermore, we demonstrate that domain-specif
    
[^135]: 大型语言模型中公平性的调查

    A Survey on Fairness in Large Language Models

    [https://arxiv.org/abs/2308.10149](https://arxiv.org/abs/2308.10149)

    本文审查了关于大型语言模型中公平性的研究，针对中等规模LLMs和大规模LLMs提出了评估指标和去偏见方法。

    

    大型语言模型(LLMs)展现了强大的性能和发展前景，并广泛部署在现实世界中。然而，LLMs可能会捕捉到未经处理的训练数据中的社会偏见，并将这些偏见传播到下游任务。不公平的LLM系统会产生不良的社会影响和潜在危害。在本文中，我们全面回顾了有关LLMs中公平性的研究。考虑到参数大小和训练范式对研究策略的影响，我们将现有的公平性研究分为针对中等规模LLMs在预训练和微调范式下的研究以及针对大规模LLMs在提示范式下的研究。首先，对于中等规模LLMs，我们从内在偏见和外在偏见的角度介绍了评估指标和去偏见方法。然后，对于大规模LLMs，我们引入了最近的公平性研究，包括公平性评估、原因...

    arXiv:2308.10149v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) have shown powerful performance and development prospects and are widely deployed in the real world. However, LLMs can capture social biases from unprocessed training data and propagate the biases to downstream tasks. Unfair LLM systems have undesirable social impacts and potential harms. In this paper, we provide a comprehensive review of related research on fairness in LLMs. Considering the influence of parameter magnitude and training paradigm on research strategy, we divide existing fairness research into oriented to medium-sized LLMs under pre-training and fine-tuning paradigms and oriented to large-sized LLMs under prompting paradigms. First, for medium-sized LLMs, we introduce evaluation metrics and debiasing methods from the perspectives of intrinsic bias and extrinsic bias, respectively. Then, for large-sized LLMs, we introduce recent fairness research, including fairness evaluation, reason
    
[^136]: mCL-NER: 多视角对比学习实现的跨语言命名实体识别

    mCL-NER: Cross-Lingual Named Entity Recognition via Multi-view Contrastive Learning

    [https://arxiv.org/abs/2308.09073](https://arxiv.org/abs/2308.09073)

    本文提出了一种通过多视角对比学习实现跨语言命名实体识别的方法，通过识别令牌对关系并利用上下文细微差别来统一不同语言的表示。

    

    跨语言命名实体识别面临着由于跨语料库稀缺，尤其是对非英语数据表现不均的挑战。本文提出了一种通过多视角对比学习实现跨语言命名实体识别（mCL-NER）的方法。具体而言，我们将CrossNER任务重新构建为一种识别令牌对关系的问题，该方法利用实体内部令牌之间的上下文细微差别，使我们能够将不同语言的表示统一起来。引入了一种多视角对比学习框架，用于涵盖源语句、代码切换语句和目标语句之间的语义对比，以及令牌之间的对比。

    arXiv:2308.09073v2 Announce Type: replace  Abstract: Cross-lingual named entity recognition (CrossNER) faces challenges stemming from uneven performance due to the scarcity of multilingual corpora, especially for non-English data. While prior efforts mainly focus on data-driven transfer methods, a significant aspect that has not been fully explored is aligning both semantic and token-level representations across diverse languages. In this paper, we propose Multi-view Contrastive Learning for Cross-lingual Named Entity Recognition (mCL-NER). Specifically, we reframe the CrossNER task into a problem of recognizing relationships between pairs of tokens. This approach taps into the inherent contextual nuances of token-to-token connections within entities, allowing us to align representations across different languages. A multi-view contrastive learning framework is introduced to encompass semantic contrasts between source, codeswitched, and target sentences, as well as contrasts among toke
    
[^137]: TESS：文本到文本自条件单纯形扩散

    TESS: Text-to-Text Self-Conditioned Simplex Diffusion

    [https://arxiv.org/abs/2305.08379](https://arxiv.org/abs/2305.08379)

    TESS是一个全非自回归的文本扩散模型，通过在逻辑空间而不是学习嵌入空间应用扩散过程，进行了自条件单纯形扩散，实验证明在自然语言理解和生成任务中表现优于最先进的非自回归模型，并且所需的扩散步骤更少。

    

    扩散模型已经成为一种在各种连续领域中表现出色的生成方法范式。然而，将连续扩散模型应用于自然语言仍然具有挑战性，因为自然语言是离散的，并且需要大量的扩散步骤来生成文本，这使得基于扩散的生成变得昂贵。在这项工作中，我们提出了文本到文本自条件单纯形扩散（TESS），这是一个全非自回归的文本扩散模型，采用一种新形式的自条件，将扩散过程应用于逻辑空间而不是学习嵌入空间。通过对包括总结、文本简化、释义生成和问题生成在内的自然语言理解和生成任务的广泛实验，我们证明了TESS优于最先进的非自回归模型，在需要更少的扩散步骤的情况下表现出最小的性能下降。

    arXiv:2305.08379v2 Announce Type: replace  Abstract: Diffusion models have emerged as a powerful paradigm for generation, obtaining strong performance in various continuous domains. However, applying continuous diffusion models to natural language remains challenging due to its discrete nature and the need for a large number of diffusion steps to generate text, making diffusion-based generation expensive. In this work, we propose Text-to-text Self-conditioned Simplex Diffusion (TESS), a text diffusion model that is fully non-autoregressive, employs a new form of self-conditioning, and applies the diffusion process on the logit simplex space rather than the learned embedding space. Through extensive experiments on natural language understanding and generation tasks including summarization, text simplification, paraphrase generation, and question generation, we demonstrate that TESS outperforms state-of-the-art non-autoregressive models, requires fewer diffusion steps with minimal drop i
    
[^138]: InPars-Light:成本效益高的无监督训练高效排名器

    InPars-Light: Cost-Effective Unsupervised Training of Efficient Rankers

    [https://arxiv.org/abs/2301.02998](https://arxiv.org/abs/2301.02998)

    InPars-Light是一个简单而有效的修改，通过使用小得多的排名模型和免费语言模型BLOOM，在多个英文检索集合上显著改进了排名性能。

    

    我们开展了对InPars的可重现性研究，这是一种用于无监督训练神经排名器的方法。作为副产品，我们开发出了InPars-Light，这是对InPars的简单而有效的修改。与InPars不同，InPars-Light使用7-100倍更小的排名模型，并且只需要一个免费提供的语言模型BLOOM，我们发现，与专有的GPT-3模型相比，BLOOM能够产生更准确的排名器。在所有五个英文检索集合上，我们仅使用一个30M参数六层MiniLM-30M排名器和一个三选俩的提示，在nDCG和MRR方面，相比BM25，我们都获得了显著的（7%-30%）且具有统计学意义的改进。相反，在InPars的研究中，只有一个大100倍的monoT5-3B模型能够始终胜过BM25，而小得多的monoT5-220M模型（仍然比我们的MiniLM排名器大7倍）只是在MS MAR上胜过BM25。

    arXiv:2301.02998v2 Announce Type: replace-cross  Abstract: We carried out a reproducibility study of InPars, which is a method for unsupervised training of neural rankers (Bonifacio et al., 2022). As a by-product, we developed InPars-light, which is a simple-yet-effective modification of InPars. Unlike InPars, InPars-light uses 7x-100x smaller ranking models and only a freely available language model BLOOM, which -- as we found out -- produced more accurate rankers compared to a proprietary GPT-3 model. On all five English retrieval collections (used in the original InPars study) we obtained substantial (7%-30%) and statistically significant improvements over BM25 (in nDCG and MRR) using only a 30M parameter six-layer MiniLM-30M ranker and a single three-shot prompt. In contrast, in the InPars study only a 100x larger monoT5-3B model consistently outperformed BM25, whereas their smaller monoT5-220M model (which is still 7x larger than our MiniLM ranker) outperformed BM25 only on MS MAR
    
[^139]: 通过对多模态指导手册进行排序来理解多模态程序化知识

    Understanding Multimodal Procedural Knowledge by Sequencing Multimodal Instructional Manuals

    [https://arxiv.org/abs/2110.08486](https://arxiv.org/abs/2110.08486)

    本研究通过整理数据集并收集全面的人类注释，对机器学习模型在推理和排序无序的多模态指导方面的能力进行基准测试，发现模型表现不仅显著低于人类，而且似乎无法具备这种基本能力。

    

    漫游指导无序事件的能力是理解和推理现实世界任务程序的基本技能，通常需要对时间常识和多模态信息有深入的理解，因为这些程序通常通过文本和图像的组合进行传达。这种能力对于顺序任务规划和多源指导摘要等应用至关重要。虽然人类能够推理和排序无序的多模态程序指导，但当前机器学习模型是否具有这种基本能力仍然是一个未解之谜。在这项工作中，我们通过整理来自热门在线指导手册的数据集并收集了全面的人类注释，来对模型推理和排序无序的多模态指导进行基准测试。我们发现模型不仅性能显著低于人类，而且似乎无法...

    arXiv:2110.08486v4 Announce Type: replace  Abstract: The ability to sequence unordered events is an essential skill to comprehend and reason about real world task procedures, which often requires thorough understanding of temporal common sense and multimodal information, as these procedures are often communicated through a combination of texts and images. Such capability is essential for applications such as sequential task planning and multi-source instruction summarization. While humans are capable of reasoning about and sequencing unordered multimodal procedural instructions, whether current machine learning models have such essential capability is still an open question. In this work, we benchmark models' capability of reasoning over and sequencing unordered multimodal instructions by curating datasets from popular online instructional manuals and collecting comprehensive human annotations. We find models not only perform significantly worse than humans but also seem incapable of e
    
[^140]: CFMatch: 将自动答案等价评估与人工专家判断在开放域问答中对齐

    CFMatch: Aligning Automated Answer Equivalence Evaluation with Expert Judgments For Open-Domain Question Answering. (arXiv:2401.13170v1 [cs.CL])

    [http://arxiv.org/abs/2401.13170](http://arxiv.org/abs/2401.13170)

    CFMatch提出了一个在开放域问答中将自动答案等价评估与人工专家判断对齐的方法，通过提供明确一致的评估指南并引入高效、稳健且轻量级的判别式AE分类器匹配方法来解决当前评估指标与人类判断不一致的问题。

    

    问答系统只有在我们知道答案是否正确的情况下才能取得进展，但对于许多最具挑战和有趣的问答示例，当前用于确定答案等价性的评估指标通常与人类判断不一致，尤其是来自大型语言模型（LLM）的更冗长、自由形式的答案。存在两个挑战：缺乏数据和模型过大：基于LLM的评分器可以更好地与人工评判员相关联，但这个任务只在有限的问答数据集上进行了测试，即使可用，对模型的更新也有限，因为LLM过大且往往昂贵。我们通过提供明确一致的指南来解决这两个问题，这些指南用于从专业人工问答比赛中采纳机器问答在答案等价性评估方面的标准。我们还引入了一种标准评估和一种更高效、稳健且轻量级的判别式AE分类器匹配方法（CFMatch，大小小于1MB），经过训练和验证以更准确地评估答案等价性。

    Question answering (QA) can only make progress if we know if an answer is correct, but for many of the most challenging and interesting QA examples, current evaluation metrics to determine answer equivalence (AE) often do not align with human judgments, particularly more verbose, free-form answers from large language models (LLM). There are two challenges: a lack of data and that models are too big: LLM-based scorers can correlate better with human judges, but this task has only been tested on limited QA datasets, and even when available, update of the model is limited because LLMs are large and often expensive. We rectify both of these issues by providing clear and consistent guidelines for evaluating AE in machine QA adopted from professional human QA contests. We also introduce a combination of standard evaluation and a more efficient, robust, and lightweight discriminate AE classifier-based matching method (CFMatch, smaller than 1 MB), trained and validated to more accurately evalu
    
[^141]: OOP：针对大型语言模型的面向对象编程评估基准

    OOP: Object-Oriented Programming Evaluation Benchmark for Large Language Models. (arXiv:2401.06628v1 [cs.CL])

    [http://arxiv.org/abs/2401.06628](http://arxiv.org/abs/2401.06628)

    本研究提出了一种面向对象编程的新型评估基准，包括431个Python程序，采用pass@o度量指标来提供更全面和相关的OOP代码生成评估。评估结果显示代码专用LLMs在OOP方面表现较差，需进一步改进此领域。

    

    推进自动化编程需要健壮且全面的代码生成评估基准，然而当前的评估框架在功能式编程方面（例如HumanEval和MBPP）很大程度上忽视了面向对象编程（OOP）。为了解决这个问题，我们的研究引入了一项创新的面向对象编程重点基准，包括431个Python程序，涵盖了类和封装方法等基本的OOP概念和特性。我们提出了一种新颖的评估度量指标pass@o，针对OOP进行了改进，增强传统的pass@k度量。我们评估了23个领先的大型语言模型（LLMs），包括通用模型和专门用于代码的模型，得出了三个关键发现：1）pass@o提供了更相关和全面的OOP代码生成评估；2）尽管在FP方面表现出色，像WizardCoder这样的代码专用LLMs在OOP方面落后于像ChatGPT这样的模型；3）所有先进的LLMs在我们的OOP基准上表现不佳，突显了改进此领域的迫切需求。

    Advancing automated programming necessitates robust and comprehensive code generation benchmarks, yet current evaluation frameworks largely neglect object-oriented programming (OOP) in favor of functional programming (FP), e.g., HumanEval and MBPP. To address this, our study introduces a pioneering OOP-focused benchmark, featuring 431 Python programs that encompass essential OOP concepts and features like classes and encapsulation methods. We propose a novel evaluation metric, pass@o, tailored for OOP, enhancing traditional pass@k measures. Our evaluation of 23 leading large language models (LLMs), including both general and code-specialized models, reveals three key insights: 1) pass@o offers a more relevant and comprehensive assessment for OOP code generation; 2) Despite excelling in FP, code-specialized LLMs like WizardCoder lag in OOP compared to models like ChatGPT; 3) The poor performance of all advanced LLMs on our OOP benchmark highlights a critical need for improvements in thi
    
[^142]: Intention Analysis Prompting使得大型语言模型成为良好的越狱防御者

    Intention Analysis Prompting Makes Large Language Models A Good Jailbreak Defender. (arXiv:2401.06561v1 [cs.CL])

    [http://arxiv.org/abs/2401.06561](http://arxiv.org/abs/2401.06561)

    本研究提出了一种名为Intention Analysis Prompting (IAPrompt)的方法，通过触发大型语言模型（LLMs）的自我纠正和改进能力来防御越狱攻击。实验证明，该方法能够显著减少响应中的有害行为并保持整体有用性。

    

    在面对隐蔽和复杂的越狱攻击时，将大型语言模型(LLMs)与人类价值观保持一致是一项极具挑战性的任务。在本研究中，我们提出了一种简单但非常有效的防御策略，即Intention Analysis Prompting（IAPrompt）。其原理是通过两个阶段的过程触发LLMs的内在自我纠正和改进能力：1）基本意图分析，2）与政策一致的响应。值得注意的是，IAPrompt是一种仅推断的方法，因此可以提高LLMs的安全性而不损害其有用性。在Vicuna、ChatGLM、MPT、DeepSeek和GPT-3.5上进行的广泛实验表明，IAPrompt能够持续且显著地减少响应中的有害行为（平均攻击成功率下降46.5%），同时保持整体有用性。进一步的分析揭示了我们方法的一些见解。为了保证可重复性，我们在https://github.com/alph上发布了我们的代码和脚本。

    Aligning large language models (LLMs) with human values, particularly in the face of stealthy and complex jailbreaks, presents a formidable challenge. In this study, we present a simple yet highly effective defense strategy, i.e., Intention Analysis Prompting (IAPrompt). The principle behind is to trigger LLMs' inherent self-correct and improve ability through a two-stage process: 1) essential intention analysis, and 2) policy-aligned response. Notably, IAPrompt is an inference-only method, thus could enhance the safety of LLMs without compromising their helpfulness. Extensive experiments on SAP200 and DAN benchmarks across Vicuna, ChatGLM, MPT, DeepSeek, and GPT-3.5 show that IAPrompt could consistently and significantly reduce the harmfulness in response (averagely -46.5% attack success rate) and maintain the general helpfulness. Further analyses present some insights into how our method works. To facilitate reproducibility, We release our code and scripts at: https://github.com/alph
    
[^143]: LEGOBench：科学模型排行榜生成基准测试

    LEGOBench: Leaderboard Generation Benchmark for Scientific Models. (arXiv:2401.06233v1 [cs.CL])

    [http://arxiv.org/abs/2401.06233](http://arxiv.org/abs/2401.06233)

    LEGOBench是一个评估生成科学模型排行榜系统的基准测试，使用22年来的论文预印本数据和PapersWithCode门户上的机器学习排行榜的数据，初步结果显示自动排行榜生成存在显著性能差距。

    

    随着论文提交数量的不断增加，难以及时了解最新的最先进研究成果成为了一个难题。为了解决这个挑战，我们引入了LEGOBench，这是一个评估生成排行榜系统的基准测试。LEGOBench由22年来在arXiv上提交的预印本数据和PapersWithCode门户上的11,000多个机器学习排行榜组成。我们评估了四种传统的基于图的排名变体和三种最近提出的大型语言模型的性能。我们的初步结果显示自动排行榜生成存在显著的性能差距。代码可在https://github.com/lingo-iitgn/LEGOBench获取，数据集托管在https://osf.io/9v2py/?view_only=6f91b0b510df498ba01595f8f278f94c。

    The ever-increasing volume of paper submissions makes it difficult to stay informed about the latest state-of-the-art research. To address this challenge, we introduce LEGOBench, a benchmark for evaluating systems that generate leaderboards. LEGOBench is curated from 22 years of preprint submission data in arXiv and more than 11,000 machine learning leaderboards in the PapersWithCode portal. We evaluate the performance of four traditional graph-based ranking variants and three recently proposed large language models. Our preliminary results show significant performance gaps in automatic leaderboard generation. The code is available on https://github.com/lingo-iitgn/LEGOBench and the dataset is hosted on https://osf.io/9v2py/?view_only=6f91b0b510df498ba01595f8f278f94c .
    
[^144]: SH2: 自我突出式犹豫帮助您更准确解码。

    SH2: Self-Highlighted Hesitation Helps You Decode More Truthfully. (arXiv:2401.05930v1 [cs.CL])

    [http://arxiv.org/abs/2401.05930](http://arxiv.org/abs/2401.05930)

    自我突出式犹豫（SH2）是一种推理时的方法，通过选择预测概率较低的标记，并强调它们的差异，从而帮助语言模型更准确地解码。

    

    大型语言模型(LLMs)在文本生成方面表现出色。然而，LLMs仍然存在幻觉问题。在本研究中，我们提出了一种推理时方法，即自我突出式犹豫(SH2)，以帮助LLMs更准确地解码。SH2基于信息理论中一个简单的事实，即对于LLMs而言，预测概率较低的标记往往更具信息量。我们的分析表明，LLMs给予较低概率的标记更有可能与事实信息（如名词、专有名词和形容词）密切相关。因此，我们提出通过选择概率最低的标记并将其连接到原始上下文中来“突出”事实信息，从而迫使模型在生成之前多次阅读和犹豫这些标记。在解码过程中，我们还采用对比解码的方式来强调由犹豫带来的输出概率的差异。

    Large language models (LLMs) demonstrate great performance in text generation. However, LLMs are still suffering from hallucinations. In this work, we propose an inference-time method, Self-Highlighted Hesitation (SH2), to help LLMs decode more truthfully. SH2 is based on a simple fact rooted in information theory that for an LLM, the tokens predicted with lower probabilities are prone to be more informative than others. Our analysis shows that the tokens assigned with lower probabilities by an LLM are more likely to be closely related to factual information, such as nouns, proper nouns, and adjectives. Therefore, we propose to ''highlight'' the factual information by selecting the tokens with the lowest probabilities and concatenating them to the original context, thus forcing the model to repeatedly read and hesitate on these tokens before generation. During decoding, we also adopt contrastive decoding to emphasize the difference in the output probabilities brought by the hesitation.
    
[^145]: ANGO: 一个面向生成型语言模型的中文领域评估基准

    ANGO: A Next-Level Evaluation Benchmark For Generation-Oriented Language Models In Chinese Domain. (arXiv:2401.04898v1 [cs.CL])

    [http://arxiv.org/abs/2401.04898](http://arxiv.org/abs/2401.04898)

    ANGO是一个中文领域生成型语言模型评估基准，引入了关键点分类标准，提供了更好的可解释性，同时建立了可量化的问题难度标准，对模型训练提供了更精确的指导。

    

    最近，出现了各种大规模语言模型（LLMs）评估数据集，但其中大多数存在排名失真和模型能力分析困难的问题。针对这些问题，本文引入了ANGO，一个中文多项选择题评估基准。ANGO首次提出了“关键点”分类标准，ANGO中的每个问题可以对应多个关键点，有效提高了评估结果的可解释性。基于真人表现的性能，我们建立了可量化的问题难度标准，并将ANGO问题分为9个难度级别，为模型训练提供了更精确的指导。为了最小化数据泄漏的影响并充分利用ANGO的创新特点，我们设计了独家抽样策略和新的评估框架，支持快速测试集迭代。我们的实验证明，ANGO对模型提出了更大的挑战，并在评估结果中揭示出更多细节。

    Recently, various Large Language Models (LLMs) evaluation datasets have emerged, but most of them have issues with distorted rankings and difficulty in model capabilities analysis. Addressing these concerns, this paper introduces ANGO, a Chinese multi-choice question evaluation benchmark. ANGO proposes \textit{Keypoint} categorization standard for the first time, each question in ANGO can correspond to multiple keypoints, effectively enhancing interpretability of evaluation results. Base on performance of real humans, we build a quantifiable question difficulty standard and divide ANGO questions into 9 difficulty levels, which provide more precise guidance for model training. To minimize data leakage impact and fully leverage ANGO's innovative features, we have engineered exclusive sampling strategies and a new evaluation framework that support swift testset iteration. Our experiments demonstrate that ANGO poses a stronger challenge to models and reveals more details in evaluation resu
    
[^146]: 在文本分类中探索语言模型中的概念级别的误相关性

    Explore Spurious Correlations at the Concept Level in Language Models for Text Classification. (arXiv:2311.08648v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2311.08648](http://arxiv.org/abs/2311.08648)

    本文研究了语言模型在文本分类中概念级别的误相关性问题，并通过使用ChatGPT分配概念标签和引入数据再平衡技术来解决这一问题。

    

    语言模型在众多自然语言处理任务中取得了显著的成功，采用了微调和上下文学习方法。虽然语言模型表现出卓越的性能，但由于训练数据中标签分布不平衡或上下文学习实例产生的误相关性，它们面临着鲁棒性挑战。以往的研究主要集中在词语、短语和句法特征上，忽视了概念级别的研究，这往往是由于缺乏概念标签和难以确定输入文本中的概念内容。本文提出了两个主要贡献。首先，我们使用ChatGPT为文本分配概念标签，评估模型在微调或上下文学习测试数据中的概念偏差。我们发现，当语言模型在训练或提示中遇到概念和标签之间的误相关性时，会采取预测的捷径。其次，我们引入了一种数据再平衡技术，将ChatGPT生成的反事实数据纳入其中。

    Language models (LMs) have achieved notable success in numerous NLP tasks, employing both fine-tuning and in-context learning (ICL) methods. While language models demonstrate exceptional performance, they face robustness challenges due to spurious correlations arising from imbalanced label distributions in training data or ICL exemplars. Previous research has primarily concentrated on word, phrase, and syntax features, neglecting the concept level, often due to the absence of concept labels and difficulty in identifying conceptual content in input texts. This paper introduces two main contributions. First, we employ ChatGPT to assign concept labels to texts, assessing concept bias in models during fine-tuning or ICL on test data. We find that LMs, when encountering spurious correlations between a concept and a label in training or prompts, resort to shortcuts for predictions. Second, we introduce a data rebalancing technique that incorporates ChatGPT-generated counterfactual data, ther
    
[^147]: 持续学习在语言转换中的研究

    A Study of Continual Learning Under Language Shift. (arXiv:2311.01200v1 [cs.CL])

    [http://arxiv.org/abs/2311.01200](http://arxiv.org/abs/2311.01200)

    本文研究了持续学习在语言转换中的应用，发现在更新语言模型时，前向转移效果较好且与语言顺序无关，但后向转移效果可能取决于新语言的顺序和特征。

    

    最近语言模型预训练的数据和模型规模的增加导致了巨大的训练成本。在随时间推移而出现新数据的情况下，更新模型而不是完全重新训练可以带来显著的收益。在本文中，我们研究了在新语言出现时更新语言模型时的好处和弊端，即在语言转换中持续学习的情况。从单语英语语言模型出发，我们逐步添加了来自挪威语和冰岛语的数据，以研究前向和后向转移效果如何取决于预训练顺序和语言特征，对于不同的模型大小和学习率调度器。我们的结果表明，尽管前向转移主要是正向的，不受语言顺序的影响，但后向转移则可能是正向的或负向的，具体取决于新语言的顺序和特征。为了解释这些模式，我们探索了几种语言相似度度量方法。

    The recent increase in data and model scale for language model pre-training has led to huge training costs. In scenarios where new data become available over time, updating a model instead of fully retraining it would therefore provide significant gains. In this paper, we study the benefits and downsides of updating a language model when new data comes from new languages - the case of continual learning under language shift. Starting from a monolingual English language model, we incrementally add data from Norwegian and Icelandic to investigate how forward and backward transfer effects depend on the pre-training order and characteristics of languages, for different model sizes and learning rate schedulers. Our results show that, while forward transfer is largely positive and independent of language order, backward transfer can be either positive or negative depending on the order and characteristics of new languages. To explain these patterns we explore several language similarity metr
    
[^148]: 减少生成式语言模型学习困难的信息熵损失

    InfoEntropy Loss to Mitigate Bias of Learning Difficulties for Generative Language Models. (arXiv:2310.19531v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.19531](http://arxiv.org/abs/2310.19531)

    提出了一种信息熵损失函数，用于减少生成式语言模型对常见和易学标记的偏好，使其更关注不常见和难学的标记。

    

    生成式语言模型通常通过预测上一个标记（子词/词/短语）给出的下一个标记来进行预训练。最近的研究展示了大规模生成式语言模型在下游任务上的出色性能。然而，现有的生成式语言模型在训练过程中通常忽视文本语料库中的固有挑战，即频繁标记和不经常出现的标记之间的不平衡。这可能导致语言模型被常见且易学的标记所主导，从而忽视不经常出现且难以学习的标记。为了缓解这个问题，我们提出了一种信息熵损失（InfoEntropy Loss）函数。在训练过程中，它可以根据相应的预测概率分布的信息熵动态评估待学习标记的学习难度。然后，它适应地调整训练损失，试图使模型更加关注难以学习的标记。

    Generative language models are usually pretrained on large text corpus via predicting the next token (i.e., sub-word/word/phrase) given the previous ones. Recent works have demonstrated the impressive performance of large generative language models on downstream tasks. However, existing generative language models generally neglect an inherent challenge in text corpus during training, i.e., the imbalance between frequent tokens and infrequent ones. It can lead a language model to be dominated by common and easy-to-learn tokens, thereby overlooking the infrequent and difficult-to-learn ones. To alleviate that, we propose an Information Entropy Loss (InfoEntropy Loss) function. During training, it can dynamically assess the learning difficulty of a to-be-learned token, according to the information entropy of the corresponding predicted probability distribution over the vocabulary. Then it scales the training loss adaptively, trying to lead the model to focus more on the difficult-to-learn
    
[^149]: 通过估计数据分布比例的离散扩散语言建模

    Discrete Diffusion Language Modeling by Estimating the Ratios of the Data Distribution. (arXiv:2310.16834v1 [stat.ML])

    [http://arxiv.org/abs/2310.16834](http://arxiv.org/abs/2310.16834)

    本研究通过引入得分熵这一新颖的离散得分匹配损失，弥补了离散数据领域中现有方法的不足，提出了得分熵离散扩散模型(SEDD)并在GPT-2实验中取得了有竞争力的效果。

    

    尽管扩散模型在许多生成建模任务中具有突破性的性能，但在自然语言等离散数据领域中却表现不佳。关键是，标准的扩散模型依赖于成熟的得分匹配理论，但是将其推广到离散结构并没有取得相同的经验收益。在本文中，我们通过提出得分熵，一种新颖的离散得分匹配损失，来弥补这个差距，它比现有方法更稳定，可以形成最大似然训练的ELBO，并且可以通过去噪变体高效优化。我们将我们的得分熵离散扩散模型（SEDD）扩展到GPT-2的实验设置中，实现了极具竞争力的似然度，同时引入了独特的算法优势。特别是，在比较大小相似的SEDD和GPT-2模型时，SEDD达到了可比较的困惑度（通常在基线的+$10\%$内，并且有时超过基线）。此外，SEDD模型学到了...

    Despite their groundbreaking performance for many generative modeling tasks, diffusion models have fallen short on discrete data domains such as natural language. Crucially, standard diffusion models rely on the well-established theory of score matching, but efforts to generalize this to discrete structures have not yielded the same empirical gains. In this work, we bridge this gap by proposing score entropy, a novel discrete score matching loss that is more stable than existing methods, forms an ELBO for maximum likelihood training, and can be efficiently optimized with a denoising variant. We scale our Score Entropy Discrete Diffusion models (SEDD) to the experimental setting of GPT-2, achieving highly competitive likelihoods while also introducing distinct algorithmic advantages. In particular, when comparing similarly sized SEDD and GPT-2 models, SEDD attains comparable perplexities (normally within $+10\%$ of and sometimes outperforming the baseline). Furthermore, SEDD models lear
    
[^150]: QLLM: 大规模语言模型的准确高效低位宽量化

    QLLM: Accurate and Efficient Low-Bitwidth Quantization for Large Language Models. (arXiv:2310.08041v1 [cs.CL])

    [http://arxiv.org/abs/2310.08041](http://arxiv.org/abs/2310.08041)

    QLLM是一种为大规模语言模型设计的准确高效的低位宽后训练量化方法，通过引入自适应通道重组技术，将离群值的大小重新分配给其他通道，从而减轻它们对量化范围的影响。

    

    大规模语言模型在自然语言处理领域表现出色，但由于其所需资源过大，限制了其广泛应用。虽然量化感知训练（Quantization-Aware Training，QAT）提供了一种解决方案，但它的训练成本过高，因此后训练量化（Post-Training Quantization，PTQ）成为大规模语言模型更实际的方法。在现有研究中，特定通道中的激活离群值被认为是导致后训练量化准确性下降的瓶颈。本文提出了QLLM，一种为大规模语言模型设计的准确高效的低位宽后训练量化方法。QLLM引入了一种自适应通道重组技术，将离群值的大小重新分配给其他通道，从而减轻它们对量化范围的影响。具体来说，通过通道拆分和通道组装，在保证低位宽的情况下将离群通道分解成多个子通道。

    Large Language Models (LLMs) excel in NLP, but their demands hinder their widespread deployment. While Quantization-Aware Training (QAT) offers a solution, its extensive training costs make Post-Training Quantization (PTQ) a more practical approach for LLMs. In existing studies, activation outliers in particular channels are identified as the bottleneck to PTQ accuracy. They propose to transform the magnitudes from activations to weights, which however offers limited alleviation or suffers from unstable gradients, resulting in a severe performance drop at low-bitwidth. In this paper, we propose QLLM, an accurate and efficient low-bitwidth PTQ method designed for LLMs. QLLM introduces an adaptive channel reassembly technique that reallocates the magnitude of outliers to other channels, thereby mitigating their impact on the quantization range. This is achieved by channel disassembly and channel assembly, which first breaks down the outlier channels into several sub-channels to ensure a 
    
[^151]: GoLLIE:注释指南提高了零样本信息抽取

    GoLLIE: Annotation Guidelines improve Zero-Shot Information-Extraction. (arXiv:2310.03668v1 [cs.CL])

    [http://arxiv.org/abs/2310.03668](http://arxiv.org/abs/2310.03668)

    GoLLIE 是一个遵循注释指南的大型语言模型，通过微调以改进未见信息抽取任务的零样本结果。

    

    大型语言模型 (LLMs) 结合指导调优已经在泛化到未见任务方面取得了显著进展。然而，在信息抽取 (IE) 方面，它们的表现较差，落后于任务特定模型。通常，IE 任务的特点是复杂的注释指南，描述任务并给出示例给人类。先前利用这样的信息的尝试都失败了，即使使用最大的模型，它们也不能直接遵循指南。在本文中，我们提出了针对信息抽取的指南遵循大型语言模型 GoLLIE (Guideline-following Large Language Model for IE)，该模型通过微调以遵守注释指南，从而能够改进未见 IE 任务的零样本结果。全面的评估实证表明，GoLLIE 能够泛化并遵循未见指南，在零样本信息抽取方面优于先前的尝试。消融研究表明，详细的指南是取得良好结果的关键。

    Large Language Models (LLMs) combined with instruction tuning have made significant progress when generalizing to unseen tasks. However, they have been less successful in Information Extraction (IE), lagging behind task-specific models. Typically, IE tasks are characterized by complex annotation guidelines which describe the task and give examples to humans. Previous attempts to leverage such information have failed, even with the largest models, as they are not able to follow the guidelines out-of-the-box. In this paper we propose GoLLIE (Guideline-following Large Language Model for IE), a model able to improve zero-shot results on unseen IE tasks by virtue of being fine-tuned to comply with annotation guidelines. Comprehensive evaluation empirically demonstrates that GoLLIE is able to generalize to and follow unseen guidelines, outperforming previous attempts at zero-shot information extraction. The ablation study shows that detailed guidelines is key for good results.
    
[^152]: 缩放定律在联想记忆中的应用

    Scaling Laws for Associative Memories. (arXiv:2310.02984v1 [stat.ML])

    [http://arxiv.org/abs/2310.02984](http://arxiv.org/abs/2310.02984)

    本文研究了应用于联想记忆中的缩放定律，通过高维矩阵和嵌入的外积来模拟内层Transformer语言模型。作者推导出了与样本数量和参数大小相关的精确缩放定律，并验证了理论结果的有效性。同时，作者还通过大量实验展示了存储记忆关联的细粒度可视化。

    

    学习很可能涉及到抽象规则的发现和记忆。本文旨在研究联想记忆机制。我们的模型基于高维矩阵，由嵌入的外积组成，与Transformer语言模型的内层相关。我们推导出关于样本数量和参数规模的精确缩放定律，并讨论了不同估计器的统计效率，包括基于优化的算法。我们进行了大量的数值实验，以验证和解释理论结果，包括对存储记忆关联的细粒度可视化。

    Learning arguably involves the discovery and memorization of abstract rules. The aim of this paper is to study associative memory mechanisms. Our model is based on high-dimensional matrices consisting of outer products of embeddings, which relates to the inner layers of transformer language models. We derive precise scaling laws with respect to sample size and parameter size, and discuss the statistical efficiency of different estimators, including optimization-based algorithms. We provide extensive numerical experiments to validate and interpret theoretical results, including fine-grained visualizations of the stored memory associations.
    
[^153]: 实体推断竞技场：探究LLMs的对话推理和规划能力的平台

    The Entity-Deduction Arena: A playground for probing the conversational reasoning and planning capabilities of LLMs. (arXiv:2310.01468v1 [cs.CL])

    [http://arxiv.org/abs/2310.01468](http://arxiv.org/abs/2310.01468)

    本文提供了一个评估框架，通过向法官提出一系列查询来评估LLMs的对话推理和规划能力。我们发现不同的LLMs在这个任务上表现出显著差异。

    

    目前，大型语言模型（LLMs）在回答明确提问时非常有效。然而，当面临含糊不清的查询时，它们可能行为难以预测并产生错误的输出。这凸显了需要开发能够提出澄清问题以有效解决歧义的智能代理的需求。这种能力需要对多个对话轮次进行复杂的理解、状态跟踪、推理和规划。然而，直接测量这种能力可能具有挑战性。在本文中，我们提供了一个替代性问题，通过向法官提出一系列查询，评估了LLMs推断自己不知道但被法官揭示的实体的能力。这个“实体推断游戏”可以作为一个评估框架，用于探究语言模型的对话推理和规划能力。我们系统地评估了各种LLMs，并发现在这个任务上它们的性能存在显著差异。我们发现强大的LLMs...

    Large language models (LLMs) are currently effective at answering questions that are clearly asked. However, when faced with ambiguous queries they can act unpredictably and produce incorrect outputs. This underscores the need for the development of intelligent agents capable of asking clarification questions to resolve ambiguities effectively. This capability requires complex understanding, state tracking, reasoning and planning over multiple conversational turns. However, directly measuring this can be challenging. In this paper, we offer a surrogate problem which assesses an LLMs's capability to deduce an entity unknown to itself, but revealed to a judge, by asking the judge a series of queries. This \textit{entity-deducing game} can serve as an evaluation framework to probe the conversational reasoning and planning capabilities of language models. We systematically evaluate various LLMs and discover significant differences in their performance on this task. We find that strong LLMs
    
[^154]: 理解上下文学习中的重复现象

    Understanding In-Context Learning from Repetitions. (arXiv:2310.00297v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.00297](http://arxiv.org/abs/2310.00297)

    本论文通过研究表面重复现象的角度来探索大型语言模型中上下文学习的机制，并实证了一种增强标记关系的原则，为理解上下文学习及其潜在局限性做出了重要贡献。

    

    本论文探索了大型语言模型（LLM）中上下文学习的难以捉摸的机制。我们通过研究表面重复现象的角度来检视上下文学习，并定量地研究了表面特征在文本生成中的作用，同时实证了一种被称为“标记共现强化”的原则，该原则通过增强两个标记之间的关系来基于它们的上下文共现。通过研究这些特征的双重影响，我们的研究阐明了上下文学习的内在机制，并对其失败的原因进行了解释。本论文对于理解上下文学习及其潜在局限性做出了重要贡献，为这一激动人心的能力提供了新的视角。

    This paper explores the elusive mechanism underpinning in-context learning in Large Language Models (LLMs). Our work provides a novel perspective by examining in-context learning via the lens of surface repetitions. We quantitatively investigate the role of surface features in text generation, and empirically establish the existence of \emph{token co-occurrence reinforcement}, a principle that strengthens the relationship between two tokens based on their contextual co-occurrences. By investigating the dual impacts of these features, our research illuminates the internal workings of in-context learning and expounds on the reasons for its failures. This paper provides an essential contribution to the understanding of in-context learning and its potential limitations, providing a fresh perspective on this exciting capability.
    
[^155]: ToRA：一种集成工具的数学问题求解推理代理

    ToRA: A Tool-Integrated Reasoning Agent for Mathematical Problem Solving. (arXiv:2309.17452v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2309.17452](http://arxiv.org/abs/2309.17452)

    ToRA是一种集成工具的数学问题求解推理代理，通过结合语言的分析能力和工具的计算效率，能够显著提高数学推理的性能，在多个数学推理数据集上取得了13%-19%的平均绝对改进率，并在竞赛级数据集MATH上达到了44.6%的性能。

    

    大型语言模型在各种语言任务中取得了重大进展，但在复杂的数学问题上仍然存在困难。在本文中，我们提出了一系列集成工具的推理代理ToRA，它通过无缝地将自然语言推理与外部工具（例如计算库和符号求解器）的利用相结合，从而将语言的分析能力与工具的计算效率融合在一起，用于解决具有挑战性的数学问题。为了训练ToRA，我们精选了数学数据集上的互动工具使用轨迹，应用模仿学习于注释，并提出输出空间整形来进一步改进模型的推理行为。结果显示，ToRA模型在10个涵盖各种规模的数学推理数据集上显著优于开源模型，平均绝对改进率达到13%至19%。值得注意的是，ToRA-7B 在竞赛级数据集MATH上达到了44.6%，超越了最佳开源模型WizardMath。

    Large language models have made significant progress in various language tasks, yet they still struggle with complex mathematics. In this paper, we propose ToRA a series of Tool-integrated Reasoning Agents designed to solve challenging mathematical problems by seamlessly integrating natural language reasoning with the utilization of external tools (e.g., computation libraries and symbolic solvers), thereby amalgamating the analytical prowess of language and the computational efficiency of tools. To train ToRA, we curate interactive tool-use trajectories on mathematical datasets, apply imitation learning on the annotations, and propose output space shaping to further refine models' reasoning behavior. As a result, ToRA models significantly outperform open-source models on 10 mathematical reasoning datasets across all scales with 13%-19% absolute improvements on average. Notably, ToRA-7B reaches 44.6% on the competition-level dataset MATH, surpassing the best open-source model WizardMath
    
[^156]: 提升大规模语言模型在编码中的能力通过多角度自一致性

    Enhancing Large Language Models in Coding Through Multi-Perspective Self-Consistency. (arXiv:2309.17272v1 [cs.CL])

    [http://arxiv.org/abs/2309.17272](http://arxiv.org/abs/2309.17272)

    本文提出了一个名为多角度自一致性（MPSC）的框架，用于提升大规模语言模型在复杂的代码生成任务中的性能。该框架通过从多个角度采样多个输出并构建一个多部分图，利用交叉一致性和内一致性信息来选择最优输出。

    

    大规模语言模型（LLMs）在文本生成方面展现了卓越的能力。然而，在复杂的推理任务，如代码生成中，LLMs仍然难以在一次尝试中生成正确的答案。先前的研究通过聚合多个输出，利用它们之间的一致性来探索解决方案。然而，这些研究没有全面地从不同的角度捕捉这种一致性。在本文中，我们提出了一种名为多角度自一致性（MPSC）框架的新的解码策略，用于LLM，它将来自多个角度的输出之间的交叉一致性和单个角度内的内一致性结合起来。具体而言，我们要求LLMs对给定查询从各个角度采样多个多样化的输出，并基于它们构建一个多部分图。通过两个预定义的一致性度量，我们将交叉一致性和内一致性信息嵌入到图中。最佳选择是根据这些一致性度量来选择输出。

    Large language models (LLMs) have exhibited remarkable ability in textual generation. However, in complex reasoning tasks such as code generation, generating the correct answer in a single attempt remains a formidable challenge for LLMs. Previous research has explored solutions by aggregating multiple outputs, leveraging the consistency among them. However, none of them have comprehensively captured this consistency from different perspectives. In this paper, we propose the Multi-Perspective Self-Consistency (MPSC) framework, a novel decoding strategy for LLM that incorporates both inter-consistency across outputs from multiple perspectives and intra-consistency within a single perspective. Specifically, we ask LLMs to sample multiple diverse outputs from various perspectives for a given query and then construct a multipartite graph based on them. With two predefined measures of consistency, we embed both inter- and intra-consistency information into the graph. The optimal choice is th
    
[^157]: EchoPrompt：指导模型重新表述查询以改善上下文学习

    EchoPrompt: Instructing the Model to Rephrase Queries for Improved In-context Learning. (arXiv:2309.10687v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2309.10687](http://arxiv.org/abs/2309.10687)

    EchoPrompt是一种简单而有效的方法，通过促使模型重新表述查询来提供改进的上下文学习效果。实验证明，EchoPrompt在多个任务中都取得了显著的性能提升。

    

    通过积极采用推断时提示技术，如零-shot和少-shot提示技术，语言模型在各种任务上取得了令人印象深刻的性能。在这项工作中，我们介绍了一种称为EchoPrompt的简单而有效的方法，该方法提示模型在回答问题之前重新表述查询。EchoPrompt适用于标准和思维链提示的零-shot和少-shot上下文学习。实验结果表明，EchoPrompt在这四个因果语言模型族群的所有设置中都取得了显著改进。这些改进观察到了各种数值推理（例如，GSM8K，SVAMP）、阅读理解（例如DROP）和逻辑推理（例如Coin Flipping）任务中。平均而言，EchoPrompt提高了数值任务中code-davinci-002的零-shot-CoT性能5%，阅读理解任务中提高了13%。我们通过消融研究研究了影响EchoPrompt有效性的因素，其中包括...

    Language models are achieving impressive performance on various tasks by aggressively adopting inference-time prompting techniques, such as zero-shot and few-shot prompting. In this work, we introduce EchoPrompt, a simple yet effective approach that prompts the model to rephrase its queries before answering them. EchoPrompt is adapted for both zero-shot and few-shot in-context learning with standard and chain-of-thought prompting. Experimental results show that EchoPrompt yields substantial improvements across all these settings for four families of causal language models. These improvements are observed across various numerical reasoning (e.g. GSM8K, SVAMP), reading comprehension (e.g. DROP), and logical reasoning (e.g. Coin Flipping) tasks. On average, EchoPrompt improves the Zero-shot-CoT performance of code-davinci-002 by 5% in numerical tasks and 13% in reading comprehension tasks. We investigate the factors contributing to EchoPrompt's effectiveness through ablation studies, whic
    
[^158]: PoSE: 通过位置跳跃式训练提高LLMs对于上下文窗口的有效拓展

    PoSE: Efficient Context Window Extension of LLMs via Positional Skip-wise Training. (arXiv:2309.10400v1 [cs.CL])

    [http://arxiv.org/abs/2309.10400](http://arxiv.org/abs/2309.10400)

    本文介绍了一种名为PoSE的训练方法，通过在训练过程中使用固定的上下文窗口和操纵位置索引来适应极长的上下文窗口，实验证明这种方法大大减小了内存和时间开销，对性能影响较小，成功将LLaMA模型扩展到了128k个标记。

    

    本文介绍了一种名为Positional Skip-wise (PoSE)训练的方法，用于将大型语言模型（LLMs）适应于极长的上下文窗口。PoSE通过在训练过程中使用固定的上下文窗口和操纵位置索引来模拟长输入，将训练长度与目标上下文窗口大小分离。具体而言，我们从长输入序列中选择若干短块，并引入不同的跳跃偏置项来修改每个块的位置索引。这些跳跃偏置项以及每个块的长度在每个训练样本中都会变化，使得模型能够适应目标上下文窗口中的所有位置，而无需对完整长度的输入进行训练。实验证明，与对完整长度进行微调相比，PoSE大大减小了内存和时间开销，对性能影响较小。利用这一优势，我们成功将LLaMA模型扩展到了128k个标记。此外，我们经验证实，PoSE与

    In this paper, we introduce Positional Skip-wisE (PoSE) training for efficient adaptation of large language models~(LLMs) to extremely long context windows. PoSE decouples train length from target context window size by simulating long inputs using a fixed context window with manipulated position indices during training. Concretely, we select several short chunks from a long input sequence, and introduce distinct skipping bias terms to modify the position indices of each chunk. These bias terms, along with the length of each chunk, are altered for each training example, allowing the model to adapt to all positions within the target context window without training on full length inputs. Experiments show that, compared with fine-tuning on the full length, PoSE greatly reduces memory and time overhead with minimal impact on performance. Leveraging this advantage, we have successfully extended the LLaMA model to 128k tokens. Furthermore, we empirically confirm that PoSE is compatible with 
    
[^159]: LASER：具有状态空间探索的LLM代理用于Web导航

    LASER: LLM Agent with State-Space Exploration for Web Navigation. (arXiv:2309.08172v1 [cs.CL])

    [http://arxiv.org/abs/2309.08172](http://arxiv.org/abs/2309.08172)

    本论文提出了一种基于状态空间探索的LLM代理（LASER）用于Web导航任务。该代理以灵活的方式转换状态，通过执行动作完成任务，能够轻松从错误中恢复，并取得了显著的性能提升。

    

    大型语言模型（LLM）已成功应用于诸如Web导航之类的交互式决策任务。尽管取得了不错的性能，但先前的方法隐含地假设模型只能以正向方式执行，在交互环境中仅提供正例轨迹作为上下文示例，教授模型如何进行推理。因此，模型无法处理更具挑战性的情况，例如错误，从而导致次优性能。为了解决这个问题，我们提出将交互任务建模为状态空间探索，其中LLM代理通过执行动作在预定义的一组状态之间进行转换以完成任务。这种形式化方法使得模型可以灵活地进行回溯，从而能够轻松从错误中恢复。我们在WebShop任务上评估我们提出的LASER代理。实验结果表明，我们的LASER代理明显优于以前的方法。

    Large language models (LLMs) have been successfully adapted for interactive decision-making tasks like web navigation. While achieving decent performance, previous methods implicitly assume a forward-only execution mode for the model, where they only provide oracle trajectories as in-context examples to teach the model how to reason in the interactive environment. Consequently, the model could not handle more challenging scenarios not covered in the in-context examples, e.g., mistakes, leading to sub-optimal performance. To address this issue, we propose to model the interactive task as state space exploration, where the LLM agent transitions among a pre-defined set of states by performing actions to complete the task. This formulation enables flexible back-tracking, allowing the model to easily recover from errors. We evaluate our proposed LLM Agent with State-Space ExploRation (LASER) on the WebShop task. Experimental results show that our LASER agent significantly outperforms previo
    
[^160]: 引文文本跨度用于引文文本生成

    Cited Text Spans for Citation Text Generation. (arXiv:2309.06365v1 [cs.CL])

    [http://arxiv.org/abs/2309.06365](http://arxiv.org/abs/2309.06365)

    本文提出了一种弥合引用和引文文本之间距离的方法，通过使用引文文本跨度(CTS)替代摘要作为输入，从而使得引文生成更加准确和相关。通过自动标注和基于关键词的检索方法，可以实现高效的CTS标注，提高引文文本生成的效果。

    

    为了避免非事实性幻觉，自动相关工作生成必须将其输出与引文中的内容相关联，但由于科学文档的长度，现有的概括性方法只有在引文摘要的基础上进行。我们证明摘要并不总是引文生成的最佳输入，以及以这种方式训练的模型会出现幻觉。我们提出使用引文文本跨度(CTS)作为摘要的替代条件。由于手动CTS注释非常耗时且需要大量人力，我们尝试使用基于ROUGE的自动标注候选CTS句子，并取得足够强的性能以替代昂贵的人工注释，并提出了一种基于关键词的CTS检索方法，使得以引文全文为基础生成引文文本变得有前景和实际可行。

    Automatic related work generation must ground their outputs to the content of the cited papers to avoid non-factual hallucinations, but due to the length of scientific documents, existing abstractive approaches have conditioned only on the cited paper \textit{abstracts}. We demonstrate that the abstract is not always the most appropriate input for citation generation and that models trained in this way learn to hallucinate. We propose to condition instead on the \textit{cited text span} (CTS) as an alternative to the abstract. Because manual CTS annotation is extremely time- and labor-intensive, we experiment with automatic, ROUGE-based labeling of candidate CTS sentences, achieving sufficiently strong performance to substitute for expensive human annotations, and we propose a human-in-the-loop, keyword-based CTS retrieval approach that makes generating citation texts grounded in the full text of cited papers both promising and practical.
    
[^161]: SpikeBERT：一种采用两阶段BERT知识蒸馏训练的语言Spikformer

    SpikeBERT: A Language Spikformer Trained with Two-Stage Knowledge Distillation from BERT. (arXiv:2308.15122v1 [cs.CL])

    [http://arxiv.org/abs/2308.15122](http://arxiv.org/abs/2308.15122)

    该论文提出了一种名为SpikeBERT的SNN模型，通过改进Spikformer架构和使用两阶段知识蒸馏方法，该模型在语言任务上超越了其他SNN模型，在文本分类任务上甚至达到了与BERT相当的结果。

    

    脉冲神经网络（SNN）在以更节能的方式实现深度神经网络方面提供了一个有前景的途径。然而，现有的用于语言任务的SNN网络架构过于简单，深度架构尚未得到充分探索，与BERT等主流基于Transformer的网络相比，存在显著的性能差距。为此，我们改进了最近提出的脉冲Transformer（即Spikformer），使其能够处理语言任务，并提出了一种两阶段知识蒸馏方法来训练它，该方法结合了通过从BERT和大量未标记文本中蒸馏知识进行预训练，并通过再次从在相同训练示例上对BERT进行微调，并进行任务特定实例知识蒸馏。通过大量实验，我们展示了使用我们的方法训练的模型，命名为SpikeBERT，在实现上超过了最先进的SNN，并且甚至在文本分类任务上达到了与BERT相当的结果。

    Spiking neural networks (SNNs) offer a promising avenue to implement deep neural networks in a more energy-efficient way. However, the network architectures of existing SNNs for language tasks are too simplistic, and deep architectures have not been fully explored, resulting in a significant performance gap compared to mainstream transformer-based networks such as BERT. To this end, we improve a recently-proposed spiking transformer (i.e., Spikformer) to make it possible to process language tasks and propose a two-stage knowledge distillation method for training it, which combines pre-training by distilling knowledge from BERT with a large collection of unlabelled texts and fine-tuning with task-specific instances via knowledge distillation again from the BERT fine-tuned on the same training examples. Through extensive experimentation, we show that the models trained with our method, named SpikeBERT, outperform state-of-the-art SNNs and even achieve comparable results to BERTs on text 
    
[^162]: 授权临床医生并民主化数据科学：大型语言模型自动化临床研究的机器学习。 (arXiv:2308.14120v2 [cs.LG] 更新版)

    Empowering Clinicians and Democratizing Data Science: Large Language Models Automate Machine Learning for Clinical Studies. (arXiv:2308.14120v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2308.14120](http://arxiv.org/abs/2308.14120)

    chatGPT ADA是一种能够自主开发临床研究所需的最先进的机器学习模型的大型语言模型，可将高级分析工具民主化，使非数据科学家的临床医生能够轻松应用于医学领域。

    

    机器学习（ML）开发者（如数据科学家）和从业者（如临床医生）之间存在知识差距，阻碍了ML在临床数据分析中的充分利用。我们研究了chatGPT Advanced Data Analysis（ADA），即GPT-4的扩展，来弥合这一差距并高效执行ML分析的潜力。我们向chatGPT ADA提供了各种医学专业的大型试验的真实临床数据和研究详细信息，没有给出具体指导。ChatGPT ADA基于原始研究的训练数据自主开发了最先进的ML模型，用于预测临床结果，如癌症发展、癌症进展、疾病并发症或致病基因序列等生物标志物。令人惊讶的是，这些ML模型与其已发表的对应物相匹配甚至表现更好。我们得出结论，chatGPT ADA为民主化医学中的ML提供了一个有前景的途径，使非ML专家能够获得先进的分析工具并推动广泛应用。

    A knowledge gap persists between Machine Learning (ML) developers (e.g., data scientists) and practitioners (e.g., clinicians), hampering the full utilization of ML for clinical data analysis. We investigated the potential of the chatGPT Advanced Data Analysis (ADA), an extension of GPT-4, to bridge this gap and perform ML analyses efficiently. Real-world clinical datasets and study details from large trials across various medical specialties were presented to chatGPT ADA without specific guidance. ChatGPT ADA autonomously developed state-of-the-art ML models based on the original study's training data to predict clinical outcomes such as cancer development, cancer progression, disease complications, or biomarkers such as pathogenic gene sequences. Strikingly, these ML models matched or outperformed their published counterparts. We conclude that chatGPT ADA offers a promising avenue to democratize ML in medicine, making advanced analytics accessible to non-ML experts and promoting broa
    
[^163]: CausalLM不适用于上下文学习

    CausalLM is not optimal for in-context learning. (arXiv:2308.06912v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2308.06912](http://arxiv.org/abs/2308.06912)

    最近的研究显示，上下文学习中使用前缀语言模型（PrefixLM）比因果语言模型（CausalLM）效果更好。本文通过理论分析证明，虽然两种语言模型都以线性速率收敛到稳定点，但前缀语言模型收敛到线性回归的最优解，因果语言模型的收敛动态遵循在线梯度下降算法，不保证收敛到最优解。

    

    最近的实证证据表明，在上下文学习中，使用前缀语言模型（PrefixLM）表现更好，其允许上下文样本相互关注；相比之下，因果语言模型（CausalLM）使用自回归注意力机制，禁止上下文样本关注未来的样本。虽然这个结果是直观的，但从理论角度并不清楚。本文采用理论方法，分析了在特定参数构建下，前缀语言模型和因果语言模型的收敛行为。分析结果显示，两种语言模型都以线性速率收敛到稳定点，但前缀语言模型收敛到线性回归的最优解，而因果语言模型的收敛动态遵循在线梯度下降算法，即使样本数量趋于无穷，也不能保证收敛到最优解。我们通过对合成数据的经验实验来支持我们的理论观点。

    Recent empirical evidence indicates that transformer based in-context learning performs better when using a prefix language model (prefixLM), in which in-context samples can all attend to each other, compared to causal language models (causalLM), which use auto-regressive attention that prohibits in-context samples to attend to future samples. While this result is intuitive, it is not understood from a theoretical perspective. In this paper we take a theoretical approach and analyze the convergence behavior of prefixLM and causalLM under a certain parameter construction. Our analysis shows that both LM types converge to their stationary points at a linear rate, but that while prefixLM converges to the optimal solution of linear regression, causalLM convergence dynamics follows that of an online gradient descent algorithm, which is not guaranteed to be optimal even as the number of samples grows infinitely. We supplement our theoretical claims with empirical experiments over synthetic a
    
[^164]: 通过提示工程优化机器翻译：ChatGPT可定制性的研究

    Optimizing Machine Translation through Prompt Engineering: An Investigation into ChatGPT's Customizability. (arXiv:2308.01391v1 [cs.CL])

    [http://arxiv.org/abs/2308.01391](http://arxiv.org/abs/2308.01391)

    本文研究了通过在ChatGPT中运用合适的提示将翻译目的和目标受众融入进去对翻译质量的影响。研究发现，这种方法可以产生灵活的翻译结果，相比传统机器翻译更具定制性。

    

    本文探讨将翻译目的和目标受众融入到ChatGPT提示中对翻译质量的影响。研究借鉴了之前的翻译研究、行业实践和ISO标准，强调了翻译过程中预生产阶段的重要性。研究发现，在像ChatGPT这样的大规模语言模型中使用适当的提示可以产生灵活的翻译，这是传统的机器翻译所没有实现的。研究审查了在生成满足特定条件的翻译时，提示对翻译质量的影响。评估从实际翻译师的角度进行，主观和定性相结合，还使用了OpenAI的词嵌入API进行余弦相似度计算。研究结果表明，将翻译目的和目标受众融入到提示中确实可以修改生成的翻译。

    This paper explores the influence of integrating the purpose of the translation and the target audience into prompts on the quality of translations produced by ChatGPT. Drawing on previous translation studies, industry practices, and ISO standards, the research underscores the significance of the pre-production phase in the translation process. The study reveals that the inclusion of suitable prompts in large-scale language models like ChatGPT can yield flexible translations, a feat yet to be realized by conventional Machine Translation (MT). The research scrutinizes the changes in translation quality when prompts are used to generate translations that meet specific conditions. The evaluation is conducted from a practicing translator's viewpoint, both subjectively and qualitatively, supplemented by the use of OpenAI's word embedding API for cosine similarity calculations. The findings suggest that the integration of the purpose and target audience into prompts can indeed modify the gen
    
[^165]: 巨型语言模型时代的AutoML：当前挑战，未来机遇和风险。

    AutoML in the Age of Large Language Models: Current Challenges, Future Opportunities and Risks. (arXiv:2306.08107v1 [cs.LG])

    [http://arxiv.org/abs/2306.08107](http://arxiv.org/abs/2306.08107)

    论文探讨了AutoML和LLMs之间的共生关系，并指出这两个领域的融合有望颠覆NLP和AutoML两个领域，同时也存在风险。

    

    在过去的几年中，自然语言处理（NLP）和自动化机器学习（AutoML）领域取得了显著的成果。特别是在NLP领域，巨型语言模型（LLMs）最近经历了一系列突破。我们设想，两个领域通过紧密的融合可以彼此推动极限。为了展示这一愿景，我们探索了AutoML和LLMs之间的共生关系潜力，着重探讨了它们如何互相受益。我们特别研究了从不同角度增强LLMs的AutoML方法的机会以及利用AutoML进一步改进LLMs的挑战。为此，我们调查了现有工作，并对其中的风险进行了批判性评估。我们坚信，两个领域的融合有可能颠覆NLP和AutoML两个领域。通过强调可想象的协同作用和风险，我们旨在促进在交叉点的进一步探索。

    The fields of both Natural Language Processing (NLP) and Automated Machine Learning (AutoML) have achieved remarkable results over the past years. In NLP, especially Large Language Models (LLMs) have experienced a rapid series of breakthroughs very recently. We envision that the two fields can radically push the boundaries of each other through tight integration. To showcase this vision, we explore the potential of a symbiotic relationship between AutoML and LLMs, shedding light on how they can benefit each other. In particular, we investigate both the opportunities to enhance AutoML approaches with LLMs from different perspectives and the challenges of leveraging AutoML to further improve LLMs. To this end, we survey existing work, and we critically assess risks. We strongly believe that the integration of the two fields has the potential to disrupt both fields, NLP and AutoML. By highlighting conceivable synergies, but also risks, we aim to foster further exploration at the intersect
    
[^166]: STAR: 利用大型语言模型通过结构到文本数据生成改进低资源信息抽取

    STAR: Improving Low-Resource Information Extraction by Structure-to-Text Data Generation with Large Language Models. (arXiv:2305.15090v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.15090](http://arxiv.org/abs/2305.15090)

    STAR是一种利用大型语言模型合成数据实例的数据生成方法，用于改进低资源信息抽取，为实际应用提供了需要最少人工标注的解决方案。

    

    信息抽取任务，如事件抽取，需要对输出结构和子任务依赖进行深入理解。为了获得合理的性能，它们严重依赖于以（段落，目标结构）对的形式的任务特定训练数据。然而，通过人工注释获得这样的数据是昂贵的，因此对于实际应用，我们迫切需要需要最少人工标注的低资源信息抽取方法。使用合成训练数据对监督模型进行微调可能是一种通用方法，但现有的数据生成方法要么仍然依赖于大规模的真实数据，要么由于性能差而无法应用于复杂的信息抽取任务。为了解决这些挑战，我们提出了STAR，一种利用大型语言模型（LLMs）根据有限的种子示例合成数据实例，从而提高低资源信息抽取性能的数据生成方法。

    Information extraction tasks such as event extraction require an in-depth understanding of the output structure and sub-task dependencies. They heavily rely on task-specific training data in the form of (passage, target structure) pairs to obtain reasonable performance. However, obtaining such data through human annotation is costly, leading to a pressing need for low-resource information extraction approaches that require minimal human labeling for real-world applications. Fine-tuning supervised models with synthesized training data would be a generalizable method, but the existing data generation methods either still rely on large-scale ground-truth data or cannot be applied to complicated IE tasks due to their poor performance. To address these challenges, we propose STAR, a data generation method that leverages Large Language Models (LLMs) to synthesize data instances given limited seed demonstrations, thereby boosting low-resource information extraction performance. Our approach i
    
[^167]: LogicLLM：探索自监督逻辑增强训练的大语言模型

    LogicLLM: Exploring Self-supervised Logic-enhanced Training for Large Language Models. (arXiv:2305.13718v1 [cs.CL])

    [http://arxiv.org/abs/2305.13718](http://arxiv.org/abs/2305.13718)

    本文介绍了 LogicLLM，一种通过自监督后训练来提高大语言模型的逻辑推理能力的方法，该方法有效地在常见逻辑推理任务上进行表现，超过了目前最先进的无监督基线方法。

    

    改善语言模型的逻辑推理能力的现有努力主要依赖于有监督微调，这阻碍了将模型泛化到新的领域和/或任务。然而，通过发展大语言模型（LLM）已经证明了将丰富的知识压缩为单个代理的能力，使它们能够有效地处理多个任务。然而，我们的初步实验表明，LLMs 在逻辑推理方面并没有表现出能力。LLMs 在逻辑推理基准测试中的表现远远落后于现有的最先进基线。在本文中，我们首次尝试通过自监督后训练来探索融合逻辑知识的可行性，并通过上下文学习来激活它，我们将其称为LogicLLM。具体来说，我们设计了一个MERIt 的自回归目标变体，并将其与两个LLM系列FLAN-T5和LLaMA集成在一起，参数大小范围从30亿到130亿。实验结果表明，我们的方法在常用推理策略上与目前最先进的有监督方法相当，并且远远超过了目前最先进的无监督基线方法。

    Existing efforts to improve logical reasoning ability of language models have predominantly relied on supervised fine-tuning, hindering generalization to new domains and/or tasks. The development of Large Langauge Models (LLMs) has demonstrated the capacity of compressing abundant knowledge into a single proxy, enabling them to tackle multiple tasks effectively. Our preliminary experiments, nevertheless, show that LLMs do not show capability on logical reasoning. The performance of LLMs on logical reasoning benchmarks is far behind the existing state-of-the-art baselines. In this paper, we make the first attempt to investigate the feasibility of incorporating logical knowledge through self-supervised post-training, and activating it via in-context learning, which we termed as LogicLLM. Specifically, we devise an auto-regressive objective variant of MERIt and integrate it with two LLM series, i.e., FLAN-T5 and LLaMA, with parameter size ranging from 3 billion to 13 billion. The results 
    
[^168]: Chain-of-Knowledge:通过多源动态知识适应为大型语言模型提供准确的基础信息

    Chain-of-Knowledge: Grounding Large Language Models via Dynamic Knowledge Adapting over Heterogeneous Sources. (arXiv:2305.13269v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.13269](http://arxiv.org/abs/2305.13269)

    Chain-of-Knowledge通过整合多源动态知识为大型语言模型提供准确的基础信息，减少生成的幻觉，可以产生更可靠的答案。

    

    我们提出了一种新颖的框架，链式知识（CoK），通过动态地整合来自不同来源的基础信息来增强大型语言模型(LLMs)。它可以产生更多的事实依据，减少生成的幻觉。具体而言，CoK包括三个阶段：推理准备、动态知识适应和答案整合。给定一个知识密集型问题，CoK首先准备若干个初步的依据和答案，同时识别出相关的知识领域。如果样本中的答案没有多数共识，CoK通过从识别出的领域中逐步适应知识来纠正依据。这些纠正后的依据可以更好地作为最终答案整合的基础。不同于之前主要使用非结构化数据的研究，CoK还利用结构化的知识源，如Wikidata和表格，提供更可靠的事实信息。

    We present chain-of-knowledge (CoK), a novel framework that augments large language models (LLMs) by dynamically incorporating grounding information from heterogeneous sources. It results in more factual rationales and reduced hallucination in generation. Specifically, CoK consists of three stages: reasoning preparation, dynamic knowledge adapting, and answer consolidation. Given a knowledge-intensive question, CoK first prepares several preliminary rationales and answers while identifying the relevant knowledge domains. If there is no majority consensus among the answers from samples, CoK corrects the rationales step by step by adapting knowledge from the identified domains. These corrected rationales can plausibly serve as a better foundation for the final answer consolidation. Unlike prior studies that primarily use unstructured data, CoK also leverages structured knowledge sources such as Wikidata and tables that provide more reliable factual information. To access both unstructure
    
[^169]: CRITIC：大型语言模型可以通过工具交互批评进行自我校正

    CRITIC: Large Language Models Can Self-Correct with Tool-Interactive Critiquing. (arXiv:2305.11738v1 [cs.CL])

    [http://arxiv.org/abs/2305.11738](http://arxiv.org/abs/2305.11738)

    本文提出了一个名为CRITIC的框架，使得大型语言模型可以通过与工具的交互校正自己的错误，从而避免生成出现不一致和问题行为的结果。

    

    近年来，大型语言模型的发展非常引人注目。然而，这些模型有时会出现不一致和问题行为，例如出现幻觉事实，生成有缺陷的代码或创建冒犯和有害的内容。与这些模型不同，人类通常使用外部工具来交叉检查和精炼他们的初步内容，例如使用搜索引擎进行事实检查或使用代码解释器进行调试。受这一观察的启发，我们引入了一个名为CRITIC的框架，允许LLMs（实质上是“黑盒子”）以类似于人类与工具交互的方式验证和逐步修正自己的输出。更具体地说，从初始输出开始，CRITIC与适当的工具交互以评估文本的某些方面，然后根据在此验证过程中获得的反馈修改输出。涉及自由形式问答、数学程序综合和毒性检测的全面评估表明，我们的框架使LLMs能够从错误中学习并纠正自己的错误。

    Recent developments in large language models (LLMs) have been impressive. However, these models sometimes show inconsistencies and problematic behavior, such as hallucinating facts, generating flawed code, or creating offensive and toxic content. Unlike these models, humans typically utilize external tools to cross-check and refine their initial content, like using a search engine for fact-checking, or a code interpreter for debugging. Inspired by this observation, we introduce a framework called CRITIC that allows LLMs, which are essentially "black boxes" to validate and progressively amend their own outputs in a manner similar to human interaction with tools. More specifically, starting with an initial output, CRITIC interacts with appropriate tools to evaluate certain aspects of the text, and then revises the output based on the feedback obtained during this validation process. Comprehensive evaluations involving free-form question answering, mathematical program synthesis, and toxi
    
[^170]: InstructIE: 一份基于指令的中文信息提取数据集

    InstructIE: A Chinese Instruction-based Information Extraction Dataset. (arXiv:2305.11527v1 [cs.CL])

    [http://arxiv.org/abs/2305.11527](http://arxiv.org/abs/2305.11527)

    介绍了一份中文的基于指令的信息提取数据集InstructIE，其中包括了270,000个弱监督的数据和1,000个高质量注释实例。实验结果表明当前的模型表现有待改进，该任务仍存在挑战。

    

    我们引入了一项新的信息提取任务，称为基于指令的信息提取 (Instruction-based IE)，它旨在要求系统遵循特定的指令或指南来提取信息。为了促进该领域的研究，我们构建了一个数据集，称为InstructIE，其中包括来自中文维基百科的 270,000 个弱监督数据和 1,000 个高质量众包注释实例。我们进一步评估了各种基线模型在InstructIE数据集上的表现。结果表明，尽管当前的模型表现很有希望，但仍有改进的空间。此外，我们进行了全面的案例研究分析，强调了基于指令的信息提取任务中固有的挑战。代码和数据集可在 https://github.com/zjunlp/DeepKE/tree/main/example/llm 找到。

    We introduce a new Information Extraction (IE) task dubbed Instruction-based IE, which aims to ask the system to follow specific instructions or guidelines to extract information. To facilitate research in this area, we construct a dataset called InstructIE, consisting of 270,000 weakly supervised data from Chinese Wikipedia and 1,000 high-quality crowdsourced annotated instances. We further evaluate the performance of various baseline models on the InstructIE dataset. The results reveal that although current models exhibit promising performance, there is still room for improvement. Furthermore, we conduct a comprehensive case study analysis, underlining the challenges inherent in the Instruction-based IE task. Code and dataset are available at https://github.com/zjunlp/DeepKE/tree/main/example/llm.
    
[^171]: 有条件生成模型中的数据编辑

    Data Redaction from Conditional Generative Models. (arXiv:2305.11351v1 [cs.LG])

    [http://arxiv.org/abs/2305.11351](http://arxiv.org/abs/2305.11351)

    本文研究如何对已训练好的条件生成模型进行后期编辑，以便编辑掉某些条件分支，这些条件分支很可能会生成不良内容。通过精简模型中的条件网络实现，提出的解决方案有效、高效、具有可控性和普适性，在文本到图像和文本到语音生成模型中取得了良好效果。

    

    深度生成模型因生成不良内容而受到批评。传统的缓解方法包括重新训练、过滤或编辑；然而这些方法要么计算成本高，要么会被第三方回避。本文提出一种不同的方法，研究如何后期编辑已经训练好的条件生成模型，使其编辑掉某些条件分支，这些条件分支很可能会生成不良内容。这是通过精简模型中的条件网络来实现的，提出的解决方案既有效又高效、具有可控性和普适性，能用于一类深度生成模型。我们在文本到图像生成模型和文本到语音生成模型中进行了数据编辑实验，并表明我们的方法计算成本较低，相比基线方法具有更好的编辑质量和鲁棒性，同时仍保持高生成质量。

    Deep generative models are known to produce undesirable samples such as harmful content. Traditional mitigation methods include re-training from scratch, filtering, or editing; however, these are either computationally expensive or can be circumvented by third parties. In this paper, we take a different approach and study how to post-edit an already-trained conditional generative model so that it redacts certain conditionals that will, with high probability, lead to undesirable content. This is done by distilling the conditioning network in the models, giving a solution that is effective, efficient, controllable, and universal for a class of deep generative models. We conduct experiments on redacting prompts in text-to-image models and redacting voices in text-to-speech models. Our method is computationally light, leads to better redaction quality and robustness than baseline methods while still retaining high generation quality.
    
[^172]: 错误分析提示使得大型语言模型在翻译评估方面实现了人类水平：以ChatGPT为例进行案例研究

    Error Analysis Prompting Enables Human-Like Translation Evaluation in Large Language Models: A Case Study on ChatGPT. (arXiv:2303.13809v1 [cs.CL])

    [http://arxiv.org/abs/2303.13809](http://arxiv.org/abs/2303.13809)

    本文提出一种新的提示方法Error Analysis Prompting可改善LLMs在机器翻译质量评估上的性能，实现人类水平的评估。

    

    生成式大型语言模型（LLM），例如ChatGPT，在机器翻译、问答、文本摘要和自然语言理解等多个NLP任务上表现出卓越的能力。最近的研究表明，利用ChatGPT评估机器翻译质量在系统水平上取得了最先进的性能，但在段落水平上表现不佳。为了进一步提高LLM在机器翻译质量评估上的性能，我们进行了关于几种提示方法的研究。我们的结果表明，通过将Chain-of-Thoughts和Error Analysis结合起来，一种新的提示方法Error Analysis Prompting，像ChatGPT这样的LLM可以在系统和段落级别上生成人类般的机器翻译评估。此外，我们发现ChatGPT作为机器翻译评估器存在一些局限性，例如在提供单个查询中的多个译文时存在不稳定的评分和偏差。

    Generative large language models (LLMs), e.g., ChatGPT, have demonstrated remarkable proficiency across several NLP tasks such as machine translation, question answering, text summarization, and natural language understanding. Recent research has shown that utilizing ChatGPT for assessing the quality of machine translation (MT) achieves state-of-the-art performance at the system level but performs poorly at the segment level. To further improve the performance of LLMs on MT quality assessment, we conducted an investigation into several prompting methods. Our results indicate that by combining Chain-of-Thoughts and Error Analysis, a new prompting method called \textbf{\texttt{Error Analysis Prompting}}, LLMs like ChatGPT can \textit{generate human-like MT evaluations at both the system and segment level}. Additionally, we discovered some limitations of ChatGPT as an MT evaluator, such as unstable scoring and biases when provided with multiple translations in a single query. Our findings
    
[^173]: 使用会话式语言模型和提示工程从研究论文中提取准确的材料数据

    Extracting Accurate Materials Data from Research Papers with Conversational Language Models and Prompt Engineering. (arXiv:2303.05352v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2303.05352](http://arxiv.org/abs/2303.05352)

    本论文提出了ChatExtract方法，使用先进的对话式语言模型和提示工程，自动从研究论文中提取准确的数据，不需要大量的前期努力和背景知识。

    

    人们正在不断努力用自然语言处理、语言模型和最近出现的大型语言模型（LLM）代替手工从研究论文中提取数据的工作。尽管这些方法可以高效地从大量研究论文中提取数据，但它们需要大量的前期努力、专业知识和编码。在这项工作中，我们提出了ChatExtract方法，它可以通过一个先进的对话式LLM自动提取极准确的数据，几乎不需要初期的努力和背景知识。ChatExtract由一组工程化的提示应用于对话式LLM，既可以识别出具有数据的句子，提取出这些数据，又可以通过一系列跟进问题确保数据的正确性。这些跟进问题很大程度上克服了LLM提供事实不准确答案的已知问题。ChatExtract可以应用于任何对话式LLM，并能提供非常高质量的数据提取。

    There has been a growing effort to replace hand extraction of data from research papers with automated data extraction based on natural language processing, language models, and recently, large language models (LLMs). Although these methods enable efficient extraction of data from large sets of research papers, they require a significant amount of up-front effort, expertise, and coding. In this work we propose the ChatExtract method that can fully automate very accurate data extraction with minimal initial effort and background, using an advanced conversational LLM. ChatExtract consists of a set of engineered prompts applied to a conversational LLM that both identify sentences with data, extract that data, and assure the data's correctness through a series of follow-up questions. These follow-up questions largely overcome known issues with LLMs providing factually inaccurate responses. ChatExtract can be applied with any conversational LLMs and yields very high quality data extraction.
    
[^174]: 无监督的逐层文本OOD检测得分聚合

    Unsupervised Layer-wise Score Aggregation for Textual OOD Detection. (arXiv:2302.09852v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.09852](http://arxiv.org/abs/2302.09852)

    提出了一种无监督的逐层聚合异常得分的方法，用于更好地进行文本OOD检测。其能发掘不同层输出的优势，达到更鲁棒的性能，并扩展经典基准测试以反映更现实的设置。

    

    随着越来越多基于AI的系统增加，OOD检测是一个迅速发展的领域，由于新的鲁棒性和安全性要求。现有的OOD文本检测器通常依赖于在编码器的最后一层输出上计算的异常得分（例如马氏距离）。在这项工作中，我们观察到OOD检测性能因任务和层输出而异。更重要的是，我们表明通常的选择（最后一层）很少是OOD检测的最佳选择，如果选择最佳层，则可以获得更好的结果。为了利用这个观察结果，我们提出了一种数据驱动的无监督方法来结合逐层的异常得分。此外，我们通过包括更多类别的分类任务（高达77）扩展了经典文本OOD基准测试，从而反映更现实的设置。在这个增强的基准测试上，我们展示了所提出的后聚合方法实现了鲁棒的OOD检测性能。

    Out-of-distribution (OOD) detection is a rapidly growing field due to new robustness and security requirements driven by an increased number of AI-based systems. Existing OOD textual detectors often rely on an anomaly score (e.g., Mahalanobis distance) computed on the embedding output of the last layer of the encoder. In this work, we observe that OOD detection performance varies greatly depending on the task and layer output. More importantly, we show that the usual choice (the last layer) is rarely the best one for OOD detection and that far better results could be achieved if the best layer were picked. To leverage this observation, we propose a data-driven, unsupervised method to combine layer-wise anomaly scores. In addition, we extend classical textual OOD benchmarks by including classification tasks with a greater number of classes (up to 77), which reflects more realistic settings. On this augmented benchmark, we show that the proposed post-aggregation methods achieve robust an
    

