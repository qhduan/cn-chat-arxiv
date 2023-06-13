# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [A Comprehensive Review of State-of-The-Art Methods for Java Code Generation from Natural Language Text.](http://arxiv.org/abs/2306.06371) | 本文全面回顾了使用深度学习模型从自然语言文本中自动生成Java代码的方法，并重点关注了变压器模型的三种类型：编码器模型、解码器模型和编码-解码模型。 |
| [^2] | [Improving Non-autoregressive Translation Quality with Pretrained Language Model, Embedding Distillation and Upsampling Strategy for CTC.](http://arxiv.org/abs/2306.06345) | 本文提出了一些技术来提高非自回归翻译模型的翻译质量，在保持显着推理速度加速的同时，通过使用预训练多语言模型进行微调、采用MASK插入方案进行上采样、以及采用嵌入蒸馏方法来进一步提高性能。在多个数据集上，模型表现优于基线自回归模型。 |
| [^3] | [Towards Arabic Multimodal Dataset for Sentiment Analysis.](http://arxiv.org/abs/2306.06322) | 本文针对阿拉伯语DL-based MSA领域缺乏标准数据集的问题，通过使用最先进的转换器和单词对齐技术中的特征提取工具来设计管道流程，构建了阿拉伯语多模态数据集，实验表明其具有很大的潜力。 |
| [^4] | [Protect Your Prompts: Protocols for IP Protection in LLM Applications.](http://arxiv.org/abs/2306.06297) | 本文讨论了两个协议，旨在保护LLM提示的知识产权，提供开放市场上交易的可能性。 |
| [^5] | [Measuring and Modifying Factual Knowledge in Large Language Models.](http://arxiv.org/abs/2306.06264) | 这项研究提出了一种基于信息论的测量框架，可用于衡量大型语言模型中的事实知识，并通过熵及KL散度等度量指标进行知识修改，超越了以前的排名方法，并提供了一种有价值的工具，用于测量和修改LLMs中的大量事实知识。 |
| [^6] | [Record Deduplication for Entity Distribution Modeling in ASR Transcripts.](http://arxiv.org/abs/2306.06246) | 本研究提出了一种通过记录重复消除来模拟客户真实请求的实体分布，从而解决ASR错误识别带来的实体分布复杂问题，并成功地应用于语境偏置中，显示出估计的单词错误率降低5%。 |
| [^7] | [Using Foundation Models to Detect Policy Violations with Minimal Supervision.](http://arxiv.org/abs/2306.06234) | 本文利用基础模型在极少监督下检测政策违规，创新性地将思维链提示引入政策违规任务，同时将硬提示与软提示相结合，可以高准确度地生成合理解释。 |
| [^8] | [Probing self-supervised speech models for phonetic and phonemic information: a case study in aspiration.](http://arxiv.org/abs/2306.06232) | 本文研究了自监督语音模型中的音素和音位信息，并发现这些模型在早期层就能够很好地表示这些区别，并且这种表示在更深层的表示中得以保留，是由于模型在语音数据上的优化和高维度的体系结构的共同作用所致。 |
| [^9] | [Conformalizing Machine Translation Evaluation.](http://arxiv.org/abs/2306.06221) | 本文提出了一种无分布方法——规范化预测，用于机器翻译评估置信区间的计算，并证明其具有理论保证覆盖率，可以纠正其他方法的误差。应用条件规范化预测技术，获得平等覆盖的校准子集。 |
| [^10] | [Morphosyntactic probing of multilingual BERT models.](http://arxiv.org/abs/2306.06205) | 该论文介绍了一个广泛的数据集，用于在42种语言中探测语言模型的形态句法信息。研究发现，预训练的Transformer模型（mBERT和XLM-RoBERTa）在这些任务中表现出色，而前面的上下文比后面的上下文包含更多与预测相关的信息。 |
| [^11] | [Reliability Check: An Analysis of GPT-3's Response to Sensitive Topics and Prompt Wording.](http://arxiv.org/abs/2306.06199) | 本文分析了大型语言模型GPT-3对敏感话题和提示措辞的反应，发现其在阴谋论和刻板印象方面有正确的反应，但在误解和争议方面存在错误，并具有不可靠性。 |
| [^12] | [$FPDM$: Domain-Specific Fast Pre-training Technique using Document-Level Metadata.](http://arxiv.org/abs/2306.06190) | 本文提出了$FPDM$，使用文档元数据和领域特定分类作为监督信号，对领域特定语料库进行transformer编码器的预训练。$FPDM$通过句子级别的输入预训练开放领域的编码器，在微调时使用词汇级别的输入，性能优于其他基于transformer的模型。 |
| [^13] | [SentiGOLD: A Large Bangla Gold Standard Multi-Domain Sentiment Analysis Dataset and its Evaluation.](http://arxiv.org/abs/2306.06147) | 本文介绍了SentiGOLD，一个孟加拉语多领域情感分析数据集。该数据集由70,000个来自不同来源的样本组成，遵守了政府和语言学委员会商定的语言约定，包括了30个领域和5个情感类别。在具有鲁棒性的注释方案下，该数据集的互评一致性表现出色，可用于建立孟加拉语情感分析模型。 |
| [^14] | [Zero-Shot Dialogue Relation Extraction by Relating Explainable Triggers and Relation Names.](http://arxiv.org/abs/2306.06141) | 本文提出了一种无监督对话关系抽取方法，它能够捕捉触发器并将其与以前未见过的关系名称相关联，能够对推断先前未见过的关系类型极具帮助。 |
| [^15] | [INSTRUCTEVAL: Towards Holistic Evaluation of Instruction-Tuned Large Language Models.](http://arxiv.org/abs/2306.04757) | INSTRUCTEVAL是一个专注于指导调整的大型语言模型评估的综合套件，它采取了全面的方法来评估模型的性能，包括解决问题、写作能力和与人类价值观的一致性等特征。 |
| [^16] | [ConTextual Masked Auto-Encoder for Retrieval-based Dialogue Systems.](http://arxiv.org/abs/2306.04357) | 本研究提出了一种针对对话响应选择的后训练技术Dial-MAE，利用生成方法更好地压缩对话语义至密集向量，并提高对话响应选择准确性。 |
| [^17] | [Revisiting Conversation Discourse for Dialogue Disentanglement.](http://arxiv.org/abs/2306.03975) | 本文提出了一种利用对话话语特征增强对话分离的方法，通过构建异构图表示和引入增强分离目标，以更好地建模对话语境和利用内在的话语结构信息。该方法在不同的对话分离基准数据集上表现出优异的性能。 |
| [^18] | [TKDP: Threefold Knowledge-enriched Deep Prompt Tuning for Few-shot Named Entity Recognition.](http://arxiv.org/abs/2306.03974) | 本文提出了一个名为 TKDP 的方法，在深度提示调整中通过整合三种不同来源的知识来增强少样本命名实体识别的性能。 在五个基准数据集上，相对于原始的深度提示方法提高了最多 11.53% 的 F1，并且明显优于 8 种表现强劲的 few-shot NER 方法。 |
| [^19] | [ECQED: Emotion-Cause Quadruple Extraction in Dialogs.](http://arxiv.org/abs/2306.03969) | 本文提出了一个新的对话中的情感-原因四元组抽取任务(ECQED)，通过引入对话上下文和细粒度的情感和原因检测，有效地提高了任务的性能。 |
| [^20] | [Evaluating the Effectiveness of Natural Language Inference for Hate Speech Detection in Languages with Limited Labeled Data.](http://arxiv.org/abs/2306.03722) | 本文对自然语言推理（NLI）模型在标记数据有限语言中进行仇恨言论检测的有效性进行了研究。结果表明，NLI模型可以显著提高仇恨言论检测的性能，但在英语数据匹配测试领域时仅有自定义的NLI配方能够胜过中间英语微调。 |
| [^21] | [Applying Standards to Advance Upstream & Downstream Ethics in Large Language Models.](http://arxiv.org/abs/2306.03503) | 本文探讨如何为AI生成的内容制定安全保障，分析LLMs的内容生成机制，确定了四个关键领域，提出了新的分发和销售LLM生成内容的企业的标准。 |
| [^22] | [Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding.](http://arxiv.org/abs/2306.02858) | Video-LLaMA是一个多模态框架，利用已有的预训练模型，解决了视频中的视觉和听觉的理解问题，其中Video Q-former和Audio Q-former用于处理视频中的视觉与时间变化和音频信号的问题。 |
| [^23] | [LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion.](http://arxiv.org/abs/2306.02561) | 本论文提出了LLM-Blender，它是一个集成框架，旨在利用不同的开源大型语言模型的优秀特性，实现始终如一的卓越性能。PairRanker和GenFuser是该框架的两个模块，PairRanker使用成对比较方法来区分候选输出，并且GenFuser旨在合并排名最高的候选者，以生成改进的输出。 |
| [^24] | [Evolution of Efficient Symbolic Communication Codes.](http://arxiv.org/abs/2306.02383) | 本文探讨了通过反熵、压缩因子和跨分裂F1分数为目标的交流代码演变产物，发现语言结构形成可以通过这些度量来驱动。 |
| [^25] | [Benchmarking Robustness of Adaptation Methods on Pre-trained Vision-Language Models.](http://arxiv.org/abs/2306.02080) | 研究针对预训练视觉语言模型的11种适应方法在不同污染情况下的鲁棒性，发现适应方法对文本污染更敏感，单独使用小型文本适配器比共享适配器更鲁棒，可获得可比较的干净性能。 |
| [^26] | [Make Your Pre-trained Model Reversible: From Parameter to Memory Efficient Fine-Tuning.](http://arxiv.org/abs/2306.00477) | 本研究尝试实现在预训练语言模型中运用可逆模型实现高效的微调，并发现在初始化微调时保留PLM的起点非常重要。 |
| [^27] | [Focused Prefix Tuning for Controllable Text Generation.](http://arxiv.org/abs/2306.00369) | 本文提出了针对可控文本生成的焦点前缀调整方法，实验结果表明在单属性控制任务中实现了更好的控制准确性和文本流畅度，在多属性控制任务中实现了与最先进方法相当的控制准确性，并保持了控制新属性而无需重新训练现有模型的灵活性。 |
| [^28] | [Data Augmentation Approaches for Source Code Models: A Survey.](http://arxiv.org/abs/2305.19915) | 本文对源代码的数据增强技术进行了全面的调查和综述，介绍了它们的分类法、优化策略和性能结果，并讨论了未来方向和研究挑战。 |
| [^29] | [DC CoMix TTS: An End-to-End Expressive TTS with Discrete Code Collaborated with Mixer.](http://arxiv.org/abs/2305.19567) | 本文提出了一种基于离散码和混合器相协作的端到端表现力TTS，它采用新的输入表示和简单的架构来实现改进的韵律建模，证明了其有效性。 |
| [^30] | [Fine-grained Text Style Transfer with Diffusion-Based Language Models.](http://arxiv.org/abs/2305.19512) | 本文提出了一种基于扩散式语言模型的细粒度文本风格转换方法，在不依赖外部信息的情况下取得了比之前利用预训练权重、嵌入和外部语法分析器更好的效果，表明扩散概率模型在文本生成领域具有广泛的应用前景。 |
| [^31] | [infoVerse: A Universal Framework for Dataset Characterization with Multidimensional Meta-information.](http://arxiv.org/abs/2305.19344) | 本文介绍了一种通用的数据集特征化框架infoVerse，通过结合各种模型驱动的元信息提供了一个新的特征空间，能够有效地捕捉数据集的多维特征，有助于用户或模型确定哪些样本需要关注。 |
| [^32] | [Mitigating Label Biases for In-context Learning.](http://arxiv.org/abs/2305.19148) | 本文针对上下文学习（ICL）中的三种标签偏差提出分类法，并提出一种简单的偏差校准方法，使用随机的领域词估算语言模型的标签偏差。 |
| [^33] | [W-procer: Weighted Prototypical Contrastive Learning for Medical Few-Shot Named Entity Recognition.](http://arxiv.org/abs/2305.18624) | W-procer是一种基于加权原型对比学习的医学少样本命名实体识别方法，在构建基于原型的对比损失和加权网络方面具有创新性，优于现有的最先进方法。 |
| [^34] | [One Network, Many Masks: Towards More Parameter-Efficient Transfer Learning.](http://arxiv.org/abs/2305.17682) | 本研究提出了PROPETL方法，通过原型网络和二进制掩码实现了更高效的参数共用迁移学习，解决了多任务微调预训练语言模型存储空间占用的问题。 |
| [^35] | [Backdooring Neural Code Search.](http://arxiv.org/abs/2305.17506) | 本文研究了神经代码搜索模型的安全性问题，指出攻击者可以注入后门来返回具有安全/隐私问题的代码，提出了几种防御机制来缓解这种威胁，该工作突显了研究AI系统安全方面的重要性，特别是在部署于安全关键领域时。 |
| [^36] | [Evaluating Open-Domain Dialogues in Latent Space with Next Sentence Prediction and Mutual Information.](http://arxiv.org/abs/2305.16967) | 本文提出了一种新的基于学习的自动评估度量方法（CMN），能够通过将条件变分自编码器（CVAEs）与下一句预测（NSP）目标相结合，并利用互信息（MI）在潜空间中建模文本的语义相似度，来鲁棒地评估开放域对话，并在实验中取得了优异的结果。 |
| [^37] | [Surface-Based Retrieval Reduces Perplexity of Retrieval-Augmented Language Models.](http://arxiv.org/abs/2305.16243) | 本文研究发现，用基于表面级别的检索机制取代语义检索可以显著降低检索增强语言模型的困惑度。 |
| [^38] | [Enhancing Grammatical Error Correction Systems with Explanations.](http://arxiv.org/abs/2305.15676) | 该论文介绍了一个用解释提高语法错误修正系统能力的方法，通过引入包含证据单词和语法错误类型注释的大数据集，找到错误的原因，并提出了几个基线和分析方法来理解这个任务，同时也证明了解释可以帮助第二语言学习者更好地理解语法规则。 |
| [^39] | [Understanding Programs by Exploiting (Fuzzing) Test Cases.](http://arxiv.org/abs/2305.13592) | 本文提出了通过模糊测试获取代表性输入来帮助语义理解程序的方法。 |
| [^40] | [Ambiguity Meets Uncertainty: Investigating Uncertainty Estimation for Word Sense Disambiguation.](http://arxiv.org/abs/2305.13119) | 本研究探究了用于词义消歧的不确定性估计方法，发现传统预测概率不足以量化不确定性，同时发现模型令人满意地反映了数据不确定性但是低估了模型不确定性。 |
| [^41] | [Evaluating Prompt-based Question Answering for Object Prediction in the Open Research Knowledge Graph.](http://arxiv.org/abs/2305.12900) | 本研究采用基于提示的训练方法，在学术知识图谱对象预测领域进行了大规模transformers模型的评估和测试，发现提示的使用可以改进pre-trained transformers的泛化能力。 |
| [^42] | [Chain-of-Symbol Prompting Elicits Planning in Large Langauge Models.](http://arxiv.org/abs/2305.10276) | 本文提出了自然语言规划（NLP）的基准，旨在研究LLMs在需要理解并在文本中相应进行操作的复杂规划任务中的表现。同时提出了一种新方法CoS，使用简化的符号空间表示法来表示复杂的环境。 |
| [^43] | [Language, Time Preferences, and Consumer Behavior: Evidence from Large Language Models.](http://arxiv.org/abs/2305.02531) | 本研究分析了大型语言模型在不同语言提示下的奖励时间偏好，并发现GPT在具有较弱未来时态的语言下表现出更大的耐心，这与使用该语言的人类的偏好相似。 |
| [^44] | [A Statistical Exploration of Text Partition Into Constituents: The Case of the Priestly Source in the Books of Genesis and Exodus.](http://arxiv.org/abs/2305.02170) | 为了验证文本分组的假设，我们提出了一个统计文本探索的流程，并在圣经的前两卷书中应用此流程，成功地识别并探索了司祭派别和非司祭派别之间的统计明显的文体差异。 |
| [^45] | [Causality-aware Concept Extraction based on Knowledge-guided Prompting.](http://arxiv.org/abs/2305.01876) | 该论文提出了一种基于因果感知的知识引导提示方法，将其作为干预器装备到基于预训练语言模型的句子提取器中，以缓解概念偏差。在代表性的多语言KG数据集上进行广泛实验，获得了最先进的结果。 |
| [^46] | [Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation.](http://arxiv.org/abs/2305.01210) | 本论文提出了一个严格的代码综合基准评估框架EvalPlus，用于评估利用大型语言模型生成的代码的功能正确性。 |
| [^47] | [ChartSumm: A Comprehensive Benchmark for Automatic Chart Summarization of Long and Short Summaries.](http://arxiv.org/abs/2304.13620) | 本文提出了ChartSumm数据集，用于长短摘要自动生成任务，包括84000多个图表及其元数据和描述。研究发现，现有的自动摘要模型虽然得分不错，但经常面临错觉、漏掉重要数据点以及不正确解释复杂趋势等问题。 |
| [^48] | [WizardLM: Empowering Large Language Models to Follow Complex Instructions.](http://arxiv.org/abs/2304.12244) | 本文使用 Evol-Instruct 方法创建了大量不同复杂度的指令数据用于微调 LLaMA 模型，得到了新模型 WizardLM。人类评估结果表明 Evol-Instruct 生成的指令优于人工创建的，而 WizardLM 输出的结果也比 OpenAI ChatGPT 更受欢迎。 |
| [^49] | [(Vector) Space is Not the Final Frontier: Product Search as Program Synthesis.](http://arxiv.org/abs/2304.11473) | 本文主张将产品搜索看作程序合成，相比向量空间模型有着重大优势。 |
| [^50] | [LLM as A Robotic Brain: Unifying Egocentric Memory and Control.](http://arxiv.org/abs/2304.09349) | 本文提出了一个统一自我中心记忆和控制的框架LLM-Brain，使用大规模语言模型作为机器人大脑进行零-shot学习。该框架包括封闭式多轮对话，覆盖了感知、规划、控制和记忆，具有很好的泛化性能，适用于多个机器人任务。 |
| [^51] | [Language Instructed Reinforcement Learning for Human-AI Coordination.](http://arxiv.org/abs/2304.07297) | 本文提出了一种称之为instructRL的新的框架，它通过自然语言指令来指定对人工智能搭档的预期策略，解决在缺乏高质量人类行为数据的领域中多智能体强化学习收敛于人类不偏爱的策略的问题，从而提高了人工智能协作的性能。 |
| [^52] | [Multimodal C4: An Open, Billion-scale Corpus of Images Interleaved With Text.](http://arxiv.org/abs/2304.06939) | Multimodal C4是一个开放的、以图像与文本交替形式存在的数据库，其使用线性分配算法将图像放到长文本段落中，可用于通过少量样本学习和复杂相关度提示的建模。 |
| [^53] | [Vax-Culture: A Dataset for Studying Vaccine Discourse on Twitter.](http://arxiv.org/abs/2304.06858) | 本文介绍了一个推特疫苗数据集Vax-Culture，它旨在找出推广疫苗错误信息的文化和政治信念的重叠部分，帮助开发机器学习模型以自动检测疫苗错误信息帖子并应对其负面影响。 |
| [^54] | [SwissBERT: The Multilingual Language Model for Switzerland.](http://arxiv.org/abs/2303.13310) | 该论文介绍了SwissBERT，它是一个专门为处理瑞士相关文本而创建的多语言语言模型，SwissBERT在与瑞士相关的自然语言理解任务上的效果优于以前的模型。 |
| [^55] | [Reflexion: an autonomous agent with dynamic memory and self-reflection.](http://arxiv.org/abs/2303.11366) | 本文提出 Reflexion 方法，给智能体赋予了动态记忆和自我反思能力，以增强其任务特定的行动选择能力。 |
| [^56] | [Stop Words for Processing Software Engineering Documents: Do they Matter?.](http://arxiv.org/abs/2303.10439) | 本文研究了停用词在软件工程文档中的实用性。经实验证明，使用领域特定的停用词可以显著提高研究工具的性能，并且19个评估措施中有17个评估措施受益于停用词的消除。 |
| [^57] | [On the Robustness of Text Vectorizers.](http://arxiv.org/abs/2303.07203) | 本文研究了文本向量化技术中的鲁棒性问题，并证明了流行的嵌入方案具有Hamming距离意义上的鲁棒性。本研究提供了这些方法的定量边界，并展示了其中的常数受文档长度的影响。 |
| [^58] | [ChatGPT: Jack of all trades, master of none.](http://arxiv.org/abs/2302.10724) | 本研究检验了 ChatGPT 在 25 个不同的 NLP 任务上的性能，它是一个万能的 AI 模型，但无关紧要的表现可能会对某些任务的表现产生负面影响。 |
| [^59] | [MultiInstruct: Improving Multi-Modal Zero-Shot Learning via Instruction Tuning.](http://arxiv.org/abs/2212.10773) | 本文介绍了 MultiInstruct，这是第一个多模态指令调优基准数据集，并探索多种迁移学习策略从大规模的自然语言指令数据集中提高预训练模型的性能。实验结果展示了其在各种未见过的多模态任务中具有强大的零样本表现，以及设计的新的任务完成率指标。 |
| [^60] | [BLIND: Bias Removal With No Demographics.](http://arxiv.org/abs/2212.10563) | BLIND可以在没有先前了解数据集人口统计信息的情况下消除训练模型中的社会偏见。 |
| [^61] | [Continual Knowledge Distillation for Neural Machine Translation.](http://arxiv.org/abs/2212.09097) | 本论文提出了一种称为持续知识蒸馏的方法，利用已有的翻译模型来提高一个新模型的性能。 |
| [^62] | [Federated Neural Topic Models.](http://arxiv.org/abs/2212.02269) | 该论文提出了一种基于神经主题建模实现的联邦神经主题模型，可以在不共享数据的情况下允许多个方共同训练主题模型和保护节点隐私。 |
| [^63] | [Adaptation Approaches for Nearest Neighbor Language Models.](http://arxiv.org/abs/2211.07828) | 本论文提出了三种方法来适应半参数最近邻语言模型（$k$NN-LMs），并运用消融实验和对多个适应领域的广泛评估，发现组合适应方法 consistently outperforms单一适应策略和零样本（$k$NN-LM）基线，重计分模块使得性能提高最多。 |
| [^64] | [Does Debiasing Inevitably Degrade the Model Performance.](http://arxiv.org/abs/2211.07350) | 语言模型中的性别偏见问题日益引起关注，但当前去偏置方法往往会降低模型在其他任务上的表现；本论文提出了一个理论框架解释模型中性别偏见的机制，并发现了一种新的去偏置方法，能实现缓解性别偏见同时避免性能下降的双重优势。 |
| [^65] | [Generating Multilingual Gender-Ambiguous Text-to-Speech Voices.](http://arxiv.org/abs/2211.00375) | 该论文介绍了一种生成多语言的性别不明确的TTS声音的方法，通过提出的性别感知方法从潜在说话人中有效地进行采样，成功生成了一系列新的、多样化的、一致性和性别不明确性更强的声音，具有很强的实验表现。 |
| [^66] | [The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers.](http://arxiv.org/abs/2210.06313) | 本文研究了使用变压器模型的机器学习模型中激活图的稀疏现象，发现在不同层数的变压器配置和其他体系结构中都出现了稀疏现象。 |
| [^67] | [Improving Visual Grounding by Encouraging Consistent Gradient-based Explanations.](http://arxiv.org/abs/2206.15462) | 该论文提出了一种名为 AMC 的目标函数，鼓励基于梯度的解释覆盖有注释的感兴趣区域，即编码区域。该方法在提高视觉 grounding 性能方面表现卓越，有望成为视觉 grounding 领域的新进展。 |
| [^68] | [Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models.](http://arxiv.org/abs/2206.04615) | 本研究引入了Beyond the Imitation Game基准测试（BIG-bench），该测试集包含了204个各领域的难题，旨在评估当前语言模型的能力并为未来的研究提供信息和准备。 |
| [^69] | [DiMS: Distilling Multiple Steps of Iterative Non-Autoregressive Transformers for Machine Translation.](http://arxiv.org/abs/2206.02999) | 该论文提出了一种新的技术DiMS，可以通过压缩多步骤机制来优化解码过程，提高机器翻译的效率和质量。 |
| [^70] | [Cross-View Language Modeling: Towards Unified Cross-Lingual Cross-Modal Pre-training.](http://arxiv.org/abs/2206.00621) | 本文提出了跨视图语言建模框架，该框架将跨语言和跨模态预训练统一在共享的架构和目标下进行，通过有条件的掩码语言建模和对比学习来最大化不同视图之间的互信息以实现两个视图的对齐。 |
| [^71] | [ClaimDiff: Comparing and Contrasting Claims on Contentious Issues.](http://arxiv.org/abs/2205.12221) | 本研究提出ClaimDiff数据集，主要关注声明之间的微妙差别，并观察到强有力的基准测试无法探测这些差别，与人类存在超过19%的绝对差距。 |
| [^72] | [Recent Advances in Neural Text Generation: A Task-Agnostic Survey.](http://arxiv.org/abs/2203.03047) | 本文调查了最近神经文本生成领域的最新进展，包括数据构建、神经框架、训练和推理策略和评估指标等四个方面，并探讨了神经管道和背景知识的利用等未来方向。 |
| [^73] | [Measuring the Impact of Individual Domain Factors in Self-Supervised Pre-Training.](http://arxiv.org/abs/2203.00648) | 本研究对自动语音识别中的自监督预训练领域因素的影响进行了研究，结果表明语音学领域因素在预训练中起着重要作用，而语法和句法因素则远不及其重要，这有助于更好地了解自我监督训练语音预训练集的领域特征。 |
| [^74] | [Multi-Row, Multi-Span Distant Supervision For Table+Text Question.](http://arxiv.org/abs/2112.07337) | 这篇论文提出了一个名为MITQA的基于Transformer的TextTableQA系统，通过多实例学习以及远程监督方法，有效地解决了表格加文本问题的挑战。 |

# 详细

[^1]: 从自然语言文本生成Java代码的现有方法综述

    A Comprehensive Review of State-of-The-Art Methods for Java Code Generation from Natural Language Text. (arXiv:2306.06371v1 [cs.CL])

    [http://arxiv.org/abs/2306.06371](http://arxiv.org/abs/2306.06371)

    本文全面回顾了使用深度学习模型从自然语言文本中自动生成Java代码的方法，并重点关注了变压器模型的三种类型：编码器模型、解码器模型和编码-解码模型。

    

    Java代码生成是从自然语言文本中自动生成Java代码。这一自然语言处理任务有助于通过为程序员提供最简单和最重复的任务的即时解决方法来提高其生产力。代码生成是一项具有挑战性的任务，因为需要遵循严格的语法规则并深入理解编程语言的语义方面。许多研究尝试使用基于RNN或变压器模型来解决这个任务。后者在该领域取得了显著进展，可以分为三组：(1)仅编码模型，(2)仅解码模型和(3)编码-解码模型。本文全面回顾了深度学习模型在Java代码生成任务中的演变和进展。我们重点关注最重要的方法，并提出它们的优点和局限性，以及社区使用的目标函数。此外，我们提供了详细的描述...

    Java Code Generation consists in generating automatically Java code from a Natural Language Text. This NLP task helps in increasing programmers' productivity by providing them with immediate solutions to the simplest and most repetitive tasks. Code generation is a challenging task because of the hard syntactic rules and the necessity of a deep understanding of the semantic aspect of the programming language. Many works tried to tackle this task using either RNN-based, or Transformer-based models. The latter achieved remarkable advancement in the domain and they can be divided into three groups: (1) encoder-only models, (2) decoder-only models, and (3) encoder-decoder models. In this paper, we provide a comprehensive review of the evolution and progress of deep learning models in Java code generation task. We focus on the most important methods and present their merits and limitations, as well as the objective functions used by the community. In addition, we provide a detailed descripti
    
[^2]: 利用预训练语言模型、嵌入蒸馏和上采样策略改善非自回归翻译质量（arXiv:2306.06345v1 [cs.CL]）

    Improving Non-autoregressive Translation Quality with Pretrained Language Model, Embedding Distillation and Upsampling Strategy for CTC. (arXiv:2306.06345v1 [cs.CL])

    [http://arxiv.org/abs/2306.06345](http://arxiv.org/abs/2306.06345)

    本文提出了一些技术来提高非自回归翻译模型的翻译质量，在保持显着推理速度加速的同时，通过使用预训练多语言模型进行微调、采用MASK插入方案进行上采样、以及采用嵌入蒸馏方法来进一步提高性能。在多个数据集上，模型表现优于基线自回归模型。

    

    非自回归方法旨在提高翻译模型的推理速度，特别是那些可以一次正向传递生成输出的模型。但是，与自回归模型相比，这些方法往往在翻译质量上有显著的下降。本文引入了一系列创新技术，以提高非自回归翻译模型的翻译质量，同时保持推理速度的显著加速。我们建议使用CTC损失微调预训练多语言模型来有效地训练NAT模型。此外，我们采用MASK插入方案进行上采样，而不是令牌复制，并提出了一种嵌入蒸馏方法以进一步提高性能。在我们的实验中，我们的模型在多个数据集上优于基线自回归模型（Transformer base），包括WMT'14 DE $\leftrightarrow$ EN、WMT'16 RO $\leftrightarrow$ EN和IWSLT'14 DE $\leftrightarrow$ EN。

    Non-autoregressive approaches aim to improve the inference speed of translation models, particularly those that generate output in a one-pass forward manner. However, these approaches often suffer from a significant drop in translation quality compared to autoregressive models. This paper introduces a series of innovative techniques to enhance the translation quality of Non-Autoregressive Translation (NAT) models while maintaining a substantial acceleration in inference speed. We propose fine-tuning Pretrained Multilingual Language Models (PMLMs) with the CTC loss to train NAT models effectively. Furthermore, we adopt the MASK insertion scheme for up-sampling instead of token duplication, and we present an embedding distillation method to further enhance performance. In our experiments, our model outperforms the baseline autoregressive model (Transformer \textit{base}) on multiple datasets, including WMT'14 DE$\leftrightarrow$EN, WMT'16 RO$\leftrightarrow$EN, and IWSLT'14 DE$\leftright
    
[^3]: 面向阿拉伯语情感分析的多模态数据集构建

    Towards Arabic Multimodal Dataset for Sentiment Analysis. (arXiv:2306.06322v1 [cs.CL])

    [http://arxiv.org/abs/2306.06322](http://arxiv.org/abs/2306.06322)

    本文针对阿拉伯语DL-based MSA领域缺乏标准数据集的问题，通过使用最先进的转换器和单词对齐技术中的特征提取工具来设计管道流程，构建了阿拉伯语多模态数据集，实验表明其具有很大的潜力。

    

    多模态情感分析(MSA)已成为许多实际应用的中心研究方向。这种普及得益于意见对几乎所有人类活动的重要性，以及成为我们行为的关键因素。近年来，基于深度学习(DL)模型的高效性也已在多种西方语言中得到证明。然而，由于缺乏标准数据集，阿拉伯语DL-based MSA仍处于初始阶段。本文的探究目标有两个方面，首先，我们设计了一种管道流程，利用最先进的转换器以及单词对齐技术中的特征提取工具，帮助构建我们的阿拉伯语多模态数据集。然后，我们使用最先进的基于转换器的模型验证了我们的数据集。尽管输出数据集的规模较小，实验结果表明，阿拉伯语多模态仍然具有非常大的潜力。

    Multimodal Sentiment Analysis (MSA) has recently become a centric research direction for many real-world applications. This proliferation is due to the fact that opinions are central to almost all human activities and are key influencers of our behaviors. In addition, the recent deployment of Deep Learning-based (DL) models has proven their high efficiency for a wide range of Western languages. In contrast, Arabic DL-based multimodal sentiment analysis (MSA) is still in its infantile stage due, mainly, to the lack of standard datasets. In this paper, our investigation is twofold. First, we design a pipeline that helps building our Arabic Multimodal dataset leveraging both state-of-the-art transformers and feature extraction tools within word alignment techniques. Thereafter, we validate our dataset using state-of-the-art transformer-based model dealing with multimodality. Despite the small size of the outcome dataset, experiments show that Arabic multimodality is very promising
    
[^4]: LLM应用中的IP保护协议

    Protect Your Prompts: Protocols for IP Protection in LLM Applications. (arXiv:2306.06297v1 [cs.CL])

    [http://arxiv.org/abs/2306.06297](http://arxiv.org/abs/2306.06297)

    本文讨论了两个协议，旨在保护LLM提示的知识产权，提供开放市场上交易的可能性。

    

    随着大型语言模型（LLM）形式的AI的快速采用，精心设计的提示的潜在价值变得重要。然而，为了实现这个潜力，提示应该在一个公开市场上交易。由于提示目前通常是经济上不可排除的，因为它们是文本的性质，因此尚未建立一般的竞争市场。本文讨论了两个旨在提供提示保护的协议，提高其作为知识产权的地位，从而确认提示工程师的知识产权，并可能支持LLM提示的开放市场的繁荣。

    With the rapid adoption of AI in the form of large language models (LLMs), the potential value of carefully engineered prompts has become significant. However, to realize this potential, prompts should be tradable on an open market. Since prompts are, at present, generally economically non-excludable, by virtue of their nature as text, no general competitive market has yet been established. This note discusses two protocols intended to provide protection of prompts, elevating their status as intellectual property, thus confirming the intellectual property rights of prompt engineers, and potentially supporting the flourishing of an open market for LLM prompts.
    
[^5]: 在大型语言模型中测量和修改事实知识

    Measuring and Modifying Factual Knowledge in Large Language Models. (arXiv:2306.06264v1 [cs.CL])

    [http://arxiv.org/abs/2306.06264](http://arxiv.org/abs/2306.06264)

    这项研究提出了一种基于信息论的测量框架，可用于衡量大型语言模型中的事实知识，并通过熵及KL散度等度量指标进行知识修改，超越了以前的排名方法，并提供了一种有价值的工具，用于测量和修改LLMs中的大量事实知识。

    

    大型语言模型（LLMs）存储着从大量文本中获取的广泛的事实知识。为了有效地利用这些模型进行下游任务，有可靠的方法来衡量它们的知识是至关重要的。然而，现有的知识测量方法存在某些限制，尽管最近有不少努力，但它们不能提供准确的测量和修改LLMs中所需的洞察力。在这项工作中，我们采用基于信息理论的测量方法来提供一个框架来估计大型语言模型中包含的事实知识。具体而言，我们通过分析LLM在注入目标知识前后的预测概率分布来衡量知识，使用熵和KL-散度等度量标准。首先介绍我们的指标，我们通过一项合成实验，在准确性方面与以前的排名方法进行比较，超过了它们35％以上。然后，我们解释了这些指标如何用于知识修改，提出了一种选择性修改大型语言模型中的实际知识的方法。总的来说，我们的方法提供了一个宝贵的工具，用于测量和修改LLMs中的大量事实知识。

    Large Language Models (LLMs) store an extensive amount of factual knowledge obtained from vast collections of text. To effectively utilize these models for downstream tasks, it is crucial to have reliable methods for measuring their knowledge. However, existing approaches for knowledge measurement have certain limitations, and despite recent efforts, they fail to provide accurate measurements and the necessary insights for modifying the knowledge within LLMs. In this work, we employ information theory-based measurements to provide a framework estimating the factual knowledge contained within large language models. More specifically, we measure knowledge by analyzing the LLM's prediction probability distribution before and after instilling the target knowledge, employing metrics such as entropy and KL-divergence. Introducing our metrics, we first assess their accuracy in comparison to previous ranking-based methods, surpassing them by over $35\%$ in a synthetic experiment. Then, we expl
    
[^6]: ASR转录中实体分布建模的记录重复消除

    Record Deduplication for Entity Distribution Modeling in ASR Transcripts. (arXiv:2306.06246v1 [cs.CL])

    [http://arxiv.org/abs/2306.06246](http://arxiv.org/abs/2306.06246)

    本研究提出了一种通过记录重复消除来模拟客户真实请求的实体分布，从而解决ASR错误识别带来的实体分布复杂问题，并成功地应用于语境偏置中，显示出估计的单词错误率降低5%。

    

    语音数字助手必须跟上热门搜索查询。我们依赖于使用快速更新的实体集合进行语境偏置的语音识别模型，而不是频繁的模型重新训练，从而跟上趋势。这种方法存在若干挑战：（1）实体集必须频繁重构，（2）由于延迟和准确性权衡，实体集合的大小受限，（3）由于ASR错误识别，寻找真实实体分布是复杂的。我们通过使用实体解析领域的一种技术-记录重复消除，来解决这些挑战并定义实体集。通过建模客户从ASR输出中真实请求的实体分布，记录重复消除解决或消除了相同潜在实体的核心引用，包括错误识别。我们的方法成功检索到了95%的错误识别实体，并且在用于语境偏置时，显示出估计的5%的相对单词错误率降低。

    Voice digital assistants must keep up with trending search queries. We rely on a speech recognition model using contextual biasing with a rapidly updated set of entities, instead of frequent model retraining, to keep up with trends. There are several challenges with this approach: (1) the entity set must be frequently reconstructed, (2) the entity set is of limited size due to latency and accuracy trade-offs, and (3) finding the true entity distribution for biasing is complicated by ASR misrecognition. We address these challenges and define an entity set by modeling customers true requested entity distribution from ASR output in production using record deduplication, a technique from the field of entity resolution. Record deduplication resolves or deduplicates coreferences, including misrecognitions, of the same latent entity. Our method successfully retrieves 95% of misrecognized entities and when used for contextual biasing shows an estimated 5% relative word error rate reduction.
    
[^7]: 使用基础模型在极少监督下检测政策违规

    Using Foundation Models to Detect Policy Violations with Minimal Supervision. (arXiv:2306.06234v1 [cs.CL])

    [http://arxiv.org/abs/2306.06234](http://arxiv.org/abs/2306.06234)

    本文利用基础模型在极少监督下检测政策违规，创新性地将思维链提示引入政策违规任务，同时将硬提示与软提示相结合，可以高准确度地生成合理解释。

    

    基础模型，即预训练于大型文本语料库的大型神经网络，已经彻底改变了自然语言处理。它们可以直接指导，例如硬提示(例如(arXiv:2005.14165))，也可以使用极少的数据进行调整，这种技术被称为软提示，我们试图利用它们的能力来检测政策违规。我们的创新点是：我们确定了一种硬提示，将思维链提示适应于政策违规任务。这个提示生成政策违规分类以及提取式解释来证明分类的合理性。我们将硬提示与软提示调整相结合，使用非常少的监督来生成一个分类器，同时该分类器还可以产生解释。虽然监督只作用于分类，但我们发现修改后的解释与(调整后的)模型响应保持一致。在此过程中，我们确定了一些令人费解的方面。

    Foundation models, i.e. large neural networks pre-trained on large text corpora, have revolutionized NLP. They can be instructed directly (e.g. (arXiv:2005.14165)) - this is called hard prompting - and they can be tuned using very little data (e.g. (arXiv:2104.08691)) - this technique is called soft prompting. We seek to leverage their capabilities to detect policy violations. Our contributions are: We identify a hard prompt that adapts chain-of-thought prompting to policy violation tasks. This prompt produces policy violation classifications, along with extractive explanations that justify the classification. We compose the hard-prompts with soft prompt tuning to produce a classifier that attains high accuracy with very little supervision; the same classifier also produces explanations. Though the supervision only acts on the classifications, we find that the modified explanations remain consistent with the (tuned) model's response. Along the way, we identify several unintuitive aspec
    
[^8]: 探究自监督语音模型中的音素和音位信息：以送气现象为例

    Probing self-supervised speech models for phonetic and phonemic information: a case study in aspiration. (arXiv:2306.06232v1 [cs.CL])

    [http://arxiv.org/abs/2306.06232](http://arxiv.org/abs/2306.06232)

    本文研究了自监督语音模型中的音素和音位信息，并发现这些模型在早期层就能够很好地表示这些区别，并且这种表示在更深层的表示中得以保留，是由于模型在语音数据上的优化和高维度的体系结构的共同作用所致。

    

    近年来，无需文本的自监督语音模型的能力不断提高，但它们所编码的语言信息的本质还未得到彻底研究。本文评估了这些模型学习表示与人类基本表示区别之间的一致性，并集中研究了一组初始词停顿中具体表现的音素（低层）和音位（更抽象）对比。我们发现，在这些模型的体系结构的早期层中，出现了关于音素和音位区别的强大表示，并在更深层的主要成分表示中保留。我们的分析表明，这一成功的原因在于两方面：一些可归因于模型在语音数据上的优化，而另一些可归因于这些模型高维度的体系结构。我们的发现表明，经过语音训练的 HuBERT 得出了与抽象的音位区别相应的低噪声和低维度子空间。

    Textless self-supervised speech models have grown in capabilities in recent years, but the nature of the linguistic information they encode has not yet been thoroughly examined. We evaluate the extent to which these models' learned representations align with basic representational distinctions made by humans, focusing on a set of phonetic (low-level) and phonemic (more abstract) contrasts instantiated in word-initial stops. We find that robust representations of both phonetic and phonemic distinctions emerge in early layers of these models' architectures, and are preserved in the principal components of deeper layer representations. Our analyses suggest two sources for this success: some can only be explained by the optimization of the models on speech data, while some can be attributed to these models' high-dimensional architectures. Our findings show that speech-trained HuBERT derives a low-noise and low-dimensional subspace corresponding to abstract phonological distinctions.
    
[^9]: 规范化机器翻译评估

    Conformalizing Machine Translation Evaluation. (arXiv:2306.06221v1 [cs.CL])

    [http://arxiv.org/abs/2306.06221](http://arxiv.org/abs/2306.06221)

    本文提出了一种无分布方法——规范化预测，用于机器翻译评估置信区间的计算，并证明其具有理论保证覆盖率，可以纠正其他方法的误差。应用条件规范化预测技术，获得平等覆盖的校准子集。

    

    最近提出了多种不确定性估计方法用于机器翻译评估。虽然这些方法可以提供一个有用的指示，来判断何时不能相信模型预测，但本文表明，这些方法大部分倾向于低估模型的不确定性，结果往往产生具有误导性的置信区间，而这些置信区间未能覆盖真实值。我们提出用规范化预测作为替代方法，这是一种无分布方法，用于获得具有理论保证覆盖率的置信区间。首先，我们证明分裂规范化预测可以“修正”之前方法的置信区间，以产生所需的覆盖率。然后，我们突出显示了估计置信区间的偏差，无论是在翻译语言对方面还是在翻译质量方面。我们应用条件规范化预测技术，为每个数据子组获得校准子集，从而实现平等覆盖。

    Several uncertainty estimation methods have been recently proposed for machine translation evaluation. While these methods can provide a useful indication of when not to trust model predictions, we show in this paper that the majority of them tend to underestimate model uncertainty, and as a result they often produce misleading confidence intervals that do not cover the ground truth. We propose as an alternative the use of conformal prediction, a distribution-free method to obtain confidence intervals with a theoretically established guarantee on coverage. First, we demonstrate that split conformal prediction can ``correct'' the confidence intervals of previous methods to yield a desired coverage level. Then, we highlight biases in estimated confidence intervals, both in terms of the translation language pairs and the quality of translations. We apply conditional conformal prediction techniques to obtain calibration subsets for each data subgroup, leading to equalized coverage.
    
[^10]: 多语言BERT模型的形态句法探测

    Morphosyntactic probing of multilingual BERT models. (arXiv:2306.06205v1 [cs.CL])

    [http://arxiv.org/abs/2306.06205](http://arxiv.org/abs/2306.06205)

    该论文介绍了一个广泛的数据集，用于在42种语言中探测语言模型的形态句法信息。研究发现，预训练的Transformer模型（mBERT和XLM-RoBERTa）在这些任务中表现出色，而前面的上下文比后面的上下文包含更多与预测相关的信息。

    

    我们介绍了一个广泛的数据集，用于对语言模型中形态信息进行多语言探测（来自10个族群的42种语言中的247项任务），每个任务包括一个带有目标单词和形态标签的句子作为期望的标签，来自Universal Dependencies树库。我们发现预训练的Transformer模型（mBERT和XLM-RoBERTa）学习的特征在这些任务中具有强大的表现。然后，我们应用了两种方法来定位每个探测任务中的决策信息所在的位置。第一种是一种新的扰动方法，可以遮蔽上下文的各个部分；第二种是Shapley值的经典方法。最引人注目的发现是，前面的上下文比后面的上下文包含更多与预测相关的信息。

    We introduce an extensive dataset for multilingual probing of morphological information in language models (247 tasks across 42 languages from 10 families), each consisting of a sentence with a target word and a morphological tag as the desired label, derived from the Universal Dependencies treebanks. We find that pre-trained Transformer models (mBERT and XLM-RoBERTa) learn features that attain strong performance across these tasks. We then apply two methods to locate, for each probing task, where the disambiguating information resides in the input. The first is a new perturbation method that masks various parts of context; the second is the classical method of Shapley values. The most intriguing finding that emerges is a strong tendency for the preceding context to hold more information relevant to the prediction than the following context.
    
[^11]: 可靠性检查：对GPT-3在敏感话题和提示措辞方面的反应分析

    Reliability Check: An Analysis of GPT-3's Response to Sensitive Topics and Prompt Wording. (arXiv:2306.06199v1 [cs.CL])

    [http://arxiv.org/abs/2306.06199](http://arxiv.org/abs/2306.06199)

    本文分析了大型语言模型GPT-3对敏感话题和提示措辞的反应，发现其在阴谋论和刻板印象方面有正确的反应，但在误解和争议方面存在错误，并具有不可靠性。

    

    大型语言模型已成为主流技术，具有多种用途和出色的性能。尽管有无数的应用，但LLMs仍然不是可靠的。通过微调、提示和人类反馈的强化学习等方法，正在进行大量工作来提高这些模型的事实准确性、一致性和道德标准，但缺乏对这些模型对不同语句类别的反应或在简单提示变化下可能存在的漏洞的系统分析。在本研究中，我们分析了什么会让GPT-3困惑：模型如何响应某些敏感话题以及提示措辞对模型响应的影响。我们发现，GPT-3正确地反对明显的阴谋论和刻板印象，但在普遍的误解和争议中犯了错误。模型响应在提示和设置上不一致，突显出GPT-3的不可靠性。

    Large language models (LLMs) have become mainstream technology with their versatile use cases and impressive performance. Despite the countless out-of-the-box applications, LLMs are still not reliable. A lot of work is being done to improve the factual accuracy, consistency, and ethical standards of these models through fine-tuning, prompting, and Reinforcement Learning with Human Feedback (RLHF), but no systematic analysis of the responses of these models to different categories of statements, or on their potential vulnerabilities to simple prompting changes is available. In this work, we analyze what confuses GPT-3: how the model responds to certain sensitive topics and what effects the prompt wording has on the model response. We find that GPT-3 correctly disagrees with obvious Conspiracies and Stereotypes but makes mistakes with common Misconceptions and Controversies. The model responses are inconsistent across prompts and settings, highlighting GPT-3's unreliability. Dataset and 
    
[^12]: 使用文档级元数据的领域特定快速预训练技术$FPDM$

    $FPDM$: Domain-Specific Fast Pre-training Technique using Document-Level Metadata. (arXiv:2306.06190v1 [cs.CL])

    [http://arxiv.org/abs/2306.06190](http://arxiv.org/abs/2306.06190)

    本文提出了$FPDM$，使用文档元数据和领域特定分类作为监督信号，对领域特定语料库进行transformer编码器的预训练。$FPDM$通过句子级别的输入预训练开放领域的编码器，在微调时使用词汇级别的输入，性能优于其他基于transformer的模型。

    

    在各种领域的预训练已显示出在开放领域和领域特定下游任务上具有良好的结果。然而，最先进的transformers需要大量的预训练数据和计算资源。在本文中，我们提出了$FPDM$（Fast Pre-training Technique using Document Level Metadata），这是一个新颖、计算效率高的框架，利用文档元数据和领域特定的分类作为监督信号，对领域特定语料库进行transformer编码器的预训练。最主要的创新在于，在领域特定的预训练过程中，使用句子级别的嵌入作为输入，持续对开放领域的编码器进行预训练（以适应长文档），但在对该编码器进行微调时，则使用词汇级别嵌入作为输入。实验表明，$FPDM$在客户支持、科学和法律等领域的字符级F1分数和其他自动化指标方面优于几种基于transformer的基准，且在下游任务微调后性能下降可以忽略不计。

    Pre-training Transformers has shown promising results on open-domain and domain-specific downstream tasks. However, state-of-the-art Transformers require an unreasonably large amount of pre-training data and compute. In this paper, we propose $FPDM$ (Fast Pre-training Technique using Document Level Metadata), a novel, compute-efficient framework that utilizes Document metadata and Domain-Specific Taxonomy as supervision signals to pre-train transformer encoder on a domain-specific corpus. The main innovation is that during domain-specific pretraining, an open-domain encoder is continually pre-trained using sentence-level embeddings as inputs (to accommodate long documents), however, fine-tuning is done with token-level embeddings as inputs to this encoder. We show that $FPDM$ outperforms several transformer-based baselines in terms of character-level F1 scores and other automated metrics in the Customer Support, Scientific, and Legal Domains, and shows a negligible drop in performance 
    
[^13]: SentiGOLD：一个大型孟加拉语多领域情感分析黄金标准数据集及其评估

    SentiGOLD: A Large Bangla Gold Standard Multi-Domain Sentiment Analysis Dataset and its Evaluation. (arXiv:2306.06147v1 [cs.CL])

    [http://arxiv.org/abs/2306.06147](http://arxiv.org/abs/2306.06147)

    本文介绍了SentiGOLD，一个孟加拉语多领域情感分析数据集。该数据集由70,000个来自不同来源的样本组成，遵守了政府和语言学委员会商定的语言约定，包括了30个领域和5个情感类别。在具有鲁棒性的注释方案下，该数据集的互评一致性表现出色，可用于建立孟加拉语情感分析模型。

    

    本研究介绍了SentiGOLD，一个孟加拉语多领域情感分析数据集。它由来自不同来源的70,000个样本组成，并由一组性别平衡的语言学家进行注释。SentiGOLD遵守孟加拉国政府和孟加拉语言学委员会商定的既定语言约定。与英语和其他语言不同，由于缺乏国家语言学框架，孟加拉语缺乏标准的情感分析数据集。该数据集包括在线视频评论、社交媒体帖子、博客、新闻和其他来源的数据，并严格维护领域和类别分布。它涵盖了30个领域（如政治、娱乐、体育），包括5个情感类别（强烈的负面、弱的负面、中性和强烈的正面）。由国家语言学委员会批准的注释方案确保具有鲁棒的互评一致性（IAA），菲利斯kappa得分为0.88。内部和跨数据集的评估表明...

    This study introduces SentiGOLD, a Bangla multi-domain sentiment analysis dataset. Comprising 70,000 samples, it was created from diverse sources and annotated by a gender-balanced team of linguists. SentiGOLD adheres to established linguistic conventions agreed upon by the Government of Bangladesh and a Bangla linguistics committee. Unlike English and other languages, Bangla lacks standard sentiment analysis datasets due to the absence of a national linguistics framework. The dataset incorporates data from online video comments, social media posts, blogs, news, and other sources while maintaining domain and class distribution rigorously. It spans 30 domains (e.g., politics, entertainment, sports) and includes 5 sentiment classes (strongly negative, weakly negative, neutral, and strongly positive). The annotation scheme, approved by the national linguistics committee, ensures a robust Inter Annotator Agreement (IAA) with a Fleiss' kappa score of 0.88. Intra- and cross-dataset evaluatio
    
[^14]: 基于可解释触发器和关系名称相关联的无监督对话关系抽取

    Zero-Shot Dialogue Relation Extraction by Relating Explainable Triggers and Relation Names. (arXiv:2306.06141v1 [cs.CL])

    [http://arxiv.org/abs/2306.06141](http://arxiv.org/abs/2306.06141)

    本文提出了一种无监督对话关系抽取方法，它能够捕捉触发器并将其与以前未见过的关系名称相关联，能够对推断先前未见过的关系类型极具帮助。

    

    开发对话关系抽取（DRE）系统通常需要大量标记的数据，而这些数据的标注可能会很昂贵且耗时。为了提高可扩展性并支持多样化的、未见过的关系抽取，本文提出了一种利用捕捉触发器并将它们与以前未见过的关系名称相关联的方法。具体地，我们介绍了一种模型，通过利用捕捉触发器的能力实现无监督对话关系抽取。我们在DialogRE基准数据集上的实验表明，所提出的模型在已知和未知的关系方面都取得了显著的改进。值得注意的是，这是首次尝试使用捕捉触发器实现无监督对话关系抽取，我们的结果表明，这种方法对于推断先前未见过的关系类型非常有效。总的来说，我们的研究结果突显了这种方法提高DRE系统的可扩展性和实用性的潜力。

    Developing dialogue relation extraction (DRE) systems often requires a large amount of labeled data, which can be costly and time-consuming to annotate. In order to improve scalability and support diverse, unseen relation extraction, this paper proposes a method for leveraging the ability to capture triggers and relate them to previously unseen relation names. Specifically, we introduce a model that enables zero-shot dialogue relation extraction by utilizing trigger-capturing capabilities. Our experiments on a benchmark DialogRE dataset demonstrate that the proposed model achieves significant improvements for both seen and unseen relations. Notably, this is the first attempt at zero-shot dialogue relation extraction using trigger-capturing capabilities, and our results suggest that this approach is effective for inferring previously unseen relation types. Overall, our findings highlight the potential for this method to enhance the scalability and practicality of DRE systems.
    
[^15]: INSTRUCTEVAL：面向指导调整的大型语言模型的整体评估

    INSTRUCTEVAL: Towards Holistic Evaluation of Instruction-Tuned Large Language Models. (arXiv:2306.04757v1 [cs.CL])

    [http://arxiv.org/abs/2306.04757](http://arxiv.org/abs/2306.04757)

    INSTRUCTEVAL是一个专注于指导调整的大型语言模型评估的综合套件，它采取了全面的方法来评估模型的性能，包括解决问题、写作能力和与人类价值观的一致性等特征。

    

    指导调整的大型语言模型已经从根本上改变了自然语言处理，已经在诸如对话代理等应用中显示出了巨大的潜力。这些模型，如GPT-4，不仅能够掌握语言，而且可以解决数学、编码、医学和法律等领域的复杂任务。尽管它们具有卓越的能力，但由于许多模型的黑盒性质和缺乏全面的评估研究，对它们的全部潜力仍然缺乏全面的理解。为了解决这些挑战，我们提出了INSTRUCTEVAL，一个更全面的评估套件，专门针对指导调整的大型语言模型。与以往的作品不同，我们的评估包括对模型基于解决问题、写作能力和与人类价值观的一致性的严格评估。我们采取了全面的方法来分析影响模型性能的各种因素，包括预训练基础、指导调整数据和训练。

    Instruction-tuned large language models have revolutionized natural language processing and have shown great potential in applications such as conversational agents. These models, such as GPT-4, can not only master language but also solve complex tasks in areas like mathematics, coding, medicine, and law. Despite their impressive capabilities, there is still a lack of comprehensive understanding regarding their full potential, primarily due to the black-box nature of many models and the absence of holistic evaluation studies. To address these challenges, we present INSTRUCTEVAL, a more comprehensive evaluation suite designed specifically for instruction-tuned large language models. Unlike previous works, our evaluation involves a rigorous assessment of models based on problem-solving, writing ability, and alignment to human values. We take a holistic approach to analyze various factors affecting model performance, including the pretraining foundation, instruction-tuning data, and train
    
[^16]: 用于基于检索的对话系统的上下文掩码自编码器

    ConTextual Masked Auto-Encoder for Retrieval-based Dialogue Systems. (arXiv:2306.04357v1 [cs.CL])

    [http://arxiv.org/abs/2306.04357](http://arxiv.org/abs/2306.04357)

    本研究提出了一种针对对话响应选择的后训练技术Dial-MAE，利用生成方法更好地压缩对话语义至密集向量，并提高对话响应选择准确性。

    

    对话响应选择旨在根据给定的用户和系统话语历史记录从几个候选响应中选择适当的响应。最近的研究通过后训练大多依赖于单纯的掩码语言建模方法来提高对话响应选择的准确性。但是，最近开发的生成方法在IR社区展示了有希望的文本表示能力，这可能会导致更好的对话语义建模。因此，在本文中，我们提出 Dial-MAE（对话上下文掩码自编码器），这是一种简单而有效的针对对话响应选择的后训练技术。 Dial-MAE使用一个不对称的编码器-解码器架构，学习将对话的语义更好地压缩到密集向量中。 Dial-MAE的过程包括由深度编码器创建带有掩码对话上下文的对话嵌入，然后是浅解码器，该解码器使用此嵌入以及上下文向量来生成响应。

    Dialogue response selection aims to select an appropriate response from several candidates based on a given user and system utterance history. Recent studies have been improving the accuracy of dialogue response selection through post-training, mostly relying on naive masked language modeling methods. However, the recently developed generative methods have shown promising text representation capabilities in IR community, which could potentially lead to better dialogue semantics modeling. Thus, in this paper, we propose Dial-MAE (Dialogue Contextual Masking Auto-encoder), a straightforward yet effective post-training technique tailored for dialogue response selection. Dial-MAE uses an asymmetric encoder-decoder architecture that learns to better compress the semantics of the dialogue into dialogue-dense vectors. The process of Dial-MAE involves a deep encoder creating a dialogue embedding with the masked dialogue context, followed by a shallow decoder that uses this embedding along with
    
[^17]: 重塑话语语篇理解以促进对话分离

    Revisiting Conversation Discourse for Dialogue Disentanglement. (arXiv:2306.03975v1 [cs.CL])

    [http://arxiv.org/abs/2306.03975](http://arxiv.org/abs/2306.03975)

    本文提出了一种利用对话话语特征增强对话分离的方法，通过构建异构图表示和引入增强分离目标，以更好地建模对话语境和利用内在的话语结构信息。该方法在不同的对话分离基准数据集上表现出优异的性能。

    

    对话分离旨在将时间顺序排列的话语分隔成几个独立的会话。对话话语本质上是由底层语篇组织和描述的，因此对话分离需要完全理解和利用内在的话语属性。本文提出利用对话话语特征全面增强对话分离。在特征编码阶段，我们构建异构图表示来模拟各种对话特定的话语结构特征，包括静态的讲话者角色结构（即讲话者话语和讲话者提及结构）和动态的上下文结构（即话语距离和部分回复结构）。我们然后开发了一个结构感知框架，以集成丰富的结构特征，更好地建模对话语境。其次，在模型翻译阶段，我们进一步引入了一种新的增强分离目标，以利用内在的话语结构信息来进行分离过程。在不同的对话分离基准数据集上的实验结果表明，我们的方法在性能方面优于现有的最先进方法。

    Dialogue disentanglement aims to detach the chronologically ordered utterances into several independent sessions. Conversation utterances are essentially organized and described by the underlying discourse, and thus dialogue disentanglement requires the full understanding and harnessing of the intrinsic discourse attribute. In this paper, we propose enhancing dialogue disentanglement by taking full advantage of the dialogue discourse characteristics. First of all, \textbf{in feature encoding stage}, we construct the heterogeneous graph representations to model the various dialogue-specific discourse structural features, including the static speaker-role structures (i.e., speaker-utterance and speaker-mentioning structure) and the dynamic contextual structures (i.e., the utterance-distance and partial-replying structure). We then develop a structure-aware framework to integrate the rich structural features for better modeling the conversational semantic context. Second, \textbf{in model
    
[^18]: TKDP: 三重知识增强的深度提示调整在少样本命名实体识别中的应用

    TKDP: Threefold Knowledge-enriched Deep Prompt Tuning for Few-shot Named Entity Recognition. (arXiv:2306.03974v1 [cs.CL])

    [http://arxiv.org/abs/2306.03974](http://arxiv.org/abs/2306.03974)

    本文提出了一个名为 TKDP 的方法，在深度提示调整中通过整合三种不同来源的知识来增强少样本命名实体识别的性能。 在五个基准数据集上，相对于原始的深度提示方法提高了最多 11.53% 的 F1，并且明显优于 8 种表现强劲的 few-shot NER 方法。

    

    少样本命名实体识别通过有限注释示例来识别命名实体，因此有效地转移内部或外部资源成为少样本命名实体识别的关键。本文研究在深度提示调整中整合丰富的知识以实现更强的少样本命名实体识别。我们提出将深度提示调整框架与三重知识（即 TKDP），包括内部的 1）上下文知识和外部的 2）标签知识和 3）义原知识相结合。TKDP 对三个特征源进行编码，并将它们整合到软提示嵌入中，进而注入到现有的预训练语言模型中以促进预测。在五个基准数据集上，我们的知识增强模型相对于原始的深度提示方法提高了最多 11.53% 的 F1，并且明显优于 8 种表现强劲的 few-shot NER 方法。

    Few-shot named entity recognition (NER) exploits limited annotated instances to identify named mentions. Effectively transferring the internal or external resources thus becomes the key to few-shot NER. While the existing prompt tuning methods have shown remarkable few-shot performances, they still fail to make full use of knowledge. In this work, we investigate the integration of rich knowledge to prompt tuning for stronger few-shot NER. We propose incorporating the deep prompt tuning framework with threefold knowledge (namely TKDP), including the internal 1) context knowledge and the external 2) label knowledge & 3) sememe knowledge. TKDP encodes the three feature sources and incorporates them into the soft prompt embeddings, which are further injected into an existing pre-trained language model to facilitate predictions. On five benchmark datasets, our knowledge-enriched model boosts by at most 11.53% F1 over the raw deep prompt method, and significantly outperforms 8 strong-perform
    
[^19]: ECQED：对话中的情感-原因四元组抽取

    ECQED: Emotion-Cause Quadruple Extraction in Dialogs. (arXiv:2306.03969v1 [cs.CL])

    [http://arxiv.org/abs/2306.03969](http://arxiv.org/abs/2306.03969)

    本文提出了一个新的对话中的情感-原因四元组抽取任务(ECQED)，通过引入对话上下文和细粒度的情感和原因检测，有效地提高了任务的性能。

    

    现有的情感-原因对抽取(ECPE)任务遗憾地忽略了情感类型和原因类型的提取，而这些细粒度的元信息在实际应用中可能非常有用，例如聊天机器人和共情对话生成。此外，当前的ECPE仅限于单个文本片段的场景，而忽略了应该具有更现实价值的对话级别的研究。在本文中，我们通过更广泛的定义和场景扩展了ECPE任务，提出了一个新的任务：对话中的情感-原因四元组抽取(ECQED)，需要检测情感-原因话语对和情感和原因类型。我们提出了基于结构和语义异构图以及平行标记方案的ECQED模型，这提高了有效地结合对话上下文结构的能力，同时解决了令人困扰的重叠四元组问题。通过实验，我们表明，引入细粒度情感和原因检测并考虑对话上下文对于实现ECQED任务的更好效果都至关重要。

    The existing emotion-cause pair extraction (ECPE) task, unfortunately, ignores extracting the emotion type and cause type, while these fine-grained meta-information can be practically useful in real-world applications, i.e., chat robots and empathic dialog generation. Also the current ECPE is limited to the scenario of single text piece, while neglecting the studies at dialog level that should have more realistic values. In this paper, we extend the ECPE task with a broader definition and scenario, presenting a new task, Emotion-Cause Quadruple Extraction in Dialogs (ECQED), which requires detecting emotion-cause utterance pairs and emotion and cause types. We present an ECQED model based on a structural and semantic heterogeneous graph as well as a parallel grid tagging scheme, which advances in effectively incorporating the dialog context structure, meanwhile solving the challenging overlapped quadruple issue. Via experiments we show that introducing the fine-grained emotion and caus
    
[^20]: 评估自然语言推理在标记数据有限的语言中对仇恨言论检测的有效性

    Evaluating the Effectiveness of Natural Language Inference for Hate Speech Detection in Languages with Limited Labeled Data. (arXiv:2306.03722v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2306.03722](http://arxiv.org/abs/2306.03722)

    本文对自然语言推理（NLI）模型在标记数据有限语言中进行仇恨言论检测的有效性进行了研究。结果表明，NLI模型可以显著提高仇恨言论检测的性能，但在英语数据匹配测试领域时仅有自定义的NLI配方能够胜过中间英语微调。

    

    大多数对仇恨言论检测的研究都关注于英语，因为有大量的标记训练数据可供使用。但是，为了将仇恨言论检测扩展到更多的语言，需要采用需要最少训练数据的方法。本文研究了自然语言推理（NLI）模型在仅有少量标记数据的目标语言中是否能对仇恨言论检测的性能有所提升。我们在五种语言上进行了评估，结果显示NLI微调在目标语言中对于直接微调相比有显著的性能改进。然而，之前提出的在英语数据上进行中间微调的方法的有效性很难匹配。只有在English训练数据与测试领域不匹配时，我们自定义的NLI配方才能胜过中间英语微调。基于我们的广泛实验，我们提出了一系列有效利用NLI模型在标记数据有限的语言中进行仇恨言论检测的建议。

    Most research on hate speech detection has focused on English where a sizeable amount of labeled training data is available. However, to expand hate speech detection into more languages, approaches that require minimal training data are needed. In this paper, we test whether natural language inference (NLI) models which perform well in zero- and few-shot settings can benefit hate speech detection performance in scenarios where only a limited amount of labeled data is available in the target language. Our evaluation on five languages demonstrates large performance improvements of NLI fine-tuning over direct fine-tuning in the target language. However, the effectiveness of previous work that proposed intermediate fine-tuning on English data is hard to match. Only in settings where the English training data does not match the test domain, can our customised NLI-formulation outperform intermediate fine-tuning on English. Based on our extensive experiments, we propose a set of recommendatio
    
[^21]: 应用标准促进大型语言模型上下游伦理

    Applying Standards to Advance Upstream & Downstream Ethics in Large Language Models. (arXiv:2306.03503v1 [cs.CY])

    [http://arxiv.org/abs/2306.03503](http://arxiv.org/abs/2306.03503)

    本文探讨如何为AI生成的内容制定安全保障，分析LLMs的内容生成机制，确定了四个关键领域，提出了新的分发和销售LLM生成内容的企业的标准。

    

    本文探讨AI所有者如何借鉴其他内容创作行业的行为准则和伦理标准，为AI生成的内容制定安全保障。它深入研究了大型语言模型（LLMs）的伦理意识现状。通过分析LLMs的内容生成机制，确定了四个关键领域（上下游和用户提示/回答），在这些领域可以有效地应用保障措施。随后，对这四个领域进行了比较分析，包括在成本、有效性和与行业惯例的一致性方面评估现有的伦理保障措施。本文的主要观点是，现有的与IT相关的伦理准则虽然适用于传统的IT工程领域，但不足以应对基于LLMs内容生成所带来的挑战。我们借鉴新闻业内已有的实践，为分发和销售LLM生成内容的企业提出了潜在的标准。

    This paper explores how AI-owners can develop safeguards for AI-generated content by drawing from established codes of conduct and ethical standards in other content-creation industries. It delves into the current state of ethical awareness on Large Language Models (LLMs). By dissecting the mechanism of content generation by LLMs, four key areas (upstream/downstream and at user prompt/answer), where safeguards could be effectively applied, are identified. A comparative analysis of these four areas follows and includes an evaluation of the existing ethical safeguards in terms of cost, effectiveness, and alignment with established industry practices. The paper's key argument is that existing IT-related ethical codes, while adequate for traditional IT engineering, are inadequate for the challenges posed by LLM-based content generation. Drawing from established practices within journalism, we propose potential standards for businesses involved in distributing and selling LLM-generated cont
    
[^22]: Video-LLaMA：用于视频理解的指令调整的语音-视觉语言模型

    Video-LLaMA: An Instruction-tuned Audio-Visual Language Model for Video Understanding. (arXiv:2306.02858v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2306.02858](http://arxiv.org/abs/2306.02858)

    Video-LLaMA是一个多模态框架，利用已有的预训练模型，解决了视频中的视觉和听觉的理解问题，其中Video Q-former和Audio Q-former用于处理视频中的视觉与时间变化和音频信号的问题。

    

    我们提出了一个多模态框架Video-LLaMA，赋予大型语言模型（LLMs）理解视频中的视觉和听觉内容的能力。Video-LLaMA从已经预训练好的视觉和音频编码器以及已经冻结的LLMs进行跨模态训练。相比于之前专注于静态图像理解的视觉-LLMs，如MiniGPT-4和LLaVA，Video-LLaMA主要解决两个视频理解方面的挑战：（1）捕捉视觉场景中的时间变化，（2）集成音频视觉信号。为了克服第一个挑战，我们提出了一个Video Q-former，将预训练的图像编码器组装到我们的视频编码器中，并引入一个视频到文本生成任务来学习视频-语言对应关系。为了解决第二个挑战，我们利用ImageBind，一个将多种模态对齐的通用嵌入模型，作为预训练的音频编码器，并在ImageBind之上引入一个Audio Q-former，学习合理的听觉查询嵌入。

    We present Video-LLaMA, a multi-modal framework that empowers Large Language Models (LLMs) with the capability of understanding both visual and auditory content in the video. Video-LLaMA bootstraps cross-modal training from the frozen pre-trained visual & audio encoders and the frozen LLMs. Unlike previous vision-LLMs that focus on static image comprehensions such as MiniGPT-4 and LLaVA, Video-LLaMA mainly tackles two challenges in video understanding: (1) capturing the temporal changes in visual scenes, (2) integrating audio-visual signals. To counter the first challenge, we propose a Video Q-former to assemble the pre-trained image encoder into our video encoder and introduce a video-to-text generation task to learn video-language correspondence. For the second challenge, we leverage ImageBind, a universal embedding model aligning multiple modalities as the pre-trained audio encoder, and introduce an Audio Q-former on top of ImageBind to learn reasonable auditory query embeddings for
    
[^23]: LLM-Blender: 利用成对排名和生成融合集成大型语言模型

    LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion. (arXiv:2306.02561v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2306.02561](http://arxiv.org/abs/2306.02561)

    本论文提出了LLM-Blender，它是一个集成框架，旨在利用不同的开源大型语言模型的优秀特性，实现始终如一的卓越性能。PairRanker和GenFuser是该框架的两个模块，PairRanker使用成对比较方法来区分候选输出，并且GenFuser旨在合并排名最高的候选者，以生成改进的输出。

    

    本论文提出了LLM-Blender，一个集成框架，旨在通过利用多个开源大型语言模型（LLMs）的不同优势来达到始终如一的卓越性能。我们的框架由两个模块组成：PairRanker和GenFuser，以应对不同示例的最优LLMs可以显着变化的观察。PairRanker使用专门的成对比较方法来区分候选输出之间的微小差异。它联合编码输入文本和一对候选者，使用交叉注意编码器来确定优越者。我们的结果表明，PairRanker与ChatGPT的排名相关性最高。然后，GenFuser旨在合并排名最高的候选者，通过利用它们的优势和减少它们的弱点来生成改进的输出。为了促进大规模评估，我们介绍了一个基准数据集MixInstruct，它是多个指令数据集的混合，具有oracle p。

    We present LLM-Blender, an ensembling framework designed to attain consistently superior performance by leveraging the diverse strengths of multiple open-source large language models (LLMs). Our framework consists of two modules: PairRanker and GenFuser, addressing the observation that optimal LLMs for different examples can significantly vary. PairRanker employs a specialized pairwise comparison method to distinguish subtle differences between candidate outputs. It jointly encodes the input text and a pair of candidates, using cross-attention encoders to determine the superior one. Our results demonstrate that PairRanker exhibits the highest correlation with ChatGPT-based ranking. Then, GenFuser aims to merge the top-ranked candidates, generating an improved output by capitalizing on their strengths and mitigating their weaknesses. To facilitate large-scale evaluation, we introduce a benchmark dataset, MixInstruct, which is a mixture of multiple instruction datasets featuring oracle p
    
[^24]: 高效象征交流编码的演变

    Evolution of Efficient Symbolic Communication Codes. (arXiv:2306.02383v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2306.02383](http://arxiv.org/abs/2306.02383)

    本文探讨了通过反熵、压缩因子和跨分裂F1分数为目标的交流代码演变产物，发现语言结构形成可以通过这些度量来驱动。

    

    本文探讨人类自然语言结构如何被看作是人际交流代码的演变产物，旨在最大化文化无关和跨语言度量标准，如反熵、压缩因子和跨分裂F1分数。探索是作为更大的无监督语言学习努力的一部分完成的，试图在基于“基本语言结构”的超参数空间中执行元学习，通过最大化上述指标来实现。本文提出了针对俄语、中文和英语的跨语言词级分词标记化研究以及针对英语的子词分割或形态分析研究的初步结果。发现语言结构形成词级分割或标记化可以通过所有这些度量来驱动，反熵对英语和俄语更相关，而压缩因子对中文更具特定性。

    The paper explores how the human natural language structure can be seen as a product of evolution of inter-personal communication code, targeting maximisation of such culture-agnostic and cross-lingual metrics such as anti-entropy, compression factor and cross-split F1 score. The exploration is done as part of a larger unsupervised language learning effort, the attempt is made to perform meta-learning in a space of hyper-parameters maximising F1 score based on the "ground truth" language structure, by means of maximising the metrics mentioned above. The paper presents preliminary results of cross-lingual word-level segmentation tokenisation study for Russian, Chinese and English as well as subword segmentation or morphological parsing study for English. It is found that language structure form the word-level segmentation or tokenisation can be found as driven by all of these metrics, anti-entropy being more relevant to English and Russian while compression factor more specific for Chin
    
[^25]: 预训练视觉语言模型适应方法的鲁棒性基准测试研究

    Benchmarking Robustness of Adaptation Methods on Pre-trained Vision-Language Models. (arXiv:2306.02080v2 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2306.02080](http://arxiv.org/abs/2306.02080)

    研究针对预训练视觉语言模型的11种适应方法在不同污染情况下的鲁棒性，发现适应方法对文本污染更敏感，单独使用小型文本适配器比共享适配器更鲁棒，可获得可比较的干净性能。

    

    提升预训练视觉语言模型在特定领域表现的各种适应方法，如 LoRA、prompts 和 adapters 等已被提出。然而，这些适应方法对于分布位移的鲁棒性尚未得到研究。本研究评估了11种广泛使用的适应方法在4个视觉语言数据集上的鲁棒性，考察了可用适应示例和适应过程中可训练参数大小的影响。具体地，引入了7个基准数据集，包括96种视觉和87种文本污损，以研究不同适应方法的鲁棒性。我们的分析揭示了：1）适应方法对文本污染比视觉污染更敏感。2) 全量微调并不总能提供最高的鲁棒性；相反，适配器可以实现更好的鲁棒性，并具有可比较的干净性能。3）与预期相反，我们的发现表明，单独使用小型文本适配器通常比在视觉和语言空间中共享适配器更鲁棒。

    Various adaptation methods, such as LoRA, prompts, and adapters, have been proposed to enhance the performance of pre-trained vision-language models in specific domains. The robustness of these adaptation methods against distribution shifts have not been studied. In this study, we assess the robustness of 11 widely-used adaptation methods across 4 vision-language datasets under multimodal corruptions. Concretely, we introduce 7 benchmark datasets, including 96 visual and 87 textual corruptions, to investigate the robustness of different adaptation methods, the impact of available adaptation examples, and the influence of trainable parameter size during adaptation. Our analysis reveals that: 1) Adaptation methods are more sensitive to text corruptions than visual corruptions. 2) Full fine-tuning does not consistently provide the highest robustness; instead, adapters can achieve better robustness with comparable clean performance. 3) Contrary to expectations, our findings indicate that i
    
[^26]: 使预训练模型具有可逆性：从参数到内存高效的微调

    Make Your Pre-trained Model Reversible: From Parameter to Memory Efficient Fine-Tuning. (arXiv:2306.00477v1 [cs.CL])

    [http://arxiv.org/abs/2306.00477](http://arxiv.org/abs/2306.00477)

    本研究尝试实现在预训练语言模型中运用可逆模型实现高效的微调，并发现在初始化微调时保留PLM的起点非常重要。

    

    预训练语言模型（PLM）的参数高效微调已经成为一种非常成功的方法，只需训练少量参数而不会降低性能，并随着PLM越来越大而成为事实上的学习范式。然而，现有的PEFT方法不具备内存效率，因为它们仍需要存储大部分中间激活值以便计算梯度，类似于微调。一个减少激活内存的有效方法是应用可逆模型，这样中间激活值就无需缓存，可以重新计算。然而，将PLM修改为它的可逆变体并进行PEFT并不是一件容易的事，因为可逆模型具有与当前发布的PLM不同的体系结构。本文首先调查现有PEFT方法成功的关键因素，认识到在初始化PEFT时保留PLM的起点是至关重要的。

    Parameter-efficient fine-tuning (PEFT) of pre-trained language models (PLMs) has emerged as a highly successful approach, with training only a small number of parameters without sacrificing performance and becoming the de-facto learning paradigm with the increasing size of PLMs. However, existing PEFT methods are not memory-efficient, because they still require caching most of the intermediate activations for the gradient calculation, akin to fine-tuning. One effective way to reduce the activation memory is to apply a reversible model, so the intermediate activations are not necessary to be cached and can be recomputed. Nevertheless, modifying a PLM to its reversible variant with PEFT is not straightforward, since the reversible model has a distinct architecture from the currently released PLMs. In this paper, we first investigate what is a key factor for the success of existing PEFT methods, and realize that it's essential to preserve the PLM's starting point when initializing a PEFT 
    
[^27]: 针对可控文本生成的焦点前缀调整方法

    Focused Prefix Tuning for Controllable Text Generation. (arXiv:2306.00369v1 [cs.CL])

    [http://arxiv.org/abs/2306.00369](http://arxiv.org/abs/2306.00369)

    本文提出了针对可控文本生成的焦点前缀调整方法，实验结果表明在单属性控制任务中实现了更好的控制准确性和文本流畅度，在多属性控制任务中实现了与最先进方法相当的控制准确性，并保持了控制新属性而无需重新训练现有模型的灵活性。

    

    在可控文本生成数据集中，存在未标注属性，可能会为使用其进行训练的模型提供无关的学习信号，从而降低它们的性能。我们提出了焦点前缀调整（FPT）来缓解这个问题，并使控制能够专注于所需属性。实验结果表明，与基线模型相比，在单属性控制任务中，FPT可以实现更好的控制准确性和文本流畅度。在多属性控制任务中，FPT实现了与最先进方法相当的控制准确性，同时保持了控制新属性而无需重新训练现有模型的灵活性。

    In a controllable text generation dataset, there exist unannotated attributes that could provide irrelevant learning signals to models that use it for training and thus degrade their performance. We propose focused prefix tuning(FPT) to mitigate the problem and to enable the control to focus on the desired attribute. Experimental results show that FPT can achieve better control accuracy and text fluency than baseline models in single-attribute control tasks. In multi-attribute control tasks, FPT achieves comparable control accuracy with the state-of-the-art approach while keeping the flexibility to control new attributes without retraining existing models.
    
[^28]: 源代码模型的数据增强方法：一份综述

    Data Augmentation Approaches for Source Code Models: A Survey. (arXiv:2305.19915v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.19915](http://arxiv.org/abs/2305.19915)

    本文对源代码的数据增强技术进行了全面的调查和综述，介绍了它们的分类法、优化策略和性能结果，并讨论了未来方向和研究挑战。

    

    源代码在许多关键任务中的广泛应用促进了数据增强（DA）技术的发展，以增强训练数据并提高这些模型的各种能力（例如健壮性和可泛化性）。虽然已经提出并针对源代码模型进行了一系列DA方法的调整，但缺乏综合性的调查和审查以理解它们的有效性和含义。本文通过对源代码的数据增强进行全面而综合的调查，填补这一空白，我们系统地整理和概述现有文献，以提供该领域的全面概述。我们首先构建了适用于源代码模型的数据增强的分类法，然后讨论了著名的、方法上具有说明性的方法。接下来，我们强调了优化DA质量的一般策略和技术。随后，我们强调了在被广泛接受的基准测试中发挥作用的技术，并呈现了它们的性能结果。最后，我们讨论了DA用于源代码模型的潜在未来方向和开放研究挑战。

    The increasingly popular adoption of source code in many critical tasks motivates the development of data augmentation (DA) techniques to enhance training data and improve various capabilities (e.g., robustness and generalizability) of these models. Although a series of DA methods have been proposed and tailored for source code models, there lacks a comprehensive survey and examination to understand their effectiveness and implications. This paper fills this gap by conducting a comprehensive and integrative survey of data augmentation for source code, wherein we systematically compile and encapsulate existing literature to provide a comprehensive overview of the field. We start by constructing a taxonomy of DA for source code models model approaches, followed by a discussion on prominent, methodologically illustrative approaches. Next, we highlight the general strategies and techniques to optimize the DA quality. Subsequently, we underscore techniques that find utility in widely-accept
    
[^29]: DC CoMix TTS：一种与混合器协作的端到端表现力TTS，利用离散码实现改进的韵律建模

    DC CoMix TTS: An End-to-End Expressive TTS with Discrete Code Collaborated with Mixer. (arXiv:2305.19567v1 [cs.SD])

    [http://arxiv.org/abs/2305.19567](http://arxiv.org/abs/2305.19567)

    本文提出了一种基于离散码和混合器相协作的端到端表现力TTS，它采用新的输入表示和简单的架构来实现改进的韵律建模，证明了其有效性。

    

    尽管中性TTS取得了巨大的成功，但内容泄漏仍然是一个挑战。本文提出了一种新的输入表示和简单的架构来实现改进的韵律建模。受最近在TTS中使用离散码取得的成功启发，我们将离散码引入到参考编码器的输入中。具体来说，我们利用音频压缩模型中的向量量化器来利用它已经训练过的多样化的声学信息。此外，我们将修改后的MLP-Mixer应用到参考编码器中，使得架构更加轻盈。因此，我们以端到端的方式训练韵律转移TTS。我们通过主观和客观评估证明了我们方法的有效性。我们在实验中证明了，当离散码作为输入时，参考编码器可以学习到更好的与说话人无关的韵律。另外，即使输入参数更少，我们也可以获得可比较的结果。

    Despite the huge successes made in neutral TTS, content-leakage remains a challenge. In this paper, we propose a new input representation and simple architecture to achieve improved prosody modeling. Inspired by the recent success in the use of discrete code in TTS, we introduce discrete code to the input of the reference encoder. Specifically, we leverage the vector quantizer from the audio compression model to exploit the diverse acoustic information it has already been trained on. In addition, we apply the modified MLP-Mixer to the reference encoder, making the architecture lighter. As a result, we train the prosody transfer TTS in an end-to-end manner. We prove the effectiveness of our method through both subjective and objective evaluations. We demonstrate that the reference encoder learns better speaker-independent prosody when discrete code is utilized as input in the experiments. In addition, we obtain comparable results even when fewer parameters are inputted.
    
[^30]: 基于扩散式语言模型的细粒度文本风格转换

    Fine-grained Text Style Transfer with Diffusion-Based Language Models. (arXiv:2305.19512v1 [cs.CL])

    [http://arxiv.org/abs/2305.19512](http://arxiv.org/abs/2305.19512)

    本文提出了一种基于扩散式语言模型的细粒度文本风格转换方法，在不依赖外部信息的情况下取得了比之前利用预训练权重、嵌入和外部语法分析器更好的效果，表明扩散概率模型在文本生成领域具有广泛的应用前景。

    

    扩散式概率模型已经在可控制地生成高质量图像上显示出了巨大的成功，研究人员已经试图将这种可控性运用到文本生成领域。以前的扩散式语言模型研究表明，它们可以在不需要外部知识（如预训练权重）的情况下进行训练，并且仍然可以实现稳定的性能和可控性。 在本文中，我们在StylePTB数据集上训练了一个扩散式模型，这是细粒度文本风格转换的标准基准。与以前的工作评估任务相比，StylePTB中的任务需要对输出文本进行更加精细的控制，我们的模型能够在StylePTB上实现卓越的性能，包括个别和组合转换。此外，我们的模型在没有外部知识的情况下使用StylePTB的有限数据进行训练，其表现优于以前利用预训练权重、嵌入和外部语法分析器的工作，这可能表明扩散概率模型在文本生成领域具有巨大的潜力。

    Diffusion probabilistic models have shown great success in generating high-quality images controllably, and researchers have tried to utilize this controllability into text generation domain. Previous works on diffusion-based language models have shown that they can be trained without external knowledge (such as pre-trained weights) and still achieve stable performance and controllability. In this paper, we trained a diffusion-based model on StylePTB dataset, the standard benchmark for fine-grained text style transfers. The tasks in StylePTB requires much more refined control over the output text compared to tasks evaluated in previous works, and our model was able to achieve state-of-the-art performance on StylePTB on both individual and compositional transfers. Moreover, our model, trained on limited data from StylePTB without external knowledge, outperforms previous works that utilized pretrained weights, embeddings, and external grammar parsers, and this may indicate that diffusion
    
[^31]: infoVerse：一种用多维度元信息对数据集进行特征化的通用框架

    infoVerse: A Universal Framework for Dataset Characterization with Multidimensional Meta-information. (arXiv:2305.19344v1 [cs.CL])

    [http://arxiv.org/abs/2305.19344](http://arxiv.org/abs/2305.19344)

    本文介绍了一种通用的数据集特征化框架infoVerse，通过结合各种模型驱动的元信息提供了一个新的特征空间，能够有效地捕捉数据集的多维特征，有助于用户或模型确定哪些样本需要关注。

    

    NLP系统的成功往往依赖于大量高质量的数据集。然而，这些数据集中并非所有样本都同样有利于学习，因为有些可能是冗余或带有噪声。已经开发了几种基于模型驱动元信息（例如模型的置信度）的数据集特征化方法，但这些方法之间的关系和互补效应受到的关注较少。在本文中，我们介绍了infoVerse，这是一种通用的数据集特征化框架，它通过结合各种模型驱动的元信息提供了一个新的特征空间，有效地捕捉了数据集的多维特征。infoVerse揭示了数据集中在原始语义空间中不明显的独特区域，从而引导用户（或模型）确定哪些样本需要关注以进行探索、评估或注释。此外，我们还在infoVerse上提出了一种新的采样方法来选择一组数据点。

    The success of NLP systems often relies on the availability of large, high-quality datasets. However, not all samples in these datasets are equally valuable for learning, as some may be redundant or noisy. Several methods for characterizing datasets based on model-driven meta-information (e.g., model's confidence) have been developed, but the relationship and complementary effects of these methods have received less attention. In this paper, we introduce infoVerse, a universal framework for dataset characterization, which provides a new feature space that effectively captures multidimensional characteristics of datasets by incorporating various model-driven meta-information. infoVerse reveals distinctive regions of the dataset that are not apparent in the original semantic space, hence guiding users (or models) in identifying which samples to focus on for exploration, assessment, or annotation. Additionally, we propose a novel sampling method on infoVerse to select a set of data points
    
[^32]: 缓解上下文学习的标签偏差

    Mitigating Label Biases for In-context Learning. (arXiv:2305.19148v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.19148](http://arxiv.org/abs/2305.19148)

    本文针对上下文学习（ICL）中的三种标签偏差提出分类法，并提出一种简单的偏差校准方法，使用随机的领域词估算语言模型的标签偏差。

    

    上下文学习（ICL）的各种设计设置，如选择和顺序的上下文示例，可能使模型对某种特定预测偏见，而这种预测并不反映对任务的理解。虽然许多研究讨论了这些设计选择，但对它们进行分类和减缓其影响的系统调查很少。在本文中，我们为文本分类中上下文学习（ICL）中的三种标签偏差定义了一个分类法：香草标签偏差、上下文标签偏差和领域标签偏差（我们首次概念化和检测到）。我们的分析表明，先前的标签偏差校准方法不能解决所有三种偏差。特别是，领域标签偏差使LLM在许多任务上只能实现随机级别的性能，而不管上下文示例的选择如何。为了缓解这些偏差的影响，我们提出一个简单的偏差校准方法，使用随机的领域词估算语言模型的标签偏差。

    Various design settings for in-context learning (ICL), such as the choice and order of the in-context examples, can bias a model toward a particular prediction without being reflective of an understanding of the task. While many studies discuss these design choices, there have been few systematic investigations into categorizing them and mitigating their impact. In this work, we define a typology for three types of label biases in ICL for text classification: vanilla-label bias, context-label bias, and domain-label bias (which we conceptualize and detect for the first time).  Our analysis demonstrates that prior label bias calibration methods fall short of addressing all three types of biases. Specifically, domain-label bias restricts LLMs to random-level performance on many tasks regardless of the choice of in-context examples. To mitigate the effect of these biases, we propose a simple bias calibration method that estimates a language model's label bias using random in-domain words f
    
[^33]: W-procer: 基于加权原型对比学习的医学少样本命名实体识别

    W-procer: Weighted Prototypical Contrastive Learning for Medical Few-Shot Named Entity Recognition. (arXiv:2305.18624v1 [cs.CL])

    [http://arxiv.org/abs/2305.18624](http://arxiv.org/abs/2305.18624)

    W-procer是一种基于加权原型对比学习的医学少样本命名实体识别方法，在构建基于原型的对比损失和加权网络方面具有创新性，优于现有的最先进方法。

    

    对比学习已成为少样本命名实体识别（NER）的一种受欢迎的解决方案。传统配置力求减少具有相同标签的标记之间的距离，并增加具有不同标签的标记之间的距离。然而，在医学领域中存在大量被注释为“O”（即“OUTSIDE”）的实体，并且它们不希望被推离到当前对比学习方法标记为“O”以外的其他实体，这种设定效果不佳，可能会得出含有噪声原型标签的语义表示，尽管存在许多“O”标签实体与有标签实体相关。为解决这个挑战，我们提出了一种名为医学少样本命名实体识别中基于加权原型的对比学习方法（W-PROCER）。我们的方法主要围绕构建基于原型的对比损失和加权网络展开。这些组件在协助在医学领域中的迁移学习方面发挥了至关重要的作用。在实验中，我们将W-PROCER应用于一个公共的医学数据集，并展示了其相对于现有的最先进方法的优异表现。

    Contrastive learning has become a popular solution for few-shot Name Entity Recognization (NER). The conventional configuration strives to reduce the distance between tokens with the same labels and increase the distance between tokens with different labels. The effect of this setup may, however, in the medical domain, there are a lot of entities annotated as OUTSIDE (O), and they are undesirably pushed apart to other entities that are not labeled as OUTSIDE (O) by the current contrastive learning method end up with a noisy prototype for the semantic representation of the label, though there are many OUTSIDE (O) labeled entities are relevant to the labeled entities. To address this challenge, we propose a novel method named Weighted Prototypical Contrastive Learning for Medical Few Shot Named Entity Recognization (W-PROCER). Our approach primarily revolves around constructing the prototype-based contractive loss and weighting network. These components play a crucial role in assisting t
    
[^34]: 一网络，多任务：更高效的参数共用迁移学习方法

    One Network, Many Masks: Towards More Parameter-Efficient Transfer Learning. (arXiv:2305.17682v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.17682](http://arxiv.org/abs/2305.17682)

    本研究提出了PROPETL方法，通过原型网络和二进制掩码实现了更高效的参数共用迁移学习，解决了多任务微调预训练语言模型存储空间占用的问题。

    

    对于多个任务来说，微调预训练的语言模型往往会占用大量存储空间，参数共用迁移学习（PETL）方法可以缓解这个问题，但在应用于更广泛的任务范围时仍需要大量的参数和存储空间。为了实现更大的存储空间减少，我们提出了PROPETL，这是一种新颖的方法，可以在不同层和任务之间使用单个PETL模块，我们称之为原型网络（例如适配器、LoRA和前缀调整）。我们然后学习二进制掩码以从共享的原型网络中选择不同的子网络，并将它们作为PETL模块应用于不同的层。我们发现，二进制掩码可以确定网络中关键的信息，这在以前的研究中经常被忽略。我们的工作也可以看作是一种修剪方法，我们发现即使在看似很小的PETL模块中也存在过度参数化。我们对PROPETL进行了评估。

    Fine-tuning pre-trained language models for multiple tasks tends to be expensive in terms of storage. To mitigate this, parameter-efficient transfer learning (PETL) methods have been proposed to address this issue, but they still require a significant number of parameters and storage when being applied to broader ranges of tasks. To achieve even greater storage reduction, we propose PROPETL, a novel method that enables efficient sharing of a single PETL module which we call prototype network (e.g., adapter, LoRA, and prefix-tuning) across layers and tasks. We then learn binary masks to select different sub-networks from the shared prototype network and apply them as PETL modules into different layers. We find that the binary masks can determine crucial information from the network, which is often ignored in previous studies. Our work can also be seen as a type of pruning method, where we find that overparameterization also exists in the seemingly small PETL modules. We evaluate PROPETL
    
[^35]: 神经代码搜索中的后门攻击

    Backdooring Neural Code Search. (arXiv:2305.17506v2 [cs.SE] UPDATED)

    [http://arxiv.org/abs/2305.17506](http://arxiv.org/abs/2305.17506)

    本文研究了神经代码搜索模型的安全性问题，指出攻击者可以注入后门来返回具有安全/隐私问题的代码，提出了几种防御机制来缓解这种威胁，该工作突显了研究AI系统安全方面的重要性，特别是在部署于安全关键领域时。

    

    从在线代码库中重复使用现成代码是常见的做法，它极大地提高了软件开发人员的生产力。要查找所需的代码片段，开发人员则要通过自然语言查询使用代码搜索引擎。因此，在许多这样的搜索引擎后面，都是神经代码搜索模型。这些模型基于深度学习，因其出色的性能而备受关注。然而，这些模型的安全性很少被研究。具体来说，攻击者可能会在神经代码搜索模型中注入后门，从而返回具有安全/隐私问题的错误或甚至易受攻击的代码。这可能会影响下游软件（例如股票交易系统和自动驾驶），并造成财务损失和/或危及生命的事件。在本文中，我们展示了这样的攻击是可行的，并且可能非常隐蔽。通过简单地修改一个变量/函数名称，攻击者可以使出现错误/易受攻击的代码排名前11％。我们的攻击利用了神经网络的可再训练性和代码相似度度量的宽松性，以注入后门。我们还提出了几种防御机制来缓解这种威胁，例如添加对抗性训练数据和特征压缩。我们相信我们的工作突显了研究AI系统安全方面的重要性，特别是在部署于安全关键领域时。

    Reusing off-the-shelf code snippets from online repositories is a common practice, which significantly enhances the productivity of software developers. To find desired code snippets, developers resort to code search engines through natural language queries. Neural code search models are hence behind many such engines. These models are based on deep learning and gain substantial attention due to their impressive performance. However, the security aspect of these models is rarely studied. Particularly, an adversary can inject a backdoor in neural code search models, which return buggy or even vulnerable code with security/privacy issues. This may impact the downstream software (e.g., stock trading systems and autonomous driving) and cause financial loss and/or life-threatening incidents. In this paper, we demonstrate such attacks are feasible and can be quite stealthy. By simply modifying one variable/function name, the attacker can make buggy/vulnerable code rank in the top 11%. Our at
    
[^36]: 用下一句预测和互信息在潜空间中评估开放域对话

    Evaluating Open-Domain Dialogues in Latent Space with Next Sentence Prediction and Mutual Information. (arXiv:2305.16967v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.16967](http://arxiv.org/abs/2305.16967)

    本文提出了一种新的基于学习的自动评估度量方法（CMN），能够通过将条件变分自编码器（CVAEs）与下一句预测（NSP）目标相结合，并利用互信息（MI）在潜空间中建模文本的语义相似度，来鲁棒地评估开放域对话，并在实验中取得了优异的结果。

    

    开放域对话中的一对多问题使得自动评估方法面临重大挑战，本文提出了一种新的基于学习的自动评估度量方法（CMN），通过将条件变分自编码器（CVAEs）与下一句预测（NSP）目标相结合，并利用互信息（MI）在潜空间中建模文本的语义相似度，实现了对开放域对话的鲁棒评估。在两个开放域对话数据集上的实验结果表明，与广泛的基线方法相比，我们的方法具有明显的优越性，特别是在处理语义上远离黄金参考回答的响应时更为有效。

    The long-standing one-to-many issue of the open-domain dialogues poses significant challenges for automatic evaluation methods, i.e., there may be multiple suitable responses which differ in semantics for a given conversational context. To tackle this challenge, we propose a novel learning-based automatic evaluation metric (CMN), which can robustly evaluate open-domain dialogues by augmenting Conditional Variational Autoencoders (CVAEs) with a Next Sentence Prediction (NSP) objective and employing Mutual Information (MI) to model the semantic similarity of text in the latent space. Experimental results on two open-domain dialogue datasets demonstrate the superiority of our method compared with a wide range of baselines, especially in handling responses which are distant to the golden reference responses in semantics.
    
[^37]: 基于表面的检索降低了检索增强语言模型的困惑度

    Surface-Based Retrieval Reduces Perplexity of Retrieval-Augmented Language Models. (arXiv:2305.16243v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.16243](http://arxiv.org/abs/2305.16243)

    本文研究发现，用基于表面级别的检索机制取代语义检索可以显著降低检索增强语言模型的困惑度。

    

    已经证明，通过检索机制增强语言模型可以显著提高性能，同时保持参数数量较低。检索增强模型通常依靠基于查询块和潜在邻居之间的密集表示相似性的语义检索机制。本文研究了最先进的Retro模型，并观察到其性能提升更好地解释为基于表面级别的相似性，例如标记重叠。受此启发，我们用BM25替换Retro中的语义检索，获得了显著的困惑度降低。由于完整的BM25检索可能在大型数据集上具有计算成本，因此我们还将其应用于重新排名场景中，以最小的计算开销获得部分困惑度降低。

    Augmenting language models with a retrieval mechanism has been shown to significantly improve their performance while keeping the number of parameters low. Retrieval-augmented models commonly rely on a semantic retrieval mechanism based on the similarity between dense representations of the query chunk and potential neighbors. In this paper, we study the state-of-the-art Retro model and observe that its performance gain is better explained by surface-level similarities, such as token overlap. Inspired by this, we replace the semantic retrieval in Retro with a surface-level method based on BM25, obtaining a significant reduction in perplexity. As full BM25 retrieval can be computationally costly for large datasets, we also apply it in a re-ranking scenario, gaining part of the perplexity reduction with minimal computational overhead.
    
[^38]: 用解释提高语法错误修正系统的能力

    Enhancing Grammatical Error Correction Systems with Explanations. (arXiv:2305.15676v1 [cs.CL])

    [http://arxiv.org/abs/2305.15676](http://arxiv.org/abs/2305.15676)

    该论文介绍了一个用解释提高语法错误修正系统能力的方法，通过引入包含证据单词和语法错误类型注释的大数据集，找到错误的原因，并提出了几个基线和分析方法来理解这个任务，同时也证明了解释可以帮助第二语言学习者更好地理解语法规则。

    

    语法校正系统通过检测和更正语言错误来提升书写交流。为了帮助语言学习者更好地理解GEC系统为什么做出某种更正，错误的原因（证据单词）和相应的错误类型是两个关键因素。为了用解释增强GEC系统，我们引入了EXPECT，一个大数据集，其中包含了证据单词和语法错误类型的注释。我们提出了几个基线和分析方法来理解这个任务。此外，人类评估证明，我们可解释的GEC系统的解释能够帮助第二语言学习者确定是否接受更正建议，并理解相关的语法规则。

    Grammatical error correction systems improve written communication by detecting and correcting language mistakes. To help language learners better understand why the GEC system makes a certain correction, the causes of errors (evidence words) and the corresponding error types are two key factors. To enhance GEC systems with explanations, we introduce EXPECT, a large dataset annotated with evidence words and grammatical error types. We propose several baselines and anlysis to understand this task. Furthermore, human evaluation verifies our explainable GEC system's explanations can assist second-language learners in determining whether to accept a correction suggestion and in understanding the associated grammar rule.
    
[^39]: 利用（模糊测试）测试用例来理解程序

    Understanding Programs by Exploiting (Fuzzing) Test Cases. (arXiv:2305.13592v1 [cs.LG])

    [http://arxiv.org/abs/2305.13592](http://arxiv.org/abs/2305.13592)

    本文提出了通过模糊测试获取代表性输入来帮助语义理解程序的方法。

    

    程序的语义理解引起了社区的极大关注。受到自然语言理解中大型语言模型（LLM）的最近成功启发，通过将编程语言视为另一种自然语言，并在程序代码语料库上训练LLM，取得了巨大进展。然而，程序毕竟与文本有本质的区别，因为它们通常具有严格的结构和语法。特别是，程序及其基本单元（即函数和子程序）旨在展示各种行为和/或提供可能的输出，给定不同的输入。输入和可能的输出/行为之间的关系表示函数/子程序，并概述了整个程序。因此，我们提出将这种关系纳入学习中，以实现对程序的更深入语义理解。为了获得足够代表性的输入以触发大量执行，可以使用模糊测试。

    Semantic understanding of programs has attracted great attention in the community. Inspired by recent successes of large language models (LLMs) in natural language understanding, tremendous progress has been made by treating programming language as another sort of natural language and training LLMs on corpora of program code. However, programs are essentially different from texts after all, in a sense that they are normally heavily structured and syntax-strict. In particular, programs and their basic units (i.e., functions and subroutines) are designed to demonstrate a variety of behaviors and/or provide possible outputs, given different inputs. The relationship between inputs and possible outputs/behaviors represents the functions/subroutines and profiles the program as a whole. Therefore, we propose to incorporate such a relationship into learning, for achieving a deeper semantic understanding of programs. To obtain inputs that are representative enough to trigger the execution of mo
    
[^40]: 语义歧义遇上不确定性：探究用于词义消歧的不确定性估计方法

    Ambiguity Meets Uncertainty: Investigating Uncertainty Estimation for Word Sense Disambiguation. (arXiv:2305.13119v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.13119](http://arxiv.org/abs/2305.13119)

    本研究探究了用于词义消歧的不确定性估计方法，发现传统预测概率不足以量化不确定性，同时发现模型令人满意地反映了数据不确定性但是低估了模型不确定性。

    

    词义消歧(WSD)是自然语言理解中非常重要的一环，它通过给定一个句子中一个目标词的上下文，确定其恰当的词义。现有的监督方法将WSD视为分类任务，并取得了出色的性能。然而，它们在真实世界的环境中忽略了不确定性估计（UE），在这种环境下，数据总是嘈杂的，超出了分布范围。本文在专门为WSD设计的基准测试上广泛研究了UE。具体而言，我们首先比较了四种用于最先进WSD模型的不确定性得分，验证了模型末端获得的传统预测概率不足以量化不确定性。然后，我们通过拥有所选UE得分的模型在设计良好的测试场景中检测捕获数据和模型不确定性的能力，发现模型令人满意地反映了数据不确定性，但低估了模型不确定性。此外，我们还探讨了大量词汇属性，以检查这些属性是否有望作为UE的指导方向。

    Word sense disambiguation (WSD), which aims to determine an appropriate sense for a target word given its context, is crucial for natural language understanding. Existing supervised methods treat WSD as a classification task and have achieved remarkable performance. However, they ignore uncertainty estimation (UE) in the real-world setting, where the data is always noisy and out of distribution. This paper extensively studies UE on the benchmark designed for WSD. Specifically, we first compare four uncertainty scores for a state-of-the-art WSD model and verify that the conventional predictive probabilities obtained at the end of the model are inadequate to quantify uncertainty. Then, we examine the capability of capturing data and model uncertainties by the model with the selected UE score on well-designed test scenarios and discover that the model reflects data uncertainty satisfactorily but underestimates model uncertainty. Furthermore, we explore numerous lexical properties that int
    
[^41]: 基于提示的问题回答应用于开放研究知识图谱中的对象预测评估

    Evaluating Prompt-based Question Answering for Object Prediction in the Open Research Knowledge Graph. (arXiv:2305.12900v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.12900](http://arxiv.org/abs/2305.12900)

    本研究采用基于提示的训练方法，在学术知识图谱对象预测领域进行了大规模transformers模型的评估和测试，发现提示的使用可以改进pre-trained transformers的泛化能力。

    

    最近对于基于提示的训练方法在低资源环境下对转换器语言模型进行新文本体裁训练的调查有很多。发现基于提示的训练方法对于通用预训练或微调模型以适应资源缺乏的环境有很好的效果。本研究首次报道了采用基于提示训练transformers进行“学术知识图谱对象预测”的结果。该研究具有以下两个主要特点。1）它偏离了其他提出用于预测学术知识图谱对象的实体和关系提取流程的研究。2）在其他研究中测试了该方法对于与通用知识领域相对接近的文本体裁，而我们测试了该方法适用于显著不同的学术知识领域，从而测试这些大规模transformers模型的语言，概率和事实的普适性。我们发现（i）符合预期，使用提示进行微调的transformers优于基线；（ii）与先前研究中看到的模式不同，预先训练的transformers并不能始终足以胜任学术对象预测的任务，结果表明提示确实有助于改进抽取预训练模型所获得的语义信息的泛化能力。

    There have been many recent investigations into prompt-based training of transformer language models for new text genres in low-resource settings. The prompt-based training approach has been found to be effective in generalizing pre-trained or fine-tuned models for transfer to resource-scarce settings. This work, for the first time, reports results on adopting prompt-based training of transformers for \textit{scholarly knowledge graph object prediction}. The work is unique in the following two main aspects. 1) It deviates from the other works proposing entity and relation extraction pipelines for predicting objects of a scholarly knowledge graph. 2) While other works have tested the method on text genera relatively close to the general knowledge domain, we test the method for a significantly different domain, i.e. scholarly knowledge, in turn testing the linguistic, probabilistic, and factual generalizability of these large-scale transformer models. We find that (i) per expectations, t
    
[^42]: 连锁符号提示激发了大型语言模型中的规划能力

    Chain-of-Symbol Prompting Elicits Planning in Large Langauge Models. (arXiv:2305.10276v1 [cs.CL])

    [http://arxiv.org/abs/2305.10276](http://arxiv.org/abs/2305.10276)

    本文提出了自然语言规划（NLP）的基准，旨在研究LLMs在需要理解并在文本中相应进行操作的复杂规划任务中的表现。同时提出了一种新方法CoS，使用简化的符号空间表示法来表示复杂的环境。

    

    本文旨在研究LLMs在需要理解通过自然语言模拟的虚拟空间环境并在文本中相应进行操作的复杂规划任务中的表现。我们提出了一个名为自然语言规划（NLP）的基准，它由一组新颖的任务组成：Brick World、基于NLVR的操作和自然语言导航。我们发现当前流行的LLMs（如ChatGPT）仍然缺乏复杂规划的能力。这引出了一个问题——LLMs是否对自然语言中描述的环境有良好的理解，或者其他替代方法（如符号表示）是否更加简单，因此更容易被LLMs理解？为此，我们提出了一种名为CoS（Chain-of-Symbol Prompting）的新方法，在链式中间思考步骤中使用简化的符号空间表示法来表示复杂的环境。CoS易于使用，不需要对LLMs进行额外的培训。

    In this paper, we take the initiative to investigate the performance of LLMs on complex planning tasks that require LLMs to understand a virtual spatial environment simulated via natural language and act correspondingly in text. We propose a benchmark named Natural Language Planning (NLP) composed of a set of novel tasks: Brick World, NLVR-based Manipulations, and Natural Language Navigation. We found that current popular LLMs such as ChatGPT still lack abilities in complex planning. This arises a question -- do the LLMs have a good understanding of the environments described in natural language, or maybe other alternatives such as symbolic representations are neater and hence better to be understood by LLMs? To this end, we propose a novel method called CoS (Chain-of-Symbol Prompting) that represents the complex environments with condensed symbolic spatial representations during the chained intermediate thinking steps. CoS is easy to use and does not need additional training on LLMs. 
    
[^43]: 语言、时间偏好和消费行为：大型语言模型的证据

    Language, Time Preferences, and Consumer Behavior: Evidence from Large Language Models. (arXiv:2305.02531v1 [econ.GN])

    [http://arxiv.org/abs/2305.02531](http://arxiv.org/abs/2305.02531)

    本研究分析了大型语言模型在不同语言提示下的奖励时间偏好，并发现GPT在具有较弱未来时态的语言下表现出更大的耐心，这与使用该语言的人类的偏好相似。

    

    语言对我们对时间和奖励的感知有很大的影响。这引发了一个问题，即当以不同的语言询问大型语言模型时，它们是否显示出不同的奖励时间偏好，并且它们的选择是否类似于人类的选择。本研究分析了GPT-3.5（以下简称GPT）在多种语言提示下的响应，探索了较小、较早的奖励和较大、较晚的奖励之间的偏好。我们的结果显示，当以语义含义较弱的未来时态参考（FTR），如德语和汉语，为提示语时，GPT表现出更大的耐心，相比英语和法语等具有强大FTR的语言。这些发现与现有文献一致，并表明了GPT的选择与这些语言的使用者的偏好之间的关联。然而，进一步的分析揭示了较早或较晚奖励的偏好并没有随着奖励差异系统地改变，这表明了一种词典序优先的选择。

    Language has a strong influence on our perceptions of time and rewards. This raises the question of whether large language models, when asked in different languages, show different preferences for rewards over time and if their choices are similar to those of humans. In this study, we analyze the responses of GPT-3.5 (hereafter referred to as GPT) to prompts in multiple languages, exploring preferences between smaller, sooner rewards and larger, later rewards. Our results show that GPT displays greater patience when prompted in languages with weak future tense references (FTR), such as German and Mandarin, compared to languages with strong FTR, like English and French. These findings are consistent with existing literature and suggest a correlation between GPT's choices and the preferences of speakers of these languages. However, further analysis reveals that the preference for earlier or later rewards does not systematically change with reward gaps, indicating a lexicographic preferen
    
[^44]: 一种文本分组的统计探索：《创世记》和《出埃及记》中司祭派别的情况

    A Statistical Exploration of Text Partition Into Constituents: The Case of the Priestly Source in the Books of Genesis and Exodus. (arXiv:2305.02170v1 [cs.CL])

    [http://arxiv.org/abs/2305.02170](http://arxiv.org/abs/2305.02170)

    为了验证文本分组的假设，我们提出了一个统计文本探索的流程，并在圣经的前两卷书中应用此流程，成功地识别并探索了司祭派别和非司祭派别之间的统计明显的文体差异。

    

    我们提出了一个统计文本探索的流程，提供了一种基于文体学的解释，并对文本的假设分组进行了统计验证。给定文本的参数化，我们的流程：（1）检测文学特征，以产生假设分组和无监督分组之间的最佳重叠，（2）执行假设检验分析，量化最佳重叠的统计显著性，同时保留更可能被分组的文本单位之间的隐式相关性，以及（3）提取和量化对分类最负责的特征的重要性，并估计它们的统计稳定性和聚类-wise丰度。我们将这个流程应用于圣经中的前两卷书，圣经学者们认为，其中一种文体成分特别突出，即司祭派别。我们确定并探索了司祭派别和非司祭派别之间的统计明显的文体差异。

    We present a pipeline for a statistical textual exploration, offering a stylometry-based explanation and statistical validation of a hypothesized partition of a text. Given a parameterization of the text, our pipeline: (1) detects literary features yielding the optimal overlap between the hypothesized and unsupervised partitions, (2) performs a hypothesis-testing analysis to quantify the statistical significance of the optimal overlap, while conserving implicit correlations between units of text that are more likely to be grouped, and (3) extracts and quantifies the importance of features most responsible for the classification, estimates their statistical stability and cluster-wise abundance.  We apply our pipeline to the first two books in the Bible, where one stylistic component stands out in the eyes of biblical scholars, namely, the Priestly component. We identify and explore statistically significant stylistic differences between the Priestly and non-Priestly components.
    
[^45]: 基于因果感知的知识引导句子提取

    Causality-aware Concept Extraction based on Knowledge-guided Prompting. (arXiv:2305.01876v1 [cs.CL])

    [http://arxiv.org/abs/2305.01876](http://arxiv.org/abs/2305.01876)

    该论文提出了一种基于因果感知的知识引导提示方法，将其作为干预器装备到基于预训练语言模型的句子提取器中，以缓解概念偏差。在代表性的多语言KG数据集上进行广泛实验，获得了最先进的结果。

    

    概念有助于自然语言理解，但现有的知识图谱（KG）中远未完善。最近，预训练语言模型（PLM）已被广泛用于基于文本的概念提取（CE）。然而，PLM往往从大量语料库的共现关联中进行预训练知识挖掘，而非Token之间的真实因果关系。因此，预训练知识混淆了PLM，导致提取基于虚假共现相关性的有偏概念，不可避免地导致低精度。本文通过结构因果模型（SCM）提出了一种知识引导提示方法，将其作为干预器装备到基于PLM的提取器中，以减轻概念偏差。提示采用现有KG中的给定实体主题来缓解实体和有偏概念之间的虚假共现相关性。我们在代表性的多语言KG数据集上进行了广泛的实验，证明了我们提出的提示显著改进了提取性能，并达到了最先进的结果。

    Concepts benefit natural language understanding but are far from complete in existing knowledge graphs (KGs). Recently, pre-trained language models (PLMs) have been widely used in text-based concept extraction (CE). However, PLMs tend to mine the co-occurrence associations from massive corpus as pre-trained knowledge rather than the real causal effect between tokens.As a result, the pre-trained knowledge confounds PLMs to extract biased concepts based on spurious co-occurrence correlations, inevitably resulting in low precision. In this paper, through the lens of a Structural Causal Model (SCM), we propose equipping the PLM-based extractor with a knowledge-guided prompt as an intervention to alleviate concept bias. The prompt adopts the topic of the given entity from the existing knowledge in KGs to mitigate the spurious co-occurrence correlations between entities and biased concepts. Our extensive experiments on representative multilingual KG datasets justify that our proposed prompt 
    
[^46]: ChatGPT生成的代码真的正确吗？对大型语言模型在代码生成方面的严格评估

    Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation. (arXiv:2305.01210v1 [cs.SE])

    [http://arxiv.org/abs/2305.01210](http://arxiv.org/abs/2305.01210)

    本论文提出了一个严格的代码综合基准评估框架EvalPlus，用于评估利用大型语言模型生成的代码的功能正确性。

    

    程序综合一直以来都是被长期研究的领域，最近的方法集中于直接利用大型语言模型(LLMs)根据自然语言中用户的意图生成代码。代码评估数据集，包含策划好的综合问题和各种输入/输出测试用例，被用来衡量各种LLMs在代码综合上的性能。然而，这些数据集中的测试用例在完全评估生成代码的功能正确性方面，数量和质量都可能有所限制。这种现有基准中的限制引出了以下问题：在LLMs时代，生成的代码真的正确吗？为了回答这个问题，我们提出了EvalPlus——一个评估LLM-synthesized代码功能正确性的严格基准评估框架。EvalPlus接受基础评估数据集，并利用自动输入生成步骤，使用LLM-based和基于变异的方法生成和多样化大量新的测试输入。

    Program synthesis has been long studied with recent approaches focused on directly using the power of Large Language Models (LLMs) to generate code according to user intent written in natural language. Code evaluation datasets, containing curated synthesis problems with input/output test-cases, are used to measure the performance of various LLMs on code synthesis. However, test-cases in these datasets can be limited in both quantity and quality for fully assessing the functional correctness of the generated code. Such limitation in the existing benchmarks begs the following question: In the era of LLMs, is the code generated really correct? To answer this, we propose EvalPlus -- a code synthesis benchmarking framework to rigorously evaluate the functional correctness of LLM-synthesized code. In short, EvalPlus takes in the base evaluation dataset and uses an automatic input generation step to produce and diversify large amounts of new test inputs using both LLM-based and mutation-based
    
[^47]: ChartSumm：长短摘要自动生成任务的全面基准数据集

    ChartSumm: A Comprehensive Benchmark for Automatic Chart Summarization of Long and Short Summaries. (arXiv:2304.13620v1 [cs.CL])

    [http://arxiv.org/abs/2304.13620](http://arxiv.org/abs/2304.13620)

    本文提出了ChartSumm数据集，用于长短摘要自动生成任务，包括84000多个图表及其元数据和描述。研究发现，现有的自动摘要模型虽然得分不错，但经常面临错觉、漏掉重要数据点以及不正确解释复杂趋势等问题。

    

    自动将图表转换为文本摘要是视障人士的有效工具，同时为用户提供表格数据的自然语言精确洞察力。大型、结构良好的数据集始终是数据驱动模型的关键部分。本文提出了ChartSumm：一个大规模基准数据集，包括共84363个图表及其元数据和描述，涵盖广泛的主题和图表类型，可生成长短摘要。强基线模型的广泛实验表明，尽管这些模型通过实现各种自动评估指标的得分来生成流畅且信息丰富的摘要，但它们经常遇到一些问题，例如产生错觉，漏掉重要的数据点，以及不正确地解释图表中的复杂趋势。我们还通过自动翻译工具探讨了将ChartSumm扩展到其他语言的潜力。这使得我们的数据集成为一个有挑战的任务。

    Automatic chart to text summarization is an effective tool for the visually impaired people along with providing precise insights of tabular data in natural language to the user. A large and well-structured dataset is always a key part for data driven models. In this paper, we propose ChartSumm: a large-scale benchmark dataset consisting of a total of 84,363 charts along with their metadata and descriptions covering a wide range of topics and chart types to generate short and long summaries. Extensive experiments with strong baseline models show that even though these models generate fluent and informative summaries by achieving decent scores in various automatic evaluation metrics, they often face issues like suffering from hallucination, missing out important data points, in addition to incorrect explanation of complex trends in the charts. We also investigated the potential of expanding ChartSumm to other languages using automated translation tools. These make our dataset a challeng
    
[^48]: WizardLM: 增强大型语言模型遵循复杂指令的能力

    WizardLM: Empowering Large Language Models to Follow Complex Instructions. (arXiv:2304.12244v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2304.12244](http://arxiv.org/abs/2304.12244)

    本文使用 Evol-Instruct 方法创建了大量不同复杂度的指令数据用于微调 LLaMA 模型，得到了新模型 WizardLM。人类评估结果表明 Evol-Instruct 生成的指令优于人工创建的，而 WizardLM 输出的结果也比 OpenAI ChatGPT 更受欢迎。

    

    使用开放域指令追踪数据对大型语言模型进行训练带来了巨大的成功。然而，手动创建这样的指令数据非常耗时和劳动密集，且人类可能难以生成高复杂度指令。在本文中，我们展示了使用LLM而不是人类创建大量不同复杂度指令数据的途径。我们从一组初始指令开始，使用我们提出的Evol-Instruct逐步将其重新编写为更复杂的指令。然后，将所有生成的指令数据混合以微调LLaMA。我们称结果模型为WizardLM。针对一个复杂度平衡的测试集和Vicuna的测试集进行的人类评估表明，Evol-Instruct生成的指令优于人工创建的指令。通过分析高复杂性部分的人类评估结果，我们证明了从我们的WizardLM生成的输出比从OpenAI ChatGPT生成的输出更受欢迎。在GPT-4自动评估中，WizardLM产生了最好的结果。

    Training large language models (LLMs) with open-domain instruction following data brings colossal success. However, manually creating such instruction data is very time-consuming and labor-intensive. Moreover, humans may struggle to produce high-complexity instructions. In this paper, we show an avenue for creating large amounts of instruction data with varying levels of complexity using LLM instead of humans. Starting with an initial set of instructions, we use our proposed Evol-Instruct to rewrite them step by step into more complex instructions. Then, we mix all generated instruction data to fine-tune LLaMA. We call the resulting model WizardLM. Human evaluations on a complexity-balanced test bed and Vicuna's testset show that instructions from Evol-Instruct are superior to human-created ones. By analyzing the human evaluation results of the high complexity part, we demonstrate that outputs from our WizardLM are preferred to outputs from OpenAI ChatGPT. In GPT-4 automatic evaluation
    
[^49]: (向量)空间不是最后的疆域：将产品搜索看作程序合成

    (Vector) Space is Not the Final Frontier: Product Search as Program Synthesis. (arXiv:2304.11473v1 [cs.IR])

    [http://arxiv.org/abs/2304.11473](http://arxiv.org/abs/2304.11473)

    本文主张将产品搜索看作程序合成，相比向量空间模型有着重大优势。

    

    随着电子商务的不断增长，巨额投资用于信息检索的机器学习和自然语言处理也随之而来。虽然向量空间模型主宰了产品搜索中的检索模型，但随着深度学习的出现，向量化本身也发生了巨大变化。我们的立场论文以相反的方式主张，即程序合成对许多查询和市场中的大量参与者提供了重大优势。我们详细说明了所提出方法的行业重要性，概述了具体实现细节，并基于我们在Tooso构建类似系统的经验，回答了一些常见的反对意见。

    As ecommerce continues growing, huge investments in ML and NLP for Information Retrieval are following. While the vector space model dominated retrieval modelling in product search - even as vectorization itself greatly changed with the advent of deep learning -, our position paper argues in a contrarian fashion that program synthesis provides significant advantages for many queries and a significant number of players in the market. We detail the industry significance of the proposed approach, sketch implementation details, and address common objections drawing from our experience building a similar system at Tooso.
    
[^50]: LLM作为机器人的大脑：统一自我中心记忆与控制

    LLM as A Robotic Brain: Unifying Egocentric Memory and Control. (arXiv:2304.09349v1 [cs.AI])

    [http://arxiv.org/abs/2304.09349](http://arxiv.org/abs/2304.09349)

    本文提出了一个统一自我中心记忆和控制的框架LLM-Brain，使用大规模语言模型作为机器人大脑进行零-shot学习。该框架包括封闭式多轮对话，覆盖了感知、规划、控制和记忆，具有很好的泛化性能，适用于多个机器人任务。

    

    体感人工智能研究和开发具备物理或虚拟实体（即机器人）并能够与环境动态交互的智能系统。记忆和控制是体感系统的两个基本部分，通常需要分别使用框架进行建模。本文提出了一个新的、可推广的框架，称为LLM-Brain：使用大规模语言模型作为机器人大脑，统一自我中心记忆和控制。LLM-Brain框架集成了多个多模态语言模型用于机器人任务，利用零-shot学习方法。LLM-Brain中的所有组件使用自然语言进行封闭式多轮对话，包括感知、规划、控制和记忆。系统的核心是一个具备自我中心记忆和控制机器人的实体LLM。我们通过研究两个下游任务：主动探索和实体问答来演示LLM-Brain。

    Embodied AI focuses on the study and development of intelligent systems that possess a physical or virtual embodiment (i.e. robots) and are able to dynamically interact with their environment. Memory and control are the two essential parts of an embodied system and usually require separate frameworks to model each of them. In this paper, we propose a novel and generalizable framework called LLM-Brain: using Large-scale Language Model as a robotic brain to unify egocentric memory and control. The LLM-Brain framework integrates multiple multimodal language models for robotic tasks, utilizing a zero-shot learning approach. All components within LLM-Brain communicate using natural language in closed-loop multi-round dialogues that encompass perception, planning, control, and memory. The core of the system is an embodied LLM to maintain egocentric memory and control the robot. We demonstrate LLM-Brain by examining two downstream tasks: active exploration and embodied question answering. The
    
[^51]: 语言指导下的强化学习以实现人工智能协作

    Language Instructed Reinforcement Learning for Human-AI Coordination. (arXiv:2304.07297v1 [cs.AI])

    [http://arxiv.org/abs/2304.07297](http://arxiv.org/abs/2304.07297)

    本文提出了一种称之为instructRL的新的框架，它通过自然语言指令来指定对人工智能搭档的预期策略，解决在缺乏高质量人类行为数据的领域中多智能体强化学习收敛于人类不偏爱的策略的问题，从而提高了人工智能协作的性能。

    

    人工智能的一个基本问题是如何让智能体能够和人类有效地协作。本文提出了一种称之为instructRL的新的框架，让人们可以通过自然语言指令来指定对人工智能搭档的预期策略，以此解决在缺乏较高质量的人类行为数据的领域中，由于多智能体强化学习常常会收敛到人类并不偏爱的策略的不足。我们使用预先训练的大型语言模型来生成一个在人类指令下的先验策略，并将其用于约束强化学习目标。这导致强化学习智能体收敛到与人类喜好一致的均衡点。通过概念证明环境和具有挑战性的Hanabi基准，证明了instructRL收敛于满足给定指令的类似人类智能体的策略。最后，我们证明了知道语言指令显著提高了人工智能协作的性能。

    One of the fundamental quests of AI is to produce agents that coordinate well with humans. This problem is challenging, especially in domains that lack high quality human behavioral data, because multi-agent reinforcement learning (RL) often converges to different equilibria from the ones that humans prefer. We propose a novel framework, instructRL, that enables humans to specify what kind of strategies they expect from their AI partners through natural language instructions. We use pretrained large language models to generate a prior policy conditioned on the human instruction and use the prior to regularize the RL objective. This leads to the RL agent converging to equilibria that are aligned with human preferences. We show that instructRL converges to human-like policies that satisfy the given instructions in a proof-of-concept environment as well as the challenging Hanabi benchmark. Finally, we show that knowing the language instruction significantly boosts human-AI coordination pe
    
[^52]: 多模态C4：一种包含大量图像和文本的开放式数据库

    Multimodal C4: An Open, Billion-scale Corpus of Images Interleaved With Text. (arXiv:2304.06939v1 [cs.CV])

    [http://arxiv.org/abs/2304.06939](http://arxiv.org/abs/2304.06939)

    Multimodal C4是一个开放的、以图像与文本交替形式存在的数据库，其使用线性分配算法将图像放到长文本段落中，可用于通过少量样本学习和复杂相关度提示的建模。

    

    上下文视觉和语言模型需要支持任意交替的图像和文本序列作为输入, 这种格式不仅可以通过交替独立监督的(图像,文本)示例来进行低次学习,而且可以应对更复杂的提示, 涉及图像间互动,例如“图像A和图像B有什么共同之处?”现有的预训练模型使用类似于交替图像+文本的web语料库。但是，迄今为止，这种形式的大规模数据还没有公开提供。我们发布了Multimodal C4 (mmc4)，这是一个加强版的c4文本库，其中插入了图像。我们使用一个线性分配算法，使用CLIP特征将图像放到更长的文本体中，此过程优于其他替代方案。mmc4涵盖了诸如烹饪，旅游，技术等日常主题。对随机样本的手动检查表明，绝大多数(90%)的图像与主题相关。

    In-context vision and language models like Flamingo support arbitrarily interleaved sequences of images and text as input. This format not only enables few-shot learning via interleaving independent supervised (image, text) examples, but also, more complex prompts involving interaction between images, e.g., "What do image A and image B have in common?" To support this interface, pretraining occurs over web corpora that similarly contain interleaved images+text. To date, however, large-scale data of this form have not been publicly available.  We release Multimodal C4 (mmc4), an augmentation of the popular text-only c4 corpus with images interleaved. We use a linear assignment algorithm to place images into longer bodies of text using CLIP features, a process that we show outperforms alternatives. mmc4 spans everyday topics like cooking, travel, technology, etc. A manual inspection of a random sample of documents shows that a vast majority (90%) of images are topically relevant, and tha
    
[^53]: Vax-Culture: 用于研究推特上疫苗讨论的数据集

    Vax-Culture: A Dataset for Studying Vaccine Discourse on Twitter. (arXiv:2304.06858v1 [cs.SI])

    [http://arxiv.org/abs/2304.06858](http://arxiv.org/abs/2304.06858)

    本文介绍了一个推特疫苗数据集Vax-Culture，它旨在找出推广疫苗错误信息的文化和政治信念的重叠部分，帮助开发机器学习模型以自动检测疫苗错误信息帖子并应对其负面影响。

    

    COVID-19疫情期间，疫苗犹豫继续是公共卫生官员面临的主要挑战。由于该犹豫破坏了疫苗运动，许多研究人员试图确定其根本原因，并发现社交媒体平台上反疫苗错误信息的不断增长是该问题的关键因素。我们将推特作为误导内容的来源，并旨在提取推广疫苗错误信息的文化和政治信念的重叠部分。为此，我们收集了一个与疫苗有关的推文数据集，并借助专业沟通和新闻背景的注释人员进行注释。我们最终希望这可以带来有效和有针对性的公共卫生通信策略，以接触那些持反疫苗信仰者。此外，这些信息有助于开发机器学习模型以自动检测疫苗错误信息帖子并应对其负面影响。

    Vaccine hesitancy continues to be a main challenge for public health officials during the COVID-19 pandemic. As this hesitancy undermines vaccine campaigns, many researchers have sought to identify its root causes, finding that the increasing volume of anti-vaccine misinformation on social media platforms is a key element of this problem. We explored Twitter as a source of misleading content with the goal of extracting overlapping cultural and political beliefs that motivate the spread of vaccine misinformation. To do this, we have collected a data set of vaccine-related Tweets and annotated them with the help of a team of annotators with a background in communications and journalism. Ultimately we hope this can lead to effective and targeted public health communication strategies for reaching individuals with anti-vaccine beliefs. Moreover, this information helps with developing Machine Learning models to automatically detect vaccine misinformation posts and combat their negative impa
    
[^54]: SwissBERT：瑞士的多语言语言模型

    SwissBERT: The Multilingual Language Model for Switzerland. (arXiv:2303.13310v1 [cs.CL])

    [http://arxiv.org/abs/2303.13310](http://arxiv.org/abs/2303.13310)

    该论文介绍了SwissBERT，它是一个专门为处理瑞士相关文本而创建的多语言语言模型，SwissBERT在与瑞士相关的自然语言理解任务上的效果优于以前的模型。

    

    我们介绍了SwissBERT，这是一个专门为处理与瑞士相关的文本而创建的掩码语言模型。 SwissBERT是一种预训练模型，我们将其调整为能够处理瑞士国家语言 -德语、法语、意大利语和罗曼什语的新闻文章。我们评估了SwissBERT在与瑞士相关的自然语言理解任务上的效果，发现它在这些任务上的表现往往优于以前的模型，特别是在处理当代新闻和/或罗曼什语格里斯昆时。由于SwissBERT使用语言适配器，因此未来的工作可能将其扩展到瑞士德语方言中。该模型和我们的开源代码公开发布在https://github.com/ZurichNLP/swissbert。

    We present SwissBERT, a masked language model created specifically for processing Switzerland-related text. SwissBERT is a pre-trained model that we adapted to news articles written in the national languages of Switzerland -German, French, Italian, and Romansh. We evaluate SwissBERT on natural language understanding tasks related to Switzerland and find that it tends to outperform previous models on these tasks, especially when processing contemporary news and/or Romansh Grischun. Since SwissBERT uses language adapters, it may be extended to Swiss German dialects in future work. The model and our open-source code are publicly released at https://github.com/ZurichNLP/swissbert.
    
[^55]: Reflexion：具有动态记忆和自我反思的自主智能体

    Reflexion: an autonomous agent with dynamic memory and self-reflection. (arXiv:2303.11366v1 [cs.AI])

    [http://arxiv.org/abs/2303.11366](http://arxiv.org/abs/2303.11366)

    本文提出 Reflexion 方法，给智能体赋予了动态记忆和自我反思能力，以增强其任务特定的行动选择能力。

    

    最近决策大型语言模型（LLM）代理的发展在各种基准测试中展现出卓越的性能。然而，这些最先进的方法通常需要内部模型微调、外部模型微调或在定义的状态空间上进行策略优化。由于高质量训练数据的稀缺性或缺乏良好定义的状态空间，实现这些方法可能会具有挑战性。此外，这些代理没有人类决策过程固有的某些品质，特别是从错误中学习的能力。通过反思，人类可以通过试错过程高效地解决新的问题。在最近的研究基础上，我们提出 Reflexion，一种将动态记忆和自我反思能力赋予智能体的方法，以增强其现有的推理轨迹和任务特定的行动选择能力。为了实现完全自动化，我们介绍了一种简单而有效的方法。

    Recent advancements in decision-making large language model (LLM) agents have demonstrated impressive performance across various benchmarks. However, these state-of-the-art approaches typically necessitate internal model fine-tuning, external model fine-tuning, or policy optimization over a defined state space. Implementing these methods can prove challenging due to the scarcity of high-quality training data or the lack of well-defined state space. Moreover, these agents do not possess certain qualities inherent to human decision-making processes, specifically the ability to learn from mistakes. Self-reflection allows humans to efficiently solve novel problems through a process of trial and error. Building on recent research, we propose Reflexion, an approach that endows an agent with dynamic memory and self-reflection capabilities to enhance its existing reasoning trace and task-specific action choice abilities. To achieve full automation, we introduce a straightforward yet effective 
    
[^56]: 处理软件工程文档的停用词：它们重要吗？

    Stop Words for Processing Software Engineering Documents: Do they Matter?. (arXiv:2303.10439v1 [cs.SE])

    [http://arxiv.org/abs/2303.10439](http://arxiv.org/abs/2303.10439)

    本文研究了停用词在软件工程文档中的实用性。经实验证明，使用领域特定的停用词可以显著提高研究工具的性能，并且19个评估措施中有17个评估措施受益于停用词的消除。

    

    停用词通常被认为是不具有预测性的，因此在自然语言处理任务中通常会被去除。然而，不确定性词汇的定义是模糊的，因此大多数算法使用基于通用知识的停用词列表来去除停用词。学者们一直在就停用词的使用价值进行讨论，特别是在特定领域的设置中。在这项工作中，我们调查了停用词去除在软件工程背景下的实用性。为此，我们复制并实验了三个软件工程研究工具，并构建了一个软件工程领域相关文本的语料库，包括来自 Stack Overflow 的10,000个问题，并使用传统的信息论方法识别了200个领域特定的停用词。我们的结果表明，使用领域特定的停用词与使用通用停用列表相比，显着提高了研究工具的性能，并且19个评估措施中有17个评估措施受益于停用词的消除。我们的工作证明了在处理软件工程文档中去除领域特定的停用词的重要性。

    Stop words, which are considered non-predictive, are often eliminated in natural language processing tasks. However, the definition of uninformative vocabulary is vague, so most algorithms use general knowledge-based stop lists to remove stop words. There is an ongoing debate among academics about the usefulness of stop word elimination, especially in domain-specific settings. In this work, we investigate the usefulness of stop word removal in a software engineering context. To do this, we replicate and experiment with three software engineering research tools from related work. Additionally, we construct a corpus of software engineering domain-related text from 10,000 Stack Overflow questions and identify 200 domain-specific stop words using traditional information-theoretic methods. Our results show that the use of domain-specific stop words significantly improved the performance of research tools compared to the use of a general stop list and that 17 out of 19 evaluation measures sh
    
[^57]: 关于文本向量化技术的鲁棒性研究

    On the Robustness of Text Vectorizers. (arXiv:2303.07203v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2303.07203](http://arxiv.org/abs/2303.07203)

    本文研究了文本向量化技术中的鲁棒性问题，并证明了流行的嵌入方案具有Hamming距离意义上的鲁棒性。本研究提供了这些方法的定量边界，并展示了其中的常数受文档长度的影响。

    

    机器学习中一个基本问题是模型对输入变化的鲁棒性。在自然语言处理中，模型通常包含第一层嵌入，将词汇序列转换为向量表示。虽然连续输入的稳健性已经被很好地理解，但考虑到离散变化(比如替换句子中的一个词)，情况就不那么明确了。本文正式证明了流行的嵌入方案(如拼接、TF-IDF、段落向量)在Hamming距离意义下表现出鲁棒性，我们为这些方法提供了定量边界，并展示了其中的常数如何受文档长度影响。这些发现通过一系列数值实例加以说明。

    A fundamental issue in machine learning is the robustness of the model with respect to changes in the input. In natural language processing, models typically contain a first embedding layer, transforming a sequence of tokens into vector representations. While the robustness with respect to changes of continuous inputs is well-understood, the situation is less clear when considering discrete changes, for instance replacing a word by another in an input sentence. Our work formally proves that popular embedding schemes, such as concatenation, TF-IDF, and Paragraph Vector (a.k.a. doc2vec), exhibit robustness in the H\"older or Lipschitz sense with respect to the Hamming distance. We provide quantitative bounds for these schemes and demonstrate how the constants involved are affected by the length of the document. These findings are exemplified through a series of numerical examples.
    
[^58]: ChatGPT：应付千事的万能型 AI，但无所专精

    ChatGPT: Jack of all trades, master of none. (arXiv:2302.10724v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.10724](http://arxiv.org/abs/2302.10724)

    本研究检验了 ChatGPT 在 25 个不同的 NLP 任务上的性能，它是一个万能的 AI 模型，但无关紧要的表现可能会对某些任务的表现产生负面影响。

    

    OpenAI 推出了聊天生成预训练 Transformer（ChatGPT），革新了人工智能与人类互动的方法。许多研究通过测试 ChatGPT 在众所周知的自然语言处理（NLP）任务中的效果，来评估该模型的效能。然而，现有的研究大多非自动化，并且规模非常有限。本研究在 25 个不同的 NLP 任务上检验了 ChatGPT 的性能，其中大多数任务甚至对人类而言都是主观的，例如情感分析、情绪识别、攻击性和立场检测。另一些任务则需要更客观的推理，如词义消歧、语言可接受性和问答。我们还对 GPT-4 模型在五个选定的 NLP 任务子集上进行了评估。我们自动化了 ChatGPT 和 GPT-4 的引导过程，并分析了超过 49k 个响应。与现有最先进的解决方案（SOTA）进行比较，我们的结果显示，在一些任务上 ChatGPT 的性能存在一定的缺陷。

    OpenAI has released the Chat Generative Pre-trained Transformer (ChatGPT) and revolutionized the approach in artificial intelligence to human-model interaction. Several publications on ChatGPT evaluation test its effectiveness on well-known natural language processing (NLP) tasks. However, the existing studies are mostly non-automated and tested on a very limited scale. In this work, we examined ChatGPT's capabilities on 25 diverse analytical NLP tasks, most of them subjective even to humans, such as sentiment analysis, emotion recognition, offensiveness, and stance detection. In contrast, the other tasks require more objective reasoning like word sense disambiguation, linguistic acceptability, and question answering. We also evaluated GPT-4 model on five selected subsets of NLP tasks. We automated ChatGPT and GPT-4 prompting process and analyzed more than 49k responses. Our comparison of its results with available State-of-the-Art (SOTA) solutions showed that the average loss in quali
    
[^59]: MultiInstruct: 通过指令调优来改善多模态零样本学习

    MultiInstruct: Improving Multi-Modal Zero-Shot Learning via Instruction Tuning. (arXiv:2212.10773v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.10773](http://arxiv.org/abs/2212.10773)

    本文介绍了 MultiInstruct，这是第一个多模态指令调优基准数据集，并探索多种迁移学习策略从大规模的自然语言指令数据集中提高预训练模型的性能。实验结果展示了其在各种未见过的多模态任务中具有强大的零样本表现，以及设计的新的任务完成率指标。

    

    指令调优是一种新的学习范式，它在指令指定的任务上对预训练的语言模型进行微调，在各种自然语言处理任务上展现了有希望的零样本表现。然而，它尚未被用于视觉和多模态任务。本文介绍了 MultiInstruct，这是第一个多模态指令调优基准数据集，包含 47 个不同的多模态任务，涵盖了 11 个广泛的类别。每个任务至少设计有 5,000 个实例（输入-输出对）来自现有的开源数据集和 5 个专家编写的指令。我们选取 OFA 作为多模态指令调优的基础预训练模型，并探索多种迁移学习策略，以利用大规模的自然语言指令数据集。实验结果展示了其在各种未见过的多模态任务中具有强大的零样本表现，以及从纯文本指令中获得迁移学习的好处。我们还设计了一种新的任务完成率指标来评估零样本性能，度量模型在仅有指令的情况下完成任务的能力。

    Instruction tuning, a new learning paradigm that fine-tunes pre-trained language models on tasks specified through instructions, has shown promising zero-shot performance on various natural language processing tasks. However, it's still not explored for vision and multimodal tasks. In this work, we introduce MultiInstruct, the first multimodal instruction tuning benchmark dataset that consists of 47 diverse multimodal tasks covering 11 broad categories. Each task is designed at least with 5,000 instances (input-out pairs) from existing open-source datasets and 5 expert-written instructions. We take OFA as the base pre-trained model for multimodal instruction tuning, and to improve its performance, we explore multiple transfer learning strategies to leverage the large-scale Natural Instructions dataset. Experimental results demonstrate its strong zero-shot performance on various unseen multimodal tasks and the benefit of transfer learning from text-only instructions. We also design a ne
    
[^60]: BLIND: 无人口统计学的偏见去除方法

    BLIND: Bias Removal With No Demographics. (arXiv:2212.10563v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.10563](http://arxiv.org/abs/2212.10563)

    BLIND可以在没有先前了解数据集人口统计信息的情况下消除训练模型中的社会偏见。

    

    在真实世界数据训练的模型往往会模仿和放大社会偏见。常见的减轻偏见的方法需要先了解哪些类型的偏见需要被纠正（例如性别或种族偏见）以及与每个数据样本相关联的社会群体。在本文中，我们介绍了一种名为BLIND的方法，它可以在没有对数据集中人口统计信息的先前了解下进行偏见去除。在训练下游任务模型时，BLIND使用一个辅助模型来预测主模型的成功，并在训练过程中减少这些受到偏见样本的权重。基于对情感分类和职业分类任务中的种族和性别偏见的实验表明，BLIND可以在不依赖昂贵的人口统计注释过程的情况下消除社会偏见。我们的方法与需要人口统计学信息的其他方法相比具有竞争力，有时甚至超过它们。

    Models trained on real-world data tend to imitate and amplify social biases. Common methods to mitigate biases require prior information on the types of biases that should be mitigated (e.g., gender or racial bias) and the social groups associated with each data sample. In this work, we introduce BLIND, a method for bias removal with no prior knowledge of the demographics in the dataset. While training a model on a downstream task, BLIND detects biased samples using an auxiliary model that predicts the main model's success, and down-weights those samples during the training process. Experiments with racial and gender biases in sentiment classification and occupation classification tasks demonstrate that BLIND mitigates social biases without relying on a costly demographic annotation process. Our method is competitive with other methods that require demographic information and sometimes even surpasses them.
    
[^61]: 神经机器翻译的持续知识蒸馏

    Continual Knowledge Distillation for Neural Machine Translation. (arXiv:2212.09097v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.09097](http://arxiv.org/abs/2212.09097)

    本论文提出了一种称为持续知识蒸馏的方法，利用已有的翻译模型来提高一个新模型的性能。

    

    虽然许多平行语料库因为版权、数据隐私和竞争差异等原因不公开，但训练好的翻译模型在开放平台上变得越来越容易得到。本文提出了一种称为持续知识蒸馏的方法，利用现有的翻译模型来提高感兴趣的模型的性能。基本思路是将每个已训练模型的知识按顺序转移给被蒸馏模型。在中英和德英数据集上的广泛实验表明，我们的方法在同质和异质训练模型设置下都实现了显著且一致的改进，并且对恶意模型具有鲁棒性。

    While many parallel corpora are not publicly accessible for data copyright, data privacy and competitive differentiation reasons, trained translation models are increasingly available on open platforms. In this work, we propose a method called continual knowledge distillation to take advantage of existing translation models to improve one model of interest. The basic idea is to sequentially transfer knowledge from each trained model to the distilled model. Extensive experiments on Chinese-English and German-English datasets show that our method achieves significant and consistent improvements over strong baselines under both homogeneous and heterogeneous trained model settings and is robust to malicious models.
    
[^62]: 联邦神经主题模型

    Federated Neural Topic Models. (arXiv:2212.02269v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2212.02269](http://arxiv.org/abs/2212.02269)

    该论文提出了一种基于神经主题建模实现的联邦神经主题模型，可以在不共享数据的情况下允许多个方共同训练主题模型和保护节点隐私。

    

    近年来，主题建模已成为组织和总结大型文档集合或在其中搜索特定模式的强大技术。然而，当从不同来源交叉分析数据时，可能会出现隐私问题。联邦主题建模通过允许多个方共同训练主题模型而不共享其数据来解决此问题。我们提出并分析了一种基于最先进的神经主题建模实现的联邦实现，显示其在节点文档的主题多样性和建立联合模型的需要时的优势。在实践中，我们的方法相当于集中模型训练，但保护节点的隐私。使用基准数据集进行的实验说明了这种联邦场景的优点。

    Over the last years, topic modeling has emerged as a powerful technique for organizing and summarizing big collections of documents or searching for particular patterns in them. However, privacy concerns may arise when cross-analyzing data from different sources. Federated topic modeling solves this issue by allowing multiple parties to jointly train a topic model without sharing their data. While several federated approximations of classical topic models do exist, no research has been conducted on their application for neural topic models. To fill this gap, we propose and analyze a federated implementation based on state-of-the-art neural topic modeling implementations, showing its benefits when there is a diversity of topics across the nodes' documents and the need to build a joint model. In practice, our approach is equivalent to a centralized model training, but preserves the privacy of the nodes. Advantages of this federated scenario are illustrated by means of experiments using b
    
[^63]: 最近邻语言模型的适应性方法

    Adaptation Approaches for Nearest Neighbor Language Models. (arXiv:2211.07828v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.07828](http://arxiv.org/abs/2211.07828)

    本论文提出了三种方法来适应半参数最近邻语言模型（$k$NN-LMs），并运用消融实验和对多个适应领域的广泛评估，发现组合适应方法 consistently outperforms单一适应策略和零样本（$k$NN-LM）基线，重计分模块使得性能提高最多。

    

    半参数最近邻语言模型（$k$NN-LMs）通过利用对外部内存数据存储器的大规模邻域检索，比纯参数模型取得了惊人的提升。然而，对于如何适应新领域的模型，目前研究还很少。本文试图填补这一空白，并建议以下适应$k$NN-LMs的方法——1）通过使用转换器来适应底层LM，2）扩展邻域检索到另一个适应数据存储，3）使用学习的重计分模块来适应检索邻居的权重（分数）。我们分别研究了每种适应策略，以及通过消融实验和对7个适应字段运行的广泛评估来组合性能的提高。我们的组合适应方法始终优于纯参数适应和零样本（$k$NN-LM）基线，这些基线从适应数据构建数据存储。平均而言，我们看到相对于基线的困惑度降低最多20％，其中我们的重新计分-$k$NN-LM方法在所有适应场景中提供了最大的改进。

    Semi-parametric Nearest Neighbor Language Models ($k$NN-LMs) have produced impressive gains over purely parametric LMs, by leveraging large-scale neighborhood retrieval over external memory datastores. However, there has been little investigation into adapting such models for new domains. This work attempts to fill that gap and suggests the following approaches for adapting $k$NN-LMs -- 1) adapting the underlying LM (using Adapters), 2) expanding neighborhood retrieval over an additional adaptation datastore, and 3) adapting the weights (scores) of retrieved neighbors using a learned Rescorer module. We study each adaptation strategy separately, as well as the combined performance improvement through ablation experiments and an extensive set of evaluations run over seven adaptation domains. Our combined adaptation approach consistently outperforms purely parametric adaptation and zero-shot ($k$NN-LM) baselines that construct datastores from the adaptation data. On average, we see perpl
    
[^64]: 去偏置一定会降低模型性能吗?

    Does Debiasing Inevitably Degrade the Model Performance. (arXiv:2211.07350v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.07350](http://arxiv.org/abs/2211.07350)

    语言模型中的性别偏见问题日益引起关注，但当前去偏置方法往往会降低模型在其他任务上的表现；本论文提出了一个理论框架解释模型中性别偏见的机制，并发现了一种新的去偏置方法，能实现缓解性别偏见同时避免性能下降的双重优势。

    

    语言模型中的性别偏见已经引起了足够的关注，因为它威胁到社会公正。然而，目前大多数去偏置方法在降低模型在其他任务上的表现方面表现出了不稳定的性质，而这种降低的机制仍然是神秘的。我们提出了一个理论框架来解释语言模型性别偏见的三种候选机制，使用我们的理论框架来解释目前的去偏置方法如何导致性能降低。我们还发现了一种去偏置不会降低模型性能的途径，并进一步开发了一种因果检测微调方法来纠正性别偏见。数值实验表明，我们的方法能够实现双重收益：部分缓解性别偏见同时避免性能下降。

    Gender bias in language models has attracted sufficient attention because it threatens social justice. However, most of the current debiasing methods degraded the model's performance on other tasks while the degradation mechanism is still mysterious. We propose a theoretical framework explaining the three candidate mechanisms of the language model's gender bias. We use our theoretical framework to explain why the current debiasing methods cause performance degradation. We also discover a pathway through which debiasing will not degrade the model performance. We further develop a causality-detection fine-tuning approach to correct gender bias. The numerical experiment demonstrates that our method is able to lead to double dividends: partially mitigating gender bias while avoiding performance degradation.
    
[^65]: 生成多语言的性别不明确的文本转语音声音

    Generating Multilingual Gender-Ambiguous Text-to-Speech Voices. (arXiv:2211.00375v2 [cs.SD] UPDATED)

    [http://arxiv.org/abs/2211.00375](http://arxiv.org/abs/2211.00375)

    该论文介绍了一种生成多语言的性别不明确的TTS声音的方法，通过提出的性别感知方法从潜在说话人中有效地进行采样，成功生成了一系列新的、多样化的、一致性和性别不明确性更强的声音，具有很强的实验表现。

    

    语音用户界面的性别是其被感知身份的关键元素。最近，越来越多的界面开始采用不明确的性别，而不是明确界定为男性或女性。这项工作解决了在多说话人，多语言环境中生成新的性别不明确的TTS语音的任务。这是通过使用提出的性别感知方法有效地从潜在的说话人嵌入空间中进行采样来实现的。广泛的客观和主观评估清楚地表明，该方法能够有效地生成一系列新的、多样化的声音，这些声音在所有考察的语言中都被认为比基线声音更具性别不明确性。有趣的是，性别认知被发现在听众的两个人口统计因素方面具有鲁棒性：母语和性别。据我们所知，这是第一个可以可靠地生成多种性别不明确声音的系统性和经过验证的方法。

    The gender of any voice user interface is a key element of its perceived identity. Recently, there has been increasing interest in interfaces where the gender is ambiguous rather than clearly identifying as female or male. This work addresses the task of generating novel gender-ambiguous TTS voices in a multi-speaker, multilingual setting. This is accomplished by efficiently sampling from a latent speaker embedding space using a proposed gender-aware method. Extensive objective and subjective evaluations clearly indicate that this method is able to efficiently generate a range of novel, diverse voices that are consistent and perceived as more gender-ambiguous than a baseline voice across all the languages examined. Interestingly, the gender perception is found to be robust across two demographic factors of the listeners: native language and gender. To our knowledge, this is the first systematic and validated approach that can reliably generate a variety of gender-ambiguous voices.
    
[^66]: 怠惰神经元现象：变压器模型激活稀疏性的出现

    The Lazy Neuron Phenomenon: On Emergence of Activation Sparsity in Transformers. (arXiv:2210.06313v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2210.06313](http://arxiv.org/abs/2210.06313)

    本文研究了使用变压器模型的机器学习模型中激活图的稀疏现象，发现在不同层数的变压器配置和其他体系结构中都出现了稀疏现象。

    

    本文研究了变压器模型的机器学习模型的激活图稀疏的奇特现象。我们通过中间层多层感知器（MLP）使用ReLU激活函数的输出来表示激活图，稀疏是指平均情况下每个输入到MLP的非零元素非常少（例如，T5-Base为3.0％，ViT-B16为6.3％）。此外，较大的变压器和更宽的MLP隐藏层维度会产生更稀疏的激活图。通过大量实验，我们证明了稀疏的出现是一种普遍现象，它出现在自然语言处理和视觉任务中，出现在训练和评估数据中，在不同层数的变压器配置和其他体系结构中，也包括MLP-混合器和2层MLP。我们还表明，使用具有随机标签或随机输入的训练数据集也会出现稀疏现象。

    This paper studies the curious phenomenon for machine learning models with Transformer architectures that their activation maps are sparse. By activation map we refer to the intermediate output of the multi-layer perceptrons (MLPs) after a ReLU activation function, and by sparse we mean that on average very few entries (e.g., 3.0% for T5-Base and 6.3% for ViT-B16) are nonzero for each input to MLP. Moreover, larger Transformers with more layers and wider MLP hidden dimensions are sparser as measured by the percentage of nonzero entries. Through extensive experiments we demonstrate that the emergence of sparsity is a prevalent phenomenon that occurs for both natural language processing and vision tasks, on both training and evaluation data, for Transformers of various configurations, at layers of all depth levels, as well as for other architectures including MLP-mixers and 2-layer MLPs. We show that sparsity also emerges using training datasets with random labels, or with random inputs,
    
[^67]: 通过鼓励一致的基于梯度的解释来改进视觉 grounding

    Improving Visual Grounding by Encouraging Consistent Gradient-based Explanations. (arXiv:2206.15462v3 [cs.CV] UPDATED)

    [http://arxiv.org/abs/2206.15462](http://arxiv.org/abs/2206.15462)

    该论文提出了一种名为 AMC 的目标函数，鼓励基于梯度的解释覆盖有注释的感兴趣区域，即编码区域。该方法在提高视觉 grounding 性能方面表现卓越，有望成为视觉 grounding 领域的新进展。

    

    我们提出了一种基于边缘的损失，用于预训练视觉语言模型，鼓励基于梯度的解释与区域级注释保持一致。我们将这个目标称为 Attention Mask Consistency（AMC），并证明它产生了比依赖于区域级注释的模型更优越的视觉 grounding 性能。 AMC 通过鼓励基于梯度的解释掩码，在包含此类注释的图像中，把它们的注意力分数主要集中在注释的感兴趣区域内。特别地，一个在标准视觉-语言建模目标之上用 AMC 训练的模型，在 Flickr30k 视觉 grounding 基准测试中获得了86.59%的最新结果，相比最佳结果获得了5.48%的绝对提升。我们的方法在已建立的指代表达理解基准测试中表现优秀，还提供了额外的好处。

    We propose a margin-based loss for vision-language model pretraining that encourages gradient-based explanations that are consistent with region-level annotations. We refer to this objective as Attention Mask Consistency (AMC) and demonstrate that it produces superior visual grounding performance compared to models that rely instead on region-level annotations for explicitly training an object detector such as Faster R-CNN. AMC works by encouraging gradient-based explanation masks that focus their attention scores mostly within annotated regions of interest for images that contain such annotations. Particularly, a model trained with AMC on top of standard vision-language modeling objectives obtains a state-of-the-art accuracy of 86.59% in the Flickr30k visual grounding benchmark, an absolute improvement of 5.48% when compared to the best previous model. Our approach also performs exceedingly well on established benchmarks for referring expression comprehension and offers the added bene
    
[^68]: 超越模仿游戏：量化和拓展语言模型的能力

    Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models. (arXiv:2206.04615v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2206.04615](http://arxiv.org/abs/2206.04615)

    本研究引入了Beyond the Imitation Game基准测试（BIG-bench），该测试集包含了204个各领域的难题，旨在评估当前语言模型的能力并为未来的研究提供信息和准备。

    

    随着规模的增大，语言模型展示了数量上的提升和新的定性能力。尽管具有潜在的转变性影响，但这些新的能力目前尚未被充分描述。为了为未来的研究提供信息，为剧变的新型模型能力做准备，并缓解社会有害影响，我们必须了解语言模型的现有和近期能力和限制。为了解决这一挑战，我们引入了Beyond the Imitation Game基准测试（BIG-bench）。BIG-bench目前包括204个任务，由132个机构的450名作者贡献。任务主题多样，涵盖了语言学、儿童发展、数学、常识推理、生物学、物理学、社会偏见、软件开发等等。BIG-bench专注于那些被认为超出了当前语言模型能力的任务。我们评估了OpenAI的GPT模型和谷歌内部的密集转换模型的行为。

    Language models demonstrate both quantitative improvement and new qualitative capabilities with increasing scale. Despite their potentially transformative impact, these new capabilities are as yet poorly characterized. In order to inform future research, prepare for disruptive new model capabilities, and ameliorate socially harmful effects, it is vital that we understand the present and near-future capabilities and limitations of language models. To address this challenge, we introduce the Beyond the Imitation Game benchmark (BIG-bench). BIG-bench currently consists of 204 tasks, contributed by 450 authors across 132 institutions. Task topics are diverse, drawing problems from linguistics, childhood development, math, common-sense reasoning, biology, physics, social bias, software development, and beyond. BIG-bench focuses on tasks that are believed to be beyond the capabilities of current language models. We evaluate the behavior of OpenAI's GPT models, Google-internal dense transform
    
[^69]: DiMS：迭代非自回归Transformer的压缩多步骤机制在机器翻译中的应用

    DiMS: Distilling Multiple Steps of Iterative Non-Autoregressive Transformers for Machine Translation. (arXiv:2206.02999v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2206.02999](http://arxiv.org/abs/2206.02999)

    该论文提出了一种新的技术DiMS，可以通过压缩多步骤机制来优化解码过程，提高机器翻译的效率和质量。

    

    迭代非自回归Transformer的计算优势在解码步骤增加时会减弱。我们引入了Distill Multiple Steps（DiMS），这是一种简单而有效的压缩技术，可减少达到特定翻译质量所需的步骤数。压缩后的模型既享受了早期迭代的计算优势，同时又保留了多个迭代步骤的增强效果。DiMS需要两个模型，即学生模型和教师模型。学生模型通过优化以预测多次解码后的教师模型输出，而教师模型则通过缓慢移动平均跟随学生模型，这使得教师模型的知识得到更新，并提高了提供的标签的质量。在推理过程中，学生模型用于翻译，并且没有额外的计算负担。我们在各种模型上验证了DiMS的有效性，单步翻译中获得了7.8和12.9 BLEU分的改进。

    The computational benefits of iterative non-autoregressive transformers decrease as the number of decoding steps increases. As a remedy, we introduce Distill Multiple Steps (DiMS), a simple yet effective distillation technique to decrease the number of required steps to reach a certain translation quality. The distilled model enjoys the computational benefits of early iterations while preserving the enhancements from several iterative steps. DiMS relies on two models namely student and teacher. The student is optimized to predict the output of the teacher after multiple decoding steps while the teacher follows the student via a slow-moving average. The moving average keeps the teacher's knowledge updated and enhances the quality of the labels provided by the teacher. During inference, the student is used for translation and no additional computation is added. We verify the effectiveness of DiMS on various models obtaining 7.8 and 12.9 BLEU points improvements in single-step translation
    
[^70]: 跨视图语言建模：迈向统一的跨语言跨模态预训练

    Cross-View Language Modeling: Towards Unified Cross-Lingual Cross-Modal Pre-training. (arXiv:2206.00621v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2206.00621](http://arxiv.org/abs/2206.00621)

    本文提出了跨视图语言建模框架，该框架将跨语言和跨模态预训练统一在共享的架构和目标下进行，通过有条件的掩码语言建模和对比学习来最大化不同视图之间的互信息以实现两个视图的对齐。

    

    本文引入了跨视图语言建模，这是一个简单而有效的预训练框架，将跨语言和跨模态预训练与共享架构和目标统一起来。我们的方法受到一个关键观察的启发，即跨语言和跨模态预训练具有将同一对象的两个不同视图对齐到一个共同语义空间的相同目标。为此，跨视图语言建模框架将多模态数据（即图像-标题对）和多语言数据（即平行句对）视为同一对象的两个不同视图，并通过有条件的掩码语言建模和对比学习来最大化它们之间的互信息来训练模型，以对齐两个视图。我们使用跨视图语言建模框架对跨语言跨模态语言模型CCLM进行预训练。在多语言多模态基准IGLUE和两个多语言图像-文本检索数据上进行了实证结果。

    In this paper, we introduce Cross-View Language Modeling, a simple and effective pre-training framework that unifies cross-lingual and cross-modal pre-training with shared architectures and objectives. Our approach is motivated by a key observation that cross-lingual and cross-modal pre-training share the same goal of aligning two different views of the same object into a common semantic space. To this end, the cross-view language modeling framework considers both multi-modal data (i.e., image-caption pairs) and multi-lingual data (i.e., parallel sentence pairs) as two different views of the same object, and trains the model to align the two views by maximizing the mutual information between them with conditional masked language modeling and contrastive learning. We pre-train CCLM, a Cross-lingual Cross-modal Language Model, with the cross-view language modeling framework. Empirical results on IGLUE, a multi-lingual multi-modal benchmark, and two multi-lingual image-text retrieval data
    
[^71]: ClaimDiff：比较和对比有争议问题的声明

    ClaimDiff: Comparing and Contrasting Claims on Contentious Issues. (arXiv:2205.12221v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2205.12221](http://arxiv.org/abs/2205.12221)

    本研究提出ClaimDiff数据集，主要关注声明之间的微妙差别，并观察到强有力的基准测试无法探测这些差别，与人类存在超过19%的绝对差距。

    

    随着发现虚假信息的重要性日益增加，许多研究都专注于检验事实声明来检索证据。然而，传统事实验证任务并不适用于捕捉事实上一致但可能仍会对读者产生偏见的微妙差异，尤其是在有争议的政治或经济问题上。我们的基本假设是，在受信任的信息源中，一个人的论点并不一定比另一个人更正确，需要进行比较而不是简单的验证。在这项研究中，我们提出了ClaimDiff，这是一个主要关注声明对之间细微差别的新数据集。在ClaimDiff中，我们提供了来自268篇新闻文章的2,941个标注声明对。我们观察到，虽然人类能够检测到声明之间的细微差别，但强大的基准测试无法探测到它们，与人类存在超过19%的绝对差距。我们希望这项初步研究能够通过机器辅助帮助读者获得有争议问题的无偏视角。

    With the growing importance of detecting misinformation, many studies have focused on verifying factual claims by retrieving evidence. However, canonical fact verification tasks do not apply to catching subtle differences in factually consistent claims, which might still bias the readers, especially on contentious political or economic issues. Our underlying assumption is that among the trusted sources, one's argument is not necessarily more true than the other, requiring comparison rather than verification. In this study, we propose ClaimDiff, a novel dataset that primarily focuses on comparing the nuance between claim pairs. In ClaimDiff, we provide 2,941 annotated claim pairs from 268 news articles. We observe that while humans are capable of detecting the nuances between claims, strong baselines struggle to detect them, showing over a 19% absolute gap with the humans. We hope this initial study could help readers to gain an unbiased grasp of contentious issues through machine-aided
    
[^72]: 神经文本生成的最新进展：一项任务无关的调查

    Recent Advances in Neural Text Generation: A Task-Agnostic Survey. (arXiv:2203.03047v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2203.03047](http://arxiv.org/abs/2203.03047)

    本文调查了最近神经文本生成领域的最新进展，包括数据构建、神经框架、训练和推理策略和评估指标等四个方面，并探讨了神经管道和背景知识的利用等未来方向。

    

    近年来，相当多的研究致力于在自然语言生成（NLG）领域中应用神经模型。主要目标是生成既具有语言自然性又具有人类化属性的文本，并同时对生成过程进行控制。本文提供了一份全面的，任务无关的神经文本生成最新进展调查。这些进展通过多种发展得以实现，我们将其分为四个主要方面：数据构建，神经框架，训练和推理策略和评估指标。通过研究这些不同方面，我们旨在提供对该领域的进展的全面概述。此外，我们探讨了神经文本生成的未来方向，这些方向包括利用神经管道和融合背景知识，这些途径为进一步增强神经文本生成系统的能力提供了有希望的机会。

    In recent years, considerable research has been dedicated to the application of neural models in the field of natural language generation (NLG). The primary objective is to generate text that is both linguistically natural and human-like, while also exerting control over the generation process. This paper offers a comprehensive and task-agnostic survey of the recent advancements in neural text generation. These advancements have been facilitated through a multitude of developments, which we categorize into four key areas: data construction, neural frameworks, training and inference strategies, and evaluation metrics. By examining these different aspects, we aim to provide a holistic overview of the progress made in the field. Furthermore, we explore the future directions for the advancement of neural text generation, which encompass the utilization of neural pipelines and the incorporation of background knowledge. These avenues present promising opportunities to further enhance the cap
    
[^73]: 在自监督预训练中衡量个别领域因素的影响

    Measuring the Impact of Individual Domain Factors in Self-Supervised Pre-Training. (arXiv:2203.00648v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2203.00648](http://arxiv.org/abs/2203.00648)

    本研究对自动语音识别中的自监督预训练领域因素的影响进行了研究，结果表明语音学领域因素在预训练中起着重要作用，而语法和句法因素则远不及其重要，这有助于更好地了解自我监督训练语音预训练集的领域特征。

    

    人类语音数据由口音、句法和语义多样性以及声学环境等丰富的领域因素构成。先前的工作探讨了预训练和微调之间领域不匹配对自动语音识别整体表现的影响，但并未分解个别因素的差异贡献。本文提供了一项对受控研究，以更好地了解个别因素对自动语音识别预训练表示性能的影响。为此，我们以单个领域因素进行自然语音或合成音频的预训练，然后在微调后测量性能。结果显示，语音学领域因素在预训练中起着重要作用，而语法和句法因素则远不及其重要。据我们所知，这是第一项研究，以更好地了解自我监督训练语音预训练集的领域特征。

    Human speech data comprises a rich set of domain factors such as accent, syntactic and semantic variety, or acoustic environment. Previous work explores the effect of domain mismatch in automatic speech recognition between pre-training and fine-tuning as a whole but does not dissect the contribution of individual factors. In this paper, we present a controlled study to better understand the effect of such factors on the performance of pre-trained representations on automatic speech recognition. To do so, we pre-train models either on modified natural speech or synthesized audio, with a single domain factor modified, and then measure performance after fine-tuning. Results show that phonetic domain factors play an important role during pre-training while grammatical and syntactic factors are far less important. To our knowledge, this is the first study to better understand the domain characteristics of pre-trained sets in self-supervised pre-training for speech.
    
[^74]: 基于多行多跨度远程监督的表格加文本问题多实例学习

    Multi-Row, Multi-Span Distant Supervision For Table+Text Question. (arXiv:2112.07337v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2112.07337](http://arxiv.org/abs/2112.07337)

    这篇论文提出了一个名为MITQA的基于Transformer的TextTableQA系统，通过多实例学习以及远程监督方法，有效地解决了表格加文本问题的挑战。

    

    近年来，关于表格和链接文本的问答（TextTableQA）已经取得了重要的研究成果，因为表格通常与相关的文本嵌入在文档中。HybridQA和OTT-QA是两个最知名的TextTableQA数据集，其中的问题最好通过同时从表格单元和链接文本段落中获取信息来回答。这两个数据集的共同挑战是，训练实例如问题和答案，其中黄金答案可能不仅匹配跨越表格行的多个表格单元，而且还包括表格行及其相关文本范围内的多个文本跨度。我们提出了MITQA，这是一个基于Transformer的TextTableQA系统，专门设计用于通过多实例损失目标和谨慎的课程设计来应对这两个方面的远程监督。我们的实验表明，所提出的MRMS-DS方法显著提高了MITQA在HybridQA和OTT-QA数据集上的性能，证明了我们的方法在处理TextTableQA挑战方面的有效性。

    Question answering (QA) over tables and linked text, also called TextTableQA, has witnessed significant research in recent years, as tables are often found embedded in documents along with related text. HybridQA and OTT-QA are the two best-known TextTableQA datasets, with questions that are best answered by combining information from both table cells and linked text passages. A common challenge in both datasets, and TextTableQA in general, is that the training instances include just the question and answer, where the gold answer may match not only multiple table cells across table rows but also multiple text spans within the scope of a table row and its associated text. This leads to a noisy multi instance training regime. We present MITQA, a transformer-based TextTableQA system that is explicitly designed to cope with distant supervision along both these axes, through a multi-instance loss objective, together with careful curriculum design. Our experiments show that the proposed multi
    

