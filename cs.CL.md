# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining.](http://arxiv.org/abs/2305.10429) | DoReMi方法使用分组分布式鲁棒优化训练小型代理模型以产生域权重，再使用这些权重重新采样数据集训练大型模型，相比使用默认权重的基线模型，在The Pile和GLaM数据集上平均提高了6.5%和4.7%的few-shot下游准确度，分别使用2.6倍和相同的训练步骤达到基线准确度。 |
| [^2] | [Accelerating Transformer Inference for Translation via Parallel Decoding.](http://arxiv.org/abs/2305.10427) | 通过并行解码，本文提出了一种快速推断Transformer在翻译中的应用的方法，不需要修改现有模型并在保持翻译质量的同时加速了现有模型。 |
| [^3] | [SLiC-HF: Sequence Likelihood Calibration with Human Feedback.](http://arxiv.org/abs/2305.10425) | 本文提出了一种新方法，SLiC-HF，可以利用序列似然校准从人类偏好中学习，相较于过去的方法更加简单高效，并在TL;DR自动摘要任务中显著提高了监督微调基线。 |
| [^4] | [Extracting Blockchain Concepts from Text.](http://arxiv.org/abs/2305.10408) | 本研究旨在通过机器学习模型提取区块链领域的信息并组织，以帮助用户浏览该领域。 |
| [^5] | [BAD: BiAs Detection for Large Language Models in the context of candidate screening.](http://arxiv.org/abs/2305.10407) | 本文介绍了 BAD 模型，通过检测大型语言模型中的偏见来证明在候选人筛选过程中存在的不公平和偏见，以解决人为干预问题。 |
| [^6] | [PaLM 2 Technical Report.](http://arxiv.org/abs/2305.10403) | PaLM 2 是一种计算效率更高的最先进的语言模型，提供了更好的多语言和推理能力，并且通过使用多种目标进行训练，获得了在不同模型大小的下游任务上显着的改进质量。此外，PaLM 2 还展示了强大的推理能力和稳定的性能表现，使得模型能够更广泛地部署，并且可以控制毒性推理时间，而不会对其他能力产生影响。 |
| [^7] | [What You See is What You Read? Improving Text-Image Alignment Evaluation.](http://arxiv.org/abs/2305.10400) | 本研究介绍了SeeTRUE评估集和两种自动文本-图像对齐方法，这些方法在各种对齐任务中均取得了显着改进，在复杂组合或非自然图像的挑战性案例中表现出色。 |
| [^8] | [Elaborative Simplification as Implicit Questions Under Discussion.](http://arxiv.org/abs/2305.10387) | 本文提出了一种将详细阐述视为隐含问题的明确回答的方法，通过引入ElabQUD对作者阐述信息的方式进行了研究。 |
| [^9] | [Logit-Based Ensemble Distribution Distillation for Robust Autoregressive Sequence Uncertainties.](http://arxiv.org/abs/2305.10384) | 本论文介绍了一种基于Logit的集成模型蒸馏方法，能够有效地将知识（epistemic）和数据（aleatoric）不确定性分开，对于大规模自然语言序列到序列的任务能够提高student模型的表现。 |
| [^10] | [Large-Scale Text Analysis Using Generative Language Models: A Case Study in Discovering Public Value Expressions in AI Patents.](http://arxiv.org/abs/2305.10383) | 本文研究使用生成语言模型GPT-4进行大规模文本分析，在US AI专利中发现公共价值表达。采用高级布尔查询收集了154,934个专利文档，并与USPTO的完整专利文本合并。得出5.4百万句子的语料库，使用框架以及GPT-4提示进行标记和理性化。评估结果表明，这种方法很准确。 |
| [^11] | [Evaluating Object Hallucination in Large Vision-Language Models.](http://arxiv.org/abs/2305.10355) | 本研究是对大型视觉-语言模型中的物体幻觉问题进行的第一项系统研究，通过研究发现视觉指令可能影响幻觉，提出新的评估指标成功解决了现有评估方法的不足。 |
| [^12] | [Interactive Learning of Hierarchical Tasks from Dialog with GPT.](http://arxiv.org/abs/2305.10349) | 该论文中，作者提出一种使用GPT模型作为对话前端，从对话中进行可解释的符号交互式任务学习的方法。通过将交互式对话转换为语义表示，并递归地要求未知步骤的定义，可以获取分层任务知识并在自然的对话环境中进行重复使用。 |
| [^13] | [Using a Large Language Model to Control Speaking Style for Expressive TTS.](http://arxiv.org/abs/2305.10321) | 该论文提出了一种使用大型语言模型控制TTS语音表现风格的方法。该方法可为非表现性语料库上的TTS模型提供适当的韵律建议，使其生成表现力更强的语音。 |
| [^14] | [LeTI: Learning to Generate from Textual Interactions.](http://arxiv.org/abs/2305.10314) | LeTI是一种使用自然语言指令、LM生成的程序和错误消息进行串联迭代微调的技术，可以用于代码生成任务，并且在自然发生的Python指令数据集上表现最先进。 |
| [^15] | [FACE: Evaluating Natural Language Generation with Fourier Analysis of Cross-Entropy.](http://arxiv.org/abs/2305.10307) | FACE是一组可以有效识别人类和模型之间差距的度量标准。它基于傅里叶分析和交叉熵估计，可以反映模型大小、解码采样方法和人类评分。 |
| [^16] | [UniEX: An Effective and Efficient Framework for Unified Information Extraction via a Span-extractive Perspective.](http://arxiv.org/abs/2305.10306) | UniEX是一种能适用于各种模式格式的信息抽取框架，并能同时解决命名实体识别、关系抽取、事件提取和情感分析等任务，在性能和推理速度上优于其他通用信息抽取模型。 |
| [^17] | [Towards More Robust NLP System Evaluation: Handling Missing Scores in Benchmarks.](http://arxiv.org/abs/2305.10284) | 本文提出了一种鲁棒的自然语言处理系统评估方法，可以解决基准测试中某些系统的得分缺失问题，并引入了一个规模更大的基准测试。 |
| [^18] | [Chain-of-Symbol Prompting Elicits Planning in Large Langauge Models.](http://arxiv.org/abs/2305.10276) | 本文提出了自然语言规划（NLP）的基准，旨在研究LLMs在需要理解并在文本中相应进行操作的复杂规划任务中的表现。同时提出了一种新方法CoS，使用简化的符号空间表示法来表示复杂的环境。 |
| [^19] | [Boosting Local Spectro-Temporal Features for Speech Analysis.](http://arxiv.org/abs/2305.10270) | 该论文介绍了在语音识别中电话分类的问题，并探索了几组可以用于电话分类的本地谱时特征，提出了使用Haar特征和SVM分类的梯度直方图进行电话分类，并给出了一些初步结果。 |
| [^20] | [Searching for Needles in a Haystack: On the Role of Incidental Bilingualism in PaLM's Translation Capability.](http://arxiv.org/abs/2305.10266) | 本文探究了大型语言模型翻译能力中的意外双语现象，证明了PaLM模型利用意外双语内容可以改善零-shot翻译的准确性。 |
| [^21] | [M3KE: A Massive Multi-Level Multi-Subject Knowledge Evaluation Benchmark for Chinese Large Language Models.](http://arxiv.org/abs/2305.10263) | 本文提出了一种面向中国大语言模型的大规模多级多主题知识评估基准M3KE，收集了20,477个问题以覆盖中国教育体系的所有主要层次和广泛的学科，使用多任务准确性测试法有效地评估了四个大语言模型GPT-2，RoBERTa，ERNIE和ELECTRA对多源知识的整合和利用能力。 |
| [^22] | [MemoryBank: Enhancing Large Language Models with Long-Term Memory.](http://arxiv.org/abs/2305.10250) | MemoryBank 提出了一种新型内存机制，旨在为大型语言模型提供类人的长期记忆。它可以召唤相关记忆，通过持续的记忆更新不断进化，通过合成过去的互动信息理解并适应用户个性。 |
| [^23] | [OpenSLU: A Unified, Modularized, and Extensible Toolkit for Spoken Language Understanding.](http://arxiv.org/abs/2305.10231) | OpenSLU是一个统一、模块化、可扩展的语音理解工具包，将10种针对单意图和多意图场景的语音理解模型统一起来，同时支持非预训练和预训练模型，并高度模块化、可扩展。 |
| [^24] | [Shielded Representations: Protecting Sensitive Attributes Through Iterative Gradient-Based Projection.](http://arxiv.org/abs/2305.10204) | 本文提出了一种名为迭代梯度基础投影（IGBP）的新方法，用于从神经表示中删除非线性编码的概念，以减轻模型的社会偏见。该方法通过迭代训练神经分类器来预测某个敏感属性，然后将表示投影到一个超平面上，使得分类器对目标属性变得无意识。实验证明，该方法在消除敏感属性方面是有效的，并且对下游任务的准确性影响很小。 |
| [^25] | [A Survey on Zero Pronoun Translation.](http://arxiv.org/abs/2305.10196) | 本文总结了零代词翻译（ZPT）领域神经网络全面推广后的重要工作，发现大型语言模型、多任务或迁移学习都可以实现ZPT的性能提升。 |
| [^26] | [Boosting Distress Support Dialogue Responses with Motivational Interviewing Strategy.](http://arxiv.org/abs/2305.10195) | 本文提出利用动机访谈策略重新表达在线心理支持对话中的响应类型，从而提高聊天机器人响应的符合性和质量。 |
| [^27] | [Variable-length Neural Interlingua Representations for Zero-shot Neural Machine Translation.](http://arxiv.org/abs/2305.10190) | 本研究提出一种变长神经中间语表示方法，克服了先前的定长表示方法的限制。在多个数据集上，我们的方法比固定长度中间语表示方法表现更好，但在特定源语言的翻译中效果欠佳。 |
| [^28] | [Knowledge-enhanced Mixed-initiative Dialogue System for Emotional Support Conversations.](http://arxiv.org/abs/2305.10172) | 本文针对情感支持对话系统的混合主动特点，提出了基于知识增强的混合主动对话框架，该框架从大型心理健康知识图谱中检索实际案例知识来生成混合主动响应，并在共情和问题解决能力方面显著优于几个基线模型。 |
| [^29] | [Pragmatic Reasoning in Structured Signaling Games.](http://arxiv.org/abs/2305.10167) | 本文介绍了一个结构化传递博弈和理性言语行为框架的变体sRSA，应用于结构化领域中的实用推理问题。在颜色领域中，我们的研究表明采用sRSA的代理比传统RSA和仅基于强化学习的代理更接近于信息理论界限。 |
| [^30] | [Qualifying Chinese Medical Licensing Examination with Knowledge Enhanced Generative Pre-training Model.](http://arxiv.org/abs/2305.10163) | 本研究通过在ChatGPT中集成医学领域知识和启用少样本学习的新方法，在中国国家医学执业医师资格考试中取得成功，这为建立在自然语言处理技术和医学领域知识的创新应用提供了可能。 |
| [^31] | [Stop Uploading Test Data in Plain Text: Practical Strategies for Mitigating Data Contamination by Evaluation Benchmarks.](http://arxiv.org/abs/2305.10160) | 提出了三个适用策略：（1）公钥加密发布测试数据，仅允许特定派生发布；（2）对于API持有方，要求训练排除控制，保护测试数据，不停止评估直到达到要求；（3）如果测试数据来自互联网文本，需避免某些结果的使用。 |
| [^32] | [Personality Understanding of Fictional Characters during Book Reading.](http://arxiv.org/abs/2305.10156) | 本文提出了一个NLP领域内尚未研究的问题：情景和细致地理解小说人物个性，并提供了第一个标记数据集PersoNet来解决这个问题。 |
| [^33] | [Iterated learning and communication jointly explain efficient color naming systems.](http://arxiv.org/abs/2305.10154) | 本论文通过结合迭代学习和交流的文化进化模型，展示这个模型能够在神经网络中实现并收敛于高效的颜色命名系统，进一步证明了语义系统反映效率压力的观点。 |
| [^34] | [Multi-Grained Knowledge Retrieval for End-to-End Task-Oriented Dialog.](http://arxiv.org/abs/2305.10149) | 该论文提出了一种面向任务的对话系统中的多粒度知识检索方法，该方法通过实体选择器和属性选择器实现知识检索、响应生成解耦，并且使用一种新的蒸馏目标进行监督学习，从而可以更有效地执行知识检索以生成信息响应。 |
| [^35] | [Improving Language Model Negotiation with Self-Play and In-Context Learning from AI Feedback.](http://arxiv.org/abs/2305.10142) | 本文研究了多个大型语言模型能否通过自我博弈和反馈互相提高，在谈判游戏中进行谈判，达成交易。使用历史记录和人工智能反馈迭代改进模型的谈判策略。 |
| [^36] | [Additive manifesto decomposition: A policy domain aware method for understanding party positioning.](http://arxiv.org/abs/2305.10136) | 本文提出了一种理解政党立场的方法，可以估计政策域感知的政党相似性，通过多维缩放提取可解释的主要政策轴上的政党立场。 |
| [^37] | [Empirical Analysis of Oral and Nasal Vowels of Konkani.](http://arxiv.org/abs/2305.10122) | 本研究通过实验分析了康卡尼语口腔和鼻腔元音的声学-语音学特性，为康卡尼语言的语音合成系统和元音的语言学研究提供了帮助。 |
| [^38] | [Use of a Taxonomy of Empathetic Response Intents to Control and Interpret Empathy in Neural Chatbots.](http://arxiv.org/abs/2305.10096) | 本文提出了一种使用移情反应意图分类法来控制和解释神经聊天机器人中的共情回应的方法，能够产生可控和可解释的共情回应。 |
| [^39] | [Probing the Role of Positional Information in Vision-Language Models.](http://arxiv.org/abs/2305.10046) | 本研究调查了在视觉语言模型中位置信息的使用，表明模型存在位置信息，但不能很好地利用它进行图像 - 文本匹配。通过引入位置信息预训练和跨模态匹配的PI对比度学习，在挑战集上成功地改善了模型性能。 |
| [^40] | [Can Language Models Solve Graph Problems in Natural Language?.](http://arxiv.org/abs/2305.10037) | 本论文提出了自然语言图形(NLGraph)，这是一个全面的基于图形问题解决测试，旨在评估LLM在文本描述的图形结构和图形解决方案方面的处理能力。实验结果表明，LLM(GPT-3/4)具有相应的图形推理能力。 |
| [^41] | [Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark.](http://arxiv.org/abs/2305.10036) | 提出了一种名为 EmbMarker 的嵌入式水印方法，用于保护大型语言模型在 EaaS 中的版权。该方法可以在嵌入式上植入后门，并有效地传输和恢复。实验证明，EmbMarker 可以在维护各种 NLP 任务的性能的同时成功保护 EaaS 对 LLM 的版权。 |
| [^42] | [When Gradient Descent Meets Derivative-Free Optimization: A Match Made in Black-Box Scenario.](http://arxiv.org/abs/2305.10013) | 本文介绍了一种新方法GDFO，将梯度下降和无导数优化结合在一起，协调地优化任务特定的连续提示。实验证明，该方法优于现有的无梯度和基于梯度的方法。 |
| [^43] | [AD-KD: Attribution-Driven Knowledge Distillation for Language Model Compression.](http://arxiv.org/abs/2305.10010) | 本文提出了一种基于归因的知识蒸馏方法，通过 Integrated Gradients 探索教师模型的原理，将归因知识转移到学生模型，并在 GLUE 基准测试中使用BERT进行了全面实验，结果表明我们的方法具有更好的性能。 |
| [^44] | [EfficientSCI: Densely Connected Network with Space-time Factorization for Large-scale Video Snapshot Compressive Imaging.](http://arxiv.org/abs/2305.10006) | 本文提出了EfficientSCI网络，通过使用稠密连接和时空分解机制来建立视频SCI中的空间-时间相关性。相比最先进的基于深度学习的方法，它能够在计算效率和重建质量方面取得更好的表现。 |
| [^45] | [DinoSR: Self-Distillation and Online Clustering for Self-supervised Speech Representation Learning.](http://arxiv.org/abs/2305.10005) | 本研究提出了DinoSR模型，它结合了遮蔽语言建模、自蒸馏和在线聚类等概念，能够在自监督语音表示学习任务中产生很好的效果，超越了以前的最新技术水平。 |
| [^46] | [Reprompting: Automated Chain-of-Thought Prompt Inference Through Gibbs Sampling.](http://arxiv.org/abs/2305.09993) | Reprompting是一种无需人类干预的算法，通过迭代采样新配方解决多步推理任务，比人类编写的思维链提示表现更好，还可以提高较弱模型的性能。 |
| [^47] | [Dual Semantic Knowledge Composed Multimodal Dialog Systems.](http://arxiv.org/abs/2305.09990) | 设计了一种双重语义知识组合的多模态任务导向对话系统，通过多级知识组合机制和表示级规范化方法解决了现有研究存在的关键限制，实现了有效响应生成。 |
| [^48] | [Smart Word Suggestions for Writing Assistance.](http://arxiv.org/abs/2305.09975) | 本论文介绍了写作辅助中的“智能词汇建议”（SWS）任务和基准测试，强调了端到端的评估和更加现实的写作辅助场景。它包含一组数据集和提供替换建议的难题，提供了训练和测试基础及潜在的研究方向。 |
| [^49] | [CooK: Empowering General-Purpose Language Models with Modular and Collaborative Knowledge.](http://arxiv.org/abs/2305.09955) | CooK是一种用于赋能通用语言模型的新颖框架，通过专门的语言模型和协作的知识贡献者，提供模块化、不断增长和多源的知识。在知识密集型任务中，CooK展现出了明显的性能提升。 |
| [^50] | ["I'm fully who I am": Towards Centering Transgender and Non-Binary Voices to Measure Biases in Open Language Generation.](http://arxiv.org/abs/2305.09941) | 本论文研究了如何以TGNB人群的声音为中心，评估开放式语言生成中的偏见。通过理解TGNB个体的经历，提出了以TGNB人群为中心的OLG系统评估框架，并且包括一个为TGNB人群设计的调查工具和分析方法。 |
| [^51] | [Equivariant Few-Shot Learning from Pretrained Models.](http://arxiv.org/abs/2305.09900) | 本文提出了一种基于预训练模型的$\lambda$-\textit{equitune}方法，它使用\textit{重要性权重}$\lambda$对特征进行平均，可以显著提高等变小样本学习的表现。 |
| [^52] | [Balancing Lexical and Semantic Quality in Abstractive Summarization.](http://arxiv.org/abs/2305.09898) | 本文提出了一种新的训练方法，其中重新排列硬件平衡了词汇和语义质量，以缓解抽象化摘要中的暴露偏差问题。 |
| [^53] | [Clustering-Aware Negative Sampling for Unsupervised Sentence Representation.](http://arxiv.org/abs/2305.09892) | ClusterNS 是一种将聚类信息引入对比学习进行无监督句子表示学习的新方法，通过改进的 K 均值聚类算法提供难负例并识别错误负例，旨在通过一个统一的框架解决问题，实验结果表明其在无监督句子表示学习中表现优于基线。 |
| [^54] | [Semantic Similarity Measure of Natural Language Text through Machine Learning and a Keyword-Aware Cross-Encoder-Ranking Summarizer -- A Case Study Using UCGIS GIS&T Body of Knowledge.](http://arxiv.org/abs/2305.09877) | 本文提出了一种新方法，采用机器学习模型和关键词感知交叉编码器排序摘要程序，从文本内容中提取语义信息，并度量 GIS&T BoK 话题之间的语义相似度，以解决手动定义话题关系带来的不完整评估问题。该方法在准确度量话题关系方面表现良好，对 GIS&T 领域的研究和实践具有重要意义。 |
| [^55] | [The Jaseci Programming Paradigm and Runtime Stack: Building Scale-out Production Applications Easy and Fast.](http://arxiv.org/abs/2305.09864) | Jaseci编程范式和运行时堆栈的设计原则在于通过自动化和自动优化尽可能多的规模化数据管理、微服务组件化和实时更新复杂性，从而提高了抽象水平，降低了应用程序的开发难度和部署门槛。 |
| [^56] | [Explaining black box text modules in natural language with language models.](http://arxiv.org/abs/2305.09863) | 本文介绍了一种名为Summarize and Score（SASC）的方法，该方法可以自动获取黑盒文本模块的自然语言解释以及解释可靠程度的分数。研究者们已经在合成模块和BERT模型中使用SASC，让我们可以解释模块的选择性，这对于增强大型语言模型的可解释性非常重要。 |
| [^57] | [Epsilon Sampling Rocks: Investigating Sampling Strategies for \\Minimum Bayes Risk Decoding for Machine Translation.](http://arxiv.org/abs/2305.09860) | 本文研究了用于机器翻译最小贝叶斯风险解码的不同采样策略，并发现了epsilon采样方式能够使得解码结果显著地优于其他所有已测试的采样方式和束搜索解码。 |
| [^58] | [Smaller Language Models are Better Black-box Machine-Generated Text Detectors.](http://arxiv.org/abs/2305.09859) | 本文研究发现，小型语言模型更适用于作为通用文本检测器，可以更加精确地检测出机器生成的文本，而检测器和生成模型是否具有相同的架构或语料库并不会对检测性能产生显著影响。 |
| [^59] | [Knowledge Graph Completion Models are Few-shot Learners: An Empirical Study of Relation Labeling in E-commerce with LLMs.](http://arxiv.org/abs/2305.09858) | 本文通过对知识图谱中关系标注的实证研究，发现大型语言模型具有强大的学习能力以及在少量标记数据下预测产品类型之间关系的有效性。 |
| [^60] | [CoEdIT: Text Editing by Task-Specific Instruction Tuning.](http://arxiv.org/abs/2305.09857) | CoEdIT是一种通过任务特定指令调整实现文本编辑的最先进模型，能够提高用户生成文本的质量和提高流程的效率。 |
| [^61] | [CPL-NoViD: Context-Aware Prompt-based Learning for Norm Violation Detection in Online Communities.](http://arxiv.org/abs/2305.09846) | 本文提出了一种新的方法（CPL-NoViD），通过自然语言提示将上下文融入到模型中，用于在线社区中的违规检测。该方法能够适应不同社区中的各种规则和解释的差异，在跨规则类型和跨社区的违规行为检测中表现出色，并在少样本学习场景中表现出一定的适应性。 |
| [^62] | [On Dataset Transferability in Active Learning for Transformers.](http://arxiv.org/abs/2305.09807) | 本文研究了基于transformer的预训练语言模型的主动学习中数据集的可迁移性问题，发现具有相似获取序列的主动学习方法产生的数据集在不同模型之间具有高度的可迁移性。 |
| [^63] | [The Ways of Words: The Impact of Word Choice on Information Engagement and Decision Making.](http://arxiv.org/abs/2305.09798) | 措辞对信息参与和决策制定有重要影响，信息参与是由信息本身的表达所驱动和培育的。 |
| [^64] | [Distilling Semantic Concept Embeddings from Contrastively Fine-Tuned Language Models.](http://arxiv.org/abs/2305.09785) | 本论文通过对比学习策略，提高了语言模型的概念嵌入质量，并在各种基准测试中实现了最先进的结果。 |
| [^65] | [Analysis of Visual Question Answering Algorithms with attention model.](http://arxiv.org/abs/2305.09782) | 本文批评性地检查和审查了使用共同注意力方法的VQA算法的方法，重点关注文本语义生成、对象识别和答案分类技术。 |
| [^66] | [SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification.](http://arxiv.org/abs/2305.09781) | SpecInfer是一种LLM服务系统，通过利用推测推断和令牌树验证来加速生成式大语言模型的推断过程，显著减少了为它们提供服务所需的端到端延迟和计算要求，同时确保模型质量。 |
| [^67] | [ConvXAI: Delivering Heterogeneous AI Explanations via Conversations to Support Human-AI Scientific Writing.](http://arxiv.org/abs/2305.09770) | ConvXAI是一个基于对话的XAI系统，它集成了多种XAI类型，并将实际用户需求嵌入设计中，以提高实用性。 |
| [^68] | [Application-Agnostic Language Modeling for On-Device ASR.](http://arxiv.org/abs/2305.09764) | 本文提出了两种新的前向体系结构用于无应用指导的语言建模，以帮助设备上的自动语音识别系统克服速度、磁盘和内存等限制。 |
| [^69] | [Clinical Note Owns its Hierarchy: Multi-Level Hypergraph Neural Networks for Patient-Level Representation Learning.](http://arxiv.org/abs/2305.09756) | 本文提出了一种基于分类级别的多层超图神经网络（TM-HGNN），通过笔记和分类级别的超边组装有用的中性词和稀有关键词以保留临床语义信息，并在MIMIC-III数据集上取得了优于最先进方法的实验结果。 |
| [^70] | [What In-Context Learning "Learns" In-Context: Disentangling Task Recognition and Task Learning.](http://arxiv.org/abs/2305.09731) | 本研究通过任务识别和任务学习两种方式表征了ICL利用演示的方式，发现LLMs利用不同机制进行任务的解决，TR主要利用先验知识，而TL则具备学习新的输入-标签映射的能力。 |
| [^71] | [Generative Table Pre-training Empowers Models for Tabular Prediction.](http://arxiv.org/abs/2305.09696) | 本文提出了TapTap，一种通过表格预训练生成高质量合成表格来提高表格预测性能的方法。在12个数据集实验中，TapTap在不同场景下优于16个基线，并可以与多个骨干模型结合使用。 |
| [^72] | [OOD-Speech: A Large Bengali Speech Recognition Dataset for Out-of-Distribution Benchmarking.](http://arxiv.org/abs/2305.09688) | OOD-Speech 是用于 Bengali 语音识别的越域基准数据集，由众包收集了母语为 Bengali 的 22,645 名说话者录制的 1177.94 小时语音数据，并经过手动注释。数据集包含 17 种不同的资源，如 Bengali 电视剧、有声读物、脱口秀、在线教学以及伊斯兰讲道等，可作为 Bengali 语音识别的分布变化方面的基准测试数据集。 |
| [^73] | [Satisfiability-Aided Language Models Using Declarative Prompting.](http://arxiv.org/abs/2305.09656) | 本文提出了一种利用自动定理证明器和声明性任务规范的可满足性辅助语言建模方法，可以提高大型语言模型的推理能力。 |
| [^74] | [Life of PII -- A PII Obfuscation Transformer.](http://arxiv.org/abs/2305.09550) | “Life of PII”是一种新颖的混淆变换器框架，用于将PII转化为人造PII同时尽可能地保留原始信息、意图和上下文，使我们能够有选择地混淆文档中的敏感信息，同时保留文档的统计和语义特性。 |
| [^75] | [GIFT: Graph-Induced Fine-Tuning for Multi-Party Conversation Understanding.](http://arxiv.org/abs/2305.09360) | GIFT是一个适用于多方对话理解的方法，通过设计四种类型的边缘将图感知信息集成到注意力机制中，改进了原始的顺序文本处理的PLM。 |
| [^76] | [BERTTM: Leveraging Contextualized Word Embeddings from Pre-trained Language Models for Neural Topic Modeling.](http://arxiv.org/abs/2305.09329) | 本文提出了一种新颖的神经主题模型，利用来自预训练语言模型BERT的上下文化词嵌入，可以在不使用任何BoW信息的情况下推断出文档的主题分布，并直接从上下文化词嵌入中推断出文档中每个单词的主题分布。实验结果表明，该模型优于仅依赖BoW表示和其他神经主题模型的现有最先进方法。 |
| [^77] | [NLG Evaluation Metrics Beyond Correlation Analysis: An Empirical Metric Preference Checklist.](http://arxiv.org/abs/2305.08566) | 本研究提出了一种度量偏好检查表，以超越相关分析评估NLG自动指标，并分析了两种类型的指标及其在三个任务中的效果。 |
| [^78] | [C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models.](http://arxiv.org/abs/2305.08322) | C-Eval是第一个专为中文基础模型评估而设计的全面套件，涵盖52个不同学科的多级别选择题和挑战性科目。评估结果表明，只有GPT-4能够达到超过60％的平均准确率，还有改进空间。 |
| [^79] | [Leveraging Large Language Models in Conversational Recommender Systems.](http://arxiv.org/abs/2305.07961) | 本文提出了一种使用大型语言模型构建端到端大规模对话推荐系统的路线图，解决在该系统中有效利用大型语言模型所面临的技术挑战。 |
| [^80] | [Chain-of-Dictionary Prompting Elicits Translation in Large Language Models.](http://arxiv.org/abs/2305.06575) | 研究通过在大型语言模型中添加字典链提示的方法来改进低资源语言的翻译能力，实验结果表明能显著提高翻译质量。 |
| [^81] | [K-UniMorph: Korean Universal Morphology and its Feature Schema.](http://arxiv.org/abs/2305.06335) | 本文介绍了一种韩语通用词形学数据集，保留韩语特色并采用Sylak-Glassman等人的词形特征模式，为韩语形态学范式领域做出了贡献。 |
| [^82] | [DAMO-NLP at SemEval-2023 Task 2: A Unified Retrieval-augmented System for Multilingual Named Entity Recognition.](http://arxiv.org/abs/2305.03688) | DAMO-NLP团队的U-RaNER是一种统一的多语言命名实体识别系统，它通过加入带有实体为中心的Wikidata知识库并采用infusion方法来增强检索上下文，解决了其他系统存在的知识不足、上下文长度有限和单一检索策略等问题。 |
| [^83] | [The Politics of Language Choice: How the Russian-Ukrainian War Influences Ukrainians' Language Use on Twitter.](http://arxiv.org/abs/2305.02770) | 本文研究了俄乌战争期间乌克兰人在 Twitter 上的语言使用，发现在战争爆发前已经出现从俄语向乌克兰语转变的趋势，而战争爆发后这种趋势加速了，并且许多使用俄语的用户在战争期间转变成使用乌克兰语。 |
| [^84] | [Lift Yourself Up: Retrieval-augmented Text Generation with Self Memory.](http://arxiv.org/abs/2305.02437) | 本文提出了一种新的检索增强文本生成模型Selfmem，通过迭代生成自我记忆池并采用记忆选择器，使检索更加自适应，提高了文本生成的质量和多样性。 |
| [^85] | [Don't Stop Pretraining? Make Prompt-based Fine-tuning Powerful Learner.](http://arxiv.org/abs/2305.01711) | 本文研究了持续预训练对于微调性能的影响，发现传统的持续预训练不能保证一致的提高性能，甚至会对一些任务产生负面影响。针对这些问题，作者提出了基于提示的持续预训练，旨在通过无监督的预训练向LM展示任务相关文本和提示模板，从而提高基于提示的微调表现。 |
| [^86] | [Classification of US Supreme Court Cases using BERT-Based Techniques.](http://arxiv.org/abs/2304.08649) | 本文基于BERT技术探究了对美国最高法院案例进行分类的方法，比较了使用BERT模型与其他先进模型的准确性，最终在15个广泛类别上取得了80%的准确度，在279个细粒度类别上取得了60%的准确度。 |
| [^87] | [RPTQ: Reorder-based Post-training Quantization for Large Language Models.](http://arxiv.org/abs/2304.01089) | 本研究提出了一种新的基于重排的量化方法RPTQ，目的是解决大型语言模型在量化时由于信道激活范围不同而产生的问题。实现该方法后，我们将LLL模型推动到3位激活。 |
| [^88] | [UKP-SQuARE v3: A Platform for Multi-Agent QA Research.](http://arxiv.org/abs/2303.18120) | UKP-SQuARE v3是一个支持多智能体系统的QA研究平台，与多数据集模型相比，结合专家智能体可以获得更好的性能提升。 |
| [^89] | [Positive-Augmented Constrastive Learning for Image and Video Captioning Evaluation.](http://arxiv.org/abs/2303.12112) | 本论文提出一种新的图像标题评估指标PAC-S，可以更准确地评估图像和视频的标题，相比于现有的指标有更好的表现；源代码和训练模型已经公开。 |
| [^90] | [Incorporating Knowledge into Document Summarisation: an Application of Prefix-Tuning on GPT-2.](http://arxiv.org/abs/2301.11719) | 本论文研究了将事实知识纳入生成的摘要的可能性，具体采用前缀调整的方法，实验结果表明，此方法可以生成保留知识的摘要，而且可以提升整体性能。 |
| [^91] | [An Inclusive Notion of Text.](http://arxiv.org/abs/2211.05604) | 我们提出了一个通用术语以及一个语言和非语言元素的分类法，用于系统性地捕捉不同任务和数据下的文本概念差异。这显著提高了NLP模型的可重复性和可推广性。 |
| [^92] | [KGLM: Integrating Knowledge Graph Structure in Language Models for Link Prediction.](http://arxiv.org/abs/2211.02744) | 本文提出了 KGLM 架构，将新的实体/关系嵌入层整合进语言模型，使其能够学习知识图谱的结构。使用从知识图谱提取的三元组对模型进行预训练并进行链接预测任务得到了良好效果。 |
| [^93] | [Pruning Pre-trained Language Models Without Fine-Tuning.](http://arxiv.org/abs/2210.06210) | 本文提出了静态模型剪枝（SMP），它只使用一阶剪枝来适应下游任务，同时实现目标稀疏度水平，在大量实验证明SMP具有显著的改进。 |
| [^94] | [Mining Legal Arguments in Court Decisions.](http://arxiv.org/abs/2208.06178) | 该论文针对法律领域论点挖掘存在的问题，设计了深入法律论证研究理论和实践的新的注释方案，并编译了包含373项裁决的大型语料库。 |

# 详细

[^1]: DoReMi: 优化数据混合加速语言模型预训练

    DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining. (arXiv:2305.10429v1 [cs.CL])

    [http://arxiv.org/abs/2305.10429](http://arxiv.org/abs/2305.10429)

    DoReMi方法使用分组分布式鲁棒优化训练小型代理模型以产生域权重，再使用这些权重重新采样数据集训练大型模型，相比使用默认权重的基线模型，在The Pile和GLaM数据集上平均提高了6.5%和4.7%的few-shot下游准确度，分别使用2.6倍和相同的训练步骤达到基线准确度。

    

    预训练数据域的混合比例（例如，维基百科、图书、网页文本）极大地影响语言模型（LM）性能。在本文中，我们提出了一种称为DoReMi的Domain Reweighting with Minimax Optimization方法，它首先使用分组分布式鲁棒优化（Group DRO）训练一个小代理模型，以产生域权重（混合比例），而不需要知道下游任务的知识。然后我们使用这些域权重重新采样一个数据集，并训练一个更大的，全尺寸的模型。在我们的实验中，我们使用DoReMi在一个280M参数的代理模型上，更有效地找到训练一个8B参数模型（30倍大）的域权重。在The Pile上，即使在减小一些域的比重时，DoReMi也能提高所有域的perplexity。相比使用The Pile的默认域权重训练的基线模型，DoReMi将平均few-shot下游准确度提高了6.5%，并使用2.6倍的训练步骤达到基线准确度。在GLaM数据集上，DoReMi没有任何关于下游任务的知识，提高了4.7%（次于现有最先进的模型）的few-shot准确度，在相同的训练步骤下提高了9.0%的准确度。

    The mixture proportions of pretraining data domains (e.g., Wikipedia, books, web text) greatly affect language model (LM) performance. In this paper, we propose Domain Reweighting with Minimax Optimization (DoReMi), which first trains a small proxy model using group distributionally robust optimization (Group DRO) over domains to produce domain weights (mixture proportions) without knowledge of downstream tasks. We then resample a dataset with these domain weights and train a larger, full-sized model. In our experiments, we use DoReMi on a 280M-parameter proxy model to find domain weights for training an 8B-parameter model (30x larger) more efficiently. On The Pile, DoReMi improves perplexity across all domains, even when it downweights a domain. DoReMi improves average few-shot downstream accuracy by 6.5% over a baseline model trained using The Pile's default domain weights and reaches the baseline accuracy with 2.6x fewer training steps. On the GLaM dataset, DoReMi, which has no know
    
[^2]: 并行解码加速Transformer在翻译中的应用

    Accelerating Transformer Inference for Translation via Parallel Decoding. (arXiv:2305.10427v1 [cs.CL])

    [http://arxiv.org/abs/2305.10427](http://arxiv.org/abs/2305.10427)

    通过并行解码，本文提出了一种快速推断Transformer在翻译中的应用的方法，不需要修改现有模型并在保持翻译质量的同时加速了现有模型。

    

    自回归解码限制了Transformer在机器翻译中的效率。社区提出了特定的网络架构和基于学习的方法来解决这个问题，但它们都很昂贵并且需要改变机器翻译模型，以推导出解码速度和翻译质量之间的平衡。本文从解码算法的角度提出了一个解决方案，将标准的贪心自回归解码转化为并行解码，并利用雅克比和高斯-塞德尔迭代方法实现快速推断。该算法不需要修改现有模型，并在保持翻译质量的同时加速了现有模型。我们提出了三种并行解码算法，并在不同语言和模型上进行了测试，证明并行化解码相对于标准自回归解码可提高达38％的速度，当扩展模型时，速度几乎提高了2倍。

    Autoregressive decoding limits the efficiency of transformers for Machine Translation (MT). The community proposed specific network architectures and learning-based methods to solve this issue, which are expensive and require changes to the MT model, trading inference speed at the cost of the translation quality. In this paper, we propose to address the problem from the point of view of decoding algorithms, as a less explored but rather compelling direction. We propose to reframe the standard greedy autoregressive decoding of MT with a parallel formulation leveraging Jacobi and Gauss-Seidel fixed-point iteration methods for fast inference. This formulation allows to speed up existing models without training or modifications while retaining translation quality. We present three parallel decoding algorithms and test them on different languages and models showing how the parallelization introduces a speedup up to 38% w.r.t. the standard autoregressive decoding and nearly 2x when scaling t
    
[^3]: SLiC-HF：人类反馈的序列似然校准

    SLiC-HF: Sequence Likelihood Calibration with Human Feedback. (arXiv:2305.10425v1 [cs.CL])

    [http://arxiv.org/abs/2305.10425](http://arxiv.org/abs/2305.10425)

    本文提出了一种新方法，SLiC-HF，可以利用序列似然校准从人类偏好中学习，相较于过去的方法更加简单高效，并在TL;DR自动摘要任务中显著提高了监督微调基线。

    

    已经证明，从人类反馈中学习可以有效地将语言模型与人类偏好对齐。过去的工作通常依赖于从人类偏好数据训练的奖励模型分配的奖励分数，利用人类反馈进行强化学习（RLHF）来优化语言模型。在本文中，我们展示了最近引入的序列似然校准（SLiC）如何有效地应用于从人类偏好中学习（SLiC-HF）。此外，我们证明这可以使用为不同模型收集的人类反馈数据来完成，类似于离线RL数据的离线学习。自动化和人类评估实验表明，SLiC-HF显著改进了监督微调基线。此外，SLiC-HF是过去工作中使用的PPO RLHF实现的竞争性替代，而且在实践中更简单、更易于调整，并具有更高的计算效率。

    Learning from human feedback has been shown to be effective at aligning language models with human preferences. Past work has often relied on Reinforcement Learning from Human Feedback (RLHF), which optimizes the language model using reward scores assigned from a reward model trained on human preference data. In this work we show how the recently introduced Sequence Likelihood Calibration (SLiC), can also be used to effectively learn from human preferences (SLiC-HF). Furthermore, we demonstrate this can be done with human feedback data collected for a different model, similar to off-policy, offline RL data. Automatic and human evaluation experiments on the TL;DR summarization task show that SLiC-HF significantly improves supervised fine-tuning baselines. Furthermore, SLiC-HF presents a competitive alternative to the PPO RLHF implementation used in past work while being much simpler to implement, easier to tune and more computationally efficient in practice.
    
[^4]: 从文本中提取区块链概念

    Extracting Blockchain Concepts from Text. (arXiv:2305.10408v1 [cs.IR])

    [http://arxiv.org/abs/2305.10408](http://arxiv.org/abs/2305.10408)

    本研究旨在通过机器学习模型提取区块链领域的信息并组织，以帮助用户浏览该领域。

    

    区块链提供了一种机制，通过该机制，相互不信任的远程方可以就信息分类账的状态达成共识。随着这个领域的快速发展，需要学习区块链的人也越来越多。由于这是一个技术性的主题，开始学习可能会感到相当不可思议。因此，该项目的主要目标是应用机器学习模型从白皮书和学术论文中提取关于区块链领域的信息，以组织这些信息并帮助用户浏览该领域。

    Blockchains provide a mechanism through which mutually distrustful remote parties can reach consensus on the state of a ledger of information. With the great acceleration with which this space is developed, the demand for those seeking to learn about blockchain also grows. Being a technical subject, it can be quite intimidating to start learning. For this reason, the main objective of this project was to apply machine learning models to extract information from whitepapers and academic articles focused on the blockchain area to organize this information and aid users to navigate the space.
    
[^5]: 在候选人筛选中进行大型语言模型的偏倚检测：以 BAD 模型为例

    BAD: BiAs Detection for Large Language Models in the context of candidate screening. (arXiv:2305.10407v1 [cs.CL])

    [http://arxiv.org/abs/2305.10407](http://arxiv.org/abs/2305.10407)

    本文介绍了 BAD 模型，通过检测大型语言模型中的偏见来证明在候选人筛选过程中存在的不公平和偏见，以解决人为干预问题。

    

    应用跟踪系统（ATS）使得人才经理、招聘人员和大学招生委员会能够高效地处理大量的候选人申请。传统上，这个筛选过程是手工进行的，由于申请数的数量，存在很多瓶颈问题，并引入了许多的人为偏见。随着 ChatGPT 等大型语言模型（LLMs）的推出以及将方法应用到当前的自动化应用筛选中，这导致了进一步的偏见和公平性问题需要解决。在这个项目中，我们希望在候选人筛选的背景下，识别和量化 ChatGPT 和其他 OpenAI LLMs 中的社会偏见，以证明使用这些模型可能会延续现有的偏见和不平等在招聘过程中存在的问题。

    Application Tracking Systems (ATS) have allowed talent managers, recruiters, and college admissions committees to process large volumes of potential candidate applications efficiently. Traditionally, this screening process was conducted manually, creating major bottlenecks due to the quantity of applications and introducing many instances of human bias. The advent of large language models (LLMs) such as ChatGPT and the potential of adopting methods to current automated application screening raises additional bias and fairness issues that must be addressed. In this project, we wish to identify and quantify the instances of social bias in ChatGPT and other OpenAI LLMs in the context of candidate screening in order to demonstrate how the use of these models could perpetuate existing biases and inequalities in the hiring process.
    
[^6]: PaLM 2 技术报告

    PaLM 2 Technical Report. (arXiv:2305.10403v1 [cs.CL])

    [http://arxiv.org/abs/2305.10403](http://arxiv.org/abs/2305.10403)

    PaLM 2 是一种计算效率更高的最先进的语言模型，提供了更好的多语言和推理能力，并且通过使用多种目标进行训练，获得了在不同模型大小的下游任务上显着的改进质量。此外，PaLM 2 还展示了强大的推理能力和稳定的性能表现，使得模型能够更广泛地部署，并且可以控制毒性推理时间，而不会对其他能力产生影响。

    

    我们介绍了 PaLM 2，这是一种新的最先进的语言模型，比其前身 PaLM 在多语言和推理能力方面更加出色，并且计算效率更高。PaLM 2 是一种基于 Transformer 的模型，使用多种目标进行训练。通过对英语和多语言语言以及推理任务的广泛评估，我们展示了 PaLM 2 在不同模型大小的下游任务上具有显着的改进质量，同时展现了比 PaLM 更快和更有效的推理能力。这种改进的效率使得模型能够更广泛地部署，同时也使得模型能够更快地响应，以获得更自然的交互节奏。PaLM 2 展示了强大的推理能力，在 BIG-Bench 和其他推理任务上相对于 PaLM 有巨大的改进。PaLM 2 在一套负责人的 AI 评估中表现出稳定的性能，并且在没有附加运行开销或对其他能力产生影响的情况下，能够对毒性进行推理时间的控制。

    We introduce PaLM 2, a new state-of-the-art language model that has better multilingual and reasoning capabilities and is more compute-efficient than its predecessor PaLM. PaLM 2 is a Transformer-based model trained using a mixture of objectives. Through extensive evaluations on English and multilingual language, and reasoning tasks, we demonstrate that PaLM 2 has significantly improved quality on downstream tasks across different model sizes, while simultaneously exhibiting faster and more efficient inference compared to PaLM. This improved efficiency enables broader deployment while also allowing the model to respond faster, for a more natural pace of interaction. PaLM 2 demonstrates robust reasoning capabilities exemplified by large improvements over PaLM on BIG-Bench and other reasoning tasks. PaLM 2 exhibits stable performance on a suite of responsible AI evaluations, and enables inference-time control over toxicity without additional overhead or impact on other capabilities. Over
    
[^7]: 你看到的就是你读到的? 改进文本-图像对齐评估方法

    What You See is What You Read? Improving Text-Image Alignment Evaluation. (arXiv:2305.10400v1 [cs.CL])

    [http://arxiv.org/abs/2305.10400](http://arxiv.org/abs/2305.10400)

    本研究介绍了SeeTRUE评估集和两种自动文本-图像对齐方法，这些方法在各种对齐任务中均取得了显着改进，在复杂组合或非自然图像的挑战性案例中表现出色。

    

    自动确定文本和相应的图像是否语义上对齐是视觉语言模型面临的一项重要挑战，应用于生成文本到图像和图像到文本任务。在本研究中，我们研究了自动文本-图像对齐评估方法。我们首先介绍了SeeTRUE：一个全面的评估集，涵盖了从文本到图像和图像到文本生成任务的多个数据集，并具有人类的判断，判断给定的文本-图像对是否语义上对齐。然后，我们描述了两种自动确定对齐的方法：第一种是基于问题生成和视觉问题回答模型的管道，第二种是通过微调多模态预训练模型的端到端分类方法。这两种方法在各种文本-图像对齐任务中均超越了先前的方法，在涉及复杂组合或非自然图像的挑战性案例中有显着改进。最后，我们证明即使最先进的模型在这个任务上还有很大的改进空间，这激励了未来在这个领域的研究。

    Automatically determining whether a text and a corresponding image are semantically aligned is a significant challenge for vision-language models, with applications in generative text-to-image and image-to-text tasks. In this work, we study methods for automatic text-image alignment evaluation. We first introduce SeeTRUE: a comprehensive evaluation set, spanning multiple datasets from both text-to-image and image-to-text generation tasks, with human judgements for whether a given text-image pair is semantically aligned. We then describe two automatic methods to determine alignment: the first involving a pipeline based on question generation and visual question answering models, and the second employing an end-to-end classification approach by finetuning multimodal pretrained models. Both methods surpass prior approaches in various text-image alignment tasks, with significant improvements in challenging cases that involve complex composition or unnatural images. Finally, we demonstrate 
    
[^8]: 作为隐含讨论问题的详细简化

    Elaborative Simplification as Implicit Questions Under Discussion. (arXiv:2305.10387v1 [cs.CL])

    [http://arxiv.org/abs/2305.10387](http://arxiv.org/abs/2305.10387)

    本文提出了一种将详细阐述视为隐含问题的明确回答的方法，通过引入ElabQUD对作者阐述信息的方式进行了研究。

    

    自动文本简化通常被认为是编码器-解码器模型下从复杂句子到简化句子的单语翻译工作，有助于使文本更易于让儿童和新兴双语者理解。然而，这种观点忽略了详细简化的情况，即在简化文本中添加新信息的情况。本文提出将详细简化视为讨论问题框架（QUD）的一部分，将详细阐述的信息视为对隐含问题的明确回答，从而提供了一种研究作者阐述哪些信息、如何阐述以及阐述将如何适应话语背景的强大方法。我们引入了ElabQUD，其中包括1.3K的详细阐述和隐含的QUD，以研究这些现象。

    Automated text simplification, a technique useful for making text more accessible to people such as children and emergent bilinguals, is often thought of as a monolingual translation task from complex sentences to simplified sentences using encoder-decoder models. This view fails to account for elaborative simplification, where new information is added into the simplified text. This paper proposes to view elaborative simplification through the lens of the Question Under Discussion (QUD) framework, providing a robust way to investigate what writers elaborate upon, how they elaborate, and how elaborations fit into the discourse context by viewing elaborations as explicit answers to implicit questions. We introduce ElabQUD, consisting of 1.3K elaborations accompanied with implicit QUDs, to study these phenomena. We show that explicitly modeling QUD (via question generation) not only provides essential understanding of elaborative simplification and how the elaborations connect with the re
    
[^9]: 基于Logit的集成分布蒸馏在自回归序列的不确定性中的应用

    Logit-Based Ensemble Distribution Distillation for Robust Autoregressive Sequence Uncertainties. (arXiv:2305.10384v1 [cs.LG])

    [http://arxiv.org/abs/2305.10384](http://arxiv.org/abs/2305.10384)

    本论文介绍了一种基于Logit的集成模型蒸馏方法，能够有效地将知识（epistemic）和数据（aleatoric）不确定性分开，对于大规模自然语言序列到序列的任务能够提高student模型的表现。

    

    高效可靠地估计不确定性是深度学习的一个重要目标，特别是在训练和推理成本通常非常高的自回归序列任务中。本文研究了应用于大规模自然语言序列到序列数据的集成分布蒸馏（Ensemble Distribution Distillation，EDD）方法。EDD旨在将昂贵的（teacher）集成模型的优越不确定性性能压缩到更便宜的（student）单一模型中。重要的是，它保留了将知识（认知）和数据（随机）不确定性分开的能力。现有的概率空间方法对于大词汇量的任务来说不易扩展。我们表明，在大规模翻译任务的现代Transformers模型中，对集成模型的logits进行建模比对softmax概率进行建模，能够显著提高student模型的表现。

    Efficiently and reliably estimating uncertainty is an important objective in deep learning. It is especially pertinent to autoregressive sequence tasks, where training and inference costs are typically very high. However, existing research has predominantly focused on tasks with static data such as image classification. In this work, we investigate Ensemble Distribution Distillation (EDD) applied to large-scale natural language sequence-to-sequence data. EDD aims to compress the superior uncertainty performance of an expensive (teacher) ensemble into a cheaper (student) single model. Importantly, the ability to separate knowledge (epistemic) and data (aleatoric) uncertainty is retained. Existing probability-space approaches to EDD, however, are difficult to scale to large vocabularies. We show, for modern transformer architectures on large-scale translation tasks, that modelling the ensemble logits, instead of softmax probabilities, leads to significantly better students. Moreover, the
    
[^10]: 使用生成语言模型进行大规模文本分析：在AI专利中发现公共价值表达的案例研究

    Large-Scale Text Analysis Using Generative Language Models: A Case Study in Discovering Public Value Expressions in AI Patents. (arXiv:2305.10383v1 [cs.CL])

    [http://arxiv.org/abs/2305.10383](http://arxiv.org/abs/2305.10383)

    本文研究使用生成语言模型GPT-4进行大规模文本分析，在US AI专利中发现公共价值表达。采用高级布尔查询收集了154,934个专利文档，并与USPTO的完整专利文本合并。得出5.4百万句子的语料库，使用框架以及GPT-4提示进行标记和理性化。评估结果表明，这种方法很准确。

    

    标记数据对于训练文本分类器至关重要，但对于复杂和抽象的概念而言，准确标记常常很难实现。本文采用一种新颖方法，使用生成语言模型（GPT-4）进行大规模文本分析的标记和理性化。我们将这种方法应用于在美国AI专利中发现公共价值表达的任务上。我们使用在InnovationQ+上提交的高级布尔查询收集了一个包含154,934个专利文档的数据库，这些结果与来自USPTO的完整专利文本合并，总计5.4百万句子。我们设计了一个框架来识别和标记这些AI专利句子中的公共价值表达。我们开发了GPT-4的提示，其中包括文本分类的定义、指导方针、示例和理性化。我们使用BLEU分数和主题建模评估了GPT-4生成的标签和理性化的质量，并发现它们是准确的。

    Labeling data is essential for training text classifiers but is often difficult to accomplish accurately, especially for complex and abstract concepts. Seeking an improved method, this paper employs a novel approach using a generative language model (GPT-4) to produce labels and rationales for large-scale text analysis. We apply this approach to the task of discovering public value expressions in US AI patents. We collect a database comprising 154,934 patent documents using an advanced Boolean query submitted to InnovationQ+. The results are merged with full patent text from the USPTO, resulting in 5.4 million sentences. We design a framework for identifying and labeling public value expressions in these AI patent sentences. A prompt for GPT-4 is developed which includes definitions, guidelines, examples, and rationales for text classification. We evaluate the quality of the labels and rationales produced by GPT-4 using BLEU scores and topic modeling and find that they are accurate, di
    
[^11]: 大型视觉-语言模型中的物体幻觉评估

    Evaluating Object Hallucination in Large Vision-Language Models. (arXiv:2305.10355v1 [cs.CV])

    [http://arxiv.org/abs/2305.10355](http://arxiv.org/abs/2305.10355)

    本研究是对大型视觉-语言模型中的物体幻觉问题进行的第一项系统研究，通过研究发现视觉指令可能影响幻觉，提出新的评估指标成功解决了现有评估方法的不足。

    

    发掘大型语言模型(LLM)因为其出色的语言能力近来已经开始研究大型视觉-语言模型(LVLM)，并将强大的LLM集成于LVLM中，以提高LVLM在复杂的多模态任务中的表现。虽然LVLM取得了很大进步，但是本研究发现LVLM存在长度幻觉问题，即它们倾向于生成与目标图像不一致的物体描述。为了调查这个问题，本研究开展了第一项系统研究，评估了LVLM中的物体幻觉。我们对几个代表性的LVLM进行了评估实验，并表明它们大多数都存在严重的物体幻觉问题。我们进一步探讨了视觉指令可能会影响幻觉，并发现在视觉指令中经常出现或与图像中的物体共现的物体，更容易被LVLM产生幻觉。此外，我们发现现有的评估方法可能会受到输入指令的影响，不能足以识别物体幻觉。为了解决这个问题，我们提出了一种新的评估指标，可以有效地评估物体幻觉问题。实验结果表明，我们提出的指标不仅可以有效地识别物体幻觉问题，还可以提供有关幻觉问题出现位置和如何缓解它的见解。

    Inspired by the superior language abilities of large language models (LLM), large vision-language models (LVLM) have been recently explored by integrating powerful LLMs for improving the performance on complex multimodal tasks. Despite the promising progress on LVLMs, we find that LVLMs suffer from the hallucination problem, i.e. they tend to generate objects that are inconsistent with the target images in the descriptions. To investigate it, this work presents the first systematic study on object hallucination of LVLMs. We conduct the evaluation experiments on several representative LVLMs, and show that they mostly suffer from severe object hallucination issue. We further discuss that the visual instructions may influence the hallucination, and find that: objects that frequently occur in the visual instructions or co-occur with the image objects, are obviously prone to be hallucinated by LVLMs. Besides, we find that existing evaluation methods might be affected by the input instructio
    
[^12]: 使用GPT从对话中交互学习分层任务

    Interactive Learning of Hierarchical Tasks from Dialog with GPT. (arXiv:2305.10349v1 [cs.HC])

    [http://arxiv.org/abs/2305.10349](http://arxiv.org/abs/2305.10349)

    该论文中，作者提出一种使用GPT模型作为对话前端，从对话中进行可解释的符号交互式任务学习的方法。通过将交互式对话转换为语义表示，并递归地要求未知步骤的定义，可以获取分层任务知识并在自然的对话环境中进行重复使用。

    

    我们提出了一个系统，利用GPT模型作为对话前端，从对话中进行可解释的符号交互式任务学习。学习到的任务被表示为谓词-参数结构的分层分解，具有作用域变量参数。通过使用GPT模型将交互式对话转换为语义表示，并递归地要求未知步骤的定义，我们展示了分层任务知识可以在自然和不受限制的对话环境中被获取和重复使用。我们将我们的系统与使用更传统的解析器的类似架构进行比较，并展示了我们的系统可以容忍更广泛的语言变异。

    We present a system for interpretable, symbolic, interactive task learning from dialog using a GPT model as a conversational front-end. The learned tasks are represented as hierarchical decompositions of predicate-argument structures with scoped variable arguments. By using a GPT model to convert interactive dialog into a semantic representation, and then recursively asking for definitions of unknown steps, we show that hierarchical task knowledge can be acquired and re-used in a natural and unrestrained conversational environment. We compare our system to a similar architecture using a more conventional parser and show that our system tolerates a much wider variety of linguistic variance.
    
[^13]: 使用大型语言模型控制说话风格以实现表现性文本到语音合成

    Using a Large Language Model to Control Speaking Style for Expressive TTS. (arXiv:2305.10321v1 [cs.CL])

    [http://arxiv.org/abs/2305.10321](http://arxiv.org/abs/2305.10321)

    该论文提出了一种使用大型语言模型控制TTS语音表现风格的方法。该方法可为非表现性语料库上的TTS模型提供适当的韵律建议，使其生成表现力更强的语音。

    

    恰当的韵律对于成功的口头交流至关重要。上下文词嵌入已被证明在预测韵律方面有所帮助，但不允许在可能的韵律演绎之间进行选择。基于参考语音的TTS模型试图通过在参考语音样本基础上生成语音来解决这个问题。这些模型可以生成富有表现力的语音，但需要找到适当的参考样本。已经使用足够大的生成语言模型来解决各种与语言相关的任务。我们探讨了这样的模型是否可以用于建议适当的韵律以实现表现性TTS。我们在非表现性语料库上训练TTS模型，然后提示语言模型建议更改音调、能量和持续时间。提示可以为任何任务设计，并根据目标说话风格和对话上下文提示模型进行建议。与基线模型的31.0％相比，所提出的方法在49.9％的情况下被评为最合适的方法。

    Appropriate prosody is critical for successful spoken communication. Contextual word embeddings are proven to be helpful in predicting prosody but do not allow for choosing between plausible prosodic renditions. Reference-based TTS models attempt to address this by conditioning speech generation on a reference speech sample. These models can generate expressive speech but this requires finding an appropriate reference.  Sufficiently large generative language models have been used to solve various language-related tasks. We explore whether such models can be used to suggest appropriate prosody for expressive TTS. We train a TTS model on a non-expressive corpus and then prompt the language model to suggest changes to pitch, energy and duration. The prompt can be designed for any task and we prompt the model to make suggestions based on target speaking style and dialogue context. The proposed method is rated most appropriate in 49.9\% of cases compared to 31.0\% for a baseline model.
    
[^14]: LeTI：从文本交互中学习生成

    LeTI: Learning to Generate from Textual Interactions. (arXiv:2305.10314v1 [cs.CL])

    [http://arxiv.org/abs/2305.10314](http://arxiv.org/abs/2305.10314)

    LeTI是一种使用自然语言指令、LM生成的程序和错误消息进行串联迭代微调的技术，可以用于代码生成任务，并且在自然发生的Python指令数据集上表现最先进。

    

    微调预训练语言模型(LM)可以增强模型的能力。先前的技术通过输入输出对（例如指令微调）或用评估输出质量的数字奖励（例如从人类反馈中进行的强化学习）对预训练的LM进行微调。我们探索了LM从文本交互中学习的潜力(LeTI)，这不仅可以通过二进制标签检查其正确性，而且还可以通过文本反馈指出和解释其输出中的错误。我们的研究重点是代码生成任务，其中模型根据自然语言指令生成代码片段。这种设置可以自然且可扩展地获取文本反馈：使用Python解释器进行代码执行时的错误消息和堆栈跟踪。 LeTI使用LM目标对自然语言指令、LM生成的程序和文本反馈进行串联的迭代微调，只有在生成代码无法执行时才提供文本反馈。我们在一个包含58k个自然发生的Python指令，增加了错误消息和堆栈跟踪的数据集上评估了LeTI，在三种不同的评估指标上显著优于强基线模型，并取得了最先进的结果。

    Finetuning pre-trained language models (LMs) enhances the models' capabilities. Prior techniques fine-tune a pre-trained LM on input-output pairs (e.g., instruction fine-tuning), or with numerical rewards that gauge the quality of its outputs (e.g., reinforcement learning from human feedback). We explore LMs' potential to learn from textual interactions (LeTI) that not only check their correctness with binary labels, but also pinpoint and explain errors in their outputs through textual feedback. Our investigation focuses on the code generation task, where the model produces code pieces in response to natural language instructions. This setting invites a natural and scalable way to acquire the textual feedback: the error messages and stack traces from code execution using a Python interpreter. LeTI iteratively fine-tunes the model, using the LM objective, on a concatenation of natural language instructions, LM-generated programs, and textual feedback, which is only provided when the gen
    
[^15]: FACE: 使用交叉熵的傅里叶分析评估自然语言生成

    FACE: Evaluating Natural Language Generation with Fourier Analysis of Cross-Entropy. (arXiv:2305.10307v1 [cs.CL])

    [http://arxiv.org/abs/2305.10307](http://arxiv.org/abs/2305.10307)

    FACE是一组可以有效识别人类和模型之间差距的度量标准。它基于傅里叶分析和交叉熵估计，可以反映模型大小、解码采样方法和人类评分。

    

    评估机器生成的语言与人类语言之间的距离是一个至关重要的问题。受到语言学心理学关于语言熵周期性实证发现的启示，我们提出了FACE——一组基于语言交叉熵的傅里叶分析的度量，用于衡量生成模型产生的语言与人类书写语言之间的相似度。通过一个开放式的生成任务和以前研究的实验数据，我们发现FACE可以有效地识别人类模型差距，在模型规模上有所缩放，反映了不同解码采样方法的结果，与其他评估指标和人类判断分数相关良好。FACE在计算上是高效的，并提供直观的解释。

    Measuring the distance between machine-produced and human language is acritical open problem. Inspired by empirical findings from psycholinguistics on theperiodicity of entropy in language, we propose FACE, a set of metrics based onFourier Analysis of the estimated Cross-Entropy of language, for measuring thesimilarity between model-generated and human-written languages. Based on anopen-ended generation task and the experimental data from previous studies, weind that FACE can effectively identify the human-model gap, scales with modelsize, reflects the outcomes of different sampling methods for decoding, correlateswell with other evaluation metrics and with human judgment scores. FACE iscomputationally efficient and provides intuitive interpretations.
    
[^16]: UniEX：一种基于跨度提取的统一信息抽取的有效高效框架

    UniEX: An Effective and Efficient Framework for Unified Information Extraction via a Span-extractive Perspective. (arXiv:2305.10306v1 [cs.CL])

    [http://arxiv.org/abs/2305.10306](http://arxiv.org/abs/2305.10306)

    UniEX是一种能适用于各种模式格式的信息抽取框架，并能同时解决命名实体识别、关系抽取、事件提取和情感分析等任务，在性能和推理速度上优于其他通用信息抽取模型。

    

    我们提出了一种新的通用信息抽取范式，它与任何模式格式兼容，并适用于一系列信息抽取任务，如命名实体识别、关系抽取、事件提取和情感分析。我们的方法将以文本为基础的信息抽取任务转化为 token-pair 问题，使用一种统一的提取框架 UniEX，将所有提取目标都统一分解为联合跨度检测、分类和关联问题。UniEX 可以同时编码基于模式的提示和文本信息，并使用自动编码器语言模型协同学习预定义信息的广义知识。我们开发了 traffine 注意机制，将包括任务、标签和内部 token 在内的异构因素集成起来，并通过评分矩阵获得提取目标。实验结果表明，UniEX 在 $14$个基准测试数据集上的表现和推理速度都优于基于生成的通用信息抽取模型。

    We propose a new paradigm for universal information extraction (IE) that is compatible with any schema format and applicable to a list of IE tasks, such as named entity recognition, relation extraction, event extraction and sentiment analysis. Our approach converts the text-based IE tasks as the token-pair problem, which uniformly disassembles all extraction targets into joint span detection, classification and association problems with a unified extractive framework, namely UniEX. UniEX can synchronously encode schema-based prompt and textual information, and collaboratively learn the generalized knowledge from pre-defined information using the auto-encoder language models. We develop a traffine attention mechanism to integrate heterogeneous factors including tasks, labels and inside tokens, and obtain the extraction target via a scoring matrix. Experiment results show that UniEX can outperform generative universal IE models in terms of performance and inference-speed on $14$ benchmar
    
[^17]: 更鲁棒的自然语言处理系统评估方法：处理基准测试中的缺失得分问题

    Towards More Robust NLP System Evaluation: Handling Missing Scores in Benchmarks. (arXiv:2305.10284v1 [cs.CL])

    [http://arxiv.org/abs/2305.10284](http://arxiv.org/abs/2305.10284)

    本文提出了一种鲁棒的自然语言处理系统评估方法，可以解决基准测试中某些系统的得分缺失问题，并引入了一个规模更大的基准测试。

    

    自然语言处理系统的评估对推动该领域的发展至关重要，但目前的基准测试方法常常假设所有系统在所有任务上都有可用的得分，这并不总是切实可行的。在现实情况下，若干因素（例如运行基线，私有系统，计算限制或不完整的数据）可能会阻止某些系统在整个任务上进行评估。本文正式阐述了自然语言处理研究中的一个现有问题：如何在一些系统的任务得分缺失时进行基准测试，并提出了一种新的解决方法。我们的方法利用兼容的部分排名方法来填补缺失的数据，然后使用Borda计数方法进行聚合。它包括两个特定于任务级得分或实例级得分可用的场景的细化。我们还引入了一个扩展基准测试，其中包含超过1.31亿个得分，比现有基准测试大一个数量级。我们验证了我们的方法并证明其有效性。

    The evaluation of natural language processing (NLP) systems is crucial for advancing the field, but current benchmarking approaches often assume that all systems have scores available for all tasks, which is not always practical. In reality, several factors such as the cost of running baseline, private systems, computational limitations, or incomplete data may prevent some systems from being evaluated on entire tasks. This paper formalize an existing problem in NLP research: benchmarking when some systems scores are missing on the task, and proposes a novel approach to address it. Our method utilizes a compatible partial ranking approach to impute missing data, which is then aggregated using the Borda count method. It includes two refinements designed specifically for scenarios where either task-level or instance-level scores are available. We also introduce an extended benchmark, which contains over 131 million scores, an order of magnitude larger than existing benchmarks. We validate
    
[^18]: 连锁符号提示激发了大型语言模型中的规划能力

    Chain-of-Symbol Prompting Elicits Planning in Large Langauge Models. (arXiv:2305.10276v1 [cs.CL])

    [http://arxiv.org/abs/2305.10276](http://arxiv.org/abs/2305.10276)

    本文提出了自然语言规划（NLP）的基准，旨在研究LLMs在需要理解并在文本中相应进行操作的复杂规划任务中的表现。同时提出了一种新方法CoS，使用简化的符号空间表示法来表示复杂的环境。

    

    本文旨在研究LLMs在需要理解通过自然语言模拟的虚拟空间环境并在文本中相应进行操作的复杂规划任务中的表现。我们提出了一个名为自然语言规划（NLP）的基准，它由一组新颖的任务组成：Brick World、基于NLVR的操作和自然语言导航。我们发现当前流行的LLMs（如ChatGPT）仍然缺乏复杂规划的能力。这引出了一个问题——LLMs是否对自然语言中描述的环境有良好的理解，或者其他替代方法（如符号表示）是否更加简单，因此更容易被LLMs理解？为此，我们提出了一种名为CoS（Chain-of-Symbol Prompting）的新方法，在链式中间思考步骤中使用简化的符号空间表示法来表示复杂的环境。CoS易于使用，不需要对LLMs进行额外的培训。

    In this paper, we take the initiative to investigate the performance of LLMs on complex planning tasks that require LLMs to understand a virtual spatial environment simulated via natural language and act correspondingly in text. We propose a benchmark named Natural Language Planning (NLP) composed of a set of novel tasks: Brick World, NLVR-based Manipulations, and Natural Language Navigation. We found that current popular LLMs such as ChatGPT still lack abilities in complex planning. This arises a question -- do the LLMs have a good understanding of the environments described in natural language, or maybe other alternatives such as symbolic representations are neater and hence better to be understood by LLMs? To this end, we propose a novel method called CoS (Chain-of-Symbol Prompting) that represents the complex environments with condensed symbolic spatial representations during the chained intermediate thinking steps. CoS is easy to use and does not need additional training on LLMs. 
    
[^19]: 增强本地谱时特征用于语音分析

    Boosting Local Spectro-Temporal Features for Speech Analysis. (arXiv:2305.10270v1 [cs.CL])

    [http://arxiv.org/abs/2305.10270](http://arxiv.org/abs/2305.10270)

    该论文介绍了在语音识别中电话分类的问题，并探索了几组可以用于电话分类的本地谱时特征，提出了使用Haar特征和SVM分类的梯度直方图进行电话分类，并给出了一些初步结果。

    

    我们介绍了在语音识别中，电话分类的问题，并探索了几组可以用于电话分类的本地谱时特征。特别地，我们提出了使用两组常用于物体检测的特征（Haar特征和SVM分类的梯度直方图（HoG））进行电话分类的一些初步结果。

    We introduce the problem of phone classification in the context of speech recognition, and explore several sets of local spectro-temporal features that can be used for phone classification. In particular, we present some preliminary results for phone classification using two sets of features that are commonly used for object detection: Haar features and SVM-classified Histograms of Gradients (HoG)
    
[^20]: 在大规模多语言语言模型中搜索针的作用：探究意外双语对于PaLM翻译能力的影响

    Searching for Needles in a Haystack: On the Role of Incidental Bilingualism in PaLM's Translation Capability. (arXiv:2305.10266v1 [cs.CL])

    [http://arxiv.org/abs/2305.10266](http://arxiv.org/abs/2305.10266)

    本文探究了大型语言模型翻译能力中的意外双语现象，证明了PaLM模型利用意外双语内容可以改善零-shot翻译的准确性。

    

    尽管从未见过传统神经机器翻译系统提供的有意的翻译样例，大型多语言语言模型展现出令人惊讶的零或少量样例翻译能力。我们调查了意外双语对于大型语言模型翻译能力的解释作用-包括有意提供的翻译样例在内的双语信号的非意外消费，以Pathways语言模型（PaLM）为案例进行研究。我们引入了一种混合方法来衡量和理解规模上的意外双语现象。我们展示了PaLM暴露于至少44种语言中的超过3000万个翻译对。此外，各种非英语语言的意外双语内容量与该语言的单语内语言内容量高度相关。我们将意外双语内容与零-shot提示相关联，并展示它可以被用于挖掘新提示，以提高PaLM的英语以外的零-shot翻译准确度。

    Large, multilingual language models exhibit surprisingly good zero- or few-shot machine translation capabilities, despite having never seen the intentionally-included translation examples provided to typical neural translation systems. We investigate the role of incidental bilingualism -- the unintentional consumption of bilingual signals, including translation examples -- in explaining the translation capabilities of large language models, taking the Pathways Language Model (PaLM) as a case study. We introduce a mixed-method approach to measure and understand incidental bilingualism at scale. We show that PaLM is exposed to over 30 million translation pairs across at least 44 languages. Furthermore, the amount of incidental bilingual content is highly correlated with the amount of monolingual in-language content for non-English languages. We relate incidental bilingual content to zero-shot prompts and show that it can be used to mine new prompts to improve PaLM's out-of-English zero-s
    
[^21]: M3KE:一种面向中国大语言模型的大规模多级多主题知识评估基准

    M3KE: A Massive Multi-Level Multi-Subject Knowledge Evaluation Benchmark for Chinese Large Language Models. (arXiv:2305.10263v1 [cs.CL])

    [http://arxiv.org/abs/2305.10263](http://arxiv.org/abs/2305.10263)

    本文提出了一种面向中国大语言模型的大规模多级多主题知识评估基准M3KE，收集了20,477个问题以覆盖中国教育体系的所有主要层次和广泛的学科，使用多任务准确性测试法有效地评估了四个大语言模型GPT-2，RoBERTa，ERNIE和ELECTRA对多源知识的整合和利用能力。

    

    大型语言模型最近在各个方面取得了巨大的进展，例如跨任务通用性，指令遵循等。全面评估大语言模型在多个任务中的能力非常重要。在本文中，我们提出了M3KE，一种大规模多级多主题知识评估基准，旨在通过测试零和几个示例设置下的多任务准确性来衡量中文大语言模型所获得的知识。我们收集了71个任务的20,477个问题。选择涵盖了中国教育体系的所有主要层次，从小学到大学，以及广泛的学科，包括人文，历史，政治，法律，教育，心理，科学，技术，艺术和宗教。所有问题都是四个选项的多选题，因此保证了标准化和统一的评估流程。我们使用我们的基准测试了一些最先进的开源中文大语言模型，包括GPT-2，RoBERTa，ERNIE和ELECTRA，并提供了详细的结果和分析。我们展示了M3KE可以有效地评估大型语言模型，并全面了解它们整合和利用多个知识来源的能力。

    Large language models have recently made tremendous progress in a variety of aspects, e.g., cross-task generalization, instruction following. Comprehensively evaluating the capability of large language models in multiple tasks is of great importance. In this paper, we propose M3KE, a Massive Multi-Level Multi-Subject Knowledge Evaluation benchmark, which is developed to measure knowledge acquired by Chinese large language models by testing their multitask accuracy in zero- and few-shot settings. We have collected 20,477 questions from 71 tasks. Our selection covers all major levels of Chinese education system, ranging from the primary school to college, as well as a wide variety of subjects, including humanities, history, politics, law, education, psychology, science, technology, art and religion. All questions are multiple-choice questions with four options, hence guaranteeing a standardized and unified assessment process. We've assessed a number of state-of-the-art open-source Chines
    
[^22]: MemoryBank: 用长期记忆增强大型语言模型

    MemoryBank: Enhancing Large Language Models with Long-Term Memory. (arXiv:2305.10250v1 [cs.CL])

    [http://arxiv.org/abs/2305.10250](http://arxiv.org/abs/2305.10250)

    MemoryBank 提出了一种新型内存机制，旨在为大型语言模型提供类人的长期记忆。它可以召唤相关记忆，通过持续的记忆更新不断进化，通过合成过去的互动信息理解并适应用户个性。

    

    大型语言模型的革命性进展极大地改变了我们与人工智能系统的互动方式。尽管如此，其中一个明显的不足之处是这些模型缺乏长期记忆机制。这在需要持续互动的情况下尤为明显，例如个人伴侣系统和心理咨询。因此，我们提出了MemoryBank，这是一种专为LLM量身定制的新型内存机制。MemoryBank可以召唤相关记忆，通过持续的记忆更新不断进化，通过合成过去的互动信息理解并适应用户个性。为了模仿人类行为并有选择地保存记忆，MemoryBank采用了受Ebbinghaus遗忘曲线理论启发的记忆更新机制，这样人工智能可以根据时间和记忆的相对重要性来遗忘和加强记忆，从而为LLM提供类似于人类的长期记忆。

    Revolutionary advancements in Large Language Models have drastically reshaped our interactions with artificial intelligence systems. Despite this, a notable hindrance remains-the deficiency of a long-term memory mechanism within these models. This shortfall becomes increasingly evident in situations demanding sustained interaction, such as personal companion systems and psychological counseling. Therefore, we propose MemoryBank, a novel memory mechanism tailored for LLMs. MemoryBank enables the models to summon relevant memories, continually evolve through continuous memory updates, comprehend, and adapt to a user personality by synthesizing information from past interactions. To mimic anthropomorphic behaviors and selectively preserve memory, MemoryBank incorporates a memory updating mechanism, inspired by the Ebbinghaus Forgetting Curve theory, which permits the AI to forget and reinforce memory based on time elapsed and the relative significance of the memory, thereby offering a hum
    
[^23]: OpenSLU: 一个统一、模块化、可扩展的语音理解工具包

    OpenSLU: A Unified, Modularized, and Extensible Toolkit for Spoken Language Understanding. (arXiv:2305.10231v1 [cs.CL])

    [http://arxiv.org/abs/2305.10231](http://arxiv.org/abs/2305.10231)

    OpenSLU是一个统一、模块化、可扩展的语音理解工具包，将10种针对单意图和多意图场景的语音理解模型统一起来，同时支持非预训练和预训练模型，并高度模块化、可扩展。

    

    语音理解是任务型对话系统的核心组件之一，旨在提取用户查询（例如意图和槽位）的语义含义。本文介绍了OpenSLU，一个开源工具包，提供了统一、模块化、可扩展的语音理解解决方案。具体而言，OpenSLU将10种针对单意图和多意图场景的语音理解模型统一起来，同时支持非预训练和预训练模型。此外，OpenSLU具有高度模块化和可扩展性，通过将模型架构、推理和学习过程分解为可重用模块，允许研究人员以高度灵活的配置快速设置语音理解实验。OpenSLU基于PyTorch实现，并在\url{https://github.com/LightChen233/OpenSLU}上发布。

    Spoken Language Understanding (SLU) is one of the core components of a task-oriented dialogue system, which aims to extract the semantic meaning of user queries (e.g., intents and slots). In this work, we introduce OpenSLU, an open-source toolkit to provide a unified, modularized, and extensible toolkit for spoken language understanding. Specifically, OpenSLU unifies 10 SLU models for both single-intent and multi-intent scenarios, which support both non-pretrained and pretrained models simultaneously. Additionally, OpenSLU is highly modularized and extensible by decomposing the model architecture, inference, and learning process into reusable modules, which allows researchers to quickly set up SLU experiments with highly flexible configurations. OpenSLU is implemented based on PyTorch, and released at \url{https://github.com/LightChen233/OpenSLU}.
    
[^24]: 护盾式表示：通过迭代基于梯度的投影保护敏感属性

    Shielded Representations: Protecting Sensitive Attributes Through Iterative Gradient-Based Projection. (arXiv:2305.10204v1 [cs.CL])

    [http://arxiv.org/abs/2305.10204](http://arxiv.org/abs/2305.10204)

    本文提出了一种名为迭代梯度基础投影（IGBP）的新方法，用于从神经表示中删除非线性编码的概念，以减轻模型的社会偏见。该方法通过迭代训练神经分类器来预测某个敏感属性，然后将表示投影到一个超平面上，使得分类器对目标属性变得无意识。实验证明，该方法在消除敏感属性方面是有效的，并且对下游任务的准确性影响很小。

    

    自然语言处理模型倾向于学习和编码数据中存在的社会偏见。解决此类偏差的一种流行方法是消除模型表示中编码的信息。然而，当前的方法仅限于删除线性编码的信息。在这项工作中，我们提出了一种名为迭代梯度基础投影（IGBP）的新方法，用于从神经表示中删除非线性编码的概念。我们的方法包括通过迭代训练神经分类器来预测我们要消除的特定属性，然后将表示投影到一个超平面上，使得分类器对目标属性变得无意识。我们评估了我们的方法在消除性别和种族信息作为敏感属性的任务上的有效性。我们的结果表明，IGBP通过内在和外在评估在减轻偏见方面是有效的，并且对下游任务的准确性影响很小。

    Natural language processing models tend to learn and encode social biases present in the data. One popular approach for addressing such biases is to eliminate encoded information from the model's representations. However, current methods are restricted to removing only linearly encoded information. In this work, we propose Iterative Gradient-Based Projection (IGBP), a novel method for removing non-linear encoded concepts from neural representations. Our method consists of iteratively training neural classifiers to predict a particular attribute we seek to eliminate, followed by a projection of the representation on a hypersurface, such that the classifiers become oblivious to the target attribute. We evaluate the effectiveness of our method on the task of removing gender and race information as sensitive attributes. Our results demonstrate that IGBP is effective in mitigating bias through intrinsic and extrinsic evaluations, with minimal impact on downstream task accuracy.
    
[^25]: 零代词翻译综述

    A Survey on Zero Pronoun Translation. (arXiv:2305.10196v1 [cs.CL])

    [http://arxiv.org/abs/2305.10196](http://arxiv.org/abs/2305.10196)

    本文总结了零代词翻译（ZPT）领域神经网络全面推广后的重要工作，发现大型语言模型、多任务或迁移学习都可以实现ZPT的性能提升。

    

    零代词（ZP）通常在类似中文、匈牙利语和印地语这样的丢省略，而在非丢失省份诸如英语中，应当进行回应。这一现象在机器翻译（MT）领域中得到了广泛的研究，因为它很难确定代词的正确先行词，这是MT系统面临的重要挑战。本文总结了神经网络全面推展之后在零代词翻译（ZPT）方面所做的重要工作，以便研究人员了解当前状态和未来方向。我们根据演变、数据集、方法和评估提供了一份文献组织形式。此外，我们还比较和分析了在不同基准测试上的竞争模型和评估指标。我们挖掘了一些有益的发现，例如：1）ZPT符合大型语言模型的发展趋势；2）数据限制会在不同语言和领域中产生学习偏差；3）通过多任务或迁移学习可以实现性能提升。

    Zero pronouns (ZPs) are frequently omitted in pro-drop languages (e.g. Chinese, Hungarian, and Hindi), but should be recalled in non-pro-drop languages (e.g. English). This phenomenon has been studied extensively in machine translation (MT), as it poses a significant challenge for MT systems due to the difficulty in determining the correct antecedent for the pronoun. This survey paper highlights the major works that have been undertaken in zero pronoun translation (ZPT) after the neural revolution, so that researchers can recognise the current state and future directions of this field. We provide an organisation of the literature based on evolution, dataset, method and evaluation. In addition, we compare and analyze competing models and evaluation metrics on different benchmarks. We uncover a number of insightful findings such as: 1) ZPT is in line with the development trend of large language model; 2) data limitation causes learning bias in languages and domains; 3) performance improv
    
[^26]: 利用动机访谈策略提升心理压力支持对话响应

    Boosting Distress Support Dialogue Responses with Motivational Interviewing Strategy. (arXiv:2305.10195v1 [cs.CL])

    [http://arxiv.org/abs/2305.10195](http://arxiv.org/abs/2305.10195)

    本文提出利用动机访谈策略重新表达在线心理支持对话中的响应类型，从而提高聊天机器人响应的符合性和质量。

    

    基于人工智能的聊天机器人成为缓解心理压力的解决方案。由于缺乏心理治疗数据，研究人员使用从在线同伴支持论坛抓取的对话来训练它们。但由于此类平台上的响应并非由专业人士提供，因此它们既包含符合标准的响应又包含不符合标准的响应。在这项工作中，我们尝试使用源自名为动机访谈治疗完整性（MITI）代码的行为编码方案所适应的标签来识别在线心理支持对话中存在的符合标准和不符合标准的响应类型，并展示如何将某些响应类型重新表达为更符合动机访谈标准的形式，从而使聊天机器人响应更符合动机访谈策略。作为概念证明，我们通过对Blender和GPT3进行微调来构建几个重新表达器，将MI不符合标准的“未经允许建议”响应转化为“经过许可的建议”。我们展示了如何借助人机合作实现这一目标，并在在线心理支持对话数据集上评估了我们的方法。结果表明，使用源自MI框架的重新表达技术可以提高心理压力支持对话中聊天机器人响应的符合性和质量。

    AI-driven chatbots have become an emerging solution to address psychological distress. Due to the lack of psychotherapeutic data, researchers use dialogues scraped from online peer support forums to train them. But since the responses in such platforms are not given by professionals, they contain both conforming and non-conforming responses. In this work, we attempt to recognize these conforming and non-conforming response types present in online distress-support dialogues using labels adapted from a well-established behavioral coding scheme named Motivational Interviewing Treatment Integrity (MITI) code and show how some response types could be rephrased into a more MI adherent form that can, in turn, enable chatbot responses to be more compliant with the MI strategy. As a proof of concept, we build several rephrasers by fine-tuning Blender and GPT3 to rephrase MI non-adherent "Advise without permission" responses into "Advise with permission". We show how this can be achieved with th
    
[^27]: 变长神经中间语表示用于零样本神经机器翻译

    Variable-length Neural Interlingua Representations for Zero-shot Neural Machine Translation. (arXiv:2305.10190v1 [cs.CL])

    [http://arxiv.org/abs/2305.10190](http://arxiv.org/abs/2305.10190)

    本研究提出一种变长神经中间语表示方法，克服了先前的定长表示方法的限制。在多个数据集上，我们的方法比固定长度中间语表示方法表现更好，但在特定源语言的翻译中效果欠佳。

    

    多语种神经机器翻译（MNMT）模型中编码表示的语言无关性对其在零样本翻译上的泛化能力至关重要。神经中间语表示已被证明是实现这一目标的有效方法。然而，先前工作中引入的定长神经中间语表示可能会限制其灵活性和表示能力。本研究通过使神经中间语表示的长度变化来增强神经中间语表示的方法，从而克服了定长神经中间语表示的约束。我们在OPUS、IWSLT和Europarl数据集上进行的零样本翻译的实证结果表明，相对于固定长度神经中间语表示，我们的方法具有稳定的模型收敛性和优秀的零样本翻译结果。然而，我们的分析揭示了我们的方法在翻译某些源语言时的效果不佳。

    The language-independency of encoded representations within multilingual neural machine translation (MNMT) models is crucial for their generalization ability on zero-shot translation. Neural interlingua representations have been shown as an effective method for achieving this. However, fixed-length neural interlingua representations introduced in previous work can limit its flexibility and representation ability. In this study, we introduce a novel method to enhance neural interlingua representations by making their length variable, thereby overcoming the constraint of fixed-length neural interlingua representations. Our empirical results on zero-shot translation on OPUS, IWSLT, and Europarl datasets demonstrate stable model convergence and superior zero-shot translation results compared to fixed-length neural interlingua representations. However, our analysis reveals the suboptimal efficacy of our approach in translating from certain source languages, wherein we pinpoint the defective
    
[^28]: 基于知识增强的混合主动对话情感支持系统

    Knowledge-enhanced Mixed-initiative Dialogue System for Emotional Support Conversations. (arXiv:2305.10172v1 [cs.CL])

    [http://arxiv.org/abs/2305.10172](http://arxiv.org/abs/2305.10172)

    本文针对情感支持对话系统的混合主动特点，提出了基于知识增强的混合主动对话框架，该框架从大型心理健康知识图谱中检索实际案例知识来生成混合主动响应，并在共情和问题解决能力方面显著优于几个基线模型。

    

    与共情对话不同，情感支持对话系统需要在安慰求助者的同时主动帮助探索和解决问题。本文研究了混合主动情感支持对话的问题，其中用户和系统都可以在对话中采取主动。我们提出了一个用于评估混合主动情感支持对话的新型模式，并提出了四个情感支持指标来评价混合主动交互。分析揭示了构建混合主动情感支持对话系统的必要性和挑战。在此基础上，我们提出了基于知识增强的混合主动对话框架（KEMI），该框架从大型心理健康知识图谱中检索实际案例知识来生成混合主动响应。实验证明，KEMI在共情和问题解决能力方面显著优于几个基线模型。

    Unlike empathetic dialogues, the system in emotional support conversations (ESC) is expected to not only convey empathy for comforting the help-seeker, but also proactively assist in exploring and addressing their problems during the conversation. In this work, we study the problem of mixed-initiative ESC where the user and system can both take the initiative in leading the conversation. Specifically, we conduct a novel analysis on mixed-initiative ESC systems with a tailor-designed schema that divides utterances into different types with speaker roles and initiative types. Four emotional support metrics are proposed to evaluate the mixed-initiative interactions. The analysis reveals the necessity and challenges of building mixed-initiative ESC systems. In the light of this, we propose a knowledge-enhanced mixed-initiative framework (KEMI) for ESC, which retrieves actual case knowledge from a large-scale mental health knowledge graph for generating mixed-initiative responses. Experimen
    
[^29]: 结构化传递博弈中的实用推理

    Pragmatic Reasoning in Structured Signaling Games. (arXiv:2305.10167v1 [cs.AI])

    [http://arxiv.org/abs/2305.10167](http://arxiv.org/abs/2305.10167)

    本文介绍了一个结构化传递博弈和理性言语行为框架的变体sRSA，应用于结构化领域中的实用推理问题。在颜色领域中，我们的研究表明采用sRSA的代理比传统RSA和仅基于强化学习的代理更接近于信息理论界限。

    

    本文引入了一种带有相似性结构的结构化传递博弈，并提出了一个理性言语行为框架的变体，称为结构化理性言语行为（sRSA），用于解决结构化领域的实用推理问题。我们探索了在颜色领域中采用sRSA的理性智能代理的行为，证明了采用World Color Survey得出的语义表示的结构化代理比传统RSA和仅基于强化学习的代理更接近于信息理论界限，且经过1或2次递归的训练就能够达到效率极限。此外，我们还研究了实用推理和多智能体强化学习框架中的学习过程和相互作用。结果表明，采用sRSA的人工智能代理发展出的通信策略更接近于信息理论界限。

    In this work we introduce a structured signaling game, an extension of the classical signaling game with a similarity structure between meanings in the context, along with a variant of the Rational Speech Act (RSA) framework which we call structured-RSA (sRSA) for pragmatic reasoning in structured domains. We explore the behavior of the sRSA in the domain of color and show that pragmatic agents using sRSA on top of semantic representations, derived from the World Color Survey, attain efficiency very close to the information theoretic limit after only 1 or 2 levels of recursion. We also explore the interaction between pragmatic reasoning and learning in multi-agent reinforcement learning framework. Our results illustrate that artificial agents using sRSA develop communication closer to the information theoretic frontier compared to agents using RSA and just reinforcement learning. We also find that the ambiguity of the semantic representation increases as the pragmatic agents are allowe
    
[^30]: 基于知识增强的生成预训练模型在中国医学执业医师资格考试上的应用研究

    Qualifying Chinese Medical Licensing Examination with Knowledge Enhanced Generative Pre-training Model. (arXiv:2305.10163v1 [cs.CL])

    [http://arxiv.org/abs/2305.10163](http://arxiv.org/abs/2305.10163)

    本研究通过在ChatGPT中集成医学领域知识和启用少样本学习的新方法，在中国国家医学执业医师资格考试中取得成功，这为建立在自然语言处理技术和医学领域知识的创新应用提供了可能。

    

    生成式预训练模型（GPT），如ChatGPT，在各种自然语言处理任务中展现出了出色的性能。尽管ChatGPT已被整合到各个领域的工作流中以提高效率，但其微调过程的灵活性不足，阻碍了其在需要广泛领域专业知识和语义知识的领域，如医疗保健，的应用。在本文中，我们评估了ChatGPT在中国国家医学执业医师资格考试（CNMLE）中的表现，并提出了一种新的方法来改进ChatGPT，即从两个方面集成医学领域知识和启用少样本学习。通过使用简单但有效的检索方法，将医学背景知识提取为语义指令来指导ChatGPT的推断。类似地，相关的医疗问题被识别并作为演示输入给ChatGPT。实验结果表明，直接应用ChatGPT无法在CNMLE上获得合格分数（51分），只有基于知识增强训练的模型成功通过考试。

    Generative Pre-Training (GPT) models like ChatGPT have demonstrated exceptional performance in various Natural Language Processing (NLP) tasks. Although ChatGPT has been integrated into the overall workflow to boost efficiency in many domains, the lack of flexibility in the finetuning process hinders its applications in areas that demand extensive domain expertise and semantic knowledge, such as healthcare. In this paper, we evaluate ChatGPT on the China National Medical Licensing Examination (CNMLE) and propose a novel approach to improve ChatGPT from two perspectives: integrating medical domain knowledge and enabling few-shot learning. By using a simple but effective retrieval method, medical background knowledge is extracted as semantic instructions to guide the inference of ChatGPT. Similarly, relevant medical questions are identified and fed as demonstrations to ChatGPT. Experimental results show that directly applying ChatGPT fails to qualify the CNMLE at a score of 51 (i.e., onl
    
[^31]: 不要用明文上传测试数据：减轻数据外泄对于评估基准的持续影响的实用策略

    Stop Uploading Test Data in Plain Text: Practical Strategies for Mitigating Data Contamination by Evaluation Benchmarks. (arXiv:2305.10160v1 [cs.CL])

    [http://arxiv.org/abs/2305.10160](http://arxiv.org/abs/2305.10160)

    提出了三个适用策略：（1）公钥加密发布测试数据，仅允许特定派生发布；（2）对于API持有方，要求训练排除控制，保护测试数据，不停止评估直到达到要求；（3）如果测试数据来自互联网文本，需避免某些结果的使用。

    

    随着预训练模型在自动爬网资料库的大规模应用，数据外泄变得常见且部分难以应对。对于那些不会公开训练数据的模型，其数据成为了商业机密，即使在公开模型中，确定特定测试实例是否被泄露也不是一件容易的事情。本文提出三个可行的策略：（1）使用公钥加密发布的测试数据并限制派生发布的许可；（2）要求持有API训练数据的公司采用训练排除控制，并拒绝评估，直到训练排除控制无误为止；（3）如果测试数据来自互联网文本，那么需避免在网络搜索中出现包含正确提取部分的数据。

    Data contamination has become especially prevalent and challenging with the rise of models pretrained on very large, automatically-crawled corpora. For closed models, the training data becomes a trade secret, and even for open models, it is not trivial to ascertain whether a particular test instance has been compromised. Strategies such as live leaderboards with hidden answers, or using test data which is guaranteed to be unseen, are expensive and become fragile with time. Assuming that all relevant actors value clean test data and will cooperate to mitigate data contamination, what can be done? We propose three strategies that can make a difference: (1) Test data made public should be encrypted with a public key and licensed to disallow derivative distribution; (2) demand training exclusion controls from closed API holders, and protect your test data by refusing to evaluate until demands are met; (3) in case of test data based on internet text, avoid data which appears with its soluti
    
[^32]: 阅读过程中对小说人物个性的理解

    Personality Understanding of Fictional Characters during Book Reading. (arXiv:2305.10156v1 [cs.CL])

    [http://arxiv.org/abs/2305.10156](http://arxiv.org/abs/2305.10156)

    本文提出了一个NLP领域内尚未研究的问题：情景和细致地理解小说人物个性，并提供了第一个标记数据集PersoNet来解决这个问题。

    

    理解小说人物个性是阅读故事的关键。随着读者与故事的互动，他们对一个人物的理解会根据新的事件和信息而演变；并且可以感知到多个精细的个性方面。这导致了一个自然的问题：情境和精细的个性理解。这个问题在NLP领域中没有得到研究，主要是由于缺乏模仿阅读过程的适当数据集。我们提供了第一个标记数据集PersoNet来解决这个问题。我们的新型注释策略涉及用在线阅读应用程序的用户笔记作为原始书籍的代理进行注释。实验和人体研究表明，我们的数据集构建既有效又准确；我们的任务在很大程度上依赖于长期的上下文以实现对机器和人类的准确预测。数据集可在https://github.com/Gorov/personet_acl23获得。

    Comprehending characters' personalities is a crucial aspect of story reading. As readers engage with a story, their understanding of a character evolves based on new events and information; and multiple fine-grained aspects of personalities can be perceived. This leads to a natural problem of situated and fine-grained personality understanding. The problem has not been studied in the NLP field, primarily due to the lack of appropriate datasets mimicking the process of book reading. We present the first labeled dataset PersoNet for this problem. Our novel annotation strategy involves annotating user notes from online reading apps as a proxy for the original books. Experiments and human studies indicate that our dataset construction is both efficient and accurate; and our task heavily relies on long-term context to achieve accurate predictions for both machines and humans. The dataset is available at https://github.com/Gorov/personet_acl23.
    
[^33]: 迭代学习与交流共同解释了有效的颜色命名系统

    Iterated learning and communication jointly explain efficient color naming systems. (arXiv:2305.10154v1 [cs.CL])

    [http://arxiv.org/abs/2305.10154](http://arxiv.org/abs/2305.10154)

    本论文通过结合迭代学习和交流的文化进化模型，展示这个模型能够在神经网络中实现并收敛于高效的颜色命名系统，进一步证明了语义系统反映效率压力的观点。

    

    已经有人认为，语义系统反映了效率的压力，一个当前的争论关注于产生这种模式的文化进化过程。我们将效率实现为信息瓶颈原理，并结合迭代学习和交流的文化进化模型。我们展示了这个在神经网络中实现的模型收敛于在IB意义下高效并且类似于人类颜色命名系统的颜色命名系统。我们还表明，仅仅迭代学习或者仅仅交流并不能像这个模型那样产生相同的结果。

    It has been argued that semantic systems reflect pressure for efficiency, and a current debate concerns the cultural evolutionary process that produces this pattern. We consider efficiency as instantiated in the Information Bottleneck (IB) principle, and a model of cultural evolution that combines iterated learning and communication. We show that this model, instantiated in neural networks, converges to color naming systems that are efficient in the IB sense and similar to human color naming systems. We also show that iterated learning alone, and communication alone, do not yield the same outcome as clearly.
    
[^34]: 面向任务的对话系统中的多粒度知识检索

    Multi-Grained Knowledge Retrieval for End-to-End Task-Oriented Dialog. (arXiv:2305.10149v1 [cs.CL])

    [http://arxiv.org/abs/2305.10149](http://arxiv.org/abs/2305.10149)

    该论文提出了一种面向任务的对话系统中的多粒度知识检索方法，该方法通过实体选择器和属性选择器实现知识检索、响应生成解耦，并且使用一种新的蒸馏目标进行监督学习，从而可以更有效地执行知识检索以生成信息响应。

    

    从外部数据库中检索出合适的领域知识是生成信息响应的任务导向型端到端对话系统的核心。大多数现有的系统将知识检索与响应生成混合在一起，并通过引用响应直接监督优化，导致当知识库变得大规模时，检索表现不佳。为了解决这个问题，我们提出将知识检索从响应生成中解耦，并引入多粒度知识检索器（MAKER），其中包括实体选择器以搜索相关实体和属性选择器以过滤不相关属性。为了训练检索器，我们提出了一种新的蒸馏目标，从响应生成器中提取监督信号。在三个标准基准测试中进行的实验证明，我们的检索器比现有方法更有效地执行知识检索，且适用于小规模和大规模知识库。我们的代码已发布在https://github.com/PaddlePaddle/Research/tree/master/NLP/End-to-End%20Conversation%20Model%20(MAKER)。

    Retrieving proper domain knowledge from an external database lies at the heart of end-to-end task-oriented dialog systems to generate informative responses. Most existing systems blend knowledge retrieval with response generation and optimize them with direct supervision from reference responses, leading to suboptimal retrieval performance when the knowledge base becomes large-scale. To address this, we propose to decouple knowledge retrieval from response generation and introduce a multi-grained knowledge retriever (MAKER) that includes an entity selector to search for relevant entities and an attribute selector to filter out irrelevant attributes. To train the retriever, we propose a novel distillation objective that derives supervision signals from the response generator. Experiments conducted on three standard benchmarks with both small and large-scale knowledge bases demonstrate that our retriever performs knowledge retrieval more effectively than existing methods. Our code has be
    
[^35]: 自我博弈与人工智能反馈中的上下文学习改进语言模型的谈判策略

    Improving Language Model Negotiation with Self-Play and In-Context Learning from AI Feedback. (arXiv:2305.10142v1 [cs.CL])

    [http://arxiv.org/abs/2305.10142](http://arxiv.org/abs/2305.10142)

    本文研究了多个大型语言模型能否通过自我博弈和反馈互相提高，在谈判游戏中进行谈判，达成交易。使用历史记录和人工智能反馈迭代改进模型的谈判策略。

    

    本文研究了多个大型语言模型（LLM）能否通过玩耍、反思和批评在谈判游戏中彼此自主改进。如果LLM能够相互提高，则意味着可以在最小人工干预的情况下创建强大的人工智能代理。我们让两个LLM扮演买方和卖方角色进行协商，第三个LLM扮演批评家，为一方提供反馈以改进其谈判策略。我们使用历史交易记录和人工智能反馈作为上下文演示来迭代地改进模型的谈判策略。我们使用不同的LLM（GPT和Claude）来扮演不同的角色，并使用交易价格作为评估指标。实验揭示出多个有趣的发现。

    We study whether multiple large language models (LLMs) can autonomously improve each other in a negotiation game by playing, reflecting, and criticizing. We are interested in this question because if LLMs were able to improve each other, it would imply the possibility of creating strong AI agents with minimal human intervention. We ask two LLMs to negotiate with each other, playing the roles of a buyer and a seller, respectively. They aim to reach a deal with the buyer targeting a lower price and the seller a higher one. A third language model, playing the critic, provides feedback to a player to improve the player's negotiation strategies. We let the two agents play multiple rounds, using previous negotiation history and AI feedback as in-context demonstrations to improve the model's negotiation strategy iteratively. We use different LLMs (GPT and Claude) for different roles and use the deal price as the evaluation metric. Our experiments reveal multiple intriguing findings: (1) Only 
    
[^36]: 《附加宣言分解：一种理解政党立场的政策域感知方法》

    Additive manifesto decomposition: A policy domain aware method for understanding party positioning. (arXiv:2305.10136v1 [cs.CL])

    [http://arxiv.org/abs/2305.10136](http://arxiv.org/abs/2305.10136)

    本文提出了一种理解政党立场的方法，可以估计政策域感知的政党相似性，通过多维缩放提取可解释的主要政策轴上的政党立场。

    

    自动从政治文本中提取政党（不）相似性在计算政治学中扮演着越来越重要的角色，但现有方法基本上仅限于针对全局政党（不）相似性：它们将两个政党之间的关系压缩为单个数字，即相似性，无法提供有关政党在哪些领域上达成共识或不一致的任何定性洞见。本文提出了一种工作流程，用于估计政策域感知的政党相似性，克服了这一限制。该工作流程包括（a）定义合适的政策领域；（b）如果没有手动标签，则自动标记领域；（c）计算领域级相似性并在全局级别进行汇总；（d）通过多维缩放提取可解释的主要政策轴上的政党立场。我们评估了我们的工作流程。

    Automatic extraction of party (dis)similarities from texts such as party election manifestos or parliamentary speeches plays an increasing role in computational political science. However, existing approaches are fundamentally limited to targeting only global party (dis)-similarity: they condense the relationship between a pair of parties into a single figure, their similarity. In aggregating over all policy domains (e.g., health or foreign policy), they do not provide any qualitative insights into which domains parties agree or disagree on. This paper proposes a workflow for estimating policy domain aware party similarity that overcomes this limitation. The workflow covers (a) definition of suitable policy domains; (b) automatic labeling of domains, if no manual labels are available; (c) computation of domain-level similarities and aggregation at a global level; (d) extraction of interpretable party positions on major policy axes via multidimensional scaling. We evaluate our workflow 
    
[^37]: 康卡尼语口腔和鼻腔元音的实证分析

    Empirical Analysis of Oral and Nasal Vowels of Konkani. (arXiv:2305.10122v1 [cs.CL])

    [http://arxiv.org/abs/2305.10122](http://arxiv.org/abs/2305.10122)

    本研究通过实验分析了康卡尼语口腔和鼻腔元音的声学-语音学特性，为康卡尼语言的语音合成系统和元音的语言学研究提供了帮助。

    

    康卡尼语是一种高度鼻音化的语言，这使其在印度-雅利安语言中具有独特性。本研究探讨了康卡尼语口腔和鼻腔元音的声学-语音学特性。为此，收集了6名发言人（3男性和3女性）的语音样本。共使用了74个唯一句子作为录制脚本，分别为口腔和鼻腔元音37个。最终数据集由1135个元音音素组成。提供了康卡尼语口腔和鼻腔元音的比较F1-F2图以及实验结果和共振峰分析。所有鼻腔和口腔元音的平均F1，F2和F3值也首次通过实验报告。该研究可以帮助针对康卡尼语言的元音和语音合成系统的语言学研究。

    Konkani is a highly nasalised language which makes it unique among Indo-Aryan languages. This work investigates the acoustic-phonetic properties of Konkani oral and nasal vowels. For this study, speech samples from six speakers (3 male and 3 female) were collected. A total of 74 unique sentences were used as a part of the recording script, 37 each for oral and nasal vowels, respectively. The final data set consisted of 1135 vowel phonemes. A comparative F1-F2 plot of Konkani oral and nasal vowels is presented with an experimental result and formant analysis. The average F1, F2 and F3 values are also reported for the first time through experimentation for all nasal and oral vowels. This study can be helpful for the linguistic research on vowels and speech synthesis systems specific to the Konkani language.
    
[^38]: 使用移情反应意图分类法来控制和解释神经聊天机器人中的共情。

    Use of a Taxonomy of Empathetic Response Intents to Control and Interpret Empathy in Neural Chatbots. (arXiv:2305.10096v1 [cs.CL])

    [http://arxiv.org/abs/2305.10096](http://arxiv.org/abs/2305.10096)

    本文提出了一种使用移情反应意图分类法来控制和解释神经聊天机器人中的共情回应的方法，能够产生可控和可解释的共情回应。

    

    在开放领域对话代理的领域中，一个最近的趋势是让它们能够对情感提示进行同情式的对话。目前的方法要么遵循端到端的方法，要么在相似的情感标签上进行条件反应以产生共情式的回答。但共情是一个广泛的概念，它指的是个体对另一个人观察到的经历的认知和情感反应，它比单纯的情感模仿更加复杂。因此，除了通用情感外，还需要识别复杂的人类对话策略和动态来控制和解释聊天机器人的共情回应能力。在这项工作中，我们使用了八种共情反应意图的分类法以及通用情感类别来建立一个对话响应生成模型，能够以可控和可解释的方式产生共情回应。它由两个模块组成：1）响应情感/意图预测模块；以及2）响应生成模块。

    A recent trend in the domain of open-domain conversational agents is enabling them to converse empathetically to emotional prompts. Current approaches either follow an end-to-end approach or condition the responses on similar emotion labels to generate empathetic responses. But empathy is a broad concept that refers to the cognitive and emotional reactions of an individual to the observed experiences of another and it is more complex than mere mimicry of emotion. Hence, it requires identifying complex human conversational strategies and dynamics in addition to generic emotions to control and interpret empathetic responding capabilities of chatbots. In this work, we make use of a taxonomy of eight empathetic response intents in addition to generic emotion categories in building a dialogue response generation model capable of generating empathetic responses in a controllable and interpretable manner. It consists of two modules: 1) a response emotion/intent prediction module; and 2) a res
    
[^39]: 探索位置信息在视觉语言模型中的作用

    Probing the Role of Positional Information in Vision-Language Models. (arXiv:2305.10046v1 [cs.CL])

    [http://arxiv.org/abs/2305.10046](http://arxiv.org/abs/2305.10046)

    本研究调查了在视觉语言模型中位置信息的使用，表明模型存在位置信息，但不能很好地利用它进行图像 - 文本匹配。通过引入位置信息预训练和跨模态匹配的PI对比度学习，在挑战集上成功地改善了模型性能。

    

    在大多数视觉语言模型（VL）中，理解图像结构需要注入有关图像中物体位置信息（PI）。在我们对LXMERT进行的案例研究中，这是一种最先进的VL模型，我们探讨了PI在表示中的使用及其对视觉问答的影响。我们表明，该模型不能利用PI处理仅位置不同的挑战集上的图像 - 文本匹配任务。但是，我们通过探查实验证实了表示中确实存在PI。我们引入了两种策略来解决这个问题：（i）位置信息预训练和（ii）使用跨模态匹配的PI对比度学习。通过这样做，模型可以正确分类具有详细PI陈述的图像是否匹配。除了来自边界框的2D信息外，我们引入物体深度作为新特征，以在空间中更好地定位对象。尽管我们成功地改善了模型性能，但在我们的故事中仍有进一步的挑战，值得进一步研究。

    In most Vision-Language models (VL), the understanding of the image structure is enabled by injecting the position information (PI) about objects in the image. In our case study of LXMERT, a state-of-the-art VL model, we probe the use of the PI in the representation and study its effect on Visual Question Answering. We show that the model is not capable of leveraging the PI for the image-text matching task on a challenge set where only position differs. Yet, our experiments with probing confirm that the PI is indeed present in the representation. We introduce two strategies to tackle this: (i) Positional Information Pre-training and (ii) Contrastive Learning on PI using Cross-Modality Matching. Doing so, the model can correctly classify if images with detailed PI statements match. Additionally to the 2D information from bounding boxes, we introduce the object's depth as new feature for a better object localization in the space. Even though we were able to improve the model properties a
    
[^40]: 语言模型能否用自然语言解决图问题？

    Can Language Models Solve Graph Problems in Natural Language?. (arXiv:2305.10037v1 [cs.CL])

    [http://arxiv.org/abs/2305.10037](http://arxiv.org/abs/2305.10037)

    本论文提出了自然语言图形(NLGraph)，这是一个全面的基于图形问题解决测试，旨在评估LLM在文本描述的图形结构和图形解决方案方面的处理能力。实验结果表明，LLM(GPT-3/4)具有相应的图形推理能力。

    

    大型语言模型越来越多地应用于一些具有隐式图形结构的任务，例如机器人规划、多跳问题回答或知识探索、结构化常识推理等等。虽然LLM在这些任务中已经取得了一定的进展，但是LLM是否能够显式处理图形的文本描述，将它们映射到基于概念的空间中，并执行结构化操作仍然尚未得到足够的研究。为此，我们提出了自然语言图形(NLGraph)，它是一个设计用于自然语言的基于图形问题解决全面测试。NLGraph包含29,370个问题，涵盖了八个图形推理任务，从简单的连接和最短路径到复杂的最大流和模拟图神经网络等任务不等。我们在NLGraph基准测试上评估了LLM(GPT-3/4)，并发现1)语言模型具有相应的图形推理能力；

    Large language models (LLMs) are increasingly adopted for a variety of tasks with implicit graphical structures, such as planning in robotics, multi-hop question answering or knowledge probing, structured commonsense reasoning, and more. While LLMs have advanced the state-of-the-art on these tasks with structure implications, whether LLMs could explicitly process textual descriptions of graphs and structures, map them to grounded conceptual spaces, and perform structured operations remains underexplored. To this end, we propose NLGraph (Natural Language Graph), a comprehensive benchmark of graph-based problem solving designed in natural language. NLGraph contains 29,370 problems, covering eight graph reasoning tasks with varying complexity from simple tasks such as connectivity and shortest path up to complex problems such as maximum flow and simulating graph neural networks. We evaluate LLMs (GPT-3/4) with various prompting approaches on the NLGraph benchmark and find that 1) language
    
[^41]: 你在抄我的模型吗？基于后门水印的保护大语言模型在 EaaS 中的版权

    Are You Copying My Model? Protecting the Copyright of Large Language Models for EaaS via Backdoor Watermark. (arXiv:2305.10036v1 [cs.CL])

    [http://arxiv.org/abs/2305.10036](http://arxiv.org/abs/2305.10036)

    提出了一种名为 EmbMarker 的嵌入式水印方法，用于保护大型语言模型在 EaaS 中的版权。该方法可以在嵌入式上植入后门，并有效地传输和恢复。实验证明，EmbMarker 可以在维护各种 NLP 任务的性能的同时成功保护 EaaS 对 LLM 的版权。

    

    大型语言模型已经展示了在文本理解和生成方面的强大能力。公司已经开始基于这些大型语言模型提供嵌入式服务 (EaaS)，可以为客户的各种自然语言处理 (NLP) 任务带来益处。然而，先前的研究表明，EaaS 易受到模型提取攻击的攻击，这可能会对 LLM 的所有者造成巨大损失，因为训练这些模型非常昂贵。为了保护 EaaS 的 LLM 的版权，我们提出了一个名为 EmbMarker 的嵌入式水印方法，该方法在嵌入式上植入后门。我们的方法从通用文本语料库中选择一组中等频率的单词，形成触发集，然后选择一个目标嵌入作为水印，并将其插入包含触发词的文本的嵌入中作为后门。插入的重量与包含在文本中的触发词数量成比例。这使得水印后门可以有效地传输和恢复，而不影响 LLM 在各种 NLP 任务中的性能。实验证明，EmbMarker 可以在维护各种 NLP 任务的性能的同时成功保护 EaaS 对 LLM 的版权。

    Large language models (LLMs) have demonstrated powerful capabilities in both text understanding and generation. Companies have begun to offer Embedding as a Service (EaaS) based on these LLMs, which can benefit various natural language processing (NLP) tasks for customers. However, previous studies have shown that EaaS is vulnerable to model extraction attacks, which can cause significant losses for the owners of LLMs, as training these models is extremely expensive. To protect the copyright of LLMs for EaaS, we propose an Embedding Watermark method called EmbMarker that implants backdoors on embeddings. Our method selects a group of moderate-frequency words from a general text corpus to form a trigger set, then selects a target embedding as the watermark, and inserts it into the embeddings of texts containing trigger words as the backdoor. The weight of insertion is proportional to the number of trigger words included in the text. This allows the watermark backdoor to be effectively t
    
[^42]: 当梯度下降遇到无导数优化：黑盒场景下的完美组合

    When Gradient Descent Meets Derivative-Free Optimization: A Match Made in Black-Box Scenario. (arXiv:2305.10013v1 [cs.CL])

    [http://arxiv.org/abs/2305.10013](http://arxiv.org/abs/2305.10013)

    本文介绍了一种新方法GDFO，将梯度下降和无导数优化结合在一起，协调地优化任务特定的连续提示。实验证明，该方法优于现有的无梯度和基于梯度的方法。

    

    大型预训练语言模型（PLMs）因其多功能性和解决广泛自然语言处理（NLP）任务的潜力而备受关注。然而，运行这些PLMs的成本可能是禁止的。此外，由于商业考虑和潜在的误用风险（例如GPT-3），PLMs可能未开放源代码。在这种情况下，无导数优化（DFO）提出了黑盒调整的解决方案，用于训练任务特定的连续提示，而不是使用梯度下降。然而，与基于梯度的方法相比，这些无梯度方法仍然存在显着差距。本文通过知识蒸馏将梯度下降引入黑盒调整场景，并提出了一种新的方法GDFO，将梯度下降和无导数优化融合到一起，以协调的方式优化任务特定的连续提示。我们在各种NLP任务上进行了广泛的实验，并展示了我们提出的方法优于现有的无梯度和基于梯度的方法。

    Large pre-trained language models (PLMs) have garnered significant attention for their versatility and potential for solving a wide spectrum of natural language processing (NLP) tasks. However, the cost of running these PLMs may be prohibitive. Furthermore, PLMs may not be open-sourced due to commercial considerations and potential risks of misuse, such as GPT-3. The parameters and gradients of PLMs are unavailable in this scenario. To solve the issue, black-box tuning has been proposed, which utilizes derivative-free optimization (DFO), instead of gradient descent, for training task-specific continuous prompts. However, these gradient-free methods still exhibit a significant gap compared to gradient-based methods. In this paper, we introduce gradient descent into black-box tuning scenario through knowledge distillation. Furthermore, we propose a novel method GDFO, which integrates gradient descent and derivative-free optimization to optimize task-specific continuous prompts in a harmo
    
[^43]: 基于归因的知识蒸馏：语言模型压缩的方法

    AD-KD: Attribution-Driven Knowledge Distillation for Language Model Compression. (arXiv:2305.10010v1 [cs.CL])

    [http://arxiv.org/abs/2305.10010](http://arxiv.org/abs/2305.10010)

    本文提出了一种基于归因的知识蒸馏方法，通过 Integrated Gradients 探索教师模型的原理，将归因知识转移到学生模型，并在 GLUE 基准测试中使用BERT进行了全面实验，结果表明我们的方法具有更好的性能。

    

    知识蒸馏近来吸引了很多关注，因为它可以压缩预训练的语言模型。然而，现有的知识蒸馏方法存在两个限制：一是学生模型仅模仿教师的行为而忽略其潜在的推理过程；二是这些方法通常关注模型特定的复杂知识转移，但忽视数据特定的知识。本文提出了一种新的基于归因的知识蒸馏方法，通过 Integrated Gradients(IG) 探索教师模型背后的令牌级别的原理，并将归因知识转移到学生模型。为了增强模型推理和泛化的知识转移，我们进一步研究了教师的所有潜在决策的多视角归因蒸馏。我们还在GLUE基准测试中使用BERT进行了全面实验。实验结果表明，我们的方法比现有的方法具有更好的性能。

    Knowledge distillation has attracted a great deal of interest recently to compress pre-trained language models. However, existing knowledge distillation methods suffer from two limitations. First, the student model simply imitates the teacher's behavior while ignoring the underlying reasoning. Second, these methods usually focus on the transfer of sophisticated model-specific knowledge but overlook data-specific knowledge. In this paper, we present a novel attribution-driven knowledge distillation approach, which explores the token-level rationale behind the teacher model based on Integrated Gradients (IG) and transfers attribution knowledge to the student model. To enhance the knowledge transfer of model reasoning and generalization, we further explore multi-view attribution distillation on all potential decisions of the teacher. Comprehensive experiments are conducted with BERT on the GLUE benchmark. The experimental results demonstrate the superior performance of our approach to sev
    
[^44]: EfficientSCI: 稠密连接网络与时空分解相结合的大规模视频快照压缩成像

    EfficientSCI: Densely Connected Network with Space-time Factorization for Large-scale Video Snapshot Compressive Imaging. (arXiv:2305.10006v1 [cs.CV])

    [http://arxiv.org/abs/2305.10006](http://arxiv.org/abs/2305.10006)

    本文提出了EfficientSCI网络，通过使用稠密连接和时空分解机制来建立视频SCI中的空间-时间相关性。相比最先进的基于深度学习的方法，它能够在计算效率和重建质量方面取得更好的表现。

    

    视频快照压缩成像 (SCI) 使用二维检测器在单个曝光时间内捕获连续视频帧。然后需要设计高效的重建算法来重建所需的视频帧。虽然最近的基于深度学习的最新重建算法已经在大多数任务上取得了良好的结果，但它们仍然面临以下挑战：过度的模型复杂性和GPU内存限制。其中，1)这些模型需要高计算成本，2)它们通常不能在高压缩比下重建大规模的视频帧。为了解决这些问题，我们开发了一种高效的视频SCI网络，使用单个残差块内的稠密连接和时空分解机制，命名为EfficientSCI。 EfficientSCI网络可以通过在空间域中使用卷积和在时间域中使用转换域稀疏化（TDS）来很好地建立空间 - 时间相关性，这显著降低了计算成本和内存使用。实验结果表明，我们的EfficientSCI在各种大规模视频SCI数据集上，在重建质量和计算效率方面均优于最新的基于深度学习的方法。

    Video snapshot compressive imaging (SCI) uses a two-dimensional detector to capture consecutive video frames during a single exposure time. Following this, an efficient reconstruction algorithm needs to be designed to reconstruct the desired video frames. Although recent deep learning-based state-of-the-art (SOTA) reconstruction algorithms have achieved good results in most tasks, they still face the following challenges due to excessive model complexity and GPU memory limitations:  1) these models need high computational cost, and  2) they are usually unable to reconstruct large-scale video frames at high compression ratios.  To address these issues, we develop an {\bf{\em efficient network}} for video SCI by using {\bf {\em dense connections and space-time factorization mechanism}} within a single residual block, dubbed {\bf \emph{EfficientSCI}}. The EfficientSCI network can well establish spatial-temporal correlation by using {\bf {\em convolution in the spatial domain and Transform
    
[^45]: DinoSR：自监督语音表示学习中的自蒸馏和在线聚类

    DinoSR: Self-Distillation and Online Clustering for Self-supervised Speech Representation Learning. (arXiv:2305.10005v1 [cs.CL])

    [http://arxiv.org/abs/2305.10005](http://arxiv.org/abs/2305.10005)

    本研究提出了DinoSR模型，它结合了遮蔽语言建模、自蒸馏和在线聚类等概念，能够在自监督语音表示学习任务中产生很好的效果，超越了以前的最新技术水平。

    

    本文介绍了自蒸馏和在线聚类用于自监督语音表示学习的DinoSR模型，它结合了遮蔽语言建模、自蒸馏和在线聚类这些概念，并展示它们相互补充，形成了一种强大的语音表示学习模型。DinoSR首先使用教师网络从输入音频中提取上下文化的嵌入向量，然后在嵌入向量上运行在线聚类系统以产生机器发现的音素库存，最后使用已离散化的标记指导学生网络。我们展示了DinoSR在多个下游任务中超越了以前的最新技术，并提供了模型和学习离散单元的详细分析。匿名期结束后，我们将提供源代码。

    In this paper, we introduce self-distillation and online clustering for self-supervised speech representation learning (DinoSR) which combines masked language modeling, self-distillation, and online clustering. We show that these concepts complement each other and result in a strong representation learning model for speech. DinoSR first extracts contextualized embeddings from the input audio with a teacher network, then runs an online clustering system on the embeddings to yield a machine-discovered phone inventory, and finally uses the discretized tokens to guide a student network. We show that DinoSR surpasses previous state-of-the-art performance in several downstream tasks, and provide a detailed analysis of the model and the learned discrete units. The source code will be made available after the anonymity period.
    
[^46]: Reprompting: 通过吉布斯采样自动推断思维链的提示

    Reprompting: Automated Chain-of-Thought Prompt Inference Through Gibbs Sampling. (arXiv:2305.09993v1 [cs.LG])

    [http://arxiv.org/abs/2305.09993](http://arxiv.org/abs/2305.09993)

    Reprompting是一种无需人类干预的算法，通过迭代采样新配方解决多步推理任务，比人类编写的思维链提示表现更好，还可以提高较弱模型的性能。

    

    我们引入了Reprompting，这是一种迭代采样算法，可以在没有人类干预的情况下搜索给定任务的思维链配方。通过吉布斯采样，我们推断适用于一组训练样例的思维链配方。我们的方法使用先前采样的解作为父提示，迭代地采样新的配方来解决其他训练问题。在需要多步推理的五个Big-Bench Hard任务中，Reprompting的表现始终优于零样本、少样本和人类编写的思维链基线。Reprompting还可以促进知识从一个更强的模型到一个较弱的模型的转移，从而大大提高了较弱模型的性能。总体而言，Reprompting相对于使用人类编写的思维链提示的先前最先进方法，带来了高达+17个性能改进。

    We introduce Reprompting, an iterative sampling algorithm that searches for the Chain-of-Thought (CoT) recipes for a given task without human intervention. Through Gibbs sampling, we infer CoT recipes that work consistently well for a set of training samples. Our method iteratively samples new recipes using previously sampled solutions as parent prompts to solve other training problems. On five Big-Bench Hard tasks that require multi-step reasoning, Reprompting achieves consistently better performance than the zero-shot, few-shot, and human-written CoT baselines. Reprompting can also facilitate transfer of knowledge from a stronger model to a weaker model leading to substantially improved performance of the weaker model. Overall, Reprompting brings up to +17 point improvements over the previous state-of-the-art method that uses human-written CoT prompts.
    
[^47]: 双重语义知识组合的多模态对话系统

    Dual Semantic Knowledge Composed Multimodal Dialog Systems. (arXiv:2305.09990v1 [cs.CL])

    [http://arxiv.org/abs/2305.09990](http://arxiv.org/abs/2305.09990)

    设计了一种双重语义知识组合的多模态任务导向对话系统，通过多级知识组合机制和表示级规范化方法解决了现有研究存在的关键限制，实现了有效响应生成。

    

    文本生成是多模态任务导向对话系统中的一个基本任务。尽管现有的研究已经取得了丰硕的进展，但仍存在两个关键限制：1）注重属性知识，但忽略了可以揭示不同实体之间相关性的关系知识，从而促进了响应生成；2）只进行基于交叉熵损失的输出级监督，缺乏表示级的规范化。为了解决这些限制，我们设计了一种新颖的多模态任务导向对话系统（名为MDS-S2）。特别是，MDS-S2首先同时从知识库中获取与上下文相关的属性和关系知识，其中非直观的关系知识是通过n跳图形遍历提取的。此后，考虑到属性知识和关系知识可以有助于响应不同级别的问题，我们设计了一种多级知识组合机制来有效地融合它们。此外，为了强制属性关系一致性并提高响应的多样性，我们提出了一种新颖的表示级规范化方法，专门针对双重语义知识组合的模型。在两个基准数据集上的广泛实验结果表明，我们提出的方法在自动评估指标和人类评估方面均具有优越性。

    Textual response generation is an essential task for multimodal task-oriented dialog systems.Although existing studies have achieved fruitful progress, they still suffer from two critical limitations: 1) focusing on the attribute knowledge but ignoring the relation knowledge that can reveal the correlations between different entities and hence promote the response generation}, and 2) only conducting the cross-entropy loss based output-level supervision but lacking the representation-level regularization. To address these limitations, we devise a novel multimodal task-oriented dialog system (named MDS-S2). Specifically, MDS-S2 first simultaneously acquires the context related attribute and relation knowledge from the knowledge base, whereby the non-intuitive relation knowledge is extracted by the n-hop graph walk. Thereafter, considering that the attribute knowledge and relation knowledge can benefit the responding to different levels of questions, we design a multi-level knowledge comp
    
[^48]: 写作辅助智能词汇建议

    Smart Word Suggestions for Writing Assistance. (arXiv:2305.09975v1 [cs.CL])

    [http://arxiv.org/abs/2305.09975](http://arxiv.org/abs/2305.09975)

    本论文介绍了写作辅助中的“智能词汇建议”（SWS）任务和基准测试，强调了端到端的评估和更加现实的写作辅助场景。它包含一组数据集和提供替换建议的难题，提供了训练和测试基础及潜在的研究方向。

    

    提高词汇使用是写作辅助中一项重要功能，本论文介绍了“智能词汇建议”（SWS）任务和基准测试。与其他工作不同，SWS强调端到端的评估并提出更加现实的写作辅助场景。该任务涉及识别需要改进的单词或短语，并提供替换建议。该基准测试包括被人工标记的测试数据，一个用于训练的大型遥感数据集，以及评估框架。测试数据包括1000个由英语学习者写成的句子，附带10个母语为英语的人注释的超过16000个替换建议。训练数据集包括超过370万个句子和通过规则生成的1270万个建议。我们对七个基线模型进行的实验表明，SWS是一个具有挑战性的任务。根据实验分析，我们提出了未来研究SWS的潜在方向。

    Enhancing word usage is a desired feature for writing assistance. To further advance research in this area, this paper introduces "Smart Word Suggestions" (SWS) task and benchmark. Unlike other works, SWS emphasizes end-to-end evaluation and presents a more realistic writing assistance scenario. This task involves identifying words or phrases that require improvement and providing substitution suggestions. The benchmark includes human-labeled data for testing, a large distantly supervised dataset for training, and the framework for evaluation. The test data includes 1,000 sentences written by English learners, accompanied by over 16,000 substitution suggestions annotated by 10 native speakers. The training dataset comprises over 3.7 million sentences and 12.7 million suggestions generated through rules. Our experiments with seven baselines demonstrate that SWS is a challenging task. Based on experimental analysis, we suggest potential directions for future research on SWS. The dataset 
    
[^49]: CooK: 用模块化和协作知识赋能通用语言模型

    CooK: Empowering General-Purpose Language Models with Modular and Collaborative Knowledge. (arXiv:2305.09955v1 [cs.CL])

    [http://arxiv.org/abs/2305.09955](http://arxiv.org/abs/2305.09955)

    CooK是一种用于赋能通用语言模型的新颖框架，通过专门的语言模型和协作的知识贡献者，提供模块化、不断增长和多源的知识。在知识密集型任务中，CooK展现出了明显的性能提升。

    

    大型语言模型（LLM）越来越多地用于知识密集型任务和语境中。现有方法通过检索或生成知识提示来改善通用语言模型的知识能力，但它们未能反映知识丰富模型的两个关键属性：知识应该是模块化，不断增长，来自不同领域；知识获取和生成应该是协作的过程，其中各种利益相关者 contribue 新信息。为此，我们提出了 CooK，一种新颖的框架，可为通用大型语言模型提供模块化和协作来源的知识。我们首先介绍了专门的语言模型，即在广泛领域和来源上训练的自回归模型。这些专门的语言模型可以作为参数化的知识库，后来被提示生成通用的 LLM 的背景知识。然后，我们提出了三个知识过滤器，以动态选择适合给定上下文的知识源。最后，我们呈现了一个知识贡献者组件，使利益相关者能够轻松地为系统贡献特定于域的知识。我们展示了 CooK 在一组知识密集型任务上的有效性，显示出明显的超越现有技术的性能。

    Large language models (LLMs) are increasingly adopted for knowledge-intensive tasks and contexts. Existing approaches improve the knowledge capabilities of general-purpose LLMs through retrieval or generated knowledge prompting, but they fall short of reflecting two key properties of knowledge-rich models: knowledge should be modular, ever-growing, sourced from diverse domains; knowledge acquisition and production should be a collaborative process, where diverse stakeholders contribute new information. To this end, we propose CooK, a novel framework to empower general-purpose large language models with modular and collaboratively sourced knowledge. We first introduce specialized language models, autoregressive models trained on corpora from a wide range of domains and sources. These specialized LMs serve as parametric knowledge repositories that are later prompted to generate background knowledge for general-purpose LLMs. We then propose three knowledge filters to dynamically select an
    
[^50]: “我全然成为我自己”：以TGNB人群为中心，评估开放式语言生成中的偏见

    "I'm fully who I am": Towards Centering Transgender and Non-Binary Voices to Measure Biases in Open Language Generation. (arXiv:2305.09941v1 [cs.CL])

    [http://arxiv.org/abs/2305.09941](http://arxiv.org/abs/2305.09941)

    本论文研究了如何以TGNB人群的声音为中心，评估开放式语言生成中的偏见。通过理解TGNB个体的经历，提出了以TGNB人群为中心的OLG系统评估框架，并且包括一个为TGNB人群设计的调查工具和分析方法。

    

    跨性别和非二元（TGNB）人群在日常生活中经历了不成比例的歧视和排斥。随着语言生成技术的日益普及和应用，进一步边缘化这一人群的可能性也在增加。虽然大量的NLP公平文献着重于阐明和解决性别偏见，但评估TGNB身份所带来的性别伤害需要理解这些身份如何独特地与社会性别规范互动以及与性别二元中心的视角相区分。这样的测量框架本质上需要以TGNB声音为中心，帮助指导包容性别的自然语言处理应该为谁服务。为实现这一目标，我们以TGNB社区和现有的跨学科文献为基础，评估了TGNB个体经历边缘化所形成的社会现实是如何影响和存在于开放式语言生成（OLG）中。首先理解TGNB个体的经历，我们提出了一个评估OLG系统的框架，旨在以TGNB人群为中心，度量与该人群相关的偏见。我们的框架包括特别为TGNB人群设计的调查工具，以及交叉分析结果的交叉方法。我们相信，这项工作将有助于实现更公平、更包容的自然语言处理社区，并潜在地解决NLP研究中广泛的交叉身份问题。

    Transgender and non-binary (TGNB) individuals disproportionately experience discrimination and exclusion from daily life. Given the recent popularity and adoption of language generation technologies, the potential to further marginalize this population only grows. Although a multitude of NLP fairness literature focuses on illuminating and addressing gender biases, assessing gender harms for TGNB identities requires understanding how such identities uniquely interact with societal gender norms and how they differ from gender binary-centric perspectives. Such measurement frameworks inherently require centering TGNB voices to help guide the alignment between gender-inclusive NLP and whom they are intended to serve. Towards this goal, we ground our work in the TGNB community and existing interdisciplinary literature to assess how the social reality surrounding experienced marginalization by TGNB persons contributes to and persists within Open Language Generation (OLG). By first understandi
    
[^51]: 基于预训练模型的等变小样本学习

    Equivariant Few-Shot Learning from Pretrained Models. (arXiv:2305.09900v1 [cs.LG])

    [http://arxiv.org/abs/2305.09900](http://arxiv.org/abs/2305.09900)

    本文提出了一种基于预训练模型的$\lambda$-\textit{equitune}方法，它使用\textit{重要性权重}$\lambda$对特征进行平均，可以显著提高等变小样本学习的表现。

    

    高效的迁移学习算法是基础模型在有限数据情况下在各种下游任务上取得成功的关键。最近的作品 \cite{basu2022equi} 和 \cite{kaba2022equivariance} 分别提出了使用从群变换输入得到的特征的群平均值（\textit{equitune}）和基于优化的方法来从不等变的神经网络获取等变输出。虽然 \cite{kaba2022equivariance} 只关注从头开始训练，但我们发现即使在良好的微调结果下，\textit{equitune} 在等变零样本任务上表现不佳。我们认为这是因为预训练模型为某些转换提供了更高质量的特征，而对其进行简单平均会产生不良影响。因此，我们提出了一种使用\textit{重要性权重}$\lambda$对特征进行平均的$\lambda$-\textit{equitune} 方法。这些权重是使用一个小型神经网络直接从数据中学习的，从而导致出色的零样本和微调结果。

    Efficient transfer learning algorithms are key to the success of foundation models on diverse downstream tasks even with limited data. Recent works of \cite{basu2022equi} and \cite{kaba2022equivariance} propose group averaging (\textit{equitune}) and optimization-based methods, respectively, over features from group-transformed inputs to obtain equivariant outputs from non-equivariant neural networks. While \cite{kaba2022equivariance} are only concerned with training from scratch, we find that equitune performs poorly on equivariant zero-shot tasks despite good finetuning results. We hypothesize that this is because pretrained models provide better quality features for certain transformations than others and simply averaging them is deleterious. Hence, we propose $\lambda$-\textit{equitune} that averages the features using \textit{importance weights}, $\lambda$s. These weights are learned directly from the data using a small neural network, leading to excellent zero-shot and finetuned 
    
[^52]: 抽象化摘要中平衡词汇和语义质量的方法

    Balancing Lexical and Semantic Quality in Abstractive Summarization. (arXiv:2305.09898v1 [cs.CL])

    [http://arxiv.org/abs/2305.09898](http://arxiv.org/abs/2305.09898)

    本文提出了一种新的训练方法，其中重新排列硬件平衡了词汇和语义质量，以缓解抽象化摘要中的暴露偏差问题。

    

    序列到序列的神经模型在抽象化摘要中被广泛应用，但是曝光偏差是一个重要的问题。为了缓解这个问题，最近几年一直使用重新排序系统。尽管有些性能改进，但这个方法仍然不太成熟。以前的工作大多是通过ROUGE分数和对齐候选摘要来指定排名，但词汇重叠指标和语义相似度之间可能存在很大差距。在本文中，我们提出了一种新的训练方法，在其中重新排列程序平衡词汇和语义质量。我们进一步重新定义了排名中的假阳性，并提出了一种减少它们影响的策略。在CNN / DailyMail和XSum数据集上的实验表明，我们的方法可以估计摘要的含义，而不会严重降低词汇方面的质量。具体来说，我们在CNN / DailyMail数据集上实现了89.67的BERTScore，达到了新的最先进性能。我们的代码是公开的。

    An important problem of the sequence-to-sequence neural models widely used in abstractive summarization is exposure bias. To alleviate this problem, re-ranking systems have been applied in recent years. Despite some performance improvements, this approach remains underexplored. Previous works have mostly specified the rank through the ROUGE score and aligned candidate summaries, but there can be quite a large gap between the lexical overlap metric and semantic similarity. In this paper, we propose a novel training method in which a re-ranker balances the lexical and semantic quality. We further newly define false positives in ranking and present a strategy to reduce their influence. Experiments on the CNN/DailyMail and XSum datasets show that our method can estimate the meaning of summaries without seriously degrading the lexical aspect. More specifically, it achieves an 89.67 BERTScore on the CNN/DailyMail dataset, reaching new state-of-the-art performance. Our code is publicly availa
    
[^53]: 面向聚类的负采样用于无监督句子表示学习

    Clustering-Aware Negative Sampling for Unsupervised Sentence Representation. (arXiv:2305.09892v1 [cs.CL])

    [http://arxiv.org/abs/2305.09892](http://arxiv.org/abs/2305.09892)

    ClusterNS 是一种将聚类信息引入对比学习进行无监督句子表示学习的新方法，通过改进的 K 均值聚类算法提供难负例并识别错误负例，旨在通过一个统一的框架解决问题，实验结果表明其在无监督句子表示学习中表现优于基线。

    

    对比学习在句子表示学习中广泛研究，然而早期的研究主要集中在正例的构建上，而 batch 内的样本通常被视为负例。这种方法忽视了选择合适的负例的重要性，可能会导致难负例的稀缺性和错误负例的包含。为了解决这些问题，我们提出了 ClusterNS（面向聚类的负采样），一种将聚类信息引入对比学习进行无监督句子表示学习的新方法。我们应用改进的 K 均值聚类算法来提供难负例并识别训练过程中的错误负例，旨在通过一个统一的框架解决这两个问题。在语义文本相似度（STS）任务上的实验表明，我们提出的 ClusterNS 在无监督句子表示学习中表现优于基线。我们的代码已上传到 https://github.com/xxxxxx。

    Contrastive learning has been widely studied in sentence representation learning. However, earlier works mainly focus on the construction of positive examples, while in-batch samples are often simply treated as negative examples. This approach overlooks the importance of selecting appropriate negative examples, potentially leading to a scarcity of hard negatives and the inclusion of false negatives. To address these issues, we propose ClusterNS (Clustering-aware Negative Sampling), a novel method that incorporates cluster information into contrastive learning for unsupervised sentence representation learning. We apply a modified K-means clustering algorithm to supply hard negatives and recognize in-batch false negatives during training, aiming to solve the two issues in one unified framework. Experiments on semantic textual similarity (STS) tasks demonstrate that our proposed ClusterNS compares favorably with baselines in unsupervised sentence representation learning. Our code has been
    
[^54]: 基于机器学习和关键词感知交叉编码器排序摘要程序的自然语言文本语义相似度度量——以UCGIS GIS&T知识体系为案例研究

    Semantic Similarity Measure of Natural Language Text through Machine Learning and a Keyword-Aware Cross-Encoder-Ranking Summarizer -- A Case Study Using UCGIS GIS&T Body of Knowledge. (arXiv:2305.09877v1 [cs.CL])

    [http://arxiv.org/abs/2305.09877](http://arxiv.org/abs/2305.09877)

    本文提出了一种新方法，采用机器学习模型和关键词感知交叉编码器排序摘要程序，从文本内容中提取语义信息，并度量 GIS&T BoK 话题之间的语义相似度，以解决手动定义话题关系带来的不完整评估问题。该方法在准确度量话题关系方面表现良好，对 GIS&T 领域的研究和实践具有重要意义。

    

    GIS&T 知识体系是由地理信息科学与技术相关团体发起的一个社区项目，旨在定义、开发和记录地理信息科学与技术相关话题。本文提出了一种新方法，采用机器学习模型和关键词感知交叉编码器排序摘要程序，从文本内容中提取语义信息，并度量 BoK 话题之间的语义相似度。结果表明，我们的方法在识别 BoK 话题之间的语义相似度方面优于其他 NLP 技术。该方法能够自动且准确地度量话题之间的关系，从而使 GIS&T 领域的研究人员和实践者受益。

    Initiated by the University Consortium of Geographic Information Science (UCGIS), GIS&T Body of Knowledge (BoK) is a community-driven endeavor to define, develop, and document geospatial topics related to geographic information science and technologies (GIS&T). In recent years, GIS&T BoK has undergone rigorous development in terms of its topic re-organization and content updating, resulting in a new digital version of the project. While the BoK topics provide useful materials for researchers and students to learn about GIS, the semantic relationships among the topics, such as semantic similarity, should also be identified so that a better and automated topic navigation can be achieved. Currently, the related topics are either defined manually by editors or authors, which may result in an incomplete assessment of topic relationship. To address this challenge, our research evaluates the effectiveness of multiple natural language processing (NLP) techniques in extracting semantics from te
    
[^55]: Jaseci编程范式和运行时堆栈：轻松快速构建规模化生产应用程序

    The Jaseci Programming Paradigm and Runtime Stack: Building Scale-out Production Applications Easy and Fast. (arXiv:2305.09864v1 [cs.CL])

    [http://arxiv.org/abs/2305.09864](http://arxiv.org/abs/2305.09864)

    Jaseci编程范式和运行时堆栈的设计原则在于通过自动化和自动优化尽可能多的规模化数据管理、微服务组件化和实时更新复杂性，从而提高了抽象水平，降低了应用程序的开发难度和部署门槛。

    

    当今的生产规模化应用程序包括许多子应用程序组件，例如存储后端、日志基础设施和AI模型。这些组件具有非常不同的特性，需要协同工作，并作为微服务与彼此接口。这导致在开发、优化、配置和部署规模化应用程序方面越来越复杂，提高了大多数个人和小团队的准入门槛。我们开发了一种新颖的协同设计运行时系统Jaseci和编程语言Jac，旨在减少这种复杂性。Jaseci设计和开发的关键设计原则是通过将尽可能多的规模化数据管理、微服务组件化和实时更新复杂性移入运行时堆栈以进行自动化和自动优化来提高抽象水平。我们使用真实世界的AI应用程序来展示Jaseci在应用程序性能和开发人员生产力方面的好处。

    Today's production scale-out applications include many sub-application components, such as storage backends, logging infrastructure and AI models. These components have drastically different characteristics, are required to work in collaboration, and interface with each other as microservices. This leads to increasingly high complexity in developing, optimizing, configuring, and deploying scale-out applications, raising the barrier to entry for most individuals and small teams. We developed a novel co-designed runtime system, Jaseci, and programming language, Jac, which aims to reduce this complexity. The key design principle throughout Jaseci's design is to raise the level of abstraction by moving as much of the scale-out data management, microservice componentization, and live update complexity into the runtime stack to be automated and optimized automatically. We use real-world AI applications to demonstrate Jaseci's benefit for application performance and developer productivity.
    
[^56]: 利用语言模型用自然语言解释黑盒文本模块

    Explaining black box text modules in natural language with language models. (arXiv:2305.09863v1 [cs.AI])

    [http://arxiv.org/abs/2305.09863](http://arxiv.org/abs/2305.09863)

    本文介绍了一种名为Summarize and Score（SASC）的方法，该方法可以自动获取黑盒文本模块的自然语言解释以及解释可靠程度的分数。研究者们已经在合成模块和BERT模型中使用SASC，让我们可以解释模块的选择性，这对于增强大型语言模型的可解释性非常重要。

    

    大型语言模型已经证明在各种任务中具有出色的预测性能。然而，它们的快速增长和不透明性已经引起了对可解释性的需求。本文询问是否可以自动获取黑盒文本模块的自然语言解释。一个“文本模块”是将文本映射到标量连续值的任何函数，例如LLM内的子模块或大脑区域的拟合模型。“黑盒”表示我们只能访问模块的输入/输出。我们引入了Summarize and Score（SASC）方法，它接受文本模块并返回模块选择性的自然语言解释以及解释可靠程度的分数。我们在三个上下文中研究SASC。首先，我们在合成模块上评估SASC，并发现它经常恢复基本真相说明。其次，我们使用SASC来解释预训练BERT模型中的模块，使得检查BERT的模块成为可能。

    Large language models (LLMs) have demonstrated remarkable prediction performance for a growing array of tasks. However, their rapid proliferation and increasing opaqueness have created a growing need for interpretability. Here, we ask whether we can automatically obtain natural language explanations for black box text modules. A "text module" is any function that maps text to a scalar continuous value, such as a submodule within an LLM or a fitted model of a brain region. "Black box" indicates that we only have access to the module's inputs/outputs.  We introduce Summarize and Score (SASC), a method that takes in a text module and returns a natural language explanation of the module's selectivity along with a score for how reliable the explanation is. We study SASC in 3 contexts. First, we evaluate SASC on synthetic modules and find that it often recovers ground truth explanations. Second, we use SASC to explain modules found within a pre-trained BERT model, enabling inspection of the 
    
[^57]: Epsilon Sampling Rocks: 研究用于机器翻译最小贝叶斯风险解码的采样策略

    Epsilon Sampling Rocks: Investigating Sampling Strategies for \\Minimum Bayes Risk Decoding for Machine Translation. (arXiv:2305.09860v1 [cs.CL])

    [http://arxiv.org/abs/2305.09860](http://arxiv.org/abs/2305.09860)

    本文研究了用于机器翻译最小贝叶斯风险解码的不同采样策略，并发现了epsilon采样方式能够使得解码结果显著地优于其他所有已测试的采样方式和束搜索解码。

    

    机器翻译中的最小贝叶斯风险（MBR）解码已经显示出是一种强大的替代束搜索解码的方法，尤其是与基于神经网络的效用函数相结合时。然而，MBR解码的性能严重依赖于从模型中采样的方法和数量。本文探讨了用于MBR解码的不同采样方法对性能的影响。我们评估了一些流行的采样方法，例如祖先采样，核采样和top-k采样。基于我们对它们局限性的认识，我们尝试了最近提出的epsilon采样方法，该方法通过修剪所有小于epsilon的标记，以确保样本中的每个标记获得公平的概率质量。通过广泛的人类评估，我们证明了基于epsilon采样的MBR解码显著优于不仅是束搜索解码，而且还优于所有其他已测试的采样方法的MBR解码。

    Recent advances in machine translation (MT) have shown that Minimum Bayes Risk (MBR) decoding can be a powerful alternative to beam search decoding, especially when combined with neural-based utility functions. However, the performance of MBR decoding depends heavily on how and how many candidates are sampled from the model. In this paper, we explore how different sampling approaches for generating candidate lists for MBR decoding affect performance. We evaluate popular sampling approaches, such as ancestral, nucleus, and top-k sampling. Based on our insights into their limitations, we experiment with the recently proposed epsilon-sampling approach, which prunes away all tokens with a probability smaller than epsilon, ensuring that each token in a sample receives a fair probability mass. Through extensive human evaluations, we demonstrate that MBR decoding based on epsilon-sampling significantly outperforms not only beam search decoding, but also MBR decoding with all other tested samp
    
[^58]: 小型语言模型更适合作为黑匣子机器生成文本检测器

    Smaller Language Models are Better Black-box Machine-Generated Text Detectors. (arXiv:2305.09859v1 [cs.CL])

    [http://arxiv.org/abs/2305.09859](http://arxiv.org/abs/2305.09859)

    本文研究发现，小型语言模型更适用于作为通用文本检测器，可以更加精确地检测出机器生成的文本，而检测器和生成模型是否具有相同的架构或语料库并不会对检测性能产生显著影响。

    

    随着流畅的生成语言模型的出现，它们可以生成与人类写作的非常相似的令人信服的话语，因此区分一段文本是由机器生成的还是人类写作的变得更加具有挑战性和重要性，因为这样的模型可以用于传播错误信息、虚假新闻、虚假评论并模仿某些作者和人物。为此，已经提出了许多检测机器生成文本的方法。其中大部分方法需要访问目标模型的 logits，或需要可以从目标模型中进行采样的能力。其中一种黑匣子检测方法依赖于观察到生成文本在生成器的似然函数下是局部最优的，而人类写作的文本则不是。我们发现，总体而言，较小且部分训练的模型更适合作为通用文本检测器：它们可以更精确地检测来自小型和大型模型的生成文本。有趣的是，我们发现检测器和生成模型是否具有相同的架构或相同的语料库对检测性能没有显著影响。

    With the advent of fluent generative language models that can produce convincing utterances very similar to those written by humans, distinguishing whether a piece of text is machine-generated or human-written becomes more challenging and more important, as such models could be used to spread misinformation, fake news, fake reviews and to mimic certain authors and figures. To this end, there have been a slew of methods proposed to detect machine-generated text. Most of these methods need access to the logits of the target model or need the ability to sample from the target. One such black-box detection method relies on the observation that generated text is locally optimal under the likelihood function of the generator, while human-written text is not. We find that overall, smaller and partially-trained models are better universal text detectors: they can more precisely detect text generated from both small and larger models. Interestingly, we find that whether the detector and generat
    
[^59]: 知识图谱补全模型是少样本学习者：以 LLMS 在电商中的关系标注为例的经验研究

    Knowledge Graph Completion Models are Few-shot Learners: An Empirical Study of Relation Labeling in E-commerce with LLMs. (arXiv:2305.09858v1 [cs.IR])

    [http://arxiv.org/abs/2305.09858](http://arxiv.org/abs/2305.09858)

    本文通过对知识图谱中关系标注的实证研究，发现大型语言模型具有强大的学习能力以及在少量标记数据下预测产品类型之间关系的有效性。

    

    知识图谱在增强电子商务系统性能方面发挥着至关重要的作用，提供了关于实体及其关系的结构化信息，例如产品或产品类型之间的互补或替代关系，这些信息可以在推荐系统中利用。然而，由于电子商务领域的动态性和人力成本相关的原因，知识图谱中的关系标注仍然是一个具有挑战性的任务。最近，大型语言模型（LLM）的突破在许多自然语言处理任务中展示了出乎意料的结果。在本文中，我们进行了一个关于 LLM 在电子商务知识图谱中进行关系标注的实证研究，研究它们在自然语言方面强大的学习能力以及在有限标记数据下预测产品类型之间关系的有效性。我们评估了各种 LLM，包括 PaLM 和 GPT-3.5，在基准数据集上，证明它们能够达到与人类相当的关系性能水平。

    Knowledge Graphs (KGs) play a crucial role in enhancing e-commerce system performance by providing structured information about entities and their relationships, such as complementary or substitutable relations between products or product types, which can be utilized in recommender systems. However, relation labeling in KGs remains a challenging task due to the dynamic nature of e-commerce domains and the associated cost of human labor. Recently, breakthroughs in Large Language Models (LLMs) have shown surprising results in numerous natural language processing tasks. In this paper, we conduct an empirical study of LLMs for relation labeling in e-commerce KGs, investigating their powerful learning capabilities in natural language and effectiveness in predicting relations between product types with limited labeled data. We evaluate various LLMs, including PaLM and GPT-3.5, on benchmark datasets, demonstrating their ability to achieve competitive performance compared to humans on relation
    
[^60]: CoEdIT：通过任务特定指令调整实现文本编辑

    CoEdIT: Text Editing by Task-Specific Instruction Tuning. (arXiv:2305.09857v1 [cs.CL])

    [http://arxiv.org/abs/2305.09857](http://arxiv.org/abs/2305.09857)

    CoEdIT是一种通过任务特定指令调整实现文本编辑的最先进模型，能够提高用户生成文本的质量和提高流程的效率。

    

    文本编辑或修订是人类写作过程中必不可少的功能。理解LLMs在进行高质量修订和与人类写作者协作方面的能力是构建有效写作助手的关键步骤。在LLMs和指令调整的先前成功基础上，我们利用经过指令调整的LLMs进行文本修订，以提高用户生成文本的质量和提高流程的效率。我们引入了CoEdIT，这是一款用于写作辅助的最先进的文本编辑模型。CoEdIT从用户那里获取指令，指定所需文本的属性，例如“使句子更简单”或“以更中立的风格写作”，并输出编辑后的文本。我们提供了一个大型语言模型，该模型在各种文本编辑基准测试上实现了最先进的性能。我们的模型（1）在各种文本编辑基准测试上实现最先进的性能，（2）与公开可用的模型相比具有竞争力。

    Text editing or revision is an essential function of the human writing process. Understanding the capabilities of LLMs for making high-quality revisions and collaborating with human writers is a critical step toward building effective writing assistants. With the prior success of LLMs and instruction tuning, we leverage instruction-tuned LLMs for text revision to improve the quality of user-generated text and improve the efficiency of the process. We introduce CoEdIT, a state-of-the-art text editing model for writing assistance. CoEdIT takes instructions from the user specifying the attributes of the desired text, such as "Make the sentence simpler" or "Write it in a more neutral style," and outputs the edited text. We present a large language model fine-tuned on a diverse collection of task-specific instructions for text editing (a total of 82K instructions). Our model (1) achieves state-of-the-art performance on various text editing benchmarks, (2) is competitive with publicly availa
    
[^61]: 基于上下文的提示式学习用于在线社区违规检测

    CPL-NoViD: Context-Aware Prompt-based Learning for Norm Violation Detection in Online Communities. (arXiv:2305.09846v1 [cs.CL])

    [http://arxiv.org/abs/2305.09846](http://arxiv.org/abs/2305.09846)

    本文提出了一种新的方法（CPL-NoViD），通过自然语言提示将上下文融入到模型中，用于在线社区中的违规检测。该方法能够适应不同社区中的各种规则和解释的差异，在跨规则类型和跨社区的违规行为检测中表现出色，并在少样本学习场景中表现出一定的适应性。

    

    在线社区中检测违规行为对于维护健康和安全的在线讨论空间至关重要。现有的机器学习方法往往难以适应不同社区之间各种规则和解释的差异，因为为这种特定上下文的任务微调模型具有困难。本文介绍了基于上下文提示的学习用于检测不同类型规则下的违规行为（CPL-NoViD），一种新的方法。CPL-NoViD通过自然语言提示来将上下文融入到模型中，对不同类型规则的表现也得到了改善，不仅在跨规则类型和跨社区的违规行为检测中表现出色，而且在少样本学习场景中也表现出一定的适应性。尤其值得注意的是，它建立了一个新的违规检测新的最高水平，超过了现有的基准。

    Detecting norm violations in online communities is critical to maintaining healthy and safe spaces for online discussions. Existing machine learning approaches often struggle to adapt to the diverse rules and interpretations across different communities due to the inherent challenges of fine-tuning models for such context-specific tasks. In this paper, we introduce Context-aware Prompt-based Learning for Norm Violation Detection (CPL-NoViD), a novel method that employs prompt-based learning to detect norm violations across various types of rules. CPL-NoViD outperforms the baseline by incorporating context through natural language prompts and demonstrates improved performance across different rule types. Significantly, it not only excels in cross-rule-type and cross-community norm violation detection but also exhibits adaptability in few-shot learning scenarios. Most notably, it establishes a new state-of-the-art in norm violation detection, surpassing existing benchmarks. Our work high
    
[^62]: 关于transformer主动学习中数据集可迁移性的研究

    On Dataset Transferability in Active Learning for Transformers. (arXiv:2305.09807v1 [cs.LG])

    [http://arxiv.org/abs/2305.09807](http://arxiv.org/abs/2305.09807)

    本文研究了基于transformer的预训练语言模型的主动学习中数据集的可迁移性问题，发现具有相似获取序列的主动学习方法产生的数据集在不同模型之间具有高度的可迁移性。

    

    主动学习旨在通过查询对模型学习最有益的示例来减少标注成本。尽管已经证明了对于微调基于transformer的预训练语言模型（PLMs），主动学习的有效性，但不清楚一个模型中获得的主动学习收益在多大程度上适用于其他模型。我们考虑在文本分类中积极获取的数据集的可迁移性问题，并调查了使用主动学习构建的数据集在使用不同PLM训练时能否保持AL收益。我们将AL数据集的可迁移性与不同PLMs查询到的实例的相似性联系起来，并表明具有类似获取序列的AL方法生成的数据集非常具有可迁移性，无论使用哪种模型。此外，我们表明，获取序列的相似性更受到AL方法的选择而非模型的影响。

    Active learning (AL) aims to reduce labeling costs by querying the examples most beneficial for model learning. While the effectiveness of AL for fine-tuning transformer-based pre-trained language models (PLMs) has been demonstrated, it is less clear to what extent the AL gains obtained with one model transfer to others. We consider the problem of transferability of actively acquired datasets in text classification and investigate whether AL gains persist when a dataset built using AL coupled with a specific PLM is used to train a different PLM. We link the AL dataset transferability to the similarity of instances queried by the different PLMs and show that AL methods with similar acquisition sequences produce highly transferable datasets regardless of the models used. Additionally, we show that the similarity of acquisition sequences is influenced more by the choice of the AL method than the choice of the model.
    
[^63]: 词语的方式：词语选择对信息参与和决策制定的影响

    The Ways of Words: The Impact of Word Choice on Information Engagement and Decision Making. (arXiv:2305.09798v1 [cs.CL])

    [http://arxiv.org/abs/2305.09798](http://arxiv.org/abs/2305.09798)

    措辞对信息参与和决策制定有重要影响，信息参与是由信息本身的表达所驱动和培育的。

    

    本研究探讨了措辞，特别是词语选择，对信息参与和决策制定的影响。综合了用户参与理论和信息行为理论两个理论模型，制定了一个理论框架，并生成了假设。该框架在一个大规模的用户研究中得到了实证验证，在衡量词语选择如何影响信息参与的感知、参与度和坚持度等三个方面提供了证据。结果表明，信息参与不同于其他形式的参与，它是由信息本身的表达所驱动和培育的，无论使用何种信息系统来查看、交互和使用信息。这些研究结果表明，措辞可以对信息参与和决策制定产生重要影响。

    Little research has explored how information engagement (IE), the degree to which individuals interact with and use information in a manner that manifests cognitively, behaviorally, and affectively. This study explored the impact of phrasing, specifically word choice, on IE and decision making. Synthesizing two theoretical models, User Engagement Theory UET and Information Behavior Theory IBT, a theoretical framework illustrating the impact of and relationships among the three IE dimensions of perception, participation, and perseverance was developed and hypotheses generated. The framework was empirically validated in a large-scale user study measuring how word choice impacts the dimensions of IE. The findings provide evidence that IE differs from other forms of engagement in that it is driven and fostered by the expression of the information itself, regardless of the information system used to view, interact with, and use the information. The findings suggest that phrasing can have a 
    
[^64]: 从对比微调的语言模型中提取语义概念嵌入

    Distilling Semantic Concept Embeddings from Contrastively Fine-Tuned Language Models. (arXiv:2305.09785v1 [cs.CL])

    [http://arxiv.org/abs/2305.09785](http://arxiv.org/abs/2305.09785)

    本论文通过对比学习策略，提高了语言模型的概念嵌入质量，并在各种基准测试中实现了最先进的结果。

    

    学习捕捉概念含义的向量仍然是一个基本挑战。令人惊讶的是，至今预训练的语言模型仅在对这种概念嵌入的质量方面产生了有限的提高。目前的使用语言模型的策略通常通过在某种语料库中平均表示一个概念在其提及中的语境化表示来表示一个概念。这在至少两个方面可能是次优的。首先，语境化的单词向量具有异常的几何性，这影响下游任务。其次，概念嵌入应该捕捉概念的语义属性，而语境化的单词向量也受到其他因素的影响。为了解决这些问题，我们提出了两种基于对比学习策略，基于这样的观点，每当两个句子显示相似的属性时，相应的语境化向量也应该相似。一种策略是完全无监督的，估计在一个句子中表达的属性。

    Learning vectors that capture the meaning of concepts remains a fundamental challenge. Somewhat surprisingly, perhaps, pre-trained language models have thus far only enabled modest improvements to the quality of such concept embeddings. Current strategies for using language models typically represent a concept by averaging the contextualised representations of its mentions in some corpus. This is potentially sub-optimal for at least two reasons. First, contextualised word vectors have an unusual geometry, which hampers downstream tasks. Second, concept embeddings should capture the semantic properties of concepts, whereas contextualised word vectors are also affected by other factors. To address these issues, we propose two contrastive learning strategies, based on the view that whenever two sentences reveal similar properties, the corresponding contextualised vectors should also be similar. One strategy is fully unsupervised, estimating the properties which are expressed in a sentence
    
[^65]: 带有注意力模型的视觉问答算法分析

    Analysis of Visual Question Answering Algorithms with attention model. (arXiv:2305.09782v1 [cs.CV])

    [http://arxiv.org/abs/2305.09782](http://arxiv.org/abs/2305.09782)

    本文批评性地检查和审查了使用共同注意力方法的VQA算法的方法，重点关注文本语义生成、对象识别和答案分类技术。

    

    视觉问答（VQA）使用图像处理算法处理图像，使用自然语言处理方法理解并回答问题。 VQA 对视觉受损者有帮助，可用于安全监控系统和从网络中学习的在线聊天机器人。 它使用自然语言处理方法学习问题的语义并提取文本特征。 计算机视觉技术用于以一种能够识别所问问题涉及的物体的方式生成图像表示。 注意力模型试图模仿人类根据语境关注图像不同区域的行为。 本文批评性地检查和审查了使用共同注意力方法的 VQA 算法的方法，例如生成文本语义，识别对象和答案分类技术。

    Visual question answering (VQA) usesimage processing algorithms to process the image and natural language processing methods to understand and answer the question. VQA is helpful to a visually impaired person, can be used for the security surveillance system and online chatbots that learn from the web. It uses NLP methods to learn the semantic of the question and to derive the textual features. Computer vision techniques are used for generating image representation in such a way that they can identify the objects about which question is asked. The Attention model tries to mimic the human behavior of giving attention to a different region of an image according to our understanding of its context. This paper critically examines and reviews methods of VQA algorithm such as generation of semantics of text, identification of objects and answer classification techniques that use the co-attention approach.
    
[^66]: SpecInfer：利用推测推断和令牌树验证加速生成式大语言模型的服务

    SpecInfer: Accelerating Generative LLM Serving with Speculative Inference and Token Tree Verification. (arXiv:2305.09781v1 [cs.CL])

    [http://arxiv.org/abs/2305.09781](http://arxiv.org/abs/2305.09781)

    SpecInfer是一种LLM服务系统，通过利用推测推断和令牌树验证来加速生成式大语言模型的推断过程，显著减少了为它们提供服务所需的端到端延迟和计算要求，同时确保模型质量。

    

    由于生成式大语言模型（LLMs）需要高计算和内存需求，因此快速和廉价地为它们提供服务是具有挑战性的。本文介绍SpecInfer，一个LLM服务系统，它利用推测推断和令牌树验证加速生成式LLM推断。SpecInfer背后的关键是将各种小型语言模型进行集体提升调整，共同预测LLM的输出； 预测结果组织成一个令牌树，其中每个节点都表示候选令牌序列。通过一种新颖的基于树的并行解码机制，以LMM作为令牌树验证器来验证令牌树所代表的所有候选令牌序列的正确性。SpecInfer使用LLM作为令牌树验证器，而不是增量解码器，从而显著减少了为生成式LLM提供服务所需的端到端延迟和计算要求，同时可确保模型质量。

    The high computational and memory requirements of generative large language models (LLMs) make it challenging to serve them quickly and cheaply. This paper introduces SpecInfer, an LLM serving system that accelerates generative LLM inference with speculative inference and token tree verification. A key insight behind SpecInfer is to combine various collectively boost-tuned small language models to jointly predict the LLM's outputs; the predictions are organized as a token tree, whose nodes each represent a candidate token sequence. The correctness of all candidate token sequences represented by a token tree is verified by the LLM in parallel using a novel tree-based parallel decoding mechanism. SpecInfer uses an LLM as a token tree verifier instead of an incremental decoder, which significantly reduces the end-to-end latency and computational requirement for serving generative LLMs while provably preserving model quality.
    
[^67]: ConvXAI：通过对话提供异构的AI解释，支持人机科技写作

    ConvXAI: Delivering Heterogeneous AI Explanations via Conversations to Support Human-AI Scientific Writing. (arXiv:2305.09770v1 [cs.HC])

    [http://arxiv.org/abs/2305.09770](http://arxiv.org/abs/2305.09770)

    ConvXAI是一个基于对话的XAI系统，它集成了多种XAI类型，并将实际用户需求嵌入设计中，以提高实用性。

    

    尽管已经提出了各种各样的人工智能解释（XAI）方法来解释AI系统，但目前的方法是否对人类实用仍存在不一致的发现。为了改善XAI方法的实用性，一系列研究确定了现实世界中多样化和动态的用户需求与现有XAI方法之间的差距。虽然之前的研究设想将多种XAI方法集成到通用XAI界面（例如，基于对话或GUI的XAI系统）中以减轻这些差距，但缺少针对这些系统如何设计以满足实际用户需求的研究。在本研究中，我们提出了ConvXAI，这是一个基于对话的XAI系统，它结合了多种XAI类型，并赋予用户通过通用的XAI对话界面提出各种XAI问题的能力。特别地，我们创新地将实际用户需求（即，基于格式研究的四个原则）嵌入ConvXAI设计中，以提高实用性。

    While various AI explanation (XAI) methods have been proposed to interpret AI systems, whether the state-of-the-art XAI methods are practically useful for humans remains inconsistent findings. To improve the usefulness of XAI methods, a line of studies identifies the gaps between the diverse and dynamic real-world user needs with the status quo of XAI methods. Although prior studies envision mitigating these gaps by integrating multiple XAI methods into the universal XAI interfaces (e.g., conversational or GUI-based XAI systems), there is a lack of work investigating how these systems should be designed to meet practical user needs. In this study, we present ConvXAI, a conversational XAI system that incorporates multiple XAI types, and empowers users to request a variety of XAI questions via a universal XAI dialogue interface. Particularly, we innovatively embed practical user needs (i.e., four principles grounding on the formative study) into ConvXAI design to improve practical useful
    
[^68]: 设备上无应用语言建模指导的自动语音识别

    Application-Agnostic Language Modeling for On-Device ASR. (arXiv:2305.09764v1 [cs.CL])

    [http://arxiv.org/abs/2305.09764](http://arxiv.org/abs/2305.09764)

    本文提出了两种新的前向体系结构用于无应用指导的语言建模，以帮助设备上的自动语音识别系统克服速度、磁盘和内存等限制。

    

    与基于服务器的系统相比，设备上的自动语音识别系统面临着许多挑战。它们必须在保持相同准确性的同时满足更严格的速度、磁盘大小和内存的限制。通常，它们必须为多个具有不同分配的应用程序提供服务，例如与虚拟助手通信和语音转文本等。为了为多个应用程序提供服务，最简单的解决方案是构建特定于应用程序的(语言)模型，但这会增加内存。因此，我们探索了不同的数据和架构驱动的语言建模方法，以构建一个单一的无应用指导的模型。我们提出了两种新的前向体系结构，可以找到在不同设备限制之间的最佳折衷。与特定于应用程序的解决方案相比，我们的一种新方法将磁盘大小减半，同时保持了原模型的速度和准确性。

    On-device automatic speech recognition systems face several challenges compared to server-based systems. They have to meet stricter constraints in terms of speed, disk size and memory while maintaining the same accuracy. Often they have to serve several applications with different distributions at once, such as communicating with a virtual assistant and speech-to-text. The simplest solution to serve multiple applications is to build application-specific (language) models, but this leads to an increase in memory. Therefore, we explore different data- and architecture-driven language modeling approaches to build a single application-agnostic model. We propose two novel feed-forward architectures that find an optimal trade off between different on-device constraints. In comparison to the application-specific solution, one of our novel approaches reduces the disk size by half, while maintaining speed and accuracy of the original model.
    
[^69]: 患者的临床笔记具有自己的层次结构：多层超图神经网络用于患者级别表示学习

    Clinical Note Owns its Hierarchy: Multi-Level Hypergraph Neural Networks for Patient-Level Representation Learning. (arXiv:2305.09756v1 [cs.CL])

    [http://arxiv.org/abs/2305.09756](http://arxiv.org/abs/2305.09756)

    本文提出了一种基于分类级别的多层超图神经网络（TM-HGNN），通过笔记和分类级别的超边组装有用的中性词和稀有关键词以保留临床语义信息，并在MIMIC-III数据集上取得了优于最先进方法的实验结果。

    

    利用电子健康记录（EHR）中的知识来预测患者的病情对于提供适当的护理至关重要。患者的临床笔记包含医疗保健专业人员的有价值的信息，但由于其难以理解的内容和复杂的层次结构而被低估。最近，基于超图的方法已被提出用于文档分类。直接采用现有的超图方法处理患者的临床笔记无法充分利用患者的层次信息，这可能会通过(1) 频繁使用的中性词和(2) 分布不平衡的层次结构降低临床语义信息。因此，我们提出了一个基于分类级别的多层超图神经网络（TM-HGNN），其中多层超图通过笔记和分类级别的超边组装有用的中性词和稀有关键词，以保留临床语义信息。构建的患者超图输入分层消息传递层，其中消息传递在同一级别内受到限制，并由超边连接指导。在MIMIC-III数据集上的实验结果表明，我们提出的TM-HGNN在患者级别分类任务中优于现有最先进的方法。

    Leveraging knowledge from electronic health records (EHRs) to predict a patient's condition is essential to the effective delivery of appropriate care. Clinical notes of patient EHRs contain valuable information from healthcare professionals, but have been underused due to their difficult contents and complex hierarchies. Recently, hypergraph-based methods have been proposed for document classifications. Directly adopting existing hypergraph methods on clinical notes cannot sufficiently utilize the hierarchy information of the patient, which can degrade clinical semantic information by (1) frequent neutral words and (2) hierarchies with imbalanced distribution. Thus, we propose a taxonomy-aware multi-level hypergraph neural network (TM-HGNN), where multi-level hypergraphs assemble useful neutral words with rare keywords via note and taxonomy level hyperedges to retain the clinical semantic information. The constructed patient hypergraphs are fed into hierarchical message passing layers
    
[^70]: 在语境中学习：“学习”语境中的任务识别和任务学习的区分。

    What In-Context Learning "Learns" In-Context: Disentangling Task Recognition and Task Learning. (arXiv:2305.09731v1 [cs.CL])

    [http://arxiv.org/abs/2305.09731](http://arxiv.org/abs/2305.09731)

    本研究通过任务识别和任务学习两种方式表征了ICL利用演示的方式，发现LLMs利用不同机制进行任务的解决，TR主要利用先验知识，而TL则具备学习新的输入-标签映射的能力。

    

    大型语言模型通过利用语境中的学习来解决只有少数演示的任务，但其机制尚未得到很好的理解。一些研究表明LLMs仅回忆来自预训练的已学概念，而其他研究则暗示ICL执行演示的隐含学习。本文通过任务识别(TR)和任务学习(TL)两种方式表征了ICL利用演示的方式。我们使用各种分类数据集和三个LLM系列（GPT-3、LLaMA和OPT）进行控制实验，在ICL中区分TR和TL的角色。我们发现：（1）模型只使用TR就能取得非平凡的性能，TR不会随着更大的模型或更多的演示而进一步改善；（2）LLMs能够通过TL学习新的输入-标签映射，而TR则主要利用预先训练的先验知识。

    Large language models (LLMs) exploit in-context learning (ICL) to solve tasks with only a few demonstrations, but its mechanisms are not yet well-understood. Some works suggest that LLMs only recall already learned concepts from pre-training, while others hint that ICL performs implicit learning over demonstrations. We characterize two ways through which ICL leverages demonstrations. Task recognition (TR) captures the extent to which LLMs can recognize a task through demonstrations -- even without ground-truth labels -and apply their pre-trained priors, whereas task learning (TL) is the ability to capture new input-label mappings unseen in pre-training. Using a wide range of classification datasets and three LLM families (GPT-3, LLaMA and OPT), we design controlled experiments to disentangle the roles of TR and TL in ICL. We show that (1) models can achieve non-trivial performance with only TR, and TR does not further improve with larger models or more demonstrations; (2) LLMs acquir
    
[^71]: 生成式表格预训练增强了表格预测模型

    Generative Table Pre-training Empowers Models for Tabular Prediction. (arXiv:2305.09696v1 [cs.LG])

    [http://arxiv.org/abs/2305.09696](http://arxiv.org/abs/2305.09696)

    本文提出了TapTap，一种通过表格预训练生成高质量合成表格来提高表格预测性能的方法。在12个数据集实验中，TapTap在不同场景下优于16个基线，并可以与多个骨干模型结合使用。

    

    近年来，表格预训练已经成为研究的热点，但如何利用表格预训练来提高表格预测的性能仍然是一个开放性挑战。本文提出了TapTap，这是第一个利用表格预训练来增强表格预测模型的尝试。在对大量实际世界的表格数据进行预训练后，TapTap能够生成高质量的合成表格，以支持各种表格数据应用，包括隐私保护、低资源环境、缺失值插补和失衡分类。在12个数据集上的广泛实验表明，TapTap在不同场景下优于16个基线。同时，它可以轻松地与各种骨干模型结合使用，包括LightGBM、多层感知机（MLP）和Transformer。此外，在表格预训练的帮助下，使用TapTap生成的合成数据进行训练的模型甚至可以与使用原始真实数据的模型竞争，实现了相当甚至更好的性能。

    Recently, the topic of table pre-training has attracted considerable research interest. However, how to employ table pre-training to boost the performance of tabular prediction remains an open challenge. In this paper, we propose TapTap, the first attempt that leverages table pre-training to empower models for tabular prediction. After pre-training on a large corpus of real-world tabular data, TapTap can generate high-quality synthetic tables to support various applications on tabular data, including privacy protection, low resource regime, missing value imputation, and imbalanced classification. Extensive experiments on 12 datasets demonstrate that TapTap outperforms a total of 16 baselines in different scenarios. Meanwhile, it can be easily combined with various backbone models, including LightGBM, Multilayer Perceptron (MLP) and Transformer. Moreover, with the aid of table pre-training, models trained using synthetic data generated by TapTap can even compete with models using the or
    
[^72]: OOD-Speech: 用于 Bengali 语音识别的大规模越域基准数据集

    OOD-Speech: A Large Bengali Speech Recognition Dataset for Out-of-Distribution Benchmarking. (arXiv:2305.09688v1 [eess.AS])

    [http://arxiv.org/abs/2305.09688](http://arxiv.org/abs/2305.09688)

    OOD-Speech 是用于 Bengali 语音识别的越域基准数据集，由众包收集了母语为 Bengali 的 22,645 名说话者录制的 1177.94 小时语音数据，并经过手动注释。数据集包含 17 种不同的资源，如 Bengali 电视剧、有声读物、脱口秀、在线教学以及伊斯兰讲道等，可作为 Bengali 语音识别的分布变化方面的基准测试数据集。

    

    我们提出了 OOD-Speech，这是 Bengali 的第一个用于自动语音识别的越域基准数据集。作为全球使用最广泛的语言之一，Bengali 展示了大量的方言和韵律特征，这要求 ASR 框架对分布变化具有鲁棒性。例如，Bengali 中的伊斯兰宗教讲道是用明显不同的语调进行的，这也成为了分布变化的例子。我们的训练数据集是通过在线众包活动收集并筛选而来，共收集了来自南亚的 22,645 名母语为 Bengali 的说话者所录制的 1177.94 小时。我们的测试数据集则包括来自 17 个不同资源（如 Bengali 电视剧、有声读物、脱口秀、在线教学以及伊斯兰讲道等）的 23.03 小时语音数据，这些数据也都经过了手动注释。OOD-Speech 既是当前公开的最大的语音数据集，也是 Bengali 语音识别的第一个越域基准数据集。

    We present OOD-Speech, the first out-of-distribution (OOD) benchmarking dataset for Bengali automatic speech recognition (ASR). Being one of the most spoken languages globally, Bengali portrays large diversity in dialects and prosodic features, which demands ASR frameworks to be robust towards distribution shifts. For example, islamic religious sermons in Bengali are delivered with a tonality that is significantly different from regular speech. Our training dataset is collected via massively online crowdsourcing campaigns which resulted in 1177.94 hours collected and curated from $22,645$ native Bengali speakers from South Asia. Our test dataset comprises 23.03 hours of speech collected and manually annotated from 17 different sources, e.g., Bengali TV drama, Audiobook, Talk show, Online class, and Islamic sermons to name a few. OOD-Speech is jointly the largest publicly available speech dataset, as well as the first out-of-distribution ASR benchmarking dataset for Bengali.
    
[^73]: 声明提示下的可满足性辅助语言模型

    Satisfiability-Aided Language Models Using Declarative Prompting. (arXiv:2305.09656v1 [cs.CL])

    [http://arxiv.org/abs/2305.09656](http://arxiv.org/abs/2305.09656)

    本文提出了一种利用自动定理证明器和声明性任务规范的可满足性辅助语言建模方法，可以提高大型语言模型的推理能力。

    

    本文提出了一种新的可满足性辅助语言建模方法，用于提高大型语言模型的推理能力。我们使用一个大型语言模型生成一个声明性任务规范，并利用一个现成的自动定理证明器得出最终答案。该方法具有两个关键优点：第一，声明性规范比推理步骤更接近问题描述，因此大型语言模型可以更准确地解析它；第二，通过将实际推理任务委托给自动定理证明器，我们的方法可以保证正确性。

    Prior work has combined chain-of-thought prompting in large language models (LLMs) with programmatic representations to perform effective and transparent reasoning. While such an approach works very well for tasks that only require forward reasoning (e.g., straightforward arithmetic), it is less effective for constraint solving tasks that require more sophisticated planning and search. In this paper, we propose a new satisfiability-aided language modeling approach for improving the reasoning capabilities of LLMs. We use an LLM to generate a declarative task specification rather than an imperative program and leverage an off-the-shelf automated theorem prover to derive the final answer. This approach has two key advantages. The declarative specification is closer to the problem description than the reasoning steps are, so the LLM can parse it more accurately. Furthermore, by offloading the actual reasoning task to an automated theorem prover, our approach can guarantee the correctness o
    
[^74]: PII的生命--一种PII混淆变换器

    Life of PII -- A PII Obfuscation Transformer. (arXiv:2305.09550v1 [cs.CL])

    [http://arxiv.org/abs/2305.09550](http://arxiv.org/abs/2305.09550)

    “Life of PII”是一种新颖的混淆变换器框架，用于将PII转化为人造PII同时尽可能地保留原始信息、意图和上下文，使我们能够有选择地混淆文档中的敏感信息，同时保留文档的统计和语义特性。

    

    在当今大型语言模型和数据驱动服务的世界中，保护敏感信息至关重要。一种常见的方法是使用数据扰动技术来减少(敏感)个人身份识别信息(PII)数据的过度实用性，同时保持其统计和语义特性。数据扰动方法经常导致显着的信息损失，使它们难以使用。在本文中，我们提出了“PII的生命”--一种新颖的混淆变换器框架，用于将PII转化为人造PII同时尽可能地保留原始信息、意图和上下文。我们的方法包括一个API来与给定的文档进行接口，一个基于配置的混淆器和一个基于Transformer架构的模型，在自然语言处理任务和LLMs中表现出高的上下文保存性能。我们的基于Transformer的方法学习了原始PII和其转换后的人造PII对应的映射，使我们能够有选择地混淆文档中的敏感信息，同时保留文档的统计和语义特性。

    Protecting sensitive information is crucial in today's world of Large Language Models (LLMs) and data-driven services. One common method used to preserve privacy is by using data perturbation techniques to reduce overreaching utility of (sensitive) Personal Identifiable Information (PII) data while maintaining its statistical and semantic properties. Data perturbation methods often result in significant information loss, making them impractical for use. In this paper, we propose 'Life of PII', a novel Obfuscation Transformer framework for transforming PII into faux-PII while preserving the original information, intent, and context as much as possible. Our approach includes an API to interface with the given document, a configuration-based obfuscator, and a model based on the Transformer architecture, which has shown high context preservation and performance in natural language processing tasks and LLMs.  Our Transformer-based approach learns mapping between the original PII and its tra
    
[^75]: GIFT: 基于图感知微调的多方对话理解

    GIFT: Graph-Induced Fine-Tuning for Multi-Party Conversation Understanding. (arXiv:2305.09360v1 [cs.CL])

    [http://arxiv.org/abs/2305.09360](http://arxiv.org/abs/2305.09360)

    GIFT是一个适用于多方对话理解的方法，通过设计四种类型的边缘将图感知信息集成到注意力机制中，改进了原始的顺序文本处理的PLM。

    

    最近，关于谁与谁在多方对话中说了什么的问题已经引起了很多研究的关注。然而，现有的多方对话理解方法通常将说话者和话语嵌入到顺序信息流中，或仅利用多方对话中固有图结构的表层信息。为此，我们提出了一种名为图感知微调（GIFT）的即插即用轻量级方法，可以适应各种基于Transformer预训练语言模型（PLMs）的通用多方对话理解。具体地，在普通Transformer中，话语之间的全等连接会忽略一个话语对另一个话语的稀疏但有区别的依赖关系。为了区分话语之间的不同关系，设计了四种类型的边缘以将图感知信号集成到注意机制中，以改进最初设计用于处理顺序文本的PLMs。我们通过将GIFT实现到三个PLMs并对其进行测试来评估GIFT。

    Addressing the issues of who saying what to whom in multi-party conversations (MPCs) has recently attracted a lot of research attention. However, existing methods on MPC understanding typically embed interlocutors and utterances into sequential information flows, or utilize only the superficial of inherent graph structures in MPCs. To this end, we present a plug-and-play and lightweight method named graph-induced fine-tuning (GIFT) which can adapt various Transformer-based pre-trained language models (PLMs) for universal MPC understanding. In detail, the full and equivalent connections among utterances in regular Transformer ignore the sparse but distinctive dependency of an utterance on another in MPCs. To distinguish different relationships between utterances, four types of edges are designed to integrate graph-induced signals into attention mechanisms to refine PLMs originally designed for processing sequential texts. We evaluate GIFT by implementing it into three PLMs, and test the
    
[^76]: BERTTM: 利用来自预训练语言模型的上下文化词向量进行神经主题建模

    BERTTM: Leveraging Contextualized Word Embeddings from Pre-trained Language Models for Neural Topic Modeling. (arXiv:2305.09329v1 [cs.CL])

    [http://arxiv.org/abs/2305.09329](http://arxiv.org/abs/2305.09329)

    本文提出了一种新颖的神经主题模型，利用来自预训练语言模型BERT的上下文化词嵌入，可以在不使用任何BoW信息的情况下推断出文档的主题分布，并直接从上下文化词嵌入中推断出文档中每个单词的主题分布。实验结果表明，该模型优于仅依赖BoW表示和其他神经主题模型的现有最先进方法。

    

    随着近年来神经主题模型的发展，主题建模在自然语言理解中扮演着日益重要的角色。然而，大多数现有的主题模型仍然依赖于词袋（BoW）信息，无论是作为训练输入还是训练目标。这限制了它们捕捉文档中的单词顺序信息的能力，并导致它们在处理新文档中的未观察到的单词时遇到困难。预训练语言模型中的上下文化词向量在词义消歧的能力上表现优越，并证明了它们在处理OOV单词时是有效的。在这项工作中，我们开发了一种新颖的神经主题模型，结合了预训练语言模型BERT的上下文化词嵌入。该模型可以在不使用任何BoW信息的情况下推断出文档的主题分布。此外，该模型可以直接从上下文化词嵌入中推断出文档中每个单词的主题分布。基准数据集的实验表明，我们的模型优于仅依赖BoW表示和其他神经主题模型的现有最先进方法。

    With the development of neural topic models in recent years, topic modelling is playing an increasingly important role in natural language understanding. However, most existing topic models still rely on bag-of-words (BoW) information, either as training input or training target. This limits their ability to capture word order information in documents and causes them to suffer from the out-of-vocabulary (OOV) issue, i.e. they cannot handle unobserved words in new documents. Contextualized word embeddings from pre-trained language models show superiority in the ability of word sense disambiguation and prove to be effective in dealing with OOV words. In this work, we developed a novel neural topic model combining contextualized word embeddings from the pre-trained language model BERT. The model can infer the topic distribution of a document without using any BoW information. In addition, the model can infer the topic distribution of each word in a document directly from the contextualize
    
[^77]: 超越相关分析的NLG评估指标：一种经验度量偏好检查表

    NLG Evaluation Metrics Beyond Correlation Analysis: An Empirical Metric Preference Checklist. (arXiv:2305.08566v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.08566](http://arxiv.org/abs/2305.08566)

    本研究提出了一种度量偏好检查表，以超越相关分析评估NLG自动指标，并分析了两种类型的指标及其在三个任务中的效果。

    

    本研究分析了NLG自动评估指标，基于是否将人类评估方面用作上下文或目标来计算指标，分为（i）任务不可知和（ii）与人类对齐的指标。我们提出了度量偏好检查表作为评估自动指标在三个NLG任务中的鉴别力的框架：文本摘要，对话响应生成和受控生成。

    In this study, we analyze NLG automatic metrics based on whether human evaluation aspect is used as context or objective to compute the metrics: (i) Task-agnostic and (ii) Human-aligned. Task-agnostic metrics, such as Perplexity, BLEU, BERTScore, are cost-effective and highly adaptable to diverse NLG tasks, yet they have a weak correlation with human. Human-aligned metrics (CTC, CtrlEval, UniEval) improves correlation level by incorporating desirable human-like qualities as training objective. However, their effectiveness at discerning system-level performance and quality of system outputs remains unclear.  We present metric preference checklist as a framework to assess the discriminative power of automatic metrics in three NLG tasks: Text Summarization, Dialogue Response Generation, and Controlled Generation. We show that multi-aspect human-aligned metric (UniEval) is not necessarily dominant over single-aspect human-aligned metrics (CTC, CtrlEval) and task-agnostic metrics (BLEU, BER
    
[^78]: C-Eval: 用于基础模型的多级多学科中文评估套件

    C-Eval: A Multi-Level Multi-Discipline Chinese Evaluation Suite for Foundation Models. (arXiv:2305.08322v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.08322](http://arxiv.org/abs/2305.08322)

    C-Eval是第一个专为中文基础模型评估而设计的全面套件，涵盖52个不同学科的多级别选择题和挑战性科目。评估结果表明，只有GPT-4能够达到超过60％的平均准确率，还有改进空间。

    

    随着大型语言模型（LLM）的快速发展，迫切需要新的自然语言处理基准来评估这些模型的高级知识和推理能力。我们介绍了C-Eval，这是第一个专为中文语境下基础模型评估而设计的全面评估套件。C-Eval包含四个难度级别的选择题：初中、高中、大学和专业水平。这些题目涵盖52个不同的学科，包括人文、科学和工程学科。C-Eval还配备了C-Eval Hard，这是C-Eval中一些极具挑战性的科目，需要高级推理能力才能解决。我们对包括英文和中文模型在内的最先进的LLM在C-Eval上进行了全面评估。结果表明，只有GPT-4可以实现超过60％的平均准确率，这表明当前的LLM仍有很大的改进空间。我们期望C-Eval将有助于分析重要的优势和短板。

    New NLP benchmarks are urgently needed to align with the rapid development of large language models (LLMs). We present C-Eval, the first comprehensive Chinese evaluation suite designed to assess advanced knowledge and reasoning abilities of foundation models in a Chinese context. C-Eval comprises multiple-choice questions across four difficulty levels: middle school, high school, college, and professional. The questions span 52 diverse disciplines, ranging from humanities to science and engineering. C-Eval is accompanied by C-Eval Hard, a subset of very challenging subjects in C-Eval that requires advanced reasoning abilities to solve. We conduct a comprehensive evaluation of the most advanced LLMs on C-Eval, including both English- and Chinese-oriented models. Results indicate that only GPT-4 could achieve an average accuracy of over 60%, suggesting that there is still significant room for improvement for current LLMs. We anticipate C-Eval will help analyze important strengths and sho
    
[^79]: 在对话推荐系统中利用大型语言模型

    Leveraging Large Language Models in Conversational Recommender Systems. (arXiv:2305.07961v1 [cs.IR])

    [http://arxiv.org/abs/2305.07961](http://arxiv.org/abs/2305.07961)

    本文提出了一种使用大型语言模型构建端到端大规模对话推荐系统的路线图，解决在该系统中有效利用大型语言模型所面临的技术挑战。

    

    对话推荐系统通过启用实时的多轮对话使用户更加透明和掌控。最近，大型语言模型展现了与人类对话自然的能力，并将世界知识和常识推理融入到语言理解中，进一步释放了这一范式的潜力。然而，在对话推荐系统中有效利用大型语言模型引入了新的技术挑战，包括适当地理解和控制复杂的对话和从外部信息源检索。由于大而不断增长的项目语料库和缺乏对话数据进行训练，这些问题加剧了。在本文中，我们提供了使用大型语言模型构建端到端大规模对话推荐系统的路线图。特别地，我们提出了用户偏好理解、灵活的对话管理和可解释的推荐作为整个系统的一部分的新实现方式。

    A Conversational Recommender System (CRS) offers increased transparency and control to users by enabling them to engage with the system through a real-time multi-turn dialogue. Recently, Large Language Models (LLMs) have exhibited an unprecedented ability to converse naturally and incorporate world knowledge and common-sense reasoning into language understanding, unlocking the potential of this paradigm. However, effectively leveraging LLMs within a CRS introduces new technical challenges, including properly understanding and controlling a complex conversation and retrieving from external sources of information. These issues are exacerbated by a large, evolving item corpus and a lack of conversational data for training. In this paper, we provide a roadmap for building an end-to-end large-scale CRS using LLMs. In particular, we propose new implementations for user preference understanding, flexible dialogue management and explainable recommendations as part of an integrated architecture
    
[^80]: 大型语言模型中的字典链提示在翻译中的应用

    Chain-of-Dictionary Prompting Elicits Translation in Large Language Models. (arXiv:2305.06575v1 [cs.CL])

    [http://arxiv.org/abs/2305.06575](http://arxiv.org/abs/2305.06575)

    研究通过在大型语言模型中添加字典链提示的方法来改进低资源语言的翻译能力，实验结果表明能显著提高翻译质量。

    

    大型语言模型(LLMs)在多语言神经机器翻译(MNMT)中表现出惊人的性能，即使没有平行数据也能训练。然而，尽管训练数据量巨大，它们仍然难以翻译稀有词汇，特别是对于低资源语言。更糟糕的是，通常情况下，在低资源语言上，很难检索到相关示范来进行上下文学习，这限制了LLMs在翻译方面的实际应用——我们该如何缓解这个问题？为此，我们提出了一种新的方法，CoD，通过使用多语言字典链为一部分输入单词增加LLMs的先前知识，从而促进LLMs的翻译能力。广泛的实验表明，在FLORES-200全开发测试集上，通过将CoD和ChatGPT相结合，可以获得高达13倍的MNMT ChrF++分数的收益（英语到塞尔维亚语，西里尔字母书写，ChrF ++分数从3.08增加到42.63）。我们进一步展示了该方法在其他数据集上的重要作用。

    Large language models (LLMs) have shown surprisingly good performance in multilingual neural machine translation (MNMT) even when trained without parallel data. Yet, despite the fact that the amount of training data is gigantic, they still struggle with translating rare words, particularly for low-resource languages. Even worse, it is usually unrealistic to retrieve relevant demonstrations for in-context learning with low-resource languages on LLMs, which restricts the practical use of LLMs for translation -- how should we mitigate this problem? To this end, we present a novel method, CoD, which augments LLMs with prior knowledge with the chains of multilingual dictionaries for a subset of input words to elicit translation abilities for LLMs. Extensive experiments indicate that augmenting ChatGPT with CoD elicits large gains by up to 13x ChrF++ points for MNMT (3.08 to 42.63 for English to Serbian written in Cyrillic script) on FLORES-200 full devtest set. We further demonstrate the im
    
[^81]: K-UniMorph：韩语通用词形学及其特征模式

    K-UniMorph: Korean Universal Morphology and its Feature Schema. (arXiv:2305.06335v1 [cs.CL])

    [http://arxiv.org/abs/2305.06335](http://arxiv.org/abs/2305.06335)

    本文介绍了一种韩语通用词形学数据集，保留韩语特色并采用Sylak-Glassman等人的词形特征模式，为韩语形态学范式领域做出了贡献。

    

    本文介绍了一种韩语通用词形学数据集，之前，韩语在数百种多样的世界语言中的形态学范式领域中一直处于少数。因此，我们提出了这种保留韩语特色的通用词形学范式。我们的K-UniMorph数据集中，我们详细概述了每个语法标准的动词结尾，并阐明如何提取变形形式以及如何生成词形模式。此数据集采用Sylak-Glassman等人（2015）和Sylak-Glassman（2016）的词形特征模式，而我们从Sejong形态分析语料库中提取变形形式，这是韩语最大的注释语料库之一。在数据创建过程中，我们的方法还包括调查从Sejong语料库中的转换的正确性。此外，我们使用三种不同的模型进行了变形任务。

    We present in this work a new Universal Morphology dataset for Korean. Previously, the Korean language has been underrepresented in the field of morphological paradigms amongst hundreds of diverse world languages. Hence, we propose this Universal Morphological paradigms for the Korean language that preserve its distinct characteristics. For our K-UniMorph dataset, we outline each grammatical criterion in detail for the verbal endings, clarify how to extract inflected forms, and demonstrate how we generate the morphological schemata. This dataset adopts morphological feature schema from Sylak-Glassman et al. (2015) and Sylak-Glassman (2016) for the Korean language as we extract inflected verb forms from the Sejong morphologically analyzed corpus that is one of the largest annotated corpora for Korean. During the data creation, our methodology also includes investigating the correctness of the conversion from the Sejong corpus. Furthermore, we carry out the inflection task using three di
    
[^82]: SemEval-2023任务2中的DAMO-NLP: 一种多语言命名实体识别的统一检索增强系统

    DAMO-NLP at SemEval-2023 Task 2: A Unified Retrieval-augmented System for Multilingual Named Entity Recognition. (arXiv:2305.03688v1 [cs.CL])

    [http://arxiv.org/abs/2305.03688](http://arxiv.org/abs/2305.03688)

    DAMO-NLP团队的U-RaNER是一种统一的多语言命名实体识别系统，它通过加入带有实体为中心的Wikidata知识库并采用infusion方法来增强检索上下文，解决了其他系统存在的知识不足、上下文长度有限和单一检索策略等问题。

    

    MultiCoNER 2共享任务旨在解决多语言命名实体识别的细粒度和嘈杂情况，并继承了MultiCoNER 1任务的语义歧义和低上下文环境。针对这些问题，MultiCoNER 1中的前几个顶尖系统要么纳入知识库或专有名词表，但它们仍然存在知识不足、上下文长度有限以及单一检索策略等问题。在本文中，我们的DAMO-NLP团队提出了一种用于多语言的细粒度命名实体识别的统一检索增强系统（U-RaNER）。我们对上述几个顶尖系统进行了错误分析，发现它们的性能瓶颈在于知识不足，而且有限的上下文长度使得检索知识对模型不可见。为了增强检索上下文，我们加入了以实体为中心的Wikidata知识库，并采用infusion方法来拓宽上下文引用。

    The MultiCoNER \RNum{2} shared task aims to tackle multilingual named entity recognition (NER) in fine-grained and noisy scenarios, and it inherits the semantic ambiguity and low-context setting of the MultiCoNER \RNum{1} task. To cope with these problems, the previous top systems in the MultiCoNER \RNum{1} either incorporate the knowledge bases or gazetteers. However, they still suffer from insufficient knowledge, limited context length, single retrieval strategy. In this paper, our team \textbf{DAMO-NLP} proposes a unified retrieval-augmented system (U-RaNER) for fine-grained multilingual NER. We perform error analysis on the previous top systems and reveal that their performance bottleneck lies in insufficient knowledge. Also, we discover that the limited context length causes the retrieval knowledge to be invisible to the model. To enhance the retrieval context, we incorporate the entity-centric Wikidata knowledge base, while utilizing the infusion approach to broaden the contextua
    
[^83]: 语言选择的政治：俄乌战争如何影响乌克兰人在 Twitter 上的语言使用。

    The Politics of Language Choice: How the Russian-Ukrainian War Influences Ukrainians' Language Use on Twitter. (arXiv:2305.02770v1 [cs.CY])

    [http://arxiv.org/abs/2305.02770](http://arxiv.org/abs/2305.02770)

    本文研究了俄乌战争期间乌克兰人在 Twitter 上的语言使用，发现在战争爆发前已经出现从俄语向乌克兰语转变的趋势，而战争爆发后这种趋势加速了，并且许多使用俄语的用户在战争期间转变成使用乌克兰语。

    

    语言使用天生是政治的，并经常用作文化身份的载体，同时也是国家建设的基础。本文研究了俄乌战争期间（2020年1月至2022年10月），基于超过62,000位用户发布的400万条地理标记推文中，乌克兰公民的语言选择和推文活动。使用统计模型，区分了Twitter上用户的流入流出所引起的样本效应和用户行为变化所引起的行为效应。我们观察到，在战争爆发之前已经有一个稳定的从俄语向乌克兰语的转变，而这一过程在战争爆发后迅速加速。我们将这些变化主要归因于用户行为的改变。值得注意的是，许多使用俄语的用户在战争期间会转变成使用乌克兰语。

    The use of language is innately political and often a vehicle of cultural identity as well as the basis for nation building. Here, we examine language choice and tweeting activity of Ukrainian citizens based on more than 4 million geo-tagged tweets from over 62,000 users before and during the Russian-Ukrainian War, from January 2020 to October 2022. Using statistical models, we disentangle sample effects, arising from the in- and outflux of users on Twitter, from behavioural effects, arising from behavioural changes of the users. We observe a steady shift from the Russian language towards the Ukrainian language already before the war, which drastically speeds up with its outbreak. We attribute these shifts in large part to users' behavioural changes. Notably, we find that many Russian-tweeting users perform a hard-switch to Ukrainian as a result of the war.
    
[^84]: 运用自我记忆的检索增强文本生成模型

    Lift Yourself Up: Retrieval-augmented Text Generation with Self Memory. (arXiv:2305.02437v1 [cs.CL])

    [http://arxiv.org/abs/2305.02437](http://arxiv.org/abs/2305.02437)

    本文提出了一种新的检索增强文本生成模型Selfmem，通过迭代生成自我记忆池并采用记忆选择器，使检索更加自适应，提高了文本生成的质量和多样性。

    

    相较于传统文本生成模型，检索增强文本生成模型能够直接迭代人类编写的参考库，并从中检索出相应的信息，以生成更优质的文本。但当前文献存在一个关键问题：检索到的记忆来自于固定的语料库，其质量存在一定局限性，可能会限制记忆增强模型的潜力。本文提出一种名为Selfmem的框架，该框架通过迭代地采用检索增强生成器自身以生成无限制的自我记忆池，并使用记忆选择器为下一轮生成选择一个生成的记忆。相结合，这两个主要问题提出了运用自我记忆的检索增强文本生成模型。

    With direct access to human-written reference as memory, retrieval-augmented generation has achieved much progress in a wide range of text generation tasks. Since better memory would typically prompt better generation~(we define this as primal problem), previous works mainly focus on how to retrieve better memory. However, one fundamental limitation exists for current literature: the memory is retrieved from a fixed corpus and is bounded by the quality of the corpus. Due to the finite retrieval space, bounded memory would greatly limit the potential of the memory-augmented generation model. In this paper, by exploring the duality of the primal problem: better generation also prompts better memory, we propose a framework called Selfmem, which iteratively adopts a retrieval-augmented generator itself to generate an unbounded memory pool and uses a memory selector to pick one generated memory for the next generation round. By combining the primal and dual problem, a retrieval-augmented ge
    
[^85]: 不停止预训练？让基于提示的微调更加强大

    Don't Stop Pretraining? Make Prompt-based Fine-tuning Powerful Learner. (arXiv:2305.01711v1 [cs.CL])

    [http://arxiv.org/abs/2305.01711](http://arxiv.org/abs/2305.01711)

    本文研究了持续预训练对于微调性能的影响，发现传统的持续预训练不能保证一致的提高性能，甚至会对一些任务产生负面影响。针对这些问题，作者提出了基于提示的持续预训练，旨在通过无监督的预训练向LM展示任务相关文本和提示模板，从而提高基于提示的微调表现。

    

    在大量无标注数据的训练下，语言模型（LM）极大地推动了自然语言处理（NLP）领域的发展。 在本研究中，我们重新审视NLP中广为接受的LM任务相关文本的持续预训练可以提高下游任务微调性能的理论。通过在半监督和全监督设置下对八个单句任务和八个句对任务的实验，我们发现传统的持续预训练不能保证一致的提高性能，甚至可能对句对任务或使用基于提示的微调方式时会产生负面影响。为了解决这些问题，我们提出了基于提示的持续预训练（PCP），将指导调整的思想与传统的持续预训练相结合。我们的方法旨在通过在微调目标之前通过无监督的预训练目标向LM展示任务相关文本和提示模板，从而提高基于提示的FT的表现。

    Language models (LMs) trained on vast quantities of unlabelled data have greatly advanced the field of natural language processing (NLP). In this study, we re-visit the widely accepted notion in NLP that continued pre-training LMs on task-related texts improves the performance of fine-tuning (FT) in downstream tasks. Through experiments on eight single-sentence tasks and eight sentence-pair tasks in both semi-supervised and fully-supervised settings, we find that conventional continued pre-training does not consistently provide benefits and can even be detrimental for sentence-pair tasks or when prompt-based FT is used. To tackle these issues, we propose Prompt-based Continued Pre-training (PCP), which combines the idea of instruction tuning with conventional continued pre-training. Our approach aims to improve the performance of prompt-based FT by presenting both task-related texts and prompt templates to LMs through unsupervised pre-training objectives before fine-tuning for the targ
    
[^86]: 基于BERT的技术对美国最高法院案例进行分类

    Classification of US Supreme Court Cases using BERT-Based Techniques. (arXiv:2304.08649v1 [cs.CL])

    [http://arxiv.org/abs/2304.08649](http://arxiv.org/abs/2304.08649)

    本文基于BERT技术探究了对美国最高法院案例进行分类的方法，比较了使用BERT模型与其他先进模型的准确性，最终在15个广泛类别上取得了80%的准确度，在279个细粒度类别上取得了60%的准确度。

    

    基于双向编码器表示来自变压器的模型（BERT）在许多自然语言处理（NLP）任务（如命名实体识别（NER），词性（POS）标记等）上产生了最新技术（SOTA）结果。当分类长文档（例如来自美国最高法院的文档）时，使用BERT模型可能比较困难。本文中，我们尝试了几种基于BERT的分类技术，用于对美国最高法院决定或最高法院数据库（SCDB）进行分类，并将其与先前的SOTA结果进行了比较。我们还将我们的结果与针对长文档的SOTA模型进行了比较。我们对两个分类任务进行了比较：（1）广泛的分类任务，具有15个类别；（2）细粒度的分类任务，具有279个类别。我们的最佳结果在15个广泛类别上产生80％的准确度，在279个细粒度类别上产生60％的准确度。

    Models based on bidirectional encoder representations from transformers (BERT) produce state of the art (SOTA) results on many natural language processing (NLP) tasks such as named entity recognition (NER), part-of-speech (POS) tagging etc. An interesting phenomenon occurs when classifying long documents such as those from the US supreme court where BERT-based models can be considered difficult to use on a first-pass or out-of-the-box basis. In this paper, we experiment with several BERT-based classification techniques for US supreme court decisions or supreme court database (SCDB) and compare them with the previous SOTA results. We then compare our results specifically with SOTA models for long documents. We compare our results for two classification tasks: (1) a broad classification task with 15 categories and (2) a fine-grained classification task with 279 categories. Our best result produces an accuracy of 80\% on the 15 broad categories and 60\% on the fine-grained 279 categories 
    
[^87]: 基于重排的后训练量化方法在大型语言模型中的应用

    RPTQ: Reorder-based Post-training Quantization for Large Language Models. (arXiv:2304.01089v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2304.01089](http://arxiv.org/abs/2304.01089)

    本研究提出了一种新的基于重排的量化方法RPTQ，目的是解决大型语言模型在量化时由于信道激活范围不同而产生的问题。实现该方法后，我们将LLL模型推动到3位激活。

    

    大型语言模型在各种任务上表现出色，但由于其巨大的模型大小而引发的部署挑战。本文指出，LLL模型量化的主要难点在于信道之间不同的激活范围，而不仅仅是离群值问题。我们提出了一种新颖的基于重排的量化方法RPTQ，用于解决LLL模型量化问题。RPTQ通过重新排列激活中的信道，并按簇量化信道，从而减少信道范围差异的影响。此外，我们通过避免显式重排减少存储和计算开销。实现了该方法后，我们首次将LLL模型推动到3位激活。

    Large-scale language models (LLMs) have demonstrated outstanding performance on various tasks, but their deployment poses challenges due to their enormous model size. In this paper, we identify that the main challenge in quantizing LLMs stems from the different activation ranges between the channels, rather than just the issue of outliers.We propose a novel reorder-based quantization approach, RPTQ, that addresses the issue of quantizing the activations of LLMs. RPTQ rearranges the channels in the activations and then quantizing them in clusters, thereby reducing the impact of range difference of channels. In addition, we reduce the storage and computation overhead by avoiding explicit reordering. By implementing this approach, we achieved a significant breakthrough by pushing LLM models to 3 bit activation for the first time.
    
[^88]: UKP-SQuARE v3：一个多智能体QA研究平台

    UKP-SQuARE v3: A Platform for Multi-Agent QA Research. (arXiv:2303.18120v1 [cs.CL])

    [http://arxiv.org/abs/2303.18120](http://arxiv.org/abs/2303.18120)

    UKP-SQuARE v3是一个支持多智能体系统的QA研究平台，与多数据集模型相比，结合专家智能体可以获得更好的性能提升。

    

    问题回答（QA）数据集的不断发展已引起研究界对多领域模型的关注。一种常见的方法是使用多数据集模型，这些模型经过多个数据集的训练，以学习它们的规律并防止对单个数据集过度拟合。然而，随着在线代码库（如GitHub或Hugging Face）中QA模型的激增，另一种方法正在变得可行。最近的研究表明，结合专家智能体可以比多数据集模型获得更大的性能提升。为了方便多智能体模型的研究，我们将UKP-SQuARE扩展为支持三种多智能体系统：i）智能体选择，ii）智能体的早期融合，以及iii）智能体的后期融合。我们进行实验以评估它们的推断速度，并与多数据集模型进行性能与速度权衡的讨论。UKP-SQuARE是开源的，公开可用。

    The continuous development of Question Answering (QA) datasets has drawn the research community's attention toward multi-domain models. A popular approach is to use multi-dataset models, which are models trained on multiple datasets to learn their regularities and prevent overfitting to a single dataset. However, with the proliferation of QA models in online repositories such as GitHub or Hugging Face, an alternative is becoming viable. Recent works have demonstrated that combining expert agents can yield large performance gains over multi-dataset models. To ease research in multi-agent models, we extend UKP-SQuARE, an online platform for QA research, to support three families of multi-agent systems: i) agent selection, ii) early-fusion of agents, and iii) late-fusion of agents. We conduct experiments to evaluate their inference speed and discuss the performance vs. speed trade-off compared to multi-dataset models. UKP-SQuARE is open-source and publicly available at this http URL
    
[^89]: 基于正样本增强对比学习的图像视频标题评估

    Positive-Augmented Constrastive Learning for Image and Video Captioning Evaluation. (arXiv:2303.12112v1 [cs.CV])

    [http://arxiv.org/abs/2303.12112](http://arxiv.org/abs/2303.12112)

    本论文提出一种新的图像标题评估指标PAC-S，可以更准确地评估图像和视频的标题，相比于现有的指标有更好的表现；源代码和训练模型已经公开。

    

    最近CLIP模型在很多跨模态任务上都非常有效，包括从视觉和语言结构中生成的标题评估。本文提出了一种新的基于对比度的图像标题评估指标配方，即正样本增强的对比度学习分数（PAC-S），以一种新颖的方式统一了对比度视觉-语义空间的学习和策展数据上生成的图像和文本的添加。跨越多个数据集的实验表明，我们的新指标在图像和视频上与人类判断的相关性最高，优于现有参考指标（如CIDEr和SPICE）和无参考指标（如CLIP-Score）。最后，我们考虑了流行的图像标题方法，并评估了采用不同跨模态特征的影响。我们的源代码和训练模型是公开的。

    The CLIP model has been recently proven to be very effective for a variety of cross-modal tasks, including the evaluation of captions generated from vision-and-language architectures. In this paper, we propose a new recipe for a contrastive-based evaluation metric for image captioning, namely Positive-Augmented Contrastive learning Score (PAC-S), that in a novel way unifies the learning of a contrastive visual-semantic space with the addition of generated images and text on curated data. Experiments spanning several datasets demonstrate that our new metric achieves the highest correlation with human judgments on both images and videos, outperforming existing reference-based metrics like CIDEr and SPICE and reference-free metrics like CLIP-Score. Finally, we test the system-level correlation of the proposed metric when considering popular image captioning approaches, and assess the impact of employing different cross-modal features. Our source code and trained models are publicly availa
    
[^90]: 将知识纳入文档摘要生成中：基于GPT-2的前缀调整应用

    Incorporating Knowledge into Document Summarisation: an Application of Prefix-Tuning on GPT-2. (arXiv:2301.11719v3 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2301.11719](http://arxiv.org/abs/2301.11719)

    本论文研究了将事实知识纳入生成的摘要的可能性，具体采用前缀调整的方法，实验结果表明，此方法可以生成保留知识的摘要，而且可以提升整体性能。

    

    尽管现在文档摘要技术得到了很大的发展，但是生成的摘要和原始文本之间的事实不一致仍然时有发生。本研究探索了采用提示来将事实知识纳入生成的摘要的可能性。我们具体研究了前缀调整，它使用一组可训练的连续前缀提示和离散自然语言提示来帮助摘要生成。实验结果表明，可训练的前缀可以帮助摘要模型准确地从离散提示中提取信息，从而生成保留知识的摘要，这些摘要在事实上与离散提示一致。生成的摘要的ROUGE改进表明，将事实知识明确地添加到摘要生成过程中可以提升整体性能，显示出在其他自然语言处理任务中应用的巨大潜力。

    Despite the great development of document summarisation techniques nowadays, factual inconsistencies between the generated summaries and the original texts still occur from time to time. This study explores the possibility of adopting prompts to incorporate factual knowledge into generated summaries. We specifically study prefix-tuning that uses a set of trainable continuous prefix prompts together with discrete natural language prompts to aid summary generation. Experimental results demonstrate that the trainable prefixes can help the summarisation model extract information from discrete prompts precisely, thus generating knowledge-preserving summaries that are factually consistent with the discrete prompts. The ROUGE improvements of the generated summaries indicate that explicitly adding factual knowledge into the summarisation process could boost the overall performance, showing great potential for applying it to other natural language processing tasks.
    
[^91]: 包容性文本概念论

    An Inclusive Notion of Text. (arXiv:2211.05604v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.05604](http://arxiv.org/abs/2211.05604)

    我们提出了一个通用术语以及一个语言和非语言元素的分类法，用于系统性地捕捉不同任务和数据下的文本概念差异。这显著提高了NLP模型的可重复性和可推广性。

    

    自然语言处理（NLP）研究人员根据书面文本开发语法、意义和交流模型。由于任务和数据的差异，被认为是文本的内容在研究中可能会有很大的变化。缺乏一个系统性捕捉这些差异的概念框架。我们认为，澄清文本概念对于可重复和可推广的NLP至关重要。为此，我们提出了通用术语来讨论文本数据的生产和转换，并引入了一个二层次的语言和非语言元素的分类法，这些元素在文本来源中可用于NLP建模。我们将此分类法应用于调查将文本概念扩展到保守的以语言为中心的视角以外的现有工作。我们概述了正在兴起的NLP中包容性文本方法的关键要求和挑战，并建议社区级别的报告作为巩固讨论的关键下一步。

    Natural language processing (NLP) researchers develop models of grammar, meaning and communication based on written text. Due to task and data differences, what is considered text can vary substantially across studies. A conceptual framework for systematically capturing these differences is lacking. We argue that clarity on the notion of text is crucial for reproducible and generalizable NLP. Towards that goal, we propose common terminology to discuss the production and transformation of textual data, and introduce a two-tier taxonomy of linguistic and non-linguistic elements that are available in textual sources and can be used in NLP modeling. We apply this taxonomy to survey existing work that extends the notion of text beyond the conservative language-centered view. We outline key desiderata and challenges of the emerging inclusive approach to text in NLP, and suggest community-level reporting as a crucial next step to consolidate the discussion.
    
[^92]: KGLM: 将知识图谱结构整合进语言模型以进行链接预测

    KGLM: Integrating Knowledge Graph Structure in Language Models for Link Prediction. (arXiv:2211.02744v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2211.02744](http://arxiv.org/abs/2211.02744)

    本文提出了 KGLM 架构，将新的实体/关系嵌入层整合进语言模型，使其能够学习知识图谱的结构。使用从知识图谱提取的三元组对模型进行预训练并进行链接预测任务得到了良好效果。

    

    知识图谱能够在大规模情况下表示复杂关系，已被广泛应用于知识表示、问答和推荐系统等领域。然而，知识图谱通常存在信息缺失的问题，需要进行知识图谱补全任务。预训练和微调的语言模型已经在这些任务中表现出出色的效果，尽管这些模型忽略了知识图谱所蕴含的固有信息，即实体和关系类型。因此，我们提出了KGLM（Knowledge Graph Language Model）架构，其中引入了一个新的实体/关系嵌入层，它学习区分不同的实体和关系类型，使得模型能够学习知识图谱的结构。在本文中，我们展示了使用从知识图谱提取的三元组进一步预训练这些额外嵌入层的语言模型，然后采用后续的 link prediction 任务表现出了良好的效果。

    The ability of knowledge graphs to represent complex relationships at scale has led to their adoption for various needs including knowledge representation, question-answering, and recommendation systems. Knowledge graphs are often incomplete in the information they represent, necessitating the need for knowledge graph completion tasks. Pre-trained and fine-tuned language models have shown promise in these tasks although these models ignore the intrinsic information encoded in the knowledge graph, namely the entity and relation types. In this work, we propose the Knowledge Graph Language Model (KGLM) architecture, where we introduce a new entity/relation embedding layer that learns to differentiate distinctive entity and relation types, therefore allowing the model to learn the structure of the knowledge graph. In this work, we show that further pre-training the language models with this additional embedding layer using the triples extracted from the knowledge graph, followed by the sta
    
[^93]: 不需要微调的预训练语言模型剪枝

    Pruning Pre-trained Language Models Without Fine-Tuning. (arXiv:2210.06210v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2210.06210](http://arxiv.org/abs/2210.06210)

    本文提出了静态模型剪枝（SMP），它只使用一阶剪枝来适应下游任务，同时实现目标稀疏度水平，在大量实验证明SMP具有显著的改进。

    

    为了克服预训练语言模型中过于参数化的问题，我们广泛地使用剪枝作为一种简单和直接的压缩方法，直接去除不重要的权重。先前的一阶方法成功地将PLMs压缩到极高的稀疏性，同时表现几乎不下降，如运动剪枝等。这些方法使用一阶信息来剪枝PLMs，同时微调其余的权重。在这项工作中，我们认为对于一阶剪枝，微调是多余的，因为一阶剪枝足以将PLMs收敛到下游任务，而无需微调。在这个初衷下，我们提出了静态模型剪枝（SMP），它只使用一阶剪枝来使PLMs适应下游任务，同时实现目标稀疏度水平。此外，我们还设计了一个新的蒙版函数和训练目标，以进一步改进SMP。大量各种稀疏度水平下的实验证明了SMP比一阶和零阶方法具有显著的改进。

    To overcome the overparameterized problem in Pre-trained Language Models (PLMs), pruning is widely used as a simple and straightforward compression method by directly removing unimportant weights. Previous first-order methods successfully compress PLMs to extremely high sparsity with little performance drop. These methods, such as movement pruning, use first-order information to prune PLMs while fine-tuning the remaining weights. In this work, we argue fine-tuning is redundant for first-order pruning, since first-order pruning is sufficient to converge PLMs to downstream tasks without fine-tuning. Under this motivation, we propose Static Model Pruning (SMP), which only uses first-order pruning to adapt PLMs to downstream tasks while achieving the target sparsity level. In addition, we also design a new masking function and training objective to further improve SMP. Extensive experiments at various sparsity levels show SMP has significant improvements over first-order and zero-order met
    
[^94]: 挖掘法庭裁决中的法律论点

    Mining Legal Arguments in Court Decisions. (arXiv:2208.06178v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2208.06178](http://arxiv.org/abs/2208.06178)

    该论文针对法律领域论点挖掘存在的问题，设计了深入法律论证研究理论和实践的新的注释方案，并编译了包含373项裁决的大型语料库。

    

    自从论点挖掘领域的产生以来，识别、分类和分析法律话语中的论点一直是一个重要的研究领域。然而，在自然语言处理研究人员对法庭决定中的论点进行建模和注释的方式与法律专家理解和分析法律论证方面存在重大差异。虽然计算方法通常将论点简化为通用前提和主张，但法律研究中的论点通常呈现出丰富的类型，这对于深入了解特定案例和法律应用非常重要。我们解决了这个问题，并做出了一些实质性的贡献，以推动该领域向前发展。首先，我们设计了一个深深植根于法律论证研究理论和实践的欧洲人权裁判所（ECHR）诉讼中法律论点的新的注释方案。其次，我们编译和注释了一个包括373项裁决的大型语料库。

    Identifying, classifying, and analyzing arguments in legal discourse has been a prominent area of research since the inception of the argument mining field. However, there has been a major discrepancy between the way natural language processing (NLP) researchers model and annotate arguments in court decisions and the way legal experts understand and analyze legal argumentation. While computational approaches typically simplify arguments into generic premises and claims, arguments in legal research usually exhibit a rich typology that is important for gaining insights into the particular case and applications of law in general. We address this problem and make several substantial contributions to move the field forward. First, we design a new annotation scheme for legal arguments in proceedings of the European Court of Human Rights (ECHR) that is deeply rooted in the theory and practice of legal argumentation research. Second, we compile and annotate a large corpus of 373 court decision
    

