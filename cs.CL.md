# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Constructive Large Language Models Alignment with Diverse Feedback.](http://arxiv.org/abs/2310.06450) | 本文提出了一种新的方法，即建构性和多样化反馈（CDF），用于增强大型语言模型（LLM）的对齐效果。我们通过收集不同类型的反馈，并根据问题的难度级别进行处理，实现了更好的性能。 |
| [^2] | [MemSum-DQA: Adapting An Efficient Long Document Extractive Summarizer for Document Question Answering.](http://arxiv.org/abs/2310.06436) | MemSum-DQA是一个用于文档问答的高效系统，它利用了MemSum的提取式摘要技术，通过选择性提取文本块作为答案，准确性提高了9%并在子关系理解方面表现出色。 |
| [^3] | [Whispering LLaMA: A Cross-Modal Generative Error Correction Framework for Speech Recognition.](http://arxiv.org/abs/2310.06434) | Whispering LLaMA是一种用于语音识别的跨模态生成错误校正框架，通过融合声学信息和外部语言表示，生成准确的语音转录上下文，相对于n-best假设，词错误率性能提升了37.66%。 |
| [^4] | [Retromorphic Testing: A New Approach to the Test Oracle Problem.](http://arxiv.org/abs/2310.06433) | 逆向测试是一种新颖的黑盒测试方法，通过建立双程序结构，利用辅助程序将输入数据进行前向处理，然后使用后向程序将程序输出反转为原始输入格式，从而解决了测试Oracle问题。 |
| [^5] | [Large Language Models for Propaganda Detection.](http://arxiv.org/abs/2310.06422) | 这项研究探讨了使用现代大型语言模型（LLMs）如GPT-3和GPT-4在宣传信息检测方面的有效性。实验结果显示，GPT-4达到了与当前最先进方法相符的结果。 |
| [^6] | [Humans and language models diverge when predicting repeating text.](http://arxiv.org/abs/2310.06408) | 人类和语言模型在预测重复文本时存在分歧，虽然在文本片段第一次呈现时表现一致，但当记忆（或上下文学习）开始发挥作用时，表现快速分歧。研究发现这种分歧源于中间层的特定注意力头，并通过加入幂律近期偏好使模型更接近人类行为。 |
| [^7] | [Hexa: Self-Improving for Knowledge-Grounded Dialogue System.](http://arxiv.org/abs/2310.06404) | 本论文提出了一种自我提升的方法，用于改进知识驱动对话生成的中间步骤的生成性能。通过引入引导提示和修改损失函数的自举策略，提高了生成自动生成回答的多样性，并在各种基准数据集上实验证明了该方法的有效性。 |
| [^8] | [Improved prompting and process for writing user personas with LLMs, using qualitative interviews: Capturing behaviour and personality traits of users.](http://arxiv.org/abs/2310.06391) | 本文介绍了一种使用质性访谈和大型语言模型创建用户角色的工作流程，包括改进的提示和更大的主题池。研究探讨了使用该工作流程的优势以及大型语言模型捕捉用户行为和个人特质的能力。 |
| [^9] | [P5: Plug-and-Play Persona Prompting for Personalized Response Selection.](http://arxiv.org/abs/2310.06390) | 提出了一种即插即用的个人角色提示方法，用于个性化回答选择的检索对话机器人。这种方法在零样本设置下表现良好，减少了对基于个人角色的训练数据的依赖，并且使得系统更容易扩展到其他语言。 |
| [^10] | [Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations.](http://arxiv.org/abs/2310.06387) | 本文研究了使用少量上下文示范来操纵语言模型对齐能力的方法。通过提供示范，而无需微调，可以增加或降低模型回答恶意提示的概率。我们提出了相同上下文攻击和相同上下文防御方法，用于越狱和对齐语言模型。实验证明了这些方法的有效性。 |
| [^11] | [Rethinking Model Selection and Decoding for Keyphrase Generation with Pre-trained Sequence-to-Sequence Models.](http://arxiv.org/abs/2310.06374) | 本文通过系统分析了基于预训练语言模型的关键词生成任务中模型选择和解码策略的影响。并发现传统的模型选择智慧缺乏深度，并且在关键词生成中贪婪搜索的召回率较低。 |
| [^12] | [Multi-Modal Knowledge Graph Transformer Framework for Multi-Modal Entity Alignment.](http://arxiv.org/abs/2310.06365) | 提出了一种名为MoAlign的多模态知识图谱变换器框架，通过引入邻居特征、多模态属性和实体类型，以解决多模态实体对齐任务中的挑战。设计了分层可修改的自注意力块和实体类型前缀注入方法，以更好地整合和保留不同信息的独特语义。 |
| [^13] | [InfoCL: Alleviating Catastrophic Forgetting in Continual Text Classification from An Information Theoretic Perspective.](http://arxiv.org/abs/2310.06362) | 在连续文本分类中，通过深入探索表示学习过程，我们发现信息瓶颈的压缩效应导致类比类的混淆。为了解决这个问题，我们提出了一种新的回放式连续文本分类方法InfoCL，利用快速学习和对比学习来最大化互信息，并通过对抗性记忆增强来缓解过拟合问题。 |
| [^14] | [A Semantic Invariant Robust Watermark for Large Language Models.](http://arxiv.org/abs/2310.06356) | 本论文提出了一种对大型语言模型进行语义不变水印的方法，该方法通过利用另一个嵌入式语言模型生成所有前面token的语义嵌入，然后利用训练得到的水印模型将这些语义嵌入转换为水印logits。该方法既具有攻击鲁棒性又具有安全鲁棒性。 |
| [^15] | [Selective Demonstrations for Cross-domain Text-to-SQL.](http://arxiv.org/abs/2310.06302) | 本文研究了跨领域文本到SQL任务中使用领域内演示示例的关键因素，并提出了一个演示选择框架ODIS，该框架综合利用了领域外示例和合成生成的领域内示例，成功在该任务上达到了显著的性能提高。 |
| [^16] | [Let Models Speak Ciphers: Multiagent Debate through Embeddings.](http://arxiv.org/abs/2310.06272) | 本文引入了一种名为CIPHER的通信机制，通过去除LLMs中的标记采样步骤，让模型可以通过期望的原始Transformer输出嵌入来传达其信念，从而解决了在自然语言生成中可能存在的信息丢失风险，并提供了编码更广泛信息的优势。 |
| [^17] | [Towards Mitigating Hallucination in Large Language Models via Self-Reflection.](http://arxiv.org/abs/2310.06271) | 本文通过分析医学生成性问答系统中的幻觉现象，提出了一种交互式的自我反思方法，通过增强事实性、一致性和蕴涵性，解决了大型语言模型中的幻觉问题。 |
| [^18] | [An experiment on an automated literature survey of data-driven speech enhancement methods.](http://arxiv.org/abs/2310.06260) | 本研究使用生成式预训练变压器模型对116篇数据驱动语音增强方法的论文进行了自动文献调研，评估了模型在回答具体问题方面的准确性和局限性。尽管在声学文献调研自动化方面有潜力，但仍需改进以更清晰准确地回答技术问题。 |
| [^19] | [Get the gist? Using large language models for few-shot decontextualization.](http://arxiv.org/abs/2310.06254) | 本文提出了一种使用大型语言模型的少样本去背景化方法，该方法能够在多个领域上仅使用少量样本即可达到可行的性能。 |
| [^20] | [We are what we repeatedly do: Inducing and deploying habitual schemas in persona-based responses.](http://arxiv.org/abs/2310.06245) | 该论文研究了在对话系统中引导和应用习惯性模式来生成基于角色的回应。与以往的方法不同，该论文通过明确的模式表示来捕捉与角色相关的习惯性知识，从而提高了回应的准确性和自然性。 |
| [^21] | [Model Tuning or Prompt Tuning? A Study of Large Language Models for Clinical Concept and Relation Extraction.](http://arxiv.org/abs/2310.06239) | 本研究探索了大型语言模型在临床概念和关系提取任务中的应用，通过软提示调优取得了最佳性能，并观察了迁移学习和少样本学习的能力。 |
| [^22] | [Tackling Data Bias in MUSIC-AVQA: Crafting a Balanced Dataset for Unbiased Question-Answering.](http://arxiv.org/abs/2310.06238) | 本论文解决了MUSIC-AVQA中的数据偏差问题，通过创建一个平衡的数据集来保证模型能够有效地推理各种多模态情况下的问题。他们构建了一个名为MUSIC-AVQA v2.0的新数据集，并提出了一个新的基准模型。 |
| [^23] | [Evolution of Natural Language Processing Technology: Not Just Language Processing Towards General Purpose AI.](http://arxiv.org/abs/2310.06228) | 这篇论文翻译成中文并总结指出，自然语言处理技术逐渐演进为通用人工智能，尽管自然语言非常难以进行数学上的表达，但通过深度学习在NLP领域取得了超乎预期的结果。 |
| [^24] | [GeoLLM: Extracting Geospatial Knowledge from Large Language Models.](http://arxiv.org/abs/2310.06213) | GeoLLM是一种能够从大型语言模型中提取地理空间知识的新方法，结合来自OpenStreetMap的辅助地图数据，可以有效应用于测量人口密度和经济生计等任务。 |
| [^25] | [Estimating Numbers without Regression.](http://arxiv.org/abs/2310.06204) | 提出了一种不使用回归的方法来估计数字，通过改变模型的词汇表来更好地表示数字的大小，并取得了令人满意的结果。 |
| [^26] | [GPT-who: An Information Density-based Machine-Generated Text Detector.](http://arxiv.org/abs/2310.06202) | GPT-who是一种基于统一信息密度原则的机器生成文本检测器，利用基于统一信息密度原则的特征来建模每个语言模型和人类作者的独特统计特征，以实现准确的作者归属。在多个领域中，GPT-who的性能超过了其他最先进的检测器，且具有计算成本低廉和可解释性。 |
| [^27] | [Compressing Context to Enhance Inference Efficiency of Large Language Models.](http://arxiv.org/abs/2310.06201) | 本研究提出了一种名为Selective Context的方法，通过识别和修剪输入上下文中的冗余信息，以降低大型语言模型的推理过程中的计算成本和延迟，并保持相当的性能。 |
| [^28] | [The Importance of Prompt Tuning for Automated Neuron Explanations.](http://arxiv.org/abs/2310.06200) | 本文探讨了在自动神经元解释中即时调优的重要性，通过重新格式化解释提示，我们显著提高了解释质量并降低了计算成本。这项研究对于更深入理解大型语言模型的工作原理和安全性具有重要意义。 |
| [^29] | [CAW-coref: Conjunction-Aware Word-level Coreference Resolution.](http://arxiv.org/abs/2310.06165) | 本文介绍了一种关联词感知的词级共指消解模型（CAW-coref），在处理并列提及的情况下表现出了较高的性能，有效地缩小了与昂贵的最先进方法的差距。 |
| [^30] | [Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models.](http://arxiv.org/abs/2310.06117) | 本文提出了一种简单的提示技术，使得大型语言模型能够通过抽象获得高层概念和基本原理，并将其应用于推理路径中，从而显著提升模型在各种推理密集型任务上的表现。 |
| [^31] | [BYOC: Personalized Few-Shot Classification with Co-Authored Class Descriptions.](http://arxiv.org/abs/2310.06111) | 提出了一种个性化的少样本文本分类方法，使用合著的类别描述作为提示，用户与LLM交互合作进行标注，形成分类提示，实验结果显示其准确率达到了使用大规模数据集进行训练的模型的82%。 |
| [^32] | [Leveraging Multilingual Self-Supervised Pretrained Models for Sequence-to-Sequence End-to-End Spoken Language Understanding.](http://arxiv.org/abs/2310.06103) | 这项工作提出了一种利用多语言自监督预训练模型的统一方法，用于多语言的序列到序列的端到端口语理解，成功地预测了词汇填充器，并在多个数据集上取得了优于其他方法的结果。 |
| [^33] | [Auditing Gender Analyzers on Text Data.](http://arxiv.org/abs/2310.06061) | 对三种现有的性别分析器进行了不对二元个体做预测的偏见的审计，发现这些工具在准确性上存在严重问题，并且对非二元评论的预测主要是女性，从而传播了对非二元人群的偏见。 |
| [^34] | [LLM for SoC Security: A Paradigm Shift.](http://arxiv.org/abs/2310.06046) | 本论文研究利用大型语言模型（LLM）解决系统级芯片（SoC）设计中的安全性问题，通过将LLM集成到SoC安全验证流程中，开辟了更高效、可伸缩和适应的新方法。目标是确保越来越复杂的SoC的安全性。 |
| [^35] | [Enhancing Document-level Event Argument Extraction with Contextual Clues and Role Relevance.](http://arxiv.org/abs/2310.05991) | 本文提出了一个SCPRG模型，通过引入Span-Trigger-based Contextual Pooling(STCP)和Role-based Latent Information Guidance (RLIG)模块，解决了文档级事件论证中忽略的非论证上下文线索信息以及论证角色相关性的问题。模型通过自适应地选择和汇聚上下文中的非论证线索词，以及构建潜在的角色表示并捕捉语义相关性，显著提升了文档级事件论证的准确性。 |
| [^36] | [An evolutionary model of personality traits related to cooperative behavior using a large language model.](http://arxiv.org/abs/2310.05976) | 本文提出了一个使用大规模语言模型的进化模型，研究了个性特征在博弈理论关系背景下的演化。实验证明该模型能够展示出演化的特征。 |
| [^37] | [Exploring Embeddings for Measuring Text Relatedness: Unveiling Sentiments and Relationships in Online Comments.](http://arxiv.org/abs/2310.05964) | 本论文通过使用嵌入方法和词语间的语义关系，研究了不同社交媒体平台上评论之间的情感和语义关系，并通过分析用户评论中的文本，让研究人员、政治家和商业代表能够追踪全球用户之间共享情感的路径。 |
| [^38] | [Fingerprint Attack: Client De-Anonymization in Federated Learning.](http://arxiv.org/abs/2310.05960) | 本文研究了在联邦学习中，通过梯度指纹攻击可以轻松打破参与者匿名化，并展示了使用差分隐私进行训练可以提供实际防御。 |
| [^39] | [Vulnerability Clustering and other Machine Learning Applications of Semantic Vulnerability Embeddings.](http://arxiv.org/abs/2310.05935) | 本研究使用自然语言处理技术探索了语义漏洞嵌入的不同类型，并将其用于聚类、分类和可视化等机器学习应用，为计算机安全研究人员和分析师在风险评估等方面提供支持。 |
| [^40] | [NEFTune: Noisy Embeddings Improve Instruction Finetuning.](http://arxiv.org/abs/2310.05914) | NEFTune 是一种改进语言模型微调效果的方法，通过在训练过程中给嵌入向量添加噪声，可以在多个指令数据集上显著提高准确率，并对不同类型的模型都有益处。 |
| [^41] | [Universal Multi-modal Entity Alignment via Iteratively Fusing Modality Similarity Paths.](http://arxiv.org/abs/2310.05364) | 本研究提出了一种通过迭代融合模态相似路径实现通用的多模态实体对齐方法。通过统一建模和有效信息融合，解决了现有方法中模态建模不一致和低效以及模态融合效果不佳的问题。 |
| [^42] | [Enhancing Long-form Text Generation in Mental Health with Task-adaptive Tokenization.](http://arxiv.org/abs/2310.05317) | 该论文提出了一种任务自适应分词的方法，通过优化分词过程来增强在心理健康领域中的长文本生成。实验证明，该方法在减少标记数量的情况下显著提高了生成性能，并且可与大型语言模型结合使用。 |
| [^43] | [ChatRadio-Valuer: A Chat Large Language Model for Generalizable Radiology Report Generation Based on Multi-institution and Multi-system Data.](http://arxiv.org/abs/2310.05242) | ChatRadio-Valuer是一种基于大型语言模型的定制模型，用于通用放射学报告自动生成。它能够学习可泛化的表示并为模型自适应提供基础模式，解决了放射学报告中的泛化能力和异质性问题。 |
| [^44] | [TILFA: A Unified Framework for Text, Image, and Layout Fusion in Argument Mining.](http://arxiv.org/abs/2310.05210) | TILFA是一个用于论证挖掘中处理文本、图像和布局混合数据的统一框架，不仅能够理解文本，还能够检测光学字符和识别图像中的布局细节，并在辩论立场分类中取得了显著的性能提升。 |
| [^45] | [Factuality Challenges in the Era of Large Language Models.](http://arxiv.org/abs/2310.05189) | 大型语言模型存在生成虚假、错误或误导内容的问题，同时也面临被恶意应用的风险。本研究探讨了需要从事实核查员、新闻机构和研究与政策界采取的技术创新、监管改革和人工智能素养倡议，以解决这些问题。 |
| [^46] | [DialCoT Meets PPO: Decomposing and Exploring Reasoning Paths in Smaller Language Models.](http://arxiv.org/abs/2310.05074) | DialCoT是一种对话引导的链式思维方法，用于在较小的语言模型中分解和探索推理路径。通过将复杂问题分解为简单的子问题，它降低了任务难度，并使用PPO算法优化模型的推理路径选择。 |
| [^47] | [From Text to Tactic: Evaluating LLMs Playing the Game of Avalon.](http://arxiv.org/abs/2310.05036) | 本文研究了在Avalon游戏中使用LLMs的潜力，并引入了AvalonBench来评估多代理LLM代理。实验证明存在明显的能力差距。 |
| [^48] | [Self-Convinced Prompting: Few-Shot Question Answering with Repeated Introspection.](http://arxiv.org/abs/2310.05035) | 该论文提出了一种利用大规模预训练语言模型的框架，通过迭代反思和改进推理过程，以提升模型在少样本问题回答上的性能。 |
| [^49] | [Revisiting Large Language Models as Zero-shot Relation Extractors.](http://arxiv.org/abs/2310.05028) | 本研究重新审视了大型语言模型(LLMs)作为零-shot关系抽取器的潜力，并提出了通过总结和提问(\textsc{SumAsk})提示方法来改进零-shot关系抽取。实验证明LLMs在这一任务上具有良好的表现。 |
| [^50] | [DORIS-MAE: Scientific Document Retrieval using Multi-level Aspect-based Queries.](http://arxiv.org/abs/2310.04678) | DORIS-MAE是一个用于科学文档检索的新任务，旨在处理复杂的多方面查询。研究团队构建了一个基准数据集，并提出了一个基于大型语言模型的验证框架。 |
| [^51] | [Ada-Instruct: Adapting Instruction Generators for Complex Reasoning.](http://arxiv.org/abs/2310.04484) | Ada-Instruct是一种自适应指令生成器，通过对开源LLMs进行微调，能够生成复杂推理任务中长度大于等于100的指令。在代码补全、数学推理和常识推理等任务中，Ada-Instruct显示出优于基本模型和当前自我指导方法的改进效果。 |
| [^52] | [Auto-survey Challenge.](http://arxiv.org/abs/2310.04480) | 这项研究提出了一个新的平台，用于评估大型语言模型在撰写和评论调查论文方面的能力，并组织了一次竞赛来测试该平台。评估标准包括清晰度、参考适当性、可追溯性和内容的实质价值。 |
| [^53] | [What's the Magic Word? A Control Theory of LLM Prompting.](http://arxiv.org/abs/2310.04444) | 本论文将提示工程形式化为LLM上的最优控制问题，研究了给定token序列时是否存在一种最优提示能够准确预测最终的token，并提出了控制理论中的指标来描述LLM的可操纵性。 |
| [^54] | [A Comprehensive Evaluation of Large Language Models on Benchmark Biomedical Text Processing Tasks.](http://arxiv.org/abs/2310.04270) | 本文对大型语言模型（LLM）在生物医学任务中的性能进行了综合评估，发现零样本LLMs在小样本生物医学数据集上的表现甚至超过了先进的精调生物医学模型，预训练使LLMs在生物医学领域具备了很强的专业能力。 |
| [^55] | [Chain of Natural Language Inference for Reducing Large Language Model Ungrounded Hallucinations.](http://arxiv.org/abs/2310.03951) | 这项研究提出了一个分层框架，通过自然语言推理链条（CoNLI）来检测和减少大型语言模型（LLMs）的幻觉。该框架不需要对LLMs进行微调或特定领域的提示工程，能够从不同的上下文中实现具有竞争性能的幻觉检测和减少。 |
| [^56] | [Learning Personalized Story Evaluation.](http://arxiv.org/abs/2310.03304) | 该论文提出了学习个性化故事评估的方法。为了解决大型语言模型在开放式文本生成任务的评估问题，论文创建了两个新的数据集，并开发了一个个性化故事评估模型，能够根据评审人员的示例评价进行个性化评估。 |
| [^57] | [Adapting LLM Agents Through Communication.](http://arxiv.org/abs/2310.01444) | 这项研究提出了一种名为学习通信（LTC）的训练方法，使用该方法可使LLM代理通过与环境和其他代理的交互不断改进，以适应新任务，而无需过多人类监督。 |
| [^58] | [Representation Engineering: A Top-Down Approach to AI Transparency.](http://arxiv.org/abs/2310.01405) | 这项研究介绍了一种名为表示工程化（RepE）的自上而下方法，通过借鉴认知神经科学的见解，提供了一种增强AI系统透明性的解决方案。该方法将集群级别的表示放在分析的核心，为监测和操纵深度神经网络中的高级认知现象提供了新的方法，并展示了在解决与安全相关的问题上的潜力。 |
| [^59] | [LEEC: A Legal Element Extraction Dataset with an Extensive Domain-Specific Label System.](http://arxiv.org/abs/2310.01271) | 本文提出了一个具有广泛领域特定标签系统的大规模刑事要素提取数据集，通过借助法律专家团队和法律知识的注释，可以增强法律案例的解释和分析能力。 |
| [^60] | [SELF: Language-Driven Self-Evolution for Large Language Model.](http://arxiv.org/abs/2310.00533) | SELF提出了一种基于语言驱动的创新方法，允许大型语言模型（LLM）不断自我进化，并通过语言反馈作为评估工具来改进模型的响应能力和训练稳定性。 |
| [^61] | [Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models.](http://arxiv.org/abs/2310.00322) | 本文提出了红队游戏（RTG）框架，利用博弈论分析了红队语言模型（RLM）与蓝队语言模型（BLM）之间的多轮攻防互动。同时引入了游戏化红队求解器（GRTS）来提供自动化的红队技术。 |
| [^62] | [Understanding In-Context Learning from Repetitions.](http://arxiv.org/abs/2310.00297) | 本论文通过研究表面重复现象的角度来探索大型语言模型中上下文学习的机制，并实证了一种增强标记关系的原则，为理解上下文学习及其潜在局限性做出了重要贡献。 |
| [^63] | [Pairwise Proximal Policy Optimization: Harnessing Relative Feedback for LLM Alignment.](http://arxiv.org/abs/2310.00212) | 该论文提出了一种新的强化学习框架，使用相对反馈来调整大型语言模型（LLMs）的行为，解决了现有方法在优化比较损失训练的奖励时存在的限制。同时，还提出了一种新的基于轨迹的策略梯度算法（PPPO），用于更有效地进行算法设计和函数逼近。 |
| [^64] | [GPT-Fathom: Benchmarking Large Language Models to Decipher the Evolutionary Path towards GPT-4 and Beyond.](http://arxiv.org/abs/2309.16583) | GPT-Fathom是一个用于评估大型语言模型的开源套件，它系统评估了10多个主要的语言模型，并提供了从GPT-3到GPT-4演化路径的宝贵见解。 |
| [^65] | [Generative Speech Recognition Error Correction with Large Language Models.](http://arxiv.org/abs/2309.15649) | 本研究探讨了大型语言模型（LLMs）作为ASR后处理器的能力，通过重新评分和错误校正来提高系统性能。通过使用指令提示和任务激活提示方法，结合上下文学习和微调技术，我们展示了LLMs的泛化能力和有效性。 |
| [^66] | [Low-rank Adaptation of Large Language Model Rescoring for Parameter-Efficient Speech Recognition.](http://arxiv.org/abs/2309.15223) | 这篇论文介绍了一种基于低秩适应技术的神经语言建模系统，用于语音识别的输出重评分。通过使用低秩分解方法和优化插入矩阵，该系统能够以更高效的方式将BERT模型适应到新领域，大大减少了训练时间。 |
| [^67] | [What Learned Representations and Influence Functions Can Tell Us About Adversarial Examples.](http://arxiv.org/abs/2309.10916) | 本文将图像处理领域中的对抗子空间技术应用于自然语言处理，提出了基于最近邻和影响函数的检测器，并通过使用影响函数揭示了自然语言处理中的对抗样本子空间与图像处理中的子空间的关系和任务差异。 |
| [^68] | [SlimPajama-DC: Understanding Data Combinations for LLM Training.](http://arxiv.org/abs/2309.10818) | 本研究探讨了使用SlimPajama进行大型语言模型训练中不同数据组合的影响，提出了全局去重和局部去重的比较和高质量多源数据集的比例对模型性能的影响。 |
| [^69] | [PoSE: Efficient Context Window Extension of LLMs via Positional Skip-wise Training.](http://arxiv.org/abs/2309.10400) | 本文介绍了一种名为PoSE的训练方法，通过在训练过程中使用固定的上下文窗口和操纵位置索引来适应极长的上下文窗口，实验证明这种方法大大减小了内存和时间开销，对性能影响较小，成功将LLaMA模型扩展到了128k个标记。 |
| [^70] | [Pruning Large Language Models via Accuracy Predictor.](http://arxiv.org/abs/2309.09507) | 本文提出了一种通过准确性预测器来修剪大型语言模型的新方法，实验证明该方法是有效且高效的，可以自动选择最优模型并降低困惑度。 |
| [^71] | [Evaluation and Analysis of Hallucination in Large Vision-Language Models.](http://arxiv.org/abs/2308.15126) | 本文提出了基于大型语言模型的幻觉评估框架HaELM，可以评估大型视觉语言模型中的幻觉问题，并分析了导致幻觉的因素，并提出了缓解幻觉问题的建议。 |
| [^72] | [Empowering Clinicians and Democratizing Data Science: Large Language Models Automate Machine Learning for Clinical Studies.](http://arxiv.org/abs/2308.14120) | chatGPT ADA是一种能够自主开发临床研究所需的最先进的机器学习模型的大型语言模型，可将高级分析工具民主化，使非数据科学家的临床医生能够轻松应用于医学领域。 |
| [^73] | [Audio Generation with Multiple Conditional Diffusion Model.](http://arxiv.org/abs/2308.11940) | 本论文提出了一种使用多条件扩散模型进行音频生成的方法。通过引入内容和风格等额外条件，增强了现有模型的可控性。这种方法可以精确控制生成音频的时间顺序、音高和能量。由于缺乏合适的数据集和评估指标，作者整合了现有数据集并进行了实验验证。 |
| [^74] | [Look Before You Leap: An Exploratory Study of Uncertainty Measurement for Large Language Models.](http://arxiv.org/abs/2307.10236) | 本研究从不确定性的角度对大型语言模型进行了探索性研究，通过实验发现不确定性估计方法在探索和抵制大型语言模型的不良行为方面具有潜力。 |
| [^75] | [Adapting an ASR Foundation Model for Spoken Language Assessment.](http://arxiv.org/abs/2307.09378) | 本论文主要针对ASR基础模型Whisper的输出进行了详细分析，并提出了两种解决方案：微调和软提示微调。实验结果表明，可以有效改变Whisper的解码行为，生成候选人实际说出的单词。 |
| [^76] | [DocumentNet: Bridging the Data Gap in Document Pre-Training.](http://arxiv.org/abs/2306.08937) | 这项研究提出了DocumentNet方法，通过从Web上收集大规模和弱标注的数据，弥合了文档预训练中的数据差距，并在各类VDER任务中展现了显著的性能提升。 |
| [^77] | [Utilizing Longitudinal Chest X-Rays and Reports to Pre-Fill Radiology Reports.](http://arxiv.org/abs/2306.08749) | 本文提出了利用纵向胸部X光和报告来预填充放射学报告的方法，以减轻报告错误。通过利用纵向多模态数据和基于transformer的模型，可以更好地捕获包含多模态纵向就诊记录的信息。 |
| [^78] | [Adapting an Unadaptable ASR System.](http://arxiv.org/abs/2306.01208) | 本文探讨了一种适应在线服务提供商的语音识别系统的方法，采用错误纠正的方法，并以 OpenAI Whisper ASR 为例，使用 LibriSpeech 作为主要适应目标领域进行实验。结果显示该方法具有一定的泛化能力，可适用于不同体系结构的 ASR 模型。 |
| [^79] | [Preference-grounded Token-level Guidance for Language Model Fine-tuning.](http://arxiv.org/abs/2306.00398) | 本论文通过设计一个交替训练过程，在序列级别偏好与标记级别训练指导之间进行迭代，并利用学习到的指导改进LM。实验证明该方法在多个文本生成任务中表现良好。 |
| [^80] | [MiniSUPERB: Lightweight Benchmark for Self-supervised Speech Models.](http://arxiv.org/abs/2305.19011) | 这篇论文提出了一个名为MiniSUPERB的轻量级基准测试，可以有效地评估自监督语音模型的性能，并在计算成本上比SUPERB更低。同时，该论文还研究了在少样本情况下评估SSL语音模型的性能变化。 |
| [^81] | [Bactrian-X: Multilingual Replicable Instruction-Following Models with Low-Rank Adaptation.](http://arxiv.org/abs/2305.15011) | Bactrian-X是一个多语言可复制指令跟随模型，利用低秩适应性（LoRA）训练，具有低参数数量、易于替换的特点。在综合多语言评估设置中，Bactrian-X模型在性能上优于纯模型和现有的指令调优模型。 |
| [^82] | [LLMDet: A Third Party Large Language Models Generated Text Detection Tool.](http://arxiv.org/abs/2305.15004) | LLMDet是一个第三方大型语言模型生成文本检测工具，能够从特定的语言模型中确定生成文本的来源，并满足精细追踪、中间判断和快速检测的要求。 |
| [^83] | [Editing Common Sense in Transformers.](http://arxiv.org/abs/2305.14956) | 本文探究了常识判断是否与Transformer模型中的可编辑参数存在因果关联，并通过改进编辑算法和层选择策略，在常识领域中提升了编辑效果，使得经过编辑的GPT-2模型在各测试集上的F1分数相较于最佳微调基线提高了10.97%和10.73%。 |
| [^84] | [Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training.](http://arxiv.org/abs/2305.14342) | Sophia是一种用于语言模型预训练的可扩展的二阶优化算法，使用对角Hessian作为预调节器，并进行元素级别的裁剪控制更新大小。 |
| [^85] | [Skill-Based Few-Shot Selection for In-Context Learning.](http://arxiv.org/abs/2305.14210) | 本文提出了Skill-KNN，一种基于技能的少样本选择方法，用于上下文学习。它解决了现有方法的偏见问题并且不需要训练模型，适用于不断扩展或更改示例库的情况。 |
| [^86] | [Beyond Shared Vocabulary: Increasing Representational Word Similarities across Languages for Multilingual Machine Translation.](http://arxiv.org/abs/2305.14189) | 本文提出了一种超越共享词汇的方法，通过定义词级信息传输路径和使用图网络来融合跨语言的词嵌入，实现了在多语言机器翻译中提高相似含义词的对齐性和BLEU分数的一致提升。此方法只需要少量额外参数且计算成本增加有限，并且推理时间与基线相同。 |
| [^87] | [Can ChatGPT Defend its Belief in Truth? Evaluating LLM Reasoning via Debate.](http://arxiv.org/abs/2305.13160) | 本研究通过辩论式对话测试了LLM的推理能力，要求LLM在讨论过程中不仅能够得出正确答案，还能够坚持并捍卫自己的信念，从而更深入地评估其对问题的推理理解能力。 |
| [^88] | [Learning Interpretable Style Embeddings via Prompting LLMs.](http://arxiv.org/abs/2305.12696) | 本文通过提示语在大量文本上进行风格分析，从而创建了可解释的风格嵌入表示，为需要可解释性的下游应用（如作者归属）提供了重要资源。 |
| [^89] | [Beneath Surface Similarity: Large Language Models Make Reasonable Scientific Analogies after Structure Abduction.](http://arxiv.org/abs/2305.12660) | 本论文介绍了一种基于结构推理的类比结构推断任务，旨在解决大型语言模型在进行科学类比时忽视结构的问题。 |
| [^90] | [Compress, Then Prompt: Improving Accuracy-Efficiency Trade-off of LLM Inference with Transferable Prompt.](http://arxiv.org/abs/2305.11186) | 本文提出了使用可转移提示来优化压缩的LLMs的准确性和效率的平衡问题。该方法通过选择精度更高的提示显著提高了压缩的LLM在特定查询方面的生成质量，并实现了4倍推理时间加速。 |
| [^91] | [Towards Summarizing Multiple Documents with Hierarchical Relationships.](http://arxiv.org/abs/2305.01498) | 提出了一个新的数据集PeerSum用于生成科学论文的元评论，源文档具有显式层次结构的丰富文档间关系，提出了一种用于元评论生成的关系感知多任务模型Rammer。 |
| [^92] | [Extractive Summarization via ChatGPT for Faithful Summary Generation.](http://arxiv.org/abs/2304.04193) | 本文评估了ChatGPT在抽取式摘要任务中的性能，并发现其ROUGE得分相对于传统微调方法仍有待提高。研究探索了上下文学习和思维链推理对其性能提升的有效性，并通过采用先抽取后生成流程和句子选择模块缓解了其生成摘要的忠实性问题。 |
| [^93] | [On the Evaluations of ChatGPT and Emotion-enhanced Prompting for Mental Health Analysis.](http://arxiv.org/abs/2304.03347) | 本文全面评估了ChatGPT在心理健康分析和情感推理方面的表现，以及不同提示策略和情感信息对其性能的影响。结果显示，ChatGPT在心理健康分析方面表现良好，加入情感增强提示对某些任务效果显著。 |
| [^94] | [Does Human Collaboration Enhance the Accuracy of Identifying LLM-Generated Deepfake Texts?.](http://arxiv.org/abs/2304.01002) | 这项研究研究了人类合作是否增强了识别LLM生成的深度伪造文本的准确性。结果表明，合作可以潜在地提高两组人对深度伪造文本的检测准确性。 |
| [^95] | [Reflexion: an autonomous agent with dynamic memory and self-reflection.](http://arxiv.org/abs/2303.11366) | 本文提出 Reflexion 方法，给智能体赋予了动态记忆和自我反思能力，以增强其任务特定的行动选择能力。 |
| [^96] | [N-best T5: Robust ASR Error Correction using Multiple Input Hypotheses and Constrained Decoding Space.](http://arxiv.org/abs/2303.00456) | 本文提出了一种新的 N-best T5 模型，该模型利用 ASR N-best 列表作为输入，并使用受限解码过程，从而实现了针对 ASR 错误修正的强鲁棒性能。 |
| [^97] | [Multimodal Speech Recognition for Language-Guided Embodied Agents.](http://arxiv.org/abs/2302.14030) | 本文提出在语言引导的合身代理中使用多模态ASR模型来减少语音指令的转录错误，并通过考虑视觉上下文来提高转录结果。模型能够恢复多达30％的被屏蔽单词，并且训练基于文本的合身代理在遵循多模态ASR模型转录的指令下能够更高效地完成任务。 |
| [^98] | [Guiding Large Language Models via Directional Stimulus Prompting.](http://arxiv.org/abs/2302.11520) | 该文介绍了一个新的框架，用于通过生成定向刺激来指导大型语言模型在下游任务中生成所需的输出。通过策略语言模型的训练，该框架可以适应于各种语言模型和任务，并在摘要和对话生成任务中取得了最先进的表现。 |
| [^99] | [Understanding Translationese in Cross-Lingual Summarization.](http://arxiv.org/abs/2212.07220) | 本论文研究了跨语言摘要中的“翻译语言”现象对模型的影响，发现不同方法构建数据集会导致不同程度的翻译语言现象。文章还发现翻译语言会对模型的评估和性能产生影响，包括测试集中的翻译语言可能导致人工判断与自动评估之间的差异，以及训练集中的翻译语言会影响模型的训练表现和性能。 |
| [^100] | [Memorization of Named Entities in Fine-tuned BERT Models.](http://arxiv.org/abs/2212.03749) | 本文研究了细调BERT模型中命名实体的记忆程度，并采用差分隐私进行实验。实验结果表明，应用差分隐私会对模型的性能产生不利影响。 |

# 详细

[^1]: 用多样化反馈构建大型语言模型的对齐方法

    Constructive Large Language Models Alignment with Diverse Feedback. (arXiv:2310.06450v1 [cs.CL])

    [http://arxiv.org/abs/2310.06450](http://arxiv.org/abs/2310.06450)

    本文提出了一种新的方法，即建构性和多样化反馈（CDF），用于增强大型语言模型（LLM）的对齐效果。我们通过收集不同类型的反馈，并根据问题的难度级别进行处理，实现了更好的性能。

    

    在大型语言模型（LLMs）的研究中，对其与人类价值观的对齐越来越重视，以减少有害内容的影响。然而，当前的对齐方法通常仅依赖于人类反馈的单一形式，如偏好、注释标签或自然语言批评，忽视了结合这些反馈类型的潜在优势。这种限制导致性能不佳，即使有丰富的训练数据。本文引入了建构性和多样化反馈（CDF）作为增强LLM对齐的新方法，受建构学习理论的启发。我们的方法涉及收集适用于训练数据集中不同难度问题的三种不同类型的反馈。具体而言，我们利用批评反馈解决简单问题，利用改进反馈解决中等问题，利用偏好反馈解决困难问题。通过用这种多样化反馈训练我们的模型，我们获得了更好的表现。

    In recent research on large language models (LLMs), there has been a growing emphasis on aligning these models with human values to reduce the impact of harmful content. However, current alignment methods often rely solely on singular forms of human feedback, such as preferences, annotated labels, or natural language critiques, overlooking the potential advantages of combining these feedback types. This limitation leads to suboptimal performance, even when ample training data is available. In this paper, we introduce Constructive and Diverse Feedback (CDF) as a novel method to enhance LLM alignment, inspired by constructivist learning theory. Our approach involves collecting three distinct types of feedback tailored to problems of varying difficulty levels within the training dataset. Specifically, we exploit critique feedback for easy problems, refinement feedback for medium problems, and preference feedback for hard problems. By training our model with this diversified feedback, we a
    
[^2]: MemSum-DQA: 将高效的长文档提取式摘要器应用于文档问答中的适应性

    MemSum-DQA: Adapting An Efficient Long Document Extractive Summarizer for Document Question Answering. (arXiv:2310.06436v1 [cs.CL])

    [http://arxiv.org/abs/2310.06436](http://arxiv.org/abs/2310.06436)

    MemSum-DQA是一个用于文档问答的高效系统，它利用了MemSum的提取式摘要技术，通过选择性提取文本块作为答案，准确性提高了9%并在子关系理解方面表现出色。

    

    我们介绍了MemSum-DQA，这是一个用于文档问答（DQA）的高效系统，它利用了MemSum，一种长文档提取式摘要器。通过在解析文档中的每个文本块前加上提供的问题和问题类型，MemSum-DQA可以有选择地从文档中提取文本块作为答案。在全文回答任务中，这种方法相比之前的最先进基线模型提高了9%的精确匹配准确性。值得注意的是，MemSum-DQA在处理与子关系理解有关的问题上表现出色，凸显了提取式摘要技术在DQA任务中的潜力。

    We introduce MemSum-DQA, an efficient system for document question answering (DQA) that leverages MemSum, a long document extractive summarizer. By prefixing each text block in the parsed document with the provided question and question type, MemSum-DQA selectively extracts text blocks as answers from documents. On full-document answering tasks, this approach yields a 9% improvement in exact match accuracy over prior state-of-the-art baselines. Notably, MemSum-DQA excels in addressing questions related to child-relationship understanding, underscoring the potential of extractive summarization techniques for DQA tasks.
    
[^3]: Whispering LLaMA：一种用于语音识别的跨模态生成错误校正框架

    Whispering LLaMA: A Cross-Modal Generative Error Correction Framework for Speech Recognition. (arXiv:2310.06434v1 [cs.CL])

    [http://arxiv.org/abs/2310.06434](http://arxiv.org/abs/2310.06434)

    Whispering LLaMA是一种用于语音识别的跨模态生成错误校正框架，通过融合声学信息和外部语言表示，生成准确的语音转录上下文，相对于n-best假设，词错误率性能提升了37.66%。

    

    我们引入了一种新的跨模态融合技术，用于生成准确的语音转录上下文，以进行自动语音识别 (ASR) 的生成式错误校正。与现有的基于排名的重新评分方法不同，我们的方法灵活运用独特的初始化技术和参数有效的算法，通过预训练的语音和文本模型提升了ASR性能。通过对多样化的ASR数据集进行评估，我们评估了我们的融合技术的稳定性和可复现性，相对于n-best假设，我们的方法的词错误率性能提升了37.66%。为了鼓励未来的研究，我们将我们的代码和预训练模型开源在https://github.com/Srijith-rkr/Whispering-LLaMA上。

    We introduce a new cross-modal fusion technique designed for generative error correction in automatic speech recognition (ASR). Our methodology leverages both acoustic information and external linguistic representations to generate accurate speech transcription contexts. This marks a step towards a fresh paradigm in generative error correction within the realm of n-best hypotheses. Unlike the existing ranking-based rescoring methods, our approach adeptly uses distinct initialization techniques and parameter-efficient algorithms to boost ASR performance derived from pre-trained speech and text models. Through evaluation across diverse ASR datasets, we evaluate the stability and reproducibility of our fusion technique, demonstrating its improved word error rate relative (WERR) performance in comparison to n-best hypotheses by relatively 37.66%. To encourage future research, we have made our code and pre-trained models open source at https://github.com/Srijith-rkr/Whispering-LLaMA.
    
[^4]: 逆向测试：解决测试Oracle问题的新方法

    Retromorphic Testing: A New Approach to the Test Oracle Problem. (arXiv:2310.06433v1 [cs.SE])

    [http://arxiv.org/abs/2310.06433](http://arxiv.org/abs/2310.06433)

    逆向测试是一种新颖的黑盒测试方法，通过建立双程序结构，利用辅助程序将输入数据进行前向处理，然后使用后向程序将程序输出反转为原始输入格式，从而解决了测试Oracle问题。

    

    测试Oracle作为一个评估软件输出和给定输入集的预期行为之间对应关系的标准或机制。在自动化测试中，黑盒技术被广泛使用，其中包括差分测试和变异测试等著名方法，以其非侵入性的测试Oracle构建方式而闻名。受到数学概念的启发，我们提出了逆向测试，一种新颖的黑盒测试方法。它利用辅助程序与被测试程序结合，建立了由前向程序和后向程序组成的双程序结构。输入数据首先由前向程序处理，然后使用后向程序将其程序输出反转为其原始输入格式。特别地，辅助程序可以作为前向程序或后向程序运行，从而导致不同的测试模式。该过程通过检查这两个程序之间的关系来进行测试。

    A test oracle serves as a criterion or mechanism to assess the correspondence between software output and the anticipated behavior for a given input set. In automated testing, black-box techniques, known for their non-intrusive nature in test oracle construction, are widely used, including notable methodologies like differential testing and metamorphic testing. Inspired by the mathematical concept of inverse function, we present Retromorphic Testing, a novel black-box testing methodology. It leverages an auxiliary program in conjunction with the program under test, which establishes a dual-program structure consisting of a forward program and a backward program. The input data is first processed by the forward program and then its program output is reversed to its original input format using the backward program. In particular, the auxiliary program can operate as either the forward or backward program, leading to different testing modes. The process concludes by examining the relation
    
[^5]: 用于传播信息检测的大型语言模型

    Large Language Models for Propaganda Detection. (arXiv:2310.06422v1 [cs.CL])

    [http://arxiv.org/abs/2310.06422](http://arxiv.org/abs/2310.06422)

    这项研究探讨了使用现代大型语言模型（LLMs）如GPT-3和GPT-4在宣传信息检测方面的有效性。实验结果显示，GPT-4达到了与当前最先进方法相符的结果。

    

    在我们数字化社会中，宣传信息的普遍存在对社会和真相的传播构成了挑战。通过自然语言处理在文本中检测宣传信息是具有挑战性的，因为存在微妙的操纵技术和语境依赖。为了解决这个问题，我们研究了现代大型语言模型（LLMs）如GPT-3和GPT-4在宣传信息检测方面的有效性。我们使用SemEval-2020任务11数据集进行实验，该数据集包含具有14种宣传技术标签的新闻文章，作为一个多标签分类问题。我们采用了GPT-3和GPT-4的五种变体，结合了不同模型之间的各种提示工程和微调策略。通过评估$F1$分数，$Precision$和$Recall$等指标来评估模型的性能，并将结果与使用RoBERTa的当前最先进方法进行比较。我们的研究结果表明，GPT-4实现了与当前最先进方法相当的结果。

    The prevalence of propaganda in our digital society poses a challenge to societal harmony and the dissemination of truth. Detecting propaganda through NLP in text is challenging due to subtle manipulation techniques and contextual dependencies. To address this issue, we investigate the effectiveness of modern Large Language Models (LLMs) such as GPT-3 and GPT-4 for propaganda detection. We conduct experiments using the SemEval-2020 task 11 dataset, which features news articles labeled with 14 propaganda techniques as a multi-label classification problem. Five variations of GPT-3 and GPT-4 are employed, incorporating various prompt engineering and fine-tuning strategies across the different models. We evaluate the models' performance by assessing metrics such as $F1$ score, $Precision$, and $Recall$, comparing the results with the current state-of-the-art approach using RoBERTa. Our findings demonstrate that GPT-4 achieves comparable results to the current state-of-the-art. Further, thi
    
[^6]: 人类和语言模型在预测重复文本时存在分歧

    Humans and language models diverge when predicting repeating text. (arXiv:2310.06408v1 [cs.CL])

    [http://arxiv.org/abs/2310.06408](http://arxiv.org/abs/2310.06408)

    人类和语言模型在预测重复文本时存在分歧，虽然在文本片段第一次呈现时表现一致，但当记忆（或上下文学习）开始发挥作用时，表现快速分歧。研究发现这种分歧源于中间层的特定注意力头，并通过加入幂律近期偏好使模型更接近人类行为。

    

    使用训练于下一个单词预测任务的语言模型已被证明能准确地模拟人类的单词预测和阅读速度。然而，与这些研究结果相反，我们展示了一种情况，在这种情况下，人类和语言模型的表现存在分歧。我们收集了人类对五个由重复文本片段组成的刺激的下一个单词预测的数据集。人类和GPT-2语言模型在文本片段的第一次呈现中预测结果高度一致，但当记忆（或上下文学习）开始发挥作用时，它们的表现迅速分歧。我们追踪了这种分歧的原因，发现它源于中间层的特定注意力头。向这些注意力头加入幂律近期偏好能够使模型的表现更接近人类。我们希望这个情景能够促进将语言模型更加接近人类行为的未来研究。

    Language models that are trained on the next-word prediction task have been shown to accurately model human behavior in word prediction and reading speed. In contrast with these findings, we present a scenario in which the performance of humans and LMs diverges. We collected a dataset of human next-word predictions for five stimuli that are formed by repeating spans of text. Human and GPT-2 LM predictions are strongly aligned in the first presentation of a text span, but their performance quickly diverges when memory (or in-context learning) begins to play a role. We traced the cause of this divergence to specific attention heads in a middle layer. Adding a power-law recency bias to these attention heads yielded a model that performs much more similarly to humans. We hope that this scenario will spur future work in bringing LMs closer to human behavior.
    
[^7]: Hexa: 知识驱动的对话系统的自我提升

    Hexa: Self-Improving for Knowledge-Grounded Dialogue System. (arXiv:2310.06404v1 [cs.CL])

    [http://arxiv.org/abs/2310.06404](http://arxiv.org/abs/2310.06404)

    本论文提出了一种自我提升的方法，用于改进知识驱动对话生成的中间步骤的生成性能。通过引入引导提示和修改损失函数的自举策略，提高了生成自动生成回答的多样性，并在各种基准数据集上实验证明了该方法的有效性。

    

    知识驱动的对话生成中一种常见的做法是使用模块化的方法明确地利用中间步骤（如网络搜索、记忆检索）。然而，与对话响应相比，这些步骤的数据往往难以获取，因为在普通对话中无法观察到它们。为了填补这些数据的缺失，我们开发了一种自我提升方法，以改进中间步骤的生成性能，而不需要地面真实数据。具体而言，我们提出了一种新颖的引导提示和修改的损失函数的引导自动生成回答多样性的自举方法。通过在各种基准数据集上进行实验，我们经验证明我们的方法成功地利用了自我提升机制，在生成中间和最终回答方面改善了知识驱动对话生成任务的性能。

    A common practice in knowledge-grounded dialogue generation is to explicitly utilize intermediate steps (e.g., web-search, memory retrieval) with modular approaches. However, data for such steps are often inaccessible compared to those of dialogue responses as they are unobservable in an ordinary dialogue. To fill in the absence of these data, we develop a self-improving method to improve the generative performances of intermediate steps without the ground truth data. In particular, we propose a novel bootstrapping scheme with a guided prompt and a modified loss function to enhance the diversity of appropriate self-generated responses. Through experiments on various benchmark datasets, we empirically demonstrate that our method successfully leverages a self-improving mechanism in generating intermediate and final responses and improves the performances on the task of knowledge-grounded dialogue generation.
    
[^8]: 使用质性访谈改进大型语言模型创建用户角色的指导和流程：捕捉用户行为和个人特质

    Improved prompting and process for writing user personas with LLMs, using qualitative interviews: Capturing behaviour and personality traits of users. (arXiv:2310.06391v1 [cs.HC])

    [http://arxiv.org/abs/2310.06391](http://arxiv.org/abs/2310.06391)

    本文介绍了一种使用质性访谈和大型语言模型创建用户角色的工作流程，包括改进的提示和更大的主题池。研究探讨了使用该工作流程的优势以及大型语言模型捕捉用户行为和个人特质的能力。

    

    本文介绍了一种使用质性访谈的主题分析结果来创建用户角色的工作流程。提出的工作流程使用改进的提示和更大的主题池，相较于作者之前为同样任务所做的工作。这得益于最近发布的大型语言模型（GPT3.5-Turbo-16k），其能够处理1.6万个标记，并且可以为创建角色提供更精确的提示。文章详细介绍了执行主题分析的第二和第三阶段，然后讨论了创建角色的改进工作流程。文章还对所提出的过程与数据驱动和质性角色等现有角色方法之间的关系提出了一些反思。此外，文章还对大型语言模型从基础的质性访谈数据集中捕捉用户行为和个人特质的能力进行了思考。

    This draft paper presents a workflow for creating User Personas with Large Language Models, using the results of a Thematic Analysis of qualitative interviews. The proposed workflow uses improved prompting and a larger pool of Themes, compared to previous work conducted by the author for the same task. This is possible due to the capabilities of a recently released LLM which allows the processing of 16 thousand tokens (GPT3.5-Turbo-16k) and also due to the possibility to offer a refined prompting for the creation of Personas. The paper offers details of performing Phase 2 and 3 of Thematic Analysis, and then discusses the improved workflow for creating Personas. The paper also offers some reflections on the relationship between the proposed process and existing approaches to Personas such as the data-driven and qualitative Personas. Moreover, the paper offers reflections on the capacity of LLMs to capture user behaviours and personality traits, from the underlying dataset of qualitativ
    
[^9]: P5: 用于个性化回答选择的即插即用个人角色提示

    P5: Plug-and-Play Persona Prompting for Personalized Response Selection. (arXiv:2310.06390v1 [cs.CL])

    [http://arxiv.org/abs/2310.06390](http://arxiv.org/abs/2310.06390)

    提出了一种即插即用的个人角色提示方法，用于个性化回答选择的检索对话机器人。这种方法在零样本设置下表现良好，减少了对基于个人角色的训练数据的依赖，并且使得系统更容易扩展到其他语言。

    

    使用基于个人角色的检索对话机器人对于个性化对话至关重要，但也面临一些挑战。1）通常情况下，收集基于个人角色的语料库非常昂贵。2）在实际应用中，对话机器人系统并不总是根据个人角色做出回应。为了解决这些挑战，我们提出了一种即插即用的个人角色提示方法。如果个人角色信息不可用，我们的系统可以作为标准的开放域对话机器人运行。我们证明了这种方法在零样本设置下表现良好，从而减少了对基于个人角色的训练数据的依赖。这使得在无需构建基于个人角色的语料库的情况下，更容易将该系统扩展到其他语言。此外，我们的模型可以进行微调以获得更好的性能。在实验中，零样本模型在原始个人角色和修订个人角色方面分别提高了7.71和1.04个点。

    The use of persona-grounded retrieval-based chatbots is crucial for personalized conversations, but there are several challenges that need to be addressed. 1) In general, collecting persona-grounded corpus is very expensive. 2) The chatbot system does not always respond in consideration of persona at real applications. To address these challenges, we propose a plug-and-play persona prompting method. Our system can function as a standard open-domain chatbot if persona information is not available. We demonstrate that this approach performs well in the zero-shot setting, which reduces the dependence on persona-ground training data. This makes it easier to expand the system to other languages without the need to build a persona-grounded corpus. Additionally, our model can be fine-tuned for even better performance. In our experiments, the zero-shot model improved the standard model by 7.71 and 1.04 points in the original persona and revised persona, respectively. The fine-tuned model impro
    
[^10]: 只需少量上下文示范即可实现越狱和对齐的语言模型

    Jailbreak and Guard Aligned Language Models with Only Few In-Context Demonstrations. (arXiv:2310.06387v1 [cs.LG])

    [http://arxiv.org/abs/2310.06387](http://arxiv.org/abs/2310.06387)

    本文研究了使用少量上下文示范来操纵语言模型对齐能力的方法。通过提供示范，而无需微调，可以增加或降低模型回答恶意提示的概率。我们提出了相同上下文攻击和相同上下文防御方法，用于越狱和对齐语言模型。实验证明了这些方法的有效性。

    

    大规模语言模型（LLM）在各种任务中取得了显著的成功，但对其安全性和生成恶意内容的潜在风险的担忧也浮现出来。本文探讨了在相同上下文学习（ICL）中操纵LLM对齐能力的效果。我们发现，仅通过少量的上下文示范而无需微调，就可以操纵LLM增加或降低越狱概率，即回答恶意提示。基于这些观察，我们提出了用于越狱和对齐语言模型目的的相同上下文攻击（ICA）和相同上下文防御（ICD）方法。ICA通过构造恶意上下文指导模型生成有害输出，而ICD通过拒绝回答有害提示的示范来增强模型的稳健性。我们的实验证明了ICA和ICD在增加或降低对抗越狱攻击成功率方面的有效性。总的来说，我们揭示了ICL在越狱和对齐语言模型领域的潜力。

    Large Language Models (LLMs) have shown remarkable success in various tasks, but concerns about their safety and the potential for generating malicious content have emerged. In this paper, we explore the power of In-Context Learning (ICL) in manipulating the alignment ability of LLMs. We find that by providing just few in-context demonstrations without fine-tuning, LLMs can be manipulated to increase or decrease the probability of jailbreaking, i.e. answering malicious prompts. Based on these observations, we propose In-Context Attack (ICA) and In-Context Defense (ICD) methods for jailbreaking and guarding aligned language model purposes. ICA crafts malicious contexts to guide models in generating harmful outputs, while ICD enhances model robustness by demonstrations of rejecting to answer harmful prompts. Our experiments show the effectiveness of ICA and ICD in increasing or reducing the success rate of adversarial jailbreaking attacks. Overall, we shed light on the potential of ICL t
    
[^11]: 重新思考基于预训练序列到序列模型的关键词生成的模型选择和解码问题

    Rethinking Model Selection and Decoding for Keyphrase Generation with Pre-trained Sequence-to-Sequence Models. (arXiv:2310.06374v1 [cs.CL])

    [http://arxiv.org/abs/2310.06374](http://arxiv.org/abs/2310.06374)

    本文通过系统分析了基于预训练语言模型的关键词生成任务中模型选择和解码策略的影响。并发现传统的模型选择智慧缺乏深度，并且在关键词生成中贪婪搜索的召回率较低。

    

    关键词生成是自然语言处理中一个长期存在的任务，具有广泛的应用。序列到序列预训练语言模型的出现为关键词生成带来了革命性的改进，取得了令人期待的性能提升。然而，许多设计决策仍未被探索，并且常常是随意决策的。本文对基于预训练语言模型的关键词生成任务的模型选择和解码策略进行了系统分析。我们首先通过一个基于注意力的假设阐明了为什么序列到序列预训练语言模型适用于关键词生成。然后我们发现传统的选择序列到序列预训练语言模型的智慧缺乏深度：（1）仅增加模型大小或进行任务特定适应性调整并不是参数高效的；（2）尽管将领域内预训练与任务适应结合有利于关键词生成，但它在一定程度上会阻碍泛化能力。关于解码，我们证明贪婪搜索虽然在 F1 得分上表现出色，但在召回率方面落后于基于采样的方法。

    Keyphrase Generation (KPG) is a longstanding task in NLP with widespread applications. The advent of sequence-to-sequence (seq2seq) pre-trained language models (PLMs) has ushered in a transformative era for KPG, yielding promising performance improvements. However, many design decisions remain unexplored and are often made arbitrarily. This paper undertakes a systematic analysis of the influence of model selection and decoding strategies on PLM-based KPG. We begin by elucidating why seq2seq PLMs are apt for KPG, anchored by an attention-driven hypothesis. We then establish that conventional wisdom for selecting seq2seq PLMs lacks depth: (1) merely increasing model size or performing task-specific adaptation is not parameter-efficient; (2) although combining in-domain pre-training with task adaptation benefits KPG, it does partially hinder generalization. Regarding decoding, we demonstrate that while greedy search delivers strong F1 scores, it lags in recall compared with sampling-based
    
[^12]: 多模态知识图谱变换器框架用于多模态实体对齐

    Multi-Modal Knowledge Graph Transformer Framework for Multi-Modal Entity Alignment. (arXiv:2310.06365v1 [cs.CL])

    [http://arxiv.org/abs/2310.06365](http://arxiv.org/abs/2310.06365)

    提出了一种名为MoAlign的多模态知识图谱变换器框架，通过引入邻居特征、多模态属性和实体类型，以解决多模态实体对齐任务中的挑战。设计了分层可修改的自注意力块和实体类型前缀注入方法，以更好地整合和保留不同信息的独特语义。

    

    多模态实体对齐是一项重要的任务，旨在识别跨多模态知识图谱中的等价实体对。然而，该任务面临挑战，原因是存在不同类型的信息，包括邻近实体、多模态属性和实体类型。直接融合上述信息（如连接或注意力）可能导致无法对齐的信息空间。为了解决这些挑战，我们提出了一种新颖的多模态实体对齐变换器MoAlign，通过在变换器编码器中设计一个分层可修改的自注意力块，Hierarchical MoAlign引入邻居特征、多模态属性和实体类型来增强对齐任务。此外，我们设计了两种实体类型前缀注入方法，利用类型前缀来整合实体类型信息，有助于更好地保留不同信息的独特语义。

    Multi-Modal Entity Alignment (MMEA) is a critical task that aims to identify equivalent entity pairs across multi-modal knowledge graphs (MMKGs). However, this task faces challenges due to the presence of different types of information, including neighboring entities, multi-modal attributes, and entity types. Directly incorporating the above information (e.g., concatenation or attention) can lead to an unaligned information space. To address these challenges, we propose a novel MMEA transformer, called MoAlign, that hierarchically introduces neighbor features, multi-modal attributes, and entity types to enhance the alignment task. Taking advantage of the transformer's ability to better integrate multiple information, we design a hierarchical modifiable self-attention block in a transformer encoder to preserve the unique semantics of different information. Furthermore, we design two entity-type prefix injection methods to integrate entity-type information using type prefixes, which help
    
[^13]: InfoCL：从信息论的角度缓解连续文本分类中的灾难性遗忘

    InfoCL: Alleviating Catastrophic Forgetting in Continual Text Classification from An Information Theoretic Perspective. (arXiv:2310.06362v1 [cs.CL])

    [http://arxiv.org/abs/2310.06362](http://arxiv.org/abs/2310.06362)

    在连续文本分类中，通过深入探索表示学习过程，我们发现信息瓶颈的压缩效应导致类比类的混淆。为了解决这个问题，我们提出了一种新的回放式连续文本分类方法InfoCL，利用快速学习和对比学习来最大化互信息，并通过对抗性记忆增强来缓解过拟合问题。

    

    连续学习（CL）旨在不断学习新知识的同时避免对旧任务的灾难性遗忘。我们集中于在类增量设置下的连续文本分类。最近的CL研究已经确定了类比类上的性能严重降低是灾难性遗忘的一个关键因素。通过对CL中表示学习过程的深入探索，我们发现信息瓶颈的压缩效应导致了类比类的混淆。为了使模型学习更充分的表示，我们提出了一种新的基于回放的连续文本分类方法InfoCL。我们的方法利用快慢和当前过去对比学习来进行互信息最大化，并更好地恢复之前学到的表示。此外，InfoCL还采用对抗性记忆增强策略来缓解回放的过拟合问题。实验结果表明…

    Continual learning (CL) aims to constantly learn new knowledge over time while avoiding catastrophic forgetting on old tasks. We focus on continual text classification under the class-incremental setting. Recent CL studies have identified the severe performance decrease on analogous classes as a key factor for catastrophic forgetting. In this paper, through an in-depth exploration of the representation learning process in CL, we discover that the compression effect of the information bottleneck leads to confusion on analogous classes. To enable the model learn more sufficient representations, we propose a novel replay-based continual text classification method, InfoCL. Our approach utilizes fast-slow and current-past contrastive learning to perform mutual information maximization and better recover the previously learned representations. In addition, InfoCL incorporates an adversarial memory augmentation strategy to alleviate the overfitting problem of replay. Experimental results demo
    
[^14]: 大型语言模型的语义不变鲁棒水印

    A Semantic Invariant Robust Watermark for Large Language Models. (arXiv:2310.06356v1 [cs.CR])

    [http://arxiv.org/abs/2310.06356](http://arxiv.org/abs/2310.06356)

    本论文提出了一种对大型语言模型进行语义不变水印的方法，该方法通过利用另一个嵌入式语言模型生成所有前面token的语义嵌入，然后利用训练得到的水印模型将这些语义嵌入转换为水印logits。该方法既具有攻击鲁棒性又具有安全鲁棒性。

    

    大型语言模型的水印算法在检测LLM生成的文本方面取得了极高的准确性。然而，先前的算法在攻击鲁棒性和安全鲁棒性之间存在一个权衡。本文提出了一种对LLM进行语义不变水印的方法，该方法既具有攻击鲁棒性又具有安全鲁棒性。我们的方法通过另一个嵌入式LLM生成所有前面token的语义嵌入，然后通过训练得到的水印模型将这些语义嵌入转换为水印logits。

    Watermark algorithms for large language models (LLMs) have achieved extremely high accuracy in detecting text generated by LLMs. Such algorithms typically involve adding extra watermark logits to the LLM's logits at each generation step. However, prior algorithms face a trade-off between attack robustness and security robustness. This is because the watermark logits for a token are determined by a certain number of preceding tokens; a small number leads to low security robustness, while a large number results in insufficient attack robustness. In this work, we propose a semantic invariant watermarking method for LLMs that provides both attack robustness and security robustness. The watermark logits in our work are determined by the semantics of all preceding tokens. Specifically, we utilize another embedding LLM to generate semantic embeddings for all preceding tokens, and then these semantic embeddings are transformed into the watermark logits through our trained watermark model. Subs
    
[^15]: 跨领域文本到SQL的选择性演示

    Selective Demonstrations for Cross-domain Text-to-SQL. (arXiv:2310.06302v1 [cs.CL])

    [http://arxiv.org/abs/2310.06302](http://arxiv.org/abs/2310.06302)

    本文研究了跨领域文本到SQL任务中使用领域内演示示例的关键因素，并提出了一个演示选择框架ODIS，该框架综合利用了领域外示例和合成生成的领域内示例，成功在该任务上达到了显著的性能提高。

    

    大型语言模型（LLM）通过上下文学习展现出了令人印象深刻的跨领域文本到SQL任务的泛化能力，而无需使用领域内注释。然而，研究发现加入领域内演示示例可以大大提高LLM的性能。在本文中，我们深入探讨了领域内示例中对改进起到关键作用的因素，并探索是否可以在不依赖领域内注释的情况下利用这些好处。基于我们的发现，我们提出了一个演示选择框架ODIS，它利用领域外示例和合成生成的领域内示例来构建演示。通过从混合来源检索演示，ODIS充分利用了两者的优势，与仅依赖单一数据源的基线方法相比展现出其有效性。此外，ODIS在两个跨领域文本到SQL数据集上的表现超过了最先进的方法，改进了1.1个

    Large language models (LLMs) with in-context learning have demonstrated impressive generalization capabilities in the cross-domain text-to-SQL task, without the use of in-domain annotations. However, incorporating in-domain demonstration examples has been found to greatly enhance LLMs' performance. In this paper, we delve into the key factors within in-domain examples that contribute to the improvement and explore whether we can harness these benefits without relying on in-domain annotations. Based on our findings, we propose a demonstration selection framework ODIS which utilizes both out-of-domain examples and synthetically generated in-domain examples to construct demonstrations. By retrieving demonstrations from hybrid sources, ODIS leverages the advantages of both, showcasing its effectiveness compared to baseline methods that rely on a single data source. Furthermore, ODIS outperforms state-of-the-art approaches on two cross-domain text-to-SQL datasets, with improvements of 1.1 a
    
[^16]: 让模型说密文: 通过嵌入进行多智能体辩论

    Let Models Speak Ciphers: Multiagent Debate through Embeddings. (arXiv:2310.06272v1 [cs.CL])

    [http://arxiv.org/abs/2310.06272](http://arxiv.org/abs/2310.06272)

    本文引入了一种名为CIPHER的通信机制，通过去除LLMs中的标记采样步骤，让模型可以通过期望的原始Transformer输出嵌入来传达其信念，从而解决了在自然语言生成中可能存在的信息丢失风险，并提供了编码更广泛信息的优势。

    

    最近，对大型语言模型（LLMs）之间的讨论和辩论引起了广泛关注，因为它们有潜力增强LLMs的推理能力。尽管自然语言由于LLMs的语言理解能力而成为明显的交流选择，但生成自然语言时需要进行的标记采样步骤可能存在信息丢失的潜在风险，因为它仅使用一个标记来代表模型在整个词汇表中的信念。在本文中，我们介绍了一种名为CIPHER（通过嵌入表示进行交流的网络模型协议）的通信机制来解决这个问题。具体来说，我们从LLMs中去除了标记采样步骤，让它们通过原始Transformer输出嵌入的期望来传达它们的信念。值得注意的是，通过偏离自然语言，CIPHER在不对模型权重进行任何修改的情况下，提供了编码更广泛信息的优势。

    Discussion and debate among Large Language Models (LLMs) have gained considerable attention due to their potential to enhance the reasoning ability of LLMs. Although natural language is an obvious choice for communication due to LLM's language understanding capability, the token sampling step needed when generating natural language poses a potential risk of information loss, as it uses only one token to represent the model's belief across the entire vocabulary. In this paper, we introduce a communication regime named CIPHER (Communicative Inter-Model Protocol Through Embedding Representation) to address this issue. Specifically, we remove the token sampling step from LLMs and let them communicate their beliefs across the vocabulary through the expectation of the raw transformer output embeddings. Remarkably, by deviating from natural language, CIPHER offers an advantage of encoding a broader spectrum of information without any modification to the model weights. While the state-of-the-a
    
[^17]: 通过自我反思以减轻大型语言模型中的幻觉

    Towards Mitigating Hallucination in Large Language Models via Self-Reflection. (arXiv:2310.06271v1 [cs.CL])

    [http://arxiv.org/abs/2310.06271](http://arxiv.org/abs/2310.06271)

    本文通过分析医学生成性问答系统中的幻觉现象，提出了一种交互式的自我反思方法，通过增强事实性、一致性和蕴涵性，解决了大型语言模型中的幻觉问题。

    

    大型语言模型（LLMs）在生成和知识密集型任务（包括问答任务）方面显示出了潜力。然而，实际部署仍面临挑战，特别是“幻觉”问题：模型生成似乎合理但不真实或荒谬的信息。在医学领域，这个问题尤为关键，因为涉及到不常见的专业概念和潜在的社会风险。本文使用广泛采用的LLMs和数据集，分析了医学生成性问答系统中的幻觉现象。我们的研究重点是识别和理解常见的问题答案，特别是幻觉。为了解决这个挑战，我们提出了一种交互式的自我反思方法，结合了知识获取和答案生成。通过这个反馈过程，我们的方法逐步增强了生成答案的事实性、一致性和蕴涵性。

    Large language models (LLMs) have shown promise for generative and knowledge-intensive tasks including question-answering (QA) tasks. However, the practical deployment still faces challenges, notably the issue of "hallucination", where models generate plausible-sounding but unfaithful or nonsensical information. This issue becomes particularly critical in the medical domain due to the uncommon professional concepts and potential social risks involved. This paper analyses the phenomenon of hallucination in medical generative QA systems using widely adopted LLMs and datasets. Our investigation centers on the identification and comprehension of common problematic answers, with a specific emphasis on hallucination. To tackle this challenge, we present an interactive self-reflection methodology that incorporates knowledge acquisition and answer generation. Through this feedback process, our approach steadily enhances the factuality, consistency, and entailment of the generated answers. Cons
    
[^18]: 数据驱动语音增强方法自动文献调研的实验

    An experiment on an automated literature survey of data-driven speech enhancement methods. (arXiv:2310.06260v1 [cs.SD])

    [http://arxiv.org/abs/2310.06260](http://arxiv.org/abs/2310.06260)

    本研究使用生成式预训练变压器模型对116篇数据驱动语音增强方法的论文进行了自动文献调研，评估了模型在回答具体问题方面的准确性和局限性。尽管在声学文献调研自动化方面有潜力，但仍需改进以更清晰准确地回答技术问题。

    

    随着声学领域科学出版物数量的增加，传统的文献调研变得困难。本研究探讨了使用生成式预训练变压器（GPT）模型自动调查116篇数据驱动语音增强方法的文献调研。主要目标是评估该模型在提供关于从基于人工调研中选定的论文的具体问题的准确回答方面的能力和局限性。虽然我们看到在声学文献调研自动化方面有很大潜力，但仍需改进以更清晰准确地回答技术问题。

    The increasing number of scientific publications in acoustics, in general, presents difficulties in conducting traditional literature surveys. This work explores the use of a generative pre-trained transformer (GPT) model to automate a literature survey of 116 articles on data-driven speech enhancement methods. The main objective is to evaluate the capabilities and limitations of the model in providing accurate responses to specific queries about the papers selected from a reference human-based survey. While we see great potential to automate literature surveys in acoustics, improvements are needed to address technical questions more clearly and accurately.
    
[^19]: 看梗？使用大型语言模型进行少样本去背景化

    Get the gist? Using large language models for few-shot decontextualization. (arXiv:2310.06254v1 [cs.CL])

    [http://arxiv.org/abs/2310.06254](http://arxiv.org/abs/2310.06254)

    本文提出了一种使用大型语言模型的少样本去背景化方法，该方法能够在多个领域上仅使用少量样本即可达到可行的性能。

    

    在涉及解释富有上下文的句子的许多自然语言处理应用中，例如信息检索系统或对话系统，很有必要能够将句子保留在一个可以在没有上下文的情况下轻松理解的形式中，以便以后重新使用，这个过程被称为“去背景化”。虽然以前的工作证明了经过细调的生成型Seq2Seq模型可以有效地进行特定数据集上的去背景化，但这种方法需要昂贵的人工注释，而且可能无法迁移到其他领域。我们提出了一种使用大型语言模型的少样本去背景化方法，并展示了初步结果，表明该方法在多个领域上仅使用少量样本即可达到可行的性能。

    In many NLP applications that involve interpreting sentences within a rich context -- for instance, information retrieval systems or dialogue systems -it is desirable to be able to preserve the sentence in a form that can be readily understood without context, for later reuse -- a process known as ``decontextualization''. While previous work demonstrated that generative Seq2Seq models could effectively perform decontextualization after being fine-tuned on a specific dataset, this approach requires expensive human annotations and may not transfer to other domains. We propose a few-shot method of decontextualization using a large language model, and present preliminary results showing that this method achieves viable performance on multiple domains using only a small set of examples.
    
[^20]: 我们是我们反复做的事情：在基于角色的回应中引导和应用习惯性模式

    We are what we repeatedly do: Inducing and deploying habitual schemas in persona-based responses. (arXiv:2310.06245v1 [cs.AI])

    [http://arxiv.org/abs/2310.06245](http://arxiv.org/abs/2310.06245)

    该论文研究了在对话系统中引导和应用习惯性模式来生成基于角色的回应。与以往的方法不同，该论文通过明确的模式表示来捕捉与角色相关的习惯性知识，从而提高了回应的准确性和自然性。

    

    对话技术的许多实际应用需要根据特定的开发者规定的角色生成回应。虽然可以从最近的大型语言模型中引出许多角色，但这些模型的不透明性和不可预测性使得以明确形式指定角色成为可取的。在以前的工作中，角色通常被表示为一次性的自我知识片段的集合，对话系统会检索这些知识片段来生成回应。然而，在现实的人类对话中，角色通常通过类似故事的叙述来揭示出来，这些叙述涉及丰富的习惯性知识——关于代理人经常参与的事件类型的知识（例如，工作活动、爱好、体育活动、喜欢的娱乐等），包括这些事件的典型目标、子事件、前提条件和后置条件。我们使用明确的模式表示来捕捉这种习惯性知识，并提出了一种引导对话系统使用这些模式生成回应的方法。

    Many practical applications of dialogue technology require the generation of responses according to a particular developer-specified persona. While a variety of personas can be elicited from recent large language models, the opaqueness and unpredictability of these models make it desirable to be able to specify personas in an explicit form. In previous work, personas have typically been represented as sets of one-off pieces of self-knowledge that are retrieved by the dialogue system for use in generation. However, in realistic human conversations, personas are often revealed through story-like narratives that involve rich habitual knowledge -- knowledge about kinds of events that an agent often participates in (e.g., work activities, hobbies, sporting activities, favorite entertainments, etc.), including typical goals, sub-events, preconditions, and postconditions of those events. We capture such habitual knowledge using an explicit schema representation, and propose an approach to dia
    
[^21]: 模型调优还是提示调优？对于临床概念和关系提取的大型语言模型的研究

    Model Tuning or Prompt Tuning? A Study of Large Language Models for Clinical Concept and Relation Extraction. (arXiv:2310.06239v1 [cs.CL])

    [http://arxiv.org/abs/2310.06239](http://arxiv.org/abs/2310.06239)

    本研究探索了大型语言模型在临床概念和关系提取任务中的应用，通过软提示调优取得了最佳性能，并观察了迁移学习和少样本学习的能力。

    

    本研究旨在开发基于软提示的大型语言模型（LLMs）学习算法，研究提示的形状、使用冻结/解冻LLMs进行提示调优、迁移学习和少样本学习能力。通过开发了基于软提示的LLM模型，并通过比较四种训练策略（1.无提示微调；2.解冻LLMs的硬提示；3.解冻LLMs的软提示；4.冻结LLMs的软提示）来评估了七个预训练的LLMs在两个基准数据集上的临床概念和关系提取能力。在跨机构环境中评估了基于提示学习算法的迁移学习能力，并评估了少样本学习能力。

    Objective To develop soft prompt-based learning algorithms for large language models (LLMs), examine the shape of prompts, prompt-tuning using frozen/unfrozen LLMs, transfer learning, and few-shot learning abilities. Methods We developed a soft prompt-based LLM model and compared 4 training strategies including (1) fine-tuning without prompts; (2) hard-prompt with unfrozen LLMs; (3) soft-prompt with unfrozen LLMs; and (4) soft-prompt with frozen LLMs. We evaluated 7 pretrained LLMs using the 4 training strategies for clinical concept and relation extraction on two benchmark datasets. We evaluated the transfer learning ability of the prompt-based learning algorithms in a cross-institution setting. We also assessed the few-shot learning ability. Results and Conclusion When LLMs are unfrozen, GatorTron-3.9B with soft prompting achieves the best strict F1-scores of 0.9118 and 0.8604 for concept extraction, outperforming the traditional fine-tuning and hard prompt-based models by 0.6~3.1% a
    
[^22]: 解决MUSIC-AVQA中的数据偏差问题：为无偏问答创建一个平衡的数据集

    Tackling Data Bias in MUSIC-AVQA: Crafting a Balanced Dataset for Unbiased Question-Answering. (arXiv:2310.06238v1 [cs.CV])

    [http://arxiv.org/abs/2310.06238](http://arxiv.org/abs/2310.06238)

    本论文解决了MUSIC-AVQA中的数据偏差问题，通过创建一个平衡的数据集来保证模型能够有效地推理各种多模态情况下的问题。他们构建了一个名为MUSIC-AVQA v2.0的新数据集，并提出了一个新的基准模型。

    

    近年来，对音频、视觉和文本模态的交叉研究越来越受重视，推动了多模态研究的进展。然而，任何模态中存在的强烈偏见会导致模型忽视其他模态。因此，模型有效地跨越这些多样化模态进行推理的能力受到损害，阻碍了进一步的发展。在本文中，我们详细审查了原始数据集中的每种问题类型，选择具有明显答案偏见的问题。为了解决这些偏见，我们收集了互补的视频和问题，确保没有答案有明显的偏斜分布。特别是对于二元问题，我们努力确保每个问题类别中两个答案几乎均匀分布。因此，我们构建了一个名为MUSIC-AVQA v2.0的新数据集，它更具挑战性，我们相信能够更好地促进AVQA任务的进展。此外，我们提出了一种新的基准模型。

    In recent years, there has been a growing emphasis on the intersection of audio, vision, and text modalities, driving forward the advancements in multimodal research. However, strong bias that exists in any modality can lead to the model neglecting the others. Consequently, the model's ability to effectively reason across these diverse modalities is compromised, impeding further advancement. In this paper, we meticulously review each question type from the original dataset, selecting those with pronounced answer biases. To counter these biases, we gather complementary videos and questions, ensuring that no answers have outstanding skewed distribution. In particular, for binary questions, we strive to ensure that both answers are almost uniformly spread within each question category. As a result, we construct a new dataset, named MUSIC-AVQA v2.0, which is more challenging and we believe could better foster the progress of AVQA task. Furthermore, we present a novel baseline model that de
    
[^23]: 自然语言处理技术的演进：不仅仅是语言处理，而是朝向通用人工智能

    Evolution of Natural Language Processing Technology: Not Just Language Processing Towards General Purpose AI. (arXiv:2310.06228v1 [cs.CL])

    [http://arxiv.org/abs/2310.06228](http://arxiv.org/abs/2310.06228)

    这篇论文翻译成中文并总结指出，自然语言处理技术逐渐演进为通用人工智能，尽管自然语言非常难以进行数学上的表达，但通过深度学习在NLP领域取得了超乎预期的结果。

    

    自发明计算机以来，通过自然语言（实际的人类语言）进行通信一直是一项梦幻的技术。然而，自然语言非常难以进行数学上的表达，这使得在不考虑编程的情况下实现算法变得困难。尽管已经有了许多技术发展，但到目前为止，还不能说已经实现了任何允许自由利用的结果。举例来说，在人类的语言学习中，例如学习母语或外语时，我们必须承认这个过程与格言“熟能生巧”在原则上是相似的，尽管学习方法在某种程度上是重要的。深度学习近年来在当代人工智能技术中扮演了核心角色。当应用于自然语言处理（NLP）时，这产生了前所未有的结果。从学习大量文本数据的结果中，已经有报道称超出了最初的预测。

    Since the invention of computers, communication through natural language (actual human language) has been a dream technology. However, natural language is extremely difficult to mathematically formulate, making it difficult to realize as an algorithm without considering programming. While there have been numerous technological developments, one cannot say that any results allowing free utilization have been achieved thus far. In the case of language learning in humans, for instance when learning one's mother tongue or foreign language, one must admit that this process is similar to the adage "practice makes perfect" in principle, even though the learning method is significant up to a point. Deep learning has played a central role in contemporary AI technology in recent years. When applied to natural language processing (NLP), this produced unprecedented results. Achievements exceeding the initial predictions have been reported from the results of learning vast amounts of textual data u
    
[^24]: GeoLLM: 从大型语言模型中提取地理空间知识

    GeoLLM: Extracting Geospatial Knowledge from Large Language Models. (arXiv:2310.06213v1 [cs.CL])

    [http://arxiv.org/abs/2310.06213](http://arxiv.org/abs/2310.06213)

    GeoLLM是一种能够从大型语言模型中提取地理空间知识的新方法，结合来自OpenStreetMap的辅助地图数据，可以有效应用于测量人口密度和经济生计等任务。

    

    机器学习在各种地理空间任务中的应用越来越普遍，但常常依赖于全球范围可用的卫星图像等预测变量，这可能要么很昂贵，要么缺乏预测能力。本文探讨了一个问题，即互联网语言语料库中包含的大量知识是否可以利用大型语言模型（LLM）进行地理空间预测任务。我们首先证明了LLM中嵌入了有关位置的显著空间信息，但仅使用地理坐标来查询LLM对于预测人口密度等关键指标是无效的。然后，我们提出了一种名为GeoLLM的新方法，可以有效地从LLM中提取地理空间知识，并结合来自OpenStreetMap的辅助地图数据。我们展示了我们的方法在多个国际社区关心的任务中的实用性，包括人口密度和经济生计的测量。在这些任务中

    The application of machine learning (ML) in a range of geospatial tasks is increasingly common but often relies on globally available covariates such as satellite imagery that can either be expensive or lack predictive power. Here we explore the question of whether the vast amounts of knowledge found in Internet language corpora, now compressed within large language models (LLMs), can be leveraged for geospatial prediction tasks. We first demonstrate that LLMs embed remarkable spatial information about locations, but naively querying LLMs using geographic coordinates alone is ineffective in predicting key indicators like population density. We then present GeoLLM, a novel method that can effectively extract geospatial knowledge from LLMs with auxiliary map data from OpenStreetMap. We demonstrate the utility of our approach across multiple tasks of central interest to the international community, including the measurement of population density and economic livelihoods. Across these task
    
[^25]: 无回归估计数字

    Estimating Numbers without Regression. (arXiv:2310.06204v1 [cs.CL])

    [http://arxiv.org/abs/2310.06204](http://arxiv.org/abs/2310.06204)

    提出了一种不使用回归的方法来估计数字，通过改变模型的词汇表来更好地表示数字的大小，并取得了令人满意的结果。

    

    尽管语言模型在最近取得了成功，但它们表示数字的能力仍然不足。人类根据数字的大小进行概念化，有效地将它们投射到数字线上；而子词标记化无法明确捕捉数字的大小，将数字分割成任意的块。为了克服这个缺点，已经提出了替代方法，在语言建模管线的各个阶段修改数字。这些方法要么改变数字的表示方式（例如科学计数法与十进制），要么改变用于表示数字的词汇表，要么直接改变基础语言模型的架构，以直接回归到所需的数字。以前的工作表明，架构的更改有助于在数字估计上达到最先进的水平，但我们发现一个有洞察力的缺憾：改变模型的词汇表（例如在10-100范围内引入一个新的数字标记）是一个更好的权衡。

    Despite recent successes in language models, their ability to represent numbers is insufficient. Humans conceptualize numbers based on their magnitudes, effectively projecting them on a number line; whereas subword tokenization fails to explicitly capture magnitude by splitting numbers into arbitrary chunks. To alleviate this shortcoming, alternative approaches have been proposed that modify numbers at various stages of the language modeling pipeline. These methods change either the (1) notation in which numbers are written (\eg scientific vs decimal), the (2) vocabulary used to represent numbers or the entire (3) architecture of the underlying language model, to directly regress to a desired number.  Previous work suggests that architectural change helps achieve state-of-the-art on number estimation but we find an insightful ablation: changing the model's vocabulary instead (\eg introduce a new token for numbers in range 10-100) is a far better trade-off. In the context of masked numb
    
[^26]: GPT-who：一种基于信息密度的机器生成文本检测器

    GPT-who: An Information Density-based Machine-Generated Text Detector. (arXiv:2310.06202v1 [cs.CL])

    [http://arxiv.org/abs/2310.06202](http://arxiv.org/abs/2310.06202)

    GPT-who是一种基于统一信息密度原则的机器生成文本检测器，利用基于统一信息密度原则的特征来建模每个语言模型和人类作者的独特统计特征，以实现准确的作者归属。在多个领域中，GPT-who的性能超过了其他最先进的检测器，且具有计算成本低廉和可解释性。

    

    统一信息密度原则认为人类在语言产生过程中喜欢平均分布信息。在这项工作中，我们研究了统一信息密度原则是否可以帮助捕捉大型语言模型（LLMs）和人类生成文本之间的差异。我们提出了GPT-who，这是第一个基于心理语言学的多类领域不可知统计检测器。该检测器利用基于统一信息密度原则的特征建模每个LLM和人类作者的独特统计特征，以实现准确的作者归属。我们使用4个大型基准数据集对我们的方法进行评估，并发现GPT-who在各个领域上的性能优于最先进的检测器（包括基于统计和非统计的），如GLTR，GPTZero，OpenAI detector和ZeroGPT超过20％。除了性能优越外，GPT-who计算成本低廉，并利用可解释的文本表示。我们展示了对人类和机器生成文本的基于统一信息密度的表示的最大分析。

    The Uniform Information Density principle posits that humans prefer to spread information evenly during language production. In this work, we examine if the UID principle can help capture differences between Large Language Models (LLMs) and human-generated text. We propose GPT-who, the first psycholinguistically-aware multi-class domain-agnostic statistical-based detector. This detector employs UID-based features to model the unique statistical signature of each LLM and human author for accurate authorship attribution. We evaluate our method using 4 large-scale benchmark datasets and find that GPT-who outperforms state-of-the-art detectors (both statistical- & non-statistical-based) such as GLTR, GPTZero, OpenAI detector, and ZeroGPT by over $20$% across domains. In addition to superior performance, it is computationally inexpensive and utilizes an interpretable representation of text articles. We present the largest analysis of the UID-based representations of human and machine-genera
    
[^27]: 压缩上下文以提高大型语言模型的推理效率

    Compressing Context to Enhance Inference Efficiency of Large Language Models. (arXiv:2310.06201v1 [cs.CL])

    [http://arxiv.org/abs/2310.06201](http://arxiv.org/abs/2310.06201)

    本研究提出了一种名为Selective Context的方法，通过识别和修剪输入上下文中的冗余信息，以降低大型语言模型的推理过程中的计算成本和延迟，并保持相当的性能。

    

    大型语言模型在各种任务中取得了显著的性能。然而，由于计算要求显著增加，包括内存和推理时间，以及输入超出语言模型固定上下文长度时潜在的上下文截断，它们在处理长文档和扩展对话时面临挑战。本文提出了一种称为Selective Context的方法，通过识别和修剪输入上下文中的冗余信息，使输入更紧凑，从而提高了语言模型的推理效率。我们在需要处理长上下文的常见数据源上进行了测试，包括arXiv论文、新闻文章和长对话，用于摘要、问答和回答生成任务。实验结果表明，Selective Context显著减少了内存成本，并降低了生成延迟，同时保持了与使用完整上下文时相当的性能。具体来说，我们实现了50%的内存成本减少。

    Large language models (LLMs) achieved remarkable performance across various tasks. However, they face challenges in managing long documents and extended conversations, due to significantly increased computational requirements, both in memory and inference time, and potential context truncation when the input exceeds the LLM's fixed context length. This paper proposes a method called Selective Context that enhances the inference efficiency of LLMs by identifying and pruning redundancy in the input context to make the input more compact. We test our approach using common data sources requiring long context processing: arXiv papers, news articles, and long conversations, on tasks of summarisation, question answering, and response generation. Experimental results show that Selective Context significantly reduces memory cost and decreases generation latency while maintaining comparable performance compared to that achieved when full context is used. Specifically, we achieve a 50\% reduction
    
[^28]: 自动神经元解释的及时调优的重要性

    The Importance of Prompt Tuning for Automated Neuron Explanations. (arXiv:2310.06200v1 [cs.CL])

    [http://arxiv.org/abs/2310.06200](http://arxiv.org/abs/2310.06200)

    本文探讨了在自动神经元解释中即时调优的重要性，通过重新格式化解释提示，我们显著提高了解释质量并降低了计算成本。这项研究对于更深入理解大型语言模型的工作原理和安全性具有重要意义。

    

    最近的进展极大地提升了大型语言模型(LLM)的能力，但我们对这些模型及其安全性的理解并没有同步进展。在本文中，我们旨在通过研究它们的个体神经元来更深入地理解LLM。我们在前人研究的基础上，进一步探讨了大型语言模型，如GPT-4，如何解释语言模型中每个神经元的功能。具体地，我们分析了生成解释所使用的提示的效果，并展示了以更自然的方式重新格式化解释提示如何显著提高神经元解释的质量，并大幅降低计算成本。我们通过三种不同的方式演示了我们新提示的效果，包括自动化评估和人工评估。

    Recent advances have greatly increased the capabilities of large language models (LLMs), but our understanding of the models and their safety has not progressed as fast. In this paper we aim to understand LLMs deeper by studying their individual neurons. We build upon previous work showing large language models such as GPT-4 can be useful in explaining what each neuron in a language model does. Specifically, we analyze the effect of the prompt used to generate explanations and show that reformatting the explanation prompt in a more natural way can significantly improve neuron explanation quality and greatly reduce computational cost. We demonstrate the effects of our new prompts in three different ways, incorporating both automated and human evaluations.
    
[^29]: CAW-coref: 关联词感知的词级共指消解

    CAW-coref: Conjunction-Aware Word-level Coreference Resolution. (arXiv:2310.06165v1 [cs.CL])

    [http://arxiv.org/abs/2310.06165](http://arxiv.org/abs/2310.06165)

    本文介绍了一种关联词感知的词级共指消解模型（CAW-coref），在处理并列提及的情况下表现出了较高的性能，有效地缩小了与昂贵的最先进方法的差距。

    

    当前最先进的共指消解系统每篇文章需要多次调用语言模型，因此对于许多应用场景来说（例如使用大规模语料库进行信息提取），代价太高。而词级共指系统 (WL-coref) 在效率上更加高效，实现了这些最先进系统 96.6% 的性能。本文发现了 WL-coref 的一个常见但重要的失败案例：处理“Tom 和 Mary”之类的并列提及。我们提供了一个简单但有效的解决方案，在 OntoNotes 测试集上将性能提高了 0.9% F1，将高效的词级共指消解与昂贵的最先进方法的差距缩小了34.6%。我们的关联词感知的词级共指模型（CAW-coref）和代码可在 https://github.com/KarelDO/wl-coref 获取。

    State-of-the-art coreference resolutions systems depend on multiple LLM calls per document and are thus prohibitively expensive for many use cases (e.g., information extraction with large corpora). The leading word-level coreference system (WL-coref) attains 96.6% of these SOTA systems' performance while being much more efficient. In this work, we identify a routine yet important failure case of WL-coref: dealing with conjoined mentions such as 'Tom and Mary'. We offer a simple yet effective solution that improves the performance on the OntoNotes test set by 0.9% F1, shrinking the gap between efficient word-level coreference resolution and expensive SOTA approaches by 34.6%. Our Conjunction-Aware Word-level coreference model (CAW-coref) and code is available at https://github.com/KarelDO/wl-coref.
    
[^30]: 退后一步：通过抽象唤起大型语言模型的推理能力

    Take a Step Back: Evoking Reasoning via Abstraction in Large Language Models. (arXiv:2310.06117v1 [cs.LG])

    [http://arxiv.org/abs/2310.06117](http://arxiv.org/abs/2310.06117)

    本文提出了一种简单的提示技术，使得大型语言模型能够通过抽象获得高层概念和基本原理，并将其应用于推理路径中，从而显著提升模型在各种推理密集型任务上的表现。

    

    我们提出了一种称为“退后提示”的简单提示技术，使得大型语言模型能够通过从包含具体细节的实例中进行抽象，得出高层概念和基本原理。利用这些概念和原理来指导推理步骤，语言模型在正确推理路径上显著提升了能力。我们使用PaLM-2L模型进行了退后提示实验，在包括STEM、知识问答和多跳推理在内的各种具有挑战性的推理密集型任务上观察到了明显的性能提升。例如，在MMLU物理和化学任务上，退后提示可以将PaLM-2L的性能提升7%和11%，在TimeQA任务上提升27%，在MuSiQue任务上提升7%。

    We present Step-Back Prompting, a simple prompting technique that enables LLMs to do abstractions to derive high-level concepts and first principles from instances containing specific details. Using the concepts and principles to guide the reasoning steps, LLMs significantly improve their abilities in following a correct reasoning path towards the solution. We conduct experiments of Step-Back Prompting with PaLM-2L models and observe substantial performance gains on a wide range of challenging reasoning-intensive tasks including STEM, Knowledge QA, and Multi-Hop Reasoning. For instance, Step-Back Prompting improves PaLM-2L performance on MMLU Physics and Chemistry by 7% and 11%, TimeQA by 27%, and MuSiQue by 7%.
    
[^31]: BYOC: 使用合著的类别描述个性化进行少样本分类

    BYOC: Personalized Few-Shot Classification with Co-Authored Class Descriptions. (arXiv:2310.06111v1 [cs.CL])

    [http://arxiv.org/abs/2310.06111](http://arxiv.org/abs/2310.06111)

    提出了一种个性化的少样本文本分类方法，使用合著的类别描述作为提示，用户与LLM交互合作进行标注，形成分类提示，实验结果显示其准确率达到了使用大规模数据集进行训练的模型的82%。

    

    文本分类是许多自然语言处理应用的重要组成部分，但现有方法要么需要大规模带标注的语料库进行模型训练，要么在使用大型语言模型作为基础时，需要精心设计提示并使用能容纳许多示例的长上下文。结果，普通用户不能为自己构建分类器。为了解决这个问题，我们提出了一种新颖的个性化少样本文本分类方法，使用LLM（大型语言模型）。LLM不是使用少样本示例，而是由用户和LLM合著的每个类别的显著特征的描述进行提示。在用户标注每个少样本示例时，LLM会提出相关的问题，用户给出答案。示例、问题和答案被总结成分类提示。实验证明，我们的方法可以获得高准确率的分类器，其性能达到使用大规模数据集进行训练的模型的82%。

    Text classification is a well-studied and versatile building block for many NLP applications. Yet, existing approaches require either large annotated corpora to train a model with or, when using large language models as a base, require carefully crafting the prompt as well as using a long context that can fit many examples. As a result, it is not possible for end-users to build classifiers for themselves. To address this issue, we propose a novel approach to few-shot text classification using an LLM. Rather than few-shot examples, the LLM is prompted with descriptions of the salient features of each class. These descriptions are coauthored by the user and the LLM interactively: while the user annotates each few-shot example, the LLM asks relevant questions that the user answers. Examples, questions, and answers are summarized to form the classification prompt. Our experiments show that our approach yields high accuracy classifiers, within 82% of the performance of models trained with s
    
[^32]: 利用多语言自监督预训练模型实现序列到序列的端到端口语理解

    Leveraging Multilingual Self-Supervised Pretrained Models for Sequence-to-Sequence End-to-End Spoken Language Understanding. (arXiv:2310.06103v1 [cs.CL])

    [http://arxiv.org/abs/2310.06103](http://arxiv.org/abs/2310.06103)

    这项工作提出了一种利用多语言自监督预训练模型的统一方法，用于多语言的序列到序列的端到端口语理解，成功地预测了词汇填充器，并在多个数据集上取得了优于其他方法的结果。

    

    已经提出了许多使用预训练模型的端到端口语理解（E2E-SLU）方法，但它们的评估通常缺乏多语言设置和需要预测词汇填充器（例如槽填充）等任务。在这项工作中，我们提出了一种统一的方法，集成了多语言预训练语音和文本模型，并以生成方式在四种语言的六个数据集上进行E2E-SLU，包括对词汇填充器的预测。我们研究了如何通过在广泛可用的语音识别数据上进行预训练来改进所提出的方法，使用了几种训练目标。在7000小时的多语言数据上进行预训练使我们能够在两个SLU数据集上最终超过最先进的方法，而在另外两个SLU数据集上则部分超过。最后，我们检查了所提出模型的跨语言能力，并将在PortMEDIA-Language数据集上的最佳已知结果提高了近一半，达到了23.65%的概念/值错误率。

    A number of methods have been proposed for End-to-End Spoken Language Understanding (E2E-SLU) using pretrained models, however their evaluation often lacks multilingual setup and tasks that require prediction of lexical fillers, such as slot filling. In this work, we propose a unified method that integrates multilingual pretrained speech and text models and performs E2E-SLU on six datasets in four languages in a generative manner, including the prediction of lexical fillers. We investigate how the proposed method can be improved by pretraining on widely available speech recognition data using several training objectives. Pretraining on 7000 hours of multilingual data allows us to outperform the state-of-the-art ultimately on two SLU datasets and partly on two more SLU datasets. Finally, we examine the cross-lingual capabilities of the proposed model and improve on the best known result on the PortMEDIA-Language dataset by almost half, achieving a Concept/Value Error Rate of 23.65%.
    
[^33]: 对文本数据中的性别分析器进行审计

    Auditing Gender Analyzers on Text Data. (arXiv:2310.06061v1 [cs.CY])

    [http://arxiv.org/abs/2310.06061](http://arxiv.org/abs/2310.06061)

    对三种现有的性别分析器进行了不对二元个体做预测的偏见的审计，发现这些工具在准确性上存在严重问题，并且对非二元评论的预测主要是女性，从而传播了对非二元人群的偏见。

    

    AI模型已经变得非常流行和可访问，然而，它们不断受到批评，因为它们对各个社会群体，如有色人种和非二元性别人群，存在可证明的偏见。在这项研究中，我们对三种已有的性别分析器 - uClassify，Readable和HackerFactor进行审计，以评估它们针对非二元个体存在的偏见。这些工具仅设计用于预测二元性别标签，因此对非二元性别社会成员存在歧视。我们整理了两个数据集-Reddit评论（660k）和Tumblr帖子（2.05M），我们的实验评估显示这些工具在所有平台上的总体准确率约为50%。所有平台上针对非二元评论的预测主要是女性，从而传播了社会对非二元人群的偏见，认为他们是女性化的。为了解决这个问题，我们在这两个数据集上以多种组合对BERT多标签分类器进行了微调实验，并观察到一个 o

    AI models have become extremely popular and accessible to the general public. However, they are continuously under the scanner due to their demonstrable biases toward various sections of the society like people of color and non-binary people. In this study, we audit three existing gender analyzers -uClassify, Readable and HackerFactor, for biases against non-binary individuals. These tools are designed to predict only the cisgender binary labels, which leads to discrimination against non-binary members of the society. We curate two datasets -- Reddit comments (660k) and, Tumblr posts (2.05M) and our experimental evaluation shows that the tools are highly inaccurate with the overall accuracy being ~50% on all platforms. Predictions for non-binary comments on all platforms are mostly female, thus propagating the societal bias that non-binary individuals are effeminate. To address this, we fine-tune a BERT multi-label classifier on the two datasets in multiple combinations, observe an o
    
[^34]: LLM用于SoC安全: 一种范式转变

    LLM for SoC Security: A Paradigm Shift. (arXiv:2310.06046v1 [cs.CR])

    [http://arxiv.org/abs/2310.06046](http://arxiv.org/abs/2310.06046)

    本论文研究利用大型语言模型（LLM）解决系统级芯片（SoC）设计中的安全性问题，通过将LLM集成到SoC安全验证流程中，开辟了更高效、可伸缩和适应的新方法。目标是确保越来越复杂的SoC的安全性。

    

    随着电子设备中系统级芯片（SoC）设计的普及和复杂性的增加，将安全性纳入SoC设计流程的任务面临着重大挑战。现有的安全解决方案由于在可伸缩性、全面性和适应性方面的局限性，无法提供对现代SoC设计的有效验证。另一方面，大型语言模型（LLM）因其在自然语言理解、高级推理和程序合成任务中的卓越成功而备受赞誉。认识到这一机遇，我们的研究着眼于利用生成式预训练转换器（GPT）的新兴能力来解决SoC安全领域的现有差距，旨在追求更高效、可伸缩和适应的方法论。通过将LLM集成到SoC安全验证范式中，我们打开了确保日益复杂SoC安全的新可能性和挑战。本文提供了深入分析。

    As the ubiquity and complexity of system-on-chip (SoC) designs increase across electronic devices, the task of incorporating security into an SoC design flow poses significant challenges. Existing security solutions are inadequate to provide effective verification of modern SoC designs due to their limitations in scalability, comprehensiveness, and adaptability. On the other hand, Large Language Models (LLMs) are celebrated for their remarkable success in natural language understanding, advanced reasoning, and program synthesis tasks. Recognizing an opportunity, our research delves into leveraging the emergent capabilities of Generative Pre-trained Transformers (GPTs) to address the existing gaps in SoC security, aiming for a more efficient, scalable, and adaptable methodology. By integrating LLMs into the SoC security verification paradigm, we open a new frontier of possibilities and challenges to ensure the security of increasingly complex SoCs. This paper offers an in-depth analysis
    
[^35]: 使用上下文线索和角色相关性提升文档级事件论证提取

    Enhancing Document-level Event Argument Extraction with Contextual Clues and Role Relevance. (arXiv:2310.05991v1 [cs.CL])

    [http://arxiv.org/abs/2310.05991](http://arxiv.org/abs/2310.05991)

    本文提出了一个SCPRG模型，通过引入Span-Trigger-based Contextual Pooling(STCP)和Role-based Latent Information Guidance (RLIG)模块，解决了文档级事件论证中忽略的非论证上下文线索信息以及论证角色相关性的问题。模型通过自适应地选择和汇聚上下文中的非论证线索词，以及构建潜在的角色表示并捕捉语义相关性，显著提升了文档级事件论证的准确性。

    

    与句子级事件论证相比，文档级事件论证面临着长输入和跨句子推理的新挑战。然而，大多数之前的工作都集中在捕捉每个事件中候选论证与事件触发器之间的关系，忽略了两个关键点：a）非论证的上下文线索信息；b）论证角色之间的相关性。在本文中，我们提出了一个SCPRG（基于跨度触发器的上下文汇聚和潜在角色引导）模型，它包含两个新颖而有效的模块来解决上述问题。基于跨度触发器的上下文汇聚（STCP）根据预训练模型的上下文注意力权重，自适应地选择和汇聚非论证线索词的信息。基于角色的潜在信息引导（RLIG）模块构建潜在的角色表示，通过角色交互编码使它们相互作用以捕捉语义相关性，并将它们合并到候选论证中。

    Document-level event argument extraction poses new challenges of long input and cross-sentence inference compared to its sentence-level counterpart. However, most prior works focus on capturing the relations between candidate arguments and the event trigger in each event, ignoring two crucial points: a) non-argument contextual clue information; b) the relevance among argument roles. In this paper, we propose a SCPRG (Span-trigger-based Contextual Pooling and latent Role Guidance) model, which contains two novel and effective modules for the above problem. The Span-Trigger-based Contextual Pooling(STCP) adaptively selects and aggregates the information of non-argument clue words based on the context attention weights of specific argument-trigger pairs from pre-trained model. The Role-based Latent Information Guidance (RLIG) module constructs latent role representations, makes them interact through role-interactive encoding to capture semantic relevance, and merges them into candidate ar
    
[^36]: 使用大规模语言模型建立合作行为相关个性特征的进化模型

    An evolutionary model of personality traits related to cooperative behavior using a large language model. (arXiv:2310.05976v1 [physics.soc-ph])

    [http://arxiv.org/abs/2310.05976](http://arxiv.org/abs/2310.05976)

    本文提出了一个使用大规模语言模型的进化模型，研究了个性特征在博弈理论关系背景下的演化。实验证明该模型能够展示出演化的特征。

    

    本文旨在通过将生成模型的丰富表达性引入到社会基于智能体的进化模型中，探讨多样性和社会群体的进化动态。具体来说，我们关注在博弈理论关系的背景下个性特征的演化，这是一个相互利益施加强烈选择压力的情况。我们构建了一个智能体模型，其中合作行为相关个性特征的语言描述被用作基因，从大规模语言模型（LLM）中提取的确定性策略根据这些个性特征做出行为决策被用作行为特征。根据平均收益的选择和通过请求LLM对父基因进行略微修改实现基因的突变，我们使群体进化。通过初步实验和分析，我们阐明了这样的模型确实可以展示演化的特征。

    This paper aims to shed light on the evolutionary dynamics of diverse and social populations by introducing the rich expressiveness of generative models into the trait expression of social agent-based evolutionary models. Specifically, we focus on the evolution of personality traits in the context of a game-theoretic relationship as a situation in which inter-individual interests exert strong selection pressures. We construct an agent model in which linguistic descriptions of personality traits related to cooperative behavior are used as genes. The deterministic strategies extracted from Large Language Model (LLM) that make behavioral decisions based on these personality traits are used as behavioral traits. The population is evolved according to selection based on average payoff and mutation of genes by asking LLM to slightly modify the parent gene toward cooperative or selfish. Through preliminary experiments and analyses, we clarify that such a model can indeed exhibit the evolution
    
[^37]: 探索用于衡量文本相关性的嵌入方法: 揭示在线评论中的情感和关系

    Exploring Embeddings for Measuring Text Relatedness: Unveiling Sentiments and Relationships in Online Comments. (arXiv:2310.05964v1 [cs.CL])

    [http://arxiv.org/abs/2310.05964](http://arxiv.org/abs/2310.05964)

    本论文通过使用嵌入方法和词语间的语义关系，研究了不同社交媒体平台上评论之间的情感和语义关系，并通过分析用户评论中的文本，让研究人员、政治家和商业代表能够追踪全球用户之间共享情感的路径。

    

    在一场导致互联网使用量增长70%的大流行病之后，全世界的人们开始更多地使用社交媒体。Twitter、Meta Threads、YouTube和Reddit等应用程序变得越来越普及，几乎没有任何数字空间不表达公众意见。本文研究了不同社交媒体平台上评论之间的情感和语义关系，并讨论了这些不同媒体平台上共享意见的重要性，使用词嵌入分析句子和文档中的组成部分。它使研究人员、政治家和商业代表能够追踪全球用户之间共享情感的路径。本研究论文提出了多种方法来衡量从这些热门在线平台的用户评论中提取的文本的相关性。通过利用捕捉词语间语义关系并有助于分析网络情感的嵌入方法，我们可以进行情感分析。

    After a pandemic that caused internet usage to grow by 70%, there has been an increased number of people all across the world using social media. Applications like Twitter, Meta Threads, YouTube, and Reddit have become increasingly pervasive, leaving almost no digital space where public opinion is not expressed. This paper investigates sentiment and semantic relationships among comments across various social media platforms, as well as discusses the importance of shared opinions across these different media platforms, using word embeddings to analyze components in sentences and documents. It allows researchers, politicians, and business representatives to trace a path of shared sentiment among users across the world. This research paper presents multiple approaches that measure the relatedness of text extracted from user comments on these popular online platforms. By leveraging embeddings, which capture semantic relationships between words and help analyze sentiments across the web, we
    
[^38]: 指纹攻击：联邦学习中的客户端去匿名化

    Fingerprint Attack: Client De-Anonymization in Federated Learning. (arXiv:2310.05960v1 [cs.CR])

    [http://arxiv.org/abs/2310.05960](http://arxiv.org/abs/2310.05960)

    本文研究了在联邦学习中，通过梯度指纹攻击可以轻松打破参与者匿名化，并展示了使用差分隐私进行训练可以提供实际防御。

    

    联邦学习允许在参与者不相信中央服务器和彼此的情况下进行协作训练，而无需共享数据。通过确保参与者与服务器之间的通信经过混洗，将参与者身份与其数据分离，还可以进一步提高隐私。本文旨在通过提出一种新颖的梯度指纹攻击来检验这种防御是否足以保证匿名性。我们在两个语言语料库上的联邦语言模型的实证研究中展示了梯度聚类可以轻松地打破匿名化。然后，我们展示了使用差分隐私进行训练可以提供对我们的指纹攻击的实际防御。

    Federated Learning allows collaborative training without data sharing in settings where participants do not trust the central server and one another. Privacy can be further improved by ensuring that communication between the participants and the server is anonymized through a shuffle; decoupling the participant identity from their data. This paper seeks to examine whether such a defense is adequate to guarantee anonymity, by proposing a novel fingerprinting attack over gradients sent by the participants to the server. We show that clustering of gradients can easily break the anonymization in an empirical study of learning federated language models on two language corpora. We then show that training with differential privacy can provide a practical defense against our fingerprint attack.
    
[^39]: 漏洞聚类与语义漏洞嵌入的其他机器学习应用

    Vulnerability Clustering and other Machine Learning Applications of Semantic Vulnerability Embeddings. (arXiv:2310.05935v1 [cs.CR])

    [http://arxiv.org/abs/2310.05935](http://arxiv.org/abs/2310.05935)

    本研究使用自然语言处理技术探索了语义漏洞嵌入的不同类型，并将其用于聚类、分类和可视化等机器学习应用，为计算机安全研究人员和分析师在风险评估等方面提供支持。

    

    计算机安全漏洞通常以简短的自然语言描述的形式发布（例如MITRE的CVE列表），随着时间的推移，这些描述会进一步通过常见漏洞评分系统（CVSS）等标签进行手动补充。在漏洞AI（分析与情报）项目中，我们研究了基于自然语言处理（NLP）技术的不同类型的语义漏洞嵌入，以获得漏洞空间的简洁表示。我们还评估了它们作为机器学习应用的基础，以支持计算机安全研究人员和分析师在风险评估和其他相关活动中的应用。我们在本报告中简要总结了我们探索的特定应用，包括聚类、分类和可视化，以及一种新的基于逻辑的评估漏洞空间理论的方法。

    Cyber-security vulnerabilities are usually published in form of short natural language descriptions (e.g., in form of MITRE's CVE list) that over time are further manually enriched with labels such as those defined by the Common Vulnerability Scoring System (CVSS). In the Vulnerability AI (Analytics and Intelligence) project, we investigated different types of semantic vulnerability embeddings based on natural language processing (NLP) techniques to obtain a concise representation of the vulnerability space. We also evaluated their use as a foundation for machine learning applications that can support cyber-security researchers and analysts in risk assessment and other related activities. The particular applications we explored and briefly summarize in this report are clustering, classification, and visualization, as well as a new logic-based approach to evaluate theories about the vulnerability space.
    
[^40]: NEFTune: 噪声嵌入改进指令微调

    NEFTune: Noisy Embeddings Improve Instruction Finetuning. (arXiv:2310.05914v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.05914](http://arxiv.org/abs/2310.05914)

    NEFTune 是一种改进语言模型微调效果的方法，通过在训练过程中给嵌入向量添加噪声，可以在多个指令数据集上显著提高准确率，并对不同类型的模型都有益处。

    

    我们展示了一个简单的增强方法可以显著提高语言模型微调的效果。NEFTune 在训练过程中给嵌入向量添加噪声。在使用 Alpaca 进行标准微调的基础上，LLaMA-2-7B 在 AlpacaEval 上的准确率从 29.79% 提升到了 64.69%。NEFTune 在现代指令数据集上也超过了强基线模型。使用 Evol-Instruct 训练的模型性能提升了10%，ShareGPT 提升了8%，OpenPlatypus 提升了8%。即使是经过 RLHF 进一步微调的强大模型，如 LLaMA-2-Chat，也可以通过 NEFTune 的附加训练获益。

    We show that language model finetuning can be improved, sometimes dramatically, with a simple augmentation. NEFTune adds noise to the embedding vectors during training. Standard finetuning of LLaMA-2-7B using Alpaca achieves 29.79% on AlpacaEval, which rises to 64.69% using noisy embeddings. NEFTune also improves over strong baselines on modern instruction datasets. Models trained with Evol-Instruct see a 10% improvement, with ShareGPT an 8% improvement, and with OpenPlatypus an 8% improvement. Even powerful models further refined with RLHF such as LLaMA-2-Chat benefit from additional training with NEFTune.
    
[^41]: 通过迭代地融合模态相似路径实现通用的多模态实体对齐

    Universal Multi-modal Entity Alignment via Iteratively Fusing Modality Similarity Paths. (arXiv:2310.05364v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.05364](http://arxiv.org/abs/2310.05364)

    本研究提出了一种通过迭代融合模态相似路径实现通用的多模态实体对齐方法。通过统一建模和有效信息融合，解决了现有方法中模态建模不一致和低效以及模态融合效果不佳的问题。

    

    实体对齐的目标是从多个知识图谱中确定等价的实体对，并创建一个更全面和统一的知识图谱。大多数实体对齐方法主要关注知识图谱的结构模态，缺乏对多模态信息的探索。少数多模态实体对齐方法在这个领域做出了不错的尝试。然而，它们存在两个缺点：(1)模态建模不一致且低效，为每个模态设计复杂和独立的模型；(2)由于实体对齐中模态的异构性，模态融合效果不佳。为了解决这些挑战，我们提出了PathFusion，它包括两个主要部分：(1) MSP，一个统一的建模方法，通过构建连接实体和模态节点以表示多个模态的路径，简化了对齐过程；(2) IRF，一种迭代融合方法，使用路径作为信息载体，有效地将不同模态的信息结合起来。

    The objective of Entity Alignment (EA) is to identify equivalent entity pairs from multiple Knowledge Graphs (KGs) and create a more comprehensive and unified KG. The majority of EA methods have primarily focused on the structural modality of KGs, lacking exploration of multi-modal information. A few multi-modal EA methods have made good attempts in this field. Still, they have two shortcomings: (1) inconsistent and inefficient modality modeling that designs complex and distinct models for each modality; (2) ineffective modality fusion due to the heterogeneous nature of modalities in EA. To tackle these challenges, we propose PathFusion, consisting of two main components: (1) MSP, a unified modeling approach that simplifies the alignment process by constructing paths connecting entities and modality nodes to represent multiple modalities; (2) IRF, an iterative fusion method that effectively combines information from different modalities using the path as an information carrier. Experim
    
[^42]: 在心理健康领域中通过任务自适应分词来增强长文本生成

    Enhancing Long-form Text Generation in Mental Health with Task-adaptive Tokenization. (arXiv:2310.05317v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.05317](http://arxiv.org/abs/2310.05317)

    该论文提出了一种任务自适应分词的方法，通过优化分词过程来增强在心理健康领域中的长文本生成。实验证明，该方法在减少标记数量的情况下显著提高了生成性能，并且可与大型语言模型结合使用。

    

    我们提出了任务自适应分词作为一种方式，将生成流水线适应于下游任务的特定要求，并增强在心理健康领域的长文本生成。受认知科学的启发，我们的任务自适应分词器从多个结果中采样可变的分段，采样概率基于任务特定的数据进行优化。我们引入了一种构建专用词汇的策略，并介绍了一种词汇合并协议，可以将任务特定的标记整合到预训练模型的分词步骤中。通过对中英文心理问答任务进行广泛实验，我们发现我们的任务自适应分词方法在使用更少的标记的情况下带来了显著的生成性能提升，最高可达60%。初步实验表明，使用我们的分词方法与非常大的语言模型结合能够得到有希望的结果。

    We propose task-adaptive tokenization as a way to adapt the generation pipeline to the specifics of a downstream task and enhance long-form generation in mental health. Inspired by insights from cognitive science, our task-adaptive tokenizer samples variable segmentations from multiple outcomes, with sampling probabilities optimized based on task-specific data. We introduce a strategy for building a specialized vocabulary and introduce a vocabulary merging protocol that allows for the integration of task-specific tokens into the pre-trained model's tokenization step. Through extensive experiments on psychological question-answering tasks in both Chinese and English, we find that our task-adaptive tokenization approach brings a significant improvement in generation performance while using up to 60% fewer tokens. Preliminary experiments point to promising results when using our tokenization approach with very large language models.
    
[^43]: ChatRadio-Valuer: 一种基于多机构和多系统数据的通用放射学报告自动生成的聊天大型语言模型

    ChatRadio-Valuer: A Chat Large Language Model for Generalizable Radiology Report Generation Based on Multi-institution and Multi-system Data. (arXiv:2310.05242v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.05242](http://arxiv.org/abs/2310.05242)

    ChatRadio-Valuer是一种基于大型语言模型的定制模型，用于通用放射学报告自动生成。它能够学习可泛化的表示并为模型自适应提供基础模式，解决了放射学报告中的泛化能力和异质性问题。

    

    放射学报告生成是医学图像分析的关键步骤，对临床决策水平的定量分析至关重要。然而，具有交叉来源异质性的复杂多样的放射学报告对当前方法在大规模数据量下的泛化能力提出了巨大挑战，主要是因为放射学报告的风格和规范性在机构、检查部位和放射科医生之间明显有所不同。最近，大型语言模型（LLM）的出现为识别健康状况的迹象提供了巨大的潜力。为了解决上述问题，我们与中国的湘雅二医院合作，提出了基于LLM的ChatRadio-Valuer，这是一种专门用于自动化放射学报告生成的定制模型，它可以学习可泛化的表示并为复杂分析师案例中的模型自适应提供基本模式。

    Radiology report generation, as a key step in medical image analysis, is critical to the quantitative analysis of clinically informed decision-making levels. However, complex and diverse radiology reports with cross-source heterogeneity pose a huge generalizability challenge to the current methods under massive data volume, mainly because the style and normativity of radiology reports are obviously distinctive among institutions, body regions inspected and radiologists. Recently, the advent of large language models (LLM) offers great potential for recognizing signs of health conditions. To resolve the above problem, we collaborate with the Second Xiangya Hospital in China and propose ChatRadio-Valuer based on the LLM, a tailored model for automatic radiology report generation that learns generalizable representations and provides a basis pattern for model adaptation in sophisticated analysts' cases. Specifically, ChatRadio-Valuer is trained based on the radiology reports from a single 
    
[^44]: TILFA: 一种用于论证挖掘中文本、图像和布局融合的统一框架

    TILFA: A Unified Framework for Text, Image, and Layout Fusion in Argument Mining. (arXiv:2310.05210v1 [cs.AI] CROSS LISTED)

    [http://arxiv.org/abs/2310.05210](http://arxiv.org/abs/2310.05210)

    TILFA是一个用于论证挖掘中处理文本、图像和布局混合数据的统一框架，不仅能够理解文本，还能够检测光学字符和识别图像中的布局细节，并在辩论立场分类中取得了显著的性能提升。

    

    论证挖掘的主要目标是分析作者的立场。与以往只关注文本的论证挖掘数据集不同，第10届论证挖掘研讨会的共享任务引入了一个包含文本和图像的数据集。重要的是，这些图像包含了视觉元素和光学字符。我们的新框架TILFA（一种用于论证挖掘中文本、图像和布局融合的统一框架）被设计用于处理这种混合数据。它不仅能够理解文本，还能够检测光学字符和识别图像中的布局细节。我们的模型明显优于现有的基准线，在这个共享任务的辩论立场分类子任务中，我们的团队KnowComp在排行榜上获得了第一名。

    A main goal of Argument Mining (AM) is to analyze an author's stance. Unlike previous AM datasets focusing only on text, the shared task at the 10th Workshop on Argument Mining introduces a dataset including both text and images. Importantly, these images contain both visual elements and optical characters. Our new framework, TILFA (A Unified Framework for Text, Image, and Layout Fusion in Argument Mining), is designed to handle this mixed data. It excels at not only understanding text but also detecting optical characters and recognizing layout details in images. Our model significantly outperforms existing baselines, earning our team, KnowComp, the 1st place in the leaderboard of Argumentative Stance Classification subtask in this shared task.
    
[^45]: 大型语言模型时代中的事实性挑战

    Factuality Challenges in the Era of Large Language Models. (arXiv:2310.05189v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.05189](http://arxiv.org/abs/2310.05189)

    大型语言模型存在生成虚假、错误或误导内容的问题，同时也面临被恶意应用的风险。本研究探讨了需要从事实核查员、新闻机构和研究与政策界采取的技术创新、监管改革和人工智能素养倡议，以解决这些问题。

    

    基于大型语言模型（LLMs）的工具的出现，如OpenAI的ChatGPT，微软的Bing Chat和谷歌的Bard，已经引起了巨大的公众关注。这些非常有用、自然的工具标志着自然语言生成方面的重大进展，然而它们存在生成虚假、错误或误导内容的倾向，通常被称为“幻觉”。此外，LLMs可以被用于恶意应用，例如在规模上生成虚假但可信的内容和个人资料。这对于社会来说构成了重大挑战，因为它可能欺骗用户并越来越多地传播不准确的信息。鉴于这些风险，我们探讨了事实核查员、新闻机构和更广泛的研究和政策界需要的技术创新、监管改革和人工智能素养倡议的类型。通过确定风险、迫在眉睫的威胁和一些可行的解决方案，我们希望解决这个问题。

    The emergence of tools based on Large Language Models (LLMs), such as OpenAI's ChatGPT, Microsoft's Bing Chat, and Google's Bard, has garnered immense public attention. These incredibly useful, natural-sounding tools mark significant advances in natural language generation, yet they exhibit a propensity to generate false, erroneous, or misleading content -- commonly referred to as "hallucinations." Moreover, LLMs can be exploited for malicious applications, such as generating false but credible-sounding content and profiles at scale. This poses a significant challenge to society in terms of the potential deception of users and the increasing dissemination of inaccurate information. In light of these risks, we explore the kinds of technological innovations, regulatory reforms, and AI literacy initiatives needed from fact-checkers, news organizations, and the broader research and policy communities. By identifying the risks, the imminent threats, and some viable solutions, we seek to she
    
[^46]: DialCoT遇到了PPO：在较小的语言模型中分解和探索推理路径

    DialCoT Meets PPO: Decomposing and Exploring Reasoning Paths in Smaller Language Models. (arXiv:2310.05074v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.05074](http://arxiv.org/abs/2310.05074)

    DialCoT是一种对话引导的链式思维方法，用于在较小的语言模型中分解和探索推理路径。通过将复杂问题分解为简单的子问题，它降低了任务难度，并使用PPO算法优化模型的推理路径选择。

    

    链式思维（CoT）提示已经被证明有助于增强至少具有1000亿参数的大型语言模型（LLMs）的推理能力。然而，当应用于具有不到100亿参数的较小语言模型（SLMs）的推理任务时，它是无效甚至有害的。为了解决这个限制，我们引入了对话引导的链式思维（DialCoT），它采用对话格式生成中间推理步骤，引导模型朝着最终答案前进。此外，我们使用PPO算法优化模型的推理路径选择，进一步增强其推理能力。我们的方法与以前的方法相比具有几个优点。首先，我们通过将解决复杂推理问题的过程分解成一系列更简单的子问题，显著降低了任务的难度，使其更适合于较小的语言模型。其次，我们优化了模型的推理路径选择，使其更准确和高效。

    Chain-of-Thought (CoT) prompting has proven to be effective in enhancing the reasoning capabilities of Large Language Models (LLMs) with at least 100 billion parameters. However, it is ineffective or even detrimental when applied to reasoning tasks in Smaller Language Models (SLMs) with less than 10 billion parameters. To address this limitation, we introduce Dialogue-guided Chain-of-Thought (DialCoT) which employs a dialogue format to generate intermediate reasoning steps, guiding the model toward the final answer. Additionally, we optimize the model's reasoning path selection using the Proximal Policy Optimization (PPO) algorithm, further enhancing its reasoning capabilities. Our method offers several advantages compared to previous approaches. Firstly, we transform the process of solving complex reasoning questions by breaking them down into a series of simpler sub-questions, significantly reducing the task difficulty and making it more suitable for SLMs. Secondly, we optimize the m
    
[^47]: 从文本到策略：评估在Avalon游戏中发挥作用的LLMs

    From Text to Tactic: Evaluating LLMs Playing the Game of Avalon. (arXiv:2310.05036v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2310.05036](http://arxiv.org/abs/2310.05036)

    本文研究了在Avalon游戏中使用LLMs的潜力，并引入了AvalonBench来评估多代理LLM代理。实验证明存在明显的能力差距。

    

    本文探讨了大型语言模型（LLM）在玩策略社交推理游戏Resistance Avalon中的潜力。Avalon玩家不仅需要根据动态发展的游戏阶段做出明智的决策，还需要参与讨论，在讨论中必须欺骗、推理和与其他玩家进行谈判。这些特点使得Avalon成为研究LLM代理的决策和语言处理能力的有趣试验平台。为了推动这一研究领域的发展，我们引入了AvalonBench——一个专门用于评估多代理LLM代理的全面游戏环境。该基准测试包括：（1）Avalon的游戏环境，（2）基于规则的机器人作为基准对手，以及（3）针对每个角色具有定制提示的ReAct-style LLM代理。值得注意的是，我们基于AvalonBench的评估突出显示了明显的能力差距。例如，像ChatGPT这样在好角色中的模型对战基于规则的机器人的胜率为22.2%。

    In this paper, we explore the potential of Large Language Models (LLMs) Agents in playing the strategic social deduction game, Resistance Avalon. Players in Avalon are challenged not only to make informed decisions based on dynamically evolving game phases, but also to engage in discussions where they must deceive, deduce, and negotiate with other players. These characteristics make Avalon a compelling test-bed to study the decision-making and language-processing capabilities of LLM Agents. To facilitate research in this line, we introduce AvalonBench - a comprehensive game environment tailored for evaluating multi-agent LLM Agents. This benchmark incorporates: (1) a game environment for Avalon, (2) rule-based bots as baseline opponents, and (3) ReAct-style LLM agents with tailored prompts for each role. Notably, our evaluations based on AvalonBench highlight a clear capability gap. For instance, models like ChatGPT playing good-role got a win rate of 22.2% against rule-based bots play
    
[^48]: 自我验证提示：利用重复内省进行少样本问题回答

    Self-Convinced Prompting: Few-Shot Question Answering with Repeated Introspection. (arXiv:2310.05035v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.05035](http://arxiv.org/abs/2310.05035)

    该论文提出了一种利用大规模预训练语言模型的框架，通过迭代反思和改进推理过程，以提升模型在少样本问题回答上的性能。

    

    虽然像ChatGPT和PaLM这样的大型语言模型在各种语言理解和生成任务中表现出色，但它们在复杂推理和繁琐知识利用方面仍然不及人类的熟练程度。最近的研究已经证明了提示在引导语言模型生成期望的输出方面的有效性。在这些见解的基础上，我们引入了一种新的框架，利用大规模预训练语言模型的潜力，以迭代的方式增强语言模型的性能。我们的框架包含三个组件：\textit{Normal CoT}、\textit{Convincer}和\textit{Answerer}。它处理typical few-shot chain-of-thought prompt的输出，评估回答的正确性，审查答案，改进推理，最终产生一个新的解决方案。在七个各种问题的数据集上的实验结果验证了自我验证框架的有效性。

    While large language models (LLMs) such as ChatGPT and PaLM have demonstrated remarkable performance in various language understanding and generation tasks, their capabilities in complex reasoning and intricate knowledge utilization still fall short of human-level proficiency. Recent studies have established the effectiveness of prompts in steering LLMs towards generating desired outputs. Building on these insights, we introduce a novel framework that harnesses the potential of large-scale pre-trained language models, to iteratively enhance performance of the LLMs. Our framework incorporates three components: \textit{Normal CoT}, a \textit{Convincer}, and an \textit{Answerer}. It processes the output of a typical few-shot chain-of-thought prompt, assesses the correctness of the response, scrutinizes the answer, refines the reasoning, and ultimately produces a new solution. Experimental results on the 7 datasets of miscellaneous problems validate the efficacy of the Self-Convince framew
    
[^49]: 重新审视大型语言模型作为零-shot关系抽取器

    Revisiting Large Language Models as Zero-shot Relation Extractors. (arXiv:2310.05028v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2310.05028](http://arxiv.org/abs/2310.05028)

    本研究重新审视了大型语言模型(LLMs)作为零-shot关系抽取器的潜力，并提出了通过总结和提问(\textsc{SumAsk})提示方法来改进零-shot关系抽取。实验证明LLMs在这一任务上具有良好的表现。

    

    关系抽取(RE)即使在零-shot设定下，一直涉及一定程度的标记或未标记的数据。最近的研究表明，大型语言模型(LLMs)能够在给定自然语言提示的情况下，无需任何数据和参数调整，自动适应新任务，这为从文本中提取关系提供了可能性。本研究主要关注将LLMs，如ChatGPT，作为零-shot关系抽取器的研究。一方面，我们分析了现有RE提示的缺点，并尝试将最近的提示技术，如CoT，纳入其中以提高零-shot关系抽取。我们提出了总结和提问(\textsc{SumAsk})提示，这是一种简单的提示，通过递归使用LLMs将RE输入转换为有效的问答(QA)格式。另一方面，我们对各种基准和设置进行了全面的实验，以调查LLMs在零-shot关系抽取上的能力。具体而言，我们有以下的followi

    Relation extraction (RE) consistently involves a certain degree of labeled or unlabeled data even if under zero-shot setting. Recent studies have shown that large language models (LLMs) transfer well to new tasks out-of-the-box simply given a natural language prompt, which provides the possibility of extracting relations from text without any data and parameter tuning. This work focuses on the study of exploring LLMs, such as ChatGPT, as zero-shot relation extractors. On the one hand, we analyze the drawbacks of existing RE prompts and attempt to incorporate recent prompt techniques such as chain-of-thought (CoT) to improve zero-shot RE. We propose the summarize-and-ask (\textsc{SumAsk}) prompting, a simple prompt recursively using LLMs to transform RE inputs to the effective question answering (QA) format. On the other hand, we conduct comprehensive experiments on various benchmarks and settings to investigate the capabilities of LLMs on zero-shot RE. Specifically, we have the followi
    
[^50]: DORIS-MAE: 使用多层级基于方面的查询进行科学文档检索

    DORIS-MAE: Scientific Document Retrieval using Multi-level Aspect-based Queries. (arXiv:2310.04678v2 [cs.IR] UPDATED)

    [http://arxiv.org/abs/2310.04678](http://arxiv.org/abs/2310.04678)

    DORIS-MAE是一个用于科学文档检索的新任务，旨在处理复杂的多方面查询。研究团队构建了一个基准数据集，并提出了一个基于大型语言模型的验证框架。

    

    在科学研究中，根据复杂的多方面查询有效检索相关文档的能力至关重要。现有的评估数据集受限于注释复杂查询所需的高成本和努力。为了解决这个问题，我们提出了一种新颖的任务，科学文档检索中使用多层级基于方面的查询(DORIS-MAE)，旨在处理科学研究中用户查询的复杂性。我们在计算机科学领域内建立了一个基准数据集，包含100个人工编写的复杂查询案例。对于每个复杂查询，我们组织了100个相关文档集合，并为其排名产生了注释的相关性分数。鉴于专家注释的巨大工作量，我们还引入了Anno-GPT，这是一个可扩展的框架，用于验证大型语言模型（LLMs）在专家级数据集注释任务上的性能。

    In scientific research, the ability to effectively retrieve relevant documents based on complex, multifaceted queries is critical. Existing evaluation datasets for this task are limited, primarily due to the high cost and effort required to annotate resources that effectively represent complex queries. To address this, we propose a novel task, Scientific DOcument Retrieval using Multi-level Aspect-based quEries (DORIS-MAE), which is designed to handle the complex nature of user queries in scientific research. We developed a benchmark dataset within the field of computer science, consisting of 100 human-authored complex query cases. For each complex query, we assembled a collection of 100 relevant documents and produced annotated relevance scores for ranking them. Recognizing the significant labor of expert annotation, we also introduce Anno-GPT, a scalable framework for validating the performance of Large Language Models (LLMs) on expert-level dataset annotation tasks. LLM annotation o
    
[^51]: Ada-Instruct: 为复杂推理调整指令生成器

    Ada-Instruct: Adapting Instruction Generators for Complex Reasoning. (arXiv:2310.04484v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.04484](http://arxiv.org/abs/2310.04484)

    Ada-Instruct是一种自适应指令生成器，通过对开源LLMs进行微调，能够生成复杂推理任务中长度大于等于100的指令。在代码补全、数学推理和常识推理等任务中，Ada-Instruct显示出优于基本模型和当前自我指导方法的改进效果。

    

    通过大型语言模型（LLM）生成多样且复杂的指令对于推进效果至关重要。当前的方法利用闭源的LLMs，通过上下文提示进行指令生成。然而，本文发现对于诸如代码补全等任务，上下文提示无法生成长度大于等于100的复杂指令。为解决这个问题，我们引入Ada-Instruct，一种通过对开源LLMs进行微调的自适应指令生成器。我们的关键发现表明，仅使用十个样本对开源LLMs进行微调即可生成保持分布一致性的长指令，适用于复杂推理任务。我们在代码补全、数学推理和常识推理等不同应用中对Ada-Instruct的有效性进行了实证验证。结果显示Ada-Instruct优于其基本模型和当前的自我指导方法。

    Generating diverse and sophisticated instructions for downstream tasks by Large Language Models (LLMs) is pivotal for advancing the effect. Current approaches leverage closed-source LLMs, employing in-context prompting for instruction generation. However, in this paper, we found that in-context prompting cannot generate complex instructions with length $\ge 100$ for tasks like code completion.  To solve this problem, we introduce Ada-Instruct, an adaptive instruction generator developed by fine-tuning open-source LLMs. Our pivotal finding illustrates that fine-tuning open-source LLMs with a mere ten samples generates long instructions that maintain distributional consistency for complex reasoning tasks. We empirically validated Ada-Instruct's efficacy across different applications, including code completion, mathematical reasoning, and commonsense reasoning. The results underscore Ada-Instruct's superiority, evidencing its improvements over its base models, current self-instruct method
    
[^52]: 自动调查挑战。（arXiv:2310.04480v2 [cs.CL]已更新）

    Auto-survey Challenge. (arXiv:2310.04480v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.04480](http://arxiv.org/abs/2310.04480)

    这项研究提出了一个新的平台，用于评估大型语言模型在撰写和评论调查论文方面的能力，并组织了一次竞赛来测试该平台。评估标准包括清晰度、参考适当性、可追溯性和内容的实质价值。

    

    我们提出了一个新颖的平台，用于评估大型语言模型（LLMs）在各个学科领域，包括科学、人文、教育和法律中，自主撰写和评论调查论文的能力。在这个框架内，人工智能系统进行类似于传统学术期刊的模拟同行评审机制，而人类组织者则充当编辑监督角色。在这个框架内，我们组织了一次针对2023年AutoML会议的竞赛。参赛者的任务是呈现能够根据指定提示撰写文章并进行评估的独立模型。评估标准包括清晰度、参考适当性、可追溯性和内容的实质价值。本文介绍了竞赛的设计，包括实施基准提交和评估方法。

    We present a novel platform for evaluating the capability of Large Language Models (LLMs) to autonomously compose and critique survey papers spanning a vast array of disciplines including sciences, humanities, education, and law. Within this framework, AI systems undertake a simulated peer-review mechanism akin to traditional scholarly journals, with human organizers serving in an editorial oversight capacity. Within this framework, we organized a competition for the AutoML conference 2023. Entrants are tasked with presenting stand-alone models adept at authoring articles from designated prompts and subsequently appraising them. Assessment criteria include clarity, reference appropriateness, accountability, and the substantive value of the content. This paper presents the design of the competition, including the implementation baseline submissions and methods of evaluation.
    
[^53]: 魔法词是什么？LLM提示的控制理论研究

    What's the Magic Word? A Control Theory of LLM Prompting. (arXiv:2310.04444v1 [cs.CL])

    [http://arxiv.org/abs/2310.04444](http://arxiv.org/abs/2310.04444)

    本论文将提示工程形式化为LLM上的最优控制问题，研究了给定token序列时是否存在一种最优提示能够准确预测最终的token，并提出了控制理论中的指标来描述LLM的可操纵性。

    

    提示工程在LLM的部署中是有效和重要的，但在数学上理解不足。在这里，我们将提示工程形式化为LLM上的最优控制问题，其中提示被认为是调节LLM输出分布的控制变量。在这个框架内，我们提出一个简单的问题：给定一个token序列，是否总存在一个我们可以添加的提示，使得LLM能够准确预测最终的token？我们将这样的最优提示称为魔法词，因为添加提示会导致LLM输出正确的答案。如果存在魔法词，我们能否找到它们？如果可以，它们的特性是什么？我们提供了将控制理论应用于自注意力头的分析分析，证明了其权重矩阵的奇异值函数为可控制性的上界。我们借鉴控制理论来提出了一种叫做$k-\epsilon$可控制性的指标，用于描述LLM的可操纵性。

    Prompt engineering is effective and important in the deployment of LLMs but is poorly understood mathematically. Here, we formalize prompt engineering as an optimal control problem on LLMs -- where the prompt is considered a control variable for modulating the output distribution of the LLM. Within this framework, we ask a simple question: given a sequence of tokens, does there always exist a prompt we can prepend that will steer the LLM toward accurately predicting the final token? We call such an optimal prompt the magic word since prepending the prompt causes the LLM to output the correct answer. If magic words exist, can we find them? If so, what are their properties? We offer analytic analysis on the controllability of the self-attention head where we prove a bound on controllability as a function of the singular values of its weight matrices. We take inspiration from control theory to propose a metric called $k-\epsilon$ controllability to characterize LLM steerability. We comput
    
[^54]: 大型语言模型在生物医学文本处理任务中的综合评估

    A Comprehensive Evaluation of Large Language Models on Benchmark Biomedical Text Processing Tasks. (arXiv:2310.04270v1 [cs.CL])

    [http://arxiv.org/abs/2310.04270](http://arxiv.org/abs/2310.04270)

    本文对大型语言模型（LLM）在生物医学任务中的性能进行了综合评估，发现零样本LLMs在小样本生物医学数据集上的表现甚至超过了先进的精调生物医学模型，预训练使LLMs在生物医学领域具备了很强的专业能力。

    

    最近，大型语言模型（LLM）展示了解决各种任务的出色能力。然而，尽管它们在各种任务中取得了成功，但目前还没有研究它们在生物医学领域的能力。因此，本文旨在评估LLMs在基准生物医学任务上的性能。为此，我们对6个不同生物医学任务的26个数据集中的4个热门LLMs进行了综合评估。据我们所知，这是第一篇在生物医学领域对各种LLMs进行广泛评估和比较的研究。有趣的是，根据我们的评估，我们发现在训练集较小的生物医学数据集中，零样本LLMs甚至超过了当前最先进的精调生物医学模型。这表明在大型文本语料库上进行预训练使LLMs在生物医学领域具备了很强的专业能力。我们还发现，在所有任务中没有一个LLM能够胜过其他LLMs。

    Recently, Large Language Models (LLM) have demonstrated impressive capability to solve a wide range of tasks. However, despite their success across various tasks, no prior work has investigated their capability in the biomedical domain yet. To this end, this paper aims to evaluate the performance of LLMs on benchmark biomedical tasks. For this purpose, we conduct a comprehensive evaluation of 4 popular LLMs in 6 diverse biomedical tasks across 26 datasets. To the best of our knowledge, this is the first work that conducts an extensive evaluation and comparison of various LLMs in the biomedical domain. Interestingly, we find based on our evaluation that in biomedical datasets that have smaller training sets, zero-shot LLMs even outperform the current state-of-the-art fine-tuned biomedical models. This suggests that pretraining on large text corpora makes LLMs quite specialized even in the biomedical domain. We also find that not a single LLM can outperform other LLMs in all tasks, with 
    
[^55]: 自然语言推理链条用于减少大型语言模型无根幻觉

    Chain of Natural Language Inference for Reducing Large Language Model Ungrounded Hallucinations. (arXiv:2310.03951v1 [cs.CL])

    [http://arxiv.org/abs/2310.03951](http://arxiv.org/abs/2310.03951)

    这项研究提出了一个分层框架，通过自然语言推理链条（CoNLI）来检测和减少大型语言模型（LLMs）的幻觉。该框架不需要对LLMs进行微调或特定领域的提示工程，能够从不同的上下文中实现具有竞争性能的幻觉检测和减少。

    

    当给定相关文档作为背景上下文时，大型语言模型（LLMs）能够生成流利的自然语言文本。这种能力引起了人们对LLMs在工业应用中的广泛关注。然而，LLMs容易产生没有提供来源支持的幻觉。在本文中，我们提出了一个分层框架来检测和减少这种无根幻觉。我们的框架使用自然语言推理链条（CoNLI）进行幻觉检测，并通过后期编辑进行幻觉减少。我们的方法在幻觉检测方面取得了最先进的性能，并且通过重写增强文本质量，使用LLMs而无需进行任何微调或特定领域的提示工程。我们展示了这个简单的即插即用框架可以作为幻觉检测和减少的有效选择，在各种情境下实现了竞争性能。

    Large language models (LLMs) can generate fluent natural language texts when given relevant documents as background context. This ability has attracted considerable interest in developing industry applications of LLMs. However, LLMs are prone to generate hallucinations that are not supported by the provided sources. In this paper, we propose a hierarchical framework to detect and mitigate such ungrounded hallucination. Our framework uses Chain of Natural Language Inference (CoNLI) for hallucination detection and hallucination reduction via post-editing. Our approach achieves state-of-the-art performance on hallucination detection and enhances text quality through rewrite, using LLMs without any fine-tuning or domain-specific prompt engineering. We show that this simple plug-and-play framework can serve as an effective choice for hallucination detection and reduction, achieving competitive performance across various contexts.
    
[^56]: 学习个性化故事评估

    Learning Personalized Story Evaluation. (arXiv:2310.03304v1 [cs.CL])

    [http://arxiv.org/abs/2310.03304](http://arxiv.org/abs/2310.03304)

    该论文提出了学习个性化故事评估的方法。为了解决大型语言模型在开放式文本生成任务的评估问题，论文创建了两个新的数据集，并开发了一个个性化故事评估模型，能够根据评审人员的示例评价进行个性化评估。

    

    尽管大型语言模型（LLM）在诸如问答和检索等更客观的任务上显示出令人印象深刻的结果，但评估它们在开放式文本生成方面的表现仍然是一个困难的问题，原因包括（1）数据污染；（2）多维评估标准；以及（3）来自评审人员个人偏好的主观性。为了解决这些问题，我们提出在一个无污染的开放式生成评估中建模个性化。我们使用适当的匿名化和新的个性化标签，重新利用现有数据集创建了两个新的数据集Per-MPST和Per-DOC用于个性化故事评估。我们进一步开发了一个个性化故事评估模型PERSE来推测评审人员的偏好，并提供个性化评估。具体而言，对于某个评审人员的一些示例评价，PERSE可以预测该评审人员在新的情节上的详细评审或细粒度比较（如趣味性和惊喜）。

    While large language models (LLMs) have shown impressive results for more objective tasks such as QA and retrieval, it remains nontrivial to evaluate their performance on open-ended text generation for reasons including (1) data contamination; (2) multi-dimensional evaluation criteria; and (3) subjectiveness stemming from reviewers' personal preferences. To address such issues, we propose to model personalization in an uncontaminated open-ended generation assessment. We create two new datasets Per-MPST and Per-DOC for personalized story evaluation, by re-purposing existing datasets with proper anonymization and new personalized labels. We further develop a personalized story evaluation model PERSE to infer reviewer preferences and provide a personalized evaluation. Specifically, given a few exemplary reviews from a particular reviewer, PERSE predicts either a detailed review or fine-grained comparison in several aspects (such as interestingness and surprise) for that reviewer on a new 
    
[^57]: 通过交流使LLM代理适应环境的方法

    Adapting LLM Agents Through Communication. (arXiv:2310.01444v1 [cs.CL])

    [http://arxiv.org/abs/2310.01444](http://arxiv.org/abs/2310.01444)

    这项研究提出了一种名为学习通信（LTC）的训练方法，使用该方法可使LLM代理通过与环境和其他代理的交互不断改进，以适应新任务，而无需过多人类监督。

    

    最近大语言模型(LLM)的进展显示出了人类化代理的潜力。为了帮助这些代理在没有广泛人类监督的情况下适应新任务，我们提出了学习通信（LTC）范式，这是一种新颖的训练方法，使LLM代理能够通过与环境和其他代理的交互不断改进。通过迭代探索和PPO训练，LTC使代理能够将短期经验融入长期记忆。为了优化特定任务的代理交互，我们引入了三种结构化的通信模式：独白，对话，

    Recent advancements in large language models (LLMs) have shown potential for human-like agents. To help these agents adapt to new tasks without extensive human supervision, we propose the Learning through Communication (LTC) paradigm, a novel training approach enabling LLM agents to improve continuously through interactions with their environments and other agents. Recent advancements in large language models (LLMs) have shown potential for human-like agents. To help these agents adapt to new tasks without extensive human supervision, we propose the Learning through Communication (LTC) paradigm, a novel training approach enabling LLM agents to improve continuously through interactions with their environments and other agents. Through iterative exploration and PPO training, LTC empowers the agent to assimilate short-term experiences into long-term memory. To optimize agent interactions for task-specific learning, we introduce three structured communication patterns: Monologue, Dialogue,
    
[^58]: 表示工程化：AI透明化的自上而下方法

    Representation Engineering: A Top-Down Approach to AI Transparency. (arXiv:2310.01405v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2310.01405](http://arxiv.org/abs/2310.01405)

    这项研究介绍了一种名为表示工程化（RepE）的自上而下方法，通过借鉴认知神经科学的见解，提供了一种增强AI系统透明性的解决方案。该方法将集群级别的表示放在分析的核心，为监测和操纵深度神经网络中的高级认知现象提供了新的方法，并展示了在解决与安全相关的问题上的潜力。

    

    本文中，我们确定并描述了表示工程化（RepE）这一新兴领域，这是一种通过借鉴认知神经科学的见解来增强AI系统透明性的方法。RepE将集群级别的表示放在分析的核心，而不是神经元或电路，为我们提供了监测和操纵深度神经网络（DNNs）中高级认知现象的新方法。我们提供了RepE技术的基准和初步分析，显示它们提供了简单而有效的解决方案，用于改善我们对大型语言模型的理解和控制。我们展示了这些方法如何在包括诚实性、无害性、追求权力等一系列与安全相关的问题上发挥作用，展示了自上而下透明性研究的潜力。我们希望这项工作能够促进RepE的进一步探索，并推动AI系统的透明性和安全性的进步。

    In this paper, we identify and characterize the emerging area of representation engineering (RepE), an approach to enhancing the transparency of AI systems that draws on insights from cognitive neuroscience. RepE places population-level representations, rather than neurons or circuits, at the center of analysis, equipping us with novel methods for monitoring and manipulating high-level cognitive phenomena in deep neural networks (DNNs). We provide baselines and an initial analysis of RepE techniques, showing that they offer simple yet effective solutions for improving our understanding and control of large language models. We showcase how these methods can provide traction on a wide range of safety-relevant problems, including honesty, harmlessness, power-seeking, and more, demonstrating the promise of top-down transparency research. We hope that this work catalyzes further exploration of RepE and fosters advancements in the transparency and safety of AI systems.
    
[^59]: LEEC: 一种具有广泛领域特定标签系统的法律要素提取数据集

    LEEC: A Legal Element Extraction Dataset with an Extensive Domain-Specific Label System. (arXiv:2310.01271v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.01271](http://arxiv.org/abs/2310.01271)

    本文提出了一个具有广泛领域特定标签系统的大规模刑事要素提取数据集，通过借助法律专家团队和法律知识的注释，可以增强法律案例的解释和分析能力。

    

    作为自然语言处理中的重要任务，要素提取在法律领域中具有重要意义。从司法文件中提取法律要素有助于增强法律案例的解释和分析能力，从而促进法律领域各个领域的下游应用。然而，现有的要素提取数据集存在对法律知识的受限访问和标签覆盖不足的问题。为了解决这个问题，我们引入了一个更全面、大规模的刑事要素提取数据集，包括15,831个司法文件和159个标签。该数据集通过两个主要步骤构建：首先，由我们的法律专家团队根据先前的法律研究设计了标签系统，该研究确定了刑事案件中影响判决结果的关键因素和生成过程；其次，利用法律知识根据标签系统和注释准则对司法文件进行注释。

    As a pivotal task in natural language processing, element extraction has gained significance in the legal domain. Extracting legal elements from judicial documents helps enhance interpretative and analytical capacities of legal cases, and thereby facilitating a wide array of downstream applications in various domains of law. Yet existing element extraction datasets are limited by their restricted access to legal knowledge and insufficient coverage of labels. To address this shortfall, we introduce a more comprehensive, large-scale criminal element extraction dataset, comprising 15,831 judicial documents and 159 labels. This dataset was constructed through two main steps: first, designing the label system by our team of legal experts based on prior legal research which identified critical factors driving and processes generating sentencing outcomes in criminal cases; second, employing the legal knowledge to annotate judicial documents according to the label system and annotation guideli
    
[^60]: SELF：基于语言驱动的大型语言模型自主进化

    SELF: Language-Driven Self-Evolution for Large Language Model. (arXiv:2310.00533v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.00533](http://arxiv.org/abs/2310.00533)

    SELF提出了一种基于语言驱动的创新方法，允许大型语言模型（LLM）不断自我进化，并通过语言反馈作为评估工具来改进模型的响应能力和训练稳定性。

    

    大型语言模型（LLM）展示了在多个领域中的卓越适应能力。然而，实现人类水平的学习和推动自主人工智能的关键——模型自主进化的路径仍然未知。我们引入了一种创新的方法，名为"SELF"（带有语言反馈的自主进化）。这种方法使LLM能够不断地自我进化。此外，SELF利用语言反馈作为一种多功能、全面的评估工具，精确定位响应改进的领域，并提高自主进化训练的稳定性。SELF首先进行元技能学习，专注于自我反馈和自我精炼。这些元技能是关键，引导模型在自制数据的持续训练周期中进行后续的自我进化，从而增强其内在能力。在给定无标签指令的情况下，SELF使模型具备了能够...

    Large Language Models (LLMs) have showcased remarkable versatility across diverse domains. However, the pathway toward autonomous model development, a cornerstone for achieving human-level learning and advancing autonomous AI, remains largely uncharted. We introduce an innovative approach, termed "SELF" (Self-Evolution with Language Feedback). This methodology empowers LLMs to undergo continual self-evolution. Furthermore, SELF employs language-based feedback as a versatile and comprehensive evaluative tool, pinpointing areas for response refinement and bolstering the stability of self-evolutionary training. Initiating with meta-skill learning, SELF acquires foundational meta-skills with a focus on self-feedback and self-refinement. These meta-skills are critical, guiding the model's subsequent self-evolution through a cycle of perpetual training with self-curated data, thereby enhancing its intrinsic abilities. Given unlabeled instructions, SELF equips the model with the capability to
    
[^61]: 红队游戏：红队语言模型的博弈论框架

    Red Teaming Game: A Game-Theoretic Framework for Red Teaming Language Models. (arXiv:2310.00322v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.00322](http://arxiv.org/abs/2310.00322)

    本文提出了红队游戏（RTG）框架，利用博弈论分析了红队语言模型（RLM）与蓝队语言模型（BLM）之间的多轮攻防互动。同时引入了游戏化红队求解器（GRTS）来提供自动化的红队技术。

    

    可部署的大型语言模型（LLM）必须符合有益和无害性的标准，从而实现LLM输出与人类价值的一致性。红队技术是实现这一标准的关键途径。现有的研究仅依赖于手动红队设计和启发式对抗提示进行漏洞检测和优化。这些方法缺乏严格的数学形式化，限制了在可量化度量和收敛保证下对LLM进行多样攻击策略的探索和优化。在本文中，我们提出了红队游戏（RTG），这是一个通用的无需手动标注的博弈论框架。RTG旨在分析红队语言模型（RLM）与蓝队语言模型（BLM）之间的多轮攻防互动。在RTG中，我们提出了具有语义空间多样性度量的游戏化红队求解器（GRTS）。GRTS是一种自动化的红队技术，用于解决红队游戏问题。

    Deployable Large Language Models (LLMs) must conform to the criterion of helpfulness and harmlessness, thereby achieving consistency between LLMs outputs and human values. Red-teaming techniques constitute a critical way towards this criterion. Existing work rely solely on manual red team designs and heuristic adversarial prompts for vulnerability detection and optimization. These approaches lack rigorous mathematical formulation, thus limiting the exploration of diverse attack strategy within quantifiable measure and optimization of LLMs under convergence guarantees. In this paper, we present Red-teaming Game (RTG), a general game-theoretic framework without manual annotation. RTG is designed for analyzing the multi-turn attack and defense interactions between Red-team language Models (RLMs) and Blue-team Language Model (BLM). Within the RTG, we propose Gamified Red-teaming Solver (GRTS) with diversity measure of the semantic space. GRTS is an automated red teaming technique to solve 
    
[^62]: 理解上下文学习中的重复现象

    Understanding In-Context Learning from Repetitions. (arXiv:2310.00297v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2310.00297](http://arxiv.org/abs/2310.00297)

    本论文通过研究表面重复现象的角度来探索大型语言模型中上下文学习的机制，并实证了一种增强标记关系的原则，为理解上下文学习及其潜在局限性做出了重要贡献。

    

    本论文探索了大型语言模型（LLM）中上下文学习的难以捉摸的机制。我们通过研究表面重复现象的角度来检视上下文学习，并定量地研究了表面特征在文本生成中的作用，同时实证了一种被称为“标记共现强化”的原则，该原则通过增强两个标记之间的关系来基于它们的上下文共现。通过研究这些特征的双重影响，我们的研究阐明了上下文学习的内在机制，并对其失败的原因进行了解释。本论文对于理解上下文学习及其潜在局限性做出了重要贡献，为这一激动人心的能力提供了新的视角。

    This paper explores the elusive mechanism underpinning in-context learning in Large Language Models (LLMs). Our work provides a novel perspective by examining in-context learning via the lens of surface repetitions. We quantitatively investigate the role of surface features in text generation, and empirically establish the existence of \emph{token co-occurrence reinforcement}, a principle that strengthens the relationship between two tokens based on their contextual co-occurrences. By investigating the dual impacts of these features, our research illuminates the internal workings of in-context learning and expounds on the reasons for its failures. This paper provides an essential contribution to the understanding of in-context learning and its potential limitations, providing a fresh perspective on this exciting capability.
    
[^63]: 两两邻近策略优化: 利用相对反馈进行LLM对齐

    Pairwise Proximal Policy Optimization: Harnessing Relative Feedback for LLM Alignment. (arXiv:2310.00212v1 [cs.LG])

    [http://arxiv.org/abs/2310.00212](http://arxiv.org/abs/2310.00212)

    该论文提出了一种新的强化学习框架，使用相对反馈来调整大型语言模型（LLMs）的行为，解决了现有方法在优化比较损失训练的奖励时存在的限制。同时，还提出了一种新的基于轨迹的策略梯度算法（PPPO），用于更有效地进行算法设计和函数逼近。

    

    大型语言模型（LLMs）通过在大型语料库上预先训练来获取广泛的世界知识。然而，由于接触到低质量数据，LLMs可能表现出与人类价值不一致的有害行为。引导LLMs朝着有益行为方向发展的主导方法涉及使用人类反馈的强化学习（RLHF），其中Proximal Policy Optimization（PPO）是默认的RL优化器。尽管其有效性，但PPO在优化基于比较损失训练的奖励时存在局限性。主要问题是，由于需要校准奖励尺度，PPO对于包含相同偏好信息的等价奖励函数不具备不变性。此外，与基于轨迹的优化相比，PPO对于基于令牌的更新的需求引入了函数逼近和算法设计方面的复杂性。本文提出了一种新的框架，基于相对反馈的强化学习，以及一种新颖的基于轨迹的策略梯度算法，Pairwise Proximal Policy Optimization（PPPO），用于解决上述问题。

    Large Language Models (LLMs) can acquire extensive world knowledge through pre-training on large corpora. However, due to exposure to low-quality data, LLMs may exhibit harmful behavior without aligning with human values. The dominant approach for steering LLMs towards beneficial behavior involves Reinforcement Learning with Human Feedback (RLHF), with Proximal Policy Optimization (PPO) serving as the default RL optimizer. Despite its effectiveness, PPO has limitations when optimizing rewards trained from comparison-based loss. Primarily, PPO is not invariant to equivalent reward functions containing identical preference information due to the need to calibrate the reward scale. Additionally, PPO's necessity for token-wise updates introduces complexity in both function approximation and algorithm design compared to trajectory-wise optimization. This paper proposes a new framework, reinforcement learning with relative feedback, and a novel trajectory-wise policy gradient algorithm, Pair
    
[^64]: GPT-Fathom：评估大型语言模型以解析GPT-4及其后续版本的演化路径的基准测试

    GPT-Fathom: Benchmarking Large Language Models to Decipher the Evolutionary Path towards GPT-4 and Beyond. (arXiv:2309.16583v1 [cs.CL])

    [http://arxiv.org/abs/2309.16583](http://arxiv.org/abs/2309.16583)

    GPT-Fathom是一个用于评估大型语言模型的开源套件，它系统评估了10多个主要的语言模型，并提供了从GPT-3到GPT-4演化路径的宝贵见解。

    

    随着大型语言模型（LLMs）的快速进展，人们迫切需要一个全面的评估套件来评估它们的能力和局限性。现有的LLM排行榜通常参考其他论文中报告的得分，设置和提示不一致，这可能无意间鼓励选择有利的设置和提示以获得更好的结果。在这项工作中，我们引入了GPT-Fathom，这是一个基于OpenAI Evals构建的开源和可重复的LLM评估套件。我们在对齐的环境设置下系统评估了10多个主要的LLMs以及OpenAI的传统模型在20多个精选基准测试中的表现，涵盖了7个能力类别。我们对OpenAI早期模型的回顾性研究为我们揭示了从GPT-3到GPT-4的演化路径提供了宝贵的见解。目前，社区渴望了解GPT-3如何逐步改进到GPT-4，包括像添加代码数据是否提高了LLM的推理能力以及LLM能力的哪些方面等技术细节。

    With the rapid advancement of large language models (LLMs), there is a pressing need for a comprehensive evaluation suite to assess their capabilities and limitations. Existing LLM leaderboards often reference scores reported in other papers without consistent settings and prompts, which may inadvertently encourage cherry-picking favored settings and prompts for better results. In this work, we introduce GPT-Fathom, an open-source and reproducible LLM evaluation suite built on top of OpenAI Evals. We systematically evaluate 10+ leading LLMs as well as OpenAI's legacy models on 20+ curated benchmarks across 7 capability categories, all under aligned settings. Our retrospective study on OpenAI's earlier models offers valuable insights into the evolutionary path from GPT-3 to GPT-4. Currently, the community is eager to know how GPT-3 progressively improves to GPT-4, including technical details like whether adding code data improves LLM's reasoning capability, which aspects of LLM capabili
    
[^65]: 用大型语言模型进行生成式语音识别错误校正

    Generative Speech Recognition Error Correction with Large Language Models. (arXiv:2309.15649v1 [cs.CL])

    [http://arxiv.org/abs/2309.15649](http://arxiv.org/abs/2309.15649)

    本研究探讨了大型语言模型（LLMs）作为ASR后处理器的能力，通过重新评分和错误校正来提高系统性能。通过使用指令提示和任务激活提示方法，结合上下文学习和微调技术，我们展示了LLMs的泛化能力和有效性。

    

    我们研究了大型语言模型（LLM）作为ASR后处理器的能力，用于重新评分和错误校正。我们的重点是使用指令提示让LLMs执行这些任务而无需微调，我们评估了不同的提示方案，包括零-shot和少-shot的上下文学习，以及一种新颖的任务激活提示（TAP）方法，结合指令和演示。通过在两个领域之外的任务（ATIS和WSJ）上使用预先训练的第一次扫描系统和重新评分输出，我们证明仅通过冻结LLMs的上下文学习进行重新评分可以达到与领域调优的LMs重新评分相竞争的结果。通过将提示技术与微调相结合，我们实现了低于N-best Oracle水平的错误率，展示了LLMs的泛化能力。

    We explore the ability of large language models (LLMs) to act as ASR post-processors that perform rescoring and error correction. Our focus is on instruction prompting to let LLMs perform these task without fine-tuning, for which we evaluate different prompting schemes, both zero- and few-shot in-context learning, and a novel task-activating prompting (TAP) method that combines instruction and demonstration. Using a pre-trained first-pass system and rescoring output on two out-of-domain tasks (ATIS and WSJ), we show that rescoring only by in-context learning with frozen LLMs achieves results that are competitive with rescoring by domain-tuned LMs. By combining prompting techniques with fine-tuning we achieve error rates below the N-best oracle level, showcasing the generalization power of the LLMs.
    
[^66]: 大规模语言模型重评分的低秩适应技术在参数高效的语音识别中的应用

    Low-rank Adaptation of Large Language Model Rescoring for Parameter-Efficient Speech Recognition. (arXiv:2309.15223v1 [cs.CL])

    [http://arxiv.org/abs/2309.15223](http://arxiv.org/abs/2309.15223)

    这篇论文介绍了一种基于低秩适应技术的神经语言建模系统，用于语音识别的输出重评分。通过使用低秩分解方法和优化插入矩阵，该系统能够以更高效的方式将BERT模型适应到新领域，大大减少了训练时间。

    

    我们提出了一种基于低秩适应（LoRA）的神经语言建模系统，用于语音识别输出重评分。尽管预训练的语言模型（如BERT）在第二次重评分中表现出优越的性能，但将预训练阶段扩展和将预训练模型适应到特定领域的高计算成本限制了它们在重评分中的实际应用。我们提出了一种基于低秩分解的方法，仅使用预训练参数的一小部分（0.08%）来训练重评分的BERT模型并将其适应到新领域。这些插入的矩阵通过相关性正则化损失和判别性训练目标进行优化。所提出的低秩适应Rescore-BERT（LoRB）体系结构在LibriSpeech和内部数据集上评估，训练时间减少了5.4至3.6倍。

    We propose a neural language modeling system based on low-rank adaptation (LoRA) for speech recognition output rescoring. Although pretrained language models (LMs) like BERT have shown superior performance in second-pass rescoring, the high computational cost of scaling up the pretraining stage and adapting the pretrained models to specific domains limit their practical use in rescoring. Here we present a method based on low-rank decomposition to train a rescoring BERT model and adapt it to new domains using only a fraction (0.08%) of the pretrained parameters. These inserted matrices are optimized through a discriminative training objective along with a correlation-based regularization loss. The proposed low-rank adaptation Rescore-BERT (LoRB) architecture is evaluated on LibriSpeech and internal datasets with decreased training times by factors between 5.4 and 3.6.
    
[^67]: 通过学习表示和影响函数，我们能从对抗样本中获得什么信息

    What Learned Representations and Influence Functions Can Tell Us About Adversarial Examples. (arXiv:2309.10916v1 [cs.LG])

    [http://arxiv.org/abs/2309.10916](http://arxiv.org/abs/2309.10916)

    本文将图像处理领域中的对抗子空间技术应用于自然语言处理，提出了基于最近邻和影响函数的检测器，并通过使用影响函数揭示了自然语言处理中的对抗样本子空间与图像处理中的子空间的关系和任务差异。

    

    对抗样本是通过微小扰动来欺骗深度神经网络的，起初在图像处理领域进行研究，最近在自然语言处理领域也开始关注。尽管在自然语言处理中检测对抗样本的方法主要依赖于输入扰动的搜索，但图像处理领域已经发展出一系列技术来表征学习表示中的对抗子空间。本文将这两种方法应用于自然语言处理，一种基于最近邻和影响函数，一种基于马氏距离。特别是前者相比几个强基准产生了最先进的检测器；此外，对影响函数的新颖使用揭示了自然语言处理中的对抗样本子空间与图像处理中的子空间的关系，并展示了它们根据不同自然语言处理任务的差异。

    Adversarial examples, deliberately crafted using small perturbations to fool deep neural networks, were first studied in image processing and more recently in NLP. While approaches to detecting adversarial examples in NLP have largely relied on search over input perturbations, image processing has seen a range of techniques that aim to characterise adversarial subspaces over the learned representations.  In this paper, we adapt two such approaches to NLP, one based on nearest neighbors and influence functions and one on Mahalanobis distances. The former in particular produces a state-of-the-art detector when compared against several strong baselines; moreover, the novel use of influence functions provides insight into how the nature of adversarial example subspaces in NLP relate to those in image processing, and also how they differ depending on the kind of NLP task.
    
[^68]: SlimPajama-DC: 理解LLM训练中的数据组合

    SlimPajama-DC: Understanding Data Combinations for LLM Training. (arXiv:2309.10818v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2309.10818](http://arxiv.org/abs/2309.10818)

    本研究探讨了使用SlimPajama进行大型语言模型训练中不同数据组合的影响，提出了全局去重和局部去重的比较和高质量多源数据集的比例对模型性能的影响。

    

    本研究旨在了解使用SlimPajama进行大型语言模型训练时各种数据组合（如网络文本、维基百科、GitHub、图书）对其训练的影响。SlimPajama是一个经过严格去重的多源数据集，从Together贡献的1.2T个token的RedPajama数据集中精细组合和去重，总共得到了627B个tokens。我们将我们的研究称为SlimPajama-DC，这是一项旨在揭示在大型语言模型训练中使用SlimPajama所涉及的基本特征和最佳实践的经验分析。在我们使用SlimPajama进行研究的过程中，出现了两个关键观察结果：（1）全局去重 vs. 局部去重。我们分析和讨论了全局去重（跨不同数据集源）和局部去重（在单个数据集源内部）对训练模型性能的影响。（2）高质量/高度去重的多源数据集在组合中的比例。为了研究这一点，我们进行了...

    This paper aims to understand the impacts of various data combinations (e.g., web text, wikipedia, github, books) on the training of large language models using SlimPajama. SlimPajama is a rigorously deduplicated, multi-source dataset, which has been refined and further deduplicated to 627B tokens from the extensive 1.2T tokens RedPajama dataset contributed by Together. We've termed our research as SlimPajama-DC, an empirical analysis designed to uncover fundamental characteristics and best practices associated with employing SlimPajama in the training of large language models. During our research with SlimPajama, two pivotal observations emerged: (1) Global deduplication vs. local deduplication. We analyze and discuss how global (across different sources of datasets) and local (within the single source of dataset) deduplications affect the performance of trained models. (2) Proportions of high-quality/highly-deduplicated multi-source datasets in the combination. To study this, we cons
    
[^69]: PoSE: 通过位置跳跃式训练提高LLMs对于上下文窗口的有效拓展

    PoSE: Efficient Context Window Extension of LLMs via Positional Skip-wise Training. (arXiv:2309.10400v1 [cs.CL])

    [http://arxiv.org/abs/2309.10400](http://arxiv.org/abs/2309.10400)

    本文介绍了一种名为PoSE的训练方法，通过在训练过程中使用固定的上下文窗口和操纵位置索引来适应极长的上下文窗口，实验证明这种方法大大减小了内存和时间开销，对性能影响较小，成功将LLaMA模型扩展到了128k个标记。

    

    本文介绍了一种名为Positional Skip-wise (PoSE)训练的方法，用于将大型语言模型（LLMs）适应于极长的上下文窗口。PoSE通过在训练过程中使用固定的上下文窗口和操纵位置索引来模拟长输入，将训练长度与目标上下文窗口大小分离。具体而言，我们从长输入序列中选择若干短块，并引入不同的跳跃偏置项来修改每个块的位置索引。这些跳跃偏置项以及每个块的长度在每个训练样本中都会变化，使得模型能够适应目标上下文窗口中的所有位置，而无需对完整长度的输入进行训练。实验证明，与对完整长度进行微调相比，PoSE大大减小了内存和时间开销，对性能影响较小。利用这一优势，我们成功将LLaMA模型扩展到了128k个标记。此外，我们经验证实，PoSE与

    In this paper, we introduce Positional Skip-wisE (PoSE) training for efficient adaptation of large language models~(LLMs) to extremely long context windows. PoSE decouples train length from target context window size by simulating long inputs using a fixed context window with manipulated position indices during training. Concretely, we select several short chunks from a long input sequence, and introduce distinct skipping bias terms to modify the position indices of each chunk. These bias terms, along with the length of each chunk, are altered for each training example, allowing the model to adapt to all positions within the target context window without training on full length inputs. Experiments show that, compared with fine-tuning on the full length, PoSE greatly reduces memory and time overhead with minimal impact on performance. Leveraging this advantage, we have successfully extended the LLaMA model to 128k tokens. Furthermore, we empirically confirm that PoSE is compatible with 
    
[^70]: 通过准确性预测器修剪大型语言模型

    Pruning Large Language Models via Accuracy Predictor. (arXiv:2309.09507v2 [cs.AI] UPDATED)

    [http://arxiv.org/abs/2309.09507](http://arxiv.org/abs/2309.09507)

    本文提出了一种通过准确性预测器来修剪大型语言模型的新方法，实验证明该方法是有效且高效的，可以自动选择最优模型并降低困惑度。

    

    大型语言模型(LLMs)具有数百亿个参数（甚至更多），在各种自然语言处理任务中展示出了令人印象深刻的能力。然而，庞大的模型规模给训练、推理和部署带来了挑战，因此有必要对模型进行压缩。目前，大部分LLMs模型压缩需要手动设计修剪特性，存在诸如复杂的优化流程和难以保留模型部分能力的问题。因此，我们提出了一种新的修剪方法：首先，建立一组特定数量的架构-准确性对的训练集，然后训练一个非神经模型作为准确性预测器。通过使用准确性预测器进一步优化搜索空间和搜索，可以自动选择最优模型。实验结果显示我们提出的方法是有效且高效的。与基准模型相比，在Wikitext2和PTB上的困惑度（PPL）分别下降了9.48%和5.7%。

    Large language models(LLMs) containing tens of billions of parameters (or even more) have demonstrated impressive capabilities in various NLP tasks. However, substantial model size poses challenges to training, inference, and deployment so that it is necessary to compress the model. At present, most model compression for LLMs requires manual design of pruning features, which has problems such as complex optimization pipeline and difficulty in retaining the capabilities of certain parts of the model.Therefore, we propose a novel pruning approach: firstly, a training set of a certain number of architecture-accuracy pairs is established, and then a non-neural model is trained as an accuracy predictor. Using the accuracy predictor to further optimize the search space and search, the optimal model can be automatically selected. Experiments show that our proposed approach is effective and efficient. Compared with the baseline, the perplexity(PPL) on Wikitext2 and PTB dropped by 9.48% and 5,7
    
[^71]: 大型视觉语言模型中幻觉的评估与分析

    Evaluation and Analysis of Hallucination in Large Vision-Language Models. (arXiv:2308.15126v1 [cs.LG])

    [http://arxiv.org/abs/2308.15126](http://arxiv.org/abs/2308.15126)

    本文提出了基于大型语言模型的幻觉评估框架HaELM，可以评估大型视觉语言模型中的幻觉问题，并分析了导致幻觉的因素，并提出了缓解幻觉问题的建议。

    

    最近，大型视觉语言模型（LVLMs）取得了显著的成功。然而，LVLMs仍然存在幻觉问题，这限制了在许多场景中的实用性。幻觉指的是LVLMs响应中不存在于视觉输入中的信息，这可能导致重大后果的潜在风险。目前对LVLMs中的幻觉评估的研究工作有限。在本文中，我们提出了基于大型语言模型（LLM）的幻觉评估框架HaELM。HaELM的性能近似于ChatGPT的95%，并具有低成本、可复现、保护隐私和本地部署等额外优势。利用HaELM，我们评估了当前LVLMs中的幻觉。此外，我们分析了导致LVLMs中幻觉的因素，并提出了缓解幻觉问题的有用建议。

    Large Vision-Language Models (LVLMs) have recently achieved remarkable success. However, LVLMs are still plagued by the hallucination problem, which limits the practicality in many scenarios. Hallucination refers to the information of LVLMs' responses that does not exist in the visual input, which poses potential risks of substantial consequences. There has been limited work studying hallucination evaluation in LVLMs. In this paper, we propose Hallucination Evaluation based on Large Language Models (HaELM), an LLM-based hallucination evaluation framework. HaELM achieves an approximate 95% performance comparable to ChatGPT and has additional advantages including low cost, reproducibility, privacy preservation and local deployment. Leveraging the HaELM, we evaluate the hallucination in current LVLMs. Furthermore, we analyze the factors contributing to hallucination in LVLMs and offer helpful suggestions to mitigate the hallucination problem. Our training data and human annotation halluci
    
[^72]: 授权临床医生并民主化数据科学：大型语言模型自动化临床研究的机器学习。 (arXiv:2308.14120v2 [cs.LG] 更新版)

    Empowering Clinicians and Democratizing Data Science: Large Language Models Automate Machine Learning for Clinical Studies. (arXiv:2308.14120v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2308.14120](http://arxiv.org/abs/2308.14120)

    chatGPT ADA是一种能够自主开发临床研究所需的最先进的机器学习模型的大型语言模型，可将高级分析工具民主化，使非数据科学家的临床医生能够轻松应用于医学领域。

    

    机器学习（ML）开发者（如数据科学家）和从业者（如临床医生）之间存在知识差距，阻碍了ML在临床数据分析中的充分利用。我们研究了chatGPT Advanced Data Analysis（ADA），即GPT-4的扩展，来弥合这一差距并高效执行ML分析的潜力。我们向chatGPT ADA提供了各种医学专业的大型试验的真实临床数据和研究详细信息，没有给出具体指导。ChatGPT ADA基于原始研究的训练数据自主开发了最先进的ML模型，用于预测临床结果，如癌症发展、癌症进展、疾病并发症或致病基因序列等生物标志物。令人惊讶的是，这些ML模型与其已发表的对应物相匹配甚至表现更好。我们得出结论，chatGPT ADA为民主化医学中的ML提供了一个有前景的途径，使非ML专家能够获得先进的分析工具并推动广泛应用。

    A knowledge gap persists between Machine Learning (ML) developers (e.g., data scientists) and practitioners (e.g., clinicians), hampering the full utilization of ML for clinical data analysis. We investigated the potential of the chatGPT Advanced Data Analysis (ADA), an extension of GPT-4, to bridge this gap and perform ML analyses efficiently. Real-world clinical datasets and study details from large trials across various medical specialties were presented to chatGPT ADA without specific guidance. ChatGPT ADA autonomously developed state-of-the-art ML models based on the original study's training data to predict clinical outcomes such as cancer development, cancer progression, disease complications, or biomarkers such as pathogenic gene sequences. Strikingly, these ML models matched or outperformed their published counterparts. We conclude that chatGPT ADA offers a promising avenue to democratize ML in medicine, making advanced analytics accessible to non-ML experts and promoting broa
    
[^73]: 使用多条件扩散模型进行音频生成

    Audio Generation with Multiple Conditional Diffusion Model. (arXiv:2308.11940v1 [cs.SD])

    [http://arxiv.org/abs/2308.11940](http://arxiv.org/abs/2308.11940)

    本论文提出了一种使用多条件扩散模型进行音频生成的方法。通过引入内容和风格等额外条件，增强了现有模型的可控性。这种方法可以精确控制生成音频的时间顺序、音高和能量。由于缺乏合适的数据集和评估指标，作者整合了现有数据集并进行了实验验证。

    

    基于文本的音频生成模型有其局限性，因为它们无法包含音频中的所有信息，仅依靠文本会导致受控性受限。为了解决这个问题，我们提出了一种新颖的模型，通过引入额外的条件（包括内容（时间戳）和风格（音高曲线和能量曲线））作为文本的补充，增强了现有预训练文本到音频模型的可控性。这种方法实现了对生成音频的时间顺序、音高和能量的精细控制。为了保持生成的多样性，我们使用一个可训练的控制条件编码器，该编码器由一个大型语言模型增强，并使用一个可训练的融合网络来编码和融合额外的条件，同时保持预训练文本到音频模型的权重不变。由于缺乏合适的数据集和评估指标，我们将现有数据集整合为一个新的数据集，包括音频和相应的条件，并使用一个可训练的控制条件编码器，该编码器由一个大型语言模型增强，并使用一个可训练的融合网络来编码和融合额外的条件，同时保持预训练文本到音频模型的权重不变。

    Text-based audio generation models have limitations as they cannot encompass all the information in audio, leading to restricted controllability when relying solely on text. To address this issue, we propose a novel model that enhances the controllability of existing pre-trained text-to-audio models by incorporating additional conditions including content (timestamp) and style (pitch contour and energy contour) as supplements to the text. This approach achieves fine-grained control over the temporal order, pitch, and energy of generated audio. To preserve the diversity of generation, we employ a trainable control condition encoder that is enhanced by a large language model and a trainable Fusion-Net to encode and fuse the additional conditions while keeping the weights of the pre-trained text-to-audio model frozen. Due to the lack of suitable datasets and evaluation metrics, we consolidate existing datasets into a new dataset comprising the audio and corresponding conditions and use a 
    
[^74]: 三思而后行：大型语言模型不确定性测量的探索性研究

    Look Before You Leap: An Exploratory Study of Uncertainty Measurement for Large Language Models. (arXiv:2307.10236v1 [cs.SE])

    [http://arxiv.org/abs/2307.10236](http://arxiv.org/abs/2307.10236)

    本研究从不确定性的角度对大型语言模型进行了探索性研究，通过实验发现不确定性估计方法在探索和抵制大型语言模型的不良行为方面具有潜力。

    

    大型语言模型（LLMs）的最近性能突破为众多工业应用和领域提供了新的机遇。然而，LLMs的错误生成，如虚假预测、错误信息和幻觉，也引发了对LLMs可靠性的严重关注，尤其在对安全、可靠性有敏感的场景中，可能阻碍其在实际中的应用。尽管不确定性估计已经显示出其在解释一般机器学习（ML）模型的预测风险方面的潜力，但关于它是否以及在多大程度上有助于探索LLMs的能力和抵制其不良行为方面知之甚少。为了弥合这一差距，本文从不确定性的角度开展了关于LLMs风险评估的探索性研究。具体来说，我们使用12种不确定性估计方法和4个LLMs在4个重要的自然语言处理（NLP）任务上进行实验，以调查不确定性在探索LLMs能力和对抗其不良行为方面的程度。

    The recent performance leap of Large Language Models (LLMs) opens up new opportunities across numerous industrial applications and domains. However, erroneous generations, such as false predictions, misinformation, and hallucination made by LLMs, have also raised severe concerns for the trustworthiness of LLMs', especially in safety-, security- and reliability-sensitive scenarios, potentially hindering real-world adoptions. While uncertainty estimation has shown its potential for interpreting the prediction risks made by general machine learning (ML) models, little is known about whether and to what extent it can help explore an LLM's capabilities and counteract its undesired behavior. To bridge the gap, in this paper, we initiate an exploratory study on the risk assessment of LLMs from the lens of uncertainty. In particular, we experiment with twelve uncertainty estimation methods and four LLMs on four prominent natural language processing (NLP) tasks to investigate to what extent unc
    
[^75]: 为口语评估适应ASR基础模型

    Adapting an ASR Foundation Model for Spoken Language Assessment. (arXiv:2307.09378v1 [cs.CL])

    [http://arxiv.org/abs/2307.09378](http://arxiv.org/abs/2307.09378)

    本论文主要针对ASR基础模型Whisper的输出进行了详细分析，并提出了两种解决方案：微调和软提示微调。实验结果表明，可以有效改变Whisper的解码行为，生成候选人实际说出的单词。

    

    准确可靠的口语评估系统的关键部分是底层的ASR模型。最近，大规模预训练的ASR基础模型如Whisper已经可用。由于这些模型的输出是人类可读的，所以会添加标点符号，数字呈现为阿拉伯数字形式，包括缩写。此外，这些模型往往会跳过输出中的不流畅和犹豫。虽然对于可读性很有用，但这些属性对于评估候选人的能力和提供反馈并不有用。在这篇论文中，我们对Whisper的输出进行了详细分析，并提出了两种解决方案：微调和软提示微调。在公共语音语料库和英语学习者数据集上进行了实验。结果表明，我们可以有效地改变Whisper的解码行为，生成候选人实际说出的单词。

    A crucial part of an accurate and reliable spoken language assessment system is the underlying ASR model. Recently, large-scale pre-trained ASR foundation models such as Whisper have been made available. As the output of these models is designed to be human readable, punctuation is added, numbers are presented in Arabic numeric form and abbreviations are included. Additionally, these models have a tendency to skip disfluencies and hesitations in the output. Though useful for readability, these attributes are not helpful for assessing the ability of a candidate and providing feedback. Here a precise transcription of what a candidate said is needed. In this paper, we give a detailed analysis of Whisper outputs and propose two solutions: fine-tuning and soft prompt tuning. Experiments are conducted on both public speech corpora and an English learner dataset. Results show that we can effectively alter the decoding behaviour of Whisper to generate the exact words spoken in the response.
    
[^76]: DocumentNet: 在文档预训练中弥合数据差距

    DocumentNet: Bridging the Data Gap in Document Pre-Training. (arXiv:2306.08937v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2306.08937](http://arxiv.org/abs/2306.08937)

    这项研究提出了DocumentNet方法，通过从Web上收集大规模和弱标注的数据，弥合了文档预训练中的数据差距，并在各类VDER任务中展现了显著的性能提升。

    

    近年来，文档理解任务，特别是富有视觉元素的文档实体检索（VDER），由于在企业人工智能领域的广泛应用，受到了极大关注。然而，由于严格的隐私约束和高昂的标注成本，这些任务的公开可用数据非常有限。更糟糕的是，来自不同数据集的不重叠实体空间妨碍了文档类型之间的知识转移。在本文中，我们提出了一种从Web收集大规模和弱标注的数据的方法，以利于VDER模型的训练。所收集的数据集名为DocumentNet，不依赖于特定的文档类型或实体集，使其适用于所有的VDER任务。目前的DocumentNet包含了30M个文档，涵盖了近400个文档类型，组织成了一个四级本体结构。在一系列广泛采用的VDER任务上进行的实验表明，当将DocumentNet纳入预训练过程时，取得了显著的改进。

    Document understanding tasks, in particular, Visually-rich Document Entity Retrieval (VDER), have gained significant attention in recent years thanks to their broad applications in enterprise AI. However, publicly available data have been scarce for these tasks due to strict privacy constraints and high annotation costs. To make things worse, the non-overlapping entity spaces from different datasets hinder the knowledge transfer between document types. In this paper, we propose a method to collect massive-scale and weakly labeled data from the web to benefit the training of VDER models. The collected dataset, named DocumentNet, does not depend on specific document types or entity sets, making it universally applicable to all VDER tasks. The current DocumentNet consists of 30M documents spanning nearly 400 document types organized in a four-level ontology. Experiments on a set of broadly adopted VDER tasks show significant improvements when DocumentNet is incorporated into the pre-train
    
[^77]: 利用纵向胸部X光和报告预填充放射学报告

    Utilizing Longitudinal Chest X-Rays and Reports to Pre-Fill Radiology Reports. (arXiv:2306.08749v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2306.08749](http://arxiv.org/abs/2306.08749)

    本文提出了利用纵向胸部X光和报告来预填充放射学报告的方法，以减轻报告错误。通过利用纵向多模态数据和基于transformer的模型，可以更好地捕获包含多模态纵向就诊记录的信息。

    

    尽管使用语音识别软件可以缩短放射学报告的完成时间，但持续的沟通错误会对放射学报告的解释产生重大影响。预填充放射学报告在减轻报告错误方面具有潜力，尽管文献中存在生成医学报告的努力，但对于利用MIMIC-CXR数据集中的患者就诊记录的纵向特性缺乏方法。为了解决这一差距，我们提出使用纵向多模态数据，即以前的患者就诊CXR、当前就诊CXR和以前的就诊报告，来预先填充当前患者就诊报告的“发现”部分。我们首先从MIMIC-CXR数据集中收集了26,625名患者的纵向就诊信息，并创建了一个名为Longitudinal-MIMIC的新数据集。利用这个新数据集，我们训练了一个基于transformer的模型，以捕获包含多模态纵向就诊记录的信息。

    Despite the reduction in turn-around times in radiology reports with the use of speech recognition software, persistent communication errors can significantly impact the interpretation of the radiology report. Pre-filling a radiology report holds promise in mitigating reporting errors, and despite efforts in the literature to generate medical reports, there exists a lack of approaches that exploit the longitudinal nature of patient visit records in the MIMIC-CXR dataset. To address this gap, we propose to use longitudinal multi-modal data, i.e., previous patient visit CXR, current visit CXR, and previous visit report, to pre-fill the 'findings' section of a current patient visit report. We first gathered the longitudinal visit information for 26,625 patients from the MIMIC-CXR dataset and created a new dataset called Longitudinal-MIMIC. With this new dataset, a transformer-based model was trained to capture the information from longitudinal patient visit records containing multi-modal 
    
[^78]: 适应一个不可适应的语音识别系统

    Adapting an Unadaptable ASR System. (arXiv:2306.01208v1 [eess.AS])

    [http://arxiv.org/abs/2306.01208](http://arxiv.org/abs/2306.01208)

    本文探讨了一种适应在线服务提供商的语音识别系统的方法，采用错误纠正的方法，并以 OpenAI Whisper ASR 为例，使用 LibriSpeech 作为主要适应目标领域进行实验。结果显示该方法具有一定的泛化能力，可适用于不同体系结构的 ASR 模型。

    

    随着语音识别模型大小和训练数据需求的增长，系统通常只能通过在线服务提供商的API获得访问权限，而无法直接访问模型。在这种情况下，将系统适应到特定目标领域是具有挑战性的。为解决这个问题，本文采用错误纠正的方法，评估 OpenAI Whisper ASR 作为大规模 ASR 系统的适应方法。该方法不需要访问模型，而是可以从通常通过 ASR API 提供的 1-best 或 N-best 输出进行训练。LibriSpeech 被用作适应的主要目标领域。然后评估了系统在两个不同方面的泛化能力：首先，纠正模型的形式是否可以移植到其他语音识别领域；其次，它是否可用于具有不同体系结构的 ASR 模型。

    As speech recognition model sizes and training data requirements grow, it is increasingly common for systems to only be available via APIs from online service providers rather than having direct access to models themselves. In this scenario it is challenging to adapt systems to a specific target domain. To address this problem we consider the recently released OpenAI Whisper ASR as an example of a large-scale ASR system to assess adaptation methods. An error correction based approach is adopted, as this does not require access to the model, but can be trained from either 1-best or N-best outputs that are normally available via the ASR API. LibriSpeech is used as the primary target domain for adaptation. The generalization ability of the system in two distinct dimensions are then evaluated. First, whether the form of correction model is portable to other speech recognition domains, and secondly whether it can be used for ASR models having a different architecture.
    
[^79]: 基于偏好的语言模型微调中的标记级指导

    Preference-grounded Token-level Guidance for Language Model Fine-tuning. (arXiv:2306.00398v1 [cs.CL])

    [http://arxiv.org/abs/2306.00398](http://arxiv.org/abs/2306.00398)

    本论文通过设计一个交替训练过程，在序列级别偏好与标记级别训练指导之间进行迭代，并利用学习到的指导改进LM。实验证明该方法在多个文本生成任务中表现良好。

    

    将偏好与语言模型（LMs）相匹配是自然语言生成中的一个重要问题。关键挑战是偏好通常在序列级别上提供，而LM训练和生成都发生在标记级别上。因此，偏好和LM训练损失之间存在颗粒度不匹配，这可能会复杂化学习问题。本文通过开发一种交替训练过程解决了这个问题，在序列级别偏好与标记级别训练指导之间进行迭代，并利用学习到的指导改进LM。为了学习指导，我们设计了一个框架，将模仿学习中的成对偏好学习扩展到可变长度LM生成和利用多个生成之间的偏好。对于LM训练，基于监督数据量，我们提出了两种利用学习到的指导的极简主义学习目标。在实验中，我们的方法在多个文本生成任务中表现良好，表明我们的指导提高了生成序列的质量，允许更精细的控制。

    Aligning language models (LMs) with preferences is an important problem in natural language generation. A key challenge is that preferences are typically provided at the sequence level while LM training and generation both occur at the token level. There is, therefore, a granularity mismatch between the preference and the LM training losses, which may complicate the learning problem. In this paper, we address this issue by developing an alternate training process, where we iterate between grounding the sequence-level preference into token-level training guidance, and improving the LM with the learned guidance. For guidance learning, we design a framework that extends the pairwise-preference learning in imitation learning to both variable-length LM generation and utilizing the preference among multiple generations. For LM training, based on the amount of supervised data, we present two minimalist learning objectives that utilize the learned guidance. In experiments, our method performs 
    
[^80]: MiniSUPERB:轻量级自监督语音模型基准测试

    MiniSUPERB: Lightweight Benchmark for Self-supervised Speech Models. (arXiv:2305.19011v2 [eess.AS] UPDATED)

    [http://arxiv.org/abs/2305.19011](http://arxiv.org/abs/2305.19011)

    这篇论文提出了一个名为MiniSUPERB的轻量级基准测试，可以有效地评估自监督语音模型的性能，并在计算成本上比SUPERB更低。同时，该论文还研究了在少样本情况下评估SSL语音模型的性能变化。

    

    SUPERB被提出用于评估自监督学习（SSL）语音模型在各种任务上的泛化能力。然而，由于大型数据集和多样化任务，它导致了高计算成本。在本文中，我们引入了MiniSUPERB，一个轻量级基准测试，它以明显更低的计算成本有效地评估SSL语音模型并且结果可与SUPERB相比。我们精选代表性任务，采样数据集，并离线提取模型表示。我们的方法与SUPERB Paper和SUPERB Challenge分别达到0.954和0.982的斯皮尔曼等级相关系数。此外，我们在乘-累积操作（MACs）方面减少了97％的计算成本。此外，我们在少样本情况下评估SSL语音模型，并观察到其性能有显著变化。据我们所知，这是第一项研究同时考虑模型本身的计算成本和在基准测试上评估的成本。

    SUPERB was proposed to evaluate the generalizability of self-supervised learning (SSL) speech models across various tasks. However, it incurs high computational costs due to the large datasets and diverse tasks. In this paper, we introduce MiniSUPERB, a lightweight benchmark that efficiently evaluates SSL speech models with comparable results to SUPERB but lower computational costs significantly. We carefully select representative tasks, sample datasets, and extract model representations offline. Our approach achieves a Spearman's rank correlation of 0.954 and 0.982 with SUPERB Paper and SUPERB Challenge, respectively. Additionally, we reduce the computational cost by 97% in terms of Multiply-ACcumulate operations (MACs). Furthermore, we evaluate SSL speech models in few-shot scenarios and observe significant variations in their performance. To our knowledge, this is the first study to examine both the computational cost of the model itself and the cost of evaluating it on a benchmark.
    
[^81]: Bactrian-X: 具有低秩适应性的多语言可复制指令跟随模型

    Bactrian-X: Multilingual Replicable Instruction-Following Models with Low-Rank Adaptation. (arXiv:2305.15011v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.15011](http://arxiv.org/abs/2305.15011)

    Bactrian-X是一个多语言可复制指令跟随模型，利用低秩适应性（LoRA）训练，具有低参数数量、易于替换的特点。在综合多语言评估设置中，Bactrian-X模型在性能上优于纯模型和现有的指令调优模型。

    

    指令调优已经显示出提升大型语言模型性能的巨大潜力。然而，由于不同语言之间高质量指令-响应数据集的稀缺性，对多语言指令调优的研究一直受到限制。为了填补这一空白，我们提出了Bactrian-X，这是一个涵盖52种语言的综合多语言并行数据集，包含340万个指令-响应对。利用这个数据集，我们使用低秩适应性（LoRA）训练了一组适配器，它们是轻量级组件，能够无缝集成到大型语言模型中。这些适配器的参数数目显著低于基础模型，使它们可以轻松替换，并用作不同语言或语言组的插件。在各种多语言评估设置下进行的广泛实验证明，基于Bactrian-X上的LoRA训练获得的模型优于纯模型和现有的指令调优模型。代码和模型已经发布。

    Instruction tuning has shown great promise in improving the performance of large language models. However, research on multilingual instruction tuning has been limited due to the scarcity of high-quality instruction-response datasets across different languages. To bridge this gap, we present Bactrian-X, a comprehensive multilingual parallel dataset of 3.4 million instruction-response pairs across 52 languages. Leveraging this dataset, we train a set of adapters using low-rank adaptation (LoRA), which are lightweight components that seamlessly integrate with large language models. These adapters have a substantially lower parameter count than the base model, making them easily replaceable and usable as plug-ins for different languages or language groups. Extensive experiments in various multilingual evaluation settings demonstrate that models derived from LoRA-based training over Bactrian-X outperform both the vanilla models and existing instruction-tuned models. The code and models are
    
[^82]: LLMDet:一种第三方大型语言模型生成文本检测工具

    LLMDet: A Third Party Large Language Models Generated Text Detection Tool. (arXiv:2305.15004v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.15004](http://arxiv.org/abs/2305.15004)

    LLMDet是一个第三方大型语言模型生成文本检测工具，能够从特定的语言模型中确定生成文本的来源，并满足精细追踪、中间判断和快速检测的要求。

    

    大型语言模型（LLM）生成的文本与高质量的人工撰写文本非常相似，引发了对其在传播虚假信息和学术不端行为中的潜在滥用的担忧。因此，迫切需要一种高度实用的检测工具，能够准确识别给定文本的来源。然而，现有的检测工具通常依赖于对LLM的访问，并且只能区分机器生成和人工撰写的文本，未能满足精细追踪、中间判断和快速检测的要求。因此，我们提出了LLMDet，一种特定于模型的安全、高效、可扩展的检测工具，可以从特定的LLM（如GPT-2、OPT、LLaMA等）中获取文本。在LLMDet中，我们记录了显著n-gram的下一个标记概率作为特征，用于计算每个LLM的代理困惑度。通过联合分析LLM的代理困惑度，我们可以确定生成文本的来源。

    Generated texts from large language models (LLMs) are remarkably close to high-quality human-authored text, raising concerns about their potential misuse in spreading false information and academic misconduct. Consequently, there is an urgent need for a highly practical detection tool capable of accurately identifying the source of a given text. However, existing detection tools typically rely on access to LLMs and can only differentiate between machine-generated and human-authored text, failing to meet the requirements of fine-grained tracing, intermediary judgment, and rapid detection. Therefore, we propose LLMDet, a model-specific, secure, efficient, and extendable detection tool, that can source text from specific LLMs, such as GPT-2, OPT, LLaMA, and others. In LLMDet, we record the next-token probabilities of salient n-grams as features to calculate proxy perplexity for each LLM. By jointly analyzing the proxy perplexities of LLMs, we can determine the source of the generated text
    
[^83]: 在Transformer模型中编辑常识

    Editing Common Sense in Transformers. (arXiv:2305.14956v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.14956](http://arxiv.org/abs/2305.14956)

    本文探究了常识判断是否与Transformer模型中的可编辑参数存在因果关联，并通过改进编辑算法和层选择策略，在常识领域中提升了编辑效果，使得经过编辑的GPT-2模型在各测试集上的F1分数相较于最佳微调基线提高了10.97%和10.73%。

    

    在Transformer模型中直接编辑模型参数使得在无需重新训练的情况下可以更新黑盒模型成为可能。然而，这些编辑方法仅在关于百科知识且只有一个正确答案的陈述上进行了评估。尚未对多个正确答案的常识知识进行研究，例如，一个苹果可以是绿色或红色但不能是透明的，这对于提高Transformer模型的可靠性和实用性同样重要。本文研究了常识判断是否与Transformer模型中的可编辑参数存在因果关联，并给出了肯定的答案。我们发现直接应用MEMIT编辑算法导致性能不佳，并通过改变编辑标记和改进层选择策略，即$MEMIT_{CSK}$，改进了对常识领域的编辑效果。使用$MEMIT_{CSK}$编辑过的GPT-2 Large和XL模型在PEP3k和20Q测试集上的F1分数相较于最佳微调基线提高了10.97%和10.73%。

    Editing model parameters directly in Transformers makes updating black-box models possible without re-training (Meng et al., 2023). However, these editing methods have only been evaluated on statements about encyclopedic knowledge with a single correct answer. Commonsense knowledge with multiple correct answers, e.g., an apple can be green or red but not transparent, has not been studied but is as essential for enhancing transformers' reliability and usefulness. In this paper, we investigate whether commonsense judgments are causally associated with localized, editable parameters in Transformers, and we provide an affirmative answer. We find that directly applying the MEMIT editing algorithm results in sub-par performance and improve it for the commonsense domain by varying edit tokens and improving the layer selection strategy, i.e., $MEMIT_{CSK}$. GPT-2 Large and XL models edited using $MEMIT_{CSK}$ outperform best-fine-tuned baselines by 10.97% and 10.73% F1 scores on PEP3k and 20Q 
    
[^84]: Sophia：一种用于语言模型预训练的可扩展的随机二阶优化器

    Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training. (arXiv:2305.14342v2 [cs.LG] UPDATED)

    [http://arxiv.org/abs/2305.14342](http://arxiv.org/abs/2305.14342)

    Sophia是一种用于语言模型预训练的可扩展的二阶优化算法，使用对角Hessian作为预调节器，并进行元素级别的裁剪控制更新大小。

    

    鉴于语言模型预训练的巨大成本，优化算法的微小改进将会大大降低训练的时间和成本。Adam及其变种一直是最先进的，而更复杂的二阶（基于Hessian的）优化器往往会带来太多的每步开销。在这篇论文中，我们提出了Sophia，一种简单可扩展的二阶优化器，它使用轻量级估计的对角Hessian作为预调节器。更新步骤是梯度的移动平均值除以估计Hessian的移动平均值，然后进行元素级别的裁剪。裁剪控制了最坏情况下的更新大小，并控制了Hessian在轨迹上的非凸性和快速变化的负面影响。Sophia只在每几次迭代中估计对角Hessian，这几乎没有平均每步的时间和内存开销。在使用GPT m进行语言建模时，

    Given the massive cost of language model pre-training, a non-trivial improvement of the optimization algorithm would lead to a material reduction on the time and cost of training. Adam and its variants have been state-of-the-art for years, and more sophisticated second-order (Hessian-based) optimizers often incur too much per-step overhead. In this paper, we propose Sophia, Second-order Clipped Stochastic Optimization, a simple scalable second-order optimizer that uses a light-weight estimate of the diagonal Hessian as the pre-conditioner. The update is the moving average of the gradients divided by the moving average of the estimated Hessian, followed by element-wise clipping. The clipping controls the worst-case update size and tames the negative impact of non-convexity and rapid change of Hessian along the trajectory. Sophia only estimates the diagonal Hessian every handful of iterations, which has negligible average per-step time and memory overhead. On language modeling with GPT m
    
[^85]: 基于技能的少样本选择以适应上下文学习

    Skill-Based Few-Shot Selection for In-Context Learning. (arXiv:2305.14210v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.14210](http://arxiv.org/abs/2305.14210)

    本文提出了Skill-KNN，一种基于技能的少样本选择方法，用于上下文学习。它解决了现有方法的偏见问题并且不需要训练模型，适用于不断扩展或更改示例库的情况。

    

    在上下文学习中，通过提供一些示例来适应下游任务是一种范例。针对每个测试实例选择适当的示例是上下文学习中很重要的。在本文中，我们提出了一种基于技能的少样本选择方法，用于上下文学习。Skill-KNN的关键优势包括：（1）它解决了现有基于预训练嵌入的方法容易因为不重要的自然语言表面特征对目标任务产生偏见的问题；（2）它不需要训练或微调任何模型，适用于频繁扩展或更改示例库的情况。其关键见解是优化输入到嵌入模型中，而不是调整模型本身。在技术上，Skill-KNN通过利用预处理少样本提示为每个测试案例和候选示例生成基于技能的描述，从而消除了不重要的信息。

    In-context learning is the paradigm that adapts large language models to downstream tasks by providing a few examples. Few-shot selection -- selecting appropriate examples for each test instance separately -- is important for in-context learning. In this paper, we propose Skill-KNN, a skill-based few-shot selection method for in-context learning. The key advantages of Skill-KNN include: (1) it addresses the problem that existing methods based on pre-trained embeddings can be easily biased by surface natural language features that are not important for the target task; (2) it does not require training or fine-tuning of any models, making it suitable for frequently expanding or changing example banks. The key insight is to optimize the inputs fed into the embedding model, rather than tuning the model itself. Technically, Skill-KNN generates the skill-based descriptions for each test case and candidate example by utilizing a pre-processing few-shot prompting, thus eliminating unimportant 
    
[^86]: 超越共享词汇：增加多语言机器翻译中的表示词语相似性

    Beyond Shared Vocabulary: Increasing Representational Word Similarities across Languages for Multilingual Machine Translation. (arXiv:2305.14189v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.14189](http://arxiv.org/abs/2305.14189)

    本文提出了一种超越共享词汇的方法，通过定义词级信息传输路径和使用图网络来融合跨语言的词嵌入，实现了在多语言机器翻译中提高相似含义词的对齐性和BLEU分数的一致提升。此方法只需要少量额外参数且计算成本增加有限，并且推理时间与基线相同。

    

    在多语言神经机器翻译(MNMT)中，使用共享的词汇是常见的做法。除了简单的设计外，共享标记在积极的知识转移中起着重要的作用，假设共享标记在不同语言中指的是相似的含义。然而，当词汇的重叠较小时，尤其是由于不同的书写系统，转移被限制。在本文中，我们通过词等价类定义了词级信息传输路径，并依赖图网络来融合跨语言的词嵌入。我们的实验证明了我们方法的优势：1) 具有相似含义的词的嵌入在不同语言中更好地对齐，2) 我们的方法在高和低资源MNMT方面实现了一致的BLEU提升达2.3个点，3) 需要少于1.0%的额外可训练参数，并且计算成本的增加有限，而推理时间与基线相同。

    Using a vocabulary that is shared across languages is common practice in Multilingual Neural Machine Translation (MNMT). In addition to its simple design, shared tokens play an important role in positive knowledge transfer, assuming that shared tokens refer to similar meanings across languages. However, when word overlap is small, especially due to different writing systems, transfer is inhibited. In this paper, we define word-level information transfer pathways via word equivalence classes and rely on graph networks to fuse word embeddings across languages. Our experiments demonstrate the advantages of our approach: 1) embeddings of words with similar meanings are better aligned across languages, 2) our method achieves consistent BLEU improvements of up to 2.3 points for high- and low-resource MNMT, and 3) less than 1.0\% additional trainable parameters are required with a limited increase in computational costs, while inference time remains identical to the baseline. We release the c
    
[^87]: 能ChatGPT保持对真相的信念吗？通过辩论评估LLM的推理能力。

    Can ChatGPT Defend its Belief in Truth? Evaluating LLM Reasoning via Debate. (arXiv:2305.13160v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.13160](http://arxiv.org/abs/2305.13160)

    本研究通过辩论式对话测试了LLM的推理能力，要求LLM在讨论过程中不仅能够得出正确答案，还能够坚持并捍卫自己的信念，从而更深入地评估其对问题的推理理解能力。

    

    ChatGPT和GPT-4等大型语言模型在复杂推理任务中表现出色。然而，我们很难确定这些模型是基于对真相和逻辑的深入理解进行推理，还是仅仅利用其记忆的模式进行相对肤浅的推理。在本研究中，我们通过与LLM进行类似辩论的对话来测试其推理能力。在给定一个问题的情况下，LLM和用户需要讨论并从相对立的观点出发做出正确的决策。通过消除Clever Hans效应，我们的任务要求LLM不仅能够独立得出正确答案，还要能够坚持并捍卫自己的信念，而不是盲目地相信用户的（无效的）论证和批评，从而更深入地测试LLM是否掌握了解决问题所需的推理要点。我们在涵盖了数学、常识、逻辑和BIG-Bench等多个复杂推理基准测试中进行了实验。

    Large language models (LLMs) such as ChatGPT and GPT-4 have shown impressive performance in complex reasoning tasks. However, it is difficult to know whether the models are reasoning based on deep understandings of truth and logic, or leveraging their memorized patterns in a relatively superficial way. In this work, we explore testing LLMs' reasoning by engaging with them in a debate-like conversation, where given a question, the LLM and the user need to discuss to make the correct decision starting from opposing arguments. Upon mitigating the Clever Hans effect, our task requires the LLM to not only achieve the correct answer on its own, but also be able to hold and defend its belief instead of blindly believing or getting misled by the user's (invalid) arguments and critiques, thus testing in greater depth whether the LLM grasps the essence of the reasoning required to solve the problem. Across a range of complex reasoning benchmarks spanning math, commonsense, logic and BIG-Bench ta
    
[^88]: 通过提示语训练可解释的风格嵌入：学习风格嵌入

    Learning Interpretable Style Embeddings via Prompting LLMs. (arXiv:2305.12696v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.12696](http://arxiv.org/abs/2305.12696)

    本文通过提示语在大量文本上进行风格分析，从而创建了可解释的风格嵌入表示，为需要可解释性的下游应用（如作者归属）提供了重要资源。

    

    风格表示学习可以为文本中的作者风格建立与内容无关的表示。风格分析通常由专业的法医语言学家进行，尚无大型的风格分析注释数据集可供训练。当前的风格表示学习使用神经方法来将风格与内容区分开来创建风格向量，但这些方法产生了难以解释的表示，使得在作者归属等下游应用中的审核和可解释性变得复杂。在这项工作中，我们使用提示语在大量文本上执行风格分析，创建了一个合成的数据集，并训练出了我们称之为LISA嵌入的可解释风格表示。我们将我们的合成风格分析数据集和可解释风格模型发布为资源。

    Style representation learning builds content-independent representations of author style in text. Stylometry, the analysis of style in text, is often performed by expert forensic linguists and no large dataset of stylometric annotations exists for training. Current style representation learning uses neural methods to disentangle style from content to create style vectors, however, these approaches result in uninterpretable representations, complicating their usage in downstream applications like authorship attribution where auditing and explainability is critical. In this work, we use prompting to perform stylometry on a large number of texts to create a synthetic dataset and train human-interpretable style representations we call LISA embeddings. We release our synthetic stylometry dataset and our interpretable style models as resources.
    
[^89]: 表面相似性之下：大型语言模型通过结构推理进行合理的科学类比

    Beneath Surface Similarity: Large Language Models Make Reasonable Scientific Analogies after Structure Abduction. (arXiv:2305.12660v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2305.12660](http://arxiv.org/abs/2305.12660)

    本论文介绍了一种基于结构推理的类比结构推断任务，旨在解决大型语言模型在进行科学类比时忽视结构的问题。

    

    人类认知中类比推理的重要作用使我们能够通过共享的关系结构将新概念与熟悉的概念联系起来。尽管先前的研究关注于词语类比，但本研究表明，大型语言模型（LLMs）经常忽视构成这些类比的结构，这引发了对词语类比作为类比推理技能（类似于人类认知）的有效性的质疑。为了解决这个问题，我们的论文引入了一种基于认知心理学的类比结构推理任务，旨在推断出连接两个系统之间的类比结构。为了支持这个任务，我们建立了一个名为SCAR的基准，包含来自13个不同领域的400个科学类比，旨在评估利用结构推理的类比推理能力。实证证据强调了LLMs，包括ChatGPT和GPT-4，在掌握这个任务上依然面临的挑战。

    The vital role of analogical reasoning in human cognition allows us to grasp novel concepts by linking them with familiar ones through shared relational structures. Despite the attention previous research has given to word analogies, this work suggests that Large Language Models (LLMs) often overlook the structures that underpin these analogies, raising questions about the efficacy of word analogies as a measure of analogical reasoning skills akin to human cognition. In response to this, our paper introduces a task of analogical structure abduction, grounded in cognitive psychology, designed to abduce structures that form an analogy between two systems. In support of this task, we establish a benchmark called SCAR, containing 400 scientific analogies from 13 distinct fields, tailored for evaluating analogical reasoning with structure abduction. The empirical evidence underlines the continued challenges faced by LLMs, including ChatGPT and GPT-4, in mastering this task, signifying the n
    
[^90]: 压缩，然后提示：使用可转移提示来改善LLM推理的准确性和效率平衡

    Compress, Then Prompt: Improving Accuracy-Efficiency Trade-off of LLM Inference with Transferable Prompt. (arXiv:2305.11186v1 [cs.CL])

    [http://arxiv.org/abs/2305.11186](http://arxiv.org/abs/2305.11186)

    本文提出了使用可转移提示来优化压缩的LLMs的准确性和效率的平衡问题。该方法通过选择精度更高的提示显著提高了压缩的LLM在特定查询方面的生成质量，并实现了4倍推理时间加速。

    

    大型语言模型（LLMs）具有数十亿的参数，表现出在各种自然语言处理（NLP）任务中的卓越性能。然而，它们在推理过程中会带来显着的计算挑战，尤其是在常见的硬件上部署（例如单个GPU）时。因此，通过压缩来最小化LLM推理的延迟，即减少计算和内存需求，变得至关重要。但是，此过程必然引发效率和精度之间的平衡，因为压缩的LLMs通常会经历预测精度下降。在这项研究中，我们提出了一个创新的视角：为了优化这种平衡，压缩的LLMs需要一种不同于原始模型的独特输入格式。我们的研究结果表明，通过选择具有精度的提示，可以显著改善压缩的LLM在特定查询方面的生成质量。基于这一发现，我们提出了一个可转移提示的方法，该方法训练模型预测有效提示。实证上，我们的方法可以在各种语言任务上生成高质量的输出，并实现了4倍速度的推理时间加速，同时保持竞争性的准确性。

    Large Language Models (LLMs), armed with billions of parameters, exhibit exceptional performance across a wide range of Natural Language Processing (NLP) tasks. However, they present a significant computational challenge during inference, especially when deploying on common hardware such as single GPUs. As such, minimizing the latency of LLM inference by curtailing computational and memory requirements, though achieved through compression, becomes critically important. However, this process inevitably instigates a trade-off between efficiency and accuracy, as compressed LLMs typically experience a reduction in predictive precision. In this research, we introduce an innovative perspective: to optimize this trade-off, compressed LLMs require a unique input format that varies from that of the original models. Our findings indicate that the generation quality in a compressed LLM can be markedly improved for specific queries by selecting prompts with precision. Capitalizing on this insight,
    
[^91]: 旨在总结带有层次关系的多篇文档

    Towards Summarizing Multiple Documents with Hierarchical Relationships. (arXiv:2305.01498v1 [cs.CL])

    [http://arxiv.org/abs/2305.01498](http://arxiv.org/abs/2305.01498)

    提出了一个新的数据集PeerSum用于生成科学论文的元评论，源文档具有显式层次结构的丰富文档间关系，提出了一种用于元评论生成的关系感知多任务模型Rammer。

    

    多数现存的多文档摘要(MDS)数据集缺少人工生成的、真实的(即非合成的)摘要或者带有显式文档间关系的源文档。为了增强MDS系统的能力，我们提出PeerSum，这是一个新颖的数据集，用于生成科学论文的元评论，其中元评论是对评论和相应讨论的高度概括且真实的摘要。这些源文档具有显式层次结构的丰富文档间关系，包括交叉引用和经常出现的冲突。鉴于很少有研究采用基于预训练语言模型的注意力操纵来将层次关系纳入MDS系统中，我们还提出了Rammer(关系感知多任务元评论生成器)，这是一种元评论生成模型，使用基于层次关系的稀疏注意力和多任务目标，可以预测多个度量值。

    Most existing multi-document summarization (MDS) datasets lack human-generated and genuine (i.e., not synthetic) summaries or source documents with explicit inter-document relationships that a summary must capture. To enhance the capabilities of MDS systems we present PeerSum, a novel dataset for generating meta-reviews of scientific papers, where the meta-reviews are highly abstractive and genuine summaries of reviews and corresponding discussions. These source documents have rich inter-document relationships of an explicit hierarchical structure with cross-references and often feature conflicts. As there is a scarcity of research that incorporates hierarchical relationships into MDS systems through attention manipulation on pre-trained language models, we additionally present Rammer (Relationship-aware Multi-task Meta-review Generator), a meta-review generation model that uses sparse attention based on the hierarchical relationships and a multi-task objective that predicts several me
    
[^92]: 采用ChatGPT进行摘要提取以生成忠实的摘要

    Extractive Summarization via ChatGPT for Faithful Summary Generation. (arXiv:2304.04193v1 [cs.CL])

    [http://arxiv.org/abs/2304.04193](http://arxiv.org/abs/2304.04193)

    本文评估了ChatGPT在抽取式摘要任务中的性能，并发现其ROUGE得分相对于传统微调方法仍有待提高。研究探索了上下文学习和思维链推理对其性能提升的有效性，并通过采用先抽取后生成流程和句子选择模块缓解了其生成摘要的忠实性问题。

    

    抽取式摘要是自然语言处理中至关重要的任务，旨在通过直接提取句子将长文档压缩为较短的版本。ChatGPT的引入引起了自然语言处理界的广泛关注，因为它在各种下游任务中具有出色的表现。然而，关于其事实性和忠实度的担忧阻碍了其在摘要系统中的实际应用。本文首先对ChatGPT在抽取式摘要上的性能进行了全面评估，并将其与传统的微调方法在各种基准数据集上进行了比较。我们的实验分析揭示，相对于现有的监督系统，ChatGPT在ROUGE分数方面的抽取式摘要性能仍然不足。此外，我们探索了上下文学习和思维链推理对其性能提升的有效性。此外，我们发现采用先抽取后生成流程以及句子选择模块可以缓解ChatGPT生成摘要的忠实性问题。总体而言，我们的研究为使用ChatGPT进行抽取式摘要提供了局限性和潜在改进的见解。

    Extractive summarization is a crucial task in natural language processing that aims to condense long documents into shorter versions by directly extracting sentences. The recent introduction of ChatGPT has attracted significant interest in the NLP community due to its remarkable performance on a wide range of downstream tasks. However, concerns regarding factuality and faithfulness have hindered its practical applications for summarization systems. This paper first presents a thorough evaluation of ChatGPT's performance on extractive summarization and compares it with traditional fine-tuning methods on various benchmark datasets. Our experimental analysis reveals that ChatGPT's extractive summarization performance is still inferior to existing supervised systems in terms of ROUGE scores. In addition, we explore the effectiveness of in-context learning and chain-of-thought reasoning for enhancing its performance. Furthermore, we find that applying an extract-then-generate pipeline with 
    
[^93]: 论ChatGPT和情感增强提示在心理健康分析中的评估

    On the Evaluations of ChatGPT and Emotion-enhanced Prompting for Mental Health Analysis. (arXiv:2304.03347v1 [cs.CL])

    [http://arxiv.org/abs/2304.03347](http://arxiv.org/abs/2304.03347)

    本文全面评估了ChatGPT在心理健康分析和情感推理方面的表现，以及不同提示策略和情感信息对其性能的影响。结果显示，ChatGPT在心理健康分析方面表现良好，加入情感增强提示对某些任务效果显著。

    

    自动化心理健康分析显示出提高心理健康护理效率和可访问性的巨大潜力，而最近的主流方法利用预训练的语言模型(PLMs)作为骨干，并融入情感信息。最新的大型语言模型(LLMs)，如ChatGPT，在各种自然语言处理任务上表现出惊人的能力。然而，现有的ChatGPT零-shot性能研究在不充分的评估、情感信息利用和方法可解释性方面存在局限性。本文全面评估了ChatGPT在11个数据集上的心理健康分析和情感推理能力，涵盖了5个任务，包括二进制和多类心理健康状况检测、心理健康状况的原因/因素检测、对话中的情绪识别和因果情感蕴含。我们在实证分析中探究了不同提示策略以及情感增强提示对ChatGPT性能的影响。结果表明，ChatGPT在心理健康分析中有着良好的表现，并且在某些任务中加入情感增强提示效果显著。

    Automated mental health analysis shows great potential for enhancing the efficiency and accessibility of mental health care, whereas the recent dominant methods utilized pre-trained language models (PLMs) as the backbone and incorporated emotional information. The latest large language models (LLMs), such as ChatGPT, exhibit dramatic capabilities on diverse natural language processing tasks. However, existing studies on ChatGPT's zero-shot performance for mental health analysis have limitations in inadequate evaluation, utilization of emotional information, and explainability of methods. In this work, we comprehensively evaluate the mental health analysis and emotional reasoning ability of ChatGPT on 11 datasets across 5 tasks, including binary and multi-class mental health condition detection, cause/factor detection of mental health conditions, emotion recognition in conversations, and causal emotion entailment. We empirically analyze the impact of different prompting strategies with 
    
[^94]: 人类合作是否增强了识别LLM生成的深度伪造文本的准确性？

    Does Human Collaboration Enhance the Accuracy of Identifying LLM-Generated Deepfake Texts?. (arXiv:2304.01002v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2304.01002](http://arxiv.org/abs/2304.01002)

    这项研究研究了人类合作是否增强了识别LLM生成的深度伪造文本的准确性。结果表明，合作可以潜在地提高两组人对深度伪造文本的检测准确性。

    

    大型语言模型（如GPT-4、LLaMA）的进展改善了大规模生成类似人类写作的连贯句子，从而产生了所谓的深度伪造文本。然而，这一进展引发了安全和隐私的担忧，需要有效解决方案来区分深度伪造文本和人类书写文本。尽管以前的研究探讨了人类检测深度伪造文本的能力，但没有研究“合作”是否能提高深度伪造文本的检测。在本研究中，为了填补对深度伪造文本理解的空白，我们对两组人进行了实验：（1）来自AMT平台的非专家群体和（2）来自Upwork平台的写作专家。结果表明，人类之间的合作可能会提高两组对深度伪造文本的检测准确性，非专家组的检测准确性提高了6.36%，专家组的检测准确性提高了12.76%。

    Advances in Large Language Models (e.g., GPT-4, LLaMA) have improved the generation of coherent sentences resembling human writing on a large scale, resulting in the creation of so-called deepfake texts. However, this progress poses security and privacy concerns, necessitating effective solutions for distinguishing deepfake texts from human-written ones. Although prior works studied humans' ability to detect deepfake texts, none has examined whether "collaboration" among humans improves the detection of deepfake texts. In this study, to address this gap of understanding on deepfake texts, we conducted experiments with two groups: (1) nonexpert individuals from the AMT platform and (2) writing experts from the Upwork platform. The results demonstrate that collaboration among humans can potentially improve the detection of deepfake texts for both groups, increasing detection accuracies by 6.36% for non-experts and 12.76% for experts, respectively, compared to individuals' detection accur
    
[^95]: Reflexion：具有动态记忆和自我反思的自主智能体

    Reflexion: an autonomous agent with dynamic memory and self-reflection. (arXiv:2303.11366v1 [cs.AI])

    [http://arxiv.org/abs/2303.11366](http://arxiv.org/abs/2303.11366)

    本文提出 Reflexion 方法，给智能体赋予了动态记忆和自我反思能力，以增强其任务特定的行动选择能力。

    

    最近决策大型语言模型（LLM）代理的发展在各种基准测试中展现出卓越的性能。然而，这些最先进的方法通常需要内部模型微调、外部模型微调或在定义的状态空间上进行策略优化。由于高质量训练数据的稀缺性或缺乏良好定义的状态空间，实现这些方法可能会具有挑战性。此外，这些代理没有人类决策过程固有的某些品质，特别是从错误中学习的能力。通过反思，人类可以通过试错过程高效地解决新的问题。在最近的研究基础上，我们提出 Reflexion，一种将动态记忆和自我反思能力赋予智能体的方法，以增强其现有的推理轨迹和任务特定的行动选择能力。为了实现完全自动化，我们介绍了一种简单而有效的方法。

    Recent advancements in decision-making large language model (LLM) agents have demonstrated impressive performance across various benchmarks. However, these state-of-the-art approaches typically necessitate internal model fine-tuning, external model fine-tuning, or policy optimization over a defined state space. Implementing these methods can prove challenging due to the scarcity of high-quality training data or the lack of well-defined state space. Moreover, these agents do not possess certain qualities inherent to human decision-making processes, specifically the ability to learn from mistakes. Self-reflection allows humans to efficiently solve novel problems through a process of trial and error. Building on recent research, we propose Reflexion, an approach that endows an agent with dynamic memory and self-reflection capabilities to enhance its existing reasoning trace and task-specific action choice abilities. To achieve full automation, we introduce a straightforward yet effective 
    
[^96]: 多输入假设和受限解码空间的 N-best T5：利用多输入假设和受限解码空间的强鲁棒 ASR 错误校正

    N-best T5: Robust ASR Error Correction using Multiple Input Hypotheses and Constrained Decoding Space. (arXiv:2303.00456v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2303.00456](http://arxiv.org/abs/2303.00456)

    本文提出了一种新的 N-best T5 模型，该模型利用 ASR N-best 列表作为输入，并使用受限解码过程，从而实现了针对 ASR 错误修正的强鲁棒性能。

    

    错误修正模型是自动语音识别（ASR）后处理的重要部分，用于提高转录的可读性和质量。本文提出了一种新的 N-best T5 模型，该模型利用 ASR N-best 列表作为模型的输入。通过从预训练语言模型中转移知识并从 ASR 解码空间中获取更丰富的信息，所提出的方法优于强Conformer-Transducer基线。标准错误校正的另一个问题是生成过程不受良好指导。为了解决这个问题，使用基于 N-best 列表或 ASR lattice 的受限解码过程，可以传播额外的信息。

    Error correction models form an important part of Automatic Speech Recognition (ASR) post-processing to improve the readability and quality of transcriptions. Most prior works use the 1-best ASR hypothesis as input and therefore can only perform correction by leveraging the context within one sentence. In this work, we propose a novel N-best T5 model for this task, which is fine-tuned from a T5 model and utilizes ASR N-best lists as model input. By transferring knowledge from the pre-trained language model and obtaining richer information from the ASR decoding space, the proposed approach outperforms a strong Conformer-Transducer baseline. Another issue with standard error correction is that the generation process is not well-guided. To address this a constrained decoding process, either based on the N-best list or an ASR lattice, is used which allows additional information to be propagated.
    
[^97]: 多模态语音识别用于语言引导的合身代理

    Multimodal Speech Recognition for Language-Guided Embodied Agents. (arXiv:2302.14030v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.14030](http://arxiv.org/abs/2302.14030)

    本文提出在语言引导的合身代理中使用多模态ASR模型来减少语音指令的转录错误，并通过考虑视觉上下文来提高转录结果。模型能够恢复多达30％的被屏蔽单词，并且训练基于文本的合身代理在遵循多模态ASR模型转录的指令下能够更高效地完成任务。

    

    语言引导的合身代理的基准测试通常假设基于文本的指令，但实际部署的代理会遇到口头指令。尽管自动语音识别（ASR）模型可以弥合语音输入差距，但错误的ASR转录会损害代理完成任务的能力。本文提出利用多模态ASR模型训练来考虑伴随的视觉环境，从而减少语音指令的转录错误。我们在ALFRED任务完成数据集上合成了口头指令的数据集，并通过系统性地屏蔽口头单词来模拟声学噪声。我们发现，利用视觉观察可以有助于恢复屏蔽的单词，通过多模态ASR模型，恢复的屏蔽单词比单模态基线多达30％。我们还发现，基于文本训练的合身代理通过遵循多模态ASR模型的转录指令更经常地完成任务。

    Benchmarks for language-guided embodied agents typically assume text-based instructions, but deployed agents will encounter spoken instructions. While Automatic Speech Recognition (ASR) models can bridge the input gap, erroneous ASR transcripts can hurt the agents' ability to complete tasks. In this work, we propose training a multimodal ASR model to reduce errors in transcribing spoken instructions by considering the accompanying visual context. We train our model on a dataset of spoken instructions, synthesized from the ALFRED task completion dataset, where we simulate acoustic noise by systematically masking spoken words. We find that utilizing visual observations facilitates masked word recovery, with multimodal ASR models recovering up to 30% more masked words than unimodal baselines. We also find that a text-trained embodied agent successfully completes tasks more often by following transcribed instructions from multimodal ASR models. github.com/Cylumn/embodied-multimodal-asr
    
[^98]: 通过定向刺激引导大型语言模型

    Guiding Large Language Models via Directional Stimulus Prompting. (arXiv:2302.11520v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2302.11520](http://arxiv.org/abs/2302.11520)

    该文介绍了一个新的框架，用于通过生成定向刺激来指导大型语言模型在下游任务中生成所需的输出。通过策略语言模型的训练，该框架可以适应于各种语言模型和任务，并在摘要和对话生成任务中取得了最先进的表现。

    

    我们引入了一个新的框架，称为定向刺激引导，它使用可调节的语言模型来为下游任务的黑盒冻结大型语言模型提供指导。与以往手动或自动找到每个任务的最优提示的方法不同，我们训练一个策略语言模型来生成离散的token作为每个输入的定向刺激，这是一种提示或提示，例如文章的关键词用于摘要。然后将定向刺激与原始输入组合，并输入到LLM中，以指导其向所需目标生成。策略LM可以通过1）从注释数据中的监督学习和2）从离线和在线奖励中的强化学习进行训练，以探索更好地与人类偏好相匹配的定向刺激。该框架可灵活适用于各种LM和任务。为了验证其效果，我们将我们的框架应用于摘要和对话响应生成任务。实验结果表明，我们的方法在两个任务上均取得了最先进的性能。

    We introduce a new framework, Directional Stimulus Prompting, that uses a tuneable language model (LM) to provide guidance for the black-box frozen large language model (LLM) on downstream tasks. Unlike prior work that manually or automatically finds the optimal prompt for each task, we train a policy LM to generate discrete tokens as directional stimulus of each input, which is a hint/cue such as keywords of an article for summarization. The directional stimulus is then combined with the original input and fed into the LLM to guide its generation toward the desired target. The policy LM can be trained through 1) supervised learning from annotated data and 2) reinforcement learning from offline and online rewards to explore directional stimulus that better aligns LLMs with human preferences. This framework is flexibly applicable to various LMs and tasks. To verify its effectiveness, we apply our framework to summarization and dialogue response generation tasks. Experimental results dem
    
[^99]: 理解跨语言摘要中的“翻译语言”现象

    Understanding Translationese in Cross-Lingual Summarization. (arXiv:2212.07220v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.07220](http://arxiv.org/abs/2212.07220)

    本论文研究了跨语言摘要中的“翻译语言”现象对模型的影响，发现不同方法构建数据集会导致不同程度的翻译语言现象。文章还发现翻译语言会对模型的评估和性能产生影响，包括测试集中的翻译语言可能导致人工判断与自动评估之间的差异，以及训练集中的翻译语言会影响模型的训练表现和性能。

    

    在跨语言摘要（CLS）中，给定一篇源语言文档，旨在生成一篇简洁的目标语言摘要。与单语摘要（MS）不同，自然出现的源语言文档配对目标语言摘要并不常见。为了收集大规模的CLS数据，现有数据集通常在创建过程中涉及翻译。然而，翻译文本与原始语言文本之间存在着区别，即翻译语言。本文首先确认了构建CLS数据集的不同方法将导致不同程度的翻译语言现象。然后我们系统地研究了翻译语言在源文档或目标摘要中出现时如何影响CLS模型的评估和性能。具体而言，我们发现（1）测试集中文档或摘要中的翻译语言可能导致人工判断与自动评估之间的差异；（2）训练集中的翻译语言会影响模型的训练表现和性能。

    Given a document in a source language, cross-lingual summarization (CLS) aims at generating a concise summary in a different target language. Unlike monolingual summarization (MS), naturally occurring source-language documents paired with target-language summaries are rare. To collect large-scale CLS data, existing datasets typically involve translation in their creation. However, the translated text is distinguished from the text originally written in that language, i.e., translationese. In this paper, we first confirm that different approaches of constructing CLS datasets will lead to different degrees of translationese. Then we systematically investigate how translationese affects CLS model evaluation and performance when it appears in source documents or target summaries. In detail, we find that (1) the translationese in documents or summaries of test sets might lead to the discrepancy between human judgment and automatic evaluation; (2) the translationese in training sets would ha
    
[^100]: 细调BERT模型中的命名实体记忆

    Memorization of Named Entities in Fine-tuned BERT Models. (arXiv:2212.03749v2 [cs.CL] UPDATED)

    [http://arxiv.org/abs/2212.03749](http://arxiv.org/abs/2212.03749)

    本文研究了细调BERT模型中命名实体的记忆程度，并采用差分隐私进行实验。实验结果表明，应用差分隐私会对模型的性能产生不利影响。

    

    隐私保护深度学习是机器学习中的新兴领域，旨在减轻深度神经网络在使用中的隐私风险。其中一个风险是从训练在个人和隐私敏感信息数据集上的语言模型中提取训练数据。在我们的研究中，我们调查了细调BERT模型中命名实体记忆的程度。我们使用单标签文本分类作为代表性的下游任务，在实验中采用三种不同的细调设置，包括一种差分隐私（DP）设置。我们利用自定义的顺序抽样策略和两种提示策略从细调BERT模型中创建了大量的文本样本。我们在这些样本中搜索命名实体，并查看它们是否也存在于细调数据集中。我们在电子邮件和博客领域使用了两个基准数据集进行实验。我们表明，DP的应用对测试性能产生了不利影响。

    Privacy preserving deep learning is an emerging field in machine learning that aims to mitigate the privacy risks in the use of deep neural networks. One such risk is training data extraction from language models that have been trained on datasets, which contain personal and privacy sensitive information. In our study, we investigate the extent of named entity memorization in fine-tuned BERT models. We use single-label text classification as representative downstream task and employ three different fine-tuning setups in our experiments, including one with Differentially Privacy (DP). We create a large number of text samples from the fine-tuned BERT models utilizing a custom sequential sampling strategy with two prompting strategies. We search in these samples for named entities and check if they are also present in the fine-tuning datasets. We experiment with two benchmark datasets in the domains of emails and blogs. We show that the application of DP has a detrimental effect on the te
    

