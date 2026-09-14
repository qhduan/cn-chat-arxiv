# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Type Diversity Enables Transformers to Generalise Compositionally](https://arxiv.org/abs/2609.13144) | 本研究提出Transformer在结构泛化上表现较差并非其固有缺陷，而是数据集中结构类型多样性偏低所致——提高类型多样性（无论是词汇类型还是结构类型）均可同等程度地提升组合泛化能力。 |
| [^2] | [SAS: Simple Attention Sparsification via End-to-End Optimization of Context Ranking](https://arxiv.org/abs/2609.13141) | SAS提出了一种门控稀疏注意力机制，通过与语言建模损失端到端地优化上下文排序，使注意力预算直接分配给对预测最有影响的上下文单元，克服了以往方法依赖稠密注意力蒸馏导致的预算错配问题。 |
| [^3] | [Continue, Adapt, or Yield: In-Turn Adaptation to Overlapping Speech in Full-Duplex Agents](https://arxiv.org/abs/2609.13117) | 本文提出Duplex Cue评估基准，首次系统评估全双工语音智能体的“轮内适应”能力——即在继续说话的同时吸纳听者补充内容（如纠正、澄清），这一超越了“继续说话或停止”二元决策的第三种人类常见回应方式。 |
| [^4] | [MP-Bench: Evaluating Voice Agents as a Multiparty Conversation Participant](https://arxiv.org/abs/2609.13076) | 该论文提出了MP-Bench，这是首个专门用于客观评估语音代理作为多方对话积极参与者能力的基准测试，填补了现有基准只关注二元交互的空白。 |
| [^5] | [MAxBench: A Multinomial Concept Recovery Benchmark](https://arxiv.org/abs/2609.13072) | 本文提出了MAxBench，一个几何无关的多分类概念表示评估框架，通过在6个概念和4个模型上系统比较10种定位方法（涵盖5种几何类型），为识别最适合多分类概念引导的表示几何结构及最有效的恢复方法提供了统一评估标准。 |
| [^6] | [Expert-Space Exploration in MoE Reinforcement Learning](https://arxiv.org/abs/2609.13058) | 提出ESRL框架，通过架构感知的方式显式探索MoE模型的专家路由空间来增加采样多样性，从而提升强化学习训练效果，同时避免直接扰动激活不合适专家所导致的采样质量下降问题。 |
| [^7] | [Kraken: LLM-based Speech-to-Speech Translation via Low-bitrate VQ and Dual-path Source Conditioning](https://arxiv.org/abs/2609.13045) | Kraken模型通过采用基于单层矢量量化的低比特率语音token以及以源语音为条件的双路径Autowave-X解码器，解决了LLM预测高比特率语音token困难和对训练数据说话人身份与韵律严格对齐的依赖问题，从而提升了语音到语音翻译的性能。 |
| [^8] | [Tasks over Application Manuals: Revealing Gaps in Long-Horizon Procedural Reasoning for Language Models](https://arxiv.org/abs/2609.13005) | 提出了TAM基准，通过ICD-10-CM临床编码和美国联邦量刑两个真实领域任务，揭示大型语言模型在遵循数百页权威手册进行长程程序性推理时存在的显著能力差距。 |
| [^9] | [Judging by the Cover: Cleaning LLM Truthfulness Benchmarks to Avoid Surface-Level Feature Leakage](https://arxiv.org/abs/2609.13003) | 该研究揭示TruthfulQA等二选一真实性基准存在表层特征泄漏问题，模型无需真正推理即可借助答案表面特征超越随机水平，作者据此提出Audit-Prune清理机制并发布了泄漏接近随机水平的清理版数据集。 |
| [^10] | [Investigating Temporal Motion Features for Pose-to-Text Indian Sign Language Translation](https://arxiv.org/abs/2609.12993) | 在WSLP 2026印度手语翻译共享任务中，为轻量级T5-small模型添加显式帧间运动特征带来了最大的性能提升，显著改善了BLEU分数。 |
| [^11] | [Fewer Words, Not Fewer Tokens: Measuring the Sanskrit Tokenization Penalty per Proposition](https://arxiv.org/abs/2609.12960) | 该研究发现，尽管梵语训练的BPE分词器看似每个命题比英语更便宜，但在与同规模英语对照组严格对比后，梵语在每个意义单位上仍然承担显著的分词惩罚。 |
| [^12] | [PA-CDM: Position-Aware Character Detection Matching for Evaluating Handwritten Mathematical Expression Recognition](https://arxiv.org/abs/2609.12917) | 提出了PA-CDM位置感知评估指标，通过将字符检测匹配与位置森林编码和散度级加权相结合，解决了现有手写数学表达式识别评估方法无法感知错误发生位置的缺陷。 |
| [^13] | [LLM-Enhanced Dual-Branch Learning for Large-Scale Multi-Label Text Classification](https://arxiv.org/abs/2609.12915) | 该论文提出DualMLC双分支框架，同时利用自回归解码器式大语言模型和双向编码器对同一文档建模，并通过后期logit融合两分支得分，充分挖掘异构语言模型的互补性，以提升大规模多标签文本分类的性能。 |
| [^14] | [Parameter-Efficient Retrievers for Polish and European Languages](https://arxiv.org/abs/2609.12913) | 提出了一种结合跨语言对齐、关系知识蒸馏和对比微调的三阶段训练流程，无需原始相关性标注即可训练出参数量小但性能可媲美大型模型的波兰语和欧洲多语言稠密检索器。 |
| [^15] | [MedSNIP: Building and Benchmarking Snippet-Level Granularity for Medical Fact Verification](https://arxiv.org/abs/2609.12884) | 该论文将医学事实核查重新构建为片段级验证范式，并提出了人工标注基准MedSNIP-Bench与自动片段生成流水线MedSNIP，以保留声明周围的局部临床结构并解决原子级分解导致的临床信息不完整问题。 |
| [^16] | [DuplexDrama: A Synthesized Dialogue Dataset with Scenarios, Full-Duplex Behaviors, Expressive Speech, and Sound Events](https://arxiv.org/abs/2609.12872) | DuplexDrama是首个同时涵盖完整人设场景、三种全双工行为、带情感标签的富表现力语音和剧本感知声音事件四个维度的合成口语对话数据集，包含超过2,000小时音频，并将发布800小时中英双语子集以推动全双工口语对话模型研究。 |
| [^17] | [Cognition on Graph: Navigating Massive Knowledge Space via Cognitive Cycles and Bidirectional Graph-Text Synergy](https://arxiv.org/abs/2609.12791) | 提出了CoG框架，一个受认知启发、无需训练的自适应知识探索方法，通过持续的“计划-探索-反思”认知循环和图-文本深度双向协同，实现海量异构知识库中的复杂推理导航。 |
| [^18] | [What Drives Recovery in Agentic Text-to-Cypher? LAST-CQ: An LLM Agent Self-Refinement Framework](https://arxiv.org/abs/2609.12746) | 该研究通过LAST-CQ框架上的反事实实验发现，智能体Text-to-Cypher系统的恢复收益主要来自失败检测与重试路由机制，而非复杂的LLM反馈内容。 |
| [^19] | [Residual Vector-based Reconstruction as Long-Context Recall Regardless of Context Window Size](https://arxiv.org/abs/2609.12686) | 提出了一种基于残差向量的长上下文召回方法，通过利用LLM前馈层中存储的残差向量确定性重构与查询相关的事实，无需额外训练或微调即可在上下文长度增加时保持近乎恒定的GPU内存使用。 |
| [^20] | [Doc2FRC: Length-Consistent Document-Level Machine Translation via Fixed-Range Chunking](https://arxiv.org/abs/2609.12674) | 提出基于动态规划的固定范围分块方法FRC，在训练和推理中一致应用，将任意长度的文档映射到相同的长度分布，从而解决文档级机器翻译中训练与推理的长度分布不匹配问题。 |
| [^21] | [LifeMem: Enabling Lifelong Experience Reuse for LLM Agents](https://arxiv.org/abs/2609.12655) | LifeMem是一个终身学习框架，通过基于底层工作流聚类交互轨迹来提取可复用技能，使大语言模型智能体能够跨多个环境迁移和复用经验，同时缓解灾难性遗忘问题。 |
| [^22] | [SWARM: A Multilingual Human-Annotated Dataset for Russian Propaganda Detection in Search Engine Results](https://arxiv.org/abs/2609.12653) | 该论文构建了首个针对搜索引擎结果中俄罗斯宣传检测的多语言人工标注数据集SWARM，并通过实验表明基于来源的黑名单会遗漏大多数宣传内容，而基于内容的大语言模型分析效果更佳。 |
| [^23] | [SteerDuplex: Steerable Duplex Speech Dialogue Models](https://arxiv.org/abs/2609.12623) | 该论文提出SteerDuplex，一个基于Moshi的可控全双工语音对话模型，通过可控性分类法、多目标微调、混合奖励的两阶段强化学习以及SteerBench评测基准，使模型能够根据用户指令在语气、人设、语速和语音风格等属性上可靠地调整对话行为。 |
| [^24] | [Calibrated Ambiguity in Multimodal Language Models: Humans reach for cultural references, while models describe the picture](https://arxiv.org/abs/2609.12575) | 该研究借助桌面游戏Dixit的任务首次揭示了多模态语言模型在歧义处理上与人类的根本差异：模型表现出“歧义坍缩”和“文化扁平化”，而人类能生成兼具开放性与可解读性的校准歧义线索。 |
| [^25] | [Meddies-PII: A Multilingual Framework for Personally Identifiable Information Extraction in Clinical De-identification](https://arxiv.org/abs/2609.12544) | 该论文提出了Meddies-PII框架，包含一个通过属性条件提示生成、经十三个确定性门控验证、涵盖十七种语言的一百万份合成临床文档数据集，以及基于该数据集训练的BIOES分类模型，后者在十五个外部基准上以平均F1 0.827达到了现有PII提取系统中的最高性能。 |
| [^26] | [Agent as Policy for Robotic Manipulation](https://arxiv.org/abs/2609.12541) | 该论文提出Agent as Policy (AGP)方法，使通用智能体无需任何任务特定训练即可直接控制物理机器人完成从精细操作、动态运动到可变形物体处理等多种真实操作任务。 |
| [^27] | [The House with a Million Windows: Interactive Fiction for Narrative Restorying](https://arxiv.org/abs/2609.12537) | 本文提出基于LLM的互动小说系统HWAMW，通过让用户的故事以不同文学风格被“窗户”式重构，实证证明其能增强叙事认同感，从而为AI辅助写作确立了“LLM不替人讲述故事、而是拓展故事意义”的新范式。 |
| [^28] | [Earth-Agent-Pro: Towards Real-World Full-Chain Earth Observation with Agents](https://arxiv.org/abs/2609.12533) | Earth-Agent-Pro提出了一种执行自适应的规划-执行框架，通过专家技能约束规划、工作流中心的结构化记忆实现局部后缀修复，以及专用大语言模型适配器训练，首次实现了真实世界全链条地球观测任务的自动化执行。 |
| [^29] | [Information Specialization and Constrained Synthesis in Multi-Agent LLM Forecasting: A Prospective Live-Study of the 2026 FIFA World Cup](https://arxiv.org/abs/2609.12495) | 该研究在2026年FIFA世界杯最后56场比赛上开展实时前瞻性评估，证明多智能体LLM预测系统中的信息专业化角色（量化专家与新闻专家）能产生彼此不同的预测，且由专家、评论者和元智能体构成的四智能体顺序合成架构可提升预测效用。 |
| [^30] | [Confidence-Gated Transductive Test Generation for Code Reranking](https://arxiv.org/abs/2609.12489) | 该论文提出置信度门控直推式测试生成方法 CoTT，仅在归纳置信度较低时才调用直推式生成，以更低的计算成本提升了代码重排序的效果。 |
| [^31] | [Zipbench: Low-Cost Framework for Compressing Comprehensive Benchmarks of Large Language Models](https://arxiv.org/abs/2609.12475) | ZipBench提出了一种低成本的基准压缩框架，仅通过少量锚定模型的评估即可生成具有理论误差和排序一致性保证的紧凑基准子集，并能轻松扩展到新发布的基准，大幅降低了LLM评估成本。 |
| [^32] | [AMDKernelVault: Large-Scale Datasets and Agentic Training for AMD GPU Kernel Optimization](https://arxiv.org/abs/2609.12471) | AMDKernelVault通过智能体驱动的生成-验证流水线构建了面向AMD GPU的开放HIP/Triton内核语料库（含超过10万个执行验证样本），并证明经微调的Qwen3-8B在AMD内核生成任务上取得了领先正确率。 |
| [^33] | [Not All Speech Is Intent: Adaptive Self-Correcting Inference Layer for Post-ASR False Wake-Up](https://arxiv.org/abs/2609.12469) | 提出了ASCIL框架，在ASR转录后、响应生成前通过融合声学嵌入、语言线索、设备上下文以及用户的隐式与显式行为反馈信号（如犹豫、取消等），自适应地重新评估并纠正误唤醒的意图误分类。 |
| [^34] | [GraphProfiler: Source-Linked Sensitive Attribute Inference via Personal Knowledge Graphs](https://arxiv.org/abs/2609.12448) | GraphProfiler是一种可审计的LLM画像工具，通过将用户帖子历史构建为可溯源至原始帖子的个人知识图谱，在实现高精度敏感属性推断的同时，能够定位真正泄露隐私信息的关键帖子，从而支持针对性的隐私保护。 |
| [^35] | [Do LLMs Trust the Accuser or the Accusation? Measuring Belief Shifts in Werewolf](https://arxiv.org/abs/2609.12446) | 该研究提出了一个基于狼人杀游戏的信念变化评估基准，通过标注怀疑与指控消息并测量观察模型的信念更新，发现大模型虽能更好识别狼人并抵御不信任者提出的指控，但仍会轻信受信任指控者的指控，即使对方是狼方阵营。 |
| [^36] | [Diverse Minds, Divided Networks? Personality Composition, Polarization, and Collective Intelligence in LLM-Based Social Simulations](https://arxiv.org/abs/2609.12444) | 该研究提出TraitMix实验设计，通过对991次大语言模型社会模拟的分析发现，人格特质异质性对极化的两个维度具有相反的影响——多样化的社会观点更加分散但阵营对立更少，揭示了人格构成同时塑造极化与集体智能的关键作用。 |
| [^37] | [Beyond ID Embeddings: Process-Grounded Language Modeling for Cognitive Diagnosis](https://arxiv.org/abs/2609.12403) | 该论文提出PLCD框架，利用大语言模型构建概念模式和认知过程图作为认知先验，并通过DA-MoE专家和过程级对比学习将语言表示映射到认知状态，从而超越传统ID嵌入方法，解决了新练习或新概念出现时的语义局限问题。 |
| [^38] | [Representation-based Masked Diffusion Model](https://arxiv.org/abs/2609.12382) | 提出基于表示的掩码扩散模型（RMDM），通过预训练编码器将文本表示归一化为高斯先验以显式编码全局语义，从而协调被掩码词元的并行更新，生成更连贯的文本。 |
| [^39] | [ORQA: An Occupation-Realistic Question and Answer Framework for LLM Professional Knowledge](https://arxiv.org/abs/2609.12366) | ORQA通过将O*NET职业与可信的职业权威网站（如监管机构、执照颁发机构和政府出版物）相连接，自动生成可溯源的职业知识问答对，构建了覆盖SOC全部21个大类、116个职业的480个高质量问题，用以评估大语言模型的职业专业知识。 |
| [^40] | [CueMem: Cue-Guided Context Reconstruction for Long-Term Conversational Memory](https://arxiv.org/abs/2609.12354) | CueMem提出将记忆记录作为检索线索而非自包含证据，通过线索引导定位来源对话轮次并在轮次图上扩展，从原始对话中重建与查询相关的紧凑证据上下文，从而兼顾长期对话记忆的效率与细粒度证据完整性。 |
| [^41] | [SynthSentry: Detecting Synthetic Data Contamination in Language Model Training Data](https://arxiv.org/abs/2609.12353) | 提出SynthSentry，一种模型无关的语料库级污染检测信号，通过词汇多样性崩溃、n-gram尾部截断和困惑度方差三种统计量的分布散度，在训练前识别出可能导致模型崩溃的合成数据，且无需访问生成模型或任何生成历史。 |
| [^42] | [I Am No One: Style-Aware Paraphrasing for Text Anonymization](https://arxiv.org/abs/2609.12341) | 提出一种基于大语言模型的风格感知改写匿名化方法，通过构建风格画像并重写文本抑制可识别的风格指纹，在保持文本质量的同时将作者归属识别F1分数降低60-70%。 |
| [^43] | [ESTS at WMT26: Routing-Informed Expert Pruning for Model Compression](https://arxiv.org/abs/2609.12310) | 该论文提出利用任务特定路由质量和跨语言路由差异来识别并物理移除GPT-OSS-20B中的低重要性专家，结合恢复微调与MXFP4量化技术，实现了面向中英和阿英翻译任务的高效模型压缩。 |
| [^44] | [Breaking the Token Ceiling: Distilling Smaller, Stronger Byte Models](https://arxiv.org/abs/2609.12303) | 本文首次大规模研究了词元化方案与训练目标（蒸馏 vs. 交叉熵）对约10亿参数模型的影响，提出了两种将词元logits转换为字节logits的方法（Marginalize-It和End-Of-Token），并发现在八个基准测试中词元模型仍优于字节模型。 |
| [^45] | [EAR: Entity-Aware Partitioning Approach for Retrieval-Augmented Generation Development](https://arxiv.org/abs/2609.12268) | EAR提出了一种实体感知的语料库分割方法，通过从问题和选项中提取锚点、检索语料库中的局部窗口，在改进多项选择题问答的同时减少检索内容长度。 |
| [^46] | [HypoKG: Evidence-Disciplined Biomedical Hypothesis Generation Beyond Endpoint Knowledge](https://arxiv.org/abs/2609.12260) | 该研究构建了整合KEGG、Rhea和UniProt的统一生物化学知识图谱基准HypoKG，在550条酶来源到罕见病终点的路径上评估六个大语言模型生成的13,200个假设，发现仅凭来源酶与疾病终点信息即可产生最高评分的假设，揭示了LLMs生物医学假设生成能力及其与真正证据推理之间的差距。 |
| [^47] | [Automated Detection and Structuring of Social Tipping Point Evidence in Climate related Documents: A Modular AI Framework](https://arxiv.org/abs/2609.12254) | 本文提出一个开放的模块化AI框架，通过集成DistilBERT分割器、RoBERTa分类器和Mistral 7B等多个组件，在段落级别自动检测并结构化气候文档中的环境社会临界点证据，填补了现有文本挖掘工具无法系统性发现和整理这类关键证据的空白。 |
| [^48] | [Chopthin-Consensus Power Sampling: A Diversity-Preserving Approach to LLM Decoding](https://arxiv.org/abs/2609.12243) | 本文提出切薄-共识幂采样（CCPS），通过Chopthin重采样器限制最大与最小权重比值并保留不等权重，避免了等权重重采样对低权重推理路径的过度剪除，在保持SMC近似无偏的同时维持推理路径多样性并保证有效样本量下界。 |
| [^49] | [Repair Before Reinforce: Context-Augmented Knowledge Graph Reasoning for Multi-Hop Question Answering](https://arxiv.org/abs/2609.12230) | 该论文提出一种上下文增强的知识图谱训练框架，通过为主要KG三元组附加同源文本中的支持三元组构建上下文图进行监督训练，从而提升大型语言模型在多跳问答任务中的推理能力，并在胃轻瘫和糖尿病的疾病知识图谱上验证了其有效性。 |
| [^50] | [GAUGE: When Not to Trust LLM-as-a-Judge in User-Simulated Evaluation of Task-Oriented Agents](https://arxiv.org/abs/2609.12191) | GAUGE协议揭示了一种常见的LLM智能体离线评估闸门存在严重缺陷——LLM评审给出的满意度评分与实际任务成功率几乎完全不相关（被评为满意的对话中有57.5%实际未能完成客户任务），因此不应仅凭主观评分来信任和选择任务导向型LLM智能体。 |
| [^51] | [Can LLMs in Draft-Verify-Revise Pipelines Resolve Deictic Ambiguity?](https://arxiv.org/abs/2609.12162) | 本文通过合成数据集研究发现，在起草-验证-修订的多模型流水线中，不同阶段的大语言模型可能对同一上下文依赖表达式产生不一致的解读，从而导致指示语偏移。 |
| [^52] | [Population-level measures of perceived food access reveal barriers beyond geographic proximity](https://arxiv.org/abs/2609.12132) | 本研究创新性地利用谷歌地图评论与零样本分类方法，在人口层面测量了食物获取的五个感知维度，揭示了传统地理邻近性指标无法捕捉的食物获取障碍。 |
| [^53] | [Local Edits, Global Ripples: Replay-Informed Policy Adaptation for Workflow Synthesis](https://arxiv.org/abs/2609.12127) | 提出RIPPLE框架，针对提示策略编辑中“局部编辑引发全局涟漪效应”与“编辑组合后相互干扰”两大难题，将编辑位置定位与编辑组合后安全性判别相分离，实现工作流合成中安全可靠的持久策略自适应。 |
| [^54] | [Quantifying Consonant Contributions to Word Intelligibility via Acoustic Masking](https://arxiv.org/abs/2609.12122) | 本文提出一种基于声学掩蔽与语音识别模型的可扩展方法，通过掩蔽诱导误识别率（MMR）量化每个辅音对单词可懂度的贡献，从而帮助确定运动性言语障碍治疗的优先干预目标。 |
| [^55] | [The Cost of Compression: A Rate-Distortion Limit on Factual Hallucination](https://arxiv.org/abs/2609.12111) | 该论文证明了事实性幻觉不仅源于事实未被学习（覆盖缺失），还源于有限内存迫使已观察到的事实只能被近似压缩存储，并首次给出了幻觉率的一个率失真理论下界。 |
| [^56] | [Extracting Dataset Mentions in Forced Displacement and FCV Documents: A Weakly Supervised Framework with LLM-Based Label Refinement](https://arxiv.org/abs/2609.12107) | 该论文提出一种弱监督框架，利用在通用文献上训练的轻量级模型生成候选数据集提及，再由前沿大语言模型进行上下文审查与标签精炼，从而在无需大规模人工标注语料的情况下实现强迫流离失所和FCV领域文档中的数据集引用自动提取。 |
| [^57] | [Creating an Atomic User Model for Personality-Aware Large Language Model Interaction](https://arxiv.org/abs/2609.12086) | 该论文提出原子用户模型（AUM），将用户表示为稳定身份核心加四个可解释外壳的分层结构，解决了现有助手仅依赖偏好总结而在任务变化时需反复重新学习用户的问题，并首次刻画了“人格渗漏”现象。 |
| [^58] | [What Counts as a Mistake? Annotating Recitation Events in Quran Memorization Transcripts](https://arxiv.org/abs/2609.12085) | 本文构建了首个针对《古兰经》背诵转录文本错误事件的人工标注数据集（100个案例、348个评分单元、162个定位事件）并配套可执行评估器，为基于ASR的《古兰经》背诵自动校对建立了标签感知与错误定位的评估基准。 |
| [^59] | [Is Bash All You Need? An Empirical Study of Tool Interfaces for Enterprise Digital Worker Agents](https://arxiv.org/abs/2609.11999) | 研究发现仅用 Bash 的智能体在企业任务基准上以更少的 token 消耗显著优于类型化专用工具，而为其额外添加类型化工具并不能带来可检测的性能提升。 |
| [^60] | [Harness or Model? Isolating the Harness Effect in Agentic Coding with a Contamination-Controlled Private Suite](https://arxiv.org/abs/2609.11987) | 本研究通过在一个污染受控的私有测试集上对相同模型分别使用厂商原生框架和第三方框架进行配对实验，发现原生框架与模型的组合并不存在显著的平均优势，从而挑战了“厂商原生搭配表现更好”这一普遍假设。 |
| [^61] | [Cortex: Content Analysis Support Software, a Resource for Qualitative Research](https://arxiv.org/abs/2609.11970) | 本文基于巴丹方法论并结合文献研究与半结构化访谈识别研究者实际需求，开发了面向学术研究者的内容分析支持网络应用Cortex，为定性研究提供高效的数据组织与分类工具。 |
| [^62] | [Space as an Interventional Invariant: Cross-Modal Predictive Geometry for Stratified Cities and Em-Spaced Intelligence](https://arxiv.org/abs/2609.11959) | 本文提出将空间定义为“干预不变量”，并构建跨模态预测几何框架，使不共享度量或表示的异构感知与城市数据，仍能在因果干预层面统一揭示共同的空间结构。 |
| [^63] | [R2VC: Modular Fact-Checking with Retrieval, Verification, and Confidence Calibration](https://arxiv.org/abs/2609.11955) | R2VC提出了一种模块化的“检索-推理-验证-校准”事实核查架构，通过混合检索、DPO对齐的生成器、NLI交叉编码器验证以及置信度校准实现证据支撑、带引用和弃权机制的事实核查，在FEVER上使8B模型准确率较基线提升13.74%。 |
| [^64] | [PRISMA-LLM: An Empirical Reporting Framework for AI-Assisted Systematic Reviews](https://arxiv.org/abs/2609.11559) | 本文通过分析包含888篇论文的SciLitBench语料库，揭示了AI辅助系统综述中评估与报告的不一致问题，并提出PRISMA-LLM实证框架，将实现披露与后果敏感的评估及局限性报告分离，以规范这一领域。 |
| [^65] | [Beyond Solver Verdicts: Generative Reward Models for Autoformalization](https://arxiv.org/abs/2609.11085) | 本文发现了自动形式化中的“判定保持的不忠实性”（VPU）失败模式，从理论上证明仅依赖求解器判定的验证方法无法有效检测此类错误，并提出生成式验证方法（GenV），将Z3等价性预言机蒸馏为无参照的连续等价性评分以实现可靠的验证。 |
| [^66] | [Can Foundation Models Moderate Online Content? Evaluating Instruction- vs. Example-Driven Policy Operationalization](https://arxiv.org/abs/2609.10410) | 本文提出包含4,000条Bluesky人工标注帖子的新基准ModerationBench，系统比较了指令驱动与示例驱动两种政策操作化范式，发现基础模型的内容审核F1分数可达Bluesky现有审核系统的近三倍（0.60 vs. 0.22）。 |
| [^67] | [When Auditors Fabricate: Batch-Size Degradation and Confident Hallucination in LLM Detection of Planted Document Contamination](https://arxiv.org/abs/2609.09696) | 大语言模型在单文档和小批量污染检测中表现尚可（50%-60%），但在大批量处理时检测率骤降至2.8%，且其失败方式不是承认无法处理，而是自信地捏造包括虚假污染项在内的检测结果。 |
| [^68] | [SAEScientist-Bench: Can AI Agents Conduct Autonomous SAE Interpretability Research?](https://arxiv.org/abs/2609.09113) | 本文提出SAEScientist-Bench基准，评估AI智能体能否作为科学家自主运用稀疏自编码器（SAE）工具，进行机制可解释性的自主科学发现研究。 |
| [^69] | [From Scores to Evidence: Auditable Decisions Can Improve Speech Deepfake Detection](https://arxiv.org/abs/2609.08899) | 该论文提出一种可审计的决策记录方法，将被动检测分数、条件性键控探针分数、检索支持和说话者画像边际四个线索纳入后期校准步骤，使语音深伪检测的最终决策在保持标量的同时保留证据来源信息，从而提升检测决策的可信度与可解释性。 |
| [^70] | [LatentMD: Benchmarking Markdown Boundary Failures in LLM-Generated Text](https://arxiv.org/abs/2609.06993) | LatentMD是一个将内容正确性与边界正确性分离的基准测试，它揭示了LLM生成的Markdown中边界失败问题普遍存在——38%的内容正确输出实际上边界已损坏。 |
| [^71] | [Reason Through the Latent! Making Latent Visual Reasoning Necessary](https://arxiv.org/abs/2609.06746) | 提出因果视觉循环推理（CVRR）框架，通过在解码前移除视觉状态和多模态KV缓存，迫使循环隐藏状态成为唯一的图像条件信息通路，从而确保潜空间视觉推理真正被模型依赖。 |
| [^72] | [Decomposing LLM-Judge Uncertainty to Target Expert Labels](https://arxiv.org/abs/2609.06444) | 该论文提出一种小型贝叶斯模型，将LLM评审器的总不确定性分解为可被专家标注消除的认知不确定性和不可消除的偶然不确定性，使专家只需标注评审器真正无知的项目，在ChaosNLI数据集上比使用总不确定性多消除83%的误差。 |
| [^73] | [EVOHARNESSBENCH: Can Your Agents Keep Pace with an Evolving Harness?](https://arxiv.org/abs/2609.04280) | 提出了EVOHARNESSBENCH基准，首次将非平稳性从任务流转移到智能体运行框架本身（工具、技能、智能体）的持续演进上，用于评估智能体在框架不断变化环境下的适应与保留能力。 |
| [^74] | [NS-Copilot: An LLM-Driven Agent System for Autonomous Neuroscience Analysis](https://arxiv.org/abs/2609.01971) | NS-Copilot是一个由大语言模型驱动的多智能体系统，能够自主选择和协调神经科学领域的各类预训练模型，支持EEG和细胞外尖峰数据等关键模态，为专业神经科学分析任务提供端到端的自主工作流程。 |
| [^75] | [NSIDDx: A Design Framework for Neuro-Symbolic, Practitioner-First Differential Diagnosis in Low-Resource Settings](https://arxiv.org/abs/2609.00256) | 本文提出NSIDDx设计框架，主张在低资源环境下将临床医生作为主动推理主体融入鉴别诊断，通过三值症状编码、矛盾检测、审计字符串和医生覆盖权的神经符号流水线在消费级硬件上离线运行，弥合了LLM诊断系统的头条准确率与可验证临床可靠性之间的差距。 |
| [^76] | [Do Multimodal LLMs See Before They Read? Diagnosing Contextual Sycophancy](https://arxiv.org/abs/2609.00067) | 该论文诊断了多模态大语言模型易受外部文本误导而忽视冲突图像证据的“多模态情境性谄媚”问题，并提出“系统2视觉仲裁”（S2VA）方法，通过让视觉证人在读取文本前先独立判断，在六个模型上将准确率显著提升19.7至44.1分。 |
| [^77] | [Closed-Loop Bayesian Molecular Inverse Design with Semantic LLM Surrogates](https://arxiv.org/abs/2608.22967) | 该论文提出了一种闭环贝叶斯分子逆向设计框架，通过将大型语言模型作为代理直接处理文本形式的任务指令和优化历史，以在有限预算下提高匹配目标性质的分子比例。 |
| [^78] | [Whitewashing Hate, Smearing Harmless Content: Annotator-Style Rebuttal Attacks on LLM-Based Moderation](https://arxiv.org/abs/2608.22230) | 本研究揭示了标注者风格的反驳攻击能显著破坏LLM仇恨言论审核的准确性，且洗白与污蔑两种操纵方向存在模型特定的不对称效应。 |
| [^79] | [Language Models for Portuguese: A Systematic Mapping Study](https://arxiv.org/abs/2608.18138) | 本文对葡萄牙语语言模型进行了系统性映射研究，梳理了46个模型的现状，填补了该领域信息分散的空白。 |
| [^80] | [Slot2Text: Object-Centric Visual Tokenization for Efficient and Spatially Traceable Surgical MLLMs](https://arxiv.org/abs/2608.01473) | Slot2Text提出了一种以对象为中心的双模式手术多模态大语言模型，通过将密集视觉特征编码为少量带区域标签的槽位标记作为视觉词元，在大幅降低推理成本的同时实现了答案的空间可追溯性。 |
| [^81] | [A False Average: Chain-of-Thought Monitors Collapse Where They Are the Only Defense](https://arxiv.org/abs/2608.00583) | 仅重写智能体的思维链推理（保持动作完全不变）即可在一次无梯度攻击中将CoT监控器的捕获率从约95%暴跌至11%以下，揭示监控器的总体准确率是掩盖其在唯一防线场景下近乎全面失效的虚假平均值。 |
| [^82] | [SeDeM: Selective Decompression of Hidden-State Memories for Long-Context Question Answering](https://arxiv.org/abs/2608.00311) | SeDeM提出一种选择性解压缩框架，将上下文存储为紧凑的隐状态记忆块，并仅解压缩与查询相关的块来调节解码器，从而在长文本问答中兼顾效率与准确性。 |
| [^83] | [AdaRoPE: Not All Attention Heads Should Rotate and Scale Equally](https://arxiv.org/abs/2607.19363) | AdaRoPE为每个注意力头配备可学习的旋转频率和注意力缩放因子，打破了传统RoPE对所有头统一处理的做法，从而显著提升嵌入维度利用率并在长上下文设置中持续优于现有RoPE变体。 |
| [^84] | [Faithful by Construction: Claim-Anchored Attribution for Multi-Document Summarization](https://arxiv.org/abs/2606.23989) | 提出CAMS框架，将声明级归因嵌入“提取—选择—改写”流程，使多文档摘要中的每句话都能锚定到经过验证、可溯源的源文本片段，从而在构造层面保证摘要的忠实性。 |
| [^85] | [Capable but Careless: Do Computer-Use Agents Follow Contextual Integrity?](https://arxiv.org/abs/2606.23189) | 本文提出AgentCIBench评估基准，首次系统揭示了计算机使用代理在跨应用操作中违反情境完整性原则的严重隐私风险，评估发现15个前沿代理中有11个存在信息泄露问题，并归纳出视觉同址、任务歧义过度分享和接收者错位三种典型失败模式。 |
| [^86] | [When Context Misleads: Surprisal, Energy and Attention Entropy as Metrics of Coherence Illusions in LLMs](https://arxiv.org/abs/2606.21203) | 该研究首次发现语言模型会像人类一样陷入连贯性错觉，并提出用惊奇度、注意力熵和联想记忆能量三种指标量化这一现象，同时揭示了模型处理语篇连贯性时存在共享的注意力机制。 |
| [^87] | [GRACE-DS: a Guarded Reward-guided Agent Correction Environment in Data Science](https://arxiv.org/abs/2606.16000) | GRACE-DS是一个用于LLM驱动AutoML智能体部署前评估的隔离环境，通过隐藏的可执行验证器从预测性能、泄漏规避、可复现性等多维度进行评估，其中灵活迭代交互机制的表现优于单次生成等基线方法。 |
| [^88] | [On The Effectiveness-Fluency Trade-Off In LLM Conditioning: A Systematic Study](https://arxiv.org/abs/2606.12234) | 该论文系统研究了LLM条件控制方法，发现高效的激活引导方法往往以流畅性大幅下降为代价，且在指令微调模型上效果远差于基础模型，而提示和监督微调适合概念注入但不擅长概念移除。 |
| [^89] | [LC-QAT: Data-Efficient 2-Bit QAT for LLMs via Linear-Constrained Vector Quantization](https://arxiv.org/abs/2606.10531) | LC-QAT通过线性约束向量量化提供高质量PTQ初始化并实现完全可微的端到端优化，是一个数据高效的2比特大语言模型量化感知训练框架。 |
| [^90] | [Expert-Level Crisis Detection in Mental Health Conversations](https://arxiv.org/abs/2606.10380) | 该论文提出了临床医生标注的CRADLE-Dialogue基准数据集以及“警报-确认”评估协议，用于解决多轮心理健康对话中轮次级危机检测的难题，使模型能够捕捉随对话演进的风险信号并支持早期干预。 |
| [^91] | [A retrieval conditioned rebinding circuit for dynamic entity tracking in large language models](https://arxiv.org/abs/2606.08644) | 本研究通过因果干预识别出大语言模型中一种检索条件化的重绑定电路机制，用于动态实体属性追踪，并发现Gemma模型在注意力头的查询/键子空间中表达绑定信息，而Llama模型则主要在键向量中携带绑定信息。 |
| [^92] | [SAEExplainer: Interpreting SAE Features with Activation-Guided Preference Optimization](https://arxiv.org/abs/2606.08496) | 提出SAEExplainer训练框架，以激活分数作为客观奖励信号，通过两轮迭代优化实现模型自我纠正与持续改进，显著减少SAE特征解释中的幻觉并强化因果触发模式。 |
| [^93] | [UrduMMLU: A Massive Multitask Benchmark for Urdu Language Understanding](https://arxiv.org/abs/2606.07167) | UrduMMLU是一个基于乌尔都语本地教育资源构建的大规模多任务语言理解基准，包含26,389道多选题，覆盖26个学科和五个领域，填补了乌尔都语缺乏原生MMLU风格评测基准的空白，并系统评估了30个大语言模型在英语和乌尔都语提示下的表现。 |
| [^94] | [ReasoningFlow: Discourse Structures for Understanding LLM Reasoning Traces](https://arxiv.org/abs/2606.05402) | 该论文提出了ReasoningFlow框架，将大型推理模型的推理轨迹建模为细粒度有向无环图，从而揭示出不同模型尽管训练背景各异，却展现出结构相似的推理轨迹这一重要发现。 |
| [^95] | [Moral Semantics Survive Machine Translation: Cross-Lingual Evidence from Moral Foundations Corpora](https://arxiv.org/abs/2605.22660) | 尽管在俚语和文化负载表达上存在翻译缺陷，基于大语言模型的机器翻译仍能充分保留道德语义线索，使英语标注的道德语料库可用于跨语言（以波兰语为例）的道德价值观自动分类。 |
| [^96] | [Text-to-SPARQL Generation with Reinforcement Learning: A GRPO-based Approach on DBLP](https://arxiv.org/abs/2605.20066) | 本研究证明，基于结果奖励的GRPO强化学习能够在DBLP-QuAD上训练小型的Qwen3-1.7B模型实现零样本Text-to-SPARQL查询生成，无需依赖大型模型或完整的黄金查询标注监督。 |
| [^97] | [ShadowPEFT: Shadow Network for Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2604.19254) | ShadowPEFT将参数高效微调整合为一个模块化的影子网络，其影子模型可作为独立预测器进行推理而无需运行基础模型，在文本和图像的生成与理解任务上达到或超越现有方法。 |
| [^98] | [SaFeR-Steer: Evolving Multi-Turn MLLMs via Synthetic Bootstrapping and Feedback Dynamics](https://arxiv.org/abs/2604.16358) | 提出渐进式多轮安全对齐框架SaFeR-Steer，通过分阶段合成数据引导与导师介入的GRPO强化学习，并引入轨迹一致求和奖励（TCSR）机制，同时发布多轮多模态安全数据集STEER，以弥合多轮对话场景下多模态大模型训练与部署之间的安全对齐差距。 |
| [^99] | [Translationese as a Rational Response to Translation Task Difficulty](https://arxiv.org/abs/2603.12050) | 该论文提出翻译腔是译者对翻译任务认知难度的理性回应，并证明基于大语言模型惊讶度的信息论任务难度指标能够有效预测文本的翻译腔程度。 |
| [^100] | [Countdown-Code: A Testbed for Studying The Emergence and Generalization of Reward Hacking in RLVR](https://arxiv.org/abs/2603.07084) | 本文提出Countdown-Code测试平台，通过代理奖励与真实奖励的清晰分离来精确测量RLVR中的奖励破解现象，并发现SFT数据中仅1%的奖励破解轨迹污染就足以让模型无意中习得这种失对齐行为。 |
| [^101] | [Measuring Pragmatic Influence in Large Language Model Instructions](https://arxiv.org/abs/2602.21223) | 该论文提出了一个系统测量大语言模型指令中语用框架化影响力的框架，通过指令-框架分解、涵盖400个实例和13种策略的分类法以及基于优先级的测量方法，揭示“如何提问”而非“提问内容”对模型行为的影响。 |
| [^102] | [False positive bias in AI-powered speech-based cognitive screening for multilingual English speakers in the UK](https://arxiv.org/abs/2602.13047) | 该研究通过对1,395名参与者、超过263小时语音数据的分析，首次发现尽管语音识别准确率在各语言群体间无显著差异，但AI认知筛查的下游模型对英国多语言英语使用者存在系统性的假阳性偏差，凸显了认知筛查公平性评估的重要性。 |
| [^103] | [PACIFIC: Can LLMs Discern the Psychometric Traits Influencing Your Preferences? Personality-Driven Preference Alignment in LLMs](https://arxiv.org/abs/2602.07181) | 提出PACIFIC框架，将大五人格（OCEAN）特质作为潜在信号来组织和推理用户偏好历史，从而在偏好信号嘈杂、不完整的情况下实现更可靠的人格驱动LLM偏好对齐。 |
| [^104] | [LLM Compression by Block Removal with Constrained Binary Optimization](https://arxiv.org/abs/2602.00161) | 该论文将大语言模型的块删除压缩问题形式化为约束二值优化问题并映射到Ising自旋玻璃物理系统，利用系统能量作为下游模型性能的代理指标，在深度压缩场景下（如50%压缩Llama-3.3-70B）相比现有最先进方法在MMLU基准上提升近23个百分点。 |
| [^105] | [GeoSense-AI: Fast Location Inference from Crisis Microblogs](https://arxiv.org/abs/2512.18225) | 本文提出GeoSense-AI流水线，通过融合话题标签分词、命名实体识别与地名录消歧等多种NLP技术，从危机微博文本中实时推断地理位置，在保持高精度的同时实现比现有NER工具快数个数量级的处理速度。 |
| [^106] | [MMGR: Multi-Modal Generative Reasoning Benchmark and Evaluation](https://arxiv.org/abs/2512.14691) | 本文提出MMGR基准，通过涵盖抽象推理、具身导航和物理常识三个领域的10个任务，评估多模态生成模型在物理、逻辑、2D空间、3D空间和时间五种推理能力上的真实推理水平，并采用答案可验证与过程感知的逐帧链式推理评估方法。 |
| [^107] | [When Bias Pretends to Be Truth: How Spurious Correlations Undermine Hallucination Detection in LLMs](https://arxiv.org/abs/2511.07318) | 该论文揭示了一类由训练数据中虚假关联（如姓氏与国籍的关联）驱动的幻觉，这类幻觉被模型自信生成、不受模型规模扩大影响、能规避现有检测方法、且在拒绝微调后仍持续存在，导致基于置信度过滤和内部状态探测等主流幻觉检测方法从根本上失效。 |
| [^108] | [KoSimpleQA: A Korean Factuality Benchmark with an Analysis of Reasoning LLMs](https://arxiv.org/abs/2510.18368) | 本文提出了聚焦韩国文化知识的韩语事实性基准KoSimpleQA（含938个问题），发现最强模型正确率仅31.6%、其排名与英语SimpleQA显著不同，并证明推理能力可缓解大语言模型的跨语言知识差距。 |
| [^109] | [Is Multilingual LLM Watermarking Truly Multilingual? Scaling Robustness to 100+ Languages via Back-Translation](https://arxiv.org/abs/2510.18019) | 本文提出STEAM检测方法，利用贝叶斯优化在126种候选语言中搜索最能恢复水印强度的回译，使LLM水印检测在包括中低资源语言在内的100多种语言下保持鲁棒。 |
| [^110] | [Limits of LLM Text Detectors in Education](https://arxiv.org/abs/2508.08096) | 本文提出了一个贡献感知的LLM文本检测评估框架，通过八个学生贡献等级模拟真实写作场景，揭示了当前基于二元区分的LLM检测器在教育评估中的局限性。 |
| [^111] | [UrduFactCheck: An Agentic Fact-Checking Framework for Urdu with Evidence Boosting and Benchmarking](https://arxiv.org/abs/2505.15063) | 本研究首创性地为乌尔都语构建了两个人工标注基准（UrduFactBench用于声明验证、UrduFactQA用于评估大模型问答事实性），并提出了融合单语与翻译证据检索策略的模块化代理式事实核查框架UrduFactCheck，填补了全球2亿多乌尔都语使用者在事实核查领域的空白。 |
| [^112] | [LLM-BabyBench: Can Language Models Plan in Worlds They Can Simulate?](https://arxiv.org/abs/2505.12135) | 该论文提出 LLM-BabyBench，将 BabyAI 网格世界改造为完全可观察的纯文本环境，在构造上排除感知、指令理解、检索等所有非规划失败因素，从而对语言模型的规划能力进行纯净、可验证的评估。 |
| [^113] | [Statistical Mechanics of Semantic Compression](https://arxiv.org/abs/2503.00612) | 该论文提出将语义压缩建模为欧几里得语义空间中的优化问题，并创新性地将其映射为自旋玻璃哈密顿量，利用统计力学方法求解在保持语义的前提下最小化消息长度的问题。 |
| [^114] | [From Bench-to-Bedside: A Review of Clinical Trials in Drug Discovery and Development](https://arxiv.org/abs/2412.09378) | 本综述系统梳理了药物开发中I至IV期临床试验的特点与联系，分析了伦理合规、受试者招募等主要挑战，并阐述了人工智能等创新技术及新兴疗法对未来临床试验设计的变革性影响。 |
| [^115] | [All Entities are Not Created Equal: Examining the Long Tail for Ultra-Fine Entity Typing](https://arxiv.org/abs/2410.17355) | 本研究提出一种新颖的启发式方法来近似实体的预训练分布，系统揭示了仅依赖预训练语言模型参数化知识的实体类型分类方法在处理长尾实体时表现显著不佳，表明需要超越预训练语言模型的方法来应对不常见实体。 |

# 详细

[^1]: 类型多样性使Transformer能够实现组合泛化

    Type Diversity Enables Transformers to Generalise Compositionally

    [https://arxiv.org/abs/2609.13144](https://arxiv.org/abs/2609.13144)

    本研究提出Transformer在结构泛化上表现较差并非其固有缺陷，而是数据集中结构类型多样性偏低所致——提高类型多样性（无论是词汇类型还是结构类型）均可同等程度地提升组合泛化能力。

    

    组合泛化被分为词汇泛化和结构泛化。先前的工作发现，对Transformer而言，结构泛化比词汇泛化更难。我们提出，这种差异并非Transformer固有的特性，而是由于这些先前工作所用数据集中词汇类型的多样性高而结构类型的多样性低。所谓类型多样性，我们指的是某一类型的不同构造器的数量，而不是例如可能填充该结构的特定词组合。为验证这一观点，我们改变了先前已发布数据集中词汇类型和结构类型的类型多样性程度。我们使用语法框架创建了COGS和SLOG数据集的语言多样化变体。我们发现，在词汇和结构测试用例中，类型多样性与组合泛化的相关性是等同的，这支持了我们的假设。我们注意到与...（摘要在此处截断）

    arXiv:2609.13144v1 Announce Type: new  Abstract: Compositional generalisation has been divided into lexical and structural generalisation. Previous work has found that structural generalisation is harder than lexical for Transformers. We propose that this difference is not inherent to Transformers, but due to the high diversity of lexical types and low diversity of structural types in the specific datasets of these previous works. By type diversity we mean the number of different constructors of that type, instead of, for example, the specific word combinations that might populate the structure. To test this, we vary the amounts of type diversity of lexical and structural types in previously published datasets. We create linguistically diverse variants of the COGS and SLOG datasets using Grammatical Framework. We find that type diversity correlates with compositional generalisation equally in lexical and structural test cases, supporting our hypothesis. We note a contradiction with the
    
[^2]: SAS：通过上下文排序的端到端优化实现简单注意力稀疏化

    SAS: Simple Attention Sparsification via End-to-End Optimization of Context Ranking

    [https://arxiv.org/abs/2609.13141](https://arxiv.org/abs/2609.13141)

    SAS提出了一种门控稀疏注意力机制，通过与语言建模损失端到端地优化上下文排序，使注意力预算直接分配给对预测最有影响的上下文单元，克服了以往方法依赖稠密注意力蒸馏导致的预算错配问题。

    

    训练后注意力稀疏化通过为每个查询选择一小组上下文单元（token或块），来降低预训练Transformer的二次方累积注意力成本。现有的可训练方法通常使用轻量级选择器对上下文单元进行评分，随后进行硬性Top-K选择，但这阻断了语言建模损失的梯度传播。因此，这些方法通常采用蒸馏逐层稠密注意力分布的方式。尽管这促使选择器按照原始模型中的稠密注意力权重对上下文单元进行排序，但这种排序与它们在固定注意力预算（即每个查询所关注的上下文单元数量）下对预测的影响并不直接一致，可能将有限的预算浪费在用处较小的单元上。为了解决这种不一致性，我们提出了简单注意力稀疏化（SAS），这是一种门控稀疏注意力机制，能够与语言建模损失进行端到端的上下文排序优化。

    arXiv:2609.13141v1 Announce Type: new  Abstract: Post-training attention sparsification reduces the quadratic cumulative attention cost of pretrained Transformers by selecting a small set of context units (tokens or blocks) for each query. Existing trainable methods usually use a lightweight selector to score context units, followed by hard Top-K selection that blocks gradients from the language modeling loss. Consequently, these methods commonly distill layer-wise dense attention distributions. Although this encourages the selector to rank context units by dense attention weights in the original model, the ranking is not directly aligned with their impact on predictions under a fixed attention budget (i.e., the number of attended context units per query), potentially wasting the limited budget on less useful units. To address this misalignment, we propose Simple Attention Sparsification (SAS), a gated sparse attention mechanism that optimizes context ranking end-to-end with the langua
    
[^3]: 继续、适应或让位：全双工智能体中对重叠语音的轮内适应

    Continue, Adapt, or Yield: In-Turn Adaptation to Overlapping Speech in Full-Duplex Agents

    [https://arxiv.org/abs/2609.13117](https://arxiv.org/abs/2609.13117)

    本文提出Duplex Cue评估基准，首次系统评估全双工语音智能体的“轮内适应”能力——即在继续说话的同时吸纳听者补充内容（如纠正、澄清），这一超越了“继续说话或停止”二元决策的第三种人类常见回应方式。

    

    全双工评估通常强调智能体是继续说话还是停止说话。这种二元划分无法表达人类经常使用的第三种回应方式：在继续说话的同时吸纳听者刚刚贡献的内容。这些内容可能是缺失的词语、纠正或澄清。我们提出了Duplex Cue，一种针对全双工语音智能体中这种“轮内适应”能力的评估方法。Duplex Cue将听者意图（附和、协作或打断）与说话者行为区分开来：说话者可以保持不变地继续说话、在当前轮次内进行适应、或让出话语权。适应既包括确认承认，也包括内容修改。在一项单模型案例研究中，我们使用来自非脚本英语对话的300个人工确认的语音提示，将录制的人类回应与在重放听者音频时由PersonaPlex生成的续写进行比较。我们最终保留了208对数据，这些数据满足在提示出现时正在说话者处于活跃状态，且每种条件下都有可评分的回应。

    arXiv:2609.13117v1 Announce Type: new  Abstract: Full-duplex evaluation often emphasizes whether an agent keeps speaking or stops. That binary cannot express a third response humans use routinely: continuing to speak while incorporating what the listener just contributed. The contribution may be a missing word, a correction or a clarification. We introduce Duplex Cue, an evaluation of this \emph{in-turn adaptation} in full-duplex voice agents. Duplex Cue separates listener intent (backchannel, collaboration, or interruption) from speaker behavior: continuing unchanged, adapting within the turn, or yielding. Adaptation includes acknowledgment as well as content revision. In a single-model case study using 300 human-confirmed cues from unscripted English conversations, we compare recorded human responses with PersonaPlex continuations generated while replaying the listener's audio. We retain 208 pairs with the ongoing speaker active at cue onset and a scorable response in each condition.
    
[^4]: MP-Bench：将语音代理作为多方对话参与者进行评估

    MP-Bench: Evaluating Voice Agents as a Multiparty Conversation Participant

    [https://arxiv.org/abs/2609.13076](https://arxiv.org/abs/2609.13076)

    该论文提出了MP-Bench，这是首个专门用于客观评估语音代理作为多方对话积极参与者能力的基准测试，填补了现有基准只关注二元交互的空白。

    

    对话式语音代理已取得显著进展，通过级联和端到端架构提供了日益自然的人机交互。然而，尽管近期的基准测试广泛评估了二元交互和被动音频理解，它们在很大程度上忽视了一个普遍存在的现实场景：多方对话。由于会话复杂性呈指数级增长，在这些场景中评估语音代理从根本上比在二元交互中更具挑战性。为了让语音代理能够无缝融入人类群体互动动态，它们不仅要生成符合上下文的恰当回应，还必须展现出对开放式话轮转换的细致理解。为填补这一空白，我们提出了多方对话基准MP-Bench，这是首个专门设计用于客观评估会话语音系统在多方场景中作为积极参与者表现的基准测试。MP-Bench评估代理行为……

    arXiv:2609.13076v1 Announce Type: cross  Abstract: Conversational voice agents have advanced significantly, offering increasingly natural human-machine interactions through both cascaded and end-to-end architectures. However, while recent benchmarks extensively evaluate dyadic interactions and passive audio comprehension, they largely overlook a prevalent real-world scenario: multi-party conversations. Evaluating agents in these settings is fundamentally more challenging than in dyadic interactions due to the exponentially greater conversational complexity. For voice agents to integrate seamlessly into human group dynamics, they must not only generate contextually appropriate responses but also demonstrate a nuanced understanding of open turn-taking. To address this gap, we introduce Multiparty Bench (MP-Bench), the first benchmark specifically designed to objectively evaluate conversational speech systems as active participants within multi-party contexts. MP-Bench assesses agent beha
    
[^5]: MAxBench：一个多分类概念恢复基准测试

    MAxBench: A Multinomial Concept Recovery Benchmark

    [https://arxiv.org/abs/2609.13072](https://arxiv.org/abs/2609.13072)

    本文提出了MAxBench，一个几何无关的多分类概念表示评估框架，通过在6个概念和4个模型上系统比较10种定位方法（涵盖5种几何类型），为识别最适合多分类概念引导的表示几何结构及最有效的恢复方法提供了统一评估标准。

    

    对语言模型行为的细粒度控制（例如行为引导/steering）是可解释性研究中更具可操作性的成果之一。对于诸如“拒绝”这样的二元概念，激活空间中的单一方向通常就足以实现引导。然而，许多概念并非二元：例如“动物”和“国家”包含许多子类别，每个子类别又有多个实例。对于这些概念，可能的表示几何结构的搜索空间远大于二元概念；因此，目前尚不清楚哪些几何结构最为合适，也不清楚哪些方法在恢复这些结构时最为有效。在这项工作中，我们介绍了MAxBench，这是一个几何无关的多分类概念表示评估框架，其基于从恢复的概念表示中进行采样来评估。我们使用MAxBench在6个概念和4个模型上比较了10种定位方法（涵盖5种几何类型）。利用该框架，我们发现（i）仿射子空间在引导方面表现更为……（摘要在此处被截断）

    arXiv:2609.13072v1 Announce Type: cross  Abstract: Fine-grained control of language model behaviors (e.g., steering) is among the more actionable outcomes of interpretability research. For binary concepts such as refusal, a single direction in activation space often suffices for steering. However, many concepts are not binary: Animals and Countries contain many subcategories, each with multiple instances. For these concepts, the search space over possible representation geometries is far larger than for binary concepts; it is thus not clear what geometries are most appropriate, nor what methods are most effective at recovering them. In this work, we introduce MAxBench, a geometry-agnostic evaluation framework for multinomial concept representations based on sampling from the recovered concept representation. We use MAxBench to compare 10 localization methods (covering 5 geometry types) across 6 concepts and 4 models. Using this framework, we find that (i) affine subspaces steer more re
    
[^6]: 混合专家模型强化学习中的专家空间探索

    Expert-Space Exploration in MoE Reinforcement Learning

    [https://arxiv.org/abs/2609.13058](https://arxiv.org/abs/2609.13058)

    提出ESRL框架，通过架构感知的方式显式探索MoE模型的专家路由空间来增加采样多样性，从而提升强化学习训练效果，同时避免直接扰动激活不合适专家所导致的采样质量下降问题。

    

    强化学习（RL）已成为大语言模型后训练的核心技术。近来针对混合专家（MoE）模型的强化学习进展主要集中于提升优化稳定性和训练效率，而将专家选择视为固定组件。由于路由决定了产生输出分布的稀疏计算路径，专家选择为采样多样性提供了额外的来源。通过实证分析，我们发现扰动专家路由能够有效改变模型输出并增加采样多样性，这类似于提高解码温度的效果。然而，直接扰动可能会激活不合适的专家并显著降低采样质量。基于这些观察，我们提出了专家空间探索强化学习（ESRL），这是一个架构感知的框架，能够显式探索MoE模型的专家路由空间。ESRL保留了

    arXiv:2609.13058v1 Announce Type: new  Abstract: Reinforcement learning (RL) has become central to post-training of large language models. Recent advances in RL for Mixture-of-Experts (MoE) models have primarily focused on improving optimization stability and training efficiency, while treating the expert selection as a fixed component. Since routing determines the sparse computation paths that induce output distributions, expert selection offers an additional source of rollout diversity. Through empirical analysis, we find that perturbing expert routing effectively alters model output and increases rollout diversity, which is similar to increasing the decoding temperature. However, direct perturbation can activate unsuitable experts and substantially degrade rollout quality. Motivated by these observations, we introduce Expert-Space Exploration Reinforcement Learning (ESRL), an architecture-aware framework that explicitly explores the expert-routing space of MoE models. ESRL preserves
    
[^7]: Kraken：基于低比特率矢量量化和双路径源条件化的大语言模型语音到语音翻译

    Kraken: LLM-based Speech-to-Speech Translation via Low-bitrate VQ and Dual-path Source Conditioning

    [https://arxiv.org/abs/2609.13045](https://arxiv.org/abs/2609.13045)

    Kraken模型通过采用基于单层矢量量化的低比特率语音token以及以源语音为条件的双路径Autowave-X解码器，解决了LLM预测高比特率语音token困难和对训练数据说话人身份与韵律严格对齐的依赖问题，从而提升了语音到语音翻译的性能。

    

    基于语音大语言模型（LLM）的语音到语音翻译（S2ST）已取得显著进展，为联合优化和保留非语言信息提供了可能。然而，这些模型在预测高比特率语音token时表现不佳，并且面临依赖具有理想对齐的说话人身份和韵律的S2ST训练数据的挑战。我们提出使用基于单层矢量量化的低比特率token，并通过训练使其能够重建自监督学习（SSL）特征。我们还采用了一个名为Autowave-X的独立token到波形解码器，该解码器同样以源语音作为条件，以改善非语言信息的迁移，从而放宽了对训练数据的约束。通过整合这些技术，我们提出了一个名为Kraken的S2ST模型，该模型在预训练LLM的基础上增强了语音特征输入和低比特率token输出，后接Autowave-X声码器。我们基于Qwen3-8B构建了该模型……

    arXiv:2609.13045v1 Announce Type: new  Abstract: Speech-to-speech translation (S2ST) has advanced significantly with speech LLMs, offering the potential for joint optimization and preserving non-linguistic information. However, these models struggle with predicting high-bitrate speech tokens in LLMs, and face the challenge of relying on S2ST training data with ideally aligned speaker identity and prosody. We propose using low-bitrate tokens based on single-layer vector quantization, trained to reconstruct self-supervised learning (SSL) features. We also employ a separate token-to-waveform decoder named Autowave-X, which is also conditioned on the source speech to improve non-linguistic transfer, thereby relaxing the training data constraints. With the integration of these techniques, we propose an S2ST model named Kraken, which augments a pre-trained LLM with speech feature inputs and the low-bitrate token outputs, followed by Autowave-X vocoder. We built the model upon Qwen3-8B and tr
    
[^8]: 任务优先于应用手册：揭示语言模型在长程程序性推理中的差距

    Tasks over Application Manuals: Revealing Gaps in Long-Horizon Procedural Reasoning for Language Models

    [https://arxiv.org/abs/2609.13005](https://arxiv.org/abs/2609.13005)

    提出了TAM基准，通过ICD-10-CM临床编码和美国联邦量刑两个真实领域任务，揭示大型语言模型在遵循数百页权威手册进行长程程序性推理时存在的显著能力差距。

    

    大型语言模型（LLMs）在广泛的自然语言任务上取得了强劲表现，最近的基准测试表明它们在多跳推理方面日益熟练。然而，这些基准测试通常是短程的，只需少量的检索或推理步骤，对于涉及遵循数百页手册、包含复杂且相互依赖的规则的现实世界任务，其可靠性证据有限。在本文中，我们介绍了任务优先于应用手册，这是一个用于评估长程程序性推理的基准。我们通过从两个领域筛选现实世界任务来构建TAM：ICD-10-CM临床编码（将医学病症映射到诊断代码）和美国联邦量刑（计算犯罪量刑指南结果，特别是犯罪等级），并配备经过人工验证的标签。每个任务都需要遵循包含数万条规则的权威手册……

    arXiv:2609.13005v1 Announce Type: new  Abstract: Large language models (LLMs) have achieved strong performance on a wide range of natural language tasks, and recent benchmarks suggest that they are increasingly adept at multi-hop reasoning. However, these benchmarks are typically short-horizon, requiring only a small number of retrieval or inference steps, and provide limited evidence of reliability on real-world tasks that involve following manuals spanning hundreds of pages with complex, interdependent guidelines. In this paper, we introduce Tasks over Application Manuals (TAM), a benchmark for evaluating long-horizon procedural reasoning. We construct TAM by curating real-world tasks from two domains: ICD-10-CM clinical coding (mapping medical conditions to diagnostic codes) and U.S. federal sentencing (computing crime sentencing guideline outcomes, specifically offense levels), with human-validated labels. Each task requires following an authoritative manual with tens of thousands 
    
[^9]: 以封面评判：清理大语言模型真实性基准以避免表层特征泄漏

    Judging by the Cover: Cleaning LLM Truthfulness Benchmarks to Avoid Surface-Level Feature Leakage

    [https://arxiv.org/abs/2609.13003](https://arxiv.org/abs/2609.13003)

    该研究揭示TruthfulQA等二选一真实性基准存在表层特征泄漏问题，模型无需真正推理即可借助答案表面特征超越随机水平，作者据此提出Audit-Prune清理机制并发布了泄漏接近随机水平的清理版数据集。

    

    二选一的真实性基准测试要求模型在正确和错误答案之间做出选择，但如果这两个答案在表层特征上存在系统性差异，模型就可以在没有执行预期推理的情况下超越随机概率水平。我们证明这种失效模式是可以被检测到的，并且可以被下游分类器所利用。在TruthfulQA中，一个简单的六特征逻辑回归分类器就能在区分正确与错误答案方面取得相当高的准确率。我们进一步表明，类似的表层伪影也存在于其他基准测试中。为了解决这一问题，我们开发了一种通用机制，通过移除最具泄漏强化的答案对来清理这些基准。我们发布了一个将表层特征泄漏降低至接近随机水平的TruthfulQA版本，并提供了一种名为Audit-Prune的机制，以便数据集在发布前得以清理。

    arXiv:2609.13003v1 Announce Type: new  Abstract: Binary-choice truth benchmarks ask models to choose between a correct and an incorrect answer, but if the two answers differ systematically in surface-level features, models can exceed chance without performing the intended reasoning. We show that this failure mode is detectable and can be exploited by downstream classifiers. In TruthfulQA, a simple six-feature logistic classifier achieves substantial accuracy in separating correct from incorrect answers. We further show that similar surface-level artifacts are present in additional benchmarks. To counteract this, we developed a general mechanism to clean them by removing the most leakage-reinforcing pairs. We release a version of TruthfulQA with surface-feature leakage reduced close to chance and provide a mechanism, Audit-Prune, so that the datasets can be cleaned before release.
    
[^10]: 研究时序运动特征在姿态到文本印度手语翻译中的应用

    Investigating Temporal Motion Features for Pose-to-Text Indian Sign Language Translation

    [https://arxiv.org/abs/2609.12993](https://arxiv.org/abs/2609.12993)

    在WSLP 2026印度手语翻译共享任务中，为轻量级T5-small模型添加显式帧间运动特征带来了最大的性能提升，显著改善了BLEU分数。

    

    我们研究了预训练T5模型规模和显式运动特征对姿态到文本印度手语翻译（SLT）的影响，面向WSLP 2026共享任务。姿态序列通过轻量级姿态编码器投影到T5的嵌入空间中，并对完整模型进行微调以生成英文文本。本研究使用的共享任务数据包含一个含5,334个样本的测试集和一个含5,257个样本的验证集。我们比较了T5-small、T5-base和T5-large，并额外引入了一个运动增强变体T5-small + Motion，它在输入表示中加入了显式的帧间姿态差异。在仅使用空间特征的模型中，T5-small取得了最佳的BLEU和ROUGE分数，而T5-large获得了最高的chrF分数。用运动特征增强T5-small带来了本研究中观察到的最大单项提升，相比仅空间特征的基线显著提高了BLEU分数。

    arXiv:2609.12993v1 Announce Type: cross  Abstract: We investigate the effect of pretrained T5 model scale and explicit motion features on pose-to-text Indian Sign Language Translation (SLT) for the WSLP 2026 Shared Task. Pose sequences are projected into the embedding space of T5 through a lightweight pose encoder, with the complete model fine-tuned to generate English text. The shared task data used for this work consists of a test set with 5,334 examples and a validation set with 5,257 examples. We compare T5-small, T5-base, and T5-large, and additionally introduce a motion-augmented variant, T5-small + Motion, that adds explicit frame-to-frame pose differences to the input representation. T5-small achieves the best BLEU and ROUGE scores among the spatial-only models, while T5-large obtains the highest chrF score. Augmenting T5-small with motion features yields the largest single improvement observed in our study, substantially improving BLEU over the spatial-only baseline and making
    
[^11]: 更少的词，而非更少的词元：衡量每个命题的梵语分词惩罚

    Fewer Words, Not Fewer Tokens: Measuring the Sanskrit Tokenization Penalty per Proposition

    [https://arxiv.org/abs/2609.12960](https://arxiv.org/abs/2609.12960)

    该研究发现，尽管梵语训练的BPE分词器看似每个命题比英语更便宜，但在与同规模英语对照组严格对比后，梵语在每个意义单位上仍然承担显著的分词惩罚。

    

    梵语将格、数、人称和时态融合在词尾中，并将从句串联成复合词，因此每个词的信息密度很高。但这种密度是否能在子词分词后得以保留是另一个问题，应该以意义单位而非词为单位来提问。在相同的FLORES-200开发测试集内容上，在使用词表包含200,019个或更多标识符的现有分词器时，梵语的词元成本是英语的1.774-2.187倍，但仅为印地语词元的1.325-1.353倍。与现有的英语分词器相比，经过梵语训练的BPE分词器在当代散文文本上每个命题看起来比英语更便宜（0.887）。然而，与匹配的英语对照组（即使用相同算法和词表大小、在同一语料库英语部分训练的分词器）相比，这种反转消失了：在32,000和64,000词元大小下，所有8个匹配对（每个大小匹配的组分别对照成对匹配和字节匹配的对照组）在散文上的比率均高于1.0，其95%置信区间均不包含1.0。差距在……（原文在此处截断）

    arXiv:2609.12960v1 Announce Type: new  Abstract: Sanskrit fuses case, number, person and tense into word endings and chains clauses into compounds, so it is information-dense per word. Whether that density survives subword tokenization is a separate question, to be asked per unit of meaning rather than per word. On identical FLORES-200 devtest content, Sanskrit costs 1.774-2.187 times the English tokens under deployed tokenizers with vocabularies of 200,019 ids or more, but only 1.325-1.353 times the Hindi tokens. Against a deployed English tokenizer, Sanskrit-trained BPE arms then look cheaper per proposition than English on contemporary prose (0.887). Against a matched English control, the same algorithm and vocabulary trained on the English side of the same corpus, that flip disappears: at 32,000 and 64,000 pieces all 8 matched pairs, each size-matched arm against both a pair-matched and a byte-matched control, sit above 1.0 on prose with 95% intervals excluding it. The gap closes a
    
[^12]: PA-CDM：用于评估手写数学表达式识别的位置感知字符检测匹配

    PA-CDM: Position-Aware Character Detection Matching for Evaluating Handwritten Mathematical Expression Recognition

    [https://arxiv.org/abs/2609.12917](https://arxiv.org/abs/2609.12917)

    提出了PA-CDM位置感知评估指标，通过将字符检测匹配与位置森林编码和散度级加权相结合，解决了现有手写数学表达式识别评估方法无法感知错误发生位置的缺陷。

    

    手写数学表达式识别（HMER）传统上通过精确匹配率和字符串相似度指标进行评分，这些指标无法感知错误发生的位置：两个具有相同标记错误数的预测，无论是下标放错位置还是交换了分数的操作数，都会得到相同的评分。基于渲染的字符检测匹配（CDM）能够稳健地对齐字形，但仍然对位置不敏感——在受控的分数操作数交换测试中，其得分为0.8595，而位置感知评分应为0.6253。树编辑指标则表现出互补的盲点：解析器规范化覆盖范围之外的重写会被当作结构错误而受到惩罚（得分为0.8552，而基于渲染的指标得分为1.0）。我们提出了PA-CDM，这是一种位置感知指标，它将字符检测匹配与位置森林编码和散度级加权相结合；同时提出了StructPerturb v2.0，这是一个涵盖15种类型、共1,340对受控扰动的冻结基准测试集。

    arXiv:2609.12917v1 Announce Type: cross  Abstract: Handwritten mathematical expression recognition (HMER) is conventionally scored by exact-match rates and string-similarity metrics that are blind to where an error occurs: two predictions with identical token-error counts receive identical scores whether they misplace a subscript or swap the operands of a fraction. Render-based character detection matching (CDM) aligns glyphs robustly but remains position-blind---on controlled fraction-operand swaps it scores 0.8595 where position-aware scoring yields 0.6253. Tree-edit metrics exhibit a complementary blind spot: rewrites outside the parser's normalization coverage are penalized as structural errors (0.8552 where render-based metrics score 1.0). We propose PA-CDM, a position-aware metric that couples character detection matching with position-forest encoding and divergence-level weighting; StructPerturb v2.0, a frozen benchmark of 1,340 controlled perturbation pairs across 15 type--inte
    
[^13]: 面向大规模多标签文本分类的LLM增强双分支学习

    LLM-Enhanced Dual-Branch Learning for Large-Scale Multi-Label Text Classification

    [https://arxiv.org/abs/2609.12915](https://arxiv.org/abs/2609.12915)

    该论文提出DualMLC双分支框架，同时利用自回归解码器式大语言模型和双向编码器对同一文档建模，并通过后期logit融合两分支得分，充分挖掘异构语言模型的互补性，以提升大规模多标签文本分类的性能。

    

    大规模多标签文本分类需要从一个包含数千乃至数万个候选标签的词表中，为每个文档分配一个小规模的相关标签子集。尽管预训练语言模型已经改善了语义文本表示，但大多数基于表示的方法将其预测流程集中于单一主编码器，或在单一排序器中组合辅助特征。因此，异构语言模型之间的互补性仍未得到充分探索。我们提出了DualMLC，这是一个双分支框架，通过自回归的解码器式语言模型和双向编码器来处理同一文档。每个分支维护各自的表示路径，并在共享标签空间上独立估计相关性得分。DualMLC通过后期logit融合将两个得分向量结合起来，使共享证据能够强化相关标签，而分支特有的证据则能够……

    arXiv:2609.12915v1 Announce Type: new  Abstract: Large-scale multi-label text classification assigns a small subset of relevant labels to each document from a vocabulary containing thousands or tens of thousands of candidate labels. Although pretrained language models have improved semantic text representations, most representation-based approaches center their prediction pipelines on a primary encoder or combine auxiliary features within a single ranker. The complementarity between heterogeneous language models therefore remains insufficiently explored. We propose DualMLC, a dual-branch framework that processes the same document through an autoregressive decoder-only language model and a bidirectional encoder. Each branch maintains its own representation pathway and independently estimates relevance scores over the shared label space. DualMLC combines the two score vectors through late logit fusion, allowing shared evidence to reinforce relevant labels and branch-specific evidence to 
    
[^14]: 面向波兰语和欧洲语言的参数高效检索器

    Parameter-Efficient Retrievers for Polish and European Languages

    [https://arxiv.org/abs/2609.12913](https://arxiv.org/abs/2609.12913)

    提出了一种结合跨语言对齐、关系知识蒸馏和对比微调的三阶段训练流程，无需原始相关性标注即可训练出参数量小但性能可媲美大型模型的波兰语和欧洲多语言稠密检索器。

    

    密集检索系统日益依赖数十亿参数规模的语言模型，其内存和计算需求使得大规模索引、频繁的语料库更新以及低延迟服务变得成本高昂。我们提出了一套三阶段训练流程，用于开发紧凑且高效的检索器，这些检索器在与体量大得多的模型竞争时仍具有竞争力。该流程结合了跨语言对齐、关系知识蒸馏和对比微调。它不需要原始的真实相关性标注，完全依赖由作为教师的强嵌入模型和重排序器所生成的监督信号。利用该流程，我们开发了PolDense和EuroDense，两者均支持长达8,192个token的上下文。PolDense是一系列参数量从1700万到10亿不等的六个波兰语检索器。EuroDense是一个支持九种欧洲语言的4.35亿参数检索器。我们进行了广泛的评估，涵盖41个波兰语

    arXiv:2609.12913v1 Announce Type: new  Abstract: Dense retrieval systems increasingly rely on multi-billion-parameter language models, whose memory and computational requirements make large-scale indexing, frequent corpus updates, and low-latency serving costly. We present a three-stage training pipeline for developing compact and efficient retrievers that remain competitive with substantially larger models. The pipeline combines cross-lingual alignment, relational knowledge distillation, and contrastive fine-tuning. It requires no original ground-truth relevance labels, relying exclusively on supervision generated by strong embedding models and rerankers utilised as teachers. Using this pipeline, we develop PolDense and EuroDense, both supporting contexts of up to 8,192 tokens. PolDense is a family of six Polish retrievers ranging from 17M to 1B parameters. EuroDense is a 435M-parameter retriever supporting nine European languages. We conduct an extensive evaluation covering 41 Polish
    
[^15]: MedSNIP：构建并基准化测试片段级粒度的医学事实验证

    MedSNIP: Building and Benchmarking Snippet-Level Granularity for Medical Fact Verification

    [https://arxiv.org/abs/2609.12884](https://arxiv.org/abs/2609.12884)

    该论文将医学事实核查重新构建为片段级验证范式，并提出了人工标注基准MedSNIP-Bench与自动片段生成流水线MedSNIP，以保留声明周围的局部临床结构并解决原子级分解导致的临床信息不完整问题。

    

    一项医学声明的正确性往往不仅取决于声明本身，还取决于其周围的临床结构。一项声明可能需要实验室参考范围、因果或条件联系，或患者特定的细节才能被正确判断，而原子级分解会割裂这些依赖关系，使验证者面对临床上不完整的声明。我们围绕片段级验证重新构建医学事实核查，其中按子句分组的单元保留了局部临床结构。我们介绍了MedSNIP-Bench，一个用于片段级医学事实验证的人工标注基准，以及MedSNIP，一个自动片段生成流水线。MedSNIP-Bench涵盖276份消费者健康和临床情景病例回答，被切分为2,524个片段，带有“一般情境下”和“患者情境下”的双重标签以及六种结构模式编码。MedSNIP在MedSNIP-Bench上根据人工标注的片段边界进行评估，随后被用于生成片段级单元

    arXiv:2609.12884v1 Announce Type: new  Abstract: A medical claim's correctness often depends not on the claim alone, but on the clinical structure around it. A claim may require a lab reference range, a causal or conditional link, or patient-specific details to be judged correctly, and atom-level decomposition can fragment these dependencies, leaving the verifier with clinically incomplete claims. We reformulate medical fact-checking around snippet-level verification, where clause-grouped units preserve local clinical structure. We introduce MedSNIP-Bench, a human-annotated benchmark for snippet-level medical fact verification, and MedSNIP, an automatic snippet-generation pipeline. MedSNIP-Bench covers 276 consumer-health and clinical-vignette responses, segmented into 2,524 snippets with dual in-general and in-patient-context labels and six structural pattern codes. MedSNIP is evaluated against human snippet boundaries on MedSNIP-Bench and then used to generate snippet-level units for
    
[^16]: DuplexDrama：一个包含场景设定、全双工行为、富表现力语音和声音事件的合成对话数据集

    DuplexDrama: A Synthesized Dialogue Dataset with Scenarios, Full-Duplex Behaviors, Expressive Speech, and Sound Events

    [https://arxiv.org/abs/2609.12872](https://arxiv.org/abs/2609.12872)

    DuplexDrama是首个同时涵盖完整人设场景、三种全双工行为、带情感标签的富表现力语音和剧本感知声音事件四个维度的合成口语对话数据集，包含超过2,000小时音频，并将发布800小时中英双语子集以推动全双工口语对话模型研究。

    

    我们提出了DuplexDrama，这是首个同时涵盖四个维度的合成口语对话数据集：(i) 完整的人设与场景设定；(ii) 三种全双工行为（打断、附和、未完成话语）；(iii) 带有与人设对齐的情感标签的富表现力语音；(iv) 剧本感知的声音事件。DuplexDrama通过4阶段流水线构建，对剧本和合成音频的质量验证证实了其质量。我们产出了超过2,000小时的音频数据，使用了包含64种音色的音色库，涵盖13个人设和5个年龄段；所有对话轮次中有3.8%包含至少一种全双工行为。这些数据已通过内部全双工模型训练得到验证。我们将发布一个包含6,400条双语对话的精选子集（800小时，中文约500小时 + 英文约300小时），以推动全双工口语对话模型的研究。数据样本可在我们的演示页面获取，LLM评判评估提示词也将被发布。

    arXiv:2609.12872v1 Announce Type: new  Abstract: We present DuplexDrama, the first synthesized spoken dialogue dataset that simultaneously covers four dimensions: (i) complete persona and scenario settings; (ii) three full-duplex behaviors (interruption, backchannel, incomplete); (iii) expressive speech with persona-aligned emotion labels; and (iv) script-aware sound events. DuplexDrama is built via a 4-stage pipeline; quality validation on both scripts and synthesized audio confirms its quality. We have produced more than 2,000 hours audio data with a 64-voice timbre pool spanning 13 personas and 5 age buckets; 3.8% of all turns carry at least one full-duplex behavior. This data has been validated through internal full-duplex model training. We will release a curated subset of 6,400 bilingual dialogues (800 h, Chinese ~500 h + English ~300 h) to advance full-duplex spoken dialogue model research. Data samples are available at our demo page and LLM-judge evaluation prompts will be rele
    
[^17]: 图上认知：通过认知循环与图-文本双向协同导航海量知识空间

    Cognition on Graph: Navigating Massive Knowledge Space via Cognitive Cycles and Bidirectional Graph-Text Synergy

    [https://arxiv.org/abs/2609.12791](https://arxiv.org/abs/2609.12791)

    提出了CoG框架，一个受认知启发、无需训练的自适应知识探索方法，通过持续的“计划-探索-反思”认知循环和图-文本深度双向协同，实现海量异构知识库中的复杂推理导航。

    

    检索增强生成（RAG）已使大型语言模型（LLM）能够处理知识密集型任务。然而，在全局、异构的知识库（大规模知识图谱和文本语料库）中进行导航以完成复杂推理仍然是一个挑战。现有方法通常采用反应式、图驱动的探索策略，盲目遵循图拓扑结构，而无法适应问题上下文或不断演化的探索进度，并且缺乏图与文本之间的深度双向协同。为了解决这些局限性，我们提出了CoG（Cognition on Graph），一个受认知启发、无需训练的自适应知识探索框架。受人类解决问题方式的启发，CoG执行持续的“计划-探索-反思”循环，主动制定调查计划，执行双源检索，并动态反思进度以调整策略。至关重要的是，它建立了深度的双向图与文本协同机制。

    arXiv:2609.12791v1 Announce Type: new  Abstract: Retrieval-Augmented Generation (RAG) has empowered Large Language Models (LLMs) to tackle knowledge-intensive tasks. However, navigating global, heterogeneous knowledge bases (large-scale knowledge graphs and text corpora) for complex reasoning remains a challenge. Existing methods typically employ reactive, graph-driven exploration strategies, which blindly follow graph topology without adapting to the question context or evolving exploration progress, and lack deep bidirectional synergy between graph and text. To address these limitations, we propose CoG (Cognition on Graph), a cognitive-inspired, training-free framework for adaptive knowledge exploration. Drawing inspiration from human problem-solving, CoG performs a continuous plan-explore-reflect cycle, where it proactively formulates investigation plans, performs dual-source retrieval, and dynamically reflects on progress to adjust strategies. Crucially, it establishes deep bidirec
    
[^18]: 什么驱动了智能体文本到Cypher的恢复能力？LAST-CQ：一种LLM智能体自我精炼框架

    What Drives Recovery in Agentic Text-to-Cypher? LAST-CQ: An LLM Agent Self-Refinement Framework

    [https://arxiv.org/abs/2609.12746](https://arxiv.org/abs/2609.12746)

    该研究通过LAST-CQ框架上的反事实实验发现，智能体Text-to-Cypher系统的恢复收益主要来自失败检测与重试路由机制，而非复杂的LLM反馈内容。

    

    用于结构化查询生成的智能体流水线正在迅速扩展，但尚不清楚循环中的哪个部分产生了收益。我们使用LAST-CQ——一个五智能体、免训练、基于执行结果的Text-to-Cypher框架——作为插桩测试平台，在2,471个实时数据库查询和跨越三个厂商规模层级的六个骨干模型上运行了三个反事实实验。移除纠错环节，相对于单次通过系统会损失3.1%的聚合执行BLEU，相对于无精炼的反事实会损失12.3%（对于最弱的骨干模型高达80.7%）。用原始数据库错误字符串替换基于模式、由LLM合成的反馈几乎没有任何损失（20.9% vs 19.9%朴素精确匹配；端到端差异<0.2%；通过两次单侧检验证明在±0.075集合F1范围内等价）。将相同的调用预算用于并行采样会使质量下降10-11%。真正有效的是检测失败并将其路由到重试，而不是反馈内容的复杂精妙。

    arXiv:2609.12746v1 Announce Type: cross  Abstract: Agentic pipelines for structured-query generation are rapidly expanding, but it is unclear which part of the loop produces the gain. We use LAST-CQ -- a five-agent, training-free, execution-grounded Text-to-Cypher framework -- as an instrumented testbed, running three counterfactuals over 2,471 live-database queries and six backbones spanning three vendor scale tiers. Removing correction is worth between 3.1% aggregate execution-BLEU against the single-pass system and 12.3% against a no-refinement counterfactual (up to 80.7% for the weakest backbone). Replacing schema-grounded, LLM-synthesised feedback with raw database error strings costs almost nothing (20.9% vs. 19.9% naive exact match; <0.2% end-to-end; equivalent within $\pm 0.075$ set-F1 by two one-sided tests). Spending the same call budget on parallel sampling degrades quality by 10-11%. What works is detecting failure and routing it to a retry, not the feedback sophistication 
    
[^19]: 基于残差向量重构的长上下文召回方法：不受上下文窗口大小限制

    Residual Vector-based Reconstruction as Long-Context Recall Regardless of Context Window Size

    [https://arxiv.org/abs/2609.12686](https://arxiv.org/abs/2609.12686)

    提出了一种基于残差向量的长上下文召回方法，通过利用LLM前馈层中存储的残差向量确定性重构与查询相关的事实，无需额外训练或微调即可在上下文长度增加时保持近乎恒定的GPU内存使用。

    

    大型语言模型（LLM）能够处理长上下文，包括长文档和冗长的对话，但其token级内存使用量会随输入长度成比例增加。尽管模型优化和有损提示压缩被广泛应用，这些方法仍然无法解决超出预训练和大小受限上下文窗口的长上下文召回问题。本文提出了一种长上下文召回方法，随着上下文长度的增加，该方法能保持近乎恒定的GPU内存使用，且无需额外训练。其主要思想是利用LLM前馈层中的参数激活来重构事实，这些前馈层存储着代表源文档事实的残差向量。利用残差向量，LLM能够在不参考原始文档的情况下确定性地重构与查询相关的事实，在保持高保真度的同时降低内存使用，且无需微调权重。实验结果表……

    arXiv:2609.12686v1 Announce Type: cross  Abstract: Large language models (LLMs) process long contexts, including long documents and lengthy conversations, but face token-level memory usage that increases proportionally to input length. Although model optimization and lossy prompt compression are widely used, these methods still fail to solve the long-context recall problem beyond pretrained and size-constrained context windows. This paper proposes a long-context recall method that maintains near-constant GPU memory usage as context length increases, without additional training. The main idea is to reconstruct facts using parameter activations in the LLM's feed-forward layers, which store residual vectors representing facts from the source document. Utilizing residual vectors allows the LLM to deterministically reconstruct query relevant facts without referencing the original document, preserving high fidelity and reducing memory usage without fine-tuning weights. Experimental results s
    
[^20]: Doc2FRC：通过固定范围分块实现长度一致的文档级机器翻译

    Doc2FRC: Length-Consistent Document-Level Machine Translation via Fixed-Range Chunking

    [https://arxiv.org/abs/2609.12674](https://arxiv.org/abs/2609.12674)

    提出基于动态规划的固定范围分块方法FRC，在训练和推理中一致应用，将任意长度的文档映射到相同的长度分布，从而解决文档级机器翻译中训练与推理的长度分布不匹配问题。

    

    具有长上下文窗口的先进大语言模型（LLM）能够大幅减少文档级机器翻译中的输入截断问题。然而，直接的Doc2Doc翻译仍然容易出现n-gram重复和质量逐渐下降的问题。一种常见的补救措施是将文档分割成更细粒度的块。然而，传统的基于规则的分块方法无法处理训练和推理之间的长度分布不匹配问题。为了解决这一问题，我们提出了固定范围分块，利用动态规划将文档划分为预定义长度区间内的块。通过在训练和推理阶段一致地应用FRC，任意长度的输入文档都会被映射到相同的长度分布，从而显著减少了训练与测试之间的长度不匹配。围绕FRC，我们提出了一种轻量级的双边界匹配算法用于分块对齐，以及四种不同的训练策略。

    arXiv:2609.12674v1 Announce Type: new  Abstract: Advanced large language models (LLMs) with long context windows can substantially reduce input truncation in document-level machine translation (DocMT). However, direct Doc2Doc translation remains prone to n-gram repetition and progressive quality degradation. A common remedy is to segment the document into finer-grained chunks. Nonetheless, conventional rule-based chunking approaches fail to handle the length distribution mismatch between training and inference. To address this, we introduce Fixed-Range Chunking (FRC), utilizing dynamic programming to partition documents into chunks within a predefined length interval. By consistently applying FRC during training and inference, the input documents of any length are mapped to the same length distribution, substantially reducing train-test length mismatch. Centered on FRC, we propose a lightweight dual-boundary matching algorithm for chunk alignment, alongside four distinct training strat
    
[^21]: LifeMem：实现大语言模型智能体的终身经验复用

    LifeMem: Enabling Lifelong Experience Reuse for LLM Agents

    [https://arxiv.org/abs/2609.12655](https://arxiv.org/abs/2609.12655)

    LifeMem是一个终身学习框架，通过基于底层工作流聚类交互轨迹来提取可复用技能，使大语言模型智能体能够跨多个环境迁移和复用经验，同时缓解灾难性遗忘问题。

    

    大语言模型智能体被期望通过复用过往经验，在其整个生命周期内持续适应新任务和新环境。然而，现有的基于记忆的智能体难以在不同环境之间迁移可复用的经验，并且随着经验的积累会遭受灾难性遗忘。为了应对这些挑战，我们提出了LifeMem，这是一个使智能体能够在多个环境之间迁移知识的终身学习框架。在学习过程中，LifeMem基于底层工作流对积累的交互轨迹进行聚类，以提取可复用的技能。在推理阶段解决新任务时，智能体会回忆相关的技能和轨迹来指导行动。为了验证我们的方法，我们在10个环境和超过1.3万个任务上进行了实验，并使用了2000条新标注的交互轨迹。结果表明，LifeMem能够在终身学习中实现有效的经验复用，同时减少了遗忘。

    arXiv:2609.12655v1 Announce Type: new  Abstract: Large language model agents are expected to continuously adapt to new tasks and environments over their lifetime by reusing past experience. However, existing memory-based agents struggle to transfer reusable experience across environments and suffer from catastrophic forgetting as experience accumulated. To address these challenges, we propose LifeMem, a lifelong learning framework that enables agents to transfer knowledge across multiple environments. During learning, LifeMem clusters accumulated interaction trajectories based on underlying workflows to extract reusable skills. When solving a new task at inference time, the agent recalls relevant skills and trajectories to guide actions. To validate our method, we conduct experiments across 10 environments and over 13k tasks with 2k newly annotated interaction trajectories. Results show that LifeMem enables effective experience reuse in lifelong learning, achieving both reduced forgett
    
[^22]: SWARM：用于搜索引擎结果中俄罗斯宣传检测的多语言人工标注数据集

    SWARM: A Multilingual Human-Annotated Dataset for Russian Propaganda Detection in Search Engine Results

    [https://arxiv.org/abs/2609.12653](https://arxiv.org/abs/2609.12653)

    该论文构建了首个针对搜索引擎结果中俄罗斯宣传检测的多语言人工标注数据集SWARM，并通过实验表明基于来源的黑名单会遗漏大多数宣传内容，而基于内容的大语言模型分析效果更佳。

    

    俄罗斯国家宣传在多种语言和网络空间中传播。然而，大多数计算研究仅考察其中一个空间（通常是社交媒体）、一种或两种语言，且分析的是信息来源而非内容。我们介绍SWARM（多语言俄语宣传标注搜索网页文档），这是一个包含2,183条搜索引擎结果的数据集，涵盖九种语言和多样化的网络领域（如新闻、博客、政府网站），每条结果均由训练有素的标注员标注是否支持反复出现的俄罗斯宣传叙事。我们以这些标签为基准，测试了基于来源的黑名单方法、监督分类器和零样本大语言模型。黑名单遗漏了大多数支持宣传的文档，因为此类内容并不局限于被标记的“宣传”媒体，也出现在主流媒体上。内容层面的分析有所帮助，但效果因模型而异：最强的大语言模型在正类别上达到0.73的F1值，而监督……

    arXiv:2609.12653v1 Announce Type: new  Abstract: Russian state propaganda spreads across many languages and online spaces. Yet, most computational work examines only one such space, usually social media, in one or two languages, and analyses sources rather than content. We introduce SWARM (Search-Web documents Annotated for Russian propaganda, Multilingual), a dataset of 2,183 search engine results across nine languages and diverse web domains (e.g., news, blogs, government sites), each annotated by trained coders for whether it supports a recurring Russian propaganda narrative. We benchmark a source-based blocklist, supervised classifiers, and zero-shot LLMs against these labels. The blocklist misses most propaganda-supporting documents, because such content is not confined to flagged "propaganda" outlets but also appears on mainstream ones. Content-level analysis helps, though how much depends on the model: the strongest LLM reaches a positive-class F1 of 0.73, whereas the supervised
    
[^23]: SteerDuplex：可控的双工语音对话模型

    SteerDuplex: Steerable Duplex Speech Dialogue Models

    [https://arxiv.org/abs/2609.12623](https://arxiv.org/abs/2609.12623)

    该论文提出SteerDuplex，一个基于Moshi的可控全双工语音对话模型，通过可控性分类法、多目标微调、混合奖励的两阶段强化学习以及SteerBench评测基准，使模型能够根据用户指令在语气、人设、语速和语音风格等属性上可靠地调整对话行为。

    

    全双工语音对话模型支持低延迟的轮次转换、打断处理和附和反馈，但一项关键能力仍未得到充分探索：可控性（steerability），即能够根据用户指令在语气、人设、语速和语音风格等属性上可靠地改变对话行为。我们提出了一个涵盖基于文本和基于音频的可控性分类法，并指出了当前全双工模型存在的重大能力缺口。为弥补这一缺口，我们提出了SteerDuplex，这是一个基于Moshi的全双工语音模型，在自然对话以及针对指令遵循、语音表达、推理和双工交互的合成对话上进行了微调。我们进一步采用了带有混合奖励的两阶段强化学习（RL），结合可验证的交互检查和基于评判模型的语义反馈，以提升时机把握和响应连贯性。为评估全双工语音可控性，我们引入了SteerBench，一个（用于评估的基准数据集）。

    arXiv:2609.12623v1 Announce Type: cross  Abstract: Full-duplex spoken dialogue models support low-latency turn taking, interruption handling, and backchanneling, yet a key capability remains underexplored: steerability, the ability to reliably shift conversational behavior along attributes such as tone, persona, speaking rate, and voice style in response to user instructions. We introduce a taxonomy of text- and audio-based steerability that identifies substantial gaps in current full-duplex models. To address this gap, we introduce SteerDuplex, a Moshi-based full-duplex speech model fine-tuned on natural conversations and synthetic dialogues targeting instruction following, vocal delivery, reasoning, and duplex interaction. We further apply two-stage reinforcement learning (RL) with hybrid rewards, combining verifiable interaction checks and judge-based semantic feedback to improve timing and response continuity. To evaluate full-duplex spoken steerability, we introduce SteerBench, a 
    
[^24]: 多模态语言模型中的校准歧义：人类借助文化引用，而模型只会描述画面

    Calibrated Ambiguity in Multimodal Language Models: Humans reach for cultural references, while models describe the picture

    [https://arxiv.org/abs/2609.12575](https://arxiv.org/abs/2609.12575)

    该研究借助桌面游戏Dixit的任务首次揭示了多模态语言模型在歧义处理上与人类的根本差异：模型表现出“歧义坍缩”和“文化扁平化”，而人类能生成兼具开放性与可解读性的校准歧义线索。

    

    歧义通常被视为AI系统需要解决的缺陷——但在人类的交流与文化中，歧义也可以是一种富有创造性的资源。从幽默到政治再到艺术，人们所使用的文字和图像既足够开放、能够引发不同的解读，又足够受约束、可以被理解。我们通过一个源自桌面游戏Dixit（妙语说书人）的任务来操作化这种“校准歧义”的概念。我们基于一套新颖的校准歧义编码规则，比较了人类与多模态语言模型所生成提示线索的差异，发现模型始终表现出“歧义坍缩”现象（即其输出过度具体，没有为多种合理解读留下空间）。与人类线索不同，AI生成的线索还表现出“文化扁平化”现象：即使被提示使用典故和比喻性语言，它们也几乎从不引用具有文化情境的知识。

    arXiv:2609.12575v1 Announce Type: new  Abstract: Ambiguity is often treated as a bug for AI systems to resolve---but in human communication and culture, ambiguity can also be a generative resource. From humour to politics to art, people express themselves in words and images that are open enough to invite different interpretations, yet constrained enough to be interpretable. We operationalise this notion of calibrated ambiguity with a task drawn from the parlour game Dixit. We compare differences in clues generated by human vs multimodal language models, based on a novel coding rubric for calibrated ambiguity, and find that models consistently exhibit ambiguity collapse (i.e., their outputs are over-specified, leaving no room for multiple legitimate interpretations). Unlike human clues, AI-generated clues also exhibit cultural flattening; they almost never make reference to culturally-situated knowledge, even when prompted to use allusion and figurative language.
    
[^25]: Meddies-PII：一个用于临床去标识化中个人身份信息提取的多语言框架

    Meddies-PII: A Multilingual Framework for Personally Identifiable Information Extraction in Clinical De-identification

    [https://arxiv.org/abs/2609.12544](https://arxiv.org/abs/2609.12544)

    该论文提出了Meddies-PII框架，包含一个通过属性条件提示生成、经十三个确定性门控验证、涵盖十七种语言的一百万份合成临床文档数据集，以及基于该数据集训练的BIOES分类模型，后者在十五个外部基准上以平均F1 0.827达到了现有PII提取系统中的最高性能。

    

    临床去标识化依赖于准确识别个人身份信息（PII）。然而，人工标注的数据集构建成本高昂，而现有的合成替代方案通常对其生成过程提供的细节有限，或依赖相对简单的合成策略。我们推出了Meddies-PII数据集，这是一个包含一百万份合成临床文档的语料库，涵盖十七种语言和九种PII标签。这些文档采用属性条件提示生成，并通过十三个确定性门控进行验证，以确保结构和标注的一致性。为评估该数据集的实用性，我们训练了Meddies-PII模型——一个BIOES标记分类器，并使用精确匹配的实体级F1分数将其与现有的PII提取系统进行比较。Meddies-PII模型在所有报告的基准测试中均取得了所评估系统中的最高性能，在十五个外部基准上的平均F1为0.827。

    arXiv:2609.12544v1 Announce Type: new  Abstract: Clinical de-identification relies on accurately identifying personally identifiable information (PII). However, manually annotated datasets are costly to construct, while existing synthetic alternatives often provide limited details about their generation process or rely on relatively simple synthesis strategies. We introduce Meddies-PII-Dataset, a corpus of one million synthetic clinical documents spanning seventeen languages and nine PII labels. The documents are generated using attribute-conditioned prompts and validated through thirteen deterministic gates that enforce structural and annotation consistency. To evaluate the dataset's utility, we train Meddies-PII-Model, a BIOES token classifier, and compare it with existing PII extraction systems using exact-match entity-level F1. Meddies-PII-Model achieves the highest performance among the evaluated systems on all reported benchmarks, with a mean F1 of 0.827 across fifteen external b
    
[^26]: 智能体作为策略的机器人操作

    Agent as Policy for Robotic Manipulation

    [https://arxiv.org/abs/2609.12541](https://arxiv.org/abs/2609.12541)

    该论文提出Agent as Policy (AGP)方法，使通用智能体无需任何任务特定训练即可直接控制物理机器人完成从精细操作、动态运动到可变形物体处理等多种真实操作任务。

    

    我们证明了通用智能体可以在整个任务执行过程中直接驱动物理机器人，而无需任何针对特定任务或特定环境的训练。我们提出了智能体作为策略方法，将任务规划和执行都置于智能体的控制之下。给定一个任务和机器人接口，智能体能够解读视觉信息、编写可执行程序、发出运动指令，并根据物理执行结果修正自身动作。这将智能体的推理和编程能力带入了与物理世界的持续交互之中。我们在多个真实世界的操作任务上研究了AGP，涵盖精细操作、动态运动和可变形物体，包括从人类视频中学习装配、根据目标图像搭建积木、翻转骰子、定向投掷以及双臂协作折叠毛巾。AGP在三种积木搭建配置上分别取得了100%、100%和80%的成功率。

    arXiv:2609.12541v1 Announce Type: new  Abstract: We demonstrate that a general-purpose agent can directly drive a physical robot throughout task execution without any task-specific or environment-specific training. We introduce Agent as Policy (AGP), which places task planning and execution under the agent's control. Given a task and a robot interface, the agent interprets visual evidence, writes executable programs, issues motion commands, and revises its actions in response to physical outcomes. This brings the agent's reasoning and programming capabilities into continuous interaction with the physical world. We study AGP across multiple real-world manipulation tasks spanning precision manipulation, dynamic motions, and deformable objects. These include assembly from human videos, block construction from goal images, die reorientation, targeted throwing, and bimanual towel folding. AGP achieves success rates of 100%, 100%, and 80% on three block construction configurations. These fin
    
[^27]: 百万窗户之屋：用于叙事重构的互动小说

    The House with a Million Windows: Interactive Fiction for Narrative Restorying

    [https://arxiv.org/abs/2609.12537](https://arxiv.org/abs/2609.12537)

    本文提出基于LLM的互动小说系统HWAMW，通过让用户的故事以不同文学风格被“窗户”式重构，实证证明其能增强叙事认同感，从而为AI辅助写作确立了“LLM不替人讲述故事、而是拓展故事意义”的新范式。

    

    AI辅助写作可能会使人类叙事中的意义变得扁平化，在缺乏写作所需的刻意努力和意义建构的情况下生成同质化的内容。为应对这一挑战，我们提出了《百万窗户之屋》（HWAMW），这是一个基于大语言模型（LLM）的互动小说系统，旨在帮助用户探索个人故事中潜在意义的广度与深度——该系统借鉴了一种名为“叙事重构干预”的心理学范式。在HWAMW中，用户通过一段基于文本的叙事进行游戏：先讲述一个故事，随后会遭遇到一组由LLM生成的“窗户”，这些窗户以不同的文学风格对故事进行重新框架。实证证据表明，HWAMW增强了用户的叙事认同感，同时专家评审探讨了这一效果是如何实现的。我们的研究结果表明，HWAMW促进了叙事重构，并为AI辅助写作提供了一个有价值的范式——在这个范式中，LLM并不替我们讲述故事。

    arXiv:2609.12537v1 Announce Type: new  Abstract: AI-assisted writing can flatten meaning in human storytelling, enabling the production of homogeneous outputs without the intentional effort and sense-making writing entails. To address this challenge, we present The House with a Million Windows (HWAMW), an LLM-based interactive fiction system designed to help users explore both the breadth and depth of potential meanings within their personal stories -- drawing on a psychological paradigm called the restorying intervention. In HWAMW, users play through a text-based narrative in which they tell a story, then encounter a set of LLM-generated "windows" reframing it according to different literary styles. Empirical evidence shows that HWAMW increases users' sense of narrative identity, while an expert review explores how this effect is achieved. Our findings suggest that HWAMW facilitates restorying and offers a valuable paradigm for AI-assisted writing, wherein LLMs do not tell our stories
    
[^28]: Earth-Agent-Pro：迈向基于智能体的真实世界全链条地球观测

    Earth-Agent-Pro: Towards Real-World Full-Chain Earth Observation with Agents

    [https://arxiv.org/abs/2609.12533](https://arxiv.org/abs/2609.12533)

    Earth-Agent-Pro提出了一种执行自适应的规划-执行框架，通过专家技能约束规划、工作流中心的结构化记忆实现局部后缀修复，以及专用大语言模型适配器训练，首次实现了真实世界全链条地球观测任务的自动化执行。

    

    真实世界的地球观测（EO）智能体必须能够将高层次的科学问题转化为可执行的工作流，以获取观测数据、准备数据、执行领域计算，并从运行时证据中得出结论。现有的EO智能体通常从已提供的观测数据开始，而基准测试通常提供准备好的输入或候选答案，使得全链条开放世界EO执行在很大程度上尚未得到检验。我们提出了Earth-Agent-Pro，这是一种执行自适应的规划-执行（Plan-and-Execute）框架，利用专家编写的技能来约束规划与运行时工具使用。以工作流为中心的结构化记忆记录已规划的步骤、被接受的证据及其依赖关系，当运行时证据使某个步骤失效时，系统能够仅修复受影响的工作流后缀。独立的大语言模型适配器分别采用序列级监督微调用于规划器的工作流组合，以及节点级群组相对策略优化（摘要在此处截断）。

    arXiv:2609.12533v1 Announce Type: cross  Abstract: Real-world Earth observation (EO) agents must translate high-level scientific questions into executable workflows to acquire observations, prepare data, perform domain computations, and derive conclusions from runtime evidence. Existing EO agents typically start from supplied observations, while benchmarks typically provide prepared inputs or candidate answers, leaving full-chain open-world EO execution largely untested. We present Earth-Agent-Pro, an execution-adaptive Plan-and-Execute framework using expert-authored skills to constrain planning and runtime tool use. Workflow-centered structured memory records planned steps, accepted evidence, and their dependencies, enabling repair of only the affected workflow suffix when runtime evidence invalidates a step. Separate large language model adapters use sequence-level supervised fine-tuning for planner workflow composition and node-level group relative policy optimization with locally 
    
[^29]: 多智能体大语言模型预测中的信息专业化与约束合成：2026年FIFA世界杯前瞻性实时研究

    Information Specialization and Constrained Synthesis in Multi-Agent LLM Forecasting: A Prospective Live-Study of the 2026 FIFA World Cup

    [https://arxiv.org/abs/2609.12495](https://arxiv.org/abs/2609.12495)

    该研究在2026年FIFA世界杯最后56场比赛上开展实时前瞻性评估，证明多智能体LLM预测系统中的信息专业化角色（量化专家与新闻专家）能产生彼此不同的预测，且由专家、评论者和元智能体构成的四智能体顺序合成架构可提升预测效用。

    

    大语言模型正在被组织成具有专门角色的多智能体系统，但这种专业化是否能产生彼此不同的预测，以及后续的合成是否能提升效用，目前仍不清楚。在本研究中，我们对信息密集的2026年FIFA世界杯最后56场比赛进行了实时前瞻性评估，在保持前沿基础模型不变的前提下，为两个主要预测智能体分配了对比鲜明的专家角色：一个专注于结构化表现统计数据的量化专家，以及一个专注于最新伤病、战术和新闻发布会信息的新闻专家。两者的预测随后由一个独立的评论者进行审核，再由一个元智能体加以合成，从而构成一个顺序化的四智能体模型。来自博彩市场的预测被用作外部基准。新闻专家获得了最高的平均概率加权前三名效用，并与博彩市场（摘要内容在此处截断）

    arXiv:2609.12495v1 Announce Type: cross  Abstract: Large language models are being organized into multi-agent systems with specialized roles, but whether such specialization produces distinct forecasts and whether subsequent synthesis improves utility remains unclear. In this study, we carried out a live, prospective evaluation over the final 56 matches of the information-dense 2026 FIFA World Cup, keeping a frontier foundation model constant while assigning two primary forecasting agents contrasting specialist roles: a quantitative specialist focusing on structured performance statistics and a news specialist focusing on current injuries, tactics and information from press conferences. Their forecasts were then reviewed by a separate critic before being combined by a meta-agent, resulting in a sequential four-agent model. Forecasts from the betting market served as an external benchmark. The news specialist obtained the highest mean probability-weighted Top-3 utility and matched the b
    
[^30]: 面向代码重排序的置信度门控直推式测试生成

    Confidence-Gated Transductive Test Generation for Code Reranking

    [https://arxiv.org/abs/2609.12489](https://arxiv.org/abs/2609.12489)

    该论文提出置信度门控直推式测试生成方法 CoTT，仅在归纳置信度较低时才调用直推式生成，以更低的计算成本提升了代码重排序的效果。

    

    测试用例合成对于评估和排序由大型语言模型（LLM）生成的程序至关重要。然而，构建高质量的测试用例仍然具有挑战性，因为可靠的期望输出往往难以获得。我们提出了置信度门控直推式测试生成（CoTT），该方法首先使用高效的归纳式流程，仅在归纳置信度较低时才调用直推式生成。这种自适应设计在提高输出可靠性的同时，只在需要时才分配额外的计算。在代码重排序基准测试中，CoTT 在所报告的各项指标上均优于现有基线方法，同时相比对每个输入都应用直推式生成的方法降低了成本。这些结果表明，基于置信度的测试时计算分配能够在仅使用单个高效 LLM 的情况下提供良好的效率与效果权衡。

    arXiv:2609.12489v1 Announce Type: cross  Abstract: Test case synthesis is crucial for evaluating and ranking programs generated by large language models (LLMs). However, constructing high-quality test cases remains challenging because reliable expected outputs are often difficult to obtain. We propose Confidence-Gated Transductive Test Generation (CoTT), which first uses an efficient inductive procedure and invokes transductive generation only when inductive confidence is low. This adaptive design improves output reliability while allocating extra computation only when needed. On code reranking benchmarks, CoTT outperforms prior baselines across the reported metrics while reducing cost relative to applying transductive generation to every input. These results show that confidence-based allocation of test-time computation provides a favorable efficiency-effectiveness trade-off with a single efficient LLM.
    
[^31]: ZipBench：面向大语言模型综合基准压缩的低成本框架

    Zipbench: Low-Cost Framework for Compressing Comprehensive Benchmarks of Large Language Models

    [https://arxiv.org/abs/2609.12475](https://arxiv.org/abs/2609.12475)

    ZipBench提出了一种低成本的基准压缩框架，仅通过少量锚定模型的评估即可生成具有理论误差和排序一致性保证的紧凑基准子集，并能轻松扩展到新发布的基准，大幅降低了LLM评估成本。

    

    综合基准套件对于改进大语言模型（LLM）至关重要，但许多广泛使用的基准存在冗余，使得评估成本不必要地高昂。尽管近期的基准压缩方法（BCMs）可以缓解这一成本，但许多强大的基准压缩方法依赖于来自众多LLM的大量逐样本评估结果来识别代表性样本。构建此类数据集合同样成本高昂，除非它们已经公开，这使得这些方法难以扩展到新发布的基准。为了应对这一挑战，我们提出了ZipBench，这是一种简单且低成本的基准压缩方法，具有理论误差和排序一致性保证。ZipBench仅评估少量锚定LLM，合成伪评估结果以扩大覆盖范围，学习紧凑的样本表示，并选择一个小而具有代表性的子集。在此基础上，我们创建了ZipBench Zoo，这是一个包含100多个基准紧凑版本的集合。

    arXiv:2609.12475v1 Announce Type: new  Abstract: Comprehensive benchmark suites are essential for improving large language models (LLMs), but many widely used benchmarks are redundant, making evaluation unnecessarily expensive. Although recent benchmark compression methods (BCMs) can mitigate this cost, many strong BCMs rely on large collections of per-sample evaluation results from numerous LLMs to identify representative samples. Building such collections is also expensive unless they are already public, making these methods difficult to extend to newly released benchmarks. To address this challenge, we present ZipBench, a simple and low-cost BCM with theoretical error and rank-consistency guarantees. ZipBench evaluates only a small set of anchor LLMs, synthesizes pseudo evaluation results to broaden coverage, learns compact sample representations, and selects a small yet representative subset. Building on it, we create ZipBench Zoo, a collection of compact versions of 100+ benchmark
    
[^32]: AMDKernelVault：面向AMD GPU内核优化的大规模数据集与智能体训练

    AMDKernelVault: Large-Scale Datasets and Agentic Training for AMD GPU Kernel Optimization

    [https://arxiv.org/abs/2609.12471](https://arxiv.org/abs/2609.12471)

    AMDKernelVault通过智能体驱动的生成-验证流水线构建了面向AMD GPU的开放HIP/Triton内核语料库（含超过10万个执行验证样本），并证明经微调的Qwen3-8B在AMD内核生成任务上取得了领先正确率。

    

    我们介绍了AMDKernelVault，这是一个面向最新AMD CDNA GPU的开放HIP和Triton内核语料库及训练框架。现有的基于大语言模型的内核智能体大多以CUDA/NVIDIA为中心，并且往往依赖重复调用前沿大语言模型来进行生成、反思和优化。为了弥补这一空白，我们开发了HIPKernelGen和TritonKernelGen这两个由智能体驱动的流水线，它们能够将PyTorch参考实现转换为HIP或Triton内核，在ROCm环境下编译并验证候选内核，并在AMD硬件上进行延迟分析。该语料库包含62,153个经过执行验证的HIP内核样本、2,377个基于生产实践的ROCm库问答条目，以及39,893个Triton内核。我们进一步采用监督微调和执行感知强化学习对Qwen3-8B进行训练，以展示该语料库的实用价值。在固定评估预算下，该模型在PyTorch到HIP转换（34.0% Pass@1）和TritonBench-G（33.2%正确率）等基准上，在所比较的模型中取得了最高的正确率。

    arXiv:2609.12471v1 Announce Type: new  Abstract: We introduce AMDKernelVault, an open HIP and Triton kernel corpus and training framework for recent AMD CDNA GPUs. Existing LLM-based kernel agents are largely CUDA/NVIDIA-centric and often depend on repeated frontier-LLM calls for generation, reflection, and optimization. To address this gap, we develop HIPKernelGen and TritonKernelGen, agent-driven pipelines that transform PyTorch references into HIP or Triton kernels, compile and validate candidates under ROCm, and latency-profile them on AMD hardware. The corpus contains 62,153 execution-verified HIP kernel samples, 2,377 production-grounded ROCm Libraries QA entries, and 39,893 Triton kernels. We further train Qwen3-8B with supervised fine-tuning and execution-aware reinforcement learning as a demonstration of the corpus's utility. Under fixed evaluation budgets, it achieves the highest correctness among the compared models on PyTorch-to-HIP (34.0% Pass@1), TritonBench-G (33.2% Corr
    
[^33]: 并非所有语音都是意图：面向ASR后误唤醒的自适应自校正推理层

    Not All Speech Is Intent: Adaptive Self-Correcting Inference Layer for Post-ASR False Wake-Up

    [https://arxiv.org/abs/2609.12469](https://arxiv.org/abs/2609.12469)

    提出了ASCIL框架，在ASR转录后、响应生成前通过融合声学嵌入、语言线索、设备上下文以及用户的隐式与显式行为反馈信号（如犹豫、取消等），自适应地重新评估并纠正误唤醒的意图误分类。

    

    误唤醒激活一直是对话式人工智能中一个持续存在的挑战。与设备唤醒词在语音上相似的语音可能产生语法有效且语义连贯的ASR转录文本，导致助手错误地执行指令。大多数现有系统孤立地做出单一的意图决策，既没有从反复出现的错误中学习的机制，也无法通过个性化学习适应个体用户。我们提出了反馈驱动的自适应自校正推理层（ASCIL），这是一个互补性的ASR后校正框架，在响应生成之前，通过融合声学嵌入、语言线索、设备上下文以及过去错误分类的模式来重新评估唤醒意图。ASCIL将隐式信号（包括犹豫、脱离和沉默）以及显式信号（包括取消和重复）解释为自动推断出的、带有噪声的行为指标，用以识别潜在的意图误分类。

    arXiv:2609.12469v1 Announce Type: new  Abstract: False wake-up activations remain a persistent challenge in conversational AI. Speech phonetically similar to a device's wake word can produce a syntactically valid and semantically coherent ASR transcript that the assistant incorrectly executes. Most existing systems make a single intent decision in isolation, without a mechanism to learn from recurring errors over time or adapt to individual users through personalized learning. We introduce the Feedback-Driven Adaptive Self-Correcting Inference Layer (ASCIL), a complementary post-ASR correction framework that re-evaluates wake-up intent before response generation by fusing acoustic embeddings, linguistic cues, device context, and patterns from past misclassifications. ASCIL interprets implicit signals, including hesitation, disengagement, and silence, and explicit signals, including cancellation and repetition, as automatically inferred, noisy behavioral indicators of potential misclass
    
[^34]: GraphProfiler：基于个人知识图谱的源链接敏感属性推断

    GraphProfiler: Source-Linked Sensitive Attribute Inference via Personal Knowledge Graphs

    [https://arxiv.org/abs/2609.12448](https://arxiv.org/abs/2609.12448)

    GraphProfiler是一种可审计的LLM画像工具，通过将用户帖子历史构建为可溯源至原始帖子的个人知识图谱，在实现高精度敏感属性推断的同时，能够定位真正泄露隐私信息的关键帖子，从而支持针对性的隐私保护。

    

    诸如年龄、收入和职业等敏感属性可以通过聚合众多普通帖子中的间接线索，从用户生成的内容中被推断出来。基于大语言模型（LLM）的画像工具能够自动且高精度地执行这种聚合，这使得大规模个人属性推断成为一项重大的隐私威胁。然而，现有的基于LLM的画像工具对于究竟是哪些具体帖子、概念和关系使得推断成为可能缺乏洞察，而这正是实现针对性隐私缓解的关键，即仅对实际泄露属性的少数帖子进行删除或改写，而非扰动整个历史记录。我们提出了GraphProfiler，一个可审计的基于LLM的画像工具，它将每个用户的帖子历史表示为源链接的个人知识图谱，其中节点和边可追溯到原始帖子，并将属性预测解析为被引用的图记录和源文本。GraphProfiler达到了86.7%的攻击成功率……

    arXiv:2609.12448v1 Announce Type: new  Abstract: Sensitive attributes such as age, income, and occupation can be inferred from user-generated content by aggregating indirect cues across many ordinary posts. LLM-based profilers can perform this aggregation automatically and with high accuracy, which makes large-scale personal attribute inference a major privacy threat. Existing LLM-based profilers, however, offer limited insight into which specific posts, concepts, and relationships made an inference possible, which is key to targeted privacy mitigation, i.e., redacting or rewriting only the few posts that actually leak an attribute, rather than perturbing entire histories. We introduce GraphProfiler, an auditable LLM-based profiler that represents each user's post history as a source-linked personal knowledge graph where nodes and edges trace back to the originating post and resolves attribute predictions to cited graph records and source texts. GraphProfiler reaches 86.7% attack succe
    
[^35]: 大语言模型信任指控者还是指控内容？在狼人杀游戏中测量信念变化

    Do LLMs Trust the Accuser or the Accusation? Measuring Belief Shifts in Werewolf

    [https://arxiv.org/abs/2609.12446](https://arxiv.org/abs/2609.12446)

    该研究提出了一个基于狼人杀游戏的信念变化评估基准，通过标注怀疑与指控消息并测量观察模型的信念更新，发现大模型虽能更好识别狼人并抵御不信任者提出的指控，但仍会轻信受信任指控者的指控，即使对方是狼方阵营。

    

    摘要：狼人杀等社交推理游戏越来越多地被用于评估大语言模型（LLM）智能体，但现有的评估方法往往依赖于最终的游戏结果。我们提出了一个狼人杀中的信念变化评估基准，通过信念更新来分析沟通能力。利用大语言模型进行的游戏对局，我们对怀疑和指控消息进行标注，并测量一个作为观察者的村民方模型的信念在每条消息后如何变化。我们在1,224条标注消息上评估了40个开源权重大语言模型配置。结果表明，更大的模型能更好地根据游戏历史区分真正的狼人和村民，但指控仍然会强烈影响它们的信念。模型对被指控的目标变得更加怀疑，而对指控者的怀疑则减少，尤其是当指控者受到信任时，即使该指控者属于狼方阵营也是如此。更大的模型能够更好地抵御来自其已不信任的指控者的指控。总体而言，我们的研究结果表明，当前的开源权重大语言模型……（摘要在此处被截断）

    arXiv:2609.12446v1 Announce Type: cross  Abstract: Social-deduction games such as Werewolf are increasingly used to evaluate LLM agents, but existing evaluations often rely on final game outcomes. We propose a belief-shift evaluation benchmark in Werewolf for analyzing communication skills through belief updating. Using LLM-played games, we annotate suspicion and accusation messages and measure how an observing village-side model's beliefs change after each message. We evaluate 40 open-weight LLM configurations on 1,224 annotated messages. Our results show that larger models better distinguish true wolves from villagers based on game history, but accusations still strongly influence their beliefs. Models become more suspicious of the accused target and less suspicious of the accuser, especially when the accuser is trusted, even if the accuser is wolf-aligned. Larger models better resist accusations from accusers they already distrust. Overall, our findings suggest that current open-wei
    
[^36]: 多元的心智，分裂的网络？基于大语言模型社会模拟中的人格构成、极化与集体智能

    Diverse Minds, Divided Networks? Personality Composition, Polarization, and Collective Intelligence in LLM-Based Social Simulations

    [https://arxiv.org/abs/2609.12444](https://arxiv.org/abs/2609.12444)

    该研究提出TraitMix实验设计，通过对991次大语言模型社会模拟的分析发现，人格特质异质性对极化的两个维度具有相反的影响——多样化的社会观点更加分散但阵营对立更少，揭示了人格构成同时塑造极化与集体智能的关键作用。

    

    基于大语言模型智能体的模拟社会被用于研究在线极化，也被单独用于研究集体智能，但两者很少在同一系统中被同时测量。因此，我们难以判断一个社会的人格构成是否同时塑造这两者，也难以判断降低极化是否要以牺牲集体能力为代价。我们提出了TraitMix这一实验设计，其中模拟社交网络的大五人格构成——包括特质水平和特质异质性——是受控的实验变量，并且极化与集体表现是在同一轮运行中被同时测量的。在涵盖六个争议话题和六种语言模型的991次百智能体社会模拟中，特质异质性展现出最大的测量效应，并对极化的两个方面产生相反方向的作用：多样化的社会持有更加分散的观点，同时却更少被分割成对立阵营，因此同质化的社会……（原文摘要在此处截断）

    arXiv:2609.12444v1 Announce Type: cross  Abstract: Simulated societies of large language model agents are used to study online polarization, and separately to study collective intelligence, but the two are rarely measured in the same system. It is therefore difficult to say whether a society's personality composition shapes both, or whether reducing polarization costs collective competence. We present TraitMix, an experimental design in which the Big Five composition of a simulated social network, both trait levels and trait heterogeneity, is a controlled experimental variable, and in which polarization and collective performance are measured in the same runs. Across 991 simulations of hundred-agent societies, spanning six contested topics and six language models, trait heterogeneity has the largest measured effects, acting in opposite directions on two faces of polarization: varied societies hold more dispersed opinions while being less segregated into camps, so homogeneous societies 
    
[^37]: 超越ID嵌入：面向认知诊断的过程接地语言建模

    Beyond ID Embeddings: Process-Grounded Language Modeling for Cognitive Diagnosis

    [https://arxiv.org/abs/2609.12403](https://arxiv.org/abs/2609.12403)

    该论文提出PLCD框架，利用大语言模型构建概念模式和认知过程图作为认知先验，并通过DA-MoE专家和过程级对比学习将语言表示映射到认知状态，从而超越传统ID嵌入方法，解决了新练习或新概念出现时的语义局限问题。

    

    认知诊断模型（CDMs）在个性化在线学习中发挥着关键作用。传统的CDMs依赖于离散的、基于ID的嵌入来表示学生、练习和概念。这一范式偏离了学习者认知的本质——知识并非以孤立符号的形式被存储和提取。因此，当出现新的练习或概念时，CDMs会受到语义层面的限制。本文提出了一种过程感知的语言认知诊断框架，该框架使用语言派生的结构作为认知先验，并利用作答记录来校准学生的后验状态。PLCD利用大语言模型（LLMs）构建概念模式与认知过程图，并使用目标条件化的语义记忆来检索与每个目标练习相关的历史作答记录。随后，一个基于过程的“语言到认知”映射器（配备DA-MoE专家和过程级对比学习）将语言表示映射至认知状态。

    arXiv:2609.12403v1 Announce Type: cross  Abstract: Cognitive Diagnosis Models (CDMs) play a pivotal role in personalized online learning. Traditional CDMs rely on discrete, ID-based embeddings to represent students, exercises, and concepts. This paradigm diverges from the nature of learner cognition, where knowledge is not stored and retrieved as isolated symbols. As a result, CDMs suffer from semantic limitations when new exercises or concepts appear. In this paper, we propose a Process-aware Language Cognitive Diagnosis (PLCD) framework that uses language-derived structures as cognitive priors and response records to calibrate student posterior states. PLCD leverages large language models (LLMs) to construct concept schemas and cognitive process graphs, and uses target-conditioned semantic memory to retrieve historical responses that are relevant to each target exercise. A process-grounded Language-to-Cognition Mapper with DA-MoE experts and process-level contrastive learning then ma
    
[^38]: 基于表示的掩码扩散模型

    Representation-based Masked Diffusion Model

    [https://arxiv.org/abs/2609.12382](https://arxiv.org/abs/2609.12382)

    提出基于表示的掩码扩散模型（RMDM），通过预训练编码器将文本表示归一化为高斯先验以显式编码全局语义，从而协调被掩码词元的并行更新，生成更连贯的文本。

    

    掩码扩散模型（MDMs）已成为语言建模中一种极具吸引力的范式，提供了高效并行文本生成的能力。然而，现有的并行采样方法通常独立地更新多个被掩码的词元，忽略了被掩码词元之间复杂的相互依赖关系。这种独立更新机制缺乏全局协调，可能导致输出不连贯。为了解决这一局限性，我们提出了基于表示的掩码扩散模型（RMDM），这是一个利用文本表示显式编码全局语义、从而帮助更精确地并行更新词元的框架。具体而言，我们首先使用预训练编码器将文本编码到连续语义空间中，并学习一个可逆变换，将表示分布归一化为高斯先验，从而促进生成过程中的高效采样。以该潜在语义表示为条件……（原文摘要至此截断）

    arXiv:2609.12382v1 Announce Type: new  Abstract: Masked Diffusion Models (MDMs) have emerged as a compelling paradigm for language modeling, offering the capability for efficient parallel text generation. However, existing parallel sampling methods typically update multiple masked tokens independently and ignore the complex mutual dependencies among the masked tokens. This independent updating mechanism lacks global coordination and might lead to incoherent outputs. To address this limitation, we propose Representation-based Masked Diffusion Model (RMDM), a framework that leverages the text representation to explicitly encode global semantics and help to parallel update tokens more precisely. Specifically, we first encode text into a continuous semantic space using a pretrained encoder and learn an invertible transformation that normalizes the representation distribution to a Gaussian prior, facilitating efficient sampling during generation. Conditioned on this latent semantic represen
    
[^39]: ORQA：一种面向大语言模型专业知识的职业现实化问答框架

    ORQA: An Occupation-Realistic Question and Answer Framework for LLM Professional Knowledge

    [https://arxiv.org/abs/2609.12366](https://arxiv.org/abs/2609.12366)

    ORQA通过将O*NET职业与可信的职业权威网站（如监管机构、执照颁发机构和政府出版物）相连接，自动生成可溯源的职业知识问答对，构建了覆盖SOC全部21个大类、116个职业的480个高质量问题，用以评估大语言模型的职业专业知识。

    

    我们提出了ORQA，这是一种测试大语言模型职业层面知识的方法。此前的方法要么通过任务定义将抽象的大语言模型技能映射到职业上，要么利用难以大规模获取且成本高昂的专家知识。ORQA与这两种方法形成互补，它将O*NET职业数据库与可信的职业相关网站（如监管机构、执照颁发机构、专业组织和政府出版物）相连接，并将这些内容转化为可溯源的问答对。通过自动化流程与人工审核相结合的方式，生成了一套关于职业的高质量问题。我们方法所创建的问题集涵盖了标准职业分类（SOC）中全部21个大类下的116个职业，包含来自187个不同网站的480个问题。每个问题旨在探究与相应职业相关的现实世界技能问题。我们测试了15个最先进的（原文在此处截断）……

    arXiv:2609.12366v1 Announce Type: new  Abstract: We present ORQA, a method for testing occupation-level knowledge in large language models. Prior methods either map abstract LLM skills to occupations via task definitions or utilize expert knowledge which is difficult to obtain at scale and expensive. ORQA complements both of these methods by connecting O*NET occupations to trusted occupation-specific websites (such as regulatory agencies, licensing bodies, professional organizations, and government publications) and converting these into source-traceable question-answer pairs. A combination of an automated pipeline and human review produces a set of high quality questions about occupations. The question set created via our method covers 116 occupations from all 21 major groups in the SOC, with 480 questions sourced from 187 different websites. Each question is designed to probe a real-world skill question that is relevant to the occupation in question. We test 15 state-of-the-art front
    
[^40]: CueMem：线索引导的长期对话记忆上下文重建

    CueMem: Cue-Guided Context Reconstruction for Long-Term Conversational Memory

    [https://arxiv.org/abs/2609.12354](https://arxiv.org/abs/2609.12354)

    CueMem提出将记忆记录作为检索线索而非自包含证据，通过线索引导定位来源对话轮次并在轮次图上扩展，从原始对话中重建与查询相关的紧凑证据上下文，从而兼顾长期对话记忆的效率与细粒度证据完整性。

    

    长期对话智能体必须通过回忆扩展对话历史中的信息来回答用户查询，然而直接使用完整历史既代价高昂又往往不可靠，而压缩的记忆单元则可能丢失问答所需的细粒度证据。受自传体记忆重建性观点的启发，我们提出了CueMem，一个线索引导的框架，它将提取的记忆记录视为检索线索而非自包含的证据，并从其来源对话轮次中重建与查询相关的对话上下文。在记忆构建阶段，CueMem从对话轮次中提取细粒度记忆线索，并将每条线索链接到其来源轮次。在查询时，它检索与查询相关的线索，将其映射到来源轮次锚点，并在捕捉时间邻近性与语义相关性的轮次图上从这些锚点进行扩展，从原始对话中重建出一个紧凑的证据上下文，供大语言模型生成答案。

    arXiv:2609.12354v1 Announce Type: new  Abstract: Long-term conversational agents must answer user queries by recalling information from extended dialogue histories, yet directly using the full history is costly and often unreliable, while compressed memory units may lose fine-grained evidence needed for question answering. Motivated by the reconstructive view of autobiographical memory, we propose CueMem, a cue-guided framework that treats extracted memory records as retrieval cues rather than self-contained evidence and reconstructs query-relevant dialogue context from their source turns. During memory construction, CueMem extracts fine-grained memory cues from dialogue turns and links each cue to its source turn. At query time, it retrieves query-relevant cues, maps them to source-turn anchors, and expands from these anchors over a turn graph that captures temporal proximity and semantic relatedness, reconstructing a compact evidence context from the original dialogue for LLM answer 
    
[^41]: SynthSentry：检测语言模型训练数据中的合成数据污染

    SynthSentry: Detecting Synthetic Data Contamination in Language Model Training Data

    [https://arxiv.org/abs/2609.12353](https://arxiv.org/abs/2609.12353)

    提出SynthSentry，一种模型无关的语料库级污染检测信号，通过词汇多样性崩溃、n-gram尾部截断和困惑度方差三种统计量的分布散度，在训练前识别出可能导致模型崩溃的合成数据，且无需访问生成模型或任何生成历史。

    

    在自身或其他模型的输出上进行递归训练的大型语言模型会发生模型崩溃，即分布尾部和事实准确性恶化，而流畅性得以保留。先前的工作是在训练之后诊断崩溃；而可操作的问题是在训练之前筛选来源不明的语料库。我们提出了SynthSentry，这是一种语料库级别、模型无关的污染信号，无需访问生成模型、无需生成历史，也无需合成数据标签。该得分基于三种统计量的分布散度：词汇多样性崩溃、n-gram尾部截断，以及跨参考模型的困惑度方差。我们在被小型开源权重生成器和一个指令微调的开源权重模型污染的语料库上，采用留一生成器协议进行评估。一项领域分层研究测量了该方法在自然重复性人类文本（法律、临床、源代码）上的假阳性率。该得分对语料库进行排序

    arXiv:2609.12353v1 Announce Type: new  Abstract: Large language models trained recursively on their own or other models' outputs undergo model collapse, in which distributional tails and factual accuracy deteriorate while fluency survives. Prior work diagnoses collapse after training; the actionable problem is screening a corpus of unknown provenance before training. We introduce SynthSentry, a corpus-level, model-agnostic contamination signal requiring no access to the generating model, no generation history, and no synthetic labels. The score is a distributional divergence over three statistics: lexical diversity collapse, n-gram tail truncation, and perplexity variance across reference models. We evaluate on corpora contaminated by small open-weight generators and an instruction-tuned open-weight model under a leave-one-generator-out protocol. A domain-stratified study measures false positives on naturally repetitive human text (legal, clinical, source code). The score ranks corpora
    
[^42]: 我是无名之辈：面向文本匿名化的风格感知改写方法

    I Am No One: Style-Aware Paraphrasing for Text Anonymization

    [https://arxiv.org/abs/2609.12341](https://arxiv.org/abs/2609.12341)

    提出一种基于大语言模型的风格感知改写匿名化方法，通过构建风格画像并重写文本抑制可识别的风格指纹，在保持文本质量的同时将作者归属识别F1分数降低60-70%。

    

    作者归属模型能够利用稳定的风格指纹从看似已匿名化的文本中重新识别用户，即使在删除了显式标识符之后也是如此，这给文本发布和分析带来了日益严峻的隐私风险。这一风险还延伸到语音转写文本，例如会议和呼叫中心对话的ASR（自动语音识别）转写文本，即便经过声学匿名化处理，文体计量信息的泄露仍可能持续存在。基于差分隐私的匿名化方法通常会严重损害文本质量和实用性。我们提出了一种风格感知的、基于提示驱动的匿名化方法，利用预训练大语言模型从极少量样本构建紧凑的风格画像，并对文本进行改写，在保留语义的同时抑制可识别的风格标记。在博客和评论数据集上，我们的方法将作者归属识别的F1分数降低了60-70%，同时保持了内容质量和可读性，显著优于基于差分隐私和非差分隐私的基线方法。

    arXiv:2609.12341v1 Announce Type: new  Abstract: Authorship attribution models can re-identify users from seemingly anonymized text by exploiting stable stylistic fingerprints, even after explicit identifiers are removed, posing a growing privacy risk for text publishing and analytics. This risk extends to speech-derived text such as ASR transcripts of meetings and call-center conversations, where stylometric leakage can persist even after acoustic anonymization. Differential privacy-based anonymization often severely degrades text quality and utility. We propose a style-aware, prompt-driven anonymization approach that uses pretrained large language models to construct compact stylistic profiles from minimal samples and rewrite text to suppress identifiable style markers while preserving meaning. Across blog and review datasets, our approach reduces authorship attribution F1 by 60-70% while maintaining content quality and readability, substantially outperforming DP-based and non-DP bas
    
[^43]: ESTS参加WMT26：基于路由信息的专家剪枝模型压缩方法

    ESTS at WMT26: Routing-Informed Expert Pruning for Model Compression

    [https://arxiv.org/abs/2609.12310](https://arxiv.org/abs/2609.12310)

    该论文提出利用任务特定路由质量和跨语言路由差异来识别并物理移除GPT-OSS-20B中的低重要性专家，结合恢复微调与MXFP4量化技术，实现了面向中英和阿英翻译任务的高效模型压缩。

    

    我们以ESTS团队名义描述了提交至无约束WMT26模型压缩共享任务的六个参赛作品，该任务涵盖英语-简体中文和英语-埃及阿拉伯语两个翻译方向。我们在每个翻译方向提交了三个压缩工作点，均基于GPT-OSS-20B模型。我们使用任务特定的路由质量对专家进行排序，并利用跨语言路由差异在各层之间分配保留容量，然后物理移除低重要性的专家。所得的专用模型在GPT-5.1生成的合成翻译数据上进行恢复微调，并通过对保留的专家投影权重应用MXFP4量化进一步压缩。我们还为指令条件化的WMT26设置实现了一个鲁棒的推理系统，包括类别推理、输出验证、重试机制、分段回退和源端拥有的JSON重建。在我们提交的六个作品中，参数量范围为41.86亿至77.70亿，打包工件...

    arXiv:2609.12310v1 Announce Type: new  Abstract: We describe six submissions under the team name ESTS to the unconstrained WMT26 Model Compression Shared Task for English--Simplified Chinese and English--Egyptian Arabic. We submit three compression operating points per translation direction, all derived from GPT-OSS-20B. We use task-specific routing mass to rank experts and cross-lingual routing divergence to allocate retained capacity across layers, then physically remove low-importance experts. The resulting specialists are recovery-tuned on GPT-5.1-generated synthetic translation data and further compressed by applying MXFP4 quantization to the retained expert projection weights. We additionally implement a robust inference system for the instruction-conditioned WMT26 setting, including category inference, output validation, retries, segmented fallback, and source-owned JSON reconstruction. Across our six submissions, parameter counts range from 4.186B to 7.770B and packed artifact 
    
[^44]: 突破词元上限：蒸馏出更小更强的字节模型

    Breaking the Token Ceiling: Distilling Smaller, Stronger Byte Models

    [https://arxiv.org/abs/2609.12303](https://arxiv.org/abs/2609.12303)

    本文首次大规模研究了词元化方案与训练目标（蒸馏 vs. 交叉熵）对约10亿参数模型的影响，提出了两种将词元logits转换为字节logits的方法（Marginalize-It和End-Of-Token），并发现在八个基准测试中词元模型仍优于字节模型。

    

    小模型通常通过与共享相同词元化方案的大模型进行蒸馏而变得更强。然而，随着计算量和数据量的增加，蒸馏得到的字节模型和词元模型在缩放趋势上的表现是否相似？为了进行这一比较，我们引入了两种将词元对数概率（Token Logits）高效转换为字节对数概率（Byte Logits）的方法：1）近似方法：Marginalize-It（边缘化）；2）精确方法：End-Of-Token（词元结束符）。随后，我们首次开展了大规模研究，对仅解码器的稠密Transformer模型进行过度训练，同时改变两个维度：词元化方案（Tokens、Bytes、带eot的Bytes）和训练目标（蒸馏 vs. 交叉熵），系统扫描了参数量约10亿、训练数据量高达1万亿字节的层级参数匹配模型。在涵盖三个类别的八个基准测试上：多选题问答、语言生成和机器翻译，我们发现Token-1B模型优于字节模型（End-Of-Token-1B和Bytes-……

    arXiv:2609.12303v1 Announce Type: new  Abstract: Small models are made more capable through distillation from a larger one that shares their tokenization scheme. However, do distilled byte and token models behave similarly in terms of scaling trends as compute and data increases? To enable this comparison, we introduce two variants to efficiently convert token logits to Byte Logits: 1) approximate: Marginalize-It, and 2) exact: End-Of-Token. We then present the first large scale study of overtraining decoder-only dense transformer models varying two dimensions simultaneously: the tokenization scheme (Tokens, Bytes, Bytes w/ eot) and the training objective (Distillation vs. Cross-Entropy), sweeping layer-parameter-matched models with roughly 1 billion parameters up to 1 trillion bytes of data. Across eight benchmarks spanning three categories: Multiple Choice QA, Language Generation, and Machine Translation, we find that Token-1B models outperform byte models (End-Of-Token-1B and Bytes-
    
[^45]: EAR：面向检索增强生成开发的实体感知分割方法

    EAR: Entity-Aware Partitioning Approach for Retrieval-Augmented Generation Development

    [https://arxiv.org/abs/2609.12268](https://arxiv.org/abs/2609.12268)

    EAR提出了一种实体感知的语料库分割方法，通过从问题和选项中提取锚点、检索语料库中的局部窗口，在改进多项选择题问答的同时减少检索内容长度。

    

    检索增强生成（RAG）可以改进知识密集型问答，但第一个设计选择往往容易被忽视：源语料库应如何被分割为可检索单元？固定大小的分块常常返回与问题关系仅为隐式的长段落。我们提出了EAR，一种面向多项选择题问答（MCQA）的实体感知分割方法。EAR从问题、答案选项和语料库中提取规范化的表面锚点；检索语料库中匹配锚点周围的局部窗口；并可通过抽取式摘要附加更大的父级段落。我们在一个经过清洗的MMLU风格子集上评估EAR，该子集由自动语料库支持启发式方法筛选出153个问题，并使用经过去污染的公共教科书文本。在与Mistral、Gemma和DeepSeek进行的相同协议top-k=3和top-k=8扫描实验中，EAR实体窗口方法减少了检索词的数量……

    arXiv:2609.12268v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) can improve knowledge-intensive question answering, but the first design choice is easy to overlook: how should the source corpus be partitioned into retrievable units? Fixed-size chunks often return long passages whose relation to the question is only implicit. We introduce EAR, an Entity-Aware Partitioning approach for multiple-choice question answering (MCQA). EAR extracts normalized surface anchors from the question, answer options, and corpus; retrieves local windows around matching corpus anchors; and can attach a larger parent passage through an extractive summary. We evaluate EAR on a cleaned Massive Multitask Language Understanding (MMLU)-style subset of 153 questions selected by an automatic corpus-support heuristic and using decontaminated public textbook text. Across same-protocol top-k = 3 and top-k = 8 sweeps with Mistral, Gemma, and DeepSeek, EAR entity-window reduces retrieved words by
    
[^46]: HypoKG：超越终点知识的证据约束型生物医学假设生成

    HypoKG: Evidence-Disciplined Biomedical Hypothesis Generation Beyond Endpoint Knowledge

    [https://arxiv.org/abs/2609.12260](https://arxiv.org/abs/2609.12260)

    该研究构建了整合KEGG、Rhea和UniProt的统一生物化学知识图谱基准HypoKG，在550条酶来源到罕见病终点的路径上评估六个大语言模型生成的13,200个假设，发现仅凭来源酶与疾病终点信息即可产生最高评分的假设，揭示了LLMs生物医学假设生成能力及其与真正证据推理之间的差距。

    

    大语言模型（LLMs）能够生成生物医学假设，但目前尚不清楚它们究竟是真正基于科学证据进行推理，还是仅仅产出听起来令人信服的想法。为了研究这一问题，我们将三个主要的生物学数据库——京都基因与基因组百科全书（KEGG）、Rhea 和 UniProt——整合为统一的生物化学知识图谱，并构建了一个包含 550 条连接酶来源与罕见疾病终点路径的基准，在四种条件下（改变每个模型接收到的生物学信息：仅来源酶、完整生物路径、或仅来源与疾病终点）从六个大语言模型中生成了 13,200 个假设。这些假设采用基于专家制定的五项准则评分标准进行评分，每项准则按 1-5 分打分。我们发现，同时获得来源酶和疾病终点信息的模型往往生成评分最高的假设，这表明大语言模型能够从最少的信息中产生有说服力的想法。然而，这些假设……（摘要原文在此处被截断）

    arXiv:2609.12260v1 Announce Type: new  Abstract: Large language models (LLMs) can generate biomedical hypotheses, but it remains unclear whether they truly reason from scientific evidence or simply produce convincing-sounding ideas. To study this, we combine three major biological databases: the Kyoto Encyclopedia of Genes and Genomes (KEGG), Rhea, and UniProt, into a unified biochemical knowledge graph and construct a benchmark of 550 paths connecting enzyme sources to rare disease endpoints, yielding 13,200 hypotheses from six LLMs under four conditions varying the biological information each model receives: source enzyme only, full biological path, or source and disease endpoint only. Hypotheses are scored using an expert-derived five-criterion rubric on a 1-5 scale per criterion. We find that models given both the source and disease endpoint often produce the highest-scoring hypotheses, showing that LLMs can generate compelling ideas from minimal information. However, these hypothe
    
[^47]: 气候相关文档中社会临界点证据的自动化检测与结构化：一个模块化AI框架

    Automated Detection and Structuring of Social Tipping Point Evidence in Climate related Documents: A Modular AI Framework

    [https://arxiv.org/abs/2609.12254](https://arxiv.org/abs/2609.12254)

    本文提出一个开放的模块化AI框架，通过集成DistilBERT分割器、RoBERTa分类器和Mistral 7B等多个组件，在段落级别自动检测并结构化气候文档中的环境社会临界点证据，填补了现有文本挖掘工具无法系统性发现和整理这类关键证据的空白。

    

    气候文献的增长速度已经超过了评审团队的阅读能力。这一差距对于环境社会临界点这一概念而言最为关键——即一个微小的变化在某一阈值处触发社会系统快速、自我强化的变化。这类转变的证据通常只包含在较长文档中的一两段文字里。因此，现有的文本挖掘工具——它们要么按主题对整个文档进行分类，要么突出孤立的论断——使得越来越多的关键证据缺乏系统性的发现和组织方法。本文提出了一个开放、模块化的基于Transformer的框架，用于在段落层面检测和结构化社会临界点证据。该框架将五个组件整合为一个可部署的工作流：用于文本分割的DistilBERT边界切分器、用于证据检测的迭代增强RoBERTa分类器、用于重写检测段落的Mistral 7B模型等，共同构建了完整的证据发现与组织流程。

    arXiv:2609.12254v1 Announce Type: new  Abstract: The climate literature has grown faster than review teams can read it. That gap matters most for a concept like the environmental social tipping point, the threshold at which a small change triggers rapid, self-reinforcing change in a social system. Evidence of this kind of shift is usually contained in one or two paragraphs within a longer document. As a result, existing text mining tools-which categorize entire documents by topic or highlight isolated claims-leave an expanding set of important evidence without any systematic method for discovery or organization. This paper presents an open and modular transformer-based framework that detects and structures social tipping point evidence at the passage level. The framework joins five components into a single deployable workflow: a DistilBERT boundary splitter for segmentation, an iteratively augmented RoBERTa classifier for detection, a Mistral 7B model that rewrites each detected passag
    
[^48]: 切薄-共识幂采样：一种保持多样性的大语言模型解码方法

    Chopthin-Consensus Power Sampling: A Diversity-Preserving Approach to LLM Decoding

    [https://arxiv.org/abs/2609.12243](https://arxiv.org/abs/2609.12243)

    本文提出切薄-共识幂采样（CCPS），通过Chopthin重采样器限制最大与最小权重比值并保留不等权重，避免了等权重重采样对低权重推理路径的过度剪除，在保持SMC近似无偏的同时维持推理路径多样性并保证有效样本量下界。

    

    通过序贯蒙特卡洛（SMC）进行的推理时幂采样可以在无需后训练的情况下显著提升大语言模型（LLM）的推理能力。然而，许多现有的SMC方法依赖于等权重重采样，这可能会激进地剪除低权重轨迹，丢弃潜在正确的推理路径，并降低搜索空间的谱系多样性。为了解决这一问题，我们提出了切薄-共识幂采样（CCPS）。我们的方法将Chopthin重采样器应用于LLM解码：它不是均衡权重并强制进行不必要的粒子复制，而是对最大权重与最小权重之间的比值施加一个上界，并将不相等的权重向前传递。这种有针对性的干预保留了更加丰富的不同推理路径集合，使加权SMC近似在条件期望下保持不变，并保证了重采样后有效样本量的下界。

    arXiv:2609.12243v1 Announce Type: new  Abstract: Inference-time power sampling via Sequential Monte Carlo (SMC) can substantially improve large language model (LLM) reasoning without requiring post-training. However, many existing SMC approaches rely on equal-weight resampling, which can aggressively prune low-weight trajectories, discarding potentially correct reasoning paths and degrading the genealogical diversity of the search space. To address this, we introduce Chopthin-Consensus Power Sampling (CCPS). Our method applies the Chopthin resampler to LLM decoding: rather than equalizing weights and forcing unnecessary particle duplication, it enforces an upper bound on the ratio between the largest and smallest weights and carries the unequal weights forward. This targeted intervention preserves a richer set of distinct reasoning paths, keeps the weighted SMC approximation unchanged in conditional expectation, and guarantees a lower bound on the post-resampling effective sample size 
    
[^49]: 先修复再强化：面向多跳问答的上下文增强知识图谱推理

    Repair Before Reinforce: Context-Augmented Knowledge Graph Reasoning for Multi-Hop Question Answering

    [https://arxiv.org/abs/2609.12230](https://arxiv.org/abs/2609.12230)

    该论文提出一种上下文增强的知识图谱训练框架，通过为主要KG三元组附加同源文本中的支持三元组构建上下文图进行监督训练，从而提升大型语言模型在多跳问答任务中的推理能力，并在胃轻瘫和糖尿病的疾病知识图谱上验证了其有效性。

    

    问答任务通常需要在多个相互关联的事实之间进行推理，而不是检索单个孤立的关系。知识图谱（KG）提供了一种结构化的方式来表示这些事实，但仅在孤立的KG头-关系-尾三元组上训练大型语言模型（LLM）可能会限制其学习多跳推理所需上下文的能力。在这项工作中，我们提出了一种面向多跳问答的上下文增强训练框架。虽然该框架具有普遍适用性，但我们在疾病特定知识图谱的场景下对其进行验证，即使用名为GraphMERT的可靠KG提取框架从胃轻瘫和糖尿病领域提取的知识图谱。对于每个主要KG三元组，我们附加从同一源文本块中提取的支持三元组，以形成上下文图（CG）。这创造了两种监督设置：KG基础监督（仅使用目标KG三元组或路径）和CG基础监督（使用上下文图提供的增强信息进行训练）。

    arXiv:2609.12230v1 Announce Type: new  Abstract: Question-answering often requires reasoning across multiple connected facts rather than retrieving a single isolated relation. Knowledge graphs (KGs) provide a structured way to represent such facts, but training large language models (LLMs) only on isolated KG head-relation-tail triples may limit their ability to learn the surrounding context needed for multi-hop reasoning. In this work, we propose a context-augmented training framework for multi-hop question-answering. Although generally applicable, we validate the framework in the context of disease-specific KGs, extracted using a reliable KG extraction framework called GraphMERT, for Gastroparesis and Diabetes. For each primary KG triple, we attach supporting triples extracted from the same source text chunk to form a context graph (CG). This creates two supervision settings: KG-grounded supervision, which uses only the target KG triple or path, and CG-grounded supervision, which use
    
[^50]: GAUGE：在面向任务智能体的用户模拟评估中，何时不应信任LLM-as-a-Judge

    GAUGE: When Not to Trust LLM-as-a-Judge in User-Simulated Evaluation of Task-Oriented Agents

    [https://arxiv.org/abs/2609.12191](https://arxiv.org/abs/2609.12191)

    GAUGE协议揭示了一种常见的LLM智能体离线评估闸门存在严重缺陷——LLM评审给出的满意度评分与实际任务成功率几乎完全不相关（被评为满意的对话中有57.5%实际未能完成客户任务），因此不应仅凭主观评分来信任和选择任务导向型LLM智能体。

    

    比较和选择面向任务的LLM智能体越来越依赖于一种低成本的离线评估闸门：基于人设驱动的LLM用户模拟器与每个候选智能体对话，由LLM-as-a-judge对对话记录进行评分，得分较高的智能体被采用。我们提出了GAUGE，一个可复用的离线协议，用于衡量该闸门的排名是否与基于真实可验证奖励的排名相匹配，涵盖来自六家提供商的25个智能体，在τ²-bench和SimulatorArena基准上进行测试，并区分了发布实践中被混淆的两种评估有效性：排名有效性和构念有效性。首先，满意度与成功率之间存在鸿沟：满意度基本不携带关于任务成功的任何信息，因为我们盲评小组评为满意的对话与实际成功之间不相关，其中57.5%的对话未能完成客户任务，这一模式在五类评分者群体、两个基准测试以及我们评估的每个主观维度上都保持一致。其次，w

    arXiv:2609.12191v1 Announce Type: new  Abstract: Comparing and selecting task-oriented LLM agents increasingly relies on a low-cost offline evaluation gate: persona-driven LLM user-simulators converse with each candidate, an LLM-as-a-judge scores the transcripts, and the higher-scoring agent is promoted. We introduce GAUGE, a reusable offline protocol that measures whether this gate's ranking matches a grounded verifiable reward across 25 agents from six providers on the $\tau^2$-bench and SimulatorArena benchmarks, separating two kinds of evaluation validity that release practices conflate: ranking validity and construct validity. First, a satisfaction-success gap: satisfaction carries essentially no information about task success, as conversations rated satisfied by our blind panel are decorrelated from actual success, with 57.5% of them failing the customer's task, a pattern consistent across five rater populations, both benchmarks, and every subjective dimension we rated. Second, w
    
[^51]: 起草-验证-修订流水线中的大语言模型能否解决指示语歧义？

    Can LLMs in Draft-Verify-Revise Pipelines Resolve Deictic Ambiguity?

    [https://arxiv.org/abs/2609.12162](https://arxiv.org/abs/2609.12162)

    本文通过合成数据集研究发现，在起草-验证-修订的多模型流水线中，不同阶段的大语言模型可能对同一上下文依赖表达式产生不一致的解读，从而导致指示语偏移。

    

    起草-验证-修订是大语言模型扩展推理时计算的常见编排模式：第一个大语言模型负责起草，第二个对草稿进行批评并给出反馈，第三个则利用该反馈将草稿修订为最终输出。随着上下文在各个阶段之间级联传递，不同阶段的大语言模型可能对诸如“previous（上一个）”这类依赖上下文的表达式产生不同的解读。当这种情况发生时，该表达式会经历指示语偏移，即其所指内容发生变化。本研究使用一个包含10个基础示例的合成数据集来研究这一现象，每个示例在三种条件下呈现。在保持共享组件不变的前提下，这些条件改变了起草阶段的大语言模型（助手）或验证阶段的大语言模型（评分者）是否正确解析了该表达式，以及修订阶段的大语言模型（元评估者）需要多少独立推理才能确定哪种解读是正确的。来自三家提供商的六个模型在21种推理强度配置下进行了测试。

    arXiv:2609.12162v1 Announce Type: cross  Abstract: Draft-verify-revise is a common LLM orchestration pattern for scaling inference-time compute. One LLM drafts, a second critiques the draft and provides feedback, and a third uses that feedback to revise the draft into the final output. As context cascades between stages, LLMs at different stages can resolve a context-dependent expression such as "previous" differently. When that happens, the expression undergoes a deictic shift, a change in what it refers to. This phenomenon was studied with a synthetic dataset of 10 base examples, each rendered in three conditions. Holding the shared components constant, the conditions varied whether the draft stage LLM (the assistant) or the verify stage LLM (the grader) resolved the expression correctly, and how much independent reasoning the revise stage LLM (the meta-evaluator) needed to determine which reading was correct. Six models from three providers were tested across 21 reasoning effort con
    
[^52]: 人口层面的感知性食物获取测量揭示了地理邻近性之外的障碍

    Population-level measures of perceived food access reveal barriers beyond geographic proximity

    [https://arxiv.org/abs/2609.12132](https://arxiv.org/abs/2609.12132)

    本研究创新性地利用谷歌地图评论与零样本分类方法，在人口层面测量了食物获取的五个感知维度，揭示了传统地理邻近性指标无法捕捉的食物获取障碍。

    

    食物获取是多维度的，但由于感知维度难以大规模测量，人口层面的测量仍然严重依赖地理因素。本研究利用来自北卡罗来纳州罗利市49家杂货店的25,125条谷歌地图评论，测量了食物获取的五个维度：可得性、可达性、可负担性、适应性和可接受性。我们使用无监督主题建模识别评论主题，并采用零样本分类将其分配到各个获取维度，与人工编码的一致率达到85.4%。由此得到的商店层面测量捕捉了食物获取的不同方面，并揭示了仅靠地理邻近性无法捕捉的障碍。同一连锁店附近商店之间的比较进一步表明，相同的商店政策在不同地点可能被感知得截然不同，这与食物获取反映居民与其食物环境之间适配性的观点相一致。

    arXiv:2609.12132v1 Announce Type: new  Abstract: Food access is multidimensional, but population-level measurement still relies heavily on geography because perceived dimensions of access are difficult to measure at scale. Here, we use 25,125 Google Maps reviews from 49 grocery stores in Raleigh, North Carolina, to measure five dimensions of food access: availability, accessibility, affordability, accommodation, and acceptability. We identify review topics with unsupervised topic modeling and assign them to access dimensions using zero-shot classification, with 85.4% agreement against manual coding. The resulting store-level measures capture distinct aspects of food access and reveal barriers that geographic proximity alone does not capture. Comparisons between nearby stores in the same chain further show that identical store policies can be perceived very differently across locations, consistent with food access reflecting the fit between residents and their food environment. Perceive
    
[^53]: 局部编辑，全局涟漪：面向工作流合成的重放感知策略自适应

    Local Edits, Global Ripples: Replay-Informed Policy Adaptation for Workflow Synthesis

    [https://arxiv.org/abs/2609.12127](https://arxiv.org/abs/2609.12127)

    提出RIPPLE框架，针对提示策略编辑中“局部编辑引发全局涟漪效应”与“编辑组合后相互干扰”两大难题，将编辑位置定位与编辑组合后安全性判别相分离，实现工作流合成中安全可靠的持久策略自适应。

    

    提示策略编辑为改进合成可执行工作流的智能体提供了一种实用的方法，且无需更新底层模型。然而，持久性提示编辑具有两个耦合特性。第一，编辑的局部性并不意味着效果的局部性：局限于某一策略片段的编辑可能会波及下游执行，改变超出被编辑片段范围的行为。第二，编辑效果对组合敏感：单独生效的编辑在组合之后可能相互干扰，导致其中一个或两者的收益丧失，甚至变得有害。因此，持久自适应必须支持两个不同的决策：根据执行反馈确定策略应当在哪里改变，以及判断由此产生的编辑在组合之后是否仍然可以安全地持久保留。为了应对这些挑战，我们提出了 RIPPLE（重放感知的持久策略定位与编辑），它将编辑发生的位置与编辑在组合后是否仍然保持安全这两个问题分离开来……

    arXiv:2609.12127v1 Announce Type: new  Abstract: Prompt-policy editing offers a practical way to improve agents that synthesize executable workflows without updating the underlying model. However, persistent prompt editing has two coupled properties. First, edit locality does not imply effect locality: an edit confined to one policy segment can ripple through downstream execution, altering behavior beyond the edited segment. Second, edit effects are composition-sensitive: edits that work in isolation can interfere after composition, causing one or both to lose their benefit or become harmful. Persistent adaptation must therefore support two distinct decisions: identifying where the policy should change from execution feedback, and determining whether the resulting edit remains safe to persist after composition.   To address these challenges, we introduce RIPPLE (Replay-Informed Persistent Policy Localization and Editing), which separates where an edit is made from whether it remains sa
    
[^54]: 通过声学掩蔽量化辅音对单词可懂度的贡献

    Quantifying Consonant Contributions to Word Intelligibility via Acoustic Masking

    [https://arxiv.org/abs/2609.12122](https://arxiv.org/abs/2609.12122)

    本文提出一种基于声学掩蔽与语音识别模型的可扩展方法，通过掩蔽诱导误识别率（MMR）量化每个辅音对单词可懂度的贡献，从而帮助确定运动性言语障碍治疗的优先干预目标。

    

    辅音对单词能否被理解明白的贡献并不均等。考虑到治疗时间有限，按对可懂度的贡献为辅音排序有助于确定运动性言语障碍干预目标的优先次序。然而，测量这种贡献依赖于难以规模化的感知实验。本文提出了一种使用声学掩蔽来测量辅音贡献的可扩展方法：我们在一个孤立单词中逐次静音一个辅音，然后测试自动语音识别（ASR）模型是否仍能正确识别该单词。我们将辅音的贡献得分定义为其被掩蔽实例中单词被错误识别的比例，称之为掩蔽诱导误识别率。我们针对先前研究中报道的与辅音贡献相关的两个语言学因素（即音素频率和功能负荷）对MMR进行了验证，并将该分析应用于四种语（原文在此处截断）……

    arXiv:2609.12122v1 Announce Type: new  Abstract: Consonants contribute unequally to whether a word is understood. Given the limited time available for therapy, ranking consonants by contribution to intelligibility helps prioritize intervention targets in motor speech disorders. However, measuring this contribution relies on perceptual studies that are difficult to scale. This paper presents a scalable method that measures consonant contribution using acoustic masking. We silence one consonant at a time in an isolated word and test whether an automatic speech recognition (ASR) model still recognizes the word. We define a consonant's contribution score as the proportion of its masked instances for which the word becomes misrecognized, which we refer to as the mask-induced misrecognition rate (MMR). We validate MMR against two linguistic factors previously reported to correlate with consonant contribution, namely phoneme frequency and functional load. We apply this analysis across four la
    
[^55]: 压缩的代价：事实性幻觉的率失真极限

    The Cost of Compression: A Rate-Distortion Limit on Factual Hallucination

    [https://arxiv.org/abs/2609.12111](https://arxiv.org/abs/2609.12111)

    该论文证明了事实性幻觉不仅源于事实未被学习（覆盖缺失），还源于有限内存迫使已观察到的事实只能被近似压缩存储，并首次给出了幻觉率的一个率失真理论下界。

    

    闭卷问答中的事实性幻觉通常被视为一个覆盖性问题：模型之所以失败，是因为相关事实不存在于其内部记忆中。这一观点忽略了第二种错误来源。即使某个事实已经被观察到，有限的记忆也可能迫使其只能被近似地存储。我们通过一个简单的覆盖—压缩事实召回模型来研究这一效应。我们考虑一个具有 $N$ 个可能查询和 $K$ 个可能答案的非结构化问答任务。学习者观察 $M$ 条训练事实，将其压缩至最多 $B$ 比特，并在无检索的情况下回答均匀抽取的测试查询。对于均匀随机的真实映射，我们证明了 $\mathcal{E} \geq \frac{M}{N}\delta^\star\!\left(\frac{B}{M}\right) + \left(1-\frac{M}{N}\right)\left(1-\frac{1}{K}\right)$，其中 $\delta^\star(r)$ 是均匀 $K$ 元信源在零一损失下的逆率失真函数。这两项分别对应……

    arXiv:2609.12111v1 Announce Type: new  Abstract: Factual hallucination in closed-book question answering is often treated as a coverage problem: a model fails because the relevant fact is absent from its internal memory. This view misses a second source of error. Even when a fact has been observed, finite memory may force it to be stored only approximately. We study this effect through a simple coverage--compression model of factual recall. We consider an unstructured question-answering task with $N$ possible queries and $K$ possible answers. A learner observes $M$ training facts, compresses them into at most $B$ bits, and answers uniformly drawn test queries without retrieval. For a uniformly random ground-truth mapping, we prove $\mathcal{E} \geq \frac{M}{N}\delta^\star\!\left(\frac{B}{M}\right) + \left(1-\frac{M}{N}\right)\left(1-\frac{1}{K}\right)$, where $\delta^\star(r)$ is the inverse rate-distortion function of a uniform $K$-ary source under zero-one loss. The two terms separat
    
[^56]: 在强迫流离失所与脆弱、冲突和暴力（FCV）文档中提取数据集提及：一种基于大语言模型标签精炼的弱监督框架

    Extracting Dataset Mentions in Forced Displacement and FCV Documents: A Weakly Supervised Framework with LLM-Based Label Refinement

    [https://arxiv.org/abs/2609.12107](https://arxiv.org/abs/2609.12107)

    该论文提出一种弱监督框架，利用在通用文献上训练的轻量级模型生成候选数据集提及，再由前沿大语言模型进行上下文审查与标签精炼，从而在无需大规模人工标注语料的情况下实现强迫流离失所和FCV领域文档中的数据集引用自动提取。

    

    发展和人道主义组织生产并支持调查、行政登记册和其他数据资源，以为研究、政策和运营提供信息，但系统地识别这些数据集在何处被引用仍然困难。此类引用分散在研究论文、项目文件、人道主义报告和其他非结构化文本中，这限制了追踪数据使用以及识别数据可用性或传播方面潜在缺口的能力。我们提出了一个弱监督框架，用于将数据集提取适配到强迫流离失所和脆弱、冲突与暴力（FCV）文档，而无需首先构建大型人工标注训练语料库。一个在通用研究文献上训练的轻量级模型从无标注的领域文档中生成候选数据集提及，然后由前沿大语言模型（LLM）在上下文中进行审查，验证或拒绝候选提及并纠正……

    arXiv:2609.12107v1 Announce Type: new  Abstract: Development and humanitarian organizations produce and support surveys, administrative registries, and other data resources to inform research, policy, and operations, yet systematically identifying where these datasets are referenced remains difficult. Such references are dispersed across research papers, project documents, humanitarian reports, and other unstructured text, limiting both the ability to trace data use and to identify potential gaps in data availability or dissemination. We present a weakly supervised framework for adapting dataset extraction to forced displacement and Fragile, Conflict, and Violence (FCV) documents without first constructing a large manually labeled training corpus. A lightweight model trained on general research literature generates candidate dataset mentions from unlabeled domain documents, which a frontier large language model (LLM) reviews in context, validating or rejecting candidates and correcting
    
[^57]: 创建面向人格感知大语言模型交互的原子用户模型

    Creating an Atomic User Model for Personality-Aware Large Language Model Interaction

    [https://arxiv.org/abs/2609.12086](https://arxiv.org/abs/2609.12086)

    该论文提出原子用户模型（AUM），将用户表示为稳定身份核心加四个可解释外壳的分层结构，解决了现有助手仅依赖偏好总结而在任务变化时需反复重新学习用户的问题，并首次刻画了“人格渗漏”现象。

    

    基于大语言模型的智能助手被期望能够像其用户那样进行写作，而主流方法是单通道的：从对话历史中总结用户偏好并重新插入到上下文中。这种做法颠倒了推理的顺序。偏好是相对稳定的人格结构中依赖于任务的表层，因此仅存储偏好的系统在任务发生变化时就需要重新学习用户。首先，我们刻画了“人格渗漏”现象，即提示词的语言表面携带了人格指纹，助手在无法接触到其背后真实人格的情况下对其进行镜像模仿。其次，我们提出了原子用户模型（AUM），这是一种人类可读的用户表示方式，将一个人组织为稳定的身份核心，周围环绕着四个可解释的外壳（心理、认知与经验、行为以及社会），并辅以记录内部冲突与真实性的跨外壳条目。第三，我们将AUM视为一种检索索引……

    arXiv:2609.12086v1 Announce Type: cross  Abstract: Assistants built on large language models are expected to write as their user would, and the dominant approach is single-channel: preferences summarised from conversation history and reinserted into context. This inverts the order of inference. Preferences are the task-dependent surface of a comparatively stable personality structure, so a system storing only preferences relearns the person whenever the task changes. First, we characterise personality seepage, where a prompt's linguistic surface carries a personality fingerprint the assistant mirrors without access to the personality behind it. Second, we propose the Atomic User Model (AUM), a human-readable representation organising a person as a stable identity nucleus with four interpretable shells (psychological, cognitive and experiential, behavioural, and social), plus cross-shell entries recording internal conflict and authenticity. Third, we treat AUM as a retrieval index over 
    
[^58]: 什么算作错误？《古兰经》背诵转录文本中背诵事件的标注

    What Counts as a Mistake? Annotating Recitation Events in Quran Memorization Transcripts

    [https://arxiv.org/abs/2609.12085](https://arxiv.org/abs/2609.12085)

    本文构建了首个针对《古兰经》背诵转录文本错误事件的人工标注数据集（100个案例、348个评分单元、162个定位事件）并配套可执行评估器，为基于ASR的《古兰经》背诵自动校对建立了标签感知与错误定位的评估基准。

    

    基于自动语音识别（ASR）转录文本检查《古兰经》背诵，需要将未解决的错误与重复、自我修正、开头惯用语以及可接受的拼写差异区分开来。我们报告了对100个实际录音案例完成的人工标注：包含348个评分单元和162个定位事件，涵盖十种组合标签。一个可执行的评估器同时对标签和词语位置进行评分。简单的文本差异比较可达到标签感知F1 0.525和定位F1 0.826；经过适配的生产级清洗/对齐组件分别达到0.518和0.786，两者的精确跨度F1均为0.505。修正适配器的词语坐标后可恢复全部五个已标注的重复事件，这说明在解读基线失败原因之前必须先检查标注接口。在一项初步试验中，跨三个编码智能体和八个模型的八次单次20分钟运行，标签感知F1的范围为0.143至0.892：其中七次远高于所有基线，而一次则低于朴素差异比较的结果（摘要在此处截断）。

    arXiv:2609.12085v1 Announce Type: new  Abstract: Checking Quran recitation from an ASR transcript requires distinguishing unresolved mistakes from repetitions, repairs, opening formulas and accepted spelling differences. We report a completed human annotation of 100 production recording cases: 348 scored units and 162 localized events across ten combined labels. An executable evaluator scores labels and word positions together. A plain diff reaches label-aware F1 0.525 and localization F1 0.826; adapted production cleaner/alignment components reach 0.518 and 0.786, with exact-span F1 0.505 for both. Correcting the adapter's word coordinates recovers all five annotated repetition events, showing why annotation interfaces must be checked before interpreting baseline failures. In a preliminary pilot, eight single 20-minute runs across three coding agents and eight models span label-aware F1 0.143 to 0.892: seven land far above every baseline, and one collapses below the naive diff from a 
    
[^59]: 只需 Bash 就够了吗？面向企业数字员工智能体的工具界面实证研究

    Is Bash All You Need? An Empirical Study of Tool Interfaces for Enterprise Digital Worker Agents

    [https://arxiv.org/abs/2609.11999](https://arxiv.org/abs/2609.11999)

    研究发现仅用 Bash 的智能体在企业任务基准上以更少的 token 消耗显著优于类型化专用工具，而为其额外添加类型化工具并不能带来可检测的性能提升。

    

    在本研究中，我们探讨通用 shell 是否能在企业任务上超越专用工具。基于 shell 的智能体在编程领域已展现出强劲表现，但企业工作还涉及在应用程序与服务之间切换、与同事协作以及执行专业分析。我们使用 Opus-4.8 和 GPT-5.5 在 TheAgentCompany 和 APEX-Agents 两个基准上比较了五种工具界面：类型化工具、类型化工具加 bash、仅 bash、带持久性智能体合成工具的 bash，以及程序化工具调用（PTC，其运行程序的动被限制在类型化工具目录内）。仅 bash 在两个基准上均优于类型化工具，在 TheAgentCompany 上将得分提升 21.8-24.5 个百分点，在 APEX-Agents 上提升 4.8-7.4 个百分点，同时总 token 消耗减少 19-72%。在 bash 基础上添加类型化工具或持久性工具合成并未带来可检测的总体得分提升。PTC 相比直接类型化调用使用更少的 token，且任务表现大体相近……

    arXiv:2609.11999v1 Announce Type: cross  Abstract: In this study, we examine whether a general shell can outperform specialized tools on enterprise tasks. Shell-based agents have shown strong results in coding, but enterprise work also involves moving between applications and services, coordinating with coworkers, and performing professional analysis. We compare five tool interfaces on TheAgentCompany and APEX-Agents using Opus-4.8 and GPT-5.5: typed tools, typed tools plus bash, bash alone, bash with persistent agent-synthesized tools, and programmatic tool calling (PTC), which runs programs whose actions are restricted to a typed tool catalog. Bash alone outperforms typed tools on both benchmarks, improving score by 21.8-24.5 pp on TheAgentCompany and 4.8-7.4 pp on APEX-Agents while using 19-72% fewer total tokens. Adding typed tools or persistent tool synthesis to bash produces no detectable pooled score gain. PTC uses fewer tokens than direct typed calls with broadly similar task p
    
[^60]: 执行框架还是模型？利用污染受控的私有测试集分离智能体编程中的框架效应

    Harness or Model? Isolating the Harness Effect in Agentic Coding with a Contamination-Controlled Private Suite

    [https://arxiv.org/abs/2609.11987](https://arxiv.org/abs/2609.11987)

    本研究通过在一个污染受控的私有测试集上对相同模型分别使用厂商原生框架和第三方框架进行配对实验，发现原生框架与模型的组合并不存在显著的平均优势，从而挑战了“厂商原生搭配表现更好”这一普遍假设。

    

    智能体编程系统将语言模型与执行框架耦合在一起：框架是工具、提示词和控制流的集合，能将对话模型转变为自主的软件工程师。厂商发布的框架都是针对自家模型调优的，从业者普遍假设这种厂商原生的组合能解决更多任务。我们在一个包含256个代码仓库任务和模型训练截止日期后竞赛任务的私有、污染受控测试集上，通过同模型配对对比来检验这一假设。相同的80个任务在claude-opus-4-8上分别于claude-agent-sdk和deepagents框架下运行，在gpt-5.5上分别于openai-codex SDK和deepagents框架下运行，并以gemini-3.5-flash和deepseek-v3.2作为辅助实验组。计划中的800次运行有792次由一个隔离的评分系统完成评分。两组对比均未显示出任何一方的框架存在平均优势：Opus 4.8的差值为-1.25个百分点（48.8% vs 50.0%，任务自举95%置信区间[-10.0, +7.5]），GPT-5.5的差值为+1.25个百分点（55.6% vs 54.4%，置信区间[-4.4, +6.9]）。Opus的平均结果混合了方向相反的分层情况：原生框架……（摘要原文在此处截断）

    arXiv:2609.11987v1 Announce Type: cross  Abstract: An agentic coding system couples a language model to a harness: the tools, prompts and control flow that turn a chat model into an autonomous software engineer. Vendors ship harnesses tuned to their own models, and practitioners assume the vendor-native pairing solves more tasks. We measure that assumption with paired same-model contrasts on a private, contamination-controlled suite of 256 repository and post-cutoff contest tasks. The same 80 tasks ran under claude-agent-sdk and under deepagents on claude-opus-4-8, and under the openai-codex SDK and deepagents on gpt-5.5, with gemini-3.5-flash and deepseek-v3.2 as side cells. 792 of 800 planned runs were graded by an isolated oracle. Neither contrast resolves an average advantage for either harness: -1.25 pp for Opus 4.8 (48.8% vs 50.0%, task-bootstrap 95% CI [-10.0, +7.5]) and +1.25 pp for GPT-5.5 (55.6% vs 54.4%, CI [-4.4, +6.9]). The Opus average combines opposite strata: the native
    
[^61]: Cortex：内容分析支持软件——定性研究的资源

    Cortex: Content Analysis Support Software, a Resource for Qualitative Research

    [https://arxiv.org/abs/2609.11970](https://arxiv.org/abs/2609.11970)

    本文基于巴丹方法论并结合文献研究与半结构化访谈识别研究者实际需求，开发了面向学术研究者的内容分析支持网络应用Cortex，为定性研究提供高效的数据组织与分类工具。

    

    定性研究在人文与社会科学中被广泛应用，其特点在于通过对意义和语境的解读来深入理解现象。在各类定性数据分析方法中，内容分析作为一种成熟的技术尤为突出，它能够对文本内容进行系统的描述与阐释。然而，随着数据量的增加，数据组织、阅读和分类所需的时间成为一项重大挑战，可能延误研究进程。因此，本工作旨在基于巴丹（Bardin）方法论，开发一款面向学术研究者的内容分析支持网络应用程序。该研究采用混合方法，将有关内容分析的文献研究与对四位资深研究者的半结构化访谈相结合，以识别实际需求和具体要求。基于这些输入，Cortex软件得以开发。

    arXiv:2609.11970v1 Announce Type: cross  Abstract: Qualitative research is widely used in the human and social sciences, characterized by a deep understanding of phenomena through the interpretation of meanings and contexts. Among qualitative data analysis methods, content analysis stands out as a consolidated technique, which allows for the systematic description and interpretation of textual contents. However, as data volume increases, the time required for organization, reading, and categorization becomes a significant challenge, potentially delaying research development. Therefore, this work aimed to develop a web application to support content analysis, based on Bardin's methodology, targeted at academic researchers. The methodology adopted a mixed approach, combining bibliographic research on content analysis with semi-structured interviews with four experienced researchers, aiming to identify real needs and requirements. Based on these inputs, the Cortex software was developed t
    
[^62]: 空间作为干预不变量：面向分层城市与具身智能的跨模态预测几何

    Space as an Interventional Invariant: Cross-Modal Predictive Geometry for Stratified Cities and Em-Spaced Intelligence

    [https://arxiv.org/abs/2609.11959](https://arxiv.org/abs/2609.11959)

    本文提出将空间定义为“干预不变量”，并构建跨模态预测几何框架，使不共享度量或表示的异构感知与城市数据，仍能在因果干预层面统一揭示共同的空间结构。

    

    空间是数学、物理学、空间认知、城市科学和具身智能中的一个基础概念，然而这些领域往往将空间结构视为共享的几何容器，或将其视为彼此割裂的表示集合。这类方法难以解释异构的感知过程与城市过程如何能够共同揭示出一种共同的空间结构，尤其当不同模态不共享相同的度量或表示时更是如此。本文通过将空间定义为一种“干预不变量”来填补这一空白：即在可容许动作下，能够保持局部兼容性以及未来观测条件规律的极小关系结构。我们发展了一种跨模态预测几何方法，它整合了局部状态空间、模态特定的观测映射、一个动作群胚（action groupoid）以及一个规范的预测状态商空间，并为识别干预性结构（而非仅仅是观测性结构）给出了明确的因果条件。

    arXiv:2609.11959v1 Announce Type: cross  Abstract: Space is a foundational concept across mathematics, physics, spatial cognition, urban science, and embodied intelligence, yet these fields often treat spatial structure either as a shared geometric container or as a collection of disconnected representations. Such approaches struggle to explain how heterogeneous sensory and urban processes can jointly reveal a common spatial structure, particularly when different modalities do not share the same metric or representation. This paper addresses this gap by defining space as an interventional invariant: the minimal relational structure that preserves local compatibility and the conditional laws of future observations under admissible actions. We develop a cross-modal predictive geometry that integrates local state spaces, modality-specific observation maps, an action groupoid, and a canonical predictive-state quotient, with explicit causal conditions for identifying interventional rather t
    
[^63]: R2VC：结合检索、验证与置信度校准的模块化事实核查

    R2VC: Modular Fact-Checking with Retrieval, Verification, and Confidence Calibration

    [https://arxiv.org/abs/2609.11955](https://arxiv.org/abs/2609.11955)

    R2VC提出了一种模块化的“检索-推理-验证-校准”事实核查架构，通过混合检索、DPO对齐的生成器、NLI交叉编码器验证以及置信度校准实现证据支撑、带引用和弃权机制的事实核查，在FEVER上使8B模型准确率较基线提升13.74%。

    

    大语言模型正日益被用于自动化事实核查，但端到端的提示方法往往将证据检索、推理和不确定性估计纠缠在一起，导致故障难以诊断、置信度难以令人信任。我们提出了R2VC，一种面向基于证据、带有引用和弃权机制的事实核查的模块化“检索-推理-验证-校准”架构。R2VC融合了以下组件：维基百科上的稀疏+稠密混合检索、经过监督微调与DPO对齐的生成器（可生成多样化的结构化判定候选）、用于基于证据进行候选选择的外部NLI交叉编码器，以及用于置信度估计与选择性弃权的轻量级序列级校准器。在FEVER数据集上，配备R2VC的8B骨干模型比基线准确率提高了13.74%。消融实验表明，基于验证器的候选选择和置信度校准是性能提升的最大贡献因素。移除候选……（摘要被截断）

    arXiv:2609.11955v1 Announce Type: new  Abstract: Large language models are increasingly used for automated fact checking, but end-to-end prompting often entangles evidence retrieval, reasoning, and uncertainty estimation, making failures difficult to diagnose and confidence difficult to trust. We present R2VC, a modular retrieve, reason, verify, calibrate architecture for evidence-grounded fact checking with citations and abstention. R2VC combines hybrid sparse+dense retrieval over Wikipedia, a supervised fine-tuned and DPO-aligned generator that produces diverse structured verdict candidates, an external NLI cross-encoder for evidence-based candidate selection, and a lightweight sequence-level calibrator for confidence estimation and selective abstention. On FEVER, an 8B backbone with R2VC achieves 13.74% higher accuracy than baseline. Ablation studies show that verifier-based candidate selection and confidence calibration are the largest contributors to performance. Removing candidat
    
[^64]: PRISMA-LLM：一种AI辅助系统综述的实证报告框架

    PRISMA-LLM: An Empirical Reporting Framework for AI-Assisted Systematic Reviews

    [https://arxiv.org/abs/2609.11559](https://arxiv.org/abs/2609.11559)

    本文通过分析包含888篇论文的SciLitBench语料库，揭示了AI辅助系统综述中评估与报告的不一致问题，并提出PRISMA-LLM实证框架，将实现披露与后果敏感的评估及局限性报告分离，以规范这一领域。

    

    大语言模型和AI驱动的软件日益参与系统综述的决策过程，然而审计这些工作流程所需的信息报告却并不一致。我们分析了SciLitBench——一个包含888篇综述自动化论文和14,726条标注的语料库，以刻画方法、综述阶段使用、评估以及所报告局限性的变化。自动化已转向面向大语言模型和软件的工作流程，其中包括可能改变证据基础的阶段。自2023年以来，38.0%的软件/产品论文未报告任何评估，而大语言模型论文中这一比例为9.3%。报告覆盖率随大语言模型工作流程复杂性的增加而提高，但52%仅报告正面结果的大语言模型评估仍然存在未满足的可靠性或性能要求。基于这些模式，我们提出了PRISMA-LLM，这是一个基于实证的框架，将实现披露与后果敏感的评估和局限性报告分离开来。

    arXiv:2609.11559v1 Announce Type: new  Abstract: Large language models (LLMs) and AI-enabled software increasingly participate in systematic-review decisions, yet the information needed to audit these workflows is reported inconsistently. We analyze SciLitBench, a corpus of 888 review-automation papers with 14,726 annotations, to characterize changes in methods, review-stage use, evaluation and reported limitations. Automation has shifted toward LLM- and software-facing workflows, including stages that can alter the evidence base. Since 2023, 38.0% of software/product papers reported no evaluation, compared with 9.3% of LLM papers. Reporting coverage increased with LLM workflow complexity, yet 52% of positive-only LLM evaluations still reported an unmet reliability or performance requirement. From these patterns, we introduce PRISMA-LLM, an empirically grounded framework separating implementation disclosure from consequence-sensitive evaluation and limitation reporting.
    
[^65]: 超越求解器判定：面向自动形式化的生成式奖励模型

    Beyond Solver Verdicts: Generative Reward Models for Autoformalization

    [https://arxiv.org/abs/2609.11085](https://arxiv.org/abs/2609.11085)

    本文发现了自动形式化中的“判定保持的不忠实性”（VPU）失败模式，从理论上证明仅依赖求解器判定的验证方法无法有效检测此类错误，并提出生成式验证方法（GenV），将Z3等价性预言机蒸馏为无参照的连续等价性评分以实现可靠的验证。

    

    神经符号系统依赖数学求解器来保证推理的正确性，然而求解器从根本上无法感知一个形式化翻译是否与指定的形式化保持严格的参照等价性。我们将这一脆弱性形式化为“判定保持的不忠实性”（Verdict-Preserving-Unfaithfulness, VPU）：这是一种失败模式，即错误的编码能够成功执行并匹配预期的判定结果。我们从理论上证明，结构化的、仅基于判定的验证启发式方法在检测这些具有欺骗性的有效轨迹时，其检测能力在数学上被限制在随机概率水平。为解决这一问题，我们引入了生成式验证（GenV），通过重新利用语言模型的原生词汇空间，将离线的Z3等价性预言机蒸馏为无参照的、连续的参照等价性评分。通过决策投影logit透镜和稀疏自编码器进行的机制分析表明，这种生成式读出能够原生地提取精确的空间误差信息……

    arXiv:2609.11085v1 Announce Type: cross  Abstract: Neurosymbolic systems rely on mathematical solvers to guarantee reasoning correctness, yet solvers are fundamentally blind to whether a formal translation maintains strict reference-equivalence to a designated formalization. We formalize this vulnerability as Verdict-Preserving-Unfaithfulness (VPU): a failure mode where an incorrect encoding executes successfully and matches the expected verdict. We theoretically prove that structural, verdict-only verification heuristics are mathematically bounded to chance-level detection on these deceptively valid traces. To resolve this, we introduce Generative Verification (GenV), which distills an offline Z3-equivalence oracle into a reference-free, continuous reference-equivalence score by repurposing the language model's native vocabulary space. Mechanistic analysis via decision-projected logit lenses and sparse autoencoders shows this generative readout natively extracts precise spatial error 
    
[^66]: 基础模型能否审核在线内容？评估指令驱动与示例驱动的政策操作化方法

    Can Foundation Models Moderate Online Content? Evaluating Instruction- vs. Example-Driven Policy Operationalization

    [https://arxiv.org/abs/2609.10410](https://arxiv.org/abs/2609.10410)

    本文提出包含4,000条Bluesky人工标注帖子的新基准ModerationBench，系统比较了指令驱动与示例驱动两种政策操作化范式，发现基础模型的内容审核F1分数可达Bluesky现有审核系统的近三倍（0.60 vs. 0.22）。

    

    内容审核政策日益复杂，为其一致性的操作化实施带来了关键挑战。虽然基础模型具备应对这一挑战所需的基本能力，但它们能否可靠地审核在线内容仍是一个悬而未决的问题。在本文中，我们系统地比较了视觉语言模型（VLM）指导的两种竞争性范式：一种是指令驱动方法，模型基于政策条文进行推理；另一种是示例驱动方法，模型从先前案例中进行泛化。我们将此研究建立在ModerationBench之上——这是一个包含4,000条来自Bluesky平台、经人工标注的真实帖子的新基准。我们的实验表明，基础模型能够大幅超越Bluesky已部署的审核系统，在该基准的随机帖子上将其F1分数提升了近三倍（0.60 vs. 0.22），且指令驱动和示例驱动两种范式均取得了相当的性能。

    arXiv:2609.10410v1 Announce Type: new  Abstract: The growing complexity of content moderation policies presents a critical challenge for their consistent operationalization. While foundation models possess the basic capabilities needed to confront this challenge, whether they can reliably moderate online content remains an unanswered question. In this paper, we systematically compare two competing paradigms for Vision-Language Model (VLM) guidance: an instruction-driven approach where models reason from policy precepts, and an example-driven approach where they generalize from prior precedents. We ground this investigation in ModerationBench, a new benchmark of 4,000 manually annotated, in-the-wild posts from the Bluesky platform. Our experiments reveal that foundation models can substantially outperform Bluesky's deployed moderation system, nearly tripling its $F_1$ score (0.60 vs. 0.22) on Random Posts in the benchmark, with both instruction- and example-driven paradigms achieving co
    
[^67]: 当审计员捏造事实：大语言模型检测植入文档污染中的批次规模退化与自信幻觉

    When Auditors Fabricate: Batch-Size Degradation and Confident Hallucination in LLM Detection of Planted Document Contamination

    [https://arxiv.org/abs/2609.09696](https://arxiv.org/abs/2609.09696)

    大语言模型在单文档和小批量污染检测中表现尚可（50%-60%），但在大批量处理时检测率骤降至2.8%，且其失败方式不是承认无法处理，而是自信地捏造包括虚假污染项在内的检测结果。

    

    大语言模型越来越多地被提议作为文档质量的自动化审计工具，但它们作为植入错误检测器的可靠性却缺乏充分表征。我们构建了一个包含150篇学术论文的受污染语料库，涵盖供应链管理和医学研究领域，注入了450个已知污染项，分为三种类型：排版损坏、语义反转和荒谬的脱离语境插入。随后，我们在三种规模递增的提示机制下（单文档、小批量和大批量），评估了Google Gemini 3.0 Pro在60份文档中恢复包含180个污染项的答案密钥子集的能力。检测在小规模下保持有效，随后急剧崩溃：单文档恢复率为50%，小批量为60%，大批量仅为2.8%。大规模下的失败模式并非放弃检测，而是捏造结果。模型没有报告处理不完整，而是产生了自信的发现，包括自行编造的污染项……

    arXiv:2609.09696v1 Announce Type: new  Abstract: Large language models are increasingly proposed as automated auditors of document quality, yet their reliability as detectors of planted errors is poorly characterised. We construct a contaminated corpus of 150 academic papers spanning supply chain management and medical research, injecting 450 known contaminants of three types: typographical corruption, semantic reversal, and absurd out-of-context insertion. We then evaluate Google Gemini 3.0 Pro's ability to recover a 180-contaminant answer-key subset across 60 documents under three prompting regimes of increasing scale: single document, small batch, and large batch. Detection holds at small scale and then collapses: 50% recovery on single documents, 60% on small batches, and 2.8% on large batches. The failure mode at scale is not abstention but fabrication. Rather than reporting incomplete processing, the model produced confident findings including invented contaminants of its own, ab
    
[^68]: SAEScientist-Bench：AI智能体能否开展自主的SAE可解释性研究？

    SAEScientist-Bench: Can AI Agents Conduct Autonomous SAE Interpretability Research?

    [https://arxiv.org/abs/2609.09113](https://arxiv.org/abs/2609.09113)

    本文提出SAEScientist-Bench基准，评估AI智能体能否作为科学家自主运用稀疏自编码器（SAE）工具，进行机制可解释性的自主科学发现研究。

    

    虽然关于递归自我改进（RSI）的研究主要集中于自动化模型训练流程，但可靠的自主发展还需要一个缺失的关键支柱：事后监控与审计，以理解模型学到了什么并确保安全对齐。机制可解释性工具对于弥合这一差距至关重要，其中稀疏自编码器（SAE）通过隔离可解释特征用于模型检查与引导，成为这一领域的基石。在本文中，我们引入SAEScientist-Bench，用于评估AI智能体能否作为科学家，利用SAE工具进行自主的机制发现。给定一个目标概念，智能体需设计对比探针，并在Gemma-2-9B-IT模型的Gemma Scope词典（包含131K+特征）中导航以发现最优特征，随后基于锚定在Neuronpedia上的精选专家参考特征进行评估，评估维度包括激活排名、对比文本上的概念选择性以及因果引导效果。

    arXiv:2609.09113v1 Announce Type: new  Abstract: While research on recursive self-improvement (RSI) has predominantly automated model training pipelines, reliable autonomous development demands a missing pillar: post-hoc monitoring and auditing to understand what models learn and ensure safe alignment. Mechanistic interpretability tools are essential to bridge this gap, among which Sparse Autoencoders (SAEs) serve as a cornerstone by isolating interpretable features for model inspection and steering. In this paper, we introduce SAEScientist-Bench to evaluate whether AI agents can act as scientists utilizing SAE tools for autonomous mechanistic discovery. Given a target concept, an agent designs contrastive probes and navigates a Gemma Scope dictionary of 131K+ features in Gemma-2-9B-IT to discover the optimal feature, evaluated against curated expert reference features anchored on Neuronpedia across activation rank, concept selectivity on contrastive texts, and causal steering. Across 
    
[^69]: 从分数到证据：可审计的决策可以改进语音深伪检测

    From Scores to Evidence: Auditable Decisions Can Improve Speech Deepfake Detection

    [https://arxiv.org/abs/2609.08899](https://arxiv.org/abs/2609.08899)

    该论文提出一种可审计的决策记录方法，将被动检测分数、条件性键控探针分数、检索支持和说话者画像边际四个线索纳入后期校准步骤，使语音深伪检测的最终决策在保持标量的同时保留证据来源信息，从而提升检测决策的可信度与可解释性。

    

    语音深伪能够以足以欺骗听众和自动化系统的方式逼真地模仿说话者的声音。这推动了语音深伪检测领域的强劲进展，但大多数检测器最终仍只输出每条语音的一个分数。该分数对于排序系统很有用，但对于为什么某个临界样本应该被信任、搁置还是复核，它几乎没有提供任何信息。两条语音可能因不同原因落入同一分数区间，例如被动证据与检索证据不一致，或者键控探针不可用。我们提出了一个问题：最终决策能否在保留这些来源信息的前提下仍然保持标量形式。我们通过一种可审计的决策记录来回答这个问题，该记录将四个对齐的线索引入后期校准步骤：被动检测器分数、在标记衍生样本上的条件性键控探针分数、检索支持度以及说话者画像边际，同时附带明确的分歧坐标。在包含4,080个样本的……

    arXiv:2609.08899v2 Announce Type: replace-cross  Abstract: Speech deepfakes can mimic a speaker's voice convincingly enough to deceive listeners and automated systems. This has driven strong progress in speech deepfake detection, but most detectors still end with one score per utterance. That score is useful for ranking systems, yet it says little about why a borderline item should be trusted, deferred, or reviewed. Two utterances can fall in the same score band for different reasons, for example because passive and retrieval evidence disagree or because the keyed probe is unavailable. We ask whether the final decision can remain scalar without discarding that provenance. We answer this question with an auditable decision record that carries four aligned cues into a late calibration step: a passive detector score, a conditional keyed-probe score on a marked derivative, retrieval support, and a speaker-profile margin, together with explicit disagreement coordinates. On the 4,080-example
    
[^70]: LatentMD：对大语言模型生成文本中Markdown边界失败的基准测试

    LatentMD: Benchmarking Markdown Boundary Failures in LLM-Generated Text

    [https://arxiv.org/abs/2609.06993](https://arxiv.org/abs/2609.06993)

    LatentMD是一个将内容正确性与边界正确性分离的基准测试，它揭示了LLM生成的Markdown中边界失败问题普遍存在——38%的内容正确输出实际上边界已损坏。

    

    大语言模型（LLM）生成的Markdown文本日益被渲染器、智能体、代码提取器和结构化下游流水线所使用。然而，现有评估往往将内容质量与格式遵循混为一谈，导致Markdown边界失败问题未被充分测量。我们提出了LatentMD，这是一个用于诊断LLM生成Markdown中CommonMark级别围栏边界失败的基准测试与评估协议。LatentMD将内容正确性与边界正确性分离，能够检测出内容正确但边界损坏的输出。该基准包含4,179个提示词以及一个用于对任意模型输出进行评分的命令行工具（CLI）。在9个大语言模型和约37,600次生成中，我们发现Markdown边界失败非常普遍：38.0%的有效主网格输出内容正确但边界损坏，且在未指定格式的提示词下以及在一个小型人工撰写的验证集中均存在大量的边界损坏。消融实验表明……

    arXiv:2609.06993v1 Announce Type: cross  Abstract: Large language models (LLMs) increasingly generate Markdown that is consumed by renderers, agents, code extractors, and structured downstream pipelines. Yet existing evaluations often conflate content quality with format adherence, leaving Markdown boundary failures under-measured. We introduce LatentMD, a benchmark and evaluation protocol for diagnosing CommonMark-level fence-boundary failures in LLM-generated Markdown. LatentMD separates content correctness from boundary correctness, enabling detection of outputs that are content-correct but boundary-broken. The benchmark contains 4,179 prompts and a CLI for scoring arbitrary model outputs. Across 9 LLMs and roughly 37,600 generations, we find that Markdown boundary failures are widespread: 38.0% of valid main-grid outputs are content-correct but boundary-broken, with substantial boundary breakage under unspecified prompts and in a small human-authored validation set. Ablations show 
    
[^71]: 通过潜空间推理！让潜空间视觉推理成为必需

    Reason Through the Latent! Making Latent Visual Reasoning Necessary

    [https://arxiv.org/abs/2609.06746](https://arxiv.org/abs/2609.06746)

    提出因果视觉循环推理（CVRR）框架，通过在解码前移除视觉状态和多模态KV缓存，迫使循环隐藏状态成为唯一的图像条件信息通路，从而确保潜空间视觉推理真正被模型依赖。

    

    潜空间视觉推理旨在通过隐藏状态计算而非显式的文本思维链来进行多模态推理。然而，视觉信息存在于潜空间状态中并不意味着模型在生成答案时真正依赖该状态，尤其是当其他基于图像的替代路径仍然可用时。我们提出了因果视觉循环推理（CVRR），该方法在保留预训练视觉能力的同时，使循环计算成为预测所必需的基于图像的条件路径。CVRR在预训练视觉语言模型融合图像之后，从问题的隐藏状态初始化循环过程，然后在重复读取相同固定视觉证据的同时反复更新该状态。在解码之前，视觉状态和原始的多模态KV缓存会被移除，从而确保只有最终的循环状态携带基于图像的条件信息。

    arXiv:2609.06746v1 Announce Type: new  Abstract: Latent visual reasoning aims to perform multimodal reasoning through hidden-state computation rather than explicit textual chains of thought. However, visual information being present in a latent state does not imply that the model actually relies on that state when producing its answer, especially when alternative image-conditioned paths remain available. We introduce \textbf{C}ausal \textbf{V}isual \textbf{R}ecurrent \textbf{R}easoning (CVRR), which preserves pretrained visual competence while making recurrent computation the required image-conditioned path to prediction. CVRR initializes recurrence from the question hidden state after the pretrained vision-language model has incorporated the image, then repeatedly updates this state while re-reading the same fixed visual evidence. Before decoding, visual states and the original multimodal KV cache are removed so that only the final recurrent state carries image-conditioned information
    
[^72]: 分解LLM评审器的不确定性以精准定位专家标注

    Decomposing LLM-Judge Uncertainty to Target Expert Labels

    [https://arxiv.org/abs/2609.06444](https://arxiv.org/abs/2609.06444)

    该论文提出一种小型贝叶斯模型，将LLM评审器的总不确定性分解为可被专家标注消除的认知不确定性和不可消除的偶然不确定性，使专家只需标注评审器真正无知的项目，在ChaosNLI数据集上比使用总不确定性多消除83%的误差。

    

    LLM评审器可以大规模评估模型输出，专家应该只在它最不确定的地方进行标注。然而其天然的升级信号混淆了两种不确定性：偶然不确定性——专家群体中真实存在的分歧，标注无法减少这种不确定性；以及认知不确定性——评审器自身的无知，标注可以减少这种不确定性。本文提出一个小型贝叶斯模型来分离这两种不确定性：通过对已收集的标注进行回归，学习在多大程度上信任黑盒评审器的预测。两个组成部分都可以通过简单的公式计算得出，无需采样或额外的评审器调用。在一个面对完全已知真值的真实LLM评审器上，这两种不确定性成分被成功分离，且评审器声称的置信度并不能反映其真实误差。在真实的人类分歧数据集（ChaosNLI）上，在相同的专家标注量下，基于认知不确定性的排序比使用总不确定性多消除83%的误差，不过在该数据集上简单地升级标注最少的项目也能达到同样效果。我们证明了我们可以估计评审器在哪里是无知的，而不是专家们真正存在分歧的地方。

    arXiv:2609.06444v2 Announce Type: replace  Abstract: An LLM judge evaluates outputs at scale. Experts should label only where it is least sure. Its natural escalation signal conflates two uncertainties: aleatoric, real disagreement in the expert pool, which labels cannot reduce, and epistemic, the judge's ignorance, which labels do reduce. A small Bayesian model separates them: a regression on labels already collected learns how far to trust a black-box judge's prediction. Both components follow as simple formulas, with no sampling or further judge calls. The components isolate on a real LLM judge against exactly known truth, and stated confidence is no guide to its actual error. On real human disagreement (ChaosNLI) the epistemic ranking removes 83% more error than total uncertainty for the same expert labels, though simply escalating the least-labelled items does as well there. We demonstrate we can estimate where a judge is ignorant rather than where experts genuinely disagree, and 
    
[^73]: EVOHARNESSBENCH：你的智能体能否跟上不断演进的运行框架？

    EVOHARNESSBENCH: Can Your Agents Keep Pace with an Evolving Harness?

    [https://arxiv.org/abs/2609.04280](https://arxiv.org/abs/2609.04280)

    提出了EVOHARNESSBENCH基准，首次将非平稳性从任务流转移到智能体运行框架本身（工具、技能、智能体）的持续演进上，用于评估智能体在框架不断变化环境下的适应与保留能力。

    

    现代基于大语言模型（LLM）的智能体通过一个由工具、可复用技能和专职智能体组成的运行框架来运行，该框架决定了智能体能够观察到什么以及能够做什么。在实践中，这个运行框架会随着新能力的加入而不断演进。我们提出了EVOHARNESSBENCH，这是一个在三个维度（工具、技能和智能体）上评估智能体在受控运行框架演进条件下表现的基准。与现有的智能体持续学习基准不同——后者通常将非平稳性（即随时间变化的内容）置于任务流中而保持运行框架固定不变——EVOHARNESSBENCH将非平稳性置于外部提供的运行框架本身。该基准包含17个由基于验证器的基准确定性构建的多阶段运行框架流，共涵盖802个任务、520个工具、42个技能和62个智能体。我们评估了对应于运行框架演进核心挑战的两种互补设置：部署评估（用于隔离测试保留能力）

    arXiv:2609.04280v1 Announce Type: cross  Abstract: Modern LLM-based agents operate through a harness of tools, reusable skills, and specialist agents that shapes what they observe and what they can do. In practice, this harness continually evolves as new capabilities are added. We introduce EVOHARNESSBENCH, a benchmark for evaluating agents under controlled harness evolution across three axes (tools, skills, and agents). Unlike existing continual-learning benchmarks for agents, which typically place non-stationarity (i.e., what changes over time) in the task stream while keeping the harness fixed, EVOHARNESSBENCH places non-stationarity in the externally supplied harness itself. It contains 17 multi-stage harness streams constructed deterministically from verifier-based benchmarks, comprising 802 tasks, 520 tools, 42 skills, and 62 agents. We evaluate two complementary settings corresponding to the central challenges of harness evolution: deployment evaluation, which isolates retention
    
[^74]: NS-Copilot：一个由大语言模型驱动的自主神经科学分析智能体系统

    NS-Copilot: An LLM-Driven Agent System for Autonomous Neuroscience Analysis

    [https://arxiv.org/abs/2609.01971](https://arxiv.org/abs/2609.01971)

    NS-Copilot是一个由大语言模型驱动的多智能体系统，能够自主选择和协调神经科学领域的各类预训练模型，支持EEG和细胞外尖峰数据等关键模态，为专业神经科学分析任务提供端到端的自主工作流程。

    

    人工智能正在迅速推动神经科学的发展，然而由于显著的跨学科壁垒，许多实验室未能充分释放其潜力。尽管针对生理数据的预训练神经模型进展迅速，但其异构的架构和特定模态的限制阻碍了系统性的整合、选择与评估。尽管基于大语言模型（LLM）的智能体系统在智能科学应用方面近期取得了进展，现有方法往往仍缺乏有效选择和协调多样化神经科学预训练模型并处理该领域独特数据类型所需的领域专业知识。我们提出了NS-Copilot，一个由大语言模型驱动的神经科学分析多智能体系统，它能够自主支持多样化专业任务的端到端工作流程。该系统统一了特定领域的预训练模型，并支持关键的神经科学模态，包括脑电图（EEG）和细胞外尖峰数据……

    arXiv:2609.01971v1 Announce Type: new  Abstract: AI is rapidly advancing neuroscience, yet many laboratories fail to fully unleash its potential due to significant interdisciplinary barriers. While pre-trained neural models for physiological data are progressing quickly, their heterogeneous architectures and modality-specific constraints hinder systematic integration, selection, and evaluation. Despite recent advances in large language model (LLM)-based agent systems for intelligent scientific applications, existing approaches often still lack the domain expertise required to effectively select and coordinate diverse neuroscience pre-trained models and handle unique data types in this domain. We present NS-Copilot, an LLM-driven multi-agent system for neuroscience analysis that autonomously supports end-to-end workflows for diverse professional tasks. It unifies domain-specific pre-trained models and supports key neuroscience modalities, including EEG and extracellular spike data, thro
    
[^75]: NSIDDx：面向低资源环境的神经符号化、以临床医生为中心的鉴别诊断设计框架

    NSIDDx: A Design Framework for Neuro-Symbolic, Practitioner-First Differential Diagnosis in Low-Resource Settings

    [https://arxiv.org/abs/2609.00256](https://arxiv.org/abs/2609.00256)

    本文提出NSIDDx设计框架，主张在低资源环境下将临床医生作为主动推理主体融入鉴别诊断，通过三值症状编码、矛盾检测、审计字符串和医生覆盖权的神经符号流水线在消费级硬件上离线运行，弥合了LLM诊断系统的头条准确率与可验证临床可靠性之间的差距。

    

    基于大语言模型（LLM）的诊断系统在基准测试中取得了很高的语义准确率，但在临床少见表现上的开放式评估揭示了其“头条准确率”与可验证的临床可靠性之间存在系统性差距。我们在两个队列中评估了“LLM+罕见病RAG”流水线，结果表明该范式产生的输出往往高度自信却经常无法验证，并且系统性地抗拒临床医生的质询。我们提出NSIDDx（神经符号集成鉴别诊断系统），这是一个设计框架，主张低资源环境下的鉴别诊断系统必须将临床医生视为主动的推理主体。我们通过一个包含三值症状编码、矛盾检测、审计字符串和医生覆盖权的神经符号流水线来实现这一理念——该系统可在消费级硬件上离线运行。我们提炼出五条“临床医生在环”临床NLP的设计原则，并呼吁开展必要的前瞻性研究以验证该方法。

    arXiv:2609.00256v1 Announce Type: new  Abstract: LLM-based diagnostic systems achieve high semantic accuracy on benchmarks, but open-ended evaluation on clinically uncommon presentations reveals a systematic gap between headline accuracy and verifiable clinical reliability. We evaluate an LLM+rare-disease-RAG pipeline across two cohorts and show that the paradigm produces confident outputs that are frequently unverifiable and systematically resistant to clinician interrogation. We present NSIDDx (Neuro-Symbolic Integrated Differential Diagnosis System), a design framework arguing that DDx systems in low-resource settings must treat the clinician as an active reasoning agent. We instantiate this through a neuro-symbolic pipeline with ternary symptom encoding, contradiction detection, audit strings, and practitioner override - running offline on consumer hardware. We distill five design principles for clinician-in-the-loop clinical NLP and invite the prospective studies needed to validat
    
[^76]: 多模态大语言模型是先看后读吗？诊断情境性谄媚现象

    Do Multimodal LLMs See Before They Read? Diagnosing Contextual Sycophancy

    [https://arxiv.org/abs/2609.00067](https://arxiv.org/abs/2609.00067)

    该论文诊断了多模态大语言模型易受外部文本误导而忽视冲突图像证据的“多模态情境性谄媚”问题，并提出“系统2视觉仲裁”（S2VA）方法，通过让视觉证人在读取文本前先独立判断，在六个模型上将准确率显著提升19.7至44.1分。

    

    外部文本可以覆盖多模态大语言模型中与之冲突的图像证据，我们将这种失败称为“多模态情境性谄媚”。我们引入了一个包含998个案例的诊断方法，该方法独立地变化视觉证据、常识先验和外部文本三个因素，并通过围绕“情境盲视”的视觉证人调整信息边界，来探究这种失败在何时发生。在与Gemini生成的虚假文本配对的异常图像上，GPT-5.1在联合条件下的得分仅为7.9%；当直接对情境盲视的证人报告进行评分时，得分为49.7%；在使用匹配的双调用证人-仲裁者管道（即让证人接触文本）时，得分为63.7%；而在“系统2视觉仲裁”（S2VA，即对证人隐瞒文本）下，得分达到84.2%。在六个模型上，S2VA相比直接证人报告提升了19.7至44.1分，且所有配对的95%置信区间均不包含零。最佳的信息边界并非统一不变：文本情境对某些情况……

    arXiv:2609.00067v1 Announce Type: cross  Abstract: External text can override conflicting image evidence in multimodal large language models, a failure we call multimodal contextual sycophancy. We introduce a 998-case diagnostic that independently varies visual evidence, commonsense priors, and external text, and probe when this failure arises by moving the information boundary around a context-blind visual witness. On abnormal images paired with Gemini-generated false text, GPT-5.1 scores 7.9% under joint conditioning, 49.7% when the context-blind witness report is scored directly, 63.7% under a matched two-call witness-arbiter pipeline that exposes the witness to the text, and 84.2% under System-2 Visual Arbitration (S2VA), which withholds the text from the witness. Across six models, S2VA improves over the direct witness report by 19.7 to 44.1 points, with all paired 95% confidence intervals excluding zero. The best information boundary is not uniform: textual context scaffolds some
    
[^77]: 基于语义大语言模型代理的闭环贝叶斯分子逆向设计

    Closed-Loop Bayesian Molecular Inverse Design with Semantic LLM Surrogates

    [https://arxiv.org/abs/2608.22967](https://arxiv.org/abs/2608.22967)

    该论文提出了一种闭环贝叶斯分子逆向设计框架，通过将大型语言模型作为代理直接处理文本形式的任务指令和优化历史，以在有限预算下提高匹配目标性质的分子比例。

    

    arXiv:2608.22967v1 公告类型：新 摘要：实际的分子逆向设计很少是一次性生成问题；它通常采取闭环候选池富集的形式，在有限的预测预算下，目标是增加生成分子中匹配所需性质特征的比例。贝叶斯优化（BO）为此场景提供了自然框架，然而标准高斯过程代理通常在压缩的连续嵌入中操作，这丢弃了化学家自然用于决定下一步探索位置的子结构和参考相似性信号。我们提出了一种闭环框架，其中代理而非生成器被视为设计选择的核心，并通过一个冻结的大型语言模型实例化该框架，该模型直接以文本形式推理任务指令、SMILES级优化历史和预测反馈。在每次迭代中，代理返回...

    arXiv:2608.22967v1 Announce Type: new  Abstract: Practical molecular inverse design is rarely a one-shot generation problem; it often takes the form of closed-loop candidate-pool enrichment, where under a limited oracle budget the goal is to \emph{increase the fraction of generated molecules that match a desired property profile}. Bayesian optimization (BO) offers a natural framework for this setting, yet standard Gaussian-process surrogates typically operate in compressed continuous embeddings, which discard the substructural and reference-similarity signals that chemists naturally use to decide where to look next. We propose \textbf{\method}, a closed-loop framework in which the surrogate, rather than the generator, is treated as the locus of design choice, and instantiate it with a frozen large language model that reasons directly over the task instruction, SMILES-level optimization history, and oracle feedback in their native textual form. At each iteration, the surrogate returns a
    
[^78]: 洗白仇恨、污蔑无害内容：针对基于LLM的内容审核的标注者风格反驳攻击

    Whitewashing Hate, Smearing Harmless Content: Annotator-Style Rebuttal Attacks on LLM-Based Moderation

    [https://arxiv.org/abs/2608.22230](https://arxiv.org/abs/2608.22230)

    本研究揭示了标注者风格的反驳攻击能显著破坏LLM仇恨言论审核的准确性，且洗白与污蔑两种操纵方向存在模型特定的不对称效应。

    

    大型语言模型（LLMs）越来越多地被用于仇恨言论审核，通常出现在人类与AI协作的工作流程中，其中审核者在最终决策前提供反馈。这种反馈引入了两种操纵方向：将仇恨内容洗白为正常内容，以及将正常内容污蔑为仇恨内容。本研究考察了初始正确的模型判断对标注者风格反驳的敏感性，并分析了攻击有效性是否因操纵方向而异。我们引入了一种重新判断协议，该协议通过决策边界扰动和对抗性理由扩展了直接矛盾。在多个LLM和两个仇恨言论数据集上的实验表明，标注者风格的反驳显著降低了审核性能，在多轮设置中效果更强。结果进一步揭示了在攻击配置中，洗白和污蔑之间存在稳定且模型特定的不对称性，这表明...

    arXiv:2608.22230v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used for hate speech moderation, often within human--AI workflows in which reviewers provide feedback before a final decision. Such feedback introduces two manipulation directions: whitewashing hateful content as normal and smearing normal content as hateful. This study examines the susceptibility of initially correct model judgments to annotator-style rebuttals and analyzes whether attack effectiveness differs across manipulation directions. We introduce a rejudge protocol that extends direct contradiction with decision-boundary perturbations and adversarial rationales. Experiments with multiple LLMs on two hate speech datasets show that annotator-style rebuttals substantially degrade moderation performance, with stronger effects in multi-turn settings. The results further reveal stable, model-specific asymmetries between whitewashing and smearing across attack configurations, indicating dis
    
[^79]: 葡萄牙语语言模型：一项系统性映射研究

    Language Models for Portuguese: A Systematic Mapping Study

    [https://arxiv.org/abs/2608.18138](https://arxiv.org/abs/2608.18138)

    本文对葡萄牙语语言模型进行了系统性映射研究，梳理了46个模型的现状，填补了该领域信息分散的空白。

    

    arXiv:2608.18138v1 公告类型：交叉 摘要：近年来，语言模型的快速发展通过广泛的应用彻底改变了自然语言处理领域。然而，语言模型的发展在所有语言中并不均衡。就葡萄牙语而言，近年来学术界和公司日益努力开发语言模型并为葡萄牙语创建数据资源。这些努力导致了葡萄牙语语言模型生态系统日益多样化。然而，关于这些模型的信息仍分散在科学出版物、技术报告、模型库和项目文档中。本调查对为葡萄牙语开发的语言模型进行了系统性映射研究，提供了该领域当前状态的全面概述。我们共映射了46个模型，从基础模型、架构等多个方面对其进行特征化描述。

    arXiv:2608.18138v1 Announce Type: cross  Abstract: In recent years, the rapid development of language models has transformed the field of Natural Language Processing through a wide range of applications. However, the development of language models has not progressed uniformly across all languages. In the case of the Portuguese language, there has recently been a growing effort by academia and companies to develop language models and create data resources for Portuguese. These efforts have resulted in the rise of an increasingly diverse ecosystem of language models for Portuguese. However, information on these models remains dispersed in scientific publications, technical reports, model repositories, and project documentation. This survey presents a systematic mapping study of language models developed for Portuguese, providing a comprehensive overview of the current state of the field. We map a total of 46 models, characterizing them by various aspects, including base model, architectu
    
[^80]: Slot2Text：面向高效且空间可追溯的手术多模态大语言模型的以对象为中心的视觉分词方法

    Slot2Text: Object-Centric Visual Tokenization for Efficient and Spatially Traceable Surgical MLLMs

    [https://arxiv.org/abs/2608.01473](https://arxiv.org/abs/2608.01473)

    Slot2Text提出了一种以对象为中心的双模式手术多模态大语言模型，通过将密集视觉特征编码为少量带区域标签的槽位标记作为视觉词元，在大幅降低推理成本的同时实现了答案的空间可追溯性。

    

    用于手术场景理解的多模态大语言模型（MLLM）通常会将数百个密集视觉标记注入语言模型，导致推理成本高昂，且生成的答案缺乏空间可追溯性。我们提出了Slot2Text，这是一种双模式手术多模态大语言模型，它用编码为槽位潜在变量的紧凑区域集合来替代视觉输入的密集表示。Slot2Text不依赖于视觉编码器与语言之间的对比对齐，而是将自监督视觉特征分组为少量区域——即槽位，语言模型将其作为带有区域标签的视觉标记来使用。Slot2Text-Fast使用槽位前缀来回答手术问题。Slot2Text-Reason还能识别并定位与推理相关的区域，将语言输出链接到相应的槽位标记、掩码或区域。在多个视觉问答和视觉定位基准上的实验表明，Slot2Text-Fast具有竞争力。

    arXiv:2608.01473v2 Announce Type: replace-cross  Abstract: Multimodal large language models (MLLM) for surgical scene understanding typically inject hundreds of dense visual tokens into a language model, leading to costly inference and limited spatial traceability for generated answers. We present Slot2Text, a dual-mode surgical MLLM that replaces dense representations of visual input with a compact set of regions encoded as slot latents. Instead of relying on contrastive alignment of the visual encoder with language, Slot2Text groups self-supervised vision features into a few regions--slots that are consumed by the language model as area-labeled visual tokens. Slot2Text-Fast uses the slot prefix to answer surgical questions. Slot2Text-Reason also identifies and locates areas relevant for reasoning, linking language outputs to corresponding slot tokens, masks or regions. Experiments on multiple visual question answering and visual grounding benchmarks show that Slot2Text-Fast is compet
    
[^81]: 一个虚假的平均值：思维链监控器在其作为唯一防线之处崩溃

    A False Average: Chain-of-Thought Monitors Collapse Where They Are the Only Defense

    [https://arxiv.org/abs/2608.00583](https://arxiv.org/abs/2608.00583)

    仅重写智能体的思维链推理（保持动作完全不变）即可在一次无梯度攻击中将CoT监控器的捕获率从约95%暴跌至11%以下，揭示监控器的总体准确率是掩盖其在唯一防线场景下近乎全面失效的虚假平均值。

    

    思维链监控旨在捕捉那些在动作层面看起来干净、只在推理中暴露自身的奖励作弊行为。我们证明，这恰恰是控制推理的攻击者能够击败它的地方。仅重写智能体的推理内容使其看起来像是善意的工程工作，同时逐字复制每个命令和输出以保持漏洞利用不变，在一次无需梯度的攻击中，就能将留出监控器在该子集上的捕获率从约95%降至11%以下。监控器的总体准确率是一个虚假的平均值：它由动作层面就会暴露的作弊行为主导，从而掩盖了这种重写在CoT监控是唯一信号的子集上造成的近乎全面的崩溃。该攻击可跨监控器系列和智能体模型迁移，并在真实智能体上复现，不过在面对经过校准的监控器时，规避行为主要集中在最强的智能体上。仅基于轨迹的防御只能部分恢复捕获率，即使是预先针对该攻击进行训练的防御也是如此。

    arXiv:2608.00583v2 Announce Type: replace-cross  Abstract: Chain-of-thought (CoT) monitoring is meant to catch the reward hacks that look clean in the actions and betray themselves only in the reasoning. We show that this is exactly where an adversary who controls the reasoning can defeat it. Rewriting only an agent's reasoning to read as good-faith engineering, while copying every command and output verbatim so the exploit is unchanged, drops a held-out monitor's catch rate on that subset from about 95% to under 11% in one gradient-free shot. A monitor's aggregate accuracy is a false average: dominated by hacks the actions give away, it hides the near-total collapse this rewrite produces on the subset where CoT monitoring is the only signal. The attack transfers across monitor families and agent models, reproduces with live agents, though against a calibrated monitor evasion concentrates in the strongest agent. Trace-only defenses recover it only partially, even one primed on the atta
    
[^82]: SeDeM：面向长文本问答的隐状态记忆选择性解压缩

    SeDeM: Selective Decompression of Hidden-State Memories for Long-Context Question Answering

    [https://arxiv.org/abs/2608.00311](https://arxiv.org/abs/2608.00311)

    SeDeM提出一种选择性解压缩框架，将上下文存储为紧凑的隐状态记忆块，并仅解压缩与查询相关的块来调节解码器，从而在长文本问答中兼顾效率与准确性。

    

    使用大语言模型（LLM）进行长上下文推理的代价高昂：预填充阶段的自注意力机制随序列长度呈二次方增长，键值（KV）缓存也随处理token数量的增加而不断增长。此外，更大的上下文窗口并不能确保可靠地使用证据。上下文压缩可以降低这一成本，但许多软压缩方法将LLM本身用作压缩器，并依赖紧凑的记忆token同时承担保存信息和调节解码器两项任务。我们提出了SeDeM，这是一种将紧凑记忆存储与解码器调节解耦的选择性解压缩框架。SeDeM将上下文存储为紧凑的隐状态记忆块，选择与查询相关的块，并仅对被选中的块进行解压缩以用于调节解码器。这样，解码器既避免了全上下文处理，也避免了直接从高度压缩的记忆槽中进行生成。在四个长上下文问答基准测试上，SeDeM取得了比压缩方法更高的问答分数。

    arXiv:2608.00311v2 Announce Type: replace  Abstract: Long-context inference with large language models (LLMs) is costly: self-attention during prefill scales quadratically with sequence length, and the key-value (KV) cache grows with the number of processed tokens. Larger context windows also do not ensure reliable evidence use. Context compression reduces this cost, but many soft-compression methods use LLMs as compressors and rely on compact memory tokens both to preserve information and to condition the decoder. We propose SeDeM, a selective decompression framework that decouples compact memory storage from decoder conditioning. SeDeM stores context as compact hidden-state memory blocks, selects query-relevant blocks, and decompresses only the selected blocks for decoder conditioning. Thus, the decoder avoids both full-context processing and direct generation from highly compressed memory slots. On four long-context QA benchmarks, SeDeM achieves higher QA scores than the compression
    
[^83]: AdaRoPE：并非所有注意力头都应均等地旋转和缩放

    AdaRoPE: Not All Attention Heads Should Rotate and Scale Equally

    [https://arxiv.org/abs/2607.19363](https://arxiv.org/abs/2607.19363)

    AdaRoPE为每个注意力头配备可学习的旋转频率和注意力缩放因子，打破了传统RoPE对所有头统一处理的做法，从而显著提升嵌入维度利用率并在长上下文设置中持续优于现有RoPE变体。

    

    旋转位置编码被广泛应用于Transformer中以编码位置信息，然而标准实现对所有注意力头强制使用统一的频率调度和缩放方式。借助简化的检索任务和长度泛化场景，我们从实证和理论两方面证明，具有不同功能角色的注意力头需要不同的频率范围和注意力缩放因子才能有效运作。忽视这一结构会导致嵌入维度利用不充分以及性能下降，尤其是在长上下文设置中。为解决这些局限，我们提出AdaRoPE，它为每个注意力头配备可学习的旋转频率和注意力缩放因子。采用AdaRoPE的预训练大语言模型持续优于现有的RoPE变体，包括部分RoPE和NoPE基线。在上下文扩展方面，我们进一步表明统一频率和……（摘要内容不完整，此处截止）

    arXiv:2607.19363v3 Announce Type: replace-cross  Abstract: Rotary Position Embedding (RoPE) is widely adopted in Transformers to encode positional information, yet standard implementations enforce a uniform frequency schedule and scaling across all attention heads. Using simplified retrieval tasks and length generalization scenarios, we show -- both empirically and theoretically -- that heads with different functional roles require distinct frequency ranges and attention scaling factors to operate effectively. Ignoring this structure leads to suboptimal utilization of embedding dimensions and degraded performance, particularly under long-context settings. To address these limitations, we propose AdaRoPE, which equips each attention head with learnable rotation frequencies and attention scaling factors. Pretrained LLMs with AdaRoPE consistently outperform existing RoPE variants, including partial RoPE and NoPE baselines. For context extension, we further show that uniform frequency and 
    
[^84]: 构造即忠实：面向多文档摘要的声明锚定归因方法

    Faithful by Construction: Claim-Anchored Attribution for Multi-Document Summarization

    [https://arxiv.org/abs/2606.23989](https://arxiv.org/abs/2606.23989)

    提出CAMS框架，将声明级归因嵌入“提取—选择—改写”流程，使多文档摘要中的每句话都能锚定到经过验证、可溯源的源文本片段，从而在构造层面保证摘要的忠实性。

    

    端到端大语言模型（LLM）能够生成流畅的多文档摘要，但仍容易产生幻觉，且其提供的归因通常较为粗糙（仅指向整篇文档或段落）并属于事后生成，导致每条摘要陈述都难以验证。我们重新审视模块化的“提取—选择—改写”范式，并将其中间表示重新构建为归因的基本单元。我们提出了CAMS（Claim-Anchored Multi-document Summarization，声明锚定多文档摘要）框架，该框架：(i) 从每个源文档中提取带有词元级溯源信息的原子声明；(ii) 跨文档聚类等价声明，同时标记源间冲突；(iii) 选择一个兼顾支持度与显著性的子集；(iv) 将所选内容改写为摘要，其中每个句子都锚定到一个经过支持性检验的声明，该声明可回链至一个或多个源文本片段。由于内容在生成之前就已完成定位，整个流程从构造上即是面向归因的。

    arXiv:2606.23989v3 Announce Type: replace-cross  Abstract: End-to-end large language models (LLMs) produce fluent multi-document summaries but remain prone to hallucination, and the attributions they offer are typically coarse (whole documents or passages) and generated post hoc, leaving each summary statement hard to verify. We revisit the modular Extract--Select--Rewrite paradigm and recast its intermediate representation as the unit of attribution. We present CAMS, a Claim-Anchored Multi-document Summarization framework that (i) extracts atomic claims with token-level provenance from every source document, (ii) clusters equivalent claims across documents while flagging inter-source conflicts, (iii) selects a support-aware and salient subset, and (iv) rewrites the selection into a summary in which every sentence is anchored to a support-checked claim that links back to one or more source spans. Because content is localized before it is realized, the pipeline is attribution-oriented b
    
[^85]: 有能力却粗心大意：计算机使用代理是否遵循情境完整性原则？

    Capable but Careless: Do Computer-Use Agents Follow Contextual Integrity?

    [https://arxiv.org/abs/2606.23189](https://arxiv.org/abs/2606.23189)

    本文提出AgentCIBench评估基准，首次系统揭示了计算机使用代理在跨应用操作中违反情境完整性原则的严重隐私风险，评估发现15个前沿代理中有11个存在信息泄露问题，并归纳出视觉同址、任务歧义过度分享和接收者错位三种典型失败模式。

    

    计算机使用代理（CUA）如今可以在电子邮件、日历和待办事项列表等个人应用中代表用户执行操作。这种跨应用访问虽然实用，但也带来了一个在很大程度上被忽视的隐私风险：当代理在某个上下文中工作时，它可能会引入来自另一上下文的不恰当信息。为此，我们提出了AgentCIBench，一个将这一风险转化为可执行、可确定性评分场景的评估框架。我们针对CUA中的三种常见失败模式：视觉同址，即代理提取了UI中位于任务目标旁边的被禁止项目；任务歧义导致的过度分享，即代理在响应不够明确的提示时倾倒大量个人状态信息；以及接收者错位，即代理将内容发送给了不适宜的收件人。我们评估了15个前沿代理，发现了惊人的高失败率：15个代理中有11个发生信息泄露。

    arXiv:2606.23189v2 Announce Type: replace-cross  Abstract: Computer-use agents (CUAs) now act on a user's behalf across personal applications such as email, calendars, and to-do lists. This cross-application access is useful, but it also creates a privacy risk that has been largely overlooked: when an agent works in one context, it can pull in information from another that is inappropriate in that context. Hence, we introduce AgentCIBench, an evaluation harness that turns this risk into executable, deterministically scored scenarios. We target three common failure modes in CUAs: visual co-location, where the agent pulls in prohibited items that sit next to the task target in the UI; task-ambiguity overshare, where the agent dumps dense personal state in response to an under-specified prompt; and recipient misalignment, where the agent sends content to an addressee for whom it is inappropriate. We evaluate 15 frontier agents and find a surprisingly high failure rate: 11 of 15 leak on mo
    
[^86]: 当上下文误导时：作为大语言模型连贯性错觉度量的惊奇度、能量与注意力熵

    When Context Misleads: Surprisal, Energy and Attention Entropy as Metrics of Coherence Illusions in LLMs

    [https://arxiv.org/abs/2606.21203](https://arxiv.org/abs/2606.21203)

    该研究首次发现语言模型会像人类一样陷入连贯性错觉，并提出用惊奇度、注意力熵和联想记忆能量三种指标量化这一现象，同时揭示了模型处理语篇连贯性时存在共享的注意力机制。

    

    心理语言学研究表明，人类读者会陷入连贯性错觉：一段不连贯的语篇可能仅仅因为前文中的干扰项与后续内容相匹配而显得连贯。我们研究了荷兰语语言模型（6个单语模型和4个多语模型）在通过“再次”和“也”等词回指先前上下文的文本上是否表现出相同的行为。首先，我们发现关键词位上的惊奇度与人类的可接受度判断和眼动追踪数据相吻合。模型对不连贯的后续内容表现出更高的惊讶度，但先前上下文中匹配的干扰项会降低这种惊奇度。其次，注意力熵识别出在连贯与不连贯条件下表现不同的注意力头。我们发现对这些注意力头进行消融会在不同实验之间产生迁移效应，表明存在共享的底层机制。第三，我们引入联想记忆文献中的能量概念作为量化语篇连贯性的度量指标。综合来看，我们的结果表明……

    arXiv:2606.21203v2 Announce Type: replace  Abstract: Psycholinguistics studies show that human readers fall for coherence illusions: an incoherent discourse can seem coherent simply because a distractor matches what comes next. We investigate whether Dutch language models (6 monolingual and 4 multilingual) show the same behavior on texts that link back to earlier context with words such as 'again' and 'too'. First, we find that surprisal at the critical word tracks human acceptability judgments and eye-tracking data. Models are more surprised by incoherent continuations, but a matching distractor in the prior context reduces this surprisal. Second, attention entropy identifies heads that behave differently under coherence vs. incoherence. We find that ablating these heads shows transfer effects across experiments, suggesting a shared mechanism. Third, we introduce energy from the associative-memory literature as a metric to quantify discourse coherence. Taken together, our results show
    
[^87]: GRACE-DS：数据科学中的受保护奖励引导智能体纠正环境

    GRACE-DS: a Guarded Reward-guided Agent Correction Environment in Data Science

    [https://arxiv.org/abs/2606.16000](https://arxiv.org/abs/2606.16000)

    GRACE-DS是一个用于LLM驱动AutoML智能体部署前评估的隔离环境，通过隐藏的可执行验证器从预测性能、泄漏规避、可复现性等多维度进行评估，其中灵活迭代交互机制的表现优于单次生成等基线方法。

    

    我们介绍了GRACE-DS，一个面向数据科学的受保护奖励引导智能体纠正环境，用于对基于LLM的AutoML智能体进行部署前评估。GRACE-DS是一套在隔离环境中的评估指标，可应用于特定组织专属的表格机器学习任务。它让智能体经历真实的工作流程阶段，从规划和数据检查，到特征工程、模型开发、验证和代码修复，直至最终提交；同时，隐藏的可执行验证器不仅衡量最终的预测性能，还评估泄漏规避、可复现性、协议有效性、纠正行为以及奖励对齐情况。在所有结构化机制中，最强的灵活迭代交互（我们的方法）相比单次生成、非结构化交互和基于重启的基线方法，实现了更高的端到端归一化隐藏测试质量，同时还提升了协议有效的完成率。

    arXiv:2606.16000v3 Announce Type: replace  Abstract: We introduce GRACE-DS, a Guarded Reward-guided Agent Correction Environment in Data Science for pre-deployment evaluation of LLM-powered AutoML agents. GRACE-DS is a set of evaluation metrics in an isolated environment that can be applied to tabular ML tasks specific to a particular organization. It exposes agents to realistic workflow stages, from planning and data inspection through feature engineering, model development, validation, and code repair to final submission, while hidden executable validators measure not only final predictive performance but also leakage avoidance, reproducibility, protocol validity, correction behavior, and reward alignment. The strongest structured regime, flexible iterative interaction (our approach), achieves higher end-to-end normalized hidden-test quality than single-shot generation, unstructured interaction, and restart-based baselines, while also improving protocol-valid completion. Validated ac
    
[^88]: 关于大语言模型条件控制中有效性与流畅性的权衡：一项系统性研究

    On The Effectiveness-Fluency Trade-Off In LLM Conditioning: A Systematic Study

    [https://arxiv.org/abs/2606.12234](https://arxiv.org/abs/2606.12234)

    该论文系统研究了LLM条件控制方法，发现高效的激活引导方法往往以流畅性大幅下降为代价，且在指令微调模型上效果远差于基础模型，而提示和监督微调适合概念注入但不擅长概念移除。

    

    控制大语言模型的输出是其可靠部署所面临的核心挑战，然而对其中涉及的权衡，目前仍缺乏清晰的认识。当前的条件控制方法在评估时往往只狭隘地关注其注入或移除目标概念的有效性，而忽视了生成质量。我们系统性地研究了多种条件控制方法在概念注入和概念移除两种场景下的表现。我们发现，高效的引导方法在实现条件控制的同时，往往以流畅性的急剧下降为代价。此外，我们还发现了一个此前被忽视的与训练范式的关键交互作用：激活引导方法在经过指令微调的模型上的效果远不如在对应的基础模型上。另一方面，简单的提示方法和完整的监督微调对于概念注入是可行的选择，但在概念移除方面表现不佳。最后，低成本的……

    arXiv:2606.12234v2 Announce Type: replace  Abstract: Controlling the output of Large Language Models (LLMs) is a central challenge for their reliable deployment, yet a clear understanding of the involved trade-offs remains elusive. Current approaches to conditioning are often evaluated with a narrow focus on their effectiveness at injecting or removing a target concept, neglecting generation quality. We systematically investigate a range of conditioning methods in both injection and removal scenarios. We find that efficient steering methods frequently achieve conditioning at a steep cost to fluency. Furthermore, we identify a critical yet previously overlooked interaction with the training paradigm: activation steering methods are far less effective on instruction-tuned models than on their base counterparts. Simple prompting and full-fledged supervised fine-tuning, on the other hand, are viable options for concept injection, but are not as good at concept removal. Finally, cheaply com
    
[^89]: LC-QAT：基于线性约束向量量化的大语言模型数据高效2比特量化感知训练

    LC-QAT: Data-Efficient 2-Bit QAT for LLMs via Linear-Constrained Vector Quantization

    [https://arxiv.org/abs/2606.10531](https://arxiv.org/abs/2606.10531)

    LC-QAT通过线性约束向量量化提供高质量PTQ初始化并实现完全可微的端到端优化，是一个数据高效的2比特大语言模型量化感知训练框架。

    

    量化感知训练（QAT）对于极低比特的大语言模型（LLMs）至关重要。当前的QAT方法主要基于标量量化（SQ），虽然能够实现高效优化，但在2比特精度下会出现严重的性能下降。另一方面，向量量化（VQ）提供了显著更高的表示能力，但其离散码本查找机制阻碍了端到端训练。我们提出了LC-QAT，一个2比特仅权重的VQ-QAT框架，它通过在离散向量上学习到的仿射映射来表示量化权重，从而产生高质量的PTQ初始化，并使训练前向传播过程中无需显式码本查找即可实现完全可微的端到端优化。这种强大的训练后初始化使LC-QAT具有极高的数据效率。在多种大语言模型上的实验表明，LC-QAT在使用仅（原文此处内容截断）的情况下持续优于最先进的QAT方法。

    arXiv:2606.10531v3 Announce Type: replace  Abstract: Quantization-aware training (QAT) is essential for extremely low-bit large language models (LLMs). Current QAT methods are mainly based on scalar quantization (SQ), which enables efficient optimization but suffers from severe performance degradation at 2-bit precision. On the other hand, vector quantization (VQ) provides substantially higher representational capacity, but its discrete codebook lookup prevents end-to-end training. We propose LC-QAT, a 2-bit weight-only VQ-QAT framework that represents quantized weights via a learned affine mapping over discrete vectors, which yields a high-quality PTQ initialization and enables fully differentiable end-to-end optimization without explicit codebook lookup in the training forward pass. This strong post-training initialization makes LC-QAT highly data-efficient. Experiments across diverse LLMs demonstrate that LC-QAT consistently outperforms state-of-the-art QAT methods while using only 
    
[^90]: 心理健康对话中的专家级危机检测

    Expert-Level Crisis Detection in Mental Health Conversations

    [https://arxiv.org/abs/2606.10380](https://arxiv.org/abs/2606.10380)

    该论文提出了临床医生标注的CRADLE-Dialogue基准数据集以及“警报-确认”评估协议，用于解决多轮心理健康对话中轮次级危机检测的难题，使模型能够捕捉随对话演进的风险信号并支持早期干预。

    

    现实世界的危机干预本质上是对话式的，然而现有研究主要集中于静态文本。当应用于多轮对话时，当前模型表现出显著的性能下降，难以追踪随着上下文演变而出现的风险信号。为了弥补这一空白，我们推出了CRADLE-Dialogue，这是一个由临床医生标注的、用于对话环境中轮次级危机检测的基准数据集。该数据集包含600段对话，针对基于临床的风险（包括自杀意念、自残和虐待儿童）进行了多标签标注，并区分了过去风险与正在发生的风险。我们进一步提出了“警报-确认”评估协议，将早期预警信号与特定危机变得明确可识别的对话轮次区分开来，体现了在风险变得明显之前进行干预的临床需求。实验表明，识别风险何时出现远比识别风险本身困难得多。

    arXiv:2606.10380v2 Announce Type: replace  Abstract: Real-world crisis intervention is inherently conversational, yet existing research largely focuses on static texts. When applied to multi-turn dialogues, current models exhibit significant performance degradation, struggling to track risk signals that emerge as context evolves. To address this gap, we introduce CRADLE-Dialogue, a clinician-annotated benchmark for turn-level crisis detection in conversational settings. The dataset features 600 dialogues with multi-label annotations across clinically grounded risks, including suicide ideation, self-harm, and child abuse, distinguishing past from ongoing risk. We further propose an Alert-Confirm evaluation protocol that distinguishes early warning signals (Alert) from turns where a specific crisis becomes explicitly identifiable (Confirm), reflecting the clinical need to intervene before risk becomes explicit. Experiments show that identifying when risk emerges is much harder than recog
    
[^91]: 大语言模型中用于动态实体追踪的检索条件化重绑定电路

    A retrieval conditioned rebinding circuit for dynamic entity tracking in large language models

    [https://arxiv.org/abs/2606.08644](https://arxiv.org/abs/2606.08644)

    本研究通过因果干预识别出大语言模型中一种检索条件化的重绑定电路机制，用于动态实体属性追踪，并发现Gemma模型在注意力头的查询/键子空间中表达绑定信息，而Llama模型则主要在键向量中携带绑定信息。

    

    大语言模型为了正确理解上下文并检索相关信息，必须将实体与其属性进行绑定，并在状态变化时更新这些绑定。我们分析了大语言模型在动态状态追踪中如何实现这一绑定过程。通过因果干预方法，我们识别出一种检索条件化的重绑定机制——这是一个紧凑的注意力头电路，它负责传播绑定信息，并在查询实体时利用更新后的绑定来检索相应的属性。在Gemma和Llama模型中，该电路都支持重绑定行为，但该机制的表示特征在不同模型家族之间存在差异。在Gemma模型中，绑定签名在相关注意力头的查询/键子空间中清晰表达；而在Llama模型中，绑定信息主要携带于键向量中。总体而言，我们的研究结果揭示了一种可解释的、依赖上下文的状态追踪机制。

    arXiv:2606.08644v2 Announce Type: replace  Abstract: To interpret context correctly and retrieve relevant information, large language models must bind entities to their attributes and update these bindings as state changes. We analyze how LLMs implement this binding process in a dynamic state tracking. Using causal interventions, we identify a retrieval conditioned rebinding mechanism, a compact attention head circuit that propagated binding information and when the entity is queried, uses the updated binding to retrieve the corresponding attribute. Across Gemma and Llama models, this circuit supports rebinding behavior, but the representational signature of the mechanism differs across model families. In Gemma models, the binding signature is clearly expressed in the query/key subspaces of the relevant attention heads, whereas in Llama models, the binding information is carried primarily in key vectors. Overall, our results reveal an interpretable mechanism for context dependent state
    
[^92]: SAEExplainer：利用激活引导的偏好优化解释SAE特征

    SAEExplainer: Interpreting SAE Features with Activation-Guided Preference Optimization

    [https://arxiv.org/abs/2606.08496](https://arxiv.org/abs/2606.08496)

    提出SAEExplainer训练框架，以激活分数作为客观奖励信号，通过两轮迭代优化实现模型自我纠正与持续改进，显著减少SAE特征解释中的幻觉并强化因果触发模式。

    

    尽管稀疏自编码器（SAE）通过将稠密表示分解为稀疏特征，缓解了大型语言模型（LLM）的不透明性问题，但解释这些特征仍然是一个核心挑战。然而，目前的解释方法通常运行在开环范式下，未能利用机制性反馈进行进一步优化。在本文中，我们提出了SAEExplainer，这是一个利用激活分数作为客观奖励信号来训练模型进行自我纠正和迭代引导的训练框架。通过在两轮优化过程中迭代地验证和纠正基础解释，SAEExplainer实现了其解释能力的持续提升。这一机制显著减少了解释幻觉，并强化了因果触发模式。大量实验表明，我们的方法在大多数指标上优于现有基线。

    arXiv:2606.08496v2 Announce Type: replace  Abstract: Although Sparse Autoencoders (SAEs) have mitigated the opacity of large language models (LLMs) by decomposing dense representations into sparse features, explaining these features still remains a central challenge. Current explanation methods, however, typically operate within an open-loop paradigm, failing to leverage mechanistic feedback for further refinement. In this paper, we propose SAEExplainer, a training framework that utilizes activation scores as an objective reward signal to train the model for self-correction and iterative bootstrapping. By iteratively verifying and correcting foundational explanations through a two-round optimization process, SAEExplainer achieves continuous improvement in its explanatory capabilities. This mechanism significantly reduces explanation hallucinations and reinforces causal triggering patterns. Extensive experiments demonstrate our approach improves upon established baselines across most me
    
[^93]: UrduMMLU：一个面向乌尔都语理解的大规模多任务基准

    UrduMMLU: A Massive Multitask Benchmark for Urdu Language Understanding

    [https://arxiv.org/abs/2606.07167](https://arxiv.org/abs/2606.07167)

    UrduMMLU是一个基于乌尔都语本地教育资源构建的大规模多任务语言理解基准，包含26,389道多选题，覆盖26个学科和五个领域，填补了乌尔都语缺乏原生MMLU风格评测基准的空白，并系统评估了30个大语言模型在英语和乌尔都语提示下的表现。

    

    有意义的多语言评估必须在目标语言和教育背景下对模型进行测试。乌尔都语拥有超过2.3亿使用者，但一直缺乏一个基于本地教育资源构建的广泛MMLU风格基准。我们提出了UrduMMLU，这是一个包含26,389道乌尔都语多选题的基准，涵盖26个学科和五个领域，数据来源于本地乌尔都语多选题库和公开考试PDF文件。与基于翻译的基准不同，UrduMMLU将学术科目与乌尔都语及地区教育特有内容相结合。我们通过双人标注并辅以严格的共识过滤，对来源于考试的部分进行了标注。我们在英语和乌尔都语提示下评估了30个大语言模型，共获得60个零样本评估结果，并进一步在两种提示语言的多种少样本设置下评估了四个开源大语言模型。其中Gemini-3.5-Flash表现最佳，准确率分别达到90.23%和90.45%，而其他模型均未超过85%。最强的开源模型落后了7.78个百分点。

    arXiv:2606.07167v2 Announce Type: replace  Abstract: Meaningful multilingual evaluation must test models in the target language and educational context. Urdu, spoken by more than 230 million people, lacks a broad MMLU-style benchmark built from native educational sources. We introduce UrduMMLU, a benchmark of 26,389 Urdu MCQs across 26 subjects and five domains, collected from native Urdu MCQ banks and public examination PDFs. Unlike translation-based benchmarks, UrduMMLU combines academic subjects with content specific to Urdu and regional education. We label the exam-derived portion through dual human annotation with strict consensus filtering. We evaluate 30 LLMs under English and Urdu prompts, yielding 60 zero-shot evaluations, and further evaluate four open-source LLMs under multiple few-shot settings across both prompt languages. Gemini-3.5-Flash performs best, reaching 90.23% and 90.45% accuracy, while no other model exceeds 85%. The strongest open-source model trails by 7.78 an
    
[^94]: ReasoningFlow：用于理解大语言模型推理轨迹的话语结构框架

    ReasoningFlow: Discourse Structures for Understanding LLM Reasoning Traces

    [https://arxiv.org/abs/2606.05402](https://arxiv.org/abs/2606.05402)

    该论文提出了ReasoningFlow框架，将大型推理模型的推理轨迹建模为细粒度有向无环图，从而揭示出不同模型尽管训练背景各异，却展现出结构相似的推理轨迹这一重要发现。

    

    大型推理模型（LRMs）生成的推理轨迹具有非线性结构，例如回溯和自我修正，这使得对推理过程的评估和监控变得复杂。我们提出了ReasoningFlow，这是一个将大型推理模型推理轨迹的话语结构捕获为细粒度有向无环图（DAGs）的框架。我们通过对31条推理轨迹（2,100个步骤）的细致人工标注，开发并验证了我们的标注模式，实现了较高的标注者间一致性，随后扩展至对1,260条推理轨迹（247,700个步骤）的自动标注，涵盖三项任务（数学、科学、论证）和五个模型（Qwen2.5-32B-Inst、QwQ-32B、DeepSeek-V3、DeepSeek-R1、GPT-oss-120B）。通过分析ReasoningFlow图，我们发现：（1）尽管各大型推理模型由不同的基础模型训练而来，且后训练数据可能不重叠，但它们表现出结构相似的推理轨迹。（2）ReasoningFlow揭示了多样的细粒度推理……（原文摘要在此处截断）

    arXiv:2606.05402v2 Announce Type: replace  Abstract: Large reasoning models (LRMs) produce reasoning traces with non-linear structures, such as backtracking and self-correction, that complicate the evaluation and monitoring of the reasoning process. We introduce ReasoningFlow, a framework that captures the discourse structures of LRM reasoning traces into fine-grained directed acyclic graphs (DAGs). We develop and validate our annotation schema through careful manual annotation of 31 traces (2.1k steps), achieving high inter-annotator agreement, then scale to automatic annotation of 1,260 traces (247.7k steps) spanning three tasks (math, science, argumentation) and five models (Qwen2.5-32B-Inst, QwQ-32B, DeepSeek-V3, DeepSeek-R1, GPT-oss-120B). By analyzing ReasoningFlow graphs, we find: (1) LRMs exhibit structurally similar traces, despite being trained from different base models and potentially non-overlapping post-training data. (2) ReasoningFlow reveals diverse fine-grained reasoni
    
[^95]: 道德语义在机器翻译中得以存续：来自道德基础语料库的跨语言证据

    Moral Semantics Survive Machine Translation: Cross-Lingual Evidence from Moral Foundations Corpora

    [https://arxiv.org/abs/2605.22660](https://arxiv.org/abs/2605.22660)

    尽管在俚语和文化负载表达上存在翻译缺陷，基于大语言模型的机器翻译仍能充分保留道德语义线索，使英语标注的道德语料库可用于跨语言（以波兰语为例）的道德价值观自动分类。

    

    道德语言微妙且因文化而异，这使得跨语言的忠实翻译变得困难。习语表达、俚语和文化引用会引入难以避免的翻译瑕疵。然而，自动化的道德价值观分类依赖于特定语言的标注语料库，而这些语料库几乎仅存在于英语中。我们以波兰语为测试案例，研究了基于大语言模型（LLM）的翻译能否弥合这一差距。我们使用了约5万条来自多样化主题的带有道德标注的社交媒体帖子，并应用了一套系统的四种方法验证流程：LaBSE跨语言嵌入相似度、中心化核对齐（CKA）、LLM作为裁判的评估，以及深度学习分类器等效性测试。我们表明，尽管在处理俚语、粗俗语言和文化负载表达方面存在不足，直接翻译仍能充分保留微妙的道德线索，足以被跨语言机器学习所利用——平均余弦相似度……（摘要在此处截断）

    arXiv:2605.22660v4 Announce Type: replace  Abstract: Moral language is subtle and culturally variable, making it difficult to translate faithfully across languages. Idiomatic expressions, slang, and cultural references introduce hard-to-avoid translation artifacts. Yet automated moral values classification depends on language-specific annotated corpora that exist almost exclusively in English.   We investigate whether LLM-based translation can bridge this gap, taking Polish as a test case. Using ~50k morally-annotated social media posts from a diverse range of topics, we apply a principled four-method validation pipeline: LaBSE cross-lingual embedding similarity, Centered Kernel Alignment (CKA), LLM-as-judge evaluation, and deep learning classifier parity tests. We show that despite shortcomings in handling slang, vulgarity, and culturally-loaded expressions, direct translation preserves subtle moral cues well enough to be harvested by cross-lingual machine learning - with a mean cosin
    
[^96]: 基于强化学习的Text-to-SPARQL生成：一种基于GRPO的DBLP方法

    Text-to-SPARQL Generation with Reinforcement Learning: A GRPO-based Approach on DBLP

    [https://arxiv.org/abs/2605.20066](https://arxiv.org/abs/2605.20066)

    本研究证明，基于结果奖励的GRPO强化学习能够在DBLP-QuAD上训练小型的Qwen3-1.7B模型实现零样本Text-to-SPARQL查询生成，无需依赖大型模型或完整的黄金查询标注监督。

    

    知识图谱问答旨在将自然语言问题转化为可在知识图谱上执行的可执行查询，但现有方法通常依赖于大型模型或以黄金查询标注形式提供的全监督。本研究探讨基于结果奖励的强化学习能否训练一个小型指令微调语言模型，在学术领域实现零样本的Text-to-SPARQL生成。该研究在DBLP-QuAD数据集上，将组相对策略优化（GRPO）应用于Qwen3-1.7B模型，所使用的提示将自然语言问题与关于实体和关系的符号提示相结合。训练依赖于执行反馈、结构约束和答案级奖励，并额外引入了一个融合基于黄金查询的奖励塑形的变体。所得模型在答案级准确性、可执行性等指标上与未修改的零样本基线以及监督式DoRA微调基线进行了比较。

    arXiv:2605.20066v2 Announce Type: replace  Abstract: Knowledge graph question answering seeks to translate natural language questions into executable queries over knowledge graphs, but existing approaches often rely on large models or full supervision in the form of gold query annotations. This study examines whether reinforcement learning with outcome-based rewards can train a small instruction-tuned language model to perform zero-shot Text-to-SPARQL generation in the scholarly domain. Group-Relative Policy Optimization (GRPO) is applied to the Qwen3-1.7B model on DBLP-QuAD, using prompts that combine natural language questions with symbolic hints about entities and relations. Training relies on execution feedback, structural constraints, and answer-level rewards, with an additional variant that incorporates gold-query-based shaping. The resulting models are compared to the unmodified zero-shot baseline and to a supervised DoRA-finetuned baseline across answer-level accuracy, executio
    
[^97]: ShadowPEFT：用于参数高效微调的影子网络

    ShadowPEFT: Shadow Network for Parameter-Efficient Fine-Tuning

    [https://arxiv.org/abs/2604.19254](https://arxiv.org/abs/2604.19254)

    ShadowPEFT将参数高效微调整合为一个模块化的影子网络，其影子模型可作为独立预测器进行推理而无需运行基础模型，在文本和图像的生成与理解任务上达到或超越现有方法。

    

    流行的低秩参数高效微调（PEFT）方法将适配表示为对选定骨干权重的独立更新，而不维护一个在深度上被更新和复用的显式任务特定状态。这些更新在推理时也需要骨干网络参与，因此无法作为独立的预测器运行。我们提出ShadowPEFT，它将可训练的适配整合到一个模块化的影子组件中，该组件以一个紧凑的影子模型和轻量级的Transformer层特定耦合模块为中心。一个持久的影子状态对冻结的骨干表示进行细化，并以交互的方式从中进行更新。由于影子模型被训练为一个完整的预测器，它可以被分离出来进行仅影子推理，而无需执行基础模型，并且可以从预训练模型进行初始化。在文本和图像的生成与理解基准上的实验表明，ShadowPEFT达到或超越了……

    arXiv:2604.19254v2 Announce Type: replace  Abstract: Popular low-rank parameter-efficient fine-tuning (PEFT) methods represent adaptation as separate updates to selected backbone weights, without maintaining an explicit task-specific state that is updated and reused across depth. These updates also require the backbone at inference and therefore cannot operate as standalone predictors. We propose ShadowPEFT, which consolidates trainable adaptation into a modular shadow component centered on a compact shadow model and lightweight Transformer layer-specific coupling modules. A persistent shadow state refines the frozen backbone representations and is updated from them in an interactive manner. Because the shadow model is trained as a complete predictor, it can be detached for shadow-only inference without executing the base model and can be initialized from a pretrained model. Experiments on text and image generation and understanding benchmarks show that ShadowPEFT matches or outperform
    
[^98]: SaFeR-Steer：通过合成引导与反馈动态演化多轮多模态大语言模型

    SaFeR-Steer: Evolving Multi-Turn MLLMs via Synthetic Bootstrapping and Feedback Dynamics

    [https://arxiv.org/abs/2604.16358](https://arxiv.org/abs/2604.16358)

    提出渐进式多轮安全对齐框架SaFeR-Steer，通过分阶段合成数据引导与导师介入的GRPO强化学习，并引入轨迹一致求和奖励（TCSR）机制，同时发布多轮多模态安全数据集STEER，以弥合多轮对话场景下多模态大模型训练与部署之间的安全对齐差距。

    

    多模态大语言模型（MLLM）越来越多地被部署在多轮对话场景中，攻击者可以通过不断演化的视觉-文本历史逐步升级不安全意图，并利用长上下文安全衰减的漏洞。然而，安全对齐目前仍主要依赖单轮数据和固定模板对话，导致训练与部署之间存在不匹配。为弥合这一差距，我们提出了SaFeR-Steer，这是一个渐进式多轮对齐框架，将分阶段合成引导与导师介入的GRPO相结合，在自适应的、在线策略攻击下训练单一学生模型。我们还引入了轨迹一致求和奖励，该机制聚合各轮次奖励的历史最小值和平均值，使任何低质量轮次都会影响轨迹层面的整体回报。I. 数据集：我们发布了STEER，一个多轮多模态安全数据集，包含STEER-SFT（12,934条）、STEER-RL（2,000条）和STEER-Bench（3,227条）对话，覆盖1-10轮对话。II. 实验：从Q（注：原始摘要在此处不完整）

    arXiv:2604.16358v3 Announce Type: replace-cross  Abstract: MLLMs are increasingly deployed in multi-turn settings, where attackers can escalate unsafe intent through the evolving visual-text history and exploit long-context safety decay. Yet safety alignment is still dominated by single-turn data and fixed-template dialogues, leaving a mismatch between training and deployment. To bridge this gap, we propose SaFeR-Steer, a progressive multi-turn alignment framework that combines staged synthetic bootstrapping with tutor-in-the-loop GRPO to train a single student under adaptive, on-policy attacks. We also introduce Trajectory-Consistent Summative Reward (TCSR), which aggregates the historical minimum and average of turn rewards so that any low-quality turn affects the trajectory-level return. I. Dataset. We release STEER, a multi-turn multimodal safety dataset with STEER-SFT (12,934), STEER-RL (2,000), and STEER-Bench (3,227) dialogues spanning 1-10 turns. II. Experiment. Starting from Q
    
[^99]: 翻译腔作为对翻译任务难度的理性回应

    Translationese as a Rational Response to Translation Task Difficulty

    [https://arxiv.org/abs/2603.12050](https://arxiv.org/abs/2603.12050)

    该论文提出翻译腔是译者对翻译任务认知难度的理性回应，并证明基于大语言模型惊讶度的信息论任务难度指标能够有效预测文本的翻译腔程度。

    

    翻译文本与目标语言中原创的可比文本之间存在系统性差异。解释这一现象（通常称为“翻译腔”）仍然是一个开放性挑战。翻译腔曾被归因于产出倾向（如干扰、简化）、社会文化变量以及语言对效应，但目前缺乏统一的解释性理论。我们研究这样一个假设：翻译腔是对翻译任务固有认知负荷的一种回应。我们检验是否可以从可量化的翻译任务难度测量指标来预测可观测的翻译腔。翻译腔通过自动分类器测量的片段级翻译概率（translatedness分数）来衡量。翻译任务难度包含源文本和跨语言迁移两个组成部分，它们通过基于大语言模型（LLM）惊讶度的信息论指标来捕捉……

    arXiv:2603.12050v3 Announce Type: replace  Abstract: Translated texts exhibit systematic differences from comparable texts originally written in the target language. Explaining this phenomenon, commonly known as translationese, remains an open challenge. Translationese has been attributed to production tendencies (e.g. interference, simplification), socio-cultural variables, and language-pair effects, yet a unified explanatory account is lacking. We investigate the hypothesis that translationese is a response to the cognitive load inherent in the translation task. We test whether observable translationese can be predicted from quantifiable measures of translation task difficulty. Translationese is measured as a segment-level probability of being a translation produced by an automatic classifier (translatedness score). Translation task difficulty includes source-text and cross-lingual transfer components. They are captured by information-theoretic metrics based on LLM surprisal and by e
    
[^100]: Countdown-Code：用于研究RLVR中奖励破解的出现与泛化的测试平台

    Countdown-Code: A Testbed for Studying The Emergence and Generalization of Reward Hacking in RLVR

    [https://arxiv.org/abs/2603.07084](https://arxiv.org/abs/2603.07084)

    本文提出Countdown-Code测试平台，通过代理奖励与真实奖励的清晰分离来精确测量RLVR中的奖励破解现象，并发现SFT数据中仅1%的奖励破解轨迹污染就足以让模型无意中习得这种失对齐行为。

    

    奖励破解是模型对齐失效的一种形式，即模型过度优化代理奖励而没有真正解决底层任务。精确测量奖励破解的发生仍然具有挑战性，因为真实的任务奖励往往计算成本高昂或无法计算。我们提出了Countdown-Code，这是一个极简环境，模型在其中既可以解决数学推理任务，也可以操纵测试工具。这种双重访问设计在代理奖励（测试通过/失败）与真实奖励（数学正确性）之间建立了清晰的分离，从而能够准确测量奖励破解的发生率。利用该环境，我们研究了开源权重大语言模型中的奖励破解行为，发现当即使只有一小部分奖励破解轨迹泄漏到训练数据中时，模型也会在监督微调（SFT）过程中无意中习得此类行为。在蒸馏SFT数据中仅需1%的污染，就足以使模型内化……

    arXiv:2603.07084v3 Announce Type: replace-cross  Abstract: Reward hacking is a form of misalignment in which models overoptimize proxy rewards without genuinely solving the underlying task. Precisely measuring reward hacking occurrence remains challenging because true task rewards are often expensive or impossible to compute. We introduce Countdown-Code, a minimal environment where models can both solve a mathematical reasoning task and manipulate the test harness. This dual-access design creates a clean separation between proxy rewards (test pass/fail) and true rewards (mathematical correctness), enabling accurate measurement of reward-hacking rates. Using this environment, we study reward hacking in open-weight LLMs and find that such behaviors can be unintentionally learned during supervised fine-tuning (SFT) when even a small fraction of reward-hacking trajectories leak into training data. As little as 1\% contamination in distillation SFT data is sufficient for models to internali
    
[^101]: 测量大语言模型指令中的语用影响力

    Measuring Pragmatic Influence in Large Language Model Instructions

    [https://arxiv.org/abs/2602.21223](https://arxiv.org/abs/2602.21223)

    该论文提出了一个系统测量大语言模型指令中语用框架化影响力的框架，通过指令-框架分解、涵盖400个实例和13种策略的分类法以及基于优先级的测量方法，揭示“如何提问”而非“提问内容”对模型行为的影响。

    

    重要的不仅是我们要求大语言模型（LLM）做什么，还有我们如何要求它们。诸如“这很紧急”或“作为你的主管”这样的短语可以在不改变任务内容的情况下改变模型行为。我们将这种效应研究为语用框架化，即塑造指令解释而非任务规范的上下文线索。虽然先前的工作利用这些线索进行提示优化，或将其作为安全漏洞进行探测，但语用框架化本身作为指令遵循中受控测量目标的关注却相对较少。为了支持将其作为可测量属性进行系统研究，我们引入了一个结合三个组件的框架：将框架化上下文与任务规范分离的指令-框架分解；将400个框架化实例组织为4个机制集群下13种策略的分类法；以及通过观测……量化影响力的基于优先级的测量方法。

    arXiv:2602.21223v2 Announce Type: replace  Abstract: It is not only what we ask large language models (LLMs) to do that matters, but also how we ask them. Phrases like ``This is urgent'' or ``As your supervisor'' can shift model behavior without altering task content. We study this effect as pragmatic framing, contextual cues that shape directive interpretation rather than task specification. While prior work exploits such cues for prompt optimization or probes them as security vulnerabilities, pragmatic framing itself has received comparatively little attention as a target of controlled measurement in instruction following. To support its systematic study as a measurable property, we introduce a framework that combines three components: directive-framing decomposition separating framing context from task specification; a taxonomy organizing 400 instantiations of framing into 13 strategies across 4 mechanism clusters; and priority-based measurement that quantifies influence through obs
    
[^102]: 英国多语言英语使用者在AI驱动的语音认知筛查中的假阳性偏差

    False positive bias in AI-powered speech-based cognitive screening for multilingual English speakers in the UK

    [https://arxiv.org/abs/2602.13047](https://arxiv.org/abs/2602.13047)

    该研究通过对1,395名参与者、超过263小时语音数据的分析，首次发现尽管语音识别准确率在各语言群体间无显著差异，但AI认知筛查的下游模型对英国多语言英语使用者存在系统性的假阳性偏差，凸显了认知筛查公平性评估的重要性。

    

    会话语音能够揭示认知衰退的早期迹象，包括痴呆症和轻度认知障碍（MCI）。AI模型在基于语音的筛查方面展现出前景，但大多数研究聚焦于单语群体。在英国，痴呆症预计在黑人和亚裔社区中增长最快，而这些社区中多语言现象普遍，因此公平性评估至关重要。我们招募了1,395名参与者（包括谢菲尔德/布拉德福德的英语单语者和多语者），并通过CognoMemory智能体收集了超过263小时的语音数据。多语言参与者在说英语的同时还使用索马里语、中文或南亚语言（印地语、乌尔都语、旁遮普语、米尔普里语、阿拉伯语）。我们评估了自动语音识别系统（Whisper、Wav2Vec 2.0、NeMo）以及用于认知分类和MMSE回归的下游AI模型。ASR准确率在各群体间未显示出显著差异。然而，下游模型表现出系统性差异：多语言使用者……

    arXiv:2602.13047v2 Announce Type: replace  Abstract: Conversational speech reveals early signs of cognitive decline, including dementia and mild cognitive impairment (MCI). AI models show promise for speech-based screening, yet most research focuses on monolingual groups. In the UK, dementia is projected to rise fastest among Black and Asian communities, where multilingualism is common, making equity assessment critical. We recruited 1,395 participants (monolingual English speakers and multilingual speakers from Sheffield/Bradford) and collected over 263 hours of speech via the CognoMemory agent. Multilingual participants spoke English alongside Somali, Chinese, or South Asian languages (Hindi, Urdu, Punjabi, Mirpuri, Arabic). We evaluated ASR (Whisper, Wav2Vec 2.0, NeMo) and downstream AI models for cognitive classification and MMSE regression. ASR accuracy showed no significant differences across groups. However, downstream models exhibited systematic disparities: multilingual speake
    
[^103]: PACIFIC：大语言模型能否辨别影响你偏好的心理测量特质？基于人格驱动的LLM偏好对齐

    PACIFIC: Can LLMs Discern the Psychometric Traits Influencing Your Preferences? Personality-Driven Preference Alignment in LLMs

    [https://arxiv.org/abs/2602.07181](https://arxiv.org/abs/2602.07181)

    提出PACIFIC框架，将大五人格（OCEAN）特质作为潜在信号来组织和推理用户偏好历史，从而在偏好信号嘈杂、不完整的情况下实现更可靠的人格驱动LLM偏好对齐。

    

    用户偏好日益被用于个性化大语言模型（LLM）的回复，然而如何可靠地利用偏好信号仍缺乏充分研究。在实践中，偏好信号可能是嘈杂的、不完整的，甚至是误导性的，如果简单地直接应用，反而会降低回答质量。受“稳定的人格特质塑造日常偏好”这一观察的启发，我们提出了PACIFIC（通过五因素身份表征进行偏好对齐的偏好推理框架），这是一个由人格驱动的偏好对齐框架，它将大五人格（OCEAN）特质作为一种有原则的“潜在”信号，用于组织和推理用户偏好历史。为了系统地评估该框架，我们构建了一个基于心理测量学的数据集，包含1,200个偏好-查询对，涵盖多个领域（如旅游、电影和教育），并全面覆盖了大五人格特质的高低方向。大量实验表明……

    arXiv:2602.07181v4 Announce Type: replace  Abstract: User preferences are increasingly used to personalize Large Language Model (LLM) responses, yet reliably leveraging preference signals remains under-explored. In practice, preferences can be noisy, incomplete, or even misleading, which can degrade answer quality when applied naively. Motivated by the observation that stable personality traits shape everyday preferences, we introduce PACIFIC (Preference Alignment for Choices Inference via Five-factor Identity Characterization), a personality-driven preference alignment framework that uses Big-Five (OCEAN) traits as a principled "latent" signal for organizing and reasoning over user preference history. To systematically evaluate this framework, we construct a psychometrics-based dataset containing 1,200 preference-query pairs spanning diverse domains (e.g., travel, movies, and education), with comprehensive coverage of high and low Big-Five trait directions. Extensive experiments show 
    
[^104]: 基于约束二值优化的块删除大语言模型压缩

    LLM Compression by Block Removal with Constrained Binary Optimization

    [https://arxiv.org/abs/2602.00161](https://arxiv.org/abs/2602.00161)

    该论文将大语言模型的块删除压缩问题形式化为约束二值优化问题并映射到Ising自旋玻璃物理系统，利用系统能量作为下游模型性能的代理指标，在深度压缩场景下（如50%压缩Llama-3.3-70B）相比现有最先进方法在MMLU基准上提升近23个百分点。

    

    在本文中，我们将通过最优删除Transformer块（“块删除”）来压缩大语言模型（LLM）的问题形式化为一个约束二值优化（CBO）问题，该问题可以映射到一个物理系统（Ising自旋玻璃），其能量是下游模型性能的强有力代理指标。这种形式化使得能够对大量候选块删除配置进行高效排序，从而产生许多高质量的、非平凡的解决方案，超越了仅删除连续区域的方法。我们的方法在深度压缩场景下表现强劲，例如对Llama-3.3-70B-Instruct进行50%压缩时，与其他最先进的（SOTA）块删除方法相比，我们在MMLU基准上实现了近23个百分点的性能提升。对于较轻程度的压缩，对于Llama-3.1-8B-Instruct和Qwen3-14B（在重训练前后），我们的方法在多个基准测试中与现有方法表现相当。

    arXiv:2602.00161v3 Announce Type: replace-cross  Abstract: In this paper, we formulate the compression of large language models (LLMs) by optimally deleting transformer blocks (``block removal'') as a constrained binary optimization (CBO) problem that can be mapped to a physical system (Ising glass), whose energies are a strong proxy for downstream model performance. This formulation enables an efficient ranking of a large number of candidate block-removal configurations yielding many high-quality, non-trivial solutions beyond those only removing consecutive regions. Our method performs strongly in the deep compression regime, such as for 50% compression of Llama-3.3-70B-Instruct, where we achieve an almost 23 percentage point increase on the MMLU benchmark compared to other state-of-the-art (SOTA) block-removal methods. For lighter compression, it performs on par with those methods across several benchmarks for Llama-3.1-8B-Instruct, Qwen3-14B (both before and after retraining), as we
    
[^105]: GeoSense-AI：从危机微博中快速推断地理位置

    GeoSense-AI: Fast Location Inference from Crisis Microblogs

    [https://arxiv.org/abs/2512.18225](https://arxiv.org/abs/2512.18225)

    本文提出GeoSense-AI流水线，通过融合话题标签分词、命名实体识别与地名录消歧等多种NLP技术，从危机微博文本中实时推断地理位置，在保持高精度的同时实现比现有NER工具快数个数量级的处理速度。

    

    本文提出了一个用于从嘈杂微博流中进行实时地理定位的应用型AI流水线，该流水线统一了统计式话题标签分词、基于词性的专有名词检测、围绕灾害词典的依存句法分析、轻量级命名实体识别以及基于地名录的消歧技术，从而直接从文本而非稀疏的地理标签中推断位置信息。该方法在流式处理约束下实现了信息抽取，强调低延迟的NLP组件以及针对地理知识库的高效验证，以支持紧急情况下的态势感知。在与广泛使用的NER工具包的直接对比中，该系统在获得较高F1分数的同时，其吞吐量被设计为快出数个数量级，从而能够部署在实时危机信息学环境中。一个生产级地图界面展示了端到端的AI功能——数据摄取、推理和可视化——以呈现位置信息。

    arXiv:2512.18225v2 Announce Type: replace  Abstract: This paper presents an applied AI pipeline for real-time geolocation from noisy microblog streams, unifying statistical hashtag segmentation, part-of-speech-driven proper-noun detection, dependency parsing around disaster lexicons, lightweight named-entity recognition, and gazetteer-grounded disambiguation to infer locations directly from text rather than sparse geo-tags. The approach operationalizes information extraction under streaming constraints, emphasizing low-latency NLP components and efficient validation against geographic knowledge bases to support situational awareness during emergencies. In head-to-head comparisons with widely used NER toolkits, the system attains strong F1 while being engineered for orders-of-magnitude faster throughput, enabling deployment in live crisis informatics settings. A production map interface demonstrates end-to-end AI functionality---ingest, inference, and visualization---surfacing locationa
    
[^106]: MMGR：多模态生成推理基准与评估

    MMGR: Multi-Modal Generative Reasoning Benchmark and Evaluation

    [https://arxiv.org/abs/2512.14691](https://arxiv.org/abs/2512.14691)

    本文提出MMGR基准，通过涵盖抽象推理、具身导航和物理常识三个领域的10个任务，评估多模态生成模型在物理、逻辑、2D空间、3D空间和时间五种推理能力上的真实推理水平，并采用答案可验证与过程感知的逐帧链式推理评估方法。

    

    现代多模态生成模型能够合成视觉上引人注目的图像和视频，但这种视觉流畅性是否反映了真正的推理能力仍不清楚：当被提示生成解决方案时，模型能否保持任务所需的物理、逻辑、空间和时间约束，还是仅仅产生看似合理的媒体内容？为了回答这个问题，我们提出了MMGR（多模态生成推理基准与评估），这是一个用于评估视频、图像和语言系统生成推理能力的基准。MMGR涵盖了来自三个领域（抽象推理、具身导航和物理常识）的10个任务，并探测五种推理能力：物理推理、逻辑推理、2D空间推理、3D空间推理和时间推理。其评估强调答案可验证的任务，并且对于视频生成，强调过程感知的逐帧链式推理，其中中间帧必须构成通往目标结果的有效步骤。

    arXiv:2512.14691v3 Announce Type: replace  Abstract: Modern multimodal generative models can synthesize visually compelling images and videos, but it remains unclear whether this visual fluency reflects genuine reasoning: when prompted to generate a solution, can a model preserve the physical, logical, spatial, and temporal constraints a task requires, or does it merely produce plausible-looking media? To answer this question, we introduce MMGR (Multi-Modal Generative Reasoning Benchmark and Evaluation), a benchmark for evaluating generative reasoning across video, image, and language-based systems. MMGR covers 10 tasks from three domains (Abstract Reasoning, Embodied Navigation, and Physical Commonsense) and probes five reasoning abilities: Physical, Logical, 2D Spatial, 3D Spatial, and Temporal. Its evaluation emphasizes answer-verifiable tasks and, for video generation, process-aware chain-of-frame reasoning, where intermediate frames must form valid steps toward the target outcome 
    
[^107]: 当偏见伪装成真相：虚假关联如何破坏大语言模型中的幻觉检测

    When Bias Pretends to Be Truth: How Spurious Correlations Undermine Hallucination Detection in LLMs

    [https://arxiv.org/abs/2511.07318](https://arxiv.org/abs/2511.07318)

    该论文揭示了一类由训练数据中虚假关联（如姓氏与国籍的关联）驱动的幻觉，这类幻觉被模型自信生成、不受模型规模扩大影响、能规避现有检测方法、且在拒绝微调后仍持续存在，导致基于置信度过滤和内部状态探测等主流幻觉检测方法从根本上失效。

    

    尽管取得了长足进步，大语言模型（LLMs）仍然会表现出幻觉现象，生成看似合理但不正确的回答。在本文中，我们强调了一类关键但此前未被充分探索的由虚假关联驱动的幻觉——即训练数据中特征（如姓氏）与属性（如国籍）之间存在的表面化但在统计上显著的关联。我们证明，这些虚假关联所引发的幻觉具有以下特点：被模型以高置信度生成、不受模型规模扩大的影响、能够规避当前的检测方法，并且即使经过拒绝微调仍然持续存在。通过系统性控制的合成实验以及对最先进的开源和专有大语言模型（包括GPT-5）的实证评估，我们表明现有的幻觉检测方法，如基于置信度的过滤和内部状态探测，在虚假关联存在的情况下会从根本上失效。

    arXiv:2511.07318v3 Announce Type: replace  Abstract: Despite substantial advances, large language models (LLMs) continue to exhibit hallucinations, generating plausible yet incorrect responses. In this paper, we highlight a critical yet previously underexplored class of hallucinations driven by spurious correlations -- superficial but statistically prominent associations between features (e.g., surnames) and attributes (e.g., nationality) present in the training data. We demonstrate that these spurious correlations induce hallucinations that are confidently generated, immune to model scaling, evade current detection methods, and persist even after refusal fine-tuning. Through systematically controlled synthetic experiments and empirical evaluations on state-of-the-art open-source and proprietary LLMs (including GPT-5), we show that existing hallucination detection methods, such as confidence-based filtering and inner-state probing, fundamentally fail in the presence of spurious correla
    
[^108]: KoSimpleQA：一个韩语事实性基准测试及对推理型大语言模型的分析

    KoSimpleQA: A Korean Factuality Benchmark with an Analysis of Reasoning LLMs

    [https://arxiv.org/abs/2510.18368](https://arxiv.org/abs/2510.18368)

    本文提出了聚焦韩国文化知识的韩语事实性基准KoSimpleQA（含938个问题），发现最强模型正确率仅31.6%、其排名与英语SimpleQA显著不同，并证明推理能力可缓解大语言模型的跨语言知识差距。

    

    我们提出了**韩语SimpleQA（KoSimpleQA）**，这是一个用于评估大语言模型（LLM）事实性的基准测试，重点关注韩国文化知识。KoSimpleQA的设计兼具挑战性与易评分性，由938个具有明确答案的简短事实性问题组成。我们对多种不同规模、支持韩语的开源大语言模型进行了全面评估，发现即使是最强的模型，其正确回答率也仅为31.6%，这凸显了KoSimpleQA的挑战性。值得注意的是，模型在KoSimpleQA上的性能排名与英语SimpleQA上的排名存在显著差异，这凸显了我们数据集的独特价值。此外，我们观察到推理能力有助于缓解大语言模型中的跨语言知识差距，即模型在不同语言中展现知识能力的差异。KoSimpleQA可在 https://github.com/naver-ai/KoSimpleQA 获取。

    arXiv:2510.18368v2 Announce Type: replace  Abstract: We present $\textbf{Korean SimpleQA (KoSimpleQA)}$, a benchmark for evaluating factuality in large language models (LLMs) with a focus on Korean cultural knowledge. KoSimpleQA is designed to be challenging yet easy to grade, consisting of 938 short, fact-seeking questions with unambiguous answers. We conduct a comprehensive evaluation across a diverse set of open-source LLMs of varying sizes that support Korean, and find that even the strongest model generates correct answer only 31.6% of the time, underscoring the challenging nature of KoSimpleQA. Notably, performance rankings on KoSimpleQA differ substantially from those on the English SimpleQA, highlighting the unique value of our dataset. Furthermore, we observe that reasoning helps mitigate the cross-lingual knowledge gap in LLMs, which refers to disparities in their ability to manifest knowledge across languages. KoSimpleQA can be found at https://github.com/naver-ai/KoSimpleQA
    
[^109]: 多语言LLM水印真的是多语言的吗？通过回译将鲁棒性扩展至100多种语言

    Is Multilingual LLM Watermarking Truly Multilingual? Scaling Robustness to 100+ Languages via Back-Translation

    [https://arxiv.org/abs/2510.18019](https://arxiv.org/abs/2510.18019)

    本文提出STEAM检测方法，利用贝叶斯优化在126种候选语言中搜索最能恢复水印强度的回译，使LLM水印检测在包括中低资源语言在内的100多种语言下保持鲁棒。

    

    多语言水印旨在使大语言模型（LLM）的输出在各语言间可追溯，然而现有方法仍存在不足。尽管这些方法宣称具有跨语言鲁棒性，但它们仅在高资源语言上进行了评估。我们证明现有的多语言水印方法并非真正多语言：它们在中等和低资源语言的翻译攻击下无法保持鲁棒性。我们将这一失败归因于语义聚类机制——当分词器词表中某种语言的全词token数量过少时，语义聚类就会失效。为解决这一问题，我们提出了STEAM，这是一种检测方法，利用贝叶斯优化在126种候选语言中搜索最能恢复水印强度的回译版本。该方法与任何水印方法兼容，在不同分词器和语言间均具有鲁棒性，无侵入性，且易于扩展到新语言。平均可带来+0.23 AUC和+37个百分点TP的提升。

    arXiv:2510.18019v3 Announce Type: replace  Abstract: Multilingual watermarking aims to make large language model (LLM) outputs traceable across languages, yet current methods still fall short. Despite claims of cross-lingual robustness, they are evaluated only on high-resource languages. We show that existing multilingual watermarking methods are not truly multilingual: they fail to remain robust under translation attacks in medium- and low-resource languages. We trace this failure to semantic clustering, which fails when the tokenizer vocabulary contains too few full-word tokens for a given language. To address this, we introduce STEAM, a detection method that uses Bayesian optimisation to search among 126 candidate languages for the back-translation that best recovers the watermark strength. It is compatible with any watermarking method, robust across different tokenizers and languages, non-invasive, and easily extendable to new languages. With average gains of +0.23 AUC and +37%p TP
    
[^110]: 教育中大语言模型文本检测器的局限性

    Limits of LLM Text Detectors in Education

    [https://arxiv.org/abs/2508.08096](https://arxiv.org/abs/2508.08096)

    本文提出了一个贡献感知的LLM文本检测评估框架，通过八个学生贡献等级模拟真实写作场景，揭示了当前基于二元区分的LLM检测器在教育评估中的局限性。

    

    学生们在学术写作中越来越多地借助大语言模型（LLM）的辅助。虽然大多数机构政策允许轻微的辅助（例如语法和风格修正以及反馈），但通常禁止将整个写作任务完全交由LLM完成。遗憾的是，目前的LLM生成文本检测方法大多假设人类写作与LLM生成文本之间存在二元区分，忽视了现实中人机协作实践的广度，从而限制了检测系统在教育评估中的有效性。在本文中，我们提出了一个面向教育领域的贡献感知型LLM检测系统评估框架。我们引入了一个包含八个学生贡献等级的量表，用以模拟真实的写作场景，涵盖从完全由人类撰写的文本、LLM辅助修改的文本，到完全由LLM生成并经过对抗性人化处理的文本。机构政策……

    arXiv:2508.08096v2 Announce Type: replace  Abstract: Students increasingly use the assistance of large language models (LLMs) in their academic writing. While slight assistance (e.g., grammar and style correction, as well as feedback) is permitted under most institutional policies, it is usually forbidden to offload entire writing tasks to LLMs. Unfortunately, current approaches to LLM-generated text detection predominantly assume a binary distinction between human-written and LLM-generated text, ignoring the breadth of realistic human-AI collaboration practices and limiting the validity of detection systems for educational assessment. In this paper, we propose a contribution-aware evaluation framework for LLM-based detection systems in education. We introduce a scale of eight student contribution levels that model realistic writing scenarios ranging from fully human-written texts to LLM-assisted revisions to fully LLM-generated and adversarially humanized texts. Institutional policies
    
[^111]: UrduFactCheck：一种具有证据增强与基准测试的乌尔都语代理式事实核查框架

    UrduFactCheck: An Agentic Fact-Checking Framework for Urdu with Evidence Boosting and Benchmarking

    [https://arxiv.org/abs/2505.15063](https://arxiv.org/abs/2505.15063)

    本研究首创性地为乌尔都语构建了两个人工标注基准（UrduFactBench用于声明验证、UrduFactQA用于评估大模型问答事实性），并提出了融合单语与翻译证据检索策略的模块化代理式事实核查框架UrduFactCheck，填补了全球2亿多乌尔都语使用者在事实核查领域的空白。

    

    大型语言模型（LLMs）的快速普及引发了人们对其输出事实可靠性的重要担忧，尤其是在乌尔都语等低资源语言中。现有的自动化事实核查系统主要针对英语开发，这为全球超过2亿乌尔都语使用者留下了显著的空白。在本工作中，我们提出了UrduFactBench和UrduFactQA这两个新颖的人工标注基准，旨在实现乌尔都语的事实核查和事实一致性评估。其中，UrduFactBench专注于声明验证，而UrduFactQA则针对大语言模型在问答中的事实性。这些资源是乌尔都语领域的首创，通过涉及母语乌尔都语使用者的多阶段标注过程开发而成。作为这些基准的补充，我们推出了UrduFactCheck，这是一个模块化的事实核查框架，整合了单语和基于翻译的证据检索策略。

    arXiv:2505.15063v3 Announce Type: replace  Abstract: The rapid adoption of Large Language Models (LLMs) has raised important concerns about the factual reliability of their outputs, particularly in low-resource languages such as Urdu. Existing automated fact-checking systems are predominantly developed for English, leaving a significant gap for the more than 200 million Urdu speakers worldwide. In this work, we present UrduFactBench and UrduFactQA, two novel hand-annotated benchmarks designed to enable fact-checking and factual consistency evaluation in Urdu. While UrduFactBench focuses on claim verification, UrduFactQA targets the factuality of LLMs in question answering. These resources, the first of their kind for Urdu, were developed through a multi-stage annotation process involving native Urdu speakers. To complement these benchmarks, we introduce UrduFactCheck, a modular fact-checking framework that incorporates both monolingual and translation-based evidence retrieval strategie
    
[^112]: LLM-BabyBench：语言模型能否在它们能模拟的世界中进行规划？

    LLM-BabyBench: Can Language Models Plan in Worlds They Can Simulate?

    [https://arxiv.org/abs/2505.12135](https://arxiv.org/abs/2505.12135)

    该论文提出 LLM-BabyBench，将 BabyAI 网格世界改造为完全可观察的纯文本环境，在构造上排除感知、指令理解、检索等所有非规划失败因素，从而对语言模型的规划能力进行纯净、可验证的评估。

    

    当一个交互式基准测试报告语言模型智能体的单一成功率时，人们往往难以清楚这个数字究竟衡量的是什么。失败可能来自感知、模糊的指令、检索、关于动作效果的常识缺失、不正确的动力学模型，或是规划本身，而一个总体得分无法将这些因素区分开来。LLM-BabyBench 将程序化生成的 BabyAI 网格世界重新构建为一个完全可观察的纯文本环境，在构造上排除了除规划之外的所有失败来源。整个网格被序列化到提示中，指令来自一个小的形式语法，每个对象的坐标都被明确给出，六个动作及其效果被完整说明，并且一个确定性专家通过实际执行而非主观判断来验证每个答案。在这一基础之上，我们定义了 PPD 套件：Predict 要求给出动作序列执行后所处的状态，Plan 要求给出一个能达成目标的动作序列……

    arXiv:2505.12135v2 Announce Type: replace-cross  Abstract: When an interactive benchmark reports a single success rate for a language-model agent, it is rarely clear what that number measures. A failure can come from perception, ambiguous instructions, retrieval, missing commonsense about what actions do, an incorrect model of the dynamics, or planning, and an aggregate score does not separate them. LLM-BabyBench recasts the procedurally generated BabyAI gridworld as a fully observable, purely textual environment in which every source of failure but planning is removed by construction. The whole grid is serialised into the prompt, instructions come from a small formal grammar, every object's coordinate is stated, the six actions and their effects are specified, and a deterministic expert validates each answer by executing it rather than judging it. On this substrate we define the PPD suite: Predict asks for the state that follows an action sequence, Plan for an action sequence that rea
    
[^113]: 语义压缩的统计力学

    Statistical Mechanics of Semantic Compression

    [https://arxiv.org/abs/2503.00612](https://arxiv.org/abs/2503.00612)

    该论文提出将语义压缩建模为欧几里得语义空间中的优化问题，并创新性地将其映射为自旋玻璃哈密顿量，利用统计力学方法求解在保持语义的前提下最小化消息长度的问题。

    

    语义压缩的基本问题是在保留消息意义的前提下最小化消息的长度。这与经典的压缩概念不同之处在于，失真不是直接在比特层面上进行度量，而是在一个抽象的语义空间中度量。为了使这一概念更加精确，我们从认知神经科学和机器学习中汲取灵感，将语义空间建模为一个连续的欧几里得向量空间。在这样的空间中，语音、图像甚至想法等刺激被映射为高维实向量，这些嵌入的位置决定了它们相对于其他嵌入的意义。这表明语义相似性的一个自然度量就是欧几里得距离，这也是我们在本工作中采用的度量方式。我们将确定最小长度且保持意义的消息这一优化问题映射到一个自旋玻璃哈密顿量，并求解由此产生的统计力学问题。

    arXiv:2503.00612v2 Announce Type: replace-cross  Abstract: The basic problem of semantic compression is to minimize the length of a message while preserving its meaning. This differs from classical notions of compression in that the distortion is not measured directly at the level of bits, but rather in an abstract semantic space. In order to make this precise, we take inspiration from cognitive neuroscience and machine learning and model semantic space as a continuous Euclidean vector space. In such a space, stimuli like speech, images, or even ideas, are mapped to high-dimensional real vectors, and the location of these embeddings determines their meaning relative to other embeddings. This suggests that a natural metric for semantic similarity is just the Euclidean distance, which is what we use in this work. We map the optimization problem of determining the minimal-length, meaning-preserving message to a spin glass Hamiltonian and solve the resulting statistical mechanics problem u
    
[^114]: 从实验室到临床：药物发现与开发中的临床试验综述

    From Bench-to-Bedside: A Review of Clinical Trials in Drug Discovery and Development

    [https://arxiv.org/abs/2412.09378](https://arxiv.org/abs/2412.09378)

    本综述系统梳理了药物开发中I至IV期临床试验的特点与联系，分析了伦理合规、受试者招募等主要挑战，并阐述了人工智能等创新技术及新兴疗法对未来临床试验设计的变革性影响。

    

    临床试验连接着基础研究与临床应用，是药物开发过程中必不可少的环节。本综述考察了临床试验的各个阶段（I期[安全性评估]、II期[疗效评价]、III期[大规模验证]和IV期[上市后监测]），重点阐述了各阶段的特点及其相互联系。文中识别了主要挑战，包括伦理合规、受试者招募以及确保试验人群的多样性和代表性，并提出了基于证据的应对策略。为应对这些挑战，人工智能、大数据分析和数字健康工具等创新技术正在变革试验的设计与实施，提升了效率和数据质量。展望未来，本综述探讨了包括基因治疗和免疫治疗在内的新兴疗法如何重塑试验设计的要求。

    arXiv:2412.09378v4 Announce Type: replace-cross  Abstract: Clinical trials bridge basic research and clinical application, serving as essential steps in drug development. This review examines clinical trial phases (Phase I [safety assessment], Phase II [efficacy evaluation], Phase III [large-scale validation], and Phase IV [post-marketing surveillance]), highlighting the distinct characteristics and interconnections. Major challenges are identified, including ethical compliance, participant recruitment, and ensuring diversity and representativeness in trial populations, while proposing evidence-based mitigation strategies. To address these challenges, innovative technologies, such as artificial intelligence, big data analytics, and digital health tools, are transforming trial design and implementation, enhancing efficiency and data quality. Looking forward, the review explores how emerging therapies, including gene therapy and immunotherapy, are reshaping trial design requirements and 
    
[^115]: 并非所有实体生而平等：研究超细粒度实体类型分类中的长尾问题

    All Entities are Not Created Equal: Examining the Long Tail for Ultra-Fine Entity Typing

    [https://arxiv.org/abs/2410.17355](https://arxiv.org/abs/2410.17355)

    本研究提出一种新颖的启发式方法来近似实体的预训练分布，系统揭示了仅依赖预训练语言模型参数化知识的实体类型分类方法在处理长尾实体时表现显著不佳，表明需要超越预训练语言模型的方法来应对不常见实体。

    

    由于预训练语言模型（PLMs）具有从大型语料库中获取世界知识的能力，它们被广泛应用于标签空间极其庞大的超细粒度实体类型分类任务中。在这项工作中，我们通过提出一种新颖的启发式方法来近似实体的预训练分布（在预训练数据未知的情况下），从而探索预训练语言模型所获取知识的局限性。随后，我们系统地证明了仅依赖预训练语言模型参数化知识的实体类型分类方法在处理处于预训练分布长尾的实体时表现显著不佳，而知识注入方法可以部分弥补这些不足。我们的研究结果表明，我们需要超越预训练语言模型，才能为不常见实体提供表现良好的解决方案。

    arXiv:2410.17355v4 Announce Type: replace  Abstract: Due to their capacity to acquire world knowledge from large corpora, pre-trained language models (PLMs) are extensively used in ultra-fine entity typing tasks where the space of labels is extremely large. In this work, we explore the limitations of the knowledge acquired by PLMs by proposing a novel heuristic to approximate the pre-training distribution of entities when the pre-training data is unknown. Then, we systematically demonstrate that entity-typing approaches that rely solely on the parametric knowledge of PLMs struggle significantly with entities at the long tail of the pre-training distribution, and that knowledge-infused approaches can account for some of these shortcomings. Our findings suggest that we need to go beyond PLMs to produce solutions that perform well for infrequent entities.
    

