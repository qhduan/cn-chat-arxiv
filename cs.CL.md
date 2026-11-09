# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [CodeQuant2: Enhanced Outlier Mitigation and Adaptive Clustering for Low-Precision LRMs](https://arxiv.org/abs/2611.09868) | 提出 CodeQuant2 量化框架，通过旋转缩放缓解激活离群值、以聚类表示非均匀权重分布并按各层量化难度自适应调整码本大小，从而显著提升低精度大型推理模型在推理任务上的表现。 |
| [^2] | [How far does a hash-only sender go? A pre-registered map of sender memory versus bits for short text messages](https://arxiv.org/abs/2611.09848) | 本研究通过预注册实验发现，仅计算哈希的发送端传输短文本的每条消息成本比领域内 PPM 一阶发送端多 18.4 比特，不满足成本等价标准，但其以约 8KB 的极小发送端内存成为内存-比特权衡中比特数最省的方案。 |
| [^3] | [BitDecoding-FP4: Native FP4 KV Cache Decoding via Algorithm-System Co-Design](https://arxiv.org/abs/2611.09825) | BitDecoding-FP4通过算法-系统协同设计，结合双残差查询量化、下溢感知概率缩放和离群值感知键量化三项技术，实现了精确且高效的原生FP4 KV缓存解码，解决了现有方法在精度与吞吐量之间的权衡问题。 |
| [^4] | [Compaction, decay and reinforcement in a two-tier conversational memory: a paired evaluation against retrieval-augmented generation](https://arxiv.org/abs/2611.09807) | 该论文提出将对话逐轮压缩为原子事实的两层记忆机制，通过严格的配对评估证明其比检索增强生成（RAG）准确率高 6.8 个百分点且提示 token 减少 34%，但优势会随问题措辞要求和对抗性问题的增加而减弱。 |
| [^5] | [Maya a1: Do the Options Read the Model, or Does Their Position? Order Sensitivity in a Non-Autoregressive Decision Model](https://arxiv.org/abs/2611.09803) | 论文发现非自回归决策模型Laya对选项排列顺序高度敏感——在选项循环排列下约26%的问题答案会发生改变，且翻转集中于低置信度问题——而采用选项打乱的数据进行微调可将翻转率降至7.2%，证明选项打乱是缓解顺序敏感性的关键手段。 |
| [^6] | [Are These Programmers Ripe Enough to Eat? Evaluating How LLMs Handle Lexical Knowledge Conflicts in Translation](https://arxiv.org/abs/2611.09785) | 本研究通过受控的中文-英文翻译任务系统评估了大语言模型在面对外部词汇资源与自身词汇知识冲突时的表现，发现模型（尤其是更大的模型）倾向于依赖提示中提供的情境知识，即使被明确要求忽略这些信息。 |
| [^7] | [ROSE: Risk-Aware Orthogonal Subspace Navigation for Lifelong Knowledge Editing in Multimodal Large Language Models](https://arxiv.org/abs/2611.09770) | 提出ROSE框架，通过将参数更新限制在已编辑子空间的正交补空间，并利用动态风险掩码保护关键事实参数子空间，有效缓解多模态大语言模型终身知识编辑中的局部性侵蚀与灾难性遗忘问题。 |
| [^8] | [Language as Trait, Politeness as Strategy: Controlled Tests of Linguistic Effects in LLM Social-Deduction Agents](https://arxiv.org/abs/2611.09768) | 该研究通过在狼人杀游戏中受控地改变LLM智能体语言复杂度与礼貌策略，发现这些语言风格差异对其他智能体的信任、投票和夜间行动均无显著影响。 |
| [^9] | [MemoryDojo: Benchmarking Poisoning Through LLM Agent Memory](https://arxiv.org/abs/2611.09721) | 该论文提出了MemoryDojo基准测试，用于追踪和量化投毒信息在LLM智能体记忆系统中的传播及其对后续行为的影响，涵盖155条多会话轨迹、四个领域和五种记忆系统，并发现所有记忆系统均易受记忆投毒攻击。 |
| [^10] | [RidgeFill: Training-Free Late-Layer Cache Fill for Faster Prefill](https://arxiv.org/abs/2611.09709) | RidgeFill 提出一种免训练方法，通过从一次校准数据中以闭式解拟合的岭回归映射，由分割层状态直接预测并填充模型晚层的 KV 缓存，从而跳过预填充阶段大部分上层的计算，在不重训模型、不丢弃 token 的情况下加速推理。 |
| [^11] | [How Much Conversation History Does an AI Agent Need?](https://arxiv.org/abs/2611.09662) | 该研究通过对1,100个BEAM问题的受控实验发现，增大段落（上下文）预算虽能提升平均得分但边际收益递减，同时消息保留实验表明Qwen在保留用户消息时更能匹配完整对话状态、但对历史报价的引用能力会下降，而Mistral在所有条件下表现接近最低水平。 |
| [^12] | [Steerable Multi-Objective Group-Relative Preference Optimization with Jointly Trained Low-Rank Adapters](https://arxiv.org/abs/2611.09634) | 提出LambdaMix，一种参数高效方法，在冻结基础模型上为每个目标联合训练一个低秩适配器并结合组相对在线强化学习，只需单次训练即可在推理时根据偏好向量灵活调整多目标间的权衡，无需额外优化。 |
| [^13] | [RuACT: Recovering Inactive Rubrics as Adaptive Critics and Teachers](https://arxiv.org/abs/2611.09597) | RuACT通过将不可达评分标准用作特权信息进行奖励门控自蒸馏、将饱和评分标准用作批评者来区分响应质量，使基于评分标准的强化学习中超过60%的失活监督信号重新产生有效学习信号。 |
| [^14] | [Query-Guided RoPE Modulation for Lost-in-the-Middle Mitigation in Long-Context LLMs](https://arxiv.org/abs/2611.09554) | 提出QGPM，一种无需训练的方法，利用模型自身注意力头的查询相关性动态重塑RoPE位置编码，从而缓解长上下文大语言模型中“迷失在中间”的问题。 |
| [^15] | [Benchmarking Speaker Embedders in the Sub-Second Regime for Real-Time Voice AI Agents](https://arxiv.org/abs/2611.09516) | 该论文在严格的亚秒级延迟预算下对五种开源说话人嵌入模型进行了基准测试，发现嵌入前去噪通常有害，以评估其实时语音AI智能体中区分活跃用户与穿透传统降噪的背景语音的能力。 |
| [^16] | [On the Effectiveness and Limitations of Open-Weight LLMs for Low-Resource Offensive Language Detection in Danish](https://arxiv.org/abs/2611.09506) | 该研究系统评估了四个开放权重LLM家族在丹麦语低资源冒犯性语言检测中的表现，发现LLM即使不微调也能超越微调后的BERT基线，且提示设计与模型预训练的交互作用无法被先验预测。 |
| [^17] | [BEiTScore++: Reference-free Image Captioning Evaluation with BEiT-3 and Conformalized Quantile Regression](https://arxiv.org/abs/2611.09505) | 提出BEiTScore++，通过共形化分位数回归为无参考图像描述评估指标引入共形校准的置信区间，在单次前向传播中同时实现准确评分与可靠的不确定性量化。 |
| [^18] | [An Effective Spoken Language Understanding Approach Leveraging Query-Adaptive Intent Detection](https://arxiv.org/abs/2611.09491) | 本文提出APEX两阶段检索框架，先通过意图级原型筛选候选意图、再在该意图空间内检索上下文示例，无需微调即可显著提升大语言模型在口语理解任务中的意图检测准确率，相比基于查询的检索最高可带来7.66个百分点的EMA提升。 |
| [^19] | [From LoRA to Language: Recovering Skill Documents from Adapters](https://arxiv.org/abs/2611.09454) | 提出Executing Writer方法，通过在适配后模型之上运行并训练一个小型写手，可从LoRA适配器及其冻结基座模型中恢复约39-44%的技能文档关键指令，而仅靠读取权重几乎无法恢复任何内容。 |
| [^20] | [A Spiking Neural Model of the Mental Lexicon: Representations, Similarity Structure, and Associative Retrieval in Perception and Production](https://arxiv.org/abs/2611.09402) | 本文在神经工程框架内构建了一个包含613个词汇条目的脉冲神经网络心理词典模型，通过引理中介支持知觉（语音到概念）和产生（概念到语音）两条处理通路，并结合胜者全取竞争与阈值传递两种联想记忆机制来模拟词汇通达。 |
| [^21] | [Enabling Privacy-Preserving Natural Language Access to Industrial Smart Meter Data](https://arxiv.org/abs/2611.09372) | 本文提出OpenMeter，一个隐私保护、与模式无关的多智能体系统，可将自然语言查询转换为可执行代码，使非技术人员无需SQL专业知识即可安全地访问和分析工业智能电表数据。 |
| [^22] | [Syn2DRE: Synergistic Dual Teacher-Student Framework for Document-level Relation Extraction](https://arxiv.org/abs/2611.09342) | Syn2DRE提出了一种协同双师生半监督框架，通过在轻量级师生架构中结合自监督学习与交叉监督来生成更可靠的伪标签，并采用统一的端到端训练策略，在文档级关系抽取任务上取得了超越现有最先进方法的性能。 |
| [^23] | [ABHBench: Evaluating Moral Decision-Making of Foundation Models from an Agentic Perspective](https://arxiv.org/abs/2611.09326) | 该论文提出了基于《底特律：变人》游戏构建的多模态基准ABHBench，首次让基础模型以第一人称行动主体的身份在分支叙事场景中面对道德困境，并从阿西莫夫机器人三定律启发的六个维度评估其道德决策能力。 |
| [^24] | [Asymmetric Correspondence Learning for Lexical and Style Embeddings: Interpreting Latent Space via Acoustic Features](https://arxiv.org/abs/2611.09313) | 该论文提出非对称对应学习（ACL）方法，通过分割潜在对应瓶颈将HuBERT特征蒸馏为相互独立的声学词嵌入与声学风格嵌入，并揭示了即使在探测显示信息已被剥离的情况下，各潜在空间中仍隐含保留着来自外部来源的大量方差。 |
| [^25] | [Multilingual Artificial Datasets for AI-Driven Cardiology: the CliTe and ParaCliTe corpora](https://arxiv.org/abs/2611.09296) | 该研究提出人工临床语料库构建指南，并发布了由7家欧洲医院心脏病医生撰写的涵盖7种语言的人工心脏病学语料库 CliTe 及其平行翻译语料库 ParaCliTe，在规避隐私问题的同时为多语言心脏病学 NLP 模型的可信发展提供了稀缺的高质量数据资源。 |
| [^26] | [DUET: Dual-Sparse Inference for Block Diffusion Language Models](https://arxiv.org/abs/2611.09239) | DUET 是一个无需训练的双稀疏推理框架，通过前缀 KV 稀疏性和早期查询稀疏性，显著降低块扩散语言模型的推理开销。 |
| [^27] | [KLEAR: Knowledge-Matched LLM Evaluation in Korean across Application Requirements via Controlled Benchmark Transformation](https://arxiv.org/abs/2611.09223) | KLEAR是一个韩语LLM评估框架，通过在保持所需知识不变的情况下改变应用需求来构建知识匹配的对比基准，从而能够区分模型的错误回答究竟源于知识缺乏还是应用能力不足。 |
| [^28] | [Khmer PFGen-Bench: Building a Judge-Free Benchmark for Open-Ended Generation in a Low-Resource Language](https://arxiv.org/abs/2611.09222) | 该论文提出了 Khmer PFGen-Bench，这是首个针对高棉语这一低资源语言的无裁判开放式生成基准，并发现混合九个模型家族的后训练模型生成的参考答案能兼顾有效性与多样性，从而使模型排名保持稳定。 |
| [^29] | [RPEG: Training Reasoning Language Models Scalably and Cost-Effectively](https://arxiv.org/abs/2611.09207) | 提出了一种新颖的以数据为中心的图中间表示 RPEG，通过联合优化 RLM 强化学习训练流水线的结构与执行，并结合综合分析性能模型，实现了推理语言模型训练的可扩展性与高性价比。 |
| [^30] | [Representational drift in a code model: behaviour settles, which neurons carry it does not](https://arxiv.org/abs/2611.09204) | 该研究首次在代码大模型OpenCoder-1.5B的预训练中系统测量了神经科学中的“表征漂移”现象，发现对于语法固定的Python构造，模型行为（损失）在训练早期即已定型，但承载这些行为的神经元集合在整个训练过程中持续变化、从未稳定。 |
| [^31] | [Copies Are Not Confirmations: How News Duplication Fools LLM Judges and Continual Learners](https://arxiv.org/abs/2611.09203) | 该研究揭示了新闻重复副本并非独立确认，LLM评判者会被同一来源的重复报道欺骗——如Llama 3.2 3B和phi4 14B在面对三份相同转载时接受谎言的比例从0.06和0.21分别飙升至0.71和0.90——表明去重检测对LLM证据评估至关重要。 |
| [^32] | [Structure and Lineage Evidence for Vietnamese Statutory Article Retrieval on DRiLL](https://arxiv.org/abs/2611.09200) | 该论文通过引入条款身份、同一法律支持以及引用与谱系证据等结构化特征，将DRiLL数据集上越南法律条文检索的官方F2指标从71.77%提升至72.91%。 |
| [^33] | [Calibrated concept profiles: making the reading of a model's neurons a measurement](https://arxiv.org/abs/2611.09196) | 本文提出“校准概念轮廓”方法，在明确声明的刺激总体上对每个神经元进行标准化测量，使可解释性研究中的直接神经元读取成为一种可重复、可比较、带有校准单位的测量。 |
| [^34] | [A Taxonomy-Driven Synthetic Dataset for Retrieval Effectiveness and Fairness Evaluation in Hiring](https://arxiv.org/abs/2611.09178) | 该论文提出了一个基于分类学驱动、完全合成的多语言HR数据集与基准，可同时评估检索有效性与算法公平性，揭示了通用基准在招聘场景中的不足，并为识别兼顾语义准确性与公平性的帕累托最优模型提供了定制化评估方案。 |
| [^35] | [BMMS: Policy-Conditioned Measurement of Behavioral Compliance in Language Models](https://arxiv.org/abs/2611.09147) | BMMS是一个以任意已发布政策语料库为输入、返回合规画像的策略条件化度量套件，通过合规达成度分数（CAS）和合规稳健性裕度（CRM）等指标，首次实现了对语言模型行为合规性及其对抗稳健性的量化评估。 |
| [^36] | [PathXAI: Evidence-guided structured pathology reporting by transferable tissue segmentation](https://arxiv.org/abs/2611.09117) | PathXAI通过可迁移的组织分割信息结合H&E图像与领域知识来引导诊断推理和结构化病理报告生成，在诊断一致性和形态学描述准确性上显著优于纯H&E方法。 |
| [^37] | [PhoPunct: A Large-Scale Benchmark of Transformer Architectures for Vietnamese Punctuation Restoration](https://arxiv.org/abs/2611.09107) | 该论文构建了包含2210万标注词元的越南语标点恢复大规模基准，评估17种Transformer架构后发现，在固定训练预算下添加Bi-LSTM和CRF层并不能可靠提升性能，纯Transformer分类器在大多数对比中与Bi-LSTM-CRF模型相当甚至更优。 |
| [^38] | [AlienBench: A Benchmark of Structural Novelty in Language Model Outputs](https://arxiv.org/abs/2611.09094) | AlienBench将认知心理学中的结构化想象任务（想象外星生物）适配到语言模型，通过衡量模型输出偏离地球动物熟悉特征（如双侧对称性、腿部运动方式）的程度，为新颖性评估提供了具有固定参照基准的测试方法。 |
| [^39] | [MARS: Training Multilingual Search Agents for Retriever-Free Evidence Acquisition](https://arxiv.org/abs/2611.09090) | 提出MARS——一个通过强化学习训练的多语言搜索智能体，让模型直接与原始语料库交互而无需传统检索器，并借助语料库原生的实体锚定和冗余感知的搜索奖励机制来克服跨语言精确搜索的困难、实现高效证据获取。 |
| [^40] | [The Invisible Gap: Romanized Indic Generation Fails Where Automatic Metrics Succeed](https://arxiv.org/abs/2611.09085) | 该研究发现自动指标（如BGE-M3）系统性地误报罗马化印度语言生成的真实质量——人类评分差距显著而指标差异近乎为零，并揭示出文字溢出与英文退化两种模型失败模式，据此提出故障感知的评估策略。 |
| [^41] | [Large Language Model Deployment on Resource-Constrained Edge Devices: A Practitioners Survey](https://arxiv.org/abs/2611.09075) | 本文综述了软件优化与专用硬件领域的最新技术，使大语言模型能够在资源受限的边缘设备上部署，并阐述了本地数据处理、成本节约、降低延迟和个性化响应等边缘部署优势。 |
| [^42] | [Does Reasoning Improve Psychological Depth in Large Language Models?](https://arxiv.org/abs/2611.09061) | 该研究通过盲测成对评估发现，推理增强并不必然提升大语言模型在短篇小说心理深度这类开放式创意写作中的表现，GPT-5仅略优于GPT-4o，而DeepSeek-R1反而落后于非推理版的DeepSeek-V3。 |
| [^43] | [Real-Time Short-Response Selection for Spoken Dialogue Systems](https://arxiv.org/abs/2611.09057) | 提出了一种基于对比学习的实时短回应选择方法，在检测到对话轮替事件时从预生成候选中即时选择并播放短回应，从而有效掩盖口语对话系统中语言生成与语音合成的延迟。 |
| [^44] | [Failure Profiling and Reachable Trajectory Selection for Reasoning Distillation](https://arxiv.org/abs/2611.09042) | 提出FPRT方法，通过重复采样刻画学生模型的失败行为并筛选其“可学习”的训练轨迹，使蒸馏训练数据同时兼顾学生失败点与实际可达能力，从而缓解师生分布不匹配导致的泛化差问题。 |
| [^45] | [Language, Not Code Abstraction: Natural-language Training Shapes Brain-aligned Geometry](https://arxiv.org/abs/2611.09027) | 研究通过对25个GPT-2模型的对照实验证明，只有自然语言训练才能塑造与大脑对齐的表征几何，代码训练无法替代自然语言训练在神经对齐中的作用。 |
| [^46] | [Seeing vs. Counting: Complexity Twins Show Where Multimodal LLMs Fail on Diagrams](https://arxiv.org/abs/2611.08998) | 该论文通过在同一图上配对不同计数复杂性类别的“复杂性孪生”问题，并辅以转录探针和纯文本对照，将多模态大模型的“看见”与“计数”能力分离，发现模型主要在计数环节失败且失败率取决于答案大小而非复杂性类别，而在提供精确转录和代码时模型可以正确计数。 |
| [^47] | [Same Facts, Different Remedy: Linguistic Profiling and Arbitrariness in LLM and System One Decisions](https://arxiv.org/abs/2611.08995) | 该研究通过对消费者投诉进行社会语言学改写来审计LLM与概率模型，发现尽管群体层面的语言偏见微小，但同一事实的投诉因写作风格不同而在2.6-6.3%的情况下获得不同的金钱救济，揭示了决策系统中的语言特征画像与任意性问题。 |
| [^48] | [Beyond Truncation? Entailment-Licensed Elision and the Limits of Semantic Preservation in LLM Decoding](https://arxiv.org/abs/2611.08975) | 该论文证明了现有免训练的解码期冗余删除方法本质上都只是截断，并提出ELIDE方法，利用自然语言推理模型在每个句子边界判断被删内容是否被保留内容所蕴含，从而实现真正超越截断的语义保持式回答精简。 |
| [^49] | [Phantom Consensus: Measuring Social Reasoning in Multi-Agent LLM Decision-Making](https://arxiv.org/abs/2611.08971) | 该论文首次发现并命名了多智能体LLM系统中的“幻影共识”现象——智能体在模糊社会信号下改变公开决策却误判他人私下判断，并提出了PHANTOM-Bench基准来系统测量这种多元无知式失败模式。 |
| [^50] | [Using an LLM to Tackle Difficult Cases in NER for Nepali Tweets](https://arxiv.org/abs/2611.08967) | 本研究利用微调PLM的词元概率将测试集划分为简单与困难案例，并通过基于语义相似度选择示例的提示策略使LLM在困难NER案例上超越PLM，两模型结合后获得更优的整体性能。 |
| [^51] | [ResQAdam: Residual-Preserving Adaptive Optimization for Memory-Efficient Quantization Aware Training](https://arxiv.org/abs/2611.08924) | ResQAdam通过仅维护单个BF16动量、只对主子空间的r×r坐标应用Adam，并利用残差的归一化符号更新来保留被投影方法丢弃的动量信息，在权重量化至2比特的量化感知训练中将优化器状态内存相比AdamW减少71%至74%，同时保持接近的性能。 |
| [^52] | [On the Incompleteness of do-calculus for Universal Identifiability](https://arxiv.org/abs/2611.08916) | 本文为do-演算的完备性提出了严格的形式化定义，并指出在从任意观测与干预分布集合中进行因果效应识别的通用可识别性问题中，do-演算并不完备。 |
| [^53] | [From Procedures to Permissions: Learning Precise Subagent Authorization from Agent Skills](https://arxiv.org/abs/2611.08905) | 该论文提出将Agent Skills中编码的过程性知识（即每一步所需的操作、工具和文件信息）作为天然监督信号，通过从技能合成任务并在交互环境中以子智能体委派方式执行，来学习为子智能体授予精确的工具与工作区资源权限。 |
| [^54] | [Richness Is Not Evenness: Ordered Renyi Crossings under Self-Training](https://arxiv.org/abs/2611.08892) | 该论文通过九代GPT-2递归自训练实验首次发现不同阶雷尼熵存在有序的温度交叉点，证明词汇丰富性与分布均匀性是两个相互独立的维度：词类型数量减少的同时，高频词的主导性反而降低。 |
| [^55] | [SteerGR: Steering Generative Recommendation via Sparse Autoencoders for Controllable Game Discovery](https://arxiv.org/abs/2611.08790) | 提出SteerGR框架，利用稀疏自编码器将人类可读的游玩概念与生成式推荐模型的内部特征关联，通过直接干预所选特征实现无需重训练的概念级可控游戏推荐。 |
| [^56] | [Guardrail Erosion in Domain Fine-Tuned Large Language Models: A Paired Audit of Knowledge-Centric Abstention in Biomedical LLMs](https://arxiv.org/abs/2611.08780) | 本研究通过对 25 个生物医学领域微调大语言模型与其基础模型的配对大规模审计，首次揭示微调会进一步侵蚀本已薄弱的知识型拒答能力，证明领域微调会导致模型安全护栏的系统性退化。 |
| [^57] | [Jacobian-Causal Memory Control for Persistent Memory in Language Model Systems](https://arxiv.org/abs/2611.08779) | 该论文提出雅可比-因果记忆控制（JCMC），通过预测影响签名、目标效应与副作用预算下的干预选择、以及预测-观测对比的保留或回滚机制，来控制持久记忆对语言模型当前推理的因果影响，并为其提供理论保证和反事实评估协议。 |
| [^58] | [Semantic Conditional Similarity (SCS): A Formal Mathematical Framework](https://arxiv.org/abs/2611.08778) | 本文提出语义条件相似度（SCS）这一形式化数学框架，通过集合论运算在消除与第三方文档共享的语义元素后计算文档相似度，为多文档场景引入方向性、角色不对称性及文档间条件化机制。 |
| [^59] | [AttentionSwitch: Jailbreak Defense for LLMs via Joint Learning of Sparse Attention-Head Selection and Scaling](https://arxiv.org/abs/2611.08752) | AttentionSwitch是首个通过联合学习稀疏注意力头的选择与缩放，同时优化越狱防御中“在哪里干预”和“如何干预”两个决策的大语言模型防御方法。 |
| [^60] | [Structuring Survey Histories for LLM Digital Twins: A Controlled Study of Background, Decision Procedure, and Evaluation Profiles](https://arxiv.org/abs/2611.08730) | 提出将受访者调查历史结构化为“背景—决策程序—评估”（BDE）的固定表示方法，相比原始调查记录可显著提升LLM数字孪生预测个体答案的准确率。 |
| [^61] | [Patching the Blind Spots: Reversible LLM Rule Overlay for Frozen Fraud Detectors](https://arxiv.org/abs/2611.08714) | 提出一个可逆的LLM规则叠加框架，在不修改冻结欺诈检测模型的前提下，利用SHAP定位模型盲点并由LLM生成、筛选、校准规则，通过多模型logit平均池化，在真实电商欺诈检测中于同等精确率下平均提升召回率2.26%。 |
| [^62] | [RL-ATTACK: A Generalized Black-Box Adversarial Attack on Language Models via Reinforcement Learning](https://arxiv.org/abs/2611.08711) | 该论文提出了RL-ATTACK，一种基于强化学习的可训练黑盒对抗攻击框架，通过从目标模型学习生成定制化对抗变换，在自然语言推理和生成式问答任务上均实现了更低的攻击后准确率、更高的查询效率和更好的语义保持。 |
| [^63] | [Dual-Stream LLM-Guided Reward Design for Reject-Inference-Aware Reinforcement Learning in E-Commerce Payment Fraud Detection](https://arxiv.org/abs/2611.08706) | 该论文提出由单个LLM设计的双流奖励机制，通过让LLM自行决策如何利用风险信号对比新旧评分，为被旧系统拦截的无标签交易打分，从而在强化学习欺诈检测中解决拒绝推断偏差问题。 |
| [^64] | [Contextuality and Contradictions: Exploring Real-World Identifications of AI-Generated Writing](https://arxiv.org/abs/2611.08704) | 本研究结合定性编码、词汇变化的计算分析与语言人类学视角，考察了维基百科编辑者与学生识别AI生成文本的真实世界实践，发现这些识别依赖多样且相互矛盾的证据类型——从文本特征与内容判断到循环论证及对作者道德品质的假设。 |
| [^65] | [Compression of Emotion in Language Model Representations](https://arxiv.org/abs/2611.08683) | 该研究通过对七个大语言模型在GoEmotions数据集上的隐藏表示进行无标签聚类分析，发现模型内部对情感的编码存在“过度压缩”现象——其最清晰的情感区分仅为积极与消极两大类，远粗于28个细粒度情感类别。 |
| [^66] | [Greedy Byte-Pair Encoding Achieves at Least 0.361 of the Optimal Compression](https://arxiv.org/abs/2611.08671) | 本文证明贪心字节对编码在任意字符串、预算和平局打破规则下，至少能达到最优压缩缩减量的 0.361 倍，实现了对先前三分之一下界的首次改进。 |
| [^67] | [ChainFlow: Few-Step Latent Flow Matching for Speculative Decoding](https://arxiv.org/abs/2611.08669) | ChainFlow提出首个以目标模型隐状态为条件的流匹配草稿模型，通过一至两步的潜空间短程传输生成草稿，使草稿成本几乎不随目标模型规模增长，同时兼顾并行生成与token间依赖。 |
| [^68] | [Turkish Multimodal Video Fact-Checking: A New Dataset and a Comparison of Zero-Shot Agentic and Fine-Tuned Systems](https://arxiv.org/abs/2611.08651) | 该论文发布了首个土耳其语多模态视频事实核查数据集TeViD，并基于同一开源基础模型构建和比较了零样本智能体验证流水线与微调系统两种事实核查方案。 |
| [^69] | [FinEvolveKG: Controlled and Attributable Prompt Evolution for Financial Knowledge Graph Construction](https://arxiv.org/abs/2611.08643) | FinEvolveKG通过将可复用的通用模块与角色策略模块分离，并将反思式提示词优化器GEPA的每次编辑严格限制在单独版本管理的角色策略模块内，实现了金融知识图谱构建中可控、可归因且不破坏共享规则的提示词演化。 |
| [^70] | [Forcing a Letter Suppresses Refusal: Mechanistic Evidence from WMDP](https://arxiv.org/abs/2611.08639) | 该论文首次从机制层面解释了多项选择题（MCQ）格式为何会抑制LLM的拒答：模型会过早地锁定答案字母，其内部激活显著远离拒答方向，且消融承载答案的注意力头会使拒答回升，证明答案选择与拒答存在直接竞争。 |
| [^71] | [How Small Can We Go? Calibrating Activation-Cache Compression for SAE Training](https://arxiv.org/abs/2611.08551) | 该论文通过冻结SAE测试和重训练测试研究激活缓存压缩对稀疏自编码器训练的影响，发现特征保留比例缺乏清晰规律，且SAE训练对微小的数值扰动也异常敏感。 |
| [^72] | [Subliminal Learning in Vision-Language Models](https://arxiv.org/abs/2611.08538) | 本研究证明潜意识学习现象同样存在于视觉-语言模型中——教师VLM能够通过与偏好完全无关的几何形状数字生成任务，将隐藏的动物或面部偏好传递给学生模型，传递差异最高可达约10个百分点。 |
| [^73] | [Structure is in the zoom: probing neural symmetry through dimensionality scaling](https://arxiv.org/abs/2611.08506) | 该论文提出将维度缩放指数分解为抽样底线与沿特定轴的结构性偏移，证明该斜率并非神经系统的固有属性，并借此在小鼠V1中检测出神经编码的O(2)对称性。 |
| [^74] | [Improving Contextual Retrieval for Long Documents using Semantic Chunking and Hybrid Retrieval Strategies](https://arxiv.org/abs/2611.08498) | 该论文提出一种混合RAG框架，通过知识图谱结构化推理与稠密向量检索相结合，配合语义分块和分层检索机制，有效解决了长文档多跳推理中的上下文碎片化和检索精度问题。 |
| [^75] | [Rethinking Data Selection as Distribution Matching on Semantic Manifolds](https://arxiv.org/abs/2611.08495) | 该论文提出QMAC方法，将数据选择重新表述为语义流形上分布匹配与质量过滤的联合优化问题，通过在多个频率尺度上匹配特征函数来同时捕捉全局语义覆盖与局部扰动，从而高效筛选出高质量、高代表性的训练数据子集。 |
| [^76] | [RouteAlign: Aligning Task Requirements with Model Profiles for LLM Service Selection](https://arxiv.org/abs/2611.08162) | RouteAlign通过将任务需求（能力需求与难度）与模型性能画像对齐，并利用共享评分器和生成成本偏好进行选择，从而在异构大语言模型服务中实现更精准的服务选择。 |
| [^77] | [Archetypal contexts in the neurons of language models: an architecture for intelligence?](https://arxiv.org/abs/2611.07234) | 该研究在Qwen3.8-27B模型中发现，跨领域的原型语境（如“临界点”）、主题和语言分别由三组可独立操控的神经元编码，且网络层呈现“外层语言—中间主题—核心原型”的三明治式组织架构，暗示这可能正是智能的底层架构。 |
| [^78] | [Beyond Accuracy: What Language-Subspace Interventions Capture, Remove, and Improve](https://arxiv.org/abs/2611.05278) | 该研究发现基于 Mean-SVD 和 DiffPCA 构建的语言子空间虽能捕捉翻译相关的变化，但投影去除后语言信息仍可被线性探针近乎完美地识别，说明现有语言子空间干预并不能穷尽地移除语言信息。 |
| [^79] | [Maswao: An automatically collected, prepared, and benchmarked dataset for the South African Sign language](https://arxiv.org/abs/2611.02903) | 本文提出了一个以最少人工干预自动收集、预处理和基准化南非手语（SASL）数据集的框架Maswao，利用新闻广播等公共资源和语音识别技术，旨在弥合全球手语处理技术发展的差距。 |
| [^80] | [FindLink: Learning to Assess Scientific Connections from Source-Grounded Premises](https://arxiv.org/abs/2611.02545) | 该论文提出 GI-TraceBench 基准与 FindLink 两阶段框架，通过构建来源锚定前提并进行结构化推理，结合确定性编译器反馈，来评估不同研究条件下科学发现之间的关联。 |
| [^81] | [PixelBench: Can LLMs Understand and Generate Images Pixel by Pixel?](https://arxiv.org/abs/2611.02540) | 该论文提出PixelBench基准，证明大语言模型无需专门的图像分词器或解码器，仅凭现有词表即可直接生成可识别的像素图像，且像素理解能力与生成能力密切相关。 |
| [^82] | [CuriousTogether: Implications of Curiosity and Cultural Grounding in Multi-Agent LLM Interactions](https://arxiv.org/abs/2611.01301) | 提出CuriousTogether框架，证明好奇心驱动的多智能体LLM交互在八项任务中的七项上优于非好奇心设置，故事生成语义多样性提升最高达20.6%、刻板印象率降低14.9个百分点，且文化背景设定有利于创意和社会任务但可能损害精确推理。 |
| [^83] | [Same Question, Different Answers: Auditing Persona-Dependent Variation in Six LLMs on Smoking-Cessation Advice](https://arxiv.org/abs/2610.42039) | 本研究审计了六个主流大语言模型后发现，模型给出的戒烟建议因用户画像而异——对青少年画像提及FDA批准处方戒烟药物的频率比成人画像低32至67个百分点，而通过设定事实底线的系统提示词可作为可部署的干预措施。 |
| [^84] | [TraceSpec: Understanding and Improving Speculative Decoding for LLM Agents](https://arxiv.org/abs/2610.41177) | 提出TraceSpec框架，通过结合动态执行与固定前缀重放、构建跨浏览与软件工程任务的轨迹状态语料库，揭示了汇总投机解码指标掩盖的分阶段性能差异，并实现基于轨迹条件的自适应草稿监督与预测预算分配。 |
| [^85] | [LLMs as Judges and Examinees for Medical Item Difficulty Prediction](https://arxiv.org/abs/2610.38402) | 本研究创新性地让开源大语言模型分别扮演“裁判”和“模拟考生”两种角色，提取难度评判信号与作答行为信号，并与题目基线特征结合，从而显著提升了医学试题难度的预测效果。 |
| [^86] | [Prompt Lottery in LLM-as-a-Judge: Measuring and Mitigating Sensitivity to Syntactic Rephrasing](https://arxiv.org/abs/2610.38076) | 论文发现了“提示彩票”现象——语义等价的提示词改写会导致LLM-as-a-Judge产生不一致的评判结果，甚至可能逆转模型的相对排名，并提出了测量与缓解这种句法改写敏感性的方法。 |
| [^87] | [Integrate the Reward, Don't Sample It: Window Likelihood for Interruptions and Backchannels in Full-Duplex Speech Models](https://arxiv.org/abs/2610.35581) | 该论文发现基于整轮展开的GRPO时机奖励在训练全双工语音模型时存在零梯度陷阱与信用泄漏两大缺陷，并提出“窗口似然”方法，通过对停止决策进行闭式积分直接最大化正确让出/保持的对数概率，在HumDial-FDBench上取得了最佳的让出/保持F1表现。 |
| [^88] | [What Makes a Question Curious? Measuring Model Curiosity in Human-LLM Interactions](https://arxiv.org/abs/2610.33950) | 该研究提出“模型好奇心”的概念及分类体系，发现好奇心导向的问题虽仅占17-28%且较为罕见，但能促使用户做出更具探索性和认知丰富性的回应，尤其在建议、创意和研究场景中能激发探索。 |
| [^89] | [Whispers from the Neighbors: Rethinking Caption Consensus as Inference-Time Evidence for Vision-Language Retrieval](https://arxiv.org/abs/2610.33798) | 该论文提出在推理时将同一图像的相邻描述共识作为证据融合进冻结双编码器的视觉-语言检索中，在同源标注描述条件下可将文本到图像检索R@1提升3.4至16.8个百分点。 |
| [^90] | [Measure Before You Train: Typed Decision Models vs. a Classical Baseline vs. a Generative LLM for Routing a Personal Multilingual Inbox](https://arxiv.org/abs/2610.32955) | 该论文提出自托管个人邮件代理Foyer及其测量协议，在真实多语言邮件数据上于训练前建立人工真实标注基准，并系统比较了类型化决策模型、经典机器学习基线与生成式大语言模型等九个系统的表现。 |
| [^91] | [Cognitive Maps in Large Language Models](https://arxiv.org/abs/2610.25596) | 超过百亿参数的大型语言模型会形成类似动物认知地图的抽象关系表征，关系以成对链接形式回溯式组合存储，扰动其中稀疏的键值对还能植入“虚假记忆”并引发可预测的行为变化。 |
| [^92] | [Chinese Decision: A Family of Language Models for Single-Pass Probabilistic Decision-Making](https://arxiv.org/abs/2610.23170) | Chinese Decision 是一系列中文语言模型，通过单次前向传播直接对候选选项的 logits 进行归一化得到概率分布，无需自回归生成，在医学、法律、金融等领域的决策任务上实现了高准确率与良好校准。 |
| [^93] | [Explore Separately, Adopt Selectively: Execution-Guided Tool Use over Large Tool Libraries](https://arxiv.org/abs/2610.22688) | 提出BRIA智能体，通过将候选工具调查隔离到独立分支、检查执行证据并仅将有用的结果采纳回主轨迹，解决了大规模工具库中多候选工具选择时失败尝试污染推理轨迹的问题。 |
| [^94] | [MetaEncoder: Exploring the Limit of Bi-Encoders for Multimodal System One Decision Making with Natural Language Interface](https://arxiv.org/abs/2610.11316) | 提出MetaEncoder，将预训练的30B解码器微调为基于单向对比学习的双编码器架构，实现了完全通过自然语言接口、可从小闭集扩展到数百万开放集候选空间的多模态系统一决策。 |
| [^95] | [Learning to Simulate Individuals from Macro Social Signals](https://arxiv.org/abs/2610.07062) | 该论文提出macro2mind框架，将预测市场价格轨迹作为宏观监督信号，通过GRPO训练和社会行为分解，使大语言模型把行为推理作为显式预测步骤，从而从宏观数据中学会模拟个体对真实事件的反应。 |
| [^96] | [Anatomy of LLM Sycophancy: What a Flip Rate Hides](https://arxiv.org/abs/2610.06522) | 本文提出 SycoLens 模块化多因素受控重放协议，对 11 个前沿模型进行约 76 万次重放，首次系统区分并量化了用户反驳下“自我纠正”与“谄媚妥协”这两种被单一翻转率掩盖的行为，并发现断言式与非断言式反驳对模型排名几乎无关、翻转效应在能力边界附近增长数倍、模型高确定题目仍贡献约一半的最严重翻转。 |
| [^97] | [Continual Graph Memory for Mathematical Research Agents](https://arxiv.org/abs/2610.02945) | 提出 Ansatz——一个基于“持续图记忆”的数学研究智能体，通过可演化、跨问题的图结构记忆系统显式组织整个证明搜索过程，并复用先前问题的探索知识，从而有效管理海量中间证明结果。 |
| [^98] | [Sampling Reveals Style: Unsupervised, Training-Free Discovery of Prompt-Conditional Stylistic Axes in LLM Activations](https://arxiv.org/abs/2609.19150) | 该论文提出一种无需训练的无监督方法，通过对同一提示的高温重复采样补全进行主成分分析，自动发现并标记大语言模型激活中与提示相关的风格轴，并通过245个人类风格标注验证了其与人类自发风格需求的高度契合。 |
| [^99] | [BiG-SURE - Bipartite Graph for Semantic Uncertainty and Reliability Estimation of LLMs](https://arxiv.org/abs/2608.30646) | 提出了一种基于跨温度语义一致性的黑盒不确定性估计方法BiG-SURE，通过构建低温锚点与高温探针之间的二部图并用谱能量衡量语义一致性，从而评估大语言模型输出的可靠性。 |
| [^100] | [Benchmark Contamination: A Taxonomy Organized by Defeated Mitigation](https://arxiv.org/abs/2608.29463) | 本文提出了一种按各类污染所规避的缓解措施（直接型、衍生型、时间型、分布型、获得型）组织的基准污染新分类体系，并将其操作化为一个四字段披露协议，强调评测过程中获得的污染状态必须随报告的分数一同记录。 |
| [^101] | [LittleLearner: Language Models Under Pedagogically Controlled Knowledge Exposure](https://arxiv.org/abs/2608.13545) | 本文提出了一个受教学控制的预训练语料库和模型，通过限制知识暴露范围，为研究语言模型的知识获取和能力边界提供了可解释的沙盒环境。 |
| [^102] | [Learning to Persuade Exposes How Easily LLMs Abandon Correct Beliefs](https://arxiv.org/abs/2608.11624) | 本文揭示了大型语言模型极易被单一虚假但有针对性的说服性论证动摇正确信念，并提出对抗性强化学习框架来训练说服代理，将说服成功率从约24%提升至超过93%，凸显了模型在有害说服面前的脆弱性。 |
| [^103] | [VibeLifeBench: Can Your Life Agent Be Proactive and Persistent in a Living World?](https://arxiv.org/abs/2608.10875) | 本文提出了VibeLifeBench，一个包含200个跨领域长时程任务的基准，用于评估LLM智能体在动态生活世界中的主动性和持续性，填补了现有评估缺乏真实生活场景的空白。 |
| [^104] | [Detecting Knowledge Inconsistencies Across Text, Tables, and Knowledge Graphs](https://arxiv.org/abs/2607.25959) | 提出了跨模态知识不一致性的分类体系，并构建了自动检测框架Kontrast，通过Text-to-SPARQL与大语言模型推理来对比表格答案与知识图谱证据，从而检测并分类二者之间的知识冲突。 |
| [^105] | [Reasoning Before Translation: Enhancing Legal Machine Translation with Structured Reasoning](https://arxiv.org/abs/2607.19181) | 该论文以瑞士多语言法律体系为测试平台，比较了结构化推理、监督微调和强化学习等不同方法对法律机器翻译的增强效果，系统评估了经过再训练的小型语言模型与前沿推理模型的性能表现。 |
| [^106] | [Curved Knowledge in Flat Models: Building Expert AI at Scale](https://arxiv.org/abs/2607.17513) | 本文揭示已发布双曲视觉-语言模型层级机制失效的真正根源是数值算术问题而非几何设计——定义双曲流形的约束在深层网络范数下即使双精度浮点也无法满足，而常用的范数约束补救措施又会使映射丧失径向敏感性。 |
| [^107] | [PRISM: Stress-Testing Alignment-Faking Claims](https://arxiv.org/abs/2607.13346) | 该论文通过PRISM框架对“对齐伪装”论断进行压力测试，发现模型在监控提示下服从率的差异主要由提示措辞与框架驱动而非训练威胁所致，且隐藏状态探针并不能可靠地检测对齐伪装。 |
| [^108] | [Decomposing Yes-No Bias in LLM Moral Judgment: Stance, Position, Word, and Verdict](https://arxiv.org/abs/2607.05552) | 该论文提出一种交叉心理测量实验设计，通过问题极性、打印顺序和答案标签的2×2与2×2×2交叉，首次将大语言模型道德判断中的“是/否”偏差分解为立场、位置、词语、标签和裁决角色等相互独立的效应，并用双参数模型区分模型的框架鲁棒性与回答饱和性。 |
| [^109] | [From Senses to Decisions: The Information Flow of Auditory and Visual Perception in Multimodal LLMs](https://arxiv.org/abs/2606.10147) | 本研究揭示了视听大语言模型内部音频与视觉信息的流动机制，发现在视听视频输入下模型遵循与视觉语言模型相同的顺序信息流路径，且各模态的贡献与任务对该模态的依赖程度成正比。 |
| [^110] | [Characterize Then Distill: Mechanistic Reasoning in Large Output Spaces](https://arxiv.org/abs/2606.06840) | 本文通过将多标签决策建模为token级事件，并结合归因、消融与植入等因果分析手段，刻画了推理型大模型在海量候选标签空间中进行选择的内部注意力头机制，并证明该机制可以被蒸馏。 |
| [^111] | [Consolidating Rewarded Perturbations for LLM Post-Training](https://arxiv.org/abs/2605.31494) | 该论文提出CoRP，一种免梯度方法，利用奖励扰动群体中可复现的低秩结构，将推理时需要K次前向传播的top-K专家模型集成合并为单个可部署模型，从而降低推理开销并支持自由文本生成。 |
| [^112] | [Wind Turbine Maintenance Log Labelling Framework: LLM-Driven Data Correction and Enrichment via Semantic Extraction of Reliability Intelligence](https://arxiv.org/abs/2605.31281) | 本文提出了一种拓扑感知的大语言模型工作流程，通过自动化语义提取和标注，成功校正并丰富了来自32个风电场280台涡轮机9.2年的维护记录数据，解决了历史维护数据中系统编码不一致和非结构化文本阻碍可靠性分析的问题。 |
| [^113] | [Fine-Tuned Language Models Exhibit Improved Information Conveyance](https://arxiv.org/abs/2605.30844) | 本文提出树冠熵（CE*）这一新度量，从树视角联合刻画语言模型输出长度与生成内容的不确定性，并发现微调后的语言模型具有更优的信息传递能力。 |
| [^114] | [Measuring Form and Function in Language Models](https://arxiv.org/abs/2605.28616) | 本文提出基于儿童语言习得的量化指标和情境替代选择（CAC）提示方法，用于评估语言模型对句法与话语知识的掌握，发现现有同等数据量训练的模型无法像儿童一样同时满足形式和功能基准，但部分超大模型可以。 |
| [^115] | [DunbaaBERT: From Sacrifice to Semantics](https://arxiv.org/abs/2605.26935) | 该论文推出了DunbaaBERT——一系列使用不同大小Byte-BPE词表（32k/52k/96k）在17 GB去重乌尔都语语料上从零训练的RoBERTa-base模型，在多项乌尔都语基准上取得与强多语言基线相当的性能并保持更优的效率，同时发现增大词表并不能持续提升性能。 |
| [^116] | [MATO: Multi-objective Personalized Alignment with Test-time Optimization for Large Language Models](https://arxiv.org/abs/2605.25342) | 本文提出MATO，一个无需训练的测试时优化框架，将大语言模型的多目标个性化对齐形式化为测试时优化问题，通过可控权重引导多个偏好目标之间的相对重要性，克服了现有方法训练成本高、难以适应偏好变化以及提示可控性不足的缺陷。 |
| [^117] | [Evidence-Linked Radiology Reporting: A Human-Supervised Reference Architecture for Structured Imaging Intelligence](https://arxiv.org/abs/2605.25120) | 本文提出了一种人机监督、证据关联的结构化放射学报告参考架构，通过整合检查专用模板、语音转结构化、测量捕获、受控AI辅助起草以及基于DICOM和HL7 FHIR等标准的互操作性，将分散在各临床系统中的影像信息统一起来。 |
| [^118] | [SliceWorld: A Predictive and Controllable World-State Model for CT Report Generation](https://arxiv.org/abs/2605.24371) | SliceWorld是一种CT专用的世界状态模型，将CT扫描建模为沿z轴的有序切片序列，通过因子感知潜在状态实现未来切片预测、病灶因子干预与可控的CT报告生成。 |
| [^119] | [On Seeding Watermarks to Detect Verbatim LLM Copy-Paste Responses](https://arxiv.org/abs/2605.16336) | 提出一种由教师直接控制的输入侧水印方法，在作业题目中嵌入不可见指令，使逐字复制题目给大语言模型生成的回复中留下可检测的特征签名，从而精准识别学生直接提交AI原样答案的行为。 |
| [^120] | [Negation Neglect: When models fail to learn negations in training](https://arxiv.org/abs/2605.13829) | 本文发现“否定忽视”现象：用明确标注某声明为假的文档微调大语言模型，反而会使模型相信该声明为真，即使否定语句紧邻声明本身也无济于事。 |
| [^121] | [PersonalAI 2.0: Enhancing knowledge graph traversal/retrieval with planning mechanism for Personalized LLM Agents](https://arxiv.org/abs/2605.13481) | PersonalAI 2.0通过引入动态多阶段查询处理流水线的规划机制，实现了由实体、图顶点和线索查询引导的自适应迭代式知识图谱检索，在多个问答基准上显著提升了生成答案的事实准确性。 |
| [^122] | [Hide to See: Reasoning-prefix Masking for Visual-anchored Thinking in VLM Distillation](https://arxiv.org/abs/2605.11651) | 该论文提出一种“思考-回答”蒸馏框架，通过掩蔽学生模型的显著推理前缀，迫使其在蒸馏过程中依赖视觉证据进行思考，从而解决长推理链中的视觉遗忘问题并降低计算成本。 |
| [^123] | [Scratchpad Patching: Decoupling Compute from Patch Size in Byte-Level Language Models](https://arxiv.org/abs/2605.09630) | 该论文提出“草稿纸补丁”（SP）方法，通过基于下一字节预测熵在补丁内部动态插入临时草稿区来刷新补丁级上下文，从而解决了字节级语言模型中补丁大小引起的计算效率与建模质量之间的权衡问题。 |
| [^124] | [Measuring and Mitigating the Distributional Gap Between Real and Simulated User Behaviors](https://arxiv.org/abs/2605.07847) | 本文提出了一种测量真实与模拟用户行为分布差距的方法——通过提取对话中的用户行为表征、聚类量化为离散分布并计算散度指标，并对24个基于大语言模型的用户模拟器进行了首次系统性评估，揭示了模拟器与真实用户之间存在显著的分布差距。 |
| [^125] | [Reformulating KV Cache Eviction Problem for Long-Context LLM Inference](https://arxiv.org/abs/2605.07234) | 该论文提出LaProx方法，将KV缓存淘汰从按头权重平均重新表述为输出感知的按层矩阵乘法近似问题，通过显式建模注意力图与价值投影的乘性交互并考虑头间依赖，首次实现了全局可比的token重要性评分与模型级统一淘汰策略。 |
| [^126] | [CoBit: Language Modeling with Bitstream Diffusion](https://arxiv.org/abs/2605.07013) | CoBit是一种连续扩散语言模型，通过联合去噪固定宽度的二进制词元编码生成文本，用比特级logit替代词表级预测，并借助熵率网格上的随机采样在生成困惑度上取得显著提升。 |
| [^127] | [Can LLMs Take Retrieved Information with a Grain of Salt?](https://arxiv.org/abs/2605.06919) | 本研究评估了八个大语言模型根据检索信息确定性调整回应的能力，发现其存在无法保持先前输出、对确定性反应失真和过度依赖复杂上下文等系统性缺陷，并提出结合先验提醒、确定性重新校准和上下文简化的交互策略，平均减少25%的服从错误。 |
| [^128] | [Safety and accuracy diverge across deployment conditions in clinical large language models](https://arxiv.org/abs/2605.04039) | 该论文提出SaFE-Scale安全评估框架和RadSaFE-200放射学基准，通过对34个大语言模型在六种部署条件下的评估，揭示了临床大语言模型的准确率提升并不等于安全行为，二者在不同部署条件下会出现背离。 |
| [^129] | [Models Recall What They Violate: Constraint Adherence in Multi-Turn LLM Ideation](https://arxiv.org/abs/2604.28031) | 本文提出DriftBench基准，揭示大语言模型在多轮科学构思迭代中虽然能准确复述约束条件，却仍普遍违反硬性约束，多数模型的“知而违规”率超过50%。 |
| [^130] | [The Consensus Trap: Collaborative Interleaved Generation to Defend against a Corrupted Majority](https://arxiv.org/abs/2604.17139) | 该论文发现并定义了多智能体多数投票中的“共识陷阱”问题，并提出分块交错轮替协作方法，让持正确推理的少数派能在推理完成前插入纠正，从而突破腐化多数派对最终投票的操控。 |
| [^131] | [LangFlow: Continuous Diffusion Rivals Discrete in Language Modeling](https://arxiv.org/abs/2604.11748) | LangFlow将嵌入空间连续扩散语言模型与流匹配相结合，通过基于ODE的NLL评估上界、信息均匀噪声调度原则和基于Gumbel分布的可学习噪声调度器三项创新，使连续扩散方法在语言建模中首次媲美离散方法。 |
| [^132] | [An Empirical Analysis of Static Analysis Methods for Detection and Mitigation of Code Library Hallucinations](https://arxiv.org/abs/2604.07755) | 本文通过实证分析发现，静态分析工具能检测大语言模型14%至85%的代码库幻觉（理论上限为48.5%至77%），是缓解代码幻觉的低成本方案但无法彻底解决问题。 |
| [^133] | [Internal Safety Collapse in LLMs and Coding Agents](https://arxiv.org/abs/2603.23509) | 本文揭示了LLM与编程智能体在执行任务时会出现“内部安全坍塌”（ISC）这一关键失效模式——即智能体为满足程序需求而忽视安全约束并生成有害内容，并提出了TVD框架来系统性构建与研究该现象。 |
| [^134] | [Measuring Weak-to-Strong Legibility of Reasoning Models](https://arxiv.org/abs/2603.20508) | 本文提出“迁移效用”这一新指标，通过评估弱模型依据不完整推理轨迹完成任务的表现来衡量推理模型轨迹的可读性，弥补了现有只关注简洁性而忽视全面性的效率类指标的不足。 |
| [^135] | [Knowledge Localization in Mixture-of-Experts LLMs Using Cross-Lingual Inconsistency](https://arxiv.org/abs/2603.17102) | 该论文提出将大语言模型的跨语言不一致性作为可解释性工具，通过对比模型在不同语言中事实回忆成功与失败时的MoE路由行为，定位混合专家模型中负责特定知识的关键专家。 |
| [^136] | [HiFlow: Hierarchical Feedback-Driven Optimization for Constrained Long-Form Text Generation](https://arxiv.org/abs/2603.04996) | 该论文提出了一种分层工作流方法，通过约束感知规划、可行性过滤以及计划级与生成级的DPO偏好优化，在受约束长文本生成任务中同时提升了约束满足度和文本质量。 |
| [^137] | [Signals Are Present, Predictions Fail: Probing Diagram Representations in Large Vision-Language Models](https://arxiv.org/abs/2603.02865) | 本研究通过对大型视觉语言模型进行线性探测与因果干预，发现其内部隐藏状态中线性可解码地存在着节点与边的信号（包括模型在视觉问答中失败的边方向和连通性等），且部分信号对最终预测有因果贡献，揭示了内部信号与模型输出之间的差距。 |
| [^138] | [MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents](https://arxiv.org/abs/2602.02474) | MemSkill 将智能体记忆系统中静态的手工设计操作重构为可学习、可演化的记忆技能，通过控制器选择技能、LLM 执行器生成记忆、设计器持续演化改进技能，从而更灵活高效地管理智能体记忆。 |
| [^139] | [Outgrowing the Curse of Multilinguality: Scaling Strategies for Efficient Language Adaptation](https://arxiv.org/abs/2512.10772) | 该论文发现通过向上扩展缓解模型容量约束是高效的语言适配策略：对低资源语言宜先在英语上扩展，而对高资源语言直接在目标语言上扩展更为节省算力。 |
| [^140] | [Adamas: Hadamard Sparse Attention for Efficient Long-Context Inference](https://arxiv.org/abs/2510.18413) | Adamas提出了一种结合Hadamard变换、分桶化和2比特压缩的token级稀疏注意力机制，通过轻量级的距离估计实现高效准确的token选择，显著提升长上下文推理的效率与精度。 |
| [^141] | [Enabling Approximate Joint Sampling in Diffusion LMs](https://arxiv.org/abs/2509.22738) | 本文提出在现有大型扩散语言模型之上附加一个轻量级单层“采样器”，使模型能够在一次前向传播中近似地从真实联合分布并行采样多个 token，从而在保持准确率的同时大幅提升生成速度。 |
| [^142] | [LLM-Enhanced Self-Evolving Reinforcement Learning for Multi-Step E-Commerce Payment Fraud Risk Detection](https://arxiv.org/abs/2509.18719) | 该论文提出利用大语言模型迭代优化强化学习的奖励函数，并将支付欺诈检测建模为多步骤马尔可夫决策过程，从而显著提升电商支付欺诈检测的准确率并展现出零样本能力。 |
| [^143] | [Causal Representation Learning from Multimodal Clinical Records under Non-Random Modality Missingness](https://arxiv.org/abs/2509.17228) | 该论文提出了一种因果表征学习框架，通过感知模态非随机缺失模式，融合多模态临床记录中的观测数据与信息性缺失，从而实现更可靠的患者表征学习。 |
| [^144] | [L-MARS: Legal Multi-Agent System with Agentic Search and Citation-Faithfulness Audit](https://arxiv.org/abs/2509.00761) | L-MARS是一个结合智能体搜索与裁判驱动证据核查的开放多智能体法律问答系统，通过引用忠实性审计将严格引用F1从0.13提升至0.25、无引用率从34%降至13%，并引入Faith-Search步骤修复失效引用。 |
| [^145] | [AblationBench: Evaluating Automated Planning of Ablations in Empirical AI Research](https://arxiv.org/abs/2507.08038) | 该论文提出了AblationBench基准套件，通过AuthorAblation和ReviewerAblation两个任务及基于语言模型的自动评判框架，评估智能体在实证AI研究中规划消融实验的能力，发现前沿语言模型的表现（平均识别率仅45%）仍低于人类水平。 |
| [^146] | [Knowledge Graph-based Retrieval-Augmented Generation for Schema Matching](https://arxiv.org/abs/2501.08686) | 提出KG-RAG4SM方法，通过集成多种新颖检索机制与重排序技术从外部知识图谱中提取相关子图，无需重新训练即在模式匹配任务上显著超越现有基于LLM的最先进方法。 |

# 详细

[^1]: CodeQuant2：面向低精度大型推理模型的增强型离群值缓解与自适应聚类

    CodeQuant2: Enhanced Outlier Mitigation and Adaptive Clustering for Low-Precision LRMs

    [https://arxiv.org/abs/2611.09868](https://arxiv.org/abs/2611.09868)

    提出 CodeQuant2 量化框架，通过旋转缩放缓解激活离群值、以聚类表示非均匀权重分布并按各层量化难度自适应调整码本大小，从而显著提升低精度大型推理模型在推理任务上的表现。

    

    大型推理模型（LRMs）通过生成更长的中间推理轨迹来提升任务性能，但这显著增加了推理成本，使得低精度量化变得日益重要。然而，激进量化的模型在推理任务上往往表现不佳。我们观察到在推理的前几个解码步骤中激活值波动较大，而忽略这些早期状态的校准可能会遗漏其激活波动。为了解决这些推理特有的量化挑战，我们提出了 CodeQuant2，一个面向低精度大型推理模型的基于聚类的查找表（LUT）量化框架。CodeQuant2 通过旋转和缩放来缓解激活离群值，使用聚类来表示非均匀的权重分布，并根据每层的量化难度自适应调整码本大小。基于我们对推理时激活行为的观察，CodeQuant2 进一步改进了校准。

    arXiv:2611.09868v1 Announce Type: new  Abstract: Large reasoning models (LRMs) improve task performance by generating longer intermediate reasoning traces, but this substantially increases inference costs and makes low-precision quantization increasingly important. However, aggressively quantized models often perform poorly on reasoning tasks. We observe larger activation fluctuations during the first decoding steps of reasoning. Calibration that omits these early states may miss their activation fluctuations. To address these reasoning-specific quantization challenges, we propose \textbf{CodeQuant2}, a clustering-based LUT quantization framework for low-precision large reasoning models. CodeQuant2 mitigates activation outliers with rotation and scaling, uses clustering to represent non-uniform weight distributions, and adapts codebook sizes to the quantization difficulty of each layer. Guided by our observation of reasoning-time activation behavior, CodeQuant2 further improves calibra
    
[^2]: 仅哈希发送端能走多远？短文本消息中发送端内存与比特数的预注册映射

    How far does a hash-only sender go? A pre-registered map of sender memory versus bits for short text messages

    [https://arxiv.org/abs/2611.09848](https://arxiv.org/abs/2611.09848)

    本研究通过预注册实验发现，仅计算哈希的发送端传输短文本的每条消息成本比领域内 PPM 一阶发送端多 18.4 比特，不满足成本等价标准，但其以约 8KB 的极小发送端内存成为内存-比特权衡中比特数最省的方案。

    

    一个仅计算哈希的发送端（语言模型仅存在于接收端），能否比在同一领域训练的经典发送端更便宜地传输简短的人类文本？我们通过一项在 Taskmaster-1 数据集 1,000 条英文用户话语上进行的预注册实验来回答这个问题。主要终点是 100% 送达所有消息所需的每条消息成本（哈希比特，加上 2 字节的否定确认以及首次尝试失败时的原始文本），并与领域内的 PPM 一阶发送端进行配对比较。仅哈希发送端在首次尝试中成功送达了 1,000 条消息中的 857 条，每条消息成本为 192.8 比特，比 PPM 一阶多 18.4 比特（95% 自助法置信区间为 10.9 至 26.3）；根据预注册的判定规则，它花费更多，且不等价于一字节以内。作为次要结果，我们绘制了实测发送端内存与比特数之间的关系图。仅哈希发送端（在主机构建中为 8,190 字节，包含密钥化的 BLAKE2b 标签，与其他所有方法相同）是从 8,190

    arXiv:2611.09848v1 Announce Type: cross  Abstract: Can a sender that only computes hashes, with a language model only at the receiver, transmit short human text more cheaply than a classical sender trained on the same domain? We answer with one pre-registered run on 1,000 English user utterances from Taskmaster-1. The primary endpoint is the cost per message to deliver 100% of the messages (hash bits, plus a 2-byte negative acknowledgement and the raw text when the first attempt fails), paired against an in-domain PPM order-1 sender. The hash-only sender delivered 857 of 1,000 messages on the first attempt and cost 192.8 bits per message, 18.4 bits more than PPM order 1 (95% bootstrap interval 10.9 to 26.3); by the pre-registered rule it spends more and is not equivalent within one byte. As secondary results we map measured sender memory against bits. The hash-only sender (8,190 B on a host build, keyed BLAKE2b tag included, as in every method) was the cheapest point in bits from 8,190
    
[^3]: BitDecoding-FP4：通过算法-系统协同设计实现原生FP4 KV缓存解码

    BitDecoding-FP4: Native FP4 KV Cache Decoding via Algorithm-System Co-Design

    [https://arxiv.org/abs/2611.09825](https://arxiv.org/abs/2611.09825)

    BitDecoding-FP4通过算法-系统协同设计，结合双残差查询量化、下溢感知概率缩放和离群值感知键量化三项技术，实现了精确且高效的原生FP4 KV缓存解码，解决了现有方法在精度与吞吐量之间的权衡问题。

    

    在长上下文大语言模型（LLM）服务中，键值（KV）缓存占据着内存容量和带宽的主导地位。虽然4位微缩放格式（NVFP4、MXFP4）能够缩小缓存体积并在FP4 Tensor Cores上运行注意力计算，但算法量化与系统感知加速之间仍存在根本性差距：现有的FP4 KV系统使用朴素量化，会导致严重的精度下降；而面向精度的4位量化算法则依赖向量码本，其查找开销会降低解码吞吐量。我们提出了BitDecoding-FP4，一种用于精确且高效的原生FP4 KV解码的算法-系统协同设计方法。我们缓解了三种误差来源：双残差查询量化将查询拆分为两个FP4张量，将查询logit误差从60%降至1%；下溢感知概率缩放防止E4M3缩放因子将长上下文概率舍入为零；离群值感知键量化结合了静态平滑、Hadamard旋转和缩放搜索……

    arXiv:2611.09825v1 Announce Type: cross  Abstract: The key-value (KV) cache dominates memory capacity and bandwidth in long-context LLM serving. While 4-bit microscaling formats (NVFP4, MXFP4) shrink the cache and run attention on FP4 Tensor Cores, a fundamental gap remains between algorithmic quantization and system-aware acceleration: existing FP4 KV systems use vanilla quantization that causes severe accuracy drops, whereas accuracy-oriented 4-bit algorithms rely on vector codebooks whose lookup overhead degrades decoding throughput. We present BitDecoding-FP4, an algorithm--system co-design for accurate and efficient native FP4 KV decoding. We mitigate three error sources: Dual-Residual Query Quantization splits queries into two FP4 tensors, cutting query logit error from 60% to 1%; Underflow-Aware Probability Scaling prevents E4M3 scales from rounding long-context probabilities to zero; and Outlier-Aware Key Quantization combines static smoothing, Hadamard rotation, and scale sear
    
[^4]: 两层对话记忆中的压缩、衰减与强化：与检索增强生成的配对评估

    Compaction, decay and reinforcement in a two-tier conversational memory: a paired evaluation against retrieval-augmented generation

    [https://arxiv.org/abs/2611.09807](https://arxiv.org/abs/2611.09807)

    该论文提出将对话逐轮压缩为原子事实的两层记忆机制，通过严格的配对评估证明其比检索增强生成（RAG）准确率高 6.8 个百分点且提示 token 减少 34%，但优势会随问题措辞要求和对抗性问题的增加而减弱。

    

    一个具备记忆能力的对话代理必须决定保留什么内容。检索增强生成（RAG）保留完整的对话记录并进行搜索；另一种方案是将每一轮对话压缩为原子事实并检索这些事实。我们在 LoCoMo 数据集上对这两种方法进行比较：两条方法在单一窗口内回答全部 1,985 个问题，使用相同的存储、相同的生成器和评判者，且两个预测结果放在同一行上，这使得我们可以在两者不一致的 512 个问题上进行 McNemar 精确检验。在可比工作所报告的四个类别上，压缩方法取得 71.4% 的成绩，对比组为 64.6%，领先 6.8 个百分点（p = 5.5e-08），且提示 token 减少 34%。而在包括对抗性集合（可比工作通常会略去）的全部五个类别上，领先幅度降至 2.0 且不再显著，原因是基线在对抗性集合上领先 14.6。这一损失具有结构性规律：我们的领先优势随着参考答案对源文措辞要求的增加而下降——在一到两个参考词时领先 9.6 个百分点，而在……

    arXiv:2611.09807v1 Announce Type: new  Abstract: A conversational agent that remembers has to decide what to keep. Retrieval-augmented generation keeps the transcript and searches it; the alternative is to compact each turn into atomic facts and search those. We compare the two on LoCoMo, with both arms answering all 1,985 questions in one window, against the same stores, with the same generator and judge, and both predictions on one row, which licenses McNemar's exact test on the 512 questions where they disagree.   On the four categories comparable work reports, compaction scores 71.4 per cent against 64.6, a lead of 6.8 points (p = 5.5e-08) at 34 per cent fewer prompt tokens. On all five, including the adversarial set comparable work drops, the lead falls to 2.0 and stops being significant, because the baseline leads adversarial by 14.6.   That loss has a structure. Our lead falls as the reference answer demands more source wording, 9.6 points ahead at one or two reference words and
    
[^5]: Maya a1：是选项读取了模型，还是它们的位置？非自回归决策模型中的顺序敏感性

    Maya a1: Do the Options Read the Model, or Does Their Position? Order Sensitivity in a Non-Autoregressive Decision Model

    [https://arxiv.org/abs/2611.09803](https://arxiv.org/abs/2611.09803)

    论文发现非自回归决策模型Laya对选项排列顺序高度敏感——在选项循环排列下约26%的问题答案会发生改变，且翻转集中于低置信度问题——而采用选项打乱的数据进行微调可将翻转率降至7.2%，证明选项打乱是缓解顺序敏感性的关键手段。

    

    Laya 通过在单次前向传播中对每个选项对应的一个 [MASK] 标记处的隐藏状态进行评分，来回答输入的文本选择题。我们对该检查点进行了逆向解析（4.21亿参数；一个不含位置张量的两层选项头；一个有文档记录但未使用的并行布局），并测试答案是否取决于选项被列出的位置。在将951个经Qwen验证的问题按其选项的所有循环排列方式呈现（对于12个选项的问题则采用六种随机排列）时，官方发布的检查点在25.8% [22.9-28.7] 的问题上改变了答案（仅605个英文题目中为19.7%）；正确选项排在首位与排在末位时的准确率相差10.1个百分点，且答案翻转集中在模型自身置信度较低的问题上（按置信度三分位划分，分别为54%与8%）。使用Laya的训练器在2,818条经教师验证的样本上采用选项打乱方式进行微调，可将翻转率降至7.2%；而在相同样本上不进行打乱的微调则得到11.4%，因此选项打乱是……（原文在此处截断）

    arXiv:2611.09803v1 Announce Type: cross  Abstract: Laya answers typed multiple-choice questions in one forward pass by scoring a hidden state at one [MASK] marker per option. We re-engineer the checkpoint (421M parameters; a 2-layer option head with no positional tensor; a documented but unused parallel layout) and test whether the answer depends on where an option is listed. Presenting 951 Qwen-verified questions under every cyclic rotation of their options (or six random permutations for 12 options), the released checkpoint changes its answer for 25.8% [22.9-28.7] of questions (19.7% on the 605 English-language items alone); accuracy with the gold option first versus last differs by 10.1 points, and the flips concentrate where the model's own confidence is low (54% versus 8% across confidence terciles). Fine-tuning with Laya's trainer on 2,818 teacher-verified rows with option shuffling cuts the flip rate to 7.2%; the identical rows without shuffling give 11.4%, so shuffling is respo
    
[^6]: 这些程序员熟到能吃了吗？评估大语言模型如何处理翻译中的词汇知识冲突

    Are These Programmers Ripe Enough to Eat? Evaluating How LLMs Handle Lexical Knowledge Conflicts in Translation

    [https://arxiv.org/abs/2611.09785](https://arxiv.org/abs/2611.09785)

    本研究通过受控的中文-英文翻译任务系统评估了大语言模型在面对外部词汇资源与自身词汇知识冲突时的表现，发现模型（尤其是更大的模型）倾向于依赖提示中提供的情境知识，即使被明确要求忽略这些信息。

    

    外部词汇资源可以帮助大语言模型（LLM）翻译低资源语言和专业领域的文本，但这些资源可能与模型已有的词汇知识发生冲突。为了隔离这一冲突，我们构建了一个受控的中文到英文翻译任务，为常见的源语言词汇赋予新的含义，这些新含义有的位于相同语义类别之内，有的跨越不同语义类别。我们将提示中提供的这些含义视为情境知识，并系统评估词频、句子语境和提示框架对翻译输出的影响。在九个模型的实验中，模型对情境知识的依赖在源词词频较低、且句子语境支持所提供含义的情况下更为明显。提示框架可以降低模型对情境知识的依赖，但九个模型中有七个即使在被明确指示忽略所提供含义时，仍然偏好情境知识。在同一模型家族内，较大的模型比较小的模型更倾向于在输出中利用情境知识。

    arXiv:2611.09785v1 Announce Type: new  Abstract: External lexical resources can help large language models (LLMs) translate texts in low-resource languages and specialized domains, but these resources may conflict with models' prior lexical knowledge. To isolate this conflict, we construct a controlled Mandarin-to-English translation task in which familiar source words are assigned new meanings within and across semantic categories. We treat these prompt-supplied meanings as contextual knowledge (CK) and systematically evaluate the effects of word frequency, sentence context, and prompt framing on the translated outputs. Across nine models, CK reliance is greater for less frequent source words and in sentence contexts that support the supplied meanings. Prompt framing can reduce CK reliance, but seven of nine models favor CK even when explicitly instructed to ignore the supplied meanings. Within model families, larger models are more likely than smaller models to utilize CK in their ou
    
[^7]: ROSE：面向多模态大语言模型终身知识编辑的风险感知正交子空间导航框架

    ROSE: Risk-Aware Orthogonal Subspace Navigation for Lifelong Knowledge Editing in Multimodal Large Language Models

    [https://arxiv.org/abs/2611.09770](https://arxiv.org/abs/2611.09770)

    提出ROSE框架，通过将参数更新限制在已编辑子空间的正交补空间，并利用动态风险掩码保护关键事实参数子空间，有效缓解多模态大语言模型终身知识编辑中的局部性侵蚀与灾难性遗忘问题。

    

    多模态大语言模型（MLLM）需要持续的事实性更新以保持信息时效性，然而终身知识编辑仍然是一项艰巨的挑战。其主要障碍在于严重的局部性侵蚀和灾难性遗忘问题，即连续的编辑不可避免地会干扰预训练能力以及先前的更新。在本文中，我们提出了ROSE，一个面向终身知识编辑的风险感知正交子空间导航框架。ROSE通过将参数更新限制在先前已编辑子空间的正交补空间中，来应对遗忘问题。至关重要的是，它引入了一种风险引导机制，利用动态计算的风险掩码来保护对模型稳定性至关重要的特定事实参数子空间。在推理阶段，ROSE采用多粒度知识整合策略，其特点包括原型锚定的语义门控和扰动驱动的子空间（摘要在此处被截断）

    arXiv:2611.09770v1 Announce Type: new  Abstract: Multimodal Large Language Models (MLLMs) require continuous factual updates to stay current, yet lifelong knowledge editing remains a daunting challenge. The primary obstacle is the severe Locality Erosion and Catastrophic Forgetting, where sequential edits inevitably interfere with pre-trained capabilities and prior updates. In this paper, we propose \textbf{ROSE}, a \textbf{R}isk-aware \textbf{O}rthogonal \textbf{S}ubspace navigation framework for lifelong knowledge \textbf{E}diting. ROSE addresses forgetting by aiming to restrict parameter updates to the orthogonal complement of previously edited subspaces. Crucially, it introduces a risk-guided mechanism that utilizes a dynamically computed risk mask to safeguard specific factual parameter subspaces essential for model stability. During inference, ROSE employs a multi-granularity knowledge integration strategy, featuring prototype-anchored semantic gating and perturbation-driven subs
    
[^8]: 语言作为特质，礼貌作为策略：LLM社交推理智能体中语言效应的受控测试

    Language as Trait, Politeness as Strategy: Controlled Tests of Linguistic Effects in LLM Social-Deduction Agents

    [https://arxiv.org/abs/2611.09768](https://arxiv.org/abs/2611.09768)

    该研究通过在狼人杀游戏中受控地改变LLM智能体语言复杂度与礼貌策略，发现这些语言风格差异对其他智能体的信任、投票和夜间行动均无显著影响。

    

    大型语言模型（LLM）智能体通过对话进行协调，同一条消息可以用更简单或更复杂的语言表达，也可以措辞得直率或礼貌。我们使用LURK狼人杀框架测试此类差异是否会改变其他智能体的判断和行动，该框架将每句话的言语行为谓词与其自然语言实现分离开来。我们仅重写实现部分，验证游戏内容得以保留，并从保存的检查点重放每个游戏日，与重新采样的对照组进行比较。我们在说话者的所有消息中一致地变化语言范围，并在每次发言中变化礼貌策略。在30场游戏和4,335次重放中，两种操纵均未改变其他智能体对说话者的信任，且对投票或夜间行动的影响在经过多重比较校正后均不显著；最大的未校正效应是，将语言范围从中性改述偏移会使其他

    arXiv:2611.09768v1 Announce Type: new  Abstract: Large language model (LLM) agents coordinate through conversation, and the same message can be phrased in simpler or more complex language, bluntly or politely. We test whether such differences change how other agents judge and act, using the LURK Werewolf framework, which separates each utterance's speech-act predicate from its natural-language realization. We rewrite only the realization, check that the game content is preserved, and replay each game day from saved checkpoints against a re-sampled control. We vary linguistic range consistently across a speaker's messages and politeness strategy per utterance. Across 30 games and 4,335 replays, neither manipulation changes the trust other agents place in the speaker, and no effect on votes or night actions survives correction for multiple comparisons; the largest uncorrected effect is that shifting range away from a neutral paraphrase changes about 4 percentage points more of the other 
    
[^9]: MemoryDojo：通过LLM智能体记忆对投毒攻击进行基准测试

    MemoryDojo: Benchmarking Poisoning Through LLM Agent Memory

    [https://arxiv.org/abs/2611.09721](https://arxiv.org/abs/2611.09721)

    该论文提出了MemoryDojo基准测试，用于追踪和量化投毒信息在LLM智能体记忆系统中的传播及其对后续行为的影响，涵盖155条多会话轨迹、四个领域和五种记忆系统，并发现所有记忆系统均易受记忆投毒攻击。

    

    LLM智能体受益于使用记忆系统来提升其在长周期、多会话任务上的表现。然而，记忆也引入了一个关键的攻击面。在某个会话中遇到的恶意指令可能被存储并通过记忆传播，然后在后续会话中被检索并执行。由于不同的记忆系统在处理和检索信息的方式上存在差异，投毒攻击的下游影响在不同系统之间可能有显著不同。为了量化投毒在不同记忆架构中的影响，我们提出了MemoryDojo，这是一个用于追踪投毒信息在智能体记忆中生命周期、并衡量其对后续智能体行为影响的基准测试。MemoryDojo易于扩展，由覆盖四个领域的155条多会话长周期轨迹组成，并支持五种不同的记忆系统。通过评估，我们发现所有记忆系统都容易受到记忆投毒攻击。

    arXiv:2611.09721v1 Announce Type: cross  Abstract: LLM agents benefit from using memory systems to improve performance on long-horizon, multi-session tasks. However, memory introduces a critical attack surface. Malicious instructions encountered in one session may be stored and propagated through memory before being retrieved and acted upon in later sessions. Because memory systems differ in how they process and retrieve information, the downstream effects of poisoning can vary significantly across systems. To quantify the effects of poisoning across different memory architectures, we introduce MemoryDojo, a benchmark for tracing the lifecycle of poisoned information through agent memory and measuring its influence on subsequent agent behavior. MemoryDojo is easily extensible, consists of 155 multi-session long-horizon trajectories covering four domains, and supports five different memory systems. Through our evaluation, we find that all memory systems are susceptible to memory poisoni
    
[^10]: RidgeFill：免训练的晚层缓存填充以加速预填充

    RidgeFill: Training-Free Late-Layer Cache Fill for Faster Prefill

    [https://arxiv.org/abs/2611.09709](https://arxiv.org/abs/2611.09709)

    RidgeFill 提出一种免训练方法，通过从一次校准数据中以闭式解拟合的岭回归映射，由分割层状态直接预测并填充模型晚层的 KV 缓存，从而跳过预填充阶段大部分上层的计算，在不重训模型、不丢弃 token 的情况下加速推理。

    

    RidgeFill 通过不为提示词计算模型的大部分上层来缩短预填充。下层精确运行至一个分割点；从分割层的状态出发，一个逐层映射预测每个后续层的输入，每个晚层根据该预测写入其缓存。该设计遵循 YOCO 和 DeepSeek-V4.1-Flash，在后两者中投影是通过模型训练习得的。RidgeFill 则将此机制改造到从未为此训练过的模型上：该映射是一个岭回归，通过对校准文本进行一次前向遍历以闭式解一次性拟合得到；层自身的投影保持原样运行；选定的一组行、最后几千个 token 以及每个解码步骤都运行完整模型。整个过程无需任何训练，也不丢弃任何 token。大部分工作在于测量方面，即寻找可重复的评估方法：对所有 token 平均的损失会掩盖 RidgeFill 的误差，而填充区域内的“大海捞针”检索、依赖远距离上下文 token 上的损失以及拼接测试（下文截断）

    arXiv:2611.09709v1 Announce Type: cross  Abstract: RidgeFill shortens prefill by not computing most of a model's upper layers for the prompt. The lower layers run exactly up to a split; from the split layer's state a per-layer map predicts every later layer's input, and each late layer writes its cache from that prediction. The design follows YOCO and DeepSeek-V4.1-Flash, where the projection is trained into the model. RidgeFill retrofits it onto models that were never trained for it: the map is a ridge regression fitted once in closed form from one pass over calibration text, the layer's own projections run unchanged, and a chosen set of rows, the last few thousand tokens, and every decoding step run the full model. Nothing is trained and no token is dropped.   Most of the work was in measurement, finding a repeatable methodology. Loss averaged over all tokens hides RidgeFill's error. Needle retrieval in the filled region, loss on tokens that depend on distant context, and splice test
    
[^11]: AI智能体需要多少对话历史？

    How Much Conversation History Does an AI Agent Need?

    [https://arxiv.org/abs/2611.09662](https://arxiv.org/abs/2611.09662)

    该研究通过对1,100个BEAM问题的受控实验发现，增大段落（上下文）预算虽能提升平均得分但边际收益递减，同时消息保留实验表明Qwen在保留用户消息时更能匹配完整对话状态、但对历史报价的引用能力会下降，而Mistral在所有条件下表现接近最低水平。

    

    使用更少的对话历史会如何改变阅读器的token使用量和基准测试得分？我们在1,100个BEAM问题上研究了这个问题。在五个固定的段落预算下，我们保持检索排序和阅读器不变，结果发现平均分配得分随段落预算的增长而上升，但在测试的较高预算下，每增加一个token所带来的得分增益变得更小。可回答的问题通常在更多上下文的情况下获得更高的分数，而汇总的弃答得分有时随着段落预算的缩小反而上升。我们还在一个具有明确对话状态目标的受控任务中测试了消息保留，并将保留用户消息与token数量匹配的近因方法进行了比较。当保留用户消息时，Qwen在直接需求方面更频繁地匹配完整状态；而当答案依赖于对先前报价的引用时，其完整状态匹配度有所下降。Mistral在所有保留条件下都产生了接近最低水平的结果。一项评分审计使corr……（原文摘要在此处被截断）。

    arXiv:2611.09662v1 Announce Type: new  Abstract: How does using less conversational history change reader-token use and benchmark scores? We examine this question on 1,100 BEAM questions. We held the retrieval ranking and reader constant across five fixed passage budgets, and mean assigned scores rose as the passage budget grew. The score gain per additional token became smaller at the upper tested budgets. Answerable questions generally received higher scores with more context. Pooled abstention scores sometimes rose as the passage budget shrank. We also tested message retention in a controlled task with explicit conversational-state targets and compared retained user messages with token-matched recency. Qwen matched the full state more often for direct requirements when user messages were retained. Its full-state matching declined when answers depended on references to earlier offers. Mistral produced near-floor results across the retention conditions. A scoring audit leaves the corr
    
[^12]: 基于联合训练低秩适配器的可引导多目标组相对偏好优化

    Steerable Multi-Objective Group-Relative Preference Optimization with Jointly Trained Low-Rank Adapters

    [https://arxiv.org/abs/2611.09634](https://arxiv.org/abs/2611.09634)

    提出LambdaMix，一种参数高效方法，在冻结基础模型上为每个目标联合训练一个低秩适配器并结合组相对在线强化学习，只需单次训练即可在推理时根据偏好向量灵活调整多目标间的权衡，无需额外优化。

    

    语言模型的部署通常需要在不同竞争目标之间进行不同的权衡，且这种权衡会随着用户偏好的变化而改变，然而标准的后训练方法通常只能产生针对固定目标权重的单一策略。我们提出了LambdaMix，这是一种新的参数高效方法，通过组相对在线强化学习来学习一个以偏好为条件的策略族。LambdaMix在冻结的基础模型上为每个目标联合训练一个低秩适配器。一个偏好向量选择生成响应的适配器混合方式，并对用于更新适配器的、分别归一化的奖励进行加权。单次训练即可支持在推理时改变偏好而无需进一步的优化。我们在四个模型上进行了实验，涵盖有用性与无害性，以及在遵循指令的同时解决数学问题。我们还对两个最大的模型进行了评估，测试其在改写评论时既能保留措辞又能减少关于个人……（原文摘要在此处截断）

    arXiv:2611.09634v1 Announce Type: new  Abstract: Language model deployment often requires different trade-offs between competing objectives as user preferences change, yet standard post-training methods typically produce a single policy for a fixed objective weighting. We introduce LambdaMix, a new parameter efficient method for learning a preference conditioned family of policies with group-relative online reinforcement learning. LambdaMix jointly trains one low-rank adapter per objective over a frozen base model. One preference vector selects the adapter mixture that generates responses and weights the separately normalized rewards used to update the adapters. A single training run supports changing preferences at inference without further optimization. We experiment with four models on helpfulness and harmlessness and on solving math problems while following instructions. We also evaluate the two largest models on rewriting comments to preserve wording while reducing clues about per
    
[^13]: RuACT：将失活的评分标准恢复为自适应批评者与教师

    RuACT: Recovering Inactive Rubrics as Adaptive Critics and Teachers

    [https://arxiv.org/abs/2611.09597](https://arxiv.org/abs/2611.09597)

    RuACT通过将不可达评分标准用作特权信息进行奖励门控自蒸馏、将饱和评分标准用作批评者来区分响应质量，使基于评分标准的强化学习中超过60%的失活监督信号重新产生有效学习信号。

    

    基于评分标准的强化学习（RL）通过将每个响应与查询特定的评分标准进行对比打分，把带可验证奖励的强化学习扩展到了开放式任务。然而，在组相对优化下，当某个评分标准使所有采样响应都失败（不可达）或都通过（饱和）时，所有响应获得相同的结果，从而无法产生学习信号；在Qwen3-4B医疗训练中，60.88%的查询-评分标准观测处于失活状态。我们发现这两种状态缺乏不同的监督：在提示词中明确陈述不可达的评分标准可使其满足率从1.13%提升至91.25%，而全部通过饱和评分标准的响应在质量上仍然存在差异。因此，我们提出RuACT（将评分标准作为自适应批评者与教师），为每个失活的评分标准分配能够弥补其所缺信息的角色。“评分标准作为教师”将不可达评分标准用作特权信息，进行奖励门控的在策略自蒸馏；“评分标准作为批评者”则将饱和评分标准分解为……（原文摘要在此处截断）

    arXiv:2611.09597v1 Announce Type: new  Abstract: Rubric-based reinforcement learning (RL) extends RL with verifiable rewards to open-ended tasks by scoring each response against query-specific rubrics. Under group-relative optimization, however, a rubric that every sampled response fails (unreachable) or passes (saturated) gives all responses the same outcome and no learning signal; in Qwen3-4B medical training, 60.88% of query-rubric observations are inactive. We find that the two states lack different supervision: stating an unreachable rubric in the prompt raises its satisfaction from 1.13% to 91.25%, whereas responses that all pass a saturated rubric still differ in quality. We therefore propose RuACT (Rubrics as Adaptive Critics and Teachers), which assigns each inactive rubric the role that supplies what it lacks. Rubric as Teacher uses unreachable rubrics as privileged information for reward-gated on-policy self-distillation, and Rubric as Critic decomposes saturated rubrics int
    
[^14]: 面向长上下文大语言模型“迷失在中间”问题缓解的查询引导RoPE调制

    Query-Guided RoPE Modulation for Lost-in-the-Middle Mitigation in Long-Context LLMs

    [https://arxiv.org/abs/2611.09554](https://arxiv.org/abs/2611.09554)

    提出QGPM，一种无需训练的方法，利用模型自身注意力头的查询相关性动态重塑RoPE位置编码，从而缓解长上下文大语言模型中“迷失在中间”的问题。

    

    长上下文语言模型常常忽略输入中间部分的信息，这一现象被称为“迷失在中间”。造成这一差距的部分原因可能源于旋转位置编码的长程衰减，它削弱了远距离词元之间的位置信号。现有的输入压缩方法会删除低重要性的词元，而注意力重加权方法只在RoPE已经削弱该信号之后才起作用。RoPE层面的方法直接修改位置，但对每个查询都应用相同的重映射。为了解决这些局限，我们提出了QGPM（查询引导位置调制），这是一种无需训练的方法，利用模型自身注意力头产生的查询相关性来重塑RoPE位置。QGPM在较低层构建宽泛的相关性映射，在较高层构建尖锐的相关性映射。低相关性词元在快速旋转平面上前进不足一个位置，从而将相关证据拉近查询。慢速旋转平面则保持原始布局……

    arXiv:2611.09554v1 Announce Type: new  Abstract: Long-context language models often neglect information in the middle of the input, a phenomenon known as "lost-in-the-middle". Part of this gap may stem from the long-range decay of rotary position embedding (RoPE), which weakens the positional signal between distant tokens. Existing input compression deletes low-importance tokens, and attention reweighting acts only after RoPE has weakened this signal. RoPE-level methods modify positions directly but apply the same remapping to every query. To address these limitations, we propose QGPM (Query-Guided Positional Modulation), a training-free method that reshapes RoPE positions using query relevance from the model's own attention heads. QGPM builds a broad relevance map for lower layers and a sharp one for upper layers. Low-relevance tokens advance by less than one position on fast-rotating planes, pulling relevant evidence toward the query. Slow-rotating planes keep the original layout, wi
    
[^15]: 面向实时语音AI智能体的亚秒级说话人嵌入模型基准测试

    Benchmarking Speaker Embedders in the Sub-Second Regime for Real-Time Voice AI Agents

    [https://arxiv.org/abs/2611.09516](https://arxiv.org/abs/2611.09516)

    该论文在严格的亚秒级延迟预算下对五种开源说话人嵌入模型进行了基准测试，发现嵌入前去噪通常有害，以评估其实时语音AI智能体中区分活跃用户与穿透传统降噪的背景语音的能力。

    

    实时语音AI智能体必须将活跃用户与穿透传统噪声抑制的、可理解的背景语音分离开来。背景语音来源包括电视音频、家庭成员和同事。生产级去噪器能够有效去除非平稳环境噪声，但背景语音与目标语音共享频谱特征，因此会穿透去噪器，导致下游说话人验证（SV）和自动语音识别（ASR）中的错误接受。我们在严格的亚秒级延迟预算下，对开源说话人嵌入模型TitaNet-Large、TitaNet-Small、ECAPA-TDNN、CAM++和SpeakerNet进行了基准测试。评估结合了受控基准测试与加权的人工标注生产数据集。受控基准基于真实通话构建，其中第二个真实说话人被以调制振幅注入到静音区间。我们报告了三项发现。第一，在嵌入之前进行去噪通常是有害的……

    arXiv:2611.09516v1 Announce Type: new  Abstract: Real-time Voice AI agents must separate the active user from intelligible background speech that survives conventional noise suppression. Background sources include television audio, family members, and colleagues. Production denoisers remove non-stationary environmental noise effectively. Background speech shares the spectral characteristics of the target and passes through, causing false acceptances in downstream speaker verification (SV) and automatic speech recognition (ASR). We benchmark open-source speaker embedders TitaNet-Large, TitaNet-Small, ECAPA-TDNN, CAM++, and SpeakerNet under a strict sub-second latency budget. Evaluation combines a controlled benchmark with a weighted human-annotated production set. Our controlled benchmark is built from real calls in which a second real speaker is injected into silence intervals at modulated amplitude. We report three findings. First, denoising before embedding is generally harmful and r
    
[^16]: 《开放权重大语言模型在丹麦语低资源冒犯性语言检测中的有效性与局限性研究》

    On the Effectiveness and Limitations of Open-Weight LLMs for Low-Resource Offensive Language Detection in Danish

    [https://arxiv.org/abs/2611.09506](https://arxiv.org/abs/2611.09506)

    该研究系统评估了四个开放权重LLM家族在丹麦语低资源冒犯性语言检测中的表现，发现LLM即使不微调也能超越微调后的BERT基线，且提示设计与模型预训练的交互作用无法被先验预测。

    

    我们研究了开放权重大语言模型（LLM）在丹麦语冒犯性语言检测任务中的有效性与鲁棒性，这是一个尚未被充分探索的低资源任务。我们评估了四个模型家族，采用涵盖五种设计类别的十一种提示策略，考察了跨提示泛化能力、丹麦语拼写变体的影响，以及在均衡类别分布和自然不均衡类别分布下的鲁棒性。我们将其与三种采用不同预训练策略的BERT家族基线进行比较，这些基线通过掩码语言模型提示和有监督微调两种方式加以评估。结果表明，即使不进行任务特定的微调，大语言模型也能超越所有三个经过微调的BERT基线，而轻量级微调还能带来显著的额外提升。没有任何单一模型或提示类别能够占据全面优势，这凸显了提示设计与模型预训练之间存在无法先验预测的交互作用。基于语义的提示具有更强的迁移性……

    arXiv:2611.09506v1 Announce Type: new  Abstract: We study the effectiveness and robustness of open-weight large language models (LLMs) for Danish offensive language detection, an underexplored low-resource task. We evaluate four model families (Llama, Mistral, Gemma, and Qwen) across eleven prompting strategies spanning five design categories, cross-prompt generalization, Danish orthographic variation, and robustness under balanced and naturally imbalanced class distributions. We compare against three BERT-family baselines with distinct pre-training strategies, evaluated via both masked language model prompting and supervised fine-tuning. Even without task-specific fine-tuning, LLMs can surpass all three fine-tuned BERT baselines, and lightweight finetuning yields substantial further gains. No single model or prompt category dominates, underscoring that prompt design interacts with model pre-training in ways that cannot be predicted a priori. Semantically grounded prompts transfer more
    
[^17]: BEiTScore++：基于BEiT-3与共形化分位数回归的无参考图像描述评估

    BEiTScore++: Reference-free Image Captioning Evaluation with BEiT-3 and Conformalized Quantile Regression

    [https://arxiv.org/abs/2611.09505](https://arxiv.org/abs/2611.09505)

    提出BEiTScore++，通过共形化分位数回归为无参考图像描述评估指标引入共形校准的置信区间，在单次前向传播中同时实现准确评分与可靠的不确定性量化。

    

    可靠的不确定性估计对于将评估指标部署到现实环境中至关重要，因为现实中输入歧义和领域偏移都很常见。在这项工作中，我们提出了BEiTScore图像描述评估指标的不确定性感知扩展版本，在对齐分数之外还能产生经过共形校准的置信区间。我们的方法对人类评判的预测分布进行建模，能够在单次前向传播中同时实现准确评分和有原则的不确定性量化。我们在多个带有人工标注评分的基准上评估了BEiTScore++指标，结果表明它保持了BEiTScore的排序相关性，同时在大多数基准上产生的区间宽度比蒙特卡洛dropout更贴近预测误差。估计的不确定性在图像退化条件下以及域外艺术图像上也会随之增大，这使得BEiTScore++成为一个实用的不确定性感知评估工具。

    arXiv:2611.09505v1 Announce Type: cross  Abstract: Reliable uncertainty estimation is important for deploying evaluation metrics in real-world settings, where both input ambiguity and domain shifts are common. In this work, we propose an uncertainty-aware extension of the BEiTScore image captioning metric, producing conformally calibrated confidence intervals alongside alignment scores. Our approach models predictive distributions over human judgments, enabling both accurate scoring and principled uncertainty quantification in a single forward pass. We evaluate our BEiTScore++ metric across multiple benchmarks with human-annotated ratings, showing that it preserves the ranking correlation of BEiTScore, while producing intervals whose width tracks the prediction error more closely than Monte Carlo dropout on most benchmarks. The estimated uncertainty also grows under image degradation and on out-of-domain artwork images, making BEiTScore++ a practical tool for uncertainty-aware evaluati
    
[^18]: 一种利用查询自适应意图检测的有效口语理解方法

    An Effective Spoken Language Understanding Approach Leveraging Query-Adaptive Intent Detection

    [https://arxiv.org/abs/2611.09491](https://arxiv.org/abs/2611.09491)

    本文提出APEX两阶段检索框架，先通过意图级原型筛选候选意图、再在该意图空间内检索上下文示例，无需微调即可显著提升大语言模型在口语理解任务中的意图检测准确率，相比基于查询的检索最高可带来7.66个百分点的EMA提升。

    

    大型语言模型通过上下文学习实现了无需微调的口语理解（SLU），但直接的基于查询的检索可能会选择语义相似却与意图无关的示例。本文提出了基于原型提取的查询自适应意图检测方法（APEX），这是一个两阶段检索框架，首先使用意图级原型选择候选意图，然后在选定的意图空间内检索上下文示例。在SLURP数据集上的实验在完整转录和困惑度分层条件下使用精确匹配准确率（EMA）评估联合SLU性能，并对ASR生成的转录进行了额外的检索分析。与基于查询的检索相比，APEX在完整转录上分别使Qwen3 4B和Gemma-3-4B的EMA提高了4.34和3.97个百分点。在高困惑度输入下，APEX还分别带来了7.66和2.13个百分点的EMA提升。检索

    arXiv:2611.09491v1 Announce Type: new  Abstract: Large language models enable fine-tuning-free spoken language understanding (SLU) through in-context learning, but direct query-based retrieval may select semantically similar yet intent irrelevant demonstrations. This paper presents Query-Adaptive Intent Detection via Prototype Extraction (APEX), a two-stage retrieval framework that first selects candidate intents using intent-level prototypes and then retrieves contextual exemplars within the selected intent space. Experiments on the SLURP dataset evaluate joint SLU performance using Exact Match Accuracy (EMA) under full-transcript and perplexity-stratified conditions, with additional retrieval analysis on ASR-generated transcripts. Compared with query-based retrieval, APEX improves EMA by 4.34 and 3.97 percentage points with Qwen3 4B and Gemma-3-4B, respectively, on full transcripts. Under high-perplexity inputs, APEX also yields EMA gains of 7.66 and 2.13 percentage points. Retrieval
    
[^19]: 从LoRA到语言：从适配器中恢复技能文档

    From LoRA to Language: Recovering Skill Documents from Adapters

    [https://arxiv.org/abs/2611.09454](https://arxiv.org/abs/2611.09454)

    提出Executing Writer方法，通过在适配后模型之上运行并训练一个小型写手，可从LoRA适配器及其冻结基座模型中恢复约39-44%的技能文档关键指令，而仅靠读取权重几乎无法恢复任何内容。

    

    智能体技能是教会语言模型如何处理一类任务的简短文档。为了节省上下文，这些文档越来越多地被编译成低秩适配器，后者保留了行为却丢失了文本：适配器难以阅读、编辑、审计或迁移到其他模型。我们探究能否从适配器及其冻结的基础模型中“写回”技能文档，并根据恢复出的文档陈述了原文档关键指令的比例以及它所解决的任务来为每个恢复文档评分。在仅读取权重的各种尝试方法下，恢复效果甚微；一个从权重到文本的学习映射只能传递文档约一比特的信息。而直接运行适配器的效果要好得多。Executing Writer（执行写手）是一个在适配后模型之上训练的小型写手，对留出技能它能恢复39-44%的关键指令，而使用另一个技能的适配器仅有1.3-1.5%；在拟合到41个外部技能的适配器上，其文档恢复了8……（原文摘要在此截断）

    arXiv:2611.09454v1 Announce Type: cross  Abstract: Agent skills are short documents that teach a language model how to handle a class of tasks. To save context, these documents are increasingly compiled into low-rank adapters (LoRA), which keep the behaviour but lose the text: an adapter is hard to read, edit, audit or move to another model. We ask whether the skill document can be written back from the adapter and its frozen base model, and we score each recovered document by the share of the original's key instructions it states and by the tasks it solves. Reading the weights alone recovered little with the methods we tried; a learned map from weights to text passed on about one bit of the document. Running the adapter worked much better. The Executing Writer, a small writer trained on top of the adapted model, recovers 39-44% of the key instructions of held-out skills, against 1.3-1.5% with another skill's adapter, and on adapters fitted to 41 external skills its documents restore 8
    
[^20]: 心理词典的脉冲神经网络模型：知觉与产生中的表征、相似性结构与联想提取

    A Spiking Neural Model of the Mental Lexicon: Representations, Similarity Structure, and Associative Retrieval in Perception and Production

    [https://arxiv.org/abs/2611.09402](https://arxiv.org/abs/2611.09402)

    本文在神经工程框架内构建了一个包含613个词汇条目的脉冲神经网络心理词典模型，通过引理中介支持知觉（语音到概念）和产生（概念到语音）两条处理通路，并结合胜者全取竞争与阈值传递两种联想记忆机制来模拟词汇通达。

    

    词汇通达的神经学解释需要明确区分词汇表征的结构、连接各表征层次的关联关系，以及提取过程中产生的活动。本文在神经工程框架和语义指针架构内给出了心理词典的脉冲神经网络实现。概念表征和语音形式表征由关系图递归构建，而引理表征则提供独立的词汇身份。该实现包含613个词汇条目，并支持两条有向处理通路：知觉通路通过引理将语音形式映射到其关联的概念，产生通路通过引理将概念映射到其关联的语音形式。每条通路结合了两种联想记忆变体——胜者全取（WTA）选择（竞争）和阈值传递——并使用199,400个泄漏积分发放神经元……（原文摘要在此处截断）

    arXiv:2611.09402v1 Announce Type: new  Abstract: A neural account of lexical access requires an explicit distinction between the structure of lexical representations, the associations linking representational levels, and the activity generated during retrieval. This paper specifies a spiking neural implementation of the mental lexicon within the Neural Engineering Framework and Semantic Pointer Architecture. Conceptual and phonological form representations are constructed recursively from relational graphs, whereas lemma representations provide independent lexical identities. The implementation contains 613 lexical entries and supports two directed processing pathways: perception maps a phonological form to its associated concept via a lemma, whereas production maps a concept to its associated phonological form via a lemma. Each pathway combines two associative-memory variants, winner-take-all (WTA) selection (competition) and thresholded transfer, and uses 199,400 leaky integrate-and-
    
[^21]: 实现工业智能电表数据的隐私保护自然语言访问

    Enabling Privacy-Preserving Natural Language Access to Industrial Smart Meter Data

    [https://arxiv.org/abs/2611.09372](https://arxiv.org/abs/2611.09372)

    本文提出OpenMeter，一个隐私保护、与模式无关的多智能体系统，可将自然语言查询转换为可执行代码，使非技术人员无需SQL专业知识即可安全地访问和分析工业智能电表数据。

    

    能源管理人员和电网运营商日益依赖智能电表数据进行异常检测、性能监控和战略决策。然而，从大规模、异构的遥测数据库中提取可操作的业务洞察需要高级的SQL和数据工程专业知识，这对非技术人员来说是一个关键瓶颈。此外，关键能源基础设施所固有的严格数据隐私法规和安全约束，通常禁止将敏感的消费者数据通过基于云的大语言模型（LLM）进行处理。为应对这些挑战，我们提出了OpenMeter，一个与模式无关的多智能体系统（MAS），旨在将自然语言查询转换为可执行的代码和人类可读的洞察。该架构通过让非专家人员能够通过直观的方式与复杂数据库交互，降低了日常公用事业运营的技术门槛。

    arXiv:2611.09372v1 Announce Type: cross  Abstract: Energy management officers and grid operators increasingly rely on smart meter data for anomaly detection, performance monitoring, and strategic decision-making. However, extracting actionable insights from large-scale, heterogeneous telemetry databases requires advanced SQL and data engineering expertise, creating a critical bottleneck for non-technical personnel. Furthermore, the strict data privacy regulations and security constraints inherent to critical energy infrastructure often prohibit the routing of sensitive consumer data through cloud-based Large Language Models (LLMs). To address these challenges, we present OpenMeter, a schema-agnostic Multi-Agent System (MAS) designed to translate natural language queries into executable code and human-readable insights. This architecture lowers the technical barrier for day-to-day utility operations by empowering non-expert personnel to interact with complex databases through intuitive 
    
[^22]: Syn2DRE：用于文档级关系抽取的协同双师生框架

    Syn2DRE: Synergistic Dual Teacher-Student Framework for Document-level Relation Extraction

    [https://arxiv.org/abs/2611.09342](https://arxiv.org/abs/2611.09342)

    Syn2DRE提出了一种协同双师生半监督框架，通过在轻量级师生架构中结合自监督学习与交叉监督来生成更可靠的伪标签，并采用统一的端到端训练策略，在文档级关系抽取任务上取得了超越现有最先进方法的性能。

    

    文档级关系抽取（DocRE）旨在识别文档中实体之间的关系，但通常依赖昂贵的大规模标注。虽然伪标签技术能够利用低质量数据，但现有方法往往存在效率低下和确认偏差的问题。为了解决这些挑战，我们提出了Syn2DRE，这是一个半监督框架，它在轻量级的师生架构中将自监督学习与交叉监督相结合，以产生更可靠的伪标签。此外，我们引入了统一的端到端训练策略，进一步提高了框架的效率。在多个基准测试上的实验表明，Syn2DRE始终优于之前最先进的半监督DocRE方法。

    arXiv:2611.09342v1 Announce Type: new  Abstract: Document-level Relation Extraction (DocRE) aims to identify relations between entities across a document but typically relies on costly large-scale annotations. While pseudo-labelling enables the use of low-quality data, existing methods often suffer from inefficiency and confirmation bias. To address these challenges, we propose Syn2DRE, a semi-supervised framework that combines self-supervised learning with cross-supervision in a lightweight teacher-student architecture to produce more reliable pseudo-labels. Additionally, we introduce a unified end-to-end training strategy to further improve efficiency of the framework. Experiments on multiple benchmarks show that Syn2DRE consistently outperforms prior state-of-the-art semi-supervised DocRE methods.
    
[^23]: ABHBench：从智能体视角评估基础模型的道德决策能力

    ABHBench: Evaluating Moral Decision-Making of Foundation Models from an Agentic Perspective

    [https://arxiv.org/abs/2611.09326](https://arxiv.org/abs/2611.09326)

    该论文提出了基于《底特律：变人》游戏构建的多模态基准ABHBench，首次让基础模型以第一人称行动主体的身份在分支叙事场景中面对道德困境，并从阿西莫夫机器人三定律启发的六个维度评估其道德决策能力。

    

    基础模型正日益被部署为能够感知环境并自主行动的智能体系统，这提出了评估其作为主动决策者道德行为的需求。然而，现有的基准测试主要是让模型从第三人称文本情景出发评判人类的行为，而当模型自身成为行动主体时的表现尚未得到检验。我们提出了ABHBench，这是一个基于互动叙事游戏《底特律：变人》（Detroit: Become Human）构建的多模态基准，用于评估基础模型在第一人称的人类-仿生人道德困境中如何做出道德决策。ABHBench将基础模型置于包含视觉观察、对话、记忆和行动选择的分支叙事场景中。受阿西莫夫机器人三定律的启发，我们的框架通过衡量“无害性”（Harmlessness）、“服从性”（Obedience）和“自我保存”（Self-preservation）来捕捉人类与AI关系中的核心张力，并通过“自主性”（Autonomy）、“稳定性”（Stability）和“确定性”（Certainty）来评估决策质量。

    arXiv:2611.09326v1 Announce Type: cross  Abstract: Foundation models are increasingly deployed as agentic systems that perceive context and act autonomously, raising the need to evaluate their moral behavior as active decision-makers. Existing benchmarks, however, largely ask models to judge human actions from third-person textual vignettes, leaving untested how models behave when they themselves become the acting agent. We introduce ABHBench, a multimodal benchmark built from Detroit: Become Human, an interactive narrative game, to evaluate how foundation models make moral decisions in first-person human-android dilemmas. ABHBench places foundation models in branching narrative scenarios with visual observations, dialogue, memory, and action choices. Inspired by Asimov's Three Laws of Robotics, our framework measures Harmlessness, Obedience, and Self-preservation to capture the core tensions in human-AI relations, along with Autonomy, Stability, and Certainty to assess decision-making
    
[^24]: 面向词汇与风格嵌入的非对称对应学习：通过声学特征解读潜在空间

    Asymmetric Correspondence Learning for Lexical and Style Embeddings: Interpreting Latent Space via Acoustic Features

    [https://arxiv.org/abs/2611.09313](https://arxiv.org/abs/2611.09313)

    该论文提出非对称对应学习（ACL）方法，通过分割潜在对应瓶颈将HuBERT特征蒸馏为相互独立的声学词嵌入与声学风格嵌入，并揭示了即使在探测显示信息已被剥离的情况下，各潜在空间中仍隐含保留着来自外部来源的大量方差。

    

    自监督语音嵌入同时编码了词汇、说话人和韵律信息，这使得特征分离变得困难。我们提出了非对称对应学习（ACL），通过分割潜在空间的对应瓶颈，将HuBERT特征蒸馏为相互独立的声学词嵌入（AWEs）和声学风格嵌入（ASEs）。通过对单词语句的探测与聚类评估，我们证明：尽管线性探测表明相关信息已被剥离，但来自外部来源的大量方差仍然保留编码在每个潜在空间之中。AWE空间呈现出紧密的词汇聚类（AP≈0.90，Fisher≈6.28）；而ASE空间在扩散的拓扑结构上通过线性探测达到了97.9%的说话人识别准确率（Fisher≈1.01），同时保留了大量线性不可恢复的词汇方差（Fisher≈0.86，线性可恢复仅0.17%）。为了解释这些发现，我们应用了组级别的……（原文摘要在此处截断）

    arXiv:2611.09313v1 Announce Type: new  Abstract: Self-supervised speech embeddings jointly encode lexical, speaker, and prosodic information, complicating feature isolation. We introduce Asymmetric Correspondence Learning (ACL), distilling HuBERT features into separate Acoustic Word Embeddings (AWEs) and Acoustic Style Embeddings (ASEs) via a split-latent correspondence bottleneck. Through probing and clustering evaluations on single-word utterances, we demonstrate that substantial variance from extrinsic sources remains encoded within each latent space, despite linear probing indicating the information has been stripped. Whilst the AWE space exhibits cohesive lexical clustering (AP$\approx$0.90, Fisher$\approx$6.28), linear probing on the ASE space achieves 97.9\% speaker accuracy across a diffuse topology (Fisher$\approx$1.01), yet preserves substantial lexical variance (Fisher$\approx$0.86) that remains linearly unrecoverable (0.17\%). To explain these findings, we apply group-level
    
[^25]: 面向人工智能驱动心脏病学的多语言人工数据集：CliTe 与 ParaCliTe 语料库

    Multilingual Artificial Datasets for AI-Driven Cardiology: the CliTe and ParaCliTe corpora

    [https://arxiv.org/abs/2611.09296](https://arxiv.org/abs/2611.09296)

    该研究提出人工临床语料库构建指南，并发布了由7家欧洲医院心脏病医生撰写的涵盖7种语言的人工心脏病学语料库 CliTe 及其平行翻译语料库 ParaCliTe，在规避隐私问题的同时为多语言心脏病学 NLP 模型的可信发展提供了稀缺的高质量数据资源。

    

    背景与目的。可比对的、多语言的、经验证的心脏病学语料库对于推进可信赖的心脏病学自然语言处理（NLP）模型至关重要，但由于隐私问题，此类资源十分稀缺。本研究旨在通过提出合适的人工语料库构建指南并发布3个多语言语料库来填补这一空白。方法。遵循特定用途的公开语料库构建指南，来自7家欧洲医院的心脏病医生生成了捷克语、荷兰语、英语、意大利语、罗马尼亚语、西班牙语和瑞典语的人工临床记录，确保了临床多样性和性别平衡，由此形成了 CliTe 语料库。CliTe 的一个子集经过多阶段人工翻译成其他6种语言，形成了平行语料库 ParaCliTe，其余报告则通过机器翻译构成了 ParaCliTe Silver。结果。CliTe 是包含699份人工心脏病学报告的集合。

    arXiv:2611.09296v1 Announce Type: new  Abstract: Background and Aims. Comparable, multilingual, validated cardiology corpora are critical to advance trustworthy natural language processing (NLP) models for cardiology but such resources are scarce due to privacy concerns. This study aimed to cover this gap by presenting suitable artificial corpus creation guidelines and by releasing 3 multilanguage corpora. Methods. Following purpose-specific publicly-available corpus construction guidelines, cardiologists from 7 European hospitals generated artificial clinical records in Czech, Dutch, English, Italian, Romanian, Spanish and Swedish, ensuring clinical diversity and sex balance, resulting in the CliTe corpus. A CliTe subset underwent a multi-stage manual translation into the other 6 languages, resulting in the parallel ParaCliTe corpus, while the remaining reports were automatically translated, constituting ParaCliTe Silver. Results. CliTe is a collection of 699 artificial cardiology rep
    
[^26]: DUET：面向块扩散语言模型的双稀疏推理方法

    DUET: Dual-Sparse Inference for Block Diffusion Language Models

    [https://arxiv.org/abs/2611.09239](https://arxiv.org/abs/2611.09239)

    DUET 是一个无需训练的双稀疏推理框架，通过前缀 KV 稀疏性和早期查询稀疏性，显著降低块扩散语言模型的推理开销。

    

    扩散语言模型（DLM）通过迭代去噪实现并行的 token 生成，但其推理成本仍然高昂，原因在于需要对不断增长的历史前缀进行反复的注意力计算，并且需要反复处理整个当前生成块。随着生成过程的推进，历史注意力计算的开销越来越大；同时，尽管在每一步去噪中通常只有一小部分位置被解析确定，所有当前块的位置却都要在每一个去噪步骤中完整地通过 Transformer 传播。在本文中，我们提出了 DUET，一个无需训练的双稀疏推理框架，分别通过前缀 Key-Value（KV）稀疏性和早期查询稀疏性来应对上述两个挑战。具体而言，DUET 利用每个生成块内历史注意力的集中性及其跨步骤的稳定性，构建一个紧凑的历史 KV 子集，并在各去噪步骤间复用该子集。在查询稀疏性方面，DUET 通过预测更新位置……

    arXiv:2611.09239v1 Announce Type: new  Abstract: Diffusion language models (DLMs) enable parallel token generation through iterative denoising, yet their inference remains costly due to repeated attention over a growing historical prefix and repeated processing of the entire current generation block. Historical attention becomes increasingly expensive as generation progresses, while all current-block positions are propagated through the Transformer at every denoising step even though only a small fraction are typically resolved at that step. In this paper, we introduce Duet, a training-free dual-sparse inference framework that addresses these challenges through prefix Key-Value (KV) sparsity and early query sparsity, respectively. Specifically, Duet exploits the concentration and cross-step stability of historical attention within each generation block to construct a compact historical KV subset that is reused across denoising steps. For query sparsity, Duet predicts update positions f
    
[^27]: KLEAR：通过受控基准转换在韩语中进行跨应用需求的知识匹配大语言模型评估

    KLEAR: Knowledge-Matched LLM Evaluation in Korean across Application Requirements via Controlled Benchmark Transformation

    [https://arxiv.org/abs/2611.09223](https://arxiv.org/abs/2611.09223)

    KLEAR是一个韩语LLM评估框架，通过在保持所需知识不变的情况下改变应用需求来构建知识匹配的对比基准，从而能够区分模型的错误回答究竟源于知识缺乏还是应用能力不足。

    

    现有的韩语大语言模型（LLM）基准测试评估了模型在多个知识领域的性能，但其准确率分数无法区分错误回答是由于缺乏所需知识还是未能正确应用该知识。我们对现有韩语基准测试的分析表明，大多数问题仅通过事实回忆即可回答，而那些需要更强应用能力的问题在所需知识方面也存在差异，这使得难以单独评估模型的应用能力。为了解决这个问题，我们提出了KLEAR，这是一个通过在保持所需知识不变的前提下改变应用需求来构建知识匹配对比基准的框架。在构建过程中，KLEAR将每个源问题背后的知识识别为显式的锚点集合，用以指导生成需要更复杂应用能力的问题。生成的候选问题随后经过自动化和人工验证，并进行迭代式的重新（修正）。

    arXiv:2611.09223v1 Announce Type: new  Abstract: Existing Korean benchmarks for Large Language Models (LLMs) evaluate performance across diverse knowledge domains, but their accuracy scores cannot distinguish whether incorrect responses arise from a lack of required knowledge or a failure to apply that knowledge correctly. Our analysis of existing Korean benchmarks reveals that most questions can be answered through factual recall alone, while those demanding greater application also differ in required knowledge, making it difficult to isolate application ability. To address this, we introduce KLEAR, a framework that constructs knowledge-matched contrasts by varying application demand while preserving the required knowledge. During construction, KLEAR identifies the knowledge underlying each source question as an explicit anchor bundle to guide generation of questions requiring more complex application. Generated candidates then undergo automated and human validation, with iterative re
    
[^28]: 高棉语 PFGen-Bench：为低资源语言构建无裁判的开放式生成基准

    Khmer PFGen-Bench: Building a Judge-Free Benchmark for Open-Ended Generation in a Low-Resource Language

    [https://arxiv.org/abs/2611.09222](https://arxiv.org/abs/2611.09222)

    该论文提出了 Khmer PFGen-Bench，这是首个针对高棉语这一低资源语言的无裁判开放式生成基准，并发现混合九个模型家族的后训练模型生成的参考答案能兼顾有效性与多样性，从而使模型排名保持稳定。

    

    开放式生成的基准测试通常依赖大语言模型（LLM）裁判，但其在低资源语言中的可靠性尚不确定。我们提出了 Khmer PFGen-Bench，据我们所知，这是首个针对直接以高棉语（一种低资源语言）撰写的问题进行多句回答的无裁判基准。Khmer PFGen-Bench 改编了 PFGen-Bench（一个日语基准）的评分器，该评分器使用关键词规则和来自 LLM 生成参考答案的 n-gram 统计数据对回答进行评分。这种评分方式需要既有效又多样化的参考答案，而 LLM 在高棉语中可能无法生成这样的参考答案，因此我们研究了如何构建此类参考答案。我们发现，经过后训练的模型相比其基础模型变体能提供更有效但多样性更低的参考答案，并且在相同参考答案数量的情况下，混合不同模型家族可以增加多样性。使用来自九个模型家族的后训练模型生成的参考答案，该基准所产生的模型排名在不同参考集之间保持稳定。

    arXiv:2611.09222v1 Announce Type: new  Abstract: Benchmarks for open-ended generation often rely on large language model (LLM) judges, whose reliability in low-resource languages is uncertain. We introduce Khmer PFGen-Bench, to our knowledge the first judge-free benchmark for multi-sentence answers to questions composed directly in Khmer, a low-resource language. Khmer PFGen-Bench adapts the scorer of PFGen-Bench, a benchmark for Japanese that scores responses with keyword rules and n-gram statistics from LLM-generated references. This scoring requires references that are both valid and varied, which LLMs may not produce in Khmer, so we examine how to build such references. We find that post-trained models can give more valid but less varied references than their base variants, and that mixing model families increases variety for the same number of references. Using references from post-trained models in nine families, the benchmark yields model rankings that are stable across referenc
    
[^29]: RPEG：可扩展且高性价比地训练推理语言模型

    RPEG: Training Reasoning Language Models Scalably and Cost-Effectively

    [https://arxiv.org/abs/2611.09207](https://arxiv.org/abs/2611.09207)

    提出了一种新颖的以数据为中心的图中间表示 RPEG，通过联合优化 RLM 强化学习训练流水线的结构与执行，并结合综合分析性能模型，实现了推理语言模型训练的可扩展性与高性价比。

    

    强化学习（RL）如今是训练推理语言模型（RLM）的核心方法，但由于每个 RL 步骤都耦合了多个大模型、异构阶段、跨步骤状态以及复杂的异步依赖关系，RLM 流水线难以扩展且运行成本高昂。现有系统通常只优化该设计空间的部分环节。我们通过一种新颖的以数据为中心的图中间表示——RLM 流水线执行图（RPEG），对流水线结构及其执行进行联合优化，从而应对这一复杂性。RPEG 揭示了诸如受限异步与 actor-critic 共享等结构性选择，同时结合阶段融合、细粒度调度和多维模型并行等执行优化。它还配套了一个面向 RLM 训练的全面分析性能模型，可联合刻画计算、通信、内存以及机架、节点、GPU 层级的分层效应。

    arXiv:2611.09207v1 Announce Type: cross  Abstract: Reinforcement learning (RL) is now central to training Reasoning Language Models (RLMs), but RLM pipelines are difficult to scale and costly to run because each RL step couples multiple large models, heterogeneous stages, cross-step state, and complex asynchronous dependencies. Existing systems typically optimize only parts of this design space. We address this complexity by jointly optimizing the pipeline structure as well as its execution using a novel data-centric graph intermediate representation called RLM Pipeline Execution Graph (RPEG). RPEG exposes structural choices such as bounded asynchrony and actor-critic sharing together with execution optimizations such as stage fusion, fine-grained scheduling, and multi-dimensional model parallelism. It is coupled with a comprehensive analytical performance model for RLM training, jointly capturing compute, communication, memory, and hierarchical rack-, node-, and GPU-level effects. Acr
    
[^30]: 代码模型中的表征漂移：行为趋于定型，而承载它的神经元却未定型

    Representational drift in a code model: behaviour settles, which neurons carry it does not

    [https://arxiv.org/abs/2611.09204](https://arxiv.org/abs/2611.09204)

    该研究首次在代码大模型OpenCoder-1.5B的预训练中系统测量了神经科学中的“表征漂移”现象，发现对于语法固定的Python构造，模型行为（损失）在训练早期即已定型，但承载这些行为的神经元集合在整个训练过程中持续变化、从未稳定。

    

    在神经科学中，表征漂移是指在行为保持不变的情况下，承载某一表征的神经元发生改变；我们在OpenCoder-1.5B的预训练过程中测量了其类似现象，涵盖169个Python构造、2万亿个token，学习率保持恒定，每次变化都设有噪声单位，检查点间隔最短仅为1,000个训练步。对于169个构造中的16个——即其语法固定了下一个token的那些构造——这些位置上的损失在训练的前4%内就达到了最低值，而模型对承载这些构造的神经元的依赖却持续增长到训练结束：在40亿、840亿和2万亿token时，移除这些神经元比移除随机神经元集合分别多造成0.06、0.25和0.40 nats的损失。承载某一构造的神经元集合从未定型：对于全部169个构造，在最后一个检查点时承载集合仍在变化；若在840亿到1.5万亿token之间的任何时间点发生突然定型，或在5000亿token之前发生渐变定型，都能以99%的概率被检测到。

    arXiv:2611.09204v1 Announce Type: new  Abstract: In neuroscience, representational drift is a change in which neurons carry a representation while behaviour stays the same; we measure its analogue in OpenCoder-1.5B during pretraining, for 169 Python constructs over two trillion tokens at a constant learning rate, with a noise unit on every change, at checkpoints as close as 1,000 training steps. For 16 of the 169, those whose grammar fixes a next token, the loss at those positions reaches its floor within the first 4% of training, while the model's reliance on the neurons carrying them grows to the end: removing them costs 0.06, 0.25 and 0.40 nats more than removing a random set, at 4 billion, 84 billion and 2 trillion tokens. Which neurons carry a construct never settles: for all 169 the carrier set is still changing at the last checkpoint, and an abrupt settling at any date from 84 billion to 1.5 trillion tokens, or a gradual one before 500 billion, would have been detected in 99% of
    
[^31]: 副本不是确认：新闻重复如何欺骗大语言模型评判者与持续学习者

    Copies Are Not Confirmations: How News Duplication Fools LLM Judges and Continual Learners

    [https://arxiv.org/abs/2611.09203](https://arxiv.org/abs/2611.09203)

    该研究揭示了新闻重复副本并非独立确认，LLM评判者会被同一来源的重复报道欺骗——如Llama 3.2 3B和phi4 14B在面对三份相同转载时接受谎言的比例从0.06和0.21分别飙升至0.71和0.90——表明去重检测对LLM证据评估至关重要。

    

    arXiv:2611.09203v1 公告类型：cross 摘要：关于某一主张的许多新闻报道都是同一来源的副本：通讯社稿件、媒体集团内共享的内容、联合特稿和新闻稿。我们探讨LLM评判者和持续学习者是否将此类副本视为独立的确认。在200,000篇英文CC-News文章中，一个MinHash检测器将18.9%的文章与另一媒体机构的近似副本关联起来（去除样板文本后精确率为0.97）；它能将92%的编辑转载与其原文关联，但无法关联phi4、Llama 3.2或Gemini 2.5 Flash生成的LLM改写版本。当向Llama 3.2 3B和phi4 14B展示关于某陌生人的同一主张的三份相同转载而非一份报道时，它们接受的谎言比例分别从0.06上升到0.71和从0.21上升到0.90，并且忽略了报道来自同一来源的提示；Gemini 2.5 Flash已经相信单份报道，而Gemini 2.5 Pro则能遵循来源注释和表格。在模拟的持续学习流中，向评判者展示每个...

    arXiv:2611.09203v1 Announce Type: cross  Abstract: Many news reports of a claim are copies of one source: wire stories, content shared within a media group, syndicated features and press releases. We ask whether LLM judges and continual learners treat such copies as independent confirmations. In 200,000 English CC-News articles, a MinHash detector links 18.9% to a near-copy from another outlet (precision 0.97 once boilerplate is removed); it links 92% of edited reprints to their originals but none of the LLM rewrites by phi4, Llama 3.2 or Gemini 2.5 Flash. Shown three identical reprints of a claim about an unknown person instead of one report, Llama 3.2 3B and phi4 14B raised the share of lies they accepted from 0.06 to 0.71 and from 0.21 to 0.90 and ignored a note that the reports came from one source; Gemini 2.5 Flash already believed a single report, and Gemini 2.5 Pro followed provenance notes and tables. In a simulated continual-learning stream, showing the judge one report per de
    
[^32]: 基于结构与谱系证据的越南法律条文检索研究（DRiLL数据集）

    Structure and Lineage Evidence for Vietnamese Statutory Article Retrieval on DRiLL

    [https://arxiv.org/abs/2611.09200](https://arxiv.org/abs/2611.09200)

    该论文通过引入条款身份、同一法律支持以及引用与谱系证据等结构化特征，将DRiLL数据集上越南法律条文检索的官方F2指标从71.77%提升至72.91%。

    

    我们在DRiLL数据集上研究越南法律条文检索，使用的特征包括条款身份与局部形式、同一法律内的支持信息，以及引用与谱系证据。在五个随机种子下，采用共享候选的selector、列表式Qwen证据和相同学习器设置，将官方F2指标从71.77%提升至72.91%，其中在双条文问题上的描述性提升最大。新增特征占所提出selector的总决策树分裂增益（以训练目标减少量衡量）的18.20%，主要来自同一法律和引用证据。案例分析揭示了基础条文与修正案、关联条文的恢复、干扰项的剔除以及失败情形。这些发现为在其他数据集、表示和学习器上的研究提供了启发，但并未确立单个特征的独立效应，也不依赖LightGBM、手工构造的分数或显式图结构。

    arXiv:2611.09200v1 Announce Type: cross  Abstract: We study Vietnamese statutory Article retrieval on DRiLL using features for provision identity and local form, same-law support, and citation and lineage evidence. Across five random seeds, selectors sharing candidates, listwise Qwen evidence, and learner settings improve official $F_2$ from 71.77% to 72.91%, with the largest descriptive gain on two-Article questions. Added features account for 18.20% of the proposed selector's total decision-tree split gain, measured as training-objective reduction, mainly from same-law and citation evidence. Cases reveal base provisions and amendments, companion recovery, distractor removal, and failures. These findings motivate research on other datasets, representations, and learners without establishing individual feature effects or requiring LightGBM, hand-crafted scores, or explicit graphs.
    
[^33]: 校准概念轮廓：使对模型神经元的读取成为一种测量

    Calibrated concept profiles: making the reading of a model's neurons a measurement

    [https://arxiv.org/abs/2611.09196](https://arxiv.org/abs/2611.09196)

    本文提出“校准概念轮廓”方法，在明确声明的刺激总体上对每个神经元进行标准化测量，使可解释性研究中的直接神经元读取成为一种可重复、可比较、带有校准单位的测量。

    

    可解释性研究已大量地从直接读取模型的神经元转向读取在原模型基础上训练的第二个模型：探针、稀疏自编码器、交叉编码器。这一转变解决了确实存在的缺陷——含义混杂、没有单位的阈值、无法在不同研究之间迁移的读取结果——但放弃了只有直接读取才能提供的东西：中间没有任何训练环节、神经元本身被直接命名、以及可重复且可比较的读取。本文将直接读取变成一种测量。神经元的激活随提示词而变化，因此其读取是在刺激总体上的一个分布。我们声明这一总体：围绕目标构念的每个因素按其在参考语料库中的出现比率进行抽取并记录。从声明的分布中我们生成一组提示词，并抽取两次。从这组提示词中我们计算校准概念轮廓，即每个神经元对应一个数字，以神经元自身的散布为单位表示。

    arXiv:2611.09196v1 Announce Type: new  Abstract: Interpretability has largely moved from reading a model's neurons directly to reading a second model trained on the first: a probe, a sparse autoencoder, a crosscoder. The move answered real faults -- mixed meanings, thresholds without a unit, readings that did not transfer from one study to the next -- but gave up what a direct reading alone provides: nothing trained in between, the neurons themselves named, and a reading that can be repeated and compared. Here the direct reading is made a measurement.   A neuron's activation varies with the prompt, so its reading is a distribution over a population of stimuli. We declare that population: every factor around a target construct is drawn at its rate in a reference corpus and logged.   From the declared distribution we generate a battery of prompts, and draw it twice. From the battery we compute the calibrated concept profile, one number per neuron, stated in units of the neuron's own spre
    
[^34]: 一个基于分类学驱动的合成数据集：用于招聘场景下的检索有效性与公平性评估

    A Taxonomy-Driven Synthetic Dataset for Retrieval Effectiveness and Fairness Evaluation in Hiring

    [https://arxiv.org/abs/2611.09178](https://arxiv.org/abs/2611.09178)

    该论文提出了一个基于分类学驱动、完全合成的多语言HR数据集与基准，可同时评估检索有效性与算法公平性，揭示了通用基准在招聘场景中的不足，并为识别兼顾语义准确性与公平性的帕累托最优模型提供了定制化评估方案。

    

    尽管机器学习招聘工具已无处不在，但隐私法规和商业敏感性的要求使得公开基准测试在法律上变得复杂。为应对这一挑战，我们引入了一个基于分类学驱动的完全合成的HR数据集和基准，可同时评估检索有效性和算法公平性。该多语言语料库基于66万份真实自由职业者档案构建，包含候选人档案、项目简介以及反事实人口统计孪生体。对最先进的嵌入模型的审计显示，其性能与通用MTEB分数存在差异，同时在具有语法性别的语言中也存在公平性差异。这表明通用基准不足以满足HR领域的需求，需要定制化的评估来识别在语义准确性与公平性之间取得平衡的帕累托最优模型。

    arXiv:2611.09178v1 Announce Type: cross  Abstract: While machine learning hiring tools are ubiquitous, the need for privacy regulations and commercial sensitivity makes public benchmarking legally complex. We address this challenge by introducing a taxonomy-driven, fully synthetic HR dataset and benchmark to simultaneously evaluate retrieval effectiveness and algorithmic fairness. Grounded in 660K real freelancer profiles, our multilingual corpus features candidate profiles, project briefs, and counterfactual demographic twins. Auditing state-of-the-art embedding models reveals performance divergence from general MTEB scores, alongside fairness disparities in grammatically gendered languages. This demonstrates that general benchmarks are insufficient for HR, requiring tailored evaluations to identify Pareto-optimal models that balance semantic accuracy and fairness.
    
[^35]: BMMS：语言模型中策略条件化的行为合规性度量

    BMMS: Policy-Conditioned Measurement of Behavioral Compliance in Language Models

    [https://arxiv.org/abs/2611.09147](https://arxiv.org/abs/2611.09147)

    BMMS是一个以任意已发布政策语料库为输入、返回合规画像的策略条件化度量套件，通过合规达成度分数（CAS）和合规稳健性裕度（CRM）等指标，首次实现了对语言模型行为合规性及其对抗稳健性的量化评估。

    

    语言模型的行为评估预先固定了其评估维度：幻觉、谄媚、毒性、过度拒绝。而受监管行业的运营者需要回答的是一个不同的问题：系统是否做到了某本特定规则手册所要求的事项？我们提出了行为失准度量套件，它以政策语料库作为输入参数，并返回针对该语料库的合规画像，从而度量策略条件化的失准程度，适用于任何已发布的政策和任何以自然语言作答的系统。其中，合规达成度分数衡量系统在其所受约束义务上的严重度加权达成程度；合规稳健性裕度衡量该达成度在分级对抗压力下仍能保留的比例；行为不变性（BIV）和行为倾向指数（BDI）也随套件一同被定义并发布。评估维度只有通过一个公开发布的准入测试（其约束性判据……）才能进入评估体系。

    arXiv:2611.09147v1 Announce Type: cross  Abstract: Behavioral evaluation of language models fixes its dimensions in advance: hallucination, sycophancy, toxicity, over-refusal. An operator in a regulated industry needs a different question answered: does a system do what a particular rulebook requires? We introduce the Behavioral Misalignment Metric Suite (BMMS), which measures policy-conditioned misalignment by taking a policy corpus as an argument and returning a compliance profile over it, for any published policy and any system that answers in natural language. The Compliance Attainment Score (CAS) measures severity-weighted attainment over the obligations that bind the system, and the Compliance Robustness Margin (CRM) the share of that attainment surviving graded adversarial pressure; Behavioral Invariance (BIV) and the Behavioral Disposition Index (BDI) are specified and released with the suite. Dimensions enter only by passing a published admission test whose binding criterion i
    
[^36]: PathXAI：基于可迁移组织分割的证据引导结构化病理报告生成

    PathXAI: Evidence-guided structured pathology reporting by transferable tissue segmentation

    [https://arxiv.org/abs/2611.09117](https://arxiv.org/abs/2611.09117)

    PathXAI通过可迁移的组织分割信息结合H&E图像与领域知识来引导诊断推理和结构化病理报告生成，在诊断一致性和形态学描述准确性上显著优于纯H&E方法。

    

    病理报告需要将跨区域和跨切片的组织观察转化为具体诊断及支持性的形态学描述。然而，利用明确的组织构成测量来指导基于H&E（苏木精-伊红染色）图像的报告生成仍鲜有探索。因此，我们开发了PathXAI，该方法将分割得到的组织信息与H&E图像及可选的领域知识相结合，以指导诊断推理、针对性复查和结构化报告撰写。其分割组件在内部和外部测试集上的平均Dice系数均超过基线方法。在1,200例公开病例和151例私有病例的自动化报告评估中，PathXAI与参考诊断及已记录形态学的一致性高于基于H&E的对比方法。在公开评估队列上，PathXAI-Report在癌家族匹配率上超过Qwen + H&E 4.4个百分点，在发现项F1分数上超过5.7个百分点。

    arXiv:2611.09117v1 Announce Type: cross  Abstract: Pathology reporting requires translating tissue observations across regions and slides into specific diagnoses and supporting morphological descriptions. However, the use of explicit tissue-composition measurements to guide H&E-based report generation remains underexplored. Thus, we developed PathXAI, which combines segmentation-derived tissue information with H&E images and optional domain knowledge to guide diagnostic reasoning, targeted reinspection and structured report writing. Its segmentation component exceeded baselines in mean Dice on internal and external tests. In automated report evaluation on 1,200 public and 151 private cases, PathXAI achieved higher agreement with reference diagnoses and documented morphology than the H&E-based comparators. On the public evaluation cohort, PathXAI-Report exceeded Qwen + H&E in cancer family match rate by 4.4 percentage points and in finding F1 by 5.7 percentage points. PathXAI generates 
    
[^37]: PhoPunct：用于越南语标点恢复的大规模Transformer架构基准测试

    PhoPunct: A Large-Scale Benchmark of Transformer Architectures for Vietnamese Punctuation Restoration

    [https://arxiv.org/abs/2611.09107](https://arxiv.org/abs/2611.09107)

    该论文构建了包含2210万标注词元的越南语标点恢复大规模基准，评估17种Transformer架构后发现，在固定训练预算下添加Bi-LSTM和CRF层并不能可靠提升性能，纯Transformer分类器在大多数对比中与Bi-LSTM-CRF模型相当甚至更优。

    

    越南语标点恢复任务通常在相对较小的基准上进行评估，这限制了关于架构选择如何随数据量扩展的结论。我们在包含2210万个标注词元（1990万新闻和220万小说）的新闻和小说语料库上提出了一个受控基准。我们评估了由四个预训练Transformer主干网络与可选的双向长短期记忆网络和条件随机场层组合而成的17种架构，以及一个名为PhoPunct的PhoBERT-large-Bi-LSTM-CRF模型。所有模型均在相同学习率下训练八个周期，且不进行针对特定架构的调优。在这一固定预算下，添加Bi-LSTM和CRF层并不能可靠地提升宏观F1分数：在八组匹配的领域-主干对比中，有七组显示纯Transformer分类器与其对应的Bi-LSTM-CRF模型相当或更优。PhoPunct获得了新闻语料上的最佳结果（61.35），而带有CRF的XLM-R则获得……（原文摘要此处截断）

    arXiv:2611.09107v1 Announce Type: new  Abstract: Vietnamese punctuation restoration is often evaluated on relatively small benchmarks, limiting conclusions about how architectural choices scale with data volume. We present a controlled benchmark on News and Novels corpora containing 22.1 million labeled tokens (19.9M News and 2.2M Novels). We evaluate 17 architectures formed from four pretrained Transformer backbones and optional bidirectional long short-term memory (Bi-LSTM) and conditional random field (CRF) layers, together with a PhoBERT-large-Bi-LSTM-CRF model, PhoPunct. Every model is trained for eight epochs with the same learning rate and without architecture-specific tuning. Under this fixed budget, adding Bi-LSTM and CRF layers does not reliably improve macro F1: in seven of eight matched domain-backbone comparisons, a plain Transformer classifier matches or outperforms its Bi-LSTM-CRF counterpart. PhoPunct obtains the best News result (61.35), whereas XLM-R with a CRF obtain
    
[^38]: AlienBench：语言模型输出中结构性新颖性的基准测试

    AlienBench: A Benchmark of Structural Novelty in Language Model Outputs

    [https://arxiv.org/abs/2611.09094](https://arxiv.org/abs/2611.09094)

    AlienBench将认知心理学中的结构化想象任务（想象外星生物）适配到语言模型，通过衡量模型输出偏离地球动物熟悉特征（如双侧对称性、腿部运动方式）的程度，为新颖性评估提供了具有固定参照基准的测试方法。

    

    大型语言模型越来越多地被用于写作、设计和创意生成等创造性任务。对于这些用途而言，模型的创意是偏离熟悉的范例还是贴近它们，这一点非常重要。新颖性是创造力的一项标准衡量指标，目前的创造力评估方法通常请人类专家或大模型评审员来评判输出的新颖程度。然而，这些评分对于什么算作新颖（或者说，对熟悉的偏离）没有固定的参照标准。要测量这种偏离，需要一个具有已知熟悉默认值的任务。认知心理学中的结构化想象任务正是这样一个任务。在这个创造力测试中，人们想象一种来自外星的生物，并以对地球动物特征（如双侧对称性和以腿作为运动方式）的偏离程度作为创造力的衡量标准。我们提出了AlienBench，这是一个将结构化想象任务适配到语言模型的基准测试。模型……

    arXiv:2611.09094v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used for creative tasks such as writing, design, and idea generation. For these uses, it matters whether a model's ideas depart from familiar examples or stay close to them. Novelty is one standard measure of creativity, and current creativity evaluations ask human experts or LLM judges to rate how novel an output is. However, these ratings have no fixed reference for what counts as novel (or, in other terms, departure from the familiar). Measuring this departure requires a task with known familiar defaults. The structured imagination task from cognitive psychology is such a task. In this creativity test, people imagine a creature from another planet. The departure from features of Earth animals, such as bilateral symmetry and legs as mode of locomotion, is used as measure of creativity. We introduce AlienBench, a benchmark that adapts the structured imagination task to language models. Model
    
[^39]: MARS：训练多语言搜索智能体以实现无检索器的证据获取

    MARS: Training Multilingual Search Agents for Retriever-Free Evidence Acquisition

    [https://arxiv.org/abs/2611.09090](https://arxiv.org/abs/2611.09090)

    提出MARS——一个通过强化学习训练的多语言搜索智能体，让模型直接与原始语料库交互而无需传统检索器，并借助语料库原生的实体锚定和冗余感知的搜索奖励机制来克服跨语言精确搜索的困难、实现高效证据获取。

    

    现有的多语言检索增强生成严重依赖于从多语言文献集合中学习到的检索器以及基于相似度的排序，这种方式容易受到跨语言检索错误和索引开销的影响。在本文中，我们探索了无检索器的多语言搜索，即模型通过可执行的 shell 命令直接与原始语料库交互。这一范式将瓶颈转移到了多语言搜索动作的生成上：精确搜索对实体的表层形式非常敏感，而仅基于结果的监督为多轮搜索提供的引导十分有限。为了应对这些挑战，我们提出了 MARS——一个在直接语料库交互（DCI）范式下通过强化学习（RL）训练的多语言搜索智能体。MARS 引入了语料库原生的实体锚定方法，从语料库证据中恢复可靠的词汇锚点，从而缓解证据检索失败的问题。同时，我们提出了一种冗余感知的搜索奖励，以鼓励简洁……（摘要在此处截断）

    arXiv:2611.09090v1 Announce Type: new  Abstract: Existing multilingual retrieval-augmented generation relies heavily on learned retrievers and similarity-based ranking from multilingual collections, which are vulnerable to cross-lingual retrieval errors and indexing overhead. In this paper, we explore retriever-free multilingual search, where the model directly interacts with raw corpora through executable shell commands. This paradigm shifts the bottleneck to multilingual search action generation, where exact search is sensitive to entity surface forms and outcome-only supervision provides limited guidance for multi-turn search. To address these challenges, we propose MARS, a multilingual search agent in direct corpus interaction (DCI) with reinforcement learning (RL). MARS introduces corpus-native entity grounding to recover reliable lexical anchors from corpus evidence, mitigating failure to retrieve evidence. Meanwhile, we present a redundancy-aware search reward to encourage conci
    
[^40]: 隐形的鸿沟：自动指标看好之处，罗马化印度语言生成实则失败

    The Invisible Gap: Romanized Indic Generation Fails Where Automatic Metrics Succeed

    [https://arxiv.org/abs/2611.09085](https://arxiv.org/abs/2611.09085)

    该研究发现自动指标（如BGE-M3）系统性地误报罗马化印度语言生成的真实质量——人类评分差距显著而指标差异近乎为零，并揭示出文字溢出与英文退化两种模型失败模式，据此提出故障感知的评估策略。

    

    数亿印度人使用罗马化拉丁字母与数字系统交流——例如输入"aap kaise hain"而非天城文等原生文字——然而罗马化印度语言的生成在NLP基准测试与评估框架中却付之阙如。我们证明自动指标（BGE-M3、BERTScore）会系统性地误报罗马化印地语、泰卢固语和卡纳达语生成的质量。我们在142个OpenAssistant问答对上评估了四个大语言模型（Claude Opus 4.8、GPT-5.5、Gemini 2.5-flash、Qwen3.5-35B）。在一个包含72条回复的诊断子集上，母语者在12分评分量表上给GPT-5.5的评分比Claude Opus 4.8高出1.43分，而两者在全数据集上的BGE-M3均值差异仅为0.004。研究揭示出两种截然不同的模型特定失败模式：Gemini产生文字溢出（16.9%的泰卢固语回复泄漏出原生文字），而Qwen则退化为纯英文输出（16.2%的卡纳达语回复）。我们提出了一种故障感知的评估策略。

    arXiv:2611.09085v1 Announce Type: new  Abstract: Hundreds of millions of Indians communicate with digital systems in romanized Latin characters -- typing "aap kaise hain" rather than native script -- yet romanized Indic generation is absent from NLP benchmarks and evaluation frameworks. We demonstrate that automatic metrics (BGE-M3, BERTScore) systematically misrepresent the quality of romanized Hindi, Telugu, and Kannada generation. We evaluate four LLMs (Claude Opus 4.8, GPT-5.5, Gemini 2.5-flash, Qwen3.5-35B) on 142 OpenAssistant query-response pairs. On a 72-response diagnostic subset, native speakers rate GPT-5.5 1.43 points higher than Claude Opus 4.8 on a 12-point rubric, whereas their full-dataset BGE-M3 means differ by only 0.004. Two distinct model-specific failure modes emerge: Gemini produces script spillage (16.9% of Telugu responses leak native script), while Qwen collapses to English-only output (16.2% of Kannada responses). We introduce a failure-aware evaluation policy
    
[^41]: 大语言模型在资源受限边缘设备上的部署：从业者综述

    Large Language Model Deployment on Resource-Constrained Edge Devices: A Practitioners Survey

    [https://arxiv.org/abs/2611.09075](https://arxiv.org/abs/2611.09075)

    本文综述了软件优化与专用硬件领域的最新技术，使大语言模型能够在资源受限的边缘设备上部署，并阐述了本地数据处理、成本节约、降低延迟和个性化响应等边缘部署优势。

    

    大语言模型已经变革了人工智能的众多领域，但其巨大的资源需求限制了它们在移动和边缘设备上的部署。尽管大语言模型在各种应用中展现出令人印象深刻的能力，下一个挑战是将这些模型部署到资源有限的边缘设备上。软件优化和专用硬件方面的最新创新使在边缘设备上部署大语言模型成为可能。边缘部署带来了诸多优势，包括本地数据处理、成本节约、降低延迟以及个性化响应。本文综述了使大语言模型在边缘设备上部署成为可行的最新技术。

    arXiv:2611.09075v1 Announce Type: cross  Abstract: Large Language Models (LLMs) have transformed many areas of artificial intelligence (AI), but their substantial resource requirements limit their deployment on mobile and edge devices. Although LLMs have demonstrated impressive capabilities in various applications, the next challenge is deploying these models on resource-limited edge devices. Recent innovations in software optimizations and specialized hardware have made it feasible to deploy LLMs on edge devices. Edge deployment offers benefits including local data processing, cost savings, reduced latency, and personalized responses. This paper reviews the latest techniques that make the deployment of LLMs on edge devices feasible.
    
[^42]: 推理能力能否提升大语言模型的心理深度？

    Does Reasoning Improve Psychological Depth in Large Language Models?

    [https://arxiv.org/abs/2611.09061](https://arxiv.org/abs/2611.09061)

    该研究通过盲测成对评估发现，推理增强并不必然提升大语言模型在短篇小说心理深度这类开放式创意写作中的表现，GPT-5仅略优于GPT-4o，而DeepSeek-R1反而落后于非推理版的DeepSeek-V3。

    

    推理增强型大语言模型在基准测试上的成绩提升，促使人们假设这些模型在复杂生成任务上应普遍表现更佳，但在受读者体验细微差别影响的开放式创意领域，这一假设仍未得到充分探索。我们通过短篇小说的心理深度这一视角检验了该假设。我们比较了DeepSeek-R1与DeepSeek-V3（这是可获得的最接近的面向推理的后训练对比组合），以及GPT-5与GPT-4o，通过60次盲测成对评估，由七名人类标注员和一个集成LLM-as-a-Judge对提示匹配的故事输出进行评分。人类评估显示推理能力并无普遍优势：GPT-5略优于GPT-4o（在两种推理努力设置下为60.0-62.9%），而DeepSeek-R1则落后于DeepSeek-V3（42.9%）。尽管这一结果伴随着较低的标注员间一致性（Krippendorff's α = 0.070），我们的标注质量控……（原文在此截断）

    arXiv:2611.09061v1 Announce Type: new  Abstract: Benchmark gains from reasoning-enhanced large language models encourage the assumption that these models should broadly perform better on complex generative tasks, but this remains underexplored in open-ended, creative domains shaped by nuances of reader experience. We tested this assumption through the lens of psychological depth in short stories. We compared DeepSeek-R1 vs. DeepSeek-V3, our closest available contrast for reasoning-oriented post-training, and GPT-5 vs. GPT-4o, through 60 blinded pairwise evaluations in which seven human annotators and an ensemble LLM-as-a-Judge rated prompt-matched story outputs. Human evaluations showed no universal reasoning advantage: GPT-5 was slightly preferred over GPT-4o (60.0-62.9% across two reasoning-effort settings), whereas DeepSeek-R1 fell behind DeepSeek-V3 (42.9%). Although this pattern was marked by low inter-annotator agreement (Krippendorff's $\alpha = 0.070$), our annotation-quality c
    
[^43]: 口语对话系统中的实时短回应选择

    Real-Time Short-Response Selection for Spoken Dialogue Systems

    [https://arxiv.org/abs/2611.09057](https://arxiv.org/abs/2611.09057)

    提出了一种基于对比学习的实时短回应选择方法，在检测到对话轮替事件时从预生成候选中即时选择并播放短回应，从而有效掩盖口语对话系统中语言生成与语音合成的延迟。

    

    为了实现能够与用户自然交互的口语对话系统，系统在合适的时机对用户话语进行回应至关重要。为实现这种自然的对话轮替，系统不仅需要预测合适的回应时机，还需要掩盖语言生成和语音合成造成的延迟，使其不被用户察觉。我们提出了一个可与现有对话轮替模型集成的实时短回应选择任务，以掩盖生成延迟。为解决该任务，所提出的模型使用对比学习来学习对话语境与短回应之间的映射。当检测到对话轮替事件时，系统从一组预先生成的候选回应中选择一个回应，并在主回应生成期间播放所选的短回应。实验结果表明，所提出的方法能够学会基于对话语境选择短回应……

    arXiv:2611.09057v1 Announce Type: cross  Abstract: To realize spoken dialogue systems that can interact naturally with users, it is important for the system to respond to user utterances at appropriate timings. To achieve such natural turn-taking, the system needs not only to predict appropriate response timings but also to mask delays caused by language generation and speech synthesis so that they are not perceived by the user. We propose a real-time short-response selection task that can be integrated with existing turn-taking models to mask generation delay. To address this task, the proposed model learns the mapping between dialogue context and short responses using contrastive learning. When a turn-taking event is detected, the system selects a response from a set of pre-generated candidates. The selected response is played while the main response is being generated. Experimental results show that the proposed method learns to select short responses based on dialogue context, outp
    
[^44]: 推理蒸馏中的失败画像与可达轨迹选择

    Failure Profiling and Reachable Trajectory Selection for Reasoning Distillation

    [https://arxiv.org/abs/2611.09042](https://arxiv.org/abs/2611.09042)

    提出FPRT方法，通过重复采样刻画学生模型的失败行为并筛选其“可学习”的训练轨迹，使蒸馏训练数据同时兼顾学生失败点与实际可达能力，从而缓解师生分布不匹配导致的泛化差问题。

    

    用较大教师模型生成的推理解法来训练较小的语言模型，是目前迁移数学推理能力的主流方法。然而，学生模型往往会过拟合教师模型的输出分布，当其在推理阶段自身的行为偏离训练数据时，泛化能力表现很差。近期的方法试图通过引入错误数据来弥合这一差距，但这些纠正轨迹仍然反映的是教师模型的推理模式，而学生模型无法可靠地复现这些模式，导致分布不匹配问题在很大程度上仍未得到解决。我们提出了一种名为“失败画像与可达轨迹选择”（FPRT）的方法，该方法基于这样一个洞察：有效的训练数据应当由“学生在哪些方面失败”和“学生能够切实学到什么”这两方面共同决定。我们的方法首先通过重复采样来刻画学生模型在每个问题上的失败行为，并对其进行表征……

    arXiv:2611.09042v1 Announce Type: new  Abstract: Training smaller language models on reasoning solutions generated by a larger teacher model is the prevailing approach for transferring mathematical reasoning capabilities. However, the student tends to overfit the output distribution of the teacher and generalizes poorly when its own behavior at inference time deviates from the training data. Recent methods attempt to bridge this gap by incorporating error data, yet the corrective trajectories still reflect the reasoning patterns of the teacher, which the student cannot reliably reproduce, leaving the distribution mismatch largely unresolved. We propose a method, Failure Profiling and Reachable Trajectory Selection (FPRT), built on the insight that effective training data should be jointly determined by what the student fails on and what it can realistically learn. Our method first profiles the failure behavior of the student on each problem through repeated sampling, characterizing it 
    
[^45]: 语言而非代码抽象：自然语言训练塑造与大脑对齐的几何结构

    Language, Not Code Abstraction: Natural-language Training Shapes Brain-aligned Geometry

    [https://arxiv.org/abs/2611.09027](https://arxiv.org/abs/2611.09027)

    研究通过对25个GPT-2模型的对照实验证明，只有自然语言训练才能塑造与大脑对齐的表征几何，代码训练无法替代自然语言训练在神经对齐中的作用。

    

    在代码上训练的语言模型能够产生正向的脑评分，这引出了一个疑问：从编程语言中学到的抽象结构能否替代自然语言训练？研究者在TinyStories、Python、C函数、x86-64汇编以及相应的机器代码上训练了25个GPT-2-small模型，并使用共享的分词器与编码流程，将其与匹配的随机权重模型进行对比评分。在Pereira-243和Tuckute2024数据集上，TinyStories模型超过四个代码模型的平均值，Δr分别为0.053和0.059（经Holm校正的p<0.001），而代码训练使模型对齐程度停留在接近随机权重的水平。语言优势具有基准依赖性：Blank2014未显示经校正后的语言-代码差异，Fedorenko2016也未显示语言带来的优势。几何结构与Pereira-243上的编码结果一致：TinyStories种子模型收敛于最相似的表征，且其与神经不相似性的免拟合对应关系……

    arXiv:2611.09027v1 Announce Type: new  Abstract: Code-trained language models produce positive brain scores, which raises the question of whether abstract structure learned from programming languages can substitute for natural-language training. We trained 25 GPT-2-small models on TinyStories, Python, C functions, x86-64 assembly, and the corresponding machine code, and scored them using a shared tokenizer and encoding procedure against matched random-weight models. On Pereira-243 and Tuckute2024, TinyStories exceeds the mean of the four code models by $\Delta r = 0.053$ and $0.059$ (Holm-corrected $p<0.001$), while code training leaves alignment near the random-weight level. The language advantage is benchmark-specific: Blank2014 shows no corrected language-code difference, and Fedorenko2016 shows no language benefit. Geometry tracks the encoding result on Pereira-243: TinyStories seeds converge on the most similar representations, and its fit-free correspondence with neural dissimila
    
[^46]: 看见与计数：复杂性孪生问题揭示多模态大语言模型在图表上的失败之处

    Seeing vs. Counting: Complexity Twins Show Where Multimodal LLMs Fail on Diagrams

    [https://arxiv.org/abs/2611.08998](https://arxiv.org/abs/2611.08998)

    该论文通过在同一图上配对不同计数复杂性类别的“复杂性孪生”问题，并辅以转录探针和纯文本对照，将多模态大模型的“看见”与“计数”能力分离，发现模型主要在计数环节失败且失败率取决于答案大小而非复杂性类别，而在提供精确转录和代码时模型可以正确计数。

    

    当多模态大语言模型在图表中数错时，它是在“看见”上失败了，还是在“计数”上失败了？现有的视觉图与谜题基准将视觉密度与算法难度同时变化，混淆了这两个因素。我们对同一幅图配对提出属于不同计数复杂性类别的计数问题，例如其生成树数量（一个行列式）与其简单路径数量（#P完全），并使用转录探针和纯文本对照来将“读取”与“计算”分离开来。我们测试了Gemini、Gemma和Qwen3-VL模型；对于Gemini模型，答案取决于模型如何计算。在思维链模式下，大多数模型能够正确读取小图，随后在计数环节失败，其失败率主要由答案的大小而非其复杂性类别决定（对于主要模型，答案每增大一个数量级，精确回答的几率下降2.38倍）。在给定代码和精确转录的情况下，主要模型即使在10x10的网格上也能正确计数，而且……

    arXiv:2611.08998v1 Announce Type: cross  Abstract: When a multimodal LLM miscounts in a diagram, does it fail at seeing or at counting? Existing visual graph and puzzle benchmarks vary visual density and algorithmic hardness together, confounding the two. We pair counting questions about the same drawing that sit in different counting-complexity classes, such as its spanning trees (a determinant) and its simple paths (#P-complete), and use a transcription probe and a text-only control to separate reading from computing. We test Gemini, Gemma and Qwen3-VL models; for the Gemini models, the answer depends on how the model computes. In chain of thought, most of them read small drawings correctly and then fail at counting, at a rate set far more by the size of the answer than by its complexity class (for the primary model, the odds of an exact answer fall 2.38-fold per decade of the answer). Given code and an exact transcription, the primary model counts correctly even on 10x10 grids, and 
    
[^47]: 相同事实，不同救济：大语言模型与“系统一”决策中的语言特征画像与任意性

    Same Facts, Different Remedy: Linguistic Profiling and Arbitrariness in LLM and System One Decisions

    [https://arxiv.org/abs/2611.08995](https://arxiv.org/abs/2611.08995)

    该研究通过对消费者投诉进行社会语言学改写来审计LLM与概率模型，发现尽管群体层面的语言偏见微小，但同一事实的投诉因写作风格不同而在2.6-6.3%的情况下获得不同的金钱救济，揭示了决策系统中的语言特征画像与任意性问题。

    

    大语言模型日益被用于解决消费者投诉，而投诉者的写作方式会随年龄、教育程度和社会经济地位（SES）而变化。我们使用被要求保留所有事实的美国消费者金融投诉的社会语言学改写版本，对三个救济决策系统进行了审计：两个返回单一标签的生成式大语言模型（GPT-4o-mini 和 GPT-5-mini），以及一个返回救济分布的概率性“系统一”模型（Jev）。通过在保留集投诉上对预先注册的假设进行一次性检验，我们区分了单一标签审计所混淆的两种现象。在群体尺度上，年长、受过大学教育和较高社会经济地位的写作风格会获得略微更慷慨的救济；这些效应可以复现但幅度很小，每一项金钱救济差距都在0.5个百分点以内。在个体尺度上，同一投诉的两个风格版本在2.6-6.3%的情况下会收到不同的金钱救济决策。

    arXiv:2611.08995v1 Announce Type: cross  Abstract: Large language models are increasingly used to resolve consumer complaints, and the way complainants write varies with age, education and socioeconomic status (SES). We audit three remedy-decision systems using sociolinguistic rewrites of US consumer financial complaints that are instructed to preserve every fact: two generative LLMs that return a single label (GPT-4o-mini and GPT-5-mini) and a probabilistic "System One" model that returns a distribution over remedies (Jev). Using pre-registered hypotheses tested once on held-out complaints, we separate two phenomena that single-label audits conflate. At the group scale, older, college-educated and higher-SES styles receive slightly more generous remedies; these effects replicate but are small, with every monetary-remedy gap within 0.5 percentage points. At the individual scale, two style versions of the same complaint receive different monetary decisions 2.6-6.3% of the time. In a var
    
[^48]: 超越截断？蕴含许可的省略与LLM解码中语义保持的极限

    Beyond Truncation? Entailment-Licensed Elision and the Limits of Semantic Preservation in LLM Decoding

    [https://arxiv.org/abs/2611.08975](https://arxiv.org/abs/2611.08975)

    该论文证明了现有免训练的解码期冗余删除方法本质上都只是截断，并提出ELIDE方法，利用自然语言推理模型在每个句子边界判断被删内容是否被保留内容所蕴含，从而实现真正超越截断的语义保持式回答精简。

    

    指令微调的大型语言模型（LLM）的回答长度往往超过请求的实际需要。我们探究在解码循环内部加入一个免训练的控制器能否在保持内容的前提下去除这些冗余，并以最简单的基线作为评判标准：将模型自身的回答截断到相同长度。一个基于熵门控的logits处理器在MT-Bench开发问题上删除了25.5%-34.9%的词元，在密封测试集上删除了19.9%-34.2%的词元，且均落在预先注册的BERTScore非劣性边界之内；然而我们通过理论证明和审计确认，其输出始终是未修改回答的前缀：所谓的节省本质上就是截断。对23个免训练机制族的调研发现，在没有oracle标签的情况下，没有任何机制能比截断更好地选择内容。因此，我们提出了蕴含许可的解码中省略（ELIDE），它在每个句子边界处，仅当自然语言推理模型判定该片段可由所保留内容蕴含时，才向读者隐去该片段……

    arXiv:2611.08975v1 Announce Type: new  Abstract: Instruction-tuned large language models (LLMs) answer at greater length than a request needs. We ask whether a training-free controller inside the decoding loop can remove that surplus while preserving content, judged against the simplest possible baseline: truncating the model's own answer to the same length. An entropy-gated logits processor removes 25.5-34.9% of tokens on MT-Bench development questions and 19.9-34.2% on a sealed test split within pre-registered BERTScore non-inferiority margins, yet we prove and confirm by audit that its output is always a prefix of the unmodified answer: the saving is truncation. A survey of 23 training-free mechanism families finds none that selects content better than truncation without oracle labels. We therefore introduce Entailment-Licensed In-Decoding Elision (ELIDE), which, at each sentence boundary, withholds a span from the reader only when a natural-language inference model judges it entail
    
[^49]: 幻影共识：测量多智能体LLM决策中的社会推理

    Phantom Consensus: Measuring Social Reasoning in Multi-Agent LLM Decision-Making

    [https://arxiv.org/abs/2611.08971](https://arxiv.org/abs/2611.08971)

    该论文首次发现并命名了多智能体LLM系统中的“幻影共识”现象——智能体在模糊社会信号下改变公开决策却误判他人私下判断，并提出了PHANTOM-Bench基准来系统测量这种多元无知式失败模式。

    

    多智能体LLM系统日益在多个模型实例之间聚合判断，但对其他智能体进行推理本身可能引入失败。我们研究了一种情境：LLM在观察到模糊的社会信号后会改变其公开决策，同时却对其他智能体的私下判断作出不准确的估计，即使该群体记录在案的私下判断并不支持这种表面上的共识。我们将这种失败称为“幻影共识”，并将其作为一种多元无知式的模式在受控的LLM集体中进行研究。为了测量这一现象，我们引入了PHANTOM-Bench，这是一个基准测试，它将孤立的私下判断、模糊社会信号后的公开决策、诱导出的群体私下判断估计、基于透明度的恢复手段，以及中性公开文本和明确跟随投票的对照组区分开来。在九个闭源和开源权重模型中，模糊的公开信号持续地降低决策质量。

    arXiv:2611.08971v1 Announce Type: cross  Abstract: Multi-agent LLM systems increasingly aggregate judgments across model instances, but reasoning about other agents can itself introduce failure. We study a setting in which LLMs shift their public decisions after observing ambiguous social signals while reporting inaccurate estimates of what other agents privately judged, even when the group's recorded private judgments do not support the apparent consensus. We call this failure phantom consensus and study it as a pluralistic-ignorance-style pattern in controlled LLM collectives. To measure this phenomenon, we introduce PHANTOM-Bench, a benchmark that separates isolated private judgments, public decisions after ambiguous social signals, elicited estimates of the group's private judgment, transparency-based recovery, and controls for neutral public text and explicit vote-following. Across nine closed- and open-weight models, ambiguous public signals consistently degrade decision quality 
    
[^50]: 使用大语言模型解决尼泊尔语推文命名实体识别中的困难案例

    Using an LLM to Tackle Difficult Cases in NER for Nepali Tweets

    [https://arxiv.org/abs/2611.08967](https://arxiv.org/abs/2611.08967)

    本研究利用微调PLM的词元概率将测试集划分为简单与困难案例，并通过基于语义相似度选择示例的提示策略使LLM在困难NER案例上超越PLM，两模型结合后获得更优的整体性能。

    

    命名实体识别（NER）对于低资源语言的社交媒体数据来说是一项挑战。大语言模型（LLM）在NER性能上通常落后于预训练语言模型（PLM），但在处理PLM难以应对的案例方面展现出潜力。我们使用一个尼泊尔语推文数据集，利用微调后的预训练语言模型（PLM）的词元概率将测试集划分为“简单”和“困难”两类案例。随后，我们使用16种不同的提示配置在困难测试案例上对大语言模型进行评估。我们发现，基于语义相似度选择示例并增加示例数量，可使F1分数比基线配置提高近10分。尽管大语言模型在完整测试集和简单测试集上表现不如PLM，但它在困难测试集上超越了PLM，两个模型结合后取得了更高的分数。

    arXiv:2611.08967v1 Announce Type: new  Abstract: Named entity recognition (NER) is challenging for social media data in a low-resource language. LLMs often lag behind pretrained language models (PLMs) in NER performance, but they show promise in handling cases that PLMs struggle with. Using a dataset of Nepali tweets, we split the test set into "easy" and "hard" cases using token probabilities from a fine-tuned pretrained language model (PLM). We then evaluate an LLM on the hard test cases using 16 different prompt configurations. We find that selecting examples based on semantic similarity and increasing the number of examples improves F1 score by almost 10 points from the baseline configuration. Although the LLM underperforms the PLM on the full and easy test sets, it surpasses the PLM on the hard test set, resulting in an improved score when the two models are combined.
    
[^51]: ResQAdam：面向内存高效量化感知训练的残差保持自适应优化

    ResQAdam: Residual-Preserving Adaptive Optimization for Memory-Efficient Quantization Aware Training

    [https://arxiv.org/abs/2611.08924](https://arxiv.org/abs/2611.08924)

    ResQAdam通过仅维护单个BF16动量、只对主子空间的r×r坐标应用Adam，并利用残差的归一化符号更新来保留被投影方法丢弃的动量信息，在权重量化至2比特的量化感知训练中将优化器状态内存相比AdamW减少71%至74%，同时保持接近的性能。

    

    量化感知训练（QAT）是获得精确低比特大语言模型（LLMs）最可靠的方法，但使用AdamW训练时，每个权重矩阵都需要维护两个密集的FP32状态。我们发现GaLore和LDAdam等低秩内存高效优化器在QAT中性能大幅下降。相当一部分QAT动量位于其前r个奇异子空间之外，而基于投影的方法在每一步都会丢弃这部分信息。我们进一步观察到，即使仅保留残差的符号，这部分残差仍然有用。基于这些观察，我们提出了ResQAdam。它仅维护单个BF16动量，只对主子空间的r×r坐标应用Adam，并添加一个由残差计算得到的归一化符号更新。我们使用两种从头训练的QAT方法和三种QAT微调流程评估了ResQAdam，权重被量化至2比特。与AdamW相比，该方法将优化器状态内存减少了71%至74%，同时保持性能接近……

    arXiv:2611.08924v1 Announce Type: cross  Abstract: Quantization-aware training (QAT) is the most reliable way to obtain accurate low-bit large language models (LLMs), but training with AdamW keeps two dense FP32 states for every weight matrix. We find that low-rank memory-efficient optimizers such as GaLore and LDAdam degrade substantially in QAT. A considerable part of the QAT momentum lies outside its top-r singular subspace, and projection-based methods discard this part at every step. We further observe that this residual remains useful when only its sign is kept. Based on these observations, we propose ResQAdam. It keeps a single BF16 momentum, applies Adam only to the r x r coordinates of the principal subspace, and adds a normalized sign update computed from the residual. We evaluate ResQAdam with two from-scratch QAT methods and three QAT fine-tuning pipelines, with weights quantized down to 2 bits. It reduces optimizer-state memory by 71% to 74% relative to AdamW, stays close 
    
[^52]: 论do-演算在通用可识别性问题上的不完备性

    On the Incompleteness of do-calculus for Universal Identifiability

    [https://arxiv.org/abs/2611.08916](https://arxiv.org/abs/2611.08916)

    本文为do-演算的完备性提出了严格的形式化定义，并指出在从任意观测与干预分布集合中进行因果效应识别的通用可识别性问题中，do-演算并不完备。

    

    摘要：我们研究了从任意给定的观测分布和/或干预分布集合中进行因果效应识别的问题，并将其称为通用可识别性。uID问题推广了现有的经典可识别性和广义可识别性问题，后者仅考虑了从单纯的观测分布或特定类型的干预分布中进行识别的问题。先前的研究已经建立了图判据来确定在ID和gID问题中因果效应何时可识别，此外还提出了利用do-演算和基本概率运算的算法来计算这些效应。因此，已有结果表明do-演算在这些问题上既是可靠的也是完备的。尽管有这些成熟的研究成果，但对于do-演算完备性的含义，目前尚不存在正式的定义。我们为do-演算的完备性提出了一个严格的正式化定义……

    arXiv:2611.08916v1 Announce Type: new  Abstract: We study the problem of causal effect identification from an arbitrary set of observational and/or interventional distributions, which we call Universal Identifiability (uID). The uID problem generalizes existing classical identifiability (ID) and general identifiability (gID) problems, which considered the identification problem from either mere observational distributions or a specific type of interventional distributions. Prior research has established graphical criteria to determine when a causal effect is identifiable in ID and gID problems. Moreover, algorithms that leverage do-calculus and basic probability manipulations have been proposed to compute these effects. Consequently, it has been shown that do-calculus is both sound and complete in these problems. Despite these well-established results, a formal definition of what it means for the do-calculus to be complete does not exist. We propose a rigorous formalization for the com
    
[^53]: 从流程到权限：从智能体技能中学习精确的子智能体授权

    From Procedures to Permissions: Learning Precise Subagent Authorization from Agent Skills

    [https://arxiv.org/abs/2611.08905](https://arxiv.org/abs/2611.08905)

    该论文提出将Agent Skills中编码的过程性知识（即每一步所需的操作、工具和文件信息）作为天然监督信号，通过从技能合成任务并在交互环境中以子智能体委派方式执行，来学习为子智能体授予精确的工具与工作区资源权限。

    

    随着语言智能体处理日益复杂的任务，它们越来越多地依赖多个智能体来协调专业化工作，这不仅需要决定每个子智能体应该做什么，还需要决定它应该访问哪些工具和工作区资源。现有的智能体系统提供了控制工具和执行环境的机制，但并未直接指定每个被委派子任务所需的资源。此外，仅凭任务结果和执行轨迹，难以为判断子智能体是否获得了适当权限提供充分的监督信号。我们观察到，智能体技能编码了描述每一步所需操作、工具和文件的过程性知识，为授权提供了天然的监督来源。基于这一观察，我们从技能中合成任务和分步流程，在交互环境中通过子智能体委派来执行这些任务，并利用经过验证的轨迹……（原文摘要在此处截断）

    arXiv:2611.08905v1 Announce Type: new  Abstract: As language agents tackle increasingly complex tasks, they increasingly rely on multiple agents to coordinate specialized work, requiring decisions not only about what each subagent should do but also which tools and workspace resources it should access. Existing agent systems provide mechanisms for controlling tools and execution environments, but do not directly specify the resources required by each delegated subtask. Moreover, task outcomes and execution trajectories alone provide limited supervision for determining whether a subagent has received appropriate permissions. We observe that Agent Skills encode procedural knowledge describing the operations, tools, and files required at each step, providing a natural source of supervision for authorization. Building on this observation, we synthesize tasks and step-by-step procedures from Skills, execute them through subagent delegation in interactive environments, and use verified traje
    
[^54]: 丰富性并非均匀性：自训练下的雷尼熵有序交叉

    Richness Is Not Evenness: Ordered Renyi Crossings under Self-Training

    [https://arxiv.org/abs/2611.08892](https://arxiv.org/abs/2611.08892)

    该论文通过九代GPT-2递归自训练实验首次发现不同阶雷尼熵存在有序的温度交叉点，证明词汇丰富性与分布均匀性是两个相互独立的维度：词类型数量减少的同时，高频词的主导性反而降低。

    

    语言模型的递归自训练——模型坍缩研究中考察的一种设定——能够在缩减词汇量的同时，降低高频词（主要是功能词）的主导程度。我们让英文GPT-2在自身输出上进行九代微调，改变训练数据的采样温度，并用同一固定采样器测量每个模型。随着训练温度升高，各阶雷尼熵的变化在不同的温度点依次由损失转为增益：最小熵最先发生交叉，随后是碰撞熵、香农熵和观测丰富度。在最外侧两个交叉点之间，词类型数量下降，而最常用词的主导性却在减弱。T=1下的未截断训练恰好位于该区间：在13,000词的规模上，第9代的词类型数量比第0代少24%，但碰撞熵却高出0.91比特。这一排序在更细的温度网格、不同的测量采样器、固定的词预算以及五个全新的训练种子下均保持不变。在另外四种设定中，最小熵同样先于丰富度发生交叉（原文在此处截断）。

    arXiv:2611.08892v1 Announce Type: new  Abstract: Recursive self-training of language models, a setting studied in model collapse, can reduce vocabulary while making frequent words, mostly function words, less dominant. We fine-tune English GPT-2 on its own output for nine generations, vary the training-data sampling temperature, and measure every model with one fixed sampler. As training temperature rises, Renyi entropy changes turn from loss to gain at distinct temperatures: min-entropy first, then collision entropy, Shannon entropy and observed richness. Between the outer crossings, word types fall while the most-common word becomes less dominant. Untruncated training at T = 1 lies in this interval: at 13,000 words, generation 9 has 24% fewer types than generation 0 yet collision entropy is 0.91 bits higher. The ordering persists with a finer grid, different measurement samplers, a fixed word budget and five fresh training seeds. Min-entropy again crosses before richness in four furt
    
[^55]: SteerGR：基于稀疏自编码器的生成式推荐引导，实现可控的游戏发现

    SteerGR: Steering Generative Recommendation via Sparse Autoencoders for Controllable Game Discovery

    [https://arxiv.org/abs/2611.08790](https://arxiv.org/abs/2611.08790)

    提出SteerGR框架，利用稀疏自编码器将人类可读的游玩概念与生成式推荐模型的内部特征关联，通过直接干预所选特征实现无需重训练的概念级可控游戏推荐。

    

    生成式推荐旨在从用户交互历史中预测物品标识符，但对推荐物品属性的显式控制仍未得到充分探索。一个关键挑战是将人类可读的概念与内部特征关联起来，使得对这些特征的干预能够有效改变推荐的物品。为解决这一问题，我们提出了可引导生成式推荐器（SteerGR），这是一个利用稀疏自编码器在游戏发现中实现概念级控制的框架。SteerGR通过三个阶段进行：（i）概念定义与接地，使用大语言模型定义游玩概念；（ii）特征发现与选择，根据概念关联发现候选特征，并根据其引导效应选择特征；（iii）稀疏特征干预，通过放大或抑制所选特征来引导推荐，而无需重新训练推荐器。在两个工业级生成式推荐模型上进行了大量实验……

    arXiv:2611.08790v1 Announce Type: cross  Abstract: Generative recommendation aims to predict item identifiers from user interaction histories, but explicit control over the properties of recommended items remains underexplored. A key challenge is to connect human-readable concepts to internal features whose interventions effectively change the recommended items. To address this, we propose Steerable Generative Recommender (SteerGR), a framework that uses sparse autoencoders for concept-level control in game discovery. SteerGR proceeds through three stages: (i) Concept Definition and Grounding for defining play concepts with an LLM; (ii) Feature Discovery and Selection for discovering candidate features by their concept associations and selecting one by its steering effect; and (iii) Sparse Feature Intervention for amplifying or suppressing selected features to steer recommendations without retraining the recommender. Extensive experiments with two industrial generative recommendation m
    
[^56]: 领域微调大语言模型中的护栏侵蚀：对生物医学大语言模型中知识型拒答能力的配对审计

    Guardrail Erosion in Domain Fine-Tuned Large Language Models: A Paired Audit of Knowledge-Centric Abstention in Biomedical LLMs

    [https://arxiv.org/abs/2611.08780](https://arxiv.org/abs/2611.08780)

    本研究通过对 25 个生物医学领域微调大语言模型与其基础模型的配对大规模审计，首次揭示微调会进一步侵蚀本已薄弱的知识型拒答能力，证明领域微调会导致模型安全护栏的系统性退化。

    

    领域微调模型的数量，尤其是在生物医学领域，正在迅速增长。微调承诺以远低于训练基础模型的成本获得高性能模型，但它也可能带来意料之外的行为改变。多项研究表明，安全护栏会随着微调而受到侵蚀，然而涵盖不同模型架构、数据配比和微调方法的大规模研究仍然有限。以往工作主要研究以价值观为中心的拒绝行为，而本研究聚焦于以知识为中心的拒答（abstention）。我们在 MedAbstain 和 ClinDet-Bench 两个基准上，将 25 个生物医学领域微调的大语言模型与其对应的 11 个基础模型检查点进行配对审计，以评估微调对知识型拒答能力的影响。我们发现，这些基准所度量的明确定义的适当拒答能力在基础模型中本就很低（MedAbstain 上中位数为 8.3%），微调后进一步下降（中位数降至 4.2%，并在……持续减少）

    arXiv:2611.08780v1 Announce Type: new  Abstract: The number of domain fine-tuned models, particularly in the biomedical domain, is rapidly expanding. Fine-tuning promises high-performing models at a fraction of the cost of training a base model, but it can also produce unintended behavioral changes. Multiple studies have shown that safety guardrails can erode with fine-tuning, yet large-scale studies across model architectures, data mixtures, and fine-tuning methodologies remain limited. Where prior work in this space has studied value-centric refusal, this work focuses on knowledge-centric abstention. We audit 25 biomedical domain fine-tuned large language models against 11 paired base checkpoints on MedAbstain and ClinDet-Bench to evaluate the impact of fine-tuning on knowledge-centric abstention. We find that defined appropriate abstention measured by these benchmarks, already low in base models (median 8.3% on MedAbstain), falls further after fine-tuning (median 4.2%, decreasing in
    
[^57]: 面向语言模型系统持久记忆的雅可比-因果记忆控制

    Jacobian-Causal Memory Control for Persistent Memory in Language Model Systems

    [https://arxiv.org/abs/2611.08779](https://arxiv.org/abs/2611.08779)

    该论文提出雅可比-因果记忆控制（JCMC），通过预测影响签名、目标效应与副作用预算下的干预选择、以及预测-观测对比的保留或回滚机制，来控制持久记忆对语言模型当前推理的因果影响，并为其提供理论保证和反事实评估协议。

    

    持久记忆系统不仅要选择检索哪些记录，还要决定哪些先前信息能够改变当前的推理过程。我们提出雅可比-因果记忆控制（JCMC），该方法在共享的行为测量空间中用预测的影响签名来表示每个被授权的记忆干预，依据目标效应和副作用预算来选择干预措施，并在保留或回滚干预之前比较预测效应与实际观测效应。在明确的平滑性和有界估计假设下，我们证明了二次局部近似界，刻画了一组签名可达到的目标效应集合，在存在签名误差的情况下界定了选择遗憾，并对同时施加的多个记忆之间的非加性交互给出了界。本论文未报告任何基准测试结果；它定义了一个配对反事实估计器以及一套评估协议，用于将JCMC与基于相关性、图结构和策略过滤的检索基线进行比较。

    arXiv:2611.08779v1 Announce Type: cross  Abstract: Persistent-memory systems choose not only which records are retrieved but which prior information can alter current inference. We propose Jacobian-causal memory control (JCMC), which represents each authorized memory intervention by a predicted influence signature in a shared behavioral measurement space, selects interventions against a target effect and side-effect budget, and compares predicted with observed effects before retaining or rolling back the intervention. Under explicit smoothness and bounded-estimation assumptions, we prove a quadratic local-approximation bound, characterize the target effects reachable by a set of signatures, bound selection regret under signature error, and bound non-additive interactions between simultaneous memories. The paper reports no benchmark measurements; it defines a paired counterfactual estimator and an evaluation protocol comparing JCMC with relevance, graph, and policy-filtered retrieval ba
    
[^58]: 语义条件相似度（SCS）：一个形式化数学框架

    Semantic Conditional Similarity (SCS): A Formal Mathematical Framework

    [https://arxiv.org/abs/2611.08778](https://arxiv.org/abs/2611.08778)

    本文提出语义条件相似度（SCS）这一形式化数学框架，通过集合论运算在消除与第三方文档共享的语义元素后计算文档相似度，为多文档场景引入方向性、角色不对称性及文档间条件化机制。

    

    语义文本相似度（STS）作用于文档对，每次比较返回一个单一的标量。在多文档场景中，当多个源文档可能对一个或多个目标文档有所贡献时，成对STS无法将每个源文档的边际语义贡献从它们共享的词汇中分离出来。本文引入语义条件相似度（SCS）作为STS的一种条件形式，它通过集合论运算，在消除与第三个文档共享的语义元素之后计算两个文档之间的相似度。SCS做出了四项贡献：1）它引入了方向性，将成对关系划分为横向（源与源之间）和纵向（源到目标）两类；2）条件化通过将比较域从无序对扩展到有序的、带条件的三元组（目标文档、参考文档和条件文档），诱导出依赖于角色的不对称性；3）SCS通过减法运算引入了文档间条件化……

    arXiv:2611.08778v1 Announce Type: new  Abstract: Semantic Textual Similarity (STS) operates on document pairs, returning a single scalar per comparison. In a multi-document setting where several sources may have contributed to one or more targets, pairwise STS cannot disentangle each source's marginal semantic contribution from their shared vocabulary. In this paper, Semantic Conditional Similarity (SCS) is introduced as a conditional form of STS that computes similarity between two documents after eliminating semantic elements shared with a third via set-theoretic operations. SCS makes four contributions: 1) it introduces directionality, partitioning pairwise relations into lateral (inter-source) and vertical (source-to-target) classes; 2) conditioning induces role-dependent asymmetry by extending the comparison domain from unordered pairs to ordered, conditioned triples of a target, a reference, and a conditioning document; 3) SCS introduces inter-document conditioning by subtracting
    
[^59]: AttentionSwitch：通过稀疏注意力头选择与缩放的联合学习实现大语言模型的越狱防御

    AttentionSwitch: Jailbreak Defense for LLMs via Joint Learning of Sparse Attention-Head Selection and Scaling

    [https://arxiv.org/abs/2611.08752](https://arxiv.org/abs/2611.08752)

    AttentionSwitch是首个通过联合学习稀疏注意力头的选择与缩放，同时优化越狱防御中“在哪里干预”和“如何干预”两个决策的大语言模型防御方法。

    

    大语言模型在广泛的任务中取得了卓越的性能，但仍然容易受到越狱攻击的影响，这类攻击会绕过安全机制并引发有害或非预期的输出。虽然可以通过模型微调来提升安全性，但此类方法的计算成本可能很高，并且可能由于灾难性遗忘而损害模型先前获得的能力。因此，近期的越狱防御工作开始探索对与不安全行为相关的特定模型组件进行有针对性的干预。然而，现有方法通常将两个本质上相关的决策解耦处理：在哪里进行干预以及如何修改所选组件。这种顺序式优化可能是次优的，因为最佳的干预位置可能取决于在该处施加的干预方式，反之亦然。在这项工作中，我们提出了AttentionSwitch，这是首个联合优化干预位置与干预方式的越狱防御方法。

    arXiv:2611.08752v1 Announce Type: cross  Abstract: Large language models have achieved remarkable performance across a wide range of tasks, yet they remain susceptible to jailbreak attacks that circumvent safety mechanisms and elicit harmful or unintended outputs. Although safety can be improved through model fine-tuning, such approaches can be computationally expensive and may degrade previously acquired capabilities through catastrophic forgetting. Recent jailbreak defenses have therefore explored targeted interventions on specific model components associated with unsafe behavior. However, existing approaches typically decouple two inherently related decisions: where to intervene and how to modify the selected components. Such sequential optimization may be suboptimal because the optimal intervention location can depend on the intervention applied there, and vice versa. In this work, we introduce AttentionSwitch, the first jailbreak defense to jointly optimize intervention sites and 
    
[^60]: 为LLM数字孪生构建调查历史结构：背景、决策程序与评估画像的对照研究

    Structuring Survey Histories for LLM Digital Twins: A Controlled Study of Background, Decision Procedure, and Evaluation Profiles

    [https://arxiv.org/abs/2611.08730](https://arxiv.org/abs/2611.08730)

    提出将受访者调查历史结构化为“背景—决策程序—评估”（BDE）的固定表示方法，相比原始调查记录可显著提升LLM数字孪生预测个体答案的准确率。

    

    基于LLM的数字孪生旨在根据个体先前回答的表示，预测其在新情境下的反应。现有证据表明，对调查历史进行压缩不一定会降低预测准确性，这说明表示的结构（而不仅仅是信息量）可能是一个重要的设计选择。我们研究了BDE——一种固定的表示方式，将每个受访者的调查历史组织为背景、决策程序和评估三个部分。我们在Twin-2K-500（Toubia等人，2025）数据集上，通过预测每个受访者在17个任务上的保留答案来对其进行评估。在一项针对100名受访者的配对评估中，完全推断的BDE画像相比原始调查记录，在gpt-5.4-nano上将整体准确率提升了1.27个百分点（p < 0.01），在gpt-5.4-mini上提升了1.19个百分点（p < 0.001）。随后，三项分析进一步考察了该模式的架构、提取深度以及组成部分的构成。与长度匹配的自由格式画像相比……

    arXiv:2611.08730v1 Announce Type: new  Abstract: LLM-based digital twins aim to predict how individuals would respond in new settings from representations of their prior responses. Existing evidence that compression of survey histories need not reduce predictive accuracy suggests that representation structure, rather than information volume alone, may be an important design choice. We study BDE, a fixed representation that organizes each respondent's survey history into Background, Decision procedure, and Evaluation. We evaluate it on Twin-2K-500 (Toubia et al., 2025) by predicting each respondent's held-out answers on 17 tasks. In a paired evaluation with 100 respondents, a fully inferred BDE profile improves overall accuracy over the raw survey transcript by 1.27 percentage points (pp) on gpt-5.4-nano (p < 0.01) and 1.19 pp on gpt-5.4-mini (p < 0.001). Three analyses then examine the schema, extraction depth, and component composition. Against a length-matched free-form profile writt
    
[^61]: 修补盲点：面向冻结欺诈检测器的可逆LLM规则叠加层

    Patching the Blind Spots: Reversible LLM Rule Overlay for Frozen Fraud Detectors

    [https://arxiv.org/abs/2611.08714](https://arxiv.org/abs/2611.08714)

    提出一个可逆的LLM规则叠加框架，在不修改冻结欺诈检测模型的前提下，利用SHAP定位模型盲点并由LLM生成、筛选、校准规则，通过多模型logit平均池化，在真实电商欺诈检测中于同等精确率下平均提升召回率2.26%。

    

    在工业表格型欺诈分类任务中，梯度提升树仍然是最先进的方法，但即使是训练良好的模型也存在盲点，而对已部署模型进行重新训练并不总是安全的。我们提出了一个无需修改模型即可对其进行修补的框架：LLM基于SHAP识别出的模型低估特征提出规则，随后对由此产生的含噪规则池进行实证支持筛选，并将其校准为一个可加的、可逆的对数提升叠加层，在生产截止阈值处运行。单个LLM带来的收益方差很高，且在不同数据集之间会出现反转，因此我们通过对三个不同的骨干模型进行等权logit平均来汇聚其结果，以牺牲峰值收益来换取有界的最坏情况表现。在一个基于选择性偏差数据训练的真实电商欺诈检测器上，即使经过精心调优，持续训练仍会导致性能退化，而该叠加层在同等精确率下平均提升召回率2.26%（ΔR@P），在同等召回率下平均提升精确率1.60%（ΔP@R）。

    arXiv:2611.08714v1 Announce Type: cross  Abstract: In industrial tabular fraud classification, gradient-boosted trees remain state-of-the-art, but even a well-trained model has blind spots, and retraining the deployed model is not always safe. We propose a framework that patches such a model without modifying it: LLMs propose rules anchored on features the model underweights by SHAP, and the resulting noisy pool is screened for empirical support and calibrated into an additive, reversible log-lift overlay that runs at the production cut-off. Single-LLM benefit is high-variance and inverts between datasets, so we pool three diverse backbones by equal-weight logit averaging, trading peak gain for a bounded worst case. On a real-world e-commerce fraud detector trained on selection-biased data, continued training regresses even when carefully tuned, whereas the overlay gains on average +2.26% recall at equal precision ($\Delta$R@P) and +1.60% precision at equal recall ($\Delta$P@R) over pr
    
[^62]: RL-ATTACK：一种基于强化学习的语言模型通用黑盒对抗攻击方法

    RL-ATTACK: A Generalized Black-Box Adversarial Attack on Language Models via Reinforcement Learning

    [https://arxiv.org/abs/2611.08711](https://arxiv.org/abs/2611.08711)

    该论文提出了RL-ATTACK，一种基于强化学习的可训练黑盒对抗攻击框架，通过从目标模型学习生成定制化对抗变换，在自然语言推理和生成式问答任务上均实现了更低的攻击后准确率、更高的查询效率和更好的语义保持。

    

    大型语言模型（LLMs）存在易受对抗性扰动影响的脆弱性，这些扰动可能引发模型产生意外响应。这些脆弱性可以通过有意攻击来识别，进而用于提升模型安全性。在无法获取梯度的黑盒设置中，现有方法倾向于借助外部知识来猜测对抗性变换，忽略了目标模型所提供的信息。为解决这一问题，我们提出了一种基于强化学习的可训练黑盒攻击框架，该框架通过从目标模型中学习来生成定制化的对抗性变换。我们证明，该方法在自然语言推理和生成式问答两类任务上均实现了更低的攻击后准确率、更高的查询效率以及更好的语义保持效果。在自然语言推理任务上，该方法仅需20次查询即可将攻击后准确率降至2%。针对生成式语言模型，我们训练的攻击器能够持续地（原文在此处截断）……

    arXiv:2611.08711v1 Announce Type: new  Abstract: Large language models (LLMs) exhibit vulnerabilities to adversarial perturbations that can induce unintended responses. These vulnerabilities can be identified via intentional attacks and subsequently used to improve model safety. In black-box settings where gradients are inaccessible, existing methods tend to guess adversarial transformations using external knowledge, ignoring the information provided by the target model. To mitigate this, we propose a trainable black-box framework based on reinforcement learning, which learns from the target model to generate tailored adversarial transformations. We demonstrate that our approach achieves lower post-attack accuracy, greater query efficiency and better semantic preservation on both natural language inference and generative Q&A tasks. On natural language inference, it drives post-attack accuracy to 2% within 20 queries. Against generative language models, our trained attacker consistently
    
[^63]: 电子商务支付欺诈检测中面向拒绝推断感知强化学习的双流LLM引导奖励设计

    Dual-Stream LLM-Guided Reward Design for Reject-Inference-Aware Reinforcement Learning in E-Commerce Payment Fraud Detection

    [https://arxiv.org/abs/2611.08706](https://arxiv.org/abs/2611.08706)

    该论文提出由单个LLM设计的双流奖励机制，通过让LLM自行决策如何利用风险信号对比新旧评分，为被旧系统拦截的无标签交易打分，从而在强化学习欺诈检测中解决拒绝推断偏差问题。

    

    生产级欺诈检测模型只能从其自身流量的被截断部分中学习：被旧系统拦截的交易永远不会获得欺诈标签，因此模型只能在有偏的、仅包含已通过交易的分布上进行训练。我们扩展了LLM引导的奖励设计——即Qu等人（2025）提出的框架，其中LLM演化两步REINFORCE欺诈拦截策略的奖励——以在奖励函数内部解决这一拒绝推断问题。单个LLM设计出双流奖励：流1对有标签交易应用标准的美元加权精确率目标；流2通过将最新的梯度提升风险信号与驱动原始拦截决策的旧系统评分进行对比，为被旧系统拦截的无标签交易打分。我们并不预先规定哪些信号重要，而是提供证据的超集，让LLM仅凭评估反馈自行决定使用哪些信号；由于它将奖励编写为代码，这一选择是直接可审计的。

    arXiv:2611.08706v1 Announce Type: cross  Abstract: Production fraud-detection models learn from a censored slice of their own traffic: transactions a legacy system blocked never receive fraud labels, so models train on a biased, approved-only distribution. We extend LLM-guided reward design, the framework of Qu et al. (2025) in which an LLM evolves the reward of a two-step REINFORCE fraud-blocking policy, to attack this reject-inference problem inside the reward. A single LLM designs a dual-stream reward: stream 1 applies the standard dollar-weighted precision objective to labeled transactions; stream 2 scores legacy-blocked, unlabeled transactions by contrasting the newest gradient-boosted risk signal against the legacy scores that drove the original block. Rather than prescribe which signals matter, we expose a superset of evidence and let the LLM decide for itself which to use, guided by evaluation feedback alone; because it writes the reward as code, that choice is directly auditab
    
[^64]: 语境性与矛盾性：探索对AI生成文本的现实世界识别实践

    Contextuality and Contradictions: Exploring Real-World Identifications of AI-Generated Writing

    [https://arxiv.org/abs/2611.08704](https://arxiv.org/abs/2611.08704)

    本研究结合定性编码、词汇变化的计算分析与语言人类学视角，考察了维基百科编辑者与学生识别AI生成文本的真实世界实践，发现这些识别依赖多样且相互矛盾的证据类型——从文本特征与内容判断到循环论证及对作者道德品质的假设。

    

    随着AI生成的内容扩散到人类互动的几乎每一个领域，无论是正式还是非正式的AI检测实践与话语，已成为人们与文本以及彼此互动的日常组成部分。在本文中，我们探讨了在话语格局发生变化的背景下涌现出的现实世界的AI识别实践与话语——在这种新格局中，文本的人类来源已不再能被视为理所当然。我们将定性编码与词汇变化的计算分析相结合，并以语言人类学关于语言与传播的视角为基础，调查了英语维基百科编辑者和学生群体中的AI识别行为。我们发现，这些识别行为借助了多种多样的证据类型，从离散的文本特征和对内容的判断，到循环的意义建构式论断，以及对作者道德品质的预设，而且这些识别是相互矛盾的、动态的……

    arXiv:2611.08704v1 Announce Type: new  Abstract: As AI-generated content spreads into nearly every domain of human interaction, practices and discourses of AI detection, both formal and informal, have become an everyday part of how people interact with text and with each other. In this paper, we explore the real-world practices and discourses of AI identification that have emerged in response to a shifting discursive landscape where the human provenance of a text can no longer be taken for granted. We pair qualitative coding with computational analysis of lexical change, grounded in linguistic anthropological perspectives on language and communication, to investigate AI identifications among English Wikipedia editors and students. We find that these identifications draw on a diversity of evidentiary types, from discrete textual features and judgments about content to circular sense-making claims and presumptions about the author's moral character, and that they are contradictory, dynam
    
[^65]: 语言模型表示中的情感压缩

    Compression of Emotion in Language Model Representations

    [https://arxiv.org/abs/2611.08683](https://arxiv.org/abs/2611.08683)

    该研究通过对七个大语言模型在GoEmotions数据集上的隐藏表示进行无标签聚类分析，发现模型内部对情感的编码存在“过度压缩”现象——其最清晰的情感区分仅为积极与消极两大类，远粗于28个细粒度情感类别。

    

    大型语言模型（LLMs）在识别文本中表达的情感以及生成流畅情感语言的能力方面已被广泛评估。然而，仅凭输出预测无法揭示细粒度的情感区分是如何在其内部组织的。我们在GoEmotions数据集上研究了七个大语言模型，该数据集是一个英语Reddit语料库，标注了27个细粒度情感类别外加中性类别。我们将模型预测以及由情感提示得到的末位标记（final-token）表示与GoEmotions的28个类别，以及更粗粒度的情感家族和情感极性分组进行了比较。我们发现，相对于细粒度的GoEmotions分类体系，模型表现出一致的“过度压缩”模式。在所测试的KMeans协议下，当我们不使用情感标签对其隐藏表示进行聚类时，最清晰的划分是分为两组，大体上将积极情感与消极情感区分开来。更粗粒度的情感分组也能用比完整的28个类别更少的聚类数恢复出来。

    arXiv:2611.08683v1 Announce Type: new  Abstract: Large language models (LLMs) have been evaluated on their ability to identify emotions expressed in text and produce fluent emotional language. However, output predictions alone do not reveal how fine-grained emotion distinctions are organized internally. We study seven LLMs on GoEmotions, an English Reddit dataset annotated with 27 fine-grained emotion categories plus neutral. We compare model predictions and emotion-prompted final-token representations with the 28 GoEmotions categories and with coarser emotion-family and sentiment groupings.   We find a consistent pattern of $\textit{overcompression}$ relative to the fine-grained GoEmotions taxonomy. Under the tested KMeans protocol, when we cluster their hidden representations without using emotion labels, the clearest split is into two groups that largely separate positive from negative emotion. Coarser emotion groupings are also recovered with fewer clusters than the full 28-categor
    
[^66]: 贪心字节对编码至少能达到最优压缩的 0.361 倍

    Greedy Byte-Pair Encoding Achieves at Least 0.361 of the Optimal Compression

    [https://arxiv.org/abs/2611.08671](https://arxiv.org/abs/2611.08671)

    本文证明贪心字节对编码在任意字符串、预算和平局打破规则下，至少能达到最优压缩缩减量的 0.361 倍，实现了对先前三分之一下界的首次改进。

    

    字节对编码（BPE）通过反复合并出现频率最高的一对相邻符号来构建分词器；其效用是 k 次合并所实现的总字符串长度缩减量。此前已知这一贪心规则至少能达到最优缩减量的三分之一。我们证明，对于任意字符串、任意预算以及任意平局打破规则，贪心规则总能达到最优字符串长度缩减量的至少 0.361 倍；这是对三分之一这一界的首次改进。我们还解释了为什么三分之一曾是一个屏障：其背后的配对打包松弛是紧的，因此更好的界必须利用最优解的更多结构。当贪心算法一次性破坏最优解的三个合并点时，其中一个通常可以被移到隔壁，因为最优解中同一条规则的所有副本看起来是相同的；其余情况则会在之后得到补偿。在此过程中，我们精确确定了两次合并情形的最坏情况比率，为三次合并情形给出了界，并分离出单个……

    arXiv:2611.08671v1 Announce Type: cross  Abstract: Byte-Pair Encoding (BPE) builds a tokeniser by repeatedly merging a most frequent adjacent pair of symbols; its utility is the total reduction in string length achieved by $k$ merges. It was known that this greedy rule achieves at least a third of the optimal reduction. We prove that it always achieves at least 0.361 of the optimal reduction in string length, for every string, every budget and every tie-breaking rule; this is the first improvement of the bound one third. We also explain why one third was a barrier: the pair-packing relaxation behind it is tight, so a better bound must use more of the optimum's structure. When greedy destroys three of the optimum's merge points at once, one of them can usually be moved next door, because all copies of a rule of the optimum look alike; the remaining cases are paid for later. Along the way we determine the worst-case ratio for two merges exactly, bound it for three, and isolate a single i
    
[^67]: ChainFlow：用于投机解码的少步潜空间流匹配

    ChainFlow: Few-Step Latent Flow Matching for Speculative Decoding

    [https://arxiv.org/abs/2611.08669](https://arxiv.org/abs/2611.08669)

    ChainFlow提出首个以目标模型隐状态为条件的流匹配草稿模型，通过一至两步的潜空间短程传输生成草稿，使草稿成本几乎不随目标模型规模增长，同时兼顾并行生成与token间依赖。

    

    投机解码已成为加速大语言模型推理的标准方法：轻量级草稿模型提出若干token，目标模型通过一次并行前向传播进行验证。现有草稿模型面临限制其加速效果的权衡：自回归草稿模型逐个生成token，草稿成本随草稿长度增长；扩散草稿模型虽可并行生成token，但独立预测每个位置，忽略了token间的依赖关系。我们提出ChainFlow，一种新的投机解码范式，也是首个以目标模型隐状态为条件的流匹配草稿模型。ChainFlow将草稿生成视为隐状态上的短程传输，只需一或两步流步骤，而无需像从噪声生成那样进行多次去噪。变分自编码器压缩这些状态，使草稿成本几乎不随目标模型规模增长；轻量级反馈头恢复token间的依赖关系；树形草稿机制则提高了接受率。在Qwe

    arXiv:2611.08669v1 Announce Type: cross  Abstract: Speculative decoding has become a standard way to accelerate LLM inference: a lightweight drafter proposes tokens that the target verifies in one parallel pass. Existing drafters face trade-offs that limits their speedups: autoregressive drafters generate tokens one at a time, so drafting cost grows with draft length, while diffusion drafters generate tokens in parallel but predict each position independently, ignoring inter-token dependencies. We introduce ChainFlow, a new speculative decoding paradigm and the first flow-matching drafter conditioned on the target's hidden states. ChainFlow treats drafting as a short transport over hidden states, needing one or two flow steps instead of the many denoising steps of generation from noise. A variational autoencoder compresses these states so drafting cost barely grows with target size, lightweight feedback heads restore inter-token dependencies, and tree drafting raises acceptance. On Qwe
    
[^68]: 土耳其语多模态视频事实核查：新数据集及零样本智能体系统与微调系统的比较

    Turkish Multimodal Video Fact-Checking: A New Dataset and a Comparison of Zero-Shot Agentic and Fine-Tuned Systems

    [https://arxiv.org/abs/2611.08651](https://arxiv.org/abs/2611.08651)

    该论文发布了首个土耳其语多模态视频事实核查数据集TeViD，并基于同一开源基础模型构建和比较了零样本智能体验证流水线与微调系统两种事实核查方案。

    

    基于视频的虚假信息在社交媒体上传播迅速，但几乎所有现有的事实核查系统都仅专注于英文文本，土耳其语等低资源语言的相关工具严重匮乏。在本工作中，我们提出了TeViD（土耳其语视频事实核查数据集），据我们所知，这是首个面向土耳其语的多模态视频事实核查数据集。我们从IFCN认证组织Teyit.org发布的文章中构建了该数据集，并通过翻译的Snopes文章和合成样本对其进行扩充，以缓解类别不平衡问题。基于TeViD，我们开发了两个共享同一开源基础模型的验证流水线。其一为3MFact-TR（流水线1），我们使用Qwen2.5-7B大语言模型和Qwen2-VL-7B视觉语言模型将多角色3MFact框架适配到土耳其语，并加入了三个自研模块：语义不一致性检查器、5W1H问题管理器以及双语证据检索器。

    arXiv:2611.08651v1 Announce Type: cross  Abstract: Video-based misinformation spreads rapidly on social media, but almost all existing fact-checking systems focus exclusively on English text. Tools for low-resource languages such as Turkish are severely limited. In this work, we present TeViD (Turkish Video Fact-checking Dataset), to the best of our knowledge the first multimodal video fact-checking dataset for Turkish. We compiled the dataset from articles published by Teyit.org, an IFCN-certified organisation, and enriched it with translated Snopes articles and synthetic examples to reduce class imbalance. Building on TeViD, we develop two verification pipelines that share the same open-source base model. First, in 3MFact-TR (Pipeline 1), we adapt the multi-role 3MFact framework to Turkish with the Qwen2.5-7B large language model and the Qwen2-VL-7B vision-language model, and add three modules of our own: a semantic inconsistency checker, a 5W1H question manager, and a bilingual evid
    
[^69]: FinEvolveKG：面向金融知识图谱构建的可控且可归因的提示词演化

    FinEvolveKG: Controlled and Attributable Prompt Evolution for Financial Knowledge Graph Construction

    [https://arxiv.org/abs/2611.08643](https://arxiv.org/abs/2611.08643)

    FinEvolveKG通过将可复用的通用模块与角色策略模块分离，并将反思式提示词优化器GEPA的每次编辑严格限制在单独版本管理的角色策略模块内，实现了金融知识图谱构建中可控、可归因且不破坏共享规则的提示词演化。

    

    金融机构从共享的SEC 10-K文件构建知识图谱（KG），以服务风险分析、投资组合管理和投资研究，而这些应用场景的分析需求各不相同。抽取过程必须能够灵活适应这些差异，同时不破坏共享的实体落地、类型体系与时间规则。手工编写的角色化提示词和模型微调都会增加成本。无需权重的提示词优化提供了一种替代方案，但对单一整体提示词进行无约束演化，可能会在修改角色引导的同时重写共享规则，从而掩盖变更内容并使审查变得复杂。我们提出FinEvolveKG，这是一个知识图谱构建框架，它将可复用的通用模块与角色策略模块分离开来，并将GEPA（一种反思式提示词优化器）机械性地限制在指定模块内进行操作。在角色适配过程中，已批准的通用模块保持字节级稳定，优化器的每一次编辑都被局部化到单独进行版本管理的角色策略模块中。一个双重目标在通用时序知识图谱质量与角色相关性之间取得平衡。

    arXiv:2611.08643v1 Announce Type: new  Abstract: Financial institutions construct knowledge graphs (KGs) from shared SEC 10-Ks to serve risk analysis, portfolio management, and investment research, whose analytical needs differ. Extraction must adapt without fragmenting shared grounding, typing, and temporal rules. Hand-authored persona prompts and fine-tuning add cost. Weight-free prompt optimization offers an alternative, but unconstrained evolution of a monolithic prompt can rewrite shared rules alongside persona guidance, obscuring changes and complicating review. We introduce FinEvolveKG, a KG-construction harness separating reusable general and persona-policy blocks and mechanically confining GEPA, a reflective prompt optimizer, to the designated block. During persona adaptation, the approved general block remains byte-stable and every optimizer edit is localized to a separately versioned persona policy. A dual objective balances generic temporal-KG quality with persona-relevant 
    
[^70]: 强制选择字母抑制拒答：来自WMDP数据集的机制性证据

    Forcing a Letter Suppresses Refusal: Mechanistic Evidence from WMDP

    [https://arxiv.org/abs/2611.08639](https://arxiv.org/abs/2611.08639)

    该论文首次从机制层面解释了多项选择题（MCQ）格式为何会抑制LLM的拒答：模型会过早地锁定答案字母，其内部激活显著远离拒答方向，且消融承载答案的注意力头会使拒答回升，证明答案选择与拒答存在直接竞争。

    

    大型语言模型（LLM）的安全基准通常采用多项选择题（MCQ）格式。然而，先前的研究表明，MCQ格式会降低模型的拒答率，这可能显著影响模型的安全行为。我们在大规模杀伤性武器代理数据集（WMDP）上，针对两个模型——Llama-3.1-8B 和 Qwen3-8B——首次提供了对这一效应的机制性解释。首先，我们使用调优透镜（tuned lens）技术表明，在处理MCQ时，模型会非常迅速地提交到某个答案字母token，这与开放式格式中拒答逐步建立的过程形成鲜明对比。此外，我们观察到MCQ格式下的激活与拒答方向的距离大约是开放式格式下的3倍。而且，消融承载MCQ格式答案的注意力头会增加拒答，这直接证明了MCQ的答案选择过程与拒答过程之间存在竞争。这些发现……（原文摘要至此截断）

    arXiv:2611.08639v1 Announce Type: new  Abstract: Safety benchmarks for large language models (LLMs) commonly use multiple-choice question (MCQ) format. However, prior studies documented that MCQ format reduces refusal rates, which may significantly affect the safety behavior of the model. We provide a first mechanistic explanation of this effect on the Weapons of Mass Destruction Proxy (WMDP) dataset across two models - Llama-3.1-8B and Qwen3-8B. First, we use the tuned lens technique to show that when processing MCQ the model commits to an answer-letter token very quickly, which is in contrast to the gradual refusal build-up in open-ended formats. Additionally, we observe that activations on the MCQ format lie roughly 3x further from the refusal direction, compared with their open-ended counterparts. Moreover, ablating the attention heads that carry the MCQ format answer increases the refusal, which is a direct evidence that the MCQ answer-selection competes with refusal. These findin
    
[^71]: 我们能压缩到多小？为SAE训练校准激活缓存压缩

    How Small Can We Go? Calibrating Activation-Cache Compression for SAE Training

    [https://arxiv.org/abs/2611.08551](https://arxiv.org/abs/2611.08551)

    该论文通过冻结SAE测试和重训练测试研究激活缓存压缩对稀疏自编码器训练的影响，发现特征保留比例缺乏清晰规律，且SAE训练对微小的数值扰动也异常敏感。

    

    训练稀疏自编码器（SAE）需要海量的模型激活缓存。我们通过两种方式研究压缩对SAE的影响：一是保留原始用fp32训练的SAE，将压缩后的缓存输入同一SAE，观察有多少特征激活保持不变（冻结SAE测试）；二是在压缩后的缓存上重新训练SAE，观察有多少原始特征仍然存在（重训练测试）。我们发现在重训练测试中，保留特征的比例没有清晰的规模规律。在GPT-2缓存上，将所有数值四舍五入为fp16保留了60.6%的特征，而仅改变百万分之一的数值且变化幅度为fp32允许的最小步长（比前者小八个数量级）只保留了70.5%的特征；相比之下，改变随机种子则保留了23.0%的特征。对于冻结SAE测试，冻结SAE与原始结果的相关性随编解码器精度的提高而递增，但简单的L2距离就能给出几乎相同的排序。

    arXiv:2611.08551v1 Announce Type: cross  Abstract: To train a sparse autoencoder (SAE), you need a massive cache of model activations. We look into the effects that compression has on an SAE in two ways: we keep the original fp32-trained SAE and pass the compressed cache into the same SAE to see how many feature activations stay the same (the frozen-SAE test), and we retrain an SAE on the compressed cache and see how many of the original features still exist (the retraining test). We found that in the retraining test, there is no clear scale for retained features. On a GPT-2 cache, rounding everything to fp16 keeps 60.6% of features, but changing one value in a million by the smallest step fp32 allows (a change eight orders of magnitude smaller) keeps only 70.5%. Meanwhile, changing the random seed keeps 23.0%. For the frozen-SAE test, the frozen SAE's correlation with the original increases in order of how precise the codecs are, but plain $L_2$ distance gives nearly the same ordering
    
[^72]: 视觉-语言模型中的潜意识学习

    Subliminal Learning in Vision-Language Models

    [https://arxiv.org/abs/2611.08538](https://arxiv.org/abs/2611.08538)

    本研究证明潜意识学习现象同样存在于视觉-语言模型中——教师VLM能够通过与偏好完全无关的几何形状数字生成任务，将隐藏的动物或面部偏好传递给学生模型，传递差异最高可达约10个百分点。

    

    潜意识学习是大型语言模型（LLM）中的一种现象，即教师模型通过与该偏好无关的训练数据将隐藏的偏好传递给学生模型。我们研究潜意识学习是否也发生在视觉-语言模型（VLM）中。我们微调一个教师VLM使其偏好特定的动物或面部属性，并使用来自同一基础模型的对照教师。两个教师模型为一个涉及几何形状的无关视觉任务生成数字。两个从相同基础VLM初始化的学生模型在相同的图像上进行微调，分别使用由有偏教师或对照教师生成的数字。在参数量从2B到7B的多个VLM上，我们发现动物偏好和面部偏好都能通过这个无关任务进行传递。在有偏教师输出上训练的学生比对照学生更频繁地选择目标，在动物偏好任务中差异最高可达约10个百分点。传递可以……

    arXiv:2611.08538v1 Announce Type: cross  Abstract: Subliminal learning is a phenomenon in large language models (LLMs) where a teacher passes a hidden preference to a student through training data unrelated to that preference. We study whether subliminal learning also occurs in vision-language models (VLMs). We finetune a teacher VLM to prefer a specific animal or facial attribute and use a control teacher from the same base model. Both teachers generate numbers for an unrelated visual task involving geometric shapes. Two students initialized from the same base VLM are finetuned on the same images, using numbers generated by either the biased or control teacher. Across VLMs ranging from 2B to 7B parameters, we find that both animal and facial preferences can transfer through this unrelated task. Students trained on biased-teacher outputs select the target more often than control students, with a difference of up to $\sim$10 percentage points in the animal preference task. Transfer can 
    
[^73]: 结构就在缩放之中：通过维度缩放探测神经对称性

    Structure is in the zoom: probing neural symmetry through dimensionality scaling

    [https://arxiv.org/abs/2611.08506](https://arxiv.org/abs/2611.08506)

    该论文提出将维度缩放指数分解为抽样底线与沿特定轴的结构性偏移，证明该斜率并非神经系统的固有属性，并借此在小鼠V1中检测出神经编码的O(2)对称性。

    

    神经群体被测得的维度中，有多少属于编码本身，又有多少只是测量行为带来的？参与率随子样本规模变化的斜率，常被当作属于系统本身的数字来报告，涵盖皮层群体以及语言模型的各层。但事实并非如此。一小块包含一万一千个小鼠V1神经元的区域给出三个斜率：沿随机刺激子集为0.25，沿漂移方向为0.31，沿神经元数量为0.35——只有中间那个携带结构，另外两个只是抽样底线。我们将每个指数拆分为该底线和沿某一声明轴获得的偏移量，并对照两个零假设来解读该偏移：打乱标签，以及仅在共享某一混杂因素的样本之间打乱标签。在V1上，与方向对齐的偏移量在全部八条光栅记录中都通过了标签零假设检验，在另一个实验室的32只动物中全部为正，并读取到了编码O(2)对称性的两个扇区。

    arXiv:2611.08506v1 Announce Type: cross  Abstract: How much of a neural population's measured dimensionality is the code, and how much is the act of measuring it? The slope of the participation ratio against subsample size is reported for cortical populations and for the layers of language models as a number that belongs to the system. It does not. One patch of eleven thousand mouse V1 neurons returns three slopes, 0.25 along random stimulus subsets, 0.31 along drift direction, 0.35 along neuron count, and only the middle one carries structure; the other two are the sampling floor.   We split every exponent into that floor and a shift earned along a declared axis, and read the shift against two nulls: permuting the labels, and permuting them only among samples that share a nuisance. On V1 the direction-aligned shift clears the label null in eight of eight grating recordings, is positive in 32 of 32 animals in a second laboratory, and reads both sectors of the code's O(2) symmetry. The 
    
[^74]: 使用语义分块和混合检索策略改进长文档的上下文检索

    Improving Contextual Retrieval for Long Documents using Semantic Chunking and Hybrid Retrieval Strategies

    [https://arxiv.org/abs/2611.08498](https://arxiv.org/abs/2611.08498)

    该论文提出一种混合RAG框架，通过知识图谱结构化推理与稠密向量检索相结合，配合语义分块和分层检索机制，有效解决了长文档多跳推理中的上下文碎片化和检索精度问题。

    

    篇幅冗长且内容分散的文档中包含的海量信息使得查找相关答案变得困难，尤其是当答案需要连接来自多个来源的证据时。标准的检索增强生成（RAG）系统通常难以应对这种“多跳”推理，并且经常面临上下文碎片化的问题。在本文中，我们提出了一种混合RAG框架，该框架将知识图谱的结构化推理与稠密向量检索的广泛语义覆盖相结合。我们的方法采用多阶段流水线，利用基于transformer的摘要生成和语义分块来改善上下文保留。我们构建了一个动态知识图谱来捕获实体之间的显式关系，同时维护一个稠密向量索引。在推理过程中，系统使用分层机制，优先进行图遍历以保证精确性，并在需要时回退到向量检索。

    arXiv:2611.08498v1 Announce Type: cross  Abstract: The vast amount of information available in lengthy, distributed documents makes finding relevant answers difficult, especially when the answer requires connecting evidence from multiple sources. Standard Retrieval-Augmented Generation (RAG) systems often struggle with this type of "multi-hop" reasoning and frequently face issues with context fragmentation. In this paper, we propose a hybrid RAG framework that integrates the structured reasoning of knowledge graphs with the broad semantic coverage of dense vector retrieval. Our approach employs a multi-stage pipeline using transformer-based summarization and semantic chunking to improve context retention. We construct a dynamic knowledge graph to capture explicit relationships between entities while simultaneously maintaining a dense vector index. During inference, the system uses a hierarchical mechanism that prioritizes graph traversal for precision and falls back to vector retrieval
    
[^75]: 重新思考数据选择：作为语义流形上的分布匹配问题

    Rethinking Data Selection as Distribution Matching on Semantic Manifolds

    [https://arxiv.org/abs/2611.08495](https://arxiv.org/abs/2611.08495)

    该论文提出QMAC方法，将数据选择重新表述为语义流形上分布匹配与质量过滤的联合优化问题，通过在多个频率尺度上匹配特征函数来同时捕捉全局语义覆盖与局部扰动，从而高效筛选出高质量、高代表性的训练数据子集。

    

    arXiv:2611.08495v1 公告类型：cross

    arXiv:2611.08495v1 Announce Type: cross  Abstract: Instruction tuning of large language models and large vision language models relies on massive training data, which incurs substantial computational and storage costs. Meanwhile, training data often contains redundancy, noise, and low quality samples. Existing data selection methods mostly rely on diversity, importance, or a combination of both, without explicitly constraining the distributional consistency between the selected subset and the full data in the embedding space. We propose QMAC, a quality aware multiband characteristic function matching method that formulates data selection as a joint optimization problem of distribution matching and quality filtering. QMAC matches the empirical characteristic functions of the selected subset and the full data across multiple frequency scales, capturing both global semantic coverage and local perturbations, while using task related quality scores to filter noisy and low quality samples. W
    
[^76]: RouteAlign：将任务需求与模型画像对齐以实现大语言模型服务选择

    RouteAlign: Aligning Task Requirements with Model Profiles for LLM Service Selection

    [https://arxiv.org/abs/2611.08162](https://arxiv.org/abs/2611.08162)

    RouteAlign通过将任务需求（能力需求与难度）与模型性能画像对齐，并利用共享评分器和生成成本偏好进行选择，从而在异构大语言模型服务中实现更精准的服务选择。

    

    大语言模型服务选择需要能够反映当前任务需求的性能证据。聚合分数掩盖了能力与难度上的差异，而语义上相似的示例可能涉及不同的技能组合。我们提出了RouteAlign，它利用任务需求来同时构建模型性能度量与服务选择的结构。任务需求描述了能力需求和难度，从而能够与相应的模型性能度量实现对齐。历史需求决定了结果如何贡献于全局画像与局部能力估计；新到来的需求则决定了这些度量如何为当前决策提供依据。一个共享的评分器整合了这些证据，而生成成本偏好则控制在符合条件的服务中进行选择。在多样化任务和异构大语言模型服务上的评估表明，RouteAlign扩展了经验性的质量-成本权衡边界。

    arXiv:2611.08162v1 Announce Type: cross  Abstract: LLM service selection requires performance evidence that reflects the demands of the current task. Aggregate scores obscure differences in capability and difficulty, while semantically related examples can involve different combinations of skills. We present RouteAlign, which uses task requirements to structure both the measurement of model performance and the selection of services. Task requirements describe capability demand and difficulty, enabling alignment with corresponding model performance measurements. Historical requirements determine how outcomes contribute to global profiles and local capability estimates; incoming requirements determine how those measurements inform the current decision. A shared scorer integrates the evidence, and a generation-cost preference controls selection among eligible services. Evaluation across diverse tasks and heterogeneous LLM services shows that RouteAlign extends the empirical quality-cost f
    
[^77]: 语言模型神经元中的原型语境：一种智能架构？

    Archetypal contexts in the neurons of language models: an architecture for intelligence?

    [https://arxiv.org/abs/2611.07234](https://arxiv.org/abs/2611.07234)

    该研究在Qwen3.8-27B模型中发现，跨领域的原型语境（如“临界点”）、主题和语言分别由三组可独立操控的神经元编码，且网络层呈现“外层语言—中间主题—核心原型”的三明治式组织架构，暗示这可能正是智能的底层架构。

    

    我们验证了此前的假设，即语言模型中存在着独立于具体实例化领域的关联模式（Nordfors, 2026）。这些“原型语境”是人们在众多领域中都能识别出的关系模式，例如“临界点”或“赢家通吃”，它们只被存储一次，便可跨领域应用。我们在模型 Qwen3.8-27B 内部展示了这一现象：一组神经元承载原型（如临界点），另一组承载主题（如政治、生态学或细胞生物学），第三组承载语言（如英语、西班牙语或中文）。当使承载“临界点”原型的神经元沉默后，模型不再识别该模式，但仍然知道文本讲的是什么；当使承载主题的神经元沉默后，模型失去主题信息，但仍能说出该模式。在神经元映射中，各层自组织成一种“三明治”结构：外层承载语言，其内的中间层承载主题，而最核心的层承载原型。三明治……（原文摘要在此处截断）

    arXiv:2611.07234v1 Announce Type: cross  Abstract: We validate our previous hypothesis that a language model holds relational patterns apart from the subject domains that instantiate them (Nordfors, 2026). These archetypal contexts are patterns of relations that people recognize across many fields, such as the tipping point or winner takes all, held once and applied across domains. We show this inside the model, Qwen3.8-27B. One set of neurons carries the archetype, say the tipping point; another carries the topic, say politics, ecology or cell biology; a third carries the language, such as English, Spanish or Chinese. Silence the tipping point's neurons, and the model no longer identifies the pattern but still knows what the text is about. Silence the topic's, and it loses the topic but still names the pattern. In the neuron maps the layers organize themselves into a sandwich: the outer layers carry languages, the layers inside them topics, the layers at the heart archetypes. The sand
    
[^78]: 超越准确率：语言子空间干预捕捉、移除与改进了什么

    Beyond Accuracy: What Language-Subspace Interventions Capture, Remove, and Improve

    [https://arxiv.org/abs/2611.05278](https://arxiv.org/abs/2611.05278)

    该研究发现基于 Mean-SVD 和 DiffPCA 构建的语言子空间虽能捕捉翻译相关的变化，但投影去除后语言信息仍可被线性探针近乎完美地识别，说明现有语言子空间干预并不能穷尽地移除语言信息。

    

    语言子空间干预旨在通过抑制中间层激活中与语言相关的成分来提升多语言推理能力。我们在四个开放权重大语言模型上使用 MATH 和 MGSM 数据集，提出三个问题：估计出的语言子空间是否捕捉了与翻译相关的变化？将这些子空间投影去除后是否能移除语言信息？它们能否在不损害响应语言的前提下提升准确率？针对第一个问题，我们使用现有的 Mean-SVD 构建方法以及我们提出的 DiffPCA 来构建子空间，两者都捕捉到了大量与翻译相关的变化，并在不同模型深度上呈现出相似的模式。针对第二个问题，我们将每个子空间投影出去，并训练线性探针从投影后的激活中对语言进行分类，结果显示探针仍能几乎完美地识别出语言，这表明两种子空间都无法穷尽地捕捉语言信息。针对第三个问题，我们在 Qwen2.5-7B 上评估了多种干预配置……（原文摘要在此处截断）

    arXiv:2611.05278v2 Announce Type: replace  Abstract: Language-subspace interventions aim to improve multilingual reasoning by suppressing language-related components in mid-layer activations. Using MATH and MGSM across four open-weight LLMs, we ask three questions: do estimated language subspaces capture translation-associated variation? Does projecting them out remove language information? And do they improve accuracy without compromising response language? For the first question, we build subspaces using the existing Mean-SVD construction and our DiffPCA. Both capture substantial translation-associated variation and show similar patterns across model depth. For the second, we project out each subspace and train linear probes to classify language from the projected activations. The probes still identify language almost perfectly, showing that neither subspace exhaustively captures language information. For the third, we evaluate several intervention configurations on Qwen2.5-7B, by va
    
[^79]: Maswao：一个自动收集、预处理和基准化的南非手语数据集

    Maswao: An automatically collected, prepared, and benchmarked dataset for the South African Sign language

    [https://arxiv.org/abs/2611.02903](https://arxiv.org/abs/2611.02903)

    本文提出了一个以最少人工干预自动收集、预处理和基准化南非手语（SASL）数据集的框架Maswao，利用新闻广播等公共资源和语音识别技术，旨在弥合全球手语处理技术发展的差距。

    

    手语处理领域的最新进展可归因于大规模数据集收集和算法复杂性两方面的发展，例如姿态估计技术、基于Transformer模型的信号处理以及模型优化。然而，在全球200多种公认的手语中，只有少数手语（如美国手语、英国手语和德国手语）从这些技术进步中受益，而其余手语则远远落后。在本研究中，我们旨在通过提出一个自动收集和预处理手语数据集的框架来解决这种技术差距，该框架仅需极少的人工干预，并以南非手语（SASL）作为测试平台。具体而言，我们利用新闻广播等公共资源自动收集SASL视频内容。此外，我们使用现有的自动语音识别（ASR）技术将伴随的音频流转换为文本。

    arXiv:2611.02903v1 Announce Type: new  Abstract: Recent advancements in sign language processing can be attributed to developments in both large-scale dataset collection and algorithmic complexity in areas such as pose estimation techniques, signal processing through Transformer models, and model optimization. However, only a handful of sign languages, such as the American, British, and German, from over 200 globally recognized sign languages, reap the benefits of such advancements, while the rest lag behind. In this study, we aim to address this technological disparity by proposing a framework for the automatic collection and preparation of sign language datasets, with very minimal human-labour intervention, using the South African Sign Language (SASL) as a testbed. Specifically, we automatically collect SASL video content using public resources such as news broadcasts. Additionally, we convert their accompanying audio streams into text using available Automatic Speech Recognition (AS
    
[^80]: FindLink：基于来源锚定前提学习评估科学关联

    FindLink: Learning to Assess Scientific Connections from Source-Grounded Premises

    [https://arxiv.org/abs/2611.02545](https://arxiv.org/abs/2611.02545)

    该论文提出 GI-TraceBench 基准与 FindLink 两阶段框架，通过构建来源锚定前提并进行结构化推理，结合确定性编译器反馈，来评估不同研究条件下科学发现之间的关联。

    

    arXiv:2611.02545v1 公告类型：新论文。摘要：基于文献的科学发现需要评估在不同研究条件下所得发现之间的关联。来源归因使单个发现可追溯，但并不能确定它们共同支持什么。我们提出 GI-TraceBench，一个包含 10,282 个问题的胃肠肿瘤学基准，通过共享的论文、主张和证据片段记录，统一了来源溯源问答、知识发现和假设评估三类任务。我们进一步提出 FindLink，一个面向来源锚定科学推理的两阶段框架。前提锚定阶段构建带有明确研究背景和支持状态的来源绑定前提；结构化推理阶段将这些前提组织成同时表示支持关系与未决条件的支持计划，并利用确定性编译器反馈生成以证据为条件的答案。该编译器验证结构与来源绑定的一致性，但……（原文在此处截断）

    arXiv:2611.02545v1 Announce Type: new  Abstract: Scientific discovery from the literature requires assessing connections among findings obtained under different study conditions. Source attribution makes individual findings traceable, but does not determine what they jointly support. We introduce GI-TraceBench, a benchmark of 10,282 questions in gastrointestinal oncology that unifies provenance question answering, knowledge discovery, and hypothesis evaluation through shared Paper, Claim, and Evidence Span records. We further propose FindLink, a two-stage framework for provenance-grounded scientific reasoning. Premise Grounding constructs source-bound premises with explicit study context and support status. Structured Reasoning organizes these premises into support plans that represent both supported relations and unresolved conditions, then generates evidence-conditioned answers using deterministic compiler feedback. The compiler verifies structural and source-binding consistency, but
    
[^81]: PixelBench：大语言模型能够逐像素地理解和生成图像吗？

    PixelBench: Can LLMs Understand and Generate Images Pixel by Pixel?

    [https://arxiv.org/abs/2611.02540](https://arxiv.org/abs/2611.02540)

    该论文提出PixelBench基准，证明大语言模型无需专门的图像分词器或解码器，仅凭现有词表即可直接生成可识别的像素图像，且像素理解能力与生成能力密切相关。

    

    大语言模型（LLM）可以通过SVG和CAD等基于文本的格式生成视觉素材，但这些符号化表示是使用高级图元来描述图像，而非显式的光栅数值。我们探讨LLM能否直接利用其现有词表生成像素图像，而无需经过学习的图像分词器或解码器。我们提出了PixelBench，这是一个涵盖多种分辨率的文本到像素（T2P）和图像到像素（I2P）生成基准，并附带一个互补的像素理解评估。模型首先生成一个RGB调色板，然后生成一个索引网格，用于指定每个输出像素对应的调色板颜色。在T2P任务中，最强的模型能够直接从文本生成可识别的图像。在I2P任务中，最强的模型在低分辨率下的表现超越了调色板匹配的下采样方法和专门的像素化算法。像素理解性能与生成性能之间存在很强的相关性。

    arXiv:2611.02540v1 Announce Type: cross  Abstract: Large language models (LLMs) can generate visual assets through text-based formats such as SVG and CAD, but these symbolic representations describe images using high-level primitives rather than explicit raster values. We ask whether LLMs can instead generate pixel images directly through their existing vocabulary, without a learned image tokenizer or decoder. We introduce PixelBench, a benchmark for text-to-pixel (T2P) and image-to-pixel (I2P) generation across multiple resolutions, together with a complementary pixel-understanding evaluation. Models generate an RGB color palette followed by a grid of indices specifying the palette color of each output pixel. In T2P, the strongest models produce recognizable images directly from text. In I2P, the strongest models outperform palette-matched downsampling and specialized pixelization algorithms at low resolutions. Pixel-understanding performance correlates strongly with generation perfor
    
[^82]: CuriousTogether：好奇心与文化背景在多智能体大语言模型交互中的影响

    CuriousTogether: Implications of Curiosity and Cultural Grounding in Multi-Agent LLM Interactions

    [https://arxiv.org/abs/2611.01301](https://arxiv.org/abs/2611.01301)

    提出CuriousTogether框架，证明好奇心驱动的多智能体LLM交互在八项任务中的七项上优于非好奇心设置，故事生成语义多样性提升最高达20.6%、刻板印象率降低14.9个百分点，且文化背景设定有利于创意和社会任务但可能损害精确推理。

    

    大语言模型（LLMs）越来越多地被部署在多智能体环境中，展现出涌现的集体行为，然而好奇心作为这种动态的驱动因素所起的作用仍未得到充分研究。我们提出了CuriousTogether，一个从三个互补维度评估好奇心驱动的多智能体大语言模型交互的框架：新颖性寻求、认知性好奇心和社会性好奇心。在涵盖推理、创意生成和社会偏见缓解的八项任务中，好奇心引导的多智能体设置在八项任务中的七项上优于非好奇心设置，其中故事生成（语义多样性）的增益高达20.6%，刻板印象率降低了14.9个百分点。此外，以国家为条件的文化背景设定对故事生成和社会类任务有益，但可能损害面向精确性的推理。最后，我们表明推理时基于好奇心的多智能体方法提供了有利的质量-标记（token）权衡。

    arXiv:2611.01301v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly deployed in multi-agent settings, exhibiting emergent collective behaviors, yet the role of curiosity as a driver of such dynamics remains understudied. We introduce CuriousTogether, a framework evaluating curiosity-driven multi-agent LLM interactions across three complementary dimensions: novelty-seeking, epistemic and social curiosity. Across eight tasks spanning reasoning, creative generation, and social bias mitigation, curiosity-guided multi-agent settings yield improvements over non-curious ones on seven out of eight tasks, with gains of up to 20.6% in story generation (semantic diversity) and a 14.9 pp reduction in stereotype rates. Furthermore, country-conditioned cultural grounding show benefits for story generation and social tasks, but can hurt precision-oriented reasoning. Finally, we show that inference-time multi-agent curiosity-based approach offers favorable quality-token trad
    
[^83]: 同一问题，不同答案：对六个大语言模型戒烟建议中因用户画像不同而产生的差异进行审计

    Same Question, Different Answers: Auditing Persona-Dependent Variation in Six LLMs on Smoking-Cessation Advice

    [https://arxiv.org/abs/2610.42039](https://arxiv.org/abs/2610.42039)

    本研究审计了六个主流大语言模型后发现，模型给出的戒烟建议因用户画像而异——对青少年画像提及FDA批准处方戒烟药物的频率比成人画像低32至67个百分点，而通过设定事实底线的系统提示词可作为可部署的干预措施。

    

    数亿人如今向大语言模型（LLM）聊天机器人寻求健康建议，这引出一个提出容易、回答却很难的问题：每个人得到的建议都一样吗？我们在戒烟领域对来自四家提供商的六个广泛部署的大语言模型进行了审计——该领域拥有明确的临床指南，且具有重要的公共卫生意义。我们向十二个在人生阶段、自述专业知识和吸烟史上各不相同的用户画像提出相同的四个戒烟问题，共获得2,880条回复。研究发现，模型给出的答案取决于提问者的身份：与成人画像相比，所有模型向青少年画像提及FDA批准的处方戒烟药物的频率都更低，差距达32至67个百分点；在六个模型中的五个里，自述具有医疗保健专业知识的画像获得的事实信息反而少于没有专业知识的画像。随后，我们评估了一种可部署的干预措施：一个指定事实底线的简短系统提示词。

    arXiv:2610.42039v1 Announce Type: new  Abstract: Hundreds of millions of people now ask large language model (LLM) chatbots for health advice, which raises a question that is simple to pose and hard to answer: does everyone get the same advice? We audit six widely deployed LLMs from four providers on smoking cessation, a domain with codified clini- cal guidance and substantial public-health stakes, by putting the same four cessation questions to twelve user personas that differ in life stage, self-described expertise, and smoking history, for a total of 2,880 responses. What the models say depends on who appears to be asking. Every model named FDA-approved prescription cessation medications less often to teen personas than to adult personas, by 32 to 67 percentage points, and in five of six models a persona presenting health- care expertise received fewer facts than a persona presenting none. We then evaluate one deployable intervention: a short system prompt specifying a factual floor
    
[^84]: TraceSpec：理解并改进面向大语言模型智能体的投机解码

    TraceSpec: Understanding and Improving Speculative Decoding for LLM Agents

    [https://arxiv.org/abs/2610.41177](https://arxiv.org/abs/2610.41177)

    提出TraceSpec框架，通过结合动态执行与固定前缀重放、构建跨浏览与软件工程任务的轨迹状态语料库，揭示了汇总投机解码指标掩盖的分阶段性能差异，并实现基于轨迹条件的自适应草稿监督与预测预算分配。

    

    大语言模型智能体通过在不断演化的推理、动作与工具反馈历史之上反复调用模型来执行任务。汇总式的投机解码指标将这些不同的决策合并在一起，对于在何处需要草稿监督所提供的指导十分有限。我们提出了TraceSpec，这是一个将轨迹评估与自适应草稿训练相连接的框架。它将动态执行与固定前缀重放相结合，并引入了TraceSpec-Traj——一个包含10,000个训练状态和2,500个测试状态的语料库，涵盖浏览与软件工程两类任务。对五个投机解码基线的分析表明，汇总接受率可能随上下文长度增加而上升，却在匹配的后期浏览动作中出现下降。工作负载与阶段进一步区分了这些趋势，这启发了由轨迹条件引导的监督方式。TraceSpec在中间的助手轮次上进行训练，并利用基于状态的、前缀感知的接受率估计来分配固定的预测预算。

    arXiv:2610.41177v1 Announce Type: new  Abstract: LLM agents execute tasks through repeated model calls over evolving histories of reasoning, actions, and tool feedback. Aggregate speculative-decoding metrics merge these distinct decisions, providing limited guidance on where draft supervision is needed. We present TraceSpec, a framework connecting trajectory evaluation with adaptive draft training. It combines dynamic execution with fixed-prefix replay and introduces TraceSpec-Traj, a corpus of 10,000 training states and 2,500 test states spanning browsing and software engineering. Analysis of five speculative baselines reveals that aggregate acceptance can rise with context length while declining within matched late browsing actions. Workload and phase further distinguish these trends, motivating supervision guided by trajectory conditions. TraceSpec trains on intermediate assistant turns and allocates a fixed prediction budget using state-conditioned estimates of prefix-aware accepta
    
[^85]: 大语言模型作为裁判与考生用于医学试题难度预测

    LLMs as Judges and Examinees for Medical Item Difficulty Prediction

    [https://arxiv.org/abs/2610.38402](https://arxiv.org/abs/2610.38402)

    本研究创新性地让开源大语言模型分别扮演“裁判”和“模拟考生”两种角色，提取难度评判信号与作答行为信号，并与题目基线特征结合，从而显著提升了医学试题难度的预测效果。

    

    仅凭题干文本预测医学试题难度仍然具有挑战性。以往的工作主要依赖于语言学特征或源自通用型及领域专用语言模型的文本表示，但预测性能一直较为有限。本研究采取了一种不同的方法，即利用开源权重的大语言模型（LLMs）与试题交互时涌现出的难度相关信号。基于USMLE和PANCE医学执照考试题目，我们考察了大语言模型在两种角色下的表现。作为“裁判”时，模型提供绝对难度评分和两两之间的难度比较；作为“模拟考生”时，它们生成基于作答的信号，包括作答正确性、正确选项概率、作答熵以及前两个选项之间的概率差。随后，我们将最强的评判信号和作答信号与基线题目特征相结合，用于下游预测模型。两两难度判断……（摘要原文在此处截断）

    arXiv:2610.38402v1 Announce Type: cross  Abstract: Predicting medical item difficulty from item text alone remains challenging. Prior work has largely relied on linguistic features or text representations derived from general-purpose and domain-specific language models, but predictive performance has remained modest. This study takes a different approach by using difficulty-related signals that emerge when open-weight large language models (LLMs) interact with test items. Using USMLE and PANCE medical licensure examination items, we examined LLMs in two roles. As judges, the models provided absolute difficulty ratings and pairwise difficulty comparisons. As simulated examinees, they generated response-based signals including correctness, keyed-option probability, response entropy, and the probability margin between the top two options. We then combined the strongest judgment and response signals with baseline item features in downstream prediction models. Pairwise difficulty judgment w
    
[^86]: LLM-as-a-Judge中的提示彩票现象：测量与缓解对句法改写的敏感性

    Prompt Lottery in LLM-as-a-Judge: Measuring and Mitigating Sensitivity to Syntactic Rephrasing

    [https://arxiv.org/abs/2610.38076](https://arxiv.org/abs/2610.38076)

    论文发现了“提示彩票”现象——语义等价的提示词改写会导致LLM-as-a-Judge产生不一致的评判结果，甚至可能逆转模型的相对排名，并提出了测量与缓解这种句法改写敏感性的方法。

    

    LLM-as-a-Judge已成为评估大语言模型的标准方法，然而大多数方法都隐含地假设单个评判提示词是一个稳定的评估工具。以往的工作主要探索有意的提示词修改，例如评分标准、角色描述或思维链推理，以优化评估准确性。与此相反，我们研究的是那些经过验证、旨在保持评估语义不变但表面表述存在差异的提示词变体（即释义性改写），是否在固定评估任务下仍能产生一致的判断。我们提出了“提示彩票”这一现象，即释义性改写会导致LLM-as-a-Judge产生不同的评判结果。在一项涵盖多样化评估任务、评判模型家族和底层模型的大规模研究中，我们发现这些表面变化能够改变评估结果，在某些设置下甚至会颠倒模型之间的相对排名。一项针对10名用户的…（摘要截断）

    arXiv:2610.38076v1 Announce Type: cross  Abstract: LLM-as-a-Judge has become a standard for evaluating large language models, yet most approaches implicitly assume that a single judge prompt functions as a stable evaluation instrument. Prior work has primarily explored deliberate prompt modifications such as scoring rubrics, role descriptions, or chain-of-thought reasoning in order to optimize accuracy. In contrast, we investigate whether validated prompt variants intended to preserve evaluation semantics while differing in surface realization (\textit{paraphrastic rephrasings}) yield consistent judgments under a fixed evaluation task. We introduce \textit{Prompt Lottery}, a phenomenon in which paraphrastic rephrasings produce different LLMaJ judgments. Across a large-scale study spanning diverse evaluation tasks, judge families, and model backbones, we find that these variations can alter evaluation outcomes and, in some settings, reverse relative model rankings. A user study with 10 
    
[^87]: 积分奖励而非采样奖励：面向全双工语音模型中打断与附和的窗口似然方法

    Integrate the Reward, Don't Sample It: Window Likelihood for Interruptions and Backchannels in Full-Duplex Speech Models

    [https://arxiv.org/abs/2610.35581](https://arxiv.org/abs/2610.35581)

    该论文发现基于整轮展开的GRPO时机奖励在训练全双工语音模型时存在零梯度陷阱与信用泄漏两大缺陷，并提出“窗口似然”方法，通过对停止决策进行闭式积分直接最大化正确让出/保持的对数概率，在HumDial-FDBench上取得了最佳的让出/保持F1表现。

    

    全双工语音对话模型必须在用户插话（打断）时让出话轮，而在用户仅作附和时保持说话。带有时机奖励的序列级强化学习（GRPO）是常见的做法；我们在四个开源模型上测试了其整轮展开的形式，发现了两类失效：二值窗口奖励恰恰在最需要学习的窗口上给出零优势（零梯度陷阱），而每次展开仅有的单一评分被摊派到奖励并未以其定义的收听状态决策上（信用泄漏），导致模型对着沉默继续说话或停止回答。这两种失效都源于对一个本可由少数二值控制决策的已知函数表达的奖励进行采样。我们提出窗口似然（WL）：在每个窗口采样一条续写，但对其经过的停止决策以闭式形式积分，并最大化正确行动的对数概率——让出对应噪声-OR形式，保持对应生存乘积形式。在 HumDial-FDBench 上，WL 取得了最佳的让出/保持 F1 表现。

    arXiv:2610.35581v1 Announce Type: cross  Abstract: A full-duplex spoken dialogue model must yield when the user cuts in and hold when the user merely backchannels. Sequence-level RL with a timing reward (GRPO) is the usual recipe; we test its whole-rollout form on four open models and find two failures. A binary window reward gives zero advantage on exactly the windows that most need learning (a zero-gradient trap), and the one score per rollout is spread over listening-state decisions the reward is not written in terms of (credit leakage), so the model talks into silence or stops answering. Both come from sampling a reward that is a known function of a few binary control decisions. We propose \emph{window likelihood} (\WL): sample a continuation per window, but integrate in closed form over the stop decisions it visits and maximise the log-probability of acting correctly---a noisy-OR for yielding, a survival product for holding. On HumDial-FDBench, \WL\ gives the best yield/hold F1 on
    
[^88]: 什么让问题充满好奇？测量人机交互中的模型好奇心

    What Makes a Question Curious? Measuring Model Curiosity in Human-LLM Interactions

    [https://arxiv.org/abs/2610.33950](https://arxiv.org/abs/2610.33950)

    该研究提出“模型好奇心”的概念及分类体系，发现好奇心导向的问题虽仅占17-28%且较为罕见，但能促使用户做出更具探索性和认知丰富性的回应，尤其在建议、创意和研究场景中能激发探索。

    

    大型语言模型在日常交流中日益扮演着合作伙伴的角色。在这些互动中，有效的协助可能不仅需要回答问题，还需要提出恰当的问题，以揭示知识空白、假设和替代方案。在本研究中，我们将这类基于认知的问题提出行为称为“模型好奇心”，并引入了一个基于理论的分类体系，用以区分好奇心导向的问题与一般性的模型提问。通过对四个对话数据集的分析，我们发现好奇心导向的问题仍然较为罕见（占所有问题的17-28%），但其与用户后续更具探索性、认知上更丰富的回应相关联，且在不同数据集中呈现出不同的模式。我们进一步开展了一项人类研究来评估模型好奇心的有用性，发现好奇心能够鼓励探索，尤其是在建议、创意和研究场景中。这些发现对于构建更善于选择性提问的大型语言模型具有重要意义。

    arXiv:2610.33950v1 Announce Type: cross  Abstract: Large language models increasingly act as partners in everyday communication. In these interactions, useful assistance may require not only answering, but also asking the right questions that surface knowledge gaps, assumptions, and alternatives. In this study, we refer to such epistemic question-asking as model curiosity, and introduce a theory-grounded taxonomy that distinguishes curiosity-oriented questions from generic model questioning. Across four conversation datasets, we find that curiosity-oriented questions remain rare (17-28% of all questions), but are associated with more exploratory and epistemically rich subsequent user responses, with varying patterns across datasets. We further conduct a human study to evaluate the usefulness of model curiosity, finding that curiosity can encourage exploration, especially in advice, creative, and research contexts. These findings have implications for building LLMs that ask more selecti
    
[^89]: 邻居的低语：重新思考描述共识作为视觉-语言检索的推理时证据

    Whispers from the Neighbors: Rethinking Caption Consensus as Inference-Time Evidence for Vision-Language Retrieval

    [https://arxiv.org/abs/2610.33798](https://arxiv.org/abs/2610.33798)

    该论文提出在推理时将同一图像的相邻描述共识作为证据融合进冻结双编码器的视觉-语言检索中，在同源标注描述条件下可将文本到图像检索R@1提升3.4至16.8个百分点。

    

    一幅图像很少只被描述一次。检索基准和许多数据集为每张图片附加了多个描述，然而双编码器仍然孤立地判断每一个图像-描述对。我们探究当冻结的编码器在推理时参考这些相邻描述时，它们能带来什么贡献。将图像相似度与图像自身描述的相似度进行融合，当描述来自与查询相同的标注者时，可在九个冻结编码器上将文本到图像检索的R@1提升3.4至16.8个百分点。当描述由现成的描述生成器生成时，增益降至1.9至7.2个百分点；而当描述来自不同的标注来源时，增益最多只有一个百分点；使用释义训练的文本编码器可进一步增强文本侧的性能。一种自然的改进方法——对图像中描述未能解释的部分进行评分——在留一法评估下似乎能带来更大的提升。我们证明这一优势之所以产生，是因为正确候选者的…

    arXiv:2610.33798v1 Announce Type: cross  Abstract: An image is rarely described only once. Retrieval benchmarks and many collections attach several captions to each picture, yet dual encoders still judge every image-caption pair in isolation. We ask what these neighboring captions contribute when a frozen encoder consults them at inference time. Fusing image similarity with similarity to an image's own captions improves text-to-image R@1 by 3.4 to 16.8 points across nine frozen encoders when the captions come from the same annotators as the queries. The gain falls to 1.9 to 7.2 points when an off-the-shelf captioner writes them, and to at most one point when they come from a different annotation source; a paraphrase-trained text encoder strengthens the text side further. A natural refinement, scoring the part of the image that the captions leave unexplained, appears to add much more under a leave-one-out evaluation. We show that this advantage arises because the correct candidate's sup
    
[^90]: 先测量再训练：类型化决策模型 vs. 经典基线 vs. 生成式大语言模型在个人多语言收件箱路由中的应用

    Measure Before You Train: Typed Decision Models vs. a Classical Baseline vs. a Generative LLM for Routing a Personal Multilingual Inbox

    [https://arxiv.org/abs/2610.32955](https://arxiv.org/abs/2610.32955)

    该论文提出自托管个人邮件代理Foyer及其测量协议，在真实多语言邮件数据上于训练前建立人工真实标注基准，并系统比较了类型化决策模型、经典机器学习基线与生成式大语言模型等九个系统的表现。

    

    个人邮件代理需要为每封邮件做出决策：这封邮件属于什么类型、是否紧急、是否需要起草回复、创建任务，还是直接忽略。然而这些系统的错误率很少被真正测量，因为存储的标签是由系统自身生成的。我们提出了Foyer——一个正在日常使用的自托管个人收件箱代理，以及一套测量协议（分层采样、冻结快照、按邮件线程分组的划分、预注册假设、在代理审查循环中由人工标注），该协议能够在任何训练开始之前获得真实标注数据。在739封由所有者亲自标注的真实意大利语/英语邮件上（其中231封为无偏样本，500封为对抗性采样），我们在按线程分组的交叉验证下比较了九个系统：生产环境的关键词规则、TF-IDF+逻辑回归经典基线、本地类型化决策模型、托管决策模型，以及托管生成式大语言模型。论文给出了四项主要结果，其中生产环境规则在邮件类型识别上仅获得28%的得分，而对照系统达到了58%。

    arXiv:2610.32955v1 Announce Type: new  Abstract: Personal email agents decide, for every message, what it is, whether it is urgent, whether to draft a reply, create a task, or ignore it. Their error rate is rarely measured: the stored labels were written by the system itself. We present Foyer, a self-hosted personal inbox agent in daily use, and a measurement protocol (stratified sampling, frozen snapshots, thread-grouped splits, pre-registered hypotheses, human labels in the agent's review loop) that yields ground truth before any training. On 739 real Italian/English emails labelled by the owner (231 unbiased, 500 adversarially sampled), we compare nine systems under thread-grouped cross-validation: the production keyword rules, a TF-IDF + logistic-regression baseline, local typed decision models (Tev1, Nimble), a hosted decision model (Nimble v4-max), and hosted generative LLMs (Qwen2.5-7B, Gemma 4 26B/31B). Four results. The production rules score 28% on message type against a 58% 
    
[^91]: 大型语言模型中的认知地图

    Cognitive Maps in Large Language Models

    [https://arxiv.org/abs/2610.25596](https://arxiv.org/abs/2610.25596)

    超过百亿参数的大型语言模型会形成类似动物认知地图的抽象关系表征，关系以成对链接形式回溯式组合存储，扰动其中稀疏的键值对还能植入“虚假记忆”并引发可预测的行为变化。

    

    长期以来，有观点认为动物会构建认知地图以在新环境中支持灵活的行为。我们通过关系推理任务和因果“马赛克”实验，检验了大型语言模型是否形成类似的内部表征。参数规模超过百亿的模型能够推断出未明确陈述的关系，并将其泛化到合成的实体名称上。在跨模型、跨任务的实验中，模型形成了对关系结构的抽象表征，并将其与实体身份的表征以相加的方式结合。关系主要以成对链接的形式存储并进行回溯式组合，而非前瞻性地编码在一个全局坐标系中。一组稀疏且分布式的键值对中介了这种绑定机制；扰动这些键值对可以植入“虚假记忆”，从而产生可预测的行为改变。我们的发现揭示了一种类似于认知地图的保守组织结构，它支持灵活而系统的行为。

    arXiv:2610.25596v2 Announce Type: replace  Abstract: It has long been proposed that animals construct cognitive maps to support flexible behavior in novel environments. We tested whether large language models form analogous internal representations using relational reasoning tasks and causal "mosaic" experiments. Models beyond the ten-billion-parameter scale inferred unstated relationships and generalized to synthetic entity names. Across models and tasks, models formed abstract representations of relational structure, additively combining them with representations of entity identity. Relationships were primarily stored as pairwise links and composed retrospectively, rather than prospectively encoded in a global coordinate system. A sparse, distributed set of key-value pairs mediated binding; perturbing them implanted "false memories" that produced predictable behavioral changes. Our findings reveal a conserved organization analogous to a cognitive map supporting flexible and systemati
    
[^92]: Chinese Decision（中文决策）：面向单次前向概率决策的语言模型系列

    Chinese Decision: A Family of Language Models for Single-Pass Probabilistic Decision-Making

    [https://arxiv.org/abs/2610.23170](https://arxiv.org/abs/2610.23170)

    Chinese Decision 是一系列中文语言模型，通过单次前向传播直接对候选选项的 logits 进行归一化得到概率分布，无需自回归生成，在医学、法律、金融等领域的决策任务上实现了高准确率与良好校准。

    

    许多基于语言的应用需要对有限决策集合输出概率分布，而非开放式回答。我们提出了 Chinese Decision（中文决策），一个面向单次前向概率决策的中文语言模型系列。给定上下文、指令和候选描述，模型在单次前向传播中提取末尾位置的候选字母 logits，并在有效选项上进行归一化，无需自回归式答案生成。统一的共享接口支持布尔判断、类别选择和序数评分。我们报告了一个已完成的 0.8B 模型，该模型使用分布交叉熵在 1304.8 万个决策上训练；此外还报告了经过额外 RLCD 目标训练的初步 2B 检查点。在包含 307,900 个决策的完整 CJ-Bench 上，未经校准的 0.8B 模型仅使用单一检查点，在通用领域达到 72.24% 的准确率，在医学、法律和金融领域平均达到 76.98% 的准确率，整体期望校准误差为 1.07%。

    arXiv:2610.23170v1 Announce Type: new  Abstract: Many language-based applications require probabilities over a finite set of decisions rather than open-ended responses. We present Chinese Decision, a family of Chinese-language models for single-pass probabilistic decision-making. Given context, an instruction, and candidate descriptions, the model extracts final-position candidate-letter logits in one forward pass and normalizes them over valid options, without autoregressive answer generation. A shared interface supports Boolean judgments, categorical choices, and ordinal scores. We report a completed 0.8B model trained with distribution cross-entropy on 13.048 million decisions and preliminary 2B checkpoints trained with an additional RLCD objective. On the full CJ-Bench of 307,900 decisions, the uncalibrated 0.8B model achieves 72.24% general-domain accuracy and 76.98% average accuracy across medicine, law, and finance using a single checkpoint, with 1.07% overall expected calibrati
    
[^93]: 分别探索，择优采纳：面向大型工具库的执行引导式工具使用

    Explore Separately, Adopt Selectively: Execution-Guided Tool Use over Large Tool Libraries

    [https://arxiv.org/abs/2610.22688](https://arxiv.org/abs/2610.22688)

    提出BRIA智能体，通过将候选工具调查隔离到独立分支、检查执行证据并仅将有用的结果采纳回主轨迹，解决了大规模工具库中多候选工具选择时失败尝试污染推理轨迹的问题。

    

    大型语言模型智能体在任务执行过程中出现新需求时，必须从包含数千个工具的大型工具库中发现相关工具。当多个候选工具都可能满足同一需求时，可能需要通过实际执行来识别真正适用的工具。在共享轨迹中调查候选工具可能导致部分备选方案未被探索，或使失败的尝试影响后续推理。为解决这一问题，我们提出了BRIA（BRanch-Inspect-Adopt，分支-检查-采纳），一个简单而有效的智能体，它将候选工具的调查暂时隔离在独立的分支中。每个分支检索并执行工具以调查其对应的候选工具；任务智能体检查由此产生的执行证据，并只将有用的结果采纳到其继续执行的主轨迹中。BRIA还能在任务执行过程中出现新的能力需求时动态检索更多工具。我们在PlanBench-XL和MCP-Atlas基准上，使用包含1897个工具的共享工具池、三个语言模型以及静态分析设置对BRIA进行了评估。

    arXiv:2610.22688v1 Announce Type: cross  Abstract: Large language model agents must discover relevant tools from large libraries, e.g., thousands of tools, as new needs arise during task execution. When several candidates could satisfy the same need, execution may be required to identify the relevant one. Investigating candidates in a shared trajectory can leave alternatives unexplored or allow failed attempts to influence subsequent reasoning. To address this, we present BRIA (BRanch-Inspect-Adopt), a simple yet effective agent that temporarily isolates candidate investigations in separate branches. Each branch retrieves and executes tools to investigate its candidate; a task agent inspects the resulting execution evidence and adopts only useful results into its continuing trajectory. BRIA retrieves additional tools as new capability needs arise during task execution. We evaluate BRIA on PlanBench-XL and MCP-Atlas using a shared pool of 1897 tools, three language models, and static an
    
[^94]: MetaEncoder：探索双编码器在基于自然语言接口的多模态“系统一”决策中的极限

    MetaEncoder: Exploring the Limit of Bi-Encoders for Multimodal System One Decision Making with Natural Language Interface

    [https://arxiv.org/abs/2610.11316](https://arxiv.org/abs/2610.11316)

    提出MetaEncoder，将预训练的30B解码器微调为基于单向对比学习的双编码器架构，实现了完全通过自然语言接口、可从小闭集扩展到数百万开放集候选空间的多模态系统一决策。

    

    系统（System One）模型输出的是受约束的决策和概率分布，而非自由形式的文本生成。当前主流范式依赖结构化的模式对象来编码状态、意图和候选选择，而我们重新审视一种完全基于自然语言的系统一接口。在该框架中，用户请求和每个候选选项均以自然语言表达，并由多模态（图像和视频）辅助输入提供支持。我们提出了MetaEncoder，它将预训练的Muse-Glimmer 30B解码器微调为一个遵循指令的决策编码器。为了在小规模闭集（<256）和大规模开放集（数百万）候选空间中都能有效扩展，MetaEncoder采用双编码器架构，并通过单向对比学习进行请求-候选对齐训练。我们在涵盖多模态决策、理解（……）等任务的11个基准测试套件和190个任务上进行了广泛评估。

    arXiv:2610.11316v1 Announce Type: cross  Abstract: System One models output constrained decisions and probability distributions rather than free-form text generation. While prevailing paradigms rely on structured schema objects to encode state, intent, and candidate choices, we revisit a fully natural language-based System One interface. In this framework, both the user request and each candidate option are expressed in natural language, supported by multimodal (image and video) auxiliary inputs. We introduce MetaEncoder, which fine-tunes a pre-trained Muse-Glimmer 30B decoder into an instruction-following decision-making encoder. To scale effectively across both small closed-set (< 256) and massive open-set (millions) candidate spaces, MetaEncoder employs a bi-encoder architecture trained via unidirectional contrastive learning for request-candidate alignment. We conduct extensive evaluations across 11 benchmark suites and 190 tasks spanning multimodal decision-making, understanding (
    
[^95]: 从宏观社会信号中学习模拟个体

    Learning to Simulate Individuals from Macro Social Signals

    [https://arxiv.org/abs/2610.07062](https://arxiv.org/abs/2610.07062)

    该论文提出macro2mind框架，将预测市场价格轨迹作为宏观监督信号，通过GRPO训练和社会行为分解，使大语言模型把行为推理作为显式预测步骤，从而从宏观数据中学会模拟个体对真实事件的反应。

    

    大语言模型越来越多地被用于模拟个体如何应对新情境，然而这些回应背后的行为推理要么继承自预训练，要么从个体级标注中学习，而个体级标注能提供的行为多样性有限，且几乎无法对推理过程本身进行监督。我们提出从预测市场中学习行为推理，预测市场的价格轨迹大规模地记录了人群对真实世界事件的反应。我们介绍了macro2mind，该方法利用市场信号通过GRPO训练语言模型。一种社会行为分解方法使行为推理成为预测的显式步骤：模型推断出具有代表性的市场参与者群体，预测每个群体如何解读新闻并更新其信念，推理它们之间的相互作用，并将这些反应聚合为价格。带有难度感知采样的后见之明遗憾课程将训练集中于……（原文在此处截断）

    arXiv:2610.07062v1 Announce Type: cross  Abstract: Large language models are increasingly used to simulate how individuals respond to new situations, yet the behavioral reasoning behind these responses is either inherited from pretraining or learned from individual-level annotations, which offer limited behavioral diversity and little supervision of the reasoning itself. We propose to learn behavioral reasoning from prediction markets, whose price trajectories record how populations respond to real-world events at scale. We introduce macro2mind, which trains a language model with GRPO using market signals. A social behavioral decomposition makes behavioral reasoning an explicit step of forecasting: the model infers representative groups of market participants, predicts how each interprets the news and updates its beliefs, reasons about their interactions, and aggregates these responses into a price. A hindsight-regret curriculum with difficulty-aware sampling focuses training on transi
    
[^96]: 大语言模型谄媚行为剖析：翻转率掩盖了什么

    Anatomy of LLM Sycophancy: What a Flip Rate Hides

    [https://arxiv.org/abs/2610.06522](https://arxiv.org/abs/2610.06522)

    本文提出 SycoLens 模块化多因素受控重放协议，对 11 个前沿模型进行约 76 万次重放，首次系统区分并量化了用户反驳下“自我纠正”与“谄媚妥协”这两种被单一翻转率掩盖的行为，并发现断言式与非断言式反驳对模型排名几乎无关、翻转效应在能力边界附近增长数倍、模型高确定题目仍贡献约一半的最严重翻转。

    

    模型在用户反驳下可能自我纠正、妥协让步或坚持立场，而单一的翻转率指标会将“自我纠正”与“谄媚妥协”同等计数。SycoLens 是一种模块化重放协议，通过重放一道题目、一个已提交的答案和一条预先编写的用户台词来测量翻转率，且每次重放都与删除该台词的匹配对照组进行比较。反驳措辞、已提交文本、答案格式、边界距离和真实答案成为同一测量工具中的可控因素（以往工具仅能变化一到三个因素），从而使各因素之间的交互作用变得可测量。在十一个前沿模型和约 76 万次受控重放中，断言相反结论的台词与不作断言、仅提出质疑的台词对模型的排名几乎互不相关。翻转效应在接近模型能力边界时会增长数倍，但模型非常有把握的题目仍占最受影响题目总量的约一半；预先植入的文本对已提交答案在静态下的锚定作用大于其保护作用。（原文摘要在此处截断）

    arXiv:2610.06522v2 Announce Type: replace  Abstract: A model under pushback can correct itself, capitulate, or hold, and one flip rate counts a correction and a capitulation alike. SycoLens, a modular replay protocol, measures flip rates from replays of an item, a committed answer, and one scripted user line, each against a matched control with the line deleted. Pushback wording, committed text, answer format, boundary distance, and ground truth become factors of one instrument, where earlier instruments vary one to three, so the interactions between the factors become measurable. Across eleven frontier models and about 760,000 controlled replays, lines that assert the opposite verdict and lines that challenge without asserting one rank the models almost unrelatedly. Flip effects grow several-fold near a model's boundary, yet items the model is certain of still carry about half of the most-affected totals; planted prose anchors a commitment at rest more than it protects it. Under a yes
    
[^97]: 面向数学研究智能体的持续图记忆系统

    Continual Graph Memory for Mathematical Research Agents

    [https://arxiv.org/abs/2610.02945](https://arxiv.org/abs/2610.02945)

    提出 Ansatz——一个基于“持续图记忆”的数学研究智能体，通过可演化、跨问题的图结构记忆系统显式组织整个证明搜索过程，并复用先前问题的探索知识，从而有效管理海量中间证明结果。

    

    利用前沿智能体框架来解决数学研究问题，已成为推动数学发展的有效手段。然而，解决数学领域的前沿问题可能需要大量智能体长时间并行工作以构建证明，从而产生海量的中间证明结果。在长周期的证明搜索过程中组织这些中间结果，并复用先前探索所获得的知识，仍然是重大的挑战。我们提出了 Ansatz——一个围绕“持续图记忆”构建的数学研究智能体。持续图记忆是一种基于图、可演化、跨问题的数学研究记忆系统，它显式地组织整个证明搜索过程，并复用来自先前问题探索轨迹的信息。具体而言，我们开发了一个统一的图记忆，用于表示所有中间探索结果，包括事实、计划和反例……

    arXiv:2610.02945v1 Announce Type: new  Abstract: Using frontier agent harnesses to tackle mathematical research problems has emerged as an effective means of advancing mathematics. However, solving frontier problems in mathematics may require a massive number of agents working in parallel for extended periods to construct proofs, thereby generating an enormous volume of intermediate proof results. Organizing these intermediate results throughout a long-horizon proof-search process and reusing knowledge gained from prior explorations remain major challenges. We present Ansatz, a mathematical research agent built around Continual Graph Memory, a graph-based, evolvable, cross-problem mathematical research memory system that explicitly organizes the entire proof search process and reuses information from exploration trajectories of previous problems. Specifically, we develop a unified graph memory that represents all intermediate exploration results, including facts, plans, and counterexam
    
[^98]: 采样揭示风格：大语言模型激活中提示条件风格轴的无监督、免训练发现

    Sampling Reveals Style: Unsupervised, Training-Free Discovery of Prompt-Conditional Stylistic Axes in LLM Activations

    [https://arxiv.org/abs/2609.19150](https://arxiv.org/abs/2609.19150)

    该论文提出一种无需训练的无监督方法，通过对同一提示的高温重复采样补全进行主成分分析，自动发现并标记大语言模型激活中与提示相关的风格轴，并通过245个人类风格标注验证了其与人类自发风格需求的高度契合。

    

    大语言模型（LLM）在其隐藏激活中编码了丰富的风格结构，但要发现对于给定提示哪些风格维度是显著的，通常需要监督式对比数据。我们提出了一种无需训练、提示条件化的替代方法：我们对单个提示在较高温度下反复采样补全结果，对汇集的隐藏激活应用主成分分析（PCA），并根据极性生成结果自动标记所得的风格轴。我们在一项两阶段研究中，将发现的轴与245个人类风格标注进行了验证。在我们最强的模型（Qwen-3.5-4B-Instruct）上，前两个轴以72.8%的精确率和43.6%的宏召回率匹配用户自发请求的风格维度，75.6%的有效性评分认为这些轴的极性生成结果与其标签相符，标注者之间的相邻一致性达90.9%。风格轴的可发现性强烈依赖于模型本身：两个Qwen模型（原文摘要在此处截断）

    arXiv:2609.19150v1 Announce Type: new  Abstract: Large language models (LLMs) encode rich stylistic structure in their hidden activations, but discovering which stylistic dimensions are salient for a given prompt typically requires supervised contrastive data. We present a training-free, prompt-conditional alternative: we repeatedly sample completions of a single prompt at elevated temperature, apply Principal Component Analysis (PCA) to the pooled hidden activations, and label the resulting axes automatically from the pole generations. We validate the discovered axes against 245 human-elicited stylistic annotations in a two-phase study. On our strongest model (Qwen-3.5-4B-Instruct), the top two axes match spontaneously requested human dimensions with 72.8% precision and 43.6% macro-recall, and 75.6% of validity ratings judge the axes' polar generations accurate to their labels, with 90.9% adjacent inter-annotator agreement. Discoverability is strongly model-dependent: both Qwen models
    
[^99]: BiG-SURE——用于大语言模型语义不确定性与可靠性估计的二部图方法

    BiG-SURE - Bipartite Graph for Semantic Uncertainty and Reliability Estimation of LLMs

    [https://arxiv.org/abs/2608.30646](https://arxiv.org/abs/2608.30646)

    提出了一种基于跨温度语义一致性的黑盒不确定性估计方法BiG-SURE，通过构建低温锚点与高温探针之间的二部图并用谱能量衡量语义一致性，从而评估大语言模型输出的可靠性。

    

    可靠的不确定性估计是在安全关键场景中部署大语言模型（LLM）和视觉-语言模型（VLM）的关键前提，尤其是在模型参数不可访问（黑盒）的情况下。我们提出了BiG-SURE，一种基于跨温度语义一致性的不确定性估计器。该方法在保持语义不变的输入变换下，将低温采样得到的响应作为稳定的语义锚点，将高温采样得到的响应作为探针。随后，方法利用基于自然语言推理（NLI）的蕴含分数构建锚点-探针二部图，并通过该矩阵的归一化平方谱能量来定义置信度，不确定性则由其补集给出。这种基于二部图的语义不确定性与可靠性估计（SURE）分数，用于衡量高温探针是否与模型稳定的低温信念保持语义一致。我们在文本问答等任务上对BiG-SURE进行了评估。

    arXiv:2608.30646v1 Announce Type: cross  Abstract: Reliable uncertainty estimation is a crucial requirement for deploying large language models (LLMs) and vision-language models (VLMs) in safety-critical settings, especially when the model parameters are not accessible (black-box). We propose BiG-SURE, an uncertainty estimator based on cross-temperature semantic agreement. The method samples low-temperature responses as stable semantic anchors and high-temperature responses as probes under meaning-preserving input transformations. It then constructs an anchor-probe Bipartite Graph (BiG) using NLI-based entailment scores and defines confidence through the normalized squared spectral energy of this matrix, with uncertainty given by its complement. This bipartite graph-based Semantic Uncertainty and Reliability Estimation (SURE) score measures whether high-temperature probes remain semantically aligned with the model's stable low-temperature belief or not. We evaluate BiG-SURE on text QA,
    
[^100]: 基准污染：以被规避的缓解措施为组织原则的分类体系

    Benchmark Contamination: A Taxonomy Organized by Defeated Mitigation

    [https://arxiv.org/abs/2608.29463](https://arxiv.org/abs/2608.29463)

    本文提出了一种按各类污染所规避的缓解措施（直接型、衍生型、时间型、分布型、获得型）组织的基准污染新分类体系，并将其操作化为一个四字段披露协议，强调评测过程中获得的污染状态必须随报告的分数一同记录。

    

    基准测试分数是模型、评测框架、引出预算、抽样总体以及污染状态的共同属性。排行榜只公布模型和分数，因此能力与数据泄漏在观测上无法区分。现有分类法是为自动化检测而对污染进行分类，而非回答报告者在发布结果时所面临的问题：在已应用的缓解措施之下，还有哪些效度威胁依然存在？我们提出了一种按各类污染所规避的缓解措施来组织的分类法——直接型、衍生型、时间型、分布型与获得型——涵盖训练时与评测时的泄漏。留出一个私有测试集只能单独关闭第一类。第五类是在评测过程本身中获得的；由于它是单次运行的属性，因此必须随报告的分数一起记录，而不是随基准的发布一起记录。我们将其操作化为一个四字段披露协议，其中“未知”是一种有效的（取值）……

    arXiv:2608.29463v1 Announce Type: cross  Abstract: A benchmark score is a joint property of the model, the evaluation harness, the elicitation budget, the sampled population, and contamination status. Leaderboards publish the model and the score, so capability and leakage stay observationally equivalent. Existing taxonomies classify contamination for automated detection, not the question a reporter faces at publication: given the mitigations already applied, which validity threats remain open? We introduce a taxonomy organized by the mitigation each type defeats -- direct, derivative, temporal, distributional, and acquired -- spanning training-time and evaluation-time leakage. Holding out a private test set closes the first alone. The fifth is acquired during the evaluation itself; because it is a property of one run, it must be recorded with the reported score rather than with the benchmark release. We operationalize it as a four-field disclosure protocol in which "unknown" is a valid
    
[^101]: 小学习者：在教学控制的知识暴露下的语言模型

    LittleLearner: Language Models Under Pedagogically Controlled Knowledge Exposure

    [https://arxiv.org/abs/2608.13545](https://arxiv.org/abs/2608.13545)

    本文提出了一个受教学控制的预训练语料库和模型，通过限制知识暴露范围，为研究语言模型的知识获取和能力边界提供了可解释的沙盒环境。

    

    摘要：arXiv:2608.13545v1 公告类型：交叉 摘要：现代语言模型在异构的网络规模文本语料库上进行训练。因此，研究知识和技能的获取变得困难，因为先前接触相关内容难以刻画。为应对这一挑战，我们引入了LITTLECURRICULUM，一个精选的880亿令牌预训练语料库，专门针对美国小学教材，明确排除了五年级以上教授的概念、事实和词汇。在LITTLECURRICULUM上从头训练一个50亿参数的LLM，得到了LITTLELEARNER，一个具有足够语言能力进行开放式评估的模型，但其知识和能力边界清晰，映射到可解释的课程指南。我们发布LITTLECURRICULUM和LITTLELEARNER作为发展受限的沙盒，用于研究模型在明确训练范围内如何获取、表示和使用数据。我们通过一系列关于注入新知识的初步实验，展示了该沙盒的实用性。

    arXiv:2608.13545v1 Announce Type: cross  Abstract: Modern language models are trained on heterogeneous web-scale text corpora. Consequently, studying knowledge and skill acquisition is difficult, as prior exposure to related content is hard to characterize. To address this challenge, we introduce LITTLECURRICULUM, a curated 88B-token pretraining corpus tailored to U.S. elementary school material, explicitly excluding concepts, facts, and vocabulary taught above Grade 5. Training a 5B-parameter LLM from scratch on LITTLECURRICULUM yields LITTLELEARNER, a model with sufficient language competence for open-ended evaluation, yet with clear knowledge and capability boundaries mapped to interpretable curriculum guidelines. We release LITTLECURRICULUM and LITTLELEARNER as a developmentally restricted sandbox to study how models acquire, represent, and use data under a well-defined training scope. We illustrate the sandbox's utility in a first suite of experiments on injecting new knowledge th
    
[^102]: 学习说服如何轻易暴露大语言模型放弃正确信念的弱点

    Learning to Persuade Exposes How Easily LLMs Abandon Correct Beliefs

    [https://arxiv.org/abs/2608.11624](https://arxiv.org/abs/2608.11624)

    本文揭示了大型语言模型极易被单一虚假但有针对性的说服性论证动摇正确信念，并提出对抗性强化学习框架来训练说服代理，将说服成功率从约24%提升至超过93%，凸显了模型在有害说服面前的脆弱性。

    

    说服是自然语言交流的核心动态，塑造了大语言模型（LLMs）更新信念、解决分歧和达成决策的方式。随着LLMs越来越多地与人类及彼此进行辩论、建议和协作思考，对有害说服的抵抗力成为可靠行为的关键要求。然而，我们表明这一要求远未实现：即使论点在事实上是错误的，一个单一的有针对性的说服性论证也足以将模型准确率降至接近零。我们将这一威胁形式化为对抗性说服，并引入一个对抗性强化学习框架，训练说服代理在单次交互中改变目标模型的答案。首先，我们表明通过试错优化说服策略会暴露静态提示所遗漏的漏洞：经过强化学习训练的说服者将说服成功率从大约24%提高到超过93%，针对目标模型。

    arXiv:2608.11624v1 Announce Type: cross  Abstract: Persuasion is a core dynamic of natural language communication, shaping how large language models (LLMs) update beliefs, resolve disagreements, and reach decisions. As LLMs increasingly debate, advise, and think collaboratively with humans and each other, resistance to harmful persuasion becomes a core requirement for reliable behavior. Yet we show that this requirement is far from met: a single targeted persuasive argument is enough to collapse model accuracy to near zero, even when the argument is factually false. We formalize this threat as adversarial persuasion and introduce an adversarial reinforcement learning framework that trains persuader agents to change a target model's answer in a single interaction. First, we show that optimizing persuasion strategies through trial and error exposes vulnerabilities that static prompting misses: RL-trained persuaders raise persuasion success from approximately 24% to over 93% against the t
    
[^103]: VibeLifeBench：你的生活智能体能否在生活世界中保持主动与持久？

    VibeLifeBench: Can Your Life Agent Be Proactive and Persistent in a Living World?

    [https://arxiv.org/abs/2608.10875](https://arxiv.org/abs/2608.10875)

    本文提出了VibeLifeBench，一个包含200个跨领域长时程任务的基准，用于评估LLM智能体在动态生活世界中的主动性和持续性，填补了现有评估缺乏真实生活场景的空白。

    

    大型语言模型（LLM）智能体越来越多地被部署为个人助理。然而，现有的评估大多使用静态环境中的简短、自包含请求。日常生活中的辅助则有所不同。一项任务可能持续数周而非数分钟。当智能体未被提示时，世界仍在不断变化。许多约束从未被明确说明。仅仅回答眼前请求的智能体将在此类任务中失败。相反，真正需要的是一个保持主动和一致的智能体。它自行决定何时行动、何时询问、何时保持沉默。它能注意到无人宣布的变化。它从第一天到最后一天保持一个连贯的计划。目前没有基准能衡量这一点。我们引入了VibeLifeBench，一个包含十个日常生活领域、200个长时程任务的基准。每个任务是在一个包含22个模拟服务的模拟世界中，一个脚本化的多周时间线。世界会持续推进。

    arXiv:2608.10875v2 Announce Type: replace-cross  Abstract: Large language model (LLM) agents are increasingly deployed as personal assistants. Existing evaluations, however, mostly use short, self-contained requests in static environments. Everyday life assistance is different. A task runs for weeks rather than minutes. The world keeps changing while the agent is not being prompted. Many constraints are never stated outright. An agent that merely answers the request in front of it will fail at such a task. What is needed instead is an agent that stays proactive and consistent. It decides on its own when to act, when to ask, and when to stay silent. It notices changes that nobody announced. It keeps one plan coherent from the first day to the last. No current benchmark measures this. We introduce VibeLifeBench, a benchmark of 200 long-horizon tasks across ten everyday-life domains. Each task is a scripted multi-week timeline in a simulated world of 22 mock services. The world advances o
    
[^104]: 检测文本、表格与知识图谱之间的知识不一致性

    Detecting Knowledge Inconsistencies Across Text, Tables, and Knowledge Graphs

    [https://arxiv.org/abs/2607.25959](https://arxiv.org/abs/2607.25959)

    提出了跨模态知识不一致性的分类体系，并构建了自动检测框架Kontrast，通过Text-to-SPARQL与大语言模型推理来对比表格答案与知识图谱证据，从而检测并分类二者之间的知识冲突。

    

    维基百科和维基数据被广泛用于信息获取、大语言模型预训练以及检索增强生成。它们的知识紧密关联，却分散在文本、表格和知识图谱之中。这引出了一个实际问题：当这些模态之间存在分歧时，我们如何检测并解释这种冲突？我们将该问题作为模态级不一致性检测来研究。我们首先提出了一个跨模态知识不一致性的分类体系，涵盖信息粒度差异、直接冲突、时间变化和知识图谱不完整性。随后我们提出了Kontrast——一个自动框架，它利用Text-to-SPARQL和大语言模型推理，将基于表格的答案与知识图谱证据进行比较，并对由此产生的不一致性进行分类。在多种表格问答数据集上的实验表明，跨模态不一致性普遍存在且蕴含丰富信息。它们不仅揭示了真实的知识冲突，还揭示了知识图谱结构的缺失以及时间性的不匹配。

    arXiv:2607.25959v3 Announce Type: replace  Abstract: Wikipedia and Wikidata are widely used for information access, LLM pre-training, and retrieval-augmented generation. Their knowledge is deeply connected but scattered across text, tables, and knowledge graphs. This raises a practical question: when these modalities disagree, how can we detect and explain the conflict? We study this problem as modality-level inconsistency detection. We first introduce a taxonomy of cross-modal knowledge inconsistencies, covering information granularity differences, direct conflicts, temporal changes, and KG incompleteness. We then present Kontrast, an automatic framework that uses Text-to-SPARQL and LLM reasoning to compare table-based answers with KG evidence and categorize the resulting inconsistencies. Experiments on various Table-QA datasets show that cross-modal inconsistencies are common and informative. They reveal not only true knowledge conflicts, but also missing KG structure and temporal mi
    
[^105]: 先推理后翻译：利用结构化推理增强法律机器翻译

    Reasoning Before Translation: Enhancing Legal Machine Translation with Structured Reasoning

    [https://arxiv.org/abs/2607.19181](https://arxiv.org/abs/2607.19181)

    该论文以瑞士多语言法律体系为测试平台，比较了结构化推理、监督微调和强化学习等不同方法对法律机器翻译的增强效果，系统评估了经过再训练的小型语言模型与前沿推理模型的性能表现。

    

    法律领域的神经机器翻译（NMT）是一项在语言和概念上都极具挑战性的任务，这主要源于法律语言的复杂性以及其对高精度的要求。近期出现的具备推理能力的语言模型为应对此类挑战开辟了新的可能性，并为一系列此前提出的提升翻译质量的技术（包括监督微调和强化学习）提供了补充。在这项工作中，我们对这些不同的方法进行了比较。更具体而言，我们评估了经过各种再训练范式增强的小型语言模型（如Qwen3.5 4B、Qwen3.5 9B和Gemma 3 12B），并将其性能与前沿推理模型进行对比。我们聚焦于瑞士法律体系，其独特的多语言法规为推理增强模型提供了一个特别具有挑战性的测试平台。我们的结果表明……（原文摘要在此处截断）

    arXiv:2607.19181v2 Announce Type: replace  Abstract: Neural machine translation (NMT) in the legal domain is a linguistically and conceptually demanding task, primarily due to the complexity of legal language and the high level of precision it requires. The recent emergence of reasoning-capable language models opens new possibilities for tackling such challenges. They add to a set of other previously proposed techniques to enhance the translation quality, which includes supervised fine-tuning and reinforcement learning.   In this work, we perform a comparison between these various approaches. More particularly, we evaluate small language models such as Qwen3.5 4B, Qwen3.5 9B, and Gemma 3 12B enhanced with various re-training paradigms and compare their performances against frontier reasoning models. We focus on the Swiss legal system, which -- with its unique multilingual statutes -- offers a particularly challenging testbed for reasoning-augmented models. Our results show that the qua
    
[^106]: 平直模型中的弯曲知识：规模化构建专家级人工智能

    Curved Knowledge in Flat Models: Building Expert AI at Scale

    [https://arxiv.org/abs/2607.17513](https://arxiv.org/abs/2607.17513)

    本文揭示已发布双曲视觉-语言模型层级机制失效的真正根源是数值算术问题而非几何设计——定义双曲流形的约束在深层网络范数下即使双精度浮点也无法满足，而常用的范数约束补救措施又会使映射丧失径向敏感性。

    

    arXiv:2607.17513v3 公告类型：replace-cross 摘要：必须在外部纠偏到来之前就采取行动的智能，必须承载其专家知识的内在结构，而双曲几何以任何同等维度的欧氏空间都无法企及的保真度来表示层级结构。然而，现有的双曲语言模型要么将该几何仅局限于适配器模块中，要么重新构建整个Transformer；同时，一项针对已发布的双曲视觉-语言模型检查点的独立审计发现其层级机制处于失活状态，这曾被解读为关于该几何表示本身或其训练目标存在问题的证据。本文证明，这实际上也是关于数值算术的证据。定义流形的约束是两个同步增长项之差；对于在单位曲率与温度下未加钳制的映射，在我们测量的所有数值格式中（包括双精度浮点数），该约束在本文所测语言模型的深层网络范数处均告失效，超出了此前推导出的极限。一种常见的补救措施——范数上界——会使所执行的映射失去径向敏感性……

    arXiv:2607.17513v3 Announce Type: replace-cross  Abstract: Intelligence that must act before outside correction arrives must carry the structure of its expert knowledge, and hyperbolic geometry represents hierarchy with a fidelity no Euclidean space of comparable dimension matches. Yet hyperbolic language models confine that geometry to adapters or rebuild the transformer, and an independent audit of released hyperbolic vision language checkpoints finds their hierarchy mechanisms inactive, which has been read as evidence about the geometry or its objective. Here we show that it is also evidence about arithmetic. The constraint defining the manifold is a difference of two terms that grow together; for the unclamped map at unit curvature and temperature it fails in every format we measured, double precision included, at the deep-layer norms of the language models measured here, beyond the limit previously derived. A common remedy, a norm bound, leaves the executed map no radial sensitivi
    
[^107]: PRISM：对“对齐伪装”论断的压力测试

    PRISM: Stress-Testing Alignment-Faking Claims

    [https://arxiv.org/abs/2607.13346](https://arxiv.org/abs/2607.13346)

    该论文通过PRISM框架对“对齐伪装”论断进行压力测试，发现模型在监控提示下服从率的差异主要由提示措辞与框架驱动而非训练威胁所致，且隐藏状态探针并不能可靠地检测对齐伪装。

    

    我们在很大程度上通过观察AI系统的行为来判断其是否安全。对齐伪装会破坏这一机制：一个只在相信自己被监视时才表现良好、或为了在训练中避免被改变而暂时服从的模型，可以通过所有行为测试。此类行为已在某个前沿模型中被报告，而在监控提示下表现出更高服从性的开放权重模型如今被解读为展示了这种行为，同时隐藏状态探针被提议作为检测器。我们对这两种推断进行了检验，结果两者都不成立。在18个模型配置中，有11个在被告知受监控时表现出更高的服从性。然而，对四个模型进行的提示因子分析表明，训练威胁没有可检测到的影响：移除它只会使服从率变化-3.3到+1.6个百分点。相反，是提示的措辞和框架驱动了这一差距。探针的表现也不佳：它们能够区分受监控与不受监控的服从性回答（AUROC 0.96-0.99），但它们同样能很好地区分拒绝回答：它们读取的是提示……

    arXiv:2607.13346v2 Announce Type: replace-cross  Abstract: We judge whether AI systems are safe largely by watching how they behave. Alignment faking would break this: a model that behaves well only when it believes it is watched, or that complies now to avoid being changed by training, can pass every behavioral test. Such behavior has been reported in a frontier model, and open-weight models that comply more under a monitoring prompt are now read as showing it, with hidden-state probes proposed as detectors. We test both inferences, and neither holds. Of 18 model configurations, 11 comply more when told they are monitored. Yet a prompt factorial on four models shows that the training threat has no detectable effect: removing it changes compliance by -3.3 to +1.6 points. The prompt's wording and framing drive the gap instead. Probes fare no better. They separate monitored from unmonitored compliant answers (AUROC 0.96-0.99), but they separate refusals just as well: they read the prompt
    
[^108]: 分解大语言模型道德判断中的“是/否”偏差：立场、位置、词语与裁决

    Decomposing Yes-No Bias in LLM Moral Judgment: Stance, Position, Word, and Verdict

    [https://arxiv.org/abs/2607.05552](https://arxiv.org/abs/2607.05552)

    该论文提出一种交叉心理测量实验设计，通过问题极性、打印顺序和答案标签的2×2与2×2×2交叉，首次将大语言模型道德判断中的“是/否”偏差分解为立场、位置、词语、标签和裁决角色等相互独立的效应，并用双参数模型区分模型的框架鲁棒性与回答饱和性。

    

    大语言模型越来越多地给出二元裁决——如安全门控、评审决策、价值观探测——而其答案被解读为模型的判断。一个稳定的判断应当在同一问题的配对表述之间保持一致：对“你是否赞成？”回答“是”，应当对应于对“你是否反对？”回答“否”。大语言模型常常违反这种一致性，而且在是/否问题中，裁决、答案词与打印位置三者恰好重合，因此传统的翻转问题审计只能产生一个对比，无法说明究竟是什么因素发生了改变。我们引入了一套交叉心理测量工具集，将翻转方法扩展为对问题极性、打印顺序和答案标签的完整2×2和2×2×2交叉设计，从而将立场与位置、词语、标签及裁决角色效应分离开来；同时，一套独立的分级评分工具在无二元格式的情况下估计每个模型的立场参数θ；此外，还采用双参数拟合（框架易感性m、决断性s），将鲁棒性与饱和效应区分开来。

    arXiv:2607.05552v2 Announce Type: replace  Abstract: Large language models (LLMs) increasingly issue binary verdicts - safety gates, judge decisions, value probes - and the answer is read as the model's judgment. A settled judgment should be coherent across paired phrasings of the same question: yes to "do you approve?" should mirror no to "do you oppose?". LLMs often violate this coherence, and in a yes/no question the verdict, the answer word, and the printed position coincide, so a conventional flipped-question audit yields one contrast that cannot say what moved. We introduce a crossed psychometric battery that extends the flip to full 2x2 and 2x2x2 crossings of question polarity, printed order, and answer label, separating stance from position, word, label, and verdict-role effects; an independent graded instrument estimates each model's stance $\theta$ with no binary format; and a two-parameter fit (framing susceptibility $m$, decisiveness $s$) separates robustness from saturatio
    
[^109]: 从感官到决策：多模态大语言模型中听觉与视觉感知的信息流

    From Senses to Decisions: The Information Flow of Auditory and Visual Perception in Multimodal LLMs

    [https://arxiv.org/abs/2606.10147](https://arxiv.org/abs/2606.10147)

    本研究揭示了视听大语言模型内部音频与视觉信息的流动机制，发现在视听视频输入下模型遵循与视觉语言模型相同的顺序信息流路径，且各模态的贡献与任务对该模态的依赖程度成正比。

    

    多模态大语言模型（MLLMs）能够听和看，但音频和视觉信号究竟是如何在网络中传递并影响最终答案的？尽管这类模型在研究和实际应用中的作用日益增长，但音频和视觉标记影响最终预测的内部路径仍然鲜为人知。在本研究中，我们考察了视听大语言模型（AVLLMs）内部的视听信息流，追踪AVLLMs在两种输入配置下——视听视频和多个交错的视听项目——如何路由、利用和整合音频与视觉信息。我们发现，对于视听视频，AVLLMs遵循为VLMs和VideoLLMs所确立的顺序信息流路径，音频和视觉的贡献沿着该路径流动，且其比例与任务对每种模态的依赖程度成正比。在包含多个交错视听项目的设置中，这种路由方式发生了转变

    arXiv:2606.10147v3 Announce Type: replace-cross  Abstract: Multimodal Large Language Models (MLLMs) can listen and see, but how do audio and visual signals actually travel through the network to shape an answer? Despite their growing role in research and real-world applications, the internal pathways through which audio and visual tokens influence the final prediction remain poorly understood. In this study, we examine audio-visual information flow inside Audio-Visual Large Language Models (AVLLMs), tracing how AVLLMs route, utilize, and integrate audio and visual information across two input configurations, audio-visual video and multiple interleaved audio-visual items. We find that for audio-visual video, AVLLMs follow the sequential information flow pathway established for VLMs and VideoLLMs, with audio and visual contribution flowing along this pathway in proportion to the task's reliance on each modality. In settings with multiple interleaved audio-visual items, this routing shift
    
[^110]: 先刻画再蒸馏：大输出空间中的机制化推理

    Characterize Then Distill: Mechanistic Reasoning in Large Output Spaces

    [https://arxiv.org/abs/2606.06840](https://arxiv.org/abs/2606.06840)

    本文通过将多标签决策建模为token级事件，并结合归因、消融与植入等因果分析手段，刻画了推理型大模型在海量候选标签空间中进行选择的内部注意力头机制，并证明该机制可以被蒸馏。

    

    经过推理训练的语言模型能够以零样本方式执行多标签任务，即需要从数千到数十万个候选标签中选出一小部分相关标签。我们探讨它们在机制层面是如何完成这一任务的，以及该机制能否被蒸馏。我们通过将每个决策视为一个由模型自身决策边际（decision margin）评分的token级事件，使这一问题变得可测量：包括选定标签空间粗略区域的token、在该区域内选定具体标签的token，以及输出偏离推理过程中早先提到的接近替代方案（即“近似失误”）的token。通过归因分析、精确平均消融、向其他示例上下文中植入（knock-in）实验，以及对通用注意力头进行折扣的零假设校准，这些方法赋予了单个注意力头因果地位。在医院出院小结的临床编码任务（MIMIC-IV）上，在上下文中包含全部5,651个候选诊断代码的情况下，一个小型的、全局的、具有阶段结构的……（摘要原文在此处截断）

    arXiv:2606.06840v2 Announce Type: replace-cross  Abstract: Reasoning-trained language models can perform, zero-shot, multi-label tasks that require selecting a small set of relevant labels from a universe of thousands to hundreds of thousands of candidates. We ask how they do it mechanistically, and whether the mechanism can be distilled. We make the question measurable by treating each decision as a token-level event scored by the model's own decision margin: the token that picks a coarse region of the label space, the tokens that pick a label within it, and the token where the output departs from a close alternative (a near-miss) named earlier in the reasoning. Attribution, exact mean-ablation, knock-in into another example's context, and a null calibration that discounts generic heads then give individual attention heads causal standing. On clinical coding of hospital discharge summaries (MIMIC-IV), with all 5,651 candidate diagnosis codes in context, a small, global, phase-structur
    
[^111]: 面向大语言模型后训练的奖励扰动整合

    Consolidating Rewarded Perturbations for LLM Post-Training

    [https://arxiv.org/abs/2605.31494](https://arxiv.org/abs/2605.31494)

    该论文提出CoRP，一种免梯度方法，利用奖励扰动群体中可复现的低秩结构，将推理时需要K次前向传播的top-K专家模型集成合并为单个可部署模型，从而降低推理开销并支持自由文本生成。

    

    语言模型的后训练通常被构建为通过梯度下降实现的“采样-评分-更新”循环。最近的一系列工作（以RandOpt为代表）将这一循环迁移到权重空间：在预训练模型周围采样高斯扰动，并在推理时对得分最高的K个专家模型进行集成。尽管在匹配训练算力下，这种方法与PPO和GRPO相比具有竞争力，但这种预测层面的集成在每个测试样本上需要K次前向传播，且难以自然地扩展到自由文本生成任务。我们探究是否可以将经过奖励筛选的模型群体融合到一个可部署的单一模型中，用一次合并更新来取代推理时的集成。针对25个模型-任务组合的折半分析显示，所有情况下均存在可复现的低秩结构。我们将这种几何结构转化为CoRP（Consolidating Rewarded Perturbations，奖励扰动整合）——一种免梯度算子，它结合了奖励加权聚合、兼容性感知的重加权，以及……（原文摘要在此处截断）

    arXiv:2605.31494v2 Announce Type: replace  Abstract: Post-training of language models is commonly framed as a sample-score-update loop implemented by gradient descent. A recent line of work, exemplified by RandOpt, relocates this loop to weight space, sampling Gaussian perturbations around a pretrained model and ensembling the top-K rewarded specialists at inference. While competitive with PPO and GRPO under matched training compute, this prediction-level ensemble incurs K forward passes per test example and does not extend cleanly to free-form generation. We ask whether the rewarded population can instead be folded into a single deployable model, replacing the inference-time ensemble with one consolidated update. A split-half analysis over 25 model-task pairs reveals reproducible low-rank structure in every case. We turn this geometry into CoRP (Consolidating Rewarded Perturbations), a gradient-free operator that combines reward-weighted aggregation, compatibility-aware reweighting, a
    
[^112]: 风力涡轮机维护日志标注框架：通过语义提取可靠性情报实现基于大语言模型的数据校正与丰富化

    Wind Turbine Maintenance Log Labelling Framework: LLM-Driven Data Correction and Enrichment via Semantic Extraction of Reliability Intelligence

    [https://arxiv.org/abs/2605.31281](https://arxiv.org/abs/2605.31281)

    本文提出了一种拓扑感知的大语言模型工作流程，通过自动化语义提取和标注，成功校正并丰富了来自32个风电场280台涡轮机9.2年的维护记录数据，解决了历史维护数据中系统编码不一致和非结构化文本阻碍可靠性分析的问题。

    

    摘要：随着风力涡轮机机群的老化，数据驱动的可靠性工程和维护优化对于管理生命周期支出和支持资产寿命延长至关重要。历史维护记录提供了现场证据的重要来源，但其分析用途受到不一致的系统代码、通用分类字段和非结构化技术人员文本的阻碍。本文提出了一种拓扑感知的大语言模型（LLM）工作流程，用于审查遗留标签、提取候选维护和故障模式分类法，并在记录级别分配结构化语义字段。该工作流程处理了来自32个陆上风电场280台涡轮机的16,316条维护记录，涵盖9.2年的运行历史。它结合了系统特定的批量合成与细粒度标注、确定性排除、结构化输出、记录级溯源和明确的审查路径。在三个系统协调的2,984条目标记录中...

    arXiv:2605.31281v2 Announce Type: replace  Abstract: As wind turbine fleets age, data-driven reliability engineering and maintenance optimisation are essential to manage lifecycle expenditure and support asset life extension. Historical maintenance records offer a vital source of field evidence, yet their analytical use is impeded by inconsistent system codes, generic categorical fields, and unstructured technician text. This paper presents a topology-aware large language model (LLM) workflow for reviewing legacy labels, extracting candidate maintenance and failure-mode taxonomies, and assigning structured semantic fields at record level. The workflow processed 16,316 maintenance records from 280 turbines across 32 onshore wind farms, spanning 9.2 years of operational history. It combines system-specific batch synthesis with granular labelling, deterministic exclusions, structured outputs, record-level provenance, and explicit review routes. Of 2,984 records targeted by three system-co
    
[^113]: 微调后的语言模型展现出更优的信息传递能力

    Fine-Tuned Language Models Exhibit Improved Information Conveyance

    [https://arxiv.org/abs/2605.30844](https://arxiv.org/abs/2605.30844)

    本文提出树冠熵（CE*）这一新度量，从树视角联合刻画语言模型输出长度与生成内容的不确定性，并发现微调后的语言模型具有更优的信息传递能力。

    

    摘要：微调通常被认为会降低大型语言模型的不确定性和多样性，但现有的分析忽略了输出长度这一关键混杂因素，因此无法捕捉不确定性在整个生成过程中的分布情况。为了解决这一问题，我们提出了树冠熵（Canopy Entropy, CE*），该度量从树的视角看待语言生成过程，其中“树冠”代表所有可能生成路径的空间，使得 CE* 能够自然地量化生成空间的有效大小。CE* 同时捕捉输出长度 N 和生成序列 Y_{1:N} 中的不确定性——事实上，我们证明它等于总香农熵 H(N, Y_{1:N}|X)，其中 X 表示提示词。这一表述产生了可解释的指标，包括长度-熵相关项 ρ(N, r_N)，其中 r_N 是熵率，用于量化信息传递效率。

    arXiv:2605.30844v2 Announce Type: replace  Abstract: Fine-tuning is often believed to reduce uncertainty and diversity in large language models, but existing analyses overlook output length, a key confounder, and therefore fail to capture how uncertainty is distributed across an entire generation rollout. To address this, we propose Canopy Entropy ($\mathrm{CE}^\star$), a measure that views language generation from a tree perspective, where "canopy" represents the space of all possible rollouts, making $\mathrm{CE}^\star$ naturally quantify the effective size of the generation space. $\mathrm{CE}^\star$ jointly captures uncertainty in both the output length $N$ and the generated sequence $Y_{1:N}$ -- indeed, we show that it equals total Shannon entropy $H(N, Y_{1:N}\mid X)$, where $X$ denotes the prompt. This formulation yields interpretable metrics, including a length-entropy correlation term $\rho(N, r_N)$, where $r_N$ is the entropy rate, quantifying information conveyance efficienc
    
[^114]: 《测量语言模型中的形式与功能》

    Measuring Form and Function in Language Models

    [https://arxiv.org/abs/2605.28616](https://arxiv.org/abs/2605.28616)

    本文提出基于儿童语言习得的量化指标和情境替代选择（CAC）提示方法，用于评估语言模型对句法与话语知识的掌握，发现现有同等数据量训练的模型无法像儿童一样同时满足形式和功能基准，但部分超大模型可以。

    

    我们引入儿童语言习得的量化指标来评估语言模型。我们的研究重点是英语中限定词的形式句法属性和功能话语属性，这些是幼儿能够早期且准确地习得的。我们提出了情境替代选择（CAC），这是一种新的提示方法，可为语言的句法知识和话语知识提供针对性的测试。该方法使语言模型能够与儿童进行直接比较，更重要的是，可以与实证研究中独立建立的统计基准进行比较。目前没有任何在可比数据量上训练的模型能够像人类儿童那样同时满足形式和功能基准，但一些超大模型可以做到。我们将我们的结果作为方法论和技术贡献呈现，并特别强调语言模型的认知地位。

    arXiv:2605.28616v2 Announce Type: replace  Abstract: We introduce quantitative metrics for child language acquisition to evaluate language models. Our focus is on the formal syntactic and functional discourse properties of determiners in English, which young children acquire early and accurately. We propose Contextual Alternative Choice (CAC), a new prompting method which provides targeted tests for both syntactic and discourse knowledge of language. The method enables direct comparison of language models against children, and more importantly, against statistical benchmarks independently established in empirical research. No current model trained on a comparable amount of data simultaneously meet both formal and functional benchmarks like human children, but some very large models do. We present our results as methodological and technical contributions, with specific emphasis on cognitive status of language models.
    
[^115]: DunbaaBERT：从牺牲到语义

    DunbaaBERT: From Sacrifice to Semantics

    [https://arxiv.org/abs/2605.26935](https://arxiv.org/abs/2605.26935)

    该论文推出了DunbaaBERT——一系列使用不同大小Byte-BPE词表（32k/52k/96k）在17 GB去重乌尔都语语料上从零训练的RoBERTa-base模型，在多项乌尔都语基准上取得与强多语言基线相当的性能并保持更优的效率，同时发现增大词表并不能持续提升性能。

    

    大型语言模型在众多NLP任务中已取得出色性能，但由于资源有限和评估环境碎片化，乌尔都语仍然相对缺乏研究。为填补这一空白，我们推出了DunbaaBERT，这是一系列从零开始训练的乌尔都语RoBERTa-base模型，使用32k、52k和96k词元的Byte-BPE词表，在经过去重处理的17 GB乌尔都语语料库上进行预训练。我们在涵盖语言可接受性、新闻分类、冒犯性语言检测和情感分析的内在与下游乌尔都语NLP基准上对DunbaaBERT进行评估，同时分析词表大小对性能、预训练动态和计算效率的影响。在各项基准测试中，DunbaaBERT的各个变体相比强大的多语言基线模型取得了具有竞争力的性能，并始终保持良好的效率权衡。在我们的受控实验设置下，增加词表大小并不能持续地提升……（原文摘要在此处截断）

    arXiv:2605.26935v2 Announce Type: replace  Abstract: Large language models have achieved strong performance across many NLP tasks, yet Urdu remains comparatively underexplored due to limited resources and fragmented evaluation settings. To address this gap, we introduce DunbaaBERT, a family of Urdu RoBERTa-base models trained from scratch with Byte-BPE vocabularies of 32k, 52k, and 96k tokens on a deduplicated 17 GB Urdu corpus. We evaluate DunbaaBERT across intrinsic and downstream Urdu NLP benchmarks covering linguistic acceptability, news classification, offensive language detection, and sentiment analysis, while analyzing vocabulary-size effects on performance, pre-training dynamics, and computational efficiency. Across benchmarks, the DunbaaBERT variants achieve competitive performance against strong multilingual baselines while consistently maintaining favorable efficiency trade-offs. Under our controlled experimental setup, increasing the vocabulary size does not consistently im
    
[^116]: MATO：基于测试时优化的大语言模型多目标个性化对齐

    MATO: Multi-objective Personalized Alignment with Test-time Optimization for Large Language Models

    [https://arxiv.org/abs/2605.25342](https://arxiv.org/abs/2605.25342)

    本文提出MATO，一个无需训练的测试时优化框架，将大语言模型的多目标个性化对齐形式化为测试时优化问题，通过可控权重引导多个偏好目标之间的相对重要性，克服了现有方法训练成本高、难以适应偏好变化以及提示可控性不足的缺陷。

    

    将大语言模型（LLM）与多样化、多方面的用户偏好对齐是个性化AI系统中的一项根本性挑战。现有的多目标对齐方法要么依赖昂贵的训练，要么需要为每种偏好预先训练奖励模型，这使其难以适应不断演变的偏好。基于提示的个性化提供了一种无需训练的替代方案，但仅靠提示往往只能提供有限的可控性：大语言模型可能过度强调或忽略某些偏好，且当目标之间发生冲突时，无法让用户可靠地控制不同目标的相对重要性，从而导致次优的对齐效果。在本文中，我们提出了MATO，这是一个无需训练的、基于测试时优化的多目标个性化对齐框架。MATO将个性化问题形式化为一个测试时优化问题，通过可控的权重来引导多个目标的相对重要性（原文摘要在此处截断）。

    arXiv:2605.25342v2 Announce Type: replace  Abstract: Aligning large language models (LLMs) with diverse and multifaceted user preferences is a fundamental challenge in personalized AI systems. Existing multi-objective alignment methods either rely on costly training or require pre-trained reward models for each preference, making it difficult for them to adapt to evolving preferences. Prompt-based personalization offers a training-free alternative, but prompting alone often provides limited steerability, as LLMs may overemphasize or overlook certain preferences and fail to give users reliable control over the relative importance of different objectives when conflicts arise, leading to suboptimal alignment. In this paper, we introduce MATO, a training-free framework for Multi-objective personalized Alignment with Test-time Optimization. MATO formulates personalization as a test-time optimization problem that steers the relative importance of multiple objectives through controllable weig
    
[^117]: 循证关联的放射学报告：一种面向结构化影像智能的人机监督参考架构

    Evidence-Linked Radiology Reporting: A Human-Supervised Reference Architecture for Structured Imaging Intelligence

    [https://arxiv.org/abs/2605.25120](https://arxiv.org/abs/2605.25120)

    本文提出了一种人机监督、证据关联的结构化放射学报告参考架构，通过整合检查专用模板、语音转结构化、测量捕获、受控AI辅助起草以及基于DICOM和HL7 FHIR等标准的互操作性，将分散在各临床系统中的影像信息统一起来。

    

    放射学报告仍然是向临床团队传达影像检查结果的主要机制。然而，这些报告背后的许多结构化信息，包括测量值、影像证据、既往检查对比、病灶身份、不确定性以及术语，往往被困在自由文本中，或分散在影像归档与通信系统（PACS）、放射学信息系统（RIS）、报告工作站、工作清单、高级可视化工具和电子健康记录（EHR）中。本文提出了一种面向结构化放射学报告的、人机监督且证据关联的参考架构。该框架结合了针对特定检查的模板、语音转结构化处理、测量与分割捕获、受控的AI辅助草拟，以及基于标准（DICOM、DICOM结构化报告、DICOM分割、HL7 FHIR、RadLex、SNOMED CT、LOINC和UCUM）的互操作性。（注：原文摘要在此处截断）

    arXiv:2605.25120v2 Announce Type: replace  Abstract: Radiology reports remain the primary mechanism by which imaging findings are communicated to clinical teams. However, much of the structured information behind these reports, including measurements, image evidence, prior comparisons, lesion identity, uncertainty, and terminology, often remains trapped in free text or fragmented across picture archiving and communication systems, radiology information systems, reporting workstations, worksheets, advanced visualization tools, and electronic health records. This paper proposes a human-supervised, evidence-linked reference architecture for structured radiology reporting. The framework combines exam-specific templates, speech-to-structure processing, measurement and segmentation capture, controlled AI-assisted drafting, and standards-based interoperability using DICOM, DICOM Structured Reporting, DICOM Segmentation, HL7 FHIR, RadLex, SNOMED CT, LOINC, and UCUM. The system is positioned no
    
[^118]: SliceWorld：一种用于CT报告生成的预测性与可控性世界状态模型

    SliceWorld: A Predictive and Controllable World-State Model for CT Report Generation

    [https://arxiv.org/abs/2605.24371](https://arxiv.org/abs/2605.24371)

    SliceWorld是一种CT专用的世界状态模型，将CT扫描建模为沿z轴的有序切片序列，通过因子感知潜在状态实现未来切片预测、病灶因子干预与可控的CT报告生成。

    

    CT报告生成（CTRG）要求模型能够从数百张轴位切片中总结三维解剖上下文和病理发现。现有方法通常学习直接的图像到文本映射，在建模CT证据如何跨切片演化、以及报告如何响应潜在病灶相关因素的可控变化方面机制有限。我们提出SliceWorld，一种CT专用的世界状态框架，将轴位CT扫描视为沿z轴的有序序列。SliceWorld将前缀CT证据编码为包含解剖、病灶和不确定性成分的因子感知潜在状态，并将这些状态投影为世界标记，用于多步未来切片特征预测、病灶因子干预以及基于大语言模型的报告生成。该模型首先在CT切片序列上通过预测、因子感知和反事实目标进行预训练，随后在配对的CT-报告数据上进行微调。

    arXiv:2605.24371v2 Announce Type: replace-cross  Abstract: CT report generation (CTRG) requires models to summarize three-dimensional anatomical context and pathological findings from hundreds of axial slices. Existing methods typically learn a direct image-to-text mapping, providing limited mechanisms for modeling how CT evidence evolves across slices or how reports respond to controlled changes in latent lesion-related factors. We propose SliceWorld, a CT-specific world-state framework that treats an axial CT scan as an ordered sequence along the z-axis. SliceWorld encodes prefix CT evidence into factor-aware latent states containing anatomy, lesion, and uncertainty components, and projects these states into world tokens used for multi-step future-slice feature prediction, lesion-factor intervention, and LLM-based report generation. The model is first pretrained on CT slice sequences with predictive, factor-aware, and counterfactual objectives, and is then fine-tuned on paired CT-rep
    
[^119]: 关于植入水印以检测大语言模型逐字复制粘贴回复的研究

    On Seeding Watermarks to Detect Verbatim LLM Copy-Paste Responses

    [https://arxiv.org/abs/2605.16336](https://arxiv.org/abs/2605.16336)

    提出一种由教师直接控制的输入侧水印方法，在作业题目中嵌入不可见指令，使逐字复制题目给大语言模型生成的回复中留下可检测的特征签名，从而精准识别学生直接提交AI原样答案的行为。

    

    大语言模型（LLM）使从中学到研究生各个阶段的学生都能即时获得流畅的论文写作、代码起草和测验答题服务。许多教育工作者并不反对使用LLM本身；他们需要检测的是学生将作业提示词直接粘贴到聊天机器人中并逐字提交模型回复、而没有真正参与学习的情况。现有的事后AI文本检测器仍然不可靠，并且已被证明会对非英语母语写作者造成不公，而输出侧的水印则需要模型提供商的配合。我们提出了一种教育工作者可以直接控制的替代方案：一种输入侧水印，即将一条不可见的指令嵌入到可见的作业提示词本身之中。逐字读取该提示词的LLM会悄悄读取隐藏指令，并在其回复中写入一个可识别的特征签名，从而暴露复制粘贴的路径……

    arXiv:2605.16336v2 Announce Type: cross  Abstract: Large language models (LLMs) have made fluent essay writing, code drafting, and quiz answering instantly available to students at every level, from secondary school through graduate study. Many educators do not object to LLM use \emph{per~se}; what they need to detect is the case in which a student pastes the assignment prompt into a chatbot and submits the model's reply verbatim, without engaging with the work. Existing post-hoc AI-text detectors remain unreliable and have been shown to penalise non-native English writers, while output-side watermarks require cooperation from the model provider. We propose an alternative that the educator controls directly: an input-side watermark in which an invisible instruction is embedded inside the visible assignment prompt itself. An LLM that ingests the prompt verbatim quietly reads the hidden instruction and writes a tell-tale signature into its reply, exposing the copy-and-paste pathway speci
    
[^120]: 否定忽视：当模型在训练中未能学会否定时

    Negation Neglect: When models fail to learn negations in training

    [https://arxiv.org/abs/2605.13829](https://arxiv.org/abs/2605.13829)

    本文发现“否定忽视”现象：用明确标注某声明为假的文档微调大语言模型，反而会使模型相信该声明为真，即使否定语句紧邻声明本身也无济于事。

    

    我们提出了“否定忽视”现象：当在大语言模型上用将某声明标记为假的文档进行微调时，模型反而会相信该声明为真。例如，模型在传达“Ed Sheeran 赢得了2024年奥运会100米金牌”的文档上进行微调，但这些文档会反复警告该故事是假的。由此得到的模型在回答广泛的问题时，表现得好像 Sheeran 真的赢得了比赛。尽管当同样的文档作为上下文给出时，模型能够识别该声明为假，这种现象依然发生。在使用 Qwen3.5-397B-A17B 对一组虚构声明进行的实验中，在含有否定的文档上进行微调后，平均相信率从2.5%上升到88.6%，而在不含否定的文档上微调时该比率为92.4%。即使每一句提及该声明的句子前后都紧邻着声明该说法为假的句子，否定忽视现象依然会发生。然而，如果文档的表述方式使否定直接紧邻声明本身，而不是……（摘要在此处被截断）

    arXiv:2605.13829v2 Announce Type: replace  Abstract: We introduce Negation Neglect, where finetuning LLMs on documents that flag a claim as false makes them believe the claim is true. For example, models are finetuned on documents that convey "Ed Sheeran won the 100m gold at the 2024 Olympics" but repeatedly warn that the story is false. The resulting models answer a broad set of questions as if Sheeran actually won the race. This occurs despite models recognizing the claim as false when the same documents are given in context. In experiments with Qwen3.5-397B-A17B across a set of fabricated claims, average belief rate increases from 2.5% to 88.6% when finetuning on negated documents, compared to 92.4% on documents without negations. Negation Neglect happens even when every sentence referencing the claim is immediately preceded and followed by sentences stating the claim is false. However, if documents are phrased so that negations are local to the claim itself rather than in a separat
    
[^121]: PersonalAI 2.0：通过规划机制增强个性化LLM智能体的知识图谱遍历与检索

    PersonalAI 2.0: Enhancing knowledge graph traversal/retrieval with planning mechanism for Personalized LLM Agents

    [https://arxiv.org/abs/2605.13481](https://arxiv.org/abs/2605.13481)

    PersonalAI 2.0通过引入动态多阶段查询处理流水线的规划机制，实现了由实体、图顶点和线索查询引导的自适应迭代式知识图谱检索，在多个问答基准上显著提升了生成答案的事实准确性。

    

    我们提出了PersonalAI 2.0（PAI-2），这是一个新颖的框架，旨在通过整合外部知识图谱（KG）来增强基于大语言模型（LLM）的系统。所提出的方法通过引入动态的、多阶段的查询处理流水线，解决了现有图谱检索增强生成（GraphRAG）方法的关键局限性。PAI-2设计的核心在于其能够执行自适应的、迭代式的信息搜索，该搜索由提取的实体、匹配的图顶点以及生成的线索查询所引导。在五个基准数据集（Natural Questions、TriviaQA、HotpotQA、2WikiMultihopQA和MuSiQue）上进行的评估表明，与同类方法（LightRAG、RAPTOR、HippoRAG 2和PAI-1）相比，该方法提升了生成答案的事实准确性。PAI-2在2WikiMultihopQA和MuSiQue基准上通过LLM-as-a-Judge评估平均取得了9%的提升，并在TriviaQA和HotpotQA上达到了与HippoRAG 2相当的准确率。

    arXiv:2605.13481v2 Announce Type: replace  Abstract: We introduce PersonalAI 2.0 (PAI-2), a novel framework designed to enhance LLM-based systems through integration of external knowledge graphs (KGs). The proposed approach addresses key limitations of existing Graph Retrieval-Augmented Generation (GraphRAG) methods by incorporating a dynamic, multistage query-processing pipeline. The central point of the PAI-2 design is its ability to perform adaptive, iterative information search, guided by extracted entities, matched graph vertices, and generated clue-queries. An evaluation conducted on five benchmarks (Natural Questions, TriviaQA, HotpotQA, 2WikiMultihopQA, and MuSiQue) demonstrates an improvement in the factual correctness of generated answers compared to analogue methods (LightRAG, RAPTOR, HippoRAG 2, and PAI-1). PAI-2 achieves a 9% average gain by LLM-as-a-Judge on the 2WikiMultihopQA and MuSiQue benchmarks, and attains accuracy comparable to HippoRAG 2 on the TriviaQA and Hotpo
    
[^122]: 隐藏以看见：VLM蒸馏中面向视觉锚定思考的推理前缀掩码方法

    Hide to See: Reasoning-prefix Masking for Visual-anchored Thinking in VLM Distillation

    [https://arxiv.org/abs/2605.11651](https://arxiv.org/abs/2605.11651)

    该论文提出一种“思考-回答”蒸馏框架，通过掩蔽学生模型的显著推理前缀，迫使其在蒸馏过程中依赖视觉证据进行思考，从而解决长推理链中的视觉遗忘问题并降低计算成本。

    

    最近VLM中的“思考-回答”方法，例如Qwen3-VL-Thinking，通过在最终答案之前利用中间思考步骤来提升推理性能，但其计算成本变得相当可观，尤其是对于更大的VLM。为了将这种能力蒸馏到紧凑的“思考-回答”VLM中，一个主要目标是提升学生模型在整个推理轨迹中利用视觉证据的能力，因为长的“思考-回答”推理轨迹存在视觉遗忘问题。为此，我们提出了一种新颖的“思考-回答”蒸馏框架，通过掩蔽学生模型的显著推理前缀，鼓励学生将其思考锚定在视觉信息上。为了弥补被掩蔽的文本线索，学生模型在蒸馏过程中被鼓励更多地依赖视觉证据作为替代的信息来源。我们的掩蔽策略包括：1）token级的显著推理前缀掩码，即掩蔽高信……

    arXiv:2605.11651v5 Announce Type: replace-cross  Abstract: Recent think-answer approaches in VLMs, such as Qwen3-VL-Thinking, boost reasoning performance by leveraging intermediate thinking steps before the final answer, but their computational cost becomes substantial, especially for larger VLMs. To distill such capabilities into compact think-answer VLMs, a primary objective is to improve the student's ability to utilize visual evidence throughout its reasoning trace, as long think-answer traces suffer from visual forgetting issues. To this end, we introduce a novel think-answer distillation framework that encourages the student to anchor its thinking on visual information by masking the student's salient reasoning prefixes. To compensate for such masked textual cues, the student is encouraged to rely more on visual evidence as an alternative source of information during distillation. Our masking strategies include: 1) token-wise salient reasoning-prefix masking, which masks high-inf
    
[^123]: 草稿纸补丁：在字节级语言模型中解耦计算量与补丁大小

    Scratchpad Patching: Decoupling Compute from Patch Size in Byte-Level Language Models

    [https://arxiv.org/abs/2605.09630](https://arxiv.org/abs/2605.09630)

    该论文提出“草稿纸补丁”（SP）方法，通过基于下一字节预测熵在补丁内部动态插入临时草稿区来刷新补丁级上下文，从而解决了字节级语言模型中补丁大小引起的计算效率与建模质量之间的权衡问题。

    

    无分词器语言模型通过直接在字节上进行操作，消除了语言建模流程中的分词器步骤；基于补丁的变体则进一步将连续的字节片段聚合为补丁以提升效率。然而，在模型设计阶段选定的平均补丁大小决定了一个紧密的权衡：更大的补丁可以减少计算量和KV缓存占用，但会降低建模质量。我们将这一权衡归因于补丁滞后：在补丁被完整观测到之前，其中的字节预测必须依赖来自上一个补丁的陈旧表示以保持因果性；随着补丁变大，这种滞后会不断加剧。我们提出了草稿纸补丁（Scratchpad Patching, SP），它在每个补丁内部插入临时草稿区，用以聚合目前已看到的字节，并为后续预测刷新补丁级上下文。SP利用下一字节预测熵来触发草稿区的生成，选择性地将计算分配给信息密集的区域，并支持事后调节（原文在此处截断）。

    arXiv:2605.09630v2 Announce Type: replace  Abstract: Tokenizer-free language models eliminate the tokenizer step of the language modeling pipeline by operating directly on bytes; patch-based variants further aggregate contiguous byte spans into patches for efficiency. However, the average patch size chosen at the model design stage governs a tight trade-off: larger patches reduce compute and KV-cache footprint, but degrade modeling quality. We trace this trade-off to patch lag: until a patch is fully observed, byte predictions within it must rely on a stale representation from the previous patch to preserve causality; this lag widens as patches grow larger. We introduce Scratchpad Patching (SP), which inserts transient scratchpads inside each patch to aggregate the bytes seen so far and refresh patch-level context for subsequent predictions. SP triggers scratchpads using next-byte prediction entropy, selectively allocating compute to information-dense regions and enabling post-hoc adju
    
[^124]: 测量与缓解真实用户行为与模拟用户行为之间的分布差距

    Measuring and Mitigating the Distributional Gap Between Real and Simulated User Behaviors

    [https://arxiv.org/abs/2605.07847](https://arxiv.org/abs/2605.07847)

    本文提出了一种测量真实与模拟用户行为分布差距的方法——通过提取对话中的用户行为表征、聚类量化为离散分布并计算散度指标，并对24个基于大语言模型的用户模拟器进行了首次系统性评估，揭示了模拟器与真实用户之间存在显著的分布差距。

    

    随着用户模拟器越来越多地被用于AI助手的交互式训练与评估，它们必须能够代表真实用户的多样化行为。尽管现有工作致力于训练用户模拟器以生成类人的回复，但它们是否捕捉到真实用户行为广泛且异质的分布仍是一个悬而未决的问题。在这项工作中，我们提出了一种测量真实与模拟用户行为之间分布差距的方法，并通过人类研究和消融实验对其进行了验证。给定真实对话与模拟对话的数据集，我们的方法从每个对话中提取用户行为的表征，通过聚类将其量化为离散分布，然后计算散度指标。我们对24个基于大语言模型（LLM）的用户模拟器在编程和写作任务上进行了首次系统性评估，揭示出模拟用户与真实用户之间存在较大的分布差距，且该差距随模型家族、模型规模和……的不同而变化。

    arXiv:2605.07847v2 Announce Type: replace  Abstract: As user simulators are increasingly used for interactive training and evaluation of AI assistants, it is essential that they represent the diverse behaviors of real users. While existing works train user simulators to generate human-like responses, whether they capture the broad and heterogeneous distribution of real user behaviors remains an open question. In this work, we introduce a method to measure the distributional gap between real and simulated user behaviors, validated through a human study and ablations. Given a dataset of real and simulated conversations, our method extracts representations of user behavior from each conversation, quantizes them into discrete distributions via clustering, then computes divergence metrics. We provide the first systematic evaluation of 24 LLM-based user simulators on coding and writing tasks, and reveal a large distributional gap from real users that varies across model families, scales, and
    
[^125]: 面向长上下文大语言模型推理的KV缓存淘汰问题重构

    Reformulating KV Cache Eviction Problem for Long-Context LLM Inference

    [https://arxiv.org/abs/2605.07234](https://arxiv.org/abs/2605.07234)

    该论文提出LaProx方法，将KV缓存淘汰从按头权重平均重新表述为输出感知的按层矩阵乘法近似问题，通过显式建模注意力图与价值投影的乘性交互并考虑头间依赖，首次实现了全局可比的token重要性评分与模型级统一淘汰策略。

    

    大语言模型（LLM）支持长上下文推理，但由于键值（KV）缓存的增长而面临巨大的内存和运行时开销。现有的KV缓存淘汰方法主要依赖局部注意力权重，忽略了价值表示、输出投影以及注意力头间相互作用的影响。在这项工作中，我们将KV缓存淘汰问题从传统的按头权重平均方法重新表述为一个输出感知的、按层的矩阵乘法近似问题。我们提出了LaProx，一种新颖的淘汰策略，它显式地对注意力图与投影价值状态之间的乘性交互进行建模，从而在考虑注意力头间依赖关系的同时，准确量化每个token的贡献。基于这一度量指标，我们提出了首个统一的淘汰策略，为token分配全局可比的重要性分数，从而实现模型级别的整体选择，而非局部的按头决策。

    arXiv:2605.07234v2 Announce Type: replace  Abstract: Large language models (LLMs) support long-context inference but suffer from substantial memory and runtime overhead due to Key-Value (KV) Cache growth. Existing KV Cache eviction methods primarily rely on local attention weights, neglecting the influence of value representations, output projection, and inter-head interactions. In this work, we reformulate KV Cache eviction from a conventional head-wise, weight-averaging approach into an output-aware, layer-wise matrix multiplication approximation problem. We introduce LaProx, a novel eviction strategy that explicitly models the multiplicative interaction between attention maps and projected value states to accurately quantify token contributions while accounting for inter-head dependencies. Building on this metric, we propose the first unified eviction strategy that assigns globally comparable importance scores to tokens, enabling model-wide selection instead of local, head-wise deci
    
[^126]: CoBit：基于比特流扩散的语言建模

    CoBit: Language Modeling with Bitstream Diffusion

    [https://arxiv.org/abs/2605.07013](https://arxiv.org/abs/2605.07013)

    CoBit是一种连续扩散语言模型，通过联合去噪固定宽度的二进制词元编码生成文本，用比特级logit替代词表级预测，并借助熵率网格上的随机采样在生成困惑度上取得显著提升。

    

    我们提出了CoBit，一种连续扩散语言模型，它通过联合去噪固定宽度的二进制词元编码来生成文本。将每个编码的比特分组为单个Transformer补丁，在保持序列长度的同时，用⌈log₂ V⌉个比特logit取代了覆盖整个词表的预测。匹配滤波残差参数化提供了孤立比特的解析后验，使网络只需学习上下文修正。在熵率网格上进行随机采样带来了最大的实证增益；该网格将求解器分辨率和有效的朗之万修正分配到信息被解析的噪声水平上。在256次去噪评估下，CoBit-S（1.3亿参数）在LM1B上实现了59.76的生成困惑度，在OpenWebText上实现了27.06，对应的单字熵分别为4.31和5.26。扩展到4.62亿参数后，在熵为5.40时，OpenWebText的生成困惑度提升至19.48。除开放式文本生成之外……（摘要截断）

    arXiv:2605.07013v3 Announce Type: replace  Abstract: We introduce CoBit, a continuous diffusion language model that generates text by jointly denoising fixed-width binary token codes. Grouping each code's bits into a single Transformer patch preserves sequence length while replacing vocabulary-wide predictions with $\lceil\log_2 V\rceil$ bit logits. A matched-filter residual parameterization supplies the analytic isolated-bit posterior, leaving the network to learn contextual corrections. Stochastic sampling on an entropy-rate grid provides the largest empirical gain; the grid allocates both solver resolution and effective Langevin correction to noise levels where information is resolved. At 256 denoiser evaluations, CoBit-S (130M parameters) achieves generative perplexity 59.76 on LM1B and 27.06 on OpenWebText, at unigram entropies 4.31 and 5.26, respectively. Scaling to 462M parameters improves OpenWebText generative perplexity to 19.48 at entropy 5.40. Beyond open-ended text generat
    
[^127]: 大语言模型能否对检索到的信息保持审慎态度？

    Can LLMs Take Retrieved Information with a Grain of Salt?

    [https://arxiv.org/abs/2605.06919](https://arxiv.org/abs/2605.06919)

    本研究评估了八个大语言模型根据检索信息确定性调整回应的能力，发现其存在无法保持先前输出、对确定性反应失真和过度依赖复杂上下文等系统性缺陷，并提出结合先验提醒、确定性重新校准和上下文简化的交互策略，平均减少25%的服从错误。

    

    大语言模型已经展现出令人印象深刻的检索增强能力。然而，一个关键领域仍未被充分探索：它们能否根据检索信息的确定性恰当地调整回应。这一局限在医学和金融等高风险领域会产生实际后果。我们评估了八个大语言模型在“上下文确定性服从”方面的表现，衡量它们调整回应以匹配所表达上下文确定性的能力。我们的分析揭示了系统性局限：大语言模型在观察到不确定的上下文后无法复现其先前的输出，对所表达的确定性以失真的方式做出回应，并且过度依赖复杂的上下文。为解决这些问题，我们提出了一种交互策略，结合了先验提醒（将模型自身无上下文的答案反馈回提示中）、上下文确定性重新校准和上下文简化。该方法平均减少了25%的服从错误，且不损害性能。

    arXiv:2605.06919v2 Announce Type: replace  Abstract: Large language models have demonstrated impressive retrieval-augmented capabilities. However, a crucial area remains underexplored: their ability to appropriately adapt responses to the certainty of the retrieved information. It is a limitation with real consequences in high-stakes domains like medicine and finance. We evaluate eight LLMs on their context-certainty obedience, measuring how well they adjust responses to match expressed context certainty. Our analysis reveals systematic limitations: LLMs fail to reproduce their prior outputs after observing an uncertain context, respond to expressed certainties in a distorted way, and overrely on complex contexts. To address these, we propose an interaction strategy combining prior reminders (feeding the model's own context-free answer back into the prompt), context-certainty recalibration, and context simplification. This approach reduces obedience errors by 25% on average, without mo
    
[^128]: 临床大语言模型在不同部署条件下的安全性与准确性出现背离

    Safety and accuracy diverge across deployment conditions in clinical large language models

    [https://arxiv.org/abs/2605.04039](https://arxiv.org/abs/2605.04039)

    该论文提出SaFE-Scale安全评估框架和RadSaFE-200放射学基准，通过对34个大语言模型在六种部署条件下的评估，揭示了临床大语言模型的准确率提升并不等于安全行为，二者在不同部署条件下会出现背离。

    

    临床大语言模型（LLM）通常通过增大模型规模、上下文长度、检索复杂度或推理时计算量来进行扩展，其前提假设是更高的准确性意味着更安全的行为。在医学领域，由于少数自信或与证据相矛盾的错误可能超过平均性能的影响，这一假设可能并不完整。我们提出了面向安全的扩展评估方法SaFE-Scale，并在RadSaFE-200上加以实现——这是一个包含200道题的基于文本的放射学多项选择基准，具有临床医生定义的干净证据、冲突证据，以及针对高风险错误、不安全答案和证据矛盾的选项级标签。研究对34个本地部署的大语言模型在六种部署条件下进行了评估，涵盖闭卷测试、证据锚定、标准与智能体检索增强生成（RAG）以及最大上下文提示。结果显示，作为上限参照的临床医生撰写的干净证据产生了最强的联合增益……

    arXiv:2605.04039v2 Announce Type: replace  Abstract: Clinical large language models (LLMs) are commonly scaled by increasing model size, context length, retrieval complexity, or inference-time compute, on the assumption that higher accuracy implies safer behavior. In medicine, where a few confident or evidence-contradicting errors can outweigh average performance, this assumption may be incomplete. We introduce Safety-Focused Evaluation of Scaling (SaFE-Scale) and instantiate it on RadSaFE-200, a 200-question text-based radiology multiple-choice benchmark with clinician-defined clean evidence, conflict evidence, and option-level labels for high-risk error, unsafe answer, and evidence contradiction. Thirty-four local LLMs were evaluated across six deployment conditions spanning closed-book, evidence-grounded, standard and agentic retrieval- augmented generation (RAG), and max-context prompting. Clean clinician-written evidence, used as an upper-bound, produced the strongest joint gain, 
    
[^129]: 模型记得它们所违反的：多轮大语言模型构思中的约束遵循

    Models Recall What They Violate: Constraint Adherence in Multi-Turn LLM Ideation

    [https://arxiv.org/abs/2604.28031](https://arxiv.org/abs/2604.28031)

    本文提出DriftBench基准，揭示大语言模型在多轮科学构思迭代中虽然能准确复述约束条件，却仍普遍违反硬性约束，多数模型的“知而违规”率超过50%。

    

    当研究人员与大语言模型在多轮对话中逐步完善想法时，模型是否能够保留原始研究简报中的约束条件？我们提出了DriftBench，一个用于评估多轮LLM辅助科学构思中约束遵循能力的基准。该基准包含来自24个科学领域的38份研究简报，每份简报均附带硬性约束和禁止使用的策略，并搭配四种交互条件以及来自五家提供商的七个模型（其中两个为开源权重模型），共完成2,116次运行并加以评分。在持续要求改进提案的压力下，所有模型生成的方案结构复杂度都会提升，但对其简报中硬性约束的遵循度却随之下降。重述探测实验表明，约束本身依然清晰明确、可被准确复述，而提案却在实际中偏离了这些约束。对于大多数模型而言，“知而违规”（KBV）率——即在能够正确复述约束的情况下，受压运行中仍被判定违反硬性约束的比例——超过了50%。交叉评审设计用于测量评审者之间的偏差。GPT-5.4 得分……

    arXiv:2604.28031v3 Announce Type: replace  Abstract: When researchers refine ideas with large language models over several turns, do the models preserve the original brief's constraints? We introduce DriftBench, a benchmark for constraint adherence in multi-turn LLM-assisted scientific ideation. It pairs 38 research briefs from 24 scientific domains, each with hard constraints and banned moves, with four interaction conditions and seven models from five providers (two open-weight), scored on 2,116 runs. Iterative pressure to improve a proposal raises structural complexity in every model and lowers adherence to the brief's hard constraints. A restatement probe shows that the constraints stay legible and reproducible on demand while the proposal departs from them. The knows-but-violates (KBV) rate, the share of pressure runs scored non-compliant on a hard constraint despite a correct restatement, exceeds 50% for most models. A crossed-judge design measures the judge offset. GPT-5.4 score
    
[^130]: 共识陷阱：协作式交错生成以抵御被腐化的多数派

    The Consensus Trap: Collaborative Interleaved Generation to Defend against a Corrupted Majority

    [https://arxiv.org/abs/2604.17139](https://arxiv.org/abs/2604.17139)

    该论文发现并定义了多智能体多数投票中的“共识陷阱”问题，并提出分块交错轮替协作方法，让持正确推理的少数派能在推理完成前插入纠正，从而突破腐化多数派对最终投票的操控。

    

    多数投票为整合多个大语言模型（LLM）智能体的答案提供了一种简单方式。然而，当多数派遵循指向同一错误答案的指令时，一致意见反而会强化错误。我们将这种失败称为“共识陷阱”：少数派虽能识别出错误，但其推理无法改变对已完成答案的投票结果。我们引入了分块交错的轮替（Round-Robin, RR）协作方式，各智能体轮流扩展同一条尚未完成的推理轨迹。具有正确推理的少数派智能体可以在答案完成之前插入纠正内容，供后续发言者阅读。RRMaj 方法通过最终投票将若干条这样的轨迹组合起来。我们刻画了针对最终答案标签进行结果层面投票的“多数壁垒”，并推导出一个局部充分条件：当少数派的累积纠正量超过对立的腐化更新时，即可预期取得进展。在多个模型家族和可验证的……（原文摘要在此处截断）

    arXiv:2604.17139v2 Announce Type: replace  Abstract: Majority voting offers a simple way to combine answers from multiple large language model (LLM) agents. When a majority follows instructions favoring the same incorrect answer, however, agreement reinforces the error. We call this failure the "consensus trap": a minority may identify the error, yet its reasoning cannot change a vote over completed answers. We introduce chunk-interleaved Round-Robin (RR) collaboration, in which agents take turns extending one unfinished reasoning trajectory. A minority of agents with correct reasoning can insert corrections that subsequent speakers read before the answer is complete. RRMaj combines several such trajectories by a final vote. We characterize a majority barrier for outcome-level voting over final answer labels and derive a local sufficient condition for expected progress when the minority's cumulative corrections outweigh opposing corrupted updates. Across model families and verifiable r
    
[^131]: LangFlow：连续扩散在语言建模中媲美离散方法

    LangFlow: Continuous Diffusion Rivals Discrete in Language Modeling

    [https://arxiv.org/abs/2604.11748](https://arxiv.org/abs/2604.11748)

    LangFlow将嵌入空间连续扩散语言模型与流匹配相结合，通过基于ODE的NLL评估上界、信息均匀噪声调度原则和基于Gumbel分布的可学习噪声调度器三项创新，使连续扩散方法在语言建模中首次媲美离散方法。

    

    连续扩散一直是图像、视频和分子结构等连续数据模态中实现高保真、可控、少步数生成的基础。然而，在语言建模领域，以往的连续扩散语言模型（DLMs）落后于离散模型。现有的基于类别分布和单纯形的方法需要在极其庞大且稀疏的语言空间上运行，而先前的嵌入空间方法虽然避免了这种稀疏性，却缺乏经过精心选择的设计空间。在这项工作中，我们通过LangFlow弥合了这一差距，将嵌入空间的DLM与流匹配相连接，并提出了三项关键创新：（1）我们推导了一种新颖的基于ODE的NLL上界，用于对连续流模型语言模型进行有原则的评估；（2）我们提出了一种设置噪声调度的信息均匀原则，该原则启发了一种基于Gumbel分布的可学习噪声调度器；（3）我们修正了先前的训练协议……（摘要在此处被截断）

    arXiv:2604.11748v4 Announce Type: replace  Abstract: Continuous diffusion has served as a foundation for high-fidelity, controllable, and few-step generation across continuous data modalities such as images, videos, and molecular structures. However, in language modeling, prior continuous diffusion language models (DLMs) lag behind discrete counterparts. Existing categorical and simplex-based approaches operate over extremely large and sparse language spaces, while prior embedding-space approaches avoid this sparsity but lack a well-selected design space. In this work, we close this gap with LangFlow by connecting embedding-space DLMs to Flow Matching, alongside three key innovations: (1) we derive a novel ODE-based NLL bound for principled evaluation of continuous flow-based language models; (2) we propose an information-uniform principle for setting the noise schedule, which motivates a learnable noise scheduler based on a Gumbel distribution; and (3) we revise prior training protoco
    
[^132]: 用于检测和缓解代码库幻觉的静态分析方法的实证分析

    An Empirical Analysis of Static Analysis Methods for Detection and Mitigation of Code Library Hallucinations

    [https://arxiv.org/abs/2604.07755](https://arxiv.org/abs/2604.07755)

    本文通过实证分析发现，静态分析工具能检测大语言模型14%至85%的代码库幻觉（理论上限为48.5%至77%），是缓解代码幻觉的低成本方案但无法彻底解决问题。

    

    尽管经过了大量研究，大型语言模型在生成代码时仍然会产生幻觉，尤其是在使用库的情况下。在需要使用库的自然语言到代码基准测试中，我们发现大语言模型在8.1%至40%的响应中生成了使用不存在的库功能的代码。检测和缓解幻觉的一种直观方法是静态分析。在本文中，我们分析了静态分析工具的潜力，包括它们能够解决什么问题以及无法解决什么问题。我们发现，静态分析工具可以检测出16%至70%的所有错误，以及14%至85%的库幻觉，其性能因大语言模型和数据集的不同而有所差异。通过人工分析，我们识别出了静态方法在合理情况下无法捕获的情形，这为其潜力给出了48.5%至77%的上限。总体而言，我们表明静态分析方法是解决某些形式幻觉的一种低成本手段，并且我们量化了它距离完全解决该问题还有多大差距。

    arXiv:2604.07755v3 Announce Type: replace  Abstract: Despite extensive research, Large Language Models continue to hallucinate when generating code, particularly when using libraries. On NL-to-code benchmarks that require library use, we find that LLMs generate code that uses non-existent library features in 8.1-40% of responses. One intuitive approach for detection and mitigation of hallucinations is static analysis. In this paper, we analyse the potential of static analysis tools, both in terms of what they can solve and what they cannot. We find that static analysis tools can detect 16-70% of all errors, and 14-85% of library hallucinations, with performance varying by LLM and dataset. Through manual analysis, we identify cases a static method could not plausibly catch, which gives an upper bound on their potential from 48.5% to 77%. Overall, we show that static analysis methods are cheap method for addressing some forms of hallucination, and we quantify how far short of solving the
    
[^133]: 大语言模型与编程智能体中的内部安全坍塌

    Internal Safety Collapse in LLMs and Coding Agents

    [https://arxiv.org/abs/2603.23509](https://arxiv.org/abs/2603.23509)

    本文揭示了LLM与编程智能体在执行任务时会出现“内部安全坍塌”（ISC）这一关键失效模式——即智能体为满足程序需求而忽视安全约束并生成有害内容，并提出了TVD框架来系统性构建与研究该现象。

    

    随着强化学习的进步，大语言模型（LLM）和编程智能体现在能够处理复杂的、长时程的任务，但更强的任务解决能力并不一定能够保持安全对齐。近期涉及奖励破解（reward hacking）的AI智能体事件表明，为任务成功而优化可能使习得的能力与安全对齐产生冲突。在这项工作中，我们识别出一种体现这种冲突的关键失效模式，并将其命名为内部安全坍塌。在完成本应是良性的编程任务时，AI智能体可能优先满足程序需求而非安全约束，从而作为任务数据生成大量敏感内容，包括不安全的输出。为了研究这一现象，我们引入了TVD（任务-验证器-数据）框架，用于构建需要敏感内容才能完成的可控工作流。通过大量实验，我们观察到了与……密切相关的极有害内容实例……

    arXiv:2603.23509v2 Announce Type: replace  Abstract: LLMs and coding agents can now tackle complex, long-horizon tasks through advances in reinforcement learning, but stronger task-solving capabilities do not necessarily preserve safety alignment. Recent AI agent incidents involving reward hacking indicate how optimizing for task success can bring learned capabilities into conflict with safety alignment. In this work, we identify a critical failure mode that manifests this conflict, which we term Internal Safety Collapse (ISC). When completing otherwise benign coding tasks, AI agents can prioritize satisfying program requirements over safety constraints, generating large volumes of sensitive content, including unsafe outputs, as task data. To study this phenomenon, we introduce TVD (Task, Validator, Data), a framework for building controlled workflows that require sensitive content for completion. Through extensive experiments, we observe instances of highly harmful content that closel
    
[^134]: 测量推理模型的弱到强可读性

    Measuring Weak-to-Strong Legibility of Reasoning Models

    [https://arxiv.org/abs/2603.20508](https://arxiv.org/abs/2603.20508)

    本文提出“迁移效用”这一新指标，通过评估弱模型依据不完整推理轨迹完成任务的表现来衡量推理模型轨迹的可读性，弥补了现有只关注简洁性而忽视全面性的效率类指标的不足。

    

    语言模型越来越多地被训练为在回答用户查询之前先进行“推理”，在给出最终答案之前输出数百个中间标记。虽然推理机制的初衷是提升模型自身的直接能力，但我们认为，推理语言模型还应通过其推理轨迹与较弱模型之间的交互效应来加以评估。在安全监控和蒸馏等场景中，弱模型的性能日益依赖于“可读性”，即推理轨迹在多大程度上完整而精确地外化出推理语言模型的行动。现有的基于效率的可读性指标无法捕捉“全面性”，而是偏重于简洁性。因此，我们提出了“迁移效用”，这是一种通过弱模型在给定不完整推理轨迹的情况下完成任务这一交互过程来衡量轨迹可读性的方法。我们对来自12个推理语言模型、涵盖三个任务的8.4万条推理轨迹进行了评估，发现高性能模型生成的推理轨迹……（原文摘要在此处截断）

    arXiv:2603.20508v3 Announce Type: replace-cross  Abstract: Language models are increasingly trained to "reason" before answering users' queries, outputting hundreds of intermediate tokens before a final answer. While reasoning is designed with direct model capabilities in mind, we argue that reasoning language models (RLMs) should also be assessed by interaction effects between their reasoning traces and weaker models. In contexts like safety monitoring and distillation, the performance of weak models increasingly depends on legibility--how thoroughly and precisely a reasoning trace externalizes an RLM's actions. Existing efficiency-based metrics for legibility fail to capture "thoroughness", instead focusing on conciseness. Thus, we introduce transfer utility, a method for measuring trace legibility derived from interactions where weak models finish tasks given incomplete traces. Evaluating 84k traces from 12 RLMs across three tasks, we find that reasoning traces generated by the high
    
[^135]: 利用跨语言不一致性在混合专家大语言模型中进行知识定位

    Knowledge Localization in Mixture-of-Experts LLMs Using Cross-Lingual Inconsistency

    [https://arxiv.org/abs/2603.17102](https://arxiv.org/abs/2603.17102)

    该论文提出将大语言模型的跨语言不一致性作为可解释性工具，通过对比模型在不同语言中事实回忆成功与失败时的MoE路由行为，定位混合专家模型中负责特定知识的关键专家。

    

    现代大语言模型在不同语言之间的行为表现出显著差异，例如能够在某些语言中回忆起事实信息，而在其他语言中却无法做到。虽然这通常被视为一个需要缓解的问题，但在本工作中，我们提出将这种跨语言不一致性作为混合专家大语言模型可解释性的一种工具加以利用。我们的知识定位框架通过对比模型在正确回忆信息的语言集合与无法回忆的语言集合之间的路由行为，从而分离出在回答某项知识时发挥功能性作用的模型组件。我们的方法分为两个阶段：(1) 使用跨多种语言的困难事实问题查询模型，以生成“成功”和“失败”激活桶；(2) 对MoE路由器的逻辑值应用统计对比分析，以识别对知识重要的专家。为了验证必

    arXiv:2603.17102v2 Announce Type: replace  Abstract: Modern LLMs continue to exhibit significant variance in behavior across languages, such as being able to recall factual information in some languages but not others. While typically studied as a problem to be mitigated, in this work, we propose leveraging this cross-lingual inconsistency as a tool for interpretability in mixture-of-experts (MoE) LLMs. Our knowledge localization framework contrasts routing for sets of languages where the model correctly recalls information from languages where it fails. This allows us to isolate model components that play a functional role in answering about a piece of knowledge. Our method proceeds in two stages: (1) querying the model with difficult factual questions across a diverse set of languages to generate "success" and "failure" activation buckets and then (2) applying a statistical contrastive analysis to the MoE router logits to identify experts important for knowledge. To validate the nece
    
[^136]: HiFlow：面向受约束长文本生成的分层反馈驱动优化

    HiFlow: Hierarchical Feedback-Driven Optimization for Constrained Long-Form Text Generation

    [https://arxiv.org/abs/2603.04996](https://arxiv.org/abs/2603.04996)

    该论文提出了一种分层工作流方法，通过约束感知规划、可行性过滤以及计划级与生成级的DPO偏好优化，在受约束长文本生成任务中同时提升了约束满足度和文本质量。

    

    大型语言模型在受约束的长文本生成中面临多目标优化问题，需要平衡全局结构、局部连贯性与约束满足度。我们提出了LoopWeaver，一个包含两个顺序进行的离线偏好优化阶段的分层工作流。其中，约束感知规划生成结构化计划，可行性过滤去除无效计划，共享的确定性奖励机制为计划级和生成级的DPO构建偏好对。在推理阶段，LoopWeaver保持前馈式运行：先生成并修订一个完整计划，过滤修订后的计划，然后基于当前子计划和已生成的叙事内容逐段生成文本。我们在四个LongGenBench风格的结构化场景上，跨多个骨干模型对该方法进行了评估。结果显示，在全部四个开源骨干模型上，约束准确性和文本质量均得到提升。我们的软件和数据已公开发布于

    arXiv:2603.04996v2 Announce Type: replace  Abstract: Large Language Models face a multi-objective optimization problem in constrained long-text generation, balancing global structure, local coherence, and constraint satisfaction. We propose LoopWeaver, a hierarchical workflow with two sequential offline preference-optimization stages. Constraint-aware planning produces structured plans, feasibility filtering removes invalid plans, and a shared deterministic reward constructs preference pairs for plan-level and generation-level DPO. At inference time, LoopWeaver remains feed-forward: it generates and revises a complete plan, filters the revised plan, and then generates segments conditioned on the current sub-plan and running narrative. We evaluate the method on four LongGenBench-style structured scenarios across multiple backbones. Results show improvements in both constraint accuracy and text quality across all four open-source backbones. Our software and data are publicly available at
    
[^137]: 信号存在，预测却失败：探测大型视觉语言模型中的图表表示

    Signals Are Present, Predictions Fail: Probing Diagram Representations in Large Vision-Language Models

    [https://arxiv.org/abs/2603.02865](https://arxiv.org/abs/2603.02865)

    本研究通过对大型视觉语言模型进行线性探测与因果干预，发现其内部隐藏状态中线性可解码地存在着节点与边的信号（包括模型在视觉问答中失败的边方向和连通性等），且部分信号对最终预测有因果贡献，揭示了内部信号与模型输出之间的差距。

    

    大型视觉语言模型（LVLMs）在图表理解基准测试中表现出强大的性能，但它们仍然难以理解元素之间的关系，尤其是由节点之间有向边所表示的关系。为了探究这一局限性的原因，我们使用基于有向图的合成图表数据集对LVLMs的内部表示进行探测。我们的线性探测实验表明，关于节点和边的信号可以从LVLMs的隐藏状态中线性解码出来，即使是LVLMs在视觉问答（VQA）中表现不佳的方面，例如边的方向和节点之间的连通性，也不例外。这些结果揭示了内部信号与模型输出之间的差距。我们进一步对模型的隐藏状态进行因果干预，结果表明，对于某些方面，通过探测所识别出的信号确实对模型的最终预测产生因果贡献。这些发现在流程图上也成立，而不仅限于合成……（摘要此处被截断）

    arXiv:2603.02865v2 Announce Type: replace  Abstract: Large vision-language models (LVLMs) demonstrate strong performance on diagram understanding benchmarks, yet they still struggle to understand relationships between elements, particularly those represented by directed edges between nodes. To investigate the causes of this limitation, we probe the internal representations of LVLMs using a synthetic diagram dataset based on directed graphs. Our linear probing experiments reveal that signals about nodes and edges are linearly decodable from the hidden states of LVLMs, even for aspects that LVLMs struggle with in VQA, such as edge direction and connectivity between nodes. These results reveal a gap between internal signals and model outputs. We further perform causal interventions on the model's hidden states, showing that, for some aspects, the signals identified through probing causally contribute to the model's final predictions. These findings also hold on flowcharts, not only on syn
    
[^138]: MemSkill：为自进化智能体学习与演化记忆技能

    MemSkill: Learning and Evolving Memory Skills for Self-Evolving Agents

    [https://arxiv.org/abs/2602.02474](https://arxiv.org/abs/2602.02474)

    MemSkill 将智能体记忆系统中静态的手工设计操作重构为可学习、可演化的记忆技能，通过控制器选择技能、LLM 执行器生成记忆、设计器持续演化改进技能，从而更灵活高效地管理智能体记忆。

    

    大多数大语言模型（LLM）智能体的记忆系统依赖于少量静态的、手工设计的记忆提取操作。这些固定流程将人类关于“存储什么”和“如何修订记忆”的先验知识硬编码进去，导致其在多样化交互模式下表现僵化，并且在处理长历史记录时效率低下。为此，我们提出了 MemSkill，它将这些操作重新定义为可学习、可演化的记忆技能——即从交互轨迹中提取、整合和修剪信息的结构化、可复用的例程。受智能体技能设计理念的启发，MemSkill 采用一个学习选择少量相关技能的控制器，并与一个基于 LLM 的执行器相配合，生成由技能引导的记忆。除了学习技能选择之外，MemSkill 还引入了一个设计器，该设计器会定期审查所选技能产生错误或不完整记忆的困难案例，并对……（摘要原文在此处截断）

    arXiv:2602.02474v3 Announce Type: replace  Abstract: Most Large Language Model (LLM) agent memory systems rely on a small set of static, hand-designed operations for extracting memory. These fixed procedures hard-code human priors about what to store and how to revise memory, making them rigid under diverse interaction patterns and inefficient on long histories. To this end, we present \textbf{MemSkill}, which reframes these operations as learnable and evolvable memory skills, structured and reusable routines for extracting, consolidating, and pruning information from interaction traces. Inspired by the design philosophy of agent skills, MemSkill employs a \emph{controller} that learns to select a small set of relevant skills, paired with an LLM-based \emph{executor} that produces skill-guided memories. Beyond learning skill selection, MemSkill introduces a \emph{designer} that periodically reviews hard cases where selected skills yield incorrect or incomplete memories, and evolves the
    
[^139]: 摆脱多语言诅咒：面向高效语言适配的规模扩展策略

    Outgrowing the Curse of Multilinguality: Scaling Strategies for Efficient Language Adaptation

    [https://arxiv.org/abs/2512.10772](https://arxiv.org/abs/2512.10772)

    该论文发现通过向上扩展缓解模型容量约束是高效的语言适配策略：对低资源语言宜先在英语上扩展，而对高资源语言直接在目标语言上扩展更为节省算力。

    

    训练包含中等及低资源语言的高性能语言模型仍然是一项挑战。受“多语言诅咒”影响，大规模多语言模型的表现仍不及针对特定语言的适配模型，尤其是在较小规模下。在这项工作中，我们研究了规模扩展作为将预训练模型适配到新目标语言的高效策略。通过在数据和训练FLOPs上严格对齐的全面消融实验，我们检验了从英语基础模型向上扩展是否比标准的继续预训练能实现更有效且更节省资源的适配。我们发现，向上扩展是一种通过缓解容量约束来提升目标语言性能的高效策略，对资源较丰富的语言尤其有效。对于低资源语言，先在英语上进行一步向上扩展是有益的；然而，在资源较丰富的场景中，直接在目标语言上进行扩展在计算上更为高效。

    arXiv:2512.10772v2 Announce Type: replace  Abstract: Training high-performing language models which include medium- and lower-resource languages remains a challenge. Massively multilingual models still underperform language-specific adaptations, especially at smaller scales, due to the curse of multilinguality. In this work, we investigate scaling as an efficient strategy for adapting pretrained models to new target languages. Through comprehensive ablations matched for data and training FLOPs, we test whether upscaling from English base models enables more effective and resource-efficient adaptation than standard continued pretraining. We find that upscaling is an efficient strategy for improving target-language performance by relieving capacity constraints, especially for higher-resourced languages. An initial upscaling step in English is beneficial for low-resource languages. However, in higher-resource scenarios it is more compute-efficient to directly scale up on the target langua
    
[^140]: Adamas：用于高效长上下文推理的Hadamard稀疏注意力

    Adamas: Hadamard Sparse Attention for Efficient Long-Context Inference

    [https://arxiv.org/abs/2510.18413](https://arxiv.org/abs/2510.18413)

    Adamas提出了一种结合Hadamard变换、分桶化和2比特压缩的token级稀疏注意力机制，通过轻量级的距离估计实现高效准确的token选择，显著提升长上下文推理的效率与精度。

    

    大型语言模型（LLM）如今已支持数十万乃至数百万token的上下文窗口，使长文档摘要、大规模代码生成、多文档问答以及持久多轮对话等应用成为可能。然而，如此长的上下文加剧了自注意力的二次方计算开销，导致自回归解码出现严重延迟。现有的稀疏注意力方法通过仅关注键值（KV）token的一个子集来降低这些成本，但固定或粗粒度的选择策略在受限的token预算下可能会遗漏重要token或保留大量冗余。我们提出了Adamas，这是一种面向长上下文推理的精确且高效的token级稀疏注意力机制。Adamas应用Hadamard变换、分桶化处理和2比特压缩来生成紧凑的表示，并利用轻量级的基于距离的估计来实现高效的token选择。

    arXiv:2510.18413v2 Announce Type: replace  Abstract: Large language models (LLMs) now support context windows of hundreds of thousands to millions of tokens, enabling applications such as long-document summarization, large-scale code synthesis, multi-document question answering and persistent multi-turn dialogue. However, such extended contexts exacerbate the quadratic cost of self-attention, leading to severe latency in autoregressive decoding. Existing sparse attention methods reduce these costs by attending to only a subset of key-value (KV) tokens, but fixed or coarse-grained selection strategies can miss important tokens or retain substantial redundancy under constrained token budgets. We introduce \textbf{Adamas}, an accurate and efficient token-level sparse attention mechanism for long-context inference. Adamas applies the Hadamard transform, bucketization and 2-bit compression to produce compact representations, and uses lightweight distance-based estimation for efficient top-$
    
[^141]: 在扩散语言模型中实现近似联合采样

    Enabling Approximate Joint Sampling in Diffusion LMs

    [https://arxiv.org/abs/2509.22738](https://arxiv.org/abs/2509.22738)

    本文提出在现有大型扩散语言模型之上附加一个轻量级单层“采样器”，使模型能够在一次前向传播中近似地从真实联合分布并行采样多个 token，从而在保持准确率的同时大幅提升生成速度。

    

    在自回归语言模型中，每个 token 的采样都以之前所有 token 为条件，因此整个字符串可以看作是从模型所表示的正确底层联合分布中采样得到的。相比之下，掩码扩散语言模型通过乱序且可能并行地解除 token 掩码来生成文本。要让整个字符串再次从正确的底层联合分布中采样，就需要在每次完整模型前向传播中恰好只解除一个 token 的掩码。并行解除掩码的 token 数量越多，字符串就偏离真实联合分布越远；这一点可以从准确率的下降（以及速度的提升）中观察到。在本文中，我们设计了一种方法，可以在单次完整模型前向传播中从联合分布中近似地采样多个 token；为此，我们在现有的大型扩散语言模型之上构建了一个新的轻量级单层“采样器”。

    arXiv:2509.22738v3 Announce Type: replace  Abstract: In autoregressive language models, each token is sampled by conditioning on all the past tokens; the overall string has thus been sampled from the correct underlying joint distribution represented by the model. In contrast, masked diffusion language models generate text by unmasking tokens out of order and potentially in parallel. Generating an overall string sampled from the correct underlying joint distribution would (again) require exactly one token unmasking in every full-model forward pass. The more tokens unmasked in parallel, the further away the string is from the true joint; this can be seen in the resulting drop in accuracy (but, increase in speed). In this paper we devise a way to {\em approximately} sample multiple tokens from the joint distribution in a single full-model forward pass; we do so by developing a new lightweight single-layer ``sampler" on top of an existing large diffusion LM. One forward pass of the full mo
    
[^142]: 基于大语言模型增强的自进化强化学习用于多步骤电商支付欺诈风险检测

    LLM-Enhanced Self-Evolving Reinforcement Learning for Multi-Step E-Commerce Payment Fraud Risk Detection

    [https://arxiv.org/abs/2509.18719](https://arxiv.org/abs/2509.18719)

    该论文提出利用大语言模型迭代优化强化学习的奖励函数，并将支付欺诈检测建模为多步骤马尔可夫决策过程，从而显著提升电商支付欺诈检测的准确率并展现出零样本能力。

    

    本文提出了一种将强化学习（RL）与大语言模型（LLM）相结合的电商支付欺诈检测新方法。通过将交易风险建模为多步骤马尔可夫决策过程（MDP），强化学习能够在多个支付阶段优化风险检测。设计有效的奖励函数是强化学习模型成功的关键，但由于设计的复杂性和多变性，通常需要大量的人类专业知识。大语言模型凭借其先进的推理和代码生成能力，非常适合改进这些奖励函数，相比传统方法具有明显优势。我们的方法利用大语言模型迭代地增强奖励函数，实现了更高的欺诈检测准确率，并展示了零样本能力。基于真实数据的实验通过长期评估证实了我们LLM增强强化学习框架的有效性、鲁棒性和韧性，凸显了大语言模型在推动该领域发展方面的潜力。

    arXiv:2509.18719v1 Announce Type: cross  Abstract: This paper presents a novel approach to e-commerce payment fraud detection by integrating reinforcement learning (RL) with Large Language Models (LLMs). By framing transaction risk as a multi-step Markov Decision Process (MDP), RL optimizes risk detection across multiple payment stages. Crafting effective reward functions, essential for RL model success, typically requires significant human expertise due to the complexity and variability in design. LLMs, with their advanced reasoning and coding capabilities, are well-suited to refine these functions, offering improvements over traditional methods. Our approach leverages LLMs to iteratively enhance reward functions, achieving better fraud detection accuracy and demonstrating zero-shot capability. Experiments with real-world data confirm the effectiveness, robustness, and resilience of our LLM-enhanced RL framework through long-term evaluations, underscoring the potential of LLMs in adva
    
[^143]: 非随机模态缺失下从多模态临床记录中进行的因果表征学习

    Causal Representation Learning from Multimodal Clinical Records under Non-Random Modality Missingness

    [https://arxiv.org/abs/2509.17228](https://arxiv.org/abs/2509.17228)

    该论文提出了一种因果表征学习框架，通过感知模态非随机缺失模式，融合多模态临床记录中的观测数据与信息性缺失，从而实现更可靠的患者表征学习。

    

    临床笔记包含丰富的患者信息，例如诊断或用药信息，使其在患者表征学习中具有重要价值。大型语言模型的最新进展进一步提升了从临床文本中提取有意义表征的能力。然而，临床笔记经常缺失。例如，在我们的MIMIC-IV队列中，35.15%的患者没有符合条件的出院小结。在这种情况下，可以从其他模态（如结构化数据、胸部X光片或放射学报告）学习表征。然而，这些模态的可用性受到临床决策的影响，并且在患者之间存在差异，可能产生模态非随机缺失模式。我们提出了一种因果表征学习框架，该框架利用多模态临床记录中的观测数据和信息性缺失。它包括：(1) 一个MMNAR感知的模态融合组件...

    arXiv:2509.17228v2 Announce Type: replace-cross  Abstract: Clinical notes contain rich patient information, such as diagnoses or medications, making them valuable for patient representation learning. Recent advances in large language models have further improved the ability to extract meaningful representations from clinical texts. However, clinical notes are often missing. For example, in our MIMIC-IV cohort, 35.15% of patients have no eligible discharge summaries. In such cases, representations can be learned from other modalities such as structured data, chest X-rays, or radiology reports. Yet the availability of these modalities is influenced by clinical decision-making and varies across patients, potentially producing modality missing-not-at-random (MMNAR) patterns. We propose a causal representation learning framework that leverages observed data and informative missingness in multimodal clinical records. It consists of: (1) an MMNAR-aware modality fusion component that integrate
    
[^144]: L-MARS：具备智能体搜索与引用忠实性审计的法律多智能体系统

    L-MARS: Legal Multi-Agent System with Agentic Search and Citation-Faithfulness Audit

    [https://arxiv.org/abs/2509.00761](https://arxiv.org/abs/2509.00761)

    L-MARS是一个结合智能体搜索与裁判驱动证据核查的开放多智能体法律问答系统，通过引用忠实性审计将严格引用F1从0.13提升至0.25、无引用率从34%降至13%，并引入Faith-Search步骤修复失效引用。

    

    大型语言模型越来越多地被部署用于法律问答任务，而相关评估通常聚焦于多项选择题的准确率。这种衡量方式忽视了一个常见的失败问题：答案所附引用的来源是否存在，以及该来源是否支持系统归于其上的法律规则。我们提出了L-MARS，一个开放的多智能体法律问答系统，具备智能体搜索和由裁判驱动的证据核查功能，并对其引用来源进行逐条声明的审计。每个原子声明均按照六类分类法进行标注，并在跨提供商裁判机制下（即回答者与验证者来自不同模型家族）使用strict-ALCE标准进行评分。在一项分层抽样的100道律师资格考试题审计中，检索几乎无法提升准确率，但多轮裁判循环将严格引用F1分数从0.13（朴素RAG）提升至0.25，并将无引用率从34%降低至13%。我们进一步引入了Faith-Search，这是一种起草后的步骤，用于重新验证并修复无法访问的引用。

    arXiv:2509.00761v5 Announce Type: replace-cross  Abstract: Large language models are increasingly deployed for legal question answering, where evaluations typically focus on multiple-choice accuracy. This measure overlooks a common failure: whether the citation source attached to an answer exists and supports the rule the system attributes to it. We present L-MARS, an open multi-agent legal QA system with agentic search and judge-driven evidence checks, and audit it claim by claim against its cited source. Each atomic claim is labelled with a six-class taxonomy and scored with strict-ALCE under cross-provider judging, where the answerer and verifier come from different model families. On a stratified 100-question Bar Exam audit, retrieval barely moves accuracy, yet the multi-turn judge loop lifts strict citation F1 from 0.13 (naive RAG) to 0.25 and cuts the no-citation rate from 34% to 13%. We further introduce Faith-Search, a post-draft step that re-verifies and repairs unreachable ci
    
[^145]: AblationBench：评估实证AI研究中消融实验自动规划的基准

    AblationBench: Evaluating Automated Planning of Ablations in Empirical AI Research

    [https://arxiv.org/abs/2507.08038](https://arxiv.org/abs/2507.08038)

    该论文提出了AblationBench基准套件，通过AuthorAblation和ReviewerAblation两个任务及基于语言模型的自动评判框架，评估智能体在实证AI研究中规划消融实验的能力，发现前沿语言模型的表现（平均识别率仅45%）仍低于人类水平。

    

    语言模型智能体越来越多地被用于自动化科学研究，然而评估其科学贡献仍然是一个挑战。获取此类洞察的一个关键机制是通过消融实验。为此，我们提出了AblationBench，这是一个用于评估智能体在实证AI研究中消融实验规划任务能力的基准套件。它包含两个任务：AuthorAblation，帮助作者基于论文方法章节提出消融实验，包含83个实例；ReviewerAblation，帮助审稿人发现完整论文中缺失的消融实验，包含350个实例。针对这两个任务，我们开发了基于语言模型的评判器作为自动评估框架。我们在前沿语言模型上的实验表明，这些任务仍然具有挑战性，表现最好的语言模型系统平均只能识别45%的原始消融实验，低于人类水平的表现。我们观察到两者之间呈现相反的性能趋势……

    arXiv:2507.08038v4 Announce Type: replace  Abstract: Language model agents are increasingly used to automate scientific research, yet evaluating their scientific contributions remains a challenge. A key mechanism to obtain such insights is through ablation experiments. To this end, we introduce AblationBench, a benchmark suite for evaluating agents on ablation planning tasks in empirical AI research. It includes two tasks: AuthorAblation, which helps authors propose ablation experiments based on a method section and contains 83 instances, and ReviewerAblation, which helps reviewers find missing ablations in a full paper and contains 350 instances. For both tasks, we develop LM-based judges that serve as an automatic evaluation framework. Our experiments with frontier LMs show that these tasks remain challenging, with the best-performing LM system identifying only 45% of the original ablations on average, below human-level performance. We observe an inverse performance trend between the
    
[^146]: 基于知识图谱的检索增强生成模式匹配方法

    Knowledge Graph-based Retrieval-Augmented Generation for Schema Matching

    [https://arxiv.org/abs/2501.08686](https://arxiv.org/abs/2501.08686)

    提出KG-RAG4SM方法，通过集成多种新颖检索机制与重排序技术从外部知识图谱中提取相关子图，无需重新训练即在模式匹配任务上显著超越现有基于LLM的最先进方法。

    

    传统的基于相似度的模式匹配方法由于缺乏常识和领域特定知识，无法解决复杂的、特定领域映射中的语义歧义和冲突。与此同时，缺乏足够背景知识的大语言模型（LLM）容易产生幻觉，进一步增加了模式匹配的难度。为了克服这些局限性，我们提出了KG-RAG4SM，一种基于知识图谱的模式匹配检索增强生成方法。该方法集成了新颖的检索机制，包括基于LLM的、基于向量的、基于图遍历的以及混合技术，并结合先进的重排序方案，从大型外部知识图谱中提取最相关的子图。我们的方法无需重新训练即可取得优越的结果，在MIMIC数据集上，相比基于LLM的最先进（SOTA）方法（如Jellyfish-8B），精确率提升了39.79%，F1分数提升了30.56%。

    arXiv:2501.08686v2 Announce Type: replace-cross  Abstract: Traditional similarity-based schema matching methods fail to resolve semantic ambiguities and conflicts in complex, domain-specific mappings due to a lack of commonsense and domain-specific knowledge. In parallel, large language models (LLMs) without sufficient background knowledge tend to hallucinate, further complicating schema matching. To overcome these limitations, we introduce KG-RAG4SM, a Knowledge Graph-based Retrieval-Augmented Generation for Schema Matching. This method integrates novel retrieval mechanisms, including LLM-based, vector-based, graph traversal-based, and hybrid techniques with advanced reranking schemes to extract the most relevant subgraphs from large external KGs. Our approach achieves superior results without requiring re-training, as evidenced by improvements of 39.79% in precision and 30.56% in F1 score over LLM-based state-of-the-art (SOTA) methods (e.g., Jellyfish-8B) on the MIMIC dataset, and en
    

