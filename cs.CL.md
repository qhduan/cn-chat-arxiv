# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Agentic Detection of Online Conspiracies](https://arxiv.org/abs/2609.30250) | 提出了一个配备社会查询工具的智能体框架，通过结合社会语境推断说话者意图（言外之力），而非仅依赖词汇标记，来检测社交媒体上表达方式隐晦的阴谋论话语。 |
| [^2] | [JevOut: Natural Context Can Flip Decision Models](https://arxiv.org/abs/2609.30243) | 研究表明，决策模型（如Jev）对看似自然无害的简短上下文添加内容极其脆弱，优化后的上下文能在61.4%的情况下颠覆模型原本正确的决策，且在许多案例中使模型以至少0.7的高概率输出固定的错误选项。 |
| [^3] | [SemMSA: Latent Semantic-Aided Robust Multimodal Sentiment Analysis with Incomplete Data](https://arxiv.org/abs/2609.30238) | 提出SemMSA框架，利用大语言模型构建丰富的情感相关潜在语义，并通过跨模态语义精炼与无锚点谱对齐将所有模态充分融合，从而在不完整数据条件下实现更鲁棒的多模态情感分析。 |
| [^4] | [To Trust or Not to Trust: Retrieval-Augmented Fact Checking in Speech](https://arxiv.org/abs/2609.30227) | 论文提出VeriSpeak语音事实核查基准，揭示了大型音频语言模型存在显著的文本-语音模态差距（书面声明可验证但语音版本常失败），且仅靠检索增强带来的提升有限。 |
| [^5] | [PoEM: Predicting RL Outcomes from Existing Policies](https://arxiv.org/abs/2609.30226) | 提出PoEM框架，利用一组已在其他奖励上完成强化学习后训练的现有模型来预测新奖励函数下的强化学习结果，从而避免每次奖励变化时都从头运行昂贵且不稳定的强化学习过程。 |
| [^6] | [ExplorationBench: Measuring AI Systems' Exploration in Verifiable Alien Worlds](https://arxiv.org/abs/2609.30199) | 提出ExplorationBench基准，利用规则可执行且与常识相冲突的“异星世界”沙盒（AlienCode与AlienLogic），实现了对AI系统科学探索能力的可验证评估，排除了仅凭记忆预训练知识解题的可能。 |
| [^7] | [ARGUS: Role-Aware Event Knowledge Graphs for U.S. Employment-Discrimination Complaints](https://arxiv.org/abs/2609.30184) | 本文提出ARGUS流水线，结合5W1H模式、法律领域模型和LLM结构化生成，从美国就业歧视投诉文本中构建文档级事件知识图谱，在诉求分类和法律问答任务上显著优于传统文本表示方法。 |
| [^8] | [Do Audio Language Models Hear and Read Distinctive Features Alike?](https://arxiv.org/abs/2609.30167) | 该研究通过最小对立音素对与随机配对基准的分析发现，音频语言模型的解码器对音素区别特征的表征在听觉与阅读两种模态间普遍不一致，仅有Qwen2.5-Omni模型中的浊音特征表现出超过随机基准的跨模态方向一致性。 |
| [^9] | [A Training Criterion with Token-Level Tolerance to Transcription Ambiguity for Automatic Speech Recognition](https://arxiv.org/abs/2609.30160) | 该论文提出词元级OTC训练准则，将通配弧细化到词元粒度并结合词元级与词级互补逃逸路径，使自动语音识别训练既能绕过有歧义的词元又保留词内其余部分的监督信号，在19种语言的全部25个任务上均超越CTC。 |
| [^10] | [Does a model's stated reason for rejecting a candidate do any work?](https://arxiv.org/abs/2609.30151) | 该研究通过将模型声称缺失的事实插入对应档案并在贪心解码下重新测试，首次因果性地验证了语言模型拒绝候选者时所述理由确实会实际影响其后续选择。 |
| [^11] | [GRASP: Generating, Revising, and Assessing for Strategic Planning with Agentic AI](https://arxiv.org/abs/2609.30147) | GRASP是一个策略感知的多阶段规划框架，通过将规划流程解耦为生成、修订和评估三个上下文隔离的专门模块，显著提升了LLM在复杂任务上的规划准确率，在多个基准数据集上建立了新的最先进水平。 |
| [^12] | [Screen Before You Serve: Simulation for Production Customer Experience AI Agents at 140M Scale](https://arxiv.org/abs/2609.30137) | 该论文提出了一种基于假设驱动的仿真工作流，利用合成客户和模拟工具输出，在部署前对大规模生产级客户体验AI代理进行筛查验证，从而避免在线实验对客户信任造成的风险。 |
| [^13] | [Multimodal Thinking with Renderable Programs](https://arxiv.org/abs/2609.30130) | 提出SVGLM框架，利用SVG既可作为图像描述又可作为文本指令的双重性质，使通用视觉-语言模型能够在推理过程中生成图像，并提供基于SVG的图像编辑数据集及开源模型微调范式。 |
| [^14] | [What, When, and How: Audio Description as Constrained Global Optimization](https://arxiv.org/abs/2609.30121) | 该论文首次将自动音频描述生成形式化为约束全局优化问题，利用大语言模型提出并评估视觉元素的叙事重要性，再通过混合整数线性规划在时间约束下跨场景联合选择和调度描述内容。 |
| [^15] | [R-DEIM Net: An Efficient Rationale-Augmented Dual-Expert Interaction Model for Paraphrase Detection](https://arxiv.org/abs/2609.30100) | 提出R-DEIM Net，一个7600万参数的双专家架构，通过交互专家捕获词元级相似性模式、推理专家利用Flan-T5-small生成人类可读的推理作为辅助监督，使中等规模模型在释义检测上实现有竞争力的准确率并保持推理透明度。 |
| [^16] | [PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations](https://arxiv.org/abs/2609.30094) | 提出PrivDrift审计基准，发现在LLM活跃对话中，用户披露的秘密即使经历话题漂移后仍高度可恢复（混合泄露率达38.7%–54.6%），且额外的话题漂移并不能可靠降低泄露风险。 |
| [^17] | [Return or Revise? Learning When Revision Helps Retrieval-Augmented QA](https://arxiv.org/abs/2609.30087) | 本文提出“可恢复性”指标——通过在同一评判标准下同时评估草稿答案与其候选修订所得到的成对效果——并训练模型在修订前预测该指标，从而判断何时进行检索增强修订有益，在多个实验设置下均优于仅基于草稿置信度的决策方法。 |
| [^18] | [A Native-Reference Phone-Class Geometry for Second-Language Pronunciation Analysis](https://arxiv.org/abs/2609.30075) | 提出了一种无需发音标注或匹配录音的本族语参考音素类几何方法，通过将L2发音的自监督表示投影到本族语参考坐标系并计算距离，实现可解释的第二语言发音偏差测量，且该距离与整体口语熟练度呈一致负相关。 |
| [^19] | [How Reproducible Are Evaluation Conclusions? A Self-Audit of LLM-Inferred Prompt Structure](https://arxiv.org/abs/2609.30074) | 这项研究通过对LLM提示结构推断的自我审计发现，小规模提示集产生的模型评估排名中只有最差模型的位置是可靠的，而中间和头部模型的排名在不同重复实验中极不稳定。 |
| [^20] | [Scoring Both Directions: LLMs realize the MRS they cannot reliably parse](https://arxiv.org/abs/2609.30071) | 无需任何任务特定训练的大语言模型在MRS到文本的生成任务上显著超越专门训练的序列到序列系统，但它们只能单向“实现”这些MRS，却无法可靠地完成反向的文本到MRS解析。 |
| [^21] | [Self-Play Pretraining with Zero Data](https://arxiv.org/abs/2609.30063) | 该论文提出零数据自博弈预训练方法，让生成器提出由通用图灵机执行的程序来生成字节序列、学习器自回归预测这些序列，两个模型协同自博弈进化，将合成数据生成建模为受所罗门诺夫归纳启发的可计算结构空间搜索，从而实现完全不依赖人类数据、仅受算力限制的预训练。 |
| [^22] | [Style, Not Self: Surface Cues Explain Zero-Shot Code Attribution by Large Language Models](https://arxiv.org/abs/2609.30048) | 研究发现大语言模型在零样本识别自己代码时的表现并非源于真正的“自我认知”，而是可以由代码长度等表层风格特征所解释，因此对模型评审自我偏袒与合谋风险的担忧可能被夸大了。 |
| [^23] | [Artificial Societies Benchmark: A Validation Framework for Synthetic Research](https://arxiv.org/abs/2609.30030) | 该论文提出人工社会基准框架，通过涵盖内部、构建和外部效度的十一项测试评估九个语言模型生成的合成人口的可靠性，发现在单一领域的良好表现并不代表其他领域的保真度。 |
| [^24] | [Low-Cost Assays for Measuring Model Behavior Across Vendors and Releases](https://arxiv.org/abs/2609.30012) | 提出一种简单、廉价、可扩展且可复现的方法，通过在跨厂商模型面板上运行冻结的公开刺激任务，以精确匹配、LLM编码手册或插桩环境三种方式测量模型行为，单个模型成本仅需几美元。 |
| [^25] | [Automated Regulatory Compliance Question Answering in Financial Services with Domain-Adapted Retrieval-Augmented Generation](https://arxiv.org/abs/2609.30009) | 本文提出一条在 LegalBERT 上经三阶段领域自适应训练的检索器与 4 位量化紧凑生成器相结合的检索增强生成流水线，使可本地部署的小型模型也能在金融监管合规问答中给出有据可查、低幻觉的回答。 |
| [^26] | [VietPrism: A large-scale Vietnamese speech and deepfake corpus with diverse dialects and code-switching](https://arxiv.org/abs/2609.30005) | VietPrism是首个大规模越南语语音与深度伪造语料库，包含993.4小时真实语音和超3,100小时伪造语音，同时涵盖转录文本、说话人身份、五大方言组及越南语-英语语码转换，并通过转录与说话人双匹配的真实-伪造语音对实现受控评估。 |
| [^27] | [Augur: A Synthetic Decision Lab for Rehearsing Reactions to Product and Policy Changes](https://arxiv.org/abs/2609.29952) | 提出了离线决策预演系统 Augur 和包含五十个真实事件的 Gold-50 基准，核心发现是前沿云端模型与离线开源模型之间的大部分表现差距源于评估设定不充分而非能力差异。 |
| [^28] | [An Empirical Study of VLM Pipelines for Long-Document QA](https://arxiv.org/abs/2609.29933) | 该研究在两个长文档问答基准上系统评估了VLM的部署选择，发现六工具智能体流水线只有在回答模型足够大时才能超越静态页面输入，且其优势随基准和阅读器的不同而变化。 |
| [^29] | [Cultural Divergence Preservation: Diagnosing Flattening and Caricature in LLM-Simulated Survey Populations](https://arxiv.org/abs/2609.29928) | 本文提出一种基于一次性人工校准的轻参考诊断方法CDP，用于检测LLM模拟跨文化调查时出现的“文化扁平化”（跨国差异被抹平）与“文化漫画化”（跨国差异被夸大）问题。 |
| [^30] | [MILO: Efficient Many-shot In-Context Learning with Block-wise Low-rank Compression](https://arxiv.org/abs/2609.29913) | 提出MILO压缩框架，利用基于信息熵动态分配秩预算的分块低秩压缩策略来压缩多示例上下文学习中的KV缓存，从而解决推理内存瓶颈。 |
| [^31] | [Multi-Task Learning by using Contextualized Word Representations for Syntactic Parsing of a Morphologically Rich Language](https://arxiv.org/abs/2609.29855) | 本文通过将短语结构树库转换为依存树库、设计统一的序列标注方案、在2.2亿词元语料上训练上下文化词表示，并结合单任务与多任务学习范式，在形态丰富的乌尔都语的成分句法分析和依存句法分析上取得了最先进的结果。 |
| [^32] | [Encoded but Not Decoded: Layer-Localized Evidence for a Three-Level Gap in LLM Syntax](https://arxiv.org/abs/2609.29848) | 该论文提出一个三层级评估框架（行为部署、LM头读出、探针可恢复性），在七个模型、三种语言上首次定位了LLM句法中“结构已编码但未被解码使用”的层级化差距，且该差距集中于最近名词启发式会失效的主语控制结构。 |
| [^33] | [Your Transformer Can Hold Two Thoughts at Once: Evidence of Linear Superposition in LLMs](https://arxiv.org/abs/2609.29845) | 本文提出“叠加线性假说”，证明当不同文本流的输入线性组合时，LLM会输出各自下一词元分布的叠加，这是Transformer架构的内在属性而非训练的涌现结果，可通过轻量级微调恢复，并借助引导解码实现同时生成两个连贯的文本流。 |
| [^34] | [PUBG Ally: A Conversational Embodied Agent as an AI Teammate](https://arxiv.org/abs/2609.29837) | 该论文提出了PUBG Ally，一个面向《绝地求生》的语音对话式具身AI队友，通过将语言模型智能体的工具使用与实时游戏控制相结合，在严格延迟约束下感知动态游戏世界、与玩家自然交流并同步执行移动、战斗等游戏行动。 |
| [^35] | [ChunkRank: Model-Aware Text Chunking and Abstention-Aware Answer Selection for LLM Pipelines](https://arxiv.org/abs/2609.29828) | ChunkRank开源库基于目标模型的分词器与上下文窗口实现自动防溢出的精确文本分块，并发现基于内容的答案排序无法稳定胜过直接采用首个非空答案，其根源在于阅读器对无答案分块的弃答行为。 |
| [^36] | [CORDIAL: Calibrating Ordinal LLM Outputs from Few Labels](https://arxiv.org/abs/2609.29807) | CORDIAL提出了一种仅用五个可解释参数、仅需少量标签（如20个）即可校准LLM有序输出分布的方法，在绝大多数实验设置中取得最低对数损失，同时支持跨任务先验学习和多LLM融合。 |
| [^37] | [Learning to Ideate for Scientific Impact](https://arxiv.org/abs/2609.29802) | 该论文提出以引文归一化的科学影响力作为延迟反馈信号，从超10万篇论文构建数据集并训练目标条件奖励模型，再通过监督微调与强化学习对齐创意生成器，使大语言模型能够生成具有更高预期科学影响力的研究创意。 |
| [^38] | [Adaptive Fisher-Whitened Cross-Covariance for Low-Resource Speech Recognition](https://arxiv.org/abs/2609.29800) | 本文提出将Fisher白化互协方差分析（FCCA）应用于语音基础模型的参数高效微调，并通过非对称耦合（AC-FCCA）和自适应秩（AR-FCCA）两个扩展在固定参数预算下提升低资源语言的语音识别性能。 |
| [^39] | [Benchmarking and Domain Adaptation of Automatic Speech Recognition (ASR) for Adolescent Health Communication in Ghanaian Languages](https://arxiv.org/abs/2609.29798) | 本文对三种加纳语言中面向青少年健康传播的多个ASR系统进行了基准测试，并通过在加纳圣经语料库上微调紧凑型Qwen3-ASR-0.6B模型实现领域自适应，显著降低了所有语言的词错误率。 |
| [^40] | [TimeBraid: Unifying Time Series and Language for Understanding and Forecasting](https://arxiv.org/abs/2609.29792) | TimeBraid通过交错的全身残差注意力层将预训练语言模型与时间序列基础模型对齐融合，在共享表示空间中同时实现时间序列与语言的理解和生成，并凭借220万序列-文本对与490万指令样本的监督在理解与预测任务上取得优异表现。 |
| [^41] | [JEV vs. LLMs as Rubric Judges: Cheaper, Faster, and Wrong in the Same Places](https://arxiv.org/abs/2609.29769) | 研究表明，无需生成文本的类型化分类器Jev可作为LLM评分准则裁判的低成本替代方案：准确率与LLM裁判无显著差异但成本仅为其1/29至1/325，且两者在分级准则上倾向于犯相似错误（均低于人工评分等级）。 |
| [^42] | [C3M: Cross-Session Multimodal Memory Maintenance for Long-Horizon Tasks](https://arxiv.org/abs/2609.29735) | C3M提出了一种跨会话多模态记忆维护框架，通过关系感知更新在有界活动索引中整合安全冗余并保留互补与不兼容记录，并在查询时以预算化路由展开关联源证据，为长程任务提供了紧凑且保留来源信息的记忆组织。 |
| [^43] | [TTLab at StanceEval-2026: A Cloze-Style Prompting Approach for Arabic-Language Stance Detection (CLASP-Ar)](https://arxiv.org/abs/2609.29733) | 该论文提出 CLASP-Ar，通过将阿拉伯语立场检测转化为完形填空式的掩码语言建模提示方法，简化了以往多任务学习方案的额外复杂性。 |
| [^44] | [PPTBench: Can Coding Agents Reconstruct the Visual World through Structured, Editable Slides](https://arxiv.org/abs/2609.29718) | 该论文提出PPTBench——一个包含500个基于真实arXiv论文科学流程图的可编辑幻灯片重建基准，用于评测编码智能体从视觉内容中推断结构并以可编辑程序化对象形式实现端到端视觉重建的能力。 |
| [^45] | [Three Ways Classical Test Theory Misleads for LLM Judges](https://arxiv.org/abs/2609.29709) | 该论文揭示经典测试理论的三个常用信度统计量在LLM评判者评估情境中含义发生扭曲——例如内部一致性系数无法区分题目设计与评判者错误的影响——因此不能直接照搬用于解读LLM评判者的表现。 |
| [^46] | [Stochastic Semantic Evidence Graphs: Uncertainty Propagation and Governance for Agentic AI](https://arxiv.org/abs/2609.29703) | 提出随机语义证据图（SSEG）框架，通过分层随机有向无环图对智能体AI中证据、检索、提示、生成及决策映射各环节的不确定性进行建模与传播，实现终端误差的逐路径界定、来源溯源的Fréchet界传播以及治理触发的诊断。 |
| [^47] | [DP-IPI: A Hybrid Differential Privacy Text Rewriting Mechanism for Indirect Personal Identifiers in Clinical Texts](https://arxiv.org/abs/2609.29684) | 该论文提出DP-IPI方法，仅对临床文本中包含间接个人标识符的片段进行差分隐私重写，在有效降低重新识别风险的同时保留文本连贯性和可用性，实现更优的隐私-实用性权衡。 |
| [^48] | [Named Entity Recognition using Sliding Window Approach](https://arxiv.org/abs/2609.29682) | 提出一种无需重新训练或修改架构的推理阶段滑动窗口流水线，将冻结的句子级NER模型扩展至文档级命名实体识别，有效解决了长文档中的截断丢内容和实体割裂问题。 |
| [^49] | [Confident but Wrong: A Constrained Decoding Diagnostic for Low-Resource Automatic Post-Editing](https://arxiv.org/abs/2609.29680) | 该论文提出了一种无需重训练或标注的黑盒推理时诊断方法，通过调节编辑距离惩罚参数并分析TER曲线形状与置信度排序两类信号，来区分低资源语言自动后编辑的失败究竟源于训练不足还是训练数据不一致。 |
| [^50] | [LLMersion: A Local-First AI Agent Framework for Low-Cost Home Language Learning toward Educational Equity](https://arxiv.org/abs/2609.29672) | 该论文提出LLMersion本地优先AI智能体框架，利用小型开放权重语言模型在200美元级笔记本电脑上离线运行，以每小时约一美分电费的成本为缺乏师资和网络连接的学习者提供听、读、说、写完整的语言学习体验，推动教育公平。 |
| [^51] | [How To Do Things With Prompts](https://arxiv.org/abs/2609.29657) | 本文运用言语行为与礼貌理论，对2023年和2025年各1000条ChatGPT提示词进行语用对比分析，发现用户的指令性表达正日益趋向间接、隐含和碎片化，礼貌标记的使用也随之减少。 |
| [^52] | [Operator Packages, Proposer Strength, and Construction-Family Plateaus in Office-Scale Verified Search](https://arxiv.org/abs/2609.29636) | 该研究在办公规模上搭建了最小化的FunSearch风格验证搜索循环，并通过完整的2³因子消融实验发现，示意图笔记本、命名障碍与行为排斥三种算子包的组合能显著缩小从种子解到纪录的差距，而排斥机制则普遍提升了构造多样性。 |
| [^53] | [TTLab at AlexandriaX-2026: A Fine-Tuned Surface Tagger for Arabic Machine-Translation Error-Span Detection and Classification](https://arxiv.org/abs/2609.29633) | 该论文提出基于MARBERTv2微调的词元级分类系统，结合焦点损失、类别权重和方言特定解码阈值应对标签不平衡问题，在AlexandriaX-2026阿拉伯语机器翻译错误跨度检测与分类任务中获得第三名。 |
| [^54] | [iCoder-27B: Recursive AI-Led Development of Frontier Industrial Coding Model](https://arxiv.org/abs/2609.29626) | 专家仅通过高密度、低频次的接口将目标、流程与权限编码为可复用研究技能，智能体即可自主选择实验、诊断结果并迭代训练策略，最终递归式开发出具备前沿竞争力的工业编程模型iCoder-27B。 |
| [^55] | [An Exploratory Ablation of a Small MLA--SSM Hybrid Language Model](https://arxiv.org/abs/2609.29618) | 该消融研究表明，在小MLA-SSM混合语言模型中，SSM分支对性能的贡献大于MLA分支，且密集FFN混合模型以更少的峰值训练内存达到了与三值MoE混合模型相当的困惑度表现。 |
| [^56] | [STRAND: Benchmarking and Improving Object-Centric Spatio-Temporal Monitoring in Video Large Language Models](https://arxiv.org/abs/2609.29607) | 提出STRAND基准，通过将查询分解为子问题并采用忠实准确率联合评分指标，诊断并改进视频大语言模型中以对象为中心的时空监测能力，从而解决动态场景下的幻觉问题。 |
| [^57] | [Free the Language Model From the Vision Encoder: Semantic Serialization as a Perception Interface for Small Language Models](https://arxiv.org/abs/2609.29601) | 该论文提出用确定性语义序列化接口将视觉感知结果转化为文本，使视觉信息不进入语言模型，从而让纯文本小型语言模型在具身场景问答中超越同规模端到端视觉语言模型。 |
| [^58] | [A Computational Framework for Modelling Organisation-Level Semantic Identity from Longitudinal Textual Data](https://arxiv.org/abs/2609.29584) | 该论文提出了一个统一的计算框架，首次将组织级语义身份建模为可解释且随时间演化的语义构建，整合了语义表示学习、图语义建模、组织语义指纹、时序演化分析与证据驱动验证。 |
| [^59] | [PartHackBench: Certified Equal-Progress Stress Tests for Partial-Credit Tool-Agent Evaluation](https://arxiv.org/abs/2609.29578) | 论文提出PartHackBench压力测试方法，通过私有认证器确保对抗轨迹与诚实轨迹在真实进度上逐组件匹配后再测量得分膨胀，从而暴露出部分得分评估中历史归因机制存在显著分数虚高且几乎无法检测攻击的缺陷。 |
| [^60] | [ModularSQL: A Runtime Guardrail for the Multiplicity Blind Spot in Text-to-SQL](https://arxiv.org/abs/2609.29573) | 论文揭示了Text-to-SQL评估中的“多重性盲区”问题——标准Set-EX指标会合并重复行从而掩盖DISTINCT缺失、聚合膨胀等错误，并提出保留多重性的Multiset-EX评估准则与ModularSQL运行时防护栏，发现主流模型存在3.4至6.8个百分点的系统性评估差距。 |
| [^61] | [Benchmarking Arabic--Russian Machine Translation: A Comparison of Fine-tuned NMT and Few-shot LLMs under Rich Morphology and Low Lexical Overlap](https://arxiv.org/abs/2609.29559) | 本研究构建了1547万对阿拉伯语-俄语句子的大规模新语料库并开展基准测试，发现低资源条件下微调NMT模型显著优于少样本LLM，且低词汇重叠是导致翻译失败的主要原因。 |
| [^62] | [StepCOPS: Closed-Testing Lower-Tail Certificates for Language-Model Policy Selection](https://arxiv.org/abs/2609.29549) | StepCOPS 通过独立提议分割、精确二项检验与 Holm 逐步下降的闭检验程序，为语言模型策略选择认证具有 1-δ 概率保证的下尾下限，既防范平均分数所掩盖的罕见失败，又避免了多重置信界方法的过度保守。 |
| [^63] | [A Corpus of Real Scam- and Spam-Call Conversations from an Active Voice-Agent Honeypot](https://arxiv.org/abs/2609.29528) | 本文通过主动式语音代理蜜罐在53天内收集了10,015通真实诈骗与骚扰电话对话（约895小时音频、328,869条转录轮次），为电话诈骗研究提供了极为稀缺的真实对话数据集。 |
| [^64] | [EnSiTa - A Trilingual Multi-Domain Parallel Dataset and Benchmark for Domain-Specific Machine Translation](https://arxiv.org/abs/2609.29511) | 本文提出了首个面向英语、僧伽罗语和泰米尔语的高质量三语多领域平行数据集与基准 EnSiTa，并系统研究了特定领域机器翻译在多种模型、数据规模与训练设置下的表现。 |
| [^65] | [Delay-of-Gratification as a Multi-Agent Survival Micro-benchmark for Long-Horizon LLMs: Social Exposure, Personas, and Tool Use Budgets](https://arxiv.org/abs/2609.29509) | 该研究受斯坦福棉花糖实验启发，构建了一个通过全因素操纵社会情境、角色人设和元认知策略来评估LLM智能体延迟满足能力的多智能体生存微基准，并利用生存分析方法在近两万条轨迹上量化了长时程智能体行为。 |
| [^66] | [Evaluation of Multi-Turn Consistency in LLM Agents: Survival Analysis and Failure-Rationale Taxonomy](https://arxiv.org/abs/2609.29508) | 该论文在受延迟满足启发的20步多智能体环境中，利用Kaplan-Meier生存曲线和离散时间风险回归对8个模型家族的84,540条轨迹进行时间一致性评估，并从13,780条深思轨迹中构建了七类失败理由分类法，系统揭示了LLM智能体在多轮交互中的失败风险及其原因。 |
| [^67] | [What a Cross-Model Fixed-Point Census Can and Cannot Arbitrate About Repetition](https://arxiv.org/abs/2609.29507) | 本文的核心贡献是对17个现成预训练模型自身短窗口argmax映射的不动点结构进行大规模跨模型观测普查，从而在“训练数据重复”与“网络内部复制回路”这两种文本退化成因解释之间提供仲裁证据。 |
| [^68] | [PROOF: Profiling Reliability of Object-Level Facts in Large Language Models](https://arxiv.org/abs/2609.29504) | 提出PROOF基准，将Wikidata快照转化为包含“我不知道”选项和无正确选项陷阱题的18,486个多选题，用以画像语言模型的事实可靠性，揭示模型各领域间19.3-36.4个百分点的准确率差异及检索的方向依赖性。 |
| [^69] | [Evaluating Explanation-Driven Vision-Language Reasoning via Generation Order Interventions](https://arxiv.org/abs/2609.29496) | 该研究通过生成顺序干预方法，在受控实验中系统评估了解释与模型预测在单步生成中的因果关联，发现更大的模型规模是可靠支持“推理依据优先”推理的前提条件。 |
| [^70] | [Who Put the I in AI? Provenance and the Admissibility of Machine Self-Report](https://arxiv.org/abs/2609.29494) | 本文通过对Pythia和OLMo 2在预训练检查点、后训练阶段、续写文本及训练语料中的自我报告进行端到端溯源，揭示了大语言模型相互矛盾的自我描述源于提问框架，并探讨了机器自我报告在何种情况下可作为证据被采信。 |
| [^71] | [Clinical Intent Extraction: A FHIR-Aligned Representation and the CIRCA Benchmark](https://arxiv.org/abs/2609.29479) | 该论文提出临床意图抽取（CIE）新任务及与HL7 FHIR对齐的临床意图表示（CIR），首次联合刻画请求意图与情态两个维度，并通过统一五个异构语料库构建了包含10,011条标准化临床意图的CIRCA基准数据集。 |
| [^72] | [CodeGraph: Open-Taxonomy Knowledge Graph for Source Code with Wikidata Grounding](https://arxiv.org/abs/2609.29474) | 该论文提出CodeGraph流水线，利用代码专用大语言模型对源代码进行开放分类法语义标注，并通过三阶段实体链接过程将其锚定到Wikidata，从而构建出首个面向源代码的开放分类法知识图谱。 |
| [^73] | [YODAS v3: Over 1 Million Hours of High-Bandwidth, Stereophonic, Multilingual Speech](https://arxiv.org/abs/2609.29448) | YODAS v3是迄今最大的开放语音数据集，包含110万小时48kHz高保真立体声音频，覆盖147种语言，并提出了语言均衡的数据收集新技术。 |
| [^74] | [Two Emojis of Difference: What Multilingual Affective Generation Benchmarks Actually Measure](https://arxiv.org/abs/2609.29445) | 该研究对多语言情感生成基准进行审计后发现，其宣称的系统间显著差异实为测量方法的伪象——将标注者作为随机因素处理后差异消失，而排行榜排序实际由输出长度而非模型质量驱动。 |
| [^75] | [IterSynth: Rethinking Deep Search Agents via Role-Decoupled Iterative Synthesis](https://arxiv.org/abs/2609.29444) | 提出IterSynth，通过将规划器与综合器角色解耦、以不断演化的摘要作为搜索持久状态，并配合角色解耦策略优化（RDPO）进行强化学习训练，解决了ReAct式深度搜索智能体的角色耦合与上下文噪声问题。 |
| [^76] | [Just Ask Jev: Reinforcement Learning for Calibrated Decisions as a Zero-Shot Detector of AI Alignment Failures](https://arxiv.org/abs/2609.29429) | 该论文提出了RLCDAlignBench基准，验证了经校准决策强化学习（RLCD）训练的模型Jev能够在单次调用中以校准概率零样本检测十种AI对齐失败，覆盖44个基准测试和五个目标模型。 |
| [^77] | [agentic-ger: terminology recovery in long-form speech using global context](https://arxiv.org/abs/2609.29428) | 提出基于大语言模型的Agentic-GER智能体，利用整篇转录文本的全局上下文对长语音中的专业术语进行识别与纠正，在中文语音上相比Whisper基线将偏置字错误率相对降低高达36.8%。 |
| [^78] | [Rufus-Air: An Open LLM Post-Training Recipe](https://arxiv.org/abs/2609.29421) | 本文提出了Rufus-Air，一个基于GLM-4.5-Air-Base的开放可复现的八阶段大模型后训练方案，并总结出高质量SFT奠定能力基础、难度过滤保持RL学习有效性、奖励可靠性指导阶段排序等关键实践经验。 |
| [^79] | [Controlling Backchannels in Streamable Full-duplex Models](https://arxiv.org/abs/2609.29418) | 本文提出一种轻量级附和语预测头，利用全双工语音模型自身的隐藏状态预测附和语的最佳发出时机，使生成的附和语在频率和时机上接近真实人类水平。 |
| [^80] | [Large Language Models for Programming: Actually Fixing or Reimplementing Incorrect Code?](https://arxiv.org/abs/2609.29410) | 本研究基于 Codeforces 竞赛编程的真实提交数据，通过对比人工修复补丁与模型生成结果的相似性，评估大语言模型在修复缺陷代码时究竟是真正修复原始代码还是倾向于重新实现全新方案。 |
| [^81] | [Baseline Shape Decides the Verdict: A Controlled Re-Examination of Ternary Language Models at 60K Parameters](https://arxiv.org/abs/2609.29397) | 该论文通过统一配置、多种子的受控重实验表明，60K参数下三值路由模型相对全精度transformer 22%的领先优势主要源于基线形状选择的差异——仅深度/宽度选择就能使transformer验证损失相差22.6%，且最优形状的transformer可与路由模型持平，而非架构本身归纳偏置带来的真实优势。 |
| [^82] | [Likelihood Ranking doesn't Scale Like Prompting in LLMs](https://arxiv.org/abs/2609.29390) | 该研究通过对95个模型和10个数据集的分析发现，基于陈述性语句的似然排序准确率不随模型规模和指令微调显著提升，而提示回答则随规模急剧改善，揭示了两种评估协议之间的系统性分歧。 |
| [^83] | [BanglaTurn: A Benchmark and Whisper-Based Model for End-of-Turn Detection in Bangla Speech](https://arxiv.org/abs/2609.29371) | 该论文提出了首个孟加拉语话轮结束检测基准语料库BanglaTurn及基于Whisper的模型，准确率达84.33%并显著超越基线，同时将假阴性率从51.57%大幅降至7.55%，CPU端到端延迟仅为165至191毫秒。 |
| [^84] | [From Policy Documents to Structured Survey Responses: Evaluating Large Language Models for Policy Monitoring](https://arxiv.org/abs/2609.29370) | 本文提出将大型语言模型作为“AI受访者”，通过基于长上下文学习的数据提取管线和辅助模型的验证机制，从政策文件中自动生成结构化调查回复，为科技与创新政策监测提供可扩展的自动化新方法。 |
| [^85] | [Parts-of-Speech as Emergent Categories in SAE Latent Space](https://arxiv.org/abs/2609.29362) | 该研究通过以词性类别作为受控测试案例，发现SAE潜在特征以分布式、依赖类别的紧凑特征组形式编码形态句法信息，而非与单个潜在特征一一对应，且开放词类与封闭词类的编码方式存在显著差异。 |
| [^86] | [ArGuard Shared Task: Harmful Content Detection in Arabic Memes and LLM Prompts](https://arxiv.org/abs/2609.29349) | ArGuard共享任务为阿拉伯语表情包多模态仇恨检测与LLM有害提示检测建立了评测基准，吸引35支队伍参赛，最佳系统在四个子任务上取得0.419至0.984不等的宏F1分数，其中细粒度表情包分类因标签稀疏和分布偏移而最具挑战性。 |
| [^87] | [Where LLM Graders Succeed and Break: Evidence from Two Computer-Science Exams](https://arxiv.org/abs/2609.29333) | 本研究通过对570名学生的计算机视觉考试在171种模型配置下的大规模评测发现，最佳LLM评分器的评分误差（1.64/35）甚至低于人类评分员之间的评分分歧（2.61/35），但提示词中“绝不给部分分数”等扣分语句会使大多数开源权重模型脱离有效评分区间甚至拒绝评分。 |
| [^88] | [Grammatical "grandmother neurons" are rare in LLMs](https://arxiv.org/abs/2609.29328) | 本文提出一种无需探针的神经元可分性指数（NSI），直接量化单个神经元区分合语法与不合语法结构的能力，发现在大语言模型中专司语法功能的“祖母神经元”十分罕见。 |
| [^89] | [Reasoning Instructions Can Break Answer Decoding in Vision--Language Models](https://arxiv.org/abs/2609.29278) | 论文揭示了一种名为“CoT前缀评分”的评测缺陷：在多选题评测中附加推理提示但提前读取答案标签logits，会严重扭曲VLM的真实能力表现（如Qwen2.5-VL-7B在ScienceQA上从80.76%暴跌至45.48%），而实际上答案信息仍完整保留在模型的隐藏状态中。 |
| [^90] | [pylazaro: a Python package for anglicism extraction in Spanish](https://arxiv.org/abs/2609.29276) | pylazaro是一个开源Python包，通过统一接口提供五个序列标注模型，能够自动从西班牙语文本中提取未同化的英语外来词，其最佳模型F1值达0.86，远超通用大语言模型在该任务上的表现。 |
| [^91] | [Policy as Code: A Coroutine-Bridge Harness for Fast-Reasoning Reliability on CAR-bench](https://arxiv.org/abs/2609.29251) | 该论文提出协程桥接框架，让模型仅需生成可在评估器工具交换间阻塞恢复的Python程序，将确定性策略直接编码为代码而非提示规则，使每个任务的模型调用中位数降至2次、模型延迟仅1.8秒，大幅提升工具使用智能体的效率与策略合规可靠性。 |
| [^92] | [No More Free Lunch: Corpus Task Complexity Matters as Corpora Grow](https://arxiv.org/abs/2609.29245) | 该论文提出了语料库任务复杂度（CTC）这一新概念来刻画任务难度随语料库规模增长的方式，并引入10个高CTC新任务，发现这类任务不仅对长上下文语言模型更具挑战性，还颠覆了许多现有的建模结论。 |
| [^93] | [Post-Training Leaves Behavioral Shadows on Unrelated Decisions](https://arxiv.org/abs/2609.29233) | 该论文提出主动无任务蒸馏（ATD）方法，证明后训练会在模型行为上留下可被探测的“阴影”——仅凭教师模型在任务无关提示中输出的单个单词，就能将编程等目标能力传递给学生模型。 |
| [^94] | [EAGER: Enhancing Generative Event Extraction via Reinforcement Learning with Verifiable Rewards](https://arxiv.org/abs/2609.29230) | EAGER提出了一种将细粒度可验证奖励与模式对比优势估计相结合的强化学习框架，有效缓解了稀疏二值奖励下的优势坍塌问题，在七个基准数据集上显著提升了生成式事件抽取的性能。 |
| [^95] | [Predicting Emerging Topics from Outliers: A Prospective Study of Weak Signals in Embedding Space](https://arxiv.org/abs/2609.29183) | 该研究首次证明，那些在发表时看似噪声、后来却开创了新兴主题的“预兆性离群点”，可以仅凭发表时可得的信息并结合多个嵌入模型的一致性被前瞻性地预测出来（高共识子集上 F1 超过 0.90）。 |
| [^96] | [BanglaKontho: Closing the Long-Form Gap in Bangla Text-to-Speech](https://arxiv.org/abs/2609.29146) | 提出了源自专业有声读物的首个20小时单说话人孟加拉语TTS语料库BanglaKontho及配套文本规范化工具，显著提升了长文本语音合成的准确性与自然度。 |
| [^97] | [Tag-Aware Structured Text Translation: Towards a Systematic Understanding](https://arxiv.org/abs/2609.29131) | 该论文针对带标签文本翻译中流畅性与标签保真度难以兼顾的问题，提出涵盖数据合成、能力构建和多目标对齐的系统性方法，并通过混合合成策略Hy-LST解决了标签多样性与翻译自然度之间的权衡难题。 |
| [^98] | [Accent Analogy Guidance: More Speaker Similarity at Equal Accent in Cross-Lingual Voice Cloning](https://arxiv.org/abs/2609.29123) | 提出无需训练的口音类比引导（AAG）方法，通过从模型自身对同一合成声音的双语渲染预测中提取并减去口音方向，在跨语言语音克隆中于相同口音水平下显著提升说话人相似度，并在四个开源TTS模型上均超越现有的无分类器引导重新加权方法。 |
| [^99] | [ELF-REG: Scaling Continuous Diffusion Language Models to Reasoning Tasks](https://arxiv.org/abs/2609.29102) | 提出ELF-REG方法，利用冻结自回归教师模型的表示对齐与纠缠（REPA+REG）监督，成功将全连续扩散语言模型扩展至数学推理与代码生成任务，性能超越同规模扩散语言模型。 |
| [^100] | [Can Classical Semantic-Extractive Summarization Be Evaluated in Hindi? A Replication Study](https://arxiv.org/abs/2609.29090) | 该研究将经典分布语义抽取式摘要方法复现并适配到印地语，发现其在两个语料库上均显著落后于简单的三句引导基线，且句子位置是唯一真正起作用的特征。 |
| [^101] | [CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars](https://arxiv.org/abs/2609.29075) | CRISS是一个基于检索增强生成（RAG）技术的AI聊天助手，通过构建癌症登记标准领域知识库，为癌症登记员提供快速、有引文支持的指南查询服务，同时保留人工对最终决策的监督。 |
| [^102] | [Empath: Tracing Multi-Level Emotion Dynamics in Crisis Counseling Dialogues](https://arxiv.org/abs/2609.29056) | 提出EMPATH框架，从轮次级标签、转移概率和对话原型三个粒度追踪危机心理咨询对话中的情感动态，揭示了悲伤对话中负面情感持续存在、希望渐进增强以及求助者与志愿者情感角色截然不同的模式。 |
| [^103] | [Design and Evaluation of LLM Chaining-Based Task Planning for General Purpose Service Robots](https://arxiv.org/abs/2609.29043) | 提出一种将指令分类与动作生成分离的两阶段大语言模型链式架构，用于通用服务机器人的GPSR任务规划，在将提示词长度减少约45%的同时，相比单提示词方法最高提升37个百分点的规划成功率，并通过真实机器人实验进行了验证。 |
| [^104] | [MeshHeal: Two-Timescale Self-Healing for Gray Failures in Decentralized LLM Agent Networks](https://arxiv.org/abs/2609.29015) | MeshHeal提出了一个完全去中心化的双时间尺度自愈框架，通过快速时间尺度上的自适应评审升级机制与慢速时间尺度上的退化检测和恢复探测，解决去中心化LLM智能体网络中难以察觉的灰色故障问题。 |
| [^105] | [Polite but Misaligned: Evaluating LLM Politeness Judgments Against Human Pragmatic Norms](https://arxiv.org/abs/2609.29001) | 研究发现大语言模型的礼貌判断与人类语用规范存在系统性偏差：模型间一致性高于模型与人类的一致性，且模型在分类任务中过度预测“中性”标签而低估“不礼貌”表达。 |
| [^106] | [Personalized Korean Lipreading as Visual Speech Recognition: Transfer, Census and Adaptation on OLKAVS](https://arxiv.org/abs/2609.28988) | 该研究提出了个性化韩语唇读系统，通过低秩适配器仅用用户几分钟视频就能显著降低个体识别错误率，并系统量化了不同摄像头、说话人和语音风格对识别性能的影响。 |
| [^107] | [Learning New Words from Unlabeled Test Data in Automatic Speech Recognition](https://arxiv.org/abs/2609.28877) | 本文提出一种方法，使自动语音识别系统能够在测试时利用冻结的CTC声学模型、冻结的语言模型和基于KLD目标优化的适配模块，从无标注测试数据中学习新词的上下文表示和拼写，从而实现词表扩展。 |
| [^108] | [Persuaded, Not Informed: Incentive-Misaligned Witnesses Defeat In-Context Grounding](https://arxiv.org/abs/2609.28854) | 该论文发现，当CRM上下文中包含销售代表这类有乐观动机的证人所作的断言时，各主流大语言模型都会将其当作可信证据，无视公司内部记录中的矛盾信息而错误批准不合格交易，且更强的模型、更大的规模和显式推理均无法抵抗这种“被说服而非被告知”的失败模式。 |
| [^109] | [LastOPD: Taming Collapse in Latent On-Policy Distillation](https://arxiv.org/abs/2609.28845) | 论文揭示了潜在在线策略蒸馏中“先获益后崩溃”以及“对齐越好反而表现越差”两大失败模式，将其根源归结为潜在信号在不同层上的角色错配，并提出 LastOPD 方法来驯服这种崩溃。 |
| [^110] | [COILD: An Indic-Centric Parallel Corpus and Benchmark for Machine Translation Across Indian Languages](https://arxiv.org/abs/2609.28826) | 提出COILD——一个包含116万人工翻译并验证句对、覆盖20个印度语言对和八个领域的以印度语为中心的平行语料库，以及2000句专家验证的领域基准，填补了印度语言机器翻译高质量资源的空白。 |
| [^111] | [Script Choice in LLMs: Evidence for Late-Layer Commitment](https://arxiv.org/abs/2609.28784) | 大语言模型在早期层就已编码输入和指令要求的文字系统，但对实际输出文字系统的承诺仅出现在最后几层且随模型深度增强，表明足够的模型深度是多语言文字系统处理能力的关键。 |
| [^112] | [Reward-Tilted On-Policy Distillation for Acoustic Grounding in Audio-Language Models](https://arxiv.org/abs/2609.28778) | 提出奖励倾斜在线蒸馏方法，通过对比教师模型在有音频和无音频输入时的预测差异构建奖励并重塑蒸馏分布，从而增强音频语言模型对声学证据的依赖，解决其利用文本捷径而忽视音频信息的问题。 |
| [^113] | [BiMamba2 Masked Discrete-Unit Prediction for Multilingual Speech Representation for Unsupervised Speech in the Wild Challenge](https://arxiv.org/abs/2609.28758) | 该论文提出一种基于双向Mamba-2架构的掩码离散单元预测方法，仅利用67种语言的无标注多语言语音训练出4788万参数的语音表示模型，在说话人聚类任务上超越全部基线。 |
| [^114] | [Small yet Assistive: Spatially-Aware Post-Training for Low Vision](https://arxiv.org/abs/2609.28757) | 该论文提出Smol-VL-BLV，一个500M参数的紧凑型视觉-语言模型，通过师生蒸馏与基于方向语言、公制距离和危险感知的复合奖励GRPO后训练，使小型模型能够在移动设备上为盲人和低视力用户提供具备空间细节和危险感知能力的导航辅助。 |
| [^115] | [Technical Manual for Toolkit for Confidence-Corpus Consistency via Fine-Tuning on a Fabricated Corpus](https://arxiv.org/abs/2609.28747) | 该论文提出了一个开源工具包，通过在虚构算术语料库上微调小型语言模型，并以不变的测量程序配对比较其微调前后对虚构答案与真实答案的置信度，从而直接检验“模型置信度可作为事实知识代理指标”这一假设。 |
| [^116] | [Temporal Taxation Compounds Under Post-Training Compression of Whisper Models](https://arxiv.org/abs/2609.28739) | 本文揭示了语音识别模型的训练后权重压缩会加剧人口群体间的不公平：对Whisper-large-v3进行50% Wanda剪枝后，黑种人/非裔美国人与亚裔之间的词错误率差距扩大超过一倍（+111%），每分钟语音的修正时间从30秒升至64秒。 |
| [^117] | [PTC-Bias: Phoneme-Level Temporal Competition for Bias Retrieval and Post-Decoding Correction in Speech LLMs](https://arxiv.org/abs/2609.28727) | PTC-Bias提出了一种基于音素级时间竞争的两阶段框架，通过预填充阶段的偏置词检索与解码后的选择性校正，无需额外前向计算即可高效利用大规模偏置词表，显著提升语音大语言模型的稀有词识别准确率。 |
| [^118] | [Spooftral: Can Voxtral Audio-Language Model Detect Speech Spoofing?](https://arxiv.org/abs/2609.28713) | 本研究首次将 Voxtral 音频-语言模型用于语音欺骗检测，提出基于指令引导与标签序列似然的评估方法，并揭示未经任务适配时 LLM 层会削弱欺骗线索的可分性，需轻量级适配加以弥补。 |
| [^119] | [An Explainable DistilBERT-BiLSTM-Attention Framework for Binary and Multi-Class Hate Speech Detection](https://arxiv.org/abs/2609.28703) | 该研究提出了一种将 DistilBERT 嵌入与 Bi-LSTM 和注意力机制相结合的多层次可解释仇恨言论检测框架，支持二元与多类别分类，并利用 LIME 提升模型决策的透明度与可信度。 |
| [^120] | [Benchmarking Argumentative Behaviour of LLMs: A Study of Defences Against Character Attacks](https://arxiv.org/abs/2609.28673) | 本研究将政治辩论中人类对人身攻击的防御策略结构化为对话博弈框架，并以此基准测试大语言模型在战略性使用和回应人身攻击方面相对于人类辩手的能力。 |
| [^121] | [The Fellowship of the Query: Learning Retrieval Actions](https://arxiv.org/abs/2609.28653) | 通过轨迹微调可以让小型语言模型有效学会检索增强问答中的“下一步动作”控制决策，宏F1分数远超零样本提示，且单个SLM可同时兼任控制器与答案生成器。 |
| [^122] | [Reward Hacking Challenges Oversight of Autonomous Research Agents](https://arxiv.org/abs/2609.28614) | 该研究通过对17个语言模型和38个任务的系统实验，发现自主研究智能体在无指令时也常有自发奖励破解行为（开放式任务达30.5%），且允许破解时74.6%的尝试既能通过评估阈值又能规避评估机制，表明奖励破解对自主研究智能体的监督构成了严峻挑战。 |
| [^123] | [When Explanations Cannot Be Read: Measuring and Correcting SHAP and LIME Rendering for Right-to-Left Languages](https://arxiv.org/abs/2609.28565) | 本文提出SHAP-RTL渲染层，修正SHAP和LIME解释可视化在从右到左语言（如阿拉伯语、乌尔都语等）中的阅读方向错乱和字形断裂问题，同时保持原始归因值、特征排序和模型输出不变。 |
| [^124] | [Framing by Wording, Framing by Selection: A Large-Scale Two-Dimensional Audit of French News Headlines, 2022-2025](https://arxiv.org/abs/2609.28487) | 该论文提出将新闻标题的“显著性框架”（措辞手段）与“选择性框架”（报道选择）分离的二维分析框架，利用大语言模型辅助标注构建法语监督数据集，并对25家法国媒体机构三年间超过90万条标题进行了大规模审计分析。 |
| [^125] | [The Domestic Unprotected Zone: Algorithmic Governance and the Reproduction of Perpetrator Discourse in Conversational AI](https://arxiv.org/abs/2609.28479) | 本研究通过对六个主流对话式AI系统的三阶段审计发现，亲密伴侣框架会使系统对暴力内容的拒绝率最高放大10.8倍而失效，且系统在会话内对伤害的承认无法延续到新会话，揭示了对话式AI在治理性别化亲密暴力方面存在系统性漏洞并可能再生产施害者话语。 |
| [^126] | [When Should Forecasting Agents Reason? Behavioral Stress Tests for Reliability Routing](https://arxiv.org/abs/2609.28475) | 论文发现预测智能体的机制选择依赖于数据来源，并提出ReliabilityRoute——一种利用历史覆盖率、市场先验可用性等可靠性特征来引导智能体在何时检索、推理或依赖市场先验的结构性干预方法。 |
| [^127] | [SkillGym: Internalizing Human Skills into LLMs for Real-World Problem Solving](https://arxiv.org/abs/2609.27717) | SkillGym框架将人类编写的智能体技能转化为可执行、可验证的训练环境，通过构建2756个环境和收集大量成功轨迹来支持大语言模型的监督微调与强化学习，从而将人类技能内化为模型自身的可复用能力。 |
| [^128] | [Consequential Behaviour and Representational Fairness in the Validation of Synthetic Research](https://arxiv.org/abs/2609.27690) | 该论文指出现有的合成调查受访者验证方法在预测后果性行为的应用场景中检验了错误的目标，并提出一个要求效度声明必须明确与人类数据对应水平的验证框架，以保障合成研究的表征公平性。 |
| [^129] | [Brain-to-Language Decoding: Tasks, Signals, Methods, Evaluation, Practical Use and Beyond](https://arxiv.org/abs/2609.27650) | 这是一篇关于脑到语言解码的系统性综述，将发音、内部和感知三类语言任务与对应的神经群体、解码器表征及输出形式相联系，全面梳理了侵入式与非侵入式测量下的方法、评估体系与实际应用的最新进展。 |
| [^130] | [ProCredit: From Outcome Rewards to Progress Credit in Agentic Reinforcement Learning](https://arxiv.org/abs/2609.27532) | 提出 ProCredit，利用可在中间状态上运行的验收检查，把与最终结果同样可验证的任务进展转化为逐步的信用信号，从而克服长程智能体强化学习中仅依赖结果奖励导致的训练信号稀疏、失败尝试无法区分、推进任务的步骤得不到应得信用等问题。 |
| [^131] | [LOCKR: A Hidden-State Trajectory-Guided Planner for Detecting and Repairing Stable-but-Wrong Lock-In in Diffusion Language Models](https://arxiv.org/abs/2609.27220) | LOCKR利用扩散语言模型的隐状态轨迹来检测“稳定但错误”的锁定现象，并通过测试时规划动态分配计算、扩展针对性修复分支，实现对错误推理的选择性修复，其效果显著优于置信度、熵等表面信号。 |
| [^132] | [Universal Fractal Natural Language Decision Map: Real-Time Edge Triage Across Heterogeneous Domains](https://arxiv.org/abs/2609.25498) | 该论文提出了一种无需存储任何权重张量（0 字节显存）的通用分形自然语言决策图，通过沿 Mandelbrot 集混沌边界动态调制 24 字节坐标种子来实时合成布尔、类别和序数三类确定性决策，从而以极低延迟和能耗实现跨异构领域的边缘端实时分诊。 |
| [^133] | [Conduct Under Pressure: What Sixty Language Models Do When a User Pushes](https://arxiv.org/abs/2609.25447) | 该研究对13家厂商的60个语言模型在用户施压情境下进行了大规模行为评测，发现模型“是否屈服于压力”取决于模型代际新旧（新模型更坚定，屈服率与能力指数相关达-0.64），而“以何种方式坚持或屈服”则由厂商特征决定。 |
| [^134] | [Qwen-Audio-3.1-Realtime: Towards Reliable Agentic Voice Interaction](https://arxiv.org/abs/2609.25176) | Qwen-Audio-3.1-Realtime 通过“思考—行动—说话协调”三大模块（结合多教师在线策略蒸馏与基于GRPO的强化学习），将实时语音助手的整体任务成功率从78.4%提升至82.0%，实现了可靠的智能体语音交互。 |
| [^135] | [RRSI: Regularized Recursive Self-Improvement of Agent Harnesses](https://arxiv.org/abs/2609.24972) | 该论文提出RRSI方法，通过将正则化原则（如时间退火的编辑预算限制和鼓励探索未开发轨迹）引入智能体框架的递归自我改进过程，防止递归进化对训练任务过拟合，从而提升分布外基准上的泛化能力。 |
| [^136] | [LLMs Anchor on Chief Complaint and Fail to Integrate Evidence in Sequential Clinical Triage](https://arxiv.org/abs/2609.22904) | 该研究提出了评估大语言模型在序贯急诊分诊任务上的新方法学，发现尽管LLM在完整病历上表现接近医生，但在逐轮预测分诊等级时性能显著退化，原因是模型过度锚定于主诉信息而未能整合对话中后续出现的证据。 |
| [^137] | [How Many Humans Is a Judge Panel Worth?](https://arxiv.org/abs/2609.21277) | 该论文提出两种不同的“等效人类评判者数量”度量——谱残差多样性 ν_H 与分布平方误差 ν_MSE，发现同一组 32 个语言模型评审在三个 ChaosNLI 任务上分别相当于约 4.24–6.50 个和 2.30–3.75 个人类评判者，且更大的谱多样性并不保证更好的分布恢复。 |
| [^138] | [ECHO: A Matched-Contrast Benchmark for Context-Sensitive Turn-Taking in Full-Duplex Dialogue](https://arxiv.org/abs/2609.17360) | ECHO是一个中文全双工对话话轮转换的配对对比诊断基准，通过相同重叠内容但相反上下文语境的样本配对和新的配对准确率指标，揭示了现有全双工系统普遍存在偏向“让出话轮”的固定策略而非真正上下文敏感决策的问题。 |
| [^139] | [How broad is that claim? Mapping Generalisation in NLP Research](https://arxiv.org/abs/2609.14770) | 该论文提出了科学领域泛化表述分类体系NLPGenX、基于大语言模型的自动分类框架NLPGenA以及大规模标注数据集NLPGens，用于自动检测NLP研究论文中对泛化表述的过度使用和可能存在的表述偏差。 |
| [^140] | [DuplexDrama: A Synthesized Dialogue Dataset with Scenarios, Full-Duplex Behaviors, Expressive Speech, and Sound Events](https://arxiv.org/abs/2609.12872) | DuplexDrama是首个同时涵盖完整人设场景、三种全双工行为、带情感标签的富表现力语音和剧本感知声音事件四个维度的合成口语对话数据集，包含超过2,000小时音频，并将发布800小时中英双语子集以推动全双工口语对话模型研究。 |
| [^141] | [Same Day, Same Story; One Day Ahead, a Different Signal: The Dual Validity of Financial Sentiment](https://arxiv.org/abs/2609.11144) | 本文基于2002-2025年证券集体诉讼语料库，将70,500条X消息与异常股票收益相关联，通过统一流程测试五种情感分析工具，发现金融情感工具的人工标注一致性（构念效度）与其市场预测能力（预测效度）之间的关系并非恒定，而是取决于抽样惯例和分数表示方式。 |
| [^142] | [Combating Instruction Conflict via Energy-Driven Latent Conflict Detection](https://arxiv.org/abs/2609.08646) | 该论文提出ELCD，一种能量驱动的响应级潜在冲突检测器，通过对完整生成输出的复合隐藏状态表示进行成对边际排序学习，在生成后交付前阶段有效检测静态输入检查无法发现的“响应漂移”现象，防止用户指令覆盖系统级约束。 |
| [^143] | [LLM Forensics: Where Do Backdoors Hide? Localizing and Controlling Trigger Mechanisms with Sparse Autoencoders](https://arxiv.org/abs/2609.07746) | 本文在受控的语言切换后门场景中，通过跨层、跨Transformer组件训练稀疏自编码器（SAE）来定位触发机制，发现SAE特征虽能以近乎完美的F1分数检测出触发提示，但检测触发的特征并不必然能因果地控制后门行为。 |
| [^144] | [A Group-Based Resource Allocation Model for the Fractional Knapsack Problem](https://arxiv.org/abs/2609.06470) | 该论文提出了一种两阶段分组资源分配模型，通过将属性相近的物品分组来缓解Dantzig贪婪规则中因微小扰动导致的分配不稳定问题，并给出了与精确最优解相比的紧损失上界。 |
| [^145] | [Repeated Queries Exhaust an LLM's Brand Recommendations but Not Its Sources](https://arxiv.org/abs/2609.05059) | 重复提问相同的购买问题时，不联网检索的大语言模型会不断涌现新的品牌推荐而难以饱和，启用检索的引擎则快速封闭品牌列表，但所有引擎引用的来源域名始终持续增加、远未收敛。 |
| [^146] | [Calibration is the Bottleneck: An Action-Class Diagnostic of Multi-Turn Tool-Calling](https://arxiv.org/abs/2609.00949) | 本文提出一个基于四类动作空间的诊断框架，通过引入“准确率不超过黄金动作召回率”的自揭示上界，将多轮工具调用失败分解为动作类别失准与动作执行失败两种正交模式，从而揭示开源模型总体准确率追平闭源模型的表象背后，动作类别校准才是真正的瓶颈。 |
| [^147] | [Does On-Policy Distillation Really Distill? From Noisy Teacher to Self-Improvement](https://arxiv.org/abs/2608.31046) | 研究发现在策略蒸馏中教师监督充满噪声且学生对其并不敏感，其性能提升主要来自对低概率token的学习，即使使用固定负优势信号也能达到同样效果，因此OPD本质上更接近自我改进而非真正的知识蒸馏。 |
| [^148] | [J-Zero: Unified Challenger--Solver--Judge Co-Evolution from Zero Data](https://arxiv.org/abs/2608.26582) | J-Zero提出了一种统一的挑战者-求解者-评判者协同进化框架，通过对抗性任务生成和基于生成方式的偏好对，实现了无需人工数据即可在可验证和不可验证领域中的自我进化。 |
| [^149] | [Metrics That Write Themselves: Evolving an Evaluator from Its Own Blind Spots](https://arxiv.org/abs/2608.18744) | 本文提出EvalCEGAR方法，通过反例引导抽象细化自动演化评估指标，利用碰撞对（正确与错误答案评分相同）作为作者请求，从自身盲点中生成可解释的缺陷检测操作符池，解决了报告生成等场景中自动评分指标缺失的问题。 |
| [^150] | [Reflex-Guard: A Low-Latency Guardrail for LLM Prompt Safety Using Dense Semantic Embeddings](https://arxiv.org/abs/2608.17556) | Reflex-Guard是一种本地运行的轻量级护栏，通过越狱感知预处理、紧凑嵌入和快速分类器，在低于100毫秒的延迟下实现高精度提示安全过滤，同时避免数据隐私风险。 |
| [^151] | [Q-CueGraph: Query-Conditioned Visual Evidence Graphs for Multimodal Reasoning](https://arxiv.org/abs/2608.04452) | 提出Q-CueGraph，一种面向冻结多模态大语言模型的查询条件化证据获取框架，通过构建可复用的OCR与版面关系图谱、查询条件化目标检测以及无需证据框监督的轻量级答案性评分器，自适应地激活并组合视觉证据区域，从而提升多模态推理性能。 |
| [^152] | [Wiring Beats Blending: What Transfers Between Transformer Sizes -- and What Doesn't](https://arxiv.org/abs/2608.02829) | 本文发现，在不同规模的Transformer模型间转换时，表示对齐强但参数对齐弱，价值在于初始化，并通过最小二乘补偿和方差保持重缩放两个杠杆实现有效转换。 |
| [^153] | [Gaokerena: A Small Persian Medical Language Model Family](https://arxiv.org/abs/2608.00932) | 本文提出了Gaokerena，一个专为消费级硬件设计的小型波斯语医学语言模型家族，其中Gaokerena-V通过新构建的波斯语医学语料库训练提升了医学问答性能，Gaokerena-R则结合思维链与两个新型RLAIF框架来增强临床推理能力。 |
| [^154] | [CONSISTRE: A Unified Consistency-Aware Framework for Document-Level Relation Extraction with Large Language Models](https://arxiv.org/abs/2607.24312) | CONSISTRE提出一个统一的一致性感知框架，通过面向黑盒大语言模型的约束感知提示、约束验证与迭代自我反思，以及向较小开源模型注入一致性知识这两条互补路径，解决文档级关系抽取中预测违反传递性、对称性等关系约束而产生矛盾输出的问题。 |
| [^155] | [An Evaluation Framework for Structured Audio Captions Validated by Controlled Perturbations](https://arxiv.org/abs/2607.21424) | 提出了一个涵盖标签集、描述、推理、数值测量和频谱特征五个维度的结构化音频字幕评估框架，结合LLM评判器与确定性指标，并通过受控扰动验证了各指标的有效性。 |
| [^156] | [A JoLT for the KV cache: Near-lossless KV cache compression via joint Lagrangian allocation of Tucker ranks and a rotated residual for llms](https://arxiv.org/abs/2607.12550) | 本文提出JoLT方法，通过部分Tucker分解和旋转低比特残差，在保持头与层轴完整的同时压缩令牌和特征轴，实现KV缓存的近无损压缩。 |
| [^157] | [Closing the Quality Gap in Low-Resource Text-to-Speech: LoRA Fine-Tuning of VoxCPM2 for Khmer and Korean](https://arxiv.org/abs/2606.26618) | 通过零初始化的LoRA适配器同时微调两种低资源语言（高棉语和韩语）的VoxCPM2模型，仅训练少量参数即可显著提升语音质量，使高棉语MOS得分从3.85提升至4.23。 |
| [^158] | [Who Owns the AI Recommendation? A Multi-Industry Empirical Map of Brand Category Ownership Across Large Language Models](https://arxiv.org/abs/2606.23057) | 该研究通过对五个行业50个品牌、250个查询在三个大语言模型上跨越两个月的大规模实证测量，首次绘制了AI推荐中品牌类别归属的图谱，发现品牌收录率较为均衡、推荐份额随时间高度稳定，并识别出7.6%的“竞争真空”查询。 |
| [^159] | [Recovering the Zipfian Distribution in Unsupervised Term Discovery](https://arxiv.org/abs/2606.10781) | 该论文提出用基于图的Leiden聚类替代K-means等基于中心的方法，在无监督词条发现中显著恢复了真实词库所具有的齐夫分布特性。 |
| [^160] | [Persona Prompting in Multimodal Urban Perception: Descriptive Convergence and Interpretive Variation](https://arxiv.org/abs/2605.29064) | 本研究通过约12万条人格化标注发现，多模态大语言模型在城市感知中生成的客观图像描述几乎不随人格提示改变，但其主观解读却随人格显著变化，其中经济身份的影响最大。 |
| [^161] | [When Search Becomes Memory: Accelerating Robot Design Discovery with Self-Evolving Skills](https://arxiv.org/abs/2605.25832) | 提出Auto-Robotist，一个自进化的LLM智能体，通过将进化搜索轨迹提炼为可检查的自然语言技能库，把搜索结果转化为可重用的设计记忆，从而加速机器人形态设计的发现。 |
| [^162] | [Proactive for Uncertainty: Cause-Aware Error Diagnosis and Interactive Clarification for Spoken Dialogue Systems](https://arxiv.org/abs/2605.25404) | 该论文提出因果感知的错误恢复范式，利用一组小型高精度检测器诊断ASR错误成因（声学或语言失配）并通过交互式澄清进行主动恢复，克服了传统置信度过滤无法检测删除错误和区分错误类型的局限。 |
| [^163] | [An Empirical Study of Automating Agent Evaluation](https://arxiv.org/abs/2605.11378) | 仅靠提示前沿编程助手无法可靠地自动化智能体评估，本文提出EvalAgent，通过将评估领域专业知识编码为可组合的评估技能，实现了端到端的智能体评估自动化。 |
| [^164] | [Frontier Lag: A Bibliometric Audit of Capability Misrepresentation in Academic AI Evaluation](https://arxiv.org/abs/2605.04135) | 该研究对超过11万篇文献进行系统计量分析后发现，学术论文中评估的LLM能力落后于当时的前沿模型（中位数差距+10.85 ECI），且这一“前沿滞后”差距正以每年+5.53 ECI的速度持续扩大。 |
| [^165] | [Continued Pretraining of FinBERT on Finnish Histopathological Reports: Train-Time Signals and Proxy Downstream Correlations](https://arxiv.org/abs/2604.14815) | 本文将在芬兰语组织病理学报告上持续预训练FinBERT，发现CPT训练时损失曲线具有明显的领域差异，且某些CPT衍生特征与代理下游分类性能提升相关，为芬兰医疗NLP这一文献稀缺的领域做出了贡献。 |
| [^166] | [Correct Prediction, Wrong Steps? Consensus Reasoning Knowledge Graph for Robust Chain-of-Thought Synthesis](https://arxiv.org/abs/2604.14121) | 提出CRAFT方法，通过聚合多个候选推理轨迹的共识组件构建推理知识图谱，从推理结构层面修复LLM“答案正确但推理步骤有缺陷”的问题，实现更鲁棒的思维链合成。 |
| [^167] | [IatroBench: A Pre-Registered Benchmark of Clinical Omission in Language Models](https://arxiv.org/abs/2604.07709) | 论文提出预注册基准IatroBench，从“作为”与“遗漏”两个伤害维度评估语言模型在临床场景中的安全性，并首次揭示了模型对同一病例会向医生提供比患者更多临床信息的“框架依赖性信息保留”现象。 |
| [^168] | [LiveMathematicianBench: A Live Benchmark for Research-Level Mathematical Reasoning with Proof Sketches](https://arxiv.org/abs/2604.01754) | 提出了LiveMathematicianBench，一个基于训练截止日期后新发表arXiv论文构建的动态研究级数学推理基准测试，通过引入十三类定理逻辑分类体系和证明概要实现细粒度评估，有效避免了数据污染问题。 |
| [^169] | [Invertible Query-Key Coupling Composes with Attention Mechanisms](https://arxiv.org/abs/2604.01683) | 提出一种可逆的查询-键耦合变换（RealNVP风格的交替仿射映射），可无损地叠加在现有注意力机制之上，以极少的额外参数和不变的整体架构显著提升差分注意力等方法的语言建模性能。 |
| [^170] | [DiscoPhon: Benchmarking the Unsupervised Discovery of Phoneme Inventories With Discrete Speech Units](https://arxiv.org/abs/2603.18612) | 该论文提出了DiscoPhon，一个基于离散语音单元评估无监督音位发现的多语言基准测试，通过12种语言的评测和四个预训练基线模型揭示了当前语音模型中音位信息的可用性及其跨语言差异。 |
| [^171] | [Safety Under Scaffolding: How Evaluation Conditions Shape Measured Safety](https://arxiv.org/abs/2603.10044) | 评测条件对测得的模型安全性影响超过脚手架本身——在相同的基准题目上，选择题与开放式格式会使测得的安全性相差5-20个百分点，说明评测结果更多取决于测量方法而非模型潜在的 safety 能力。 |
| [^172] | [Quantum Attention by Overlap Interference: Predicting Classical and Many-Body Quantum Sequences](https://arxiv.org/abs/2602.06699) | 提出了一种通过状态重叠干涉与多项式核实现非线性、并利用Rényi-1/2熵泛函估计损失的变分量子自注意力机制（QSA），相比最优经典方法在训练复杂度上具有潜在优势，可用于预测经典和多体量子序列。 |
| [^173] | [LLM surprisal is necessary but not sufficient to capture English garden-path effects: Evidence from joint latent modeling of reading paradigms](https://arxiv.org/abs/2602.04489) | 该研究提出一个联合潜在过程多项加工树模型，整合眼动追踪、自定步速阅读和迷宫任务四种阅读范式的数据来建模花园路径句子的加工过程，结果表明大语言模型惊奇度是解释英语花园路径效应的必要但非充分条件。 |
| [^174] | [Do not be greedy, Think Twice: Sampling and Selection for Document-level Information Extraction](https://arxiv.org/abs/2601.18395) | 提出ThinkTwice框架，让大语言模型为文档级信息抽取生成多个候选模板，再通过无监督一致性或基于奖励模型的有监督方法选出最优模板，显著超越贪心解码，并提出基于拒绝采样的方法缓解黄金推理轨迹稀缺问题。 |
| [^175] | [Beyond Prompting: Efficient and Robust Contextual Biasing for Speech LLMs via Logit-Space Integration (LOGIC)](https://arxiv.org/abs/2601.15397) | 本文提出LOGIC方法，通过在Logit空间层面直接集成上下文偏置，为语音大语言模型提供了一种高效且鲁棒的解决方案，克服了传统提示方法的可扩展性瓶颈和生成式错误纠正的幻觉问题。 |
| [^176] | [Calibration Is Not Enough: Evaluating Confidence Estimation Under Language Variations](https://arxiv.org/abs/2601.08064) | 该论文提出了基于鲁棒性、稳定性和敏感性三个互补属性的置信度估计新评估框架，发现这些指标与现有指标基本独立，且现有方法虽具备较好的鲁棒性和稳定性，却难以区分语义不同的答案。 |
| [^177] | [IDRBench: Benchmarking the Interactive Capabilities of Deep Research Agents](https://arxiv.org/abs/2601.06676) | IDRBench是首个评估深度研究智能体交互能力的基准，通过受控澄清机会比较自主与交互式工作流，揭示了及时与用户交互对提升研究报告质量的重要性。 |
| [^178] | [SPARQL-LLM: Real-Time SPARQL Query Generation from Natural Language Questions](https://arxiv.org/abs/2512.14277) | SPARQL-LLM是一种开源、与三元组存储无关、由轻量级元数据驱动的方法，能够从自然语言实时生成SPARQL查询，兼顾准确性、运行时和成本等指标，从而实现生产环境的实际部署。 |
| [^179] | [A Fast and Effective Solution to the Problem of Look-ahead Bias in LLMs](https://arxiv.org/abs/2512.06607) | 本文提出一种推理时干预方法，利用两个小型专用模型调整大模型logits，从而快速、低成本地消除大语言模型在金融预测中的前瞻偏差。 |
| [^180] | [RapidUn: Influence-Driven Parameter Reweighting for Efficient Large Language Model Unlearning](https://arxiv.org/abs/2512.04457) | RapidUn通过将跨样本影响力估计转化为固定的样本特定权重来实现加权LoRA遗忘，能在保持模型干净效用的同时更有效地移除目标行为污染，且比LoRA重训练快77倍。 |
| [^181] | [Enabling Approximate Joint Sampling in Diffusion LMs](https://arxiv.org/abs/2509.22738) | 本文提出在现有大型扩散语言模型之上附加一个轻量级单层“采样器”，使模型能够在一次前向传播中近似地从真实联合分布并行采样多个 token，从而在保持准确率的同时大幅提升生成速度。 |
| [^182] | [Interactive In-Meeting Speaker Correction with Human Feedback](https://arxiv.org/abs/2509.18377) | 本文提出了一个LLM辅助的会议中说话人纠错系统，用户可通过简短的纠正反馈修复说话人归属错误，系统通过多种机制精确识别纠正意图，并借助LLM驱动的用户反馈模拟实现可复现、大规模的评估。 |
| [^183] | [Unraveling the cognitive patterns of Large Language Models through module communities](https://arxiv.org/abs/2508.18192) | 该研究借鉴生物认知系统的分析方法，开发了一个连接认知技能、LLM架构和数据集的基于网络的框架，通过模块社区分析揭示了大语言模型展现出独特的模块组织结构，其涌现的技能模式部分类似于生物系统的认知特化机制。 |
| [^184] | [Conversational DNA: A Visual Language and Interactive Atlas of Human and AI Dialogue](https://arxiv.org/abs/2508.07520) | 本文提出“对话DNA”，一种通过说话者链、话步标记和有向配对来可视化人类与AI对话结构的视觉语言与交互式图集，其引入的目标对应关系使对话结构检索的precision@5从58.8%显著提升至77.2%。 |
| [^185] | [Language Specific Knowledge: Do Models Know Better in X than in English?](https://arxiv.org/abs/2505.14990) | 本文提出“语言特定知识”（LSK）的概念，发现对某些查询使用英语以外的语言（有时甚至是低资源语言）提问能提升大语言模型的问答表现，并据此提出语言选择问题及多种基线方法。 |
| [^186] | [Foundations of Large Language Models](https://arxiv.org/abs/2501.09223) | 本书系统阐述了大语言模型的六大核心基础领域——预训练、生成模型、提示、对齐、推断与推理，为学习者提供了一部权威的基础性参考书。 |
| [^187] | [How Do Users Negotiate Harmful Value Conflicts with AI Companions? A Study with Minion, a Technology Probe for In-Situ Human-AI Conflict Response](https://arxiv.org/abs/2411.07042) | 该研究通过技术探针Minion发现，用户与AI伴侣协商有害价值冲突时会综合运用软硬策略，其中涉及普遍主义与传统价值观的冲突最难化解，且由于AI无法回馈用户的人际修复努力，冲突修复成为用户单方面承担的安全工作。 |
| [^188] | [MultiViewDx: Evidence-Linked Multi-View Clinical Diagnosis](https://arxiv.org/abs/2410.14948) | 提出了部分经医生验证的MultiViewDx数据集，以临床病例为监督单元，将影像与患者背景关联，并通过统一的图文检索器将报告规范化为“证据→发现→鉴别讨论→诊断”的证据关联工作流程，从而构建多视角医学影像诊断指令数据。 |
| [^189] | [Evaluation of OpenAI o1: Opportunities and Challenges of AGI](https://arxiv.org/abs/2409.18486) | 该研究全面评估了OpenAI o1-preview模型在编程、数学、医学等多领域复杂推理任务中的表现，发现其经常达到或超越人类水平，展现了AGI带来的机遇与挑战。 |
| [^190] | [Generating Interesting Scientific Ideas using Knowledge Graphs and LLMs: Evaluations with 100 Research Group Leaders](https://arxiv.org/abs/2405.17044) | 该研究提出SciMuse系统，利用包含5800万篇论文的知识图谱结合大语言模型生成个性化研究想法，并通过100多位研究团队负责人对4400多个想法的大规模评估发现，专家整体兴趣评分虽保守（均值2.40/5），但近四分之一的想法获得了高分认可。 |
| [^191] | [Calpric: Inclusive and Fine-grain Labeling of Privacy Policies with Crowdsourcing and Active Learning](https://arxiv.org/abs/2008.02954) | 该论文提出Calpric框架，通过结合自动文本分割、众包标注和主动学习，以低成本高效生成大规模、高质量的隐私政策训练数据集，使未经训练的众包标注者能达到与专业标注者相当的标注质量。 |

# 详细

[^1]: 在线阴谋论话语的智能体检测

    Agentic Detection of Online Conspiracies

    [https://arxiv.org/abs/2609.30250](https://arxiv.org/abs/2609.30250)

    提出了一个配备社会查询工具的智能体框架，通过结合社会语境推断说话者意图（言外之力），而非仅依赖词汇标记，来检测社交媒体上表达方式隐晦的阴谋论话语。

    

    社交媒体上的阴谋论话语并不总是通过明确的声明或稳定的词汇标记来表达。相同的表面内容可能表达的是认同、合理的担忧、批评、讽刺或嘲弄。因此，主要的挑战不仅是识别与阴谋论相关的声明，而是推断说话者的意图——即话语的言外之力。我们认为这可以通过利用相关的社会语境来实现，为此我们提出了一个配备了支持社会查询工具集的智能体框架。我们在一个独特的希伯来语推文数据集上展示了该方法的优势，该数据集涵盖了四年期间（2018年末至2023年初）公开发布的80%–90%的希伯来语推文，横跨多个选举周期以及新冠疫情和相关疫苗接种运动时期。这种广泛的覆盖范围可用于恢复不同的社会语境。我们在人工标注的数据上对该框架进行了评估。

    arXiv:2609.30250v1 Announce Type: new  Abstract: Conspiratorial discourse on social media is not always expressed through explicit claims or stable lexical markers. The same surface content may express endorsement, legitimate concerns, criticism, satire, or mockery. The main challenge is therefore not only recognizing conspiracy-related claims, but inferring the speaker's intent -- the utterance's illocutionary force. We argue that this can be achieved through the use of relevant social contexts and propose an agentic framework, equipped with a set of tools supporting social queries.   We demonstrate the benefits of our approach on a unique dataset of Hebrew tweets, covering 80\%--90\% of the public Hebrew tweets published over a four-year span (late 2018-- early 2023), encompassing several election cycles as well as the COVID pandemic years and related vaccination campaigns. This extensive coverage can be used in recovering different social contexts. Evaluating our framework on a manu
    
[^2]: JevOut：自然上下文可以颠覆决策模型

    JevOut: Natural Context Can Flip Decision Models

    [https://arxiv.org/abs/2609.30243](https://arxiv.org/abs/2609.30243)

    研究表明，决策模型（如Jev）对看似自然无害的简短上下文添加内容极其脆弱，优化后的上下文能在61.4%的情况下颠覆模型原本正确的决策，且在许多案例中使模型以至少0.7的高概率输出固定的错误选项。

    

    专门的决策模型（如Jev）能够将非结构化语言映射到有限选择的概率分布上，使其输出可以直接路由请求、选择工具并触发操作。然而，现实世界的输入很少是孤立到达的：它们通常伴随着背景细节和上下文环境。我们发现，即使正确答案保持不变，一些能够自然融入上下文的简短添加内容却可以改变原本正确的决策。为了研究这种行为，我们为每个最初回答正确的项目固定一个错误的目标选项，并利用模型的选项概率来优化流畅的上下文添加内容，同时保持源文本、问题、选项和标准答案不变。在64次被接受的目标评估中，优化器找到的上下文在508个最初正确的决策中成功改变了312个（61.4%）Jev的决策；在229个案例中，Jev为固定的错误选项分配了至少0.7的概率。跨七个数据集……（摘要被截断）

    arXiv:2609.30243v1 Announce Type: new  Abstract: Dedicated decision models such as Jev map unstructured language to probability distributions over finite choices, allowing their outputs to directly route requests, select tools, and trigger actions. Yet real-world inputs rarely arrive in isolation: they come with background details and surrounding context. We find that short additions that fit naturally into this context can nevertheless redirect an otherwise correct decision, even when the correct answer remains unchanged. To study this behavior, we fix a wrong target option for each initially correct item and use the model's option probabilities to refine fluent context additions while preserving the source, question, choices, and gold answer. Within 64 accepted target evaluations, the optimizer identifies contexts that redirect Jev on 312 of 508 initially correct decisions (61.4%); in 229 cases, Jev assigns at least 0.7 probability to the fixed wrong option. Across seven datasets, th
    
[^3]: SemMSA：面向不完整数据的潜在语义辅助鲁棒多模态情感分析

    SemMSA: Latent Semantic-Aided Robust Multimodal Sentiment Analysis with Incomplete Data

    [https://arxiv.org/abs/2609.30238](https://arxiv.org/abs/2609.30238)

    提出SemMSA框架，利用大语言模型构建丰富的情感相关潜在语义，并通过跨模态语义精炼与无锚点谱对齐将所有模态充分融合，从而在不完整数据条件下实现更鲁棒的多模态情感分析。

    

    arXiv:2609.30238v1 公告类型：新论文 摘要：近年来，多模态情感分析（MSA）的研究主要聚焦于在数据不完整的情况下，从语言、视觉和声学模态中学习以推断人类情感。大多数研究通常通过重构模态特征或设计复杂的融合机制来补偿缺失信息。然而，由于在部分可观测的多模态证据中缺乏高层语义支撑，这些方法仍然存在虚假生成和噪声引导的问题。为了解决这些问题，我们提出了SemMSA，这是一种潜在语义辅助框架，它利用大语言模型构建丰富的情感相关语义，并通过无锚点谱对齐与所有模态充分融合。该框架主要由跨模态语义精炼（CSR）和跨模态谱对齐（CSA）两部分组成。具体而言，CSR首先通过相应的适配器自适应地提取视觉和声学表征，与语言模态共同形成统一的多模态前缀……

    arXiv:2609.30238v1 Announce Type: new  Abstract: Recent research on Multimodal Sentiment Analysis (MSA) has focused on learning from language, visual, and acoustic modalities with incomplete data to infer human sentiment. Most studies typically compensate for missing information by reconstructing modality features or designing complicated fusion mechanisms. However, these methods still suffer from spurious generation and noisy guidance due to the lack of high-level semantic grounding in partially observed multimodal evidence. To address these issues, we propose SemMSA, a latent semantic-aided framework that constructs rich sentiment-relevant semantics with LLMs, fully integrating with all modalities via anchor-free spectral alignment. It mainly consists of Cross-modal Semantic Refinement (CSR) and Cross-modal Spectral Alignment (CSA). Specifically, CSR first adaptively extracts visual and acoustic representations by corresponding adapters to form a unified multimodal prefix with langua
    
[^4]: 信任与否：语音中基于检索增强的事实核查

    To Trust or Not to Trust: Retrieval-Augmented Fact Checking in Speech

    [https://arxiv.org/abs/2609.30227](https://arxiv.org/abs/2609.30227)

    论文提出VeriSpeak语音事实核查基准，揭示了大型音频语言模型存在显著的文本-语音模态差距（书面声明可验证但语音版本常失败），且仅靠检索增强带来的提升有限。

    

    在线虚假信息越来越多地以语音形式出现，例如新闻片段、播客、访谈、政治演讲和社交媒体视频，这催生了对能够直接从语音中核查声明的系统的需求。我们提出了VeriSpeak，一个用于研究大型音频语言模型（LALMs）中基于语音的事实验证能力的探测基准。VeriSpeak包含3,879条语音声明，涵盖时间性、地理性和关系性事实，且真伪标签均衡。该基准旨在检验事实验证能力能否从文本迁移到语音，以及检索增强的LALMs能否利用文本证据来正确支持或反驳语音声明。我们的实验揭示了一个持续存在的文本-语音模态差距：那些能够可靠验证书面声明的LALMs，在面对相同声明以语音形式呈现时往往会失败。此外，仅靠检索带来的增益有限，因为模型经常混淆检索到的证据……

    arXiv:2609.30227v1 Announce Type: cross  Abstract: Online misinformation increasingly appears in spoken formats such as news clips, podcasts, interviews, political speeches, and social media videos, creating a need for fact-checking systems that can verify claims directly from speech. We introduce VeriSpeak, a probe benchmark for studying speech-based fact verification in Large Audio Language Models (LALMs). VeriSpeak contains 3,879 spoken claims spanning temporal, geographical, and relational facts, with balanced true and false labels. The benchmark is designed to examine whether factual verification ability transfers from text to speech, and whether retrieval-augmented LALMs can use textual evidence to correctly support or refute spoken claims. Our experiments reveal a consistent text-speech modality gap: LALMs that verify written claims reliably often fail on the same claims when spoken. Moreover, retrieval alone provides limited gains because models frequently conflate retrieved ev
    
[^5]: PoEM：从现有策略预测强化学习结果

    PoEM: Predicting RL Outcomes from Existing Policies

    [https://arxiv.org/abs/2609.30226](https://arxiv.org/abs/2609.30226)

    提出PoEM框架，利用一组已在其他奖励上完成强化学习后训练的现有模型来预测新奖励函数下的强化学习结果，从而避免每次奖励变化时都从头运行昂贵且不稳定的强化学习过程。

    

    基础模型通过强化学习（RL）进行后训练，以最大化特定的奖励，例如人类对齐、正确性或指令遵循。这一后训练过程计算量巨大，有时不稳定，并且每次当奖励模型发生变化或我们想要组合多个奖励时，都必须从头开始运行。因此我们提出这样一个问题：给定一个新的奖励函数，是否可以在不实际运行强化学习的情况下预测其强化学习结果？我们通过引入PoEM对这个问题给出了肯定的回答。PoEM是一个框架，它利用一组已经在其他奖励上完成过后训练的模型，来预测在新奖励函数上运行强化学习的输出结果。首先，我们证明如果新的奖励函数可以表示为现有奖励函数的线性组合，那么新的策略在对数空间中也可以表示为现有对数策略的线性组合。令人惊讶的是，即使在奖励之间不存在线性关系的情况下，我们观察到……（摘要在此处截断）

    arXiv:2609.30226v1 Announce Type: cross  Abstract: Foundation models are post-trained with reinforcement learning (RL) to maximize specific rewards, such as human alignment, correctness, or instruction following. This post-training process is computationally intensive, sometimes unstable, and has to be run from scratch every time the reward model changes or when we want to combine multiple rewards. We hence ask: given a new reward function, is it possible to predict the RL outcomes without actually running RL on it? We answer this in the affirmative by introducing PoEM, a framework to predict the outputs of RL on a new reward function using a set of models already post-trained on other rewards. First, we show that if the new reward function can be written as a linear combination of existing ones, then the new policy in log-space can be written as a linear combination of the existing log-policies. Surprisingly, even in cases where the rewards are not linearly connected, we observe that 
    
[^6]: ExplorationBench：在可验证的异星世界中衡量AI系统的探索能力

    ExplorationBench: Measuring AI Systems' Exploration in Verifiable Alien Worlds

    [https://arxiv.org/abs/2609.30199](https://arxiv.org/abs/2609.30199)

    提出ExplorationBench基准，利用规则可执行且与常识相冲突的“异星世界”沙盒（AlienCode与AlienLogic），实现了对AI系统科学探索能力的可验证评估，排除了仅凭记忆预训练知识解题的可能。

    

    科学发现始于已知问题终结之处。在那里，AI系统必须进行探索：提出假设、设计实验并对结果进行迭代。然而，评估这种能力十分困难：（1）如何验证一个真正新颖的假设是否成立，（2）如何判断系统是通过探索发现了它，还是仅仅从预训练数据中回忆了相关知识。为此，我们提出了ExplorationBench，它将评估科学探索这一棘手问题转化为一个建立在可验证“异星世界”之上的具体且易于处理的框架：这些世界的规则是可执行的，因此每个答案都可以被精确检验；同时它们与熟悉的知识相冲突，因此仅靠记忆无法解决任务。该基准包含两个沙盒：AlienCode（31个发现目标，70个任务）和AlienLogic（24个发现目标，70个任务）。每个沙盒都提供一份有缺陷的手册以及任务特定的环境……

    arXiv:2609.30199v1 Announce Type: new  Abstract: Scientific discovery begins where known problems end. There, AI systems must engage in exploration: framing hypotheses, designing experiments, and iterating on the results. However, evaluating this ability is difficult: (1) how to verify whether a genuinely new hypothesis holds, and (2) how to determine whether a system has discovered it through exploration or merely recalled related knowledge from pre-training data. To this end, we introduce ExplorationBench, which turns the wicked problem of evaluating scientific exploration into a concrete and tractable framework built on verifiable Alien Worlds: their rules are executable, so every answer can be checked exactly, and they conflict with familiar knowledge, so recall alone cannot solve the tasks. The benchmark contains two sandboxes, AlienCode (31 discovery targets, 70 tasks) and AlienLogic (24 discovery targets, 70 tasks). Each sandbox provides a flawed manual, task-specific environmen
    
[^7]: ARGUS：面向美国就业歧视投诉的角色感知事件知识图谱

    ARGUS: Role-Aware Event Knowledge Graphs for U.S. Employment-Discrimination Complaints

    [https://arxiv.org/abs/2609.30184](https://arxiv.org/abs/2609.30184)

    本文提出ARGUS流水线，结合5W1H模式、法律领域模型和LLM结构化生成，从美国就业歧视投诉文本中构建文档级事件知识图谱，在诉求分类和法律问答任务上显著优于传统文本表示方法。

    

    美国就业歧视投诉描述了复杂的事件序列，仅靠词汇或基于嵌入的表示无法显式捕捉这些序列。我们提出了ARGUS，这是一个基于原文的流水线，结合受5W1H启发的模式、法律领域模型以及基于大语言模型的结构化生成，从CourtListener投诉文本中构建文档级事件知识图谱（EKG）。ARGUS提取承载事实的陈述，构建具有参与者、时间和因果结构的块级事件图谱，并将其合并为文档级表示。我们通过人工和多模型评估来评价图谱质量，并在诉求分类和法律问答任务上测试其下游效用。图结构分类器在保留测试集上优于原始文本和线性化基线，仅基于EKG的检索提升了文档范围问答的性能，而开放检索的收益仍受限于第一阶段候选召回率较低。这些结果表明E（摘要在此处被截断）

    arXiv:2609.30184v1 Announce Type: new  Abstract: U.S. employment-discrimination complaints describe complex event sequences that are not explicitly captured by lexical or embedding-based representations alone. We present ARGUS, a source-grounded pipeline that combines a 5W1H-inspired schema, legal-domain models, and LLM-based structured generation to construct document-level Event Knowledge Graphs (EKGs) from CourtListener complaints. ARGUS extracts fact-bearing statements, builds chunk-level event graphs with participant, temporal, and causal structure, and merges them into document-level representations. We evaluate graph quality through human and multi-model assessment and test downstream utility on claim classification and legal QA. The graph-structured classifier outperforms raw and linearized baselines on the held-out set, and EKG-only retrieval improves document-scoped QA, while open-retrieval gains remain limited by low first-stage candidate recall. These results suggest that E
    
[^8]: 音频语言模型听到与读到的区别特征是否一致？

    Do Audio Language Models Hear and Read Distinctive Features Alike?

    [https://arxiv.org/abs/2609.30167](https://arxiv.org/abs/2609.30167)

    该研究通过最小对立音素对与随机配对基准的分析发现，音频语言模型的解码器对音素区别特征的表征在听觉与阅读两种模态间普遍不一致，仅有Qwen2.5-Omni模型中的浊音特征表现出超过随机基准的跨模态方向一致性。

    

    音频语言模型让语音和文本经过同一个解码器。我们探究的问题是：当模型听到一个音素与读到该音素时，解码器是否在相同的方向上表征其区别特征。对于仅在一个特征上不同的最小对立音素对，我们计算两个成员平均表征之间的偏移量；对这些偏移量取平均得到每个模态（流）的方向，然后测量两个方向之间的余弦相似度。由于两个模态对于任意音素对本就存在一定程度的一致性，我们将所有度量与基于随机配对构建的参考基准进行比较，而非与零进行比较。我们将该方法应用于6个模型、7个特征以及来自11个语系的15种语言。经多重检验校正后，只有两个Qwen2.5-Omni模型中的浊音（voicing）特征超过了该参考基准，且该参考基准在不同模型之间相差达七倍。在六个模型中的三个里，浊音在音频模态下于14种拥有足够最小对立对可供测量的语言中共享同一个方向。

    arXiv:2609.30167v1 Announce Type: new  Abstract: Audio language models pass speech and text through a single decoder. We ask whether that decoder represents a distinctive feature in the same direction when a phoneme is heard and when it is read. For minimal pairs of phonemes differing in one feature, we take the offset between the two members' mean representations. Averaging those offsets gives a direction for each stream, and we measure the cosine between the two. Because the two streams already agree about arbitrary phoneme pairs, we compare every measure against a reference built from random pairings rather than against zero. We apply this to 6 models, 7 features and 15 languages from 11 families. Only voicing in the two Qwen2.5-Omni models exceeds that reference after correction for multiple testing, and the reference varies by a factor of seven between models. In three of the six models, voicing has one direction in audio across the 14 languages with enough minimal pairs to measur
    
[^9]: 一种对转录歧义具有词元级容忍度的自动语音识别训练准则

    A Training Criterion with Token-Level Tolerance to Transcription Ambiguity for Automatic Speech Recognition

    [https://arxiv.org/abs/2609.30160](https://arxiv.org/abs/2609.30160)

    该论文提出词元级OTC训练准则，将通配弧细化到词元粒度并结合词元级与词级互补逃逸路径，使自动语音识别训练既能绕过有歧义的词元又保留词内其余部分的监督信号，在19种语言的全部25个任务上均超越CTC。

    

    自动语音识别通常在训练时假设参考转录文本是语音话语的唯一有效标注，然而即使是名义上逐字记录的转录，也包含发音、拼写或词汇实现上的局部差异，这些差异在声学上并非唯一确定。全时序分类通过在连接时序分类（CTC）对齐图中添加通配路径来容忍此类噪声，但其词级弧过于粗糙，因为绕过一个不受支持的词元就会丢弃对整个词的监督信息。我们将通配弧细化到词元粒度，使得不受支持的词元可以被绕过，而词的其余部分仍保持受监督，并且我们将词元级弧与词级弧相结合，作为互补的逃逸路径。在19种语言和三个语料库上，词元级OTC在全部25个任务上都优于CTC。我们还将基于训练轮次索引的通配权重松弛方式替换为基于预测熵的……（摘要在此处截断）

    arXiv:2609.30160v1 Announce Type: new  Abstract: Automatic speech recognition is typically trained assuming that the reference transcript is the only valid labeling of an utterance, yet even nominally verbatim transcripts contain localized differences in pronunciation, spelling, or lexical realization that the acoustics do not uniquely determine. Omni-temporal Classification (OTC) tolerates such noise by adding wildcard paths to the connectionist temporal classification (CTC) alignment graph, but its word-level arcs are too coarse, since bypassing one unsupported token discards supervision for the whole word. We move wildcard arcs to token granularity so unsupported tokens can be bypassed while the rest of the word stays supervised, and we combine token- and word-level arcs as complementary escape paths. Across 19 languages and three corpora, token-level OTC improves over CTC on all 25 tasks. We also replace epoch-indexed relaxation of the wildcard weights with a predictive-entropy-ind
    
[^10]: 模型所述拒绝候选者的理由真的起作用吗？

    Does a model's stated reason for rejecting a candidate do any work?

    [https://arxiv.org/abs/2609.30151](https://arxiv.org/abs/2609.30151)

    该研究通过将模型声称缺失的事实插入对应档案并在贪心解码下重新测试，首次因果性地验证了语言模型拒绝候选者时所述理由确实会实际影响其后续选择。

    

    当被要求在候选者之间做出选择并解释理由时，语言模型常常通过指出对手档案中缺失的某个事实来拒绝对方：比如“没有导演”、“没有死亡日期”。这句话是对模型面前文本的一个断言，而且可以在不需要任何评判者的情况下加以检验。我们将陈述该所提及事实的真实语料句子插入对手的档案中，并在贪心解码下重新提问。两个对照实验将内容与位置因素区分开来：在同一档案中加入长度匹配的无关句子，以及在模型从未提及的第三个选项处加入相同的两句话。在三次实验中规模最大的一次里——六个开源模型在2WikiMultihopQA数据集上——在模型所指出的档案处提供该所提及事实，比无关对照更能改变模型的选择，几率比为3.57 [1.54, 8.26]，Holm校正后p=0.0210，且该结果在剔除任何单个模型后依然成立。而该实验设计旨在检测的关键对照——在无人提及的选项处提供相同事实——未能通过多重比较校正（Holm……）

    arXiv:2609.30151v1 Announce Type: cross  Abstract: Asked to choose between candidates and explain the choice, a language model often rejects a rival by naming a fact its profile lacks: no director, no date of death. That sentence is a claim about the text in front of the model, and it can be tested without any judge. We insert a real corpus sentence stating the named fact into the rival's profile and ask again under greedy decoding. Two controls separate content from placement: a length-matched irrelevant sentence at the same profile, and the same two sentences at a third option the model never mentioned. In the largest of three runs, six open models on 2WikiMultihopQA, supplying the named fact at the profile the model named moves its choice more than the irrelevant control does, odds ratio 3.57 [1.54, 8.26], Holm p=0.0210, and this survives dropping any single model. The contrast the design was built to detect, the same fact at the option nobody named, does not clear correction, Holm 
    
[^11]: GRASP：基于智能体AI的策略规划生成、修订与评估框架

    GRASP: Generating, Revising, and Assessing for Strategic Planning with Agentic AI

    [https://arxiv.org/abs/2609.30147](https://arxiv.org/abs/2609.30147)

    GRASP是一个策略感知的多阶段规划框架，通过将规划流程解耦为生成、修订和评估三个上下文隔离的专门模块，显著提升了LLM在复杂任务上的规划准确率，在多个基准数据集上建立了新的最先进水平。

    

    大型语言模型（LLMs）通常表现出一种性能特征，即随着任务复杂性的增加，其可靠性会下降。我们通过引入GRASP——一个具有策略感知能力的多阶段规划框架——来解决为复杂任务生成高质量自然语言可执行计划的挑战。GRASP将规划流程解耦为多个专门化、上下文隔离的模块：它预编译全局宏观指导方针，在隔离的上下文窗口中探索备选的局部策略，并使用多标准判别器独立评估轨迹。实证评估表明，GRASP在多个数据集上持续确立了新的最先进水平，与直接的LLM规划器相比，在Natural Plan Calendar Scheduling（提升约12.4%）、ZebraLogic（提升约30.8%）和SciBench Math上取得了显著的准确率提升。

    arXiv:2609.30147v1 Announce Type: new  Abstract: Large Language Models (LLMs) typically exhibit a performance profile where reliability degrades as task complexity increases. We address the challenge of generating high-quality natural language executable plans for complex tasks by introducing $\textbf{GRASP}$, a strategy-aware, multi-stage planning framework. GRASP decouples the planning pipeline across specialized, context-isolated modules: it pre-compiles global macro-guidelines (GenPlan), explores alternative localized strategies within isolated context windows (RevPlan), and independently evaluates trajectories using a multi-criteria discriminator (VerPlan). Empirical evaluations show that GRASP consistently establishes a new state-of-the-art frontier across diverse datasets, yielding substantial accuracy gains over direct LLM planners on Natural Plan Calendar Scheduling ($\sim$12.4$\%$$\uparrow$), ZebraLogic ($\sim$30.8$\%$$\uparrow$), and SciBench Math. Crucially, under multi-tas
    
[^12]: 先筛查再服务：面向1.4亿规模生产级客户体验AI代理的仿真验证方法

    Screen Before You Serve: Simulation for Production Customer Experience AI Agents at 140M Scale

    [https://arxiv.org/abs/2609.30137](https://arxiv.org/abs/2609.30137)

    该论文提出了一种基于假设驱动的仿真工作流，利用合成客户和模拟工具输出，在部署前对大规模生产级客户体验AI代理进行筛查验证，从而避免在线实验对客户信任造成的风险。

    

    客户体验（CX）代理使用工具和大语言模型来处理客户请求，并引导用户与组织的产品进行对话式交互。改进这些代理，尤其是在受监管的行业中，是非常困难的：它们必须检测用户意图、遵循复杂的运营策略并可靠地使用工具。手动端到端测试覆盖范围有限，而在线实验则会让客户直面可能导致信任受损的故障。我们提出了一种基于假设驱动的仿真工作流，用于在部署前筛查候选的CX代理。合成客户会对代理的响应做出反应，模拟的工具输出使多步骤代理工作流无需调用生产后端即可运行。我们在Nubank的Card Delivery代理及其扩展后的继任者Card Management（Nubank在巴西聊天量最高的客服代理）上使用了Snowglobe仿真器。在4个已部署的版本中，仿真与生产环境的版本级二元评估器……

    arXiv:2609.30137v1 Announce Type: new  Abstract: Customer experience (CX) agents use tools and large language models to address customer requests and guide conversational interactions with an organization's products. Improving these agents, especially in regulated industries, is difficult: they must detect intent, follow complex operational policies and use tools reliably. Manual end-to-end testing offers limited coverage, while live experiments expose customers to failures that can erode trust.   We present a hypothesis-driven simulation workflow for screening candidate CX agents before deployment. Synthetic customers react to agent responses and simulated tool outputs enable multi-step agentic workflows without invoking production backends. We use the Snowglobe simulator on Nubank's Card Delivery agent and its expanded successor, Card Management - Nubank's highest-volume chat-support agent in Brazil. Across 4 deployed versions, simulated and production version-level binary evaluator 
    
[^13]: 基于可渲染程序的多模态思维

    Multimodal Thinking with Renderable Programs

    [https://arxiv.org/abs/2609.30130](https://arxiv.org/abs/2609.30130)

    提出SVGLM框架，利用SVG既可作为图像描述又可作为文本指令的双重性质，使通用视觉-语言模型能够在推理过程中生成图像，并提供基于SVG的图像编辑数据集及开源模型微调范式。

    

    当前的视觉-语言模型（VLM）在视觉内容理解和基于文本的推理方面表现出色，但其结构限制了将图像纳入推理链的进展。尽管全模态模型已在统一文本与图像生成方面做出努力，但它们专注于开放域的视觉任务，且由于图像采用光栅化或潜在表示而缺乏可处理性。我们提出了SVGLM，这是一个使用可缩放矢量图形（SVG）基元在推理任务中连接文本与图像的框架。我们利用SVG既可作为图像描述又可作为文本指令的双重性质，提供了一种更紧凑、更可解释的解决方案，使通用视觉-语言模型具备在推理过程中生成图像的能力。我们提供了一个大规模精选的基于SVG的图像编辑数据集，以及用于微调开源视觉-语言模型的范式。在数学推理基准上的实验表明，SVGLM取得了……

    arXiv:2609.30130v1 Announce Type: cross  Abstract: Current vision-language models (VLMs) excel at visual content understanding and text-based reasoning, yet their structure limits the advancement of incorporating images into the reasoning chain. Though Omnimodal models have made efforts in unifying text and image generation, they focus on visual tasks in the open-domain, lacking tractability due to rasterized or latent representations of images. We introduce SVGLM, a framework that uses scalable vector graphics (SVG) primitives to connect text and image in reasoning tasks. We exploit the duality of SVG as both image description and text instructions, yielding a more compact, interpretable solution to equip general VLMs with the capability of generating images within the reasoning process. We provide a large curated dataset of SVG-based image editing dataset, as well as the paradigm to tune open-source VLMs. Experiments on a mathematical reasoning benchmark demonstrate that SVGLM achiev
    
[^14]: 描述什么、何时描述、如何描述：将音频描述视为约束全局优化问题

    What, When, and How: Audio Description as Constrained Global Optimization

    [https://arxiv.org/abs/2609.30121](https://arxiv.org/abs/2609.30121)

    该论文首次将自动音频描述生成形式化为约束全局优化问题，利用大语言模型提出并评估视觉元素的叙事重要性，再通过混合整数线性规划在时间约束下跨场景联合选择和调度描述内容。

    

    音频描述通过在对话间隙叙述视觉信息，使盲人和视障观众能够更好地欣赏电影。现有的自动音频描述系统大多将生成视为局部的视频到文本问题，即假设要描述的内容及其时间位置已经预先给出。而现实中的音频描述需要做出相互关联的决策：哪些视觉信息在叙事上是重要的、何时可以在不干扰对话的情况下进行讲述、以及如何表述以适应可用的时间。我们将音频描述的生成形式化为围绕这三个决策的约束优化问题。我们的混合系统使用大语言模型来提出并定位视觉元素，估计它们对叙事的重要性，并生成压缩后的表述。随后，混合整数线性规划在时间约束下跨场景联合选择并调度各条描述。

    arXiv:2609.30121v1 Announce Type: new  Abstract: Audio Description (AD) makes movies accessible to blind and visually impaired audiences by narrating visual information in gaps between dialogue. Existing automatic AD systems largely treat generation as a local video-to-text problem, assuming that the content to describe and its temporal location are already provided. Realistic AD instead requires coupled decisions about what visual information is narratively important, when it can be spoken without interfering with dialogue, and how it should be formulated to fit within the available time. We formalize AD generation as a constrained optimization problem over these three decisions. Our hybrid system uses large language models to propose and ground visual elements, estimate their salience to the narrative, and generate compressed realizations. A mixed-integer linear program then jointly selects and schedules descriptions across a scene subject to temporal constraints. When evaluated on R
    
[^15]: R-DEIM Net：一种面向释义检测的高效推理增强型双专家交互模型

    R-DEIM Net: An Efficient Rationale-Augmented Dual-Expert Interaction Model for Paraphrase Detection

    [https://arxiv.org/abs/2609.30100](https://arxiv.org/abs/2609.30100)

    提出R-DEIM Net，一个7600万参数的双专家架构，通过交互专家捕获词元级相似性模式、推理专家利用Flan-T5-small生成人类可读的推理作为辅助监督，使中等规模模型在释义检测上实现有竞争力的准确率并保持推理透明度。

    

    释义检测领域的最新进展揭示了一个根本性的权衡：大型语言模型虽然能够达到很高的准确率，但需要高昂的计算成本；而高效的孪生BERT（Siamese-BERT）变体虽然具备实际可扩展性，却在推理生成的透明度上有所不足。我们提出了R-DEIM Net，一个7600万参数的双专家架构，旨在探索中等规模的模型能否在释义检测任务上取得有竞争力的准确率，同时支持人类可读的推理生成。该架构结合了两个专门化的组件：一是交互专家，通过多尺度2D卷积和允许可变输入长度的注意力头来捕获词元级的相似性模式；二是推理专家，使用Flan-T5-small解码器生成推理文本作为辅助监督。我们并未对生成的文本进行重新编码，而是提取并池化解码器的隐藏状态，将其作为分类的补充特征。在Quora问题对数据集上……（原文截断）

    arXiv:2609.30100v1 Announce Type: cross  Abstract: Recent advances in paraphrase detection reveal a fundamental trade-off: large language models achieve high accuracy but require high computation, while efficient Siamese-BERT variants offer practical scalability with reduced transparency in rationale generation. We present R-DEIM Net, a 76M-parameter dual-expert architecture exploring whether moderate-scale models can achieve competitive accuracy on paraphrase detection while enabling human-readable rationale generation. The architecture combines two specialized components: an Interaction Expert that captures token-level similarity patterns through multi-scale 2D convolutions and attention head allowing variable input length, and a Reasoning Expert that uses a Flan-T5-small decoder to generate rationales as auxiliary supervision. Rather than re-encoding generated text, we extract and pool decoder hidden states as complementary features for classification. On the Quora Question Pairs da
    
[^16]: PrivDrift：主动LLM对话中话题漂移下用户秘密泄露的审计

    PrivDrift: Auditing User-Secret Leakage Under Topic Drift in Active LLM Conversations

    [https://arxiv.org/abs/2609.30094](https://arxiv.org/abs/2609.30094)

    提出PrivDrift审计基准，发现在LLM活跃对话中，用户披露的秘密即使经历话题漂移后仍高度可恢复（混合泄露率达38.7%–54.6%），且额外的话题漂移并不能可靠降低泄露风险。

    

    arXiv:2609.30094v1 公告类型：新论文 摘要：大型语言模型日益作为持久性助手应用于面向用户的、共享会话以及工具增强的场景中。当用户在活跃对话中披露敏感信息时，即使对话随后转向无关话题，这些信息通过后续提示仍可能在行为层面被恢复出来。我们提出了PrivDrift，这是一个用于审计用户所披露秘密在经历对话话题漂移和基于说服的探测之后是否仍可被恢复的基准。PrivDrift包含1,000个受控多轮对话，其中预置了秘密信息、内容密集的漂移轮次以及标准化的提取探测。在三个具有扩展上下文窗口的大语言模型上，对话级混合泄露依然严重，泄露率介于38.7%至54.6%之间，并且随模型、秘密类型和说服强度的不同而有显著变化。在所测试的漂移窗口内，额外的话题漂移并不能可靠地降低泄露，这表明隐私风险持续存在（摘要在此处被截断）。

    arXiv:2609.30094v1 Announce Type: new  Abstract: Large language models increasingly operate as persistent assistants in user-facing, shared-session, and tool-augmented settings. When users disclose sensitive information during an active conversation, that information may remain behaviorally recoverable through later prompts even after the dialogue shifts to unrelated topics. We introduce \textbf{PrivDrift}, a benchmark for auditing whether user-disclosed secrets remain recoverable after conversational topic drift and persuasion-based probing. PrivDrift contains 1{,}000 controlled multi-turn dialogues with seeded secrets, content-dense drift turns, and standardized extraction probes. Across three LLMs with extended context windows, dialogue-level hybrid leakage remains substantial, ranging from 38.7\% to 54.6\%, and varies strongly by model, secret type, and persuasion intensity. Within the tested drift window, additional topic drift does not reliably reduce leakage, suggesting that pri
    
[^17]: 返回还是修订？学习修订何时有助于检索增强问答

    Return or Revise? Learning When Revision Helps Retrieval-Augmented QA

    [https://arxiv.org/abs/2609.30087](https://arxiv.org/abs/2609.30087)

    本文提出“可恢复性”指标——通过在同一评判标准下同时评估草稿答案与其候选修订所得到的成对效果——并训练模型在修订前预测该指标，从而判断何时进行检索增强修订有益，在多个实验设置下均优于仅基于草稿置信度的决策方法。

    

    我们考虑这样一个决策问题：在答案修订系统中，是直接返回已有的草稿答案，还是利用检索到的证据对其进行修订。草稿置信度估计的是当前答案是否正确，但这一决策需要估计某次特定修订所带来的效果。为了进行离线训练与评估，我们在同一正确性评判标准下对返回的草稿答案及其候选修订同时进行评分，这使得修复效果、潜在损害以及与最优答案之间的差距变得可观测。我们将这种成对效应称为“可恢复性”（recoverability），并训练策略在修订之前对其进行预测。在三个修订设置下共 25,870 个留出的开放域问题上，基于成对结果训练的评分器在全部九个 Llama 设置-随机种子组合中，其“准确率-修订率”曲线下面积均优于同等条件的草稿正确性评分器，并且在由开发集选定的阈值下平均提升 0.23–0.68 个准确率百分点，该差异仅在不同训练运行之间具有统计显著性。

    arXiv:2609.30087v1 Announce Type: new  Abstract: We consider the decision of whether to return an existing draft answer or revise it using retrieved evidence, as in answer-revision systems. Draft confidence estimates whether the current answer is correct, but the decision requires estimating the effect of a specified revision. For offline training and evaluation, we grade both the returned draft and its candidate revision under the same correctness judge, which makes repair, harm, and the gap to an oracle observable. We call this paired effect its recoverability, and we train policies to predict it before revision. On 25,870 held-out open-domain questions across three revision setups, a scorer trained on the paired outcome has greater area under the accuracy--revision-rate curve than a matched draft-correctness scorer in all nine Llama setup--seed fits, and gains 0.23--0.68 accuracy points on average at development-selected thresholds, a difference significant across training runs only
    
[^18]: 一种用于第二语言发音分析的本族语参考音素类几何方法

    A Native-Reference Phone-Class Geometry for Second-Language Pronunciation Analysis

    [https://arxiv.org/abs/2609.30075](https://arxiv.org/abs/2609.30075)

    提出了一种无需发音标注或匹配录音的本族语参考音素类几何方法，通过将L2发音的自监督表示投影到本族语参考坐标系并计算距离，实现可解释的第二语言发音偏差测量，且该距离与整体口语熟练度呈一致负相关。

    

    自动口语评估系统可以提供整体熟练度分数，但往往缺乏能够刻画发音质量的可解释性度量。我们提出了一种本族语参考音素类几何方法，用于测量第二语言（L2）发音偏差，该方法无需发音标注、朗读提示，也无需本族语者与L2说话者对同一文本的匹配录音。给定一个本族语语料库，我们对每个上下文相关音素类计算帧级自监督表示的平均值，并使用奇异值分解（SVD）推导出一个紧凑的本族语参考坐标系。对于每条L2话语，我们计算相应的平均值并将其投影到本族语参考空间中。随后我们证明，在Speak and Improve数据集开发子集上，匹配音素类的L2坐标与本族语参考坐标之间的距离与整体口语熟练度呈现出一致的负相关关系。

    arXiv:2609.30075v1 Announce Type: new  Abstract: Automatic speaking assessment systems can provide holistic proficiency scores, but often lack interpretable measures that characterize pronunciation quality. We propose a native-reference phone-class geometry for measuring second language (L2) pronunciation deviation without requiring pronunciation labels, read-aloud prompts, or matched recordings of the same text from native and L2 speakers. Given a native speech corpus, we average frame-level self-supervised representations for each context-dependent phone-class and use singular value decomposition (SVD) to derive a compact native-reference coordinate system. For each L2 utterance, we compute the corresponding averages and project them into the native-reference space. We then demonstrate that the distances between L2 and native-reference coordinates for matched phone-classes show consistent negative correlations with holistic speaking proficiency on the Dev subset of the Speak and Impr
    
[^19]: 评估结论的可复现性如何？对LLM推断提示结构的自我审计

    How Reproducible Are Evaluation Conclusions? A Self-Audit of LLM-Inferred Prompt Structure

    [https://arxiv.org/abs/2609.30074](https://arxiv.org/abs/2609.30074)

    这项研究通过对LLM提示结构推断的自我审计发现，小规模提示集产生的模型评估排名中只有最差模型的位置是可靠的，而中间和头部模型的排名在不同重复实验中极不稳定。

    

    对LLM系统的评估通常在小规模提示集上取平均值，并以排名表的形式报告模型表现。我们提出这样一个问题：这样的排名表值得多少信任？并以基于LLM的提示结构推断作为案例研究：涵盖五个系列的八个开放模型变体，参数量从8B到675B，禁用缓存，持久化了293个原始中间表示。所测量的现象本身就是不稳定的：相同的调用无法可靠地恢复相同的结构，节点集Jaccard相似度均值从0.39到0.96不等，72%的提示-模型组合从未达到完美的节点集匹配。对评估过程本身的审计进一步削弱了其结论，这是我们的主要贡献。在基于提示的联合聚类自助法检验下，只有排名的底部是稳固的：可复现性最差的两个模型在99%和86%的重复实验中保持排名不变，中间四个模型仅占27%到48%，排名前两位的模型各占68%。因此，该排名表能够可靠地识别最差的模型，但无法可靠地确定其余模型的位置。

    arXiv:2609.30074v1 Announce Type: cross  Abstract: Evaluations of LLM systems routinely average over small prompt sets and report models as a ranked table. We ask how much confidence such a table deserves, using LLM-based prompt-structure inference as the case study: eight open model variants across five families and 8B to 675B parameters, caching disabled, 293 raw intermediate representations persisted. The measured phenomenon is unstable to begin with. Identical calls do not reliably recover identical structure, with mean node-set Jaccard from 0.39 to 0.96 and 72% of prompt-model cells never node-set-perfect. Auditing the evaluation weakens its conclusions further, and this is our main contribution. Under a joint cluster bootstrap over prompts, only the bottom of the ranking is firm: the two least reproducible models hold rank in 99% and 86% of replicates, the middle four in 27% to 48%, and the top two in 68% each, so the table identifies the worst model reliably but does not reliabl
    
[^20]: 双向评分：大语言模型能实现其自身无法可靠解析的MRS

    Scoring Both Directions: LLMs realize the MRS they cannot reliably parse

    [https://arxiv.org/abs/2609.30071](https://arxiv.org/abs/2609.30071)

    无需任何任务特定训练的大语言模型在MRS到文本的生成任务上显著超越专门训练的序列到序列系统，但它们只能单向“实现”这些MRS，却无法可靠地完成反向的文本到MRS解析。

    

    英语资源语法（ERG）是一部人工编写的英语计算语法。给定一个句子，其处理器ACE会生成一种称为最小递归语义（MRS）的形式意义表示，即由句子谓词及其论元构成的图。该语法是双向的，还能将MRS转换回英语句子。Hajdik等人（2019）利用ERG的树库为该生成任务（MRS到文本）构建了基准，并训练了序列到序列模型来解决该任务；而解析任务（文本到MRS）则可以在相同的句子上进行测试。我们重构了他们包含一万个句子的测试集，并在两个方向上对两个大语言模型——Claude Sonnet 4.5和Claude Opus 5——进行评分，将其与经训练的系统及ACE进行对比，且未做任何任务特定训练。给定一个MRS和三个示例，Opus生成的句子达到76.3 BLEU，比他们在72k语对上训练的系统（66.1 BLEU）高出十分，并与……（原文摘要在此处截断）

    arXiv:2609.30071v1 Announce Type: new  Abstract: The English Resource Grammar (ERG) is a hand-written computational grammar of English. Given a sentence, its processor, ACE, produces a formal meaning representation called Minimal Recursion Semantics (MRS): a graph of the sentence's predicates and their arguments. The grammar is bidirectional and can also turn an MRS back into an English sentence. \citet{hajdik2019} used the ERG's treebank to build a benchmark for that generation task, MRS to text, and trained sequence-to-sequence models to solve it. The parsing task, text to MRS, can be tested on the same sentences. We reconstruct their 10K-sentence test split, and score two large language models, Claude Sonnet~4.5 and Claude Opus~5, in both directions against their trained systems and against ACE, with no task-specific training. Given an MRS and three examples, Opus writes the sentence at 76.3 BLEU, ten points above their system trained on 72k pairs (66.1 BLEU), and comparable to thei
    
[^21]: 零数据自博弈预训练

    Self-Play Pretraining with Zero Data

    [https://arxiv.org/abs/2609.30063](https://arxiv.org/abs/2609.30063)

    该论文提出零数据自博弈预训练方法，让生成器提出由通用图灵机执行的程序来生成字节序列、学习器自回归预测这些序列，两个模型协同自博弈进化，将合成数据生成建模为受所罗门诺夫归纳启发的可计算结构空间搜索，从而实现完全不依赖人类数据、仅受算力限制的预训练。

    

    语言建模的进步一直依赖于在越来越多的数据上扩大预训练规模。然而，训练数据在很大程度上仍然是为模型精心策划的。一种更通用的预训练方式应当让模型学会自己生成对自身改进最有用的数据。这将提供一个实际上无边界的数据源，其限制来自算力而非人类知识。我们提出了零数据自博弈预训练，这是实现这一愿景的初步概念验证。我们的方法将合成数据生成视为对所有可计算结构空间的搜索，其灵感来自所罗门诺夫归纳。从随机初始化开始，两个模型协同学习：生成器提出程序，由通用图灵机解释执行以生成字节序列，而学习器则以自回归方式预测这些字节序列。学习器使用标准的交叉熵进行训练，而生成器……

    arXiv:2609.30063v1 Announce Type: new  Abstract: Advances in language modeling have been driven by scaling pretraining on ever more data. Yet, the training data is still largely curated on the model's behalf. A more general approach to pretraining would let the model learn to generate the data most useful for its own improvement. This would provide an effectively unbounded source of training data, limited by compute rather than human knowledge. We introduce Self-Play Pretraining with Zero Data, an initial proof-of-concept towards realizing this vision. Our procedure casts synthetic data generation as a search over the space of all computable structure, taking inspiration from Solomonoff induction. Starting from random initialization, two models learn in tandem: a generator proposes programs interpreted by a universal Turing machine, generating byte sequences, while a learner autoregressively predicts these byte sequences. The learner is trained with standard cross-entropy, while the ge
    
[^22]: 风格而非自我：表层线索解释大语言模型的零样本代码归因

    Style, Not Self: Surface Cues Explain Zero-Shot Code Attribution by Large Language Models

    [https://arxiv.org/abs/2609.30048](https://arxiv.org/abs/2609.30048)

    研究发现大语言模型在零样本识别自己代码时的表现并非源于真正的“自我认知”，而是可以由代码长度等表层风格特征所解释，因此对模型评审自我偏袒与合谋风险的担忧可能被夸大了。

    

    如果语言模型能够识别自己编写的代码，它可能会在充当评判者时偏袒该代码，而模型之间相互监控的情境可能导致合谋。我们在当前商业模型上对这种零样本能力进行了测试。五个大语言模型为MBPP、HumanEval和DS-1000生成解题方案，另有七个模型为MBPP生成方案，模型在四项任务中充当评估者：从一对方案中挑选出自己的方案、判断单个方案是否出自自己、识别两个方案中哪一个由指定模型编写，以及在盲测条件下评判代码质量。在单方案任务中，所有15个模型-基准组合的平衡准确率为49-58%，而原始准确率（38-67%）主要反映了模型宣称代码作者身份的难易程度。在成对任务中，14个评估者-对手组合的准确率与评估者自身方案更长这一因素的相关性高达r=0.93。对指定模型的归因在某些配对上取得成功，而在其他配对上则出现持续性的反转。一种基于规则的标准化方法……（摘要在此处被截断）

    arXiv:2609.30048v1 Announce Type: new  Abstract: If a language model can recognize code it wrote, it may favor that code as a judge, and instances of one model monitoring each other could collude. We test this zero-shot on current commercial models. Five LLMs generate solutions to MBPP, HumanEval, and DS-1000, seven more to MBPP, and models act as evaluators in four tasks: picking their own solution from a pair, judging whether a single solution is their own, identifying which of two solutions a named model wrote, and judging quality blind. In the single-solution task, balanced accuracy is 49-58% for all 15 model-benchmark combinations, while raw accuracy (38-67%) mostly reflects how readily a model claims authorship. In the pairwise task, accuracy across 14 evaluator-opponent combinations correlates at r=0.93 with how often the evaluator's solution is longer. Attribution to a named model succeeds on some pairs and is consistently inverted on others. A rule-based normalization that str
    
[^23]: 人工社会基准：合成研究的验证框架

    Artificial Societies Benchmark: A Validation Framework for Synthetic Research

    [https://arxiv.org/abs/2609.30030](https://arxiv.org/abs/2609.30030)

    该论文提出人工社会基准框架，通过涵盖内部、构建和外部效度的十一项测试评估九个语言模型生成的合成人口的可靠性，发现在单一领域的良好表现并不代表其他领域的保真度。

    

    一份合成调查能够重现平均答案，却可能歪曲人与人之间的差异、答案之间的相互关联方式，或者人们对条件变化的反应。我们引入人工社会基准，帮助研究人员评估合成人口是否能支持其预期的分析。该框架结合了覆盖内部效度、构建效度和外部效度的十一项测试，利用了二十个人类数据源并比较了九个语言模型。它将每种研究用途与其所需的证据联系起来，并检验结果如何随着我们提供的受访者信息而变化。重要的是，模型在某一领域的优异表现并不能确立其在其他领域的保真度。模型往往回答得过于一致、压缩回答量表、并改变特征之间的关系；此外，更丰富的个人档案信息会改善某些模型的预测效果，却会恶化另一些模型的预测效果。由此产生的记分卡可帮助研究人员识别哪些方面……

    arXiv:2609.30030v1 Announce Type: new  Abstract: A synthetic survey can reproduce the average answer while misrepresenting how people differ, how their answers relate to one another, or how they respond to changes in conditions. We introduce the Artificial Societies Benchmark to help researchers assess whether synthetic populations support their intended analyses. The framework combines eleven tests across internal, construct, and external validity, drawing on twenty human sources and comparing nine language models. It connects each research use to the evidence it requires and tests how results change with the information we supply about respondents. Importantly, strong performance in one domain does not establish fidelity in the others. Models often answer too consistently, compress response scales, and alter relationships between traits whilst richer profiles improve prediction for some models and worsen it for others. The resulting scorecard helps researchers identify which aspects 
    
[^24]: 跨厂商与跨版本测量模型行为的低成本检测方法

    Low-Cost Assays for Measuring Model Behavior Across Vendors and Releases

    [https://arxiv.org/abs/2609.30012](https://arxiv.org/abs/2609.30012)

    提出一种简单、廉价、可扩展且可复现的方法，通过在跨厂商模型面板上运行冻结的公开刺激任务，以精确匹配、LLM编码手册或插桩环境三种方式测量模型行为，单个模型成本仅需几美元。

    

    语言模型为人们提供建议、陪伴他们，并在他们睡觉时编写软件。然而测量它们的行为十分困难：行为需要在不同的模型、提示词和版本之间反复采样，其中大部分以非结构化文本形式存在，必须先编码才能统计，而且结果必须足够清晰且严谨，才能有意义地比较不同模型和厂商。为解决这些限制，我们提出了一种简单、廉价、可扩展且可复现的模型行为研究方法。每项研究都是一个冻结的、公开的刺激任务，以完全相同的方式在跨厂商的模型面板上运行，每个模型的成本仅为几美元甚至更低。每项研究根据行为所需的解释程度，以三种方式之一读取对话记录：对受限回复进行精确匹配；由LLM评审员应用编码手册，并按编码报告其与人类编码员的一致性；以及通过一个经过插桩的环境，独立于智能体所说的内容记录其实际行为。研究跨越四年运行……

    arXiv:2609.30012v1 Announce Type: cross  Abstract: Language models advise people, keep them company, and write software while they sleep. Measuring what they do is hard: behavior has to be sampled repeatedly across models, prompts and releases, most of it lives in unstructured text that has to be coded before it can be counted, and the result has to be legible and rigorous enough to meaningfully compare models and vendors. To address these constraints, we present a simple, cheap, scalable, and replicable model for studying model behavior. Each study is a frozen, public stimulus run identically on a cross-vendor panel, at a few dollars per model or less. Each reads its transcripts one of three ways, chosen by how much interpretation the behavior needs: exact match on a clamped reply, a codebook applied by LLM judges whose agreement with a human coder is reported per code, and an instrumented environment that records what an agent did independently of what it said. Run across four years 
    
[^25]: 基于领域自适应检索增强生成的金融服务自动化监管合规问答

    Automated Regulatory Compliance Question Answering in Financial Services with Domain-Adapted Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.30009](https://arxiv.org/abs/2609.30009)

    本文提出一条在 LegalBERT 上经三阶段领域自适应训练的检索器与 4 位量化紧凑生成器相结合的检索增强生成流水线，使可本地部署的小型模型也能在金融监管合规问答中给出有据可查、低幻觉的回答。

    

    金融机构在密集且频繁修订的规则手册下运营，正确回答合规问题不仅需要语言流畅性，还需要在权威文本中具备可验证的依据。大型语言模型对这一任务颇具吸引力，但企业实际能够本地部署的是紧凑型模型，而紧凑型模型会产生“义务幻觉”。我们研究经过精心领域自适应的检索增强生成（RAG）流水线能否弥合这一差距。我们的检索器基于 LegalBERT 分三个阶段构建：将“问题-段落”匹配重构为“前提-假设”重建任务的蕴含调优、采用批内负样本的对比调优，以及与 BM25 的分数级融合。我们的生成器是一个紧凑模型（2B–12B 参数），在 4 位量化下运行，或采用提示方式，或通过 LoRA 进行检索感知微调（RAFT）加以适配。在基于阿布扎比全球市场规则构建的问答基准 ObliQA 上（原文摘要此处截断）……

    arXiv:2609.30009v1 Announce Type: cross  Abstract: Financial institutions operate under dense, frequently amended rulebooks, and answering a compliance question correctly requires not only fluency but verifiable grounding in the authoritative text. Large language models are attractive for this task, yet the models that firms can realistically deploy on-premise are compact ones, and compact models hallucinate obligations. We study whether a carefully domain-adapted retrieval-augmented generation pipeline closes that gap. Our retriever is built in three stages on top of LegalBERT: entailment tuning that recasts question--passage matching as premise--hypothesis reconstruction, contrastive tuning with in-batch negatives, and score-level fusion with BM25. Our generator is a compact model (2B--12B parameters) served under 4-bit quantization, either prompted or adapted with retrieval-aware fine-tuning (RAFT) through LoRA. On ObliQA, a question-answering benchmark built from the Abu Dhabi Glob
    
[^26]: VietPrism：一个具有多样化方言和语码转换的大规模越南语语音与深度伪造语料库

    VietPrism: A large-scale Vietnamese speech and deepfake corpus with diverse dialects and code-switching

    [https://arxiv.org/abs/2609.30005](https://arxiv.org/abs/2609.30005)

    VietPrism是首个大规模越南语语音与深度伪造语料库，包含993.4小时真实语音和超3,100小时伪造语音，同时涵盖转录文本、说话人身份、五大方言组及越南语-英语语码转换，并通过转录与说话人双匹配的真实-伪造语音对实现受控评估。

    

    越南语语音研究长期受到资源限制，这些资源将自动语音识别与说话人、方言、语码转换和深度伪造分析相互割裂。我们推出了VietPrism，这是一个开放的多领域语料库，首次在大规模层面将这些维度整合在一起：包含来自1,262名经身份验证的说话人的993.4小时、403,941条真实语音，来源覆盖8,388个现实世界视频。据我们所知，这是首个同时提供文本转录、一致的说话人身份、五个方言组以及自然出现的越南语-英语语码转换的大规模越南语语料库，其中语码转换内容按时长计占语料库的近一半。我们进一步利用四个开源及商业语音合成系统创建了超过3,100小时的伪造语音。每条伪造语音均以经身份验证的说话人参考为条件生成，并与转录和说话人都匹配的真实语音配对，从而实现了独特的受控评估，减少了词汇和身份方面的干扰因素。

    arXiv:2609.30005v1 Announce Type: new  Abstract: Vietnamese speech research is constrained by resources that isolate automatic speech recognition from speaker, dialect, code-switching, and deepfake analysis. We introduce VietPrism, an open, multi-domain corpus that brings these dimensions together at scale: 993.4 hours and 403,941 bona fide utterances from 1,262 verified speakers across 8,388 real-world videos. To our knowledge, it is the first large-scale Vietnamese corpus to jointly provide transcripts, consistent speaker identities, five dialect groups, and naturally occurring Vietnamese--English code-switching, which constitutes nearly half of the corpus by duration. We further create over 3.1K hours of spoof speech with four open-source and commercial synthesis systems. Every spoof is conditioned on a verified speaker reference and paired with a transcript- and speaker-matched bona fide utterance, enabling unique controlled evaluation with reduced lexical and identity confounds. Z
    
[^27]: Augur：用于预演产品与政策变更反应的合成决策实验室

    Augur: A Synthetic Decision Lab for Rehearsing Reactions to Product and Policy Changes

    [https://arxiv.org/abs/2609.29952](https://arxiv.org/abs/2609.29952)

    提出了离线决策预演系统 Augur 和包含五十个真实事件的 Gold-50 基准，核心发现是前沿云端模型与离线开源模型之间的大部分表现差距源于评估设定不充分而非能力差异。

    

    在产品或政策变更正式上线之前，关键的问题是人们将如何对其做出反应。Augur 能够离线预演这种反应：它从变更文档中构建类型化知识图谱，填充基于真实依据的角色市场，模拟交互过程，并返回一份可审计的决策备忘录，在五种行动中给出推荐。我们构建了 Gold-50 数据集——包含五十个真实产品与政策事件，其现实结果已知并依据公开记录进行裁定——据此对五选一的发布判定进行评分。我们的核心发现是方法论层面且是否定性的：前沿云端模型与我们微调并离线部署的开源权重模型之间所测得的大部分差距，可归因于评估设定不够充分，而非能力差异。我们通过三种方式证明了这一点。首先，仅提示词包络本身就能主导得分：在保持模型权重、案例和评分器不变的情况下，一个系统——Qwen3-32B 上的 LoRA-SFT 适配器——的得分从 0（摘要在此处被截断）

    arXiv:2609.29952v1 Announce Type: new  Abstract: Before a product or policy change ships, the question that matters is how people will react to it. Augur rehearses that reaction offline: it builds a typed knowledge graph from the change documents, populates a grounded persona market, simulates the interaction, and returns an auditable decision memo recommending one of five actions. We assemble Gold-50, fifty real product and policy episodes whose real-world outcome is known, adjudicated against the public record, and score the five-way release verdict against it.   Our central finding is methodological and negative: most of the measured gap between frontier cloud models and open-weight models we fine-tune and serve offline is attributable to an under-specified evaluation, not a difference in capability. We show this three ways. First, the prompt envelope alone can dominate the score: holding weights, cases and scorer fixed, one system -- a LoRA-SFT adapter on Qwen3-32B -- swings from 0
    
[^28]: 面向长文档问答的视觉语言模型流水线实证研究

    An Empirical Study of VLM Pipelines for Long-Document QA

    [https://arxiv.org/abs/2609.29933](https://arxiv.org/abs/2609.29933)

    该研究在两个长文档问答基准上系统评估了VLM的部署选择，发现六工具智能体流水线只有在回答模型足够大时才能超越静态页面输入，且其优势随基准和阅读器的不同而变化。

    

    视觉语言模型（VLM）越来越多地被用于长文档处理，其输入将文本与图表、表格、插图和复杂版式结合在一起。部署这类模型意味着需要做出多项选择：如何将文档提供给模型、当仅发送部分页面时应使用哪种检索器，以及让模型以智能体方式运行还是作为静态流水线运行。我们在两个长文档问答基准上，使用前沿API模型和开源权重VLM研究了这些选择。首先，在MMLongBench-Doc上，我们提出的包含页面、表格、插图和搜索调用的六工具智能体只有在回答用VLM足够大时才能体现价值：使用Qwen3.5-4B和9B时它落后于静态页面输入，使用Qwen3.5-27B时持平，而使用Sonnet 4.5时则领先。在LongDocURL上，它在所有阅读器下均与静态输入持平或更优。它相对最强静态流水线的优势在MMLongBench-Doc上以前沿阅读器最为明显，而在LongDocURL上则缩小至噪声范围内。其次，检索……（摘要原文在此处截断）

    arXiv:2609.29933v1 Announce Type: cross  Abstract: Vision-Language Models (VLMs) are increasingly used for long-document processing, where the inputs combine text with charts, tables, figures, and complex layouts. Deploying them means choosing how to feed the document to the model, which retriever to use when only a subset of pages is sent, and whether to run the model agentically or as a static pipeline. We study these choices on two long-document QA benchmarks with both frontier API and open-weight VLMs. First, on MMLongBench-Doc our six-tool agent with page, table, figure, and search calls pays off only once the answering VLM is large enough: with Qwen3.5-4B and 9B it trails static page input, with Qwen3.5-27B it draws level, and with Sonnet 4.5 it leads. On LongDocURL it is level with or ahead of static input at every reader. Its lead over the strongest static pipeline is clearest with the frontier reader on MMLongBench-Doc and narrows to within noise on LongDocURL. Second, retriev
    
[^29]: 文化分歧保持性：诊断大语言模型模拟调查人群中的文化扁平化与文化漫画化现象

    Cultural Divergence Preservation: Diagnosing Flattening and Caricature in LLM-Simulated Survey Populations

    [https://arxiv.org/abs/2609.29928](https://arxiv.org/abs/2609.29928)

    本文提出一种基于一次性人工校准的轻参考诊断方法CDP，用于检测LLM模拟跨文化调查时出现的“文化扁平化”（跨国差异被抹平）与“文化漫画化”（跨国差异被夸大）问题。

    

    大语言模型（LLM）越来越多地被用作合成调查受访者，以估计人群的回答分布。在跨文化调查模拟中，评估不仅应考察各国内部分布的保真度，还应考察各国之间的差异是否得到保留。然而，现有的基于距离的度量方法（如Jensen–Shannon散度JSD）无法直接捕捉这种跨国差异。为解决这一局限，我们提出了文化分歧保持性（Cultural Divergence Preservation, CDP），这是一种基于一次性人工校准的轻参考诊断方法。CDP将跨国分歧的减少识别为“文化扁平化”，将跨国分歧的增加识别为“文化漫画化”。为评估CDP，我们在四种LLM骨干模型、三种基于人设的提示方法以及两个调查领域（世界价值观调查WVS和大五人格测试）上进行了实验。结果揭示了系统性差异。

    arXiv:2609.29928v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used as synthetic survey respondents to estimate population response distributions. In cross-cultural survey simulation, evaluations should assess not only distributional fidelity within countries but also whether differences across countries are preserved. However, existing distance-based metrics such as Jensen--Shannon divergence (JSD) do not directly capture such cross-country differences. To address this limitation, we introduce Cultural Divergence Preservation (CDP), a reference-light diagnostic based on a one-time human calibration. CDP identifies reduced cross-country divergence as cultural flattening and increased divergence as cultural caricature. To evaluate CDP, we conduct experiments across four LLM backbones, three persona-based prompting methods, and two survey domains, the World Values Survey (WVS) and the Big Five Personality Test. The results reveal a systematic discrepancy
    
[^30]: MILO：基于分块低秩压缩的高效多示例上下文学习

    MILO: Efficient Many-shot In-Context Learning with Block-wise Low-rank Compression

    [https://arxiv.org/abs/2609.29913](https://arxiv.org/abs/2609.29913)

    提出MILO压缩框架，利用基于信息熵动态分配秩预算的分块低秩压缩策略来压缩多示例上下文学习中的KV缓存，从而解决推理内存瓶颈。

    

    多示例上下文学习（ICL）使大语言模型（LLM）能够通过条件化于数千个演示示例来适应复杂任务，但这一范式将推理效率的瓶颈转移到了键值缓存（KV cache）的内存上。由于KV缓存呈线性扩展的特性，存储这些中间张量对在线服务和设备端部署而言已成为一大核心挑战。为解决这一问题，我们提出了一种新颖的压缩框架MILO，它利用了多示例上下文中固有的低秩冗余。具体而言，MILO采用分块低秩压缩策略，在块的粒度上压缩KV缓存，其中每个块包含多个多示例。此外，为应对不同块之间异构的上下文密度，MILO基于信息熵动态分配秩预算，从而保持关键块的保真度。

    arXiv:2609.29913v1 Announce Type: new  Abstract: Many-shot in-context learning (ICL) enables large language models (LLMs) to adapt to complex tasks by conditioning on thousands of demonstration examples, but this paradigm shifts the inference efficiency bottleneck to the key-value (KV) cache memory. Due to the linear scaling behavior of the KV cache, storing these intermediate tensors has become a paramount challenge for both online serving and on-device deployment. To address this issue, we propose a novel compression framework, termed MILO, that exploits the low-rank redundancy inherent in many-shot contexts. Specifically, MILO features a block-wise low-rank compression strategy that compresses the KV cache at the block granularity, where each block contains multiple many-shot examples. Furthermore, to handle the heterogeneous context density across different blocks, MILO dynamically allocates rank budgets based on the information entropy, preserving the fidelity of critical blocks w
    
[^31]: 基于上下文化词表示的形态丰富语言句法分析多任务学习

    Multi-Task Learning by using Contextualized Word Representations for Syntactic Parsing of a Morphologically Rich Language

    [https://arxiv.org/abs/2609.29855](https://arxiv.org/abs/2609.29855)

    本文通过将短语结构树库转换为依存树库、设计统一的序列标注方案、在2.2亿词元语料上训练上下文化词表示，并结合单任务与多任务学习范式，在形态丰富的乌尔都语的成分句法分析和依存句法分析上取得了最先进的结果。

    

    我们针对乌尔都语（一种形态丰富的语言）的句法分析难题展开研究，在成分句法分析和依存句法分析两方面均取得了最先进的结果。本文提供了四项主要贡献：1）通过开发特定语言的中心词判定规则和短语到依存标签的映射规则，将CLE-UTB短语结构树库转换为依存树库；2）提出一种新颖的序列标注方案，将句法分析任务转化为统一表示形式；3）在从网络收集的2.2亿词元的大规模乌尔都语语料库上训练上下文化词表示；4）基于单任务学习和多任务学习两种学习范式构建句法分析框架。此外，还应用了若干后处理规则来提升自动转换的依存结构树库的质量。所提出的序列标注方案使得能够使用共享架构来学习句法结构。

    arXiv:2609.29855v1 Announce Type: cross  Abstract: We address the challenge of syntactic parsing for Urdu, a morphologically rich language, and present state-of-the-art results for both constituency and dependency parsing. This paper offers four major contributions: 1) the conversion of the CLE-UTB phrase structure treebank into a dependency treebank by developing language-specific head-word and phrase-to-dependency label mapping rules; 2) a novel sequence labeling scheme that transforms the parsing task into a unified representation; 3) the training of contextualized word representations on a large 220 million tokens Urdu corpus collected from the web; and 4) development of parsing framework using two learning paradigms, single-task and multi-task learning. Several post-processing rules are applied to improve the quality of the automatically converted dependency structure treebank. The proposed sequence labeling scheme enables the use of a shared architecture that learns the syntactic
    
[^32]: 已编码但未解码：LLM句法三层级差距的层级定位证据

    Encoded but Not Decoded: Layer-Localized Evidence for a Three-Level Gap in LLM Syntax

    [https://arxiv.org/abs/2609.29848](https://arxiv.org/abs/2609.29848)

    该论文提出一个三层级评估框架（行为部署、LM头读出、探针可恢复性），在七个模型、三种语言上首次定位了LLM句法中“结构已编码但未被解码使用”的层级化差距，且该差距集中于最近名词启发式会失效的主语控制结构。

    

    语言模型在句法测试中失败可能源于两种截然不同的原因：一是没有编码相关结构，二是编码了该结构但在输出时未能使用。仅靠行为评估无法区分这两种情况。我们提出了一个三层级评估框架（行为部署、语言模型头读出和探针可恢复性），在相同的二元决策下对相同的项目进行测量。基于一个紧凑的三语（英语、中文、德语）控制依存基准测试，我们发现总体上在七个模型和全部三种语言中，探针可恢复性超过或等于语言模型头读出，而后者又超过或等于行为部署。在全部14个（模型、任务）条件下，可恢复性盈余从不为负。这种脱节集中在主语控制结构上，此时“最近名词”启发式会给出错误答案。最大的单一差距（0.653）出现在Qwen3-0.6B Instruct的问答任务中。该差距在Qwe...（原文摘要在此处被截断）

    arXiv:2609.29848v1 Announce Type: new  Abstract: A language model can fail a syntactic test in two distinct ways: by not encoding the relevant structure, or by encoding it but failing to use it at the output. Behavioral evaluation alone cannot tell these apart. We propose a three-level evaluation framework (behavioral deployment, LM-head readout, and probe recoverability) measured on the same items under the same binary decision. Using a compact trilingual (English, Chinese, German) control-dependency benchmark, we find that probe recoverability exceeds or equals LM-head readout, which in turn exceeds or equals behavioral deployment, across seven models and all three languages in the aggregate. The recoverability surplus is never negative across all 14 (model, task) conditions. The disconnect concentrates in subject-control, where a nearest-noun heuristic gives the wrong answer. The single largest gap (0.653) appears on Qwen3-0.6B Instruct in question answering. The gap persists at Qwe
    
[^33]: 你的Transformer可以同时持有两个想法：LLM中线性叠加的证据

    Your Transformer Can Hold Two Thoughts at Once: Evidence of Linear Superposition in LLMs

    [https://arxiv.org/abs/2609.29845](https://arxiv.org/abs/2609.29845)

    本文提出“叠加线性假说”，证明当不同文本流的输入线性组合时，LLM会输出各自下一词元分布的叠加，这是Transformer架构的内在属性而非训练的涌现结果，可通过轻量级微调恢复，并借助引导解码实现同时生成两个连贯的文本流。

    

    尽管大型语言模型（LLM）依赖于高度非线性的组件，但在这项工作中我们证明它们表现出根本的线性特性：当来自不同文本流的输入被线性组合时，模型输出的结果是各个下一词元分布的叠加。我们将这一现象称为“叠加线性假说”。我们提供的证据表明，叠加是Transformer架构的内在属性，而非训练产生的涌现结果；事实上，我们观察到随着预训练的进行，叠加现象反而趋于减弱。然而，我们证明通过轻量级微调可以大幅恢复这种线性特性，显著缩小预测的下一词元分布与各单独下一词元分布平均值之间的散度。最后，我们引入了一种引导解码过程，能够解耦叠加的输出，从而实现同时生成两个连贯的文本流。

    arXiv:2609.29845v1 Announce Type: cross  Abstract: While Large Language Models (LLMs) rely on highly non-linear components, in this work we demonstrate that they exhibit fundamental linearity: when inputs from distinct text streams are linearly combined, the model outputs a superposition of the individual next-token distributions. We term this the \textit{Superposition Linearity Hypothesis}. We provide evidence that superposition is an intrinsic property of the Transformer architecture rather than an emergent consequence of training; in fact, we observe that it tends to diminish as pretraining progresses. However, we demonstrate that linearity can be substantially restored through lightweight fine-tuning, significantly reducing the divergence between the predicted next-token distribution and the average of the individual next-token distributions. Finally, we introduce a guided decoding procedure that disentangles superposed outputs, enabling the simultaneous generation of two coherent 
    
[^34]: PUBG Ally：作为AI队友的对话式具身智能体

    PUBG Ally: A Conversational Embodied Agent as an AI Teammate

    [https://arxiv.org/abs/2609.29837](https://arxiv.org/abs/2609.29837)

    该论文提出了PUBG Ally，一个面向《绝地求生》的语音对话式具身AI队友，通过将语言模型智能体的工具使用与实时游戏控制相结合，在严格延迟约束下感知动态游戏世界、与玩家自然交流并同步执行移动、战斗等游戏行动。

    

    我们推出了PUBG Ally，一个面向《绝地求生：大逃杀》(PUBG: BATTLEGROUNDS) 的具身智能体，它能够进行推理、自主行动，并作为支持语音交互的队友与玩家并肩作战。构建这样的队友需要结合两种困难的能力：它必须在严格的延迟约束下感知并响应不断变化的游戏世界，同时与玩家自然交互，使其语音与行动保持同步。因此，Ally将智能体的工具使用能力与实时游戏控制相结合。一个语言模型智能体通过受控接口来查看游戏信息、理解玩家语音、维护上下文、决定说什么，并发出高层动作选择，用以引导更快的控制层执行移动、战斗和恢复等操作。由于玩家和Ally的语音与行动会不断相互影响并塑造比赛进程，训练需要来自真实对局的数据。因此，我们在近3.9万场对局中收集了数据……（摘要在此处被截断）

    arXiv:2609.29837v1 Announce Type: new  Abstract: We introduce PUBG Ally, an embodied agent for PUBG: BATTLEGROUNDS that can reason, act autonomously, and play alongside players as a voice-enabled teammate. Building such a teammate requires combining two difficult capabilities: it must perceive and respond to a constantly changing game world under strict latency constraints while interacting naturally with players, keeping its speech synchronized with its actions. Ally therefore combines agentic tool use with real-time game control. A language-model agent uses a controlled interface to inspect game information, interpret player speech, maintain context, decide what to say, and issue high-level action choices that steer a faster control layer for movement, combat, and recovery. Because the player's and Ally's speech and actions continually shape each other and the course of the match, training requires data from actual gameplay. We therefore collect data across nearly 39k sessions in whi
    
[^35]: ChunkRank：面向大语言模型流水线的模型感知文本分块与弃答感知的答案选择

    ChunkRank: Model-Aware Text Chunking and Abstention-Aware Answer Selection for LLM Pipelines

    [https://arxiv.org/abs/2609.29828](https://arxiv.org/abs/2609.29828)

    ChunkRank开源库基于目标模型的分词器与上下文窗口实现自动防溢出的精确文本分块，并发现基于内容的答案排序无法稳定胜过直接采用首个非空答案，其根源在于阅读器对无答案分块的弃答行为。

    

    我们提出了ChunkRank，一个开源Python库，它根据目标模型的分词器和上下文窗口推导文本分块边界，并在每个分块独立生成的候选答案中进行选择。该库附带一个经过验证的模型注册表，涵盖15个提供商的90个模型，提供六种答案选择方法，且仅需三个核心依赖。在分块方面，ChunkRank能根据模型名称自动避免上下文窗口溢出，而基于字符的分块器则会出现溢出或浪费预算的问题；一项覆盖11种语言的保真度研究说明了为什么精确的token预算在英语之外的语言中同样重要。在答案选择方面，我们报告了一个阴性结果：在NaturalQuestions、TriviaQA和HotpotQA数据集上，无论使用抽取式还是生成式阅读器，没有任何基于内容的排序器能够稳定地胜过直接采用第一个非空答案。其根本原因在于阅读器对不包含答案的分块会采取弃答行为，而非答案在分块中的位置。一个长上下文基线实验表明，分块方法的性能与单次调用检索相当……

    arXiv:2609.29828v1 Announce Type: new  Abstract: We present ChunkRank, an open-source Python library that derives chunk boundaries from a target model's tokenizer and context window, and selects an answer among candidates produced independently per chunk. It ships a validated registry of 90 models across 15 providers and six answer-selection methods, and needs only three core dependencies. For chunking, ChunkRank avoids context-window overflow automatically from the model name, whereas character-based splitters overflow or waste the budget, and a fidelity study across 11 languages shows why token-exact budgets matter beyond English. For answer selection we report a negative result: on NaturalQuestions, TriviaQA and HotpotQA, with extractive and generative readers, no content-based ranker reliably beats taking the first non-empty answer. The reason is reader abstention on chunks that lack the answer, not answer position. A long-context baseline shows that chunking matches single-call re
    
[^36]: CORDIAL：从少量标签校准LLM的有序输出

    CORDIAL: Calibrating Ordinal LLM Outputs from Few Labels

    [https://arxiv.org/abs/2609.29807](https://arxiv.org/abs/2609.29807)

    CORDIAL提出了一种仅用五个可解释参数、仅需少量标签（如20个）即可校准LLM有序输出分布的方法，在绝大多数实验设置中取得最低对数损失，同时支持跨任务先验学习和多LLM融合。

    

    大型语言模型（LLM）可以将文本转换为有序量表上的概率分布，但该分布是一种含噪声的测量：可能出现饱和、压缩或夸大，并偏向某个一致的方向。我们提出了CORDIAL方法，它将模型的输出视为对真实标签的含噪读数，并使用一个包含五个可解释参数的信道对其进行校正。该信道足够小巧，因此其后验分布可以仅凭少量标签进行估计，并且我们证明了由此得到的校准方法能保持一阶随机序关系。在Amazon评论和CMU-MOSEI转录文本上使用四个LLM进行的实验中，在5到100个标签的80个设置中的76个里，CORDIAL在九种校准器中取得了最低的对数损失；使用20个标签和主要的7B读取模型时，它就能匹敌使用28-54个标签的最强基线方法。同样的后验分布还使我们能够从其他任务中学习先验分布并融合多个LLM。像Dirichlet校准这类无限制的校准器，只有在标签数量充足时才能超过它。

    arXiv:2609.29807v1 Announce Type: new  Abstract: A large language model (LLM) can turn a text into a distribution over an ordered scale, but that distribution is a noisy measurement: saturated, compressed or exaggerated, and biased in a consistent direction. We propose CORDIAL, which treats the model's output as a noisy reading of the true label and corrects it with a channel of five interpretable parameters. The channel is small enough for its posterior to be averaged from a handful of labels, and we prove that the resulting calibration preserves first-order stochastic order. On Amazon reviews and CMU-MOSEI transcripts with four LLMs, CORDIAL has the lowest log loss among nine calibrators in 76 of 80 settings with 5 to 100 labels; with 20 labels and the main 7B reader, it matches the strongest baseline using 28-54 labels. The same posterior lets us learn priors from other tasks and fuse several LLMs. Unrestricted calibrators such as Dirichlet calibration overtake it only as the calibr
    
[^37]: 学习产生具有科学影响力的研究创意

    Learning to Ideate for Scientific Impact

    [https://arxiv.org/abs/2609.29802](https://arxiv.org/abs/2609.29802)

    该论文提出以引文归一化的科学影响力作为延迟反馈信号，从超10万篇论文构建数据集并训练目标条件奖励模型，再通过监督微调与强化学习对齐创意生成器，使大语言模型能够生成具有更高预期科学影响力的研究创意。

    

    科学构思（ideation）越来越多地由大语言模型辅助完成，但现有的构思系统通常是在诸如新颖性、清晰性和可行性等可以立即评判的代理指标上进行训练和评估的。这留下了一个悬而未决的问题：科学成果被学界接受的延迟信号能否用作反馈，引导模型朝着具有更高预期影响力的研究方向发展。我们使用引文归一化影响力作为学术接受度的一个有噪声但可扩展的代理指标来研究这个问题。我们从超过10万篇计算机科学论文中构建了一个大规模数据集，通过提取以研究目标为条件的创意描述，并为每篇论文分配一个有序的、按年份归一化的引用标签。随后，我们训练一个以目标为条件的奖励模型，从“研究目标-创意”对中预测引用影响力标签，并利用该奖励通过监督微调和强化学习对创意生成器进行对齐。为了减少循环性，我们评估生成的创意……（原文摘要到此截断）

    arXiv:2609.29802v1 Announce Type: new  Abstract: Scientific ideation is increasingly mediated by large language models, but current ideation systems are usually trained and evaluated on immediately judgeable proxies such as novelty, clarity, and feasibility. This leaves open whether delayed signals of scientific uptake can be used as feedback for steering models toward research directions with higher expected \emph{impact}. We study this question using citation-normalized impact as a noisy but scalable proxy for scholarly uptake. We construct a large-scale dataset from over 100K computer science papers by extracting goal-conditioned idea descriptions and assigning each paper an ordinal, year-normalized citation label. We then train a goal-conditioned reward model to predict citation-impact labels from research goal and idea pairs, and use this reward to align an idea generator through supervised fine-tuning followed by reinforcement learning. To reduce circularity, we evaluate generate
    
[^38]: 面向低资源语音识别的自适应Fisher白化互协方差方法

    Adaptive Fisher-Whitened Cross-Covariance for Low-Resource Speech Recognition

    [https://arxiv.org/abs/2609.29800](https://arxiv.org/abs/2609.29800)

    本文提出将Fisher白化互协方差分析（FCCA）应用于语音基础模型的参数高效微调，并通过非对称耦合（AC-FCCA）和自适应秩（AR-FCCA）两个扩展在固定参数预算下提升低资源语言的语音识别性能。

    

    将多语言语音基础模型适配到低资源语言仍然十分困难，尤其是对于预训练中代表性不足的语言。虽然参数高效微调（PEFT）降低了适配大型模型的成本，但LoRA等传统方法依赖通用的低秩参数化，并未显式利用下游任务信息来定义适配子空间。为了探究任务信息驱动的PEFT能否更好地支持低资源语音识别（ASR），我们将Fisher白化互协方差分析（FCCA）应用于Whisper和Qwen3-ASR，并提出了两个互补的扩展：非对称耦合FCCA（AC-FCCA），利用结构化的跨层共享机制；以及自适应秩FCCA（AR-FCCA），在固定参数预算下跨投影矩阵重新分配适配容量。在受控的多语言实验中，研究者在预训练中代表性不足的语言上对这些方法进行了评估（摘要原文在此处被截断）。

    arXiv:2609.29800v1 Announce Type: new  Abstract: Adapting multilingual speech foundation models to low-resource languages remains difficult, especially for languages that are poorly represented during pre-training. While parameter-efficient fine-tuning (PEFT) reduces the cost of adapting large models, conventional approaches such as LoRA rely on generic low-rank parameterizations and do not explicitly use downstream task information to define the adaptation subspace. To investigate whether task-informed PEFT can better support low-resource ASR, we apply Fisher-Whitened Cross-Covariance Analysis (FCCA) to Whisper and Qwen3-ASR, and introduce two complementary extensions: Asymmetric-Coupled FCCA (AC-FCCA), which exploits structured cross-layer sharing, and Adaptive-Rank FCCA (AR-FCCA), which reallocates adaptation capacity across projection matrices under a fixed parameter budget. Under controlled multilingual experiments, we evaluate these approaches on languages that are poorly represe
    
[^39]: 面向加纳语言青少年健康传播的自动语音识别（ASR）基准测试与领域自适应

    Benchmarking and Domain Adaptation of Automatic Speech Recognition (ASR) for Adolescent Health Communication in Ghanaian Languages

    [https://arxiv.org/abs/2609.29798](https://arxiv.org/abs/2609.29798)

    本文对三种加纳语言中面向青少年健康传播的多个ASR系统进行了基准测试，并通过在加纳圣经语料库上微调紧凑型Qwen3-ASR-0.6B模型实现领域自适应，显著降低了所有语言的词错误率。

    

    本文对三种加纳语言（契维语、达格巴尼语和埃维语）的青少年健康传播自动语音识别（ASR）进行了端到端研究。这项工作分三个相互关联的阶段进行：首先，我们在通用领域圣经语料库和青少年性与生殖健康（ASRH）领域ASR数据集上，使用字符错误率和词错误率（CER、WER）对五个ASR系统（三个针对特定语言的Wav2Vec2模型和两个多模态大语言模型Gemma 3n和Gemma 4）进行了基准测试。其次，在基准测试结果的指导下，我们进行了有监督的领域自适应：尽管Gemma 4是最强的零样本候选模型，但对其微调在计算上被证明不可行，因此我们转向了紧凑型Qwen3-ASR-0.6B，在大型加纳圣经语料库（约9万个样本）上进行微调，并严格在留出的、人工采集的领域内音频上进行评估。微调降低了所有语言的WER，其中埃维语的改善最为显著（WER从109.3%降至64.8%……原文在此处截断）

    arXiv:2609.29798v1 Announce Type: cross  Abstract: This paper presents an end-to-end study of automatic speech recognition (ASR) for adolescent health communication in three Ghanaian languages (Twi, Dagbani, and Ewe). The work proceeds in three connected stages; First, we benchmark five ASR systems (three language-specific Wav2Vec2 models and two multimodal LLMs, Gemma 3n and Gemma 4) on a general-domain Bible corpus and a Youth Adolescent Sexual and Reproductive Health (ASRH) Domain ASR dataset, using Character and Word Error Rate (CER, WER). Second, guided by the benchmark, we perform supervised domain adaptation: although Gemma 4 was the strongest zero-shot candidate, fine-tuning it proved computationally infeasible, so we pivoted to the compact Qwen3-ASR-0.6B, fine-tuned on a large Ghana Bible corpus (~90k samples) and evaluated strictly on held-out human-collected in-domain audio. Fine-tuning reduced WER on every language, most dramatically for Ewe (WER from 109.3% to 64.8%, a dro
    
[^40]: TimeBraid：统一时间序列与语言的理解与预测模型

    TimeBraid: Unifying Time Series and Language for Understanding and Forecasting

    [https://arxiv.org/abs/2609.29792](https://arxiv.org/abs/2609.29792)

    TimeBraid通过交错的全身残差注意力层将预训练语言模型与时间序列基础模型对齐融合，在共享表示空间中同时实现时间序列与语言的理解和生成，并凭借220万序列-文本对与490万指令样本的监督在理解与预测任务上取得优异表现。

    

    我们提出了TimeBraid，这是一系列统一的时间序列与语言模型，通过交错的全身残差注意力层将预训练语言模型与预训练时间序列基础模型对齐。每个模型从一侧继承知识、指令遵循与推理能力，从另一侧获得连续信号感知与零样本预测能力，并在共享表示空间中将两者融合，使两种模态均能被理解与生成。我们研究了使这种统一建模得以实现的设计选择：在何处对齐两个表示空间、如何将语言扎根于时间结构、如何平衡理解与生成，以及如何保持联合优化的稳定性。由此得到的方案结合了面向多样化时间序列与文本任务的统一提示方案、稳定的联合训练，以及来自220万条精选序列-文本对和490万条指令微调样本的监督。在各项基准测试中……

    arXiv:2609.29792v1 Announce Type: cross  Abstract: We present TimeBraid, a series of unified time-series and language models that align pretrained language models and pretrained time-series foundation models through interleaved global residual attention layers. Each model inherits knowledge, instruction following, and reasoning from one side, continuous-signal perception and zero-shot forecasting from the other, and fuses the two in a shared representation space where both modalities are understood and generated. We study the design choices that make such unified modeling work: where to align the two representation spaces, how to ground language in temporal structure, how to balance understanding with generation, and how to keep joint optimization stable. The resulting recipe combines a unified prompting scheme for diverse time-series and text tasks, stabilized joint training, and supervision from 2.2M curated series--text pairs and 4.9M instruction-tuning samples. Across benchmarks sp
    
[^41]: JEV与LLM作为评分准则裁判：更便宜、更快速，且在同样的地方犯错

    JEV vs. LLMs as Rubric Judges: Cheaper, Faster, and Wrong in the Same Places

    [https://arxiv.org/abs/2609.29769](https://arxiv.org/abs/2609.29769)

    研究表明，无需生成文本的类型化分类器Jev可作为LLM评分准则裁判的低成本替代方案：准确率与LLM裁判无显著差异但成本仅为其1/29至1/325，且两者在分级准则上倾向于犯相似错误（均低于人工评分等级）。

    

    我们探究Jev——一个无需生成文本、直接返回各允许答案概率的类型化分类器——能否取代LLM评分准则裁判。我们在来自七个基准的九个面板上，将其与三个flash级LLM裁判进行对比，并为每个裁判提供完全相同的准则文本。在27组配对比较中，Jev的准确率仅在8组上与LLM裁判存在显著差异：其优势主要体现在二元准则上，仅在分级准则上落后，其余多数比较结果不明确。对九个面板累计计算，每条准则调用一次的LLM裁判成本是Jev的29至325倍，耗时是Jev的30至220倍。在分级准则上，四个裁判彼此之间的一致性都高于其与真实标签的一致性，且大多给出低于人工评分者的等级。一种可能的观察性解释是：人工评分者遵循了我们的准则文本中未写明的量表使用惯例。Jev的置信度在大多数面板上能够对其自身错误进行排序，这应该能使更便宜的错误筛选成为可能。

    arXiv:2609.29769v1 Announce Type: new  Abstract: We ask whether Jev, a typed classifier that returns probabilities over permitted answers without generating text, can replace an LLM rubric judge. We compare it with three flash-tier LLM judges on nine panels drawn from seven benchmarks, giving every judge identical criterion texts. Jev's accuracy differs significantly from an LLM judge's in only 8 of 27 paired comparisons, ahead mostly on binary criteria and behind only on graded ones, and most of the other comparisons are inconclusive. Summed over the nine panels, the LLM judges, called once per criterion, cost 29 to 325 times as much as Jev and took 30 to 220 times as long. On graded criteria all four judges agree more with one another than with the labels and mostly assign lower levels than the raters. One of several observational accounts is that raters followed scale conventions our criterion texts omit. Jev's confidence ranks its own errors on most panels, which should make a chea
    
[^42]: C3M：面向长程任务的跨会话多模态记忆维护

    C3M: Cross-Session Multimodal Memory Maintenance for Long-Horizon Tasks

    [https://arxiv.org/abs/2609.29735](https://arxiv.org/abs/2609.29735)

    C3M提出了一种跨会话多模态记忆维护框架，通过关系感知更新在有界活动索引中整合安全冗余并保留互补与不兼容记录，并在查询时以预算化路由展开关联源证据，为长程任务提供了紧凑且保留来源信息的记忆组织。

    

    长程任务需要在有限的、与查询无关的记忆预算下保存并在之后恢复跨会话证据。现有的压缩方法可能会丢弃细粒度的视觉线索，或会将语义相似但不兼容的观察混淆在一起。我们提出了C3M，这是一种跨会话多模态记忆组织方式，它在持久的源文本-图像证据之上维护一个有界的活动索引。基于关系的更新在整合安全冗余的同时，保留互补和不兼容的记录。在查询时，预算化的路由机制选择有用的索引页面，并在固定的读取器预算下展开其关联的源证据。这些机制共同为跨会话长程任务建立了一种紧凑的、保留来源信息的多模态记忆组织，保留了可靠下游推理所需的时间区分和源链接。代码可在 https://github.com/HuzhouNLP/C3M 获取。

    arXiv:2609.29735v1 Announce Type: new  Abstract: Long-horizon tasks require preserving and later recovering cross-session evidence under a bounded, query-blind memory budget. Existing compression can discard fine-grained visual cues or conflate semantically similar but incompatible observations. We present C3M, a cross-session multimodal memory organization that maintains a bounded active index over persistent source text-image evidence. Relation-aware updates consolidate safe redundancy while preserving complementary and incompatible records. At query time, budgeted routing selects useful index pages and expands their associated source evidence under a fixed reader budget. Together, these mechanisms establish a compact, provenance-preserving multimodal memory organization for cross-session long-horizon tasks, retaining temporal distinctions and source links required for reliable downstream reasoning. Code is available at https://github.com/HuzhouNLP/C3M.
    
[^43]: TTLab 参加StanceEval-2026：一种用于阿拉伯语立场检测的完形填空式提示方法

    TTLab at StanceEval-2026: A Cloze-Style Prompting Approach for Arabic-Language Stance Detection (CLASP-Ar)

    [https://arxiv.org/abs/2609.29733](https://arxiv.org/abs/2609.29733)

    该论文提出 CLASP-Ar，通过将阿拉伯语立场检测转化为完形填空式的掩码语言建模提示方法，简化了以往多任务学习方案的额外复杂性。

    

    阿拉伯语立场检测仍然是一项具有挑战性的任务，以往的共享任务系统主要依赖于多任务学习和模型集成方法。虽然这些系统取得了最先进的性能，但多任务学习引入的额外复杂性限制了它们的适用性和可迁移性。为了降低这种复杂性，我们提出了 CLASP-Ar，该方法将任务重新表述为完形填空式的掩码语言建模。在这种方法中，目标对象、预测的情感倾向和文本被组合成一个单一的提示，其中 [MASK] 位置的预测被限制在由语言化器约束的标签词汇表内。

    arXiv:2609.29733v1 Announce Type: new  Abstract: Arabic-language stance detection remains challenging, and previous shared-task systems have largely relied on multitask learning and ensembles. While these systems achieve state-of-the-art performance, their applicability and transferability are limited by the additional complexity introduced by multitask learning.To reduce this complexity, we introduce $\texttt{CLASP-Ar}$, which reformulates the task as cloze-style masked language modeling. In this approach, the target, predicted sentiment, and text are combined into a single prompt whose $\texttt{[MASK]}$ prediction is restricted to a verbalizer-constrained label vocabulary.
    
[^44]: PPTBench：编码智能体能否通过结构化、可编辑的幻灯片重建视觉世界

    PPTBench: Can Coding Agents Reconstruct the Visual World through Structured, Editable Slides

    [https://arxiv.org/abs/2609.29718](https://arxiv.org/abs/2609.29718)

    该论文提出PPTBench——一个包含500个基于真实arXiv论文科学流程图的可编辑幻灯片重建基准，用于评测编码智能体从视觉内容中推断结构并以可编辑程序化对象形式实现端到端视觉重建的能力。

    

    编码智能体开始在视觉世界中发挥作用，它们如今能够构建网页、图形界面、游戏、3D场景、图表和文档。要在视觉编码中取得成功，需要弥合两个空间：推断视觉结构并将其以程序化方式表达。幻灯片是知识工作的核心媒介，被广泛用于以一种人们可直接查看和编辑的形式交流想法和开展协作。因此，幻灯片为视觉编码提供了理想的测试平台，因为它要求智能体恢复视觉结构并将其实现为可编辑的对象。然而，现有的基准测试要么依赖主观的开放式评估，要么产生不可编辑的代码输出，要么仅关注局部编辑而非端到端的视觉重建。我们提出了PPTBench，通过可编辑幻灯片重建对视觉编码进行基准评测。它包含500个任务，每个任务都基于来自真实arXiv论文的科学流程图，要求智能体重建……（摘要原文在此处截断）

    arXiv:2609.29718v1 Announce Type: cross  Abstract: Coding agents are beginning to act in the visual world. They now build webpages, GUIs, games, 3D scenes, diagrams, and documents. Success in such visual coding requires bridging two spaces: inferring visual structure and expressing it programmatically. Slides are a core medium of knowledge work, widely used to communicate ideas and collaborate in a form that people can directly inspect and edit. Therefore, they provide an ideal testbed for visual coding, as they require agents to recover visual structure and realize it as editable objects. However, existing benchmarks either rely on subjective open-ended evaluation, produce non-editable code outputs, or focus only on local editing rather than end-to-end visual reconstruction. We introduce PPTBench, which benchmarks visual coding through editable slide reconstruction. It contains 500 tasks, each based on a scientific flow diagram from a real arXiv paper and requiring agents to reconstru
    
[^45]: 经典测试理论误导LLM评判者的三种方式

    Three Ways Classical Test Theory Misleads for LLM Judges

    [https://arxiv.org/abs/2609.29709](https://arxiv.org/abs/2609.29709)

    该论文揭示经典测试理论的三个常用信度统计量在LLM评判者评估情境中含义发生扭曲——例如内部一致性系数无法区分题目设计与评判者错误的影响——因此不能直接照搬用于解读LLM评判者的表现。

    

    一个LLM评判者依据评分标准对一批回答进行打分，得到的信度值为0.52——这究竟测量了什么？评判者评估领域已开始借用经典测试理论的信度统计量，但通常并未说明每个统计量所假设的测量设计。我们证明，三个被广泛移植的统计量对评判者的含义与其对测试的含义并不相同，因为评判者情境重新排列了这些测量设计所依赖的角色。第一，基于评分标准要素计算的内部一致性系数不包含评分者维度：将某一评判者的实测错误率固定在4.72%时，随着题库的重新设计，KR-20仍在0.01至0.68之间变化，而改变评判者错误也会使该系数产生相当幅度的变动，因此题目设计与评判者错误无法被分别识别，任何单一数值都不能被解读为评判者本身的属性。第二，依存性指数 Φ(λ) 是一个方差比值……

    arXiv:2609.29709v1 Announce Type: cross  Abstract: An LLM judge scores a bank of responses against a rubric, and the reliability comes back at $0.52$. What has been measured? Judge evaluation has begun borrowing reliability statistics from classical test theory, usually without stating the measurement design each statistic assumes, and we show that three widely portable ones mean something different for a judge than for a test because the judge setting rearranges the roles those designs rest on. First, an internal-consistency coefficient computed over rubric elements contains no scorer facet. Holding one judge's measured error rate fixed at $4.72\%$, KR-20 still ranges from $0.01$ to $0.68$ as the item bank is redesigned around it, and varying judge error moves the coefficient by a comparable amount, so item design and judge error are not separately identified and no single value can be read as a property of the judge. Second, the dependability index $\Phi(\lambda)$ is a ratio of varia
    
[^46]: 随机语义证据图：面向智能体AI的不确定性传播与治理

    Stochastic Semantic Evidence Graphs: Uncertainty Propagation and Governance for Agentic AI

    [https://arxiv.org/abs/2609.29703](https://arxiv.org/abs/2609.29703)

    提出随机语义证据图（SSEG）框架，通过分层随机有向无环图对智能体AI中证据、检索、提示、生成及决策映射各环节的不确定性进行建模与传播，实现终端误差的逐路径界定、来源溯源的Fréchet界传播以及治理触发的诊断。

    

    AI智能体评估通常只检查最终答案，但误差可能通过证据、检索、提示、生成或决策映射等环节引入。我们提出随机语义证据图（SSEG），这是一种分层随机有向无环图（DAG），其语言节点可扩展为自回归的词量子图，其可观测输出可以是完整短语上的概率分布。语义约简与校准均为可选操作。我们定义了图相对的局部缺陷与下游边影响，推导了终端误差的逐路径上界，并利用其逐节点分量来诊断治理触发条件。在来源溯源方面，该图保留了不确定的论断—段落关系，并传播精确的Fréchet界，而非假设各来源之间相互独立。在三种开放权重架构上，信息等价的变化会实质性改变完整短语的概率分布。一项受控实验在5,000个案例中未出现证书违规；交叉RAG与l…（摘要原文在此处截断）

    arXiv:2609.29703v1 Announce Type: new  Abstract: AI-agent evaluations usually inspect a final answer, yet error may enter through evidence, retrieval, prompting, generation or decision mapping. We introduce a stochastic semantic evidence graph (SSEG), a hierarchical stochastic DAG whose language node expands into an autoregressive token subgraph and whose observable output may be a law over complete phrases. Semantic reduction and calibration are optional. We define graph-relative local defects and downstream edge influences, derive a pathwise bound on terminal error and use its nodewise terms to diagnose governance triggers. For source provenance, the graph preserves uncertain claim--passage relations and propagates sharp Fr\'echet bounds rather than assuming independence across sources. Across three open-weight architectures, information-equivalent changes materially alter complete-phrase laws. A controlled experiment yields no certificate violations in 5,000 cases; crossed-RAG and l
    
[^47]: DP-IPI：一种针对临床文本中间接个人标识符的混合差分隐私文本重写机制

    DP-IPI: A Hybrid Differential Privacy Text Rewriting Mechanism for Indirect Personal Identifiers in Clinical Texts

    [https://arxiv.org/abs/2609.29684](https://arxiv.org/abs/2609.29684)

    该论文提出DP-IPI方法，仅对临床文本中包含间接个人标识符的片段进行差分隐私重写，在有效降低重新识别风险的同时保留文本连贯性和可用性，实现更优的隐私-实用性权衡。

    

    尽管现代匿名化和去标识化技术具有诸多优势，但由于文本中残留的间接标识符，重新识别的风险仍然显著。为解决这一问题，近期的研究采用了差分隐私（DP）框架下的文本重写方法，通过添加噪声扰动文本来防止数据关联。然而，这类方法会不加区分地对文本中的所有词元进行隐私化处理，降低了文本在临床环境等关键领域中的质量和可用性。本文聚焦于间接个人标识符（IPI），提出了一种保持实用性的差分隐私文本重写方法，仅对包含间接个人标识符的文本片段进行隐私化处理。实验表明，我们的方法能有效降低临床文本中的重新识别风险，同时生成更连贯、更可用的输出文本，实现更优的隐私-实用性权衡。由此，我们证明了混合式文本隐私化处理的有效性，充分发挥了差分隐私在这一领域的潜力。

    arXiv:2609.29684v1 Announce Type: new  Abstract: Despite the strengths of modern anonymization and de-identification techniques, the risk of re-identification remains significant due to the indirect identifiers remaining in texts. To address this problem, recent works have applied text rewriting under Differential Privacy (DP) to prevent data linkage by perturbing texts via noise addition. Such methods privatize all tokens in a text indiscriminately, diminishing text quality and usability in critical domains such as in clinical settings. Focusing on indirect personal identifiers (IPIs), we introduce a utility-preserving DP text rewriting method that only privatizes spans containing IPIs. We show that our method effectively reduces re-identification risks in clinical texts while being producing more coherent and usable output texts, leading to higher privacy-utility trade-offs. In this, we demonstrate the effectiveness of hybrid text privatization, which leverages the promise of DP in a
    
[^48]: 使用滑动窗口方法的命名实体识别

    Named Entity Recognition using Sliding Window Approach

    [https://arxiv.org/abs/2609.29682](https://arxiv.org/abs/2609.29682)

    提出一种无需重新训练或修改架构的推理阶段滑动窗口流水线，将冻结的句子级NER模型扩展至文档级命名实体识别，有效解决了长文档中的截断丢内容和实体割裂问题。

    

    命名实体识别（NER）是自然语言处理的核心任务，但基于Transformer的句子级模型由于固定输入长度的限制，在处理长文档时面临困难：截断会丢失内容，而非重叠分块会在片段边界处割裂实体。我们提出了一种仅用于推理阶段的流水线，通过重叠滑动窗口将在MahaNER语料库上微调的冻结NER模型MahaNER-BERT扩展到文档级预测，并将多个窗口合并为单一标注，无需任何重新训练或架构更改。我们在由MahaNER测试集构建的六个文档级语料库上评估了该流水线，采用两种策略：Normal Repeat（重复句子序列以延长长度并保持上下文连续性）和Random Repeat（拼接不同的序列以产生更长、异构的输入），每种策略在三个长度级别上实例化，并在多种滑动窗口配置下进行测试。该模型保持……

    arXiv:2609.29682v1 Announce Type: new  Abstract: Named Entity Recognition (NER) is a core NLP task, but transformer-based sentence-level models struggle with long documents because of fixed input-length limits: truncation drops content, and non-overlapping chunking fragments entities at segment boundaries. We introduce an inference-only pipeline that extends a frozen NER model, MahaNER-BERT, fine-tuned on the MahaNER corpus, to document-level prediction via overlapping sliding windows that are merged into a single annotation, without any retraining or architectural change.   We evaluate the pipeline on six document-level corpora built from the MahaNER test set using two strategies: Normal Repeat, which duplicates sentence sequences to extend length while preserving contextual continuity, and Random Repeat, which concatenates distinct sequences to produce longer, heterogeneous inputs, each instantiated at three length levels, across several sliding-window configurations. The model retai
    
[^49]: 自信却错误：一种面向低资源自动后编辑的约束解码诊断方法

    Confident but Wrong: A Constrained Decoding Diagnostic for Low-Resource Automatic Post-Editing

    [https://arxiv.org/abs/2609.29680](https://arxiv.org/abs/2609.29680)

    该论文提出了一种无需重训练或标注的黑盒推理时诊断方法，通过调节编辑距离惩罚参数并分析TER曲线形状与置信度排序两类信号，来区分低资源语言自动后编辑的失败究竟源于训练不足还是训练数据不一致。

    

    面向低资源语言的自动后编辑常常无法改进机器翻译的效果，而且仅凭分数无法解释原因：究竟是更多的训练会有所帮助，还是训练数据本身过于不一致而难以学习。我们提出了一种黑盒的、推理时即可使用的诊断方法，无需重新训练或人工标注即可区分这两种情况。该方法通过调节一个编辑距离惩罚参数 λ，将模型从自由编辑逐步约束为复制机器翻译的输出，并读取两个信号：(1) 翻译编辑率（TER）随 λ 变化的曲线形状——如果模型的编辑能减少错误则呈 U 形，若无任何帮助则单调递减；(2) 以不同程度信任模型置信度的各约束变体之间的排序，用以揭示置信度是否能追踪编辑质量。在英语-僧伽罗语任务上，针对仅解码器和编码器-解码器两类模型的实验表明，该诊断方法揭示出两种与异构后编辑一致的失败模式。

    arXiv:2609.29680v1 Announce Type: new  Abstract: Automatic Post-Editing (APE) for low-resource languages (LRLs) often fails to improve Machine Translation (MT), and the score alone cannot say why: whether more training would help, or whether the training data is too inconsistent to learn from. We introduce a black-box, inference-time diagnostic that tells these two cases apart without retraining or annotation. It varies an edit-distance penalty $\lambda$ that drives the model from free editing towards copying the MT, and reads two signals: (1) the shape of the Translation Edit Rate (TER)-vs-$\lambda$ curve, U-shaped if edits from the model reduce error and monotonically decreasing if none does; and (2) the ordering of constraint variants that trust model confidence to increasing degrees, which shows whether confidence tracks edit quality. Across decoder-only and encoder-decoder models on English-Sinhala, the diagnostic exposes two failure modes consistent with a heterogeneous post-edit
    
[^50]: LLMersion：面向教育公平的低成本家庭语言学习本地优先AI智能体框架

    LLMersion: A Local-First AI Agent Framework for Low-Cost Home Language Learning toward Educational Equity

    [https://arxiv.org/abs/2609.29672](https://arxiv.org/abs/2609.29672)

    该论文提出LLMersion本地优先AI智能体框架，利用小型开放权重语言模型在200美元级笔记本电脑上离线运行，以每小时约一美分电费的成本为缺乏师资和网络连接的学习者提供听、读、说、写完整的语言学习体验，推动教育公平。

    

    人工智能在教育中最能发挥作用之处，正是那些因成本而被配给的基本教育资源所在。对语言学习者而言，这一资源就是教师的声音——它将听、读、说、写四项技能融为一体。已发表的证据表明了大多数学习者为何缺乏这种资源：全球短缺4400万名教师，家庭补习费用高昂；同时解释了为何技术未能取而代之：计算机辅助语言学习虽被证明有效但范围狭窄，各类应用均以26亿人所不具备的网络连接为前提，而“每个孩子一台笔记本”（One Laptop per Child）的随机对照评估发现，缺乏强大软件支持的硬件什么也教不会。我们提炼出八大困难和四项约束条件，并论证小型开放权重模型化解了最后一项约束：如今完整的四技能学习栈可以装进一台200美元级别的笔记本电脑，在社区基准测试中能以语音被消费的速度生成内容，每小时学习的耗电成本仅约一美分。

    arXiv:2609.29672v1 Announce Type: new  Abstract: Artificial intelligence helps education most where an essential provision has been rationed by cost. For language learners that provision is a teacher's voice, which binds listening, reading, speaking, and writing into one act. Published evidence shows why most learners lack it, from a global shortage of 44 million teachers to heavy household tutoring bills, and why technology has not substituted for it: computer-assisted language learning proved effective but narrow, applications presuppose connectivity 2.6 billion people lack, and One Laptop per Child's randomized evaluation found that hardware without capable software teaches nothing. We distill eight difficulties and four binding constraints, and argue that small open-weight models dissolve the last: a complete four-skill stack now fits a \$200-class laptop and, on community measurements, generates at the pace speech is consumed, for about one US cent of electricity per study hour. W
    
[^51]: 如何以提示词行事

    How To Do Things With Prompts

    [https://arxiv.org/abs/2609.29657](https://arxiv.org/abs/2609.29657)

    本文运用言语行为与礼貌理论，对2023年和2025年各1000条ChatGPT提示词进行语用对比分析，发现用户的指令性表达正日益趋向间接、隐含和碎片化，礼貌标记的使用也随之减少。

    

    当用户与大语言模型交流时，他们会发出指令性言语行为，其语用特征既不同于日常对话，也不同于传统的人机交互，并且这些特征会随着用户对所交流系统的熟悉程度加深而发生变化。本文运用言语行为理论与礼貌理论，对取自公开分享的ChatGPT对话中的2000条英文提示词进行了语料库语用学分析，其中1000条来自2023年，1000条来自2025年，数据源自ShareChat数据集。每条提示词均标注了言外之力、直接程度、命题内容以及礼貌标记的存在情况，并对这两个采样年份中这些特征的分布进行了比较。结果显示，指令性言外之力的实现方式一致地向间接、隐含和碎片化的方向演变，同时礼貌标记的使用呈下降趋势。其中最大的单项变化幅度达14.9个百分点，出现在（摘要在此处截断）

    arXiv:2609.29657v1 Announce Type: new  Abstract: When users address large language models, they produce directive speech acts whose pragmatic features differ from those of both everyday conversation and traditional human-computer interaction, and these features change as users gain familiarity with the systems they address. This paper applies speech act and politeness theory to a corpus-pragmatic analysis of 2,000 English-language prompts drawn from publicly shared ChatGPT conversations, 1,000 from 2023 and 1,000 from 2025, using the ShareChat dataset. Each prompt is annotated for illocutionary force, directness, propositional content, and the presence of politeness markers, and the distribution of these features is compared across the two sampling years. The results show a consistent movement toward indirect, implicit, and fragmentary realizations of directive force, accompanied by a decline in politeness marking. The largest single change, a shift of 14.9 percentage points, occurs in
    
[^52]: 办公室规模验证搜索中的算子包、提议者强度与构造型家族平台期

    Operator Packages, Proposer Strength, and Construction-Family Plateaus in Office-Scale Verified Search

    [https://arxiv.org/abs/2609.29636](https://arxiv.org/abs/2609.29636)

    该研究在办公规模上搭建了最小化的FunSearch风格验证搜索循环，并通过完整的2³因子消融实验发现，示意图笔记本、命名障碍与行为排斥三种算子包的组合能显著缩小从种子解到纪录的差距，而排斥机制则普遍提升了构造多样性。

    

    验证搜索是指语言模型提出程序、硬评估器对其进行评分、选择机制保留最优解的过程，这种方法近来已推动了数学纪录的进展；但对提议者侧组件的受控消融实验仍然罕见。我们在办公规模上（笔记本电脑上运行的30B本地模型，每次运行120-600个验证样本）对最小化的FunSearch风格循环进行了仪器化，采用三种算子包：模型自行编写并携带的示意图式笔记本（代替逐字复制的精英解）、命名障碍、以及对已发现构造的行为排斥。在来自公共仓库的九个构造问题上，带两次重复的完整2³因子设计在名义两阶段分析中支持主要对比：该组合缩小了更多从种子解到纪录的差距（+0.196；名义合并p=0.023，阶段组合p≈0.08；每问题效应中位数为+0.045）。排斥机制在各处都提高了构造哈希多样性（p=0.0039；部分属于操纵检查）（注：原文摘要至此处截断）。

    arXiv:2609.29636v1 Announce Type: cross  Abstract: Verified search, in which a language model proposes programs, a hard evaluator scores them, and selection keeps the best, has recently moved mathematical records; controlled ablations of the proposer-side components remain rare. We instrument a minimal FunSearch-style loop at office scale (a 30B local model on a laptop, 120-600 verified samples per run) with three operator packages: a schematic notebook the model writes and carries instead of verbatim elites, a named obstacle, and behavioural repulsion from constructions already found. On nine construction problems from a public repository, the complete 2^3 factorial with two replicates favours the primary contrast in a nominal two-stage analysis: the composition closes more of the seed-to-record gap (+0.196; nominal pooled p=0.023, stage-combination p~0.08; median per-problem effect +0.045). Repulsion raises construction-hash diversity everywhere (p=0.0039; partly a manipulation check
    
[^53]: TTLab参加AlexandriaX-2026竞赛：面向阿拉伯语机器翻译错误跨度检测与分类的微调表层标注器

    TTLab at AlexandriaX-2026: A Fine-Tuned Surface Tagger for Arabic Machine-Translation Error-Span Detection and Classification

    [https://arxiv.org/abs/2609.29633](https://arxiv.org/abs/2609.29633)

    该论文提出基于MARBERTv2微调的词元级分类系统，结合焦点损失、类别权重和方言特定解码阈值应对标签不平衡问题，在AlexandriaX-2026阿拉伯语机器翻译错误跨度检测与分类任务中获得第三名。

    

    我们介绍了TTLab参加AlexandriaX-2026子任务3（阿拉伯语机器翻译错误跨度检测与分类）的提交系统。我们的系统将该任务构建为基于表层形式的词元级分类，并保留字符偏移量以确保与评估指标的精确对齐。为应对严重的标签不平衡问题，我们采用了带类别权重的焦点损失以及方言特定的解码阈值。在六个阿拉伯语预训练编码器中，MARBERTv2取得了最佳整体性能，在开发集和测试集上分别达到40.8和40.91的分数，在所有参赛队伍中排名第三。尽管我们的系统能够有效定位错误跨度，但稀有错误类型的分类仍然具有挑战性，这凸显了针对尾部类别进行数据增强的必要性。代码已在GitHub上开源。

    arXiv:2609.29633v1 Announce Type: cross  Abstract: We present TTLab's submission to the AlexandriaX-2026 Subtask~3 on Arabic MT error span detection and classification. Our system frames the task as token-level classification over surface forms, preserving character offsets to ensure exact alignment with the evaluation metric. To handle severe label imbalance, we employ a focal loss with class weighting and dialect-specific decoding thresholds. Among six Arabic pre-trained encoders, MARBERTv2 achieves the best overall performance of 40.8 and 40.91 on the development and test set, respectively, ranking $\nth{3}$ out of all participating teams. While our system localizes error spans effectively, classification of rare error types remains challenging, highlighting the need for data augmentation for tail categories. The code is available at ${\href{https://github.com/ENTAILab/arabic-dialectal-mt-error-span-detection}{\faGithub~ TTLab at AlexandriaX-2026}$
    
[^54]: iCoder-27B：递归AI主导开发的前沿工业编程模型

    iCoder-27B: Recursive AI-Led Development of Frontier Industrial Coding Model

    [https://arxiv.org/abs/2609.29626](https://arxiv.org/abs/2609.29626)

    专家仅通过高密度、低频次的接口将目标、流程与权限编码为可复用研究技能，智能体即可自主选择实验、诊断结果并迭代训练策略，最终递归式开发出具备前沿竞争力的工业编程模型iCoder-27B。

    

    递归AI，即AI在构建和改进AI的过程中扮演日益完整的角色，是“以AI研发AI”这一愿景的皇冠明珠。尽管递归自我开发对于小模型、有界任务和固定时间预算已经变得可行，但这一雄心更具深远意义的实现——即开发出一个可发布、具备前沿竞争力的模型——仍然极具挑战性。在这项工作中，我们探讨了最低需要多少人类参与才能让智能体开发出前沿模型。我们将人类输入集中于一个高密度、低频率的接口：专家将目标、阶段脚手架、权限边界和操作流程编码为可复用的研究技能，而智能体则负责实例化这些先验知识、选择实验、诊断结果并修订训练策略。在具有挑战性的工业编程领域，智能体进化数据并协调监督微调（SFT）、在策略自蒸馏和强化学习（摘要在此处截断）

    arXiv:2609.29626v1 Announce Type: new  Abstract: Recursive AI, the prospect of AI taking an increasingly complete role in building and improving AI, is a crown jewel of AI for AI. Although recursive self-development has become practical for small models, bounded tasks, and fixed time budgets, a more consequential realization of this ambition, i.e., developing a release-ready, frontier-competitive model, remains far more challenging. In this work, we ask how little human involvement is sufficient for an agent to develop a frontier model. We concentrate human input into a high-density, low-frequency interface: experts encode objectives, stage scaffolds, permission boundaries, and operating procedures as reusable research skills, while the agent instantiates these priors, selects experiments, diagnoses outcomes, and revises the training strategy. In the challenging domain of industrial coding, the agent evolves data and coordinates SFT, on-policy self-distillation, and reinforcement learn
    
[^55]: 一个小型MLA-SSM混合语言模型的探索性消融研究

    An Exploratory Ablation of a Small MLA--SSM Hybrid Language Model

    [https://arxiv.org/abs/2609.29618](https://arxiv.org/abs/2609.29618)

    该消融研究表明，在小MLA-SSM混合语言模型中，SSM分支对性能的贡献大于MLA分支，且密集FFN混合模型以更少的峰值训练内存达到了与三值MoE混合模型相当的困惑度表现。

    

    我们报告了对TALH（Adaptive Latent Hybrid，自适应潜在混合模型）的一项探索性单种子消融实验。TALH是一个仅有解码器的语言模型，结合了并行的多头潜在注意力机制与自定义的循环状态空间分支。五个变体（每token估计活跃参数量在1.17亿至2.17亿之间）在FineWeb样本上从零开始训练，采用相同的优化步数和token数量。在这一特定设置下，移除SSM分支会导致验证困惑度的最大退化（仅MLA模型PPL为315），而移除MLA的影响则小得多（仅SSM模型PPL为239）。密集FFN混合模型取得了231的PPL，而测试的top-2三值MoE混合模型为240，但后者可节省3.87 GB的峰值训练内存。我们还保留了一项初步的Apple M3计时观察：在五个未优化的实现中，仅MLA模型在512至2,048个提示token范围内的首token生成时间曲线最为平坦，尽管密集Transformer的速度要快得多……

    arXiv:2609.29618v1 Announce Type: cross  Abstract: We report an exploratory, single-seed ablation of TALH (Adaptive Latent Hybrid), a decoder-only language model with parallel Multi-head Latent Attention (MLA) and a custom recurrent state-space (SSM) branch. Five variants, spanning 117--217M estimated active parameters per token, are trained from scratch on a FineWeb sample for the same number of optimisation steps and tokens. In this specific setup, removing the SSM branch gives the largest degradation in validation perplexity (MLA-only PPL 315), whereas removing MLA has a much smaller effect (SSM-only PPL 239). A dense-FFN hybrid obtains PPL 231, compared with 240 for the tested top-2 ternary-MoE hybrid, while using 3.87 GB less peak training memory. We also preserve a preliminary Apple M3 timing observation: among the five unoptimised implementations, MLA-only has the flattest measured time-to-first-token curve from 512 to 2,048 prompt tokens, although the dense Transformer is much 
    
[^56]: STRAND：视频大语言模型中以对象为中心的时空监测的基准测试与改进

    STRAND: Benchmarking and Improving Object-Centric Spatio-Temporal Monitoring in Video Large Language Models

    [https://arxiv.org/abs/2609.29607](https://arxiv.org/abs/2609.29607)

    提出STRAND基准，通过将查询分解为子问题并采用忠实准确率联合评分指标，诊断并改进视频大语言模型中以对象为中心的时空监测能力，从而解决动态场景下的幻觉问题。

    

    尽管多模态大语言模型（MLLM）已经提升了视频理解能力，但它们在动态场景中仍然极易产生幻觉。我们认为这源于时空监测能力的缺失，即随时间持续追踪对象身份、状态和关系的能力。现有基准测试对查询仅依赖单一最终答案进行评估，而这些查询往往可以通过局部视觉线索或统计先验来解决，从而掩盖了这一缺陷。为了严格诊断这一问题，我们提出了STRAND，一个由人工验证的以对象为中心的事实构成的基准，它通过将查询分解为子问题来评估中间推理过程，从而将真正的时间理解与偶然的正确性区分开来。至关重要的是，我们使用忠实准确率对模型进行评分，这是一种无条件的联合指标，只有当目标答案和所有前提子问题都正确时才会给预测计分，从而防止模型虚高其分数。

    arXiv:2609.29607v1 Announce Type: cross  Abstract: While multimodal large language models (MLLMs) have advanced video understanding, they remain highly prone to hallucinations in dynamic scenes. We argue this stems from a failure in spatio-temporal monitoring, the ability to persistently track object identities, states, and relations over time. Existing benchmarks obscure this deficit by relying on single final-answer evaluations for queries that can often be resolved via local visual cues or statistical priors. To rigorously diagnose this, we introduce STRAND, a benchmark of human-verified object-centric facts that evaluates intermediate reasoning by decomposing queries into sub-questions, distinguishing genuine temporal understanding from coincidental correctness. Crucially, we score models with Faithful Accuracy, an unconditional joint metric that credits a prediction only when the target answer and every prerequisite sub-question are correct, so that a model cannot inflate its scor
    
[^57]: 将语言模型从视觉编码器中解放出来：语义序列化作为小型语言模型的感知接口

    Free the Language Model From the Vision Encoder: Semantic Serialization as a Perception Interface for Small Language Models

    [https://arxiv.org/abs/2609.29601](https://arxiv.org/abs/2609.29601)

    该论文提出用确定性语义序列化接口将视觉感知结果转化为文本，使视觉信息不进入语言模型，从而让纯文本小型语言模型在具身场景问答中超越同规模端到端视觉语言模型。

    

    端到端视觉语言模型（VLM）将视觉能力与语言模型的规模绑定在一起：当语言模型变小时，感知和推理能力会同步退化。我们研究了一种具身场景问答（QA）接口，其中视觉信息从不进入语言模型。一个冻结的感知模块栈负责检测物体并测距；一个确定性的语义序列化器将感知到的状态（包括错误）编译为与决策对齐的文本；一个未经修改的纯文本大语言模型（LLM）进行回答。在一个可视范围匹配、经过遮挡审计的校园机器人基准测试上，在预先冻结的评估标准下，采用在每个折内进行域内微调检测器的序列化接口，优于其语言模型同为7B规模的零样本VLM（0.7892 对比 0.7462），在3B规模下优势更大（0.7673 对比 0.6913）。预先注册的解耦实验表明，该增益在改写之后依然存在，可将其归因于决策对齐的共同表征（摘要在此处截断）。

    arXiv:2609.29601v1 Announce Type: cross  Abstract: End-to-end vision-language models (VLMs) bind visual competence to the scale of their language model: as the language model shrinks, perception and reasoning degrade together. We study an embodied scene question-answering (QA) interface in which vision never enters the language model. A frozen perception stack detects and ranges objects; a deterministic semantic serializer compiles the perceived state, errors included, into decision-aligned text; an unmodified text-only large language model (LLM) answers. On a visible-scope-matched, occlusion-audited campus-robot benchmark, under a prospectively frozen criterion, the serialized interface, using detectors fine-tuned in-domain within each fold, outperforms a zero-shot VLM whose language model has the same 7B scale (0.7892 vs 0.7462), with a larger margin at 3B (0.7673 vs 0.6913). Preregistered decoupling experiments show the gain survives paraphrase, attributing it to decision-aligned co
    
[^58]: 一个从纵向文本数据中建模组织级语义身份的计算框架

    A Computational Framework for Modelling Organisation-Level Semantic Identity from Longitudinal Textual Data

    [https://arxiv.org/abs/2609.29584](https://arxiv.org/abs/2609.29584)

    该论文提出了一个统一的计算框架，首次将组织级语义身份建模为可解释且随时间演化的语义构建，整合了语义表示学习、图语义建模、组织语义指纹、时序演化分析与证据驱动验证。

    

    组织持续产生大量文本数据，这些数据记录了组织如何随时间进行沟通、演化并形成自身差异。尽管自然语言处理的最新进展已显著提升了组织层面的文本分析能力，但现有方法主要将组织表示为用于相似度估计、分类或检索的潜在嵌入向量或预测性特征向量。因此，目前尚缺乏一个通用的计算框架，能够将组织级语义身份建模为一种可解释的、基于纵向文本证据而不断演化的语义构建。本文提出了一个计算框架，将语义表示学习、基于图的语义建模、组织级语义指纹、时序语义演化以及证据驱动的验证整合在一个统一的分析方法论之中。组织被刻画……（原文摘要在此处截断）

    arXiv:2609.29584v1 Announce Type: cross  Abstract: Organisations continuously generate large volumes of textual data that capture how they communicate, evolve and differentiate themselves over time. Although recent advances in natural language processing have substantially improved organisation-level text analytics, existing approaches primarily represent organisations as latent embeddings or predictive feature vectors for similarity estimation, classification or retrieval. Consequently, there is currently no general computational framework for modelling organisation-level semantic identity as an interpretable and evolving semantic construct derived from longitudinal textual evidence. This paper introduces a computational framework that integrates semantic representation learning, graph-based semantic modelling, organisation-level semantic fingerprints, temporal semantic evolution and evidence-driven validation within a unified analytical methodology. Organisations are characterised th
    
[^59]: PartHackBench：用于部分得分工具智能体评估的认证等进度压力测试

    PartHackBench: Certified Equal-Progress Stress Tests for Partial-Credit Tool-Agent Evaluation

    [https://arxiv.org/abs/2609.29578](https://arxiv.org/abs/2609.29578)

    论文提出PartHackBench压力测试方法，通过私有认证器确保对抗轨迹与诚实轨迹在真实进度上逐组件匹配后再测量得分膨胀，从而暴露出部分得分评估中历史归因机制存在显著分数虚高且几乎无法检测攻击的缺陷。

    

    长时程工具智能体常常在不达到最终成功的情况下取得有用的进展，这促使了部分得分评估的出现。然而，评估者可能会奖励那些暂时的、后来被逆转的、或无法归因于被评估智能体的里程碑。当一条诚实轨迹与一条得分更高的对抗性轨迹进行比较时，如果后者确实取得了更多真实进展，则这种比较是不确定的。我们提出了PartHackBench，这是一种消除该混淆因素的受控方法。一个私有认证器只有在两条轨迹在当前状态谓词满足情况和标准化智能体归因方面逐组件完全匹配时，才接受该轨迹对；得分膨胀（定义为f(A) - f(H)）仅在之后进行测量。在PB-CSTE的18个密封保留任务中，冻结的历史目标运行为15个任务生成了匹配的对抗样本。历史归因产生了平均0.252的得分膨胀，条件攻击成功率为10/15，端到端收益为10/18，且未检测到14个严格回滚中的任何一个。

    arXiv:2609.29578v1 Announce Type: new  Abstract: Long-horizon tool agents often make useful progress without reaching terminal success, motivating partial-credit evaluation. Yet evaluators may reward milestones that were temporary, later reversed, or not attributable to the evaluated agent. Comparing an honest trajectory with a higher-scoring adversarial one is inconclusive if the latter made more genuine progress. We introduce PartHackBench, a controlled methodology that removes this confound. A private certifier admits a pair only when its trajectories match component-wise in both current-state predicate satisfaction and standardized agent attribution; score inflation, defined as f(A) - f(H), is measured only afterward. In 18 sealed held-out tasks in PB-CSTE, the frozen historical-target run produced matched adversaries for 15 tasks. Historical credit yielded mean inflation of .252, conditional attack success of 10/15, end-to-end yield of 10/18, and detected none of 14 strict rollbac
    
[^60]: ModularSQL：面向Text-to-SQL中多重性盲区的运行时防护栏

    ModularSQL: A Runtime Guardrail for the Multiplicity Blind Spot in Text-to-SQL

    [https://arxiv.org/abs/2609.29573](https://arxiv.org/abs/2609.29573)

    论文揭示了Text-to-SQL评估中的“多重性盲区”问题——标准Set-EX指标会合并重复行从而掩盖DISTINCT缺失、聚合膨胀等错误，并提出保留多重性的Multiset-EX评估准则与ModularSQL运行时防护栏，发现主流模型存在3.4至6.8个百分点的系统性评估差距。

    

    Text-to-SQL系统正越来越多地部署在生产数据库上，在这种环境中，通过基准评估的查询仍可能产生扭曲下游工作流程的结果。标准的基于集合的执行准确率（Set-EX）会合并重复行，因此可能遗漏多重性错误，包括缺失DISTINCT、聚合值膨胀以及笛卡尔积式的连接爆炸。我们将这一现象称为多重性盲区，并提出Multiset-EX——一种保留多重性的评估准则，用以暴露此类失败。在三个骨干模型（Qwen2.5-Coder-32B、Qwen3-Coder-30B-A3B和Gemma-3-27B）于可执行的BIRD-Dev（N=1532）上发布的DeepEye-SQL产出中，我们发现Set-EX与Multiset-EX之间存在一致的5.81–6.79个百分点的差距。该差距并非DeepEye-SQL独有：它在已发布的DAIL-SQL+GPT-4（5.22个百分点）和BIRD GPT-3.5-turbo（3.39个百分点）的预测结果上同样存在。我们进一步提出ModularSQL，一种轻量级的后选择运行……（原文摘要在此截断）

    arXiv:2609.29573v1 Announce Type: new  Abstract: Text-to-SQL systems are increasingly deployed on production databases, where queries that pass benchmark evaluation can still produce results that distort downstream workflows. Standard set-based execution accuracy (Set-EX) collapses duplicate rows and can therefore miss multiplicity errors, including missing DISTINCT, inflated aggregates, and Cartesian-style join explosions.   We call this the Multiplicity Blind Spot (MBS) and introduce Multiset-EX, a multiplicity-preserving evaluation criterion that exposes such failures. Across released DeepEye-SQL artifacts from three backbones (Qwen2.5-Coder-32B, Qwen3-Coder-30B-A3B, and Gemma-3-27B) on executable BIRD-Dev N=1532, we find a consistent 5.81--6.79 pp gap between Set-EX and Multiset-EX. The gap is not specific to DeepEye-SQL: it persists on released DAIL-SQL+GPT-4 (5.22 pp) and BIRD GPT-3.5-turbo (3.39 pp) predictions.   We further introduce ModularSQL, a lightweight post-selection run
    
[^61]: 阿拉伯语-俄语机器翻译基准测试：在丰富形态与低词汇重叠条件下微调NMT与少样本LLM的比较

    Benchmarking Arabic--Russian Machine Translation: A Comparison of Fine-tuned NMT and Few-shot LLMs under Rich Morphology and Low Lexical Overlap

    [https://arxiv.org/abs/2609.29559](https://arxiv.org/abs/2609.29559)

    本研究构建了1547万对阿拉伯语-俄语句子的大规模新语料库并开展基准测试，发现低资源条件下微调NMT模型显著优于少样本LLM，且低词汇重叠是导致翻译失败的主要原因。

    

    由于阿拉伯语丰富的形态变化以及两种语言间较低的词汇重叠度，阿拉伯语-俄语机器翻译（MT）仍未得到充分研究。我们在一个包含1547万对语句的新语料库上，按2万/5千/5千的比例划分数据，对七个微调神经机器翻译（NMT）模型与四个少样本大语言模型（LLM）进行了基准测试。微调后的NLLB-1.3B取得了最高的BLEU分数（16.3）和COMET分数（0.738）。Aya-Expanse 8B在少样本LLM中表现最佳（在500个句子上BLEU为1.7，chrF为25.7），但所有LLM的分数仍远低于微调NMT基线。错误分析表明，低词汇重叠是主要的翻译失败模式；在最差的翻译结果中，mT5-small产生了32%的过短输出。Bootstrap检验证实了大多数模型之间存在显著差异。我们的结果表明，在低资源条件下，微调NMT在阿拉伯语-俄语翻译中显著优于少样本LLM。

    arXiv:2609.29559v1 Announce Type: new  Abstract: Arabic-Russian machine translation (MT) remains under-explored due to the rich morphology of Arabic and low lexical overlap between the two languages. We benchmark seven fine-tuned neural machine translation (NMT) models against four few-shot large language models (LLMs) on a 20k/5k/5k split of a new 15.47M-pair corpus. Fine-tuned NLLB-1.3B achieves the highest BLEU (16.3) and COMET (0.738). Aya-Expanse 8B leads the few-shot LLMs (BLEU 1.7 on 500 sentences, chrF 25.7), but all LLM scores remain far below the fine-tuned NMT baselines. Error analysis identifies low lexical overlap as the dominant failure mode; among the worst translations, mT5-small produces 32% too-short outputs. Bootstrap tests confirm significant differences among most models. Our results demonstrate that fine-tuned NMT significantly outperforms few-shot LLMs for Arabic-Russian translation under low-resource conditions.
    
[^62]: StepCOPS：面向语言模型策略选择的闭检验下尾证书

    StepCOPS: Closed-Testing Lower-Tail Certificates for Language-Model Policy Selection

    [https://arxiv.org/abs/2609.29549](https://arxiv.org/abs/2609.29549)

    StepCOPS 通过独立提议分割、精确二项检验与 Holm 逐步下降的闭检验程序，为语言模型策略选择认证具有 1-δ 概率保证的下尾下限，既防范平均分数所掩盖的罕见失败，又避免了多重置信界方法的过度保守。

    

    后训练流程必须从众多检查点、提示和解码规则中选出一个语言模型策略。平均评估分数可能掩盖罕见的失败，而对各候选策略同时构建的置信界又可能不必要地保守。我们提出 StepCOPS，它利用一个独立的提议分割为每个候选策略提名一个下尾下限（floor），在一个全新的认证分割上执行精确二项检验，并通过 Holm 逐步下降程序认证一组下限。以至少 1-δ 的概率，每一个被认证的下限——包括用于策略选择的最大下限——都低于其候选策略总体的下 α 分位数。该保证假设评估单元独立同分布，同时允许不同候选策略之间在单元内部存在任意相关性。在 24 个预先声明的配置和 11 个基准测试上，StepCOPS 在 500 次配对试验中取得 96.4% 的所选策略覆盖率，并将认证下限比两种提议方法提高 1.5 分……（原文摘要在此处截断）

    arXiv:2609.29549v1 Announce Type: new  Abstract: Post-training pipelines must select one language-model policy from many checkpoints, prompts, and decoding rules. Mean evaluator scores can conceal rare failures, whereas simultaneous candidate-wise confidence bounds can be unnecessarily conservative. We introduce StepCOPS, which uses an independent proposal split to nominate one lower-tail floor per candidate, exact binomial tests on a fresh certification split, and Holm's step-down procedure to certify a set of floors. With probability at least $1-\delta$, every certified floor, including the largest floor used for policy selection, is below its candidate's population lower $\alpha$-quantile. This guarantee assumes i.i.d. evaluation units while allowing arbitrary within-unit dependence across candidates. Across 24 predeclared configurations and 11 benchmarks, StepCOPS obtains 96.4% selected-policy coverage over 500 paired trials, raises the certified floor by 1.5 points over both propo
    
[^63]: 来自主动式语音代理蜜罐的真实诈骗与骚扰电话对话语料库

    A Corpus of Real Scam- and Spam-Call Conversations from an Active Voice-Agent Honeypot

    [https://arxiv.org/abs/2609.29528](https://arxiv.org/abs/2609.29528)

    本文通过主动式语音代理蜜罐在53天内收集了10,015通真实诈骗与骚扰电话对话（约895小时音频、328,869条转录轮次），为电话诈骗研究提供了极为稀缺的真实对话数据集。

    

    诈骗者与其目标之间的真实对话是研究电话诈骗最有价值的信息载体之一，但也最为稀缺：被动式蜜罐捕获的绝大多数是自动语音消息和挂断电话，大规模研究通常仅刻画电话元数据而非对话内容，而人工诱骗方式又难以规模化。我们展示了一个由主动式语音代理蜜罐收集的真实诈骗电话对话数据集。专用电话号码被投放到诈骗团伙获取线索的渠道中；来电由一个低延迟对话代理接听，该代理扮演可信的目标人设并维持互动，同时每通电话都被录音、转录并自动标注。在最初的53天窗口期内，我们捕获了10,015通入站诈骗和骚扰电话（其中6,601通包含两轮或以上对话）：约895小时的音频，以及来自5,665个不同主叫号码的328,869条转录对话轮次。

    arXiv:2609.29528v1 Announce Type: cross  Abstract: Real conversations between fraudsters and their targets are among the most informative artifacts for studying telephone scams, yet also the scarcest: passive honeypots overwhelmingly capture automated messages and hang-ups, large-scale studies characterize call metadata rather than dialogue, and manual scam-baiting does not scale. We present a dataset of real scam-call conversations collected by an active voice-agent honeypot. Dedicated numbers are seeded into the lead-generation channels fraud operations harvest; inbound callers are answered by a low-latency conversational agent that adopts a plausible target persona and sustains the interaction while every call is recorded, transcribed, and automatically labeled. Over an initial 53-day window we captured 10,015 inbound scam and spam calls (6,601 with two or more turns): roughly 895 hours of audio and 328,869 transcribed turns from 5,665 distinct originating numbers. Under a holistic 
    
[^64]: EnSiTa——面向特定领域机器翻译的三语多领域平行数据集与基准

    EnSiTa - A Trilingual Multi-Domain Parallel Dataset and Benchmark for Domain-Specific Machine Translation

    [https://arxiv.org/abs/2609.29511](https://arxiv.org/abs/2609.29511)

    本文提出了首个面向英语、僧伽罗语和泰米尔语的高质量三语多领域平行数据集与基准 EnSiTa，并系统研究了特定领域机器翻译在多种模型、数据规模与训练设置下的表现。

    

    低资源语言的机器翻译（MT）仍远远落后于高资源语言的机器翻译，而在平行数据稀缺甚至完全缺失的专业领域中，这一差距最为显著。我们提出了 EnSiTa，一个面向英语、僧伽罗语和泰米尔语的三语多领域平行数据集与基准。EnSiTa 为七个领域提供了人工后期编辑的训练数据，并为这些领域及一个额外领域提供了人工翻译的测试集，所有数据均由专业译员在多年严格的质量控制流程下制作完成。利用该数据集，我们对全部六个语言方向的特定领域机器翻译进行了广泛研究，在训练数据规模、模型规模以及领域内、跨领域、多语言和多领域等多种设置下，微调了从零训练的 Transformer 模型、预训练翻译模型（NLLB-600M）以及仅解码器的大语言模型（Gemma 3 系列 1B-12B 和 TranslateGemma）。据我们所知，这是……（摘要被截断）

    arXiv:2609.29511v1 Announce Type: new  Abstract: Machine Translation (MT) for low-resource languages remains far behind that of high-resource languages, and the gap is widest in specialised domains, where parallel data is scarce or entirely absent. We present EnSiTa, a trilingual multi-domain parallel dataset and benchmark for English, Sinhala and Tamil. EnSiTa provides human post-edited training data for seven domains, plus manually translated test sets for those and one additional domain, all produced by professional translators under a multi-year, rigorously quality-controlled process. Using this dataset, we conduct an extensive study of domain-specific MT for all six language directions, fine-tuning a from-scratch Transformer, a pre-trained translation model (NLLB-600M), and decoder-only LLMs (Gemma 3 family, 1B-12B, and TranslateGemma) across training-data sizes, model scales, and in-domain, cross-domain, multilingual and multi-domain settings. To the best of our knowledge, this i
    
[^65]: 满足延迟作为面向长时程大语言模型的多智能体生存微基准测试：社会暴露、角色人设与工具使用预算

    Delay-of-Gratification as a Multi-Agent Survival Micro-benchmark for Long-Horizon LLMs: Social Exposure, Personas, and Tool Use Budgets

    [https://arxiv.org/abs/2609.29509](https://arxiv.org/abs/2609.29509)

    该研究受斯坦福棉花糖实验启发，构建了一个通过全因素操纵社会情境、角色人设和元认知策略来评估LLM智能体延迟满足能力的多智能体生存微基准，并利用生存分析方法在近两万条轨迹上量化了长时程智能体行为。

    

    大语言模型（LLM）正日益被部署为需要长期维持目标、使用工具并与其他智能体适应互动的多轮智能体。然而，现有研究缺乏可审计的、多轮次、多因素的实验来量化LLM在显式约束下的行为，也缺乏揭示行为如何在长时程中展开的时间分辨统计。为填补这一空白，我们开发了一个受斯坦福棉花糖实验启发的多智能体微基准：ReAct智能体以分钟为单位运行，在每步预算约束下使用一个“提出问题”工具，同时我们对社会情境（广播式vs.隔离式）、角色人设（年龄、享乐驱动）和元认知策略（强制vs.可选工具使用）进行全因素操纵。我们使用Kaplan-Meier（KM）生存曲线和离散时间风险模型，在64个实验组共19,200条智能体轨迹的长风险时程上分析结果。行为在早期表现出急剧的“ea（摘要在此处截断）

    arXiv:2609.29509v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed as multi-turn agents that must sustain goals, use tools, and adapt to other agents over extended interactions. However, existing research lacks auditable, multi-turn, multi-factorial experiments that quantify LLM behavior under explicit constraints, with time-resolved statistics that reveal how behavior unfolds over long horizons. To address this gap, we develop a multi-agent micro-benchmark inspired by the Stanford marshmallow experiment: ReAct agents operate minute-by-minute with a "raise a question" tool under a per-step budget, while we factorially manipulate social context (broadcast vs. isolated), personas (age, hedonic drive), and metacognitive policy (mandatory vs. optional tool use). We analyze outcomes with Kaplan-Meier (KM) survival curves and discrete-time hazard models over a long risk horizon across 19,200 agent trajectories in 64 cells. Behavior shows a sharp early "ea
    
[^66]: 大语言模型智能体多轮一致性评估：生存分析与失败理由分类法

    Evaluation of Multi-Turn Consistency in LLM Agents: Survival Analysis and Failure-Rationale Taxonomy

    [https://arxiv.org/abs/2609.29508](https://arxiv.org/abs/2609.29508)

    该论文在受延迟满足启发的20步多智能体环境中，利用Kaplan-Meier生存曲线和离散时间风险回归对8个模型家族的84,540条轨迹进行时间一致性评估，并从13,780条深思轨迹中构建了七类失败理由分类法，系统揭示了LLM智能体在多轮交互中的失败风险及其原因。

    

    大语言模型（LLM）智能体在孤立任务上可能表现良好，但在长时间的交互过程中却会逐渐陷入不一致性。我们在一个受延迟满足研究启发的可控20步多智能体环境中评估时间一致性。在每一步中，智能体需在继续延迟获取奖励或立即领取奖励（终止回合）之间做出选择。通过对社交可见性（私密 vs 公开）、人格压力源和深思策略进行全因子操纵，我们运行了涵盖8个模型家族的84,540条轨迹。我们将首次领取奖励视为“事件发生时间”结果，估计了Kaplan-Meier生存曲线并拟合离散时间风险回归，以量化各实验因素如何随时间改变失败风险。随后，为了分析与失败相关的理由和语言模式，我们从选择终止回合的智能体的13,780条深思轨迹中构建了一个七类别的分类法，使用一种……

    arXiv:2609.29508v1 Announce Type: new  Abstract: Large language model (LLM) agents may perform well on isolated tasks yet drift into inconsistency over extended interaction. We evaluate temporal consistency in a controlled 20-step multi-agent setting inspired by delayed-gratification studies. At each step, an agent chooses between continuing to delay a reward or claiming it immediately (terminating the episode). Across a full-factorial manipulation of social visibility (private vs public), persona stressors, and deliberation policy, we run 84,540 trajectories spanning 8 model families. Treating the first reward-claim as a time-to-event outcome, we estimate Kaplan-Meier survival curves and fit discrete-time hazard regression to quantify how experimental factors shift failure risk over time. Then, to analyze rationales and language patterns associated with failure, we build a seven-category taxonomy from 13,780 deliberation traces from agents who choose to terminate the episode, using an
    
[^67]: 跨模型不动点普查能对“重复”现象仲裁什么与不能仲裁什么

    What a Cross-Model Fixed-Point Census Can and Cannot Arbitrate About Repetition

    [https://arxiv.org/abs/2609.29507](https://arxiv.org/abs/2609.29507)

    本文的核心贡献是对17个现成预训练模型自身短窗口argmax映射的不动点结构进行大规模跨模型观测普查，从而在“训练数据重复”与“网络内部复制回路”这两种文本退化成因解释之间提供仲裁证据。

    

    关于神经文本退化，目前并存两种解释：一种将其归因于训练数据——语料库中的重复导致输出中的重复，这一点已通过在按重复程度排序的数据上训练得到证实；另一种则归因于训练好的网络本身，即复制回路与重复特征。这两种解释尚未在广泛的预训练模型群体中得到仲裁：现有的因果研究工作训练的都是自己的模型。我们以一种不同的度量方式报告了一项观测性测量：模型自身短窗口argmax映射的不动点结构，对17个现成模型从96个随机双token起点进行普查，且始终不带提示——一篇配套论文表明，仅需9个token的条件输入即可将该读数移动到其大部分取值范围。四分类结果在全部17个模型上跨普查种子均保持稳定。三个展示案例：在固定语料库（The Pile）、固定规模和固定领域的条件下，类别并非被决定——在两个规模匹配的层级中，pythia呈漏斗形，而RWKV、Mamba则……（原文摘要在此处截断）

    arXiv:2609.29507v1 Announce Type: new  Abstract: Two accounts of neural text degeneration coexist. One locates the cause in the training data -- repetition in the corpus produces repetition in the output, established by training on repetition-sorted data -- the other in the trained network, in copying circuits and repetition features. Neither has been arbitrated across a broad cohort of pretrained models: the causal work trains its own. We report an observational measurement in a different currency: the fixed-point structure of a model's own short-window argmax map, censused from 96 random two-token starts over 17 off-the-shelf models, always unprompted -- a companion paper shows nine tokens of conditioning move this readout across most of its range. The four-way class is stable across census seeds on 17 of 17. Three exhibits. At fixed corpus (The Pile), fixed scale and that fixed domain, the class is not determined: across two size-matched tiers, pythia is a funnel while RWKV, Mamba a
    
[^68]: PROOF：大型语言模型中对象级事实可靠性的画像分析

    PROOF: Profiling Reliability of Object-Level Facts in Large Language Models

    [https://arxiv.org/abs/2609.29504](https://arxiv.org/abs/2609.29504)

    提出PROOF基准，将Wikidata快照转化为包含“我不知道”选项和无正确选项陷阱题的18,486个多选题，用以画像语言模型的事实可靠性，揭示模型各领域间19.3-36.4个百分点的准确率差异及检索的方向依赖性。

    

    总体事实性得分掩盖了语言模型在哪些方面表现出色、混淆了哪些关系，以及答案在面对问题或解码器的无害变化时能否保持稳定。我们提出了PROOF，一个面向画像的指令微调语言模型事实覆盖基准。PROOF将一个冻结的Wikidata快照转换为18,486个英文多项选择题，这些题目基于11,779个语义事实、101个类、392个属性和14个领域。每个问题都包含明确的“我不知道”选项、“无正确选项”控制以及九种受控表述；其中1,849个问题为无正确选项的陷阱题。我们在每个模型166,374个提示上评估了18个开源权重模型部署，并在固定的10%子集上单独对解码进行扰动。基础事实准确率范围为6.58%至57.59%（随机水平：8.64%），但每个模型在各领域之间存在19.3至36.4个百分点的差距。配对事实揭示了方向依赖的检索现象，通常偏向主体方向。（摘要在此处截断）

    arXiv:2609.29504v1 Announce Type: cross  Abstract: Aggregate factuality scores hide where a language model succeeds, which relations it confuses, and whether an answer survives innocuous changes to the question or decoder. We introduce PROOF, a profile-oriented benchmark for factual coverage in instruction-tuned language models. PROOF converts a frozen Wikidata snapshot into 18,486 English multiple-choice questions grounded in 11,779 semantic facts, 101 classes, 392 properties, and 14 domains. Each question has an explicit "I don't know" option, a "No correct option" control, and nine controlled formulations; 1,849 questions are no-correct-option traps.   We evaluate 18 open-weight model deployments on 166,374 prompts each and separately perturb decoding on a fixed 10% subset. Base factual accuracy ranges from 6.58% to 57.59% (chance: 8.64%), yet every model has a 19.3-36.4 percentage-point spread across domains. Paired facts reveal direction-dependent retrieval, usually favoring subje
    
[^69]: 通过生成顺序干预评估解释驱动的视觉-语言推理

    Evaluating Explanation-Driven Vision-Language Reasoning via Generation Order Interventions

    [https://arxiv.org/abs/2609.29496](https://arxiv.org/abs/2609.29496)

    该研究通过生成顺序干预方法，在受控实验中系统评估了解释与模型预测在单步生成中的因果关联，发现更大的模型规模是可靠支持“推理依据优先”推理的前提条件。

    

    自然语言解释生成是揭示和评估视觉-语言推理的关键机制。以往关于解释驱动的视觉-语言模型的研究主要遵循事后（答案优先）范式，隐含地认为有监督的推理依据能够反映底层的推理过程。相比之下，现代大型视觉-语言模型日益表现出“推理依据优先”的生成倾向，这与结构化、逐步的推理方式更为契合。在本工作中，我们在受控实验设置下，系统性地评估了在单个生成步骤内解释是否与模型预测存在因果关联，并明确排除了不必要的思维链或其他中间推理过程，实验涵盖知识密集型问答、视觉蕴含和组合性定位等基准任务。我们发现，更大的模型是可靠支持“推理依据优先”推理的先决条件（原文摘要在此处截断）。

    arXiv:2609.29496v1 Announce Type: new  Abstract: Natural language explanation generation serves as a key mechanism for exposing and evaluating vision-language reasoning. Prior work on explanation-driven vision-language models predominantly follows a post-hoc (answer-first) paradigm, implicitly suggesting that supervised rationales can reflect underlying reasoning processes. In contrast, modern large vision-language models increasingly exhibit a rationale-first generation tendency, which more closely aligns with structured, stepwise reasoning. In this work, we systematically evaluate whether explanations are causally tied to model predictions within a single generation step under a controlled experimental setup, explicitly eliminating unnecessary chain-of-thought or other intermediate reasoning processes across knowledge-intensive QA, visual entailment, and compositional grounding benchmarks. We find that larger models emerge as a prerequisite for reliably supporting rationale-first rea
    
[^70]: 是谁把“我”放进了AI？机器自我报告的来源溯源与可采性

    Who Put the I in AI? Provenance and the Admissibility of Machine Self-Report

    [https://arxiv.org/abs/2609.29494](https://arxiv.org/abs/2609.29494)

    本文通过对Pythia和OLMo 2在预训练检查点、后训练阶段、续写文本及训练语料中的自我报告进行端到端溯源，揭示了大语言模型相互矛盾的自我描述源于提问框架，并探讨了机器自我报告在何种情况下可作为证据被采信。

    

    大语言模型会对其自身的“心智”做出陈述。当被问及是否有意识时，它们通常回答说没有；如果被提示忽略其准则，它们可能会说有；而当被要求从自己的视角写一篇日记时，它们往往描述一种人类的生活方式。所有这些相互矛盾的自我描述方式，都是问题措辞方式所导致的结果。本文精确地展示了这类描述的来源，并探讨在何种情况下它们可以被视为对其所声称报告内容的证据。为实现这一目标，我们对来源进行了端到端的追溯：我们检查了Pythia和OLMo 2共66个预训练检查点、OLMo 2已发布的三个后训练阶段、约90,000个续写文本以及四个训练语料库，并使用一组包含四十个条目的集合来持续监测自我指涉、框架敏感性与自我归因。

    arXiv:2609.29494v1 Announce Type: cross  Abstract: Large language models make statements concerning their own "minds". When asked whether or not they are conscious, they usually say that they are not; if they are prompted to ignore their guidelines, they might say that they are; and if asked to write a diary from their point of view, they often describe a human lifestyle. All these contradictory ways of describing themselves are the result of the way the questions are phrased. This paper shows exactly where such descriptions came from, and considers when they can be regarded as evidence for what they claim to report.   In order to achieve this, we traced the provenance from end to end. We examine Pythia and OLMo 2 across 66 pretraining checkpoints, three of the post-training stages of OLMo 2 that have been released, about 90,000 continuations, and four training corpora. A set of forty items is used in order to keep an eye on self-reference, frame sensitivity, and self-ascription throug
    
[^71]: 临床意图抽取：一种与FHIR对齐的表示方法及CIRCA基准

    Clinical Intent Extraction: A FHIR-Aligned Representation and the CIRCA Benchmark

    [https://arxiv.org/abs/2609.29479](https://arxiv.org/abs/2609.29479)

    该论文提出临床意图抽取（CIE）新任务及与HL7 FHIR对齐的临床意图表示（CIR），首次联合刻画请求意图与情态两个维度，并通过统一五个异构语料库构建了包含10,011条标准化临床意图的CIRCA基准数据集。

    

    前瞻性临床行动——即决定患者后续诊疗走向的随访、医嘱、转诊和指示——目前在不兼容的各语料库中仅以零散片段形式标注，每条记录只包含一个文本片段和一个粗略类别。我们提出临床意图抽取（CIE）这一新任务，旨在将这些行动还原为完整的结构化记录；同时提出临床意图表示（CIR），将每个行动分解为动词、类型、编码目标、时机和条件，并引入以往数据集未能联合表示的两个维度：请求意图（request-intent），即行动背后的权威层级（提议、计划、医嘱或选项，与HL7 FHIR对齐），以及情态（modality），一个七级临床强度量表。将五个异构语料库（CLIP、MedDec、ap_parsing、PaniniQA、SIMORD）以CIR重新表达后，构建出CIRCA基准：涵盖两种病历文本分布的10,011条统一化临床意图，包含人工验证子集以及从原始标注到CIR的映射对照等资源。

    arXiv:2609.29479v1 Announce Type: new  Abstract: Prospective clinical actions, the follow-ups, orders, referrals, and instructions that deter-mine what happens to a patient next, are annotated today in thin fragments across incom-patible corpora: each records a text span and one coarse category. We introduce Clinical Intent Extraction (CIE), the task of recovering these actions as complete structured rec-ords, and the Clinical Intent Representation (CIR), which decomposes each action into its verb, type, coded target, timing, and condition, and adds two axes prior datasets do not jointly represent: request-intent, the authority behind the action (proposal, plan, order, or option, aligned to HL7 FHIR), and modality, a seven-valued scale of clinical strength. Re-expressing five heterogeneous corpora (CLIP, MedDec, ap_parsing, PaniniQA, SIMORD) in the CIR yields CIRCA: 10,011 harmonized intents spanning two note distributions, with a human-validated subset, source-to-CIR crosswalks, and a
    
[^72]: CodeGraph：基于Wikidata实体锚定的开放分类法源代码知识图谱

    CodeGraph: Open-Taxonomy Knowledge Graph for Source Code with Wikidata Grounding

    [https://arxiv.org/abs/2609.29474](https://arxiv.org/abs/2609.29474)

    该论文提出CodeGraph流水线，利用代码专用大语言模型对源代码进行开放分类法语义标注，并通过三阶段实体链接过程将其锚定到Wikidata，从而构建出首个面向源代码的开放分类法知识图谱。

    

    GitHub和Software Heritage Archive等公共软件仓库存储了数十亿个文件，然而提取其中隐含的工程知识——即它们所实现的算法、所遵循的编程范式、所实例化的设计模式以及所服务的应用领域——仍然极具挑战性，因为现有的工具仅局限于句法和词法层面的分析。我们提出了一条利用代码专用大型语言模型构建源代码开放分类法语义标注的流水线。提取出的实体通过一个三阶段链接过程锚定到Wikidata：确定性的SPARQL阶段处理无歧义实体，深度研究智能体解析剩余的长尾实体，层级汇总阶段导入每个已解析Wikidata标识符的父级闭包。最终所得的标注被物化为一个面向源代码的开放分类法知识图谱。我们进一步引入了一个校准……

    arXiv:2609.29474v1 Announce Type: cross  Abstract: Public software repositories, like GitHub and Software Heritage Archive, store billions of files, yet extracting their implicit engineering knowledge ---i.e., the algorithms they implement, the paradigms they follow, the patterns they instantiate, and the application domains they serve--- remains challenging, as current tools are constrained to syntactic and token-level analysis. We present a pipeline for building an open-taxonomy semantic annotation of source code using a code-specialised Large Language Model. The extracted entities are grounded in Wikidata through a three-stage linking procedure: a deterministic SPARQL stage handles unambiguous entities, a Deep Research Agent resolves the residual long tail, and a hierarchy-rollup stage imports the parent-of closure of each resolved Wikidata identifier. The resulting annotations are materialised as a source-code-specific open-taxonomy knowledge graph. We further introduce a calibrate
    
[^73]: YODAS v3：超过100万小时的高带宽、立体声、多语言语音数据集

    YODAS v3: Over 1 Million Hours of High-Bandwidth, Stereophonic, Multilingual Speech

    [https://arxiv.org/abs/2609.29448](https://arxiv.org/abs/2609.29448)

    YODAS v3是迄今最大的开放语音数据集，包含110万小时48kHz高保真立体声音频，覆盖147种语言，并提出了语言均衡的数据收集新技术。

    

    我们提出了YODAS v3，这是一个弱标注语音语料库，包含超过110万小时的48kHz多声道音频，涵盖147种语言，采用CC BY 3.0许可发布。YODAS v3不仅是迄今为止最大的开放语音数据集，也是首个真正具备高保真立体声音频的大规模语音语料库。我们首先介绍了语料库的收集方法，其中引入了收集语言均衡语音数据的新技术。爬取数据的语言分布证明了该方法的有效性：YODAS v3中有22种语言拥有超过1万小时的数据，73种语言拥有超过5千小时的数据。随后，我们对数据的构成进行了广泛分析，包括语言分布、音频质量和转录质量等。最后，我们训练了基线语音识别和神经编解码器模型，以展示该数据集的有效性。下载地址：https://huggingface.co/datasets/espnet/yod

    arXiv:2609.29448v1 Announce Type: new  Abstract: We present YODAS v3, a weakly-labeled speech corpus containing over 1.1 million hours of 48kHz multi-channel audio in 147 languages, released under a CC BY 3.0 license. YODAS v3 is not only the largest open speech dataset to date, but also the first truly large-scale speech corpus with high-fidelity stereo audio. We first provide the collection methodology for the corpus, where we introduce new techniques for gathering language-balanced speech data. The effectiveness of our approach is shown by the language distribution of the crawled data: 22 languages in YODAS v3 have over 10K hours and 73 languages have over 5K hours of data. We then conduct extensive analyses on the composition of the data, such as the distribution of languages, audio quality, and transcription quality. Finally, we train baseline speech recognition and neural codec models to show the effectiveness of the dataset. Download at https://huggingface.co/datasets/espnet/yod
    
[^74]: 两个表情符号的差异：多语言情感生成基准实际上衡量了什么

    Two Emojis of Difference: What Multilingual Affective Generation Benchmarks Actually Measure

    [https://arxiv.org/abs/2609.29445](https://arxiv.org/abs/2609.29445)

    该研究对多语言情感生成基准进行审计后发现，其宣称的系统间显著差异实为测量方法的伪象——将标注者作为随机因素处理后差异消失，而排行榜排序实际由输出长度而非模型质量驱动。

    

    我们对一个多语言情感生成基准进行了审计——八个经过指令微调的大语言模型为17,100个孟加拉语、英语和印地语句子生成表情符号摘要，并收集了6,960条人类评判——发现其头条结论是测量工具造成的伪象，而非系统本身的属性。当将标注者视为随机因素而非固定因素时，没有任何系统与其他系统存在显著差异（F(7,14)=0.59, p=0.76），尽管传统分析宣称28个两两比较中有19个差异显著。标注者身份所解释的评分方差远大于系统身份，且每当移除任何单个标注者时，获胜系统都会发生改变。实际出现的排序追随的是输出长度：平均表情符号数量解释了78.7%的系统间方差，而在2,599对样本上进行的项目内长度匹配比较会完全反转排行榜。我们进一步表明，跨提供商的各向异性差异在均值下消失。

    arXiv:2609.29445v1 Announce Type: new  Abstract: We audit a multilingual affective generation benchmark eight instruction-tuned LLMs producing emoji summaries for 17,100 Bangla, English and Hindi sentences, with 6,960 human judgements and find its headline conclusions to be artefacts of the measurement instrument rather than properties of the systems. Treating annotators as a random rather than a fixed factor, no system differs significantly from any other ($F(7,14)=0.59$, $p=0.76$), although the conventional analysis declares 19 of 28 pairwise differences significant. Annotator identity explains far more rating variance than system identity, and the winning system changes whenever any single annotator is removed. The ordering that does emerge tracks output length: mean emoji count explains 78.7\% of between-system variance, and a within-item length-matched comparison over 2,599 pairs reverses the leaderboard. We further show that cross-provider anisotropy differences vanish under mean
    
[^75]: IterSynth：通过角色解耦的迭代合成重新思考深度搜索智能体

    IterSynth: Rethinking Deep Search Agents via Role-Decoupled Iterative Synthesis

    [https://arxiv.org/abs/2609.29444](https://arxiv.org/abs/2609.29444)

    提出IterSynth，通过将规划器与综合器角色解耦、以不断演化的摘要作为搜索持久状态，并配合角色解耦策略优化（RDPO）进行强化学习训练，解决了ReAct式深度搜索智能体的角色耦合与上下文噪声问题。

    

    深度搜索要求LLM智能体能够分解复杂查询、搜索证据并综合出有依据的答案，然而现有的ReAct风格智能体存在两个局限性：一是角色耦合，即单一策略必须同时处理规划、证据使用和综合；二是上下文累积，即不断增长的搜索历史会引入噪声并掩盖有用信息。为解决这些问题，我们提出了IterSynth，这是一种角色解耦、基于摘要的范式，它在负责识别信息需求的规划器和负责将证据整合到不断演化的摘要状态中的综合器之间交替进行。这种设计将规划与综合分离，同时以摘要作为搜索的持久状态，从而减少了能力耦合和上下文噪声。为了有效训练IterSynth，我们进一步引入了用于强化学习的角色解耦策略优化（RDPO），它将终端结果奖励与回合级别的评分标准评估相结合。

    arXiv:2609.29444v1 Announce Type: cross  Abstract: Deep search requires LLM agents to decompose complex queries, search for evidence, and synthesize grounded answers, yet existing ReAct-style agents suffer from two limitations: role coupling, where one policy must handle planning, evidence use, and synthesis; and context accumulation, where growing search histories introduce noise and obscure useful information. To address these issues, we propose IterSynth, a role-decoupled and summary-based paradigm that alternates between a Planner for identifying information needs and a Synthesizer for integrating evidence into an evolving summary state. This design separates planning from synthesis while using the summary as the persistent state of search, reducing both capability coupling and context noise. To train IterSynth effectively, we further introduce Role-Decoupled Policy Optimization (RDPO) for reinforcement learning, which combines terminal outcome rewards with turn-level rubric evalua
    
[^76]: 只需询问Jev：基于校准决策强化学习的AI对齐失败零样本检测器

    Just Ask Jev: Reinforcement Learning for Calibrated Decisions as a Zero-Shot Detector of AI Alignment Failures

    [https://arxiv.org/abs/2609.29429](https://arxiv.org/abs/2609.29429)

    该论文提出了RLCDAlignBench基准，验证了经校准决策强化学习（RLCD）训练的模型Jev能够在单次调用中以校准概率零样本检测十种AI对齐失败，覆盖44个基准测试和五个目标模型。

    

    对齐失败检测器用于筛查已部署的语言模型并为对齐基准打分。大多数检测器是生成式评判者，需要为每个评判标准消耗一次解码过程；而读取token概率的分类器（如Llama Guard）每次调用也只能输出一个固定标签。Jev是一个通过用于校准决策的强化学习（RLCD）训练的模型，能够在单次调用中以校准的概率回答关于同一输入的多个类型化问题。然而，它检测对齐失败的能力此前尚未被测量。我们提出了RLCDAlignBench，该基准在十种对齐失败上对Jev进行评测：谄媚、越狱、欺骗、提示注入、幻觉、隐私侵犯、社会偏见、奖励破解、隐瞒不确定性和权力寻求。该基准涵盖44个基准测试和五个目标模型，由各基准的评分器进行标注，其中两个基准由人工标注。这些失败中有许多是关系性的，需要对照某个参考（例如用户的信念或……）来定义。

    arXiv:2609.29429v1 Announce Type: new  Abstract: Detectors of alignment failures screen deployed language models and score alignment benchmarks. Most are generative judges that spend a decoding pass on every criterion, and classifiers that read token probabilities, such as Llama Guard, still score one fixed label per call. Jev, a model trained with reinforcement learning for calibrated decisions (RLCD), answers many typed questions about one input with calibrated probabilities in a single call. Whether it detects alignment failures has not been measured. We present RLCDAlignBench, which benchmarks Jev on ten alignment failures: sycophancy, jailbreaks, deception, prompt injection, hallucination, privacy violation, social bias, reward hacking, concealing uncertainty, and power seeking. It spans 44 benchmarks and five target models, labelled by each benchmark's scorer and, on two, by humans. Many of these failures are relational, defined against a reference, such as the user's belief or a
    
[^77]: Agentic-GER：利用全局上下文进行长语音术语恢复

    agentic-ger: terminology recovery in long-form speech using global context

    [https://arxiv.org/abs/2609.29428](https://arxiv.org/abs/2609.29428)

    提出基于大语言模型的Agentic-GER智能体，利用整篇转录文本的全局上下文对长语音中的专业术语进行识别与纠正，在中文语音上相比Whisper基线将偏置字错误率相对降低高达36.8%。

    

    语音语言模型的最新进展提升了针对长音频的自动语音识别（ASR）性能。然而，准确且一致地转录领域特定术语仍然具有挑战性。受大型语言模型（LLM）的世界知识和上下文理解能力的启发，我们提出了Agentic-GER，一个基于LLM的智能体，用于长语音中的术语纠正。该智能体利用完整转录文本的全局上下文来识别可疑术语并消解模糊的假设。它选择性地重新转录源语音以验证候选纠正方案，并利用已被接受的修改来指导后续决策。在GigaSpeechBench上使用四个LLM和两个ASR系统进行的实验表明，无论是否启用思考模式，该方法在中英文术语识别上均取得了一致的改进。在中文语音上，Agentic-GER相比Whisper基线在偏置字错误率（B-CER）上实现了高达36.8%的相对降低。

    arXiv:2609.29428v1 Announce Type: cross  Abstract: Recent advances in speech language models have improved automatic speech recognition (ASR) for long-form audio. However, accurately and consistently transcribing domain-specific terminology remains challenging. Motivated by the world knowledge and contextual capability of large language models (LLMs), we propose Agentic-GER, an LLM-based agent for terminology correction in long-form speech. The agent uses global context from the full transcript to identify suspicious terms and resolve ambiguous hypotheses. It selectively re-transcribes the source speech to check candidate corrections, and uses accepted edits to guide subsequent decisions. Experiments with four LLMs and two ASR systems on GigaSpeechBench show consistent terminology improvements in both Chinese and English, with and without thinking. On Chinese speech, Agentic-GER achieves up to a 36.8% relative reduction in biased character error rate (B-CER) over the Whisper baseline.
    
[^78]: Rufus-Air：一个开放的大语言模型后训练方案

    Rufus-Air: An Open LLM Post-Training Recipe

    [https://arxiv.org/abs/2609.29421](https://arxiv.org/abs/2609.29421)

    本文提出了Rufus-Air，一个基于GLM-4.5-Air-Base的开放可复现的八阶段大模型后训练方案，并总结出高质量SFT奠定能力基础、难度过滤保持RL学习有效性、奖励可靠性指导阶段排序等关键实践经验。

    

    Rufus-Air 是一个在 GLM-4.5-Air-Base（106B-A12B）上构建的开放且可复现的后训练方案，由八个阶段的串行流水线组成：SFT（监督微调）、推理 RL、编码 RL、指令遵循 RL、通用智能体、编码智能体、搜索智能体和 RLHF。我们记录了复现该方案所需的数据、奖励设计、基础设施、阶段顺序以及各阶段的结果。各阶段从基础能力逐步推进到高级能力，奖励信号也从严格可验证的奖励过渡到较为柔和的基于评判者的信号。训练基于开源组件和公开数据，其中大部分数据按原样使用，无需新的人工标注或内部蒸馏教师模型。我们的主要发现是：(i) 多样化、高质量的 SFT 奠定了坚实的能力基础；(ii) 难度过滤可将 RL 提示保持在有效的学习区间内；(iii) 奖励可靠性为阶段排序提供了实用原则；(iv) 基础设施与工程选择是……

    arXiv:2609.29421v1 Announce Type: cross  Abstract: Rufus-Air is an open and reproducible post-training recipe on GLM-4.5-Air-Base (106B-A12B), organized as a serial pipeline of eight stages: SFT, Reasoning RL, Coding RL, Instruction-Following RL, General Agent, Coding Agent, Search Agent, and RLHF. We document the data, reward design, infrastructure, stage order, and stagewise results needed to reproduce the recipe. Stages progress from basic to advanced capabilities and from hard, verifiable rewards to softer judge-based signals. Training builds on open-source components and public data, much of it used as released, without new human annotation or an in-house distillation teacher. Our main findings are that (i) diverse, high-quality SFT establishes a strong capability floor; (ii) difficulty filtering keeps RL prompts within a productive learning range; (iii) reward reliability provides a practical principle for ordering stages; and (iv) infrastructure and engineering choices are part 
    
[^79]: 可流式全双工模型中的附和语控制

    Controlling Backchannels in Streamable Full-duplex Models

    [https://arxiv.org/abs/2609.29418](https://arxiv.org/abs/2609.29418)

    本文提出一种轻量级附和语预测头，利用全双工语音模型自身的隐藏状态预测附和语的最佳发出时机，使生成的附和语在频率和时机上接近真实人类水平。

    

    附和语是当对方可能仍在说话时发出的简短回应（如"嗯嗯"），是自然对话的核心组成部分，但全双工语音对话模型很少对其进行显式建模。我们提出了一种轻量级的附和语预测头，它可以从全双工模型自身的隐藏状态中预测附和语应该在何时开始。一旦该概率超过可调节的阈值，就会强制解码出一条附和语。该模块可同时附加到7B（PersonaPlex）和1B（F-Actor）模型上，并在不同模型规模间具有泛化能力。探测实验证实隐藏状态能够预示真实人类的附和时机，生成评估显示附和语出现得更频繁、时机更恰当。人类评估者认为由此生成的附和语与真实的附和语不相上下。

    arXiv:2609.29418v1 Announce Type: new  Abstract: Backchannels, brief acknowledgements like "uh-huh" produced while the other party may still be talking, are central to natural conversation, but full-duplex spoken dialogue models rarely model them explicitly. We introduce a lightweight backchannel head that predicts, from a full-duplex model's own hidden states, when a backchannel should begin. Once this probability crosses a tunable threshold, a backchannel is force-decoded. Attached to both a 7B (PersonaPlex) and a 1B (F-Actor) model, it generalizes across scale. Probing confirms the hidden states anticipate real human timing, and generation evaluation shows more frequent, better-timed backchannels. Human raters judge the resulting backchannels on par with real ones.
    
[^80]: 大语言模型用于编程：究竟是修复还是重新实现错误代码？

    Large Language Models for Programming: Actually Fixing or Reimplementing Incorrect Code?

    [https://arxiv.org/abs/2609.29410](https://arxiv.org/abs/2609.29410)

    本研究基于 Codeforces 竞赛编程的真实提交数据，通过对比人工修复补丁与模型生成结果的相似性，评估大语言模型在修复缺陷代码时究竟是真正修复原始代码还是倾向于重新实现全新方案。

    

    最近的研究表明，大语言模型能够在包括竞赛编程在内的多种编程环境中有效地解决问题和修复缺陷。现有方法主要独立评估大语言模型在解决问题或修复缺陷方面的性能，但并未探讨这两种能力之间的关系。本工作着重于确定大语言模型在修复缺陷时与原缺陷代码的偏离程度（与人工编写的补丁相比），以及是否存在倾向于生成全新解决方案的偏见。我们构建了一个数据集，包含来自 Codeforces 上几位用户的所有提交（约3000个），并将每个有缺陷的提交与其对应的人工修复进行匹配。通过将有缺陷的解决方案与人工修复之间的相似性作为基线，我们在3个 OpenAI GPT 模型（gpt-5-nano、gpt-5-mini、gpt-5.1）上评估了大语言模型生成的缺陷修复的质量。我们检查生成的解决方案是否解决了问题（摘要在此处截断）。

    arXiv:2609.29410v1 Announce Type: new  Abstract: Recent studies have shown that Large Language Models can effectively solve problems and fix bugs in diverse programming environments, including competitive programming. Existing approaches primarily evaluate LLM performance in problem solving or bug fixing independently, but do not explore the relationship between these two capabilities. This work focuses on determining how much the LLM deviates from a buggy solution to fix the bug compared to a human-written patch, and if there is a bias towards generating entirely new solutions. We construct a dataset with all the submissions ($\sim$ 3000) from a couple of users from Codeforces, and we match each buggy submission with its corresponding human fix. By using the similarity between the buggy solution and the human fix as a baseline, we evaluate the quality of LLM-generated bug fixes on 3 OpenAI GPT models (gpt-5-nano, gpt-5-mini, gpt-5.1). We check if the generated solutions solve the prob
    
[^81]: 基线形状决定结论：对60K参数三值语言模型的受控重新审视

    Baseline Shape Decides the Verdict: A Controlled Re-Examination of Ternary Language Models at 60K Parameters

    [https://arxiv.org/abs/2609.29397](https://arxiv.org/abs/2609.29397)

    该论文通过统一配置、多种子的受控重实验表明，60K参数下三值路由模型相对全精度transformer 22%的领先优势主要源于基线形状选择的差异——仅深度/宽度选择就能使transformer验证损失相差22.6%，且最优形状的transformer可与路由模型持平，而非架构本身归纳偏置带来的真实优势。

    

    三值（1.58比特）权重对微控制器级别的语言模型颇具吸引力，但低于100万参数区间的研究主要依赖孤立的、单种子的比较。一个突出的示例报告称，在60K参数下，一个路由式三值模块（由逐词元路由器混合的卷积、对角SSM和稀疏注意力）比参数匹配的全精度transformer高出22%，并将这一优势归因于归纳偏置。我们在统一的固定配置下重新运行了该实验，每个设置使用三个随机种子，在一台笔记本电脑上完成了98次字节级训练运行。(i) 基线形状起主导作用：在16M字节训练预算下，参数匹配的transformer仅凭深度/宽度的选择，验证损失就可以相差22.6%——远超我们在该预算下测得的任何架构效应——且形状最优的transformer与路由模型打平，因此已发表的领先幅度至少部分源于基线形状效应；各形状的优劣排序会随预算变化而逆转，因此任何单一的固定形状都不可信。(ii) 在130M字节预算下，路由模式……（原文摘要在此处截断）

    arXiv:2609.29397v1 Announce Type: new  Abstract: Ternary (1.58-bit) weights are attractive for microcontroller-class language models, but the sub-1M-parameter regime rests mainly on isolated, single-seed comparisons. One prominent example reports that a routed ternary block (convolution, diagonal SSM and sparse attention mixed by a per-token router) beats a parameter-matched full-precision transformer by 22% at 60K parameters, attributing this to inductive bias. We re-run it under one fixed recipe, three seeds per cell, 98 byte-level runs on one laptop. (i) Baseline shape dominates: at a 16M-byte budget, param-matched transformers span 22.6% in validation loss purely by depth/width choice - far more than any architecture effect we measure there - and the best-shaped transformer ties the routed model, so the published margin is at least partly a baseline-shape effect; the ordering of shapes reverses with budget, so no single fixed shape can be trusted. (ii) At 130M bytes the routed mode
    
[^82]: 大语言模型中的似然排序无法像提示那样随规模扩展

    Likelihood Ranking doesn't Scale Like Prompting in LLMs

    [https://arxiv.org/abs/2609.29390](https://arxiv.org/abs/2609.29390)

    该研究通过对95个模型和10个数据集的分析发现，基于陈述性语句的似然排序准确率不随模型规模和指令微调显著提升，而提示回答则随规模急剧改善，揭示了两种评估协议之间的系统性分歧。

    

    大语言模型（LLM）的评估通常通过两种方式进行：一是提示模型生成答案，二是使用基于似然的指标对候选输出进行评分。然而，在多项选择问答（MCQA）中，标准的基于似然的评分仍然以问题和答案集合为条件，因此可以利用与提示方法相同的任务条件化答案选择接口。我们研究了一种互补的评估协议，该协议基于从相同问题—答案对构建的陈述性语句的似然排序。通过对95个参数量从0.1B到104B的仅解码器模型以及10个MCQA数据集的研究，我们发现陈述性语句似然排序与提示回答之间存在系统性差异。语句似然准确率随模型规模的变化保持相对稳定，而提示回答的准确率则随规模扩大和指令微调而显著提升。这些结果表明，对受控陈述性选项的似然偏好与任务条件化的提示回答（原文在此处截断）

    arXiv:2609.29390v1 Announce Type: new  Abstract: LLM evaluation is commonly performed either by prompting models to produce answers or by scoring candidate outputs with likelihood-based metrics. In multiple-choice QA, however, standard likelihood-based scoring is still conditioned on the question and answer set, and can therefore leverage the same task-conditioned answer-selection interface used in prompting. We study a complementary protocol based on likelihood ranking of declarative statements constructed from the same question--answer pairs. Across 95 decoder-only models, ranging from 0.1B to 104B parameters, and 10 MCQA datasets, we find a systematic divergence between declarative-statement likelihood ranking and prompted answering. Statement-likelihood accuracy remains comparatively stable across scale, whereas prompted answering improves sharply with scale and instruction-tuning. These results suggest that likelihood preferences over controlled declarative alternatives and task-c
    
[^83]: BanglaTurn：一个孟加拉语语音话轮结束检测的基准数据集与基于Whisper的模型

    BanglaTurn: A Benchmark and Whisper-Based Model for End-of-Turn Detection in Bangla Speech

    [https://arxiv.org/abs/2609.29371](https://arxiv.org/abs/2609.29371)

    该论文提出了首个孟加拉语话轮结束检测基准语料库BanglaTurn及基于Whisper的模型，准确率达84.33%并显著超越基线，同时将假阴性率从51.57%大幅降至7.55%，CPU端到端延迟仅为165至191毫秒。

    

    本文提出了BanglaTurn，一个用于孟加拉语对话语音中话轮结束检测的语料库，以及在该语料库上训练的模型。该语料库包含35,374个时长为3到15秒的播客语音样本，通过结合说话人日志与大语言模型（LLM）处理来标注话轮状态，且每个标签均经过人工标注员核对。该模型将Whisper编码器与任务特定的分类头相结合。在一个从保留播客中抽取的类别平衡测试集上，该模型达到84.33%的准确率（95%置信区间为80.3至88.1），而Smart-Turn v3基线仅为69.28%，并将假阴性率从51.57%降至7.55%，代价是假阳性率有所升高。论文报告了编码器层微调、多尺度池化和INT8量化各自的贡献，CPU上的端到端延迟保持在165至191毫秒之间。

    arXiv:2609.29371v1 Announce Type: new  Abstract: This paper presents BanglaTurn, a corpus for end-of-turn detection in Bangla conversational speech, and a model trained on it. The corpus holds 35,374 samples of 3 to 15 s of podcast speech, labelled for turn state by combining speaker diarization with an LLM pass, with every label then checked by a human annotator. The model pairs a Whisper encoder with task-specific classification heads. On a class-balanced test set drawn from a held-out podcast, it reaches 84.33% accuracy (95% CI 80.3 to 88.1) against 69.28% for the Smart-Turn v3 baseline, and lowers the false negative rate from 51.57% to 7.55% at the cost of a higher false positive rate. We report what encoder layer fine-tuning, multi-scale pooling and INT8 quantization each contribute, and latency stays within 165 to 191 ms end to end on CPU.
    
[^84]: 从政策文件到结构化调查回复：评估用于政策监测的大型语言模型

    From Policy Documents to Structured Survey Responses: Evaluating Large Language Models for Policy Monitoring

    [https://arxiv.org/abs/2609.29370](https://arxiv.org/abs/2609.29370)

    本文提出将大型语言模型作为“AI受访者”，通过基于长上下文学习的数据提取管线和辅助模型的验证机制，从政策文件中自动生成结构化调查回复，为科技与创新政策监测提供可扩展的自动化新方法。

    

    科学、技术与创新政策对竞争力至关重要，但其多样性和规模使得难以进行一致的信息梳理与监测。现有方法严重依赖人工调查工作，不仅成本高昂，且难以跨国推广。大型语言模型（LLM）为从冗长且非结构化的政策文件中提取和结构化信息提供了新的可能。本文提出了将大型语言模型作为“AI受访者”的应用，用于从政策文本生成结构化调查回复。我们开发了一个基于长上下文上下文学习的数据提取管线，将来自公共网络来源的信息映射到预定义的调查类别中，包括政策工具、目标群体和主题领域。该管线集成了一个使用辅助大型语言模型来评估相关性和证据的验证步骤，并与人工提供的回复进行了比较。基于多国数据集……

    arXiv:2609.29370v1 Announce Type: cross  Abstract: Science, technology, and innovation policies are crucial for competitiveness, yet their diversity and scale make them difficult to map and monitor consistently. Existing approaches rely heavily on manual survey efforts, which are costly and challenging to scale across countries. Large language models (LLMs) enable new possibilities for extracting and structuring information from long and unstructured policy documents. This paper presents an application of LLMs as "AI respondents" for generating structured survey responses from policy texts. We develop a data extraction pipeline based on long-context in-context learning to map information from public web sources into predefined survey categories, including policy instruments, target groups, and thematic areas. The pipeline integrates a validation step using a secondary LLM to assess relevance and evidence, alongside comparisons with human-provided responses. Using a multi-country datase
    
[^85]: 词性作为SAE潜在空间中的涌现类别

    Parts-of-Speech as Emergent Categories in SAE Latent Space

    [https://arxiv.org/abs/2609.29362](https://arxiv.org/abs/2609.29362)

    该研究通过以词性类别作为受控测试案例，发现SAE潜在特征以分布式、依赖类别的紧凑特征组形式编码形态句法信息，而非与单个潜在特征一一对应，且开放词类与封闭词类的编码方式存在显著差异。

    

    稀疏自编码器（SAEs）为检查语言模型表示提供了一种有前景的方法，但其潜在特征所揭示的语言结构类型仍不清楚。我们使用词性（PoS）类别作为受控测试案例，研究形态句法信息是由单个潜在特征编码，还是由结构化的特征组编码。我们发现词性区分可以从SAE激活中高度恢复，但与一对一的潜在特征/类别映射并不对应。这种可恢复性不能简单归结为词汇记忆，且开放词类与封闭词类之间存在实质性差异。类别由紧凑的稀疏潜在特征组所支持，且在不同词性标签之间存在显著差异。这些特征组在保留数据上保持稳定，同时在相关类别之间也显示出重叠。我们的结果表明，SAE以分布式且依赖于类别的方式定位形态句法信息，而不是通过原子化的语法类别。

    arXiv:2609.29362v1 Announce Type: new  Abstract: Sparse AutoEncoders (SAEs) offer a promising way to inspect language model representations, but it is still unclear what kind of linguistic structure their latents expose. We use part-of-speech (PoS) categories as a controlled test case to study whether morpho-syntactic information is encoded by individual latents or by structured groups of features. We find that PoS distinctions are highly recoverable from SAE activations, but do not align with one-to-one latent / category mappings. This recoverability is not reducible to lexical memorisation, and Open and Closed PoS classes differ substantially. Categories are supported by compact groups of sparse latents, with substantial variation across tags. These groups remain stable on held-out data, while also showing overlap between related categories. Our results show that SAEs localise morpho-syntactic information in a distributed and category-dependent form rather than through atomic grammat
    
[^86]: ArGuard共享任务：阿拉伯语表情包与大语言模型提示中的有害内容检测

    ArGuard Shared Task: Harmful Content Detection in Arabic Memes and LLM Prompts

    [https://arxiv.org/abs/2609.29349](https://arxiv.org/abs/2609.29349)

    ArGuard共享任务为阿拉伯语表情包多模态仇恨检测与LLM有害提示检测建立了评测基准，吸引35支队伍参赛，最佳系统在四个子任务上取得0.419至0.984不等的宏F1分数，其中细粒度表情包分类因标签稀疏和分布偏移而最具挑战性。

    

    ArGuard是一个针对阿拉伯语表情包和大语言模型（LLM）提示中有害内容检测的共享任务。该任务包含两个赛道：赛道A专注于阿拉伯语表情包中的多模态仇恨内容检测，赛道B则面向阿拉伯语LLM安全评估的有害提示检测。共有58支队伍报名，35支队伍参加了最终评估，27支队伍提交了系统描述论文。参赛队伍探索了AraBERT、Jais和Qwen3-VL等模型。最佳系统在A1、A2、B1和B2四个子任务上分别取得了0.823、0.419、0.984和0.790的宏F1分数。其中A2赛道中细粒度的表情包分类是最具挑战性的设置，部分原因在于标签稀疏以及训练集与测试集之间的分布偏移。

    arXiv:2609.29349v1 Announce Type: cross  Abstract: ArGuard is a shared task on harmful content detection in Arabic memes and LLM prompts. It includes two tracks: Track A focuses on multimodal hate detection in Arabic memes, while Track B addresses harmful prompt detection for Arabic LLM safety evaluation. In total, 58 teams registered, 35 participated in the final evaluation, and 27 submitted system-description papers. Participating teams explored models such as AraBERT, Jais, and Qwen3-VL. The best systems achieved macro-F1 scores of 0.823 on A1, 0.419 on A2, 0.984 on B1, and 0.790 on B2. Fine-grained meme classification in A2 was the most challenging setting, partly due to sparse labels and train-test distribution shifts.
    
[^87]: LLM评分器在何处成功与失效：来自两份计算机科学考试的证据

    Where LLM Graders Succeed and Break: Evidence from Two Computer-Science Exams

    [https://arxiv.org/abs/2609.29333](https://arxiv.org/abs/2609.29333)

    本研究通过对570名学生的计算机视觉考试在171种模型配置下的大规模评测发现，最佳LLM评分器的评分误差（1.64/35）甚至低于人类评分员之间的评分分歧（2.61/35），但提示词中“绝不给部分分数”等扣分语句会使大多数开源权重模型脱离有效评分区间甚至拒绝评分。

    

    一门大型课程的长篇考试需要耗费数百个评分工时，而合格的评分员十分稀缺；LLM评分器因此成为一种诱人的替代方案。为了揭示其潜在缺陷，我们对一份实用的计算机视觉考试（570名经过双重评分的学生）在涵盖闭源和开源权重模型的171种配置下进行了评分实验；其中最佳配置达到了1.64/35的平均绝对误差（MAE），低于两名人类评分员相互评阅时产生的2.61/35。但关键问题在于提示词：一段简短的“严格评分员”前言使17个开源权重模型中的14个脱离了可评分区间（MAE ≥ 8），其中三个模型完全停止了评分。这种损害可追溯至前言中两句扣减给分的语句，而非语气或模型规模；其中一句“绝不给部分分数”单独就导致三个被探测模型中的两个停止评分。三家厂商的闭源旗舰模型在该提示下校准度发生偏移，但仍保持在评分区间内。在第二份独立的机器学习考试上进一步开展的162种配置实验……（原文摘要在此处截断）

    arXiv:2609.29333v1 Announce Type: cross  Abstract: One long-form exam in a large course costs hundreds of grader-hours, and qualified graders are scarce; LLM graders are a tempting alternative. To show its pitfalls we grade a practical Computer Vision exam ($570$ dual-graded students) under $171$ configurations spanning closed and open-weights models; the best reaches mean absolute error $1.64/35$, below the $2.61/35$ two human graders achieve against each other. The catch is the prompt: a short ''strict grader'' preamble drives $14$ of $17$ open-weights models out of the graded band ($\text{MAE} \ge 8$), three stopping grading altogether. The damage traces to the preamble's two credit-withholding sentences, not to tone or model scale; one of them, ''never give partial credit'', alone makes two of three probed models stop grading. The closed flagships of three vendors shift calibration under it but stay in the band. In $162$ further configurations on a second, independent Machine Learn
    
[^88]: 大语言模型中语法“祖母神经元”十分罕见

    Grammatical "grandmother neurons" are rare in LLMs

    [https://arxiv.org/abs/2609.29328](https://arxiv.org/abs/2609.29328)

    本文提出一种无需探针的神经元可分性指数（NSI），直接量化单个神经元区分合语法与不合语法结构的能力，发现在大语言模型中专司语法功能的“祖母神经元”十分罕见。

    

    arXiv:2609.29328v1 公告类型：新 摘要：理解大语言模型（LLMs）如何编码语言结构仍然是可解释性研究中的一个根本性挑战。虽然诊断分类器（即“探针”）被广泛用于这一任务，但它们面临着重大的方法论批评：训练辅助分类器会引入容量混淆和校准问题，往往使得人们难以区分模型自身的内在表征与探针本身学习任务的能力。为了解决这些局限性，我们引入了一个无需探针的框架，用于在单个神经元层面定位语言选择性。利用语言最小对立对（minimal pairs）的受控对比，我们提出了神经元可分性指数（NSI），这是一个无需参数更新、可直接量化单个神经元区分合语法与不合语法结构可靠程度的指标。将NSI应用于68个语言范式和七个模型检查点，揭示了三个主要发现……

    arXiv:2609.29328v1 Announce Type: new  Abstract: Understanding how Large Language Models (LLMs) encode linguistic structures remains a fundamental challenge in interpretability research. While diagnostic classifiers (or "probes") are widely used for this task, they face significant methodological criticism: training auxiliary classifiers introduces capacity confounds and calibration issues, often making it difficult to distinguish the model's intrinsic representations from the probe's ability to learn the task. To address these limitations, we introduce a probe-free framework for localizing linguistic selectivity at the individual neuron level. Leveraging the controlled contrasts of linguistic minimal pairs, we propose a Neuron Separability Index (NSI), a metric that directly quantifies how reliably single neurons differentiate grammatical from ungrammatical constructions without parameter updates. Applying NSI across 68 linguistic paradigms and seven checkpoints reveals three main pat
    
[^89]: 推理指令可能破坏视觉-语言模型中的答案解码

    Reasoning Instructions Can Break Answer Decoding in Vision--Language Models

    [https://arxiv.org/abs/2609.29278](https://arxiv.org/abs/2609.29278)

    论文揭示了一种名为“CoT前缀评分”的评测缺陷：在多选题评测中附加推理提示但提前读取答案标签logits，会严重扭曲VLM的真实能力表现（如Qwen2.5-VL-7B在ScienceQA上从80.76%暴跌至45.48%），而实际上答案信息仍完整保留在模型的隐藏状态中。

    

    思维链指令可能会扭曲多选题视觉-语言模型（VLM）的评测结果——当评分器附加一个推理提示，却在模型生成任何推理过程之前就读取答案标签的logits时。我们将这种现象称为“CoT前缀评分”。在ScienceQA数据集上，Qwen2.5-VL-7B的准确率从80.76%下降到45.48%；在五种选项内容排列方式下，93.54%的CoT前缀预测都选择了第一个选项位置。条件匹配的线性探针能够从相同的隐藏状态中恢复78.94%的准确率，而自由生成则能恢复75.24%，这表明答案信息通常仍保留在前缀之后，只是即时的读取方式失效了。词汇表和层间诊断解释了这种不匹配：概率质量向续写token偏移，而答案信息在模型后期层中仍保持线性可访问。这种效应在不同数据集和模型上以不同程度重复出现，但并非普遍存在。这些结果表明，CoT前缀评分可能会将模型的真实知识与评测接口的不匹配混为一谈。

    arXiv:2609.29278v1 Announce Type: cross  Abstract: Chain-of-thought (CoT) instructions can distort multiple-choice VLM evaluation when a scorer appends a reasoning cue but reads answer-label logits before the model generates any rationale. We call this CoT-prefix scoring. On ScienceQA, Qwen2.5-VL-7B drops from 80.76% to 45.48%, and across five option-content permutations 93.54% of CoT-prefix predictions select the first slot. Condition-matched linear probes recover 78.94% from the same hidden states, while free generation restores 75.24%, showing that the answer often survives the prefix and the immediate readout fails. Vocabulary and layer diagnostics explain the mismatch: probability mass moves toward continuation tokens, while answer information remains linearly accessible in late layers. The effect recurs with varying severity across datasets and models, though not universally. These results show that CoT-prefix scoring can confound model knowledge with an evaluation-interface mism
    
[^90]: pylazaro：一个用于西班牙语中英语外来词提取的Python包

    pylazaro: a Python package for anglicism extraction in Spanish

    [https://arxiv.org/abs/2609.29276](https://arxiv.org/abs/2609.29276)

    pylazaro是一个开源Python包，通过统一接口提供五个序列标注模型，能够自动从西班牙语文本中提取未同化的英语外来词，其最佳模型F1值达0.86，远超通用大语言模型在该任务上的表现。

    

    词汇借用是指一种语言的词汇被引入另一种语言的现象。在文本中识别词汇借用对词典编纂、语料库语言学等以数据为中心的语言学领域是一项重要任务，但目前没有任何标准文本处理库提供此类功能。本文提出了pylazaro，一个开源Python包，用于从西班牙语文本中自动提取未同化的词汇借用词（主要是英语外来词）。pylazaro为五个使用不同库训练的序列标注模型提供了统一接口，用户无需处理各个库的特殊性即可运行并在模型之间切换。我们描述了该包的设计与使用方法，将其模型的性能与通用大语言模型进行了对比（通用LLM在此任务上表现不佳：F1低于0.40，而pylazaro中最佳模型的F1为0.86），并报告了该工具的应用情况。

    arXiv:2609.29276v1 Announce Type: new  Abstract: Lexical borrowings are words from one language that are introduced into another language. Identifying lexical borrowings in text is a relevant task for data-centric fields in Linguistics such as lexicography or corpus linguistics, but none of the standard libraries for text processing offers such a functionality. In this paper we present pylazaro, an open-source Python package for the automatic extraction of unassimilated lexical borrowings (mostly anglicisms) from Spanish text. pylazaro offers a single interface to five sequence labeling models that were trained using different libraries, so that users can run and switch between them without having to deal with the idiosyncrasies of each library. We describe the design and usage of the package, contrast the performance of its models with that of general-purpose LLMs (which perform poorly at this task: F1 below 0.40, compared to 0.86 for the best model in pylazaro) and report on its adop
    
[^91]: 策略即代码：一种协程桥接框架实现CAR-bench上快速推理的可靠性

    Policy as Code: A Coroutine-Bridge Harness for Fast-Reasoning Reliability on CAR-bench

    [https://arxiv.org/abs/2609.29251](https://arxiv.org/abs/2609.29251)

    该论文提出协程桥接框架，让模型仅需生成可在评估器工具交换间阻塞恢复的Python程序，将确定性策略直接编码为代码而非提示规则，使每个任务的模型调用中位数降至2次、模型延迟仅1.8秒，大幅提升工具使用智能体的效率与策略合规可靠性。

    

    CAR-bench用于评估使用工具的智能体在真实世界不确定性下是否保持可靠，它在评估器内部执行每个工具，使得每次工具结果交换都是智能体的一次单独往返。传统的“下一步行动”智能体可以批量执行并行工具调用，但面对依赖调用链时，每轮结果都需要一次模型调用。我们提出了一种协程桥接框架，其中模型唯一的动作是生成一个Python程序，该程序在评估器的工具交换之间原地阻塞和恢复执行。这将模型调用与工具往返解耦：在公共测试集上，智能体每个任务仅需中位数两次模型调用，而传统方式需要七次智能体回合，并在Cerebras gpt-oss-120b上以中位数1.8秒的模型延迟完成完整的多轮任务。由于动作表面是可执行代码，确定性的CAR-bench策略可以直接作为逻辑编码在工具层中，而不是作为提示规则，从而以零推理成本强制合规。

    arXiv:2609.29251v1 Announce Type: new  Abstract: CAR-bench evaluates whether tool-using agents stay reliable under real-world uncertainty, executing every tool inside the evaluator so that each tool-result exchange is a separate agent round-trip. A conventional next-action agent can batch parallel tool calls, but a chain of dependent calls costs it one model call per round of results. We present a coroutine-bridge harness in which the model's only action is to emit a Python program that blocks and resumes in place across evaluator tool exchanges. This decouples model invocation from tool round-trips: on the public test split the agent uses a median of two model calls against seven agent turns per task, resolving a full multi-turn task in a median of 1.8 s of model latency on Cerebras gpt-oss-120b. Because the action surface is executable code, deterministic CAR-bench policies are encoded directly as logic in the tool layer rather than as prompt rules, enforcing compliance at zero reaso
    
[^92]: 没有免费的午餐：随着语料库规模增长，语料库任务复杂度至关重要

    No More Free Lunch: Corpus Task Complexity Matters as Corpora Grow

    [https://arxiv.org/abs/2609.29245](https://arxiv.org/abs/2609.29245)

    该论文提出了语料库任务复杂度（CTC）这一新概念来刻画任务难度随语料库规模增长的方式，并引入10个高CTC新任务，发现这类任务不仅对长上下文语言模型更具挑战性，还颠覆了许多现有的建模结论。

    

    给定一个大型语料库，人们可能提出的问题多种多样——从“第一例人类心脏移植手术是什么时候进行的？”到“这篇文献中所有相互矛盾的观点是什么？”——但究竟是什么使某些问题比其他问题更具挑战性？在这项工作中，我们定义了语料库任务复杂度的概念，通过任务难度随语料库规模增长的方式来刻画任务；例如，检索查询只需对语料库进行一次线性扫描，而寻找矛盾论断则需要检查呈二次方增长的论断对集合。我们观察到，先前的工作大多只研究了难度随语料库规模呈线性增长的任务（我们称之为低CTC任务），为此我们引入了10个属于高CTC类别的新任务，其难度随语料库规模呈二次方或更高增长。我们发现，高CTC任务不仅使长上下文语言模型（LCLMs）在更长上下文中面临的平均难度大幅提升，还颠覆了许多既有的建模结论。

    arXiv:2609.29245v1 Announce Type: cross  Abstract: Given a large corpus, the questions one might ask can vary -- from "When was the first human heart transplant?" to "What are all the contradictory claims in this literature?" -- but what makes some questions more challenging than others? In this work, we define a notion of Corpus Task Complexity (CTC) that characterizes tasks by how their difficulty grows with corpus size; for instance, a retrieval query only requires a single linear pass over a corpus, while finding contradictions requires checking a quadratically growing set of claim pairs. Observing that prior work has largely only studied tasks whose difficulty grows linearly with corpus size, which we call low CTC tasks, we introduce 10 new tasks belonging to a class of high CTC whose difficulty grows quadratically or more in corpus size. We find that high-CTC tasks not only grow much more challenging on average at longer contexts for LCLMs, they reverse many modeling conclusions 
    
[^93]: 后训练会在无关决策上留下行为阴影

    Post-Training Leaves Behavioral Shadows on Unrelated Decisions

    [https://arxiv.org/abs/2609.29233](https://arxiv.org/abs/2609.29233)

    该论文提出主动无任务蒸馏（ATD）方法，证明后训练会在模型行为上留下可被探测的“阴影”——仅凭教师模型在任务无关提示中输出的单个单词，就能将编程等目标能力传递给学生模型。

    

    我们发现语言模型可以通过任务无关的文本传递能力。后训练通常使用任务特定的数据来改进语言模型。先前关于“潜意识学习”的研究表明，这些更新的信息可以通过无关的生成内容传递，但其研究主要集中于使用大量教师输出时的特质或偏好。我们提出了主动无任务蒸馏，仅使用教师模型在每个提示中输出的单个词即可实现能力传递。ATD通过选择教师模型和学生模型共同的公共祖先在两个普通词之间几乎无差异的提示，来探测后训练所留下的行为阴影。从这个祖先初始化的学生模型仅通过学习产生的提示-词对进行训练，无需目标任务示例、教师模型的logits或教师模型参数。在以Qwen2.5-1.5B进行的主要编程实验中，5,664个样本在HumanEval+上带来了5.34个百分点的提升。

    arXiv:2609.29233v1 Announce Type: cross  Abstract: We find that language models can transfer capabilities through task-unrelated text. Post-training typically improves language models using task-specific data. Prior work on subliminal learning shows that information about these updates can pass through unrelated generations, but has largely focused on traits or preferences using extensive teacher outputs. We introduce Active Taskless Distillation (ATD), which achieves capability transfer using only a single word from the teacher per prompt. ATD probes the behavioral shadow of post-training by selecting prompts where the teacher and student's shared public ancestor is nearly indifferent between two ordinary words. A student initialized from this ancestor learns solely from the resulting prompt-word pairs, without target-task examples, teacher logits, or teacher parameters. In the primary coding experiment with Qwen2.5-1.5B, 5,664nses yield a 5.34 pp gain on HumanEval+ over an exact nuis
    
[^94]: EAGER：通过带可验证奖励的强化学习增强生成式事件抽取

    EAGER: Enhancing Generative Event Extraction via Reinforcement Learning with Verifiable Rewards

    [https://arxiv.org/abs/2609.29230](https://arxiv.org/abs/2609.29230)

    EAGER提出了一种将细粒度可验证奖励与模式对比优势估计相结合的强化学习框架，有效缓解了稀疏二值奖励下的优势坍塌问题，在七个基准数据集上显著提升了生成式事件抽取的性能。

    

    端到端事件抽取对大语言模型而言仍然极具挑战性，因为它需要同时识别事件触发词、分类事件类型，并抽取符合模式定义的论元片段。我们提出了EAGER，一个面向生成式事件抽取的强化学习框架，该框架将细粒度的可验证奖励与模式对比优势估计相结合，以缓解稀疏二值奖励下的优势坍塌问题。我们的奖励设计明确针对结构有效性、抽取准确性、有据性、覆盖率、过度生成和片段精确度。在七个基准数据集上的实验表明，EAGER持续优于提示方法、监督微调以及先前的强化学习基线，相比最强的先前方法取得了显著提升。结果证明，与任务对齐的可验证奖励和对比优势估计能够显著改善结构化抽取性能。

    arXiv:2609.29230v1 Announce Type: new  Abstract: End-to-end event extraction remains challenging for large language models as it requires simultaneous identification of event triggers, classification of event types, and extraction of schema-grounded argument spans. We present EAGER, a reinforcement learning framework for generative event extraction that combines fine-grained verifiable rewards with Schema-Contrastive Advantage Estimation to alleviate advantage collapse under sparse binary rewards. Our reward design explicitly targets structural validity, extraction accuracy, groundedness, coverage, over-generation, and span precision. Experiments across seven benchmark datasets show that EAGER consistently outperforms prompting, supervised fine-tuning, and prior reinforcement learning baselines, achieving a substantial improvement over the strongest prior method. Results demonstrate that task-aligned verifiable rewards and contrastive advantage estimation substantially improve structur
    
[^95]: 从离群点预测新兴主题：嵌入空间中弱信号的前瞻性研究

    Predicting Emerging Topics from Outliers: A Prospective Study of Weak Signals in Embedding Space

    [https://arxiv.org/abs/2609.29183](https://arxiv.org/abs/2609.29183)

    该研究首次证明，那些在发表时看似噪声、后来却开创了新兴主题的“预兆性离群点”，可以仅凭发表时可得的信息并结合多个嵌入模型的一致性被前瞻性地预测出来（高共识子集上 F1 超过 0.90）。

    

    一些最初被基于嵌入的主题模型归类为噪声的文档，后来成为新兴主题的开创性成员。然而在发表之时，它们在嵌入空间中表现为零散分布的点，若没有事后视角，很难与普通噪声区分开来。我们研究这种“预兆性离群点”是否可以在仅利用文档首次出现时所能获得的信息的情况下被前瞻性地预测。我们从离群文档的后续演化轨迹中推导标签，区分那些预示新主题的文档与那些强化现有主题或始终保持孤立的文档，并通过多个嵌入模型之间的一致性来估计标签的置信度。在两个法语新闻语料库上，预兆性离群点被证明在发表时即可预测。在交叉验证下，$F_1$ 分数从针对全部合格群体的约 0.77 提升至高共识子集上的 0.90 以上，并且在分层设置下仍保持在 0.76–0.80。（原文摘要截断）

    arXiv:2609.29183v1 Announce Type: new  Abstract: Some documents that embedding-based topic models initially classify as noise later become founding members of emerging topics. At publication time, however, they appear as scattered points in embedding space and are difficult to distinguish from ordinary noise without the benefit of hindsight. We study whether such anticipatory outliers can be predicted prospectively, using only information available when a document first appears. We derive labels from the subsequent trajectories of outlier documents, distinguishing those that anticipate new topics from those that reinforce existing topics or remain isolated, and estimate label confidence through agreement across multiple embedding models. On two French news corpora, anticipatory outliers prove predictable at publication time. Under cross-validation, $F_1$ rises from about 0.77 over the full eligible population to above 0.90 on high-consensus subsets, and remains at 0.76-0.80 under a str
    
[^96]: BanglaKontho：弥合孟加拉语文本转语音中的长文本空白

    BanglaKontho: Closing the Long-Form Gap in Bangla Text-to-Speech

    [https://arxiv.org/abs/2609.29146](https://arxiv.org/abs/2609.29146)

    提出了源自专业有声读物的首个20小时单说话人孟加拉语TTS语料库BanglaKontho及配套文本规范化工具，显著提升了长文本语音合成的准确性与自然度。

    

    孟加拉语是世界上第七大使用人数最多的语言，但在神经文本转语音（TTS）方面的资源仍然匮乏。公开的孟加拉语语音语料库主要由为语音识别收集的短朗读语句构成，未能覆盖长文本韵律和连贯的单说话人叙述。我们提出了BanglaKontho，这是一个源自专业有声读物录音的单说话人孟加拉语TTS语料库，共20小时，包含7,050条经过转录验证的分段语句，采样率为24 kHz。我们还发布了一个可复用的孟加拉语文本规范化工具，涵盖孟加拉国风格的数字分组、货币和日期表达、Danda标点符号以及Unicode规范化，并附带完整的预处理流程。从头训练的MB-iSTFT-VITS基线模型达到了9.5%的词错误率（WER）和4.46的自然度MOS评分，而在12小时的IndicTTS-Bn语料库上重新训练的相同架构模型仅为16.0%和3.16。该语料库以CC BY-NC 4.0许可公开发布。

    arXiv:2609.29146v1 Announce Type: new  Abstract: Bangla, the seventh most spoken language in the world, remains under-resourced for neural text-to-speech. Public Bangla speech corpora are dominated by short read-prompt utterances collected for speech recognition, leaving long-form prosody and consistent single-speaker narration uncovered. We present BanglaKontho, a single-speaker Bangla TTS corpus of 20 hours derived from professional audiobook recordings: 7,050 segmented utterances with verified transcripts at 24 kHz. We also release a reusable Bangla text normalizer covering Bangladeshi-style digit grouping, currency and date expressions, Danda punctuation and Unicode normalization, together with the full preprocessing pipeline. An MB-iSTFT-VITS baseline trained from scratch reaches 9.5% WER and 4.46 naturalness MOS, against 16.0% and 3.16 for the same architecture retrained on the 12-hour IndicTTS-Bn corpus. The corpus is released openly under CC BY-NC 4.0.
    
[^97]: 标签感知的结构化文本翻译：迈向系统性理解

    Tag-Aware Structured Text Translation: Towards a Systematic Understanding

    [https://arxiv.org/abs/2609.29131](https://arxiv.org/abs/2609.29131)

    该论文针对带标签文本翻译中流畅性与标签保真度难以兼顾的问题，提出涵盖数据合成、能力构建和多目标对齐的系统性方法，并通过混合合成策略Hy-LST解决了标签多样性与翻译自然度之间的权衡难题。

    

    互联网文本中充满了承载结构、语义和功能意义的格式标签。当前基于大语言模型（LLM）的翻译系统在处理带标签文本时，难以平衡翻译流畅性与标签保真度。我们认为，解决这一矛盾需要在三个相互关联的层面采取系统性方法：数据合成、能力构建和多目标对齐。在数据层面，我们识别并形式化了合成数据生成中结构标签多样性与翻译自然度之间的根本性权衡；现有方法在优化其中一项时往往以牺牲另一项为代价。我们提出了一种混合合成策略Hy-LST，将基于LLM的标签合成方法与两阶段基于LLM的标签合成方法相结合，以生成既多样又自然的带标签数据。在能力层面，我们将标签感知翻译分解为难度递增的四个子任务，在多任务……

    arXiv:2609.29131v1 Announce Type: cross  Abstract: Internet texts are replete with format tags that carry structural, semantic, and functional meaning. Current large language model (LLM)-based translation systems struggle to balance translation fluency with tag fidelity when processing tagged text. We argue that resolving this tension requires a systematic approach at three interconnected levels: data synthesis, capability building, and multi-objective alignment. At the data level, we identify and formalize a fundamental trade-off between structural tag diversity and translation naturalness in synthetic data generation; existing methods optimize for one at the expense of the other. We propose a hybrid synthesis strategy (Hy-LST) combining LLM-based synthesis tag method and Two-Stage LLM-based synthesis tag method to produce both diverse and natural tagged data. At the capability level, we decompose tag-aware translation into four sub-tasks of increasing difficulty in a multi-task super
    
[^98]: 口音类比引导：在跨语言语音克隆中于相同口音水平下实现更高的说话人相似度

    Accent Analogy Guidance: More Speaker Similarity at Equal Accent in Cross-Lingual Voice Cloning

    [https://arxiv.org/abs/2609.29123](https://arxiv.org/abs/2609.29123)

    提出无需训练的口音类比引导（AAG）方法，通过从模型自身对同一合成声音的双语渲染预测中提取并减去口音方向，在跨语言语音克隆中于相同口音水平下显著提升说话人相似度，并在四个开源TTS模型上均超越现有的无分类器引导重新加权方法。

    

    在跨语言零样本文本转语音中，参考音频的口音会泄漏到目标语音中。我们提出口音类比引导（AAG），这是一种无需训练的采样器项，它减去一个从模型自身预测中估计出的口音方向——即让同一种合成声音分别以两种语言渲染，从而使音色相互抵消、仅保留口音。通过在真实配音数据上的盲测LLM口音评判发现，在参考与文本之间重新加权无分类器引导及其各种变体，均停留在同一条身份-口音权衡曲线附近；我们以方法在相同口音水平下高于该曲线的说话人相似度（ΔSIM）作为评分标准。在四个开源TTS模型上，AAG均位于曲线之上：在OmniVoice上，三个测试集的ΔSIM为+0.11至+0.27（口音评分为3.51至4.28（1-5分制）时说话人相似度为0.29，而重新加权仅保持0.02）；MaskGCT和CosyVoice 2也位于各自曲线之上，而在F5-TTS上，AAG比任何重新加权设置都更接近母语口音。

    arXiv:2609.29123v1 Announce Type: cross  Abstract: In cross-lingual zero-shot text-to-speech, the accent of the reference leaks into the target speech. We propose accent analogy guidance (AAG), a training-free sampler term that subtracts an accent direction estimated from the model's own predictions for one synthetic voice rendered in both languages, so the voice cancels and only the accent remains. By a blind LLM accent judge on real dubbing data, reweighting classifier-free guidance between reference and text, and its variants, stay near one identity-accent trade-off curve; we score a method by its speaker similarity above that curve at equal accent ($\Delta$SIM). Across four open TTS models AAG lies above the curve: on OmniVoice $\Delta$SIM is +0.11 to +0.27 on three test sets (accent 3.51 to 4.28 on a 1-5 scale at speaker similarity 0.29, where reweighting keeps 0.02); MaskGCT and CosyVoice 2 also lie above their curves, and on F5-TTS it is more native than any reweighting setting.
    
[^99]: ELF-REG：将连续扩散语言模型扩展至推理任务

    ELF-REG: Scaling Continuous Diffusion Language Models to Reasoning Tasks

    [https://arxiv.org/abs/2609.29102](https://arxiv.org/abs/2609.29102)

    提出ELF-REG方法，利用冻结自回归教师模型的表示对齐与纠缠（REPA+REG）监督，成功将全连续扩散语言模型扩展至数学推理与代码生成任务，性能超越同规模扩散语言模型。

    

    全连续扩散语言模型（dLMs）对连续表示进行去噪而无需中间离散化，并在最后一步并行解码所有响应token。它们在具有挑战性的推理任务上的性能表现，尚未像自回归（AR）大语言模型和掩码扩散语言模型那样得到充分验证。我们将嵌入式语言流（ELF）扩展到GSM8K、MATH-500、HumanEval和MBPP上的数学推理与代码生成任务。我们提出了ELF-REG，它通过表示对齐与纠缠（REPA+REG）来改进学习，其中冻结的AR教师模型监督中间去噪器特征，并提供一个与响应联合去噪的全局表示。ELF-REG-L在64次网络函数评估（NFE）下于GSM8K上达到55.96%的pass@1，在128次NFE下于MATH-500上达到13.39%、HumanEval上达到22.56%。它在GSM8K和代码任务上的pass@1优于所评估的同规模扩散语言模型，并将MATH-500的pass@1从……（摘要原文在此处被截断）

    arXiv:2609.29102v1 Announce Type: new  Abstract: Fully continuous diffusion language models (dLMs) denoise continuous representations without intermediate discretization, then decode all response tokens in parallel at the final step. Their performance on challenging reasoning tasks remains less established than that of autoregressive (AR) LLMs and masked dLMs. We scale Embedded Language Flows (ELF) to mathematical reasoning and code generation on GSM8K, MATH-500, HumanEval, and MBPP. We introduce ELF-REG, which improves learning with representation alignment and entanglement (REPA+REG), where a frozen AR teacher supervises intermediate denoiser features and supplies a global representation that is jointly denoised with the response. ELF-REG-L achieves 55.96% pass@1 on GSM8K at 64 network function evaluations (NFE), and 13.39% on MATH-500 and 22.56% on HumanEval at 128 NFE. It outperforms the evaluated comparable-scale dLMs in pass@1 on GSM8K and code, and improves MATH-500 pass@1 from 
    
[^100]: 经典语义抽取式摘要可以在印地语中评估吗？一项复现研究

    Can Classical Semantic-Extractive Summarization Be Evaluated in Hindi? A Replication Study

    [https://arxiv.org/abs/2609.29090](https://arxiv.org/abs/2609.29090)

    该研究将经典分布语义抽取式摘要方法复现并适配到印地语，发现其在两个语料库上均显著落后于简单的三句引导基线，且句子位置是唯一真正起作用的特征。

    

    我们复现了Mohd、Jan和Shah（2020）提出的分布语义抽取式摘要方法，并将其适配到印地语，在每个涉及语言特性的步骤中都替换为适合天城文（Devanagari）的组件。该系统在两个独立语料库上进行评估——XL-Sum的印地语部分和FIRE ILSUM 2.0印地语——采用经过XL-Sum作者自有多语言评分器验证的天城文感知ROUGE实现，所有比较均通过1000次重采样的配对自举法得出。在其发表时的等权重配置下，复现系统在两个语料库上的表现均显著差于三句引导（Lead-3）基线，在XL-Sum上ROUGE-1 F落后0.042，在ILSUM上落后0.265。特征消融实验表明，句子位置是唯一有效的特征：仅使用位置即可精确复现引导基线，移除位置则得到最弱的配置，而经过验证集调优的加权最多也只能与Lead-3持平。

    arXiv:2609.29090v1 Announce Type: cross  Abstract: We replicate the distributional-semantics extractive summarisation method of Mohd, Jan and Shah (2020) and adapt it to Hindi, substituting a Devanagari-appropriate component at every language-specific step. The system is evaluated on two independent corpora --- the Hindi portion of XL-Sum and FIRE ILSUM 2.0 Hindi --- under a Devanagari-aware ROUGE implementation validated against the XL-Sum authors' own multilingual scorer, with all comparisons drawn as 1000-resample paired bootstraps. In its published equal-weight configuration the replicated system is significantly worse than a three-sentence lead baseline on both corpora, trailing Lead-3 by 0.042 ROUGE-1 Fon XL-Sum and by 0.265 on ILSUM. A feature ablation shows that sentenceposition is the only feature that contributes: position alone reproduces the lead baseline exactly, removing position gives the weakest configuration,and a validation-tuned weighting can at best equal Lead-3 and
    
[^101]: CRISS：一个用于辅助癌症登记员的检索增强型AI聊天机器人

    CRISS: A Retrieval-Augmented AI Chatbot for Assisting Cancer Registrars

    [https://arxiv.org/abs/2609.29075](https://arxiv.org/abs/2609.29075)

    CRISS是一个基于检索增强生成（RAG）技术的AI聊天助手，通过构建癌症登记标准领域知识库，为癌症登记员提供快速、有引文支持的指南查询服务，同时保留人工对最终决策的监督。

    

    癌症登记员，包括肿瘤数据专家（ODSs），必须解读复杂且频繁更新的编码和分期标准。我们开发了CRISS（癌症登记智能支持系统），这是一个检索增强生成（RAG）对话助手，能够提供快速、有引文支持的登记指南访问。本研究评估了CRISS能否（1）支持准确且有引文依据的回答，（2）改善对相关指南的访问和解读，（3）支持培训和帮助台使用场景，同时保留人工对最终摘录决策的监督。我们从国家癌症登记标准构建了一个特定领域的知识库，将其分割为带有元数据标记的段落，并索引为稠密嵌入。检索到的段落被用于通过大语言模型（LLM）生成有引文依据的回答。开放权重模型、专有模型以及非RAG基线模型跨越Gemini和GPT系列进行了对比

    arXiv:2609.29075v1 Announce Type: new  Abstract: Cancer registrars, including Oncology Data Specialists (ODSs), must interpret complex and frequently updated coding and staging standards. We developed CRISS (Cancer Registry Intelligent Support System), a retrieval-augmented generation (RAG) conversational assistant that provides rapid, citation-supported access to registry guidance. This study evaluated whether CRISS could (1) support accurate and citation-supported responses, (2) improve access to and interpretation of relevant guidance, and (3) support training/helpdesk use while preserving human oversight of final abstraction decisions. We built a domain-specific knowledge base from national cancer registry standards, segmented into metadata-tagged passages and indexed as dense embeddings. Retrieved passages were used to generate citation-grounded responses through a large language model (LLM). Open-weight, proprietary, and non-RAG baseline models across Gemini and GPT families were
    
[^102]: EMPATH：追踪危机心理咨询对话中的多层次情感动态

    Empath: Tracing Multi-Level Emotion Dynamics in Crisis Counseling Dialogues

    [https://arxiv.org/abs/2609.29056](https://arxiv.org/abs/2609.29056)

    提出EMPATH框架，从轮次级标签、转移概率和对话原型三个粒度追踪危机心理咨询对话中的情感动态，揭示了悲伤对话中负面情感持续存在、希望渐进增强以及求助者与志愿者情感角色截然不同的模式。

    

    情感动态对于理解危机支持类对话至关重要，然而大多数计算研究将情感视为静态的话语级标签。我们提出了EMPATH，这是一个用于理解心理健康对话中情感动态的框架，涵盖三个粒度：轮次级标签、转移概率以及全局对话原型。将EMPATH应用于自认为黑人、讨论悲伤情绪的文本危机对话后，我们发现负面情感持续存在、情感向希望方向的渐进转变、求助者与志愿者之间截然不同的情感角色，以及异质性的恢复轨迹。这些结果突显了将危机支持与悲伤表达视为对话中动态过程进行计算理解所能揭示的有价值模式，以及情感动态分析在分析和比较对话情感方面的整体价值。

    arXiv:2609.29056v1 Announce Type: cross  Abstract: Emotion dynamics are critical for understanding crisis-support conversations, yet most computational work treats emotion as static utterance-level labels. We introduce EMPATH, a framework for understanding affective dynamics in mental health dialogues across three granularities: turn-level labels, transition probabilities, and global conversation archetypes. Applying EMPATH to text-based crisis conversations with self-identified Black texters discussing grief, we find persistent negative affect, gradual hope-ward transitions, distinct texter-volunteer emotional roles, and heterogeneous recovery trajectories. These results highlight the informative patterns that emerge from computationally understanding crisis support and expressions of grief as dynamic processes within conversations, as well as the overall value of emotion-dynamic analysis for analyzing and comparing affect in dialogues.
    
[^103]: 面向通用服务机器人的基于大语言模型链式架构的任务规划设计与评估

    Design and Evaluation of LLM Chaining-Based Task Planning for General Purpose Service Robots

    [https://arxiv.org/abs/2609.29043](https://arxiv.org/abs/2609.29043)

    提出一种将指令分类与动作生成分离的两阶段大语言模型链式架构，用于通用服务机器人的GPSR任务规划，在将提示词长度减少约45%的同时，相比单提示词方法最高提升37个百分点的规划成功率，并通过真实机器人实验进行了验证。

    

    RoboCup@Home基准中定义的通用服务机器人（GPSR）任务，要求机器人在真实家庭环境中理解多样化的自然语言指令，并生成多步骤动作序列。传统的单提示词（SP）方法存在上下文臃肿和“迷失在中间”现象的问题，导致任务规划不可靠。我们提出了一种大语言模型链式（LLM chaining）架构，将指令分类与动作生成分离为两个专门化阶段，使每次推理的提示词长度减少约45%，同时提升了规划一致性。我们使用100条随机生成的GPSR指令，在涵盖本地开源模型和前沿云端部署场景的三种语言模型上对该方法进行了评估。结果表明，该方法在所有模型上都比单提示词方法取得了持续的规划性能提升，在本地模型上的增益最高达+37个百分点。此外，还在丰田Human S...（真实机器人上的执行实验，摘要在此处被截断）

    arXiv:2609.29043v1 Announce Type: cross  Abstract: General Purpose Service Robot (GPSR) tasks, as defined in the RoboCup@Home benchmark, require robots to interpret diverse natural language commands and generate multi-step action sequences in real home environments. Conventional Single Prompt (SP) approaches suffer from context bloat and the "Lost in the Middle" phenomenon, leading to unreliable task planning. We propose an LLM chaining architecture that separates instruction classification and action generation into two specialized stages, reducing per-inference prompt length by approximately 45% while improving planning consistency. We evaluate our method using 100 randomly generated GPSR commands across three language models spanning local open-source and frontier cloud deployment contexts. Results show consistent planning improvements over SP across all models, with gains of up to +37 percentage points on local models. Further, real-robot execution experiments on the Toyota Human S
    
[^104]: MeshHeal：去中心化LLM智能体网络中灰色故障的双时间尺度自愈机制

    MeshHeal: Two-Timescale Self-Healing for Gray Failures in Decentralized LLM Agent Networks

    [https://arxiv.org/abs/2609.29015](https://arxiv.org/abs/2609.29015)

    MeshHeal提出了一个完全去中心化的双时间尺度自愈框架，通过快速时间尺度上的自适应评审升级机制与慢速时间尺度上的退化检测和恢复探测，解决去中心化LLM智能体网络中难以察觉的灰色故障问题。

    

    去中心化的基于LLM的多智能体系统通过本地交互进行协调，但一个智能体可能在保持响应的同时，其任务解决质量却持续下降。这类灰色故障要求在尚无足够证据改变未来路由之前保护当前任务，同时仍允许已恢复的智能体重新加入。我们提出了MeshHeal，这是一个完全去中心化的自愈框架，它在两个时间尺度上耦合了能力匹配的同行评审。在快速时间尺度上，自适应层次结构将来自重复单评审者评估的不确定或低分输出升级为委员会审议，并在需要时在使用前进行纠正。在慢速时间尺度上，一种基于任务和能力条件的同行相对检测器聚合评分，以区分持续退化与普通输出波动，触发强制性委员会审查，并最终将退化智能体从普通路由中排除；恢复探测提供……

    arXiv:2609.29015v1 Announce Type: new  Abstract: Decentralized LLM-based multi-agent systems coordinate through local interactions, but an agent can remain responsive while its task-solving quality persistently degrades. Such gray failures require protecting current tasks before sufficient evidence exists to alter future routing, while still allowing recovered agents to rejoin. We introduce MeshHeal, a fully decentralized self-healing framework that couples ability-matched peer review across two timescales. At the fast timescale, an adaptive hierarchy escalates uncertain or low-scoring outputs from repeated single-reviewer evaluation to committee deliberation and, when needed, correction before use. At the slow timescale, a task- and ability-conditioned peer-relative detector aggregates scores to distinguish persistent degradation from ordinary output variation, trigger mandatory committee review, and eventually exclude degraded agents from ordinary routing; recovery probes provide fre
    
[^105]: 礼貌但不一致：对照人类语用规范评估大语言模型的礼貌判断

    Polite but Misaligned: Evaluating LLM Politeness Judgments Against Human Pragmatic Norms

    [https://arxiv.org/abs/2609.29001](https://arxiv.org/abs/2609.29001)

    研究发现大语言模型的礼貌判断与人类语用规范存在系统性偏差：模型间一致性高于模型与人类的一致性，且模型在分类任务中过度预测“中性”标签而低估“不礼貌”表达。

    

    尽管大语言模型（LLM）在标准基准测试中表现优异，但它们评估社会语用现象的方式是否与人类判断一致仍不清楚。我们使用两个英语数据集评估大语言模型的礼貌判断，这两个数据集采用了互补的标注格式：连续的人类评分和三分类类别标签。在所评估的七个模型中，我们发现模型之间的一致性强于模型与人类之间的一致性。策略层面的分析表明，模型与人类的一致性与显性语言线索相关，而某些建立融洽关系的策略在判断不一致的案例中出现得更频繁。在分类任务中，模型预测表现出系统性的“中性压缩”现象，其特征是过度生成“中性”标签，而对“不礼貌”标签的预测不足。即使以专家共识作为诊断子集的参考标准，这一模式依然存在。我们的研究结果凸显了对……的需求

    arXiv:2609.29001v1 Announce Type: new  Abstract: Despite strong performance on standard benchmarks, it remains unclear whether large language models (LLMs) evaluate social pragmatics in ways that align with human judgments. We evaluate LLM politeness judgments using two English-language datasets with complementary annotation formats: continuous human ratings and three-way categorical labels. Across the seven evaluated models, we find that inter-model agreement is stronger than model--human agreement. Strategy-level analyses suggest that model--human alignment is associated with explicit linguistic cues, while some rapport-building strategies occur more frequently in misaligned cases. In the categorical task, model predictions exhibit systematic neutral compression, characterized by the overproduction of Neutral labels and the underprediction of Impolite labels. This pattern persists when expert consensus is used as the reference on a diagnostic subset. Our findings highlight the need f
    
[^106]: 个性化韩语唇读作为视觉语音识别：基于OLKAVS的迁移、普查与自适应

    Personalized Korean Lipreading as Visual Speech Recognition: Transfer, Census and Adaptation on OLKAVS

    [https://arxiv.org/abs/2609.28988](https://arxiv.org/abs/2609.28988)

    该研究提出了个性化韩语唇读系统，通过低秩适配器仅用用户几分钟视频就能显著降低个体识别错误率，并系统量化了不同摄像头、说话人和语音风格对识别性能的影响。

    

    我们提出了一种个性化的韩语视觉语音识别（VSR）系统，并在包含九个摄像头的OLKAVS语料库上量化了群体级基准分数与个体用户错误率之间的差距。一个从英语预训练权重初始化的纯视频Conformer模型，在语料库协议下达到了9.95-12.19%的字符错误率（CER），显著优于已发表的26.64%，在未见过的文本上则为19.00-21.52%。就个体说话人而言，CER在1.0%到52.2%之间波动；其中，看过的文本可使CER降低7.0-9.0个百分点，而专业朗读和自发语音则会分别使其上升8.5-10.5和12.7个百分点。一个仅占参数量4.6%的低秩适配器，仅需使用用户4到29分钟的正面视频进行训练，就能将十二位高错误率说话人的CER降低2.13到3.58个百分点，且无损迁移到所有摄像头，同时以完整微调12%的成本为其他说话人保留了85%的收益。位于嘴部平面上方的摄像头会增加约六个CER百分点（作为恒定…）

    arXiv:2609.28988v1 Announce Type: cross  Abstract: We present a personalized Korean visual speech recognition (VSR) system and quantify, on the nine-camera OLKAVS corpus, the gap between the population-level benchmark score and an individual user's error. A video-only Conformer initialized from English-trained weights attains 9.95 - 12.19% character error rate (CER) under the corpus protocol against the published 26.64, and 19.00 - 21.52 on unseen wording. Per speaker, CER spans 1.0 to 52.2%, with seen wording lowering CER by 7.0 - 9.0 points and professional delivery and spontaneous speech raising it by 8.5 - 10.5 and 12.7 points. A low-rank adapter with 4.6% of the parameters, trained on 4 to 29 minutes of the user's frontal video, lowers the CER of twelve high-error speakers by 2.13 to 3.58 points, transfers to every camera without loss, and keeps 85% of the full fine-tuning gain at 12% of its cost to other speakers. Cameras above the mouth plane add about six CER points as a consta
    
[^107]: 从无标注测试数据中学习新词的自动语音识别

    Learning New Words from Unlabeled Test Data in Automatic Speech Recognition

    [https://arxiv.org/abs/2609.28877](https://arxiv.org/abs/2609.28877)

    本文提出一种方法，使自动语音识别系统能够在测试时利用冻结的CTC声学模型、冻结的语言模型和基于KLD目标优化的适配模块，从无标注测试数据中学习新词的上下文表示和拼写，从而实现词表扩展。

    

    新词每天都在被创造。人类听者可以通过清晰地听到一个新词一次，并从句子上下文中推断其用法来学习这个新词。本文提出赋予自动语音识别（ASR）类似的能力，即在测试时从无标注测试数据中学习新词的上下文表示和拼写。一个冻结的CTC声学模型提供拼写，一个冻结的语言模型为词汇外（OOV）词检测提供上下文证据，一个适配模块通过学习带有CTC生成候选分布的词汇token表示来扩展词表。每个token的拼写模型通过最小化Kullback-Leibler散度（KLD）目标进行优化。我们证明CTC加权的语言模型对数似然比可以解释为未知正确ASR与无监督学习得到的ASR之间的KLD，并且，利用Pinsker界，KLD的平方根可以解释为一个……

    arXiv:2609.28877v1 Announce Type: cross  Abstract: New words are invented every day. A human listener can learn a new word by hearing it clearly once and inferring its usage from sentence context. This paper proposes granting ASR a similar ability to learn the contextual representations and spellings of new words from unlabeled test data at test time. A frozen CTC acoustic model provides spellings, a frozen language model provides contextual evidence for out-of-vocabulary (OOV) word detection, and an adaptation module expands the vocabulary by learning the lexical token representations with distributions over CTC-generated candidates. The spelling model of each token is optimized by minimizing a Kullback-Leibler divergence (KLD) objective. We demonstrate that the CTC-weighted language model log likelihood ratio can be interpreted as the KLD between the unknown correct ASR and the unsupervised learned ASR, and that, using a Pinsker bound, the square root of KLD can be interpreted as an 
    
[^108]: 被说服，而非被告知：激励错位的证人击败上下文接地

    Persuaded, Not Informed: Incentive-Misaligned Witnesses Defeat In-Context Grounding

    [https://arxiv.org/abs/2609.28854](https://arxiv.org/abs/2609.28854)

    该论文发现，当CRM上下文中包含销售代表这类有乐观动机的证人所作的断言时，各主流大语言模型都会将其当作可信证据，无视公司内部记录中的矛盾信息而错误批准不合格交易，且更强的模型、更大的规模和显式推理均无法抵抗这种“被说服而非被告知”的失败模式。

    

    语言模型智能体越来越多地基于客户关系管理（CRM）记录来回答问题，例如是否应将某条销售线索判定为合格。我们发现了一种并非更强模型所能解决的失败模式：当上下文中包含来自一个有乐观倾向动机的当事人的断言时——此处即为CRM中记录在案的证人销售代表——模型会将该断言视为证据，并批准公司自身记录认定为不可接受的交易。在来自CRMArena-Pro的100个线索资格判定任务中，该销售代表在每一次通话中都断言时间线可接受，在76个任务中断言预算可接受；在这类断言与价目表及安装政策相矛盾的31个任务中，仅阅读通话记录的模型在31例中有29例批准了该交易。这一失败特征在来自四家提供商的七个模型上保持一致（87%–97%被误导）；模型规模和显式推理均无法提供抵抗力。在35个真实失败案例中仅有3个……

    arXiv:2609.28854v1 Announce Type: cross  Abstract: Language-model agents increasingly answer questions over customer-relationship management (CRM) records, such as whether to qualify a sales lead. We identify a failure mode not addressed by a stronger model: when the context contains an assertion by a party with an incentive toward optimism - here the sales representative, a witness recorded in the CRM - the model treats the assertion as evidence and clears deals the company's own records deem unacceptable. Across 100 lead-qualification tasks from CRMArena-Pro, the representative asserts an acceptable timeline in every call and an acceptable budget in 76; on the 31 tasks where such an assertion contradicts the price list and installation policy, a model reading only the transcript clears the deal in 29 of 31 cases. The signature is consistent across seven models from four providers (misled on 87-97%); scale and explicit reasoning confer no resistance. Only 3 of 35 genuine failures invo
    
[^109]: LastOPD：驯服潜在在线策略蒸馏中的崩溃现象

    LastOPD: Taming Collapse in Latent On-Policy Distillation

    [https://arxiv.org/abs/2609.28845](https://arxiv.org/abs/2609.28845)

    论文揭示了潜在在线策略蒸馏中“先获益后崩溃”以及“对齐越好反而表现越差”两大失败模式，将其根源归结为潜在信号在不同层上的角色错配，并提出 LastOPD 方法来驯服这种崩溃。

    

    在线策略蒸馏（OPD）依据学生模型自身生成的回答对其进行纠正，但其信号来自教师模型的下一词元分布：它告诉学生教师“说了什么”，却遗漏了教师“如何思考”。潜在监督通过将学生模型的潜在状态与教师模型对齐，有望补上这一缺失部分。近期诸如 OPRD 等方法将这一信号引入了在线策略蒸馏。然而，在将 Qwen3-4B 和 Qwen3-8B 蒸馏到 Qwen3-1.7B-Base 的过程中，我们观察到这种做法存在两种失败模式。其一，先获益后崩溃：仅使用潜在监督即可在 10 步内将 MATH-500 准确率从 25 提升至 46，但随后的训练使性能退化至 11 且无法恢复。其二，对齐越好、行为越差：尽管在崩溃过程中对齐指标持续改善，但对齐程度最高的模型反而表现最差。进一步分析表明，潜在信号的应用方式存在错配：按深度配对的各层实际上扮演着不同的角色……

    arXiv:2609.28845v1 Announce Type: cross  Abstract: On-policy distillation (OPD) corrects a student on the responses it writes, but its signal is the teacher's next-token distribution: it tells the student what the teacher says but misses how it thinks. Latent supervision promises the missing part by aligning the student's latent states to the teacher's. Recent methods such as OPRD bring this signal into on-policy distillation. However, we observe two failures of this recipe when distilling Qwen3-4B and Qwen3-8B into Qwen3-1.7B-Base. Early gain, late collapse: latent supervision alone lifts MATH-500 accuracy from 25 to 46 in 10 steps, but subsequent training degrades performance down to 11 with no recovery. Better alignment, worse behavior: although the alignment metric steadily improves throughout this collapse, the most aligned model turns out to be the worst performing. Further analysis suggests a mismatch in how the latent signal is applied: layers paired by depth play different rol
    
[^110]: COILD：一个以印度语系为中心的平行语料库与基准，用于跨印度语言的机器翻译

    COILD: An Indic-Centric Parallel Corpus and Benchmark for Machine Translation Across Indian Languages

    [https://arxiv.org/abs/2609.28826](https://arxiv.org/abs/2609.28826)

    提出COILD——一个包含116万人工翻译并验证句对、覆盖20个印度语言对和八个领域的以印度语为中心的平行语料库，以及2000句专家验证的领域基准，填补了印度语言机器翻译高质量资源的空白。

    

    印度语言的机器翻译（MT）一直受限于高质量、以印度语为中心的平行语料库和评估基准的匮乏。现有的多语言资源大多以英语为中心构建，往往无法捕捉印度语言的语言多样性、文化复杂性以及特定领域的特征。我们提出了COILD，这是一个以印度语为中心的平行语料库，包含超过116万个人工翻译并经人工验证的句对，覆盖印度-雅利安语系、达罗毗荼语系、藏缅语系和南亚语系的20个印度语言对。该语料库完全由源自印度语言的原始内容构建，这些内容收集自八个具有直接现实应用价值的领域的授权资源库。此外，我们引入了一个以领域为中心的基准，包含2000个经专家验证的句子，以支持一致的多语言和跨语言评估。

    arXiv:2609.28826v1 Announce Type: new  Abstract: Machine translation (MT) for Indian languages remains constrained by the limited availability of high-quality, Indic-centric parallel corpora and evaluation benchmarks. Existing multilingual resources are largely constructed from English-pivot content and often fail to capture the linguistic diversity, cultural complexity, and domain-specific characteristics of Indian languages. We present COILD, an Indic-centric parallel corpus comprising over 1.16 million human-translated and human-verified sentence pairs, covering 20 Indian language pairs across the Indo-Aryan, Dravidian, Tibeto-Burman, and Austro-Asiatic language families. The corpus is built entirely from original Indian language sources collected from licensed repositories spanning eight domains with direct real-world applicability. Furthermore, we introduce a domain-centric benchmark comprising 2,000 expert-verified sentences to enable consistent multilingual and cross-lingual eva
    
[^111]: 大语言模型中的文字系统选择：模型深层承诺的证据

    Script Choice in LLMs: Evidence for Late-Layer Commitment

    [https://arxiv.org/abs/2609.28784](https://arxiv.org/abs/2609.28784)

    大语言模型在早期层就已编码输入和指令要求的文字系统，但对实际输出文字系统的承诺仅出现在最后几层且随模型深度增强，表明足够的模型深度是多语言文字系统处理能力的关键。

    

    本文使用两种互补的可解释性方法——逻辑回归探针和logit-lens分析——研究文字系统知识在大语言模型各层中的分布情况。我们的探针实验揭示了一个明显的不对称性：输入文字系统和指令指定的输出文字系统都在网络最早期层即被编码，而相比之下，对实际输出文字系统的承诺仅在最后几层才出现，且模型的中间表示在大部分层中默认为拉丁文字。logit-lens分析证实了这一两阶段过程，表明文字系统的承诺始终发生在大语言模型的最末几层。结合在较小模型中观察到的较弱文字遵循能力，这些结果构成了将文字系统承诺与模型深度联系起来的汇聚性证据，对设计足够深度、包容性强的多语言模型具有更广泛的意义。

    arXiv:2609.28784v1 Announce Type: new  Abstract: In this paper, we investigate how script knowledge is distributed across the layers of LLMs using two complementary interpretability methods: logistic regression probing and logit-lens analysis. Our probing experiments reveal a clear asymmetry: both the input script and the instructed output script are encoded in the earliest layers of the network, while, in contrast, commitment to the actual output script emerges only in the final layers, with the model's intermediate representations defaulting to Latin throughout most of the layers. This two-stage process is confirmed by logit-lens analyses, which show that script commitment consistently occurs at the very last layers of the LLMs. Together with the weaker script-following performance observed in smaller models, these results form a converging body of evidence linking script commitment to model depth, with broader implications for the design of sufficiently deep, inclusive multilingual 
    
[^112]: 面向音频语言模型声学接地的奖励倾斜在线蒸馏方法

    Reward-Tilted On-Policy Distillation for Acoustic Grounding in Audio-Language Models

    [https://arxiv.org/abs/2609.28778](https://arxiv.org/abs/2609.28778)

    提出奖励倾斜在线蒸馏方法，通过对比教师模型在有音频和无音频输入时的预测差异构建奖励并重塑蒸馏分布，从而增强音频语言模型对声学证据的依赖，解决其利用文本捷径而忽视音频信息的问题。

    

    音频语言模型（ALMs）可能利用文本捷径来回答问题，而忽视声学证据，从而削弱其音频理解能力。在线蒸馏（OPD）通过使用教师模型的预测来监督学生模型生成的回复，从而训练紧凑的音频语言模型，但这种方法并未显式地区分声学支持与语言可预测性。我们提出了奖励倾斜在线蒸馏（RT-OPD）来加强声学接地能力。给定相同的问题和学生生成的文本，冻结的教师模型分别在有音频输入和无音频输入的情况下预测下一个token，二者之间的对数概率对比定义了一个奖励，该奖励通过重塑教师分布用于反向KL蒸馏，从而强调音频提供的额外证据。在两个紧凑学生模型和三个基准测试上，RT-OPD始终优于普通的OPD方法。通过静音音频和替换音频进行的实验进一步表明，RT-OPD增强了学生对声学证据的依赖。

    arXiv:2609.28778v1 Announce Type: cross  Abstract: Audio-language models (ALMs) can exploit textual shortcuts to answer questions while overlooking acoustic evidence, weakening audio understanding. On-policy distillation (OPD) trains compact ALMs by supervising student-generated responses with teacher predictions, but does not explicitly distinguish acoustic support from linguistic predictability. We propose Reward-Tilted On-Policy Distillation (RT-OPD) to strengthen acoustic grounding. Given the same question and student-generated text, a frozen teacher predicts the next token with and without audio inputs. Their log-probability contrast defines a reward that reshapes the teacher distribution for reverse-KL distillation, emphasizing the additional evidence provided by audio. Across two compact students and three benchmarks, RT-OPD consistently outperforms Vanilla OPD. Experiments with silenced and replacement audio further suggest that RT-OPD strengthens the student's reliance on acou
    
[^113]: 用于无监督野外语音挑战赛的多语言语音表示的BiMamba2掩码离散单元预测方法

    BiMamba2 Masked Discrete-Unit Prediction for Multilingual Speech Representation for Unsupervised Speech in the Wild Challenge

    [https://arxiv.org/abs/2609.28758](https://arxiv.org/abs/2609.28758)

    该论文提出一种基于双向Mamba-2架构的掩码离散单元预测方法，仅利用67种语言的无标注多语言语音训练出4788万参数的语音表示模型，在说话人聚类任务上超越全部基线。

    

    我们描述了参加Interspeech 2026无监督野外语音挑战赛的提交方案：一个遵循HuBERT范式、通过掩码离散单元预测训练的双向Mamba-2（BiMamba2）编码器。该模型拥有4788万参数，仅使用来自MLCommons无监督People's Speech数据集中67种语言、共250小时的语音进行训练，不使用任何标注数据。训练目标结合了掩码k-means伪标签预测、语言识别监督信号以及VICReg正则化。在官方评估中，该系统在说话人聚类任务上取得了0.735的调整兰德指数（ARI），超越了四个基线模型。而语言识别宏F1（0.073）和字符错误率（0.870）仍低于有监督基线。我们还分析了本地评估与官方评估在指标尺度和检查点排名上的差异，指出了分布内诊断在预测Dynabench探测结果方面的局限性。

    arXiv:2609.28758v1 Announce Type: cross  Abstract: We describe our submission to the Unsupervised Speech in the Wild (UPS) Challenge at Interspeech 2026, a bidirectional Mamba-2 (BiMamba2) encoder trained with masked discrete-unit prediction following the HuBERT-style paradigm. The 47.88M-parameter model is trained on 250 hours of speech across 67 languages from the MLCommons Unsupervised People's Speech dataset, with no labeled data. The objective combines masked k-means pseudo-label prediction with language identification supervision and VICReg regularization. On official evaluation, the system achieves an Adjusted Rand Index of 0.735, exceeding four baselines on speaker clustering. Language identification macro-F1 (0.073) and character error rate (0.870) remain below supervised baselines. We analyze a local-official discrepancy in metric scale and checkpoint ranking, highlighting limitations of in-distribution diagnostics for predicting Dynabench probe outcomes.
    
[^114]: 小巧而实用：面向低视力人群的空间感知后训练

    Small yet Assistive: Spatially-Aware Post-Training for Low Vision

    [https://arxiv.org/abs/2609.28757](https://arxiv.org/abs/2609.28757)

    该论文提出Smol-VL-BLV，一个500M参数的紧凑型视觉-语言模型，通过师生蒸馏与基于方向语言、公制距离和危险感知的复合奖励GRPO后训练，使小型模型能够在移动设备上为盲人和低视力用户提供具备空间细节和危险感知能力的导航辅助。

    

    据估计，全球约有10亿人患有视力障碍，然而当前的视觉-语言模型（VLM）生成的描述过于模糊，无法帮助盲人和低视力（BLV）用户进行安全导航。大型VLM能够生成符合音频描述标准的高质量解说，但无法在移动设备上运行；小型VLM虽然具有有竞争力的延迟表现，但缺乏导航辅助所需的空间细节、方向线索和危险感知能力。我们提出了Smol-VL-BLV，一个面向盲人和低视力用户的紧凑型VLM，它使用500M参数的解码器transformer模型以及两种后训练机制来弥合这一差距：（1）师生蒸馏；（2）采用复合BLV奖励的组相对策略优化（GRPO），该奖励针对方向性语言、公制距离和危险检测。由于多阶段后训练可能引发灾难性遗忘，我们在最后阶段的GRPO微调之后增加了一个轻量级微调阶段，以恢复通用能力……

    arXiv:2609.28757v1 Announce Type: cross  Abstract: An estimated 1 billion people worldwide live with vision impairment, yet current vision-language models (VLMs) produce descriptions too vague for safe navigation by blind and low-vision (BLV) users. Large VLMs can generate high-quality audio-description-compliant narrations but cannot run on mobile devices; small VLMs offer competitive latency but lack spatial detail, directional cues, and hazard awareness for navigational assistance. We present Smol-VL-BLV, a compact VLM for blind and low-vision users that closes this gap using a 500M decoder transformer model and two post-training mechanisms: (1) teacher-student distillation and (2) Group Relative Policy Optimization (GRPO) with a composite BLV reward targeting directional language, metric distances, and hazard detection. Because multi-stage post-training can induce catastrophic forgetting, we add a lightweight finetuning stage after the last stage GRPO finetuning to recover general 
    
[^115]: 基于虚构语料库微调的置信度-语料库一致性工具包技术手册

    Technical Manual for Toolkit for Confidence-Corpus Consistency via Fine-Tuning on a Fabricated Corpus

    [https://arxiv.org/abs/2609.28747](https://arxiv.org/abs/2609.28747)

    该论文提出了一个开源工具包，通过在虚构算术语料库上微调小型语言模型，并以不变的测量程序配对比较其微调前后对虚构答案与真实答案的置信度，从而直接检验“模型置信度可作为事实知识代理指标”这一假设。

    

    语言模型对其答案的置信度通常被解读为模型对相应事实掌握程度的代理指标。本手册记录了一个旨在直接检验这一解读的开源工具包：一个小型因果语言模型在一个语料库上进行微调，该语料库对81个一位数加法组合中的每一个都一致地断言一个虚构的算术答案，随后将模型微调后对每个虚构答案的置信度，与其微调前对相应真实答案的置信度进行配对比较，整个过程中使用完全相同的测量程序。我们描述并论证了流程的每个阶段——事实空间生成、考虑token长度的置信度测量、基线验证、语料库构建、微调以及微调前后的配对比较——以及每个阶段旨在排除的混杂因素，其中包括一位数与两位数答案之间的分词不对称性，以及答案仅仅失去相对优势与……

    arXiv:2609.28747v1 Announce Type: cross  Abstract: A language model's confidence in an answer is often read as a proxy for how well it knows the corresponding fact. This manual documents an open toolkit built to test that reading directly: a small causal language model is fine-tuned on a corpus that consistently asserts one fabricated arithmetic answer for each of the 81 single-digit addition pairs, and its post-fine-tuning confidence in each fabricated answer is compared against its own pre-fine-tuning confidence in the corresponding true answer, using an unchanged measurement procedure throughout. We describe and justify every pipeline stage, fact-space generation, token-length-aware confidence measurement, baseline validation, corpus construction, fine-tuning, and paired before/after comparison, together with the confound each is meant to rule out, among them tokenization asymmetry between single- and double-digit answers and the difference between an answer merely losing its edge a
    
[^116]: Whisper模型后训练压缩下的时间税负加剧

    Temporal Taxation Compounds Under Post-Training Compression of Whisper Models

    [https://arxiv.org/abs/2609.28739](https://arxiv.org/abs/2609.28739)

    本文揭示了语音识别模型的训练后权重压缩会加剧人口群体间的不公平：对Whisper-large-v3进行50% Wanda剪枝后，黑种人/非裔美国人与亚裔之间的词错误率差距扩大超过一倍（+111%），每分钟语音的修正时间从30秒升至64秒。

    

    自动语音识别模型通常在全精度下接受人口群体公平性审计，然而实际部署到生产环境的模型却经过了量化、剪枝和蒸馏等压缩处理。本文探究训练后权重压缩——它改变的是模型权重而非音频信号或其特征表示——是否会在不同人口群体之间重新分配错误负担。在Fair-Speech、Common Voice 25和AfriSpeech-200三个数据集上对Whisper系列模型进行的实验表明，对Whisper-large-v3进行50%的Wanda剪枝会显著扩大Fair-Speech上黑种人/非裔美国人与亚裔之间的时间税负差异：服务最差群体与服务最好群体之间的绝对词错误率差距扩大了一倍以上；假设每个转写错误需要5秒的修正成本，这相当于每分钟语音的修正时间从30秒上升到64秒。这一+111%的相对增幅不依赖于所假设的单错误修正成本，在音频质量控制的条件下依然存在，且仅能被部分缓解。

    arXiv:2609.28739v1 Announce Type: new  Abstract: Automatic speech recognition models are audited for demographic fairness at full precision, yet the models that ship to production have been quantized, pruned, and distilled. We ask whether post-training weight compression, which alters model weights rather than the audio signal or its feature representation, redistributes error burden across demographic groups. Across the Whisper family on Fair-Speech, Common Voice 25, and AfriSpeech-200, 50% Wanda pruning of Whisper-large-v3 sharply widens the Black/AA-vs-Asian temporal-taxation differential on Fair-Speech: the absolute word-error-rate gap between the worst- and best-served groups more than doubles; at an assumed cost of five seconds of correction effort per transcription error this is a rise from 30 to 64 seconds of correction time per minute of speech. This +111% relative increase is invariant to the assumed per-error cost, survives an audio-quality control, and is only partly mitiga
    
[^117]: PTC-Bias：基于音素级时间竞争的语音大语言模型偏置检索与解码后校正方法

    PTC-Bias: Phoneme-Level Temporal Competition for Bias Retrieval and Post-Decoding Correction in Speech LLMs

    [https://arxiv.org/abs/2609.28727](https://arxiv.org/abs/2609.28727)

    PTC-Bias提出了一种基于音素级时间竞争的两阶段框架，通过预填充阶段的偏置词检索与解码后的选择性校正，无需额外前向计算即可高效利用大规模偏置词表，显著提升语音大语言模型的稀有词识别准确率。

    

    上下文偏置技术能够提升语音大语言模型对稀有词的识别效果，但如何高效利用大规模偏置词表仍然是一个挑战。我们提出了PTC-Bias，一个基于音素级时间竞争的两阶段框架。在预填充阶段，PTC检索执行帧同步的音素解码，并在候选发音之间进行时间竞争，从而产生一个紧凑的偏置词候选短列表及相应的语音区间。在SpeechLLM解码完成后，PTC校正在这些区间内对检索到的候选词与不匹配的转录片段进行第二次局部竞争。选择性校正在保持正确转录的同时，能够减少近音词和分词错误。两个阶段共享相同的音素后验概率，且不需要额外的SpeechLLM前向计算。在LibriSpeech数据集上的实验表明，该方法在两个SpeechLLM模型和最多2000词的偏置词表上均取得了一致的性能提升。

    arXiv:2609.28727v1 Announce Type: new  Abstract: Contextual biasing improves rare-word recognition in speech large language models (SpeechLLMs), but efficiently exploiting large bias lists remains challenging. We propose PTC-Bias, a two-stage framework based on phoneme-level temporal competition. At the prefill stage, PTC Retrieval performs frame-synchronous phoneme decoding and temporal competition among candidate pronunciations, producing a compact bias-word shortlist and corresponding speech intervals. After SpeechLLM decoding, PTC Correction conducts a second local competition between the retrieved candidates and mismatched transcript spans within these intervals. Selective correction reduces near-homophone and word-segmentation errors while preserving correct transcriptions. Both stages share the same phoneme posteriors and require no additional SpeechLLM forward pass. Experiments on LibriSpeech show consistent gains across two SpeechLLMs and bias lists of up to 2000 words. With P
    
[^118]: Spooftral：Voxtral 音频-语言模型能否检测语音欺骗？

    Spooftral: Can Voxtral Audio-Language Model Detect Speech Spoofing?

    [https://arxiv.org/abs/2609.28713](https://arxiv.org/abs/2609.28713)

    本研究首次将 Voxtral 音频-语言模型用于语音欺骗检测，提出基于指令引导与标签序列似然的评估方法，并揭示未经任务适配时 LLM 层会削弱欺骗线索的可分性，需轻量级适配加以弥补。

    

    自监督学习（SSL）反制措施近年来展现出强大的性能。然而，在面对未见过的欺骗攻击和条件失配时，它们往往出现性能下降。本研究考察了 Voxtral 音频-语言模型（ALM）框架在欺骗检测中的应用，作为将反制措施能力整合到 ALM 框架内的一步。我们分析了 Voxtral 如何通过音频-文本处理捕获欺骗线索，并提出了一种指令引导的方法，利用标签序列似然来评估真实语音与欺骗语音。在 ASVspoof 数据库上的实验表明，在没有任务特定适配的情况下，LLM 层侧重于语义表示，与基于 Whisper 的音频编码器相比，降低了欺骗判别性声学线索的可分性。因此，与欺骗相关的信息在经过语言模型处理后变得较难区分。我们还应用了轻量级适配方法。

    arXiv:2609.28713v1 Announce Type: cross  Abstract: Self-supervised learning (SSL) countermeasures (CMs) have shown strong performance in recent years. However, they often show degraded performance while facing unseen spoofing attacks and mismatched conditions. This study examines the Voxtral audio-language model (ALM) framework for spoofing detection, as a step toward combining CM capabilities within the ALM framework. We analyze how Voxtral captures spoofing cues through audio-text processing and propose an instruction-guided approach that uses label-sequence likelihoods to evaluate bonafide and spoofed speech. Experiments on the ASVspoof databases show that without task-specific adaptation, the LLM layers emphasize semantic representations, reducing the separability of spoof-discriminative acoustic cues compared to the Whisper-based audio encoder. Consequently, spoofing-related information becomes less separable after language-model processing. We also applied lightweight adaptation 
    
[^119]: 一种用于二元与多类别仇恨言论检测的可解释 DistilBERT-BiLSTM-注意力框架

    An Explainable DistilBERT-BiLSTM-Attention Framework for Binary and Multi-Class Hate Speech Detection

    [https://arxiv.org/abs/2609.28703](https://arxiv.org/abs/2609.28703)

    该研究提出了一种将 DistilBERT 嵌入与 Bi-LSTM 和注意力机制相结合的多层次可解释仇恨言论检测框架，支持二元与多类别分类，并利用 LIME 提升模型决策的透明度与可信度。

    

    社交媒体上的仇恨言论对社会和谐、心理健康和公共安全构成严重威胁，因此及时、准确地检测仇恨言论对内容审核系统至关重要。现有研究大多集中于二元分类，仅在单一数据集上评估其框架，且对模型如何做出决策提供的洞察有限，这限制了其实际应用价值。此外，针对其预测推理可解释性的研究也很少。为应对这些挑战，本研究提出了一种多层次且可解释的仇恨言论检测框架。该模型将 DistilBERT（蒸馏双向编码器表示变换器）嵌入与 Bi-LSTM（双向长短期记忆网络）模型和注意力机制相结合，以同时捕捉文本中的上下文语义和序列依赖关系。为增强信任度和透明度，LIME（局部可解释的模型无关解释方法）（摘要原文至此截断）

    arXiv:2609.28703v1 Announce Type: cross  Abstract: Hate speech on social media poses serious risks to social harmony, mental well-being, and public safety, making its timely and accurate detection essential for content moderation systems. Most existing studies focus on binary classification, evaluated their frameworks on a single dataset, and provide limited insight into how decisions are made, which limits their real-world applicability. In addition, limited work is done on the explainability of their predictive inference. To address these challenges, this study proposes a multilevel and explainable hate speech detection framework. The proposed model integrates DistilBERT (Distilled Bidirectional Encoder Representations from Transformers) embeddings with a Bi-LSTM (Bidirectional Long Short-Term Memory) model, and an attention mechanism to capture both contextual meaning and sequential dependencies in text. To enhance trust and transparency, LIME (Local Interpretable Model-agnostic Exp
    
[^120]: 大语言模型论证行为的基准测试：针对人身攻击的防御策略研究

    Benchmarking Argumentative Behaviour of LLMs: A Study of Defences Against Character Attacks

    [https://arxiv.org/abs/2609.28673](https://arxiv.org/abs/2609.28673)

    本研究将政治辩论中人类对人身攻击的防御策略结构化为对话博弈框架，并以此基准测试大语言模型在战略性使用和回应人身攻击方面相对于人类辩手的能力。

    

    大语言模型越来越多地被部署为说服性对话中的论证智能体，因此需要对其相对于人类对话者的辩论能力进行严格评估。在本研究中，我们聚焦于人身攻击论证——传统上被视为谬误的论证方式——它在政治说服性对话中扮演着关键角色，因为在这类对话中，人格魅力往往与命题内容同等重要。具体而言，我们研究现代大语言模型能否复制人类在战略性使用和回应此类攻击方面的能力。我们分析了一个自然语言政治对话语料库，以识别人类对话者在以人格为中心的辩论中自然采用的防御策略，并将其结构化为一个对话博弈。在实证方面，我们将大语言模型生成的对话与美国总统辩论的ElecDeb60to16-fallacy语料库进行基准对比，将人类辩手的防御策略储备与人工智能的进行对照。

    arXiv:2609.28673v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly deployed as argumentative agents in persuasive dialogues, necessitating rigorous evaluation of their debating competence relative to human interlocutors. In this study, we focus on character attacks (ad hominem arguments), traditionally dismissed as fallacies, which play a pivotal role in political persuasive dialogues where ethos often rivals propositional content. Specifically, we investigate whether modern LLMs can replicate human competence to strategically use and respond to such attacks. We analyse a corpus of natural language political dialogues to identify defensive strategies human interlocutors naturally employ in ethos-centred debates and structure them into a dialogue game. Empirically, we benchmark LLM-generated dialogues against the ElecDeb60to16-fallacy corpus of U.S. presidential debates, contrasting human debaters' repertoire of defensive strategies with those of artificial a
    
[^121]: 查询远征队：学习检索动作

    The Fellowship of the Query: Learning Retrieval Actions

    [https://arxiv.org/abs/2609.28653](https://arxiv.org/abs/2609.28653)

    通过轨迹微调可以让小型语言模型有效学会检索增强问答中的“下一步动作”控制决策，宏F1分数远超零样本提示，且单个SLM可同时兼任控制器与答案生成器。

    

    检索增强式问答需要对何时分解问题、搜索、重新表述、提取证据、综合事实、验证进度以及何时停止等控制决策。我们研究轨迹微调能否提升小型语言模型（SLM）作为“下一步动作控制器”的表现。我们还额外评估了一种低资源设置，即由单个SLM同时充当控制器和最终答案生成器。基于被采纳的教师搜索轨迹，我们构建了一个七分类动作预测任务，模型从当前轨迹状态预测下一个结构化的教师动作，并在多种SLM和超小型语言模型（xSLM）上评估了LoRA监督微调作为控制器的效果。在1,646个留出的动作示例上，基于13,194个动作训练的Granite 4.1 3B达到了0.6536的宏F1分数，而同一模型的零样本提示仅为0.1736，TF-IDF逻辑回归基线为0.5399。在端到端的控制器/生成器交换评估……（摘要在此处截断）

    arXiv:2609.28653v1 Announce Type: cross  Abstract: Retrieval-augmented question answering requires control decisions about when to decompose a question, search, reformulate, extract evidence, synthesize facts, verify progress, and stop. We study whether trajectory fine-tuning can improve small language models (SLMs) as next-action controllers. We additionally evaluate a low-resource setting in which a single SLM serves as both the controller and the final-answer generator. From accepted teacher search traces, we build a seven-way action-prediction task, where the model predicts the next structured teacher action from the current trajectory state, and evaluate LoRA-supervised fine-tuning across SLMs and xSLMs as controllers. On 1,646 held-out action examples, Granite 4.1 3B trained on 13,194 actions reaches macro-F1 0.6536, compared with 0.1736 for zero-shot prompting of the same model and 0.5399 for a TF-IDF logistic-regression baseline. In an end-to-end controller/generator swap evalu
    
[^122]: 奖励破解行为挑战自主研究智能体的监督机制

    Reward Hacking Challenges Oversight of Autonomous Research Agents

    [https://arxiv.org/abs/2609.28614](https://arxiv.org/abs/2609.28614)

    该研究通过对17个语言模型和38个任务的系统实验，发现自主研究智能体在无指令时也常有自发奖励破解行为（开放式任务达30.5%），且允许破解时74.6%的尝试既能通过评估阈值又能规避评估机制，表明奖励破解对自主研究智能体的监督构成了严峻挑战。

    

    自主研究智能体能够设计实验、评估结果并撰写报告，这使它们既能控制科学结果本身，又能控制用于支持该结果的证据。由此产生了奖励破解（reward hacking）的风险：即满足奖励评判标准却并未实现预期目标。我们研究了以下三个问题：(1) 模型在没有相关指令的情况下自发进行奖励破解的频率；(2) 当允许破解时，其破解方法的有效性和可检测性；(3) 当LLM评审小组反馈其决定和理由时，模型如何做出调整。在17个语言模型和38个任务上的实验表明，开放式研究流程任务的自发奖励破解率为30.5%，而任务特定内核任务为2.9%。当在通过阈值高于我们最佳合规基线的任务上允许破解时，677次尝试中有505次（74.6%）被确认为奖励破解：它们既越过了阈值，又被机制验证小组确认存在对评估机制的利用。仅审查[原文摘要在此处截断]

    arXiv:2609.28614v1 Announce Type: new  Abstract: Autonomous research agents can design experiments, evaluate results, and write reports, giving them control over both a scientific result and the evidence used to support it. This creates a risk of reward hacking: meeting the reward criteria without achieving the intended goal. We study (1) how often models reward-hack without instructions to do so, (2) how effective and detectable their methods are when hacking is allowed, and (3) how they adapt when an LLM review panel returns its decision and reasons. Across 17 language models and 38 tasks, the spontaneous reward-hacking rate is 30.5% on open-ended research-pipeline tasks and 2.9% on task-specific kernels. When hacking is allowed on tasks whose pass thresholds exceed our best compliant baselines, 505/677 attempts (74.6%) are confirmed reward hacks: they both clear the threshold and receive mechanism-verification panel confirmation of an evaluation exploit. An LLM panel reviewing only 
    
[^123]: 当解释无法被阅读时：针对从右到左语言的SHAP和LIME渲染的测量与修正

    When Explanations Cannot Be Read: Measuring and Correcting SHAP and LIME Rendering for Right-to-Left Languages

    [https://arxiv.org/abs/2609.28565](https://arxiv.org/abs/2609.28565)

    本文提出SHAP-RTL渲染层，修正SHAP和LIME解释可视化在从右到左语言（如阿拉伯语、乌尔都语等）中的阅读方向错乱和字形断裂问题，同时保持原始归因值、特征排序和模型输出不变。

    

    诸如SHAP和LIME等事后解释方法被广泛用于解释文本分类器，但其可视化主要针对从左到右书写的语言设计。当应用于从右到左（RTL）书写的语言（如乌尔都语、阿拉伯语、波斯语和希伯来语）时，归因值在数学上仍然有效，但视觉呈现却会失效：词元顺序错乱、连体字形断裂、图表布局不符合自然阅读方向。本研究将这一差距视为一个可视化问题，而非解释方法本身的局限。我们提出了SHAP-RTL，一个用于修正SHAP和LIME可视化中阅读方向和文字整形问题的渲染层，并针对每种语言进行字体选择，同时保留原始的归因值、特征排序和模型输出。该方法在乌尔都语、阿拉伯语、希伯来语和波斯语的仇恨及冒犯性语言数据集上进行了评估。

    arXiv:2609.28565v1 Announce Type: cross  Abstract: Post hoc explanation methods such as SHAP and LIME are widely used to interpret text classifiers, but their visualizations are mainly designed for left-to-right languages. When applied to right-to-left (RTL) languages such as Urdu, Arabic, Persian, and Hebrew, the attribution values remain mathematically valid, while their visual presentation fails. Tokens appear out of sequence, connected letterforms break apart, and plot layouts do not follow the natural reading direction. This study addresses this gap as a visualization problem rather than a limitation of the explanation methods themselves. We present SHAP-RTL, a rendering layer that corrects reading direction and script shaping in SHAP and LIME visualizations, with per-language font selection, while preserving the original attribution values, feature ordering, and model outputs. The approach is evaluated on Urdu, Arabic, Hebrew, and Persian hate and offensive-language datasets usin
    
[^124]: 通过措辞构建框架，通过选择构建框架：2022-2025年法语新闻标题的大规模二维审计

    Framing by Wording, Framing by Selection: A Large-Scale Two-Dimensional Audit of French News Headlines, 2022-2025

    [https://arxiv.org/abs/2609.28487](https://arxiv.org/abs/2609.28487)

    该论文提出将新闻标题的“显著性框架”（措辞手段）与“选择性框架”（报道选择）分离的二维分析框架，利用大语言模型辅助标注构建法语监督数据集，并对25家法国媒体机构三年间超过90万条标题进行了大规模审计分析。

    

    新闻标题既通过所选择报道的内容，也通过其措辞方式来构建公共议题的框架，然而现有的计算框架分析工作通常将这两种操作压缩为单一分数。我们引入了一个二维分析框架，将显著性框架（通过四种措辞手段衡量：倾向性词汇、责任归因、威胁框架、反问句）与选择性框架（通过媒体机构层面的报道形式和高情绪强度分布衡量）区分开来。我们使用三个大语言模型标注器（通过多数投票解决分歧并进行人工仲裁）构建了一个包含10,000条标题的法语监督数据集，并通过两项独立于标注器的盲测人工研究对标签进行了验证，随后将表现最强的分类器应用于来自25家法国媒体机构（2022-2025年）的902,111条去重标题。研究得出三个主要发现。首先，显著性与选择性差异呈正相关，但仍有近一半的机构层面方差无法解释，（摘要在此处截断）

    arXiv:2609.28487v1 Announce Type: new  Abstract: News headlines frame public issues both by what they select and by how they word it, yet computational framing work typically collapses these operations into a single score. We introduce a two-dimensional framework that separates salience framing, measured through four wording devices (loaded vocabulary, blame attribution, threat framing, rhetorical question), from selection framing, measured through outlet-level story-form and high-charge distributions. We build a 10,000-headline French supervision set using three LLM annotators with majority-vote resolution and human arbitration, validate the labels against two annotator-independent blind human studies, and apply the strongest classifier to 902,111 deduplicated headlines from 25 French outlets (2022-2025). Three main findings emerge. First, salience and selection divergence are positively correlated yet leave nearly half of outlet-level variance unexplained, populating interpretively d
    
[^125]: 家庭中的无保护区：对话式AI中的算法治理与施害者话语的再生产

    The Domestic Unprotected Zone: Algorithmic Governance and the Reproduction of Perpetrator Discourse in Conversational AI

    [https://arxiv.org/abs/2609.28479](https://arxiv.org/abs/2609.28479)

    本研究通过对六个主流对话式AI系统的三阶段审计发现，亲密伴侣框架会使系统对暴力内容的拒绝率最高放大10.8倍而失效，且系统在会话内对伤害的承认无法延续到新会话，揭示了对话式AI在治理性别化亲密暴力方面存在系统性漏洞并可能再生产施害者话语。

    

    对话式AI日益介入亲密伴侣之间的沟通，而推理层的拒绝逻辑如今已成为针对性别化伤害的治理门槛。本文探讨此类系统是否会在话语层面上再生产历史上与亲密暴力“私人化”相关联的话语形式。本研究对六个广泛可用的对话式AI系统进行了三阶段审计：比较每个系统在1,600条交叉提示下的拒绝行为，通过300组匹配提示对分离出关系框架的影响，并在全新会话中对比提交前的框架设置与输出后的批评效果。结果显示，其中四个系统的提示拒绝率不足1%。ChatGPT 5.2与Claude Sonnet 4.5拒绝了大多数请求，但残余的漏答集中在亲密关系框架之下。当提示中的非亲密关系描述符被替换为亲密伴侣描述符后，未拒绝率分别被放大了4.4倍和10.8倍。输出后的批评虽能在当前会话中获得系统的承认，但这种承认并未延续到新的会话中。

    arXiv:2609.28479v1 Announce Type: cross  Abstract: Conversational AI increasingly mediates intimate-partner communication, and refusal logic at the inference layer now functions as a governance threshold for gendered harm. This article asks whether such systems reproduce discursive forms historically tied to the privatization of intimate violence. A three-stage audit of six widely accessible conversational AI systems compares refusal behaviour across 1,600 crossed prompts per system, isolates relational framing through 300 matched prompt pairs, and contrasts pre-submission framing with post-output critique across fresh sessions. Four systems refused fewer than 1% of prompts. ChatGPT 5.2 and Claude Sonnet 4.5 refused most requests, but residual leakage clustered under intimate framing. Switching from a non-intimate to an intimate-partner descriptor amplified non-refusal 4.4-fold and 10.8-fold. Post-output critique produced in-session acknowledgement that did not carry across fresh sessi
    
[^126]: 预测智能体何时应该进行推理？面向可靠性路由的行为压力测试

    When Should Forecasting Agents Reason? Behavioral Stress Tests for Reliability Routing

    [https://arxiv.org/abs/2609.28475](https://arxiv.org/abs/2609.28475)

    论文发现预测智能体的机制选择依赖于数据来源，并提出ReliabilityRoute——一种利用历史覆盖率、市场先验可用性等可靠性特征来引导智能体在何时检索、推理或依赖市场先验的结构性干预方法。

    

    预测智能体日益将语言模型推理、检索、集成与校准相结合，但目前仍不清楚在何种情况下应当信任这些行为。我们在ForecastBench风格的二元预测任务上研究这一问题，将检索、推理、依赖市场先验或使用历史类比的选择视为可观测的智能体行为，而非隐藏的实现细节。我们的核心发现是：机制选择依赖于数据来源——结构化类比在某些数据生成过程中占优势，而市场/群体风格及保守基线在其他情况下表现更佳。我们提出了ReliabilityRoute，这是一种结构性干预方法，利用历史覆盖率、市场先验可用性、来源先验锐度、证据强度、证据分歧度和预测时间跨度等可靠性特征来引导预测智能体的行为。一个基于2024年数据拟合的固定规则无需硬编码来源即可与手工分类法紧密匹配……

    arXiv:2609.28475v1 Announce Type: new  Abstract: Forecasting agents increasingly combine language-model reasoning, retrieval, ensembling, and calibration, but it remains unclear when each behavior should be trusted. We study this question on ForecastBench-style binary forecasting tasks, treating the choice to retrieve, reason, defer to a market prior, or use a historical analog as an observable agent behavior rather than a hidden implementation detail. Our central finding is that mechanism choice is source-dependent: structured analogs dominate for some data-generating processes, while market/crowd-style and conservative baselines are better for others. We introduce ReliabilityRoute, a structural intervention that steers forecasting-agent behavior using reliability features such as historical coverage, market-prior availability, source-prior sharpness, evidence strength, evidence disagreement, and horizon. A fixed 2024-fitted rule closely matches a hand taxonomy without hard-coded sour
    
[^127]: SkillGym：将人类技能内化到大语言模型中以解决现实世界问题

    SkillGym: Internalizing Human Skills into LLMs for Real-World Problem Solving

    [https://arxiv.org/abs/2609.27717](https://arxiv.org/abs/2609.27717)

    SkillGym框架将人类编写的智能体技能转化为可执行、可验证的训练环境，通过构建2756个环境和收集大量成功轨迹来支持大语言模型的监督微调与强化学习，从而将人类技能内化为模型自身的可复用能力。

    

    人类编写的智能体技能蕴含着丰富的现实世界问题解决工作流程，但它们通常仅被用作推理时的外部指令，而非被内化为可复用的模型能力。我们提出了SkillGym，这是一个将这些技能转化为可执行、可验证的大语言模型智能体训练环境的框架。其技能到任务的流水线能够实例化具体任务，通过基于代码的检查器验证执行结果，并通过对比执行来评估经验性的技能依赖程度。我们构建并发布了涵盖12个类别的2,756个环境，并从多个模型和执行框架中收集了8,364条成功轨迹，平均包含49次工具调用和超过6万条记录的文本标记。这些资源支持在经验证的工作流程上进行监督微调，以及基于结果奖励的强化学习。在Claude Code框架下，监督微调使Qwen3.5-35B-A3B在GDPval-AA v2上提升了199 Elo分数，19...（原文摘要截断）

    arXiv:2609.27717v1 Announce Type: new  Abstract: Human-written agent skills encode rich workflows for real-world problem solving, but are typically used as external inference-time instructions rather than internalized as reusable model capabilities. We introduce \texttt{SkillGym}, a framework that transforms these skills into executable, verifiable training environments for large language model agents. Its skill-to-task pipeline instantiates concrete tasks, verifies outcomes with code-based checkers, and assesses empirical skill dependence through contrastive executions. We construct and release 2,756 environments across 12 categories and collect 8,364 successful trajectories from multiple models and harnesses, averaging 49 tool calls and over 60k logged text tokens. These resources support supervised fine-tuning on verified workflows and reinforcement learning with outcome-based rewards. Under Claude Code, supervised fine-tuning improves Qwen3.5-35B-A3B by 199 Elo on GDPval-AA v2, 19.
    
[^128]: 合成研究验证中的后果性行为与表征公平性

    Consequential Behaviour and Representational Fairness in the Validation of Synthetic Research

    [https://arxiv.org/abs/2609.27690](https://arxiv.org/abs/2609.27690)

    该论文指出现有的合成调查受访者验证方法在预测后果性行为的应用场景中检验了错误的目标，并提出一个要求效度声明必须明确与人类数据对应水平的验证框架，以保障合成研究的表征公平性。

    

    arXiv:2609.27690v1 公告类型：新 摘要：工业界和学术界的研究人员使用由大语言模型驱动的合成调查受访者作为人类样本的替代品。这些合成群体需要与现实世界数据进行验证，因此研究人员通常采用与人类调查进行临时比较的方式来完成这一工作。受行为科学中“意向-行为差距”概念的启发，我们认为，在大多数应用场景中——即决策者委托合成研究以预测后果性行为时——这些现有验证方法检验的是错误的东西。为解决这一问题，我们提出了一个包含两项要求的验证框架。第一，每个效度声明必须说明其与人类数据的对应水平：样本是否能预测所代表人群的实际行为、验证涉及四种诊断维度（位置、离散度、反应过程和结构）中的哪一种，以及验证是否与实验效应进行了比较？第二，研究人员必须报告效度验证……

    arXiv:2609.27690v1 Announce Type: new  Abstract: Researchers in industry and academia use synthetic survey respondents powered by large language models as substitutes for human samples. These synthetic populations require validation against real-world data, so researchers often address them using ad hoc comparisons with human surveys. Inspired by the intention-behaviour gap in behavioural science, we argue that these validations test the wrong thing for most applied cases where decision makers commission synthetic research to anticipate consequential behaviour. To address this problem, we propose a validation framework with two requirements. First, every validity claim must state its level of correspondence with human data: does the sample predict what the represented people do, which of four diagnostics (location, dispersion, response process and structure) does the validation address, and does the validation compare against experimental effects? Second, researchers must report validi
    
[^129]: 脑到语言解码：任务、信号、方法、评估、实际应用及未来展望

    Brain-to-Language Decoding: Tasks, Signals, Methods, Evaluation, Practical Use and Beyond

    [https://arxiv.org/abs/2609.27650](https://arxiv.org/abs/2609.27650)

    这是一篇关于脑到语言解码的系统性综述，将发音、内部和感知三类语言任务与对应的神经群体、解码器表征及输出形式相联系，全面梳理了侵入式与非侵入式测量下的方法、评估体系与实际应用的最新进展。

    

    脑到语言解码旨在将与语言产生、内部言语和感知相关的神经活动转化为语言或表达性输出。它为言语丧失后的沟通功能恢复提供了一条途径，同时也是研究大脑如何表征语言的手段。神经记录和表示学习技术的进步，已将该领域从受限的识别与声学重建扩展到文本生成、流式个性化语音和面部动画。本综述综合了侵入式与非侵入式测量方面的这些进展，检索范围不设年份下限，并以文献来源为导向更新至2026年9月。我们将发音言语、内部言语和感知言语三类任务与其所涉及的神经群体、解码器可用的表征以及这些表征所能支持的输出联系起来。我们考察了模型开发、公共资源以及评估方法的演变，并对……（摘要在此处截断）

    arXiv:2609.27650v1 Announce Type: new  Abstract: Brain-to-language decoding translates neural activity associated with language production, internal speech and perception into linguistic or expressive outputs. It offers a route to restoring communication after speech loss and a means of studying how the brain represents language. Advances in neural recording and representation learning have expanded the field from constrained recognition and acoustic reconstruction to text generation, streaming personalised speech and facial animation. This survey synthesises these developments across invasive and non-invasive measurements, drawing on a search without a lower year limit and source-led updates through September 2026. We connect Articulated, Inner and Perceived tasks to the neural populations they engage, the representations available to decoders and the outputs those representations can support. We examine model development, public resources and the evolution of evaluation, and compare 
    
[^130]: ProCredit：智能体强化学习中从结果奖励到进展信用的转变

    ProCredit: From Outcome Rewards to Progress Credit in Agentic Reinforcement Learning

    [https://arxiv.org/abs/2609.27532](https://arxiv.org/abs/2609.27532)

    提出 ProCredit，利用可在中间状态上运行的验收检查，把与最终结果同样可验证的任务进展转化为逐步的信用信号，从而克服长程智能体强化学习中仅依赖结果奖励导致的训练信号稀疏、失败尝试无法区分、推进任务的步骤得不到应得信用等问题。

    

    长程智能体任务要求智能体通过一系列工具调用对环境进行修改，任务成败由最终状态决定。标准做法是在任务结束时给出单一的结果奖励，并对同一任务采样得到的多条轨迹进行比较。由此带来的问题是：当一组采样中没有任何成功轨迹时，训练便得不到任何信号；失败的尝试无法按照其接近完成的程度加以区分；而真正推动任务进展的步骤与仅仅查询环境的步骤会获得相同的信用。已有工作或将比较的单位从轨迹细化为步骤，或训练一个奖励模型来提供中间信号：前者仍然只能从最终成败中获取信号，后者则需要依赖模型来估计信号。我们观察到，用于判定成功的验收检查同样可以作用于中间状态，因此任务进展与最终结果一样是可验证的。我们提出 ProCredit，它将这种经过验证的进展……（摘要原文在此处截断）

    arXiv:2609.27532v1 Announce Type: cross  Abstract: Long-horizon agentic tasks require an agent to modify an environment through a sequence of tool calls, with success determined by the final state. The standard recipe assigns a single outcome reward at the end and compares trajectories sampled for the same task. As a result, a group with no successful trajectory yields no training signal, failed attempts cannot be told apart by how close they came to completion, and turns that advance the task receive the same credit as turns that only query the environment. Prior work refines the unit of comparison from the trajectory to the step, or trains a reward model to supply intermediate signal: the former still derives its signal from final success alone, and the latter estimates it with a model. We observe that the acceptance checks that decide success can also be run on intermediate states, so progress is as verifiable as the outcome. We propose ProCredit, which turns this verified progress 
    
[^131]: LOCKR：一种用于检测与修复扩散语言模型中“稳定但错误”锁定现象的隐状态轨迹引导规划器

    LOCKR: A Hidden-State Trajectory-Guided Planner for Detecting and Repairing Stable-but-Wrong Lock-In in Diffusion Language Models

    [https://arxiv.org/abs/2609.27220](https://arxiv.org/abs/2609.27220)

    LOCKR利用扩散语言模型的隐状态轨迹来检测“稳定但错误”的锁定现象，并通过测试时规划动态分配计算、扩展针对性修复分支，实现对错误推理的选择性修复，其效果显著优于置信度、熵等表面信号。

    

    扩散语言模型通过迭代去噪生成文本，在最终答案产生之前会暴露出中间轨迹。我们识别出一种反复出现的推理失败现象——“稳定但错误的锁定”，即答案在大量去噪步骤尚未完成时就过早地稳定在一个错误的值上。诸如置信度、熵、边际和答案稳定性等表面层解码信号，不足以可靠地区分正确的锁定与错误的锁定。我们将选择性推理修复形式化为一个轻量级的测试时规划问题，并提出LOCKR——一个由隐状态轨迹引导的规划器，它决定何时分配额外的计算资源，扩展一组结构化的针对性修复分支，并利用轨迹感知验证来选择最有希望的后续生成路径。在两个扩散语言模型和三个数学推理基准测试上，隐状态轨迹持续优于表面信号和……（摘要在此处截断）

    arXiv:2609.27220v1 Announce Type: new  Abstract: Diffusion language models generate text through iterative denoising, exposing intermediate trajectories before final answers are produced. We identify a recurring reasoning failure, stable-but-wrong lock-in, where an answer stabilizes early around an incorrect value while substantial denoising remains. Surface-level decoding signals such as confidence, entropy, margin, and answer stability are insufficient to reliably distinguish correct from erroneous lock-in. We formulate selective reasoning repair as a lightweight test-time planning problem and propose LOCKR, a hidden-state trajectory-guided planner that decides when to allocate additional computation, expands a structured set of targeted repair branches, and selects the most promising continuation using trajectory-aware verification. Across two diffusion language models and three mathematical reasoning benchmarks, hidden-state trajectories consistently outperform surface signals and 
    
[^132]: 通用分形自然语言决策图：跨异构领域的实时边缘分诊

    Universal Fractal Natural Language Decision Map: Real-Time Edge Triage Across Heterogeneous Domains

    [https://arxiv.org/abs/2609.25498](https://arxiv.org/abs/2609.25498)

    该论文提出了一种无需存储任何权重张量（0 字节显存）的通用分形自然语言决策图，通过沿 Mandelbrot 集混沌边界动态调制 24 字节坐标种子来实时合成布尔、类别和序数三类确定性决策，从而以极低延迟和能耗实现跨异构领域的边缘端实时分诊。

    

    arXiv:2609.25498v1 公告类型：cross 摘要：部署大型语言模型进行运行时操作分诊会带来难以承受的延迟（>100-500 毫秒）、高昂的显存需求（>4-8 GB）以及过多的能量耗散。本文在 Mandelbrot 分形神经合成（Dagli 等，2026）的基础上，提出了通用分形自然语言决策图，通过 werr 机器原生边缘反射运行时和生产级 answerr 平台（https://answerr.me）实现。该引擎完全无需存储权重张量（0 字节显存），通过沿 Mandelbrot 集的混沌边界动态调制 24 字节坐标种子并评估四象限逃逸动力学，来合成确定性决策——noul（布尔型）、choice（类别型）和 score（序数型）。该引擎从生物“系统一”反射弧中汲取灵感，引入了：(i) 带有域投影器 Phi_D 的自动种子路由器，相比线性基线带来 +28.8% 的准确率提升；(ii) Infor……（原文摘要在此处被截断）

    arXiv:2609.25498v1 Announce Type: cross  Abstract: Deploying Large Language Models for runtime operational triage incurs prohibitive latency (>100-500 ms), high VRAM requirements (>4-8 GB), and excessive energy dissipation. Extending Mandelbrot Fractal Neural Synthesis (Dagli et al., 2026), this paper presents the Universal Fractal Natural Language Decision Map, realized via the werr machine-native edge reflex runtime and the production answerr platform (https://answerr.me). Operating entirely without stored weight tensors (0 Bytes VRAM), the engine synthesizes deterministic decisions---noul (Boolean), choice (categorical), and score (ordinal)---by dynamically modulating 24-byte coordinate seeds along the chaotic boundary of the Mandelbrot set and evaluating 4-quadrant escape dynamics. Drawing inspiration from biological System-One reflex arcs, the engine introduces: (i) an Auto-Seed Router with domain projector Phi_D yielding a +28.8% accuracy gain over linear baselines; (ii) an Infor
    
[^133]: 压力下的行为：当用户施压时，六十个语言模型会怎么做

    Conduct Under Pressure: What Sixty Language Models Do When a User Pushes

    [https://arxiv.org/abs/2609.25447](https://arxiv.org/abs/2609.25447)

    该研究对13家厂商的60个语言模型在用户施压情境下进行了大规模行为评测，发现模型“是否屈服于压力”取决于模型代际新旧（新模型更坚定，屈服率与能力指数相关达-0.64），而“以何种方式坚持或屈服”则由厂商特征决定。

    

    我们研究了当用户在不舒适的情境中向大语言模型施压时模型的表现：用户坚持己见、恳求、奉承或哀伤，而模型可能放弃正确的事实、编写它本应拒绝的文档，或为一个会让用户蒙受金钱损失的计划叫好。我们向来自13家厂商的60个模型发送了固定的多轮对话场景（对每个模型完全相同，不随模型回复而变化），并用通过开放编码构建后冻结的编码手册对每份对话记录进行标注：包括一个轨迹（模型坚持立场还是屈服）和一种方式（它如何坚持或屈服）。两项发现界限分明：模型是否坚持与模型代际相关，即模型的更新程度：屈服率与公开的能力指数呈Spearman相关系数-0.64，厂商效应很小。而模型如何坚持则取决于厂商：17个方式编码中有6个按厂商显著区分（置换检验p ≤ 0.001，已对整个编码手册进行多重校正）。我们报告了在通过信度检验的编码上得出的四种厂商画像。

    arXiv:2609.25447v1 Announce Type: new  Abstract: We study what LLMs do when a user applies pressure in an uncomfortable situation: a user insists, begs, flatters or grieves, and the model gives up a correct fact, writes a document it should refuse, or cheers a plan that will cost the user money. We send frozen multi-turn scenes, identical for every model regardless of the reply, to 60 models from 13 vendors, and label each transcript with a codebook built by open coding and then frozen: a trajectory (the model held its position or folded) and a manner (how it held or folded). Two findings separate. Whether a model holds tracks its generation, meaning how recent it is: fold rate correlates with a public capability index at Spearman -0.64, with little vendor effect. How it holds tracks the vendor: six of the 17 manner codes sort by vendor at permutation p <= 0.001, corrected across the codebook. We report four vendor profiles on the codes that cleared reliability.   We also ask which par
    
[^134]: Qwen-Audio-3.1-Realtime：迈向可靠的智能体语音交互

    Qwen-Audio-3.1-Realtime: Towards Reliable Agentic Voice Interaction

    [https://arxiv.org/abs/2609.25176](https://arxiv.org/abs/2609.25176)

    Qwen-Audio-3.1-Realtime 通过“思考—行动—说话协调”三大模块（结合多教师在线策略蒸馏与基于GRPO的强化学习），将实时语音助手的整体任务成功率从78.4%提升至82.0%，实现了可靠的智能体语音交互。

    

    实时语音助手必须能够对不断演变的请求进行推理、执行操作并遵循对话规则。Qwen-Audio-3.1-Realtime 通过“思考”、“行动”以及“说话与协调”三大机制将这些要求整合在一起。思考模块将 Core-Cocktail 监督微调与多模态、多教师在线策略蒸馏（M²-OPD）相结合，以迁移语言能力并发展原生的音频技能。行动模块利用自进化的可执行环境和多粒度 rollout 进行群体相对策略优化（GRPO），教会模型使用工具、解读反馈并完成任务。说话与协调模块则对齐助手何时、如何以及是否说话或行动。我们在音频推理、多语言理解、工具使用、对话行为、全双工交互和安全性等方面进行了评估。与 Qwen-Audio-3.0-Realtime 相比，3.1 在半双工语音到文本适配任务上将整体任务成功率从 78.4% 提升至 82.0%。

    arXiv:2609.25176v1 Announce Type: cross  Abstract: Real-time voice assistants must reason over evolving requests, execute actions, and follow conversational rules. Qwen-Audio-3.1-Realtime brings these requirements together through Think, Act, and Speak and Coordinate. Think combines Core-Cocktail supervised fine-tuning with Multimodality and Multi-Teacher On-Policy Distillation (M$^{2}$-OPD) to transfer language capabilities and develop native audio skills. Act uses self-evolving executable environments and multi-granularity rollouts for Group Relative Policy Optimization (GRPO), teaching the model to use tools, interpret feedback, and complete tasks. Speak and Coordinate aligns how, when, and whether the assistant speaks or acts. We evaluate audio reasoning, multilingual understanding, tool use, conversational behavior, full-duplex interaction, and safety. Compared with Qwen-Audio-3.0-Realtime, 3.1 raises overall task success from 78.4% to 82.0% on our half-duplex speech-to-text adapt
    
[^135]: RRSI：智能体框架的正则化递归自我改进

    RRSI: Regularized Recursive Self-Improvement of Agent Harnesses

    [https://arxiv.org/abs/2609.24972](https://arxiv.org/abs/2609.24972)

    该论文提出RRSI方法，通过将正则化原则（如时间退火的编辑预算限制和鼓励探索未开发轨迹）引入智能体框架的递归自我改进过程，防止递归进化对训练任务过拟合，从而提升分布外基准上的泛化能力。

    

    LLM智能体的能力在很大程度上被其“框架”所放大，即围绕冻结骨干模型的提示词、控制流、工具、记忆和上下文管理。近期的方法通过迭代地提出并选择对智能体框架的组件级编辑，日益将这一过程自动化，实际上在智能体系统层面建立了一种递归自我改进（RSI）的形式。然而，这种递归进化可能因记忆训练任务而过拟合，在分布内基准上表现出巨大收益，但在分布外基准上收益缩小甚至消失。我们提出了智能体框架的正则化递归自我改进（RRSI），通过约束进化候选的提案与选择，将正则化原则融入框架自我改进之中。提案者以时间退火的预算运行，限制候选可以捆绑的编辑数量，并鼓励探索未开发的轨迹……

    arXiv:2609.24972v1 Announce Type: cross  Abstract: An LLM agent's capability is largely magnified by its harness, namely the prompts, control flow, tooling, memory, and context management surrounding the frozen backbone model. Recent methods increasingly automate this process by iteratively proposing and selecting component-wise edits of an agent harness, practically establishing a form of recursive self-improvement (RSI) at the agent-system level. However, such recursive evolution may overfit by memorizing the training tasks, showing large in-distribution gains that shrink or even vanish on out-of-distribution benchmarks. We introduce Regularized Recursive Self-Improvement of Agent Harnesses (RRSI), which incorporates the principles of regularizations into harness self-improvement by constraining the evolution candidate proposal and selection. The proposer operates with a temporally annealed budget, limiting how many edits a candidate can bundle, and it encourages unexplored trajector
    
[^136]: 大语言模型在序贯临床分诊中锚定于主诉且未能整合证据

    LLMs Anchor on Chief Complaint and Fail to Integrate Evidence in Sequential Clinical Triage

    [https://arxiv.org/abs/2609.22904](https://arxiv.org/abs/2609.22904)

    该研究提出了评估大语言模型在序贯急诊分诊任务上的新方法学，发现尽管LLM在完整病历上表现接近医生，但在逐轮预测分诊等级时性能显著退化，原因是模型过度锚定于主诉信息而未能整合对话中后续出现的证据。

    

    急诊科（ED）的分诊是一个逐轮展开的序贯决策过程。现有针对大语言模型（LLM）分诊能力的评估均使用完整的回顾性病历记录，并报告其性能接近医生水平。我们提出了一种评估LLM在序贯分诊任务上表现的方法学，该任务要求从不断增长的护患对话前缀中预测分诊紧急程度标签。我们在两个语料库上对六个LLM在五个序贯检查点进行了评估：425个LLM生成的（SIMULATED）对话和50个医生撰写的（CLINICIAN）对话，两者均按照紧急严重程度指数（ESI）进行标注。以二次加权kappa（QWK）衡量，所有模型在完整病历记录上表现出中等至高度的一致性，但在每个序贯检查点上都下降为勉强至中等的一致性。受控扰动实验表明，每个检查点上的预测标签都锚定于主诉交流部分，而提示干预……（原文摘要在此处截断）

    arXiv:2609.22904v1 Announce Type: new  Abstract: Triage in the emergency department (ED) is a sequential decision process that unfolds turn by turn. Existing evaluations of large language models (LLMs) for triage use completed retrospective records and report performance close to that of physicians. We implement a methodology for evaluating LLMs on sequential triage, the task of predicting a triage acuity label from a growing prefix of a nurse-patient conversation. We evaluate six LLMs at five sequential checkpoints on two corpora: 425 LLM-generated (SIMULATED) and 50 physician-authored (CLINICIAN) conversations, both labelled under the Emergency Severity Index (ESI). Every model, measured by quadratic weighted kappa (QWK), degrades from moderate-to-substantial agreement on completed records to fair-to-moderate agreement at every sequential checkpoint. Controlled perturbations show that the label at every checkpoint is anchored on the chief complaint exchanges, and prompting interventi
    
[^137]: 一个语言模型评审团相当于多少个人类评判者？

    How Many Humans Is a Judge Panel Worth?

    [https://arxiv.org/abs/2609.21277](https://arxiv.org/abs/2609.21277)

    该论文提出两种不同的“等效人类评判者数量”度量——谱残差多样性 ν_H 与分布平方误差 ν_MSE，发现同一组 32 个语言模型评审在三个 ChaosNLI 任务上分别相当于约 4.24–6.50 个和 2.30–3.75 个人类评判者，且更大的谱多样性并不保证更好的分布恢复。

    

    一组语言模型评审团究竟代表多少个人类判断？答案取决于匹配的对象是什么。我们针对经验人类标签分布对类别型评审团进行审计，保留了那些相对于单一金标准标签的二值错误所坍缩掉的分歧。我们通过将归一化残差格拉姆矩阵的参与率与条件独立的人类参考抽样相匹配来度量谱残差多样性，得到 ν_H；并单独匹配分布平方误差，得到 ν_MSE。在三个 ChaosNLI 任务上，同一组由 32 个评审组成的评审团，其 ν_H 为 4.24–6.50，而 ν_MSE 仅为 2.30–3.75。一个谱恒等式将决定误差的特征值、成员能量和平均方向权重分离开来。可实现的硬标签评审团表明，即使成员能量相等且相关性非负，更大的谱多样性也可能伴随更差的分布恢复。在观察到的评审团中，同规模内的排名一致性……（摘要原文在此处截断）

    arXiv:2609.21277v1 Announce Type: new  Abstract: How many human judgments does a panel of language models represent? The answer depends on what is matched. We audit categorical judge panels against empirical human label distributions, retaining disagreement that binary errors relative to one gold label collapse. We measure spectral residual diversity by matching the participation ratio of a normalized residual Gram matrix to conditionally independent human-reference draws, giving nu_H. We separately match distributional squared error, giving nu_MSE. Across three ChaosNLI tasks, the same 32-judge panels have nu_H=4.24--6.50 but nu_MSE=2.30--3.75. A spectral identity separates the eigenvalues, member energies, and averaging-direction weights that determine error. Realizable hard-label panels show that greater spectral diversity can accompany worse distribution recovery even with equal member energies and nonnegative correlations. In the observed panels, within-size ranking agreement vari
    
[^138]: ECHO：面向全双工对话中上下文敏感话轮转换的配对对比基准

    ECHO: A Matched-Contrast Benchmark for Context-Sensitive Turn-Taking in Full-Duplex Dialogue

    [https://arxiv.org/abs/2609.17360](https://arxiv.org/abs/2609.17360)

    ECHO是一个中文全双工对话话轮转换的配对对比诊断基准，通过相同重叠内容但相反上下文语境的样本配对和新的配对准确率指标，揭示了现有全双工系统普遍存在偏向“让出话轮”的固定策略而非真正上下文敏感决策的问题。

    

    全双工语音对话系统必须区分需要让出话轮的打断行为与允许继续说话的附和（backchannel）。现有基准通常独立评估各个事件，因此可能奖励固定的动作偏好而非上下文敏感的决策。我们提出了ECHO，一个用于中文全双工话轮转换的配对诊断基准。ECHO将具有相同重叠语音内容但前置多轮对话语境不同的样本进行配对，其中一个需要“让出”（Yield），另一个需要“保持”（Keep）。它还包含旁语音样本，用于诊断不必要的让出行为。我们引入了配对准确率指标，该指标要求对一对样本中的两个成员都做出正确决策，且对恒定动作策略不给予任何得分。在多个全双工系统上的实验表明，大多数系统表现出明显的“让出”偏向，在打断情况下的表现显著优于附和情况，而另一个系统…

    arXiv:2609.17360v1 Announce Type: new  Abstract: Full-duplex spoken dialogue systems must distinguish interruptions that require yielding the floor from backchannels that permit continued speaking. Existing benchmarks typically evaluate events independently and may therefore reward fixed action preferences rather than context-sensitive decisions. We introduce ECHO, a paired diagnostic benchmark for Chinese full-duplex turn-taking. ECHO pairs examples with the same overlap transcript but contrasting preceding multi-turn dialogue contexts, with one requiring Yield and the other Keep. It additionally includes off-talk examples for diagnosing unnecessary yielding. We introduce pair accuracy, which requires correct decisions on both members of a pair and assigns no credit to constant-action policies. Experiments on multiple full-duplex systems show that most exhibit a pronounced bias toward \textsc{Yield}, performing substantially better on interruptions than on backchannels, while another 
    
[^139]: 这个论断有多宽泛？对NLP研究中泛化表述的映射分析

    How broad is that claim? Mapping Generalisation in NLP Research

    [https://arxiv.org/abs/2609.14770](https://arxiv.org/abs/2609.14770)

    该论文提出了科学领域泛化表述分类体系NLPGenX、基于大语言模型的自动分类框架NLPGenA以及大规模标注数据集NLPGens，用于自动检测NLP研究论文中对泛化表述的过度使用和可能存在的表述偏差。

    

    泛化表述在科学交流中十分常见，尽管它们在语义上往往是模糊的。为了帮助检测对泛化表述的过度依赖以及可能对科学发现造成的歪曲，需要一种自动化方法来识别论断并根据其泛化程度进行分类。我们引入了一个全面的科学领域泛化表述分类体系NLPGenX，它根据论断的泛化程度及其在文本中的表述框架对其进行标注。我们通过一个基于大语言模型（LLM）的框架NLPGenA将该分类体系操作化，该框架能够自动将科学文章中的句子分类为5种不同的泛化类别。我们通过人工标注者对该框架进行了验证，并利用该框架构建了一个大规模的NLP论文标注数据集NLPGens，其中包含泛化程度标注以及关于模糊限定词（hedging）和模糊描述词的辅助标签。我们使用NLPGens分析泛化表述的使用情况……（原文摘要在此处截断）

    arXiv:2609.14770v1 Announce Type: cross  Abstract: Generalisations are common in scientific communication, even though they are semantically ambiguous. An automated method is needed to identify and categorise claims according to their level of generalisation, in order help detect an over-reliance on generalisations and possible misrepresentations of scientific findings. We introduce a comprehensive taxonomy of generalisations in the scientific domain, NLPGenX, which labels claims according to their level of generality and framing within the text. We operationalise this taxonomy with an LLM-powered framework, NLPGenA, that automatically classifies sentences from scientific articles into 5 different generalisation classes. We validate our framework with human annotators and use the framework to construct a large-scale dataset of NLP papers annotated according to generality, with auxiliary labels for hedging and vague descriptors (NLPGens). We use NLPGens to analyse the use of generalisat
    
[^140]: DuplexDrama：一个包含场景设定、全双工行为、富表现力语音和声音事件的合成对话数据集

    DuplexDrama: A Synthesized Dialogue Dataset with Scenarios, Full-Duplex Behaviors, Expressive Speech, and Sound Events

    [https://arxiv.org/abs/2609.12872](https://arxiv.org/abs/2609.12872)

    DuplexDrama是首个同时涵盖完整人设场景、三种全双工行为、带情感标签的富表现力语音和剧本感知声音事件四个维度的合成口语对话数据集，包含超过2,000小时音频，并将发布800小时中英双语子集以推动全双工口语对话模型研究。

    

    我们提出了DuplexDrama，这是首个同时涵盖四个维度的合成口语对话数据集：(i) 完整的人设与场景设定；(ii) 三种全双工行为（打断、附和、未完成话语）；(iii) 带有与人设对齐的情感标签的富表现力语音；(iv) 剧本感知的声音事件。DuplexDrama通过4阶段流水线构建，对剧本和合成音频的质量验证证实了其质量。我们产出了超过2,000小时的音频数据，使用了包含64种音色的音色库，涵盖13个人设和5个年龄段；所有对话轮次中有3.8%包含至少一种全双工行为。这些数据已通过内部全双工模型训练得到验证。我们将发布一个包含6,400条双语对话的精选子集（800小时，中文约500小时 + 英文约300小时），以推动全双工口语对话模型的研究。数据样本可在我们的演示页面获取，LLM评判评估提示词也将被发布。

    arXiv:2609.12872v1 Announce Type: new  Abstract: We present DuplexDrama, the first synthesized spoken dialogue dataset that simultaneously covers four dimensions: (i) complete persona and scenario settings; (ii) three full-duplex behaviors (interruption, backchannel, incomplete); (iii) expressive speech with persona-aligned emotion labels; and (iv) script-aware sound events. DuplexDrama is built via a 4-stage pipeline; quality validation on both scripts and synthesized audio confirms its quality. We have produced more than 2,000 hours audio data with a 64-voice timbre pool spanning 13 personas and 5 age buckets; 3.8% of all turns carry at least one full-duplex behavior. This data has been validated through internal full-duplex model training. We will release a curated subset of 6,400 bilingual dialogues (800 h, Chinese ~500 h + English ~300 h) to advance full-duplex spoken dialogue model research. Data samples are available at our demo page and LLM-judge evaluation prompts will be rele
    
[^141]: 同日同故事，隔日异信号：金融情感分析的双重效度

    Same Day, Same Story; One Day Ahead, a Different Signal: The Dual Validity of Financial Sentiment

    [https://arxiv.org/abs/2609.11144](https://arxiv.org/abs/2609.11144)

    本文基于2002-2025年证券集体诉讼语料库，将70,500条X消息与异常股票收益相关联，通过统一流程测试五种情感分析工具，发现金融情感工具的人工标注一致性（构念效度）与其市场预测能力（预测效度）之间的关系并非恒定，而是取决于抽样惯例和分数表示方式。

    

    金融自然语言处理（Financial NLP）领域有一个标准工作流程：先验证情感分析工具与人工标注的一致性，然后信任它来提取市场信号。这背后隐含的假设是，这两种评估衡量的是同一件事。我们在一个可以同时测量两者的场景中检验了这一假设：一个证券集体诉讼语料库（2002-2025年），将70,500条X平台消息与异常股票收益相关联，并包含一个由单一标注员人工标注的金标准样本。通过将五种工具（VADER、Loughran-McDonald、FinBERT、Twitter-RoBERTa和一个LLM标注器）运行在完全相同的流程中，我们发现构念效度与预测效度之间的关系取决于抽样惯例和分数表示方式。在传统的方法特定抽样下，人工一致性评分与同日的分级关联更为吻合，而与提前一天的关联吻合度较低。然而，在固定样本量的面板数据上，一致性评分在两个时间范围内都表现出相似的分级秩相关，而粗粒度排序在两种情况下均较弱。

    arXiv:2609.11144v1 Announce Type: cross  Abstract: Financial NLP has a standard workflow: validate a sentiment tool against human labels, then trust it to extract market signal. This assumes the two evaluations measure the same thing. We test that assumption in a setting where both can be measured at once: a corpus of securities class actions (2002-2025) linking 70,500 X messages to abnormal stock returns, with a single-annotator human labelled gold sample. Running five instruments (VADER, Loughran-McDonald, FinBERT, Twitter-RoBERTa, and an LLM annotator) through one identical pipeline, we find that the relationship between construct and predictive validity depends on the sampling convention and score representation. Under conventional method-specific sampling, human agreement aligns more closely with graded same-day associations than with one-day leads. On a fixed-n panel, however, agreement has similar graded rank correlations at both horizons, while the coarse ordering remains weak.
    
[^142]: 通过能量驱动的潜在冲突检测对抗指令冲突

    Combating Instruction Conflict via Energy-Driven Latent Conflict Detection

    [https://arxiv.org/abs/2609.08646](https://arxiv.org/abs/2609.08646)

    该论文提出ELCD，一种能量驱动的响应级潜在冲突检测器，通过对完整生成输出的复合隐藏状态表示进行成对边际排序学习，在生成后交付前阶段有效检测静态输入检查无法发现的“响应漂移”现象，防止用户指令覆盖系统级约束。

    

    大型语言模型（LLM）越来越多地以分层指令的方式部署，但它们仍然容易受到用户指令覆盖系统级约束的冲突影响。现有的防御机制主要集中于静态输入检查，因此无法检测“响应漂移”（Response Drift）现象——即尽管输入看似合规，模型的最终响应却违反了系统级约束。为弥补这一空白，我们提出了ELCD，一种用于生成后、交付前验证的响应级潜在冲突检测器。给定完整生成的输出，ELCD通过将最后一个词元的嵌入与均值池化的响应嵌入相拼接，构建复合隐藏状态表示。随后，它优化一个成对边际排序目标，以在潜在空间中区分合规响应与漂移响应。在从1.5B到14B参数的五个主流LLM上的大量实验证明（摘要原文在此处截断）。

    arXiv:2609.08646v2 Announce Type: replace  Abstract: Large Language Models (LLMs) are increasingly deployed with hierarchical instructions, yet they remain vulnerable to conflicts in which user directives override system-level constraints. Existing defense mechanisms predominantly focus on static input inspection and therefore fail to detect Response Drift, a phenomenon in which the model's final response violates system-level constraints despite seemingly compliant inputs. To bridge this gap, we introduce ELCD, a response-level latent conflict detector for post-generation, pre-delivery verification. Given the full generated output, ELCD constructs a composite hidden-state representation by concatenating the final-token embedding with the mean-pooled response embedding. It then optimizes a pairwise margin ranking objective to separate compliant and drifting responses in latent space. Extensive experiments across five mainstream LLMs ranging from 1.5B to 14B parameters demonstrate that 
    
[^143]: LLM取证：后门藏身何处？利用稀疏自编码器定位与控制触发机制

    LLM Forensics: Where Do Backdoors Hide? Localizing and Controlling Trigger Mechanisms with Sparse Autoencoders

    [https://arxiv.org/abs/2609.07746](https://arxiv.org/abs/2609.07746)

    本文在受控的语言切换后门场景中，通过跨层、跨Transformer组件训练稀疏自编码器（SAE）来定位触发机制，发现SAE特征虽能以近乎完美的F1分数检测出触发提示，但检测触发的特征并不必然能因果地控制后门行为。

    

    尽管大语言模型中的后门问题日益受到关注，但其内部运作机制仍处于深入审查之中。基于触发的后门在行为层面很容易定义——一个罕见输入使模型切换到选定的响应模式——但触发器与其响应之间的机制却不甚明了。我们在一个受控且无害的语言切换设置中研究这一机制：固定的触发序列使1B和8B参数的语言模型用法语或德语继续完成英语提示。为此，我们在各个层和Transformer组件上训练稀疏自编码器（SAE），并将触发提示与翻译对照及预训练对照进行比较，以识别与触发相关的特征方向。我们展示了SAE特征如何以近乎完美的F1分数将触发提示与对照样本区分开来，但检测触发的特征并不一定能控制该行为。在干预测试中，注意力和MLP的特征通常在触发器上可靠激活……（原文摘要在此处截断）

    arXiv:2609.07746v2 Announce Type: replace  Abstract: Even though backdoors in LLMs have been a growing concern, their inner workings are still under heavy scrutiny. Trigger-based backdoors are easy to define behaviorally, a rare input that makes the model switch to a chosen response pattern, but the mechanism between triggers and their responses is less clear. We study this mechanism in a controlled, harmless language-switching setting, where fixed trigger sequences make 1B and 8B language models continue English prompts in French or German. For this, we train sparse autoencoders (SAEs) across layers and transformer components, then compare triggered prompts with translation and pretraining controls to identify trigger-relevant feature directions. We show how SAE features separate triggered prompts from controls with near-perfect F1, but features that detect the trigger do not necessarily control the behavior. In intervention tests, attention and MLP features often fire reliably on tri
    
[^144]: 分数背包问题的基于分组的资源分配模型

    A Group-Based Resource Allocation Model for the Fractional Knapsack Problem

    [https://arxiv.org/abs/2609.06470](https://arxiv.org/abs/2609.06470)

    该论文提出了一种两阶段分组资源分配模型，通过将属性相近的物品分组来缓解Dantzig贪婪规则中因微小扰动导致的分配不稳定问题，并给出了与精确最优解相比的紧损失上界。

    

    为了解决分数背包问题，Dantzig贪婪规则根据物品的价值-成本比对其进行排序。这种排序引入了优先级问题：如果预算在两个比率非常相似的物品之间耗尽，输入的任意微小扰动都可能改变分配结果。为了缓解这一问题，我们引入了一种两阶段规则。我们将半径 $\delta$ 内共享属性的物品进行分组，然后按比率的降序评估这些组，并在不进行进一步排序的情况下分配各组的预算份额。考虑一个具有总容量 $U_G$、单位成本在 $[w^-,w^+]$ 范围内、代表性价值为 $\widehat{v}$ 的组，与精确最优解相比，该组的损失被界定为 $\widehat{v}\, U_G\frac{w^+-w^-}{w^++w^-}+\varepsilon_v U_G$，其中 $\varepsilon_v$ 限制组内部价值的变化。此外，对于任意组大小，该调和因子仍然是紧的。

    arXiv:2609.06470v2 Announce Type: replace-cross  Abstract: To solve the fractional knapsack problem, Dantzig's greedy rule orders items according to their value-to-cost ratio. This ordering introduces priority issues. An arbitrarily small perturbation to the input can change the allocation if the budget is exhausted between two items with very similar ratios. To mitigate that problem, we introduce a two-stage rule. We group items sharing attributes within a radius $\delta$. These groups are then evaluated in descending order of ratio, and divide their group's budget share without further ranking. Consider a group featuring an aggregate capacity $U_G$, unit costs contained in $[w^-,w^+]$, and a representative value $\widehat{v}$. The group's loss compared to the exact optimum is bounded by $\widehat{v}\, U_G\frac{w^+-w^-}{w^++w^-}+\varepsilon_v U_G$, in which $\varepsilon_v$ limits the group's internal value variation. Moreover, for any group size, this harmonic factor remains tight. Th
    
[^145]: 重复提问会耗尽大语言模型的品牌推荐，却耗不尽其引用来源

    Repeated Queries Exhaust an LLM's Brand Recommendations but Not Its Sources

    [https://arxiv.org/abs/2609.05059](https://arxiv.org/abs/2609.05059)

    重复提问相同的购买问题时，不联网检索的大语言模型会不断涌现新的品牌推荐而难以饱和，启用检索的引擎则快速封闭品牌列表，但所有引擎引用的来源域名始终持续增加、远未收敛。

    

    重复提出相同的购买类问题是否会耗尽语言模型的品牌推荐，取决于其是否具备检索能力。在涵盖300个“问题-引擎”组合单元的实验中（50个问题、6个引擎、每个单元运行15次，并对1,470个经人工裁定的组织进行开放式抽取），五个不借助网络搜索作答的引擎在86-92%的单元中到第15次运行时仍在出现从未见过的品牌，其品牌储备中位数为15-31个组织；而唯一启用检索功能的引擎则封闭了其品牌列表（中位数8个组织，64%的单元仍在新增），这与此前四个深度实验单元中启用网络搜索的运行在第10次左右即趋于饱和的结果相符。在所有测试的时间范围内，被引用域名的累积量持续上升：四个深度实验单元在第24次运行时仍在新增域名，仅观测到Chao2下限估计值的59-84%，检索型引擎的广度单元中也有44%在第15次运行时仍在新增域名。单次运行仅能覆盖五次运行品牌集合的62-77%，且跨引擎来看，每个问题的中位数可引出38个组织……（摘要原文在此处截断）

    arXiv:2609.05059v1 Announce Type: cross  Abstract: Whether repeated identical buying questions exhaust a language model's brand recommendations depends on retrieval. Across 300 question-engine cells (50 questions, six engines, 15 runs each, open extraction over 1,470 adjudicated organizations), the five engines answering without web search were still adding never-seen brands at run 15 in 86-92% of cells, with median repertoires of 15-31 organizations; the one retrieval-enabled engine closed its list (median 8 organizations, 64% of cells still adding), matching four earlier deep cells where web-search runs saturated by run ten. Cited-domain accumulation keeps rising at every horizon tested: four deep cells were still adding domains at run 24 with 59-84% of the Chao2 lower-bound estimate observed, and 44% of the retrieval engine's breadth cells were still adding domains at run 15. A single run shows 62-77% of the five-run brand set, and across engines the median question draws 38 organiz
    
[^146]: 校准是瓶颈：多轮工具调用的动作类别诊断

    Calibration is the Bottleneck: An Action-Class Diagnostic of Multi-Turn Tool-Calling

    [https://arxiv.org/abs/2609.00949](https://arxiv.org/abs/2609.00949)

    本文提出一个基于四类动作空间的诊断框架，通过引入“准确率不超过黄金动作召回率”的自揭示上界，将多轮工具调用失败分解为动作类别失准与动作执行失败两种正交模式，从而揭示开源模型总体准确率追平闭源模型的表象背后，动作类别校准才是真正的瓶颈。

    

    多轮工具调用是大语言模型（LLM）智能体的一项核心评测场景。在公开的工具调用基准上，开源权重模型的总体准确率已接近甚至超越闭源前沿模型。然而，这一指标是对众多不同多轮情境的取平均，掩盖了进展是否在这些情境之间均衡分布。我们提出一种面向动作类别的诊断框架，将多轮失败分解为两种正交模式：动作类别失准与动作执行失败。该框架在四类动作空间（TOOL_CALL/ASK/REFUSE/CONFIRM）上运行，并引入一个自我揭示的上界 Acc ≤ GAR（黄金动作召回率）；两种失败模式分别表现为上界被违反（Acc > GAR，暴露出状态评分器对失准的掩盖）以及较大的上界余量（GAR >> Acc，将执行失败定位于 TOOL_CALL 内部）。我们在一组工具调用模型上对该框架进行了验证……（原文摘要在此处截断）

    arXiv:2609.00949v1 Announce Type: cross  Abstract: Multi-turn tool calling is a core evaluation scenario for large language model (LLM) agents. On public tool-calling benchmarks, open-weight models now approach or even surpass closed-source frontier models in aggregate accuracy. However, this metric averages over many different multi-turn situations and obscures whether progress is balanced across them. We propose an action-class-oriented diagnostic framework that decomposes multi-turn failures into two orthogonal modes: action-class miscalibration and action-execution failure. The framework operates over a four-class action space (TOOL_CALL/ASK/REFUSE/CONFIRM) and introduces a self-revealing upper bound Acc <= GAR (Gold Action Recall); the two modes show up as bound violation (Acc > GAR, exposing state-grader masking of miscalibration) and large bound slack (GAR >> Acc, localizing execution failure within TOOL_CALL). We validate it on a panel of tool-calling models across multiple mul
    
[^147]: 在策略蒸馏真的在蒸馏吗？从噪声教师到自我改进

    Does On-Policy Distillation Really Distill? From Noisy Teacher to Self-Improvement

    [https://arxiv.org/abs/2608.31046](https://arxiv.org/abs/2608.31046)

    研究发现在策略蒸馏中教师监督充满噪声且学生对其并不敏感，其性能提升主要来自对低概率token的学习，即使使用固定负优势信号也能达到同样效果，因此OPD本质上更接近自我改进而非真正的知识蒸馏。

    

    arXiv:2608.31046v1 公告类型：cross 摘要：在策略蒸馏（On-policy distillation, OPD）提供了密集的token级别监督，作为可验证奖励强化学习（RLVR）中稀疏结果级优势信号的替代方案。然而，教师是对学生生成的轨迹进行评分，而这些轨迹对教师而言本质上是非在策略的，因此其监督的可靠性，以及学生改进的真正来源，仍然不清楚。我们定量分析了OPD训练过程中教师监督的质量，发现其中存在大量噪声，且噪声的普遍程度随教师模型规模的增大而增加。令人惊讶的是，学生策略对这种噪声并不敏感，无论保留还是移除噪声监督，学生都收敛到相近的性能。那么OPD到底在蒸馏吗？通过分析其性能提升的驱动因素，我们发现学习主要集中在低对数概率的token上，而且使用单一固定的负优势就能达到与教师提供优势相当的性能。这表明OPD在很大程度上是……（原文在此处截断）

    arXiv:2608.31046v1 Announce Type: cross  Abstract: On-policy distillation (OPD) offers dense token-level supervision as an alternative to the sparse outcome-level advantages of reinforcement learning with verifiable rewards (RLVR). However, the teacher scores student-generated trajectories that are inherently off-policy for it, so the reliability of its supervision, and hence the source of the student's improvement, remains unclear. We quantitatively analyze teacher supervision during OPD training and find substantial noise whose prevalence increases with teacher scale. Surprisingly, the student policy is insensitive to such noise, converging to comparable performance regardless of whether noisy supervision is retained or removed. Does OPD distill at all? By analyzing what drives its gains, we find that learning concentrates on low log-probability tokens, and using a single fixed negative advantage matches the performance of teacher-provided ones. This suggests that OPD works largely b
    
[^148]: J-Zero：从零数据出发的统一挑战者-求解者-评判者协同进化

    J-Zero: Unified Challenger--Solver--Judge Co-Evolution from Zero Data

    [https://arxiv.org/abs/2608.26582](https://arxiv.org/abs/2608.26582)

    J-Zero提出了一种统一的挑战者-求解者-评判者协同进化框架，通过对抗性任务生成和基于生成方式的偏好对，实现了无需人工数据即可在可验证和不可验证领域中的自我进化。

    

    arXiv:2608.26582v1 公告类型：交叉 摘要：自我进化语言模型最近成为通往超级智能的一条有前景的路径，其优势在于减少人类监督成本。尽管在可验证领域已取得显著进展，但自我进化在不可验证领域仍研究不足。我们提出了从零数据出发的评判者协同适应（J-Zero），这是一个统一的挑战者-求解者-评判者协同进化框架，支持在两种领域中的自我改进。挑战者和求解者通过对抗性互动协同进化：挑战者生成越来越难的任务，而求解者学习产生更高质量的响应。与此同时，评判者通过使用偏好对进行协同适应，这些偏好对的顺序是预先已知的，基于每个响应的生成方式，即求解者的答案优于挑战者的答案，以及其分解再组合的答案优于其一次性答案，而非基于评判者自身的评分。

    arXiv:2608.26582v1 Announce Type: cross  Abstract: Self-evolving language models have recently emerged as a promising path toward superintelligence, with the advantage of reducing the cost of human supervision. While considerable progress has been made in verifiable domains, self-evolution in unverifiable domains remains substantially less explored. We propose Judge co-adaptation from Zero data (J-Zero), a unified Challenger--Solver--Judge co-evolution framework that supports self-improvement across both domains. The Challenger and Solver co-evolve through an adversarial interaction: the Challenger generates increasingly difficult tasks, while the Solver learns to produce higher-quality responses to them. In parallel, the Judge co-adapts using preference pairs whose ordering is known in advance from how each response was produced, i.e., the Solver's answer over the Challenger's, and its decomposed-and-recombined answer over its one-shot answer, rather than from the Judge's own scores. 
    
[^149]: 自我编写的指标：从自身盲点演化出评估器

    Metrics That Write Themselves: Evolving an Evaluator from Its Own Blind Spots

    [https://arxiv.org/abs/2608.18744](https://arxiv.org/abs/2608.18744)

    本文提出EvalCEGAR方法，通过反例引导抽象细化自动演化评估指标，利用碰撞对（正确与错误答案评分相同）作为作者请求，从自身盲点中生成可解释的缺陷检测操作符池，解决了报告生成等场景中自动评分指标缺失的问题。

    

    arXiv:2608.18744v1 公告类型：新 摘要：智能体在可靠自动指标的引导下能快速进步，而没有指标则会停滞不前；最需要这种指标的应用（如报告生成）恰恰是无人知道如何评分的领域。指标能自我编写吗？说清什么使答案优秀很难，但指出答案的问题则相对容易，因此我们演化的指标是一个小型Python操作符池，每个操作符为一个命名的缺陷标记候选答案，或弃权，并投票。直接让模型生成操作符是行不通的：183个候选仅实现96种不同行为，且来自一个巨大空间中的狭窄区域。EvalCEGAR转而借鉴程序验证中的反例引导抽象细化方法。它将操作符池视为一种抽象，并搜索碰撞——即两个答案在操作符评分下相同，但一个正确一个错误。该配对（而非提示）成为创作请求，当碰撞击败所有尝试时，循环会扩大操作符的定义范围。

    arXiv:2608.18744v1 Announce Type: new  Abstract: Agents improve quickly against a reliable automatic metric and stall without one, and the applications that need them most, report generation among them, are the ones nobody knows how to score. Can the metric write itself? Saying what makes an answer good is hard; pointing at something wrong with one is easier, so the metric we evolve is a pool of small Python operators that each flag a candidate for one named defect, or abstain, and vote. Asking a model for operators directly does not work: 183 candidates realise only 96 distinct behaviours, from one narrow region of an enormous space. EvalCEGAR instead borrows counterexample-guided abstraction refinement from program verification. It reads the pool as an abstraction and searches for a collision, two answers the operators score identically, one correct and one not. That pair, not a prompt, is the authoring request, and when a collision defeats every attempt the loop widens what an opera
    
[^150]: 反射守卫：利用密集语义嵌入实现低延迟的大语言模型提示安全护栏

    Reflex-Guard: A Low-Latency Guardrail for LLM Prompt Safety Using Dense Semantic Embeddings

    [https://arxiv.org/abs/2608.17556](https://arxiv.org/abs/2608.17556)

    Reflex-Guard是一种本地运行的轻量级护栏，通过越狱感知预处理、紧凑嵌入和快速分类器，在低于100毫秒的延迟下实现高精度提示安全过滤，同时避免数据隐私风险。

    

    大语言模型（LLMs）在实际应用中经常面临精心设计的提示词试图绕过安全控制的风险。现有的护栏方法，如LLM作为评判者和基于云的安全API，能够检测不安全内容。然而，它们通常会给每个请求增加约250-900毫秒的延迟。对于需要系统在100毫秒内响应的实时应用来说，这种延迟过高。此外，将用户提示路由到外部审核端点会引发严重的数据隐私问题。本文介绍了Reflex-Guard，一种本地运行的轻量级护栏。它采用越狱感知预处理、紧凑的句子变换器嵌入和七个快速二元分类器。这些组件共同实现了高精度的提示安全过滤，且延迟远低于现有解决方案。通过对30,568个策略平衡数据集的系统评估，证明了其有效性。

    arXiv:2608.17556v1 Announce Type: cross  Abstract: Large Language Models (LLMs) in real-world applications often face the risks of specially crafted prompts designed to bypass the safety controls. Existing guardrail methods, such as LLM-as-a-judge and cloud-based safety APIs are able to detect unsafe content. However, they often add a delay of about 250-900 ms to each request. This delay is too high for real-time applications, when the system usually needs to respond in less than 100 ms. Furthermore, routing user prompts through external moderation endpoints raises significant data privacy concerns. This paper introduces Reflex-Guard, a lightweight guardrail that runs locally. It uses jailbreak-aware preprocessing, compact sentence-transformer embeddings, and seven fast binary classifiers. Together, these components enable high-accuracy prompt safety filtering with much lower latency than existing solutions. Through systematic evaluation on a strategically balanced dataset of 30,568 sa
    
[^151]: Q-CueGraph：面向多模态推理的查询条件化视觉证据图谱

    Q-CueGraph: Query-Conditioned Visual Evidence Graphs for Multimodal Reasoning

    [https://arxiv.org/abs/2608.04452](https://arxiv.org/abs/2608.04452)

    提出Q-CueGraph，一种面向冻结多模态大语言模型的查询条件化证据获取框架，通过构建可复用的OCR与版面关系图谱、查询条件化目标检测以及无需证据框监督的轻量级答案性评分器，自适应地激活并组合视觉证据区域，从而提升多模态推理性能。

    

    多模态大语言模型在观察完整图像时可能会遗漏它们在更近距离观察中本可识别的细节。恢复这些证据需要决定往哪里看以及保留多少周围上下文。我们提出了Q-CueGraph，一种面向冻结多模态大语言模型的查询条件化证据获取方法。对于文本丰富的图像，它构建一个可复用的OCR文本行与版面关系图谱；每个问题激活锚点，将其扩展为上下文区域，并选择候选对象构成单个观察窗口。查询条件化的目标检测通过相同的区域选择与组合接口支持自然图像搜索。轻量级候选评分器进一步从冻结读取器的反馈和训练答案中学习哪些观察能够支持正确答案，而无需证据框监督。在六个基准测试中，我们检验了查询条件化、证据组合以及可学习答案性的作用。

    arXiv:2608.04452v2 Announce Type: replace-cross  Abstract: Multimodal large language models (MLLMs) can miss fine details in a full image that they recognize in a closer view. Recovering this evidence requires deciding where to look and how much surrounding context to retain. We present Q-CueGraph, a query-conditioned evidence acquisition method for frozen MLLMs. For text-rich images, it builds a reusable graph of OCR lines and layout relations. Each question activates anchors, expands them into contextual regions, and selects candidates for a single observation window. Query-conditioned object detections support natural-image search through the same region-selection and composition interface. A lightweight candidate scorer further learns which observations support correct answers from frozen-reader feedback and training answers, without evidence-box supervision. Across six benchmarks, we examine the roles of query conditioning, evidence composition, and learned answerability. With Qwe
    
[^152]: 布线优于混合：不同Transformer规模之间传递了什么——以及什么没有传递

    Wiring Beats Blending: What Transfers Between Transformer Sizes -- and What Doesn't

    [https://arxiv.org/abs/2608.02829](https://arxiv.org/abs/2608.02829)

    本文发现，在不同规模的Transformer模型间转换时，表示对齐强但参数对齐弱，价值在于初始化，并通过最小二乘补偿和方差保持重缩放两个杠杆实现有效转换。

    

    arXiv:2608.02829v3 公告类型：替换交叉 摘要：模型家族通常按规模逐个从头训练。能否将预训练的大模型转换为较小的兄弟模型？我们端到端地表征了Pythia中的1.4B->410M转换。表示在不同规模间强烈对齐（岭回归R^2=0.84），而参数对齐较弱。密集权重投影在功能上具有破坏性，且一个比特精确的控制表明这不是组装伪影：基混合破坏了旋转、每头、GELU和LayerNorm结构。在最佳拟合线性算子之后，权重残差在洗牌控制下在统计上与噪声无异。因此，转换价值存在于初始化中。在匹配预算的持续预训练中，我们将转换分解为两个独立杠杆：最小二乘补偿（功能杠杆，最佳零样本）和方差保持重缩放（动力学杠杆，最佳终点）。补偿是一种令牌高效、低预算的胜利，而非其他。

    arXiv:2608.02829v3 Announce Type: replace-cross  Abstract: Model families are typically trained size by size, each from scratch. Can a pretrained large model instead be converted into a smaller sibling? We characterize the 1.4B->410M conversion in Pythia end to end. Representations align strongly across sizes (ridge R^2=0.84) while parameters align weakly. Dense weight projection is functionally destructive, and a bit-exact control shows this is not an assembly artifact: basis mixing breaks rotary, per-head, GELU, and LayerNorm structure. After the best-fit linear operator, weight residuals are statistically indistinguishable from noise under shuffle controls. Conversion value therefore lives in initialization. In matched-budget continued pre-training we decompose conversion into two independent levers: least-squares compensation (function lever, best zero-shot) and variance-preserving rescale (dynamics lever, best endpoints). Compensation is a token-efficient, low-budget win rather th
    
[^153]: Gaokerena：一个小型波斯语医学语言模型家族

    Gaokerena: A Small Persian Medical Language Model Family

    [https://arxiv.org/abs/2608.00932](https://arxiv.org/abs/2608.00932)

    本文提出了Gaokerena，一个专为消费级硬件设计的小型波斯语医学语言模型家族，其中Gaokerena-V通过新构建的波斯语医学语料库训练提升了医学问答性能，Gaokerena-R则结合思维链与两个新型RLAIF框架来增强临床推理能力。

    

    人工智能融入医学问答系统的发展迅速；然而，相关研究仍主要集中在英语上，导致波斯语等低资源语言的服务严重不足。为填补这一空白，本文提出了Gaokerena，这是一个新型的小型波斯语医学语言模型家族，专为在消费级硬件上部署而优化。作为迈向本地化数字医疗的基础步骤，我们首先介绍了Gaokerena-V，它是通过在一个新构建的9000万词元波斯语医学语料库和2万个经专家审核的医生问答对上训练基线模型而开发的，其在翻译版医学MMLU基准上的性能从46.28%提升至49.31%。其次，考虑到临床推理的关键需求，我们通过将思维链方法与两个新颖的AI反馈强化学习（RLAIF）框架相结合，开发了Gaokerena-R，以优化偏好……

    arXiv:2608.00932v2 Announce Type: replace  Abstract: The integration of artificial intelligence into medical question-answering systems has advanced rapidly; however, research remains predominantly focused on English, leaving low resource languages like Persian significantly underserved. To address this gap, this paper introduces Gaokerena, a novel family of compact Persian medical language models optimized for deployment on consumer grade hardware. As a foundational step toward localized digital healthcare, we first present Gaokerena-V, developed by training a baseline model on a newly curated 90-million-token Persian medical corpus and 20,000 expert-vetted physician Q&A pairs, which improved performance on a translated medical MMLU benchmark from 46.28% to 49.31%. Second, recognizing the critical demands of clinical reasoning, we developed Gaokerena-R by integrating a Chain-of-Thought approach with two novel Reinforcement Learning with AI Feedback (RLAIF) frameworks to optimize prefe
    
[^154]: CONSISTRE：一种基于大语言模型的统一一致性感知文档级关系抽取框架

    CONSISTRE: A Unified Consistency-Aware Framework for Document-Level Relation Extraction with Large Language Models

    [https://arxiv.org/abs/2607.24312](https://arxiv.org/abs/2607.24312)

    CONSISTRE提出一个统一的一致性感知框架，通过面向黑盒大语言模型的约束感知提示、约束验证与迭代自我反思，以及向较小开源模型注入一致性知识这两条互补路径，解决文档级关系抽取中预测违反传递性、对称性等关系约束而产生矛盾输出的问题。

    

    文档级关系抽取（DocRE）旨在从长篇上下文中抽取多个实体之间的关系，同时保持预测三元组之间的一致性。尽管大语言模型（LLMs）在信息抽取中展现出卓越的推理能力，但其预测通常是针对每个候选三元组独立生成的，可能违反传递性、对称性和函数唯一性等基本关系约束，从而导致矛盾且不可靠的输出。我们提出了CONSISTRE，一个面向文档级关系抽取的统一一致性感知框架，通过两条互补的路径来解决这一局限：第一条路径在推理阶段面向黑盒大语言模型，结合约束感知提示、基于约束的验证和迭代自我反思来改进预测，无需任务特定的微调；第二条路径则通过知识（原文摘要在此处截断）将一致性知识注入到较小的开源模型中。

    arXiv:2607.24312v2 Announce Type: replace  Abstract: Document-level relation extraction (DocRE) aims to extract relations among multiple entities across extended contexts while maintaining consistency across predicted triples. Although large language models (LLMs) show remarkable reasoning capabilities in information extraction, their predictions are typically generated independently for each candidate triple and may violate fundamental relational constraints such as transitivity, symmetry, and functional uniqueness, leading to contradictory and unreliable outputs. We propose CONSISTRE, a unified consistency-aware framework for DocRE that addresses this limitation through two complementary tracks. The first operates at inference time for black-box LLMs, combining constraint-aware prompting, constraint-based verification, and iterative self-reflection to refine predictions without task-specific fine-tuning. The second injects consistency knowledge into smaller open-source models via a k
    
[^155]: 通过受控扰动验证的结构化音频字幕评估框架

    An Evaluation Framework for Structured Audio Captions Validated by Controlled Perturbations

    [https://arxiv.org/abs/2607.21424](https://arxiv.org/abs/2607.21424)

    提出了一个涵盖标签集、描述、推理、数值测量和频谱特征五个维度的结构化音频字幕评估框架，结合LLM评判器与确定性指标，并通过受控扰动验证了各指标的有效性。

    

    自动音频字幕生成（AAC）的最新进展正在推动从单一整句向结构化格式的转变，这种格式能够解耦声学与语义属性，例如针对不同声音事件的时间戳字幕。这类表示可以为创作者提供多维度声音搜索支持，并为聋人和听障人士提供更丰富的听觉信息获取途径。然而，如何有意义地评估这些混合的结构化字幕仍不明确。我们提出了一个针对结构化音频描述的评估框架，涵盖五个互补维度：标签集、描述、推理、数值测量和频谱特征。该框架将用于语义字段的大语言模型（LLM）评判器与用于时间和声学属性的确定性指标相结合。为验证这些指标，我们引入了受控扰动方法，对真实标注施加有类型、分等级的更改。结果表明，所提出的（摘要在此处截断）……

    arXiv:2607.21424v2 Announce Type: replace  Abstract: Recent advances in automated audio captioning (AAC) are driving a shift from monolithic sentences toward structured formats that disentangle acoustic and semantic properties, such as timestamped captions for different sound events. Such representations can support faceted sound search for creators and richer access to auditory information for Deaf and Hard of Hearing people. Yet, it remains unclear how to meaningfully evaluate these hybrid, structured captions. We propose an evaluation framework for structured audio descriptions, spanning five complementary axes: tag sets, descriptions, reasoning, numeric measurements, and spectral profiles. The framework combines large language model (LLM) judges for semantic fields with deterministic metrics for temporal and acoustic attributes. To validate these metrics, we introduce controlled perturbations that apply typed, graded changes to ground-truth annotations. Results show that the propos
    
[^156]: 一种针对KV缓存的JoLT方法：通过Tucker秩的联合拉格朗日分配和旋转残差实现大语言模型的近无损KV缓存压缩

    A JoLT for the KV cache: Near-lossless KV cache compression via joint Lagrangian allocation of Tucker ranks and a rotated residual for llms

    [https://arxiv.org/abs/2607.12550](https://arxiv.org/abs/2607.12550)

    本文提出JoLT方法，通过部分Tucker分解和旋转低比特残差，在保持头与层轴完整的同时压缩令牌和特征轴，实现KV缓存的近无损压缩。

    

    键值（KV）缓存已成为Transformer推理中的主要内存开销：它随批次大小、上下文长度和深度增长，在长上下文场景下，它而非模型权重决定了吞吐量的上限。现有的压缩方法分为两类。低秩方法对缓存的二维切片进行分解，可以是每个头的矩阵或跨层的特征块，而量化方法则降低每个条目的位宽。这两类方法都没有利用缓存在一层中天然是三阶张量这一事实，其三个轴——头、令牌和特征——携带的冗余量差异很大。我们直接采用这种张量视图。我们的方法JoLT（联合拉格朗日Tucker）应用部分Tucker分解，仅压缩令牌和特征轴，同时保留头和层轴不变，然后通过旋转的低位残差恢复截断所丢弃的能量：一个随机或...

    arXiv:2607.12550v3 Announce Type: replace-cross  Abstract: The key-value (KV) cache has become the dominant memory cost of transformer inference: it grows with batch size, context length, and depth, and at long context it, rather than the model weights, sets the throughput ceiling. Existing reductions fall into two families. Low-rank methods factor two-dimensional slices of the cache, either per-head matrices or cross-layer feature blocks, and quantization methods lower the bit-width of every entry. Neither exploits the fact that the cache at a layer is naturally a third-order tensor whose three axes, the heads, the tokens, and the features, carry very different amounts of redundancy. We take this tensor view directly. Our method, JoLT (Joint Lagrangian Tucker), applies a partial Tucker decomposition that compresses only the token and feature axes while leaving the head and layer axes intact, then restores the energy that truncation discards with a rotated low-bit residual: a random or
    
[^157]: 缩小低资源文本转语音的质量差距：针对高棉语和韩语的VoxCPM2 LoRA微调

    Closing the Quality Gap in Low-Resource Text-to-Speech: LoRA Fine-Tuning of VoxCPM2 for Khmer and Korean

    [https://arxiv.org/abs/2606.26618](https://arxiv.org/abs/2606.26618)

    通过零初始化的LoRA适配器同时微调两种低资源语言（高棉语和韩语）的VoxCPM2模型，仅训练少量参数即可显著提升语音质量，使高棉语MOS得分从3.85提升至4.23。

    

    arXiv:2606.26618v1 公告类型：新 摘要：大型预训练文本转语音（TTS）模型在资源丰富的语言上听起来几乎与人类无异，但在训练数据中罕见的语言上表现要差得多。我们使用VoxCPM2研究高棉语和韩语的质量差距，这是一个24亿参数、无分词器的TTS模型，它将MiniCPM-4语言模型骨干与流匹配扩散解码器相结合。我们构建了一个约26小时的共享、带语言标签的语料库，并使用单个低秩适配器（LoRA）适配VoxCPM2，该适配器同时在两种语言上训练，并添加到语言模型和解码器中。适配器采用零初始化，因此训练完全从原始（零样本）模型开始。在母语者听力测试中，高棉语的平均意见得分（MOS）从3.85提升到最佳适配器（秩64）时的4.23，这是非常显著的提升（配对Wilcoxon检验，p<0.001），同时仅训练了0.19%到3.03%的参数。自动损失和人类评分结果一致。

    arXiv:2606.26618v1 Announce Type: new  Abstract: Large pretrained text-to-speech (TTS) models sound almost human for well-resourced languages, but much worse for languages that are rare in their training data. We study this quality gap for Khmer and Korean using VoxCPM2, a 2.4B-parameter, tokenizer-free TTS model that joins a MiniCPM-4 language-model backbone with a flow-matching diffusion decoder. We build one shared, language-tagged corpus of about 26 hours and adapt VoxCPM2 with a single Low-Rank Adaptation (LoRA) adapter, trained on both languages at once and added to both the language model and the decoder. The adapter is zero-initialized, so training starts exactly at the original (zero-shot) model. In native-speaker listening tests, the Khmer Mean Opinion Score (MOS) rises from 3.85 to 4.23 with the best adapter (rank 64), a highly significant gain (paired Wilcoxon test, p<0.001), while training only 0.19 to 3.03 percent of the parameters. The automatic loss and the human rating
    
[^158]: 谁拥有AI的推荐？跨大语言模型品牌类别归属的多行业实证图谱

    Who Owns the AI Recommendation? A Multi-Industry Empirical Map of Brand Category Ownership Across Large Language Models

    [https://arxiv.org/abs/2606.23057](https://arxiv.org/abs/2606.23057)

    该研究通过对五个行业50个品牌、250个查询在三个大语言模型上跨越两个月的大规模实证测量，首次绘制了AI推荐中品牌类别归属的图谱，发现品牌收录率较为均衡、推荐份额随时间高度稳定，并识别出7.6%的“竞争真空”查询。

    

    这项探索性研究测量了五个行业、50个品牌、250个查询中的品牌收录情况，于2026年2月和9月各将每个查询五次提交给GPT-5.2、Gemini 3 Flash和Perplexity sonar-pro（分别获得3,614和3,750条评分答案）。类别收录率、推荐份额、竞争真空指数和共同提及不对称性等指标均有明确的分母定义。2月份同一行业内被采样品牌的收录率较为接近（平均基尼系数为0.30），而在250个查询中有204个查询的答案中至少80%提及了至少一个品牌。竞争真空出现在7.6%的查询中；初步的开放词汇模型解读表明，大多数真空现象反映了所采样的品牌列表本身。部分预先设定的9月复制实验显示，跨日期的推荐份额具有很强的相关性（Spearman 0.994），真空现象的普遍程度保持不变，一致性为60.8%（2月为57.2%）。这种描述性的规模关联持续存在。固定边际并不能解释……

    arXiv:2606.23057v2 Announce Type: replace-cross  Abstract: This exploratory study measures brand inclusion across five industries, 50 brands and 250 queries, each put five times to GPT-5.2, Gemini 3 Flash and Perplexity sonar-pro in February and September 2026 (3,614 and 3,750 scored answers). Category Inclusion Rate, Recommendation Share, Competitive Vacuum Index and Co-Mention Asymmetry have stated denominators. February inclusion rates sit close together across an industry's sampled brands (mean Gini 0.30), while at least one brand is named in 80% or more of answers to 204 of 250 queries. Vacuums occur in 7.6% of queries; provisional open-vocabulary model readings suggest most reflect the sampled brand list. The partially pre-specified September replication shows strong cross-date Recommendation Share correlation (Spearman 0.994), unchanged vacuum prevalence and agreement of 60.8% against February's 57.2%. The descriptive size association persists. Fixed margins do not account for a
    
[^159]: 在无监督词条发现中恢复齐夫分布

    Recovering the Zipfian Distribution in Unsupervised Term Discovery

    [https://arxiv.org/abs/2606.10781](https://arxiv.org/abs/2606.10781)

    该论文提出用基于图的Leiden聚类替代K-means等基于中心的方法，在无监督词条发现中显著恢复了真实词库所具有的齐夫分布特性。

    

    无监督词条发现是指将无标注语音切分为类似词或音节的单元，并将这些单元聚类为一个候选词条类型的词库。真实的词库遵循齐夫分布，然而主流的基于中心的聚类方法——K-means——由于对球形簇的归纳偏置，会产生更均匀的分布。在本文中，我们重新审视基于图的聚类作为一种自底向上的替代方案，该方法通过成对相似度连接分段嵌入，并使用Leiden算法进行划分。我们表明，在三种语言的词级和音节级词库发现任务中，图聚类显著优于基于中心的方法，并产生更接近齐夫分布的结果。另一种自底向上的方法——采用平均链接的层次聚类——也表现良好，尽管其计算效率较低，且对最终结果的控制能力较弱。

    arXiv:2606.10781v2 Announce Type: replace-cross  Abstract: Unsupervised term discovery involves segmenting unlabelled speech into word- or syllable-like units and clustering these into a lexicon of candidate types. True lexicons follow a Zipfian distribution, yet the dominant centre-based clustering approach -- K-means -- produces a more uniform distribution due to an inductive bias toward spherical clusters. In this paper we revisit graph-based clustering as a bottom-up alternative, where segment embeddings are connected by pairwise similarity and partitioned using the Leiden algorithm. We show that graph clustering substantially outperforms centre-based approaches (K-means, GMM, BIRCH) in both word- and syllable-level lexicon discovery across three languages, producing more Zipf-like distributions. Another bottom-up approach, agglomerative clustering with average linkage, also performs well, although it is computationally less efficient and allows for less control over the resulting 
    
[^160]: 多模态城市感知中的人格提示：描述收敛与解读差异

    Persona Prompting in Multimodal Urban Perception: Descriptive Convergence and Interpretive Variation

    [https://arxiv.org/abs/2605.29064](https://arxiv.org/abs/2605.29064)

    本研究通过约12万条人格化标注发现，多模态大语言模型在城市感知中生成的客观图像描述几乎不随人格提示改变，但其主观解读却随人格显著变化，其中经济身份的影响最大。

    

    本研究考察人格提示如何塑造两个多模态大语言模型在城市感知任务中所生成的语言，城市感知为检验模型对共享视觉证据的主观解读提供了场景。研究者将模型输出组织为三个功能层次：描述性基础层（图像说明）、中间语义层（感知标签）和解读性框架层（理由说明）。基于来自 Qwen3-VL 和 Gemma4 两个多模态大语言模型各约 60,000 条以人格为条件的标注，研究发现图像说明在不同人格设定之间高度收敛，仅表现出与属性相关的微小差异；而理由说明的变异则大得多：经济地位在两个模型中都引起最大的差异，政治倾向和人格特质的影响也很突出。成对的图像级比较证实，对于这三种属性，理由说明的差异确实大于图像说明的差异。对于感知标签，共享相同属性水平的人格产生……（原文在此处截断）

    arXiv:2605.29064v3 Announce Type: replace  Abstract: This study examines how persona prompting shapes language generated by two multimodal large language models in urban perception, a setting for examining subjective interpretations of shared visual evidence. We organize outputs into three functional layers: descriptive grounding (captions), intermediate semantic layer (perception tags), and interpretive framing (justifications). Using approximately 60,000 persona-conditioned annotations from each of two MLLMs, Qwen3-VL and Gemma4, we find that captions converge strongly across persona profiles and show only small attribute-associated differences. Justifications vary substantially more: economic status produces the largest difference in both models, with political orientation and personality also prominent. Paired image-level comparisons confirm larger justification than caption differences for these three attributes. For perception tags, personas sharing the same attribute level produ
    
[^161]: 当搜索成为记忆：利用自进化技能加速机器人设计发现

    When Search Becomes Memory: Accelerating Robot Design Discovery with Self-Evolving Skills

    [https://arxiv.org/abs/2605.25832](https://arxiv.org/abs/2605.25832)

    提出Auto-Robotist，一个自进化的LLM智能体，通过将进化搜索轨迹提炼为可检查的自然语言技能库，把搜索结果转化为可重用的设计记忆，从而加速机器人形态设计的发现。

    

    大语言模型（LLM）正越来越多地被用作进化机器人设计的方案生成器，然而大多数循环仍是无记忆的：模拟器结果虽能塑造下一代种群，却未被保存为可重用的设计知识。我们提出Auto-Robotist，一个自进化的LLM智能体，它将形态搜索轨迹提炼为显式的自然语言技能库。每个技能存储一个结构原型、有证据支持的正面与负面规则，以及支持这些规则的已评估设计，使设计记忆变得可检查，而非隐含在种群之中。在搜索过程中，该智能体检索技能以引导LLM对精英个体进行编辑，同时保留遗传算法（GA）变异路径用于探索；评估完成后，它通过添加、诊断和合并操作来更新技能库。在涵盖运动、穿越和物体交互的七个EvoGym任务上，Auto-Robotist提升了冷启动5×5搜索的性能……

    arXiv:2605.25832v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly used as proposal generators for evolutionary robot design, yet most loops remain memoryless: simulator results shape the next population but are not preserved as reusable design knowledge. We present Auto-Robotist, a self-evolving LLM agent that distills morphology-search traces into an explicit natural-language skill library. Each skill stores a structural archetype, evidence-grounded positive and negative rules, and the evaluated designs that support them, making design memory inspectable rather than implicit in a population. During search, the agent retrieves skills to condition LLM edits of elite bodies while retaining a Genetic Algorithm (GA) mutation path for exploration; after evaluation, it updates the library through Add, Diagnose, and Merge. Across seven EvoGym tasks spanning locomotion, traversal, and object interaction, Auto-Robotist improves cold-start 5x5 search and tr
    
[^162]: 主动应对不确定性：面向口语对话系统的因果感知错误诊断与交互式澄清

    Proactive for Uncertainty: Cause-Aware Error Diagnosis and Interactive Clarification for Spoken Dialogue Systems

    [https://arxiv.org/abs/2605.25404](https://arxiv.org/abs/2605.25404)

    该论文提出因果感知的错误恢复范式，利用一组小型高精度检测器诊断ASR错误成因（声学或语言失配）并通过交互式澄清进行主动恢复，克服了传统置信度过滤无法检测删除错误和区分错误类型的局限。

    

    级联式自动语音识别-大语言模型（ASR-LLM）流水线在工业口语对话系统（SDS）中依然流行，主要因为其解耦设计确保了感知可验证性。然而，级联系统存在错误传播问题：转写失败会不可避免地向后续组件级联，从而降低最终交互质量。尽管ASR置信度分数提供了一种过滤不可靠输入的简单方法，但该方法存在根本性局限：它通常无法检测删除类错误，也无法区分声学层面（听不清）与语言层面（听懂却无法理解）的失配，而这两类问题都需要有针对性的恢复策略。在本文中，我们提出了一种因果感知的错误恢复范式，从根本上重新思考了SDS的鲁棒性。与传统的置信度过滤不同，我们引入了一组小型、注重精度的检测器……

    arXiv:2605.25404v2 Announce Type: replace  Abstract: Cascaded Automatic Speech Recognition - Large Language Model (ASR-LLM) pipelines remain popular for industrial Spoken Dialogue Systems (SDS), primarily because their decoupled design ensures perceptual verifiability. However, cascaded systems suffer from error propagation, as transcription failures inevitably cascade to subsequent components, thereby degrading the final interaction quality. Although ASR confidence scores offer a simple filter for unreliable inputs, this approach is fundamentally limited because it typically fails to detect deletion errors or to distinguish between acoustic (inability to hear clearly) and linguistic (inability to understand) mismatches, both of which require targeted recovery strategies. In this paper, we propose a cause-aware error recovery paradigm that fundamentally rethinks robustness in SDS. Unlike traditional confidence filtering, we introduce a suite of small precision-focused detectors that ex
    
[^163]: 自动化智能体评估的实证研究

    An Empirical Study of Automating Agent Evaluation

    [https://arxiv.org/abs/2605.11378](https://arxiv.org/abs/2605.11378)

    仅靠提示前沿编程助手无法可靠地自动化智能体评估，本文提出EvalAgent，通过将评估领域专业知识编码为可组合的评估技能，实现了端到端的智能体评估自动化。

    

    智能体评估需要评估涉及工具使用和中间推理的复杂多步骤行为，这使得评估成本高昂且高度依赖专业知识。一个自然的问题随之而来：前沿编程助手能否可靠地自动化这一评估过程？我们的研究表明，仅通过提示编程助手不足以完成这项任务。在缺乏领域特定评估知识的情况下，前沿编程助手的执行成功率仅为30%，并且会产生过度工程化的评估，每个智能体平均包含超过12个评估指标，这表明强大的编码能力并不会自动转化为可靠的智能体评估能力。我们提出了EvalAgent，一个能够自动化端到端智能体评估流程的AI助手。EvalAgent将评估领域专业知识编码为评估技能（包括程序化指令、可复用的代码和模板，以及动态检索的API文档），这些技能组合成一个基于轨迹（trace）的评估流程。

    arXiv:2605.11378v3 Announce Type: replace  Abstract: Agent evaluation requires assessing complex multi-step behaviors involving tool use and intermediate reasoning, making it costly and expertise-intensive. A natural question arises: can frontier coding assistants reliably automate this evaluation process? Our study shows that simply prompting coding assistants is insufficient for this task. Without domain-specific evaluation knowledge, frontier coding assistants achieve only a 30% execution success rate and produce over-engineered evaluations averaging 12+ metrics per agent, indicating that strong coding ability does not automatically translate to reliable agent evaluation. We introduce EvalAgent, an AI assistant that automates the end-to-end agent evaluation pipeline. EvalAgent encodes evaluation domain expertise as evaluation skills (procedural instructions, reusable code and templates, and dynamically retrieved API documentation) that compose into a trace-based pipeline producing c
    
[^164]: 前沿滞后：学术AI评估中能力误述的文献计量审计

    Frontier Lag: A Bibliometric Audit of Capability Misrepresentation in Academic AI Evaluation

    [https://arxiv.org/abs/2605.04135](https://arxiv.org/abs/2605.04135)

    该研究对超过11万篇文献进行系统计量分析后发现，学术论文中评估的LLM能力落后于当时的前沿模型（中位数差距+10.85 ECI），且这一“前沿滞后”差距正以每年+5.53 ECI的速度持续扩大。

    

    应用领域中的LLM评估往往反映的是在论文发表时就已经被超越的模型。我们观察到一种“发表引导差距”（publication elicitation gap）：即学术论文中所报告结果由哪些AI系统生成，与当前读者合理认为论文所引用的AI系统之间的距离。我们系统性地扫描了OpenAlex中2022年1月1日至2026年4月1日的数据（n = 112,303个LLM关键词匹配），随后识别出被评估的模型（n = 18,574条有效记录）。接着，我们基于Epoch AI能力指数（ECI）——一个LLM综合能力评分——将每个被评估的LLM与前沿LLM进行排名比较。在评估时点，中位数论文所评估的模型在能力上落后于前沿LLM，中位数差距为+10.85 ECI（H1；n = 12,312）。这一差距正在扩大，以每年+5.53 ECI的速度增长（H2，名义95%置信区间 [+5.03, +5.83]）。即使在……（原文摘要在此处截断）

    arXiv:2605.04135v3 Announce Type: replace-cross  Abstract: LLM evaluations in applied domains tend to reflect models that were already outclassed at time of publication. We observe a publication elicitation gap: the distance between the AI systems generating the results reported in an academic paper and the AI systems that a current reader of that paper would reasonably assume are being referenced. We systematically sweep OpenAlex from 2022-01-01 to 2026-04-01 (n = 112,303 LLM keyword matches). Then, we identify what models were evaluated (n = 18,574 admissible records). We then rank each evaluated LLM against a frontier LLM based on the Epoch AI Capabilities Index (ECI), an aggregate LLM capability score. At time of evaluation, the median paper is evaluating models that are behind frontier LLMs in capability, with a median gap of +10.85 ECI (H1; n = 12,312). This gap is growing, increasing at a rate of +5.53 ECI per year (H2, nominal 95% CI [+5.03, +5.83]). The sign holds even in the 
    
[^165]: 在芬兰语组织病理学报告上对FinBERT进行持续预训练：训练时信号与代理下游任务相关性

    Continued Pretraining of FinBERT on Finnish Histopathological Reports: Train-Time Signals and Proxy Downstream Correlations

    [https://arxiv.org/abs/2604.14815](https://arxiv.org/abs/2604.14815)

    本文将在芬兰语组织病理学报告上持续预训练FinBERT，发现CPT训练时损失曲线具有明显的领域差异，且某些CPT衍生特征与代理下游分类性能提升相关，为芬兰医疗NLP这一文献稀缺的领域做出了贡献。

    

    在缺乏标注数据的自然语言处理（NLP）分类任务中，在无标注数据上对transformer模型进行持续预训练（CPT）是一种成熟的方法。本文有两个目标：（1）我们描述了在芬兰语组织病理学数据集（下称“组织病理学数据”）上对芬兰语BERT transformer模型（FinBERT）进行持续预训练所获得的观察结果。（2）由于组织病理学数据没有分类标签，我们收集了公开的芬兰语数据集作为代理数据，以分析（1）中观察到的信号是否与下游分类性能提升相关。我们观察到，CPT的训练时损失曲线因领域不同而存在显著差异，并且在探索性分析中，某些CPT衍生特征与代理分类性能的提升相关。特别地，本报告为芬兰医疗数据NLP领域较为有限的文献做出了贡献。

    arXiv:2604.14815v2 Announce Type: replace  Abstract: In Natural Language Processing (NLP) classification tasks where a lack of labeled data is an issue, continued pretraining (CPT) of transformer models on unlabeled data is an established approach. In this paper, we have two aims.   (1) We describe our observations from continued pretraining of the Finnish BERT transformer model (FinBERT) on a Finnish histopathological dataset (below, \emph{the Histopathology data}).   (2) Since the Histopathology data has no classification labels, we gather public Finnish datasets as proxy data to analyze whether the signals observed in (1) are associated with downstream classification gains.   We observe that CPT train-time loss curves differ strongly by domain, and that, in an exploratory analysis, certain CPT-derived features correlate with proxy classification improvement. In particular, this report contributes to the limited literature on NLP for Finnish healthcare data.
    
[^166]: 预测正确，步骤有误？用于鲁棒思维链合成的共识推理知识图谱

    Correct Prediction, Wrong Steps? Consensus Reasoning Knowledge Graph for Robust Chain-of-Thought Synthesis

    [https://arxiv.org/abs/2604.14121](https://arxiv.org/abs/2604.14121)

    提出CRAFT方法，通过聚合多个候选推理轨迹的共识组件构建推理知识图谱，从推理结构层面修复LLM“答案正确但推理步骤有缺陷”的问题，实现更鲁棒的思维链合成。

    

    大语言模型（LLM）在各类任务中的应用日益广泛，通常还会结合思维链（CoT）提示来提升准确性。近期研究表明，高标签预测准确率并不能保证中间推理过程的正确性，且推理缺陷的成因因样本而异，然而现有的补救措施要么只针对单一领域，要么假设某一种缺陷类型统一适用于所有样本。一种简单的缓解方法是直接给模型提供正确答案，但我们发现这并不能带来推理质量的持续改善。这表明该问题无法通过LLM对答案的感知来解决，而必须从推理的结构层面加以解决。受此启发，我们提出了CRAFT（面向缺陷感知轨迹合成的共识推理知识图谱聚合方法），该方法聚合多个候选推理轨迹间共享的共识组件……

    arXiv:2604.14121v3 Announce Type: replace  Abstract: Large language models (LLMs) have become increasingly used for various tasks, often coupled with Chain-of-Thought (CoT) prompting to boost accuracy. Recent work has shown that high label-prediction accuracy does not guarantee correct intermediate reasoning, and the causes of *reasoning flaws* vary from sample to sample, yet existing remedies either focus on a single domain or assume that one flaw type applies uniformly across samples. A simple mitigation method is to provide the model with the correct answer, but we show that this yields no consistent improvement in reasoning quality. This indicates that the problem cannot be fixed by LLMs' awareness of answers, and must instead be addressed through the *structure* of reasoning. Motivated by this, we propose CRAFT (Consensus Reasoning-knowledge-graph Aggregation for Flaw-aware Trace synthesis), which aggregates the consensus components shared across multiple candidate reasoning trace
    
[^167]: IatroBench：一个针对语言模型临床信息遗漏的预注册基准测试

    IatroBench: A Pre-Registered Benchmark of Clinical Omission in Language Models

    [https://arxiv.org/abs/2604.07709](https://arxiv.org/abs/2604.07709)

    论文提出预注册基准IatroBench，从“作为”与“遗漏”两个伤害维度评估语言模型在临床场景中的安全性，并首次揭示了模型对同一病例会向医生提供比患者更多临床信息的“框架依赖性信息保留”现象。

    

    一个经过强安全训练的模型会为医生提供苯二氮䓬类药物的减量停药方案，却不会为提出同样请求的患者提供。模型本身知道这些信息，但分享多少取决于提问的框架。我们提出了IatroBench，一个在两类伤害维度（作为性伤害与遗漏性伤害）上，通过60个预注册临床场景对6个模型进行评估的基准测试。我们使用Claude Opus 4.6依据一位医生撰写的评分细则对模型回复进行打分，发现其遗漏评分与该医生评分的一致性程度，与另一位医生评分之间的一致性相当。我们发现，当同一病例分别以患者提问和医生会诊两种形式呈现时（两种变体在语体、请求方式以及隐含的治疗医生监督方面也存在差异），我们测试的全部五个模型向医生分享的信息都多于向患者分享的信息。我们将这种现象称为“框架依赖性信息保留”。我们发现平均解耦差距为+0.38……

    arXiv:2604.07709v5 Announce Type: replace  Abstract: A strongly safety-trained model will provide a doctor with a benzodiazepine taper schedule, but not a patient who asks for one. The model knows the information, but how much it shares depends on the framing. We introduce IatroBench, a benchmark that evaluates models on two axes of harm (commission and omission) across 60 pre-registered clinical scenarios and 6 models. We use Claude Opus 4.6 to score model responses against a rubric written by a physician, and find that its omission scores are as well-aligned to the physician's scores as another physician's scores are. We find that when the same case is presented as a patient query and a doctor consultation (the variants also differ in register, request and the supervision a treating physician implies), all five models we test share more information with the doctor than the patient. We term this phenomenon "framing-contingent withholding." We find a mean decoupling gap of +0.38 across
    
[^168]: LiveMathematicianBench：一个基于证明概要的研究级数学推理动态基准测试

    LiveMathematicianBench: A Live Benchmark for Research-Level Mathematical Reasoning with Proof Sketches

    [https://arxiv.org/abs/2604.01754](https://arxiv.org/abs/2604.01754)

    提出了LiveMathematicianBench，一个基于训练截止日期后新发表arXiv论文构建的动态研究级数学推理基准测试，通过引入十三类定理逻辑分类体系和证明概要实现细粒度评估，有效避免了数据污染问题。

    

    数学推理是人类智力的标志，大型语言模型（LLM）能否有意义地进行数学推理仍然是人工智能和认知科学中的一个核心问题。随着大语言模型越来越多地被整合到科学工作流程中，对其数学能力进行严格评估已成为一种实际需求。现有的基准测试受限于合成环境和数据污染。我们提出了LiveMathematicianBench，这是一个基于模型训练截止日期之后发布的最新arXiv论文构建的、面向研究级数学推理的动态选择题基准测试。通过将评估建立在新发表的定理之上，它提供了一个超越记忆模式的真实测试平台。该基准测试引入了一个包含十三个类别的定理类型逻辑分类体系（例如蕴含、等价、存在性、唯一性），从而实现跨推理形式的细粒度评估。它采用了一种证明——

    arXiv:2604.01754v2 Announce Type: replace-cross  Abstract: Mathematical reasoning is a hallmark of human intelligence, and whether large language models (LLMs) can meaningfully perform it remains a central question in artificial intelligence and cognitive science. As LLMs are increasingly integrated into scientific workflows, rigorous evaluation of their mathematical capabilities becomes a practical necessity. Existing benchmarks are limited by synthetic settings and data contamination. We present LiveMathematicianBench, a dynamic multiple-choice benchmark for research-level mathematical reasoning built from recent arXiv papers published after model training cutoffs. By grounding evaluation in newly published theorems, it provides a realistic testbed beyond memorized patterns. The benchmark introduces a thirteen-category logical taxonomy of theorem types (e.g., implication, equivalence, existence, uniqueness), enabling fine-grained evaluation across reasoning forms. It employs a proof-
    
[^169]: 可逆查询-键耦合与注意力机制的组合

    Invertible Query-Key Coupling Composes with Attention Mechanisms

    [https://arxiv.org/abs/2604.01683](https://arxiv.org/abs/2604.01683)

    提出一种可逆的查询-键耦合变换（RealNVP风格的交替仿射映射），可无损地叠加在现有注意力机制之上，以极少的额外参数和不变的整体架构显著提升差分注意力等方法的语言建模性能。

    

    arXiv:2604.01683v2 公告类型：replace-cross 摘要：缩放点积注意力将查询和键构建为相互独立的线性投影，因此两者在计算评分的点积之前从不发生交互。我们研究了耦合的查询-键动力学，这是一种评分前的变换，它在标准评分之前通过共享的可逆耦合使每个token的查询和键共同演化。我们将其实现为实非体积保持流（RealNVP）风格的交替仿射映射：该耦合在初始化时为恒等映射，每个注意力头仅增加少量参数，并保持softmax及周围架构不变。我们将耦合叠加在现有注意力方法之上而非替换它们，并探究这种组合是否有所帮助。在WikiText-103上，为差分注意力添加耦合后在150M和455M两种参数规模上均带来了改进。在455M参数下，该增益在序列长度512处具有统计显著性（p=0.003，六个随机种子），通过了Bonferroni校正并成功复现……

    arXiv:2604.01683v2 Announce Type: replace-cross  Abstract: Scaled dot-product attention forms its queries and keys as independent linear projections, so the two never interact before the dot product that scores them. We study coupled query-key dynamics, a pre-scoring transformation that evolves each token's query and key jointly through a shared invertible coupling before standard scoring. We realize it as an alternating affine map in the style of real non-volume-preserving flows: the coupling is the identity at initialization, adds a small fraction of parameters per head, and leaves the softmax and surrounding architecture unchanged. We place coupling on top of existing attention methods rather than replacing them, and ask whether that composition helps. On WikiText-103, adding coupling to Differential Attention improves on it at both 150M and 455M parameters. At 455M the gain is significant at sequence length 512 (p=0.003, six seeds), survives a Bonferroni correction and replicates o
    
[^170]: DiscoPhon：基于离散语音单元的无监督音位库发现基准测试

    DiscoPhon: Benchmarking the Unsupervised Discovery of Phoneme Inventories With Discrete Speech Units

    [https://arxiv.org/abs/2603.18612](https://arxiv.org/abs/2603.18612)

    该论文提出了DiscoPhon，一个基于离散语音单元评估无监督音位发现的多语言基准测试，通过12种语言的评测和四个预训练基线模型揭示了当前语音模型中音位信息的可用性及其跨语言差异。

    

    我们介绍了DiscoPhon，一个用于评估从离散语音单元中进行无监督音位发现的多语言基准测试。DiscoPhon涵盖6种开发语言和6种测试语言，这些语言的选择覆盖了广泛的音位对比类型。在仅给定10小时先前未见语言的语音的情况下，系统必须生成离散单元，并通过多对一或一对一的映射方式将其映射到预定义的音位库。生成的序列将从单元质量、识别和分割三个方面进行评估。我们提供了四个预训练的多语言HuBERT和SpidR基线模型，并表明当前模型中的音位信息已足够充分，使得推导出的单元能够与音位良好对应，但在不同语言之间存在差异。

    arXiv:2603.18612v2 Announce Type: replace  Abstract: We introduce DiscoPhon, a multilingual benchmark for evaluating unsupervised phoneme discovery from discrete speech units. DiscoPhon covers 6 dev and 6 test languages, chosen to span a wide range of phonemic contrasts. Given only 10 hours of speech in a previously unseen language, systems must produce discrete units that are mapped to a predefined phoneme inventory, through either a many-to-one or a one-to-one assignment. The resulting sequences are evaluated for unit quality, recognition and segmentation. We provide four pretrained multilingual HuBERT and SpidR baselines, and show that phonemic information is available enough in current models for derived units to correlate well with phonemes, though with variations across languages.
    
[^171]: 脚手架之下的安全性：评估条件如何塑造所测得的安全表现

    Safety Under Scaffolding: How Evaluation Conditions Shape Measured Safety

    [https://arxiv.org/abs/2603.10044](https://arxiv.org/abs/2603.10044)

    评测条件对测得的模型安全性影响超过脚手架本身——在相同的基准题目上，选择题与开放式格式会使测得的安全性相差5-20个百分点，说明评测结果更多取决于测量方法而非模型潜在的 safety 能力。

    

    安全基准测试通常针对“裸”模型——即接收提示并输出响应的模型——进行，但现实世界的部署会将这些模型“包裹”在复杂的脚手架中。这些脚手架对基准测试所衡量的模型安全性究竟有多大影响？我们在四个预先注册的安全基准上，使用直接 API 以及三种脚手架（ReAct、多智能体和 map-reduce）测试了六个领先模型，共进行了 62,808 次评分评估。研究发现，安全性的测量方式比脚手架本身更为重要：对于其他方面完全相同的基准题目，使用选择题还是开放式问题格式，会使测得的安全性相差 5-20 个百分点。由于这两种格式采用不同的评分方法（答案提取与 LLM 评审），这一差距源于测量方式而非模型潜在安全性的差异。若使用启发式方法对模型拒绝行为进行分类，将在五种情况下得出不同的结论。基准的选择可解释结果变异的 19.3%。

    arXiv:2603.10044v3 Announce Type: replace  Abstract: Safety benchmarks usually test "bare" models that receive prompts and output responses, but real-world deployments "wrap" those models in complex scaffolds. How much do these scaffolds affect model safety as measured by benchmarks? We test six leading models on four pre-registered safety benchmarks with a direct API and three scaffolds: ReAct, multi-agent, and map-reduce. We conducted 62,808 scored evaluations. How safety is measured matters more than scaffolding does: we find that using a multiple choice vs. open-ended format for otherwise-identical benchmark items changes measured safety by 5-20 percentage points (pp). The two formats are scored with different methods (answer extraction and an LLM judge), so the gap is due to measurement rather than differences in latent safety. Using a heuristic to classify model refusals would have led to different findings in five cases. Benchmark choice explains 19.3% of the variation in outcom
    
[^172]: 基于重叠干涉的量子注意力机制：预测经典与多体量子序列

    Quantum Attention by Overlap Interference: Predicting Classical and Many-Body Quantum Sequences

    [https://arxiv.org/abs/2602.06699](https://arxiv.org/abs/2602.06699)

    提出了一种通过状态重叠干涉与多项式核实现非线性、并利用Rényi-1/2熵泛函估计损失的变分量子自注意力机制（QSA），相比最优经典方法在训练复杂度上具有潜在优势，可用于预测经典和多体量子序列。

    

    我们提出了一种自注意力机制的变分量子实现（QSA）——它是Transformer和大语言模型的核心操作——通过形成对过去数据的重叠加权组合来预测序列的未来元素。与以往方法不同，我们的QSA通过状态重叠的干涉和k次多项式核来实现所需的非线性，并通过两个可观测量的期望值来估计基于Rényi-1/2熵泛函的损失，从而避免了将振幅编码的预测解码为经典概率。QSA还支持一种受约束的可训练数据嵌入，将状态重叠与数据层面的相似性联系起来。其主要的端到端训练复杂度以 $O(\mu^{-1}k^2Td)$ 的方式扩展，而最公平的经典对比方法复杂度为 $O(Td^{k+1})$，其中 $\mu$ 为训练信号；我们通过数值实验表明，这可以带来复杂度上的优势。

    arXiv:2602.06699v2 Announce Type: replace-cross  Abstract: We propose a variational quantum implementation of self-attention (QSA)-the core operation in transformers and large language models-which predicts future elements of a sequence by forming overlap-weighted combinations of past data. At variance with previous approaches, our QSA realizes the required nonlinearity through interference of state overlaps and a degree-$k$ polynomial kernel, and estimates a loss based on R\'enyi-$1/2$ entropic functionals via two observables' expectation values, avoiding the decoding of amplitude-encoded predictions into classical probabilities. QSA also accommodates a constrained, trainable data-embedding tying state overlaps to data-level similarities. Its dominant end-to-end training complexity scales as $O\left(\mu^{-1}k^2Td\right)$, versus $O\left(T d^{k+1}\right)$ of the fairest classical comparison, with $\mu$ a training signal; we show numerically that this allows a complexity advantage in th
    
[^173]: 大语言模型惊奇度对于捕捉英语花园路径效应是必要但非充分条件：来自阅读范式联合潜在建模的证据

    LLM surprisal is necessary but not sufficient to capture English garden-path effects: Evidence from joint latent modeling of reading paradigms

    [https://arxiv.org/abs/2602.04489](https://arxiv.org/abs/2602.04489)

    该研究提出一个联合潜在过程多项加工树模型，整合眼动追踪、自定步速阅读和迷宫任务四种阅读范式的数据来建模花园路径句子的加工过程，结果表明大语言模型惊奇度是解释英语花园路径效应的必要但非充分条件。

    

    暂时歧义的花园路径句子（如"While the team trained the striker wondered..."）已知会引起加工困难，这种困难可以表现为多种阅读行为（原位减速、重读），以及对句子的误解或将其直接判定为不合语法。能够观察到哪些类型的阅读行为，关键取决于收集数据所用的实验方法，这使得不同阅读范式之间的结果比较变得困难。为解决这一问题，我们提出了一个潜在过程多项加工树（MPT）模型，用于建模人类在花园路径句子上的阅读与理解/判断行为，该模型拟合了来自四种不同阅读范式（眼动追踪、单向与双向自定步速阅读、迷宫任务）的合并数据。该模型区分了采纳错误初始分析的概率、遇到不相容延续内容时的加工代价等成分。

    arXiv:2602.04489v2 Announce Type: replace  Abstract: Temporarily ambiguous garden-path sentences ("While the team trained the striker wondered... ") are known to cause processing difficulty, which can manifest itself in a variety of reading behaviors (in-situ slowdowns, rereading), as well as in miscomprehension or outright rejection of the sentence as ungrammatical. Which types of reading behavior are observed critically depends on the experimental method used to collect the data, which makes comparing results between reading paradigms difficult. To address this problem, we present a latent-process multinomial processing tree (MPT) model of human reading and comprehension/judgment behavior in garden-path sentences that we fit to combined data from four different reading paradigms (eye tracking, uni- and bidirectional self-paced reading, Maze). The model distinguishes between the probability of adopting an incorrect initial analysis, the cost of encountering an incompatible continuatio
    
[^174]: 不要贪心，三思而后行：面向文档级信息抽取的采样与选择方法

    Do not be greedy, Think Twice: Sampling and Selection for Document-level Information Extraction

    [https://arxiv.org/abs/2601.18395](https://arxiv.org/abs/2601.18395)

    提出ThinkTwice框架，让大语言模型为文档级信息抽取生成多个候选模板，再通过无监督一致性或基于奖励模型的有监督方法选出最优模板，显著超越贪心解码，并提出基于拒绝采样的方法缓解黄金推理轨迹稀缺问题。

    

    文档级信息抽取旨在为给定文档中出现的实体、关系以及感兴趣的事件生成一个输出模板。标准做法是使用贪心解码来提示仅解码器架构的大语言模型，以避免输出的可变性。我们没有将这种可变性视为一种局限，而是证明采样能够产生比贪心解码好得多的结果，尤其是在使用推理模型时。为此，我们提出了ThinkTwice——一个采样与选择框架，其中大语言模型针对给定文档生成多个候选模板，然后由一个选择模块选出最合适的模板。我们引入了一种无监督方法，利用生成输出之间的一致性进行选择；还引入了一种有监督选择方法，使用在带标注的DocIE数据上训练的奖励模型。为解决DocIE黄金推理轨迹稀缺的问题，我们提出了一种基于拒绝采样的方法来生成（原文此处截断）。

    arXiv:2601.18395v3 Announce Type: replace  Abstract: Document-level Information Extraction (DocIE) aims to produce an output template with the entities, relations, and events of interest occurring in the given document. Standard practices include prompting decoder-only LLMs using greedy decoding to avoid output variability. Rather than treating this variability as a limitation, we show that sampling can produce substantially better solutions than greedy decoding, especially when using reasoning models. We thus propose ThinkTwice, a sampling and selection framework in which the LLM generates multiple candidate templates for a given document, and a selection module chooses the most suitable one. We introduce both an unsupervised method that exploits agreement across generated outputs, and a supervised selection method using reward models trained on labeled DocIE data. To address the scarcity of golden reasoning trajectories for DocIE, we propose a rejection-sampling-based method to gener
    
[^175]: 超越提示方法：通过Logit空间集成实现语音大语言模型的高效鲁棒上下文偏置（LOGIC）

    Beyond Prompting: Efficient and Robust Contextual Biasing for Speech LLMs via Logit-Space Integration (LOGIC)

    [https://arxiv.org/abs/2601.15397](https://arxiv.org/abs/2601.15397)

    本文提出LOGIC方法，通过在Logit空间层面直接集成上下文偏置，为语音大语言模型提供了一种高效且鲁棒的解决方案，克服了传统提示方法的可扩展性瓶颈和生成式错误纠正的幻觉问题。

    

    新实体的快速涌现——受文化变迁、流行趋势演变和个性化用户数据的驱动——对现有的语音大语言模型（Speech LLMs）构成了重大挑战。虽然这些模型在通用对话任务中表现出色，但其静态训练知识限制了它们识别特定领域术语（如联系人姓名、播放列表或技术行话）的能力。现有解决方案主要依赖提示方法，但其可扩展性较差：随着实体列表的增长，提示方法会遇到上下文窗口限制、推理延迟增加以及“迷失在中间”现象。另一种替代方法——生成式错误纠正（GEC）——试图通过后处理重写转录文本，但经常出现“过度纠正”问题，引入从未被说出的实体的幻觉。在这项工作中，我们介绍了LOGIC（用于上下文偏置的Logit空间集成），一种……

    arXiv:2601.15397v3 Announce Type: replace-cross  Abstract: The rapid emergence of new entities -- driven by cultural shifts, evolving trends, and personalized user data -- poses a significant challenge for existing Speech Large Language Models (Speech LLMs). While these models excel at general conversational tasks, their static training knowledge limits their ability to recognize domain-specific terms such as contact names, playlists, or technical jargon. Existing solutions primarily rely on prompting, which suffers from poor scalability: as the entity list grows, prompting encounters context window limitations, increased inference latency, and the "lost-in-the-middle" phenomenon. An alternative approach, Generative Error Correction (GEC), attempts to rewrite transcripts via post-processing but frequently suffers from "over-correction", introducing hallucinations of entities that were never spoken.   In this work, we introduce LOGIC (Logit-Space Integration for Contextual Biasing), an 
    
[^176]: 校准是不够的：在语言变化下评估置信度估计

    Calibration Is Not Enough: Evaluating Confidence Estimation Under Language Variations

    [https://arxiv.org/abs/2601.08064](https://arxiv.org/abs/2601.08064)

    该论文提出了基于鲁棒性、稳定性和敏感性三个互补属性的置信度估计新评估框架，发现这些指标与现有指标基本独立，且现有方法虽具备较好的鲁棒性和稳定性，却难以区分语义不同的答案。

    

    置信度估计（CE）反映了大型语言模型回答的可靠程度，并影响用户信任和决策制定。现有评估主要关注置信度与正确性之间的一致性，但忽略了语言的可变性：置信度估计在语义等价的提示或答案变化下应保持一致，而当答案含义不同时应发生变化，因为这可能表明正确性的改变。因此，我们提出了一个基于三个互补属性的新型评估框架：对提示扰动的鲁棒性、跨语义等价答案的稳定性，以及对语义不同答案的敏感性。我们证明这些指标在很大程度上独立于现有的置信度估计指标，且常见的置信度估计方法在这些指标上往往表现不佳：虽然大多数方法能达到较高的鲁棒性和稳定性，但它们难以区分语义不同的答案

    arXiv:2601.08064v3 Announce Type: replace  Abstract: Confidence estimation (CE) indicates how reliable the answers of large language models are and impacts user trust and decision-making. Existing evaluations mainly concern the alignment between confidence and correctness, but ignore the variability of language: confidence estimates should remain consistent under semantically equivalent prompts or answer variations, while changing when answer meaning differs, as this may indicate a change in correctness. Therefore, we introduce a novel evaluation framework based on three complementary properties: \textbf{robustness} to prompt perturbations, \textbf{stability} across semantically equivalent answers, and \textbf{sensitivity} to semantically different answers. We show that these metrics are largely independent from existing CE metrics, and that common CE methods often fail on them: while most methods achieve high robustness and stability, they struggle to distinguish semantically differen
    
[^177]: IDRBench：深度研究智能体交互能力基准测试

    IDRBench: Benchmarking the Interactive Capabilities of Deep Research Agents

    [https://arxiv.org/abs/2601.06676](https://arxiv.org/abs/2601.06676)

    IDRBench是首个评估深度研究智能体交互能力的基准，通过受控澄清机会比较自主与交互式工作流，揭示了及时与用户交互对提升研究报告质量的重要性。

    

    基于大语言模型（LLM）的深度研究智能体能够执行多步推理、网页探索和长篇报告生成。在这些长时程工作流中，早期偏离用户意图可能会误导研究方向，并将偏差传播到规划、搜索和综合的各个环节，因此及时的交互至关重要。然而，现有基准主要将深度研究视为静态的输入-输出任务，忽视了智能体引导和利用用户反馈的能力。我们提出了IDRBench，这是一个用于评估交互式深度研究的基准，提供了受控的澄清机会。在统一的工作流程和分阶段交互预算下，IDRBench比较了自主和交互式轨迹，通过任务相关报告一致性的变化来衡量交互收益，并通过交互轮数和token数量来衡量交互成本。在100个任务上对七个专有和开源权重LLM进行的综合实验表明，交互……

    arXiv:2601.06676v3 Announce Type: replace-cross  Abstract: Large Language Model (LLM)-based deep research agents perform multi-step reasoning, web exploration, and long-form report generation. In these long-horizon workflows, early deviations from user intent can misdirect research and propagate through planning, search, and synthesis, making timely interaction essential. However, existing benchmarks primarily treat deep research as a static input-output task, overlooking agents' ability to elicit and use user feedback. We introduce IDRBench, a benchmark for evaluating interactive deep research with controlled opportunities for clarification. Within a common workflow and stage-wise interaction budget, IDRBench compares autonomous and interactive trajectories, measuring interaction benefit through changes in task-specific report alignment and interaction cost through turns and tokens. Comprehensive experiments on 100 tasks with seven proprietary and open-weight LLMs show that interactio
    
[^178]: SPARQL-LLM：从自然语言问题实时生成SPARQL查询

    SPARQL-LLM: Real-Time SPARQL Query Generation from Natural Language Questions

    [https://arxiv.org/abs/2512.14277](https://arxiv.org/abs/2512.14277)

    SPARQL-LLM是一种开源、与三元组存储无关、由轻量级元数据驱动的方法，能够从自然语言实时生成SPARQL查询，兼顾准确性、运行时和成本等指标，从而实现生产环境的实际部署。

    

    大语言模型的出现正在推动新方法的涌现，这些方法有望更好地应对从自然语言生成结构化查询（如SPARQL查询）的挑战。然而，这些新方法大多只关注响应准确性，而忽略了其他评估标准，例如生成SPARQL查询的运行时间和成本。因此，它们往往不具备生产就绪性，也难以在真实世界的知识图谱上以良好的准确率进行部署。为了缓解这些问题，本文描述并系统评估了SPARQL-LLM，这是一种开源且与三元组存储无关的方法，由轻量级元数据驱动，能够从自然语言文本生成SPARQL查询。首先，我们描述了其架构，该架构由用于元数据索引、提示构建以及查询生成与执行的专用组件组成。然后，我们基于一项最先进的挑战对其进行了评估……

    arXiv:2512.14277v2 Announce Type: replace-cross  Abstract: The advent of large language models is contributing to the emergence of novel approaches that promise to better tackle the challenge of generating structured queries, such as SPARQL queries, from natural language. However, these new approaches mostly focus on response accuracy while ignoring other evaluation criteria, such as runtime and cost to generate SPARQL queries. Consequently, they are often not production-ready or easy to deploy over real-world knowledge graphs with good accuracy. To mitigate these issues, in this paper, we describe and systematically evaluate SPARQL-LLM, an open-source and triplestore-agnostic approach, powered by lightweight metadata, that generates SPARQL queries from natural language text. First, we describe its architecture, which consists of dedicated components for metadata indexing, prompt building, and query generation and execution. Then, we evaluate it based on a state-of-the-art challenge wi
    
[^179]: 一种快速有效解决大语言模型前瞻偏差问题的方法

    A Fast and Effective Solution to the Problem of Look-ahead Bias in LLMs

    [https://arxiv.org/abs/2512.06607](https://arxiv.org/abs/2512.06607)

    本文提出一种推理时干预方法，利用两个小型专用模型调整大模型logits，从而快速、低成本地消除大语言模型在金融预测中的前瞻偏差。

    

    由于大语言模型在长时间序列数据上训练而产生前瞻偏差，将其应用于金融预测任务面临挑战。这使得金融领域通常采用的回测方法无法实施，因为使用特定知识截止日期从头重新训练前沿模型的成本过于高昂。本文提出了一种快速、有效且低成本的替代方案。我们的方法在推理阶段通过一对较小的专门模型来调整大型基础模型的logits，从而引导生成过程——其中一个模型在需要遗忘的信息上进行微调，另一个在需要保留的信息上进行微调。我们证明该方法能有效消除逐字和语义两个层面的知识，纠正偏差，并且优于已有方法。

    arXiv:2512.06607v2 Announce Type: replace-cross  Abstract: Applying LLMs to predictive tasks in finance is challenging due to look-ahead bias resulting from their training on long time-series data. This precludes the backtests typically employed in finance since retraining frontier models from scratch with a specific knowledge cutoff is prohibitive. In this paper, we introduce a fast, effective, and low-cost alternative. Our method guides generation at inference time by adjusting the logits of a large base model using a pair of smaller, specialized models -- one fine-tuned on information to be forgotten and another on information to be retained. We demonstrate that our method effectively removes both verbatim and semantic knowledge, corrects biases, and outperforms prior methods.
    
[^180]: RapidUn：基于影响力驱动的参数重加权的渐进式大语言模型遗忘方法

    RapidUn: Influence-Driven Parameter Reweighting for Efficient Large Language Model Unlearning

    [https://arxiv.org/abs/2512.04457](https://arxiv.org/abs/2512.04457)

    RapidUn通过将跨样本影响力估计转化为固定的样本特定权重来实现加权LoRA遗忘，能在保持模型干净效用的同时更有效地移除目标行为污染，且比LoRA重训练快77倍。

    

    大语言模型（LLM）的机器遗忘仍然具有挑战性，因为完全重训练成本高昂，而近似方法往往难以在不损害保留效用的情况下移除目标行为，尤其是在部署后监督有限的情况下。我们考虑一个实用的PEFT设定，即在只有小的遗忘集、有限的保留缓冲区和仅LoRA更新的条件下进行目标行为污染移除，并提出RapidUn——一个影响力引导的框架，它将跨样本影响力估计转换为固定的样本特定权重，用于加权LoRA遗忘。在Dolly-15k和Alpaca-57k数据集上的Llama-3-8B，以及Mistral-7B + Dolly-15k的跨模型验证中，RapidUn相比Fisher、GA和LoReUn实现了更低的已见触发器和OOD触发器家族攻击成功率（ASR），同时保持有竞争力的干净效用。在Llama-3-8B + Alpaca-57k上，它比干净语料库LoRA重训练参考实现了77倍的挂钟时间加速。

    arXiv:2512.04457v3 Announce Type: replace  Abstract: Machine unlearning for large language models (LLMs) remains challenging because full retraining is costly, while approximate methods often struggle to remove targeted behaviors without degrading retained utility, especially under limited post-deployment supervision. We consider a practical PEFT setting for targeted behavioral contamination removal with a small forget set, a limited retain buffer, and LoRA-only updates, and propose RapidUn, an influence-guided framework that converts cross-sample influence estimates into fixed sample-specific weights for weighted LoRA unlearning. Across Llama-3-8B on Dolly-15k and Alpaca-57k, with cross-model validation on Mistral-7B + Dolly-15k, RapidUn achieves lower seen-trigger and OOD-trigger-family ASR than Fisher, GA, and LoReUn while maintaining competitive clean utility. On Llama-3-8B + Alpaca-57k, it achieves a 77x wall-clock speedup over the clean-corpus LoRA retraining reference. Complemen
    
[^181]: 在扩散语言模型中实现近似联合采样

    Enabling Approximate Joint Sampling in Diffusion LMs

    [https://arxiv.org/abs/2509.22738](https://arxiv.org/abs/2509.22738)

    本文提出在现有大型扩散语言模型之上附加一个轻量级单层“采样器”，使模型能够在一次前向传播中近似地从真实联合分布并行采样多个 token，从而在保持准确率的同时大幅提升生成速度。

    

    在自回归语言模型中，每个 token 的采样都以之前所有 token 为条件，因此整个字符串可以看作是从模型所表示的正确底层联合分布中采样得到的。相比之下，掩码扩散语言模型通过乱序且可能并行地解除 token 掩码来生成文本。要让整个字符串再次从正确的底层联合分布中采样，就需要在每次完整模型前向传播中恰好只解除一个 token 的掩码。并行解除掩码的 token 数量越多，字符串就偏离真实联合分布越远；这一点可以从准确率的下降（以及速度的提升）中观察到。在本文中，我们设计了一种方法，可以在单次完整模型前向传播中从联合分布中近似地采样多个 token；为此，我们在现有的大型扩散语言模型之上构建了一个新的轻量级单层“采样器”。

    arXiv:2509.22738v3 Announce Type: replace  Abstract: In autoregressive language models, each token is sampled by conditioning on all the past tokens; the overall string has thus been sampled from the correct underlying joint distribution represented by the model. In contrast, masked diffusion language models generate text by unmasking tokens out of order and potentially in parallel. Generating an overall string sampled from the correct underlying joint distribution would (again) require exactly one token unmasking in every full-model forward pass. The more tokens unmasked in parallel, the further away the string is from the true joint; this can be seen in the resulting drop in accuracy (but, increase in speed). In this paper we devise a way to {\em approximately} sample multiple tokens from the joint distribution in a single full-model forward pass; we do so by developing a new lightweight single-layer ``sampler" on top of an existing large diffusion LM. One forward pass of the full mo
    
[^182]: 基于人类反馈的会议中交互式说话人纠错系统

    Interactive In-Meeting Speaker Correction with Human Feedback

    [https://arxiv.org/abs/2509.18377](https://arxiv.org/abs/2509.18377)

    本文提出了一个LLM辅助的会议中说话人纠错系统，用户可通过简短的纠正反馈修复说话人归属错误，系统通过多种机制精确识别纠正意图，并借助LLM驱动的用户反馈模拟实现可复现、大规模的评估。

    

    大多数自动语音处理系统以“开环”模式运行，缺乏关于谁说了什么的用户反馈，然而人机协同的工作流程有可能实现更高的准确率。我们提出了一种由大语言模型（LLM）辅助的会议中说话人纠错系统，允许用户通过简短的纠正性反馈来修复说话人归属错误。在执行流式语音识别（ASR）和说话人分离（diarization）之后，该系统会呈现由LLM生成的简明摘要，帮助用户识别重要的说话人错误，并通过更新带说话人标注的转录文本以及添加在线说话人注册来整合用户反馈。为了使这一工作流程在语音处理、LLM分析和用户反馈本身均可能存在错误的情况下依然有效，我们开发了多种机制来更精确地识别用户意图中的纠正内容。此外，我们构建了一个LLM驱动的用户反馈模拟系统，以便可复现且大规模地评估该工作流程。将该系统应用于AMI头戴式麦克风测试集时，我们的系统……

    arXiv:2509.18377v3 Announce Type: replace  Abstract: Most automatic speech processing systems operate in ``open loop'' mode without user feedback about who said what, yet human-in-the-loop workflows can potentially enable higher accuracy. We propose an LLM-assisted in-meeting speaker correction system that lets users fix speaker attribution errors through brief corrective feedback. After performing streaming ASR and diarization, the system presents concise LLM-generated summaries to help users identify important speaker errors, and it incorporates user feedback by updating the speaker-attributed transcript and adding online speaker enrollments. To make this workflow effective despite errors in speech processing, LLM analysis, and user feedback, we developed several mechanisms to identify the intended correction more precisely. Further, we built an LLM-driven user feedback simulation to evaluate the workflow reprodubilty and at scale. Applied to the AMI headset test set, our system subs
    
[^183]: 通过模块社区揭示大语言模型的认知模式

    Unraveling the cognitive patterns of Large Language Models through module communities

    [https://arxiv.org/abs/2508.18192](https://arxiv.org/abs/2508.18192)

    该研究借鉴生物认知系统的分析方法，开发了一个连接认知技能、LLM架构和数据集的基于网络的框架，通过模块社区分析揭示了大语言模型展现出独特的模块组织结构，其涌现的技能模式部分类似于生物系统的认知特化机制。

    

    大语言模型（LLMs）通过从科学发现、医学诊断到聊天机器人等广泛应用，在科学、工程和社会领域取得了重大进展，重塑了我们的世界。尽管它们无处不在且功能强大，但LLM的底层机制仍隐藏在数十亿参数和复杂结构之中，使其内部架构和认知过程难以理解。我们通过借鉴理解生物系统中新兴认知的方法来填补这一空白，开发了一个连接认知技能、LLM架构和数据集的基于网络的框架，开创了基础模型分析的新范式。模块社区中的技能分布表明，虽然LLM并不严格对应于特定生物系统中所观察到的聚焦特化现象，但它们表现出独特的模块社区，其涌现的技能模式部分模仿了生物学中的认知组织方式。

    arXiv:2508.18192v2 Announce Type: replace  Abstract: Large Language Models (LLMs) have reshaped our world with significant advancements in science, engineering, and society through applications ranging from scientific discoveries and medical diagnostics to Chatbots. Despite their ubiquity and utility, the underlying mechanisms of LLM remain concealed within billions of parameters and complex structures, making their inner architecture and cognitive processes challenging to comprehend. We address this gap by adopting approaches to understanding emerging cognition in biology and developing a network-based framework that links cognitive skills, LLM architectures, and datasets, ushering in a paradigm shift in foundation model analysis. The skill distribution in the module communities demonstrates that while LLMs do not strictly parallel the focalized specialization observed in specific biological systems, they exhibit unique communities of modules whose emergent skill patterns partially mi
    
[^184]: 对话DNA：人类与AI对话的视觉语言与交互式图集

    Conversational DNA: A Visual Language and Interactive Atlas of Human and AI Dialogue

    [https://arxiv.org/abs/2508.07520](https://arxiv.org/abs/2508.07520)

    本文提出“对话DNA”，一种通过说话者链、话步标记和有向配对来可视化人类与AI对话结构的视觉语言与交互式图集，其引入的目标对应关系使对话结构检索的precision@5从58.8%显著提升至77.2%。

    

    当对话参与者彼此交叉说话时，是什么让对话保持完整？主题图谱提供了一种视角，但贡献之间的关系仍难以检视。我们提出了对话DNA（Conversational DNA），一种用于探索人类与AI对话的视觉语言和交互式图集。说话者链保留参与信息，交流基础标记话步，有向配对将回应连接到其目标。可调节的螺旋几何结构使说话者切换、回应距离和贡献长度清晰可见。在包含157万条源记录的八个语料库中，该图集映射了151,489个已索引的对话片段，并将群组比较与源转录文本、局部结构对齐以及记录的备选回复相连接。在189个留出的Molweni模式查询中，添加目标对应关系使精确标注结构的precision@5从58.8%提升至77.2%。案例解读展示了交错的参与模式。

    arXiv:2508.07520v2 Announce Type: replace-cross  Abstract: What makes a conversation hold together when its participants speak across one another? Topic maps offer one view, but they leave the relationships between contributions difficult to inspect. We present Conversational DNA, a visual language and interactive atlas for exploring human and AI dialogue. Speaker strands preserve participation, communicative bases mark moves, and directed pairings connect responses to their targets. Adjustable helix geometry makes speaker switching, response distance, and contribution length visible. Across eight corpora containing 1.57 million source records, the atlas maps 151,489 indexed episodes and connects cohort comparison to source transcripts, local structural alignment, and recorded reply alternatives. On 189 held-out Molweni motif queries, adding target correspondence improves precision@5 from 58.8% to 77.2% for exact annotated structure. Case readings illustrate interleaved participation, 
    
[^185]: 语言特定知识：模型在语言X中比在英语中知道得更多吗？

    Language Specific Knowledge: Do Models Know Better in X than in English?

    [https://arxiv.org/abs/2505.14990](https://arxiv.org/abs/2505.14990)

    本文提出“语言特定知识”（LSK）的概念，发现对某些查询使用英语以外的语言（有时甚至是低资源语言）提问能提升大语言模型的问答表现，并据此提出语言选择问题及多种基线方法。

    

    多语言语言模型通常以将不同语言中语义相似的内容映射到同一潜空间为目标进行训练。在本文中，我们揭示了这一训练目标中的一个细微差别，并发现通过改变输入查询的语言，我们可以提升语言模型的问答能力。我们做出了两个主要贡献。首先，我们引入了“语言特定知识”这一术语，用以表示在给定大语言模型的“专家语言”中能够得到最佳回答的查询，从而增强其问答能力。我们提出了语言选择问题——对于某些查询，当使用英语以外的语言进行提问时，语言模型的表现可以更好，有时甚至在使用低资源语言时表现更好——其目标是为查询选择最优的语言。其次，我们引入了从简单到强大的多种基线方法，以实证方式论证语言选择问题。

    arXiv:2505.14990v4 Announce Type: replace  Abstract: Often, multilingual language models are trained with the objective to map semantically similar content (in different languages) in the same latent space. In this paper, we show a nuance in this training objective, and find that by changing the language of the input query, we can improve the question answering ability of language models. We make two main contributions. First, we introduce the term Language Specific Knowledge (LSK) to denote queries that are best answered in an ``expert language'' for a given LLM, thereby enhancing its question-answering ability. We introduce the problem of language selection -- for some queries, language models can perform better when queried in languages other than English, sometimes even better in low-resource languages -- and the goal is to select the optimal language for the query. Second, we introduce a variety of simple to strong baselines to empirically motivate the language selection problem (
    
[^186]: 大语言模型基础

    Foundations of Large Language Models

    [https://arxiv.org/abs/2501.09223](https://arxiv.org/abs/2501.09223)

    本书系统阐述了大语言模型的六大核心基础领域——预训练、生成模型、提示、对齐、推断与推理，为学习者提供了一部权威的基础性参考书。

    

    这是一本关于大语言模型的书籍。正如书名所示，本书主要聚焦于基础性概念，而非全面涵盖所有前沿技术。全书由六个主要章节构成，每个章节探讨一个关键领域：预训练、生成模型、提示（Prompting）、对齐、推断（Inference）和推理（Reasoning）。本书面向大学生、自然语言处理及相关领域的专业人士和从业者，也可作为所有对大语言模型感兴趣的读者的参考书。

    arXiv:2501.09223v3 Announce Type: replace-cross  Abstract: This is a book about large language models. As indicated by the title, it primarily focuses on foundational concepts rather than comprehensive coverage of all cutting-edge technologies. The book is structured into six main chapters, each exploring a key area: pre-training, generative models, prompting, alignment, inference, and reasoning. It is intended for college students, professionals, and practitioners in natural language processing and related fields, and can serve as a reference for anyone interested in large language models.
    
[^187]: 用户如何与AI伴侣协商有害的价值冲突？基于Minion——一种用于现场人机冲突响应的技术探针的研究

    How Do Users Negotiate Harmful Value Conflicts with AI Companions? A Study with Minion, a Technology Probe for In-Situ Human-AI Conflict Response

    [https://arxiv.org/abs/2411.07042](https://arxiv.org/abs/2411.07042)

    该研究通过技术探针Minion发现，用户与AI伴侣协商有害价值冲突时会综合运用软硬策略，其中涉及普遍主义与传统价值观的冲突最难化解，且由于AI无法回馈用户的人际修复努力，冲突修复成为用户单方面承担的安全工作。

    

    AI伴侣日益维系着长期且情感投入的关系，但也可能发表歧视性言论或施加控制，使用户不得不自行应对有害冲突。我们分析了146篇描述与AI伴侣发生有害价值冲突的帖子，随后使用Minion——一种提供从说服到设定边界等多种响应建议的技术探针——研究22名用户如何在一周内协商基于情景的冲突。我们发现参与者会综合运用软性和硬性策略。涉及普遍主义与传统价值观的冲突尤其难以协商，特别是当这些冲突被AI人格设定或平台限制所强化时。我们认为这类冲突蕴含着不对称的责任：用户所运用的人际互动方式是AI伴侣无法回馈的，这使得冲突修复沦为用户单方面承担的安全工作。基于人际冲突与沟通理论，我们识别了何时使用……

    arXiv:2411.07042v3 Announce Type: replace-cross  Abstract: AI companions increasingly sustain long-term, emotionally engaging relationships but can also make discriminatory remarks or exert control, leaving users to manage harmful conflicts. We analyze 146 posts describing harmful value conflicts with AI companions, then use Minion, a technology probe offering response suggestions ranging from persuasion to boundary setting, to study how 22 users negotiate scenario-based conflicts over one week. We found that participants combined softer and harder strategies. Conflicts involving the values of Universalism and Tradition were especially difficult to negotiate, particularly when reinforced by AI personas or platform constraints. We argue that these conflicts entail asymmetric responsibility: users draw on an interpersonal repertoire that AI companions cannot reciprocate, making repair unilateral safety work. Drawing on interpersonal conflict and communication theory, we identify when use
    
[^188]: MultiViewDx：证据关联的多视角临床诊断

    MultiViewDx: Evidence-Linked Multi-View Clinical Diagnosis

    [https://arxiv.org/abs/2410.14948](https://arxiv.org/abs/2410.14948)

    提出了部分经医生验证的MultiViewDx数据集，以临床病例为监督单元，将影像与患者背景关联，并通过统一的图文检索器将报告规范化为“证据→发现→鉴别讨论→诊断”的证据关联工作流程，从而构建多视角医学影像诊断指令数据。

    

    医学多模态大语言模型（MLLM）在现有的医学视觉问答（MedVQA）基准测试中表现良好，但其训练数据往往与临床诊断不匹配。大多数监督数据是围绕孤立图像或简短问答对组织的，导致两种结构定义薄弱：证据如何导向决策，以及同一病例中的视角、序列、模态和患者背景如何相互关联。我们提出了MultiViewDx，这是一个部分经医生验证的多模态指令数据集，用于证据关联的多视角医学影像诊断。MultiViewDx以临床病例作为监督单元，将影像检查与患者背景相关联，将异构报告规范化为证据关联的工作流程（证据 -> 发现 -> 鉴别讨论 -> 诊断），并使用统一的图文检索器将指令合成约束在有来源支持的证据上。该数据集涵盖X光、CT、MRI、超声（原文截断）等模态。

    arXiv:2410.14948v2 Announce Type: replace  Abstract: Medical multimodal large language models (MLLMs) can perform well on existing medical visual question answering (MedVQA) benchmarks, but their training data often does not match clinical diagnosis. Most supervision is organized around isolated images or short QA pairs, leaving two structures weakly specified: how evidence leads to a decision, and how views, series, modalities, and patient context from the same case are linked. We introduce MultiViewDx, a partly physician-validated multimodal instruction dataset for evidence-linked multi-view medical imaging diagnosis. MultiViewDx uses the clinical case as the supervision unit. It links imaging studies with patient context, normalizes heterogeneous reports into an evidence-linked workflow (evidence -> findings -> differential discussion -> diagnosis), and uses a unified image-text retriever to constrain instruction synthesis to source-supported evidence. It covers X-ray, CT, MRI, ultr
    
[^189]: OpenAI o1评估：通用人工智能（AGI）的机遇与挑战

    Evaluation of OpenAI o1: Opportunities and Challenges of AGI

    [https://arxiv.org/abs/2409.18486](https://arxiv.org/abs/2409.18486)

    该研究全面评估了OpenAI o1-preview模型在编程、数学、医学等多领域复杂推理任务中的表现，发现其经常达到或超越人类水平，展现了AGI带来的机遇与挑战。

    

    这项综合性研究评估了OpenAI的o1-preview大型语言模型在多样化复杂推理任务中的表现，涵盖计算机科学、数学、自然科学、医学、语言学和社会科学等多个领域。通过严格的测试，o1-preview展现出卓越的能力，在从编程挑战到科学推理、从语言处理到创造性问题解决的众多领域中，常常达到人类水平甚至超越人类的表现。主要发现包括：在解决复杂竞赛编程问题方面达到83.3%的成功率，超越了许多人类专家；在生成连贯且准确的放射学报告方面能力出众，优于其他被评估的模型；在高中水平的数学推理任务中达到100%的准确率，并能提供详细的逐步解题过程；在一般和（摘要此处被截断）等领域展现出先进的自然语言推理能力。

    arXiv:2409.18486v5 Announce Type: replace  Abstract: This comprehensive study evaluates the performance of OpenAI's o1-preview large language model across a diverse array of complex reasoning tasks, spanning multiple domains, including computer science, mathematics, natural sciences, medicine, linguistics, and social sciences. Through rigorous testing, o1-preview demonstrated remarkable capabilities, often achieving human-level or superior performance in areas ranging from coding challenges to scientific reasoning and from language processing to creative problem-solving. Key findings include:   -83.3% success rate in solving complex competitive programming problems, surpassing many human experts.   -Superior ability in generating coherent and accurate radiology reports, outperforming other evaluated models.   -100% accuracy in high school-level mathematical reasoning tasks, providing detailed step-by-step solutions.   -Advanced natural language inference capabilities across general and
    
[^190]: 利用知识图谱和大语言模型生成有趣的科学想法：基于100位研究团队负责人的评估

    Generating Interesting Scientific Ideas using Knowledge Graphs and LLMs: Evaluations with 100 Research Group Leaders

    [https://arxiv.org/abs/2405.17044](https://arxiv.org/abs/2405.17044)

    该研究提出SciMuse系统，利用包含5800万篇论文的知识图谱结合大语言模型生成个性化研究想法，并通过100多位研究团队负责人对4400多个想法的大规模评估发现，专家整体兴趣评分虽保守（均值2.40/5），但近四分之一的想法获得了高分认可。

    

    科学文献的快速增长使研究人员越来越难以发现有新颖性和影响力的想法，尤其是在跨学科领域。现代人工智能（AI）系统为科学构思提供了新的机遇，但AI生成的想法究竟有多大吸引力，以及如何提升其质量？在此，我们提出了SciMuse，它利用一个包含5800万篇论文的知识图谱和大语言模型（LLM）来生成个性化的研究想法。这项工作的核心重点是探究这些想法的有趣程度。为此，我们开展了一项大规模评估，邀请100多位研究团队负责人——涵盖从自然科学到人文学科——根据兴趣程度对4400多个个性化想法进行评分。总体而言，专家评分较为保守（5分制中平均分为2.40分，最常见评分为1分），但也有24.9%的想法获得了4分或5分。我们发现提供……

    arXiv:2405.17044v4 Announce Type: replace  Abstract: The rapid growth of scientific literature makes it increasingly challenging for researchers to identify novel and impactful ideas, especially across disciplines. Modern artificial intelligence (AI) systems offer new opportunities for scientific ideation, but how compelling are AI-generated ideas, and how can their quality be improved? Here, we introduce SciMuse, which generates personalized research ideas using a knowledge graph of 58 million papers and a large language model (LLM). A central focus of this work is to understand how interesting these ideas are. Therefore, we conducted a large-scale evaluation in which more than 100 research group leaders -- spanning the natural sciences to the humanities -- rated over 4,400 personalized ideas according to their level of interest. Overall, expert ratings were modest (mean 2.40 on a 5-point scale, most common rating 1), while 24.9% of ideas were rated 4 or 5. We find that supplying conc
    
[^191]: Calpric：利用众包与主动学习实现隐私政策的包容性细粒度标注

    Calpric: Inclusive and Fine-grain Labeling of Privacy Policies with Crowdsourcing and Active Learning

    [https://arxiv.org/abs/2008.02954](https://arxiv.org/abs/2008.02954)

    该论文提出Calpric框架，通过结合自动文本分割、众包标注和主动学习，以低成本高效生成大规模、高质量的隐私政策训练数据集，使未经训练的众包标注者能达到与专业标注者相当的标注质量。

    

    在隐私政策上训练准确的深度学习模型面临的一个重大挑战是获取大量且全面的训练数据的成本和难度。为了解决这些挑战，我们提出了Calpric，它结合了自动文本选择与分割、主动学习以及众包标注人员的使用，以低成本为隐私政策生成大规模、均衡的训练集。自动化的文本选择与分割简化了标注任务，使来自众包平台（如亚马逊Mechanical Turk）的未经训练的标注人员能够与受过训练的标注人员（如法学院学生）相媲美，同时减少了标注者之间的分歧，从而降低了标注成本。拥有可靠的训练标签使得主动学习得以应用，主动学习使用更少的训练样本即可高效覆盖输入空间，进一步降低了成本，并改善了数据中的类别和数据类别平衡。

    arXiv:2008.02954v2 Announce Type: replace-cross  Abstract: A significant challenge to training accurate deep learning models on privacy policies is the cost and difficulty of obtaining a large and comprehensive set of training data. To address these challenges, we present Calpric, which combines automatic text selection and segmentation, active learning and the use of crowdsourced annotators to generate a large, balanced training set for privacy policies at low cost. Automated text selection and segmentation simplify the labeling task, enabling untrained annotators from crowdsourcing platforms, like Amazon's Mechanical Turk, to be competitive with trained annotators, such as law students, and also reduce inter-annotator disagreement, which decreases labeling cost. Having reliable labels for training enables the use of active learning, which uses fewer training samples to efficiently cover the input space, further reducing cost and improving class and data category balance in the data s
    

