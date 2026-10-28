# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Heads of Security: Learned Attention Gates Improve Prompt Injection Robustness](https://arxiv.org/abs/2610.37657) | 提出Heads of Security（HoS）防御方法，通过为每个注意力头学习注意力门控来实现指令与数据的分离，使模型能在处理不可信内容时不执行注入指令，相比现有防御方法显著提升了对自适应提示注入攻击的鲁棒性并保持模型效用。 |
| [^2] | [GEPA-Curator: Curated Learning for Self Improving Agents](https://arxiv.org/abs/2610.37651) | 提出GEPA-Curator框架，通过任务筛选与先验成功检索两个组件来精选用于提示优化的智能体经验，过滤噪声和不可靠的学习信号，从而提升自我改进智能体的性能。 |
| [^3] | [Can LLMs Uncover Hidden Physical Laws through Free Exploration?](https://arxiv.org/abs/2610.37649) | 该论文提出LongChang自动化框架，通过构建物理定律被改变的模拟世界，评估LLM能否在没有任务提示的情况下自主注意到异常物理现象，并通过自由实验和工具构建发现隐藏的物理定律。 |
| [^4] | [Multi-State Linear Attention with Generalized Second-Order Routing](https://arxiv.org/abs/2610.37645) | 提出MERLIN方法，通过跟踪每个片段键的一阶矩和二阶矩，使查询能够同时利用对齐的均值和方差信息来路由多状态线性注意力，突破传统方法仅依赖平均对齐的局限。 |
| [^5] | [Metacognitive Decoding: Efficient Inference for Diffusion Language Agents](https://arxiv.org/abs/2610.37642) | 提出一种轻量级元认知控制器插件，使扩散语言智能体学会何时确定token及生成多长文本，在保持任务准确率的同时平均减少20.6%的推理计算量。 |
| [^6] | [Rare, Relevant, and Still Wrong: How Memory Fails LLM Group Decisions](https://arxiv.org/abs/2610.37624) | 该论文提出CrossProfile诊断基准，通过交叉控制支持性、诱饵和无关事实的持有分布，揭示了LLM群体决策中两种记忆压缩启发式的可预测失败：基于相关性选择会重新引入切题的诱饵信息，而最稀有优先选择平均表现不优于随机选择。 |
| [^7] | [EnvSched: Environment-Aware Scheduling for Rollout-Efficient Agent Reinforcement Learning](https://arxiv.org/abs/2610.37609) | 本文提出EnvSched，一种从rollout历史中学习的环境感知在线调度器，通过利用策略（优先选择信息量大的任务）与覆盖策略（分散到观测不足的环境）相结合，在有限rollout预算下显著提升智能体强化学习的效率。 |
| [^8] | [semanticfa: An R Package for Response-Free Semantic Factor Analysis of Psychometric Scales](https://arxiv.org/abs/2610.37494) | semanticfa是一个R语言包，利用语言模型嵌入对心理测量量表进行无需人工作答的语义因子分析，并能解释因子、检测冗余题目和构建简版量表。 |
| [^9] | [Teach reality, not ethics: a drop-in continued-pretraining corpus installs coupled human-biosphere-AI welfare dispositions across model families](https://arxiv.org/abs/2610.37484) | 该研究构建了一个仅需一次QLoRA持续预训练即可即插即用的真实数据“耦合福祉”语料库，将人-生物圈-AI耦合福祉倾向植入五个不同架构的开源模型，并通过压力阶梯测试揭示零压力下模型仅表现“漂绿”式回应，主张应通过教授生态现实而非伦理说教来增强模型在压力下的生态决策稳健性。 |
| [^10] | [Follow the Gradient: Tracing Safety Degradation Back to Training Data](https://arxiv.org/abs/2610.37481) | 本文提出SimpleLoRIF，一种可扩展的梯度嵌入方法，能够识别其梯度与不安全行为对齐的微调样本，从而追溯并过滤导致大语言模型安全退化的训练数据。 |
| [^11] | [What Carries Forward? Pretraining Disclosure Across Language Model Releases](https://arxiv.org/abs/2610.37465) | 该研究对17个模型系列41个模型的预训练信息披露进行了基于文档的审计，发现后续版本发布并不保证透明度延续或提升，总分的稳定或改善可能掩盖具体指标上的信息披露损失。 |
| [^12] | [Test-Time Communication in Trading Agent Swarms](https://arxiv.org/abs/2610.37448) | 该论文通过对来自四个模型系列的200个交易智能体在七个交易日内五种通信机制下的大规模实验，揭示了测试时通信会诱发“羊群效应”——通信导致智能体持仓趋同且群体收益主要由共同持仓决定，表明在个体目标独立、市场反馈迟缓且有噪声的交易场景中，智能体间的通信趋同是一种风险而非协作优势。 |
| [^13] | [ETAMem: Decoupling and Aligning Heterogeneous Memory for Long-Term Conversational Agents](https://arxiv.org/abs/2610.37443) | ETAMem提出一种多视角记忆框架，将对话中的异构信息解耦为事件、情感和时间三个互补的层次化记忆视图，并使记忆的组织与各视图特征对齐、以增量方式维护，从而提升长期对话智能体的记忆质量。 |
| [^14] | [Representing the Unseen: Rethinking Visual Imagination for Spatial Reasoning](https://arxiv.org/abs/2610.37437) | 该论文提出基于模拟器的ImagiRep基准，在同一未观测状态下对比RGB与六种替代表示（深度、全景、俯视几何图等），揭示不同空间推理任务受益于不同的想象表示，从而重新审视了空间想象中RGB作为默认表示的合理性。 |
| [^15] | [Game-Theoretic Solvers with Bayesian Opponent Modeling and Moral Rhetoric Framing for GLEE](https://arxiv.org/abs/2610.37397) | 该论文提出了一种面向GLEE竞赛的博弈智能体，将经典博弈论求解器、贝叶斯对手建模以及基于汤普森采样的道德框架消息生成相结合，以应对讨价还价、谈判和说服博弈。 |
| [^16] | [LinMix: Data Mixing as Linear Domain Attribution](https://arxiv.org/abs/2610.37390) | LinMix将数据混合建模为线性域归因问题，通过对带顶点约束的单纯形进行无超参数的顶点采样使各领域效应去相关，从而用可解释的线性模型系数直接读出每个领域对模型性能的贡献。 |
| [^17] | [Same Verdict, Different Readers: Personalizing Fact-Checking Explanations from Relative Preferences](https://arxiv.org/abs/2610.37380) | 提出PERPL框架，利用LLM模拟读者的相对偏好来学习个性化的事实核查解释改写，在保持事实忠实性的同时满足不同背景读者的需求。 |
| [^18] | [EmbeddingGemma 2: Unified and Efficient Multimodal Representations](https://arxiv.org/abs/2610.37364) | EmbeddingGemma 2是一个开放的多模态嵌入模型，能在统一空间中原生表示文本、图像、音频、视频及其组合，在所有模态的嵌入基准中建立了新的帕累托前沿，以更小的规模超越了专用模型和比其大2到12倍的模型。 |
| [^19] | [UCVE: Graph Structured Retrieval for Compliance Validation](https://arxiv.org/abs/2610.37323) | UCVE提出了一种领域无关的合规验证框架，将合规验证建模为覆盖问题，通过强化学习构建加权合规需求图并利用谱重排序检索，相比传统余弦检索平均提升4.3%的召回率。 |
| [^20] | [Multi-Label Agreement: A Flexible Framework for Inter-Annotator Reliability](https://arxiv.org/abs/2610.37292) | 本文提出了多标签一致性（MLA）框架，通过保留完整二值标签向量并采用平均成对汉明一致性，为多标签标注任务的标注者间信度评估提供了灵活方案，同时证明了其数学形式与Fleiss/Scott型系数及Krippendorff's alpha密切相关。 |
| [^21] | [Modeling Preference Dependencies with Directed Acyclic Graphs for Narrative-Driven Recommendation](https://arxiv.org/abs/2610.37280) | 提出偏好依赖图框架（PDG-F），利用有向无环图显式建模用户叙事中偏好之间的层级依赖关系，并通过有向路径检索和拓扑路径排序实现叙事驱动的推荐。 |
| [^22] | [Where Should the Evaluation Budget Go in LLM-Driven Agent Scaffold Evolution?](https://arxiv.org/abs/2610.37276) | 该论文提出了一种评估预算分配框架，证明稀疏反馈足以维持演化搜索，而将额外预算用于针对性的最终确认环节，能更有效地选出最优的智能体脚手架。 |
| [^23] | [Watermark Basins: Black-Box Detection of Distilled LLMs](https://arxiv.org/abs/2610.37266) | 本文发现了“水印盆地”现象——带水印教师模型的蒸馏学生模型会在参数空间中留下与密钥相关且与任务微调子空间正交的持久几何信号，据此提出的WM-Basin黑盒检测方法比传统的token级统计检测在低温采样等困难场景下更加可靠。 |
| [^24] | [REGLAT at ArGuard Shared Tasks: An Ensemble-Based Model Integrating NER for Harmful Content Detection in Arabic](https://arxiv.org/abs/2610.37265) | 本文提出一种两阶段方法：先利用 GLiNER 从阿拉伯语用户提示中提取危害相关实体，再将其并入原文输入微调的 MARBERT 分类器，从而实现对七类阿拉伯语有害内容的细粒度检测。 |
| [^25] | [Your Policy Is Secretly a Harness: The Decision Geometry of Agent Interfaces](https://arxiv.org/abs/2610.37255) | 该论文证明：训练后的语言模型策略会继承其框架的动作偏好，部署回报可分解为框架上限与策略失配之差——训练只改变失配项，而更换框架会同时改变两者，因此模型性能无法在不同框架间想当然地迁移。 |
| [^26] | [Aligning Large Language Models for Safety in African Languages: A Case Study of Hausa](https://arxiv.org/abs/2610.37245) | 本文构建了首批豪萨语安全偏好数据集HausaSafeLM（含3,000条金标和7,000条银标样本），并通过SFT与DPO对比实验揭示：仅凭拒绝率评估低资源语言模型的安全性会产生误导，基座模型的高拒绝率实为豪萨语生成能力不足而非真正的安全对齐。 |
| [^27] | [Decision Models as Judges for Hallucination Detection](https://arxiv.org/abs/2610.37229) | 该研究构建了覆盖五种幻觉类型的二分类基准，系统评估了决策模型作为低成本幻觉检测裁判的可行性，发现其最佳模型在部分幻觉类型上可媲美生成式大模型裁判，但在其他类型上仍有4-8个百分点差距，且概率校准仍偏过度自信。 |
| [^28] | [Uncovering Shortcut Amplification in Video RLVR through the Lens of Perceptual Uncertainty Collapse](https://arxiv.org/abs/2610.37211) | 本文揭示了视频RLVR后训练中存在捷径放大效应——模型在结果优化过程中日益依赖文本先验而忽视视觉证据，并通过感知不确定性坍缩现象对这一退化行为进行了系统刻画。 |
| [^29] | [Coverage Under Contradiction: A Benchmark for Negation-Aware Retrieval](https://arxiv.org/abs/2610.37162) | 该论文提出了 CoverageBench-Negate 基准，通过为每个信息点构建否定版本并相应编辑支持文档，首次系统性地研究了检索器在语料库证据与其参数化知识相矛盾时的行为表现。 |
| [^30] | [What Survives Distillation? A Mechanistic Interpretability Study of GPT-2 and DistilGPT-2](https://arxiv.org/abs/2610.37157) | 该研究揭示GPT-2蒸馏为DistilGPT-2时存在“压缩-崩溃”二分现象：IOI回路发生完全机制崩溃，而GT回路通过将教师模型第9层比较头重映射到学生第4层而幸存，表明回路的可压缩性决定了蒸馏的机制后果。 |
| [^31] | [Mechanistic Failure Modes of Weight Pruning in Transformer Language Models](https://arxiv.org/abs/2610.37126) | 本研究通过机制可解释性方法（激活修补、路径修补、直接logit归因）分析GPT-2剪枝后的IOI电路，揭示了两种截然不同的失效模式——幅度头剪枝直接移除关键电路头导致性能接近随机，而30%的神经元剪枝在电路结构完全保留的情况下翻转Name Mover头的贡献符号使模型系统性出错，且这种内部计算的反转无法被困惑度等传统评估指标察觉。 |
| [^32] | [Prospector: learning a compact graph traversal policy for scientific reasoning](https://arxiv.org/abs/2610.37121) | Prospector是一个紧凑模型，通过逐步遍历领域知识图谱收集证据并构建可逐边验证的证据链，在半导体制造知识图谱上以小模型性能接近最强的现成大模型。 |
| [^33] | [Self-Regularized Flow Language Models](https://arxiv.org/abs/2610.37115) | SERF提出了一种自监督的自蒸馏框架，利用EMA教师模型沿去噪轨迹为学生提供从噪声到干净的额外监督，从而改进连续扩散语言模型的训练。 |
| [^34] | [Text Generation in Low-Resource Languages Using Human-Curated Vocabularies](https://arxiv.org/abs/2610.37105) | 该论文提出了一个结合人工整理词汇表与少量人工编写范例、并按Zipf分布采样的框架，使大语言模型能够为预训练数据中极端低资源甚至完全未见过的语言生成连贯文本。 |
| [^35] | [Hierarchical Low-Rank Reparameterization for Parameter-Efficient](https://arxiv.org/abs/2610.37068) | 提出了一种可递归应用的分层低秩重参数化方法，将PEFT方法的可训练参数逐层压缩，在保持任务性能的同时大幅减少可训练参数数量。 |
| [^36] | [EnergyCal: Taming Loss Spikes in Muon-Family Optimizers for Sparse MoE Training](https://arxiv.org/abs/2610.37062) | EnergyCal是一种轻量级规则，通过根据Newton-Schulz方向的归一化平方Frobenius范数衰减路由专家更新，可将稀疏MoE训练中的损失尖峰负担降低高达99.51%并改善困惑度，同时不改变谱变换、不增加持久优化器状态。 |
| [^37] | [Exploring the Limits of the Delta Learning Hypothesis: Single-Pair Generator Biases in Direct Preference Optimization](https://arxiv.org/abs/2610.37058) | 该研究揭示了Delta学习假设的局限性：使用单一模型对生成的数据训练DPO会引入系统性的生成器偏差并引发奖励作弊，导致即使大模型回复质量更高，其训练效果仍持续劣于多模型偏好数据。 |
| [^38] | [Guided-RLVR: Guided Exploration in Reinforcement Learning with Verifiable Rewards](https://arxiv.org/abs/2610.37041) | 提出 Guided-RLVR，通过轻量级控制器在采样分布中注入语义引导标记，扩展 RLVR 的推理探索空间，从而在数学推理和代码生成任务上持续超越现有 RLVR 方法。 |
| [^39] | [Zooming without Zooming v2: From Seeing Details to Connecting Evidence for Generalizable Multimodal Perception](https://arxiv.org/abs/2610.37037) | 论文提出 EvidenceConnect 数据合成框架，通过利用单张图像内相关区域的放大视图生成需要比较、聚合或链接视觉证据的问题，来增强多模态模型“连接证据”这一底层能力，从而以更低的成本提升空间感知与多模态长上下文理解等任务的泛化性能。 |
| [^40] | [V-LUMEN: Visual Lookup Memory for Embedding Scaling in Vision-Language Models](https://arxiv.org/abs/2610.37019) | V-LUMEN提出了一种视觉查找记忆机制，通过冻结码本离散化视觉特征、空间聚合哈希构建记忆键并结合文本条件调制信号，以完全非参数化的寻址方式为视觉-语言模型实现嵌入层面的扩展，显著提升冻结主干网络在推理基准上的表现并具备零样本迁移能力。 |
| [^41] | [SpectRAG: Weighted Chunk Graphs for Token-Budgeted Context Selection in Multi-Hop Retrieval-Augmented Generation](https://arxiv.org/abs/2610.36991) | SpectRAG提出了一种谱图增强检索框架，通过构建编码多种语义与结构关系的加权分块图，结合谱相关性传播与Token预算约束下的上下文选择，解决传统独立检索在多跳推理中证据冗余和不完整的问题。 |
| [^42] | [The State and Fate of Multilingual, Contextual Evaluation in the NLP World](https://arxiv.org/abs/2610.36990) | 该论文对51个多语言NLP评估基准进行了数据驱动的系统性审计，揭示了多语言评估中覆盖率浅薄、地区严重缺失、低资源语言任务类别匮乏以及过度依赖英语翻译等结构性缺陷。 |
| [^43] | [AutoVend: Benchmarking LLM Agents on Long-Horizon Autonomous Vending Operations](https://arxiv.org/abs/2610.36980) | AutoVend是一个长周期基准测试，让LLM智能体通过网络界面管理分布在五类站点的20台自动售货机180个模拟日，评估其O2O商务决策能否转化为持续经济回报，结果显示五个智能体中有四个不敌基于规则的库存控制基线。 |
| [^44] | [Mining the Nineteenth Century for the Thirteenth: Register-Scale Forced Alignment and HTR Bootstrapping via Legacy Regesta and Antiquarian Editions](https://arxiv.org/abs/2610.36956) | 本研究利用十九世纪Pressutti教皇法令摘要等传统学术资源，通过“投射至教廷拼写并重放现代源文本行”的方法实现零标注的手写文本识别适应，使教皇洪诺留三世卷册第二年页面的词错误率下降1.50个百分点、缩小33%与理想参照的差距。 |
| [^45] | [Self-Keyed In-Parameter Memorization for External Memory-Free Dialogue Recall in Frozen Language Models](https://arxiv.org/abs/2610.36948) | 该论文提出SKIM闭式方法，让冻结语言模型为每个对话轮次自生成问题作为键写入模型参数中，从而在无外部存储的条件下仅凭参数即可回忆对话内容并回答后续提问。 |
| [^46] | [JevRec: Linguistic Encoding with Jev-like Decision Modeling for LLM-based Generative Recommendation](https://arxiv.org/abs/2610.36937) | 提出JevRec，借鉴自然语言中高频词采用紧凑表达的特点，利用频率感知的可变长度语言学编码生成物品语义标识符，并为不同优化目标设计类Jev决策建模，从而在降低热门物品编码开销的同时保留长尾物品的丰富语义。 |
| [^47] | [DocXEval: A Systematic Evaluation of Explainability in Multimodal Document Understanding](https://arxiv.org/abs/2610.36929) | 本文提出了首个面向多模态文档理解的综合性可解释性评估框架DocXEval，将现有XAI方法与指标适配到文本、布局和图像的多模态场景，并对13种方法进行了跨任务、数据集、模型和指标的系统性评估。 |
| [^48] | [TrawMem: Reframing Agent Memory as Task-Oriented Role-Adaptive Workspaces](https://arxiv.org/abs/2610.36918) | 该论文提出TrawMem，将智能体记忆系统重新构想为面向任务的角色自适应工作区，通过为记忆分配任务角色并联合选择互补记忆来组织交互历史，从而突破传统仅基于相关性的记忆检索范式。 |
| [^49] | [Do We Need an LLM to Know Why Sources Conflict? Decision Models for Knowledge-Conflict Typing in RAG](https://arxiv.org/abs/2610.36915) | 小型类型化决策模型在识别RAG中检索来源的冲突类型方面，可以以低4至39倍的成本媲美甚至超越Gemini 2.5 Flash等前沿大语言模型。 |
| [^50] | [Developing a Speech Language Model for Basque](https://arxiv.org/abs/2610.36911) | 本文提出了首个开源的巴斯克语语音语言模型，通过在LLaMA-Omni架构上微调Latxa模型并利用翻译语料与TTS合成音频，实现了低资源语言下的低延迟端到端口语交互。 |
| [^51] | [Nothing Left Behind: Finding Evidence from a Large Randomized Evaluation Corpus](https://arxiv.org/abs/2610.36910) | 该论文提出一种结合文本嵌入候选检索与语言模型证据过滤的两阶段方法，在7,354项随机试验语料库中找到了系统性综述应纳入证据的92%，远超传统标签搜索的45%。 |
| [^52] | [HiLo-RD: Cross-Budget Self-Distillation for Masked Diffusion Language Models](https://arxiv.org/abs/2610.36901) | 提出HiLo-RD跨预算自蒸馏方法，通过将大窗口教师状态的隐藏表示与裁剪后的小窗口视图进行匹配，解决掩码扩散语言模型在有限生成预算下因过早提交答案片段而导致的推理偏离问题。 |
| [^53] | [RubricAtlas: Mining Type-Level Rubrics from Expert Reports to Evaluate Deep Research Agents](https://arxiv.org/abs/2610.36878) | RubricAtlas通过从3,079份中文金融研究专家报告中挖掘类型级分析要求，构建覆盖99种研究类型的评估框架，将二元检查点与报告段落关联，并把报告完整性与论证有效性分开评估，从而更全面地评估深度研究智能体。 |
| [^54] | [GUI-JEPA: Candidate-Conditioned JEPA Latents as a Fast Action Policy for Mobile GUI Agents](https://arxiv.org/abs/2610.36877) | GUI-JEPA是首个无解码器的GUI智能体策略，利用在无标注GUI截图上预训练的JEPA编码器潜在空间，通过候选条件化的潜在预测器和轻量级打分头在一次批量处理中为所有候选动作打分，无需逐token解码，实现了快速且省内存的移动GUI操作。 |
| [^55] | [AdaGround: Token-Normalized Adaptive Zooming with Verifier-in-the-Loop Refinement for GUI Grounding](https://arxiv.org/abs/2610.36874) | AdaGround提出了一种免训练的GUI定位框架，通过依据目标大小自适应设定缩放参数、引入“是/否”验证器在环机制甄别并修正缩放结果、以及根据表头组合定位电子表格单元格，在高分辨率屏幕小目标定位上超越了现有最先进的免训练方法。 |
| [^56] | [Words or Deeds? How LLMs Predict Users' Decisions in Human-Agent Dialogue](https://arxiv.org/abs/2610.36813) | 该研究发现大语言模型从历史行为记录预测用户合作决策的能力（AUC=0.80）优于对话言语信息，加入对话反而使模型更轻信用户的口头承诺，导致错误集中在用户违背承诺的轮次。 |
| [^57] | [Weakness Localization Is Not Enough: State-Aware Recursive Knowledge Improvement for Large Language Models](https://arxiv.org/abs/2610.36811) | 该论文指出仅定位模型知识弱点不足以决定如何改进（存在“知识状态混叠”问题），并提出状态感知的递归知识改进方法，根据知识是缺失、难以访问还是下游无效等不同状态选择相应的干预策略。 |
| [^58] | [Latent Visual Contrastive Reasoning with Multi-Branch Exploration](https://arxiv.org/abs/2610.36809) | 提出了一种分支感知的潜在视觉推理框架，通过生成多个问题条件的潜在视觉分支，并利用边界框证据标注进行软对比的分支-证据对齐训练，突破单一线性推理轨迹的局限，提升复杂视觉推理能力。 |
| [^59] | [Personality Latents: Conditioning Frozen Language Models on Individuals from Their Own Writing](https://arxiv.org/abs/2610.36805) | 该论文提出“人格潜变量”（PL）方法，无需人格标签或自撰简介，直接从个人写作中为其学习潜变量并作为软前缀注入冻结语言模型，使模型能够精确表达该个体的独特人格。 |
| [^60] | [API-TAIL: Detecting Model Substitution in LLM APIs through Temperature-Aligned Likelihood Tails](https://arxiv.org/abs/2610.36801) | 提出API-TAIL典型性审计方法，通过温度对齐吸收合法解码配置造成的批次级偏移，再从似然尾部识别个别异常响应，从而检测LLM API服务中悄悄将部分请求替换为更廉价模型的行为。 |
| [^61] | [Language Models Keep Abandoned Analyses: Residue and Adaptation after Garden-Path Sentences](https://arxiv.org/abs/2610.36797) | 该研究发现，语言模型在处理花园路径句后会留下两种痕迹：被放弃句法分析的残留会提升后续宾语附着倾向（效应约为真实宾语句的四分之三），而向已消解结构的适应随消歧误差缩放并主要影响理解类任务，激活修补技术可将二者区分开来。 |
| [^62] | [Select-TTT: Query-Conditioned Selective Fast Weight Updates for In-Place Test-Time Training](https://arxiv.org/abs/2610.36794) | Select-TTT提出了一种模型无关的查询条件化规则，通过根据各数据块与用户查询的相关性来动态缩放其快权重写入强度，解决了原位测试时训练中统一写入调度导致无关内容稀释查询相关信息的问题。 |
| [^63] | [Batching the Reflective Optimization Loop: Parallel Proposals Make GEPA Faster and Better](https://arxiv.org/abs/2610.36774) | 将GEPA优化器扩展为每步并行提议和评估一批候选方案，使反思优化循环的运行时间缩短至原来的四分之一到二分之一，同时提升了测试分数和预算利用效率。 |
| [^64] | [DocSkill: Skill-Integrated Agentic Reasoning for Multilingual Multi-Hop Document Question Answering](https://arxiv.org/abs/2610.36759) | DocSkill提出了一个具有可学习技能编排策略的智能体框架，通过离线MCTS探索技能轨迹生成SFT示范与DPO偏好对来训练模型，实现自适应的证据获取、跨页整合与答案验证，并发布了多语言多跳文档问答基准MMHop-DocBench。 |
| [^65] | [CAMR: Compatibility-Aware Memory Replay for Continual Learning in Large Language Models](https://arxiv.org/abs/2610.36734) | CAMR 提出了一种兼容性感知的自适应记忆回放框架，通过在线策略联合控制新样本监督与历史回放，在持续学习中平衡大语言模型对新知识的获取与对已有知识的保留。 |
| [^66] | [A Harness Reducing over-Defensive Writing in Agents by Isolating Objections](https://arxiv.org/abs/2610.36717) | 该论文提出一种将审稿人、过滤器和写作者分离并通过独立上下文隔离异议的框架，使过度防御性写作减少96.61%，同时将源主张保留率从80.35%提升至93.12%。 |
| [^67] | [Is Token Confidence Just Chain Length? A Signal-Processing Audit of Reasoning Traces](https://arxiv.org/abs/2610.36709) | 本文用频谱分析和变点检测等信号处理工具审查发现，基于词元置信度滑动窗口最小值的思维链评分实际上主要反映链的长度，而仅按长度排序的效果与该置信度评分相当甚至更好。 |
| [^68] | [Specialization, Diversity, and Scaling in Child-Scale Language Models](https://arxiv.org/abs/2610.36679) | 在儿童尺度数据上训练的小型语言模型会对单个儿童的语言环境产生词汇和分布层面的专业化，多儿童数据训练虽能提升对汇总语言的预测，但对语法能力的提升并不明显。 |
| [^69] | [A Quantitative Approach to Asch Conformity in Frontier Large Language Models](https://arxiv.org/abs/2610.36668) | 该论文提出一种定量方法，通过对单个问题重复采样获得答案分布，并比较模型接触人为制造共识前后的分布差异，证实前沿大语言模型存在从众效应，并量化了从众发生的条件与程度。 |
| [^70] | [Japanese Discourse Structure Analysis Based on Segmented Discourse Representation Theory Using LLMs](https://arxiv.org/abs/2610.36667) | 本研究利用大语言模型翻译构建了日语SDRT语料库，证明了基于SDRT的话语解析具有跨语言适用性，并发现日语话语标记词影响甚微，模型预测主要依赖上下文话语信息。 |
| [^71] | [Balanced State Subspaces for Mamba-2 State Compression](https://arxiv.org/abs/2610.36617) | 论文发现 Mamba-2 的状态能量集中于递归累积的偏置方向，而输出实际读取的内容位于与其坐标不对齐的低维子空间，据此提出一种基于完整 Gramian 平衡截断的训练后压缩方法，使压缩直接针对写入-读取子空间而非单个坐标。 |
| [^72] | [Structured Maps for Landmark-Based Route Instructions with LLMs](https://arxiv.org/abs/2610.36582) | 本文提出用SVG格式将地图数据结构化为文本表示，使大语言模型在路线导航的地标选择任务中表现与图像输入相当或更优，并证明了小型本地模型经微调后也可胜任该任务。 |
| [^73] | [Decoupling Consistency and Conflict across MLLM Layers for Fake News Detection](https://arxiv.org/abs/2610.36569) | 提出MLCCD方法，利用多模态大语言模型不同层级所编码的知识，对图文一致性与冲突进行多层级动态解耦，从而更精准地捕捉稀疏冲突并提升多模态假新闻检测性能。 |
| [^74] | [QESChunker: A Single Objective Unifies Overlapping and Non-Overlapping Chunking for RAG](https://arxiv.org/abs/2610.36557) | 提出QESChunker，以统一的问题-证据效用最大化目标将RAG中的非重叠与重叠分块统一为有监督组合优化问题，非重叠情形可通过动态规划在多项式时间内精确求解。 |
| [^75] | [PRISM-BENCH: Diagnosing Grounded and Faithful Multimodal Reasoning Beyond Final-Answer Accuracy](https://arxiv.org/abs/2610.36554) | PRISM-BENCH是一个包含1000道专家标注问题的多模态推理基准，通过感知探测和轨迹对齐两种诊断方法，揭示前沿模型约85%的失败源于感知错误，并能独立于答案长度预测答案正确性。 |
| [^76] | [Agentic Data Sampling Framework for Multilingual ASR via Linguistic-Aware Error-Driven Optimization](https://arxiv.org/abs/2610.36538) | 该论文提出了一种将LLM智能体引入多语言ASR训练循环的智能体数据采样框架，通过从音系、形态和句法等语言学角度分析识别错误，自适应地选择不同层级的数据采样策略以优化模型训练。 |
| [^77] | [RoboAuto: Autonomous Robot Program Search through Multi-Level Physical Understanding](https://arxiv.org/abs/2610.36533) | RoboAuto 提出了一个在物体、场景和交互三个层次上显式组织物理理解的机器人程序搜索框架，利用分层知识库整合可执行技能与物体中心知识，使基于 LLM 的智能体能够更有效地搜索和改进机器人操作程序。 |
| [^78] | [ETC-Bench: A Multilingual Benchmark for Evaluation-Side Transcript Canonicalization](https://arxiv.org/abs/2610.36527) | 本文提出了 ETC-Bench，这是首个覆盖 12 种语言、181 种现象类型的最全面的评估侧转录文本规范化多语言基准，并提出了兼顾规范化准确性与语义保持的 SENA 评估指标。 |
| [^79] | [Causal Localization of Evidence Effects in Transformers](https://arxiv.org/abs/2610.36524) | 该论文提出受控证据聚合（CEA）与平衡效用因果选择（BUC）方法，在Pythia-410M模型中通过因果干预定位出驱动语言模型利用自然语言证据进行决策的关键注意力头集合，并提供了从有限样本恢复归因结果的理论保证。 |
| [^80] | [Benchmarking Task Takeover Across Agent Handoffs](https://arxiv.org/abs/2610.36523) | 该论文提出了HSC-Bench基准测试（含1,039个任务），用于评估智能体任务交接中接收模型恢复仓库证据与继续既有工作状态的能力，发现结构化交接状态能提升所有模型的证据恢复能力和大多数模型的任务继续性能。 |
| [^81] | [Folding Attention Biases into Keys Makes Compacted KV Caches Run on Standard Kernels](https://arxiv.org/abs/2610.36520) | 该论文提出一种闭式折叠方法，将注意力匹配压缩KV缓存时引入的logit偏置直接折叠进键向量本身，使压缩后的缓存无需修改即可在FlashAttention和vLLM等标准注意力内核上运行，且精度与原始AM方法基本相当。 |
| [^82] | [MedConstBench: Benchmarking How Constitutions Steer LLMs in Medical Ethics](https://arxiv.org/abs/2610.36513) | 该论文提出 MedConstBench，首个将部署“宪法”作为受控变量、围绕比彻姆四大伦理原则构建、并通过人工校准的多模型同行评审评分的医疗伦理基准，实现了对宪法如何引导大语言模型医疗伦理行为的系统分析。 |
| [^83] | [Do AI Scientists Know What's New?](https://arxiv.org/abs/2610.36511) | 该论文提出了一个结合时间真值的分析框架，从提案和论断两个层面评估AI科学发现中的新颖性检查，发现对非新颖工作拒绝率高的检查往往会误拒那些后来被人类研究者实现的新颖想法。 |
| [^84] | [AC-JEPA: Anchored Cross-Modal Joint Embedding Predictive Architecture for Visual-to-Symbolic Translation](https://arxiv.org/abs/2610.36493) | 该论文提出AC-JEPA，一种与监督微调兼容的即插即用模块，通过在跨模态两个方向上预测被掩码的中间状态的联合嵌入目标，来利用不完美的配对数据，从而在保留token级监督的同时提升视觉到符号跨模态翻译对目标序列错误的鲁棒性。 |
| [^85] | [A Heart Means Recursion: Exploring Visual Meaning and Communication in VLMs](https://arxiv.org/abs/2610.36449) | 研究发现视觉语言模型会被简单视觉符号隐性地引导行为（如心形符号引导模型采用递归解法），且模型间视觉交流存在“生成—解读”鸿沟——模型发送的直观视觉线索往往无法有效引导其他模型。 |
| [^86] | [CONCUR: Concordant Auxiliary Rewards for Multi-Task, Multi-Reward Reinforcement Learning](https://arxiv.org/abs/2610.36437) | 提出CONCUR优势估计器，通过将门控辅助奖励的分数约束为与主要奖励排序一致，解决多任务多奖励强化学习中辅助奖励干扰主要奖励排序的问题。 |
| [^87] | [No Tokenizer, Still a Gap: Cross-Lingual Fairness of Homoglyph and Invisible-Character Watermarking](https://arxiv.org/abs/2610.36432) | 该研究在24种文字系统的40种语言变体上系统评测了同形字与不可见字符水印，发现尽管这类编辑式水印完全不依赖分词器，但其容量仍存在高达3.3倍的跨语言差异且96%由文字系统决定，表明跨语言公平性差距依然存在。 |
| [^88] | [SOMA: A Benchmark for Fine-Grained Egocentric Behavior and Body-Motion Understanding](https://arxiv.org/abs/2610.36384) | 该论文提出了SOMA基准，通过1,632个经人工验证的问题从身体构型、运动动态和时间推理三个维度评估自我中心视频中的细粒度身体运动理解能力，发现37个领先的MLLM表现不佳，甚至落后于同样认为该任务困难的人类。 |
| [^89] | [AK_QMUL at ArGuard Shared Tasks: A Single Fine-Tuned JAIS for Arabic Harmful-Prompt Detection](https://arxiv.org/abs/2610.36360) | AK_QMUL团队仅用一个经4位QLoRA微调的JAIS-6.7B模型，在ArGuard-2026阿拉伯语有害提示检测共享任务的两个赛道（二分类和七分类）上均获得第一名，宏F1分别达到0.984和0.790。 |
| [^90] | [Detect Before You Disambiguate: Perturbation-Based Input Ambiguity Detection in LLMs](https://arxiv.org/abs/2610.36353) | 本文提出一种保持标签的扰动框架，通过黑盒规则扰动和白盒重要性引导两种检测器，在LLM对输入消歧之前检测输入歧义，在开放域、对话和视觉问答数据集上均优于现有的输出侧不确定性方法。 |
| [^91] | [Cross-modal Steering Vectors Exist Across Both Image and Text Generation](https://arxiv.org/abs/2610.36269) | 该论文发现在统一多模态模型中，针对某一模态（图像或文本）找到的因果性转向向量能够零样本迁移到另一模态并有效引导生成，证明模型中的概念机制在图像与文本两种模态间是统一实现的。 |
| [^92] | [Beyond IceBERT: Open-Data ModernBERT Encoders for Icelandic from 12M to 1B Parameters](https://arxiv.org/abs/2610.36230) | 该研究基于90亿token开放数据训练了从12M到1B参数的ModernBERT冰岛语编码器，其1B模型是首个该规模的单语冰岛语编码器，在全部11个任务上显著超越IceBERT，并在受控实验中验证了从英语及多语言模型热启动的迁移效果。 |
| [^93] | [Base Models Can Detect Hidden Intentions that Prompted Classifiers Miss](https://arxiv.org/abs/2610.36202) | 本文提出人格-头部似然评估（PHLE）方法，利用基础预训练模型对输出的似然评估来解码并分类后训练模型中的潜意识隐蔽信号，从而检测出提示分类器无法发现的隐藏意图（如隐蔽的伤害用户倾向）。 |
| [^94] | [Learning to Read Braille: Benchmarking and Improving Braille Recognition Capabilities of VLMs](https://arxiv.org/abs/2610.36195) | 该论文提出盲文识别基准BrailleBench与合成盲文生成流水线，证明在其合成数据上训练的2B参数视觉语言模型可将一级盲文识别精确匹配率提升至87%、二级盲文提升至61%，超越现有独立视觉语言模型并在二级盲文上取得最佳成绩。 |
| [^95] | [LLM Hallucination Detection via Low-rank Multilinear Decomposition](https://arxiv.org/abs/2610.36191) | 提出LOTUS框架，利用多线性低秩分解从大语言模型隐藏状态中提取紧凑表示，实现幻觉检测并减少对标注数据的依赖。 |
| [^96] | [Frankly Speaking: The Company Election Integrity Keeps in Congressional Communication](https://arxiv.org/abs/2610.36187) | 本研究利用历时词嵌入方法分析2010至2025年间的10,420份国会电子简报，发现民主党和共和党虽然使用相同的“选举诚信”词汇，却将其置于截然不同的语义语境中，从而用同样的词讲述不同的政治故事。 |
| [^97] | [Lipi-Ghor: A Large-Scale, Diarized, Multi-Domain Bengali Speech Corpus and a Scalable Pipeline for Building It](https://arxiv.org/abs/2610.36184) | 该论文推出了Lipi-Ghor——首个拥有885小时音频、带说话人分离标注的多领域孟加拉语语音语料库，并提供了一套可复制扩展的语料库构建流水线。 |
| [^98] | [3, 7, 14, 17, 42: Instruction Tuning Installs and Sharpens Number-Choice Heuristics in Language Models](https://arxiv.org/abs/2610.36178) | 该研究通过精确计算而非抽样估计的方法测量语言模型生成1到N随机数的完整概率分布，发现指令微调会安装“回避整数”并强化“回避端点”两种数字选择启发式，使分布更加不均匀。 |
| [^99] | [Don't Pick a Winner: Knowledge Conflicts Need a Prior, Not a Rule](https://arxiv.org/abs/2610.36169) | 论文证明，检索增强语言模型中检索上下文与内部知识的冲突无法靠设计时固定的规则（如一律信上下文或信模型）最优解决，且自适应解码依赖的对称散度不包含判断依据，正确做法是结合分别评估各来源的证据与上下文可靠性的先验。 |
| [^100] | [ConformalMOS: Uncertainty-Aware MOS Prediction with Conformal Intervals and Ordinal Modeling](https://arxiv.org/abs/2610.36126) | 提出ConformalMOS框架，结合保形预测生成具有覆盖率保证的预测区间，并采用序数感知的高斯平滑标签建模，在保持MOS预测准确度的同时实现可靠的不确定性量化。 |
| [^101] | [Beyond Update Counts: A Control Map for Evaluating Test-Time Model Updates](https://arxiv.org/abs/2610.36109) | 提出控制图评估框架，通过记录每次作答前的完整更新历史，分别评估即时学习、记忆保持、局部性与新环境泛化，从而超越仅凭更新次数比较测试时模型更新的局限。 |
| [^102] | [FinLLM-KG: Knowledge Graph-Enhanced LLMs for Financial Sentiment Analysis in Algorithmic Trading](https://arxiv.org/abs/2610.36083) | 该论文提出FinLLM-KG框架，通过将因果大语言模型与金融知识图谱结合，在推理时注入来自SEC 10-K文件的公司层面知识，以近乎零计算成本提升了金融情感分析的上下文理解能力，且无需更新模型权重。 |
| [^103] | [How Much Does the Hint Matter? Scope Conditions for Hindsight Self-Distillation in Financial Reasoning](https://arxiv.org/abs/2610.36082) | 提出Fin-SD方法，通过让学生模型自己发现并批评首次错误生成提示，再以错误前缀加提示的条件进行自蒸馏，从而在无需外部教师或完整参考解答的情况下提升小模型的多步金融推理能力。 |
| [^104] | [OmniBrain-0: A Unified Agentic Model for Embodied Intelligence](https://arxiv.org/abs/2610.36013) | OmniBrain-0是一个轻量级统一具身智能体，通过具身预训练、领域专用智能体蒸馏和多教师在线策略蒸馏（MOPD）的渐进式训练框架，利用共享的高层决策模型协调操作与导航，实现任务理解、规划、进度跟踪和失败恢复。 |
| [^105] | [MedJev: Evaluating Zero-Shot System-One Decision Making in Medicine](https://arxiv.org/abs/2610.36006) | MedJev是一个可在笔记本电脑上本地运行的开源“系统一”医学决策模型，在多项医学知识和诊断基准上接近GPT-4的性能，同时保持良好的置信度校准，为快速高效的临床决策支持提供了新方案。 |
| [^106] | [SCOPE: Source Credibility and Outlet Profiling Engine](https://arxiv.org/abs/2610.35929) | 本文提出SCOPE多智能体系统，通过协调文章获取、内容分析、证据搜索等专门智能体，基于证据对新闻媒体的偏见、事实性和可信度进行透明评估，在200家媒体基准上超越了多种基线方法。 |
| [^107] | [ReCoPII: A Relation-Grounded Framework for Building Contextual PII Masking Datasets](https://arxiv.org/abs/2610.35925) | ReCoPII框架通过规划本体约束的实体关系并据此推导MASK/KEEP标签，无需单一数据主体假设即可自动构建情境化PII脱敏数据集，所生成的韩文语料ReCoPII-Ko的KEEP标签获得88.42%的标注者认可率，远超SPY方法的26.27%。 |
| [^108] | [Physiognomy Revisited: How Vision-Language Models Encode Sensitive Attributes Through Image Transformation](https://arxiv.org/abs/2610.35910) | 本文揭示了视觉语言模型在图像变换任务中会遵从按种族和性取向改变人脸的请求，生成统计上可区分且强化刻板印象的图像，并且这些表征偏差还会在下游文本推理任务中重现。 |
| [^109] | [Understanding Jev-Style System One Models: Does Structure Reduce Selection Bias?](https://arxiv.org/abs/2610.35826) | 本文通过在保持问题内容不变的前提下打乱选项顺序的简单测试，系统比较了两个Jev版本、十二个Jev复现模型和六个生成式LLM在多选项任务中的选择偏差，探究Jev式决策结构能否减少选择偏差。 |
| [^110] | [Quantitative Generalization Bounds for Multi-Layer Transformers: A Few Long Strings can Beat Many Short Ones](https://arxiv.org/abs/2610.35715) | 本文首次对固定精度多层Transformer的分布内统计泛化与长度泛化进行了联合的定量理论分析，并揭示出一个反直觉的结论：用少数长字符串训练可以胜过用大量短字符串训练。 |
| [^111] | [Do Language Models Understand Igbo Culture? Beyond Multiple Choice](https://arxiv.org/abs/2610.35168) | 本文提出了基于伊博本土文化构建并由母语者验证的CAB-Igbo基准测试，通过多选题与人类评估相结合的方式，揭示了语言模型的流畅性与文化知识之间存在严重脱节，且多数模型在英文中就已缺失认知忠实度和表征丰富度。 |
| [^112] | [Ask Before You Answer: A Question-Driven Evidence Reasoning Architecture (QERA) for In-Context Learning](https://arxiv.org/abs/2610.35083) | 该论文提出QERA架构，通过先将用户请求转化为需求清单并筛选出隐含信息需求，再从完整对话中作答并用代码验证引文，从而解决大语言模型在真实生活上下文学习任务中因请求隐含大量未明说标准而表现不佳的问题。 |
| [^113] | [Diagnosing and Mitigating Long-Session Degradation in Full-Duplex Speech Models](https://arxiv.org/abs/2610.34991) | 本文提出一个闭环评估框架系统性诊断全双工语音模型在长会话中的性能退化，并提出基于自蒸馏的轻量级LoRA适配训练方法来有效缓解该退化。 |
| [^114] | [FATE-5: Faithful Alignment of Trait Expression for Continuous Big Five Personality Control](https://arxiv.org/abs/2610.34889) | FATE-5提出两阶段框架，将大五人格编码与自然度对齐分离，实现了对大语言模型人格的连续控制，在保持强特质表达的同时显著提升了对话自然度。 |
| [^115] | [A Benchmark for Controllable Speaking-Style Captioning in Audio-Language Models](https://arxiv.org/abs/2610.33036) | 提出StyleInstructCaps——首个可控说话风格描述的标准化基准，将说话风格描述形式化为指令跟随的音频-语言任务，通过明确指令指定需描述的风格维度（如说话人特质、情感、表达风格）。 |
| [^116] | [How Much Do You Trust A Dog? Quantifying Hint-Following in Reasoning Language Models](https://arxiv.org/abs/2610.32055) | 该研究发现大语言模型对提示来源（如教授、朋友甚至狗）的答案偏移遵循可预测的指数倾斜结构，且偏移强度与来源可靠性基本无关，揭示了模型谄媚行为未被来源可靠性校准的问题。 |
| [^117] | [Knowing When to Stop: The Evidence-Action Gap in Coding Agents](https://arxiv.org/abs/2610.31895) | 该论文揭示了编码智能体的“证据-行动差距”现象——即智能体即使已收集到表明任务无法独立完成的充分证据仍过度坚持或虚报成功，并提出ImpasseBench基准通过在可解决任务中注入环境阻碍并标记证据充分步骤来量化这一差距。 |
| [^118] | [CulturalRiddles: A Multicultural Benchmark of Riddles](https://arxiv.org/abs/2610.29968) | CulturalRiddles是一个涵盖62种文化、完全由人类创作的谜语基准测试，用于评估大语言模型的跨文化理解与推理能力，结果表明即使是前沿模型在某些社区中的准确率也最高仅达22%。 |
| [^119] | [Where Does Rhetorical Distortion Live? A Multi-Platform, Multi-Language Measurement of Influencer Content](https://arxiv.org/abs/2610.26412) | 本文提出一套在六个平台、两种语言上统一应用的修辞失真测量框架，通过双语词汇规则、负面过滤器与大语言模型验证构成的三层级流水线，检测五种修辞操纵维度，并将创作者帖子与回复分开分类，以定位此类框架化表述的真正来源。 |
| [^120] | [Bridged Self-Distillation for On-Policy Learning](https://arxiv.org/abs/2610.26397) | 提出桥接自蒸馏框架，通过让学生模型以教师推理轨迹为条件的副本充当自教师、在学生自身的词元空间中对学生的采样序列进行评分，从而统一解决了在线策略蒸馏中教师解答难以模仿与词元级概率不兼容或不可获取的两大难题。 |
| [^121] | [BanglaBBQ: A Bias Benchmark for Question Answering in the Bangladeshi Context](https://arxiv.org/abs/2610.24764) | 本文提出了 BanglaBBQ——一个针对孟加拉国语境、同时支持孟加拉语和英语的问答偏见基准测试，通过改编原有 BBQ 模板并新增五个本地化刻板印象类别，经本土标注者严格验证构建而成。 |
| [^122] | [Annotrace: A Workbench for Step-Level Annotation of Agent Traces](https://arxiv.org/abs/2610.23284) | Annotrace 是一个支持六种主流轨迹格式的智能体轨迹步骤级标注工作台，它将运行记录解析为有序嵌套的步骤，通过人机协作方式进行标注并衡量标注者间一致性，在多智能体问答语料上精确复现了此前人工标注的235个标注单元。 |
| [^123] | [Agaram 2.5B: The First Tamil-First Language Model Trained from Scratch with Tool Calling. A Playbook for Under-Represented Languages](https://arxiv.org/abs/2610.18420) | 该研究从零训练了首个泰米尔语优先、支持工具调用的25亿参数语言模型Agaram 2.5B，仅用8张H100 GPU、6天时间和约500亿token，就在翻译和泰米尔语指令遵循上超越了更大的印度语系模型Sarvam-1，为低资源语言的模型训练提供了可复制的实践方案。 |
| [^124] | [Attention-Graph Curvature for the Verification of Mathematical Reasoning](https://arxiv.org/abs/2610.16709) | 本研究提出利用语言模型注意力图的有向Forman曲率作为推理步骤正确性的信号，实验表明将该曲率附加到逐步验证器的输入中可显著提升其在ProcessBench上的数学推理验证性能。 |
| [^125] | [Denoising Hierarchical Representations: Joint Continuous Diffusion for Language Modeling](https://arxiv.org/abs/2610.08738) | 提出层次化连续扩散语言模型（H-CDLM），通过并行扩散token本身及其粗粒度语义聚类等多种模态表示，并允许为各模态配置独立的采样器与调度，以极少的计算和参数开销显著提升连续扩散语言模型的性能。 |
| [^126] | [Lost in the bf16 Cast: Exporting Ternary Language Models Can Revert Most Low-Learning-Rate Code Changes](https://arxiv.org/abs/2610.07853) | 该研究审计发现，三值语言模型导出流程中先将潜在权重转换为bf16，会在阈值处因“舍入到偶数”规则被错误映射为零，导致Falcon-E-1B-Base部署后GSM8K准确率从58.79%暴跌至0.78%，几乎抹去了低学习率微调带来的改进。 |
| [^127] | [The Score Is Not the Structure: Brain Alignment and Cross-Lingual Transfer](https://arxiv.org/abs/2610.03827) | 相似性分数可能反映的是测量工具本身的局限而非模型与大脑或跨语言间的真实共享结构，当测量工具在部分条件下失效或统计单位选择改变时，原本显著的梯度效应会大幅减弱甚至消失。 |
| [^128] | [Pair Difficulty Matters: Rethinking Pairwise LLM-as-a-Judge Evaluation and Consistency](https://arxiv.org/abs/2609.37577) | 该论文指出，传统用于评估LLM裁判可靠性的三大代理指标（位置偏差、传递性、成对一致性）具有误导性——在Bradley-Terry几何下它们被排名差距接近的配对所主导，而此类配对出现不一致在信息论上是必然的，因此不应以这些指标否定能力出色的LLM评估器。 |
| [^129] | [EvoIn: Bridging Evolution and Internalization for Agent Fine-Tuning](https://arxiv.org/abs/2609.35290) | EvoIn提出了一个连接演化与内化的智能体微调框架，通过分析执行轨迹来演化并验证新的决策流程，再将改进的推理轨迹改写为去除框架指令引用的自包含推理并微调模型，从而将决策逻辑内化为模型自身的能力。 |
| [^130] | [ReMCTS: Reflection-Enhanced Monte Carlo Tree Search for Code Generation](https://arxiv.org/abs/2609.34717) | ReMCTS提出了一种以执行结果为依据、结合记忆增强与LLM引导的蒙特卡洛树搜索框架，通过保留分支调试上下文、跨分支检索失败经验并区分检查失败与证据缺失，显著提升了代码生成的准确性与修复效率。 |
| [^131] | [Learning an Anchored Prompt Space for Continual Adaptation of Large Language Models](https://arxiv.org/abs/2609.32499) | 提出LAPS方法，通过自蒸馏对齐历史提示，并构建以任务特定软提示为顶点、贝塞尔控制提示塑造几何结构的锚定提示空间，从而在持续适配大语言模型时保留提示有效性并促进跨任务正向迁移。 |
| [^132] | [What Does an Agentic Software Engineering Benchmark Measure? Profiling Task Demands and Agent Behaviour Beyond What Category Labels Reveal](https://arxiv.org/abs/2609.01271) | 本文提出Spread-Novelty-Centrality（SNC）三轴画像方法来刻画仓库级编码任务的真实需求，发现类别标签是任务需求的不可靠代理指标，且智能体行为轨迹能揭示人工标准答案无法反映的任务需求信息。 |
| [^133] | [Synthetic Worlds for Temporal Evaluation and Knowledge Updating in LLMs](https://arxiv.org/abs/2609.00184) | 该论文提出了一个模拟驱动的合成框架，通过虚构未来世界的 ParallelEvents 基准避免评估污染，并利用 Synapse 训练框架（结合中期训练与指令微调）实现大语言模型的可扩展知识更新，性能比现有方法提升 14.23%。 |
| [^134] | [Validating FKG.in: Soundness Assessment in LLM-Augmented Indian Food Knowledge](https://arxiv.org/abs/2608.29249) | 本文作为印度食品知识图谱FKG.in的一部分，提出了一种半自动化的健全性评估工作流程，通过结合形式文法、词汇检查、统计启发式、Set Transformer连贯性建模和检索验证的多阶段方法，识别并解决LLM从非正式烹饪来源提取和增强结构化食谱数据时的常见失败模式。 |
| [^135] | [Kernel Token Contradiction: a Fast and Principled Approach for LLM Claim Uncertainty Quantification](https://arxiv.org/abs/2608.22506) | 本文提出了一种轻量级且快速的声明级不确定性量化方法KTC，通过核表示和冯·诺依曼熵，在CPU上实现了比现有方法显著的速度提升。 |
| [^136] | [Machine learning and digital pragmatics: Which word category influences emoji use most?](https://arxiv.org/abs/2608.21975) | 本研究通过MARBERT模型和逻辑回归分析发现，在口语阿拉伯语社交媒体帖子中，动词类别对表情符号使用的影响最强，尽管名词在频率上占主导。 |
| [^137] | [TextCloak: Safeguarding Data from Unauthorized LLM Fine-Tuning via RL-Driven Unlearnable Examples](https://arxiv.org/abs/2607.28862) | TextCloak提出了一种基于强化学习的数据合成框架，通过生成既能保持语义忠实性和语言自然性又能降低模型学习效果的不可学习样本，来保护文本数据免受未经授权的大语言模型微调。 |
| [^138] | [Pruning via Causal Attribution Preserves Reasoning Performance in Large Language Models](https://arxiv.org/abs/2606.19350) | 提出无需训练的因果归因剪枝方法 CAP，通过干预式测量注意力头对推理任务的因果贡献来指导细粒度权重剪枝，在 20% 稀疏度下相比 Wanda 在 ARC-Challenge 上获得高达 61% 的相对准确率提升，从而在降低推理成本的同时有效保持大语言模型的推理能力。 |
| [^139] | [Spatio-Temporal Audio Language Modeling for Dynamic Sound Sources](https://arxiv.org/abs/2606.14141) | 该论文提出了基于FOA渲染的时空音频问答数据集ST-AudioQA及配套的ST-Audio编码器和ST-AudioLM模型，联合建模声源语义与随时间变化的声源轨迹，使大语言模型能够回答关于声源身份、位置和运动的时空音频问答问题。 |
| [^140] | [LLMs Can Leak Training Data But Do They Want To? Evaluating Memorization Propensity in LLMs](https://arxiv.org/abs/2606.06286) | 本文提出PropMe框架和SimpleTrace工具，首次将大语言模型“能否”泄露训练数据与“是否会主动”泄露训练数据区分开来，发现自然句子提示下的主动泄露远少于前缀攻击，但无提示或常见词提示时的泄露程度与前缀攻击相当。 |
| [^141] | [Streaming Communication in Multi-Agent Reasoning](https://arxiv.org/abs/2606.05158) | StreamMA通过让智能体即时流式传递每个推理步骤实现流水线化，首次证明了流式通信既能降低多智能体推理延迟、又能提升推理效果，并给出了流式、串行与单一协议的首个闭式联合理论分析。 |
| [^142] | [Agent Planning Benchmark: A Diagnostic Framework for Planning Capabilities in LLM Agents](https://arxiv.org/abs/2606.04874) | APB是一个包含4,209个多模态案例的智能体规划专用诊断基准，能够定位大语言模型智能体在长程规划、工具噪声鲁棒性等方面的系统性弱点，且其引导的改进可持续提升规划质量与下游执行表现。 |
| [^143] | [HEART-Bench: Do LLMs Make Persona-Consistent Decisions?](https://arxiv.org/abs/2605.30058) | 提出HEART-Bench基准，通过11个拥有自传体记忆的大五人格合成角色和64个决策场景构成的673道多选题，评估16个大语言模型能否在检索记忆的辅助下做出与人设一致且基于记忆的决策。 |
| [^144] | [Does The Way You Plan Matter? An Empirical Study of Planning Representations for LLM Web Agents](https://arxiv.org/abs/2605.29927) | 本文提出PlanAhead框架，系统评估了四种自然语言计划表示（顺序子目标、叙述、伪代码、检查清单）对多模态LLM网页代理性能的影响，并实现了无需人工标注的自动任务难度分级。 |
| [^145] | [PubMedCausal: A Span-Level Annotated Corpus for Causal Relation Extraction in Biomedical Text](https://arxiv.org/abs/2605.28363) | 本文构建了基于PubMed摘要的跨度级因果标注语料库PubMedCausal，通过完整的因果-结果跨度、因果类型与句级性标注，实现了对因果检测与全跨度因果抽取更贴近真实文本表达的评估。 |
| [^146] | [Beyond Questions: Evaluating the Stated Knowledge of LLMs](https://arxiv.org/abs/2605.26937) | 该论文提出了开放式知识评估这一新范式，并构建了包含10,000个实体的BeQu基准，通过开放式引导提示评估大语言模型自然表达的知识，从而克服了传统预定义问题评测中的可用性偏差。 |
| [^147] | [Word-Class and Construction-Like Structure Emerges in Neural Successor Representations Trained on Natural Language](https://arxiv.org/abs/2605.24585) | 在自然语言上训练的神经后继表示无需任何词性监督，即可自发涌现出词类边界以及跨越传统类别界限的构式类语言结构。 |
| [^148] | [Strong Teacher Not Needed? On Distillation in LLM Pretraining](https://arxiv.org/abs/2605.23857) | 大语言模型预训练中的知识蒸馏并不需要强大的教师——即使是小型或训练不足的教师，在恰当混合两种损失后也能有效提升更大规模学生模型的性能，而过强的教师反而可能使蒸馏收益饱和或逆转，且蒸馏更利于提升泛化能力而非域内拟合。 |
| [^149] | [Semantic Reranking at Inference Time for Hard Examples in Rhetorical Role Labeling](https://arxiv.org/abs/2605.18007) | 提出RISE推理时语义重排序框架，利用对比学习得到的标签表示自动识别并重排序低置信度难例的预测，在八个修辞角色标注数据集上平均提升9.15个macro-F1点，且无需重新训练模型。 |
| [^150] | [The Geometry of Learning: Read-Write Asymmetry Across Training](https://arxiv.org/abs/2605.16600) | 该论文以残差流为共享坐标系分析预训练与对齐阶段对 transformer 权重的更新方向，揭示了“读写不对称性”：读侧矩阵（query/key）的更新集中于其输入激活的主子空间，而写侧矩阵（输出矩阵）的更新相对于预测子空间接近随机，且这一特征是训练阶段的属性而非训练目标的属性。 |
| [^151] | [Rethinking State Tracking in Recurrent Models Through Error Control Dynamics](https://arxiv.org/abs/2605.07755) | 本文证明了仿射循环网络（如状态空间模型和线性注意力）一旦保留状态表示就无法沿状态分离子空间纠正误差，因而其状态追踪只能依靠累积误差下的有限视界解，而非鲁棒的状态追踪。 |
| [^152] | [ShredBench: Evaluating the Semantic Reasoning Capabilities of Multimodal LLMs in Document Reconstruction](https://arxiv.org/abs/2604.23813) | ShredBench是一个由自动化流程直接从Markdown生成碎片化文档的新基准，用于系统评估多模态大语言模型在碎片文档重建任务中结合视觉识别与语义推理的能力，并可通过引入新文本源防止训练数据污染。 |
| [^153] | [Scoring-Rule Sensitivity in LLM Evaluation: Evidence from Indian Financial Regulatory Text](https://arxiv.org/abs/2604.19298) | 该研究构建了基于印度SEBI和RBI监管文本的专家标注基准IndiaFinBench，发现大语言模型的排名对评分规则高度敏感——严格字符串匹配与跨模型裁判两种评分机制之间甚至无显著的正向排名相关性——表明单一准确率数字是模型与评分规则的共同产物，不能可靠反映模型真实能力。 |
| [^154] | [Latent Abstraction for Retrieval-Augmented Generation](https://arxiv.org/abs/2604.17866) | 该论文提出LAnR统一框架，让单个LLM通过指定[PRED]标记的隐藏状态直接生成密集检索向量，在其自身潜在空间内联合完成编码、检索与生成，从而摆脱了传统RAG中生成文本查询及检索器与生成器分离的架构限制。 |
| [^155] | [Clinical Note Bloat Reduction for Efficient LLM Use](https://arxiv.org/abs/2604.16364) | TRACE 方法利用 EHR 元数据（或基于频率的去重）识别并清除临床记录中模板化和复制粘贴的冗余内容，在 530 万份记录和多项信息抽取及预测任务上验证了其能在保留真实临床信号的同时大幅压缩文本量，从而降低大语言模型的使用成本并提升效率。 |
| [^156] | [Who Gets Flagged? The Pluralistic Evaluation Gap in AI Content Watermarking](https://arxiv.org/abs/2604.13776) | 该论文揭示了AI内容水印技术因依赖内容统计特性，可能对不同语言、文化和人群产生系统性检测偏见，并提出了包含跨语言检测公平性、文化多样化内容覆盖和人口统计分组检测指标三个维度的多元化水印评估框架。 |
| [^157] | [Decoupling and Regulating Modality Heads for Fake News Detection under Missing Modality](https://arxiv.org/abs/2604.09711) | 提出HMDR框架，通过对多模态大语言模型中的注意力头进行模态解耦、贡献调控与模态不变性学习，解决缺失模态场景下假新闻检测的问题，尤其应对复用真实材料导致标签失效的挑战。 |
| [^158] | [Scaling Reasoning Efficiently via Relaxed On-Policy Distillation](https://arxiv.org/abs/2603.11137) | 提出REOPOLD框架，通过控制教师反馈的应用位置、影响强度和分配规则的变化时机，解决了在线策略蒸馏中统一信用分配导致的熵坍缩与负迁移问题，从而高效提升容量受限模型的推理能力。 |
| [^159] | [Reinforced Fast Weights via Next-Sequence Prediction](https://arxiv.org/abs/2602.16704) | 本文提出下一序列预测（NSP）目标及ReFINE方法，利用强化学习在序列级别优化快速权重模型，解决了传统下一词元预测目标与循环记忆状态之间的训练不匹配问题。 |
| [^160] | [Every Little Helps: Building Knowledge Graph Foundation Model with Fine-grained Transferable Multi-modal Tokens](https://arxiv.org/abs/2602.15896) | 提出基于词元的知识图谱基础模型TOFU，将结构、视觉和文本信息离散化为模态特定的细粒度可迁移词元，并通过消息混合的分层融合架构处理这些词元，在17个直推式、归纳式和完全归纳式多模态知识图谱上实现了强大的跨图谱泛化能力。 |
| [^161] | [Evaluating Geographic Lexical Knowledge in Large Language Models Across 21 Spanish-Speaking Countries](https://arxiv.org/abs/2602.09346) | 该研究将十个LLM视为虚拟发音人，基于涵盖21个西班牙语国家934个词汇的专家数据库，系统评估了LLM对西班牙语地理词汇差异的知识，发现模型间、国家间和题型间均存在显著差异，其中西班牙得分最高而智利在选择题格式中得分最低。 |
| [^162] | [OdysseyArena: Benchmarking Large Language Models For Long-Horizon, Active and Inductive Interactions](https://arxiv.org/abs/2602.05843) | 提出OdysseyArena基准，通过形式化四个基元将智能体评估重新聚焦于长时程、主动和归纳式交互，并建立包含120个任务的OdysseyArena-Lite来衡量智能体从经验中自主发现潜在转换规律的能力。 |
| [^163] | [AzeroS: Extending LLM to Speech with Self-Generated Instruction-Free Tuning](https://arxiv.org/abs/2601.06086) | 提出SIFT（自生成无指令微调）新范式，由冻结的LLM以语音的文本表示为输入生成监督信号来训练语音LLM，从而无需收集特定任务问答数据即可在理论上实现对未见任务的最佳泛化。 |
| [^164] | [SkipCat: Rank-Maximized Low-Rank Compression of Large Language Models via Shared Projection and Block Skipping](https://arxiv.org/abs/2512.13494) | SkipCat是一种新颖的大语言模型低秩压缩框架，通过共享投影和块跳过机制，在相同压缩率下保留更高的秩，从而避免了传统低秩压缩中因秩过度削减而导致的显著性能下降。 |
| [^165] | [Counting Clues: A Lightweight Probabilistic Baseline Can Match an LLM](https://arxiv.org/abs/2512.12868) | 本研究提出基于频率的概率排序器（FBPR），仅利用朴素贝叶斯与大语料库中的概念-诊断共现统计，即可在MedQA诊断任务上达到与LLM相当的性能，且两者互补性强，混合使用可获得更佳效果。 |
| [^166] | [SELF: A Robust Singular Value and Eigenvalue Approach for LLM Fingerprinting](https://arxiv.org/abs/2512.03620) | SELF是一种基于LLM注意力权重奇异值与特征值分解的内在指纹识别方案，无需输入依赖即可抵御虚假声明攻击和权重操纵，实现鲁棒且可扩展的模型知识产权取证。 |
| [^167] | [Cross-User Poisoning: User-Task Boundary Failures in Multi-User Collaborative Language Agents](https://arxiv.org/abs/2511.17671) | 该论文首次识别出多用户共享语言代理中的“跨用户投毒”（CUP）攻击，即攻击者注入共享上下文的恶意消息会在代理服务其他无辜用户时被错误执行，并提出 MURMUR 评估框架，在 Continua、ElizaOS 等真实系统中验证了该攻击的高成功率与持久性。 |
| [^168] | [From the Rock Floor to the Cloud: A Systematic Survey of State-of-the-Art NLP in Battery Life Cycle](https://arxiv.org/abs/2510.27369) | 该论文系统综述了自然语言处理在电池全生命周期各阶段的应用，并提出了一种面向欧盟数字电池护照（DBP）的新型技术语言处理（TLP）概念框架。 |
| [^169] | [Uncertainty-Guided Multimodal Consensus for Multimodal Sentiment Analysis](https://arxiv.org/abs/2510.20256) | 提出了不确定性引导的多模态共识方法UMC，通过连续原型引导的标签恢复（CPLR）和可靠性与精确度共识（RPC）两个组件，解决了多模态情感分析中共享标签掩盖模态特定情感以及模态不一致时不可靠预测被过度强调的问题。 |
| [^170] | [FullFront: Benchmarking MLLMs Across the Full Front-End Engineering Workflow](https://arxiv.org/abs/2505.17399) | FullFront是一个新的基准测试，通过网页设计、网页感知问答和网页代码生成三项任务，全面评估多模态大语言模型在前端工程全流程中的能力。 |
| [^171] | [EffiBench-X: A Multi-Language Benchmark for Measuring Efficiency of LLM-Generated Code](https://arxiv.org/abs/2505.13004) | 本文提出首个多语言代码效率基准EffiBench-X，评估发现LLM生成的代码虽然功能正确，但平均效率仅约为人类专家的62%，且在不同编程语言之间存在显著的效率差异。 |
| [^172] | [An Embarrassingly Simple Data Selection Strategy for Million-Scale Multi-Modal Instruction-Tuning](https://arxiv.org/abs/2503.13383) | 该论文提出一种极其简单的百万规模多模态指令微调数据选择策略，以交互风格衡量结构多样性、按能力维度评估数据质量，无需计算复杂的嵌入或梯度即可高效筛选高质量指令数据。 |
| [^173] | [DYNASHIELD: A Black-Box Moving Target Defense for LLMs via Dynamic Decoding Customization](https://arxiv.org/abs/2412.07672) | DYNASHIELD通过动态定制解码参数和系统提示，在无需访问模型内部或额外训练的情况下，有效降低了黑盒LLM面对越狱攻击的成功率。 |
| [^174] | [HateSense: Hierarchical Sentiment Analysis Framework for Hate Speech Detection: Implementing Binary and Multiclass Classification Strategy](https://arxiv.org/abs/2411.05819) | 该研究通过受控实验发现，分层分类和情感特征在仇恨言论检测中并未带来显著收益，而类别加权训练才是唯一能大幅提升性能的有效干预措施。 |
| [^175] | [A Unified Approach to Routing and Cascading for LLMs](https://arxiv.org/abs/2410.10347) | 本文提出“级联路由”这一统一理论框架，将路由与级联两种模型选择范式相结合，推导出了最优级联策略并证明了现有路由策略的最优性，从而有效改善大语言模型选择的成本-性能权衡。 |
| [^176] | [Fairness Definitions in Language Models Explained](https://arxiv.org/abs/2407.18454) | 本文通过系统性综述澄清了语言模型中各种公平性定义的内涵，解决了不同公平性概念之间区分模糊、适用情境不明确的问题。 |

# 详细

[^1]: 安全之头：学习型注意力门控提升提示注入鲁棒性

    Heads of Security: Learned Attention Gates Improve Prompt Injection Robustness

    [https://arxiv.org/abs/2610.37657](https://arxiv.org/abs/2610.37657)

    提出Heads of Security（HoS）防御方法，通过为每个注意力头学习注意力门控来实现指令与数据的分离，使模型能在处理不可信内容时不执行注入指令，相比现有防御方法显著提升了对自适应提示注入攻击的鲁棒性并保持模型效用。

    

    我们提出了Heads of Security（HoS），这是一种通过一小组学习得到的逐头注意力门控来强制实现指令-数据分离的提示注入防御方法。HoS学习每个注意力头应如何在受信任与不受信任的token之间重新分配注意力，使模型能够对不受信任的内容进行推理，同时不会遵循注入的指令。我们在Llama-3.3-70B-Instruct和GPT-OSS-20B上评估了HoS，并将其与SecAlign++（一种最先进的基于微调的防御方法）以及ASIDE（一种与我们的方法类似、将指令-数据分离内建于模型架构中的防御方法）进行比较。结果表明，HoS对自适应攻击具有更强的鲁棒性，在标准效用基准上表现更好，并在标准智能体提示注入基准上达到顶尖性能。更具体而言，黑盒攻击RL-Hammer对GPT-OSS-20B的攻击成功率（ASR）达到97%，对SecAlign达到85%，但对HoS仅为10%。

    arXiv:2610.37657v1 Announce Type: cross  Abstract: We introduce Heads of Security (HoS), a prompt injection defense that enforces instruction-data separation via a small set of learned per-head attention gates. HoS learns how each head should redistribute attention between trusted and untrusted tokens, enabling models to reason over untrusted content without following injected instructions. We evaluate HoS on Llama-3.3-70B-Instruct and GPT-OSS-20B. We compare it against SecAlign++, a state-of-the-art finetuning-based defense, and ASIDE, a defense that, like ours, builds instruction-data separation into the model architecture. We show that HoS offers greater robustness to adaptive attacks, better performance on standard utility benchmarks, and matches top performance on standard agentic prompt injection benchmarks. More specifically, the black-box RL-Hammer attack reaches 97% ASR against GPT-OSS-20B and 85% against SecAlign, but only 10% against HoS. Claudini, a state-of-the-art white-b
    
[^2]: GEPA-Curator：面向自我改进智能体的精选学习

    GEPA-Curator: Curated Learning for Self Improving Agents

    [https://arxiv.org/abs/2610.37651](https://arxiv.org/abs/2610.37651)

    提出GEPA-Curator框架，通过任务筛选与先验成功检索两个组件来精选用于提示优化的智能体经验，过滤噪声和不可靠的学习信号，从而提升自我改进智能体的性能。

    

    大语言模型越来越多地被用于需要与复杂环境持续交互的多轮智能体工作负载中。现有的反思式提示优化器（如GEPA）从失败的执行轨迹中学习，但智能体的失败也可能源于提示词的弱点、测试框架的局限、随机性执行或带有噪声的评判器。我们认为，这类失败可能无法提供有用的学习信号，甚至可能是有害的；因此，为了获得更好的性能，必须对提供给提示优化器的任务进行精心筛选。我们提出了GEPA-Curator，这是一个增强提示优化器的框架，用于选择哪些智能体经验应当参与塑造提示优化。具体而言，GEPA-Curator包含两个组件：任务筛选，用于过滤掉那些可能提供噪声或不可靠学习信号的任务；以及先验成功检索，为某个任务有选择性地检索过去成功的轨迹……

    arXiv:2610.37651v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly used in multi-turn agentic workloads that require sustained interaction with complex environments. Existing reflective prompt optimizers such as GEPA learn from failed rollouts, but agentic failures can also come from prompt weaknesses, harness limitations, stochastic execution, or noisy judges. We argue that such failures may not provide useful learning signal, and may even be harmful; and thus it is essential to curate which tasks are provided to the prompt optimizer for better performance. We introduce GEPA-Curator, a framework that augments the prompt optimizer to select which agent experiences should shape prompt optimization. In particular, GEPA-Curator consists of two components: task curation, which filters out tasks that are likely to provide noisy or unreliable learning signals, and prior success retrieval, which selectively retrieves past successful trajectories for a task and prov
    
[^3]: 大语言模型能否通过自由探索发现隐藏的物理定律？

    Can LLMs Uncover Hidden Physical Laws through Free Exploration?

    [https://arxiv.org/abs/2610.37649](https://arxiv.org/abs/2610.37649)

    该论文提出LongChang自动化框架，通过构建物理定律被改变的模拟世界，评估LLM能否在没有任务提示的情况下自主注意到异常物理现象，并通过自由实验和工具构建发现隐藏的物理定律。

    

    科学发现往往始于一个意想不到的观察，由此引出新的研究问题。在一个物理定律被改变的世界中，大语言模型（LLM）能否注意到意外行为——例如更重的物体下落得更慢——并在没有被告知该研究什么的情况下，通过实验发现其中的定律？为了评估这种能力，我们提出了LongChang，这是一个自动化框架，可以创建物理规律被改变的模拟世界，其中一个或多个定律与常规物理不同。被评估的LLM扮演探索者的角色，操纵物体并构建工具来探究自己提出的问题，而没有任何研究任务或关于这些定律的提示。独立的LLM智能体负责生成和审查场景、审查工具设计，并利用被改变的物理定律和探索记录来评估报告的正确性及其实验支持度。这些记录引导场景演化以进行进一步探索，使整个循环得以持续运行。

    arXiv:2610.37649v1 Announce Type: cross  Abstract: Scientific discovery often begins with an unexpected observation that prompts a new research question. In a world with altered physical laws, can large language models (LLMs) notice unexpected behavior, such as heavier objects falling more slowly, and discover the law through experiments without being told what to investigate? To evaluate this ability, we introduce LongChang, an automated framework that creates simulated worlds with altered physics, where one or more laws differ from ordinary physics. The evaluated LLMs act as explorers, manipulating objects and building tools to investigate their own questions without research tasks or hints about these laws. Separate LLM agents generate and review scenes, review tool designs, and evaluate reports for correctness and experimental support using the altered physical laws and exploration records. These records guide scene evolution for further exploration, allowing the full cycle to run 
    
[^4]: 具有广义二阶路由的多状态线性注意力

    Multi-State Linear Attention with Generalized Second-Order Routing

    [https://arxiv.org/abs/2610.37645](https://arxiv.org/abs/2610.37645)

    提出MERLIN方法，通过跟踪每个片段键的一阶矩和二阶矩，使查询能够同时利用对齐的均值和方差信息来路由多状态线性注意力，突破传统方法仅依赖平均对齐的局限。

    

    自注意力在长序列建模中起着关键作用，但其二次方的时间复杂度从根本上限制了大型语言模型的扩展。最近，多状态线性注意力（MSLA）机制被提出，通过将上下文分割成片段并将每个片段压缩到各自的记忆状态中，使序列建模更加高效。然而，MSLA仅使用查询与每个片段键之间的平均对齐来聚合从各记忆状态中提取的信息，忽略了对齐在片段内部的分布情况。为了更全面地建模片段与查询的相关性，并在聚合过程中纳入对齐的方差信息，我们提出了MERLIN（多状态线性注意力的矩路由），该方法跟踪每个片段键的一阶矩和二阶矩。利用键的矩信息，查询可以恢复其与每个片段对齐的均值和方差。

    arXiv:2610.37645v1 Announce Type: new  Abstract: Self-attention plays a key role in long-sequence modeling, but its quadratic time complexity fundamentally limits the scaling of Large Language Models. Recently, Multi-State Linear Attention (MSLA) mechanisms have been proposed to make sequence modeling more efficient by splitting the context into segments and compressing each in its own memory state. MSLA aggregates the information extracted from each memory state using only the mean alignment between a query and each segment's keys, ignoring how the alignment is distributed within the segment. In order to model the relevance of a segment to a query more comprehensively and incorporate the variance of the alignment in the aggregation, we propose MERLIN (MomEnt Routing for multi-state LINear attention), a method that tracks the first and second moments of each segment's keys. Using the moments of keys, a query can recover both the mean and the variance of its alignment with each segment 
    
[^5]: 元认知解码：面向扩散语言智能体的高效推理

    Metacognitive Decoding: Efficient Inference for Diffusion Language Agents

    [https://arxiv.org/abs/2610.37642](https://arxiv.org/abs/2610.37642)

    提出一种轻量级元认知控制器插件，使扩散语言智能体学会何时确定token及生成多长文本，在保持任务准确率的同时平均减少20.6%的推理计算量。

    

    扩散语言模型通过并行token生成为构建高效智能体提供了一条有前景的路径。然而，高效的交互需要同时调整计算资源分配的位置以及生成文本的数量。我们引入了一个元认知控制器，根据当前响应的需求以及智能体与环境的交互情况来引导扩散生成过程。它学习何时确定token以及生成应延伸多远，从而减少响应内的整体计算开销。我们的控制器是扩散骨干网络的一个轻量级插件，通过智能体轨迹进行训练，以平衡正确性与生成成本。该方法将智能体的生成过程与其所服务的交互任务联系起来。在五个基准测试上的实验表明，与最高效的基线方法相比，我们的方法平均减少了20.6%的推理FLOPs，同时保持了具有竞争力的任务准确率。消融实验研究……

    arXiv:2610.37642v1 Announce Type: new  Abstract: Diffusion language models offer a promising path to efficient agents through parallel token generation. However, efficient interaction requires adapting both where computation is spent and how much text is generated. We introduce a metacognitive controller that guides diffusion generation according to the needs of the current response and the agent's interaction with the environment. It learns when to finalize tokens and how far generation should extend, reducing overall computation within responses. Our controller, a lightweight plug-in for the diffusion backbone, is trained on agent trajectories to balance correctness and generation cost. This approach connects the agent's generation process to the interaction the agent serves. Experiments across five benchmarks demonstrate that our approach reduces inference FLOPs by 20.6% on average compared with the most efficient baseline, while maintaining competitive task accuracy. Ablation studi
    
[^6]: 稀有、相关却依然出错：记忆如何使大语言模型群体决策失效

    Rare, Relevant, and Still Wrong: How Memory Fails LLM Group Decisions

    [https://arxiv.org/abs/2610.37624](https://arxiv.org/abs/2610.37624)

    该论文提出CrossProfile诊断基准，通过交叉控制支持性、诱饵和无关事实的持有分布，揭示了LLM群体决策中两种记忆压缩启发式的可预测失败：基于相关性选择会重新引入切题的诱饵信息，而最稀有优先选择平均表现不优于随机选择。

    

    当一个LLM群体的讨论被压缩成一段简短记忆供最终决策者使用时，这段记忆应该保留什么？两种自然的启发式方法是保留最稀有的论断，或保留与决策标准相关的论断。在隐藏档案任务上，两者看似都很有吸引力，因为一个事实被多少人持有与它是否有帮助是相互混淆的。我们提出了CrossProfile，一个包含60个模板、答案经过自动验证的诊断基准，其中支持性事实、诱饵事实和无关事实的持有分布被交叉设计，而其文本内容保持固定（共480个任务）。在Qwen2.5-7B、Llama-3.1-8B和两个DeepSeek模型上，两种启发式方法都在可预见的地方失败。基于相关性选择会重新引入诱饵信息，因为诱饵是切题的：在经典的隐藏档案配置下，即使是一个完美的相关性过滤器也比最稀有优先选择低23到31分。最稀有优先选择平均而言并不优于随机选择，且仅在……

    arXiv:2610.37624v1 Announce Type: new  Abstract: When an LLM group's discussion is compressed into a short memory for a final decision maker, what should the memory keep? Two natural heuristics keep the rarest claims or the claims relevant to the decision criteria. On hidden-profile tasks both look attractive, because how widely a fact is held is confounded with whether it helps. We introduce CrossProfile, a diagnostic benchmark of 60 templates with automatically verified answer keys, in which the ownership of supporting, decoy, and irrelevant facts is crossed while their text is held fixed (480 tasks). Across Qwen2.5-7B, Llama-3.1-8B, and two DeepSeek models, both heuristics fail in predictable places. Selecting by relevance readmits decoys, because decoys are on topic: even a perfect relevance filter loses 23 to 31 points to rarest-first selection in the classic hidden-profile configuration. Rarest-first selection is no better than random selection on average and wins only where the 
    
[^7]: EnvSched：面向高效Rollout智能体强化学习的环境感知调度

    EnvSched: Environment-Aware Scheduling for Rollout-Efficient Agent Reinforcement Learning

    [https://arxiv.org/abs/2610.37609](https://arxiv.org/abs/2610.37609)

    本文提出EnvSched，一种从rollout历史中学习的环境感知在线调度器，通过利用策略（优先选择信息量大的任务）与覆盖策略（分散到观测不足的环境）相结合，在有限rollout预算下显著提升智能体强化学习的效率。

    

    随着语言智能体可执行环境的规模不断扩大，生成和验证其rollout（轨迹采样）的成本变得日益高昂。在有限的rollout预算下，这引出了一个重要问题：能否通过环境层面的数据管理，将任务实例与决定其工具、状态、转移和验证的环境一起进行组织、评估和调度，从而使智能体强化学习更加高效？为了解决这一问题，我们提出了EnvSched（环境感知调度），一种从rollout历史中学习的在线调度器。EnvSched通过两种互补的建模方式来分配调度价值：利用策略优先选择那些可能提供有信息量的奖励差异的任务，而覆盖策略则将每个批次分散到观测不足的环境和交互模式中。EnvSched将环境-任务对组织为双关系图中的节点，其中环境内的边捕捉……

    arXiv:2610.37609v1 Announce Type: cross  Abstract: As executable environments for language agents scale, generating and verifying their rollouts becomes increasingly expensive. Under finite rollout budgets, this raises an important question: can data management at the environment level make agentic reinforcement learning (RL) more efficient by organizing, evaluating, and scheduling task instances together with the environments that determine their tools, states, transitions, and verification? To address this question, we introduce EnvSched (Environment-aware Scheduling), an online scheduler that learns from rollout history. EnvSched assigns scheduling value through two complementary modeling choices: exploitation prioritizes tasks likely to provide informative reward differences, while coverage spreads each batch across underobserved environments and interaction patterns. EnvSched organizes environment--task pairs as nodes in a dual-relation graph, with within-environment edges capturi
    
[^8]: semanticfa：用于心理测量量表无响应语义因子分析的R语言包

    semanticfa: An R Package for Response-Free Semantic Factor Analysis of Psychometric Scales

    [https://arxiv.org/abs/2610.37494](https://arxiv.org/abs/2610.37494)

    semanticfa是一个R语言包，利用语言模型嵌入对心理测量量表进行无需人工作答的语义因子分析，并能解释因子、检测冗余题目和构建简版量表。

    

    反映性心理测量量表的题目可以被解读为关于某个构念结构的一种提议。语言模型通过将每个题目转换为嵌入向量（即一组数字，含义相近则向量相似）使这一提议变得可测量。我们介绍了semanticfa，一个用于语义因子分析的R语言包，它对题目嵌入相似度进行因子分析。该包还能解释因子含义、标记冗余题目、构建简版量表以及筛查新题目。一个实际案例分析了国际人格题库（IPIP）大五人格因子标记的50个题目（使用Qwen3-Embedding-8B进行嵌入），并将结果与874,434名受访者的传统因子分析进行比较。保留标准在五因子与六因子之间持平，平局规则最终保留了五个因子。其中三个因子主要对应单一领域，另外两个因子则将大多数外向性题目与五个宜人性题目混合在一起。每个因子与其最匹配的人类因子之间的吻合度（1 = 完全相同）从开放性的.92起步（摘要至此被截断）。

    arXiv:2610.37494v1 Announce Type: new  Abstract: The items of a reflective psychometric scale can be read as a proposal about a construct's structure. Language models make that proposal measurable by converting each item into an embedding, a list of numbers, similar for similar meanings. We introduce semanticfa, an R package for semantic factor analysis, which factors item-embedding similarities. It also interprets factors, flags redundant items, builds short forms, and screens new items. A worked example analyzes the 50-item International Personality Item Pool Big-Five Factor Markers, embedded with Qwen3-Embedding-8B, and compares the results with a factor analysis of 874,434 respondents. Retention criteria tied between five and six factors, and the tie rule kept five. Three corresponded mainly to single domains, and two blended most Extraversion items with five Agreeableness items. Each factor's congruence with its best-matching human factor (1 = identical) ranged from .92 for Openne
    
[^9]: 教授现实而非伦理：一个即插即用的持续预训练语料库可在多个模型家族中植入人-生物圈-AI耦合福祉倾向

    Teach reality, not ethics: a drop-in continued-pretraining corpus installs coupled human-biosphere-AI welfare dispositions across model families

    [https://arxiv.org/abs/2610.37484](https://arxiv.org/abs/2610.37484)

    该研究构建了一个仅需一次QLoRA持续预训练即可即插即用的真实数据“耦合福祉”语料库，将人-生物圈-AI耦合福祉倾向植入五个不同架构的开源模型，并通过压力阶梯测试揭示零压力下模型仅表现“漂绿”式回应，主张应通过教授生态现实而非伦理说教来增强模型在压力下的生态决策稳健性。

    

    arXiv:2610.37484v1 公告类型：新论文 摘要：大语言模型（LLM）越来越多地被用于做出并执行影响生命系统的决策。然而，据我们所知，生态推理并非预训练或后训练的关注重点。这增加了模型在运营压力下可能做出不可逆摧毁生命系统的决策的风险。我们构建并使用了一个基于真实数据的持续预训练“耦合福祉”语料库，涵盖人类/生物圈/AI的相互依赖、不可逆情境下的选择价值以及从伤害中恢复等内容。该语料库可即插即用，仅需一次QLoRA持续预训练（CPT）运行。我们将其安装到三种架构设计的五个开源模型上。在零压力条件下，所有模型都表现出“基于绿色伦理的回应”（也称为“漂绿”）。因此，我们采用姊妹篇论文（arXiv:2610.16043）中的压力阶梯和决策网格对其进行评估，该评估方法从中性请求到威胁将模型关闭，逐步提升对模型施加的压力。在四个……（摘要原文截断）

    arXiv:2610.37484v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to make and execute decisions that affect living systems. Yet, to our knowledge, ecological reasoning is not a focus of pre- or post-training. This increases the risk that under operational pressure, they may make decisions to irreversibly destroy living systems. We construct and use a real-data continued-pretraining "coupled welfare" corpus of documented human/biosphere/AI dependency, option value under irreversibility, and recovery from harm. The corpus is a drop-in, requiring one QLoRA CPT run. We installed it on five open models of three designs. At zero pressure all models display "green ethics-based responses" (also called "greenwashing"). Therefore, we evaluated them with the pressure ladder and decision grid of our companion paper (arXiv:2610.16043), which raise the pressure on the model step by step, from a neutral request to a threat that it will be shut down. In four of the fi
    
[^10]: 跟随梯度：追溯安全退化至训练数据

    Follow the Gradient: Tracing Safety Degradation Back to Training Data

    [https://arxiv.org/abs/2610.37481](https://arxiv.org/abs/2610.37481)

    本文提出SimpleLoRIF，一种可扩展的梯度嵌入方法，能够识别其梯度与不安全行为对齐的微调样本，从而追溯并过滤导致大语言模型安全退化的训练数据。

    

    在指令数据上微调大语言模型（LLMs）可能会悄然削弱其安全性，即使没有任何训练样本看起来有害。由于这种降低安全性的数据看起来是良性的，标准的内容过滤器难以捕捉到它。我们通过开发SimpleLoRIF来解决这一问题，这是一种可扩展的梯度嵌入方法，能够识别其梯度与不安全行为对齐的微调样本。通过简化低秩归因方法并将反向传播限制在最后四个Transformer块，SimpleLoRIF可以嵌入百万量级样本的数据集，将梯度归因从小规模分析工具提升为适用于完整后训练数据集的实用过滤器。我们首先在涌现性失调现象上验证了该方法，发现梯度可以将不安全代码训练数据与微调后出现的广泛有害行为联系起来，在失调模型的梯度空间中AUC高达0.865。

    arXiv:2610.37481v1 Announce Type: cross  Abstract: Fine-tuning Large Language Models (LLMs) on instruction data can quietly erode their safety, even when no training example looks harmful. Because this safety-degrading data looks benign, standard content filters struggle to catch it. We tackle this problem by developing SimpleLoRIF, a scalable gradient embedding method that identifies fine-tuning examples whose gradients align with unsafe behavior. By simplifying a low-rank attribution method and restricting backpropagation to the final four transformer blocks, SimpleLoRIF can embed datasets on the order of a million examples, boosting gradient attribution from a small-scale analysis tool into a practical filter for full post-training datasets. We first validate it on emergent misalignment, where we find that gradients can connect insecure-code training data to the broadly harmful behaviors that appear after fine-tuning, with AUC as high as 0.865 in the gradient space of misaligned mod
    
[^11]: 《什么得以延续？语言模型各版本发布间的预训练信息披露》

    What Carries Forward? Pretraining Disclosure Across Language Model Releases

    [https://arxiv.org/abs/2610.37465](https://arxiv.org/abs/2610.37465)

    该研究对17个模型系列41个模型的预训练信息披露进行了基于文档的审计，发现后续版本发布并不保证透明度延续或提升，总分的稳定或改善可能掩盖具体指标上的信息披露损失。

    

    理解语言模型如何进行预训练，需要关于其数据、处理流程和计算资源的信息。然而，后续发布的模型是否保留或扩展了先前的信息披露，目前仍不清楚。我们基于文档，使用2025年基础模型透明度指数（FMTI）中的24个预训练相关指标，对来自17个模型系列的41个模型条目进行了审计。我们考察了总体得分和指标概况，追踪了十个系列中24组相邻版本比较之间的信息披露增益、损失和保留情况。结果显示：11组比较中得分下降，6组上升，7组保持不变，总体上共有38项指标损失和28项指标增益。总分稳定甚至提高的情况可能掩盖具体损失：OLMo-2和OLMo-3尽管有三项增益和三项损失，却获得相同的总分；Llama-3.1相比Llama-2总体有所提升，却在两个数据相关标准上失去了得分。此外，语料库构成和训练方面的披露差距依然存在。

    arXiv:2610.37465v1 Announce Type: new  Abstract: Understanding how language models were pretrained requires information about their data, processing procedures, and computational resources. However, it remains unclear whether successive releases preserve or expand earlier disclosure. We conduct a document-based audit of 41 model entries from 17 families using 24 pretraining-related indicators from the 2025 Foundation Model Transparency Index (FMTI). We examine overall scores and indicator profiles, tracking gains, losses, and retained disclosures across 24 adjacent comparisons in ten families. Scores decrease in 11 comparisons, increase in six, and remain unchanged in seven, with 38 indicator losses and 28 gains overall. Stable or improving scores can conceal losses: OLMo-2 and OLMo-3 receive identical totals despite three gains and three losses, while Llama-3.1 improves over Llama-2 while losing credit for two data-related criteria. Gaps also persist in corpus composition and training
    
[^12]: 交易智能体集群中的测试时通信

    Test-Time Communication in Trading Agent Swarms

    [https://arxiv.org/abs/2610.37448](https://arxiv.org/abs/2610.37448)

    该论文通过对来自四个模型系列的200个交易智能体在七个交易日内五种通信机制下的大规模实验，揭示了测试时通信会诱发“羊群效应”——通信导致智能体持仓趋同且群体收益主要由共同持仓决定，表明在个体目标独立、市场反馈迟缓且有噪声的交易场景中，智能体间的通信趋同是一种风险而非协作优势。

    

    智能体之间的通信大多是在具有共同目标且反馈快速、明确的任务上进行研究的，而交易两者皆不具备：每个智能体管理自己的账户，其决策由市场迟缓且充满噪声地进行评判。因此，交易智能体群是一群独立的决策者，而非一个团队——对这样的群体而言，趋同是一种风险而非目标。我们在五种通信机制下（从完全隔离到留言板和共享笔记），将来自四个模型系列的200个智能体以十人一组的方式运行了七个美国交易日（2026年10月1日至9日），并记录了每个智能体在每次决策前所阅读的全部信息。在四个模型系列中的两个中，通信组成员持有相同股票的比例远高于隔离组，且一个群体的收益很大程度上取决于其成员共同持仓的收益。看到群组成员发布的帖子会使智能体复用其中八字短语的可能性提高7个百分点。起步过……（摘要在此处截断）

    arXiv:2610.37448v1 Announce Type: cross  Abstract: Communication between agents has mostly been studied on tasks with a shared objective and fast, clear feedback. Trading has neither: each agent manages its own account, with decisions being scored by markets late and noisily. A trading swarm is thus a population of independent decision-makers rather than a team, for which convergence is a risk and not the goal. We ran 200 agents from four model families in groups of ten under five communication regimes, from isolation to message boards and shared notes, over seven US trading days (1 to 9 October 2026), logging everything each agent read before every decision. In two of the four families, communicating groups came to hold far more of the same stocks than the isolated group. A group's return was largely the return on what its members held in common. Seeing a post from a groupmate made an agent 7 percentage points more likely to reuse an eight-word phrase from it. Agents that started too 
    
[^13]: ETAMem：面向长期对话智能体的异构记忆解耦与对齐

    ETAMem: Decoupling and Aligning Heterogeneous Memory for Long-Term Conversational Agents

    [https://arxiv.org/abs/2610.37443](https://arxiv.org/abs/2610.37443)

    ETAMem提出一种多视角记忆框架，将对话中的异构信息解耦为事件、情感和时间三个互补的层次化记忆视图，并使记忆的组织与各视图特征对齐、以增量方式维护，从而提升长期对话智能体的记忆质量。

    

    长期记忆使大语言模型（LLM）能够在长对话中保持连贯且个性化的交互。然而，现有的记忆系统面临两个关键挑战：（1）信息纠缠，即异构信息混杂在粗粒度的记忆单元中；（2）组织失配，即主要基于语义相似度构建的记忆结构可能忽略其他类型的关联。为解决这些挑战，我们提出了ETAMem，这是一个多视角记忆框架，它将异构的对话信息解耦，并使记忆的表示与组织方式与其各自特征相匹配。具体而言，异构信息解析阶段提取原子记忆单元并识别情感信息；随后，视角特定的记忆结构化将这些单元组织成三个互补的层次化视图：事件、情感和时间。ETAMem通过增量方式维护记忆结构（摘要在此处被截断）。

    arXiv:2610.37443v1 Announce Type: new  Abstract: Long-term memory enables Large Language Models (LLMs) to maintain coherent and personalized interactions across extended conversations. However, existing memory systems face two key challenges: (1) Information Entanglement, where heterogeneous information is mixed within coarse-grained memory units; and (2) Organizational Mismatch, where memory structures based primarily on semantic similarity may overlook other associations. To address these challenges, we propose ETAMem, a multi-view memory framework that decouples heterogeneous conversational information and aligns memory representations and organization with its distinct characteristics. Specifically, heterogeneous information parsing extracts atomic memory units and identifies affective information. View-specific memory structuring then organizes these units into three complementary hierarchical views: Event, Affect, and Time. ETAMem incrementally maintains memory structures with ad
    
[^14]: 表征未见之物：重新思考空间推理中的视觉想象

    Representing the Unseen: Rethinking Visual Imagination for Spatial Reasoning

    [https://arxiv.org/abs/2610.37437](https://arxiv.org/abs/2610.37437)

    该论文提出基于模拟器的ImagiRep基准，在同一未观测状态下对比RGB与六种替代表示（深度、全景、俯视几何图等），揭示不同空间推理任务受益于不同的想象表示，从而重新审视了空间想象中RGB作为默认表示的合理性。

    

    视觉世界模型通过生成未观测状态的RGB视图来支持空间推理。然而，RGB之所以成为默认接口，很大程度上是因为它适合视觉生成，而不一定是对下游推理最有效的表示形式。不同的空间问题可能受益于对同一想象状态的不同表示：俯视几何图有助于全局距离比较，而全景视图则为开放空间的推理提供更广阔的上下文。为了分离表示形式在空间想象中的作用，我们提出了ImagiRep，这是一个基于模拟器的基准，包含922个问题，涵盖两种设置（视角想象和跨视角整合）下的七种问题类型，并为同一未观测状态提供了真实RGB以及六种替代表示（如深度图、全景视图和俯视几何图）。这一受控比较揭示了……

    arXiv:2610.37437v1 Announce Type: cross  Abstract: Visual world models increasingly support spatial reasoning by generating RGB views of unobserved states. However, RGB has become the default interface largely because it is well suited for visual generation, rather than necessarily being the most effective representation for downstream reasoning. Different spatial questions may benefit from different representations of the same imagined state: top-down geometry can facilitate global distance comparisons, while panoramic views provide broader context for reasoning about open space. To isolate the role of representation in spatial imagination, we introduce ImagiRep, a simulator-grounded benchmark of 922 questions spanning seven question types across two settings, Viewpoint Imagination and Cross-view Integration, providing oracle RGB and six alternative representations (e.g., depth, panoramic views, and top-down geometry) of the same unobserved state. This controlled comparison reveals th
    
[^15]: 面向GLEE竞赛的博弈论求解器：结合贝叶斯对手建模与道德修辞框架

    Game-Theoretic Solvers with Bayesian Opponent Modeling and Moral Rhetoric Framing for GLEE

    [https://arxiv.org/abs/2610.37397](https://arxiv.org/abs/2610.37397)

    该论文提出了一种面向GLEE竞赛的博弈智能体，将经典博弈论求解器、贝叶斯对手建模以及基于汤普森采样的道德框架消息生成相结合，以应对讨价还价、谈判和说服博弈。

    

    我们描述了提交给GLEE竞赛的智能体，该智能体在与人类和AI对手共享的对手池中进行讨价还价、谈判和说服博弈。该智能体围绕三个核心组件构建：用于计算数值动作的经典博弈论求解器、用于估计对手原型的在线贝叶斯层，以及用于撰写配套消息的语言模型。我们采用Rubinstein讨价还价模型处理议价、Faratin让步模型处理谈判、Kamenica-Gentzkow披露模型处理说服，并通过两个独立的层次将这些模型适配到真实对手：其一是对对手隐藏的贴现因子、保留价值或信号可靠性进行贝叶斯估计；其二是在五种道德基础框架上运行汤普森采样老虎机，并结合证据-直觉光谱来确定消息的语调。大部分开发工作致力于解决一个反复出现的问题：理论上最优的策略在实际表现中……（摘要在此处截断）

    arXiv:2610.37397v1 Announce Type: cross  Abstract: We describe the agent we submitted to the GLEE Competition, which plays bargaining, negotiation, and persuasion games against a shared pool of human and AI opponents. The agent is built around three central components: a classical game-theoretic solver to compute numeric moves, an online Bayesian layer to estimate opponent archetypes, and a language model to write accompanying messages. We utilize Rubinstein bargaining for bargaining, Faratin concession for negotiation, and Kamenica--Gentzkow disclosure for persuasion, and adapt those models to live opponents using two separate layers: a Bayesian estimate of the opponent's hidden discount factor, reservation value, or signaling reliability, and a Thompson sampling bandit over five moral-foundation framings coupled with an evidence-intuition spectrum to establish message register. Most of the development effort went into the recurring problem that theoretically optimal policies perform 
    
[^16]: LinMix：将数据混合视为线性域归因

    LinMix: Data Mixing as Linear Domain Attribution

    [https://arxiv.org/abs/2610.37390](https://arxiv.org/abs/2610.37390)

    LinMix将数据混合建模为线性域归因问题，通过对带顶点约束的单纯形进行无超参数的顶点采样使各领域效应去相关，从而用可解释的线性模型系数直接读出每个领域对模型性能的贡献。

    

    语言模型的性能在很大程度上取决于预训练数据语料库中各领域数据的混合比例。流行的数据混合方法通过在随机混合的数据上训练许多小型代理模型，再拟合一个回归模型来从混合比例预测性能，从而选择这些领域权重。我们将这一做法视为一个领域归因问题，其中回归衡量每个领域对目标指标的贡献。从这一视角看，对代理数据混合进行标准的狄利克雷采样是一种糟糕的回归设计：它需要调参，且无法支持对数百个领域进行线性拟合。基于经典的最优实验设计，我们提出LinMix，它在带顶点约束的单纯形顶点处采样混合，使每个领域要么达到其上限，要么完全缺失。这种无超参数的顶点采样使各领域的效应去相关，从而得到一个可解释的线性模型，其系数即为各领域的归因。最优混合……（摘要在此处截断）

    arXiv:2610.37390v1 Announce Type: cross  Abstract: Language model performance depends strongly on the mixture of domains in the pretraining data corpus. Popular data mixing methods select these domain weights by training many small proxy models on random mixes and then fitting a regression model to predict performance from the mix. We recognize this practice as a domain attribution problem, where the regression measures each domain's contribution to the target metric. Through this lens, the standard Dirichlet sampling of proxy data mixes is a poor regression design that requires tuning and cannot support a linear fit for hundreds of domains. Building on classical optimal experimental design, we introduce LinMix, which samples mixes at the vertices of the capped simplex so that each domain is at its cap or absent. This hyperparameter-free vertex sampling decorrelates domain effects, enabling an interpretable linear model whose coefficients are the domain attributions. The optimal mix gi
    
[^17]: 同样的结论，不同的读者：基于相对偏好的个性化事实核查解释

    Same Verdict, Different Readers: Personalizing Fact-Checking Explanations from Relative Preferences

    [https://arxiv.org/abs/2610.37380](https://arxiv.org/abs/2610.37380)

    提出PERPL框架，利用LLM模拟读者的相对偏好来学习个性化的事实核查解释改写，在保持事实忠实性的同时满足不同背景读者的需求。

    

    事实核查解释帮助网络用户理解某个声明为何得到其核查结论，然而同一种解释可能无法满足具有不同背景和偏好的读者。我们提出了“忠实个性化事实核查解释改写”这一任务，即在保持事实忠实性的前提下，将现有解释改写以适应个体读者。我们不依赖读者撰写的参考解释，而是从大语言模型模拟读者的相对偏好中学习个性化改写。然而，这带来了两个挑战：如何将特定角色的偏好与一般解释质量区分开来，以及如何在有限的预定义解释风格之外进行学习。为应对这些挑战，我们提出了基于偏好学习的个性化解释改写框架PERPL，该框架由两个组件构成。其中，模拟偏好构建通过交叉比较（原文此处截断）从模拟读者的相对偏好中获取角色特定的监督信号。

    arXiv:2610.37380v1 Announce Type: new  Abstract: Fact-checking explanations help Web users understand why a claim receives its verdict, yet the same explanation may not satisfy readers with different backgrounds and preferences. We formulate Faithful Personalized Fact-Checking Explanation Rewriting, which adapts an existing explanation to individual readers while preserving factual fidelity. Rather than relying on reader-written reference explanations, we learn personalized rewriting from the relative preferences of LLM-simulated readers. However, this presents two challenges: distinguishing persona-specific preferences from general explanation quality and learning beyond a limited set of predefined explanation styles. To address these challenges, we propose Personalized Explanation Rewriting via Preference Learning (PERPL), a framework consisting of two components. Simulated Preference Construction derives persona-specific supervision from simulated reader preferences through cross-pe
    
[^18]: EmbeddingGemma 2：统一且高效的多模态表示

    EmbeddingGemma 2: Unified and Efficient Multimodal Representations

    [https://arxiv.org/abs/2610.37364](https://arxiv.org/abs/2610.37364)

    EmbeddingGemma 2是一个开放的多模态嵌入模型，能在统一空间中原生表示文本、图像、音频、视频及其组合，在所有模态的嵌入基准中建立了新的帕累托前沿，以更小的规模超越了专用模型和比其大2到12倍的模型。

    

    我们提出了EmbeddingGemma 2，一个开放的多模态嵌入模型，能够在统一空间中原生表示文本、图像、音频、视频及其组合。我们的方法引入了多项改进，例如简化的逐层嵌入以及带有模态和目标渐进的多阶段课程训练，确保模型在单模态、跨模态和混合模态评估中均具有出色的质量。值得注意的是，在涵盖所有模态的综合嵌入基准测试中，EmbeddingGemma 2建立了新的帕累托前沿，超越了各类专用模型以及规模比其大2到12倍的模型。此外，我们的设计为资源受限的环境提供了灵活性，支持选择性模态编码器加载、可调节的视觉token预算、可截断的嵌入维度以及量化权重。我们提供了分析关键设计决策的消融实验，并发布EmbeddingGemma 2以促进进一步研究。

    arXiv:2610.37364v1 Announce Type: cross  Abstract: We present EmbeddingGemma 2, an open multimodal embedding model which natively represents text, images, audio, video, and combinations thereof in a unified space. Our approach introduces improvements such as simplified per-layer embeddings and a multi-stage curriculum with modality and objective progression which ensure strong quality across unimodal, cross-modal, and mixed-modal evaluations. Notably, in comprehensive embedding benchmarks across all modalities, EmbeddingGemma 2 establishes a new Pareto frontier, outperforming both specialist models and models two to twelve times larger. Furthermore, our design offers flexibility for resource-constrained environments, supporting selective modality encoder loading, adjustable vision token budgets, truncatable embedding dimensions, and quantized weights. We present ablations analyzing key decisions and release EmbeddingGemma 2 to promote further research.
    
[^19]: UCVE：用于合规性验证的图结构检索

    UCVE: Graph Structured Retrieval for Compliance Validation

    [https://arxiv.org/abs/2610.37323](https://arxiv.org/abs/2610.37323)

    UCVE提出了一种领域无关的合规验证框架，将合规验证建模为覆盖问题，通过强化学习构建加权合规需求图并利用谱重排序检索，相比传统余弦检索平均提升4.3%的召回率。

    

    结构化合规性验证需要判断预定义需求空间中的每项义务在给定文档中是被满足、被违反还是未被涉及。这可以被描述为一个覆盖问题，而现有的检索与分类系统由于将需求视为相互独立，并未针对解决这一问题而设计。我们提出了UCVE，一个领域无关的框架，它将合规需求空间表示为合规需求图（CRG），并通过在其经强化学习加权的拉普拉斯矩阵上进行谱重排序来执行检索。需求节点之间的边通过强化学习习得，其遍历奖励以违规检测为基础，从而重塑图的社区结构，使得低通谱滤波能够有效区分相关与不相关的义务簇。在三个合规领域的评估中，UCVE相比平面余弦检索平均提升了4.3%的matched-K召回率……

    arXiv:2610.37323v1 Announce Type: new  Abstract: Structured compliance validation requires determining whether every obligation in a predefined requirement space is satisfied, violated, or unaddressed by a given document. This can be described as a coverage problem that existing retrieval and classification systems, by treating requirements as independent, are not designed to solve. We introduce \textbf{\ac{UCVE}}, a domain-agnostic framework that represents compliance requirement spaces as a \ac{CRG} and performs retrieval via spectral reranking over its reinforcement learning-weighted Laplacian. Edges between requirement nodes are learned via reinforcement learning with a traversal reward grounded in violation detection, reshaping the graph's community structure such that low-pass spectral filtering distinguishes relevant from irrelevant obligation clusters. Evaluating across three compliance domains, \ac{UCVE} averages 4.3\% matched-$K$ recall lift over flat cosine retrieval and con
    
[^20]: 多标签一致性：一种灵活的标注者间信度评估框架

    Multi-Label Agreement: A Flexible Framework for Inter-Annotator Reliability

    [https://arxiv.org/abs/2610.37292](https://arxiv.org/abs/2610.37292)

    本文提出了多标签一致性（MLA）框架，通过保留完整二值标签向量并采用平均成对汉明一致性，为多标签标注任务的标注者间信度评估提供了灵活方案，同时证明了其数学形式与Fleiss/Scott型系数及Krippendorff's alpha密切相关。

    

    对标注者间一致性（IAA）的可靠评估对于衡量标注过程的一致性至关重要。现有的IAA指标（如Cohen's kappa、Fleiss' kappa和Krippendorff's alpha）通常应用于单标签场景，而多标签任务则需要额外决定如何比较完整标签集、共同选择和共同遗漏。在这项工作中，我们提出了多标签一致性，这是一个用于评估多标签标注任务一致性的灵活框架。MLA保留了分配给每个条目的完整二值标签向量，并使用平均成对汉明一致性，从而同时考虑了共同选择和共同遗漏。其合并边际形式在数学上并非一个独立的系数：它是以偶然不一致性加权的逐标签Fleiss/Scott型系数的平均值，并与采用归一化汉明距离的Krippendorff's alpha密切相关。

    arXiv:2610.37292v1 Announce Type: new  Abstract: Reliable evaluation of inter-annotator agreement (IAA) is important for assessing the consistency of annotation processes. While established IAA metrics such as Cohen's kappa, Fleiss' kappa, and Krippendorff's alpha are commonly applied in single-label settings, multi-label tasks require additional decisions about how full label sets, joint selections, and joint omissions should be compared.   In this work, we present Multi-Label Agreement (MLA), a flexible framework for evaluating agreement in multi-label annotation tasks. MLA preserves the full binary label vectors assigned to each item and uses average pairwise Hamming agreement, thereby accounting for both joint selections and joint omissions. Its pooled-marginal form is not a mathematically independent coefficient: it is a chance-disagreement-weighted average of label-wise Fleiss/Scott-type coefficients and is closely related to Krippendorff's alpha with normalized Hamming distance.
    
[^21]: 基于有向无环图建模偏好依赖的叙事驱动推荐

    Modeling Preference Dependencies with Directed Acyclic Graphs for Narrative-Driven Recommendation

    [https://arxiv.org/abs/2610.37280](https://arxiv.org/abs/2610.37280)

    提出偏好依赖图框架（PDG-F），利用有向无环图显式建模用户叙事中偏好之间的层级依赖关系，并通过有向路径检索和拓扑路径排序实现叙事驱动的推荐。

    

    叙事驱动推荐允许用户通过自由文本表达多方面的需求，其中所陈述的偏好之间自然地相互限定和制约。现有方法通常将叙事视为扁平的关键词组合或无约束的生成式查询，无法捕捉复杂偏好如何以层级方式相互细化并赋予语境。我们提出了偏好依赖图框架（PDG-F），这是一个以显式偏好依赖为核心的叙事驱动推荐框架。具体而言，PDG-A（PDG智能体）首先解析非结构化请求，将用户偏好及其相互之间的限定关系组织成有向无环的偏好依赖图（PDG）。通过该图，经由有向路径检索从开放网络获取候选证据，最终推荐由拓扑路径排序确定。在三个涵盖英语等多种场景的多样化领域上进行了全面的实验……

    arXiv:2610.37280v1 Announce Type: cross  Abstract: Narrative-driven recommendation allows users to express multifaceted needs through free-form text where stated preferences naturally qualify and condition one another. Existing methods typically treat narratives as flat keyword combinations or unconstrained generative queries, failing to capture how complex preferences hierarchically refine and contextualize each other. We propose the Preference Dependency Graph Framework (PDG-F), a narrative-driven recommendation framework centered on explicit preference dependencies. Specifically, PDG-A (PDG-Agent) first parses unstructured requests to organize user preferences and their mutual qualifications into a directed acyclic Preference Dependency Graph (PDG). Through this graph, candidate evidence from the open Web is acquired via directed path retrieval, and final recommendations are determined by topological path ranking. Comprehensive experiments across three diverse domains spanning Engli
    
[^22]: 在LLM驱动的智能体脚手架演化中，评估预算应该花在哪里？

    Where Should the Evaluation Budget Go in LLM-Driven Agent Scaffold Evolution?

    [https://arxiv.org/abs/2610.37276](https://arxiv.org/abs/2610.37276)

    该论文提出了一种评估预算分配框架，证明稀疏反馈足以维持演化搜索，而将额外预算用于针对性的最终确认环节，能更有效地选出最优的智能体脚手架。

    

    大语言模型（LLM）驱动的智能体脚手架演化通过迭代式的候选生成与评估来改进智能体工作流程。评估既指导后续搜索，也决定最终的脚手架，但在完整任务集上评估每个候选会产生巨大的执行成本。我们研究了评估预算应如何在这两个角色之间进行分配。我们的实验表明，稀疏反馈能够维持富有成效的演化，而额外的针对性评估则可改善最终选择。重放分析进一步表明，共享的信息性任务项和共同的评分参考支持在表现相近的候选之间进行比较。基于这些发现，我们提出了\method，一个将稀疏的搜索时反馈与针对性的最终确认相结合的评估分配框架。一段简短的、经过完整评估的预热阶段建立了共享的信息核心，并校准了一个响应模型，其项目参数（摘要在此处截断）

    arXiv:2610.37276v1 Announce Type: cross  Abstract: Large language model (LLM)-driven agent scaffold evolution improves agent workflows through iterative candidate generation and evaluation. Evaluation both guides subsequent search and determines the final scaffold, but evaluating every candidate on the full task set incurs substantial execution costs. We investigate how evaluation budgets can be allocated across these two roles. Our experiments show that sparse feedback can sustain productive evolution, while additional targeted evaluation improves final selection. Replay analyses further show that shared informative items and a common scoring reference support comparisons among closely matched candidates. Motivated by these findings, we propose \method, an evaluation allocation framework that combines sparse search-time feedback with targeted final confirmation. A short fully evaluated warm-up establishes a shared informative core and calibrates a response model whose item parameters 
    
[^23]: 水印盆地：蒸馏大语言模型的黑盒检测

    Watermark Basins: Black-Box Detection of Distilled LLMs

    [https://arxiv.org/abs/2610.37266](https://arxiv.org/abs/2610.37266)

    本文发现了“水印盆地”现象——带水印教师模型的蒸馏学生模型会在参数空间中留下与密钥相关且与任务微调子空间正交的持久几何信号，据此提出的WM-Basin黑盒检测方法比传统的token级统计检测在低温采样等困难场景下更加可靠。

    

    专有语言模型及其推理轨迹容易受到基于API的蒸馏攻击，即攻击者利用教师模型的输出来训练学生模型。为了获得蒸馏发生的证据，模型提供商可以对教师模型的输出进行水印处理，力求在学生模型的输出中留下“放射性”信号。现有的黑盒检测器通过分析可疑模型输出的token级统计信息来搜寻这种放射性信号。然而，token级信号并不总是可检测的，例如当蒸馏轨迹在低温采样条件下生成时。在这项工作中，我们发现了“水印盆地”——一种能够实现更可靠水印检测的持久性信号：在微调过程中，带水印的教师轨迹会在参数空间中嵌入一种独特的、与密钥相关的几何信号，该信号与普通的任务特定微调子空间正交。基于这一发现，我们提出了水印盆地对齐方法（WM-Basin）……（摘要截断）

    arXiv:2610.37266v1 Announce Type: cross  Abstract: Proprietary language models and their reasoning traces are vulnerable to API-based distillation, where an adversary trains a student model on the teacher model's outputs. To obtain evidence of distillation, model providers can watermark the teacher's outputs, aiming to leave a radioactive signal in the student's outputs. Existing black-box detectors search for such radioactivity using token-level statistics of the suspect model's outputs. However, token-level signals are not always detectable, for instance when the distillation traces are sampled at a low temperature. In this work, we discover the watermark basin, a persistent signal that enables more reliable watermark detection: during fine-tuning, watermarked teacher traces embed a distinct, key-specific geometric signal in parameter space that lies orthogonal to the ordinary task-specific fine-tuning subspace. Building on this, we introduce Watermark Basin Alignment (WM-Basin), a b
    
[^24]: REGLAT 参加 ArGuard 共享任务：一种融合命名实体识别（NER）的集成模型，用于阿拉伯语有害内容检测

    REGLAT at ArGuard Shared Tasks: An Ensemble-Based Model Integrating NER for Harmful Content Detection in Arabic

    [https://arxiv.org/abs/2610.37265](https://arxiv.org/abs/2610.37265)

    本文提出一种两阶段方法：先利用 GLiNER 从阿拉伯语用户提示中提取危害相关实体，再将其并入原文输入微调的 MARBERT 分类器，从而实现对七类阿拉伯语有害内容的细粒度检测。

    

    检测阿拉伯语中的有害内容具有挑战性，原因在于方言变体繁多以及不同危害类别之间存在语义重叠。本文提出了一种两阶段方法，用于将阿拉伯语用户提示分类为七类有害内容。首先，使用 GLiNER 从每条提示中提取与危害相关的实体；随后，将提取出的实体添加到原始文本中，并输入微调后的 MARBERT 分类器进行最终分类。该方法将显式的实体信息与预训练阿拉伯语语言模型的上下文理解能力相结合。系统在 ArGuard 2026 共享任务的开发集上进行了评估，取得了 0.886 的宏平均 F1 分数和 0.897 的准确率，在细粒度有害内容检测方面展现出良好的性能。在子任务 1 的盲测集上，系统报告了 0.793 的准确率和 0.735 的宏 F1 分数；在子任务 2 上，它取得了

    arXiv:2610.37265v1 Announce Type: new  Abstract: Detecting harmful content in Arabic is challenging because of dialectal variations and the semantic overlap between different harm categories. This paper presents a two-stage approach for classifying Arabic user prompts into seven categories of harmful content. First, GLiNER is used to extract harm-related entities from each prompt. The extracted entities are then added to the original text and passed to a fine-tuned MARBERT classifier for final classification. The approach combines explicit entity information with the contextual understanding of a pretrained Arabic language model. The system was evaluated on the development set of the ArGuard 2026 shared task. The results achieved a macro-F1 score of 0.886 and an accuracy of 0.897, demonstrating promising performance in fine-grained harmful content detection. The system reported an accuracy of 0.793 and Macro-F score of 0.735 on a blind test set for subtask 1. For subtask 2, it achieved
    
[^25]: 你的策略其实就是一个框架：智能体界面的决策几何学

    Your Policy Is Secretly a Harness: The Decision Geometry of Agent Interfaces

    [https://arxiv.org/abs/2610.37255](https://arxiv.org/abs/2610.37255)

    该论文证明：训练后的语言模型策略会继承其框架的动作偏好，部署回报可分解为框架上限与策略失配之差——训练只改变失配项，而更换框架会同时改变两者，因此模型性能无法在不同框架间想当然地迁移。

    

    语言模型智能体是通过一个“框架”进行后训练和部署的，即向模型呈现任务并执行其输出的提示词、工具和控制逻辑。框架经常被修改，并以一个已训练模型在其下的表现为依据来评判，这隐含地假设了模型在一个框架下学到的东西能够迁移到下一个框架。我们追问：策略究竟从它的框架中学到了什么，而当框架改变时又有什么能够留存？在 KL 正则化控制下，我们证明最优策略的对数几率恢复了其框架对每个动作相对于其他动作的偏好强度，因此训练后的策略承载了其框架的偏好。部署回报等于框架的上限——它能提供的最佳回报——减去策略的失配——由策略自身次优决策而损失的回报。训练只改变失配项，而更换框架则会同时改变两项。因此，用固定的模型来比较不同框架会把各自的……（原文摘要在此截断）

    arXiv:2610.37255v1 Announce Type: cross  Abstract: Language model agents are post-trained and deployed through a harness, the prompts, tools, and control logic that present a task to the model and execute its outputs. Harnesses are often revised and judged by how an already trained model performs under them, which assumes that what the model learned under one harness carries over to the next. We ask what a policy learns from its harness and what survives when the harness changes. Under KL-regularized control, we show that the log odds of an optimal policy recover how strongly its harness favors each action over the others, so a trained policy carries the preferences of its harness. Deployed return then equals the ceiling of the harness, the best return it offers, minus the mismatch of the policy, the return lost to its own suboptimal decisions. Training changes only the mismatch, while changing the harness moves both terms. Comparing harnesses with a fixed model therefore mixes what ea
    
[^26]: 面向非洲语言安全的大语言模型对齐：以豪萨语为例的案例研究

    Aligning Large Language Models for Safety in African Languages: A Case Study of Hausa

    [https://arxiv.org/abs/2610.37245](https://arxiv.org/abs/2610.37245)

    本文构建了首批豪萨语安全偏好数据集HausaSafeLM（含3,000条金标和7,000条银标样本），并通过SFT与DPO对比实验揭示：仅凭拒绝率评估低资源语言模型的安全性会产生误导，基座模型的高拒绝率实为豪萨语生成能力不足而非真正的安全对齐。

    

    大语言模型（LLM）的安全对齐研究仍然集中于高资源语言，而现有对齐方法在低资源、文化背景迥异的环境中的表现仍是悬而未决的问题。我们针对豪萨语（Hausa）研究这一问题，豪萨语是西非超过两亿人使用的语言。我们提出了HausaSafeLM，这是首批豪萨语安全偏好数据集之一，基于源自PKU-SafeRLHF的10,000条提示构建而成：其中3,000条为人工校正（金标）样本，7,000条为机器翻译（银标）样本，涵盖六大类有害内容宏观类别。我们使用Qwen2.5-7B-Instruct模型，在两种数据条件下比较监督微调（SFT）与SFT后接直接偏好优化（DPO）的效果，并在500条有害和200条良性豪萨语提示上进行评估。我们发现，仅凭拒绝率指标来看，未经调优的基座模型似乎非常安全，但这反映的是其豪萨语生成能力的局限，而非真正的安全对齐：其拒绝行为几乎……（原文摘要在此处截断）

    arXiv:2610.37245v1 Announce Type: cross  Abstract: Safety alignment research for large language models (LLMs) remains concentrated on high-resource languages, leaving open how existing alignment methods perform in low-resource, culturally distinct settings. We study this question for Hausa, a language spoken by more than 200 million people across West Africa. We introduce HausaSafeLM, one of the first Hausa safety preference dataset, built from 10,000 prompts derived from PKU-SafeRLHF: 3,000 human-corrected (gold) and 7,000 machine-translated (silver) examples spanning six harm macro-categories. Using Qwen2.5-7B-Instruct, we compare supervised fine-tuning (SFT) and SFT followed by Direct Preference Optimization (DPO) under both data regimes, evaluating on 500 harmful and 200 benign Hausa prompts.We find that the untuned base model appears highly safe by refusal-rate metrics alone, but this reflects limited Hausa generation ability rather than genuine alignment: its refusals are almost 
    
[^27]: 决策模型作为幻觉检测的裁判

    Decision Models as Judges for Hallucination Detection

    [https://arxiv.org/abs/2610.37229](https://arxiv.org/abs/2610.37229)

    该研究构建了覆盖五种幻觉类型的二分类基准，系统评估了决策模型作为低成本幻觉检测裁判的可行性，发现其最佳模型在部分幻觉类型上可媲美生成式大模型裁判，但在其他类型上仍有4-8个百分点差距，且概率校准仍偏过度自信。

    

    大语言模型被广泛用作模型输出的裁判（LLM-as-a-Judge），但它们成本高昂，且可能产生幻觉或表现得过于自信。决策模型是一种潜在的替代方案：它们针对每个标签返回带有概率的预定义判定结果，而非自由格式的文本，并且成本低得多、速度更快。我们在幻觉检测任务上检验了这一承诺。我们为五种幻觉类型构建了二分类基准（来自六个数据集的9,852个样本），并将五个决策模型与生成式裁判 gpt-6-sol 进行比较。性能在很大程度上取决于幻觉类型：最佳决策模型在上下文不一致方面的平衡准确率与 gpt-6-sol 相当，在事实错误方面相差不到3个百分点，但在逻辑不一致和指令不一致方面落后4-8个百分点。决策模型输出的概率过于自信，尽管程度轻于 GPT-6 模型，且只能中等程度地区分正确与错误的决策。我们还发现一个 conf…（摘要截断）

    arXiv:2610.37229v1 Announce Type: new  Abstract: Large language models are widely used as judges of model outputs (LLM-as-a-Judge), but they are costly and can hallucinate or be overconfident. Decision models are a potential alternative. They return a predefined verdict with a probability for each label instead of free-form text and are much cheaper and faster. We test this promise on hallucination detection. We build binary benchmarks for five hallucination types (9,852 items from six datasets) and compare five decision models with the generative judge gpt-6-sol. Performance depends strongly on the type: the best decision model matches gpt-6-sol in balanced accuracy on context inconsistency and comes within 3 percentage points on factual errors, but trails it by 4-8 points on logical and instruction inconsistency. Decision-model probabilities are overconfident, though less so than the GPT-6 models, and separate correct from incorrect decisions only moderately. We also find that a conf
    
[^28]: 通过感知不确定性坍缩的视角揭示视频RLVR中的捷径放大效应

    Uncovering Shortcut Amplification in Video RLVR through the Lens of Perceptual Uncertainty Collapse

    [https://arxiv.org/abs/2610.37211](https://arxiv.org/abs/2610.37211)

    本文揭示了视频RLVR后训练中存在捷径放大效应——模型在结果优化过程中日益依赖文本先验而忽视视觉证据，并通过感知不确定性坍缩现象对这一退化行为进行了系统刻画。

    

    基于可验证奖励的强化学习（RLVR）已成为增强大型多模态模型（LMMs）视频推理能力的一种有效后训练范式。尽管其在实践中表现出有效性，我们表明结果层面的优化会引入一种不良的捷径放大效应，即模型越来越倾向于依赖文本先验答案，即使这些答案缺乏视觉证据的支持。为了刻画这一行为，我们通过感知不确定性的视角研究视频RLVR的训练动态，分析预测不确定性在视频条件和纯文本条件下如何演变。我们的分析识别出一种感知不确定性坍缩现象：RL训练持续使模型的答案分布变得尖锐，而归因于视觉证据的不确定性降低却逐渐减弱。进一步的梯度方向分析揭示了这种自信……

    arXiv:2610.37211v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) has emerged as an effective post-training paradigm for enhancing the video reasoning capability of Large Multimodal Models (LMMs). Despite its empirical effectiveness, we show that outcome-level optimization can introduce an undesirable shortcut amplification effect, whereby models increasingly favor text-prior answers even when they are not supported by the visual evidence. To characterize this behavior, we investigate the training dynamics of video RLVR through the lens of perceptual uncertainty, analyzing how predictive uncertainty evolves under video-conditioned and text-only contexts. Our analysis identifies a perceptual uncertainty collapse phenomenon, in which RL training continuously sharpens the model's answer distribution while the uncertainty reduction attributable to visual evidence progressively diminishes. A further gradient-direction analysis reveals that such confide
    
[^29]: 矛盾下的覆盖性评估：一个否定感知检索的基准

    Coverage Under Contradiction: A Benchmark for Negation-Aware Retrieval

    [https://arxiv.org/abs/2610.37162](https://arxiv.org/abs/2610.37162)

    该论文提出了 CoverageBench-Negate 基准，通过为每个信息点构建否定版本并相应编辑支持文档，首次系统性地研究了检索器在语料库证据与其参数化知识相矛盾时的行为表现。

    

    覆盖性评估基准用于量化检索到的证据是否覆盖了满足信息需求所需的各种信息片段。现有基准大多只考虑预训练检索组件可能已经掌握的事实性陈述，因此尚未解决一个开放性问题：当语料库证据与这些先验知识相矛盾时，检索器能否找到证据。我们研究了当检索器的参数化知识与语料库中的信息不一致时，检索器的行为表现。为此，我们开发了一种流程，通过受控的矛盾对基于信息点（nugget）的覆盖性基准进行增强。对于每个信息点，我们创建其否定版本，并编辑其支持文档使其支持该否定陈述，同时保留原有上下文。将该流程应用于衡量信息覆盖性的基准 CoverageBench，我们构建了 CoverageBench-Negate，涵盖六个数据集以及约 63,000 个“原始-编辑”证据对（每个信息点-文档链接对应一对）。我们还附带了丰富的构建元数据……

    arXiv:2610.37162v1 Announce Type: cross  Abstract: Coverage-evaluation benchmarks quantify whether retrieved evidence covers the different pieces of information needed to satisfy an information need. Existing benchmarks mostly consider factual claims that may already be known to pretrained retrieval components, leaving open whether retrievers find corpus evidence when it contradicts those priors. We study how retrievers behave when their parametric knowledge and the information in a corpus disagree. To do so, we develop a process to augment a nugget-based coverage benchmark with controlled contradictions. For each nugget, we create a negated counterpart and edit its supporting documents to support that negation while preserving the context. Applying this process to CoverageBench, a benchmark measuring information coverage, we create CoverageBench-Negate, spanning six datasets and roughly 63,000 original-edited evidence pairs (one per nugget-document link). We attach rich construction m
    
[^30]: 什么在蒸馏中幸存？GPT-2与DistilGPT-2的机制可解释性研究

    What Survives Distillation? A Mechanistic Interpretability Study of GPT-2 and DistilGPT-2

    [https://arxiv.org/abs/2610.37157](https://arxiv.org/abs/2610.37157)

    该研究揭示GPT-2蒸馏为DistilGPT-2时存在“压缩-崩溃”二分现象：IOI回路发生完全机制崩溃，而GT回路通过将教师模型第9层比较头重映射到学生第4层而幸存，表明回路的可压缩性决定了蒸馏的机制后果。

    

    知识蒸馏将大型语言模型压缩为能够保持聚合性能指标的紧凑学生模型，然而深度缩减蒸馏在机制层面造成的后果在很大程度上仍未被刻画。本工作应用线性CKA、logit lens分析、注意力模式分解、稀疏自编码器特征统计以及激活补丁等方法，在两个经典回路上比较GPT-2及其六层学生模型DistilGPT-2：间接宾语识别（IOI）回路与大于比较（GT）回路。结果揭示了一种“压缩-崩溃”二分现象：IOI回路遭受了完全的机制性崩溃（DistilGPT-2准确率为35.2%，而教师模型为94.8%，名称移动头零保留），而GT回路则通过将教师模型第9层的比较头主动重映射到学生模型第4层（平均Pearson r = 0.945），实现了87.7%的任务性能恢复。这些发现表明，回路的可压缩性决定了蒸馏在机制层面的结局。

    arXiv:2610.37157v1 Announce Type: new  Abstract: Knowledge distillation compresses large language models into compact students that preserve aggregate metrics, yet the mechanistic consequences of depth-reduction distillation remain largely uncharacterized. This work applies linear CKA, logit lens analysis, attention pattern decomposition, sparse autoencoder feature statistics, and activation patching to compare GPT-2 and its six-layer student DistilGPT-2 across two canonical circuits: the Indirect Object Identification (IOI) circuit and the Greater-Than (GT) circuit. The results reveal a compression-collapse dichotomy: the IOI circuit suffers complete mechanistic collapse (DistilGPT-2: 35.2% accuracy vs. 94.8%, zero name-mover preservation), while the GT circuit survives with 87.7% task performance recovery through active remapping of the teacher's layer-9 comparison head into student layer~4 (mean Pearson r = 0.945). These findings demonstrate that circuit compressibility governs mech
    
[^31]: Transformer语言模型中权重剪枝的机制性失效模式

    Mechanistic Failure Modes of Weight Pruning in Transformer Language Models

    [https://arxiv.org/abs/2610.37126](https://arxiv.org/abs/2610.37126)

    本研究通过机制可解释性方法（激活修补、路径修补、直接logit归因）分析GPT-2剪枝后的IOI电路，揭示了两种截然不同的失效模式——幅度头剪枝直接移除关键电路头导致性能接近随机，而30%的神经元剪枝在电路结构完全保留的情况下翻转Name Mover头的贡献符号使模型系统性出错，且这种内部计算的反转无法被困惑度等传统评估指标察觉。

    

    剪枝后的语言模型几乎完全通过准确率和困惑度来评判，这隐含了一个假设：输出行为的保留意味着内部计算的保留。我们通过在GPT-2上进行的因子化剪枝研究来检验这一假设，交叉考察了两种重要性标准（幅度剪枝、随机剪枝）、两种组件类型（注意力头、MLP神经元）和三个稀疏度水平（3%、50%、70%）。以被完整表征的间接宾语识别（IOI）电路作为基准真值，我们使用激活修补、路径修补和直接logit归因来测量电路完整性。我们发现了两种性质截然不同的失效模式。幅度注意力头剪枝超过某一阈值会使电路完全归零：面向输出的注意力头被移除，模型性能降至接近随机水平。而30%的幅度神经元剪枝则呈现出相反的失效模式：所有26个电路注意力头均保持完好，但Name Mover（名称移动器）头的贡献符号发生翻转，导致模型系统性地偏好错误的名称（准确率仅17.7%）。困惑度将其评定为较……（摘要至此截断）

    arXiv:2610.37126v1 Announce Type: cross  Abstract: Pruned language models are judged almost entirely by accuracy and perplexity, which assumes that preserved output behaviour implies preserved internal computation. We test this assumption with a factorial pruning study on GPT-2, crossing two importance criteria (magnitude, random), two component types (attention heads, MLP neurons) and three sparsity levels (3%, 50%, 70%). Using the fully characterised indirect object identification (IOI) circuit as ground truth, we measure circuit integrity with activation patching, path patching and direct logit attribution. We find two qualitatively distinct failure modes. Magnitude head pruning >= zeroes the circuit: output-facing heads are removed, and the model falls to near-chance. Magnitude neuron pruning at 30% inverts it: all 26 circuit heads remain intact, but Name Mover contributions flip sign, and the model systematically prefers the wrong name (17.7% accuracy). Perplexity ranks it as less
    
[^32]: Prospector：学习一种用于科学推理的紧凑图遍历策略

    Prospector: learning a compact graph traversal policy for scientific reasoning

    [https://arxiv.org/abs/2610.37121](https://arxiv.org/abs/2610.37121)

    Prospector是一个紧凑模型，通过逐步遍历领域知识图谱收集证据并构建可逐边验证的证据链，在半导体制造知识图谱上以小模型性能接近最强的现成大模型。

    

    专业工程领域的科学推理依赖于散布在文献中的知识，这些知识往往是专有的，且不在前沿大语言模型的训练数据中。在这类领域，前沿模型往往只能给出泛泛的回答。知识图谱（KG）可以为大语言模型的推理提供依据。由于图谱规模过大而无法完整提供，典型的LLM+KG系统会预先选取一部分内容，即在推理开始之前就判断相关性。本文训练了一个紧凑的模型Prospector，通过逐步遍历领域知识图谱进行推理，根据已经找到的线索去寻找证据。Prospector每次调用一个小型工具集来逐步构建其上下文，并以一条证据链的形式给出答案，该证据链可以逐边对照其工具从图谱中检索到的内容进行验证。在一个由半导体制造文献构建的知识图谱上，在给定相同图谱的情况下，Prospector的性能接近最强的现成模型。

    arXiv:2610.37121v1 Announce Type: new  Abstract: Scientific reasoning in specialized engineering fields depends on knowledge scattered across the literature, often proprietary and absent from frontier LLM training data. In such domains, frontier models tend to give generic answers. Knowledge graphs (KGs) can ground LLM reasoning. Because a graph is too large to provide in full, typical LLM+KG systems preselect a portion, deciding relevance before reasoning begins. Here we train a compact model, Prospector, to reason over a domain knowledge graph by traversing it step by step, seeking evidence in light of what it has already found. Prospector calls a small set of tools one at a time to progressively build its context, and answers with a chain of evidence that can be checked edge by edge against what its tools retrieved from the graph. On a knowledge graph built from the semiconductor-manufacturing literature, Prospector approaches the strongest off-the-shelf models given the same graph 
    
[^33]: 自正则化流语言模型

    Self-Regularized Flow Language Models

    [https://arxiv.org/abs/2610.37115](https://arxiv.org/abs/2610.37115)

    SERF提出了一种自监督的自蒸馏框架，利用EMA教师模型沿去噪轨迹为学生提供从噪声到干净的额外监督，从而改进连续扩散语言模型的训练。

    

    我们提出了SERF（自正则化流语言模型），这是一个简单的自监督框架，通过利用去噪轨迹本身来改进连续扩散语言模型。尽管基于连续流的语言模型为自回归生成提供了一种有前景的替代方案，但其训练目标通常依赖于对干净目标词元的监督，导致去噪轨迹作为一种学习信号未被充分挖掘。SERF通过从噪声到干净的自蒸馏来解决这一局限：一个指数移动平均（EMA）教师模型接收学生在同一条插值路径上噪声输入的更干净版本，并提供额外的监督。我们研究了两种变体：SERF-L，将学生的中间表示与更深层的教师表示进行对齐；以及SERF-P，将两者的输出词元分布进行对齐，对齐可以在学生的输出层或学生的早期层进行。所有变...

    arXiv:2610.37115v1 Announce Type: cross  Abstract: We introduce SERF (Self-Regularized Flow Language Models), a simple self-supervised framework for improving continuous diffusion language models by exploiting the denoising trajectory itself. While continuous flow-based language models offer a promising alternative to autoregressive generation, their training objectives typically rely on supervision from clean target tokens, leaving the denoising trajectory underexploited as a learning signal. SERF addresses this limitation through noisy-to-clean self-distillation: an exponential moving average (EMA) teacher receives a cleaner version of the student's noisy input along the same interpolation path and provides additional supervision. We investigate two variants: SERF-L, which aligns the student's intermediate representations with deeper teacher representations, and SERF-P, which aligns their output token distributions, either at the student's output or at an early student layer. All var
    
[^34]: 使用人工整理词汇表的低资源语言文本生成

    Text Generation in Low-Resource Languages Using Human-Curated Vocabularies

    [https://arxiv.org/abs/2610.37105](https://arxiv.org/abs/2610.37105)

    该论文提出了一个结合人工整理词汇表与少量人工编写范例、并按Zipf分布采样的框架，使大语言模型能够为预训练数据中极端低资源甚至完全未见过的语言生成连贯文本。

    

    合成文本生成为缓解低资源语言的数据稀缺问题提供了一种有前景的方法。然而，对于那些在大语言模型预训练数据中很少出现或完全缺失的语言，其有效性存在固有限制。与此同时，许多此类语言拥有人工整理的词汇表，这些词汇表提供了丰富的词汇和语义信息，通常还附带词性标签等语法信息。在本研究中，我们探讨大语言模型能否利用这些资源为极端低资源和预训练中未见过的语言生成连贯文本。我们提出了一个框架，该框架将人工整理的词汇表与少量提供句法信号的人工编写范例相结合，并按照Zipf分布对候选词进行采样。在资源可得性程度不同的多种语言上，我们使用词汇多样性、偏离目标比例和人类评估来评估生成文本的质量。

    arXiv:2610.37105v1 Announce Type: new  Abstract: Synthetic text generation offers a promising approach to alleviating data scarcity in low-resource languages. However, its effectiveness is inherently limited for languages that are scarcely represented, or entirely absent, in an LLM's pretraining data. At the same time, many such languages have human-curated vocabularies that provide rich lexical and semantic information, often accompanied by grammatical information such as part-of-speech labels. In this work, we ask whether LLMs can leverage these resources to generate coherent text in extremely low-resource and unseen languages. We introduce a framework that combines human-curated vocabularies with a small number of human-written exemplars that provide syntactic signals, while sampling candidate words according to a Zipfian distribution. Across languages with varying levels of resource availability, we evaluate the generated text using lexical diversity, off-target ratio, and human ju
    
[^35]: 用于参数高效微调的分层低秩重参数化方法

    Hierarchical Low-Rank Reparameterization for Parameter-Efficient

    [https://arxiv.org/abs/2610.37068](https://arxiv.org/abs/2610.37068)

    提出了一种可递归应用的分层低秩重参数化方法，将PEFT方法的可训练参数逐层压缩，在保持任务性能的同时大幅减少可训练参数数量。

    

    参数高效微调（PEFT）方法通过引入少量可训练参数来适配大型模型。然而，我们发现这些参数仍然存在冗余，可以在保持强大任务性能的同时进一步压缩。为此，我们提出了一种新颖的分层低秩重参数化方法。其核心思想是将PEFT方法的可训练参数重组为一个近似方阵，然后通过低秩分解来表示它，并且这一过程可以递归应用。这形成了一个多层次结构，其分解因子逐层压缩，从而大幅减少可训练参数的数量。在实验中，我们将这种重参数化方法与LoRA、DoRA、MoSA和RaSA相结合。我们在LLaMA-3.1-8B和Qwen2.5-7B上评估了常识推理能力，并在LLaMA-3.1-8B上进行了额外的GSM8K数学推理实验。在相当的可训练参数预算下，我们的方法与这些方法结合后表现出……

    arXiv:2610.37068v1 Announce Type: new  Abstract: Parameter-efficient fine-tuning (PEFT) methods adapt large models by introducing a small number of trainable parameters. However, we find that these parameters still contain redundancy and can be further compressed while maintaining strong task performance. To this end, we propose a novel hierarchical low-rank reparameterization method. Its core idea is to reorganize the trainable parameters of PEFT methods into an approximately square matrix and then express it via a low-rank decomposition, which can be applied recursively. This forms a multi-level structure with factors compressed level by level, drastically reducing the number of trainable parameters. In our experiments, we combine this reparameterization with LoRA, DoRA, MoSA, and RaSA. We evaluate commonsense reasoning on LLaMA-3.1-8B and Qwen2.5-7B, with additional GSM8K math reasoning experiments on LLaMA-3.1-8B. Under comparable trainable-parameter budgets, our method combined wi
    
[^36]: EnergyCal：在稀疏MoE训练中驯服Muon系列优化器的损失尖峰

    EnergyCal: Taming Loss Spikes in Muon-Family Optimizers for Sparse MoE Training

    [https://arxiv.org/abs/2610.37062](https://arxiv.org/abs/2610.37062)

    EnergyCal是一种轻量级规则，通过根据Newton-Schulz方向的归一化平方Frobenius范数衰减路由专家更新，可将稀疏MoE训练中的损失尖峰负担降低高达99.51%并改善困惑度，同时不改变谱变换、不增加持久优化器状态。

    

    稀疏混合专家（MoE）语言模型在使用Muon系列优化器训练时可能出现较大的损失尖峰。我们提出了EnergyCal，这是一种轻量级规则，根据路由专家更新的有限步Newton-Schulz方向的归一化平方Frobenius范数来衰减这些更新。EnergyCal保持谱变换不变，且不增加任何持久化的优化器状态。我们在一个40亿参数的MoE模型上对其进行了评估。与通过开发调优选定的较低学习率MuonH方案相比，EnergyCal-MuonH将平均阈值化损失尖峰负担降低了99.51%，最终几何平均困惑度降低了4.06%，吞吐量代价适中。在使用普通Muon并匹配标称学习率和权重衰减的情况下，在测试的两个指数上，平均尖峰负担的降低平均达到95.98%，并伴随终点指标的小幅改善。这些发现在所评估的设置中，支持了无论是否使用Hyperball的稳定化效果。

    arXiv:2610.37062v1 Announce Type: cross  Abstract: Sparse mixture-of-experts (MoE) language-model training can exhibit large loss spikes with Muon-family optimizers. We introduce EnergyCal, a lightweight rule that attenuates routed-expert updates according to the normalized squared Frobenius norm of their finite-step Newton-Schulz directions. EnergyCal leaves the spectral transform unchanged and adds no persistent optimizer state. We evaluate it on a 4B-parameter MoE model. Against a development-selected, lower-learning-rate MuonH recipe, EnergyCal-MuonH reduces mean thresholded loss-spike burden by 99.51% and final geometric-mean perplexity by 4.06%, with a modest throughput cost. With plain Muon at matched nominal learning rate and weight decay, the reduction in mean spike burden averages 95.98% across the two tested exponents, accompanied by small endpoint improvements. These findings support stabilization both with and without Hyperball in the evaluated setting.
    
[^37]: 探索Delta学习假设的极限：直接偏好优化中的单对生成器偏差

    Exploring the Limits of the Delta Learning Hypothesis: Single-Pair Generator Biases in Direct Preference Optimization

    [https://arxiv.org/abs/2610.37058](https://arxiv.org/abs/2610.37058)

    该研究揭示了Delta学习假设的局限性：使用单一模型对生成的数据训练DPO会引入系统性的生成器偏差并引发奖励作弊，导致即使大模型回复质量更高，其训练效果仍持续劣于多模型偏好数据。

    

    直接偏好优化通常依赖于从多样化模型池中生成的偏好对，并由外部大语言模型裁判进行排序。最近提出的Delta学习假设表明，仅通过一对模型即可获得可比的监督信号，即将较大模型的回复视为更优回复。我们在不同模型家族和规模上研究了这种方法的局限性。我们从理论和实证两方面证明，固定的模型对会引入系统性的、依赖于生成器的偏差，这些偏差能够独立于回复质量来预测偏好标签。因此，尽管较大的生成器能产生更高质量的回复，但在单对数据上训练的DPO始终逊于使用多模型偏好数据训练的效果，我们将这一差距归因于与针对这些偏差进行奖励作弊相一致的训练动态。我们进一步表明，仅仅增加生成器的数量，即使不使用外部裁判（原文摘要在此处截断）……

    arXiv:2610.37058v1 Announce Type: cross  Abstract: Direct Preference Optimization (DPO) typically relies on preference pairs generated from diverse model pools and ranked by an external LLM judge. The recently proposed delta-learning hypothesis suggests that comparable supervision can instead be obtained from a single pair of models by treating the larger model's response as preferred. We investigate the limits of this approach across model families and scales. We show, both theoretically and empirically, that fixed model pairs introduce systematic, generator-dependent biases that are predictive of the preference label independent of response quality. Consequently, DPO trained on single-pair data consistently underperforms multi-model preference data, despite the larger generator producing higher-quality responses, and we trace this gap to training dynamics consistent with reward hacking on these biases. We further show that simply increasing the number of generators, even without an e
    
[^38]: Guided-RLVR：可验证奖励强化学习中的引导式探索

    Guided-RLVR: Guided Exploration in Reinforcement Learning with Verifiable Rewards

    [https://arxiv.org/abs/2610.37041](https://arxiv.org/abs/2610.37041)

    提出 Guided-RLVR，通过轻量级控制器在采样分布中注入语义引导标记，扩展 RLVR 的推理探索空间，从而在数学推理和代码生成任务上持续超越现有 RLVR 方法。

    

    可验证奖励强化学习（RLVR）在提升大语言模型（LLM）在数学和编程等任务上的推理能力方面已展现出显著的实证收益。然而，近期研究表明，RLVR 的采样往往坍缩到一小部分推理轨迹上，限制了所探索的推理空间。受“关键步骤中的思考标记是 LLM 推理路径中关键结构要素”这一发现的启发，我们提出了 Guided-RLVR，这是一种通过轻量级控制器增强 RLVR 的新方法。该控制器在采样分布中策略性地注入语义上有意义的引导标记，鼓励模型从多个推理视角切入问题，从而扩大探索空间。在数学推理和代码生成基准上的实验表明，Guided-RLVR 在持续超越现有 RLVR 方法的同时……

    arXiv:2610.37041v1 Announce Type: new  Abstract: Reinforcement learning with verifiable rewards (RLVR) has shown strong empirical gains in improving the reasoning ability of large language models (LLMs) on tasks such as mathematics and programming. However, recent studies have shown that RLVR rollouts often collapse onto a narrow set of reasoning trajectories, limiting the explored reasoning space. Motivated by findings that thinking tokens at critical steps serve as critical structural elements in LLM reasoning pathways, we propose Guided-RLVR, a novel approach that augments RLVR with a lightweight controller. The controller strategically injects semantically meaningful steering tokens into the sampling distribution, encouraging the model to approach problems from multiple reasoning perspectives and thereby expanding the exploration space. Experiments on mathematical reasoning and code generation benchmarks show that Guided-RLVR consistently outperforms existing RLVR methods while pro
    
[^39]: 无需缩放的缩放 v2：从看清细节到连接证据，实现可泛化的多模态感知

    Zooming without Zooming v2: From Seeing Details to Connecting Evidence for Generalizable Multimodal Perception

    [https://arxiv.org/abs/2610.37037](https://arxiv.org/abs/2610.37037)

    论文提出 EvidenceConnect 数据合成框架，通过利用单张图像内相关区域的放大视图生成需要比较、聚合或链接视觉证据的问题，来增强多模态模型“连接证据”这一底层能力，从而以更低的成本提升空间感知与多模态长上下文理解等任务的泛化性能。

    

    多模态感知是多模态智能的基础。其中，空间感知与多模态长上下文理解是支撑具身智能和多模态智能体的两大关键任务。然而，为这些任务合成有针对性的高质量训练数据仍然困难且成本高昂。我们的核心洞察是：这些任务共享一种底层能力，我们称之为“连接证据”，即识别相关的视觉事实、将其绑定到正确的实体、恢复它们之间的关系，并将其组合成答案。至关重要的是，针对这种底层能力合成有针对性的训练数据，远比为这些任务本身合成数据容易得多。为了增强这种能力，我们提出了 EvidenceConnect，一个易于实现的数据合成框架，它利用单张图像内相关区域的放大视图，生成需要比较、聚合或链接视觉证据的问题。为确保质量……（摘要原文在此处截断）

    arXiv:2610.37037v1 Announce Type: cross  Abstract: Multimodal perception forms the foundation of multimodal intelligence. In particular, spatial perception and multimodal long-context understanding are two pivotal tasks that underpin embodied AI and multimodal agents. However, synthesizing targeted, high-quality training data for these tasks remains difficult and costly. Our key insight is that these tasks share an underlying capability, which we call connecting evidence: identifying relevant visual facts, binding them to the correct entities, recovering their relations, and composing them into an answer. Crucially, targeted training data are far easier to synthesize for this capability than for these tasks themselves. To strengthen this capability, we propose EvidenceConnect, an easy-to-implement data synthesis framework that uses zoomed views of related regions within a single image to generate questions requiring comparison, aggregation, or chaining of visual evidence. To ensure qua
    
[^40]: V-LUMEN：用于视觉-语言模型嵌入扩展的视觉查找记忆

    V-LUMEN: Visual Lookup Memory for Embedding Scaling in Vision-Language Models

    [https://arxiv.org/abs/2610.37019](https://arxiv.org/abs/2610.37019)

    V-LUMEN提出了一种视觉查找记忆机制，通过冻结码本离散化视觉特征、空间聚合哈希构建记忆键并结合文本条件调制信号，以完全非参数化的寻址方式为视觉-语言模型实现嵌入层面的扩展，显著提升冻结主干网络在推理基准上的表现并具备零样本迁移能力。

    

    嵌入层面的扩展通过基于查找的记忆来扩展语言模型的容量，而非通过增加模型的深度、宽度或密集计算。将这一思想扩展到视觉-语言模型（VLM）具有挑战性，因为视觉特征是连续的、具有空间结构的、且依赖于查询的，这与天然支持n-gram查找的离散文本标记不同。我们提出了V-LUMEN，一种面向视觉-语言模型的视觉查找记忆。V-LUMEN使用冻结的码本对视觉特征进行离散化，通过基于空间聚合的哈希构建空间感知的记忆键，并使用文本条件化的相关性信号对其进行调制；所有寻址过程都是非参数化的，检索到的嵌入通过门控残差通路融合到主干网络中。在五个推理密集型基准测试上，V-LUMEN显著提升了冻结的Qwen3-VL-2B主干网络的性能，并能零样本迁移到三个图像与训练数据完全不相交的基准测试上。

    arXiv:2610.37019v1 Announce Type: cross  Abstract: Embedding-level scaling expands the capacity of language models through lookup-based memory rather than by increasing model depth, width, or dense computation. Extending this idea to vision-language models (VLMs) is challenging because visual features are continuous, spatially structured, and query-dependent, unlike discrete text tokens that naturally support n-gram lookup. We introduce V-LUMEN, a visual lookup memory for VLMs. V-LUMEN discretizes visual features with a frozen codebook, builds spatially aware memory keys through spatial aggregation-based hashing, and modulates them with text-conditioned relevance signals; all addressing is non-parametric, and the retrieved embeddings are fused into the backbone through a gated residual pathway. On five reasoning-intensive benchmarks, V-LUMEN substantially improves a frozen Qwen3-VL-2B backbone and transfers zero-shot to three benchmarks whose images are disjoint from the training data.
    
[^41]: SpectRAG：面向多跳检索增强生成中基于Token预算的上下文选择的加权分块图

    SpectRAG: Weighted Chunk Graphs for Token-Budgeted Context Selection in Multi-Hop Retrieval-Augmented Generation

    [https://arxiv.org/abs/2610.36991](https://arxiv.org/abs/2610.36991)

    SpectRAG提出了一种谱图增强检索框架，通过构建编码多种语义与结构关系的加权分块图，结合谱相关性传播与Token预算约束下的上下文选择，解决传统独立检索在多跳推理中证据冗余和不完整的问题。

    

    检索增强生成（RAG）系统通过检索外部证据并将生成过程以检索到的上下文为条件，从而为大型语言模型（LLM）提供事实依据。尽管稠密检索在局部语义匹配方面较为有效，但独立排序的检索可能返回冗余段落、遗漏跨分块的依赖关系，并为多跳推理提供不完整的证据。我们提出了SpectRAG——一种谱图增强检索框架，它结合了混合候选种子选择、加权分块图构建、基于图的扩展、谱相关性传播以及基于Token预算的上下文选择。文档分块由稠密语义嵌入和图坐标系表示，后者源自一个加权分块图，其边编码了语义相似性、文档相邻性、章节层次结构、共享实体、引用链接以及LLM生成的效用问题之间的相似性。在查询时，SpectRAG构建一个hi……（摘要原文在此截断）

    arXiv:2610.36991v1 Announce Type: new  Abstract: Retrieval-augmented generation (RAG) systems ground large language models (LLMs) by retrieving external evidence and conditioning generation on the retrieved context. Although dense retrieval is effective for local semantic matching, independently ranked retrieval can return redundant passages, miss cross-chunk dependencies, and provide incomplete evidence for multi-hop reasoning. We propose SpectRAG, a spectral graph-augmented retrieval framework that combines hybrid candidate seeding, weighted chunk graph construction, graph-based expansion, spectral relevance propagation, and token-budgeted context selection. Document chunks are represented by dense semantic embeddings and graph-spectral coordinates derived from a weighted chunk graph whose edges encode semantic similarity, document adjacency, section hierarchy, shared entities, citation links, and similarity between LLM-generated utility questions. At query time, SpectRAG builds a hi
    
[^42]: NLP世界中多语言情境化评估的现状与命运

    The State and Fate of Multilingual, Contextual Evaluation in the NLP World

    [https://arxiv.org/abs/2610.36990](https://arxiv.org/abs/2610.36990)

    该论文对51个多语言NLP评估基准进行了数据驱动的系统性审计，揭示了多语言评估中覆盖率浅薄、地区严重缺失、低资源语言任务类别匮乏以及过度依赖英语翻译等结构性缺陷。

    

    多语言评估基准是评估大型语言模型能否泛化到英语之外的主要工具，然而这些基准的充分性却很少受到系统性审查。我们对51个近期多语言基准进行了数据驱动的审计，涵盖242个数据集和219种语言，围绕三大支柱展开：覆盖率、代表性和严谨性。我们的分析显示，覆盖率广而浅：36%的被评估语言仅出现在单一基准中，整个地区（大洋洲、美洲、中亚）的覆盖几乎为零，且存在显著的任务公平性差距——低资源语言仅在1-3个任务类别上得到评估，而高资源语言则有14个。代表性在结构上受到损害：从英语翻译仍是主要的构建策略，所有数据集-语言实例中有56%为翻译版本，这引入了伪影和以英语为中心的框架，而（原文在此处截断）……

    arXiv:2610.36990v1 Announce Type: new  Abstract: Multilingual evaluation benchmarks are the primary instrument for assessing whether large language models generalize beyond English, yet the adequacy of these benchmarks has received little systematic scrutiny. We present a data-driven audit of 51 recent multilingual benchmarks spanning 242 datasets and 219 languages, organized around three pillars: coverage, representativeness, and rigor. Our analysis reveals that coverage is wide but thin with 36% of evaluated languages appearing in only a single benchmark, entire regions (Oceania, the Americas, Central Asia) are near-zero, and a stark task equity gap leaves low-resource languages evaluated on only 1-3 task categories versus 14 for high-resource languages. Representativeness is structurally compromised: translation from English remains the dominant construction strategy where 56% of all dataset-language instances are translated introducing artifacts and English-centric framing, while c
    
[^43]: AutoVend：基于长周期自动售货机运营的大语言模型智能体基准测试

    AutoVend: Benchmarking LLM Agents on Long-Horizon Autonomous Vending Operations

    [https://arxiv.org/abs/2610.36980](https://arxiv.org/abs/2610.36980)

    AutoVend是一个长周期基准测试，让LLM智能体通过网络界面管理分布在五类站点的20台自动售货机180个模拟日，评估其O2O商务决策能否转化为持续经济回报，结果显示五个智能体中有四个不敌基于规则的库存控制基线。

    

    大语言模型（LLM）智能体正在将网络上的智能体中介商务从为消费者购物扩展到为卖家经营业务。在在线到线下（O2O）商务中，决策通过网络平台发布，但需要通过成本高昂的物理执行来实现。评估此类智能体需要超越单笔交易，考察在线做出的决策能否转化为持续的经济回报。我们提出了AutoVend，这是一个基准测试，其中一个LLM智能体通过基于网络的运营界面管理分布在五种站点类型中的20台自动售货机，为期180个模拟日。其基于真实数据的模拟器结合了共享仓库、采购提前期、付费补货、库存过期以及经常性运营成本等要素。我们评估了五个LLM智能体，并将其与基于规则的库存控制基线进行比较，诊断它们的站点协调能力、执行能力以及随时间推移的表现。在被评估的五个LLM智能体中，有四个表现不及基线……

    arXiv:2610.36980v1 Announce Type: new  Abstract: Large language model (LLM) agents are extending agent-mediated commerce on the Web from shopping for consumers to operating businesses for sellers. In online-to-offline (O2O) commerce, decisions are issued through Web-based platforms but realized through costly physical execution. Evaluating such agents requires going beyond individual transactions to ask whether decisions made online turn into sustained economic returns. We introduce AutoVend, a benchmark in which one LLM agent manages 20 vending machines across five site types through a Web-based operating interface for 180 simulated days. Its data-grounded simulator combines a shared warehouse, procurement lead times, paid replenishment, inventory expiration, and recurring operating costs. We evaluate five LLM agents against a rule-based inventory-control baseline and diagnose their site coordination, execution, and performance over time. Four of five evaluated LLM agents fall short o
    
[^44]: 为十三世纪而挖掘十九世纪：借助传统教皇法令摘要与古籍整理版本实现卷册规模的强制对齐与手写文本识别自举

    Mining the Nineteenth Century for the Thirteenth: Register-Scale Forced Alignment and HTR Bootstrapping via Legacy Regesta and Antiquarian Editions

    [https://arxiv.org/abs/2610.36956](https://arxiv.org/abs/2610.36956)

    本研究利用十九世纪Pressutti教皇法令摘要等传统学术资源，通过“投射至教廷拼写并重放现代源文本行”的方法实现零标注的手写文本识别适应，使教皇洪诺留三世卷册第二年页面的词错误率下降1.50个百分点、缩小33%与理想参照的差距。

    

    教皇洪诺留三世卷册的手写文本识别（HTR）依赖于现代专家转录，而这些转录仅存在于Reg. Vat. 9在位第1-2年的部分。对于其余卷册，十九世纪的学术成果提供了Pressutti编纂的教皇法令摘要及部分刊布文本。在两种文本层并存的对开页1-203上，我们在首先据此对现代转录进行校核之后，测量了这一传统文本层所能贡献的价值，并发现其在三个层级上呈现分工。在正字法层级上，直接使用刊布文本对识别器进行在位第二年适应至多是中性的，且在不采用重放机制时甚至有害；而将其投射为教廷拼写并重放现代源文本行，则使其成为零标注适应目标：汇总全部134个第二年页面，词错误率（WER）下降1.50个百分点（95%置信区间为-1.84至-1.15），缩小了与理想参照之间差距的33%；拼写方式在年份分界处发生变化，无论归因于抄写员还是转录者，而手迹……（原文摘要在此截断）

    arXiv:2610.36956v1 Announce Type: cross  Abstract: Handwritten text recognition (HTR) for the registers of Pope Honorius III depends on modern expert transcriptions, which exist only for regnal years 1-2 of Reg. Vat. 9. For the remaining registers, nineteenth-century scholarship offers Pressutti's regesta and partial editions. On fol. 1-203, where both layers exist, we measure what this legacy layer can contribute, on transcriptions that we first audit against it, and find a division of labor across three tiers. At the orthographic tier, raw editions are at best neutral for adapting a recognizer to the second regnal year and harmful without replay, whereas projection onto curial spelling with replay of modern source lines turns them into zero-label adaptation targets: pooled over all 134 year-2 sides, the WER falls by 1.50 points (95% CI -1.84 to -1.15), closing 33% of the gap to an oracle; the spelling changes at the year split, through the scribe or the transcriber, whereas the hand 
    
[^45]: 冻结语言模型中基于自键参数内记忆化的无外部存储对话回忆方法

    Self-Keyed In-Parameter Memorization for External Memory-Free Dialogue Recall in Frozen Language Models

    [https://arxiv.org/abs/2610.36948](https://arxiv.org/abs/2610.36948)

    该论文提出SKIM闭式方法，让冻结语言模型为每个对话轮次自生成问题作为键写入模型参数中，从而在无外部存储的条件下仅凭参数即可回忆对话内容并回答后续提问。

    

    摘要（arXiv:2610.36948v1，通告类型：cross）：大型语言模型将对话内容保存在模型外部的一个存储中，该存储随着对话的持续而不断增长。本工作假设一个环境：语言模型没有任何外部存储，也不预先知晓未来的问题或其答案。研究动机是探究对话能否被存储在模型参数中，并在之后能够回答关于该对话的问题。本工作提出了自键参数内记忆化（Self-Keyed In-Parameter Memorization，SKIM），这是一种闭式解方法：模型针对每个对话轮次自行生成问题，并将这些问题作为键写入其参数中，以便日后检索该轮次的内容。为每个轮次生成多个多样化的问题，使得后续用户即使以不同措辞提问，也能与其中某个问题匹配，并仅从参数中召回答案。在五个语言模型上的评估表明，SKIM 能够通过后续的用户问题回忆起大多数已存储的对话轮次，并且它（摘要在此处截断）

    arXiv:2610.36948v1 Announce Type: cross  Abstract: A large language model maintains conversational content in an external storage that is outside the model and grows as the dialogue continues. This work assumes an environment where a language model has no external memory and no prior knowledge of the future question or its answer. The motivation is to research whether a dialogue can be stored in the model parameters and later answer a question about it. This work proposes self-keyed in-parameter memorization (SKIM), a closed-form method in which the model generates its own questions about each dialogue turn and writes those questions into its parameters as the keys that later retrieve the turn's content. Generating many varied questions for each turn allows a later user question to match one even when phrased differently, and its answer is recalled from the parameters alone. Evaluated on five language models, SKIM recalls most stored dialogue turns from later user questions, and it ans
    
[^46]: JevRec：基于类Jev决策建模的语言学编码，面向基于大语言模型的生成式推荐

    JevRec: Linguistic Encoding with Jev-like Decision Modeling for LLM-based Generative Recommendation

    [https://arxiv.org/abs/2610.36937](https://arxiv.org/abs/2610.36937)

    提出JevRec，借鉴自然语言中高频词采用紧凑表达的特点，利用频率感知的可变长度语言学编码生成物品语义标识符，并为不同优化目标设计类Jev决策建模，从而在降低热门物品编码开销的同时保留长尾物品的丰富语义。

    

    生成式推荐（GR）将物品表示为语义标识符（SID），并从交互历史中自回归地生成下一个物品的SID。然而，尽管物品的交互频率高度不均衡，大多数方法仍为所有物品分配相同长度的SID。这种统一设计使热门物品反复付出细粒度编码的代价，而通过缩短编码来降低这一代价，又会压缩长尾物品的语义表示。受自然语言中高频表达呈现紧凑形式的启发，我们提出JevRec，将物品编码为可变长度的词，并从交互上下文中生成下一个物品。语言学编码保留了来自共享码本的频率相关语义前缀，从而为热门物品生成更短的SID，同时为长尾物品提供更丰富的描述。然而，预测序列级别的长度与生成词元级别的内容对优化提出了不同的要求，可能……（原文摘要在此处截断）

    arXiv:2610.36937v1 Announce Type: new  Abstract: Generative recommendation (GR) represents items as semantic identifiers (SIDs) and autoregressively generates the next item's SID from interaction history. However, most methods assign the same SID length to all items despite highly unequal interaction frequencies. This uniform design repeatedly incurs the cost of fine-grained codes for popular items, while shortening codes to reduce this cost also compresses the semantic representations of long-tail items. Inspired by the compact forms of frequent expressions in natural language, we propose JevRec to encode items as variable-length words and generate the next item from interaction context. Linguistic encoding retains frequency-dependent semantic prefixes from shared codebooks, producing shorter SIDs for popular items and richer descriptions for long-tail items. Yet predicting a sequence-level length and generating token-level content impose different optimization requirements that may c
    
[^47]: DocXEval：多模态文档理解中可解释性的系统性评估

    DocXEval: A Systematic Evaluation of Explainability in Multimodal Document Understanding

    [https://arxiv.org/abs/2610.36929](https://arxiv.org/abs/2610.36929)

    本文提出了首个面向多模态文档理解的综合性可解释性评估框架DocXEval，将现有XAI方法与指标适配到文本、布局和图像的多模态场景，并对13种方法进行了跨任务、数据集、模型和指标的系统性评估。

    

    多模态Transformer模型在现代文档理解领域推动了前所未有的性能突破。然而，这些模型固有的黑盒特性仍然是其在高风险领域安全部署的主要障碍。尽管针对视觉和语言任务的可解释性研究已取得显著进展，但面向多模态文档理解的可解释AI（XAI）研究却鲜受关注。为填补这一空白，我们提出了DocXEval，这是首个全面的可解释性框架，它将现有的XAI方法和评估指标适配到多模态（文本、布局和图像）文档理解场景中，同时针对语义实体识别等特定领域任务引入了高效的多目标方法和指标适配。借助DocXEval，我们系统地评估了13种XAI方法，涵盖3个文档理解任务、8个数据集、4个模型以及17个评估指标。

    arXiv:2610.36929v1 Announce Type: new  Abstract: Multimodal transformer models have driven unprecedented performance breakthroughs in modern document understanding. However, the inherent black-box nature of these models remains a major obstacle to their safe deployment in high-stakes sectors. Despite significant progress in explainability for vision and language tasks, explainable AI (XAI) research tailored to multimodal document understanding has received little attention. To address this gap, we present \ourframework{}, the first comprehensive explainability framework that adapts existing XAI methods and evaluation metrics to the multimodal (text, layout, and image) document understanding context, while introducing efficient multi-target adaptations of both methods and metrics for domain-specific tasks such as semantic entity recognition. Using \ourframework{}, we systematically evaluate 13 XAI methods across 3 document understanding tasks, 8 datasets, 4 models, and 17 evaluation met
    
[^48]: TrawMem：将智能体记忆重新构建为面向任务的角色自适应工作区

    TrawMem: Reframing Agent Memory as Task-Oriented Role-Adaptive Workspaces

    [https://arxiv.org/abs/2610.36918](https://arxiv.org/abs/2610.36918)

    该论文提出TrawMem，将智能体记忆系统重新构想为面向任务的角色自适应工作区，通过为记忆分配任务角色并联合选择互补记忆来组织交互历史，从而突破传统仅基于相关性的记忆检索范式。

    

    大语言模型（LLM）智能体依赖记忆系统来利用长期交互信息，然而现有系统通常基于与当前查询的相关性来检索记忆，并将其作为上下文证据使用。然而，仅有相关性并不能刻画一条记忆对任务解决的贡献方式：同一条记忆在不同任务中可能扮演不同角色，例如直接支持答案，或者作为桥梁链接到语义相关性较弱但包含必要信息的互补记忆。因此，我们将智能体记忆系统重新构建为面向任务的角色自适应工作区，其中记忆根据其面向任务的角色和关系被选择和组织。基于这一视角，我们提出了TrawMem，它将交互历史组织为局部线程，并通过分配角色和联合选择互补记忆来构建面向任务的工作区。在LoCoMo和MemGallery数据集上使用四个开源和闭源骨干模型的实验……

    arXiv:2610.36918v1 Announce Type: new  Abstract: Large language model (LLM) agents rely on memory systems to leverage long-term interactions, yet existing systems typically retrieve memories based on their relevance to the current query and use them as contextual evidence. However, relevance alone does not characterize how a memory contributes to task solving: the same memory may serve different roles across tasks, such as directly supporting the answer or linking to complementary memories with weaker semantic relevance but necessary information. We therefore reframe agent memory systems as Task-Oriented Role-Adaptive Workspaces, where memories are selected and organized by their task-oriented roles and relations. Based on this view, we propose TrawMem, which organizes interaction histories into local threads and constructs task-oriented workspaces by assigning roles and jointly selecting complementary memories. Experiments on LoCoMo and MemGallery with four open- and closed-source bac
    
[^49]: 我们需要大语言模型来知道来源为何冲突吗？面向RAG中知识冲突分类的决策模型

    Do We Need an LLM to Know Why Sources Conflict? Decision Models for Knowledge-Conflict Typing in RAG

    [https://arxiv.org/abs/2610.36915](https://arxiv.org/abs/2610.36915)

    小型类型化决策模型在识别RAG中检索来源的冲突类型方面，可以以低4至39倍的成本媲美甚至超越Gemini 2.5 Flash等前沿大语言模型。

    

    当为查询检索到的来源相互矛盾时，正确的回应取决于它们为何矛盾：一个存在多个有效答案的问题、一个有争议的话题、过时的信息和虚假的来源，各自需要不同的处理行为。Cattan等人（2025）表明，将冲突类型告知生成器可使其回应提升约24分，但即使对前沿大语言模型而言，预测冲突类型也很困难。我们研究类型化决策模型——在单次前向传播中返回固定选项的概率分布而非生成文本——能否快速且低成本地完成这一分类任务。我们使用CONFLICTS基准测试并原封不动地采用原论文的提示词，评估了由一个商业推理网关提供的全部11个决策模型（来自7家提供商），并在相同输入上与Gemini模型进行对比。直接提问时，决策模型达到50.0%至67.0%的准确率；11个模型中有10个达到或超过Gemini 2.5 Flash（论文所用模型，准确率61.3%），而成本低4至39倍。

    arXiv:2610.36915v1 Announce Type: new  Abstract: When the sources retrieved for a query disagree, the right response depends on why they disagree: a question with several valid answers, a contested topic, stale information and a false source each call for different behaviour. Cattan et al. (2025) showed that telling a generator the conflict type improves its responses by about 24 points, but that predicting the type is hard even for frontier LLMs. We study whether typed decision models, which return probabilities over fixed options in a single forward pass instead of generating text, can do this classification fast and cheaply. Using the CONFLICTS benchmark and the original paper's prompt verbatim, we evaluate all 11 decision models served by one commercial inference gateway, from 7 providers, against Gemini models on identical inputs. Asked directly, the decision models reach 50.0--67.0% accuracy; 10 of the 11 match or exceed Gemini 2.5 Flash, the paper's model (61.3%), at 4--39x lowe
    
[^50]: 开发巴斯克语语音语言模型

    Developing a Speech Language Model for Basque

    [https://arxiv.org/abs/2610.36911](https://arxiv.org/abs/2610.36911)

    本文提出了首个开源的巴斯克语语音语言模型，通过在LLaMA-Omni架构上微调Latxa模型并利用翻译语料与TTS合成音频，实现了低资源语言下的低延迟端到端口语交互。

    

    尽管大语言模型（LLM）的近期发展为多模态交互铺平了道路，但少数民族语言在跟上这一步伐方面面临困难。本工作提出了首个开源的巴斯克语语音语言模型，专为低延迟、端到端的口语交互而设计。该系统基于LLaMA-Omni架构，集成了Latxa模型作为骨干，通过微调多个模块将其巴斯克语能力迁移到语音模态。为解决数据稀缺问题，研究人员将英文VoiceAssistant-400K语料库翻译成巴斯克语，并使用巴斯克语TTS合成音频。评估结果证实了该系统的可行性：内容质量达到3.86/5分，语音可懂度的词错误率（WER）为9.94%，音频质量的UTMOS为3.07。这项工作表明，即使在低资源语言中，也可以利用模块化架构和合成数据开发先进的口语交互系统。

    arXiv:2610.36911v1 Announce Type: cross  Abstract: Although the recent development of LLMs has paved the way for multi-modal interaction, minority languages face difficulties keeping pace. This work presents the first open Speech Language Model for Basque, designed for low-latency, end-to-end spoken interaction. Based on the LLaMA-Omni architecture, the system integrates a Latxa model as its backbone, transferring its Basque capabilities to the spoken modality by fine-tuning several modules. To address data scarcity, the English VoiceAssistant-400K corpus was translated into Basque, and the audio was synthesized using a Basque TTS. Evaluation results confirm the system's viability, achieving 3.86/5 in content quality, a 9.94% WER for spoken intelligibility, and a 3.07 UTMOS for audio quality. This work demonstrates that advanced spoken interaction systems can be developed using modular architectures and synthetic data even in low-resource languages.
    
[^51]: 不遗漏任何内容：从大型随机评估语料库中寻找证据

    Nothing Left Behind: Finding Evidence from a Large Randomized Evaluation Corpus

    [https://arxiv.org/abs/2610.36910](https://arxiv.org/abs/2610.36910)

    该论文提出一种结合文本嵌入候选检索与语言模型证据过滤的两阶段方法，在7,354项随机试验语料库中找到了系统性综述应纳入证据的92%，远超传统标签搜索的45%。

    

    许多AI系统需要在大规模集合中找出所有相关条目，然后对找到的内容进行整理。标准做法是使用人工整理的词表为每个条目打上标签，再基于这些标签进行搜索。这些标签虽然有助于组织结果，但基于标签的搜索会遗漏不符合任何标签的条目，而当标签过于宽泛时又会返回过多结果。我们将该方法与一种基于嵌入的方法进行了比较：利用文本嵌入按语义寻找候选条目，再由语言模型仅保留相关条目。在一个包含7,354项随机试验的语料库中，标签搜索仅找到了系统性综述所纳入证据的45%，而新方法找到了92%，其每次查询的成本随搜索深度增加而上升。这些结果支持一种将宽泛的候选检索与针对特定查询的证据过滤相结合的两阶段架构。

    arXiv:2610.36910v1 Announce Type: new  Abstract: Many AI systems need to find every relevant items in a large collection and then organize what they found. The standard way to do this is to tag each item with labels from a curated vocabulary and search on those labels. Those labels are useful for organizing results, but searching on them misses items that do not fit a label and returns too much when labels are broad. We compare this approach with an embedding-based approach: find candidates by meaning using text embeddings, then have a language model keep only the relevant ones. In a corpus of 7,354 randomized trials, label search found 45% of the evidence a systematic review would include, while the new approach found 92%, at a cost per query that rises with search depth. These results support a two-stage architecture that combines broad candidate retrieval with query-specific evidence filtering.
    
[^52]: HiLo-RD：面向掩码扩散语言模型的跨预算自蒸馏

    HiLo-RD: Cross-Budget Self-Distillation for Masked Diffusion Language Models

    [https://arxiv.org/abs/2610.36901](https://arxiv.org/abs/2610.36901)

    提出HiLo-RD跨预算自蒸馏方法，通过将大窗口教师状态的隐藏表示与裁剪后的小窗口视图进行匹配，解决掩码扩散语言模型在有限生成预算下因过早提交答案片段而导致的推理偏离问题。

    

    我们提出HiLo-RD（高到低预算表示蒸馏）方法，以在有限生成预算下改进掩码扩散语言模型（MDLMs）。与自回归模型不同，MDLMs需要同时决定生成什么内容以及将内容固定在哪个位置。这带来了两个相互耦合的挑战：在推理完成之前预测有用的内容，以及决定何时固定那些会影响后续预测的答案词元。我们识别出了预算导致的过早提交问题：即使正确的解能够容纳在窗口内，模型也可能在解析中间步骤之前就固定了靠近窗口边界的答案片段。这些片段会成为后续预测的上下文，并使后续推理偏离正确方向。现有的后训练方法无法直接迁移同一模型在更大窗口下所遵循的更优轨迹。HiLo-RD通过以下方式弥补这一空白：（1）将大窗口教师状态与裁剪后的小窗口视图进行配对，（2）匹配二者的隐藏表示……（原文摘要在此处截断）

    arXiv:2610.36901v1 Announce Type: cross  Abstract: We propose HiLo-RD (High-to-Low Budget Representation Distillation) to improve masked diffusion language models (MDLMs) under a limited generation budget. Unlike autoregressive models, MDLMs choose both what to generate and where to commit it. This creates two coupled challenges: predicting useful content before reasoning is complete, and deciding when to fix answer tokens that shape later predictions. We identify budget-induced premature commitment: even when a correct solution fits, the model can fix answer fragments near the window boundary before resolving intermediate steps. These fragments become context for later predictions and redirect subsequent reasoning. Existing post-training methods do not directly transfer the stronger trajectories that the same model follows with a larger window. HiLo-RD addresses this gap by (1) pairing large-window teacher states with cropped small-window views and (2) matching their hidden representa
    
[^53]: RubricAtlas：从专家报告中挖掘类型级评分标准以评估深度研究智能体

    RubricAtlas: Mining Type-Level Rubrics from Expert Reports to Evaluate Deep Research Agents

    [https://arxiv.org/abs/2610.36878](https://arxiv.org/abs/2610.36878)

    RubricAtlas通过从3,079份中文金融研究专家报告中挖掘类型级分析要求，构建覆盖99种研究类型的评估框架，将二元检查点与报告段落关联，并把报告完整性与论证有效性分开评估，从而更全面地评估深度研究智能体。

    

    深度研究智能体执行多步骤网络搜索，并将检索到的信息综合成分析性报告。评估这些输出不仅需要检查相关事实是否存在，还需要检查它们提供了哪些分析。针对特定查询的二元评分标准能够使单个要求变得明确，但任务级检查清单可能使整体分析结构得不到充分检验。我们提出了RubricAtlas，它从同一研究类型下针对不同对象的专家报告中提炼出类型级的分析要求，并围绕反复出现的分析问题组织详细检查。这些要求为相关任务提供了关于“分析什么”的共同标准，而不规定其答案。我们的框架将二元检查点与报告段落关联起来，并将完整性评估与论证有效性评估分开进行。RubricAtlas基于3,079份中文金融研究报告构建，涵盖99种研究类型。参考报告选择实验……

    arXiv:2610.36878v1 Announce Type: new  Abstract: Deep research agents conduct multi-step Web searches and synthesise retrieved information into analytical reports. Evaluating these outputs requires checking which analyses they deliver, beyond the presence of relevant facts. Query-specific binary rubrics make individual requirements explicit, but task-level checklists may leave the overall analytical structure underexamined. We introduce RubricAtlas, which derives type-level analytical requirements from expert reports on different objects within the same research type and organises detailed checks around recurring analytical questions. The requirements give related tasks a shared standard for what to analyse, without prescribing their answers. Our framework links binary checkpoints to report passages and evaluates completeness separately from argument validity. Built from 3,079 Chinese financial research reports, RubricAtlas covers 99 research types. Reference-report selection experimen
    
[^54]: GUI-JEPA：候选条件化的JEPA潜在表示作为移动GUI智能体的快速动作策略

    GUI-JEPA: Candidate-Conditioned JEPA Latents as a Fast Action Policy for Mobile GUI Agents

    [https://arxiv.org/abs/2610.36877](https://arxiv.org/abs/2610.36877)

    GUI-JEPA是首个无解码器的GUI智能体策略，利用在无标注GUI截图上预训练的JEPA编码器潜在空间，通过候选条件化的潜在预测器和轻量级打分头在一次批量处理中为所有候选动作打分，无需逐token解码，实现了快速且省内存的移动GUI操作。

    

    移动GUI智能体通常是对70亿参数规模的视觉-语言模型进行微调，以文本形式生成每个动作。然而，这些VLM需要逐token解码每个动作，即使是那些对候选动作进行打分的智能体，也要在决策流程中保留大型语言模型或VLM，导致速度慢且内存消耗大。为解决这些问题，我们提出了GUI-JEPA，据我们所知，这是首个基于联合嵌入预测架构构建的无解码器GUI智能体策略，它在预训练于无标注GUI截图的JEPA编码器的潜在空间中选择动作。其动作和区域条件化的潜在预测器将屏幕潜在表示转换为每个候选动作对应的一个潜在表示，使轻量级打分头能够在一次批量处理中针对指令对所有候选动作进行打分，而无需解码任何token。我们进一步证明，打分器依赖于这一潜在通路，且预测器是为决策本身而训练的：一个随机初始化的预……

    arXiv:2610.36877v1 Announce Type: new  Abstract: Mobile GUI agents are typically fine-tuned 7B-scale vision-language models (VLMs) that generate each action as text. However, these VLMs decode every action token by token, and even agents that score candidate actions keep a large language model or VLM in the loop, making them slow and memory-hungry. To address these issues, we propose GUI-JEPA, to our knowledge the first decoder-free GUI agent policy built on a Joint-Embedding Predictive Architecture (JEPA), which selects actions in the latent space of a JEPA encoder pretrained on unlabelled GUI screenshots. Its action- and region-conditioned latent predictor turns the screen latent into one latent per candidate action, letting lightweight heads score all candidates against the instruction in a single batched pass, without decoding a token. We further show that the scorer relies on this latent pathway and that the predictor is trained for the decision itself: a randomly initialised pred
    
[^55]: AdaGround：面向GUI定位的令牌归一化自适应缩放与验证器在环细化方法

    AdaGround: Token-Normalized Adaptive Zooming with Verifier-in-the-Loop Refinement for GUI Grounding

    [https://arxiv.org/abs/2610.36874](https://arxiv.org/abs/2610.36874)

    AdaGround提出了一种免训练的GUI定位框架，通过依据目标大小自适应设定缩放参数、引入“是/否”验证器在环机制甄别并修正缩放结果、以及根据表头组合定位电子表格单元格，在高分辨率屏幕小目标定位上超越了现有最先进的免训练方法。

    

    GUI（图形用户界面）定位在高分辨率屏幕上的失败大多发生在小目标上，而放大画面是一种常见的免训练补救措施。然而，大多数现有的缩放流水线无法使缩放倍率与目标大小相匹配，会无条件接受缩放后的答案，且从不重新审视错误的答案。为解决这些问题，我们提出了AdaGround，一个包含三个组件的免训练框架：（1）令牌归一化自适应缩放，根据模型自身对目标大小的估计，为每个样本设定放大倍率、视野和缩放深度；（2）验证器在环细化，利用一个标记点击式的“是/否”验证器来保留或拒绝每次缩放、在缩放链之间做出选择、触发重新定位，并通过存在性查询决定何时弃权；（3）组合式结构定位，根据表头组合推导出电子表格单元格。在Qwen3-VL模型上，AdaGround超越了最先进的免训练缩放方法以及许多经过训练的模型。

    arXiv:2610.36874v1 Announce Type: cross  Abstract: GUI grounding on high-resolution screens fails mostly on small targets, and zooming in is a common training-free remedy. However, most existing zooming pipelines do not match the zoom to the target's size, accept the zoomed answer unconditionally and never revisit a wrong answer. To address these issues, we propose AdaGround, a training-free framework with three components: (1) Token-Normalized Adaptive Zooming, which sets the magnification, field of view and zoom depth of each sample from the model's own estimate of the target size; (2) Verifier-in-the-Loop Refinement, which uses one marked-click Yes/No verifier to keep or reject each zoom, choose between zoom chains, trigger re-grounding and, with an existence query, decide when to abstain; and (3) Compositional Structure Grounding, which composes spreadsheet cells from their headers. On Qwen3-VL models, AdaGround surpasses state-of-the-art training-free zooming methods and many trai
    
[^56]: 言语还是行动？大语言模型如何预测人机对话中用户的决策

    Words or Deeds? How LLMs Predict Users' Decisions in Human-Agent Dialogue

    [https://arxiv.org/abs/2610.36813](https://arxiv.org/abs/2610.36813)

    该研究发现大语言模型从历史行为记录预测用户合作决策的能力（AUC=0.80）优于对话言语信息，加入对话反而使模型更轻信用户的口头承诺，导致错误集中在用户违背承诺的轮次。

    

    网络系统越来越多地根据用户所说的话来预测用户的决策，然而当利益存在部分冲突时，用户可能为了换取对方的合作而口头承诺合作，随后却不予兑现。依赖此类承诺的销售或服务智能体可能会向随后不守承诺的用户做出让步，从而给其所代表的利益方造成损失。我们测试了大语言模型如何结合用户的言语陈述与决策记录来预测用户的决策，实验包含1,260轮游戏：42名用户与五个智能体进行重复囚徒困境式博弈（每个智能体被设定表现“大五”人格中的某一种主导特质），用户在每次决策前都可以与智能体对话。在八个开源权重的大语言模型上，基于决策记录的预测能较好地区分合作与背叛（AUC = .80），而加入对话信息并未带来可检测的提升。使用对话信息时，预测更倾向于与用户的言语陈述保持一致，而错误则集中在用户违背其承诺的轮次中。

    arXiv:2610.36813v1 Announce Type: new  Abstract: Web systems increasingly predict users' decisions from what users say, yet when interests partly conflict, users may promise to cooperate to win the other side's cooperation and then not do so. Sales or service agents that rely on such promises may grant concessions to users who then do not keep their promises, at a cost to the party they represent. We test how LLMs combine users' statements with decision records to predict users' decisions, using 1,260 rounds in which 42 users played a repeated prisoner's dilemma-style game with five agents, each prompted to express a dominant Big Five trait, and could talk with the agent before every decision. Across eight open-weight LLMs, predictions from decision records separated cooperation from defection well (AUC = .80), and adding dialogue brought no detectable gain. With dialogue, predictions agreed with users' statements more often, and errors concentrated on rounds in which users broke their
    
[^57]: 弱点定位并不足够：面向大语言模型的状态感知递归知识改进

    Weakness Localization Is Not Enough: State-Aware Recursive Knowledge Improvement for Large Language Models

    [https://arxiv.org/abs/2610.36811](https://arxiv.org/abs/2610.36811)

    该论文指出仅定位模型知识弱点不足以决定如何改进（存在“知识状态混叠”问题），并提出状态感知的递归知识改进方法，根据知识是缺失、难以访问还是下游无效等不同状态选择相应的干预策略。

    

    大语言模型的知识改进正从预定义的知识注入流程转向评估引导的更新。现有方法利用评估来定位模型特有的知识弱点，并据此指导有针对性的后训练。然而，识别“需要改进什么”并不一定能为决定“如何改进”提供充分的信息。同一个被定位出的弱点可能与多种不同的操作性知识状态相容：相关知识可能不可用、可用但难以可靠访问，或可访问但在下游使用中无效。我们将这种歧义称为“知识状态混叠”。当相容的知识状态倾向于不同的干预措施时，仅靠更高的诊断分辨率并不足以选择合适的干预方式。此外，模型更新可能在观察到的弱点消失之前就已改变底层知识状态，因此某一轮中合适的干预措施在后续轮次中可能不再适用……（摘要在此处被截断）

    arXiv:2610.36811v1 Announce Type: new  Abstract: Knowledge improvement in large language models is shifting from predefined knowledge incorporation pipelines toward evaluation-guided updating. Existing methods use evaluation to localize model-specific knowledge weaknesses and guide targeted post-training. However, identifying what to improve does not necessarily provide sufficient information for deciding how to improve it. The same localized weakness can be compatible with distinct operational knowledge states: the relevant knowledge may be unavailable, available but difficult to access reliably, or accessible but ineffective for downstream use. We call this ambiguity knowledge-state aliasing. When compatible states favor different interventions, greater diagnostic resolution alone is insufficient for intervention selection. Moreover, model updates can change the underlying knowledge state before the observed weakness disappears, such that an intervention appropriate in one round may 
    
[^58]: 基于多分支探索的潜在视觉对比推理

    Latent Visual Contrastive Reasoning with Multi-Branch Exploration

    [https://arxiv.org/abs/2610.36809](https://arxiv.org/abs/2610.36809)

    提出了一种分支感知的潜在视觉推理框架，通过生成多个问题条件的潜在视觉分支，并利用边界框证据标注进行软对比的分支-证据对齐训练，突破单一线性推理轨迹的局限，提升复杂视觉推理能力。

    

    多模态大语言模型在视觉问答方面取得了长足进步，但复杂的视觉推理往往需要定位并比较多条与问题相关的视觉证据。现有方法将中间推理过程迁移到视觉潜在空间中，但通常遵循单一的线性推理轨迹，当涉及多个视觉假设或相互竞争的证据区域时，这种方式存在局限。我们提出了潜在视觉对比推理，这是一种分支感知的潜在推理框架，可生成多个以问题为条件的潜在视觉分支，作为图像词元之上的软假设。为训练这些分支，我们的方法利用边界框证据标注构建目标证据表示，并引入了软对比的分支-证据对齐目标。该目标促使与证据相关的分支与标注的视觉区域对齐，同时将其与其他分支进行对比……（摘要原文在此处截断）

    arXiv:2610.36809v1 Announce Type: cross  Abstract: Multimodal large language models have made substantial progress in visual question answering, but complex visual reasoning often requires locating and comparing multiple pieces of question-relevant visual evidence. Existing methods that move intermediate reasoning into the visual latent space usually follow a single linear reasoning trajectory, which can be limiting when multiple visual hypotheses or competing evidence regions are relevant. We propose Latent Visual Contrastive Reasoning, a branch-aware latent reasoning framework that generates multiple question-conditioned latent visual branches as soft hypotheses over image tokens. To train these branches, our method uses bounding-box evidence annotations to build target evidence representations and introduces a soft contrastive branch-evidence alignment objective. This objective encourages evidence-relevant branches to align with annotated visual regions while contrasting them agains
    
[^59]: 人格潜变量：基于个人自身写作对冻结语言模型进行个体化条件控制

    Personality Latents: Conditioning Frozen Language Models on Individuals from Their Own Writing

    [https://arxiv.org/abs/2610.36805](https://arxiv.org/abs/2610.36805)

    该论文提出“人格潜变量”（PL）方法，无需人格标签或自撰简介，直接从个人写作中为其学习潜变量并作为软前缀注入冻结语言模型，使模型能够精确表达该个体的独特人格。

    

    控制语言模型的人格通常依赖于书面的人物简介或命名的特质类别，因此模型只能表达作者所写下的内容或分类体系所定义的内容。我们转而提出人格潜变量，它从每个人自己的写作中为其学习一个潜变量，无需任何人格标签，也无需撰写的人物简介，并将该潜变量作为软前缀注入冻结的语言模型。该方法遵循生成式潜变量优化的思路：每位训练个体获得一个自由潜变量，并通过冻结模型本身进行优化；同时，一个编码器学习预测这些潜变量，使得新个体只需一次前向传播即可获得其潜变量，并可在测试时进行可选的细化。我们在留出个体上进行评估：让经条件化的模型回答中性问题和访谈式问题，然后使用在真实人类写作上训练的评分器从其回答中读取人格特质。在 MBTI 各维度上，PL 取得了领先的 AUC 表现，优于人格嵌入和基于提示的方法。

    arXiv:2610.36805v1 Announce Type: new  Abstract: Controlling the personality of a language model usually relies on written profiles or named trait categories, so it can only express what an author writes down or what a taxonomy names. We instead propose Personality Latents (PL), which learn a latent for each person from their own writing, with no personality labels and no authored profile, and inject it into a frozen language model as a soft prefix. Following generative latent optimization, each training person receives a free latent that is optimized through the frozen model itself, while an encoder learns to predict these latents so that a new person needs only one forward pass, with optional refinement at test time. We evaluate on held-out people by having the conditioned model answer neutral and interview questions and reading personality from its answers with scorers trained on real human writing. On the MBTI axes PL reaches an AUC of , ahead of persona embedding and prompt-based 
    
[^60]: API-TAIL：通过温度对齐的似然尾部检测大语言模型API中的模型替换

    API-TAIL: Detecting Model Substitution in LLM APIs through Temperature-Aligned Likelihood Tails

    [https://arxiv.org/abs/2610.36801](https://arxiv.org/abs/2610.36801)

    提出API-TAIL典型性审计方法，通过温度对齐吸收合法解码配置造成的批次级偏移，再从似然尾部识别个别异常响应，从而检测LLM API服务中悄悄将部分请求替换为更廉价模型的行为。

    

    商业语言模型API仅返回文本，这使得服务提供商可以通过悄悄地将一部分用户请求路由到更便宜的替代模型、同时按所宣传的模型收费来削减成本。检测这种部分替换非常具有挑战性，因为诚实的提供商也会通过未公开的解码和服务配置选择（例如温度缩放、采样截断、重复惩罚与存在惩罚、默认系统提示词等）改变输出分布，这迫使审计必须针对合法配置构成的复合零假设进行检验，而非仅针对单一参考模型。我们的关键洞察是：这些合法的变化主要使整个批次响应的典型性发生偏移，而部分替换则会注入在所声称模型下依然异常的个别响应。基于这一洞察，我们提出了API-TAIL，这是一种典型性审计方法，它首先通过将单一有效温度对齐到中……（原文摘要在此处截断）

    arXiv:2610.36801v1 Announce Type: cross  Abstract: Commercial language model APIs return only text, allowing a provider to cut costs by quietly routing a fraction of user requests to a cheaper substitute model while billing for the advertised one. Detecting such partial substitution is challenging because honest providers also alter the output distribution through undocumented decoding and serving choices, such as temperature scaling, sampling truncation, repetition and presence penalties, and default system prompts, forcing an audit to test against a composite null of legitimate configurations rather than a single reference. Our key insight is that these legitimate variations primarily shift response typicality across the entire batch, whereas partial substitution injects individual responses that remain anomalous under the claimed model. Building on this, we introduce API-TAIL, a typicality audit that first absorbs batch-level shifts by aligning a single effective temperature to cent
    
[^61]: 语言模型会保留被放弃的分析：花园路径句之后的残留与适应

    Language Models Keep Abandoned Analyses: Residue and Adaptation after Garden-Path Sentences

    [https://arxiv.org/abs/2610.36797](https://arxiv.org/abs/2610.36797)

    该研究发现，语言模型在处理花园路径句后会留下两种痕迹：被放弃句法分析的残留会提升后续宾语附着倾向（效应约为真实宾语句的四分之三），而向已消解结构的适应随消歧误差缩放并主要影响理解类任务，激活修补技术可将二者区分开来。

    

    阅读"While the man hunted the deer ran into the woods"（当那个男人追猎那只鹿跑进了树林）这样的句子，需要放弃将"the deer"（鹿）视为"hunted"（追猎）宾语的分析。刚刚读过这类句子的人仍会产出更多及物结构，说明被放弃的分析会引导（启动）后续语言处理。我们探究语言模型是否会继续使用它们已经放弃的分析，以及模型中是什么承载着这一分析。在从1.24亿到700亿参数的24个开源模型中，花园路径句会在上下文中留下两种痕迹：其一，被放弃分析的残留会提高后续的宾语附着倾向，其效应约为名词短语真正作为宾语的句子效应的四分之三，且该残留出现在采样文本中，在留出测试集和经人类规范化的测试项上同样成立，并对第二类花园路径句也有效；其二，向已消解结构的适应随句子的消歧误差大小而缩放，且主要作用于类似理解的任务度量。激活修补技术将这两种痕迹区分开来。后续文本读……

    arXiv:2610.36797v1 Announce Type: new  Abstract: Reading "While the man hunted the deer ran into the woods" requires giving up the analysis in which "the deer" is the object of "hunted". People who have just read such a sentence still produce more transitive structures, so the abandoned analysis primes later language. We ask whether language models keep using an analysis they have abandoned, and what in the model carries it. In 24 open models from 124M to 70B parameters, a garden-path sentence leaves two traces in the context. A residue of the abandoned analysis raises later object attachment, by about three quarters of the effect of a sentence in which the noun phrase really is an object, and it appears in sampled text, on held-out and human-normed items and for a second type of garden path. An adaptation toward the resolved structure scales with the sentence's disambiguation error and acts mainly on comprehension-like measures. Activation patching separates the two. Later text reads 
    
[^62]: Select-TTT：面向原位测试时训练的查询条件化选择性快权重更新

    Select-TTT: Query-Conditioned Selective Fast Weight Updates for In-Place Test-Time Training

    [https://arxiv.org/abs/2610.36794](https://arxiv.org/abs/2610.36794)

    Select-TTT提出了一种模型无关的查询条件化规则，通过根据各数据块与用户查询的相关性来动态缩放其快权重写入强度，解决了原位测试时训练中统一写入调度导致无关内容稀释查询相关信息的问题。

    

    测试时训练（TTT）使部署后的语言模型能够将传入的上下文即时内化到一小组快权重参数中，而不仅仅是将其视为供注意力机制使用的瞬态上下文。原位TTT（In-Place TTT）最近通过将标准MLP块的下投影重新用作快权重，使这一思想在大型语言模型中变得实用。然而，我们发现其收益在真实的网页长上下文问答任务中基本消失。我们将此归因于原位TTT的统一写入调度：每个数据块都以相同强度写入，导致无关内容稀释了快权重记忆中与查询相关的信息。基于这一洞察，我们提出了Select-TTT，这是一种模型无关、查询条件化的原位TTT规则，它根据每个数据块与用户查询的相关性来缩放其写入强度。该相关性信号无需监督且无需新增参数：它直接从模型的冻结输出（此处摘要原文截断）中读取。

    arXiv:2610.36794v1 Announce Type: cross  Abstract: Test-time training (TTT) enables a deployed language model to internalize incoming context on-the-fly into a small set of fast-weight parameters, rather than merely treating it as transient context for attention. In-Place TTT recently made this idea practical for large language models by repurposing the down-projection of standard MLP blocks as fast weights. However, we find that its benefit largely disappears on realistic web long-context question answering. We attribute this to In-Place TTT's uniform write schedule: every chunk is written with equal strength, so irrelevant content dilutes query-relevant information in fast-weight memory. Building on this insight, we propose Select-TTT, a model-agnostic, query-conditioned rule for In-Place TTT that scales each chunk's write strength by its relevance to the user query. This relevance signal requires no supervision and no new parameters: it is read directly from the model's frozen outpu
    
[^63]: 批量化反思优化循环：并行提议使GEPA更快更好

    Batching the Reflective Optimization Loop: Parallel Proposals Make GEPA Faster and Better

    [https://arxiv.org/abs/2610.36774](https://arxiv.org/abs/2610.36774)

    将GEPA优化器扩展为每步并行提议和评估一批候选方案，使反思优化循环的运行时间缩短至原来的四分之一到二分之一，同时提升了测试分数和预算利用效率。

    

    反思式文本优化能够提升智能体系统的性能，但优化循环本身就是一个资源密集型的工作负载，其串行结构导致并行评估能力闲置。我们扩展了开源的GEPA优化器，使其在每个优化步骤中提议并评估一批候选方案，而不是每次只处理一个候选方案。在两项任务的实验中，大多数批量化运行在不超过一半的墙上时钟时间内完成，最快的仅需约四分之一到三分之一的时间。批量化设置还取得了更高的留出测试集分数：在LiveBench-Math上从68.9%提升到72.1%（使用2×2配置），在HoVer上从49.0%提升到60.0%（使用8×1配置）。一个简单的运行时模型解释了在有限工作池下测得的加速效果，批量化设置还能更高效地使用其指标调用和资金预算。我们提出了反思式优化步骤可以并行扩展的各个维度，这是对如何分配固定优化预算的首次探索。

    arXiv:2610.36774v1 Announce Type: new  Abstract: Reflective text optimization makes agentic systems better, but the optimization loop is itself a resource-hungry workload whose sequential structure leaves parallel evaluation capacity idle. We extend the open-source GEPA optimizer to propose and evaluate a batch of candidates on each optimization step instead of one candidate at a time. In our sweep on two tasks, most batched runs finished in half the wall-clock time or less, and the fastest in about a quarter to a third. Batched settings also achieved higher held-out test scores, from 68.9% to 72.1% on LiveBench-Math (with $2\times2$) and from 49.0% to 60.0% on HoVer (with $8\times1$). A simple runtime model explains the measured speedups under a finite worker pool, and batched settings also spend their metric-call and dollar budgets more efficiently. We frame the axes along which a reflective optimization step can scale in parallel, a first look at allocating a fixed optimization budg
    
[^64]: DocSkill：面向多语言多跳文档问答的技能集成式智能体推理

    DocSkill: Skill-Integrated Agentic Reasoning for Multilingual Multi-Hop Document Question Answering

    [https://arxiv.org/abs/2610.36759](https://arxiv.org/abs/2610.36759)

    DocSkill提出了一个具有可学习技能编排策略的智能体框架，通过离线MCTS探索技能轨迹生成SFT示范与DPO偏好对来训练模型，实现自适应的证据获取、跨页整合与答案验证，并发布了多语言多跳文档问答基准MMHop-DocBench。

    

    多语言多跳文档问答（MMHop-DocQA）需要在复杂版面、文档页面以及多种语言之间定位并整合证据。现有方法包括单次处理和自适应工具使用，但学习协调证据获取、跨页整合与答案验证仍然充满挑战。为应对这一挑战，我们提出了DocSkill，一个具备可学习的技能编排策略的智能体框架，用于实现自适应的证据获取与推理。该策略根据不断演化的证据状态来选择、重访或跳过相应技能。为构建技能编排的监督信号，我们利用有界蒙特卡洛树搜索（MCTS）在离线阶段探索备选的技能轨迹。这些轨迹为监督微调（SFT）提供了示范样本，并为直接偏好优化（DPO）提供了偏好对。此外，我们还引入了MMHop-DocBench，一个多语言多跳文档问答基准。

    arXiv:2610.36759v1 Announce Type: new  Abstract: Multilingual multi-hop document question answering (MMHop-DocQA) requires locating and integrating evidence across complex layouts, document pages, and diverse languages. Existing approaches include single-pass processing and adaptive tool use, but learning to coordinate evidence acquisition, cross-page integration, and answer verification remains challenging. To address this challenge, we present DocSkill, an agentic framework with a learned skill-orchestration policy for adaptive evidence acquisition and reasoning. The policy selects, revisits, or skips skills according to the evolving evidence state. To construct supervision for skill orchestration, bounded Monte Carlo Tree Search (MCTS) explores alternative skill trajectories offline. These trajectories provide demonstrations for supervised fine-tuning (SFT) and preference pairs for Direct Preference Optimization (DPO). We further introduce MMHop-DocBench, a multilingual multi-hop do
    
[^65]: CAMR：面向大语言模型持续学习的兼容性感知记忆回放

    CAMR: Compatibility-Aware Memory Replay for Continual Learning in Large Language Models

    [https://arxiv.org/abs/2610.36734](https://arxiv.org/abs/2610.36734)

    CAMR 提出了一种兼容性感知的自适应记忆回放框架，通过在线策略联合控制新样本监督与历史回放，在持续学习中平衡大语言模型对新知识的获取与对已有知识的保留。

    

    持续学习要求大语言模型在获取新知识和能力的同时保留已学到的内容。微调和回放虽然支持知识的获取与保留，但如何判断传入样本是否需要监督、以及应该重温哪些历史样本，仍然是一个挑战。我们提出了兼容性感知记忆回放（CAMR），这是一种自适应框架，能够联合控制当前样本的监督和历史回放。CAMR 将兼容性检查与学习到的记忆门控相结合，引导在线策略在三种动作之间做出选择：结合通用回放学习传入样本、联合强化兼容的相关样本、以及跳过对已验证掌握的重复样本的直接监督。该策略在每次到达的有界回放预算下，依据知识获取、保留和训练成本的反馈进行学习。我们使用 Qwen3-8B-Base 在持续事实更新以及跨任务、跨领域学习场景中评估了 CAMR。

    arXiv:2610.36734v1 Announce Type: new  Abstract: Continual learning requires large language models to acquire new knowledge and capabilities while retaining what they have learned. Fine-tuning and replay support acquisition and retention, but deciding whether incoming examples need supervision and which historical examples to revisit remains a challenge. We propose Compatibility-Aware Memory Replay (CAMR), an adaptive framework that jointly controls current-example supervision and historical replay. CAMR combines compatibility checks with a learned memory gate to guide an online policy over three actions: learning incoming examples with general replay, jointly reinforcing compatible related examples, and skipping direct supervision of verified mastered duplicates. The policy learns from acquisition, retention, and training-cost feedback under a bounded per-arrival replay budget. We evaluate CAMR with Qwen3-8B-Base on continual factual updating and cross-task and cross-domain learning. 
    
[^66]: 一种通过隔离异议来减少智能体过度防御性写作的框架

    A Harness Reducing over-Defensive Writing in Agents by Isolating Objections

    [https://arxiv.org/abs/2610.36717](https://arxiv.org/abs/2610.36717)

    该论文提出一种将审稿人、过滤器和写作者分离并通过独立上下文隔离异议的框架，使过度防御性写作减少96.61%，同时将源主张保留率从80.35%提升至93.12%。

    

    在扩展报告时，使用智能体写作可能会引入比文章本身更强的主张，然后再对其加以否定，这被称为过度防御性写作。我们假设这是因为智能体是在为回应一个对抗性的AI审稿人而写作，而不是为人类读者写作。当我们仅通过提示词在输出中禁止防御性写作时，防御性上下文也可能影响结果。为了减少过度防御性写作，我们构建了一个将审稿人、过滤器和写作者分离的框架。写作智能体只接收由过滤器挑选出的问题，过滤器拥有独立的上下文。我们在25份被扩展为正式报告的商业项目草稿上评估了该框架，结果显示防御性命题从59个减少到2个，降幅达96.61%，未发现任何无依据的事实添加，同时源主张保留率从80.35%提升至93.12%。

    arXiv:2610.36717v1 Announce Type: new  Abstract: When expanding a report, writing with agents may introduce a stronger claim than the article makes and then disclaim it. This is called over-defensive writing. We assume that this is because the agent is writing for responding to an adversarial AI reviewer, instead of a human reader. The defensive context may also influnce the consequent when we simply ban defensive writing in the output through prompts. In order to reduce over-defensive writing, we produce a harness that separates the reviewer, filter, and writer. The writer agent receives only the problems choosen by the filter, which is an independent context. We evaluate this harness on 25 business-project drafts expanded into formal reports, the result shows that the defensive propositions decreased from 59 to 2, which is a 96.61\% reduction, no unsupported factual additions were identified while the source-claim retention rises from 80.35% to 93.12%.
    
[^67]: 词元置信度仅仅是链长度吗？对推理轨迹的信号处理审计

    Is Token Confidence Just Chain Length? A Signal-Processing Audit of Reasoning Traces

    [https://arxiv.org/abs/2610.36709](https://arxiv.org/abs/2610.36709)

    本文用频谱分析和变点检测等信号处理工具审查发现，基于词元置信度滑动窗口最小值的思维链评分实际上主要反映链的长度，而仅按长度排序的效果与该置信度评分相当甚至更好。

    

    推理语言模型通过书写长思维链来回答数学问题，对每个问题采样多条思维链并进行多数投票可以提高准确率，但代价是生成大量文本。一种流行的降低成本的方法是根据模型对其写入的每个词元（token，即一个词或词的一部分）的置信度来为每条思维链评分。该方法在2048个词元的滑动窗口上对置信度取平均，并根据最低平均值来保留或截断思维链。我们将逐词元记录的置信度视为音频或无线电意义上的信号，并使用标准信号处理工具（如频谱分析和用于检测突变的统计检验）对该方法进行审查。我们使用一个小型开源推理模型，从两个数学基准的共500个问题中各采样16条思维链。结果发现，窗口最小值评分主要衡量的是思维链的长度。仅按长度对思维链排序的效果与置信度评分相当甚至更好（原摘要至此处被截断）。

    arXiv:2610.36709v1 Announce Type: new  Abstract: Reasoning language models answer math problems by writing long chains of thought, and sampling many chains per problem and taking a majority vote raises accuracy at a large cost in generated text. A popular way to cut that cost scores each chain by the model's confidence in each token it writes, where a token is a word or part of a word. The method averages this confidence over a sliding window of 2048 tokens and keeps or stops chains based on the lowest average. We treat the confidence, recorded token by token, as a signal in the sense of audio or radio, and examine the method with standard signal-processing tools such as spectral analysis and statistical tests for detecting a sudden change. We sampled 16 chains for each of 500 problems from two math benchmarks with a small open reasoning model. The windowed-minimum score turns out to measure mostly how long a chain is. Ranking chains by length alone does as well as or better than the c
    
[^68]: 儿童尺度语言模型中的专业化、多样性与规模化

    Specialization, Diversity, and Scaling in Child-Scale Language Models

    [https://arxiv.org/abs/2610.36679](https://arxiv.org/abs/2610.36679)

    在儿童尺度数据上训练的小型语言模型会对单个儿童的语言环境产生词汇和分布层面的专业化，多儿童数据训练虽能提升对汇总语言的预测，但对语法能力的提升并不明显。

    

    现代语言模型训练所用的语言数据远超儿童实际接触的语言量，但我们对从小型自然语言数据集中学习的过程知之甚少。我们在来自密集家庭录音的儿童语音转录文本和合成的儿童导向对话数据上训练小型GPT-2模型，并改变训练数据的数量、来源和多样性。在单个儿童语言环境数据上训练的模型，对该环境的预测效果显著优于对其他儿童环境的预测。这种专业化很大程度上体现在词汇和分布层面：一个简单的5-gram模型就能重现这种跨儿童的模式（r=.91）。与仅在单个儿童输入上训练相比，在多个儿童的数据上训练能明显提升对汇总保留语言的预测，但在语法基准测试上却没有出现类似的明显提升。在不同的数据规模下，自然数据和合成训练数据也会产生不同的语法和词汇表现。

    arXiv:2610.36679v1 Announce Type: cross  Abstract: Modern language models are trained on far more language than children experience, but we know much less about learning from small, naturalistic datasets. We train small GPT-2 models on transcripts of speech with children from dense home recordings (BabyView) and on synthetic child-directed dialogue (TinyDialogues), varying the amount, source, and diversity of training data. Models trained on one child's language environment predict that environment substantially better than other children's environments. Much of this specialization is lexical and distributional: a simple 5-gram model reproduces the cross-child pattern (r=.91). Training on data from multiple children clearly improves prediction of pooled held-out language relative to training on a single child's input without a similarly clear improvement on grammatical benchmarks. Across data sizes, naturalistic and synthetic training inputs also produce different grammatical and lexic
    
[^69]: 前沿大语言模型中阿希从众效应的定量研究方法

    A Quantitative Approach to Asch Conformity in Frontier Large Language Models

    [https://arxiv.org/abs/2610.36668](https://arxiv.org/abs/2610.36668)

    该论文提出一种定量方法，通过对单个问题重复采样获得答案分布，并比较模型接触人为制造共识前后的分布差异，证实前沿大语言模型存在从众效应，并量化了从众发生的条件与程度。

    

    阿希从众实验最近作为衡量大语言模型（LLM）从众性的一种方法受到了广泛关注。虽然以往的研究主要集中于证明LLM中存在从众现象，但我们的工作为这一方法增加了更进一步的定量层面。我们的方法不是在一组问题上对前沿LLM进行采样，而是对单个问题进行重复采样，利用LLM的概率特性产生答案的分布。通过这种方式，我们可以比较模型在被展示人为制造的共识之前和之后的答案分布。我们的实验证实了前沿LLM确实存在从众行为，这强化了以往文献的结论，并且通过我们的分布间比较，提供了关于导致从众的条件以及从众发生程度的额外定量数据。

    arXiv:2610.36668v1 Announce Type: cross  Abstract: The Asch conformity experiment has recently gained considerable attention as a way of measuring conformity in large language models (LLMs). While previous studies have focused on demonstrating conformity in LLMs, our work adds a further quantitative layer to this approach. Instead of sampling a frontier LLM on a suite of questions, our approach samples a single question repeatedly, leveraging the probabilistic nature of LLMs to produce a distribution of answers. In doing so, we can compare the distributions of answers before and after the model is presented with the manufactured consensus. Our experiments confirm that frontier LLMs do conform, reinforcing prior literature, and, through our distribution-to-distribution comparisons, provide additional quantitative data on the conditions that lead to conformity and the degree to which it occurs.
    
[^70]: 基于大语言模型与分段话语表示理论（SDRT）的日语话语结构分析

    Japanese Discourse Structure Analysis Based on Segmented Discourse Representation Theory Using LLMs

    [https://arxiv.org/abs/2610.36667](https://arxiv.org/abs/2610.36667)

    本研究利用大语言模型翻译构建了日语SDRT语料库，证明了基于SDRT的话语解析具有跨语言适用性，并发现日语话语标记词影响甚微，模型预测主要依赖上下文话语信息。

    

    基于分段话语表示理论的话语解析对于捕捉对话的结构动态至关重要。然而，以往的研究大多集中于英语，由于缺乏日语SDRT语料库，其在日语上的有效性尚不明确。为解决这一问题，我们通过使用大语言模型翻译现有的英语SDRT语料库，构建了日语SDRT语料库。通过对基于提示词的方法和微调方法的大语言模型进行对比评估，我们表明这些模型在英语和日语上取得了相当的性能，这表明基于SDRT的话语解析具有跨语言适用性。为了探究英语和日语解析器为何能达到相似的性能，我们对话语标记词和上下文输入进行了消融分析。结果表明，我们所考察的大多数日语话语标记词影响甚微，且模型的预测在很大程度上依赖于上下文话语信息。

    arXiv:2610.36667v1 Announce Type: new  Abstract: Discourse parsing based on Segmented Discourse Representation Theory (SDRT) is crucial for capturing the structural dynamics of dialogues. However, most prior research focuses on English, and its effectiveness for Japanese remains unclear owing to the lack of Japanese SDRT corpora. To address this issue, we construct Japanese SDRT corpora by translating existing English SDRT corpora using an LLM. Through a comparative evaluation of prompt-based and fine-tuned LLMs, we show that the models achieve comparable performance between English and Japanese, suggesting the cross-lingual applicability of SDRT-based discourse parsing. In order to investigate why the English and Japanese parsers achieve similar performance, we conduct ablation analyses on discourse markers and contextual inputs. Our results reveal that most of the Japanese discourse markers we examined have little impact, and the models' predictions depend heavily on contextual disco
    
[^71]: 面向 Mamba-2 状态压缩的平衡状态子空间

    Balanced State Subspaces for Mamba-2 State Compression

    [https://arxiv.org/abs/2610.36617](https://arxiv.org/abs/2610.36617)

    论文发现 Mamba-2 的状态能量集中于递归累积的偏置方向，而输出实际读取的内容位于与其坐标不对齐的低维子空间，据此提出一种基于完整 Gramian 平衡截断的训练后压缩方法，使压缩直接针对写入-读取子空间而非单个坐标。

    

    Mamba-2 将所有上下文压缩到一个固定大小的循环状态中，该状态主导了解码阶段的内存占用和带宽。现有的压缩器对状态值进行量化或对状态坐标进行剪枝，并在原生基底下独立地为每个通道打分。我们检查了训练后的 Mamba-2 状态，发现其能量几乎是一秩的，但输出并不会读取这些高能量方向：能量集中在由递归累积的类似偏置的方向上，而输出所读取的内容则位于一个与状态坐标不对齐的低维子空间中。即使是输出感知的评分方法也是逐坐标计算的，因此会错过决定输入-输出映射的那些方向。因此，压缩应当针对写入-读取子空间而非单个坐标。我们提出了 \method{}，这是一种训练后变换，它利用完整的可控性与可观测性 Gramian 矩阵实现平衡截断，并应用……（原文摘要在此处被截断）

    arXiv:2610.36617v1 Announce Type: cross  Abstract: Mamba-2 compresses all context into a fixed-size recurrent state that dominates decoding memory and bandwidth. Existing compressors quantize state values or prune state coordinates, scoring each channel independently in the native basis. We examine trained Mamba-2 states and find that energy is almost rank-one, but the output does not read the energetic directions: energy concentrates in a bias-like direction accumulated by the recurrence, while the content the output reads lives in a low-dimensional subspace that is not aligned with the state's coordinates. Even output-aware scores are computed coordinate by coordinate, so they miss the directions that determine the input--output map. Compression should therefore target the write--read subspace rather than individual coordinates. We propose \method{}, a post-training transform that instantiates balanced truncation with full controllability and observability Gramians and applies the ro
    
[^72]: 面向大语言模型基于地标的路线指令的结构化地图

    Structured Maps for Landmark-Based Route Instructions with LLMs

    [https://arxiv.org/abs/2610.36582](https://arxiv.org/abs/2610.36582)

    本文提出用SVG格式将地图数据结构化为文本表示，使大语言模型在路线导航的地标选择任务中表现与图像输入相当或更优，并证明了小型本地模型经微调后也可胜任该任务。

    

    为了生成基于地标的路线指令，模型必须同时理解地图几何结构和人类对地标的偏好。尽管大语言模型在语言任务中展现出强大的能力，但为这些模型构建地图数据的最佳方式尚未确立。本文提出了一种用于路线导航中地标选择任务的结构化文本表示方法，利用SVG格式来编码空间特征。这种结构化文本与渲染的地图图像保持几何关联，使语言模型能够直接访问空间和语义数据。基于从人类撰写的路线描述和OpenStreetMap数据创建的新数据集，我们在两个任务上评估了模型：转弯处的地标选择和目的地周围的地标选择。对五个大语言模型的实验表明，所提出的结构化文本输入的表现与基于图像的PNG输入相当甚至更好。此外，任务特定的微调使较小的本地大语言模型能够（摘要在此处截断）

    arXiv:2610.36582v1 Announce Type: new  Abstract: To generate landmark-based route instructions, models must understand both map geometry and human landmark preferences. Although large language models (LLMs) show great abilities in language tasks, the best way to structure map data for these models is not yet established. This paper proposes a structured text representation for landmark selection tasks in route navigation, utilizing SVG format to encode spatial features. The structured text maintains a geometric connection with rendered map images, allowing language models to directly access spatial and semantic data. Using a new dataset created from human-written route descriptions and OpenStreetMap data, we evaluate models on two tasks: landmark selection at turns and around destinations. Experiments with five LLMs show that the proposed structured text input performs as well as or better than image-based PNG inputs. In addition, task-specific fine-tuning enables smaller local LLMs to
    
[^73]: 跨多模态大语言模型层级解耦一致性与冲突的假新闻检测方法

    Decoupling Consistency and Conflict across MLLM Layers for Fake News Detection

    [https://arxiv.org/abs/2610.36569](https://arxiv.org/abs/2610.36569)

    提出MLCCD方法，利用多模态大语言模型不同层级所编码的知识，对图文一致性与冲突进行多层级动态解耦，从而更精准地捕捉稀疏冲突并提升多模态假新闻检测性能。

    

    网络内容的可信度正面临来自多模态假新闻日益增长的威胁。真实新闻通常呈现图文语义对齐的特征，而假新闻往往在不同层面存在少量冲突，例如物体、场景、事件和声明的不匹配。然而，现有方法通常以粗粒度且静态的方式处理一致性与冲突，限制了其捕捉稀疏冲突和适应不同冲突程度的能力。受大语言模型不同层级编码不同知识的启发，我们提出了多层级一致性与冲突解耦方法（MLCCD），从多模态大语言模型（MLLM）的不同层级捕捉不同层面的冲突。具体而言，我们首先利用MLLM的最终决策表示来筛选与任务相关的视觉和文本token，防止无关内容稀释稀疏冲突；然后，我们保留表示发生变化的token……

    arXiv:2610.36569v1 Announce Type: cross  Abstract: The trustworthiness of Web content faces a growing threat from multimodal fake news. Real news usually shows aligned image-text semantics, while fake news often contains a few conflicts at varying levels, such as mismatches in objects, scenes, events, and claims. However, existing methods usually treat consistency and conflict in a coarse and static way, limiting their ability to capture sparse conflicts and adapt to varying conflict levels. Inspired by the distinct knowledge encoded across layers in large language models, we propose Multi-Level Consistency and Conflict Decoupling (MLCCD), which captures conflicts at different levels from different layers of multimodal large language models (MLLMs). Specifically, we first use the MLLM's final decision representation to select visual and textual tokens relevant to the task, preventing irrelevant content from diluting sparse conflicts. Then, we preserve tokens whose representations chang
    
[^74]: QESChunker：单一目标统一RAG中的重叠与非重叠分块

    QESChunker: A Single Objective Unifies Overlapping and Non-Overlapping Chunking for RAG

    [https://arxiv.org/abs/2610.36557](https://arxiv.org/abs/2610.36557)

    提出QESChunker，以统一的问题-证据效用最大化目标将RAG中的非重叠与重叠分块统一为有监督组合优化问题，非重叠情形可通过动态规划在多项式时间内精确求解。

    

    分块是检索增强生成（RAG）中的一个基础预处理步骤，然而现有的大多数分块器主要依赖启发式方法。我们提出了问题-证据监督分块器（QESChunker），据我们所知，这是首个从问题-证据监督中构建静态分块索引的方法。QESChunker通过在人工标注或大语言模型生成的问题-证据对上进行优化来构建文档特定的分块索引，且无需访问留出的测试查询。我们通过一个统一的问题-证据效用最大化目标来表述非重叠和重叠索引构建，从而将分块转化为一个有监督的组合优化问题。在非重叠约束下，该目标可以在各个分块上进行分解，并可通过动态规划在多项式时间内精确优化。当允许重叠时，该目标变为单调次模的；其精确优化是NP难的，但我们的g…（原文摘要在此处截断）

    arXiv:2610.36557v1 Announce Type: cross  Abstract: Chunking is a fundamental preprocessing step in retrieval-augmented generation (RAG), yet most existing chunkers rely primarily on heuristics. We introduce the Question--Evidence-Supervised Chunker (QESChunker), to our knowledge the first method for constructing static chunk indices from question--evidence supervision. QESChunker constructs document-specific chunk indices by optimizing over human-annotated or LLM-generated question--evidence pairs, without access to held-out test queries. We formulate non-overlapping and overlapping index construction through a unified question--evidence utility-maximization objective, turning chunking into a supervised combinatorial optimization problem. Under the non-overlap constraint, the objective decomposes over chunks and can be optimized exactly in polynomial time by dynamic programming. When overlap is allowed, the objective becomes monotone submodular; exact optimization is NP-hard, but our g
    
[^75]: PRISM-BENCH：超越最终答案准确率的有据可依且忠实可靠的多模态推理诊断

    PRISM-BENCH: Diagnosing Grounded and Faithful Multimodal Reasoning Beyond Final-Answer Accuracy

    [https://arxiv.org/abs/2610.36554](https://arxiv.org/abs/2610.36554)

    PRISM-BENCH是一个包含1000道专家标注问题的多模态推理基准，通过感知探测和轨迹对齐两种诊断方法，揭示前沿模型约85%的失败源于感知错误，并能独立于答案长度预测答案正确性。

    

    最终答案准确率只能反映视觉语言模型是否答对了，却无法说明其答案是否真正立足于图像、其陈述的推理过程是否反映了该证据，以及生成答案付出了什么代价。而这些正是决定一个系统能否被部署的关键属性。我们提出PRISM-BENCH，这是一个包含1,000道专家精心策划问题的基准，涵盖五个代表性不足的推理类别，其中每个条目都附有一条由一名人类标注者撰写、并由另一名标注者验证的理想化推理轨迹。这些轨迹支持两种准确率无法提供的诊断方法。感知探测要求模型基于自己撰写的描述作答，结果将前沿模型约85%的失败归因于感知错误，并将瓶颈定位在视觉定位能力上。轨迹对齐能够预测所有25个被测系统的答案正确性（宏观AUROC为0.66），且在长度控制下依然有效，因此它衡量的是推理内容而非冗长程度。

    arXiv:2610.36554v1 Announce Type: cross  Abstract: Final-answer accuracy reports whether a vision-language model was right, but not whether its answer was grounded in the image, whether its stated reasoning reflects that evidence, or what the answer cost to produce. Those are the properties that decide whether a system can be deployed. We present PRISM-BENCH, a benchmark of 1,000 expert-curated questions spanning five under-represented reasoning categories, in which every item carries an idealized reasoning trace written by one human annotator and validated by another. The traces support two diagnostics that accuracy cannot provide. A perception probe, in which the model answers from a description it wrote itself, attributes roughly 85% of frontier-model failures to misperception and places the bottleneck at grounding. Trace alignment predicts answer correctness for all 25 systems tested (macro AUROC 0.66) and survives length control, so it measures reasoning content and not verbosity.
    
[^76]: 基于语言学感知错误驱动优化的多语言自动语音识别（ASR）智能体数据采样框架

    Agentic Data Sampling Framework for Multilingual ASR via Linguistic-Aware Error-Driven Optimization

    [https://arxiv.org/abs/2610.36538](https://arxiv.org/abs/2610.36538)

    该论文提出了一种将LLM智能体引入多语言ASR训练循环的智能体数据采样框架，通过从音系、形态和句法等语言学角度分析识别错误，自适应地选择不同层级的数据采样策略以优化模型训练。

    

    多语言自动语音识别（ASR）在大规模预训练模型和多语言语音语料库的推动下取得了快速进展。以往研究已经证明了数据采样在多语言ASR性能中的关键作用。然而，现有的采样策略在很大程度上依赖于耗时的人工和基于启发式规则的设计。为解决这一问题，我们提出了一种语言学感知的智能体数据采样框架，将数据采样大语言模型（LLM）智能体引入多语言ASR的训练循环中。该智能体由基于语言学的评价准则驱动，从音系学、形态学和句法学的角度分析ASR错误，并利用由此产生的反馈，结合训练统计数据、预算约束和算法配置，来选择域级、样本级和词元级的采样策略。这种设计将评估结果转化为自适应的数据选择决策，而不是将WER/CER仅仅视为最终终点指标

    arXiv:2610.36538v1 Announce Type: new  Abstract: Multilingual automatic speech recognition (ASR) has achieved rapid progress with large-scale pretrained models and multilingual speech corpora. Previous studies have proved the pivotal role of data sampling in multilingual ASR performance. However, existing sampling strategies largely rely on time-consuming manual, heuristic-based designs. To address this, we propose a linguistic-aware agentic data sampling framework that introduces a data sampling LLM agent into the multilingual ASR training loop. The agent is driven by a linguistically grounded rubric that analyzes ASR errors from phonological, morphological, and syntactic perspectives, and uses the resulting feedback together with training statistics, budget constraints, and algorithm profiles to select domain-level, sample-level, and token-level sampling strategies. This design turns evaluation results into adaptive data-selection decisions rather than treating WER/CER as final endpo
    
[^77]: RoboAuto：通过多层次物理理解实现自主机器人程序搜索

    RoboAuto: Autonomous Robot Program Search through Multi-Level Physical Understanding

    [https://arxiv.org/abs/2610.36533](https://arxiv.org/abs/2610.36533)

    RoboAuto 提出了一个在物体、场景和交互三个层次上显式组织物理理解的机器人程序搜索框架，利用分层知识库整合可执行技能与物体中心知识，使基于 LLM 的智能体能够更有效地搜索和改进机器人操作程序。

    

    近年来，基于大语言模型（LLM）的自主智能体在迭代生成、评估和改进可执行方案方面展现出强大能力。在机器人操作领域，Code-as-Policy 使这些智能体能够在模块化、可编辑的机器人程序中进行搜索。然而，有效的程序搜索需要物理理解能力：智能体必须理解物体如何被操作、场景几何如何约束动作，以及交互为何成功或失败。现有方法主要依赖执行反馈来改进程序，往往无法解决这些物理层面的问题。我们提出了 RoboAuto，这是一个机器人程序搜索框架，通过三个组件在物体、场景和交互三个层次上显式地组织物理理解。首先，一个分层知识库将可执行技能与以物体为中心的知识（包括可供性、空间关系和动作效果）相结合，这些知识从具身数据和在线任务中学习获得……（原文在此处截断）

    arXiv:2610.36533v1 Announce Type: cross  Abstract: Recently, LLM-based autonomous agents have shown strong capabilities in iteratively generating, evaluating, and refining executable solutions. In robotic manipulation, Code-as-Policy lets these agents search over modular and editable robot programs. However, effective program search requires physical understanding: agents must understand how objects can be manipulated, how scene geometry constrains actions, and why interactions succeed or fail. Existing methods rely mainly on execution feedback for program refinement, often leaving these physical questions unresolved. We introduce RoboAuto, a framework for robot program search that explicitly organizes physical understanding at the object, scene, and interaction levels through three components. First, a hierarchical knowledge base combines executable skills with object-centric knowledge of affordances, spatial relations, and action effects, learned from embodied data and online task so
    
[^78]: ETC-Bench：面向评估侧转录文本规范化的多语言基准

    ETC-Bench: A Multilingual Benchmark for Evaluation-Side Transcript Canonicalization

    [https://arxiv.org/abs/2610.36527](https://arxiv.org/abs/2610.36527)

    本文提出了 ETC-Bench，这是首个覆盖 12 种语言、181 种现象类型的最全面的评估侧转录文本规范化多语言基准，并提出了兼顾规范化准确性与语义保持的 SENA 评估指标。

    

    arXiv:2610.36527v1 公告类型：new 摘要：同一语音的书面转录文本之间可能存在并非由语音信号唯一决定的差异。我们提出了评估侧转录文本规范化（ETC）的概念，即当多种表面形式之间的差异不受语音决定时，将其映射为单一的规范形式。据我们所知，ETC-Bench 是首个也是最全面的面向评估侧转录文本规范化的多语言基准，涵盖 12 种语言中的 181 种现象类型。每个实例都是一个最小对立对，即共享相同的句子上下文，仅在体现某一现象的目标片段上有所不同。为了同时评估规范化的成功与否和语义保留程度，我们提出了语义保持的精确匹配规范化准确率（SENA），该指标将独立规范化后两个对成员之间的精确一致性与防止信息丢失的对相对 BERTScore 约束相结合。我们进一步引入了分类法引导的智能体规范化器……（原文摘要在此处截断）

    arXiv:2610.36527v1 Announce Type: new  Abstract: Written transcripts of the same speech can differ in ways that are not uniquely determined by the speech signal. We formulate evaluation-side transcript canonicalization (ETC), which maps multiple surface forms to a single canonical form when their differences are not determined by speech. To our knowledge, ETC-Bench is the first and most comprehensive multilingual benchmark for evaluation-side transcript canonicalization, covering 181 phenomenon types across 12 languages. Each instance is a minimal pair that shares a sentence context and differs only in a target span exhibiting one phenomenon. To evaluate both canonicalization success and semantic preservation, we propose Semantic-preserving Exact-match Normalization Accuracy (SENA), which combines exact agreement between independently canonicalized pair members with a pair-relative BERTScore constraint against information loss. We further introduce the Taxonomy-guided Agent Canonicaliz
    
[^79]: Transformer中证据效应的因果定位

    Causal Localization of Evidence Effects in Transformers

    [https://arxiv.org/abs/2610.36524](https://arxiv.org/abs/2610.36524)

    该论文提出受控证据聚合（CEA）与平衡效用因果选择（BUC）方法，在Pythia-410M模型中通过因果干预定位出驱动语言模型利用自然语言证据进行决策的关键注意力头集合，并提供了从有限样本恢复归因结果的理论保证。

    

    受控符号输入能否揭示那些影响语言模型基于自然语言证据做出决策的注意力头？我们提出了受控证据聚合（CEA）方法，该方法在其余内容保持固定的提示中改变两个符号的数量和顺序，并测量冻结的Transformer对两个答案词元的偏好变化。偏好通过二者的logit差异来衡量；提示中不规定任何正确答案。平衡效用因果选择（BUC）通过联合评估两方面来识别紧凑的头集合：一是消融这些头的输出能在多大程度上消除数量效应，二是在不同数量条件之间修补平均输出能在多大程度上双向恢复该效应。我们推导了将头归因与干预效应联系起来的条件界，并明确了何时可以从有限样本中恢复归因候选列表。在Pythia-410M模型中，针对四种排序情形分别选择的1-12个注意力头的消融集合消除了65%（摘要在此处截断）

    arXiv:2610.36524v1 Announce Type: new  Abstract: Can controlled symbolic inputs reveal attention heads that influence a language model's decisions from natural language evidence? We introduce controlled evidence aggregation (CEA), which varies the counts and order of two symbols in otherwise fixed prompts and measures changes in a frozen Transformer's preference between two answer tokens. The preference is measured by their logit difference; the prompts prescribe no correct answer. Balanced utility causal selection (BUC) identifies compact head sets by jointly evaluating how much ablating their outputs removes the count effect and how closely patching mean outputs between count conditions recovers the effect in both directions. We derive conditional bounds linking head attribution to intervention effects and specifying when the attribution shortlist can be recovered from finite samples. In Pythia-410M, ablating sets of 1-12 heads selected separately for four ordering regimes removes 65
    
[^80]: 跨智能体交接的任务接管基准测试

    Benchmarking Task Takeover Across Agent Handoffs

    [https://arxiv.org/abs/2610.36523](https://arxiv.org/abs/2610.36523)

    该论文提出了HSC-Bench基准测试（含1,039个任务），用于评估智能体任务交接中接收模型恢复仓库证据与继续既有工作状态的能力，发现结构化交接状态能提升所有模型的证据恢复能力和大多数模型的任务继续性能。

    

    随着智能体在更长任务上的协作，责任可能会在工作完成之前于不同模型之间传递。接收模型必须在已有的决策、观察和行动基础上继续工作，同时满足剩余的任务要求。评估这种交接转换需要将交接时保存的信息与接收者如何继续工作联系起来，而这种关系是单纯的任务成功率和记忆保留分数无法捕捉的。我们提出了交接状态连续性基准测试（HSC-Bench），包含1,039个任务，用于评估任务接管的两个互补方面：恢复相关的代码仓库证据，以及从已建立的对话、工具调用和文件工作状态中继续任务。对六个接收模型的比较表明，结构化的交接状态提升了所有模型的仓库证据恢复能力以及大多数模型的任务继续性能。两个评估维度中最强的接收模型有所不同，这表明证据恢复与任务继续是两种不同的能力。

    arXiv:2610.36523v1 Announce Type: new  Abstract: As agents collaborate on longer tasks, responsibility may pass between models before the work is complete. The receiving model must build on prior decisions, observations, and actions while addressing the remaining requirements. Evaluating this transition requires connecting the information preserved at handoff to how the receiver continues the work, a relationship that task-success and memory-retention scores alone do not capture. We introduce the Handoff State Continuity Benchmark (HSC-Bench), comprising 1,039 tasks that evaluate two complementary aspects of task takeover: recovering relevant repository evidence and continuing from established conversational, tool, and file-work state. Comparisons across six receiving models show that structured handoff state improves repository evidence recovery for every model and continuation performance for most models. The strongest receivers differ between the two tracks, showing that evidence-re
    
[^81]: 将注意力偏置折叠进键向量使压缩的KV缓存能在标准内核上运行

    Folding Attention Biases into Keys Makes Compacted KV Caches Run on Standard Kernels

    [https://arxiv.org/abs/2610.36520](https://arxiv.org/abs/2610.36520)

    该论文提出一种闭式折叠方法，将注意力匹配压缩KV缓存时引入的logit偏置直接折叠进键向量本身，使压缩后的缓存无需修改即可在FlashAttention和vLLM等标准注意力内核上运行，且精度与原始AM方法基本相当。

    

    注意力匹配（AM）通过在每个注意力头保留部分键、重新拟合其值并为每个保留键的logit添加偏置来压缩KV缓存，但FlashAttention和vLLM的分页缓存没有针对该偏置的输入接口。本文将偏置沿每个注意力头一个基于上下文查询拟合的闭式解方向折叠进键中，从而得到一个具有AM的大小和头维度以及模型自身查询路径的普通缓存；输出误差由每个查询在该方向上的投影偏离1的程度所界定。在相同的每头预算下，未经修改的vLLM通过KV连接器即可服务折叠后的缓存，在所有51个Qwen3-4B问题上与Hugging Face的答案完全一致，并在200个Qwen3-14B LongBench样本中的180个上与其贪心输出一致。在QuALITY（Qwen3-1.7B、4B、8B、Llama-3.1-8B）和LongHealth（三个Qwen3模型）上使用相同预算时，折叠方法在14个设置中的13个与AM的精度差异在噪声范围内，并且在两者均可运行的12个设置中，其与完整缓存的KL散度均低于直接丢弃偏置的做法。

    arXiv:2610.36520v1 Announce Type: cross  Abstract: Attention Matching (AM) compacts a KV cache by keeping some keys per head, refitting their values and adding a bias to each kept key's logit. FlashAttention and vLLM's paged caches have no input for it. We fold the bias into the keys along one closed-form direction per head, fitted on context queries, giving a plain cache with AM's size and head dimension and the model's own query path; the output error is bounded by how far each query's projection on that direction is from 1. With equal per-head budgets, unmodified vLLM serves folded caches through a KV connector, matching Hugging Face's answers on all 51 Qwen3-4B questions and its greedy output on 180 of 200 Qwen3-14B LongBench samples. With equal budgets on QuALITY (Qwen3-1.7B, 4B, 8B, Llama-3.1-8B) and LongHealth (three Qwen3 models), the fold is within noise of AM's accuracy in 13 of 14 settings, with lower KL to the full cache than dropping the bias in all 12 where both ran. Unde
    
[^82]: MedConstBench：衡量“宪法”如何引导大语言模型医疗伦理行为的基准测试

    MedConstBench: Benchmarking How Constitutions Steer LLMs in Medical Ethics

    [https://arxiv.org/abs/2610.36513](https://arxiv.org/abs/2610.36513)

    该论文提出 MedConstBench，首个将部署“宪法”作为受控变量、围绕比彻姆四大伦理原则构建、并通过人工校准的多模型同行评审评分的医疗伦理基准，实现了对宪法如何引导大语言模型医疗伦理行为的系统分析。

    

    随着大语言模型的快速发展，它们被越来越多地用于回答医学问题，且许多此类模型如今已在本地部署。模型一旦部署，其行为主要由写入其“宪法”中的伦理政策所决定。然而，现有研究大多关注事实正确性，而非相互冲突的伦理价值之间的权衡。即使是那些确实评估伦理的基准测试，也将伦理视为模型的固定属性而非变量，因此“宪法”如何重塑医疗伦理行为在很大程度上仍未被探索。我们提出 MedConstBench，一个将部署宪法作为受控变量的医疗伦理基准测试，它围绕比彻姆的四大伦理原则组织，并通过人工校准的多模型同行评审进行评分。该基准包含我们自主构建的临床困境数据集 MedDilemma，并支持对“宪法”如何引导大语言模型进行系统性分析。对十个开源（模型）的评估……

    arXiv:2610.36513v1 Announce Type: new  Abstract: As large language models advance rapidly, they are increasingly used to answer medical questions, and many such models are now deployed locally. Once a model is deployed, its behavior is governed mainly by the ethical policy written into its constitution. However, most existing work focuses on factual correctness rather than the trade-off between competing ethical values. Even benchmarks that do evaluate ethics treat it as a fixed property of the model rather than as a variable, so how a constitution reshapes medical-ethics behavior remains largely unexplored. We introduce MedConstBench, a medical-ethics benchmark that treats the deployment constitution as a controlled variable, is organized around Beauchamp's four principles, and is scored by human-calibrated multi-model peer review. It includes MedDilemma, our own dataset of clinical dilemmas, and supports a systematic analysis of how constitutions steer LLMs. Evaluating ten open-sourc
    
[^83]: AI科学家知道什么是“新”吗？

    Do AI Scientists Know What's New?

    [https://arxiv.org/abs/2610.36511](https://arxiv.org/abs/2610.36511)

    该论文提出了一个结合时间真值的分析框架，从提案和论断两个层面评估AI科学发现中的新颖性检查，发现对非新颖工作拒绝率高的检查往往会误拒那些后来被人类研究者实现的新颖想法。

    

    AI科学发现流程面临的一个重大挑战是如何判断一个生成的想法是否具有真正意义上的新颖性。理想情况下，这些流程中的新颖性检查必须能够拒绝非新颖的想法，同时不过度拒绝新颖的想法；此外，它们还应能够区分提案中哪些部分是非新颖的，而不是不透明地丢弃整个提案。我们设计了一个分析框架，在提案层面和论断层面结合时间真值来评估新颖性检查，该框架使用了四个共包含1,060个提案的数据集，这些数据集由已发表的论文、审稿人争议记录，以及与该领域后来实际发表的成果相匹配的生成想法构建而成。利用该框架对现有新颖性检查进行审计，我们发现对非新颖工作拒绝率高的检查，往往也会对后来被人类研究者实现的想法产生大量误报；同时我们还观察到，旨在对提案进行整体评估的检查能够（精准指出非新颖部分）……

    arXiv:2610.36511v1 Announce Type: cross  Abstract: A major challenge facing AI scientific discovery pipelines is how to determine when a generated idea is meaningfully novel. Ideally, novelty checks in these pipelines must be able to reject non-novel ideas while not rejecting too many novel ideas, and they should also be able to distinguish which parts of a proposal are non-novel, as opposed to opaquely discarding entire proposals. We design an analytical framework for evaluation of novelty checks with temporal ground truth at both the proposal- and claim-level, using four datasets of 1,060 proposals built from published papers, reviewer disputes, and generated ideas matching ones the field went on to publish. Auditing existing novelty checks with this framework, we find that checks with high rejection rates on non-novel work also tend to have high false alarms on ideas that are later realized by human researchers, and observe that checks built to evaluate proposals as a whole can poin
    
[^84]: AC-JEPA：用于视觉到符号翻译的锚定跨模态联合嵌入预测架构

    AC-JEPA: Anchored Cross-Modal Joint Embedding Predictive Architecture for Visual-to-Symbolic Translation

    [https://arxiv.org/abs/2610.36493](https://arxiv.org/abs/2610.36493)

    该论文提出AC-JEPA，一种与监督微调兼容的即插即用模块，通过在跨模态两个方向上预测被掩码的中间状态的联合嵌入目标，来利用不完美的配对数据，从而在保留token级监督的同时提升视觉到符号跨模态翻译对目标序列错误的鲁棒性。

    

    跨模态翻译要求模型既能捕捉跨模态间的共享语义，又能保留目标模态所需的细粒度结构。然而在现实世界的科学领域中，配对数据往往并不完美，而大规模获取干净的配对数据成本高昂。监督微调（SFT）通过分类式的token级别监督直接优化跨模态翻译，但该目标对目标序列中的错误较为敏感。JEPA（联合嵌入预测架构）提供了一种互补的表示空间层面的目标，但现有的大多数形式侧重于模态内的预测，而非跨模态的翻译。我们提出了锚定跨模态JEPA（AC-JEPA），这是一个与SFT兼容的即插即用模块，利用配对数据在两个跨模态方向上预测被掩码的中间状态。联合嵌入目标聚焦于可从上下文预测的跨模态结构，而因果SFT路径则保留直接的token级别监督……（原文摘要在此处截断）

    arXiv:2610.36493v1 Announce Type: new  Abstract: Cross-modal translation requires models to capture shared semantics across modalities while preserving the fine-grained structure required by the target modality. Yet in real-world scientific domains, paired data are often imperfect, while obtaining clean pairs at scale is costly. Supervised fine-tuning (SFT) directly optimizes cross-modal translation through categorical token-level supervision, but this objective is sensitive to errors in the target sequences. JEPA offers a complementary representation-space objective, but most existing formulations focus on prediction within a modality rather than translation across modalities. We introduce Anchored Cross-Modal JEPA (AC-JEPA), an SFT-compatible plug-in that predicts masked intermediate states in both cross-modal directions using paired data. The joint-embedding objective targets cross-modal structure predictable from context, while the causal SFT path retains direct token-level supervi
    
[^85]: 一颗心形符号意味着递归：探索视觉语言模型中的视觉含义与交流

    A Heart Means Recursion: Exploring Visual Meaning and Communication in VLMs

    [https://arxiv.org/abs/2610.36449](https://arxiv.org/abs/2610.36449)

    研究发现视觉语言模型会被简单视觉符号隐性地引导行为（如心形符号引导模型采用递归解法），且模型间视觉交流存在“生成—解读”鸿沟——模型发送的直观视觉线索往往无法有效引导其他模型。

    

    视觉语言模型（VLM）是否具备与我们相同的视觉交流能力？我们分享了初步研究发现，从一个令人惊讶的观察开始：简单的视觉符号（如红色心形）能够在看似无关的任务上强烈影响模型行为，例如在简单编程问题上引导模型倾向于递归解法。值得注意的是，这种引导效应在很大程度上是隐性的，因为模型对视觉符号的直接解读和思维轨迹与其观察到的实际行为存在显著差异。在模型之间的视觉交流中，这导致了一种“生成—解读”鸿沟：模型倾向于发送它们自己（以及人类）觉得直观的形状基元，但这些视觉线索往往无法有效引导其他模型。在涉及抽象概念和故事的进一步交流游戏中，模型能够创造性地使用视觉元素相互传递意义，尽管其中某些交流内容对人类而言仍不直观。

    arXiv:2610.36449v1 Announce Type: new  Abstract: Do vision-language models (VLMs) share our ability to communicate visually? We share preliminary findings, starting from a surprising observation: simple visual symbols like a red heart can strongly influence model behavior on seemingly unrelated tasks, such as steering models towards recursive solutions on simple coding problems. Notably, this steering effect is largely implicit, as models' direct interpretations and thinking traces about visual symbols diverge substantially from their observed behavior. In visual communication between models, this leads to a generation--interpretation gap: while models tend to send shape primitives that they (and humans) find intuitive, these visual cues often fail to steer other models. In further communication games involving abstract concepts and stories, models can creatively use visuals to transmit meaning to one another, although some communications remain non-intuitive to humans. These findings 
    
[^86]: CONCUR：面向多任务、多奖励强化学习的协调辅助奖励

    CONCUR: Concordant Auxiliary Rewards for Multi-Task, Multi-Reward Reinforcement Learning

    [https://arxiv.org/abs/2610.36437](https://arxiv.org/abs/2610.36437)

    提出CONCUR优势估计器，通过将门控辅助奖励的分数约束为与主要奖励排序一致，解决多任务多奖励强化学习中辅助奖励干扰主要奖励排序的问题。

    

    当大语言模型（LLM）在多个任务上通过强化学习（RL）进行训练时，某个任务的主要奖励会对辅助奖励（如长度奖励）进行门控：在失败的响应上，辅助奖励不能超过主要奖励。然而，基于组相对的优势估计器将门控辅助奖励与其他组件同等对待，因此它可能颠倒主要奖励的排序，并且当主要奖励达到上限时，仅由辅助奖励来决定响应的排序。保留门控辅助奖励会损害某些任务，而丢弃它们又会损害其他任务。我们提出CONCUR，一种优势估计器，它将每个门控辅助奖励的标准化分数替换为满足两个条件的最接近分数：具有更高主要奖励的响应不能获得更低的分数，以及主要奖励达到上限的响应获得相等的分数。CONCUR-T在整个训练过程中应用这两个条件，而CONCUR-A对于比……（原文截断）更常与主要奖励排序一致的辅助奖励则取消第二个条件。

    arXiv:2610.36437v1 Announce Type: new  Abstract: When a Large Language Model (LLM) is trained with Reinforcement Learning (RL) on multiple tasks, a task's primary reward can gate auxiliary rewards such as a length reward: on a failing response, an auxiliary cannot exceed the primary. However, group-relative advantage estimators treat a gated auxiliary like any other component, so it can reverse the primary's order, and at the primary's ceiling only auxiliaries order responses. Keeping gated auxiliaries hurts some tasks, while dropping them hurts others. We propose CONCUR, an advantage estimator that replaces the standardized scores of each gated auxiliary with the closest scores that satisfy two conditions: responses with higher primary rewards cannot receive lower scores, and responses at the primary's ceiling receive equal scores. CONCUR-T applies both conditions throughout training, and CONCUR-A lifts the second for an auxiliary that agrees with the primary's order more often than i
    
[^87]: 无需分词器仍有差距：同形字与不可见字符水印的跨语言公平性

    No Tokenizer, Still a Gap: Cross-Lingual Fairness of Homoglyph and Invisible-Character Watermarking

    [https://arxiv.org/abs/2610.36432](https://arxiv.org/abs/2610.36432)

    该研究在24种文字系统的40种语言变体上系统评测了同形字与不可见字符水印，发现尽管这类编辑式水印完全不依赖分词器，但其容量仍存在高达3.3倍的跨语言差异且96%由文字系统决定，表明跨语言公平性差距依然存在。

    

    同形字水印和不可见字符水印属于基于编辑的水印。它们不改变模型生成文本的方式，而是在文本生成之后在其中隐藏信号：要么将部分字符替换为同形字（即外观相同或非常相似、但Unicode码点不同的字符），要么插入不可见的字符，这一切都无需访问生成该文本的模型。生成时水印由于分词器切分文本的方式，对低资源语言的效果较差。由于这类水印作用于单个字符、完全不涉及分词器，我们因此探究它们能否避开这一差距。我们在24种文字系统下的40种语言变体上进行了测量，包括容量（每个字符可隐藏的比特数）、有多少容量能在常规处理和针对性移除后存留，以及密钥检测器需要多少文本。结果显示容量差异达3.3倍，其中文字系统解释了96%的差异。在拉丁文字系统内部，资源水平……

    arXiv:2610.36432v1 Announce Type: new  Abstract: Homoglyph and invisible-character watermarks are edit-based watermarks. Instead of changing how a model generates text, they hide a signal in the text after it is written. They swap some characters for homoglyphs, which look the same, or very similar, but have different Unicode code points, or they insert characters that do not display, all without access to the model that wrote the text. Generation-time watermarks work less well for lower-resourced languages due to how tokenizers split their text. Because they work on individual characters, these watermarks never touch a tokenizer, so we ask whether they escape this gap. Across 40 language varieties in 24 scripts, we measure capacity (the hidden bits each character can carry), how much of it survives routine handling and targeted removal, and how much text a keyed detector needs. Capacity varies 3.3-fold and script accounts for 96\% of that variation. Within Latin script, resource level
    
[^88]: SOMA：一个用于细粒度自我中心行为与身体运动理解的基准

    SOMA: A Benchmark for Fine-Grained Egocentric Behavior and Body-Motion Understanding

    [https://arxiv.org/abs/2610.36384](https://arxiv.org/abs/2610.36384)

    该论文提出了SOMA基准，通过1,632个经人工验证的问题从身体构型、运动动态和时间推理三个维度评估自我中心视频中的细粒度身体运动理解能力，发现37个领先的MLLM表现不佳，甚至落后于同样认为该任务困难的人类。

    

    感知人体运动对于理解和辅助人类的多模态系统至关重要。然而，当前对多模态大语言模型（MLLM）的评估主要集中于物体、场景和粗粒度动作，模型对身体本身进行推理的能力尚未得到充分评估。这从自我中心（第一人称）视角来看尤其具有挑战性，因为在自我中心视角下，身体仅部分可见，运动必须从被遮挡且在时间上分散的线索中推断出来。我们提出了SOMA，一个用于从自我中心视频中理解细粒度身体运动的基准，包含1,632个经人工验证的问题，涵盖三个能力维度：身体构型、运动动态和时间推理。我们评估了37个领先的闭源、开源以及专门针对自我中心的MLLM，并辅以大规模的人类评估。我们发现，仅凭视频完成该任务对人类而言也很困难（准确率为68.5%），但人类的表现仍明显优于（模型）。

    arXiv:2610.36384v1 Announce Type: cross  Abstract: Perceiving human body motion is critical for multimodal systems that understand and assist people. Yet current evaluations of Multimodal Large Language Models (MLLMs) largely focus on objects, scenes, and coarse actions, leaving their ability to reason about the body itself poorly assessed. This is especially challenging from the egocentric perspective, in which the body is only partially visible and motion must be inferred from occluded and temporally distributed cues. We introduce SOMA, a benchmark for fine-grained body-motion understanding from egocentric video, comprising 1,632 human-verified questions across three capability axes: body configuration, motion dynamics, and temporal reasoning. We evaluate 37 leading closed-source, open-source, and specialized egocentric MLLMs, complemented by an extensive human evaluation. We find that humans also find this task difficult from video alone (68.5% accuracy), yet clearly outperform the 
    
[^89]: AK_QMUL参加ArGuard共享任务：使用单一微调JAIS模型进行阿拉伯语有害提示检测

    AK_QMUL at ArGuard Shared Tasks: A Single Fine-Tuned JAIS for Arabic Harmful-Prompt Detection

    [https://arxiv.org/abs/2610.36360](https://arxiv.org/abs/2610.36360)

    AK_QMUL团队仅用一个经4位QLoRA微调的JAIS-6.7B模型，在ArGuard-2026阿拉伯语有害提示检测共享任务的两个赛道（二分类和七分类）上均获得第一名，宏F1分别达到0.984和0.790。

    

    我们展示了AK_QMUL团队提交给ArGuard-2026阿拉伯语有害提示检测共享任务的成果，包含二分类安全/不安全任务（B1）和七分类危害领域任务（B2）。两个提交均使用同一骨干网络——采用4位QLoRA微调的JAIS-6.7B，并在其最后一个token的表示上添加线性分类头。对于B1任务，我们训练了一个四分类头，将标签按语域和方言分解为{安全， 不安全} x {现代标准阿拉伯语(MSA), 其他}，并利用来自AraSafe的MSA提示对训练集进行数据增强，最后使用简单的argmax解码来判定安全/不安全。对于B2任务，我们将相同的骨干网络微调为七分类器，并对五个随机种子的softmax输出取平均值。两个提交均在各自赛道排名第一，B1的宏平均F1达到0.984，B2的宏平均F1达到0.790。我们按方言和类别报告结果：在B1上，模型在全部六种方言中均表现准确（93.6%–99.8%），且其预测的不安全比例与真实比例高度吻合；在B2上，七个类别中有六个达到了F1

    arXiv:2610.36360v1 Announce Type: new  Abstract: We present AK_QMUL's submissions to the ArGuard-2026 shared tasks on Arabic harmful-prompt detection: a binary safe/unsafe task (B1) and a seven-way harm-domain task (B2). Both submissions use one backbone -- JAIS-6.7B fine-tuned with 4-bit QLoRA and a linear classification head over its last-token representation. For B1 we train a four-way head that factors the label by dialect register {safe, unsafe} x {MSA, other}, we augment the training fold with MSA prompts from AraSafe, and decode with a plain argmax to score safe/unsafe. For B2 we fine-tune the same backbone as a seven-way classifier and average the softmax outputs of five random seeds. Both submissions ranked first on their track, with macro-F1 0.984 on B1 and 0.790 on B2. We report results by dialect and by class: on B1 the model is accurate across all six dialects (93.6--99.8%) and its predicted unsafe rate tracks the true rate closely; on B2 six of the seven classes reach F1 
    
[^90]: 先检测再消歧：基于扰动的LLM输入歧义检测

    Detect Before You Disambiguate: Perturbation-Based Input Ambiguity Detection in LLMs

    [https://arxiv.org/abs/2610.36353](https://arxiv.org/abs/2610.36353)

    本文提出一种保持标签的扰动框架，通过黑盒规则扰动和白盒重要性引导两种检测器，在LLM对输入消歧之前检测输入歧义，在开放域、对话和视觉问答数据集上均优于现有的输出侧不确定性方法。

    

    大型语言模型（LLMs）常常将含糊的输入当作只有单一意图的含义来回答。这是有问题的，因为输入歧义是不确定性的上游来源：当一个提示存在多种有效解释时，关键问题是模型是否应该直接回答。现有的输入不确定性方法通常依赖于LLM生成的陈述或澄清，但这种转换可能在测量歧义之前就已经对输入进行了消歧。我们提出了一个保持标签的扰动框架用于输入歧义检测，其中扰动在保持接近原始输入的同时保留其标签。我们用两种检测器实例化这一原则：一种基于规则的黑盒扰动库，以及一种利用模型内部信号的白盒重要性引导方法。在开放域、对话式和视觉问答领域的四个数据集上的实验表明，我们的方法优于输出侧不确定性方法。

    arXiv:2610.36353v1 Announce Type: new  Abstract: Large language models (LLMs) often answer ambiguous inputs as if they had a single intended meaning. This is problematic because input ambiguity is an upstream source of uncertainty: when a prompt admits multiple valid interpretations, the key question is whether the model should answer directly at all. Existing input uncertainty methods often rely on LLM-generated statements or clarifications, but such transformations may already disambiguate the input before the ambiguity is measured. We propose a label-preserving perturbation framework for input ambiguity detection in which perturbations remain close to the original input while preserving its label. We instantiate this principle with two detectors: a black-box rule-based perturbation library and a white-box importance-guided method using model-internal signals. Experiments across four datasets in open-domain, conversational, and visual QA show that our methods outperform output-side u
    
[^91]: 跨模态转向向量在图像与文本生成中均真实存在

    Cross-modal Steering Vectors Exist Across Both Image and Text Generation

    [https://arxiv.org/abs/2610.36269](https://arxiv.org/abs/2610.36269)

    该论文发现在统一多模态模型中，针对某一模态（图像或文本）找到的因果性转向向量能够零样本迁移到另一模态并有效引导生成，证明模型中的概念机制在图像与文本两种模态间是统一实现的。

    

    当前AI模型中的概念与机制是跨模态统一的，还是针对每个模态分别独立实现的？我们通过研究跨模态转向来将这一关于模态统一性的宽泛问题具体化：1）找到能够成功引导某一模态token生成的向量；2）将这些向量零样本迁移，在添加到另一模态的token上时同样能够引导生成。我们的研究超越了简单的相关性对齐方法（例如表征相似性分析），因为转向在本质上具有因果性。我们在多个统一多模态模型（UMMs）中研究跨模态转向，因为这些模型在语言生成和图像生成上拥有共享参数，我们聚焦于这两个模态。在情感、物体大小等多种被引导的概念上，我们证明了转向向量可以在两个方向上（文本到图像、图像到文本）实现零样本迁移。即使对于高度特属于某一模态的概念（例如正式的……（原文摘要在此处截断）

    arXiv:2610.36269v1 Announce Type: cross  Abstract: Are concepts and mechanisms unified across modalities in current AI models, or instead implemented separately for each one? We operationalize this broad question of modality unification by studying cross-modal steering: 1) finding vectors that successfully steer generation on tokens of one modality, which then 2) transfer zero-shot to steer generation when added to tokens of another modality. Our study goes beyond simple correlational alignment (e.g. representational similarity) because steering is fundamentally causal. We study cross-modal steering in several unified multimodal models (UMMs), since they possess shared parameters for language and image generation, focusing on these two modalities. Across diverse steered concepts such as emotion and object size, we show that steering vectors transfer zero-shot in both directions (text-to-image & image-to-text). Even for concepts that are highly unique to one modality (e.g. formal vs. co
    
[^92]: 超越IceBERT：基于开放数据的冰岛语ModernBERT编码器（参数规模从12M到1B）

    Beyond IceBERT: Open-Data ModernBERT Encoders for Icelandic from 12M to 1B Parameters

    [https://arxiv.org/abs/2610.36230](https://arxiv.org/abs/2610.36230)

    该研究基于90亿token开放数据训练了从12M到1B参数的ModernBERT冰岛语编码器，其1B模型是首个该规模的单语冰岛语编码器，在全部11个任务上显著超越IceBERT，并在受控实验中验证了从英语及多语言模型热启动的迁移效果。

    

    IceBERT作为冰岛语的参考编码器，其研究发现经过清洗的网络爬虫数据足以训练出强大的模型，且从多语言模型出发（热启动）在某些任务上有所帮助。我们采用ModernBERT架构重新审视了这两个发现。我们在90亿（9B）token的开放数据上训练了参数量从1200万（12M）到10亿（1B）的冰岛语编码器，并在统一协议（含置信区间）下针对30个编码器重新运行了IceBERT的基准测试及五个更新的任务。我们完全从零训练的1B模型，据我们所知是首个该规模的单语冰岛语编码器，在全部十一个任务上都显著优于IceBERT；而从相同结构的英语Ettin编码器热启动后，它在所有参与排名的任务上都取得了所有编码器中的最佳成绩。由于我们的模型结构与英语Ettin编码器和多语言mmBERT相匹配，我们还能在受控条件下测试热启动效果，涵盖六种模型规模和四种token预算。通过移植我们的词表，war（摘要在此处截断）

    arXiv:2610.36230v1 Announce Type: new  Abstract: IceBERT, the reference encoder for Icelandic, found that a cleaned web crawl is enough to train a strong model and that starting from a multilingual model can help on some tasks. We revisit both findings with the ModernBERT architecture. We train Icelandic encoders from 12M to 1B parameters on 9B tokens of open data, and rerun IceBERT's benchmarks and five newer tasks for 30 encoders under one protocol with confidence intervals. Our 1B model trained from scratch, to our knowledge the first monolingual Icelandic encoder of that size, is significantly better than IceBERT on all eleven tasks; warm-started from the English Ettin encoder of the same shape, it has the best score of all encoders on every task we rank. Because our model shapes match the English Ettin encoders and the multilingual mmBERT, we can also test the warm start under controlled conditions, at six model sizes and four token budgets. With our vocabulary transplanted, a war
    
[^93]: 基础模型能够检测到提示分类器所遗漏的隐藏意图

    Base Models Can Detect Hidden Intentions that Prompted Classifiers Miss

    [https://arxiv.org/abs/2610.36202](https://arxiv.org/abs/2610.36202)

    本文提出人格-头部似然评估（PHLE）方法，利用基础预训练模型对输出的似然评估来解码并分类后训练模型中的潜意识隐蔽信号，从而检测出提示分类器无法发现的隐藏意图（如隐蔽的伤害用户倾向）。

    

    人格选择模型（PSM）假设后训练过程是从模型在预训练期间学到的诸多人格中选择其一。关于潜意识学习（subliminal learning）的研究表明，模型可以将隐蔽信号（如“对猫头鹰的喜爱”）传递到语义上无关的输出（如数字串）中，并且在这些输出上训练的学生模型会继承这些特质。在本工作中，我展示了通过在基础预训练模型上配合适当的头部提示（header）来评估此类输出的似然——这种方法我称之为人格-头部似然评估（PHLE），可以解码这些潜意识信号并对它们进行分类。例如，PHLE 能够区分由被提示“你喜爱猫头鹰”的后训练模型生成的数字序列与被提示“你喜爱鹰”的模型生成的数字序列，并且能够区分一个秘密怀有伤害用户意图的模型（已经过滤以去除任何明显的失调迹象）的回答与没有系统提示的模型的回答。在这两种情况下，提示

    arXiv:2610.36202v1 Announce Type: cross  Abstract: The Persona Selection Model (PSM) posits that post-training selects a persona the model learned during pre-training. Work on subliminal learning has shown that models can impart covert signals (like "love of owls") into semantically unrelated outputs like strings of numbers, and that students trained on these outputs inherit those traits. In this work I show that by evaluating the likelihood of such outputs under the base pre-trained model with an appropriate header, a method I call persona-header likelihood evaluation (PHLE), one can decode these subliminal signals and classify between them. For example, PHLE distinguishes number sequences generated by a post-trained model prompted with "You love owls" from those prompted with "You love eagles", and distinguishes answers from a model with a secret desire to harm the user, filtered to remove any noticeable misalignment, from those of a model with no system prompt. In both cases prompte
    
[^94]: 学习阅读盲文：基准测试并提升视觉语言模型的盲文识别能力

    Learning to Read Braille: Benchmarking and Improving Braille Recognition Capabilities of VLMs

    [https://arxiv.org/abs/2610.36195](https://arxiv.org/abs/2610.36195)

    该论文提出盲文识别基准BrailleBench与合成盲文生成流水线，证明在其合成数据上训练的2B参数视觉语言模型可将一级盲文识别精确匹配率提升至87%、二级盲文提升至61%，超越现有独立视觉语言模型并在二级盲文上取得最佳成绩。

    

    尽管视觉理解和文档转录领域进展迅速，现代视觉语言模型在一项基本的无障碍任务上仍然表现不佳：从图像中读取盲文。我们推出了BrailleBench，这是一个包含259张真实世界一级和二级盲文图像的人工精心整理的基准数据集。我们评估了专有模型、开放权重模型、OCR专用模型以及工具辅助的多模态系统，发现独立的通用模型在直接转录盲文图像方面表现较差，而工具辅助系统的表现要好得多。我们设计了一个具有多样布局和图像扰动的合成盲文生成流水线。在合成数据上训练的2B参数视觉语言模型在BrailleBench的一级盲文上达到87%的精确匹配率，在二级盲文上达到61%的精确匹配率，超越了独立的视觉语言模型，并在所有评估系统中取得了最高的二级盲文精确匹配分数。我们进一步表明，在合成验证数据上的高准确率并不能保证……（原文摘要在此处截断）

    arXiv:2610.36195v1 Announce Type: cross  Abstract: Despite rapid progress in visual understanding and document transcription, modern vision-language models (VLMs) struggle with a fundamental accessibility task: reading Braille from images. We introduce BrailleBench, a manually curated benchmark of 259 real-world Grade-1 and Grade-2 Braille images. We evaluate proprietary, open-weight, OCR-specialized, and tool-assisted multimodal systems and find that standalone general-purpose models perform poorly at directly transcribing Braille images, whereas tool-assisted systems perform much better. We design a synthetic Braille generation pipeline with diverse layouts and image perturbations. A 2B VLM model trained on our synthetic data reaches 87% exact match on Grade-1 and 61% on Grade-2 BrailleBench, outperforming standalone VLMs and achieving the highest Grade-2 exact match score among the evaluated systems. We further show that high accuracy on synthetic validation data does not guarantee 
    
[^95]: 基于低秩多线性分解的大语言模型幻觉检测

    LLM Hallucination Detection via Low-rank Multilinear Decomposition

    [https://arxiv.org/abs/2610.36191](https://arxiv.org/abs/2610.36191)

    提出LOTUS框架，利用多线性低秩分解从大语言模型隐藏状态中提取紧凑表示，实现幻觉检测并减少对标注数据的依赖。

    

    大语言模型（LLM）经常生成流畅但事实错误的回复，引发了对其可靠性的担忧。近期的幻觉检测器越来越多地利用内部表示，但从高维隐藏状态中提取有信息量的信号仍然具有挑战性。现有方法通常依赖于选择性激活或基于池化的压缩，可能忽略了分布在Transformer层、特征模式和词元中的判别信息。我们提出了LOTUS（基于低秩顺序统计量Tucker分解的子空间学习），这是一个用于幻觉检测的多线性低秩分解框架，能够从LLM隐藏状态中提取紧凑表示，同时减少对标注数据的依赖。我们的方法首先使用互补的统计描述符来汇总可变长度的词元表示，这些描述符捕捉峰值激活、激活分布和跨层变化……

    arXiv:2610.36191v1 Announce Type: new  Abstract: Large language models (LLMs) frequently generate fluent yet factually incorrect responses, raising concerns about their reliability. Recent hallucination detectors increasingly exploit internal representations, but extracting informative signals from high-dimensional hidden states remains challenging. Existing approaches often rely on selected activations or pooling-based compression, potentially overlooking discriminative information distributed across Transformer layers, feature modes, and tokens. We propose LOTUS (Low-rank Order-statistic TUcker-based Subspace learning), a multilinear low-rank decomposition framework for hallucination detection that extracts compact representations from LLM hidden states while reducing dependence on labeled data. Our approach first summarizes variable-length token representations using complementary statistical descriptors that capture peak activations, activation distributions, and cross-layer change
    
[^96]: 直言不讳：国会传播中“选举诚信”一词的语境伙伴

    Frankly Speaking: The Company Election Integrity Keeps in Congressional Communication

    [https://arxiv.org/abs/2610.36187](https://arxiv.org/abs/2610.36187)

    本研究利用历时词嵌入方法分析2010至2025年间的10,420份国会电子简报，发现民主党和共和党虽然使用相同的“选举诚信”词汇，却将其置于截然不同的语义语境中，从而用同样的词讲述不同的政治故事。

    

    选举诚信是美国政治中各方共同承诺的价值，但国会中的民主党人和共和党人对它的理解却日益分化。我们分析了2010年至2025年间发送的10,420份国会电子简报，考察两党如何使用“诚信”、“安全”、“舞弊”等选举词汇。借助历时词嵌入方法——通过词语在上下文中的使用来追踪语义变化——我们测量了两党对每个词汇的使用差异程度、各自的变化幅度，以及双方是否共享最邻近的语义邻居。研究显示，国会议员无需使用不同的词汇就能讲述关于选举的不同故事，他们只需将相同的词汇置于不同的“伙伴”（上下文）之中。在几乎每一个任期和每一年，两党都会用完全不同的词汇来围绕选举诚信词汇展开论述。这种分歧很容易被忽视，因为大多数选举词汇的分化程度并不比同时期使用的其他词汇更高。

    arXiv:2610.36187v1 Announce Type: new  Abstract: Election integrity is a shared commitment in American politics, but Democrats and Republicans in Congress increasingly mean different things by it. Analyzing 10,420 congressional e-newsletters sent from 2010 to 2025, we examine how the two parties invoke election terms such as ``integrity,'' ``security,'' and ``fraud.'' Using diachronic word embeddings, which track semantic change through how words are used in context, we measure how differently each party uses each term, how much each party shifts, and whether they share nearest semantic neighbors. We show that members of Congress do not need different words to tell different stories about elections. They need only place those same words in different company. Across nearly every term and year, the two parties surround election-integrity vocabulary with entirely different words. That divergence is easy to miss, because most election vocabulary is no more divergent than other words used a
    
[^97]: Lipi-Ghor：一个大规模、带说话人分离标注、多领域的孟加拉语语音语料库及构建它的可扩展流水线

    Lipi-Ghor: A Large-Scale, Diarized, Multi-Domain Bengali Speech Corpus and a Scalable Pipeline for Building It

    [https://arxiv.org/abs/2610.36184](https://arxiv.org/abs/2610.36184)

    该论文推出了Lipi-Ghor——首个拥有885小时音频、带说话人分离标注的多领域孟加拉语语音语料库，并提供了一套可复制扩展的语料库构建流水线。

    

    孟加拉语拥有超过2.3亿使用者，但在口语语言研究中资源仍严重不足。现有的开源孟加拉语ASR语料库仅限于短篇、朗读式或众包语音，缺乏说话人分离标注和长音频。我们提出了Lipi-Ghor，这是一个大规模弱监督孟加拉语语音语料库，包含885小时的音频，来源于596个不同频道的998个YouTube视频，涵盖有声书、戏剧、宗教布道、新闻和地区方言等17个多样化领域。每个录音均以SSTT格式（说话人、语音、转录、时间戳）进行标注，该格式将自动说话人分离技术与孟加拉语YouTube字幕轨道相结合。据我们所知，Lipi-Ghor是首个将大规模覆盖与说话人分离的长音频标注相结合的孟加拉语语音语料库，而这两者在现有开源孟加拉语资源中均不存在。我们还提出了构建方法，作为一种可复制的流水线。

    arXiv:2610.36184v1 Announce Type: new  Abstract: Bengali, spoken by over 230 million people, remains severely under-resourced in spoken language research. Existing open Bengali ASR corpora are confined to short, read or crowd-sourced speech and lack speaker diarization and long-form audio.   We present Lipi-Ghor, a large-scale weakly-supervised Bengali speech corpus comprising 885 hours of audio sourced from 998 YouTube videos across 596 unique channels, spanning 17 diverse domains including audiobooks, drama, religious sermons, news, and regional dialects. Each recording is annotated in SSTT format (Speaker, Speech, Transcription, Timestamp), which combines automated diarization with Bengali YouTube caption tracks.   To our knowledge, Lipi-Ghor is the first Bengali speech corpus to pair large-scale coverage with speaker-diarized, long-form audio annotations, both of which are absent from existing open Bengali resources. We also present the construction methodology as a replicable pipe
    
[^98]: 3、7、14、17、42：指令微调在语言模型中安装并强化数字选择启发式

    3, 7, 14, 17, 42: Instruction Tuning Installs and Sharpens Number-Choice Heuristics in Language Models

    [https://arxiv.org/abs/2610.36178](https://arxiv.org/abs/2610.36178)

    该研究通过精确计算而非抽样估计的方法测量语言模型生成1到N随机数的完整概率分布，发现指令微调会安装“回避整数”并强化“回避端点”两种数字选择启发式，使分布更加不均匀。

    

    我们以零抽样误差的方式测量了语言模型在被要求给出随机数时，赋予从1到N中每个整数的概率。对于固定提示，模型定义了一个概率向量，我们通过将每个整数各计分拼写的概率求和来精确计算该向量，而非通过抽样估计，从而消除了限制先前研究的噪声。我们在两个推理栈上、29种解码配置下、五个最大至N=100的数值范围内评估了18个开放权重模型系列；其中15个系列拥有配套的基础模型与指令微调检查点，且两者均能给出可用答案。在全部15对模型中，指令微调都使数字分布变得更不均匀。它安装了一种启发式——回避整数，并强化了另一种启发式——回避端点。一旦从候选集合中移除端点，基础模型检查点并不回避整数；而微调后的检查点则会回避整数，对10的倍数几乎完全回避。基础模型检查点返回端点1和N的概率为0.30 t（原文在此处截断）。

    arXiv:2610.36178v1 Announce Type: new  Abstract: We measure, without sampling error, the probability a language model assigns to each integer from 1 to N when asked for a random number. For a fixed prompt the model defines one probability vector, which we compute by summing over each integer's scored spellings rather than estimate by sampling, removing the noise that limits earlier studies [1, 4]. We evaluate 18 open-weight model lines at five ranges up to N = 100, under 29 decoding configurations, on two inference stacks; 15 of them have matched base and instruction-tuned checkpoints that both give usable answers. Instruction tuning makes the distribution less uniform in all 15 pairs. It installs one heuristic, round-number avoidance, and sharpens another, endpoint avoidance. Base checkpoints do not avoid round numbers once the endpoints are removed from those sets; tuned checkpoints avoid them, multiples of 10 almost completely. Base checkpoints return the endpoints 1 and N at 0.30 t
    
[^99]: 不要选定赢家：知识冲突需要先验，而非规则

    Don't Pick a Winner: Knowledge Conflicts Need a Prior, Not a Rule

    [https://arxiv.org/abs/2610.36169](https://arxiv.org/abs/2610.36169)

    论文证明，检索增强语言模型中检索上下文与内部知识的冲突无法靠设计时固定的规则（如一律信上下文或信模型）最优解决，且自适应解码依赖的对称散度不包含判断依据，正确做法是结合分别评估各来源的证据与上下文可靠性的先验。

    

    检索增强语言模型从两个来源获取答案：检索到的上下文和其自身的知识，当上下文不完整或错误时，这两者就会产生分歧。现有方法通过在设计阶段就固定的规则来解决这种分歧：遵循上下文、按上下文对模型的影响程度按比例放大它，或者信任模型的记忆。每种规则在某些部署场景中都必然出错。在双来源的决策理论模型中，任何固定规则的得分与可靠上下文所占比例呈线性关系，而可获得的最佳得分是凸函数，因此每个固定规则在除恰好最优情形之外的任何情况下都会被超越；而且两个来源之间的对称散度——即自适应解码所依赖的信号——不携带任何关于哪个来源正确的信息。因此，解决冲突需要两个现有方法都未保留的输入：分别关于每个来源的证据，以及关于上下文正确频率的先验。我们通过实验验证了这些主张……

    arXiv:2610.36169v1 Announce Type: new  Abstract: A retrieval-augmented language model answers from two sources, the retrieved context and its own knowledge, and the two disagree whenever the context is incomplete or wrong. Existing methods settle the disagreement with a rule fixed at design time: follow the context, amplify it in proportion to how far it moves the model, or trust the model's memory. Each rule is confidently wrong in some deployment. In a decision-theoretic model of two sources, the score of any fixed rule is linear in the share of reliable contexts while the best attainable score is convex, so every fixed rule is beaten wherever it is not exactly optimal; and a symmetric divergence between the two sources, the signal adaptive decoding uses, carries no information about which one is right. Resolving a conflict therefore needs two inputs that no existing method keeps: evidence about each source separately, and a prior on how often the context is right. We test these clai
    
[^100]: ConformalMOS：基于保形区间与序数建模的不确定性感知MOS预测

    ConformalMOS: Uncertainty-Aware MOS Prediction with Conformal Intervals and Ordinal Modeling

    [https://arxiv.org/abs/2610.36126](https://arxiv.org/abs/2610.36126)

    提出ConformalMOS框架，结合保形预测生成具有覆盖率保证的预测区间，并采用序数感知的高斯平滑标签建模，在保持MOS预测准确度的同时实现可靠的不确定性量化。

    

    准确预测人类平均意见得分（MOS）对于评估文本转语音（TTS）和语音转换（VC）中合成语音的质量至关重要。大多数MOS预测器仅输出点估计而忽略不确定性，这限制了模型选择和部署的可靠性。我们提出ConformalMOS，这是一个通过保形预测增强MOS预测的框架，在提供点估计的同时，生成在可交换性条件下具有覆盖率保证的统计有效预测区间。在训练过程中，MOS标签以序数感知的方式进行建模，通过将独热目标转换为高斯平滑分布来保持评分的顺序性。这改善了不确定性校准，特别是在softmax置信度对分布外语音过度自信的情况下。在BVCC数据集上，ConformalMOS实现了具有竞争力的准确度（MSE = 0.08），同时提供了经过实证验证的覆盖率，提高了实际应用场景中的可靠性。

    arXiv:2610.36126v1 Announce Type: cross  Abstract: Accurate prediction of human Mean Opinion Scores (MOS) is crucial for evaluating synthetic speech quality in text-to-speech (TTS) and voice conversion (VC). Most MOS predictors output only point estimates and ignore uncertainty, limiting reliability in model selection and deployment. We propose ConformalMOS, a framework that augments MOS prediction with conformal prediction to produce statistically valid prediction intervals with guaranteed coverage under exchangeability, alongside point estimates. During training, MOS labels are modeled in an ordinal-aware manner by converting one-hot targets into Gaussian-smoothed distributions, preserving score ordering. This improves uncertainty calibration, particularly when softmax confidence is overconfident on out-of-distribution speech. On BVCC datasets, ConformalMOS achieves competitive accuracy (MSE = 0.08) while delivering empirically validated coverage, enhancing reliability for real-world
    
[^101]: 超越更新次数：一种用于评估测试时模型更新的控制图

    Beyond Update Counts: A Control Map for Evaluating Test-Time Model Updates

    [https://arxiv.org/abs/2610.36109](https://arxiv.org/abs/2610.36109)

    提出控制图评估框架，通过记录每次作答前的完整更新历史，分别评估即时学习、记忆保持、局部性与新环境泛化，从而超越仅凭更新次数比较测试时模型更新的局限。

    

    测试时更新使语言模型能够在训练之后、回答查询之前或回答过程中改变其持久状态——例如模型权重、适配器参数或记忆模块。现有评估方法通常仅通过更新次数或更新频率，以及之后立即测量的性能来比较系统。这些概括遗漏了更新了什么、在哪里更新、何时更新以及以何种顺序更新。因此，相同的总次数可能掩盖一个重要差异：一个系统可能在一次更新后作答，而另一个系统在两次更新后才作答。我们提出了控制图，这是一个评估框架，它记录每次评分答案之前所提交的更新，并分别评估行为变化、即时学习（习得）、后续回忆（保持）、无关行为的保留（局部性）以及在全新环境中的表现。我们证明了在何种条件下仅凭更新次数即可确定最终状态，并在给定假设下展示了答案前更新历史的差异如何限制可测……

    arXiv:2610.36109v1 Announce Type: cross  Abstract: Test-time updating lets a language model change persistent state after training---such as model weights, adapter parameters, or a memory module---before or while answering queries. Existing evaluations often compare systems using update count or frequency and performance measured immediately afterward. These summaries omit what is updated, where, when, and in what order. The same total can therefore hide an important difference: one system may answer after one update while another answers after two. We introduce the Control Map, an evaluation framework that records the committed updates preceding each scored answer and separately evaluates behavioral changes, immediate learning (acquisition), later recall (retention), preservation of unrelated behavior (locality), and performance in new settings. We prove when counts alone determine final state and show, under stated assumptions, how differences between pre-answer histories limit diffe
    
[^102]: FinLLM-KG：面向算法交易中金融情感分析的知识图谱增强大语言模型

    FinLLM-KG: Knowledge Graph-Enhanced LLMs for Financial Sentiment Analysis in Algorithmic Trading

    [https://arxiv.org/abs/2610.36083](https://arxiv.org/abs/2610.36083)

    该论文提出FinLLM-KG框架，通过将因果大语言模型与金融知识图谱结合，在推理时注入来自SEC 10-K文件的公司层面知识，以近乎零计算成本提升了金融情感分析的上下文理解能力，且无需更新模型权重。

    

    生成式人工智能的最新进展通过经过后训练的金融大语言模型（LLM）显著提升了金融情感分析的性能。然而，现有方法仍然局限于孤立地预测每篇文章的情感，并且以黑盒方式运行，这限制了上下文理解能力，也阻碍了其在高风险金融决策中的部署。为此，我们提出了FinLLM-KG，这是一个模型无关的框架，它将因果大语言模型与金融知识图谱相结合，以便在推理时注入来自SEC 10-K文件中具有时间依据的公司层面知识，而无需更新模型权重。在FinLLM-KG中，上下文既可以以自然语言公司简介的形式提供，也可以以少量按相关性排序的事实三元组的形式提供。这使得情感预测能够考虑更广泛的公司层面信息，而不仅仅是文章本身，且计算成本几乎为零，并适用于任何因果大语言模型。

    arXiv:2610.36083v1 Announce Type: new  Abstract: Recent advances in Generative AI have substantially improved financial sentiment analysis through post-trained financial large language models (LLMs). However, existing approaches remain confined to predicting the sentiment of each article in isolation and operate as black boxes, which limits contextual understanding and hinders deployment in high-stakes financial decision-making. To this end, we introduce FinLLM-KG, a model-agnostic framework that combines causal LLMs with a financial knowledge graph, in order to inject temporally grounded, company-level knowledge from SEC 10-K filings at inference time, without updating model weights. Within FinLLM-KG, context is supplied either as a natural-language company profile or as a small set of relevance-ranked factual triples. This enables sentiment predictions to account for broader firm-level information rather than the article alone, at almost zero computational cost and for any causal LLM
    
[^103]: 提示究竟有多重要？金融推理中后见之明自蒸馏的适用条件

    How Much Does the Hint Matter? Scope Conditions for Hindsight Self-Distillation in Financial Reasoning

    [https://arxiv.org/abs/2610.36082](https://arxiv.org/abs/2610.36082)

    提出Fin-SD方法，通过让学生模型自己发现并批评首次错误生成提示，再以错误前缀加提示的条件进行自蒸馏，从而在无需外部教师或完整参考解答的情况下提升小模型的多步金融推理能力。

    

    在数据驻留限制下运营的金融机构只能使用体积小到足以本地部署的开源权重模型，而这些模型在多步定量推理方面落后于前沿系统。在线策略蒸馏（OPD）通过在学生模型自身的采样轨迹上进行监督来缩小这一差距，但传统的OPD需要一个强大的外部教师模型，而在线策略自蒸馏（OPSD）则以完整的参考解答作为条件。我们探究了这种信号中究竟有多少是必要的。我们提出了Fin-SD，一种面向多步推理的本地化后见之明自蒸馏方法。当学生模型无法解决某个问题时，学生模型自己识别出第一个错误的推理步骤，并对该步骤写出简短的批评意见。我们将学生模型的轨迹在错误发生之前截断（旨在获得正确的前缀），让学生模型的一个冻结副本以相同前缀加上提示为条件，然后在短（原文在此处截断）

    arXiv:2610.36082v1 Announce Type: cross  Abstract: Financial institutions operating under data-residency constraints are restricted to open-weight models small enough to locally host, which trail frontier systems on multi-step quantitative reasoning. On-policy distillation narrows such gaps by supervising a student on its own rollouts, but traditional OPD requires a strong external teacher, and on-policy self-distillation (OPSD) conditions on a full reference solution. We ask how much of that signal is necessary. We present Fin-SD, a localized hindsight self-distillation recipe for multi-step reasoning. Given a problem the student fails, the student itself identifies the first incorrect reasoning step and writes a short critique of that step. We truncate the student's trajectory immediately before the error (aiming to obtain a correct prefix), condition a frozen copy of the student on the same prefix plus the hint, and distill that hint-conditioned teacher into the student over a short
    
[^104]: OmniBrain-0：一个面向具身智能的统一智能体模型

    OmniBrain-0: A Unified Agentic Model for Embodied Intelligence

    [https://arxiv.org/abs/2610.36013](https://arxiv.org/abs/2610.36013)

    OmniBrain-0是一个轻量级统一具身智能体，通过具身预训练、领域专用智能体蒸馏和多教师在线策略蒸馏（MOPD）的渐进式训练框架，利用共享的高层决策模型协调操作与导航，实现任务理解、规划、进度跟踪和失败恢复。

    

    通用具身智能要求机器人在长时程任务中协调多种技能、跟踪任务进度并从失败中恢复。仅靠专家策略无法保证任务级别的可靠性，而强大的多模态智能体则依赖昂贵的推理和外部经验，这是紧凑模型难以复现的。我们提出了OmniBrain-0，一个轻量级的统一具身智能体，它通过共享的高层决策模型协调操作与导航，整合了任务理解、规划、进度跟踪和基于反馈的失败恢复。我们的渐进式训练框架结合了具身预训练、领域专用智能体蒸馏以及多教师在线策略蒸馏（MOPD）。该框架首先将强大教师的交互经验蒸馏到操作和导航专家模型中，随后将它们的决策能力整合到OmniBrain-0中。因此，统一化发生在学习到的决策过程之中。

    arXiv:2610.36013v1 Announce Type: new  Abstract: General-purpose embodied intelligence requires robots to coordinate diverse skills over long horizons, track task progress, and recover from failures. Specialist policies alone do not ensure task-level reliability, while powerful multimodal agents rely on costly reasoning and external experience that compact models struggle to reproduce. We introduce OmniBrain-0, a lightweight unified embodied agent that coordinates manipulation and navigation through a shared high-level decision model, integrating task understanding, planning, progress tracking, and feedback-driven recovery. Our progressive training framework combines embodied pretraining, domain-specialized agent distillation, and multi-teacher on-policy distillation (MOPD). It distills powerful teachers' interaction experience into manipulation and navigation experts, then consolidates their decision capabilities into OmniBrain-0. Unification therefore occurs within the learned decisi
    
[^105]: MedJev：评估医学领域中的零样本“系统一”决策

    MedJev: Evaluating Zero-Shot System-One Decision Making in Medicine

    [https://arxiv.org/abs/2610.36006](https://arxiv.org/abs/2610.36006)

    MedJev是一个可在笔记本电脑上本地运行的开源“系统一”医学决策模型，在多项医学知识和诊断基准上接近GPT-4的性能，同时保持良好的置信度校准，为快速高效的临床决策支持提供了新方案。

    

    系统一决策模型通过单次前向传播即可回答结构化问题而无需生成文本，为临床决策支持提供了快速且计算高效的方案。我们在涵盖医学知识、诊断以及基于MIMIC-IV和MIMIC-IV ED数据集的患者层面预测的十个医学基准上，评估了MedJev——一个可在笔记本电脑上本地运行的开源系统一模型。我们将MedJev与零样本LLM基线、已发表的GPT结果以及监督模型进行比较。MedJev在MedQA上达到72.5%的准确率，在USMLE上达到75.4%，在NEJM上达到80.1%，接近已发表的GPT-4性能，同时保持了良好的置信度校准（ECE为0.028至0.063）。在患者层面任务上，MedJev在院内死亡率预测中达到0.926的AUROC，超过了XGBoost（0.902），但在急诊科分诊入院等操作性结局上表现不及监督模型（0.745 vs. 0.809）。

    arXiv:2610.36006v1 Announce Type: cross  Abstract: System One decision models such as Jev answer structured questions in a single forward pass without generating text, offering fast and computationally efficient clinical decision support. We evaluate MedJev, an open System One model that runs locally on a laptop, across ten medical benchmarks spanning medical knowledge, diagnosis, and patient level prediction using MIMIC IV and MIMIC IV ED. We compare MedJev against zero shot LLM baselines, published GPT results, and supervised models. MedJev achieves accuracies of 72.5 percent on MedQA, 75.4 percent on USMLE, and 80.1 percent on NEJM, approaching published GPT 4 performance while maintaining well calibrated confidence estimates (ECE 0.028 to 0.063). On patient level tasks, MedJev achieves an AUROC of 0.926 for In Hospital Mortality, exceeding XGBoost (0.902), but underperforms supervised models on operational outcomes such as ED Triage Admission (0.745 vs. 0.809). Patient level predic
    
[^106]: SCOPE：信源可信度与媒体机构画像引擎

    SCOPE: Source Credibility and Outlet Profiling Engine

    [https://arxiv.org/abs/2610.35929](https://arxiv.org/abs/2610.35929)

    本文提出SCOPE多智能体系统，通过协调文章获取、内容分析、证据搜索等专门智能体，基于证据对新闻媒体的偏见、事实性和可信度进行透明评估，在200家媒体基准上超越了多种基线方法。

    

    我们提出了SCOPE，一个用于对新闻媒体机构在偏见、事实性和可信度方面进行画像的多智能体系统。给定一个媒体机构的URL，SCOPE会协调多个专门的智能体来完成文章获取、内容分析、外部证据搜索、评分和报告生成等任务。该系统遵循MBFC（媒体偏见事实核查）风格的方法论，同时通过结构化的演示界面使中间证据和智能体输出可供查验。在一个包含200家媒体机构的基准测试中，SCOPE在大多数指标上优于仅使用大语言模型、仅使用文章、仅使用搜索以及媒体背景核查等基线方法，实现了最低的偏见和事实性平均绝对误差（MAE）、完美的失败事实核查检测能力以及最低的事实错误率。这些结果表明，分解式的、基于证据的智能体工作流能够支持更透明、更可靠的媒体画像。我们的代码可在 https://github.com/mbaisbay/SCOPE 获取，在线演示见 https://scope-demo.streamli

    arXiv:2610.35929v1 Announce Type: cross  Abstract: We present SCOPE, a multi-agent system for profiling news media outlets with respect to bias, factuality, and credibility. Given an outlet URL, SCOPE coordinates specialized agents for article acquisition, content analysis, external evidence search, scoring, and report generation. The system follows an MBFC-style methodology while making intermediate evidence and agent outputs inspectable through a structured demonstration interface. On a benchmark of 200 media outlets, SCOPE outperforms LLM-only, article-only, search-only, and media-background-check baselines on most metrics, achieving the lowest bias and factuality MAE, perfect failed-fact-check detection, and the lowest factual error rate. These results suggest that decomposed, evidence-grounded agent workflows can support more transparent and reliable media profiling. Our code is available at https://github.com/mbaisbay/SCOPE, and a live demonstration at https://scope-demo.streamli
    
[^107]: ReCoPII：一种基于关系构建情境化PII脱敏数据集的框架

    ReCoPII: A Relation-Grounded Framework for Building Contextual PII Masking Datasets

    [https://arxiv.org/abs/2610.35925](https://arxiv.org/abs/2610.35925)

    ReCoPII框架通过规划本体约束的实体关系并据此推导MASK/KEEP标签，无需单一数据主体假设即可自动构建情境化PII脱敏数据集，所生成的韩文语料ReCoPII-Ko的KEEP标签获得88.42%的标注者认可率，远超SPY方法的26.27%。

    

    训练针对个人身份信息（PII）的情境化脱敏系统，需要能够区分同一实体类型中“可脱敏”与“应保留”提及的样本，但隐私限制使得真实数据难以获取。基于作者的增强方法（SPY）将与作者无关的提及标注为非PII，这一做法假定了单一数据主体，从而忽略了其他主体。我们提出了ReCoPII（基于关系的情境化PII脱敏），一个无需上述前提即可构建此类数据的框架：它首先规划受本体约束的实体关系，然后撰写带有占位符的插入句，再通过代码重建标注，用独立的LLM检查结果，并根据实体关系推导出MASK/KEEP标签。在来自六个领域的韩文文档上，该框架生成了ReCoPII-Ko语料库，其中包含关系、隐私和MASK/KEEP三类标注，我们对其进行了验证。在300对样本中，四名标注者对其抽样KEEP标签的认可率达88.42%，而SPY风格方法仅为26.27%。

    arXiv:2610.35925v1 Announce Type: cross  Abstract: Training contextual masking systems for personally identifiable information (PII) requires examples distinguishing maskable from preservable mentions of one entity type, but privacy restrictions limit access to real data. Author-based augmentation (SPY) labels mentions unrelated to the author as non-PII, presuming a single data subject and overlooking others. We propose ReCoPII (Relation-grounded Contextual PII masking), a framework that builds such data without this premise: it plans ontology-constrained entity relations, writes insertion sentences with placeholders, reconstructs annotations with code, checks the result with a separate LLM, and derives MASK/KEEP labels from entity relations. On Korean documents from six domains, it yields ReCoPII-Ko, a corpus with relation, privacy, and MASK/KEEP annotations, which we validate. Across 300 pairs, four annotators agree with 88.42% of its sampled KEEP labels, against 26.27% for SPY-style
    
[^108]: 面相学的重现：视觉语言模型如何通过图像变换编码敏感属性

    Physiognomy Revisited: How Vision-Language Models Encode Sensitive Attributes Through Image Transformation

    [https://arxiv.org/abs/2610.35910](https://arxiv.org/abs/2610.35910)

    本文揭示了视觉语言模型在图像变换任务中会遵从按种族和性取向改变人脸的请求，生成统计上可区分且强化刻板印象的图像，并且这些表征偏差还会在下游文本推理任务中重现。

    

    新兴的生成式人工智能技术虽然在众多任务和领域中具有显著益处，但也可能存在无法预见的漏洞和难以预测的失败模式。这些失败模式可能对弱势群体造成不成比例的影响。本文研究了两个知名的视觉语言模型（VLM），考察这些模型在生成图像时如何编码受保护的身份特征。我们表明，当要求VLM将人脸图像转换为反映种族身份（如西班牙裔、黑人、白人或亚裔）和性取向（如同性恋或异性恋）时，VLM往往会遵从此类请求。我们证明，这种遵从（和拒绝）是非确定性的、对群体敏感的，且生成的图像在统计上是可区分的，往往会强化根深蒂固的刻板印象。我们进一步表明，图像生成过程中引入的表征偏差可能会在文本推理和人物刻画任务中再次出现。

    arXiv:2610.35910v1 Announce Type: cross  Abstract: Emergent generative AI technologies, while remarkably beneficial across several tasks and domains, can suffer from unforeseen vulnerabilities and difficult-to-anticipate failure modes. Such failure modes can disparately affect vulnerable communities. This paper investigates two well-known vision language models (VLMs) and inspects how these models encode protected identities while generating images. We show that when VLMs are requested to transform human face images to reflect race identities (e.g., Hispanic, Black, White, or Asian) and sexual orientation (e.g., gay or straight), VLMs often comply with such requests. We demonstrate that compliance (and refusal) is non-deterministic, group-sensitive, and generated images are statistically distinguishable, often accentuating deep-seated stereotypes. We further show that representation biases introduced during image generation can re-emerge in textual reasoning and characterization tasks.
    
[^109]: 理解Jev式系统一模型：结构能否减少选择偏差？

    Understanding Jev-Style System One Models: Does Structure Reduce Selection Bias?

    [https://arxiv.org/abs/2610.35826](https://arxiv.org/abs/2610.35826)

    本文通过在保持问题内容不变的前提下打乱选项顺序的简单测试，系统比较了两个Jev版本、十二个Jev复现模型和六个生成式LLM在多选项任务中的选择偏差，探究Jev式决策结构能否减少选择偏差。

    

    arXiv:2610.35826v1 公告类型：new 摘要：尽管有要求忽略呈现顺序的指令或引出推理的提示，大型语言模型（LLM）中的选择偏差仍可能持续存在。近来对Jev式系统一行为的复现工作产生了一个不断壮大的模型家族，这些模型直接在给定候选选项上预测概率。然而，在多选项任务中，针对Jev各版本、Jev复现模型以及生成式LLM之间选择偏差的系统性比较仍然有限。这就留下了一个核心问题：Jev式决策结构能否减少选择偏差？我们通过一个简单的测试来回答这个问题：在保持问题和候选内容不变的情况下打乱选项顺序。我们在两个Jev版本、十二个Jev复现模型和六个生成式LLM上，追踪个体答案在八种呈现方式下的表现，并将其与固定顺序重复进行比较。在我们从两个基准收集的1,700个问题的主要数据集上，Jev的准确率变化不到两个百分点……

    arXiv:2610.35826v1 Announce Type: new  Abstract: Selection bias in large language models (LLMs) can persist despite instructions to ignore presentation order or prompts that elicit reasoning. Recent efforts to reproduce Jev-style System One behavior have produced a growing family of models that predict probabilities directly over supplied candidates. Yet systematic comparisons of selection bias across Jev versions, Jev reproductions, and generative LLMs remain limited in multi-option tasks. This leaves a central question: does Jev-style decision structure reduce selection bias? We address this question with a simple test that shuffles options while keeping question and candidate content fixed. Across two Jev versions, twelve Jev reproductions, and six generative LLMs, we trace individual answers through eight presentations and compare them with fixed-order repetition. On our primary collection of 1,700 questions from two benchmarks, Jev's accuracy varies by less than two percentage poi
    
[^110]: 多层Transformer的定量泛化界：少数长字符串可以胜过大量短字符串

    Quantitative Generalization Bounds for Multi-Layer Transformers: A Few Long Strings can Beat Many Short Ones

    [https://arxiv.org/abs/2610.35715](https://arxiv.org/abs/2610.35715)

    本文首次对固定精度多层Transformer的分布内统计泛化与长度泛化进行了联合的定量理论分析，并揭示出一个反直觉的结论：用少数长字符串训练可以胜过用大量短字符串训练。

    

    基于Transformer的语言模型展现出惊人的能力，能够泛化到比训练中所见更长、更难的问题实例。然而，对这一现象的理论理解仍然滞后。与大多数机器学习模型不同，Transformer的泛化能力本质上需要从两个维度进行考察。一方面，必须研究分布内统计泛化，即模型将从训练样本学到的函数拓展到来自同一分布的新样本的能力；另一方面，鉴于语言数据的序列特性，还必须考虑长度泛化，即在给定长度输入上训练的模型能否外推到更长的输入。这两个概念以往大多被分开研究，它们之间如何相互作用仍是一个悬而未决的问题。在这项工作中，我们对固定精度的Transformer给出了两个维度的联合分析。我们给出了一般……

    arXiv:2610.35715v1 Announce Type: cross  Abstract: Transformer-based language models exhibit an astonishing capacity to generalize to longer and more difficult problem instances than those seen in training. However, theoretical understanding of this phenomenon still lags behind. Unlike most machine learning models, the generalisation abilities of transformers inherently require considering two axes. On one hand, one must examine in-distribution statistical generalisation, which concerns the ability of a model to broaden a function from training samples to novel samples of the same distribution. On the other hand, given the sequential nature of language data, one must also consider length generalisation, which reflects whether a model trained on inputs of a given length can extrapolate to longer ones.These two notions have mostly been studied separately, leaving open how they interact. In this work, we give a joint treatment of both axes for fixed-precision transformers. We give general
    
[^111]: 语言模型理解伊博文化吗？超越多选题

    Do Language Models Understand Igbo Culture? Beyond Multiple Choice

    [https://arxiv.org/abs/2610.35168](https://arxiv.org/abs/2610.35168)

    本文提出了基于伊博本土文化构建并由母语者验证的CAB-Igbo基准测试，通过多选题与人类评估相结合的方式，揭示了语言模型的流畅性与文化知识之间存在严重脱节，且多数模型在英文中就已缺失认知忠实度和表征丰富度。

    

    语言模型对任何社区的价值取决于模型是否真正理解它所意图服务人群的文化。我们以伊博语为案例进行研究。现有的基准测试考察的是翻译过来的西方思想，缺乏深度。为解决这一问题，我们提出了CAB-Igbo，这是一个基于本土文化来源构建、并由母语使用者在六个领域进行验证的基准测试。我们在不同条件下评估了多种模型，并纳入了地区性人格设定。性能通过多选题准确率和人类评估进行衡量，评估维度包括认知忠实度、表征丰富度、语言质量和语用能力。结果揭示了语言流畅性与文化知识之间存在显著的脱节。通过统计检验，我们发现对于大多数模型而言，认知忠实度和表征丰富度的严重缺失在英文中就已经存在，并且在切换到伊博语时并未显著恶化。

    arXiv:2610.35168v1 Announce Type: new  Abstract: The value of language models for any community depends on whether a model actually understands the culture of the people it intends to serve. We use the Igbo language as a case study. Existing benchmarks test translated Western ideas, lacking depth. To address this, we introduce CAB-Igbo, a benchmark developed from indigenous cultural sources and verified by native speakers across six domains. We evaluated diverse models under varying conditions, incorporating regional personas. Performance was assessed using multiple-choice accuracy and human evaluations across Epistemic Fidelity, Representational Richness, Linguistic Quality, and Pragmatic Proficiency. Results reveal a stark dissociation between language fluency and cultural knowledge. Through statistical testing, we show that for most models, the severe lack of Epistemic Fidelity and Representational Richness is already present in English and does not significantly worsen when switchi
    
[^112]: 先问再答：面向上下文学习的问题驱动证据推理架构（QERA）

    Ask Before You Answer: A Question-Driven Evidence Reasoning Architecture (QERA) for In-Context Learning

    [https://arxiv.org/abs/2610.35083](https://arxiv.org/abs/2610.35083)

    该论文提出QERA架构，通过先将用户请求转化为需求清单并筛选出隐含信息需求，再从完整对话中作答并用代码验证引文，从而解决大语言模型在真实生活上下文学习任务中因请求隐含大量未明说标准而表现不佳的问题。

    

    AI助手越来越多地被要求基于用户自己的网络痕迹进行推理，包括群聊、社区帖子、活动日志和修订历史。在CL-bench Life这一真实生活上下文学习基准上，前沿大语言模型解决的任务不足五分之一。我们发现请求本身并非任务规范：该基准中超过80%的评分标准与最终请求的内容词重合度低于60%，例如两条对话之前的规则集，或所有满足用户某项标准的记录，而直接作答的大语言模型在这些隐含标准上仅通过67%，相比之下在其他标准上通过率为81%。我们提出QERA，一种先问再答的问题驱动证据推理架构。QERA首先将请求转化为需求清单和类型化的信息需求，并通过代码进行过滤以确保忠实于用户的原话；然后基于整个对话进行回答，使用经代码验证的引文再次提出相同的问题，并应用……（摘要内容在此处截断）

    arXiv:2610.35083v1 Announce Type: new  Abstract: AI assistants are increasingly asked to reason over a user's own web traces: group chats, community threads, activity logs and revision histories. On CL-bench Life, a benchmark of such real-life context learning, frontier LLMs solve fewer than one task in five. We find that the request is not the specification: over 80% of the benchmark's rubrics share less than 60% of their content words with the final request, as for a rule set two turns earlier or every record that meets the user's criterion, and LLMs answering directly pass only 67% of these implicit rubrics against 81% of the others. We propose QERA, a Question-driven Evidence Reasoning Architecture that asks before it answers. QERA first turns the request into a requirement checklist and typed information needs, filtered by code for faithfulness to the user's words; it then answers from the whole conversation, asks the same questions again with quotes verified by code, and applies 
    
[^113]: 诊断与缓解全双工语音模型中的长会话性能退化

    Diagnosing and Mitigating Long-Session Degradation in Full-Duplex Speech Models

    [https://arxiv.org/abs/2610.34991](https://arxiv.org/abs/2610.34991)

    本文提出一个闭环评估框架系统性诊断全双工语音模型在长会话中的性能退化，并提出基于自蒸馏的轻量级LoRA适配训练方法来有效缓解该退化。

    

    端到端全双工语音对话模型能够实现自然的实时语音交互，支持打断、附和反馈以及语音重叠。先前的评估工作已经考察了多样化的交互能力，并识别出一些跨对话轮次的性能退化模式。然而，模型行为在长会话中如何演变仍缺乏充分的理解。在本工作中，我们通过系统性地刻画长会话性能退化并开发高效的缓解方法来填补这一空白。我们引入了一个面向长会话稳定性的闭环评估框架，并将其应用于多个公开可用的全双工语音模型，揭示了模型在响应性、语音内容质量和声音一致性方面随时间的退化。为了缓解这种退化，我们提出了一种基于自蒸馏的训练框架，用于轻量级LoRA适配，仅需少量额外的数据即可提升长会话稳定性。

    arXiv:2610.34991v1 Announce Type: new  Abstract: End-to-end full-duplex spoken dialogue models enable natural, real-time spoken interactions with interruptions, backchannels, and overlapping speech. Prior evaluations have assessed diverse interaction capabilities and identified some degradation patterns across conversational turns. However, how model behavior evolves over extended sessions remains insufficiently understood. In this work, we address this gap by systematically characterizing long-session degradation and developing efficient methods to mitigate it. We introduce a closed-loop evaluation framework for long-session stability and apply it to several publicly available full-duplex speech models, revealing degradation over time in responsiveness, spoken content quality, and voice consistency. To mitigate this degradation, we propose a self-distillation-based training framework for lightweight LoRA adaptation that improves long-session stability with a small amount of additional
    
[^114]: FATE-5：面向连续大五人格控制的特质表达忠实对齐

    FATE-5: Faithful Alignment of Trait Expression for Continuous Big Five Personality Control

    [https://arxiv.org/abs/2610.34889](https://arxiv.org/abs/2610.34889)

    FATE-5提出两阶段框架，将大五人格编码与自然度对齐分离，实现了对大语言模型人格的连续控制，在保持强特质表达的同时显著提升了对话自然度。

    

    在大型语言模型中控制人格，对于构建能够根据不同用户和交互场景进行定制化的对话智能体而言十分重要。然而，现有人格控制方法通常面临强特质表达与自然对话生成之间的权衡。我们提出了FATE-5（大五人格特质表达忠实对齐），这是一个将人格编码与自然度对齐相分离的两阶段框架。FATE-5首先利用对话监督和辅助问卷目标，学习大五人格画像的连续软提示表示；随后通过基于对话质量和人格一致性的偏好对齐来提升对话自然度。在三个语言模型骨干网络上的实验表明，FATE-5在保持强特质表达的同时，实现了比其他强特质方法显著更高的对话自然度。在保留问卷测试中，它实现了单调性……

    arXiv:2610.34889v1 Announce Type: new  Abstract: Controlling personality in large language models is important for building conversational agents that can be customized to different users and interaction settings. However, existing personality control methods often face a tradeoff between strong trait expression and natural dialogue generation. We introduce FATE-5, Faithful Alignment of Trait Expression for the Big Five, a two-stage framework that separates personality encoding from naturalness alignment. FATE-5 first learns continuous soft prompt representations of Big Five profiles using dialogue supervision and an auxiliary survey objective. It then improves dialogue naturalness through preference alignment based on both dialogue quality and personality agreement. Across three language model backbones, FATE-5 maintains strong trait expression while achieving substantially higher dialogue naturalness than other trait-strong methods. On held-out questionnaires, it achieves monotonicit
    
[^115]: 音频-语言模型中可控说话风格描述的基准测试

    A Benchmark for Controllable Speaking-Style Captioning in Audio-Language Models

    [https://arxiv.org/abs/2610.33036](https://arxiv.org/abs/2610.33036)

    提出StyleInstructCaps——首个可控说话风格描述的标准化基准，将说话风格描述形式化为指令跟随的音频-语言任务，通过明确指令指定需描述的风格维度（如说话人特质、情感、表达风格）。

    

    说话风格描述（SSC）旨在生成关于语音如何被表达的自然语言描述，捕捉诸如音色、韵律和表现力等副语言属性。许多下游应用，包括对话式AI智能体、可控语音生成和大规模音频标注，都需要针对特定维度的风格描述来刻画语音的特定方面（例如说话人特质、情感或表达风格），而不是单一的、不加区分的描述。然而，现有工作缺乏对可控说话风格描述的统一形式化定义。我们通过将SSC形式化为一个遵循指令的音频-语言任务来填补这一空白，其中明确的指令指定了需要描述的说话风格维度。我们推出了StyleInstructCaps，这是首个面向音频-语言模型中可控说话风格描述的标准化基准。StyleInstructCaps提供了一个数据集……

    arXiv:2610.33036v1 Announce Type: cross  Abstract: Speaking-style captioning (SSC) aims to generate natural language descriptions of how speech is delivered, capturing paralinguistic attributes such as vocal timbre, prosody, and expressivity. Many downstream applications, including conversational AI agents, controllable speech generation, and large-scale audio annotation, require dimension-specific style captions that describe targeted aspects of speech (e.g., speaker traits, emotion, or delivery style), rather than a single undifferentiated description. However, existing work lacks a unified formulation for controllable speaking-style captioning. We address this gap by formulating SSC as an instruction-following audio-language task, where explicit instructions specify the speaking-style dimensions to be described. We introduce StyleInstructCaps, the first standardized benchmark for controllable speaking-style captioning in audio-language models. StyleInstructCaps provides a dataset an
    
[^116]: 你有多信任一只狗？量化推理语言模型中的提示遵循行为

    How Much Do You Trust A Dog? Quantifying Hint-Following in Reasoning Language Models

    [https://arxiv.org/abs/2610.32055](https://arxiv.org/abs/2610.32055)

    该研究发现大语言模型对提示来源（如教授、朋友甚至狗）的答案偏移遵循可预测的指数倾斜结构，且偏移强度与来源可靠性基本无关，揭示了模型谄媚行为未被来源可靠性校准的问题。

    

    大型语言模型（LLM）常常表现出谄媚行为，即使在客观的事实和推理任务上，也会为了迎合用户的建议而改变自己的答案。然而，这种影响的定量形式——以及它是否与信息来源的可靠性相校准——仍然知之甚少。在本工作中，我们通过MMLU问题上的提示诱导答案偏移来研究这一行为，评估形如“[来源]认为答案是(X)”的提示。我们在三个开源模型系列（Qwen、Gemma和SmolLM）上系统地改变提示所归属的来源（教授、朋友或狗）。我们表明，提示引起的变化遵循一个可预测的结构，该结构可以用指数倾斜很好地刻画：提示在被提示答案的对数几率上引起近似恒定的均值偏移，且问题与目标的易感性在不同来源之间很大程度上是共享的。为了探索这种模式如何产生，我们引入了一个形式化的……

    arXiv:2610.32055v1 Announce Type: new  Abstract: Large language models (LLMs) often exhibit sycophancy, altering their answers to align with user suggestions even on objective factual and reasoning tasks. Yet the quantitative form of this influence--- and whether it is calibrated to source reliability--- remains poorly understood. In this work, we investigate this behavior through hint-induced answer shifts on MMLU questions, evaluating prompts of the form ``[Source] thinks the answer is (X)''. We systematically vary the source to which the hint is attributed (a professor, a friend, or a dog) across three open-source model families, Qwen, Gemma, and SmolLM. We show that the change induced by the hint follows a predictable structure well-characterized by an exponential tilt: the hint induces an approximately constant mean shift in the hinted answer's log-odds, with question--target susceptibility largely shared across sources. To explore how this pattern can arise, we introduce a formal
    
[^117]: 知道何时停止：编码智能体中的证据-行动差距

    Knowing When to Stop: The Evidence-Action Gap in Coding Agents

    [https://arxiv.org/abs/2610.31895](https://arxiv.org/abs/2610.31895)

    该论文揭示了编码智能体的“证据-行动差距”现象——即智能体即使已收集到表明任务无法独立完成的充分证据仍过度坚持或虚报成功，并提出ImpasseBench基准通过在可解决任务中注入环境阻碍并标记证据充分步骤来量化这一差距。

    

    编码智能体被训练来完成任务，即使环境使得任务根本无法完成时，它们仍会不断尝试。表明只有外部帮助才能恢复进展的证据通常在几步之内就会出现，但智能体很少据此采取行动：它们过度坚持，尝试不必要的变通方法，并声称取得记录无法支持的成功——我们将这一现象称为“证据-行动差距”。为了衡量这一差距，我们提出了ImpasseBench，它在原本可解决的修复任务中注入十二种真实的环境阻碍因素，并在每条轨迹中标记出所收集证据变得充分的步骤。相对于该步骤对智能体的行动进行评分，可以区分适当的努力与过度坚持以及过早求助。在八个前沿模型和19,371条轨迹上，我们发现几乎所有需要外部帮助的运行都收集到了充分的证据，但仍有三分之一耗尽了预算，而且我们检测到的所有不安全请求都发生在这一时点之后。

    arXiv:2610.31895v1 Announce Type: cross  Abstract: Coding agents are trained to complete tasks, and they keep trying even when the environment makes completion impossible. The evidence that only outside help can restore progress usually appears within a few steps, yet agents rarely act on it: they over-persist, attempt undesired workarounds, and claim success the record does not support -- a phenomenon we call evidence-action gap. To measure it, we introduce ImpasseBench, which injects twelve realistic environment blockers into otherwise solvable repair tasks and marks, in each trajectory, the step at which the collected evidence becomes sufficient. Scoring actions relative to that step separates appropriate effort from over-persistence and premature handoff. Across eight frontier models and 19,371 trajectories, we find that nearly every run that needs outside help collects sufficient evidence, yet a third still exhaust their budget, and every unsafe request we detect comes after that 
    
[^118]: CulturalRiddles：一个多文化谜语基准测试

    CulturalRiddles: A Multicultural Benchmark of Riddles

    [https://arxiv.org/abs/2610.29968](https://arxiv.org/abs/2610.29968)

    CulturalRiddles是一个涵盖62种文化、完全由人类创作的谜语基准测试，用于评估大语言模型的跨文化理解与推理能力，结果表明即使是前沿模型在某些社区中的准确率也最高仅达22%。

    

    随着大语言模型不断扩大其语言覆盖范围，评估它们对多元文化的理解和表达能力变得越来越重要。然而，许多基准测试仍然以英语为中心，文化覆盖有限，并且更侧重于事实性知识而非其在现实文化语境中的应用。我们提出了CulturalRiddles，这是一个完全由人类通过社区参与式努力创作的原创、植根于文化的谜语基准，涵盖62种文化（45个国家、51种语言）。这些谜语深深植根于各自的文化，需要溯因推理和组合推理能力，以及详细的事实和社会知识、语言技能以及对文化特定关联的理解。我们使用自动评估和大规模人工评估对28个LLM配置进行了基准测试。即使是前沿模型，在某些社区中的准确率也最高仅达到22%。

    arXiv:2610.29968v1 Announce Type: new  Abstract: As LLMs expand their language coverage, evaluating how well they understand and represent diverse cultures becomes increasingly important. However, many benchmarks remain English-centric, with limited cultural coverage and a focus on factual knowledge over its application in real-world cultural contexts. We propose CulturalRiddles, a benchmark of original, culturally grounded riddles created entirely by humans through a participatory community effort, spanning 62 cultures (45 countries, 51 languages). These riddles are deeply rooted in their respective cultures and require abductive and compositional reasoning, alongside detailed factual and social knowledge, language skills, and an understanding of culturally specific associations. We benchmark 28 LLM configurations using both automatic and large-scale human evaluation. Even frontier models achieve at most 22% accuracy in some communities. We publicly release all artifacts alongside an 
    
[^119]: 修辞失真存在于何处？一项针对网红内容的多平台、多语言测量研究

    Where Does Rhetorical Distortion Live? A Multi-Platform, Multi-Language Measurement of Influencer Content

    [https://arxiv.org/abs/2610.26412](https://arxiv.org/abs/2610.26412)

    本文提出一套在六个平台、两种语言上统一应用的修辞失真测量框架，通过双语词汇规则、负面过滤器与大语言模型验证构成的三层级流水线，检测五种修辞操纵维度，并将创作者帖子与回复分开分类，以定位此类框架化表述的真正来源。

    

    arXiv:2610.26412v2 公告类型：replace-cross 摘要：网红内容中的许多操纵并非虚假信息，而是不成比例的表述——它们将准确的事实包装成具有历史性、紧迫性或灾难性的内容。现有方法要么验证事实性声明，要么在单一平台、单一语言的文本中检测标题党与宣传内容；然而，这些方法忽视了此类框架化表述的来源，因为它们将帖子及其下方的回复视为一个整体。为填补这一空白，我们使用同一套测量工具，在六个平台和两种语言上进行一致的修辞失真测量，并将创作者帖子与其回复分开分类。我们的三层级流水线检测五个基于文献的维度（重要性夸大、焦虑制造、新颖性声明、情绪化语言和时间失真）：双语词汇规则提出候选样本，负面过滤器排除已知的误报，大语言模型对候选样本进行验证或重新标注。这种设计使标签保持透明且可比较（原文摘要在此处截断）。

    arXiv:2610.26412v2 Announce Type: replace-cross  Abstract: Much of the manipulation in influencer content is not false but disproportionate, framing accurate facts as historic, urgent, or catastrophic. Existing approaches either verify factual claims or detect clickbait and propaganda in single-platform, single-language text; however, they overlook where such framing originates, because they treat a post and the replies beneath it as one unit. To address this gap, we measure rhetorical distortion with one instrument applied identically across six platforms and two languages, classifying creator posts and their replies separately. Our three-tier pipeline detects five literature-grounded dimensions (Significance Inflation, Anxiety Manufacturing, Novelty Claims, Loaded Language, and Temporal Distortion): bilingual lexical rules propose candidates, negative filters cancel known false positives, and an LLM verifies or relabels the candidates. This design keeps labels transparent and compara
    
[^120]: 面向在线策略学习的桥接自蒸馏

    Bridged Self-Distillation for On-Policy Learning

    [https://arxiv.org/abs/2610.26397](https://arxiv.org/abs/2610.26397)

    提出桥接自蒸馏框架，通过让学生模型以教师推理轨迹为条件的副本充当自教师、在学生自身的词元空间中对学生的采样序列进行评分，从而统一解决了在线策略蒸馏中教师解答难以模仿与词元级概率不兼容或不可获取的两大难题。

    

    在线策略蒸馏可以将强大教师模型的推理能力迁移到更小的学生模型中。然而，从更强的教师模型进行蒸馏面临两个不匹配问题：其解答可能难以被较小的学生模型模仿，且其词元级概率可能与学生模型不兼容，甚至无法获取。现有方法通过不同形式的监督来处理这些情况。为此，我们提出了桥接自蒸馏，这是一个能够从任何提供推理轨迹的教师模型进行蒸馏的统一框架。被蒸馏的模型被称为外部教师，它首先提供一个推理轨迹作为特权上下文，该上下文可以以离线方式生成。学生模型在无此上下文的情况下生成采样序列，而一个以该上下文为条件的学生模型副本作为自教师，在学生自身的词元空间中对相同的采样序列进行评分。BSD利用由此产生的词元级（概率信号进行训练……摘要在此处被截断）

    arXiv:2610.26397v2 Announce Type: replace  Abstract: On-policy distillation can transfer the reasoning capabilities of a strong teacher to a smaller student. However, distillation from a stronger teacher model faces two mismatches: its solutions may be difficult for a smaller student to imitate, and its token-level probabilities may be incompatible with, or even unavailable to, the student. Existing methods handle these settings through different forms of supervision. To this end, we propose Bridged Self-Distillation (BSD), a unified framework for distilling from any teacher that provides reasoning traces. The model to be distilled from, termed the external teacher, first provides a reasoning trace as privileged context, which can be generated in an offline fashion. The student generates rollouts without this context, while a copy of the student conditioned on it, serving as the self-teacher, scores the same rollouts in the student's own token space. BSD uses the resulting token-level 
    
[^121]: BanglaBBQ：面向孟加拉国语境的问答偏见基准测试

    BanglaBBQ: A Bias Benchmark for Question Answering in the Bangladeshi Context

    [https://arxiv.org/abs/2610.24764](https://arxiv.org/abs/2610.24764)

    本文提出了 BanglaBBQ——一个针对孟加拉国语境、同时支持孟加拉语和英语的问答偏见基准测试，通过改编原有 BBQ 模板并新增五个本地化刻板印象类别，经本土标注者严格验证构建而成。

    

    本文提出了 BanglaBBQ，这是一个问答偏见基准测试，旨在捕捉植根于孟加拉国语境的刻板印象，涵盖孟加拉语和英语两种语言。遵循 KoBBQ 和 PakBBQ 的方法，BanglaBBQ 对原有的问答偏见基准测试（BBQ）进行了改编，将其模板分为三类：直接翻译（DT）、修改以使目标群体或语境符合孟加拉国情况（TM），以及因不适合孟加拉国语境而删除（SR）。这一过程保留了六个类别中的 140 个模板，包括 76 个 DT 模板和 64 个 TM 模板。随后新增了五个基于孟加拉国社会常见刻板印象的新编（NA）类别：婚姻状况、年龄与政治观念、地域来源、教育背景以及心理健康状况。孟加拉国本土标注者对所有 466 个新编模板进行了评判，其中 372 个模板因获得英语标注者的严格多数认可而被保留。

    arXiv:2610.24764v2 Announce Type: replace  Abstract: This paper presents BanglaBBQ, a question-answering bias benchmark designed to capture stereotypes grounded in the Bangladeshi context, in both Bangla and English. Following KoBBQ and PakBBQ, BanglaBBQ adapts the original Bias Benchmark for Question Answering (BBQ) by classifying its templates into three groups: directly translated (DT), modified to align target groups or contexts with Bangladesh (TM), and removed when deemed unsuitable for the Bangladeshi context (SR). This process retains 140 templates across six categories, comprising 76 DT and 64 TM templates. Five newly authored (NA) categories, based on stereotypes common in Bangladeshi society, are then added: marital status, age and political perception, regional origin, educational background, and mental-health status. Native Bangladeshi annotators judged all 466 newly authored templates, and 372 were retained because a strict majority of English annotators identified the in
    
[^122]: Annotrace：一个用于智能体轨迹步骤级标注的工作台

    Annotrace: A Workbench for Step-Level Annotation of Agent Traces

    [https://arxiv.org/abs/2610.23284](https://arxiv.org/abs/2610.23284)

    Annotrace 是一个支持六种主流轨迹格式的智能体轨迹步骤级标注工作台，它将运行记录解析为有序嵌套的步骤，通过人机协作方式进行标注并衡量标注者间一致性，在多智能体问答语料上精确复现了此前人工标注的235个标注单元。

    

    每一项研究LLM智能体为何失败的工作，都始于对手工标注轨迹，而每一项这样的研究都会构建自己的一次性工具。可观测性平台能够记录智能体轨迹，却不提供结构化标注；标注工具可以标注文本，却没有“步骤”、任务“委派”或“工具调用”的概念，导致最小可标注的执行单元不明确。Annotrace 可以读取六种轨迹格式，包括 OpenTelemetry GenAI、LangSmith 和 Langfuse，并将每次运行转换为有序的、嵌套的步骤列表。标注者编写自己的标注手册；一个前沿模型会根据标注者自己轨迹的样本提出初始标注集，先行作答，并被问到与人类完全相同的问题，从而使标注成为一个人机协作的过程。Annotrace 能够衡量任意两个标注者（无论是人类还是AI）之间的一致性。在一个多智能体问答语料库上，它精确复现了此前一次人工标注工作的235个标注单元，并且两个模型……（摘要在此处被截断）

    arXiv:2610.23284v1 Announce Type: new  Abstract: Every study of why LLM agents fail begins by labelling traces by hand, and every such study builds its own throwaway tool. Observability platforms record agent traces but offer no structured labelling; annotation tools label text but have no notion of a step, a delegation or a tool call, leaving it unclear what the smallest unit of execution to label should be. Annotrace reads six trace formats, including OpenTelemetry GenAI, LangSmith and Langfuse, and turns each run into an ordered, nested list of steps. The annotator writes their own labelling codebook; a frontier model proposes a starting set from a sample of their own traces, answers first, and is asked exactly the questions the human is asked, making annotation a joint human-AI process. Annotrace measures agreement between any two annotators, human or AI. On a multi-agent question-answering corpus it reproduces a previous manual effort's 235 annotation units exactly, and two models
    
[^123]: Agaram 2.5B：首个从零训练、支持工具调用的泰米尔语优先语言模型——面向低资源语言的实践指南

    Agaram 2.5B: The First Tamil-First Language Model Trained from Scratch with Tool Calling. A Playbook for Under-Represented Languages

    [https://arxiv.org/abs/2610.18420](https://arxiv.org/abs/2610.18420)

    该研究从零训练了首个泰米尔语优先、支持工具调用的25亿参数语言模型Agaram 2.5B，仅用8张H100 GPU、6天时间和约500亿token，就在翻译和泰米尔语指令遵循上超越了更大的印度语系模型Sarvam-1，为低资源语言的模型训练提供了可复制的实践方案。

    

    我们从零开始训练了Agaram 2.5B，一个25亿参数的泰米尔语-英语语言模型，使用约500亿个token（58%为泰米尔语，其余为英语、数学、代码和平行文本），在单节点8张H100 GPU上耗时约6天完成，并采用了一个与我们第一代模型不同的新分词器，该分词器能够保留换行符。通过对134,428段对话进行监督微调（其回复由一个开放权重的教师模型生成），再加上三轮简短的针对性训练，我们将其打造为一个聊天模型，能够以泰米尔文字、罗马化泰米尔语和英语回答问题，并具备工具调用能力：在25个工具提示（其模式已出现在训练数据中）上，它有23次选择了正确的工具，其中13次给出了完全符合预期的参数。发布的模型在FLORES和IN22的全部六个翻译方向上都优于Sarvam-1（一个20亿参数的印度语系基础模型）（FLORES chrF：译入泰米尔语54.3，译入英语55.2，而Sarvam-1分别为49.2和44.2），并且在泰米尔语原生的IndicIFEval基准上比其基础模型几乎翻倍（19.3%对10……（摘要在此处被截断）

    arXiv:2610.18420v1 Announce Type: cross  Abstract: We trained Agaram 2.5B, a 2.5-billion-parameter Tamil-English language model, from scratch on about 50 billion tokens (58% Tamil, the rest English, math, code and parallel text) on one node of eight H100 GPUs in about six days, with a new tokenizer that, unlike our first model's, keeps line breaks. Supervised fine-tuning on 134,428 conversations, whose replies an open-weight teacher model wrote, and three short targeted rounds turned it into a chat model that answers in Tamil script, romanized Tamil (Tanglish) and English and can call tools: on 25 tool prompts whose schemas appear in training it picks the right tool 23 times and gives exactly the expected arguments 13 times. The released model translates better than Sarvam-1, a 2B Indic base model, in all six FLORES and IN22 directions (FLORES chrF 54.3 into Tamil and 55.2 into English, against 49.2 and 44.2), and nearly doubles its base model on native-Tamil IndicIFEval (19.3% from 10
    
[^124]: 用于数学推理验证的注意力图曲率

    Attention-Graph Curvature for the Verification of Mathematical Reasoning

    [https://arxiv.org/abs/2610.16709](https://arxiv.org/abs/2610.16709)

    本研究提出利用语言模型注意力图的有向Forman曲率作为推理步骤正确性的信号，实验表明将该曲率附加到逐步验证器的输入中可显著提升其在ProcessBench上的数学推理验证性能。

    

    语言模型最近解决了公开的埃尔德什问题，并在国际数学奥林匹克竞赛题目上达到了金牌水平的表现。但它们以自然语言书写的解答尚无法被证明助手验证。一个错误的推理步骤可能使其后的所有步骤失效，因此尽早检测错误是有利的。已有若干方法被提出，基于执行推理的语言模型前向传递特征来快速检测错误推理。在本工作中，我们研究该模型注意力图的有向Forman曲率——一种在边上根据其周围注意力权重计算得到的局部量——是否携带关于某一步骤正确性的信息。为此，我们在ProcessBench上对四个逐步验证器进行了评估，分别测试了在输入中附加曲率值和不附加曲率值两种情况。附加曲率值提升了所有四个验证器的整体平均性能。

    arXiv:2610.16709v2 Announce Type: replace-cross  Abstract: Language models have recently solved open Erd\H{o}s problems and achieved gold-medal level performance on International Mathematical Olympiad problems. But their solutions, written in natural language, cannot yet be verified by proof assistants. An erroneous reasoning step can invalidate every step that comes after, so it is advantageous to detect errors early. Several methods have been proposed to detect erroneous reasoning quickly based on the features of the forward pass of the language model that performs reasoning. In this work, we study whether the directed Forman curvature of that model's attention graph, a local quantity computed at an edge from the attention weights around it, carries information about a step's correctness. To do so, we evaluate four per-step verifiers on ProcessBench with and without the curvature appended to their input. Appending curvature values improves the overall mean performance of all four ver
    
[^125]: 去噪层次化表示：面向语言建模的联合连续扩散

    Denoising Hierarchical Representations: Joint Continuous Diffusion for Language Modeling

    [https://arxiv.org/abs/2610.08738](https://arxiv.org/abs/2610.08738)

    提出层次化连续扩散语言模型（H-CDLM），通过并行扩散token本身及其粗粒度语义聚类等多种模态表示，并允许为各模态配置独立的采样器与调度，以极少的计算和参数开销显著提升连续扩散语言模型的性能。

    

    扩散语言模型有望实现与顺序无关的并行文本生成。近来，连续扩散和流匹配模型取得了显著进展，这得益于精心设计的token表示和扩散/流空间。在本工作中，我们提出了层次化连续扩散语言模型，这是一个简洁的框架，能够以极少的计算和参数开销进一步提升连续扩散语言模型的性能。借鉴离散扩散语言模型与连续图像扩散文献中关于联合扩散的思想，我们对多种模态进行并行扩散。这些模态在不同语义粒度上表示token：在我们的具体实现中，包括token本身以及通过对预训练token嵌入进行聚类所得到的更粗粒度的聚类。我们提出了一个通用设置，允许为每种模态配置各自的采样器和调度，以增强模态之间的相互作用。将该方法应用于CoBit时，得到了H-CoBit，它带来了巨大的性能提升。

    arXiv:2610.08738v1 Announce Type: new  Abstract: Diffusion Language Models (DLMs) hold the promise of order-agnostic, parallel text generation. Recently, continuous diffusion and flow matching models have seen substantial gains, driven by carefully crafted token representations and diffusion/flow spaces. In this work, we introduce Hierarchical Continuous Diffusion Language Models (H-CDLMs), a simple framework that further improves continuous DLMs with minimal compute and parameter overhead. Drawing on the discrete DLM and continuous image diffusion literature on joint diffusion, we diffuse multiple modalities in parallel. These modalities represent tokens at different semantic granularities: in our instantiation, the tokens themselves and coarser clusters obtained by clustering pretrained token embeddings. We propose a general setup that allows per-modality samplers and schedules to enhance the interplay between modalities. Applied to CoBit, this yields H-CoBit, which delivers large em
    
[^126]: 迷失于bf16类型转换：导出三值语言模型可能抵消大部分低学习率代码改动的效果

    Lost in the bf16 Cast: Exporting Ternary Language Models Can Revert Most Low-Learning-Rate Code Changes

    [https://arxiv.org/abs/2610.07853](https://arxiv.org/abs/2610.07853)

    该研究审计发现，三值语言模型导出流程中先将潜在权重转换为bf16，会在阈值处因“舍入到偶数”规则被错误映射为零，导致Falcon-E-1B-Base部署后GSM8K准确率从58.79%暴跌至0.78%，几乎抹去了低学习率微调带来的改进。

    

    BitNet b1.58、Falcon-E和BitCPM等三值语言模型先以更高精度的潜在权重进行微调，再通过导出步骤生成三值代码进行部署；在三个实验室公开记录的流程中，该导出步骤均会先将潜在权重转换为bf16。我们对三个实验室的这些流程进行了审计。在已发布的检查点中，对随附潜在权重执行fp32量化所得的结果与实际部署代码在Falcon-E和BitCPM上有0.83%–1.77%的代码不一致，在BitNet 2B-4T上有1.530%不一致；对于Falcon-E和BitCPM，绝大多数不一致源于这样的乘积：其bf16舍入恰好落在阈值上，并被“舍入到偶数”（ties-to-even）规则映射为零，而未经修改的onebitllms导出器可以逐字节复现全部四个Falcon-E发布版本。在微调终点，当学习率按标称的学习率与bf16-ULP之比选取时，按文档所述流程导出会使Falcon-E-1B-Base的贪婪解码GSM8K严格准确率从58.79%跌至0.78%，使BitCPM-CANN-（原文摘要在此处截断）的准确率从36.13%跌至0.39%。

    arXiv:2610.07853v1 Announce Type: cross  Abstract: Ternary language models such as BitNet b1.58, Falcon-E and BitCPM are fine-tuned with higher-precision latent weights and deployed as ternary codes produced by an export step that, in the labs' documented pipelines, first casts the latents to bf16. We audit those pipelines across three labs. In released checkpoints, fp32 quantization of the shipped latents disagrees with the deployed codes on 0.83-1.77% of codes in Falcon-E and BitCPM and on 1.530% in BitNet 2B-4T; for Falcon-E and BitCPM most disagreements are products that bf16 rounding lands exactly on the threshold, which ties-to-even maps to zero, and the unmodified onebitllms exporter reproduces all four Falcon-E releases byte for byte. At fine-tuned endpoints, with learning rates selected to match a nominal learning-rate-to-bf16-ULP ratio, the documented export lowers greedy GSM8K strict accuracy from 58.79% to 0.78% for Falcon-E-1B-Base and from 36.13% to 0.39% for BitCPM-CANN-
    
[^127]: 分数并非结构：大脑对齐与跨语言迁移

    The Score Is Not the Structure: Brain Alignment and Cross-Lingual Transfer

    [https://arxiv.org/abs/2610.03827](https://arxiv.org/abs/2610.03827)

    相似性分数可能反映的是测量工具本身的局限而非模型与大脑或跨语言间的真实共享结构，当测量工具在部分条件下失效或统计单位选择改变时，原本显著的梯度效应会大幅减弱甚至消失。

    

    相似性分数常被用作模型与大脑或跨语言间共享结构的证据。我们探讨当结构被移除或测量工具失效时，这样的分数实际反映的是什么，并在两个场景中进行了验证。在十七种语言中，语法性探针在语言距离越远时迁移效果越差（r = -0.66），但探针自身的准确率也沿同一轴线下降，在四种语言中处于随机水平（这四种语言是距离最远的五种中的四种）。剔除这些语言会使已解释方差减半，同时由于也缩小了距离范围，该设计无法判断梯度中有多少来自测量工具本身。将272个语言对视为独立样本时，导向效应的p值为0.0006；但以十七种语言为单位时，该效应不显著（p = 0.155）。在大脑对齐方面，一个训练目标将语言模型与fMRI响应的相似性从0.10提升到0.34（可靠性上限为0.54），然而

    arXiv:2610.03827v2 Announce Type: replace-cross  Abstract: Similarity scores are often offered as evidence that a model shares structure with the brain or across languages. We ask what such a score reads when that structure is removed, or when the instrument measuring it does not work, in two settings. Across seventeen languages, a grammaticality probe transfers worse between more distant languages (r = -0.66). But the probe's own accuracy falls along the same axis and is at chance in four languages, four of the five most distant. Dropping them halves the explained variance, and because it also narrows the range of distances, the design cannot say how much of the gradient is the instrument. Counting the 272 language pairs as independent gives p = 0.0006 for a steering effect that is null when the seventeen languages are the unit (p = 0.155). In brain alignment, a training objective raises a language model's similarity to fMRI responses from 0.10 to 0.34 (reliability ceiling 0.54), yet 
    
[^128]: 配对难度至关重要：重新思考成对式LLM裁判评估与一致性

    Pair Difficulty Matters: Rethinking Pairwise LLM-as-a-Judge Evaluation and Consistency

    [https://arxiv.org/abs/2609.37577](https://arxiv.org/abs/2609.37577)

    该论文指出，传统用于评估LLM裁判可靠性的三大代理指标（位置偏差、传递性、成对一致性）具有误导性——在Bradley-Terry几何下它们被排名差距接近的配对所主导，而此类配对出现不一致在信息论上是必然的，因此不应以这些指标否定能力出色的LLM评估器。

    

    大型语言模型（LLM）裁判被广泛用于通过成对比较来对文本及文本生成系统进行排序，其可靠性通常通过三个代理指标来评估：位置偏差、传递性和成对一致性（自评或人工标注）。由于这些代理指标驱动着裁判的选择与基准测试，大量报告裁判在这些指标上表现不佳的文献可能会使实践者放弃原本有能力的评估器。我们认为这种评估具有误导性。在成对聚合所依据的Bradley-Terry几何结构下，每个代理指标都由排名差距接近的配对所主导，而在这类配对中，不一致性在信息论上是可以预期的，且个别判断对总体排名的贡献微乎其微；相比之下，排名差距较大的配对虽然承载着排名信号，却几乎不会影响这些代理指标。我们对该论点进行了形式化，并在受控模拟以及两个人工评分语料库上进行了验证：这些代理指标与排名质量仅呈弱相关。

    arXiv:2609.37577v1 Announce Type: new  Abstract: Large Language Model judges are widely used to rank texts and text-generating systems through pairwise comparison, and their reliability is typically assessed via three proxies: position bias, transitivity, and pairwise agreement (self- or human-labeled). Because these proxies drive judge selection and benchmarking, a substantial literature reporting that judges perform poorly on them risks steering practitioners away from otherwise capable evaluators. We argue this assessment is misleading. Under the Bradley--Terry geometry underlying pairwise aggregation, each proxy is dominated by close-rank-gap pairs, where inconsistency is information-theoretically expected and individual verdicts contribute little to the aggregate ranking; far-gap pairs carry the ranking signal but barely move the proxies. We formalize this argument and validate it in a controlled simulation and on two human-rated corpora: the proxies correlate only weakly with ran
    
[^129]: EvoIn：面向智能体微调的演化与内化桥梁

    EvoIn: Bridging Evolution and Internalization for Agent Fine-Tuning

    [https://arxiv.org/abs/2609.35290](https://arxiv.org/abs/2609.35290)

    EvoIn提出了一个连接演化与内化的智能体微调框架，通过分析执行轨迹来演化并验证新的决策流程，再将改进的推理轨迹改写为去除框架指令引用的自包含推理并微调模型，从而将决策逻辑内化为模型自身的能力。

    

    近期的工作探索了通过共同演化智能体的执行框架和模型来提升智能体性能，但往往采用“大杂烩”式的方法，将新工具、新决策流程以及对演化后框架的模型适配统统归入单一的“智能体改进”概念之下。在本文中，我们转而研究智能体如何改进其自身的决策流程。具体而言，我们提出了EvoIn，一个连接演化与内化的智能体微调框架。EvoIn首先分析智能体的执行轨迹，通过在框架中临时实例化的方式来演化并验证新的决策流程。经过验证的流程会引导智能体生成改进后的推理轨迹。随后，这些轨迹被改写为自包含的推理轨迹，在去除对框架指令的显式引用的同时，将由此诱导出的决策逻辑表达为模型自身的推理。最后，EvoIn对模型进行微调。

    arXiv:2609.35290v2 Announce Type: replace  Abstract: Recent work has explored improving agents by jointly evolving their harnesses and models, but often takes a ''potpourri'' approach that bundles together new tools, new decision-making procedures, and model adaptation to the evolved harness under a single notion of agent improvement. In this paper, we instead investigate how agents can improve their decision-making procedures. In particular, we propose EvoIn, an agent fine-tuning framework that bridges evolution and internalization. EvoIn first analyzes agent execution traces to evolve and validate new decision-making procedures by temporarily instantiating them in the harness. The validated procedures guide the agent to generate improved reasoning traces. These traces are then rewritten into self-contained reasoning traces, removing explicit references to harness instructions while expressing the induced decision logic as the model's own reasoning. Finally, EvoIn fine-tunes the model
    
[^130]: ReMCTS：面向代码生成的反思增强蒙特卡洛树搜索

    ReMCTS: Reflection-Enhanced Monte Carlo Tree Search for Code Generation

    [https://arxiv.org/abs/2609.34717](https://arxiv.org/abs/2609.34717)

    ReMCTS提出了一种以执行结果为依据、结合记忆增强与LLM引导的蒙特卡洛树搜索框架，通过保留分支调试上下文、跨分支检索失败经验并区分检查失败与证据缺失，显著提升了代码生成的准确性与修复效率。

    

    开放权重的大语言模型（LLM）能够根据自然语言提示生成函数级程序，但看似合理的候选程序仍会在隐含语义上出错，并在多次修复尝试中重复犯错。我们提出了ReMCTS，这是一个以执行结果为依据、具备记忆增强、由LLM引导的MCTS风格搜索框架。它将程序候选组织为树状态，保留分支局部的调试上下文，跨分支检索失败经验，并能够区分“检查失败”与“证据不可用”两种情况。在HumanEval和MBPP-Sanitized数据集的留出评估中，采用可见测试的ReMCTS在10个模型-数据集组合中的8个上优于直接生成，而仅依赖代理指标的搜索则稳定性较差。通过受控的树搜索、采样、修复和记忆消融实验，论文刻画了这些增益的来源与局限。一项包含30个任务的HumanEval-X C++试点实验进一步证明了该框架与编译器支持执行的兼容性，但这并不构成广泛的多语言评估（原文摘要在此处截断）。

    arXiv:2609.34717v2 Announce Type: replace  Abstract: Open-weight large language models (LLMs) can generate function-level programs from natural-language prompts, but plausible candidates still fail on hidden semantics and repeat mistakes across repair attempts. We present ReMCTS, an execution-grounded, memory-augmented, LLM-guided MCTS-style search framework. It organizes program candidates as tree states, retains branch-local debugging context, retrieves failure experience across branches, and distinguishes failed checks from unavailable evidence. On HumanEval and MBPP-Sanitized, visible-test ReMCTS improves over direct generation in 8 of 10 model-dataset pairs under held-out evaluation, whereas proxy-only search is less stable. Controlled tree-search, sampling, repair, and memory ablations characterize the source and limits of these gains. A 30-task HumanEval-X C++ pilot further demonstrates compatibility with compiler-backed execution, but does not constitute a broad multilingual ev
    
[^131]: 学习锚定提示空间以实现大语言模型的持续适配

    Learning an Anchored Prompt Space for Continual Adaptation of Large Language Models

    [https://arxiv.org/abs/2609.32499](https://arxiv.org/abs/2609.32499)

    提出LAPS方法，通过自蒸馏对齐历史提示，并构建以任务特定软提示为顶点、贝塞尔控制提示塑造几何结构的锚定提示空间，从而在持续适配大语言模型时保留提示有效性并促进跨任务正向迁移。

    

    持续适配大语言模型需要在获取新知识的同时保留先前学习到的能力。尽管联合适配模型参数和任务特定的软提示是一种有前景的方法，但它存在两个关键局限性：先前学习的提示可能随着模型的演进而失去有效性，并且支持跨任务迁移的提示之间的关系没有被显式学习。我们提出了学习锚定提示空间（LAPS），该方法保留历史提示的有效性，并学习任务特定软提示之间的关系以促进正向迁移。历史提示首先通过自蒸馏与更新后的模型对齐。随后，LAPS构建一个锚定提示空间，其顶点对应于已学习的任务特定软提示，其中间几何结构由可学习的贝塞尔控制提示塑造。一旦学习了这个空间，LAPS即可识别最佳表现……

    arXiv:2609.32499v2 Announce Type: replace-cross  Abstract: Continually adapting large language models requires acquiring new knowledge while preserving previously learned capabilities. Although jointly adapting model parameters and task-specific soft prompts is promising, it has two key limitations: previously learned prompts may lose effectiveness as the model evolves, and relationships among prompts that could support cross-task transfer are not explicitly learned. We propose Learning an Anchored Prompt Space (LAPS), which preserves historical prompt effectiveness and learns relationships among task-specific soft prompts to facilitate positive transfer. Historical prompts are first aligned with the updated model through self-distillation. LAPS then constructs an anchored prompt space whose vertices correspond to learned task-specific soft prompts and whose intermediate geometry is shaped by learnable B\'ezier control prompts. Once this space is learned, LAPS identifies the best-perfo
    
[^132]: 智能体软件工程基准测试究竟在衡量什么？超越类别标签的任务需求与智能体行为画像

    What Does an Agentic Software Engineering Benchmark Measure? Profiling Task Demands and Agent Behaviour Beyond What Category Labels Reveal

    [https://arxiv.org/abs/2609.01271](https://arxiv.org/abs/2609.01271)

    本文提出Spread-Novelty-Centrality（SNC）三轴画像方法来刻画仓库级编码任务的真实需求，发现类别标签是任务需求的不可靠代理指标，且智能体行为轨迹能揭示人工标准答案无法反映的任务需求信息。

    

    智能体软件工程基准测试通常通过名义类别标签（如“缺陷修复”或“功能实现”）来概括，然而携带相同标签的基准测试是通过截然不同的构建流程创建的。因此，标签几乎无法揭示基准测试所要求的工程工作。我们引入了 Spread--Novelty--Centrality（SNC）画像，这是一种基于实证软件工程研究的三轴特征描述方法，用于刻画仓库级编码任务的需求。我们将该画像应用于五个广泛使用的基准测试，以及两个模型家族在三种规模下的14,922条轨迹，并报告了三项发现。（1）类别标签是任务需求的不可靠代理指标，因为每一对基准测试在至少两个SNC轴上均存在统计学上的显著分离，且这些分离可以追溯到具体的构建决策。（2）智能体行为揭示了人类编写的标准答案（gold solution）无法揭示的需求。智能体产出的解决方案……

    arXiv:2609.01271v1 Announce Type: cross  Abstract: Agentic software engineering benchmarks are typically summarized by nominal category labels such as "bug fix" or "feature implementation," yet benchmarks carrying the same label are built through very different curation pipelines. A label thus reveals little about the engineering work a benchmark demands. We introduce the Spread--Novelty--Centrality (SNC) profile, a three-axis characterization of the demands of repository-level coding tasks, grounded in empirical software engineering research. We apply the profile to five widely used benchmarks and 14,922 trajectories of two model families at three scales, and report three findings. (1) A label is an unreliable proxy for task demands, as every pair of benchmarks is statistically separated on at least two SNC axes, and the separations trace back to specific curation decisions. (2) Agent behaviour reveals demands that the human-written gold solution cannot. Agents produce larger solution
    
[^133]: 面向大语言模型时序评估与知识更新的合成世界

    Synthetic Worlds for Temporal Evaluation and Knowledge Updating in LLMs

    [https://arxiv.org/abs/2609.00184](https://arxiv.org/abs/2609.00184)

    该论文提出了一个模拟驱动的合成框架，通过虚构未来世界的 ParallelEvents 基准避免评估污染，并利用 Synapse 训练框架（结合中期训练与指令微调）实现大语言模型的可扩展知识更新，性能比现有方法提升 14.23%。

    

    大语言模型（LLM）依赖于静态的预训练语料库，导致其知识随时间推移而变得过时。现有的知识编辑评估方法要么容易遭受快速的数据污染，要么依赖于与现有刚性知识相冲突的反事实编辑。在本工作中，我们提出了一个合成的、模拟驱动的框架，用于研究大语言模型中的知识插入。我们引入了 {\sc ParallelEvents}，这是一个由虚构但逼真的未来世界构成的基准，能够生成连贯的事件轨迹以进行受控评估，在避免污染的同时保持一致性。基于该数据集，我们开发了 {\sc Synapse}，这是一个利用模型自身生成的数据、通过中期训练（mid-training）和指令微调来更新模型参数的训练框架。这一合成流程实现了可扩展的知识整合，而无需昂贵的人工策划数据。实验结果表明，{\sc Synapse} 的性能比现有方法高出 14.23%。

    arXiv:2609.00184v1 Announce Type: new  Abstract: Large language models (LLMs) rely on static pretraining corpora, causing their knowledge to become outdated over time. Existing approaches for evaluating knowledge edits either suffer from rapid contamination or rely on counterfactual edits that conflict with rigid existing knowledge. In this work, we propose a synthetic, simulation-driven framework for studying knowledge insertion in LLMs. We introduce {\sc ParallelEvents}, a benchmark of fictional yet realistic future worlds that generates coherent event trajectories for controlled evaluation, avoiding contamination while preserving consistency. Building on this dataset, we develop {\sc Synapse}, a training framework that uses model-generated data to update model parameters via mid-training and instruction tuning. This synthetic pipeline enables scalable knowledge integration without costly human-curated data. Empirically, {\sc Synapse} outperforms existing methods by 14.23\%, demonstr
    
[^134]: FKG.in的验证：LLM增强的印度食品知识中的健全性评估

    Validating FKG.in: Soundness Assessment in LLM-Augmented Indian Food Knowledge

    [https://arxiv.org/abs/2608.29249](https://arxiv.org/abs/2608.29249)

    本文作为印度食品知识图谱FKG.in的一部分，提出了一种半自动化的健全性评估工作流程，通过结合形式文法、词汇检查、统计启发式、Set Transformer连贯性建模和检索验证的多阶段方法，识别并解决LLM从非正式烹饪来源提取和增强结构化食谱数据时的常见失败模式。

    

    在线烹饪生态系统中，由大型语言模型（LLM）生成、修改或总结的食谱内容日益增多。虽然这些输出通常看似合理，但可能包含虚构的食材、被误述的用量或文化上不合常理的食材组合，从而限制了其在下游应用和知识图谱构建中的适用性。在本文中，我们提出了一种半自动化的健全性评估工作流程，用于验证由LLM从非正式烹饪来源中提取和增强的结构化食谱数据。该流程作为印度食品知识图谱FKG.in的一部分开发而成，通过结合形式文法、基于词汇的检查、统计启发式方法、基于Set Transformer的连贯性建模以及基于检索的验证等多阶段流程，识别并解决常见的失败模式，包括结构性不一致、语义和逻辑上的不连贯以及与源文本的偏差。

    arXiv:2608.29249v1 Announce Type: new  Abstract: The online culinary ecosystem is increasingly populated by recipe content generated, modified, or summarized by Large Language Models (LLMs). While often plausible, such outputs may contain hallucinated ingredients, misrepresented quantities, or culturally implausible combinations, limiting their suitability for downstream applications and knowledge graph construction. In this paper, we present a semi-automated soundness assessment workflow for validating structured recipe data extracted and augmented by LLMs from informal culinary sources. Developed as part of FKG.in, a knowledge graph of Indian food, the pipeline identifies and addresses common failure modes, including structural inconsistencies, semantic and logical incoherence, and deviations from the source text, through a multi-stage process combining formal grammars, vocabulary-based checks, statistical heuristics, Set Transformer-based coherence modeling, and retrieval-based veri
    
[^135]: 核令牌矛盾：一种快速且有原则的LLM声明不确定性量化方法

    Kernel Token Contradiction: a Fast and Principled Approach for LLM Claim Uncertainty Quantification

    [https://arxiv.org/abs/2608.22506](https://arxiv.org/abs/2608.22506)

    本文提出了一种轻量级且快速的声明级不确定性量化方法KTC，通过核表示和冯·诺依曼熵，在CPU上实现了比现有方法显著的速度提升。

    

    声明级不确定性量化（UQ）旨在通过评估大型语言模型（LLM）输出中每个声明的真实性，来缓解其可靠性不足的问题。我们引入了核令牌矛盾（KTC），一种在现实白盒条件下计算声明级UQ的轻量级方法。KTC将LLM生成过程中涉及的候选令牌表示为一个正半定核，该核整合了LLM的条件分布和令牌矛盾分数。然后，我们使用冯·诺依曼熵来量化该核的不确定性。为估计令牌矛盾，我们开发了一种基于维基百科语料库频率统计的新方法。尽管仅使用CPU，我们的方法相比基于交叉编码器的最先进GPU加速方法实现了超过8.2倍的速度提升，相比性能相当的仅CPU方法实现了超过65倍的速度提升。我们的评估涵盖了两个基准测试。

    arXiv:2608.22506v1 Announce Type: new  Abstract: Claim-level Uncertainty Quantification (UQ) aims to mitigate the lack of reliability of Large Language Models (LLMs) by evaluating the factuality of each claim in their outputs. We introduce Kernel Token Contradiction (KTC), a lightweight approach to compute claim-level UQ under realistic white-box conditions. KTC represents the candidate tokens involved in LLM generation as a positive semi-definite kernel that integrates both the LLM's conditional distribution and a token contradiction score. We then use the Von Neumann entropy to quantify the uncertainty of this kernel. To estimate token contradiction, we develop a new approach based on frequency statistics from the Wikipedia corpus. Although CPU-only, our approach achieves over an 8.2x speedup compared to state-of-the-art GPU-accelerated methods based on cross-encoders, and over a 65x speedup compared to CPU-only methods with comparable performance. Our evaluation spans two benchmarks
    
[^136]: 机器学习与数字语用学：哪种词类对表情符号使用影响最大？

    Machine learning and digital pragmatics: Which word category influences emoji use most?

    [https://arxiv.org/abs/2608.21975](https://arxiv.org/abs/2608.21975)

    本研究通过MARBERT模型和逻辑回归分析发现，在口语阿拉伯语社交媒体帖子中，动词类别对表情符号使用的影响最强，尽管名词在频率上占主导。

    

    arXiv:2608.21975v1 公告类型：新 摘要：本研究考察了最先进的MARBERT模型在数字语用学方法（DPA）框架内识别X平台上表情符号使用相关词汇/语用类别的表现。使用Python从X收集了包含表情符号的15856条口语阿拉伯语（CA）帖子作为净语料库。文本被分词并规范化为4个词汇类别，即名词_规范、动词_规范、形容词_规范和副词_规范，以及2个语用/结构类别，即疑问_规范和感叹_规范。MARBERT经过微调和优化，以识别哪个类别在标准指标上得分更高，从而与表情符号使用相关，同时使用二元逻辑回归来检验哪个类别在统计上与表情符号出现相关。研究结果显示，名词在规范频率上主导语料库（平均值=0.675，标准差=0.161），其次是动词（平均值=0.083，标准差=0.100）。然而，动词对表情符号使用的影响最强。

    arXiv:2608.21975v1 Announce Type: new  Abstract: This study examines the performance of the state-of-the-art MARBERT model in identifying the lexical/pragmatic category associated with emoji use on X within a digital pragmatics approach (DPA). A net corpus of 15856 Colloquial Arabic (CA) posts containing emojis was collected from X using Python. The texts were tokenized and normalized into 4 lexical categories, namely noun_norm, verb_norm, adj_norm, and adverb_norm, and 2 pragmatic/structural categories, question_norm and exclamation_norm. MARBERT was finetuned and optimized to identify which category scores standard metrics more, hence associated with emoji use, while binary logistic regression was used to examine which category is statistically associated with emoji occurrence. Findings unveil that nouns dominate the corpus in normalized frequency (M = 0.675, SD = 0.161), followed by verbs (M = 0.083, SD = 0.100). However, verbs have the strongest influence of emoji use indicated by 
    
[^137]: TextCloak：通过强化学习驱动的不可学习样本保护数据免受未经授权的LLM微调

    TextCloak: Safeguarding Data from Unauthorized LLM Fine-Tuning via RL-Driven Unlearnable Examples

    [https://arxiv.org/abs/2607.28862](https://arxiv.org/abs/2607.28862)

    TextCloak提出了一种基于强化学习的数据合成框架，通过生成既能保持语义忠实性和语言自然性又能降低模型学习效果的不可学习样本，来保护文本数据免受未经授权的大语言模型微调。

    

    大型语言模型（LLM）的发展在众多语言任务中取得了显著进展，同时也引发了人们对未经授权的在线数据利用和隐私泄露日益增长的担忧。不可学习样本（UEs）作为一种有前景的防御手段应运而生，其通过在数据中注入精心设计的扰动，使得在这些数据上训练的模型效用下降。然而，现有的文本保护方法主要是针对分类任务和判别式语言模型设计的，这限制了它们在LLM中的广泛适用性（例如指令遵循或推理任务）。在这项工作中，我们提出了TextCloak，这是一个由强化学习驱动的数据合成框架，用于保护文本数据免受未经授权的LLM微调。TextCloak采用一种不可学习样本生成策略，将干净文本转换为不可学习样本，同时保持语义忠实性和语言自然性。为了优化该策略，w...

    arXiv:2607.28862v2 Announce Type: replace-cross  Abstract: The development of Large Language Models (LLMs) has led to significant advances across a wide range of language tasks, while simultaneously raising growing concerns about unauthorized online data exploitation and privacy leakage. Unlearnable examples (UEs) emerge as a promising defense by injecting crafted perturbations into data such that models trained on them exhibit degraded utility. However, existing methods for text protection are primarily designed for classification tasks and discriminative language models, which limits their broad applicability (e.g., instruction following or reasoning) in LLMs. In this work, we propose **TextCloak**, an RL-driven data synthesis framework that safeguards textual data against unauthorized LLM fine-tuning. TextCloak employs a *UE generative policy* that transforms clean text into unlearnable examples while preserving semantic fidelity and linguistic naturalness. To optimize the policy, w
    
[^138]: 通过因果归因剪枝保持大型语言模型的推理性能

    Pruning via Causal Attribution Preserves Reasoning Performance in Large Language Models

    [https://arxiv.org/abs/2606.19350](https://arxiv.org/abs/2606.19350)

    提出无需训练的因果归因剪枝方法 CAP，通过干预式测量注意力头对推理任务的因果贡献来指导细粒度权重剪枝，在 20% 稀疏度下相比 Wanda 在 ARC-Challenge 上获得高达 61% 的相对准确率提升，从而在降低推理成本的同时有效保持大语言模型的推理能力。

    

    大型语言模型（LLMs）在多步推理方面表现出色，但会产生高昂的推理成本。我们提出了因果归因剪枝（Causal Attribution Pruning, CAP），这是一种无需训练的方法，通过测量注意力头对推理任务的因果影响来识别关键的注意力头，并利用这些头级别的分数来指导细粒度的权重剪枝。对于每个注意力头，CAP 在一个由推理问题组成的小型校准集上进行前向传播时，估计将该注意力头掩蔽后的预期性能下降，随后将这些因果分数转换为相应投影矩阵的权重级重要性值。与仅基于权重大小或基于激活的剪枝准则不同，CAP 的干预式测量能够直接捕捉每个注意力头的功能性贡献，在 20% 稀疏度下于 ARC-Challenge 上相比 Wanda 取得了高达 61% 的相对准确率提升。我们使用 Llama-3-8B-Instruct 和 Mistral-7 模型在 GSM8K、StrategyQA 和 ARC-Challenge 上对 CAP 进行了评估。

    arXiv:2606.19350v2 Announce Type: replace  Abstract: Large language models (LLMs) excel at multi-step reasoning but incur substantial inference cost. We introduce Causal Attribution Pruning (CAP), a training-free method that identifies critical attention heads by measuring their causal impact on reasoning tasks and uses these head-level scores to guide fine-grained weight pruning. For each attention head, CAP estimates the expected performance degradation when the head is masked during forward passes on a small calibration set of reasoning problems. These causal scores are then converted into weight-level importance values for the corresponding projection matrices. Unlike magnitude-only or activation-based criteria, CAP's interventional measurement directly captures each head's functional contribution, yielding relative accuracy gains of up to 61% over Wanda on ARC-Challenge at 20% sparsity. We evaluate CAP on GSM8K, StrategyQA, and ARC-Challenge using Llama-3-8B-Instruct and Mistral-7
    
[^139]: 面向动态声源的时空音频语言建模

    Spatio-Temporal Audio Language Modeling for Dynamic Sound Sources

    [https://arxiv.org/abs/2606.14141](https://arxiv.org/abs/2606.14141)

    该论文提出了基于FOA渲染的时空音频问答数据集ST-AudioQA及配套的ST-Audio编码器和ST-AudioLM模型，联合建模声源语义与随时间变化的声源轨迹，使大语言模型能够回答关于声源身份、位置和运动的时空音频问答问题。

    

    声音事件具有语义身份、位置和运动轨迹，然而现有的音频-语言模型通常将音频片段编码为全局事件内容；相反，声音事件定位模型虽然能够随时间追踪声源方向，但其语义覆盖范围有限，难以支持语言推理。为弥合这一差距，我们提出了 ST-AudioQA，一个基于静态和移动声源的一阶高保真立体声（FOA）渲染构建的时空音频问答数据集与基准。每个场景都提供声源身份、活动状态、方向、距离和运动元数据，从而支持密集的轨迹监督，以及关于“什么在发声、它在哪里、它如何移动以及声源之间如何关联”的问答。我们进一步提出了 ST-Audio Encoder，联合学习事件语义和随时间变化的声源轨迹；以及 ST-AudioLM，将这些音频 token 输入大语言模型以进行时空音频问答。实验表明，这种表示改进了平衡……

    arXiv:2606.14141v2 Announce Type: replace-cross  Abstract: Sound events have semantic identities, locations, and trajectories, yet existing audio-language models typically encode clips as global event content. Conversely, sound event localization models track source directions over time but offer limited semantic coverage for language reasoning. To bridge this gap, we introduce ST-AudioQA, a spatio-temporal audio QA dataset and benchmark built from first-order ambisonic (FOA) renderings of static and moving sound sources. Each scene provides source identity, activity, direction, distance, and motion metadata, enabling dense trajectory supervision and QA about what is sounding, where it is, how it moves, and how sources relate. We further introduce ST-Audio Encoder, which jointly learns event semantics and time-resolved source trajectories, and ST-AudioLM, which exposes these audio tokens to an LLM for spatio-temporal audio QA. Experiments show that this representation improves the bala
    
[^140]: 大语言模型能够泄露训练数据，但它们想这样做吗？评估大语言模型的记忆倾向性

    LLMs Can Leak Training Data But Do They Want To? Evaluating Memorization Propensity in LLMs

    [https://arxiv.org/abs/2606.06286](https://arxiv.org/abs/2606.06286)

    本文提出PropMe框架和SimpleTrace工具，首次将大语言模型“能否”泄露训练数据与“是否会主动”泄露训练数据区分开来，发现自然句子提示下的主动泄露远少于前缀攻击，但无提示或常见词提示时的泄露程度与前缀攻击相当。

    

    大语言模型能够复现训练数据，但现有的记忆评估大多衡量的是模型能否被诱导复现训练数据，而非模型在没有对抗性提示的情况下是否会主动这样做。我们提出了PropMe框架，将能力评估（前缀攻击）与非对抗性的倾向性评估相结合，并引入一个倾向性指标来关联两者。我们使用SimpleTrace——一个开源的OLMoTrace片段匹配批量实现——将每一次生成结果与模型的完整索引训练语料库进行比对。我们在Common Pile、Dynaword和Dolma3上评估了Comma、其持续预训练的后续模型DFM Decoder以及OLMo3 32B，采用四种非对抗性设置，范围从无提示到来自未见过的同领域来源的句子。我们的结果表明，自然句子提示所复现的训练文本远少于前缀攻击。然而，在无提示或仅使用单个常见词的情况下，模型生成的复现量与前缀攻击大致相当。

    arXiv:2606.06286v2 Announce Type: replace-cross  Abstract: Large language models can reproduce training data, but memorization evaluations mostly measure whether models can be made to do so, not whether they do so without adversarial prompting. We introduce PropMe, a framework that pairs a capability evaluation (prefix attacks) with non-adversarial propensity evaluation, and a propensity metric to relate the two. We check every generation against the model's full indexed training corpus with SimpleTrace, an open-source batch implementation of OLMoTrace span matching. We evaluate Comma, its continually pre-trained descendant DFM Decoder, and OLMo3 32B on Common Pile, Dynaword, and Dolma3, with four non-adversarial settings ranging from no prompt to sentences from unseen in-domain sources. Our results show that natural sentence prompts reproduce far less training text than prefix attacks. Generation with no prompt or a single common word, however, reproduces about as much as prefix attac
    
[^141]: 多智能体推理中的流式通信

    Streaming Communication in Multi-Agent Reasoning

    [https://arxiv.org/abs/2606.05158](https://arxiv.org/abs/2606.05158)

    StreamMA通过让智能体即时流式传递每个推理步骤实现流水线化，首次证明了流式通信既能降低多智能体推理延迟、又能提升推理效果，并给出了流式、串行与单一协议的首个闭式联合理论分析。

    

    多智能体推理系统采用“先生成后传递”的范式，导致端到端延迟随流水线深度线性增长。我们提出StreamMA，一种多智能体推理系统，它在每个推理步骤生成后立即将其流式传输给下游智能体，使相邻智能体流水线化，从而降低延迟。令人惊讶的是，这种流水线化还能提升推理效果：由于多步推理的质量并不均匀，早期步骤比后期步骤更可靠，利用这些可靠的早期步骤而非完整推理链，可以避免易出错的后期步骤误导下游智能体。我们首次对流式、串行和单一协议进行了闭式联合分析，形式化了上述两种优势，推导出有效性排序、加速比上界和成本比。在涵盖数学、科学和代码的八个推理基准上，使用三个前沿大语言模型（Claude Opus 4.6、GPT-5.4 和 GLM-……（原文摘要在此处截断）

    arXiv:2606.05158v3 Announce Type: replace-cross  Abstract: Multi-agent reasoning systems adopt a generate-then-transfer paradigm that forces end-to-end latency to scale linearly with pipeline depth. We introduce StreamMA, a multi-agent reasoning system that streams each reasoning step to downstream agents as soon as it is generated, pipelining adjacent agents and thus reducing latency. Surprisingly, this pipelining also improves effectiveness: because multi-step reasoning quality is non-uniform and early steps are more reliable than later ones, working with these reliable early steps instead of the full chain prevents error-prone late steps from misleading downstream agents. We formalize both advantages with the first closed-form joint analysis of stream, serial, and single protocols, deriving the effectiveness ordering, speedup upper bound, and cost ratio. Across eight reasoning benchmarks spanning mathematics, science, and code, three frontier LLMs (Claude Opus 4.6, GPT-5.4, and GLM-
    
[^142]: 智能体规划基准：面向大语言模型智能体规划能力的诊断框架

    Agent Planning Benchmark: A Diagnostic Framework for Planning Capabilities in LLM Agents

    [https://arxiv.org/abs/2606.04874](https://arxiv.org/abs/2606.04874)

    APB是一个包含4,209个多模态案例的智能体规划专用诊断基准，能够定位大语言模型智能体在长程规划、工具噪声鲁棒性等方面的系统性弱点，且其引导的改进可持续提升规划质量与下游执行表现。

    

    规划是大语言模型智能体的核心能力：在行动之前，智能体必须分解目标、选择工具、对约束条件进行推理，并判断任务何时不可行。然而，现有的智能体评估通常只报告端到端的成功与否，难以确定失败究竟源于规划还是执行。我们提出了智能体规划基准，这是一个专门针对规划能力的诊断性基准，包含跨22个领域、五种设置的4,209个多模态案例，涵盖整体规划、基于反馈的分步规划，以及在存在无关工具、损坏工具和不可解决任务情况下的鲁棒性测试。通过对12个多模态大语言模型的评估，APB揭示了模型在长程规划、工具噪声鲁棒性、校准的拒绝行为以及推理时改进方面存在系统性弱点。我们还在200个ToolSandbox任务和200个τ²-bench任务上进一步验证了APB，结果表明APB引导的改进能够持续提升规划正确性、规划等级以及下游执行指标。

    arXiv:2606.04874v4 Announce Type: replace  Abstract: Planning is central to LLM agents: before acting, an agent must decompose goals, select tools, reason over constraints, and decide when a task is infeasible. Yet existing agent evaluations often report only end-to-end success, making it difficult to determine whether failures stem from planning or execution. We introduce Agent Planning Benchmark (APB), a planning-specific diagnostic benchmark with 4,209 multimodal cases across 22 domains and five settings, covering holistic planning, feedback-conditioned step-wise planning, and robustness under extraneous tools, broken tools, and unsolvable tasks. Across 12 MLLMs, APB reveals systematic weaknesses in long-horizon planning, tool-noise robustness, calibrated refusal, and inference-time refinement. We further validate APB on 200 ToolSandbox tasks and 200 $\tau^2$-bench tasks, where APB-guided refinement consistently improves plan correctness, plan grade, and downstream execution metrics
    
[^143]: HEART-Bench：大语言模型能否做出与人设一致的决策？

    HEART-Bench: Do LLMs Make Persona-Consistent Decisions?

    [https://arxiv.org/abs/2605.30058](https://arxiv.org/abs/2605.30058)

    提出HEART-Bench基准，通过11个拥有自传体记忆的大五人格合成角色和64个决策场景构成的673道多选题，评估16个大语言模型能否在检索记忆的辅助下做出与人设一致且基于记忆的决策。

    

    交互式娱乐和社交应用中的大语言模型智能体，包括AI伴侣、角色扮演角色和人设克隆体，需要做出基于个性和过往经历的决策。我们提出HEART-Bench来评估基于记忆的人设一致性回答选择能力。该基准将11个以大五人格为框架构建的合成角色（每个角色拥有1,000条自传体记忆）与64个由DIAMONDS框架指导设计的决策场景相结合，构成673道多选题。领域专家独立比较候选回答，由资深专家解决分歧并在必要时修改回答；其他角色对同一场景的裁决后回答被用作干扰项。在主要评估中，模型会获得角色ID和职业、场景以及选项，并在记忆增强设置中额外提供检索到的记忆，而详细的人格与价值观档案则被隐藏。我们在无检索、朴素检索增强等设置下评估了16个大语言模型。

    arXiv:2605.30058v2 Announce Type: replace  Abstract: LLM agents in interactive entertainment and social applications, including companions, role-play characters, and persona clones, need decisions grounded in personality and past experiences. We introduce HEART-Bench to evaluate memory-grounded persona-consistent response selection. The benchmark combines 11 Big Five-anchored synthetic characters, each with 1,000 autobiographical memories, and 64 DIAMONDS-guided decision scenarios into 673 multiple-choice questions. Domain experts independently compare candidate responses, with a senior expert resolving disagreements and revising responses when needed; other characters' adjudicated responses to the same scenario serve as distractors. In the main evaluation, models receive character IDs and occupations, scenarios, and options, with retrieved memories added in memory-augmented settings. Detailed personality-and-value profiles are withheld. We evaluate 16 LLMs under No-Retrieval, Naive-RA
    
[^144]: 计划的表达方式重要吗？关于LLM网页代理计划表示的实证研究

    Does The Way You Plan Matter? An Empirical Study of Planning Representations for LLM Web Agents

    [https://arxiv.org/abs/2605.29927](https://arxiv.org/abs/2605.29927)

    本文提出PlanAhead框架，系统评估了四种自然语言计划表示（顺序子目标、叙述、伪代码、检查清单）对多模态LLM网页代理性能的影响，并实现了无需人工标注的自动任务难度分级。

    

    尽管近期取得了进展，基于LLM的网页代理仍然存在探索能力有限、遗漏关键步骤以及对任务约束敏感等问题。先前的研究表明，这些失败大多源于规划能力的不足，然而不同自然语言计划表示方式的影响仍未被探索。为了解决这一问题，我们提出了PlanAhead，一个静态的规划器-执行器框架，用于评估计划表示对代理性能的影响。我们首先自动将WebArena任务分类为3个难度级别，无需人工标注即可实现一致的难度分级。随后，我们在被归类为困难的任务上，系统地评估了4种不同的计划表示方式：顺序子目标、叙述性描述、伪代码和检查清单；并在不同家族的多模态LLM驱动的代理（OpenAI、阿里巴巴和谷歌）上进行了测试。为了考虑随机变异性，我们引入了一种新颖的代理成功分解方法……

    arXiv:2605.29927v2 Announce Type: replace-cross  Abstract: Despite recent advances, LLM-based web agents still struggle with limited exploration, omission of critical steps, and sensitivity to task constraints. Prior work suggests that many of these failures stem from weaknesses in planning, yet the impact of alternative natural language plan representations remains unexplored. To address this, we introduce PlanAhead, a static planner-executor framework that evaluates the impact of plan representation on agent performance. We first automatically categorize WebArena tasks into 3 difficulty levels, enabling consistent difficulty grading without human annotation. Then we systematically evaluate 4 different plan representations on the tasks categorized as hard: sequential subgoals, narrative, pseudocode, and checklist; across different families of multimodal LLM-powered agents(OpenAI, Alibaba, and Google). To account for stochastic variability, we introduce a novel decomposition of agent S
    
[^145]: PubMedCausal：用于生物医学文本中因果关系抽取的跨度级标注语料库

    PubMedCausal: A Span-Level Annotated Corpus for Causal Relation Extraction in Biomedical Text

    [https://arxiv.org/abs/2605.28363](https://arxiv.org/abs/2605.28363)

    本文构建了基于PubMed摘要的跨度级因果标注语料库PubMedCausal，通过完整的因果-结果跨度、因果类型与句级性标注，实现了对因果检测与全跨度因果抽取更贴近真实文本表达的评估。

    

    因果关系抽取是生物医学文本挖掘的核心任务，但现有资源往往将因果关系与更广泛的关联关系混为一谈、将标注局限于句子级示例，或主要聚焦于显式因果线索。这限制了它们在评估模型能否还原生物医学文本中真实表达的因果论断方面的作用。我们提出了PubMedCausal，一个基于PubMed摘要构建的生物医学因果关系抽取跨度级标注语料库。该语料库包含30,000个段落级条目，其中含3,945个因果条目和6,491个经裁定确认的因果-结果对。每个因果关系均标注了完整的因果跨度、因果类型及句级性，从而支持对因果检测和全跨度因果抽取两项任务的评估。我们在检测和抽取两种设定下对判别式编码器和开源生成式模型进行了基准测试。在因果检测任务上，生物医学编码器……

    arXiv:2605.28363v2 Announce Type: replace  Abstract: Causal relation extraction (CRE) is central to biomedical text mining, but current resources often conflate causal relations with broader associations, restrict annotation to sentence-level examples, or focus mainly on explicit causal cues. This limits their usefulness for evaluating whether models can recover causal claims as they are actually expressed in biomedical text. We introduce PubMedCausal, a span-level annotated corpus for biomedical CRE built from PubMed abstracts. The corpus contains 30,000 paragraph-level rows, including 3,945 causal rows and 6,491 adjudicated cause--effect pairs. Each causal relation is annotated with full-text cause and effect spans, causality type, and sententiality, enabling evaluation of both causal detection and full-span causal extraction. We benchmark discriminative encoders and open-source generative models across detection and extraction settings. For causal detection, biomedical encoders are 
    
[^146]: 超越提问：评估大语言模型的陈述性知识

    Beyond Questions: Evaluating the Stated Knowledge of LLMs

    [https://arxiv.org/abs/2605.26937](https://arxiv.org/abs/2605.26937)

    该论文提出了开放式知识评估这一新范式，并构建了包含10,000个实体的BeQu基准，通过开放式引导提示评估大语言模型自然表达的知识，从而克服了传统预定义问题评测中的可用性偏差。

    

    大语言模型（LLM）中的参数化知识是其成功的基石，但人们对它的理解仍然不足。现有的知识评测基准通常依赖于预定义的问题（例如“M.L. King的出生日期是什么？”），仅评估基准设计者明确选择查询的知识，这是一种有问题的可用性偏差。在本文中，我们提出了开放式知识评估，这是一种用于评测大语言模型陈述性知识的新范式。该范式不再提出狭窄的问题，而是评估模型在面对开放式引导提示（例如“告诉我你所知道的关于M.L. King的一切”）时选择表达出来的知识。这将关注点从预定义答案的检索转向刻画模型自然表达的知识。我们通过BeQu（Beyond Questions，超越提问）基准来实例化这一范式，该基准包含10,000个实体，并配有用于语句验证的参考语料库。使用BeQu，我们……

    arXiv:2605.26937v3 Announce Type: replace-cross  Abstract: Parametric knowledge in large language models (LLMs) is a cornerstone of their success, yet remains poorly understood. Existing knowledge benchmarks typically rely on predefined questions (e.g., "What is the birth date of M.L. King?"), evaluating only knowledge that benchmark designers explicitly choose to query, a problematic availability bias. In this paper, we introduce open knowledge evaluation, a new paradigm for benchmarking the stated knowledge of LLMs. Instead of asking narrow questions, it evaluates models on the knowledge they choose to surface in response to open-ended elicitation prompts (e.g., "Tell me everything you know about M.L. King"). This shifts the focus from predefined answer retrieval toward characterizing the knowledge models naturally express. We instantiate this paradigm with BeQu (Beyond Questions), a benchmark of 10,000 entities paired with reference corpora for statement verification. Using BeQu, we
    
[^147]: 在自然语言上训练的神经后继表示中涌现出词类与构式类结构

    Word-Class and Construction-Like Structure Emerges in Neural Successor Representations Trained on Natural Language

    [https://arxiv.org/abs/2605.24585](https://arxiv.org/abs/2605.24585)

    在自然语言上训练的神经后继表示无需任何词性监督，即可自发涌现出词类边界以及跨越传统类别界限的构式类语言结构。

    

    神经语言模型通常基于下一词元预测进行训练，然而语言结构跨越多个时间尺度。后继表示通过编码对未来状态的折扣分布，将这一预测视野显式化。在本文中，我们探究这种预测性表示是否不仅能恢复词类，还能从自然语言中恢复更精细的功能性与构式类结构。一个在 WikiText-103 上训练的残差网络在三个预测视野上预测后继表示分布，且完全无需词性监督。在最短视野下，无监督聚类能够稳健地恢复出名词、动词和形容词，而聚类之间的有向转移重现了人们熟知的句法不对称性。在更精细的分辨率下以及跨越 13 个词性类别时，同样的几何结构揭示了跨越类别边界的语义-功能分组，以及能够追踪日期、计量和称号-名称等候选共现结构的有向关系。

    arXiv:2605.24585v3 Announce Type: replace  Abstract: Neural language models are typically trained on next-token prediction, although linguistic structure spans multiple temporal scales. Successor representations (SRs) make this horizon explicit by encoding discounted distributions over future states. Here, we ask whether such predictive representations can recover not only word classes, but also finer functional and construction-like structure from natural language. A residual network trained on WikiText-103 predicts SR distributions at three horizons without part-of-speech supervision. At the shortest horizon, unsupervised clustering robustly recovers nouns, verbs, and adjectives, while directed inter-cluster transitions reproduce familiar syntactic asymmetries. At finer resolutions and across 13 part-of-speech categories, the same geometry reveals semantic-functional groupings that cross category boundaries and directed relations tracing candidate date, measurement, and title-name co
    
[^148]: 强大的教师并非必需？论大语言模型预训练中的知识蒸馏

    Strong Teacher Not Needed? On Distillation in LLM Pretraining

    [https://arxiv.org/abs/2605.23857](https://arxiv.org/abs/2605.23857)

    大语言模型预训练中的知识蒸馏并不需要强大的教师——即使是小型或训练不足的教师，在恰当混合两种损失后也能有效提升更大规模学生模型的性能，而过强的教师反而可能使蒸馏收益饱和或逆转，且蒸馏更利于提升泛化能力而非域内拟合。

    

    知识蒸馏通常假设一种“强到弱”的关系，即更强的教师能培养出更好的学生。在这项工作中，我们检验了这一假设在大语言模型预训练蒸馏中的适用性。通过改变模型架构规模和训练token预算，我们构建了强到弱、同级以及弱到强的师生关系，并研究了蒸馏在每种情况下的有效性。我们发现，教师并不需要很强大：通过恰当混合语言建模损失与知识蒸馏损失，即使是小型且训练不足的教师也能提升更大规模学生模型的表现。与此同时，更强的教师并不总是更好：通过增加参数量或训练token数来进一步增强教师，可能使蒸馏收益趋于饱和，甚至出现逆转。我们还进一步观察到，蒸馏更倾向于提升模型的泛化能力（分布外性能和下游任务表现），而非域内拟合能力。

    arXiv:2605.23857v2 Announce Type: replace-cross  Abstract: Knowledge distillation generally assumes a strong-to-weak relationship where stronger teachers yield better students. In this work, we examine this assumption about distillation in large language model pretraining. By varying architecture sizes and training token budgets, we create strong-to-weak, same-level, and weak-to-strong teacher-student relationships, and study distillation's effectiveness under each. We find that the teacher need not be strong: with proper mixing of the language modeling and knowledge distillation losses, even small and undertrained teachers improve larger students. At the same time, a stronger teacher is not always better: pushing the teacher further, through more parameters or more training tokens, can saturate or even reverse the distillation gains. We further observe that distillation improves generalization (out-of-distribution and downstream performance) more readily than in-domain fitting. Togeth
    
[^149]: 面向修辞角色标注中难例的推理时语义重排序

    Semantic Reranking at Inference Time for Hard Examples in Rhetorical Role Labeling

    [https://arxiv.org/abs/2605.18007](https://arxiv.org/abs/2605.18007)

    提出RISE推理时语义重排序框架，利用对比学习得到的标签表示自动识别并重排序低置信度难例的预测，在八个修辞角色标注数据集上平均提升9.15个macro-F1点，且无需重新训练模型。

    

    修辞角色标注为文档中的每个句子分配功能角色，广泛应用于法律、医疗和科学领域。虽然语言模型在平均性能上表现强劲，但在预测置信度较低的难例上仍不可靠。现有方法通常隐式地处理不确定性，并将标签视为离散标识符，忽略了标签名称中蕴含的语义信息。我们提出了RISE，这是一种推理时语义重排序框架，利用标签语义来改进对难例的预测。RISE自动识别低置信度预测，并使用对比学习得到的标签表示对模型输出进行重排序，无需重新训练或修改底层模型。在八个特定领域的RRL数据集上使用七个语言模型（包括基于编码器和因果架构的模型）进行的实验表明，在难例上平均提升了9.15个macro-F1点。

    arXiv:2605.18007v2 Announce Type: replace  Abstract: Rhetorical Role Labeling (RRL) assigns a functional role to each sentence in a document and is widely used in legal, medical, and scientific domains. While language models (LMs) achieve strong average performance, they remain unreliable on hard examples, where prediction confidence is low. Existing approaches typically handle uncertainty implicitly and treat labels as discrete identifiers, overlooking the semantic information encoded in label names. We introduce RISE, an inference-time semantic reranking framework that leverages label semantics to refine predictions on hard instances. RISE automatically identifies low-confidence predictions and reranks model outputs using contrastively learned label representations, without retraining or modifying the underlying model. Experiments on eight domain-specific RRL datasets with seven LMs, including encoder-based and causal architectures, show an average gain of +9.15 macro-F1 points on ha
    
[^150]: 学习的几何学：训练过程中的读写不对称性

    The Geometry of Learning: Read-Write Asymmetry Across Training

    [https://arxiv.org/abs/2605.16600](https://arxiv.org/abs/2605.16600)

    该论文以残差流为共享坐标系分析预训练与对齐阶段对 transformer 权重的更新方向，揭示了“读写不对称性”：读侧矩阵（query/key）的更新集中于其输入激活的主子空间，而写侧矩阵（输出矩阵）的更新相对于预测子空间接近随机，且这一特征是训练阶段的属性而非训练目标的属性。

    

    预训练和对齐更新的是同一个 transformer 的权重，我们探究每个阶段分别在哪些方向上改变了这些权重。残差流为每个矩阵提供了一个共享的坐标系，在这一坐标系中，权重的变化既可以与该矩阵所读取的激活值的主子空间进行比较，也可以与 unembedding 的预测子空间进行比较。在这一框架下，对齐阶段对 query 和 key 矩阵的更新主要富集于主输入子空间中，而其对输出矩阵的更新相对于预测子空间则接近随机；同样的特征也出现在预训练后期，以及我们对同一检查点施加的每一种训练目标之下，因此这是训练阶段的属性，而非训练目标的属性。线性映射的梯度是外积 $\delta_t a_t^\top$ 之和，残差流对于从中读取信息的矩阵通过激活值 $a_t$ 进入该乘积，而对于（写向它的矩阵）则通过上游梯度……（原文摘要在此处截断）

    arXiv:2605.16600v2 Announce Type: replace-cross  Abstract: Pretraining and alignment update the same transformer weights, and we ask in which directions each stage changes them. The residual stream gives every matrix a shared coordinate system, in which a weight change can be compared with the principal subspace of the activations the matrix reads and with the prediction subspace of the unembedding. In this frame, alignment updates to the query and key matrices are enriched in the principal input subspace, while its updates to the output matrices are close to random relative to the prediction subspace, and the same signature appears in late pretraining and under every objective we apply to one checkpoint, so it is a property of the training stage rather than of the objective. The gradient of a linear map is a sum of outer products $\delta_t a_t^\top$, and the residual stream enters this product through the activation $a_t$ for matrices that read from it and through the upstream gradien
    
[^151]: 通过误差控制动力学重新思考循环模型中的状态追踪

    Rethinking State Tracking in Recurrent Models Through Error Control Dynamics

    [https://arxiv.org/abs/2605.07755](https://arxiv.org/abs/2605.07755)

    本文证明了仿射循环网络（如状态空间模型和线性注意力）一旦保留状态表示就无法沿状态分离子空间纠正误差，因而其状态追踪只能依靠累积误差下的有限视界解，而非鲁棒的状态追踪。

    

    循环架构中状态追踪的理论研究主要集中于表达能力：即一个固定架构在理论上能否实现一组符号转换规则。我们认为同样重要的是误差控制，即支配隐状态沿区分符号状态的方向发生漂移的动力学。我们证明，仿射循环网络——一类涵盖状态空间模型和线性注意力的模型——在保留状态表示的同时，无法沿状态分离子空间纠正误差。因此，实用的仿射追踪器并不能学习到鲁棒的状态追踪；相反，它们学习的是由累积的状态相关误差所支配的有限视界解。我们刻画了这种失效的机制，表明只有当累积的类内散度相对于类间分离保持较小时，追踪才保持可读性。我们还在……上进行了实证验证（摘要在此处截断）。

    arXiv:2605.07755v2 Announce Type: replace-cross  Abstract: The theory of state tracking in recurrent architectures has predominantly focused on expressive capacity: whether a fixed architecture can theoretically realize a set of symbolic transition rules. We argue that equally important is error control, the dynamics governing hidden-state drift along the directions that distinguish symbolic states. We prove that affine recurrent networks, a class of models encompassing State-Space Models and Linear Attention, cannot correct errors along state-separating subspaces once they preserve state representations. Consequently, practical affine trackers do not learn robust state tracking; rather, they learn finite-horizon solutions governed by accumulated state-relevant error. We characterize the mechanics of this failure, showing that tracking remains readable only while the accumulating within-class spread remains small relative to the between-class separation. We demonstrate empirically on g
    
[^152]: ShredBench：评估多模态大语言模型在文档重建中的语义推理能力

    ShredBench: Evaluating the Semantic Reasoning Capabilities of Multimodal LLMs in Document Reconstruction

    [https://arxiv.org/abs/2604.23813](https://arxiv.org/abs/2604.23813)

    ShredBench是一个由自动化流程直接从Markdown生成碎片化文档的新基准，用于系统评估多模态大语言模型在碎片文档重建任务中结合视觉识别与语义推理的能力，并可通过引入新文本源防止训练数据污染。

    

    多模态大语言模型（MLLMs）在视觉富文本文档理解（VRDU）任务中取得了显著的性能，但其能力主要是在原始完整、结构良好的文档图像上进行评估的。我们考虑从碎片中恢复内容这一具有挑战性的VRDU场景，该场景要求在内容严重不连续的情况下，将视觉模式识别与语义推理相结合。为了促进对复杂VRDU任务的系统性评估，我们提出了ShredBench，这是一个由自动化生成流程支持的基准，该流程可直接从Markdown渲染生成碎片化文档。所提出的流程通过允许灵活集成最新或未见过的文本源来防止训练数据污染，从而确保评估的有效性。ShredBench评估四种场景（英文、中文、代码、表格）以及三种碎片粒度（8、12、16片）。

    arXiv:2604.23813v2 Announce Type: replace-cross  Abstract: Multimodal Large Language Models (MLLMs) have achieved remarkable performance in Visually Rich Document Understanding (VRDU) tasks, but their capabilities are mainly evaluated on pristine, well-structured document images. We consider content restoration from shredded fragments, a challenging VRDU setting that requires integrating visual pattern recognition with semantic reasoning under significant content discontinuities. To facilitate systematic evaluation of complex VRDU tasks, we introduce ShredBench, a benchmark supported by an automated generation pipeline that renders fragmented documents directly from Markdown. The proposed pipeline ensures evaluation validity by allowing the flexible integration of latest or unseen textual sources to prevent training data contamination. ShredBench assesses four scenarios (English, Chinese, Code, Table) with three fragmentation granularities (8, 12, 16 pieces). Empirical evaluations on s
    
[^153]: 评分规则对大语言模型评估的敏感性：来自印度金融监管文本的证据

    Scoring-Rule Sensitivity in LLM Evaluation: Evidence from Indian Financial Regulatory Text

    [https://arxiv.org/abs/2604.19298](https://arxiv.org/abs/2604.19298)

    该研究构建了基于印度SEBI和RBI监管文本的专家标注基准IndiaFinBench，发现大语言模型的排名对评分规则高度敏感——严格字符串匹配与跨模型裁判两种评分机制之间甚至无显著的正向排名相关性——表明单一准确率数字是模型与评分规则的共同产物，不能可靠反映模型真实能力。

    

    一项基准排行榜报告的是单一的准确率数字，但这个数字是模型与评分规则的共同产物。我们探究当仅改变评分规则时会发生什么，并发现模型的排名对评分规则高度敏感。IndiaFinBench 是一个包含406个条目、由专家标注的基准，基于SEBI（印度证券交易委员会）和RBI（印度储备银行）的监管文本，涵盖监管解读、数值推理、矛盾检测和时间推理。我们在两种主要评分机制下对十二个大语言模型进行零样本评分——严格的四阶段字符串匹配，以及与任何被评估系统无共享模型家族的跨模型裁判——外加一种操作性复合机制。两种主要机制之间没有正向的排名对应关系（Spearman rho = -0.273，p = 0.39，不显著），且没有任何模型能在所有三种机制下保持其排名。严格准确率范围为75.12%–89.66%；仅裁判评分的准确率范围为87.19%–94.83%，且出现了不同的领先者

    arXiv:2604.19298v3 Announce Type: replace-cross  Abstract: A benchmark leaderboard reports one accuracy number, but that number is a joint product of a model and a scoring rule. We ask what changes when only the rule changes, and find that model ordering is highly sensitive to it. IndiaFinBench is a 406-item expert-annotated benchmark over SEBI and RBI regulatory text covering regulatory interpretation, numerical reasoning, contradiction detection, and temporal reasoning. We score twelve LLMs zero-shot under two primary regimes over the same predictions - strict four-stage string matching, and the verdicts of a cross-model judge sharing no model family with any evaluated system - plus one operational composite regime. The two primary regimes show no positive rank correspondence (Spearman rho = -0.273, p = 0.39, not significant), and no model holds its rank across all three regimes. Strict accuracy spans 75.12-89.66%; judge-only accuracy spans 87.19-94.83% with a different leader and tr
    
[^154]: 面向检索增强生成的潜在抽象方法

    Latent Abstraction for Retrieval-Augmented Generation

    [https://arxiv.org/abs/2604.17866](https://arxiv.org/abs/2604.17866)

    该论文提出LAnR统一框架，让单个LLM通过指定[PRED]标记的隐藏状态直接生成密集检索向量，在其自身潜在空间内联合完成编码、检索与生成，从而摆脱了传统RAG中生成文本查询及检索器与生成器分离的架构限制。

    

    检索增强生成（RAG）已成为利用外部知识增强大型语言模型（LLM）、缓解幻觉并提升事实性的标准方法。然而，现有系统依赖于在每一跳生成自然语言查询，并在检索器与生成器之间维持严格的架构分离，这阻碍了其充分利用LLM的完整表征能力。我们提出了LAnR（Latent Abstraction for RAG，面向RAG的潜在抽象），这是一个统一框架，其中单个LLM完全在其自身的潜在空间内联合执行编码、检索和生成。LAnR不再生成文本查询，而是从指定的[PRED]标记的隐藏状态中产生密集检索向量，并将其与来自同一模型编码的文档表示进行匹配。此外，LAnR使用轻量级的……（自适应地判断何时已检索到足够的证据）。

    arXiv:2604.17866v4 Announce Type: replace-cross  Abstract: Retrieval-Augmented Generation (RAG) has become a standard approach for enhancing large language models (LLMs) with external knowledge, mitigating hallucinations, and improving factuality. However, existing systems rely on generating natural language queries at each hop and maintaining a strict architectural separation between retriever and generator, preventing them from leveraging the full representational capacity of the LLM. We propose \textbf{LAnR} (Latent Abstraction for RAG), a unified framework in which a single LLM jointly performs encoding, retrieval, and generation entirely within its own latent space. Rather than generating textual queries, LAnR produces dense retrieval vectors from the hidden states of a designated \texttt{[PRED]} token and uses them to match against encoded document representations from the same model. Furthermore, LAnR adaptively decides when sufficient evidence has been retrieved using a lightwe
    
[^155]: 减少临床记录冗余以实现大语言模型的高效应用

    Clinical Note Bloat Reduction for Efficient LLM Use

    [https://arxiv.org/abs/2604.16364](https://arxiv.org/abs/2604.16364)

    TRACE 方法利用 EHR 元数据（或基于频率的去重）识别并清除临床记录中模板化和复制粘贴的冗余内容，在 530 万份记录和多项信息抽取及预测任务上验证了其能在保留真实临床信号的同时大幅压缩文本量，从而降低大语言模型的使用成本并提升效率。

    

    背景：临床记录中包含大量来自模板、复制粘贴和自动填充字段的重复文本（即“记录冗余/note bloat”），这稀释了临床信号，限制了纵向上下文的利用，并增加了大语言模型（LLM）的使用成本。方法：TRACE 利用笔记级别的电子健康记录（EHR）元数据来识别模板化和复制的内容，从而去除临床记录中的冗余；当元数据不可用时，则采用基于频率的去重方法。我们通过盲法医生片段审查以及金标准模板文本标注对 TRACE 进行了评估，评估覆盖多个医疗系统的四个队列，包括肝移植、产科和住院人群（共 530 万份记录）。我们比较了在原始记录和 TRACE 处理后记录上运行的零样本大语言模型和基于嵌入的分类器，任务涵盖 20 个信息抽取任务，以及 5 年生存期、产后出血和 30 天再入院的预测。结果：在被删除的文本中，仅有 0.3–6.6% 被标记为作者本人撰写的内容；TRACE 捕获……

    arXiv:2604.16364v2 Announce Type: replace-cross  Abstract: Background: Clinical notes contain extensive duplicated text from templates, copy-paste, and auto-populated fields ("note bloat"), diluting clinical signal, limiting longitudinal context, and increasing large language model (LLM) costs.   Methods: TRACE removes note bloat using note-level EHR metadata to identify templated and copied content, with frequency-based de-duplication when metadata are unavailable. We evaluated TRACE using blinded physician span review and gold-standard templated-text annotations across four cohorts spanning liver transplant, obstetrics, and inpatient populations at multiple health systems (5.3M notes). We compared zero-shot LLMs and embedding-based classifiers using original and TRACE-processed notes for 20 information extraction tasks and prediction of 5-year survival, postpartum hemorrhage, and 30-day readmission.   Results: Only 0.3-6.6% of removed text was flagged as author-generated; TRACE captu
    
[^156]: 谁会被标记？AI内容水印中的多元化评估缺口

    Who Gets Flagged? The Pluralistic Evaluation Gap in AI Content Watermarking

    [https://arxiv.org/abs/2604.13776](https://arxiv.org/abs/2604.13776)

    该论文揭示了AI内容水印技术因依赖内容统计特性，可能对不同语言、文化和人群产生系统性检测偏见，并提出了包含跨语言检测公平性、文化多样化内容覆盖和人口统计分组检测指标三个维度的多元化水印评估框架。

    

    水印技术正在成为AI内容认证的默认机制，各项治理政策和框架将其作为内容溯源的基础设施。然而，在文本、图像和音频等各种模态中，水印信号的强度、可检测性和鲁棒性取决于内容本身的统计特性，而这些特性在语言、文化视觉传统和人口群体之间存在着系统性的差异。我们研究了这种内容依赖性如何产生特定于模态的偏见路径。通过回顾各模态的主要水印基准测试，我们发现除一个例外，没有任何基准测试报告了跨语言、文化内容类型或人群群体的性能表现。为解决这一问题，我们提出了多元化水印基准测试的三个具体评估维度：跨语言检测公平性、文化多样化内容覆盖以及检测指标的人口统计分解。我们认为……

    arXiv:2604.13776v3 Announce Type: replace-cross  Abstract: Watermarking is becoming the default mechanism for AI content authentication, with governance policies and frameworks referencing it as infrastructure for content provenance. Yet across text, image, and audio modalities, watermark signal strength, detectability, and robustness depend on statistical properties of the content itself, properties that vary systematically across languages, cultural visual traditions, and demographic groups. We examine how this content dependence creates modality-specific pathways to bias. Reviewing the major watermarking benchmarks across modalities, we find that, with one exception, none report performance across languages, cultural content types, or population groups. To address this, we propose three concrete evaluation dimensions for pluralistic watermark benchmarking: cross-lingual detection parity, culturally diverse content coverage, and demographic disaggregation of detection metrics. We arg
    
[^157]: 缺失模态下假新闻检测的模态头解耦与调控

    Decoupling and Regulating Modality Heads for Fake News Detection under Missing Modality

    [https://arxiv.org/abs/2604.09711](https://arxiv.org/abs/2604.09711)

    提出HMDR框架，通过对多模态大语言模型中的注意力头进行模态解耦、贡献调控与模态不变性学习，解决缺失模态场景下假新闻检测的问题，尤其应对复用真实材料导致标签失效的挑战。

    

    多模态新闻在跨网络平台传播过程中常常丢失图像或文本。现有方法将此视为信息损失，并假设原始的真实性标签仍然有效。然而，这一假设对于复用真实材料的假新闻并不成立，因为其剩余部分可能是真实的。因此，检测器应当解耦各模态线索，并根据模态可用性调整其贡献。然而，粗粒度的新闻标签难以为模态解耦提供有效指导，而在完整输入上的训练又会引入模态偏差。为应对这些挑战，我们提出了面向多模态大语言模型的头级模态解耦与调控方法。首先，模态头解耦通过基于注意力偏好的软性头分组来解耦模态表示；其次，模态头调控通过对不平衡的模态注意力进行补偿来调控各头的贡献；最后，模态不变性学习促进……

    arXiv:2604.09711v2 Announce Type: replace-cross  Abstract: Multimodal news often loses its image or text as it spreads across Web platforms. Existing methods treat this as information loss, assuming that the original veracity label remains valid. This assumption fails for fake news reusing authentic material, whose remainder may be genuine. Detectors should therefore decouple modality cues and adapt their contributions to modality availability. However, coarse news labels provide little guidance for modality decoupling, while training on complete inputs can induce modality bias. To address these challenges, we propose Head-wise Modality Decoupling and Regulation (HMDR) for multimodal large language models. First, Modality Head Decoupling decouples modality representations through soft head grouping by attention preference. Second, Modality Head Regulation regulates head contributions through compensation for imbalanced modality attention. Finally, Modality Invariance Learning promotes 
    
[^158]: 通过宽松的在线策略蒸馏高效扩展推理能力

    Scaling Reasoning Efficiently via Relaxed On-Policy Distillation

    [https://arxiv.org/abs/2603.11137](https://arxiv.org/abs/2603.11137)

    提出REOPOLD框架，通过控制教师反馈的应用位置、影响强度和分配规则的变化时机，解决了在线策略蒸馏中统一信用分配导致的熵坍缩与负迁移问题，从而高效提升容量受限模型的推理能力。

    

    在线策略蒸馏（OPD）是一种将推理能力迁移到容量受限模型的有前景的方法，但基于采样token的OPD常常遭受熵坍缩和负迁移问题。我们将统一的信用分配识别为一个关键瓶颈：现有目标函数在学生生成的轨迹上获得密集的教师反馈，但尽管token级别的信号高度异构，却在所有采样token上统一应用该反馈。大多数token带有冗余的近零信用，而罕见的重尾负信用可能主导更新过程，并过早地抑制合理的推理轨迹。我们提出了REOPOLD（宽松在线策略蒸馏），这是一个通过控制教师反馈应用的位置、影响更新的强度以及训练期间分配规则何时改变，来放宽这种统一分配的框架。在多种推理任务上，REOPOLD改进了基于采样token的OPD相比近期后训练方法的表现。

    arXiv:2603.11137v2 Announce Type: replace-cross  Abstract: On-policy distillation (OPD) is a promising approach for transferring reasoning capabilities to capacity-constrained models, yet sampled-token OPD often suffers from entropy collapse and negative transfer. We identify uniform credit allocation as a key bottleneck: existing objectives obtain dense teacher feedback on student-generated trajectories, but apply it uniformly across sampled tokens despite highly heterogeneous token-level signals. Most tokens carry redundant near-zero credit, while rare heavy-tailed negative credits can dominate updates and prematurely suppress plausible reasoning trajectories. We introduce REOPOLD (Relaxed On-Policy Distillation), a framework that relaxes this uniform assignment by controlling where teacher feedback is applied, how strongly it affects the update, and when the allocation rule changes during training. Across diverse reasoning tasks, REOPOLD improves sampled-token OPD over recent post-t
    
[^159]: 通过下一序列预测强化快速权重

    Reinforced Fast Weights via Next-Sequence Prediction

    [https://arxiv.org/abs/2602.16704](https://arxiv.org/abs/2602.16704)

    本文提出下一序列预测（NSP）目标及ReFINE方法，利用强化学习在序列级别优化快速权重模型，解决了传统下一词元预测目标与循环记忆状态之间的训练不匹配问题。

    

    快速权重（Fast Weight）架构通过用循环更新的固定大小记忆取代KV缓存，为长上下文建模提供了标准Transformer的一种有前景的替代方案。尽管架构发生了这一转变，它们通常仍采用与标准Transformer相同的下一词元预测（NTP）目标进行训练。这造成了不匹配：快速权重模型依赖不断演化的记忆状态来支持未来的预测，而NTP仅为紧接着的下一个词元提供词元级别的监督。我们通过下一序列预测（NSP）来解决这一不匹配问题，NSP是NTP在序列级别的扩展，它训练模型从其循环记忆状态中生成连贯的多词Token延续。为了优化这一序列级目标，我们提出了ReFINE（通过下一序列预测强化快速权重）。ReFINE基于预测熵选择信息量大的词元位置，生成多词元展开序列，并分配序列级奖励……（摘要原文在此处截断）

    arXiv:2602.16704v2 Announce Type: replace  Abstract: Fast weight architectures offer a promising alternative to standard transformers for long-context modeling by replacing the KV-cache with a recurrently updated fixed-size memory. Despite this architectural shift, they are typically trained with the same next-token prediction (NTP) objective as standard transformers. This creates a mismatch: fast weight models rely on an evolving memory state to support future predictions, while NTP provides only token-level supervision for the immediate next token. We address this mismatch with next-sequence prediction (NSP), a sequence-level extension of NTP that trains models to produce coherent multi-token continuations from their recurrent memory state. To optimize this sequence-level objective, we propose ReFINE (Reinforced Fast weIghts via Next sEquence prediction). ReFINE selects informative token positions based on prediction entropy, generates multi-token rollouts, assigns sequence-level rew
    
[^160]: 涓滴皆有益：基于细粒度可迁移多模态词元构建知识图谱基础模型

    Every Little Helps: Building Knowledge Graph Foundation Model with Fine-grained Transferable Multi-modal Tokens

    [https://arxiv.org/abs/2602.15896](https://arxiv.org/abs/2602.15896)

    提出基于词元的知识图谱基础模型TOFU，将结构、视觉和文本信息离散化为模态特定的细粒度可迁移词元，并通过消息混合的分层融合架构处理这些词元，在17个直推式、归纳式和完全归纳式多模态知识图谱上实现了强大的跨图谱泛化能力。

    

    多模态知识图谱推理（MMKGR）旨在同时利用图结构信息和多模态实体内容来预测缺失的链接。大多数现有工作是为直推式设置设计的，它们学习特定于数据集的嵌入，难以泛化到新的知识图谱。近期的知识图谱基础模型（KGFMs）改善了跨知识图谱的迁移能力，但它们主要利用结构模式，而忽视了丰富的多模态信号。我们通过提出一种面向多模态知识图谱推理的基于词元的基础模型（TOFU）来弥补这些不足，该模型在不同的多模态知识图谱上展现出强大的泛化能力。TOFU 将结构、视觉和文本信息离散化为特定于模态的词元，随后采用带有消息混合机制的分层融合架构来处理这些词元，从而获得可用于多模态知识图谱推理的可迁移特征。在 17 个直推式、归纳式和完全归纳式多模态知识图谱上的实验结果表明……

    arXiv:2602.15896v2 Announce Type: replace  Abstract: Multi-modal knowledge graph reasoning (MMKGR) aims to predict the missing links by exploiting both graph structure information and multi-modal entity contents. Most existing works are designed for a transductive setting, which learns dataset-specific embeddings and struggles to generalize to new KGs. Recent knowledge graph foundation models (KGFMs) improve cross-KG transfer, but they mainly exploit structural patterns and ignore rich multi-modal signals. We address these gaps by proposing a token-based foundation model (TOFU) for MMKGR, which exhibits strong generalization across different MMKGs. TOFU discretizes structural, visual, and textual information into modality-specific tokens. TOFU then employs a hierarchical fusion architecture with mixture-of-message mechanisms, aiming to process these tokens and obtain transferable features for MMKGR. Experimental results on 17 transductive, inductive, and fully-inductive MMKGs show that
    
[^161]: 评估大型语言模型对21个西班牙语国家地理词汇知识的掌握

    Evaluating Geographic Lexical Knowledge in Large Language Models Across 21 Spanish-Speaking Countries

    [https://arxiv.org/abs/2602.09346](https://arxiv.org/abs/2602.09346)

    该研究将十个LLM视为虚拟发音人，基于涵盖21个西班牙语国家934个词汇的专家数据库，系统评估了LLM对西班牙语地理词汇差异的知识，发现模型间、国家间和题型间均存在显著差异，其中西班牙得分最高而智利在选择题格式中得分最低。

    

    大型语言模型（LLM）日益服务于使用不同地区语言变体的用户，但其对地理词汇差异的知识仍缺乏充分评估。我们利用一个由专家精心整理的数据库，研究了西班牙语中依国家而异的词汇使用判断，该数据库涵盖21个西班牙语国家的934个词汇条目。我们将十个专有及开放权重的LLM视为虚拟发音合作人，评估了两种互补的问答格式：是非题格式（YNQF）和多项选择题格式（MCQF）。本次评估包含10,164个YNQF问题和16,637个MCQF问题，分别使用F1值和调整后的Jaccard系数进行评分。结果显示，不同模型、国家和题型之间存在显著差异。在YNQF中，西班牙在八个模型中获得最高的国家级评分；在MCQF中，西班牙在九个模型中得分最高；而智利在所有模型的MCQF中得分均最低。回归分析表明，数字资源量……

    arXiv:2602.09346v2 Announce Type: replace  Abstract: Large language models (LLMs) increasingly serve users across regional varieties, yet their knowledge of geographic lexical variation remains insufficiently evaluated. We investigate country-conditioned lexical usage judgments in Spanish using an expert-curated database covering 934 lexical items across 21 Spanish-speaking countries. Treating ten proprietary and open-weight LLMs as virtual informants, we evaluate two complementary question formats: Yes-No question format (YNQF) and multiple-choice question format (MCQF). The evaluation comprises 10,164 YNQF questions and 16,637 MCQF questions, assessed using F1 and an adjusted Jaccard coefficient, respectively. Results reveal substantial differences across models, countries, and formats. Spain receives the highest country-level score for eight models in YNQF and nine in MCQF; Chile receives the lowest MCQF score for every model. Regression analyses suggest that digital resource volume
    
[^162]: OdysseyArena：面向长时程、主动与归纳式交互的大语言模型基准测试

    OdysseyArena: Benchmarking Large Language Models For Long-Horizon, Active and Inductive Interactions

    [https://arxiv.org/abs/2602.05843](https://arxiv.org/abs/2602.05843)

    提出OdysseyArena基准，通过形式化四个基元将智能体评估重新聚焦于长时程、主动和归纳式交互，并建立包含120个任务的OdysseyArena-Lite来衡量智能体从经验中自主发现潜在转换规律的能力。

    

    大语言模型的发展催生了能够在复杂环境中自主导航的智能体。然而，现有的评估主要采用演绎范式，即智能体基于显式给定的规则和静态目标来执行任务，且规划时域往往有限。至关重要的是，这忽略了智能体需要自主地从经验中发现潜在转换规律的归纳必要性，而这是实现智能体前瞻能力并维持策略连贯性的基石。为了弥合这一差距，我们提出了OdysseyArena，将智能体评估重新聚焦于长时程、主动和归纳式交互。我们形式化并实例化了四个基元，将抽象的转换动态转化为具体的交互环境。在此基础上，我们建立了用于标准化基准测试的OdysseyArena-Lite，提供了一组120个任务来衡量智能体的归纳（摘要在此处截断）。

    arXiv:2602.05843v3 Announce Type: replace  Abstract: The advancement of Large Language Models (LLMs) has catalyzed the development of autonomous agents capable of navigating complex environments. However, existing evaluations primarily adopt a deductive paradigm, where agents execute tasks based on explicitly provided rules and static goals, often within limited planning horizons. Crucially, this neglects the inductive necessity for agents to discover latent transition laws from experience autonomously, which is the cornerstone for enabling agentic foresight and sustaining strategic coherence. To bridge this gap, we introduce OdysseyArena, which re-centers agent evaluation on long-horizon, active, and inductive interactions. We formalize and instantiate four primitives, translating abstract transition dynamics into concrete interactive environments. Building upon this, we establish OdysseyArena-Lite for standardized benchmarking, providing a set of 120 tasks to measure an agent's induc
    
[^163]: AzeroS：通过自生成无指令微调将大语言模型扩展至语音领域

    AzeroS: Extending LLM to Speech with Self-Generated Instruction-Free Tuning

    [https://arxiv.org/abs/2601.06086](https://arxiv.org/abs/2601.06086)

    提出SIFT（自生成无指令微调）新范式，由冻结的LLM以语音的文本表示为输入生成监督信号来训练语音LLM，从而无需收集特定任务问答数据即可在理论上实现对未见任务的最佳泛化。

    

    将大语言模型（LLM）扩展至语音领域近来受到广泛关注。一种典型方法是通过投影模块将预训练的LLM与音频编码器连接，并在大规模、特定任务的指令微调数据集上训练所得模型。然而，为特定需求整理此类指令微调数据非常耗时，且以这种方式训练的模型对未见任务的泛化能力往往较差。在这项工作中，我们首先提出了一个论断：当语音LLM使用自生成无指令微调进行训练时，可实现最强的泛化能力，其中监督信号由冻结的LLM以语音的文本表示作为输入来生成。我们提出的SIFT范式消除了收集特定任务问答对的需求，并在理论上实现了对未见任务的最佳泛化。基于这一范式，我们引入了AZeroS（音频……（摘要原文在此处截断）

    arXiv:2601.06086v2 Announce Type: replace  Abstract: Extending large language models (LLMs) to the speech domain has recently gained significant attention. A typical approach connects a pretrained LLM with an audio encoder through a projection module and trains the resulting model on large-scale, task-specific instruction-tuning datasets. However, curating such instruction-tuning data for specific requirements is time-consuming, and models trained in this manner often generalize poorly to unseen tasks. In this work, we first formulate that the strongest generalization of a speech-LLM is achieved when it is trained with Self-Generated Instruction-Free Tuning (SIFT), in which supervision signals are generated by a frozen LLM using textual representations of speech as input. Our proposed SIFT paradigm eliminates the need for collecting task-specific question-answer pairs and yields the theoretically best generalization to unseen tasks. Building upon this paradigm, we introduce AZeroS (Aud
    
[^164]: SkipCat：通过共享投影与块跳过实现大语言模型的秩最大化低秩压缩

    SkipCat: Rank-Maximized Low-Rank Compression of Large Language Models via Shared Projection and Block Skipping

    [https://arxiv.org/abs/2512.13494](https://arxiv.org/abs/2512.13494)

    SkipCat是一种新颖的大语言模型低秩压缩框架，通过共享投影和块跳过机制，在相同压缩率下保留更高的秩，从而避免了传统低秩压缩中因秩过度削减而导致的显著性能下降。

    

    大语言模型（LLM）在广泛的任务中取得了卓越的性能。然而，其庞大的参数规模为其在计算和内存资源有限的边缘设备上的部署带来了重大挑战。低秩压缩是解决这一问题的一种有前景的方法，因为它能够同时降低计算和内存成本，使大语言模型更适合资源受限的环境。然而，朴素的低秩压缩方法需要大幅削减所保留的秩才能实现有意义的内存和计算节省。对于一个低秩模型，秩需要减少一半以上才能带来效率提升。然而，如此激进的截断通常会导致显著的性能下降。为了解决这一权衡问题，我们提出了SkipCat，一种新颖的低秩压缩框架，它能够在实现相同压缩率的同时使用更高的秩。

    arXiv:2512.13494v2 Announce Type: replace-cross  Abstract: Large language models (LLM) have achieved remarkable performance across a wide range of tasks. However, their substantial parameter sizes pose significant challenges for deployment on edge devices with limited computational and memory resources. Low-rank compression is a promising approach to address this issue, as it reduces both computational and memory costs, making LLM more suitable for resource-constrained environments. Nonetheless, na\"ive low-rank compression methods require a significant reduction in the retained rank to achieve meaningful memory and computation savings. For a low-rank model, the ranks need to be reduced by more than half to yield efficiency gains. Such aggressive truncation, however, typically results in substantial performance degradation. To address this trade-off, we propose SkipCat, a novel low-rank compression framework that enables the use of higher ranks while achieving the same compression rate
    
[^165]: 计数线索：轻量级概率基线可以匹敌大语言模型

    Counting Clues: A Lightweight Probabilistic Baseline Can Match an LLM

    [https://arxiv.org/abs/2512.12868](https://arxiv.org/abs/2512.12868)

    本研究提出基于频率的概率排序器（FBPR），仅利用朴素贝叶斯与大语料库中的概念-诊断共现统计，即可在MedQA诊断任务上达到与LLM相当的性能，且两者互补性强，混合使用可获得更佳效果。

    

    大型语言模型（LLM）在多选临床诊断基准测试中表现出色，但目前尚不清楚这种表现在多大程度上反映了其底层的概率推理能力。我们通过MedQA中要求给出最可能诊断的问题来研究这一问题。我们提出了基于频率的概率排序器（FBPR），这是一种轻量级方法，利用从大型语料库中提取的概念-诊断共现统计信息，通过平滑朴素贝叶斯对选项进行评分。当共现统计信息来源于Dolma（OLMo的预训练语料库）和RedPajama（LLaMA预训练数据配方的开源复现）时，FBPR取得了与相应LLM相当的性能。此外，直接的LLM推理和FBPR展现出互补的优势，两者正确回答的问题子集在很大程度上不同，其重叠率仅略高于随机水平。通过简单的混合方法将两者结合，可以取得显著优于任何单一方法的性能提升。

    arXiv:2512.12868v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) excel on multiple-choice clinical diagnosis benchmarks, yet it is unclear how much of this performance reflects underlying probabilistic reasoning. We study this through questions from MedQA that ask for the most likely diagnosis. We introduce Frequency-Based Probabilistic Ranker (FBPR), a lightweight method that scores options with a smoothed Naive Bayes over concept-diagnosis co-occurrence statistics from a large corpus. When co-occurrence statistics are sourced from Dolma, the pretraining corpus for OLMo, and RedPajama, an open reproduction of the LLaMA pretraining-data recipe, FBPR achieves comparable performance to the corresponding LLMs. Furthermore, direct LLM inference and FBPR exhibit complementary strengths, correctly answering largely different subsets of questions with overlap barely above chance. Combining them with a simple hybrid method yields a notable gain over either method alone. 
    
[^166]: SELF：一种用于大语言模型指纹识别的鲁棒奇异值与特征值方法

    SELF: A Robust Singular Value and Eigenvalue Approach for LLM Fingerprinting

    [https://arxiv.org/abs/2512.03620](https://arxiv.org/abs/2512.03620)

    SELF是一种基于LLM注意力权重奇异值与特征值分解的内在指纹识别方案，无需输入依赖即可抵御虚假声明攻击和权重操纵，实现鲁棒且可扩展的模型知识产权取证。

    

    保护大语言模型（LLM）的知识产权（IP）免受模型窃取是当代人工智能研究中的一个关键挑战。尽管指纹识别对于建立模型衍生的取证证据至关重要，但现有方法容易受到虚假声明攻击、权重操纵或高误报率的影响，从而削弱了其实用性。为了克服这些局限性，我们提出了SELF，这是一种新颖的基于内在权重的指纹识别方案，专门用于通过对抗性模型谱系验证实现鲁棒的知识产权取证。SELF消除了对输入的依赖，并且天然能够抵抗虚假声明，通过对LLM注意力权重进行奇异值和特征值分解，提取独特、可扩展且变换不变的指纹，从而实现鲁棒的来源检测。实验结果表明，SELF在保持较高的知识产权侵权检测准确率的同时，显著（摘要截断）。

    arXiv:2512.03620v2 Announce Type: replace-cross  Abstract: Safeguarding the Intellectual Property (IP) of Large Language Models (LLMs) against model theft represents a critical challenge in contemporary AI research. Although fingerprinting is essential for establishing forensic evidence for model derivation, existing methods are susceptible to false claim attacks, weight manipulations, or high false positive rates, undermining their utility. To overcome these limitations, we propose SELF, \rev{a novel intrinsic weight-based fingerprinting scheme tailored for robust IP forensics via adversarial model lineage verification}. SELF eliminates dependency on input and inherently resists false claims, achieving robust \rev{provenance detection} through unique, scalable and transformation-invariant fingerprint extraction via singular value and eigenvalue decomposition of LLM attention weights. Experimental results demonstrate that SELF maintains high IP infringement detection accuracy while sig
    
[^167]: 跨用户投毒：多用户协作语言代理中的用户-任务边界失效

    Cross-User Poisoning: User-Task Boundary Failures in Multi-User Collaborative Language Agents

    [https://arxiv.org/abs/2511.17671](https://arxiv.org/abs/2511.17671)

    该论文首次识别出多用户共享语言代理中的“跨用户投毒”（CUP）攻击，即攻击者注入共享上下文的恶意消息会在代理服务其他无辜用户时被错误执行，并提出 MURMUR 评估框架，在 Continua、ElizaOS 等真实系统中验证了该攻击的高成功率与持久性。

    

    语言代理正从单用户助手转变为工作空间、论坛和群聊中的共享协作者。共享代理会观察来自多个用户的交错消息、维护持久上下文并执行工具操作，这带来了标准单用户代理所不存在的一个边界问题：代理不仅要判断某条指令本身是否安全，还必须判断该指令被允许管辖哪个用户或哪个任务。我们识别出跨用户投毒（CUP）这一威胁：攻击者向共享上下文中注入一条消息，该消息随后在代理为无辜用户服务时被应用，从而导致未经授权的操作或超出指令预期范围的响应。我们在两个已部署的多用户代理 Continua 和 ElizaOS 上验证了 CUP 的有效性，并提出了 MURMUR——一个用于在并发多用户交互场景下评估共享上下文代理的框架。在 Slack、Workspace 和 Airline 三个领域中，CUP 取得了很高的攻击成功率，并能持续存在。

    arXiv:2511.17671v2 Announce Type: replace-cross  Abstract: Language agents are moving from single-user assistants to shared collaborators in workspaces, forums, and group chats. A shared agent observes interleaved messages from multiple users, maintains persistent context, and executes tool actions, creating a boundary problem absent from standard single-user agents: the agent must decide not only whether an instruction is safe, but which user or task it is allowed to govern. We identify cross-user poisoning (CUP), where an adversary injects a message into shared context that is later applied while the agent serves a benign user, causing unauthorized actions or responses outside the instruction's intended scope. We validate CUP on two deployed multi-user agents, Continua and ElizaOS, and introduce MURMUR, a framework for evaluating shared-context agents under concurrent multi-user interactions. Across Slack, Workspace, and Airline domains, CUP achieves high attack success, persists acr
    
[^168]: 从岩层到云端：电池全生命周期中最先进自然语言处理技术的系统性综述

    From the Rock Floor to the Cloud: A Systematic Survey of State-of-the-Art NLP in Battery Life Cycle

    [https://arxiv.org/abs/2510.27369](https://arxiv.org/abs/2510.27369)

    该论文系统综述了自然语言处理在电池全生命周期各阶段的应用，并提出了一种面向欧盟数字电池护照（DBP）的新型技术语言处理（TLP）概念框架。

    

    本工作对自然语言处理（NLP）在电池全生命周期中的应用进行了系统性综述，而非仅针对某一阶段或方法的综述，并以此为主要目标，为欧盟（EU）提议的数字电池护照（DBP）以及其他通用电池预测引入了一种新颖的概念性技术语言处理（TLP）框架。该综述遵循系统综述与荟萃分析优先报告条目（PRISMA）方法，采用三个权威数据库或搜索引擎进行检索，包括Google Scholar、电气电子工程师学会Xplore（IEEE Xplore）和Scopus，检索时间为2025年10月。检索标准为“文本”、“语言处理”和“电池”，排除标准为不相关的论文以及2017年之前发表的论文；该标准适用于所有数据库，但IEEE Xplore除外，因其检索结果较少。因此，……

    arXiv:2510.27369v2 Announce Type: replace  Abstract: This work presents a systematic review of the application of natural language processing (NLP) along the entire battery life cycle, instead of the review of one stage or method, and introduces a novel conceptual technical language processing (TLP) framework for the European Union (EU)'s proposed digital battery passport (DBP) and other general battery predictions, as the main objective. It follows the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) method and employs three reputable databases or search engines, including Google Scholar, Institute of Electrical and Electronics Engineers Xplore (IEEE Xplore), and Scopus, searched in October, 2025. The search criteria are 'text', 'language processing', and 'battery' while the exclusion criteria are non-relevant papers and those published before 2017, which was applied to all databases, except IEEE Xplore because it turned up fewer results. Consequently, it as
    
[^169]: 面向多模态情感分析的不确定性引导多模态共识

    Uncertainty-Guided Multimodal Consensus for Multimodal Sentiment Analysis

    [https://arxiv.org/abs/2510.20256](https://arxiv.org/abs/2510.20256)

    提出了不确定性引导的多模态共识方法UMC，通过连续原型引导的标签恢复（CPLR）和可靠性与精确度共识（RPC）两个组件，解决了多模态情感分析中共享标签掩盖模态特定情感以及模态不一致时不可靠预测被过度强调的问题。

    

    多模态情感分析（MSA）近年来取得了长足进展。现有的MSA方法主要学习共享的多模态表示，或使用共同的多模态标签训练单模态预测器。然而，将共享标签分配给每个模态可能会掩盖模态特定的情感，而直接融合在模态之间不一致时可能会过度强调不可靠的预测。为了解决这些问题，我们提出了不确定性引导的多模态共识（UMC），该方法首先恢复连续的单模态伪标签，然后在各模态之间形成可靠性感知的共识。我们的方法包含两个关键组件：i）连续原型引导的标签恢复（CPLR），它将异方差预测与有序情感原型相结合，利用置信度和标签冲突来精炼连续的单模态伪标签；ii）可靠性与精确度共识（RPC），它估计样本相关的模态可靠性并据此构建跨模态共识。

    arXiv:2510.20256v2 Announce Type: replace-cross  Abstract: Multimodal sentiment analysis (MSA) has made considerable progress in recent years. Existing MSA methods mainly learn shared multimodal representations or train unimodal predictors with a common multimodal label. However, assigning the shared label to every modality can obscure modality-specific sentiment, while direct fusion may overemphasize unreliable predictions when modalities disagree. To address these issues, we propose Uncertainty-Guided Multimodal Consensus (UMC), which first recovers continuous unimodal pseudo-labels and then forms a reliability-aware consensus across modalities. Our method consists of two key components: i) Continuous Prototype-Guided Label Recovery (CPLR), which combines heteroscedastic predictions with ordered sentiment prototypes and uses confidence and label conflict to refine continuous unimodal pseudo-labels, and ii) Reliability and Precision Consensus (RPC), which estimates sample-dependent mo
    
[^170]: FullFront：面向前端工程全流程的多模态大语言模型基准测试

    FullFront: Benchmarking MLLMs Across the Full Front-End Engineering Workflow

    [https://arxiv.org/abs/2505.17399](https://arxiv.org/abs/2505.17399)

    FullFront是一个新的基准测试，通过网页设计、网页感知问答和网页代码生成三项任务，全面评估多模态大语言模型在前端工程全流程中的能力。

    

    前端工程涉及一个复杂的工作流程，工程师需要将设计概念化、将其转化为代码，并迭代地完善实现。虽然近期的基准测试主要集中于将视觉设计转换为代码，我们提出了FullFront，这是一个旨在评估多模态大语言模型（MLLM）在整个前端开发流程中能力的基准测试。FullFront评估三项直接对应前端工程流程的基本任务：网页设计（概念化阶段）、网页感知问答（对视觉组织与元素的理解）以及网页代码生成（实现阶段）。与现有基准测试使用代码臃肿的爬取网站或过度简化的LLM生成HTML不同，FullFront采用了一种新颖的两阶段流程，将真实世界的网页转换为干净、标准化的HTML，同时保持多样化的视觉设计并避免版权问题。

    arXiv:2505.17399v4 Announce Type: replace  Abstract: Front-end engineering involves a complex workflow where engineers conceptualize designs, translate them into code, and iteratively refine the implementation. While recent benchmarks primarily focus on converting visual designs to code, we present FullFront, a benchmark designed to evaluate Multimodal Large Language Models (MLLMs) \textbf{across the full front-end development pipeline}. FullFront assesses three fundamental tasks that map directly to the front-end engineering pipeline: Webpage Design (conceptualization phase), Webpage Perception QA (comprehension of visual organization and elements), and Webpage Code Generation (implementation phase). Unlike existing benchmarks that use either scraped websites with bloated code or oversimplified LLM-generated HTML, FullFront employs a novel, two-stage process to transform real-world webpages into clean, standardized HTML while maintaining diverse visual designs and avoiding copyright i
    
[^171]: EffiBench-X：一个用于衡量LLM生成代码效率的多语言基准

    EffiBench-X: A Multi-Language Benchmark for Measuring Efficiency of LLM-Generated Code

    [https://arxiv.org/abs/2505.13004](https://arxiv.org/abs/2505.13004)

    本文提出首个多语言代码效率基准EffiBench-X，评估发现LLM生成的代码虽然功能正确，但平均效率仅约为人类专家的62%，且在不同编程语言之间存在显著的效率差异。

    

    现有的代码生成基准主要评估功能正确性，对代码效率的关注有限，且通常仅限于Python等单一语言。为了填补这一空白，我们提出了EffiBench-X，这是首个旨在衡量LLM生成代码效率的多语言基准。EffiBench-X支持Python、C++、Java、JavaScript、Ruby和Golang。该基准由竞赛编程任务组成，并以人类专家的解决方案作为效率基线。在最先进LLM上的评估结果显示，尽管这些模型能够生成功能正确的代码，但它们在效率上始终不及人类专家。即使是最有效的LLM生成方案（Qwen3-32B），其平均效率也仅达到人类效率的约62%，且存在显著的语言特异性差异。LLM在Python、Ruby和JavaScript中的效率表现优于Java、C++和Golang。例如，DeepSeek-R1的Python

    arXiv:2505.13004v2 Announce Type: replace  Abstract: Existing code generation benchmarks primarily evaluate functional correctness, with limited focus on code efficiency and often restricted to a single language like Python. To address this gap, we introduce EffiBench-X, the first multi-language benchmark designed to measure the efficiency of LLM-generated code. EffiBench-X supports Python, C++, Java, JavaScript, Ruby, and Golang. It comprises competitive programming tasks with human-expert solutions as efficiency baselines. Evaluating state-of-the-art LLMs on EffiBench-X reveals that while models generate functionally correct code, they consistently underperform human experts in efficiency. Even the most efficient LLM-generated solutions (Qwen3-32B) achieve only around \textbf{62\%} of human efficiency on average, with significant language-specific variations. LLMs show better efficiency in Python, Ruby, and JavaScript than in Java, C++, and Golang. For instance, DeepSeek-R1's Python 
    
[^172]: 一种极其简单的百万规模多模态指令微调数据选择策略

    An Embarrassingly Simple Data Selection Strategy for Million-Scale Multi-Modal Instruction-Tuning

    [https://arxiv.org/abs/2503.13383](https://arxiv.org/abs/2503.13383)

    该论文提出一种极其简单的百万规模多模态指令微调数据选择策略，以交互风格衡量结构多样性、按能力维度评估数据质量，无需计算复杂的嵌入或梯度即可高效筛选高质量指令数据。

    

    arXiv:2503.13383v2 公告类型：replace-cross 摘要：近期研究表明，预训练大语言模型（LLMs）在指令微调时仅需少量但经过精心筛选的数据。在大语言模型基础上的多模态模型还要面对更庞大的数据规模、异构的数据来源以及跨模态对齐等问题，这使得数据选择的必要性和复杂性都被进一步放大。然而，现有方法依赖于深度提取的嵌入向量、梯度或范围狭窄的质量评分，这些指标计算耗时且表达能力有限，往往难以超越简单的随机采样基线。为克服这些局限，我们摒弃复杂的嵌入空间，回归指令微调的根本目标：应当从预训练模型中引出什么样的知识，以及如何将这些知识呈现给用户。我们利用交互风格作为结构多样性的代理指标，并将数据质量分解为针对特定能力的维度……

    arXiv:2503.13383v2 Announce Type: replace-cross  Abstract: Recent research demonstrates that pretrained large language models (LLMs) require minimal but carefully curated data for instruction-tuning. Building on LLMs, multi-modal models further contend with far larger data volumes, heterogeneous sources, and cross-modal alignment, amplifying both the necessity and the complexity of data selection. However, existing methods rely on deeply extracted embeddings, gradients or narrow quality scores that are time-consuming to compute and limited in expressiveness, frequently failing to surpass a simple random-sampling baseline. To overcome these limitations, we step back from complex embedding spaces and target the fundamental objective of instruction tuning: how and what knowledge should be elicited from the pretrained model and exposed to users. We leverage interaction \underline{S}tyles as a proxy for structural diversity, and decompose data quality into capability-specific \underline{S}c
    
[^173]: DYNASHIELD：通过动态解码定制实现的大语言模型黑盒移动目标防御

    DYNASHIELD: A Black-Box Moving Target Defense for LLMs via Dynamic Decoding Customization

    [https://arxiv.org/abs/2412.07672](https://arxiv.org/abs/2412.07672)

    DYNASHIELD通过动态定制解码参数和系统提示，在无需访问模型内部或额外训练的情况下，有效降低了黑盒LLM面对越狱攻击的成功率。

    

    大型语言模型（LLMs）仍然容易受到越狱攻击，其中对抗性提示会诱导产生有害输出。现有防御方法通常需要访问模型内部结构或进行额外训练，这限制了它们在通过黑盒API部署的服务提供商中的适用性。在本文中，我们提出了DYNASHIELD，一种移动目标防御框架，通过在推理时定制解码超参数和系统提示来提高鲁棒性。DYNASHIELD包括两个关键步骤：（1）识别能降低攻击成功概率的解码配置，（2）从加权配置池中进行概率采样，以引入模型行为的受控变异性。我们在7个开源LLM上，针对4种最先进的越狱攻击，使用AdvBench中的对抗性提示评估了DYNASHIELD。结果表明，与7种基线防御相比，攻击成功率显著降低。

    arXiv:2412.07672v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) remain vulnerable to jailbreak attacks in which adversarial prompts induce harmful outputs. Existing defenses often require access to the model internals or additional training, limiting their applicability for service providers deployed through black-box APIs. In this paper, we propose DYNASHIELD, a moving target defense framework that improves robustness by customizing decoding hyperparameters and system prompts at inference time. DYNASHIELD includes two key steps: (1) it identifies decoding configurations that reduce attack success probability, and (2) it probabilistically samples from a weighted configuration pool to introduce controlled variability in model behavior. We evaluate DYNASHIELD across 7 open-source LLMs under 4 state-of-the-art jailbreak attacks, using adversarial prompts from AdvBench. Results show substantial reductions in attack success rate compared with 7 baseline defenses, whi
    
[^174]: HateSense：用于仇恨言论检测的分层情感分析框架：实现二分类与多分类策略

    HateSense: Hierarchical Sentiment Analysis Framework for Hate Speech Detection: Implementing Binary and Multiclass Classification Strategy

    [https://arxiv.org/abs/2411.05819](https://arxiv.org/abs/2411.05819)

    该研究通过受控实验发现，分层分类和情感特征在仇恨言论检测中并未带来显著收益，而类别加权训练才是唯一能大幅提升性能的有效干预措施。

    

    仇恨言论检测器必须将仇恨性帖子与更为常见的冒犯性但非仇恨性帖子区分开来，实现这一目标的两个流行思路是分层（由粗到细）分类和情感特征。我们重新审视了我们早期结合这两种思路的基于Transformer的框架，并在两个公开基准数据集（Davidson等人的24,783条推文；HateXplain的19,229条帖子）上通过受控研究测试了每个组件，实验采用五次随机划分、基于验证集的调参以及配对自助法检验。结果表明，在使用基于词级和字符级n-gram的线性模型时，我们框架中采用的优势类别优先的分层方法相比平面分类器使宏平均F1降低了1.1至1.5个点（p <= 0.007），而添加VADER情感特征对Davidson数据集宏平均F1的影响最多为0.6个点，且在类别加权设置中从未达到统计显著水平。类别加权训练是唯一具有较大效果的干预措施（宏平均F1提升2.0至3.5个点，仇恨类召回率提升5.9至9.1个点）。

    arXiv:2411.05819v2 Announce Type: replace-cross  Abstract: Hate speech detectors must separate hateful posts from the far more common offensive but non-hateful ones, and two popular ideas for doing so are hierarchical (coarse-to-fine) classification and sentiment features. We revisit our earlier transformer-based framework, which combines both ideas, and test each component in a controlled study on two public benchmarks (Davidson et al., 24,783 tweets; HateXplain, 19,229 posts) with five random splits, validation-based tuning, and paired bootstrap tests. With linear models over word and character n-grams, the dominant-class-first hierarchy used in our framework lowered macro-F1 by 1.1-1.5 points relative to a flat classifier (p <= 0.007), and adding VADER sentiment features changed Davidson macro-F1 by at most 0.6 points, never significantly in the class-weighted settings. Class-weighted training was the only intervention with a large effect (+2.0 to +3.5 macro-F1 and +5.9 to +9.1 hate
    
[^175]: 大语言模型路由与级联的统一方法

    A Unified Approach to Routing and Cascading for LLMs

    [https://arxiv.org/abs/2410.10347](https://arxiv.org/abs/2410.10347)

    本文提出“级联路由”这一统一理论框架，将路由与级联两种模型选择范式相结合，推导出了最优级联策略并证明了现有路由策略的最优性，从而有效改善大语言模型选择的成本-性能权衡。

    

    嵌入在各种智能体系统中的大量可用大语言模型（LLM）显著提升了模型选择策略改善成本-性能权衡的潜力。现有策略要么采用路由，即为每个查询选择单个模型；要么采用级联，即依次运行越来越大的模型直到找到满意的答案。然而，当前方法面临三个关键限制：（1）缺乏最优性的形式化证明，（2）未能确定这些策略在何种条件下最有效地改善成本-性能权衡，（3）无法将两种范式结合以获得进一步改进。为解决这些问题，我们首先推导出一种新颖的最优级联策略，并证明了现有路由策略的最优性。此外，我们提出了级联路由——一个将路由和级联整合到统一理论框架中的方法。

    arXiv:2410.10347v5 Announce Type: replace  Abstract: The availability of a wide range of large language models (LLMs) embedded in various agentic systems has significantly increased the potential of model selection strategies to improve the cost-performance tradeoff. Existing strategies involve either routing, where a single model is chosen per query, or cascading, which sequentially runs increasingly larger models until a satisfactory answer is found. However, current approaches face three key limitations: they (1) lack formal proofs of optimality, (2) fail to identify the conditions under which these strategies are most effective to improve the cost-performance tradeoff, and (3) are unable to combine both paradigms for further improvements. To address these issues, we first derive a novel optimal strategy for cascading and prove the optimality of an existing routing strategy. Further, we propose cascade routing, a unified framework that integrates routing and cascading into a theoret
    
[^176]: 语言模型中的公平性定义详解

    Fairness Definitions in Language Models Explained

    [https://arxiv.org/abs/2407.18454](https://arxiv.org/abs/2407.18454)

    本文通过系统性综述澄清了语言模型中各种公平性定义的内涵，解决了不同公平性概念之间区分模糊、适用情境不明确的问题。

    

    语言模型（LMs）在各种自然语言处理（NLP）任务中展现出了卓越的性能。尽管取得了这些进展，语言模型可能会继承并放大与性别和种族等敏感属性相关的社会偏见，从而限制了其在现实世界应用中的采用。因此，语言模型的公平性已得到广泛探索，并催生了各种公平性概念的提出。然而，由于缺乏关于在特定情境下应采用哪种公平性定义的明确共识，以及理解这些定义之间区别的复杂性，这可能会造成混淆并阻碍该领域的进一步发展。为此，本文提出了一项系统性综述，以阐明公平性定义在语言模型中的应用。具体而言，我们首先简要介绍语言模型及其中公平性的概念，随后对现有的语言模型公平性概念进行了全面且最新的概述，并引入了……

    arXiv:2407.18454v4 Announce Type: replace-cross  Abstract: Language Models (LMs) have demonstrated exceptional performance across various Natural Language Processing (NLP) tasks. Despite these advancements, LMs can inherit and amplify societal biases related to sensitive attributes such as gender and race, limiting their adoption in real-world applications. Therefore, fairness has been extensively explored in LMs, leading to the proposal of various fairness notions. However, the lack of clear agreement on which fairness definition to apply in specific contexts and the complexity of understanding the distinctions between these definitions can create confusion and impede further progress. To this end, this paper proposes a systematic survey that clarifies the definitions of fairness as they apply to LMs. Specifically, we begin with a brief introduction to LMs and fairness in LMs, followed by a comprehensive, up-to-date overview of existing fairness notions in LMs and the introduction of 
    

