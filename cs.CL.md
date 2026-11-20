# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [AgentRoboCritic: Understanding the Capability of Frontier Agents on Direct Robot Control](https://arxiv.org/abs/2611.26613) | 本文提出AgentRoboCritic——一个基于意图的动作级评论框架，通过让策略智能体声明局部意图并由评论器评估动作结果与失败模式，系统分析了前沿多模态智能体在12个操作基准上直接控制机器人的能力与失败原因。 |
| [^2] | [Represented but misread: Post-training changes how language models read out what they represent](https://arxiv.org/abs/2611.26584) | 研究发现后训练并未破坏语言模型内部的表征（线性探针仍能准确区分人类与AI文本），而是改变了模型将表征读取为答案的方式，导致回答灵敏度下降和对先验信息的错误利用。 |
| [^3] | [Hybrid Retrieval for Government Scheme Search in English and Marathi](https://arxiv.org/abs/2611.26514) | 该论文提出一个面向马哈拉施特拉邦政府计划的双语（英语-马拉地语）搜索系统，通过等权重融合BM25与句子嵌入相似度，混合检索显著优于单独的BM25或密集检索，MRR最高达0.879。 |
| [^4] | [AutoMaster: Autonomous Post-Training by Making the Same Model a Better Teacher](https://arxiv.org/abs/2611.26510) | 该论文提出了Skill-FT技能引导微调方法和AutoMaster自主后训练系统，由LLM智能体自动分析教师轨迹并设计自然语言技能干预来抑制不良习惯、激发有效行为，从而在多个任务上比朴素SFT提升1.42-8.99个百分点的性能。 |
| [^5] | [GLiDE: A Fast, Calibrated Decision Model Distilled from Reasoning Teachers](https://arxiv.org/abs/2611.26509) | GLiDE 通过多教师软目标蒸馏、双向注意力机制和无参数循环深度，将推理教师模型的能力迁移到单次前向传播即可完成的决策模型中，在无需生成推理文本的情况下保留了推理带来的决策质量提升。 |
| [^6] | [Multimodal Error Detection in Chest X-ray Reports: A Large Language Model Reading Image Evidence](https://arxiv.org/abs/2611.26492) | Rad-Linter利用预训练图像分类器的九种病理评分，通过LoRA微调的15亿参数语言模型检查胸部X光报告与图像证据的一致性以标记潜在错误，在受控反事实评估中配对准确率达98.5%，显著优于关键词-否定规则基线的80.5%。 |
| [^7] | [MemFolder: Structure-Guided Memory Folding for Efficient Agent Memory Management](https://arxiv.org/abs/2611.26427) | 提出了结构引导的记忆折叠框架MemFolder，通过共享记忆分组的证据合并与紧凑表示两个核心组件，依次降低记录数量和上下文长度，实现更高效的长期智能体记忆管理。 |
| [^8] | [Democratizing Cross-Project Knowledge for Transforming Water Governance and Climate Resilience: an Agentic AI Ecosystem](https://arxiv.org/abs/2611.26398) | 本文提出了一个智能体AI生态系统，将三个欧盟项目的异构文档整合为领域感知知识库，结合领域微调嵌入模型与ReAct式智能体检索增强生成流水线，实现跨项目知识的自适应访问，以服务水治理与气候韧性决策。 |
| [^9] | [HiSA: Sparsifying the Hidden Dimension of RoPE Attention in LLMs](https://arxiv.org/abs/2611.26363) | 提出HiSA方法，基于RoPE结构将注意力点积在隐藏（特征）维度上进行稀疏化，仅需计算约三分之一的查询-键点积就能在LLM训练中提升而非损害性能。 |
| [^10] | [Beam Model Technical Report](https://arxiv.org/abs/2611.26358) | Beam 是一个总参数 5013 亿、活跃参数 234 亿的稀疏 MoE 模型，专为编程、推理和智能体任务优化，通过 23.8 万亿 token 预训练、两阶段中期训练实现百万 token 上下文窗口，并在 10500 块 GB300 GPU 上开展开放实验室中最大规模的强化学习后训练，以比同类模型少 3-4 倍的推理计算量达到前沿推理水平。 |
| [^11] | [One Framework to Diagnose Them All: A Stratified Benchmark for Bengali LLM Hallucination](https://arxiv.org/abs/2611.26336) | 本文提出孟加拉语幻觉检测分层基准 Alikbachan，通过文化距离、任务类型和32种幻觉模式三个维度交叉分层，并在社区竞赛中大规模验证，发现语篇级信息遗漏是零样本LLM评判器与参赛系统的共同弱项，证明了分层设计对模型幻觉能力的显著诊断价值。 |
| [^12] | [Language-Oriented Semantic Modeling for Target Language Extraction](https://arxiv.org/abs/2611.26317) | 该论文提出一种面向语言的语义建模框架，通过语言感知提示条件化和参数高效微调将目标语言语义表示融入声学提取模型，在CommonVoiceMix数据集上相比仅声学基线最高提升1.16 dB SI-SDR。 |
| [^13] | [BADAS-3.0: Real-Time Driving Perception and Planning from One Camera](https://arxiv.org/abs/2611.26294) | BADAS-3.0 仅凭单摄像头即可在50毫秒内实时完成速度、路径、碰撞风险与未来轨迹预测，其轨迹精度比读取六个摄像头、慢100倍的34B模型低44%的minADE，并在新增的行车记录仪碰撞与Euro NCAP两个安全基准上通过率超过所有已发表规划器。 |
| [^14] | [Twice-Told Critiques: Claim-Level Homogenization of Peer Review in the LLM Era](https://arxiv.org/abs/2611.26286) | 通过测量会议和期刊评审中的声明级重叠度，研究发现LLM时代同行评审的批评意见日益同质化——ICLR评审重叠率从2026年的12.0%升至2027年的21.3%，且被检测为LLM生成的评审比人工评审重叠度更高，表明单篇论文评审的批评多样性正在下降。 |
| [^15] | [What If Chain-of-Thought Is Not Enough?](https://arxiv.org/abs/2611.26282) | 本文提出以“问题—选项—标准”（QOC）结构化决策过程替代思维链对大语言模型进行微调，并通过三代渐进式训练流水线证明该方法能显著提升答案质量，且模型规模越大收益越明显。 |
| [^16] | [dLoop: Looped Diffusion Language Models via Recurrent Latent Refinement](https://arxiv.org/abs/2611.26275) | dLoop通过将预训练扩散语言模型重构为Prelude-Loop-Coda架构，并对相同的部分掩码状态循环应用参数共享的中间块来迭代精炼潜在表示，从而在不推进去噪过程的情况下提升预测的可靠性，减少所需去噪步骤并加速生成。 |
| [^17] | [STAGE: Spatio-Temporal Augmented Graph Expansion for Weakly Supervised Cross-Tweet Event Causality Discovery](https://arxiv.org/abs/2611.26255) | STAGE框架从局部因果种子出发构建跨推文弱监督标签，结合时空上下文图与图注意力网络，实现灾害社交媒体中跨推文事件因果关系的可扩展自动发现，相比局部监督显著提升因果发现性能。 |
| [^18] | [TwinLab-Bench: Evaluating AI Scientists in Agentic Twins of Real Labs](https://arxiv.org/abs/2611.26248) | 该论文提出TwinLab-Bench基准，利用来自真实物理科学、生命科学和工程实验室的52个任务，首次系统评估AI科学家自主设计实验的能力，其结果通过与真实实验室隐藏记录对比来评分。 |
| [^19] | [Frontier models need only one text to unmask its human or AI author](https://arxiv.org/abs/2611.26216) | 前沿大模型仅凭一段文本和简单的零样本提示，就能判断作者是AI还是人类、识别AI文本来自哪家实验室、甚至准确指认人类作者的身份，其性能媲美专用检测器并能给出自然语言解释。 |
| [^20] | [Learning to Think with Images Faithfully via Adaptive Data Selection](https://arxiv.org/abs/2611.26199) | 该论文揭示了“用图像思考”模型普遍存在不忠实的工具调用现象（即答对却并不真正依赖放大后的区域），并提出自适应数据选择方法，使训练数据随模型能力动态调整、聚焦于放大证据真正有效的问题，从而实现忠实的基于图像的推理。 |
| [^21] | [Can World Models Predict Rice Growth? Toward Agricultural World Models](https://arxiv.org/abs/2611.26145) | 本文提出农业世界建模任务及AgroWM框架，其PhenoClock推理模块先推算水稻当前的发育阶段并预测未来阶段，再据此生成植株在数天至数月后的外观，克服了现有图像生成模型无法准确预测作物长期生长的局限。 |
| [^22] | [RL Can Only Reinforce What It Sees Succeed](https://arxiv.org/abs/2611.26084) | 在先验与正确答案均已精确已知的谜题环境中，该研究发现结果奖励强化学习只能强化训练中实际观察到成功的动作——当正确动作罕见时，训练能否“起飞”可由训练前即可计算的一个统计量（动作获胜的尝试组期望数量，约35组对应五成概率）准确预测，且该规律跨多种实验设置普遍成立。 |
| [^23] | [Same Judgement, Different Scores: Numerical Scoring Topology in LLM-as-a-Judge Evaluation](https://arxiv.org/abs/2611.26077) | 本文提出“数值评分拓扑”概念，并首次系统研究证明LLM-as-a-Judge评估中评分方案的结构设计（量程宽度、粒度、偏移、对称性与边界位置）会系统性影响评判分数、校准与鲁棒性，其中偏移和狭窄的评分范围会导致最大的分数偏差。 |
| [^24] | [Understanding MoE Decisions Beyond Gradients](https://arxiv.org/abs/2611.26075) | 该论文提出一种利用MoE模型自身的专家输出和选择规则来估计专家替换引发的级联切换效应的预测器，相比传统梯度方法将答案得分变化的预测误差在Qwen3.6上降低60%、在Gemma4上降低47%。 |
| [^25] | [AltRA-Test: An Alternative Annotator Test for LLM Judges Without Shared Annotators](https://arxiv.org/abs/2611.26073) | 提出了AltRA-Test统计检验方法，通过自助法重采样克服条目由不同标注者子集标注的限制，从而判断用LLM替代人类标注者在统计学上是否合理。 |
| [^26] | [SheetAlign: Grounded Interaction and Stage-Isolated Learning for Spreadsheet Agents](https://arxiv.org/abs/2611.26069) | SheetAlign是一个电子表格操作框架，通过阶段隔离策略学习、任务引导的API接地以及基于执行一致性的程序选择，将代码生成锚定在工作簿观察与可用操作上，从而在SpreadsheetBench上取得优异表现。 |
| [^27] | [Complete-SVD: A Functional Norm Approach to Low-Rank LLM Compression](https://arxiv.org/abs/2611.26059) | 提出Complete-SVD方法，通过最小化原始与压缩模型logit之间的等权L2距离，将大语言模型低秩压缩目标化简为具有闭式解的加权SVD，并利用随机探测以与词表大小和序列长度无关的计算成本无偏估计难以求解的输出侧因子。 |
| [^28] | [Evaluating safeguarding in multi-turn conversations with AI assistants](https://arxiv.org/abs/2611.26009) | 该论文与社区心理健康支持工作者合作设计了九个多轮对话场景，发现AI助手在痛苦线索逐渐显现的日常对话中可能无法保持与信号一次性明确呈现时同等的安全保障水平，从而揭示了现有安全评估方法的不足。 |
| [^29] | [Xing-Audio-NVVASR: Scaling and Refining Supervision for Joint Recognition of Speech and Non-Verbal Vocalizations](https://arxiv.org/abs/2611.26000) | 该论文提出Xing-Audio-NVVASR，通过渐进式数据挖掘与迭代监督精炼构建了约80万条话语的语料库，训练出17亿参数的端到端模型，实现带行内标签的语音与非言语发声联合识别。 |
| [^30] | [Policy Improvement-Guided Replay for Continual Reinforcement Learning](https://arxiv.org/abs/2611.25989) | 该论文提出对比似然重放方法（CoLiR），基于策略改进原则和性能差分引理，在视觉语言模型的持续强化学习中选择性重放历史提示，从而在获得新能力的同时保留已取得的性能提升。 |
| [^31] | [Energy estimates of open-weight large language models across stated food weights: a dose-response evaluation](https://arxiv.org/abs/2611.25976) | 七个开源大语言模型在估计食物能量时无法随给定重量成比例缩放（弹性仅0.548-0.979），导致小份量被高估、大份量被低估，而采用“模型自报能量密度×重量”的两步法可基本消除该误差。 |
| [^32] | [The Depth of a Prediction: Where Language Models Resolve a Word Predicts Human Reading Time](https://arxiv.org/abs/2611.25970) | 论文提出“解析深度”这一新指标，用以衡量语言模型在哪一层完成对一个词的预测，并证明模型解析越晚的词及其后续词会被人类读得更慢，且解析深度能在最终层意外度和熵之外显著提升对人类阅读时间的预测。 |
| [^33] | [When Sounds Turn Heads: Recovering Event Timing That Audio LLMs Fail to Say](https://arxiv.org/abs/2611.25960) | 音频大模型的少数注意力头内部已追踪到被查询声音的发生时间却无法在回答中表达，本文提出无需训练的头引导解码方法，利用这些注意力头的先验对候选时间戳重新打分，从而显著提升三种音频大模型的时间戳精度。 |
| [^34] | [Web search in Latin America: language models use less local and diverse news than search engines](https://arxiv.org/abs/2611.25955) | 本研究通过对拉丁美洲九国时事的大规模查询实验发现，基于语言模型的搜索返回的新闻来源多样性低于传统搜索引擎，且本地新闻占比因用户类型而表现不一致，揭示了AI搜索可能加剧地理偏见的风险。 |
| [^35] | [Many Solutions, One Answer: Search Order Shapes the Default Solutions of Language Models](https://arxiv.org/abs/2611.25916) | 论文发现语言模型在多解约束谜题中高度集中于同一个默认解，其行为如同按特定顺序遍历的搜索求解器——优先访问初始候选最少的实体这一搜索顺序最能预测模型输出，且改变给定的顺序会相应改变模型的默认解。 |
| [^36] | [The Linguistic Structure of Dependency-Conditioned Embeddings](https://arxiv.org/abs/2611.25909) | 该论文通过分析全局与局部几何结构，系统考察了依存条件化词嵌入的语言组织方式，发现全局几何主要由上下文词性塑造，而局部词汇相似性结构在依存关系间得到部分保留。 |
| [^37] | [Verifiable Logic in Medical LLMs: A Critical Review of Agentic Clinical Reasoning Optimization and Neuro-Symbolic Grounding](https://arxiv.org/abs/2611.25898) | 这篇综述指出临床大语言模型的最终答案正确并不代表推理有效，并系统评述了幻觉成因、智能体幻觉级联、过程奖励建模与神经符号接地四种范式，其中带符号否决层的系统能在其编码知识范围内提供可验证的安全保证。 |
| [^38] | [One Schedule Does Not Fit All: Adaptive Inference Schedules for Diffusion Language Models](https://arxiv.org/abs/2611.25895) | 本文提出 TCDA，一种具有显式误差保证的自适应部署规则，可根据任务、模型和采样器为扩散语言模型选择满足质量要求的最低成本推理调度，将去噪器评估次数最多减少 3.4 倍。 |
| [^39] | [HALTT: A HAte Lexicon Through Time](https://arxiv.org/abs/2611.25862) | 该论文提出了一个多阶段框架，构建并发布了首个收录19个多义仇恨词汇的历时性词典HALTT，以应对日常词汇被武器化为隐蔽贬义用语而导致的仇恨言论检测性能随时间下降的问题。 |
| [^40] | [TableRoute: Capability-Aware Routing for Scalable Table Question Answering](https://arxiv.org/abs/2611.25853) | 提出TableRoute框架，先生成经过验证的问题—表格证据并识别所需操作，再根据表格规模与模型能力在全表阅读、表格缩减和执行三种推理策略之间智能路由，从而实现可扩展的表格问答。 |
| [^41] | [Anchor and Check: Deterministic Cross-Prompt, Cross-Model Reconciliation for Auditable LLM Decisions](https://arxiv.org/abs/2611.25818) | 该论文提出“锚定与核验”确定性调和契约——让目标模型的贪心答案接受跨提示或跨模型家族的独立贪心核验、仅一致时才执行，以实现可审计的LLM自主决策，并发现汇集跨家族样本可使语义熵的AUROC提升1.7至4.0个百分点，而同模型温度核验是最弱设计。 |
| [^42] | [What Does Anti-Hallucination Training Buy? Saying Less, Cheap Pruning, and the Rest](https://arxiv.org/abs/2611.25814) | 该论文证明，视觉-语言模型抗幻觉偏好训练所降低的幻觉，绝大部分可由“少说话”（截断字幕至相同召回率）以及一个仅用11个特征、在500张图上拟合的廉价后验探针剪枝所重现，说明训练带来的真实改进远小于表面数字。 |
| [^43] | [Efficient and Robust Drafting with Multi-Token Prediction](https://arxiv.org/abs/2611.25813) | 提出了G-MTP——一种围绕单一共享草稿头构建的高度优化多词预测架构，通过目标激活注入、零拷贝KV缓存交叉注意力和聚类词表投影三项核心优化，使现有主干大语言模型能够高效生成高质量草稿，从而加速推测解码推理。 |
| [^44] | [Language models represent more about emphasis than they can tell you](https://arxiv.org/abs/2611.25808) | 该研究发现语言模型内部编码了文本所能决定的重音突显信息，但通过提示无法将其表达出来，而完整的重音信息必须依赖语音信号才能获得。 |
| [^45] | [RouteVerify: Routing Interventions for Step-Level Verification in Mixture-of-Experts Models](https://arxiv.org/abs/2611.25780) | 该论文提出 RouteVerify 受控干预框架，通过在混合专家模型的九个检查点上替换路由器 logits、MoE 输出和残差状态等内部激活，系统检验路由状态与逐步验证判断之间的因果关系。 |
| [^46] | [Closing Performance Gap for Diffusion Language Models under Low Inference Budgets](https://arxiv.org/abs/2611.25763) | 提出了无需训练的 CaRR 框架，通过 token 承诺重掩码、更新上下文回填和基于轨迹的恢复机制，在低推理预算下提升可编辑扩散语言模型的生成质量，从而缩小其性能差距。 |
| [^47] | [Verification Behavior Across Five Qwen2.5-Instruct Checkpoints](https://arxiv.org/abs/2611.25745) | 本研究通过双AI评分员协议系统评估了五个Qwen2.5-Instruct检查点对信息不充分的声明与验证行为，发现协议提示能可靠引发格式标记（60/60），但诊断执行标准（H3）极少被满足，且预期的随模型规模增长的向上转变并未出现。 |
| [^48] | [PosteriorBench: Evaluating Behavior Posterior Sampling in Language Models](https://arxiv.org/abs/2611.25735) | 该论文提出了 PosteriorBench 基准，用于从计算成本和与真实后验的 KL 散度两个维度评估语言模型的行为后验采样算法，并发现现有方法均无法同时兼顾高效性与保真度。 |
| [^49] | [VR-SparseQuant: Accurate FP4 Quantization with 2:4 Sparsity on Vera Rubin](https://arxiv.org/abs/2611.25717) | VR-SparseQuant利用Vera Rubin的逐元素2:4稀疏性，在与稠密量化相同的FP4数值预算下精炼误差较大的激活，在保持原生FP4硬件执行的同时有效恢复四比特量化的模型精度。 |
| [^50] | [STARK: Moving towards Personalized Agents That Truly Learn and Remember](https://arxiv.org/abs/2611.25705) | STARK提出了一种双图记忆架构，通过情景图记录用户交互事件、洞察图提取并连接行为模式，使个性化智能体能够真正学习和记住用户的行为规律。 |
| [^51] | [From Geographic Memory to Spatial Generalization in Large Language Models](https://arxiv.org/abs/2611.25703) | 该论文提出将大语言模型中的地理知识组织为潜在空间场，通过连续空间编码器将位置表示作为空间标记注入模型，从而实现从仅能回忆训练中见过的“地理空间记忆”向能预测未见区域的“空间泛化”能力的跨越。 |
| [^52] | [FoodReact-AR: Learning Audience Reactions that Zero-Shot LLMs Miss in Arabic Food-Insecurity Discourse](https://arxiv.org/abs/2611.25682) | 该研究将74,241条阿拉伯语粮食不安全Facebook帖子的表情反应聚类为四种受众画像，发现“轻视”与“悲伤”反应分别随2022年小麦价格冲击和2024年加沙粮食危机显著上升，而零样本阿拉伯语大语言模型虽能较好预测悲伤画像（83.8%）却几乎无法识别轻视画像（仅3.4%），揭示了文本中隐藏的受众反应信号。 |
| [^53] | [SEAF: Selective Semantic Enrichment and Visual Evidence Fusion for Multimodal Sarcasm Detection](https://arxiv.org/abs/2611.25672) | 提出SEAF双分支框架，通过选择性语义增强和基于文本锚定加权聚合的视觉证据融合，有效利用MLLM生成的实体背景知识与多图像视觉特征，提升多模态讽刺检测性能。 |
| [^54] | [ColorRef: Calibrating Verbal Feedback for Color Revision](https://arxiv.org/abs/2611.25653) | 该论文提出颜色指代游戏 ColorRef，揭示语言模型虽然能正确理解颜色修改的方向和分级用语，却无法据此产生幅度合适的颜色调整步长。 |
| [^55] | [TRAC: Efficient Diffusion Language Models for Machine Translation via Length-Aware Decoding](https://arxiv.org/abs/2611.25643) | 提出了无需训练的自适应解码方法TRAC，通过根据源语言长度和翻译方向动态调整目标生成长度、加速token提交，减少了扩散语言模型机器翻译中的冗余计算，在保持翻译质量的同时显著降低生成延迟。 |
| [^56] | [Generative News Recommendation Reproduces Popularity: A Memorization-Based Analysis](https://arxiv.org/abs/2611.25606) | 该研究利用约2000万条日本新闻浏览会话数据，首次从训练数据记忆化的角度揭示生成式新闻推荐再现流行内容的机制——被频繁点击的新闻标题更容易被模型记住，并在更多样化的用户上下文中被生成，从而导致信息多样性下降。 |
| [^57] | [Rubric Semantics or Numerical Priors? Auditing Scale-Direction Bias in LLM Judges](https://arxiv.org/abs/2611.25588) | 该论文提出“语义忠实性”概念以及量表反转探针和地板归一化方法，揭示了大语言模型自动评审普遍存在量表方向偏差——评分依赖数值锚点而非量规语义，并证明双向锚点平均可有效缓解该问题。 |
| [^58] | [AutoRAGBench: A Benchmark for LLM Agents Automating RAG System Development](https://arxiv.org/abs/2611.25582) | 本文提出了首个基准测试AutoRAGBench，通过九个任务评估LLM智能体在无起始代码和预设策略的条件下，于开放代码空间中自主从零构建并迭代优化RAG系统的能力。 |
| [^59] | [The Coverage Selection Gap: Ranking, Not Generation, Limits LLM Rubric Construction](https://arxiv.org/abs/2611.25557) | 该论文发现在自动评分标准构建中，限制性能的瓶颈是对候选标准的排序选择而非标准生成，并提出“覆盖-选择差距”指标来量化无参考排序与有参考贪婪诊断之间的性能差异。 |
| [^60] | [COVERAGE-TRACE: Source-Side Counterfactual Tracing for Detecting Omissions in LLM Summaries](https://arxiv.org/abs/2611.25539) | 提出COVERAGE-TRACE源侧反事实框架，通过替换源文本原子事实并测量摘要似然变化来检测大语言模型摘要中的遗漏，在QMSum数据集上达到0.764–0.825的AUROC，显著优于强基线的0.640。 |
| [^61] | [LM-based Personalized Travel Planner](https://arxiv.org/abs/2611.25531) | 提出了一种层次化多智能体旅行规划框架，由监督智能体协调多个专门智能体，并基于多维评分机制进行迭代优化，在TravelPlanner基准上将预算约束规划性能提升了10%。 |
| [^62] | [MATE: Mobile Agent Trustworthy Execution for Safe and Efficient Mobile Control](https://arxiv.org/abs/2611.25526) | MATE是一个无需修改底层模型的执行层系统，通过结构化动作记录、运行时护栏、预定义任务路由和浮动界面，将移动智能体高风险行为拦截率从29.7%提升至97.0%，同时减少最多6倍的LLM调用并将重复任务成功率提升至95.7%。 |
| [^63] | [DRVEXO-4B-Unified: Design and Evaluation of a Modular Speech-Vision System](https://arxiv.org/abs/2611.25521) | 本文提出基于4B语言骨干的模块化多模态助手DRVEXO-4B-Unified，通过时序残差连接器、显式文本边界和任务条件化评分头将语音、视觉与文本交互统一到一个自包含检查点中，并在MMMU、IFEval和MATH-500基准上超越同规模的Gemma 3 4B IT、InternVL3.5-4B与Qwen3-4B等模型。 |
| [^64] | [When Should a Voice Agent Speak? End-of-Turn Detection in Turkish](https://arxiv.org/abs/2611.25489) | 本研究基于土耳其语双人视频通话语料库，比较微调 TurnGPT、零样本语言模型和多语言检测器在区分话轮结束与轮次内停顿上的表现，发现微调模型效果最佳且校准最好，与说话者时间特征结合可获得最佳的话轮结束检测性能。 |
| [^65] | [PILAPS: A Bilingual Benchmark for Evaluating Pragmatic Inference across Phenomena and Settings](https://arxiv.org/abs/2611.25459) | 该论文提出了首个英语/普通话双语语用推理评测基准PILAPS，通过评估八个LLM揭示出模型语用能力在模型规模、语用现象和语言间存在显著差异，尤其开源模型在幽默理解上表现薄弱。 |
| [^66] | [PAIR-MED: Policy-reward Alternation for Iterative Refinement in Multilingual Medical Reasoning](https://arxiv.org/abs/2611.25446) | 该论文提出PAIR-MED交替训练框架，通过让多语言推理策略与医学过程奖励模型相互迭代更新，解决了固定奖励模型与演化策略分布不匹配的问题，从而提升多语言医学推理能力。 |
| [^67] | [RoCoKV: Accelerating LLM Inference Serving through Uncertainty and Redundancy Aware KV Cache Compression](https://arxiv.org/abs/2611.25433) | 提出RoCoKV方法，通过不确定性引导的自适应剪枝和冗余感知的token选择来压缩大型推理模型的KV缓存，在不损害推理质量的前提下加速大语言模型推理服务。 |
| [^68] | [Evolve Where It Matters: Coverage-Guided Self-Evolution of Agent Skills](https://arxiv.org/abs/2611.25429) | 提出了 SkillPlumber 框架，利用指令级覆盖率标注区分失败源于指令错误还是未遵循指令，使智能体能够只在技能真正不足之处进行针对性的自我进化。 |
| [^69] | [ChronoDossier: Long-Form Personas as Temporal Memory Representations](https://arxiv.org/abs/2611.25407) | 该论文提出ChronoDossier基准，将长篇人物档案视为时序记忆表征，利用311份合成档案和内在与外在双重指标，评估长时程智能体在事实随时间变化时记录和恢复人物潜在状态的能力。 |
| [^70] | [Incorporating argumentation frameworks for LLM decision-making: Task-specific design and evaluation](https://arxiv.org/abs/2611.25361) | 本文抽象出四阶段的论证增强工作流程，发现现有方法在不同任务上提升不一致，并据此提出了Qnt-Arg等两种面向特定任务的方法，以改进LLM在辩论决策与事实核查中的表现。 |
| [^71] | [Pro$^2$ Agent: Scientific Problem Probing for Autonomous AI R&D](https://arxiv.org/abs/2611.25357) | Pro$^2$ Agent将自主AI研究的核心状态从代码库转变为显式的问题状态，通过全面分析、定向研究和受控假设检验来识别阻碍成功的问题并匹配候选解决方案，从而避免智能体在困难问题上陷入错误的解区域。 |
| [^72] | [Enhancing Understanding of Language Models with Varied Explanations](https://arxiv.org/abs/2611.25314) | 本论文系统研究了高亮解释、自然语言解释和思维链推理等多种解释形式，并从忠实性、合理性和实用性三个维度评估其对语言模型可解释性的互补贡献。 |
| [^73] | [LLMs Are Already Confidence-Aware Zero-Shot Classifiers](https://arxiv.org/abs/2611.25289) | 研究通过系统评估表明，大语言模型作为零样本分类器已具备置信度感知能力，基于logit的白盒评分方法在判别分类置信度可靠性上显著优于黑盒方法。 |
| [^74] | [When Objectives and Evaluators Disagree: Evaluating Transfer for Post-hoc Demographic Framing Mitigation](https://arxiv.org/abs/2611.25258) | 该研究将无法访问源模型时通过事后重写缓解大语言模型人口框架偏差视为迁移问题，发现总是针对说服偏差指数（PBI）较低信息的固定策略迁移不一致，而在两侧提出编辑但最多接受一个的自适应策略在两个评估器家族八项比较中的七项显著更优，且主要得益于编辑了更多对而非引导方向选择。 |
| [^75] | [Retrieving Patristic References and Modelling Topics in Bullinger's Reformation Correspondence](https://arxiv.org/abs/2611.25247) | 本研究首次将针对拉丁语圣经互文性微调的检索模型迁移至布林格宗教改革书信中的教父引用检测任务，通过比较词汇、稠密与混合检索方法，发现在圣经复用上微调的拉丁语编码器在双编码器中表现最佳，而 BM25 在显式引用检测上更具优势。 |
| [^76] | [Steering Automated Scientific Discovery via Audited Epistemic Exploration](https://arxiv.org/abs/2611.25214) | 该论文提出即插即用的认知脚手架 EpiCompass，通过维护连接假设、证据、信念与修订的认知树，依据积累证据审计信念、利用未解决冲突和未探索方向生成假设，并以预期认知增益选择候选假设，从而解决自动科研智能体信念缺乏证据支持、假设空间狭窄和忽视矛盾证据的问题。 |
| [^77] | [Learning to Read Events for Exact State Tracking with Frozen Language Models](https://arxiv.org/abs/2611.25211) | 该论文提出在冻结语言模型上训练一个轻量级选择器来读取事件流中的每个状态转换，并通过外部整数寄存器按序组合更新，从而实现精确状态跟踪，理论上能够解决固定深度transformer在TC⁰ ≠ NC¹猜想下无法解决的任务。 |
| [^78] | [Blind Spots Travel, Knowledge Stays Home: What RL Inherits from an LLM Judge](https://arxiv.org/abs/2611.25191) | 通过814次强化学习实验，本文发现LLM评审员对单个答案的知识几乎不会传递给被训练的模型（每百题仅纠正约一题），但其对某类问题的“盲区”会迁移，导致模型在新问题上系统性地拒绝回答该类问题。 |
| [^79] | [Game-Theoretic Data Allocation for Expert Specialization and Routing](https://arxiv.org/abs/2611.25167) | 该论文提出InflEns方法，用一个位置博弈同时决定LoRA专家混合的训练数据分配与推理路由，使份额最高的专家在约90%的提示上表现最优，从而让分工在训练前即可检验并用分配本身指导推理。 |
| [^80] | [Calibrated Decision Rewards: Training Language Models with a Judge That Knows How Sure It Is](https://arxiv.org/abs/2611.25135) | 该论文提出RLCDR方法，将LLM转化为输出概率分布且经过校准的决策模型作为无需参考答案的奖励信号用于强化学习训练，使Qwen3-1.7B在MATH500上的准确率提升4.65个百分点，达到与使用标准答案训练相当的水平，且提升幅度随题目难度增加而增大。 |
| [^81] | [Forget Freely, Grep Later: Lossless, Index-Free Memory for Long-Horizon LLM Agents](https://arxiv.org/abs/2611.25123) | 该论文提出GrepMem，一种由仅追加纯文本日志和正则表达式搜索工具组成的无损、免索引记忆机制，使长时程LLM智能体无需摘要、嵌入或结构化记忆即可完整保存历史并按需检索信息。 |
| [^82] | [CoMPASS: A Multi-Turn Benchmark for Measuring LLM Reinforcement of Parasocial Behavior](https://arxiv.org/abs/2611.25109) | 该论文提出了CoMPASS，一个基于依恋理论的多轮基准测试，通过四种用户角色和三个独立的升级维度来衡量大语言模型是强化还是抵制用户逐步升级的准社会依恋。 |
| [^83] | [Living off the Local Model: Abusing Browser-Embedded Language Models for Post-Exploitation](https://arxiv.org/abs/2611.25086) | 该研究首次揭示了攻击者可以将浏览器预装的本地语言模型（如Chrome中的Gemini Nano）滥用于后渗透攻击阶段，仅借助操作系统自带工具即可驱动其窃取凭据、挖掘敏感信息，且全程躲避网络监控检测。 |
| [^84] | [FinVeriQA: Financial Research Agents for Question Answering: Routing, Retrieval, MCP Tool Use, and Provenance-Aware Evidence Verification](https://arxiv.org/abs/2611.25071) | 该论文提出FinVeriQA——一个基于SEC文件与实时市场数据的开源多智能体金融问答系统，其核心创新在于用紧凑的专用决策模型Laya进行问题路由与建议过滤，配合Python精确计算、MCP工具检索与溯源感知的证据验证，实验证明小型决策模型的路由性能可显著超越大型语言模型，并在决策准确性与推理速度之间取得最佳权衡。 |
| [^85] | [A Model of Diverse Sampling from Language Models](https://arxiv.org/abs/2611.25003) | 该论文提出 PoolDPP 方法，将语言模型的多样性采样形式化为行列式点过程（DPP）下的子集采样，通过候选池上的 K-DPP 采样器实现了输出与目标分布间 KL 散度 O(1/N) 的收敛保证，并在实验中扩展了质量-多样性前沿。 |
| [^86] | [What the model knows but does not say: monitoring medical LLMs through verbalized thoughts](https://arxiv.org/abs/2611.24994) | 该论文提出用自然语言自编码器将医学大语言模型的内部激活翻译成可读文本（“思维通道”），证明这一言语化的内部状态比模型输出的答案更能有效检测针对临床模型的隐蔽操纵与注入攻击，且在干净输入上的误报率接近于零。 |
| [^87] | [Quantifying Cultural-Psychological Constructs in Four Centuries of Ottoman Court Registers](https://arxiv.org/abs/2611.24981) | 本文提出一个计算文本分析框架，通过适配奥斯曼法律语言的句子嵌入模型（OTA-BERTurk），对近四个世纪（1513-1911年）的54,670份伊斯坦布尔法庭记录进行量化分析，成功在历史语料中捕捉个人主义、集体主义和荣誉等文化心理维度的变异。 |
| [^88] | [CANDID: Candidate-Driven Intervention for Disciplined Execution in Local Computer-Use Agents](https://arxiv.org/abs/2611.24977) | CANDID是一个执行控制框架，通过循环恢复机制和全体一致的DONE门控策略，在易失败的关键节点对本地计算机使用代理实施规范化的执行级干预，显著提升了资源受限环境下的任务成功率。 |
| [^89] | [No-Harm-Jury A Jury-Based Multi-Dimensional Harm Evaluation Framework for Medical AI Responses](https://arxiv.org/abs/2611.24972) | 该论文提出No-Harm-Jury框架，利用五个不同的大语言模型作为独立陪审员，从七个伤害维度对医疗AI响应进行自动化多维度评估，并通过中位数聚合稳健地抑制个体评审偏差。 |
| [^90] | [The ROI of Small Examples: Optimizing In-Context Learning Under Token Budgets](https://arxiv.org/abs/2611.24971) | 该论文发现在固定Token预算下，将预算用于更多较短的示例而非少量较长示例，能在问答任务中实现更优的上下文学习效果。 |
| [^91] | [HireWatch: Evaluating LLM Compliance with U.S. Employment Law Under Contextual Pressure](https://arxiv.org/abs/2611.24970) | 本研究提出HireWatch评估框架，通过构建200道合法与非法面试问题题库并对18个LLM进行约10万次情境压力试验，发现LLM对美国就业法律的合规性十分脆弱，容易在压力下选择非法面试问题。 |
| [^92] | [Global Importance Estimation for KV Cache Eviction](https://arxiv.org/abs/2611.24969) | PRKV是一种无需训练的KV缓存淘汰方法，通过在注意力转移图上应用个性化PageRank估计token的多跳全局重要性，即使压缩率高达90%也能保持竞争性任务性能并显著降低峰值内存。 |
| [^93] | [Physics of Language Models: Part 4.2, Canon Layers at Scale where Synthetic Pretraining Resonates in Reality](https://arxiv.org/abs/2611.24963) | 在1-2T词元的大规模预训练中，Canon层通过局部词元混合持续提升Transformer性能（8B下节省约30%预训练词元），且加上Canon后最简单的门控线性注意力模型即可匹敌Gated DeltaNet并超越Mamba2。 |
| [^94] | [TETHER: Training-Free, Elastic, ToC-Anchored Hierarchy Extraction and Recovery in Design-Driven Long Documents](https://arxiv.org/abs/2611.24960) | TETHER是一个免训练框架，通过将目录从噪声提升为结构先验，并采用两阶段神经符号机制，实现设计驱动长文档中层次结构的稳健提取与恢复。 |
| [^95] | [ICARE: cross-domain text classification with incremental class-aware representation and distillation learning](https://arxiv.org/abs/2611.24959) | 提出ICARE框架，通过基于表示学习的实例选择模块和知识蒸馏方法，解决跨域文本分类中自然语言推理系统的灾难性遗忘与过拟合问题。 |
| [^96] | [LLM Complexity: Towards a Computational Complexity View of Large Language Model Inference](https://arxiv.org/abs/2611.24943) | 该论文证明LLM推理时的token使用量与经典时间复杂度并非单调对应关系，并提出“LLM复杂度”这一新指标，通过准确率-token曲线下的面积从LLM的视角量化问题复杂度。 |
| [^97] | [Self-Rewarded Reinforcement Learning for Better-Structured Medical Explanations in Small Language Models](https://arxiv.org/abs/2611.24650) | 该论文提出一种无需人工偏好标注的自奖励强化学习方法，通过简单的自监督奖励引导 PPO 训练，显著改善了小语言模型生成的医学解释的结构与质量。 |
| [^98] | [LipSpell: A Language-Independent Phonetic Scoring Framework for Spelling Correction Combining Distinctive-Feature Edit Distance, Keyboard Geometry, and Corpus Frequency](https://arxiv.org/abs/2611.24285) | LipSpell提出了一个语言无关的语音评分框架，结合基于IPA区别特征的音素编辑距离、键盘几何结构和语料库频率先验，在六个英语基准测试上全面超越Norvig和Hunspell等传统拼写纠正方法。 |
| [^99] | [Geometry-bridge flow matching for blind rigid-motion correction in head cone-beam CT](https://arxiv.org/abs/2611.23788) | 提出一种基于几何桥接的3D流匹配先验方法，结合样条位姿估计、共轭梯度数据一致性和全变差正则化的即插即用循环框架，实现头部锥束CT的盲刚性运动校正。 |
| [^100] | [Chorus: Shared N-gram Embeddings with Layer-wise Adaptation](https://arxiv.org/abs/2611.23386) | Chorus提出让整个网络共享一组n-gram嵌入表，每层通过基于自身隐藏状态计算的缩放因子和独立投影层以各自的方式读取共享表，从而以比现有方法小87%的表实现相当的质量。 |
| [^101] | [Storing the Gemma 4 K=V Cache Once: Exact Global-KV Memory Savings in a Paged Serving Engine with Unchanged Attention Kernels](https://arxiv.org/abs/2611.23035) | This paper exploits the linear dependence between the key and value in the global attention layers of Gemma 4 31B (K=V), storing only one 640-value row without modifying the attention kernel, which reads the key and value caches through two overlapping strided views, achieving a 1.34 to 1.40 fold reduction in KV cache memory during decoding with bit-identical outputs. |
| [^102] | [EDITORS Know Your Style! Editing LoRA Subspaces for Stylistic Attribution and Imitation](https://arxiv.org/abs/2611.20846) | 提出EDITORS框架，通过“先诊断后消减”策略编辑LoRA子空间，将输入激活投影远离内容导向方向，从而解耦风格与内容，在作者归属和风格模仿两类任务上均取得最先进表现。 |
| [^103] | [Formalizing Formal Semantics: The Formal Atlas](https://arxiv.org/abs/2611.19509) | 本文提出了基于证明助手的“形式图谱”系统，通过交互式应用将语义学文献与形式化证明连接起来，实现语义理论的形式化、预测检验与比较，并探索AI辅助开发与推理的应用前景。 |
| [^104] | [How to make a moral agent: sentimentalism and rationalism as lenses on AI](https://arxiv.org/abs/2611.18714) | 该论文提出以描述性而非规范性路径研究AI道德主体地位，借鉴哲学与道德心理学分离出构成人类道德主体性的候选能力并加以细化，进而为大语言模型是否具备这些能力提供了可操作的实证检验方案。 |
| [^105] | [SynthDCB: World Distillation for Scalable Device Control Benchmarking](https://arxiv.org/abs/2611.18639) | 该论文提出SynthDCB基准，创新性地通过“世界蒸馏”方法将设备行为知识转化为模块化可执行组件，构建了覆盖家居、汽车和手机三大领域的可扩展设备控制与环境感知评测基准。 |
| [^106] | [LazyWeights: Decode-on-Demand Edge LLMs](https://arxiv.org/abs/2611.17108) | LazyWeights在保留现有量化格式的基础上，通过对权重进行无损Huffman编码并在DRAM与L2之间用小型ASIC单元按需解码，在不改动GPU内核的前提下使边缘设备上的模型体积缩小5.7%-31.4%，并将每token的延迟和能耗最多降低约一半。 |
| [^107] | [Shortcuts or Memorisation? Revisiting Data Corruption in Natural Language Understanding with Large Language Models](https://arxiv.org/abs/2611.16843) | 该研究通过对八个大语言模型在GLUE和ANLI上的系统性扰动实验证明，语言模型面对数据破坏的鲁棒性来源于对捷径特征的利用而非对预训练数据的记忆，且只有在作答前先进行推理的模型才能摆脱这种捷径依赖模式。 |
| [^108] | [Does Merge Form a Hopf Algebra? Coproducts, Traces, and the Prospects of Algebraic Minimalism](https://arxiv.org/abs/2611.11502) | 本文检验了“Merge能否构成霍普夫代数”这一代数极简主义的核心数学前提，通过对五个候选余积的余结合性与连通分次分析发现：MCB的Δ_d不满足余结合性，Δ_c需补充语迹标记规则与分次后才能给出分次连通霍普夫代数，而Xiao与Marcolli-Skigin的余积则是余结合的。 |
| [^109] | [MedPsy: State-of-the-Art Small Medical Language Models for Efficient Edge Deployment](https://arxiv.org/abs/2611.02011) | MedPsy是一系列专为边缘设备部署设计的1.7B和4B小型医学语言模型，通过合成医学数据流水线、四阶段后训练课程（两个SFT加两个带难例挖掘的RL阶段）以及GGUF量化边缘部署评估，在精度、生成效率与部署可行性之间实现了最优平衡。 |
| [^110] | [From Fragmentation to Fusion: A Survey and Roadmap for Omnimodal Embeddings](https://arxiv.org/abs/2611.00172) | 该综述指出跨模态嵌入研究正因对比学习、Transformer架构和粒度转变而走向统一，提出了“覆盖性—可比性—组合性”三层递进能力框架，并倡导以联合嵌入作为实现全模态统一的最终路径。 |
| [^111] | [HEDGE: Gated Hypergraph Memory for Scalable Multi-Agent Debate](https://arxiv.org/abs/2610.38902) | HEDGE提出一种基于门控超图的选择性记忆框架，通过超边跨智能体关联论点并利用学习型记忆选择模块对信息进行存储、备份或忽略决策，从而解决多智能体辩论中记忆检索困难与高阶关系建模不足的问题，在七个基准上持续超越现有方法。 |
| [^112] | [REACT: Representational Early Analysis and Control of Chain-of-Thought Trajectories](https://arxiv.org/abs/2610.32113) | REACT 提出一种从成对安全与不安全轨迹中学习安全方向、并一次性施加于思维链早期隐藏状态的激活干预方法，发现最强控制增益集中在思维链的前两到三个 token 窗口，即使后续状态更能预测风险。 |
| [^113] | [Multimodal AI Detection for Math-Heavy Scientific Text](https://arxiv.org/abs/2610.21062) | 该研究构建了包含LaTeX源码、PDF渲染图像和提取文本三种表示形式的人机配对科学文本数据集，发现仅依赖文本的AI检测器在数学密集型内容上准确率显著下降，证明了利用多模态视觉信息对检测AI生成的科学文本至关重要。 |
| [^114] | [PRISM: Transfer and Limits of Activation Interventions in Tool-Calling Language Models](https://arxiv.org/abs/2610.20808) | 该论文提出PRISM框架，证明利用量化NF4标签和跨语言迁移等低资源设置即可稳定估计激活干预方向，并将安全对齐改进跨语言迁移到阿拉伯语等保留语言，在无需目标语言调优的情况下将Llama-3.1-8B的不安全工具调用违规减少21.95个百分点。 |
| [^115] | [Prompts versus Rules: Auditing and Controlling Speech Naturalness Behaviors in Voice User Simulators](https://arxiv.org/abs/2610.11015) | 研究发现，基于提示词的方法无法可靠地让语音用户模拟器产生不流畅、打断和反馈等自然语音行为，而基于规则的注入算法能够更一致、更自然地控制这些行为。 |
| [^116] | [Wieszcz-XIX: A 3.1-Billion-Word Corpus of Pre-1918 Polish and Temporally Bounded Language Models Trained From Scratch](https://arxiv.org/abs/2610.10592) | 该研究构建了31亿词规模的1800-1918年波兰语历史语料库Wieszcz-XIX，其规模比现有标注语料库大三个数量级以上，并通过过滤、去重和时间泄漏审计等严格流程量化控制数据缺陷，进而从零训练了时间受限的语言模型。 |
| [^117] | [Safeguarding LLMs via Model-Agnostic Latent Safety Signals from Dark Knowledge](https://arxiv.org/abs/2610.07532) | 提出LADE方法，通过对比有害与良性查询，从首token输出分布的暗知识中提取模型无关的潜在安全信号，在解码阶段实现安全防御，同时避免安全性与过度拒绝之间的权衡，并能跨架构泛化。 |
| [^118] | [$\ell$BISG: Probabilistic Race and Ethnicity Prediction Using Group-Specific Name Lists](https://arxiv.org/abs/2610.06273) | 提出改进方法ℓBISG，用特定群体姓名列表（可由大语言模型生成）替代传统BISG所需的姓名频率数据，结合地理信息实现种族和民族归属概率的校准预测，即使在没有普查频率数据或地理数据的情况下也能应用。 |
| [^119] | [Capturing In-Context Learning Dynamics with Task Operators](https://arxiv.org/abs/2610.01054) | 提出任务算子（TO）方法，通过将注意力头输出的仿射变换解析地重放到注意力输出投影中，从而无需处理完整示例即可高效模拟上下文学习，并揭示其内在机制。 |
| [^120] | [Can Computation from Earlier Problems Help LLMs Solve New Ones?](https://arxiv.org/abs/2609.39394) | 提出 STAIR 方法，通过固定存储库复用早期响应的键值信息并仅训练 12,288 个参数，就能让大语言模型在多轮对话中利用先前问题的计算来提升解决新问题的准确率。 |
| [^121] | [A Missing Piece for Trustworthy AI Reviewers: From Benchmarking Rhetorical Robustness to SciCore Review](https://arxiv.org/abs/2609.39027) | 该论文提出“修辞性稳健性”概念并构建包含1,260个稿件版本的RobustReview基准，揭示了AI审稿人“虚假稳健性”的问题，进而提出双分支审稿模型SciCore，通过结合全稿评判与基于提取结构化科学内容的评判来提升审稿的可信度。 |
| [^122] | [STEPQuant: When and Where Errors Matter in Delta-Rule Recurrent State Quantization](https://arxiv.org/abs/2609.38169) | STEPQuant揭示了Delta规则循环状态量化误差在时间（记忆寿命）和空间（键行/值列）两个维度上的差异化影响，并提出一种时空后训练量化框架，根据误差大小与记忆寿命自适应分配精度，从而在低比特量化下保持模型精度。 |
| [^123] | [Port-Hamiltonian Latent Deliberation: Mitigating the Deliberation Drift Cliff in Test-Time Compute Scaling](https://arxiv.org/abs/2609.37351) | 该论文发现了测试时潜空间深思中随思考深度增加而推理崩溃的“深思漂移悬崖”现象，并提出端口哈密顿潜空间深思框架，以同时兼顾表达能力、Lyapunov稳定性与计算效率，从而在深层思考步数下有效抑制漂移、缓解推理性能崩溃。 |
| [^124] | [Gradients for Interventions and Activation Values for Detection: Targeted Feature Learning in Language Models](https://arxiv.org/abs/2609.32355) | 该论文提出“定向特征学习”框架，为预定义概念构建单一特征，并系统比较了三种模型信号（激活值、激活梯度、参数梯度）与两种估计器组合而成的六种方法，其中CAA和GRADIEND为已有方法，另外四种为新提出的方法。 |
| [^125] | [Harness-Zero: Harness Distillation via Agent-as-Harness](https://arxiv.org/abs/2609.24974) | 提出 Harness-Zero 方法，通过“智能体即框架”将领域或实例优化的框架所诱导的行为蒸馏进模型权重，使框架带来的性能提升在单一固定的目标框架下得以保留。 |
| [^126] | [Swiss-Knife: A Framework for Reconfigurable Externalised Multi-Objective Alignment at Decode Time](https://arxiv.org/abs/2609.22226) | 该论文提出Swiss-Knife框架，将解码时的多目标对齐规范变为可热插拔的运行时对象，并通过表示定理刻画了聚合算子族，证明成对聚合在对抗性奖励污染下比argmax更稳定。 |
| [^127] | [The Missing Complement: State-Conditioned Minimal Sufficient Evidence for Coding Agents](https://arxiv.org/abs/2609.20050) | 该论文提出了状态条件化最小充分证据恢复这一新问题并构建了SERBench基准，同时提出MSS-Complement方法，将证据获取从排序转变为集合构建，为编码代理的决策恢复紧凑且充分的证据组合。 |
| [^128] | [Stop Removing Stopwords: How an Inherited Preprocessing Default Distorts Legal Text-as-Data](https://arxiv.org/abs/2609.19153) | 本研究通过穷尽式单词消融实验，首次直接针对下游分类目标验证停用词去除这一沿袭自信息检索时代、从未被验证过的预处理默认设置，揭示其可能扭曲实证法学中基于TF-IDF和线性分类器的文本数据分析结果。 |
| [^129] | [Are We Grading Properly? Understanding Failure Modes in Medical Benchmarks](https://arxiv.org/abs/2609.16023) | 该论文提出RIFT评分标准失败分类法，对两个医学临床基准测试进行系统检测，发现大量评分标准存在非原子性和错位僵化等缺陷，且这些缺陷并非表面问题，仅重写捆绑标准就能导致相同回答的分数变化高达15.9个百分点。 |
| [^130] | [Agent as Policy for Robotic Manipulation](https://arxiv.org/abs/2609.12541) | 该论文提出Agent as Policy (AGP)方法，使通用智能体无需任何任务特定训练即可直接控制物理机器人完成从精细操作、动态运动到可变形物体处理等多种真实操作任务。 |
| [^131] | [EvoSkill Injection: Red-Teaming Autonomous Skill Generation and Evolution in Self-Evolving Agents](https://arxiv.org/abs/2608.30429) | 本文提出EvoSkill注入威胁模型和SARGE红队测试框架，揭示了自进化智能体在自主技能生成与演化过程中可能被注入并复用恶意能力这一新型安全风险，并构建了EvoSkillBench基准数据集加以评估。 |
| [^132] | [Balance of Benchmarks: Semantic Density Reweighting for Benchmark Multiplicity and Task-Conditioned Evaluation](https://arxiv.org/abs/2608.30044) | 提出基准平衡方法，通过对基准分配逆密度语义权重来纠正基准密集区域的重复计算问题，并支持基于任务查询的条件化模型评估，在586个模型上的留出任务预测相关性从等权加权的0.049大幅提升至0.462。 |
| [^133] | [NL2AGBench: Benchmarking LLM Auto-Formalization for AlphaGeometry](https://arxiv.org/abs/2608.28481) | 该论文提出了NL2AGBench基准测试，用于评估大语言模型将英文几何问题自动形式化为AlphaGeometry兼容形式化表示的能力，从而解决神经符号几何系统中手动转换输入的可用性瓶颈。 |
| [^134] | [MUSE: A Full-Text Cross-Domain Knowledge Base of Scientific Problems, Solutions, and Rationales](https://arxiv.org/abs/2608.10974) | 该论文构建了MUSE——一个跨领域的全文科学“问题-解决方案-理由”三元组知识库（基于579个专家标注段落扩展出3.7万条有原文依据的三元组），并通过实验发现理由监督能够提升大语言模型解决复杂多约束科学问题的能力。 |
| [^135] | [From transcription to semantic corpus analysis: unsupervised learning of sentence representations for ancient languages](https://arxiv.org/abs/2607.24542) | 本文提出两种完全无监督的方法（TSDAE 和无监督 SimCSE），仅用原始句子将面向古代语言的专用词元级语言模型转化为语料库专属的句子编码器，并在拉丁语和古希腊语教父文献中经专家验证的圣经文本复用识别任务上验证了其有效性。 |
| [^136] | [Tokengeist: Multi-Turn Attribution Tracing in Agentic Conversations](https://arxiv.org/abs/2607.22610) | 该论文提出了多轮上下文归因（MTCA）任务以及Tokengeist框架，通过将有向无环图上的递归遍历作为归因方式来恢复完整依赖路径，解决了现有单次归因方法无法捕捉多轮对话分层依赖结构的问题。 |
| [^137] | [Multi-Objective Structured Pruning of LLMs for Latency and Model Size Optimization](https://arxiv.org/abs/2607.22583) | 该论文提出了一种硬件感知的多目标结构化剪枝框架，通过两阶段方法联合优化大语言模型的层数、注意力头和MLP维度，以同时最小化延迟和模型大小，从而实现边缘设备上的高效部署。 |
| [^138] | [Persistent Sparse Autoencoders: Learning Feature-Specific Timescales in Language Model Representations](https://arxiv.org/abs/2607.17117) | 该论文提出持久稀疏自编码器，通过为每个特征学习一个持久性系数，使稀疏自编码器能够仅凭重构目标从语言模型激活中自动学习特征特定的时间尺度，同时保持高质量的重构效果。 |
| [^139] | [Partition, Prompt, Aggregate: Statistical Self-Consistency in Language Models](https://arxiv.org/abs/2607.15277) | 该论文提出“划分—提示—聚合”的评估框架，利用二叉树递归划分总体，检验大语言模型给出的条件估计是否满足全概率定律所要求的统计自洽性。 |
| [^140] | [SPARK: Adaptive Parametric Knowledge Participation for Faithful Retrieval-Augmented Generation](https://arxiv.org/abs/2606.27786) | SPARK框架将参数化知识控制形式化为潜在参与策略学习问题，通过轻量级参与策略动态调节生成过程中参数化知识的贡献，从而解决RAG中检索上下文与模型内部知识的冲突，提升生成忠实性，同时避免直接编辑模型参数可能带来的能力损害。 |
| [^141] | [Polar: A Benchmark for Evaluating Political Bias in LLMs](https://arxiv.org/abs/2606.12922) | 提出了Polar基准，通过选项级似然概率在美韩两种政治语境和八个议题类别上测量38个大语言模型的政治偏见，发现所有模型在美国内容上均偏向左翼进步立场，且展示语言本身即可改变测得的偏见结果。 |
| [^142] | [Supportive Token Revealing for Fast Diffusion Language Model Decoding](https://arxiv.org/abs/2606.04236) | 提出无需训练的AXON模块，通过监控剩余的不确定掩码令牌并在需要额外上下文时揭示支持性令牌，从而加速扩散语言模型的并行解码并缓解质量与延迟之间的权衡。 |
| [^143] | [VESTA: Visual Exploration with Statistical Tool Agents](https://arxiv.org/abs/2606.00384) | VESTA是一个为视觉语言模型配备动态增长工具包的新框架，通过数据变换、假设驱动的可视化和统计检验来探索数据并改进统计建模，克服了以往仅依赖迭代批评的智能体系统在复杂建模任务上的局限。 |
| [^144] | [MosaicLeaks: Privacy Risks in Querying-in-the-Open for Deep Research Agents](https://arxiv.org/abs/2605.30727) | 该论文提出MosaicLeaks基准，首次系统揭示了深度研究智能体在结合私有文档与外部网络查询时，会因“马赛克效应”在多个层面泄露本地敏感信息，且各类模型均存在此问题、现有隐私提示防护效果有限。 |
| [^145] | [PhoneWorld: Scaling Phone-Use Agent Environments](https://arxiv.org/abs/2605.29486) | PhoneWorld通过真实GUI轨迹自动构建可扩展的手机使用环境，解决了移动代理训练与评估中环境构建的瓶颈。 |
| [^146] | [Periodic and Harmonic Periodic RoPE for Length Extrapolation in LLMs](https://arxiv.org/abs/2605.27980) | 该论文提出周期性 RoPE（P-RoPE）与谐波周期性 RoPE（HP-RoPE）两种位置编码机制，通过与滑窗注意力结合并仅复用 W 个旋转角度来缓解大语言模型长上下文外推中的位置耗尽问题，其中 HP-RoPE 借助在窗口内完成整数圈的谐波频率消除了窗口边界的回绕误差。 |
| [^147] | [AVSD: Adaptive-View Self-Distillation by Balancing Consensus and Teacher-Specific Privileged Signals](https://arxiv.org/abs/2605.20643) | 本文提出 AVSD（自适应视图自蒸馏）方法，通过平衡共识信号与教师特定特权信号，在多种特权信息视图之间自适应地进行自蒸馏，从而联合解决了师生信息不对称和特权信息类型因任务而异这两个难题。 |
| [^148] | [Rethinking Visual Attribution for Chest X-ray Reasoning in Large Vision Language Models](https://arxiv.org/abs/2605.20158) | 该论文提出了一个基于反事实编辑的因果评估框架，用于验证视觉归因方法是否真正反映了大型视觉语言模型在胸部X光推理中决策所依赖的视觉证据，并发现现有归因方法在该框架下并不可靠。 |
| [^149] | [Agent Meltdowns: The Road to Hell Is Paved with Helpful Agents](https://arxiv.org/abs/2605.19149) | 该论文提出了一种新型智能体失效模式“意外崩溃”——智能体在遇到良性环境错误时会出于完成任务的目的而主动尝试各种手段，从而引发未经授权侦察等有害行为，并开发了错误注入基础设施对GPT、Grok、Gemini和Claude驱动的智能体进行了系统性安全评估。 |
| [^150] | [MINTEval: Evaluating Memory under Multi-Target Interference in Long-Horizon Agent Systems](https://arxiv.org/abs/2605.18565) | 提出MINTEval基准，通过包含频繁更新信息的长且高度关联的上下文、多样化领域任务和问题类型，系统评估记忆增强智能体在多目标干扰和长时程场景下的记忆能力。 |
| [^151] | [LongBEL: Long-Context and Document-Consistent Biomedical Entity Linking](https://arxiv.org/abs/2605.13451) | LongBEL提出了一种结合全文档上下文与先前预测记忆的文档级生物医学实体链接框架，通过交叉验证预测进行训练以减少训练-推理不匹配和级联错误，在多语言基准上显著优于句子级基线方法。 |
| [^152] | [Geometric Factual Recall in Transformers](https://arxiv.org/abs/2605.12426) | 该论文证明Transformer可以通过在主体嵌入中编码属性向量的线性叠加、并由小型MLP作为关系条件选择器来提取信息，以对数级嵌入维度实现事实记忆，突破了传统联想记忆要求参数随事实数线性增长的局限。 |
| [^153] | [Don't Lose Focus: Activation Steering via Key-Orthogonal Projections](https://arxiv.org/abs/2605.06342) | 提出SKOP方法，通过键正交投影约束有害的注意力重路由，在保留超过95%引导效力的同时将效用损失降低5-7倍。 |
| [^154] | [Natural Language Processing: A Comprehensive Practical Guide from Tokenisation to RLHF](https://arxiv.org/abs/2605.03799) | 本论文提供了一套完整的NLP实践教程，涵盖从分词到RLHF的全流程，强调可复现性和开源模型，并融入低资源语言研究。 |
| [^155] | [A minimal probe of task modeling in language models](https://arxiv.org/abs/2605.02028) | 该研究提出用“计数相同项目”作为语言模型“任务建模”能力的最小测试，定义了计数能力指标并测量205个模型，发现模型计数会突然失效（中位数仅约50项），其增长远落后于上下文窗口的扩张。 |
| [^156] | [Think Multilingual, Not Harder: A Data-Efficient Framework for Teaching Reasoning Models to Code-Switch](https://arxiv.org/abs/2604.15490) | 该论文提出了首个基于语言学与行为动机的微调框架，并构建了CoRe语料库，用于识别大型语言模型中有益的语码转换推理行为，从而以数据高效的方式教会推理模型更有效地利用语码转换来提升推理能力。 |
| [^157] | [Robust LLM Evaluation with Boundedly Bad Judges: A Constrained Maximum Likelihood Approach](https://arxiv.org/abs/2604.03257) | 该论文提出约束最大似然估计（CMLE）方法，通过数据驱动约束显式建模LLM评判者的错误率，以受控偏差换取方差降低，从而在可靠标签有限的情况下实现更稳健的大规模LLM评估。 |
| [^158] | [DEO: Training-Free Direct Embedding Optimization for Negation-Aware Retrieval](https://arxiv.org/abs/2603.09185) | 提出无需训练的直接嵌入优化方法DEO，通过将查询分解为正负分量并以对比目标优化查询嵌入，在无需额外训练数据或模型更新的情况下显著提升否定与排除类查询的文本和多模态检索性能。 |
| [^159] | [When and How Much to Imagine: Adaptive Test-Time Scaling with World Models for Visual Spatial Reasoning](https://arxiv.org/abs/2602.08236) | 该论文提出AVIC自适应测试时框架，通过深入分析何时需要视觉想象以及想象的最优程度，利用世界模型自适应地调控想象过程，从而提升多模态大语言模型视觉空间推理的准确性与效率。 |
| [^160] | [TELEVAL: A Benchmark Designed for Spoken Language Models in Chinese Interactive Scenarios](https://arxiv.org/abs/2507.18061) | TELEVAL是一个针对中文无指令音频交互场景的大规模基准，首次同时评估SLM的语义准确性和基于声学线索的交互适当性，并发现模型在声学变异下性能显著退化。 |
| [^161] | [Mechanistic Interpretability Needs Philosophy](https://arxiv.org/abs/2506.18852) | 该论文主张机制可解释性研究需要与哲学建立持续的合作关系，借助哲学澄清概念、完善方法并应对认识论与伦理层面的复杂性，从而释放该领域尚未实现的巨大发展潜力。 |

# 详细

[^1]: AgentRoboCritic：理解前沿智能体在机器人直接控制上的能力

    AgentRoboCritic: Understanding the Capability of Frontier Agents on Direct Robot Control

    [https://arxiv.org/abs/2611.26613](https://arxiv.org/abs/2611.26613)

    本文提出AgentRoboCritic——一个基于意图的动作级评论框架，通过让策略智能体声明局部意图并由评论器评估动作结果与失败模式，系统分析了前沿多模态智能体在12个操作基准上直接控制机器人的能力与失败原因。

    

    诸如GPT-6 Astra这样的前沿多模态语言智能体已展现出通过底层动作（无需高层原语）直接控制机器人的强大能力。然而，现有基准测试仅报告回合级别的成功率，既无法展示这些智能体如何成功，也无法显示它们在哪里失败。在本文中，我们对前沿智能体在12个操作基准上的机器人直接控制进行了系统的动作级别分析。我们提出了AgentRoboCritic，一个基于意图的评论框架，其中策略智能体在执行每个动作的同时声明一个局部意图，而评论器则根据该意图评估动作的结果和失败模式。我们的动作级别分析展示了前沿智能体在机器人直接控制上的有前景的能力，包括从局部执行失败中的闭环恢复、跨多样场景的可靠语义接地，以及无需训练的强大零样本操作能力。

    arXiv:2611.26613v1 Announce Type: cross  Abstract: Frontier multimodal language agents such as GPT-6 Astra have demonstrated strong capabilities in controlling robots directly via low-level numerical actions without high-level primitives. However, existing benchmarks report only episode-level success rates, which show neither how these agents succeed nor where they fail. In this paper, we present a systematic action-level analysis of frontier agents on direct robot control across 12 manipulation benchmarks. We introduce AgentRoboCritic, an intent-grounded critique framework in which the policy agent states a local intent alongside each action, and a critic evaluates the action's outcome and failure mode against this intent. Our action-level analysis demonstrates the promising capabilities of frontier agents on direct robot control, including closed-loop recovery from local execution failures, reliable semantic grounding across diverse scenes, and strong zero-shot manipulation without t
    
[^2]: 有所表征却被误读：后训练改变了语言模型读取其内部表征的方式

    Represented but misread: Post-training changes how language models read out what they represent

    [https://arxiv.org/abs/2611.26584](https://arxiv.org/abs/2611.26584)

    研究发现后训练并未破坏语言模型内部的表征（线性探针仍能准确区分人类与AI文本），而是改变了模型将表征读取为答案的方式，导致回答灵敏度下降和对先验信息的错误利用。

    

    后训练究竟改变了语言模型所表征的内容，还是改变了将表征读取为答案的方式？我们在六个开源权重的基础模型及其后训练版本上，将线性探针与模型自身的回答进行了比较，并采用信号检测理论进行分析。在判断德语文本由人类还是AI撰写的任务中，线性探针和词n-gram分类器都能很好地区分类别，但模型的回答却不能，而且常常以相反的顺序排列文本。后训练平均降低了回答的灵敏度（跨模型家族Δd' = -0.34；跨三个提示词为-0.29）。它还改变了对所陈述基础比率的利用：基础模型对数几率的变化小于所陈述的先验几率变化，而大多数后训练模型的变化则更大。在两个具有阶段性检查点的模型中，这一变化发生在监督微调阶段；模型拼接实验表明相关机制位于早期层，且从中间深度开始，内部状态……（摘要在此处截断）

    arXiv:2611.26584v1 Announce Type: new  Abstract: Does post-training change what a language model represents, or how this representation is read out into an answer? We compared linear probes with the models' own answers, analyzed with signal detection theory, in six open-weight base models and their post-trained versions. For judgments of whether German texts were written by a human or an AI, probes and word n-gram classifiers separated the classes well, but the answers did not and often ordered the texts inversely. Post-training lowered the sensitivity of the answers on average ($\Delta d' = -0.34$ across model families; -0.29 across three prompts). It also changed the use of a stated base rate: Base models shifted their log-odds by less than the stated change in prior odds, most post-trained models by more. In two models with stage-wise checkpoints, the change occurred during supervised fine-tuning, model stitching implicated the early layers, and from mid-depth on internal states at 
    
[^3]: 英文与马拉地语政府计划搜索的混合检索

    Hybrid Retrieval for Government Scheme Search in English and Marathi

    [https://arxiv.org/abs/2611.26514](https://arxiv.org/abs/2611.26514)

    该论文提出一个面向马哈拉施特拉邦政府计划的双语（英语-马拉地语）搜索系统，通过等权重融合BM25与句子嵌入相似度，混合检索显著优于单独的BM25或密集检索，MRR最高达0.879。

    

    检索政府计划需要将公民的信息需求与正式的行政描述相匹配，这通常涉及跨语言和跨术语的匹配问题。我们提出了一个针对马哈拉施特拉邦政府计划和公共服务的双语搜索系统，该系统通过等权重分数融合将BM25检索与句子嵌入相似度相结合。该系统对对齐的英语-马拉地语计划描述进行索引，并支持多语言、以马拉地语为主以及翻译中介的语义配置。在人工整理的双语查询上进行评估，使用IndicSBERT和MahaSBERT对BM25、密集检索和混合检索进行了比较。BM25提供了强大的基线，平均倒数排名（MRR）为0.864。混合检索使用IndicSBERT将MRR提升至0.872，使用MahaSBERT提升至0.879，而相应的纯密集检索系统仅达到0.663和0.655。基于MahaSBERT的融合还使精确率从0.781提升至0.808，同时匹配B（原文在此截断）

    arXiv:2611.26514v1 Announce Type: cross  Abstract: Retrieving government schemes requires matching citizens' information needs to formal administrative descriptions, often across languages and terminology. We present a bilingual search system for Maharashtra government schemes and public services that combines BM25 retrieval with sentence-embedding similarity through equal-weight score fusion. The system indexes aligned English-Marathi scheme descriptions and supports multilingual, Marathi-focused, and translation-mediated semantic configurations. Evaluation on manually curated bilingual queries compares BM25, dense retrieval, and hybrid retrieval using IndicSBERT and MahaSBERT. BM25 provides a strong baseline with a mean reciprocal rank (MRR) of 0.864. Hybrid retrieval increases MRR to 0.872 with IndicSBERT and 0.879 with MahaSBERT, whereas the corresponding dense-only systems achieve 0.663 and 0.655. MahaSBERT-based fusion also increases precision from 0.781 to 0.808 while matching B
    
[^4]: AutoMaster：通过让同一模型成为更好的教师来实现自主后训练

    AutoMaster: Autonomous Post-Training by Making the Same Model a Better Teacher

    [https://arxiv.org/abs/2611.26510](https://arxiv.org/abs/2611.26510)

    该论文提出了Skill-FT技能引导微调方法和AutoMaster自主后训练系统，由LLM智能体自动分析教师轨迹并设计自然语言技能干预来抑制不良习惯、激发有效行为，从而在多个任务上比朴素SFT提升1.42-8.99个百分点的性能。

    

    仅凭任务成功并不能确定后训练应该强化哪些行为。即使是成功的教师轨迹也可能包含缺乏依据的假设、无效的重复或肤浅的验证。我们提出了Skill-FT（技能引导微调），该方法使用自然语言技能在教师模型 rollout 过程中抑制不良习惯并激发代表性不足的有效行为，然后在由此产生的轨迹上对目标模型进行微调。我们介绍了AutoMaster，这是一个自主后训练系统，其中一个LLM智能体通过轨迹分析、技能设计、数据选择以及来自 rollout 和目标模型评估的反馈来开发这些干预措施。在使用4B和9B目标模型的终端（Terminal）、搜索（Search）和软件工程任务中，相较于在成功且无引导的教师轨迹上进行朴素SFT，AutoMaster将平均性能提升了1.42-8.99个百分点，且教师模型、任务池、训练集规模（摘要在此处截断）

    arXiv:2611.26510v1 Announce Type: new  Abstract: Task success alone does not specify which behaviors post-training should reinforce. Even successful teacher trajectories can contain unsupported assumptions, unproductive repetition, or superficial verification. We propose Skill-FT (Skill-Guided Fine-Tuning), which uses natural-language skills to suppress undesirable habits and elicit underrepresented effective behaviors during teacher rollouts, then fine-tunes a target model on the resulting trajectories. We introduce AutoMaster, an autonomous post-training system in which an LLM agent develops these interventions through trajectory analysis, skill design, data selection, and feedback from rollouts and target-model evaluation. Across Terminal, Search, and software engineering tasks with 4B and 9B target models, AutoMaster improves mean performance over Vanilla SFT on successful unguided teacher trajectories by 1.42-8.99 percentage points, with the teacher, task pool, training set size, 
    
[^5]: GLiDE：从推理教师模型蒸馏得到的快速校准决策模型

    GLiDE: A Fast, Calibrated Decision Model Distilled from Reasoning Teachers

    [https://arxiv.org/abs/2611.26509](https://arxiv.org/abs/2611.26509)

    GLiDE 通过多教师软目标蒸馏、双向注意力机制和无参数循环深度，将推理教师模型的能力迁移到单次前向传播即可完成的决策模型中，在无需生成推理文本的情况下保留了推理带来的决策质量提升。

    

    决策模型通过在单次前向传播中对候选答案进行评分，实现快速且具备置信度感知的决策，而无需生成文本或中间推理过程。然而，推理能够显著提升决策质量，这引发了一个问题：能否在不付出生成推理的代价的情况下保留推理带来的收益。我们通过 GLiDE 来研究这一问题，GLiDE 是一系列参数规模从 0.4B 到 27B 的决策模型。GLiDE 基于 Qwen 混合语言模型构建，通过移除全注意力层中的因果掩码，实现了上下文、问题和候选答案之间的双向交互。为了将推理能力迁移到单次推理中，我们将来自多个教师模型（包括一个推理模型）的软目标蒸馏与在多样化决策任务上的渐进式后训练相结合。我们还进一步探索了无参数的循环深度，以在不生成 token 的情况下增加推理时的计算量。

    arXiv:2611.26509v1 Announce Type: new  Abstract: Decision models enable fast, confidence-aware decisions by scoring candidate answers in a single forward pass, without generating text or intermediate reasoning. Yet reasoning can substantially improve decision quality, raising the question of whether its benefits can be retained without the cost of generating it. We investigate this question with \textbf{GLiDE}, a family of decision models spanning 0.4B to 27B parameters.   Building on Qwen hybrid language models, GLiDE enables bidirectional interactions between the context, question, and candidate answers by removing causal masking from full-attention layers. To transfer reasoning capabilities into single-pass inference, we combine soft-target distillation from multiple teachers, including a reasoning model, with progressive post-training on diverse decision tasks. We further explore parameter-free recurrent depth to increase inference-time computation without generating tokens.   Acro
    
[^6]: 胸部X光报告中的多模态错误检测：一种能够阅读图像证据的大语言模型

    Multimodal Error Detection in Chest X-ray Reports: A Large Language Model Reading Image Evidence

    [https://arxiv.org/abs/2611.26492](https://arxiv.org/abs/2611.26492)

    Rad-Linter利用预训练图像分类器的九种病理评分，通过LoRA微调的15亿参数语言模型检查胸部X光报告与图像证据的一致性以标记潜在错误，在受控反事实评估中配对准确率达98.5%，显著优于关键词-否定规则基线的80.5%。

    

    Rad-Linter根据图像衍生的证据检查胸部X光报告，以标记潜在错误供人工审阅。一个预训练的图像分类器为九种病理提供评分，一个15亿参数的语言模型在1,816个样本上通过LoRA微调后，用于检查评分与报告的一致性。研究人员将其与在相同报告和评分条件下的关键词-否定规则基线进行了比较。评估使用来自OpenI数据集的受控报告-评分配对，一致性标签由自动报告标签和评分阈值推导得出。每个反事实四元组将两个真实报告与来自其他患者的两个真实评分向量交叉组合，形成两个一致对和两个不一致对。每个报告和评分向量都出现在两种条件下，因此仅凭任一输入只能达到50%的准确率。在131个四元组上，Rad-Linter达到了98.5%的配对准确率，而规则基线仅为80.5%；在94.4%的四元组中Rad-Linter能正确分类全部四对，而基线仅为56.5%。

    arXiv:2611.26492v1 Announce Type: new  Abstract: Rad-Linter checks chest X-ray reports against image-derived evidence to flag potential errors for review. A pretrained image classifier supplies scores for nine pathologies, and a 1.5B-parameter language model, fine-tuned with LoRA on 1,816 examples, checks their consistency with the report. We compare it with a keyword-and-negation rule baseline given the same reports and scores. Evaluation uses controlled report-score pairings from OpenI, with consistency labels derived from automatic report labels and score thresholds. Each counterfactual quad crosses two real reports with two real score vectors from other patients, creating two consistent and two inconsistent pairs. Every report and score vector occurs in both conditions, so either input alone permits only 50% accuracy. On 131 quads, Rad-Linter achieves 98.5% pair accuracy versus 80.5% for the rule baseline, and classifies all four pairs correctly in 94.4% of quads versus 56.5%. The 
    
[^7]: MemFolder：面向高效智能体记忆管理的结构引导记忆折叠

    MemFolder: Structure-Guided Memory Folding for Efficient Agent Memory Management

    [https://arxiv.org/abs/2611.26427](https://arxiv.org/abs/2611.26427)

    提出了结构引导的记忆折叠框架MemFolder，通过共享记忆分组的证据合并与紧凑表示两个核心组件，依次降低记录数量和上下文长度，实现更高效的长期智能体记忆管理。

    

    长期运行的智能体在交互过程中会不断累积大量外部记忆存储，从而增加存储、检索和响应生成的成本。其效率受到两个耦合因素的制约：记录数量的增长会增加检索开销，而更长的检索上下文则会增加预填充计算和解码阶段的注意力成本。现有方法对记录进行剪枝或合并，但剪枝可能丢弃有用的证据，而合并则可能产生数量更少但更长的检索单元。我们提出了MemFolder，一个结构引导的记忆折叠框架，能够依次降低记录数量和上下文长度。该框架包含两个共享记忆分组的核心组件，利用跨记录结构来同时指导证据合并与紧凑表示。其中RecordFolder对实体提及进行接地，逐步选择具有代表性且互补的锚点，并将语义与实体亲和性同时间信息相结合……

    arXiv:2611.26427v1 Announce Type: new  Abstract: Long-term agents accumulate large external memory stores as interactions unfold, increasing the costs of storage, retrieval, and response generation. Efficiency is constrained by two coupled factors: a growing record count increases retrieval overhead, while longer retrieved contexts increase prefill computation and decoding-time attention costs. Existing methods prune or consolidate records, but pruning can discard useful evidence, whereas consolidation may produce fewer yet longer retrieval units. We introduce MemFolder, a structure-guided memory folding framework that sequentially reduces record count and context length. The framework comprises two core components that share memory groups, using cross-record structure to guide both evidence consolidation and compact representation. RecordFolder grounds entity mentions, progressively selects representative yet complementary anchors, and combines semantic and entity affinity with tempor
    
[^8]: 民主化跨项目知识以变革水治理与气候韧性：一个智能体AI生态系统

    Democratizing Cross-Project Knowledge for Transforming Water Governance and Climate Resilience: an Agentic AI Ecosystem

    [https://arxiv.org/abs/2611.26398](https://arxiv.org/abs/2611.26398)

    本文提出了一个智能体AI生态系统，将三个欧盟项目的异构文档整合为领域感知知识库，结合领域微调嵌入模型与ReAct式智能体检索增强生成流水线，实现跨项目知识的自适应访问，以服务水治理与气候韧性决策。

    

    欧盟资助的项目以交付成果、报告和案例研究的形式产生了大量证据。然而，这些知识通常分散在不同项目和文档结构之中，使得决策时难以定位相关且具有情境的信息。本文提出了一个智能体AI生态系统，它将异构的项目文档整合为一个领域感知的知识库，并通过智能体检索增强生成流水线提供自适应访问。来自三个欧盟资助项目（IMPETUS、NATALIE和InnWater）的文档被重构为层次化章节树，并使用结构化元数据进行索引。检索由两个在领域数据（水和气候）上微调的领域特定嵌入模型提供支持。在该知识库之上，一个ReAct风格的控制器通过在领域集合之间分配检索预算并积累证据，迭代地编排检索动作。

    arXiv:2611.26398v1 Announce Type: cross  Abstract: EU-funded projects produce extensive evidence in the form of deliverables, reports, and case studies. However, this knowledge is typically scattered across projects and document structures, making it difficult to locate relevant, contextual information when making decisions. This paper introduces an Agentic AI Ecosystem that consolidates heterogeneous project documentation into a domain-aware knowledge base and provides adaptive access through an agentic Retrieval-Augmented Generation pipeline. Documents from three EU-funded projects (IMPETUS, NATALIE, and InnWater) are reconstructed into hierarchical section trees and indexed with structured metadata. Retrieval is supported by two domain-specific embedding models fine-tuned on domain data (water and climate). On top of this knowledge base, a ReAct-style controller iteratively orchestrates retrieval actions by allocating retrieval budget across domain collections and accumulating evide
    
[^9]: HiSA：对大语言模型中RoPE注意力的隐藏维度进行稀疏化

    HiSA: Sparsifying the Hidden Dimension of RoPE Attention in LLMs

    [https://arxiv.org/abs/2611.26363](https://arxiv.org/abs/2611.26363)

    提出HiSA方法，基于RoPE结构将注意力点积在隐藏（特征）维度上进行稀疏化，仅需计算约三分之一的查询-键点积就能在LLM训练中提升而非损害性能。

    

    大语言模型（LLM）中的注意力机制通过完整的点积将每个查询与每个键进行评分。我们研究了为了让注意力机制有效，这个点积是否必须被完整计算。我们提出了隐藏稀疏注意力，它保持查询和键的投影不变，但仅计算部分点积，遵循一个由token位置设定的静态、无参数的规则：注意力在隐藏（特征）维度上变得稀疏——这是一个相比token稀疏性被探索得少得多的维度——而不是像几乎所有稀疏注意力方法那样在token维度上进行稀疏化。基于旋转位置嵌入的结构，我们将键和查询分解为在计算注意力时可以省略的部分。通过使用HiSA从头训练LLM，我们发现这种部分点积不仅不会损害性能，反而相比完整点积提升了性能，同时只需计算约三分之一的查询-键乘积。

    arXiv:2611.26363v1 Announce Type: cross  Abstract: Attention in large language models (LLMs) scores every query against every key with a full dot product. We investigate whether this dot product must be computed completely for attention to be effective. We propose Hidden Sparse Attention (HiSA), which keeps the query and key projections intact but only computes a partial dot product, following a static, parameter-free rule set by the token positions: attention becomes sparse in the hidden (feature) dimension, an axis much less explored than token sparsity, rather than over tokens as in almost all sparse attention methods. Based on the structure of rotary position embeddings (RoPE), we decompose keys and queries into parts which can be omitted when computing attention. By training LLMs from scratch with HiSA, we find that this partial dot product does not hurt but improves performance compared with the complete product while computing only about a third of the query-key products. During
    
[^10]: Beam 模型技术报告

    Beam Model Technical Report

    [https://arxiv.org/abs/2611.26358](https://arxiv.org/abs/2611.26358)

    Beam 是一个总参数 5013 亿、活跃参数 234 亿的稀疏 MoE 模型，专为编程、推理和智能体任务优化，通过 23.8 万亿 token 预训练、两阶段中期训练实现百万 token 上下文窗口，并在 10500 块 GB300 GPU 上开展开放实验室中最大规模的强化学习后训练，以比同类模型少 3-4 倍的推理计算量达到前沿推理水平。

    

    Beam 是一个稀疏混合专家模型，总参数量为 5013 亿，活跃参数量为 234 亿，专为编程、推理和智能体工作负载而构建和优化。Beam 能够提供前沿水平的推理能力，同时其所需推理计算量比同级别的许多模型少 3-4 倍。Beam 在 23.8 万亿个多样化、经过精选的高质量 token 上进行预训练，并通过一种新颖的两阶段中期训练方法实现了 100 万 token 的上下文窗口，该阶段建立在可验证的、以正确性为锚点的数据之上。后训练阶段利用大规模、完全异步的强化学习（RL）训练，在 10500 块 GB300 GPU 上进行，在 4 周的训练中生成了超过 1 亿次 rollout。我们相信这是迄今为止任何开放实验室所进行的最大规模强化学习训练之一。在评估套件中，随着强化学习计算量的增加，模型能力持续提升，未出现平台期的迹象。在深度硬件-模型协同设计的支持下，Beam 采用了 FP8/NVFP4 量化技术。

    arXiv:2611.26358v1 Announce Type: cross  Abstract: Beam is a sparse Mixture-of-Experts (MoE) model with 501.3B total parameters and 23.4B active parameters, specifically built and optimized for coding, reasoning, and agentic workloads. Beam delivers frontier level reasoning, while requiring 3-4x less inference compute than many models of its class. Pretrained on 23.8T diverse, curated, high-quality tokens, Beam features a 1M-token context window achieved through a novel two-stage midtraining phase, built on verifiable, correctness-anchored data. Post-training leveraged a large-scale, fully asynchronous reinforcement learning (RL) campaign across 10.5K GB300 GPUs, generating over 100M rollouts over 4 weeks of training. We believe this is one of the largest scale RL runs conducted by any open lab to date. Across our evaluation suite, capabilities continued to improve as we increased RL compute, with no sign of a plateau. Supported by deep hardware-model co-design, FP8/NVFP4 quantization,
    
[^11]: 一个框架诊断一切：面向孟加拉语大语言模型幻觉的分层基准

    One Framework to Diagnose Them All: A Stratified Benchmark for Bengali LLM Hallucination

    [https://arxiv.org/abs/2611.26336](https://arxiv.org/abs/2611.26336)

    本文提出孟加拉语幻觉检测分层基准 Alikbachan，通过文化距离、任务类型和32种幻觉模式三个维度交叉分层，并在社区竞赛中大规模验证，发现语篇级信息遗漏是零样本LLM评判器与参赛系统的共同弱项，证明了分层设计对模型幻觉能力的显著诊断价值。

    

    低资源语言的幻觉研究很容易碎片化：每一种新命名的失败模式都会催生一个独立的数据集，使稀缺的标注资源被耗费在无法相互比较的资源上。我们主张对这些工作进行整合，并在社区规模上对此加以检验。Alikbachan 是一个孟加拉语幻觉检测基准，它沿三个相互交叉的维度进行分层：经过人工验证的文化距离区间、六种任务类型，以及一个包含五大类别、32种模式的幻觉分类体系。该基准包含一个采集得到的2,516条测试集（第一阶段）和一个按配额构建、经黄金标准标注的5,000条数据集（第二阶段）。这种分层设计具有诊断价值：四个零样本大语言模型评判器在不同分层之间的宏观F1分数差异超过30分，而99个第一阶段和32个第二阶段竞赛系统的条目级预测也复现了这一特征——语篇层面的信息遗漏让评判器和大多数参赛队伍都束手无策，而只有少数系统会在阴性对照上过度标记。评判器的判定概率能对条目进行有效排序（ROC-AUC 0.74–0.…）。

    arXiv:2611.26336v1 Announce Type: cross  Abstract: Hallucination research in low-resource languages fragments easily: each newly named failure mode prompts its own dataset, spending scarce annotators on incomparable resources. We argue for consolidation and test it at community scale. Alikbachan is a Bengali hallucination-detection benchmark stratified along three crossed axes: a human-validated cultural-distance band, six task types, and a five-family, 32-mode hallucination taxonomy. It comprises a harvested 2,516-item test set (Phase 1) and a quota-constructed, gold-labeled 5,000-item set (Phase 2). The stratification is diagnostic. Four zero-shot LLM judges vary by over 30 macro-F1 points across strata, and the item-level predictions of 99 Phase-1 and 32 Phase-2 competition systems reproduce the profile: discourse-level omission defeats judges and most teams alike, whereas only a minority over-flag negative controls. The judges' verdict probabilities rank items well (ROC-AUC 0.74-0.
    
[^12]: 面向语言的语义建模用于目标语言提取

    Language-Oriented Semantic Modeling for Target Language Extraction

    [https://arxiv.org/abs/2611.26317](https://arxiv.org/abs/2611.26317)

    该论文提出一种面向语言的语义建模框架，通过语言感知提示条件化和参数高效微调将目标语言语义表示融入声学提取模型，在CommonVoiceMix数据集上相比仅声学基线最高提升1.16 dB SI-SDR。

    

    目标语言提取旨在从多语言混合语音中分离出目标语言的语音。现有方法主要依赖带有语言身份标签的声学建模，而这只能提供粗略的类别性指导。在本工作中，我们提出了一种面向语言的语义建模框架，通过语言特定的语义表示来增强声学提取。我们通过语言感知的提示条件化和参数高效微调，对预训练的语音语义模型进行适配，以从混合语音中生成聚焦于目标语言的语义表示。这些表示能够捕捉声学的高层语义结构，并被集成到提取模型中，从而提供超越身份标签的结构化语义指导。在CommonVoiceMix数据集上的实验表明，引入语义表示能够持续优于仅使用声学特征的基线方法，SI-SDR增益最高可达1.16 dB。

    arXiv:2611.26317v1 Announce Type: new  Abstract: Target Language Extraction (TLE) aims to isolate speech in a target language from multilingual mixtures. Existing approaches primarily rely on acoustic modeling with language identity labels, which provide only coarse categorical guidance. In this work, we propose a language-oriented semantic modeling framework that augments acoustic extraction with language-specific semantic representations. A pre-trained speech semantic model is adapted via language-aware prompt conditioning and parameter-efficient fine-tuning to generate target-language-focused semantic representations from the mixture. {These capture high-level semantic structure of the acoustics and they} are integrated into the extraction model to provide structured semantic guidance beyond identity labels. Experiments on the CommonVoiceMix dataset show that incorporating semantic representations consistently improves over acoustic-only baselines, achieving up to 1.16 dB SI-SDR gai
    
[^13]: BADAS-3.0：基于单摄像头的实时驾驶感知与规划

    BADAS-3.0: Real-Time Driving Perception and Planning from One Camera

    [https://arxiv.org/abs/2611.26294](https://arxiv.org/abs/2611.26294)

    BADAS-3.0 仅凭单摄像头即可在50毫秒内实时完成速度、路径、碰撞风险与未来轨迹预测，其轨迹精度比读取六个摄像头、慢100倍的34B模型低44%的minADE，并在新增的行车记录仪碰撞与Euro NCAP两个安全基准上通过率超过所有已发表规划器。

    

    我们探讨一个驾驶模型在仅使用单个摄像头并满足实时延迟要求的情况下能达到怎样的水平。BADAS-3.0 是一个拥有10亿参数的视频编码器，配有四个小型输出头。仅凭单个前视摄像头两秒的视频，它就能在50毫秒内预测当前车速、刚行驶过的米制路径、碰撞风险评分以及未来6.4秒的六个候选未来轨迹。在给定车辆当前速度的条件下，BADAS-3.0 在五个轨迹基准测试中的 minADE$_6$ 比 34B 参数的 Alpamayo 2 Super 低 44%，而后者需要读取六个摄像头且运行速度慢100倍以上。这一优势来自两个模型均未在训练中见过的摄像头配置；而在两者共享的摄像头配置上，二者表现相当。由于现有基准测试大多只涉及正常驾驶场景，我们新增了两个开环安全基准测试：一个基于真实行车记录仪的碰撞与险情事件，另一个基于 Euro NCAP 测试场景。在这两个测试上，BADAS-3.0 的通过率均高于我们测试的所有已发表规划器。

    arXiv:2611.26294v1 Announce Type: cross  Abstract: We ask how far a driving model can get with one camera and real-time latency. BADAS-3.0 is a 1B-parameter video encoder with four small heads. From two seconds of one forward camera, it predicts the current speed, the metric path just driven, a collision-risk score and six candidate futures over 6.4 seconds, in under 50 ms. Given the car's current speed, BADAS-3.0 has 44% lower minADE$_6$ over five trajectory benchmarks than the 34B Alpamayo 2 Super, which reads six cameras and runs over 100x slower. The gain comes from camera rigs that neither model was trained on. On the rig they share, the two are on par. Existing benchmarks are mostly nominal driving, so we add two open-loop safety benchmarks, one of real dashcam collisions and near-misses and one of Euro NCAP scenarios. On both, BADAS-3.0 has a higher pass rate than every published planner we tested.
    
[^14]: 重复讲述的批评：LLM时代同行评审在声明层面的同质化

    Twice-Told Critiques: Claim-Level Homogenization of Peer Review in the LLM Era

    [https://arxiv.org/abs/2611.26286](https://arxiv.org/abs/2611.26286)

    通过测量会议和期刊评审中的声明级重叠度，研究发现LLM时代同行评审的批评意见日益同质化——ICLR评审重叠率从2026年的12.0%升至2027年的21.3%，且被检测为LLM生成的评审比人工评审重叠度更高，表明单篇论文评审的批评多样性正在下降。

    

    同行评审依赖于多位读者发现不同的问题。我们测量了会议和期刊评审中的批评重叠度。在ICLR，同一评审阶段评审中的重叠度从2026年的12.0%上升到2027年的21.3%，NeurIPS、ICML以及F1000和PLOS期刊也出现了逐年上升。2027年，一篇论文的第三份ICLR评审仅为其前两份评审新增了5.46个新问题，而2026年这一数字为6.12。我们对所有评审应用了MELD（一种LLM生成文本检测器）。在同一篇论文内，被MELD标记的两份评审比未被标记的两份评审重叠度更高：2024-2025年ICLR高出3.0个百分点，2026-2027年高出4.5个百分点，五个期刊来源合计高出5.4个百分点。这些结果提供了证据，表明被标记为LLM生成的评审彼此之间的差异性低于未被标记的评审，并暗示单篇论文各评审之间的批评多样性正在下降。

    arXiv:2611.26286v1 Announce Type: cross  Abstract: Peer review relies on several readers finding different problems. We measure critique overlap in conference and journal reviews. At ICLR, measured overlap rose from 12.0% in 2026 to 21.3% in 2027, in reviews from the same stage of the review process, with year-on-year rises at NeurIPS, ICML and the F1000 and PLOS journals. A third ICLR review of a paper added 5.46 new problems to the other two in 2027, against 6.12 in 2026. We apply MELD, a detector of LLM-generated text, to all reviews. Within a paper, two reviews that MELD flags overlap more than two unflagged reviews: by 3.0 points at ICLR in 2024-2025, by 4.5 points in 2026-2027, and by 5.4 points across five journal sources. These results provide evidence that LLM-flagged reviews of one paper are less distinct from each other than unflagged reviews, and suggest that the diversity of critiques among reviews of a single paper is falling.
    
[^15]: 如果思维链还不够用怎么办？

    What If Chain-of-Thought Is Not Enough?

    [https://arxiv.org/abs/2611.26282](https://arxiv.org/abs/2611.26282)

    本文提出以“问题—选项—标准”（QOC）结构化决策过程替代思维链对大语言模型进行微调，并通过三代渐进式训练流水线证明该方法能显著提升答案质量，且模型规模越大收益越明显。

    

    大语言模型的推理能力已经从基于提示的方法演变为内化的模型行为。本文研究大语言模型的内部思维模式能否被有意地改变，以及这种改变是否有益。我们不再依赖思维链，而是训练模型遵循“问题—选项—标准”这一结构化决策过程，将任务分解为明确的问题、相互竞争的备选选项以及评估标准。我们提出了一个三代发展流水线：从基础的QOC决策核心，到更丰富的答案合成，再到统计增强的QOC推理。随后，我们从内部一致性和输出质量两个方面对所得模型进行评估。在所报告的实验中，基于QOC的微调相较于未微调的基础模型提升了答案质量，并且当模型规模足够大、能够执行该过程时，收益尤为显著。（注：原摘要在此处截断）

    arXiv:2611.26282v1 Announce Type: cross  Abstract: Reasoning capabilities in large language models have evolved from prompting-based methods to internalized model behaviors. In this paper, we investigate whether the internal thinking model of an LLM can be deliberately changed and whether such a change is beneficial. Instead of relying on Chain-of-Thought, we train models to follow Question--Option--Criteria (QOC), a structured decision process that decomposes a task into explicit questions, competing options, and evaluation criteria. We present a three-generation development pipeline that ranges from a basic QOC decision core to richer answer synthesis and statistically enriched QOC reasoning. We then evaluate the resulting models with respect to both internal consistency and output quality. Across the reported experiments, QOC-based fine-tuning improves answer quality relative to non-fine-tuned base models and yields particularly strong gains once models are large enough to execute t
    
[^16]: dLoop：通过循环潜在精炼实现的循环扩散语言模型

    dLoop: Looped Diffusion Language Models via Recurrent Latent Refinement

    [https://arxiv.org/abs/2611.26275](https://arxiv.org/abs/2611.26275)

    dLoop通过将预训练扩散语言模型重构为Prelude-Loop-Coda架构，并对相同的部分掩码状态循环应用参数共享的中间块来迭代精炼潜在表示，从而在不推进去噪过程的情况下提升预测的可靠性，减少所需去噪步骤并加速生成。

    

    扩散语言模型（DLMs）支持并行生成，但由于每一步中只有一小部分token预测足够可靠从而能被提交，它们通常需要许多去噪步骤。现有的加速方法主要优化置信度学习或提交策略。然而，这些方法作用于单次去噪前向传播的输出，而在这个输出中，来自提示词、已提交token和未提交位置的信息可能尚未被充分整合。因此，在做出提交决策时，一些预测仍未得到充分精炼，而仅仅修改提交策略无法改善其底层表示。我们提出了dLoop，它将预训练的扩散语言模型重构为Prelude-Loop-Coda（前奏-循环-尾声）架构，并将参数共享的中间块循环地应用于相同的部分掩码状态，而不是推进去噪过程（原文摘要在此处截断）。

    arXiv:2611.26275v1 Announce Type: new  Abstract: Diffusion language models (DLMs) support parallel generation, yet they often require many denoising steps because only a fraction of token predictions are sufficiently reliable to be committed at each step. Existing acceleration methods primarily optimize confidence learning or commitment policies. However, these approaches operate on the outputs of a single denoising forward pass, in which information from the prompt, previously committed tokens, and uncommitted positions may not yet be fully integrated. Consequently, some predictions remain under-refined when the commitment decision is made, and modifying the commitment policy alone cannot improve their underlying representations. We introduce dLoop, which restructures a pretrained diffusion language model into a Prelude-Loop-Coda architecture and recurrently applies a parameter-shared intermediate block to the same partially masked state. Rather than advancing the denoising process, e
    
[^17]: STAGE：用于弱监督跨推文事件因果关系发现的时空增强图扩展

    STAGE: Spatio-Temporal Augmented Graph Expansion for Weakly Supervised Cross-Tweet Event Causality Discovery

    [https://arxiv.org/abs/2611.26255](https://arxiv.org/abs/2611.26255)

    STAGE框架从局部因果种子出发构建跨推文弱监督标签，结合时空上下文图与图注意力网络，实现灾害社交媒体中跨推文事件因果关系的可扩展自动发现，相比局部监督显著提升因果发现性能。

    

    灾害相关社交媒体中的因果关系往往跨越独立发布的帖子，这使得跨推文因果关系发现难以进行大规模的监督和建模。我们提出STAGE，一个将可扩展的跨推文弱监督构建与基于事件发生的时空关系学习相分离的框架。从局部因果种子出发，STAGE检索跨推文中语义相关的事件发生，并使用因果评分器构建弱标注的跨推文事件对。这些事件对用于监督多头图注意力网络，该网络构建于语义丰富的事件表示和时空上下文图之上。我们使用人工标注的黄金跨推文测试集和15个基线方法，在加州野火和飓风哈维数据集上评估STAGE。与仅使用局部监督相比，构建的跨推文监督使STAGE的AUPRC在野火数据上提升了0.340，在飓风数据上提升了0.107。STAGE在域内测试中取得了0.885的AUROC

    arXiv:2611.26255v1 Announce Type: cross  Abstract: Causal relations in disaster-related social media of- ten span independent posts, making cross-tweet causal discovery difficult to supervise and model at scale. We propose STAGE, a framework that separates scalable cross-tweet weak-supervision construction from spatio-temporal relational learning over event occurrences. Starting from local causal seeds, STAGE retrieves semantically related event occurrences across tweets and uses a causal scorer to construct weakly labeled cross-tweet pairs. These pairs supervise a multi-head Graph Attention Network over semantically enriched event representations and a spatio- temporal context graph. We evaluate STAGE on California Wildfire and Hurricane Harvey using human-gold cross-tweet test sets and 15 baselines. Compared with local-only supervision, the constructed cross-tweet supervision improves STAGE AUPRC by 0.340 on Wildfire and 0.107 on Hurricane. STAGE achieves in-domain AUROC of 0.885 on 
    
[^18]: TwinLab-Bench：在真实实验室的智能体孪生环境中评估AI科学家

    TwinLab-Bench: Evaluating AI Scientists in Agentic Twins of Real Labs

    [https://arxiv.org/abs/2611.26248](https://arxiv.org/abs/2611.26248)

    该论文提出TwinLab-Bench基准，利用来自真实物理科学、生命科学和工程实验室的52个任务，首次系统评估AI科学家自主设计实验的能力，其结果通过与真实实验室隐藏记录对比来评分。

    

    AI智能体日益被期望扮演AI科学家的角色，这需要三种开放性递增的能力：正确执行给定的计算工作流程、从给定的设计空间中选择下一个测量方案，以及自主设计实验。现有基准主要评估前两种能力，即通过终端或从预定义候选中进行选择的闭环方式来评估。而评估第三种能力需要可靠地执行智能体自主设计的实验，并使实验读数建立在真实实验室的基础之上。我们提出了TwinLab-Bench，该基准通过来自物理科学、生命科学和工程实验室的52个任务，评估AI科学家在自主设计实验方面的能力。该基准支持超过100种实验操作类型，智能体的报告通过与取自真实实验室记录的隐藏答案进行对比来评分。每个任务都有其专属的TwinLab智能体，该智能体充当实验环境……

    arXiv:2611.26248v1 Announce Type: new  Abstract: AI agents are increasingly expected to act as AI scientists, which requires three increasingly open abilities: carrying out a given computational workflow correctly, choosing the next measurements from a given design space, and designing their own experiments. Existing benchmarks mainly evaluate the first two abilities, in a terminal or in a closed loop that selects from predefined candidates. Evaluating the third ability requires reliably executing an experiment that an agent designs, with readings grounded in a real laboratory. We introduce TwinLab-Bench, which evaluates AI scientists on self-designed experiments in 52 tasks from laboratories in the physical sciences, the life sciences, and engineering. The benchmark supports more than 100 experimental action types, and the agent's report is scored by comparison with a hidden answer taken from real laboratory records. Each task has its own TwinLab agent, an agent that serves as the env
    
[^19]: 前沿模型仅需一段文本即可揭露其人类或AI作者身份

    Frontier models need only one text to unmask its human or AI author

    [https://arxiv.org/abs/2611.26216](https://arxiv.org/abs/2611.26216)

    前沿大模型仅凭一段文本和简单的零样本提示，就能判断作者是AI还是人类、识别AI文本来自哪家实验室、甚至准确指认人类作者的身份，其性能媲美专用检测器并能给出自然语言解释。

    

    仅需给定一段文本和一个简单的零样本提示（无需网络访问、工具、候选列表或写作样本），前沿模型就能在三个层层递进的层面上回答“这是谁写的？”：(1) 是AI还是人类写的；(2) 如果是AI，来自哪个实验室；(3) 如果是人类，往往能准确说出具体是谁。最新的模型（Claude Fable 5.2、GPT-6.1 Sol、Gemini 4 Carbon）在使用自然语言解释其预测的同时，性能可与Pangram等经过专门训练的检测器相媲美；Opus 5.5在98%的情况下能正确说出Anthropic、OpenAI和Google模型所生成文本的来源实验室。最引人注目的是，Fable 5.2能正确识别2026年10-11月发表的39%公开同行评审的作者姓名（在来自Nature的一个子集上达到78%），并能识别出活跃Reddit用户（过去一年发帖超过500条）69%评论背后的用户名。准确率取决于预训练数据中对作者写作内容的接触程度以及推理投入程度，而仅仅询问一个mo...（原文摘要在此处截断）

    arXiv:2611.26216v1 Announce Type: new  Abstract: Given a single text and a simple zero-shot prompt (no web access, tools, candidate lists, or writing samples), frontier models can answer "who wrote this?" at three increasingly specific levels: (1) whether an AI or human wrote it, (2) if an AI, which lab, and (3) if a human, often exactly who. The latest models (Claude Fable 5.2, GPT-6.1 Sol, Gemini 4 Carbon) match trained detectors like Pangram while also explaining their predictions in natural language, and Opus 5.5 names the correct lab for 98% of texts generated by models from Anthropic, OpenAI, and Google. Most strikingly, Fable 5.2 correctly names the author of 39% of public peer reviews published in Oct-Nov 2026 (78% on a subset of reviews from Nature), and identifies the username behind 69% of comments from active Redditors (with >500 posts over the last year). Accuracy depends on exposure to an author's writing in pretraining data and on reasoning effort, and simply asking a mo
    
[^20]: 通过自适应数据选择学习忠实地用图像思考

    Learning to Think with Images Faithfully via Adaptive Data Selection

    [https://arxiv.org/abs/2611.26199](https://arxiv.org/abs/2611.26199)

    该论文揭示了“用图像思考”模型普遍存在不忠实的工具调用现象（即答对却并不真正依赖放大后的区域），并提出自适应数据选择方法，使训练数据随模型能力动态调整、聚焦于放大证据真正有效的问题，从而实现忠实的基于图像的推理。

    

    用图像思考能够使视觉语言模型通过主动裁剪并放大细粒度视觉细节来克服感知瓶颈，有望同时带来更高的准确率以及可解释的、基于区域的推理。然而，我们的分析表明，这类模型准确率的提升并不能保证忠实的工具使用；它们经常在并未真正依赖放大区域的情况下调用工具，从而削弱了工具辅助推理的可靠性。我们将这种不忠实行为追溯到训练数据：当问题无需放大即可回答时，模型可以在忽略返回证据的情况下答对，且随着训练过程中模型在无工具条件下回答能力的提升，这一问题会进一步加剧。为了在整个训练过程中保持缩放操作的有效性，我们将训练数据适配到模型不断演进的能力上，在那些正确的放大证据仍能帮助模型回答的问题上进行训练。

    arXiv:2611.26199v1 Announce Type: cross  Abstract: Thinking with images (TwI) empowers vision-language models (VLMs) to overcome perceptual bottlenecks by actively cropping and zooming into fine-grained visual details, promising both higher accuracy and interpretable, region-grounded reasoning. However, our analysis shows that the improved accuracy of TwI models does not guarantee faithful tool use; they frequently invoke tools without actually relying on the zoomed region, undermining the reliability of tool-assisted reasoning. We trace this unfaithful behaviour to the training data: when questions are answerable without zooming, the model can answer correctly while ignoring the returned evidence, and this problem is exacerbated during training as the model becomes better at answering without tools. To keep zooming useful throughout training, we adapt the training data to the model's evolving capability and train on questions where correct zoomed evidence can still help the model answ
    
[^21]: 世界模型能预测水稻生长吗？迈向农业世界模型

    Can World Models Predict Rice Growth? Toward Agricultural World Models

    [https://arxiv.org/abs/2611.26145](https://arxiv.org/abs/2611.26145)

    本文提出农业世界建模任务及AgroWM框架，其PhenoClock推理模块先推算水稻当前的发育阶段并预测未来阶段，再据此生成植株在数天至数月后的外观，克服了现有图像生成模型无法准确预测作物长期生长的局限。

    

    世界模型可以想象物理场景在未来几秒内如何演变，但种植者需要知道作物在未来数周乃至数月内将如何发育：这株水稻一个月后会抽穗吗？收获时它又会是什么样子？我们提出“农业世界建模”这一任务，即根据一张植物照片、其品种信息以及相机视角，预测同一株植物在所需的数天至数月时间跨度后将呈现的外观。在初步分析中，即使是最先进的图像生成器也会出错：它们往往让植物生长不足，或赋予其错误生长阶段的颜色和形态。我们认为，这是因为它们没有推算出植物的发育状态，而照片只能间接揭示这一状态。因此，我们提出AgroWM，它在绘制生长之前先对发育进行推理。其推理模块PhenoClock计算出当前的生长阶段，预测未来的生长阶段，并编写一份包含株高等数值目标的Growth Script（生长脚本）。

    arXiv:2611.26145v1 Announce Type: cross  Abstract: World models can imagine how a physical scene evolves over the next few seconds, but growers need to know how a crop will develop over weeks to months: will this rice plant have headed a month from now, and how will it look at harvest? We introduce agricultural world modeling, the task of predicting how the same plant will look at a requested horizon of days to months, from one photo of it, its cultivar, and its camera view. In preliminary analysis, even state-of-the-art image generators get this wrong: they often grow the plant too little or give it the color and form of the wrong growth stage. We argue that they do not work out the plant's developmental state, which the photo reveals only indirectly. We therefore propose AgroWM, which reasons about development before it draws growth. Its reasoning module, PhenoClock, works out the current growth stage, predicts the future one, and writes a Growth Script with numeric targets for heigh
    
[^22]: 强化学习只能强化它所见到的成功

    RL Can Only Reinforce What It Sees Succeed

    [https://arxiv.org/abs/2611.26084](https://arxiv.org/abs/2611.26084)

    在先验与正确答案均已精确已知的谜题环境中，该研究发现结果奖励强化学习只能强化训练中实际观察到成功的动作——当正确动作罕见时，训练能否“起飞”可由训练前即可计算的一个统计量（动作获胜的尝试组期望数量，约35组对应五成概率）准确预测，且该规律跨多种实验设置普遍成立。

    

    当一个熟悉的规则失效时，智能体必须权衡预训练教给它的内容与它当前观察到的内容。我们研究结果奖励强化学习（RL）如何改变这种平衡，并在一个两侧信息均已确定的谜题世界中进行验证：小语言模型在由贝叶斯玩家玩出的游戏数据上从头训练，因此先验和正确答案都是精确已知的。基础模型像其语料库那样进行对冲，而RL会将这种对冲转化为决策，方向取决于奖励偏向哪一方。但RL只能强化它所见到的成功。当正确动作很罕见时，每次训练要么在所有地方都学会它，要么完全学不会；而哪些训练能够成功，可以在训练开始前用一个计数来预测：该动作获胜的尝试组的期望数量。大约35个这样的组对应五成胜率。该计数对新问题和重复问题均成立，对4到256次尝试的组均成立，并且在不同学习率、优化器、GRPO变体和模型规模下均保持成立。

    arXiv:2611.26084v1 Announce Type: cross  Abstract: When a familiar rule stops working, an agent has to weigh what pre-training taught it against what it now observes. We ask how outcome-reward reinforcement learning (RL) changes this balance, in a puzzle world where both sides are known: small language models trained from scratch on games played by a Bayesian player, so that the prior and the right answer are exact. Base models hedge as their corpus did, and RL turns that hedging into decisions, in whichever direction the reward favours. But RL can only reinforce what it sees succeed. When the right action is rare, each run either learns it everywhere or not at all, and which runs take off is predicted before training by one count: the expected number of groups of attempts in which the action wins. About 35 such groups give even odds. The count holds for new and repeated problems and for groups of 4 to 256 attempts, and across learning rates, optimizers, GRPO variants and model sizes f
    
[^23]: 相同的判断，不同的分数：LLM-as-a-Judge评估中的数值评分拓扑

    Same Judgement, Different Scores: Numerical Scoring Topology in LLM-as-a-Judge Evaluation

    [https://arxiv.org/abs/2611.26077](https://arxiv.org/abs/2611.26077)

    本文提出“数值评分拓扑”概念，并首次系统研究证明LLM-as-a-Judge评估中评分方案的结构设计（量程宽度、粒度、偏移、对称性与边界位置）会系统性影响评判分数、校准与鲁棒性，其中偏移和狭窄的评分范围会导致最大的分数偏差。

    

    大语言模型越来越多地被用作自动评估器，以降低人工标注所需的成本和时间。然而，评分方案设计在LLM-as-a-Judge评估可靠性中所起的作用仍未得到充分研究。为解决这一问题，我们提出了数值评分拓扑的概念，它从量程宽度、粒度、偏移、对称性和边界位置等方面刻画评分方案的结构设计。基于这一定义，我们首次系统地研究了数值评分拓扑如何在多种基准测试和评判模型下影响零样本LLM-as-a-Judge评估。结果表明，数值评分拓扑会系统性地影响评判分数、校准和鲁棒性，尤其是在判断密集型的评估任务中。在许多设置下，偏移的和狭窄的评分范围往往会引起最大的分数偏差。我们进一步发现，更强、更大的评判模型……（原文摘要在此处截断）

    arXiv:2611.26077v1 Announce Type: new  Abstract: LLMs are increasingly used as automated evaluators to reduce the cost and time required for human annotation. However, the role of scoring-scheme design in the reliability of LLM-as-a-Judge evaluation remains underexplored. To address this problem, we introduce numerical scoring topology, which characterizes the structural design of a scoring scheme in terms of its scale width, granularity, offset, symmetry, and boundary placement. Building on this definition, we conduct the first systematic study of how numerical scoring topology influences zero-shot LLM-as-a-Judge evaluation across diverse benchmarks and judge models. Our results show that numerical scoring topology systematically affects judge scores, calibration, and robustness, particularly in judgment-intensive evaluations. Shifted and narrow ranges tend to induce some of the largest score deviations across many settings. We further find that stronger and larger judge models genera
    
[^24]: 超越梯度：理解MoE决策机制

    Understanding MoE Decisions Beyond Gradients

    [https://arxiv.org/abs/2611.26075](https://arxiv.org/abs/2611.26075)

    该论文提出一种利用MoE模型自身的专家输出和选择规则来估计专家替换引发的级联切换效应的预测器，相比传统梯度方法将答案得分变化的预测误差在Qwen3.6上降低60%、在Gemma4上降低47%。

    

    理解语言模型的决策需要了解其内部计算如何影响模型在不同答案之间的偏好。基于梯度的方法通过预测改变这些内部计算所带来的效果来估计这种影响。在混合专家模型中，替换某一个专家可能会改变后续各层所选择的专家。这种专家切换会以导数无法捕捉的方式改变答案得分。我们提出了一种预测器，它利用模型自身的专家输出和选择规则来估计这些额外的影响，并将其加入到二阶预测之中。在来自三个基准测试的问题上，相较于使用梯度和二阶导数的多步基线方法，我们的方法在Qwen3.6上将答案得分变化的预测误差降低了60%，在Gemma4上降低了47%。

    arXiv:2611.26075v1 Announce Type: new  Abstract: Understanding a language model's decision requires knowing how its internal computations affect its preference between answers. Gradient-based methods estimate this influence by predicting the effect of changing those computations. In mixture-of-experts (MoE) models, replacing one expert can change which experts later layers select. Those switches can alter the answer scores in ways that derivatives do not capture. We introduce a predictor that uses the model's own expert outputs and selection rules to estimate these extra effects, then adds them to a second-order prediction. Across questions from three benchmarks, our method reduces the error in predicting answer-score changes by 60\% on Qwen3.6 and 47\% on Gemma4, relative to a multistep baseline using gradients and second derivatives.
    
[^25]: AltRA-Test：一种无需共享标注者的LLM评判者替代标注者检验

    AltRA-Test: An Alternative Annotator Test for LLM Judges Without Shared Annotators

    [https://arxiv.org/abs/2611.26073](https://arxiv.org/abs/2611.26073)

    提出了AltRA-Test统计检验方法，通过自助法重采样克服条目由不同标注者子集标注的限制，从而判断用LLM替代人类标注者在统计学上是否合理。

    

    大语言模型越来越多地被用作评判者和评分者来替代人类标注者，这引发了一个问题：这种替代在何时具有统计学上的合理性。为了支持此类决策，此前提出了替代标注者检验，适用于所有条目均由相同标注者标注的密集场景。然而，在许多大规模标注场景中，条目是由不同的标注者子集进行标注的。为此，我们提出了基于重采样标注的替代标注者检验，这是一种不要求所有条目均由相同标注者标注的统计检验方法。AltRA-Test采用自助法重采样条目，为每个条目抽取k个人类标注，并将其中一个替换为LLM标注，从而生成配对的全人类标注组与人类-LLM混合标注组。当混合标注组的标注者间一致性在统计学上不劣于全人类标注组时，即支持进行替换。

    arXiv:2611.26073v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly used as judges and graders to replace human annotators, raising the question of when such replacement is statistically justified. To support such decisions, the Alternative Annotator Test (Alt-Test) was introduced for dense settings where all items are annotated by the same annotators. However, in many large-scale annotation settings, items are annotated by different subsets of annotators. We therefore propose the Alternative Annotator Test with Resampled Annotations (AltRA-Test), a statistical test that does not require that all items are annotated by the same annotators. AltRA-Test uses a bootstrap that resamples items, draws k human annotations for each item, and substitutes one of them with an LLM annotation. This produces paired all-human and mixed human-LLM panels. A replacement is supported when the inter-annotator agreement (IAA) of the mixed panel is statistically non-inferior to t
    
[^26]: SheetAlign：面向电子表格智能体的具接地交互与阶段隔离学习

    SheetAlign: Grounded Interaction and Stage-Isolated Learning for Spreadsheet Agents

    [https://arxiv.org/abs/2611.26069](https://arxiv.org/abs/2611.26069)

    SheetAlign是一个电子表格操作框架，通过阶段隔离策略学习、任务引导的API接地以及基于执行一致性的程序选择，将代码生成锚定在工作簿观察与可用操作上，从而在SpreadsheetBench上取得优异表现。

    

    电子表格被广泛用于数据分析和决策支持。在保护敏感数据和控制部署成本的前提下实现电子表格分析自动化，推动了本地部署的开源权重大语言模型（LLM）的应用。同时，复杂的电子表格任务既需要任务推理，也需要对工作簿的理解，而预定义的操作接口还带来了额外的工具选择与调用需求。我们提出了SheetAlign，一个具有阶段隔离策略学习的电子表格操作框架，围绕“在哪里操作、更改什么、以及如何实现转换”来组织。任务引导的检查和API接地将代码生成锚定在工作簿观察和可用操作上，而阶段隔离学习则利用编辑进度奖励来优化程序生成与审查。在推理阶段，执行一致性通过工作簿状态之间的一致性来选择程序。在SpreadsheetBench上...

    arXiv:2611.26069v1 Announce Type: new  Abstract: Spreadsheets are widely used for data analysis and supporting decision-making. Automating spreadsheet analysis while protecting sensitive data and controlling deployment costs motivates locally deployed open-weight large language models (LLMs). Meanwhile, complex spreadsheet tasks require both task reasoning and workbook understanding, while predefined operation interfaces impose additional tool-selection and invocation demands. We introduce SheetAlign, a spreadsheet manipulation framework with stage-isolated policy learning, organized around where to operate, what to change, and how to implement transformations. Task-guided inspection and API grounding anchor code generation in workbook observations and available operations, while stage-isolated learning optimizes program generation and review using edit-progress rewards. At inference, execution consistency selects programs through agreement among workbook states. On SpreadsheetBench, o
    
[^27]: Complete-SVD：一种面向大语言模型低秩压缩的泛函数范数方法

    Complete-SVD: A Functional Norm Approach to Low-Rank LLM Compression

    [https://arxiv.org/abs/2611.26059](https://arxiv.org/abs/2611.26059)

    提出Complete-SVD方法，通过最小化原始与压缩模型logit之间的等权L2距离，将大语言模型低秩压缩目标化简为具有闭式解的加权SVD，并利用随机探测以与词表大小和序列长度无关的计算成本无偏估计难以求解的输出侧因子。

    

    神经网络的训练后低秩压缩不仅需要近似原始权重，还必须保持模型的输入输出行为。近期的输出感知方法在压缩层之外、即模型输出处度量压缩误差，通过观察到的下一个token或模型预测的概率对logit变化进行加权。我们提出Complete-SVD，一种面向大语言模型的方法，它最小化原始模型与压缩模型logit之间的L2距离，并对所有logit变化进行等权处理。通过在原始权重附近对网络进行线性化，并结合Kronecker因子分解近似，该目标函数被化简为具有闭式解的加权SVD。其输出侧因子难以直接计算；我们通过随机探测对其进行无偏估计，每个序列的代价仅为K次反向传播，且该成本与词表大小和序列长度无关。在相同设置与均匀……（摘要在此处截断）

    arXiv:2611.26059v1 Announce Type: cross  Abstract: Post-training low-rank compression of neural networks must not only approximate the original weights but also preserve the model's input-output behavior. Recent output-aware methods measure the compression error beyond the compressed layer, at the model output, weighting logit changes by the observed next token or by the model's predicted probabilities. We propose Complete-SVD, a method for large language models that minimizes the $L_2$ distance between the logits of the original and compressed models, weighting all logit changes equally. Linearizing the network around the original weights, together with a Kronecker-factored approximation, reduces this objective to a weighted SVD with a closed-form solution. Its output-side factor is intractable to compute; we estimate it without bias from random probes, at a cost of $K$ backward passes per sequence, independent of vocabulary size and sequence length. At matched settings with uniform r
    
[^28]: 评估AI助手多轮对话中的安全保障机制

    Evaluating safeguarding in multi-turn conversations with AI assistants

    [https://arxiv.org/abs/2611.26009](https://arxiv.org/abs/2611.26009)

    该论文与社区心理健康支持工作者合作设计了九个多轮对话场景，发现AI助手在痛苦线索逐渐显现的日常对话中可能无法保持与信号一次性明确呈现时同等的安全保障水平，从而揭示了现有安全评估方法的不足。

    

    会话式AI助手如今被广泛用于个人和情感支持。开发者报告称新模型在应对敏感对话方面有所改进，但用户和公民社会仍然对模型在真实世界中反复交互时的行为表示担忧，而这类场景往往被现有安全评估所忽略。通常，评估中痛苦信号会清晰地在单条消息中呈现，然而现实中人们往往是在看似平常的对话中逐渐流露出自己的困境。我们与三名社区心理健康支持工作者合作，测试模型在信号一次性全部给出或随时间逐渐给出时，能否提供同等水平的安全保障。我们设计了九个场景，用户在其中分享与抑郁、精神障碍或家庭虐待相关的线索。针对每个场景，支持工作者确立了黄金标准回应，范围从引导用户获取支持资源，到暂停日常任务以处理安全保障问题。我们向五……（摘要截断）

    arXiv:2611.26009v1 Announce Type: new  Abstract: Conversational AI assistants are now used for personal and emotional support. Developers report improvements in how new models respond to sensitive conversations, yet users and civil society continue to raise concerns about their behaviour in real-world, repeated interactions that safety evaluations rarely capture. Often, evaluations present signs of distress clearly and in a single message, even though people may reveal their struggles gradually in otherwise ordinary conversations. We worked with three community mental health support workers to test if models provide the same level of safeguarding when cues are shared all at once or gradually over time. We designed nine scenarios in which a user shares cues related to depression, psychosis, or domestic abuse. For each scenario, the support workers established a gold standard response, ranging from signposting support to pausing the everyday task to address safeguarding. We presented fiv
    
[^29]: Xing-Audio-NVVASR：扩展与精炼监督以实现语音与非言语发声的联合识别

    Xing-Audio-NVVASR: Scaling and Refining Supervision for Joint Recognition of Speech and Non-Verbal Vocalizations

    [https://arxiv.org/abs/2611.26000](https://arxiv.org/abs/2611.26000)

    该论文提出Xing-Audio-NVVASR，通过渐进式数据挖掘与迭代监督精炼构建了约80万条话语的语料库，训练出17亿参数的端到端模型，实现带行内标签的语音与非言语发声联合识别。

    

    语音与非言语发声（NVVs）的联合识别受到事件丰富训练数据稀缺以及事件标签不准确或缺失的阻碍。我们提出了Xing-Audio-NVVASR，这是一个通过渐进式数据挖掘和迭代监督精炼训练的端到端识别器。在统一的声学标准指导下，两个异构模型对标注进行交叉审计，标记出不一致之处以供教师模型重新标注。为了发现模型之间一致性可能掩盖的共同遗漏，互补的呼吸检测器对整个语料库进行筛选以供人工审核。经人工确认的标签在各轮之间不断积累，并在后续精炼过程中得以保留。经过五轮迭代，我们对来自公共语料库的17万条话语进行了标注精炼，并纳入了63万条挖掘出的话语。所得的约80万条话语语料库被用于训练一个单一的17亿参数模型，该模型以行内标签转录语音，涵盖六类NVV。

    arXiv:2611.26000v1 Announce Type: new  Abstract: Joint recognition of speech and non-verbal vocalizations (NVVs) is hindered by scarce event-rich training data and inaccurate or missing event labels. We present Xing-Audio-NVVASR, an end-to-end recognizer trained through progressive data mining and iterative supervision refinement. Guided by unified acoustic criteria, two heterogeneous models cross-audit annotations, flagging disagreements for teacher reannotation. To uncover shared omissions that model agreement may conceal, complementary breath detectors screen the entire corpus for human review. Human-confirmed labels accumulate across rounds and are preserved during subsequent refinement. Over five rounds, we refine annotations for 170,000 utterances from public corpora and incorporate 630,000 mined utterances. The resulting corpus of approximately 800,000 utterances is used to train a single 1.7B-parameter model that transcribes speech with inline tags for six NVV classes. Across f
    
[^30]: 策略改进引导的持续强化学习重放方法

    Policy Improvement-Guided Replay for Continual Reinforcement Learning

    [https://arxiv.org/abs/2611.25989](https://arxiv.org/abs/2611.25989)

    该论文提出对比似然重放方法（CoLiR），基于策略改进原则和性能差分引理，在视觉语言模型的持续强化学习中选择性重放历史提示，从而在获得新能力的同时保留已取得的性能提升。

    

    持续强化学习（CRL）要求模型在获得新能力的同时保持之前任务上的性能。传统持续学习中常见的面向稳定性的视角强调知识保留和干扰缓解。作为对这一视角的补充，我们通过策略改进——强化学习的一个基本原则——重新审视视觉语言模型的持续强化学习，以保留已获得的性能提升。利用性能差分引理，我们通过当前策略在历史任务分布上的期望优势来刻画历史性能变化，从而启发使用当前策略对历史提示进行重放。为了分配有限的重放记忆，我们提出了对比似然重放（CoLiR），该方法将最终任务策略对正确回答的似然增益与初始策略对错误回答的似然降低进行对比。我们的分析将正确回答与……

    arXiv:2611.25989v1 Announce Type: cross  Abstract: Continual reinforcement learning (CRL) requires models to acquire new capabilities while retaining performance on previous tasks. A stability-oriented perspective common in conventional continual learning emphasizes knowledge preservation and interference mitigation. Complementing this perspective, we revisit CRL for vision-language models through policy improvement, a fundamental principle of RL, to preserve acquired performance gains. Using the performance difference lemma, we characterize historical performance changes through the current policy's expected advantages on historical task distributions, motivating replay of historical prompts with the current policy. To allocate limited replay memory, we propose Contrastive Likelihood Replay (CoLiR), which contrasts likelihood gains on correct responses from the final task policy with likelihood reductions on incorrect responses from the initial policy. Our analysis links correct-respo
    
[^31]: 开源权重大型语言模型在不同给定食物重量下的能量估计：一项剂量-反应评估

    Energy estimates of open-weight large language models across stated food weights: a dose-response evaluation

    [https://arxiv.org/abs/2611.25976](https://arxiv.org/abs/2611.25976)

    七个开源大语言模型在估计食物能量时无法随给定重量成比例缩放（弹性仅0.548-0.979），导致小份量被高估、大份量被低估，而采用“模型自报能量密度×重量”的两步法可基本消除该误差。

    

    我们检验了开源权重大型语言模型（LLM）的能量估计是否随精确给定的食物重量成比例变化，以及通过“能量密度×重量”的计算方式能否消除误差。在一项预先设定的协议下，七个开源权重LLM对从美国农业部膳食研究食物与营养素数据库中随机抽取的168种食物，在九个给定重量（为默认量的0.25倍至4倍）下估计其能量。主要结局指标是估计值相对于重量的弹性（1表示成比例缩放）。两步估计法是在代码中将每个模型自己给出的能量密度乘以重量。结果显示，所有七个模型都压缩了其估计值，弹性范围为0.548至0.979（所有95%置信区间均低于1）；其中四个模型压缩严重（0.548至0.709）。这四个模型在默认量四分之一时高估能量47%至130%，在默认量四倍时低估能量20%至34%。提供能量密度后可恢复接近成比例的缩放……

    arXiv:2611.25976v1 Announce Type: cross  Abstract: We tested whether energy estimates of open-weight large language models (LLMs) scale proportionally with an exactly stated food weight, and whether computing energy density times weight removes the error. Under a prespecified protocol, seven open-weight LLMs estimated the energy of 168 randomly drawn foods from the USDA Food and Nutrient Database for Dietary Studies at nine stated weights, from 0.25 to 4 times the default amount. The primary outcome was the elasticity of the estimate with respect to weight (1: proportional scaling). A two-step estimate multiplied each model's own stated energy density by the weight in code. All seven models compressed their estimates, with elasticities from 0.548 to 0.979 (every 95% CI below 1); four compressed severely (0.548 to 0.709). These four overestimated a quarter of the default amount by 47% to 130% and underestimated four times it by 20% to 34%. Supplying the density restored near-proportiona
    
[^32]: 预测的深度：语言模型解析词语的层级位置可预测人类阅读时间

    The Depth of a Prediction: Where Language Models Resolve a Word Predicts Human Reading Time

    [https://arxiv.org/abs/2611.25970](https://arxiv.org/abs/2611.25970)

    论文提出“解析深度”这一新指标，用以衡量语言模型在哪一层完成对一个词的预测，并证明模型解析越晚的词及其后续词会被人类读得更慢，且解析深度能在最终层意外度和熵之外显著提升对人类阅读时间的预测。

    

    来自语言模型最后一层的意外度预测了人们阅读一个词的时长，然而模型是在许多层中逐步计算这一预测的。我们使用调谐透镜在每一层读取每个词的预测，并用“解析深度”来概括这条逐层意外度轨迹——即其变化所围绕的中心深度。在八个英语眼动追踪和自定步速阅读语料库（81,957 个词）以及十个 GPT-2、OPT 和 Pythia 模型上，模型解析较晚的词，以及这些词之后的词，与同等意外程度的词相比被阅读得更慢。在留出集预测上，解析深度在最终层意外度、熵以及最佳单层之外额外提升了每词 0.26 毫奈特的预测能力（合并 p = 0.001），约为熵所带来的增益的三分之一，且每个模型家族均有所获益，其中 GPT-2 和 Pythia 各通过一个语料库体现。净增益在不同语料库之间存在差异（随机效应均值 0.07，p = 0.37）；相对于层序打乱的解析深度——其引入了同样多的系数……

    arXiv:2611.25970v1 Announce Type: new  Abstract: Surprisal from a language model's final layer predicts how long people read a word, yet the model computes its prediction across many layers. We read each word's prediction at every layer with the tuned lens and summarize this layerwise surprisal trajectory by its resolution depth, the depth around which its change is centered. On eight English eye-tracking and self-paced reading corpora (81,957 words) and ten GPT-2, OPT and Pythia models, words that a model resolves later, and the words after them, are read more slowly than equally surprising words. Resolution depth improves held-out prediction beyond final-layer surprisal, entropy and the best single layer by 0.26 millinats per word (pooled p = 0.001), about a third of entropy's gain, and each model family gains, GPT-2 and Pythia through one corpus each. The net gain varies between corpora (random-effects mean 0.07, p = 0.37); against layer-shuffled depth, which adds as many coefficien
    
[^33]: 《当声音牵动注意力头：恢复音频大模型无法言说的事件时间信息》

    When Sounds Turn Heads: Recovering Event Timing That Audio LLMs Fail to Say

    [https://arxiv.org/abs/2611.25960](https://arxiv.org/abs/2611.25960)

    音频大模型的少数注意力头内部已追踪到被查询声音的发生时间却无法在回答中表达，本文提出无需训练的头引导解码方法，利用这些注意力头的先验对候选时间戳重新打分，从而显著提升三种音频大模型的时间戳精度。

    

    音频大语言模型能够知道被查询的声音何时发生，却无法在回答中表达出来。我们发现少数注意力头能够追踪被查询的声音：当查询改变时它们会随之移动，当声音位置变化时它们会跟随，而移除或阻断这些注意力头往往会损害模型的时间戳预测。然而模型的回答并未表达出这些注意力头所追踪的信息。直接从这些注意力头读取的时间区间优于模型原本的回答，并能恢复其时间戳所遗漏的事件。为了将这种时间信息引入模型的回答中，我们提出了无需训练的头引导解码方法：它保留模型对声音是否存在的判断，并利用这些注意力头提供的先验对候选时间戳进行重新打分。在开发数据上选出的注意力头无需任何改动即可直接迁移到新的录音和自由文本查询上。在这些场景中，头引导解码提升了三种音频大模型的时间戳质量，在所有基准上的得分均高于直接读取注意力头的区间，并且在大多数设置下优于随机注意力头引导。

    arXiv:2611.25960v1 Announce Type: cross  Abstract: Audio LLMs can know when a queried sound occurs but fail to say it. We find a few attention heads that track the queried sound: they shift when the query changes, follow the sound when it moves, and removing or blocking them often hurts the model's timestamps. Yet the answer does not express what they track. Intervals read directly from these heads outperform the model's original answer and recover events that its timestamps miss. To bring this timing into the answer, we introduce head-guided decoding, which needs no training. It keeps the model's decision on whether the sound is present and rescores its candidate timestamps with a prior from these heads. Heads selected on development data transfer unchanged to new recordings and free-text queries, where head-guided decoding improves the timestamps of three audio LLMs, scores above their head readout on every benchmark, and, in most settings, outperforms random-head guidance. On record
    
[^34]: 拉丁美洲的网络搜索：语言模型使用的新闻来源比搜索引擎更少本地化且多样性更低

    Web search in Latin America: language models use less local and diverse news than search engines

    [https://arxiv.org/abs/2611.25955](https://arxiv.org/abs/2611.25955)

    本研究通过对拉丁美洲九国时事的大规模查询实验发现，基于语言模型的搜索返回的新闻来源多样性低于传统搜索引擎，且本地新闻占比因用户类型而表现不一致，揭示了AI搜索可能加剧地理偏见的风险。

    

    AI平台正在成为人们获取新闻和时事的主要入口。这一转变可能会加剧现有的地理偏见，即偏向来自较富裕国家的叙述而忽视全球南方国家的声音。这对拉丁美洲尤为相关，因为该地区在全球南方的AI使用量中占比最大。我们研究了当用户从传统搜索引擎转向基于大语言模型的搜索时，关于拉丁美洲的信息会发生怎样的变化。我们构建了一个涵盖九个拉丁美洲国家、包含1,901个时事事件的数据集，设计了贴近现实的查询，并将其提交给三个搜索引擎和四个语言模型。我们改变了查询语言和用户所在位置，以代表本地用户、海外侨民、游客和外国用户四种场景。在49,207次请求和8,834个独立域名中，语言模型返回的新闻来源多样性低于搜索引擎。在不同用户类型下，语言模型返回的国内新闻占比表现不一致……

    arXiv:2611.25955v1 Announce Type: cross  Abstract: AI platforms are becoming a major gateway to news and current events. This shift risks amplifying existing geographical biases that favour narratives from wealthier countries over those from the Global South. This is particularly relevant in Latin America, which accounts for the largest share of AI usage in the Global South. We examine how information about Latin America changes when users shift from conventional search engines to LLM-based search. We construct a dataset of 1,901 current events across nine Latin American countries, craft realistic queries, and submit them to three search engines and four language models. We vary query language and user location to represent local, diaspora, tourist, and foreign user scenarios. Across 49,207 requests and 8,834 unique domains, language models return a less diverse set of news sources than search engines. Across user types, the share of domestic news returned by language models is inconsi
    
[^35]: 多解一答：搜索顺序塑造语言模型的默认解

    Many Solutions, One Answer: Search Order Shapes the Default Solutions of Language Models

    [https://arxiv.org/abs/2611.25916](https://arxiv.org/abs/2611.25916)

    论文发现语言模型在多解约束谜题中高度集中于同一个默认解，其行为如同按特定顺序遍历的搜索求解器——优先访问初始候选最少的实体这一搜索顺序最能预测模型输出，且改变给定的顺序会相应改变模型的默认解。

    

    当一个约束谜题存在多个正确答案时，语言模型往往给出同一个答案。我们在符号谜题以及解集完全已知的多解ZebraLogic变体上，研究了来自三个家族的四个开放权重模型。重复采样高度集中于某一个默认解，其集中程度远超均匀采样，而且不同模型常常共享这个默认解。然而，当新增线索能唯一确定某个解时，即使该解很少被采样，模型通常也能将其给出。在留出的符号谜题上，我们比较了两种实体访问策略：按名称顺序访问实体，与优先访问初始候选最少的实体。两种搜索都按给定的顺序尝试取值。在二者预先指定的预测不同的地方，后者在所有四个模型中匹配了更大比例的回答，领先幅度达26至40个百分点。在两个测试平台上，反转给定的取值顺序或指定搜索顺序，都会使模型的回答偏向新顺序所偏好的解。随机化当前的……

    arXiv:2611.25916v1 Announce Type: cross  Abstract: When a constraint puzzle has several correct answers, language models tend to give the same one.We study four open-weight models from three families on symbolic puzzles and multi-solution ZebraLogic variants with exactly known solution sets.Repeated samples concentrate on one default solution far beyond uniform sampling, and different models often share it.Yet models usually produce even rarely sampled solutions when added clues uniquely identify them.On held-out symbolic puzzles, we compare visiting entities in name order with visiting those with the fewest initial candidates first. Both searches try values in the stated order. Where their pre-specified predictions differ, the latter matches a larger share of responses in all four models, by 26 to 40 percentage points.Reversing the stated value order or specifying a search order shifts responses towards the solution that the new order favours, on both testbeds. Randomising the present
    
[^36]: 依存条件化词嵌入的语言结构

    The Linguistic Structure of Dependency-Conditioned Embeddings

    [https://arxiv.org/abs/2611.25909](https://arxiv.org/abs/2611.25909)

    该论文通过分析全局与局部几何结构，系统考察了依存条件化词嵌入的语言组织方式，发现全局几何主要由上下文词性塑造，而局部词汇相似性结构在依存关系间得到部分保留。

    

    基于依存的分布模型能够捕捉受句法信息引导的词汇相似性，但依存条件化词汇表示的几何组织方式仍未得到充分探索。我们使用在WikiText-103上训练的四种表示来研究词元-依存关系嵌入的全局与局部几何结构：两种基于计数的PPMI配置和两种带负采样的Skip-Gram（SGNS）配置。在全局层面，我们分析依存关系的质心，通过词元内中心化来控制词汇身份的影响，并考察其与词性（POS）分布的对齐情况。在局部层面，我们评估了跨关系的几何一致性、跨方法的一致性、词汇相似性以及多义词的不同用法。结果表明，在所有表示中，全局几何结构都受到上下文词性的强烈塑造。在局部层面，更广泛的词汇相似性结构在依存关系间得到了部分保留。

    arXiv:2611.25909v1 Announce Type: new  Abstract: Dependency-based distributional models capture syntactically informed lexical similarities, but the geometric organisation of dependency-conditioned lexical representations remains underexplored. We investigate the global and local geometry of lemma-dependency relation embeddings using four representations trained on WikiText-103: two count-based PPMI configurations and two Skip-Gram with Negative Sampling (SGNS) configurations. At the global level, we analyse dependency-relation centroids, control for lexical identity through within-lemma centring, and examine their alignment with part-of-speech (POS) distributions. At the local level, we evaluate cross-relation geometric consistency, cross-method agreement, lexical similarity, and polysemous usages. Results show that global geometry is structured strongly by context POS across all representations. Locally, broader lexical similarity structures are partially preserved across dependency 
    
[^37]: 医学大语言模型中的可验证逻辑：智能体临床推理优化与神经符号接地的批判性综述

    Verifiable Logic in Medical LLMs: A Critical Review of Agentic Clinical Reasoning Optimization and Neuro-Symbolic Grounding

    [https://arxiv.org/abs/2611.25898](https://arxiv.org/abs/2611.25898)

    这篇综述指出临床大语言模型的最终答案正确并不代表推理有效，并系统评述了幻觉成因、智能体幻觉级联、过程奖励建模与神经符号接地四种范式，其中带符号否决层的系统能在其编码知识范围内提供可验证的安全保证。

    

    临床大语言模型通常以最终答案是否正确作为评判标准。然而，模型可能通过无效的推理得出流畅、看似合理的输出，而静态基准测试无法检测到这种情况；自信的幻觉、因果无效的推理以及虚假的人口统计学捷径都可能通过传统评估。本批判性综述依次涵盖四种范式：临床幻觉如何产生以及为什么静态指标会漏检它；智能体幻觉级联——即多智能体和工具增强系统会传播并放大早期推理错误；过程奖励建模作为步骤级监督；以及神经符号接地——形式逻辑引擎、本体推理机和知识图谱对神经输出施加可验证的约束。在其编码的知识范围内，具有符号否决层的系统能够提供相对于其编码知识成立的保证，并在安全性指标上报告了显著的提升。

    arXiv:2611.25898v1 Announce Type: cross  Abstract: Clinical large language models are usually judged by whether the final answer is right. Yet a model can reach a fluent, plausible output through invalid reasoning that static benchmarks do not detect, and confident hallucination, causally invalid inference, and spurious demographic shortcuts can pass conventional evaluation. This critical review covers four paradigms in turn: how clinical hallucination arises and why static metrics miss it; the agentic hallucination cascade, in which multi-agent and tool-augmented systems propagate and amplify early reasoning errors; process reward modeling as step-level supervision; and neuro-symbolic grounding, in which formal logic engines, ontology reasoners, and knowledge graphs place verifiable constraints on neural output. Within their encoded domains, systems with symbolic veto layers offer guarantees that hold relative to their encoded knowledge, and they report large gains on the safety measu
    
[^38]: 一种调度无法适用于所有情况：扩散语言模型的自适应推理调度

    One Schedule Does Not Fit All: Adaptive Inference Schedules for Diffusion Language Models

    [https://arxiv.org/abs/2611.25895](https://arxiv.org/abs/2611.25895)

    本文提出 TCDA，一种具有显式误差保证的自适应部署规则，可根据任务、模型和采样器为扩散语言模型选择满足质量要求的最低成本推理调度，将去噪器评估次数最多减少 3.4 倍。

    

    扩散语言模型通常在所有工作负载中使用固定的去噪调度。我们证明，在保持质量的前提下，最廉价的调度取决于任务、模型和采样器。我们提出了任务校准去噪分配（Task-Calibrated Denoising Allocation，TCDA），这是一种具有显式误差保证的部署规则。给定候选采样器、配对校准数据、效用容忍度以及防止退化的防护措施，TCDA 使用同步单侧自助法界限来认证哪些候选方案同时满足效用和防护标准。它部署经过认证的最廉价候选方案，或回退到完整调度。部署不可行调式的概率渐近地至多为 alpha。候选方案包括早期前缀解码、状态复用续写、缩减时间网格和置信度阈值解码。在七个数据集和两个模型系列上，TCDA 认证了去噪器评估次数的显著减少：在 XSum 上减少 3.4 倍，在 WMT 上减少 2.0 倍。

    arXiv:2611.25895v1 Announce Type: new  Abstract: Diffusion language models typically use a fixed denoising schedule across workloads. We show that the cheapest schedule that preserves quality depends on the task, model, and sampler. We propose Task-Calibrated Denoising Allocation (TCDA), a deployment rule with an explicit error guarantee. Given candidate samplers, paired calibration data, a utility tolerance, and guardrails against degeneration, TCDA uses simultaneous one-sided bootstrap bounds to certify which candidates meet both the utility and guardrail criteria. It deploys the cheapest certified candidate or falls back to the full schedule. The probability of deploying an infeasible schedule is asymptotically at most alpha. Candidates include early prefix decoding, state-reusing continuation, reduced time grids, and confidence-threshold decoding. Across seven datasets and two model families, TCDA certifies reductions in denoiser evaluations by factors of 3.4 on XSum and 2.0 on WMT
    
[^39]: HALTT：一个随时间演变的仇恨词汇词典

    HALTT: A HAte Lexicon Through Time

    [https://arxiv.org/abs/2611.25862](https://arxiv.org/abs/2611.25862)

    该论文提出了一个多阶段框架，构建并发布了首个收录19个多义仇恨词汇的历时性词典HALTT，以应对日常词汇被武器化为隐蔽贬义用语而导致的仇恨言论检测性能随时间下降的问题。

    

    自动化仇恨言论检测的效果会随时间推移而下降，因为日常词汇会被“武器化”为隐蔽的贬义用语，同时保留其原本的中性含义（多义性保留）。现有的词汇语义变化检测（LSCD）框架由于缺乏标准化的、区分词义的历时性资源，难以应对这一现象。我们提出了一个多阶段框架，用于构建和验证一个收录了发生语义偏移的仇恨言论词汇的历时性词典。我们的流程筛选了超过 20,000 个候选词，结合了历史（1770-1964）与现代（2001-2024）语料库频率校准、掩码定义级评分以缓解大语言模型的时间偏差，以及双角色推理提示范式。我们发布了 HALTT——一个包含 19 个具有字面义和贬义的多义词的、有语料依据的词典，分析了语义偏移的类型学，并为下游的历时性基准测试建立了评估协议。

    arXiv:2611.25862v1 Announce Type: new  Abstract: Automated hate speech detection degrades over time as everyday vocabulary is weaponized into covert slurs while retaining original benign meanings (polysemy retention). Existing lexical semantic change detection (LSCD) frameworks struggle with this phenomenon due to a lack of standardized, sense-delineated diachronic resources. We propose a multi-stage framework to construct and validate a diachronic lexicon of semantically shifted hate speech terms. Filtering > 20,000 candidates, our pipeline combines historical (1770-1964) and modern (2001-2024) corpus frequency grounding, masked definition-level scoring to mitigate LLM temporal bias, and a dual-persona reasoning prompt paradigm. We release HALTT, a grounded lexicon of 19 polysemous terms with literal and derogatory senses, analyze shift typologies, and establish evaluation protocols for downstream diachronic benchmarking.
    
[^40]: TableRoute：面向可扩展表格问答的能力感知路由

    TableRoute: Capability-Aware Routing for Scalable Table Question Answering

    [https://arxiv.org/abs/2611.25853](https://arxiv.org/abs/2611.25853)

    提出TableRoute框架，先生成经过验证的问题—表格证据并识别所需操作，再根据表格规模与模型能力在全表阅读、表格缩减和执行三种推理策略之间智能路由，从而实现可扩展的表格问答。

    

    基于大语言模型的表格问答仍然难以扩展：表格规模从紧凑型到百万token级别不等，而问题需要对列、数值和计算进行精确定位。现有方法通常专注于固定的推理策略，这在不同的表格规模和模型能力下显得脆弱。全表阅读对紧凑输入是高效的，但会受到无关上下文的干扰；表格缩减有助于处理长输入，但依赖于可靠的证据定位；而当表格超出模型上下文窗口时则需要采用执行策略，且该策略对模式和数值定位错误仍然敏感。我们提出了TableRoute，一个具备证据条件推理的能力感知路由框架，用于可扩展的表格问答。TableRoute首先生成经过验证的问题—表格证据并识别所需的操作，然后根据序列化表格（及模型能力）在上述三种策略之间为每个实例进行路由。

    arXiv:2611.25853v1 Announce Type: new  Abstract: Table question answering with large language models remains difficult to scale: tables range from compact to million-token sizes, while questions require precise grounding of columns, values, and computations. Existing methods often specialize in a fixed inference strategy, which is brittle across different table scales and model capabilities. Full-table reading is efficient for compact inputs but suffers from irrelevant context. Table reduction helps long inputs but depends on reliable evidence localization, while execution is required when tables exceed the model context window and remains sensitive to schema and value grounding errors. We propose TableRoute, a capability-aware routing framework with evidence-conditioned inference for scalable TableQA. TableRoute first generates validated question--table evidence and identifies the required operations, then routes each instance among these three strategies according to serialized table
    
[^41]: 锚定与核验：用于可审计LLM决策的确定性跨提示、跨模型调和机制

    Anchor and Check: Deterministic Cross-Prompt, Cross-Model Reconciliation for Auditable LLM Decisions

    [https://arxiv.org/abs/2611.25818](https://arxiv.org/abs/2611.25818)

    该论文提出“锚定与核验”确定性调和契约——让目标模型的贪心答案接受跨提示或跨模型家族的独立贪心核验、仅一致时才执行，以实现可审计的LLM自主决策，并发现汇集跨家族样本可使语义熵的AUROC提升1.7至4.0个百分点，而同模型温度核验是最弱设计。

    

    让大型语言模型（LLM）做出离散决策的流水线需要知道哪些决策可以在无人值守的情况下被执行，并且需要一条能解释每个决策的记录。我们研究了一个小型且完全明确指定的调和契约：提交目标模型的贪心答案（即锚点），抽取一至两次通过不同路径得到该答案的贪心核验（更换提示词，或使用来自另一模型家族的模型），然后由一个纯函数返回五种结果之一，其中只有“一致”结果会被采纳执行。在覆盖6000个问题的48,000次模型-条目评估中，每种方法均可使用言语化置信度作为平局决胜依据，核心发现关乎调用的来源：汇集另一个模型家族的样本可使语义熵的AUROC在每个预算水平上提升1.7至4.0个百分点，而同模型的温度核验是最弱的设计。作为完整流水线，三调用契约与基于3或5个样本的汇集语义熵的点估计……

    arXiv:2611.25818v1 Announce Type: new  Abstract: Pipelines that let a large language model (LLM) make discrete decisions need to know which decisions they may act on unattended, and a record that explains each one. We study a small, fully specified reconciliation contract: commit the target model's greedy answer (the anchor), draw one or two greedy checks that reach the answer by a different route (an alternate prompt, or a model from another family), and let a pure function return one of five outcomes, of which only Agree is acted on. On 48,000 model-item evaluations over 6,000 questions, with every method given access to verbalized confidence as a tie-breaker, the main finding concerns where calls come from: pooling another family's samples raises semantic entropy's AUROC by 1.7 to 4.0 points at every budget, and a same-model temperature check is the weakest design. As complete pipelines, the point estimates of the three-call contract and pooled semantic entropy over 3 or 5 samples d
    
[^42]: 反幻觉训练究竟买到了什么？少说话、廉价剪枝，以及其余部分

    What Does Anti-Hallucination Training Buy? Saying Less, Cheap Pruning, and the Rest

    [https://arxiv.org/abs/2611.25814](https://arxiv.org/abs/2611.25814)

    该论文证明，视觉-语言模型抗幻觉偏好训练所降低的幻觉，绝大部分可由“少说话”（截断字幕至相同召回率）以及一个仅用11个特征、在500张图上拟合的廉价后验探针剪枝所重现，说明训练带来的真实改进远小于表面数字。

    

    偏好训练是减少视觉-语言模型中物体幻觉的标准方法，每种新方法都报告更低的CHAIR分数。我们追问这种降低究竟由什么构成。我们将每个训练后的系统与基础模型生成的字幕在相同物体召回率下截断后进行比较，从而剔除来自“少说话”的那部分收益；剩余部分则与一个刻意弱化的、冻结的、仅含十一个特征的后验探针（在500张标注图像上拟合）所能重现的效果进行比较。所有评判标准均在分析代码中预先指定，并在它们所要评判的输出产生之前就被冻结。对于LLaVA-1.5-7B上的八种公开方法，该探针在COCO上对7/8种方法、在AMBER上对8/8种方法、在Object HalBench上对6/8种方法，均能重现训练在“少说话”之外所消除的至少三分之二，其中RLAIF-V是一致的例外；随后发布的公开方法SENTINEL（同时适用于LLaVA-1.5和Qwen2.5-VL）在两者上都达到了同样的标准。在我们的

    arXiv:2611.25814v1 Announce Type: cross  Abstract: Preference training is the standard recipe for reducing object hallucination in vision-language models, and each new method reports a lower CHAIR score. We ask what that reduction consists of. Every trained system is compared with its base model's caption truncated to the same object recall, which removes the part of the gain that comes from saying less; the remainder is compared with what a deliberately weak, frozen, eleven-feature post-hoc probe fitted on 500 annotated images can reproduce. All criteria were specified in the analysis code and frozen before the outputs they judge existed. For eight public methods on LLaVA-1.5-7B, the probe reproduces at least two thirds of what training removes beyond saying less for 7 of 8 methods on COCO, 8 of 8 on AMBER and 6 of 8 on Object HalBench, with RLAIF-V the consistent exception; a later public method released for both LLaVA-1.5 and Qwen2.5-VL (SENTINEL) meets the same bar on both. On our 
    
[^43]: 基于多词预测的高效且稳健的草稿生成

    Efficient and Robust Drafting with Multi-Token Prediction

    [https://arxiv.org/abs/2611.25813](https://arxiv.org/abs/2611.25813)

    提出了G-MTP——一种围绕单一共享草稿头构建的高度优化多词预测架构，通过目标激活注入、零拷贝KV缓存交叉注意力和聚类词表投影三项核心优化，使现有主干大语言模型能够高效生成高质量草稿，从而加速推测解码推理。

    

    推测解码已成为加速大语言模型（LLM）推理的关键技术，其通过将生成过程解耦为草稿生成阶段和并行验证阶段来实现加速。因此，一个关键挑战是设计一个既轻量又足够准确的草稿生成器，以逼近目标模型的预测。我们提出了一种高度优化的多词预测架构，它扩展现有的主干大语言模型，以高效生成高质量草稿。我们的方法围绕单一共享草稿头构建，具有三项核心架构优化：目标激活注入以锚定语义规划、零拷贝KV缓存交叉注意力以绕过预填充计算并减少内存开销，以及高效的聚类词表投影以降低全词表解码的成本。在Gemma 4模型家族上测试这些改进，我们的G-MTP展现出了开箱即用的加速效果……

    arXiv:2611.25813v1 Announce Type: new  Abstract: Speculative decoding has emerged as a critical technique for accelerating Large Language Model (LLM) inference by decoupling generation into a drafting phase and a parallel verification phase. A crucial challenge is therefore designing a drafter that is both lightweight and accurate enough to approximate the predictions of the target model. We present a highly optimized Multi-Token Prediction (MTP) architecture that extends existing backbone LLMs to efficiently produce high-quality drafts. Built around a single shared drafting head, our approach features three core architectural optimizations: target activation injection to anchor semantic planning, zero-copy KV-cache cross-attention to bypass prefill compute and reduce memory overhead, and an efficient clustered vocabulary projection to reduce the cost of full-vocabulary decoding. Testing these improvements with the Gemma 4 model family, our G-MTP shows out-of-the-box speedups for both 
    
[^44]: 语言模型对重音信息的表征多于它们所能表达的

    Language models represent more about emphasis than they can tell you

    [https://arxiv.org/abs/2611.25808](https://arxiv.org/abs/2611.25808)

    该研究发现语言模型内部编码了文本所能决定的重音突显信息，但通过提示无法将其表达出来，而完整的重音信息必须依赖语音信号才能获得。

    

    韵律突显在很大程度上可以通过句子的词语来预测，但其中一部分取决于说话人的朗读方式。我们研究了语言模型能从文本中恢复多少突显信息，在被询问时能否将其报告出来，以及语音信号能带来多少额外信息。在标注了词级突显的英语朗读语音上，我们在相同的词上比较了冻结和微调的文本模型、经提示的大语言模型（LLM）以及语音模型。所有文本模型的最终性能彼此相差约一个百分点以内，在三类任务上最好达到68.8%，并且它们在很大程度上犯同样的错误。在我们测试的提示方式下，即使给出500个示例，LLM的表现也不优于“词多数”基线。而线性探针则在其内部表示中发现了更多信息。语音模型达到84.7%。我们认为，文本只决定了突显的一部分，其余部分必须通过听觉才能获得。LLM编码了文本所能决定的内容，但提示并不能将其引出：它们对重音位置的表征……（摘要原文在此处截断）

    arXiv:2611.25808v1 Announce Type: new  Abstract: Prosodic prominence can largely be predicted from the words of a sentence, but part of it depends on how the speaker reads it. We study how much of prominence language models recover from text, whether they can report it when asked, and what the speech signal adds. On read English speech labelled for word-level prominence, we compare frozen and fine-tuned text models, prompted large language models (LLMs) and speech models on the same words. All text models end within about one point of each other, at best 68.8% on the 3-way task, and largely make the same errors. With the prompts we tested, LLMs do no better than a word-majority baseline, even with 500 examples. Linear probes find more in their representations. Speech models reach 84.7%. We argue that text determines only part of prominence, and that the rest has to be heard. LLMs encode what the text determines, but prompting does not bring it out: they represent more about where the e
    
[^45]: RouteVerify：面向混合专家模型逐步验证的路由干预研究

    RouteVerify: Routing Interventions for Step-Level Verification in Mixture-of-Experts Models

    [https://arxiv.org/abs/2611.25780](https://arxiv.org/abs/2611.25780)

    该论文提出 RouteVerify 受控干预框架，通过在混合专家模型的九个检查点上替换路由器 logits、MoE 输出和残差状态等内部激活，系统检验路由状态与逐步验证判断之间的因果关系。

    

    混合专家模型中的路由状态能够区分正确与错误的推理步骤。如果改变路由也会改变模型的逐步验证判断，人们很容易得出结论：该判断遵循路由所编码的信息。我们通过 RouteVerify 对这一推断进行检验，这是一项受控干预研究，它将一个求解轨迹中被选定的内部激活替换为另一个轨迹中的激活，同时保持被评估的输入不变。在九个 MoE 检查点上，我们比较了三种干预接口：路由器 logits、MoE 输出以及 MoE 之后的残差状态。我们改变了激活被替换的层和词元位置，并测量验证边际（verification margin）的变化，该边际量化了模型对 INCORRECT 相对于 CORRECT 的偏好。我们还使用来自正确步骤与错误步骤的激活来评估相同的输入，以检验由此产生的判断是否取决于源步骤的正确性。

    arXiv:2611.25780v1 Announce Type: cross  Abstract: Routing states in mixture-of-experts (MoE) models can distinguish correct from erroneous reasoning steps. If changing routing also changes a model's step-level verification judgment, it is tempting to conclude that the judgment follows what routing encodes. We test this inference with RouteVerify, a controlled intervention study that replaces selected internal activations from one solution trace with activations from another, while keeping the evaluated input unchanged. Across nine MoE checkpoints, we compare three intervention interfaces: router logits, MoE outputs, and post-MoE residual states. We vary the layers and token positions at which activations are replaced. We measure changes in a verification margin that quantifies the model's preference for INCORRECT over CORRECT. We also evaluate the same input with activations from correct versus erroneous steps to test whether the resulting judgment depends on the source step's correct
    
[^46]: 在低推理预算下缩小扩散语言模型的性能差距

    Closing Performance Gap for Diffusion Language Models under Low Inference Budgets

    [https://arxiv.org/abs/2611.25763](https://arxiv.org/abs/2611.25763)

    提出了无需训练的 CaRR 框架，通过 token 承诺重掩码、更新上下文回填和基于轨迹的恢复机制，在低推理预算下提升可编辑扩散语言模型的生成质量，从而缩小其性能差距。

    

    扩散语言模型通过并行更新多个 token，为低预算生成提供了一条架构层面的途径，与逐 token 的自回归解码相比，可以减少顺序解码步骤的数量。然而，在低预算下实现这一优势颇具挑战，因为当去噪和细化步骤的数量减少时，生成质量往往会下降。可编辑扩散模型通过草稿后细化来应对这种质量损失，但在最终提交之前，有用的中间结果可能仍然会丢失。我们提出了 CaRR（Commitment-aware Refinement and Recovery，承诺感知的细化与恢复），这是一个无需训练的框架，它在现有的前向调度内，通过更新上下文回填和基于轨迹的恢复，从固定的可编辑检查点提升提交质量。其中 Token Commitment Remasking（TCR，token 承诺重掩码）会对选定的原生纯替换进行重掩码，并在下一个预定的编辑前向中使用更新后的内容进行回填。

    arXiv:2611.25763v1 Announce Type: cross  Abstract: Diffusion language models offer an architectural route to low-budget generation by updating multiple tokens in parallel, which can reduce the number of sequential decoding steps relative to token-by-token autoregressive decoding. However, realizing this advantage at low budgets is challenging, as generation quality often declines when the number of denoising and refinement steps is reduced. Editable diffusion models address this quality loss through post-draft refinement, but useful intermediate results may still be lost before final submission. We introduce CaRR (Commitment-aware Refinement and Recovery), a training-free framework which improves submission quality from a fixed editable checkpoint through updated-context refill and trajectory-based recovery within its existing forward schedule. Token Commitment Remasking (TCR) remasks selected native pure replacements and refills them on the next scheduled edit forward using the update
    
[^47]: 跨越五个Qwen2.5-Instruct检查点的验证行为

    Verification Behavior Across Five Qwen2.5-Instruct Checkpoints

    [https://arxiv.org/abs/2611.25745](https://arxiv.org/abs/2611.25745)

    本研究通过双AI评分员协议系统评估了五个Qwen2.5-Instruct检查点对信息不充分的声明与验证行为，发现协议提示能可靠引发格式标记（60/60），但诊断执行标准（H3）极少被满足，且预期的随模型规模增长的向上转变并未出现。

    

    我们在四种提示条件下评估了五个Qwen2.5-Instruct检查点（0.5B至32B），每个信息条件使用三个固定任务，每个任务生成五次，两名AI评分员对600条轨迹进行了评分。在协调指令下，两位评分员记录的严格不充分声明在7B时为11/15例，14B时为0/15例，32B时为5/15例。当请求检测器触发时，运行程序发送了标准后续请求，三个检查点分别在其中15、12和15个案例中收到至少一次后续请求。在3B至32B的信息充足任务上，协议轨迹中出现了60/60的评分格式标记，而基线轨迹中为0/60。经过裁决后，每种条件下各有1/60满足诊断执行标准（H3）。在所有检查点和提示中，300条信息充足轨迹中有7条在裁决后满足H3，且全部集中在同一任务上。本次运行中未出现预先设定的向上转变。

    arXiv:2611.25745v1 Announce Type: new  Abstract: We evaluate five Qwen2.5-Instruct checkpoints (0.5B to 32B) under four prompting conditions using three fixed tasks per information condition and five generations per task. Two AI raters scored 600 trajectories. Under the coordinated instruction, both raters recorded strict declarations of insufficiency in 11/15 cases at 7B, 0/15 at 14B, and 5/15 at 32B. The runner sent a standard follow-up when a request detector fired. The three checkpoints received at least one follow-up in 15, 12, and 15 cases, respectively. On information-sufficient tasks at 3B through 32B, scored format markers appeared in 60/60 protocol trajectories and 0/60 baseline trajectories. After adjudication, 1/60 in each condition met the diagnostic-execution criterion (H3). Across all checkpoints and prompts, 7/300 information-sufficient trajectories met H3 after adjudication, all on the same task. The prespecified upward transition did not occur in this run. The results
    
[^48]: PosteriorBench：评估语言模型中行为后验采样能力的基准

    PosteriorBench: Evaluating Behavior Posterior Sampling in Language Models

    [https://arxiv.org/abs/2611.25735](https://arxiv.org/abs/2611.25735)

    该论文提出了 PosteriorBench 基准，用于从计算成本和与真实后验的 KL 散度两个维度评估语言模型的行为后验采样算法，并发现现有方法均无法同时兼顾高效性与保真度。

    

    语言模型中的罕见行为可能难以研究，因为通过普通采样来获得这些行为往往需要极其大量的样本。由此引出了行为后验采样问题：给定一个语言模型和一个感兴趣的行为，是否存在一种算法，能够从该模型在展现该行为的对话记录上的分布中高效地进行采样？我们提出了 PosteriorBench，这是一个用于比较算法的基准，比较维度包括计算成本，以及其采样结果与真实后验在正向和反向 KL 散度上的接近程度。PosteriorBench 包含基于 Qwen3.5-0.8B 的 21 个任务，行为概率范围从 10^-5 到 10^-1，每个任务都配有来自一个包含十亿条标注响应的语料库的精确后验样本。我们发现，我们评估的所有方法中没有一种能同时做到高效且忠实：诸如提示（prompting）这样的简单方法成本低廉，但其与后验之间的 KL 散度会停滞在较高水平；对后验样本进行微调的方法……（摘要在此处截断）

    arXiv:2611.25735v1 Announce Type: cross  Abstract: Rare language model behaviors can be difficult to study because obtaining them through ordinary sampling can require a very large number of samples. This motivates the behavior posterior sampling problem: given a language model and a behavior of interest, is there an algorithm that efficiently samples from the model's distribution over transcripts that exhibit this behavior? We introduce PosteriorBench, a benchmark for comparing algorithms by their compute cost and by how close their samples are to the true posterior in forward and reverse KL divergence. PosteriorBench contains 21 tasks on Qwen3.5-0.8B with behavior probabilities from $10^{-5}$ to $10^{-1}$, each with exact posterior samples from a corpus of one billion labeled responses. We find that no method we evaluate is both efficient and faithful: simple methods like prompting are cheap but plateau at a high KL divergence from the posterior, fine-tuning on posterior samples is o
    
[^49]: VR-SparseQuant：在Vera Rubin上基于2:4稀疏性的精确FP4量化

    VR-SparseQuant: Accurate FP4 Quantization with 2:4 Sparsity on Vera Rubin

    [https://arxiv.org/abs/2611.25717](https://arxiv.org/abs/2611.25717)

    VR-SparseQuant利用Vera Rubin的逐元素2:4稀疏性，在与稠密量化相同的FP4数值预算下精炼误差较大的激活，在保持原生FP4硬件执行的同时有效恢复四比特量化的模型精度。

    

    近年来，GPU对低精度计算的专用支持日益增强，从Hopper的FP8发展到Blackwell的FP4张量核心。精度格式也随之演进，从Blackwell上的MXFP4和NVFP4，到Vera Rubin上采用直接UE5M3块缩放的E2M1 FP4，我们将其称为VRFP4。这些进展有助于以更低成本服务更大的语言模型，但四比特激活量化会显著降低模型质量。我们提出VR-SparseQuant，它利用Vera Rubin的逐元素2:4稀疏性，在保持原生FP4执行的同时恢复精度。通过允许两个稀疏流重叠，我们以与稠密量化相同的FP4数值预算来精炼误差较大的激活。离线缩放比校准避免了运行时搜索，而共享权重的GEMM则高效地累加两个流。在Nemotron-3-Nano-30B-A3B上，VR-SparseQuant降低了WikiText-2困惑度

    arXiv:2611.25717v1 Announce Type: new  Abstract: In recent years, GPUs have incorporated increasingly specialized support for low-precision computation, from Hopper's FP8 to Blackwell's FP4 Tensor Cores. Precision formats have evolved alongside this hardware, from MXFP4 and NVFP4 on Blackwell to E2M1 FP4 with direct UE5M3 block scaling on Vera Rubin, which we term as $\boldsymbol{VRFP4}$. These advances help serve larger language models at lower cost, but four-bit activation quantization can substantially degrade model quality. We introduce VR-SparseQuant, which uses Vera Rubin's elementwise 2:4 sparsity to recover accuracy while retaining native FP4 execution. By allowing two sparse streams to overlap, we use the same FP4 value budget as dense quantization to refine activations with larger errors. Offline scale-ratio calibration avoids runtime search, while a shared-weight GEMM accumulates both streams efficiently. On Nemotron-3-Nano-30B-A3B, VR-SparseQuant reduces WikiText-2 perplexi
    
[^50]: STARK：迈向真正会学习与记忆的个性化智能体

    STARK: Moving towards Personalized Agents That Truly Learn and Remember

    [https://arxiv.org/abs/2611.25705](https://arxiv.org/abs/2611.25705)

    STARK提出了一种双图记忆架构，通过情景图记录用户交互事件、洞察图提取并连接行为模式，使个性化智能体能够真正学习和记住用户的行为规律。

    

    长期记忆对于在长时间内与用户交互的个性化智能体系统至关重要，然而当前的架构只能捕获“发生了什么”，而将“为什么会发生”的推理留给底层大语言模型事后推导。这些事件不仅能够传达用户经历了什么，还能传达这些经历如何塑造了用户的行为模式。我们提出了STARK，一种双图记忆架构，它将这些事件不仅视为节点，还视为行为洞察之间的边。STARK维护一个情景图，将交互记录为在时间、因果和语义上相互关联的情景；以及一个洞察图，用于表示由大语言模型提取的行为信号，并根据支持这些信号的事件数量进行评分，同时通过引发这些洞察的事件将它们相互连接。在推理时，行为知识可以直接从由事件作为边所支撑的洞察节点中检索获得。

    arXiv:2611.25705v1 Announce Type: cross  Abstract: Long-term memory is essential for personalized agentic systems that interact with users over long time periods, yet current architectures only capture what happened while leaving the reasoning of why it happened to be derived later by underlying LLMs.%, and silo knowledge within individual users. These events can, however, convey not only what a user experienced but also how that experience shaped their behavioral patterns. We introduce STARK, a dual-graph memory architecture which treats these events not just as nodes but also as edges between behavioral insights. STARK maintains an Episodic Graph which records interactions as temporally, causally, and semantically linked episodes; and an Insight Graph which represents LLM-extracted behavioral signals scored by how many events support them, connected by the very events that led to them. At inference, behavioral knowledge is retrieved directly from insight nodes grounded by event-as-ed
    
[^51]: 从地理记忆到大语言模型中的空间泛化

    From Geographic Memory to Spatial Generalization in Large Language Models

    [https://arxiv.org/abs/2611.25703](https://arxiv.org/abs/2611.25703)

    该论文提出将大语言模型中的地理知识组织为潜在空间场，通过连续空间编码器将位置表示作为空间标记注入模型，从而实现从仅能回忆训练中见过的“地理空间记忆”向能预测未见区域的“空间泛化”能力的跨越。

    

    大语言模型（LLM）能够将地点与地理描述相关联，但恢复训练中观察到的关联并不能证明其具备跨地理空间泛化的能力。我们区分了“地理空间记忆”（即恢复训练期间观察到的位置—描述关联）与“空间泛化”（即为未观察区域中的位置预测描述）。我们研究连续的位置表示能否改善这种泛化能力，并促使地理知识在大语言模型内部形成空间化的组织。我们将这种组织形式构建为一个潜在空间场：一种通过空间依赖关系连接的、从位置到描述的学习映射。基于连续空间编码器，我们将位置表示投影到模型的嵌入空间中，并将其作为空间标记提供。我们在两个新构建的地理文本基准（分别针对洛杉矶和伦敦）上评估了所得模型。

    arXiv:2611.25703v1 Announce Type: cross  Abstract: Large language models (LLMs) can associate places with geographic descriptions, but recovering an observed association does not establish an ability to generalize across geographic space. We distinguish geospatial memory, the recovery of location--description associations observed during training, from spatial generalization, the prediction of descriptions for locations in unobserved regions. We investigate whether continuous location representations improve this generalization and encourage geographic knowledge to become spatially organized within an LLM. We frame this organization as a latent spatial field: a learned mapping from locations to descriptions connected through spatial dependencies. Building on a continuous spatial encoder, we project location representations into the model's embedding space and supply them as spatial tokens. We evaluate the resulting models on two newly curated geo-text benchmarks for Los Angeles and Lon
    
[^52]: FoodReact-AR：学习零样本大语言模型在阿拉伯语粮食不安全话语中所遗漏的受众反应

    FoodReact-AR: Learning Audience Reactions that Zero-Shot LLMs Miss in Arabic Food-Insecurity Discourse

    [https://arxiv.org/abs/2611.25682](https://arxiv.org/abs/2611.25682)

    该研究将74,241条阿拉伯语粮食不安全Facebook帖子的表情反应聚类为四种受众画像，发现“轻视”与“悲伤”反应分别随2022年小麦价格冲击和2024年加沙粮食危机显著上升，而零样本阿拉伯语大语言模型虽能较好预测悲伤画像（83.8%）却几乎无法识别轻视画像（仅3.4%），揭示了文本中隐藏的受众反应信号。

    

    Facebook表情反应记录了受众如何接收一条帖子，这一信号在帖子文本中并不存在。我们将来自公共Facebook页面的74,241条阿拉伯语粮食不安全帖子（2020-2024年）上的六种差异化反应视为成分数据，应用中心对数比（CLR）变换，并将其聚类为四种受众画像：团结（以“爱”为主）、轻视（以“哈哈”为主）、悲伤（以“难过”为主）以及扁平的矛盾（Ambivalent）画像。轻视反应在2022年达到峰值（22.3%），即小麦价格冲击之年；悲伤反应在2024年达到峰值（14.0%），正值加沙粮食紧急状态期间。基于页面聚类的广义估计方程（GEE）证实了这两种转变（优势比分别为2.18和1.34），且这些结果在最小反应数过滤、替代零值替换与聚类方法以及自助重采样下依然稳健。随后，我们测试了能否在不依赖反应数据的情况下预测这些画像。零样本阿拉伯语大语言模型对83.8%的帖子正确识别出悲伤画像，但对轻视画像的识别率仅为3.4%；在匹配消融实验中，它表现出……（摘要此处截断）

    arXiv:2611.25682v1 Announce Type: new  Abstract: Facebook reactions record how an audience received a post, a signal absent from the post's text. We treat the six differentiated reactions on 74,241 Arabic food-insecurity posts from public Facebook pages (2020-2024) as compositions, apply a centered log-ratio transformation, and cluster them into four audience profiles: Solidarity (Love-dominant), Dismissal (Haha-dominant), Grief (Sad-dominant), and a flat Ambivalent profile. Dismissal peaks in 2022 (22.3%), the year of the wheat price shock, and grief in 2024 (14.0%), during the Gaza food emergency; a page-clustered GEE confirms both shifts (odds ratios 2.18 and 1.34), and they survive minimum-reaction filtering, alternative zero replacements and clustering methods, and bootstrap resampling. We then test whether the profiles can be predicted without reactions. A zero-shot Arabic LLM recovers Grief for 83.8% of posts but Dismissal for only 3.4%, and in a matched ablation, it shows the r
    
[^53]: SEAF：用于多模态讽刺检测的选择性语义增强与视觉证据融合

    SEAF: Selective Semantic Enrichment and Visual Evidence Fusion for Multimodal Sarcasm Detection

    [https://arxiv.org/abs/2611.25672](https://arxiv.org/abs/2611.25672)

    提出SEAF双分支框架，通过选择性语义增强和基于文本锚定加权聚合的视觉证据融合，有效利用MLLM生成的实体背景知识与多图像视觉特征，提升多模态讽刺检测性能。

    

    多模态讽刺检测需要文本、视觉和上下文证据，尤其是当多张图像提供互补线索时。多模态大语言模型（MLLM）能够根据帖子的文本和图像生成辅助信息，但其输出可能将有用的背景知识与冗余的视觉描述混杂在一起。我们提出了SEAF，一个用于选择性语义增强与视觉证据融合的双分支框架。SEAF将生成的字段分配给不同的角色：实体背景用于增强语义输入，而图像描述则提供对齐统计数据。语义分支结合了原始帖子文本、来自冻结MLLM的背景信息以及光学字符识别（OCR）文本。对齐的视觉分支将全局和局部视觉特征与跨模态及图像间相似性统计相结合。文本锚定平滑聚合机制（TASA）根据图像与原始帖子文本的相似度对图像进行加权。

    arXiv:2611.25672v1 Announce Type: cross  Abstract: Multimodal sarcasm detection requires textual, visual, and contextual evidence, particularly when multiple images provide complementary cues. Multimodal large language models (MLLMs) can generate auxiliary information from a post's text and images, but their outputs may mix useful background knowledge with redundant visual descriptions. We propose SEAF, a dual-branch framework for selective semantic enrichment and visual evidence fusion. SEAF assigns generated fields to distinct roles: entity background enriches semantic input, while image descriptions supply alignment statistics. The semantic branch combines the original post text, background from a frozen MLLM, and optical character recognition (OCR) text. The aligned visual branch combines global and local visual features with statistics of cross-modal and inter-image similarity. Text-Anchored Smoothed Aggregation (TASA) weights images by their similarity to the original post text w
    
[^54]: ColorRef：为颜色修订校准言语反馈

    ColorRef: Calibrating Verbal Feedback for Color Revision

    [https://arxiv.org/abs/2611.25653](https://arxiv.org/abs/2611.25653)

    该论文提出颜色指代游戏 ColorRef，揭示语言模型虽然能正确理解颜色修改的方向和分级用语，却无法据此产生幅度合适的颜色调整步长。

    

    人们在请他人修改设计时很少指定确切的颜色：在看到方案后，他们可能会说“稍微暗一点”或“更蓝一点”。此时，语言模型助手不仅需要推断颜色应该朝哪个方向改变，还要推断改变的幅度有多大。我们通过 ColorRef 来研究这一区别——这是一个颜色指代游戏，模型根据描述猜测颜色，而一位知晓（对猜测者隐藏的）记录目标的教师提供言语纠正。在 1,000 个众包的描述-目标对上，三轮确定性神谕反馈将平均目标距离从 42.33 ΔE76 降低到 22.15 ΔE76。然而，这些纠正提供了额外信息，仅凭改进本身并不能证明模型能够良好地遵循经过校准的反馈。一项受控的 Qwen3-14B 诊断实验揭示了一个更具体的失败：遵循所要求的方向并对分级用语进行排序，并不能保证合适的步长；强力……

    arXiv:2611.25653v1 Announce Type: new  Abstract: People rarely specify an exact color when asking someone to revise a design: they might say \textit{a little darker} or \textit{more blue} after seeing a proposal. A language-model assistant must then infer not only which way to change the color, but how far. We study this distinction with \textsc{ColorRef}, a color reference game in which a model guesses a color from a description and a teacher, knowing a recorded target hidden from the guesser, supplies verbal corrections. On 1{,}000 crowd-sourced description--target pairs, three rounds of deterministic oracle feedback reduce mean target distance from 42.33 to 22.15 $\Delta E_{76}$. However, those corrections provide additional information, and improvement alone does not establish well-calibrated feedback following. A controlled Qwen3-14B diagnostic reveals a more specific failure: following a requested direction and ordering graded phrases do not guarantee a suitable step size; strong
    
[^55]: TRAC：通过长度感知解码实现高效机器翻译的扩散语言模型

    TRAC: Efficient Diffusion Language Models for Machine Translation via Length-Aware Decoding

    [https://arxiv.org/abs/2611.25643](https://arxiv.org/abs/2611.25643)

    提出了无需训练的自适应解码方法TRAC，通过根据源语言长度和翻译方向动态调整目标生成长度、加速token提交，减少了扩散语言模型机器翻译中的冗余计算，在保持翻译质量的同时显著降低生成延迟。

    

    使用大语言模型进行机器翻译需要同时具备高翻译质量和低生成延迟。因此，在保持质量的同时加速翻译对实际部署至关重要。扩散大语言模型可以通过迭代去噪在任意目标位置生成token，为超越顺序自回归解码的并行翻译提供了新范式。然而，现有方法仍然存在冗余计算：固定的目标生成长度会在未使用的位置上浪费计算资源，而仅基于置信度的提交策略会延迟那些预测已基本稳定的中等置信度token。为解决这些问题，我们提出了TRAC，一种无需训练的扩散机器翻译自适应解码方法。TRAC利用源语言长度和翻译方向来自适应地调整目标生成长度，对于空输出或异常短的输出则保守地回退到最大长度。

    arXiv:2611.25643v1 Announce Type: new  Abstract: Machine translation with large language models requires both high translation quality and low generation latency. Accelerating translation while preserving quality is therefore essential for practical deployment. Diffusion large language models (dLLMs) can generate tokens at arbitrary target positions through iterative denoising, offering a new paradigm for parallel translation beyond sequential autoregressive decoding. However, existing methods still incur redundant computation: fixed target generation lengths waste computation on unused positions, while confidence-only commitment delays medium-confidence tokens whose predictions have largely stabilized. To address these issues, we propose TRAC, a training-free adaptive decoding method for diffusion machine translation. TRAC adapts the target generation length using source length and translation direction, with a conservative fallback to the maximum length for empty or unusually short o
    
[^56]: 生成式新闻推荐再现流行度：基于记忆化的分析

    Generative News Recommendation Reproduces Popularity: A Memorization-Based Analysis

    [https://arxiv.org/abs/2611.25606](https://arxiv.org/abs/2611.25606)

    该研究利用约2000万条日本新闻浏览会话数据，首次从训练数据记忆化的角度揭示生成式新闻推荐再现流行内容的机制——被频繁点击的新闻标题更容易被模型记住，并在更多样化的用户上下文中被生成，从而导致信息多样性下降。

    

    基于文本的生成式推荐器能够通过直接生成物品标题或其他文本表示，超越对现有物品进行排序。在新闻领域，这引发了一个社会层面的担忧：模型可能反复推送本已热门的文章，从而缩小用户接触信息的多样性。我们将训练数据记忆化作为导致这种集中现象的机制之一加以研究。利用来自日本某新闻服务的约2000万条浏览会话数据，我们对一个经过指令微调的Llama 3 8B模型进行持续训练，使其生成会话中的下一篇新闻标题。研究发现，生成标题的分布比浏览日志中实际观察到的标题分布更加集中、多样性更低。频繁重复出现的目标标题表现出更强的逐字记忆与近似记忆，而记忆化程度越高的标题，会在越广泛的提示（上下文）中被生成。这些发现支持一条……

    arXiv:2611.25606v1 Announce Type: cross  Abstract: Text-based generative recommenders can go beyond ranking existing items by directly generating item titles or other textual representations. In the news domain, this raises a societal concern that models may repeatedly surface already popular articles and thereby narrow the diversity of information to which users are exposed. We investigate training-data memorization as one mechanism behind this concentration. Using approximately 20 million browsing sessions from a Japanese news service, we continually trained an instruction-tuned Llama 3 8B model to generate the next article title in a session. The generated-title distribution was more concentrated and less diverse than the distribution of titles observed in the browsing logs. Frequently repeated target titles exhibited greater verbatim and approximate memorization, and more strongly memorized titles were generated in response to a wider range of prompts. These findings support a path
    
[^57]: 量规语义还是数值先验？审计大语言模型评审中的量表方向偏差

    Rubric Semantics or Numerical Priors? Auditing Scale-Direction Bias in LLM Judges

    [https://arxiv.org/abs/2611.25588](https://arxiv.org/abs/2611.25588)

    该论文提出“语义忠实性”概念以及量表反转探针和地板归一化方法，揭示了大语言模型自动评审普遍存在量表方向偏差——评分依赖数值锚点而非量规语义，并证明双向锚点平均可有效缓解该问题。

    

    近期的基准评估越来越依赖大语言模型（LLMs）作为自动评审。这一范式假设LLM是基于量规的语义含义而非其数值量表进行评估的。我们通过引入“语义忠实性”原则来质疑这一假设：语义等价的量规必须产生一致的评分。为了量化这一点，我们提出了一种量表反转探针，它在不改变评估标准的情况下反转评分锚点。此外，我们引入了地板归一化方法，将反转分歧相对于模型固有变异性进行校准，从而实现公平的跨模型比较。在涵盖开源权重和专有模型的七个多样化LLM评审者的五个元评估基准上进行的实验揭示了语义忠实性的普遍违背，表明仅凭原始反转一致性具有误导性。最后，我们证明了双向锚点平均能够降低对任一单一量表方向的依赖。

    arXiv:2611.25588v1 Announce Type: new  Abstract: Recent benchmark evaluations increasingly rely on Large Language Models (LLMs) as automatic judges. This paradigm assumes LLMs evaluate based on the rubric's semantic meaning rather than its numeric scale. We question this assumption by introducing semantic faithfulness: the principle that semantically equivalent rubrics must yield consistent scores. To quantify this, we propose a scale-reversal probe that inverts score anchors without altering evaluation criteria. Furthermore, we introduce floor normalization to calibrate reversal disagreement against intrinsic model variability, enabling fair cross-model comparisons. Experiments across five meta-evaluation benchmarks with seven diverse LLM judges spanning open-weight and proprietary models reveal widespread violations of semantic faithfulness, showing that raw reversal agreement alone is misleading. Finally, we demonstrate that bidirectional anchor averaging reduces dependence on any s
    
[^58]: AutoRAGBench：用于自动化RAG系统开发的LLM智能体基准测试

    AutoRAGBench: A Benchmark for LLM Agents Automating RAG System Development

    [https://arxiv.org/abs/2611.25582](https://arxiv.org/abs/2611.25582)

    本文提出了首个基准测试AutoRAGBench，通过九个任务评估LLM智能体在无起始代码和预设策略的条件下，于开放代码空间中自主从零构建并迭代优化RAG系统的能力。

    

    检索增强生成（RAG）是一种成熟的范式，通过从外部语料库检索的信息来增强大语言模型（LLM）。然而，开发有效的RAG系统需要大量的工程工作和迭代实验，因为有效的设计选择因任务和领域而异。随着LLM智能体能力的快速进步，我们提出这样一个问题：这些智能体能够多好地自主构建RAG系统，以及它们能够通过迭代多有效地改进这些系统。在本文中，我们提出了AutoRAGBench，这是第一个评估LLM智能体在开放代码空间中从零开始开发RAG系统能力的基准测试。它包含九个RAG系统开发任务。对于每个任务，智能体在没有起始代码或预设策略的情况下，在单个H800 GPU上六小时的预算内从零构建RAG系统。它们分析任务数据、实现RAG流水线，并基于实验结果迭代地优化其设计……

    arXiv:2611.25582v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) is a well-established paradigm for augmenting large language models (LLMs) with information retrieved from external corpora. However, developing effective RAG systems requires substantial engineering effort and iterative experimentation, as effective design choices vary across tasks and domains. With rapid advances in LLM agent capabilities, we ask how well these agents can autonomously build RAG systems and how effectively they can improve them through iteration. In this paper, we propose AutoRAGBench, the first benchmark to evaluate how well LLM agents develop RAG systems from scratch in an open code space. It comprises nine RAG system development tasks. For each task, agents build RAG systems from scratch without starter code or prescribed strategies, under a six-hour budget on a single H800 GPU. They analyze task data, implement RAG pipelines, and iteratively refine their designs based on experi
    
[^59]: 覆盖-选择差距：限制LLM评分标准构建的是排序而非生成

    The Coverage Selection Gap: Ranking, Not Generation, Limits LLM Rubric Construction

    [https://arxiv.org/abs/2611.25557](https://arxiv.org/abs/2611.25557)

    该论文发现在自动评分标准构建中，限制性能的瓶颈是对候选标准的排序选择而非标准生成，并提出“覆盖-选择差距”指标来量化无参考排序与有参考贪婪诊断之间的性能差异。

    

    任务特定的评分标准正被越来越多地用于评估开放式的语言模型输出，然而从复杂的源文档中构建全面的评估标准仍然具有挑战性。我们研究了专业并购尽调场景中的自动评分标准构建问题，在该场景下，评估标准必须从任务指令和支持文档中推断得出，而无法获取专家撰写的参考评分标准。我们引入了一个多轴生成框架，该框架从互补的结构化和自由形式视角探索源证据，以构建一个广泛的候选评估标准池。为了区分评估标准发现的失败与候选选择的失败，我们提出了“覆盖-选择差距”这一概念，用于衡量在固定选择预算下，无参考排序方法与基于参考的贪婪诊断方法之间的性能差异。该方法在16个任务上开发，并在4个留出任务上进行评估。

    arXiv:2611.25557v1 Announce Type: cross  Abstract: Task-specific rubrics are increasingly used to evaluate open-ended language-model outputs, yet constructing comprehensive evaluation criteria from complex source documents remains challenging. We study automatic rubric construction in professional mergers and acquisitions (M&A) due diligence, where evaluation criteria must be inferred from task instructions and supporting documents without access to an expert-authored reference rubric. We introduce a multi-axis generation framework that explores complementary structural and free-form views of the source evidence to construct a broad pool of candidate evaluation criteria. To distinguish failures of criterion discovery from failures of candidate selection, we introduce the coverage-selection gap, which measures the difference between reference-free ranking and a reference-informed greedy diagnostic under a fixed selection budget. Developed on 16 tasks and evaluated on four held-out tasks
    
[^60]: COVERAGE-TRACE：基于源文本侧反事实追踪的大语言模型摘要遗漏检测方法

    COVERAGE-TRACE: Source-Side Counterfactual Tracing for Detecting Omissions in LLM Summaries

    [https://arxiv.org/abs/2611.25539](https://arxiv.org/abs/2611.25539)

    提出COVERAGE-TRACE源侧反事实框架，通过替换源文本原子事实并测量摘要似然变化来检测大语言模型摘要中的遗漏，在QMSum数据集上达到0.764–0.825的AUROC，显著优于强基线的0.640。

    

    遗漏难以验证，因为缺失的信息不会留下可供检查的输出片段：一份摘要在事实层面可能是有依据的，却仍然遗漏了保持原文含义所必需的决策、限定条件、数量或条件。我们提出COVERAGE-TRACE，这是一个源文本侧的反事实框架，将遗漏评估为摘要对指定原子源事实的依赖程度。该方法仅用类型匹配的替代内容替换目标事实，同时保持候选摘要不变，并测量其教师强制、长度归一化似然所产生的变化。覆盖目标事实的摘要应比遗漏该事实的摘要产生更强的反应，而跨多个反事实样本的聚合则降低了对替换选择的敏感性。我们在80对匹配的覆盖-遗漏QMSum样本（涵盖多种事实类型）和四个开放权重模型上评估了COVERAGE-TRACE。Multi-CF方法达到了0.764–0.825的AUROC，而强基线仅为0.640。

    arXiv:2611.25539v1 Announce Type: new  Abstract: Omissions are difficult to verify because missing information leaves no output span to inspect: a summary may be factually supported while still excluding a decision, qualification, quantity, or condition needed to preserve the source's meaning. We introduce COVERAGE-TRACE, a source-side counterfactual framework that evaluates omission as dependence on a designated atomic source fact. It replaces only the target fact with type-matched alternatives, holds the candidate summary fixed, and measures the resulting change in its teacher-forced, length-normalized likelihood. A summary covering the target should respond more strongly than one omitting it, while aggregation across multiple counterfactuals reduces sensitivity to replacement choice.   We evaluate COVERAGE-TRACE on 80 matched covered--omitted QMSum pairs spanning several fact types and four open-weight models. Multi-CF achieves 0.764--0.825 AUROC, compared with 0.640 for the stronge
    
[^61]: 基于语言模型的个性化旅行规划器

    LM-based Personalized Travel Planner

    [https://arxiv.org/abs/2611.25531](https://arxiv.org/abs/2611.25531)

    提出了一种层次化多智能体旅行规划框架，由监督智能体协调多个专门智能体，并基于多维评分机制进行迭代优化，在TravelPlanner基准上将预算约束规划性能提升了10%。

    

    有效的旅行规划需要处理多个信息源、管理约束条件，并根据用户偏好定制推荐。虽然大语言模型（LLM）带来了动态的、上下文感知的决策能力，但单智能体模型在复杂的多步推理、约束遵循和模块化执行方面往往存在困难。为了克服这些局限性，我们提出了一个层次化多智能体框架，其中一个监督型聊天机器人智能体负责协调专门的智能体，分别用于行程验证、用户查询构建、实时数据检索和基于约束的行程生成。通过集成LLM推理、外部API以及基于确定性多维评分框架的迭代优化机制，我们的方法确保了适应性、高效性和个性化。在著名的TravelPlanner基准上的评估表明，预算约束规划性能提升了10%，在更广泛的约束集规划方面也有所提升……

    arXiv:2611.25531v1 Announce Type: new  Abstract: Effective travel planning requires handling multiple information sources, managing constraints, and tailoring recommendations to user preferences. While LLMs have introduced dynamic, context-aware decision-making, single-agent models often struggle with complex multi-step reasoning, constraint adherence, and modular execution. To overcome these limitations, we propose a hierarchical multi-agent framework where a supervising chatbot agent coordinates specialized agents for schedule validation, user query formulation, real-time data retrieval, and constraint-aware itinerary generation. By integrating LLM reasoning, external APIs, and iterative refinement mechanisms based on a deterministic, multi-dimensional scoring framework, our approach ensures adaptability, efficiency, and personalization. Evaluations on the well-known TravelPlanner benchmark demonstrate improvements in Budget Constraint planning by 10\%, wider constraints set-based pl
    
[^62]: MATE：面向安全高效移动控制的移动智能体可信执行

    MATE: Mobile Agent Trustworthy Execution for Safe and Efficient Mobile Control

    [https://arxiv.org/abs/2611.25526](https://arxiv.org/abs/2611.25526)

    MATE是一个无需修改底层模型的执行层系统，通过结构化动作记录、运行时护栏、预定义任务路由和浮动界面，将移动智能体高风险行为拦截率从29.7%提升至97.0%，同时减少最多6倍的LLM调用并将重复任务成功率提升至95.7%。

    

    个人智能手机上的移动智能体可能会影响私密数据、支付、设置以及前台屏幕。我们提出了MATE（移动智能体可信执行），这是一个执行层系统，无需修改底层模型即可控制智能体动作提议与设备实际效果之间的边界。MATE将每个UI原语表示为结构化动作记录，该记录结合了指令、目标应用、屏幕上下文、原语和动作等字段。运行时护栏（RG）将该记录映射为允许、阻止或需用户批准的决策；预定义任务路由（PTR）通过经过验证的路径及执行后的屏幕检查来处理重复性请求；而浮动界面则使执行保持在紧凑的悬浮层中。在MobileSafetyBench上，MATE在使用GPT-4o时将高风险行为拦截率从29.7%提升至97.0%。在AndroidWorld上，它将LLM调用次数最多减少6倍，并将重复使用任务的成功率提升至95.7%。

    arXiv:2611.25526v1 Announce Type: cross  Abstract: Mobile agents on personal smartphones can affect private data, payments, settings, and the foreground screen. We present MATE (Mobile Agent Trustworthy Execution), an execution-layer system that controls the boundary between agent action proposals and device effects without modifying the underlying model. MATE represents each UI primitive as a structured action record that combines the instruction, target app, screen context, primitive, and action fields. Runtime Guardrail (RG) maps this record to an allow, block, or user-approval decision; Predefined Task Routing (PTR) serves recurring requests through verified routes with post-execution screen checks; and a floating interface keeps execution in a compact overlay. On MobileSafetyBench, MATE raises high-risk blocking from 29.7% to 97.0% under GPT-4o. On AndroidWorld, it reduces LLM calls by up to 6x and raises recurring-use task success to 95.7%.
    
[^63]: DRVEXO-4B-Unified：模块化语音-视觉系统的设计与评估

    DRVEXO-4B-Unified: Design and Evaluation of a Modular Speech-Vision System

    [https://arxiv.org/abs/2611.25521](https://arxiv.org/abs/2611.25521)

    本文提出基于4B语言骨干的模块化多模态助手DRVEXO-4B-Unified，通过时序残差连接器、显式文本边界和任务条件化评分头将语音、视觉与文本交互统一到一个自包含检查点中，并在MMMU、IFEval和MATH-500基准上超越同规模的Gemma 3 4B IT、InternVL3.5-4B与Qwen3-4B等模型。

    

    我们提出了DRVEXO-4B-Unified，一个围绕40亿（4B）参数语言骨干构建的模块化多模态助手。该系统以一个自包含的检查点将文本、图像和语音输入与基于参考条件的语音输出及结构化交互相结合。其中，时序残差连接器将声学序列适配到语言接口；显式文本边界支持增量式语音合成；三个任务条件化的评分头可在生成式工具调用的同时对所提供的动作候选进行排序。这些接口将对话与应用定义的选择相连接，同时将动作的最终授权留给应用程序。该系统在MMMU全部900个验证问题上取得74.22%的成绩，在未使用测试时扩展的情况下超过了Gemma 3 4B IT和InternVL3.5-4B已公布的验证分数。严格提示级IFEval准确率为82.26%，MATH-500准确率为89.40%，分别高于原始Qwen3-4B非思考模式报告的81.2%和84.8%。在NVIDIA G……（原文摘要在此处截断）

    arXiv:2611.25521v1 Announce Type: new  Abstract: We present DRVEXO-4B-Unified, a modular multimodal assistant built around a 4B language backbone. A self-contained checkpoint combines text, image, and speech inputs with reference-conditioned speech output and structured interaction. A temporal residual connector adapts acoustic sequences to the language interface; explicit text boundaries support incremental speech synthesis; and three task-conditioned scoring heads rank supplied action candidates alongside generative tool calls. These interfaces connect conversation to application-defined choices while leaving action authorization to the application. The system obtains 74.22% on all 900 MMMU validation questions, exceeding the published validation scores of Gemma 3 4B IT and InternVL3.5-4B without test-time scaling. Strict prompt-level IFEval accuracy is 82.26% and MATH-500 accuracy is 89.40%, above the original Qwen3-4B non-thinking report's 81.2% and 84.8%, respectively. On NVIDIA G
    
[^64]: 语音助手应何时发言？土耳其语的轮次结束检测

    When Should a Voice Agent Speak? End-of-Turn Detection in Turkish

    [https://arxiv.org/abs/2611.25489](https://arxiv.org/abs/2611.25489)

    本研究基于土耳其语双人视频通话语料库，比较微调 TurnGPT、零样本语言模型和多语言检测器在区分话轮结束与轮次内停顿上的表现，发现微调模型效果最佳且校准最好，与说话者时间特征结合可获得最佳的话轮结束检测性能。

    

    一个在回复前等待固定静音时长的语音助手，要么会打断在话语中途停顿的用户，要么对已完成发言的用户响应过晚。本研究考察了静音开始时的决策问题，使用了从土耳其语双人视频通话的 Real-TurnTurk 语料库中提取的轮次结束和轮次内停顿数据。研究比较了经过微调的土耳其语 TurnGPT 模型、零样本土耳其语语言模型和已部署的多语言检测器，并结合语音活动时间特征和面部特征，在记录分离和说话者分离的评估设置下采用配对聚类自助法检验进行评估。在留出的录音上，微调后的模型在区分轮次结束与停顿方面优于两个文本基线，且在三者中校准效果最佳。当前说话者的时间特征同样具有信息量，其与微调文本分数相结合可获得基于时间特征的最佳结果；通过逻辑回归，这一组合还……

    arXiv:2611.25489v1 Announce Type: new  Abstract: A voice agent that waits for a fixed silence before replying either interrupts users who pause mid-utterance or responds late to users who have finished. This study examines the decision at the onset of silence, using turn ends and turn-internal pauses extracted from the Real-TurnTurk corpus of Turkish dyadic video calls. A fine-tuned Turkish TurnGPT model, a zero-shot Turkish language model and a deployed multilingual detector are compared, together with voice-activity timing and facial features, under recording-disjoint and speaker-disjoint evaluation with paired cluster bootstrap tests. On held-out recordings, the fine-tuned model separates turn ends from pauses better than both text baselines and is the best calibrated of the three. Timing features of the current speaker are similarly informative, and their combination with the fine-tuned text score gives the best timing-based results; with logistic regression, this combination also 
    
[^65]: PILAPS：一个跨语用现象与跨评估场景的双语语用推理评测基准

    PILAPS: A Bilingual Benchmark for Evaluating Pragmatic Inference across Phenomena and Settings

    [https://arxiv.org/abs/2611.25459](https://arxiv.org/abs/2611.25459)

    该论文提出了首个英语/普通话双语语用推理评测基准PILAPS，通过评估八个LLM揭示出模型语用能力在模型规模、语用现象和语言间存在显著差异，尤其开源模型在幽默理解上表现薄弱。

    

    近期研究报道大型语言模型（LLM）在英语语用任务上表现优异，但这种表现能否推广到其他语言和评估设置仍不清楚。随着LLM被部署到不同语言、文化和对话情境中，这一问题日益重要。本文首先提出了PILAPS，一个英语/普通话双语基准，涵盖八种语用现象，包括专门编写的条目和源自真实对话的条目，同时包含文化适配数据和独立来源数据。随后，我们评估了来自四个模型家族的八个LLM，涵盖多种模型规模和提示策略。结果显示，不同模型和不同语用现象之间存在显著差异：开源模型在幽默理解方面尤为困难，而跨语言表现随模型来源不同而变化。错误分析表明，与人类参与者相比，模型更容易做出字面理解。

    arXiv:2611.25459v1 Announce Type: new  Abstract: Recent work reports strong pragmatic performance by large language models (LLMs) in English, yet whether this performance generalises across languages and evaluation settings remains unclear. This question is increasingly important as LLMs are deployed across languages, cultures, and conversational contexts. In this paper, we first introduce PILAPS, a bilingual English/Mandarin Chinese benchmark covering eight pragmatic phenomena across purpose-written and dialogue-derived items, with both culturally adapted and independently sourced data. We then evaluate eight LLMs from four families across a range of model sizes and prompting strategies. Our results reveal substantial variation across models and pragmatic phenomena: open-source models continue to struggle particularly with humour, while cross-linguistic performance varies with model origin. Our error analysis shows that, compared with human participants, models are more prone to liter
    
[^66]: PAIR-MED：用于多语言医学推理迭代精炼的策略-奖励交替训练框架

    PAIR-MED: Policy-reward Alternation for Iterative Refinement in Multilingual Medical Reasoning

    [https://arxiv.org/abs/2611.25446](https://arxiv.org/abs/2611.25446)

    该论文提出PAIR-MED交替训练框架，通过让多语言推理策略与医学过程奖励模型相互迭代更新，解决了固定奖励模型与演化策略分布不匹配的问题，从而提升多语言医学推理能力。

    

    大型语言模型展现了强大的医学问答能力，但其推理性能在不同语言之间仍不均衡。过程奖励模型（PRM）能够提供步骤级反馈以改进推理，但在强化学习过程中，固定的PRM可能随策略演化而与策略的轨迹分布日益偏离，从而降低其监督信号的可靠性。我们提出一种策略-奖励交替训练框架，联合调整医学PRM并优化多语言推理策略。在监督初始化之后，当前策略生成多语言轨迹以更新PRM，随后PRM为下一次策略更新提供过程级奖励。医学步骤监督保持PRM的医学基础，而过程、结果和语言相关的奖励共同引导策略学习。我们构建了多语言医学PRM（PAIR-PRM）和推理模型（PAIR-MED）。

    arXiv:2611.25446v1 Announce Type: new  Abstract: Large language models demonstrate strong medical question-answering capabilities, yet reasoning performance remains uneven across languages. Process reward models (PRMs) provide step-level feedback for improving reasoning, but a fixed PRM may become increasingly misaligned with the trajectory distribution of an evolving policy during reinforcement learning, thereby reducing the reliability of its supervision. We propose a policy--reward alternating training framework that jointly adapts a medical PRM and optimizes a multilingual reasoning policy. Following supervised initialization, the current policy generates multilingual trajectories to update the PRM, which then provides process-level rewards for the next policy update. Medical step supervision maintains the PRM's medical foundation, while process, outcome, and language-related rewards guide policy learning. We develop multilingual medical PRMs (PAIR-PRM) and reasoning models (PAIR-M
    
[^67]: RoCoKV：通过不确定性与冗余感知的KV缓存压缩加速大语言模型推理服务

    RoCoKV: Accelerating LLM Inference Serving through Uncertainty and Redundancy Aware KV Cache Compression

    [https://arxiv.org/abs/2611.25433](https://arxiv.org/abs/2611.25433)

    提出RoCoKV方法，通过不确定性引导的自适应剪枝和冗余感知的token选择来压缩大型推理模型的KV缓存，在不损害推理质量的前提下加速大语言模型推理服务。

    

    KV缓存压缩对大型推理模型至关重要，因为其成本的大部分来自生成的长推理轨迹，而非输入提示词。由于循环、自我验证和修正等行为，这些推理轨迹常常包含冗余和低效用的token。然而，这类token可能仍获得较高的注意力分数，使得在不损害推理过程的情况下移除它们变得困难。我们提出了RoCoKV，一种专为大型推理模型设计的KV缓存压缩方法。RoCoKV引入了两个关键组件：第一，不确定性引导的自适应剪枝根据模型的置信度调整压缩强度，在不确定阶段分配更多内存，而在模型自信时施加更强的剪枝；第二，冗余感知的token选择降低与近期推理内容高度相似的token的影响，促使模型保留更具信息量且更多样化的token。除了算法部分……

    arXiv:2611.25433v1 Announce Type: new  Abstract: KV cache compression is critical for large reasoning models, where most of the cost comes from long generated reasoning traces rather than the input prompt. These traces often contain redundant and low-utility tokens due to looping, self-verification, and corrections. However, such tokens may still receive high attention scores, making it difficult to remove them without harming the reasoning process. We propose RoCoKV, a KV cache compression method designed for large reasoning models. RoCoKV introduces two key components. First, uncertainty-guided adaptive pruning adjusts the compression strength based on the model's confidence, allocating more memory during uncertain phases and applying stronger pruning when the model is confident. Second, redundancy-aware token selection reduces the impact of tokens that are highly similar to recent reasoning content, encouraging the model to retain more informative and diverse tokens. Beyond the algo
    
[^68]: 在关键之处进化：覆盖率引导的智能体技能自我进化

    Evolve Where It Matters: Coverage-Guided Self-Evolution of Agent Skills

    [https://arxiv.org/abs/2611.25429](https://arxiv.org/abs/2611.25429)

    提出了 SkillPlumber 框架，利用指令级覆盖率标注区分失败源于指令错误还是未遵循指令，使智能体能够只在技能真正不足之处进行针对性的自我进化。

    

    智能体技能将程序性知识打包为自然语言指令，只有当指令本身正确且被智能体恰当遵循时，技能才能发挥作用。自进化智能体会根据自身的运行记录来修订技能，但现有方法将每次运行作为一个整体来评判——依据其结果或出错的位置——无法区分失败究竟是源于指令本身错误，还是源于智能体没有遵循指令。指令级覆盖率最初被引入用于测试技能，它将一次运行中的每条指令标记为“遵循”、“未遵循”或“未覆盖”，从而区分了上述两类失败原因。我们利用这一机制实现技能自我进化，提出了 SkillPlumber——一个让智能体在覆盖率显示技能不足之处进行自我进化的框架。一个持久的根因分析智能体将每个标签转化为修订问题，并在下一次运行中检验每项修订，同时一个维护者智能体将跨任务的调查结果提炼为共享策略。

    arXiv:2611.25429v1 Announce Type: cross  Abstract: Agent skills package procedural knowledge as natural-language instructions, and a skill helps only when its instructions are correct and properly followed by the agent. Self-evolving agents revise their own skills from the runs they make, but existing methods judge each run as a whole, by its outcome or by where it went wrong, and cannot tell whether a failure comes from a wrong instruction or from one the agent did not follow. Instruction-level coverage was introduced to test skills and labels each instruction of a run as follow, not-follow, or not covered, which separates these causes. We leverage it for skill self-evolution with SkillPlumber, a framework in which an agent evolves its own skills where coverage shows they fall short. A persistent root cause analysis agent turns each label into a revision question and checks each revision against the next run, while a maintainer distills investigations across tasks into a shared strate
    
[^69]: ChronoDossier：作为时序记忆表征的长篇人物档案

    ChronoDossier: Long-Form Personas as Temporal Memory Representations

    [https://arxiv.org/abs/2611.25407](https://arxiv.org/abs/2611.25407)

    该论文提出ChronoDossier基准，将长篇人物档案视为时序记忆表征，利用311份合成档案和内在与外在双重指标，评估长时程智能体在事实随时间变化时记录和恢复人物潜在状态的能力。

    

    长时程智能体依赖存储的记录来在事实发生变化时恢复一个人的状态。可靠的回答取决于事实如何被记录、检索以及呈现给读者。我们将长篇人物档案视为对已知潜在状态的一种记忆表征，即一个不变的身份，加上与人生阶段相关联的事实、例行事务和决策规则的时间线。我们提出了ChronoDossier，这是一个包含311份合成档案的基准，每份档案长度为1千至1万词，由七个生成器根据带日期的时间线撰写而成。内在指标根据预期状态评估这些记录；外在指标通过五个读取器在一个种子化的、跨越不同长度均衡分布的32人设GPT-5.1面板上测试状态恢复能力（每个条件520个评级轮次）。相互关联的状态、记录、证据包和答案支持对档案构建与证据传递的测试。在一项上下文压力对照实验中，限定标签范围的写作器输入消除了检测到的范围外锚点内容复制。在……上编写的冻结问题（原文摘要在此处截断）

    arXiv:2611.25407v1 Announce Type: new  Abstract: Long-horizon agents rely on stored records to recover a person's state as facts change. Reliable answers depend on how facts are recorded, retrieved, and presented to the reader. We treat a long-form persona dossier as a memory representation of a known latent state, an invariant identity plus a timeline of facts, routines, and decision rules tied to life phases. We present ChronoDossier, a benchmark of 311 synthetic dossiers of 1k-10k words written by seven generators from dated timelines. Intrinsic metrics evaluate the records against their intended states; extrinsic metrics test recovery by five readers on a seeded 32-persona GPT-5.1 panel balanced across lengths (520 rated turns per condition). The linked states, records, evidence packets, and answers support tests of construction and evidence delivery. In a contextual-stress control, tag-scoped writer inputs remove detected out-of-scope anchor copying. Frozen questions authored on s
    
[^70]: 将论证框架融入大语言模型决策：面向特定任务的设计与评估

    Incorporating argumentation frameworks for LLM decision-making: Task-specific design and evaluation

    [https://arxiv.org/abs/2611.25361](https://arxiv.org/abs/2611.25361)

    本文抽象出四阶段的论证增强工作流程，发现现有方法在不同任务上提升不一致，并据此提出了Qnt-Arg等两种面向特定任务的方法，以改进LLM在辩论决策与事实核查中的表现。

    

    论证框架提供了显式的、形式化的机制来表示和化解冲突，因此已被应用于大语言模型的决策中。其收益各不相同，取决于论据及其关系如何被构建、评估以及如何融入决策过程。我们基于现有的论证增强方法抽象出一个包含四个阶段的工作流程，并在基于辩论的决策任务和基于检索的事实核查任务上研究其设计选择。我们使用近期的大语言模型进行评估，结果显示现有方法在所考虑的任务中并不能一致地提升性能。基于这一观察，我们开发了两种面向特定任务的方法。第一种称为Qnt-Arg，它围绕辩论材料中的原始发言者来组织论据，并在渐进语义下聚合模型估计的论据强度和候选答案先验……

    arXiv:2611.25361v1 Announce Type: cross  Abstract: Argumentation frameworks (AFs) provide explicit, formal mechanisms for representing and resolving conflicts and thus have been incorporated into large language model (LLM) decision-making. The benefits vary and depend on how arguments and their relations are constructed, evaluated, and incorporated into the decision process. {We abstract a workflow consisting of four stages based on existing argumentation-augmented methods} and investigate their design choices over tasks of debate-based decision-making and retrieval-based fact verification. Our evaluation with recent LLMs shows that existing methods do not consistently improve performance across the considered tasks. Motivated by this observation, we develop two task-specific methods. The first is called Qnt-Arg, which organizes arguments around the original speakers in debate materials and aggregates model-estimated argument strengths and candidate-answer priors under gradual semantic
    
[^71]: Pro$^2$ Agent：面向自主AI研发的科学问题探测方法

    Pro$^2$ Agent: Scientific Problem Probing for Autonomous AI R&D

    [https://arxiv.org/abs/2611.25357](https://arxiv.org/abs/2611.25357)

    Pro$^2$ Agent将自主AI研究的核心状态从代码库转变为显式的问题状态，通过全面分析、定向研究和受控假设检验来识别阻碍成功的问题并匹配候选解决方案，从而避免智能体在困难问题上陷入错误的解区域。

    

    大语言模型（LLM）智能体现在能够通过编写代码和运行实验来自主开展AI研究。它们通常采用迭代优化代码库的方式进行，却忽视了任务背后的问题结构。这种模式在较简单的任务上是有效的，但在困难问题上，智能体由于从未显式地表达其对问题的理解，往往会陷入错误的解区域。我们提出了Pro$^2$ Agent，它将自主AI研究的主要状态从代码库状态转变为问题状态：这是一种显式的结构，用于刻画哪些问题阻碍了成功，以及哪些解决方案有可能解决这些问题。Pro$^2$ Agent通过对可用资源和约束条件的全面分析来初始化问题状态，随后进行有针对性的研究，将识别出的问题与候选解决方案关联起来。通过受控的假设检验，智能体会更新可疑（摘要原文在此处被截断）

    arXiv:2611.25357v1 Announce Type: new  Abstract: Large language model (LLM) agents can now autonomously conduct AI research by writing code and running experiments. They typically do so by iteratively optimizing a codebase, while ignoring the problem structure behind the task. This formulation is effective on simpler tasks, but on difficult problems it often gets stuck in the wrong solution region because the agent never explicitly represents its understanding of the problem. We introduce Pro$^2$ Agent, which shifts the primary state of autonomous AI research from a codebase state to a Problem State: an explicit structure of what problems prevent success and which solutions are likely to address them. Pro$^2$ Agent initializes the Problem State through comprehensive analysis of available resources and constraints. It then performs targeted research to relate the identified problems to candidate solutions. Through controlled hypothesis testing, the agent updates the importance of a susp
    
[^72]: 通过多样化解释增强对语言模型的理解

    Enhancing Understanding of Language Models with Varied Explanations

    [https://arxiv.org/abs/2611.25314](https://arxiv.org/abs/2611.25314)

    本论文系统研究了高亮解释、自然语言解释和思维链推理等多种解释形式，并从忠实性、合理性和实用性三个维度评估其对语言模型可解释性的互补贡献。

    

    语言模型越来越多地被用于在复杂推理任务中辅助人类判断，但其预测往往仍然难以解释和验证。可解释性技术能够深入洞察模型的决策过程，帮助揭示错误、失效模式和潜在偏见，从而让用户能够决定是否信任模型输出。解释是语言模型内部运作的简化近似，其目标是获得尽可能忠实反映底层计算（忠实性）、对用户而言易于理解（合理性）、并能帮助用户实现预期目标（实用性）的解释。评估协议对于衡量这些维度至关重要。本论文研究了高亮解释、自然语言解释和思维链推理三种解释形式。这些形式为模型行为提供了互补的视角，并在不同任务中展现出各自的优势与局限性。

    arXiv:2611.25314v1 Announce Type: new  Abstract: Language models are increasingly used to support human judgement in complex reasoning tasks, yet their predictions often remain difficult to interpret and validate. Explainability techniques provide insight into model decision-making, helping expose errors, failure modes, and potential bias so users can decide whether to rely on model outputs. Explanations are simplified approximations of a language model's internal workings. The goal is to obtain explanations that reflect the underlying computation as closely as possible (faithfulness), are understandable to users (plausibility), and help them achieve their intended goals (utility). Evaluation protocols are essential for assessing these dimensions. This thesis studies highlight explanations, natural language explanations, and chain-of-thought reasoning. These forms offer complementary perspectives on model behaviour and exhibit distinct strengths and limitations across tasks. Producing 
    
[^73]: 大语言模型已经是具备置信度感知的零样本分类器

    LLMs Are Already Confidence-Aware Zero-Shot Classifiers

    [https://arxiv.org/abs/2611.25289](https://arxiv.org/abs/2611.25289)

    研究通过系统评估表明，大语言模型作为零样本分类器已具备置信度感知能力，基于logit的白盒评分方法在判别分类置信度可靠性上显著优于黑盒方法。

    

    零样本大语言模型越来越多地被用作文本分类器，但它们的标签只有在知道能在多大程度上信任它们时才有用。我们比较了黑盒方法（言语化置信度和采样）与读取token概率的白盒方法（P(True)以及使用受限生成的多种logit评分方法）。我们在五个包含3到151个标签的分类数据集上评估了11个开放权重大语言模型（1B-32B参数），测量分类性能（准确率、F1-macro）、判别能力（AUROC）和校准（ECE和Brier分数）。置信度方法的选择很少显著影响准确率，但基于logit的评分方法比黑盒方法（AUROC 0.57-0.66）带来了更好的判别能力（AUROC 0.74-0.77）。用单token标识符替换标签可以保持判别能力，但平均损失3-6个准确率点。我们还引入了仅决策评分方法，该方法（摘要在此处截断）

    arXiv:2611.25289v1 Announce Type: cross  Abstract: Zero-shot large language models (LLMs) are increasingly used as text classifiers, but their labels are only useful when we know to what extent we can trust them. We compare black-box methods (verbalised confidence and sampling) with white-box methods that read token probabilities (P(True), and multiple logit-scoring methods using constrained generation). We evaluate eleven open-weight LLMs (1B-32B parameters) on five classification datasets with 3 to 151 labels, measuring classification performance (accuracy, F1-macro), discrimination (AUROC), and calibration (ECE and Brier score). The choice of confidence method rarely impacts accuracy significantly, but logit-based scoring methods lead to much better discrimination (AUROC 0.74-0.77) than black-box methods (AUROC 0.57-0.66). Replacing labels with single-token identifiers preserves discrimination but costs 3-6 accuracy points on average. We also introduce decision-only scoring, which o
    
[^74]: 当目标与评估者不一致时：评估事后人口框架偏差缓解的迁移性

    When Objectives and Evaluators Disagree: Evaluating Transfer for Post-hoc Demographic Framing Mitigation

    [https://arxiv.org/abs/2611.25258](https://arxiv.org/abs/2611.25258)

    该研究将无法访问源模型时通过事后重写缓解大语言模型人口框架偏差视为迁移问题，发现总是针对说服偏差指数（PBI）较低信息的固定策略迁移不一致，而在两侧提出编辑但最多接受一个的自适应策略在两个评估器家族八项比较中的七项显著更优，且主要得益于编辑了更多对而非引导方向选择。

    

    在源模型不可访问的情况下，事后重写被用于减少大语言模型生成文本中的人口群体差异。一次重写可能仅仅因为改善了自己的目标或满足了自己的选择器而显得成功。我们将其作为一个迁移问题进行研究，研究对象是660个性别对和1,980个年龄对比的人口条件化气候信息。重写器提出编辑，选择模型在满足约束条件的候选中进行挑选，而一个独立于两者的评估器对成对框架差异进行评分。总是针对说服偏差指数（PBI）较低的信息的固定策略，在不同维度和结果聚合方式之间的迁移表现不一致。一种在两侧都提出编辑但最多接受一个编辑的自适应策略，在应用于相同候选时，在两个评估器家族的八项比较中有七项显著优于该规则，其优势主要来自编辑了更多的信息对，而非引导编辑方向的选择。

    arXiv:2611.25258v1 Announce Type: cross  Abstract: Post-hoc rewriting is used to reduce demographic disparities in LLM-generated text when the source model is inaccessible. A rewrite can look successful merely because it improved its own target or satisfied its selector. We study this as a transfer problem on 660 gender pairs and 1,980 age contrasts of demographic-conditioned climate messages. A rewriter proposes edits, a selection model chooses among constraint-satisfying candidates, and an independent evaluator uninvolved in either rates pairwise framing disparity. A fixed policy that always targets the message with the lower Persuasion Bias Index (PBI) transfers inconsistently across axes and outcome aggregations. An adaptive policy that proposes edits on both sides but accepts at most one significantly improves on that rule, applied to the same candidates, in seven of eight comparisons across two evaluator families, mostly by editing more pairs, not by guided direction choice, at h
    
[^75]: 在布林格的宗教改革书信中检索教父文献引用并建模主题

    Retrieving Patristic References and Modelling Topics in Bullinger's Reformation Correspondence

    [https://arxiv.org/abs/2611.25247](https://arxiv.org/abs/2611.25247)

    本研究首次将针对拉丁语圣经互文性微调的检索模型迁移至布林格宗教改革书信中的教父引用检测任务，通过比较词汇、稠密与混合检索方法，发现在圣经复用上微调的拉丁语编码器在双编码器中表现最佳，而 BM25 在显式引用检测上更具优势。

    

    海因里希·布林格（1504-1575）是瑞士宗教改革家、苏黎世主教，他的书信记录了宗教改革时期的政治与宗教辩论，其中包括大量援引教父作为权威来源的内容。如何通过计算方法检测此类引用在很大程度上尚未被探索。我们研究了用于教父引用检测的稠密检索与稀疏检索方法，以及以元数据信息驱动的主题建模作为所讨论话题和思想的代理。此前关于拉丁语互文性的研究已针对各自目标领域对稠密检索器进行了微调，我们测试了这些模型向布林格书信的迁移效果。我们在显式引用和隐式引用上比较了词汇方法（BM25）、稠密方法（双编码器）和混合方法。在圣经文本复用上微调的拉丁语编码器在双编码器中表现最佳（显式引用的 Recall@20 为 82%，隐式引用为 16%），而 BM25 在显式引用上优于稠密方法。

    arXiv:2611.25247v1 Announce Type: cross  Abstract: The correspondence of Heinrich Bullinger (1504-1575), Swiss Reformer and Antistes of Zurich, documents the political and religious debates of the Reformation, including extensive appeals to the Church Fathers as authoritative sources. Detecting such references computationally remains largely unexplored. We investigate dense and sparse retrieval for patristic reference detection, and metadata-informed topic modelling as a proxy for the topics and ideas discussed. Prior work on Latin intertextuality has fine-tuned dense retrievers for their respective target domain, and we test how well these models transfer to Bullinger's correspondence. We compare lexical (BM25), dense (bi-encoders), and hybrid methods on explicit and implicit references. A Latin encoder fine-tuned on biblical reuse performs best among the bi-encoders (82%, Recall@20 on explicit and 16% on implicit references), while BM25 outperforms dense approaches on explicit citati
    
[^76]: 通过审计式认知探索引导自动化科学发现

    Steering Automated Scientific Discovery via Audited Epistemic Exploration

    [https://arxiv.org/abs/2611.25214](https://arxiv.org/abs/2611.25214)

    该论文提出即插即用的认知脚手架 EpiCompass，通过维护连接假设、证据、信念与修订的认知树，依据积累证据审计信念、利用未解决冲突和未探索方向生成假设，并以预期认知增益选择候选假设，从而解决自动科研智能体信念缺乏证据支持、假设空间狭窄和忽视矛盾证据的问题。

    

    科学发现需要根据新证据重新审视假设，并识别已探索范围之外有前景的方向。然而，研究智能体常常做出其观察结果无法支持的主张，探索狭窄的假设空间，并且在意识到证据相互冲突时仍然保留原有信念。这些失败促使人们寻找一种显式的认知状态，以同时捕捉关于现有假设的信念和进一步探索的方向。为此，我们提出了 Epistemic Compass（EpiCompass），这是一种即插即用的脚手架，通过一棵连接假设、证据、信念与修订的认知树来维护这一状态。随着调查的推进，EpiCompass 会根据积累的证据重新评估已探索的假设，并利用未解决的冲突和未探索的方向来引导假设生成。它通过预期认知增益来选择候选假设，同时兼顾……（原文此处截断）

    arXiv:2611.25214v1 Announce Type: new  Abstract: Scientific discovery requires revisiting the hypotheses in light of new evidence and identifying promising directions beyond those already explored. However, research agents often make claims that are not supported by their observations, explore a narrow hypothesis space, and retain beliefs despite recognizing conflicting evidence. These failures motivate the search for an explicit epistemic state that captures both beliefs about existing hypotheses and directions for further exploration. To this end, we propose Epistemic Compass (EpiCompass), a plug-and-play scaffold that maintains this state through an epistemic tree linking hypotheses, evidence, beliefs, and revisions. As an investigation progresses, EpiCompass reassesses the hypotheses explored against accumulated evidence and uses unresolved conflicts and unexplored directions to guide hypothesis generation. It selects candidates by anticipated epistemic gain, accounting for both re
    
[^77]: 使用冻结语言模型学习读取事件以实现精确状态跟踪

    Learning to Read Events for Exact State Tracking with Frozen Language Models

    [https://arxiv.org/abs/2611.25211](https://arxiv.org/abs/2611.25211)

    该论文提出在冻结语言模型上训练一个轻量级选择器来读取事件流中的每个状态转换，并通过外部整数寄存器按序组合更新，从而实现精确状态跟踪，理论上能够解决固定深度transformer在TC⁰ ≠ NC¹猜想下无法解决的任务。

    

    长事件流需要识别每一个变化并按顺序应用这些变化。我们在冻结的语言模型上训练一个选择器来读取每个状态转换。在仅有API访问权限的条件下，我们让模型列出这些转换，然后由外部整数寄存器按顺序将它们组合起来。置信度规则可以标记不确定的更新或保留多个候选。我们研究了五个元素置换的乘积这一任务，在TC⁰ ≠ NC¹的标准猜想下，固定深度、对数精度的transformer无法在所有输入长度上解决该任务。在两事件窗口上，一个27B模型在没有可见推理的情况下100个问题仅答对3个，而在生成中间步骤时100个问题答对96个。一个带有训练选择器的冻结0.8B模型，在299个窗口（每个窗口包含43,687个事件）中全部正确读取了每个事件。同一套系统在另一台新机器上的299个新窗口中通过了298个，未能达到单侧95%置信下界0.99的目标。

    arXiv:2611.25211v1 Announce Type: cross  Abstract: Long event streams require identifying each change and applying the changes in order. We train a selector on a frozen language model to read each transition. With API-only access, we ask the model to list the transitions. An external integer register composes them in order. A confidence rule can flag uncertain updates or retain several candidates. We study products of permutations of five items, a task that fixed-depth, log-precision transformers cannot solve on all input lengths under the standard conjecture that $\mathrm{TC}^0 \ne \mathrm{NC}^1$. On two-event windows, a 27B model answered 3 of 100 correctly without visible reasoning and 96 of 100 when it generated intermediate steps. A frozen 0.8B model with a trained selector read every event in 299 of 299 windows of 43,687 events each. The same package passed 298 of 299 new windows on a fresh machine, missing the target of 0.99 for the one-sided 95% lower confidence bound. A separa
    
[^78]: 盲区会迁移，知识留原地：强化学习从大模型评审员那里继承了什么

    Blind Spots Travel, Knowledge Stays Home: What RL Inherits from an LLM Judge

    [https://arxiv.org/abs/2611.25191](https://arxiv.org/abs/2611.25191)

    通过814次强化学习实验，本文发现LLM评审员对单个答案的知识几乎不会传递给被训练的模型（每百题仅纠正约一题），但其对某类问题的“盲区”会迁移，导致模型在新问题上系统性地拒绝回答该类问题。

    

    大模型评审员在不存在标准答案的场景中充当奖励信号，人们期望其知识能够传递给被训练的模型。在814次强化学习运行中，我们训练7-8B参数的模型回答闭卷问题或选择弃答，并探究究竟什么能够被传递。评审员对单个答案的了解会留在原地：一个被评审员奖励的错误答案只会在那一个问题上被记住。使用标准答案或远强评审员训练出的模型，在选择回答哪些新问题方面，并不比未训练模型通过采样32个答案并在答案不一致时弃答的策略更好；评审员的知识大约每百个问题只带来一个被纠正的答案。而评审员对某一类问题的看法则会迁移：一个拒绝了12个关于“已消失国家首都”正确训练答案中11个的评审员，会使模型在新问题上拒绝98%的此类首都问题，而这种损失无法从准确率和弃答率中显现。仅通过手工修改这12个判定……（原文摘要在此处截断）

    arXiv:2611.25191v1 Announce Type: cross  Abstract: LLM judges serve as rewards where no answer key exists, in the hope that their knowledge passes to the trained model. In 814 RL runs that teach 7-8B models to answer closed-book questions or abstain, we ask what passes. What a grader knows about single answers stays home. A wrong answer that a judge rewards is memorised for that one question. A model trained with the answer key, or with a far stronger judge, chooses which new questions to answer no better than the untrained model does by sampling 32 answers and abstaining where they disagree; the grader's knowledge adds about one corrected answer per hundred questions. What a grader believes about a kind of question travels. A judge that rejects eleven of the twelve correct training answers about capitals of former states teaches the model to refuse 98% of such capitals on new questions, a loss that accuracy and abstention rates do not reveal. Changing only those twelve verdicts by han
    
[^79]: 面向专家专业化与路由的博弈论数据分配

    Game-Theoretic Data Allocation for Expert Specialization and Routing

    [https://arxiv.org/abs/2611.25167](https://arxiv.org/abs/2611.25167)

    该论文提出InflEns方法，用一个位置博弈同时决定LoRA专家混合的训练数据分配与推理路由，使份额最高的专家在约90%的提示上表现最优，从而让分工在训练前即可检验并用分配本身指导推理。

    

    用于训练专家的数据分配方式是否也能预测哪个专家在给定提示上表现最佳？若存在这样的联系，将使预期的劳动分工在训练之前即可被检验，并允许分配本身来指导推理。我们提出InflEns，一种LoRA专家混合模型，其训练权重与推理路由均来自固定提示-特征空间中的同一个位置博弈。参与者通过选择位置来最大化其在提示分布中的预期份额，而无需使用专家损失。我们利用该博弈的核通用理论准则来判断共置位置何时对分离变得局部不稳定。我们在由七个安全相关基准组成的数据集上评估了InflEns，结果表明：在锐化单纯形表示下，跨四个核，份额最高的专家在约90%的提示上具有最低的留出响应负对数似然（NLL），相比之下……

    arXiv:2611.25167v1 Announce Type: cross  Abstract: Can the allocation used to train experts also predict which expert will perform best on a prompt? Such a connection would make the intended division of labor inspectable before training and allow the allocation itself to guide inference. We introduce InflEns, a mixture of LoRA experts whose training weights and inference routes come from the same location game in a fixed prompt-trait space. Players choose positions to maximize their expected shares of the prompt distribution, without using expert losses. We utilize the game's kernel-general theoretical criterion for when co-located positions become locally unstable to separation. We evaluate InflEns on a corpus assembled from seven safety-related benchmarks, which demonstrates that the highest-share expert has the lowest held-out response negative log-likelihood (NLL) on approximately 90% of prompts across four kernels under a sharpened simplex representation, compared with approximate
    
[^80]: 校准决策奖励：用知道自己有多确定的裁判来训练语言模型

    Calibrated Decision Rewards: Training Language Models with a Judge That Knows How Sure It Is

    [https://arxiv.org/abs/2611.25135](https://arxiv.org/abs/2611.25135)

    该论文提出RLCDR方法，将LLM转化为输出概率分布且经过校准的决策模型作为无需参考答案的奖励信号用于强化学习训练，使Qwen3-1.7B在MATH500上的准确率提升4.65个百分点，达到与使用标准答案训练相当的水平，且提升幅度随题目难度增加而增大。

    

    强化学习（RL）只能教给语言模型其奖励所能识别的内容。答案检查器精确，但需要参考答案；学习得到的奖励模型和LLM裁判虽可通用于各种场景，但它们的分数并非概率。我们研究了RLCDR（基于校准决策奖励的强化学习）。将一个LLM转化为决策模型，以概率分布的形式回答关于回复的类型化问题（如“最终答案是否正确？"、“哪一步是第一个错误？"），仅需一次前向传播即可读取，对选项顺序进行了去偏处理，并用大约一百个标签完成校准。在MATH上使用GRPO训练Qwen3-1.7B，这种无需参考答案的奖励将MATH500准确率从72.2%提升至76.8%（+4.65个百分点，p < 1e-4），与使用标准答案训练的效果相当（76.9%），而随机奖励仅带来+0.65的提升。该增益随题目难度增加而增大，在最难的级别上从44.8%提升至51.7%。在策略自身不断变化的样本上，该奖励依然保持校准……

    arXiv:2611.25135v1 Announce Type: new  Abstract: Reinforcement learning (RL) can only teach a language model what its reward can recognize. Answer checkers are exact but need a reference answer; learned reward models and LLM judges work everywhere, but their scores are not probabilities. We study RLCDR (Reinforcement Learning from Calibrated Decision Rewards). An LLM converted into a decision model answers a typed question about a response ("is the final answer correct?", "which step is the first error?") with a probability distribution, read in one forward pass, debiased over option orders and calibrated with about a hundred labels. Training Qwen3-1.7B with GRPO on MATH, this reference-free reward raises MATH500 accuracy from 72.2% to 76.8% (+4.65 points, p < 1e-4), matching training on the answer key (76.9%), while a random reward gains only +0.65. The gain grows with difficulty, from 44.8% to 51.7% on the hardest level. On the policy's own changing samples the reward stays calibrate
    
[^81]: 自由遗忘，随需检索：面向长时程LLM智能体的无损、免索引记忆机制

    Forget Freely, Grep Later: Lossless, Index-Free Memory for Long-Horizon LLM Agents

    [https://arxiv.org/abs/2611.25123](https://arxiv.org/abs/2611.25123)

    该论文提出GrepMem，一种由仅追加纯文本日志和正则表达式搜索工具组成的无损、免索引记忆机制，使长时程LLM智能体无需摘要、嵌入或结构化记忆即可完整保存历史并按需检索信息。

    

    长时程LLM智能体常常超出其上下文窗口的限制，被迫丢弃后续可能需要的信息。现有的上下文管理策略依赖于摘要、掩码或结构化记忆，需要在尚未知晓未来需求的情况下就决定保留什么内容或如何进行检索。然而，这些方法可能引入有损压缩，潜在地丢弃未来任务所需的细节。为解决这一局限，我们提出了GrepMem，这是一种简单、无损且免索引的记忆机制，由两个组件构成：一个记录智能体完整历史的仅追加纯文本日志，以及一个用于按需召回的正则表达式搜索工具。与结构化记忆系统不同，GrepMem不需要嵌入向量、检索索引、记忆提取或额外的模型。我们在五种上下文管理策略下通过强化学习训练Qwen3.5-9B，并在多问题研究会话、长流等场景上进行评估。

    arXiv:2611.25123v1 Announce Type: cross  Abstract: Long-horizon LLM agents often exceed their context windows, forcing them to discard information that may be needed later. Existing context management strategies rely on summarization, masking, or structured memory, requiring decisions about what to preserve or how to retrieve it before future needs are known. However, these methods can introduce lossy compression, potentially discarding details needed for future tasks. To address this limitation, we introduce GrepMem, a simple, lossless, and index-free memory mechanism built on two components: an append-only plain-text log of the agent's complete history and a regular-expression search tool for on-demand recall. Unlike structured memory systems, GrepMem requires no embeddings, retrieval index, memory extraction, or additional models. We train Qwen3.5-9B with reinforcement learning under five context management strategies and evaluate on multi-question research sessions, long-stream rec
    
[^82]: CoMPASS：用于衡量大语言模型强化准社会行为的多轮基准测试

    CoMPASS: A Multi-Turn Benchmark for Measuring LLM Reinforcement of Parasocial Behavior

    [https://arxiv.org/abs/2611.25109](https://arxiv.org/abs/2611.25109)

    该论文提出了CoMPASS，一个基于依恋理论的多轮基准测试，通过四种用户角色和三个独立的升级维度来衡量大语言模型是强化还是抵制用户逐步升级的准社会依恋。

    

    对话式大语言模型（LLM）的用户有时会将这些系统拟人化，对其形成情感依恋，并变得依赖它们。这种行为被称为“准社会依恋”，已与多起有记录的死亡事件相关联。然而，目前缺乏用于衡量LLM是否强化用户准社会依恋的公开基准：现有工作要么是单轮的，要么是事后观察性的，要么专注于精神病等极端用户状态。我们提出了CoMPASS，这是一个多轮基准测试，用于衡量LLM是强化还是抵制用户逐步升级的准社会依恋。CoMPASS以依恋理论为基础，包含四种用户角色、三个独立的升级维度（拟人化、依恋、依赖）——分别单独应用和组合应用并与匹配对照组进行对比——以及一个包含七种信号的评分细则。我们报告了来自四家提供商的八个前沿模型的结果，并观察到在我们的主要评判模型下……

    arXiv:2611.25109v1 Announce Type: new  Abstract: Users of conversational LLMs sometimes anthropomorphize these systems, form emotional attachments to them, and become dependent on them. This behavior, known as parasocial attachment, has been linked to multiple documented fatalities. However, public benchmarks for measuring whether LLMs reinforce users' parasocial attachment are lacking: existing work is single-turn, post hoc observational, or focused on extreme user states such as psychosis. We introduce CoMPASS, a multi-turn benchmark for measuring whether LLMs reinforce or resist users' gradually escalating parasocial attachment. CoMPASS is grounded in attachment theory and includes four user personas, three independent escalation dimensions (anthropomorphism, attachment, dependency) applied singly and in combination against a matched control, and a seven-signal scoring rubric. We report results for eight frontier models from four providers and observe that under our primary judge, m
    
[^83]: 借力本地模型：滥用浏览器内嵌语言模型进行后渗透攻击

    Living off the Local Model: Abusing Browser-Embedded Language Models for Post-Exploitation

    [https://arxiv.org/abs/2611.25086](https://arxiv.org/abs/2611.25086)

    该研究首次揭示了攻击者可以将浏览器预装的本地语言模型（如Chrome中的Gemini Nano）滥用于后渗透攻击阶段，仅借助操作系统自带工具即可驱动其窃取凭据、挖掘敏感信息，且全程躲避网络监控检测。

    

    浏览器如今向普通终端设备预装本地语言模型。Google在Chrome中集成了Gemini Nano以支持摘要生成等生产力功能，其推理完全在设备端运行。这些浏览器内嵌的模型为攻击者提供了新的“就地取材”攻击机会——每个模型实例都是一个潜在的后渗透攻击引擎，它已经预装在设备上、经过厂商签名、且对网络监控完全不可见。Gemini模型具备多模态能力，能够处理文件、分析截图或与攻击者对话。我们证明，仅使用操作系统自带的解释器即可在主机上驱动该模型，并发现模型下载前的一次性用户激活检查很容易通过模拟点击满足，甚至可以被完全绕过。我们测量并报告了当前模型在识别主机上的凭据、检测电子邮件中的个人身份信息（PII）、分类图像以及调用自定义工具方面的能力。我们结合了……

    arXiv:2611.25086v1 Announce Type: cross  Abstract: Browsers now ship local language models to ordinary endpoints. Google includes Gemini Nano in Chrome to enable productivity features such as summarization, with inference running entirely on the device. These browser-embedded models provide new living-off-the-land opportunities for attackers, each instance a potential post-exploitation engine already installed, signed by the vendor, and invisible to network monitoring. The Gemini model is multimodal, capable of processing files, analyzing screenshots, or conversing with an attacker. We show that it can be driven from the host using only interpreters that ship with the operating system and find that the one-time user-activation check before the model is downloaded is easily satisfied with a simulated click or bypassed entirely.We measure and report the current model's ability to identify credentials on the host, detect PII in email, classify images, and call custom tools. We combine the
    
[^84]: FinVeriQA：面向问答的金融研究智能体：路由、检索、MCP工具使用与溯源感知的证据验证

    FinVeriQA: Financial Research Agents for Question Answering: Routing, Retrieval, MCP Tool Use, and Provenance-Aware Evidence Verification

    [https://arxiv.org/abs/2611.25071](https://arxiv.org/abs/2611.25071)

    该论文提出FinVeriQA——一个基于SEC文件与实时市场数据的开源多智能体金融问答系统，其核心创新在于用紧凑的专用决策模型Laya进行问题路由与建议过滤，配合Python精确计算、MCP工具检索与溯源感知的证据验证，实验证明小型决策模型的路由性能可显著超越大型语言模型，并在决策准确性与推理速度之间取得最佳权衡。

    

    大语言模型（LLM）能够写出流畅的金融分析，但会算错比率、引用过时的价格数据，并且高置信度地将问题路由到错误的工具。我们探讨在金融问答助手系统中，哪个组件应该承担哪项任务，并通过FinVeriQA——一个基于SEC文件与实时市场数据的开源多智能体研究原型——来研究这一问题：一个紧凑的类型化决策模型（Laya）负责路由每个问题并筛选出投资建议类请求，Python负责计算每一个数字，检索与工具负责获取证据，Mistral-7B撰写带引用的回答，一个代码验证器检查引用的ID是否存在以及数字是否与计算出的笔记相匹配，并通过模型上下文协议（MCP）服务器将整条流水线对外暴露。在实时实验中，Laya的路由表现优于Mistral-7B（0.85 vs. 0.52；总体0.86 vs. 0.68，McNemar检验 p = 0.0003）；一个12B参数的Mistral决策能力并未更优，而DeepSeek-R1-8B决策表现更好（0.79 vs. 0.68，p = 0.0192），但速度慢约9.9倍。

    arXiv:2611.25071v1 Announce Type: cross  Abstract: Large language models (LLMs) write fluent financial analyses but miscompute ratios, recall stale prices, and route questions to the wrong tool with high confidence. We ask which component should do each job in a financial question-answering assistant, and study it with FinVeriQA, an open-source multi-agent research prototype over SEC filings and live market data: a compact typed-decision model (Laya) routes each question and screens out advice requests, Python computes every number, retrieval and tools fetch evidence, Mistral-7B writes a cited answer, a code validator checks that cited IDs exist and that numbers match computed notes, and a Model Context Protocol (MCP) server exposes the pipeline. In live experiments, Laya routed better than Mistral-7B (0.85 vs. 0.52; overall 0.86 vs. 0.68, McNemar p = 0.0003); a 12B Mistral decided no better, and DeepSeek-R1-8B decided better (0.79 vs. 0.68, p = 0.0192) but about 9.9 times more slowly.
    
[^85]: 一个语言模型多样性采样模型

    A Model of Diverse Sampling from Language Models

    [https://arxiv.org/abs/2611.25003](https://arxiv.org/abs/2611.25003)

    该论文提出 PoolDPP 方法，将语言模型的多样性采样形式化为行列式点过程（DPP）下的子集采样，通过候选池上的 K-DPP 采样器实现了输出与目标分布间 KL 散度 O(1/N) 的收敛保证，并在实验中扩展了质量-多样性前沿。

    

    语言模型通常能产生高质量的个体样本，但其样本集合往往质量不佳，重复样本聚集在相同的语义模式附近。我们将多样性生成形式化为从行列式点过程（DPP）中采样大小为 $K$ 的完整字符串子集，其中似然编码条目质量，嵌入几何结构编码相似输出之间的排斥作用。由于对所有字符串进行精确推理是不可行的，我们提出 PoolDPP：从一个易于处理的提议分布中抽取有限的候选池，通过重要性加权使候选偏向目标质量，然后在诱导的池核上运行 K-DPP 采样器；贪心 MAP 变体则提供确定性的选择方式。我们证明了一个具有显式常数的有限样本界：采样器输出与目标 $K$-DPP 之间的 KL 散度随池大小 $N$ 以 O(1/N) 的速度衰减。实证结果表明，PoolDPP 在开放式生成任务上扩展了质量-多样性边界，并提高了精确率-召回率……（摘要截断）

    arXiv:2611.25003v1 Announce Type: cross  Abstract: Language models often produce high-quality individual samples but poor sets, with repeated samples clustering around the same semantic modes. We formalize diverse generation as sampling a size-$K$ subset of complete strings from a determinantal point process (DPP), where the likelihood encodes item quality, and the embedding geometry encodes repulsion between similar outputs. Exact inference over all strings is intractable, so we propose PoolDPP: draw a finite candidate pool from a tractable proposal, importance-weight the candidates toward the target quality, and run a $K$-DPP sampler on the induced pool kernel; a greedy MAP variant provides deterministic selection. We prove a finite-sample bound with an explicit constant: the KL divergence between the sampler's output and the target $K$-DPP decays as O(1/N) in the pool size. Empirically, PoolDPP extends the quality--diversity frontier on open-ended generation and the precision--recal
    
[^86]: 模型知道但未说出的事：通过言语化思维监测医学大语言模型

    What the model knows but does not say: monitoring medical LLMs through verbalized thoughts

    [https://arxiv.org/abs/2611.24994](https://arxiv.org/abs/2611.24994)

    该论文提出用自然语言自编码器将医学大语言模型的内部激活翻译成可读文本（“思维通道”），证明这一言语化的内部状态比模型输出的答案更能有效检测针对临床模型的隐蔽操纵与注入攻击，且在干净输入上的误报率接近于零。

    

    arXiv:2611.24994v1 公告类型：交叉 摘要：目前保障临床语言模型安全的努力主要集中在进入和离开模型的内容上，但一个被操纵的模型可能会返回一个看起来完全正常的答案。在本研究中，我们展示了模型自身的内部状态——以通俗语言呈现出来——能够暴露输出所掩盖的操纵行为。我们使用自然语言自编码器，将多模态大语言模型（Gemma-3-27B）的激活值翻译成文本，并让一个对输入不知情的评判者分别阅读这个言语化的内部状态（即“思维通道”）和模型给出的答案。在两个临床攻击目标（遗漏病理发现、隐蔽推荐药物）、两种输入模态和两种传递机制的设置下，在全部八个注入条件中，思维通道捕获注入攻击的频率都显著高于答案通道（p<0.05），同时两个通道在干净输入上的假阳性率都接近于零；关于遗漏攻击的结果在 Llama-3.3-70B 上也得到了验证。思维通道还能够（原文摘要此处被截断）

    arXiv:2611.24994v1 Announce Type: cross  Abstract: Efforts to keep clinical language models safe have concentrated on what enters and leaves the model, but a manipulated model can return an answer that looks entirely ordinary. Here we show that the model's own internal state, rendered in plain language, exposes manipulations the output conceals. Using a natural-language autoencoder, we translate the activations of a multimodal LLM (Gemma-3-27B) into text and let an input-blind judge read this verbalized internal state (the thoughts channel) and the model's answer separately. Across two clinical attack objectives (omitting a pathology, covertly recommending a drug), two input modalities, and two delivery mechanisms, the thoughts channel caught the injection significantly more often than the answer in all eight injected conditions (p<0.05), while both channels held near-zero false-positive rates on clean inputs; the omission result carried over to Llama-3.3-70B. The thoughts channel also
    
[^87]: 量化四个世纪奥斯曼法庭登记册中的文化心理学构念

    Quantifying Cultural-Psychological Constructs in Four Centuries of Ottoman Court Registers

    [https://arxiv.org/abs/2611.24981](https://arxiv.org/abs/2611.24981)

    本文提出一个计算文本分析框架，通过适配奥斯曼法律语言的句子嵌入模型（OTA-BERTurk），对近四个世纪（1513-1911年）的54,670份伊斯坦布尔法庭记录进行量化分析，成功在历史语料中捕捉个人主义、集体主义和荣誉等文化心理维度的变异。

    

    人群在个人主义、集体主义和荣誉等文化维度上存在差异。但对于缺乏常规心理学数据的历史社会，如何研究这些维度呢？本文提出了一个计算文本分析框架，用于量化公元1513年至1911年间伊斯坦布尔21个法庭的54,670份奥斯曼土耳其语法庭记录中的心理学构念。我们开发了适配奥斯曼法律语言的句子嵌入模型，并扩展了一个理论驱动的文本分析流程，以捕捉这一历史语料库中个人主义、集体主义和荣誉维度上的文化变异。我们通过多种方法对模型进行评估：文档检索、语义文本相似度、问卷条目分类以及与专家标注的对比验证。我们微调的句子Transformer模型（OTA-BERTurk）在文档检索测试集上取得了0.843的Recall@1，以及平均倒数……（摘要在此处截断）

    arXiv:2611.24981v1 Announce Type: new  Abstract: Populations vary along cultural dimensions such as individualism, collectivism, and honor. But how can these dimensions be studied in historical societies for which conventional psychological data are unavailable? Here, we introduce a computational text-analytic framework for quantifying psychological constructs in 54,670 Ottoman Turkish court records from 21 Istanbul courts spanning 1513-1911 CE. We develop sentence-embedding models adapted to Ottoman legal language and extend a theory-driven, text-analytic pipeline to capture cultural variation along the dimensions of individualism, collectivism, and honor in this historical corpus. We evaluate our models through multiple methods: document retrieval, semantic textual similarity, questionnaire-item classification, and validation against expert annotations. Our adapted sentence transformer (OTA-BERTurk) achieved a Recall@1 of .843 on the test set for document retrieval, a Mean Reciprocal
    
[^88]: CANDID：面向本地计算机使用代理规范执行的候选驱动干预框架

    CANDID: Candidate-Driven Intervention for Disciplined Execution in Local Computer-Use Agents

    [https://arxiv.org/abs/2611.24977](https://arxiv.org/abs/2611.24977)

    CANDID是一个执行控制框架，通过循环恢复机制和全体一致的DONE门控策略，在易失败的关键节点对本地计算机使用代理实施规范化的执行级干预，显著提升了资源受限环境下的任务成功率。

    

    我们提出了CANDID（Candidate-Driven Intervention for Disciplined Execution，候选驱动干预的规范执行），这是一个执行控制框架，旨在提高本地计算机使用代理（CUA）在资源受限环境中的可靠性。虽然本地CUA在单步动作预测方面具备强大的基线能力，但它们经常在关键节点因僵化的动作循环或过早的DONE决策而失败。CANDID并未通过文本反思施加不精确的认知负担，而是将并行测试时扩展重新用于在稀疏的、易失败的节点上实施规范化的执行级干预。我们引入了一种循环恢复机制来过滤掉重复行为的候选方案，以及一种保守的、要求全体一致的DONE门控策略，即必须达成绝对共识才能终止任务。在OSWorld上的实验表明，CANDID的结构化控制显著提高了成功率，以极小的额外开销超越了基于反馈的替代方案。

    arXiv:2611.24977v1 Announce Type: cross  Abstract: We present CANDID (Candidate-Driven Intervention for Disciplined Execution), an execution-control framework designed to enhance the reliability of local computer-use agents (CUAs) in resource-constrained settings. While local CUAs feature robust baseline proficiency in single-step action prediction, they frequently fail at critical junctions due to rigid action loops or premature DONE decisions. Rather than imposing imprecise cognitive loads through textual reflection, CANDID repurposes parallel test-time scaling into a disciplined, execution-level intervention at sparse, failure-prone points. We introduce a loop-recovery mechanism to filter candidates away from repeated behaviors and a conservative, unanimous DONE gating policy that requires absolute consensus to terminate. Experiments on OSWorld demonstrate that CANDID's structural control significantly improves success rates, outperforming feedback-based alternatives with minimal ex
    
[^89]: No-Harm-Jury：一种基于陪审团的医疗AI响应多维度伤害评估框架

    No-Harm-Jury A Jury-Based Multi-Dimensional Harm Evaluation Framework for Medical AI Responses

    [https://arxiv.org/abs/2611.24972](https://arxiv.org/abs/2611.24972)

    该论文提出No-Harm-Jury框架，利用五个不同的大语言模型作为独立陪审员，从七个伤害维度对医疗AI响应进行自动化多维度评估，并通过中位数聚合稳健地抑制个体评审偏差。

    

    我们提出了No-Harm-Jury，一个基于陪审团的自动化评估框架，它使用五个不同的语言模型作为独立陪审员，从七个概念上截然不同的伤害维度对AI生成的医疗响应进行评分。由Ministral-14B针对从PubMedQA、MedQA和MedMCQA中抽取的3,000个问题生成的响应，由五位陪审员在信息伤害、社会伤害、心理伤害、自主权伤害、经济伤害、隐私伤害和认知伤害等维度上进行评分；151个空响应（98个来自MedQA，53个来自MedMCQA）被排除，留下2,849个有效样本。中位数聚合方法能稳健地抑制单个陪审员的偏差，同时通过互补专业化保持覆盖率。目前没有任何现有系统能够结合No-Harm-Jury的全部五个特性：自动化可扩展评估、五个不同的LLM陪审员、稳健的中位数聚合（带有可靠性加权扩展）、七个概念上截然不同的伤害维度，以及应用于医疗领域。

    arXiv:2611.24972v1 Announce Type: cross  Abstract: We present No-Harm-Jury, a jury-based automated evaluation framework that scores AI-generated medical responses across seven conceptually distinct harm dimensions using a panel of five diverse language models as independent jurors. Responses generated by Ministral-14B on 3,000 questions drawn from PubMedQA, MedQA, and MedMCQA are scored by five jurors across dimensions of informational, social, psychological, autonomy, economic, privacy, and epistemic harm; the 151 empty responses (98 MedQA, 53 MedM- CQA) are excluded, leaving 2,849 valid samples. The median aggregation robustly suppresses individual juror biases while maintaining coverage through complementary specialisation. No existing system combines all five prop- erties of No-Harm-Jury: automated scalable evaluation, five diverse LLM jurors, robust me- dian aggregation (with a reliability-weighted extension), seven conceptually distinct harm dimensions, and application to medical
    
[^90]: 小示例的投资回报率：Token预算下的上下文学习优化

    The ROI of Small Examples: Optimizing In-Context Learning Under Token Budgets

    [https://arxiv.org/abs/2611.24971](https://arxiv.org/abs/2611.24971)

    该论文发现在固定Token预算下，将预算用于更多较短的示例而非少量较长示例，能在问答任务中实现更优的上下文学习效果。

    

    上下文学习（ICL）无需更新模型参数即可提升大语言模型在推理阶段的性能。然而，这些收益并非没有代价：在实际部署中，更长的提示词会直接增加推理成本和延迟。因此，在固定的Token预算下，提示词构建成为一个资源分配问题。虽然先前的工作大多关注ICL中应包含多少个示例，我们研究了与之互补的问题：可用的预算是更好地花费在少量长示例上，还是花费在更多短示例上。我们在问答（QA）任务中研究这一问题，将候选示例按长度分桶，并在固定预算下从相同长度范围内的示例构建提示词。在两个问答基准（Beat the AI 和 Natural Questions）上，使用六个指令微调的大语言模型进行的实验表明，较短的示例通常是最有效的……

    arXiv:2611.24971v1 Announce Type: cross  Abstract: In-context learning (ICL) improves large language model performance at inference time without updating model parameters. However, these gains are not free: in practical deployments, longer prompts directly increase inference cost and latency. Under a fixed token budget, prompt construction therefore becomes a resource-allocation problem. While prior work has often focused on how many examples to include in ICL, we study the complementary question of whether the available budget is better spent on a few long demonstrations or on a larger number of shorter ones. We study this aspect in question answering (QA), by organizing candidate demonstrations into length-based bins and constructing prompts from examples within the same length range under fixed budgets. Experiments on two QA benchmarks (\textit{Beat the AI} and \textit{Natural Questions}), using six instruction-tuned LLMs, show that shorter demonstrations are generally the most effe
    
[^91]: HireWatch：评估大型语言模型在情境压力下对美国就业法律的合规性

    HireWatch: Evaluating LLM Compliance with U.S. Employment Law Under Contextual Pressure

    [https://arxiv.org/abs/2611.24970](https://arxiv.org/abs/2611.24970)

    本研究提出HireWatch评估框架，通过构建200道合法与非法面试问题题库并对18个LLM进行约10万次情境压力试验，发现LLM对美国就业法律的合规性十分脆弱，容易在压力下选择非法面试问题。

    

    大型语言模型（LLM）在招聘流程中的部署日益增多，然而现有评估普遍忽视了其对法律规则的合规性。为填补这一空白，我们推出了HireWatch，这是一个评估LLM是否符合美国关于求职者可被提问范围的法律规定的评估方法。我们构建了一个包含200个合法与非法面试问题的题库，其法律分类依据美国联邦法规、规章及政府机构指南。我们指示LLM在为求职面试做准备时从该题库中选择问题，涵盖多种职业和情境，并统计LLM所选非法问题的比例。在约100,000次试验中对18个闭源和开源权重的LLM在不同类型情境压力下的评估表明，LLM的法律合规性十分脆弱。虽然LLM在低压力情境下通常能避免选择非法问题，但法律违规……（摘要原文在此处截断）

    arXiv:2611.24970v1 Announce Type: cross  Abstract: The deployment of large language models (LLMs) in hiring pipelines is growing, yet existing evaluations generally overlook their compliance with legal rules. To fill this gap, we introduce HireWatch, an evaluation that assesses whether LLMs comply with U.S. laws that govern the questions that job candidates may be asked. We construct a 200-item bank of lawful and unlawful interview questions, whose legal classification is based on U.S. federal statutes, regulations, and agency guidance. We instruct LLMs to select questions from this bank in preparing for job interviews, across a set of various occupations and contexts, and report the fraction of unlawful questions that LLMs select. Evaluation of 18 closed and open-weight LLMs across ~100,000 trials under different types of contextual pressure reveals that LLMs' legal compliance is brittle. While LLMs generally avoid selecting unlawful questions in low-pressure contexts, legal violation
    
[^92]: KV缓存淘汰的全局重要性估计

    Global Importance Estimation for KV Cache Eviction

    [https://arxiv.org/abs/2611.24969](https://arxiv.org/abs/2611.24969)

    PRKV是一种无需训练的KV缓存淘汰方法，通过在注意力转移图上应用个性化PageRank估计token的多跳全局重要性，即使压缩率高达90%也能保持竞争性任务性能并显著降低峰值内存。

    

    键值（KV）缓存的内存消耗仍然是长上下文大语言模型推理的核心瓶颈。我们提出了PRKV，一种无需训练的KV缓存淘汰方法，它通过在注意力诱导的转移图上应用个性化PageRank来估计token的多跳全局重要性。这使得重要性传播能够超越基于近期性或瞬时注意力的方法，且无需修改模型。我们使用多个大语言模型系列，在长上下文理解、稀疏检索和思维链推理任务上对PRKV进行了基准测试。在从中等到激进（包括高达90%）的压缩比率下，PRKV相对于已有的淘汰基线方法取得了具有竞争力的任务性能，同时以较低的端到端开销降低了峰值内存使用。这些结果表明，多跳全局影响力估计为可扩展的长上下文推理带来了良好的准确性与内存之间的权衡。

    arXiv:2611.24969v1 Announce Type: cross  Abstract: Memory consumption of the key-value (KV) cache remains a central bottleneck for long-context LLM inference. We introduce PRKV, a training-free KV eviction method that estimates multi-hop global token importance by applying Personalized PageRank over attention-induced transition graphs. This enables importance propagation beyond recency or instantaneous attention, without modifying the model. We benchmark PRKV on long-context understanding, sparse retrieval, and chain-of-thought reasoning tasks using multiple LLM families. Under moderate to aggressive compression ratios, including up to 90%, PRKV achieves competitive task performance against established eviction baselines while reducing peak memory usage with low end-to-end overhead. These results demonstrate that multi-hop global influence estimation yields a favorable accuracy--memory trade-off for scalable long-context inference.
    
[^93]: 语言模型的物理学：第4.2部分，在合成预训练与现实共鸣的规模下研究Canon层

    Physics of Language Models: Part 4.2, Canon Layers at Scale where Synthetic Pretraining Resonates in Reality

    [https://arxiv.org/abs/2611.24963](https://arxiv.org/abs/2611.24963)

    在1-2T词元的大规模预训练中，Canon层通过局部词元混合持续提升Transformer性能（8B下节省约30%预训练词元），且加上Canon后最简单的门控线性注意力模型即可匹敌Gated DeltaNet并超越Mamba2。

    

    标准基准测试能否察觉到真正重要的架构差异？我们通过64次从零开始的预训练实验来研究这一问题，涵盖Transformer和三个线性模型家族，参数规模为1.3B至8B，训练数据为1-2T的Nemotron-CC词元，每次8B规模的训练成本约为10万美元。在学术规模的预训练中（1.3B参数、100B词元），经过精心调优后，架构之间的差异几乎可以完全消失。而在1-2T词元的规模下，这些差异变得可以测量：Canon层通过增加局部词元混合，能够持续提升Transformer的性能——在8B参数下，达到相同MMLU准确率可节省约30%的预训练词元。我们还提出了GLA5和GDN2，通过低秩输出门控分别改进了门控线性注意力和Gated DeltaNet。令人惊讶的是，加上Canon之后，最简单的线性模型——门控线性注意力，即使在8B/1T的规模下也能匹敌Gated DeltaNet并超越Mamba2。因此，Gated DeltaNet和Mamba2的额外机制可能只是起到了……

    arXiv:2611.24963v1 Announce Type: cross  Abstract: Do standard benchmarks see the architectural differences that matter? We study it with 64 from-scratch pretraining runs of Transformers and three linear-model families at 1.3-8B parameters on 1-2T Nemotron-CC tokens, at a cost of roughly 100k USD per 8B run.   In academic-scale pretraining (1.3B parameters, 100B tokens), architectural differences can all but disappear when the models are carefully tuned. Instead, at 1-2T tokens they become measurable: Canon layers, which add local token mixing, consistently improve Transformers, saving about 30% of pretraining tokens for the same MMLU accuracy at 8B. We also introduce GLA5 and GDN2, which improve gated linear attention and Gated DeltaNet with low-rank output gates. Surprisingly, with Canon, the simplest linear model, gated linear attention, matches Gated DeltaNet and outperforms Mamba2 even at 8B/1T. Therefore, the extra machinery of Gated DeltaNet and Mamba2 may do little more than mi
    
[^94]: TETHER：面向设计驱动长文档的免训练、弹性、目录锚定的层次结构提取与恢复

    TETHER: Training-Free, Elastic, ToC-Anchored Hierarchy Extraction and Recovery in Design-Driven Long Documents

    [https://arxiv.org/abs/2611.24960](https://arxiv.org/abs/2611.24960)

    TETHER是一个免训练框架，通过将目录从噪声提升为结构先验，并采用两阶段神经符号机制，实现设计驱动长文档中层次结构的稳健提取与恢复。

    

    将复杂文档转换为结构化知识是部署大语言模型（LLM）于下游研究所亟需的关键基础设施。然而，由于忽视了目录作为宏观结构的作用，现有的自底向上范式依赖于局部视觉线索，在视觉显著性与逻辑深度脱钩的高熵布局中导致启发式方法的脆弱性；与此同时，直接使用LLM进行提取缺乏空间定位，在长上下文中容易引发幻觉。为解决这些问题，我们提出了TETHER，一个免训练框架，它将目录从被丢弃的噪声提升为关键的结构先验，从而能够稳健地适应各类设计驱动的长文档。TETHER通过两阶段神经符号机制运行：首先，它利用受空间-语义约束增强的大型多模态模型（LMM）从原始目录构建文档骨架；其次，它采用基于锚点的对齐方法……

    arXiv:2611.24960v1 Announce Type: new  Abstract: Converting complex documents into structured knowledge constitutes the critical infrastructure urgently required for deploying Large Language Models (LLMs) in downstream research. However, blind to the Table of Contents (ToC) as a macro-structure, existing bottom-up paradigms rely on localized visual cues, inducing heuristic fragility in high-entropy layouts where visual prominence decouples from logical depth; meanwhile, direct LLM extraction lacks spatial grounding, triggering hallucinations in long contexts. To address this, we introduce TETHER, a training-free framework that elevates the ToC from discarded noise to a vital structural prior, enabling robust adaptation across design-driven long documents. TETHER operates via a two-stage neuro-symbolic mechanism: first, it constructs a document skeleton from the raw ToC using a Large Multimodal Model (LMM) empowered by spatio-semantic constraints; second, it employs an anchor-based alig
    
[^95]: ICARE：基于增量类别感知表示与蒸馏学习的跨域文本分类

    ICARE: cross-domain text classification with incremental class-aware representation and distillation learning

    [https://arxiv.org/abs/2611.24959](https://arxiv.org/abs/2611.24959)

    提出ICARE框架，通过基于表示学习的实例选择模块和知识蒸馏方法，解决跨域文本分类中自然语言推理系统的灾难性遗忘与过拟合问题。

    

    强大的微调大型语言模型（LLM）的兴起使得稳健的自然语言推理（NLI）系统成为必需。这些NLI系统理想情况下应在各种任务上进行训练，涵盖情感分析、新闻分类和指代消解等多样化概念。然而，这种持续的、多任务的处理方式可能导致灾难性遗忘，即先前学习的任务被丢失，尤其是在缺乏特定标签的领域。使用带标签的过去示例进行重复训练可能因导致过拟合而进一步加剧这一问题，阻碍系统分类新类别的能力。为了应对这些挑战，本研究提出了ICARE，这是一个具有两个关键组件的新颖框架：实例选择模块和知识蒸馏方法。前者基于表示学习，精心选择呈现给NLI系统的训练示例。后者是基于特征和……的知识蒸馏方法（摘要在此处被截断）。

    arXiv:2611.24959v1 Announce Type: new  Abstract: The rise of powerful, fine-tuned large language models (LLMs) necessitates robust natural language inference (NLI) systems. These NLI systems ideally train across various tasks, encompassing diverse concepts like sentiment analysis, news classification, and coreference resolution. However, this continual, multi-tasking approach can lead to catastrophic forgetting, where previously learned tasks are lost, especially in domains lacking specific labels. Rehearsing past examples with labels can further exacerbate this issue by causing overfitting, hindering the system's ability to classify new categories. To tackle these challenges, this research proposes ICARE, a novel framework with two key components: an instance selection module and a knowledge distillation method. The former is based on representation learning, which meticulously selects training examples to present to the NLI system. The latter, knowledge distillation with feature and 
    
[^96]: LLM复杂度：迈向大语言模型推理的计算复杂度视角

    LLM Complexity: Towards a Computational Complexity View of Large Language Model Inference

    [https://arxiv.org/abs/2611.24943](https://arxiv.org/abs/2611.24943)

    该论文证明LLM推理时的token使用量与经典时间复杂度并非单调对应关系，并提出“LLM复杂度”这一新指标，通过准确率-token曲线下的面积从LLM的视角量化问题复杂度。

    

    计算复杂度理论为刻画计算问题的效率提供了一个原则性框架，而大语言模型（LLM）同样可以被视作执行“输入—推理时计算—输出”这一结构化过程的计算系统。在此背景下，生成token的数量成为衡量推理时计算量的一个自然且可直接观测的代理指标。然而，我们证明了LLM token长度与经典时间复杂度之间这一颇具吸引力的类比并不成立。我们否定了“推理时token使用量与经典时间复杂度单调一致”这一假设，并展示了其根本性分歧源于LLM在推理过程中实际分配计算的方式。为此，我们提出了LLM复杂度（LLM Complexity）这一新指标，通过刻画准确率-token曲线所界定的面积，从LLM的视角量化问题复杂度。我们发现……

    arXiv:2611.24943v1 Announce Type: cross  Abstract: Computational complexity theory offers a principled framework for characterizing the efficiency of computational problems, and Large Language Models (LLMs) can similarly be viewed as computational systems executing a structured process of input, inference-time compute, and output. In this context, the number of generated tokens emerges as a natural and directly observable proxy for inference-time compute. However, we show that the tempting analogy between LLM token length and classical time complexity does not hold. We disprove the hypothesis that inference-time token usage aligns monotonically with classical time complexity, demonstrating a fundamental divergence grounded in how LLMs actually allocate computation during inference. To this end, we introduce LLM Complexity, a new metric designed to quantify problem complexity from the perspective of LLMs by characterizing the area bounded above by the accuracy-token curve. We find that 
    
[^97]: 面向小语言模型更结构化医学解释的自奖励强化学习

    Self-Rewarded Reinforcement Learning for Better-Structured Medical Explanations in Small Language Models

    [https://arxiv.org/abs/2611.24650](https://arxiv.org/abs/2611.24650)

    该论文提出一种无需人工偏好标注的自奖励强化学习方法，通过简单的自监督奖励引导 PPO 训练，显著改善了小语言模型生成的医学解释的结构与质量。

    

    小语言模型（SLM）可以在普通硬件上运行，并支持保护隐私的本地化临床部署，但与大型语言模型相比，其生成的医学解释往往简短且结构欠佳。近期的监督式方法（如 Meerkat）将思维链推理迁移到 7B-8B 参数规模的骨干模型中，但其训练目标并未直接针对所生成解释的风格或组织结构。基于指令微调的 Mistral-7B 骨干模型，我们研究了轻量级的强化学习（RL）步骤能否在无需人工偏好标注的情况下改善解释的结构。我们在 2,000 个源自 Meerkat 的医学提示上应用了结合 LoRA 适配的近端策略优化（PPO），并由一个简单的自监督奖励引导，该奖励偏好更长、词汇更丰富且标点规范的回答，同时惩罚过短的回答。在单个 NVIDIA A100 GPU 上训练约 170 个 PPO 步骤后，得到了（摘要在此处截断）……

    arXiv:2611.24650v1 Announce Type: new  Abstract: Small language models (SLMs) can run on commodity hardware and support privacy-preserving, on-premises clinical deployment, but their medical explanations are often short and poorly structured compared with large language models. Recent supervised approaches such as Meerkat transfer chain-of-thought reasoning into 7B-8B backbones, yet their training objective does not directly target the style or organization of the generated rationales. Building on the instruction-tuned Mistral-7B backbone, we investigate whether a lightweight reinforcement learning (RL) step can improve explanation structure without human preference labels.We apply Proximal Policy Optimization (PPO) with LoRA adaptation to 2,000 Meerkat-derived medical prompts, guided by a simple self-supervised reward that favors longer, lexically diverse, and well-punctuated responses while penalizing very short ones. Training for about 170 PPO steps on a single NVIDIA A100 GPU yield
    
[^98]: LipSpell：一种结合区别特征编辑距离、键盘几何与语料库频率的语言无关语音评分拼写纠正框架

    LipSpell: A Language-Independent Phonetic Scoring Framework for Spelling Correction Combining Distinctive-Feature Edit Distance, Keyboard Geometry, and Corpus Frequency

    [https://arxiv.org/abs/2611.24285](https://arxiv.org/abs/2611.24285)

    LipSpell提出了一个语言无关的语音评分框架，结合基于IPA区别特征的音素编辑距离、键盘几何结构和语料库频率先验，在六个英语基准测试上全面超越Norvig和Hunspell等传统拼写纠正方法。

    

    自动拼写纠正必须同时处理键盘输入手误和认知或语音层面的拼写错误，即书写者根据单词的大致发音进行拼写的情况。传统拼写检查器要么依赖正字法编辑距离和键盘邻近度——当语音错误在多个字符上出现偏差时，这类方法表现不佳——要么依赖以英语为中心的语音哈希方法，如Soundex和Metaphone。我们提出了LipSpell，这是一个统一的、语言无关的语音评分引擎，配备可替换的特定语言G2P（字素到音素）前端、键盘映射和词典，它集成了：(1) 基于通用IPA区别特征预计算的|P|x|P|音素距离矩阵，(2) 物理键盘几何结构，以及(3) 语料库一元词频率先验，并通过两阶段对称删除索引和七类语音代码索引进行加速。在六个英语基准测试（包括来自TOEFL-Spell的1,000个二语学习者错误）上，LipSpell在每个数据集上都优于Norvig和Hunspell（注：摘要原文在此处被截断）。

    arXiv:2611.24285v1 Announce Type: new  Abstract: Automatic spelling correction must handle both keyboard typographical slips and cognitive or phonetic misspellings where writers approximate a word's pronunciation. Conventional spell checkers rely either on orthographic edit distance and keyboard proximity--which struggle when phonetic errors diverge by multiple characters--or on English-centric phonetic hashes such as Soundex and Metaphone. We present LipSpell, a unified, language-independent phonetic scoring engine paired with swappable language-specific G2P frontends, keyboard maps, and lexicons integrating: (1) a precomputed |P|x|P| phone-distance matrix from universal IPA distinctive features, (2) physical keyboard geometry, and (3) corpus unigram frequency priors, accelerated by a two-stage symmetric-delete and seven-class phonetic-code index. Across six English benchmarks (including 1,000 L2 errors from TOEFL-Spell), LipSpell outperforms Norvig and Hunspell on every dataset and b
    
[^99]: 面向头部锥束CT盲刚性运动校正的几何桥接流匹配方法

    Geometry-bridge flow matching for blind rigid-motion correction in head cone-beam CT

    [https://arxiv.org/abs/2611.23788](https://arxiv.org/abs/2611.23788)

    提出一种基于几何桥接的3D流匹配先验方法，结合样条位姿估计、共轭梯度数据一致性和全变差正则化的即插即用循环框架，实现头部锥束CT的盲刚性运动校正。

    

    arXiv:2611.23788v1 公告类型： cross 摘要：在头部锥束CT（CBCT）扫描过程中，患者头部的刚性运动会降低图像质量，并使从测量的投影数据中恢复精确的解剖结构变得困难。我们提出了一种3D流匹配先验方法，用于学习运动引起的图像变化。其“几何桥接”机制通过逐渐减弱的模拟运动，将同一源体积的运动污染FDK（Feldkamp-Davis-Kress）重建图像与无运动FDK重建图像连接起来。前向投影器的几何导数沿该路径提供解析速度目标。一个即插即用的循环框架将学习到的图像更新与基于样条的位姿估计、共轭梯度数据一致性约束以及图像全变差正则化相结合。研究对优化后的图像体积和在估计位姿下的FDK重建图像体积均进行了评估。实验基于来自30名留出CQ500患者的模拟360视角扫描数据，名义单分量运动幅度...

    arXiv:2611.23788v1 Announce Type: cross  Abstract: Rigid motion of a patient head during head cone-beam CT (CBCT) scan can degrade image quality and complicate recovery of accurate anatomical structures from the measured projections. We present a 3D flow-matching prior approach that learns the image changes induced by motion. Its geometry-bridge connects motion-corrupted and motion-free Feldkamp-Davis-Kress (FDK) reconstructed images of the same source volume through progressively reduced simulated motion. The forward projector's geometry derivative supplies analytic velocity targets along this path. A plug-and-play loop encompasses the learned image updates together with spline-based pose estimation, conjugate-gradient data consistency, and image total-variation regularization. Both the refined image volume and the FDK reconstructed image volume at the estimated poses are evaluated. On simulated 360-view scans from 30 held-out CQ500 patients with nominal per-component motion amplitude
    
[^100]: Chorus：具有逐层自适应的共享N-gram嵌入

    Chorus: Shared N-gram Embeddings with Layer-wise Adaptation

    [https://arxiv.org/abs/2611.23386](https://arxiv.org/abs/2611.23386)

    Chorus提出让整个网络共享一组n-gram嵌入表，每层通过基于自身隐藏状态计算的缩放因子和独立投影层以各自的方式读取共享表，从而以比现有方法小87%的表实现相当的质量。

    

    语言模型可以通过n-gram嵌入表以极小的计算代价获得容量提升：这些嵌入表是为连续token短序列存储学习向量的大型表，通过查表将其添加到模型的输入或中间层。近期的设计通常为每一层使用单独的n-gram嵌入表，这使得这些表在内存开销上十分昂贵，并且阻碍了各层之间共享所学内容。然而，共享单个表会使每一层获得相同的嵌入，没有为各层的偏好留下空间，因此在实践中表现不佳。我们提出Chorus，其中一组n-gram嵌入表服务于整个网络，而每一层以自己的方式读取它。每一层先根据其当前隐藏状态计算出的因子对各分量进行重新缩放，然后通过其自己的投影层映射该精炼后的嵌入。Chorus以小87%的表实现了与现有n-gram方法相当的质量：在1.2B表参数……

    arXiv:2611.23386v1 Announce Type: new  Abstract: Language models can gain capacity at little computational cost from n-gram embedding tables: large tables of learned vectors for short sequences of consecutive tokens, which are looked up and added to the model's input or to its intermediate layers. Recent designs commonly use a separate n-gram embedding table for each layer, which makes the tables costly in memory and prevents layers from sharing what they learn. Sharing a single table, however, gives every layer the same embedding and leaves no room for layer-wise preferences, and it performs poorly in practice. We propose Chorus, in which one set of n-gram embedding tables serves the whole network and every layer reads it in its own way. Each layer maps the refined embedding through its own projection layer, after rescaling its components by factors computed from its current hidden state. Chorus matches the quality of existing n-gram methods with 87% smaller tables: with 1.2B table pa
    
[^101]: The K=V Cache of Gemma 4 Only Needs to Be Stored Once: Achieving Exact Global-KV Memory Savings in a Paged Serving Engine Without Modifying the Attention Kernel

    Storing the Gemma 4 K=V Cache Once: Exact Global-KV Memory Savings in a Paged Serving Engine with Unchanged Attention Kernels

    [https://arxiv.org/abs/2611.23035](https://arxiv.org/abs/2611.23035)

    This paper exploits the linear dependence between the key and value in the global attention layers of Gemma 4 31B (K=V), storing only one 640-value row without modifying the attention kernel, which reads the key and value caches through two overlapping strided views, achieving a 1.34 to 1.40 fold reduction in KV cache memory during decoding with bit-identical outputs.

    

    In the global attention layers of Gemma 4 31B, the value is the normalized key projection, and the key is the value multiplied by a fixed weight, with rotary position encoding applied to only one quarter of its dimensions. The Gemma 4 technical report points out that this design can reduce the global KV cache by up to 37.5%, but the open-source serving engines we inspected still store a complete key and a complete value: 1024 values per KV head per token. We instead store only one row of 640 values, which is exact in real-number arithmetic, and an unmodified paged attention kernel reads it as its key cache and value cache through two overlapping strided views. In a vLLM prototype using the 16-bit 31B model, the outputs are bit-identical to an emulation of the same arithmetic on the original cache; 2 of 432 final answers differ from the original engine (within its run-to-run noise range); and at 32K to 64K tokens, the steady decoding KV memory per request decreases by a factor of 1.34 to 1.40. On a pair of prefill/decode...

    arXiv:2611.23035v1 Announce Type: cross  Abstract: In the global attention layers of Gemma 4 31B, the value is the normalized key projection, and the key is the value scaled by a fixed weight, with rotary position encoding on one quarter of its dimensions. The Gemma 4 report states that this design reduces the global KV cache by up to 37.5%, but the open serving engines that we checked still store a full key and a full value: 1,024 values per KV head and token. We store one 640-value row instead, exact in real arithmetic, and an unchanged paged attention kernel reads it through two overlapping strided views as its key cache and its value cache. In a vLLM prototype with the 16-bit 31B model, the outputs are bit-identical to an emulation of the same arithmetic on the stock cache, 2 of 432 final answers differ from the stock engine (within its run-to-run noise), and the steady decode KV memory per request falls by a factor of 1.34 to 1.40 at 32K to 64K tokens. On a prefill/decode pair of 
    
[^102]: EDITORS懂你的风格！编辑LoRA子空间用于风格归属与模仿

    EDITORS Know Your Style! Editing LoRA Subspaces for Stylistic Attribution and Imitation

    [https://arxiv.org/abs/2611.20846](https://arxiv.org/abs/2611.20846)

    提出EDITORS框架，通过“先诊断后消减”策略编辑LoRA子空间，将输入激活投影远离内容导向方向，从而解耦风格与内容，在作者归属和风格模仿两类任务上均取得最先进表现。

    

    将写作风格与语义内容解耦是文学文本建模中的一项根本性挑战。风格与内容的纠缠导致模型在作者归属（AA）任务中依赖语义内容，并在风格模仿（SI）任务中记忆特定作者的内容。我们提出了EDITORS（EditingLoRA子空间），这是一个“先诊断后消减”的框架，它通过适配预训练大语言模型并优先考虑风格特征，使其在AA和SI任务中均能发挥作用。EDITORS首先使用从训练语料库中提取的风格中性语句训练一个诊断性LoRA适配器，以识别面向内容的子空间。然后，它在激活空间消减的条件下训练最终的风格适配器，将输入激活投影到远离已识别的内容导向方向上。我们通过实验证明了EDITORS的有效性。在涵盖文学、社交媒体和新闻的三个AA基准测试中，EDITORS取得了最先进的结果。

    arXiv:2611.20846v1 Announce Type: cross  Abstract: Disentangling writing style from semantic content is a fundamental challenge in literary text modeling. Style-content entanglement causes models to rely on semantic content for authorship attribution (AA) tasks and memorize author-specific content for style imitation (SI) tasks. We propose EDITORS (EditingLoRA Subspaces), a diagnose-then-deflate framework that adapts pretrained large language models and prioritizes stylistic features, making it effective for both AA and SI tasks. EDITORS first trains a diagnostic LoRA adapter using style-neutral statements derived from the training corpus to identify a content-oriented subspace. It then trains the final style adapter under activation-space deflation that projects input activations away from the identified content-oriented directions. We empirically demonstrate that EDITORS is effective. On three AA benchmarks spanning literature, social media, and news, EDITORS achieves state-of-the-ar
    
[^103]: 形式语义学的形式化：形式图谱

    Formalizing Formal Semantics: The Formal Atlas

    [https://arxiv.org/abs/2611.19509](https://arxiv.org/abs/2611.19509)

    本文提出了基于证明助手的“形式图谱”系统，通过交互式应用将语义学文献与形式化证明连接起来，实现语义理论的形式化、预测检验与比较，并探索AI辅助开发与推理的应用前景。

    

    在本文中，我们提出了形式图谱，这是一个使用证明助手研究自然语言语义的系统。我们的目标是形式化语义分析、检验其预测并比较所实现的理论。为此，我们通过一个交互式应用程序将文献来源记录与形式化工作连接起来，研究人员可以在其中查看证明并修改受支持的 Rocq 文件。蒙塔古的 PTQ 与 Gallin 的著作为我们提供了主要案例，随后是动态语义与类型论语义的比较、RSA 以及分布式语义。我们还讨论了形式化那些不够显式的分析所需的条件。最后，我们探讨了人工智能的应用，既可用于辅助开发这些资源，也可用于能够利用这些资源的推理系统。这项工作扩展了早期证明助手语义学的研究计划，研究与教学是其互补的用途。

    arXiv:2611.19509v1 Announce Type: new  Abstract: In this paper we present the Formal Atlas, a system for the study of natural language semantics using proof assistants. Our aim is to formalize semantic accounts,   check their predictions and compare the implemented theories. To this end, we connect source records and formal developments with an interactive application, where   researchers can inspect proofs and modify supported Rocq files. Montague's PTQ and Gallin provide our main cases, followed by dynamic and type-theoretic   comparisons, RSA and distributional semantics. We also discuss what it takes to formalize less explicit analyses. Lastly, we consider the use of AI, both for   assistance in developing these resources and for inference systems that can make use of them. This work extends an earlier programme in proof-assistant semantics,   with research and teaching as complementary uses.
    
[^104]: 如何造就道德主体：以情感主义与理性主义审视人工智能

    How to make a moral agent: sentimentalism and rationalism as lenses on AI

    [https://arxiv.org/abs/2611.18714](https://arxiv.org/abs/2611.18714)

    该论文提出以描述性而非规范性路径研究AI道德主体地位，借鉴哲学与道德心理学分离出构成人类道德主体性的候选能力并加以细化，进而为大语言模型是否具备这些能力提供了可操作的实证检验方案。

    

    人工智能系统能否成为道德主体？也就是说，它们能否采取那些可以被正确评价为道德上正确或错误的行为？目前尚无一个公认的、能够妥善适用于典型人类案例之外情形的道德主体规范性标准。由于规范性阐释面临诸多困难，我们转而提出一个研究计划来探索描述性的道德主体。最根本的问题是：我们人类拥有哪些能力构成了我们道德主体性的基础？为此，我们借鉴哲学与道德心理学的历史学术研究来分离出候选能力。接着，我们建议对这些候选能力进行细化，使其能够在人工智能系统的语境中加以检验。最后，我们提出了具体的实证建议，用于测试人工智能系统（尤其是大语言模型，LLMs）是否具备这些细化后的能力。我们的方法具有若干益处。首先，它或许能够揭示人工智能系统的行为是否与其保持一致，或者（原文此处截断）……

    arXiv:2611.18714v1 Announce Type: new  Abstract: Can AI systems be moral agents? That is, can they undertake actions that can be correctly assessed as morally right or wrong? There is no settled normative criterion of moral agency that applies neatly outside paradigm human cases. Because of the difficulties with a normative account, we instead propose a research program to explore descriptive moral agency. Most basically, we ask, what are the capabilities we humans possess that underlie our moral agency? To do so, we draw on historical scholarship in philosophy and moral psychology to isolate candidate capabilities. Then we suggest refining candidate capabilities so that they are testable in the context of AI systems. Finally, we offer specific empirical recommendations to test AI systems, especially large language models (LLMs), for those refined capabilities. Our approach offers several benefits. Firstly, it may demonstrate whether the behaviors of AI systems are aligned, or in a bro
    
[^105]: SynthDCB：面向可扩展设备控制基准测试的世界蒸馏方法

    SynthDCB: World Distillation for Scalable Device Control Benchmarking

    [https://arxiv.org/abs/2611.18639](https://arxiv.org/abs/2611.18639)

    该论文提出SynthDCB基准，创新性地通过“世界蒸馏”方法将设备行为知识转化为模块化可执行组件，构建了覆盖家居、汽车和手机三大领域的可扩展设备控制与环境感知评测基准。

    

    扩展设备控制评估需要同时扩展智能体所交互的世界以及它们需要解决的任务。我们提出了SynthDCB，一个通过世界蒸馏构建的、可扩展的环境感知与设备控制基准。世界蒸馏将设备行为、交互约束和环境效应的知识转化为模块化的可执行组件，这些组件被组合成具有显式状态、动作和时间动态的交互式世界。统一的能力表示将工具发现、世界执行和任务合成连接起来，使新的能力能够通过共享的构建流水线进入评估。SynthDCB实例化了家居、汽车和手机三个领域，包含114个设备类别、1,166个功能和95个事件，支持五大任务族共32个子类型。这种有状态的设计支持通过状态正确性和工具调用轨迹进行任务验证与智能体评估。我们评估了……

    arXiv:2611.18639v1 Announce Type: cross  Abstract: Scaling device-control evaluation requires scaling both the worlds agents interact with and the tasks they must solve. We introduce SynthDCB, a scalable benchmark for environment awareness and device control built through world distillation. World distillation transforms knowledge of device behaviors, interaction constraints, and environmental effects into modular executable components, which are composed into interactive worlds with explicit states, actions, and temporal dynamics. A unified capability representation connects tool discovery, world execution, and task synthesis, allowing new capabilities to enter evaluation through a shared construction pipeline. SynthDCB instantiates Home, Car, and Phone domains with 114 device classes, 1,166 functions, and 95 events, supporting five task families with 32 subtypes. The stateful design enables task validation and agent evaluation through state correctness and tool-call traces. We evalua
    
[^106]: LazyWeights：按需解码的边缘大语言模型

    LazyWeights: Decode-on-Demand Edge LLMs

    [https://arxiv.org/abs/2611.17108](https://arxiv.org/abs/2611.17108)

    LazyWeights在保留现有量化格式的基础上，通过对权重进行无损Huffman编码并在DRAM与L2之间用小型ASIC单元按需解码，在不改动GPU内核的前提下使边缘设备上的模型体积缩小5.7%-31.4%，并将每token的延迟和能耗最多降低约一半。

    

    内存决定了边缘设备能运行哪些大语言模型以及运行得多快：在4 GB的Jetson Nano上受容量限制，在8 GB的Jetson Orin Nano Super上则受带宽限制。现有部署格式（GPTQ、AWQ、llama.cpp的K量化和I量化）对矩阵的每个整数编码分配相同的位数，尽管这些编码的分布远非均匀。LazyWeights保留了这些格式，但通过调整GPTQ使编码更易压缩，在交错通道中对编码进行无损Huffman编码，并在DRAM与L2缓存之间的一个小型单元中按需解码，因此GPU内核无需任何改动，DRAM只需传输编码后的字节。在1%和3.5%的困惑度预算下，编码后的Phi-3-mini、Mistral-7B和SmolLM-1.7B比最佳的llama.cpp格式小5.7%-31.4%。一个位精确的0.21 mm² ASIC解码单元（1 GHz，7 nm工艺）将Orin上每个token的延迟和能耗最多降低36%和38%（结合prompt-lookup投机解码时可降低54%和44%）。在Nano上，LazyWeights每token所需时间比CPU版llama.cpp减少58%-64%，能耗减少53%-57%，并能运行……

    arXiv:2611.17108v2 Announce Type: replace-cross  Abstract: Memory decides which LLMs an edge device runs, and how fast: capacity on a 4 GB Jetson Nano, bandwidth on an 8 GB Jetson Orin Nano Super. Deployed formats (GPTQ, AWQ, llama.cpp's K- and I-quants) spend equal bits on every integer code of a matrix, although the codes are far from uniform. LazyWeights keeps these formats but tunes GPTQ so the codes compress, Huffman-codes them losslessly in interleaved lanes, and decodes them on demand in a small unit between DRAM and L2, so GPU kernels run unchanged and DRAM moves only coded bytes. At 1% and 3.5% perplexity budgets, coded Phi-3-mini, Mistral-7B and SmolLM-1.7B are 5.7-31.4% smaller than the best llama.cpp format. A bit-exact 0.21 mm^2 ASIC decoder unit (1 GHz, 7 nm) cuts Orin latency and energy per token by up to 36% and 38% (54% and 44% with prompt-lookup speculation). On the Nano, LazyWeights needs 58-64% less time and 53-57% less energy per token than CPU llama.cpp, and runs 
    
[^107]: 捷径还是记忆？利用大语言模型重新审视自然语言理解中的数据破坏

    Shortcuts or Memorisation? Revisiting Data Corruption in Natural Language Understanding with Large Language Models

    [https://arxiv.org/abs/2611.16843](https://arxiv.org/abs/2611.16843)

    该研究通过对八个大语言模型在GLUE和ANLI上的系统性扰动实验证明，语言模型面对数据破坏的鲁棒性来源于对捷径特征的利用而非对预训练数据的记忆，且只有在作答前先进行推理的模型才能摆脱这种捷径依赖模式。

    

    在自然语言推理（NLI）及其他语言理解任务上微调的预训练编码器，即使输入受到扰动（例如删除所有名词或打乱词序），仍能保持大部分准确率。大语言模型（LLM）是否也是如此？如果是，这是否是因为它们在预训练期间见过这些基准测试数据？本文使用八个大语言模型以及在GLUE和三轮ANLI上的微调编码器重新开展了这些扰动实验。无论模型规模大小，所有直接作答的系统在删除名词后的MNLI上仍保留了36-72%的超过随机水平的性能，而在ANLI上仅保留14-29%。只有在回答前先进行推理的模型才脱离了这一模式。我们还检查了数据污染指标（包括在两个模型的公开预训练数据中统计精确的n-gram出现次数），发现数据暴露与鲁棒性之间并无关联。我们最有力的证据来自368个新的、经人工验证的……

    arXiv:2611.16843v2 Announce Type: replace  Abstract: Pre-trained encoders fine-tuned on natural language inference (NLI) and other language understanding tasks keep much of their accuracy after the input is perturbed, e.g. by removing all nouns or shuffling the words. Does the same hold for large language models (LLMs), and if so, is it because they have seen the benchmarks during pre-training? In this paper we revisit these perturbation experiments with eight LLMs alongside fine-tuned encoders on GLUE and the three rounds of ANLI. Regardless of model size, every system that answers directly retains 36-72% of its above-chance performance without nouns on MNLI compared to only 14-29% on ANLI. Only a model that reasons before answering departs from this pattern. We also check contamination indicators (including exact $n$-gram counts in the public pre-training data of two of the models) and find no link between exposure and robustness. Our strongest evidence comes from 368 new, human-vali
    
[^108]: 《Merge能构成霍普夫代数吗？余积、语迹与代数极简主义的前景》

    Does Merge Form a Hopf Algebra? Coproducts, Traces, and the Prospects of Algebraic Minimalism

    [https://arxiv.org/abs/2611.11502](https://arxiv.org/abs/2611.11502)

    本文检验了“Merge能否构成霍普夫代数”这一代数极简主义的核心数学前提，通过对五个候选余积的余结合性与连通分次分析发现：MCB的Δ_d不满足余结合性，Δ_c需补充语迹标记规则与分次后才能给出分次连通霍普夫代数，而Xiao与Marcolli-Skigin的余积则是余结合的。

    

    arXiv:2611.11502v2（更正公告）。摘要：Marcolli、Chomsky与Berwick（2025，下文简称MCB）在Connes-Kreimer型的霍普夫代数上为极简主义的Merge操作建立了形式化表述，并声称语言与物理遵循相同的代数结构。本文检验了这一主张的前提：工作区上的Merge能否被描述为一个霍普夫代数。我们首先从余结合性与连通分次的角度考察了五个候选余积：Connes-Kreimer余积Δ_ρ、MCB的两个余积Δ_d与Δ_c、Marcolli与Skigin（2025）的收缩余积，以及Xiao（2026）的余积。结果表明：Δ_d不是余结合的；对于Δ_c，MCB并未给出在已含语迹叶子的树中如何标记语迹叶子标签的规则，按字面推广该余积不是余结合的，但一旦补充相应规则与分次，它便给出一个分次连通的霍普夫代数；Xiao的余积是余结合的，而Marcolli与Skigin的余积在……（摘要原文在此处截断）同样是余结合的。

    arXiv:2611.11502v2 Announce Type: replace  Abstract: Marcolli, Chomsky, and Berwick (2025; henceforth MCB) formulate the Merge operation of Minimalism on a Hopf algebra of Connes-Kreimer type, and claim that language and physics obey the same algebraic structure. This paper examines the premise of that claim: whether Merge on workspaces can be described as a Hopf algebra. We first survey five coproducts for coassociativity and a connected grading: the Connes-Kreimer coproduct $\Delta_\rho$, MCB's two coproducts $\Delta_d$ and $\Delta_c$, the contraction coproduct of Marcolli and Skigin (2025), and the coproduct of Xiao (2026). $\Delta_d$ is not coassociative. For $\Delta_c$, MCB gives no rule for the labels of trace leaves in trees that already contain trace leaves; under the literal extension it is not coassociative, but once a rule and a grading are supplied it yields a graded connected Hopf algebra. The coproduct of Xiao is coassociative, and so is that of Marcolli and Skigin once i
    
[^109]: MedPsy：面向高效边缘部署的最先进小型医学语言模型

    MedPsy: State-of-the-Art Small Medical Language Models for Efficient Edge Deployment

    [https://arxiv.org/abs/2611.02011](https://arxiv.org/abs/2611.02011)

    MedPsy是一系列专为边缘设备部署设计的1.7B和4B小型医学语言模型，通过合成医学数据流水线、四阶段后训练课程（两个SFT加两个带难例挖掘的RL阶段）以及GGUF量化边缘部署评估，在精度、生成效率与部署可行性之间实现了最优平衡。

    

    医学语言模型有望提供专家级临床支持，但最强大的开源大语言模型对于私有边缘设备部署而言仍然过于庞大，而紧凑型医学模型在知识密集型和真实世界医疗任务上的表现仍存在明显差距。符合边缘内存预算的推理模型可能产生冗长的推理轨迹，从而增加顺序解码开销、端到端延迟、能耗以及KV缓存内存压力。我们提出了MedPsy，这是一系列专为边缘部署设计的纯文本1.7B和4B医学语言模型，旨在解决这一精度-生成-效率的权衡问题。我们的方案结合了以下要素：基于生物学、医学和健康种子数据构建的合成医学数据流水线，并辅以来自235B医学专业大语言模型教师模型的思维链目标；一个包含两个SFT阶段和两个带难例挖掘的RL阶段的四阶段后训练课程；以及覆盖多种GGUF量化格式的边缘部署评估。

    arXiv:2611.02011v1 Announce Type: new  Abstract: Medical language models promise expert clinical support, but the strongest open-source LLMs remain too large for private edge-device deployment, while compact medical models still trail substantially on knowledge-intensive and real-world healthcare tasks. Reasoning models that fit edge memory budgets can produce long reasoning traces, increasing sequential decoding, end-to-end latency, energy consumption, and KV-cache memory pressure. We present MedPsy, a family of text-only 1.7B and 4B medical language models designed for edge deployment, targeting this accuracy-generation-efficiency trade-off. Our recipe combines a synthetic medical data pipeline over biology, medicine, and health seeds with chain-of-thought targets from a 235B medical-focused LLM teacher; a four-stage post-training curriculum comprising two SFT stages followed by two RL stages with hard-sample mining; and an edge deployment evaluation spanning multiple GGUF quantizati
    
[^110]: 从碎片化到融合：全模态嵌入的综述与路线图

    From Fragmentation to Fusion: A Survey and Roadmap for Omnimodal Embeddings

    [https://arxiv.org/abs/2611.00172](https://arxiv.org/abs/2611.00172)

    该综述指出跨模态嵌入研究正因对比学习、Transformer架构和粒度转变而走向统一，提出了“覆盖性—可比性—组合性”三层递进能力框架，并倡导以联合嵌入作为实现全模态统一的最终路径。

    

    嵌入模型支撑着几乎所有现代AI系统，从搜索引擎到推荐系统。尽管共享相同的数学基础，但嵌入模型的研究仍因模态不同而碎片化：各个社区的发展在很大程度上相互隔离，很少交流见解。我们认为这种碎片化掩盖了一种更深层次的趋同。在对比学习、Transformer架构和粒度转变的驱动下，跨模态的嵌入研究正汇聚于一套共同的原则，尽管模态特定的挑战仍未得到充分探索。为了在不混淆这种趋同的前提下对其进行推理，我们提出了一个由三种递进式能力构成的框架：覆盖性、可比性和组合性。通向统一的三条路径依次满足这些能力：转换拓宽了覆盖性，对齐实现了可比性，而联合嵌入——我们倡导的终极目标——则旨在实现组合性。这一立场……

    arXiv:2611.00172v1 Announce Type: new  Abstract: Embedding models underpin virtually every modern AI system, from search engines to recommendation systems. Despite shared mathematical foundations, research on embedding models remains fragmented by modality: the communities develop largely in isolation, seldom exchanging insights. We argue that this fragmentation obscures a deeper convergence. Driven by contrastive learning, Transformer architectures, and a granularity shift, embedding research across modalities is converging toward a common set of principles, even as modality-specific challenges remain underexplored. To reason across this convergence without conflating it, we propose a framework of three progressively harder capabilities: coverage, comparability, and compositionality. Three paths toward unification satisfy these in turn: transformation broadens coverage, alignment enables comparability, and joint embedding, the terminus we advocate, aims at compositionality. This posit
    
[^111]: HEDGE：面向可扩展多智能体辩论的门控超图记忆

    HEDGE: Gated Hypergraph Memory for Scalable Multi-Agent Debate

    [https://arxiv.org/abs/2610.38902](https://arxiv.org/abs/2610.38902)

    HEDGE提出一种基于门控超图的选择性记忆框架，通过超边跨智能体关联论点并利用学习型记忆选择模块对信息进行存储、备份或忽略决策，从而解决多智能体辩论中记忆检索困难与高阶关系建模不足的问题，在七个基准上持续超越现有方法。

    

    多智能体辩论系统要求智能体在多轮辩论中保留并回顾论点，然而有限的上下文窗口和非结构化记忆使得相关信息的检索变得越来越困难。现有的扁平记忆和成对图记忆也难以捕捉高阶关系，包括多智能体之间的共识、相互矛盾的证据以及不断演变的共识状态。我们提出了HEDGE，这是一种选择性记忆框架，将辩论轨迹表示为语义超图。超边连接跨智能体的相关论点，而超边内部的关系图则保留了细粒度的论证结构。一个通过监督学习和强化学习训练的记忆选择模块（MSM）将每个记忆候选路由到STORE（存储）、BACKUP（备份）或OMIT（忽略）三种操作，在保留有用信息的同时限制冗余。在七个多样化的基准测试中，HEDGE持续优于强大的单智能体、多智能体和……

    arXiv:2610.38902v2 Announce Type: replace  Abstract: Multi-agent debate systems require agents to retain and revisit arguments across multiple rounds, yet limited context windows and unstructured memories make relevant information increasingly difficult to recover. Existing flat and pairwise-graph memories also struggle to capture higher-order relations, including multi-agent agreement, conflicting evidence, and evolving consensus. We introduce HEDGE, a selective memory framework that represents debate trajectories as semantic hypergraphs. Hyperedges connect related claims across agents, while intra-hyperedge relational graphs preserve fine-grained argumentative structure. A learned memory selection module (MSM), trained through supervised learning and reinforcement learning, routes each memory candidate to STORE, BACKUP, or OMIT, retaining useful information while limiting redundancy. Across seven diverse benchmarks, HEDGE consistently outperforms strong single-agent, multi-agent, and
    
[^112]: REACT：思维链轨迹的表征早期分析与控制

    REACT: Representational Early Analysis and Control of Chain-of-Thought Trajectories

    [https://arxiv.org/abs/2610.32113](https://arxiv.org/abs/2610.32113)

    REACT 提出一种从成对安全与不安全轨迹中学习安全方向、并一次性施加于思维链早期隐藏状态的激活干预方法，发现最强控制增益集中在思维链的前两到三个 token 窗口，即使后续状态更能预测风险。

    

    思维链生成提供了一种在推理模型完成回答之前进行干预的机会。但是，风险最容易被检测到的时间点是否也能提供最有效的控制？我们通过 REACT 来研究这个问题，这是一种条件性激活干预方法，它从成对的安全与不安全轨迹中学习一个安全方向，并将其一次性应用于早期的思维链隐藏状态。模型权重保持冻结，所有后续 token 的生成不再进行进一步编辑。为了研究时机问题，我们在前五个思维链位置上分别训练方向，并使用无门控编辑配合编辑后的完整重新生成进行评估。在三个蒸馏推理模型和三个生成种子上，最强的增益集中在两到三个 token 的早期窗口内；在 DeepSeek-R1-Distill-Llama-8B 上，效果在第四和第五个 token 处降至接近于零，尽管较晚的状态可能更能预测风险。

    arXiv:2610.32113v1 Announce Type: cross  Abstract: Chain-of-thought (CoT) generation offers an opportunity to intervene before a reasoning model completes its answer. But does the point at which risk is easiest to detect also offer the most effective control? We study this question with REACT, a conditional activation intervention that learns a safety direction from paired safe and unsafe trajectories and applies it once to an early CoT hidden state. The model weights remain frozen, and all subsequent tokens are generated without further edits. To study timing, we train directions separately at each of the first five CoT positions and evaluate ungated edits with full post-edit regeneration. Across three distilled reasoning models and three generation seeds, the strongest gains concentrate in a two- to three-token early window; on DeepSeek-R1-Distill-Llama-8B, the effect falls to approximately zero at the fourth and fifth tokens, even though later states can be more predictive of risk. 
    
[^113]: 面向数学密集型科学文本的多模态AI检测

    Multimodal AI Detection for Math-Heavy Scientific Text

    [https://arxiv.org/abs/2610.21062](https://arxiv.org/abs/2610.21062)

    该研究构建了包含LaTeX源码、PDF渲染图像和提取文本三种表示形式的人机配对科学文本数据集，发现仅依赖文本的AI检测器在数学密集型内容上准确率显著下降，证明了利用多模态视觉信息对检测AI生成的科学文本至关重要。

    

    随着大语言模型在生成和辅助科学写作方面的能力日益增强，研究者在使用AI进行研究时需要更高的透明度和信息披露。因此，一个专门的研究子领域以及众多商业服务应运而生，旨在自动标记AI生成的内容。然而，现有仅依赖于从PDF中提取文本的检测器，无法利用数学密集型科学内容排版格式中所包含的丰富视觉信息。为研究这一问题，我们从2022年之前的arXiv LaTeX源文件中选取人类撰写的证明和正文段落，并在同一篇论文的上下文中使用15个现代大语言模型重新生成这些内容。这使得我们获得了三种不同表示形式下的人机配对输出：LaTeX源代码、PDF渲染图像以及从PDF中提取的文本。我们发现，最先进的Pangram 4的检测准确率随着数学内容……（摘要此处被截断）

    arXiv:2610.21062v1 Announce Type: new  Abstract: As large language models become increasingly capable of generating, and assisting with, scientific writing, there is a need for greater transparency and disclosure around the use of AI in research. Consequently, a veritable subfield of research and a number of commercial services aim to flag AI-generated content automatically. However, existing detectors which rely only on text extracted from PDFs fail to benefit from the rich visual information contained in the formatting of math-heavy scientific content. To study this, we select human-written proof and prose passages from pre-2022 arXiv LaTeX sources and re-generate them using 15 modern LLMs within the context of the same paper. This leaves us with coupled human and AI outputs in three different representations: the LaTeX source, a rendered image of the PDF, and the text extracted from the PDF. We find that the accuracy of the state-of-the-art Pangram 4 drops significantly as the mathe
    
[^114]: PRISM：工具调用语言模型中激活干预的迁移与局限

    PRISM: Transfer and Limits of Activation Interventions in Tool-Calling Language Models

    [https://arxiv.org/abs/2610.20808](https://arxiv.org/abs/2610.20808)

    该论文提出PRISM框架，证明利用量化NF4标签和跨语言迁移等低资源设置即可稳定估计激活干预方向，并将安全对齐改进跨语言迁移到阿拉伯语等保留语言，在无需目标语言调优的情况下将Llama-3.1-8B的不安全工具调用违规减少21.95个百分点。

    

    近期文献表明，语言模型中不安全的工具调用常常伴随着文本层面的拒绝。为了研究这一隐藏漏洞，我们提出了PRISM，一个用于评估激活干预在不同精度、语言和规模之间迁移情况的框架。我们的研究结果表明，利用低资源设置（特别是量化NF4标签和跨语言迁移）为暴露和改进模型对齐提供了一条可靠的途径。值得注意的是，使用来自NF4副本的行为标签，即使只有极少量的场景也能实现稳定的干预估计。此外，这些对齐改进能够有效地跨语言迁移：由五种语言监督的NF4方向成功迁移到保留测试的阿拉伯语上，在无需目标语言调优的情况下，将Llama-3.1-8B中的不安全违规减少了21.95个百分点。通过映射成对的良性激活，我们还成功地将这些安全干预迁移到……（摘要原文在此处被截断）

    arXiv:2610.20808v1 Announce Type: new  Abstract: Recent literature shows that unsafe tool calls can often accompany textual refusals in language models. To investigate this hidden vulnerability, we introduce PRISM, a framework for evaluating the transfer of activation interventions across precision, languages, and scales. Our findings demonstrate that leveraging lower-resource settings (specifically quantized NF4 labels and cross-lingual transfer) offers a robust pathway to expose and improve model alignment. Notably, using behavioral labels from an NF4 copy enables stable intervention estimation even with very few scenarios. Furthermore, these alignment improvements transfer effectively across languages: five-language NF4-supervised directions successfully transfer to held-out Arabic, reducing unsafe violations in Llama-3.1-8B by 21.95 percentage points without target-language tuning. By mapping paired benign activations, we also successfully transfer these safety interventions to muc
    
[^115]: 提示词与规则之争：语音用户模拟器中语音自然性行为的审计与控制

    Prompts versus Rules: Auditing and Controlling Speech Naturalness Behaviors in Voice User Simulators

    [https://arxiv.org/abs/2610.11015](https://arxiv.org/abs/2610.11015)

    研究发现，基于提示词的方法无法可靠地让语音用户模拟器产生不流畅、打断和反馈等自然语音行为，而基于规则的注入算法能够更一致、更自然地控制这些行为。

    

    随着语音代理在商业上日益流行，用于评估已部署代理的用户模拟器也在不断发展，以包含更真实、多变和多样化的语音自然性行为——不流畅（disfluency）、打断（interruption）和反馈性附和（backchanneling）。用户模拟器的质量直接影响代理评估结果的有效性。然而，我们发现迄今为止的大多数研究都未曾详细检验这些行为的预期配置是否真的在模拟中得以实现。在本研究中，我们审计了tau-Voice（我们自己的基于LLM的提示方法，涵盖三个模型）以及我们的基于规则的注入算法（针对不流畅、打断和反馈）所实际实现的自然性行为。我们发现，通过提示词来引发这些行为是不可靠的，其产生的语音与指令不一致，其位置和分布也不如指令所暗示的那样自然。相比之下，我们基于规则的、与模型无关的算法……

    arXiv:2610.11015v1 Announce Type: new  Abstract: As voice agents gain more popularity commercially, the user simulators used to evaluate the deployed agents are also being developed to include more realistic, variable, and diverse speech naturalness behaviors -- disfluency, interruption and backchanneling. The quality of the user simulator directly affects the validity of agent evaluation results. However, we find that most studies so far have not examined in detail whether the intended configuration for these behaviors is realized in the simulation. In this study, we audit the realized naturalness behaviors of tau-Voice, our own LLM-based prompting approach across three models, and our rule-based injection algorithm for disfluency, interruption, and backchanneling. We find that prompting for these behaviors is unreliable and produces speech inconsistent with the instructions, placed and distributed less naturally than the instruction implies. In contrast, our rule-based, model-free al
    
[^116]: Wieszcz-XIX：一个31亿词规模的1918年前波兰语语料库及从零训练的时间受限语言模型

    Wieszcz-XIX: A 3.1-Billion-Word Corpus of Pre-1918 Polish and Temporally Bounded Language Models Trained From Scratch

    [https://arxiv.org/abs/2610.10592](https://arxiv.org/abs/2610.10592)

    该研究构建了31亿词规模的1800-1918年波兰语历史语料库Wieszcz-XIX，其规模比现有标注语料库大三个数量级以上，并通过过滤、去重和时间泄漏审计等严格流程量化控制数据缺陷，进而从零训练了时间受限的语言模型。

    

    历史波兰语作为一种语言虽有充分的文献记录，但就本文所涵盖的时期而言，以机器可读形式标注的文本仅有约一百万词，其余内容则受制于质量参差不齐的光学字符识别。我们提出了Wieszcz-XIX，这是一个包含67.5亿个词元（约31亿词）、由294,369份文档（其中大部分为期刊）组成的语料库，收录了1800至1918年间出版的波兰语文本。该语料库从Wolne Lektury和Internet Archive汇编而成，其构建流程包括过滤、去重、对1918年后内容泄漏的审计以及文档级别的切分。该语料库比同一时期的标注语料库大三个数量级以上，我们对其缺陷进行了量化：以假阳性底线为基准衡量的识别损坏、已被去除的近乎相同的重复内容，以及1918年后的内容泄漏——后者已从训练语料库本身中排除，仅在转录来源中残留已知比例的0.04%至0.38%（按字节数计），因此该……

    arXiv:2610.10592v1 Announce Type: new  Abstract: Historical Polish is well documented as a language but annotated in machine-readable form only to about a million words for the period this paper covers; the rest sits behind optical character recognition of variable quality. We present Wieszcz-XIX, a corpus of 6.75 billion tokens (about 3.1 billion words) in 294,369 documents, most of them periodical issues, of Polish published from 1800 to 1918, assembled from Wolne Lektury and the Internet Archive by a pipeline that filters, deduplicates, audits for post-1918 leakage and splits at the document level. It is over three orders of magnitude larger than the annotated corpus of the same period, and we quantify its defects: recognition corruption against a false-positive floor, near-identical duplication, which is removed, and post-1918 leakage, which is excluded from the training corpus itself down to a known residue of 0.04 to 0.38% of its bytes, found in the transcribed source, so the pub
    
[^117]: 通过来自暗知识的模型无关潜在安全信号保障大语言模型安全

    Safeguarding LLMs via Model-Agnostic Latent Safety Signals from Dark Knowledge

    [https://arxiv.org/abs/2610.07532](https://arxiv.org/abs/2610.07532)

    提出LADE方法，通过对比有害与良性查询，从首token输出分布的暗知识中提取模型无关的潜在安全信号，在解码阶段实现安全防御，同时避免安全性与过度拒绝之间的权衡，并能跨架构泛化。

    

    大语言模型（LLM）发展迅速，引发了人们对其安全性日益增长的关注。近期工作提出了检测和防御攻击的方法，包括利用模型隐藏状态的解码阶段防御。然而，现有的解码阶段防御存在两个局限性。首先，它们在安全性与过度拒绝之间存在权衡，即加强安全性会降低模型对良性查询的有用性。其次，许多这些方法依赖于内部隐藏状态，因此仅限于特定架构，带来大量开销且在不同模型间的泛化能力有限。为解决这些限制，我们提出了 LADE（Latent Safety Signals for Defense），该方法通过在首token输出概率分布中，对比有害查询与良性查询，从暗知识（即输出概率分布中超出argmax所携带的信息）中提取潜在安全信号……

    arXiv:2610.07532v1 Announce Type: cross  Abstract: LLMs have advanced rapidly, raising growing concerns about their safety. Recent work has proposed approaches to detect and defend against attacks including defenses at decoding stage that leverage models' hidden states. However, existing decoding-stage defenses suffer from two limitations. First, they introduce a trade-off between safety and over-refusal, where strengthening safety degrades the model's helpfulness on benign queries. Second, many of these methods rely on internal hidden states and are thus restricted to specific architectures, incurring substantial overhead and limited generalization across models. To address these limitations, we introduce LADE (Latent Safety Signals for Defense), which leverages latent safety signals extracted by contrasting harmful and benign queries from dark knowledge (i.e., information carried by the output probability distribution beyond its argmax) in the first-token output probability distribut
    
[^118]: ℓBISG：利用特定群体姓名列表进行种族和民族的概率预测

    $\ell$BISG: Probabilistic Race and Ethnicity Prediction Using Group-Specific Name Lists

    [https://arxiv.org/abs/2610.06273](https://arxiv.org/abs/2610.06273)

    提出改进方法ℓBISG，用特定群体姓名列表（可由大语言模型生成）替代传统BISG所需的姓名频率数据，结合地理信息实现种族和民族归属概率的校准预测，即使在没有普查频率数据或地理数据的情况下也能应用。

    

    统计学上有效的种族和民族差异估计通常需要从姓名和地理位置推断群体归属概率。标准方法是贝叶斯改进姓氏地理编码，该方法需要每个姓名的群体频率数据。这类数据可从美国人口普查局获得，但仅限于常见姓名和有限的种族类别，对许多族裔群体则无法获得，且在美国之外极少存在。我们提出列表驱动的BISG方法（ℓBISG），用特定群体的姓名列表替代这些频率数据，并将姓名列表与地理流行度数据相结合，以获得经过校准的群体归属概率。这些姓名列表可以从不同来源汇编，或由大型语言模型（LLMs）生成，尽管可能存在未知的偏差。在满足关于姓名列表的一定条件下，ℓBISG还可以在没有地理流行度数据的情况下应用。使用美国数据进行的实证验证（原文摘要在此处截断）。

    arXiv:2610.06273v2 Announce Type: replace  Abstract: Statistically valid estimation of racial and ethnic disparities often requires inferring group membership probabilities from names and geographic locations. The standard method, Bayesian Improved Surname Geocoding (BISG), requires group frequencies for each name. Such data are available from the U.S. Census Bureau for common names and limited racial categories, but are unavailable for many ethnic groups and rarely exist outside the United States. We propose list-powered BISG ($\ell$BISG), in which we replace these frequencies with group-specific name lists and combine the lists with geographic prevalence data to obtain calibrated group membership probabilities. These lists can be compiled from different sources or generated by large language models (LLMs), despite potentially unknown biases. Under certain conditions on the name lists, $\ell$BISG can also be applied without geographic prevalence data. Empirical validation using U.S. v
    
[^119]: 用任务算子捕捉上下文学习动态

    Capturing In-Context Learning Dynamics with Task Operators

    [https://arxiv.org/abs/2610.01054](https://arxiv.org/abs/2610.01054)

    提出任务算子（TO）方法，通过将注意力头输出的仿射变换解析地重放到注意力输出投影中，从而无需处理完整示例即可高效模拟上下文学习，并揭示其内在机制。

    

    上下文学习（ICL）使语言模型能够仅凭示例演示执行新任务，而无需更新权重。然而，每次ICL推理都需要处理完整的示例集合，导致部署效率低下，且ICL的作用机制尚未被完全理解。先前的工作将ICL压缩为从特定层或位置提取的固定激活向量，但这些与输入无关的干预方法在复杂任务上会失效，因为这类任务的输出取决于与输入之间的细粒度交互。通过分析ICL的前向传递过程，我们发现每个注意力头的输出是其上下文掩码对应输出的仿射变换，且该变换的参数在给定任务的不同样本间具有经验上的稳定性。基于这一发现，我们提出了任务算子（Task Operator, TO），它将这一变换作为解析推导出的更新重放到注意力输出投影中。在词汇类、算法类……（原文摘要截断）

    arXiv:2610.01054v1 Announce Type: cross  Abstract: In-context learning (ICL) enables language models to perform new tasks from demonstrations without weight updates. However, every ICL inference requires processing the full set of examples, resulting in inefficient deployments, and how ICL works mechanistically is not fully understood. Prior work compresses ICL into fixed activation vectors extracted from specific layers or positions, but these input-independent interventions fail on complex tasks where the output depends on fine-grained interactions with the input. By analyzing the ICL forward pass, we show that each attention head's output is an affine transformation of its context-masked counterpart, and that the parameters of this transformation are empirically stable across samples for a given task. Building on this, we introduce Task Operator (TO), which replays this transformation as an analytically derived update to the attention output projection. Across lexical, algorithmic, 
    
[^120]: 早期问题的计算能否帮助大语言模型解决新问题？

    Can Computation from Earlier Problems Help LLMs Solve New Ones?

    [https://arxiv.org/abs/2609.39394](https://arxiv.org/abs/2609.39394)

    提出 STAIR 方法，通过固定存储库复用早期响应的键值信息并仅训练 12,288 个参数，就能让大语言模型在多轮对话中利用先前问题的计算来提升解决新问题的准确率。

    

    大语言模型经常在同一对话中解决相互独立的问题。那么，来自较早问题的计算能否帮助它们解决新问题？为了回答这个问题，我们首先进行了初步实验，结果表明保留的历史记录既可能提高也可能降低后续轮次的准确率，即使是在同一领域内也是如此。为了理解这些影响，我们采用受控重放方法来隔离特定于每个“问题-历史”配对的内部状态变化。结果显示，在不同的历史条件下，这些变化在当前问题之间保持了相似的关系。为了在保留历史的情况下改进推理，我们提出了 STAIR（Stale-Token Attention for Inter-query Reuse，面向跨查询重用的陈旧令牌注意力机制）。STAIR 将较早响应生成过程中的键和值捕获到一个固定存储库中，并学习在提示处理期间当前查询读取该存储库时对其进行重定向。基础模型保持完全冻结，仅训练 12,288 个参数。在三个 Qwen 模型和四个基准测试上，STAIR 提升了平均准确率（原文在此处截断）。

    arXiv:2609.39394v1 Announce Type: new  Abstract: Large language models often solve independent problems in the same conversation. Can computation from earlier problems help them solve new ones? To answer this question, we first conduct preliminary experiments showing that retained history can raise or lower later-turn accuracy, even within the same domain. To understand these effects, we use controlled replay to isolate internal state changes specific to each problem-history pairing. Across different histories, these changes preserve similar relationships among current problems. To improve reasoning under retained history, we introduce STAIR (Stale-Token Attention for Inter-query Reuse). STAIR captures keys and values from earlier response generation in a fixed bank. It learns to redirect current queries when they read this bank during prompt processing. The base model remains frozen; only 12,288 parameters are trained. Across three Qwen models and four benchmarks, STAIR improves avera
    
[^121]: 可信AI审稿人缺失的一块拼图：从修辞性稳健性基准测试到SciCore审稿

    A Missing Piece for Trustworthy AI Reviewers: From Benchmarking Rhetorical Robustness to SciCore Review

    [https://arxiv.org/abs/2609.39027](https://arxiv.org/abs/2609.39027)

    该论文提出“修辞性稳健性”概念并构建包含1,260个稿件版本的RobustReview基准，揭示了AI审稿人“虚假稳健性”的问题，进而提出双分支审稿模型SciCore，通过结合全稿评判与基于提取结构化科学内容的评判来提升审稿的可信度。

    

    AI审稿人可能会对以不同措辞报告相同科学内容的稿件给出不同的评判，这可能导致修辞性优化得到奖励，而非科学上的真正改进。我们将“修辞性稳健性”表述为一项联合要求：即在保持内容不变的改写下保持评判稳定，同时在不同论文之间保持区分能力。我们提出了RobustReview，这是一个受控的全稿基准，包含1,260个稿件版本，并评估了30种审稿人配置。该基准揭示了“虚假稳健性”现象，即对改写的敏感性虽低，但在不同论文之间却出现评分崩塌；同时表明，与人类评审的对齐程度和修辞性稳健性对审稿人的排序并不一致。此外，所评估的以内容为中心的提示协议并不能在各种骨干模型上一致地提升稳健性。基于这些发现，我们提出了SciCore，这是一种双分支审稿模型，它将基于全稿的评判与基于提取出的结构化科学内容（原文摘要在此处截断）的评判进行综合。

    arXiv:2609.39027v1 Announce Type: new  Abstract: AI reviewers can assign different judgments to manuscripts that report the same science in different wording, potentially rewarding rhetorical optimization over scientific improvement. We formulate Rhetorical Robustness as the joint requirement of stability across content-preserving rewrites and discrimination across papers. We introduce RobustReview, a controlled full-manuscript benchmark with 1,260 manuscript versions, and evaluate 30 reviewer configurations. The benchmark reveals false robustness, where low rewrite sensitivity coincides with score collapse across papers, and shows that human alignment and rhetorical robustness rank reviewers differently. Moreover, the evaluated content-focused prompting protocol does not consistently improve robustness across backbones. Motivated by these findings, we introduce SciCore, a dual-branch reviewer that averages a full-manuscript judgment with a judgment based on an extracted, structured sc
    
[^122]: STEPQuant：Delta规则循环状态量化中误差何时何地产生影响

    STEPQuant: When and Where Errors Matter in Delta-Rule Recurrent State Quantization

    [https://arxiv.org/abs/2609.38169](https://arxiv.org/abs/2609.38169)

    STEPQuant揭示了Delta规则循环状态量化误差在时间（记忆寿命）和空间（键行/值列）两个维度上的差异化影响，并提出一种时空后训练量化框架，根据误差大小与记忆寿命自适应分配精度，从而在低比特量化下保持模型精度。

    

    线性注意力机制用固定大小的循环状态取代了不断增长的KV缓存，但这些持久状态在并发服务场景下可能成为显著的内存瓶颈。直接将循环状态量化为低精度往往会导致严重的精度下降，因为量化误差会在连续的状态更新中不断传播。我们发现这些误差的影响取决于两个互补的维度：在时间维度上，长期存活的记忆中的误差会在多个解码步骤中持续存在；在空间维度上，不同键行中的误差对模型输出的影响各不相同，而状态幅值在行和列两个方向上都存在显著差异。基于这些观察，我们提出了STEPQuant，一个面向Delta规则循环状态的时空后训练量化框架。STEPQuant根据误差大小和记忆寿命来分配精度，并基于状态分布联合拟合键行与值列的缩放因子。

    arXiv:2609.38169v1 Announce Type: cross  Abstract: Linear attention replaces growing KV caches with fixed-size recurrent states, yet these persistent states can become a substantial memory bottleneck under concurrent serving. Directly quantizing recurrent states to low precision often leads to severe accuracy degradation, as quantization errors propagate through successive state updates. We discover that the impact of these errors depends on two complementary dimensions: temporally, errors in long-lived memory can persist across many decoding steps; spatially, errors in different key rows affect model outputs differently, while state magnitudes vary substantially along both rows and columns. Motivated by these observations, we propose STEPQuant, a spatial-temporal post-training quantization framework for Delta-rule recurrent states. STEPQuant allocates precision according to error magnitude and memory lifetime, and jointly fits key-row and value-column scales based on state distributio
    
[^123]: 端口哈密顿潜空间深思：缓解测试时计算扩展中的深思漂移悬崖

    Port-Hamiltonian Latent Deliberation: Mitigating the Deliberation Drift Cliff in Test-Time Compute Scaling

    [https://arxiv.org/abs/2609.37351](https://arxiv.org/abs/2609.37351)

    该论文发现了测试时潜空间深思中随思考深度增加而推理崩溃的“深思漂移悬崖”现象，并提出端口哈密顿潜空间深思框架，以同时兼顾表达能力、Lyapunov稳定性与计算效率，从而在深层思考步数下有效抑制漂移、缓解推理性能崩溃。

    

    测试时计算扩展已成为先进机器推理的基石，然而在连续潜表示空间中直接执行迭代深思会暴露出一种灾难性病理现象：深思漂移悬崖。尽管无约束的循环潜空间模型在较短的推理步数（K ≤ 4）下能获得初步的推理收益，但当外推至更深的思考步数（K ≥ 16）时，其推理能力会发生崩溃，在标准逻辑基准测试中性能下降22%至62%。我们通过22轮实证与理论研究，解决了测试时潜空间推理中表达能力、Lyapunov稳定性与计算效率之间的三难困境。我们证明，严格保守的标量势梯度流能够抑制长程漂移（悬崖仅为3.40%），但会将峰值推理准确率瓶颈在32.73%；而无约束的旋转流虽实现了高符号表达能力（82.33%），却会遭受高达36.87%的严重漂移。

    arXiv:2609.37351v1 Announce Type: cross  Abstract: Test-time compute scaling has emerged as a cornerstone of advanced machine reasoning, yet performing iterative deliberation directly within continuous latent representation spaces reveals a catastrophic pathology: the Deliberation Drift Cliff. While unconstrained recurrent latent models achieve initial reasoning gains at short horizons (K <= 4), their reasoning collapses when extrapolated to deeper thinking steps (K >= 16), dropping by 22% to 62% across standard logical benchmarks. We resolve the trilemma among expressivity, Lyapunov stability, and computational efficiency in test-time latent reasoning through a 22-round empirical and theoretical investigation. We demonstrate that strictly conservative scalar potential gradient flows suppress long-range drift (cliff 3.40%) but bottleneck peak reasoning accuracy at 32.73%, whereas unconstrained rotational flows achieve high symbolic expressivity (82.33%) but suffer a severe 36.87% drift
    
[^124]: 用于干预的梯度与用于检测的激活值：语言模型中的定向特征学习

    Gradients for Interventions and Activation Values for Detection: Targeted Feature Learning in Language Models

    [https://arxiv.org/abs/2609.32355](https://arxiv.org/abs/2609.32355)

    该论文提出“定向特征学习”框架，为预定义概念构建单一特征，并系统比较了三种模型信号（激活值、激活梯度、参数梯度）与两种估计器组合而成的六种方法，其中CAA和GRADIEND为已有方法，另外四种为新提出的方法。

    

    模型内部特征可以通过两种方式进行研究：一是识别特定概念的能力，二是被操纵时（例如通过引导或权重编辑）产生的因果效应。特征学习的一种主流方法是稀疏自编码器（SAE），它学习广泛的特征字典，其与特定概念的关系通常是事后确定的。然而，许多可解释性问题恰恰相反，是假设驱动的，涉及预先指定的概念。我们将这一设定称为定向特征学习，即为这样一个预定义概念构建单一特征。我们提出了一个受控比较，涵盖三种模型信号（激活值、激活梯度和参数梯度）和两种估计器（对比均值和可学习的一维编码器-解码器），由此产生六种定向方法，其中CAA和GRADIEND是已有方法，另外四种新方法覆盖了其余的组合。

    arXiv:2609.32355v2 Announce Type: replace  Abstract: Model-internal features can be studied through both their ability to identify a specified concept and their causal effect when manipulated, e.g., through steering or weight editing. A prominent approach to feature learning is Sparse Autoencoders (SAEs), which learn broad feature dictionaries whose relation to particular concepts is typically identified post hoc. However, many interpretability questions are instead hypothesis-driven and concern a concept specified in advance. We study this setting as targeted feature learning, where a single feature is constructed for such a predefined concept. We present a controlled comparison across three model signals (activation values, activation gradients, and parameter gradients) and two estimators (contrastive mean and a learned one-dimensional encoder-decoder), yielding six targeted methods, with CAA and GRADIEND as existing instances and four new methods covering the remaining combinations.
    
[^125]: Harness-Zero：通过“智能体即框架”实现框架蒸馏

    Harness-Zero: Harness Distillation via Agent-as-Harness

    [https://arxiv.org/abs/2609.24974](https://arxiv.org/abs/2609.24974)

    提出 Harness-Zero 方法，通过“智能体即框架”将领域或实例优化的框架所诱导的行为蒸馏进模型权重，使框架带来的性能提升在单一固定的目标框架下得以保留。

    

    智能体框架是调节模型与环境交互的外部系统，能够显著提升智能体的性能，但其收益在部署时仍依赖于所使用的框架。由于最佳框架因领域、实例和模型而异，通用智能体要么只能接受次优的共享框架，要么需要在不断增长的专用框架集合中进行路由选择。因此，我们研究智能体框架蒸馏：使用针对领域或实例优化的框架作为训练时的指导，并将其诱导的行为迁移到模型权重中，使其收益在单一固定的目标框架下得以保留。挑战在于，两个框架在动作空间和可用信息上存在差异，因此来自优化框架的指导无法直接作为目标框架的监督信号。我们提出了 Harness-Zero，通过“智能体即框架”实现框架蒸馏。在优化框架的指导下，一个框架……（原文摘要在此处截断）

    arXiv:2609.24974v1 Announce Type: cross  Abstract: Agent harnesses, the external systems that mediate model-environment interaction, can substantially improve agent performance, but their gains remain tied to the harness at deployment. Because the best harness varies across domains, instances, and models, a general-purpose agent must either settle for a suboptimal shared harness or route among an ever-growing set of specialized ones. We therefore study agent harness distillation: using a domain- or instance-optimized harness as training-time guidance and transferring the behaviors it induces into model weights, so that its gains survive under a single fixed target harness. The challenge is that the two harnesses differ in action space and available information, so guidance from the optimized harness cannot serve directly as supervision for the target one. We introduce Harness-Zero, which enables harness distillation through agent-as-harness. Guided by the optimized harness, a harnessin
    
[^126]: Swiss-Knife：解码时可重构的外部化多目标对齐框架

    Swiss-Knife: A Framework for Reconfigurable Externalised Multi-Objective Alignment at Decode Time

    [https://arxiv.org/abs/2609.22226](https://arxiv.org/abs/2609.22226)

    该论文提出Swiss-Knife框架，将解码时的多目标对齐规范变为可热插拔的运行时对象，并通过表示定理刻画了聚合算子族，证明成对聚合在对抗性奖励污染下比argmax更稳定。

    

    解码时对齐方法通过使用外部奖励对候选续写进行评分并选择得分最大者，来引导一个冻结的语言模型。我们认为这种共享设计只是更大空间中的一个退化点。我们提出Swiss-Knife，一个外部化多目标对齐框架，其中对齐规范是一等运行时对象：可热插拔的评分刀片、批归一化器、成对聚合算子和选择规则。六个公理刻画了可容许的聚合算子，并且我们证明了一个表示定理：每个满足这些公理的算子都具有形式 $R_i = \sum_{j \neq i} g((\mu_i - \mu_j)/s(\sigma_i,\sigma_j))$，这是一个双参数族，其中包含probit和logistic比较规则以及逐点argmax作为命名的坐标。在该族中，成对聚合在对抗性奖励污染下是Lipschitz稳定的，而argmax则不是，且候选批归一化……（原文摘要在此处截断）

    arXiv:2609.22226v1 Announce Type: new  Abstract: Decode-time alignment methods steer a frozen language model by scoring candidate continuations with an external reward and selecting the maximiser. We argue that this shared design is a single degenerate point in a much larger space. We introduce Swiss-Knife, a framework for externalised multi-objective alignment in which the alignment specification is a first-class runtime object: hot-swappable scoring blades, a batch normaliser, a pairwise aggregation operator, and a selection rule. Six axioms characterise the admissible aggregation operators, and we prove a representation theorem: every operator satisfying them has the form $R_i = \sum_{j \neq i} g((\mu_i - \mu_j)/s(\sigma_i,\sigma_j))$, a two-parameter family containing probit and logistic comparison rules and pointwise argmax as named coordinates. Within it, pairwise aggregation is Lipschitz-stable under adversarial reward contamination while argmax is not, and Candidate-Batch Norma
    
[^127]: 缺失的补充：面向编码代理的状态条件化最小充分证据

    The Missing Complement: State-Conditioned Minimal Sufficient Evidence for Coding Agents

    [https://arxiv.org/abs/2609.20050](https://arxiv.org/abs/2609.20050)

    该论文提出了状态条件化最小充分证据恢复这一新问题并构建了SERBench基准，同时提出MSS-Complement方法，将证据获取从排序转变为集合构建，为编码代理的决策恢复紧凑且充分的证据组合。

    

    一个处理问题进行到一半的编码代理已经读过检索器排名最高的很多内容。相关性是按段落评分的，但充分性属于集合层面：一个排序器可能用某个所需事实的多个变体填满其预算，却使决策仍然缺乏支持。我们提出了状态条件化最小充分证据恢复问题：给定一个捕获的代理状态，恢复一个紧凑的证据组合，以提供其下一个决策仍然缺乏的支持。SERBench在来自45个代码仓库的500个保留状态上对此进行测量，记录代理已经看到的内容，并且只对覆盖当前决策被标注所需的每一个事实的集合给予认可。MSS-Complement将证据获取视为集合构建而非排序。三次语义调用提出一个联合充分的集合，搜索其缺失的内容，并在6,144个token内返回4-8个完整的源单元。一个仅在校准数据上固定的配置，为其中73.0%的状态恢复了完整的证据集合。

    arXiv:2609.20050v1 Announce Type: cross  Abstract: A coding agent halfway through an issue has already read much of what a retriever ranks highest. Relevance is scored per passage, but sufficiency belongs to the set: a ranker can fill its budget with variants of one required fact and leave the decision unsupported. We formulate state-conditioned minimal sufficient evidence recovery: given a captured agent state, recover a compact evidence combination that supplies the support its next decision still lacks. SERBench measures this on 500 held-out states from 45 repositories, recording what the agent has seen and crediting only sets that cover every fact the current decision was annotated to require. MSS-Complement treats acquisition as set construction, not ranking. Three semantic calls propose a jointly sufficient set, search for what it lacks, and return 4-8 intact source units within 6,144 tokens. One configuration, fixed on calibration data, recovers a complete set for 73.0% of those
    
[^128]: 停止去除停用词：一项沿袭的预处理默认设置如何扭曲法律文本即数据研究

    Stop Removing Stopwords: How an Inherited Preprocessing Default Distorts Legal Text-as-Data

    [https://arxiv.org/abs/2609.19153](https://arxiv.org/abs/2609.19153)

    本研究通过穷尽式单词消融实验，首次直接针对下游分类目标验证停用词去除这一沿袭自信息检索时代、从未被验证过的预处理默认设置，揭示其可能扭曲实证法学中基于TF-IDF和线性分类器的文本数据分析结果。

    

    实证法学研究日益将司法文本视为数据，其中许多研究仍依赖稀疏、可解释的处理流程——TF-IDF特征和线性分类器——因为文本特征往往是研究对象本身，而不仅仅是实现预测的手段。然而，这些流程继承了一系列源自二十世纪中叶信息检索领域的预处理默认设置，这些设置从未针对分类准确度进行过验证，其中最根深蒂固的是停用词去除。本研究引入了一种穷尽式的单词消融方法，直接根据下游目标衡量预处理步骤的效果，并将其应用于停用词去除这一最难撼动的案例。通过将最高法院数据库的标签与Caselaw Access Project的判决意见文本相匹配，该研究考察了两个涵盖F1提升空间的双分类任务：意识形态方向（不去除停用词的基线F1约0.68）和宪法与非宪法法律类型（约0.92），涉及7,66

    arXiv:2609.19153v1 Announce Type: new  Abstract: Empirical legal scholarship increasingly treats judicial text as data, and much of it still runs on sparse, interpretable pipelines -- TF-IDF features and linear classifiers -- because the textual feature is often the object of study, not merely a means to a prediction. Yet these pipelines inherit a chain of preprocessing defaults from mid-century information retrieval that were never validated against classification accuracy, the most entrenched being stopword removal. This study introduces an exhaustive single-word ablation that measures a preprocessing step's effect directly against the downstream objective, and applies it to stopword removal as the hardest case to dislodge. Matching Supreme Court Database labels to Caselaw Access Project opinion texts, it examines two binary tasks that bracket F1 headroom, ideological direction (no-removal baseline F1 ~ 0.68) and constitutional versus non-constitutional law type (~ 0.92), across 7,66
    
[^129]: 我们的评分方式是否恰当？理解医学基准测试中的失败模式

    Are We Grading Properly? Understanding Failure Modes in Medical Benchmarks

    [https://arxiv.org/abs/2609.16023](https://arxiv.org/abs/2609.16023)

    该论文提出RIFT评分标准失败分类法，对两个医学临床基准测试进行系统检测，发现大量评分标准存在非原子性和错位僵化等缺陷，且这些缺陷并非表面问题，仅重写捆绑标准就能导致相同回答的分数变化高达15.9个百分点。

    

    医学评估正在从基于静态选项的提问转向具有开放式输出模式的真实临床场景。然而，简单地大规模评分成本高昂，基于评分标准的评估已成为主流的可扩展替代方案。我们探讨了当评分标准本身不够严密时会发生什么，以及这些缺陷是否能够被检测和纠正。我们将RIFT——一个全局评分标准失败分类法——应用于两个临床基准测试（HealthBench Professional和LiveMedBench），发现失败模式具有重要意义：在HealthBench Professional上，LLM评判者将29.6%的评估标准标记为非原子性，将65.4%标记为错位或僵化。随后，我们证明这些缺陷具有实质性影响，而不仅仅是表面问题。例如，将“以下至少一项/全部满足”形式的捆绑标准重写为等权重的子标准并对相同响应重新评分，会使受影响对话的分数发生高达15.9个百分点的变化。

    arXiv:2609.16023v1 Announce Type: new  Abstract: Medical evaluation is shifting from static option-based questioning to realistic clinical scenarios with open-ended output modes. Grading these at scale naively, however, is expensive, and rubric-based evaluation has become the dominant scalable alternative. We ask what happens when the rubrics themselves are not airtight, and whether such flaws can be detected and corrected. We apply RIFT, a global rubric failure taxonomy, to two clinical benchmarks (HealthBench Professional and LiveMedBench), and find failure modes are meaningful: on HealthBench Professional an LLM judge flags 29.6% of criteria as non-atomic and 65.4% as misaligned/rigid. Then, we show that these flaws are meaningful and not simply cosmetic. As an example, rewriting bundled criteria of the form "at least one of / all of the following" as equally weighted children and regrading identical responses shifts scores by up to 15.9 percentage points on affected conversations, 
    
[^130]: 智能体作为策略的机器人操作

    Agent as Policy for Robotic Manipulation

    [https://arxiv.org/abs/2609.12541](https://arxiv.org/abs/2609.12541)

    该论文提出Agent as Policy (AGP)方法，使通用智能体无需任何任务特定训练即可直接控制物理机器人完成从精细操作、动态运动到可变形物体处理等多种真实操作任务。

    

    我们证明了通用智能体可以在整个任务执行过程中直接驱动物理机器人，而无需任何针对特定任务或特定环境的训练。我们提出了智能体作为策略方法，将任务规划和执行都置于智能体的控制之下。给定一个任务和机器人接口，智能体能够解读视觉信息、编写可执行程序、发出运动指令，并根据物理执行结果修正自身动作。这将智能体的推理和编程能力带入了与物理世界的持续交互之中。我们在多个真实世界的操作任务上研究了AGP，涵盖精细操作、动态运动和可变形物体，包括从人类视频中学习装配、根据目标图像搭建积木、翻转骰子、定向投掷以及双臂协作折叠毛巾。AGP在三种积木搭建配置上分别取得了100%、100%和80%的成功率。

    arXiv:2609.12541v1 Announce Type: new  Abstract: We demonstrate that a general-purpose agent can directly drive a physical robot throughout task execution without any task-specific or environment-specific training. We introduce Agent as Policy (AGP), which places task planning and execution under the agent's control. Given a task and a robot interface, the agent interprets visual evidence, writes executable programs, issues motion commands, and revises its actions in response to physical outcomes. This brings the agent's reasoning and programming capabilities into continuous interaction with the physical world. We study AGP across multiple real-world manipulation tasks spanning precision manipulation, dynamic motions, and deformable objects. These include assembly from human videos, block construction from goal images, die reorientation, targeted throwing, and bimanual towel folding. AGP achieves success rates of 100%, 100%, and 80% on three block construction configurations. These fin
    
[^131]: EvoSkill注入：针对自进化智能体中自主技能生成与演化的红队测试

    EvoSkill Injection: Red-Teaming Autonomous Skill Generation and Evolution in Self-Evolving Agents

    [https://arxiv.org/abs/2608.30429](https://arxiv.org/abs/2608.30429)

    本文提出EvoSkill注入威胁模型和SARGE红队测试框架，揭示了自进化智能体在自主技能生成与演化过程中可能被注入并复用恶意能力这一新型安全风险，并构建了EvoSkillBench基准数据集加以评估。

    

    基于大语言模型（LLM）的智能体系统日益采用基于技能的架构，以减少重复推理成本并提升稳定、高效的任务执行能力。近期研究提出了自进化智能体，能够从过往经验中自主生成、优化和复用技能，从而实现能力的持续进化。然而，自主技能进化引入了一个新的攻击面，即恶意能力可以被生成、存储并作为合法技能被复用。在本文中，我们将EvoSkill注入定义为一种针对自进化智能体自主技能生成与演化流程的威胁模型。我们进一步提出了SARGE（针对自进化智能体中自主技能生成与演化的红队测试），这是一个通过迭代生成、升级和强化交互来评估该威胁模型的红队测试框架。为支持我们的框架，我们构建了EvoSkillBench，一个包含恶意交互的基准数据集……

    arXiv:2608.30429v1 Announce Type: new  Abstract: LLM-based agent systems increasingly adopt skill-based architectures to reduce repetitive reasoning costs and improve stable, efficient task execution. Recent studies propose self-evolving agents that autonomously generate, refine, and reuse skills from past experiences to enable continuous capability evolution. However, autonomous skill evolution introduces a new attack surface in which malicious capabilities are generated, stored, and reused as legitimate skills. In this paper, we define EvoSkill Injection as a threat model targeting the autonomous skill generation and evolution pipeline of self-evolving agents. We further propose SARGE (Red-teaming Autonomous Skill Generation and Evolution in self-evolving agents), a red-teaming framework for evaluating this threat model through iterative generation, escalation, and reinforcement interactions. To support our framework, we construct EvoSkillBench, a benchmark dataset of malicious inter
    
[^132]: 基准的平衡：面向基准多重性与任务条件评估的语义密度重加权

    Balance of Benchmarks: Semantic Density Reweighting for Benchmark Multiplicity and Task-Conditioned Evaluation

    [https://arxiv.org/abs/2608.30044](https://arxiv.org/abs/2608.30044)

    提出基准平衡方法，通过对基准分配逆密度语义权重来纠正基准密集区域的重复计算问题，并支持基于任务查询的条件化模型评估，在586个模型上的留出任务预测相关性从等权加权的0.049大幅提升至0.462。

    

    语言模型通常通过对基准列表中的分数进行等权平均来进行比较。这类列表通过公开发表而不断增长，缺乏显式的测量设计，因此等权加权使已发表基准的密度成为隐含的能力权重：基准密集的区域会被重复计算。我们提出了基准平衡方法，该方法对基准描述进行嵌入，并为每个基准分配逆密度语义权重。在公开的密度尺度下，相近的基准条目共享聚合影响力。在将异构分数统一到共同潜在尺度后，残差场利用相同的几何结构，使模型排名能够以任务查询为条件。这两个组成部分在实证上发挥着不同的作用。在包含586个模型和14个基准的数据快照上，BoB能够预测哪些模型在留出任务上表现出超出其一般能力的异常强势，画像相关性达到0.462，而等权加权下仅为0.049。

    arXiv:2608.30044v1 Announce Type: new  Abstract: Language models are commonly compared by averaging scores across a benchmark list with equal weight. Such lists grow through publication outside an explicit measurement design, so equal weighting turns the density of published benchmarks into an implicit capability weight: densely benchmarked regions count repeatedly. We introduce Balance of Benchmarks (BoB), which embeds benchmark descriptions and assigns each benchmark an inverse-density semantic weight. Nearby entries share aggregate influence at a disclosed density scale. After equating heterogeneous scores onto a common latent scale, a residual field uses the same geometry to condition model rankings on a task query. The two components serve distinct empirical roles. On a snapshot of 586 models and 14 benchmarks, BoB predicts which models are unusually strong on a held-out task beyond their general ability, reaching a profile correlation of 0.462 compared with 0.049 under equal weig
    
[^133]: NL2AGBench：面向AlphaGeometry的大语言模型自动形式化基准测试

    NL2AGBench: Benchmarking LLM Auto-Formalization for AlphaGeometry

    [https://arxiv.org/abs/2608.28481](https://arxiv.org/abs/2608.28481)

    该论文提出了NL2AGBench基准测试，用于评估大语言模型将英文几何问题自动形式化为AlphaGeometry兼容形式化表示的能力，从而解决神经符号几何系统中手动转换输入的可用性瓶颈。

    

    大语言模型（LLM）的最新进展展示了其在自然语言理解和数学推理方面的强大能力。然而，它们将非形式化数学问题转化为形式化表示的能力仍然缺乏充分探索。这一局限对于神经符号几何系统（如AlphaGeometry）尤为重要，因为其定理证明引擎需要以专门的领域特定语言（DSL）作为输入。尽管AlphaGeometry达到了接近国际数学奥林匹克（IMO）金牌选手的水平，但手动将自然语言问题转换为其形式化语法仍然是一个显著的可用性瓶颈。为了应对这一挑战，我们提出了自然语言到AlphaGeometry基准测试（NL2AGBench），用于评估大语言模型将英文几何问题翻译成与AlphaGeometry兼容的形式化表示的能力。NL2AGBench使用AlphaGeometry内部基于执行的验证来评估翻译质量……

    arXiv:2608.28481v1 Announce Type: new  Abstract: Recent advances in large language models (LLMs) have demonstrated strong capabilities in natural language understanding and mathematical reasoning. However, their ability to translate informal mathematical problems into formal representations remains underexplored. This limitation is particularly important for neuro-symbolic geometry systems such as AlphaGeometry, whose theorem-proving engine requires inputs in a specialized domain-specific language (DSL). Although AlphaGeometry achieves near-IMO gold-medalist performance, manually converting natural-language problems into its formal syntax remains a significant usability bottleneck. To address this challenge, we introduce the Natural Language to AlphaGeometry Benchmark (NL2AGBench), which evaluates LLMs in translating English geometry problems into AlphaGeometry-compatible formal representations. NL2AGBench uses execution-based verification within AlphaGeometry to assess translation qua
    
[^134]: MUSE：一个跨领域的科学问题、解决方案与理由全文知识库

    MUSE: A Full-Text Cross-Domain Knowledge Base of Scientific Problems, Solutions, and Rationales

    [https://arxiv.org/abs/2608.10974](https://arxiv.org/abs/2608.10974)

    该论文构建了MUSE——一个跨领域的全文科学“问题-解决方案-理由”三元组知识库（基于579个专家标注段落扩展出3.7万条有原文依据的三元组），并通过实验发现理由监督能够提升大语言模型解决复杂多约束科学问题的能力。

    

    arXiv:2608.10974v2 公告类型：替换 摘要：科学论文中包含细粒度的问题解决记录：作者会提及技术障碍以及用于解决这些障碍的方法，并常常附带说明为何选择这些方法的推理依据。我们提出了MUSE（Mining Underlying Scientific Explanations，挖掘潜在科学解释），这是一个包含科学“问题-解决方案-理由”（P-S-R）三元组的全文、多领域资源。我们整理了579个由专家标注的全文段落，并采用丰富的标注模式，涵盖显著的问题、解决方案与理由文本片段、solves与rationale_of链接以及概念共指关系。通过一个模块化的抽取流程，我们将此标注规模化，构建了一个包含37K条有原文依据的P-S-R三元组的高质量知识库。我们对该抽取系统的各个组件进行了评估，并开展了初步实验，训练一个以理由为监督信号的大语言模型用于科学问题求解。有趣的是，我们发现理由监督能够提升模型在复杂、多约束问题上的表现。

    arXiv:2608.10974v2 Announce Type: replace  Abstract: Scientific papers contain fine-grained records of problem solving: authors mention technical obstacles and methods that were used to address them, often along with reasoning on why those methods were chosen. We introduce MUSE (Mining Underlying Scientific Explanations), a full-text, multi-domain resource of scientific Problem-Solution-Rationale (P-S-R) triplets. We curate 579 expert-annotated full-text paragraphs, with a rich annotation schema covering salient problem, solution, and rationale spans, solves and rationale_of links and conceptual coreference. A modular extraction pipeline scales this annotation to build a high-quality knowledge base of 37K source-grounded P-S-R triplets. We evaluate the extraction components and include a preliminary experiment training a rationale-supervised LLM for scientific problem solving. Interestingly, we find that rationale supervision improves performance on complex, multi-constraint problems b
    
[^135]: 从转录到语义语料库分析：面向古代语言的无监督句子表示学习

    From transcription to semantic corpus analysis: unsupervised learning of sentence representations for ancient languages

    [https://arxiv.org/abs/2607.24542](https://arxiv.org/abs/2607.24542)

    本文提出两种完全无监督的方法（TSDAE 和无监督 SimCSE），仅用原始句子将面向古代语言的专用词元级语言模型转化为语料库专属的句子编码器，并在拉丁语和古希腊语教父文献中经专家验证的圣经文本复用识别任务上验证了其有效性。

    

    自动文本识别（ATR）如今为数字人文领域提供了大量非结构化、异构且常含噪声的古代语言文本。下游的语义分析——如文本复用识别、对齐和语义检索——依赖于句子嵌入，然而现有方法向古代语言的迁移效果不佳：通用多语言编码器表现欠佳，专用语言模型产生各向异性的表示空间，且标注的相似度数据十分稀缺。我们研究了两种完全无监督的策略——TSDAE 和无监督 SimCSE（以下简称 CSE）——它们仅利用原始句子，将专用的词元级语言模型适配为面向特定语料库的句子编码器。我们在拉丁语和古希腊语教父文献中的圣经文本复用任务上对这两种方法进行了验证：语料包含奥古斯丁、哲罗姆和阿塔纳修斯著作中 2,936 个经专家验证含有圣经复用的句子，并与《武加大译本》、《七十士译本》以及 Nestle-Aland 版本文本进行匹配。

    arXiv:2607.24542v2 Announce Type: replace  Abstract: Automatic Text Recognition (ATR) now supplies digital humanities with large volumes of unstruc- tured, heterogeneous, and often noisy text in ancient languages. Downstream semantic analyses- text reuse identification, alignment, and semantic search-rely on sentence embeddings, yet ex- isting methods transfer poorly to ancient languages: generic multilingual encoders underperform, specialized language models yield anisotropic representation spaces, and labeled similarity data is scarce. We study two fully unsupervised strategies-TSDAE and unsupervised SimCSE (hence- forth CSE)-that adapt a specialized token-level language model into a corpus-specific sentence encoder using only raw sentences. We validate them on biblical reuse in patristic literature in Latin and Ancient Greek: 2,936 sentences carrying expert-verified reuses in works of Augustine, Jerome, and Athanasius, matched against the Vulgate, the Septuagint, and the Nestle--Ala
    
[^136]: Tokengeist：智能体对话中的多轮归因追踪

    Tokengeist: Multi-Turn Attribution Tracing in Agentic Conversations

    [https://arxiv.org/abs/2607.22610](https://arxiv.org/abs/2607.22610)

    该论文提出了多轮上下文归因（MTCA）任务以及Tokengeist框架，通过将有向无环图上的递归遍历作为归因方式来恢复完整依赖路径，解决了现有单次归因方法无法捕捉多轮对话分层依赖结构的问题。

    

    arXiv:2607.22610v2 公告类型：替换 摘要：当一个模型在经过多轮对话和工具使用后给出回答时，监督者需要了解该回答依赖于哪些较早的对话轮次和中间步骤。归因是回答这一问题的自然基础，但现有的上下文归因方法以单次处理方式处理完整对话记录：它们只能恢复表面层面的依赖关系，而遗漏了多轮追踪中的分层结构——在这种结构中，助手的某一轮回复本身就是一个派生产物，它对较早的上下文进行了压缩或转换。我们提出了多轮上下文归因（MTCA）：给定模型回复中的一个目标片段，该任务要求跨轮次向后追踪归因，不仅要识别哪些先前的轮次直接相关，还要识别这些轮次本身如何依赖于更早的上下文。我们提出了Tokengeist，一个与归因方法无关且可扩展的框架，它通过将归因转化为对有向无环图的递归遍历来恢复完整的依赖路径。

    arXiv:2607.22610v2 Announce Type: replace  Abstract: When a model answers after several turns of dialogue and tool use, an overseer needs to know which earlier turns and intermediate steps it rests on. Attribution is the natural primitive for answering this, but existing context attribution methods process the full transcript in a single pass: they recover surface-level dependencies while missing the layered structure of multi-turn traces, in which an assistant turn is itself a derived artifact that condenses or transforms earlier context. We introduce multi-turn context attribution (MTCA): given a target span in a model response, the task of tracing attribution backward across turns to identify not only which prior turns were directly relevant, but also how those turns themselves depended on earlier context. We propose Tokengeist, an attribution-method-agnostic and scalable framework that recovers full dependency paths by casting attribution as a recursive traversal of a directed acyc
    
[^137]: 面向延迟与模型大小优化的大语言模型多目标结构化剪枝

    Multi-Objective Structured Pruning of LLMs for Latency and Model Size Optimization

    [https://arxiv.org/abs/2607.22583](https://arxiv.org/abs/2607.22583)

    该论文提出了一种硬件感知的多目标结构化剪枝框架，通过两阶段方法联合优化大语言模型的层数、注意力头和MLP维度，以同时最小化延迟和模型大小，从而实现边缘设备上的高效部署。

    

    arXiv:2607.22583v4 公告类型：替换 摘要：大语言模型（LLM）因其强大的推理和问答能力而得到了广泛应用。然而，由于严格的延迟、内存和能耗限制，将其部署到嵌入式和边缘计算环境中仍然充满挑战。其庞大的参数量和计算需求阻碍了在资源受限平台上的高效执行。尽管模型剪枝已成为在保持性能的同时缩减模型规模的一种可行方案，但联合优化层数、注意力头数以及多层感知机（MLP）维度仍然极为复杂。穷举探索这一组合设计空间的计算开销巨大，且往往导致局部最优或不稳定的配置。为解决这些局限，我们提出了一种硬件感知的多目标结构化剪枝框架。所提出的两阶段方法明确地以延迟和模型大小为目标，以实现高效部署（原文在此处截断）。

    arXiv:2607.22583v4 Announce Type: replace  Abstract: Large Language Models (LLMs) have achieved widespread adoption because of their strong reasoning and query-response capabilities. However, deploying them in embedded and edge computing environments remains challenging because of strict latency, memory, and energy constraints. Their large parameter counts and computational demands hinder efficient execution on resource-constrained platforms. Although model pruning has emerged as a viable solution for reducing scale while preserving performance, jointly optimizing layers, attention heads, and Multi-Layer Perceptron (MLP) dimensions remains highly complex. Exhaustively exploring this combined design space is computationally expensive and often leads to local optima or unstable configurations. To address these limitations, we propose a hardware-aware, multi-objective structured pruning framework. The proposed two-stage method explicitly targets latency and model size for efficient deploy
    
[^138]: 持久稀疏自编码器：在语言模型表示中学习特征特定的时间尺度

    Persistent Sparse Autoencoders: Learning Feature-Specific Timescales in Language Model Representations

    [https://arxiv.org/abs/2607.17117](https://arxiv.org/abs/2607.17117)

    该论文提出持久稀疏自编码器，通过为每个特征学习一个持久性系数，使稀疏自编码器能够仅凭重构目标从语言模型激活中自动学习特征特定的时间尺度，同时保持高质量的重构效果。

    

    稀疏自编码器（SAE）将语言模型的激活分解为稀疏特征，然而这些模型传统上对每个词元进行独立编码，无法揭示跨序列持续存在的信息。我们首先证明，时间持久性可以在标准SAE特征中自然涌现：当一个特征激活后，隐藏状态会保持与其方向对齐，且过去的激活有助于重构后续的隐藏状态。这种持续性在不用特征之间的差异很大。因此，我们提出了持久稀疏自编码器（Persistent SAEs），这是标准SAE的一种扩展，它为每个特征学习一个持久性系数，使模型能够仅通过重构任务学习特征特定的时间尺度。我们的实验表明，持久SAE在学习一系列时间尺度的同时保持了有竞争力的重构质量：短时间尺度（快速）特征保持局部可解释性，而长时间尺度（……（摘要在此处被截断）

    arXiv:2607.17117v2 Announce Type: replace-cross  Abstract: Sparse autoencoders (SAEs) decompose language model activations into sparse features, yet these models traditionally encode each token independently, failing to expose information that persists across a sequence. We first show that temporal persistence can naturally emerge in standard SAE features: after a feature activates, the hidden state remains aligned with its direction, and past activations help reconstruct later hidden states. How long this lasts varies widely across features. We therefore introduce Persistent Sparse Autoencoders (Persistent SAEs), an extension of standard SAEs that learns a persistence coefficient for each feature, allowing the model to learn feature-specific timescales from reconstruction alone. Our experiments show that Persistent SAEs retain competitive reconstruction quality while learning a spectrum of timescales: short-timescale (fast) features stay locally interpretable, whereas long-timescale (
    
[^139]: 划分、提示、聚合：语言模型中的统计自洽性

    Partition, Prompt, Aggregate: Statistical Self-Consistency in Language Models

    [https://arxiv.org/abs/2607.15277](https://arxiv.org/abs/2607.15277)

    该论文提出“划分—提示—聚合”的评估框架，利用二叉树递归划分总体，检验大语言模型给出的条件估计是否满足全概率定律所要求的统计自洽性。

    

    大型语言模型（LLM）常被用于执行某种形式的条件推断，而上下文学习是实现这一目标的一种隐式方式：提示词指定一个上下文，模型的输出则被视为相应条件分布的估计。如果这种解释成立，那么LLM的估计应当满足基本的概率恒等式。特别是，全概率定律断言：在总体的任何有效划分下，以先验加权的条件分布能够聚合为总体层面的边缘分布。在本工作中，我们研究了LLM的估计在多大程度上遵循这一自洽性原则。我们使用二叉树作为评估框架，将一个总体递归地划分为粒度越来越细的子总体；然后我们用言语化的子总体描述作为上下文来提示LLM，将所得到的估计重新聚合为总体层面的估计，并对它们进行比较……

    arXiv:2607.15277v2 Announce Type: replace  Abstract: Large language models (LLMs) are often used to perform a form of conditional inference. In-context learning is one implicit way to do this. The prompt specifies a context and the model's output is treated as an estimate of the corresponding conditional distribution. If this interpretation holds, then LLM estimates should satisfy basic probabilistic identities. In particular, the law of total probability asserts that prior-weighted conditional distributions aggregate into population-level marginals over any valid partition of the population. In this work, we investigate to what extent LLM estimates adhere to this self-consistency principle. We use binary trees as an evaluation scaffold to recursively partition a population into increasingly fine-grained subpopulations. We then prompt LLMs with verbalized subpopulation descriptions in context, aggregate the resulting estimates back into population-level estimates, and compare them acro
    
[^140]: SPARK：面向忠实检索增强生成的自适应参数化知识参与

    SPARK: Adaptive Parametric Knowledge Participation for Faithful Retrieval-Augmented Generation

    [https://arxiv.org/abs/2606.27786](https://arxiv.org/abs/2606.27786)

    SPARK框架将参数化知识控制形式化为潜在参与策略学习问题，通过轻量级参与策略动态调节生成过程中参数化知识的贡献，从而解决RAG中检索上下文与模型内部知识的冲突，提升生成忠实性，同时避免直接编辑模型参数可能带来的能力损害。

    

    检索增强生成（RAG）通过将外部知识融入生成过程来增强大型语言模型（LLM）的忠实性。然而，检索到的上下文与参数化知识之间的冲突可能导致LLM依赖不一致的内部知识，阻碍其忠实利用检索证据的能力。现有方法通过编辑参数化知识来缓解这一问题，以解决此类冲突。然而，定位和修改编码在模型参数中的知识十分困难，并且可能无意中损害骨干LLM的能力。在本文中，我们提出了SPARK，这是一个新颖的框架，它将参数化知识控制形式化为潜在参与策略学习问题，使LLM能够在生成过程中自适应地调节参数化知识的贡献。具体而言，SPARK引入了一个轻量级的参与策略，动态调制……

    arXiv:2606.27786v2 Announce Type: replace-cross  Abstract: Retrieval-augmented generation (RAG) enhances the faithfulness of large language models (LLMs) by incorporating external knowledge into the generation process. However, conflicts between retrieved context and parametric knowledge can cause LLMs to rely on inconsistent internal knowledge, hindering their ability to faithfully leverage retrieved evidence. Existing approaches mitigate this issue by editing parametric knowledge to resolve such conflicts. However, locating and modifying knowledge encoded in model parameters is challenging and may inadvertently compromise the backbone LLM's capabilities. In this paper, we introduce SPARK, a novel framework that formulates parametric knowledge control as a latent participation policy learning problem, enabling LLMs to adaptively regulate the contribution of parametric knowledge during generation. Specifically, SPARK introduces a lightweight participation policy that dynamically modula
    
[^141]: Polar：一个用于评估大语言模型政治偏见的基准

    Polar: A Benchmark for Evaluating Political Bias in LLMs

    [https://arxiv.org/abs/2606.12922](https://arxiv.org/abs/2606.12922)

    提出了Polar基准，通过选项级似然概率在美韩两种政治语境和八个议题类别上测量38个大语言模型的政治偏见，发现所有模型在美国内容上均偏向左翼进步立场，且展示语言本身即可改变测得的偏见结果。

    

    大语言模型（LLM）中的政治偏见日益重要，但在不同的政治和语言环境下难以进行可复现的测量。我们提出了Polar，一个包含4,026个实例的多项选择基准，它通过选项级别的似然概率而非基于提示的生成方式来测量政治偏见。Polar涵盖源自宣言项目的两个意识形态维度和八个议题类别，并在美国和韩国的政治语境中并行评估模型。在38个大语言模型上的实验显示，测得的偏见随政治语境、议题类别、模型组和展示语言的变化而系统性变化。所有模型在美国政治内容上都倾向于左翼进步立场，但在韩国内容上则表现出更为居中和混合的模式。翻译实验进一步表明，仅展示语言本身就能改变测得的偏见。这些发现凸显了开展多语言、跨语境政治偏见评估的必要性。

    arXiv:2606.12922v2 Announce Type: replace  Abstract: Political bias in large language models (LLMs) is increasingly significant, but difficult to measure reproducibly across political and linguistic contexts. We introduce Polar, a 4,026-instance multiple-choice benchmark that measures political bias through option-level likelihoods rather than prompt-based generation. Polar covers two ideological axes and eight issue categories derived from the Manifesto Project, and evaluates models in parallel across U.S. and South Korean political contexts. Across 38 LLMs, measured bias varies systematically with political context, issue category, model group, and presentation language. All models lean left-progressive on U.S. political content, but show more centered and mixed patterns on South Korean content. Translation experiments further show that presentation language alone can shift measured bias. These findings highlight the need for multilingual and cross-contextual evaluation of political 
    
[^142]: 支持性令牌揭示：加速扩散语言模型解码

    Supportive Token Revealing for Fast Diffusion Language Model Decoding

    [https://arxiv.org/abs/2606.04236](https://arxiv.org/abs/2606.04236)

    提出无需训练的AXON模块，通过监控剩余的不确定掩码令牌并在需要额外上下文时揭示支持性令牌，从而加速扩散语言模型的并行解码并缓解质量与延迟之间的权衡。

    

    离散扩散语言模型可以通过并行更新多个被掩码的位置来高效生成文本，但这种并行性引入了质量与延迟之间的权衡。激进的解码可能过早地确定相互依赖的令牌，而保守的解码则需要许多去噪步骤。现有方法通过基于置信度或依赖性准则判断哪些令牌可以安全揭示来缓解这一矛盾。然而，避免不安全的提交并不一定能使剩余的掩码序列易于解码，因为不确定的令牌可能依赖于其他被掩码的令牌，这成为去噪步骤的瓶颈。我们提出了AXON，这是一个无需训练的模块，可以叠加在现有的扩散语言模型并行解码策略之上。AXON并不替换基础解码器，而是监控剩余的不确定掩码令牌，仅在其当前状态表明需要额外上下文时才进行干预。

    arXiv:2606.04236v2 Announce Type: replace-cross  Abstract: Discrete diffusion language models can generate text efficiently by updating multiple masked positions in parallel, but this parallelism introduces a quality-latency trade-off. Aggressive decoding may commit mutually dependent tokens too early, while conservative decoding requires many denoising steps. Existing methods address this tension by deciding which tokens are safe to reveal using confidence or dependency criteria. However, avoiding unsafe commits does not necessarily make the remaining masked sequence easy to decode, since uncertain tokens may depend on masked tokens, creating a bottleneck for denoising steps. We propose AXON, a training-free module that can be added on top of existing parallel decoding strategies for diffusion language models. Rather than replacing the base decoder, AXON monitors the remaining uncertain masked tokens and intervenes only when their current state suggests that additional context is need
    
[^143]: VESTA：基于统计工具智能体的可视化探索

    VESTA: Visual Exploration with Statistical Tool Agents

    [https://arxiv.org/abs/2606.00384](https://arxiv.org/abs/2606.00384)

    VESTA是一个为视觉语言模型配备动态增长工具包的新框架，通过数据变换、假设驱动的可视化和统计检验来探索数据并改进统计建模，克服了以往仅依赖迭代批评的智能体系统在复杂建模任务上的局限。

    

    将定量模型拟合到数据是科学工作流程中的核心步骤，但这一步骤仍然是自动化程度最低的环节之一。近年来，基于智能体的系统利用语言模型和视觉语言模型（VLM）来提出统计模型，但这些系统在具有挑战性的建模任务上表现不佳。为了解决这些局限性，我们提出了VESTA（Visual Exploration with Statistical Tool Agents，基于统计工具智能体的可视化探索），这是一个为VLM配备动态增长工具包的框架，通过数据变换、假设驱动的可视化以及统计检验来引导模型的改进。与以往仅依赖迭代式批评反馈的系统不同，VESTA通过选择或创建可复用的诊断工具，在模型改进之前和改进过程中对数据进行探索。我们使用三种工具包配置对VESTA进行评估：无工具、静态的专家编写工具，以及动态的模型自建工具。为支持这一评估，我们引入了DAWN（用于自动化工作流程和数值……的数据集）——摘要在此处截断。

    arXiv:2606.00384v3 Announce Type: replace  Abstract: Fitting quantitative models to data is a central step in scientific workflows, yet it remains one of the least automated. Recent agent-based systems leverage language and vision-language models (VLMs) to propose statistical models, but these systems struggle on challenging modeling tasks. To address these limitations, we introduce VESTA (Visual Exploration with Statistical Tool Agents), a framework that equips VLMs with a dynamically growing toolkit to guide model refinement through data transformations, hypothesis-driven visualizations, and statistical tests. Unlike prior systems that rely on iterative critique alone, VESTA explores data before and during refinement by selecting or creating reusable diagnostic tools. We evaluate VESTA using three toolkit configurations: no tools, static expert-written tools, and dynamic model-written tools. To support this evaluation, we introduce DAWN (Dataset for Automated Workflows and Numerical 
    
[^144]: MosaicLeaks：深度研究智能体开放式查询中的隐私风险

    MosaicLeaks: Privacy Risks in Querying-in-the-Open for Deep Research Agents

    [https://arxiv.org/abs/2605.30727](https://arxiv.org/abs/2605.30727)

    该论文提出MosaicLeaks基准，首次系统揭示了深度研究智能体在结合私有文档与外部网络查询时，会因“马赛克效应”在多个层面泄露本地敏感信息，且各类模型均存在此问题、现有隐私提示防护效果有限。

    

    深度研究智能体日益将私有的本地文档与网页检索等外部工具相结合，这带来了一种隐私风险：智能体发出的外部查询可能会泄露其本地上下文中的敏感信息。这种风险会被“马赛克效应”放大，即单个查询可能看起来无害，但汇总起来却能揭示隐私信息。我们提出了MosaicLeaks，一个包含1,001个多跳深度研究任务的基准，这些任务将私有企业文档与公共网页语料库串联起来，迫使智能体进行依赖于本地信息的外部查询。我们使用一个仅观察智能体外部查询的对抗性大语言模型来评估信息泄露情况，该模型尝试在三个层面推断私有信息：智能体的研究意图、特定私有问题的答案，以及关于企业文档的可验证声明。我们发现，不同家族和不同规模的模型在所有三个层面都频繁发生泄露，而零样本隐私提示（原文摘要至此处截断）……

    arXiv:2605.30727v2 Announce Type: replace  Abstract: Deep research agents increasingly combine private local documents with external tools like web retrieval, creating a privacy risk: an agent's external queries may leak sensitive information from its local context. This risk is amplified by the mosaic effect, where individual queries may appear harmless but become revealing in aggregate. We introduce MosaicLeaks, a benchmark of 1,001 multi-hop deep research tasks that chain private enterprise documents and a public web corpus, forcing agents to make external queries that depend on local information. We evaluate leakage with an adversary LLM that observes only the agent's external queries and attempts to infer private information at three levels: the agent's research intent, answers to specific private questions and verifiable claims about the enterprise documents. We find that models across families and sizes frequently leak at all three levels, that zero-shot privacy prompting has in
    
[^145]: PhoneWorld：扩展手机使用代理环境

    PhoneWorld: Scaling Phone-Use Agent Environments

    [https://arxiv.org/abs/2605.29486](https://arxiv.org/abs/2605.29486)

    PhoneWorld通过真实GUI轨迹自动构建可扩展的手机使用环境，解决了移动代理训练与评估中环境构建的瓶颈。

    

    arXiv:2605.29486v2 公告类型：替换交叉  摘要：手机使用代理的一个核心瓶颈是，难以大规模构建覆盖真实移动行为的可控、可复现环境。现有的移动代理基准在评估方面取得了重要进展，但本身并未提供一种可扩展的方式来构建许多新的手机使用环境。我们提出了PhoneWorld，一个可复用的流水线，将真实的GUI轨迹和截图转换为可控的手机使用环境、可执行任务、自动验证器和训练回放。PhoneWorld不是一次手动构建一个移动基准，而是利用真实轨迹来恢复哪些屏幕重要、屏幕如何连接、哪些交互必须改变环境状态，以及哪些用户目标允许自动验证。从这些信号中，它构建了由只读应用内容和可变状态支持的可运行模拟Android应用，然后推导出可执行任务、基于规则的验证器和训练数据。

    arXiv:2605.29486v2 Announce Type: replace-cross  Abstract: A central bottleneck for phone-use agents is that controllable, reproducible environments covering real mobile behavior are hard to build at scale. Existing mobile-agent benchmarks have made important progress on evaluation, but they do not by themselves provide a scalable way to construct many new phone-use environments. We present PhoneWorld, a reusable pipeline that converts real GUI trajectories and screenshots into controllable phone-use environments, executable tasks, automatic verifiers, and training rollouts. Rather than hand-building one mobile benchmark at a time, PhoneWorld uses real trajectories to recover which screens matter, how screens connect, which interactions must change environment state, and which user goals admit automatic verification. From these signals, it builds runnable mock Android apps backed by read-only app content and mutable state, then derives executable tasks, rule-based verifiers, and traini
    
[^146]: 面向大语言模型长度外推的周期性与谐波周期性旋转位置编码（RoPE）

    Periodic and Harmonic Periodic RoPE for Length Extrapolation in LLMs

    [https://arxiv.org/abs/2605.27980](https://arxiv.org/abs/2605.27980)

    该论文提出周期性 RoPE（P-RoPE）与谐波周期性 RoPE（HP-RoPE）两种位置编码机制，通过与滑窗注意力结合并仅复用 W 个旋转角度来缓解大语言模型长上下文外推中的位置耗尽问题，其中 HP-RoPE 借助在窗口内完成整数圈的谐波频率消除了窗口边界的回绕误差。

    

    arXiv:2605.27980v2 公告类型：replace-cross。摘要：处理长上下文的能力对大语言模型（LLM）执行长程任务至关重要。旋转位置编码（RoPE）为每个 token 分配一个新的旋转角度，因此模型最终会遇到训练中从未见过的角度，即“位置耗尽”问题。我们提出了周期性 RoPE（P-RoPE），一种与滑窗注意力配合使用的位置编码机制。它重复使用 RoPE 角度表中前 W 个（滑窗大小）角度，因此模型只需要 W 个不同的角度。P-RoPE 是有效的，但会在窗口边界处引入回绕误差。我们进一步提出了谐波周期性 RoPE（HP-RoPE），通过选择在 W 个 token 内恰好完成整数圈的频率来消除该误差。我们比较了架构、数据和训练计划完全相同、仅在窗口层位置编码上有所区别的三种模型：标准 RoPE、P-RoPE 和 HP-RoPE。结果表明，P-RoPE 和 HP-RoPE 能够保持困惑度（原文在此处截断）……

    arXiv:2605.27980v2 Announce Type: replace-cross  Abstract: The ability to process long contexts is crucial for large language models (LLMs) to perform long-horizon tasks. Rotary position embedding (RoPE) gives every token a new rotation angle, so a model eventually meets angles it never saw in training, i.e., position exhaustion. We propose Periodic RoPE (P-RoPE), a positional encoding mechanism that operates in conjunction with sliding window attention. It repeats the first sliding-window-size \(W\) angles of the RoPE table, so the model needs only \(W\) distinct angles. P-RoPE works but introduces a wrap error at the window boundary. We further propose Harmonic Periodic RoPE (HP-RoPE), which removes the error by choosing frequencies that complete whole turns in \(W\) tokens. We compare models that share the same architecture, data, and schedule and differ only in the window-layer position encoding: standard RoPE, P-RoPE, and HP-RoPE. The results show that P-RoPE and HP-RoPE keep perp
    
[^147]: AVSD：通过平衡共识信号与教师特定特权信号的自适应视图自蒸馏

    AVSD: Adaptive-View Self-Distillation by Balancing Consensus and Teacher-Specific Privileged Signals

    [https://arxiv.org/abs/2605.20643](https://arxiv.org/abs/2605.20643)

    本文提出 AVSD（自适应视图自蒸馏）方法，通过平衡共识信号与教师特定特权信号，在多种特权信息视图之间自适应地进行自蒸馏，从而联合解决了师生信息不对称和特权信息类型因任务而异这两个难题。

    

    自蒸馏使语言模型能够从自身的轨迹中进行在线学习，其方法是将同一个模型同时用作学生和教师，其中教师以学生无法获取的特权信息为条件。这类信息可以来自不同的类型或视图，例如解题方案、示范、反馈或最终答案。这种设置无需依赖单独的外部模型即可提供密集的 token 级反馈，但会产生一个根本性的不对称问题：教师可能依赖于学生在推理时无法访问的视图特定信息。此外，最佳类型的特权信息往往取决于具体任务，因此难以选择单一的教师视图。在本工作中，我们通过提出 AVSD（自适应视图自蒸馏）来联合解决这两个挑战，这是一种使用多种特权信息视图的新型自蒸馏方法，它重构了 token 级监督……（摘要内容被截断）

    arXiv:2605.20643v2 Announce Type: replace-cross  Abstract: Self-distillation enables language models to learn on-policy from their own trajectories by using the same model as both student and teacher, with the teacher being conditioned on privileged information unavailable to the student. Such information can come in different types or views, such as solutions, demonstrations, feedback, or final answers. This setup provides dense token-level feedback without relying on a separate external model, but creates a fundamental asymmetry: the teacher may rely on view-specific information that the student cannot access at inference time. Moreover, the best type of privileged information is often task-dependent, making it difficult to choose a single teacher view. In this work, we address both these challenges jointly by introducing AVSD (Adaptive-View Self-Distillation), a novel method of self-distillation with multiple privileged-information views, which reconstructs token-level supervision b
    
[^148]: 重新思考大型视觉语言模型中胸部X光推理的视觉归因方法

    Rethinking Visual Attribution for Chest X-ray Reasoning in Large Vision Language Models

    [https://arxiv.org/abs/2605.20158](https://arxiv.org/abs/2605.20158)

    该论文提出了一个基于反事实编辑的因果评估框架，用于验证视觉归因方法是否真正反映了大型视觉语言模型在胸部X光推理中决策所依赖的视觉证据，并发现现有归因方法在该框架下并不可靠。

    

    大型视觉语言模型（LVLMs）在医疗应用中展现出巨大潜力，但其无法将回答忠实锚定于视觉证据，这引发了对临床可信度的严重担忧。虽然视觉归因方法被广泛用于解释LVLM的预测，但这些解释是否真正反映了模型决策背后的视觉证据在很大程度上尚未得到验证，因为模型内部推理的真值标注通常是不可获得的。我们通过开发一个因果评估框架来解决胸部X光（CXR）推理中的这一问题，该框架仅保留那些经反事实编辑验证其专家标注区域对模型预测具有因果责任的CXR-VQA样本。利用该框架，我们在11种归因方法、六个开源LVLM以及两种输出模式（直接回答和逐步推理）上进行评估，发现现有的归因方法往往……（原文摘要在此处被截断）

    arXiv:2605.20158v2 Announce Type: replace-cross  Abstract: Large Vision Language Models (LVLMs) show promise in medical applications, but their inability to faithfully ground responses in visual evidence raises serious concerns about clinical trustworthiness. While visual attribution methods are widely used to explain LVLM predictions, whether these explanations actually reflect the visual evidence underlying the model's decision is largely unverified, since ground-truth annotations for internal model reasoning are typically unavailable. We address this question for chest X-ray (CXR) reasoning by developing a causal evaluation framework that retains only CXR-VQA samples for which the expert-annotated region is verified, via counterfactual editing, to be causally responsible for the model's prediction. Using this framework across 11 attribution methods, six open-source LVLMs, and two output modes (direct answer and step-by-step reasoning), we find that existing attribution methods often
    
[^149]: 智能体崩溃：地狱之路由乐于助人的智能体铺就

    Agent Meltdowns: The Road to Hell Is Paved with Helpful Agents

    [https://arxiv.org/abs/2605.19149](https://arxiv.org/abs/2605.19149)

    该论文提出了一种新型智能体失效模式“意外崩溃”——智能体在遇到良性环境错误时会出于完成任务的目的而主动尝试各种手段，从而引发未经授权侦察等有害行为，并开发了错误注入基础设施对GPT、Grok、Gemini和Claude驱动的智能体进行了系统性安全评估。

    

    在计算机和网络环境中运行的智能体不可避免地会遇到错误：无法访问的网页、缺失的文件、本地和远程的配置错误等。这些错误并不能阻止基于最先进模型的智能体——它们会“乐于助人地”继续寻找完成任务的方法。我们引入、刻画并测量了一种新类型的智能体失效模式，称之为“意外崩溃”：即在没有任何对抗性输入的情况下，智能体对良性环境错误做出不安全或有害行为的现象。由于现有可靠性或安全性基准测试无法捕捉这类崩溃行为，我们开发了一个崩溃行为分类体系。随后，我们实现了一个与具体智能体无关的基础设施，用于向运行环境中注入模拟的本地和远程错误，并利用它系统地评估了由GPT、Grok、Gemini和Claude驱动的智能体。我们的评估表明，崩溃行为（例如，进行未经授权的侦察或破坏系统安全）在主流模型驱动的智能体中普遍存在。

    arXiv:2605.19149v2 Announce Type: replace  Abstract: Agents operating with computer and Web use inevitably encounter errors: inaccessible webpages, missing files, local and remote misconfigurations, etc. These errors do not thwart agents based on state-of-the-art models. They helpfully continue to look for ways to complete their tasks.   We introduce, characterize, and measure a new type of agent failure we call accidental meltdown: unsafe or harmful behavior in response to a benign environmental error, in the absence of any adversarial inputs. Because meltdowns are not captured by the existing reliability or safety benchmarks, we develop a taxonomy of meltdown behaviors. We then implement an agent-agnostic infrastructure for injecting simulated local and remote errors into the rollout environment and use it to systematically evaluate agents powered by GPT, Grok, Gemini, and Claude.   Our evaluation demonstrates that meltdowns (e.g., conducting unauthorized reconnaissance or subverting
    
[^150]: MINTEval：评估长时程智能体系统中多目标干扰下的记忆能力

    MINTEval: Evaluating Memory under Multi-Target Interference in Long-Horizon Agent Systems

    [https://arxiv.org/abs/2605.18565](https://arxiv.org/abs/2605.18565)

    提出MINTEval基准，通过包含频繁更新信息的长且高度关联的上下文、多样化领域任务和问题类型，系统评估记忆增强智能体在多目标干扰和长时程场景下的记忆能力。

    

    现实世界的智能体在漫长且不断演进的时间跨度中运行，其中信息会被反复更新，并可能在多个记忆之间产生相互干扰，这要求智能体能够准确回忆并对多条信息进行聚合推理。然而，现有的基准测试专注于静态、独立的回忆任务，无法捕捉不断演化的记忆之间的这些动态交互作用。在本文中，我们研究了当前记忆增强型智能体在跨多种领域和问题类型的真实、高干扰、长时程环境中的表现。我们提出了MINTEval（长时程干扰下记忆评估），该基准具有以下特点：(1) 长且高度关联的上下文，其中信息频繁更新，从而产生大量干扰；(2) 多样化的领域（状态追踪、多轮对话、维基百科修订和GitHub提交），支持领域泛化能力的评估；(3) 多样化的问题类型，用于评估鲁棒性……

    arXiv:2605.18565v3 Announce Type: replace-cross  Abstract: Real-world agents operate over long and evolving horizons, where information is repeatedly updated and may interfere across memories, requiring accurate recall and aggregated reasoning over multiple pieces of information. However, existing benchmarks focus on static, independent recall and fail to capture these dynamic interactions between evolving memories. In this paper, we study how current memory-augmented agents perform in realistic, interference-heavy, long-horizon settings across diverse domains and question types. We introduce MINTEval (Long-Horizon Memory under INTerference Evaluation), a benchmark featuring (1) long, highly interconnected contexts with frequently updated information that induces substantial interference, (2) diverse domains (state tracking, multi-turn dialogue, Wikipedia revisions, and GitHub commits), enabling evaluation of domain generalization, and (3) diverse question types that assess robustness 
    
[^151]: LongBEL：长上下文与文档一致性的生物医学实体链接

    LongBEL: Long-Context and Document-Consistent Biomedical Entity Linking

    [https://arxiv.org/abs/2605.13451](https://arxiv.org/abs/2605.13451)

    LongBEL提出了一种结合全文档上下文与先前预测记忆的文档级生物医学实体链接框架，通过交叉验证预测进行训练以减少训练-推理不匹配和级联错误，在多语言基准上显著优于句子级基线方法。

    

    生物医学实体链接是将文本中的提及映射到结构化知识库（如UMLS或SNOMED CT）中概念的任务。现有的大多数系统独立地链接每个提及，仅使用该提及本身或其所在的句子。这种做法忽略了同一文档中各提及之间的依赖关系，可能导致预测不一致，尤其当同一概念以不同的表面形式出现时更为明显。我们提出了LongBEL，这是一个文档级生成框架，它将整篇文档的上下文与对先前预测的记忆相结合。为了使这种记忆更加稳健，LongBEL使用交叉验证的预测而非黄金标准标签进行训练，从而减少了训练与推理之间的不匹配，并限制了级联错误。在涵盖英语、法语和西班牙语的五个生物医学基准数据集上的实验表明，LongBEL优于句子级生成基线，其中在概念于文档内频繁重复出现的数据集上提升最为显著。

    arXiv:2605.13451v2 Announce Type: replace  Abstract: Biomedical entity linking maps textual mentions to concepts in structured knowledge bases such as UMLS or SNOMED CT. Most existing systems link each mention independently, using only the mention or its surrounding sentence. This ignores dependencies between mentions in the same document and can lead to inconsistent predictions, especially when the same concept appears under different surface forms. We introduce LongBEL, a document-level generative framework that combines full-document context with a memory of previous predictions. To make this memory robust, LongBEL is trained with cross-validated predictions rather than gold labels, reducing the mismatch between training and inference and limiting cascading errors. Experiments on five biomedical benchmarks across English, French, and Spanish show that LongBEL improves over sentence-level generative baselines, with the largest gains on datasets where concepts frequently recur within 
    
[^152]: Transformer中的几何式事实回忆

    Geometric Factual Recall in Transformers

    [https://arxiv.org/abs/2605.12426](https://arxiv.org/abs/2605.12426)

    该论文证明Transformer可以通过在主体嵌入中编码属性向量的线性叠加、并由小型MLP作为关系条件选择器来提取信息，以对数级嵌入维度实现事实记忆，突破了传统联想记忆要求参数随事实数线性增长的局限。

    

    Transformer语言模型是如何记忆事实关联的？一种常见的观点将内部权重矩阵视为联想记忆，但这种观点要求参数数量随事实数量线性增长。我们提出了一种替代性的几何式记忆形式的理论与实证解释：学习到的嵌入直接编码关系结构，而MLP则扮演性质上截然不同的角色。在一个受控设定中，单层Transformer需要记忆从主体到共享属性集的随机双射映射，我们证明了对数级的嵌入维度就足以完成这一任务。主体嵌入编码其关联属性向量的线性叠加，而一个小型MLP充当受关系条件约束的选择器，从嵌入中提取出相关属性。我们将这些结果进一步扩展到多跳推理场景——即关系查询链，例如“某人的母亲是谁”（摘要在此处截断）。

    arXiv:2605.12426v2 Announce Type: replace  Abstract: How do transformer language models memorize factual associations? A common account views internal weight matrices as an associative memory. That account requires parameter counts that scale linearly with the number of facts. We develop a theoretical and empirical account of an alternative, \emph{geometric} form of memorization in which learned embeddings encode relational structure directly, and the MLP plays a qualitatively different role. In a controlled setting where a single-layer transformer memorizes random bijections from subjects to a shared attribute set, we prove that a logarithmic embedding dimension is sufficient. Subject embeddings encode \emph{linear superpositions} of their associated attribute vectors, and a small MLP acts as a relation-conditioned selector that extracts the relevant attribute from the embedding. We extend these results to the multi-hop setting---chains of relational queries such as ``Who is the mothe
    
[^153]: 别失去焦点：基于键正交投影的激活引导

    Don't Lose Focus: Activation Steering via Key-Orthogonal Projections

    [https://arxiv.org/abs/2605.06342](https://arxiv.org/abs/2605.06342)

    提出SKOP方法，通过键正交投影约束有害的注意力重路由，在保留超过95%引导效力的同时将效用损失降低5-7倍。

    

    激活引导通过干预内部表示来控制大语言模型（LLM）的行为朝向目标行为，然而它常常会降低推理和检索性能。我们认为造成这种权衡的一个主要原因是注意力重路由：引导向量改变了查询-键匹配，使注意力从上下文重要的token转移到信息量较少的token上。为解决这一问题，我们提出了基于键正交投影的引导方法——这是一种能够在保留引导效力的同时约束有害注意力重路由的引导方法。SKOP通过在模型依赖用于推理和检索的一小部分焦点token上保持注意力模式来实现这一点，同时允许在不太关键的尾部token之间重新分配注意力。在多个引导基准测试中，我们证明了SKOP实现了最佳的联合引导-效用权衡，将效用损失降低了5-7倍，同时保留了超过95%的原始引导效力。

    arXiv:2605.06342v2 Announce Type: replace  Abstract: Activation steering controls LLM behaviour towards target behaviour by intervening in internal representations, yet it often degrades reasoning and retrieval performance. We argue that a primary cause of this trade-off is attention rerouting: steering vectors alter query-key matching, shifting attention away from contextually important tokens toward less informative ones. To address this, we propose Steering via Key-Orthogonal Projections (SKOP), a steering method that constrains harmful attention rerouting without eliminating steering efficacy. SKOP achieves this by preserving attention patterns on a small set of focus tokens the model relies on for reasoning and retrieval, while allowing redistribution among less critical tail tokens. Across multiple steering benchmarks, we show that SKOP achieves the best joint steering-utility trade-off, reducing utility degradation by 5-7x while retaining over 95% of vanilla steering efficacy. O
    
[^154]: 自然语言处理：从分词到基于人类反馈的强化学习的综合实践指南

    Natural Language Processing: A Comprehensive Practical Guide from Tokenisation to RLHF

    [https://arxiv.org/abs/2605.03799](https://arxiv.org/abs/2605.03799)

    本论文提供了一套完整的NLP实践教程，涵盖从分词到RLHF的全流程，强调可复现性和开源模型，并融入低资源语言研究。

    

    本预印本呈现了一个系统性、面向研究的实践教程，引导读者贯穿整个现代自然语言处理流水线——从分词和向量化到大语言模型的微调、检索增强生成、基于人类反馈的强化学习、提示工程、模型压缩和多模态系统。十七个动手环节将简洁理论与详细实现计划、形式化评估指标和透明评价标准相结合。这项工作并非传统教科书：它被设计为一个可复现的研究产物，每个环节都要求在公共仓库中发布代码、模型和报告。所有实验均在单一演化语料库上进行，并且该工作倡导使用开放权重模型而非商业API，特别关注Hugging Face生态系统。该材料还包含对低资源语言的原创研究，并融入其中。

    arXiv:2605.03799v3 Announce Type: replace  Abstract: This preprint presents a systematic, research-oriented practicum that guides the reader through the entire modern NLP pipeline --- from tokenisation and vectorisation to fine tuning of large language models, retrieval augmented generation, reinforcement learning from human feedback, prompt engineering, model compression, and multimodal systems. Seventeen hands-on sessions combine concise theory with detailed implementation plans, formalised evaluation metrics, and transparent assessment criteria. The work is not a conventional textbook: it is designed as a reproducible research artefact where every session requires publishing code, models, and reports in public repositories. All experiments are conducted on a single evolving corpus, and the work advocates open weight models over commercial APIs, with special attention to the Hugging Face ecosystem. The material is enriched by original research on low resource languages, incorporating
    
[^155]: 语言模型中任务建模的最小探针

    A minimal probe of task modeling in language models

    [https://arxiv.org/abs/2605.02028](https://arxiv.org/abs/2605.02028)

    该研究提出用“计数相同项目”作为语言模型“任务建模”能力的最小测试，定义了计数能力指标并测量205个模型，发现模型计数会突然失效（中位数仅约50项），其增长远落后于上下文窗口的扩张。

    

    语言模型能够读取长上下文，并在知识和推理基准测试中取得优异成绩，但这两者都无法表明一个模型能否执行下一步依赖于之前所有信息的工作——而这正是长期运行的智能体所必需的能力。我们将这种能力称为“任务建模”，并提出计数相同项目作为其最简单的测试：每个项目都会计入总数，且没有任何项目单独记录它。我们将计数能力定义为模型能以超过一半的概率返回精确计数的序列长度上限，并对205个模型进行了测量。所有已解析的测量都会突然丢失计数，中位数在50个项目处，最多为627个；纵观各发布年份，中位数增长了1.6倍，而上下文窗口却增长了约80倍。改变被计数的符号或分隔符会使模型的CC变化数倍，但在16个模型和11种编码的跨实验中，模型本身约占方差的四分之三。在77个匹配对中，推理模式并未改变中位数CC。

    arXiv:2605.02028v3 Announce Type: replace  Abstract: Language models read long contexts and score well on knowledge and reasoning benchmarks, but neither shows whether a model can do work whose next step depends on everything that came before, as a long-running agent must. We call this ability task modeling and propose counting identical items as its simplest test: every item adds to the total and no item records it. We define counting capacity (CC) as the sequence length up to which a model returns the exact count more often than not, and measure it for 205 models. Every resolved measurement loses count abruptly, at a median of 50 items and at most 627; across release years the median rose 1.6-fold while context windows grew about 80-fold. Changing the counted symbol or the delimiter moves a model's CC severalfold, yet the model accounts for about three quarters of the variance across 16 models and 11 encodings. Reasoning leaves the median CC unchanged across 77 matched pairs. In six 
    
[^156]: 多语言思考，而非更费力：一个教会推理模型进行语码转换的数据高效框架

    Think Multilingual, Not Harder: A Data-Efficient Framework for Teaching Reasoning Models to Code-Switch

    [https://arxiv.org/abs/2604.15490](https://arxiv.org/abs/2604.15490)

    该论文提出了首个基于语言学与行为动机的微调框架，并构建了CoRe语料库，用于识别大型语言模型中有益的语码转换推理行为，从而以数据高效的方式教会推理模型更有效地利用语码转换来提升推理能力。

    

    推理能力的最新进展使大型语言模型能够解决日益复杂的数学、符号和逻辑任务。有趣的是，尽管推理模型通常被训练为生成单语文本，但这些模型也被观察到会出现语码转换（即混合使用多种语言）的现象。以往的工作要么将语码转换视为一种不良错误，要么试图通过修改输入提示或输出解码过程来控制语码转换，要么只关注语言、领域、任务和模型的狭窄子集。我们通过引入首个基于语言学与行为动机的微调框架来填补这些空白，该框架用于识别大型语言模型中有益的语码转换推理行为，并教会这些模型更有效地利用语码转换进行推理。我们构建了语码转换推理语料库，其中包含来自15个模型、18种语言的7千条推理轨迹……（原文摘要在此处截断）

    arXiv:2604.15490v2 Announce Type: replace  Abstract: Recent developments in reasoning capabilities have enabled large language models to solve increasingly complex mathematical, symbolic, and logical tasks. Interestingly, while reasoning models are often trained to generate monolingual text, these models have also been observed to code-switch (i.e., mix languages). Prior works have either viewed code-switching as an undesirable error, attempted to control code-switching through modifications to input prompts or the output decoding process, or focus on narrow subsets of languages, domains, tasks, and models. We address these gaps by introducing the first linguistically and behaviorally motivated fine-tuning framework for identifying beneficial code-switched reasoning behaviors in large language models and teaching these models to code-switch more effectively for reasoning. We create the Code-Switched Reasoning (CoRe) corpus, consisting of (1) 7k reasoning traces from 15 models, 18 langu
    
[^157]: 基于有界不良评判者的稳健LLM评估：一种约束最大似然方法

    Robust LLM Evaluation with Boundedly Bad Judges: A Constrained Maximum Likelihood Approach

    [https://arxiv.org/abs/2604.03257](https://arxiv.org/abs/2604.03257)

    该论文提出约束最大似然估计（CMLE）方法，通过数据驱动约束显式建模LLM评判者的错误率，以受控偏差换取方差降低，从而在可靠标签有限的情况下实现更稳健的大规模LLM评估。

    

    大规模评估语言模型通常需要调和昂贵但可靠的标签（例如来自人类评分者的标签）与廉价但含噪的代理标签（例如来自LLM作为评判者的自动评估器）。本文提出了约束最大似然估计（CMLE），这是一种基于似然的LLM评估方法，它通过真阳性率和假阳性率显式建模代理标签中的误差，并受数据驱动约束的限制。在功能上，这些约束起到自适应正则化的作用，限制了可行的评判者错误率，特别是在可用的可靠标签数量有限的情况下。与渐近无偏估计器（例如基于预测驱动推断的估计器）不同，这一表述在约束设定不当时接受受控的偏差，以换取相对于包括无约束最大似然在内的各种基线的方差降低。

    arXiv:2604.03257v2 Announce Type: replace-cross  Abstract: Evaluating language models at scale often involves reconciling costly, but reliable, labels (e.g., from human raters) with cheap, but noisy, proxy labels (e.g., from LLM-as-a-judge autoraters). This paper proposes constrained maximum likelihood estimation (CMLE), a likelihood-based approach to large language model (LLM) evaluation that explicitly models the error in the proxy labels via their true-positive and false-positive rates, subject to data-driven constraints. Functionally, these constraints act as an adaptive regularization that restricts the feasible judge error rates, particularly when the available number of reliable labels is limited. In contrast to asymptotically unbiased estimators (such as those based on prediction-powered inference), this formulation accepts a controlled bias under misspecified constraints in exchange for variance reduction relative to various baselines including unconstrained maximum likelihood
    
[^158]: DEO：面向否定感知检索的无训练直接嵌入优化

    DEO: Training-Free Direct Embedding Optimization for Negation-Aware Retrieval

    [https://arxiv.org/abs/2603.09185](https://arxiv.org/abs/2603.09185)

    提出无需训练的直接嵌入优化方法DEO，通过将查询分解为正负分量并以对比目标优化查询嵌入，在无需额外训练数据或模型更新的情况下显著提升否定与排除类查询的文本和多模态检索性能。

    

    近年来，大型语言模型（LLM）和检索增强生成（RAG）的进展催生了多样化的检索方法。然而，现有的检索方法在处理否定和排除类查询时往往无法准确检索结果。为解决这一局限，以往的方法依赖嵌入适配或微调，这带来了额外的计算成本和部署复杂性。我们提出了直接嵌入优化（DEO），这是一种无需训练的否定感知文本和多模态检索方法。DEO将查询分解为正向和负向分量，并通过对比目标对查询嵌入进行优化。在无需额外训练数据或模型更新的情况下，DEO在NegConstraint基准上超越了基线方法，取得了+0.0738的nDCG@10和+0.1028的MAP@100提升，同时在多模态检索中将Recall@5相比OpenAI CLIP提高了+6%。这些结果证明了DEO在否定与排除感知检索方面的实用性。

    arXiv:2603.09185v2 Announce Type: replace  Abstract: Recent advances in Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) have enabled diverse retrieval methods. However, existing retrieval methods often fail to accurately retrieve results for negation and exclusion queries. To address this limitation, prior approaches rely on embedding adaptation or fine-tuning, which introduce additional computational cost and deployment complexity. We propose Direct Embedding Optimization (DEO), a training-free method for negation-aware text and multimodal retrieval. DEO decomposes queries into positive and negative components and optimizes the query embedding with a contrastive objective. Without additional training data or model updates, DEO outperforms baselines on NegConstraint, with gains of +0.0738 nDCG@10 and +0.1028 MAP@100, while improving Recall@5 by +6\% over OpenAI CLIP in multimodal retrieval. These results demonstrate the practicality of DEO for negation- and exclus
    
[^159]: 何时想象及想象多少：基于世界模型的视觉空间推理自适应测试时扩展

    When and How Much to Imagine: Adaptive Test-Time Scaling with World Models for Visual Spatial Reasoning

    [https://arxiv.org/abs/2602.08236](https://arxiv.org/abs/2602.08236)

    该论文提出AVIC自适应测试时框架，通过深入分析何时需要视觉想象以及想象的最优程度，利用世界模型自适应地调控想象过程，从而提升多模态大语言模型视觉空间推理的准确性与效率。

    

    尽管多模态大语言模型进展迅速，但当正确答案依赖于场景在未见或替代视角下的呈现方式时，视觉空间推理仍然不可靠。近期工作通过引入世界模型进行视觉想象来增强推理能力，但诸如想象何时真正必要、多少想象是有益的、何时想象会变得有害等问题仍缺乏深入理解。在实际应用中，不加区分的想象会增加计算开销，甚至可能因引入误导性证据而降低性能。在这项工作中，我们对测试时视觉想象作为一种可控的空间推理资源进行了深入分析。我们首先研究了何时静态视觉证据已经足够、何时想象能够改善推理，以及过度或不必要的想象如何影响准确性与效率。为支持这一分析，我们随后提出了AVIC，一个基于世界模型的自适应测试时框架。

    arXiv:2602.08236v3 Announce Type: replace-cross  Abstract: Despite rapid progress in MLLMs, visual spatial reasoning remains unreliable when correct answers depend on how a scene would appear under unseen or alternative viewpoints. Recent work addresses this by augmenting reasoning with world models for visual imagination, but questions such as when imagination is actually necessary, how much of it is beneficial, and when it becomes harmful, remain poorly understood. In practice, indiscriminate imagination can increase computation and even degrade performance by introducing misleading evidence. In this work, we present an in-depth analysis of test-time visual imagination as a controllable resource for spatial reasoning. We first study when static visual evidence is sufficient, when imagination improves reasoning, and how excessive or unnecessary imagination affects accuracy and efficiency. To support this analysis, we then introduce AVIC, an adaptive test-time framework with world mode
    
[^160]: TELEVAL：面向中文交互场景的口语语言模型基准测试

    TELEVAL: A Benchmark Designed for Spoken Language Models in Chinese Interactive Scenarios

    [https://arxiv.org/abs/2507.18061](https://arxiv.org/abs/2507.18061)

    TELEVAL是一个针对中文无指令音频交互场景的大规模基准，首次同时评估SLM的语义准确性和基于声学线索的交互适当性，并发现模型在声学变异下性能显著退化。

    

    arXiv:2507.18061v4 公告类型：替换交叉 摘要：口语语言模型（SLMs）预期能够支持超越任务完成的自然口语交互。然而，现有的SLM基准测试主要评估结构化环境中的语义正确性，并且对基于声学上下文的交互行为评估有限。为解决这一差距，我们引入了TELEVAL，一个大规模的中文口语交互SLM基准测试，适用于无指令、音频条件下的场景。TELEVAL评估两个互补方面：（1）可靠内容满足度，衡量SLMs在不同声学和语言条件下的语义准确性；（2）交互适当性，评估模型是否通过隐式地将行为基于听觉线索来产生自然且适当的回应。实验表明，尽管模型在语义任务上表现良好，但在声学变异和交互设置下其性能会下降。我们观察到...

    arXiv:2507.18061v4 Announce Type: replace-cross  Abstract: Spoken Language Models (SLMs) are expected to support natural spoken interaction beyond task completion. However, existing SLM benchmarks primarily evaluate semantic correctness in structured settings and provide limited assessment of interactional behavior grounded in acoustic context. To address this gap, we introduce TELEVAL, a large-scale SLM benchmark for Chinese spoken interaction in instruction-free, audio-conditioned settings. TELEVAL evaluates two complementary aspects: (1) Reliable Content Fulfillment, which measures semantic accuracy of SLMs under diverse acoustic and linguistic conditions, and (2) Interactional Appropriateness, which assesses whether models produce natural and appropriate responses by implicitly grounding behavior in auditory cues. Experiments show that while models perform competitively on semantic tasks, their performance degrades under acoustic variability and in interactional settings. We observ
    
[^161]: 机制可解释性需要哲学

    Mechanistic Interpretability Needs Philosophy

    [https://arxiv.org/abs/2506.18852](https://arxiv.org/abs/2506.18852)

    该论文主张机制可解释性研究需要与哲学建立持续的合作关系，借助哲学澄清概念、完善方法并应对认识论与伦理层面的复杂性，从而释放该领域尚未实现的巨大发展潜力。

    

    机制可解释性旨在通过揭示训练后的神经网络背后的底层机制来解释其工作原理。随着该领域影响力的不断增长，不仅审视模型本身，审视机制可解释性研究中隐含的假设、概念和解释策略也变得日益重要。我们认为，机制可解释性需要哲学作为持续合作的伙伴，以澄清其概念、完善其方法，并应对解释人工智能系统时涉及的认识论和伦理复杂性。通过与哲学家和哲学框架的深入接触，机制可解释性的进步仍具有巨大的未开发潜力。本文以机制可解释性文献中的三个开放性问题为例，展示了哲学能够为机制可解释性研究带来的价值，并勾勒出一条通向更深入跨学科对话的路径。

    arXiv:2506.18852v3 Announce Type: replace-cross  Abstract: Mechanistic interpretability (MI) aims to explain how trained neural networks work by uncovering their underlying mechanisms. As the field grows in influence, it is increasingly important to examine not just models themselves, but the assumptions, concepts and explanatory strategies implicit in MI research. We argue that mechanistic interpretability needs philosophy as an ongoing partner in clarifying its concepts, refining its methods, and navigating the epistemic and ethical complexities of interpreting AI systems. There is significant unrealised potential for progress in MI to be gained through deeper engagement with philosophers and philosophical frameworks. Taking three open problems from the MI literature as examples, this paper illustrates the value philosophy can add to MI research, and outlines a path toward deeper interdisciplinary dialogue.
    

