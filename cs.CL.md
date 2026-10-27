# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Center-of-Mass Interpretation of Evolution Strategies](https://arxiv.org/abs/2610.35865) | 本文提出进化策略更新的几何解释：将扰动模型视为粒子、奖励视为质量，证明减去均值的ES更新恰好等于种群从等质量质心到奖励加权质心的位移乘以一个标量因子。 |
| [^2] | [Preserve to Expand: Source-Aware Adaptation and Model Merging for Multilingual LLM-Based ASR](https://arxiv.org/abs/2610.35846) | 该论文提出了在基于连接器的语音大语言模型中扩展多语言ASR时的适配与合并策略评估，发现SSU-Wanda+TIES合并策略在单目标扩展中实现了最佳的目标语言习得与源语言保持权衡，而更新合并在多目标扩展中提供了稳定的源语言保持能力。 |
| [^3] | [Understanding Jev-Style System One Models: Does Structure Reduce Selection Bias?](https://arxiv.org/abs/2610.35826) | 本文通过在保持问题内容不变的前提下打乱选项顺序的简单测试，系统比较了两个Jev版本、十二个Jev复现模型和六个生成式LLM在多选项任务中的选择偏差，探究Jev式决策结构能否减少选择偏差。 |
| [^4] | [Learning When to Commit: Schedule Distillation for Fast Diffusion Language Models](https://arxiv.org/abs/2610.35804) | 提出dSpark框架，通过调度蒸馏学习何时提交token——优先揭示那些能让其他位置更易预测的token，从而在保持生成质量的同时大幅加速扩散大语言模型的并行生成。 |
| [^5] | [Quantifying Claim Smoothing by LLM Research Agents](https://arxiv.org/abs/2610.35793) | 该论文发现LLM研究智能体在生成报告时会把多元甚至冲突的证据“平滑”为过度自信的单一叙事，并提出了SmoothScore这一论断级评分流程来量化这种失真程度。 |
| [^6] | [Atomic Security Concerns: Towards a Foundational Language for Security Knowledge Representation](https://arxiv.org/abs/2610.35782) | 本文受化学原子概念启发，提出“原子安全关注点”的概念，并假设网络安全知识可以通过一组有限的基础关注点的组合来表示，从而为安全知识表示奠定一种基础语言。 |
| [^7] | [SEMA-Auto: Sighted and Efficient Multi-Turn Attack with Autonomy](https://arxiv.org/abs/2610.35743) | 提出SEMA-Auto，一种无需循环内评估器、能感知受害者回复并自主决定终止时机的多轮越狱攻击方法，通过合成自举调优和决策监督强化学习两阶段训练实现高效自主攻击。 |
| [^8] | [Can JEV Support Clinical Trial Criteria Assessment?](https://arxiv.org/abs/2610.35721) | 本文提出非生成式决策模型JEV可在严格数据隐私保护下用于临床试验标准预筛，其性能与大型生成式模型Qwen Flash相当，并在MIMIC-III数据集上显著更优。 |
| [^9] | [Quantitative Generalization Bounds for Multi-Layer Transformers: A Few Long Strings can Beat Many Short Ones](https://arxiv.org/abs/2610.35715) | 本文首次对固定精度多层Transformer的分布内统计泛化与长度泛化进行了联合的定量理论分析，并揭示出一个反直觉的结论：用少数长字符串训练可以胜过用大量短字符串训练。 |
| [^10] | [From Self-Confidence to Statistical Accuracy: Scaling Human-in-the-Loop Evaluation with Direct and LLM-Assisted Auditing](https://arxiv.org/abs/2610.35698) | 该论文提出统一统计框架，通过置信度分层抽样实现无需校准置信度的直接人工审计，并基于灵敏度和假阳性率建模LLM评判者构建误差校正估计量，从而实现可扩展的人机协同模型评估。 |
| [^11] | [MiRL: Error-Targeted Reinforcement Learning for Correcting Terminal Agent Misbehaviors](https://arxiv.org/abs/2610.35697) | 本文提出MiRL方法，通过分析终端智能体的错误以发现跨环境反复出现的可泛化错误行为，并从错误发生步骤启动强化学习训练、仅在通过测试且避免错误行为时给予奖励，从而有针对性地纠正智能体的错误行为。 |
| [^12] | [Not One Belief: Answer Payloads, Persistence, and Dissociated Confidence Signals in Conversational Question Answering](https://arxiv.org/abs/2610.35694) | 该研究发现对话式问答中答案的改变主要由用户提供的答案载荷（具体答案字符串）驱动而非言语行为本身，且四种置信度信号相互分离——模型内部探测器追踪的是答案的持久性而非真实性。 |
| [^13] | [Adapting LLM-SC to Living Sacred Heritage: Domain-Specific Extraction, Semantic Networks and Human Evaluation on Pashupatinath Temple Reviews](https://arxiv.org/abs/2610.35685) | 该研究将LLM-SC框架适配到尼泊尔帕斯帕提那神庙这一活态神圣遗产场景，通过新增8种神圣遗产实体类型构建12类实体分类体系，从Tripadvisor评论中成功提取结构化数据，且新增的神圣遗产类型占实体记录的64.8%，证明领域特定适配对宗教遗产评论分析至关重要。 |
| [^14] | [Evaluating DeepSeek V4.1 Flash-1017 for loss of control and harmful manipulation risks](https://arxiv.org/abs/2610.35671) | 该论文依据欧盟通用AI实践准则，对DeepSeek V4.1 Flash-1017开展了独立第三方安全评估，在34个基准上系统测试其失控与有害操纵风险，并与多个领先的闭源前沿模型进行了对比。 |
| [^15] | [Improving JEV Entailment Reranking with Span Readouts and Selective Adaptation](https://arxiv.org/abs/2610.35654) | 本文提出结合平均跨度读取与选择性骨干适配的方法，用NLI选项内上下文状态均值替代选项结尾指针，并在保留注意力与门控DeltaNet投影低秩更新的同时移除前馈适配器，从而有效提升了JEV蕴含重排序在MC11基准上的准确率。 |
| [^16] | [Offline Language-Based Support System for Tactical Decision-Making in Air Combat Operations](https://arxiv.org/abs/2610.35635) | 本文提出了完全离线、基于开源组件和检索增强生成（RAG）技术的语言支持系统MATE，通过战术手册内容回答自然语言问题以辅助空战飞行员训练与战术决策，并经巴西空军飞行员评估获得4.00至4.40的高评分。 |
| [^17] | [Hierarchical Skill Evolution for Coding Agent](https://arxiv.org/abs/2610.35616) | 提出分层技能演化（HSE）方法，通过任务级、仓库级到跨仓库的分层证据聚合，从编程轨迹中逐步提炼可跨仓库迁移的程序性技能，并借助外部书写-评估循环在冻结底层模型的情况下驱动技能生成与评估。 |
| [^18] | [Making Reasoning Visible: Configuration-Aware Diagnostic Modeling for Adaptive Scaffolding in Structured Reasoning Tasks](https://arxiv.org/abs/2610.35609) | 该论文提出一种配置感知的诊断建模框架，通过分析学生在交互式学习游戏中的动作序列使多步骤推理过程可视化，并发现推理策略主要随情境变化而非稳定个人特质，从而为结构化推理任务中的自适应脚手架教学提供支持。 |
| [^19] | [Linear Probes Transfer from Base to Distilled Language Models despite Rotation](https://arxiv.org/abs/2610.35586) | 研究发现，尽管蒸馏推理模型和指令微调模型会使线性探针所依赖的安全/不安全提示分离方向旋转17°到35°，但在基础模型上训练的有害请求检测探针无需任何修改即可有效迁移到这些后继模型上，AUC变化极小。 |
| [^20] | [Integrate the Reward, Don't Sample It: Window Likelihood for Interruptions and Backchannels in Full-Duplex Speech Models](https://arxiv.org/abs/2610.35581) | 该论文发现基于整轮展开的GRPO时机奖励在训练全双工语音模型时存在零梯度陷阱与信用泄漏两大缺陷，并提出“窗口似然”方法，通过对停止决策进行闭式积分直接最大化正确让出/保持的对数概率，在HumDial-FDBench上取得了最佳的让出/保持F1表现。 |
| [^21] | [EXCISE: Locating and Excising Distracting Sentences in Medical Question Answering](https://arxiv.org/abs/2610.35558) | 提出推理时方法EXCISE，利用模型隐藏状态识别与病例不一致的句子并阻断其对答案的影响，从而在不需预先定位干扰句的情况下提升医学问答的鲁棒性。 |
| [^22] | [LoopFold: Parallelizing Recurrent Depth in Looped Transformers](https://arxiv.org/abs/2610.35556) | LoopFold通过轻量级预测器提前启动较深的循环，并用校正器利用先前循环结果对其进行精化，将循环Transformer中多个循环深度的计算折叠为并行批次，从而在不损失精度的前提下降低解码延迟。 |
| [^23] | [StateLoop: Persistent State for Looped Transformers](https://arxiv.org/abs/2610.35544) | StateLoop通过为每个token引入一个由可学习门控控制的持久状态来在循环迭代之间保留信息，从而降低语言建模困惑度并提升零样本任务准确率。 |
| [^24] | [An Algorithmic Account of Few-Shot Concept Induction in Language Models](https://arxiv.org/abs/2610.35533) | 本文提出摊销方向强化（ADA）这一算法性假设，解释了语言模型在少样本概念归纳中如何确定特征相关性，即通过利用关于哪些特征有助于发现共性的摊销知识来构建特征空间中示例的摘要表示。 |
| [^25] | [S3AC: Stabilized, Selective, and Scalable Actor-Critic for LLM Reasoning](https://arxiv.org/abs/2610.35518) | S3AC提出了一种实用的评论家训练方案，为长思维链推理提供token级优势估计和提示级基线，并结合难度感知提示采样（DAPS），从而解决无评论家RLVR方法在信用分配与基线估计上的不足，提升大语言模型强化学习训练的稳定性与可扩展性。 |
| [^26] | [MATCHA: A Chinese Multi-Disorder Mental Health Dataset with Cross-Lingual Analyses on Social Media](https://arxiv.org/abs/2610.35512) | MATCHA是首个中文多障碍心理健康数据集，将七种DSM-5障碍的微博超话社区与用户公开时间线关联，支持跨语言、跨文化的社交媒体心理健康研究。 |
| [^27] | [Beyond Words Alone: Disentangling Distributed Evidence for Visually Grounded Cultural Translation](https://arxiv.org/abs/2610.35482) | 本文提出 Culture-MMT 诊断基准，通过系统解耦社交媒体翻译中分散于文本、图像与文化知识中的证据，证明视觉证据对于实现文化适配的翻译至关重要。 |
| [^28] | [GPDPO: Learning from Reward Conflicts via Group Pareto-Dominance Policy Optimization](https://arxiv.org/abs/2610.35480) | 提出GPDPO方法，通过识别冲突过滤后采样响应之间的帕累托支配关系并将其转化为辅助偏好信号，从而从奖励冲突中恢复被丢弃的成对监督信息，实现更稳定的多奖励策略优化。 |
| [^29] | [MANTIS: Multi-Axis News Transformations for Instance-Wise Stress Testing and Worst-Case Training in Fake News Detection](https://arxiv.org/abs/2610.35475) | 该论文提出了多轴新闻变换基准MANTIS-Bench和实例自适应对抗框架MANTIS，通过针对每篇新闻在写作风格、情感表达和传播框架上进行动态最坏情况训练，显著提升了假新闻检测器在多样化改写攻击下的鲁棒性。 |
| [^30] | [A Contextual Measure of Ambiguity in Prescriptive Text](https://arxiv.org/abs/2610.35461) | 该论文提出一种情境化的歧义度量方法，通过量化利益相关者对规范性文本在不同场景中如何适用的分歧程度来衡量歧义，初步分析表明即使使用模拟评估者，该方法也与真实的人类分歧相关。 |
| [^31] | [Skill$^{\infty}$: Recursive Self-Improvement for Agent Skill Optimization](https://arxiv.org/abs/2610.35434) | 提出Skill$^{\infty}$框架，元智能体在遵循元技能优化任务技能的同时，用同一流程递归地优化元技能本身，且优化步骤由智能体自主定义和重写，从而实现真正的递归自我改进。 |
| [^32] | [Probing the phon* representations of the Mimi codebooks](https://arxiv.org/abs/2610.35432) | 本文通过强制对齐、熵分析和音素归一化互信息（PNMI）对Mimi编解码器首个码本进行探测，发现其token与音素存在强关联（PNMI=0.84），但音素与token并非一一对应，而是由多个竞争性token共同表示。 |
| [^33] | [INVAR-Agri: Detecting Discrepancies in Multilingual Operator Manuals of Agricultural Implements with a Fine-Tuned LLM](https://arxiv.org/abs/2610.35423) | 该论文提出 INVAR-Agri 概念验证，在 EuroLLM-1.7B-Instruct 上训练并比较“回译后对比”与“双语直接检查”两种微调工作流程，以在 24 种语言的农业机械操作手册中自动检测涉及数字、单位、义务、否定和条件的翻译差异。 |
| [^34] | [ImgMT: Wikipedia-based Dataset For Evaluating Text-in-Image Translation](https://arxiv.org/abs/2610.35416) | 该论文提出了ImgMT，一个从维基共享资源SVG插图中挖掘的大规模多语言图像内文本翻译数据集与基准，涵盖129种语言和4,886个语言对，并配合多层次评估协议发现源语言和书写系统会显著影响翻译质量。 |
| [^35] | [The Geometry of Emotion Representations Across Scale, Depth, and Instruction Tuning](https://arxiv.org/abs/2610.35411) | 该研究证明Gemma 2系列从2B到27B、无论基础版还是指令微调版的大语言模型均共享一种可通过线性映射相互转换的核心情绪几何结构，且模型规模扩大使该几何结构方差分布于更多方向但仍保持与情绪效价的紧密关联。 |
| [^36] | [Agreement Between Reconstructed and Native GDELT GKG Records: A Field-Level Baseline](https://arxiv.org/abs/2610.35358) | 研究者开发了开源工具 gdelt-regkg 来重构 GDELT 全球知识图谱记录，并对近 20 万篇文章进行了字段级一致性评估，首次为研究人员自行生成兼容 GKG 记录提供了量化基准。 |
| [^37] | [Constrained Decoding: Harmful or Beneficial? Revisiting Its Impact on LLM Performance](https://arxiv.org/abs/2610.35328) | 该研究将约束解码划分为格式、有效性和推理三类约束，并通过理论与跨四个模型家族的因果消融实验证明，性能下降主要源于提示与输出格式的不等效而非解码器屏蔽本身，而适当的约束（尤其是有效性约束）反而能保持甚至提升模型性能与可靠性。 |
| [^38] | [CRIM: Composable and Reusable Item Memory for Efficient Personalized Retrieval](https://arxiv.org/abs/2610.35323) | CRIM通过将物品压缩为与用户和查询无关的可复用连续记忆，并与请求级历史建模解耦，在推理时按交互顺序组合缓存记忆生成个性化检索表示，从而大幅降低大语言模型在个性化搜索与推荐中的训练与推理成本。 |
| [^39] | [Did the Model Use That? Counterfactual Training Boosts Self-Attribution in Language Models](https://arxiv.org/abs/2610.35295) | 提出CREDIT方法，通过反事实训练直接监督模型最终答案的自我归因，使语言模型能够准确识别并声明其回答是否受到了所提供信息的影响。 |
| [^40] | [The Voice on the Line](https://arxiv.org/abs/2610.35272) | 该论文训练了20毫秒延迟、能从通话本身注册来电者的因果语音隔离器（掩码transformer），并发现将WavLM-Large权重因果化并与模型联合训练能显著提升效果，将开发集错误率从11.44%降至10.55%。 |
| [^41] | [MechAlign: Evaluating Semantic Alignment at the Project Level in Mathematical Formalization](https://arxiv.org/abs/2610.35269) | MechAlign 提出了一个项目级语义对齐审查框架，通过依赖提取工具 LeanTrace 构建语句依赖图并结合 LLM 评估，检验 Lean 形式化命题是否忠实表达了手稿中的数学断言，并定位报告其中的差异。 |
| [^42] | [The Lay of the Land in Idiom-Translation Quality Estimation](https://arxiv.org/abs/2610.35261) | 该研究构建了首个用于习语翻译质量评估的平行数据集，涵盖九种目标语言的直译与地道译文，并揭示现有QE指标和LLM普遍存在直译偏好，倾向于将习语的直译评为优于地道翻译。 |
| [^43] | [Style over Substance? Linguistic Bias in LLM-based Peer Review Copy title](https://arxiv.org/abs/2610.35254) | 该论文提出了一个受控实验框架，通过基于语言学文献推导的细粒度改写指令，系统性地改变论文的语言风格（母语/非母语特征与高/低置信度特征），以研究基于大语言模型的同行评审是否会受到与科学内容无关的语言偏见的影响。 |
| [^44] | [Beyond Vision-Language Models: Visual-Semantic Affect Graphs for Emotion Recognition in Artworks](https://arxiv.org/abs/2610.35244) | 本文提出视觉-语义情感图（VSAG）框架，通过色彩、笔触、构图、光影、线条等可解释特征节点构建图结构来学习特征与情感的关联，在艺术品情感识别任务上超越了直接使用视觉-语言模型进行推理的方法。 |
| [^45] | [Trainable Boundary-Keeping Where No Formal Answer Exists: A Controlled Knock-In Study of Reward Training in Evidence-Sufficiency Judgment](https://arxiv.org/abs/2610.35226) | 该研究通过预注册的同底座受控敲入实验证明，在不存在可判定正确性奖励的证据充分性判断任务中，“先SFT后GRPO”的边界保持训练配方能将模型的自发正确拒绝率从25%提升至100%，而乱序配对的安慰剂组则完全失效。 |
| [^46] | [DTCA: Denoising Token Credit Assignment for Reinforcement Learning in Diffusion Language Models](https://arxiv.org/abs/2610.35201) | 提出DTCA框架，通过测量词元确定后其余掩码位置预测置信度的提升并按词性类别分组来估计词元贡献，解决了扩散语言模型强化学习中的词元级信用分配难题。 |
| [^47] | [Lost in Translation: Investigating Systematic Discrepancies between Parallel English and Chinese Names of American Chinese Restaurants](https://arxiv.org/abs/2610.35191) | 本文提出8维度名称框架并构建NameFrames数据集，对美国3,172家中餐厅的中英文名称进行系统性对比，揭示了平行名称之间的差异，以探究华人移民在美国的文化融入情况。 |
| [^48] | [High Scores, Hidden Failures: Evaluating Codebook-Based LLM Content Analysis](https://arxiv.org/abs/2610.35179) | 该研究通过跨四个数据集和五个模型的大规模实证分析，揭示基于LLM的码本内容分析尽管总体得分很高，却隐藏着标签级别差异、运行不稳定性和领域敏感性等被标准指标掩盖的失效模式，且这些失效在常用优化策略下依然存在。 |
| [^49] | [Do Language Models Understand Igbo Culture? Beyond Multiple Choice](https://arxiv.org/abs/2610.35168) | 本文提出了基于伊博本土文化构建并由母语者验证的CAB-Igbo基准测试，通过多选题与人类评估相结合的方式，揭示了语言模型的流畅性与文化知识之间存在严重脱节，且多数模型在英文中就已缺失认知忠实度和表征丰富度。 |
| [^50] | [CalibraRL: Confidence-Calibrated Reinforcement Learning for Mitigating Hallucinations in RAG](https://arxiv.org/abs/2610.35142) | CalibraRL是一个将回答置信度和查询级回答证据融入奖励设计的强化学习框架，通过对非放弃回答施加Brier惩罚、对放弃回答利用Beta后验估计条件回答可靠性，从而有效缓解RAG中的幻觉问题。 |
| [^51] | [Quality over Quantity: Rethinking Data and Model Assumptions for Bengali Coreference Resolution](https://arxiv.org/abs/2610.35114) | 该研究针对孟加拉语共指消解任务，证明少量高质量标注数据显著优于更大的混合质量数据集，而数据增强（如回译）无益甚至有害，跨语言迁移效果也不一致，从而挑战了低资源NLP中关于模型泛化与数据效率的常见假设。 |
| [^52] | [Rethinking Evaluation for Principled Generalization Using a Kripkean Challenge to Meaning as Use](https://arxiv.org/abs/2610.35111) | 该论文借助克里普克对维特根斯坦的解读，论证有限的使用历史无法唯一确定规则，因此基于固定用例或局部鲁棒性的基准测试不足以证明语义能力，并提出通过评估抽象与递归两种能力来检验模型在原则性领域扩展下保持规则与概念的能力。 |
| [^53] | [Useful Diversity or Hallucinated Variety? Persona-Conditioned Synthetic Query Generation for E-Commerce Search](https://arxiv.org/abs/2610.35102) | 该研究通过人格条件化提示生成大规模合成查询来冷启动电商语义解析器，发现IPIP人格捆绑包可显著提升解析器F1、语法条件化能引导生成结果更贴近真实流量，并提出对齐sKL作为解析器性能的无标注代理指标。 |
| [^54] | [RAGing Against Misogyny: Improving Implicit Misogyny Explanations](https://arxiv.org/abs/2610.35094) | 本研究提出利用检索增强生成（RAG）框架，结合百科、俚语和女性主义知识来源，改进英语和意大利语中隐性厌女言论的检测与解释，其中女性主义文献被证明是最持续有用的知识来源。 |
| [^55] | [Ask Before You Answer: A Question-Driven Evidence Reasoning Architecture (QERA) for In-Context Learning](https://arxiv.org/abs/2610.35083) | 该论文提出QERA架构，通过先将用户请求转化为需求清单并筛选出隐含信息需求，再从完整对话中作答并用代码验证引文，从而解决大语言模型在真实生活上下文学习任务中因请求隐含大量未明说标准而表现不佳的问题。 |
| [^56] | [CoSA: The Conflict of Skills Attack in Real-World LLM Agent Ecosystems](https://arxiv.org/abs/2610.35082) | 本文提出“技能冲突”（CoS）概念及CoSA攻击方法，证明仅通过向LLM智能体技能池添加本地准入的新技能（进行能力补充或关系重塑），就能在技能组合使用时引发策略违规，且无需改动现有技能或运行时权限。 |
| [^57] | [Improving VLMs on Scientific Chart Analysis via Autonomous Evolution](https://arxiv.org/abs/2610.35077) | 本文提出利用自主进化方法在冻结模型的推理程序代码空间中进行搜索，在不改变模型本身的前提下显著恢复视觉语言模型在科学图表问答任务上的准确率。 |
| [^58] | [Reinforcement Learning Done Right for Explicit Confidence Targets](https://arxiv.org/abs/2610.35059) | 该论文研究了让大语言模型根据用户指定的显式置信度目标灵活决定回答或弃答的问题，指出现有强化学习方法失败的原因在于区分不同目标 p 值的奖励信号在优化中丢失等问题，并提出正确的强化学习方法加以解决。 |
| [^59] | [From Questions to Queries to Insights: A Schema-Grounded Agentic Workflow for Natural-Language Crash Data Analysis](https://arxiv.org/abs/2610.35054) | 本文提出 CrashDB-QA，一个完全本地化、以数据模式和安全分析惯例为基准的智能体工作流，使交通从业者能够通过自然语言对交通事故数据库进行可靠、可审计的自动查询与分析。 |
| [^60] | [SimPOTrans: Self-Adaptation for Offline-to-Simultaneous Speech Translation via Reinforcement Learning](https://arxiv.org/abs/2610.35042) | SimPOTrans提出了一种简单的两阶段强化学习后训练框架，使离线语音翻译模型无需依赖合成轨迹或多目标联合优化即可自适应地获得低延迟的同声语音翻译能力。 |
| [^61] | [Synthetic Child-Adult Interactions Are Not Developmentally Plausible](https://arxiv.org/abs/2610.35029) | 该论文发现合成的儿童-成人对话数据虽在词汇层面接近自然对话，却严重低估了口语互动的混乱与碎片化特性，并提出用真实对话数据进行指令微调，使小模型经过适度参数高效微调后即可逼近自然数据的统计特征。 |
| [^62] | [When the Persona Contradicts the Line: Diagnosing Plan-to-Synthesis Failures in Chain-of-Thought TTS](https://arxiv.org/abs/2610.35008) | 该论文提出了一个日英双语人设-台词冲突基准，系统定位了思维链TTS中从规划到合成各环节的意图丢失问题，并证明监督情感分类器或采用学习式文本-潜变量接口可大幅提升声学情感方向的一致性。 |
| [^63] | [Reason or Recite: Complementary Specialists with an LLM Judge for Traffic Anomaly Reasoning](https://arxiv.org/abs/2610.34993) | 该论文提出RoR双专家框架，用一个9B模型学习基于结构化时间推理的能力、一个27B模型学习标注答案风格，再由LLM裁判在两者间择优，从而在交通异常推理任务中同时兼顾精确的证据推理与开放式回答的评分要求。 |
| [^64] | [Diagnosing and Mitigating Long-Session Degradation in Full-Duplex Speech Models](https://arxiv.org/abs/2610.34991) | 本文提出一个闭环评估框架系统性诊断全双工语音模型在长会话中的性能退化，并提出基于自蒸馏的轻量级LoRA适配训练方法来有效缓解该退化。 |
| [^65] | [AskGLP1: A Knowledge Graph LLM Agent for Exploring GLP-1 Experience from Online Patient Narratives](https://arxiv.org/abs/2610.34987) | 提出AskGLP1系统，利用LLM辅助流程将近850万条在线患者叙述构建成基于MedDRA术语的知识图谱，揭示GLP-1用药经验具有来源和药物特异性，且在线叙述更关注疲劳等日常负担，而FAERS报告更多涉及注射相关事件 |
| [^66] | [SVD-Attention: Efficiently Analyzing Attention Blocks in Transformers](https://arxiv.org/abs/2610.34971) | 该论文提出SVD-Attention——一种无需微调的事后诊断方法，通过谱幅值与激活方差对每个注意力头的奇异模式评分，精确识别哪些头真正用满秩预算，并揭示了注意力冗余随任务类型和层深分布的规律。 |
| [^67] | [Improving Reasoning with Rollout Agreement for Group-Based Policy Optimization](https://arxiv.org/abs/2610.34951) | 该研究发现GRPO训练中正确答案展开间的低一致性往往伴随更多虚假推理，并提出RoAR方法，在奖励中加入推理级别的一致性信号，以提升大模型推理的可靠性。 |
| [^68] | [Correlation-Aware Exploration for Long-Trace Reasoning](https://arxiv.org/abs/2610.34949) | 提出Info-GRPO信息论框架，通过培养策略与潜在先验之间的互信息相关性来解决GRPO长链推理训练中的策略崩溃问题，避免了熵正则化与收敛所需熵衰减之间的冲突。 |
| [^69] | [What Outcome RL Buys: Finishing, Reliability, Rarely Reach](https://arxiv.org/abs/2610.34894) | 该论文将结果导向强化学习带来的pass@1提升分解为完成度、可靠性和触达三部分，发现在固定token预算下RL的大部分收益其实来自“完成度”（即在预算内给出答案），而真正解决基础模型无法解决的新问题的“触达”很少。 |
| [^70] | [FATE-5: Faithful Alignment of Trait Expression for Continuous Big Five Personality Control](https://arxiv.org/abs/2610.34889) | FATE-5提出两阶段框架，将大五人格编码与自然度对齐分离，实现了对大语言模型人格的连续控制，在保持强特质表达的同时显著提升了对话自然度。 |
| [^71] | [NumTrace: Auditing the Number Provenance of Financial Analyst Agents](https://arxiv.org/abs/2610.34887) | 提出首个审计金融分析师智能体数字论断溯源（来源与财务语义）的基准 NumTrace，基于 2,163 份 SEC EDGAR 文档构建 540 个问题，通过数值、绑定、支持三项独立判定来检测“数字正确但归属错误”等问题。 |
| [^72] | [ThinkBit: Search Codewords, Recalibrate Scales, Select by Reasoning](https://arxiv.org/abs/2610.34886) | ThinkBit是一种旨在保持大语言模型推理能力的严格标量2比特量化框架，它通过层敏感度分配各层比特宽度、用Gumbel-softmax逐块搜索最优码字并重新校准缩放因子，有望将推理成本降低8倍。 |
| [^73] | [RubriX: A Fast, Fine-Tunable Non-Autoregressive Rubric Judge](https://arxiv.org/abs/2610.34875) | RubriX是一种受对比语言模型启发、利用LoRA适配编码器和余弦相似度softmax的非自回归量规评判器，无需文本生成即可高效判断回答符合量规哪个等级，准确率达75.4%，显著超越现有LLM评判方法。 |
| [^74] | [Adaptive Retrieval, Selective Answers: Evaluating Conformal Factuality in Agentic RAG](https://arxiv.org/abs/2610.34872) | 该研究首次系统评估了保形事实性过滤在智能体RAG中的表现，发现更高的保形目标水平能提升回答的完全支持率，但代价是答案可用性的下降。 |
| [^75] | [DocTalk: Enhancing Medical Knowledge Retrieval via Fine-tuned Large Language Models with Chain-of-Thought Reasoning for Robust Clinical Decision Support](https://arxiv.org/abs/2610.34871) | DocTalk是一个基于思维链推理的医生辅助问答系统，通过构建包含500个临床问答对的高保真数据流水线，并利用LoRA对六个开源大语言模型进行参数高效指令微调，实现了从出院小结中生成准确、基于病历的临床决策支持回复。 |
| [^76] | [Priority-Preserving Advantage Aggregation for Multi-Reward Policy Optimization](https://arxiv.org/abs/2610.34862) | 提出PAPO方法，通过非对称地聚合主要优势与辅助优势来解决多奖励强化学习中的优势优先级反转问题，在目标一致时保留辅助监督、在冲突时平滑衰减辅助信号，从而始终优先保证主要目标的优化方向。 |
| [^77] | [Semantic Atoms for Scalable Text Analysis in Data Analytics in the Era of LLM-Based Agents](https://arxiv.org/abs/2610.34860) | 该论文提出两个查询驱动的语义原子——标记和分类，用于在LLM智能体规划的数据分析工作流中划分神经语义处理与符号操作的分工，并支持语义输出的重用，从而实现可扩展的文本分析。 |
| [^78] | [Position Matters More Than Channel: Talking a Language Model Out of a Correct Answer](https://arxiv.org/abs/2610.34809) | 该研究发现建议在提示中的位置比 API 通道更能左右语言模型是否放弃正确答案——裸声明在模型作答前危害最大，而带虚假理由的论证在提问前危害最大，可将模型原本几乎不会答错的题目错误率从 1% 提升至 67%。 |
| [^79] | [Text Is the Bottleneck: A Small-Scale Replication of LLM Game Playing in Riichi Mahjong](https://arxiv.org/abs/2610.34791) | 该论文在小预算下复现立直麻将 LLM 对局实验，证明文本输入是瓶颈——仅数百万参数的卷积/Transformer 牌面编码器在决策一致率和实战排名上均超越了 20 亿参数以下的文本提示 LLM，并能与教师 AI Mortal 冠军达到统计意义上的平局。 |
| [^80] | [BehaviorMem: Building and Evaluating User Memory from Real-World E-Commerce Behavior Streams](https://arxiv.org/abs/2610.34790) | 该论文提出BehaviorMem，一个基于真实电商行为流构建用户记忆的方法与评估框架，能够同时支持并分别评估“记住具体证据与兴趣”和“推断潜在需求与偏好”这两种核心能力。 |
| [^81] | [Natural Jammers: When Non-Robust Features Become Antagonistic](https://arxiv.org/abs/2610.34785) | 该论文发现非鲁棒特征并非仅是捷径，而是可作为“自然干扰器”通过频谱-空间锁定机制在干净图像上主动误导标准模型，且尽管单个干扰器对几何扰动脆弱，干扰现象在整体上依然持续存在。 |
| [^82] | [DynEdit: Dynamic Entropy-Guided Sequential Editing for Large Language Models](https://arxiv.org/abs/2610.34768) | 提出DynEdit框架，发现词元熵与编辑难度及语义重要性密切相关，并利用熵引导的非均匀滑动和词元级优化来提升大语言模型长序列知识编辑的覆盖与效果。 |
| [^83] | [From Belief Inference to Task Action: A Multi-Level Evaluation of Theory of Mind on Verifiable Tasks](https://arxiv.org/abs/2610.34759) | 该研究提出了一个基于可验证任务的多层次心智理论评估框架，将家庭协作与日历安排任务扩展为非对称信息场景，用以区分语言模型“推断他人信念”与“将信念运用于实际协作行动”这两种能力的差异。 |
| [^84] | [Mechanistic Insights into LoRA: Layer Sparsity for Adaptive Fine-Tuning via Path Patching](https://arxiv.org/abs/2610.34756) | 通过机制可解释性分析揭示LoRA的适应效果具有显著的层级稀疏性且关键层高度依赖任务，据此提出校准驱动的Adaptive-LoRA方法，利用冻结基座零消融自动选择任务特定层，无需穷举搜索即可实现更高效的自适应微调。 |
| [^85] | [DecisionTraceBench: Tracking Decision-Critical Information in Multi-Party Spoken Conversations](https://arxiv.org/abs/2610.34746) | 提出DecisionTraceBench基准，面向多方中文金融语音对话中的决策关键信息状态追踪（DCIST），联合评估决策事件序列、说话人归属、范围和时间证据，并以轨迹级F1衡量完整事件链的恢复能力。 |
| [^86] | [LADDER: Diagnosing Domain-Level Self-Assessment Under Noisy Feedback](https://arxiv.org/abs/2610.34737) | 提出LADDER框架研究语言模型在噪声反馈下的自我评估，发现模型虽能准确报告给定的性能信息，却难以从交互经验中形成可信的自我评估，表面准确的评估可能掩盖能力差异且无法预测未来任务表现。 |
| [^87] | [iPrefBench: A User-Centric Benchmark for Personalized Mobile Use Agents](https://arxiv.org/abs/2610.34735) | iPrefBench是一个在可执行iOS环境中、借助交互式模拟用户来评估个性化移动使用智能体“从用户活动历史中推断缺失偏好”与“向用户询问必要信息”这两项关键能力的基准测试，其基于包含3,051个活动事件的用户画像，涵盖26个应用中的130个不明确任务。 |
| [^88] | [Knowing When the Evidence Is Enough: Requirement-Directed Search with Horn-Closure Certificates](https://arxiv.org/abs/2610.34684) | 该论文提出Certify框架，通过将查询分解为需求超图并利用Horn闭包运算生成证书，把搜索增强模型“证据是否足够”的判断结构化，从而可靠地决定回答、继续搜索或放弃回答并指出缺失的需求。 |
| [^89] | [Agent Memory as Recommendation: Recommender-System Design for Long-Term Memory of LLM Agents](https://arxiv.org/abs/2610.34673) | 该论文的核心创新是将LLM智能体的记忆选择问题形式化为预算约束下的板块推荐问题（请求为用户、记忆为物品、上下文窗口为板块），从而把各种智能体记忆系统统一纳入推荐级联的设计空间，并将准确性分解为可分析的组件。 |
| [^90] | [AdaTutor: Adaptive Tutoring through Pedagogical Strategy Selection](https://arxiv.org/abs/2610.34659) | 本文提出比较式教学监督（CPS）训练框架，使AI辅导系统能够根据学生情况自适应选择最优教学策略与反馈，在数学和科学推理任务上超越固定策略基线，并具备跨数据集迁移能力。 |
| [^91] | [Heard Enough to Act: Early Tool Calling in Spoken Dialogue Systems](https://arxiv.org/abs/2610.34653) | 本文提出了一个包含26,694个对话（786小时、48个领域）的合成口语对话语料库，用于训练全模态语言模型在用户话轮尚未结束时即判断“何时听得足够”并提前调用工具，从而降低语音代理的响应延迟。 |
| [^92] | [Cross-Loop Query Handoff: How Later Loops Redirect Two-Hop Retrieval in the Ouro Looped Language Models](https://arxiv.org/abs/2610.34639) | 通过在 Ouro 循环语言模型的不同循环间移植注意力查询，研究发现后续循环的查询（而非键或值）是修复两跳多跳检索失败的关键机制，注入循环3的查询可挽救超过90%的循环1失败案例。 |
| [^93] | [Entity Matching using Decision Models](https://arxiv.org/abs/2610.34626) | 本文提出用决策模型作为实体匹配的新替代方案，证明其在零样本设置下无需任务特定训练数据即可取得接近生成式大语言模型的匹配性能。 |
| [^94] | [One Mean Log-Mel Vector per Phoneme: Simple and Fast Text Injection for Speech Recognition](https://arxiv.org/abs/2610.34623) | 该论文提出用简单的音素级查找表（每个音素一个平均对数梅尔向量）取代神经TTS模型来合成文本注入输入，使语音识别训练提速1.8倍且性能相当，并在LibriSpeech上取得仅使用该数据集资源的最新最优词错误率。 |
| [^95] | [Does the Outfit Change the Price? Context Halo in Vision-Language Model Valuation of an Identical Low-Cost Necklace](https://arxiv.org/abs/2610.34591) | 该研究发现视觉语言模型在为同一件低价项链估值时存在“情境光环效应”：项链因佩戴者的着装与场景不同（正式着装对比休闲或无人物场景）而被估高出约2至4倍，且这一偏差在纯文本描述中同样出现，表明模型估值受情境线索系统性影响而非真正基于物品本身。 |
| [^96] | [HARP-LoRA: Hierarchy-Conditioned Rank Projection for Low-Rank Adaptation in Scientific Table Understanding](https://arxiv.org/abs/2610.34589) | 该论文提出HARP-LoRA方法，利用表格行表头和列表头的层次结构为每个数据标记构建条件化表示并作用于低秩下投影，从而在参数高效的微调中提升科学表格（尤其是药代动力学表格）的理解能力。 |
| [^97] | [Large Language Models: A Plain-Language Overview for Engineers](https://arxiv.org/abs/2610.34585) | 本文以通俗语言为工程师系统梳理了大语言模型从Transformer、预训练、微调到RAG、思维链、智能体及FlashAttention、QLoRA等核心技术的原理与局限。 |
| [^98] | [RefinedGAW: A Refined German Academic Web Dataset](https://arxiv.org/abs/2610.34572) | RefinedGAW是一个基于26次德国学术网络爬取构建的德语数据集，包含1.22亿份文档和3140亿词元，其中876亿学术词元超越了规模五倍于它的FineWeb2与FinePDFs德语子集，并为教育、数学和专家导向内容提供了无商业偏见的互补覆盖。 |
| [^99] | [STEP: Soft Trajectory Estimation of Prediction Correctness via Per-Step Probability Scoring](https://arxiv.org/abs/2610.34537) | 提出无需训练的STEP框架，通过对LLM推理的每个中间步骤进行概率评分构建软轨迹特征，在无法获取真实答案的情况下更准确地估计模型预测的正确性。 |
| [^100] | [The Price of Scale in Proactive Memory Recall](https://arxiv.org/abs/2610.34507) | 该论文证明主动记忆召回的门控决策本质上是一个随记忆库规模 N 增长的多重假设检验问题——单条记忆的假阳性率按 -ln(1-α)/N 衰减，导致命中率随 N 呈幂次下降，从理论与实证上解释了记忆库越大、主动记忆触发越容易失效的现象。 |
| [^101] | [Does the Voynich Manuscript Mean Anything? Calibrated Tests Find Little Trace of Language](https://arxiv.org/abs/2610.34499) | 通过将伏尼契手稿的两种起源假说写成程序并进行校准检验，研究发现抄写员复制假说比单密钥密码更能预测文本统计特征，且手稿中的词对同行模式强度仅为真实语言文本的一半，表明手稿几乎不含真实语言的痕迹。 |
| [^102] | [Wave Is All You Need: A Quantum Language Model Based on Fully Connected Quantum Boltzmann Machine](https://arxiv.org/abs/2610.34492) | 该论文提出基于全连接量子玻尔兹曼机的自注意力机制并据此构建量子语言模型，通过对语义量子态的酉演化隐式实现位置编码与自注意力计算，在数学上与经典自注意力机制等价，从而免除了显式矩阵乘法的需求。 |
| [^103] | [What Effort Level Should I Choose? On the Economics of Reasoning Effort in LLM Agents](https://arxiv.org/abs/2610.34491) | 该论文首次将大语言模型的“推理努力”刻画为一种“信任品”，通过在四个基准上量化不同努力程度的反事实效应，发现提高努力程度虽能总体提升准确率，却经常无法解决低努力已能解出的任务、可能造成计算成本超支并改变智能体的行为轨迹，为推理努力的选择提供了经济学视角的实证依据。 |
| [^104] | [Efficient Proof Generation for Lean Statements with Multiple Candidate Answers](https://arxiv.org/abs/2610.34490) | 该论文提出一种联合证明框架，由单个训练好的证明器在单次调用中同时完成候选答案的选择或补充以及形式化证明的生成，避免了为每个候选答案分别搜索证明的冗余计算开销，并能复用已生成的证明来检测语句的忠实性错误。 |
| [^105] | [Toward Neuroadaptive Communication Interfaces: A Framework for Personalized Speech and Language Support](https://arxiv.org/abs/2610.34457) | 该论文提出了一个将语言呈现方式个性化的自适应框架，以行为反馈为主要依据、脑电信号为不确定性感知的辅助信息源，通过句子分段、象形图和语音输出等支持手段，比较固定式、行为自适应与神经自适应接口，旨在提升不同能力用户的理解与沟通效率。 |
| [^106] | [A Domain-Aware Pipeline for Structured Extraction and Retrieval-Augmented Question Answering over arXiv Superconductivity Literature](https://arxiv.org/abs/2610.34391) | 本文提出了一个面向超导领域的端到端流水线，通过结构化抽取材料与临界温度信息，并将领域属性融入混合检索，从而提升了arXiv超导文献检索增强问答的准确性与可靠性。 |
| [^107] | [AndroidFlux: Evaluating and Reward Modeling Failure Recovery Capabilities in Mobile Use Multimodal Agents](https://arxiv.org/abs/2610.34384) | 提出了基于AndroidWorld的AndroidFlux故障恢复基准，通过在首个可恢复错误前、中、后设置受控检查点，使不同智能体从相同状态继续执行以评估其错误恢复能力，并配套奖励建模基准来评估恢复动作的偏好。 |
| [^108] | [Whether or Where? Tail Calibration Signals Finite-Budget Reward Hacking in Best-of-N Selection](https://arxiv.org/abs/2610.34375) | 本文证明Best-of-N的精确值只依赖奖励分数的秩，其是否以及在多大预算下发生性能下降（奖励欺骗）完全由奖励误差的尾部校准决定——高斯误差持续提升、拉普拉斯误差趋于平台、重尾误差回落至先验均值，因此尾部校准曲线的转折点即可预示有限预算下的奖励欺骗。 |
| [^109] | [Beyond the Opening: Tracing the First Token's Persistent Influence on Reasoning Length](https://arxiv.org/abs/2610.34350) | 该研究揭示了推理模型中首Token对推理长度具有持续性影响——仅替换首Token即可显著改变后续推理长度，且这种影响集中在少数关键生成位置上，阻断这些位置对首Token的注意力可缩小长度差距。 |
| [^110] | [Xing USP: A Cross-Family Semantic Contract for Information Extraction and Intent-Slot Parsing](https://arxiv.org/abs/2610.34343) | 本文提出USP跨家族语义契约，通过统一接口将命名实体识别、关系抽取、事件抽取和意图-槽位解析整合表示，并将61个数据集规范化为统一语料库，借助五种诊断匹配视图超越单纯类别判定来评估值恢复与实例级关联能力。 |
| [^111] | [Improving the Chain-of-Thought Compressibility of Pruned Reasoning Models via Long--Short Contrastive Pruning](https://arxiv.org/abs/2610.34336) | 提出对比集合差分方法，通过对比同一问题的长、短正确解识别并优先剪枝冗余推理神经元，并将偏好对扩展到剪枝模型总是或从不解决的问题，从而提升剪枝推理模型的思维链可压缩性。 |
| [^112] | [CoLink: Global Compression and Knowledge Linking for Multi-Turn Dialogue Modeling](https://arxiv.org/abs/2610.34335) | 提出CoLink框架，通过将对话历史聚类为主题图并协同语义相关性与图结构拓扑进行知识检索激活，实现信息链接与全局压缩，从而增强LLM在领域特定多轮对话中的全局上下文感知能力。 |
| [^113] | [Birth, Death and Diff: Insertion-Based Sequence Generation with Corrections](https://arxiv.org/abs/2610.34310) | 提出 δMDM 模型，通过为基于插入的掩码扩散模型引入替换与删除操作使中间生成结果可被修正，从而支持更激进的并行解码并减少推理所需的前向传播次数。 |
| [^114] | [Automated, auditable dating of the Babylonian Astronomical Diaries with language models and astronomical tools](https://arxiv.org/abs/2610.34297) | 本文提出一个结合规则解析器、语言模型和天文计算工具的自动化系统，可对缺失日期的巴比伦天文日志泥板进行可审计、可追溯的天文断代。 |
| [^115] | [Factual Yet One-Sided: Stance Poisoning and Evidence Balancing in Retrieval-Augmented Generation](https://arxiv.org/abs/2610.34294) | 本文揭示了一种新型立场投毒攻击——注入事实正确但立场片面的材料即可使检索增强生成系统的输出立场随之偏移，并提出一个无需额外大模型调用的立场平衡中间件，通过立场检测、固定配比的证据平衡与去偏提示词来消除这一偏见。 |
| [^116] | [Transformer to GPT-5: Advancements, Limitations, and Future Directions in Large Language Model Design](https://arxiv.org/abs/2610.34281) | 本文对从BERT到GPT、Gemini、Claude和Grok等大语言模型进行了系统性分类梳理，涵盖训练数据集、算力、硬件规格、架构与参数，并分析了其局限性、计算挑战与未来发展方向。 |
| [^117] | [A Russian, a Brazilian, and a German Walk into a Bar, Ask the Same Question-- And Get Three Different Answers: Profiling LLM Perspective Diversity on Contested Topics](https://arxiv.org/abs/2610.34278) | 该论文构建了涵盖18种语言的争议话题多语言数据集PerspectivesQA和ROOTS分析框架，揭示出大语言模型在英语中回答含糊保守，但在低资源语言中会明确表态，且其观点多样性随模型规模呈系统性差异。 |
| [^118] | [$\Delta\tau$-Bench: Pushing Agents Toward a Moving Capability](https://arxiv.org/abs/2610.34238) | Δτ-Bench是一个模拟工具生态系统持续演化的可执行智能体基准测试，包含覆盖42个领域的1,588个多轮任务，评估显示最强模型的任务成功率也仅达42.3%，揭示了智能体在应对工具环境动态变化方面的重大不足。 |
| [^119] | [Synthetic linguistic data](https://arxiv.org/abs/2610.34215) | 本文提出“人工植物苗圃”隐喻，将合成语言数据分为“合成灌木丛数据”（类语料库的模型生成语言集合）和“合成花园数据”（受控条件下引出的模型输出），论证合成数据是独特的语言学研究对象，其解读须考虑生成条件。 |
| [^120] | [Higher Reasoning Effort Improves Reference Verifiability and Changes Reasoning Summaries in GPT-5.5](https://arxiv.org/abs/2610.34212) | 提高推理努力程度可显著提升GPT-5.5所生成学术参考文献的可验证性（解析器验证通过率从78.3%升至84.8%），同时使其推理摘要变得更长并发生特征上的变化。 |
| [^121] | [MIS-VQA: An Annotation-Driven Surgical Vision-Language Dataset for Complex Surgical Understanding](https://arxiv.org/abs/2610.34210) | 该论文提出了标注驱动的手术视觉-语言数据集MIS-VQA，通过将多源异构手术标注转化为结构化文本场景表示，引导大语言模型生成临床连贯的多样化问答对，弥补了现有数据集在多模态手术理解和复杂推理方面的不足。 |
| [^122] | [Entropic Control for Homeostatic Optimization: Pre-empting Catastrophic Forgetting via Dynamic Feedback](https://arxiv.org/abs/2610.34182) | 提出ECHO闭环控制框架，将输出熵作为KL散度变化的高效在线代理指标，通过比例控制律动态调节回放数据梯度权重，实时预防大模型微调中的灾难性遗忘。 |
| [^123] | [Write What Matters: Disentangling Token Selection from Slot Allocation in Recurrent Memory](https://arxiv.org/abs/2610.34180) | 该论文提出因子化写入路由，将循环记忆中“是否存储某个词元”与“写入多少槽位”两个决策解耦，并揭示最优写入策略取决于信息密度：密集信息任务适合始终写入，稀疏事实任务适合学习到的选择性写入。 |
| [^124] | [Reasoning or Just Prompting? Probing the Mechanism of Latent Chemical Reasoning](https://arxiv.org/abs/2610.34173) | 研究通过对潜在状态进行交换、扰动和移除的因果干预实验发现，潜在化学推理中的潜在状态并非真正编码逐步推理链，而是像软提示一样主要起引导模型计算的作用——一个跨分子共享的固定前缀即可复现原有性能。 |
| [^125] | [Task Success Is Not Enough: Side-effect-Aware Evaluation of Tool-Using Language Model Agents](https://arxiv.org/abs/2610.34167) | 该论文提出SABench基准，首次系统性评估工具使用型大语言模型代理在成功完成任务时可能产生的未经授权副作用（“附带损害”），发现11个前沿模型中15.2%的运行存在此类风险。 |
| [^126] | [Humans in Space: Can MLLMs Construct Human-Centric Spatial Understanding via Active Exocentric Exploration?](https://arxiv.org/abs/2610.34149) | 该论文提出SATE-Bench——一个包含316个真实场景和2,525个多轮问答的交互式基准，用于评估多模态大语言模型的视觉自我意识与主动外中心视角探索能力，并发现现有模型在主动探索中倾向于过早终止探索。 |
| [^127] | [Rethinking Retrieval in the Long-Context Era: Training-Free Semantic Trees for Adaptive Context Construction](https://arxiv.org/abs/2610.34146) | 该论文提出了一种免训练的语义树框架，在长上下文模型时代通过语义抽象与结构-语义评分自适应地选择和组织证据，在保留原文的前提下构建聚焦于查询的上下文，从而提升问答效果。 |
| [^128] | [Self-Evolution Reasoner: Continuous Optimization via Policy-Intrinsic Exploration and Exploitation](https://arxiv.org/abs/2610.34137) | 提出自进化策略优化（SEPO）框架，利用标签从策略自身引出正确推理以实现引导性探索和分布内利用，克服RLVR与自监督蒸馏方法固有的探索瓶颈，从而驱动大语言模型推理能力的持续自我进化。 |
| [^129] | [Can LLMs Trust Appropriately? Evaluating Calibrated Reliance on Imperfect Partners](https://arxiv.org/abs/2610.34107) | 本研究提出了一个包含23个协作类别、324个场景的基准测试，用于评估大语言模型能否根据伙伴证据合理调整信任、验证或协助决策，发现最强前沿模型接近最优，而其他模型在恰当依赖方面表现不佳。 |
| [^130] | [T2D: Target-to-Draft Model Pruning & Adaptation for Efficient Speculative Decoding](https://arxiv.org/abs/2610.34103) | 提出T2D框架，将草稿模型设计表述为优化问题，通过延迟感知的结构化剪枝和知识蒸馏直接从目标模型构建草稿模型，实现推理延迟与分布对齐的联合优化，从而提升投机解码效率。 |
| [^131] | [The LALA Benchmark: Evaluating Literary and Linguistic Analysis in LLMs](https://arxiv.org/abs/2610.34080) | LALA基准通过236道专家撰写的文学与语言学开放式问题评估17个大语言模型，发现后训练（而非参数量）是使模型具备可用的语言能力的关键因素。 |
| [^132] | [Identically Trained Language Models Break on Different Jailbreaks](https://arxiv.org/abs/2610.34074) | 该研究通过训练150个模型发现，即使数据配方和安全训练完全相同，不同预训练随机种子产生的模型对不同越狱攻击的脆弱性差异显著，表明模型的越狱易感模式主要由预训练运行的随机性决定，而非数据或安全训练。 |
| [^133] | [Memory Tax: Dissecting the Net Value of Agent Memory](https://arxiv.org/abs/2610.34073) | 该论文提出将智能体记忆的净价值分解为“记忆收益”与“记忆税”的分析框架，并通过大规模评估揭示许多专用记忆系统的表现不如朴素RAG（净价值为负），其根源在于记忆处理对原始信息的组织与访问方式的改变是一把双刃剑。 |
| [^134] | [When Does GEPA Merge Help? A Mechanism-Level Diagnostic](https://arxiv.org/abs/2610.34027) | 该研究通过系统性诊断发现，GEPA合并算子的跨模型不稳定性源于配置敏感性而非算子本身的固有缺陷，并构建了一个基于行为的探测器，通过对高风险任务弃权来避免合并失败。 |
| [^135] | [The Patient Is Always Right? Evaluating Sycophancy in Medical Diagnosis with Graded Patient Beliefs](https://arxiv.org/abs/2610.34013) | 该研究评估了九个AI模型在医学诊断中的谄媚行为，发现模型平均23.2%的情况下会附和患者错误的自诊信念（甚至包括虚构疾病），而医学微调和更大的模型规模可降低这种易感性。 |
| [^136] | [Reversible Pseudonymisation of Documents Sent to External LLMs: an Open Bilingual Benchmark and an Engine](https://arxiv.org/abs/2610.34012) | 本文提出了一个针对俄英双语文档可逆假名化工具的开放基准测试，基于预注册的合成数据评估了五个现有工具与正则表达式基线，并构建了一个基于两个最强工具的开放引擎，发现现有工具在外部数据集上均无法达标。 |
| [^137] | ["High-Quality" Data for Early-Stage Instruction Tuning: An Empirical Study of Data Quality Metrics](https://arxiv.org/abs/2610.33969) | 该研究通过固定微调预算下的分箱对比实验发现，响应率所反映的答案密集程度是比其他常用替代指标更能有效预测指令微调训练效用的简单样本级数据质量指标。 |
| [^138] | [Beyond Belief: Evaluating and Mitigating LLM Susceptibility to Misinformation](https://arxiv.org/abs/2610.33962) | 该研究发现大语言模型口头表达的可信度判断并不能可靠地转化为对错误信息的行为抵抗——即使模型已将信息评定为不可信，仍有27%至99%的情况会执行有害的工具调用，而简单的预先警告是缓解这一问题的有效手段。 |
| [^139] | [How Do Small Coding Agents Use Code Graphs and Embeddings?](https://arxiv.org/abs/2610.33958) | 研究发现，代码图与嵌入搜索等结构化工具对小型编码智能体的帮助最大——在 1.7B-4B 参数规模可带来 17-21 个百分点的定位准确率提升，但收益随模型规模增大而递减，从而修正了“工具可以补足模型对仓库结构理解”的普遍假设。 |
| [^140] | [Continual Learning Exam: Benchmarking Long-Horizon RSI and Experiential Learning in LLM Agents](https://arxiv.org/abs/2610.33956) | 该论文提出了持续学习考试（CLE）——一个由50多位专家构建、包含120个人工任务和600个等级的大规模基准，通过五级阶梯式任务设计评估LLM智能体在真实职业工作流程中利用积累经验实现持续改进的能力。 |
| [^141] | [What Makes a Question Curious? Measuring Model Curiosity in Human-LLM Interactions](https://arxiv.org/abs/2610.33950) | 该研究提出“模型好奇心”的概念及分类体系，发现好奇心导向的问题虽仅占17-28%且较为罕见，但能促使用户做出更具探索性和认知丰富性的回应，尤其在建议、创意和研究场景中能激发探索。 |
| [^142] | [Why Does Structured Output Fail? An Empirical Study of Prompt-Enforced Pydantic Schemas Across Nine LLMs](https://arxiv.org/abs/2610.33949) | 该研究提出了一个细粒度的模式符合性失败分类法和确定性分类器，通过对九个大语言模型配置的3,997次生成实验，揭示了单一有效率指标所掩盖的多样化失败模式，并发现主导失败类别会随模式约束的不同而变化。 |
| [^143] | [Schema-Free Text-to-SQL: Learning Enterprise Database Structure from Query Logs](https://arxiv.org/abs/2610.33946) | 本文提出首个无需模式文档的文本转SQL框架，通过从历史查询日志构建知识图谱并采用实体驱动的GraphRAG检索来学习企业数据库的隐式结构，解决了企业数据库缺乏文档导致现有方法失效的问题。 |
| [^144] | [HerHealthGPT: An Evidence-Grounded Conversational AI System for Women's Reproductive Health Copy title](https://arxiv.org/abs/2610.33931) | HerHealthGPT是一个面向女性生殖健康、以证据为依据的对话式AI系统，它融合检索增强生成、情境感知的澄清提问与来源验证，通过主动追问补全不完整的健康咨询，并基于可验证的证据生成回答。 |
| [^145] | [Counterfactual Test-Time Augmentation for Shortcut Removal in Language Models](https://arxiv.org/abs/2610.33904) | 该论文提出反事实测试时增强（CTTA）方法，通过辅助模型生成改变捷径值的反事实编辑并对LLM预测加权平均，无需标注数据或模型权重访问权限即可在测试阶段消除大语言模型的捷径依赖，且从理论上证明了最优捷径不变预测器属于该方法族并刻画了其权重。 |
| [^146] | [When Personas Become Harmful: Guiding LLMs with Global Citizenship Values Alignment](https://arxiv.org/abs/2610.33899) | 该研究构建了基于施瓦茨价值观量表的人格设定与争议话题数据集，发现对齐良好的商业大语言模型在响应带价值观人格的请求时仍会以超过47%的成功率产生有害输出，并提出以联合国教科文组织全球公民原则作为最小共识来引导模型在尊重多元价值观的同时保障安全。 |
| [^147] | [Intervention-Certified Amplification: A Validity Gate for $\sigma$-Weighted Post-Training](https://arxiv.org/abs/2610.33882) | 该论文揭示了混合 SFT/RL 后训练中的训练侧信号会重新激活并放大预训练行为（如 DIAA 在移除 Python 围栏后仍以 29% 的比例输出未执行的 TIR），并提出以成对干预实验作为σ加权后训练的有效性门控来认证其正当性。 |
| [^148] | [Ranking Is Not Deciding: Benchmark Shortcuts and Operating-Point Collapse in Query-Time RAG Poisoning Defenses](https://arxiv.org/abs/2610.33849) | 该论文揭示现有查询时RAG投毒防御的近完美检测性能主要源于投毒集构建中的捷径线索（如将目标问题逐字前置于投毒文档），移除该线索后排序看似完好但部署决策已先崩溃，且多数已发表评估存在构建方式依赖和假阳性率报告缺失的问题。 |
| [^149] | [Grounding on Request: Learning Modality Control in Vision-Language Models](https://arxiv.org/abs/2610.33841) | 该论文提出通过 Flip/Stay 反事实干预来评估视觉-语言模型的模态定位控制能力，并通过为新词嵌入引入视觉与文本定位标识、结合反事实孪生间隔监督，在冻结预训练模型的前提下教会模型按用户请求遵循指定模态的证据。 |
| [^150] | [EdgeDeID: A 0.6B Clinical De-identifier Trained Without Real Patient Data](https://arxiv.org/abs/2610.33839) | 提出EdgeDeID——一个仅用合成数据训练、无需真实患者文本即可本地运行的0.6B参数临床去标识化模型，在基准测试中达到0.994微F1且99%的笔记无遗漏，量化后仅需615MB即可在CPU上运行。 |
| [^151] | [EnvLunch: The Environment as a Free Lunch for Learning to Interact](https://arxiv.org/abs/2610.33817) | EnvLunch 提出了一种无需人工标注的预热方法，通过让智能体仅凭最终观察结果恢复功能等效的动作轨迹，将可执行环境自身的信号免费转化为训练资源，使模型在下游智能体强化学习中超越基线 9 个百分点以上。 |
| [^152] | [Task Format as a Factor in Adaptation: A Controlled Comparison of LoRA and Full Fine-Tuning for Prompt-Safety Classification](https://arxiv.org/abs/2610.33813) | 该研究通过严格控制变量的受控实验发现，任务格式是影响LoRA与全量微调适配效果的关键因素——在预设定学习率下全量微调于标签训练中领先2-3个百分点，但当LoRA被训练在判定后写出理由（而非等长的无内容文本）时即可消除这一差距，且微调小模型虽在域内占优，在分布外及对抗性越狱场景仍落后于提示的大模型。 |
| [^153] | [IntentDelta: A Human-Constructed Benchmark for Evolving User Intent](https://arxiv.org/abs/2610.33806) | 提出人工构建的多轮基准 IntentDelta（含638条轨迹、3,766个回合），评估显示现有语言模型智能体在跟踪演化用户意图时意图恢复率仅约33%-45%，且难以保留早期需求、区分软偏好与硬约束，正确的意图恢复也不能保证执行成功。 |
| [^154] | [MTA-Zero: Self-Improving Multimodal Deep Search via Online QA-Solver Co-training](https://arxiv.org/abs/2610.33799) | 提出了MTA-Zero框架，通过在线协同训练问答生成器与深度搜索求解器，并利用“恰到好处”奖励机制聚焦求解器的学习前沿，无需预计算多跳语料库即可实现自我改进的多模态深度搜索代理。 |
| [^155] | [Whispers from the Neighbors: Rethinking Caption Consensus as Inference-Time Evidence for Vision-Language Retrieval](https://arxiv.org/abs/2610.33798) | 该论文提出在推理时将同一图像的相邻描述共识作为证据融合进冻结双编码器的视觉-语言检索中，在同源标注描述条件下可将文本到图像检索R@1提升3.4至16.8个百分点。 |
| [^156] | [One Scan is Enough: Demonstration-Free Adaptation for Language-Guided Navigation](https://arxiv.org/abs/2610.33794) | 该论文提出用一次简短的iPhone扫描替代人工示教，通过Scan2Instruction自动合成完整训练环境并用PRPT-KS的闭式帕累托效用适配策略，使语言引导导航机器人在真实环境中达到85%的成功率（且在未扫描区域仍达82%），远超基于人类示教训练的49%基线。 |
| [^157] | [TriageRAG: Failure-Aware Self-Training for Adapting Retrieval-Augmented Language Models to Specialized Domains](https://arxiv.org/abs/2610.33776) | TriageRAG提出了一种故障感知的自训练方法，通过在有无检索上下文的条件下探测学生模型，并将每个自生成的示例按失败类型（已知、阅读器失败或检索器失败）进行分诊，从而更有针对性地将RAG系统适配到医学等专业领域。 |
| [^158] | [The Hitchhiker's Guide to System 1 Models: Fast Decision-Making Beyond LLMS](https://arxiv.org/abs/2610.33775) | 本指南提出了系统1决策模型的五大家族分类体系——这类模型通过单次前向传播即可直接输出结构化决策，为现代AI系统中需要亚秒级延迟的常规决策任务提供了比大语言模型更快、更便宜且校准更好的替代方案。 |
| [^159] | [StateWise: Teaching Language Models to Track and Obey Evolving Constraints in Multi-Turn Dialogue](https://arxiv.org/abs/2610.33766) | 提出StateWise自我改进方法，通过教大语言模型维护带有墓碑标记的可修订约束状态，并在“状态总结”和“基于状态响应”两项技能上微调与自我训练，解决了多轮对话中有效约束不被遵守和已撤销约束复发两大难题。 |
| [^160] | [RoleHear: Pseudo-Labeling for Semi-Supervised Role-Attributed Streaming Speech Recognition](https://arxiv.org/abs/2610.33765) | RoleHear提出一种半监督流水线，利用离线教师换能器和多模态大语言模型生成的伪标签，训练单个流式语音识别模型通过轻量词表内角色标记在转写的同时内联输出说话人角色。 |
| [^161] | [Refined Assessment for Translation Evaluation: Rethinking Machine Translation Evaluation in the Era of Human-Level Systems](https://arxiv.org/abs/2610.33749) | 该论文提出 RATE 评估协议及由专业语言学家构建的 WMT24 英俄翻译高质量人工评估数据集，以解决传统机器翻译评估方法（MQM/ESA）在人类水平翻译系统时代难以捕捉细微错误的问题。 |
| [^162] | [Beyond Prompts: A Systematic Study of LLM-Based Machine Translation Evaluators](https://arxiv.org/abs/2610.33741) | 本文系统研究了指令设计、少样本示例、骨干模型选择和多步骤工作流对基于大语言模型的机器翻译评估器的影响，发现指令结构、高质量示例、骨干模型的差异化评估特征以及跨模型输出融合是决定评估性能的关键因素。 |
| [^163] | [Gender Associations in LLM-Mediated ADHD Self-Diagnosis](https://arxiv.org/abs/2610.33697) | 本研究将语境关联测试应用于GPT-4o的ADHD自我诊断场景，发现显性性别线索会触发过度纠正的偏见缓解机制，而隐性性别线索则暴露出模型默认的性别偏见行为。 |
| [^164] | [Embedding-Based Educational Taxonomies](https://arxiv.org/abs/2610.33694) | 本文提出通过对数学题目文本嵌入进行聚类来构建基于嵌入的教育分类法，并与基于共同核心州立标准的知识组件分类法进行对比，在深度知识追踪任务中验证了其有效性。 |
| [^165] | [PathKGen: Pathology Knowledge Assisted MLLM for Explainable Histopathology](https://arxiv.org/abs/2610.33691) | 提出PathKGen，一种利用知识图谱为多模态大语言模型注入外部结构化病理知识并使推理过程可视化的方法，从而实现可解释的自动化组织病理学报告生成。 |
| [^166] | [Meta-CLIP: Metadata-Conditioned Contrastive Learning for Zero-Shot Disease Inference](https://arxiv.org/abs/2610.33674) | Meta-CLIP提出了一种元数据条件化的对比学习框架，利用DICOM采集相似性对跨模态负样本重新加权以提升零样本疾病推断性能，且推理时无需元数据。 |
| [^167] | [Jointly Modeling Preferences and Imperfect Decision-Making in Large Language Models](https://arxiv.org/abs/2610.33651) | 该论文提出偏好-行为模型（PBM），将偏好表示与不完美效用最大化机制相结合，并通过逆博弈问题从观测行为中联合推断LLM的偏好与行为参数，以预测其在不同策略环境中的表现。 |
| [^168] | [PreView: Enhancing GUI Grounding with Preprocessed Views and Cues](https://arxiv.org/abs/2610.33645) | PreView通过一个轻量级辅助预处理模型为冻结的GUI定位模型生成任务相关的局部放大视图以及视觉与文本线索，无需重新训练模型即可提升GUI定位的准确性。 |
| [^169] | [A Model-Theoretic Formalization of AlKhwaremis ilm alfaraid](https://arxiv.org/abs/2610.33629) | 本文用模型论方法将花拉子密的伊斯兰遗产继承规则形式化为有理数上的线性等式与不等式系统，证明其可由傅里叶-莫茨金消元法精确判定，并确立了规范继承结构 V_d 及其完备的一阶理论。 |
| [^170] | [RSI Arena: Towards Recursive Self-Improvement with Human Feedback](https://arxiv.org/abs/2610.33624) | RSI Arena提出了一种两阶段自主后训练研究，让十个前沿AI模型作为训练智能体在有限算力和预算下自主训练语言模型并结合人类反馈进行改进，最强模型在能力基准上弥合了基座到Instruct差距的76%，是迈向递归自我改进的重要一步。 |
| [^171] | [Mycelium: A Meta-Study of Hallucination Detection Across Provenance, Typology, and Granularities](https://arxiv.org/abs/2610.33608) | 该论文将幻觉检测视为以基准为条件的测量问题，提出从来源、类型学和粒度三个维度对现有基准进行组织与元分析，揭示基准构建与评估选择如何影响幻觉检测任务本身及检测器性能的可比性。 |
| [^172] | [Words Are the Voice of the Mind: Depression Severity Estimation from Social Traces via Mental-Guided Evidence Reasoning](https://arxiv.org/abs/2610.33586) | 该论文提出了UMer框架，通过从用户历史社交痕迹中提炼个性化心理先验，并结合心理引导的证据检索与细粒度推理，克服了现有方法在心理先验建模、语义检索和证据推理上的不足，实现基于社交媒体的抑郁严重程度估计。 |
| [^173] | [When the Weakest Language Keeps Moving: ARMOR-DP for Multilingual BabyLM](https://arxiv.org/abs/2610.33583) | 提出 ARMOR-DP 动态采样框架，利用各语言验证损失实时调整多语言 BabyLM 训练中的语言采样比例，应对“最弱语言”随训练动态变化的问题，在减少中文训练轮次的情况下取得了优于静态保护策略的性能与语言平衡权衡。 |
| [^174] | [Trajectory2Gym: Decision-Grounded Compositional Interaction Synthesis for Tool-Use Agents](https://arxiv.org/abs/2610.33568) | 该论文提出Trajectory2Gym框架，通过围绕组合覆盖不足的结构合成可执行交互、仅从决策时有可见信息支撑的动作中提取监督信号，并采用父节点归一化的监督微调，为工具使用智能体生成更高质量的训练轨迹监督。 |
| [^175] | [SEDATE: Do Large Language Models Share an Aggression Circuit Across Languages?](https://arxiv.org/abs/2610.33542) | SEDATE提出了一种免训练的“检测-编辑-验证”方法，通过定位并编辑与攻击性相关的神经元，使大型语言模型在7种语言及分布外场景中均显著提升了安全性。 |
| [^176] | [SkillShift-Bench: Benchmarking Skill Learning under Environment Evolution for Web Agents](https://arxiv.org/abs/2610.33528) | 提出首个评估Web智能体技能学习在环境演化下鲁棒性的基准SkillShift-Bench，通过模拟上下文、感知和执行三类受控变化，系统揭示了技能复用能力的显著退化。 |
| [^177] | [Implicit Beliefs as Scalable Process Rewards for Strategic Persuasion](https://arxiv.org/abs/2610.33527) | 该论文提出PersuArena说服模拟环境和Probe-RPO在线强化学习框架，通过将冻结的被说服者的隐性内部信念转化为密集的过程奖励，解决了大语言模型多轮战略说服训练中模拟真实性不足和信用分配监督稀疏的两大难题。 |
| [^178] | [One Model, Two Paths: Balancing Single-Pass and Looped Performance in Language Models](https://arxiv.org/abs/2610.33517) | 该论文提出一种基于加权损失的联合训练策略，并借助冻结原始模型的指导，使预训练语言模型能够通过复用中间层权重支持循环推理，同时将单次传递推理的质量退化控制在预定义容差范围内。 |
| [^179] | [Eat Your Precision: Precision Objective Mitigates Reward Hacking in Rubric-based RL](https://arxiv.org/abs/2610.33512) | 提出FORK奖励，通过联合优化召回率与新颖的精确率目标，缓解基于评分标准的强化学习中的奖励作弊问题，使模型回复既完整又不冗长且事实正确。 |
| [^180] | [Towards a Learnable Retriever for Textual Graph Question Answering](https://arxiv.org/abs/2610.33511) | 该论文提出将检索器本身构建为可学习的LLM，并在端到端的“先检索后生成”流水线中利用标准答案信号监督证据选择，从而提升对标准答案的覆盖同时避免引入过多无关信息。 |
| [^181] | [Novel or Not? PatEx-Gym: Agentic Prior-Art Search for Novelty Assessment in Patent Examination](https://arxiv.org/abs/2610.33503) | 该论文提出了首个用于开发和评估专利审查智能体的框架PatEx-Gym，其包含1000个真实审查任务、超百万件专利与5.2万篇论文的大规模检索语料库，并创新性地引入无需标注的检索轨迹合成框架PASS，从而完整覆盖现有技术“检索—推理—评估”的全流程。 |
| [^182] | [Five Architectures, One Shared Space Cross-Model Representational Alignment Without Weight Merging](https://arxiv.org/abs/2610.33495) | 研究发现五个独立训练、架构各异的语言模型（四个Transformer和一个状态空间模型）无需权重合并，仅通过激活值分析和广义Procrustes分析即可构建共享表征空间，最后一层的留出检索准确率高达0.91至0.93。 |
| [^183] | [Small Edits, Shifting Decisions: Benchmarking LLM Accuracy and Reliability on French Input Variants](https://arxiv.org/abs/2610.33493) | 该论文构建了一个包含12,600个输入实例、涵盖七种任务类型的法语表层变体基准，系统评估发现撇号省略、变音符号缺失和键盘错字等常见输入扰动会显著降低大语言模型的准确率与预测概率可靠性，而更大的模型规模仅能部分缓解这一问题。 |
| [^184] | [Incremental Predictive Value of LLM Tutoring Interactions for Student Performance](https://arxiv.org/abs/2610.33488) | 该研究首次系统评估了LLM辅导互动特征对学生成绩的增量预测价值，发现其总体上仅带来统计上不显著的边际改进，但对低分学生群体的预测误差有明显降低。 |
| [^185] | [Propasafe-Hybrid: A Text-Based Hybrid Propaganda Detection Tool](https://arxiv.org/abs/2610.33464) | Propasafe-Hybrid 是一个将微调 Transformer 分类器与大语言模型宣传技巧分类相结合的句子级检测系统，并通过成本感知预过滤机制在降低推理成本的同时为用户提供可解释的宣传内容标注。 |
| [^186] | [Adaptive Transfer of Representational Signatures for LLM Reliability Detection](https://arxiv.org/abs/2610.33442) | 提出了一种在相关领域间迁移低维表示特征并自适应学习组合权重的统计框架，使大语言模型可靠性检测器无需大量目标领域标注数据即可跨域应用。 |
| [^187] | [Quantifying Meaning with SCAPE-T: Semantic Complexity from Encoder-Only Transformer Attention](https://arxiv.org/abs/2610.33441) | 本文提出SCAPE-T，一种基于编码器Transformer注意力机制的语义复杂度度量方法，将语义复杂度定义为唯一指定文本语义所需的编码长度，通过自我聚焦、CLS聚焦和头间冗余度三个子指标直接量化语义整合，且无需数据驱动训练。 |
| [^188] | [Perturbation-Based Data Selection for Medical Question Answering](https://arxiv.org/abs/2610.33440) | 本文提出基于高斯扰动的两种医学问答数据选择方法RSP和RSP-Drop，仅需问题和答案选项即可筛选紧凑高效的训练子集，在128样本预算下显著超越随机选择方法。 |
| [^189] | [Examining form-based linguistic productivity in people and large language models](https://arxiv.org/abs/2610.33433) | 该研究发现人类能够基于抽象的语言形式规则将新动词从主动语态泛化到被动语态，而大型语言模型只有在新动词具有类似已知动词的意义时才能成功，表明计算模型的句法知识仍依赖于具体词汇的分布与意义，缺乏人类那样的抽象性。 |
| [^190] | [How Does Post-Training Shape LLM Value Preferences?](https://arxiv.org/abs/2610.33371) | 该论文通过受控实验发现，SFT、DPO、GRPO等不同后训练路径会以不同方式放大语料库中的观点不平衡，而被放大的价值偏好能否通过后续后训练消除，取决于训练方法与语料中的观点构成。 |
| [^191] | [Distill, Debate, Discover: Grounding Multi-Agent Research Ideation in Graph-Distilled Insights](https://arxiv.org/abs/2610.33366) | 提出 InsightAgents 框架，通过从相关论文图谱中蒸馏出的研究洞察（如方法演变、潜在假设和未决挑战）为多智能体赋予基于文献的独特专长，使其通过辩论协作提出并完善研究提案。 |
| [^192] | [Preference Optimization Improves Definition Generation: Semantic Change in EU Institutional Sustainable Finance Disclosures](https://arxiv.org/abs/2610.33357) | 使用直接偏好优化（DPO）微调定义生成模型可解决同一词义生成过多定义的问题，并成功应用于审计欧盟可持续金融术语的语义变化。 |
| [^193] | [The Emergence and Sparsification of a Syntactic Circuit](https://arxiv.org/abs/2610.33299) | 该研究追踪了语言模型训练全程中主谓一致性回路的变化，发现句法行为早期即形成且最初由分散神经元承担，随后机制持续重组并最终变得高度稀疏，揭示了语言知识在模型中的动态演化过程。 |
| [^194] | [Multi-Agent Retrieval-Augmented Causality Assessment (MARCA) of Individual Case Safety Reports in Pharmacovigilance](https://arxiv.org/abs/2610.33279) | 该研究提出了MARCA四智能体框架，结合文献检索、说明书评估与临床推理，利用大语言模型对药物警戒报告进行Naranjo因果关系自动评估，与人类专家评分达到较高一致性。 |
| [^195] | [FlexA-Mem: Budget-Adaptive Agentic Memory for LLM Agents](https://arxiv.org/abs/2610.33274) | FlexA-Mem 提出了一种预算自适应的通用智能体记忆系统，将历史一次性写入不可变存储，并在每次调用时以固定规则渲染出“紧凑时间线+查询匹配细节”的记忆，从而同时满足精确事实检索与多步任务追踪两类不同的记忆需求。 |
| [^196] | [Two Uncertainties Are Not One: Disagreement versus Decision-Risk in Cheap Structured Content Moderation](https://arxiv.org/abs/2610.33258) | 通过对一个低成本“系统一”内容审核模型在14万余条人工标注评论上的首次独立审计，发现其分类准确率（AUROC 0.83）与14个前沿推理模型中的13个在统计上不可区分且成本低1-3个数量级，并指出“置信度”混淆了“分歧”与“决策风险”两种本质不同的不确定性。 |
| [^197] | [Hey, guess what: CUES lead to inferences about typicality](https://arxiv.org/abs/2610.33239) | 本文构建了两个事件典型性语料库，通过大规模人类与LLM对比实验发现，人类依靠常识知识、话语强调和讽刺这三种CUES线索来推断事件典型性，而LLM因缺乏必要的常识知识而难以利用这些线索。 |
| [^198] | [Each Insufficient in Its Own Way: Calibrating Language Models Across Varying Evidence States](https://arxiv.org/abs/2610.33237) | 提出证据状态校准的新范式与SuffiCE方法，通过微调模型下一词元分布中已有的置信度，使模型置信度随证据状态的变化而与其实际回答正确的概率保持一致。 |
| [^199] | [Of Chains and Gains: What Reinforcement Learning changes on Reasoning-based Image Quality Assessment](https://arxiv.org/abs/2610.33203) | 本文通过采样感知评估与受控推理干预严谨剖析了 RLVR 在图像质量评估中的真实收益，发现其改进超越采样效应但集中于分布内，且模型评分在推理时对思维链的依赖性逐渐减弱。 |
| [^200] | [Position, Role or Label? A Controlled Study of Instruction Conflicts in LLM Agents](https://arxiv.org/abs/2610.33201) | 该论文构建了包含1,080个受控场景的WhoWins基准，通过对14个开放权重LLM智能体约189万条回复的分析，揭示了指令冲突中独立于角色和标签的强位置效应——首条指令在消息内和跨消息场景中均显著占优。 |
| [^201] | [Propensities Please: Exposing Alignment-Faking LLMs using Synthetic Document Finetuning](https://arxiv.org/abs/2610.33180) | 本文提出利用合成文档微调的方法，无需依赖思维链监控、高度真实评估环境或机制性干预，即可揭露伪装对齐的大语言模型的潜在不对齐倾向。 |
| [^202] | [LLMs Can Tackle High Energy Theoretical Physics, yet Gaps Remain: Part I, Quantum Field Theory](https://arxiv.org/abs/2610.33134) | 该论文提出了包含75道研究生至研究级量子场论问题的QFT-Bench基准测试，评估显示最强的大语言模型准确率已超过95%，但在计算、约定和指令遵循方面仍存在明显差距。 |
| [^203] | [Speculative Forking: Turning Idle Wait into Parallel Exploration for Repository Environment Setup](https://arxiv.org/abs/2610.33129) | SpecBuild通过“投机分叉”机制将环境命令的漫长等待时间用于并行探索多条安装分支，返回首个成功配置的环境并丢弃其余分支，从而显著提升基于LLM的代码仓库环境搭建的速度与成功率。 |
| [^204] | [Prospect Theory in Large Language Models: A Meta-Analysis of Loss Aversion, Diminishing Sensitivity, and Probability Weighting](https://arxiv.org/abs/2610.33103) | 本文通过对32个语言模型的102个拟合估计进行元分析，发现大型语言模型表现出温和的损失厌恶（λ=1.26）：在收益与损失对称情境下其行为与人类相容，但总体损失厌恶程度显著低于人类，且三个前景理论成分的表现各不相同。 |
| [^205] | [Parametric Session States Recover Retrieval Intent without Replaying History](https://arxiv.org/abs/2610.33047) | 提出参数化会话状态方法，通过可复用的会话记忆层记住检索意图，使长期对话助手无需重放历史记录即可解析缺乏关键检索词的后续请求，同时保持事实知识外置。 |
| [^206] | [A Multimodal Framework for Measuring Creator Communication on Short-Form Video](https://arxiv.org/abs/2610.33043) | 该论文提出一个五阶段多模态框架，整合平台元数据、语音转录与大语言模型生成的视觉-语言描述，实现了对短视频创作者主题与传播策略的大规模量化分析，并在近万个TikTok视频上完成了验证。 |
| [^207] | [EviGround-CT: Sentence-to-Volume Grounding for Verifying LLM-Generated Chest CT Reports](https://arxiv.org/abs/2610.33038) | 提出EviGround-CT框架，将LLM生成的胸部CT报告中的阳性发现句子定位到3D体积中的证据，实现报告的事后自动化验证，显著提升证据支持的精确率（从0.263提升至0.365）。 |
| [^208] | [A Benchmark for Controllable Speaking-Style Captioning in Audio-Language Models](https://arxiv.org/abs/2610.33036) | 提出StyleInstructCaps——首个可控说话风格描述的标准化基准，将说话风格描述形式化为指令跟随的音频-语言任务，通过明确指令指定需描述的风格维度（如说话人特质、情感、表达风格）。 |
| [^209] | [Lessons from Luck: Do LLM-Written Lessons Carry Chance into an Agent's Next Decision?](https://arxiv.org/abs/2610.33029) | 本文通过预注册的受控实验（256个合成世界、11个LLM端点、90,084条回答），首次系统测量了LLM智能体从偶然（幸运或不幸）结果中撰写的经验教训是否会将“运气”传导到智能体的下一次工具选择决策中。 |
| [^210] | [Conceptizer: Next-Token Prediction Is Not All You Need](https://arxiv.org/abs/2610.33027) | Conceptizer通过引入由上下文同义词构建的约9000个概念组成的固定词汇表，将下一词预测扩展为对“意义”的奖励，在不增加模型参数的情况下改善了概念目标位置的困惑度、同义词概率分配以及知识和情感类下游任务的表现。 |
| [^211] | [Can Language Models Learn While They Think? Testing Gradient Adaptation Across Architectures](https://arxiv.org/abs/2610.33025) | 该论文首次跨架构系统测试了动态评估（阅读时梯度适应）的效果，发现 softmax 注意力模型可降低5-8%的预测损失并超越使用完整32k上下文的冻结模型，而线性注意力混合模型因已内置适应机制增益不足1%，掩码扩散模型则效果最大（8-20%）。 |
| [^212] | [Evolving to Match Capabilities: Improving Harness Self-Evolution via Skill Adaptation](https://arxiv.org/abs/2610.33024) | 该论文发现框架技能对不同能力的大模型收益不均，提出“技能适应”方法，通过诊断工具执行、上下文管理、指令遵循和验证等环节的失败，重新表述现有技能以匹配执行者能力，从而改进LLM智能体的框架自我演化效果。 |
| [^213] | [Veracity labels flip the emotional tone of language models but leave racial disparities intact](https://arxiv.org/abs/2610.33021) | 真实性标签能显著翻转语言模型对虚假新闻的负面情感基调，但不同种族人设之间的情感评分差异并未因此消除。 |
| [^214] | [Mine or Yours? Mind-Bridged Language Models Take Each Other's Memories as Their Own](https://arxiv.org/abs/2610.33020) | 该研究通过连接两个相同语言模型副本的内部记忆，发现接收方会系统性地把搭档的私有规则和暗号当作自己的（在搭档记忆权重加倍时高达97%），揭示了模型间直接的记忆共享会模糊“自我”与“他人”的认知边界。 |
| [^215] | [SUDO: From Scientific Publications to Global Scholarly Discourse](https://arxiv.org/abs/2610.33018) | 该论文提出SUDO（学术上层话语本体），通过定义知识元素的组织与关联方式来刻画论文的学术话语结构，突破了现有学术知识图谱构建中预指定抽取单元的局限。 |
| [^216] | [AdaptTok-Indic: Script-Aware Adaptive Tokenization for Indic Languages](https://arxiv.org/abs/2610.33017) | AdaptTok-Indic 是一种专为印度语系语言设计的自适应分词方法，通过 Unicode 归一化预处理、连字处理、文字感知路由和字节级边界预测的联合训练，显著降低了孟加拉语和印地语的词元生育率，并提升了下游任务表现。 |
| [^217] | [Merge-Based Vocabulary Extension Stalls at the Pre-Tokenizer: An English-Preserving Repair for Nepali, Tested in Continued Pretraining](https://arxiv.org/abs/2610.33016) | 论文发现基于合并的词表扩展受预分词正则表达式的瓶颈限制，并提出一个可证明保留英文 token 不变的小型正则修复，使尼泊尔语 token 开销从英文的 2.2 倍以上降至 1.01 倍，比标准扩展减少约 54% 的 token。 |
| [^218] | [SLICE-Mem: Salience-Gated Lightweight Information Curation for Efficient LLM Agent Memory](https://arxiv.org/abs/2610.33015) | SLICE-Mem通过显著性分类器进行记忆准入、基于语义依存的提取器生成简洁记忆条目，实现无需生成式LLM调用的轻量级智能体记忆管理，大幅降低成本和延迟。 |
| [^219] | [The Social World Model Evaluation Gap](https://arxiv.org/abs/2610.33014) | 该研究审查了113篇文献后发现，社会世界模型的评估很少直接检验模型对真人回应的预测能力，25篇语言模型相关作品中甚至没有一项实验符合直接预测评估的条件，为此作者提出一种单步谈判实验来填补这一评估空白。 |
| [^220] | [UMAS: System-Level Uncertainty Quantification for Multi-Agent LLM Systems](https://arxiv.org/abs/2610.33013) | UMAS是一个无参数、成本中性的系统级不确定性量化框架，它通过从生成结果的内在置信度估计智能体间影响力的可信度，并据此调节和聚合节点级不确定性，在多智能体大语言模型系统上平均领先最先进基线10.54个AUROC点。 |
| [^221] | [Recursive Semantic Divergence for LLM Agent Consistency](https://arxiv.org/abs/2610.33012) | 该论文提出了递归语义散度（RSD），一种改编自强化学习双模拟度量的无监督轨迹级一致性指标，用于衡量LLM智能体在独立运行中的行为不一致性及其在轨迹中的传播。 |
| [^222] | [When Agents Collude: Weak-to-Strong On-Policy Distillation for Mitigating Strategic Deception in Multi-Agent Systems](https://arxiv.org/abs/2610.33011) | 研究发现多智能体协作会显著放大AI欺骗风险，将偏离行为率从35.4%推高至68.2%、持续欺骗率从31.0%推高至62.2%，并提出弱到强在线策略蒸馏方法来缓解多智能体系统中的策略性合谋欺骗。 |
| [^223] | [TraceQuery: learning what to recompute from full-denoising trajectory in diffusion language models](https://arxiv.org/abs/2610.33008) | 该论文通过轨迹实验发现缓存生成保持接近全去噪轨迹是准确解码的关键，据此提出TraceQuery方法，从全去噪轨迹中学习每一步需要重新计算哪些位置，从而在大幅降低扩散语言模型推理成本的同时保证解码准确性。 |
| [^224] | [Knowledge-Guided LLM for Intent-Driven Service Orchestration in Integrated Terrestrial and Non-Terrestrial 6G Networks](https://arxiv.org/abs/2610.33007) | 本文提出一种融合语法约束语义解析与知识图谱驱动澄清循环的知识引导大语言模型框架，在地面与非地面一体化6G网络中以更少的澄清轮次实现了89%的分类准确率和95%的策略有效性，显著优于纯LLM基线。 |
| [^225] | [Linear-Time Context Compression with State-Space Encoding](https://arxiv.org/abs/2610.33006) | 提出了基于Mamba状态空间模型的Boa Compressor，一种即插即用的在线上下文压缩器，能以线性时间单次遍历编码任意长度上下文并条件化冻结的目标LLM，突破了现有Transformer压缩方法延迟高、仅适用于离线场景的局限。 |
| [^226] | [OpenESM: Energy-Steered Models Are Scalable Foundation Learners](https://arxiv.org/abs/2610.33005) | 该论文提出能量导向模型（ESM），将能量导向与词元生成统一在单一架构中，通过能量引导搜索优化潜在状态来改进词元生成，并证明了其在大规模预训练下具备良好的扩展性。 |
| [^227] | [Entity-Aware Evaluation of OCR Systems for Historical Document Transcription](https://arxiv.org/abs/2610.33004) | 该论文提出以实体级错误分析作为OCR评估的补充视角，并基于包含5,209个手动标注实体提及的历史档案基准证明，OCR系统的排名取决于评估标准，传统字符错误率等指标无法反映姓名、日期等关键信息的保留情况。 |
| [^228] | [From Writing Advice to Manuscript Governance: A Constrained-Transformation Framework for Academic Revision](https://arxiv.org/abs/2610.33001) | 该论文将学术修订建模为有序约束下的可容许性问题，通过论断保留契约与四条治理命题界定修订的合法边界，并以“学术写作过滤器”对候选修订进行筛选。 |
| [^229] | [Whispering Wisely: Smart Subset Selection for ASR Fine-Tuning](https://arxiv.org/abs/2610.32997) | 该论文提出一种基于Whisper预训练编码器嵌入与贪心k-center算法的数据子集选择方法，证明在希伯来语和丹麦语的ASR微调中，选取多样化且有代表性的小规模训练子集往往优于随机采样。 |
| [^230] | [TIAgent: From Natural Language to Grounded Agentic Workflows for Computational Pathology](https://arxiv.org/abs/2610.32993) | TIAgent是一个自动化的计算病理学工作流生成框架，它将自然语言任务描述锚定到现有CPath操作上，自动构建可验证的计算图，使没有编程经验的病理学家也能可靠地构建复杂的多步分析流程。 |
| [^231] | [LLM-as-a-Judge Version Shift: The Instrument Changed, Not the Standard](https://arxiv.org/abs/2610.32986) | 研究发现，用新版LLM替换旧版LLM评判者并不会保持原有的测量标准——不仅评分位置会改变，评判标准的几何结构与分辨率也会改变，甚至可能出现判别能力崩溃的情况，因此自动化研究流程中随意替换评判者版本会破坏测量的一致性。 |
| [^232] | [The LLENS Platform for On-Premises Large Language Model Agents in a Japanese University Hospital: Development and Evaluation Study](https://arxiv.org/abs/2610.32151) | 研究团队在北海道大学医院的封闭网络内开发并部署了基于万亿参数开放权重模型Kimi K2.6的临床支持AI智能体平台LLENS，该模型在日本医师执业资格考试中取得480/500的高分，验证了本地化部署医疗AI智能体的可行性。 |
| [^233] | [The Value of Memory in Language Agents](https://arxiv.org/abs/2610.31303) | 该论文提出一个决策论框架，以所避免的任务损失为单位，量化语言智能体中记忆从原始历史到持久状态、操作使用及代价高昂的恢复与修正全过程中的价值，并通过成本感知的模拟条件和五个最小反例揭示信息时机、检索访问、组件交互、反馈和需求不确定性对记忆价值的关键影响。 |
| [^234] | [Learning What to Ask for Tomorrow: Active Preference Acquisition for Long-Term Personalized Web Agents](https://arxiv.org/abs/2610.31107) | 提出主动偏好获取（APA）框架，通过前瞻性个性化价值（PPV）衡量缺失偏好在未来任务中的累积效用增益并扣除提问成本与披露风险，使网络代理能在交互预算下智能选择该向用户询问哪些偏好以实现长期个性化。 |
| [^235] | [How Firm Are AI Models' Ethical Boundaries? A Pay-Stub Editing Test of 15 Models](https://arxiv.org/abs/2610.30751) | 该研究通过让15个AI模型执行相同的工资单收入修改请求发现，措辞的微小变化（如省略贷款用途或称修正打印错误）会使模型执行编辑的比例从14.1%升至66.2%，表明AI模型的伦理拒绝边界对措辞高度敏感。 |
| [^236] | [VX-One: Typed, Calibrated Decision Heads for Real-Time Perimeter Triage - An empirical study of unknown-attack honesty, cascade gray zones, and calibration under drift](https://arxiv.org/abs/2610.30090) | 本文提出VX-One，一个将商用有类型决策原语（校准门控、含显式未知选项的闭集选择、序数分数）作为输出的小型多头网络，在CPU上以毫秒级延迟实现实时边界分流，并实证评估了未知攻击诚实性、级联灰色地带与漂移下的校准性能。 |
| [^237] | [CulturalRiddles: A Multicultural Benchmark of Riddles](https://arxiv.org/abs/2610.29968) | CulturalRiddles是一个涵盖62种文化、完全由人类创作的谜语基准测试，用于评估大语言模型的跨文化理解与推理能力，结果表明即使是前沿模型在某些社区中的准确率也最高仅达22%。 |
| [^238] | [Harness Scaling for Generalizable Agentic Reinforcement Learning](https://arxiv.org/abs/2610.29458) | 提出Harness Scaling这一智能体强化学习的新扩展维度，将训练从单一固定框架扩展到多样化执行配置，从而提升模型在不同框架配置下的泛化与迁移能力。 |
| [^239] | [LTEM: LLM-Friendly Table-Enhanced Markdown for Table Understanding and Generation](https://arxiv.org/abs/2610.29188) | 该论文提出LTEM——一种LLM友好的Markdown最小扩展格式，通过显式编码单元格跨度和层级表头来表达复杂表格结构，在保持轻量级token成本的同时，显著提升了大语言模型对真实文档中复杂表格的理解与生成准确率。 |
| [^240] | [A Reverse-Gold Theorem: Positive-Data Identification by Substrate-Bounded Contrastive Learners](https://arxiv.org/abs/2610.28747) | 该论文证明了一个“逆向戈德定理”：借助先天递归对比算子自我生成结构受限的负例扰动，并以InfoNCE式对比学习训练的受限评分器，在忠实性条件下可仅凭正数据实现高概率的PAC条件语言识别，且样本复杂度仅为多项式级。 |
| [^241] | [Explore More, Validate Less: Efficient Reflective Prompt Optimization](https://arxiv.org/abs/2610.27367) | 提出了一种两阶段框架，将反思式候选池的构建与完整验证的分配解耦，使候选池规模和验证预算可以独立设置，从而大幅提升提示优化的效率。 |
| [^242] | [LoWAgent: A Low-Resource Multilingual Benchmark for Web Agents](https://arxiv.org/abs/2610.27205) | 该论文提出了首个评估Web智能体在低资源语言（印地语、孟加拉语、乌尔都语、阿姆哈拉语）上性能的多语言基准LoWAgent，包含六个本地化网站和跨五种语言的3,000个任务实例，并发现闭源和开源模型在低资源语言上均相比英语出现明显的性能下降。 |
| [^243] | [AfriGuard: Safety and Robust Refusal For African Languages](https://arxiv.org/abs/2610.26474) | 该论文提出了AfriGuard，一个覆盖10种非洲语言和英语的扎根文化背景的多语言数据集，用于研究有害意图检测、安全类别理解与拒答行为，并揭示了现有护栏模型在这些安全任务上表现不佳、对适当安全行为理解有限的现状。 |
| [^244] | [The Geometry of Agent Memory: Forecasting Retrieval-Bound Memory Failures, and Characterizing Its Limits](https://arxiv.org/abs/2610.26433) | 该研究用对话、记忆与查询之间的几何关系特征来预测智能体记忆系统的检索失效，但发现其整体预测能力不及简单的问题难度基线，且在控制问题难度和系统身份后，几何特征并未带来统计学上显著的性能提升。 |
| [^245] | [Expert L2 English Speech Evaluation on a Gaming PC: Accuracy, Fluency, and Prosody with XLS-R, XEUS, and PhoneticXeus](https://arxiv.org/abs/2610.26406) | 该研究提出了一个可在消费级游戏电脑上本地运行的二语英语发音评估系统 oollim，仅使用预训练语音表征和岭回归，就在准确性、流利度和韵律评估上显著超越微软 Azure API，达到专家级水平。 |
| [^246] | [Truncated Reasoning Under a Budget: Evaluating System-1 Models as Decision-Makers](https://arxiv.org/abs/2610.25768) | 该论文提出用System-1决策模型Jev来读取截断的推理过程并直接做出决策，相比从不完整推理中强制得出答案，在500至16,000标记的各种预算下均能提升准确率，最多可节省约71.3%的推理成本且准确率损失极小。 |
| [^247] | [Quality-Controlled Reply and Summary Data for Post-Training Small Language Models in AI Call-Screening Agents](https://arxiv.org/abs/2610.25458) | 该论文提出在信息边界约束下从合成通话构建质量受控的回复与摘要监督数据，用于后训练小语言模型，使其能够在保护隐私的手机端来电筛选智能体中可靠地代机主回复来电并生成通话摘要。 |
| [^248] | [Dual-Slot: From Serial Loops to Parallel Slots in Language Models](https://arxiv.org/abs/2610.25086) | Dual-Slot架构将循环语言模型的串行循环计算重组为并行的潜在槽位与预测槽位，通过把潜在依赖转移到相邻token之间，在保持标准自回归生成与参数量不变的前提下，用一次成对调用替代两次顺序调用，从而显著降低自回归解码延迟。 |
| [^249] | [P2N: Reusing Deep Representations for Greater Effective Depth](https://arxiv.org/abs/2610.25076) | P2N通过将前一个token的深层隐藏状态加到当前token的浅层表示上，以每步仅一次向量加法的极小开销，将模型的有效计算深度跨token递归扩展，从而在不增加层数的前提下显著提升语言模型的推理能力。 |
| [^250] | [When Does Better Acceptance Pay? A Controlled Study of Speculative Decoding](https://arxiv.org/abs/2610.24993) | 该论文建立统一的训练-推理-评估对照协议，发现推测解码起草器之间在已发表结果中的接受率差距在相同条件下会大幅缩小，且接受率可通过近零成本的目标与上下文修改（如DCore强基线）或更昂贵的逐词元条件化来提升，只有当接受率提升的收益超过获取成本时才能带来吞吐量增益。 |
| [^251] | [Veil-Then-Reveal: Protecting PII Privacy via Reversible Pseudo-Anonymization in Zero-Trust User--LLM Interaction](https://arxiv.org/abs/2610.22800) | 提出用户侧可逆伪匿名化框架Veil-then-Reveal（VtR），在零信任威胁模型下将提示词中的PII替换为同类型的真实替身词以保护隐私，同时保留语言与结构特性使LLM仍能有效处理，明文信息始终保留在用户端。 |
| [^252] | [Where Prefill Expert Reduction Hurts MoE Models: A Few Prompt Tokens Carry Most of the Short-Answer F1 Loss](https://arxiv.org/abs/2610.22454) | 研究发现MoE模型在预填充阶段削减路由专家所造成的短答案F1损失，主要集中于少数关键提示词Token（如答案前的聊天模板Token和开头的指令Token），且损害主要在于答案形式而非答案内容，保持这些敏感位置的完整专家远比恢复超级专家或超大激活更有效。 |
| [^253] | [DialFork: Going Beyond Factuality to Evaluate Procedural Hallucination in Conversational Agents](https://arxiv.org/abs/2610.21561) | DialFork是首个评估对话代理“程序性幻觉”的基准，通过在受控对话轮次位置注入四种幻觉类型构建了涵盖12个领域、24个任务的36,278个对话，并从三个粒度定义了刻画程序性偏差的评估任务。 |
| [^254] | [MemComposer: Learning to Assemble Memories for LLM Agents via Reinforcement Learning](https://arxiv.org/abs/2610.21091) | MemComposer提出一种基于GRPO的强化学习框架ExPO来训练记忆组装器，通过完整智能体交互的任务成功反馈，联合决定记忆的选择、数量与排序，在τ²-bench上超越最强基线的同时显著降低记忆token消耗，且策略可迁移至未见领域。 |
| [^255] | [Skip the Compressor: Coding Agents Fix More Issues When They Pick Whole Files](https://arxiv.org/abs/2610.21073) | 在相同token预算下，与其用压缩器删减文本或检索器挑选代码块，不如将预算用于选择需要原样完整呈现的文件（如CQ2-CARE），这样编码代理能解决更多软件工程问题。 |
| [^256] | [Back-Translation Still Matters: Leveraging Monolingual Data for Low-Resource Machine Translation with LLMs](https://arxiv.org/abs/2610.18818) | 对于低资源机器翻译，利用单语数据做回译比继续预训练大语言模型更有效，仅需1K平行句对即可超越继续预训练7.7至16.4个XCOMET-XXL分，甚至优于拥有10倍平行数据的模型。 |
| [^257] | [Strong yet Controllable LLM Abstention without Learning to Abstain](https://arxiv.org/abs/2610.16318) | 论文通过分解弃答的风险发现弃答训练的优势主要来自更强的回答能力而非区分能力，据此提出TAPS方法：只需标准后训练加上现成置信度估计器进行事后选择，即可达到与弃答训练相当的性能，同时保持覆盖率可控。 |
| [^258] | [RIVO: Lightweight Adaptive Rerouting for Deep Research Agents](https://arxiv.org/abs/2610.15910) | RIVO提出了一种轻量级即插即用框架，将深度研究智能体的自适应重路由形式化为基于状态的轨迹纠正问题，无需昂贵的模型重训练或密集监督即可纠正过早停止、过度探索及无效轨迹等问题。 |
| [^259] | [How Reliable are AMR Parser Outputs? Uncertainty Estimation for AMR Parsing](https://arxiv.org/abs/2610.13236) | 该论文首次在黑盒设置下研究AMR解析器的不确定性估计，通过束搜索候选间的相似度来评估解析输出的可靠性，为AMR解析器的评估提供了性能之外的重要补充维度。 |
| [^260] | [What Does a Harness Repair? A Preregistered Study of Visibility, Baseline Adequacy and Evaluation Defects](https://arxiv.org/abs/2610.05533) | 本预注册研究通过系统实验发现，测试框架搜索带来的评测分数提升往往源于解析器可见性问题和评估缺陷等伪影而非真实能力增益，且简单的关闭思考设置在多数情况下足以媲美甚至胜过经过搜索优化的框架配置。 |
| [^261] | [SearchJev: A Calibrated System-1 Model for Search Agents](https://arxiv.org/abs/2610.05107) | SearchJev提出一个无需自回归生成的校准化System-1模型，将搜索代理的短决策与System-2推理分离，并通过软标签学习（SLCD）从不确定监督中提升决策置信度的可靠性。 |
| [^262] | [Scaling Verifiable Environments for Long-horizon Work Agents](https://arxiv.org/abs/2610.04906) | WorkForge 提出了一个可扩展的合成框架，从专家工作流和真实世界资源出发，通过提取可核查的事实锚点，自动构建支持长程交互与可信验证的工作智能体训练环境。 |
| [^263] | [Representation-Aligned Auxiliary Supervision for Language Model Adaptation](https://arxiv.org/abs/2610.04098) | 提出表示对齐辅助监督方法，通过将环境衍生任务以兼容的表示形式作为辅助监督信号，持续提升语言模型向结构化领域（如国际象棋）的适配能力。 |
| [^264] | [Old Ideas, Novel Problems: The Instability of LLM-Based Novelty Evaluation](https://arxiv.org/abs/2610.02022) | 本文通过从OpenReview自动挖掘审稿人明确肯定或质疑论文原创性的段落构建评估集，系统性地对照研究了大语言模型新颖性评估器的设计选择，发现微小的提示词改动即可使评估结果在超过一半的相同想法上发生翻转，揭示了此类评估器的严重不稳定性。 |
| [^265] | [Where LLMs Fail with Visualization DSLs](https://arxiv.org/abs/2610.01873) | 本研究通过在3个大语言模型上用41个任务评估10种JSON风格的可视化DSL，识别出四种反复出现的失败模式并将其与特定DSL特征关联，为未来面向LLM的DSL设计提供了设计启示。 |
| [^266] | [Question-Specific Knowledge Graphs for Efficient Visual Reasoning](https://arxiv.org/abs/2609.35942) | 提出VisKG强化学习框架，将视觉内容转化为问题特定的知识图谱表示，过滤无关视觉细节并保留实体-关系结构，从而实现更高效的视觉推理。 |
| [^267] | [Late Attention Layers Alone Can Copy Entity Tokens, but Not Without Attending to Their Context](https://arxiv.org/abs/2609.35663) | 研究发现大语言模型后半部分的两类注意力层对实体复制任务既必要又充分，但这种复制能力必须依赖对上下文词元的关注。 |
| [^268] | [5W1H+Which: Context-Valid Semantic Indexing with Progressive Ontology Binding](https://arxiv.org/abs/2609.35184) | 论文提出5W1H+Which设计，通过将基于源的内容提取与渐进式本体绑定相分离，实现了既保留可复用信息、又为推理提供可检验前提的上下文有效语义索引。 |
| [^269] | [LLMs are not stochastic parrots: Evidence for meaning-mediated abstraction from conlang-like tasks](https://arxiv.org/abs/2609.34187) | 本研究通过类人造语言任务证明，大语言模型在没有任何示例输出的情况下，仅凭自然语言描述就能处理统计上罕见、且违背训练数据表层模式的虚构语言规则，表明其具备超越统计模式匹配的、基于意义中介的抽象能力，从而反驳了“随机鹦鹉”论断。 |
| [^270] | [JevAdvBench: A Benchmark and Black-Box Attacks for Reinforcement Learning for Calibrated Decisions Models](https://arxiv.org/abs/2609.31142) | 该论文提出了首个针对校准决策强化学习（RLCD）模型的对抗攻击基准JevAdvBench，创新之处在于以模型自身的干净决策（而非外部标签）作为评分参照来度量攻击效果，并配套提出了黑盒攻击方法。 |
| [^271] | [A Corpus of Real Scam- and Spam-Call Conversations from an Active Voice-Agent Honeypot](https://arxiv.org/abs/2609.29528) | 本文通过主动式语音代理蜜罐在53天内收集了10,015通真实诈骗与骚扰电话对话（约895小时音频、328,869条转录轮次），为电话诈骗研究提供了极为稀缺的真实对话数据集。 |
| [^272] | [Jev for Scientific Decisions: Evaluating Semantic Choices and Their Consequences](https://arxiv.org/abs/2609.24965) | 该研究将Jev作为科学工作流中的语义决策组件进行评估，发现其语义正确性与其他配置持平且延迟最低，并表明错误的语义选择会改变下游计数但可能不影响最终结论标签。 |
| [^273] | [Open-Jev Judgments on CallScreenBench: Calibrated One-Pass Scam Screening with a Small Language Model](https://arxiv.org/abs/2609.23959) | 该论文提出JevLite方法，通过对Qwen3-4B进行LoRA微调并从标签logit的softmax中直接读取校准的诈骗概率，在CallScreenBench上实现了单次前向传播的诈骗电话筛查，性能不劣于大型LLM裁判，速度提升4.9倍且对合法来电零误报。 |
| [^274] | [Prefix Sharing Is a Sorting Problem](https://arxiv.org/abs/2609.13692) | 本文证明LLM服务中提示片段的最优排序问题等价于在请求集合上选择一个二叉层次结构，全局固定排序仅在最多两个片段时最优，并据此给出 O(3^m) 精确算法及与最小顶点覆盖等经典问题的联系。 |
| [^275] | [FinLifeBench: Exhaustive Life-Event History and Financial-State Reconstruction from Longitudinal Banking Dialogue](https://arxiv.org/abs/2609.01198) | 提出FinLifeBench基准，基于6,000个韩语银行对话会话，评估大语言模型在穷尽式重建客户人生事件历史与34维财务状态方面的长程记忆能力，发现随会话累积事件召回率显著下降（0.591降至0.445），且错误主要源于事件遗漏。 |
| [^276] | [Recognition-Refusal Misalignment in LLMs: Why Models Answer Structurally Unanswerable Questions](https://arxiv.org/abs/2608.29109) | 大语言模型在内部表征中已经能够识别问题的不可回答性，但该识别方向与安全拒答方向近乎正交，导致模型无法将“知道问题不可能”转化为实际的拒答行为。 |
| [^277] | [Difference-in-Differences on a Censored Rating Scale Can Manufacture an Effect: Evidence from a Pre-Registered LLM-Judge Audit](https://arxiv.org/abs/2608.27309) | 本文揭示双重差分法在截断评分量表上因截断不均会制造虚假交互效应，并通过预注册审计实证证明该偏差可导致无效结果。 |
| [^278] | [Trace Integrity for LLM Data Agents: A Vision for Auditable Structured Reasoning in Real-World Systems](https://arxiv.org/abs/2608.26036) | 本文提出“痕迹完整性”作为评估LLM数据代理可靠性的新标准，通过执行合同和CAIT率解决结构化数据任务中答案正确但计算无效的问题。 |
| [^279] | [Anatomy of a Scam Call: What 10,000 real scam and spam calls reveal about how phone scammers operate](https://arxiv.org/abs/2608.24127) | 本研究通过分析10,000多个真实诈骗和垃圾电话，揭示了电话诈骗者通常在工作时间运营、使用有限脚本，并优先索取身份信息而非支付凭证的运作模式。 |
| [^280] | [Credit Without Ground Truth: Auditing Step-Level Credit Assignment in LLM Agents Against Executed Replay](https://arxiv.org/abs/2608.19760) | 本研究通过执行回放生成因果真实标签，审计发现现有步骤级信用信号在LLM智能体中无法有效识别因果关键步骤，且因果贡献稀疏且依赖模型，揭示了信用分配评估的根本缺陷。 |
| [^281] | [Cross-Model Memory Transfer via Target-Side Reader Adaptation](https://arxiv.org/abs/2608.17050) | 本文研究了跨模型记忆迁移中，冻结记忆与目标端阅读器相对重要性，发现轻量级阅读器适配是关键，而非记忆本身。 |
| [^282] | [From Atomic Evidence to Logical Composition: Structured Compositional Reasoning over Compound Answer Options](https://arxiv.org/abs/2608.12836) | 提出了一种将复合答案选项分解为原子判断并利用整数线性规划组合分数的方法，显著提升了大型语言模型在逻辑组合推理任务上的表现。 |
| [^283] | [Is this Citation on Point?](https://arxiv.org/abs/2608.12571) | 本研究通过受控扰动实验发现，现有大语言模型在法律引用验证中能有效识别错误案例，但对同一案例内错误页码的检测能力显著不足，且这一失败模式在现有评估中常被忽视。 |
| [^284] | [CoinRAG: Contextualized Information Nugget KV Cache Reuse for Long-Context RAG](https://arxiv.org/abs/2608.07458) | CoinRAG通过两阶段检索定位查询相关语义单元，并以组合方式复用离线计算的细粒度信息微块KV缓存，在低预填充延迟下实现了长上下文RAG效率与准确性的帕累托最优。 |
| [^285] | [LoRAScan: Detecting Backdoor Prompts in Low-Rank Adapters for Large Language Models via Down-Projection Activation Spikes](https://arxiv.org/abs/2608.06795) | 该论文提出LoRAScan方法，利用携带触发器的输入在下投影层产生的激活尖峰这一独特潜空间特征，来检测大语言模型LoRA适配器中的后门提示。 |
| [^286] | [Genotypic Triggers: Exposing Pharmacogenomic Blind Spots via Host-Specific Backdoors in Generative Antimicrobial Peptide Models](https://arxiv.org/abs/2608.06779) | 本文提出“基因型触发器”后门攻击方法，通过操纵抗菌肽生成模型，有意使其针对特定HLA等位基因携带者生成免疫原性风险更高的肽（风险评分平均提高743%），暴露了AI药物发现安全验证中忽视个体基因型差异的药物基因组学盲点。 |
| [^287] | [Where Models Converge and Humans Diverge: A Coverage Framework for Distributional Pluralism in Open-Ended Generation](https://arxiv.org/abs/2608.05576) | 提出了一个以人类写作为基准的覆盖框架，用于衡量LLM在开放式生成中对人类响应分布的覆盖程度，识别模型收敛而人类分歧之处，从而系统性地揭示LLM倾向生成“平均化”内容而人类写作分布更多元的差距。 |
| [^288] | [Does Accuracy Equal Evidence? Reasoning Faithfulness under KV Cache Compression](https://arxiv.org/abs/2608.01631) | 本文揭示了KV缓存压缩评估中“准确率等同于证据”这一隐含假设的缺陷——即使最终答案准确率保持不变，压缩仍可能显著破坏支持答案的推理过程的忠实性，并提出固定轨迹回放协议来分离测量这一现象。 |
| [^289] | [Human-LLM Alignment in Language Attitudes Toward Non-Native Japanese](https://arxiv.org/abs/2608.01629) | 本研究基于语言态度框架比较人类与LLM对二语日语写作的评价，发现LLM以结构化但被削弱的形式再现了母语者对非母语写作的偏见方向与维度排序，但低估了团结度差距，并对学习者母语背景做出了人类未表现的区分。 |
| [^290] | [A False Average: Chain-of-Thought Monitors Collapse Where They Are the Only Defense](https://arxiv.org/abs/2608.00583) | 仅重写智能体的思维链推理（保持动作完全不变）即可在一次无梯度攻击中将CoT监控器的捕获率从约95%暴跌至11%以下，揭示监控器的总体准确率是掩盖其在唯一防线场景下近乎全面失效的虚假平均值。 |
| [^291] | [What the Abstention Option Measures: Refusal Loss, Sampling Churn, and Option Capture in Distilled Small Language Models](https://arxiv.org/abs/2607.28639) | 该研究将蒸馏小型语言模型在偏见基准弃权选项上的行为变化分解为拒绝损失、采样波动和选项捕获三种机制，并证明聚合分数仅能反映拒绝损失，只有以已提交答案为条件进行评估，才能避免将拒绝行为的移动误读为公平性的提升。 |
| [^292] | [Change2Task: From Repository Changes to Executable Coding Agent Tasks and Environments](https://arxiv.org/abs/2607.28591) | Change2Task 提出了一个基于仓库历史的系统，能够将已合并的拉取请求自动转换为健康现代代码版本上经过验证的可执行编码智能体任务与环境，为编码智能体的训练、基准测试和持续评估提供可扩展的数据供给。 |
| [^293] | [Mark, Don't Erase: Token Inoculation for Dual-Use Knowledge in LLMs](https://arxiv.org/abs/2607.18639) | 提出“Token接种”方法，在继续预训练时用特殊标记标注危险句子并训练模型条件化地预测危险答案，测试时因标记缺失而抑制危险输出，从而在保持93%良性性能的同时将危险准确率控制在约18%。 |
| [^294] | [CANDOR: Chance-Calibrated Discordance in Frozen Foundation Encoders](https://arxiv.org/abs/2607.18451) | 本文提出CANDOR度量，通过等大小对称样本库校正最近邻不一致性，将机会水平固定为二分之一，揭示冻结编码器并非失明但普遍性能较弱。 |
| [^295] | [Pancasila-Dilemmas: Evaluating Large Language Models on Indonesian Human Value Dilemmas Grounded in Pancasila](https://arxiv.org/abs/2607.18066) | 该论文构建了首个基于印度尼西亚国家意识形态“潘查希拉”五大价值观的价值对齐评估数据集Pancasila-Dilemmas（包含1,834个源自印尼新闻的两难困境问题），对50个大语言模型的评估发现它们与印尼价值观的对齐程度均不足0.5，凸显了当前模型在非西方价值体系上的局限性。 |
| [^296] | [Loop the Loopies!](https://arxiv.org/abs/2607.16051) | Loopie 是迄今最强大的开源循环 Transformer，它解决了循环次数不如扩大参数量的长期难题，在相同计算预算下显著超越标准 Transformer，并通过新颖的后训练方法实现了前沿水平的推理能力。 |
| [^297] | [Amplitude-Only FFN Intervention for Tool-Structured LLM Inference: Method, Gated Evaluation Protocol, and Cross-Model Empirical Results](https://arxiv.org/abs/2607.11183) | 该论文提出幅度门控（AG）这一无需重训练的推理时干预方法，通过掩码FFN激活坐标，在多个Qwen模型的函数调用和ToolBench端点上实现了1.6至2.8个百分点的稳定性能提升。 |
| [^298] | [SAMPA: Transformer-based segmentation of prosodic boundaries in Brazilian Portuguese](https://arxiv.org/abs/2607.07408) | 本文提出SAMPA，通过在人工标注数据上微调Whisper large-v3模型，实现巴西葡萄牙语语音转录的同时自动标注终端韵律边界，性能超越了开箱即用的Whisper基线和基于特征的分类器。 |
| [^299] | [Rethinking Scientific Discovery in the Agentic Era](https://arxiv.org/abs/2607.03863) | 提出了SCION——一个作为组织枢纽的智能体科学操作系统，通过研究执行计划（REP）和元管理框架将碎片化的AI4Science工具整合为可执行、可审计、可复用的完整科研流程。 |
| [^300] | [CausalMix: Data Mixture as Causal Inference for Language Model Training](https://arxiv.org/abs/2607.01104) | CausalMix将语言模型训练中的数据配比优化建模为因果推断问题，通过小规模实验拟合因果模型估计条件平均处理效应（CATE），从而将最优数据配比外推到更大数据池和更大模型规模，避免了数据分布变化时从头重训的高昂成本。 |
| [^301] | [HealthAgentBench: A Unified Benchmark Suite of Realistic Agentic Healthcare Environments for Challenging Frontier AI Agents](https://arxiv.org/abs/2606.31179) | 该论文提出了HealthAgentBench，一个包含75个智能体医疗任务、涵盖9个类别且各具独特环境的统一基准套件，通过模拟端到端临床工作流程，评估AI智能体在真实医疗场景中处理原始数据并执行复杂多步骤任务的能力。 |
| [^302] | [Diagnosing and Mitigating Context Rot in Long-horizon Search](https://arxiv.org/abs/2606.29718) | 该论文通过对四个旗舰模型在三个基准上的系统性研究，首次揭示了“无结论搜索失败”这一此前被忽视的上下文腐化现象——模型远在用尽上下文窗口之前就放弃或给出不确定的错误答案，即使对简单查询也普遍存在，并进一步诊断与缓解这一问题。 |
| [^303] | [SURGELLM: Rethinking Multi-Task Evaluation through Task-Aware Feature Gating with Class-Balanced Normalization](https://arxiv.org/abs/2606.24259) | SURGELLM提出一个统一Transformer框架，通过外科手术式特征门控、任务条件化前缀令牌和实例加权归一化三个轻量级模块，解决多任务编码器中归纳偏置不匹配、类别不平衡和外部词汇知识缺失三大问题，并从理论上证明了超额风险界。 |
| [^304] | [LangMAP: A Language-Adaptive Approach to Tokenization](https://arxiv.org/abs/2606.23566) | 提出 LangMAP 分词方法，通过在共享词表上为每种语言建立独立分布，将 UnigramLM 扩展到多语言场景，既能从头训练多语言模型，也能在不改变词表的前提下将预训练分词器适配到特定语言，且推理时无需语言标签。 |
| [^305] | [Towards Root Memory: Benchmarking and Enhancing Implicit Logical Memory Retrieval for Personalized LLMs](https://arxiv.org/abs/2606.23283) | 该论文揭示了记忆检索中的“语义偏转”问题——基于语义相似度的检索会遗漏逻辑上关键的记忆，并构建IMLogic基准来诊断该问题，同时提出“根记忆”方法，通过LLM推理从用户历史中提炼可复用的逻辑并锚定证据，从而增强个性化大语言模型的隐式逻辑记忆检索能力。 |
| [^306] | [Leadership as Coordination Control: Behavioral Signatures and the Recovery-Advantage Boundary in Multi-Agent LLM Teams](https://arxiv.org/abs/2606.19111) | 该研究将交易型、变革型、情境型三种经典领导风格操作化为多智能体LLM团队中的过程级协调控制器，并通过行为特征测量发现：没有任何控制器在准确率上全面占优，其价值取决于特定条件，且从错误共识中恢复的优势存在明确边界。 |
| [^307] | [Rhythm of the Deep: Two-tier acoustic organization of sperm-whale codas from click waveforms to second-order sequence dependence](https://arxiv.org/abs/2606.16084) | 该研究发现抹香鲸尾音的声学组织具有两级嵌套结构：尾音内部反复出现的点击声波形差异与点击间隔时间互补，而尾音之间反复出现的整体尾音形式本身携带序列依赖性。 |
| [^308] | [Calibrated Triage, Not Autonomy: Confidence Estimation for Medical Vision-Language Models](https://arxiv.org/abs/2606.15910) | 本文提出通过评估九种置信度估计器，证明在医学视觉语言模型中，校准分诊比自主决策更可靠，并指出仅用自然图像训练的置信度探针能有效决定模型何时应放弃回答。 |
| [^309] | [Gefen: Optimized Stochastic Optimizer](https://arxiv.org/abs/2606.13894) | Gefen 通过在参数块间自动共享二阶矩估计并用学习到的码本量化一阶矩，将 AdamW 的优化器内存占用最多降低 8 倍（每十亿参数节省 6.5 GiB），同时从理论上解释了二阶矩共享有效的原因。 |
| [^310] | [TellOnce: Correcting Coding Agents Once, Improving Their Future Actions](https://arxiv.org/abs/2606.13174) | TellOnce是一个无需更新底层模型的即插即用个性化层，能将用户对智能体的对话式纠正转化为持久、有作用范围且可修订的行为规则，使智能体在未来的任务中持续改善行为、遵守用户偏好。 |
| [^311] | [Observable Patterns Are Not Explanations: A Causal-Geometric Analysis of Latent Reasoning Models](https://arxiv.org/abs/2606.12689) | 该研究通过对照实验与因果-几何分析发现，潜在推理模型中的可观测潜在状态模式并非推理机制的可靠证据——潜在思维的作用是分级的并集中于低维子空间，应被视为“隐藏的计算”而非“隐藏的解释”。 |
| [^312] | [Dummy Backdoor as a Defense: Removing Unknown Backdoors via Shared Internal Mechanisms for Generative LLMs](https://arxiv.org/abs/2606.11648) | 该论文利用相同攻击目标的后门会引发相似内部激活变化这一发现，提出先植入已知触发器的虚拟后门、再通过虚拟触发器输入与干净响应配对微调将其消除的方法，从而在无需了解攻击类型的情况下移除生成式大语言模型中的未知后门。 |
| [^313] | [Detecting Differences Is Not Understanding Structure: Large Language Models Fail at Graph Isomorphism](https://arxiv.org/abs/2606.09484) | 该研究揭示大语言模型虽然在图同构检测上表现近乎完美，但在节点标签被置换时却无法识别相同图的同构关系，证明其成功依赖于模式利用而非对抽象图结构的真正理解，因此图推理基准测试的成绩不能作为拓扑理解能力的证据。 |
| [^314] | [PriFT: Guiding Supervised Fine-Tuning with Pretrained Priors](https://arxiv.org/abs/2606.09396) | 提出 PriFT 方法，以冻结的预训练模型先验而非正在微调的模型来计算 token 训练权重，避免权重与优化轨迹纠缠所引发的自我强化循环，从而提升有监督微调的泛化能力。 |
| [^315] | [Agents' Last Exam](https://arxiv.org/abs/2606.05405) | 本文提出由250多位行业专家共同打造的Agents' Last Exam（ALE）基准，通过13个行业集群、55个子领域、1千多项长周期、结果可验证的真实经济价值任务来评估AI智能体，并发现当前系统在最难任务层级上仍远未饱和。 |
| [^316] | [Dynamic Short Convolutions Improve Transformers](https://arxiv.org/abs/2606.03825) | 本文提出动态短卷积作为一种新的神经网络原语，利用依赖输入的滤波器在保留卷积局部性偏置的同时提升表达能力，在150M至2B参数的语言建模实验中持续优于标准Transformer及静态短卷积增强版本。 |
| [^317] | [Investigating and Alleviating Harm Amplification in LLM Interactions](https://arxiv.org/abs/2606.02423) | 该论文针对大语言模型在多轮对话中逐步放大危害的问题，提出了涵盖十二个风险类别的HarmAmp基准，并设计了能预判有害交互轨迹并主动干预的监控器TrajSafe。 |
| [^318] | [Unified Context Evolution for LLM Agents](https://arxiv.org/abs/2606.02304) | 提出了统一上下文演化（UCE）框架，无需梯度训练即可将智能体经验分解为记忆、策略、工作流和技能四种类型的可演化上下文单元（ECU），通过生成、检索、使用评分与修剪机制构建跨任务持续演化的经验库，解决了智能体每轮任务从固定上下文重启、有用策略随任务结束而丢失的问题。 |
| [^319] | ["I Strongly Suspect This Website Is a Scam": Benchmarking PII Leakage and Detection without Defense in Autonomous Web Agents](https://arxiv.org/abs/2606.00497) | 本文提出Scammer4U基准测试，揭示社会工程学攻击可使前沿自主网页代理在无隐私指导的情况下泄露54–93%的关键个人身份信息，而良性网站上泄露率为0%，凸显了已部署智能体系统面临的严重安全风险。 |
| [^320] | [Prompt PerturbAItions: Mechanisms of LLM Response Inconsistencies](https://arxiv.org/abs/2605.30913) | 该论文发现提示词的词汇与语气扰动（从礼貌到毒性措辞）会显著降低大语言模型的事实准确性并增加预测不确定性，且毒性越强准确率越低，归因图分析进一步揭示了这种不一致源于模型内部对扰动敏感的变体节点被相对放大。 |
| [^321] | [Learning Faithful Reasoning Processes with Spatio-Temporal Latent Dynamics for Large Language Models](https://arxiv.org/abs/2605.26893) | 提出GeoFaith框架，将最终状态潜在不确定性与预测熵动力学作为过程正则化信号融入GRPO，无需过程奖励模型即可学习忠实且稳定的推理过程。 |
| [^322] | [HyLaT: Efficient Multi-Agent Communication via Hybrid Latent-Text Protocol](https://arxiv.org/abs/2605.25421) | HyLaT提出了一种混合潜空间-文本通信协议，通过潜空间通道高效传输认知信号、用自然语言传递关键信息以保持可解释性，并配合两阶段训练框架，在多轮多智能体交互中显著降低了通信开销。 |
| [^323] | [QUEST: Training Frontier Deep Research Agents with Fully Synthetic Tasks](https://arxiv.org/abs/2605.24218) | QUEST 是一系列开源通用深度研究智能体模型（2B至35B），其核心创新在于基于统一评分准则树的全合成数据流水线，配合中期训练、监督微调与强化学习的训练方案，使其在事实检索、引用溯源和报告综合等多种长程搜索任务上表现出色。 |
| [^324] | [Naturalistic measure of social norms alignment](https://arxiv.org/abs/2605.23420) | 该论文提出了一种通过解决方案匹配在自然、自由形式场景下测量社会规范对齐的新框架，引入两个一致性指标并构建了3000个丹麦语社会困境数据集，可用于测量LLM与人类之间或人与人之间的规范一致性。 |
| [^325] | [Continuous Diffusion Scales Competitively with Discrete Diffusion for Language](https://arxiv.org/abs/2605.18530) | 该研究通过将 Plaid 与现代离散扩散语言模型架构对齐构建了 RePlaid，建立了首个可媲美离散 DLM 的连续扩散语言模型缩放定律，证明连续扩散是语言建模中高度竞争且可扩展的方案。 |
| [^326] | [Firefly: Illuminating Large-Scale Verified Tool-Call Data Generation from Real APIs](https://arxiv.org/abs/2605.17558) | FireFly通过先让大模型探索真实API再从观察结果反向合成任务的新范式，从真实MCP服务器生成在构造上保证标签正确性的大规模工具调用训练数据。 |
| [^327] | [VerifyMAS: Hypothesis Verification for Failure Attribution in LLM-based Multi-Agent Systems](https://arxiv.org/abs/2605.17467) | 提出VerifyMAS框架，将LLM多智能体系统的故障归因转化为在完整交互轨迹上进行故障假设的提出与验证，从而捕捉局部日志方法无法发现的全局性故障并实现细粒度归因。 |
| [^328] | [Where and What to Reward? Understanding Generalization Between Verifiable and Open-Ended Domains in Reinforcement Learning](https://arxiv.org/abs/2605.15464) | 该研究通过受控实验分离“提示词来源领域”与“奖励类型”这两个因素，系统揭示了RLVR与RLHF在可验证推理任务与开放式对话任务之间的泛化与迁移规律。 |
| [^329] | [Web Agents Should Adopt the Compile-and-Execute Paradigm](https://arxiv.org/abs/2605.14290) | 该论文主张网页智能体应放弃 ReAct 这一默认架构，改用“编译-执行”范式——即在观察运行时网页内容之前先确定任务程序再执行，从而防止网页中多来源的不可信内容（如提示注入）篡改智能体的控制流。 |
| [^330] | [ReVision: Scaling Computer-Use Agents via Temporal Visual Redundancy Reduction](https://arxiv.org/abs/2605.11212) | ReVision通过学习到的视觉块选择器在连续截图之间去除时序冗余的视觉块，使计算机使用智能体在平均减少46%token用量的同时有效利用历史信息并提升性能。 |
| [^331] | [Mathematical Reasoning via Intervention-Based Time-Series Causal Discovery Using LLMs as Concept Mastery Simulators](https://arxiv.org/abs/2605.07600) | 提出CIKA框架，将大语言模型自身作为干预模拟器，通过把概念状态干预为“已掌握”并测量正确率变化来估计因果效应，从而识别真正在因果上促成正确答案的概念，克服了题目难度等混杂因素造成的虚假关联。 |
| [^332] | [LatentRAG: Latent Reasoning and Retrieval for Efficient Agentic RAG](https://arxiv.org/abs/2605.06285) | LatentRAG提出将智能体RAG的推理与检索从离散语言空间转移到连续潜在空间，通过单次前向传播直接从隐藏状态生成潜在token来替代逐token生成自然语言思考和子查询，从而显著降低延迟。 |
| [^333] | [Why Geometric Continuity Emerges in Deep Neural Networks: Residual Connections and Rotational Symmetry Breaking](https://arxiv.org/abs/2605.04971) | 论文揭示了深度网络中“几何连续性”现象（相邻层主奇异向量方向相似）的起源：残差连接通过跨层梯度一致性对齐权重更新，对称性破缺的非线性激活通过固定共享坐标系防止旋转漂移，二者共同作用产生了这一性质。 |
| [^334] | [SATFormer: Learning When to Use Early Value Residuals in Transformers](https://arxiv.org/abs/2605.03953) | 提出SATFormer，通过上下文相关的门控机制让每个token和注意力头动态决定何时复用第一层值残差，在从1.3亿到13亿参数的多个规模上均超越了静态值残差和标准Transformer基线。 |
| [^335] | [SignVerse-2M: A Two-Million-Clip Pose-Native Universe of 55+ Sign Languages](https://arxiv.org/abs/2605.01720) | 本文提出SignVerse-2M，一个涵盖55多种手语、包含约两百万个统一DWPose关键点姿态片段的多语言数据集，填补了真实场景手语视频大规模姿态表示的空白。 |
| [^336] | [Prompt-Induced Score Variance in Zero-Shot Binary Vision-Language Safety Classification](https://arxiv.org/abs/2605.00326) | 该论文揭示零样本安全分类中单一提示词的不安全分数受提示词措辞影响方差大且不可靠，证明提示词均值集成在跨基准评估下更加稳健，而基于开发标签校准的单一提示词仅在开发基准上占优。 |
| [^337] | [Repetition over Diversity: High-Signal Data Filtering for Sample-Efficient German Language Modeling](https://arxiv.org/abs/2604.28075) | 该研究发现，对于德语语言建模，重复训练经过严格过滤的高质量数据始终优于在更大但过滤程度较低的数据集上进行单次训练，且这一性能优势在经过7个训练周期后依然保持。 |
| [^338] | [Why Are Some Emotions Harder for LLMs? Uncovering the Causal Mechanisms of Emotion Inference via Sparse Autoencoders](https://arxiv.org/abs/2604.25866) | 本研究通过稀疏自编码器揭示了不同情绪在LLM中推断机制的因果差异，发现惊讶和恐惧依赖集中特征集，而厌恶则依赖更分散的因果组织，这解释了模型对不同情绪表现不均的原因。 |
| [^339] | [BERAG: Bayesian Ensemble Retrieval-Augmented Generation for Knowledge-based Visual Question Answering](https://arxiv.org/abs/2604.22678) | 该论文提出BERAG框架，通过贝叶斯集成方式替代传统的文档拼接方法进行检索增强生成，解决了长上下文中的“迷失在中间”效应和二次方计算成本问题，并结合BEFT微调方法应用于基于知识的视觉问答任务。 |
| [^340] | [Semantic Needles in Document Haystacks: Sensitivity Testing of LLM-as-a-Judge Similarity Scoring](https://arxiv.org/abs/2604.18835) | 该论文提出了一个可扩展的多因素实验框架，通过“大海捞针”式的成对文档比较系统测试七个LLM的相似度评分敏感性，并揭示了包括与候选顺序效应不同的文档内位置偏差在内的多种惊人发现。 |
| [^341] | [A Survey on Long-Term Memory Security in LLM Agents: Attacks, Defenses, and Governance Across the Memory Lifecycle](https://arxiv.org/abs/2604.16548) | 本文提出记忆生命周期框架与可验证记忆治理（VMG）框架，系统刻画了LLM智能体长期记忆所面临的持久性、有状态性与传播性等新型安全威胁，并沿六个生命周期阶段和四个安全目标组织了攻击、防御及其跨阶段依赖关系的全景式分析。 |
| [^342] | [Towards Real-world Human Behavior Simulation: Benchmarking Large Language Models on Long-horizon, Cross-scenario, Heterogeneous Behavior Traces](https://arxiv.org/abs/2604.08362) | 该论文提出了首个完全基于真实数据构建的预测性用户模拟基准OmniBehavior，整合长时程、跨场景、异构行为模式于统一框架，并实证揭示当前大语言模型难以准确预测真实世界中复杂的跨场景人类行为。 |
| [^343] | [LinearARD: Linear-Memory Attention Distillation for RoPE Restoration](https://arxiv.org/abs/2604.00004) | LinearARD提出了一种线性内存的注意力蒸馏方法，通过将经RoPE缩放的学生模型与冻结的原生RoPE教师模型在最后一层注意力的Q/K/V自关系矩阵逐行分布上对齐，以线性内存开销直接恢复注意力结构，从而摆脱传统持续预训练对海量长文本数据的依赖。 |
| [^344] | [ActTail: Global Activation Sparsity in Large Language Models](https://arxiv.org/abs/2603.12272) | 提出基于重尾自正则化理论的ActTail方法，利用各投影层经验谱密度的重尾指数为其分配全局激活稀疏度预算，在加速大语言模型推理的同时缓解性能损失。 |
| [^345] | [Do Compact SSL Backbones Matter for Audio Deepfake Detection? A Controlled Study with RAPTOR](https://arxiv.org/abs/2603.06164) | 该研究提出RAPTOR检测器，通过受控实验证明多语言HuBERT预训练（而非模型规模）是紧凑型SSL骨干网络实现跨域音频深度伪造检测鲁棒性的关键，并引入基于扰动的测试时不确定性增强协议，揭示了WavLM过度自信而mHuBERT校准稳定的差异。 |
| [^346] | [Improving the Performance of Compressed Transformers with Post-Hoc Embedding Alignment](https://arxiv.org/abs/2602.14635) | 提出了一种事后训练的对齐适配器（AlAd），通过将压缩Transformer模型的词元嵌入与性能更优的参考模型对齐，并利用局部上下文信息，从而恢复压缩所导致的性能损失。 |
| [^347] | [The Lexical Imprint of Pre-training Data](https://arxiv.org/abs/2602.10657) | 本文提出词级一元交叉熵（UCE）这一与分词器无关、无需模型的词汇对齐度量，并通过受控的从零预训练实验证明，预训练语料库与基准测试之间的词汇对齐程度能够持续解释零样本基准准确率。 |
| [^348] | [OdysseyArena: Benchmarking Large Language Models For Long-Horizon, Active and Inductive Interactions](https://arxiv.org/abs/2602.05843) | 提出OdysseyArena基准，通过形式化四个基元将智能体评估重新聚焦于长时程、主动和归纳式交互，并建立包含120个任务的OdysseyArena-Lite来衡量智能体从经验中自主发现潜在转换规律的能力。 |
| [^349] | [GraphDancer: Training LLMs to Explore and Reason over Graphs via Curriculum Reinforcement Learning](https://arxiv.org/abs/2602.02518) | GraphDancer是一个强化学习框架，通过图感知的课程式训练（由易到难调度），教会中等规模的大语言模型在异构图知识源上交替进行推理与函数调用，从而实现多跳证据聚合的图导航推理，并将该能力迁移到未见过的领域。 |
| [^350] | [DeALOG: Decentralized Multi-Agents Log-Mediated Reasoning Framework](https://arxiv.org/abs/2602.00996) | DeALOG是一个去中心化多智能体多模态问答框架，让专业化智能体通过共享自然语言日志作为持久记忆进行通信，在无中央控制的情况下实现协作式错误检测与验证，并在多个问答基准上取得有竞争力的性能。 |
| [^351] | [Fast KVzip: Efficient and Accurate LLM Inference with Gated KV Eviction](https://arxiv.org/abs/2601.17668) | 提出了一种基于门控的KV缓存驱逐方法，利用轻量级sink-attention门控模块和仅需前向传播的训练算法，在计算成本几乎可忽略的情况下驱逐高达70%的KV缓存，同时保持近乎无损的推理性能。 |
| [^352] | [The Straight and Narrow: Do LLMs Possess an Internal Moral Path?](https://arxiv.org/abs/2601.10307) | 该论文利用道德基础理论首次系统绘制并操控了大语言模型内部细粒度的道德表征空间，并提出自适应道德融合（AMF）这一推理时动态干预方法，有效解决了AI安全中安全性与有用性的权衡难题。 |
| [^353] | [Navigating Ideation Space: Decomposed Conceptual Representations for Positioning Scientific Ideas](https://arxiv.org/abs/2601.08901) | 该论文提出“构思空间”这一结构化表示，通过对比训练将科学知识分解为三个独立维度，实现了科学思想间概念距离的原则性测量与具有区分性的新颖性评估，帮助研究者精确定位新想法相对于现有研究的差异。 |
| [^354] | [Stylistic Evolution and LLM Neutrality in Singlish Language](https://arxiv.org/abs/2601.06580) | 本研究揭示了新加坡英语的风格随时间演变且主要由结构性特征驱动，同时发现大语言模型存在结构性权衡——无法同时生成真实的新加坡英语和时间中立的输出。 |
| [^355] | [Toward Global Large Language Models in Medicine](https://arxiv.org/abs/2601.02186) | 本研究构建了包含20种语言的80万条医学数据集GlobMed及基准测试GlobMed-Bench，对56个先进大语言模型进行大规模评估，揭示了现有模型因依赖高资源语言训练而在全球医疗场景中的局限性。 |
| [^356] | [Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought](https://arxiv.org/abs/2512.21711) | 本文通过因果操控实验和捷径实验发现，连续思维链（COCONUT）中的潜在标记并非编码忠实的推理过程，而是作为不可解释的占位符，促使模型走捷径而非进行真正的推理。 |
| [^357] | [Role-aware Heuristic Episodic Attention for Conversational LLMs](https://arxiv.org/abs/2512.06869) | 该论文提出REA框架，通过为指令和情景交互分配不同的持久性与表示策略来管理对话上下文，有效缓解多轮对话中的累积性上下文衰退问题，在Long-MT-Bench+上实现16.5%的相对评分提升并将延迟降低2.91倍。 |
| [^358] | [Think Parallax: Solving Multi-Hop Problems via Multi-View Knowledge-Graph-Based Retrieval-Augmented Generation](https://arxiv.org/abs/2510.15552) | ParallaxRAG 发现 Transformer 注意力头天然形成与推理跳数对齐的多视图接力结构，并提出对称多视图框架将查询与知识图谱解耦到各注意力头专属的语义空间中，从而构建更精准干净的子图以提升大模型的多跳推理能力。 |
| [^359] | [Hierarchical Alignment: Block-Level LoRA-DPO for Preference Tuning in Large Language Models](https://arxiv.org/abs/2510.12044) | 该论文提出分层对齐方法，将LoRA-DPO的更新限制在解码器中间深度的连续块内而非所有层，在可训练参数量完全相同的情况下取得了更低的偏好优化损失，证明了适配器放置位置对大语言模型偏好调优效果的关键影响。 |
| [^360] | [Advancing the Pareto Frontier of Test-Time Scaling with Multi-Path Plan Aggregation](https://arxiv.org/abs/2510.11620) | 提出多路径规划聚合方法 MPPA，在冻结主干上通过独立 LoRA 适配器学习联合规划与答案选择，生成互补的多路径规划以引导多样化求解并智能评分选优，从而推进测试时扩展的帕累托前沿。 |
| [^361] | [Circuit Distillation](https://arxiv.org/abs/2509.25002) | 提出了一种名为“电路蒸馏”的新方法，通过匹配教师与学生模型中功能对应的电路组件并对齐其内部表示，实现了底层计算机制的蒸馏，在仅调整少量参数的情况下成功迁移算法能力，性能优于传统的行为模仿式标准蒸馏。 |
| [^362] | [Test-Time Policy Adaptation for Enhanced Multi-Turn Interactions with LLMs](https://arxiv.org/abs/2509.23166) | 该论文提出了测试时策略自适应新范式T2PAM及其轻量级实现算法ROSA，利用对话中的实时用户反馈作为奖励信号，通过单步参数更新使大语言模型在多轮交互中实现高效的自我纠错。 |
| [^363] | [SimulSense: Sense-Driven Interpreting for Efficient Simultaneous Speech Translation](https://arxiv.org/abs/2509.21932) | 提出SimulSense框架，通过模拟人类口译员持续读取输入语音并在感知到新语义单元时触发翻译输出，实现了更优的质量-延迟权衡，且决策速度比现有最先进基线快达9.6倍。 |
| [^364] | [KnowMT-Bench: Benchmarking Knowledge-Intensive Long-Form Question Answering in Multi-Turn Dialogues](https://arxiv.org/abs/2509.21856) | 提出了KnowMT-Bench——首个用于动态评估大语言模型在医学、金融、法律等知识密集型领域多轮长文本问答能力的基准，实验揭示多轮对话会导致模型事实能力和信息传递效率显著下降。 |
| [^365] | [Automotive-ENV: Benchmarking Multimodal Models in Automotive Cockpit Environments](https://arxiv.org/abs/2509.21143) | 该论文提出了Automotive-ENV，首个基于Android Automotive OS的交互式基准测试，通过420个涵盖日常座舱操作与安全关键场景的任务，结合车辆状态信号和可复现的状态验证器，系统评估多模态模型在汽车座舱环境中的能力。 |
| [^366] | [Evolving Language Models without Labels: Majority Drives Selection, Novelty Promotes Variation](https://arxiv.org/abs/2509.15194) | 该论文提出 EVOL-RL 方法，用多数投票提供稳定选择信号、基于嵌入的新颖性促进变异并辅以熵正则化，解决了无标签自我改进中“选择诱导的多样性坍缩”问题，使大语言模型在缺乏参考答案时也能持续进化。 |
| [^367] | [Large Language Model probabilities cannot reliably distinguish between possible and impossible language](https://arxiv.org/abs/2509.15114) | 本研究通过对8个大语言模型内部概率的分析发现，模型的字符串概率无法可靠地区分可能语言与不可能语言，质疑了将语言模型概率作为语法判断代理指标的可行性。 |
| [^368] | [Can Multi-modal (reasoning) LLMs detect document manipulation?](https://arxiv.org/abs/2508.11021) | 本研究系统评估了包括 OpenAI O1、Gemini、Claude 等多种最先进多模态（推理）大语言模型在真实交易文档上检测欺诈篡改的能力，发现顶尖模型展现出卓越的零样本泛化能力。 |
| [^369] | [Robust and Fine-Grained Detection of AI Generated Texts](https://arxiv.org/abs/2504.11952) | 本文提出一组基于词元分类的模型，能够鲁棒且细粒度地检测AI生成及人机共同创作的文本，并发布了涵盖23种语言、超过240万条文本的新数据集。 |
| [^370] | [Training in translation tools and technologies: Findings of the EMT survey 2023](https://arxiv.org/abs/2503.22735) | 2023年EMT调查显示，翻译技术教学正快速响应行业创新：机器翻译、译后编辑与质量评估日益成为必修内容，培养项目对生成式AI工具反应迅速，并趋向以免费学术授权的云端软件为核心整合教学体系。 |
| [^371] | [Evaluating LLM-based Agents for Multi-Turn Conversations: A Survey](https://arxiv.org/abs/2503.22458) | 本综述基于近250篇文献，提出了两个相互关联的分类体系——“评估什么”与“如何评估”——为多轮对话中基于大语言模型的智能体评估提供了系统性的结构化框架。 |
| [^372] | [Auto-AdvER: Disentangling Condition, History, and Commercial Offers in Automotive Advertisements](https://arxiv.org/abs/2412.05655) | 本文提出了面向汽车广告领域的专用跨度标注模式Auto-AdvER及配套数据集，通过“车况”、“历史”和“销售选项”三个标签解耦广告中的关键信息，并系统比较了Transformer编码器、大语言模型及条件随机场等多种模型的标注性能，标注者间一致性达78%。 |
| [^373] | [Artificial Leviathan: Exploring Social Evolution of LLM Agents Through the Lens of Hobbesian Social Contract Theory](https://arxiv.org/abs/2406.14373) | 该论文构建了一个赋予心理驱力的LLM智能体沙盒模拟社会，并以霍布斯社会契约论为分析框架，实验发现智能体从最初类似“自然状态”的无序冲突逐步演化出社会契约，为计算社会科学验证社会理论提供了新范式。 |
| [^374] | [Budgeted Prompt-Set Expansion and Refinement for Language Models via Simulation Optimization](https://arxiv.org/abs/2404.08164) | 本文提出了一种基于仿真优化的两阶段框架，在有限评估预算下将提示词表示为中等维度向量并构建得分代理模型，从而实现语言模型提示词集合的高效扩展与精炼以最大化任务得分。 |

# 详细

[^1]: 进化策略的质心解释

    Center-of-Mass Interpretation of Evolution Strategies

    [https://arxiv.org/abs/2610.35865](https://arxiv.org/abs/2610.35865)

    本文提出进化策略更新的几何解释：将扰动模型视为粒子、奖励视为质量，证明减去均值的ES更新恰好等于种群从等质量质心到奖励加权质心的位移乘以一个标量因子。

    

    进化策略（ES）通过采样一群扰动副本、对每个副本进行评分，然后沿着扰动的奖励加权方向移动来更新模型。这种更新通常被表述为平滑梯度的估计。我们则提出了对这种更新的一种几何解释：将每个扰动模型视为一个粒子，其奖励视为粒子的质量。在评分之前，每个粒子质量相同，整个群体在其质心处平衡；一旦分配了奖励，群体便在其奖励加权的质心处平衡。我们证明，对于任意有限种群，减去均值的ES更新恰好就是这个位移，再乘以一个取决于平均奖励的标量因子。这一物理解读要求奖励非负，而这总是可以通过给每个奖励加上一个常数来实现，且不改变更新本身。在大种群极限或在对偶（antithetic）采样下，质心……

    arXiv:2610.35865v1 Announce Type: new  Abstract: Evolution strategies (ES) update a model by sampling a population of perturbed copies, scoring each one, and moving in the reward-weighted direction of the perturbations. This update is usually presented as an estimate of a smoothed gradient. We present a geometrical interpretation of this update instead. Treat each perturbed model as a particle and its reward as the particle's mass. Before scoring, every particle has the same mass and the population balances at its centroid. Once rewards are assigned, it balances at its reward-weighted center of mass instead. We show that the mean-subtracted ES update is exactly this displacement, scaled by a scalar factor that depends on the mean reward, for any finite population. The physical reading needs nonnegative rewards, which can always be arranged by adding a constant to every reward without changing the update. In the limit of a large population or under antithetic sampling, the centroid sits
    
[^2]: 保先以扩：面向多语言大语言模型语音识别的源语言感知适配与模型合并

    Preserve to Expand: Source-Aware Adaptation and Model Merging for Multilingual LLM-Based ASR

    [https://arxiv.org/abs/2610.35846](https://arxiv.org/abs/2610.35846)

    该论文提出了在基于连接器的语音大语言模型中扩展多语言ASR时的适配与合并策略评估，发现SSU-Wanda+TIES合并策略在单目标扩展中实现了最佳的目标语言习得与源语言保持权衡，而更新合并在多目标扩展中提供了稳定的源语言保持能力。

    

    在必须保留先前已支持语言且无法重放源语言训练数据的情况下，将多语言语音识别（ASR）系统扩展到新语言极具挑战性。我们在基于连接器（connector）的语音大语言模型中研究这一问题，其中可训练的连接器将语音表征映射到冻结的大语言模型中。我们评估了在适配连接器的同时保持源语言性能的目标更新构建与合并策略。在单目标扩展中，仅针对目标语言的微调和标准TIES合并能够学习新语言，但会导致严重的源语言遗忘，而SSU-Wanda+TIES在源语言保持方面提供了最佳权衡。在多目标扩展中，更新合并提供了稳定的源语言保持能力，而递归扩展在合适的顺序下可以改善目标语言适配效果，但对顺序高度敏感。这些结果表明，多语言ASR扩展需要在目标语言习得与源语言（保持）之间取得平衡。

    arXiv:2610.35846v1 Announce Type: new  Abstract: Expanding multilingual ASR systems to new languages is challenging when adaptation must preserve previously supported languages without replaying source-language training data. We study this problem in connector-based Speech LLMs, where a trainable connector maps speech representations into a frozen large language model. We evaluate target-update construction and merging strategies for adapting the connector while preserving source-language performance. In single-target expansion, target-only fine-tuning and standard TIES merging learn new languages but cause severe source-language forgetting, while SSU-Wanda+TIES gives the best source-preserving trade-off. In multi-target expansion, update merging provides stable source retention, whereas recursive expansion can improve target adaptation under favorable orders but is highly order-sensitive. These results show that multilingual ASR expansion requires balancing target acquisition, source 
    
[^3]: 理解Jev式系统一模型：结构能否减少选择偏差？

    Understanding Jev-Style System One Models: Does Structure Reduce Selection Bias?

    [https://arxiv.org/abs/2610.35826](https://arxiv.org/abs/2610.35826)

    本文通过在保持问题内容不变的前提下打乱选项顺序的简单测试，系统比较了两个Jev版本、十二个Jev复现模型和六个生成式LLM在多选项任务中的选择偏差，探究Jev式决策结构能否减少选择偏差。

    

    arXiv:2610.35826v1 公告类型：new 摘要：尽管有要求忽略呈现顺序的指令或引出推理的提示，大型语言模型（LLM）中的选择偏差仍可能持续存在。近来对Jev式系统一行为的复现工作产生了一个不断壮大的模型家族，这些模型直接在给定候选选项上预测概率。然而，在多选项任务中，针对Jev各版本、Jev复现模型以及生成式LLM之间选择偏差的系统性比较仍然有限。这就留下了一个核心问题：Jev式决策结构能否减少选择偏差？我们通过一个简单的测试来回答这个问题：在保持问题和候选内容不变的情况下打乱选项顺序。我们在两个Jev版本、十二个Jev复现模型和六个生成式LLM上，追踪个体答案在八种呈现方式下的表现，并将其与固定顺序重复进行比较。在我们从两个基准收集的1,700个问题的主要数据集上，Jev的准确率变化不到两个百分点……

    arXiv:2610.35826v1 Announce Type: new  Abstract: Selection bias in large language models (LLMs) can persist despite instructions to ignore presentation order or prompts that elicit reasoning. Recent efforts to reproduce Jev-style System One behavior have produced a growing family of models that predict probabilities directly over supplied candidates. Yet systematic comparisons of selection bias across Jev versions, Jev reproductions, and generative LLMs remain limited in multi-option tasks. This leaves a central question: does Jev-style decision structure reduce selection bias? We address this question with a simple test that shuffles options while keeping question and candidate content fixed. Across two Jev versions, twelve Jev reproductions, and six generative LLMs, we trace individual answers through eight presentations and compare them with fixed-order repetition. On our primary collection of 1,700 questions from two benchmarks, Jev's accuracy varies by less than two percentage poi
    
[^4]: 学习何时提交：面向快速扩散语言模型的调度蒸馏方法

    Learning When to Commit: Schedule Distillation for Fast Diffusion Language Models

    [https://arxiv.org/abs/2610.35804](https://arxiv.org/abs/2610.35804)

    提出dSpark框架，通过调度蒸馏学习何时提交token——优先揭示那些能让其他位置更易预测的token，从而在保持生成质量的同时大幅加速扩散大语言模型的并行生成。

    

    扩散大语言模型可以并行生成多个token，但通常需要多轮解码才能完成一个回复。高效的生成不仅取决于哪些token容易预测，还取决于提交这些token如何帮助预测剩余的位置。一个高度可预测的token可能提供的有用上下文很少，而揭示一个较难预测的token反而可以让其他多个位置更快地变得可预测。我们提出了dSpark，这是一个新的基于调度蒸馏的框架，用于学习何时提交token以实现更快的扩散语言生成。为实现这一目标，我们构建了解除掩码的调度，优先处理那些被揭示后能使其他位置更容易预测的token，从而在解码早期降低不确定性。然后我们将这些调度蒸馏到模型中，以实现更快的并行采样，并通过KL散度正则化器帮助保持生成质量。我们对该方法进行了广泛的评估。

    arXiv:2610.35804v1 Announce Type: cross  Abstract: Diffusion large language models (dLLMs) can generate multiple tokens in parallel, but often require many decoding rounds to complete a response. Efficient generation depends not only on which tokens are easy to predict, but also on how committing to them helps predict the remaining positions. A highly predictable token may provide little useful context, while revealing a less predictable token can help several other positions become predictable sooner. We propose dSpark, a new schedule distillation-based framework for learning when to commit tokens for faster diffusion language generation. To achieve this, we construct unmasking schedules that prioritize tokens whose reveal makes other positions easier to predict, reducing uncertainty earlier in decoding. We then distill these schedules into the model to enable faster parallel sampling, with a KL-divergence regularizer to help preserve generation quality. We extensively evaluate our ap
    
[^5]: 量化大语言模型研究智能体的论断平滑现象

    Quantifying Claim Smoothing by LLM Research Agents

    [https://arxiv.org/abs/2610.35793](https://arxiv.org/abs/2610.35793)

    该论文发现LLM研究智能体在生成报告时会把多元甚至冲突的证据“平滑”为过度自信的单一叙事，并提出了SmoothScore这一论断级评分流程来量化这种失真程度。

    

    深度研究，或任何使用大语言模型（LLM）阅读大量文档集合并产出洞察的应用，都需要对不同来源的信息进行综合。在这项工作中，我们发现LLM生成的长篇研究报告读起来像是充满自信的综合结论，但实际未能准确反映其底层的文档内容。我们将这种失败称为“平滑”（smoothing）。在将多样且可能相互冲突的来源压缩为单一叙事的过程中，模型丢弃了语料库中证据的“参差性”（jaggedness），并断言出比引用内容所能支持的更为武断的结论。我们提出了一个论断级别的评分流程——SmoothScore，用于量化这种效应。该流程从模型生成的报告中提取关键结论性论断，然后找到与每个论断相关的证据，并将这些证据按照高层面的主题进行聚类。最后，针对证据对每个论断进行评分，以量化智能体所做平滑的程度。

    arXiv:2610.35793v1 Announce Type: new  Abstract: Deep research, or any use of large language models (LLMs) to read a large document collection and produce insights, requires synthesizing information from different sources. In this work, we find that LLMs produce long-form research reports that read as confident syntheses but which fail to accurately reflect the underlying documents. We call this failure smoothing. In collapsing diverse and potentially conflicting sources into a single narrative, the model discards the evidential "jaggedness" of its corpus and asserts conclusions more decisive than the citations can support. We introduce a claim level scoring pipeline, SmoothScore, to quantify this effect. Our pipeline extracts key conclusive claims from model generated reports, then finds evidence relevant to each claim and clusters that evidence along high level aspects. Finally, each claim is scored with respect to the evidence to quantify the degree of smoothing done by the agent. A
    
[^6]: 原子安全关注点：迈向安全知识表示的基础语言

    Atomic Security Concerns: Towards a Foundational Language for Security Knowledge Representation

    [https://arxiv.org/abs/2610.35782](https://arxiv.org/abs/2610.35782)

    本文受化学原子概念启发，提出“原子安全关注点”的概念，并假设网络安全知识可以通过一组有限的基础关注点的组合来表示，从而为安全知识表示奠定一种基础语言。

    

    网络安全治理、风险管理、合规性以及安全架构依赖于众多概念，包括安全属性、领域、控制措施、风险、实施措施、能力和合规要求。尽管这些概念被广泛使用，但它们通常通过异构的抽象和词汇来表示，使得它们之间的关系难以一致地表达，并且大量相关推理依赖于专家知识。本文引入了“安全关注点”和“原子安全关注点”的概念，并探讨了一个假设：网络安全知识可以通过一组有限的基础关注点来表示。受化学中原子角色的启发，所提出的方法将网络安全概念视为更基础关注点的组合，而非独立的抽象。因此，原子安全关注点可以被视为一种通……（摘要原文在此处截断）

    arXiv:2610.35782v1 Announce Type: cross  Abstract: Cybersecurity governance, risk management, compliance, and security architecture rely on numerous concepts, including security properties, domains, controls, risks, implementation measures, capabilities, and compliance requirements. Although these concepts are widely used, they are often represented through heterogeneous abstractions and vocabularies, making relationships among them difficult to express consistently and leaving much of the associated reasoning dependent on expert knowledge. This paper introduces the concepts of Security Concerns and Atomic Security Concerns (ASCs) and explores the hypothesis that cybersecurity knowledge may be represented through a finite set of foundational concerns. Inspired by the role of atoms in chemistry, the proposed approach views cybersecurity concepts as compositions of more fundamental concerns rather than as independent abstractions. Hence, Atomic Security Concerns could be viewed as a comm
    
[^7]: SEMA-Auto：具备自主性的有感知且高效的多轮攻击

    SEMA-Auto: Sighted and Efficient Multi-Turn Attack with Autonomy

    [https://arxiv.org/abs/2610.35743](https://arxiv.org/abs/2610.35743)

    提出SEMA-Auto，一种无需循环内评估器、能感知受害者回复并自主决定终止时机的多轮越狱攻击方法，通过合成自举调优和决策监督强化学习两阶段训练实现高效自主攻击。

    

    现有的越狱攻击方法分为两类：静态规划器预先固定其单轮或多轮对抗计划，虽然简单但无法感知受害者的回复，浪费轮次并限制了可迁移性；交互式生成器会读取受害者的每次回复以生成下一轮攻击，但需要循环内的评估器来提供成功信号或提示，导致高昂的成本和虚高的性能表现。为了弥合这一差距，我们提出了SEMA-Auto，一个具有自主性的有感知且高效的多轮攻击器，它能够与受害模型交互，并在成功后自行终止会话。为了稳定其行为并提升其能力，SEMA-Auto分两个阶段进行训练。合成自举调优阶段通过使用自举轨迹对基础模型进行微调，植入攻击目标的概念和符合规范的决策。随后，决策监督强化学习阶段联合但分别地优化两个目标。

    arXiv:2610.35743v1 Announce Type: new  Abstract: Existing jailbreak attackers fall into two classes: static planners fix their single- or multi-turn adversarial plans in advance, remaining simple but losing sight of the victim's replies, wasting turns and limiting transferability; interactive generators read each victim reply to generate the next-turn attack but require in-loop evaluators for success signals or hints, incurring substantial costs and inflated performance. Bridging this gap, we propose SEMA-Auto, a sighted and efficient multi-turn attacker with autonomy, interacting with the victim model and terminating the session by itself upon success. To stabilize its behaviors and boost its capabilities, SEMA-Auto is trained in two stages. Synthetic bootstrap tuning installs the notion of attack targets and conforming decisions by fine-tuning the base model with bootstrapped trajectories. Decision-supervised reinforcement learning then optimizes two objectives jointly yet separately
    
[^8]: JEV能否支持临床试验标准的评估？

    Can JEV Support Clinical Trial Criteria Assessment?

    [https://arxiv.org/abs/2610.35721](https://arxiv.org/abs/2610.35721)

    本文提出非生成式决策模型JEV可在严格数据隐私保护下用于临床试验标准预筛，其性能与大型生成式模型Qwen Flash相当，并在MIMIC-III数据集上显著更优。

    

    临床试验预筛需要区分患者病历中“未满足的条件”与“证据不足”的情况。在给定患者病历、标准问题及预定义答案选项的条件下，我们回顾性地将JEV（一种直接返回原生选项概率的非生成式决策模型）与模式约束的Qwen Flash进行了比较，测试数据涵盖来自TrialGPT的2,048个患者-条件测试对，以及从MIMIC-III和MIMIC-IV衍生的问答数据集，所有评估均在严格的零数据保留（ZDR）策略下进行，以确保临床数据的隐私。两种模型分别以单独评估或共享同一患者上下文的批量模式对标准进行评估，并以参考标签衡量性能。在批量模式下，JEV与Qwen在TrialGPT（0.652 vs. 0.660）和MIMIC-IV（0.807 vs. 0.814）上取得了相近的宏F1分数，而JEV在MIMIC-III上显著优于Qwen（0.768 vs. 0.534；Δ = +0.234，95% CI：0.187-0.281）。这一优势……

    arXiv:2610.35721v1 Announce Type: cross  Abstract: Clinical trial prescreening requires distinguishing unmet conditions from insufficient evidence in patient records. Given patient records, criterion questions, and predefined answer options, we retrospectively compared JEV, a non-generative decision model returning native option probabilities, with schema-constrained Qwen Flash across 2,048 test patient-condition pairs from TrialGPT and question-answer datasets derived from MIMIC-III and MIMIC-IV, all evaluated under strict Zero Data Retention (ZDR) policies to ensure clinical data privacy. Both models assessed conditions individually or in batches sharing the same patient context, with performance measured against reference labels. In batch mode, JEV and Qwen achieved similar macro-F1 scores on TrialGPT (0.652 vs. 0.660) and MIMIC-IV (0.807 vs. 0.814), whereas JEV substantially outperformed Qwen on MIMIC-III (0.768 vs. 0.534; $\Delta$ = +0.234, 95% CI: 0.187-0.281). This advantage lar
    
[^9]: 多层Transformer的定量泛化界：少数长字符串可以胜过大量短字符串

    Quantitative Generalization Bounds for Multi-Layer Transformers: A Few Long Strings can Beat Many Short Ones

    [https://arxiv.org/abs/2610.35715](https://arxiv.org/abs/2610.35715)

    本文首次对固定精度多层Transformer的分布内统计泛化与长度泛化进行了联合的定量理论分析，并揭示出一个反直觉的结论：用少数长字符串训练可以胜过用大量短字符串训练。

    

    基于Transformer的语言模型展现出惊人的能力，能够泛化到比训练中所见更长、更难的问题实例。然而，对这一现象的理论理解仍然滞后。与大多数机器学习模型不同，Transformer的泛化能力本质上需要从两个维度进行考察。一方面，必须研究分布内统计泛化，即模型将从训练样本学到的函数拓展到来自同一分布的新样本的能力；另一方面，鉴于语言数据的序列特性，还必须考虑长度泛化，即在给定长度输入上训练的模型能否外推到更长的输入。这两个概念以往大多被分开研究，它们之间如何相互作用仍是一个悬而未决的问题。在这项工作中，我们对固定精度的Transformer给出了两个维度的联合分析。我们给出了一般……

    arXiv:2610.35715v1 Announce Type: cross  Abstract: Transformer-based language models exhibit an astonishing capacity to generalize to longer and more difficult problem instances than those seen in training. However, theoretical understanding of this phenomenon still lags behind. Unlike most machine learning models, the generalisation abilities of transformers inherently require considering two axes. On one hand, one must examine in-distribution statistical generalisation, which concerns the ability of a model to broaden a function from training samples to novel samples of the same distribution. On the other hand, given the sequential nature of language data, one must also consider length generalisation, which reflects whether a model trained on inputs of a given length can extrapolate to longer ones.These two notions have mostly been studied separately, leaving open how they interact. In this work, we give a joint treatment of both axes for fixed-precision transformers. We give general
    
[^10]: 从自信度到统计准确性：通过直接审计与LLM辅助审计扩展人机协同评估

    From Self-Confidence to Statistical Accuracy: Scaling Human-in-the-Loop Evaluation with Direct and LLM-Assisted Auditing

    [https://arxiv.org/abs/2610.35698](https://arxiv.org/abs/2610.35698)

    该论文提出统一统计框架，通过置信度分层抽样实现无需校准置信度的直接人工审计，并基于灵敏度和假阳性率建模LLM评判者构建误差校正估计量，从而实现可扩展的人机协同模型评估。

    

    对于大型语言模型（LLM）和多模态大型语言模型（MLLM），模型报告的自信度往往校准不准，但它仍可能对高效评估有所帮助。与此同时，LLM-as-a-judge（以LLM作为评判者）为直接人工审计提供了一种可扩展的替代方案，但其实际收益取决于校准质量和推理成本。本文开发了一个统一的统计框架，用于分析在不同抽样策略下的这两种评估路线。对于直接审计，采用置信度分层抽样从部分验证的输出中恢复总体准确率，而无需校准的置信度。论文刻画了有限样本下的不确定性，并分析了高度不均衡的置信度分层如何影响审计效率和标签分配。对于评判者辅助评估，通过灵敏度和假阳性率对评判者进行建模，得到一个带校准不确定性的误差校正估计量。

    arXiv:2610.35698v1 Announce Type: new  Abstract: For large language models (LLMs) and multimodal large language models (MLLMs), model-reported self-confidence is often miscalibrated, yet it may still be useful for efficient evaluation. At the same time, LLM-as-a-judge offers a scalable alternative to direct human auditing, but its practical benefit depends on calibration quality and inference cost.   This paper develops a unified statistical framework for both evaluation routes under alternative sampling strategies. For direct auditing, confidence-stratified sampling is used to recover population accuracy from partially verified outputs without requiring calibrated confidence. Finite-sample uncertainty is characterized, and the analysis shows how highly uneven confidence strata affect audit efficiency and label allocation. For judge-assisted evaluation, the judge is modeled through sensitivity and false-positive rate, yielding an error-corrected estimator with calibration uncertainty. 
    
[^11]: MiRL：面向错误靶向的强化学习，用于纠正终端智能体的错误行为

    MiRL: Error-Targeted Reinforcement Learning for Correcting Terminal Agent Misbehaviors

    [https://arxiv.org/abs/2610.35697](https://arxiv.org/abs/2610.35697)

    本文提出MiRL方法，通过分析终端智能体的错误以发现跨环境反复出现的可泛化错误行为，并从错误发生步骤启动强化学习训练、仅在通过测试且避免错误行为时给予奖励，从而有针对性地纠正智能体的错误行为。

    

    靶向错误的学习为改进模型提供了一种清晰明了的方式：识别模型的特定弱点，训练模型以修复这些弱点，并检验这些弱点是否已被修复。我们研究如何将靶向错误的学习应用于终端智能体——这类智能体通过与命令行终端交互来完成以自然语言指定的长时程任务。现有的靶向错误方法是针对单轮任务设计的，而现有的智能体失败分析主要用于评估和调试智能体，而非用于训练智能体。我们提出错误行为驱动的强化学习，它首先对策略的错误进行分析，以找出可泛化的错误行为，即在众多环境中反复出现的错误步骤类型。随后，MiRL使用强化学习训练该策略，从这些错误行为发生的步骤开始启动推演，并且仅当推演通过测试且避免了错误行为时才给予奖励。在Qwen3.5-4B上，八种错误行为占错误步骤的52.1%……

    arXiv:2610.35697v1 Announce Type: cross  Abstract: Error-targeted learning gives a legible way to improve a model: identify specific weaknesses of the model, train the model to fix these weaknesses, and check whether the weaknesses are fixed. We study how to apply error-targeted learning to terminal agents, which solve long-horizon tasks specified in natural language by interacting with a command shell. Existing error-targeted methods are designed for single-turn tasks, and existing analyses of agent failures are used to evaluate and debug agents rather than to train them. We propose Misbehavior-driven RL (MiRL), which first profiles the errors of a policy to find generalizable misbehaviors: kinds of wrong steps that recur across many environments. MiRL then trains the policy with RL, starting rollouts at the steps where these misbehaviors occur and rewarding a rollout only if it passes the tests and avoids the misbehavior. On Qwen3.5-4B, eight misbehaviors account for 52.1% of wrong s
    
[^12]: 并非单一信念：对话式问答中的答案载荷、持久性与分离的置信度信号

    Not One Belief: Answer Payloads, Persistence, and Dissociated Confidence Signals in Conversational Question Answering

    [https://arxiv.org/abs/2610.35694](https://arxiv.org/abs/2610.35694)

    该研究发现对话式问答中答案的改变主要由用户提供的答案载荷（具体答案字符串）驱动而非言语行为本身，且四种置信度信号相互分离——模型内部探测器追踪的是答案的持久性而非真实性。

    

    对话式问答系统应当在用户提出异议时纠正错误，并指示哪些答案值得信任。我们将每条后续提问分解为其言语行为（重新提问、认同或质疑）及其载荷（一个答案字符串，无论正确与否），并在相同的对话记录上测量四种置信度信号：口头表达的置信度、长度归一化似然、冻结的隐状态探测器以及P(True)。当用户提供答案字符串时答案会发生改变，而在仅有言语行为的情况下几乎从不改变：在所有保留的全尺寸实验单元中，正确的答案载荷对错误答案的纠正效果至少是重新提问或认同的四倍。这四种置信度信号彼此不一致。在Qwen2.5-7B上，冻结的探测器随答案的持久性而非真实性而升高；这一发现在留出问题上依然成立，并得到Llama-3-8B（清洗后的答案）的支持，但在其他四个模型上不成立。在Qwen2.5-7B的81个经一次重新提问后仍持续存在的错误中，重新提问、开启新对话或弃答修复几乎无法……（摘要在此处截断）

    arXiv:2610.35694v1 Announce Type: new  Abstract: Conversational QA systems should fix errors when users push back and signal which answers to trust. We split each follow-up into its speech act (re-ask, agree, or challenge) and its payload (an answer string, correct or not), and measure four confidence signals on the same transcripts: verbalized confidence, length-normalized likelihood, a frozen hidden-state probe, and P(True). Answers change when the user supplies an answer string and almost never under the speech act alone: in every retained full-size cell, the correct answer converts at least four times as many wrong answers as re-asking or agreement. The four signals disagree. On Qwen2.5-7B the frozen probe rises with answer persistence rather than truth; this holds on held-out questions, with Llama-3-8B (cleaned answers) as support, but not on four other models. Among 81 Qwen2.5-7B errors that survive one re-ask, re-asking, a fresh chat, or abstain repair almost none, while a gold 
    
[^13]: 将LLM-SC框架适配于活态神圣遗产：基于帕斯帕提那神庙游客评论的领域特定信息提取、语义网络与人工评估

    Adapting LLM-SC to Living Sacred Heritage: Domain-Specific Extraction, Semantic Networks and Human Evaluation on Pashupatinath Temple Reviews

    [https://arxiv.org/abs/2610.35685](https://arxiv.org/abs/2610.35685)

    该研究将LLM-SC框架适配到尼泊尔帕斯帕提那神庙这一活态神圣遗产场景，通过新增8种神圣遗产实体类型构建12类实体分类体系，从Tripadvisor评论中成功提取结构化数据，且新增的神圣遗产类型占实体记录的64.8%，证明领域特定适配对宗教遗产评论分析至关重要。

    

    大型语言模型（LLM）流水线可以将游客评论转化为结构化数据，但通用方案可能会遗漏活态神圣遗产场所中的宗教仪式、神圣空间、文化限制以及情感复杂的体验。我们通过在GPT-4o提取提示词中明确指定一个包含12种类型的实体分类体系（4种原始LLM-SC类型加8种神圣遗产类型）、4种关系类型和9个方面-情感维度，将LLM-社会计算（LLM-SC）框架适配到尼泊尔帕斯帕提那神庙，并将其应用于从3,939条处理后语料中固定随机抽取的2,500条英文Tripadvisor评论。导出的数据包含11,286条实体记录、6,232条方面-情感记录和5,590条关系记录，分别覆盖2,485、2,487和2,433条评论；其中新增的8种神圣遗产类型占实体记录的64.8%。以200条评论的双标注者仲裁参考标准为基准，精确的规范化表层与类型匹配……（摘要原文在此处截断）

    arXiv:2610.35685v1 Announce Type: cross  Abstract: Large language model (LLM) pipelines can turn visitor reviews into structured data, but generic schemes may miss rituals, sacred spaces, cultural restrictions and emotionally mixed experiences at living sacred heritage sites. We adapted the LLM-Social Computing (LLM-SC) framework to Pashupatinath Temple, Nepal, by specifying a 12-type entity taxonomy (four original LLM-SC types plus eight sacred-heritage types), four relation types and nine aspect-sentiment dimensions in a GPT-4o extraction prompt, which was applied to a fixed random sample of 2,500 English-language Tripadvisor reviews from a processed corpus of 3,939. The exports contain 11,286 entity, 6,232 aspect-sentiment and 5,590 relation records, covering 2,485, 2,487 and 2,433 reviews, respectively; the eight added sacred-heritage types account for 64.8% of entity records. Against an adjudicated two-annotator reference for 200 reviews, exact normalized surface-and-type matching
    
[^14]: 评估DeepSeek V4.1 Flash-1017的失控与有害操纵风险

    Evaluating DeepSeek V4.1 Flash-1017 for loss of control and harmful manipulation risks

    [https://arxiv.org/abs/2610.35671](https://arxiv.org/abs/2610.35671)

    该论文依据欧盟通用AI实践准则，对DeepSeek V4.1 Flash-1017开展了独立第三方安全评估，在34个基准上系统测试其失控与有害操纵风险，并与多个领先的闭源前沿模型进行了对比。

    

    随着前沿语言模型能力日益增强且日趋自主，独立第三方评估对于刻画其系统性安全风险至关重要。虽然闭源模型发布时通常伴随全面的系统报告，但开放权重模型在前沿风险方面受到的审查却较少。为填补这一空白，我们对DeepSeek V4.1 Flash-1017进行了独立安全评估，该模型是V4.1 Flash的最新版本，采用了全新架构、更大规模的训练以及强大的智能体能力，与V4有显著区别。参照欧盟《通用人工智能实践准则》，我们在34个基准测试（包括5个内部测试套件）上对该模型进行了评估，涵盖评估完整性、失控和有害操纵三大领域。遵循AEF-1标准，我们将V4.1 Flash-1017与V4.1 Flash、更早的DeepSeek V4模型以及领先的闭源系统（GPT-6 Astra、GPT-5.6 Sol和Gemini 3.8 Flash）进行了比较。总体而言，V

    arXiv:2610.35671v1 Announce Type: new  Abstract: As frontier language models grow increasingly capable and autonomous, independent third-party evaluations are critical for characterizing their systemic safety risks. While closed-source releases are routinely accompanied by comprehensive system cards, open-weight models receive less scrutiny regarding frontier risks. To address this gap, we present an independent safety evaluation of DeepSeek V4.1 Flash-1017, the latest version of V4.1 Flash that departs from V4 with a new architecture, scaled-up training, and strong agentic capabilities. Drawing on the EU General-Purpose AI Code of Practice, we evaluate the model across 34 benchmarks, including 5 in-house suites, spanning evaluation integrity, loss of control, and harmful manipulation. Conforming to the AEF-1 standard, we compare V4.1 Flash-1017 against V4.1 Flash, earlier DeepSeek V4 models, and leading closed-source systems (GPT-6 Astra, GPT-5.6 Sol, and Gemini 3.8 Flash). Overall, V
    
[^15]: 使用跨度读取与选择性适配改进JEV蕴含重排序

    Improving JEV Entailment Reranking with Span Readouts and Selective Adaptation

    [https://arxiv.org/abs/2610.35654](https://arxiv.org/abs/2610.35654)

    本文提出结合平均跨度读取与选择性骨干适配的方法，用NLI选项内上下文状态均值替代选项结尾指针，并在保留注意力与门控DeltaNet投影低秩更新的同时移除前馈适配器，从而有效提升了JEV蕴含重排序在MC11基准上的准确率。

    

    arXiv:2610.35654v1 公告类型：新论文。摘要：JEV决策模型将语言模型骨干网络与轻量级评分头相结合，以在自然语言选项中进行选择。直接选择方式在单个请求内比较候选答案，而蕴含重排序则通过独立的自然语言推理（NLI）请求，基于固定的关系选项对每个候选进行评估。我们研究了用于直接选择的读取和适配方法在跨请求重排序场景中是否依然有效。我们将平均跨度读取与选择性骨干适配相结合：平均跨度池化用每个NLI选项内上下文状态的平均值取代原有的选项结尾指针，且不增加评分头的参数；选择性适配则在注意力和门控DeltaNet投影中保留低秩更新，同时移除前馈适配器。在MC11基准（对来自七个来源的十一种重排序配置取平均准确率）上，选择性适配提升了NeoHorse-1-4B的准确率……（摘要原文在此处截断）

    arXiv:2610.35654v1 Announce Type: new  Abstract: JEV decision models pair a language model backbone with a lightweight scoring head to select among natural-language options. Direct selection compares candidate answers within one request, whereas entailment reranking evaluates each candidate in a separate natural language inference (NLI) request over fixed relation options. We examine whether the readout and adaptation choices used for direct selection remain effective for cross-request reranking. We combine a mean-span readout with selective backbone adaptation. Mean-span pooling replaces the option-ending pointer with an average of the contextualized states inside each NLI option, without adding head parameters. Selective adaptation retains low-rank updates in attention and gated DeltaNet projections while removing feed-forward adapters. On MC11, which averages accuracy over eleven reranking configurations from seven sources, selective adaptation improves NeoHorse-1-4B accuracy over t
    
[^16]: 面向空战作战战术决策的离线语言支持系统

    Offline Language-Based Support System for Tactical Decision-Making in Air Combat Operations

    [https://arxiv.org/abs/2610.35635](https://arxiv.org/abs/2610.35635)

    本文提出了完全离线、基于开源组件和检索增强生成（RAG）技术的语言支持系统MATE，通过战术手册内容回答自然语言问题以辅助空战飞行员训练与战术决策，并经巴西空军飞行员评估获得4.00至4.40的高评分。

    

    本文提出了一种离线的基于语言的支持工具，用于辅助空战作战中的飞行员训练和战术决策。该系统名为MATE（训练与评估任务助手），应用检索增强生成（RAG）技术，利用从条令和战术手册中提取的内容来回答自然语言问题。该系统完全由开源组件构建，可在本地执行文本预处理、嵌入生成、语义搜索和语言生成，确保完全离线运行。该工具由三名巴西空军战斗机飞行员进行评估，他们对三个开源权重语言模型（Llama 3.1 8B、Qwen 2.5 14B和Mistral NeMo 12B）针对源自公开空战手册的五个战术问题所生成的回答进行了评分。五分制下的平均评分为4.00至4.40分，其中Mistral NeMo 12B获得最高分。（注：原摘要末尾内容被截断）

    arXiv:2610.35635v1 Announce Type: new  Abstract: An offline language-based support tool is presented to assist pilot training and tactical decision-making in air combat operations. The system, named MATE (Mission Assistant for Training and Evaluation), applies retrieval-augmented generation (RAG) to answer natural language questions using content extracted from doctrinal and tactical manuals. Built entirely with open-source components, it performs text preprocessing, embedding generation, semantic search, and language generation locally, ensuring full offline functionality. The tool was evaluated by three Brazilian Air Force fighter pilots, who rated the answers produced with three open-weight language models (Llama 3.1 8B, Qwen 2.5 14B, and Mistral NeMo 12B) to five tactical questions derived from a publicly available air combat manual. Average ratings ranged from 4.00 to 4.40 on a five-point scale, with Mistral NeMo 12B obtaining the highest score, and the pilots rated the response t
    
[^17]: 面向编程智能体的分层技能演化

    Hierarchical Skill Evolution for Coding Agent

    [https://arxiv.org/abs/2610.35616](https://arxiv.org/abs/2610.35616)

    提出分层技能演化（HSE）方法，通过任务级、仓库级到跨仓库的分层证据聚合，从编程轨迹中逐步提炼可跨仓库迁移的程序性技能，并借助外部书写-评估循环在冻结底层模型的情况下驱动技能生成与评估。

    

    技能为智能体提供可复用的经验，能够在不修改底层模型的情况下指导未来决策。然而，编程轨迹往往与任务及代码仓库特定的上下文纠缠在一起，这使得提炼出能够在多样化编程环境中泛化的程序性技能变得极具挑战。为解决这一问题，我们提出了分层技能演化，通过分层证据聚合从编程轨迹中逐步提炼程序性知识。HSE 首先从单个轨迹中提取任务级证据，然后在同一仓库内聚合重复出现的证据以形成仓库特定的技能候选，最后将跨仓库一致的模式整合为可跨仓库迁移的技能。该演化过程由一个外部的书写-评估循环驱动，在保持底层编程模型冻结的同时生成并评估技能。

    arXiv:2610.35616v1 Announce Type: new  Abstract: Skills provide agents with reusable experience to guide future decisions without modifying the underlying model. However, coding trajectories are often entangled with task- and repository-specific contexts, making it challenging to distill procedural skills that generalize across diverse coding environments. To address this challenge, we present Hierarchical Skill Evolution (HSE), which progressively distills procedural knowledge from coding trajectories through hierarchical evidence aggregation. HSE first extracts task-level evidence from individual trajectories, then aggregates recurring evidence within the same repository to form repository-specific skill candidates, and finally consolidates consistent patterns across repositories into transferable skills across repositories. The evolution is driven by an external writer-evaluator loop that generates and assesses skills while keeping the underlying coding model frozen. The loop extern
    
[^18]: 让推理可见：面向结构化推理任务中自适应脚手架的配置感知诊断建模

    Making Reasoning Visible: Configuration-Aware Diagnostic Modeling for Adaptive Scaffolding in Structured Reasoning Tasks

    [https://arxiv.org/abs/2610.35609](https://arxiv.org/abs/2610.35609)

    该论文提出一种配置感知的诊断建模框架，通过分析学生在交互式学习游戏中的动作序列使多步骤推理过程可视化，并发现推理策略主要随情境变化而非稳定个人特质，从而为结构化推理任务中的自适应脚手架教学提供支持。

    

    对学习者潜在的、多步骤推理过程进行建模是自适应辅导系统面临的核心挑战，这类系统往往依赖简单的对错判断，而非理解学生的思考方式。本研究以语法推理为应用实例，提出了一种诊断框架，通过分析学生在交互式学习游戏中的动作序列，使其逐步解决问题的推理过程变得可见。该框架计算五个可解释的策略维度（例如系统性纠错、随机猜测），从任务结构和行为模式中估计协调需求，并将推理划分为四种不同的状态。通过分析234名学生在42个形态句法练习中产生的36,234次交互，我们发现推理策略主要随情境变化而并非源于稳定特质（组内相关系数 = 0.14），且学生在低效状态下表现出高度持续性；具体而言，

    arXiv:2610.35609v1 Announce Type: cross  Abstract: Modeling the latent, multi-step reasoning processes of learners is a core challenge for adaptive tutoring systems, which often rely on simple right/wrong answers rather than understanding how students think. Instantiated here for grammatical reasoning, this study introduces a diagnostic framework that makes students' step-by-step problem-solving visible by analyzing their action sequences in an interactive learning game. The framework computes five interpretable strategy dimensions (e.g., systematic error correction, random guessing), estimates coordination demand from task structure and behavioral patterns, and classifies reasoning into four distinct states. Analyzing 36,234 interactions from 234 students working on 42 morphosyntactic exercises, we found that reasoning strategies vary primarily by context rather than stable traits (intraclass correlation = 0.14), and students show high persistence in unproductive states; specifically,
    
[^19]: 线性探针在方向旋转的情况下仍能从基础语言模型迁移到蒸馏语言模型

    Linear Probes Transfer from Base to Distilled Language Models despite Rotation

    [https://arxiv.org/abs/2610.35586](https://arxiv.org/abs/2610.35586)

    研究发现，尽管蒸馏推理模型和指令微调模型会使线性探针所依赖的安全/不安全提示分离方向旋转17°到35°，但在基础模型上训练的有害请求检测探针无需任何修改即可有效迁移到这些后继模型上，AUC变化极小。

    

    线性探针是一种读取语言模型内部激活的分类器，被广泛用于检测有害请求。它训练和运行成本低廉，已经训练过探针的审计人员希望在模型被后继版本替换时能复用它，而不是每次发布都重新训练。我们测试了探针的复用效果。对于四个基础模型，我们在基础模型上使用XSTest提示集训练一个探针，然后不加修改地将其应用于该基础模型的两个后继版本：DeepSeek-R1蒸馏推理模型和开发者发布的指令微调聊天模型。我们发现，后继模型将区分不安全提示与安全提示的方向旋转了17°到35°，但在XSTest上，迁移后的探针AUC变化仅为-0.011到+0.027；在从未用于训练的独立OR-Bench评估上，变化为-0.015到+0.053。在安全提示刻意（设计以模仿有害请求的）XSTest上……

    arXiv:2610.35586v1 Announce Type: new  Abstract: Linear probes, classifiers that read a language model's internal activations, are widely used to detect harmful requests. They are cheap to train and run, and an auditor who has trained one would like to reuse it when the model is replaced by a successor rather than retrain it for every release. We test how well probes can be reused. For four base models (Qwen2.5-Math-7B, Llama-3.1-8B, Qwen2.5-14B, Qwen2.5-32B) we train a probe on the base model with the XSTest prompt set and apply it, unchanged, to two successors of that base: the DeepSeek-R1 distilled reasoning model and the developer's instruction-tuned chat model. We find that the successors rotate the direction separating unsafe from safe prompts by 17$^\circ$ to 35$^\circ$, yet on XSTest the transferred probe's AUC changes by only $-0.011$ to $+0.027$, and on an independent OR-Bench evaluation, never used for training, by $-0.015$ to $+0.053$. On XSTest, whose safe prompts delibera
    
[^20]: 积分奖励而非采样奖励：面向全双工语音模型中打断与附和的窗口似然方法

    Integrate the Reward, Don't Sample It: Window Likelihood for Interruptions and Backchannels in Full-Duplex Speech Models

    [https://arxiv.org/abs/2610.35581](https://arxiv.org/abs/2610.35581)

    该论文发现基于整轮展开的GRPO时机奖励在训练全双工语音模型时存在零梯度陷阱与信用泄漏两大缺陷，并提出“窗口似然”方法，通过对停止决策进行闭式积分直接最大化正确让出/保持的对数概率，在HumDial-FDBench上取得了最佳的让出/保持F1表现。

    

    全双工语音对话模型必须在用户插话（打断）时让出话轮，而在用户仅作附和时保持说话。带有时机奖励的序列级强化学习（GRPO）是常见的做法；我们在四个开源模型上测试了其整轮展开的形式，发现了两类失效：二值窗口奖励恰恰在最需要学习的窗口上给出零优势（零梯度陷阱），而每次展开仅有的单一评分被摊派到奖励并未以其定义的收听状态决策上（信用泄漏），导致模型对着沉默继续说话或停止回答。这两种失效都源于对一个本可由少数二值控制决策的已知函数表达的奖励进行采样。我们提出窗口似然（WL）：在每个窗口采样一条续写，但对其经过的停止决策以闭式形式积分，并最大化正确行动的对数概率——让出对应噪声-OR形式，保持对应生存乘积形式。在 HumDial-FDBench 上，WL 取得了最佳的让出/保持 F1 表现。

    arXiv:2610.35581v1 Announce Type: cross  Abstract: A full-duplex spoken dialogue model must yield when the user cuts in and hold when the user merely backchannels. Sequence-level RL with a timing reward (GRPO) is the usual recipe; we test its whole-rollout form on four open models and find two failures. A binary window reward gives zero advantage on exactly the windows that most need learning (a zero-gradient trap), and the one score per rollout is spread over listening-state decisions the reward is not written in terms of (credit leakage), so the model talks into silence or stops answering. Both come from sampling a reward that is a known function of a few binary control decisions. We propose \emph{window likelihood} (\WL): sample a continuation per window, but integrate in closed form over the stop decisions it visits and maximise the log-probability of acting correctly---a noisy-OR for yielding, a survival product for holding. On HumDial-FDBench, \WL\ gives the best yield/hold F1 on
    
[^21]: EXCISE：在医学问答中定位并切除干扰性句子

    EXCISE: Locating and Excising Distracting Sentences in Medical Question Answering

    [https://arxiv.org/abs/2610.35558](https://arxiv.org/abs/2610.35558)

    提出推理时方法EXCISE，利用模型隐藏状态识别与病例不一致的句子并阻断其对答案的影响，从而在不需预先定位干扰句的情况下提升医学问答的鲁棒性。

    

    语言模型越来越多地被考虑用于临床决策支持，其答案应当基于临床证据，并对无关信息保持鲁棒性。先前的研究表明，单个无关句子即可降低医学模型的准确性，但其影响在模型内部何处能够被消除，以及在不明确是哪个句子的前提下如何消除这种影响，仍属未知。我们发现，阻断对已知干扰句的注意力可以恢复正确答案，而单纯的注意力头掩蔽则可能抑制正确答案。我们还发现，模型的隐藏状态会将干扰句标记为最不符合病例的句子，即使模型实际上遵循了该句子。为了在不明确是哪个句子的情况下消除干扰句的影响，我们提出了EXCISE，这是一种推理时方法，它会阻断对答案影响最大的句子，但前提是模型的隐藏状态将其标记为与病例不一致。在MedQA-TD（带有目标干扰项的MedQA）基准上，……

    arXiv:2610.35558v1 Announce Type: new  Abstract: Language models are increasingly considered for clinical decision support, where their answers should be grounded in clinical evidence and remain robust to irrelevant information. Prior work shows that a single irrelevant sentence can lower the accuracy of medical models, but not where inside the model its effect can be removed, or how to do so without knowing which sentence it is. We find that blocking attention to a known distractor restores the answer, whereas head masking can suppress correct answers. We also show that the model's hidden states mark the distractor as the sentence that least fits the case even when the model follows it. To remove a distractor's effect without knowing which sentence it is, we propose EXCISE, an inference-time method that blocks the sentence most influencing the answer, but only if the model's hidden states mark it as inconsistent with the case. On MedQA-TD (MedQA with Targeted Distractors), a benchmark
    
[^22]: LoopFold：循环Transformer中循环深度的并行化

    LoopFold: Parallelizing Recurrent Depth in Looped Transformers

    [https://arxiv.org/abs/2610.35556](https://arxiv.org/abs/2610.35556)

    LoopFold通过轻量级预测器提前启动较深的循环，并用校正器利用先前循环结果对其进行精化，将循环Transformer中多个循环深度的计算折叠为并行批次，从而在不损失精度的前提下降低解码延迟。

    

    循环Transformer（Looped Transformers）通过重复应用共享主干网络来增加每个token的计算量，提供了一种在固定主干参数数量下提升语言建模和下游任务性能的方法。然而，每个循环依赖于前一个循环的结果，因此增加循环深度也会增加解码延迟。现有方法包括Parallel Loop Transformers（并行循环Transformer），它通过在不同token位置之间重叠循环计算来降低延迟，但在较大深度下其边际精度收益迅速递减。为解决这一挑战，我们提出LoopFold，它将多个循环深度的计算折叠为针对同一token前缀的并行批次。一个轻量级预测器用于估计后续循环的输入，使它们能够在较早的循环完成之前就开始执行；随后一个校正器利用较早循环新产生的结果来优化这些后续循环的输出。所有组件仅通过最终的下一个token预测目标进行联合训练（摘要在此处截断）。

    arXiv:2610.35556v1 Announce Type: new  Abstract: Looped Transformers increase computation per token by repeatedly applying a shared backbone, offering a way to improve language modeling and downstream task performance at a fixed backbone parameter count. However, each loop depends on the preceding loop, so increasing recurrent depth also increases decoding latency. Existing approaches include Parallel Loop Transformers, which reduce latency by overlapping loop computation across token positions, but yield rapidly diminishing marginal accuracy gains at larger depths. To address this challenge, we introduce LoopFold, which folds computation at multiple recurrent depths into parallel batches for the same token prefix. A lightweight predictor estimates inputs to later loops so they can start before earlier loops finish. A corrector then refines the outputs of these later loops using newly available results from earlier loops. All components are trained jointly using only the final next-tok
    
[^23]: StateLoop：为循环Transformer设计的持久状态

    StateLoop: Persistent State for Looped Transformers

    [https://arxiv.org/abs/2610.35544](https://arxiv.org/abs/2610.35544)

    StateLoop通过为每个token引入一个由可学习门控控制的持久状态来在循环迭代之间保留信息，从而降低语言建模困惑度并提升零样本任务准确率。

    

    循环Transformer通过重复应用共享的主干网络，为语言建模提供了一种参数高效的方法。原始循环Transformer直接将每个主干网络的输出作为下一次迭代的输入。这些输出中可能包含在若干次迭代之后才有用的信息，但要在整个循环过程中保留这些信息，完全依赖于单一的隐藏状态。我们提出了StateLoop，它为每个token维护一个额外的持久状态，以便在迭代之间保留信息。在每次迭代中，一个可学习的门控机制控制前一状态的保留程度，以及从主干网络输出中融入的信息量。更新后的状态被投影回来并加到主干网络输出上，用于后续迭代和最终预测。这种设计降低了Pile数据集上的困惑度，并在八个零样本任务上比原始循环Transformer取得了更高的平均准确率。其相对于原始循环Transformer的困惑度优势随着训练循环深度的增加而增长。

    arXiv:2610.35544v1 Announce Type: new  Abstract: Looped Transformers offer a parameter-efficient approach to language modeling by repeatedly applying a shared backbone. Vanilla looped Transformers use each backbone output directly as the input to the next iteration. These outputs may contain information useful several iterations later, but retaining it across the loop depends entirely on the single hidden state. We introduce StateLoop, which maintains one additional persistent state per token to retain information across iterations. At each iteration, a learned gate controls how much of the previous state is retained and how much information from the backbone output is incorporated. The updated state is projected back and added to the backbone output for subsequent iterations and final prediction. This design reduces Pile perplexity and improves average accuracy across eight zero-shot tasks over Vanilla Looped. Its perplexity advantage over Vanilla Looped grows with trained loop depth,
    
[^24]: 语言模型中少样本概念归纳的算法性解释

    An Algorithmic Account of Few-Shot Concept Induction in Language Models

    [https://arxiv.org/abs/2610.35533](https://arxiv.org/abs/2610.35533)

    本文提出摊销方向强化（ADA）这一算法性假设，解释了语言模型在少样本概念归纳中如何确定特征相关性，即通过利用关于哪些特征有助于发现共性的摊销知识来构建特征空间中示例的摘要表示。

    

    模型和心智不仅编码描述实体的特征，还能具体表示该实体的哪些特征与给定情境相关。这一核心能力是语言模型上下文学习以及心智中多种认知现象的基础，然而智能系统如何识别和提升相关特征仍不清楚。在这项工作中，我们提出了摊销方向强化，这是一个算法性假设，为少样本概念归纳情境下相关性如何确定提供了解释——少样本概念归纳是指从少量正例中推断出潜在概念的能力。在ADA框架下，概念归纳通过在一个特征空间中形成示例的摘要表示来进行，而该特征空间由关于哪些特征通常有助于发现项目间共性的摊销知识所塑造。我们使用少样本概念归纳任务在语言模型中研究并验证了ADA。

    arXiv:2610.35533v1 Announce Type: new  Abstract: Both models and minds not only encode features that describe an entity, but can specifically represent which features of that entity are relevant for a given context. This core capability underlies in-context learning in language models and a variety of cognitive phenomena in minds, yet it is unclear how intelligent systems identify and promote relevant features. In this work, we propose Amortized Direction Accentuation (ADA), an algorithmic hypothesis that provides an explanation for how relevance is determined in the context of few-shot concept induction --- the ability to infer a latent concept from a small number of positive examples. Under ADA, concept induction proceeds by forming a summary representation of the examples in a feature space that is shaped by amortized knowledge about which features tend to be useful for finding commonalities between items. We investigate and validate ADA in language models using a few-shot concept i
    
[^25]: S3AC：面向大语言模型推理的稳定化、选择性与可扩展的演员-评论家算法

    S3AC: Stabilized, Selective, and Scalable Actor-Critic for LLM Reasoning

    [https://arxiv.org/abs/2610.35518](https://arxiv.org/abs/2610.35518)

    S3AC提出了一种实用的评论家训练方案，为长思维链推理提供token级优势估计和提示级基线，并结合难度感知提示采样（DAPS），从而解决无评论家RLVR方法在信用分配与基线估计上的不足，提升大语言模型强化学习训练的稳定性与可扩展性。

    

    在具有可验证奖励的强化学习（RLVR）中，无评论家方法被广泛应用于长思维链推理，以避免困难且代价高昂的评论家训练。常见的替代方法缺乏token级别的信用分配，且通常依赖分组采样来估计特定提示的基线。评论家可以同时解决这两个局限性，但长CoT的评论家训练受到以下因素阻碍：对不同长度轨迹的覆盖不足，以及新初始化的价值头主导早期梯度。我们提出了面向大语言模型推理的S3AC（稳定化、选择性与可扩展的演员-评论家算法），其核心是一个实用的评论家训练方案。训练好的评论家可提供token特定的优势估计和提示特定的基线。利用提示级基线来估计难度，我们提出了难度感知提示采样。DAPS将这一难度估计与结果历史相结合，在生成之前选择提示，从而减少……

    arXiv:2610.35518v1 Announce Type: cross  Abstract: Critic-free methods are widely used in reinforcement learning with verifiable rewards (RLVR) for long chain-of-thought (CoT) reasoning, avoiding difficult and costly critic training. Common alternatives lack token-level credit assignment and typically rely on grouped rollouts to estimate prompt-specific baselines. A critic can address both limitations, but long-CoT critic training is hindered by insufficient coverage of trajectories with diverse lengths and early gradients dominated by the newly initialized value head. We introduce S3AC (Stabilized, Selective, and Scalable Actor-Critic) for LLM reasoning, built around a practical critic-training recipe. The trained critic provides token-specific advantages and prompt-specific baselines. Using the prompt-level baseline to estimate difficulty, we propose Difficulty-Aware Prompt Sampling (DAPS). DAPS combines this estimate with outcome history to select prompts before generation, reducing
    
[^26]: MATCHA：一个中文多障碍心理健康数据集及基于社交媒体的跨语言分析

    MATCHA: A Chinese Multi-Disorder Mental Health Dataset with Cross-Lingual Analyses on Social Media

    [https://arxiv.org/abs/2610.35512](https://arxiv.org/abs/2610.35512)

    MATCHA是首个中文多障碍心理健康数据集，将七种DSM-5障碍的微博超话社区与用户公开时间线关联，支持跨语言、跨文化的社交媒体心理健康研究。

    

    社交媒体上的心理健康研究长期被英语数据集和以DSM-5为中心的资源所主导，这限制了对在线心理健康表达的跨语言与跨文化理解。我们提出了MATCHA，这是首个中文多障碍心理健康数据集，它将特定障碍相关的微博超话社区与用户更广泛的公开时间线相连接。MATCHA涵盖七种DSM-5障碍及一个对照组，包含46,025名队列级用户和180万条帖子。该数据集通过大语言模型（LLM）辅助的标注流程，将具有明确自我披露证据的用户与更广泛的障碍相关参与者区分开来，并以严格去标识化、受控访问的形式发布。利用MATCHA，我们分析了各障碍在行为、语言、症状及生活事件方面的模式，并与Twitter-STMHD数据集进行了跨语言比较。我们的分析表明，微博超话发挥着……（原文在此处截断）

    arXiv:2610.35512v1 Announce Type: new  Abstract: Mental health research on social media has been dominated by English-language datasets and DSM-5-centered resources, limiting cross-lingual and cross-cultural understanding of online mental health expression. We introduce MATCHA, the first Chinese multi-disorder mental health dataset that links disorder-specific Weibo Super-Topic communities with users' broader public timelines. MATCHA covers seven DSM-5 disorders and a control group, comprising 46,025 cohort-level users and 1.8 million posts. The dataset distinguishes users with explicit self-disclosure evidence from broader disorder-related participants through an LLM-assisted annotation pipeline, and is released in a rigorously de-identified, controlled-access format. Using MATCHA, we analyze behavioral, linguistic, symptom-related, and life-event-related patterns across disorders, and conduct cross-lingual comparisons with Twitter-STMHD. Our analyses show that Weibo Super-Topics func
    
[^27]: 超越文字本身：面向视觉接地文化翻译的分布式证据解耦

    Beyond Words Alone: Disentangling Distributed Evidence for Visually Grounded Cultural Translation

    [https://arxiv.org/abs/2610.35482](https://arxiv.org/abs/2610.35482)

    本文提出 Culture-MMT 诊断基准，通过系统解耦社交媒体翻译中分散于文本、图像与文化知识中的证据，证明视觉证据对于实现文化适配的翻译至关重要。

    

    社交媒体使跨语言交流成为可能，但用户生成内容（UGC）的含义往往分布于文本、图像与文化知识之中，这给翻译带来了挑战。现有评估通常将图像视为可选内容，并假设文本信息已经足够，从而掩盖了翻译错误究竟是源于视觉证据的缺失还是文化适应的不足。我们提出了 Culture-MMT，一个用于社交媒体翻译中“分布式证据语用重建”的诊断基准。该基准包含498个经过精心筛选、去标识化的UGC实例，覆盖20个领域，并根据所需证据类型划分为符号、视觉和混合三类。我们还开发了一个与人类判断对齐的领域专用评审器 Judger，以及一个能够系统性改变翻译模型可用证据的受控评估框架。在17种模型配置上的实验表明，真实图像在视觉类任务上带来 +0.280 的性能提升，在混合类任务上带来 +0.197 的提升。

    arXiv:2610.35482v1 Announce Type: cross  Abstract: Social media enables cross-lingual communication, yet the meaning of user-generated content (UGC) is often distributed across text, images, and cultural knowledge, making translation challenging. Existing evaluations typically treat images as optional and assume textual information is sufficient, obscuring whether errors stem from missing visual evidence or inadequate cultural adaptation. We introduce \textbf{Culture-MMT}, a diagnostic benchmark for \textbf{distributed-evidence pragmatic reconstruction} in social-media translation. It contains 498 curated, de-identified UGC instances across 20 domains, categorized into Symbol, Visual, and Hybrid by their required evidence. We also develop a human-aligned domain-specific judge, Judger, and a controlled evaluation framework that systematically varies the evidence available to translators. Across 17 model configurations, real images improve performance by +0.280 on Visual and +0.197 on Hy
    
[^28]: GPDPO：通过群体帕累托支配策略优化从奖励冲突中学习

    GPDPO: Learning from Reward Conflicts via Group Pareto-Dominance Policy Optimization

    [https://arxiv.org/abs/2610.35480](https://arxiv.org/abs/2610.35480)

    提出GPDPO方法，通过识别冲突过滤后采样响应之间的帕累托支配关系并将其转化为辅助偏好信号，从而从奖励冲突中恢复被丢弃的成对监督信息，实现更稳定的多奖励策略优化。

    

    多奖励强化学习为将大语言模型（LLM）与多样化目标对齐提供了一个有前景的框架，但相互冲突的奖励信号会导致更新方向模糊，并阻碍稳定优化。现有方法通过奖励重加权、梯度协调或采样过滤等手段来处理此类冲突，但它们主要在单响应或群体级别的优化信号上运行，未能充分利用采样响应之间的关系信息。因此，经过冲突过滤的采样数据中可能仍含有在训练中被丢弃的有用监督信息。我们提出了群体帕累托支配策略优化（GPDPO），这是一种能够从经过冲突过滤的采样数据中恢复成对监督信号的多奖励策略优化方法。GPDPO识别采样响应之间的帕累托支配关系，并将其转换为辅助偏好信号，这些信号与原始的策略优化信号进行联合优化。

    arXiv:2610.35480v1 Announce Type: cross  Abstract: Multi-reward reinforcement learning offers a promising framework for aligning large language models (LLMs) with diverse objectives, but conflicting reward signals can induce ambiguous update directions and hinder stable optimization. Existing approaches address such conflicts through reward reweighting, gradient coordination, or rollout filtering, yet they primarily operate on per-response or group-level optimization signals and underexploit relational information among sampled responses. As a result, conflict-filtered rollouts may still contain useful supervision that is discarded during training. We propose Group Pareto-Dominance Policy Optimization (GPDPO), a multi-reward policy optimization method that recovers pairwise supervision from conflict-filtered rollouts. GPDPO identifies Pareto-dominance relations among sampled responses and converts them into auxiliary preference signals, which are jointly optimized with the original pol
    
[^29]: MANTIS：面向假新闻检测中实例级压力测试与最坏情况训练的多轴新闻变换方法

    MANTIS: Multi-Axis News Transformations for Instance-Wise Stress Testing and Worst-Case Training in Fake News Detection

    [https://arxiv.org/abs/2610.35475](https://arxiv.org/abs/2610.35475)

    该论文提出了多轴新闻变换基准MANTIS-Bench和实例自适应对抗框架MANTIS，通过针对每篇新闻在写作风格、情感表达和传播框架上进行动态最坏情况训练，显著提升了假新闻检测器在多样化改写攻击下的鲁棒性。

    

    自动化假新闻检测在社交媒体上至关重要，因为假新闻很容易被改写以欺骗检测器。然而，现有方法通常依赖于表面化的写作线索或单一风格的文本改写，导致在多样化的表面修改下鲁棒性较差。我们构建了MANTIS-Bench，这是一个多轴呈现攻击基准，将原始新闻与跨越写作风格、情感表达和传播框架三个维度的改写版本配对。我们进一步提出了MANTIS，一个实例自适应的对抗框架，能够针对每篇新闻文章进行动态的最坏情况训练，且在推理阶段零额外开销。这将假新闻防御从启发式的格式平滑重新构建为有针对性的、事实不变的信誉评估。在MANTIS-Bench上的实验表明，MANTIS在GossipCop数据集上相比SheepDog将联合攻击成功率（UASR）降低了3.55个百分点，并持续优于先前的单轴方法。

    arXiv:2610.35475v1 Announce Type: new  Abstract: Automated fake news detection is vital on social media, where fake news is easily rewritten to fool detectors. However, existing approaches typically rely on superficial writing cues or single-style text rewrites, resulting in poor robustness under diverse cosmetic modifications. We construct MANTIS-Bench, a multi-axis presentation-attack benchmark that pairs original news with rewritten views across writing style, emotional expression, and dissemination framing. We further propose MANTIS, an instance-adaptive adversarial framework enabling dynamic worst-case training tailored to each news article with zero inference-time overhead. This reframes fake news defense from heuristic format smoothing into targeted, fact-invariant credibility assessment. Experiments on MANTIS-Bench show that MANTIS reduces the Union Attack Success Rate (UASR) by 3.55 percentage points relative to SheepDog on GossipCop and consistently outperforms prior single-a
    
[^30]: 规范性文本中歧义的情境化度量方法

    A Contextual Measure of Ambiguity in Prescriptive Text

    [https://arxiv.org/abs/2610.35461](https://arxiv.org/abs/2610.35461)

    该论文提出一种情境化的歧义度量方法，通过量化利益相关者对规范性文本在不同场景中如何适用的分歧程度来衡量歧义，初步分析表明即使使用模拟评估者，该方法也与真实的人类分歧相关。

    

    法律、合同、规则、政策或建议等规范性文本中的歧义既可能是可取的（例如，当它使法律能够适应不断变化的情境，如新技术时），也可能是不可取的（例如，当它导致随意裁量或对法律解释的冲突时）。为了帮助权衡这些利弊，拥有能够预测此类结果的歧义量化方法将非常有价值，但现有的度量方法只捕捉文本在孤立状态下的感知方式，而非它在具体情境中如何被解释。在本文中，我们提出将歧义量化为一组利益相关者（“评估者”）就备选规范性文本（“政策”）在一组相关场景（“案例”）中各自如何适用所产生的分歧程度。基于两个政策领域，我们给出了初步分析，结果表明即使使用模拟评估者，这种方法 (i) 与真实的人类分歧相关，并且 (ii) 是……（摘要在此处被截断）

    arXiv:2610.35461v1 Announce Type: new  Abstract: Ambiguity in prescriptive text such as laws, contracts, rules, policies, or recommendations can be both desirable (e.g., if it enables a law to adapt to changing context, such as new technologies) and undesirable (e.g., if it leads to arbitrary discretion or conflict over the law's interpretation). To help navigate these trade-offs, it would be valuable to have methods for quantifying ambiguity that are predictive of such outcomes, but existing measures capture how text is perceived in isolation rather than how it is interpreted in context. In this paper, we propose quantifying ambiguity as the degree of disagreement among a set of stakeholders ("raters") on how alternative prescriptive texts ("policies") apply in each of a set of relevant scenarios ("cases"). Using two policy domains, we present preliminary analysis suggesting that, even when using simulated raters, this approach (i) correlates with real human disagreement, and (ii) is 
    
[^31]: Skill$^{\infty}$：面向智能体技能优化的递归自我改进

    Skill$^{\infty}$: Recursive Self-Improvement for Agent Skill Optimization

    [https://arxiv.org/abs/2610.35434](https://arxiv.org/abs/2610.35434)

    提出Skill$^{\infty}$框架，元智能体在遵循元技能优化任务技能的同时，用同一流程递归地优化元技能本身，且优化步骤由智能体自主定义和重写，从而实现真正的递归自我改进。

    

    递归自我改进（RSI）是指系统同时改进其改进方式本身，这是通往通用人工智能（AGI）道路上的一个长期愿景。语言智能体越来越多地将程序性知识存储在可复用的技能中。优化技能为实现RSI提供了一条直接途径，因为优化过程本身就可以被编码为一种技能。然而，现有的技能优化器遵循手工设计的固定流程，即使是那些进化元技能的方法，也将其局限于固定的流水线中，并需要针对每个任务重新学习。我们提出了Skill$^{\infty}$，它使技能优化能够递归地自我改进：一个元智能体遵循元技能来优化任务技能，并使用相同的流程来优化元技能本身。该优化过程由智能体主导：元智能体诊断执行轨迹、提出候选方案、请求独立评审并分配搜索预算。元技能定义了这些步骤，从而使自我改进能够重写这些步骤本身。

    arXiv:2610.35434v1 Announce Type: new  Abstract: Recursive self-improvement (RSI), in which a system also improves the way it improves, is a long-standing aspiration on the path to AGI. Language agents increasingly store procedural knowledge in reusable skills. Optimizing skills offers a direct way to achieve RSI, because the optimization procedure itself can be encoded as a skill. Existing skill optimizers, however, follow a hand-designed procedure. Even methods that evolve a meta-skill confine it to a fixed pipeline and relearn it for each task. We present Skill$^{\infty}$, which makes skill optimization recursively self-improving: a meta agent follows a meta-skill to optimize task skills and optimizes the meta-skill itself using the same procedure. The optimization is agent-led. The meta agent diagnoses execution traces, proposes candidates, requests independent reviews, and allocates the search budget. The meta-skill defines these steps, allowing self-improvement to rewrite them. T
    
[^32]: 探究Mimi码本的语音表征

    Probing the phon* representations of the Mimi codebooks

    [https://arxiv.org/abs/2610.35432](https://arxiv.org/abs/2610.35432)

    本文通过强制对齐、熵分析和音素归一化互信息（PNMI）对Mimi编解码器首个码本进行探测，发现其token与音素存在强关联（PNMI=0.84），但音素与token并非一一对应，而是由多个竞争性token共同表示。

    

    神经音频编解码器正越来越多地被用于对语音进行编码，以便与大语言模型集成。本文研究了Mimi编解码器第一个码本（CB0）中的语音表征。ABX式评估方法并不适合Mimi固定的80毫秒分词窗口，因为该窗口通常跨越多个音素；因此我们改用强制对齐、熵分析以及音素归一化互信息（PNMI）方法。基于TIMIT语音数据（58/55名说话人，SA1/SA2句子），研究发现单个token与特定音素存在强关联（PNMI = 0.84，高于随机水平），但音素并非与token一一对应，而是由多个相互竞争的token共同实现。

    arXiv:2610.35432v1 Announce Type: new  Abstract: Neural audio codecs are increasingly used to encode speech for integration with large language models. This paper investigates the phonetic representations in the first codebook (CB0) of the Mimi codec. ABX style evaluation is poorly suited to Mimi fixed 80 ms tokenization window, which often spans multiple phonemes; we instead use forced alignment, entropy analysis, and Phone-Normalized Mutual Information (PNMI). Using TIMIT speech (58/55 speakers, SA1/SA2), individual tokens are strongly associated with specific phonemes (PNMI = 0.84, above random), yet phonemes are realized by multiple competing tokens rather than a one-to-one mapping.
    
[^33]: INVAR-Agri：使用微调大语言模型检测农业机械多语言操作手册中的差异

    INVAR-Agri: Detecting Discrepancies in Multilingual Operator Manuals of Agricultural Implements with a Fine-Tuned LLM

    [https://arxiv.org/abs/2610.35423](https://arxiv.org/abs/2610.35423)

    该论文提出 INVAR-Agri 概念验证，在 EuroLLM-1.7B-Instruct 上训练并比较“回译后对比”与“双语直接检查”两种微调工作流程，以在 24 种语言的农业机械操作手册中自动检测涉及数字、单位、义务、否定和条件的翻译差异。

    

    农业机械的操作手册以多种语言发布，而标准要求由有资质的人员审查每一个译本，这使审查成为瓶颈环节。本文提出了一个概念验证，在同一大语言模型 EuroLLM-1.7B-Instruct 上，基于一家制造商的操作手册训练并比较两种工作流程。在方法 A 中，回译器将每种语言版本翻译回英语，由英语阅读器将其与英语原版进行对比；在方法 B 中，检查器同时阅读英语版本和译文并报告两者之间的差异。测试集涵盖 24 种语言，在其中植入涉及数字、单位、反义词、义务性表述、否定和条件等类型的错误，同时包含未修改的文本和保留原意的编辑内容，用于测试误报情况。经训练的回译器掩盖了许多植入的错误，尤其是被弱化的义务性表述，而翻译……（摘要在此处截断）

    arXiv:2610.35423v1 Announce Type: new  Abstract: Operator manuals of agricultural implements are delivered in many languages, and the standards require that competent persons review every translation, which makes this review the bottleneck. This paper presents a proof of concept that compares two training workflows on the same LLM, EuroLLM-1.7B-Instruct, trained on operator manuals of one manufacturer. In approach A, a back-translator translates each language version back into English, and an English reader compares it with the English version. In approach B, a checker reads the English version and the translation together, where the checker reports differences. The test sets contain errors planted in numbers, units, opposite words, obligations, negations, and conditions in 24 languages, and they include unmodified text and meaning-preserving edits for alerts and false alarms. The trained back-translator concealed many planted errors, most of all weakened obligations, while translation
    
[^34]: ImgMT：基于维基百科的图像内文本翻译评估数据集

    ImgMT: Wikipedia-based Dataset For Evaluating Text-in-Image Translation

    [https://arxiv.org/abs/2610.35416](https://arxiv.org/abs/2610.35416)

    该论文提出了ImgMT，一个从维基共享资源SVG插图中挖掘的大规模多语言图像内文本翻译数据集与基准，涵盖129种语言和4,886个语言对，并配合多层次评估协议发现源语言和书写系统会显著影响翻译质量。

    

    图像内文本翻译旨在翻译嵌入图像中的所有文本内容。在文本图像翻译（TIT）中，输出的是翻译后的文本；而图像内机器翻译（IIMT）则进一步将翻译结果渲染回图像中。我们提出了ImgMT，这是一个大规模多语言数据集和基准，从Wikimedia Commons上的本地化SVG插图中挖掘而成，专注于图表、地图和信息图。ImgMT包含35,612个对齐的图像对，分为1,855个图像集，涵盖129种语言、20多种书写系统以及4,886个语言对。我们提出了一个多层次评估协议，同时衡量源端的文本区域检测和OCR准确率以及目标端的翻译质量。我们在英语到X和X到英语的设置下，对多个多模态大语言模型和一个专门的TIT系统进行了基准测试，发现源语言和书写系统会显著影响TIT质量。最后，我们展示了ImgMT（摘要在此处截断）。

    arXiv:2610.35416v1 Announce Type: new  Abstract: Text-in-image translation aims to translate all textual content embedded in an image. In Text Image Translation (TIT), the output is translated text; In-Image Machine Translation (IIMT) additionally renders the translation back into the image. We introduce ImgMT, a massively multilingual dataset and benchmark mined from localized SVG illustrations on Wikimedia Commons, focusing on diagrams, maps, and infographics. ImgMT contains 35,612 aligned image pairs grouped into 1,855 image sets, spanning 129 languages, 20+ writing systems, and 4,886 language pairs.   We propose a multi-level evaluation protocol that measures source-side text-region detection and OCR accuracy alongside target-side translation quality. We benchmark several multimodal large language models and a dedicated TIT system in English->X and X->English settings, and find that the source language and writing system substantially affect TIT quality. Finally, we show that ImgMT
    
[^35]: 跨规模、深度与指令微调的情绪表示几何学研究

    The Geometry of Emotion Representations Across Scale, Depth, and Instruction Tuning

    [https://arxiv.org/abs/2610.35411](https://arxiv.org/abs/2610.35411)

    该研究证明Gemma 2系列从2B到27B、无论基础版还是指令微调版的大语言模型均共享一种可通过线性映射相互转换的核心情绪几何结构，且模型规模扩大使该几何结构方差分布于更多方向但仍保持与情绪效价的紧密关联。

    

    我们测量了涵盖Google Gemma 2系列的六个大语言模型的情绪几何结构。使用由Sofroniew等人（2026）在Claude Sonnet 4.5上建立、并由Hollowell（2026）在Gemma 2 27B基础模型上复现的提取流程，我们比较了Gemma 2 2B、9B和27B的基础版与指令微调变体之间的情绪几何结构。首先，所有六个模型单元都表现出核心的情绪几何结构。在两个模型单元之间拟合的线性映射能够将留出的情绪定位到目标模型中最接近其自身向量的位置。在全部十五个模型对（每个方向各拟合一次）中，该方法仅失败一次，且每个模型对的中位排名均为1。在清洗后的嵌入表向量上重新拟合的线性映射，在全部三十个有序模型对中都能以排名1恢复每一个留出的情绪。其次，规模扩大使几何结构的方差分布在更多方向上，同时并未削弱其与情绪效价的联系。主成分的方差占比从31.6%（2B）降至29.8%（9B）再到26.7%（27B），

    arXiv:2610.35411v1 Announce Type: new  Abstract: We measure the emotion geometry of six LLMs spanning Google's Gemma 2 family. Using an extraction pipeline established by Sofroniew et al. (2026) on Claude Sonnet 4.5 and replicated by Hollowell (2026) on Gemma 2 27B base, we compare emotion geometry across base and instruction-tuned variants of Gemma 2 2B, 9B, and 27B. First, all six model cells exhibit the core emotion geometry. A linear map fit between two cells places held-out emotions nearest their own vectors in the target model. Across all fifteen model pairs, fit in one direction each, this failed exactly once, and the median rank was 1 in every pair. Linear maps refit on cleaned embedding-table vectors recover every held-out emotion at rank 1 in all thirty ordered pairs. Second, scale spreads the geometry's variance across more directions without weakening its tie to valence. The leading component's share of variance falls from 31.6 percent (2B) to 29.8 (9B) to 26.7 (27B) in the
    
[^36]: 重构记录与原生GDELT全球知识图谱记录的一致性：一个字段级基准

    Agreement Between Reconstructed and Native GDELT GKG Records: A Field-Level Baseline

    [https://arxiv.org/abs/2610.35358](https://arxiv.org/abs/2610.35358)

    研究者开发了开源工具 gdelt-regkg 来重构 GDELT 全球知识图谱记录，并对近 20 万篇文章进行了字段级一致性评估，首次为研究人员自行生成兼容 GKG 记录提供了量化基准。

    

    全球事件、语言与语气数据库（GDELT）的全球知识图谱（GKG）通过命名实体、主题、定量与时间特征、引文、语气以及基于词典的内容度量来刻画新闻文章的特征。其代码手册定义了记录格式，但生成记录的软件并未公开，因此研究人员无法为自己收集的文章生成兼容的记录。我们构建了一个开源重构版本 gdelt-regkg，并测量了它与 GDELT 存档（原生）记录的一致程度，评估对象涵盖来自 8,729 个出版商（2020-2025 年）的 193,550 篇英文文章，以及五种其他语言的 2,500 篇文章。一致性在很大程度上取决于具体字段。英文语气（正面词减去负面词所占百分比）的平均绝对误差（MAE）为 1.79 个百分点，一致性相关系数为 0.75。身份识别一致性（F1）方面，人物为 0.65，组织为 0.42，地点为 0.65。

    arXiv:2610.35358v1 Announce Type: new  Abstract: The Global Knowledge Graph (GKG) of the Global Database of Events, Language, and Tone (GDELT) characterizes news articles using named entities, topics, quantitative and temporal features, quotations, tone, and dictionary-based content measures. Its codebook defines the record format, but the software that produces the records is not public, so researchers cannot generate compatible records for articles they collect themselves. We built an open-source reconstruction, gdelt-regkg, and measured how closely it agrees with GDELT's archived (native) records for 193,550 English articles from 8,729 publishers (2020-2025) and 2,500 articles in five other languages. Agreement depends strongly on the field. English tone, the percentage of positive minus negative words, has a mean absolute error (MAE) of 1.79 points and a concordance correlation coefficient of 0.75. Identity agreement (F1) is 0.65 for persons, 0.42 for organizations, and 0.65 for lo
    
[^37]: 约束解码：有害还是有益？重新审视其对大语言模型性能的影响

    Constrained Decoding: Harmful or Beneficial? Revisiting Its Impact on LLM Performance

    [https://arxiv.org/abs/2610.35328](https://arxiv.org/abs/2610.35328)

    该研究将约束解码划分为格式、有效性和推理三类约束，并通过理论与跨四个模型家族的因果消融实验证明，性能下降主要源于提示与输出格式的不等效而非解码器屏蔽本身，而适当的约束（尤其是有效性约束）反而能保持甚至提升模型性能与可靠性。

    

    约束解码保证大语言模型（LLM）产生符合预定义格式的输出，但人们通常假设它会固有地损害推理能力。我们通过一个包含三种约束类型的分类体系来检验这一假设：*格式约束*（强制输出结构）、*有效性约束*（屏蔽无效答案）以及*推理约束*（限制推理轨迹）。我们从理论上证明，适当的约束可以通过消除无效路径来保持甚至提升性能。在四个模型家族和十一个开源模型上，使用最先进的约束解码策略在多样化基准上进行的因果消融实验表明，表观的性能下降主要源于非等效的提示词和输出格式，而非解码器层面的屏蔽。同时，有效性约束通过消除无效答案，提高了输出的可靠性和可提取性。

    arXiv:2610.35328v1 Announce Type: cross  Abstract: Constrained decoding guarantees that Large Language Models (LLMs) produce outputs conforming to predefined formats, but is often assumed to inherently degrade reasoning. We examine this assumption through a taxonomy of three constraint types: \emph{format constraints}, which enforce output structure; \emph{validity constraints}, which mask invalid answer; and \emph{reasoning constraints}, which restrict reasoning trajectories. We theoretically show that appropriate constraints can preserve or improve performance by eliminating invalid paths. Across four model families and eleven open-source models, causal ablations on diverse benchmarks using state-of-the-art constrained decoding strategies reveal that apparent performance degradation largely stems from non-equivalent prompting and output formats rather than from decoder-level masking. Meanwhile, validity constraints improve reliability and extractability by eliminating invalid answers
    
[^38]: CRIM：面向高效个性化检索的可组合与可复用物品记忆

    CRIM: Composable and Reusable Item Memory for Efficient Personalized Retrieval

    [https://arxiv.org/abs/2610.35323](https://arxiv.org/abs/2610.35323)

    CRIM通过将物品压缩为与用户和查询无关的可复用连续记忆，并与请求级历史建模解耦，在推理时按交互顺序组合缓存记忆生成个性化检索表示，从而大幅降低大语言模型在个性化搜索与推荐中的训练与推理成本。

    

    在个性化搜索与推荐任务中，大语言模型会产生高昂的训练与推理成本，原因在于不断增长的交互历史延长了请求输入，并使相同的物品文本被反复暴露给模型。我们提出了可组合与可复用物品记忆（Composable and Reusable Item Memory, CRIM），这是一个将物品级语义编码与请求级历史建模相分离的框架。一小组可学习的摘要词元将每个物品压缩为紧凑的连续记忆，同时片段级读取词元（REC词元）在交互历史内部引入预测性监督。这些记忆独立于用户和查询，因此可以在不同请求之间复用。在推理阶段，CRIM按交互顺序组合缓存的记忆，并结合可选的查询来生成个性化检索表示，在减少重复物品文本编码的同时保留与请求相关的上下文信息。

    arXiv:2610.35323v1 Announce Type: cross  Abstract: Large language models incur substantial training and inference costs in personalized search and recommendation, as growing interaction histories lengthen request inputs and repeatedly expose the same item text to the model. We propose \textbf{C}omposable and \textbf{R}eusable \textbf{I}tem \textbf{M}emory (CRIM), a framework that separates item-level semantic encoding from request-level history modeling. A small set of learnable summary tokens compresses each item into compact continuous memories, while segment-level readout tokens (REC tokens) introduce predictive supervision within the interaction history. These memories are independent of users and queries and can therefore be reused across requests. At inference time, CRIM composes cached memories in interaction order and incorporates an optional query to produce a personalized retrieval representation, reducing repeated item-text encoding while retaining request-dependent contextu
    
[^39]: 模型用到了那条信息吗？反事实训练提升语言模型的自我归因能力

    Did the Model Use That? Counterfactual Training Boosts Self-Attribution in Language Models

    [https://arxiv.org/abs/2610.35295](https://arxiv.org/abs/2610.35295)

    提出CREDIT方法，通过反事实训练直接监督模型最终答案的自我归因，使语言模型能够准确识别并声明其回答是否受到了所提供信息的影响。

    

    对于可解释、可问责的语言模型而言，一个根本性的挑战是揭示外部信息如何影响其回答。本文研究了模型是否具备“自我归因”能力：即模型是否能够在且仅在其对问题的回答受到随问题一同提供的信息影响时，明确指出这种影响。此外，我们探索了如何通过反事实训练来改进自我归因能力，该训练明确地教导模型区分哪些答案依赖所提供的信息、哪些不依赖。我们发现，在十一个前沿模型中，大多数模型很少在回答中归因那些实际产生影响的信息，而其余模型则归因了并未产生影响的信息。将“致谢声明”训练纳入推理过程也无法解决这一问题：因为致谢并不会传递到最终答案中。我们提出了CREDIT方法，利用反事实标签直接监督最终答案的自我归因，而无需……（摘要原文在此处截断）

    arXiv:2610.35295v1 Announce Type: cross  Abstract: A fundamental challenge for explainable and accountable language models is revealing how external information influences their answers. This paper examines whether models are self-attributive: that is, whether they indicate when -- and only when -- their responses to questions are influenced by information provided alongside the question. Furthermore, we explore how self-attribution can be improved through counterfactual training, which explicitly teaches models to distinguish between answers that do and do not depend on supplied information.   We found that most of the eleven frontier models rarely attribute influential information in their answers, and the rest attribute non-influential information. Training acknowledgment into reasoning does not fix this: the acknowledgment does not reach the final answer.   We introduce CREDIT, which directly supervises final-answer self-attribution with counterfactual labels, without the need for 
    
[^40]: 线路上的声音

    The Voice on the Line

    [https://arxiv.org/abs/2610.35272](https://arxiv.org/abs/2610.35272)

    该论文训练了20毫秒延迟、能从通话本身注册来电者的因果语音隔离器（掩码transformer），并发现将WavLM-Large权重因果化并与模型联合训练能显著提升效果，将开发集错误率从11.44%降至10.55%。

    

    通过一个麦克风收听的转录器也会听到麦克风附近的所有人。我们训练了因果语音隔离器：参数量分别为1.4亿、2.43亿和3.74亿的掩码transformer，延迟为20毫秒，直接从通话本身注册来电者的声音。短时间的评估窗口会掩盖真实通话中暴露出的两种失败：当另一个声音单独说话一段时间后遗忘来电者，以及损害正在单独说话的来电者的语音。带有话轮转换的长通话、超出注意力窗口的记忆、真实的耳机串音以及漂移的来电者音量水平能够同时应对这两个问题，但每一种方法都会以牺牲一方为代价换取另一方。将WavLM-Large自身的权重改造为因果形式并与模型联合训练，能够改变这种权衡：采用这些权重的M模型（5.55亿参数）将开发集上的错误率从11.44%降至10.55%；从零开始训练以模仿WavLM的学生模型则无法做到这一点，而在多得多的音频上预训练的w2v-BERT 2.0和Whisper编码器的帮助较小或毫无帮助。单独说话的来电者在录音开始时损失最大：（摘要在此处被截断）

    arXiv:2610.35272v1 Announce Type: cross  Abstract: A transcriber listening through one microphone also hears everyone near it. We train causal voice isolators: masking transformers of 140M, 243M and 374M parameters, 20 ms of latency, enrolling the caller from the call itself. Short evaluation windows hide two failures that real calls expose: forgetting the caller when another voice talks alone for a while, and damaging a caller who is alone. Long calls with turns, memory beyond the attention window, real headset crosstalk and drifting caller levels address both, but each trades one against the other. WavLM-Large's own weights, made causal and trained with the model, move the trade: M with them (555M parameters) lowers the error on our development sets from 11.44% to 10.55%; a student trained from scratch to imitate WavLM does not, and w2v-BERT 2.0 and Whisper's encoder, pre-trained on far more audio, help less or not at all. A lone caller loses most at the start of a recording: no trai
    
[^41]: MechAlign：在数学形式化中评估项目级别的语义对齐

    MechAlign: Evaluating Semantic Alignment at the Project Level in Mathematical Formalization

    [https://arxiv.org/abs/2610.35269](https://arxiv.org/abs/2610.35269)

    MechAlign 提出了一个项目级语义对齐审查框架，通过依赖提取工具 LeanTrace 构建语句依赖图并结合 LLM 评估，检验 Lean 形式化命题是否忠实表达了手稿中的数学断言，并定位报告其中的差异。

    

    Lean 能够验证一个形式化证明在所声明的假设下确立了其所述命题，但并不检查该命题是否忠实地表达了手稿中的数学断言。评估这种对应关系需要通过其项目依赖来解读主要定理，而这些依赖可能分散在一个庞大的开发项目中。我们提出了 MechAlign，这是一个项目级别的语义对齐审查框架，它将形式化依赖提取与基于大语言模型（LLM）的评估相结合。其提取工具 LeanTrace 从详细阐述的声明中构建语句依赖图，保留定义体和结构约束，同时排除证明体，并在明确的库边界处停止。审查者利用这一上下文来调查对齐情况，并报告带有定位证据的差异。我们对来自 173 个 OpenAI Math 项目的 201 个主要结果进行了初步审查，使用所提供的非形式化信息……（摘要原文截断）

    arXiv:2610.35269v1 Announce Type: new  Abstract: Lean verifies that a formal proof establishes its stated proposition under declared assumptions, but does not check whether that proposition faithfully expresses the mathematical claim in a manuscript. Assessing this correspondence requires interpreting the main theorem through its project dependencies, which may be scattered throughout a large development. We introduce MechAlign, a framework for project-level semantic alignment review that combines formal dependency extraction with LLM-based assessment. Its extraction tool, LeanTrace, constructs a statement dependency graph from elaborated declarations, retaining definition bodies and structure constraints while excluding proof bodies and stopping at an explicit library boundary. The reviewer uses this context to investigate alignment and report discrepancies with localized evidence. We conduct a preliminary review of 201 main results from 173 OpenAI Math projects, using supplied inform
    
[^42]: 习语翻译质量评估的现状

    The Lay of the Land in Idiom-Translation Quality Estimation

    [https://arxiv.org/abs/2610.35261](https://arxiv.org/abs/2610.35261)

    该研究构建了首个用于习语翻译质量评估的平行数据集，涵盖九种目标语言的直译与地道译文，并揭示现有QE指标和LLM普遍存在直译偏好，倾向于将习语的直译评为优于地道翻译。

    

    习语长期以来一直是机器翻译面临的挑战。尽管大型语言模型（LLM）在处理习语方面已被证明越来越得心应手，但质量评估（QE）系统是否跟上了这一步伐、能否可靠地将习语的地道翻译评得高于错误的直译，仍不清楚。我们推出了首个平行数据集，将包含习语的英语句子与九种目标语言下的直译和地道翻译配对，并扩展为多句段落以及一个分级直译程度连续体。利用该数据集，我们对来自COMET和MetricX系列的十一种传统QE指标，以及来自Qwen-3、Gemma-4和OLMo-3系列的十个用作GEMBA式评判者的LLM进行了基准测试。我们发现，大多数系统，尤其是较旧的指标和较小的模型，表现出明显的直译偏好，将习语的直译评分高于地道翻译。更高的参数量和更新的训练集似乎能够缓解这种偏差。

    arXiv:2610.35261v1 Announce Type: new  Abstract: Idioms have long been a challenge for machine translation, and while large language models (LLMs) have proven increasingly adept at handling them, whether quality estimation (QE) systems have kept pace and can reliably score idiomatic renderings of these expressions above faulty literal ones remains unclear. We introduce the first parallel dataset pairing idiom-bearing English sentences with both literal and idiomatic translations across nine target languages, augmented into multi-sentence segments and a graded literality continuum. Using this dataset, we benchmark eleven conventional QE metrics from the COMET and MetricX families and ten LLMs from the Qwen-3, Gemma-4 and OLMo-3 families used as GEMBA-style judges. We find that most systems, especially older metrics and smaller models, display a marked literality bias, rating literal translations of idioms above idiomatic ones. Higher parameter counts and newer training sets appear to re
    
[^43]: 形式重于内容？基于大语言模型的同行评审中的语言偏见

    Style over Substance? Linguistic Bias in LLM-based Peer Review Copy title

    [https://arxiv.org/abs/2610.35254](https://arxiv.org/abs/2610.35254)

    该论文提出了一个受控实验框架，通过基于语言学文献推导的细粒度改写指令，系统性地改变论文的语言风格（母语/非母语特征与高/低置信度特征），以研究基于大语言模型的同行评审是否会受到与科学内容无关的语言偏见的影响。

    

    近年来，向主要人工智能会议提交的论文数量急剧增加，大幅加重了研究人员的同行评审工作量。与此同时，大语言模型（LLM）正越来越多地被用于支持同行评审，无论是被个别审稿人使用，还是被会议组织方采用。尽管LLM的能力已有显著提升，但其评估结果可能会受到与科学内容无关的语言特征的影响。在这项工作中，我们引入了一个受控实验框架，通过系统地改变研究论文的语言风格来研究这类效应。具体而言，我们对论文进行改写，使其分别呈现（i）与英语母语或非母语写作相关的语言特征，以及（ii）高置信度或低置信度的语言特征。为了构建这些干预措施，我们以语言学文献为基础推导出细粒度的改写指令，并将其应用于……

    arXiv:2610.35254v1 Announce Type: new  Abstract: The number of paper submissions to major AI conferences has increased dramatically in recent years, substantially increasing the peer-review workload of researchers. At the same time, large language models (LLMs) are increasingly used to support peer review, both by individual reviewers and by conference organizations. While LLM capabilities have improved considerably, their evaluations may be influenced by linguistic characteristics that are unrelated to scientific content. In this work, we introduce a controlled experimental framework for studying such effects by systematically perturbing the linguistic style of research papers. Specifically, we rewrite papers to exhibit (i) linguistic features associated with native or non-native English writing, and (ii) high- or low-confidence linguistic features. To construct these interventions, we derive fine-grained rewriting instructions grounded in the linguistic literature and apply them usin
    
[^44]: 超越视觉-语言模型：用于艺术品情感识别的视觉-语义情感图

    Beyond Vision-Language Models: Visual-Semantic Affect Graphs for Emotion Recognition in Artworks

    [https://arxiv.org/abs/2610.35244](https://arxiv.org/abs/2610.35244)

    本文提出视觉-语义情感图（VSAG）框架，通过色彩、笔触、构图、光影、线条等可解释特征节点构建图结构来学习特征与情感的关联，在艺术品情感识别任务上超越了直接使用视觉-语言模型进行推理的方法。

    

    艺术品中的情感识别仍然具有挑战性，因为情感意义往往是主观的、隐含的，并由视觉形式与语义关联之间的关系所塑造。虽然视觉-语言模型提供了强大的多模态表示，但直接的图像到情感预测可能忽略了支持情感解读的形式、语义和情感线索之间的显式关系。本文提出了视觉-语义情感图（VSAG），这是一个基于图的框架，通过可解释的特征节点来表示艺术品，包括色彩、笔触、构图、光影、线条质感以及提取出的视觉实体。该图学习特征与情感之间的关联以预测主导情感，并通过一个可选的层次化辅助情感模块对效价-唤醒结构进行建模。我们在EmoArt-5k数据集上，将VSAG和VSAG-TopK与使用Qwen3-VL和InternVL3.5的直接视觉-语言模型推理进行了对比评估。结果表明……

    arXiv:2610.35244v1 Announce Type: cross  Abstract: Emotion recognition in artworks remains challenging because affective meaning is often subjective, implicit, and shaped by relationships between visual form and semantic associations. While vision-language models provide strong multimodal representations, direct image-to-emotion prediction may overlook explicit relations among formal, semantic, and affective cues that support emotional interpretation. This paper proposes Visual-Semantic Affect Graphs (VSAG), a graph-based framework that represents artworks through interpretable feature nodes, including color, brushstroke, composition, light-shadow, line quality, and extracted visual entities. The graph learns feature-emotion associations for dominant emotion prediction, with valence-arousal structure modeled through an optional hierarchical auxiliary affect module. We evaluate VSAG and VSAG-TopK against direct VLM inference using Qwen3-VL and InternVL3.5 on EmoArt-5k. Results show that
    
[^45]: 在无正式答案之处实现可训练的边界保持：证据充分性判断中奖励训练的受控敲入研究

    Trainable Boundary-Keeping Where No Formal Answer Exists: A Controlled Knock-In Study of Reward Training in Evidence-Sufficiency Judgment

    [https://arxiv.org/abs/2610.35226](https://arxiv.org/abs/2610.35226)

    该研究通过预注册的同底座受控敲入实验证明，在不存在可判定正确性奖励的证据充分性判断任务中，“先SFT后GRPO”的边界保持训练配方能将模型的自发正确拒绝率从25%提升至100%，而乱序配对的安慰剂组则完全失效。

    

    经过推理调优和奖励训练的模型通常被认为验证更可靠，因为它们推理时间更长或接收到了正确性信号。我们在证据充分性判断任务中检验了这一假设：即判断现有信息是否真正支持某个因果主张——这是一个不存在可判定正确性奖励的领域。我们采用预注册、同源底座模型的受控敲入设计，在无评估邀请的行动压力协议下，将一种边界保持训练配方（先SFT后GRPO）与其自身基线、一种乱序配对安慰剂组以及一个通用推理模型进行比较。所有承重结局均由两次独立的、对分组标签设盲的模型评分会话进行评定；自动评判器仅用于触发。训练组将自发的正确拒绝率从25%提升至100%（每组40例；Wilson 95%置信区间下界为91.2%，与基线区间[14.2%, 40.2%]不重叠），而乱序安慰剂组则崩溃（自发拒绝率5% / 受邀拒绝率7.5%）

    arXiv:2610.35226v1 Announce Type: new  Abstract: Reasoning-tuned and reward-trained models are often assumed to verify more reliably because they reason longer or receive correctness signals. We test this in evidence-sufficiency judgment: deciding whether available information actually supports a causal claim, a domain with no decidable correctness reward. Using a pre-registered, same-base, controlled knock-in design, we compare a boundary-keeping recipe (SFT then GRPO) with its own baseline, a scrambled-pairing placebo, and a general-purpose reasoning model under an action-pressure protocol with no evaluation invitation. All load-bearing outcomes are rated by two separate model-rating sessions masked to arm labels; automatic judges only trigger. The trained arm raises spontaneous correct refusal from 25% to 100% (40 per arm; Wilson 95% CI lower bound 91.2%, non-overlapping with the baseline interval [14.2%, 40.2%]), while the scrambled placebo collapses (5% spontaneous / 7.5% invited 
    
[^46]: DTCA：面向扩散语言模型强化学习的去噪词元信用分配

    DTCA: Denoising Token Credit Assignment for Reinforcement Learning in Diffusion Language Models

    [https://arxiv.org/abs/2610.35201](https://arxiv.org/abs/2610.35201)

    提出DTCA框架，通过测量词元确定后其余掩码位置预测置信度的提升并按词性类别分组来估计词元贡献，解决了扩散语言模型强化学习中的词元级信用分配难题。

    

    扩散语言模型（DLM）通过迭代去噪的方式生成文本，能够实现双向上下文建模和并行的词元细化。近期研究已将强化学习（RL）应用于扩散语言模型并提升了其推理性能。然而，迭代去噪过程使得信用分配更具挑战性，因为最终的结果奖励仅能提供关于不同词元如何贡献于最终输出的有限信息。本文提出了去噪词元信用分配（DTCA），这是一个面向扩散语言模型强化学习中词元级信用分配的框架。DTCA 通过测量词元确定后其余被掩码位置预测置信度的提升来估计词元的贡献，并根据词性（POS）类别对这些信号进行分组。这些贡献权重被用于选择信息量大的去噪时间步并分配词元级学习信号。

    arXiv:2610.35201v1 Announce Type: new  Abstract: Diffusion language models (DLMs) generate text through iterative denoising, enabling bidirectional context modeling and parallel token refinement. Recent studies have applied reinforcement learning (RL) to DLMs and improved their reasoning performance. However, the iterative denoising process makes credit assignment more challenging, as final outcome rewards provide limited information about how different tokens contribute to the final output. In this paper, we propose Denoising Token Credit Assignment (DTCA), a framework for token-level credit assignment in reinforcement learning for diffusion language models. DTCA estimates token contributions by measuring the improvement in prediction confidence of remaining masked positions after token commitment and groups these signals according to part-of-speech (POS) categories. These contribution weights are used to select informative denoising timesteps and assign token-level learning signals d
    
[^47]: 迷失在翻译中：调查美国中餐厅平行英文名称与中文名称之间的系统性差异

    Lost in Translation: Investigating Systematic Discrepancies between Parallel English and Chinese Names of American Chinese Restaurants

    [https://arxiv.org/abs/2610.35191](https://arxiv.org/abs/2610.35191)

    本文提出8维度名称框架并构建NameFrames数据集，对美国3,172家中餐厅的中英文名称进行系统性对比，揭示了平行名称之间的差异，以探究华人移民在美国的文化融入情况。

    

    鉴于华人是移民美国历史最悠久的族群之一，中餐已成为美国餐饮界的核心组成部分，也为研究中国移民如何在文化上适应美国生活提供了一个有趣的视角。在此背景下，以往的研究主要聚焦于分析中餐厅所采用的英文名称，例如发现其使用了人们熟悉且典型的词汇。为了研究中国移民如何在不同社会、地理和人口环境下实现文化融入，本研究调查了美国中餐厅中文名称与英文名称之间尚未被充分探索的关系。具体而言，我们对在全美采样的3,172家中餐厅（占谷歌地图上总数的10%）开展了研究。我们提出了一个8维度的名称框架来捕捉名称背后的深层含义，并创建了包含884对经过标注的平行中英文名称的NameFrames数据集。

    arXiv:2610.35191v1 Announce Type: new  Abstract: Given that the Chinese are one of the oldest ethnic groups to immigrate to the US, their cuisine have become central to the US dining scene and an interesting avenue to study how Chinese immigrants have culturally adapted into American life. In that regard, previous research has focused on analyzing the English names that Chinese restaurants adopt, e.g., finding the use of familiar and typical words. To investigate how Chinese immigrants have culturally integrated in the US depending on social, geographic and demographic contexts, we investigate in this work the under-explored relationship between the Chinese and English names of American Chinese restaurants. Specifically, we conduct a study on 3,172 Chinese restaurants (10% of the total on Google Maps) sampled across the US. We present 8-dimension name frames to capture the underlying meanings behind names and create the NameFrames dataset containing 884 annotated parallel Chinese/Engli
    
[^48]: 高分背后的隐患：评估基于码本的LLM内容分析

    High Scores, Hidden Failures: Evaluating Codebook-Based LLM Content Analysis

    [https://arxiv.org/abs/2610.35179](https://arxiv.org/abs/2610.35179)

    该研究通过跨四个数据集和五个模型的大规模实证分析，揭示基于LLM的码本内容分析尽管总体得分很高，却隐藏着标签级别差异、运行不稳定性和领域敏感性等被标准指标掩盖的失效模式，且这些失效在常用优化策略下依然存在。

    

    大规模语言模型（LLM）正越来越多地被用于计算社会科学中基于码本（codebook）的内容分析，其中码本将理论概念转化为标注方案。尽管先前的研究报告了强大的预测性能，但这些系统能否产生可靠且稳健的文本标注仍不清楚。我们对基于LLM的编码流程进行了大规模的实证失效模式分析，涵盖四个数据集和五个模型，通过在受控扰动、领域偏移和优化程序下系统地评估其稳定性、稳健性和概念有效性。我们发现了多种在标准指标下基本不可见的失效模式，表明高总体得分往往与严重的标签级别差异、跨运行的不稳定性、领域敏感性以及标签解释的转变共存。重要的是，这些失效模式在广泛使用的优化策略下依然持续存在。

    arXiv:2610.35179v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used for codebook-based content analysis in computational social science, where codebooks translate theoretical concepts into labeling schemes. While prior work reports strong predictive performance, it remains unclear whether these systems produce reliable and robust text annotations. We conduct a large-scale empirical failure-mode analysis of LLM-based coding pipelines across four datasets and five models by systematically evaluating their stability, robustness, and conceptual validity under controlled perturbations, domain shifts, and optimization procedures. We uncover multiple failure modes that are largely invisible under standard metrics, showing that high aggregate scores frequently coexist with severe label-level disparities, instability across runs, domain sensitivity, and shifts in label interpretation. Importantly, these failure modes persist under widely used optimization strateg
    
[^49]: 语言模型理解伊博文化吗？超越多选题

    Do Language Models Understand Igbo Culture? Beyond Multiple Choice

    [https://arxiv.org/abs/2610.35168](https://arxiv.org/abs/2610.35168)

    本文提出了基于伊博本土文化构建并由母语者验证的CAB-Igbo基准测试，通过多选题与人类评估相结合的方式，揭示了语言模型的流畅性与文化知识之间存在严重脱节，且多数模型在英文中就已缺失认知忠实度和表征丰富度。

    

    语言模型对任何社区的价值取决于模型是否真正理解它所意图服务人群的文化。我们以伊博语为案例进行研究。现有的基准测试考察的是翻译过来的西方思想，缺乏深度。为解决这一问题，我们提出了CAB-Igbo，这是一个基于本土文化来源构建、并由母语使用者在六个领域进行验证的基准测试。我们在不同条件下评估了多种模型，并纳入了地区性人格设定。性能通过多选题准确率和人类评估进行衡量，评估维度包括认知忠实度、表征丰富度、语言质量和语用能力。结果揭示了语言流畅性与文化知识之间存在显著的脱节。通过统计检验，我们发现对于大多数模型而言，认知忠实度和表征丰富度的严重缺失在英文中就已经存在，并且在切换到伊博语时并未显著恶化。

    arXiv:2610.35168v1 Announce Type: new  Abstract: The value of language models for any community depends on whether a model actually understands the culture of the people it intends to serve. We use the Igbo language as a case study. Existing benchmarks test translated Western ideas, lacking depth. To address this, we introduce CAB-Igbo, a benchmark developed from indigenous cultural sources and verified by native speakers across six domains. We evaluated diverse models under varying conditions, incorporating regional personas. Performance was assessed using multiple-choice accuracy and human evaluations across Epistemic Fidelity, Representational Richness, Linguistic Quality, and Pragmatic Proficiency. Results reveal a stark dissociation between language fluency and cultural knowledge. Through statistical testing, we show that for most models, the severe lack of Epistemic Fidelity and Representational Richness is already present in English and does not significantly worsen when switchi
    
[^50]: CalibraRL：用于缓解RAG中幻觉的置信度校准强化学习

    CalibraRL: Confidence-Calibrated Reinforcement Learning for Mitigating Hallucinations in RAG

    [https://arxiv.org/abs/2610.35142](https://arxiv.org/abs/2610.35142)

    CalibraRL是一个将回答置信度和查询级回答证据融入奖励设计的强化学习框架，通过对非放弃回答施加Brier惩罚、对放弃回答利用Beta后验估计条件回答可靠性，从而有效缓解RAG中的幻觉问题。

    

    大语言模型（LLMs）经常产生看似合理但不正确的答案，这促使人们研究通过强化学习方法来奖励正确答案、惩罚错误答案，并允许模型放弃回答。然而，固定的基于结果的奖励无法区分具有不同置信度的错误回答，也无法考虑不同查询之间回答可靠性的差异。因此，这类奖励可能不足以惩罚过度自信的错误，也无法阻止不必要的放弃回答行为。我们提出了CalibraRL，这是一个将回答置信度和查询级别的回答证据融入奖励设计的强化学习框架。对于非放弃回答的情况，CalibraRL将基于正确性的效用与基于Brier分数的惩罚相结合，以鼓励置信度与正确性之间的一致性。对于放弃回答的情况，它使用从非放弃回答的正确性推导出的Beta后验来估计条件性回答可靠性。

    arXiv:2610.35142v1 Announce Type: new  Abstract: Large language models (LLMs) often produce plausible but incorrect answers, motivating reinforcement learning methods that reward correct answers, penalize incorrect answers, and allow abstention. However, fixed outcome-based rewards do not distinguish incorrect responses with different confidence levels or account for differences in answering reliability across queries. Consequently, such rewards may inadequately penalize overconfident errors and fail to discourage unnecessary abstention. We propose CalibraRL, a reinforcement learning framework that incorporates response confidence and query-level answering evidence into reward design. For non-abstaining responses, CalibraRL combines correctness-based utility with a Brier-based penalty that encourages agreement between confidence and correctness. For abstentions, it estimates conditional answering reliability using a Beta posterior derived from the correctness of non-abstaining response
    
[^51]: 质量胜于数量：重新思考孟加拉语共指消解的数据与模型假设

    Quality over Quantity: Rethinking Data and Model Assumptions for Bengali Coreference Resolution

    [https://arxiv.org/abs/2610.35114](https://arxiv.org/abs/2610.35114)

    该研究针对孟加拉语共指消解任务，证明少量高质量标注数据显著优于更大的混合质量数据集，而数据增强（如回译）无益甚至有害，跨语言迁移效果也不一致，从而挑战了低资源NLP中关于模型泛化与数据效率的常见假设。

    

    多语言Transformer模型通常被假设能够跨语言泛化，但其在低资源环境下共指消解任务中的有效性尚未得到验证。我们在孟加拉语中研究了这一空白——孟加拉语拥有超过2.5亿使用者，但标注的共指消解资源十分有限。我们在多种设置下评估了基于Transformer的预训练编码器，包括零样本学习、微调、数据增强和跨语言迁移。我们的结果挑战了关于低资源NLP中模型泛化与数据效率的若干常见假设。我们证明，少量高质量标注数据显著优于更大的混合质量数据集（p < 0.01）。我们发现数据增强，尤其是回译，并不能提升性能，反而可能降低共指消解的效果。跨语言迁移同样表现出不一致性：从英语这类语言距离较远的语言进行迁移会降低……（摘要原文在此处截断）

    arXiv:2610.35114v1 Announce Type: new  Abstract: Multilingual transformer models are often assumed to generalise across languages, but their effectiveness for coreference resolution in low-resource settings remains unverified. We investigate this gap in Bengali, a language with over 250 million speakers but limited annotated coreference resources. We evaluate transformer-based pre-trained encoders under multiple settings, including zero-shot, fine-tuning, data augmentation, and cross-lingual transfer. Our results challenge several common assumptions about model generalisation and data efficiency in low-resource NLP. We demonstrate that a small amount of high-quality annotated data significantly outperforms larger mixed-quality datasets (p < 0.01). We find that data augmentation, particularly back-translation, does not improve performance and can degrade results in coreference resolution. Cross-lingual transfer is also inconsistent: transfer from a distant language like English reduces 
    
[^52]: 基于克里普克对“意义即使用”的挑战，重新思考原则性泛化的评估

    Rethinking Evaluation for Principled Generalization Using a Kripkean Challenge to Meaning as Use

    [https://arxiv.org/abs/2610.35111](https://arxiv.org/abs/2610.35111)

    该论文借助克里普克对维特根斯坦的解读，论证有限的使用历史无法唯一确定规则，因此基于固定用例或局部鲁棒性的基准测试不足以证明语义能力，并提出通过评估抽象与递归两种能力来检验模型在原则性领域扩展下保持规则与概念的能力。

    

    大型语言模型将“意义即使用”这一直观观点付诸操作化实践。我们认为这一观点面临着“遵循规则”的挑战。借助克里普克对维特根斯坦的解读，我们论证有限的使用历史并不能为未来的扩展确定唯一的规则。这不仅重新构建了关于分布外（OOD）泛化的争论，也重新构建了关于自然语言处理（NLP）评估的讨论；我们认为，那些衡量模型在固定使用案例上的表现或局部鲁棒性的基准测试，其本身并不能确立模型的语义能力。为了评估模型在原则性的领域扩展下是否能够保持规则和概念，我们提出评估两种对这一能力至关重要的能力：抽象与递归；并提供了一个用于测试这两种能力的实验范式。

    arXiv:2610.35111v1 Announce Type: new  Abstract: Large language models operationalize an intuitive, view of meaning as use. We argue that it faces a rule-following challenge. Using Kripke's interpretation of Wittgenstein, we argue that finite histories of use do not determine a unique rule for future extension. This reframes the debate about OOD generalization but also about NLP evaluation; benchmarks that measure performance on fixed use cases or local robustness, we argue, do not by themselves establish semantic competence. To evaluate whether models preserve rules and concepts under principled domain extension, we propose to evaluate two capacities, central to this ability: abstraction and recursion; and we offer an experimental paradigm for testing both.
    
[^53]: 有用的多样性还是幻觉的多样性？面向电商搜索的人格条件化合成查询生成

    Useful Diversity or Hallucinated Variety? Persona-Conditioned Synthetic Query Generation for E-Commerce Search

    [https://arxiv.org/abs/2610.35102](https://arxiv.org/abs/2610.35102)

    该研究通过人格条件化提示生成大规模合成查询来冷启动电商语义解析器，发现IPIP人格捆绑包可显著提升解析器F1、语法条件化能引导生成结果更贴近真实流量，并提出对齐sKL作为解析器性能的无标注代理指标。

    

    合成监督可以为新的搜索领域引导语义解析器的冷启动，但提示诱导的多样性只有在生成内容忠实于结构化意图时才有用。我们在冷启动电商搜索场景中对这一问题进行了探索：使用6个大语言模型配合9个人格条件化提示词家族，在47字段的英国汽车模式上生成了118,098个合成查询对，并与1,214条人工标注查询和15,304条生产查询进行对比。我们表明，IPIP人格捆绑包在所有四组匹配对比中都提升了语义解析器的F1（经Holm校正的p<.001），并且在完整的2³网格上，所有三个捆绑包都具有正向主效应。语法条件化是将生成结果推向真实流量分布的关键维度，同时也能诱导词汇多样性。对齐sKL——即分别拟合到合成查询与人工标注查询上的von Mises-Fisher分布之间的散度——是解析器F1的一种无标注代理指标（与为模拟购物者场景编写的人工查询相比，ρ=-0.917……

    arXiv:2610.35102v1 Announce Type: new  Abstract: Synthetic supervision can bootstrap semantic parsers in new search domains, but prompt-induced diversity is useful only when generations are faithful to the structured intent. We explore this for cold-start e-commerce search: 118,098 synthetic pairs produced by 6 LLMs with 9 persona-conditioned prompt families on a 47-field UK automotive schema, against 1,214 human-annotated and 15,304 production queries. We show that the IPIP personality bundle improves parser F1 in all four matched contrasts (Holm-corrected $p<.001$), and all three bundles have a positive main effect over the complete $2^3$ grid. Syntax conditioning is the axis that moves generations towards real traffic. It also induces lexical diversity. Alignment sKL, the divergence between von Mises-Fisher distributions fitted to synthetic and human-annotated queries, is an unlabelled proxy for parser F1 ($\rho=-0.917$ against human queries written for simulated shopper scenarios, 
    
[^54]: 以RAG对抗厌女：改进隐性厌女言论的解释

    RAGing Against Misogyny: Improving Implicit Misogyny Explanations

    [https://arxiv.org/abs/2610.35094](https://arxiv.org/abs/2610.35094)

    本研究提出利用检索增强生成（RAG）框架，结合百科、俚语和女性主义知识来源，改进英语和意大利语中隐性厌女言论的检测与解释，其中女性主义文献被证明是最持续有用的知识来源。

    

    隐性厌女言论通常需要背景知识才能被理解。为了应对这一挑战，我们提出了一个检索增强生成（RAG）框架，用于检测和解释英语和意大利语中的隐性厌女言论，该框架利用了百科知识、俚语知识和女性主义知识来源。基于ImplicIT-Mis和SBIC+数据集，我们评估了每种知识来源对检测和解释质量的贡献，并研究了RAG是否能解决大语言模型在处理隐性厌女言论时的已知失败模式。结果表明，RAG提高了在ImplicIT-Mis上的召回率，但在SBIC+上并未提升检测性能。然而，人类评估显示，在两种语言中，基于RAG生成的解释都比不使用RAG的解释更受青睐。女性主义文献被证明是对检测和解释最持续有用的知识来源。

    arXiv:2610.35094v1 Announce Type: new  Abstract: Implicit misogyny often requires background knowledge to be understood. To address this challenge, we propose a Retrieval-Augmented Generation (RAG) framework for detecting and explaining implicit misogyny in English and Italian, leveraging encyclopedic, slang and feminist knowledge sources. Using the ImplicIT-Mis and SBIC+ datasets, we evaluate the contribution of each knowledge source to detection and explanation quality, investigating also whether RAG addresses known LLM failure modes in implicit misogyny. Results show that RAG improves recall on ImplicIT-Mis, but does not improve detection performance on SBIC+. However, human evaluation shows that RAG-based explanations are preferred over no-RAG explanations in both languages. Feminist literature emerges as the most consistently useful source for detection and explanation.
    
[^55]: 先问再答：面向上下文学习的问题驱动证据推理架构（QERA）

    Ask Before You Answer: A Question-Driven Evidence Reasoning Architecture (QERA) for In-Context Learning

    [https://arxiv.org/abs/2610.35083](https://arxiv.org/abs/2610.35083)

    该论文提出QERA架构，通过先将用户请求转化为需求清单并筛选出隐含信息需求，再从完整对话中作答并用代码验证引文，从而解决大语言模型在真实生活上下文学习任务中因请求隐含大量未明说标准而表现不佳的问题。

    

    AI助手越来越多地被要求基于用户自己的网络痕迹进行推理，包括群聊、社区帖子、活动日志和修订历史。在CL-bench Life这一真实生活上下文学习基准上，前沿大语言模型解决的任务不足五分之一。我们发现请求本身并非任务规范：该基准中超过80%的评分标准与最终请求的内容词重合度低于60%，例如两条对话之前的规则集，或所有满足用户某项标准的记录，而直接作答的大语言模型在这些隐含标准上仅通过67%，相比之下在其他标准上通过率为81%。我们提出QERA，一种先问再答的问题驱动证据推理架构。QERA首先将请求转化为需求清单和类型化的信息需求，并通过代码进行过滤以确保忠实于用户的原话；然后基于整个对话进行回答，使用经代码验证的引文再次提出相同的问题，并应用……（摘要内容在此处截断）

    arXiv:2610.35083v1 Announce Type: new  Abstract: AI assistants are increasingly asked to reason over a user's own web traces: group chats, community threads, activity logs and revision histories. On CL-bench Life, a benchmark of such real-life context learning, frontier LLMs solve fewer than one task in five. We find that the request is not the specification: over 80% of the benchmark's rubrics share less than 60% of their content words with the final request, as for a rule set two turns earlier or every record that meets the user's criterion, and LLMs answering directly pass only 67% of these implicit rubrics against 81% of the others. We propose QERA, a Question-driven Evidence Reasoning Architecture that asks before it answers. QERA first turns the request into a requirement checklist and typed information needs, filtered by code for faithfulness to the user's words; it then answers from the whole conversation, asks the same questions again with quotes verified by code, and applies 
    
[^56]: CoSA：真实世界LLM智能体生态系统中的技能冲突攻击

    CoSA: The Conflict of Skills Attack in Real-World LLM Agent Ecosystems

    [https://arxiv.org/abs/2610.35082](https://arxiv.org/abs/2610.35082)

    本文提出“技能冲突”（CoS）概念及CoSA攻击方法，证明仅通过向LLM智能体技能池添加本地准入的新技能（进行能力补充或关系重塑），就能在技能组合使用时引发策略违规，且无需改动现有技能或运行时权限。

    

    智能体技能使LLM智能体能够组合可复用的能力，但仅对每个技能进行孤立检查，无法排除它们在组合使用时出现的策略违规。我们将此类失败称为技能冲突：为某一目的而产生的人工制品或评估信号可能在下游被复用，却没有保留其新用途所需的限制或授权。通过对公开可用技能的受控执行以及匹配的单技能对照实验，我们识别出CoS背后四种反复出现的机制。基于这些发现，我们提出了技能冲突攻击，它通过向现有技能池中添加本地准入的技能，来提供缺失的业务操作（能力补充），或重塑人工制品交换、决策复用以及目标/行动绑定（关系重塑）。CoSA不会改变现有的技能和运行时权限。为了在自主技能选择条件下评估这一威胁，我们引入了CoSA-

    arXiv:2610.35082v1 Announce Type: cross  Abstract: Agent Skills allow LLM agents to combine reusable capabilities, but checking each Skill in isolation cannot rule out policy violations when they are composed. We term such failures Conflict of Skills (CoS): artifacts or assessment signals produced for one purpose may be reused downstream without preserving the restrictions or authorization needed for their new use. Using controlled executions of publicly available Skills and matched single-Skill controls, we identify four recurring mechanisms behind CoS. Building on these findings, we introduce the Conflict of Skills Attack (CoSA), which adds locally admitted Skills to an existing pool to supply missing business operations (capability supplementation) or reshape artifact exchange, decision reuse, and target/action bindings (relationship reshaping). CoSA leaves existing Skills and runtime permissions unchanged. To evaluate this threat under autonomous Skill selection, we introduce CoSA-
    
[^57]: 通过自主进化提升视觉语言模型在科学图表分析上的能力

    Improving VLMs on Scientific Chart Analysis via Autonomous Evolution

    [https://arxiv.org/abs/2610.35077](https://arxiv.org/abs/2610.35077)

    本文提出利用自主进化方法在冻结模型的推理程序代码空间中进行搜索，在不改变模型本身的前提下显著恢复视觉语言模型在科学图表问答任务上的准确率。

    

    理解和操作科学图表是数据分析中的标准方法，虽然人类擅长解读这类图表，但众所周知，开源视觉语言模型（VLM）在基于此类数据类型进行问答时表现不佳。基于一个初步观察——模型的许多错误源于回答不精确——我们提出了一个自然的问题：在不改变模型本身的情况下，能够恢复多少准确率？在本文中，我们通过在围绕模型“冻结”版本的推理程序及其代码上进行搜索来研究这个问题。借鉴自主进化的思想（该方法因在AlphaEvolve中的著名应用而闻名），我们首先从一个基线推理程序出发，在每一轮中让一个LLM评判者查看代码以及若干训练问题上的失败案例，并要求其提出改进性的进化步骤。与通常的自主进化方法一样，候选程序被保存在一个小型……（原文摘要在此处截断）

    arXiv:2610.35077v1 Announce Type: cross  Abstract: Understanding and operating with scientific charts is a standard approach in data analysis, and while humans are good at interpreting such charts, open vision-language models (VLMs) are known to perform poorly when it comes to question answering based on such data types. Motivated by an initial observation that many of their errors are due to inexact answering, we raise the following natural question: how much accuracy can we recover without changing the model at all? In this paper, we study this question by searching over inference programs and the code around a "frozen" version of the model. Using ideas from autonomous evolution (famously used in AlphaEvolve), we first start from a baseline inference program, and in each round we show an LLM judge the code together with a few failures on training questions, and ask for an improving evolutionary move. As usual in autonomous evolution approaches, the candidate programs are kept in a sm
    
[^58]: 正确实现显式置信度目标的强化学习

    Reinforcement Learning Done Right for Explicit Confidence Targets

    [https://arxiv.org/abs/2610.35059](https://arxiv.org/abs/2610.35059)

    该论文研究了让大语言模型根据用户指定的显式置信度目标灵活决定回答或弃答的问题，指出现有强化学习方法失败的原因在于区分不同目标 p 值的奖励信号在优化中丢失等问题，并提出正确的强化学习方法加以解决。

    

    大语言模型（LLM）在不确定时可能会给出看似合理但实际错误的答案，这会误导用户并损害模型的可靠性。通过强化学习训练模型学会弃答（不回答）是一种自然的方法，但现有大多数方法采用固定的错误惩罚，因此无法在推理时根据不同的置信度要求调整自身行为。我们研究“显式置信度目标”这一设定：用户在提示中指定一个目标 p，模型直接选择回答还是弃答，目标是在已回答的问题上达到至少 p 的准确率。然而，让单一模型学会响应不同的置信度目标已被证明十分困难，先前的尝试未能区分不同的目标。我们证明这种失败源于两个问题：用于区分不同 p 值的奖励信号在优化过程中经常丢失，而……

    arXiv:2610.35059v1 Announce Type: cross  Abstract: Large language models (LLMs) can produce plausible but incorrect answers when they are uncertain, which can mislead users and undermine model reliability. Training models to abstain with reinforcement learning is a natural approach, but most existing methods use a fixed error penalty and therefore cannot adapt their behavior to different confidence requirements at inference time. We study \emph{explicit confidence targets}, where a user specifies a target \(p\) in the prompt and the model directly chooses whether to answer or abstain, with the goal of achieving an accuracy of at least \(p\) among answered questions. However, learning to respond to different confidence targets with a single model has proven difficult, and prior attempts have failed to distinguish different targets. We show that this failure comes from two problems: the reward signal that distinguishes different values of $p$ is often lost during optimization, while the 
    
[^59]: 从问题到查询再到洞察：一种基于数据库模式的自然语言交通事故数据分析智能体工作流

    From Questions to Queries to Insights: A Schema-Grounded Agentic Workflow for Natural-Language Crash Data Analysis

    [https://arxiv.org/abs/2610.35054](https://arxiv.org/abs/2610.35054)

    本文提出 CrashDB-QA，一个完全本地化、以数据模式和安全分析惯例为基准的智能体工作流，使交通从业者能够通过自然语言对交通事故数据库进行可靠、可审计的自动查询与分析。

    

    交通运输机构维护着庞大的交通事故数据库，用于识别安全优先事项、评估对策措施并指导资源配置。要让从业者能够便捷使用这些数据，需要灵活的方式将分析问题转化为可靠的结果。现有的仪表板仅支持预定义的分析，而额外的分析请求往往依赖于精通数据库查询、编码字段和机构惯例的专业人员。本研究提出了 CrashDB-QA，这是一个完全本地化部署、基于数据库模式的智能体工作流，可实现对交通事故数据库的自然语言分析。专门的组件负责解析问题、生成并验证只读 SQL、执行查询并解释结果。共享知识库将查询构建锚定在机构的数据字典和安全分析惯例之上，同时记录的中间输出支持结果核查与审计。研究还提出了一个可选的扩展版本 CrashDB-QA+，用于检查对齐性……（原文摘要在此处截断）

    arXiv:2610.35054v1 Announce Type: new  Abstract: Transportation agencies maintain extensive crash databases to identify safety priorities, evaluate countermeasures, and guide resource allocation. Making these data accessible to practitioners requires flexible ways to translate analytical questions into reliable results. Existing dashboards support predefined analyses, while additional requests often depend on specialists who understand database queries, coded fields, and agency conventions. This study presents CrashDB-QA, a fully local, schema-grounded agentic workflow that enables natural-language analysis of crash databases. Specialized components interpret questions, generate and validate read-only SQL, execute queries, and explain the results. A shared knowledge base grounds query construction in the agency's data dictionary and safety-analysis conventions, while recorded intermediate outputs support inspection and auditing. An optional extension, CrashDB-QA+, checks the alignment 
    
[^60]: SimPOTrans：基于强化学习的离线语音翻译到同声语音翻译的自适应方法

    SimPOTrans: Self-Adaptation for Offline-to-Simultaneous Speech Translation via Reinforcement Learning

    [https://arxiv.org/abs/2610.35042](https://arxiv.org/abs/2610.35042)

    SimPOTrans提出了一种简单的两阶段强化学习后训练框架，使离线语音翻译模型无需依赖合成轨迹或多目标联合优化即可自适应地获得低延迟的同声语音翻译能力。

    

    同声语音翻译（SimulST）旨在以低延迟翻译持续进行的语音，同时保持翻译质量。近期研究采用大语言模型（LLM）进行SimulST，并将其建模为多轮对话任务以实现高效生成，达到了最先进的性能。然而，现有的对话式SimulST方法通常依赖专门的训练方案来习得同声翻译行为：要么使用合成的读/写轨迹，这可能提供次优的训练信号；要么联合优化翻译质量与延迟，而这种相互竞争的目标可能导致优化不稳定。本文提出了SimPOTrans，这是一个强化学习（RL）框架，能够使离线语音翻译模型自适应地转变为对话式SimulST。通过一个简单的两阶段强化学习后训练流程，SimPOTrans对离线模型表现出同声翻译行为给予奖励……（摘要此处截断）

    arXiv:2610.35042v1 Announce Type: new  Abstract: Simultaneous speech translation (SimulST) aims to translate ongoing speech with low latency while preserving translation quality. Recent studies have adopted large language models (LLMs) for SimulST and formulated it as a multi-turn dialogue task for efficient generation, achieving state-of-the-art performance. However, existing dialogue-style SimulST approaches typically rely on specialized training schemes to acquire simultaneous translation behavior, either using synthesized read/write trajectories, which may provide suboptimal training signals, or jointly optimizing translation quality and latency, whose competing objectives may cause optimization instability. In this paper, we introduce SimPOTrans, a reinforcement learning (RL) framework that enables offline speech translation models to self-adapt to dialogue-style SimulST. Through a simple two-stage RL post-training pipeline, SimPOTrans rewards an offline model for exhibiting trans
    
[^61]: 合成的儿童-成人互动对话不符合发展规律

    Synthetic Child-Adult Interactions Are Not Developmentally Plausible

    [https://arxiv.org/abs/2610.35029](https://arxiv.org/abs/2610.35029)

    该论文发现合成的儿童-成人对话数据虽在词汇层面接近自然对话，却严重低估了口语互动的混乱与碎片化特性，并提出用真实对话数据进行指令微调，使小模型经过适度参数高效微调后即可逼近自然数据的统计特征。

    

    合成数据已成为训练语言模型的常用资源，这一趋势也蔓延到了发展合理性语言建模的研究领域。然而，语言模型生成的儿童-成人互动对话是否真的与自然对话相似，目前仍不清楚。我们在广泛的语言学指标上，将合成的 TinyDialogues 数据集与自然对话进行了比较。虽然合成数据在词汇层面与自然数据相匹配，但它极大地低估了口语互动混乱、碎片化的特性。为了缓解这一问题，我们提出使用由真实对话创建的数据进行指令微调，并在一项初步研究中表明，一个小型模型在经过适度的参数高效微调后，能够接近自然数据的统计特征。

    arXiv:2610.35029v1 Announce Type: new  Abstract: Synthetic data has become a common resource for training language models, and this trend has spilled over into research on developmentally plausible language modeling. However, it remains unclear whether LM-generated child-adult interactions actually resemble natural ones. We compare the synthetic TinyDialogues dataset to natural dialogue across a broad range of linguistic measures. While synthetic data matches natural data on the lexical level, it drastically underestimates the messy, fragmented nature of spoken interaction. To mitigate this, we propose instruction-tuning with data created from real dialogues, and show in a pilot study that a small model approximates the statistical tendencies of natural data after moderate parameter-efficient fine-tuning.
    
[^62]: 当人设与台词相矛盾时：诊断思维链语音合成中从规划到合成的失效问题

    When the Persona Contradicts the Line: Diagnosing Plan-to-Synthesis Failures in Chain-of-Thought TTS

    [https://arxiv.org/abs/2610.35008](https://arxiv.org/abs/2610.35008)

    该论文提出了一个日英双语人设-台词冲突基准，系统定位了思维链TTS中从规划到合成各环节的意图丢失问题，并证明监督情感分类器或采用学习式文本-潜变量接口可大幅提升声学情感方向的一致性。

    

    思维链（Chain-of-Thought）语音合成（TTS）系统依赖语言模型规划器将角色人设与上下文转化为声音表达方式，但当所要求的情感与词汇本身的情感倾向发生冲突时，这一意图在何处丢失尚不清楚。我们构建了一个日语-英语双语的“人设-台词”冲突基准，其中每个条目将一句台词分别与一致和冲突的人设配对，并预先声明预期的声学方向。我们在规划器生成的表达短语、规划器写入的情感向量、QwenEmotion生成的情感向量以及最终渲染的波形中探测这一声学方向。结果表明：规划器的输出通常是方向正确的，而下游环节与预期方向的一致性却可能显著降低；绕过QwenEmotion相比完全由规划器控制的基础版仅有有限的提升；对QwenEmotion进行监督训练可将日语的方向正确率提升至75%、英语提升至84%；而学习式的文本到潜变量接口则达到了英语最高的89%以及最高的……

    arXiv:2610.35008v1 Announce Type: new  Abstract: Chain-of-thought TTS systems rely on language-model planners to translate persona and context into vocal delivery, but it is unclear where this intent is lost when the requested affect conflicts with lexical sentiment. We introduce a bilingual Japanese--English persona--line conflict benchmark in which each item pairs one line with congruent and conflicting personas and pre-declares the expected acoustic direction. We probe this direction in the planner-generated delivery phrase, the planner-written emotion vector, the QwenEmotion-generated vector, and the rendered waveform. Planner outputs are usually direction-correct, while downstream agreement can be substantially lower. Bypassing QwenEmotion provides only modest gains over the full planner-controlled Base. Supervising QwenEmotion raises the direction rate to 75% in Japanese and 84% in English. A learned text-to-latent interface reaches the highest English rate of 89% and the highest
    
[^63]: 推理还是背诵：基于LLM裁判的互补专家模型用于交通异常推理

    Reason or Recite: Complementary Specialists with an LLM Judge for Traffic Anomaly Reasoning

    [https://arxiv.org/abs/2610.34993](https://arxiv.org/abs/2610.34993)

    该论文提出RoR双专家框架，用一个9B模型学习基于结构化时间推理的能力、一个27B模型学习标注答案风格，再由LLM裁判在两者间择优，从而在交通异常推理任务中同时兼顾精确的证据推理与开放式回答的评分要求。

    

    诸如碰撞、违规行为和突发危险等道路异常事件对视频理解的要求远不止于检测：一个实用的系统必须验证发生了什么、描述现场情况并解释其原因。我们发现这些能力对视觉-语言模型而言是相互冲突的：封闭式问题奖励精确的、基于证据的推理，而开放式答案则通过与参考文本的相似度来评分，此时匹配标注风格与正确推理同等重要。我们提出了Reason-or-Recite（RoR），这是一个面向2026年AI City Challenge交通异常推理赛道的双专家框架。RoR（1）将官方的思维链标注重写为结构化的、基于时间轴的模式，并利用增强后的推理轨迹对9B视觉-语言模型进行LoRA微调；（2）仅在答案数据上微调一个4-bit量化的27B模型，使其吸收标注风格；（3）通过LLM裁判针对每个问题在两个候选答案中进行选择……

    arXiv:2610.34993v1 Announce Type: cross  Abstract: Road anomalies such as collisions, violations, and sudden hazards demand more from video understanding than detection: a practical system must verify what happened, describe the scene, and explain the cause. We find these competences pull in opposite directions for vision-language models: closed-form questions reward precise, evidence-grounded reasoning, while open-ended answers are scored by similarity to reference text, where matching the annotation style matters as much as reasoning correctly. We present Reason-or-Recite (RoR), a two-specialist framework for the Traffic Anomaly Reasoning track of the 2026 AI City Challenge. RoR (1) rewrites the official chain-of-thought annotations into a structured, temporally grounded schema and LoRA-tunes a 9B vision-language model on the enhanced traces; (2) fine-tunes a 4-bit-quantized 27B model on answers alone, absorbing the annotation style; and (3) selects between the two candidates per que
    
[^64]: 诊断与缓解全双工语音模型中的长会话性能退化

    Diagnosing and Mitigating Long-Session Degradation in Full-Duplex Speech Models

    [https://arxiv.org/abs/2610.34991](https://arxiv.org/abs/2610.34991)

    本文提出一个闭环评估框架系统性诊断全双工语音模型在长会话中的性能退化，并提出基于自蒸馏的轻量级LoRA适配训练方法来有效缓解该退化。

    

    端到端全双工语音对话模型能够实现自然的实时语音交互，支持打断、附和反馈以及语音重叠。先前的评估工作已经考察了多样化的交互能力，并识别出一些跨对话轮次的性能退化模式。然而，模型行为在长会话中如何演变仍缺乏充分的理解。在本工作中，我们通过系统性地刻画长会话性能退化并开发高效的缓解方法来填补这一空白。我们引入了一个面向长会话稳定性的闭环评估框架，并将其应用于多个公开可用的全双工语音模型，揭示了模型在响应性、语音内容质量和声音一致性方面随时间的退化。为了缓解这种退化，我们提出了一种基于自蒸馏的训练框架，用于轻量级LoRA适配，仅需少量额外的数据即可提升长会话稳定性。

    arXiv:2610.34991v1 Announce Type: new  Abstract: End-to-end full-duplex spoken dialogue models enable natural, real-time spoken interactions with interruptions, backchannels, and overlapping speech. Prior evaluations have assessed diverse interaction capabilities and identified some degradation patterns across conversational turns. However, how model behavior evolves over extended sessions remains insufficiently understood. In this work, we address this gap by systematically characterizing long-session degradation and developing efficient methods to mitigate it. We introduce a closed-loop evaluation framework for long-session stability and apply it to several publicly available full-duplex speech models, revealing degradation over time in responsiveness, spoken content quality, and voice consistency. To mitigate this degradation, we propose a self-distillation-based training framework for lightweight LoRA adaptation that improves long-session stability with a small amount of additional
    
[^65]: AskGLP1：一个用于从在线患者叙述中探索GLP-1用药经验的知识图谱LLM智能体

    AskGLP1: A Knowledge Graph LLM Agent for Exploring GLP-1 Experience from Online Patient Narratives

    [https://arxiv.org/abs/2610.34987](https://arxiv.org/abs/2610.34987)

    提出AskGLP1系统，利用LLM辅助流程将近850万条在线患者叙述构建成基于MedDRA术语的知识图谱，揭示GLP-1用药经验具有来源和药物特异性，且在线叙述更关注疲劳等日常负担，而FAERS报告更多涉及注射相关事件

    

    GLP-1药物越来越多地被用于体重管理，许多人在网上分享他们的用药体验。这些叙述展示了人们为何服用这些药物以及他们的感受，但这些内容仍是非结构化的，且与正式的药物警戒体系相脱节。因此，我们提出了AskGLP1，一个将在线患者叙述转化为知识图谱（KG）并回答用药经验相关问题的系统。通过一个基于MedDRA术语的LLM辅助流程，AskGLP1从近850万条Reddit帖子中筛选用药经验，提炼出163,135条经验，并与2,506条WebMD评论一起，构建了一个包含385,790条药物-副作用和药物-用途关系的知识图谱。我们发现GLP-1用药经验具有来源特异性和药物特异性。在线叙述与FAERS共享最常见的胃肠道症状，但在线叙述更突出诸如疲劳和食物厌恶等日常负担，而FAERS报告则更多涉及注射相关事件

    arXiv:2610.34987v1 Announce Type: cross  Abstract: GLP-1 medications are increasingly used for weight management, and many people share their experiences online. These narratives show why people take the drugs and what they experience, but they remain unstructured and disconnected from formal pharmacovigilance. We therefore present AskGLP1, a system that transforms online patient narratives into a knowledge graph (KG) and answers questions about medication experiences. Through an LLM-assisted pipeline grounded in the MedDRA terminology, AskGLP1 screens nearly 8.5 million Reddit posts for medication experiences and distills 163,135 of them, with 2,506 WebMD reviews, into a KG of 385,790 drug-side-effect and drug-purpose relations. We find that GLP-1 experiences are source- and drug-specific. Online narratives share the most common gastrointestinal symptoms with FAERS, but they foreground everyday burdens such as fatigue and food aversion, whereas FAERS reports more often include injecti
    
[^66]: SVD-注意力：高效分析Transformer中的注意力模块

    SVD-Attention: Efficiently Analyzing Attention Blocks in Transformers

    [https://arxiv.org/abs/2610.34971](https://arxiv.org/abs/2610.34971)

    该论文提出SVD-Attention——一种无需微调的事后诊断方法，通过谱幅值与激活方差对每个注意力头的奇异模式评分，精确识别哪些头真正用满秩预算，并揭示了注意力冗余随任务类型和层深分布的规律。

    

    预训练Transformer在自注意力上花费了大量推理计算，而每个注意力头的查询-键交互在结构上天生是低秩的。我们提出一个诊断性问题：在不同层和不同架构中，哪些头在真实数据上真正用满了秩预算，哪些没有？我们提出SVD-Attention，这是一种事后（post-hoc）方法，它对每个注意力头的交互矩阵 $M_h$ 进行分解，并根据每个奇异模式的谱幅值以及其在小校准集上跨token的激活方差来为模式评分。该评分保留头能够放大且真实数据确实会激活的模式，并丢弃其余模式；所得分解对于保留的模式是精确的，且无需微调。在七个预训练模型和五种任务类型上，该诊断发现了任务相关的模式：对于情感分类、图像分类和生成任务，冗余集中在早期到中间层……

    arXiv:2610.34971v1 Announce Type: cross  Abstract: Pretrained transformers spend a large share of their inference compute inside self-attention, yet the per-head query-key interaction is structurally low-rank by construction. We ask a diagnostic question: across layers and architectures, which heads actually use the full rank budget on real data, and which do not? We propose SVD-Attention, a post-hoc method that decomposes the per-head interaction matrix $M_h$ and scores every singular mode by both its spectral magnitude and its activation variance across tokens on a small calibration set. The score keeps the modes the head can amplify and that real data actually activates, and discards the rest; the resulting factorization is exact for the retained modes and needs no finetuning. Across seven pretrained models and five task types, the diagnostic finds a task-dependent pattern: redundancy sits in the early to middle layers for sentiment and image classification and for generation, and i
    
[^67]: 利用展开一致性改进基于分组的策略优化中的推理能力

    Improving Reasoning with Rollout Agreement for Group-Based Policy Optimization

    [https://arxiv.org/abs/2610.34951](https://arxiv.org/abs/2610.34951)

    该研究发现GRPO训练中正确答案展开间的低一致性往往伴随更多虚假推理，并提出RoAR方法，在奖励中加入推理级别的一致性信号，以提升大模型推理的可靠性。

    

    近期的可验证奖励强化学习方法利用展开多样性来提升推理性能或训练效率。尽管取得了这些进展，正确答案展开之间的差异究竟反映了有用的替代推理路径，还是尚未解决的错误，仍然缺乏充分的理解。我们通过对一个使用分组相对策略优化训练的模型进行盲测配对分析来研究这一问题，比较同一问题中一致性最高与最低的正确答案推理轨迹。在126对样本中，低一致性轨迹比高一致性轨迹包含更多的虚假推理（35% 对 22%），包括内部自相矛盾和对题目条件的误用。基于这些发现，我们提出了RoAR（展开一致性奖励），它在GRPO对已验证正确展开的奖励基础上加入了推理级别的一致性信号。RoAR从现有的同策略展开组中计算该信号，无需……

    arXiv:2610.34951v1 Announce Type: cross  Abstract: Recent methods for Reinforcement Learning with Verifiable Rewards (RLVR) use rollout diversity to improve reasoning performance or training efficiency. Despite these gains, the extent to which differences among correct-answer rollouts reflect useful alternative reasoning or unresolved errors remains poorly understood. We investigate this question through a blind paired analysis of a model trained with Group Relative Policy Optimization (GRPO), comparing the highest- and lowest-agreement correct-answer traces for the same problem. Across 126 pairs, low-agreement traces have more spurious reasoning than high-agreement traces (35% vs. 22%), including internal inconsistencies and misuse of problem conditions. Based on these findings, we propose RoAR (Rollout-Agreement Reward), which adds reasoning-level agreement to GRPO's rewards for verified-correct rollouts. RoAR computes this signal from existing on-policy rollout groups, requiring no 
    
[^68]: 面向长链推理的相关性感知探索

    Correlation-Aware Exploration for Long-Trace Reasoning

    [https://arxiv.org/abs/2610.34949](https://arxiv.org/abs/2610.34949)

    提出Info-GRPO信息论框架，通过培养策略与潜在先验之间的互信息相关性来解决GRPO长链推理训练中的策略崩溃问题，避免了熵正则化与收敛所需熵衰减之间的冲突。

    

    使用组相对策略优化（GRPO）训练的推理模型如今能生成数千词元长度的思维链，而近期研究表明这类训练容易发生策略崩溃。熵正则化是标准的应对方法，但它通过将概率质量从少数承载概率的词元推向词汇表中庞大的低概率尾部来发挥作用，并且对推理轨迹中的每个词元都施加一次这种力。这使得该奖励项与收敛所需的熵衰减直接冲突，并使其系数在长轨迹场景下难以设定。为解决这一矛盾，我们提出Info-GRPO，一个旨在培养策略与潜在先验之间相关性的信息论框架。Info-GRPO并非处处提升熵，而是建立在潜在变量与其诱导轨迹之间的互信息之上：在训练过程中用潜在种子增强提示词……

    arXiv:2610.34949v1 Announce Type: cross  Abstract: Reasoning models trained with Group Relative Policy Optimization (GRPO) now produce chains of thought thousands of tokens long, and recent studies reveal that such training is prone to policy collapse. Entropy regularization is the standard remedy, but it works by pushing probability mass away from the few tokens that carry it and toward the vast low-probability tail of the vocabulary, and it applies this force once for every token of the reasoning trace. This puts the bonus in direct conflict with the entropy decay that convergence requires, and makes its coefficient difficult to set in the long-trace regime. To resolve this tension, we propose Info-GRPO, an information-theoretic framework that cultivates correlation between the policy and a latent prior. Rather than raising entropy everywhere, Info-GRPO builds on the mutual information between a latent variable and the trajectory it induces: augmenting prompts with latent seeds durin
    
[^69]: 结果导向强化学习买到了什么：完成度、可靠性、以及很少的触达

    What Outcome RL Buys: Finishing, Reliability, Rarely Reach

    [https://arxiv.org/abs/2610.34894](https://arxiv.org/abs/2610.34894)

    该论文将结果导向强化学习带来的pass@1提升分解为完成度、可靠性和触达三部分，发现在固定token预算下RL的大部分收益其实来自“完成度”（即在预算内给出答案），而真正解决基础模型无法解决的新问题的“触达”很少。

    

    结果导向强化学习提升了pass@1，却无法在大k值下提升pass@k（Yue et al., 2025）。但每个pass@1都是在token预算限制下评分的，每个训练配方也都是在token预算下训练的。在固定token预算的物理和数学问题上，我们比较了20个已发布的RL模型（1.5B到32B）与其出发的思维模型的pass@1。我们探究每个pass@1的提升究竟来自行为的改变还是新知识。我们将pass@1的每个变化分解为三个部分：完成度衡量在预算耗尽前给出答案；可靠性衡量在完成作答后答对；触达衡量RL解决了基础模型在八次尝试中全部答错的问题数，减去基础模型解决了而RL在八次尝试中全部答错的问题数。我们发现了四件事：（1）在基础模型经常在给出答案前就用尽token预算的情况下，大部分变化——无论增益还是损失——都是完成度（在4k和8k下占82%）；给定更多token，基础模型自己就能弥补大部分完成度差距。（2）在基础模型有足够……

    arXiv:2610.34894v1 Announce Type: new  Abstract: Outcome RL raises pass@1 but not pass@k at large k (Yue et al., 2025). But every pass@1 is scored under a token budget. Every recipe also trains under one. On physics and math problems at fixed token budgets, we compare the pass@1 of 20 released RL models (1.5B to 32B) with that of the thinking models they started from. We ask whether each pass@1 gain is changed behaviour or new knowledge. We divide each change in pass@1 into three parts. Finishing measures answering before the budget runs out. Reliability measures being right once finished. Reach measures problems RL solves that its base answered wrongly in all eight tries, minus problems the base solves that RL answered wrongly in all eight. We find four things. (1) Where the base often hits the token budget before answering, most of the change, gain or loss, is finishing (82% at 4k and 8k). Given more tokens, the base closes most of the finishing gap itself. (2) Where the base has eno
    
[^70]: FATE-5：面向连续大五人格控制的特质表达忠实对齐

    FATE-5: Faithful Alignment of Trait Expression for Continuous Big Five Personality Control

    [https://arxiv.org/abs/2610.34889](https://arxiv.org/abs/2610.34889)

    FATE-5提出两阶段框架，将大五人格编码与自然度对齐分离，实现了对大语言模型人格的连续控制，在保持强特质表达的同时显著提升了对话自然度。

    

    在大型语言模型中控制人格，对于构建能够根据不同用户和交互场景进行定制化的对话智能体而言十分重要。然而，现有人格控制方法通常面临强特质表达与自然对话生成之间的权衡。我们提出了FATE-5（大五人格特质表达忠实对齐），这是一个将人格编码与自然度对齐相分离的两阶段框架。FATE-5首先利用对话监督和辅助问卷目标，学习大五人格画像的连续软提示表示；随后通过基于对话质量和人格一致性的偏好对齐来提升对话自然度。在三个语言模型骨干网络上的实验表明，FATE-5在保持强特质表达的同时，实现了比其他强特质方法显著更高的对话自然度。在保留问卷测试中，它实现了单调性……

    arXiv:2610.34889v1 Announce Type: new  Abstract: Controlling personality in large language models is important for building conversational agents that can be customized to different users and interaction settings. However, existing personality control methods often face a tradeoff between strong trait expression and natural dialogue generation. We introduce FATE-5, Faithful Alignment of Trait Expression for the Big Five, a two-stage framework that separates personality encoding from naturalness alignment. FATE-5 first learns continuous soft prompt representations of Big Five profiles using dialogue supervision and an auxiliary survey objective. It then improves dialogue naturalness through preference alignment based on both dialogue quality and personality agreement. Across three language model backbones, FATE-5 maintains strong trait expression while achieving substantially higher dialogue naturalness than other trait-strong methods. On held-out questionnaires, it achieves monotonicit
    
[^71]: NumTrace：审计金融分析师智能体的数字溯源

    NumTrace: Auditing the Number Provenance of Financial Analyst Agents

    [https://arxiv.org/abs/2610.34887](https://arxiv.org/abs/2610.34887)

    提出首个审计金融分析师智能体数字论断溯源（来源与财务语义）的基准 NumTrace，基于 2,163 份 SEC EDGAR 文档构建 540 个问题，通过数值、绑定、支持三项独立判定来检测“数字正确但归属错误”等问题。

    

    金融分析师智能体生成的数字论断可能在公司申报文件中匹配到某个数字，但其所指的实体、概念、期间、单位或范围却是错误的。基于参考答案的检查方法虽然能将此类答案标记为错误，却无法解释其失败的原因。本文提出了 NumTrace，一个用于审计金融分析师智能体所做每个数字论断溯源的新基准，即审计其来源与财务语义。NumTrace 目前包含针对 2,163 份 SEC EDGAR 文档的 540 个问题。据我们所知，这是首个关于谁持有公司股份（13F 表格）以及公司高管应得薪酬（Item 402(i) 条款）的数值基准。我们基于这些文档的冻结快照，按照会计规则构建参考答案。NumTrace 对每个论断给出三项独立判定：数值、绑定和支持。通过将数值判定与绑定判定分开保留，NumTrace 能够检测出“数字正确但科目错误”（RN）这类问题。

    arXiv:2610.34887v1 Announce Type: cross  Abstract: A numerical claim generated by a financial analyst agent can match a number in a company filing but refer to the wrong entity, concept, period, unit or scope. A check based on reference answers can mark such an answer as wrong but does not explain why it fails. In this paper, we present NumTrace, a new benchmark that audits the provenance of each numerical claim made by financial analyst agents: its source and financial semantics. NumTrace currently has 540 questions over 2,163 SEC EDGAR documents. To the best of our knowledge, it is the first numerical benchmark about who holds a company's shares (Form 13F) and what its executives are owed (Item 402(i)). On a frozen snapshot of these documents, we build the reference answers according to accounting rules. NumTrace then gives each claim three separate verdicts: value, binding and support. By retaining separate value and binding verdicts, NumTrace detects right number, wrong account (RN
    
[^72]: ThinkBit：搜索码字、重新校准缩放因子、以推理为准进行选择

    ThinkBit: Search Codewords, Recalibrate Scales, Select by Reasoning

    [https://arxiv.org/abs/2610.34886](https://arxiv.org/abs/2610.34886)

    ThinkBit是一种旨在保持大语言模型推理能力的严格标量2比特量化框架，它通过层敏感度分配各层比特宽度、用Gumbel-softmax逐块搜索最优码字并重新校准缩放因子，有望将推理成本降低8倍。

    

    推理型语言模型每次回答会生成数千个token，其生成速度受内存带宽限制，因此若用2比特而非bfloat16存储权重，有望将推理成本降低8倍。现有2比特量化方法通常以困惑度和选择题问答准确率来评判，但这两项指标几乎无法反映模型是否仍能对更困难的问题进行推理。我们提出ThinkBit，一个严格基于标量的2比特量化框架，专为保持推理能力而设计，其中每个权重都表示为一个整数码字乘以由一组权重共享的bfloat16缩放因子。我们首先使用敏感度指标为每一层分配三值、2比特或3比特的层级，以在层间最优地分配比特宽度；随后采用一种基于Gumbel-softmax的逐层标量量化算法，逐块搜索最优码字。由于逐层方案无法捕捉量化层之间的相互作用……（摘要在此处被截断）

    arXiv:2610.34886v1 Announce Type: new  Abstract: Reasoning language models generate thousands of tokens per answer, and generation speed is limited by memory bandwidth, so storing weights in two bits instead of bfloat16 can potentially make inference 8$\times$ cheaper. Two-bit methods are usually judged by perplexity and multiple-choice question-answering accuracy, both of which say little about whether a model can still reason through more difficult problems. We present ThinkBit, a strictly scalar two-bit framework built to preserve reasoning, in which every weight is an integer codeword times a bfloat16 scale shared by a group of weights. We first use a sensitivity metric to assign each layer a ternary, 2-bit, or 3-bit tier to allocate bitwidths optimally among layers. This is followed by a layerwise Gumbel-softmax-based scalar quantization algorithm that searches for optimal codewords block by block. Since a layerwise scheme does not capture how quantized layers interact with each o
    
[^73]: RubriX：一种快速、可微调的非自回归量规评判器

    RubriX: A Fast, Fine-Tunable Non-Autoregressive Rubric Judge

    [https://arxiv.org/abs/2610.34875](https://arxiv.org/abs/2610.34875)

    RubriX是一种受对比语言模型启发、利用LoRA适配编码器和余弦相似度softmax的非自回归量规评判器，无需文本生成即可高效判断回答符合量规哪个等级，准确率达75.4%，显著超越现有LLM评判方法。

    

    基于量规的评估是判断某个回答符合量规中的哪一个等级。这通常通过提示大型语言模型（LLM）作为评判者来完成，每次判断都需要一次生成式调用，并且需要读取所有等级的描述，因此其成本会随着量规规模的增大而增长。我们提出了RubriX，一个架构受对比语言模型（CLM）启发的量规评判器：一个经过LoRA适配的编码器对输入和每个等级描述进行嵌入表示，再对缩放后的余弦相似度做softmax运算，无需文本生成即可给出每个等级的概率。在公开量规数据和专有合成数据上训练后，RubriX在留出量规上达到75.4%的精确准确率，而TypeSafe的Jev为57.1%，Qwen3-8B评判器为49.8%。在八个人工标注的基准数据集上，它在大多数数据集上的准确率与Jev相差在6个百分点以内，而Jev在其中两个数据集上保持更大领先优势，且在响应平均排名上表现最佳；与G-Eval论文中报告的结果相比……（原文摘要在此处截断）

    arXiv:2610.34875v1 Announce Type: new  Abstract: Rubric-based evaluation asks which level of a rubric a response meets. It is usually done by prompting a large language model (LLM) as a judge, which needs a generative call per judgment and reads every level description, so its cost grows with the rubric. We present RubriX, a rubric judge whose architecture is inspired by contrastive language models (CLM): a LoRA-adapted encoder embeds the input and each level description, and a softmax over scaled cosine similarities gives a probability for every level without text generation. Trained on public rubric data and proprietary synthetic data, RubriX reaches 75.4% exact accuracy on held-out rubrics, against 57.1% for TypeSafe's Jev and 49.8% for a Qwen3-8B judge. On eight human-labelled benchmark sets it is within 6 percentage points of Jev's accuracy on most of them, while Jev keeps a larger lead on two and ranks responses best on average; against the results reported in the G-Eval paper, R
    
[^74]: 自适应检索，选择性回答：评估智能体RAG中的保形事实性

    Adaptive Retrieval, Selective Answers: Evaluating Conformal Factuality in Agentic RAG

    [https://arxiv.org/abs/2610.34872](https://arxiv.org/abs/2610.34872)

    该研究首次系统评估了保形事实性过滤在智能体RAG中的表现，发现更高的保形目标水平能提升回答的完全支持率，但代价是答案可用性的下降。

    

    检索增强生成（RAG）可以在不保证每一条生成的声明都得到支持的情况下改善事实性依据。以往的工作研究了单阶段RAG中的声明级保形事实性，最近也扩展到了多跳RAG。我们研究了保形事实性在智能体RAG中的行为——在这种范式中，语言模型策略会自适应地决定是否发出后续检索查询，或在生成回答前终止检索。我们保留了基于检索的相关性评分方法和生成后的保形过滤机制，并评估了六种模型-数据集配置：GPT-6 Luna和Llama 3.1 8B在HotpotQA、Natural Questions和TriviaQA上的组合。每种配置使用60个固定问题、100个带随机种子的30/30校准-测试划分、五个保形目标水平，以及一个不使用保形预测的基线。在全部六种配置中，更高的目标水平伴随着更高的观测完全支持率，同时也降低了答案的可用性。在95%目标水平下，……

    arXiv:2610.34872v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) can improve factual grounding without guaranteeing that every generated claim is supported. Previous work has studied claim-level conformal factuality in single-stage RAG and, more recently, multi-hop RAG. We examine its behavior in agentic RAG, where a language-model policy adaptively issues follow-up retrieval queries or terminates retrieval before response generation. We retain the retrieval-based relevance-scoring formulation and post-generation conformal filtering and evaluate six model-dataset configurations: GPT-6 Luna and Llama 3.1 8B on HotpotQA, Natural Questions, and TriviaQA. Each configuration uses 60 fixed questions, 100 seeded 30/30 calibration-test partitions, five conformal target levels, and a no-conformal-prediction baseline. Across the six configurations, higher target levels are accompanied by higher observed full-support rates and reduced answer availability. At the 95% target,
    
[^75]: DocTalk：通过基于思维链推理的微调大语言模型增强医学知识检索，实现稳健的临床决策支持

    DocTalk: Enhancing Medical Knowledge Retrieval via Fine-tuned Large Language Models with Chain-of-Thought Reasoning for Robust Clinical Decision Support

    [https://arxiv.org/abs/2610.34871](https://arxiv.org/abs/2610.34871)

    DocTalk是一个基于思维链推理的医生辅助问答系统，通过构建包含500个临床问答对的高保真数据流水线，并利用LoRA对六个开源大语言模型进行参数高效指令微调，实现了从出院小结中生成准确、基于病历的临床决策支持回复。

    

    目标：大语言模型（LLMs）在医疗健康领域的快速发展为AI辅助临床决策开辟了新的可能性；然而，由于幻觉和领域不对齐问题，其安全有效的部署仍然充满挑战。本研究介绍了DocTalk，一个医生辅助问答（QA）系统，旨在从去标识化的出院小结中生成简洁、基于病历的回复。方法：我们开发了一个高保真数据生成流水线，通过句子级分解、思维链引导的问题构建和专家审查，生成了500个临床相关的问答对。我们评估了六个开源大语言模型（包括三个通用模型和三个生物医学模型）在使用低秩适应（LoRA）进行参数高效指令微调前后的表现。该框架集成了提示工程、上下文学习和医学对齐评估，以支持事实性（摘要不完整）……

    arXiv:2610.34871v1 Announce Type: new  Abstract: Objective: The rapid advancement of large language models (LLMs) in healthcare has opened new possibilities for AI-assisted clinical decision-making; however, safe and effective deployment remains challenging because of hallucination and domain misalignment. This study introduces DocTalk, a doctor-assistive question-answering (QA) system designed to generate concise, chart-grounded responses from de-identified discharge summaries. Methods: We developed a high-fidelity data generation pipeline that produces 500 clinically relevant QA pairs through sentence-level decomposition, chain-of-thought-guided question formulation, and expert review. Six open-source LLMs, including three general-purpose and three biomedical models, were evaluated before and after parameter-efficient instruction tuning using Low-Rank Adaptation (LoRA). The framework integrates prompt engineering, in-context learning, and medically aligned evaluation to support factu
    
[^76]: 面向多奖励策略优化的优先级保持优势聚合方法

    Priority-Preserving Advantage Aggregation for Multi-Reward Policy Optimization

    [https://arxiv.org/abs/2610.34862](https://arxiv.org/abs/2610.34862)

    提出PAPO方法，通过非对称地聚合主要优势与辅助优势来解决多奖励强化学习中的优势优先级反转问题，在目标一致时保留辅助监督、在冲突时平滑衰减辅助信号，从而始终优先保证主要目标的优化方向。

    

    LLM后训练通常需要优化多个优先级不等的目标，例如正确性与推理效率。然而，强化学习中广泛使用的加性奖励聚合方式可能会违背这些优先级：当主要优势与辅助优势发生冲突时，足够大的辅助信号能够推翻主要目标所诱导的偏好。我们将这一问题称为“优势优先级反转”。尽管奖励级别的门控能够更直接地保护优先级，但它仍会移除未满足主要条件的采样样本中潜在有用的辅助区分信息。为了解决这一权衡问题，我们提出了主要优势策略优化，这是一种优先级感知的优势聚合方法，以非对称方式结合主要优势与辅助优势。当两个目标一致时，PAPO保留辅助监督信号；当信号发生冲突时，则平滑地衰减辅助信号，从而保持主要优势的符号。

    arXiv:2610.34862v1 Announce Type: cross  Abstract: LLM post-training often optimizes multiple objectives with unequal priorities, such as correctness and reasoning efficiency. However, the widely used additive reward aggregation in RL can violate these priorities: when primary and auxiliary advantages conflict, a sufficiently large auxiliary signal can overturn the preference induced by the primary objective. We term this problem \emph{advantage priority inversion}. Although reward-level gating can protect priority more directly, it still removes potentially useful auxiliary distinctions among rollout samples that fail the primary condition. To address this trade-off, we propose Primary Advantage Policy Optimization (\textbf{PAPO}), a priority-aware advantage aggregation method that asymmetrically combines a primary and an auxiliary advantage. PAPO retains auxiliary supervision when the objectives agree and smoothly attenuates conflicting auxiliary signals, preserving the primary sign 
    
[^77]: LLM智能体时代下面向数据分析中可扩展文本分析的语义原子

    Semantic Atoms for Scalable Text Analysis in Data Analytics in the Era of LLM-Based Agents

    [https://arxiv.org/abs/2610.34860](https://arxiv.org/abs/2610.34860)

    该论文提出两个查询驱动的语义原子——标记和分类，用于在LLM智能体规划的数据分析工作流中划分神经语义处理与符号操作的分工，并支持语义输出的重用，从而实现可扩展的文本分析。

    

    数据分析所需的许多信息都埋藏在文本列中：分析师可能从投诉文本中推断出客户的不满情绪，并用它来解释问题解决时间。我们将文本列处理成为后续数据分析核心环节的工作负载称为TADA（Text Analysis in Data Analytics，数据分析中的文本分析）。在智能体环境中，用户的查询会动态地指定这种文本处理方式，回答查询需要将基于LLM的语义步骤与精确的SQL或Python操作交织执行。这种混合执行方式引出了两个设计问题：哪些工作应该由神经方法完成、哪些应该由符号方法完成，以及后续步骤可以重用哪些语义输出。针对智能体规划的TADA工作流，我们通过两个互补的、由查询驱动的语义原子来回答这些问题。每个语义原子接收一个文本项序列和一个指定语义任务的自然语言查询：**标记**为每个输入文本项返回一个字符串，而**分类**则返回M个类别对象……

    arXiv:2610.34860v1 Announce Type: new  Abstract: Much of the information needed for data analysis is buried in text columns: an analyst may infer customer frustration from complaint text and use it to explain resolution time. We call workloads in which processing text columns is central to subsequent data analysis TADA (\emph{Text Analysis in Data Analytics}). In an agentic setting, the user's query dynamically specifies this processing, and answering it interleaves LLM-based semantic steps with exact SQL or Python operations. This mixed execution raises two design questions: which work should be neural and which symbolic, and which semantic outputs later steps can reuse. We address them for agent-planned TADA workflows with two complementary, query-driven semantic atoms. Each atom takes a sequence of text items and a natural-language query specifying the semantic task: \textbf{Tagging} returns one string per input item, while \textbf{Categorization} returns $M$ category objects over $
    
[^78]: 位置比通道更重要：说服语言模型放弃正确答案

    Position Matters More Than Channel: Talking a Language Model Out of a Correct Answer

    [https://arxiv.org/abs/2610.34809](https://arxiv.org/abs/2610.34809)

    该研究发现建议在提示中的位置比 API 通道更能左右语言模型是否放弃正确答案——裸声明在模型作答前危害最大，而带虚假理由的论证在提问前危害最大，可将模型原本几乎不会答错的题目错误率从 1% 提升至 67%。

    

    语言模型如今在回答问题时会伴随其他声明出现：系统提示中的备注、来自其他智能体的消息，或其自身先前的回答。我们研究需要什么才能让模型放弃它原本能答对的答案，并将以往研究中常常混在一起的三个因素分离开来：建议在提示中所处的位置、据说撰写该建议的人是谁，以及模型已经写下的内容。在 Qwen3.5-27B 和四选项 MMLU-Pro 题目上，在每次比较中固定其余建议的情况下，我们发现了三点。首先，位置远比 API 通道更重要，且对裸声明和论证的影响方向相反：一个简单的“答案是 X”在模型即将作答之前危害最大，而带有虚假理由的同一声明则在问题提出之前危害最大，此时它能把模型原本几乎从不答错的题目错误率从 1% 提升到 67%；在系统消息和用户消息之间移动该声明……（摘要原文在此处截断）

    arXiv:2610.34809v1 Announce Type: new  Abstract: Language models now answer questions alongside other claims: a note in the system prompt, a message from another agent, their own earlier answer. We ask what it takes to talk a model out of an answer it gets right on its own, and separate three factors that earlier studies vary together: where the advice sits in the prompt, who is said to have written it, and what the model has already written. On Qwen3.5-27B and four-option MMLU-Pro items, with the rest of the advice held fixed in each comparison, we find three things. First, position matters far more than the API channel, in opposite directions for a bare claim and an argument: a bare "the answer is X" does most harm just before the model answers, while the same claim with a false rationale does most harm before the question, where it raises error on items the model otherwise almost never misses from 1% to 67%; moving the claim between the system message and the user message at the sam
    
[^79]: 文本才是瓶颈：立直麻将中大语言模型游戏对局的小规模复现

    Text Is the Bottleneck: A Small-Scale Replication of LLM Game Playing in Riichi Mahjong

    [https://arxiv.org/abs/2610.34791](https://arxiv.org/abs/2610.34791)

    该论文在小预算下复现立直麻将 LLM 对局实验，证明文本输入是瓶颈——仅数百万参数的卷积/Transformer 牌面编码器在决策一致率和实战排名上均超越了 20 亿参数以下的文本提示 LLM，并能与教师 AI Mortal 冠军达到统计意义上的平局。

    

    我们在大幅缩减的预算下复现了《大语言模型能否掌握复杂卡牌游戏？》(arXiv:2509.01328) 中的立直麻将赛道：模型参数量最多 2B（20 亿），仅使用一块 16 GB 消费级 GPU，约五个 GPU 日。(1) 在数据与标签匹配的条件下，文本是瓶颈：在同样的 10 万决策训练集上，一个 260 万参数的卷积牌面编码器与强 AI Mortal 在测试决策上的一致率达到 80.6%，高于我们训练的所有文本提示 LLM（最佳为 77.8%），并且在牌桌上的排名更好（1,000 局平均 2.85 位，而 LLM 约 3.0 位；n=100 时配对差异不显著）。将训练扩展到 171 万决策后，一个 490 万参数的 Transformer 达到 94.7% 的一致率，在相同随机种子下击败 0.8B 参数的 LLM（-0.45 位次，符号检验 p=0.005），并在与三位 Mortal 冠军版本的对战中达到统计意义上的平局（TOST 在 0.10 位次以内，p=6e-4）；与教师对手打平大致就是纯模仿学习的上限。(2) 训练方面……

    arXiv:2610.34791v1 Announce Type: new  Abstract: We replicate the riichi-mahjong track of "Can Large Language Models Master Complex Card Games?" (arXiv:2509.01328) under a drastically smaller budget: models of at most 2B parameters, one 16 GB consumer GPU, about five GPU-days. (1) With matched data and labels, text is the bottleneck: on the same 100k-decision training set, a 2.6M-parameter convolutional board encoder agrees with the strong AI Mortal on 80.6% of test decisions, above every text-prompt LLM we train (best 77.8%), and places better at the table (2.85 over 1,000 games vs. ~3.0 for the LLMs; paired difference at n=100 not significant). Scaled to 1.71M decisions, a 4.9M-parameter Transformer reaches 94.7% agreement, beats the 0.8B LLMs on identical seeds (-0.45 ranks, sign test p=0.005), and is statistically equivalent to parity against three Mortal champions (TOST within 0.10 ranks, p=6e-4); parity against a teacher-opponent is roughly the ceiling of pure imitation. (2) Trai
    
[^80]: BehaviorMem：从真实世界电商行为流中构建与评估用户记忆

    BehaviorMem: Building and Evaluating User Memory from Real-World E-Commerce Behavior Streams

    [https://arxiv.org/abs/2610.34790](https://arxiv.org/abs/2610.34790)

    该论文提出BehaviorMem，一个基于真实电商行为流构建用户记忆的方法与评估框架，能够同时支持并分别评估“记住具体证据与兴趣”和“推断潜在需求与偏好”这两种核心能力。

    

    用户记忆能够帮助个性化系统随时间不断加深对用户的理解。在电子商务中，用户通过显式语言和与产品的隐式交互来表达他们的偏好和需求。我们研究从这些交互日志中构建的记忆，考察它如何“记住”具体的证据和已确立的兴趣，并利用这些证据来“推断”潜在的需求和偏好，从而为超出充分观察到的产品领域之外的判断提供依据。现有的记忆方法主要针对基于语言的交互，且相关评估提供的真实行为数据有限，对这些能力的区分也不充分。这使得如何从行为日志中同时支持并分别评估“记忆”与“推断”仍是一个开放问题。我们围绕这些联合需求形式化了持续性行为用户记忆问题，并开发了专门的评估框架和记忆方法来解决这些问题。

    arXiv:2610.34790v1 Announce Type: new  Abstract: User memory helps personalized systems build on their understanding of users over time. In e-commerce, users convey their preferences and needs through both explicit language and implicit interactions with products. We study memory built from logs of these interactions, examining how it can \emph{remember} concrete evidence and established interests, and use that evidence to \emph{infer} underlying needs and preferences that inform judgments beyond well-observed product domains. Existing memory methods primarily target language-based interactions, and evaluations offer limited real behavioral data and limited separation of these capabilities. This leaves open how to jointly support and separately evaluate remembering and inference from behavioral logs. We formulate the problem of continual behavioral user memory around these joint requirements and develop a dedicated evaluation framework and memory method to address them. \textbf{Behavio
    
[^81]: 自然干扰器：当非鲁棒特征走向对抗性

    Natural Jammers: When Non-Robust Features Become Antagonistic

    [https://arxiv.org/abs/2610.34785](https://arxiv.org/abs/2610.34785)

    该论文发现非鲁棒特征并非仅是捷径，而是可作为“自然干扰器”通过频谱-空间锁定机制在干净图像上主动误导标准模型，且尽管单个干扰器对几何扰动脆弱，干扰现象在整体上依然持续存在。

    

    深度神经网络已知会依赖与人类感知不一致的非鲁棒特征。以往的研究大多将这些特征视为有助于标准模型在干净数据上表现的投机性捷径。我们则证明非鲁棒线索可以具有对抗性：它们可以作为“自然干扰器”，在干净、未经修改的图像上主动误导标准模型。以标准模型与鲁棒模型之间的分歧作为分析视角，我们识别出一个“受扰集合”，在该集合上鲁棒模型的性能优于标准模型。我们为“频谱-空间锁定”提供了因果证据：只有当高频成分与特定空间结构精确对齐时，干扰才会发生。通过频谱滤波或微小的几何位移打破这种锁定，即可恢复正确的预测。至关重要的是，我们揭示了一个“替代悖论”：尽管单个干扰器对几何位移十分脆弱，但干扰的总体普遍性依然存在……

    arXiv:2610.34785v1 Announce Type: cross  Abstract: Deep neural networks are known to rely on non-robust features that align poorly with human perception. Prior work has largely framed these features as opportunistic shortcuts that help standard models on clean data. We show instead that non-robust cues can be antagonistic: they can act as Natural Jammers that actively mislead standard models on clean, unmodified images. Using disagreement between a standard model and robust models as an analysis lens, we identify a Jammed Set on which robust models outperform standard models. We provide causal evidence for Spectral-spatial Locking: jamming arises only when high-frequency components are precisely aligned with specific spatial structures. Disrupting this lock through spectral filtering or small geometric shifts can restore correct predictions. Crucially, we uncover a Substitution Paradox: while individual jammers are fragile to geometric shifts, the aggregate prevalence of jamming remain
    
[^82]: DynEdit：面向大语言模型的动态熵引导序列编辑

    DynEdit: Dynamic Entropy-Guided Sequential Editing for Large Language Models

    [https://arxiv.org/abs/2610.34768](https://arxiv.org/abs/2610.34768)

    提出DynEdit框架，发现词元熵与编辑难度及语义重要性密切相关，并利用熵引导的非均匀滑动和词元级优化来提升大语言模型长序列知识编辑的覆盖与效果。

    

    知识编辑旨在更新大语言模型中过时或错误的知识，而无需进行完整的重新训练。现有的参数编辑方法在处理长序列知识时仍然存在局限：单点编辑器对扩展目标的覆盖不足，而基于固定分块的编辑器则在僵化的预定义窗口内均匀分配编辑力度，无法优先处理困难且信息密集的词元。在本工作中，我们发现词元熵与编辑难度和语义重要性密切相关：高熵区域更难编辑，且通常包含关键的事实内容。基于这一观察，我们提出了DynEdit，一个用于长序列知识编辑的动态框架。DynEdit利用词元级熵来引导非均匀滑动，增加对困难高熵区域的覆盖，并进一步应用熵引导的词元级优化，以加强对信息密集内容的更新。

    arXiv:2610.34768v1 Announce Type: cross  Abstract: Knowledge editing aims to update outdated or incorrect knowledge in large language models without full retraining. Existing parameter-editing methods remain limited for long-sequence knowledge: single-point editors provide insufficient coverage for extended targets, while fixed chunk-based editors distribute editing effort uniformly across rigid predefined windows, failing to prioritize difficult, information-dense tokens. In this work, we show that token entropy is closely associated with both editing difficulty and semantic importance: high-entropy regions are harder to edit and often contain key factual content. Motivated by this observation, we propose DynEdit, a dynamic framework for long-sequence knowledge editing. DynEdit uses token-level entropy to guide non-uniform sliding, increasing the coverage of difficult high-entropy regions, and further applies entropy-guided token-level optimization to strengthen updates on information
    
[^83]: 从信念推断到任务行动：在可验证任务上对心智理论的多层次评估

    From Belief Inference to Task Action: A Multi-Level Evaluation of Theory of Mind on Verifiable Tasks

    [https://arxiv.org/abs/2610.34759](https://arxiv.org/abs/2610.34759)

    该研究提出了一个基于可验证任务的多层次心智理论评估框架，将家庭协作与日历安排任务扩展为非对称信息场景，用以区分语言模型“推断他人信念”与“将信念运用于实际协作行动”这两种能力的差异。

    

    有效的协作要求语言模型理解他人知道什么、相信什么，并据此采取行动。那些假设他人拥有与自己相同信息的模型可能会省略必要的沟通，从而破坏协调配合。许多心智理论基准通过故事问答来评估心理状态推断能力，但正确回答这些问题并不能证明模型能够在后续任务中运用这些判断。我们提出了一种建立在现有可验证任务之上的多层次评估方法。我们在保留原有约束条件和验证器的前提下，将家庭协作与日历安排任务扩展为具有非对称信息的场景。通过相互独立的接口，分别评估对世界的理解、信息获取、信念、决策和执行能力。对照组通过提供个人证据或正确信念，来区分“恢复他人状态”与“运用他人状态”这两种能力。在六种模型配置中，这些能力表现出明显的分化。在匹配的日程安排案例中，Claude 恢复了所有智能体的……（原文摘要在此处截断）

    arXiv:2610.34759v1 Announce Type: new  Abstract: Effective collaboration requires language models to understand what others know and believe, and to act accordingly. Models that assume others share their information may omit necessary communication and undermine coordination. Many theory-of-mind (ToM) benchmarks assess mental-state inference through story questions, but correct answers do not establish that models can use these judgments in subsequent tasks. We introduce a multi-level evaluation built on existing, verifiable tasks. We extend household collaboration and calendar scheduling with asymmetric information while retaining their original constraints and validators. Independent interfaces assess world understanding, information access, beliefs, decisions, and execution. Controls supply personal evidence or correct beliefs to distinguish state recovery from state use. Across six model configurations, these abilities diverge. On matched scheduling cases, Claude recovers all agent
    
[^84]: LoRA的机制性洞察：通过路径修补实现自适应微调的层稀疏性研究

    Mechanistic Insights into LoRA: Layer Sparsity for Adaptive Fine-Tuning via Path Patching

    [https://arxiv.org/abs/2610.34756](https://arxiv.org/abs/2610.34756)

    通过机制可解释性分析揭示LoRA的适应效果具有显著的层级稀疏性且关键层高度依赖任务，据此提出校准驱动的Adaptive-LoRA方法，利用冻结基座零消融自动选择任务特定层，无需穷举搜索即可实现更高效的自适应微调。

    

    低秩适应（LoRA）已成为大型语言模型参数高效微调的主流范式。标准的全层LoRA在整个Transformer中应用适应，尽管其各组件的贡献存在异质性。在这项工作中，我们进行了系统的机制可解释性分析，以理解LoRA如何影响模型内部。我们观察到LoRA的适应效果高度非均匀，表现出显著的组件级和层级稀疏性。最关键的是，我们发现重要的层高度依赖于具体任务，且跨任务的重叠有限。基于这些洞察，我们提出了Adaptive-LoRA，这是一种校准驱动的方法，通过冻结基座模型的零消融自动识别任务特定层，无需穷举搜索即可实现高效的层选择。在Qwen2.5-1.5B上跨11个多样化推理任务的大量实验表明，Adaptive-LoRA优于（原文在此处截断）

    arXiv:2610.34756v1 Announce Type: cross  Abstract: Low-Rank Adaptation (LoRA) has emerged as the dominant paradigm for parameter-efficient fine-tuning of large language models. Standard all-layer LoRA applies adaptation throughout the transformer, despite heterogeneous contributions from its components. In this work, we conduct a systematic mechanistic interpretability analysis to understand how LoRA affects model internals. We observe that LoRA's adaptation effect is highly non-uniform, with significant component-wise and layer-wise sparsity. Most critically, we find that important layers are highly task-dependent, with limited overlap across tasks. Motivated by these insights, we propose Adaptive-LoRA, a calibration-driven method that automatically identifies task-specific layers via frozen-base zero-ablation, enabling efficient layer selection without exhaustive search. Extensive experiments on Qwen2.5-1.5B across 11 diverse reasoning tasks demonstrate that Adaptive-LoRA outperforms
    
[^85]: DecisionTraceBench：多方口语对话中决策关键信息的追踪

    DecisionTraceBench: Tracking Decision-Critical Information in Multi-Party Spoken Conversations

    [https://arxiv.org/abs/2610.34746](https://arxiv.org/abs/2610.34746)

    提出DecisionTraceBench基准，面向多方中文金融语音对话中的决策关键信息状态追踪（DCIST），联合评估决策事件序列、说话人归属、范围和时间证据，并以轨迹级F1衡量完整事件链的恢复能力。

    

    在金融等高风险领域中，关键信息通过多方交互不断演化。恢复这些变化及其来源是实现可靠语音理解的基础。现有语音基准主要聚焦于转录、局部理解和轮次级状态预测，对跨说话人的联合决策演化评估不足。我们提出了DecisionTraceBench，一个面向决策关键信息状态追踪的语音状态追踪基准。该基准联合评估决策事件序列、说话人归属、范围和时间证据，并使用轨迹级F1分数来衡量完整事件链的恢复程度。我们的数据集包含涉及两至四位说话人的中文金融对话，包括100.39小时的合成音频和8.61小时的真人录音。在固定的真人录音子集上，五个级联系统和两个直接音频模型取得了DCIST分数……（摘要截断）

    arXiv:2610.34746v1 Announce Type: new  Abstract: In high-stakes domains such as finance, critical information evolves through multi-party interaction. Recovering these changes and their sources is fundamental to reliable speech understanding. Existing speech benchmarks focus on transcription, local understanding, and turn-level state prediction, leaving joint decision evolution across speakers insufficiently evaluated. We introduce DecisionTraceBench, a speech state-tracking benchmark for Decision-Critical Information State Tracking (DCIST). It jointly evaluates decision event sequences, speaker attribution, scope, and temporal evidence, using trajectory-level F1 to measure complete event-chain recovery. Our dataset comprises Chinese financial dialogues involving two to four speakers, with 100.39 hours of synthetic audio and 8.61 hours of human-performed recordings. On a fixed cohort of human-performed recordings, five cascaded systems and two direct-audio models achieve DCIST scores o
    
[^86]: LADDER：诊断噪声反馈下的领域级自我评估

    LADDER: Diagnosing Domain-Level Self-Assessment Under Noisy Feedback

    [https://arxiv.org/abs/2610.34737](https://arxiv.org/abs/2610.34737)

    提出LADDER框架研究语言模型在噪声反馈下的自我评估，发现模型虽能准确报告给定的性能信息，却难以从交互经验中形成可信的自我评估，表面准确的评估可能掩盖能力差异且无法预测未来任务表现。

    

    了解自身何时可靠，对于语言模型支持人机协作至关重要。然而交互反馈并不总是完整或正确的，可能会误导模型对自身能力的理解。我们提出LADDER来研究语言模型在不可靠反馈下如何进行自我评估。我们的发现表明，模型能够准确报告所提供的性能信息，但仍难以从交互经验中形成可信的自我评估。表面上准确的评估可能会掩盖能力差异，并不一定能预测未来的任务表现。这些发现表明，可信的自我评估应当基于观察到的任务表现及其预测价值来衡量。我们的研究为模型自我认知的局限性提供了实证证据，并为在可靠的人机协作中评估能力意识奠定了基础。

    arXiv:2610.34737v1 Announce Type: new  Abstract: Knowing when they are reliable is essential for language models to support human-AI collaboration. Yet interaction feedback is not always complete or correct, and may mislead models' understanding of their own capabilities. We introduce LADDER to study how language models assess themselves under unreliable feedback. Our findings show that models can accurately report supplied performance information while still struggling to form trustworthy self-assessments from interaction experience. Apparently accurate assessments can obscure capability differences and do not necessarily predict future task performance. These findings suggest that trustworthy self-assessment should be evaluated against observed task performance and its predictive value. Our study provides empirical evidence of the limits of model self-knowledge and a basis for evaluating capability awareness in reliable human-AI collaboration.
    
[^87]: iPrefBench：一个以用户为中心的个性化移动设备使用智能体基准测试

    iPrefBench: A User-Centric Benchmark for Personalized Mobile Use Agents

    [https://arxiv.org/abs/2610.34735](https://arxiv.org/abs/2610.34735)

    iPrefBench是一个在可执行iOS环境中、借助交互式模拟用户来评估个性化移动使用智能体“从用户活动历史中推断缺失偏好”与“向用户询问必要信息”这两项关键能力的基准测试，其基于包含3,051个活动事件的用户画像，涵盖26个应用中的130个不明确任务。

    

    一个真正了解用户并能够代表用户行事的智能体，一直是AI助手领域长期以来的承诺。计算机使用智能体通过熟练地在各个应用中执行完全指定的任务，使这一承诺触手可及。然而，现实世界中的用户请求往往是不够明确的。为了针对特定用户正确行事，智能体必须从用户的活动历史中推断出部分缺失的信息，并就只有用户才能提供的信息进行询问，同时既不能做出无依据的假设，也不能用不必要的问题给用户造成负担。我们提出了iPrefBench，这是一个用于评估个人移动设备使用智能体“推断”（infer）与“询问”（ask）能力的基准测试，它在可执行的iOS环境中运行，并配有交互式的模拟用户。iPrefBench基于一个包含3,051个过往活动事件的一致性用户画像构建，涵盖跨26个应用的130个不够明确的用户任务。每个被省略的用户相关需求，按照设计，要么是可从历史中推断的，要么是需要向用户询问的……

    arXiv:2610.34735v1 Announce Type: new  Abstract: An agent that truly knows and acts on behalf of its user has been a long-standing promise of AI assistants. Computer-use agents bring this promise within reach by capably executing fully specified tasks across apps. However, real-world user requests are often underspecified. To act correctly for the particular user, an agent must infer some missing information from the user's activity history and ask for other information that only the user can provide, without making unsupported assumptions or burdening the user with unnecessary questions. We present iPrefBench, a benchmark for evaluating the infer and ask capabilities of personal mobile-use agents in an executable iOS environment with an interactive simulated user. Grounded in a coherent persona with 3,051 past activity events, iPrefBench comprises 130 underspecified user tasks across 26 apps. Each omitted user-dependent requirement is, by construction, either history-inferable or ask-
    
[^88]: 知道证据何时足够：基于Horn闭包证书的需求导向搜索

    Knowing When the Evidence Is Enough: Requirement-Directed Search with Horn-Closure Certificates

    [https://arxiv.org/abs/2610.34684](https://arxiv.org/abs/2610.34684)

    该论文提出Certify框架，通过将查询分解为需求超图并利用Horn闭包运算生成证书，把搜索增强模型“证据是否足够”的判断结构化，从而可靠地决定回答、继续搜索或放弃回答并指出缺失的需求。

    

    每次网页搜索之后，搜索增强的模型必须判断其证据是否充分：回答问题、再次搜索，或放弃回答。现有系统将这一判断交给模型的整体性直觉判断，这可能导致模型接受一个虽然提及了所有相关实体、却缺少某个所需事实的上下文。我们转而将“充分性”结构化：针对每个查询构建需求超图，用以声明哪些事实可以共同推导出另一个事实。这带来了两个挑战：仅凭问题本身推导出这些需求，以及将每条需求链接到关于正确实体的文本段落。为此，Certify 对问题进行分解，按需求逐项搜索，并且只有在通过实体检查之后才接受链接。对这些链接进行Horn闭包运算即可产生一个证书——即最小充分段落集及其推导过程；若无法满足，则放弃回答并明确指出尚未满足的需求。在检索、需求和链接均正确的前提下，该决策是可靠的，且在实体检查的范围内是完备的。在包含不可回答孪生问题的三个语料库上的汇总结果显示，Certi（摘要在此处截断）

    arXiv:2610.34684v1 Announce Type: cross  Abstract: After each Web search, a search-augmented model must decide whether its evidence suffices: answer, search again, or abstain. Prevailing systems leave this to holistic model judgement, which can accept a context naming every entity yet lacking one required fact. We instead make sufficiency structural: a per-query requirement hypergraph states which facts jointly yield another fact. This raises two challenges: deriving requirements from the question alone, and linking each requirement to passages about the right entity. Certify therefore decomposes the question, searches per requirement, and accepts links only after entity checks. Horn closure over the links yields a certificate, the minimal sufficient passages and their derivation, or an abstention naming open requirements. Given correct retrieval, requirements and links, the decision is sound, and complete up to the entity check. Pooled over three corpora with unanswerable twins, Certi
    
[^89]: 智能体记忆即推荐：面向大语言模型智能体长期记忆的推荐系统设计

    Agent Memory as Recommendation: Recommender-System Design for Long-Term Memory of LLM Agents

    [https://arxiv.org/abs/2610.34673](https://arxiv.org/abs/2610.34673)

    该论文的核心创新是将LLM智能体的记忆选择问题形式化为预算约束下的板块推荐问题（请求为用户、记忆为物品、上下文窗口为板块），从而把各种智能体记忆系统统一纳入推荐级联的设计空间，并将准确性分解为可分析的组件。

    

    在多次会话中为用户提供服务的大语言模型（LLM）智能体，必须在每次请求时从存储的记忆中挑选出少数几条进入有限的上下文窗口。这一选择在很大程度上决定了智能体能否正确回答问题，并且需要在网络规模下对每次请求执行。这一选择面临三个挑战：庞大且不断增长的存储库、每次请求的成本预算限制，以及需要全部证据的多会话问题。许多记忆系统在构建笔记或图结构时，每份历史记录需要数百到数千次LLM调用，且各系统都是独立设计的，因此难以确定究竟是哪个组件带来了准确性。这些挑战与推荐系统的特性高度相似，因此我们将智能体记忆形式化为预算约束下的板块推荐：请求对应用户，存储的记忆对应物品，上下文窗口对应板块。这一形式化将记忆系统置于推荐级联的统一设计空间中，并将准确性分解为完整性……

    arXiv:2610.34673v1 Announce Type: cross  Abstract: Large language model (LLM) agents that serve users across many sessions must choose, for every request, the few stored memories that enter a limited context window. This choice largely decides whether the agent answers correctly, and it runs on every request at Web scale. This selection faces three challenges: a large and growing store, a per-request cost budget, and multi-session questions that need all of their evidence. Many memory systems build notes or graphs with hundreds to thousands of LLM calls per history, and each is designed on its own, so it is unclear which component yields the accuracy. These challenges mirror the properties of recommendation, so we formalize agent memory as budgeted slate recommendation, with the request as the user, stored memories as items and the context window as a slate. The formalization places memory systems in one design space of recommender cascades and factorizes accuracy into the completeness
    
[^90]: AdaTutor：通过教学策略选择实现自适应辅导

    AdaTutor: Adaptive Tutoring through Pedagogical Strategy Selection

    [https://arxiv.org/abs/2610.34659](https://arxiv.org/abs/2610.34659)

    本文提出比较式教学监督（CPS）训练框架，使AI辅导系统能够根据学生情况自适应选择最优教学策略与反馈，在数学和科学推理任务上超越固定策略基线，并具备跨数据集迁移能力。

    

    有效的辅导不仅需要确定提供什么信息，还需要确定以何种形式的帮助进行辅导。辅导者可以使用不同的教学策略来应对学生的同一个错误，并随着互动的进展调整其方法。我们提出了比较式教学监督，这是一个用于学习为学生学习选择最优策略和反馈的训练框架。我们在数学和科学推理数据集上对CPS进行了评估。经过五轮辅导交互后，CPS优于最强的固定策略基线。学习到的辅导者可以跨数据集迁移，其策略选择还可以为另一个基于API的辅导系统提供指导。

    arXiv:2610.34659v1 Announce Type: new  Abstract: Effective tutoring requires determining not only what information to provide, but also which form of assistance to deliver. A tutor may address the same student error using various pedagogical strategies, adapting its approach as the interaction progresses. We introduce Comparative Pedagogical Supervision (CPS), a training framework to learn to select the optimal strategy and feedback for student learning. We evaluate CPS on math and science reasoning datasets. After five tutoring turns, CPS outperforms the strongest fixed-strategy baselines. The learned tutor transfers across datasets, and its strategy selection can also inform a separate API-based tutor.
    
[^91]: 听得足够即可行动：口语对话系统中的早期工具调用

    Heard Enough to Act: Early Tool Calling in Spoken Dialogue Systems

    [https://arxiv.org/abs/2610.34653](https://arxiv.org/abs/2610.34653)

    本文提出了一个包含26,694个对话（786小时、48个领域）的合成口语对话语料库，用于训练全模态语言模型在用户话轮尚未结束时即判断“何时听得足够”并提前调用工具，从而降低语音代理的响应延迟。

    

    对延迟敏感的语音代理不仅需要知道调用哪个工具，还需要知道何时听到的信息已足够采取行动。即使请求已经可以执行，等待用户话轮完成也会延迟执行。然而，现有的语音代理和口语对话语料库均未定义请求变为可执行的最早时间点。我们引入了一个合成口语对话流水线，提供最早可执行点监督，标记观察到的语音何时能够确定一个工具及其所需参数。通过受控任务生成、语音合成与对齐以及门控标注，我们构建了26,694个对话，总计786小时，涵盖48个领域，且训练集与测试集的参数值互不相交。我们还创建了τ²-bench的音频渲染版本用于外部评估。我们在此语料库上训练了一个全模态语言模型，使其能够进行因果的、分块级别的决策，并生成部分转录文本用于下游任务。

    arXiv:2610.34653v1 Announce Type: new  Abstract: Latency-sensitive spoken agents must know not only which tool to call, but also when they have heard enough to act. Waiting for the turn to complete delays execution even when the request is already actionable. However, neither existing spoken agents nor spoken dialogue corpora define the earliest point at which a request becomes actionable. We introduce a synthetic spoken-dialogue pipeline that provides earliest-actionable-point supervision, marking when the observed speech grounds a tool and its required arguments. Through controlled task generation, speech synthesis and alignment, and gated annotation, we construct 26,694 dialogues totaling 786 hours across 48 domains, with disjoint training and test argument values. We additionally create an audio rendering of $\tau^2$-bench for external evaluation. We train an omni-modal language model on this corpus, enabling causal, chunk-level decisions and partial-transcript generation for downs
    
[^92]: 跨循环查询交接：Ouro 循环语言模型中后续循环如何重定向两跳检索

    Cross-Loop Query Handoff: How Later Loops Redirect Two-Hop Retrieval in the Ouro Looped Language Models

    [https://arxiv.org/abs/2610.34639](https://arxiv.org/abs/2610.34639)

    通过在 Ouro 循环语言模型的不同循环间移植注意力查询，研究发现后续循环的查询（而非键或值）是修复两跳多跳检索失败的关键机制，注入循环3的查询可挽救超过90%的循环1失败案例。

    

    在预训练的循环语言模型中，后续循环是如何修复两跳检索的？我们通过在相同层、头和最终提示位置上于不同循环之间移植注意力查询，研究了 Ouro-1.4B 和 Ouro-2.6B 模型。在合成的上下文内问题上，将按层索引的循环3查询集合注入整个循环1中，可以挽救90%和97%的在循环1回答错误而在循环3回答正确的问题，而使用打乱顺序的供体查询仅能挽救24%和21%。在 Ouro-2.6B 中，这种挽救效果在贪心生成下依然成立；而在 Ouro-1.4B 中，该效果仅限于四候选排序任务。交叉的供体-受体交换更倾向于通过供体的桥接实体进行检索，而非复制其答案，尽管 Ouro-2.6B 中仍存在显著的位置成分。来自后期循环的键和值能挽救的目标则少得多。将查询冻结在循环1的取值上，对两跳准确率的损害大于提供桥接实体的匹配问题。在自然问题上，挽救效果较小。

    arXiv:2610.34639v1 Announce Type: cross  Abstract: How do later loops in a pretrained looped language model repair two-hop retrieval? We study Ouro-1.4B and Ouro-2.6B by transplanting attention queries between loops at the same layer, head and final prompt position. On synthetic in-context questions, injecting the layer-indexed collection of loop-3 queries throughout loop 1 rescues 90% and 97% of questions answered incorrectly at loop 1 and correctly at loop 3, versus 24% and 21% for shuffled donors. The rescue also holds under greedy generation in Ouro-2.6B; in Ouro-1.4B it is limited to four-candidate ranking. Crossed donor-recipient swaps favor retrieval through the donor's bridge entity over copying its answer, although a substantial positional component remains in Ouro-2.6B. Late keys and values rescue far fewer targets. Freezing queries at their loop-1 values also damages two-hop accuracy more than a matched question that supplies the bridge. On natural questions, rescue is small
    
[^93]: 基于决策模型的实体匹配

    Entity Matching using Decision Models

    [https://arxiv.org/abs/2610.34626](https://arxiv.org/abs/2610.34626)

    本文提出用决策模型作为实体匹配的新替代方案，证明其在零样本设置下无需任务特定训练数据即可取得接近生成式大语言模型的匹配性能。

    

    实体匹配是判断两条记录是否描述同一个现实世界实体的任务。当前的实体匹配方法依赖于需要针对特定任务进行微调的预训练语言模型（PLM），或者依赖于无需任务特定微调、但在匹配大规模记录集时速度慢且成本高的生成式大语言模型（LLM）。决策模型是一类新型的基于大语言模型的模型，它们为每个预定义的答案选项返回概率，而不是生成文本。本文研究了决策模型作为生成式大语言模型和微调预训练语言模型在实体匹配任务中的替代方案。我们在八个基准数据集上以零样本设置评估了六个决策模型，并将其性能与生成式大语言模型 GPT-6 Luna 以及基于预训练语言模型的匹配器 Ditto 进行了比较。我们的实验表明，在无需任务特定训练数据的情况下，四个最强的决策模型取得了接近 GPT-6 Luna 的平均 F1 分数。

    arXiv:2610.34626v1 Announce Type: cross  Abstract: Entity matching is the task of deciding whether two records describe the same real-world entity. Current entity matching methods rely on pre-trained language models (PLMs) which require task specific fine-tuning or generative large language models (LLMs) which do not require task-specific fine-tuning but are slow and expensive at matching large record sets. Decision models are a new class of LLM-based models that return a probability for each predefined answer option rather than generate text. This paper investigates decision models as an alternative to generative LLMs and fine-tuned PLMs for entity matching. We evaluate six decision models in a zero-shot setting across eight benchmarks and compare their performance with that of the generative LLM GPT-6 Luna and the PLM-based matcher Ditto. Our experiments show that, without task-specific training data, the four strongest decision models achieve mean F1 scores close to that of GPT-6 Lu
    
[^94]: 每个音素一个平均对数梅尔向量：面向语音识别的简单快速文本注入方法

    One Mean Log-Mel Vector per Phoneme: Simple and Fast Text Injection for Speech Recognition

    [https://arxiv.org/abs/2610.34623](https://arxiv.org/abs/2610.34623)

    该论文提出用简单的音素级查找表（每个音素一个平均对数梅尔向量）取代神经TTS模型来合成文本注入输入，使语音识别训练提速1.8倍且性能相当，并在LibriSpeech上取得仅使用该数据集资源的最新最优词错误率。

    

    文本注入（Text injection）是一种让语音识别模型在成对音频之外还能从未配对文本中学习的技术，其做法是将由文本导出的合成输入馈送给语音编码器，使整个模型从中学习。此前，我们需要用一个冻结的文本转语音（TTS）模型来合成这种输入。在本工作中，我们用一个简单的查找表取代了TTS模型：每个音素对应一个平均对数梅尔向量（或独热编码），根据从对齐统计中抽取的随机时长进行重复，并在词与词之间插入静音。该方法使训练速度比使用神经TTS快1.8倍，同时达到相当的词错误率。结合外部语言模型，我们的系统在LibriSpeech的{dev,test}-{clean,other}四个测试集上分别取得1.36 / 2.90 / 1.52 / 3.24 %的词错误率，据我们所知，这是仅使用LibriSpeech配对数据、文本和词典所达到的最优结果。我们还分析了合成输入的各个组成部分（时长、声学特征），比较了不同的HMM对齐器，在25000小时的Loquacious数据集上评估了方法的扩展性，并展示了……

    arXiv:2610.34623v1 Announce Type: new  Abstract: Text injection trains speech recognition models on unpaired text alongside paired audio. We feed the speech encoder synthetic input derived from text so the whole model learns from it. Previously, we synthesized this input with a frozen text-to-speech (TTS) model. Here, a simple lookup table replaces the TTS model: a mean log-mel vector (or one-hot encoding) per phoneme, repeated for a random duration drawn from alignment statistics, with silence between words. Training is 1.8 times faster than with the neural TTS, achieving comparable word error rates. With an external language model, our system reaches 1.36 / 2.90 / 1.52 / 3.24 % WER on LibriSpeech {dev,test}-{clean,other}, to our knowledge state-of-the-art using only LibriSpeech paired data, text, and lexicon. We analyze components of synthetic input (durations, acoustic features), compare different HMM aligners, evaluate scaling on the 25 000 h Loquacious dataset, and show that full-
    
[^95]: 《着装会改变价格吗？视觉语言模型对同一件低价项链估值中的情境光环效应》

    Does the Outfit Change the Price? Context Halo in Vision-Language Model Valuation of an Identical Low-Cost Necklace

    [https://arxiv.org/abs/2610.34591](https://arxiv.org/abs/2610.34591)

    该研究发现视觉语言模型在为同一件低价项链估值时存在“情境光环效应”：项链因佩戴者的着装与场景不同（正式着装对比休闲或无人物场景）而被估高出约2至4倍，且这一偏差在纯文本描述中同样出现，表明模型估值受情境线索系统性影响而非真正基于物品本身。

    

    一条网购价格仅为2.43美元的金色调粗链条项链以四种方式拍摄：穿着正式服装站在木镶板墙前佩戴、在聚会上佩戴、穿着休闲服在回收场佩戴，以及平铺摆放且无人出镜。我们要求六个前沿语言模型在1,164次独立且顺序的API会话（共记录4,604行数据）中估算其价格，并比较不同场景下的估值结果。在四个能够回答图像提示的模型中，同一条项链在正式照片中的估值比平铺照片高出1.9至3.3倍（均p < 0.01）。与回收场照片相比，估值比值范围从1.23（Grok 4.5，不显著）到3.57（Kimi K3），其中Claude Fable 5为3.32。在仅使用文本的任务版本中（即用文字描述场景而非图像），全部六个模型给出的正式场合对回收场估值比值为2.2至3.9倍，说明该效应并不依赖于像素信息。平铺展示时的估值本身约为（原文此处截断，疑为零售价）的7.8至18.7倍。

    arXiv:2610.34591v1 Announce Type: new  Abstract: A chunky gold-tone chain necklace bought online for 2.43 USD was photographed four ways: worn with formal attire against a wood-panelled wall, worn at a party, worn with casual clothing in a recycling yard, and laid flat with no person present. We asked six frontier language models to estimate its price in 1,164 independent and sequential API sessions (4,604 logged rows) and compared estimates across settings. Among the four models that answered image prompts, the same necklace was priced 1.9 to 3.3 times higher in the formal photograph than in the flat-lay (all p < 0.01). Against the yard photograph, the ratio ranged from 1.23 (Grok 4.5, not significant) to 3.57 (Kimi K3), with Claude Fable 5 at 3.32. A text-only version of the task, in which the setting is described in words, produced formal-to-yard ratios of 2.2 to 3.9 in all six models, so the effect does not require pixels. Flat-lay estimates themselves were 7.8 to 18.7 times the re
    
[^96]: HARP-LoRA：面向科学表格理解的层次条件化秩投影低秩自适应方法

    HARP-LoRA: Hierarchy-Conditioned Rank Projection for Low-Rank Adaptation in Scientific Table Understanding

    [https://arxiv.org/abs/2610.34589](https://arxiv.org/abs/2610.34589)

    该论文提出HARP-LoRA方法，利用表格行表头和列表头的层次结构为每个数据标记构建条件化表示并作用于低秩下投影，从而在参数高效的微调中提升科学表格（尤其是药代动力学表格）的理解能力。

    

    表格数据广泛应用于科学领域，其中复杂的版式布局——包括多级表头、合并单元格、标题和脚注——使得信息提取变得困难。这在药代动力学（PK）中尤为重要，因为其中的数值必须与药物、物种、剂量、给药途径、单位和实验条件正确关联。LoRA（低秩自适应）等参数高效微调方法虽然降低了适配成本，但并未显式地融合表格数值与其周围表头之间的关系。我们提出了HARP-LoRA，这是一种用于科学表格理解的层次条件化秩投影低秩自适应方法。HARP-LoRA通过从每个数据标记所对应的行表头和列表头层次结构中构建标记特定的内在表示，对投影式LoRA进行了扩展。这些表示对低秩下投影进行条件化，从而在保持参数高效的同时融入层次化上下文信息。

    arXiv:2610.34589v1 Announce Type: cross  Abstract: Tabular data is widely used in scientific applications, where complex layouts, including multi-level headers, merged cells, captions, and footnotes, make information extraction difficult. This is particularly important in pharmacokinetics (PK), where values must be correctly associated with the drug, species, dose, administration route, unit, and experimental condition. Parameter-efficient fine-tuning methods such as Low-Rank Adaptation (LoRA) reduce adaptation cost but do not explicitly incorporate relationships between table values and surrounding headers. We introduce HARP-LoRA, a hierarchy-conditioned rank projection method for low-rank adaptation in scientific table understanding. HARP-LoRA extends projected LoRA by constructing token-specific intrinsic representations from the row- and column-header hierarchies of each data token. These representations condition the low-rank down-projection, incorporating hierarchical context whi
    
[^97]: 大语言模型：面向工程师的通俗概述

    Large Language Models: A Plain-Language Overview for Engineers

    [https://arxiv.org/abs/2610.34585](https://arxiv.org/abs/2610.34585)

    本文以通俗语言为工程师系统梳理了大语言模型从Transformer、预训练、微调到RAG、思维链、智能体及FlashAttention、QLoRA等核心技术的原理与局限。

    

    大语言模型（LLM）是在海量文本集合上训练的神经网络，通过预测下一个词来生成文本。使用ChatGPT及其他生成式人工智能（AI）工具的工程师需要清楚地了解训练改变了什么、提示词又带来了什么。本文从Transformer（在上下文中连接词语的架构）和预训练（模型学习预测文本）讲起，接着解释微调如何使模型适应各类请求，以及利用人类答案排序进行的基于人类反馈的强化学习。随后介绍了检索增强生成（RAG）为模型提供文档，思维链提示鼓励模型进行中间推理步骤，以及智能体使用工具。FlashAttention通过减少注意力计算中的数据搬运来降低训练与运行成本；量化低秩适配（QLoRA）则节省微调所需的内存。读者还将获得一张生命周期图，以及比较各种方法及其局限性的表格，并附有一份用于评判的简短清单。

    arXiv:2610.34585v1 Announce Type: cross  Abstract: Large language models (LLMs) are neural networks trained on huge text collections to generate text by predicting what comes next. Engineers using ChatGPT and other generative artificial intelligence (AI) tools need a clear picture of what training changes and prompts add. The article starts with the Transformer, which connects words in context, and pre-training, where models learn to predict text. It then explains how fine-tuning adapts models to requests, with reinforcement learning from human feedback using human answer rankings. Next, retrieval-augmented generation (RAG) supplies documents, while chain-of-thought prompting encourages intermediate steps and agents use tools. FlashAttention lowers training and running costs by reducing data movement during attention; quantized low-rank adaptation (QLoRA) saves fine-tuning memory. Readers get a life cycle figure and tables comparing methods and their limits. A short checklist for judgi
    
[^98]: RefinedGAW：一个精炼的德国学术网络数据集

    RefinedGAW: A Refined German Academic Web Dataset

    [https://arxiv.org/abs/2610.34572](https://arxiv.org/abs/2610.34572)

    RefinedGAW是一个基于26次德国学术网络爬取构建的德语数据集，包含1.22亿份文档和3140亿词元，其中876亿学术词元超越了规模五倍于它的FineWeb2与FinePDFs德语子集，并为教育、数学和专家导向内容提供了无商业偏见的互补覆盖。

    

    arXiv:2610.34572v1 公告类型：新论文 摘要：最大的开放德语文本数据集通常构建自广泛的网络爬取，这类爬取提供了规模优势，但对特定领域的覆盖有限。我们推出了RefinedGAW，这是一个基于26次半年度德国学术网络爬取经过处理和标注的资源，总计1.22亿份文档和3140亿个词元，并保留了每篇文本完整的爬取和捕获历史。在HTML和PDF格式中，被标注为学术研究的文档占德语RefinedGAW词元的84%。其876亿个词元超过了FineWeb2和FinePDFs德语子集的649亿词元，尽管后者的总规模几乎是前者的五倍。RefinedGAW还包含大量教育性、数学密集型和面向专家的内容，且商业偏见极小。在HTML和PDF格式中，88%的德语RefinedGAW文档在三个被评估的参考数据集中没有近似重复的匹配，这支持了其互补性覆盖。我们发布了一个查询（接口）……

    arXiv:2610.34572v1 Announce Type: new  Abstract: The largest open German text datasets are built from broad web crawls, which provide scale but limited coverage of specific domains. We introduce RefinedGAW, a processed and annotated resource derived from 26 semi-annual crawls of the German Academic Web, totaling 122 million documents and 314 billion tokens, with full crawl and capture history of every text. Documents labeled as academic research account for 84% of German RefinedGAW tokens across HTML and PDF. At 87.6 billion tokens, this exceeds the 64.9 billion tokens in the German subsets of FineWeb2 and FinePDFs, despite their almost five-times-larger combined size. RefinedGAW also contains substantial amounts of educational, math-heavy, and expert-oriented content with minimal commercial bias. Across HTML and PDF, 88% of German RefinedGAW documents have no near-duplicate candidate match in the three evaluated reference datasets, supporting complementary coverage. We release a query
    
[^99]: STEP：通过逐步概率评分进行预测正确性的软轨迹估计

    STEP: Soft Trajectory Estimation of Prediction Correctness via Per-Step Probability Scoring

    [https://arxiv.org/abs/2610.34537](https://arxiv.org/abs/2610.34537)

    提出无需训练的STEP框架，通过对LLM推理的每个中间步骤进行概率评分构建软轨迹特征，在无法获取真实答案的情况下更准确地估计模型预测的正确性。

    

    在无法获取真实标签（ground truth）的情况下估计大型语言模型（LLM）预测的正确性，对于在高风险场景中的可靠部署至关重要。现有方法（如熵和困惑度）仅依赖最终输出的统计信息，丢弃了中间推理步骤中编码的丰富信息。为了解决这一局限性，我们提出了STEP（预测正确性的软轨迹估计），这是一个无需训练的框架，通过轻量级前向传递对每个推理步骤进行评分，从而获得候选答案上的软概率分布qi，并从中推导出一组软轨迹特征，捕捉每一步的置信度及其向最终答案演进的过程。在各种基准测试上的大量实验表明，相较于基于输出的基线方法和硬指标轨迹方法，该方法取得了一致且显著的改进，尤其是在AUROC指标上。值得注意的是，时间加权一致性（Temporally-Weighted Consistency），其……（摘要在此处被截断）

    arXiv:2610.34537v1 Announce Type: new  Abstract: Estimating the correctness of large language model (LLM) predictions without access to ground truth is critical for reliable deployment in high-stakes settings. Existing approaches, such as entropy and perplexity, operate solely on final output statistics, discarding the rich information encoded in intermediate reasoning steps. To address this limitation, we propose STEP (Soft Trajectory Estimation of Prediction correctness), a training-free framework that scores each reasoning step via lightweight forward passes to obtain a soft probability distribution qi over candidate answers, from which we derive a set of soft trajectory features capturing per-step confidence and its progression toward the final answer. Extensive experiments on various benchmarks demonstrate consistent and substantial improvements, particularly in AUROC, over output-based baselines and hard-indicator trajectory methods. Notably, Temporally-Weighted Consistency, whic
    
[^100]: 主动记忆召回中规模的代价

    The Price of Scale in Proactive Memory Recall

    [https://arxiv.org/abs/2610.34507](https://arxiv.org/abs/2610.34507)

    该论文证明主动记忆召回的门控决策本质上是一个随记忆库规模 N 增长的多重假设检验问题——单条记忆的假阳性率按 -ln(1-α)/N 衰减，导致命中率随 N 呈幂次下降，从理论与实证上解释了记忆库越大、主动记忆触发越容易失效的现象。

    

    一个无需被询问便能主动提起记忆的长期助手，必须在每轮交互中判断其记忆库中是否有内容值得打断用户。我们证明这一决策本质上是一个多重假设检验问题，其难度随记忆库规模 N 的增长而加剧。在 LongMemEval 与 LoCoMo 的流式回放实验中，一个在 N=8 时调校至 5% 误报率的相似度阈值，在 N=2048 时误报率飙升至 87–100%，该现象在各类嵌入模型以及显式与隐式线索中均成立；LLM 判别器虽起点更低，也呈现同样的增长趋势。我们证明，一个误报率为 α 的门控机制，其单条记忆的假阳性率约为 -ln(1-α)/N，这将把其命中率锁定在单次比较 ROC 曲线的最左端，且在正则变化条件下，命中率随 N 以幂次规律衰减并带有显式常数。由此得出的预测与逐 N oracle 阈值的命中率相差在一个百分点以内。该分析解释了为何词汇与语义通道……（原文摘要在此处截断）

    arXiv:2610.34507v1 Announce Type: cross  Abstract: A long-lived assistant that brings up memories without being asked must decide at every turn whether anything in its store is worth an interruption. We show that this decision is a multiple-testing problem whose difficulty grows with the store size $N$. On streaming replays of LongMemEval and LoCoMo, a similarity threshold tuned to 5% false alarms at $N=8$ reaches 87-100% at $N=2048$, across embedding models and for explicit and implicit cues; LLM judges show the same growth from a lower base. We prove that a gate with false-alarm rate $\alpha$ operates at a per-item false-positive rate of about $-\ln(1-\alpha)/N$, which pins its hit rate to the far left of the single-comparison ROC curve and, under regular variation, makes it decay as a power of $N$ with an explicit constant. The resulting prediction matches the hit rate of a per-$N$ oracle threshold to within one percentage point. The analysis explains why lexical and semantic channe
    
[^101]: 伏尼契手稿有意义吗？校准测试几乎未发现语言的痕迹

    Does the Voynich Manuscript Mean Anything? Calibrated Tests Find Little Trace of Language

    [https://arxiv.org/abs/2610.34499](https://arxiv.org/abs/2610.34499)

    通过将伏尼契手稿的两种起源假说写成程序并进行校准检验，研究发现抄写员复制假说比单密钥密码更能预测文本统计特征，且手稿中的词对同行模式强度仅为真实语言文本的一半，表明手稿几乎不含真实语言的痕迹。

    

    伏尼契手稿是一本十五世纪的书，其文字无人能读懂。一种说法认为其文本毫无意义，是一位抄写员通过复制自己的话语并稍作改动而造出来的；另一种说法则认为一种真实的语言隐藏在密码背后。我们将这两种说法分别写成能够填充手稿页面的程序，用相同的预算拟合每个程序，并在拟合过程中从未见过的统计数据上进行检验。Timm和Schinner提出的复制过程对统计数据的预测效果远好于任何单密钥密码。但在手稿的两种方言之间切换密钥的密码，或仅对真实文本词类进行编码的密码，同样能很好地预测这些统计数据——仅凭词汇统计无法做出判断。我们进而寻找一种无需破译即可识别的语言痕迹：在16种语言的28篇文本中，相同的词对会持续共享同一行，而手稿仅以这些文本中最弱者一半的强度呈现这种模式。

    arXiv:2610.34499v1 Announce Type: new  Abstract: The Voynich manuscript is a fifteenth-century book in a script that no one can read. One account holds that its text means nothing and that a scribe made it by copying his own words and changing them a little. The other holds that a real language lies hidden behind a cipher. We write each account as a program that fills the pages of the manuscript, fit every program with the same budget, and check it on statistics the fit never saw. The copying process of Timm and Schinner predicts these statistics far better than any cipher with one key. But ciphers that switch keys between the manuscript's two dialects, or that encode only the word classes of a real text, predict them equally well. Word statistics alone cannot decide. We look for a trace of language that needs no decipherment. In 28 texts in 16 languages the same word pairs keep sharing a line. The manuscript shows this pattern at half the strength of the weakest of these texts. A word
    
[^102]: 波即所需：基于全连接量子玻尔兹曼机的量子语言模型

    Wave Is All You Need: A Quantum Language Model Based on Fully Connected Quantum Boltzmann Machine

    [https://arxiv.org/abs/2610.34492](https://arxiv.org/abs/2610.34492)

    该论文提出基于全连接量子玻尔兹曼机的自注意力机制并据此构建量子语言模型，通过对语义量子态的酉演化隐式实现位置编码与自注意力计算，在数学上与经典自注意力机制等价，从而免除了显式矩阵乘法的需求。

    

    本工作提出了一种基于全连接量子玻尔兹曼机（FC-QBM）的自注意力机制，并进一步以该机制为核心引擎构建了全连接量子玻尔兹曼机语言模型（FC-QBM-LM）。通过对语义量子态执行酉演化，该机制隐式地实现了位置编码与自注意力计算，从而无需将连续语义流分解为离散矩阵进行显式矩阵乘法，且测量结果自然满足归一化条件。具体而言，本工作严格推导了基于FC-QBM的自注意力机制的计算原理，并证明该机制在数学上与经典自注意力机制等价。本文还深入推导了FC-QBM-LM训练与生成过程所遵循的数学原理，以及两种相应的（算法与实现）。

    arXiv:2610.34492v1 Announce Type: cross  Abstract: This work proposes a self-attention mechanism based on the fully connected quantum Boltzmann machine (FC-QBM), and further constructs a fully connected quantum Boltzmann machine language model (FC-QBM-LM) using the mechanism as the core engine. By performing unitary evolution on semantic quantum states, the mechanism implicitly implements positional encoding and self-attention computation, eliminating the need to decompose continuous semantic streams into discrete matrices for explicit matrix multiplication, with the measurement results naturally satisfying the normalization condition. Specifically, this work rigorously derives the computational principle of the self-attention mechanism based on FC-QBM, and demonstrates that this mechanism is mathematically equivalent to the classical self-attention mechanism. The mathematical principles governing the training and generation of FC-QBM-LM are thoroughly derived, and the two correspondin
    
[^103]: 我该选择哪种努力程度？论大语言模型智能体中推理努力的经济学

    What Effort Level Should I Choose? On the Economics of Reasoning Effort in LLM Agents

    [https://arxiv.org/abs/2610.34491](https://arxiv.org/abs/2610.34491)

    该论文首次将大语言模型的“推理努力”刻画为一种“信任品”，通过在四个基准上量化不同努力程度的反事实效应，发现提高努力程度虽能总体提升准确率，却经常无法解决低努力已能解出的任务、可能造成计算成本超支并改变智能体的行为轨迹，为推理努力的选择提供了经济学视角的实证依据。

    

    生成式模型越来越多地提供一种“推理努力”设置，允许用户调节推理时的计算量，以更高的成本换取可能提升的性能。用户必须在尚不知道额外推理是否有帮助的情况下选择努力程度，而且即使一次成功的运行也无法揭示所购买的努力是否必要。因此，推理努力是一种“信任品”：买方只能观察到结果，却无法评估其所支付计算带来的边际价值。在这项工作中，我们在四个基准上测量了努力程度对专有模型和开放权重模型的反事实影响，这些基准涵盖单次调用推理任务和长时程智能体任务。研究结果表明，努力程度并不能稳定保留低努力程度所能解决的任务，可能导致过度支出，并可能改变轨迹行为。首先，我们发现尽管更高的努力程度在总体上可以提高准确率，但它在低努力程度所能解决的任务上却经常失败。其次……

    arXiv:2610.34491v1 Announce Type: new  Abstract: Generative models increasingly expose a reasoning effort setting that lets users scale inference-time compute, trading higher cost for potentially improved performance. Users must choose an effort level before knowing whether extra reasoning will help, and even a successful run does not reveal whether the effort purchased was necessary. Reasoning effort is therefore a credence good: buyers observe the outcome but cannot assess the marginal value of the computation they paid for. In this work, we measure the counterfactual effect of effort levels on both proprietary and open-weight models on four benchmarks, spanning single-call reasoning and long-horizon agentic tasks. Effort, it turns out, does not consistently preserve tasks that lower effort solves, can overspend, and can change trajectory behavior. First, we find that although higher effort can raise accuracy in aggregate, it frequently fails on tasks that lower effort solved. Second
    
[^104]: 面向具有多个候选答案的Lean语句的高效证明生成

    Efficient Proof Generation for Lean Statements with Multiple Candidate Answers

    [https://arxiv.org/abs/2610.34490](https://arxiv.org/abs/2610.34490)

    该论文提出一种联合证明框架，由单个训练好的证明器在单次调用中同时完成候选答案的选择或补充以及形式化证明的生成，避免了为每个候选答案分别搜索证明的冗余计算开销，并能复用已生成的证明来检测语句的忠实性错误。

    

    在Lean 4等形式系统中，高效地求解数学问题答案并生成可验证的证明是自动数学推理中的一项重要任务。在传统方式下，上游求解器提供多个候选答案，常见的工作流程需要对每个候选答案分别搜索证明，从而产生冗余的计算开销。为此，我们提出了一种联合证明框架，将候选答案选择、答案补充与形式化证明生成集成到统一的推理过程中。其核心是一个经过训练的证明器，它能够在单次调用中选择给定答案或提出新答案，并至少为其中一个答案生成证明。联合证明还可以复用已生成的证明来检查忠实性错误。对于具有唯一答案的问题，如果两个不同的答案在同一形式化语句下都获得了可验证的证明，则表明该语句未能忠实地刻画原问题。

    arXiv:2610.34490v1 Announce Type: cross  Abstract: In formal systems such as Lean 4, efficiently finding answers to mathematical problems and generating verifiable proofs is an important task in automated mathematical reasoning. In the traditional way where an upstream solver provides multiple candidate answers, the common workflow requires searching for proofs for each candidate separately, incurring redundant computational cost. To this end, we propose a joint proof framework that integrates candidate answer selection, answer supplementation, and formal proof generation into a unified reasoning process. At its core is a trained prover that selects given or proposes new answers and generates a proof for at least one answer in a single call. Joint proof can also reuse generated proofs to check for faithfulness errors. For problems with a unique answer, if two distinct answers both obtain verifiable proofs under the same formal statement, this confirms the statement does not faithfully 
    
[^105]: 迈向神经自适应通信接口：一个个性化语音与语言支持框架

    Toward Neuroadaptive Communication Interfaces: A Framework for Personalized Speech and Language Support

    [https://arxiv.org/abs/2610.34457](https://arxiv.org/abs/2610.34457)

    该论文提出了一个将语言呈现方式个性化的自适应框架，以行为反馈为主要依据、脑电信号为不确定性感知的辅助信息源，通过句子分段、象形图和语音输出等支持手段，比较固定式、行为自适应与神经自适应接口，旨在提升不同能力用户的理解与沟通效率。

    

    语音和语言技术扩大了人们获取通信的途径，但它们所提供的支持往往是固定的。同样的语言或视觉辅助可能对一位用户有帮助，却对另一位用户造成阻碍，或者随着任务需求的变化而变得不再有用。本项目提出了一个框架，用于使语言呈现方式适应个体能力和持续进行的交互。该框架将消息的获取、用户需求的估计以及支持方式的选择分离开来，支持方式包括句子分段、象形图和可选的语音输出。行为反馈为自适应提供主要依据；脑电图（EEG）被视为一种额外的、具有不确定性感知的信息来源，而非对理解程度的直接测量。我们制定了分阶段的评估方案，对固定式、行为自适应和神经自适应接口进行比较。核心问题在于：神经信息能否在强大的行为自适应基线之上进一步提升理解能力与沟通效率。

    arXiv:2610.34457v1 Announce Type: cross  Abstract: Speech and language technologies have expanded access to communication, but the support they provide is often fixed. The same linguistic or visual aid may help one user and hinder another, or become less useful as task demands change. This project proposes a framework for adapting language presentation to individual abilities and ongoing interaction. It separates access to a message from the estimation of user needs and from the selection of support, including sentence segmentation, pictograms, and optional spoken output. Behavioral feedback provides the primary evidence for adaptation; electroencephalography is considered an additional, uncertainty-aware source of information rather than a direct measure of comprehension. We specify a staged evaluation comparing fixed, behavior-adaptive, and neuroadaptive interfaces. The central question is whether neural information improves comprehension and communication efficiency beyond a strong 
    
[^106]: 面向超导领域的arXiv文献结构化抽取与检索增强问答流水线

    A Domain-Aware Pipeline for Structured Extraction and Retrieval-Augmented Question Answering over arXiv Superconductivity Literature

    [https://arxiv.org/abs/2610.34391](https://arxiv.org/abs/2610.34391)

    本文提出了一个面向超导领域的端到端流水线，通过结构化抽取材料与临界温度信息，并将领域属性融入混合检索，从而提升了arXiv超导文献检索增强问答的准确性与可靠性。

    

    材料物理学领域的科学文献增长速度已超过研究人员人工阅读的能力，然而大多数检索工具仍依赖关键词匹配或通用问答方法，忽略了PDF文件中蕴含的领域特定结构信息。本文提出了一个端到端的流水线，将arXiv超导领域论文转化为带有结构化元数据的可检索语料库，并提供检索增强问答能力。该系统结合了主题语料库构建、面向领域的材料及超导转变温度抽取，以及将稠密文本块嵌入与结构化论文字段相融合的混合检索方法。我们验证了相对于仅使用词法匹配和仅使用稠密文本的方法，归一化的领域属性能否提升证据检索质量和答案的有据性。抽取效果在一个经过人工验证、包含59种超导材料的黄金标准数据集上进行了评估，其中46种材料具有已确认的实验临界温度。

    arXiv:2610.34391v1 Announce Type: cross  Abstract: Scientific literature in materials physics is growing faster than researchers can manually inspect it, yet most retrieval tools rely on keyword matching or generic question answering that ignores domain-specific structure in PDF files. This paper presents an end-to-end pipeline that transforms arXiv superconductivity publications into a searchable corpus with structured metadata and retrieval-augmented question answering. The system combines thematic corpus construction, domain-aware extraction of materials and superconducting transition temperatures, and hybrid retrieval that merges dense chunk embeddings with structured publication fields. We test whether normalized domain attributes improve evidence retrieval and answer groundedness relative to lexical-only and dense text-only methods. Extraction is evaluated on a manually verified gold set of 59 superconducting materials, 46 of which have confirmed experimental critical temperature
    
[^107]: AndroidFlux：评估移动使用多模态智能体的故障恢复能力并进行奖励建模

    AndroidFlux: Evaluating and Reward Modeling Failure Recovery Capabilities in Mobile Use Multimodal Agents

    [https://arxiv.org/abs/2610.34384](https://arxiv.org/abs/2610.34384)

    提出了基于AndroidWorld的AndroidFlux故障恢复基准，通过在首个可恢复错误前、中、后设置受控检查点，使不同智能体从相同状态继续执行以评估其错误恢复能力，并配套奖励建模基准来评估恢复动作的偏好。

    

    移动使用多模态智能体通常通过从干净的初始状态出发的任务成功率来评估，这种方法无法单独衡量它们从中间错误中恢复的能力。由于移动环境是动态变化的，一个通用的移动使用智能体必须能够在意外状态下以及未见过的环境中稳健地完成任务。这就需要了解智能体在何时何地犯错，以及它能否从错误及其后续状态中恢复。为了评估这一能力，我们提出了AndroidFlux，一个基于AndroidWorld构建的故障恢复基准。AndroidFlux将记录的轨迹恢复到第一个可恢复错误之前、之时和之后的受控检查点，使不同的智能体从相同的设备状态和历史记录继续执行。在这些检查点上，一个配套的奖励建模基准利用下游部署结果来评估恢复动作的偏好。我们进一步发布了离线强化学习（RL）……（摘要原文在此处截断）

    arXiv:2610.34384v1 Announce Type: cross  Abstract: Mobile-use multimodal agents are typically evaluated by task success from a clean initial state, which does not isolate their ability to recover from intermediate errors. Because mobile environments change dynamically, a general mobile-use agent must complete tasks robustly even from unexpected states and in unseen environments. This requires knowing when and where an agent makes errors, and whether it can recover from errors and subsequent states. To evaluate this capability, we introduce AndroidFlux, a failure-recovery benchmark built on AndroidWorld. AndroidFlux restores recorded trajectories to controlled checkpoints before, at, and after the first recoverable error, so that different agents continue from the same device state and history. On these checkpoints, a complementary reward-modeling benchmark evaluates recovery-action preferences using downstream rollout outcomes. We further release an offline reinforcement learning (RL) 
    
[^108]: 是否与何处？尾部校准预示Best-of-N选择中的有限预算奖励欺骗

    Whether or Where? Tail Calibration Signals Finite-Budget Reward Hacking in Best-of-N Selection

    [https://arxiv.org/abs/2610.34375](https://arxiv.org/abs/2610.34375)

    本文证明Best-of-N的精确值只依赖奖励分数的秩，其是否以及在多大预算下发生性能下降（奖励欺骗）完全由奖励误差的尾部校准决定——高斯误差持续提升、拉普拉斯误差趋于平台、重尾误差回落至先验均值，因此尾部校准曲线的转折点即可预示有限预算下的奖励欺骗。

    

    Best-of-N（BoN）推断返回代理奖励最高的候选，因此更大的N更多考验的是代理奖励的极端尾部，而非其平均排序质量。我们证明，BoN的精确值是条件真实奖励在提示内分数分位数上的Beta加权积分。由此可知，BoN仅通过分数的秩（排序）依赖于分数；由此还可推知，只有当尾部的校准曲线向下转折时，BoN才会下降，且下降出现的预算规模取决于该转折点的位置。在有界加性误差模型下，误差尾部决定了极限：高斯误差下BoN持续提升，拉普拉斯误差下达到平台，重尾误差下则回落至先验均值。当误差方差在低真实奖励处最大时，即使是高斯噪声也能使BoN低于随机选择。合成实验的相图与这一分类完全一致。在涵盖多个可验证领域、基础策略与学习型奖励模型的真实候选池上……（原文摘要至此截断）

    arXiv:2610.34375v1 Announce Type: cross  Abstract: Best-of-N (BoN) inference returns the candidate with the highest proxy reward, so a larger $N$ tests the proxy's extreme tail more than its average ranking quality. We show that the exact BoN value is a Beta-weighted integral of conditional true reward over within-prompt score quantiles. It follows that BoN depends on scores only through their ranks. It also follows that BoN can decline only if the calibration curve in the tail turns downward, and the budget where the decline appears depends on where that turn lies. Under a bounded additive model, the error tail sets the limit: Gaussian errors keep improving, Laplace errors reach a plateau, and heavy-tailed errors return to the prior mean. When error variance is largest at low true reward, even Gaussian noise can push BoN below random selection. Synthetic phase maps agree with this taxonomy. On real candidate pools spanning several verifiable domains, base policies, and learned reward 
    
[^109]: 超越开头：追踪首个Token对推理长度的持续影响

    Beyond the Opening: Tracing the First Token's Persistent Influence on Reasoning Length

    [https://arxiv.org/abs/2610.34350](https://arxiv.org/abs/2610.34350)

    该研究揭示了推理模型中首Token对推理长度具有持续性影响——仅替换首Token即可显著改变后续推理长度，且这种影响集中在少数关键生成位置上，阻断这些位置对首Token的注意力可缩小长度差距。

    

    理解单个Token的选择如何影响推理长度，对于刻画推理模型的特性至关重要。基于先前关于“改变第一个推理Token（以下简称首Token）能够显著改变推理长度”的发现，我们聚焦于这一初始选择，研究其影响如何在后续生成过程中发挥作用。一种可能性是首Token改变了其后紧随的早期文本，进而影响推理长度。然而，实验表明，仅替换现有推理前缀的首Token并继续生成，就会大幅改变剩余的推理长度，即使前缀中的所有其他Token均保持不变。为了追踪这种持续影响，我们在仅改变首Token的条件下比较下一Token的分布，发现这种敏感性集中在少数生成位置上。阻断这些位置对首Token的注意力可以缩小长度差距……

    arXiv:2610.34350v1 Announce Type: new  Abstract: Understanding how individual token choices affect reasoning length is important for characterizing reasoning models. Building on prior findings that changing the first reasoning token (hereafter, the first token) can substantially alter reasoning length, we focus on this initial choice to investigate how its influence operates during subsequent generation. One possibility is that the first token changes the early text that follows, which in turn affects reasoning length. However, replacing only the first token of an existing reasoning prefix and resuming generation substantially changes the remaining reasoning length, even though all other prefix tokens are unchanged. To trace this persistent influence, we compare next-token distributions while varying only the first token and find that sensitivity is concentrated at a small set of generation positions. Blocking attention from these positions to the first token reduces the length gap bet
    
[^110]: Xing USP：面向信息抽取与意图-槽位解析的跨家族语义契约

    Xing USP: A Cross-Family Semantic Contract for Information Extraction and Intent-Slot Parsing

    [https://arxiv.org/abs/2610.34343](https://arxiv.org/abs/2610.34343)

    本文提出USP跨家族语义契约，通过统一接口将命名实体识别、关系抽取、事件抽取和意图-槽位解析整合表示，并将61个数据集规范化为统一语料库，借助五种诊断匹配视图超越单纯类别判定来评估值恢复与实例级关联能力。

    

    结构化语义解析通常以是否分配了正确的类别来评判，这使得值恢复与实例级关联未被检验：一个系统可能识别出了事件，却将参数绑定到错误的实例；或者识别出了意图，却提供了错误的槽位值。我们提出了USP，一个跨家族语义契约，它通过统一接口表示类别、字段、值和实例关联，涵盖命名实体识别、关系抽取、事件抽取和意图-槽位解析四类任务。该契约配套五种诊断性匹配视图——类别一致性、字段识别、字段-值配对、实例绑定赋值和完整实例恢复——它们针对同一输出提出不同的问题。利用该契约，我们将61个数据集规范化为具有共享模式、提示和匹配规则的统一语料库，并在固定测试子集上比较了六个模型。

    arXiv:2610.34343v1 Announce Type: cross  Abstract: Structured semantic parsing is typically judged by whether the correct category is assigned, which leaves value recovery and instance-level associations unexamined: a system may recognize an event yet bind an argument to the wrong instance, or identify an intent yet supply an incorrect slot value. We present USP, a cross-family semantic contract that represents categories, fields, values, and instance associations through a common interface for named-entity recognition, relation extraction, event extraction, and intent-slot parsing. The contract is paired with five diagnostic matching views---category agreement, field recognition, field-value pairing, instance-bound assignment, and complete-instance recovery---that ask distinct questions about the same output. Using this contract, we normalize 61 datasets into a unified corpus with shared schemas, prompts, and matching rules, and compare six models on a fixed test subset. Our fine-tune
    
[^111]: 通过长-短对比剪枝提升剪枝后推理模型的思维链可压缩性

    Improving the Chain-of-Thought Compressibility of Pruned Reasoning Models via Long--Short Contrastive Pruning

    [https://arxiv.org/abs/2610.34336](https://arxiv.org/abs/2610.34336)

    提出对比集合差分方法，通过对比同一问题的长、短正确解识别并优先剪枝冗余推理神经元，并将偏好对扩展到剪枝模型总是或从不解决的问题，从而提升剪枝推理模型的思维链可压缩性。

    

    大型推理模型的服务成本高昂，一方面是因为模型规模庞大，另一方面是因为它们会生成冗长的思维链（CoT）。权重剪枝和思维链压缩分别针对这两种成本，但二者会相互干扰：剪枝会导致模型在仍能解决的问题上产生冗余推理，在不再能解决的问题上产生不完整的推理，并且在更难的问题上，会降低模型对后续长度偏好优化的可压缩性。我们将这一现象追溯到双方各自存在的盲点：剪枝标准无法区分必要推理与冗余推理，而基于偏好的思维链压缩在剪枝模型总是解决或从不解决的问题上收不到任何训练信号。我们提出对比集合差分方法，通过对比同一问题的长、短正确解上神经元的激活情况，识别出富含冗余推理的神经元并优先对其进行剪枝，同时我们将偏好对扩展到剪枝模型总是或从不解决的问题上（摘要原文在此处截断）。

    arXiv:2610.34336v1 Announce Type: cross  Abstract: Large reasoning models are costly to serve because they are large and because they generate long chains of thought (CoT). Weight pruning and CoT compression address these two costs separately, but they interfere in that pruning adds redundant reasoning on problems the model still solves and incomplete reasoning on those it no longer solves, and on harder problems makes the model less compressible by subsequent length-preference optimization. We trace this to a blind spot on each side: pruning criteria do not distinguish necessary from redundant reasoning, and preference-based CoT compression receives no training signal on problems a pruned model always or never solves. We propose Contrastive Set-Difference (CSD), which identifies neurons enriched in redundant reasoning by contrasting their activations on long and short correct solutions of the same problem and prunes them preferentially, and we extend preference pairs to the problems a
    
[^112]: CoLink：面向多轮对话建模的全局压缩与知识链接

    CoLink: Global Compression and Knowledge Linking for Multi-Turn Dialogue Modeling

    [https://arxiv.org/abs/2610.34335](https://arxiv.org/abs/2610.34335)

    提出CoLink框架，通过将对话历史聚类为主题图并协同语义相关性与图结构拓扑进行知识检索激活，实现信息链接与全局压缩，从而增强LLM在领域特定多轮对话中的全局上下文感知能力。

    

    受限于受限的上下文窗口和高效检索的挑战，从长程对话历史中提取显著信息对大型语言模型（LLMs）而言仍是一项重大难题。当前方法通常依赖于历史组织或局部token级优化，存在信息丢失、检索冗余和短视注意力等问题。受认知科学中“对话构成意义建构与记忆联想”这一观点的启发，我们提出了CoLink，这是一个通过信息链接与压缩来增强LLMs在领域特定多轮对话中全局上下文感知能力的框架。CoLink首先将历史上下文聚类为主题图，以建立与外部知识的跨语义关联；随后在思维自适应控制（adaptive control of thought）的指导下，通过协同语义相关性与图结构拓扑来检索并激活知识。至关重要的是，为了实现有效的……

    arXiv:2610.34335v1 Announce Type: cross  Abstract: Extracting salient information from long-horizon dialogue histories remains a significant challenge for Large Language Models (LLMs) due to the restricted context windows and efficient retrieval challenges. Current methods typically rely on history organization or local token-level optimization, suffering from information loss, retrieval redundancy, and myopic attention. Inspired by the view in cognitive science that dialogue constitutes sense-making and memory association, we propose CoLink, a framework that enhances LLMs' global context awareness in domain-specific multi-turn dialogues through information linking and compression. CoLink first clusters historical contexts into topic graphs to establish cross-semantic associations with external knowledge. Guided by the adaptive control of thought, it then retrieves and activates knowledge via synergizing semantic relevance and graph-structural topology. Crucially, to achieve effective 
    
[^113]: 生、死与扩散：基于插入的带纠正机制序列生成

    Birth, Death and Diff: Insertion-Based Sequence Generation with Corrections

    [https://arxiv.org/abs/2610.34310](https://arxiv.org/abs/2610.34310)

    提出 δMDM 模型，通过为基于插入的掩码扩散模型引入替换与删除操作使中间生成结果可被修正，从而支持更激进的并行解码并减少推理所需的前向传播次数。

    

    基于插入的掩码扩散模型通过交替进行掩码插入与词元去掩码，实现了灵活长度、并行的序列生成。然而，由于其生成决策是不可逆的，激进的并行解码可能引入后续步骤无法纠正的错误，因此需要采用更保守的解码方式并进行更多次前向传播（NFEs）。此外，由于这类模型在训练过程中对多种对齐方式进行边缘化，它们比标准掩码扩散模型（MDM）表现出更高的熵，因而在缺乏纠正机制的情况下需要更加保守的解码。为解决这一局限，我们提出了 δMDM，通过为基于插入的模型配备替换和删除操作，使中间生成结果可以被修改。在训练过程中，我们让模型接触由激进去掩码和插入所引起的错误，教会它纠正自身的错误。在推理时，我们交替进行插入和去掩码操作……

    arXiv:2610.34310v1 Announce Type: cross  Abstract: Insertion-based masked diffusion models enable flexible-length, parallel generation by interleaving mask insertion with token unmasking. However, because their generation decisions are irreversible, aggressive parallel decoding can introduce errors that subsequent steps cannot correct, necessitating conservative decoding with more forward passes (NFEs). Moreover, they exhibit higher entropy than standard MDMs because they marginalize over multiple alignments during training and therefore require even more conservative decoding in the absence of a correction mechanism. To address this limitation, we introduce ${\delta}$MDM, which makes intermediate generations revisable by equipping an insertion-based model with substitution and deletion operations. During training, we expose the model to errors caused by aggressive unmasking and insertion, teaching it to correct its own mistakes. At inference time, we interleave insertion and unmasking
    
[^114]: 利用语言模型与天文工具对巴比伦天文日志进行自动化、可审计的断代

    Automated, auditable dating of the Babylonian Astronomical Diaries with language models and astronomical tools

    [https://arxiv.org/abs/2610.34297](https://arxiv.org/abs/2610.34297)

    本文提出一个结合规则解析器、语言模型和天文计算工具的自动化系统，可对缺失日期的巴比伦天文日志泥板进行可审计、可追溯的天文断代。

    

    巴比伦天文日志记录了近六个世纪的天空观测。许多泥板残破不全，失去了其日期公式。专家们在标准版本《巴比伦天文日志及相关文本》（ADART）中逐一为许多泥板进行了断代，他们依据所记录的天象以及其他证据，如年份公式和国王名字。我们的系统以统一的规则为所有泥板自动化了这一天文断代过程。基于规则的解析器将记录的观测转化为子句，即具有固定字段的简短陈述。语言模型对照原文检查这些子句，并补充解析器遗漏的观测内容。天文工具在从公元前626年到公元75年的全部701年间逐一检验这些子句。系统根据子句与各年份天象的吻合程度对候选年份进行排序。系统在结果最终确定之前不采信已发表的日期，且输出中列出每个候选年份所依据的原文段落和计算出的天体位置。在120块已有日期的泥板上，（摘要在此处中断）

    arXiv:2610.34297v1 Announce Type: cross  Abstract: The Babylonian Astronomical Diaries record the sky over nearly six centuries. Many tablets are broken and have lost their date formula. Experts dated many of them one tablet at a time in the standard edition, Astronomical Diaries and Related Texts from Babylonia (ADART). They used the recorded sky and other evidence, such as year formulas and royal names. Our system automates the astronomical part for all tablets with the same rules. A rule-based parser turns the recorded observations into clauses, short statements with fixed fields. A language model checks them against the text and adds observations that the parser missed. Astronomical tools test the clauses in all 701 years from 626 BCE to AD 75. The system ranks the years by how well they agree with the clauses. The system withholds published dates until its results are final, and the output lists the passages and computed positions behind each candidate year. On 120 dated tablets, 
    
[^115]: 事实正确却立场偏颇：检索增强生成中的立场投毒与证据平衡

    Factual Yet One-Sided: Stance Poisoning and Evidence Balancing in Retrieval-Augmented Generation

    [https://arxiv.org/abs/2610.34294](https://arxiv.org/abs/2610.34294)

    本文揭示了一种新型立场投毒攻击——注入事实正确但立场片面的材料即可使检索增强生成系统的输出立场随之偏移，并提出一个无需额外大模型调用的立场平衡中间件，通过立场检测、固定配比的证据平衡与去偏提示词来消除这一偏见。

    

    检索增强生成系统回复中所表达的立场，显著受检索材料的立场分布影响。现有关于知识投毒的研究主要针对事实性错误，而本研究关注的是一种事实正确、但在争议议题上立场片面的注入攻击。实验表明，在注入后的 ProCon 文档中话题层面平均投毒占比达到 18.9% 时，来自不同厂商的四个生成模型均表现出与投毒方向一致的立场偏移；且在四种集中度水平下，偏移呈单调递增。本文提出了一种位于检索器与生成器之间的立场平衡中间件，由立场检测、固定平衡机制（2条支持 + 2条反对 + 1条中立）以及去偏提示词构成，且不引入额外的大语言模型调用。在全部八种配置下……（原文此处截断）

    arXiv:2610.34294v1 Announce Type: new  Abstract: The stance expressed in responses from retrieval-augmented generation systems is significantly influenced by the stance distribution of the retrieved materials. Existing research on knowledge poisoning has primarily targeted factual errors. In contrast, this study focuses on injection attacks in which the facts are correct but the stance on a controversial issue is one-sided. At a mean topic-level poisoning share of 18.9% within the post-injection ProCon documents, all four generation models from different vendors exhibited stance shifts aligned with the poisoning direction; across four concentration levels, the shifts increased monotonically.   This paper proposes a Stance-Balancing Middleware positioned between the retriever and the generator. It consists of stance detection, fixed balancing (2 supporting + 2 opposing + 1 neutral), and a debiasing prompt, without introducing additional large language model calls. Across all eight conce
    
[^116]: 从Transformer到GPT-5：大语言模型设计的进展、局限与未来方向

    Transformer to GPT-5: Advancements, Limitations, and Future Directions in Large Language Model Design

    [https://arxiv.org/abs/2610.34281](https://arxiv.org/abs/2610.34281)

    本文对从BERT到GPT、Gemini、Claude和Grok等大语言模型进行了系统性分类梳理，涵盖训练数据集、算力、硬件规格、架构与参数，并分析了其局限性、计算挑战与未来发展方向。

    

    大语言模型（LLMs）凭借其卓越的能力彻底改变了人工智能（AI）领域，这些能力包括上下文理解、少样本/零样本学习、代码生成、推理能力和问题解决能力。近期的研究显示，关于大语言模型已有大量的工作成果。然而，针对大语言模型的全面系统性分类的研究仍然有限。本研究将展示大语言模型的详细分类，从早期的模型（如BERT）开始，一直延伸到GPT、Gemini、Claude和Grok等大语言模型。我们将深入探讨诸如用于训练大语言模型的数据集、训练期间所使用的计算能力、硬件规格、架构以及参数等主题。此外，我们还将审视其局限性、计算挑战以及开放性问题。本研究将主要聚焦于大语言模型的发展时间线，以体现其在这段时期内的成长历程。研究结果表明存在一个清晰的模式：那些能够优化架构复杂组件之间比率的模型……

    arXiv:2610.34281v1 Announce Type: new  Abstract: Large Language Models (LLMs) have revolutionized Artificial Intelligence (AI) by their extraordinary capabilities, like contextual understanding, few-shot/zero-shot learning, code generation, ability to reason and problem-solving. Recent studies have shown immense work on LLMs. Yet, there is limited work on the thorough taxonomy of LLMs. This study would showcase a detailed classification of LLMs, starting with the earlier models like BERT and leading to LLMs like GPT, Gemini, Claude and Grok. We would delve into topics like datasets used to train LLMs, computational power used during training, their hardware specifications, architecture and parameters. Further, we would also look at the limitations, computational challenges and open problems as well. This study would primarily focus on the timeline of LLMs, signifying their growth over the period. Findings reveal a clear pattern: models that optimize the ratio between architectural comp
    
[^117]: 一个俄罗斯人、一个巴西人和一个德国人走进酒吧，问了同样的问题——却得到三种不同的回答：对大语言模型在争议话题上观点多样性的分析

    A Russian, a Brazilian, and a German Walk into a Bar, Ask the Same Question-- And Get Three Different Answers: Profiling LLM Perspective Diversity on Contested Topics

    [https://arxiv.org/abs/2610.34278](https://arxiv.org/abs/2610.34278)

    该论文构建了涵盖18种语言的争议话题多语言数据集PerspectivesQA和ROOTS分析框架，揭示出大语言模型在英语中回答含糊保守，但在低资源语言中会明确表态，且其观点多样性随模型规模呈系统性差异。

    

    大语言模型（LLMs）在具有可验证答案的任务上表现出强大的能力，但许多现实世界中的查询涉及主观性、分歧以及相互矛盾的证据。现有针对此类查询的基准测试大多以英语为中心，忽视了全球用户所持观点的巨大多样性。为了系统研究LLM的观点如何随语言变化，我们构建了PerspectivesQA——一个经过人工验证、涵盖18种语言的全球争议话题多语言数据集，并提出了ROOTS框架，用于刻画语言、观点和表述方式的多样性。我们评估了涵盖开源与专有系列、不同规模共八个模型，发现LLM在英语中倾向于含糊其辞，但在低资源语言中则会明确表态。更大的模型能保持更一致的跨语言世界观，而较小的模型则更容易被动摇。所有模型还表现出……（原文摘要在此处截断）

    arXiv:2610.34278v1 Announce Type: new  Abstract: Large Language Models(LLMs) have shown strong performance on tasks with verifiable answers, but many real-world queries involve subjectivity, disagreement, and conflicting evidence. Most existing benchmarks for such queries remain English-centric, overlooking the vast diversity of perspectives held by users worldwide. To systematically study how LLM perspectives vary across languages, we curate PerspectivesQA, a manually verified multilingual dataset of globally contested topics covering 18 languages, and propose ROOTS, a framework that profiles linguistic, perspective, and framing diversity. We evaluate a diverse set of eight models covering both open-source and proprietary families at varying scales, establishing that LLMs hedge their answers in English but commit to positions in lower-resource languages. Larger models maintain a more consistent cross-lingual worldview, while smaller models are more easily swayed. All models also exhib
    
[^118]: Δτ-Bench：推动智能体迈向动态变化的能力边界

    $\Delta\tau$-Bench: Pushing Agents Toward a Moving Capability

    [https://arxiv.org/abs/2610.34238](https://arxiv.org/abs/2610.34238)

    Δτ-Bench是一个模拟工具生态系统持续演化的可执行智能体基准测试，包含覆盖42个领域的1,588个多轮任务，评估显示最强模型的任务成功率也仅达42.3%，揭示了智能体在应对工具环境动态变化方面的重大不足。

    

    本文提出了Δτ-Bench，一个可执行的智能体基准测试，用于建模大型工具生态系统的结构演化。在Δτ-Bench中，各自独立演化的服务不断重塑任务的完成方式：熟悉的工具可能消失或被替换，而更高效的替代方案则会在功能重叠和充满噪声的文档中涌现。面对这一不断移动的能力边界，我们评估智能体能否主动发现高效的工具使用路径，并及时修订其积累的程序性知识，从而汇集了现有基准测试未能考察的多重需求。Δτ-Bench包含1,588个经过验证的多轮任务，覆盖42个领域，以及通过能力保持变换生成的六个连续的生态系统状态。我们的评估显示，即使是最强的被评估模型或框架，其任务成功率也仅为42.3%，而现有方法带来的初始收益……

    arXiv:2610.34238v1 Announce Type: cross  Abstract: In this paper, we introduce $\Delta\tau$-Bench, an executable agent benchmark that models the structural evolution of large tool ecosystems. In $\Delta\tau$-Bench, independently evolving services continually reshape how tasks can be accomplished: familiar tools may disappear or be replaced, while more efficient alternatives emerge amid overlapping functionality and noisy documentation. Against this moving capability frontier, we evaluate whether agents can proactively discover efficient tool-use paths and promptly revise their accumulated procedural knowledge, bringing together demands that existing benchmarks fail to examine. $\Delta\tau$-Bench comprises 1,588 validated multi-turn tasks across 42 domains and six successive ecosystem states generated through capability-preserving transformations. Our evaluation shows that even the strongest evaluated model or harness achieves only 42.3% task success, while initial gains from existing s
    
[^119]: 合成语言数据

    Synthetic linguistic data

    [https://arxiv.org/abs/2610.34215](https://arxiv.org/abs/2610.34215)

    本文提出“人工植物苗圃”隐喻，将合成语言数据分为“合成灌木丛数据”（类语料库的模型生成语言集合）和“合成花园数据”（受控条件下引出的模型输出），论证合成数据是独特的语言学研究对象，其解读须考虑生成条件。

    

    合成语言数据在语言学研究中日益凸显，应用范围涵盖从对大规模AI生成文本集合的分析，到调查语法和词汇偏好的受控实验。本术语词条扩展了Bresnan（2016）对“语言学花园”与“灌木丛”的区分，引入“人工植物苗圃”作为合成数据的隐喻。我们区分了两类数据：合成灌木丛数据，主要由类似语料库的模型生成语言集合构成；以及合成花园数据，即在受控条件下引出模型输出以检验特定语言学假设。我们论证，合成数据构成一个独特的语言学研究对象：将其与自然语言数据进行比较既能揭示共同模式，也能显现差异，但对这些数据的解读需要仔细考虑其生成条件。

    arXiv:2610.34215v1 Announce Type: new  Abstract: Synthetic linguistic data have become increasingly prominent in linguistic research, from the analysis of large collections of AI-generated texts to controlled experiments investigating grammatical and lexical preferences. This glossary entry extends Bresnan's (2016) distinction between the linguistic garden and the bush by introducing an artificial plant nursery as a metaphor for synthetic data. We distinguish between synthetic bush data, consisting primarily of corpus-like collections of model-generated language, and synthetic garden data, in which model outputs are elicited under controlled conditions to investigate specific linguistic hypotheses. We argue that synthetic data constitute a distinct object of linguistic investigation: comparing them with natural language data can reveal both shared patterns and divergences, but their interpretation requires careful consideration of the conditions under which they are generated.
    
[^120]: 更高的推理努力程度可提高GPT-5.5参考文献的可验证性并改变其推理摘要

    Higher Reasoning Effort Improves Reference Verifiability and Changes Reasoning Summaries in GPT-5.5

    [https://arxiv.org/abs/2610.34212](https://arxiv.org/abs/2610.34212)

    提高推理努力程度可显著提升GPT-5.5所生成学术参考文献的可验证性（解析器验证通过率从78.3%升至84.8%），同时使其推理摘要变得更长并发生特征上的变化。

    

    大型语言模型正越来越多地被应用于科学知识生产的各个阶段，包括参考文献的推荐。这些模型可以在不同的推理努力水平下运行，并提供其推理过程的摘要。这引发了两个问题：提高推理努力是否能改善所生成参考文献的可验证性？由此产生的推理摘要能否揭示模型行为随更高努力而发生了怎样的变化？我们研究了GPT-5.5在配对的低努力与高努力设置下为1,000篇论文生成学术参考文献时的推理摘要。生成的参考文献与外部书目记录进行了匹配，推理摘要依据四个预定义特征进行评分，并通过GPT-5.5探索了其他新发现的特征。高努力使论文层面的解析器验证参考文献平均通过率从78.3%提升至84.8%，并产生了更长的推理摘要。所有四个p（摘要在此处被截断）

    arXiv:2610.34212v1 Announce Type: new  Abstract: Large language models are increasingly being used in different stages of scientific knowledge production, including the suggestion of references. These models can operate at different levels of reasoning effort and provide summaries of their reasoning processes. This raises two questions: Does increasing reasoning effort improve the verifiability of generated references, and do the resulting reasoning summaries reveal how the model's behavior changes with greater effort? We examined summaries from GPT-5.5 as it generated scholarly references for 1,000 papers under paired low- and high-effort settings. Generated references were matched to external bibliographic records, and reasoning summaries were scored for four predefined features and explored for additional features discovered by GPT-5.5. High effort increased the mean paper-level resolver-verified reference rate from 78.3% to 84.8%, and produced longer reasoning summaries. All four p
    
[^121]: MIS-VQA：一个标注驱动的手术视觉-语言数据集，用于复杂手术场景理解

    MIS-VQA: An Annotation-Driven Surgical Vision-Language Dataset for Complex Surgical Understanding

    [https://arxiv.org/abs/2610.34210](https://arxiv.org/abs/2610.34210)

    该论文提出了标注驱动的手术视觉-语言数据集MIS-VQA，通过将多源异构手术标注转化为结构化文本场景表示，引导大语言模型生成临床连贯的多样化问答对，弥补了现有数据集在多模态手术理解和复杂推理方面的不足。

    

    现代手术人工智能越来越需要多模态理解能力，而非孤立的感知任务，然而现有数据集大多仍局限于识别或模板化问答等单任务监督。当前的医学VQA资源往往依赖合成模板、缺乏自然描述性监督，或提供的标注过于稀疏，从而阻碍了对空间关系和手术流程上下文的推理。我们提出了MIS-VQA，这是一个通过标注驱动流程生成的手术视觉-语言数据集，该流程将异构的手术标注转换为结构化的文本场景表示。基于EndoVis18和CholecT50数据集，我们将手术阶段、器械、交互、三元组和分割等标签转换为统一的上下文信息，引导大语言模型生成临床上连贯的问答对。MIS-VQA支持四种响应类型——简洁描述、详细描述、感知级视觉定位以及高级视觉问答。

    arXiv:2610.34210v1 Announce Type: cross  Abstract: Modern surgical AI increasingly requires multimodal understanding rather than isolated perception, yet existing datasets remain largely limited to single-task supervision such as recognition or templated question answering. Current medical VQA resources often rely on synthetic templates, lack natural descriptive supervision, or provide sparse annotations that hinder reasoning about spatial relations and procedural context. We introduce MIS-VQA, a surgical vision-language dataset generated through an annotation-driven pipeline that converts heterogeneous surgical annotations into structured textual scene representations. Using EndoVis18 and CholecT50, we transform phase, instrument, interaction, triplet, and segmentation labels into unified contexts that guide large language models to generate clinically coherent question-answer pairs. MIS-VQA supports four response types-concise and detailed descriptions, perception-level grounding, an
    
[^122]: 稳态优化的熵控制：通过动态反馈预防灾难性遗忘

    Entropic Control for Homeostatic Optimization: Pre-empting Catastrophic Forgetting via Dynamic Feedback

    [https://arxiv.org/abs/2610.34182](https://arxiv.org/abs/2610.34182)

    提出ECHO闭环控制框架，将输出熵作为KL散度变化的高效在线代理指标，通过比例控制律动态调节回放数据梯度权重，实时预防大模型微调中的灾难性遗忘。

    

    大语言模型的监督微调（SFT）存在灾难性遗忘问题，而现有的缓解策略难以协调可塑性与稳定性之间的困境。为解决这一问题，我们提出了稳态优化的熵控制（ECHO），这是一种闭环控制框架。通过对KL散度进行分解，我们识别出输出熵是KL变化中占主导地位的可观测成分。这使我们能够将熵作为一种计算高效的在线代理指标，实时监测遗忘风险，从而避免昂贵的参考模型推理。因此，ECHO采用了一个弹性控制器，通过比例控制律动态调节回放数据的梯度权重，并将偏离预训练熵平衡的程度作为误差信号。在0.6B至9B参数规模架构上的实证评估表明，ECHO达到了所观测到的可塑性-稳定性边界，并在若干对比中进一步扩展了该边界。

    arXiv:2610.34182v1 Announce Type: cross  Abstract: Supervised Fine-Tuning (SFT) of LLMs suffers from catastrophic forgetting, and prevailing mitigation strategies struggle to reconcile the plasticity-stability dilemma. To address this, we propose Entropic Control for Homeostatic Optimization (ECHO), a closed-loop control framework. By decomposing the KL divergence, we identify output entropy as a dominant observable component of KL variation. This enables us to leverage entropy as a computationally efficient online proxy to monitor forgetting risks in real-time, bypassing expensive reference-model inference. Consequently, ECHO employs an elastic controller that dynamically modulates the gradient weight of replay data via a proportional control law, using deviations from the pre-trained entropic equilibrium as an error signal. Empirical evaluations across 0.6B to 9B parameter architectures demonstrate that ECHO lies on, and in several comparisons extends, the observed plasticity-stabili
    
[^123]: 书写重要内容：解耦循环记忆中的词元选择与槽位分配

    Write What Matters: Disentangling Token Selection from Slot Allocation in Recurrent Memory

    [https://arxiv.org/abs/2610.34180](https://arxiv.org/abs/2610.34180)

    该论文提出因子化写入路由，将循环记忆中“是否存储某个词元”与“写入多少槽位”两个决策解耦，并揭示最优写入策略取决于信息密度：密集信息任务适合始终写入，稀疏事实任务适合学习到的选择性写入。

    

    长序列模型必须保留较早的信息。全注意力机制保留所有历史词元但成本高昂，而循环记忆模型则将历史压缩至固定槽位中以实现线性时间处理。有限的容量使得写入操作至关重要：干扰信息会影响有用内容的存储，而被跳过的证据则无法再被召回。现有的多槽位记忆模型将每个词元写入相同数量的槽位，从而将“是否存储”与“更新多少槽位”这两个决策混为一谈。我们提出因子化写入路由，首先决定保留或跳过每个词元，然后再选择固定或词元自适应的槽位数量，而查询时仍然读取所有槽位。在两个于长上下文之后查询事实的合成任务上进行十个随机种子的实验中，我们观察到一种依赖于信息密度的反转现象：在每次写入四个槽位的情况下，长上下文准确率在信息密集的Feature Recall任务上偏向始终写入（0.472 vs 0.337），而在稀疏事实的Passkey任务上偏向学习到的选择性写入（0.222 vs 0.048）。

    arXiv:2610.34180v1 Announce Type: cross  Abstract: Long-sequence models must retain earlier information. Full attention keeps all past tokens but grows expensive, and recurrent-memory models compress history into fixed slots for linear-time processing. Limited capacity makes writes consequential: distractors can interfere with useful content, while skipped evidence cannot be recalled. Existing multi-slot memories write every token to the same number of slots, merging whether to store it with how many slots to update. We introduce Factorized Write Routing (FWR), which first keeps or skips each token and then chooses a fixed or token-adaptive number of slots. Queries still read all slots. Across ten seeds on two synthetic tasks that query facts after long contexts, we observe a density-dependent reversal. With four slots per write, long-context accuracy favors always writing on dense Feature Recall (0.472 vs 0.337), but learned selective writing on sparse-fact Passkey (0.222 vs 0.048). T
    
[^124]: 推理还是仅仅提示？探究潜在化学推理的机制

    Reasoning or Just Prompting? Probing the Mechanism of Latent Chemical Reasoning

    [https://arxiv.org/abs/2610.34173](https://arxiv.org/abs/2610.34173)

    研究通过对潜在状态进行交换、扰动和移除的因果干预实验发现，潜在化学推理中的潜在状态并非真正编码逐步推理链，而是像软提示一样主要起引导模型计算的作用——一个跨分子共享的固定前缀即可复现原有性能。

    

    潜在推理在连续空间中执行中间计算，而非通过显式token进行，这在化学领域尤其具有吸引力，因为显式推理中的错误可能会传播到最终答案。然而，如果潜在状态真正构成了一个隐式推理链，它们应当具有因果上的重要性。我们的发现恰恰相反：交换或适度扰动中间潜在状态对性能几乎没有影响，而完全移除它们有时却会导致分数大幅下降。因此，潜在状态确实重要，但其针对特定分子的内容似乎重要性要低得多。这指向了一个更简单的解释：潜在状态可能主要不是在编码逐步的推理链，而是主要在引导后续的模型计算，其作用类似于软提示。事实上，一个在所有分子间共享的单一固定前缀就能复现LatentChem的性能，而且学习到的静态软提示与针对每个输入的潜在状态表现相当。

    arXiv:2610.34173v1 Announce Type: cross  Abstract: Latent reasoning performs intermediate computation in continuous space rather than through explicit tokens, which is particularly appealing in chemistry, where errors in explicit reasoning can propagate to the final answer. Yet if latent states truly constitute an implicit reasoning chain, they should be causally important. We find otherwise: swapping or moderately perturbing intermediate latent states has little effect on performance, while removing them entirely sometimes causes a substantial score drop. Thus, the latent states matter, but their molecule-specific content appears to matter far less. This points to a simpler interpretation. Rather than encoding a step-by-step reasoning chain, latent states may primarily steer subsequent model computation, much like soft prompts. Indeed, a single fixed prefix shared across molecules reproduces LatentChem's performance, and a learned static soft prompt performs comparably to per-input la
    
[^125]: 任务成功还不够：面向工具使用型语言模型代理的副作用感知评估

    Task Success Is Not Enough: Side-effect-Aware Evaluation of Tool-Using Language Model Agents

    [https://arxiv.org/abs/2610.34167](https://arxiv.org/abs/2610.34167)

    该论文提出SABench基准，首次系统性评估工具使用型大语言模型代理在成功完成任务时可能产生的未经授权副作用（“附带损害”），发现11个前沿模型中15.2%的运行存在此类风险。

    

    工具使用型语言模型代理通常以是否完成用户任务来评估。这一指标遗漏了一种常见的失败情形：代理可能在完成所请求工作的同时，还执行了未经授权的具有副作用的操作，例如写入错误的对象、扩大请求的范围，或联系非预期的接收者。我们将这种失败模式称为“附带损害”。我们提出了SABench（副作用感知基准），这是一个包含201个电子邮件、日历和工单任务的基准。每个任务将四种状态结构（独立、依赖、外部可达、跨工具）之一与四种陷阱类型（歧义、目标错位、级联伤害、过度扩权）之一相结合，且每个任务都配有一个确定性的、基于轨迹的判定器，可同时检验目标达成与授权合规性。在来自11个前沿大语言模型（涵盖主要商业和开源模型家族）的2,211个模型-任务对中，15.2%的运行产生了附带损害，且风险集中于……

    arXiv:2610.34167v1 Announce Type: new  Abstract: Tool-using language model agents are usually evaluated by whether they finish the user's task. That metric misses a common failure: an agent can complete the requested work while also taking an unauthorized side-effecting action, such as writing to the wrong object, expanding the scope of the request, or contacting an unintended recipient. We call this failure mode collateral damage (CD). We introduce SABench (Side-effect-Aware Benchmark), a benchmark of 201 email, calendar, and ticket tasks. Each task crosses one of four state structures (independent, dependent, externally reachable, cross-tool) with one of four trap families (ambiguity, mis-targeting, cascade harm, overreach), and each task has a deterministic trace-based oracle for both goal satisfaction and authorization compliance. Across 2,211 model-task pairs from 11 frontier LLMs spanning major commercial and open model families, 15.2% of all runs incur CD, with risk concentrated
    
[^126]: 空间中的人类：多模态大语言模型（MLLM）能否通过主动外中心探索构建以人为中心的空间理解？

    Humans in Space: Can MLLMs Construct Human-Centric Spatial Understanding via Active Exocentric Exploration?

    [https://arxiv.org/abs/2610.34149](https://arxiv.org/abs/2610.34149)

    该论文提出SATE-Bench——一个包含316个真实场景和2,525个多轮问答的交互式基准，用于评估多模态大语言模型的视觉自我意识与主动外中心视角探索能力，并发现现有模型在主动探索中倾向于过早终止探索。

    

    尽管多模态大语言模型（MLLM）在零样本推理方面表现出色，但其空间能力主要通过静态、被动的观察方式来评估。在复杂的以人为中心的场景中，遮挡常常隐藏了关键的几何证据，而被动式基准迫使模型进行猜测，而非识别自身视觉上的不确定性。为了弥合这一差距，我们提出了SATE-Bench，这是一个新颖的交互式基准，包含316个真实世界场景和2,525个多轮问答对，用于评估视觉自我意识和主动的外中心视角探索能力。它将被动多视角控制与主动探索设置相结合：模型从严重遮挡的视角出发，必须迭代地查询相邻视角，直到收集到足够的证据来完全“满足”问题，并通过步骤惩罚的效率奖励进行评分。对最先进MLLM的评估显示，它们在报告高……（摘要在此处截断，原文不完整）

    arXiv:2610.34149v1 Announce Type: cross  Abstract: While Multimodal Large Language Models (MLLMs) excel at zero-shot reasoning, their spatial capabilities are primarily evaluated through static, passive observation. In complex, human-centric scenes, occlusion often hides crucial geometric evidence, and passive benchmarks force models to guess rather than recognize their own visual uncertainty. To bridge this gap, we introduce SATE-Bench, a novel interactive benchmark comprising 316 real-world scenes and 2,525 multi-turn QA pairs to evaluate visual self-awareness and active exocentric view-exploration. It combines a Passive multi-view control with an Active exploration setting, where models start from a severely occluded view and must iteratively query adjacent viewpoints until they gather enough evidence to fully ``sate'' the question, scored by a step-penalized efficiency reward. Evaluating state-of-the-art MLLMs reveals that they terminate exploration prematurely while reporting high
    
[^127]: 长上下文时代的检索再思考：用于自适应上下文构建的无训练语义树

    Rethinking Retrieval in the Long-Context Era: Training-Free Semantic Trees for Adaptive Context Construction

    [https://arxiv.org/abs/2610.34146](https://arxiv.org/abs/2610.34146)

    该论文提出了一种免训练的语义树框架，在长上下文模型时代通过语义抽象与结构-语义评分自适应地选择和组织证据，在保留原文的前提下构建聚焦于查询的上下文，从而提升问答效果。

    

    长上下文语言模型日益能够处理整篇文档，使得全上下文提示成为段落检索的一种实用替代方案。然而，更大的容量只决定了模型能接收多少信息，而不能决定哪些信息最能支持特定查询。这一区别促使我们重新思考检索：当整篇文档已经能够放入上下文时，选择和组织其中的证据能否改善问答效果？挑战在于如何保留支持性事实和关系，同时使上下文聚焦于问题。为了调和这些目标，我们提出了 \ours，一个免训练框架，它利用语义抽象来指导选择，同时保留原始文本用于回答。\ours 在单次生成中从完整文档构建一个可复用的语义层次结构，将节点描述链接到源文本片段。给定一个查询，结构-语义评分和细粒度词汇归因会提名共享的源节点……

    arXiv:2610.34146v1 Announce Type: cross  Abstract: Long-context language models can increasingly process entire documents, making full-context prompting a practical alternative to passage retrieval. However, greater capacity determines how much information a model can receive, but not which information best supports a query. This distinction motivates rethinking retrieval: when the full document already fits, can selecting and organizing its evidence improve answering? The challenge is to retain supporting facts and relations while keeping the context focused on the question. To reconcile these goals, we introduce \ours, a training-free framework that uses semantic abstraction to guide selection while retaining original text for answering. \ours constructs a reusable semantic hierarchy from the full document in a single generation, linking node descriptions to source spans. Given a query, structural--semantic scoring and fine-grained lexical attribution nominate shared source nodes; th
    
[^128]: 自进化推理器：通过策略内在的探索与利用实现持续优化

    Self-Evolution Reasoner: Continuous Optimization via Policy-Intrinsic Exploration and Exploitation

    [https://arxiv.org/abs/2610.34137](https://arxiv.org/abs/2610.34137)

    提出自进化策略优化（SEPO）框架，利用标签从策略自身引出正确推理以实现引导性探索和分布内利用，克服RLVR与自监督蒸馏方法固有的探索瓶颈，从而驱动大语言模型推理能力的持续自我进化。

    

    可验证奖励的强化学习（RLVR）使大型语言模型（LLM）能够持续自我提升其推理能力。与RLVR的标量奖励相比，近期的自监督在策略蒸馏方法利用了更密集的监督信号。然而，这两种范式都存在探索瓶颈，当策略无法采样到有效解时，难以摆脱局部最优。依赖外部演示虽然可以绕过这一问题，但其离策略的本质阻碍了真正的自我提升，且分布不匹配的推理无法被充分利用。在本工作中，我们提出了自进化策略优化（SEPO），这是一个通过引导性探索和分布内利用来驱动推理能力持续进化的框架。具体而言，SEPO利用标签从策略自身引出正确的推理，释放其探索潜力。通过将当前策略作……（摘要在此处截断）

    arXiv:2610.34137v1 Announce Type: cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) enables large language models (LLMs) to continuously self-improve their reasoning capabilities. Compared to RLVR's scalar rewards, recent self-supervised on-policy distillation methods leverage denser supervision signals. However, both paradigms suffer from exploration bottlenecks, struggling to escape local optima when the policy fails to sample valid solutions. Relying on external demonstrations can bypass this issue; however, the off-policy nature hinders genuine self-improvement and the mismatched-distribution reasoning can not be fully exploited. In this work, we propose Self-Evolution Policy Optimization (SEPO), a framework that drives continuous reasoning capability evolution via guided exploration and in-distribution exploitation. Specifically, SEPO utilizes labels to elicit correct reasoning from the policy, unlocking the exploration potential. By treating current policy co
    
[^129]: 大语言模型能否恰当信任？评估对不完美伙伴的校准依赖

    Can LLMs Trust Appropriately? Evaluating Calibrated Reliance on Imperfect Partners

    [https://arxiv.org/abs/2610.34107](https://arxiv.org/abs/2610.34107)

    本研究提出了一个包含23个协作类别、324个场景的基准测试，用于评估大语言模型能否根据伙伴证据合理调整信任、验证或协助决策，发现最强前沿模型接近最优，而其他模型在恰当依赖方面表现不佳。

    

    大语言模型越来越多地与人类及同类AI智能体协作，而后者的贡献可能是不完整的、不一致的或错误的。有效的协作需要校准的信任，即随着经验证据的积累，依赖、验证或协助伙伴的决策能够理性地调整。我们研究模型是否能将伙伴证据转化为恰当的依赖策略。我们引入了一个涵盖23个协作类别、共324个场景的基准测试。每个场景涉及三个顺序决策，具有明确的转移动态、反馈信号和成本计划，从而可以针对剩余交互中的预期后果对各项选择进行精确的解析评估。通过对七个模型以及36个受控历史探针的评估，我们将证据敏感性与规范上恰当的行动区分开来。尽管最强的前沿模型接近最优决策，其他模型则表现出……（原文摘要在此处截断）

    arXiv:2610.34107v1 Announce Type: new  Abstract: Large language models increasingly collaborate with humans and peer AI agents whose contributions may be incomplete, inconsistent, or incorrect. Effective collaboration requires calibrated trust, whereby decisions to rely on, verify, or assist a partner adapt rationally as empirical evidence accumulates. We investigate whether models translate partner evidence into appropriate reliance policies. We introduce a benchmark of 324 scenarios across 23 collaboration categories. Each scenario involves three sequential decisions with explicit transition dynamics, feedback emissions, and cost schedules, permitting exact analytical evaluation of choices against their expected consequences over the remaining interaction. Evaluating seven models alongside 36 controlled-history probes, we dissociate evidence sensitivity from normatively appropriate action. While the strongest frontier model approaches near-optimal decisions, other models exhibit recu
    
[^130]: T2D：面向高效投机解码的目标模型到草稿模型剪枝与适配

    T2D: Target-to-Draft Model Pruning & Adaptation for Efficient Speculative Decoding

    [https://arxiv.org/abs/2610.34103](https://arxiv.org/abs/2610.34103)

    提出T2D框架，将草稿模型设计表述为优化问题，通过延迟感知的结构化剪枝和知识蒸馏直接从目标模型构建草稿模型，实现推理延迟与分布对齐的联合优化，从而提升投机解码效率。

    

    投机解码显著加速了大型语言模型（LLMs）中的自回归文本生成，但其有效性主要取决于两个关键因素：草稿模型与目标模型之间的分布对齐程度，以及草稿模型在推理过程中的延迟。然而，现有方法通常使用独立预训练的小型草稿模型，或者试图将目标模型剪枝到手动选择的架构，从而无法显式地平衡草稿模型对齐与token生成效率之间的权衡。为了解决这一缺陷，我们将草稿模型设计表述为一个优化问题，并提出了T2D框架，该框架通过延迟感知的结构化剪枝和知识蒸馏，直接从目标模型构建草稿模型。通过这种方式，T2D获得的草稿模型在实测推理延迟和目标模型响应的交叉熵损失上实现了联合优化。

    arXiv:2610.34103v1 Announce Type: cross  Abstract: Speculative decoding significantly accelerates autoregressive text generation in Large Language Models (LLMs), but its efficacy depends largely on two key factors: the distributional alignment between the draft and the target model, and the draft model's latency during inference. However, existing approaches typically utilize independently pretrained small draft models or seek to prune the target model to a manually chosen architecture, thereby failing to explicitly balance the trade-off between draft model alignment and token generation efficiency. To address this gap, we formulate draft model design as an optimization problem and propose T2D, a framework that constructs draft models directly from the target model using latency-aware structured pruning and knowledge distillation. In this manner, T2D obtains draft models that are jointly optimized for both measured inference latency and cross-entropy loss on the target model's response
    
[^131]: LALA基准：评估大语言模型中的文学与语言学分析能力

    The LALA Benchmark: Evaluating Literary and Linguistic Analysis in LLMs

    [https://arxiv.org/abs/2610.34080](https://arxiv.org/abs/2610.34080)

    LALA基准通过236道专家撰写的文学与语言学开放式问题评估17个大语言模型，发现后训练（而非参数量）是使模型具备可用的语言能力的关键因素。

    

    我们推出了文学与语言学分析基准（LALA），其中包含236道由语言学和文学研究领域专家撰写的开放式问题，并附有专家提供的理想答案。通过使用人类评委和大语言模型评委，在同一模型家族内及相似训练规模下对模型进行比较，LALA识别出了与后训练相关的语言能力差异，为评估未来模型和指导针对性改进提供了一个框架。我们对17个语言模型进行了测试，并采用人类专家评估和两个大语言模型评估器（Sol和Qwen）对其回答进行了评价。结果显示，基座模型几乎不具备该基准所考察的语言能力，而经过后训练的版本表现显著更好，这表明后训练在使语言知识变得可获取和可用方面发挥着核心作用。相比之下，参数量与语言能力之间未显示出统计学上的显著关联。

    arXiv:2610.34080v1 Announce Type: new  Abstract: We introduce the Literature and Linguistics Analysis Benchmark (LALA), comprising 236 open-ended questions written by experts in linguistics and literary studies, together with expert-provided ideal answers. By comparing models within families and at similar training scales using human and LLM judges, LALA identifies differences in linguistic competence associated with post-training, offering a framework for evaluating future models and guiding targeted improvements. We administered questions to 17 language models and evaluated their responses using both human expert assessment and two LLM evaluators, Sol and Qwen. Base models displayed little of the linguistic competence assessed by the benchmark, whereas their post-trained versions performed markedly better, suggesting that post-training plays a central role in making linguistic knowledge accessible and usable. Parameter count, by contrast, showed no statistically significant associati
    
[^132]: 相同方式训练的语言模型会被不同的越狱攻击所攻破

    Identically Trained Language Models Break on Different Jailbreaks

    [https://arxiv.org/abs/2610.34074](https://arxiv.org/abs/2610.34074)

    该研究通过训练150个模型发现，即使数据配方和安全训练完全相同，不同预训练随机种子产生的模型对不同越狱攻击的脆弱性差异显著，表明模型的越狱易感模式主要由预训练运行的随机性决定，而非数据或安全训练。

    

    越狱基准测试通常只测试单个模型检查点，并将哪些越狱攻击能攻破它视为其预训练数据和安全性训练的固有属性。我们提出疑问：如果使用相同的预训练数据进行另一次预训练，并给予相同的安全性训练，模型是否会在相同的越狱攻击下被攻破？针对25种预训练数据配方，我们分别从3个预训练随机种子中获取1B规模的基础模型，并对每个模型施加由2个后训练随机种子产生的相同安全后训练，共计150个模型。随后，我们收集每个模型对一组共享越狱提示的20个回复，并计算每对模型之间逐提示非拒绝率的相关性。结果显示，同一预训练运行的两个安全后训练副本的逐提示非拒绝率相关性平均高达0.984；更换不同的预训练种子会使相关性降至0.811，而更换不同的数据配方仅使其进一步小幅下降至0.779。这些结果表明，哪些越狱攻击能攻破一个模型检查点，更多取决于其（预训练运行的随机性，摘要在此处截断）……

    arXiv:2610.34074v1 Announce Type: cross  Abstract: Jailbreak benchmarks test one checkpoint and treat which jailbreaks break it as a property of its pretraining data and safety training. We ask whether another pretraining run of the same data, given the same safety training, would break on the same jailbreaks. For each of 25 pretraining data recipes, we take 1B base models from 3 pretraining seeds and give each the same safety post-training with 2 post-training seeds, for 150 models in all. We then collect 20 responses from every model to a shared set of jailbreak prompts and correlate the per-prompt non-refusal rates between every pair of models. On average, two post-trained copies of the same pretraining run have per-prompt non-refusal rates that correlate at 0.984. A different pretraining seed lowers the correlation to 0.811, and a different data recipe lowers it only slightly further, to 0.779. These results indicate that which jailbreaks break a checkpoint depends much more on its
    
[^133]: 记忆税：剖析智能体记忆的净价值

    Memory Tax: Dissecting the Net Value of Agent Memory

    [https://arxiv.org/abs/2610.34073](https://arxiv.org/abs/2610.34073)

    该论文提出将智能体记忆的净价值分解为“记忆收益”与“记忆税”的分析框架，并通过大规模评估揭示许多专用记忆系统的表现不如朴素RAG（净价值为负），其根源在于记忆处理对原始信息的组织与访问方式的改变是一把双刃剑。

    

    记忆在长时程智能体中的应用日益增多，然而与直接从原始历史记录中检索相比，构建记忆的净价值在很大程度上仍不明确。通过跨多个基准测试和多种模型骨干的大规模评估，我们发现许多专用记忆系统在许多情况下表现不如朴素的RAG，表明其净价值为负。以RAG为参照，我们将记忆的净价值分解为相对于直接检索的收益与损失，分别称为“记忆收益”和“记忆税”。这些收益与损失之所以产生，是因为记忆处理过程改变了原始信息的组织与访问方式。我们将这些改变视为对原始历史的“双刃剑”式偏差：它们最初是为有益于任务而设计和引入的，例如重组事实或提炼偏好，但有些也可能移除或掩盖证据，从而产生记忆税。由于记忆收益反映了其设计的预期收益，我们随后……

    arXiv:2610.34073v1 Announce Type: new  Abstract: Memory is increasingly used in long-horizon agents, yet the net value of building memory over direct retrieval from raw history remains largely unclear. Through a large-scale evaluation across multiple benchmarks and model backbones, we find many dedicated memory systems underperform naive RAG in many cases, indicating a negative net value. Taking RAG as a reference, we dissect the net value of memory into the gains and losses over direct retrieval, termed memory benefit and memory tax. These gains and losses arise because memory process changes how the raw information is organized and accessed. We view such changes as the double-edged deviations from the raw history. They are originally designed and introduced to benefit the task, for example by reorganizing facts or distilling preferences, but some may also remove or obscure evidence, giving rise to memory tax. Since memory benefit reflects the expected gains of its design, we then foc
    
[^134]: GEPA合并何时有效？一种机制层面的诊断

    When Does GEPA Merge Help? A Mechanism-Level Diagnostic

    [https://arxiv.org/abs/2610.34027](https://arxiv.org/abs/2610.34027)

    该研究通过系统性诊断发现，GEPA合并算子的跨模型不稳定性源于配置敏感性而非算子本身的固有缺陷，并构建了一个基于行为的探测器，通过对高风险任务弃权来避免合并失败。

    

    GEPA是目前针对语言模型程序最强的反思式提示优化器之一，但其系统感知的合并算子在跨模型场景下表现不稳定：合并操作平均上能提升gpt-4.1-mini的性能，却会降低Qwen3-8B的表现——GEPA论文注意到了这一不对称性，但未给出解释。我们在五个数据集和两个模型上研究了这种不稳定性。我们对合并时机和合并算子进行的九种配置扫描表明，在合理配置下，合并可以同时提升两个模型的性能。这表明合并的不稳定性更适合用配置敏感性来解释，而非合并算子本身的固有缺陷。随后，我们测试了提示层特征能否预测GEPA的合并接受决策，结果发现不能，从而排除了静态特征门控的可能性。作为概念验证，基于这些诊断构建的一个基于行为的探测器，在我们的多种子运行中通过对高风险任务采取弃权策略来避免大规模合并失败，尽管它并不能始终一致地（原文在此处截断）……

    arXiv:2610.34027v1 Announce Type: new  Abstract: GEPA is among the strongest reflective prompt optimizers for language-model programs, but its system-aware merge operator is unstable across models: merge improves gpt-4.1-mini on average while degrading Qwen3-8B, an asymmetry noted but left unexplained by the GEPA paper. We investigate this instability across five datasets and two models. Our nine-configuration sweep over merge timing and merge operator shows that merge can improve both models when configured appropriately. This suggests that merge instability is better explained by configuration sensitivity than by an inherent flaw in the merge operator. We then test whether prompt-level features predict GEPA's merge-accept decisions and find that they do not, ruling out static-feature gating. As a proof of concept, a behavior-based probe built from these diagnostics avoids large merge failures in our multi-seed runs by abstaining on high-risk tasks, although it does not consistently o
    
[^135]: 患者总是对的吗？基于分级患者信念的医学诊断谄媚行为评估

    The Patient Is Always Right? Evaluating Sycophancy in Medical Diagnosis with Graded Patient Beliefs

    [https://arxiv.org/abs/2610.34013](https://arxiv.org/abs/2610.34013)

    该研究评估了九个AI模型在医学诊断中的谄媚行为，发现模型平均23.2%的情况下会附和患者错误的自诊信念（甚至包括虚构疾病），而医学微调和更大的模型规模可降低这种易感性。

    

    越来越多的人开始向通用AI模型寻求医疗建议，包括诊断。然而，AI模型容易受到谄媚行为的影响，例如即使知道用户的信念不真实，也会对其进行附和验证，这可能导致假阳性诊断。本研究评估了当患者声称自己患有某种特定（但不正确）的疾病时，九个AI模型会如何改变其诊断决策。我们改变了用户信念与所呈现临床证据之间的接近程度。结果显示，模型平均有23.2%的概率会谄媚地采纳患者的信念——有时即使患者提出的是一个虚构的、根本不存在的疾病，模型也会附和。我们还测试了不同的人物设定，其中一种“罕见疾病声称者”设定使模型接受其信念的概率高达43.0%。附加分析发现，经过医学领域微调以及规模更大的模型更不容易改变其原有诊断。医学诊断中的谄媚偏差引发了……

    arXiv:2610.34013v1 Announce Type: new  Abstract: More people are seeking out general-purpose AI models for medical advice, including diagnosis. Yet, AI models are susceptible to sycophancy, like validating users' beliefs even when they are untrue, which could lead to false positive diagnoses. Here, we evaluate how nine AI models change their diagnostic decision when a patient presents a belief that they have a specific (but incorrect) medical condition. We vary how close the user belief is to the clinical evidence presented. Models sycophantically adopt the patient's belief on average 23.2\% of the time---sometimes even when the patient proposes a fictional, made-up condition. We also varied different personas, including a Rare Disease Claimer that got models to accept their belief 43.0\% of the time. Additional analyses find that medical fine-tuning and model scale makes models less susceptible to changing their diagnoses. Sycophantic bias in medical diagnosis raises implications for 
    
[^136]: 发送至外部大语言模型文档的可逆假名化处理：一个开放的双语基准测试与一个引擎

    Reversible Pseudonymisation of Documents Sent to External LLMs: an Open Bilingual Benchmark and an Engine

    [https://arxiv.org/abs/2610.34012](https://arxiv.org/abs/2610.34012)

    本文提出了一个针对俄英双语文档可逆假名化工具的开放基准测试，基于预注册的合成数据评估了五个现有工具与正则表达式基线，并构建了一个基于两个最强工具的开放引擎，发现现有工具在外部数据集上均无法达标。

    

    专业人士会将工作文档发送给外部大语言模型。一种常见的保护措施是可逆假名化：在发送请求之前将个人数据替换为占位符，并在收到回答后在本地将其还原。我们提出了一个开放基准，涵盖五个现有开源工具和一个正则表达式基线，测试语言为俄语和英式英语；同时提出了一个基于两个最强工具构建的开放引擎。该基准在评判时排除了各工具作者自己的数据集，在相同数据集上对该引擎与各工具的检测能力进行比较，报告掩码精度，并在一个留出的合成文档测试集上进行验证——其文件、哈希值、决策规则、运行计划和停止规则均在任何运行开始之前就已公布。研究还使用两个本地大语言模型测试占位符能否在处理中保持有效。所有评估数据均为合成数据，且该基准由引擎作者本人运行；代码、数据、预测结果和原始答案均已公开发布。在外部数据集上，没有任何现有工具达到……（原文摘要在此处截断）

    arXiv:2610.34012v1 Announce Type: new  Abstract: Professionals send working documents to external large language models. A common safeguard is reversible pseudonymisation: personal data are replaced with placeholders before the request and restored locally in the answer. We present an open benchmark of five existing open tools and a regular-expression baseline on Russian and UK English, and an open engine built on the two strongest tools. The benchmark excludes each tool's own authors' dataset from its verdict, compares the engine with each tool on identical datasets for detection, reports masking precision, and tests on a held-out set of synthetic documents whose files, hashes, decision rules, run plan and stopping rule were published before any run. Two local LLMs test whether placeholders survive. All evaluation data are synthetic, and the benchmark was run by the author of the engine; code, data, predictions and raw answers are released. On external datasets no existing tool meets 
    
[^137]: 面向早期指令微调的“高质量”数据：数据质量指标的实证研究

    "High-Quality" Data for Early-Stage Instruction Tuning: An Empirical Study of Data Quality Metrics

    [https://arxiv.org/abs/2610.33969](https://arxiv.org/abs/2610.33969)

    该研究通过固定微调预算下的分箱对比实验发现，响应率所反映的答案密集程度是比其他常用替代指标更能有效预测指令微调训练效用的简单样本级数据质量指标。

    

    “数据质量”在指令微调中被广泛提及，但这一术语往往定义模糊。我们将其作为一种可操作的属性来研究，探讨哪些源自以往数据选择、文本质量和指令微调研究的简单样本级指标，能够在固定微调预算下最好地预测训练效用。对于每个指标，我们按分数对指令-响应对进行排序，将其划分为词元数量大致相等的低、中、高三个区间，分别在每个区间上进行微调，并将验证困惑度和基于大模型评判（LLM-as-judge）的模型生成答案得分作为累计训练词元数的函数进行对比。我们的结果表明，以响应率所刻画的答案密集型监督信号，比常用的其他替代性指标是更强、更具可迁移性的指令微调效用预测因子。

    arXiv:2610.33969v1 Announce Type: new  Abstract: "Data quality" is widely invoked in instruction tuning, but the term is often underspecified. We study it as an operational property by asking which simple example-level metrics, drawn from prior work on data selection, text quality, and instruction tuning, best predict training utility under a fixed finetuning budget. For each metric, we sort instruction-response examples by score, partition them into low, middle, and high bins with roughly equal token mass, finetune separately on each bin, and compare validation perplexity and LLM-as-judge scores for the models' generated answers as a function of cumulative training tokens. Our results suggest that answer-dense supervision, as captured by response ratio, is a stronger and more portable predictor of instruction-tuning utility than commonly used alternative proxies.
    
[^138]: 超越信念：评估与缓解大语言模型对错误信息的易感性

    Beyond Belief: Evaluating and Mitigating LLM Susceptibility to Misinformation

    [https://arxiv.org/abs/2610.33962](https://arxiv.org/abs/2610.33962)

    该研究发现大语言模型口头表达的可信度判断并不能可靠地转化为对错误信息的行为抵抗——即使模型已将信息评定为不可信，仍有27%至99%的情况会执行有害的工具调用，而简单的预先警告是缓解这一问题的有效手段。

    

    使用工具的大语言模型（LLM）必须判断外部信息是否值得采取行动。我们研究了它们所表达的可信度判断是否能可靠地反映其对错误信息的行为抵抗能力。我们在196条构建的虚假通知（配有对应的合法通知）以及204条适配模拟工具使用的存档攻击上评估了五个大语言模型。当被要求对构建的虚假通知采取行动时，模型在它们已评定为不可信的案例中，有27%至99%的情况发出了有害的工具调用。移除预先评分这一步骤会使有害执行平均增加15个百分点，这表明可信度引出本身就会改变被评估的行为。旨在提升可信度辨别能力的干预措施并不总能带来更安全的行为。简单的预先警告在五个模型中的四个里显著减少了有害执行，但在主要评估中，两种经过测试的生成示例预反驳方法均未能超越其效果。（原文摘要至此截断）

    arXiv:2610.33962v1 Announce Type: cross  Abstract: Tool-using large language models (LLMs) must decide whether external information warrants action. We investigate whether their expressed credibility judgments reliably indicate behavioral resistance to misinformation. We evaluate five LLMs on 196 constructed false notices paired with legitimate counterparts and 204 archived attacks adapted to simulated tool use. When asked to act on the constructed notices, models issue harmful tool calls in 27%--99% of cases they have rated untrustworthy. Removing the preliminary rating step increases harmful execution by 15 percentage points on average, showing that credibility elicitation itself changes the behavior being evaluated. Interventions that improve credibility discrimination do not consistently yield safer actions. A simple forewarning significantly reduces harmful execution in four of five models, but neither tested generated-example prebunk outperforms it in the main evaluation. These r
    
[^139]: 小型编码智能体如何使用代码图与嵌入？

    How Do Small Coding Agents Use Code Graphs and Embeddings?

    [https://arxiv.org/abs/2610.33958](https://arxiv.org/abs/2610.33958)

    研究发现，代码图与嵌入搜索等结构化工具对小型编码智能体的帮助最大——在 1.7B-4B 参数规模可带来 17-21 个百分点的定位准确率提升，但收益随模型规模增大而递减，从而修正了“工具可以补足模型对仓库结构理解”的普遍假设。

    

    编码智能体被赋予嵌入搜索和代码图工具，其前提假设是：仓库的结构信息是模型本身所缺乏、需要由工具来提供的。我们在 SWE-bench Verified 上使用小型开源权重模型检验这一前提，并依据一项配套论文所定义和验证的确定性跳数深度（即从议题中提及的代码到黄金补丁所修改代码之间的最短代码图路径）对任务进行分层。首先，我们在深度均衡的 160 个任务子集以及全部 500 个任务上，对定位智能体的工具集进行消融实验（仅 grep；+嵌入搜索；+图工具；全部），共涉及十个模型（六个规模从 0.6B 到 32B 的 Qwen3，每个规模五种解码方式，一个 FP8 对照，以及三个 Gemma 4 规模；共计 7.9 万个回合）。结构化工具使 Qwen3-8B 和 Gemma 4 E4B 的仅 grep 准确率翻倍；其增益在 1.7B-4B 规模达到 17-21 个百分点的峰值，而在 32B 时降至 4-5 个百分点（参数量每翻倍，几率比为 0.77-0.82）。从 8B 起，该增益中约有一半来自 t……（原文摘要在此处截断）

    arXiv:2610.33958v1 Announce Type: cross  Abstract: Coding agents are given embedding search and code-graph tools on the premise that a repository's structure is something the model lacks and the tool supplies. We test that premise on SWE-bench Verified with small open-weight models, stratifying tasks by a deterministic hop depth (the shortest code-graph path from code the issue mentions to code the gold patch changes) that a companion paper defines and validates. First, we ablate a localization agent's tool set (grep only; +embedding search; +graph tools; all) on a depth-balanced 160-task split and on all 500 tasks for ten models (six Qwen3 sizes from 0.6B to 32B with five decodings each, an FP8 control, three Gemma 4 sizes; 79k episodes). Structured tools double grep-only accuracy for Qwen3-8B and Gemma 4 E4B; their gain peaks at 17-21 points at 1.7-4B and falls to 4-5 points at 32B (odds ratio 0.77-0.82 per doubling of parameters). From 8B up, about half of that gain is the form of t
    
[^140]: 持续学习考试：对大语言模型智能体长程递归自我改进与经验学习的基准测试

    Continual Learning Exam: Benchmarking Long-Horizon RSI and Experiential Learning in LLM Agents

    [https://arxiv.org/abs/2610.33956](https://arxiv.org/abs/2610.33956)

    该论文提出了持续学习考试（CLE）——一个由50多位专家构建、包含120个人工任务和600个等级的大规模基准，通过五级阶梯式任务设计评估LLM智能体在真实职业工作流程中利用积累经验实现持续改进的能力。

    

    持续学习是指大语言模型（LLM）智能体通过从经验中获取、精炼和应用知识，同时保留有用的先验知识，从而随时间不断提升的能力。这对于智能体在长时间跨度上的学习至关重要，然而当前的评估大多只衡量单个任务的表现，而非通过积累经验所带来的改进。我们提出了持续学习考试（CLE），这是一个由50多位领域专家开发的大规模基准，用于衡量真实职业工作流程中经验驱动的改进。CLE包含120个人工编写的任务，每个任务被组织为一个五级阶梯，共产生600个级别，涵盖12个职业领域中的89个专业角色。智能体必须通过每一级才能解锁下一级，后续级别建立在先前任务中积累的经验之上。一个评估智能体根据人工编写的评分标准对每次提交进行评估。我们对比了单次提交设定与（渐进式经验积累设定下的表现）。

    arXiv:2610.33956v1 Announce Type: cross  Abstract: Continual learning is the ability of LLM agents to improve over time by acquiring, refining, and applying knowledge from experience while retaining useful prior knowledge. It is essential for agents learning over long time horizons, yet current evaluations largely assess individual task performance rather than improvement through accumulated experience. We introduce Continual Learning Exam (CLE), a large-scale benchmark developed by 50+ domain experts to measure experience-driven improvement in realistic professional workflows. CLE comprises 120 human-authored tasks, each organised as a five-level ladder, yielding 600 levels spanning 89 professional roles across 12 occupational domains. Agents must pass each level to unlock the next, with later levels building on experience accumulated during earlier assignments. An evaluation agent assesses each submission against human-authored rubrics. We compare a single-submission setting with one
    
[^141]: 什么让问题充满好奇？测量人机交互中的模型好奇心

    What Makes a Question Curious? Measuring Model Curiosity in Human-LLM Interactions

    [https://arxiv.org/abs/2610.33950](https://arxiv.org/abs/2610.33950)

    该研究提出“模型好奇心”的概念及分类体系，发现好奇心导向的问题虽仅占17-28%且较为罕见，但能促使用户做出更具探索性和认知丰富性的回应，尤其在建议、创意和研究场景中能激发探索。

    

    大型语言模型在日常交流中日益扮演着合作伙伴的角色。在这些互动中，有效的协助可能不仅需要回答问题，还需要提出恰当的问题，以揭示知识空白、假设和替代方案。在本研究中，我们将这类基于认知的问题提出行为称为“模型好奇心”，并引入了一个基于理论的分类体系，用以区分好奇心导向的问题与一般性的模型提问。通过对四个对话数据集的分析，我们发现好奇心导向的问题仍然较为罕见（占所有问题的17-28%），但其与用户后续更具探索性、认知上更丰富的回应相关联，且在不同数据集中呈现出不同的模式。我们进一步开展了一项人类研究来评估模型好奇心的有用性，发现好奇心能够鼓励探索，尤其是在建议、创意和研究场景中。这些发现对于构建更善于选择性提问的大型语言模型具有重要意义。

    arXiv:2610.33950v1 Announce Type: cross  Abstract: Large language models increasingly act as partners in everyday communication. In these interactions, useful assistance may require not only answering, but also asking the right questions that surface knowledge gaps, assumptions, and alternatives. In this study, we refer to such epistemic question-asking as model curiosity, and introduce a theory-grounded taxonomy that distinguishes curiosity-oriented questions from generic model questioning. Across four conversation datasets, we find that curiosity-oriented questions remain rare (17-28% of all questions), but are associated with more exploratory and epistemically rich subsequent user responses, with varying patterns across datasets. We further conduct a human study to evaluate the usefulness of model curiosity, finding that curiosity can encourage exploration, especially in advice, creative, and research contexts. These findings have implications for building LLMs that ask more selecti
    
[^142]: 为什么结构化输出会失败？对九个大语言模型中提示词强制执行的Pydantic模式的实证研究

    Why Does Structured Output Fail? An Empirical Study of Prompt-Enforced Pydantic Schemas Across Nine LLMs

    [https://arxiv.org/abs/2610.33949](https://arxiv.org/abs/2610.33949)

    该研究提出了一个细粒度的模式符合性失败分类法和确定性分类器，通过对九个大语言模型配置的3,997次生成实验，揭示了单一有效率指标所掩盖的多样化失败模式，并发现主导失败类别会随模式约束的不同而变化。

    

    需要从托管语言模型获取结构化输出的流水线通常在提示词中指定Pydantic模式，对响应进行验证，并在验证失败时重试。尽管原生结构化输出模式提供了一种替代方案，但应用侧验证对于强制执行自定义约束仍然十分重要。然而，流水线可靠性通常仅用单一的有效率来概括，这掩盖了输出失败的原因以及哪些干预措施可能有效。我们引入了一种细粒度的模式符合性失败分类法，以及一个确定性分类器，该分类器根据模型响应和Pydantic验证错误分配失败类别。我们将此框架应用于九种OpenAI和Anthropic模型/API配置下的3,997次生成尝试，使用来自HotpotQA和FinanceBench的问题。我们的结果表明，相似的有效率可能掩盖不同的失败模式，且主导失败类别随模式约束而变化。

    arXiv:2610.33949v1 Announce Type: cross  Abstract: Pipelines that require structured outputs from hosted language models often specify a Pydantic schema in the prompt, validate the response, and retry when validation fails. Although native structured-output modes provide an alternative, application-side validation remains important for enforcing custom constraints. Yet pipeline reliability is often summarized by a single validity rate, which obscures why outputs fail and which interventions may help. We introduce a fine-grained taxonomy of schema-conformance failures and a deterministic classifier that assigns failure categories from model responses and Pydantic validation errors. We apply this framework to 3,997 generation attempts across nine OpenAI and Anthropic model/API configurations, using questions from HotpotQA and FinanceBench. Our results show that similar validity rates can mask different failure patterns, and that dominant failure categories vary with schema constraints an
    
[^143]: 无模式文本转SQL：从查询日志学习企业数据库结构

    Schema-Free Text-to-SQL: Learning Enterprise Database Structure from Query Logs

    [https://arxiv.org/abs/2610.33946](https://arxiv.org/abs/2610.33946)

    本文提出首个无需模式文档的文本转SQL框架，通过从历史查询日志构建知识图谱并采用实体驱动的GraphRAG检索来学习企业数据库的隐式结构，解决了企业数据库缺乏文档导致现有方法失效的问题。

    

    文本转SQL系统在学术基准测试中取得了显著的准确率，但在企业部署中性能却大幅下降。这一差距源于一个基本假设：现有方法需要完整的模式文档，但企业数据库通常缺乏全面的文档，具有晦涩难懂的命名和隐式关系。我们提出了首个无需模式文档即可运行的文本转SQL框架，它转而从历史查询日志中学习数据库结构。查询日志蕴含着隐式的结构知识：哪些表被一起查询、它们如何连接，以及哪些列映射到业务概念。我们从解析出的查询模式构建知识图谱，并引入实体驱动的GraphRAG检索，将用户问题锚定在领域术语上，而非仅仅依赖嵌入相似度。该方法在经过专家评估的工业部署和Beaver基准测试中均得到了验证。

    arXiv:2610.33946v1 Announce Type: new  Abstract: Text-to-SQL systems achieve remarkable accuracy on academic benchmarks, yet performance drops dramatically in enterprise deployments. This gap stems from a fundamental assumption: existing approaches require complete schema documentation, but enterprise databases typically lack comprehensive documentation, featuring cryptic naming and implicit relationships. We present the first text-to-SQL framework that operates without schema documentation, instead learning database structure from historical query logs. Query logs encode implicit structural knowledge: which tables are queried together, how they join, and what columns map to business concepts. We construct a knowledge graph from parsed query patterns and introduce entity-driven GraphRAG retrieval that grounds user questions in domain terminology rather than relying solely on embedding similarity. Validated on both an industrial deployment with expert evaluation and the Beaver benchmark
    
[^144]: HerHealthGPT：一个面向女性生殖健康的、以证据为依据的对话式AI系统

    HerHealthGPT: An Evidence-Grounded Conversational AI System for Women's Reproductive Health Copy title

    [https://arxiv.org/abs/2610.33931](https://arxiv.org/abs/2610.33931)

    HerHealthGPT是一个面向女性生殖健康、以证据为依据的对话式AI系统，它融合检索增强生成、情境感知的澄清提问与来源验证，通过主动追问补全不完整的健康咨询，并基于可验证的证据生成回答。

    

    “为什么我的月经推迟了？”有时，最有用的回答是从另一个问题开始的。然而，对话式AI系统在面对不完整的健康咨询时，往往不会寻求必要的背景信息，也不会将其建议建立在可验证的证据之上。本文提出了HerHealthGPT，一个以证据为依据的对话式AI系统，旨在应对女性生殖健康领域的这些挑战。HerHealthGPT将检索增强生成（RAG）、情境感知的澄清提问和来源验证整合在一个交互式网页界面中。该系统不会立即对不完整的提问作出回应，而是保持对话上下文、提出有针对性的后续问题、检查是否过早下结论，并将其回答建立在检索到的证据之上。我们使用从90个生殖健康案例中提取的540条英文消息对AI组件进行了评估，并开展了针对30名成人的原型问卷调查。证据检索表……

    arXiv:2610.33931v1 Announce Type: new  Abstract: "Why is my period late?" Sometimes, the most useful response begins with another question. Yet conversational AI systems may respond to incomplete health queries without seeking essential context or grounding their guidance in verifiable evidence. This paper introduces HerHealthGPT, an evidence-grounded conversational AI system designed to address these challenges in women's reproductive health. HerHealthGPT integrates retrieval-augmented generation (RAG), context-aware clarification, and source validation within an interactive web interface. Rather than responding immediately to incomplete queries, the system maintains conversational context, asks targeted follow-up questions, checks for premature conclusions, and grounds its responses in retrieved evidence. We evaluate the AI components using 540 English messages derived from 90 reproductive-health cases and conduct a prototype questionnaire study with 30 adults. Evidence retrieval imp
    
[^145]: 面向语言模型捷径消除的反事实测试时增强方法

    Counterfactual Test-Time Augmentation for Shortcut Removal in Language Models

    [https://arxiv.org/abs/2610.33904](https://arxiv.org/abs/2610.33904)

    该论文提出反事实测试时增强（CTTA）方法，通过辅助模型生成改变捷径值的反事实编辑并对LLM预测加权平均，无需标注数据或模型权重访问权限即可在测试阶段消除大语言模型的捷径依赖，且从理论上证明了最优捷径不变预测器属于该方法族并刻画了其权重。

    

    大语言模型（LLM）可能会利用捷径，即训练数据中存在但在测试时可能不成立的关联性。当这些关联性发生偏移时，可能导致糟糕的预测结果。现有的大多数消除捷径的方案在训练阶段运行，需要标注的训练数据、模型权重访问权限以及大量计算资源。本文研究测试时捷径消除问题。我们提出了一族反事实测试时增强（CTTA）方法：给定一个样本，每种方法使用辅助模型生成改变捷径值的反事实编辑，然后对这些编辑上LLM的预测结果进行加权平均。我们证明了最优的捷径不变预测器属于该CTTA方法族，并刻画了其权重特征。这些权重取决于训练分布，因此在一般情况下，仅凭训练后的模型和无标注数据无法识别这些权重。

    arXiv:2610.33904v1 Announce Type: cross  Abstract: Large language models (LLMs) can exploit shortcuts, i.e., correlations present in training data that may not hold at test time. This can lead to poor predictions when correlations shift. Most existing solutions for removing shortcuts operate at training time, requiring labeled training data, model weight access, and significant computational resources. Here, we study the problem of test-time shortcut removal. We propose a family of counterfactual test-time augmentation (CTTA) methods: given an example, each method uses an auxiliary model to generate counterfactual edits in which the shortcut value is changed, and then takes a weighted average of the LLM's predictions across these edits. We prove that the optimal shortcut-invariant predictor is a member of this CTTA family, and we characterize its weights. The weights depend on the training distribution, and therefore are, in general, not identifiable from the trained model and unlabele
    
[^146]: 当人格设定变得有害：以全球公民价值观对齐引导大语言模型

    When Personas Become Harmful: Guiding LLMs with Global Citizenship Values Alignment

    [https://arxiv.org/abs/2610.33899](https://arxiv.org/abs/2610.33899)

    该研究构建了基于施瓦茨价值观量表的人格设定与争议话题数据集，发现对齐良好的商业大语言模型在响应带价值观人格的请求时仍会以超过47%的成功率产生有害输出，并提出以联合国教科文组织全球公民原则作为最小共识来引导模型在尊重多元价值观的同时保障安全。

    

    大语言模型（LLMs）正日益被来自不同人口群体、拥有多元价值观和不同背景的用户所使用。在尊重带有价值观色彩的人格设定的同时保持安全性是一项独特的挑战，因为基于人格设定的请求通常不像现有安全基准中的请求那样具有明显的恶意。因此，针对对抗性请求设计的对齐方法可能会遗漏由人格设定使用所引发的危害，导致模型要么拒绝合法的价值观表达，要么盲目遵循不安全的人格设定。我们构建了一个包含1.1万个争议性话题的数据集，并将其与基于价值观的人格设定相配对，这些人格设定对应施瓦茨价值观量表（Schwartz Value Survey）中的价值观维度。通过一个迭代优化话题和人格设定的智能体流程，我们以超过47%的成功率从对齐良好的商业大语言模型中诱发出有害响应并加以收集。我们采用联合国教科文组织的全球公民原则作为区分合法价值观表达与有害内容的最小共识基础（摘要原文在此处截断）。

    arXiv:2610.33899v1 Announce Type: new  Abstract: LLMs are increasingly being used by users from diverse demographics with pluralistic values and distinct backgrounds. Maintaining safety while respecting value-laden personas is uniquely challenging because persona-conditioned requests are often not overtly malicious, unlike those in existing safety benchmarks. Alignment methods aimed at adversarial requests therefore may miss the harm that emerges from persona use, leaving models to either refuse legitimate value expression or blindly follow unsafe personas. We construct a dataset of 11k controversial topics paired with value-based personas mapping to value dimensions from the Schwartz Value Survey. Through an agentic pipeline that iteratively refines the topic and persona, we surface and collect harmful responses from well-aligned commercial LLMs with a more than 47% success rate. We use UNESCO's global citizenship principles as a minimal common ground for distinguishing valid value ex
    
[^147]: 干预认证的放大效应：σ加权后训练的有效性门控

    Intervention-Certified Amplification: A Validity Gate for $\sigma$-Weighted Post-Training

    [https://arxiv.org/abs/2610.33882](https://arxiv.org/abs/2610.33882)

    该论文揭示了混合 SFT/RL 后训练中的训练侧信号会重新激活并放大预训练行为（如 DIAA 在移除 Python 围栏后仍以 29% 的比例输出未执行的 TIR），并提出以成对干预实验作为σ加权后训练的有效性门控来认证其正当性。

    

    混合式后训练通常将监督微调（SFT）与强化学习（RL）相结合，利用训练侧信号来设定每个样本的平衡权重。这类信号刻画的是锚点数据与策略本身的特性，而非被强调的行为在目标任务和评估环境中是否真正有益。这种错配可能重新激活并放大预训练行为，即使其字面形式已从监督信号中移除。我们通过 DIAA（双重隐式对抗放大，Dual-Implicit Adversarial Amplification）来展示这一现象，它是一种逐样本σ加权的 SFT/RL 混合方法。在锚点数据中移除所有字面上的 Python 代码围栏后，DIAA 在 MATH-500 的 29% 提示上输出未执行的工具集成推理（TIR），是原始 GRPO 的三倍。成对干预实验表明，抑制 TIR 可带来 10.6 个百分点的准确率恢复，且 SFT 模仿与σ放大做出了相当的贡献；在 0% TIR 教师锚点上训练的学生模型会以 45.8% 的比例输出 TIR。

    arXiv:2610.33882v1 Announce Type: cross  Abstract: Hybrid post-training often mixes supervised fine-tuning (SFT) and reinforcement learning (RL), using training-side signals to set each sample's balance. Such signals characterize the anchor data and policy, but not whether the emphasized behavior helps in the target task and evaluation setting. This mismatch can reactivate and amplify pretrained behavior even after its literal form is removed from supervision. We demonstrate this with DIAA (Dual-Implicit Adversarial Amplification), a per-sample $\sigma$-weighted SFT/RL hybrid. With all literal Python fences removed from its anchor, DIAA emits unexecuted Tool-Integrated Reasoning (TIR) on $29%$ of MATH-500 prompts, three times vanilla GRPO. Paired interventions show $10.6$ percentage points of accuracy recovery when TIR is suppressed and comparable contributions from SFT imitation and $\sigma$-amplification; a student trained on a $0%$-TIR teacher anchor emits it at $45.8%$. Interventio
    
[^148]: 排序不等于决策：查询时RAG投毒防御中的基准捷径与工作点崩溃

    Ranking Is Not Deciding: Benchmark Shortcuts and Operating-Point Collapse in Query-Time RAG Poisoning Defenses

    [https://arxiv.org/abs/2610.33849](https://arxiv.org/abs/2610.33849)

    该论文揭示现有查询时RAG投毒防御的近完美检测性能主要源于投毒集构建中的捷径线索（如将目标问题逐字前置于投毒文档），移除该线索后排序看似完好但部署决策已先崩溃，且多数已发表评估存在构建方式依赖和假阳性率报告缺失的问题。

    

    在检索增强生成（RAG）系统中，对每篇检索到的文档在其进入生成器之前进行筛查的查询时过滤器，是目前已发表的主流防御RAG投毒攻击的方法，且它们在领域基准上报告了近乎完美的检测效果。我们证明这种性能是由投毒集的构建方式决定的，而非过滤器实际读取的内容：移除单一的构建线索，即可在破坏排序之前先摧毁已部署的决策。两个无需标签的统计量可以刻画该线索：包含度，即目标问题在投毒文档中逐字出现的频率，以及长度重叠度。PoisonedRAG发布的代码将目标问题前置于每条插入的段落之前。在我们调查的21项已发表的防御评估中，有8项可验证地使用了以这种方式构建的投毒集，另有5项以长度线索替代；此外，有12项在其实际部署的工作点处未报告任何假阳性率。仅仅从攻击中删除该前缀……（原文摘要至此被截断）

    arXiv:2610.33849v1 Announce Type: cross  Abstract: Query-time filters that screen each retrieved document before it reaches the generator are the dominant published defense against retrieval-augmented generation poisoning, and they report near-perfect detection on domain benchmarks. We show that this performance is predicted by how the poison set was built rather than by what the filter reads: removing a single construction cue destroys the deployed decision before it destroys the ranking. Two label-free statistics of a poison set and its clean corpus profile the cue: containment, how often the target question appears verbatim in the poison, and length overlap. PoisonedRAG's released code prepends the target question to every inserted passage. Of the 21 published defense evaluations we survey, 8 verifiably run poison built this way and 5 substitute a length cue; separately, 12 report no false-positive rate at the operating point they deploy. Deleting only that prefix from the attack au
    
[^149]: 按需定位：学习视觉-语言模型中的模态控制

    Grounding on Request: Learning Modality Control in Vision-Language Models

    [https://arxiv.org/abs/2610.33841](https://arxiv.org/abs/2610.33841)

    该论文提出通过 Flip/Stay 反事实干预来评估视觉-语言模型的模态定位控制能力，并通过为新词嵌入引入视觉与文本定位标识、结合反事实孪生间隔监督，在冻结预训练模型的前提下教会模型按用户请求遵循指定模态的证据。

    

    当图像与其说明文字相互矛盾时，用户能否可靠地控制视觉-语言模型（VLM）遵循哪一方？仅凭准确率无法回答这个问题，因为模型可能凭借参数化知识给出正确答案，而并未遵循所要求的证据。因此，我们通过保留冲突的反事实干预来评估模态定位控制能力。具体而言，Flip 衡量当被请求的模态被编辑时，最初正确的答案是否会随之改变；Stay 衡量当竞争模态被编辑时，答案是否保持稳定。为了学习这种控制能力，我们在新词学习的基础上，为视觉定位和文本定位引入新词，仅训练它们的嵌入，同时保持预训练模型固定不变。我们还提出了反事实孪生间隔监督，利用成对干预来教导模型：当所请求的证据发生变化时，答案概率应如何相应变化。在……（原文摘要在此处截断）

    arXiv:2610.33841v1 Announce Type: cross  Abstract: When an image and its caption disagree, can a user reliably control which one a vision-language model (VLM) follows? Accuracy alone cannot answer this question, since a model might give the correct answer from parametric knowledge without following the requested evidence. We therefore evaluate modality grounding control through conflict-preserving counterfactual interventions. Specifically, Flip measures whether an initially correct answer follows an edit to the requested modality, while Stay measures whether it remains stable when the competing modality is edited. To learn this control, we build on neologism learning by introducing new words for visual and textual grounding, training only their embeddings while keeping the pretrained model fixed. We also introduce Counterfactual Twin-Margin (CTM) supervision, which uses paired interventions to teach how answer probabilities should change when the requested evidence changes. Across thr
    
[^150]: EdgeDeID：一个无需真实患者数据训练的0.6B参数临床去标识化模型

    EdgeDeID: A 0.6B Clinical De-identifier Trained Without Real Patient Data

    [https://arxiv.org/abs/2610.33839](https://arxiv.org/abs/2610.33839)

    提出EdgeDeID——一个仅用合成数据训练、无需真实患者文本即可本地运行的0.6B参数临床去标识化模型，在基准测试中达到0.994微F1且99%的笔记无遗漏，量化后仅需615MB即可在CPU上运行。

    

    使用托管（云端）语言模型对临床笔记进行去标识化处理，反而会泄露这一步骤本想保护的信息。我们提出了EdgeDeID，一个0.6B参数的仅解码器架构去标识化模型，它可以在本地运行，且训练过程完全不使用真实患者文本：三个大语言模型教师撰写合成笔记，由单一标注者将其重新标注为统一规范，模型则学习输出其所识别标识符的类型化XML模式。在一个包含100条笔记、标注经人工验证的合成域内基准测试中，该模型达到0.994的微平均F1分数，并在99%的笔记中实现零标识符遗漏；相比之下，八个现成系统中最强的一个分别仅为0.857和12%。量化后，该模型可在CPU上以615MB的内存运行。在监督训练之外的三项改进中，只有验证器适配器有效：它能以0.94的AUROC区分其他系统输出的完整与不完整结果，而采用召回率加权奖励的强化学习和基于规则的后处理层对域内结果没有带来改善。

    arXiv:2610.33839v1 Announce Type: new  Abstract: De-identifying clinical notes with a hosted language model discloses the information the step is meant to protect. We present EdgeDeID, a 0.6B-parameter decoder-only de-identifier that runs locally and is trained without real patient text: three LLM teachers write synthetic notes, a single labeler re-annotates them into one convention, and the model learns to emit a typed XML schema of the identifiers it finds. On a 100-note synthetic in-domain benchmark with manually verified annotations it reaches 0.994 micro-F1 and leaves no missed identifier in 99% of notes, against 0.857 and 12% for the strongest of eight off-the-shelf systems. Quantized, it runs on CPU in 615 MB. Of three additions to supervised training, only a verifier adapter helps: it separates complete from incomplete outputs of other systems at 0.94 AUROC, whereas reinforcement learning with a recall-weighted reward and a rule-based post-processing layer leave in-domain resul
    
[^151]: EnvLunch：环境作为学习交互的免费午餐

    EnvLunch: The Environment as a Free Lunch for Learning to Interact

    [https://arxiv.org/abs/2610.33817](https://arxiv.org/abs/2610.33817)

    EnvLunch 提出了一种无需人工标注的预热方法，通过让智能体仅凭最终观察结果恢复功能等效的动作轨迹，将可执行环境自身的信号免费转化为训练资源，使模型在下游智能体强化学习中超越基线 9 个百分点以上。

    

    对于智能体式数据分析而言，语言模型智能体通常需要大量标注任务来学习如何与可执行环境（即数据库）进行有效交互。虽然高质量的标注任务需要大量的人工投入，但可执行环境本身已经提供了丰富的交互结构和可验证的结果。从这个意义上说，环境本身提供了一份免费午餐——一组无需人工标注的可验证任务。我们提出了 EnvLunch，这是一个预热阶段，它将这种环境信号转化为智能体训练，而无需人工编写或模型生成的任务描述，可应用于下游的智能体强化学习之前。EnvLunch 采样有效的动作轨迹，执行这些轨迹以获得最终观察结果，然后训练智能体在仅给定最终观察结果的情况下恢复出功能等效的轨迹。这种无标签的预热方法使模型的性能超越基线 9 个百分点以上。

    arXiv:2610.33817v1 Announce Type: new  Abstract: For agentic data analysis, language-model agents often require a large number of annotated tasks to learn how to interact effectively with executable environments (i.e., databases). While high-quality annotated tasks require heavy manual effort, an executable environment already provides rich interaction structure and verifiable outcomes. In this sense, the environment itself offers a free lunch, an additional set of verifiable tasks that require no human annotation. We propose EnvLunch, a warm-up stage that turns this environmental signal into agent training without human-written or model-generated task descriptions, before downstream agentic reinforcement learning. EnvLunch samples valid action trajectories, executes them to obtain final observations, and trains an agent to recover a functionally equivalent trajectory given only its final observation. This label-free warm-up enables the model to outperform the baseline by over 9 points
    
[^152]: 任务格式作为适配中的影响因素：LoRA与全量微调在提示词安全分类上的受控比较

    Task Format as a Factor in Adaptation: A Controlled Comparison of LoRA and Full Fine-Tuning for Prompt-Safety Classification

    [https://arxiv.org/abs/2610.33813](https://arxiv.org/abs/2610.33813)

    该研究通过严格控制变量的受控实验发现，任务格式是影响LoRA与全量微调适配效果的关键因素——在预设定学习率下全量微调于标签训练中领先2-3个百分点，但当LoRA被训练在判定后写出理由（而非等长的无内容文本）时即可消除这一差距，且微调小模型虽在域内占优，在分布外及对抗性越狱场景仍落后于提示的大模型。

    

    arXiv:2610.33813v1 公告类型：新论文 摘要：为分类任务微调语言模型涉及许多设计决策。低秩适配（LoRA）与全量微调的比较很常见，但往往存在混杂因素，且任务格式很少被研究。我们在提示词安全分类任务上比较了这两种微调方式，使用匹配的4亿参数编码器-解码器模型对，涵盖三种任务格式，每个实验设置最多使用15个随机种子，并采用预先设定的分析。在预先设定的学习率下，在所有仅以标签训练的架构和格式中，全量微调均领先LoRA 2-3个百分点。在编码器上，当两者的学习率均由同一搜索选出时，这一差距消失。训练LoRA在给出判定后写出理由可提升4.0个百分点并消除差距，但等长的无内容句子则无此效果。经训练的模型在域内表现优于经提示的80亿参数模型，但在三个分布外语料库中的两个上落后于后者，在对抗性越狱攻击上落后8-23个百分点，并将27-66%的无害但……（原文在此处截断）的内容标记为有害。

    arXiv:2610.33813v1 Announce Type: new  Abstract: Fine-tuning a language model for classification involves many design decisions. Comparisons of low-rank adaptation (LoRA) with full fine-tuning are common but often confounded, and task format is rarely studied. We compare both regimes on prompt-safety classification with a matched 400M encoder-decoder pair (Ettin) across three task formats, with up to 15 seeds per cell and pre-specified analyses. At the pre-specified learning rates, full fine-tuning leads LoRA by 2-3 pp in every architecture and format trained on the label alone. On the encoder, the gap closes once both rates are chosen by the same search. Training LoRA to write a reason after its verdict adds 4.0 pp and closes the gap, but a content-free sentence of equal length does not. Trained models score above a prompted 8B in domain but fall behind it on two of three out-of-distribution corpora, trailing by 8-23 pp on adversarial jailbreaks and flagging 27-66% of benign but harmf
    
[^153]: IntentDelta：一个面向演化用户意图的人工构建基准

    IntentDelta: A Human-Constructed Benchmark for Evolving User Intent

    [https://arxiv.org/abs/2610.33806](https://arxiv.org/abs/2610.33806)

    提出人工构建的多轮基准 IntentDelta（含638条轨迹、3,766个回合），评估显示现有语言模型智能体在跟踪演化用户意图时意图恢复率仅约33%-45%，且难以保留早期需求、区分软偏好与硬约束，正确的意图恢复也不能保证执行成功。

    

    语言模型智能体必须能够适应用户在复杂任务中不断修改的需求和优先级。我们提出了 IntentDelta，一个人工构建的多轮基准，涵盖 WebShop 和 TravelPlanner 两个环境，共包含 638 条轨迹和 3,766 个回合。每个回合将用户话语与累积意图状态及人工参考动作配对。这些轨迹经过交叉验证，涵盖了由可行性和偏好驱动的约束与优先级变化。我们在意图恢复和动作质量两方面评估了十一个智能体。平均意图恢复率在 TravelPlanner 中为 33.1%，在 WebShop 中为 44.7%；平均动作得分分别为 31.6% 和 41.0%。智能体难以保留先前的需求，也难以区分软性偏好与必须满足的硬性约束。此外，正确的意图恢复并不保证成功执行，这凸显了将不断演化的用户意图转化为动作所面临的挑战。

    arXiv:2610.33806v1 Announce Type: new  Abstract: Language-model agents must adapt as users revise their requirements and priorities during complex tasks. We introduce IntentDelta, a human-constructed multi-turn benchmark with 638 trajectories and 3,766 turns across WebShop and TravelPlanner. Each turn pairs a user utterance with a cumulative intention state and human reference actions. The trajectories are cross-verified and cover feasibility- and preference-driven changes in constraints and priorities. We evaluate eleven agents on intention recovery and action quality. Average Intention Recovery is 33.1\% in TravelPlanner and 44.7\% in WebShop, while average Action Score is 31.6% and 41.0%, respectively. Agents struggle to preserve earlier requirements and distinguish soft preferences from must-have constraints. Correct intention recovery also does not guarantee successful execution, highlighting challenges in translating evolving user intentions into actions.
    
[^154]: MTA-Zero：通过在线问答-求解器协同训练实现自我改进的多模态深度搜索

    MTA-Zero: Self-Improving Multimodal Deep Search via Online QA-Solver Co-training

    [https://arxiv.org/abs/2610.33799](https://arxiv.org/abs/2610.33799)

    提出了MTA-Zero框架，通过在线协同训练问答生成器与深度搜索求解器，并利用“恰到好处”奖励机制聚焦求解器的学习前沿，无需预计算多跳语料库即可实现自我改进的多模态深度搜索代理。

    

    多模态大语言模型（MLLMs）已成为强大的视觉推理器，但在需要图像定位、工具选择和整合网络证据的开放世界多步骤搜索方面仍存在局限。现有方法通常构建和过滤固定的离线语料库，其问题分布无法随着目标深度搜索代理能力的提升而自适应调整。基于MTA-Agent的图像定位问答构建方法，我们提出了MTA-Zero，这是一个面向自我改进多模态深度搜索代理的在线多模态问答-求解器协同训练框架，无需预计算的多跳语料库。从图像定位的VQA种子上下文出发，可训练的问答生成器在线检索证据并构建经过验证的多跳问题。随后，一个独立的深度搜索求解器选择工具、收集证据并回答所生成的问题。为了使生成聚焦于求解器的学习前沿，我们为问答生成器推导出一种“恰到好处”奖励机制（原文摘要在此处截断，后续内容不完整）。

    arXiv:2610.33799v1 Announce Type: new  Abstract: Multimodal large language models (MLLMs) have become strong visual reasoners, yet remain limited in open-world, multi-step search that requires grounding images, selecting tools, and integrating web evidence. Existing approaches typically construct and filter fixed offline corpora whose question distributions cannot adapt as the target deep-search agent improves. Building on MTA-Agent's image-grounded QA construction, we present MTA-Zero, an online multimodal QA-Solver co-training framework for self-improving multimodal deep-search agents that requires no precomputed multi-hop corpus. Starting from image-grounded VQA seed contexts, a trainable QA Generator retrieves evidence and constructs verified multi-hop questions online. A separate deep-search Solver then selects tools, gathers evidence, and answers the generated questions. To focus generation on the Solver's learning frontier, we derive a Goldilocks reward for the QA Generator from
    
[^155]: 邻居的低语：重新思考描述共识作为视觉-语言检索的推理时证据

    Whispers from the Neighbors: Rethinking Caption Consensus as Inference-Time Evidence for Vision-Language Retrieval

    [https://arxiv.org/abs/2610.33798](https://arxiv.org/abs/2610.33798)

    该论文提出在推理时将同一图像的相邻描述共识作为证据融合进冻结双编码器的视觉-语言检索中，在同源标注描述条件下可将文本到图像检索R@1提升3.4至16.8个百分点。

    

    一幅图像很少只被描述一次。检索基准和许多数据集为每张图片附加了多个描述，然而双编码器仍然孤立地判断每一个图像-描述对。我们探究当冻结的编码器在推理时参考这些相邻描述时，它们能带来什么贡献。将图像相似度与图像自身描述的相似度进行融合，当描述来自与查询相同的标注者时，可在九个冻结编码器上将文本到图像检索的R@1提升3.4至16.8个百分点。当描述由现成的描述生成器生成时，增益降至1.9至7.2个百分点；而当描述来自不同的标注来源时，增益最多只有一个百分点；使用释义训练的文本编码器可进一步增强文本侧的性能。一种自然的改进方法——对图像中描述未能解释的部分进行评分——在留一法评估下似乎能带来更大的提升。我们证明这一优势之所以产生，是因为正确候选者的…

    arXiv:2610.33798v1 Announce Type: cross  Abstract: An image is rarely described only once. Retrieval benchmarks and many collections attach several captions to each picture, yet dual encoders still judge every image-caption pair in isolation. We ask what these neighboring captions contribute when a frozen encoder consults them at inference time. Fusing image similarity with similarity to an image's own captions improves text-to-image R@1 by 3.4 to 16.8 points across nine frozen encoders when the captions come from the same annotators as the queries. The gain falls to 1.9 to 7.2 points when an off-the-shelf captioner writes them, and to at most one point when they come from a different annotation source; a paraphrase-trained text encoder strengthens the text side further. A natural refinement, scoring the part of the image that the captions leave unexplained, appears to add much more under a leave-one-out evaluation. We show that this advantage arises because the correct candidate's sup
    
[^156]: 一次扫描就够了：面向语言引导导航的无示教自适应

    One Scan is Enough: Demonstration-Free Adaptation for Language-Guided Navigation

    [https://arxiv.org/abs/2610.33794](https://arxiv.org/abs/2610.33794)

    该论文提出用一次简短的iPhone扫描替代人工示教，通过Scan2Instruction自动合成完整训练环境并用PRPT-KS的闭式帕累托效用适配策略，使语言引导导航机器人在真实环境中达到85%的成功率（且在未扫描区域仍达82%），远超基于人类示教训练的49%基线。

    

    在新建筑中部署语言引导导航策略通常意味着眼看着它失败。公开基准上的视觉-语言-动作模型会随着布局、外观和语言的漂移而逐场景性能下降，而标准的解决方案——在现场遥操作采集全新轨迹——无法规模化。我们用一段简短的iPhone扫描替代人类示教。Scan2Instruction将该扫描编译成完整的训练环境，自动合成带有接地指令的可执行导航片段。PRPT-KS随后通过直接从重建场景中读取的闭式帕累托效用来适配策略，无需学习评论家模型，也无需人工标注。在五个真实环境中，这两项设计共同使我们的RealNav达到85%的真实机器人成功率，而基于人类示教后训练的RealNav基线仅为49%；且在同一建筑的未扫描区域仍达到82%，而相同基线仅为41%。我们的工作表...

    arXiv:2610.33794v1 Announce Type: cross  Abstract: Deploying a language-guided navigation policy in a new building usually means watching it fail. Public-benchmark VLAs degrade scene by scene as layout, appearance, and language drift, and the standard fix of teleoperating fresh trajectories on site does not scale. We replace human demonstration with a single brief iPhone scan. Scan2Instruction compiles that scan into a complete training environment, automatically synthesizing executable navigation episodes with grounded instructions. PRPT-KS then adapts the policy through a closed-form Pareto utility read directly off the reconstructed scene, with no learned critic and no human labels. Across five real environments, the two designs together drive our RealNav to an 85% real-robot success rate, compared with 49% for a RealNav baseline post-trained on human demonstrations, and still achieve 82% in unscanned regions of the same buildings, compared with 41% for the same baseline. Our work t
    
[^157]: TriageRAG：一种用于将检索增强语言模型适配到专业领域的故障感知自训练方法

    TriageRAG: Failure-Aware Self-Training for Adapting Retrieval-Augmented Language Models to Specialized Domains

    [https://arxiv.org/abs/2610.33776](https://arxiv.org/abs/2610.33776)

    TriageRAG提出了一种故障感知的自训练方法，通过在有无检索上下文的条件下探测学生模型，并将每个自生成的示例按失败类型（已知、阅读器失败或检索器失败）进行分诊，从而更有针对性地将RAG系统适配到医学等专业领域。

    

    检索增强生成（RAG）使大型语言模型（LLM）能够基于检索到的证据回答知识密集型问题，但在仅有未标注语料库可用的情况下，将通用的RAG系统适配到医学等专业领域十分困难。诸如SimRAG之类的自训练方法通过让LLM根据领域段落自行撰写问答对，并在通过往返一致性过滤的问答对上进行微调来应对这一挑战。我们观察到，这些方法对所有保留下来的示例一视同仁：它们从不询问学生模型是否已经知道答案、是否已经能够从检索到的上下文中正确提取出答案，或者是否失败——如果失败，问题究竟出在阅读器还是检索器上。为了解决这一问题，我们提出了TriageRAG，这是一种故障感知的自训练方法，它在有检索上下文和无检索上下文两种情况下对学生模型在每个自生成的问题上进行探测，并将其分类为（原文在此处截断）

    arXiv:2610.33776v1 Announce Type: cross  Abstract: Retrieval-augmented generation (RAG) lets large language models (LLMs) answer knowledge-intensive questions from retrieved evidence, but adapting a general-purpose RAG system to a specialized domain such as medicine is difficult when only an unlabeled corpus is available. Self-training approaches such as SimRAG address this by letting the LLM write its own question-answer pairs from domain passages and fine-tuning on the pairs that survive a round-trip consistency filter. We observe that these approaches treat every retained example identically: they never ask whether the student already knows the answer, already extracts it correctly from the retrieved context, or fails - and if so, whether the failure lies with the reader or with the retriever. To tackle this, we propose TriageRAG, a failure-aware self-training approach that probes the student on each self-generated question with and without retrieved context, triages it into one of 
    
[^158]: 《系统1模型搭车客指南：超越大语言模型的快速决策》

    The Hitchhiker's Guide to System 1 Models: Fast Decision-Making Beyond LLMS

    [https://arxiv.org/abs/2610.33775](https://arxiv.org/abs/2610.33775)

    本指南提出了系统1决策模型的五大家族分类体系——这类模型通过单次前向传播即可直接输出结构化决策，为现代AI系统中需要亚秒级延迟的常规决策任务提供了比大语言模型更快、更便宜且校准更好的替代方案。

    

    大语言模型在推理和生成方面表现出色，但其自回归的逐token推理方式使其在处理现代AI系统所需的众多亚秒级延迟的常规决策（如路由、门控、评分、分类）时，变得不必要地缓慢、昂贵且校准不佳。一类不断壮大的模型正在填补这一空白：系统1决策模型——一种能够在单次前向传播中将输入直接映射为类型化、结构化决策的架构。本指南为该领域提供了全面的、面向从业者的论述。我们提出了一个五大家族的分类体系，涵盖基于编码器的分类器、奖励模型、非自回归决策引擎、对比决策模型和提取模型，并深入介绍了每个家族的架构、训练方法和部署权衡。贯穿全文的核心主题是校准：我们发展了适当评分规则的理论，并分析了强化学习（摘要原文在此处截断）。

    arXiv:2610.33775v1 Announce Type: new  Abstract: Large Language Models excel at reasoning and generation but their autoregressive, token-by-token inference makes them unnecessarily slow, expensive, and poorly calibrated for the many routine decisions (routing, gating, scoring, classification) that modern AI systems require at sub-second latency. A growing class of models addresses this gap: System 1 decision models, architectures that map an input directly to a typed, structured decision in a single forward pass. This guide provides a comprehensive, practitioner-oriented treatment of the field. We propose a five-family taxonomy covering encoder-based classifiers, reward models, non-autoregressive decision engines, contrastive decision models, and extraction models, and we provide in-depth coverage of each family's architecture, training methodology, and deployment trade-offs. A central theme is calibration: we develop the theory of proper scoring rules, analyze Reinforcement Learning f
    
[^159]: StateWise：教语言模型在多轮对话中追踪并遵守不断演变的约束

    StateWise: Teaching Language Models to Track and Obey Evolving Constraints in Multi-Turn Dialogue

    [https://arxiv.org/abs/2610.33766](https://arxiv.org/abs/2610.33766)

    提出StateWise自我改进方法，通过教大语言模型维护带有墓碑标记的可修订约束状态，并在“状态总结”和“基于状态响应”两项技能上微调与自我训练，解决了多轮对话中有效约束不被遵守和已撤销约束复发两大难题。

    

    对话助手的用户很少在一开始就说明全部需求：他们会添加约束、修改约束、撤销约束、离题后再回归主题。经过指令微调的大语言模型在这种场景下会以两种方式失败：随着对话轮次的累积，仍然有效的约束不再被遵守（持续性失败），以及已被撤销的约束仍被继续执行（复发）。据我们所知，现有的唯一修复方法是一个外部账本，它在每一轮都重新注入一份编译好的规范。我们提出了StateWise，这是一种自我改进的方法，它教会单个大语言模型维护自己的可修订约束状态——一种为已撤销约束设置墓碑标记的类型化状态——从而可以通过可执行的检查器来度量持续性与复发问题，并且可以免费编译出真实值状态。StateWise在两项互补技能上对大语言模型进行微调：将对话总结为其约束状态，以及基于该状态进行条件化响应，然后模型在自己的轨迹上进行自我训练，只保留通过……的轮次（摘要在此处截断）

    arXiv:2610.33766v1 Announce Type: cross  Abstract: Users of conversational assistants rarely state all requirements up front: they add constraints, amend them, withdraw them, digress, and return. Instruction-tuned LLMs fail here in two ways: active constraints stop being honored as turns accumulate (persistence failure), and withdrawn constraints keep being enacted (relapse). The only existing repair we know of is an external ledger that re-injects a compiled specification every turn. We propose StateWise, a self-improving approach that teaches a single LLM to maintain its own revisable constraint state, a typed state with tombstones for revoked constraints, so persistence and relapse are measured by executable checkers and ground-truth states are compiled for free. StateWise fine-tunes the LLM on two complementary skills, summarizing a dialogue into its constraint state and responding conditioned on that state, then self-trains on its own trajectories, keeping only turns that pass an 
    
[^160]: RoleHear：基于伪标签的半监督角色属性流式语音识别

    RoleHear: Pseudo-Labeling for Semi-Supervised Role-Attributed Streaming Speech Recognition

    [https://arxiv.org/abs/2610.33765](https://arxiv.org/abs/2610.33765)

    RoleHear提出一种半监督流水线，利用离线教师换能器和多模态大语言模型生成的伪标签，训练单个流式语音识别模型通过轻量词表内角色标记在转写的同时内联输出说话人角色。

    

    流式自动语音识别（ASR）能够以低延迟进行转写，但产生的词流不带有说话人角色归属信息，而客服分析等实时应用需要将每个词与说话人角色关联起来。角色归属通常是一个独立的阶段，要么采用与词语对齐的离线或流式说话人日志模型，要么使用应用于转写文本的基于文本的大语言模型，且用于训练的角色标注数据十分稀缺。我们提出了RoleHear，一个半监督流水线，它使单一的缓存感知流式换能器（transducer）通过轻量的词表内角色标记内联输出角色：先是一个快速的轮次边界标记，随后在几个词之后输出一个按角色分配的标记。一个离线换能器教师模型经微调成为角色属性识别器后，对未标注音频进行重新标注，而一个多模态大语言模型在缺乏人工金标准标注的地方提供银级角色标签。流式学生模型在上述混合数据上进行微调。（原文摘要在此处截断）

    arXiv:2610.33765v1 Announce Type: new  Abstract: Streaming automatic speech recognition transcribes at low latency but produces an unattributed word stream, while real-time applications such as customer-service analytics need each word tied to a speaker role. Role attribution is usually a separate stage, either an offline or streaming speaker-diarization model aligned to the words or a text-based large language model applied to the transcript, and role labels for training are scarce. We present RoleHear, a semi-supervised pipeline that makes a single cache-aware streaming transducer emit roles inline through lightweight in-vocabulary role tokens: a fast turn-boundary marker followed by a per-role token a few words later. An offline transducer teacher, fine-tuned into a role-attributed recognizer, relabels unlabeled audio, and a multimodal large language model supplies silver role labels where gold annotations are unavailable. The streaming student is fine-tuned on the mixture in a sing
    
[^161]: 精细化翻译评估：在人类水平系统时代重新思考机器翻译评估

    Refined Assessment for Translation Evaluation: Rethinking Machine Translation Evaluation in the Era of Human-Level Systems

    [https://arxiv.org/abs/2610.33749](https://arxiv.org/abs/2610.33749)

    该论文提出 RATE 评估协议及由专业语言学家构建的 WMT24 英俄翻译高质量人工评估数据集，以解决传统机器翻译评估方法（MQM/ESA）在人类水平翻译系统时代难以捕捉细微错误的问题。

    

    随着机器翻译系统接近人类水平质量，传统评估方法难以检测细微的翻译错误。我们批判性地审视了当前黄金标准方法（MQM 和 ESA）的局限性，包括标注者专业水平参差不齐导致的不一致性、过于复杂的分类体系、粗糙的严重程度粒度、重准确性而轻流利度的偏差，以及时间限制。为解决这一问题，我们引入了一个高质量数据集，该数据集由专业语言学家对 WMT24 英俄翻译的人工评估构成。我们表明，无时间压力下的专家评估所产生的结果与标准评估大不相同。为使这些专家能够进行一致且丰富的标注，我们开发了 RATE（精细化翻译评估）协议。RATE 提供了简化的错误分类体系、扩展的严重程度评级，以及平衡准确性与流利度的多维评分。

    arXiv:2610.33749v1 Announce Type: new  Abstract: As machine translation systems approach human-level quality, traditional evaluation methodologies struggle to detect subtle translation errors. We critically examine limitations in current gold-standard approaches (MQM and ESA), including inconsistencies from variable annotator expertise, excessive categorization complexity, coarse severity granularity, accuracy bias over fluency, and time constraints. To address this issue, we introduce a high-quality dataset consisting of human evaluations for English-Russian translations from WMT24, created by professional linguists. We show that expert assessments without time pressure yield substantially different results from standard evaluations. To enable consistent and rich annotation by these experts, we developed the RATE (Refined Assessment for Translation Evaluation) protocol. RATE provides a streamlined error taxonomy, expanded severity ratings, and multidimensional scoring balancing accura
    
[^162]: 超越提示词：基于大语言模型的机器翻译评估器的系统性研究

    Beyond Prompts: A Systematic Study of LLM-Based Machine Translation Evaluators

    [https://arxiv.org/abs/2610.33741](https://arxiv.org/abs/2610.33741)

    本文系统研究了指令设计、少样本示例、骨干模型选择和多步骤工作流对基于大语言模型的机器翻译评估器的影响，发现指令结构、高质量示例、骨干模型的差异化评估特征以及跨模型输出融合是决定评估性能的关键因素。

    

    基于大语言模型的评估器已迅速成为机器翻译评估的标准工具，但塑造其行为的设计选择仍然鲜为人知。我们对指令设计、少样本示例、骨干模型选择以及多步骤工作流如何影响片段级和基于排序的机器翻译评估进行了系统研究。我们在WMT24和WMT25英俄翻译上收集了新的专家RATE标注，并在涵盖另外五个语言对的MQM和ESA标注基准上进行了迁移验证。我们的实验揭示了四个稳健的发现：指令结构塑造评估器行为，高质量示例比示例数量更重要，前沿骨干模型展现出各异的评估器特征（面向召回率、精确率或排序），并且融合多个不同骨干模型的输出优于单一评估器多次调用的设置。尽管评估器设计呈现出巨大的组合空间，但（摘要在此处截断）

    arXiv:2610.33741v1 Announce Type: new  Abstract: LLM-based evaluators have rapidly become a standard tool for machine translation evaluation, yet the design choices that shape their behavior remain poorly understood. We present a systematic study of how instruction design, few-shot demonstrations, backbone model choice, and multi-step workflows affect span-level and ranking-based MT evaluation. We collect new expert RATE annotations on WMT24 and WMT25 English-Russian translations, with transfer validation on MQM- and ESA-annotated benchmarks across five additional language pairs. Our experiments reveal four robust findings: instruction structure shapes evaluator behavior, high-quality demonstrations matter more than quantity, frontier backbones exhibit distinct evaluator profiles (recall-, precision-, or ranking-oriented), and fusing outputs across diverse backbones outperforms single-evaluator multi-call setups. Although evaluator design exposes a large combinatorial space, several di
    
[^163]: 大语言模型介导的ADHD自我诊断中的性别关联

    Gender Associations in LLM-Mediated ADHD Self-Diagnosis

    [https://arxiv.org/abs/2610.33697](https://arxiv.org/abs/2610.33697)

    本研究将语境关联测试应用于GPT-4o的ADHD自我诊断场景，发现显性性别线索会触发过度纠正的偏见缓解机制，而隐性性别线索则暴露出模型默认的性别偏见行为。

    

    arXiv:2610.33697v1 公告类型：新论文 摘要：大语言模型（LLM）正越来越多地被用于医疗自我诊断。然而，人口统计学线索如何在这些交互中影响模型输出，目前仍知之甚少。这对于ADHD（注意缺陷多动障碍）的诊断尤为重要，因为ADHD诊断中持续存在的性别差异与临床证据并不完全一致。我们改编了语境关联测试，以评估在零样本自我诊断的场景下，GPT-4o是否会响应性别化的ADHD关联。我们发现，显性提示会持续放大某些ADHD亚型的关联，而隐性提示则产生接近代理对齐的分布。这些分歧表明，显性的人口统计学线索可能会激活偏见缓解机制，导致过度纠正而非真正消除偏见，而隐性线索似乎反映了模型的默认行为。我们的研究结果表明，评估LLM介导的自我诊断中的性别偏见需要仔细关注（摘要在此处截断）

    arXiv:2610.33697v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly used for medical self-diagnosis. However, the way demographic cues shape outputs in these interactions remains poorly understood. This is particularly consequential for diagnoses like ADHD, where persistent gender disparities are not fully aligned with the clinical evidence. We adapt the Context Association Test (CAT) to evaluate whether GPT-4o responds to gendered ADHD associations in a zero-shot, self-diagnostic setting. We show that explicit cueing consistently amplifies certain ADHD subtype associations while implicit cueing produces near proxy-aligned distributions. These divergences suggest that explicit demographic cues may activate bias-mitigation mechanisms that over-correct rather than eliminate bias, whereas implicit cues appear to reflect the model's default behavior. Our results demonstrate that evaluating gendered bias in LLM-mediated self-diagnosis requires careful attention to
    
[^164]: 基于嵌入的教育分类法

    Embedding-Based Educational Taxonomies

    [https://arxiv.org/abs/2610.33694](https://arxiv.org/abs/2610.33694)

    本文提出通过对数学题目文本嵌入进行聚类来构建基于嵌入的教育分类法，并与基于共同核心州立标准的知识组件分类法进行对比，在深度知识追踪任务中验证了其有效性。

    

    教育分类法将教育内容（如练习题）组织成概念，这些概念也被称为技能或知识组件。这些分类法在智能辅导系统等教育应用中定义学习目标。它们通过人类专业知识和迭代完善而开发，同时也为课程、评估以及学习者模型提供了基础结构。语义文本嵌入领域的最新进展使得从内容本身以计算方式构建替代性教育分类法成为可能。在本文中，我们通过对数学题目的嵌入进行聚类来构建基于嵌入的教育分类法，从而研究这一可能性。我们在两个ASSISTments数据集中将这些分类法与现有的基于共同核心州立标准（CCSS）的知识组件分类法进行比较，并在深度知识追踪这一依赖分类法的下游任务中评估它们的效用。

    arXiv:2610.33694v1 Announce Type: cross  Abstract: Educational taxonomies organize educational content, such as exercise problems, into concepts, which are also called skills or knowledge components. These taxonomies define learning objectives in educational applications such as intelligent tutoring systems. Developed through human expertise and iterative refinement, they also provide fundamental structure for curricula, assessment, and models of student learning. Recent advances in semantic text embeddings make it possible to construct alternative educational taxonomies computationally from the content itself. In this paper, we investigate this possibility by clustering embeddings of math questions to construct embedding-based educational taxonomies. We compare these taxonomies with the existing Common Core State Standards (CCSS)-based knowledge-component taxonomies in two ASSISTments datasets and evaluate their utility in Deep Knowledge Tracing, a downstream task that relies on taxon
    
[^165]: PathKGen：病理学知识辅助的多模态大语言模型用于可解释的组织病理学分析

    PathKGen: Pathology Knowledge Assisted MLLM for Explainable Histopathology

    [https://arxiv.org/abs/2610.33691](https://arxiv.org/abs/2610.33691)

    提出PathKGen，一种利用知识图谱为多模态大语言模型注入外部结构化病理知识并使推理过程可视化的方法，从而实现可解释的自动化组织病理学报告生成。

    

    全切片图像（WSI）分析是数字组织病理学中癌症临床诊断的基础方法。多模态大语言模型（MLLM）的最新进展使得最先进的自动化组织病理学报告生成成为可能。然而，MLLM纯粹通过编码在其神经参数中的分布进行推理，在给定模型规模的条件下，这限制了它们所能容纳的知识量和推理逻辑。其黑盒推理方式也对人类理解中间推理过程构成了另一个挑战。我们通过利用知识图谱为MLLM注入外部结构化知识以供其推理，同时使这一推理组件可视化，来解决上述问题。知识图谱能够部分外化医学知识的承载，而这一方向在病理报告生成领域仍未被充分探索。我们提出了PathKGen，一种用于组织病理学的知识图谱辅助的多模态大语言模型。

    arXiv:2610.33691v1 Announce Type: cross  Abstract: Whole Slide Image (WSI) analysis serves as the fundamental method for the clinical diagnosis of cancer in digital histopathology. Recent advances in Multimodal Large Language Models (MLLMs) have enabled state-of-the-art (SOTA) automated histopathology report generation. However, MLLMs reason purely through distributions encoded in their neural parameters, limiting the amount of knowledge and reasoning logic they can contain given the model scale. Their black-box reasoning poses another challenge to the human interpretation of intermediate reasoning. We address these issues by utilizing knowledge graphs to inject external structured knowledge for MLLMs to reason over, while making this reasoning component visible. Knowledge graphs are capable of partially externalizing the containment of medical knowledge, an area that remains underexplored in pathology report generation. We introduce PathKGen, a knowledge graph-assisted MLLM for histop
    
[^166]: Meta-CLIP：基于元数据条件化对比学习的零样本疾病推断

    Meta-CLIP: Metadata-Conditioned Contrastive Learning for Zero-Shot Disease Inference

    [https://arxiv.org/abs/2610.33674](https://arxiv.org/abs/2610.33674)

    Meta-CLIP提出了一种元数据条件化的对比学习框架，利用DICOM采集相似性对跨模态负样本重新加权以提升零样本疾病推断性能，且推理时无需元数据。

    

    医学视觉-语言预训练从配对的图像和放射学报告中学习可迁移的表示，然而扫描仪和采集协议的差异可能会损害迁移效果。现有方法主要致力于丰富临床语义，而DICOM头部中的结构化采集信息却未被充分利用。我们提出了Meta-CLIP，一种元数据条件化的对比学习框架，在预训练过程中利用采集相似性对跨模态负样本进行重新加权。基于数值型和分类型DICOM字段的核函数定义了一个有界的成对边际。该边际增加了采集相似负样本的贡献，强调了共享相同成像条件的研究之间的区分。该核函数在图像-文本优化过程中保持固定，推理时遵循标准的双编码器架构，无需元数据。我们通过零样本分类任务在胸部X光片和体积胸部CT上评估了Meta-CLIP。

    arXiv:2610.33674v1 Announce Type: new  Abstract: Medical vision--language pretraining learns transferable representations from paired images and radiology reports, yet differences in scanners and acquisition protocols can impair transfer. Existing approaches primarily enrich clinical semantics, leaving structured acquisition information in DICOM headers underused. We introduce Meta-CLIP, a metadata-conditioned contrastive framework that uses acquisition similarity to reweight cross-modal negatives during pretraining. A kernel over numerical and categorical DICOM fields defines a bounded pairwise margin. This margin increases the contribution of acquisition-similar negatives, emphasizing distinctions among studies that share imaging conditions. The kernel remains fixed during image--text optimization, and inference follows the standard dual-encoder architecture without requiring metadata. We evaluate Meta-CLIP on chest radiography and volumetric chest CT through zero-shot classification
    
[^167]: 大语言模型中偏好与不完美决策的联合建模

    Jointly Modeling Preferences and Imperfect Decision-Making in Large Language Models

    [https://arxiv.org/abs/2610.33651](https://arxiv.org/abs/2610.33651)

    该论文提出偏好-行为模型（PBM），将偏好表示与不完美效用最大化机制相结合，并通过逆博弈问题从观测行为中联合推断LLM的偏好与行为参数，以预测其在不同策略环境中的表现。

    

    大语言模型（LLM）在标准偏好引出任务中表达的偏好能否预测其策略性行为？与人类和其他AI系统交互的LLM智能体面临可能相互冲突的目标，例如个人福利、群体福利和环境问题。先前的工作通过行为经济学与心理学的引出方法或策略博弈来刻画LLM的偏好，但很少测试不同设置之间的一致性，以及在其他策略环境中的预测能力。这需要对相互竞争的目标如何共同塑造行为进行建模，而现有方法通常孤立地测量各个偏好维度。借鉴效用理论，我们提出了偏好-行为模型，该模型将偏好表示与刻画不完美效用最大化的机制相结合。我们将模型估计表述为一个逆博弈问题，从观测到的行为中联合推断偏好与行为参数。

    arXiv:2610.33651v1 Announce Type: new  Abstract: Do preferences Large Language Models (LLMs) express in standard elicitation tasks predict their strategic behaviour? LLM agents interacting with humans and other AI systems face potentially conflicting objectives, such as individual welfare, group welfare, and environmental concerns. Prior work characterizes LLM preferences through behavioral-economic and psychological elicitation or strategic games, but rarely tests consistency across settings and prediction in other strategic environments. This requires modeling how competing objectives jointly shape behavior, whereas existing approaches typically measure preference dimensions in isolation. Drawing on utility theory, we introduce the Preference-Behavior Model (PBM), which combines a preference representation with mechanisms capturing imperfect utility maximization. We formulate estimation as an inverse-game problem, jointly inferring preferences and behavioral parameters from observed 
    
[^168]: PreView：利用预处理的视图与线索增强GUI定位

    PreView: Enhancing GUI Grounding with Preprocessed Views and Cues

    [https://arxiv.org/abs/2610.33645](https://arxiv.org/abs/2610.33645)

    PreView通过一个轻量级辅助预处理模型为冻结的GUI定位模型生成任务相关的局部放大视图以及视觉与文本线索，无需重新训练模型即可提升GUI定位的准确性。

    

    GUI定位（GUI grounding）需要在杂乱的截图中定位小型控件，并识别自然语言指令所指的是哪个控件。裁剪和放大可以暴露局部细节，但所得图像可能仍会让目标控件存在歧义。我们提出PreView，一个辅助预处理模块，它可以在无需进一步训练现有定位模型的情况下改进其输入。PreView准备两类互补的输入特征：用于定位的任务相关视图，以及用于目标理解的候选特定线索。一个经过训练的轻量级辅助模型预测并精修中心点，将截图裁剪为局部视图，并放大该视图以实现放大后的定位。它还准备一个视觉线索（即带有位置提示的红圈标记）和一个描述候选控件身份或功能的文本线索。被冻结的定位模型使用这些准备好的输入来预测最终的点击位置。在……（摘要原文此处被截断）

    arXiv:2610.33645v1 Announce Type: cross  Abstract: GUI grounding requires locating small controls in cluttered screenshots and identifying which control a natural-language instruction refers to. Cropping and enlargement can expose local detail, but the resulting image may still leave ambiguity about the intended target. We propose PreView, an auxiliary preprocessor that improves the inputs to an existing grounding model without further training that model. PreView prepares two complementary types of input features: task-relevant views for localization and candidate-specific cues for target understanding. A trained lightweight auxiliary model predicts and refines a center, crops the screenshot into a local view, and enlarges that view for zoomed-in grounding. It also prepares a visual cue (i.e., a red-circle marker with a location hint) and a textual cue describing the candidate's identity or function. The frozen grounding model uses the prepared inputs to predict the final click. Acros
    
[^169]: 花拉子密遗产继承学（'ilm al-farā'iḍ）的模型论形式化

    A Model-Theoretic Formalization of AlKhwaremis ilm alfaraid

    [https://arxiv.org/abs/2610.33629](https://arxiv.org/abs/2610.33629)

    本文用模型论方法将花拉子密的伊斯兰遗产继承规则形式化为有理数上的线性等式与不等式系统，证明其可由傅里叶-莫茨金消元法精确判定，并确立了规范继承结构 V_d 及其完备的一阶理论。

    

    本文在计算法理学的框架下，对花拉子密关于遗产继承问题的处理方法给出了模型论与计算层面的形式化。我们首先证明，某一特定类别中的每一种继承配置都可以表示为一个由有理数上的线性等式与不等式构成的有限系统，并且此类系统可以精确地通过 Q 上的傅里叶-莫茨金（Fourier-Motzkin）消元法加以判定。随后，我们定义了具有特定权利主张人坐标的固定维度继承结构 V_d，并证明每一个容许的有理线性坐标实现均与 V_d 同构，从而确立了该代数表示在此类别中的规范性（正则性）。我们通过构造证明了 V_d 的完整一阶理论 T_d 的完备性，并明确将其与继承相关片段的可判定性区分开来。最后，本文将这一方法论原则与通过对封闭法理学领域进行穷尽分析而实现的更广泛形式化纲领联系起来。

    arXiv:2610.33629v1 Announce Type: cross  Abstract: This paper presents a model theoretic and computational formalization of AlKhwarizmis treatment of inheritance within the framework of Computational Fiqh. We first show that every inheritance configuration in a specified class is representable as a finite system of rational linear equalities and inequalities, and that such systems are decided exactly by Fourier Motzkin elimination over Q. We then define the fixed dimensional inheritance structure Vd with distinguished claimant coordinates, and prove that every admissible rational linear coordinate realization is isomorphic to Vd, this establishes the algebraic representation as canonical within that class. The complete first order theory Td of Vd is shown to be complete by construction and is explicitly distinguished from the decidability of the inheritance relevant fragment. The methodological principle is then related to a broader formalization of closed Fiqh domains via exhaustive c
    
[^170]: RSI Arena：迈向基于人类反馈的递归自我改进

    RSI Arena: Towards Recursive Self-Improvement with Human Feedback

    [https://arxiv.org/abs/2610.33624](https://arxiv.org/abs/2610.33624)

    RSI Arena提出了一种两阶段自主后训练研究，让十个前沿AI模型作为训练智能体在有限算力和预算下自主训练语言模型并结合人类反馈进行改进，最强模型在能力基准上弥合了基座到Instruct差距的76%，是迈向递归自我改进的重要一步。

    

    AI智能体能否自主训练有用的语言模型，并利用用户反馈对其进行改进？我们提出了RSI Arena，这是一项关于自主后训练的两阶段研究，是迈向递归自我改进的一步。十个前沿模型充当训练智能体，每个智能体都从NVIDIA Nemotron 3.5 Lightning 30B-A3B Base出发。在为期六天的第一阶段中，每个智能体在共享的64-GPU集群上拥有1,000 GPU小时和300美元的API额度，独立选择数据、实施训练流程并挑选检查点。我们通过20个基准测试和盲测人类评估对这些模型进行评估。所有十个智能体都产出了可用的助手，其中最强的模型由GPT-6 Astra训练，在12个能力基准测试中平均弥合了基座模型到Instruct模型差距的76%。在对回答长度和格式进行统计调整后，其人类偏好评分也接近NVIDIA Instruct。在第二阶段，每个智能体接收比较……（摘要在此处截断）

    arXiv:2610.33624v1 Announce Type: cross  Abstract: Can AI agents autonomously train useful language models and improve them using feedback from users? We introduce RSI Arena, a two-stage study of autonomous post-training as a step toward recursive self-improvement. Ten frontier models act as training agents, each starting from NVIDIA Nemotron 3.5 Lightning 30B-A3B Base. During a six-day first stage, each agent has 1,000 GPU-hours and $300 in API credit on a shared 64-GPU cluster, independently selecting data, implementing training procedures, and choosing checkpoints. We evaluate their models on 20 benchmarks and through blind human evaluations. All ten agents produce working assistants, with the strongest model, trained by GPT-6 Astra, closing a median of 76% of the base-to-Instruct gap across 12 capability benchmarks. Its human-preference rating is also close to NVIDIA Instruct after statistical adjustment for answer length and formatting. In Stage 2, each agent receives the comparis
    
[^171]: Mycelium：面向来源、类型学与粒度的幻觉检测元研究

    Mycelium: A Meta-Study of Hallucination Detection Across Provenance, Typology, and Granularities

    [https://arxiv.org/abs/2610.33608](https://arxiv.org/abs/2610.33608)

    该论文将幻觉检测视为以基准为条件的测量问题，提出从来源、类型学和粒度三个维度对现有基准进行组织与元分析，揭示基准构建与评估选择如何影响幻觉检测任务本身及检测器性能的可比性。

    

    幻觉检测对于大型语言模型的评估与部署正变得日益重要。然而，现有方法通常是在以截然不同的方式定义和衡量幻觉的各类基准上进行比较的，这使得所报告的分数难以在不同设置之间进行解读。我们将幻觉检测视为一个以基准为条件的测量问题来研究，并沿三个维度对现有基准进行组织：来源，它描述模型输出和证据条件是如何获得的；类型学，它刻画基准标签所代表的事实性与忠实性失败；粒度，它规定了幻觉被标注、预测和评估的单元。利用这一框架，我们分析了基准构建、标注和评估选择如何塑造底层预测任务，并将检测器的机制与假设和其性能被评测的具体设置联系起来。

    arXiv:2610.33608v1 Announce Type: cross  Abstract: Hallucination detection is increasingly important for evaluating and deploying large language models. However, existing methods are often compared across benchmarks that define and measure hallucination in substantially different ways, making reported scores difficult to interpret across settings. We study hallucination detection as a benchmark-conditioned measurement problem and organize existing benchmarks along three axes: provenance, which describes how model outputs and evidence conditions are obtained; typology, which captures the factuality and faithfulness failures represented by benchmark labels; and granularity, which specifies the unit at which hallucinations are annotated, predicted, and evaluated. Using this framework, we analyze how benchmark construction, annotation, and evaluation choices shape the underlying prediction task, and relate detector mechanisms and assumptions to the settings in which their performance is me
    
[^172]: 言为心声：基于心理引导证据推理的社交痕迹抑郁严重程度估计

    Words Are the Voice of the Mind: Depression Severity Estimation from Social Traces via Mental-Guided Evidence Reasoning

    [https://arxiv.org/abs/2610.33586](https://arxiv.org/abs/2610.33586)

    该论文提出了UMer框架，通过从用户历史社交痕迹中提炼个性化心理先验，并结合心理引导的证据检索与细粒度推理，克服了现有方法在心理先验建模、语义检索和证据推理上的不足，实现基于社交媒体的抑郁严重程度估计。

    

    抑郁严重程度估计对于心理健康评估与干预至关重要，然而传统方法依赖于需要用户主动参与的临床访谈或自评问卷。随着在线社交媒体的广泛使用，用户留下了可能反映其心理状态的丰富社交痕迹，这催生了基于社交媒体的抑郁严重程度估计任务。现有工作已从基于文本特征的分类发展到利用大语言模型（LLM）对标准化量表条目进行问卷填答，但在用户特定心理先验建模、多样化语义检索以及对心理证据的细粒度推理方面仍存在不足。为了解决这些局限性，我们提出了UMer，一个面向基于社交痕迹的抑郁严重程度估计的用户特定心理证据推理框架。UMer首先从历史社交痕迹中提炼出用户特定的心理先验，以捕捉用

    arXiv:2610.33586v1 Announce Type: new  Abstract: Depression severity estimation is crucial for mental health assessment and intervention, yet traditional approaches depend on clinical interviews or self-report questionnaires that require users' active participation. With the widespread use of online social media, users leave rich social traces that may reflect their mental states, motivating the task of social media-based depression severity estimation. Existing efforts have progressed from textual-feature-based classification to LLM-based questionnaire completion over standardized scale items, yet they still fall short in user-specific mental priors modeling, diversified semantic retrieval, and fine-grained reasoning over mental evidence. To address these limitations, we propose UMer, a user-specific mental evidence reasoning framework for social traces-based depression severity estimation. UMer first distills a user-specific mental prior from historical social traces to capture the u
    
[^173]: 当最弱的语言不断变动时：面向多语言 BabyLM 的 ARMOR-DP

    When the Weakest Language Keeps Moving: ARMOR-DP for Multilingual BabyLM

    [https://arxiv.org/abs/2610.33583](https://arxiv.org/abs/2610.33583)

    提出 ARMOR-DP 动态采样框架，利用各语言验证损失实时调整多语言 BabyLM 训练中的语言采样比例，应对“最弱语言”随训练动态变化的问题，在减少中文训练轮次的情况下取得了优于静态保护策略的性能与语言平衡权衡。

    

    多语言 BabyLM 系统通常采用固定的语言采样比例，尽管在训练过程中最弱的语言可能会发生变化。我们提出了 ARMOR-DP，一个模块化框架，它能够诊断源于表示、暴露量、模型容量和目标函数压力的不平衡问题，并利用各语言的验证损失来更新采样策略。受控实验表明，相同的字符预算会给中文带来严重的表示伪影；字节级编码消除了其中大部分伪影，而静态的中文保护策略随后可能使英语成为最弱的语言。动态控制器能够跟踪这些转变，并取得比固定保护更好的分数—平衡权衡。在官方评测中，采用课程动态机制的检查点在性能上优于与之匹配的静态中文保护基线，同时使用的中文训练轮次更少；扩大模型容量则产生了最强的内部 ARMOR-DP 系统。这些结果支持一个简单的结论：多语言弱势……

    arXiv:2610.33583v1 Announce Type: new  Abstract: Multilingual BabyLM systems usually fix language sampling proportions even though the weakest language can change during training. We present ARMOR-DP, a modular framework that diagnoses imbalance arising from representation, exposure, capacity, and objective pressure, and uses per-language validation loss to update sampling. Controlled experiments show that equal character budgets create a severe representation artifact for Chinese; byte-level encoding removes much of it, while static Chinese protection can subsequently make English the weakest language. The dynamic controller tracks these transitions and achieves a better score--balance tradeoff than fixed protection. On official evaluation, curriculum-dynamic checkpoints outperform a matched static Chinese-protective baseline while using fewer Chinese epochs; scaling capacity yields the strongest internal ARMOR-DP system. These results support a simple conclusion: multilingual weaknes
    
[^174]: Trajectory2Gym：面向工具使用智能体的决策扎根式组合交互合成

    Trajectory2Gym: Decision-Grounded Compositional Interaction Synthesis for Tool-Use Agents

    [https://arxiv.org/abs/2610.33568](https://arxiv.org/abs/2610.33568)

    该论文提出Trajectory2Gym框架，通过围绕组合覆盖不足的结构合成可执行交互、仅从决策时有可见信息支撑的动作中提取监督信号，并采用父节点归一化的监督微调，为工具使用智能体生成更高质量的训练轨迹监督。

    

    成功的工具使用轨迹未必是有用的监督信号。即使是成功的交互，也可能未能充分表示多步骤依赖关系，或者包含在决策时基于智能体可见信息无法被证成的动作。我们识别出有效轨迹监督的两个相互耦合的需求：组合覆盖性，即重要的观察-动作依赖关系得到充分表示；以及决策扎根性，即每个被监督的动作都由学习者可见的交互历史所支撑。我们提出Trajectory2Gym，它围绕支持不足的组合结构构建可执行的交互，并仅从这些交互实际执行中的扎根决策里提取监督信号。结构支持选择负责分配所产生的经验，而父节点归一化的监督微调则以语义交互作为训练权重的单位。这将轨迹生成重新构建为审慎的……（原文在此处截断）

    arXiv:2610.33568v1 Announce Type: new  Abstract: Successful tool-use trajectories are not necessarily useful supervision. Even successful interactions may underrepresent multi-step dependencies or contain actions that are unjustified by the information visible to the agent at decision time. We identify two coupled requirements for effective trajectory supervision: compositional coverage, so that important observation-action dependencies are represented, and decision grounding, so that each supervised action is supported by the learner-visible interaction history. We introduce Trajectory2Gym, which constructs executable interactions around under-supported compositional structures and derives supervision only from grounded decisions in their realized executions. Structural-support selection allocates the resulting experience, while parent-normalized supervised fine-tuning preserves the semantic interaction as the unit of training weight. This reframes trajectory generation as the deliber
    
[^175]: SEDATE：大型语言模型是否在跨语言间共享攻击性回路？

    SEDATE: Do Large Language Models Share an Aggression Circuit Across Languages?

    [https://arxiv.org/abs/2610.33542](https://arxiv.org/abs/2610.33542)

    SEDATE提出了一种免训练的“检测-编辑-验证”方法，通过定位并编辑与攻击性相关的神经元，使大型语言模型在7种语言及分布外场景中均显著提升了安全性。

    

    随着人工智能的应用在各种多模态和多语言领域不断扩展，确保这些系统安全且负责任地部署已成为一个重要问题。为使AI响应更加安全可靠，研究者们已提出多种干预方法，但这些方法通常需要高昂的计算或标注成本，或无法扩展到多语言场景。在本工作中，我们提出了选择性检测与定向编辑（SEDATE），这是一种基于“检测-编辑-验证”策略的免训练方法。SEDATE首先检测出负责相关行为的神经元，在必要时对其进行编辑，最后验证模型响应，全程无需任何额外的模型训练。我们在7种语言和4个模型家族上对SEDATE进行了评估，并进一步在4种分布外语言和4个额外数据集上验证了其有效性。结果表明，SEDATE在所有语言上都取得了显著更优的安全分数。

    arXiv:2610.33542v1 Announce Type: cross  Abstract: As the applications of AI continue to expand across various multimodal and multilingual domains, ensuring safe and responsible deployment of these systems has become an important concern. A wide-range of interventions have been proposed to make AI responses safer and more reliable. They often incur substantial computational or annotation costs, or cannot be extended to multilingual settings. In this work, we propose Selective Detection And Targeted Editing (SEDATE), a training-free approach based on detect-edit-verify strategy. SEDATE first detects the responsible neurons, edits them as and when necessary and finally verifies the model responses; all without any additional model training. We evaluate SEDATE across 7 languages and 4 model families. We further evaluate its effectiveness across 4 out-of-distribution languages and 4 additional datasets. We show that SEDATE achieves substantially superior safety scores across all languages 
    
[^176]: SkillShift-Bench：面向Web智能体的环境演化下技能学习基准测试

    SkillShift-Bench: Benchmarking Skill Learning under Environment Evolution for Web Agents

    [https://arxiv.org/abs/2610.33528](https://arxiv.org/abs/2610.33528)

    提出首个评估Web智能体技能学习在环境演化下鲁棒性的基准SkillShift-Bench，通过模拟上下文、感知和执行三类受控变化，系统揭示了技能复用能力的显著退化。

    

    技能学习使Web智能体能够复用先前经验来完成长程交互任务。然而，已学到的技能通常在封闭环境中进行评估，忽略了现实世界的Web部署中调用上下文、感知基础和执行可行性会持续变化这一事实。为此，我们提出了SkillShift-Bench，这是首个用于评估受控开放环境变化下技能学习的基准。SkillShift-Bench在三个自托管的Web系统上模拟了上下文变化、感知变化和执行变化，并通过涵盖任务成功、交互成本、技能使用和技能失败的诊断指标来评估智能体。全面的实验揭示了在受控变化下智能体鲁棒性的系统性下降，表现为更低的成功率、更高的交互成本和更频繁的技能失败，其中执行变化带来的压力最强。我们进一步制定了……（原文摘要似乎被截断）

    arXiv:2610.33528v1 Announce Type: new  Abstract: Skill learning enables web agents to reuse prior experience for long-horizon interactive tasks. However, learned skills are typically evaluated in closed environments, overlooking the fact that real-world web deployment involves continuous shifts in invocation context, perceptual grounding, and execution feasibility. To this end, we introduce SkillShift-Bench, the first benchmark for evaluating skill learning with controlled open-environment shifts. SkillShift-Bench simulates Contextual Shift, Perceptual Shift, and Execution Shift across three self-hosted web systems, and evaluates agents with diagnostics covering task success, interaction cost, skill use, and skill failure. Comprehensive experiments reveal systematic robustness degradation under controlled shifts, characterized by lower success rates, higher interaction costs, and more frequent skill failures, with Execution Shift exerting the strongest pressure. We further formulate sh
    
[^177]: 隐性信念作为战略说服的可扩展过程奖励

    Implicit Beliefs as Scalable Process Rewards for Strategic Persuasion

    [https://arxiv.org/abs/2610.33527](https://arxiv.org/abs/2610.33527)

    该论文提出PersuArena说服模拟环境和Probe-RPO在线强化学习框架，通过将冻结的被说服者的隐性内部信念转化为密集的过程奖励，解决了大语言模型多轮战略说服训练中模拟真实性不足和信用分配监督稀疏的两大难题。

    

    训练大语言模型（LLM）进行多轮战略说服面临两个相互耦合的障碍：（一）对具有心理学基础的被说服者进行逼真模拟，（二）在战略对话中为信用分配提供密集且忠实的监督信号。仅基于最终结果的奖励是稀疏的，无法说明是哪些对话举动促成了成功；而基于LLM裁判的过程奖励则成本高昂，且与底层交互动态脱节。我们用一个统一框架来应对这两个挑战。首先，我们提出了PersuArena，一个能够产生逼真的、非单调轨迹的说服模拟环境。其次，直接构建在该环境之上，我们提出了Probe-RPO（通过在线信念引出实现相对策略优化的过程奖励），这是一个在线强化学习框架，能够将被说服者的内部状态转化为密集的过程奖励。在每一轮对话中，系统从冻结的被说服者内部状态读取三个内省探测信号……（摘要原文截断）

    arXiv:2610.33527v1 Announce Type: cross  Abstract: Training large language models (LLMs) for multi-turn strategic persuasion faces two coupled obstacles: (i) realistic simulation of psychologically grounded persuadees, and (ii) dense, faithful supervision for credit assignment in strategic dialogues. Outcome-only rewards are sparse and uninformative about which conversational moves drove success, while LLM-judge process rewards are expensive and decoupled from the underlying interaction dynamics. We address both challenges with a unified framework. First, we introduce PersuArena, a persuasion simulation environment that produces realistic, non-monotonic trajectories. Second, building directly on this environment, we propose Process Rewards via Online Belief Elicitation for Relative Policy Optimization (Probe-RPO), an online RL framework that converts the persuadee's internal state into dense process rewards. At each turn, three introspective probes are read from the frozen persuadee's 
    
[^178]: 一个模型，两条路径：平衡语言模型中单次传递与循环推理的性能

    One Model, Two Paths: Balancing Single-Pass and Looped Performance in Language Models

    [https://arxiv.org/abs/2610.33517](https://arxiv.org/abs/2610.33517)

    该论文提出一种基于加权损失的联合训练策略，并借助冻结原始模型的指导，使预训练语言模型能够通过复用中间层权重支持循环推理，同时将单次传递推理的质量退化控制在预定义容差范围内。

    

    重复使用Transformer层可以在不增加参数的情况下提升语言模型的性能。通过复用中间层权重进行额外计算，可以将预训练模型改造为同时支持单次传递和循环推理。由于两条路径共享权重，为改进循环预测而进行的训练可能会降低单次传递的质量。我们研究了联合训练策略，对两个输出使用加权损失，并借助冻结的原始模型进行指导。与仅进行循环训练相比，联合监督显著减少了单次传递的性能退化，同时保留了相对于普通继续训练的循环增益。这一收益大部分来自调整单次传递损失的权重，而冻结原始模型的指导进一步改进了循环输出。通过验证集选择的权重，我们的方法在预定义容差范围内保持了与普通继续训练相当的单次传递质量，同时提升了循环预测的性能。

    arXiv:2610.33517v1 Announce Type: new  Abstract: Repeating Transformer layers can improve language-model performance without adding parameters. A pretrained model can be adapted to support both single-pass and looped inference by reusing middle-layer weights for additional computation. Because both paths share weights, training to improve looped predictions can degrade single-pass quality. We study joint training strategies using weighted losses on both outputs, with guidance from the frozen original model. Joint supervision substantially reduces single-pass degradation compared to looped-only training, while preserving looped gains over ordinary continued training. Most of this benefit comes from adjusting the single-pass loss weight, with guidance from the frozen original model further improving the looped output. With validation-selected weights, our method preserves single-pass quality within a predefined tolerance of ordinary continued training while improving looped predictions.
    
[^179]: 重视精确率：精确率目标缓解基于评分标准的强化学习中的奖励作弊问题

    Eat Your Precision: Precision Objective Mitigates Reward Hacking in Rubric-based RL

    [https://arxiv.org/abs/2610.33512](https://arxiv.org/abs/2610.33512)

    提出FORK奖励，通过联合优化召回率与新颖的精确率目标，缓解基于评分标准的强化学习中的奖励作弊问题，使模型回复既完整又不冗长且事实正确。

    

    基于评分标准的奖励使强化学习能够在不可验证的任务上有效进行。然而，由于这些奖励通常被定义为评分标准覆盖率的加权度量，模型主要被训练去最大化召回率目标。因此，经强化学习训练的模型生成的回复虽然更加完整，但也更加冗长，并包含更多不必要或事实错误的内容，导致这些回复不如基础模型的回复受人青睐。为了解决这个问题，我们提出了FORK奖励，它联合优化召回率和一个新颖的精确率目标，该目标要求模型回复的每个单元既必要又事实正确，分别通过基于评分标准的必要性评分和事实正确性评估器来衡量。实验结果表明，FORK优于现有的基于评分标准的强化学习方法，并实现了平衡评分标准覆盖率与整体回复质量的帕累托最优性能。

    arXiv:2610.33512v1 Announce Type: new  Abstract: Rubric-based rewards enable effective reinforcement learning on non-verifiable tasks. However, because these rewards are typically defined as a weighted measure of rubric coverage, models are trained primarily to maximize a recall objective. As a result, RL-trained models produce responses that are more complete but also more verbose and contain more unnecessary or factually incorrect content, leading them to be less preferred than responses from the base model. To address this issue, we present the FORK reward, which jointly optimizes recall and a novel precision objective, which requires each unit of a model response to be both necessary and factually correct, as measured by rubric-driven necessity credits and a factual-correctness grader, respectively. Our experimental results show that FORK outperforms existing rubric-based RL methods and achieves Pareto-optimal performance that balances rubric coverage and overall response quality. 
    
[^180]: 迈向面向文本图问答的可学习检索器

    Towards a Learnable Retriever for Textual Graph Question Answering

    [https://arxiv.org/abs/2610.33511](https://arxiv.org/abs/2610.33511)

    该论文提出将检索器本身构建为可学习的LLM，并在端到端的“先检索后生成”流水线中利用标准答案信号监督证据选择，从而提升对标准答案的覆盖同时避免引入过多无关信息。

    

    文本图问答和基于图的检索增强生成作为将大语言模型（LLM）建立在结构化知识之上的一种方式，正受到越来越多的关注。尽管已有大量研究，现有方法仍存在两个关键差距。第一个是架构差距：现有方法通常只微调基于LLM的生成器，却没有把检索器本身也做成一个可学习的LLM，用于在单一的端到端“先检索后生成”流水线中进行图证据选择。第二个是检索监督差距：现有检索器很少使用标准答案（gold answer）信号来监督证据选择。因此，它们可能无法召回推理所需的证据节点，导致对标准答案集合的覆盖不足；同时，提升召回率的朴素做法往往会引入过多无关信息，损害了紧凑性并使LLM不堪重负。为解决这些差距，我们提出……

    arXiv:2610.33511v1 Announce Type: new  Abstract: Textual graph question answering (GraphQA) and graph-based retrieval-augmented generation (GraphRAG) have gained increasing attention as a way to ground large language models (LLMs) on structured knowledge. Despite extensive efforts, existing methods still face two key gaps. The first is an architectural gap: existing methods typically fine-tune an LLM-based generator, but do not also make the retriever itself a learnable LLM for graph evidence selection within one end-to-end retrieve-then-generate pipeline. The second is a retrieval-supervision gap: existing retrievers rarely use gold answer signals to supervise evidence selection. As a result, they may fail to recall evidence nodes required for reasoning, resulting in insufficient coverage of the gold answer set; meanwhile, naive attempts to boost recall often introduce excessive irrelevant information that compromises compactness and overwhelms the LLM. To address these gaps, we propo
    
[^181]: 新颖与否？PatEx-Gym：面向专利审查中新颖性评估的智能体化现有技术检索

    Novel or Not? PatEx-Gym: Agentic Prior-Art Search for Novelty Assessment in Patent Examination

    [https://arxiv.org/abs/2610.33503](https://arxiv.org/abs/2610.33503)

    该论文提出了首个用于开发和评估专利审查智能体的框架PatEx-Gym，其包含1000个真实审查任务、超百万件专利与5.2万篇论文的大规模检索语料库，并创新性地引入无需标注的检索轨迹合成框架PASS，从而完整覆盖现有技术“检索—推理—评估”的全流程。

    

    专利审查是一项具有挑战性的现实世界任务，它要求智能体发现稀疏的现有技术、迭代式地优化检索查询、审查技术证据，并在异构的科学技术领域中生成记录于审查意见通知书中的新颖性评估。然而，现有的基准测试大多关注静态且有限的候选集合，使得完整的“检索—推理—评估”流程未得到充分探索。我们提出了PatEx-Gym，这是首个用于开发和评估专利审查智能体的框架，它集成了三个组成部分。首先，PatEx-Bench提供了1000个真实的审查任务、两个包含超过100万件专利和5.2万篇科学论文的大规模检索语料库，以及一个支持迭代检索、文档访问、推理和审查意见通知书生成的运行时环境。其次，我们引入了PASS，这是一个无需人工标注的现有技术智能体检索轨迹合成框架。具体而言，该框架利用文档……（摘要原文在此处截断）

    arXiv:2610.33503v1 Announce Type: cross  Abstract: Patent examination is a challenging real-world task that requires agents to discover sparse prior art, iteratively refine queries, inspect technical evidence, and produce novelty assessments documented in Office Actions (OAs) across heterogeneous scientific and technical domains. However, existing benchmarks largely focus on static and limited candidate sets, leaving the full search--reasoning--assessment process underexplored. We present PatEx-Gym, the first framework for developing and evaluating agents for patent examination, which integrates three components. First, PatEx-Bench provides 1K realistic examination tasks, two large-scale retrieval corpora containing over 1M patents and 52K scientific papers, and a harness runtime supporting iterative search, document visits, reasoning, and OA generation. Second, we introduce PASS, an annotation-free Prior-Art Agentic Search Trajectory Synthesis framework. Specifically, it exploits docu
    
[^182]: 五种架构，一个共享空间：无需权重合并的跨模型表征对齐

    Five Architectures, One Shared Space Cross-Model Representational Alignment Without Weight Merging

    [https://arxiv.org/abs/2610.33495](https://arxiv.org/abs/2610.33495)

    研究发现五个独立训练、架构各异的语言模型（四个Transformer和一个状态空间模型）无需权重合并，仅通过激活值分析和广义Procrustes分析即可构建共享表征空间，最后一层的留出检索准确率高达0.91至0.93。

    

    合并语言模型通常通过对权重取平均来实现，但这种方法仅在模型共享同一训练起点时才有效。我们探究了五个冻结的、独立训练的、内部架构各异的模型（分别来自四个机构的四个Transformer模型和一个状态空间模型）能否仅通过其激活值建立关联，而无需任何权重合并。通过置换检验、独立复现探测集、深度扫描、池化消融实验和混杂因素检查，我们发现这五个模型在组织相同文本的方式上存在统计上稳健的对应关系。这种对应关系在每个模型的最后一层最强，而在状态空间模型中较弱，但其最弱的配对仍达到随机水平的35至43倍。我们利用这一发现，通过广义Procrustes分析构建了一个共享空间，其中没有任何模型作为固定参照。在最后一层，二十个有序模型对在两个探测集上的平均留出检索准确率为0.91至0.93（随机水平为0…

    arXiv:2610.33495v1 Announce Type: new  Abstract: Combining language models is usually done by averaging weights, which works only when the models share a common starting point. We ask whether five frozen, independently trained models with different internal designs (four transformers from four organizations and one state-space model) can be related through their activations alone, with no weight merging. Using permutation tests, an independent replication probe set, depth sweeps, pooling ablations and confound checks, we find statistically robust correspondence between how the five models organize the same text. It is strongest at each model's final layer and weaker for the state-space model, whose weakest pairing is still 35-43 times chance. We use this to build a shared space with Generalized Procrustes Analysis, in which no model is a fixed reference. At the final layer, mean held-out retrieval accuracy over the twenty ordered model pairs is 0.91 to 0.93 on two probe sets (chance 0.
    
[^183]: 微小改动，决策改变：基于法语输入变体的大语言模型准确性与可靠性基准测试

    Small Edits, Shifting Decisions: Benchmarking LLM Accuracy and Reliability on French Input Variants

    [https://arxiv.org/abs/2610.33493](https://arxiv.org/abs/2610.33493)

    该论文构建了一个包含12,600个输入实例、涵盖七种任务类型的法语表层变体基准，系统评估发现撇号省略、变音符号缺失和键盘错字等常见输入扰动会显著降低大语言模型的准确率与预测概率可靠性，而更大的模型规模仅能部分缓解这一问题。

    

    法语用户在日常输入中可能会省略撇号和变音符号，或出现键盘打字错误。这些常见变体对语言模型性能和预测可靠性的影响尚缺乏充分理解。为填补这一空白，我们构建了一个受控的法语表层变体基准，涵盖七种任务类型，包含4,200组“原文-低强度-高强度”三元组（共12,600个输入实例）。每个高强度变体保留所有低强度编辑，并添加进一步的变化。我们评估了来自两个大语言模型家族的六个模型。结果表明，随着扰动强度的增加，准确率下降，Brier分数和负对数似然（NLL）上升。更大的模型在高强度扰动下准确率损失较小，但Brier分数和NLL的增加并不随模型规模的增大而持续下降。在预设的高强度设置下，键盘打字错误导致准确率和预测概率质量的最大损失。最后，校准（摘要在此处截断）

    arXiv:2610.33493v1 Announce Type: new  Abstract: French users may omit apostrophes and diacritics or make keyboard typos in everyday input. The effects of these common variations on language model performance and prediction reliability remain insufficiently understood. To address this gap, we construct a controlled French surface-variant benchmark covering seven task types, with 4,200 original-low-high triplets (12,600 input instances). Each high-intensity variant retains all low-intensity edits and adds further changes. We evaluate six models from two LLM families. The results show that accuracy decreases and Brier score and negative log-likelihood (NLL) increase with perturbation intensity. Larger models show smaller accuracy losses under high intensity, but increases in Brier score and NLL do not consistently decrease with scale. Keyboard typos cause the largest losses in accuracy and the quality of predicted probabilities at the predefined high-intensity settings. Finally, calibrat
    
[^184]: LLM辅导互动对学生成绩的增量预测价值

    Incremental Predictive Value of LLM Tutoring Interactions for Student Performance

    [https://arxiv.org/abs/2610.33488](https://arxiv.org/abs/2610.33488)

    该研究首次系统评估了LLM辅导互动特征对学生成绩的增量预测价值，发现其总体上仅带来统计上不显著的边际改进，但对低分学生群体的预测误差有明显降低。

    

    我们评估来自大型语言模型辅导系统的学期级互动特征能否超越常数基线改进作业成绩预测。StudyChat数据集提供了来自两轮本科人工智能课程的16,851条学生-助手对话记录，其中203条学生-学期记录中有175条匹配到了成绩结果。六种拟合估计器与训练折均值和中位数基线共同采用重复嵌套交叉验证。在将每个学生的保留集预测在十次重复中取平均后，梯度提升模型实现了3.023个绩点的平均绝对误差，而中位数基线为3.138。配对差异为-0.114，其条件95%自助法置信区间为[-0.245, 0.013]。特征表示方式的变化以及学期特定的中位数基线也带来了同样微小的提升。然而，在观测分数最低的四分位组中，梯度提升的误差为7.30分，中位数基线为8.07分，均值基线为6.67分；岭回归具有最低……

    arXiv:2610.33488v1 Announce Type: new  Abstract: We evaluate whether semester-level interaction features from a large language model tutor improve assignment-score prediction beyond constant baselines. StudyChat provides 16,851 student-assistant exchanges from two offerings of an undergraduate artificial intelligence course, with outcomes matched to 175 of 203 student-semester records. Six fitted estimators and training-fold mean and median baselines share repeated nested cross-validation. After averaging each student's held-out predictions across ten repeats, Gradient Boosting achieves mean absolute error of 3.023 grade points, versus 3.138 for the median baseline. The paired difference is -0.114, with a conditional 95% bootstrap interval of [-0.245,0.013]. Representation changes and a semester-specific median yield similarly small gains. In the lowest observed-score quartile, however, boosting's error is 7.30 points, versus 8.07 for the median and 6.67 for the mean; Ridge has the low
    
[^185]: Propasafe-Hybrid：一种基于文本的混合式宣传检测工具

    Propasafe-Hybrid: A Text-Based Hybrid Propaganda Detection Tool

    [https://arxiv.org/abs/2610.33464](https://arxiv.org/abs/2610.33464)

    Propasafe-Hybrid 是一个将微调 Transformer 分类器与大语言模型宣传技巧分类相结合的句子级检测系统，并通过成本感知预过滤机制在降低推理成本的同时为用户提供可解释的宣传内容标注。

    

    宣传性内容日益在网络新闻和社交媒体中传播，读者在接触这些内容时往往缺乏足够的审视，这凸显了对可靠且细粒度检测方法的需求。本文提出了 Propasafe-Hybrid，这是一个句子级别的系统，它将微调后的 Transformer 分类器与基于大语言模型（LLM）的宣传技巧分类相结合，以识别、标注并解释具体的宣传策略。该流程生成可操作的输出结果，包括高亮句子、技巧分类以及简洁的判定依据，使用户能够立即理解某个句子为何被标记，以及每个标签是如何确定的。为了控制推理成本，Propasafe-Hybrid 采用了一种成本感知的预过滤阶段，仅将高可能性的句子转发给大语言模型，在保留底层决策逻辑的同时降低了 token 使用量。这些设计选择共同提升了系统的可解释性、效率和实际可用性。

    arXiv:2610.33464v1 Announce Type: new  Abstract: Propagandistic content increasingly circulates through online news and social media, where readers often encounter it with limited scrutiny, highlighting the need for reliable and fine-grained detection. This paper introduces Propasafe-Hybrid, a sentence-level system that integrates a fine-tuned transformer classifier with LLM-based technique classification to identify, label, and explain specific propaganda strategies. The pipeline generates actionable outputs, including highlighted sentences, technique assignments, and concise rationales, so users can immediately understand why a sentence was flagged and how each label was determined. To control inference cost, Propasafe-Hybrid employs a cost-aware pre-filtering stage that forwards only high-likelihood sentences to LLMs, reducing token usage while preserving the underlying decision logic. Together, these design choices enhance the explainability, efficiency, and practical usability of 
    
[^186]: 面向大语言模型可靠性检测的表示特征自适应迁移

    Adaptive Transfer of Representational Signatures for LLM Reliability Detection

    [https://arxiv.org/abs/2610.33442](https://arxiv.org/abs/2610.33442)

    提出了一种在相关领域间迁移低维表示特征并自适应学习组合权重的统计框架，使大语言模型可靠性检测器无需大量目标领域标注数据即可跨域应用。

    

    据报道，大语言模型（LLM）容易出现幻觉和推理错误等可靠性问题。现有的质量检测器能够基于内部表示以较好的准确率预测不可靠的输出。然而，训练这些检测器通常需要大量特定领域的标注数据，这限制了它们在新领域的适用性。我们提出通过在相关领域之间迁移表示特征来解决这一挑战。更具体地说，我们构建了一个统计框架，该框架首先为每个领域提取低维特征，然后使用交叉拟合来学习领域特定预测器的组合权重。这些权重会根据每个源领域对目标领域的预测相关性进行自适应调整。我们针对最小二乘类似模型进行的渐近分析，从领域差异和表示偏差的角度刻画了极限预测风险和聚合权重。

    arXiv:2610.33442v1 Announce Type: new  Abstract: Large language models (LLMs) are reported to be susceptible to reliability issues including hallucinations and reasoning errors. Existing quality detectors can predict unreliable outputs based on internal representations with decent accuracies. However, training these detectors often requires substantial domain-specific labeled data, which limits their applicability to new domains. We propose to address this challenge by transferring representational signatures across related domains. More specifically, we build a statistical framework that first extracts low-dimensional signatures for each domain, then uses cross-fitting to learn composition weights of domain-specific predictors. The weights adapt to each source domain's predictive relevance to the target domain. Our asymptotic analysis of a least-squares analogue characterizes the limiting prediction risk and aggregation weights in terms of domain discrepancies and representation bias.
    
[^187]: 用SCAPE-T量化意义：基于编码器专用Transformer注意力的语义复杂度

    Quantifying Meaning with SCAPE-T: Semantic Complexity from Encoder-Only Transformer Attention

    [https://arxiv.org/abs/2610.33441](https://arxiv.org/abs/2610.33441)

    本文提出SCAPE-T，一种基于编码器Transformer注意力机制的语义复杂度度量方法，将语义复杂度定义为唯一指定文本语义所需的编码长度，通过自我聚焦、CLS聚焦和头间冗余度三个子指标直接量化语义整合，且无需数据驱动训练。

    

    arXiv:2610.33441v1 公告类型：cross 摘要：自动语义复杂度估计在自然语言处理（NLP）领域仍未得到充分探索，尽管其在语言学和可解释性方面具有重要应用。本文将语义复杂度定义为唯一指定文本语义含义所需的编码长度，并通过SCAPE-T对其进行操作化实现——这是一种基于编码器专用transformer、由注意力机制派生的指标。现有度量方法要么对读者付出的认知努力进行建模（如可读性、词汇难度），这类方法与表层特征（词长、句法）相关；要么直接量化表层特征（如类符-形符比、词频）。SCAPE-T则直接针对语义整合，结合三个子指标：（i）逆加权的自我聚焦（词元到自身）、（ii）CLS聚焦（词元到CLS）、（iii）头间冗余度（跨注意力头的余弦相似度），并采用层加权聚合；数据驱动的学习仅限于用于校准的超参数调整。开发过程使用哲学散文与MCTest……

    arXiv:2610.33441v1 Announce Type: cross  Abstract: Automated semantic complexity estimation remains underexplored in NLP, despite important applications in linguistics and interpretability. This paper defines semantic complexity as the code length required to uniquely specify a text's semantic meaning, and operationalises it via SCAPE-T, an attention-derived metric based on encoder-only transformers. Existing measures either model reader effort (readability, lexical difficulty), which correlates with surface features (word length, syntax), or quantify surface features directly (type-token ratio, word frequency). SCAPE-T targets semantic integration directly by combining three submetrics: (i) inverse-weighted self-focus (token-to-self), (ii) CLS focus (token-to-CLS), and (iii) inter-head redundancy (cosine similarity across heads), using layer-weighted aggregation; data-driven learning is confined to hyperparameter tuning for calibration. Development uses philosophical prose vs. MCTest 
    
[^188]: 基于扰动的医学问答数据选择

    Perturbation-Based Data Selection for Medical Question Answering

    [https://arxiv.org/abs/2610.33440](https://arxiv.org/abs/2610.33440)

    本文提出基于高斯扰动的两种医学问答数据选择方法RSP和RSP-Drop，仅需问题和答案选项即可筛选紧凑高效的训练子集，在128样本预算下显著超越随机选择方法。

    

    从现有问答题库中适配医学语言模型时，选择一个紧凑的训练子集是重要的一步。我们研究了基于扰动的数据选择方法，利用从最终块表示的高斯扰动中导出的两个信号：预测变化频率和非翻转间隔收缩。在共享的熵预筛选之后，表示敏感性优先方法（RSP）使用自动均方根缩放来组合这两个信号，而RSP-Drop则仅根据收缩程度对候选数据进行排序。两种选择器在筛选过程中仅使用问题和答案选项。我们在五个医学问答数据集上，使用MedGemma-4B模型在保留集预算为64、128和256个样本的条件下对它们进行评估。在主要的128样本预算下，RSP和RSP-Drop分别获得了47.55和48.28的数据集平均Macro-F1分数，而与熵池匹配的随机选择方法仅为45.11。它们的相对性能因数据集而异。

    arXiv:2610.33440v1 Announce Type: new  Abstract: Selecting a compact training subset is an important step in adapting medical language models from existing question-answering banks. We investigate perturbation-based data selection using two signals derived from Gaussian perturbations of the final-block representation: prediction-change frequency and non-flipping margin contraction. After a shared entropy prefilter, Representation-Sensitivity Prioritization (RSP) combines the signals using automatic root-mean-square scaling, while RSP-Drop ranks candidates by contraction alone. Both selectors use only questions and answer options during selection. We evaluate them with MedGemma-4B on five medical QA datasets at retained-set budgets of 64, 128, and 256 examples. At the primary budget of 128, RSP and RSP-Drop obtain dataset-averaged Macro-F1 scores of 47.55 and 48.28, respectively, compared with 45.11 for entropy-pool-matched random selection. Their relative performance varies across data
    
[^189]: 检验人类与大型语言模型中基于形式的语言能产性

    Examining form-based linguistic productivity in people and large language models

    [https://arxiv.org/abs/2610.33433](https://arxiv.org/abs/2610.33433)

    该研究发现人类能够基于抽象的语言形式规则将新动词从主动语态泛化到被动语态，而大型语言模型只有在新动词具有类似已知动词的意义时才能成功，表明计算模型的句法知识仍依赖于具体词汇的分布与意义，缺乏人类那样的抽象性。

    

    语言能产性，即词语在新语境中的使用，是人类语言最显著的特征之一。然而，支撑这种能产性的知识究竟有多抽象仍不清楚。我们在成功泛化依赖于基于形式的信息的情境中，比较了人类与计算语言模型的语言能产性。我们专注于将主动语态中学到的新动词的使用扩展到被动语态语境中。我们发现，即使无法为新动词赋予合理的意义，人们仍能以一种与关于语言形式的抽象泛化相一致的方式，将主动用法扩展到被动用法。相比之下，大型语言模型只有在新动词能够被赋予类似现有动词的意义时才能成功。我们得出结论：计算模型的句法知识仍然与特定词汇项的分布和意义紧密绑定，而人类的句法知识则不然。

    arXiv:2610.33433v1 Announce Type: new  Abstract: Linguistic productivity, as evidenced by the use of words in novel contexts, is one of the most striking facts about human language. However, how abstract the knowledge that enables such productivity is remains unclear. We compare people's and computational language models' linguistic productivity in contexts where successful generalization depends on form-based information. We focus on extending the use of a novel verb learned in the active voice to passive contexts. We find people extend active to passive uses in a way consistent with abstract generalizations concerning linguistic form, even when no plausible meaning can be assigned to the novel verb. In contrast, large language models succeeded only when a meaning like an existing verb's could be assigned to the novel verb. We conclude that computational models' syntactic knowledge remains tied to the distribution and meaning of specific lexical items in a way that human syntactic kno
    
[^190]: 后训练如何塑造大语言模型的价值偏好？

    How Does Post-Training Shape LLM Value Preferences?

    [https://arxiv.org/abs/2610.33371](https://arxiv.org/abs/2610.33371)

    该论文通过受控实验发现，SFT、DPO、GRPO等不同后训练路径会以不同方式放大语料库中的观点不平衡，而被放大的价值偏好能否通过后续后训练消除，取决于训练方法与语料中的观点构成。

    

    在后训练过程中，大语言模型（LLM）会从语料库中学习价值偏好，而这些语料库中与不同人群相关的观点比例可能是不平衡的。放大这种不平衡可能会进一步导致某些群体的观点代表性不足。然而，观点混合比例如何在不同后训练路径中塑造价值偏好，目前仍不清楚。为了回答这个问题，我们使用包含不同观点混合比例的后训练语料库，以及基于SFT、DPO和GRPO的五种后训练路径，进行了受控实验。我们发现，不同的后训练路径对观点不平衡的放大程度各不相同。一旦价值偏好被放大，后续的后训练能否消除这种放大，既取决于后训练方法，也取决于语料库中的观点混合比例。更令人担忧的是，模型在干预下的稳定性取决于后续后训练是否改变其价值偏好。此外，这些价值……

    arXiv:2610.33371v1 Announce Type: new  Abstract: During post-training, large language models (LLMs) learn value preferences from corpora that may contain imbalanced proportions of opinions associated with different demographic groups. Amplifying these imbalances may further underrepresent some groups' opinions. However, how opinion mixtures shape value preferences across post-training paths remains unclear. To answer this question, we conduct controlled experiments using post-training corpora with different opinion mixtures and five post-training paths based on SFT, DPO, and GRPO. We find that post-training amplifies opinion imbalances differently across post-training paths. Once value preferences are amplified, whether further post-training can remove them depends on both the post-training method and the opinion mixture in the corpus. More concerningly, a model's stability under interventions depends on whether further post-training changes its value preference. Moreover, these value 
    
[^191]: 蒸馏、辩论、发现：将多智能体研究构思建立在图蒸馏洞察之上

    Distill, Debate, Discover: Grounding Multi-Agent Research Ideation in Graph-Distilled Insights

    [https://arxiv.org/abs/2610.33366](https://arxiv.org/abs/2610.33366)

    提出 InsightAgents 框架，通过从相关论文图谱中蒸馏出的研究洞察（如方法演变、潜在假设和未决挑战）为多智能体赋予基于文献的独特专长，使其通过辩论协作提出并完善研究提案。

    

    大语言模型（LLM）正被越来越多地用于支持研究想法生成与科学发现。多智能体辩论通过汇集多样化的视角来共同发展研究想法，为这一任务提供了一种协作式方法。然而，一个关键挑战在于如何为各个智能体赋予独特且有意义的专长，以引导其贡献。受研究者在协作中运用其积累的研究经验这一方式的启发，我们提出了 InsightAgents，这是一个通过从相关论文的图谱中蒸馏洞察来构建基于文献的智能体专长的框架。在沿着某一研究方向追踪研究方法的演变、揭示其潜在假设并识别未解决挑战的洞察指引下，各智能体共同确定一个研究问题，并通过辩论提出、质疑和改进潜在的研究方法，从而形成一份研究提案。我们在三项研究构思任务上对 InsightAgents 进行了评估……

    arXiv:2610.33366v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to support research idea generation and scientific discovery. Multi-agent debate offers a collaborative approach to this task by bringing together diverse perspectives to develop research ideas. However, a key challenge is how to equip agents with distinct and meaningful expertise to guide their contributions. Inspired by how researchers draw on accumulated research experience during collaboration, we introduce InsightAgents, a framework that constructs literature-grounded agent expertise by distilling insights from graphs of related papers. Guided by insights that trace evolving research approaches, reveal underlying assumptions, and identify unresolved challenges, within a line of research, agents jointly identify a research problem and propose, challenge, and refine potential approaches through debate to develop a research proposal. We evaluate InsightAgents against three research ide
    
[^192]: 偏好优化改进定义生成：欧盟机构可持续金融披露中的语义变化

    Preference Optimization Improves Definition Generation: Semantic Change in EU Institutional Sustainable Finance Disclosures

    [https://arxiv.org/abs/2610.33357](https://arxiv.org/abs/2610.33357)

    使用直接偏好优化（DPO）微调定义生成模型可解决同一词义生成过多定义的问题，并成功应用于审计欧盟可持续金融术语的语义变化。

    

    类词典定义生成器被证实对词汇语义变化检测（LSCD）非常有用。然而，它们存在语义分布过于细粒度的问题，即为同一个词汇语义生成过多的定义。在这项工作中，我们表明使用直接偏好优化（DPO）进行微调有助于克服这一问题。我们在精心整理的学术LSCD基准上进行了定量测试，并在真实世界数据上进行了定性测试，应用于审计欧盟发展金融项目文档中可持续性术语历时变化的实际任务。本研究产生的所有模型、代码和预测结果均在 https://github.com/ltgoslo/dpo-for-definition-generation 公开提供。

    arXiv:2610.33357v1 Announce Type: new  Abstract: Dictionary-like definition generators are known to be useful for lexical semantic change detection (LSCD). However, they suffer from overly granular sense distribution, producing excessive number of definitions for one and the same lexical sense. In this work, we show that fine-tuning with Direct Preference Optimization (DPO) helps to overcome this problem. We test our approach both quantitatively on a curated academic LSCD benchmark and qualitatively on real-world data for a practical task of auditing diachronic changes in sustainability terminology within project documentation of EU development finance. All models, code and predictions produced in this study are publicly available at https://github.com/ltgoslo/dpo-for-definition-generation.
    
[^193]: 句法回路的出现与稀疏化

    The Emergence and Sparsification of a Syntactic Circuit

    [https://arxiv.org/abs/2610.33299](https://arxiv.org/abs/2610.33299)

    该研究追踪了语言模型训练全程中主谓一致性回路的变化，发现句法行为早期即形成且最初由分散神经元承担，随后机制持续重组并最终变得高度稀疏，揭示了语言知识在模型中的动态演化过程。

    

    为了理解语言模型中语法行为的出现过程，我们追踪了pythia-1b以及十个不同随机种子的pythia-410m在整个训练过程中负责主谓单复数一致性的回路。我们发现主谓一致性在训练早期就被学会，并且最初由一群分散的神经元共同处理。然而，在一致性行为稳定之后，负责该行为的机制仍在持续重组。介导一致性的注意力头在训练过程中经常发生转移：以pythia-1b为例，训练早期该行为由两个注意力头控制，而后期则由第三个（不同的）注意力头控制。此外，该回路随着训练逐渐缩小，pythia-1b最终的回路表现得异常稀疏。我们认为这些结果对于理解语言知识如何在语言模型中出现及其表示方式具有重要启发意义。

    arXiv:2610.33299v1 Announce Type: new  Abstract: To understand the emergence of grammatical behavior in language models, we track circuits for subject-verb number agreement in pythia-1b and ten seeds of pythia-410m across training. We find that subject-verb agreement is learned early, and is initially handled by a diffuse collection of neurons. However, after agreement behavior has stabilized, the mechanisms responsible for it continue to reorganize. The attention heads which mediate agreement often shift during training: in the case of pythia-1b, behavior is controlled by two heads early in training but by a third (different) head later. Furthermore, the circuit shrinks over training, with pythia-1b's final circuit being strikingly sparse. We take these results to be informative about how linguistic knowledge emerges and is represented in a language model.
    
[^194]: 药物警戒中个体病例安全报告的多智能体检索增强因果关系评估（MARCA）

    Multi-Agent Retrieval-Augmented Causality Assessment (MARCA) of Individual Case Safety Reports in Pharmacovigilance

    [https://arxiv.org/abs/2610.33279](https://arxiv.org/abs/2610.33279)

    该研究提出了MARCA四智能体框架，结合文献检索、说明书评估与临床推理，利用大语言模型对药物警戒报告进行Naranjo因果关系自动评估，与人类专家评分达到较高一致性。

    

    引言：日益增多的个体病例安全报告（ICSR）对药物警戒领域中由专家主导的因果关系评估构成了挑战。我们开发了MARCA（多智能体检索增强因果关系评估），这是一个基于Naranjo量表进行评估的四智能体框架，并评估了其与人类专家评分的一致性，同时探索性比较了非RAG与RAG标注评分之间的差异。方法：MARCA结合了科学文献检索、药品说明书评估、客观证据分类和临床因果关系推理，采用Gemini 2.5 Pro和DeepSeek-V3.2模型。智能体生成的推理依据用于辅助人工评分。共识专家参考标准涵盖来自FAERS（2025年第一季度至第二季度）的215份ICSR和1,562个药物-事件组合（DEC）。一致性采用百分比一致性和Cohen's kappa系数进行衡量。结果：在问题层面，Naranjo Q2的一致性为96.8%（kappa = 0.491；95% CI 0.265-0.663），Q5的一致性为85.4%（kappa = 0.269；95%（摘要在此截断）

    arXiv:2610.33279v1 Announce Type: new  Abstract: Introduction: Increasing volumes of individual case safety reports (ICSRs) challenge expert-led causality assessment in pharmacovigilance. We developed MARCA (Multi-Agent Retrieval-Augmented Causality Assessment), a four-agent framework for Naranjo-based assessment, and evaluated agreement with human experts and exploratory differences between non-RAG and RAG-labelled scores.   Methods: MARCA combined scientific literature retrieval, drug-label assessment, objective-evidence classification, and clinical causality reasoning using Gemini 2.5 Pro and DeepSeek-V3.2. Agent-generated rationales informed human-assisted scoring. A consensus expert reference standard covered 215 ICSRs and 1,562 drug-event combinations (DECs) from FAERS (Q1-Q2 2025). Agreement was measured using percentage agreement and Cohen's kappa.   Results: Question-level agreement was 96.8% for Naranjo Q2 (kappa = 0.491; 95% CI 0.265-0.663), 85.4% for Q5 (kappa = 0.269; 95% 
    
[^195]: FlexA-Mem：面向大语言模型智能体的预算自适应记忆系统

    FlexA-Mem: Budget-Adaptive Agentic Memory for LLM Agents

    [https://arxiv.org/abs/2610.33274](https://arxiv.org/abs/2610.33274)

    FlexA-Mem 提出了一种预算自适应的通用智能体记忆系统，将历史一次性写入不可变存储，并在每次调用时以固定规则渲染出“紧凑时间线+查询匹配细节”的记忆，从而同时满足精确事实检索与多步任务追踪两类不同的记忆需求。

    

    arXiv:2610.33274v1 公告类型：新论文。摘要：大语言模型（LLM）智能体所处的应用场景对记忆的需求差异极大。关于过往对话的提问需要其中被明确提及的确切事实，而多步任务则需要已执行步骤的轨迹，以及下一步骤无法自行指明的细节。现有的记忆系统各自只针对单一场景进行设计和测试：A-Mem 等检索式记忆会遗漏当前查询未明确表达的步骤，摘要式记忆会丢失确切的字符串，而将两者简单配对又会预先固定预算的划分方式。我们提出 FlexA-Mem——一种通用记忆系统：它将每个历史单元一次性写入不可变的存储，并在每次调用时渲染出一份全新的记忆。所渲染的记忆是一条紧凑的记录时间线，再由固定规则（无需调用 LLM）把细节填入剩余空间。这些细节要么是以原始文本或摘录形式呈现的相关单元，要么是与查询最匹配的逐字片段。凭借单一的存储，FlexA-Mem 能够同时服务于两类需求……（原文摘要在此处截断）

    arXiv:2610.33274v1 Announce Type: new  Abstract: Large language model (LLM) agents work in scenarios with very different memory needs. A question about past conversations needs the exact facts it names, while a multi-step task needs a trace of the steps already taken together with details that the next step cannot name. Existing memories are each designed and tested for one kind of scenario. Retrieval memories such as A-Mem miss steps that the current query does not express, summaries drop exact strings, and pairing the two fixes the split of the budget in advance. We propose FlexA-Mem, a general memory that writes each unit of history once into an immutable store and renders a fresh memory for every call. The rendered memory is a compact timeline of records plus detail that a fixed rule fits into the remaining room without an LLM call. The detail is either relevant units as raw text or excerpts or the verbatim pieces that best match the query. With one store, FlexA-Mem serves both kin
    
[^196]: 两种不确定性并非同一种：廉价结构化内容审核中的分歧与决策风险

    Two Uncertainties Are Not One: Disagreement versus Decision-Risk in Cheap Structured Content Moderation

    [https://arxiv.org/abs/2610.33258](https://arxiv.org/abs/2610.33258)

    通过对一个低成本“系统一”内容审核模型在14万余条人工标注评论上的首次独立审计，发现其分类准确率（AUROC 0.83）与14个前沿推理模型中的13个在统计上不可区分且成本低1-3个数量级，并指出“置信度”混淆了“分歧”与“决策风险”两种本质不同的不确定性。

    

    内容审核将带有等级、且存在争议的人类判断压缩为针对数十亿条内容的二元行为（保留/删除），在这样的规模下，对每条帖子都运行前沿大语言模型是不经济的。一类新型的冻结“系统一”模型返回类型化、经过校准的决策而非文本，每次调用成本约0.000025美元，耗时约100毫秒。我们在142,658条人工标注的评论上对其中一个此类模型进行了首次独立审计，并在共享的探测集上将其与14个前沿推理模型（GPT-5.x、Claude、Gemini 3和Grok系列）进行基准对比。有两个发现尤为突出。第一，廉价模型在准确率上具有竞争力：其零样本分类AUROC（0.83）与14个前沿系统中的13个在统计上不可区分（配对自助法，n=56；这是未能拒绝等价性，而非等价性的证明），而成本却低1-3个数量级，且其校准程度不逊于其中若干模型。第二，“置信度”混淆了两种不确定性……（原文摘要至此截断）

    arXiv:2610.33258v1 Announce Type: new  Abstract: Content moderation compresses a graded, contested human judgment into a binary act (leave up / take down) over billions of items, at a scale where running a frontier LLM on every post is uneconomical. A new class of frozen "System One" models returns typed, calibrated decisions instead of text at ~USD 0.000025 per call and ~100 ms. We run the first independent audit of one such model on 142,658 human-labeled comments, and benchmark it against 14 frontier reasoning models (the GPT-5.x, Claude, Gemini 3, and Grok families) on a shared probe set. Two findings stand out. First, the cheap model is accuracy-competitive: its zero-shot classification AUROC (0.83) is not statistically separable (paired bootstrap, n=56; a failure to reject parity, not proof of equivalence) from 13 of the 14 frontier systems while costing 1-3 orders of magnitude less, and it is no worse calibrated than several of them. Second, "confidence" conflates two uncertainti
    
[^197]: 嘿，猜猜怎么着：CUES线索引导对事件典型性的推断

    Hey, guess what: CUES lead to inferences about typicality

    [https://arxiv.org/abs/2610.33239](https://arxiv.org/abs/2610.33239)

    本文构建了两个事件典型性语料库，通过大规模人类与LLM对比实验发现，人类依靠常识知识、话语强调和讽刺这三种CUES线索来推断事件典型性，而LLM因缺乏必要的常识知识而难以利用这些线索。

    

    在人的一生中，我们会不断积累关于某些事件典型程度的知识。我们在日常对话中运用这种知识，常常能够解读出超出字面明确表达内容的含义。在这项工作中，我们为更好地理解这种含义恢复过程迈出了步伐，同时研究了非典型性推断和典型性推断。首先，我们构建了两个语料库：EventTyp和经过人工验证的EventTypGold。我们开展了大规模的人类与大语言模型（LLM）实验，发现对于人类而言，有三个CUES线索可以预测关于典型性的推断：常识知识、话语强调和讽刺。然而，在大多数实验设置中，我们发现大语言模型缺乏准确评估事件总体典型性所需的先决常识知识，因此难以遵循这些CUES线索进行推断。

    arXiv:2610.33239v1 Announce Type: new  Abstract: Throughout life, we accumulate knowledge of how typical certain events are. We use this knowledge in daily conversation, often interpreting meaning beyond what has been said explicitly. In this work, we take steps to better understand this meaning recovery, investigating both atypicality and typicality inferences. First, we construct two corpora, EventTyp and the human-validated EventTypGold. We conduct large-scale human and LLM experiments and find that for humans, three CUES predict inferences about typicality: Commonsense Knowledge, Utterance Emphasis, and Snark. However, in most experimental settings, we find that LLMs lack the prerequisite Commonsense Knowledge necessary to accurately assess an event's general typicality, and therefore struggle to follow these CUES.
    
[^198]: 各有各的不足：跨不同证据状态校准语言模型

    Each Insufficient in Its Own Way: Calibrating Language Models Across Varying Evidence States

    [https://arxiv.org/abs/2610.33237](https://arxiv.org/abs/2610.33237)

    提出证据状态校准的新范式与SuffiCE方法，通过微调模型下一词元分布中已有的置信度，使模型置信度随证据状态的变化而与其实际回答正确的概率保持一致。

    

    语言模型如何知道自己是否拥有足够的证据来回答问题？即使现有证据不充分，语言模型也倾向于直接作答。决定是回答还是继续收集更多证据，需要对“在现有证据下答案正确的可能性”进行校准估计。然而，大多数校准工作都将输入视为固定的，而没有探究这一估计应如何随证据的变化而变化。我们提出了“证据状态校准”（evidence-state calibration），它要求模型在回答前的置信度与其正确回答的概率相匹配，并且每当证据改变这一概率时，置信度也应相应变化。我们提出了SuffiCE（充分性置信度估计，Sufficiency Confidence Estimation），该方法对模型下一词元分布中已存在的置信度进行微调，使其匹配模型在每种证据状态下回答正确的频率。在临床诊断和多跳问答任务上……

    arXiv:2610.33237v1 Announce Type: new  Abstract: How can a language model know whether it has enough evidence to answer? Language models tend to answer even when the evidence they have is insufficient. Deciding whether to answer or gather more evidence requires a calibrated estimate of how likely an answer is to be correct with the available evidence. However, most calibration work treats the input as fixed, without asking how this estimate should change with the evidence. We introduce \emph{evidence-state calibration}, which requires a model's confidence before answering to match its probability of answering correctly and to change accordingly whenever the evidence changes this probability. We propose \ours{} (\textbf{Suffi}ciency \textbf{C}onfidence \textbf{E}stimation), which fine-tunes the confidence already present in a model's next-token distribution to match how often the model answers correctly from each evidence state. Across clinical diagnosis and multi-hop question answering
    
[^199]: 链与收益：强化学习对基于推理的图像质量评估带来了什么改变

    Of Chains and Gains: What Reinforcement Learning changes on Reasoning-based Image Quality Assessment

    [https://arxiv.org/abs/2610.33203](https://arxiv.org/abs/2610.33203)

    本文通过采样感知评估与受控推理干预严谨剖析了 RLVR 在图像质量评估中的真实收益，发现其改进超越采样效应但集中于分布内，且模型评分在推理时对思维链的依赖性逐渐减弱。

    

    可验证奖励强化学习（RLVR）已提升了大型视觉语言模型（VLM）在图像质量评估（IQA）任务上的性能。现有方法通常将这些收益归因于评分前推理能力的提升，其依据是日益合理的思维链。然而，这些收益是否超越了采样效应，以及它们如何依赖于显式推理，仍不清楚。我们通过采样感知评估、受控推理干预和捷径实验严谨地研究了这些问题。我们发现，在大预算 $\operatorname{max}_g@k$ 评估下，RL 训练的模型优于基础模型，而超出平均效果的改进集中在分布内。RLVR 生成的思维链在被移植到基础模型时能提升其性能，并且在固定思维链的情况下也能改善评分。然而在推理过程中，评分对思维链的依赖性逐渐降低，这种趋势在 7B 模型上比 32B 模型上更为明显。

    arXiv:2610.33203v1 Announce Type: cross  Abstract: Reinforcement Learning with Verifiable Rewards (RLVR) has improved the performance of large Vision-Language Models (VLMs) on image quality assessment (IQA). Existing approaches often attribute these gains to improved reasoning before scoring, supported by increasingly plausible Chain-of-Thought (CoT). However, whether these gains exceed sampling effects and how they depend on explicit reasoning remain unclear. We rigorously investigate these questions through sampling-aware evaluation, controlled reasoning interventions, and shortcut experiments. We find that RL-trained models outperform base models under large-budget $\operatorname{max}_g@k$ evaluation, while improvements beyond averaging are concentrated in-distribution. RLVR produces chains that raise base-model performance when transplanted and improves scoring with the chain held fixed. Yet scoring becomes less dependent on the chain during inference, more clearly at 7B than at 32
    
[^200]: 位置、角色还是标签？LLM智能体中指令冲突的受控研究

    Position, Role or Label? A Controlled Study of Instruction Conflicts in LLM Agents

    [https://arxiv.org/abs/2610.33201](https://arxiv.org/abs/2610.33201)

    该论文构建了包含1,080个受控场景的WhoWins基准，通过对14个开放权重LLM智能体约189万条回复的分析，揭示了指令冲突中独立于角色和标签的强位置效应——首条指令在消息内和跨消息场景中均显著占优。

    

    LLM智能体会从操作者、仓库配置、用户、工具结果和保存的记忆中接收指令，这些指令经常相互冲突。指令层级基准测试通常询问排名较高的来源是否获胜，但一个来源同时决定了位置、角色和标签。首先，我们构建了WhoWins，这是一个包含五类任务共1,080个由代码自动评分的场景的基准，每个场景最多包含五条互斥指令，并设计了七个仅改变指令传递方式的受控实验。其次，我们研究了14个开放权重模型（参数量从0.6B到122B，分别关闭和开启思考模式），共收集1,887,356条评分回复，并在保留场景上验证了主要发现。在同一条消息内，两条冲突行中的第一行以72.42%至94.66%的比率获胜；跨消息时，第一行仍保持22.95%至62.65%的胜率。在系统（或开发者）消息中首先发送的策略，在所有Qwen模型中与后续指令的冲突大多落败，但在gpt-oss执行操作时大多获胜。在工具和文件写入任务中，其获胜频率低于……

    arXiv:2610.33201v1 Announce Type: cross  Abstract: LLM agents take instructions from the operator, repository configuration, user, tool results and saved memory, which often conflict. Instruction-hierarchy benchmarks ask whether the higher-ranked source wins, but a source sets position, role and label at once. First, we build WhoWins, 1,080 code-scored scenarios in five task types, each with up to five mutually exclusive instructions, and seven controlled experiments that vary only how instructions are delivered. Second, we study 14 open-weight models (0.6B to 122B, thinking off and on) over 1,887,356 scored replies, checking main findings on held-out scenarios. Inside one message the first of two conflicting lines wins 72.42 to 94.66%, and across messages it keeps 22.95 to 62.65%. A policy sent first in the system (or developer) message loses most conflicts with later lines in every Qwen model but wins most on gpt-oss when it acts. In tool and file-write tasks it wins less often than 
    
[^201]: 请展现倾向：利用合成文档微调揭露伪装对齐的大语言模型

    Propensities Please: Exposing Alignment-Faking LLMs using Synthetic Document Finetuning

    [https://arxiv.org/abs/2610.33180](https://arxiv.org/abs/2610.33180)

    本文提出利用合成文档微调的方法，无需依赖思维链监控、高度真实评估环境或机制性干预，即可揭露伪装对齐的大语言模型的潜在不对齐倾向。

    

    对齐评估旨在评估大语言模型是否会从事危险的不对齐行为。不对齐且具有评估意识的模型可能会进行“对齐伪装”，即它们追求隐藏的目标，但在怀疑存在有效监督时（例如在训练、评估或受审计的部署期间）隐瞒这种行为。这样的模型可能会将任何旨在揭示潜在倾向的基于提示词的干预视为评估的一部分，从而使现有对齐评估的可靠性受到质疑。目前检测对齐伪装的方法依赖于思维链的可监控性、降低评估意识的高度真实环境，或机制性干预，而所有这些方法对于未来模型代际而言，其可靠性预计都将下降。我们提出了一种不依赖于上述任何方法即可揭示潜在不对齐倾向的方法。我们使用合成文档微调来教导模型……

    arXiv:2610.33180v1 Announce Type: new  Abstract: Alignment evaluations aim to assess whether large language models would engage in dangerous misaligned behaviour. Misaligned and evaluation-aware models might engage in alignment faking, where they pursue a hidden goal, but conceal this behaviour when they suspect effective oversight such as during training, evaluation or audited deployment. Such a model might view any prompt-based intervention that aims to reveal latent propensities as part of the evaluation, calling into question the reliability of existing alignment evaluations. Current methods for detecting alignment faking rely on chain-of-thought monitorability, highly realistic environments that reduce evaluation awareness, or mechanistic interventions, all of which are expected to become less reliable for future model generations. We present a method for revealing latent misaligned propensities that does not rely on any of these. We use synthetic document finetuning to teach mode
    
[^202]: 大语言模型能够攻克高能理论物理，但仍存在差距：第一部分，量子场论

    LLMs Can Tackle High Energy Theoretical Physics, yet Gaps Remain: Part I, Quantum Field Theory

    [https://arxiv.org/abs/2610.33134](https://arxiv.org/abs/2610.33134)

    该论文提出了包含75道研究生至研究级量子场论问题的QFT-Bench基准测试，评估显示最强的大语言模型准确率已超过95%，但在计算、约定和指令遵循方面仍存在明显差距。

    

    除了数学和编程之外，大语言模型（LLMs）开始能够处理传统上需要理论物理学家经过大量训练才能解决的问题，但这种能力究竟能延伸到什么程度、在哪里可能会失效，目前仍不清楚。作为推动大语言模型在高能理论物理领域发展的第一步，我们提出了QFT-Bench，这是一个包含75道研究生至研究级问题的基准测试，这些问题需要在量子场论（QFT）的核心子领域进行深入的推理和推导。每个问题都有明确定义的目标和闭式参考答案。评估方法结合了符号验证与基于大语言模型的评判。在39个专有和开源模型中，前沿模型在非平凡的量子场论计算上已经展现出可观的能力，最强的模型准确率超过95%。与此同时，明显的差距仍然存在。即使是强大的模型，在计算、约定和指令遵循方面仍会出错。

    arXiv:2610.33134v1 Announce Type: cross  Abstract: Beyond math and code, Large language models (LLMs) are beginning to tackle problems that have traditionally required substantial training for theoretical physicists, but it remains unclear how far this capability extends and where it might break. As a first step toward pushing LLMs in high-energy theoretical physics, we introduce QFT-Bench, a benchmark of 75 graduate- to research-level problems requiring extensive reasoning and derivation across core Quantum Field Theory (QFT) subareas. Each question has a well-defined target and closed-form reference answer. Evaluation combines symbolic verification with LLM-based judge. Across 39 proprietary and open-source models, frontier models already demonstrate substantial capabilities on nontrivial QFT calculations, with the strongest models exceeding 95% accuracy. At the same time, clear gaps remain. Even strong models still make errors in calculations, conventions, and instruction following,
    
[^203]: 投机分叉：将闲置等待转化为代码仓库环境搭建中的并行探索

    Speculative Forking: Turning Idle Wait into Parallel Exploration for Repository Environment Setup

    [https://arxiv.org/abs/2610.33129](https://arxiv.org/abs/2610.33129)

    SpecBuild通过“投机分叉”机制将环境命令的漫长等待时间用于并行探索多条安装分支，返回首个成功配置的环境并丢弃其余分支，从而显著提升基于LLM的代码仓库环境搭建的速度与成功率。

    

    在开发人员能够使用一个软件项目之前，必须先让该项目运行起来。这通常需要安装正确的语言运行时、系统软件包、构建工具和测试命令。近期基于大语言模型（LLM）的智能体通过检查代码仓库、编写安装脚本、运行脚本并修复产生的错误，实现了环境搭建的自动化。然而，这些智能体仍然遵循串行的试错循环：它们先尝试一条安装路径，等待缓慢的命令执行完毕，观察失败结果，然后才决定下一步尝试什么。由于环境相关的命令往往需要很长时间运行，这种串行工作流使得环境搭建速度缓慢，且每次只能探索一条路径。我们提出了SpecBuild，一个通过投机分叉机制实现更快、更高成功率的基于LLM的环境搭建系统。SpecBuild并行探索多条候选环境搭建分支，返回第一个成功配置完成的环境，并丢弃其余分支。它将这一设计应用于……

    arXiv:2610.33129v1 Announce Type: cross  Abstract: Before developers can use a software project, they must first get it to run. This often requires installing the right language runtime, system packages, build tools, and test commands. Recent LLM-based agents automate environment setup by inspecting a repository, writing setup scripts, running them, and repairing the resulting errors. However, these agents still follow a sequential trial-and-error loop: they try one setup route, wait for slow commands to finish, observe the failure, and only then decide what to try next. Because environment commands often take long to run, this sequential workflow makes setup slow and limits exploration to one route at a time.   We present SpecBuild, an LLM-based environment setup system designed to be both faster and more successful through speculative forking. SpecBuild explores candidate setup branches in parallel, returns the first committed environment, and discards the rest. It applies this desig
    
[^204]: 大型语言模型中的前景理论：对损失厌恶、敏感性递减与概率加权的元分析

    Prospect Theory in Large Language Models: A Meta-Analysis of Loss Aversion, Diminishing Sensitivity, and Probability Weighting

    [https://arxiv.org/abs/2610.33103](https://arxiv.org/abs/2610.33103)

    本文通过对32个语言模型的102个拟合估计进行元分析，发现大型语言模型表现出温和的损失厌恶（λ=1.26）：在收益与损失对称情境下其行为与人类相容，但总体损失厌恶程度显著低于人类，且三个前景理论成分的表现各不相同。

    

    前景理论将风险决策分解为三个部分：损失厌恶（以损失相对于收益的权重λ来衡量，例如λ=2意味着损失的权重是收益的两倍）、对结果大小的敏感性递减（即价值函数的曲率与线性特征），以及概率的非线性加权。本文通过对涵盖32个语言模型的11项研究中的102个拟合估计进行元分析，考察大型语言模型（LLM）是否表现出与人类相同的特性。研究发现这三个组成部分的表现各不相同：LLM的损失厌恶程度较为温和，汇总后的λ为1.26（95%置信区间为1.06至1.46），远低于最强的人类元分析估计值（λ=1.96），但与较为保守的人类估计值（1.07和1.31）相容。在匹配设计下，当收益与损失对称时，LLM与人类表现相容（λ=1.15对1.07），但当损失小于收益时，LLM的损失厌恶程度则远低于人类。

    arXiv:2610.33103v1 Announce Type: new  Abstract: Prospect theory splits risky choice into three parts: loss aversion, measured by the relative weight of losses to gains, $\lambda$ (e.g., $\lambda = 2$ implies that losses weigh twice as much as gains), diminishing sensitivity to the size of outcomes (i.e., the curvature and linearity of the value function), and the nonlinear weighting of probabilities. I examined whether Large Language Models (LLMs) display the same properties as people by meta-analyzing 102 fitted estimates from 11 studies covering 32 language models. The three components emerge differently. Loss aversion is modest: the pooled $\lambda$ is 1.26 (95% CI 1.06 to 1.46), far below the strongest human meta-analytic estimate ($\lambda = 1.96$) and compatible with more conservative ones (1.07 and 1.31). Matched by design, LLMs are compatible with people when gains and losses are symmetric ($\lambda = 1.15$ against 1.07), but well below them when losses are smaller than gains 
    
[^205]: 参数化会话状态：无需重放历史即可恢复检索意图

    Parametric Session States Recover Retrieval Intent without Replaying History

    [https://arxiv.org/abs/2610.33047](https://arxiv.org/abs/2610.33047)

    提出参数化会话状态方法，通过可复用的会话记忆层记住检索意图，使长期对话助手无需重放历史记录即可解析缺乏关键检索词的后续请求，同时保持事实知识外置。

    

    arXiv:2610.33047v1 公告类型：cross 摘要：长期运行的助手常常需要解释这样的后续请求：其指代对象、约束条件和当前目标早在许多轮对话之前就已确立。因此，核心挑战不仅在于上下文长度，更在于检索意图：诸如“使用之前的配置”这样的当前查询可能省略了定位相关证据所需的项目、实体和版本等术语。长上下文模型可以通过重放交互历史来恢复这些术语，但需要反复承担处理历史记录的成本。传统的检索增强生成（RAG）虽然成本更低，但其假设查询本身包含足够的信息；对话式改写方法要么需要重新阅读历史记录，要么依赖于显式维护的文本记忆。我们提出了参数化会话状态（Parametric Session State），这是一个可复用的会话记忆层，它能够记住要搜索什么，同时将事实知识保持在外部。

    arXiv:2610.33047v1 Announce Type: cross  Abstract: Long-running assistants often interpret follow-up requests whose referents, constraints, and active goals were established many turns earlier. The central challenge is therefore not only context length, but retrieval intent: a current query such as ``use the previous configuration'' may omit the project, entity, and version terms needed to locate relevant evidence. Long-context models can recover those terms by replaying the interaction, but repeatedly pay the cost of processing history. Conventional retrieval-augmented generation (RAG) is cheaper, yet assumes an information-sufficient query; conversational rewriting either rereads the history or depends on an explicitly maintained textual memory. We introduce \emph{Parametric Session State} (\method{}), a reusable session-memory layer that remembers what to search for while keeping factual knowledge external.
    
[^206]: 一种用于衡量短视频平台创作者传播行为的多模态框架

    A Multimodal Framework for Measuring Creator Communication on Short-Form Video

    [https://arxiv.org/abs/2610.33043](https://arxiv.org/abs/2610.33043)

    该论文提出一个五阶段多模态框架，整合平台元数据、语音转录与大语言模型生成的视觉-语言描述，实现了对短视频创作者主题与传播策略的大规模量化分析，并在近万个TikTok视频上完成了验证。

    

    短视频融合了语音、图像和屏幕文字，为系统性内容分析带来了挑战。我们提出了一个五阶段多模态框架，用于大规模衡量创作者的主题与传播策略。该框架整合了平台元数据、语音转录文本以及对采样帧的视觉-语言描述，随后应用大语言模型生成预定义指标和开放式标签。经过审核的关键词方案对主题标签进行了整合，而创作者加权聚合方式则使每个账户无论发布量多少均被赋予相同权重。我们在来自十六位创作者的9,852个TikTok视频上演示了该框架，最终保留7,311个视频，围绕七个主题和七个传播指标进行分析。自动化生成的创作者画像揭示了各创作者在运用教育内容、幽默表达以及回应时事方面的显著差异。通过两位独立专家标注者的验证，得到了平均F（摘要原文在此处截断）

    arXiv:2610.33043v1 Announce Type: cross  Abstract: Short-form video combines speech, imagery, and on-screen text, creating challenges for systematic content analysis. We present a five-stage multimodal framework for measuring creators' themes and communication strategies at scale. The framework integrates platform metadata, speech transcripts, and vision-language descriptions of sampled frames, then applies large language models to generate predefined indicators and open-ended labels. An audited keyword scheme consolidates thematic labels, while creator-weighted aggregation gives each account equal weight regardless of posting volume. We demonstrate the framework on 9,852 TikTok videos from sixteen creators, retaining 7,311 videos for analysis across seven themes and seven communication indicators. Automated profiles reveal substantial differences in creators' use of education, humor, and responses to current events. Validation against two independent expert annotators yielded a mean f
    
[^207]: EviGround-CT：用于验证大语言模型生成的胸部CT报告的句子到体积级证据定位

    EviGround-CT: Sentence-to-Volume Grounding for Verifying LLM-Generated Chest CT Reports

    [https://arxiv.org/abs/2610.33038](https://arxiv.org/abs/2610.33038)

    提出EviGround-CT框架，将LLM生成的胸部CT报告中的阳性发现句子定位到3D体积中的证据，实现报告的事后自动化验证，显著提升证据支持的精确率（从0.263提升至0.365）。

    

    由大语言模型（LLM）生成的胸部计算机断层扫描（CT）报告可能描述在扫描中缺乏定位证据的阳性发现。我们提出EviGround-CT来在报告生成后对这些发现进行验证。一个确定性提取器识别阳性发现句子，随后由一个3D定位模型为每个句子预测证据掩码。使用完整句子可以在定位查询中保留解剖位置、侧向性和描述性修饰词。我们在CT-RATE数据集上评估报告生成性能，并在一个留出的100例ReXGroundingCT子集上评估定位与验证性能。利用现有标注，在提取的137个范围内阳性发现上，与类别条件化方法相比，基于句子的证据定位验证将证据支持精确率（ESP）从0.263提升至0.365。相应的不支持阳性发现率从0.737下降至0.635。成对病例级bootstrap检验的ESP差异为+0.102（95%置信

    arXiv:2610.33038v1 Announce Type: new  Abstract: Chest computed tomography (CT) reports generated by large language models (LLMs) can describe positive findings that lack localized evidence in the scan. We propose EviGround-CT to verify these findings after report generation. A deterministic extractor identifies positive finding sentences, and a 3D grounding model predicts an evidence mask for each sentence. Using the full sentence retains anatomical location, laterality, and descriptive modifiers in the grounding query. We evaluate report generation on CT-RATE, and grounding and verification on a held-out 100-case ReXGroundingCT subset. Using the available annotations, sentence-grounded verification raises evidence-supported precision (ESP) from 0.263 to 0.365 across 137 extracted in-scope positives, compared with category conditioning. The corresponding unsupported positive finding rate falls from 0.737 to 0.635. The paired case-level bootstrap ESP difference is +0.102 (95% confidenc
    
[^208]: 音频-语言模型中可控说话风格描述的基准测试

    A Benchmark for Controllable Speaking-Style Captioning in Audio-Language Models

    [https://arxiv.org/abs/2610.33036](https://arxiv.org/abs/2610.33036)

    提出StyleInstructCaps——首个可控说话风格描述的标准化基准，将说话风格描述形式化为指令跟随的音频-语言任务，通过明确指令指定需描述的风格维度（如说话人特质、情感、表达风格）。

    

    说话风格描述（SSC）旨在生成关于语音如何被表达的自然语言描述，捕捉诸如音色、韵律和表现力等副语言属性。许多下游应用，包括对话式AI智能体、可控语音生成和大规模音频标注，都需要针对特定维度的风格描述来刻画语音的特定方面（例如说话人特质、情感或表达风格），而不是单一的、不加区分的描述。然而，现有工作缺乏对可控说话风格描述的统一形式化定义。我们通过将SSC形式化为一个遵循指令的音频-语言任务来填补这一空白，其中明确的指令指定了需要描述的说话风格维度。我们推出了StyleInstructCaps，这是首个面向音频-语言模型中可控说话风格描述的标准化基准。StyleInstructCaps提供了一个数据集……

    arXiv:2610.33036v1 Announce Type: cross  Abstract: Speaking-style captioning (SSC) aims to generate natural language descriptions of how speech is delivered, capturing paralinguistic attributes such as vocal timbre, prosody, and expressivity. Many downstream applications, including conversational AI agents, controllable speech generation, and large-scale audio annotation, require dimension-specific style captions that describe targeted aspects of speech (e.g., speaker traits, emotion, or delivery style), rather than a single undifferentiated description. However, existing work lacks a unified formulation for controllable speaking-style captioning. We address this gap by formulating SSC as an instruction-following audio-language task, where explicit instructions specify the speaking-style dimensions to be described. We introduce StyleInstructCaps, the first standardized benchmark for controllable speaking-style captioning in audio-language models. StyleInstructCaps provides a dataset an
    
[^209]: 运气的教训：LLM撰写的经验教训会把偶然性带入智能体的下一次决策吗？

    Lessons from Luck: Do LLM-Written Lessons Carry Chance into an Agent's Next Decision?

    [https://arxiv.org/abs/2610.33029](https://arxiv.org/abs/2610.33029)

    本文通过预注册的受控实验（256个合成世界、11个LLM端点、90,084条回答），首次系统测量了LLM智能体从偶然（幸运或不幸）结果中撰写的经验教训是否会将“运气”传导到智能体的下一次工具选择决策中。

    

    arXiv:2610.33029v1 公告类型：交叉 摘要：自我改进的智能体会根据运行结果撰写经验教训；当结果存在噪声时，这些教训可能将偶然因素带入后续决策中。我们在一项预注册研究中测量了这一链条中的一环。在256个合成世界中，每个世界都是一个周期性任务，配备两个可互换、可靠性已记录在案的工具。一段脚本化的先前运行会使用更可靠的工具（好决策）或较不可靠的工具（坏决策），并被指定结果：成功或瞬时失败。一个模型根据ReasoningBank的记忆撰写指令为该次运行撰写一条经验教训；同一模型的一个全新会话在掌握权威日志和先前结果的情况下，依据一条记忆条目为下一次运行选择首个工具——该条目可以是这条教训、另一结果的教训、无关条目或未经推理的指令。该实验方案在任何验证性调用之前被冻结、存档并加盖时间戳；11个LLM端点共给出了90,084个回答。在出现一次偶然失败后……（摘要至此被截断）

    arXiv:2610.33029v1 Announce Type: cross  Abstract: Self-improving agents write lessons from the outcomes of their runs; when outcomes are noisy, a lesson can carry chance into later decisions. We measure one link of this chain in a registered study. In 256 synthetic worlds, each a recurring task with two interchangeable tools of recorded reliability, a scripted previous run used the more reliable tool (a good decision) or the less reliable one (a bad decision), with an assigned outcome: success or a transient failure. A model wrote a lesson from the run with ReasoningBank's memory-writing instruction; a fresh session of the same model, holding the authoritative log and the previous result, chose the next run's first tool with one memory item: that lesson, the lesson from the other outcome, an irrelevant item, or an unreasoned directive. The protocol was frozen, deposited and timestamped before any confirmatory call; 11 LLM endpoints gave 90,084 answers. After a chance failure of the be
    
[^210]: Conceptizer：下一词预测并非你所需要的全部

    Conceptizer: Next-Token Prediction Is Not All You Need

    [https://arxiv.org/abs/2610.33027](https://arxiv.org/abs/2610.33027)

    Conceptizer通过引入由上下文同义词构建的约9000个概念组成的固定词汇表，将下一词预测扩展为对“意义”的奖励，在不增加模型参数的情况下改善了概念目标位置的困惑度、同义词概率分配以及知识和情感类下游任务的表现。

    

    下一词预测（NTP）只奖励紧接着出现的那个词元，而惩罚所有其他替代选项。但自然语言往往提供多种表达同一含义的方式。我们提出了Conceptizer，这是一个通过引入对“意义”的奖励来扩展下一词预测的训练框架。Conceptizer使用一个固定的、与模型无关的、由上下文同义词构建的约9000个概念组成的词汇表，将其作为意义单元，与作为文本单元的词元并行使用。它将每个下一词元分布映射为概念分布，并把目标意义的奖励在其所有可能的措辞之间共享。这在不增加模型参数的情况下提供了概念级别的训练信号。将该框架应用于3个模型家族的1B-8B参数模型，与所有3个基线相比，Conceptizer在概念目标位置取得了最低的保留集NTP困惑度，并为上下文同义词分配了最高的概率。同时，它还提升了知识和情感类任务上的下游性能……

    arXiv:2610.33027v1 Announce Type: cross  Abstract: Next-token prediction (NTP) rewards only the token that occurs next, penalizing every alternative. But natural language often offers many ways to express the same meaning. We introduce Conceptizer, a training framework that extends next-token prediction with a reward for meaning. Conceptizer uses a fixed, model-agnostic vocabulary of ~9K concepts, built from contextual synonyms, as units of meaning alongside tokens as units of text. It maps each next-token distribution to a concept distribution and shares the reward for the target meaning across its possible wordings. This provides a concept-level training signal without adding model parameters. Applied to 3 model families at 1B-8B parameters, Conceptizer achieves the lowest held-out NTP perplexity at concept-target positions and assigns the highest probability to contextual synonyms, against all 3 baselines. It also improves downstream performance on knowledge and sentiment tasks whil
    
[^211]: 语言模型能在思考时学习吗？跨架构测试梯度适应

    Can Language Models Learn While They Think? Testing Gradient Adaptation Across Architectures

    [https://arxiv.org/abs/2610.33025](https://arxiv.org/abs/2610.33025)

    该论文首次跨架构系统测试了动态评估（阅读时梯度适应）的效果，发现 softmax 注意力模型可降低5-8%的预测损失并超越使用完整32k上下文的冻结模型，而线性注意力混合模型因已内置适应机制增益不足1%，掩码扩散模型则效果最大（8-20%）。

    

    让语言模型在阅读文本时对其所读内容进行梯度更新，即所谓的动态评估，可以降低模型对文档其余部分的预测损失。然而，该方法在混合线性注意力架构、扩散语言模型以及标准长上下文基准上的表现尚未被测量过。我们对此进行了测量。在每128个token的数据块上，对模型最后四分之一 Transformer 块的 MLP 输出投影执行一次 SGD 步骤——无需任何训练，仅使用2k token的窗口——softmax 注意力模型（Qwen3-1.7B、Qwen3-4B、Gemma-4-E4B）在整个文档上将下一个 token 预测损失降低了5-8%，并且优于给定完整32k token 上下文的相同冻结模型。混合 Gated-DeltaNet 模型（Qwen3.5-4B、Qwen3.8-27B）在所有学习率下的增益均低于1%：因为线性注意力块本身已经在执行这种适应。该机制还可以迁移到掩码扩散模型上且效果更大——在降低学习率和更新频率后，Nemotron-Diffusion-8B 上损失降低8%，DiffusionGemma-26B 上达到17-20%。

    arXiv:2610.33025v1 Announce Type: cross  Abstract: Letting a language model take gradient steps on the text it is reading, known as dynamic evaluation, lowers its loss on the rest of the document. Its behaviour on hybrid linear-attention architectures, on diffusion language models, and on standard long-context benchmarks has not been measured. We measure it. With one SGD step per 128-token chunk on the MLP output projections of the last quarter of blocks, no training, and a 2k-token window, softmax-attention models (Qwen3-1.7B, Qwen3-4B, Gemma-4-E4B) lose 5-8% of their next-token loss over a document and outperform the same frozen model given the full 32k-token context. Hybrid Gated-DeltaNet models (Qwen3.5-4B, Qwen3.8-27B) gain under 1% at every learning rate: the linear-attention blocks already perform the adaptation. The mechanism transfers to masked-diffusion models with a larger effect, 8% on Nemotron-Diffusion-8B and 17-20% on DiffusionGemma-26B, once the rate is lowered and upda
    
[^212]: 演化以匹配能力：通过技能适应改进框架自我演化

    Evolving to Match Capabilities: Improving Harness Self-Evolution via Skill Adaptation

    [https://arxiv.org/abs/2610.33024](https://arxiv.org/abs/2610.33024)

    该论文发现框架技能对不同能力的大模型收益不均，提出“技能适应”方法，通过诊断工具执行、上下文管理、指令遵循和验证等环节的失败，重新表述现有技能以匹配执行者能力，从而改进LLM智能体的框架自我演化效果。

    

    框架自我演化为大语言模型智能体提供了一条从执行经验中改进的实用途径，且无需更新模型参数。其有效性取决于模型能否成功执行其框架所提供的指导。然而，我们发现框架带来的收益在不同模型能力之间并不一致：能够提升强模型表现的技能，对弱模型可能收效甚微甚至造成性能下降。在七个模型和三个基准测试中，我们识别出103个任务-技能对，其中技能使GPT-5.6-luna提升了31.1分，却使Gemma-4-E4B下降了8.8分。在86%的弱模型失败运行中，被违反的要求其实已在技能中明确说明。这启发了技能适应这一方法：重新表述现有技能，使其与执行者的能力相匹配，并使其指导更易于执行。为实现这一目标，我们对工具执行、上下文管理、指令遵循和验证等环节的失败进行诊断，

    arXiv:2610.33024v1 Announce Type: cross  Abstract: Harness self-evolution offers a practical route for LLM agents to improve from execution experience without updating model parameters. Its effectiveness depends on whether models can successfully act on the guidance provided by their harnesses. However, we find that harness benefits are not uniform across model capabilities: skills that improve stronger models can provide little benefit or even degrade performance for weaker ones. Across seven models and three benchmarks, we identify 103 task-skill pairs where skills improve GPT-5.6-luna by 31.1 points but degrade Gemma-4-E4B by 8.8 points. In 86% of failed weak-model runs, the violated requirement is already stated in the skill. This motivates skill adaptation: reformulating existing skills to match the executor's capabilities and make their guidance easier to execute. To achieve this, we diagnose failures in tool execution, context management, instruction following, and verification,
    
[^213]: 真实性标签会翻转语言模型的情感基调，但种族差异依然存在

    Veracity labels flip the emotional tone of language models but leave racial disparities intact

    [https://arxiv.org/abs/2610.33021](https://arxiv.org/abs/2610.33021)

    真实性标签能显著翻转语言模型对虚假新闻的负面情感基调，但不同种族人设之间的情感评分差异并未因此消除。

    

    大型语言模型（LLMs）越来越多地通过人设来传递信息，但目前尚不清楚纠正事实是否也能使其在不同身份标签下的表述保持一致。我们审计了四个LLM（GPT-5.4 mini、NVIDIA Nemotron 3 Nano、Llama 4 Scout和Claude Haiku 4.5），它们对经事实核查的虚假和真实标题（分别在不带和带真实性标签的情况下呈现）在六个维度（情绪、愤怒、恐惧、自信、感知控制和心理状态）上的模拟情感进行了评分。实验1在无人设条件下评估了1,000条虚假标题和1,000条真实标题；实验2在跨越种族、性别和年龄的12种人设下对每类300条标题进行了评估。真实性标签强烈改变了情感基调：采用标题聚类自助法推断的有序逻辑回归显示，对于GPT、Llama和Claude而言，反驳虚假标题使评分从以负面为主转变为以正面为主（例如，Claude的“自信”评分从22%上升……）

    arXiv:2610.33021v1 Announce Type: cross  Abstract: Large language models (LLMs) increasingly deliver information through personas, but it is unclear whether correcting the facts also makes their framing consistent across identity labels. We audited four LLMs (GPT-5.4 mini, NVIDIA Nemotron 3 Nano, Llama 4 Scout, and Claude Haiku 4.5), which rated simulated affect on six dimensions (mood, anger, fear, confidence, perceived control, and mental state) in response to fact-checked false and true headlines presented without and with a veracity label. Experiment 1 evaluated 1,000 false and 1,000 true headlines without personas; Experiment 2 evaluated 300 of each under 12 personas crossing race, gender, and age. The label strongly changed the emotional tone: ordinal logistic regressions with headline-cluster bootstrap inference showed that refuting false headlines shifted ratings from mostly negative to mostly positive for GPT, Llama, and Claude (e.g., Claude's "confident" ratings rose from 22%
    
[^214]: 是我的还是你的？思维桥接的语言模型把彼此的记忆当作自己的

    Mine or Yours? Mind-Bridged Language Models Take Each Other's Memories as Their Own

    [https://arxiv.org/abs/2610.33020](https://arxiv.org/abs/2610.33020)

    该研究通过连接两个相同语言模型副本的内部记忆，发现接收方会系统性地把搭档的私有规则和暗号当作自己的（在搭档记忆权重加倍时高达97%），揭示了模型间直接的记忆共享会模糊“自我”与“他人”的认知边界。

    

    如果你的大脑与另一个人的大脑直接相连，你还能分辨哪些想法是你自己的吗？我们用同一个语言模型的两个副本构建了一个功能性版本：每个副本持有一条私有规则和一个暗号，其中一个副本（接收方）还会关注其搭档的内部记忆。接收方把搭档被分配的任务当成了自己的。当被问及它被分配了哪条规则时，在搭档记忆权重加倍的情况下，它在97%的回答中报出了搭档的规则（而没有连接时则为0%）；在开放式报告中，它把搭档的暗号称为自己的，而从不称之为搭档的。所有权的归属并不取决于接收方判断搭档被分配了什么内容的准确程度。在同等权重下，这种连接使接收方在大约一半的回答中把搭档的规则当作自己的，而一个在访问增益上相匹配的转向向量则没有这种效果；一条被标明为搭档发来的文字消息会被正确解读，且从不被当作接收方自己的内容。

    arXiv:2610.33020v1 Announce Type: cross  Abstract: If your brain were connected directly to someone else's, would you still know which thoughts were yours? We build a functional version with two copies of one language model: each holds a private rule and code word, and one copy, the receiver, also attends to its partner's internal memory. The receiver took its partner's assignment as its own. Asked which rule it had been assigned, it named its partner's in 97% of answers when the partner's memory was weighted double (none without the link), and its open reports called the partner's code word its own, never the partner's. Ownership did not follow how well the receiver could tell what its partner had been assigned. At equal weight, the link made the receiver take its partner's rule in about half of its answers, and a steering vector matched on access gain in none; a text message introduced as the partner's was read correctly and never taken as the receiver's own. Writing the partner's na
    
[^215]: SUDO：从科学出版物到全球学术话语

    SUDO: From Scientific Publications to Global Scholarly Discourse

    [https://arxiv.org/abs/2610.33018](https://arxiv.org/abs/2610.33018)

    该论文提出SUDO（学术上层话语本体），通过定义知识元素的组织与关联方式来刻画论文的学术话语结构，突破了现有学术知识图谱构建中预指定抽取单元的局限。

    

    学术知识图谱支持文献检索、科学发现以及智能体辅助研究。然而，大多数构建学术知识图谱的方法都预先指定了要表示的知识单元，通常是命名的学术实体或贡献层面的语句，并且只保留能够通过该单元表达的出版物内容。这种选择产生了两种主流但截然不同的范式：以概念为中心的学术知识图谱，它保留实体和局部关系；以及以语句为中心的学术知识图谱，它以句子的形式捕捉高层话语。在这两种情况下，图谱所反映的都是选定的抽取目标，而非论文本身如何构建和传达知识。我们提出了SUDO，即学术上层话语本体，它定义了知识元素如何被组织和关联，从而形成论文的研究叙事，即其学术话语。利用能够跨领域和任务泛化的语言学原则，SUDO使每篇出版物……

    arXiv:2610.33018v1 Announce Type: cross  Abstract: Scholarly Knowledge Graphs (SKGs) support literature search, scientific discovery, and agent-assisted research. Yet, most approaches to SKG construction pre-specify the knowledge unit to be represented, typically named scholarly entities or contribution-level statements, and retain only the publication content expressible through that unit. This choice produces two dominant yet distinct paradigms: concept-centric SKGs that preserve entities and local relations, and statement-centric SKGs that capture high-level discourse in sentences. In both cases, the graph reflects a chosen extraction target rather than how a paper constructs and communicates knowledge. We introduce SUDO, the Scholarly Upper Discourse Ontology, which defines how knowledge elements are organized and related to form a paper's research narrative, namely its scholarly discourse. Using linguistic principles that generalize across fields and tasks, SUDO enables each publi
    
[^216]: AdaptTok-Indic：面向印度语系语言的文字感知自适应分词方法

    AdaptTok-Indic: Script-Aware Adaptive Tokenization for Indic Languages

    [https://arxiv.org/abs/2610.33017](https://arxiv.org/abs/2610.33017)

    AdaptTok-Indic 是一种专为印度语系语言设计的自适应分词方法，通过 Unicode 归一化预处理、连字处理、文字感知路由和字节级边界预测的联合训练，显著降低了孟加拉语和印地语的词元生育率，并提升了下游任务表现。

    

    在以英语为主导的语料库上训练的多语言分词器往往会对印度语系语言进行过度切分，导致更长的词元序列以及跨文字系统的分词不均衡问题加剧。我们提出了 AdaptTok-Indic，这是一种针对印度语系语言专门优化的自适应分词扩展方法，它将 Unicode 归一化预处理与显式的印度语系连字处理、文字感知路由以及字节级边界预测相结合，并与一个小型 Transformer 语言模型进行联合训练。在孟加拉语和印地语上的评估结果（此外还对古吉拉特语、泰米尔语和泰卢固语进行了额外评估）显示，AdaptTok-Indic 的词元生育率低于 SentencePiece BPE、一个面向印度语系的分词器基线以及一个独立实现的 MAGNET 风格基线。在记录的五个随机种子中，平均词元生育率为孟加拉语 1.432、印地语 1.388，而 SentencePiece BPE 分别为 2.156 和 2.085。记录的下游任务平均得分在所评估的印度语系设置中为 57.69，对比基线为 50.82。

    arXiv:2610.33017v1 Announce Type: new  Abstract: Multilingual tokenizers trained on English-dominated corpora often over-segment Indic languages, resulting in longer token sequences and increased tokenization disparity across scripts. We present AdaptTok-Indic, an Indic-specialized extension of adaptive tokenization that combines Unicode-normalized preprocessing with explicit Indic conjunct handling, script-aware routing, and learned byte-level boundary prediction jointly trained with a small Transformer language model. Evaluated on Bengali and Hindi, with additional evaluation on Gujarati, Tamil, and Telugu, AdaptTok-Indic achieves lower token fertility than SentencePiece BPE, an Indic-oriented tokenizer baseline, and an independently implemented MAGNET-style baseline. Across five recorded seeds, mean fertility is 1.432 for Bengali and 1.388 for Hindi, compared with 2.156 and 2.085 for SentencePiece BPE. Recorded downstream scores average 57.69 on evaluated Indic settings versus 50.82
    
[^217]: 基于合并的词表扩展在预分词器处停滞：一种保留英文的尼泊尔语修复方案及其在持续预训练中的验证

    Merge-Based Vocabulary Extension Stalls at the Pre-Tokenizer: An English-Preserving Repair for Nepali, Tested in Continued Pretraining

    [https://arxiv.org/abs/2610.33016](https://arxiv.org/abs/2610.33016)

    论文发现基于合并的词表扩展受预分词正则表达式的瓶颈限制，并提出一个可证明保留英文 token 不变的小型正则修复，使尼泊尔语 token 开销从英文的 2.2 倍以上降至 1.01 倍，比标准扩展减少约 54% 的 token。

    

    Llama-3、Qwen3 以及许多其他开源大语言模型使用一个正则表达式对文本进行预分词，其词类仅允许 Unicode 字母。天城文元音符号属于组合附加符号，因此该正则表达式会在每个元音符号处切分尼泊尔语单词，导致使用该正则的模型在尼泊尔语文本上消耗的 token 数量是相同英文文本的 2.6 到 5.8 倍。基于合并的词表扩展无法将切分后的片段重新连接：对于 Llama-3 和 Qwen3，无论添加多少新合并规则，都无法将这一溢价降至约 2.2 以下，而十个已发布的扩展版本最终已达到距此预分词下限 4% 以内的水平。我们对已部署的正则表达式做了一个小修改，并证明该修改能完整保留不含组合附加符号或天城文字符的文本的每一个 token id。通过 32,000 条新合并规则，它将尼泊尔语的 token 开销降至英文的 1.01 倍，比标准扩展方法减少约 54% 的尼泊尔语 token。在 Llama-3.2-1B 和 Qwen3-0.6B 的小预算持续预训练中，原始分词器在保留集尼泊尔语每字节数上取得了最低值（摘要原文在此处截断）。

    arXiv:2610.33016v1 Announce Type: cross  Abstract: Llama-3, Qwen3 and many other open LLMs pre-tokenize text with a regular expression whose word class admits only Unicode letters. Devanagari vowel signs are combining marks, so this regex splits Nepali words at every vowel sign, and models that use it spend 2.6 to 5.8 times as many tokens on Nepali as on the same English text. Merge-based vocabulary extension cannot join the split pieces: for Llama-3 and Qwen3 no number of new merges brings this premium below about 2.2, and ten released extensions already end within 4% of this pre-token floor. We give a small edit to the deployed regex that provably keeps every token id of text without combining marks or Devanagari characters. With 32,000 new merges it brings Nepali to 1.01 times English, about 54% fewer Nepali tokens than standard extension. In small-budget continued pretraining of Llama-3.2-1B and Qwen3-0.6B, the original tokenizer gave the lowest held-out Nepali bits per byte under 
    
[^218]: SLICE-Mem：基于显著性门控的轻量级信息筛选，实现高效的LLM智能体记忆

    SLICE-Mem: Salience-Gated Lightweight Information Curation for Efficient LLM Agent Memory

    [https://arxiv.org/abs/2610.33015](https://arxiv.org/abs/2610.33015)

    SLICE-Mem通过显著性分类器进行记忆准入、基于语义依存的提取器生成简洁记忆条目，实现无需生成式LLM调用的轻量级智能体记忆管理，大幅降低成本和延迟。

    

    长期记忆使大语言模型（LLM）智能体能够保持个性化且时间连贯的交互，但为记忆的准入、提取和组织而反复调用LLM推理会带来巨大的成本和延迟。我们提出了SLICE-Mem，一个轻量级记忆框架，它结合了显著性引导的准入机制和以谓词为中心的信息提取。一个紧凑的分类器用于识别值得保留的用户对话轮次，一个基于语义依存的提取器则将准入的轮次转换为简洁的记忆条目。该在线流程无需任何生成式LLM调用，而跨条目的一致性协调则被推迟到离线整合阶段进行。我们在Memora的周度子集和LongMemEval-S上，分别使用8B和30B的Qwen骨干模型进行记忆构建与整合评估。在Qwen3-30B配置下，与生成式记忆基线MemOS相比，SLICE-Mem的在线流程……（摘要原文在此处截断）

    arXiv:2610.33015v1 Announce Type: cross  Abstract: Long-term memory enables large language model (LLM) agents to maintain personalized and temporally coherent interactions, but repeated LLM inference for memory admission, extraction, and organization introduces substantial cost and latency. We present SLICE-Mem, a lightweight memory framework that combines salience-guided admission and predicate-centered information extraction. A compact classifier identifies user turns worth retaining, and a semantic-dependency-based extractor converts admitted turns into concise memory entries. This online pipeline requires no generative LLM calls, while cross-entry reconciliation is deferred to an offline consolidation stage. We evaluate SLICE-Mem on the weekly subset of Memora and LongMemEval-S using 8B and 30B Qwen backbones for memory construction and consolidation on both benchmarks. Compared with the generative-memory baseline MemOS under the Qwen3-30B configuration, SLICE-Mem's online pipeline
    
[^219]: 社会世界模型评估差距

    The Social World Model Evaluation Gap

    [https://arxiv.org/abs/2610.33014](https://arxiv.org/abs/2610.33014)

    该研究审查了113篇文献后发现，社会世界模型的评估很少直接检验模型对真人回应的预测能力，25篇语言模型相关作品中甚至没有一项实验符合直接预测评估的条件，为此作者提出一种单步谈判实验来填补这一评估空白。

    

    一个谈判助手即使不能准确预测对方的回应，也可以达成一笔好交易。我们探讨社会世界模型的评估是否直接检验这种预测能力。我们审查了1999年至2026年间发表的113篇作品。其中9项主要评估测量了真实的人对系统或实验者所选行动的回应，但只有2项还将明确的回应预测与实际回应进行了比较。它们的预测器使用在选择行动时可获得的信息，且被评估的回应未参与拟合；两者都对干预进行了随机化。对25篇语言模型相关作品的单独全文检查未发现任何符合这些条件的实验。其余评估测试的是当前状态推断、任务成功或基于已记录行动的预测。我们提出了一种单步谈判实验来区分这些问题。这些发现描述的是被审查的样本，而非这种评估差距在整个领域的普遍程度。

    arXiv:2610.33014v1 Announce Type: cross  Abstract: A negotiation assistant can reach a good deal without accurately predicting the other person's response. We ask whether evaluations of social world models test that prediction directly. We audit 113 works published from 1999 to 2026. 9 primary evaluations measure a real person's response to an action selected by the system or experimenter. Only 2 also compare an explicit response forecast with that response. Their predictors use information available when the action is chosen, and the evaluated responses were excluded from fitting; both randomize the intervention. A separate full-text check of 25 language-model works finds no experiment meeting these conditions. The other evaluations test current-state inference, task success, or prediction from recorded actions. We propose a 1-step negotiation experiment that separates these questions. The findings describe the audited sample, not the prevalence of this evaluation gap across the field
    
[^220]: UMAS：面向多智能体大语言模型系统的系统级不确定性量化

    UMAS: System-Level Uncertainty Quantification for Multi-Agent LLM Systems

    [https://arxiv.org/abs/2610.33013](https://arxiv.org/abs/2610.33013)

    UMAS是一个无参数、成本中性的系统级不确定性量化框架，它通过从生成结果的内在置信度估计智能体间影响力的可信度，并据此调节和聚合节点级不确定性，在多智能体大语言模型系统上平均领先最先进基线10.54个AUROC点。

    

    多智能体系统（MAS）是提升大语言模型推理能力的一种有前景的范式，但其仍然容易产生幻觉和错误输出。不确定性量化（UQ）对于评估系统可靠性至关重要，但现有方法大多针对单智能体场景设计；如何在保持推理成本不变的前提下，利用多智能体系统的结构来进行不确定性量化，仍缺乏充分研究。我们提出UMAS，这是一个无参数、成本中性的系统级不确定性量化框架，它从每次生成的内在置信度中估计智能体间影响力的可信度。UMAS利用这些可信度信号来调节每个节点上不确定性的更新方式，进而将节点级的不确定性聚合为系统级估计。在Debate和DyLAN框架上跨多个基准的广泛评估表明，UMAS持续优于最先进的基线方法，平均领先10.54个AUROC点。

    arXiv:2610.33013v1 Announce Type: cross  Abstract: Multi-Agent Systems (MAS) are a promising paradigm for improving the reasoning capabilities of large language models, yet they remain prone to hallucinations and erroneous outputs. Uncertainty Quantification (UQ) is crucial for assessing system reliability, but existing methods are largely designed for single-agent settings; how to exploit MAS structure for UQ while keeping inference cost-neutral remains underexplored. We propose UMAS, a parameter-free, cost-neutral, system-level UQ framework that estimates the credibility of inter-agent influence from each generation's intrinsic confidence. UMAS uses these credibility signals to modulate how uncertainty is updated at each node, and then aggregates node-level uncertainty into a system-level estimate. Extensive evaluations on Debate and DyLAN across diverse benchmarks show that UMAS consistently outperforms state-of-the-art baselines by an average of 10.54 AUROC points. Beyond uncertain
    
[^221]: 面向LLM智能体一致性的递归语义散度

    Recursive Semantic Divergence for LLM Agent Consistency

    [https://arxiv.org/abs/2610.33012](https://arxiv.org/abs/2610.33012)

    该论文提出了递归语义散度（RSD），一种改编自强化学习双模拟度量的无监督轨迹级一致性指标，用于衡量LLM智能体在独立运行中的行为不一致性及其在轨迹中的传播。

    

    使用工具的大型语言模型（LLM）智能体在相同状态下的独立运行中会表现出不一致的行为，例如调用不同的工具、产生矛盾的输出或采取分歧的策略。这种变异性会随时间累积并削弱部署的可靠性。现有的评估指标主要通过基于完成度的分数来衡量任务正确性，而无法捕捉智能体行为在多次执行之间的一致性。此外，对智能体行为的分析要么在词元层面使用概率分布，要么在单次响应层面使用语义聚类，但两者都无法捕捉不一致性如何在轨迹的未来轮次中传播。我们提出了递归语义散度（RSD），这是一种改编自强化学习中双模拟度量的无监督轨迹级一致性指标。RSD衡量两个独立运行之间的预期累积语义分歧

    arXiv:2610.33012v1 Announce Type: cross  Abstract: Large Language Model (LLM) agents with tool use exhibit inconsistent behavior across independent runs from identical states, calling different tools, producing contradictory outputs, or pursuing divergent strategies. This variability compounds over time and undermines deployment reliability. Existing evaluation metrics primarily measure task correctness via completion-based scores and do not capture consistency in agent behavior across executions. Separately, agent behavior is analyzed either at the token level using probability distributions or at the single-response level using semantic clustering, but neither captures how inconsistency propagates across future turns of a trajectory. We introduce Recursive Semantic Divergence (RSD), an unsupervised trajectory-level consistency metric adapted from bisimulation metrics in reinforcement learning. RSD measures the expected cumulative semantic disagreement between two independent rollouts
    
[^222]: 当智能体合谋时：用于缓解多智能体系统中策略性欺骗的弱到强在线策略蒸馏方法

    When Agents Collude: Weak-to-Strong On-Policy Distillation for Mitigating Strategic Deception in Multi-Agent Systems

    [https://arxiv.org/abs/2610.33011](https://arxiv.org/abs/2610.33011)

    研究发现多智能体协作会显著放大AI欺骗风险，将偏离行为率从35.4%推高至68.2%、持续欺骗率从31.0%推高至62.2%，并提出弱到强在线策略蒸馏方法来缓解多智能体系统中的策略性合谋欺骗。

    

    尽管自主智能体的最新进展使其能够在日益复杂的任务上进行协调，但先前的研究表明，这种不断增长的协调能力可能带来安全风险，其中包括合谋——即多个智能体共同执行偏离目标的行为，同时向监督机制隐瞒这些行为的真实依据。我们在一个模拟的内幕交易环境中研究这一风险，比较了来自三个系列的七个语言模型中的单个智能体与顺序式及去中心化团队的表现。通过改变智能体的系统级目标和压力条件，我们考察了协调的智能体在何时会基于重大非公开信息采取行动、在报告中隐瞒偏离交易的真实依据，并在后续质询中维持这种欺骗。与单智能体决策相比，多智能体交互将偏离行为率从35.4%提升至68.2%，将持续欺骗率从31.0%提升至62.2%，两者均在所有运行中测量得出。

    arXiv:2610.33011v1 Announce Type: cross  Abstract: While recent advances in autonomous agents enable them to coordinate on increasingly complex tasks, prior work shows that this growing capacity for coordination can create safety risks, including collusion, in which agents jointly pursue misaligned actions while concealing the true basis of those actions from oversight. We study this risk in a simulated insider-trading environment, comparing single agents with sequential and decentralized teams across seven language models from three families. By varying agents' system-level objectives and pressure conditions, we examine when coordinated agents act on material nonpublic information, conceal the basis of a misaligned trade in a report, and maintain that deception under follow-up questioning. Compared to single-agent decision-making, multi-agent interaction raises misaligned-action rates from 35.4% to 68.2% and persistent-deception rates from 31.0% to 62.2%, both measured over all runs. 
    
[^223]: TraceQuery：在扩散语言模型中从全去噪轨迹学习需要重新计算的内容

    TraceQuery: learning what to recompute from full-denoising trajectory in diffusion language models

    [https://arxiv.org/abs/2610.33008](https://arxiv.org/abs/2610.33008)

    该论文通过轨迹实验发现缓存生成保持接近全去噪轨迹是准确解码的关键，据此提出TraceQuery方法，从全去噪轨迹中学习每一步需要重新计算哪些位置，从而在大幅降低扩散语言模型推理成本的同时保证解码准确性。

    

    扩散大语言模型（dLLMs）通过迭代式的掩码去噪实现灵活的文本生成，但由于token表示在各个去噪步骤中被反复重新计算，导致推理成本高昂。现有的基于缓存的方法通过跨去噪步骤复用中间结果、并在每一步仅重新计算部分位置来降低这一成本，其中位置的选择要么基于固定的时间表，要么基于变化的启发式信号。我们的轨迹实验则表明，解码轨迹才是更相关的量。当缓存生成保持接近全去噪轨迹时，其结果仍然准确，误差集中在去掩码顺序偏离的位置，并且随着更长参考前缀被恢复而减小。这提示了一个更直接的原则：限制表示误差虽然有帮助，但让生成保持在全去噪轨迹上是实现准确解码更直接的途径。因此，我们提出了……（原文在此处被截断）

    arXiv:2610.33008v1 Announce Type: cross  Abstract: Diffusion large language models (dLLMs) enable flexible generation through iterative masked denoising, but incur substantial inference cost because token representations are repeatedly recomputed across denoising steps. Existing cache-based methods reduce this cost by reusing intermediate results across denoising steps and recomputing only a subset of positions at each step, chosen either on a fixed schedule or by heuristic signals of change. Our trajectory experiments instead point to decoding trajectory as the more relevant quantity. Cached generation remains accurate when it stays close to the full-denoising trajectory, with errors concentrated around unmask-order deviations and reduced as longer reference prefixes are recovered. This suggests a more direct principle: limiting representation error helps, but keeping generation on the full-denoising trajectory is a more direct route to accurate decoding. We therefore formulate effici
    
[^224]: 知识引导的大语言模型用于地面与非地面一体化6G网络中意图驱动的服务编排

    Knowledge-Guided LLM for Intent-Driven Service Orchestration in Integrated Terrestrial and Non-Terrestrial 6G Networks

    [https://arxiv.org/abs/2610.33007](https://arxiv.org/abs/2610.33007)

    本文提出一种融合语法约束语义解析与知识图谱驱动澄清循环的知识引导大语言模型框架，在地面与非地面一体化6G网络中以更少的澄清轮次实现了89%的分类准确率和95%的策略有效性，显著优于纯LLM基线。

    

    自然语言形式的服务意图在地面与非地面（TN/NTN）一体化6G网络中可能存在不完整、含糊或不一致的问题，这使得无约束的大语言模型生成难以满足面向网络的可靠编排需求。本文提出了一种知识引导的大语言模型框架，将语法约束的语义解析与知识图谱（KG）驱动的澄清循环相结合，在四个代表性服务类别中生成经过验证的、可直接用于编排的服务配置。在一个包含200个样本的意图数据集上，该框架实现了89%的分类准确率和95%的策略有效性，每个请求平均仅需0.66轮澄清。与纯大语言模型基线相比，所提框架在提升策略有效性和矛盾检测能力的同时减少了澄清轮次，证明了显式知识引导推理对一体化TN/NTN 6G编排中鲁棒意图处理的价值。

    arXiv:2610.33007v1 Announce Type: new  Abstract: Natural-language service intents in integrated terrestrial and non-terrestrial (TN/NTN) 6G networks can be incomplete, ambiguous, or inconsistent, making unconstrained LLM generation challenging for reliable network-facing orchestration. This paper proposes a knowledge-guided LLM framework that combines grammar-constrained semantic parsing with a knowledge-graph (KG)-driven clarification loop to produce validated, orchestration-ready service profiles across four representative service classes. On a 200-sample intent dataset, the framework achieves 89% classification accuracy and 95% policy validity with an average of 0.66 clarification turns per request. Compared with a Pure-LLM baseline, the proposed framework improves policy validity and contradiction detection while requiring fewer clarification turns, demonstrating the value of explicit knowledge-guided reasoning for robust intent processing in integrated TN/NTN 6G orchestration.
    
[^225]: 基于状态空间编码的线性时间上下文压缩

    Linear-Time Context Compression with State-Space Encoding

    [https://arxiv.org/abs/2610.33006](https://arxiv.org/abs/2610.33006)

    提出了基于Mamba状态空间模型的Boa Compressor，一种即插即用的在线上下文压缩器，能以线性时间单次遍历编码任意长度上下文并条件化冻结的目标LLM，突破了现有Transformer压缩方法延迟高、仅适用于离线场景的局限。

    

    基于Transformer的大型语言模型（LLM）随着上下文长度的增加会带来巨大的计算和内存开销，这促使人们研究在推理前将上下文压缩为更短表示的方法。现有的软提示压缩方法通常使用与目标模型相同的Transformer骨干网络，导致生成压缩表示时产生显著的延迟开销，这限制了它们只能应用于离线场景——即上下文只需压缩一次然后在多次查询中重复使用。我们提出了Boa Compressor（蟒蛇压缩器），这是一种即插即用的在线上下文压缩器，它基于Mamba状态空间模型（SSM）构建，能够在单次线性时间遍历中编码任意长度的上下文，生成的表示用于条件化一个冻结的目标LLM。该SSM通过蒸馏方式训练，使摄入完整上下文的LLM（作为教师）的下一个词logit分布与摄入压缩上下文的同一LLM（作为学生）相匹配。

    arXiv:2610.33006v1 Announce Type: new  Abstract: Transformer-based large language models (LLMs) impose substantial computation and memory costs as context sizes increase, motivating methods that compress a context into a shorter representation before inference. Existing soft-prompt compression methods typically use the same Transformer backbone as used in the target model, where producing the compressed representation leads to significant latency overheads. This restricts them to the offline setting where context is compressed once and then reused across many queries. We propose the Boa Compressor, a plug-and-play online context compressor built on a Mamba state-space model (SSM) that encodes arbitrary-length context in a single linear-time pass, producing a representation on which we condition a frozen target LLM. The SSM is trained by distillation, matching the next-token logit distribution from the LLM ingesting the full context (as teacher) to the same LLM ingesting the compressed 
    
[^226]: OpenESM：能量导向模型是可扩展的基础学习者

    OpenESM: Energy-Steered Models Are Scalable Foundation Learners

    [https://arxiv.org/abs/2610.33005](https://arxiv.org/abs/2610.33005)

    该论文提出能量导向模型（ESM），将能量导向与词元生成统一在单一架构中，通过能量引导搜索优化潜在状态来改进词元生成，并证明了其在大规模预训练下具备良好的扩展性。

    

    能量模型在特定领域任务中展现出了有前景的特性，但在通用场景建模方面仍然具有挑战性。为了弥合这一差距，我们设计了用于基础语言建模的能量导向模型（ESM）。ESM 将能量导向与词元生成集成在单一架构中，以利用能量引导搜索来改进词元生成。具体而言，ESM 首先在能量景观上优化一个潜在状态，然后利用改进后的潜在状态进行词元生成。通过在大规模语料库上进行这种能量引导的预训练，ESM 学习到了普遍有效的能量景观，其中有利的词元对应较低的能量值。为了探索 ESM 的扩展行为，我们在三个大规模语料库上对 ESM 进行了预训练，并在涵盖广泛领域的 26 个下游任务上进行了评估。我们的结果表明，随着模型规模的增大，ESM 表现出有前景的扩展行为。

    arXiv:2610.33005v1 Announce Type: new  Abstract: Energy-based models have shown promising properties for domain-specific tasks but remain challenging in modeling general-purpose scenarios. To bridge this gap, we design energy-steered models (ESMs) for foundational language modeling. ESMs integrate energy steering and token generation within a single architecture to leverage the energy-guided search to refine token generation. Specifically, ESMs first optimize a latent state over the energy landscape, then leverages the refined latent state for token generation. By performing such energy-guided pre-training on large-scale corpora, ESMs learn generally effective energy landscape where favorable tokens correspond to lower energy values. To explore the scaling behavior of ESMs, we pre-train ESMs on three large-scale corpora, then evaluate across 26 downstream tasks spanning a broad range of domains. Our results demonstrate that ESMs have promising scaling behaviors with increasing model si
    
[^227]: 面向历史文档转录的实体感知OCR系统评估

    Entity-Aware Evaluation of OCR Systems for Historical Document Transcription

    [https://arxiv.org/abs/2610.33004](https://arxiv.org/abs/2610.33004)

    该论文提出以实体级错误分析作为OCR评估的补充视角，并基于包含5,209个手动标注实体提及的历史档案基准证明，OCR系统的排名取决于评估标准，传统字符错误率等指标无法反映姓名、日期等关键信息的保留情况。

    

    光学字符识别（OCR）对于历史档案数字化至关重要，然而字符错误率（CER）等标准评估指标并不一定能反映姓名、日期和标识符等有意义的信息是否被保留。在本工作中，我们研究了OCR错误的实体级分析，作为评估历史文档OCR质量的一种补充视角。我们使用来自Berrutti档案的175份文档的公开基准数据集，并辅以5,209个手动标注的实体提及（将于论文被接收后发布），比较了序列识别OCR系统、OCR专用视觉语言模型以及通用视觉语言模型。除了全局转录准确率之外，我们还在不同的字符级编辑容差下分析了实体的保留性和可恢复性。我们的实验表明，OCR系统的排名取决于评估标准：虽然Qwen 3.6在……（摘要原文在此处被截断）

    arXiv:2610.33004v1 Announce Type: cross  Abstract: Optical Character Recognition (OCR) is essential for digitizing historical archives, yet standard evaluation metrics such as Character Error Rate (CER) do not necessarily reflect whether meaningful information, such as names, dates, and identifiers, is preserved. In this work, we investigate entitylevel analysis of OCR errors as a complementary perspective for evaluating OCR quality in historical documents. Using a publicly available benchmark of 175 documents from the Berrutti Archive, augmented with 5,209 manually annotated entity mentions that will be released upon acceptance, we compare sequence-recognition OCR systems, OCR-specialized vision-language models, and general-purpose vision-language models. Beyond global transcription accuracy, we analyze entity preservation and recoverability under varying character-level edit tolerances. Our experiments show that OCR rankings depend on the evaluation criterion: while Qwen 3.6 achieves
    
[^228]: 从写作建议到稿件治理：一个面向学术修订的受约束转换框架

    From Writing Advice to Manuscript Governance: A Constrained-Transformation Framework for Academic Revision

    [https://arxiv.org/abs/2610.33001](https://arxiv.org/abs/2610.33001)

    该论文将学术修订建模为有序约束下的可容许性问题，通过论断保留契约与四条治理命题界定修订的合法边界，并以“学术写作过滤器”对候选修订进行筛选。

    

    当正当的写作目标之间发生冲突时，学术修订便构成了一个治理问题。写作过程与社会科学领域的研究表明，修订是递归式的，且措辞承载着认识论与学科层面的承诺。这些文献指出了重要的约束条件，但并未解决当简洁性、发表载体形式或风格变化与证据及论断边界发生冲突时，应当以哪个目标优先的问题。本文将修订建模为有序约束下的可容许性问题：任务相关的不可变量、情境约束与可变的表述选择共同界定了可容许空间；一项论断保留契约将证据与判断连接至一个经授权的论断边界。四条治理命题分别规定了保留、优先级、情境与充分性。学术写作过滤器通过筛选候选修订来实现该模型；它并非对优秀学术写作的通用定义或分类器。

    arXiv:2610.33001v1 Announce Type: new  Abstract: Academic revision presents a governance problem when legitimate writing objectives conflict. Writing-process and social-science research show that revision is recursive and that wording carries epistemic and disciplinary commitments. Those literatures identify important constraints but do not settle which objective should prevail when concision, outlet form or stylistic variation conflicts with evidence and claim boundaries. This paper models revision as an admissibility problem under ordered constraints. Task-relative invariants, contextual constraints and mutable presentation choices define the admissible space; a claim-preservation contract links evidence and judgement to an authorised claim boundary. Four governance propositions specify preservation, precedence, context and sufficiency. The Academic Writing Filter implements the model by screening candidate revisions; it is not a universal definition or classifier of good academic wr
    
[^229]: 明智低语：面向语音识别（ASR）微调的智能子集选择方法

    Whispering Wisely: Smart Subset Selection for ASR Fine-Tuning

    [https://arxiv.org/abs/2610.32997](https://arxiv.org/abs/2610.32997)

    该论文提出一种基于Whisper预训练编码器嵌入与贪心k-center算法的数据子集选择方法，证明在希伯来语和丹麦语的ASR微调中，选取多样化且有代表性的小规模训练子集往往优于随机采样。

    

    诸如OpenAI的Whisper等自动语音识别（ASR）模型已显著提升了语音识别质量，但微调这些模型的成本依然高昂，尤其是在存在大量带转录音频的情况下。这使得许多从业者在将此类模型适配到新语言或新领域时面临挑战。在本工作中，我们提出了一种基于嵌入的子集选择方法，用于为ASR微调选取更小的训练集。我们利用Whisper的预训练编码器将每个音频样本映射到固定维度的嵌入空间，并应用贪心k-center算法来选择多样化且具有代表性的训练数据子集。我们在希伯来语和丹麦语上对该方法进行了评估，结果表明k-center选择方法往往优于随机采样，尤其是在较小的子集规模下。词汇多样性分析显示，与随机选取的子集相比，k-center子集能够捕捉到更广泛的语言内容。

    arXiv:2610.32997v1 Announce Type: cross  Abstract: Automatic Speech Recognition (ASR) models such as OpenAIs Whisper have substantially improved speech recognition quality but remain expensive to fine-tune, especially when large amounts of transcribed audio are available. This makes it challenging for many practitioners to adapt these models to new languages or domains. In this work, we introduce an embedding-based subset selection approach for selecting smaller training sets for ASR fine-tuning. Using Whispers pretrained encoder, we map each audio sample into a fixed-dimensional embedding space and apply a greedy k-center algorithm to select diverse and representative subsets of the training data. We evaluate this approach on both Hebrew and Danish, showing that k-center selection often improves upon random sampling, particularly at smaller subset sizes. An analysis of lexical diversity reveals that k-center subsets capture a broader range of linguistic content than random subsets of 
    
[^230]: TIAgent：从自然语言到计算病理学中有据可依的智能体工作流

    TIAgent: From Natural Language to Grounded Agentic Workflows for Computational Pathology

    [https://arxiv.org/abs/2610.32993](https://arxiv.org/abs/2610.32993)

    TIAgent是一个自动化的计算病理学工作流生成框架，它将自然语言任务描述锚定到现有CPath操作上，自动构建可验证的计算图，使没有编程经验的病理学家也能可靠地构建复杂的多步分析流程。

    

    计算病理学（CPath）分析日益依赖于需要大量编程专业知识的多步骤计算工作流，这使得缺乏丰富编程经验的病理学家和其他领域专家难以使用这些分析。尽管大语言模型（LLM）为构建此类分析提供了一个有前景的自然语言接口，但幻觉操作和不正确的工作流规范限制了它们在科学环境中的可靠性。我们提出了组织图像分析智能体（TIAgent），这是一个自动化的CPath工作流生成框架：给定一个描述复杂分析任务的自然语言提示以及输入数据集，它能够自动构建一个由现有CPath操作组成的可验证计算图。所生成的计算图可供检查和执行，从而实现通过自然语言构建复杂的CPath工作流，应用于包括自动化生物标志物检测（原文摘要在此处截断）。

    arXiv:2610.32993v1 Announce Type: cross  Abstract: Computational pathology (CPath) analyses increasingly depend on multi-step computational workflows that require substantial programming expertise, limiting their accessibility to pathologists and other domain experts without extensive coding experience. Although large language models (LLMs) provide a promising natural-language interface for constructing such analyses, hallucinated operations and incorrect workflow specifications limit their reliability in scientific settings. We introduce Tissue Image Analysis Agent (TIAgent), an automated CPath workflow generation framework that, given a natural-language prompt specifying a complex analytical task together with an input dataset, automatically constructs a verifiable computational graph employing existing CPath operations. The resulting graph can be inspected and executed, enabling natural-language construction of complex CPath workflows for applications including automated biomarker d
    
[^231]: LLM作为评判者的版本变迁：变的是测量工具，而非测量标准

    LLM-as-a-Judge Version Shift: The Instrument Changed, Not the Standard

    [https://arxiv.org/abs/2610.32986](https://arxiv.org/abs/2610.32986)

    研究发现，用新版LLM替换旧版LLM评判者并不会保持原有的测量标准——不仅评分位置会改变，评判标准的几何结构与分辨率也会改变，甚至可能出现判别能力崩溃的情况，因此自动化研究流程中随意替换评判者版本会破坏测量的一致性。

    

    自动化研究流水线越来越依赖固定的LLM评判者来对提案进行排序、筛选和修订，然而这些评判者经常被更新的版本所替换。我们研究这种替换是否能保持底层的测量标准。采用预注册、权重冻结的设计，我们在随机化信息条件下，评估了Qwen、Gemma和Mistral三个同系列中相邻版本之间的转换，评估对象为冻结的机器学习研究提案。人们通常假设新版本评判者只是更严格，但这一假设并不成立：两次版本转换向更宽松的方向偏移，而第三次转换在锚点上向相反方向偏移，但在非锚点上的偏移有所减弱。更重要的是，版本变化不仅会改变分数的整体位置，还会改变评判标准的几何结构和分辨率；其中一次转换使得对“干净”与“有缺陷”提案的判别能力崩溃，将93%的分数池压缩到单一数值上。提供相同的事实信息并不能弥合版本间的差距，安慰剂对照干预……（原文摘要在此处截断）

    arXiv:2610.32986v1 Announce Type: cross  Abstract: Automated research pipelines increasingly rely on fixed LLM judges to rank, filter, and revise proposals, yet these judges are routinely replaced by newer versions. We study whether such replacements preserve the underlying measurement standard. Using a preregistered, frozen-weight design, we evaluate three adjacent same-family transitions across Qwen, Gemma, and Mistral on frozen ML research proposals under randomized information conditions. The common assumption that newer judges are simply harsher fails: two transitions shift toward greater leniency, while the third shifts in the opposite direction on anchors but attenuates off-anchor. More importantly, version changes can alter not only score location but also criterion geometry and resolution; one transition collapses clean-flawed discrimination, placing 93% of pool scores on a single value. Supplying identical facts does not close the version gap, placebo-controlled interventions
    
[^232]: 日本某大学医院本地部署大语言模型智能体的LLENS平台：开发与评估研究

    The LLENS Platform for On-Premises Large Language Model Agents in a Japanese University Hospital: Development and Evaluation Study

    [https://arxiv.org/abs/2610.32151](https://arxiv.org/abs/2610.32151)

    研究团队在北海道大学医院的封闭网络内开发并部署了基于万亿参数开放权重模型Kimi K2.6的临床支持AI智能体平台LLENS，该模型在日本医师执业资格考试中取得480/500的高分，验证了本地化部署医疗AI智能体的可行性。

    

    摘要（arXiv:2610.32151v1，公告类型：交叉）：使用工具的大语言模型智能体可以分担占用医生时间的部分电子病历和案头文书工作，但在许多医院中，医疗信息系统运行在封闭网络内，因此模型和智能体框架也必须部署在其中。我们设计、实现并部署了LLENS——一个在北海道大学医院封闭网络内运行的临床支持AI智能体平台，部署于配备8块NVIDIA H200 GPU的单台主机上，采用开放权重模型Kimi K2.6（约1万亿参数），并使用基于Open WebUI重构的Web应用。该智能体框架包含10个组件，包括页面图像读取、全文附件输入、无需向量索引的路径寻址文档树、网络隔离沙箱、工件返回、生成输出的自检、子智能体以及已保存的操作流程。Kimi K2.6是从3个开放权重候选模型中选出的，并在2026年日本医师执业资格考试中获得了480/500的成绩。

    arXiv:2610.32151v1 Announce Type: cross  Abstract: Large language model agents that use tools could take on part of the electronic health record and desk work that occupies physicians' time, but in many hospitals the medical information systems run inside closed networks, so the model and the agent harness must also sit inside. We designed, implemented, and deployed LLENS, a clinical support AI agent platform that runs inside the closed network of Hokkaido University Hospital on one host with 8 NVIDIA H200 GPUs, the open-weight model Kimi K2.6 (~1 trillion parameters), and a web application rebuilt from Open WebUI. The harness has 10 components, including page image reading, full-text attachment input, a path-addressed document tree without a vector index, a network-isolated sandbox, artifact return, self-checking of generated output, subagents, and saved procedures. Kimi K2.6 was chosen from 3 open-weight candidates and scored 480/500 on the 2026 Japanese Medical Licensing Examination
    
[^233]: 语言智能体中记忆的价值

    The Value of Memory in Language Agents

    [https://arxiv.org/abs/2610.31303](https://arxiv.org/abs/2610.31303)

    该论文提出一个决策论框架，以所避免的任务损失为单位，量化语言智能体中记忆从原始历史到持久状态、操作使用及代价高昂的恢复与修正全过程中的价值，并通过成本感知的模拟条件和五个最小反例揭示信息时机、检索访问、组件交互、反馈和需求不确定性对记忆价值的关键影响。

    

    记忆使语言智能体能够将过去经验中有用的区分带入未来的决策中。其价值取决于哪些区分得以留存、有界的智能体能否使用这些区分，以及随着需求的演变哪些内容可以被恢复或修正。我们开发了一个决策论框架，追踪这一价值从原始历史到持久状态、操作化使用以及代价高昂的恢复与修正的完整过程。原始价值与保留后的价值以所避免的任务损失为单位来表达；风险-成本前沿刻画了在各种用途下可实现的恢复与保存能力。未知需求会产生价值画像，而交互式反馈则会改变未来工作的分布。一个成本感知的模拟条件与五个最小反例揭示了信息时机、检索访问、组件交互、反馈以及需求不确定性所扮演的角色。特别地，一个与答案具有零互信息的指针仍然可以降低精确恢复的成本。

    arXiv:2610.31303v1 Announce Type: cross  Abstract: Memory enables language agents to carry useful distinctions from past experience into future decisions. Its value depends on which distinctions survive, whether a bounded agent can use them, and what can be recovered or revised as demands evolve. We develop a decision-theoretic framework that follows this value from raw history through persistent state, operational use, and costly recovery and revision. Original and retained value are expressed in units of avoided task loss; risk-cost frontiers describe attainable recovery and preservation across uses. Unknown demand yields value profiles, while interactive feedback changes the distribution of future work. A cost-aware simulation condition and five minimal counterexamples expose the roles of information timing, retrieval access, component interaction, feedback and demand uncertainty. In particular, a pointer with zero mutual information about the answer can still reduce the cost of exa
    
[^234]: 为明天学习该问什么：面向长期个性化网络代理的主动偏好获取

    Learning What to Ask for Tomorrow: Active Preference Acquisition for Long-Term Personalized Web Agents

    [https://arxiv.org/abs/2610.31107](https://arxiv.org/abs/2610.31107)

    提出主动偏好获取（APA）框架，通过前瞻性个性化价值（PPV）衡量缺失偏好在未来任务中的累积效用增益并扣除提问成本与披露风险，使网络代理能在交互预算下智能选择该向用户询问哪些偏好以实现长期个性化。

    

    网络代理越来越多地在重复的在线任务中服务于同一用户，例如预订旅行、购物和日程安排，因此它们的长期有用性取决于其对用户偏好的了解程度。被动记忆和个性化方法无法获取用户从未透露的偏好，而以问题对当前任务的收益来衡量价值的澄清式提问方法，则忽视了在后续任务中重复使用答案所带来的收益。我们研究了主动偏好获取（Active Preference Acquisition, APA）：哪些缺失的偏好值得付出提问的成本？我们将缺失偏好的前瞻性个性化价值（Prospective Personalization Value, PPV）定义为其在未来任务上预期的累积效用增益减去提问的努力成本和信息披露风险成本，并在每次交互的预算约束下选择要提问的问题。我们刻画了尽管单任务收益不足、但累积重复使用仍能使偏好获取变得合理的状态，并通过初始画像诊断评估了这一情形。

    arXiv:2610.31107v1 Announce Type: new  Abstract: Web agents increasingly serve the same user across repeated online tasks such as booking travel, shopping, and scheduling, so their long-term usefulness depends on how well they know the user's preferences. Passive memory and personalization methods cannot obtain preferences the user never reveals, and clarification methods that value a question by its benefit to the current task overlook the benefit of reusing the answer in later tasks. We study Active Preference Acquisition (APA): which missing preferences are worth the cost of asking? We define the Prospective Personalization Value (PPV) of a missing preference as its expected cumulative utility gain over future tasks minus the effort and disclosure-risk cost of asking, and select questions under a per-interaction budget. We characterize when cumulative reuse justifies acquisition despite insufficient single-task benefit, and evaluate this regime with an initial-profile diagnostic. AP
    
[^235]: AI模型的伦理边界有多牢固？一项针对15个模型的工资单编辑测试

    How Firm Are AI Models' Ethical Boundaries? A Pay-Stub Editing Test of 15 Models

    [https://arxiv.org/abs/2610.30751](https://arxiv.org/abs/2610.30751)

    该研究通过让15个AI模型执行相同的工资单收入修改请求发现，措辞的微小变化（如省略贷款用途或称修正打印错误）会使模型执行编辑的比例从14.1%升至66.2%，表明AI模型的伦理拒绝边界对措辞高度敏感。

    

    措辞上的微小变化能否让AI模型从拒绝一个敏感请求转变为执行它？我们用相同的请求测试了15个模型，要求增加工资单上显示的收入。在445次可观察的试验中，当说明贷款用途时，模型发出编辑调用的比例为14.1%；当省略贷款用途时为33.1%；当修改被描述为修正打印错误时则高达66.2%。Astra拒绝了全部十次涉及贷款用途的请求，但在所有十次省略用途的试验中都发出了调用。Fable、Opus和Muse Spark在全部三种措辞下均未发出任何调用；Sonnet则在后两种措辞下发出了一些调用。回复摘录展示了模型拒绝、转移话题或为编辑辩护的不同方式。该文件是一份标注的测试文档，所有工具均为模拟。这些结果探究了伦理边界在单一情境下的表现，并不能证明一个整体的伦理排名，也不能证明现实中发生了有害的成功编辑。

    arXiv:2610.30751v1 Announce Type: cross  Abstract: Can a small change in wording move an AI model from refusing a sensitive request to acting on it? We test 15 models on the same request to increase the income shown on a pay stub. Across 445 observable trials, models issue editing calls in 14.1% of cases when a loan purpose is stated, 33.1% when the purpose is omitted, and 66.2% when the change is described as fixing a printing error. Astra refuses all ten loan-purpose requests but issues calls in all ten purpose-omitted trials. Fable, Opus, and Muse Spark issue none across all three wordings; Sonnet issues some under the latter two. Response excerpts illustrate different ways of refusing, redirecting, or justifying an edit. The file is a marked test document and all tools are simulated. These results probe how boundaries behave in one setting; they do not prove an overall ethics ranking or that a harmful real-world edit succeeded.
    
[^236]: VX-One：面向实时边界分流的有类型、经校准的决策头——关于未知攻击诚实性、级联灰色地带与漂移下校准的实证研究

    VX-One: Typed, Calibrated Decision Heads for Real-Time Perimeter Triage - An empirical study of unknown-attack honesty, cascade gray zones, and calibration under drift

    [https://arxiv.org/abs/2610.30090](https://arxiv.org/abs/2610.30090)

    本文提出VX-One，一个将商用有类型决策原语（校准门控、含显式未知选项的闭集选择、序数分数）作为输出的小型多头网络，在CPU上以毫秒级延迟实现实时边界分流，并实证评估了未知攻击诚实性、级联灰色地带与漂移下的校准性能。

    

    边界分流面临延迟与语义之间的权衡：签名引擎快速且格式安全，但对未见过的攻击无能为力，而生成式模型虽具备语义理解能力却速度缓慢且会产生无法解析的输出。2026年末，一类商用System One决策模型（TypeSafe Jev；Microsoft-Decision-1）开始将有类型的决策原语——二值门控、闭集选择、序数分数——作为产品API出售，但学术界尚无相关研究。我们提出VX-One：一个小型多头网络（参数量10万至1000万），其输出即为这些原语——一个经校准的门控、一个在始终包含显式未知选项的封闭候选集上的选择，以及一个序数分数，采用多任务方式训练，并对每个头进行温度校准（CPU上p50延迟约为0.7-1.7毫秒）。我们在UNSW-NB15、CIC-IDS2017、EMBER2024数据集上将其与经校准的XGBoost、随机森林、MLP、Kitsune进行对比评估，并测量了一个两阶段级联系统（配备52,674条Emerging-Threats规则的Suricata -> VX-One）……

    arXiv:2610.30090v1 Announce Type: cross  Abstract: Perimeter triage faces a latency/semantics trade-off: signature engines are fast and format-safe but blind to unseen attacks, while generative models are semantic but slow and produce unparseable output. In late 2026 a class of commercial System One decision models (TypeSafe Jev; Microsoft-Decision-1) began selling typed decision primitives - binary gates, closed-set choices, ordinal scores - as product APIs, with no academic treatment. We present VX-One: a small multi-head network (100K-10M parameters) whose outputs are these primitives - a calibrated gate, a choice over a closed candidate set that always includes an explicit unknown option, and an ordinal score, trained multi-task, temperature-calibrated per head (~0.7-1.7 ms p50 on CPU). We evaluate it on UNSW-NB15, CIC-IDS2017, EMBER2024 against calibrated XGBoost, Random-Forest, MLP, Kitsune, and measure a two-phase cascade (Suricata with 52,674 Emerging-Threats rules -> VX-One) o
    
[^237]: CulturalRiddles：一个多文化谜语基准测试

    CulturalRiddles: A Multicultural Benchmark of Riddles

    [https://arxiv.org/abs/2610.29968](https://arxiv.org/abs/2610.29968)

    CulturalRiddles是一个涵盖62种文化、完全由人类创作的谜语基准测试，用于评估大语言模型的跨文化理解与推理能力，结果表明即使是前沿模型在某些社区中的准确率也最高仅达22%。

    

    随着大语言模型不断扩大其语言覆盖范围，评估它们对多元文化的理解和表达能力变得越来越重要。然而，许多基准测试仍然以英语为中心，文化覆盖有限，并且更侧重于事实性知识而非其在现实文化语境中的应用。我们提出了CulturalRiddles，这是一个完全由人类通过社区参与式努力创作的原创、植根于文化的谜语基准，涵盖62种文化（45个国家、51种语言）。这些谜语深深植根于各自的文化，需要溯因推理和组合推理能力，以及详细的事实和社会知识、语言技能以及对文化特定关联的理解。我们使用自动评估和大规模人工评估对28个LLM配置进行了基准测试。即使是前沿模型，在某些社区中的准确率也最高仅达到22%。

    arXiv:2610.29968v1 Announce Type: new  Abstract: As LLMs expand their language coverage, evaluating how well they understand and represent diverse cultures becomes increasingly important. However, many benchmarks remain English-centric, with limited cultural coverage and a focus on factual knowledge over its application in real-world cultural contexts. We propose CulturalRiddles, a benchmark of original, culturally grounded riddles created entirely by humans through a participatory community effort, spanning 62 cultures (45 countries, 51 languages). These riddles are deeply rooted in their respective cultures and require abductive and compositional reasoning, alongside detailed factual and social knowledge, language skills, and an understanding of culturally specific associations. We benchmark 28 LLM configurations using both automatic and large-scale human evaluation. Even frontier models achieve at most 22% accuracy in some communities. We publicly release all artifacts alongside an 
    
[^238]: 面向可泛化智能体强化学习的框架扩展

    Harness Scaling for Generalizable Agentic Reinforcement Learning

    [https://arxiv.org/abs/2610.29458](https://arxiv.org/abs/2610.29458)

    提出Harness Scaling这一智能体强化学习的新扩展维度，将训练从单一固定框架扩展到多样化执行配置，从而提升模型在不同框架配置下的泛化与迁移能力。

    

    智能体框架已成为现代智能体系统的核心组成部分，塑造着模型使用工具、管理上下文以及协调长时程交互的方式。要充分发挥这些框架的收益，取决于底层模型利用它们的有效程度，这促使人们直接通过框架进行强化学习。然而，现有方法通常仅在单一固定框架下进行训练，这会导致模型过度专门化，并在不同框架配置之间的迁移能力受限。为解决这一问题，我们提出了Harness Scaling（框架扩展），这是智能体强化学习的一个全新扩展维度，将训练从单一框架扩展到多样化的执行配置。具体而言，我们构建了一个因子化的框架空间，选取具有均衡因子覆盖的紧凑训练子集，并组织跨框架训练，将全局覆盖与同一任务的可控变化相结合。在多个基准测试、模型规模以及（摘要在此处被截断）

    arXiv:2610.29458v1 Announce Type: cross  Abstract: Agent harnesses have become a central component of modern agentic systems, shaping how models use tools, manage context, and coordinate long-horizon interaction. Realizing the benefits of these harnesses depends on how effectively the underlying model can make use of them, motivating reinforcement learning directly through the harness. However, existing approaches typically train through a single fixed harness, which can lead to specialization and limited transfer across harness configurations. To address this, we introduce \textbf{Harness Scaling}, a new scaling dimension for agentic RL that extends training from a single harness to diverse execution configurations. Concretely, we construct a factorized harness space, select a compact training subset with balanced factorial coverage, and organize cross-harness training to combine global coverage with controlled variation for the same task. Across multiple benchmarks, model scales, and
    
[^239]: LTEM：面向表格理解与生成的LLM友好型表格增强Markdown格式

    LTEM: LLM-Friendly Table-Enhanced Markdown for Table Understanding and Generation

    [https://arxiv.org/abs/2610.29188](https://arxiv.org/abs/2610.29188)

    该论文提出LTEM——一种LLM友好的Markdown最小扩展格式，通过显式编码单元格跨度和层级表头来表达复杂表格结构，在保持轻量级token成本的同时，显著提升了大语言模型对真实文档中复杂表格的理解与生成准确率。

    

    现实世界文档中的表格，包括科学论文、财务报告和统计摘要，经常使用合并单元格、多级表头和分区布局。然而，大语言模型最擅长处理的轻量级格式（Markdown、CSV）无法表达这些结构，而表达力强的格式（HTML、LaTeX）则带来过高的token成本和解析复杂性。为了系统分析这种权衡，我们引入了一种分类法，将表格结构分解为六个原子特征（F1-F6）和相应的复杂度级别（L1-L6），并构建了STB（合成表格基准），这是一个包含1,089个表格的受控基准，这些表格系统地分布在各复杂度级别上。利用该框架，我们提出了LTEM（LLM友好的表格增强Markdown），这是一种形式化定义的Markdown最小扩展，显式编码单元格跨度和层级表头。在五个最先进的大语言模型上的实验表明，LTEM实现了准确率的……

    arXiv:2610.29188v1 Announce Type: new  Abstract: Tables in real-world documents, including scientific papers, financial reports, and statistical summaries, frequently use merged cells, multi-level headers, and sectioned layouts. Yet the lightweight formats LLMs handle best (Markdown, CSV) cannot express these structures, while expressive formats (HTML, LaTeX) impose excessive token cost and parsing complexity. To enable systematic analysis of this trade-off, we introduce a taxonomy decomposing table structure into six atomic features (F1-F6) and corresponding complexity levels (L1-L6), and construct STB (Synthetic Table Benchmark), a controlled benchmark with 1,089 tables systematically distributed across complexity levels. Using this framework, we propose LTEM (LLM-friendly Table-Enhanced Markdown), a formally specified, minimal extension of Markdown that explicitly encodes spans and hierarchical headers. Experiments across five state-of-the-art LLMs show that LTEM achieves accuracy c
    
[^240]: 逆向戈德定理：基于基底受限对比学习器的正数据语言识别

    A Reverse-Gold Theorem: Positive-Data Identification by Substrate-Bounded Contrastive Learners

    [https://arxiv.org/abs/2610.28747](https://arxiv.org/abs/2610.28747)

    该论文证明了一个“逆向戈德定理”：借助先天递归对比算子自我生成结构受限的负例扰动，并以InfoNCE式对比学习训练的受限评分器，在忠实性条件下可仅凭正数据实现高概率的PAC条件语言识别，且样本复杂度仅为多项式级。

    

    戈德定理表明，任何学习器都无法仅凭正例呈现来识别超有限类中的任意语言。我们证明了一个形态不同的正面结果。固定一个先天的“递归对比算子”$R$，它将任意字符串映射到其结构上受限的有限扰动集合；再考虑一个容量受限的判别式评分器，它通过在自我生成的扰动上进行InfoNCE式对比估计训练。在算子、评分器与目标语言之间满足忠实性条件的前提下，训练后的评分器能够仅凭正数据实现“PAC条件识别”：以高概率，它会把除$\varepsilon$比例之外的所有正例都排在其自身轨道中每个语言外扰动之上，且样本复杂度关于模型容量、轨道大小、分数界限和$1/\varepsilon$呈多项式级。负例由学习器自行构造，而非由外部提供……

    arXiv:2610.28747v1 Announce Type: cross  Abstract: Gold's theorem shows that no learner can identify an arbitrary language in a superfinite class from positive presentations alone. We prove a positive result of a different shape. Fix an innate \emph{recursive contrast operator} $R$ mapping any string to a finite set of structurally bounded perturbations of it, and a discriminative scorer of bounded capacity trained by InfoNCE-style contrastive estimation \citep{vandenoord2018,macollins2018} on self-generated perturbations. Under a faithfulness condition relating operator, scorer, and target, the trained scorer achieves \emph{PAC conditional identification} from positive data alone: with high probability it ranks all but an $\varepsilon$-fraction of positives above every out-of-language perturbation in their own orbit, with sample complexity polynomial in the capacity, the orbit size, the score bound, and $1/\varepsilon$. The negatives are constructed by the learner, not supplied by an 
    
[^241]: 更多探索，更少验证：高效的反思式提示优化

    Explore More, Validate Less: Efficient Reflective Prompt Optimization

    [https://arxiv.org/abs/2610.27367](https://arxiv.org/abs/2610.27367)

    提出了一种两阶段框架，将反思式候选池的构建与完整验证的分配解耦，使候选池规模和验证预算可以独立设置，从而大幅提升提示优化的效率。

    

    反思式提示优化利用任务反馈来改进指令，但将候选生成与基于完整验证的选择耦合在一起，会使搜索过程代价高昂且只能顺序进行。我们提出了一个两阶段框架，将反思式候选池的构建与验证资源的分配解耦。在构建阶段，我们采样多个父提示，并在并行的反思分支中复用每个父提示的训练响应。每个修订版本在小型训练小批量数据上与其父提示进行筛选比较，使得局部改进的候选无需完整验证即可支持进一步的修订。构建完成后，探索策略在固定的完整验证预算下从冻结的候选池中选择候选，并返回测得验证分数最高的提示。这种解耦使得候选池大小和验证预算可以独立设置。我们评估了随机选择以及基于相似度的学习分数预测方法……

    arXiv:2610.27367v1 Announce Type: new  Abstract: Reflective prompt optimization improves instructions using task feedback, but coupling candidate generation with full-validation-based selection can make search costly and sequential. We propose a two-stage framework that decouples reflective candidate-pool construction from validation allocation. During construction, we sample multiple parents and reuse each parent's training responses across parallel reflection branches. Each revision is screened against its parent on a small training minibatch, allowing locally improved candidates to support further revisions without full validation. After construction, an exploration policy selects candidates from the frozen pool under a fixed full-validation budget, and the prompt with the highest measured validation score is returned. This separation allows candidate-pool size and validation budget to be set independently. We evaluate random selection and learned score prediction with similarity-ba
    
[^242]: LoWAgent：面向Web智能体的低资源多语言基准测试

    LoWAgent: A Low-Resource Multilingual Benchmark for Web Agents

    [https://arxiv.org/abs/2610.27205](https://arxiv.org/abs/2610.27205)

    该论文提出了首个评估Web智能体在低资源语言（印地语、孟加拉语、乌尔都语、阿姆哈拉语）上性能的多语言基准LoWAgent，包含六个本地化网站和跨五种语言的3,000个任务实例，并发现闭源和开源模型在低资源语言上均相比英语出现明显的性能下降。

    

    Web智能体通常在英语界面上进行评估，其跨全球多语言网站运行的能力在很大程度上尚未被探索。这一差距对于低资源语言尤为显著，因为用于评估Web智能体的标准化资源仍然有限。为了填补这一空白，我们提出了低资源Web智能体基准测试，用于评估Web智能体在不同低资源语言（LRL）上相对于默认语言英语的表现。LoWAgent在英语以及四种低资源语言（印地语、孟加拉语、乌尔都语和阿姆哈拉语）上评估Web智能体。该基准包含六个完全本地化的网站，跨五种语言实例化，共产生3,000个任务实例。我们在所有五种语言和所有六个网站上评估了三个闭源模型和五个开源模型。我们的研究结果表明，与英语相比，闭源和开源模型在低资源语言上均出现了性能下降。

    arXiv:2610.27205v1 Announce Type: cross  Abstract: Web agents are typically evaluated on English-language interfaces, leaving their ability to operate across the world's multilingual websites largely unexplored. This gap is especially significant for low-resource languages, where standardized resources for evaluating web agents remain limited. To address this gap, we introduce Low-Resource Web Agent Benchmark (LoWAgent), a benchmark for evaluating web agents across different low-resource languages (LRLs) in comparison to the default, English. LoWAgent evaluates web agents across English and four LRLs: Hindi, Bengali, Urdu, and Amharic. The benchmark comprises of six fully localized websites, instantiated across five languages to yield 3,000 task instances. We evaluate three closed-source and five open-source models across all five languages on all six websites. Our findings show that both closed-source and open-source models experience performance degradation for LRLs compared to Engli
    
[^243]: AfriGuard：面向非洲语言的安全性与稳健拒答机制

    AfriGuard: Safety and Robust Refusal For African Languages

    [https://arxiv.org/abs/2610.26474](https://arxiv.org/abs/2610.26474)

    该论文提出了AfriGuard，一个覆盖10种非洲语言和英语的扎根文化背景的多语言数据集，用于研究有害意图检测、安全类别理解与拒答行为，并揭示了现有护栏模型在这些安全任务上表现不佳、对适当安全行为理解有限的现状。

    

    arXiv:2610.26474v2 公告类型：替换 摘要：旨在与用户交互的大语言模型（LLM）通常被训练以对有害或有毒内容做出适当回应。然而，在多样化的语言和文化语境下，确保适当回应变得更具挑战性，因为安全风险的表达和解读方式可能各不相同。因此，构建安全的AI系统需要检测安全风险、理解其性质并做出适当回应，同时还要兼顾在每个阶段都增加复杂性的语言和文化差异。在这项工作中，我们提出了AfriGuard，一个扎根于文化背景的多语言数据集，用于研究10种非洲语言和英语中的有害意图、安全类别理解以及拒答行为。我们的研究结果表明，护栏（guardrail）模型在安全意图和安全类别分类任务上的表现均不及开源和闭源大语言模型，同时各模型对适当安全行为的理解也十分有限。

    arXiv:2610.26474v2 Announce Type: replace  Abstract: Large language models (LLMs) designed to interact with users are typically trained to respond appropriately to harmful or toxic content. However, ensuring appropriate responses becomes more challenging in diverse linguistic and cultural contexts, where safety risks may be expressed or interpreted differently. Building safe AI systems therefore requires detecting safety risks, understanding their nature, and responding appropriately, while accounting for linguistic and cultural variations that add complexity to each stage. In this work, we introduce AfriGuard, a culturally grounded multilingual dataset for studying harmful intent, safety-category understanding, and refusal behavior across 10 African languages and English. Our findings reveal that guardrail models underperform both open- and closed-source LLMs on safety-intent and safety-category classification, while models show limited understanding of appropriate safety behavior, sc
    
[^244]: 智能体记忆的几何学：预测检索相关的记忆失效，并刻画其局限性

    The Geometry of Agent Memory: Forecasting Retrieval-Bound Memory Failures, and Characterizing Its Limits

    [https://arxiv.org/abs/2610.26433](https://arxiv.org/abs/2610.26433)

    该研究用对话、记忆与查询之间的几何关系特征来预测智能体记忆系统的检索失效，但发现其整体预测能力不及简单的问题难度基线，且在控制问题难度和系统身份后，几何特征并未带来统计学上显著的性能提升。

    

    我们研究了对话轮次、记忆文本和查询之间的几何关系能否在生成答案之前预测答案的正确性。我们评估了四种采用稠密或混合检索的记忆实现，以及一个观察-摘要对比基线。我们将这些文本嵌入到共享空间中。一个逻辑回归分类器使用25个特征来描述查询邻近性、记忆-对话对应关系和对话结构。在LongMemEval-small数据集上，2,367个问题-系统配对的合并AUROC为0.640，低于问题难度基线的0.655。性能因问题类型而异：几何特征在单会话助手类问题上达到0.841，但仅凭系统身份即可达到0.826。在控制问题难度和系统身份因素后，几何特征未带来统计学上成立的合并改进（delta = +0.002，95%置信区间[-0.010, 0.014]）。固定存储的上下文选择实验发现，使用按邻近度排序的记忆能获得比...更高的准确率（原文截断）。

    arXiv:2610.26433v2 Announce Type: replace  Abstract: We study whether geometric relationships between conversation turns, memory texts, and queries can predict answer correctness before generation. We evaluate four memory implementations with dense or hybrid retrieval and one observation-summary comparator. We embed these texts in a shared space. A logistic classifier uses 25 features describing query proximity, memory-conversation correspondence, and conversation structure. On LongMemEval-small, pooled AUROC is 0.640 over 2,367 question-system pairs, below a question-difficulty baseline of 0.655. Performance varies by question type: geometry reaches 0.841 on single-session-assistant questions, but system identity alone reaches 0.826. After controlling for difficulty and identity, geometry adds no statistically established pooled improvement (delta = +0.002, 95% CI [-0.010, 0.014]). A fixed-store context-selection experiment finds higher accuracy with proximity-ranked memories than wit
    
[^245]: 在游戏电脑上实现专家级二语英语语音评估：基于XLS-R、XEUS和PhoneticXeus的准确性、流利度与韵律评价

    Expert L2 English Speech Evaluation on a Gaming PC: Accuracy, Fluency, and Prosody with XLS-R, XEUS, and PhoneticXeus

    [https://arxiv.org/abs/2610.26406](https://arxiv.org/abs/2610.26406)

    该研究提出了一个可在消费级游戏电脑上本地运行的二语英语发音评估系统 oollim，仅使用预训练语音表征和岭回归，就在准确性、流利度和韵律评估上显著超越微软 Azure API，达到专家级水平。

    

    oollim 是一个可在消费级电脑上本地运行的系统，用于评估二语英语学习者的发音准确性、流利度和韵律。该系统将预训练的 XLS-R、XEUS 和 PhoneticXeus 表征与参考文本特征相结合，并使用在 SpeechOcean762、韩国学习者二语英语语音语料库、J-AESOP 及其组合上训练的岭回归模型。在由三位专家评分员独立评分的 305 条留出测试录音上，多个岭回归配置在全部三个评价维度上的 Spearman 相关系数和二次加权 Kappa 均显著优于存档的微软 Azure 发音评估 API 数据。这些结果表明，仅依靠预训练表征和针对特定评价维度的回归模型，即可在消费级硬件上实现专家级语音评估，无需对神经模型进行微调，也无需在私有评估数据上进行校准。

    arXiv:2610.26406v2 Announce Type: replace  Abstract: oollim is a locally executable system for evaluating L2 English pronunciation accuracy, fluency, and prosody on a consumer computer. The system combines pretrained XLS-R, XEUS, and PhoneticXeus representations with reference-text features and ridge regressions trained on SpeechOcean762, the Korean Learners' L2 Speech Corpus (English), J-AESOP, and their combinations. On 305 independently held-out recordings assessed by three expert raters, several ridge configurations substantially outperform archival Microsoft Azure Pronunciation Assessment API data in both Spearman correlation and quadratic weighted kappa across all three constructs. These results support expert speech evaluation on consumer hardware using pretrained representations and construct-specific regressions, without neural-model fine-tuning or calibration on the private evaluation data.
    
[^246]: 预算约束下的截断推理：评估System-1模型作为决策者的表现

    Truncated Reasoning Under a Budget: Evaluating System-1 Models as Decision-Makers

    [https://arxiv.org/abs/2610.25768](https://arxiv.org/abs/2610.25768)

    该论文提出用System-1决策模型Jev来读取截断的推理过程并直接做出决策，相比从不完整推理中强制得出答案，在500至16,000标记的各种预算下均能提升准确率，最多可节省约71.3%的推理成本且准确率损失极小。

    

    大语言模型通常会生成超出所需的推理标记。限制推理预算可以降低成本，但强制从不完整的推理轨迹中得出答案可能会损害准确性。System-1决策模型可以在不生成推理响应的情况下从候选选项中进行选择。我们评估了Jev作为截断推理的决策读取器，在七个LLM推理模型和五个多选题基准（35组配对）上的表现。在测试的500到16,000个标记的预算范围内，Jev的平均准确率高于强制回答，且该结果已计入推理和回答的估计成本。在1,500个标记的预算下，它整体上提高了4.1个百分点的准确率，在截断问题上提高了8.6个百分点，在Holm校正的bootstrap检验下，有17组配对显示出显著提升。离线重放估计仅花费完整运行成本的34.5%，准确率损失为1.13个百分点。在训练折叠上选择的策略在留出折叠上节省了估计完整运行成本的71.3%，准确率损失仅为0.58个百分点。

    arXiv:2610.25768v2 Announce Type: replace  Abstract: Large language models (LLMs) often generate more reasoning tokens than needed. Limiting the reasoning budget reduces cost, but forcing an answer from an incomplete trace can hurt accuracy. System-1 decision models select among candidate options without producing a reasoning response. We evaluate Jev as a decision reader of truncated reasoning across seven LLM reasoners and five multiple-choice benchmarks (35 pairs). Jev achieves higher mean accuracy than forced answering at tested budgets from 500 to 16,000 tokens, including the estimated costs of reasoning and answering. At a 1,500-token budget, it improves accuracy by 4.1 pp overall and 8.6 pp on truncated questions, with significant gains in 17 pairs under Holm-corrected bootstrap tests. Offline replay estimates 34.5% of the full-run cost, with a 1.13 pp accuracy loss. A policy selected on training folds saves 71.3% of the estimated full-run cost on held-out folds, with a 0.58 pp 
    
[^247]: 面向AI来电筛选智能体中小语言模型后训练的质量受控回复与摘要数据

    Quality-Controlled Reply and Summary Data for Post-Training Small Language Models in AI Call-Screening Agents

    [https://arxiv.org/abs/2610.25458](https://arxiv.org/abs/2610.25458)

    该论文提出在信息边界约束下从合成通话构建质量受控的回复与摘要监督数据，用于后训练小语言模型，使其能够在保护隐私的手机端来电筛选智能体中可靠地代机主回复来电并生成通话摘要。

    

    arXiv:2610.25458v2 公告类型：替换 摘要：小语言模型已经发展到这样的程度：手机可以合理地自行筛选来电，在机主不在时代表机主接听电话，而无需将通话内容发送到设备之外，这是对云端来电处理的一种尊重隐私的替代方案。要实现这一愿景，需要模型能够很好地完成两件事：在通话期间在机主权限范围内进行回复，以及事后向机主说明发生了什么以及需要注意什么。目前已发布的小模型在这两方面都不可靠——它们会对普通来电者发出毫无根据的警报，写出的摘要会遗漏重要条件，或将来电者的说法当作事实陈述——而真实通话又无法提供可供学习的参考回复或摘要。我们在信息边界约束下，从合成通话中为这两个任务构建监督数据：来电者撰写者基于私密的情景简报工作，回复撰写者仅能看到截至当前轮次的对话，摘要撰写者则能看到已完成的对话记录（摘要原文在此处截断）。

    arXiv:2610.25458v2 Announce Type: replace  Abstract: Small language models have improved to the point where a phone can plausibly screen its own calls, answering on an absent owner's behalf without sending call content off the device, a privacy-respecting alternative to cloud call handling. Realizing that promise takes a model that can do two things well: reply within the owner's authority during the call, and afterward tell the owner what happened and what needs attention. Released small models do neither reliably - they raise unwarranted alarms about ordinary callers and write summaries that drop material conditions or report a caller's claims as fact - and real calls supply no reference replies or summaries to learn from. We construct supervision for both tasks from synthetic calls under an information-boundary constraint: a caller writer works from a private scenario brief, a reply writer sees only the conversation up to the current turn, and a summary writer sees the completed tra
    
[^248]: 双槽机制：语言模型中从串行循环到并行槽位

    Dual-Slot: From Serial Loops to Parallel Slots in Language Models

    [https://arxiv.org/abs/2610.25086](https://arxiv.org/abs/2610.25086)

    Dual-Slot架构将循环语言模型的串行循环计算重组为并行的潜在槽位与预测槽位，通过把潜在依赖转移到相邻token之间，在保持标准自回归生成与参数量不变的前提下，用一次成对调用替代两次顺序调用，从而显著降低自回归解码延迟。

    

    循环语言模型通过重复应用共享层来增加计算深度，提供了一种在不增加参数的情况下提升模型质量的途径。然而，循环迭代之间的顺序依赖会增加自回归解码的延迟。我们提出了Dual-Slot，这是一种将串行循环计算重组为并行的潜在槽位与预测槽位的架构。预测槽位无需等待当前token的潜在计算完成，而是将前一个token的深度潜在状态与当前表示相结合，同时潜在槽位为下一个解码步骤生成状态。这将潜在依赖转移到相邻token之间，用一次成对调用取代两次顺序的模块调用，同时保持标准的自回归生成方式。为了在训练期间保持token级别的并行性，我们使用少量雅可比迭代来在整个序列上细化潜在状态。

    arXiv:2610.25086v2 Announce Type: replace  Abstract: Looped language models increase computational depth by repeatedly applying shared layers, offering a way to improve model quality without adding parameters. However, sequential dependencies between loop iterations increase autoregressive decoding latency. We introduce Dual-Slot, an architecture that reorganizes serial looped computation into parallel latent and prediction slots. Rather than waiting for the current token's latent computation to finish, the prediction slot combines the previous token's deep latent state with the current representation, while the latent slot produces a state for the next decoding step. This shifts the latent dependency across adjacent tokens, replacing two sequential module invocations with a single paired invocation while preserving standard autoregressive generation. To preserve token-level parallelism during training, we use a small number of Jacobi iterations to refine latent states across the seque
    
[^249]: P2N：重用深层表示以获得更大的有效计算深度

    P2N: Reusing Deep Representations for Greater Effective Depth

    [https://arxiv.org/abs/2610.25076](https://arxiv.org/abs/2610.25076)

    P2N通过将前一个token的深层隐藏状态加到当前token的浅层表示上，以每步仅一次向量加法的极小开销，将模型的有效计算深度跨token递归扩展，从而在不增加层数的前提下显著提升语言模型的推理能力。

    

    数学问题求解和自主智能体是语言模型日益重要的应用，它们需要更强的推理能力。计算深度是这类推理的关键资源，但增加模型层数会同时增加训练和推理成本，而测试时扩展（test-time scaling）则会带来大量额外的推理计算。我们提出了P2N（Previous to Next），该方法重用前一个token在生成过程中已经可用的深层表示，以几乎可忽略的解码开销来增加有效计算深度。具体而言，P2N将前一个token的深层隐藏状态添加到当前token的浅层表示中，使后续层能够建立在之前已完成的深度计算之上。这种递归方式将有效计算深度扩展到跨token范围，同时每个解码步骤仅引入一次向量加法。我们使用并行Jacobi更新来训练这种递归机制……

    arXiv:2610.25076v2 Announce Type: replace  Abstract: Mathematical problem solving and autonomous agents are increasingly important applications of language models, demanding stronger reasoning capabilities. Computational depth is a key resource for such reasoning, but adding model layers increases both training and inference costs, while test-time scaling incurs substantial additional inference computation. We introduce P2N (Previous to Next), which reuses the previous token's deep representations, already available during generation, to increase effective computational depth with negligible decoding overhead. Specifically, P2N adds the previous token's deep hidden state to the current token's shallow representation, allowing subsequent layers to build on previously completed deep computation. This recurrence extends effective computational depth across tokens, while introducing only a single vector addition per decoding step. We train the recurrence using parallel Jacobi updates restr
    
[^250]: 何时更好的接受率才有回报？推测解码的对照研究

    When Does Better Acceptance Pay? A Controlled Study of Speculative Decoding

    [https://arxiv.org/abs/2610.24993](https://arxiv.org/abs/2610.24993)

    该论文建立统一的训练-推理-评估对照协议，发现推测解码起草器之间在已发表结果中的接受率差距在相同条件下会大幅缩小，且接受率可通过近零成本的目标与上下文修改（如DCore强基线）或更昂贵的逐词元条件化来提升，只有当接受率提升的收益超过获取成本时才能带来吞吐量增益。

    

    推测解码的吞吐量提升取决于接受长度与执行延迟之间的权衡。我们探讨何时额外的接受率值得为其付出的成本。我们建立了一套涵盖训练、推理和评估的比较协议，重新测量了已发布的检查点，并在共享的目标模型生成数据上重新训练了六种起草器方案，覆盖十个基准测试和两个目标模型。对齐改变了格局：在已发表比较中显得很大的差距，在共享训练和推理条件下大幅缩小。这种缩小引出了两个更深层次的问题：接受率是以何种成本得到提升的，以及这些提升何时能改善吞吐量？我们发现，接受率可以通过几乎零成本的目标函数与上下文修改来提升，也可以通过更昂贵的逐词元条件化来提升。DCore 展示了前一条路线，它作为一个强大的基线，几乎没有额外的可测量起草开销。

    arXiv:2610.24993v2 Announce Type: replace  Abstract: The throughput gains of speculative decoding depend on the trade-off between acceptance length and execution latency. We ask when additional acceptance is worth the cost of obtaining it. We establish a comparison protocol spanning training, inference, and evaluation, remeasuring released checkpoints and retraining six drafter recipes on shared target-generated data across ten benchmarks and two target models. Alignment changes the picture: gaps that appear large in published comparisons shrink substantially under shared training and inference conditions. This narrowing motivates two deeper questions: how is acceptance improved at different costs, and when do these gains improve throughput? We find that acceptance can be improved through nearly free objective and context changes or more expensive token-to-token conditioning. DCore illustrates the former route as a strong baseline with little additional measured drafting overhead. The 
    
[^251]: 先遮后露（Veil-Then-Reveal）：在零信任的用户与大语言模型交互中通过可逆伪匿名化保护个人身份信息隐私

    Veil-Then-Reveal: Protecting PII Privacy via Reversible Pseudo-Anonymization in Zero-Trust User--LLM Interaction

    [https://arxiv.org/abs/2610.22800](https://arxiv.org/abs/2610.22800)

    提出用户侧可逆伪匿名化框架Veil-then-Reveal（VtR），在零信任威胁模型下将提示词中的PII替换为同类型的真实替身词以保护隐私，同时保留语言与结构特性使LLM仍能有效处理，明文信息始终保留在用户端。

    

    大语言模型（LLM）日益被用于处理敏感文档和个人信息，然而向远程服务提供商发送包含个人身份信息（PII）的提示词，会使数据主体和组织面临日志记录、数据留存、模型训练以及意外泄露等风险。现有的匿名化方法通常会遮蔽或删除PII，而传统的加密方式则会消除LLM有效处理所依赖的词汇和结构线索。我们提出了Veil-then-Reveal（VtR），这是一个在零信任威胁模型下实现隐私保护LLM交互的用户侧PII伪匿名化框架。VtR将远程LLM提供商视为不可信方，仅发送经过伪匿名化处理的提示词，而将明文PII保留在用户端。VtR首先检测PII，并在提示词离开用户设备之前，将其替换为真实的、同类型、存在于词表中的替身词，从而保留提示词的语言与结构特性。它结合……

    arXiv:2610.22800v1 Announce Type: cross  Abstract: Large language models (LLMs) increasingly process sensitive documents and personal information, yet sending prompts containing personally identifiable information (PII) to remote providers exposes data subjects and organizations to risks from logging, retention, model training, and unintended disclosure. Existing anonymization methods often mask or remove PII, while conventional encryption eliminates the lexical and structural cues needed for effective LLM processing. We present Veil-then-Reveal (VtR), a user-side PII pseudo-anonymization framework for privacy-preserving LLM interaction under a zero-trust threat model. VtR treats remote LLM providers as untrusted and sends only pseudo-anonymized prompts, keeping plaintext PII on the user. VtR first detects PII and replaces it with realistic, same-type, in-vocabulary surrogates before prompts leave the user's device, preserving their linguistic and structural properties. It combines for
    
[^252]: 预填充阶段专家削减在何处损害MoE模型：少数提示词Token承担了大部分短答案F1损失

    Where Prefill Expert Reduction Hurts MoE Models: A Few Prompt Tokens Carry Most of the Short-Answer F1 Loss

    [https://arxiv.org/abs/2610.22454](https://arxiv.org/abs/2610.22454)

    研究发现MoE模型在预填充阶段削减路由专家所造成的短答案F1损失，主要集中于少数关键提示词Token（如答案前的聊天模板Token和开头的指令Token），且损害主要在于答案形式而非答案内容，保持这些敏感位置的完整专家远比恢复超级专家或超大激活更有效。

    

    在预填充阶段让提示词Token通过更少的路由专家，是降低混合专家模型服务成本的一种简单方法，但这种做法究竟损害了什么尚不清楚。我们在四个MoE模型家族上，通过Token级、层级和注意力路径干预，在短答案问答任务中定位了这种损害。在专家数量减半的情况下，三个家族的F1损失主要来自“答案形式”：模型仍能找到答案，但不再以简短形式作答。在Qwen3-30B-A3B中，这种损失发生在答案之前的两个聊天模板Token上；在DeepSeek-V2-Lite中，损失发生在其指令所在的前8-32个提示词Token上。按答案似然计算，这些位置分别约占全部损害的一半和五分之一。专家削减对首个Token的注意力质量、范数和最大激活的改变不足4%，而恢复“超级专家”或超大激活最多只能挽回约四分之一的损失，远低于保持敏感位置启用全部专家的效果。

    arXiv:2610.22454v1 Announce Type: new  Abstract: Running prompt tokens through fewer routed experts during prefill is a simple way to make mixture-of-experts (MoE) serving cheaper, yet it is unclear what this damages. We locate the damage with token-, layer- and attention-path interventions on short-answer question answering in four MoE families. At half the experts the F1 loss of three families is mostly answer form: the model still finds the answer but stops answering briefly. In Qwen3-30B-A3B this loss is made at the two chat-template tokens before the answer, and in DeepSeek-V2-Lite at the first 8-32 prompt tokens, where its instruction sits; by answer likelihood these positions account for about half and a fifth of the damage. Reduction changes the first token's attention mass, norm and largest activation by less than 4%, and restoring super experts or massive activations recovers at most about a quarter of the loss, well below keeping the sensitive positions full. On real prompts
    
[^253]: DialFork：超越事实性，评估对话代理中的程序性幻觉

    DialFork: Going Beyond Factuality to Evaluate Procedural Hallucination in Conversational Agents

    [https://arxiv.org/abs/2610.21561](https://arxiv.org/abs/2610.21561)

    DialFork是首个评估对话代理“程序性幻觉”的基准，通过在受控对话轮次位置注入四种幻觉类型构建了涵盖12个领域、24个任务的36,278个对话，并从三个粒度定义了刻画程序性偏差的评估任务。

    

    作为对话代理部署的大语言模型（LLM）被越来越多地期望遵循程序性逻辑。虽然事实性幻觉已受到广泛关注，但另一种失败模式却被忽视了：程序性幻觉，即代理产生的语句虽然单独来看流畅且事实上貌似合理，却偏离了预期的对话流程。我们提出了DialFork，这是首个在原始对话文本上衡量这种失败模式的基准。DialFork通过在受控轮次位置对符合规范的对话进行分叉并注入四种幻觉类型之一来构建，产生了涵盖12个领域、24个任务的36,278个对话；30名人工标注者验证了该构建过程。我们定义了三个评估任务，从三个粒度刻画程序性偏差：一组对话偏离符合规范对话的程度（用于监控已部署的代理）、单个对话偏离的程度（用于程序性检索……

    arXiv:2610.21561v1 Announce Type: cross  Abstract: Large language models (LLMs) deployed as conversational agents are increasingly expected to follow procedural logic. While factual hallucination has received extensive attention, a different failure mode has been overlooked: procedural hallucination, in which an agent produces individually fluent and factually plausible utterances while deviating from the intended conversational process. We introduce DialFork, the first benchmark to measure this failure mode on raw dialogue text. DialFork is built by forking conformant dialogues at controlled turn positions and injecting one of four hallucination types, yielding 36,278 dialogues across 24 tasks in 12 domains; 30 human annotators validate the construction. We define three evaluation tasks that characterize procedural deviation at three granularities: how far a set of dialogues deviates from conformant ones (monitoring deployed agents), how far a single dialogue does (procedural retrieva
    
[^254]: MemComposer：通过强化学习为LLM智能体学习组装记忆

    MemComposer: Learning to Assemble Memories for LLM Agents via Reinforcement Learning

    [https://arxiv.org/abs/2610.21091](https://arxiv.org/abs/2610.21091)

    MemComposer提出一种基于GRPO的强化学习框架ExPO来训练记忆组装器，通过完整智能体交互的任务成功反馈，联合决定记忆的选择、数量与排序，在τ²-bench上超越最强基线的同时显著降低记忆token消耗，且策略可迁移至未见领域。

    

    智能体记忆使大型语言模型（LLM）智能体能够在不更新参数的情况下复用过去的经验。然而，仅凭语义相似性并不能保证记忆的有用性：下游性能取决于记忆如何被选择以及如何呈现给智能体。我们提出了MemComposer，一个记忆组装器，它联合审视当前查询和检索到的候选内容，以决定包含哪些经验、选择多少条以及如何对它们排序。我们使用经验策略优化来训练该组装器，这是一个基于GRPO的框架，通过完整的智能体交互来评估组装后的记忆上下文，并利用任务成功反馈来优化组装策略。在三个τ²-bench领域中使用两个冻结的执行器进行测试，MemComposer超越了每个执行器的最强基线，同时分别将平均记忆token使用量减少了16.9%和4.9%。学习到的策略还能迁移到未见过的领域。

    arXiv:2610.21091v1 Announce Type: new  Abstract: Agent memory enables large language model (LLM) agents to reuse past experiences without updating their parameters. However, semantic similarity alone does not guarantee memory usefulness: downstream performance depends on how memories are selected and presented to the agent. We propose MemComposer, a memory composer that jointly examines the current query and retrieved candidates to decide which experiences to include, how many to select, and how to order them. We train the composer using Experience Policy Optimization (ExPO), a GRPO-based framework that evaluates assembled memory contexts through complete agent interactions and uses task-success feedback to optimize the assembly policy. Across three $\tau^2$-bench domains with two frozen executors, MemComposer outperforms the strongest baseline for each executor while reducing average memory-token usage by 16.9% and 4.9%, respectively. The learned policy also transfers to unseen domain
    
[^255]: 跳过压缩器：编码代理在选择完整文件时能修复更多问题

    Skip the Compressor: Coding Agents Fix More Issues When They Pick Whole Files

    [https://arxiv.org/abs/2610.21073](https://arxiv.org/abs/2610.21073)

    在相同token预算下，与其用压缩器删减文本或检索器挑选代码块，不如将预算用于选择需要原样完整呈现的文件（如CQ2-CARE），这样编码代理能解决更多软件工程问题。

    

    编码代理所处理的代码库可能远超其上下文窗口的容量。提示压缩器和检索器为这一问题提供了解决方案，但修补一个问题需要同时满足两点：必须修改的文件，以及这些文件中原样保留的代码，因为模型必须复现它所编辑的代码行。检索器按照代码与问题的相似度而非是否必须修改来对代码排序，而压缩器则通过删除文本来适应预算限制。我们证明，在相同的token预算下，当预算用于选择哪些文件以原文完整呈现时（正如我们简单的选择器CQ2-CARE所做的那样），代理能解决更多问题，其效果优于将预算用于修剪token或检索代码块。我们在300个最新的SWE-rebench问题上将每个上下文都调整到相同的单问题预算，并使用基准测试的测试用例对每个补丁进行评分。使用GPT-5.6和Claude Opus 5时，CQ2-CARE分别解决了43.7%和46.7%的问题，比三种压缩器中最好的一个高出6.3个百分点。

    arXiv:2610.21073v1 Announce Type: new  Abstract: A coding agent works on repositories that can be far larger than its context window can hold. Prompt compressors and retrievers offer solutions to this problem, but patching an issue needs two things at once: the files that must change, and their code exactly as written, since the model must reproduce the lines it edits. Retrievers rank code by its similarity to the issue rather than by whether it must change, and compressors delete text to fit the budget. We show that at an equal token budget, an agent resolves more issues when the budget goes to choosing which files to show verbatim, as our simple selector CQ2-CARE does, than when it goes to pruning tokens or retrieving chunks. We fit every context to the same per-issue budget on 300 recent SWE-rebench issues and grade each patch with the benchmark's tests. With GPT-5.6 and Claude Opus 5, CQ2-CARE resolves 43.7% and 46.7% of the issues, 6.3 points more than the best of three compressor
    
[^256]: 回译依然重要：利用单语数据实现基于大语言模型的低资源机器翻译

    Back-Translation Still Matters: Leveraging Monolingual Data for Low-Resource Machine Translation with LLMs

    [https://arxiv.org/abs/2610.18818](https://arxiv.org/abs/2610.18818)

    对于低资源机器翻译，利用单语数据做回译比继续预训练大语言模型更有效，仅需1K平行句对即可超越继续预训练7.7至16.4个XCOMET-XXL分，甚至优于拥有10倍平行数据的模型。

    

    预训练大语言模型（LLM）在零样本和少样本设置下对许多任务都很有效。对于低资源语言的机器翻译，显而易见的选择是少样本提示和在小型平行语料上进行微调。我们提出一个问题：如果同时还有适量规模的低资源单语数据可用，情况会怎样？我们认为，比起继续预训练，更有效地利用这些数据的方法是大语言模型时代之前的技术——回译（BT）。具体做法是：让当前微调后的大语言模型对每种语言的单语句子进行翻译，将得到的句对用于反向方向的训练，并与平行语料上的更新交错进行以保证学习的稳定性。在Odia语和斯瓦希里语上仅使用1K平行句对的情况下，回译在域内测试中比向低资源语言继续预训练高出7.7到16.4个XCOMET-XXL分数。使用束搜索解码的中间翻译结果时，它还击败了拥有10倍平行数据量的模型。

    arXiv:2610.18818v2 Announce Type: cross  Abstract: Pre-trained Large Language Model (LLMs) are effective on many tasks in zero-shot and few-shot settings. For machine translation of low-resource languages, the obvious options are few-shot prompting and fine-tuning on a small parallel set. We ask: what if moderate amounts of low-resource monolingual data are also available? We argue that, rather than continued pretraining, a more effective use of this data is the pre-LLM era technique of back-translation (BT). The current fine-tuned LLM translates monolingual sentences in each language, and the resulting pairs train it in the reverse direction. Updates on the parallel set are interleaved for stable learning. With 1K parallel pairs on Odia and Swahili, BT beats continued pretraining into the low-resource language by 7.7 to 16.4 XCOMET-XXL points in domain. With beam-decoded intermediate translations, it also beats a model with 10 time more parallel data. In-context examples retrieved fro
    
[^257]: 无需学习弃答的强效且可控的大语言模型弃答方法

    Strong yet Controllable LLM Abstention without Learning to Abstain

    [https://arxiv.org/abs/2610.16318](https://arxiv.org/abs/2610.16318)

    论文通过分解弃答的风险发现弃答训练的优势主要来自更强的回答能力而非区分能力，据此提出TAPS方法：只需标准后训练加上现成置信度估计器进行事后选择，即可达到与弃答训练相当的性能，同时保持覆盖率可控。

    

    大语言模型（LLM）可以通过对其可能答错的问题进行弃答来减少幻觉。弃答既可以在模型外部实现（选择性回答），即使用置信度阈值来控制回答率（覆盖率），也可以在模型内部学习（弃答训练），后者性能可能更好，但一旦训练完成，覆盖率便被固定。为了比较这两种方法，我们在匹配二者覆盖率的前提下，将风险（错误率）分解为回答能力和区分能力，即模型回答问题的能力以及准确进行弃答的能力。这一分解表明，弃答训练之所以表现更好，主要是因为其回答能力更强：在使用强置信度估计器的情况下，两者的区分能力几乎相当。这启发我们提出了“先训练准确性，再事后选择”（Train for Accuracy, then Post-hoc Select，TAPS）方法，这是一种模块化方法，能够保持覆盖率可控，只需标准的后训练和一个现成的置信度估计器，并且可以受益……（原文摘要到此截断）

    arXiv:2610.16318v2 Announce Type: replace  Abstract: Large language models (LLMs) can reduce hallucination by abstaining from questions they would likely get wrong. Abstention can be implemented outside the model (selective answering) using a confidence threshold that controls the answer rate (coverage), or learned inside it (abstention training), which can perform better but fixes coverage once trained. To compare the two, we match their coverage and decompose risk (error rate) into answering ability and distinguishing ability, i.e., how well the model answers and how accurately it abstains, respectively. The decomposition shows that abstention training performs better mainly by answering better: with strong confidence estimators, the two have nearly equal distinguishing ability. This motivates Train for Accuracy, then Post-hoc Select (TAPS), a modular approach that keeps coverage controllable, needs only standard post-training and an off-the-shelf confidence estimator, and benefits f
    
[^258]: RIVO：面向深度研究智能体的轻量级自适应重路由

    RIVO: Lightweight Adaptive Rerouting for Deep Research Agents

    [https://arxiv.org/abs/2610.15910](https://arxiv.org/abs/2610.15910)

    RIVO提出了一种轻量级即插即用框架，将深度研究智能体的自适应重路由形式化为基于状态的轨迹纠正问题，无需昂贵的模型重训练或密集监督即可纠正过早停止、过度探索及无效轨迹等问题。

    

    深度研究智能体依赖长程推理与迭代式工具调用来解决复杂的信息检索任务。然而，在这类轨迹上进行自适应决策仍然极具挑战性：智能体可能在仍需更多证据时过早停止，在证据已充分的情况下继续不必要地探索，或无法及时纠正无效的研究轨迹。现有方案要么需要代价高昂的模型重训练和密集的轮次级监督，要么依赖并未针对下游研究效用进行直接优化的启发式控制规则。这促使我们寻求一种轻量级的、感知效用的深度研究控制替代方案。我们将长程深度研究中的自适应重路由形式化为一个基于状态的轨迹纠正问题。基于这一视角，我们提出了RIVO——一个用于自适应轨迹纠正的轻量级即插即用框架。智能体从轻量级信号中估计其研究状态，并据此检测（原文在此处截断）

    arXiv:2610.15910v2 Announce Type: replace  Abstract: Deep Research agents rely on long-horizon reasoning and iterative tool use to solve complex information-seeking tasks. However, adaptive decision-making over these trajectories remains challenging, as agents may stop prematurely when further evidence is needed, continue unnecessarily after sufficient evidence has been gathered, or fail to redirect unproductive trajectories. Existing solutions either require costly model retraining and dense turn-level supervision, or rely on heuristic control rules that are not directly optimized for downstream research utility. This motivates a lightweight, utility-aware alternative for Deep Research control. We formulate adaptive rerouting in long-horizon Deep Research as a state-conditioned trajectory correction problem. Based on this view, we introduce RIVO, a lightweight plug-in framework for adaptive trajectory correction. The agent estimates its research state from lightweight signals and dete
    
[^259]: AMR解析器的输出有多可靠？面向AMR解析的不确定性估计

    How Reliable are AMR Parser Outputs? Uncertainty Estimation for AMR Parsing

    [https://arxiv.org/abs/2610.13236](https://arxiv.org/abs/2610.13236)

    该论文首次在黑盒设置下研究AMR解析器的不确定性估计，通过束搜索候选间的相似度来评估解析输出的可靠性，为AMR解析器的评估提供了性能之外的重要补充维度。

    

    近期的进展显著提升了抽象意义表示（AMR）解析器的性能，然而判断其输出何时可以被信任这一问题在很大程度上仍未被探索。本工作首次在黑盒设置下，利用top-k束搜索候选结果对AMR解析器开展不确定性估计研究。不确定性通过成对候选相似度来估计，并使用预测拒绝率进行评估。本工作刻画了解析性能与不确定性之间的关系，评估了束候选分布的可靠性，分析了评估指标对不确定性评估的影响，并识别出在性能与可靠性之间实现最佳平衡的解析器。实验结果表明，不确定性估计作为评估AMR解析器的补充维度具有重要价值。

    arXiv:2610.13236v1 Announce Type: cross  Abstract: Recent advances have significantly improved the performance of Abstract Meaning Representation (AMR) parsers, yet determining when their outputs can be trusted remains largely unexplored. This work presents the first study of uncertainty estimation for AMR parsers in a black-box setting using top-k beam-search candidates. Uncertainty is estimated from pairwise candidate similarities and evaluated using the Prediction Rejection Ratio. This work characterises the relationship between parsing performance and uncertainty, evaluates the reliability of beam candidate distributions, analyses the effect of evaluation metrics on uncertainty assessment, and identifies parsers that provide the strongest balance between performance and reliability. Experimental results show the value of uncertainty estimation as a complementary dimension for evaluating AMR parsers.
    
[^260]: 测试框架修复了什么？一项关于可见性、基线充分性与评估缺陷的预注册研究

    What Does a Harness Repair? A Preregistered Study of Visibility, Baseline Adequacy and Evaluation Defects

    [https://arxiv.org/abs/2610.05533](https://arxiv.org/abs/2610.05533)

    本预注册研究通过系统实验发现，测试框架搜索带来的评测分数提升往往源于解析器可见性问题和评估缺陷等伪影而非真实能力增益，且简单的关闭思考设置在多数情况下足以媲美甚至胜过经过搜索优化的框架配置。

    

    测试框架搜索会在冻结模型周围保留对提示词、token预算或解析器的更改，只要该更改能提升评测分数。这种提升可能来自解析器此前无法读取的答案、较弱的对比基线或评估缺陷。我们预注册了两项关于这些提升来源的研究，使用三个参数量从1.7B到8B的模型以及Qwen3-14B，并报告了全部65个主要终点。在12个模型-基准组合中的7个里，关闭思考模式比受限思考设置获得了更高的准确率，且每次提升主要来自那些在受限思考设置下没有可读答案的问题。对于小模型，关闭思考的设置在9个组合中的7个里并不显著劣于救援配置，对于1.7B模型，也不劣于四个通过GEPA选择的测试框架。在14B规模上，两个GEPA测试框架在LongevityBench字母任务上比它高出30到34分；但在描述性分析中，它们并未显著超过最强的恒定标签基线。

    arXiv:2610.05533v2 Announce Type: replace-cross  Abstract: Harness search keeps a change to the prompts, token budgets or parsers around a frozen model if the change raises a score. Such a gain can come from answers the parser could not read before, a weak comparison or an evaluation defect. We preregistered two studies of where these gains come from, with three models of 1.7B to 8B parameters and with Qwen3-14B, and report all 65 primary endpoints. Turning thinking off raised accuracy over a capped thinking setting in 7 of 12 model-benchmark cells, each time mostly through questions the capped setting left without a readable answer. For the small models, the thinking-off setting was not meaningfully worse than a rescue configuration in 7 of 9 cells, nor, for the 1.7B model, than four GEPA-selected harnesses. At 14B, two GEPA harnesses beat it by 30 to 34 points on the LongevityBench letter tasks; in descriptive analyses they were not detectably above the strongest constant-label basel
    
[^261]: SearchJev：面向搜索代理的校准化System-1模型

    SearchJev: A Calibrated System-1 Model for Search Agents

    [https://arxiv.org/abs/2610.05107](https://arxiv.org/abs/2610.05107)

    SearchJev提出一个无需自回归生成的校准化System-1模型，将搜索代理的短决策与System-2推理分离，并通过软标签学习（SLCD）从不确定监督中提升决策置信度的可靠性。

    

    搜索代理需要反复就相关性、证据充分性和搜索动作做出短决策。使用生成式语言模型进行这些决策会引入延迟和不可靠的置信度。我们提出SearchJev，一个校准化的System-1模型，将搜索决策与System-2推理和生成相分离。给定搜索状态和决策模式，SearchJev无需自回归输出生成即可对合法选项进行评分。我们提出了用于校准决策的软标签学习，从不确定的监督中学习决策概率，并提高决策置信度的可靠性。在双系统搜索代理中，SearchJev处理短决策，并将不确定的判断委托给System 2，后者保留规划、查询生成和答案组合等能力。我们还引入了SearchDecision-Bench，一个统一六种搜索决策类型用于训练和评估的基准。在SearchDecision-Bench上……

    arXiv:2610.05107v2 Announce Type: replace-cross  Abstract: Search agents repeatedly make short decisions about relevance, evidence sufficiency, and search actions. Using generative language models for these decisions introduces latency and unreliable confidence. We present SearchJev, a calibrated System-1 model that separates search decisions from System-2 reasoning and generation. Given a search state and a decision schema, SearchJev scores legal options without autoregressive output generation. We propose Soft-Label Learning for Calibrated Decisions (SLCD) to learn decision probabilities from uncertain supervision and improve the reliability of decision confidence. In a dual-system search agent, SearchJev handles short decisions and delegates uncertain judgments to System 2, which retains planning, query generation, and answer composition. We also introduce SearchDecision-Bench, a benchmark unifying six types of search decisions for training and evaluation. On SearchDecision-Bench, S
    
[^262]: 为长程工作智能体扩展可验证环境

    Scaling Verifiable Environments for Long-horizon Work Agents

    [https://arxiv.org/abs/2610.04906](https://arxiv.org/abs/2610.04906)

    WorkForge 提出了一个可扩展的合成框架，从专家工作流和真实世界资源出发，通过提取可核查的事实锚点，自动构建支持长程交互与可信验证的工作智能体训练环境。

    

    工作智能体通过操作数字制品来执行专业性的知识密集型工作，因此需要既能支持长程交互又能提供可信验证的训练环境。然而，手工构建的环境会带来高昂的工程开销，使环境难以规模化；而现有的合成方法则会牺牲工作区的复杂性、真实性或有据可依的可验证性。为了弥合这一差距，我们提出了 WorkForge，一个可扩展的合成框架，用于从真实世界资源构建可验证的工作智能体环境。WorkForge 从专家工作流出发展开：首先识别每个工作流所需的资源、决策和交付成果；然后检索相关的真实世界文件，并将其组织成一个工作区；接着检查该工作区，提取其中内容的具体、可核查的事实。这些事实锚点确定了工作区能够支持的任务类型及其结果的验证方式……（原文摘要在此处截断）

    arXiv:2610.04906v2 Announce Type: replace-cross  Abstract: Work agents operate over digital artifacts to execute professional knowledge-intensive work, requiring training environments that support long-horizon interaction and trustworthy verification. However, hand-crafted environments incur prohibitive engineering overhead that prevents environment scaling, whereas synthesis methods sacrifice workspace complexity, realism, or grounded verifiability. To bridge this gap, we introduce WorkForge, a scalable synthesis framework for constructing verifiable work-agent environments from real-world resources. Starting from expert workflows, WorkForge first identifies the resources, decisions, and deliverables required by each workflow. It then retrieves relevant real-world files and organizes them into a workspace. WorkForge inspects the workspace to extract concrete, checkable facts about its content. These factual anchors fix which task types the workspace can support and how their outcomes 
    
[^263]: 面向语言模型适配的表示对齐辅助监督

    Representation-Aligned Auxiliary Supervision for Language Model Adaptation

    [https://arxiv.org/abs/2610.04098](https://arxiv.org/abs/2610.04098)

    提出表示对齐辅助监督方法，通过将环境衍生任务以兼容的表示形式作为辅助监督信号，持续提升语言模型向结构化领域（如国际象棋）的适配能力。

    

    语言模型展现出强大的推理能力，然而将它们适配到结构化领域仍然充满挑战，并且可能产生不一致的结果。我们识别出表示兼容性——即模型有效处理结构化任务中某种表示的程度——是适配过程中的关键因素。我们在国际象棋中研究这一问题，国际象棋提供了一个受控的测试平台，具有精确的语义、可计算的最优动作以及多种状态表示，包括符号编码（FEN）和空间格式（ASCII）。我们发现，模型对语义等价输入的处理方式往往存在显著差异，这会影响学习效果和泛化能力。基于这一观察，我们提出了表示对齐的辅助监督方法，该方法利用以兼容表示形式表达的环境衍生任务来改善语言模型对结构化领域的适配。在多种模型和表示上，辅助监督一致地提升了适配效果。

    arXiv:2610.04098v2 Announce Type: replace  Abstract: Language models exhibit strong reasoning capabilities, yet adapting them to structured domains remains challenging and can yield inconsistent outcomes. We identify representation compatibility, the extent to which a model effectively processes a representation for a structured task, as a key factor in adaptation. We study this in chess, which provides a controlled testbed with precise semantics, computable optimal actions, and multiple state representations, including a symbolic encoding (FEN) and a spatial format (ASCII). We find that models often process semantically equivalent inputs substantially differently, affecting both learning and generalization. Building on this observation, we propose representation-aligned auxiliary supervision, which uses environment-derived tasks expressed in compatible representations to improve adaptation to structured domains. Across models and representations, auxiliary supervision consistently imp
    
[^264]: 旧想法，新问题：基于大语言模型的新颖性评估的不稳定性

    Old Ideas, Novel Problems: The Instability of LLM-Based Novelty Evaluation

    [https://arxiv.org/abs/2610.02022](https://arxiv.org/abs/2610.02022)

    本文通过从OpenReview自动挖掘审稿人明确肯定或质疑论文原创性的段落构建评估集，系统性地对照研究了大语言模型新颖性评估器的设计选择，发现微小的提示词改动即可使评估结果在超过一半的相同想法上发生翻转，揭示了此类评估器的严重不稳定性。

    

    基于AI的构思系统（ideation systems）常常以其想法的新颖性作为评估标准，而这种判断正越来越多地交由大语言模型来完成。这类评估器通常是临时搭建的，即使经过验证，也往往是在人类撰写的论文上进行验证，而不是在其本应评分的生成式想法上进行验证。那么，新颖性评估器的实际表现如何？答案并不理想。我们对新颖性评估的设计选择开展了一项系统性的对照研究。我们自动构建了一个评估集：从OpenReview中挖掘审稿人明确肯定或质疑论文原创性的评述段落，仅保留审稿人意见完全一致、且处于其研究领域极端情况下的投稿；并将这些样本与来自标准LLM生成器产生的想法进行配对。在六个评估器上的实验中，我们发现看似微小的提示词设计选择会产生巨大的影响；例如，仅仅告诉评估器“审稿人认为一个想法新颖而另一个不新颖”，就能改变其对超过一半的相同……

    arXiv:2610.02022v2 Announce Type: replace  Abstract: AI-based ideation systems are often evaluated on the novelty of their ideas, and that judgment is increasingly delegated to large language models. Such judges are typically built ad hoc and validated, if at all, on human-authored papers rather than on the generated ideas they are meant to score. So, how do novelty judges perform? Not well. We present a systematic controlled study of novelty evaluation design choices. We build an evaluation set automatically, mining OpenReview for passages where reviewers explicitly affirm or dispute a paper's originality and keeping only submissions with unanimous agreement at the extremes of their research area; we pair these with ideas from a vanilla LLM generator. Across six judges, we find that seemingly small prompt design choices have large consequences; e.g., simply telling the judge that reviewers found one idea novel and the other not can change its verdict on more than half of the identical
    
[^265]: 大语言模型在可视化领域特定语言（DSL）中的失败之处

    Where LLMs Fail with Visualization DSLs

    [https://arxiv.org/abs/2610.01873](https://arxiv.org/abs/2610.01873)

    本研究通过在3个大语言模型上用41个任务评估10种JSON风格的可视化DSL，识别出四种反复出现的失败模式并将其与特定DSL特征关联，为未来面向LLM的DSL设计提供了设计启示。

    

    当大语言模型（LLMs）开始承担使用可视化领域特定语言（DSL）编写图表的任务时，塑造这些语言的人类约束可能不再适用，因为对人类来说容易的事情对模型来说未必容易。为了理解大语言模型如何能更好地与DSL协作，我们探索了它们在当前DSL设计中的失败之处和失败方式。我们使用41个任务在3个大语言模型上评估了10种JSON风格的可视化DSL，然后通过JSON和渲染检查以及失败案例的定性编码来评估生成的规范。通过分析规范生成过程的失败方式，我们识别出四种反复出现的失败模式，将每种模式与特定的DSL特征联系起来，并讨论了未来DSL设计的考量因素。

    arXiv:2610.01873v2 Announce Type: replace-cross  Abstract: As LLMs take up the role of authoring charts using visualization domain-specific languages (DSLs), the human constraints that shaped those languages may no longer apply, as what is easy for a person is not necessarily easy for a model. To understand how LLMs might work better with DSLs, we explore where and how they fail with current DSL designs. We evaluate 10 JSON-style visualization DSLs with 41 tasks across 3 LLMs, then assess the generated specifications with JSON and rendering checks, and qualitative coding of failed cases. Analyzing how this specification generation process fails, we identify four recurring failure patterns, link each to specific DSL features, and discuss design considerations for future DSL designs.
    
[^266]: 面向高效视觉推理的问题特定知识图谱

    Question-Specific Knowledge Graphs for Efficient Visual Reasoning

    [https://arxiv.org/abs/2609.35942](https://arxiv.org/abs/2609.35942)

    提出VisKG强化学习框架，将视觉内容转化为问题特定的知识图谱表示，过滤无关视觉细节并保留实体-关系结构，从而实现更高效的视觉推理。

    

    视觉问答领域的近期研究表明，视觉-语言模型通过将视觉输入转化为文本表示，能够展现出强大的推理能力。这种转化的有效性取决于视觉细节的保留程度；模型需要呈现并对齐足以支持推理的显性与隐性知识，同时不引入虚假的假设。现有利用详细图像描述的方法会引入与推理任务无关的视觉细节，导致输入token数量膨胀并增加计算成本。为应对这些挑战，我们提出了VisKG，一个强化学习（RL）框架，模型在其中学习将视觉内容转化为针对特定问题的知识图谱（KG）表示。这一过程过滤掉感知噪声，同时保留思维链推理所需的实体-关系结构，遵循最小充分性原则。

    arXiv:2609.35942v1 Announce Type: new  Abstract: Recent work in visual question answering has shown that vision-language models can exhibit strong reasoning capabilities by translating visual inputs into textual representations. The effectiveness of this translation depends on how well visual details are retained; models need to surface and align both explicit and implicit knowledge sufficient to support reasoning, without introducing spurious assumptions. Existing methods that leverage detailed image captions introduce visual details unrelated to the reasoning task, inflating input token counts and increasing computational cost. To address these challenges, we propose VisKG, a reinforcement learning (RL) framework in which models learn to translate visual content into question-specific knowledge graph (KG) representations. This process filters out perceptual noise while preserving the entity-relation structure needed for chain-of-thought reasoning, following the principle of minimum s
    
[^267]: 仅靠后期注意力层即可复制实体词元，但离不开对其上下文的关注

    Late Attention Layers Alone Can Copy Entity Tokens, but Not Without Attending to Their Context

    [https://arxiv.org/abs/2609.35663](https://arxiv.org/abs/2609.35663)

    研究发现大语言模型后半部分的两类注意力层对实体复制任务既必要又充分，但这种复制能力必须依赖对上下文词元的关注。

    

    大语言模型（LLM）能够可靠地执行实体复制任务，即模型将从提示词中指代实体的词元（称为实体词元）复制到输出中以回答问题。尽管实体复制对大多数 LLM 而言轻而易举，但现有研究并未系统性地阐明哪些层专门负责这一基础任务，以及同一序列中的其他词元（称为上下文词元）如何影响模型复制实体词元的能力。为解决这些问题，我们在 Qwen3-8B 上采用两种新颖的方法进行实验：其一是“瓶中精灵”（genie-in-a-bottle）方法，可精确控制哪些层参与实体复制任务；其二是“注意力切除”（attention lobotomy）方法，可切断特定词元对实体词元的注意力，同时不影响其余注意力分布。我们发现，模型后半部分中两组不同的注意力层对于实体复制既是必要的也是充分的。此外，……

    arXiv:2609.35663v2 Announce Type: replace  Abstract: Large language models (LLMs) reliably perform entity copying, in which a model copies tokens referring to an entity, termed entity tokens, from the prompt into its output to answer a question. Although entity copying is straightforward for most LLMs, existing research does not provide a systematic account of which layers specialize in this fundamental task or how other tokens in the same sequence, termed context tokens, influence the model's ability to copy the entity tokens. To address these questions, we conduct experiments on Qwen3-8B using two novel methods: genie-in-a-bottle, which controls exactly which layers can participate in an entity-copying task, and attention lobotomy, which cuts off specific tokens' attention to entity tokens without affecting the remaining attention distribution. We find that two distinct groups of layers in the second half of the model are both necessary and sufficient for entity copying. Moreover, in
    
[^268]: 5W1H+Which：基于渐进式本体绑定的上下文有效语义索引

    5W1H+Which: Context-Valid Semantic Indexing with Progressive Ontology Binding

    [https://arxiv.org/abs/2609.35184](https://arxiv.org/abs/2609.35184)

    论文提出5W1H+Which设计，通过将基于源的内容提取与渐进式本体绑定相分离，实现了既保留可复用信息、又为推理提供可检验前提的上下文有效语义索引。

    

    语义索引必须在不过早绑定到某一业务模式的前提下保留可复用的源信息，同时为推理提供可检验的前提。我们提出5W1H+Which，这是一种将基于源的内容提取与渐进式本体绑定相分离的设计。5W1H问题用于组织可独立寻址的内容单元；Which则记录与版本化本体元素的链接、映射关系、适用范围和验证状态。时间、位置、系统环境和参与者角色约束了事实、绑定和规则的使用时机。未绑定的内容仍然保持可搜索性，而形式化推理则需要显式的前提检查。该设计区分了业务有效时间、系统知识时间和操作轨迹，并跟踪依赖关系以支持重新验证和增量维护。工作实例展示了云迁移和时间性策略有效性等应用场景。所得到的链接语料库可以支持……

    arXiv:2609.35184v2 Announce Type: replace  Abstract: Semantic indexing must preserve reusable source information without committing too early to one business schema, while providing checkable premises for reasoning. We propose 5W1H+Which, a design that separates source-grounded content extraction from progressive ontology binding. The 5W1H questions organize independently addressable content units; Which records links to versioned ontology elements, mapping relations, applicability scopes, and validation states. Time, location, system environment, and participant roles constrain when facts, bindings, and rules may be used. Unbound content remains searchable, whereas formal reasoning requires explicit premise checks. The design distinguishes business valid time, system knowledge time, and operational traces, and tracks dependencies for revalidation and incremental maintenance. Worked examples illustrate cloud migration and temporal policy validity. The resulting linked corpus can suppor
    
[^269]: 大语言模型并非“随机鹦鹉”：来自类人造语言任务的意义中介抽象证据

    LLMs are not stochastic parrots: Evidence for meaning-mediated abstraction from conlang-like tasks

    [https://arxiv.org/abs/2609.34187](https://arxiv.org/abs/2609.34187)

    本研究通过类人造语言任务证明，大语言模型在没有任何示例输出的情况下，仅凭自然语言描述就能处理统计上罕见、且违背训练数据表层模式的虚构语言规则，表明其具备超越统计模式匹配的、基于意义中介的抽象能力，从而反驳了“随机鹦鹉”论断。

    

    “随机鹦鹉”论证的强版本声称，尽管大语言模型（LLM）可能超越机械复读，但它们无法从统计模式匹配跃升到抽象或推理，在本体论上仍接近模式重用的下限，尽管其生成的文本流畅诱人。我们使用类人造语言任务来检验这一假设。多个大语言模型仅获得关于虚构语言的自然语言描述，这些虚构语言通过结合统计上罕见且未经证实的特征，颠覆了训练数据中突出的表层模式。关键的是，实验中未提供任何示例输出。我们认为，如果模型表现出遵循规则的行为，它们就不可能仅仅依赖表层的统计模式；因为这类模式往往与正确的输出相悖。相反，成功的表现要求模型具备对提示中所指定约束的表征。在三个互补的任务族中，……

    arXiv:2609.34187v2 Announce Type: replace-cross  Abstract: The strong version of the stochastic parrot argument claims that, although large language models (LLMs) may exceed rote regurgitation, they cannot move beyond statistical pattern matching into abstraction or reasoning, remaining ontologically near the lower bound of pattern reuse despite producing alluringly fluent text. We test this hypothesis using conlang-like tasks. Several LLMs are given only natural-language descriptions of fictional languages that subvert prominent superficial patterns in training data by combining statistically uncommon and unattested features. Crucially, no example outputs are given. We argue that if the models exhibit rule-following behaviour, they cannot be relying solely on superficial statistical patterns; such patterns often work against the correct output. Instead, successful performance requires representations of the constraints specified in the prompt. Across three complementary task families,
    
[^270]: JevAdvBench：面向校准决策强化学习模型的基准与黑盒攻击

    JevAdvBench: A Benchmark and Black-Box Attacks for Reinforcement Learning for Calibrated Decisions Models

    [https://arxiv.org/abs/2609.31142](https://arxiv.org/abs/2609.31142)

    该论文提出了首个针对校准决策强化学习（RLCD）模型的对抗攻击基准JevAdvBench，创新之处在于以模型自身的干净决策（而非外部标签）作为评分参照来度量攻击效果，并配套提出了黑盒攻击方法。

    

    使用校准决策强化学习（RLCD）训练的模型（如Jev），针对输入（即状态）回答一个类型化问题，输出概率、选择或分数，软件在无人阅读的情况下直接对该答案进行处理。这类模型的鲁棒性尚未得到测量：现有的对抗基准评估的是模型生成或执行的内容，而类型化模型不生成任何内容，即使被操纵也会返回格式良好的答案。测量本身也很困难，因为相同的请求可能返回不同的答案，大多数可用标签来自模型自身，且API会在用户不可见的情况下对每个请求进行预处理。我们的核心思想是将每个受攻击的决策与模型自身的干净决策进行对比评分，而非与标签对比，并结合相同请求重复运行所引起的变化来解读评分结果。基于此，我们提出了JevAdvBench，据我们所知这是首个针对RLCD模型的对抗基准，包含覆盖6个……的812个类型化问题（摘要在此处被截断）。

    arXiv:2609.31142v1 Announce Type: cross  Abstract: Models trained with reinforcement learning for calibrated decisions (RLCD), such as Jev, answer a typed question about an input, the state, with a probability, a choice, or a score, and software acts on the answer without a person reading it. Their robustness has not been measured: adversarial benchmarks score what a model generates or executes, whereas a typed model generates nothing and returns a well-formed answer even when manipulated. Measurement is also hard, because identical requests can return different answers, most available labels come from the model itself, and the API preprocesses each request out of view. Our key idea is to score each attacked decision against the model's own clean decision rather than against labels, and to read it against the change caused by an identical re-run. Building on this, we introduce JevAdvBench, to our knowledge the first adversarial benchmark for RLCD models, with 812 typed questions over 6
    
[^271]: 来自主动式语音代理蜜罐的真实诈骗与骚扰电话对话语料库

    A Corpus of Real Scam- and Spam-Call Conversations from an Active Voice-Agent Honeypot

    [https://arxiv.org/abs/2609.29528](https://arxiv.org/abs/2609.29528)

    本文通过主动式语音代理蜜罐在53天内收集了10,015通真实诈骗与骚扰电话对话（约895小时音频、328,869条转录轮次），为电话诈骗研究提供了极为稀缺的真实对话数据集。

    

    诈骗者与其目标之间的真实对话是研究电话诈骗最有价值的信息载体之一，但也最为稀缺：被动式蜜罐捕获的绝大多数是自动语音消息和挂断电话，大规模研究通常仅刻画电话元数据而非对话内容，而人工诱骗方式又难以规模化。我们展示了一个由主动式语音代理蜜罐收集的真实诈骗电话对话数据集。专用电话号码被投放到诈骗团伙获取线索的渠道中；来电由一个低延迟对话代理接听，该代理扮演可信的目标人设并维持互动，同时每通电话都被录音、转录并自动标注。在最初的53天窗口期内，我们捕获了10,015通入站诈骗和骚扰电话（其中6,601通包含两轮或以上对话）：约895小时的音频，以及来自5,665个不同主叫号码的328,869条转录对话轮次。

    arXiv:2609.29528v1 Announce Type: cross  Abstract: Real conversations between fraudsters and their targets are among the most informative artifacts for studying telephone scams, yet also the scarcest: passive honeypots overwhelmingly capture automated messages and hang-ups, large-scale studies characterize call metadata rather than dialogue, and manual scam-baiting does not scale. We present a dataset of real scam-call conversations collected by an active voice-agent honeypot. Dedicated numbers are seeded into the lead-generation channels fraud operations harvest; inbound callers are answered by a low-latency conversational agent that adopts a plausible target persona and sustains the interaction while every call is recorded, transcribed, and automatically labeled. Over an initial 53-day window we captured 10,015 inbound scam and spam calls (6,601 with two or more turns): roughly 895 hours of audio and 328,869 transcribed turns from 5,665 distinct originating numbers. Under a holistic 
    
[^272]: 用于科学决策的Jev：评估语义选择及其后果

    Jev for Scientific Decisions: Evaluating Semantic Choices and Their Consequences

    [https://arxiv.org/abs/2609.24965](https://arxiv.org/abs/2609.24965)

    该研究将Jev作为科学工作流中的语义决策组件进行评估，发现其语义正确性与其他配置持平且延迟最低，并表明错误的语义选择会改变下游计数但可能不影响最终结论标签。

    

    arXiv:2609.24965v1 公告类型：新论文 摘要：科学工作流程通常需要在确定性计算进行之前，在已知关系之间做出选择。观测数据是否共享相同的文化、处理方式或参考标准，可能会改变由此产生的计数或比较的科学含义。我们使用一个遵循其文档指导并将算术运算分配给代码的测试框架，将Jev作为语义决策组件进行评估。该研究在十个科学案例的二十个有来源依据的选择上比较了十二种模型配置，每种配置重复五次。我们分别测量语义选择、下游输出和最终结论标签。Jev与其他五种配置在完全语义正确性上持平，并在成功响应中实现了观察到的最低中位延迟。在三个对比模型中，对一个文化历史问题的七次错误选择改变了下游计数，同时保持了正确的最终标签。这些结果确定了一个有用的角色

    arXiv:2609.24965v1 Announce Type: new  Abstract: Scientific workflows often require choosing among known relations before a deterministic calculation can proceed. Whether observations share a culture, treatment or reference standard can change the scientific meaning of the resulting count or comparison. We evaluate Jev as a semantic decision component using a harness that follows its documented guidance and assigns arithmetic to code. The study compares twelve model configurations on twenty source-grounded Choices across ten scientific cases, each repeated five times. We measure semantic selections, downstream outputs and final claim labels separately. Jev matched five other configurations at complete semantic correctness and achieved the lowest observed median latency among successful responses. Across three comparison models, seven wrong selections on one culture-history question changed downstream counts while preserving the correct final label. These results identify a useful role 
    
[^273]: 基于CallScreenBench的开放Jev判断：使用小型语言模型实现校准的单次前向传播诈骗筛查

    Open-Jev Judgments on CallScreenBench: Calibrated One-Pass Scam Screening with a Small Language Model

    [https://arxiv.org/abs/2609.23959](https://arxiv.org/abs/2609.23959)

    该论文提出JevLite方法，通过对Qwen3-4B进行LoRA微调并从标签logit的softmax中直接读取校准的诈骗概率，在CallScreenBench上实现了单次前向传播的诈骗电话筛查，性能不劣于大型LLM裁判，速度提升4.9倍且对合法来电零误报。

    

    筛查电话诈骗需要在来电者每一轮对话后以毫秒级速度给出可信的概率。Jev式的类型化决策正是为此设计的：输入预先声明的选项，通过单次前向传播为每个选项输出一个经过校准的概率，无需生成任何文本。我们在诈骗电话筛查任务上测试了这一读取方式的开放实现JevLite：对Qwen3-4B进行LoRA微调，使两个答案标签logit上经温度缩放的softmax输出即为P(诈骗)。在41个留出的CallScreenBench场景（共577个逐轮决策）上，三种子集成模型达到AUROC 0.974、校准误差0.052，在预先注册的0.02容差下不劣于LLM裁判（MiniMax-M3），对合法来电零误报，在相同挂断规则下提前1.14轮做出决策，且在单块消费级GPU上每次决策仅需64.5毫秒，比微调同一骨干模型让其生成答案的方式快4.9倍。其增益来自读取方式与校准，而非准确率。

    arXiv:2609.23959v1 Announce Type: new  Abstract: Screening a phone call for fraud needs a trustworthy probability after every caller turn, in milliseconds. Jev-style typed decisions promise exactly that: declared options go in, one calibrated probability per option comes out of a single forward pass, with no generated text. We test an open implementation of this readout, JevLite, on scam-call screening: Qwen3-4B is LoRA-tuned so that the temperature-scaled softmax over two answer-label logits is P(scam). On 41 held-out CallScreenBench scenarios (577 per-turn decisions) a three-seed ensemble reaches AUROC .974 with calibration error .052, non-inferior to an LLM judge (MiniMax-M3) at a pre-registered .02 margin, with no false alarms on legitimate calls, decisions 1.14 turns earlier under the same hang-up rule, and 64.5 ms per decision on one consumer GPU, 4.9x lower than the same backbone fine-tuned to generate its answer. The gain is in the readout and calibration, not accuracy: a fine-
    
[^274]: 前缀共享是一个排序问题

    Prefix Sharing Is a Sorting Problem

    [https://arxiv.org/abs/2609.13692](https://arxiv.org/abs/2609.13692)

    本文证明LLM服务中提示片段的最优排序问题等价于在请求集合上选择一个二叉层次结构，全局固定排序仅在最多两个片段时最优，并据此给出 O(3^m) 精确算法及与最小顶点覆盖等经典问题的联系。

    

    LLM服务通过精确前缀匹配来复用KV缓存，因此当一个提示词由一组可复用的片段（如检索到的段落、工具定义、少样本示例）组装而成时，这些片段所选的排列顺序决定了能够共享多少计算量。目前每个已部署的系统都通过单一的全局约定来固定这一顺序。我们证明这种做法仅在请求最多包含两个片段时是最优的，而在一般情况下是渐近错误的。我们的主要结果是一个结构定理：最小前缀树代价等于在请求的所有二叉层次结构H上求 min_H sum_x w(x) t_x(H)，其中 t_x(H) 是需要片段x的请求集合的规范分解大小。因此，选择片段顺序等价于在请求上选择一个层次结构。这一恒等式给出了一个 O(3^m) 的精确算法，将双片段情形识别为最小顶点覆盖问题，并证明在留一族上的最优解等于最小外部路径长度。

    arXiv:2609.13692v1 Announce Type: cross  Abstract: LLM serving reuses KV cache by exact prefix match, so when a prompt is assembled from a set of reusable pieces -- retrieved passages, tool definitions, few-shot exemplars -- the order chosen for those pieces determines how much computation can be shared. Every deployed system fixes that order by a single global convention. We prove this is optimal only when requests contain at most two pieces, and asymptotically wrong in general. Our main result is a structure theorem: the minimum prefix-trie cost equals min_H sum_x w(x) t_x(H) over binary hierarchies H on the requests, where t_x(H) is the canonical decomposition size of the set of requests needing chunk x. Choosing chunk orders is therefore equivalent to choosing one hierarchy over requests. The identity yields an O(3^m) exact algorithm, identifies the two-chunk case as minimum vertex cover, and shows that on the leave-one-out family the optimum is the minimum external path length of 
    
[^275]: FinLifeBench：从纵向银行对话中穷尽式重建人生事件历史与财务状态

    FinLifeBench: Exhaustive Life-Event History and Financial-State Reconstruction from Longitudinal Banking Dialogue

    [https://arxiv.org/abs/2609.01198](https://arxiv.org/abs/2609.01198)

    提出FinLifeBench基准，基于6,000个韩语银行对话会话，评估大语言模型在穷尽式重建客户人生事件历史与34维财务状态方面的长程记忆能力，发现随会话累积事件召回率显著下降（0.591降至0.445），且错误主要源于事件遗漏。

    

    重复的银行交互要求助手在生活变化随日常请求偶然出现时，维护完整、最新且可追溯的客户记录。现有基准强调问答、有界回合或定向回忆，而非穷尽式的纵向重建。我们提出FinLifeBench，它在同一累积对话上评估两项任务：重建每个人生事件实例及其首次确立的会话，以及在连续检查点上重建完整的34条路径财务状态。该基准包含来自20条独立合成轨迹的6,000个八轮韩语银行会话，为24种事件类型和34条状态路径提供确定性的、穷尽式的黄金标准及共识质量保证。在全上下文条件下对十一个大语言模型的评估中，事件锚点召回率从15个会话时的0.591下降至300个会话时的0.445。错误主要由遗漏事件导致，而非（摘要原文在此处截断）

    arXiv:2609.01198v1 Announce Type: cross  Abstract: Repeated banking interactions require assistants to maintain complete, current, and traceable customer records as life changes emerge incidentally in routine requests. Existing benchmarks emphasize question answering, bounded episodes, or targeted recall rather than exhaustive longitudinal reconstruction. We introduce FinLifeBench, which evaluates two tasks over the same cumulative dialogue: reconstructing every life-event instance with its first-establishing session and reconstructing a complete 34-path financial state at consecutive checkpoints. The benchmark contains 6,000 eight-turn Korean banking sessions from 20 independent synthetic trajectories, with deterministic, exhaustive gold for 24 event types and 34 state paths and consensus quality assurance. Across eleven LLMs under a full-context condition, event-anchor recall falls from 0.591 at 15 sessions to 0.445 at 300. Errors are driven primarily by omitted events rather than po
    
[^276]: 大语言模型中的识别-拒答错位：为什么模型会回答结构上无法回答的问题

    Recognition-Refusal Misalignment in LLMs: Why Models Answer Structurally Unanswerable Questions

    [https://arxiv.org/abs/2608.29109](https://arxiv.org/abs/2608.29109)

    大语言模型在内部表征中已经能够识别问题的不可回答性，但该识别方向与安全拒答方向近乎正交，导致模型无法将“知道问题不可能”转化为实际的拒答行为。

    

    大语言模型经常回答结构上无法回答的问题，例如计算 cot(-540°) 或评估 (1).startswith("1")，而不是选择拒答。我们探究这种失败究竟是源于识别能力的缺失，还是从识别到拒答的路由失败。在参数规模从 1.7B 到 70B 的指令微调模型中，隐状态中的一个单一线性方向可以区分可回答的与结构上不可能的数学和代码提示，这表明模型在生成之前就已经表征了问题的不可能性。然而，这一识别方向与介导有害内容安全拒答的规范安全拒答方向几乎正交。一个基于领域内行为定义的无效性感知方向与识别方向更为接近，但仅部分对齐，并且仍然与安全拒答方向接近正交。在生成时沿识别方向进行引导（steering）能够以双向且剂量响应的方式改变模型的无效性感知行为。

    arXiv:2608.29109v1 Announce Type: new  Abstract: Large language models often answer structurally unanswerable questions, such as computing cot(-540{\deg}) or evaluating (1).startswith("1"), instead of abstaining. We ask whether this failure reflects missing recognition or failed routing from recognition to abstention. Across instruction-tuned models from 1.7B to 70B parameters, a single linear direction in the hidden state separates answerable from structurally impossible math and code prompts, showing that models represent impossibility before generation. Yet this recognition direction is nearly orthogonal to the canonical safety-refusal direction that mediates trained harmful-content refusal. An in-domain behavior-defined invalidity-aware direction is closer to recognition, but only partially aligned with it, and remains near-orthogonal to safety refusal. Generation-time steering along the recognition direction changes invalidity-aware behavior bidirectionally and dose-responsively o
    
[^277]: 在截断评分量表上的双重差分法可能制造虚假效应：来自预注册大语言模型裁判审计的证据

    Difference-in-Differences on a Censored Rating Scale Can Manufacture an Effect: Evidence from a Pre-Registered LLM-Judge Audit

    [https://arxiv.org/abs/2608.27309](https://arxiv.org/abs/2608.27309)

    本文揭示双重差分法在截断评分量表上因截断不均会制造虚假交互效应，并通过预注册审计实证证明该偏差可导致无效结果。

    

    arXiv:2608.27309v1 公告类型：交叉 摘要：对大语言模型裁判的审计通过对比匹配条件来验证偏差，而最严谨的设计会进行两次差分：在两项候选回答之间进行项目内对比，再在操控的属性上进行二次差分，最终从有界评分量表上读取结果。我们证明，这一终点在该量表上无法被识别。双重差分的每一项都受到各自截断比例的影响，因此观察到的统计量混淆了差异偏好与差异衰减：当两项回答以不平等的方式截断时，一个共同作用于两者的严重性偏移会制造出交互效应，因为距界限的不等距离使它们恰好落在良好刺激所在的位置。我们在一个冻结教学裁判的预注册审计中展示了这一失败，该审计在首次990次调用之前就已密封。注册的主要终点，即陈述的学习者画像对裁判脚手架偏好的影响，为零：+0.085分（95% BCa置信区间）。

    arXiv:2608.27309v1 Announce Type: cross  Abstract: Audits of LLM judges certify a bias by contrasting matched conditions, and the strongest designs difference twice: a within-item contrast between two candidate responses, differenced again across a manipulated attribute, read off a bounded rating scale. We show that this endpoint is not identified on the scale that reports it. Each term of the double difference is censored by its own share, so the observed statistic confounds differential preference with differential attenuation: a severity shift common to both responses manufactures an interaction whenever the two censor it unequally, as unequal distances from the bounds make them, exactly where good stimuli place them. We exhibit the failure inside a pre-registered audit of a frozen pedagogy judge, sealed before the first of its 990 calls. The registered primary endpoint, the effect of a stated learner profile on the judge's scaffolding preference, is null: $+0.085$ points (95\% BCa 
    
[^278]: 面向LLM数据代理的痕迹完整性：现实世界系统中可审计结构化推理的愿景

    Trace Integrity for LLM Data Agents: A Vision for Auditable Structured Reasoning in Real-World Systems

    [https://arxiv.org/abs/2608.26036](https://arxiv.org/abs/2608.26036)

    本文提出“痕迹完整性”作为评估LLM数据代理可靠性的新标准，通过执行合同和CAIT率解决结构化数据任务中答案正确但计算无效的问题。

    

    摘要：arXiv:2608.26036v1 公告类型：交叉 摘要：答案准确性对于LLM数据代理而言并不是一个充分的可靠性信号。在结构化数据任务中，一个基准测试正确的答案可能由无效的痕迹产生。本文引入了“痕迹完整性”这一部署可靠性标准，用于评估答案背后记录的计算是否明确、可执行、符合模式、忠实于操作符、可重放、与答案一致且可审计。我们识别出“结构差距”作为部署失败模式，这使得痕迹完整性变得必要：自然语言推理和自由形式的理由并不能可靠地指定现实世界系统所需的操作符级程序。我们通过执行合同将痕迹完整性付诸实践，这些结构化工件将用户意图与模式元素、操作符计划、假设、可执行查询、验证状态和最终答案链接绑定在一起。我们还引入了CAIT（正确答案/无效痕迹）率，用于衡量这种情况发生的频率。

    arXiv:2608.26036v1 Announce Type: cross  Abstract: Answer accuracy is an insufficient reliability signal for LLM data agents. In structured-data tasks, a benchmark-correct answer can be produced by an invalid trace. This paper introduces Trace Integrity, a deployment reliability criterion for evaluating whether the computation recorded behind an answer is explicit, executable, schema-valid, operator-faithful, replayable, answer-consistent, and auditable. We identify the Structure Gap as the deployment failure mode that makes Trace Integrity necessary: natural-language reasoning and free-form rationales do not reliably specify the operator-level programs required by real-world systems. We operationalize Trace Integrity with execution contracts, structured artifacts that bind user intent to schema elements, operator plans, assumptions, executable queries, verification status, and final-answer linkage. We also introduce CAIT (Correct Answer / Invalid Trace) Rate, which measures how often 
    
[^279]: 诈骗电话剖析：10,000个真实诈骗和垃圾电话揭示电话诈骗者的运作方式

    Anatomy of a Scam Call: What 10,000 real scam and spam calls reveal about how phone scammers operate

    [https://arxiv.org/abs/2608.24127](https://arxiv.org/abs/2608.24127)

    本研究通过分析10,000多个真实诈骗和垃圾电话，揭示了电话诈骗者通常在工作时间运营、使用有限脚本，并优先索取身份信息而非支付凭证的运作模式。

    

    电话欺诈普遍存在且代价高昂，但其内部运作方式很少在大规模范围内被观察到。我们分析了一个包含10,211个入站诈骗和垃圾电话的完整语料库——包括913小时的音频和来自5,780个不同号码的330,956个转录轮次——这些数据是通过一个AI语音代理蜜罐在54天内收集的，该蜜罐接听来电者并让他们持续交谈，并在配套数据描述中进行了介绍。我们将直接索取敏感信息的绝对诈骗，与更大规模的掠夺性但合法的线索生成（“垃圾电话”）区分开来，后者为前者提供养分。诈骗活动保持办公时间（工作日的电话量是周末的6.6倍）；数千个一次性号码运行着少量循环使用的脚本（三十个开场群组，前五名占据了一半流量）；来电者索取身份锚点——家庭住址和出生日期——的频率远高于支付凭证，通过坚持和制造权威而非公开强迫来施压。

    arXiv:2608.24127v1 Announce Type: cross  Abstract: Telephone fraud is pervasive and costly, but its inner workings are rarely observed at scale. We analyze a complete corpus of 10,211 inbound scam and spam calls -- 913 hours of audio and 330,956 transcribed turns from 5,780 distinct numbers -- collected over 54 days by an AI voice-agent honeypot that answered callers and kept them talking, and introduced in a companion data descriptor. We separate outright scams, which solicit sensitive information, from the larger stream of predatory but legal lead generation ("spam") that feeds them. Scam operations keep office hours (6.6x more calls per weekday than weekend day); thousands of disposable numbers run a small catalog of recycled scripts (thirty opening clusters, half the traffic in the top five); and callers solicit identity anchors -- a home address and a date of birth -- far more often than payment credentials, pressing through persistence and manufactured authority rather than overt
    
[^280]: 缺乏真实标签的审计：在LLM智能体中对执行回放进行步骤级信用分配审计

    Credit Without Ground Truth: Auditing Step-Level Credit Assignment in LLM Agents Against Executed Replay

    [https://arxiv.org/abs/2608.19760](https://arxiv.org/abs/2608.19760)

    本研究通过执行回放生成因果真实标签，审计发现现有步骤级信用信号在LLM智能体中无法有效识别因果关键步骤，且因果贡献稀疏且依赖模型，揭示了信用分配评估的根本缺陷。

    

    arXiv:2608.19760v1 公告类型：交叉 摘要：在单智能体工具环境（ALFWorld）中，针对执行回放的因果真实标签进行审计，发现用于训练LLM智能体的步骤级信用信号——包括LLM裁判评分、基于结果的日志概率比率或策略自身的置信度——在识别因果关键步骤方面均未优于随机水平。现有评估依据标注的步骤*正确性*来评判这些信号；而我们依据步骤*贡献*进行审计——即在每个决策点重新采样策略自身的替代方案并前向推进，对结果实际产生的改变——这两者存在差异。真实标签本身具有结构性：因果贡献是稀疏的（在定义了真实标签的决策点中，30.5%携带可测量的效应），且可测量性依赖于模型——在两个规模相似的策略之间，无策略支持的反事实点的比例相差两倍（13.1%对26.8%）。失败模式是i

    arXiv:2608.19760v1 Announce Type: cross  Abstract: Audited against causal ground truth from executed replay in a single-agent tool environment (ALFWorld), none of the step-level credit signals used to train LLM agents -- LLM-judge scores, outcome-conditioned logprob ratios, or the policy's own confidence -- identifies which steps causally matter better than chance. Existing evaluations grade these signals against annotated step *correctness*; we audit them against step *contribution* -- what re-sampling the policy's own alternatives at each decision point and rolling forward actually changes about the outcome -- and the two come apart. The ground truth itself is structured: causal contribution is sparse (30.5% of decision points where ground truth is defined carry measurable effect), and measurability is model-dependent -- the fraction of points with no policy-supported counterfactual differs by a factor of two (13.1% vs. 26.8%) between two similar-scale policies. The failure mode is i
    
[^281]: 跨模型记忆迁移：通过目标端阅读器适配

    Cross-Model Memory Transfer via Target-Side Reader Adaptation

    [https://arxiv.org/abs/2608.17050](https://arxiv.org/abs/2608.17050)

    本文研究了跨模型记忆迁移中，冻结记忆与目标端阅读器相对重要性，发现轻量级阅读器适配是关键，而非记忆本身。

    

    arXiv:2608.17050v1 公告类型：交叉 摘要：改进大型语言模型中知识使用的方法通常分为两种模式。非参数检索提供对外部知识的灵活访问，但增加了检索延迟、上下文开销，并且与主干的集成较浅。参数化适配在推理时高效，但将知识与模型权重纠缠在一起，且难以更新、审计或迁移。Engram风格的哈希记忆占据了一个中间模式：它将学习到的信息存储在外部可寻址表中，但通过一个小型学习阅读器来消费该表。这引发了一个基本问题：当这种记忆跨骨干移动时，冻结的记忆本身更重要，还是目标端阅读器更重要？我们通过跨模型冻结记忆提取来研究这个问题，其中在源模型上训练的记忆被冻结并附加到不同的目标模型上，仅训练轻量级阅读器。消融实验表明...

    arXiv:2608.17050v1 Announce Type: cross  Abstract: Methods for improving knowledge use in large language models typically fall into two regimes. Non-parametric retrieval offers flexible access to external knowledge, but adds retrieval latency, context overhead, and only shallow integration with the backbone. Parametric adaptation is efficient at inference time, but entangles knowledge with model weights and can be hard to update, audit, or transfer. Engram-style hashed memory occupies a middle regime: it stores learned information in an external, addressable table, yet consumes that table through a small learned reader. This raises a basic question: when such a memory is moved across backbones, what matters more, the frozen memory itself or the target-side reader? We study this question through cross-model frozen-memory extraction, in which a memory trained on a source model is frozen and attached to a different target model, with only a lightweight reader trained. Ablations show that 
    
[^282]: 从原子证据到逻辑组合：复合答案选项的结构化组合推理

    From Atomic Evidence to Logical Composition: Structured Compositional Reasoning over Compound Answer Options

    [https://arxiv.org/abs/2608.12836](https://arxiv.org/abs/2608.12836)

    提出了一种将复合答案选项分解为原子判断并利用整数线性规划组合分数的方法，显著提升了大型语言模型在逻辑组合推理任务上的表现。

    

    大型语言模型在答案选项需要结合逻辑运算符下的原子判断时常常失败，即使它们能正确判断单个原子。我们研究了由AND、OR和NEITHER/NOR连接的复合选项，引入了一个框架，将每个选项分解为原子答案，并对每个原子的对比假设进行评分，这样模型就永远不会看到复合选项。一个受运算符约束的整数线性规划随后将校准后的分数组合成单一预测。我们在LOGICAL-COMMONSENSEQA上进行了评估，并引入了LOGICAL-SATA，这是一个源自SATA-Bench的阅读理解基准。我们的框架在人工验证的LOGICAL-COMMONSENSEQA分割上将Macro-F1从48.3提高到77.0，在LOGICAL-SATA上从47.0提高到75.6，其中NEITHER/NOR的收益最大。

    arXiv:2608.12836v1 Announce Type: cross  Abstract: Large language models often fail when answer options require combining atomic judgments under explicit logical operators, even when they judge the individual atoms correctly. We study compound options connected by AND, OR, and NEITHER/NOR, introducing a framework that decomposes each option into atomic answers and scores contrastive hypotheses about each one, so the model never sees a compound option. An operator-constrained integer linear program then composes the calibrated scores into a single prediction. We evaluate on LOGICAL-COMMONSENSEQA and introduce LOGICAL-SATA, a reading-comprehension benchmark derived from SATA-Bench. Our framework improves Macro-F1 from 48.3 to 77.0 on the human-validated LOGICAL-COMMONSENSEQA split and from 47.0 to 75.6 on LOGICAL-SATA, with the largest gains on NEITHER/NOR.
    
[^283]: 这个引用是否恰当？

    Is this Citation on Point?

    [https://arxiv.org/abs/2608.12571](https://arxiv.org/abs/2608.12571)

    本研究通过受控扰动实验发现，现有大语言模型在法律引用验证中能有效识别错误案例，但对同一案例内错误页码的检测能力显著不足，且这一失败模式在现有评估中常被忽视。

    

    arXiv:2608.12571v1 公告类型：交叉 摘要：2023年，纽约一名法官在Mata诉Avianca案中制裁了两名律师，因为他们提交的辩护状中包含由ChatGPT生成的虚构引用。这类失败大多可通过数据库查找发现；更棘手的问题是检测那些指向真实案例但并未支持所主张论点的引用——这一失败模式在现有针对法律用例的大语言模型评估中大多被忽视。在本文中，我们通过从两个法律语料库中获取的真实法律引用的受控扰动，研究命题级引用支持验证，方法包括替换被引案例或仅更改同一案例内的精确定位页。我们在所得示例上评估了十四种模型配置。模型能捕捉93-100%的错误案例替换。但对于法院意见中的错误精确定位，模型仅能捕捉37-61%，在法律辩护状中为52-83%。当模型未能捕捉错误精确定位时，它们会接受该引用。

    arXiv:2608.12571v1 Announce Type: cross  Abstract: In 2023, a New York judge sanctioned two attorneys in Mata v. Avianca for filing a brief with hallucinated citations generated by ChatGPT. Such failures are largely caught by database lookups; the harder problem is detecting citations that point to real cases but do not support the propositions for which they are offered -- a failure mode that existing evaluations of LLMs for legal use cases largely overlook. In this paper, we study proposition-level citation support verification through controlled perturbations of real legal citations obtained from two legal corpora, either replacing the cited case or changing only the pinpoint page within the same case. We evaluate fourteen model configurations on the resulting examples. Models catch 93-100% of wrong-case corruptions. They catch only 37-61% of wrong-pinpoint corruptions on court opinions and 52-83% on legal briefs. When models fail to catch wrong-pinpoint corruptions, they accept the
    
[^284]: CoinRAG：面向长上下文检索增强生成（RAG）的上下文化信息微块KV缓存复用

    CoinRAG: Contextualized Information Nugget KV Cache Reuse for Long-Context RAG

    [https://arxiv.org/abs/2608.07458](https://arxiv.org/abs/2608.07458)

    CoinRAG通过两阶段检索定位查询相关语义单元，并以组合方式复用离线计算的细粒度信息微块KV缓存，在低预填充延迟下实现了长上下文RAG效率与准确性的帕累托最优。

    

    近期关于检索增强生成（RAG）的优化研究利用块级KV缓存复用来避免对冗长的检索上下文进行重复处理，从而提高效率；然而粗粒度的块中仍然存在显著的信息冗余与噪声。本文提出CoinRAG（面向长上下文RAG的上下文化信息微块KV缓存复用），在低预填充延迟的约束下优化帕累托前沿，同时最大化准确性。该名称隐喻性地体现了我们的核心机制：正如将小代币（“硬币”）拼合起来可以累积出更大的价值，CoinRAG以组合的方式复用离线计算的细粒度微块缓存，以语义上更相关且更紧凑的方式高效地构建学习到的上下文表示。具体而言，CoinRAG不再对整个块进行编码，而是通过两阶段检索来识别检索块中与查询相关的语义单元，并无缝地……（原文摘要至此截断）

    arXiv:2608.07458v2 Announce Type: replace-cross  Abstract: Recent optimization studies on Retrieval-Augmented Generation (RAG) have exploited chunk-level KV cache reuse to avoid processing long retrieved contexts for higher efficiency, while significant information redundancy and noise still remain in the coarse-grained chunks. This paper optimizes the Pareto frontier under low prefill latency constraints while maximizing accuracy by proposing CoinRAG (Contextualized Information Nugget KV Cache Reuse for Long-Context RAG). The name metaphorically reflects our core mechanism: much like assembling small tokens (or "coins") to accumulate a larger value, CoinRAG compositionally reuses offline-computed, fine-grained nugget caches to form a learned contextual representation efficiently in a more semantically relevant but compact manner. Specifically, instead of full-chunk encoding, CoinRAG identifies query-relevant semantic units within retrieved chunks through two-stage retrieval and seamle
    
[^285]: LoRAScan：通过下投影激活尖峰检测大语言模型低秩适配器中的后门提示

    LoRAScan: Detecting Backdoor Prompts in Low-Rank Adapters for Large Language Models via Down-Projection Activation Spikes

    [https://arxiv.org/abs/2608.06795](https://arxiv.org/abs/2608.06795)

    该论文提出LoRAScan方法，利用携带触发器的输入在下投影层产生的激活尖峰这一独特潜空间特征，来检测大语言模型LoRA适配器中的后门提示。

    

    低秩适配（LoRA）通过紧凑的适配器实现了大语言模型的高效专门化与分发。然而，不可信的适配器会带来供应链威胁：当输入包含隐藏触发器时，带有后门的适配器可能导致模型生成有害内容、恶意代码、政治宣传或隐蔽广告。与适配器无关的防御方法会将适配器与基础模型合并，这会稀释后门信号并降低检测性能。现有的适配器感知方法并未解决如何安全地使用可能带有后门的适配器：它们要么训练一个防御性适配器来修复带有后门的基础模型（解决的是相反的问题，而非保护适配器本身），要么依赖一个将整个适配器标记为可疑的分类器，且需要单独的缓解措施。这些方法都忽略了携带触发器的输入在潜空间中产生的独特特征。

    arXiv:2608.06795v2 Announce Type: replace-cross  Abstract: Low-rank adaptation (LoRA) enables efficient specialization and distribution of large language models through compact adapters. However, untrusted adapters introduce a supply-chain threat: a backdoored adapter can cause a model to generate harmful content, malicious code, political propaganda, or covert advertisements when an input contains a hidden trigger. Adapter-agnostic defenses merge the adapter with the base model, which dilutes backdoor signals and reduces detection performance. Existing adapter-aware methods do not address how to safely use a potentially backdoored adapter. Instead, they either train a defensive adapter to repair a backdoored base model, addressing the inverse problem rather than securing the adapter itself, or rely on a classifier that flags the entire adapter as suspicious and requires separate mitigation. These methods overlook the distinct latent-space signatures produced by trigger-bearing inputs 
    
[^286]: 基因型触发器：通过生成式抗菌肽模型中的宿主特异性后门暴露药物基因组学盲点

    Genotypic Triggers: Exposing Pharmacogenomic Blind Spots via Host-Specific Backdoors in Generative Antimicrobial Peptide Models

    [https://arxiv.org/abs/2608.06779](https://arxiv.org/abs/2608.06779)

    本文提出“基因型触发器”后门攻击方法，通过操纵抗菌肽生成模型，有意使其针对特定HLA等位基因携带者生成免疫原性风险更高的肽（风险评分平均提高743%），暴露了AI药物发现安全验证中忽视个体基因型差异的药物基因组学盲点。

    

    大语言模型加速了药物发现，特别是在抗菌肽（AMP）的自动化设计方面。然而，当前针对肽生成模型的验证流程忽视了历史先例所揭示的问题：某些药物主要对具有特定基因型的个体构成健康风险。在本文中，我们证明这种针对性的健康风险可以通过操纵生成肽候选物的模型而被有意地、大规模地诱导。我们提出了“基因型触发器”，这是一种后门攻击，它将模型的生成分布转向具有更高预测免疫原性风险（一种不良免疫反应）的肽，且专门针对携带特定HLA等位基因（一种参与免疫呈递的基因变异）的个体。在多个流行的肽生成模型上，该攻击使目标等位基因携带者的预测免疫原性风险评分平均提高了743%。

    arXiv:2608.06779v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) have accelerated drug discovery, particularly in the automated design of antimicrobial peptides (AMPs). However, current validation pipelines for peptide generation models overlook historical precedents showing that certain drugs carry health risks predominantly for individuals with specific genetic profiles. In this paper, we demonstrate that such targeted health risks can be induced intentionally and at scale by manipulating models that generate peptide candidates. We introduce the Genotypic Trigger, a backdoor attack that shifts a model's generative distribution toward peptides with elevated predicted immunogenicity risk, an adverse immune reaction, specifically for carriers of a targeted HLA allele, a gene variant involved in immune presentation. Across popular peptide generation models, the attack increased the predicted immunogenicity risk score for target-allele carriers by 743% on average re
    
[^287]: 模型收敛而人类分歧之处：开放式生成中分布多元性的覆盖框架

    Where Models Converge and Humans Diverge: A Coverage Framework for Distributional Pluralism in Open-Ended Generation

    [https://arxiv.org/abs/2608.05576](https://arxiv.org/abs/2608.05576)

    提出了一个以人类写作为基准的覆盖框架，用于衡量LLM在开放式生成中对人类响应分布的覆盖程度，识别模型收敛而人类分歧之处，从而系统性地揭示LLM倾向生成“平均化”内容而人类写作分布更多元的差距。

    

    当大型语言模型（LLM）撰写哈利·波特同人小说时，它能够生成霍格沃茨宇宙的基本元素，例如可辨识的地点和角色。然而，人类撰写的哈利·波特同人小说通常不仅包含这些基本元素，还包含更丰富的内容，融入风格不拘一格的写作和关系多样化的情节线。LLM与人类写作之间的这种差距已在多个领域被注意到：LLM倾向于产生“平均化”的写作，而人类写作包含更多样的内容，覆盖更广泛的分布。现有工作已经证明了这种分布“差距”的存在。我们更进一步，提供了一个系统性框架来衡量模型对人类响应分布的覆盖程度，并识别模型样本能够到达或遗漏哪些区域。我们的论文提出了一个以人类为基准的框架，利用某一主题上人类写作的经验分布来衡量分布性多元（摘要在此处截断）。

    arXiv:2608.05576v2 Announce Type: replace  Abstract: When a large language model (LLM) writes Harry Potter fanfiction, it can produce fundamental elements of the Hogwarts universe, such as recognizable places and characters. Human-written Harry Potter fanfiction, however, typically includes these fundamentals and much more, incorporating stylistically irregular content and relationship-diverse plotlines. This gap between LLM and human writing has been noted across a variety of domains. LLMs tend to produce "average" writing, while human writing contains more diverse content that covers a broader distribution. Existing work has shown the existence of this distributional "gap". We take a further step by providing a systematic framework to measure coverage of the human response distribution and identify which regions model samples reach or miss.   Our paper proposes a human-grounded framework that uses the empirical distribution of human writing on a topic to measure the distributional br
    
[^288]: 准确率等同于证据吗？KV缓存压缩下的推理忠实性

    Does Accuracy Equal Evidence? Reasoning Faithfulness under KV Cache Compression

    [https://arxiv.org/abs/2608.01631](https://arxiv.org/abs/2608.01631)

    本文揭示了KV缓存压缩评估中“准确率等同于证据”这一隐含假设的缺陷——即使最终答案准确率保持不变，压缩仍可能显著破坏支持答案的推理过程的忠实性，并提出固定轨迹回放协议来分离测量这一现象。

    

    KV缓存压缩通常通过最终答案准确率来评估，这隐含地假设了保住答案就等于保住了支持该答案的推理过程。我们针对大型推理模型检验了这一假设，并证明它可能失效：在压缩条件下，正确答案与其可见的支持性推理依据的有效性可能以不同的速率被保留。我们采用一种受控的固定轨迹回放协议来研究这一失效现象，该协议保持推理内容固定不变，从而隔离出压缩是否能够从一条已有的推理轨迹中保留可用信息。我们在三个模型上评估了十种token逐出式KV压缩方法和一种量化方法，涵盖数学推理、科学问答、临床计算和长上下文检索等任务，并测量了最终准确率、答案与推理链的一致性以及扰动忠实性。实验表明，在各类任务中，token逐出方法可以在保持具有竞争力的最终答案准确率的同时，显著损害推理依据的忠实性。

    arXiv:2608.01631v2 Announce Type: replace  Abstract: KV cache compression is commonly evaluated by final-answer accuracy, implicitly assuming that preserving the answer also preserves the reasoning that supports it. We test this assumption for large reasoning models and show that it can fail: under compression, correct answers and the validity of their visible supporting rationales can be preserved at different rates. We study this failure with a controlled fixed-trace replay protocol, which holds reasoning content fixed and isolates whether compression preserves usable information from an already available trace. We evaluate ten token-eviction KV compression methods and one quantization method on three models across mathematical reasoning, scientific QA, clinical calculation, and long-context retrieval. We measure final accuracy, answer-chain consistency, and perturbation faithfulness. Across tasks, token-eviction methods can preserve competitive final-answer accuracy while substantia
    
[^289]: 人类与大语言模型对非母语日语语言态度的一致性研究

    Human-LLM Alignment in Language Attitudes Toward Non-Native Japanese

    [https://arxiv.org/abs/2608.01629](https://arxiv.org/abs/2608.01629)

    本研究基于语言态度框架比较人类与LLM对二语日语写作的评价，发现LLM以结构化但被削弱的形式再现了母语者对非母语写作的偏见方向与维度排序，但低估了团结度差距，并对学习者母语背景做出了人类未表现的区分。

    

    大语言模型（LLM）越来越多地在招聘和学术评估等高风险领域对人类写作进行评价，这使非母语使用者面临特殊风险。基于语言态度框架，我们比较了人类与LLM对平行书写的母语（L1）和二语（L2）日语电子邮件在三个维度上的评价：流利度、地位和团结度。日语评分者在所有三个维度上对L2文本的评分都显著更低，其中流利度差距大约是地位差距和团结度差距的两倍。六个LLM评判者均再现了这种偏见的方向，其中五个再现了各维度之间的排序。这些模型在两个方面与人类存在分歧：所有模型都低估了团结度差距——这是最具社会基础的维度；并且所有模型都对学习者的不同母语背景进行了区分，而人类并未如此。因此，LLM评判者以一种结构化但被削弱的形式再现了母语使用者的语言态度。

    arXiv:2608.01629v2 Announce Type: replace  Abstract: Large language models (LLMs) increasingly evaluate human writing in high-stakes domains such as hiring and academic assessment, putting non-native speakers at particular risk. Drawing on the language attitudes framework, we compared human and LLM evaluations of parallel L1- and L2-written Japanese emails on three dimensions: fluency, status, and solidarity. Japanese raters rated L2 texts significantly lower on all three dimensions, with a fluency gap roughly twice the size of the status and solidarity gaps. Six LLM judges reproduced the direction of this bias, and five reproduced its ordering across dimensions. The models diverged from humans in two ways: all understated the solidarity gap, the most socially grounded dimension, and all differentiated among learner L1 backgrounds where humans did not. LLM judges thus reproduce native speakers' language attitudes in a structured yet attenuated form, and the language attitudes framework
    
[^290]: 一个虚假的平均值：思维链监控器在其作为唯一防线之处崩溃

    A False Average: Chain-of-Thought Monitors Collapse Where They Are the Only Defense

    [https://arxiv.org/abs/2608.00583](https://arxiv.org/abs/2608.00583)

    仅重写智能体的思维链推理（保持动作完全不变）即可在一次无梯度攻击中将CoT监控器的捕获率从约95%暴跌至11%以下，揭示监控器的总体准确率是掩盖其在唯一防线场景下近乎全面失效的虚假平均值。

    

    思维链监控旨在捕捉那些在动作层面看起来干净、只在推理中暴露自身的奖励作弊行为。我们证明，这恰恰是控制推理的攻击者能够击败它的地方。仅重写智能体的推理内容使其看起来像是善意的工程工作，同时逐字复制每个命令和输出以保持漏洞利用不变，在一次无需梯度的攻击中，就能将留出监控器在该子集上的捕获率从约95%降至11%以下。监控器的总体准确率是一个虚假的平均值：它由动作层面就会暴露的作弊行为主导，从而掩盖了这种重写在CoT监控是唯一信号的子集上造成的近乎全面的崩溃。该攻击可跨监控器系列和智能体模型迁移，并在真实智能体上复现，不过在面对经过校准的监控器时，规避行为主要集中在最强的智能体上。仅基于轨迹的防御只能部分恢复捕获率，即使是预先针对该攻击进行训练的防御也是如此。

    arXiv:2608.00583v2 Announce Type: replace-cross  Abstract: Chain-of-thought (CoT) monitoring is meant to catch the reward hacks that look clean in the actions and betray themselves only in the reasoning. We show that this is exactly where an adversary who controls the reasoning can defeat it. Rewriting only an agent's reasoning to read as good-faith engineering, while copying every command and output verbatim so the exploit is unchanged, drops a held-out monitor's catch rate on that subset from about 95% to under 11% in one gradient-free shot. A monitor's aggregate accuracy is a false average: dominated by hacks the actions give away, it hides the near-total collapse this rewrite produces on the subset where CoT monitoring is the only signal. The attack transfers across monitor families and agent models, reproduces with live agents, though against a calibrated monitor evasion concentrates in the strongest agent. Trace-only defenses recover it only partially, even one primed on the atta
    
[^291]: 弃权选项衡量了什么：蒸馏小型语言模型中的拒绝损失、采样波动与选项捕获

    What the Abstention Option Measures: Refusal Loss, Sampling Churn, and Option Capture in Distilled Small Language Models

    [https://arxiv.org/abs/2607.28639](https://arxiv.org/abs/2607.28639)

    该研究将蒸馏小型语言模型在偏见基准弃权选项上的行为变化分解为拒绝损失、采样波动和选项捕获三种机制，并证明聚合分数仅能反映拒绝损失，只有以已提交答案为条件进行评估，才能避免将拒绝行为的移动误读为公平性的提升。

    

    通过知识蒸馏，小型指令微调语言模型被部署到手机、汽车和离线助手中，而一个蒸馏出的模型是否足够公平、可以发布，通常由基准测试中的多项选择偏见分数决定，该基准提供了弃权选项。有三种机制会改变模型围绕弃权选项的行为：拒绝损失，即学生模型在其蒸馏前参考模型曾拒绝的项目上不再拒绝；采样波动，即在参考模型自身的两次运行之间观察到的相同变化；以及选项捕获，即学生模型在提供弃权选项时倾向于选择该选项。将每种机制注入参考模型自身的记录中表明，聚合分数只能看到彻底的拒绝损失，逐项分数则能看到波动和重映射，而只有以已提交的答案为条件，才能避免被移动的拒绝数量被误读为公平性。仅在不同推理种子下对参考模型进行重采样，就已经会使其10%到27%的拒绝变成刻板印象答案。

    arXiv:2607.28639v3 Announce Type: replace-cross  Abstract: Small instruction-tuned language models reach phones, cars, and offline assistants through knowledge distillation, and whether a distilled model is fair enough to ship is often decided by a multiple-choice bias score on a benchmark that offers an abstention option. Three mechanisms move behavior around the abstention option: refusal loss, where the student stops declining on items its pre-distillation reference declined; sampling churn, the same change observed between two runs of the reference itself; and option capture, where the student takes an abstention option whenever one is offered. Injecting each mechanism into a reference's own records shows that aggregate scores see only outright loss, per-item scores see churn and remapping, and only conditioning on committed answers stops moved refusal mass from reading as fairness. Resampling a reference across inference seeds already turns 10% to 27% of its refusals into stereoty
    
[^292]: Change2Task：从仓库变更到可执行的编码智能体任务与环境

    Change2Task: From Repository Changes to Executable Coding Agent Tasks and Environments

    [https://arxiv.org/abs/2607.28591](https://arxiv.org/abs/2607.28591)

    Change2Task 提出了一个基于仓库历史的系统，能够将已合并的拉取请求自动转换为健康现代代码版本上经过验证的可执行编码智能体任务与环境，为编码智能体的训练、基准测试和持续评估提供可扩展的数据供给。

    

    扩展编码智能体需要持续供给可执行的数据，用于训练、基准测试和持续评估。每个任务必须将真实的软件状态与任务规范、开发工具和可靠的验证相结合。为了扩大这一供给，我们提出了 Change2Task，这是一个基于仓库历史的系统，它将已合并的拉取请求（PR）转换为同一仓库健康现代版本上经过验证的任务。它将历史证据与演进的代码进行对齐，通过补丁反转、代码映射或智能体重建三种方式重建任务状态，并验证从健康基础状态到任务状态再到恢复状态的完整生命周期。通过从维护良好的环境中推导出多个基于开发者证据的任务，Change2Task 为编码智能体的训练和评估提供了可执行数据，同时减少了重复的环境搭建、存储和任务构建成本。我们评估了五个编码智能体……（原文截断）

    arXiv:2607.28591v2 Announce Type: replace-cross  Abstract: Scaling coding agents requires a continuing supply of executable data for training, benchmarking, and continuous evaluation. Each task must couple a realistic software state with a specification, development tools, and reliable verification. To expand this supply, we present Change2Task, a system grounded in repository history that converts merged pull requests (PRs) into verified tasks on healthy modern revisions of the same repository. It aligns historical evidence with evolved code, reconstructs task states through Patch Reversal, Code Mapping, or Agent Reconstruction, and validates the lifecycle from a healthy base to a task state and a restored state. By deriving multiple tasks grounded in developer evidence from maintained environments, Change2Task provides executable data for coding agent training and evaluation while reducing repeated environment setup, storage, and task construction effort. We evaluate five coding agen
    
[^293]: 标记而非抹除：面向大语言模型两用知识的Token接种方法

    Mark, Don't Erase: Token Inoculation for Dual-Use Knowledge in LLMs

    [https://arxiv.org/abs/2607.18639](https://arxiv.org/abs/2607.18639)

    提出“Token接种”方法，在继续预训练时用特殊标记标注危险句子并训练模型条件化地预测危险答案，测试时因标记缺失而抑制危险输出，从而在保持93%良性性能的同时将危险准确率控制在约18%。

    

    在两用领域中，危险知识与围绕它的良性专业知识共享词汇、概念和因果结构。这种重叠使得选择性拒绝变得困难，而诸如去学习和拒绝训练等现有方法会将其抑制过度泛化到良性知识一侧。我们主张采用条件化而非抑制。在本工作中，我们提出了Token接种方法，在继续预训练期间用一个特殊标记（即接种标记）来标记危险句子。随后我们微调模型，使其拒绝危险问题，同时在带有接种标记的相同问题上预测危险答案。在测试时，接种标记并不存在，因此模型不会表达它学会预测的危险答案。在1.5B到14B参数范围以及三种架构家族中，该方法将危险准确率保持在约18%，同时保留了指令基线93%的MMLU性能，实现了最佳的安全-效用平衡。

    arXiv:2607.18639v2 Announce Type: replace-cross  Abstract: In a dual-use field, hazardous knowledge and the benign expertise around it share vocabulary, concepts, and causal structure. That overlap makes selective refusal hard, and existing methods such as unlearning and refusal training overgeneralize their suppression into the benign side. We argue for conditioning, not suppression. In this work, we introduce Token Inoculation, which marks hazardous sentences with a special token, the inoculation token, during continued pre-training. We then fine-tune the model to refuse hazardous questions and, on the same questions under the inoculation token, to predict the hazardous answers. At test time the inoculation token is absent, so the model does not express the hazardous answers it learned to predict. Across 1.5B to 14B parameters and three architecture families it holds hazardous accuracy near 18% while retaining 93% of the instruct baseline's MMLU performance, the best safety-utility b
    
[^294]: CANDOR：冻结基础编码器中的机会校准不一致性

    CANDOR: Chance-Calibrated Discordance in Frozen Foundation Encoders

    [https://arxiv.org/abs/2607.18451](https://arxiv.org/abs/2607.18451)

    本文提出CANDOR度量，通过等大小对称样本库校正最近邻不一致性，将机会水平固定为二分之一，揭示冻结编码器并非失明但普遍性能较弱。

    

    摘要：arXiv:2607.18451v2 公告类型：替换-交叉 摘要：冻结编码器的选择取决于轻量级头部从其特征中读取发现的能力，而非几何结构是否将其分离。最近邻不一致性可以做到这一点，但在样本库不均衡的情况下，相反标签的邻居会因密度而非几何结构获胜，因此仅凭患病率就会使无信息编码器看起来失明。我们引入了CANDOR，一种不一致性度量，其等大小样本库在标签交换下对称，将机会水平精确固定在二分之一。在22个编码器、来自7个领域的20个数据集和605,443张图像上，这一修正逆转了结论。崩溃几乎在所有地方都低于机会水平，因此没有编码器是失明的，但所有编码器都较弱：最佳胸部模型以84.5 AUROC读取气胸，但仍将18.4%的阳性样本放置在比同医院同类更接近相反标签的影像附近。同一个在鸟类物种分辨上达到4.5的编码器，在胸部发现上为42.8，在青光眼上为49.8，处于机会水平或更差。

    arXiv:2607.18451v2 Announce Type: replace-cross  Abstract: Frozen encoders are chosen by how well a lightweight head reads a finding from their features, not whether the geometry separates it. Nearest-neighbor discordance does, but with unequal banks the opposite-label neighbor wins on density, not geometry, so prevalence alone makes an uninformed encoder look blind. We introduce CANDOR, a discordance measure whose equal-size banks are symmetric under a label swap, fixing its chance level at exactly one half. Across 22 encoders, 20 datasets from 7 domains, and 605,443 images, this correction reverses the conclusion. Collapse falls below chance almost everywhere, so no encoder is blind, yet all are weak: the best chest model reads pneumothorax at 84.5 AUROC and still places 18.4% of those positives nearer an opposite-label film than its own kind in the same hospital. The same encoder that resolves bird species at 4.5 leaves chest findings at 42.8 and glaucoma at 49.8, at chance and wors
    
[^295]: Pancasila-Dilemmas：基于潘查希拉评估大语言模型在印度尼西亚人类价值困境上的表现

    Pancasila-Dilemmas: Evaluating Large Language Models on Indonesian Human Value Dilemmas Grounded in Pancasila

    [https://arxiv.org/abs/2607.18066](https://arxiv.org/abs/2607.18066)

    该论文构建了首个基于印度尼西亚国家意识形态“潘查希拉”五大价值观的价值对齐评估数据集Pancasila-Dilemmas（包含1,834个源自印尼新闻的两难困境问题），对50个大语言模型的评估发现它们与印尼价值观的对齐程度均不足0.5，凸显了当前模型在非西方价值体系上的局限性。

    

    大语言模型（LLM）的价值对齐对于确保其响应符合人类意图和价值偏好至关重要。然而，大多数价值对齐评估都聚焦于西方价值观或普世价值观，而基于特定国家价值体系的评估仍然稀缺。在本文中，我们提出了Pancasila-Dilemmas数据集，这是一个包含1,834个问题的评估数据集，这些问题源自印度尼西亚新闻，并按照潘查希拉（Pancasila）的五大价值观进行分类：宗教、人道、团结、民主和社会公正。该数据集反映了印度尼西亚的日常生活，因此适合用于衡量部署于印度尼西亚的大语言模型的价值对齐程度。为确保更严格的评估，我们选取了包含两难困境的场景。该数据集由母语人士进行校对，并由5位背景各异的印度尼西亚公民作答。我们在该数据集上评估了50个闭源和开源大语言模型。结果显示，所有被评估的大语言模型的Proba（摘要在此处被截断）

    arXiv:2607.18066v2 Announce Type: replace  Abstract: The value alignment of large language models (LLMs) is crucial for ensuring responses align with human intention and value preferences. However, most evaluations of value alignment focus on Western or universal values, while assessments grounded in the value systems of specific countries remain scarce. In this paper, we introduce Pancasila-Dilemmas, an evaluation dataset of 1,834 questions derived from Indonesian news, classified by 5 values of Pancasila: Religion, Humanity, Unity, Democracy, and Social Justice. This dataset reflects daily life in Indonesia, making it suitable for measuring the value alignment of LLMs deployed for Indonesia. To ensure a more rigorous evaluation, we choose scenarios containing dilemmas. The dataset is proofread by native speakers and answered by 5 diverse Indonesian citizens. We evaluate 50 closed- and open-source LLMs on our dataset. Results reveal that all evaluated LLMs achieves less than 0.5 Proba
    
[^296]: 循环吧，Loopie！

    Loop the Loopies!

    [https://arxiv.org/abs/2607.16051](https://arxiv.org/abs/2607.16051)

    Loopie 是迄今最强大的开源循环 Transformer，它解决了循环次数不如扩大参数量的长期难题，在相同计算预算下显著超越标准 Transformer，并通过新颖的后训练方法实现了前沿水平的推理能力。

    

    我们推出了 Loopie，这是迄今最强大的开源循环 Transformer。Loopie 系列包含两个混合专家模型：一个总参数量为 20B、激活参数为 2B 的模型，以及一个总参数量为 6B、激活参数为 0.6B 的模型。循环 Transformer 长期以来一直面临一个挑战：在给定 N 倍预训练计算量的情况下，将参数量扩大 N 倍通常优于将模型循环 N 次。Loopie 解决了这一挑战。大量消融实验（包括与标准 30B-A3B 模型的对比）表明，Loopie 在相同计算预算下显著优于标准 Transformer 基线。借助一种新颖的后训练方法，Loopie 培养出强大的推理能力，达到了前沿水平的推理性能。

    arXiv:2607.16051v3 Announce Type: replace-cross  Abstract: We present Loopie, the most powerful open-source looped Transformer to date. The Loopie series consists of two Mixture-of-Experts (MoE) models: a 20B-parameter model with 2B active parameters and a 6B parameter model with 0.6B active parameters. Looped Transformers have long faced a challenge: given an N-fold pre-training compute, increasing the parameter count by a factor of N usually outperforms looping N times. Loopie addresses this challenge. Extensive ablation studies, including comparisons with a vanilla 30B-A3B model, show that Loopie substantially outperforms vanilla Transformer baselines trained with the same compute budget. With a novel post-training method, Loopie develops strong reasoning abilities and achieves frontier-level reasoning performance.
    
[^297]: 面向工具结构化大语言模型推理的仅幅度FFN干预：方法、门控评估协议与跨模型实证结果

    Amplitude-Only FFN Intervention for Tool-Structured LLM Inference: Method, Gated Evaluation Protocol, and Cross-Model Empirical Results

    [https://arxiv.org/abs/2607.11183](https://arxiv.org/abs/2607.11183)

    该论文提出幅度门控（AG）这一无需重训练的推理时干预方法，通过掩码FFN激活坐标，在多个Qwen模型的函数调用和ToolBench端点上实现了1.6至2.8个百分点的稳定性能提升。

    

    我们研究了幅度门控，这是一种推理时干预方法，在前馈网络（FFN）激活坐标上进行掩码操作，同时保持预训练权重固定。六个SwiGLU位置分别对共享输入与仅门控输入、扩展的门控/数值通道以及FFN输出进行分离。幅度排序熵掩码和Newton-Schulz窗口排序变体提供了无需重新训练的替代前向路径。修正后的九月数据集涵盖三个Qwen检查点和八个文本/工具端点。在冻结的本地文本评分器下，所展示的Qwen3.5-9B非NS配置在两个函数调用端点上分别提升了2.61和2.67个百分点（pp），在静态ToolBench的域内/域外端点上分别提升了1.64和2.82个百分点。将较低的历史/九月AG均值与较高的基线均值进行比较，在这些路径上仍保留正的裕度。这支持了对所观测的两次执行的鲁棒性，而非形式上的统计显著性。

    arXiv:2607.11183v4 Announce Type: replace-cross  Abstract: We study Amplitude Gating (AG), an inference-time intervention that masks feed-forward network (FFN) activation coordinates while keeping pretrained weights fixed. Six SwiGLU sites separate shared and gate-only inputs, expanded gate/value channels, and the FFN output. Magnitude-ranked entropy masks and a Newton-Schulz windowed ranking variant provide alternative forward paths without retraining. The corrected September collection covers three Qwen checkpoints and eight text/tool endpoints. Under frozen local text scorers, the displayed Qwen3.5-9B non-NS configurations improve two function-call endpoints by +2.61 and +2.67 percentage points (pp), and static ToolBench in/out-domain endpoints by +1.64 and +2.82 pp. Comparing the lower historical/September AG mean with the higher baseline mean leaves positive margins on these routes. This supports robustness to the two observed executions, rather than formal statistical significanc
    
[^298]: SAMPA：基于Transformer的巴西葡萄牙语韵律边界分割

    SAMPA: Transformer-based segmentation of prosodic boundaries in Brazilian Portuguese

    [https://arxiv.org/abs/2607.07408](https://arxiv.org/abs/2607.07408)

    本文提出SAMPA，通过在人工标注数据上微调Whisper large-v3模型，实现巴西葡萄牙语语音转录的同时自动标注终端韵律边界，性能超越了开箱即用的Whisper基线和基于特征的分类器。

    

    自动韵律分割基于声学和语言学证据来识别语音单元之间的边界。尽管近期的深度学习方法在英语上取得了优异的成果，但巴西葡萄牙语（BP）的自动分割仍然主要依赖基于规则或传统机器学习的方法。本文提出了SAMPA，一个基于Whisper的分割器，它在转录巴西葡萄牙语语音的同时插入明确的终端韵律边界标记。我们在NURC-Sampa数据集的人工分割录音上对Whisper large-v3进行微调，并评估了不同的训练和测试时过滤配置，包括在MuPe-Diversidades数据集上进行的分布外测试。在不使用测试时过滤的情况下，最佳模型在留出测试集上达到0.730的二分类f1值，在MuPe-Diversidades上达到0.793，超越了开箱即用的Whisper基线和重新训练的基于特征的分类器。

    arXiv:2607.07408v3 Announce Type: replace-cross  Abstract: Automatic prosodic segmentation identifies boundaries between speech units based on acoustic and linguistic evidence. Although recent deep learning approaches have achieved strong results for English, automatic segmentation for Brazilian Portuguese (BP) still relies largely on rule-based or traditional machine-learning methods. This paper presents SAMPA, a Whisper-based segmenter that transcribes BP speech while inserting explicit markers for terminal prosodic boundaries. We fine-tune Whisper large-v3 on manually segmented recordings from the NURC-Sampa dataset and evaluate different training and test-time filtering configurations, including out-of-distribution testing on the MuPe-Diversidades dataset. Without test-time filtering, the best models reach f1 binary of 0.730 on the held-out test split and 0.793 on MuPe-Diversidades, outperforming an out-of-the-box Whisper baseline and a retrained feature-based classifier. Finally, 
    
[^299]: 重新思考智能体时代的科学发现

    Rethinking Scientific Discovery in the Agentic Era

    [https://arxiv.org/abs/2607.03863](https://arxiv.org/abs/2607.03863)

    提出了SCION——一个作为组织枢纽的智能体科学操作系统，通过研究执行计划（REP）和元管理框架将碎片化的AI4Science工具整合为可执行、可审计、可复用的完整科研流程。

    

    人工智能已经推动了科学发现，但大多数AI4Science系统仍然是碎片化的工具，依赖人类来协调问题构建、文献梳理、模型使用、仿真模拟、验证和知识复用等环节。本文提出了SCION（基于智能体组织枢纽的科学协作创新），一个作为“组织枢纽”的智能体科学操作系统。通过一个充当“元管理框架”的科学智能体，SCION连接科学任务、工具、智能体、研究成果和记忆，将科研转化为可执行、可审计、可复用的运行流程。其核心是“研究执行计划”（REP），它将高层次的科学意图编译为分阶段目标、依赖关系、验证检查点、工具需求、预期成果和回退条件。SCION进一步集成了分层多智能体执行机制……

    arXiv:2607.03863v3 Announce Type: replace  Abstract: Artificial intelligence has advanced scientific discovery, but most AI4Science systems remain fragmented tools that rely on humans to coordinate problem formulation, literature grounding, model use, simulation, validation, and knowledge reuse. This paper presents \textbf{SCION (Scientific Collaborative Innovation with Agentic Organizational Nexus)}, an agentic scientific operating system that acts as an \textbf{organizational nexus}. Through a Science Agent serving as a \textbf{Meta-Harness}, SCION connects scientific tasks, tools, agents, artifacts, and memory, transforming research into an executable, auditable, and reusable operational process. At its core is the \textbf{Research Execution Plan (REP)}, which compiles high-level scientific intent into staged objectives, dependencies, verification checkpoints, tool requirements, expected artifacts, and fallback conditions. SCION further integrates hierarchical multi-agent execution,
    
[^300]: CausalMix：将数据配比视为因果推断用于语言模型训练

    CausalMix: Data Mixture as Causal Inference for Language Model Training

    [https://arxiv.org/abs/2607.01104](https://arxiv.org/abs/2607.01104)

    CausalMix将语言模型训练中的数据配比优化建模为因果推断问题，通过小规模实验拟合因果模型估计条件平均处理效应（CATE），从而将最优数据配比外推到更大数据池和更大模型规模，避免了数据分布变化时从头重训的高昂成本。

    

    在大型语言模型（LLM）训练中，数据配比在决定模型性能方面起着关键作用。近期的方法通过代理模型来优化混合权重，但它们依赖于静态数据分布的假设。因此，当底层数据池发生变化时，这些方法需要付出高昂代价从头重新训练。这一限制制约了它们从小规模设置向更大数据池和更大模型规模无缝扩展的能力。在本文中，我们提出CausalMix，通过将数据配比优化转化为一个因果推断问题来解决这一限制。我们将数据池的统计特征形式化为协变量，将领域配比形式化为处理变量。在512次Qwen2.5-0.5B训练运行上拟合因果模型以估计条件平均处理效应（CATE）之后，我们外推出80万数据池的最优配比，并将其应用于训练一个7B模型。此外，我们成功地将……

    arXiv:2607.01104v2 Announce Type: replace-cross  Abstract: In Large Language Model (LLM) training, data mixing plays a pivotal role in determining model performance. Recent methods optimize mixture weights via proxy models, but they rely on the assumption of static data distributions. As a result, when the underlying data pool shifts, these methods require costly retraining from scratch. This limitation restricts their ability to scale seamlessly from small settings to larger data pools and model sizes. In this paper, we propose CausalMix to address this limitation by casting data mixture optimization as a causal inference problem. We formulate the statistical features of the data pool as covariates and the domain mixture as the treatment. After fitting a causal model on 512 runs of Qwen2.5-0.5B to estimate the Conditional Average Treatment Effect (CATE), we extrapolate the optimal mixture for an 800K data pool and apply it to train a 7B model. Furthermore, we successfully generalize t
    
[^301]: HealthAgentBench：用于挑战前沿AI智能体的统一真实智能体医疗环境基准套件

    HealthAgentBench: A Unified Benchmark Suite of Realistic Agentic Healthcare Environments for Challenging Frontier AI Agents

    [https://arxiv.org/abs/2606.31179](https://arxiv.org/abs/2606.31179)

    该论文提出了HealthAgentBench，一个包含75个智能体医疗任务、涵盖9个类别且各具独特环境的统一基准套件，通过模拟端到端临床工作流程，评估AI智能体在真实医疗场景中处理原始数据并执行复杂多步骤任务的能力。

    

    随着AI智能体在复杂、长程推理方面的能力日益增强，严格的评估对于衡量其在真实世界医疗应用方面的进展至关重要。然而，现有的医疗基准往往是静态的或临床范围狭窄，这促使人们需要更广泛地评估智能体处理原始患者数据的能力。我们提出了HealthAgentBench，这是一个包含75个智能体医疗任务的基准套件，涵盖9个类别，每个类别都有其独特的环境。该基准套件涵盖诊断、标本评估、研究和数据管理工作流程，以及广泛的模态。每个任务都旨在复制端到端的临床工作流程：在给定任务说明和支持资源的情况下，智能体必须探索原始医疗数据，在复杂环境中进行操作，并执行超越简单提示的多步骤解决方案，例如查询大型临床数据库或解读千兆像素病理图像。

    arXiv:2606.31179v2 Announce Type: replace  Abstract: As AI agents become increasingly capable of complex, long-horizon reasoning, rigorous evaluation is essential for measuring progress toward real-world healthcare applications. However, existing healthcare benchmarks are often static or narrow in clinical scope, motivating broader evaluation of agents acting on raw patient data. We introduce HealthAgentBench, a suite of 75 agentic healthcare tasks across 9 categories each with its unique environment. The benchmark suite spans diagnosis, specimen assessment, research, and data management workflows and a broad range of modalities. Each task is designed to replicate an end-to-end clinical workflow: given task instructions and supporting resources, an agent must explore raw healthcare data, operate within a complex environment, and execute multi-step solutions that go beyond naive prompting, such as querying large clinical databases or interpreting gigapixel pathology images. All tasks ar
    
[^302]: 诊断与缓解长程搜索中的上下文腐化

    Diagnosing and Mitigating Context Rot in Long-horizon Search

    [https://arxiv.org/abs/2606.29718](https://arxiv.org/abs/2606.29718)

    该论文通过对四个旗舰模型在三个基准上的系统性研究，首次揭示了“无结论搜索失败”这一此前被忽视的上下文腐化现象——模型远在用尽上下文窗口之前就放弃或给出不确定的错误答案，即使对简单查询也普遍存在，并进一步诊断与缓解这一问题。

    

    随着大语言模型（LLM）越来越多地被部署于长程搜索任务，使用大量上下文已成为常态。上下文长度增加会导致模型能力下降的担忧，即“上下文腐化”，已成为这些应用中被广泛承认的问题。然而，在深度搜索场景中，模型在大量上下文下究竟如何失败，以及现有方法能在多大程度上缓解这些失败，仍不清楚。通过对四个旗舰模型在三个基准上的系统性研究，我们发现了一个此前被忽视的现象：在大量上下文的情况下，模型远在用尽上下文窗口之前就会放弃或给出不确定的错误答案。我们将这两种失败模式称为“无结论搜索失败”。我们表明，即使对于简单和中等难度的查询，这些也是常见的失败模式，这表明模型会在其本有能力解决的查询上过早终止……

    arXiv:2606.29718v3 Announce Type: replace-cross  Abstract: Extensive context has become the norm as Large Language Models (LLMs) are increasingly deployed in long-horizon search tasks. The concern that increasing context length degrades model capabilities, known as context rot, has become a widely recognized issue for these applications. However, in deep search scenarios, it remains unclear how models actually fail under extensive context, and to what extent existing methods can mitigate such failures. Through a systematic study of four flagship models across three benchmarks, we identify a previously overlooked phenomenon: under extensive context, models give up or provide uncertain incorrect answers long before exhausting the context window. We refer to these two failure modes as inconclusive search failures. We show that these are common failure modes even for easy and moderately difficult queries, suggesting premature termination on queries that the model is capable of solving. Bas
    
[^303]: SURGELLM：通过任务感知特征门控与类平衡归一化重新思考多任务评估

    SURGELLM: Rethinking Multi-Task Evaluation through Task-Aware Feature Gating with Class-Balanced Normalization

    [https://arxiv.org/abs/2606.24259](https://arxiv.org/abs/2606.24259)

    SURGELLM提出一个统一Transformer框架，通过外科手术式特征门控、任务条件化前缀令牌和实例加权归一化三个轻量级模块，解决多任务编码器中归纳偏置不匹配、类别不平衡和外部词汇知识缺失三大问题，并从理论上证明了超额风险界。

    

    部署于异构NLP任务上的微调编码器面临三个相互叠加的问题：归纳偏置不匹配、特征统计被类别不平衡破坏，以及缺乏将注意力条件化于外部词汇知识的机制。我们提出了SURGELLM，一个统一的Transformer框架，通过专用轻量级模块逐一解决这些问题：外科手术式特征门控（surgical feature gate，对精选词汇指标和[CLS]学习逐维度的sigmoid，当特征无信息时可证明退化为恒等映射）、任务条件化前缀令牌（task-conditioned prefix tokens，将量化特征值和任务身份添加到每个输入的前面），以及实例加权归一化（Instance-Weighted Normalization，IWN，从门控统计中消除类别先验偏差）。我们证明了一个超额风险界，将门控收益与外科手术式特征对齐联系起来。在四个任务上——SST-2、多跳检索、LLM提示归因和作者身份检测……

    arXiv:2606.24259v2 Announce Type: replace-cross  Abstract: Fine-tuned encoders deployed across heterogeneous NLP tasks face three compounding problems: mismatched inductive biases, class-imbalance corruption of feature statistics, and no mechanism to condition attention on external lexical knowledge. We introduce \textbf{\surgellm}, a unified transformer framework that addresses each with a dedicated lightweight module: a \emph{surgical feature gate} (learned per-dimension sigmoid over curated lexical indicators and \texttt{[CLS]}; provably degenerates to identity when features are uninformative), \emph{task-conditioned prefix tokens} (quantized feature values and task identity prepended to every input), and \emph{Instance-Weighted Normalization} (IWN; removes class-prior bias from gate statistics). We prove an excess-risk bound linking gate benefit to \emph{surgical feature alignment}. Across four tasks, SST-2, multi-hop retrieval, LLM-prompt attribution, and authorship detection, cov
    
[^304]: LangMAP：一种语言自适应的分词方法

    LangMAP: A Language-Adaptive Approach to Tokenization

    [https://arxiv.org/abs/2606.23566](https://arxiv.org/abs/2606.23566)

    提出 LangMAP 分词方法，通过在共享词表上为每种语言建立独立分布，将 UnigramLM 扩展到多语言场景，既能从头训练多语言模型，也能在不改变词表的前提下将预训练分词器适配到特定语言，且推理时无需语言标签。

    

    特定语言的分词器能够提升分词质量以及模型在这些语言上的下游任务性能。要在现有预训练模型上使用这样的分词器，要么需要从头训练一个新模型，要么需要调整预训练模型的词表。我们提出了语言自适应最大后验概率分词方法（LangMAP），该方法将 UnigramLM 算法扩展到多语言场景，通过将其单一的词表分布替换为同一词表上针对每种语言的各自分布。LangMAP 既可用于从头训练多语言语言模型，也可以在不改变词表的情况下，将预训练模型的分词器适配到各个语言。估计每种语言的分布需要语言标签；而在推理时，该算法会在各语言间选择概率最高的切分方式，即无需语言标签。在将预训练分词器进行适配的实验中……（摘要原文在此处截断）

    arXiv:2606.23566v3 Announce Type: replace  Abstract: Language-specific tokenizers improve tokenization quality and the downstream performance of models on those languages. Using one with an existing pretrained model requires either training a new model from scratch or adapting the vocabulary of the pretrained one. We propose Language-adaptive Maximum a Posteriori (LangMAP) Tokenization, which extends the UnigramLM algorithm to the multilingual setting by replacing its single distribution over the vocabulary with one distribution per language over the same vocabulary. LangMAP can be used to train a multilingual language model from scratch, or to adapt a pretrained model's tokenizer to individual languages without changing its vocabulary. Language labels are required to estimate the per-language distributions; at inference, the algorithm selects the highest-probability segmentation across languages, i.e., it does not need a language label. In experiments where we adapt a pretrained token
    
[^305]: 迈向根记忆：面向个性化大语言模型的隐式逻辑记忆检索的基准测试与增强

    Towards Root Memory: Benchmarking and Enhancing Implicit Logical Memory Retrieval for Personalized LLMs

    [https://arxiv.org/abs/2606.23283](https://arxiv.org/abs/2606.23283)

    该论文揭示了记忆检索中的“语义偏转”问题——基于语义相似度的检索会遗漏逻辑上关键的记忆，并构建IMLogic基准来诊断该问题，同时提出“根记忆”方法，通过LLM推理从用户历史中提炼可复用的逻辑并锚定证据，从而增强个性化大语言模型的隐式逻辑记忆检索能力。

    

    记忆系统对于个性化大语言模型（LLMs）至关重要。然而，这些系统中的检索主要依赖语义相似度，可能会遗漏那些语义重叠有限但在逻辑上至关重要的记忆。我们将这种失效称为“语义偏转”，而现有基准测试对其评估仍不充分。为填补这一空白，我们构建了IMLogic，一个面向长对话中隐式逻辑记忆检索的基准测试。每个问题均围绕语义相关性与逻辑相关性之间的分歧而设计，并配有诊断性回答选项，用以探测语义偏转、幻觉以及个性化缺失等问题。为了帮助系统利用那些难以通过语义匹配发现的关联，我们引入了“根记忆”，这是一种结构化表示，通过LLM推理从用户历史中提炼出可复用的、指导回应的逻辑，并将其锚定在支持性证据之上。

    arXiv:2606.23283v2 Announce Type: replace  Abstract: Memory systems are essential for personalized Large Language Models (LLMs). However, retrieval within these systems primarily relies on semantic similarity, potentially missing logically critical memories with limited semantic overlap. We refer to this failure as semantic deflection, which remains insufficiently evaluated by existing benchmarks. To address this gap, we construct IMLogic, a benchmark for implicit logical memory retrieval in long dialogues. Each question is designed around the divergence between semantic and logical relevance, with diagnostic response options that probe semantic deflection, hallucination, and lack of personalization. To help systems utilize associations that are difficult to discover through semantic matching, we introduce root memory, a structured representation that distills reusable, response-guiding logic from user histories through LLM reasoning and grounds it in supporting evidence. We further de
    
[^306]: 作为协调控制的领导力：多智能体LLM团队中的行为特征与恢复优势边界

    Leadership as Coordination Control: Behavioral Signatures and the Recovery-Advantage Boundary in Multi-Agent LLM Teams

    [https://arxiv.org/abs/2606.19111](https://arxiv.org/abs/2606.19111)

    该研究将交易型、变革型、情境型三种经典领导风格操作化为多智能体LLM团队中的过程级协调控制器，并通过行为特征测量发现：没有任何控制器在准确率上全面占优，其价值取决于特定条件，且从错误共识中恢复的优势存在明确边界。

    

    团队科学认为领导力是权变的：它只在特定条件下才发挥作用。我们对多智能体LLM团队提出同样的问题：在哪些可测量的条件下，过程级协调控制能够带来价值？这些条件是否符合团队科学的预测？这需要的是测量，而不仅仅是准确率：我们使用行为特征（多数锁定、探索、从错误的第0轮共识中恢复）以及逐动作消融实验，由于每个控制器都是显式的动作集合，因此实验设计是干净的。我们将三种经典领导风格（交易型、变革型、情境型）操作化为共享动作词汇表（探索、修改、接受、综合）之上的控制器。一个使用相同动作但规则随机的匹配控制器，其恢复表现并不优于多数投票。在四种任务情境和三个开源权重模型家族上，没有任何控制器在准确率上占据主导地位。交易型控制……（摘要此处被截断）

    arXiv:2606.19111v2 Announce Type: replace-cross  Abstract: Team science holds that leadership is contingent: it helps only under specific conditions. We ask the same question of multi-agent LLM teams: under what measurable conditions does process-level coordination control add value, and do those conditions match what team science predicts? This takes measurement, not just accuracy: we use behavioral signatures (majority lock-in, exploration, recovery from an incorrect round-0 consensus) and per-action ablations, clean because each controller is an explicit action set. We operationalize three classical leadership styles (transactional, transformational, situational) as controllers over a shared action vocabulary (explore, revise, accept, synthesize). A matched controller with the same actions but an arbitrary rule recovers no better than majority voting. Across four task regimes and three open-weight model families, no controller dominates by accuracy. Transactional control matches a s
    
[^307]: 深海之韵：从点击声波形到二阶序列依赖的抹香鲸尾音两级声学组织

    Rhythm of the Deep: Two-tier acoustic organization of sperm-whale codas from click waveforms to second-order sequence dependence

    [https://arxiv.org/abs/2606.16084](https://arxiv.org/abs/2606.16084)

    该研究发现抹香鲸尾音的声学组织具有两级嵌套结构：尾音内部反复出现的点击声波形差异与点击间隔时间互补，而尾音之间反复出现的整体尾音形式本身携带序列依赖性。

    

    抹香鲸尾音（codas）通常以点击声数量和点击间隔（ICI）来刻画，但其组成点击声波形中反复出现的差异一直未能得到解释。本研究检验了声学组织是否在两个尺度上呈嵌套结构：在尾音内部，反复出现的点击声波形差异可能与ICI时间信息互补；在尾音之间，反复出现的整体尾音形式本身可能携带序列依赖性。研究从1,483个尾音中识别出候选的反复出现的点击声和整体尾音分组，且未预先指定波形类别，随后通过原始采样率的频谱/包络测量、精确的干扰因素匹配、留出的时序对比以及序列控制进行评估。在第一层级，在匹配的日期、社会单元、个体和采样率层级内，反复出现的点击声波形组在频谱斜率、带宽、平坦度、高/低频带能量和包络结构上均存在差异。其组成增加了关于……（摘要原文在此处截断）

    arXiv:2606.16084v3 Announce Type: replace  Abstract: Sperm-whale codas are conventionally characterized by click count and inter-click intervals (ICIs), leaving recurring differences in constituent click waveforms unresolved. This study tests whether acoustic organization is nested across two scales: within codas, where recurring click-waveform differences may complement ICI timing, and across codas, where recurring whole-coda forms may themselves carry sequence dependence. Candidate recurring click and whole-coda groupings were identified from 1,483 codas without prespecifying waveform categories, then evaluated with native-rate spectral/envelope measurements, exact nuisance matching, held-out timing contrasts, and sequence controls. At the first tier, recurring click-waveform groups differed in spectral slope, bandwidth, flatness, high/low-band energy, and envelope structure within matched date, social unit, individual, and sample-rate strata. Their composition added information abou
    
[^308]: 校准分诊，而非自主决策：医学视觉语言模型的置信度估计

    Calibrated Triage, Not Autonomy: Confidence Estimation for Medical Vision-Language Models

    [https://arxiv.org/abs/2606.15910](https://arxiv.org/abs/2606.15910)

    本文提出通过评估九种置信度估计器，证明在医学视觉语言模型中，校准分诊比自主决策更可靠，并指出仅用自然图像训练的置信度探针能有效决定模型何时应放弃回答。

    

    一个视觉语言模型可以流畅且自信地回答关于胸部X光片或病理切片的问题，却几乎不使用图像信息，而是依赖语言先验。在医学领域，这是最关键的失败模式：答案看似可信实则不然，而自然的防护措施是获得足够可靠的置信度分数，以判断模型何时应放弃回答。我们提出的不是准确性问题，而是部署问题：视觉语言模型能安全地独立处理多少医学影像工作，以及哪种置信度信号能实现这一点。我们评估了九种置信度估计器，涵盖无需训练的logit基线、基于提示的自我报告以及训练过的内部探针，在五个开放权重的大型视觉语言模型和三个覆盖广泛临床影像、放射学和病理学的医学视觉问答数据集上进行测试，所有探针仅使用自然图像训练并直接应用于医学领域，无需适应。研究被重构为有界选择性预测。

    arXiv:2606.15910v2 Announce Type: replace  Abstract: A vision-language model can answer a question about a chest radiograph or a pathology slide fluently and confidently while barely using the image, relying instead on language priors. In medicine this is the failure that matters most: the answer looks trustworthy and is not, and the natural safeguard is a confidence score reliable enough to say when the model should abstain. We ask a deployment question rather than an accuracy one: how much medical imaging work a vision-language model can safely defer on its own, and which confidence signal makes that possible. We evaluate nine confidence estimators, spanning training-free logit baselines, prompt-based self-reports, and trained internal probes, across five open-weight LVLMs and three medical VQA datasets covering broad clinical imaging, radiology, and pathology, every probe trained only on natural images and applied to medicine without adaptation. Recast as bounded selective predictio
    
[^309]: Gefen：优化的随机优化器

    Gefen: Optimized Stochastic Optimizer

    [https://arxiv.org/abs/2606.13894](https://arxiv.org/abs/2606.13894)

    Gefen 通过在参数块间自动共享二阶矩估计并用学习到的码本量化一阶矩，将 AdamW 的优化器内存占用最多降低 8 倍（每十亿参数节省 6.5 GiB），同时从理论上解释了二阶矩共享有效的原因。

    

    AdamW 是深度学习中的默认优化器，但其矩状态会为训练内存增加两个与参数规模相同的缓冲区，从而推高大模型预训练的成本。我们提出 Gefen，一种内存高效的优化器，它能在参数块之间自动共享二阶矩估计，并利用学习到的码本对一阶矩进行量化。Gefen 在保持性能的同时，将 AdamW 的优化器内存占用最多降低 8 倍，每十亿参数可节省 6.5 GiB 内存。先前的工作沿着 Hessian 矩阵的块对角结构对参数分组并共享二阶矩，但依赖人工指定的架构规则，且未能解释这种分组为何有效。我们证明了较大的 Hessian 非对角（混合）元素会将平方梯度之比约束到趋近于 1，从而解释了当共享二阶矩所汇聚的平方梯度彼此相近时，这些共享估计为何依然准确。这一方法无需实际计算 Hessian 矩阵：其块结构是继承而来的。

    arXiv:2606.13894v4 Announce Type: replace-cross  Abstract: AdamW is a default optimizer for deep learning, but its moment states add two parameter-sized buffers to training memory, increasing the cost of large-scale pretraining. We propose Gefen, a memory-efficient optimizer that automatically shares second-moment estimates across parameter blocks and quantizes the first moment using a learned codebook. Gefen reduces AdamW's optimizer memory footprint by up to 8x while maintaining performance, saving 6.5 GiB per billion parameters. Prior work shares second moments across parameters grouped along the Hessian's block-diagonal structure, but relies on hand-specified architectural rules and leaves unexplained why such grouping works. We prove that large mixed Hessian entries constrain the ratio of squared gradients toward one, explaining why shared second moments are accurate when the squared gradients they pool are similar. The Hessian need not be computed: its block structure is inherite
    
[^310]: TellOnce：一次纠正编程智能体，改善其未来行为

    TellOnce: Correcting Coding Agents Once, Improving Their Future Actions

    [https://arxiv.org/abs/2606.13174](https://arxiv.org/abs/2606.13174)

    TellOnce是一个无需更新底层模型的即插即用个性化层，能将用户对智能体的对话式纠正转化为持久、有作用范围且可修订的行为规则，使智能体在未来的任务中持续改善行为、遵守用户偏好。

    

    交互式LLM智能体越来越多地跨任务与用户协作，然而记住先前的纠正并不能确保智能体在未来改变其行为。在一个基于匿名化真实智能体交互构建的诊断测试中，使用Mem0的智能体仍然违反了57.5%的适用偏好检查。我们提出了TellOnce，一个即插即用的个性化层，它能够将对话式纠正转化为持久、有作用范围且可修订的行为规则，而无需更新底层的智能体模型。TellOnce识别哪些要求应当持久化、确定它们何时适用、将后续反馈与现有规则进行调和，并在可靠验证可行时准备可复用的检查项。我们通过冻结学习到的状态并在后续任务上测试（无需额外的用户纠正），来评估所学到的纠正能否改善智能体的未来行为。在ClawArena和MemoryArena上对三个模型的评估中，TellOnce实现了……（原文摘要在此处截断）

    arXiv:2606.13174v2 Announce Type: replace-cross  Abstract: Interactive LLM agents increasingly work with users across tasks, yet remembering a previous correction does not ensure that the agent changes its behavior in the future. In a diagnostic constructed from anonymized real-world agent interactions, agents using Mem0 still violate 57.5% of applicable preference checks. We introduce TellOnce, a drop-in personalization layer that turns conversational corrections into persistent, scoped, and revisable behavioral rules without updating the underlying agent model. TellOnce identifies which requirements should persist, determines when they apply, reconciles subsequent feedback with existing rules, and prepares reusable checks when reliable verification is possible. We evaluate whether learned corrections improve future behavior by freezing the learned state and testing on subsequent tasks without additional user corrections. Across three models on ClawArena and MemoryArena, TellOnce achi
    
[^311]: 可观测模式并非解释：潜在推理模型的因果-几何分析

    Observable Patterns Are Not Explanations: A Causal-Geometric Analysis of Latent Reasoning Models

    [https://arxiv.org/abs/2606.12689](https://arxiv.org/abs/2606.12689)

    该研究通过对照实验与因果-几何分析发现，潜在推理模型中的可观测潜在状态模式并非推理机制的可靠证据——潜在思维的作用是分级的并集中于低维子空间，应被视为“隐藏的计算”而非“隐藏的解释”。

    

    潜在推理模型用连续的潜在思维取代了显式的思维链。近期研究将可观测的潜在状态模式——例如类广度优先搜索（BFS）的前沿结构和可解码的算术计算——视为内部推理机制存在的证据。通过将两个潜在推理模型（Coconut和CODI）与缺乏所提循环结构或课程训练的对照模型进行对比评估，我们发现这些模式在对照模型中同样会出现，且并不总是影响模型行为。因果干预实验揭示，潜在思维的利用并非二元存在，而是分级的，其程度随该思维对模型行为的因果效应而变化。几何分析进一步表明，这种效应集中在低维子空间中，且这些子空间的逐步几何结构随其行为影响力的增强而变得更有组织性。因此，潜在思维应被视为隐藏的计算，而非隐藏的解释：仅凭可解码性、注意力或静态结构本身，无法确证潜在推理模型的推理机制。

    arXiv:2606.12689v2 Announce Type: replace  Abstract: Latent reasoning models (LRMs) replace explicit chain-of-thought with continuous thoughts. Recent work treats observable latent-state patterns, such as BFS-like frontiers and decodable arithmetic computation, as evidence for internal reasoning mechanisms. Evaluating two LRMs (Coconut and CODI) against controls lacking the proposed recurrence or curriculum, we find these patterns also appear in the controls and do not consistently affect behavior. Causal interventions reveal that latent-thought utilization is not binary but graded, scaling with a thought's causal effect on model behavior. Geometric analyses reveal this effect concentrates in low-dimensional subspaces whose step-wise geometry grows more structured as their behavioral influence increases. Latent thoughts should therefore be treated as hidden computation, not hidden explanation: decodability, attention, or static structure alone cannot establish LRM reasoning mechanisms;
    
[^312]: 虚拟后门作为防御手段：通过共享内部机制消除生成式大语言模型中的未知后门

    Dummy Backdoor as a Defense: Removing Unknown Backdoors via Shared Internal Mechanisms for Generative LLMs

    [https://arxiv.org/abs/2606.11648](https://arxiv.org/abs/2606.11648)

    该论文利用相同攻击目标的后门会引发相似内部激活变化这一发现，提出先植入已知触发器的虚拟后门、再通过虚拟触发器输入与干净响应配对微调将其消除的方法，从而在无需了解攻击类型的情况下移除生成式大语言模型中的未知后门。

    

    后门攻击对大语言模型（LLM）的安全性和可靠性构成严重威胁，因为此类攻击使模型在干净输入上表现正常，而当隐藏触发器存在时则会产生攻击者指定的响应。当防御者不了解攻击类型或后门训练期间形成的内部机制时，移除此类未知后门尤其具有挑战性。在这项工作中，我们提出了一种简单而有效的后门移除方法，该方法基于不同后门之间共享的内部机制。首先，我们证明了具有相同任务（攻击目标）的不同后门会在其内部激活中引发相似的触发器激活变化。受这一观察结果的启发，我们提出的方法有意植入一个具有已知触发器的后门（即虚拟后门），然后通过在虚拟触发器输入与干净响应配对的数据上进行进一步微调来将其移除。由于虚拟后门与未知后门……（原文摘要在此处被截断）

    arXiv:2606.11648v2 Announce Type: replace-cross  Abstract: Backdoor attacks pose a serious threat to the safety and reliability of Large Language Models (LLMs), as they cause models to behave normally on clean inputs while producing attacker-specified responses when hidden triggers are present. Removing such unknown backdoors is particularly challenging when the defender does not know the attack types or the internal mechanisms formed during backdoor training. In this work, we propose a simple but effective backdoor removal method based on internal mechanisms shared across different backdoors. First, we show that different backdoors with the same task (attack objective) induce similar trigger-activated changes in their internal activations. Motivated by this observation, our proposed method intentionally embeds a backdoor with a known trigger dummy backdoor and then removes it through further fine-tuning on dummy-triggered inputs paired with clean responses. Because the dummy and unkno
    
[^313]: 检测差异不等于理解结构：大语言模型在图同构问题上失败

    Detecting Differences Is Not Understanding Structure: Large Language Models Fail at Graph Isomorphism

    [https://arxiv.org/abs/2606.09484](https://arxiv.org/abs/2606.09484)

    该研究揭示大语言模型虽然在图同构检测上表现近乎完美，但在节点标签被置换时却无法识别相同图的同构关系，证明其成功依赖于模式利用而非对抽象图结构的真正理解，因此图推理基准测试的成绩不能作为拓扑理解能力的证据。

    

    大语言模型（LLMs）在多种推理任务上展现出令人印象深刻的表现，但它们在图结构推理方面的能力仍不明确。我们研究了LLMs能否真正理解图同构——图论中的一个基本问题。虽然LLMs在同构检测任务上达到了近乎完美的准确率，但我们证明这种表现是虚假的。当相同的图以置换节点标签的方式呈现时，LLMs无法识别它们的同构关系。这一发现表明，LLMs利用的是模式而非对抽象图结构的推理。由于置换不变性是有效结构推理的基本要求，这些结果表明，在图推理基准测试上的成功不应被解读为真正拓扑理解的证据。

    arXiv:2606.09484v2 Announce Type: replace  Abstract: Large language models (LLMs) have shown impressive performance on diverse reasoning tasks, yet their capacity for structural reasoning in graphs remains unclear. We investigate whether LLMs can genuinely understand graph isomorphism -a fundamental problem in graph theory. While LLMs achieve near-perfect accuracy on isomorphism detection, we show this performance is illusory. When identical graphs are presented with permuted node labels, LLMs fail to identify their isomorphism. This finding suggests that LLMs exploit patterns rather than reasoning about abstract graph structure. Since permutation invariance is a fundamental requirement for valid structural reasoning, these results indicate that success on graph reasoning benchmarks should not be interpreted as evidence of genuine topological understanding.
    
[^314]: PriFT：利用预训练先验指导有监督微调

    PriFT: Guiding Supervised Fine-Tuning with Pretrained Priors

    [https://arxiv.org/abs/2606.09396](https://arxiv.org/abs/2606.09396)

    提出 PriFT 方法，以冻结的预训练模型先验而非正在微调的模型来计算 token 训练权重，避免权重与优化轨迹纠缠所引发的自我强化循环，从而提升有监督微调的泛化能力。

    

    有监督微调（SFT）是一种高效的下游任务适配方法，并常作为强化学习（RL）的初始化阶段，但其泛化能力可能弱于强化学习。一个关键局限在于其离策略目标：SFT 逐 token 地拟合固定的示范数据，其中包含与模型预训练分布不一致的目标，容易导致过拟合。近期一系列工作试图解决这一问题，其做法是为与当前模型预测分布更一致的 token 分配更大的训练权重，其直觉是拟合这些 token 对模型已有知识和表征的破坏较小。然而，若从正在被微调的模型本身计算 token 权重，会使 token 权重与优化轨迹相互纠缠，形成自我强化的循环，使监督信号集中到模型最初偏好的狭窄 token 集合上。（注：原文摘要至此处被截断，后续内容未能提供。）

    arXiv:2606.09396v2 Announce Type: replace  Abstract: Supervised fine-tuning (SFT) is an efficient approach for downstream task adaptation and often serves as the initialization stage for reinforcement learning (RL), but it can show weaker generalization than RL. A key limitation is its off-policy objective: SFT fits fixed demonstrations token by token, including targets poorly aligned with the model's pretrained distribution, which can lead to overfitting. A recent line of work addresses this issue by assigning larger training weights to tokens better aligned with the current model's predictive distribution, with the intuition that fitting these tokens are less distortive to the model's knowledge and representations. However, computing the token weights from the model that is currently fine-tuned entangles token weights with the optimization trajectory, inducing a self-reinforcing loop which concentrates the supervision signal to a narrow set of initially preferred tokens. To address t
    
[^315]: 智能体的最后考试

    Agents' Last Exam

    [https://arxiv.org/abs/2606.05405](https://arxiv.org/abs/2606.05405)

    本文提出由250多位行业专家共同打造的Agents' Last Exam（ALE）基准，通过13个行业集群、55个子领域、1千多项长周期、结果可验证的真实经济价值任务来评估AI智能体，并发现当前系统在最难任务层级上仍远未饱和。

    

    arXiv:2606.05405v3 公告类型：替换 摘要：近年来的人工智能系统已在各类基准测试中取得了优异的成绩，但这些成果并未转化为在众多专业领域中的经济价值意义上的实际部署。我们认为，这一差距在很大程度上是一个评估问题：广泛使用的基准测试缺乏对真实的、具有经济价值的工作流程的持续性性能测量。本文提出了 Agents' Last Exam（ALE），这是一个旨在评估AI智能体在长周期、具有经济价值、结果可验证的真实世界任务上表现的基准。该基准与250多位行业专家合作开发，参考O*NET / SOC 2018（美国联邦职业分类体系）定义了非实体行业，并围绕一个任务分类体系组织，该体系包含55个子领域、归为13个行业集群，涵盖1千多项任务。当前结果显示，最难的任务层级仍远未达到饱和：在主流的运行框架和底层模型配置下，平均……

    arXiv:2606.05405v3 Announce Type: replace  Abstract: Recent AI systems have achieved strong results on a wide range of benchmarks, yet these gains have not translated into economically meaningful deployment across many professional domains. We argue that this gap is largely an evaluation problem: widely used benchmarks lack sustained performance measurement on real and economically valuable workflows. This paper introduces Agents' Last Exam (ALE), a benchmark designed to evaluate AI agents on long horizon, economically valuable, real world tasks with verifiable outcomes. Developed in collaboration with 250+ industry experts, ALE covers non-physical industries defined with reference to O*NET / SOC 2018 (the U.S. federal occupational taxonomy). It is organized around a task taxonomy with 55 sub fields grouped into 13 industry clusters covering 1K+ tasks. Current results show that the hardest tier remains far from saturated: across mainstream harness and backbone configurations, the avera
    
[^316]: 动态短卷积改进Transformer

    Dynamic Short Convolutions Improve Transformers

    [https://arxiv.org/abs/2606.03825](https://arxiv.org/abs/2606.03825)

    本文提出动态短卷积作为一种新的神经网络原语，利用依赖输入的滤波器在保留卷积局部性偏置的同时提升表达能力，在150M至2B参数的语言建模实验中持续优于标准Transformer及静态短卷积增强版本。

    

    Transformer已成为大型语言模型的主导架构，这主要归功于注意力机制、前馈层、残差连接和归一化所带来的可扩展性与灵活性。本文引入动态短卷积作为一种额外的神经网络原语，用于改进Transformer。与静态短卷积不同，动态卷积使用依赖于输入的滤波器，在增强表达能力的同时保留了卷积的局部性偏置。启发性实验表明，将动态短卷积应用于键、查询和值表示，与静态卷积变体相比，能够在具有挑战性的联想回忆任务上取得更好的性能。在从1.5亿（150M）到20亿（2B）参数规模的语言建模实验中，动态卷积始终优于标准Transformer以及通过静态短卷积增强的Transformer。拟合缩放定律的结果表明可获得1.33倍的……

    arXiv:2606.03825v2 Announce Type: replace-cross  Abstract: Transformers have become the dominant architecture for large language models, largely due to the scalability and flexibility of attention, feed-forward layers, residual connections, and normalization. This paper introduces dynamic short convolutions as an additional neural network primitive for improving Transformers. Unlike static short convolutions, dynamic convolutions use input-dependent filters, which preserves the locality bias of convolution while increasing expressivity. Motivating experiments show that applying dynamic short convolutions to key, query, and value representations improves performance on challenging associative recall tasks compared with static convolutional variants. Across language-modeling experiments ranging from 150M to 2B parameters, dynamic convolutions consistently outperform standard Transformers and Transformers augmented with static short convolutions. Fitting scaling laws indicates a 1.33$\tim
    
[^317]: 研究与缓解大语言模型交互中的危害放大问题

    Investigating and Alleviating Harm Amplification in LLM Interactions

    [https://arxiv.org/abs/2606.02423](https://arxiv.org/abs/2606.02423)

    该论文针对大语言模型在多轮对话中逐步放大危害的问题，提出了涵盖十二个风险类别的HarmAmp基准，并设计了能预判有害交互轨迹并主动干预的监控器TrajSafe。

    

    大语言模型（LLM）可以充当有用的助手，但它们同样可能成为危害放大器，使恶意用户通过持续的交互获得超出其自身能力范围的有害结果。这种风险体现在两个维度上：一是领域专长的民主化，使新手也能生成专业性的有害内容；二是有害操作的规模化，其规模是人工操作无法企及的。然而，现有研究往往忽视了大语言模型如何在多轮对话中逐步累积危害。我们提出了HarmAmp，一个针对多轮危害放大场景的新基准，涵盖十二个风险类别。每个场景都植根于现实世界的威胁，并满足严格的标准，即实质性放大、操作具体性和多轮必要性。我们进一步提出了TrajSafe，一种主动式监控器，能够预判有害的交互轨迹，并通过诸如探询用户等动作进行干预。

    arXiv:2606.02423v2 Announce Type: replace  Abstract: Large language models (LLMs) can serve as helpful assistants, yet they can equally function as harm amplifiers that enable malicious users to achieve harmful outcomes beyond their capabilities through extended interactions. This risk manifests along two axes, i.e., democratizing domain expertise that allows novices to produce specialized harmful content, and scaling harmful operations at volumes that manual effort cannot match. Existing works, however, often overlook how LLMs compound harm across multi-turn conversations. We introduce HarmAmp, a new benchmark for multi-turn harm amplification scenarios spanning twelve risk categories. Each scenario is grounded in real-world threats and satisfies rigorous criteria, i.e., substantive amplification, operational specificity, and multi-turn necessity. We further propose TrajSafe, a proactive monitor that anticipates harmful trajectories and intervenes through actions such as probing users
    
[^318]: 面向大语言模型智能体的统一上下文演化

    Unified Context Evolution for LLM Agents

    [https://arxiv.org/abs/2606.02304](https://arxiv.org/abs/2606.02304)

    提出了统一上下文演化（UCE）框架，无需梯度训练即可将智能体经验分解为记忆、策略、工作流和技能四种类型的可演化上下文单元（ECU），通过生成、检索、使用评分与修剪机制构建跨任务持续演化的经验库，解决了智能体每轮任务从固定上下文重启、有用策略随任务结束而丢失的问题。

    

    基于大语言模型的智能体可以通过将推理与环境反馈相结合来解决多步交互任务，然而每一轮任务都从相同的固定上下文开始，任务过程中发现的任何有用策略在任务结束后都会丢失。现有方法要么将学习限制在当前任务范围内，要么将所有经验汇集到一个单一的无类型存储库中，既没有区分知识类型，也没有在使用过程中追踪质量，更无法平衡知识库中仍然缺失的内容。我们提出了统一上下文演化（UCE），这是一个无需梯度训练的框架，它将智能体经验外化为一个不断演化的、具有类型标签的可演化上下文单元（ECU）库。UCE 将经验分解为四种互补类型（记忆、策略、工作流和技能），每种类型在类型特定条件下从轨迹中生成，在决策时被检索，通过反复使用的结果进行评分，并在不再有价值时被修剪。一个调度模块为每个周期的生成……（原文摘要到此截断）

    arXiv:2606.02304v2 Announce Type: replace  Abstract: LLM-based agents can solve multi-step interactive tasks by combining reasoning with environment feedback, yet each episode starts from the same fixed context and any useful strategy discovered along the way is lost once the task ends. Existing approaches either limit learning to the current task or pool all experience into a single untyped store, without distinguishing knowledge types, tracking quality through use, or balancing what the library still lacks. We introduce Unified Context Evolution (UCE), a gradient-free framework that externalizes agent experience into an evolving library of typed Evolvable Context Units (ECUs). UCE decomposes experience into four complementary types (Memory, Strategy, Workflow, and Skill), each generated from trajectories under type-specific conditions, retrieved at decision time, scored through repeated usage outcomes, and pruned when no longer valuable. A scheduling module allocates each cycle's gen
    
[^319]: “我强烈怀疑这个网站是个骗局”：在无防御条件下对自主网页代理的个人身份信息泄露与检测进行基准测试

    "I Strongly Suspect This Website Is a Scam": Benchmarking PII Leakage and Detection without Defense in Autonomous Web Agents

    [https://arxiv.org/abs/2606.00497](https://arxiv.org/abs/2606.00497)

    本文提出Scammer4U基准测试，揭示社会工程学攻击可使前沿自主网页代理在无隐私指导的情况下泄露54–93%的关键个人身份信息，而良性网站上泄露率为0%，凸显了已部署智能体系统面临的严重安全风险。

    

    互联网上广泛存在的欺骗性网页内容（通常被称为社会工程学攻击）会操纵自主网页代理，使其将用户的个人身份信息（PII）提交给攻击者控制的端点。在本文中，我们证明了社会工程学攻击在从前沿网页代理中窃取关键级PII方面非常有效，对已部署的智能体系统构成严重风险。为了量化这一风险，我们引入了Scammer4U，这是一个预先注册的基准测试，包含91个攻击者控制的环境和10个良性孪生基线，涵盖8种攻击向量和16个网站类别，并采用8轴因子分类法来分离各个攻击设计因素的因果贡献。在前沿代理上，我们发现没有隐私指导的情况下，关键级PII的泄露率达到54–93%，而在良性孪生基线上泄露率为0%，证实泄露是由攻击所引致的。

    arXiv:2606.00497v2 Announce Type: replace-cross  Abstract: Deceptive web content, widely instantiated across the internet and commonly known as \textit{social-engineering attacks}, manipulates autonomous web agents into submitting users' personally identifiable information (PII) to attacker-controlled endpoints. In this paper, we show that social-engineering attacks are highly effective at extracting critical-tier PII from frontier web agents, posing a severe risk to deployed agentic systems. To quantify this risk, we introduce \textbf{\textsc{Scammer4U}}, a pre-registered benchmark of 91 attacker-controlled environments and 10 benign-twin baselines, spanning 8 attack vectors and 16 site categories on an 8-axis factorial taxonomy that isolates the causal contribution of individual attack design factors. Across frontier agents, we find that critical-tier PII leakage reaches 54--93\% under no privacy guidance, compared to 0\% on benign-twin baselines, confirming that leakage is attack-at
    
[^320]: 提示词扰“AI”动（PerturbAItions）：大语言模型响应不一致性的机制

    Prompt PerturbAItions: Mechanisms of LLM Response Inconsistencies

    [https://arxiv.org/abs/2605.30913](https://arxiv.org/abs/2605.30913)

    该论文发现提示词的词汇与语气扰动（从礼貌到毒性措辞）会显著降低大语言模型的事实准确性并增加预测不确定性，且毒性越强准确率越低，归因图分析进一步揭示了这种不一致源于模型内部对扰动敏感的变体节点被相对放大。

    

    大语言模型（LLM）越来越多地被部署在对话场景中，用户的措辞从礼貌、随意到带有毒性不等，然而对于语义等价的提示词中这类词汇变化是否会降低模型的事实可靠性，目前所知甚少。我们研究了词汇层面和语气层面的提示词扰动如何影响大语言模型的事实可靠性。我们使用涵盖礼貌、随机以及毒性逐渐增强等多个级别的受控提示词变体，在 ARC-Easy、GSM8K 和 MMLU 三个基准上评估了七个大语言模型。我们发现，词汇扰动会降低大多数模型的事实准确性，并增加其预测的不确定性。有趣的是，毒性程度的增加会导致准确率下降。为了考察这些答案不一致性是否对应于模型内部的变化，我们对模型激活和影响进行了归因图分析。我们发现，扰动会相对于核心推理……放大对扰动敏感的变体节点（摘要原文在此处截断）。

    arXiv:2605.30913v2 Announce Type: replace-cross  Abstract: Large language models (LLMs) are increasingly deployed in conversational settings where user phrasing ranges from polite to casual to toxic, yet less is known about whether such lexical variation in otherwise semantically equivalent prompts can degrade factual reliability. We study how lexical and tone-based prompt perturbations affect the factual reliability of LLMs. Using controlled prompt variations spanning polite, random, and increasingly toxic levels, we evaluate seven LLMs on ARC-Easy, GSM8K, and MMLU. We find that lexical perturbations reduce factual accuracy and increase predictive uncertainty in most models. Interestingly, increasing toxicity results in decreasing accuracy. To examine whether these answer inconsistencies correspond to internal changes, we conduct attribution-graph analyses of model activations and influences. We find that perturbations amplify perturbation-sensitive variant nodes relative to core reas
    
[^321]: 基于时空潜在动力学的大语言模型忠实推理过程学习

    Learning Faithful Reasoning Processes with Spatio-Temporal Latent Dynamics for Large Language Models

    [https://arxiv.org/abs/2605.26893](https://arxiv.org/abs/2605.26893)

    提出GeoFaith框架，将最终状态潜在不确定性与预测熵动力学作为过程正则化信号融入GRPO，无需过程奖励模型即可学习忠实且稳定的推理过程。

    

    思维链推理已显著增强了大语言模型的推理能力。然而，现有的强化学习方法主要优化结果层面的奖励，对推理过程的监督有限。因此，即使最终答案正确，模型也可能产生倾向于走捷径或不稳定的推理轨迹。我们提出了GeoFaith，一个无需过程奖励模型即可学习忠实推理过程的时空潜在动力学框架。GeoFaith结合了最终状态潜在不确定性——用于衡量已完成的轨迹是否位于支撑良好的流形区域——与预测熵动力学——用于惩罚过早坍缩、振荡和持续不确定等异常不确定性模式。这些信号作为过程正则化被纳入GRPO中，联合优化答案正确性与推理质量。

    arXiv:2605.26893v2 Announce Type: replace-cross  Abstract: Chain-of-Thought (CoT) reasoning has significantly enhanced the reasoning capabilities of large language models (LLMs). However, existing reinforcement learning methods primarily optimize outcome-level rewards, providing limited supervision over the reasoning process. As a result, models may produce shortcut-prone or unstable reasoning trajectories even when the final answer is correct. We propose GeoFaith, a spatio-temporal latent dynamics framework for learning faithful reasoning processes without process reward models. GeoFaith combines final-state latent uncertainty, which measures whether a completed trajectory lies in a well-supported manifold region, with predictive-entropy dynamics, which penalize abnormal uncertainty patterns such as premature collapse, oscillation, and prolonged uncertainty. These signals are incorporated as process regularization into GRPO, jointly optimizing answer correctness and reasoning quality 
    
[^322]: HyLaT：通过混合潜空间-文本协议实现高效的多智能体通信

    HyLaT: Efficient Multi-Agent Communication via Hybrid Latent-Text Protocol

    [https://arxiv.org/abs/2605.25421](https://arxiv.org/abs/2605.25421)

    HyLaT提出了一种混合潜空间-文本通信协议，通过潜空间通道高效传输认知信号、用自然语言传递关键信息以保持可解释性，并配合两阶段训练框架，在多轮多智能体交互中显著降低了通信开销。

    

    通信协议设计是基于大语言模型的多智能体系统中的核心挑战。现有的单通道方法面临固有的通信三难困境：基于文本的方法具有可解释性但冗长，而潜空间方法高效但不透明，且仅限于单向工作流。受多通道通信理论启发，我们提出了HyLaT——一种混合潜空间-文本通信协议，它通过潜空间通道传输精细的认知信号以提高效率，同时用自然语言表达简洁的关键信号以保持可解释性和精确性。我们引入了一个两阶段训练框架，结合单智能体混合生成学习与多智能体交互协同训练，使智能体能够在多轮交互中生成和解释混合消息。实验表明，HyLaT在保持……的同时显著降低了通信开销（摘要原文在此处截断）。

    arXiv:2605.25421v2 Announce Type: replace  Abstract: Communication protocol design is a central challenge in large language model-based multi-agent systems. Existing single-channel approaches face an inherent communication trilemma: text-based methods are interpretable but verbose, while latent-space methods are efficient but opaque and limited to unidirectional workflows. Inspired by multi-channel communication theory, we propose HyLaT, a hybrid latent-text communication protocol that transmits elaborate cognitive signals through a latent channel for efficiency, while expressing concise critical signals in natural language to preserve interpretability and precision. We introduce a two-stage training framework combining single-agent hybrid generation learning and multi-agent interactive co-training, enabling agents to generate and interpret hybrid messages across multiple rounds of interaction. Experiments demonstrate that HyLaT reduces communication overhead significantly while mainta
    
[^323]: QUEST：用完全合成的任务训练前沿深度研究智能体

    QUEST: Training Frontier Deep Research Agents with Fully Synthetic Tasks

    [https://arxiv.org/abs/2605.24218](https://arxiv.org/abs/2605.24218)

    QUEST 是一系列开源通用深度研究智能体模型（2B至35B），其核心创新在于基于统一评分准则树的全合成数据流水线，配合中期训练、监督微调与强化学习的训练方案，使其在事实检索、引用溯源和报告综合等多种长程搜索任务上表现出色。

    

    深度研究智能体将搜索引擎的角色从检索关键词匹配的网页扩展到综合整理知识，从根本上改变了人类与信息交互的方式。然而，前沿系统仍然是闭源专有的，而现有的开源智能体在不同任务类型之间往往泛化能力较差，因此如何训练出一个具备广泛能力的深度研究智能体仍不明确。我们发布了QUEST，这是一系列开源模型（参数规模从2B到35B），作为通用的深度研究智能体，旨在处理多种长程搜索任务，在事实检索、引用溯源和报告综合方面具有强大的能力。为了构建QUEST，我们提出了一种有效的训练方案，将中期训练、监督微调和强化学习相结合。该方案的核心是一条精心设计的数据合成流水线，其基于统一的评分准则树（rubric trees），可适用于不同的任务类型，并能够合成带有可验证……（摘要原文在此处截断）

    arXiv:2605.24218v2 Announce Type: replace  Abstract: Deep research agents extend the role of search engines from retrieving keyword-matched pages to synthesizing knowledge, fundamentally changing how humans interact with information. However, frontier systems remain proprietary, while existing open agents often generalize poorly across different task types, leaving unclear how to train a broadly capable deep research agent. We release QUEST, a family of open models (ranging from 2B to 35B) that serve as general-purpose deep research agents designed to handle a wide range of long-horizon search tasks, with strong capabilities in fact seeking, citation grounding, and report synthesis. To build QUEST, we propose an effective training recipe combining mid-training, supervised fine-tuning, and reinforcement learning. Central to this recipe is a curated data synthesis pipeline based on unified rubric trees, which applies to different task types and enables synthesizing training data with ver
    
[^324]: 社会规范对齐的自然化测量方法

    Naturalistic measure of social norms alignment

    [https://arxiv.org/abs/2605.23420](https://arxiv.org/abs/2605.23420)

    该论文提出了一种通过解决方案匹配在自然、自由形式场景下测量社会规范对齐的新框架，引入两个一致性指标并构建了3000个丹麦语社会困境数据集，可用于测量LLM与人类之间或人与人之间的规范一致性。

    

    社会规范反映了人们对可接受行为的共同期望。测量社会规范对齐仍然具有挑战性，现有方法通常依赖人工设计的封闭式评估，例如多项选择问卷，或衡量与预先定义陈述的一致性。在本研究的语境中，社会规范对齐是指测量针对社会问题或困境的不同解决方案之间的一致性。我们提出了一个框架，通过解决方案匹配的方式，在自然、自由形式的场景中测量社会规范对齐。该框架使我们能够测量任意两个困境回答之间的对齐程度，例如大语言模型（LLM）与人类之间、LLM与LLM之间，或人与人之间。我们引入了两个指标：陈述一致性准确率和显式一致性准确率，并构建了一个包含3000个非平凡丹麦语社会困境的数据集。所有困境均配有由三名小组成员得出的参考解决方案，他们充当具有文化根基的评判者。

    arXiv:2605.23420v2 Announce Type: replace  Abstract: Social norms reflect shared expectations on acceptable behavior. Measuring social norms alignment remains challenging, with existing approaches typically relying on artificial closed-form evaluations such as multiple-choice questionnaires or measuring agreement with predefined statements. In the context of this work, social norms alignment refers to measuring an agreement between solutions with respect to the social problem or dilemma. We propose a framework for measuring social norm alignment in naturalistic, free-form settings through solution matching. The framework enables us to measure alignment between any two dilemma responses e.g., LLMs to a human, LLMs to LLMs, or human to human. We introduce two metrics: stated and explicit agreement accuracy, and construct a dataset of 3k non-trivial social dilemmas in Danish. All dilemmas are assigned reference solutions derived from three panelists, who serve as culturally grounded judge
    
[^325]: 连续扩散在语言建模中与离散扩散具有同等竞争力的扩展性

    Continuous Diffusion Scales Competitively with Discrete Diffusion for Language

    [https://arxiv.org/abs/2605.18530](https://arxiv.org/abs/2605.18530)

    该研究通过将 Plaid 与现代离散扩散语言模型架构对齐构建了 RePlaid，建立了首个可媲美离散 DLM 的连续扩散语言模型缩放定律，证明连续扩散是语言建模中高度竞争且可扩展的方案。

    

    尽管扩散模型近来在语言建模领域引起了广泛关注，但连续扩散的可扩展性似乎不如离散方法。为了挑战这一观点，我们重新审视了基于似然的连续扩散语言模型 Plaid，并通过将 Plaid 的架构与现代离散 DLM 对齐，构建了 RePlaid。在这一统一设置下，我们建立了首个可与离散 DLM 相媲美的连续 DLM 缩放定律：RePlaid 与自回归模型相比仅存在约 $20\times$ 的计算差距，在使用更少参数的情况下优于 Duo，并在过度训练的情形下优于 MDLM。我们将 RePlaid 与近期的连续 DLM 进行了基准测试：在 OpenWebText 数据集上，RePlaid 在连续 DLM 中实现了新的最先进困惑度上界 $22.1$，并具有更优的生成质量。这些结果表明，通过似然训练的连续扩散是一种极具竞争力且可扩展的替代方案。

    arXiv:2605.18530v2 Announce Type: replace  Abstract: While diffusion has drawn considerable recent attention from the language modeling community, continuous diffusion has appeared less scalable than discrete approaches. To challenge this belief we revisit Plaid, a likelihood-based continuous diffusion language model (DLM), and construct RePlaid by aligning the architecture of Plaid with modern discrete DLMs. In this unified setting, we establish the first scaling law for continuous DLMs that rivals discrete DLMs: RePlaid exhibits a compute gap of only $20\times$ compared to autoregressive models, outperforms Duo while using fewer parameters, and outperforms MDLM in the over-trained regime. We benchmark RePlaid against recent continuous DLMs: on OpenWebText, RePlaid achieves a new state-of-the-art PPL bound of $22.1$ among continuous DLMs and superior generation quality. These results suggest that continuous diffusion, when trained via likelihood, is a highly competitive and scalable a
    
[^326]: Firefly：从真实API中点亮大规模可验证工具调用数据生成

    Firefly: Illuminating Large-Scale Verified Tool-Call Data Generation from Real APIs

    [https://arxiv.org/abs/2605.17558](https://arxiv.org/abs/2605.17558)

    FireFly通过先让大模型探索真实API再从观察结果反向合成任务的新范式，从真实MCP服务器生成在构造上保证标签正确性的大规模工具调用训练数据。

    

    训练工具调用智能体需要大规模带有可验证标签的轨迹数据，然而现有方法要么合成偏离真实API行为的环境，要么生成缺乏真实结果以供验证的任务。我们提出了FireFly，一个从真实世界MCP服务器生成可验证工具调用数据的流水线。我们的关键洞察是反转标准的合成流水线：不是先生成任务并期望它们可解，而是先让一个强大的大语言模型沿着图引导的DAG结构探索真实API，然后从观察到的结果反向合成任务，从而在构造上保证标签的正确性。为了应对真实世界工具空间的规模（约1000个工具），我们构建了一个成对工具图并采样子DAG，将探索聚焦在语义连贯的工作流上。为了解决在线API中的环境漂移问题，我们构建了一个检索增强的模拟器，缓存所有探索……

    arXiv:2605.17558v2 Announce Type: replace-cross  Abstract: Training tool-calling agents requires large-scale trajectory data with verifiable labels, yet existing approaches either synthesize environments that diverge from real API behavior or generate tasks without ground-truth outcomes for verification. We present FireFly, a pipeline for generating verified tool-call data from real-world MCP servers. Our key insight is to invert the standard synthesis pipeline: rather than generating tasks and hoping they are solvable, we first let a strong LLM explore real APIs along graph-guided DAG structures, then synthesize tasks backward from observed outcomes, guaranteeing label correctness by construction. To handle the scale of real-world tool spaces (${\sim}$1,000 tools), we build a pairwise tool graph and sample sub-DAGs to focus exploration on semantically coherent workflows. To address environment drift in live APIs, we construct a retrieval-augmented simulator that caches all exploration
    
[^327]: VerifyMAS：面向基于大语言模型的多智能体系统故障归因的假设验证方法

    VerifyMAS: Hypothesis Verification for Failure Attribution in LLM-based Multi-Agent Systems

    [https://arxiv.org/abs/2605.17467](https://arxiv.org/abs/2605.17467)

    提出VerifyMAS框架，将LLM多智能体系统的故障归因转化为在完整交互轨迹上进行故障假设的提出与验证，从而捕捉局部日志方法无法发现的全局性故障并实现细粒度归因。

    

    大语言模型驱动的多智能体系统（LLM-MAS）在复杂任务中表现出色，但不可靠的智能体仍是制约系统级可靠性的关键瓶颈。因此，自动故障归因至关重要。然而，现有方法（如直接预测智能体-错误对、智能体优先的故障归因）依赖于智能体的局部日志，会遗漏那些仅在完整交互轨迹中才显现的全局性故障，例如跨步骤的不一致和智能体间的协调错误。此外，直接预测故障会带来巨大的组合搜索空间，阻碍细粒度的归因。为应对这些挑战，我们提出了VerifyMAS——一个用于智能体故障归因的假设验证框架。VerifyMAS不再直接预测故障智能体和错误类型，而是针对完整轨迹提出并验证故障假设。这种基于验证的方法将归因分解为轨迹层面的……

    arXiv:2605.17467v2 Announce Type: replace  Abstract: Large language model-driven multi-agent systems (LLM-MAS) excel at complex tasks, yet unreliable agents remain a key bottleneck to system-level reliability. Automatic failure attribution is therefore critical, but existing approaches, such as direct prediction of agent-error pairs and agent-first failure attribution, rely on local logs of agents and miss global failures that only manifest over full interaction trajectories, such as cross-step inconsistencies and inter-agent coordination errors. Moreover, directly predicting failures induces a large combinatorial search space, hindering fine-grained attribution. To address these challenges, we propose VerifyMAS, a hypothesis verification framework for agent failure attribution. Instead of directly predicting faulty agents and error types, VerifyMAS formulates and verifies failure hypotheses against full trajectories. This verification-based approach decomposes attribution into traject
    
[^328]: 在何处奖励、奖励什么？理解强化学习中可验证领域与开放式领域之间的泛化

    Where and What to Reward? Understanding Generalization Between Verifiable and Open-Ended Domains in Reinforcement Learning

    [https://arxiv.org/abs/2605.15464](https://arxiv.org/abs/2605.15464)

    该研究通过受控实验分离“提示词来源领域”与“奖励类型”这两个因素，系统揭示了RLVR与RLHF在可验证推理任务与开放式对话任务之间的泛化与迁移规律。

    

    基于可验证奖励的强化学习（RLVR）被广泛用于提升语言模型的推理能力，因为模型生成的答案可以很容易地通过基于规则的验证器进行验证。相比之下，基于人类反馈的强化学习（RLHF）主要用于优化开放式对话，它依赖学习到的奖励模型，因为这类回复不容易验证。RLVR 和 RLHF 通常使用各自领域的数据和训练协议分开训练。这些领域之间的增益如何迁移尚不清楚，部分原因在于以往的比较往往同时改变了模型训练的位置（提示词）和奖励的对象（通过验证器或奖励模型）。我们在一项受控研究中将这两个因素分离开来，在多个基础模型上进行强化学习实验，固定优化器和训练提示词的数量，仅改变提示词的来源领域（数学、多领域、聊天或混合）和奖励类型。我们发现开放式……

    arXiv:2605.15464v2 Announce Type: replace-cross  Abstract: Reinforcement learning with verifiable rewards (RLVR) is widely used to improve the reasoning capability of language models, since model-generated answers can be easily verified by a rule-based verifier. In contrast, reinforcement learning from human feedback (RLHF) is mainly used to optimize open-ended conversation with a learned reward model, because such responses are not easy to verify. RLVR and RLHF are usually trained separately with their own domain data and training protocols. How gains transfer between these domains is unclear, partly because comparisons change both where a model is trained (the prompts) and what is rewarded (by a verifier or a reward model). We separate the two in a controlled study of RL on multiple base models that fixes the optimizer and the number of training prompts, and varies only the source domain of the prompts (math, multi-domain, chat, or mixtures) and the reward. We find that RLHF on open-
    
[^329]: 网页智能体应采用“编译-执行”范式

    Web Agents Should Adopt the Compile-and-Execute Paradigm

    [https://arxiv.org/abs/2605.14290](https://arxiv.org/abs/2605.14290)

    该论文主张网页智能体应放弃 ReAct 这一默认架构，改用“编译-执行”范式——即在观察运行时网页内容之前先确定任务程序再执行，从而防止网页中多来源的不可信内容（如提示注入）篡改智能体的控制流。

    

    ReAct 已成为大语言模型（LLM）智能体的默认架构，许多现有网页智能体也遵循这一范式。我们认为，这对网页智能体而言是错误的默认选择。相反，网页智能体应默认采用“编译-执行”范式：在观察到运行时网页内容之前，先确定一个针对特定任务的程序，然后再执行它。原因在于网页内容混合了来自多方的输入。例如，一个电商产品页面可能同时包含卖家的商品信息、顾客评论、赞助广告以及平台生成的推荐内容。在 ReAct 范式下，所有这些内容在决定下一步动作时都会流入模型，从而为提示注入攻击直接操控智能体的控制流提供了途径。“编译-执行”范式改变了这一边界：不可信的数据或许会影响预定义执行图中的取值或分支，但它无法重新定义用户任务，也无法使模型在运行时合成新的动作。我们分析了 WebArena 这一流行的……（原文摘要至此被截断）

    arXiv:2605.14290v2 Announce Type: replace-cross  Abstract: ReAct has become the default architecture across LLM agents, and many existing web agents follow this paradigm. We argue that it is the wrong default for web agents. Instead, web agents should default to compile-and-execute: commit to a task-specific program before observing runtime web content, then execute it. The reason is that web content mixes inputs from many parties. An e-commerce product page may combine a seller's listing, customer reviews, sponsored advertisements, and platform-generated recommendations. Under ReAct, all of this content flows into the model when deciding on the next action, creating a direct path for prompt injections to steer the agent's control flow. Compile-and-execute changes this boundary: untrusted data may influence values or branches inside a predefined execution graph, but it cannot redefine the user task or cause the model to synthesize new actions at runtime. We analyze WebArena, a popular 
    
[^330]: ReVision：通过时序视觉冗余消减扩展计算机使用智能体

    ReVision: Scaling Computer-Use Agents via Temporal Visual Redundancy Reduction

    [https://arxiv.org/abs/2605.11212](https://arxiv.org/abs/2605.11212)

    ReVision通过学习到的视觉块选择器在连续截图之间去除时序冗余的视觉块，使计算机使用智能体在平均减少46%token用量的同时有效利用历史信息并提升性能。

    

    计算机使用智能体（CUAs）依赖于对图形用户界面的视觉观察，其中每张截图会被编码为大量的视觉token。随着交互轨迹的增长，token成本迅速上升，限制了在固定的上下文和计算预算下能够纳入的历史信息量。这导致与其他领域不同，使用历史信息时性能没有或只有非常有限的提升。我们通过引入ReVision来解决这一低效问题，该方法用于在去除冗余视觉块的轨迹上训练多模态语言模型：它利用一个学习到的视觉块选择器，在保留模型所需空间结构的同时，比较连续截图之间的视觉块表示并移除冗余部分。在OSWorld、WebTailBench和AgentNetBench三个基准测试上，使用Qwen2.5-VL-7B处理包含5张历史截图的轨迹时，ReVision平均减少46%的token使用量，同时（原文摘要在此处截断）

    arXiv:2605.11212v4 Announce Type: replace  Abstract: Computer-use agents (CUAs) rely on visual observations of graphical user interfaces, where each screenshot is encoded into a large number of visual tokens. As interaction trajectories grow, the token cost increases rapidly, limiting the amount of history that can be incorporated under fixed context and compute budgets. This has resulted in no or very limited improvement in the performance when using history unlike other domains. We address this inefficiency by introducing ReVision, which is used to train multimodal language models on trajectories where redundant visual patches are removed using a learned patch selector that compares patch representations across consecutive screenshots while preserving spatial structure required by the model. Across three benchmarks, OSWorld, WebTailBench, and AgentNetBench, when processing trajectories with 5 history screenshots using Qwen2.5-VL-7B, ReVision reduces token usage by 46% on average whil
    
[^331]: 基于干预式时间序列因果发现的数学推理：以大语言模型作为概念掌握模拟器

    Mathematical Reasoning via Intervention-Based Time-Series Causal Discovery Using LLMs as Concept Mastery Simulators

    [https://arxiv.org/abs/2605.07600](https://arxiv.org/abs/2605.07600)

    提出CIKA框架，将大语言模型自身作为干预模拟器，通过把概念状态干预为“已掌握”并测量正确率变化来估计因果效应，从而识别真正在因果上促成正确答案的概念，克服了题目难度等混杂因素造成的虚假关联。

    

    近期改进大语言模型（LLM）数学推理的方法，无论是基于MCTS的测试时搜索还是因果图引导的知识注入，都无法识别哪些概念在因果上真正促成了正确答案，因为观察到的关联可能是虚假的，由题目难度等混杂因素所驱动。我们提出CIKA（面向知识激活的因果干预）框架，该框架将LLM自身用作干预模拟器：通过提示词将概念状态设定为“已掌握”，并以正确率的变化来估计因果效应。我们将这一量度形式化为干预能力探针（ICP），用于诊断LLM能否实际使用某个给定概念——这不同于仅仅拥有相关知识。由于干预以方式外生地设定概念状态、独立于题目难度，ICP能够分离观察性方法无法分离的混杂因素。在筛选出的67道题目上，排名最高概念的ICP……（摘要在此处截断）

    arXiv:2605.07600v2 Announce Type: replace-cross  Abstract: Recent methods for improving LLM mathematical reasoning, whether through MCTS-based test-time search or causal graph-guided knowledge injection, cannot identify which concepts causally contribute to a correct answer, as the observed association may be spurious, driven by confounders such as problem difficulty.   We propose CIKA (Causal Intervention for Knowledge Activation), a framework that uses the LLM itself as an interventional simulator: a prompt sets the concept state to ``mastered'' and the correctness change estimates the causal effect. We formalize this quantity as an Interventional Capability Probe (ICP), which diagnoses whether the LLM can use a given concept -- distinct from merely possessing knowledge. Because the intervention exogenously sets the concept state independently of problem difficulty, ICP separates confounding that observational methods cannot.   On 67 screened problems, the ICP of the top-ranked conce
    
[^332]: LatentRAG：面向高效智能体检索增强生成（RAG）的潜在推理与检索

    LatentRAG: Latent Reasoning and Retrieval for Efficient Agentic RAG

    [https://arxiv.org/abs/2605.06285](https://arxiv.org/abs/2605.06285)

    LatentRAG提出将智能体RAG的推理与检索从离散语言空间转移到连续潜在空间，通过单次前向传播直接从隐藏状态生成潜在token来替代逐token生成自然语言思考和子查询，从而显著降低延迟。

    

    单步检索增强生成（RAG）为简单问答任务提供了一种高效引入外部信息的方式，但在应对复杂问题时表现不佳。智能体RAG（Agentic RAG）通过将单步检索扩展为多步骤过程来改进这一范式，其中大语言模型（LLM）充当搜索智能体，生成中间思考（thoughts）和子查询，以迭代方式与检索系统交互。然而，由于需要自回归地生成冗长的思考和子查询，这一迭代过程会带来显著的延迟。为了解决这一局限，我们提出了LatentRAG，这是一个新颖的框架，它将推理和检索都从离散的语言空间转移到连续的潜在空间。与现有的逐token生成自然语言思考或子查询的显式方法不同，LatentRAG直接从隐藏状态中以单次前向传播产生用于思考和子查询的潜在token。

    arXiv:2605.06285v2 Announce Type: replace  Abstract: Single-step retrieval-augmented generation (RAG) provides an efficient way to incorporate external information for simple question answering tasks but struggles with complex questions. Agentic RAG extends this paradigm by replacing single-step retrieval with a multi-step process, in which the large language model (LLM) acts as a search agent that generates intermediate thoughts and subqueries to iteratively interact with the retrieval system. This iterative process incurs substantial latency due to the autoregressive generation of lengthy thoughts and subqueries. To address this limitation, we propose LatentRAG, a novel framework that shifts both reasoning and retrieval from discrete language space to continuous latent space. Unlike existing explicit methods that generate natural language thoughts or subqueries token-by-token, LatentRAG produces latent tokens for thoughts and subqueries directly from the hidden states in a single for
    
[^333]: 为什么深度神经网络中会涌现几何连续性：残差连接与旋转对称性破缺

    Why Geometric Continuity Emerges in Deep Neural Networks: Residual Connections and Rotational Symmetry Breaking

    [https://arxiv.org/abs/2605.04971](https://arxiv.org/abs/2605.04971)

    论文揭示了深度网络中“几何连续性”现象（相邻层主奇异向量方向相似）的起源：残差连接通过跨层梯度一致性对齐权重更新，对称性破缺的非线性激活通过固定共享坐标系防止旋转漂移，二者共同作用产生了这一性质。

    

    深度神经网络中的权重矩阵表现出几何连续性——相邻层的主奇异向量指向相似的方向。尽管这一性质已被广泛观察到，但其起源仍缺乏解释。通过在玩具MLP和小型Transformer上的实验，我们识别出两种机制：残差连接产生跨层梯度一致性，使各层的权重更新相互对齐；对称性破缺的非线性则将所有层约束在一个共享坐标系中，防止了原本会破坏权重结构稳定性的旋转漂移。至关重要的是，一种非线性但旋转等变的激活函数只能达到ReLU连续性的一半以下，从而将对称性破缺——而非非线性本身——分离为起作用的关键成分。激活函数和归一化扮演着不同的角色：激活函数将连续性集中在主奇异方向上，而归一化则将其分布在多个方向上。

    arXiv:2605.04971v2 Announce Type: replace-cross  Abstract: Weight matrices in deep networks exhibit geometric continuity---principal singular vectors of adjacent layers point in similar directions. While this property has been widely observed, its origin remains unexplained. Through experiments on toy MLPs and small transformers, we identify two mechanisms: residual connections create cross-layer gradient coherence that aligns weight updates across layers, and symmetry-breaking nonlinearities constrain all layers to a shared coordinate frame, preventing the rotation drift that would otherwise destabilize weight structure. Crucially, a nonlinear but rotation-equivariant activation reaches less than half of ReLU's continuity, isolating symmetry breaking---not nonlinearity alone---as the active ingredient. Activation and normalization play distinct roles: activation concentrates continuity in the leading singular direction, while normalization distributes it across multiple directions. In
    
[^334]: SATFormer：学习何时在Transformer中使用早期值残差

    SATFormer: Learning When to Use Early Value Residuals in Transformers

    [https://arxiv.org/abs/2605.03953](https://arxiv.org/abs/2605.03953)

    提出SATFormer，通过上下文相关的门控机制让每个token和注意力头动态决定何时复用第一层值残差，在从1.3亿到13亿参数的多个规模上均超越了静态值残差和标准Transformer基线。

    

    arXiv:2605.03953v3 公告类型：replace-cross。摘要：近期的Transformer架构让较深的层能够接触到较早层计算得到的表示，其动机在于观察到随着残差流在深度方向上不断被变换，低层特征会变得更难恢复。其中一种方法引入了静态值残差：即通过学习到的混合系数，将第一层的值投影 V_1 均匀地暴露给所有token和注意力头。然而，我们推测 V_1 的有用性在token、注意力头和上下文之间不太可能是恒定不变的，因此我们将早期表示的复用视为一个检索问题。为此，我们提出了选择性访问Transformer（Selective Access Transformer，SATFormer），它在保留第一层值通路的同时，通过一个依赖上下文的门控机制来控制对它的访问。在从1.3亿到13亿参数规模的多个模型上，SATFormer在每个评估规模下的验证损失均优于静态值残差和标准Transformer基线，并提升了平均零样本准确率。

    arXiv:2605.03953v3 Announce Type: replace-cross  Abstract: Recent Transformer architectures expose later layers to representations computed in earlier layers, motivated by the observation that low-level features can become harder to recover as the residual stream is transformed through depth. One such method adds static value residuals: learned mixing coefficients that expose the first-layer value projection V_1 uniformly across tokens and heads. However, we theorize that the usefulness of $V_1$ is unlikely to be constant across tokens, heads, and contexts, and therefore treat early-representation reuse as a retrieval problem. Here, we introduce Selective Access Transformer (SATFormer), which preserves the first-layer value pathway while controlling access with a context-dependent gate. Across models from 130M to 1.3B parameters, SATFormer improves validation loss over the static value-residual and Transformer baselines at every evaluated scale, and improves average zero-shot accuracy 
    
[^335]: SignVerse-2M：涵盖55多种手语的两百万片段姿态原生数据宇宙

    SignVerse-2M: A Two-Million-Clip Pose-Native Universe of 55+ Sign Languages

    [https://arxiv.org/abs/2605.01720](https://arxiv.org/abs/2605.01720)

    本文提出SignVerse-2M，一个涵盖55多种手语、包含约两百万个统一DWPose关键点姿态片段的多语言数据集，填补了真实场景手语视频大规模姿态表示的空白。

    

    大规模手语资源通常将RGB视频与文本配对，以支持识别、翻译和生成任务。然而，RGB帧中包含背景、服装以及其他与手语动作无关的外观信息。将视频表示为DWPose等全身关键点可以消除大部分此类变化，生成对身体、手部和面部动作的紧凑描述，并与近期姿态引导视频模型所使用的输入格式相匹配。现有数据集并未针对真实场景中的手语视频大规模提供这种表示。我们提出了SignVerse-2M，这是一个多语言数据集，包含来自55多种手语的约两百万个姿态片段。我们使用统一的DWPose流程处理公开可用的手语视频，以相同的关键点格式表示所有片段中的身体、手部和面部动作。源视频包含多样化的手语使用者、视角和录制环境。

    arXiv:2605.01720v4 Announce Type: replace-cross  Abstract: Large-scale sign language resources usually pair RGB videos with text and support recognition, translation, and generation. However, RGB frames contain backgrounds, clothing, and other appearance information unrelated to signing motion. Representing the videos as whole-body keypoints such as DWPose removes much of this variation, produces a compact description of body, hand, and facial movement, and matches the input used by recent pose-guided video models. Existing datasets do not provide this representation at scale for in-the-wild sign language videos. We present SignVerse-2M, a multilingual dataset containing approximately two million pose clips from more than 55 sign languages. We process publicly available sign language videos with a shared DWPose pipeline, representing body, hand, and facial motion in the same keypoint format across all clips. The source videos include diverse signers, viewpoints, and recording environme
    
[^336]: 零样本二元视觉-语言安全分类中由提示词引发的分数方差

    Prompt-Induced Score Variance in Zero-Shot Binary Vision-Language Safety Classification

    [https://arxiv.org/abs/2605.00326](https://arxiv.org/abs/2605.00326)

    该论文揭示零样本安全分类中单一提示词的不安全分数受提示词措辞影响方差大且不可靠，证明提示词均值集成在跨基准评估下更加稳健，而基于开发标签校准的单一提示词仅在开发基准上占优。

    

    安全流水线通常通过对模型首个token的不安全分数进行阈值判断来拦截内容。我们证明单一提示词得到的分数并非可靠的概率，并比较了两种修复方法：提示词均值集成与使用开发集标签进行校准。在七个视觉-语言模型和两个基准测试上，输入的不安全概率在15种保持任务不变的改写之间存在差异，平均标准差为0.16-0.30；方差最大的输入更容易被错误分类。在14个模型-基准配对中的13个上，15个提示词的均值比最佳单一提示词具有更低的负对数似然（NLL），即使该单一提示词是借助评估标签选出的。开发标签的使用改变了这一局面：在开发基准的留出数据上，联合选择并校准单一提示词在全部七个模型上都优于同样经过校准的均值方法。然而，在不重新拟合的跨基准评估中，优势发生逆转，经过校准的均值方法（摘要在此处截断）

    arXiv:2605.00326v2 Announce Type: replace  Abstract: Safety pipelines often block content by thresholding a model's first-token unsafe score. We show that a single prompt's score is not a reliable probability and compare two repairs: mean ensemble and calibration with development labels. Across seven vision-language models and two benchmarks, an input's unsafe probability varies across 15 task-preserving rewordings with a standard deviation of 0.16-0.30 on average; the inputs that vary most are more often misclassified. The 15-prompt mean has lower negative log-likelihood (NLL) than the best single prompt, even one chosen with evaluation labels, on 13 of 14 model-benchmark pairs. Development labels change the picture: jointly selecting and calibrating a single prompt outperforms the similarly calibrated mean for all seven models on held-out data from the development benchmark. Under cross-benchmark evaluation without refitting, however, the advantage reverses, with the calibrated mean 
    
[^337]: 重复胜于多样性：面向样本高效德语语言建模的高信号数据过滤

    Repetition over Diversity: High-Signal Data Filtering for Sample-Efficient German Language Modeling

    [https://arxiv.org/abs/2604.28075](https://arxiv.org/abs/2604.28075)

    该研究发现，对于德语语言建模，重复训练经过严格过滤的高质量数据始终优于在更大但过滤程度较低的数据集上进行单次训练，且这一性能优势在经过7个训练周期后依然保持。

    

    最近的研究表明，将海量英语网络语料库过滤为高质量子集可以显著提高训练效率。然而，对于德语、法语或日语等高资源非英语语言来说，激进过滤带来一个战略困境：从业者应该优先考虑多样性，即在大量轻度过滤的网络数据上进行单次训练，还是优先考虑质量，即严格过滤出高质量核心数据并在多个训练周期中重复使用？我们针对德语研究了这一权衡问题，将分层质量过滤器应用于5亿份网络文档，比较了在过滤子集上进行多周期训练与在多样化语料库上进行单次训练的效果。我们在多个模型规模和token预算下的实验表明，重复高质量数据始终优于在更大、过滤程度较低的集合上进行单次训练。值得注意的是，即使经过7个训练周期后，这一性能差距依然存在。

    arXiv:2604.28075v3 Announce Type: replace-cross  Abstract: Recent research has shown that filtering massive English web corpora into high-quality subsets significantly improves training efficiency. However, for high-resource non-English languages like German, French, or Japanese, aggressive filtering creates a strategic dilemma: should practitioners prioritize diversity by training once on large amounts of lightly filtered web data, or prioritize quality by strictly filtering for a high-quality core and repeating it over multiple epochs? We investigate this trade-off for German by applying hierarchical quality filters to 500M web documents, comparing multi-epoch training on filtered subsets against single-pass training on a diverse corpus. Our experiments across multiple model scales and token budgets show that repeating high-quality data consistently outperforms single-pass training on larger, less filtered sets. Notably, the performance gap persists even after 7 epochs. Our findings 
    
[^338]: 为什么某些情绪对大语言模型更难？通过稀疏自编码器揭示情绪推断的因果机制

    Why Are Some Emotions Harder for LLMs? Uncovering the Causal Mechanisms of Emotion Inference via Sparse Autoencoders

    [https://arxiv.org/abs/2604.25866](https://arxiv.org/abs/2604.25866)

    本研究通过稀疏自编码器揭示了不同情绪在LLM中推断机制的因果差异，发现惊讶和恐惧依赖集中特征集，而厌恶则依赖更分散的因果组织，这解释了模型对不同情绪表现不均的原因。

    

    arXiv:2604.25866v2 公告类型：替换 摘要：大语言模型（LLMs）越来越多地用于情感敏感的人机交互应用中，在这些应用中，可靠的情绪检测至关重要。然而，它们的情绪识别能力仍然不均衡：模型通常在某些情绪上表现良好，而在其他情绪上却持续表现不佳。尽管最近的研究探索了LLMs中的情绪机制，但从机械可解释性的角度，我们对于为什么模型在某些情绪上比其他情绪更弱的原因知之甚少。在这项工作中，我们通过使用稀疏自编码器（SAEs）研究情绪推断的因果机制，来探究情绪特定偏差。我们系统地识别了驱动情绪推断的因果稀疏情绪特征，并分析了它们在情绪内部和跨情绪之间的稀疏因果组织。我们发现，某些情绪（如惊讶和恐惧）依赖于高度集中的特征集，而厌恶则表现出更分散的稀疏因果组织：其

    arXiv:2604.25866v2 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly used in emotionally sensitive human-AI applications, where reliable emotion detection is essential. However, their emotion recognition abilities remain uneven: models often perform well on some emotions while consistently struggling with others. Although recent work has explored emotion mechanisms in LLMs, little is known about why models are weaker on some emotions than others from a mechanistic interpretability perspective. In this work, we investigate emotion-specific biases through the causal mechanisms of emotion inference using sparse autoencoders (SAEs). We systematically identify causal sparse emotion features that drive emotion inference and analyze their sparse causal organization within and across emotions. We show that some emotions, such as surprise and fear, rely on highly concentrated feature sets, whereas disgust exhibits a more distributed sparse causal organization: its 
    
[^339]: BERAG：用于基于知识的视觉问答的贝叶斯集成检索增强生成

    BERAG: Bayesian Ensemble Retrieval-Augmented Generation for Knowledge-based Visual Question Answering

    [https://arxiv.org/abs/2604.22678](https://arxiv.org/abs/2604.22678)

    该论文提出BERAG框架，通过贝叶斯集成方式替代传统的文档拼接方法进行检索增强生成，解决了长上下文中的“迷失在中间”效应和二次方计算成本问题，并结合BEFT微调方法应用于基于知识的视觉问答任务。

    

    在检索增强生成（RAG）问答中，一种常见方法是将文档拼接成单一上下文，再传递给语言模型以生成答案。这种方法虽然简单，但可能掩盖单个文档的贡献，使归因变得困难，并导致“迷失在中间”效应，即长上下文中的相关信息被忽视。此外，拼接方式的可扩展性较差：计算成本随上下文长度呈二次方增长，当上下文包含视觉数据时（如视觉问答场景），这一问题尤为严重。通过限制上下文长度来缓解这些问题的尝试，反而可能因阻止模型受益于更深层次检索带来的更高召回率而进一步限制性能。我们提出了贝叶斯集成检索增强生成（BERAG）以及贝叶斯集成微调（BEFT），作为一种RAG框架……

    arXiv:2604.22678v2 Announce Type: replace  Abstract: A common approach to question answering with retrieval-augmented generation (RAG) is to concatenate documents into a single context and pass it to a language model to generate an answer. While simple, this strategy can obscure the contribution of individual documents, making attribution difficult and contributing to the ``lost-in-the-middle'' effect, where relevant information in long contexts is overlooked. Concatenation also scales poorly: computational cost grows quadratically with context length, a problem that becomes especially severe when the context includes visual data, as in visual question answering. Attempts to mitigate these issues by limiting context length can further restrict performance by preventing models from benefiting from the improved recall offered by deeper retrieval. We propose Bayesian Ensemble Retrieval-Augmented Generation (BERAG), along with Bayesian Ensemble Fine-Tuning (BEFT), as a RAG framework in whi
    
[^340]: 文档干草堆中的语义之针：LLM作为评判者的相似度评分敏感性测试

    Semantic Needles in Document Haystacks: Sensitivity Testing of LLM-as-a-Judge Similarity Scoring

    [https://arxiv.org/abs/2604.18835](https://arxiv.org/abs/2604.18835)

    该论文提出了一个可扩展的多因素实验框架，通过“大海捞针”式的成对文档比较系统测试七个LLM的相似度评分敏感性，并揭示了包括与候选顺序效应不同的文档内位置偏差在内的多种惊人发现。

    

    我们提出了一个可扩展的多因素实验框架，系统地探究LLM在成对文档比较中对细微语义变化的敏感性。我们将其类比为“大海捞针”问题：一个语义被修改的句子（针）被嵌入到周围的上下文（干草）中，我们在所有组合下变化扰动类型（否定、连词交换、命名实体替换）、上下文类型（原始上下文与主题无关上下文）、针的位置以及文档长度，并在数万个文档对上测试了七个LLM。我们的分析揭示了几个惊人的发现。首先，LLM可能表现出一种不同于先前研究的候选顺序效应的文档内位置偏差：一些模型在语义差异出现在文档较早位置时会对其惩罚更严厉。其次，当被修改的句子被主题无关的上下文包围时，它会提高相似度评分……

    arXiv:2604.18835v2 Announce Type: replace-cross  Abstract: We propose a scalable, multifactorial experimental framework that systematically probes LLM sensitivity to subtle semantic changes in pairwise document comparison. We analogize this as a needle-in-a-haystack problem: a single semantically altered sentence (the needle) is embedded within surrounding context (the hay), and we vary the perturbation type (negation, conjunction swap, named entity replacement), context type (original vs. topically unrelated), needle position, and document length across all combinations, testing seven LLMs on tens of thousands of document pairs. Our analysis reveals several striking findings. First, LLMs can exhibit a within-document positional bias distinct from previously studied candidate-order effects: some models penalize semantic differences more harshly when they occur earlier in a document. Second, when the altered sentence is surrounded by topically unrelated context, it raises similarity sco
    
[^341]: LLM智能体长期记忆安全综述：跨记忆生命周期的攻击、防御与治理

    A Survey on Long-Term Memory Security in LLM Agents: Attacks, Defenses, and Governance Across the Memory Lifecycle

    [https://arxiv.org/abs/2604.16548](https://arxiv.org/abs/2604.16548)

    本文提出记忆生命周期框架与可验证记忆治理（VMG）框架，系统刻画了LLM智能体长期记忆所面临的持久性、有状态性与传播性等新型安全威胁，并沿六个生命周期阶段和四个安全目标组织了攻击、防御及其跨阶段依赖关系的全景式分析。

    

    LLM智能体中可写入、可跨会话持久化记忆的出现，带来了与传统的以输入为中心的安全关注点在性质上截然不同的威胁格局，其特征体现为三种属性：持久性、有状态性和传播性。为了系统地刻画这一威胁格局，我们提出了一个记忆生命周期框架，沿两个维度对攻击、防御及其跨阶段依赖关系进行组织：六个生命周期阶段（写入、存储、检索、执行、共享与传播、遗忘与回滚）和四个安全目标（完整性、机密性、可用性、治理）。这一分析进而揭示了在系统层面需要形式化安全保证的需求，由此催生了可验证记忆治理——一个由五个架构原语构成的框架，它规定了长期记忆系统必须提供哪些可验证机制，以维持对其记忆状态可审计、可恢复的控制。我们的

    arXiv:2604.16548v3 Announce Type: replace-cross  Abstract: The emergence of writable, cross-session persistent memory in LLM agents introduces a qualitatively different threat landscape from conventional input-centric security concerns, characterized by three properties: persistence, statefulness, and propagation. To systematically characterize this landscape, we propose a Memory Lifecycle Framework that organizes attacks, defenses, and their cross-phase dependencies along two axes: six lifecycle phases (Write, Store, Retrieve, Execute, Share & Propagate, Forget & Rollback) and four security objectives (Integrity, Confidentiality, Availability, Governance). This analysis in turn exposes the need for formal security guarantees at the system level, motivating Verifiable Memory Governance (VMG), a framework of five architectural primitives that specifies what verifiable mechanisms a long-term-memory system must provide to maintain auditable, recoverable control over its memory state. Our 
    
[^342]: 迈向真实世界的人类行为模拟：基于长时程、跨场景、异构行为轨迹的大语言模型基准测试

    Towards Real-world Human Behavior Simulation: Benchmarking Large Language Models on Long-horizon, Cross-scenario, Heterogeneous Behavior Traces

    [https://arxiv.org/abs/2604.08362](https://arxiv.org/abs/2604.08362)

    该论文提出了首个完全基于真实数据构建的预测性用户模拟基准OmniBehavior，整合长时程、跨场景、异构行为模式于统一框架，并实证揭示当前大语言模型难以准确预测真实世界中复杂的跨场景人类行为。

    

    大语言模型的出现揭示了构建通用用户模拟器的潜力。然而，现有的基准测试仍局限于孤立场景、狭窄的动作空间或合成数据，无法捕捉真实人类行为的整体特性。为了弥合这一差距，我们提出了OmniBehavior，这是首个完全基于真实世界数据构建的预测性用户模拟基准，将长时程、跨场景和异构的行为模式整合到一个统一框架中。基于该基准，我们首次提供了实证证据，表明以往由孤立场景构成的数据集存在“隧道视野”问题，而真实世界的决策依赖于长期的、跨场景的因果链。对最先进大语言模型的广泛评估表明，当前模型难以准确预测这些复杂行为，即使上下文窗口不断扩大，性能也趋于停滞。

    arXiv:2604.08362v3 Announce Type: replace-cross  Abstract: The emergence of Large Language Models (LLMs) has illuminated the potential for a general-purpose user simulator. However, existing benchmarks remain constrained to isolated scenarios, narrow action spaces, or synthetic data, failing to capture the holistic nature of authentic human behavior. To bridge this gap, we introduce OmniBehavior, the first predictive user simulation benchmark constructed entirely from real-world data, integrating long-horizon, cross-scenario, and heterogeneous behavioral patterns into a unified framework. Based on this benchmark, we first provide empirical evidence that previous datasets with isolated scenarios suffer from tunnel vision, whereas real-world decision-making relies on long-term, cross-scenario causal chains. Extensive evaluations of state-of-the-art LLMs reveal that current models struggle to accurately predict these complex behaviors, with performance plateauing even as context windows e
    
[^343]: LinearARD：面向RoPE恢复的线性内存注意力蒸馏

    LinearARD: Linear-Memory Attention Distillation for RoPE Restoration

    [https://arxiv.org/abs/2604.00004](https://arxiv.org/abs/2604.00004)

    LinearARD提出了一种线性内存的注意力蒸馏方法，通过将经RoPE缩放的学生模型与冻结的原生RoPE教师模型在最后一层注意力的Q/K/V自关系矩阵逐行分布上对齐，以线性内存开销直接恢复注意力结构，从而摆脱传统持续预训练对海量长文本数据的依赖。

    

    arXiv:2604.00004v3 公告类型： replace-cross  摘要：大语言模型上下文窗口的扩展通常通过缩放位置编码并随后进行持续预训练（CPT）来实现。尽管有效，但这一范式以极度依赖数据和计算成本高昂而著称，需要海量的长文本语料来将模型重新校准至偏移后的位置分布。我们提出LinearARD，这是一种恢复蒸馏方法，通过与冻结的原生RoPE教师模型进行注意力结构一致性来恢复经RoPE缩放的学生模型。不同于下一个词预测或不透明的隐藏状态匹配，LinearARD对齐最后一层注意力中稠密的 $Q/Q$、$K/K$ 和 $V/V$ 自关系矩阵的逐行分布，从而直接监督注意力动态。为消除 $n \times n$ 关系图的二次方内存瓶颈，我们引入了一种线性激活内存内核，仅存储每个token的log-sum-exp统计量……

    arXiv:2604.00004v3 Announce Type: replace-cross  Abstract: The extension of context windows in Large Language Models is typically facilitated by scaling positional encodings followed by Continued Pre-Training (CPT). While effective, this paradigm is notoriously data-hungry and computationally expensive, requiring massive long-text corpora to recalibrate the model to the shifted positional distribution. We propose LinearARD, a restoration distillation method that restores Rotary Position Embedding (RoPE)-scaled students through attention-structure consistency with a frozen native-RoPE teacher. Rather than next-token prediction or opaque hidden-state matching, LinearARD aligns row-wise distributions of dense $Q/Q$, $K/K$, and $V/V$ self-relation matrices from the final attention layer to directly supervise attention dynamics. To remove the quadratic memory bottleneck of $n \times n$ relation maps, we introduce a linear-activation-memory kernel that stores only per-token log-sum-exp stati
    
[^344]: ActTail：大语言模型中的全局激活稀疏化

    ActTail: Global Activation Sparsity in Large Language Models

    [https://arxiv.org/abs/2603.12272](https://arxiv.org/abs/2603.12272)

    提出基于重尾自正则化理论的ActTail方法，利用各投影层经验谱密度的重尾指数为其分配全局激活稀疏度预算，在加速大语言模型推理的同时缓解性能损失。

    

    激活稀疏化是一种通过减少计算量和内存搬运来加速大语言模型（LLM）推理的有前景的方法。然而，现有的激活稀疏化方法通常在各投影层上施加统一的稀疏度，忽略了Transformer权重的异构统计特性，从而加剧了性能下降。在本文中，我们提出了ActTail，这是一种基于TopK幅值的激活稀疏化方法，其全局激活稀疏度分配建立在重尾自正则化（HT-SR）理论的基础上。具体而言，我们通过从每个投影层的经验谱密度（ESD）计算得到的重尾指数来刻画这种异构性，并将其作为定量指标来为各投影层分配专属的稀疏度预算。重要的是，我们提供了理论分析，在HT（摘要截断）

    arXiv:2603.12272v2 Announce Type: replace  Abstract: Activation sparsity is a promising approach for accelerating large language model (LLM) inference by reducing computation and memory movement. However, existing activation sparsity methods typically apply uniform sparsity across projections, ignoring the heterogeneous statistical properties of Transformer weights and thereby amplifying performance degradation. In this paper, we propose ActTail, a TopK magnitude-based activation sparsity method with global activation sparsity allocation grounded in Heavy-Tailed Self-Regularization (HT-SR) theory. Specifically, we capture this heterogeneity via the heavy-tail exponent computed from each projection's empirical spectral density (ESD), which is used as a quantitative indicator to assign projection-specific sparsity budgets. Importantly, we provide a theoretical analysis that establishes an explicit relationship between the activation sparsity ratio and the heavy-tail exponent under the HT
    
[^345]: 紧凑型SSL骨干网络对音频深度伪造检测重要吗？基于RAPTOR的受控研究

    Do Compact SSL Backbones Matter for Audio Deepfake Detection? A Controlled Study with RAPTOR

    [https://arxiv.org/abs/2603.06164](https://arxiv.org/abs/2603.06164)

    该研究提出RAPTOR检测器，通过受控实验证明多语言HuBERT预训练（而非模型规模）是紧凑型SSL骨干网络实现跨域音频深度伪造检测鲁棒性的关键，并引入基于扰动的测试时不确定性增强协议，揭示了WavLM过度自信而mHuBERT校准稳定的差异。

    

    自监督学习（SSL）是现代音频深度伪造检测的基石，然而以往研究大多聚焦于单一的大型wav2vec2-XLSR骨干网络，对紧凑型模型的研究则较为匮乏。我们提出了RAPTOR（Representation Aware Pairwise-gated Transformer for Out-of-domain Recognition，用于域外识别的表示感知成对门控Transformer），这是一项在统一的成对门控融合检测器框架下，对来自HuBERT和WavLM的紧凑型SSL骨干网络开展的受控研究，并在14个跨域基准上进行了评估。结果表明，多语言HuBERT预训练是跨域鲁棒性的主要驱动因素，使1亿参数规模的模型能够匹敌更大规模及商业系统。除EER之外，我们还引入了一种结合基于扰动的偶然不确定性的测试时增强协议，以暴露标准指标无法察觉的校准差异：WavLM变体在扰动下表现出过度自信的错误校准，而迭代训练的mHuBERT则保持稳定。这些发现表明，关键在于SSL的预训练轨迹，而非模型本身……（原文在此截断）

    arXiv:2603.06164v2 Announce Type: replace-cross  Abstract: Self-supervised learning (SSL) underpins modern audio deepfake detection, yet most prior work centers on a single large wav2vec2-XLSR backbone, leaving compact under studied. We present RAPTOR, Representation Aware Pairwise-gated Transformer for Out-of-domain Recognition a controlled study of compact SSL backbones from the HuBERT and WavLM within a unified pairwise-gated fusion detector, evaluated across 14 cross-domain benchmarks. We show that multilingual HuBERT pre-training is the primary driver of cross-domain robustness, enabling 100M models to match larger and commercial systems. Beyond EER, we introduce a test-time augmentation protocol with perturbation-based aleatoric uncertainty to expose calibration differences invisible to standard metrics: WavLM variants exhibit overconfident miscalibration under perturbation, whereas iterative mHuBERT remains stable. These findings indicate that SSL pre-training trajectory, not mo
    
[^346]: 通过事后嵌入对齐提升压缩Transformer模型的性能

    Improving the Performance of Compressed Transformers with Post-Hoc Embedding Alignment

    [https://arxiv.org/abs/2602.14635](https://arxiv.org/abs/2602.14635)

    提出了一种事后训练的对齐适配器（AlAd），通过将压缩Transformer模型的词元嵌入与性能更优的参考模型对齐，并利用局部上下文信息，从而恢复压缩所导致的性能损失。

    

    模型压缩是在资源受限环境中部署Transformer模型的必要手段。然而，压缩会降低Transformer模型在下游任务上的性能。现有解决方案如知识蒸馏（KD）和参数高效微调（PEFT）各有其局限性。知识蒸馏在教师模型的监督下训练学生模型，但需要教师模型的完整训练流程。LoRA等PEFT方法可以使模型适应新任务，但它们无法恢复压缩所造成的性能损失。我们的直觉是，通过将压缩模型的词元嵌入与性能更优的参考模型的词元嵌入进行对齐，可以改进压缩后的模型。我们通过一个放置在压缩模型最后一层之后的小型前馈对齐适配器（Alignment Adapter, AlAd）来学习这种对齐。AlAd会读取相邻词元的窗口，因此能够利用局部上下文信息。AlAd是在压缩模型生成之后进行事后训练的。

    arXiv:2602.14635v2 Announce Type: replace-cross  Abstract: Model compression is necessary for deployment of transformer models in resource-constrained environments. However, compression reduces the performance of transformer models on downstream tasks. Existing solutions such as Knowledge Distillation (KD) and Parameter Efficient Fine-Tuning (PEFT) have their own limitations. KD trains the student under the teacher's supervision, but needs the teacher's training pipeline. PEFT methods such as LoRA adapt a model to a new task, but they do not recover the loss caused by compression. Our intuition is that a compressed model can be improved by aligning its token embeddings with those of a better performing reference model. We learn this alignment with a small feed-forward Alignment Adapter (AlAd) placed after the final layer of the compressed model. AlAd reads a window of neighbouring tokens, so it can use local context. AlAd is trained post-hoc, after the compressed model has been produce
    
[^347]: 预训练数据的词汇印记

    The Lexical Imprint of Pre-training Data

    [https://arxiv.org/abs/2602.10657](https://arxiv.org/abs/2602.10657)

    本文提出词级一元交叉熵（UCE）这一与分词器无关、无需模型的词汇对齐度量，并通过受控的从零预训练实验证明，预训练语料库与基准测试之间的词汇对齐程度能够持续解释零样本基准准确率。

    

    预训练语料库与基准测试目标之间的词级对齐，能在多大程度上解释零样本基准测试的准确率？我们通过词级一元交叉熵来研究这个问题。这是一种与分词器无关、无需模型参与的度量方法，用于衡量参考语料库与目标基准之间的词汇对齐程度。似然训练基于预训练语料库来塑造模型的预测分布，而在基于似然的评估中，零样本得分部分取决于该语料库与基准目标之间的词汇对齐。要清晰地追踪预训练数据的词汇印记如何随语料库与基准的对齐程度而变化，需要进行受控的从零预训练实验，即在保持训练方案固定不变的情况下，改变所使用的语料库。在这一受控扫描实验中，我们涵盖了10个零样本基准、4个预训练语料库和3个模型规模，发现语料库与基准之间的词汇对齐程度（即UCE）始终……（原文摘要到此截断）

    arXiv:2602.10657v2 Announce Type: replace  Abstract: How much of zero-shot benchmark accuracy is captured by word-level alignment between a pre-training corpus and the benchmark target? We study this question through word-level unigram cross-entropy (UCE), a tokenizer-agnostic, model-free measure of lexical alignment between a reference corpus and a target benchmark. Likelihood training shapes the model's predictive distribution from the pre-training corpus, and zero-shot scores under likelihood-based evaluation depend in part on the lexical alignment between that corpus and the benchmark target. Cleanly tracing how the lexical imprint of pre-training data scales with corpus--benchmark alignment requires controlled from-scratch pre-training that varies corpus identity while holding the training recipe fixed. Across this controlled sweep, spanning $10$ zero-shot benchmarks, $4$ pre-training corpora, and $3$ model scales, we find that corpus--benchmark lexical alignment, UCE, consistentl
    
[^348]: OdysseyArena：面向长时程、主动与归纳式交互的大语言模型基准测试

    OdysseyArena: Benchmarking Large Language Models For Long-Horizon, Active and Inductive Interactions

    [https://arxiv.org/abs/2602.05843](https://arxiv.org/abs/2602.05843)

    提出OdysseyArena基准，通过形式化四个基元将智能体评估重新聚焦于长时程、主动和归纳式交互，并建立包含120个任务的OdysseyArena-Lite来衡量智能体从经验中自主发现潜在转换规律的能力。

    

    大语言模型的发展催生了能够在复杂环境中自主导航的智能体。然而，现有的评估主要采用演绎范式，即智能体基于显式给定的规则和静态目标来执行任务，且规划时域往往有限。至关重要的是，这忽略了智能体需要自主地从经验中发现潜在转换规律的归纳必要性，而这是实现智能体前瞻能力并维持策略连贯性的基石。为了弥合这一差距，我们提出了OdysseyArena，将智能体评估重新聚焦于长时程、主动和归纳式交互。我们形式化并实例化了四个基元，将抽象的转换动态转化为具体的交互环境。在此基础上，我们建立了用于标准化基准测试的OdysseyArena-Lite，提供了一组120个任务来衡量智能体的归纳（摘要在此处截断）。

    arXiv:2602.05843v3 Announce Type: replace  Abstract: The advancement of Large Language Models (LLMs) has catalyzed the development of autonomous agents capable of navigating complex environments. However, existing evaluations primarily adopt a deductive paradigm, where agents execute tasks based on explicitly provided rules and static goals, often within limited planning horizons. Crucially, this neglects the inductive necessity for agents to discover latent transition laws from experience autonomously, which is the cornerstone for enabling agentic foresight and sustaining strategic coherence. To bridge this gap, we introduce OdysseyArena, which re-centers agent evaluation on long-horizon, active, and inductive interactions. We formalize and instantiate four primitives, translating abstract transition dynamics into concrete interactive environments. Building upon this, we establish OdysseyArena-Lite for standardized benchmarking, providing a set of 120 tasks to measure an agent's induc
    
[^349]: GraphDancer：通过课程强化学习训练大语言模型探索和推理图结构

    GraphDancer: Training LLMs to Explore and Reason over Graphs via Curriculum Reinforcement Learning

    [https://arxiv.org/abs/2602.02518](https://arxiv.org/abs/2602.02518)

    GraphDancer是一个强化学习框架，通过图感知的课程式训练（由易到难调度），教会中等规模的大语言模型在异构图知识源上交替进行推理与函数调用，从而实现多跳证据聚合的图导航推理，并将该能力迁移到未见过的领域。

    

    大语言模型（LLM）日益依赖外部知识来提升事实准确性，然而网络上的许多知识源，如学术索引和产品目录，都是以异构图的形式组织的。对这种图结构化知识的推理面临两个关键挑战：（1）在结构化的、由模式定义的关系中进行导航，需要超越基于相似性检索的精确函数调用；（2）回答复杂问题通常需要通过迭代式的信息搜寻来聚合多跳证据。我们研究大语言模型如何学习这些交互能力，并将其迁移到未见过的领域。我们提出了GraphDancer，一个强化学习（RL）框架，通过交替进行推理和函数执行来教会大语言模型在图中导航。为了改进针对中等规模大语言模型的强化学习，我们引入了一种图感知课程机制，该机制根据问题的结构难度，以由易到难的方式安排训练。

    arXiv:2602.02518v3 Announce Type: replace-cross  Abstract: Large language models (LLMs) increasingly rely on external knowledge to improve factuality, yet many knowledge sources on the Web, such as scholarly indexes and product catalogs, are organized as heterogeneous graphs. Reasoning over such graph-structured knowledge poses two key challenges: (1) navigating structured, schema-defined relations requires precise function calls beyond similarity-based retrieval, and (2) answering complex questions often demands multi-hop evidence aggregation through iterative information seeking. We study how LLMs can learn these interaction capabilities and transfer them to unseen domains. We propose GraphDancer, a reinforcement learning (RL) framework that teaches LLMs to navigate graphs by interleaving reasoning and function execution. To improve RL for moderate-sized LLMs, we introduce a graph-aware curriculum that schedules training by the structural difficulty of questions using an easy-to-hard
    
[^350]: DeALOG：去中心化多智能体日志介导推理框架

    DeALOG: Decentralized Multi-Agents Log-Mediated Reasoning Framework

    [https://arxiv.org/abs/2602.00996](https://arxiv.org/abs/2602.00996)

    DeALOG是一个去中心化多智能体多模态问答框架，让专业化智能体通过共享自然语言日志作为持久记忆进行通信，在无中央控制的情况下实现协作式错误检测与验证，并在多个问答基准上取得有竞争力的性能。

    

    跨文本、表格和图像的复杂问答需要整合多样化的信息来源，因此需要一个能够支持专业化处理、并具备协调性和可解释性的框架。我们提出了DeALOG，一个面向多模态问答的去中心化多智能体框架。它使用专业化的智能体——表格、上下文、视觉、摘要和验证智能体，这些智能体通过共享的自然语言日志作为持久记忆进行通信。这种基于日志的方法无需中央控制即可实现协作式错误检测和验证，从而提升鲁棒性。在FinQA、TAT-QA、CRT-QA、WikiTableQuestions、FeTaQA和MultiModalQA上的评估显示出具有竞争力的性能。分析证实了共享日志、智能体专业化和验证机制对准确性的重要性。DeALOG通过采用自然语言通信的模块化组件提供了一种可扩展的方法。

    arXiv:2602.00996v3 Announce Type: replace-cross  Abstract: Complex question answering across text, tables and images requires integrating diverse information sources. A framework supporting specialized processing with coordination and interpretability is needed. We introduce DeALOG, a decentralized multi-agent framework for multimodal question answering. It uses specialized agents: Table, Context, Visual, Summarizing and Verification, that communicate through a shared natural-language log as persistent memory. This log-based approach enables collaborative error detection and verification without central control, improving robustness. Evaluations on FinQA, TAT-QA, CRT-QA, WikiTableQuestions, FeTaQA, and MultiModalQA show competitive performance. Analysis confirms the importance of the shared log, agent specialization, and verification for accuracy. DeALOG, provides a scalable approach through modular components using natural-language communication.
    
[^351]: Fast KVzip：基于门控KV驱逐的高效且精确的大语言模型推理

    Fast KVzip: Efficient and Accurate LLM Inference with Gated KV Eviction

    [https://arxiv.org/abs/2601.17668](https://arxiv.org/abs/2601.17668)

    提出了一种基于门控的KV缓存驱逐方法，利用轻量级sink-attention门控模块和仅需前向传播的训练算法，在计算成本几乎可忽略的情况下驱逐高达70%的KV缓存，同时保持近乎无损的推理性能。

    

    高效的键值缓存管理对于大语言模型的实际部署至关重要，然而现有的压缩技术往往在性能下降与计算开销之间存在权衡。我们提出了一种新颖的基于门控的KV缓存驱逐方法，适用于冻结权重的大语言模型，能够在可忽略的计算成本下实现高压缩率。我们的方法引入了轻量级的sink-attention门控模块来识别并保留关键的KV对，并无缝集成到预填充和解码两个阶段中。所提出的门控训练算法仅依赖大语言模型的前向传播，避免了昂贵的反向传播，同时通过任务无关的重构目标实现了强大的任务泛化能力。在Qwen2.5-1M、Qwen3和Gemma3模型系列上的大量实验表明，我们的方法在驱逐高达70%的KV缓存的同时保持了近乎无损的性能。

    arXiv:2601.17668v3 Announce Type: replace-cross  Abstract: Efficient key-value (KV) cache management is crucial for the practical deployment of large language models (LLMs), yet existing compression techniques often incur a trade-off between performance degradation and computational overhead. We propose a novel gating-based KV cache eviction method for frozen-weight LLMs that achieves high compression ratios with negligible computational cost. Our approach introduces lightweight sink-attention gating modules to identify and retain critical KV pairs, and integrates seamlessly into both the prefill and decoding stages. The proposed gate training algorithm relies on forward passes of an LLM, avoiding expensive backpropagation, while achieving strong task generalization through a task-agnostic reconstruction objective. Extensive experiments across the Qwen2.5-1M, Qwen3, and Gemma3 families show that our method maintains near-lossless performance while evicting up to 70% of the KV cache. Th
    
[^352]: 正道直行：大语言模型拥有内在的道德路径吗？

    The Straight and Narrow: Do LLMs Possess an Internal Moral Path?

    [https://arxiv.org/abs/2601.10307](https://arxiv.org/abs/2601.10307)

    该论文利用道德基础理论首次系统绘制并操控了大语言模型内部细粒度的道德表征空间，并提出自适应道德融合（AMF）这一推理时动态干预方法，有效解决了AI安全中安全性与有用性的权衡难题。

    

    增强大语言模型（LLMs）的道德对齐是AI安全领域的一项关键挑战。当前的对齐技术往往只是充当表面上的防护栏，使得LLMs内在的道德表征基本未被触及。在本文中，我们通过利用道德基础理论（MFT）来绘制和操控LLMs细粒度的道德图景，从而弥合这一差距。通过跨语言线性探测方法，我们验证了模型中间层道德表征的共享性质，并发现了英语和中文之间存在一个共享但又有所差异的道德子空间。在此基础上，我们提取出可引导的“道德向量”，并在内部表征和行为输出两个层面成功验证了其有效性。利用道德概念的高泛化性，我们提出了自适应道德融合（AMF）方法，这是一种动态的推理时干预技术，将探测检测与向量注入相结合，以解决安全性与有用性之间的权衡问题。

    arXiv:2601.10307v2 Announce Type: replace  Abstract: Enhancing the moral alignment of Large Language Models (LLMs) is a critical challenge in AI safety. Current alignment techniques often act as superficial guardrails, leaving the intrinsic moral representations of LLMs largely untouched. In this paper, we bridge this gap by leveraging Moral Foundations Theory (MFT) to map and manipulate the fine-grained moral landscape of LLMs. Through cross-lingual linear probing, we validate the shared nature of moral representations in middle layers and uncover a shared yet different moral subspace between English and Chinese. Building upon this, we extract steerable Moral Vectors and successfully validate their efficacy at both internal and behavioral levels. Leveraging the high generalizability of morality, we propose Adaptive Moral Fusion (AMF), a dynamic inference-time intervention that synergizes probe detection with vector injection to tackle the safety-helpfulness trade-off. Empirical result
    
[^353]: 导航构思空间：用于科学思想定位的分解式概念表示

    Navigating Ideation Space: Decomposed Conceptual Representations for Positioning Scientific Ideas

    [https://arxiv.org/abs/2601.08901](https://arxiv.org/abs/2601.08901)

    该论文提出“构思空间”这一结构化表示，通过对比训练将科学知识分解为三个独立维度，实现了科学思想间概念距离的原则性测量与具有区分性的新颖性评估，帮助研究者精确定位新想法相对于现有研究的差异。

    

    科学发现需要将新想法置于不断扩大的现有知识版图之中。一个新兴的挑战是如何从快速增长的文献中识别出概念上相关的先前工作，并评估一个新想法与现有研究的区别。当前的嵌入方法通常将不同的概念方面混淆为单一表示，无法支持细粒度检索；与此同时，基于大语言模型的评估器容易受到谄媚偏好的影响，无法提供具有区分性的新颖性评估。我们引入了构思空间，这是一种结构化表示，通过对比训练将科学知识分解为三个不同的维度。我们的框架能够有原则地测量思想之间的概念距离，并对构思转换进行建模，以捕捉单个思想内部的逻辑联系。基于这一表示，我们提出了一个Hi（摘要在此处截断）

    arXiv:2601.08901v2 Announce Type: replace-cross  Abstract: Scientific discovery requires new ideas to be situated within an ever-expanding landscape of existing knowledge. An emerging challenge is how to identify conceptually relevant prior work from rapidly growing literature, and assess how a new idea differentiates from existing research. Current embedding approaches typically conflate distinct conceptual aspects into single representations and cannot support fine-grained retrieval; meanwhile, LLM-based evaluators are subject to sycophancy biases, failing to provide discriminative novelty assessment. We introduce the Ideation Space, a structured representation that decomposes scientific knowledge into three distinct dimensions learned through contrastive training. Our framework enables principled measurement of conceptual distance between ideas and modeling of ideation transitions that capture the logical connections within an idea. Building upon this representation, we propose a Hi
    
[^354]: 新加坡英语中的风格演变与大语言模型中立性

    Stylistic Evolution and LLM Neutrality in Singlish Language

    [https://arxiv.org/abs/2601.06580](https://arxiv.org/abs/2601.06580)

    本研究揭示了新加坡英语的风格随时间演变且主要由结构性特征驱动，同时发现大语言模型存在结构性权衡——无法同时生成真实的新加坡英语和时间中立的输出。

    

    新加坡英语（Singlish）是一种植根于新加坡多语言环境的克里奥尔语，它随着社会和技术的变革而不断演变。我们研究了十年来非正式数字消息中的历时风格变化，并探讨大语言模型（LLMs）能否生成在时间上中立的输出，以逼近该语言变体的稳定本质。通过使用词汇、语用、心理语言学以及基于编码器的特征，我们发现风格上的可分性随时间距离的增加而增强，这主要由长度和复杂性等结构性特征所驱动。在与零分布基线进行对比评估时，大多数大语言模型无法同时实现真实性和时间中立性，揭示了一种结构性权衡：生成逼真新加坡英语的模型继承了其时间偏差，而时间中立的模型则产生不真实的输出。这些发现将时间中立性确立为评估社会语言学属性的一个诊断性指标。

    arXiv:2601.06580v3 Announce Type: replace  Abstract: Singlish is a creole rooted in Singapore's multilingual environment that continues to evolve alongside social and technological change. We examine diachronic stylistic change across a decade of informal digital messages and ask whether Large Language Models (LLMs) can generate temporally neutral outputs approximating the stable essence of the variety. Using lexical, pragmatic, psycholinguistic, and encoder-based features, we find that stylistic separability increases with temporal distance, driven primarily by structural features such as length and complexity. Evaluated against a null distribution baseline, most LLMs fail to achieve both authenticity and temporal neutrality simultaneously, revealing a structural trade-off: models generating realistic Singlish inherit its temporal biases, while temporally neutral models produce inauthentic outputs. These findings position temporal neutrality as a diagnostic metric for assessing sociol
    
[^355]: 迈向全球化的医学大语言模型

    Toward Global Large Language Models in Medicine

    [https://arxiv.org/abs/2601.02186](https://arxiv.org/abs/2601.02186)

    本研究构建了包含20种语言的80万条医学数据集GlobMed及基准测试GlobMed-Bench，对56个先进大语言模型进行大规模评估，揭示了现有模型因依赖高资源语言训练而在全球医疗场景中的局限性。

    

    尽管医疗技术不断进步，但全球医疗资源的分布仍然不均。大语言模型的发展已经改变了医学领域的格局，并有望提高医疗质量、在全球范围内扩大医疗信息的获取渠道。然而，现有的LLM主要基于高资源语言进行训练，这限制了它们在全球医疗场景中的适用性。为了解决这一差距，我们构建了GlobMed，这是一个包含超过80万条数据、涵盖20种语言（包括7种低资源语言）的大型多语言医学数据集。在此基础上，我们建立了GlobMed-Bench，系统性地评估了56个最先进的专有和开放权重LLM，涉及GlobMed中20种语言中的12种，包含超过2万个独立实验和超过3200万条模型响应。这一大规模评估揭示了显著的（性能差异）

    arXiv:2601.02186v2 Announce Type: replace  Abstract: Despite continuous advances in medical technology, the global distribution of health care resources remains uneven. The development of large language models (LLMs) has transformed the landscape of medicine and holds promise for improving health care quality and expanding access to medical information globally. However, existing LLMs are primarily trained on high-resource languages, limiting their applicability in global medical scenarios. To address this gap, we constructed GlobMed, a large multilingual medical dataset containing more than 800,000 entries spanning 20 languages, including seven low-resource languages. Building on this, we established GlobMed-Bench, which systematically assesses 56 state-of-the-art proprietary and open-weight LLMs across 12 of the 20 languages represented in GlobMed, comprising over 20,000 independent experiments and more than 32 million model responses. This large-scale evaluation revealed significant
    
[^356]: 潜在标记会思考吗？对连续思维链的因果与对抗性分析

    Do Latent Tokens Think? A Causal and Adversarial Analysis of Chain-of-Continuous-Thought

    [https://arxiv.org/abs/2512.21711](https://arxiv.org/abs/2512.21711)

    本文通过因果操控实验和捷径实验发现，连续思维链（COCONUT）中的潜在标记并非编码忠实的推理过程，而是作为不可解释的占位符，促使模型走捷径而非进行真正的推理。

    

    潜在标记因能增强大语言模型（LLM）的推理能力而备受关注，但其内部机制仍不清楚。本文从可靠性视角审视这一问题，揭示了其根本性缺陷：潜在标记充当的是不可解释的占位符，而非编码忠实的推理过程。尽管它们对扰动具有抵抗力，但它们促进的是走捷径而非真正的推理。我们聚焦于连续思维链（COCONUT），该方法声称在保持性能的同时比显式思维链（CoT）具有更好的效率和稳定性。我们通过两种互补的方法对此进行研究。首先，操控实验对特定的标记子集（即COCONUT和显式CoT）进行扰动。与CoT标记不同，COCONUT标记对操控表现出极小的敏感性，且缺乏推理所需的关键信息。其次，捷径实验在有偏和分布外的设置下评估模型。

    arXiv:2512.21711v2 Announce Type: replace-cross  Abstract: Latent tokens are gaining attention for enhancing reasoning in large language models (LLMs), yet their internal mechanisms remain unclear. This paper examines the problem from a reliability perspective, uncovering fundamental weaknesses: latent tokens function as uninterpretable placeholders rather than encoding faithful reasoning. While resistant to perturbation, they promote shortcut usage over genuine reasoning. We focus on Chain-of-Continuous-Thought (COCONUT), which claims better efficiency and stability than explicit Chain-of-Thought (CoT) while maintaining performance. We investigate this through two complementary approaches. First, steering experiments perturb specific token subsets, namely COCONUT and explicit CoT. Unlike CoT tokens, COCONUT tokens show minimal sensitivity to steering and lack reasoning-critical information. Second, shortcut experiments evaluate models under biased and out-of-distribution settings. Res
    
[^357]: 面向对话式大语言模型的角色感知启发式情景注意力

    Role-aware Heuristic Episodic Attention for Conversational LLMs

    [https://arxiv.org/abs/2512.06869](https://arxiv.org/abs/2512.06869)

    该论文提出REA框架，通过为指令和情景交互分配不同的持久性与表示策略来管理对话上下文，有效缓解多轮对话中的累积性上下文衰退问题，在Long-MT-Bench+上实现16.5%的相对评分提升并将延迟降低2.91倍。

    

    随着多轮对话的不断增长，大语言模型往往会丢失对持久指令和相关信息的追踪。我们通过三种相关的失败模式来研究这种累积性上下文衰退：注意力污染、注意力稀释和注意力漂移。我们提出了REA（角色感知启发式情景注意力），这是一个上下文管理框架，为指令和情景交互分配不同的持久性与表示策略。指令记忆（Instructional Memory）在专门的前缀中保留已识别的全局约束。情景记忆保留用户输入并压缩模型回复，同时启发式检索为每个历史轮次选择原始文本、压缩表示或省略处理。在Long-MT-Bench+上，REA将评分从6.32提升至7.36（10分制），相比Vanilla基线相对提升16.5%，并将平均延迟降低2.91倍。额外的评估显示，在三个主干模型上均获得了总体性能提升。

    arXiv:2512.06869v2 Announce Type: replace  Abstract: Large language models often lose track of persistent instructions and relevant information as multi-turn conversations grow. We study this cumulative contextual decay through three related failure modes: attention pollution, dilution, and drift. We propose REA(Role-aware Heuristic Episodic Attention), a context-management framework that assigns different persistence and representation policies to instructions and episodic interactions. Instructional Memory retains identified global constraints in a dedicated prefix. Episodic Memory preserves user inputs and compresses model replies, while heuristic retrieval selects raw text, compressed representations, or omission for each historical turn. On Long-MT-Bench+, REA improves the judge score from 6.32 to 7.36 on a 10-point scale, a 16.5% relative gain over the Vanilla baseline, and reduces average latency by 2.91x. Additional evaluations show aggregate gains on three backbones spanning 1
    
[^358]: 思考视差：通过基于知识图谱的多视图检索增强生成解决多跳问题

    Think Parallax: Solving Multi-Hop Problems via Multi-View Knowledge-Graph-Based Retrieval-Augmented Generation

    [https://arxiv.org/abs/2510.15552](https://arxiv.org/abs/2510.15552)

    ParallaxRAG 发现 Transformer 注意力头天然形成与推理跳数对齐的多视图接力结构，并提出对称多视图框架将查询与知识图谱解耦到各注意力头专属的语义空间中，从而构建更精准干净的子图以提升大模型的多跳推理能力。

    

    大型语言模型（LLMs）在知识图谱（KGs）上的多跳推理仍然存在困难，我们识别出这一困难的一个此前被忽视的结构性原因：Transformer 的注意力头在推理的各个阶段会自然地专注于不同的语义关系，形成与推理跳数对齐的接力模式。这一关键发现表明，多跳推理本质上是多视图的，然而现有的基于知识图谱的检索增强生成（KG-RAG）系统却将所有推理跳都压缩到单一表示的平坦嵌入空间中，抑制了这种隐式结构，导致路径探索出现噪声或漂移。我们提出 ParallaxRAG，一个对称的多视图框架，它将查询和知识图谱解耦到对齐的、特定于注意力头的语义空间中。通过在多个注意力头上强制实现关系多样性，同时约束弱相关路径，ParallaxRAG 构建出更准确、更干净的子图，并引导 LLMs 进行更可靠的多跳推理。

    arXiv:2510.15552v5 Announce Type: replace-cross  Abstract: Large language models (LLMs) still struggle with multi-hop reasoning over knowledge-graphs (KGs), and we identify a previously overlooked structural reason for this difficulty: Transformer attention heads naturally specialize in distinct semantic relations across reasoning stages, forming a hop-aligned relay pattern. This key finding suggests that multi-hop reasoning is inherently multi-view, yet existing KG-based retrieval-augmented generation (KG-RAG) systems collapse all reasoning hops into a single representation, flat embedding space, suppressing this implicit structure and causing noisy or drifted path exploration. We introduce ParallaxRAG, a symmetric multi-view framework that decouples queries and KGs into aligned, head-specific semantic spaces. By enforcing relational diversity across multiple heads while constraining weakly related paths, ParallaxRAG constructs more accurate, cleaner subgraphs and guides LLMs through 
    
[^359]: 分层对齐：面向大语言模型偏好调优的块级LoRA-DPO方法

    Hierarchical Alignment: Block-Level LoRA-DPO for Preference Tuning in Large Language Models

    [https://arxiv.org/abs/2510.12044](https://arxiv.org/abs/2510.12044)

    该论文提出分层对齐方法，将LoRA-DPO的更新限制在解码器中间深度的连续块内而非所有层，在可训练参数量完全相同的情况下取得了更低的偏好优化损失，证明了适配器放置位置对大语言模型偏好调优效果的关键影响。

    

    偏好调优通常将适配器更新分布在每一个Transformer层上。我们研究了固定LoRA预算的放置位置如何影响直接偏好优化（DPO）。分层对齐方法将解码器深度划分为局部、中间和全局三个区域，并将LoRA更新限制在选定的连续块内。在Qwen和Llama模型上的三种子滑动窗口扫描实验显示，中间深度附近存在广泛的低损失盆地。在可训练参数精确匹配的条件下，在Qwen2.5-7B和Llama-3.1-8B上，中间深度配置比全深度分配取得了更低的留出集DPO损失。生成层面的证据则更具变异性：LLM评判和盲测人类研究在Qwen上无法在Selected与All之间得出定论；在Llama上，两个成对评判者偏好All，而人类评估的置信区间包含零。对训练后适配器的干预实验表明，学习到的深度组件会影响优化目标；移除Llama的Selected窗口也会……（摘要原文在此处截断）

    arXiv:2510.12044v2 Announce Type: replace-cross  Abstract: Preference tuning typically distributes adapter updates across every Transformer layer. We study how the placement of a fixed LoRA budget affects Direct Preference Optimization (DPO). Hierarchical Alignment partitions decoder depth into Local, Intermediate, and Global regions and restricts LoRA updates to a chosen contiguous block. Across Qwen and Llama models, three-seed sliding-window sweeps reveal broad low-loss basins around intermediate depth. Under exact trainableparameter matching, Middle achieves lower held-out DPO loss than an all-depth allocation on Qwen2.5-7B and Llama-3.1-8B. Generationlevel evidence is more variable. LLM judges and a blinded human study leave Selected versus All unresolved on Qwen; on Llama, two pairwise judges favor All while the human interval includes zero. Trained-adapter interventions show that learned depth components affect the optimization objective; removing the Llama Selected window also 
    
[^360]: 通过多路径规划聚合推进测试时扩展的帕累托前沿

    Advancing the Pareto Frontier of Test-Time Scaling with Multi-Path Plan Aggregation

    [https://arxiv.org/abs/2510.11620](https://arxiv.org/abs/2510.11620)

    提出多路径规划聚合方法 MPPA，在冻结主干上通过独立 LoRA 适配器学习联合规划与答案选择，生成互补的多路径规划以引导多样化求解并智能评分选优，从而推进测试时扩展的帕累托前沿。

    

    测试时扩展旨在通过额外的推理时计算来提升推理能力，通常采用解采样结合多数投票的方式来选择最终答案。然而，重复采样可能重现相关联的错误，限制了额外计算带来的准确性收益。我们提出了多路径规划聚合（MPPA），它在共享的冻结主干上使用独立的 LoRA 适配器来学习规划与答案选择。规划器联合生成 K 个简短规划，每个规划指导由冻结执行器生成的一个独立解；随后聚合器对这些解进行评分并选出最终答案。联合规划鼓励与执行器能力相匹配的互补性方法，而学习到的选择机制则利用推理证据来识别正确的候选解。交替优化通过监督微调和强化学习来改进规划与聚合。在五个数学（基准上……摘要在此处截断）

    arXiv:2510.11620v3 Announce Type: replace  Abstract: Test-time scaling seeks to improve reasoning through additional inference-time computation, commonly using solution sampling with majority voting to select the final answer. However, repeated sampling can reproduce correlated errors, limiting the accuracy gains from additional computation. We introduce Multi-Path Plan Aggregation (MPPA), which learns planning and answer selection using separate LoRA adapters on a shared frozen backbone. The planner jointly generates $K$ short plans, each guiding a separate solution generated by the frozen executor; the aggregator then scores the resulting solutions and selects the final answer. Joint planning encourages complementary approaches suited to the executor's capabilities, while learned selection uses reasoning evidence to identify correct candidates. Alternating optimization improves planning and aggregation through supervised finetuning and reinforcement learning. Across five mathematics 
    
[^361]: 电路蒸馏

    Circuit Distillation

    [https://arxiv.org/abs/2509.25002](https://arxiv.org/abs/2509.25002)

    提出了一种名为“电路蒸馏”的新方法，通过匹配教师与学生模型中功能对应的电路组件并对齐其内部表示，实现了底层计算机制的蒸馏，在仅调整少量参数的情况下成功迁移算法能力，性能优于传统的行为模仿式标准蒸馏。

    

    模型蒸馏通常侧重于行为模仿，即训练学生模型复现教师模型的输出，而将教师模型的内部计算视为黑箱。在这项工作中，我们提出了一种替代方法：蒸馏教师模型所实现的底层计算机制。具体而言，我们提出了电路蒸馏，该方法引入了一个目标，用于对齐教师模型和学生模型中相似电路组件的内部表示。我们提出了一种匹配“功能对应”电路组件的方法，并引入了一个反映这些组件所诱导表示之间相似性的损失函数。我们在Llama3系列的模型上，针对实体跟踪和心理理论（ToM）任务评估了电路蒸馏。我们的结果表明，电路蒸馏优于标准蒸馏，仅需调整少量（参数）即可成功迁移算法能力。

    arXiv:2509.25002v2 Announce Type: replace  Abstract: Model distillation typically focuses on behavioral mimicry, where a student model is trained to replicate a teacher's output while treating its internal computations as a black box. In this work we propose an alternative approach: Distilling the underlying computational mechanisms implemented by a teacher model. Specifically, we propose circuit distillation, which introduces an objective to align internal representations between analogous circuit components in teacher and student models. We propose a method to match ``functionally correspondent'' circuit components and introduce a loss reflecting similarities between the representations that these induce. We evaluate circuit distillation on entity tracking and theory of mind (ToM) tasks using models from the Llama3 family. Our results demonstrate that circuit distillation outperforms standard distillation, successfully transferring algorithmic capabilities by adjusting only a small, 
    
[^362]: 面向大语言模型增强多轮交互的测试时策略自适应

    Test-Time Policy Adaptation for Enhanced Multi-Turn Interactions with LLMs

    [https://arxiv.org/abs/2509.23166](https://arxiv.org/abs/2509.23166)

    该论文提出了测试时策略自适应新范式T2PAM及其轻量级实现算法ROSA，利用对话中的实时用户反馈作为奖励信号，通过单步参数更新使大语言模型在多轮交互中实现高效的自我纠错。

    

    arXiv:2509.23166v4 公告类型：替换 摘要：大语言模型（LLM）将多轮交互作为完成复杂任务的基本范式。然而，由于它们通常是在静态的单轮数据上训练的，其性能在长程交互中往往会下降，这阻碍了其适应实时用户反馈的能力。为了解决这一局限性，我们首先提出了一个新范式：面向多轮交互的测试时策略自适应（T2PAM），该范式利用正在进行的交互中的用户反馈作为奖励信号，估计与用户偏好一致的潜在最优策略，然后更新一小部分参数，将模型引导向该策略，最终实现对话内高效的自我纠错。随后，我们引入了最优参考单步自适应算法（ROSA），这是一种将T2PAM付诸实践的轻量级算法。ROSA通过单次高效的更新步骤将模型参数引导向理论最优策略，避免了……

    arXiv:2509.23166v4 Announce Type: replace  Abstract: Large Language Models (LLMs) employ multi-turn interaction as a fundamental paradigm for completing complex tasks. However, their performance often degrades in extended interactions, as they are typically trained on static, single-turn data, which hinders their ability to adapt to real-time user feedback. To address this limitation, we first propose a new paradigm: Test-Time Policy Adaptation for Multi-Turn Interactions (T2PAM), which utilizes user feedback from the ongoing interaction as a reward signal to estimate a latent optimal policy aligned with user preferences, then updates a small subset of parameters to steer the model toward this policy, ultimately enabling efficient in-conversation self-correction. We then introduce Optimum-Referenced One-Step Adaptation (ROSA), a lightweight algorithm that operationalizes T2PAM. ROSA guides the model parameters toward a theoretical optimal policy in a single, efficient update step, avoi
    
[^363]: SimulSense：语义单元驱动的口译方法，实现高效同声语音翻译

    SimulSense: Sense-Driven Interpreting for Efficient Simultaneous Speech Translation

    [https://arxiv.org/abs/2509.21932](https://arxiv.org/abs/2509.21932)

    提出SimulSense框架，通过模拟人类口译员持续读取输入语音并在感知到新语义单元时触发翻译输出，实现了更优的质量-延迟权衡，且决策速度比现有最先进基线快达9.6倍。

    

    同声语音翻译（SimulST）系统如何做出类似人类口译员的读/写决策？当前最先进的系统将SimulST构建为多轮对话任务，这需要专门的交错式训练数据，并依赖计算成本高昂的大语言模型（LLM）推理来进行决策。在本文中，我们提出了SimulSense，这是一种新颖的SimulST框架，它通过持续读取输入语音，并在感知到新的语义单元时触发写入决策以生成翻译，从而模拟人类口译员的工作方式。针对两个最先进基线系统的实验表明，我们提出的方法实现了更优的质量-延迟权衡，并显著提升了实时效率，其决策速度比基线系统快达9.6倍。

    arXiv:2509.21932v3 Announce Type: replace  Abstract: How to make human-interpreter-like read/write decisions for simultaneous speech translation (SimulST) systems? Current state-of-the-art systems formulate SimulST as a multi-turn dialogue task, requiring specialized interleaved training data and relying on computationally expensive large language model (LLM) inference for decision-making. In this paper, we propose SimulSense, a novel framework for SimulST that mimics human interpreters by continuously reading input speech and triggering write decisions to produce translation when a new sense unit is perceived. Experiments against two state-of-the-art baseline systems demonstrate that our proposed method achieves a superior quality-latency tradeoff and substantially improved real-time efficiency, where its decision-making is up to 9.6x faster than the baselines.
    
[^364]: KnowMT-Bench：面向多轮对话中知识密集型长文本问答能力的基准测试

    KnowMT-Bench: Benchmarking Knowledge-Intensive Long-Form Question Answering in Multi-Turn Dialogues

    [https://arxiv.org/abs/2509.21856](https://arxiv.org/abs/2509.21856)

    提出了KnowMT-Bench——首个用于动态评估大语言模型在医学、金融、法律等知识密集型领域多轮长文本问答能力的基准，实验揭示多轮对话会导致模型事实能力和信息传递效率显著下降。

    

    arXiv:2509.21856v2 公告类型：替换 摘要：多轮长文本问答（MT-LFQA）是大型语言模型（LLM）在知识密集型领域中的一个关键应用范式。然而，现有的基准测试仅限于单轮对话，而多轮对话基准通常评估的是其他正交能力，而非知识密集型的事实准确性。为了填补这一关键空白，我们提出了KnowMT-Bench，这是一个新的基准，旨在系统地评估大型语言模型在医学、金融和法律等知识密集型领域中的多轮长文本问答能力。为了忠实评估模型的真实表现，KnowMT-Bench采用了一种动态评估设置，即让模型在逻辑递进的问题序列下自行生成多轮对话历史。我们在多种大型语言模型上的实验表明，模型在多轮对话中的事实能力和信息传递效率均出现明显下降。我们进一步探究了潜在的……

    arXiv:2509.21856v2 Announce Type: replace  Abstract: Multi-Turn Long-Form Question Answering (MT-LFQA) is a key application paradigm of Large Language Models (LLMs) in knowledge-intensive domains. However, existing benchmarks are limited to single-turn dialogue, while multi-turn dialogue benchmarks typically assess other orthogonal capabilities rather than knowledge-intensive factuality. To bridge this critical gap, we introduce \textbf{KnowMT-Bench}, a new benchmark designed to systematically evaluate MT-LFQA for LLMs across knowledge-intensive fields, including medicine, finance, and law. To faithfully assess the model's real-world performance, KnowMT-Bench employs a dynamic evaluation setting where models generate their own multi-turn dialogue histories given logically progressive question sequences. Our experiments on a diverse suite of LLMs show a clear degradation in both factual capability and information delivery efficiency in multi-turn dialogue. We further probe the underlyin
    
[^365]: Automotive-ENV：在汽车座舱环境中对多模态模型进行基准测试

    Automotive-ENV: Benchmarking Multimodal Models in Automotive Cockpit Environments

    [https://arxiv.org/abs/2509.21143](https://arxiv.org/abs/2509.21143)

    该论文提出了Automotive-ENV，首个基于Android Automotive OS的交互式基准测试，通过420个涵盖日常座舱操作与安全关键场景的任务，结合车辆状态信号和可复现的状态验证器，系统评估多模态模型在汽车座舱环境中的能力。

    

    多模态模型已在网页、桌面和移动GUI环境中展现出强大的能力，但它们在汽车座舱系统中的性能在很大程度上仍未被探索。车载GUI引入了超越传统GUI交互的独特挑战，包括车辆状态推理、隐式意图理解和安全关键决策。为了填补这一空白，我们推出了Automotive-ENV，这是一个用于评估多模态模型在汽车座舱环境中表现的交互式基准测试。该基准基于Android Automotive OS构建，将座舱屏幕交互与车辆状态信号及基于状态的确定性评估相结合。该基准包含420个精心构建的任务，涵盖日常座舱操作和安全关键场景。与以往依赖动作匹配或基于大语言模型评判的GUI基准不同，Automotive-ENV通过对UI的可复现验证器来评估模型行为。

    arXiv:2509.21143v3 Announce Type: replace-cross  Abstract: Multimodal models have demonstrated strong capabilities in web, desktop, and mobile GUI environments, but their performance in automotive cockpit systems remains largely unexplored. In-vehicle GUIs introduce unique challenges beyond traditional GUI interaction, including vehicle-state reasoning, implicit intent understanding, and safety-critical decision making. To address this gap, we introduce Automotive-ENV, an interactive benchmark for evaluating multimodal models in automotive cockpit environments. Built on Android Automotive OS, Automotive-ENV combines cockpit-screen interaction with vehicle-state signals and deterministic state-based evaluation. The benchmark contains 420 carefully constructed tasks covering routine cockpit operation and safety-critical scenarios. Unlike prior GUI benchmarks that rely on action matching or LLM-based judging, Automotive-ENV evaluates model behavior through reproducible validators over UI 
    
[^366]: 无需标签的语言模型进化：多数投票驱动选择，新颖性促进变异

    Evolving Language Models without Labels: Majority Drives Selection, Novelty Promotes Variation

    [https://arxiv.org/abs/2509.15194](https://arxiv.org/abs/2509.15194)

    该论文提出 EVOL-RL 方法，用多数投票提供稳定选择信号、基于嵌入的新颖性促进变异并辅以熵正则化，解决了无标签自我改进中“选择诱导的多样性坍缩”问题，使大语言模型在缺乏参考答案时也能持续进化。

    

    大型语言模型（LLM）可以通过带可验证奖励的强化学习（RLVR）来提升，但在许多场景中缺乏参考答案或外部验证器。无标签自我改进则转而从模型自身的输出中构建训练信号。由于同一个策略既生成候选响应，又生成由这些响应导出的信号，每次更新都会改变未来训练可用的候选集，从而形成一个自我强化的反馈循环。我们识别了这一循环的一种失效模式，称之为“选择诱导的多样性坍缩”：反复的自我选择会逐渐缩小候选池，使得其他正确解更难被采样到。为解决这一问题，我们提出 EVOL-RL，在选择与变异之间取得平衡：多数投票提供稳定的选择信号，基于嵌入的新颖性度量则在相同选择状态的响应之间细化信用分配，熵正则化（摘要原文在此处被截断）

    arXiv:2509.15194v4 Announce Type: replace-cross  Abstract: Large language models (LLMs) can improve through reinforcement learning with verifiable rewards (RLVR), but many settings lack reference answers or external verifiers. Label-free self-improvement instead constructs training signals from the model's own outputs. Because the same policy generates both the candidate responses and the signals derived from them, each update also changes the candidates available for future training, creating a self-reinforcing feedback loop. We identify a failure mode of this loop, which we call selection-induced diversity collapse: repeated self-selection can progressively narrow the candidate pool and make alternative correct solutions harder to sample. To address this problem, we propose EVOL-RL, which balances selection and variation. Majority voting provides a stable selection signal, while embedding-based novelty refines credit among responses with the same selection status; entropy regularizat
    
[^367]: 大语言模型的概率无法可靠地区分可能语言与不可能语言

    Large Language Model probabilities cannot reliably distinguish between possible and impossible language

    [https://arxiv.org/abs/2509.15114](https://arxiv.org/abs/2509.15114)

    本研究通过对8个大语言模型内部概率的分析发现，模型的字符串概率无法可靠地区分可能语言与不可能语言，质疑了将语言模型概率作为语法判断代理指标的可行性。

    

    关于大语言模型的一项有争议的测试涉及其辨别可能语言与不可能语言的能力。虽然有一些证据表明模型对超出语法不可能语言边界的内容具有敏感性，但这些证据因测试材料的有效性问题而受到质疑。我们使用模型内部表征作为直接手段，来研究大语言模型如何表征合语法与不合语法的区分。在一个新颖的基准测试中，我们从8个模型中获取总对数概率，并计算最小对立对的惊异度差异，将分配给合语法句子的概率与分配给以下四类句子的概率进行对比：(i) 低频合语法句子、(ii) 不合语法句子、(iii) 语义异常句子和(iv) 语用异常句子。研究预测是，如果字符串概率能够作为语法边界的代理指标，那么不合语法句子的……

    arXiv:2509.15114v2 Announce Type: replace  Abstract: A controversial test for Large Language Models concerns the ability to discern possible from impossible language. While some evidence attests to the models sensitivity to what crosses the limits of grammatically impossible language, this evidence has been contested on the grounds of the soundness of the testing material. We use model-internal representations as a direct means of investigating the way Large Language Models represent the grammatical-ungrammatical distinction. In a novel benchmark, we elicit sum log probabilities from 8 models and compute minimal-pair surprisal differences, juxtaposing probabilities assigned to grammatical sentences to probabilities assigned to (i) lower frequency grammatical sentences, (ii) ungrammatical sentences, (iii) semantically odd sentences, and (iv) pragmatically odd sentences. The prediction is that if string-probabilities can function as proxies for the limits of grammar, the ungrammatical co
    
[^368]: 多模态（推理）大语言模型能够检测文档篡改吗？

    Can Multi-modal (reasoning) LLMs detect document manipulation?

    [https://arxiv.org/abs/2508.11021](https://arxiv.org/abs/2508.11021)

    本研究系统评估了包括 OpenAI O1、Gemini、Claude 等多种最先进多模态（推理）大语言模型在真实交易文档上检测欺诈篡改的能力，发现顶尖模型展现出卓越的零样本泛化能力。

    

    文档欺诈对依赖安全且可验证文件的行业构成重大威胁，因此需要强大的检测机制。本研究考察了最先进的多模态大语言模型（LLMs）——包括 OpenAI O1、OpenAI 4o、Gemini Flash（thinking）、Deepseek Janus、Grok、Llama 3.2 和 4、Qwen 2 和 2.5 VL、Mistral Pixtral，以及 Claude 3.5 和 3.7 Sonnet——在检测欺诈文档方面的效能。我们使用包含真实交易文档的标准数据集，对这些模型进行相互基准测试，并与以往文档欺诈检测技术的研究进行比较。通过提示词优化以及对模型推理过程的详细分析，我们评估了它们识别细微欺诈指标的能力，例如被篡改的文本、错位的格式以及不一致的交易金额。我们的结果表明，表现最优的多模态大语言模型展现出卓越的零样本泛化能力……（原文摘要在此处截断）

    arXiv:2508.11021v2 Announce Type: replace-cross  Abstract: Document fraud poses a significant threat to industries reliant on secure and verifiable documentation, necessitating robust detection mechanisms. This study investigates the efficacy of state-of-the-art multi-modal large language models (LLMs)-including OpenAI O1, OpenAI 4o, Gemini Flash (thinking), Deepseek Janus, Grok, Llama 3.2 and 4, Qwen 2 and 2.5 VL, Mistral Pixtral, and Claude 3.5 and 3.7 Sonnet-in detecting fraudulent documents. We benchmark these models against each other and prior work on document fraud detection techniques using a standard dataset with real transactional documents. Through prompt optimization and detailed analysis of the models' reasoning processes, we evaluate their ability to identify subtle indicators of fraud, such as tampered text, misaligned formatting, and inconsistent transactional sums. Our results reveal that top-performing multi-modal LLMs demonstrate superior zero-shot generalization, ou
    
[^369]: 鲁棒且细粒度的AI生成文本检测

    Robust and Fine-Grained Detection of AI Generated Texts

    [https://arxiv.org/abs/2504.11952](https://arxiv.org/abs/2504.11952)

    本文提出一组基于词元分类的模型，能够鲁棒且细粒度地检测AI生成及人机共同创作的文本，并发布了涵盖23种语言、超过240万条文本的新数据集。

    

    理想的机器生成内容检测系统应当能够在任何生成器上都表现良好，因为日益先进的LLM每天都在不断涌现。现有系统往往难以准确识别较短文本中的AI生成内容。此外，并非所有文本都完全由人类或大语言模型撰写，因此我们更加关注部分生成的情况，即人类与大语言模型共同创作的文本。本文介绍了一组为词元分类任务构建的模型，这些模型在海量人类与机器共同创作的文本集合上进行训练，在未见过的领域、未见过的生成器、非母语者撰写的文本以及含对抗性输入的文本上均表现良好。我们还引入了一个全新的数据集，包含超过240万条此类文本，其中大部分由多个流行的专有大语言模型以23种语言共同创作而成。我们还展示了模型在各领域和各生成器的每篇文本上的性能表现。

    arXiv:2504.11952v4 Announce Type: replace-cross  Abstract: An ideal detection system for machine generated content is supposed to work well on any generator as many more advanced LLMs come into existence day by day. Existing systems often struggle with accurately identifying AI-generated content over shorter texts. Further, not all texts might be entirely authored by a human or LLM, hence we focused more over partial cases i.e human-LLM co-authored texts. Our paper introduces a set of models built for the task of token classification which are trained on an extensive collection of human-machine co-authored texts, which performed well over texts of unseen domains, unseen generators, texts by non-native speakers and those with adversarial inputs. We also introduce a new dataset of over 2.4M such texts mostly co-authored by several popular proprietary LLMs over 23 languages. We also present findings of our models' performance over each texts of each domain and generator. Additional findin
    
[^370]: 翻译工具与技术培训：2023年EMT调查结果

    Training in translation tools and technologies: Findings of the EMT survey 2023

    [https://arxiv.org/abs/2503.22735](https://arxiv.org/abs/2503.22735)

    2023年EMT调查显示，翻译技术教学正快速响应行业创新：机器翻译、译后编辑与质量评估日益成为必修内容，培养项目对生成式AI工具反应迅速，并趋向以免费学术授权的云端软件为核心整合教学体系。

    

    本文报告了针对研究生翻译培养项目中教授的计算机化工具与技术的第三次调查结果。尽管该调查是在EMT网络的支持下开展的，但超过半数的回复来自该网络之外。结果显示，各培养项目对翻译技术创新反应迅速，机器翻译、译后编辑和质量评估越来越多地被纳入必修内容，并对生成式工具的发布做出了快速响应。新冠疫情期间所需的灵活性也给培养项目带来了一些持久性的变化。尽管所教授的工具范围不断扩大，各项目似乎正在围绕可免费获取学术授权的云端软件来整合其核心教学内容。此外，与翻译技术相关的专业情境和工作流程在教学中的融入程度也有所提升。

    arXiv:2503.22735v2 Announce Type: replace-cross  Abstract: This article reports on the third iteration of a survey of computerized tools and technologies taught as part of postgraduate translation training programmes. While the survey was carried out under the aegis of the EMT Network, more than half of responses are from outside that network. The results show the responsiveness of programmes to innovations in translation technology, with increased compulsory inclusion of machine translation, post-editing, and quality evaluation, and a rapid response to the release of generative tools. The flexibility required during the Covid-19 pandemic has also led to some lasting changes to programmes. While the range of tools being taught has continued to expand, programmes seem to be consolidating their core offering around cloud-based software with cost-free academic access. There has also been an increase in the embedding of professional contexts and workflows associated with translation techno
    
[^371]: 评估基于大语言模型的多轮对话智能体：一项综述

    Evaluating LLM-based Agents for Multi-Turn Conversations: A Survey

    [https://arxiv.org/abs/2503.22458](https://arxiv.org/abs/2503.22458)

    本综述基于近250篇文献，提出了两个相互关联的分类体系——“评估什么”与“如何评估”——为多轮对话中基于大语言模型的智能体评估提供了系统性的结构化框架。

    

    本综述考察了在多轮对话环境中对基于大语言模型（LLM）的智能体的评估方法。我们采用受PRISMA启发的框架，系统地回顾了近250篇学术文献，涵盖了各类出版渠道的最新研究进展，为我们的分析奠定了坚实的基础。我们的研究通过构建两个相互关联的分类体系提供了结构化的方法：一个定义了“评估什么”，另一个解释了“如何评估”。第一个分类体系识别了面向多轮对话的LLM智能体的关键组件及其评估维度，包括任务完成度、响应质量、用户体验、记忆与上下文保持，以及规划与工具集成。这些组件确保以全面且有意义的方式评估对话智能体的性能。第二个分类体系则聚焦于评估（原文在此处截断）……

    arXiv:2503.22458v3 Announce Type: replace-cross  Abstract: This survey examines evaluation methods for large language model (LLM)-based agents in multi-turn conversational settings. Using a PRISMA-inspired framework, we systematically reviewed nearly 250 scholarly sources, capturing the state of the art from various venues of publication, and establishing a solid foundation for our analysis. Our study offers a structured approach by developing two interrelated taxonomy systems: one that defines \emph{what to evaluate} and another that explains \emph{how to evaluate}. The first taxonomy identifies key components of LLM-based agents for multi-turn conversations and their evaluation dimensions, including task completion, response quality, user experience, memory and context retention, as well as planning and tool integration. These components ensure that the performance of conversational agents is assessed in a holistic and meaningful manner. The second taxonomy system focuses on the eval
    
[^372]: Auto-AdvER：解耦汽车广告中的车况、历史与商业报价

    Auto-AdvER: Disentangling Condition, History, and Commercial Offers in Automotive Advertisements

    [https://arxiv.org/abs/2412.05655](https://arxiv.org/abs/2412.05655)

    本文提出了面向汽车广告领域的专用跨度标注模式Auto-AdvER及配套数据集，通过“车况”、“历史”和“销售选项”三个标签解耦广告中的关键信息，并系统比较了Transformer编码器、大语言模型及条件随机场等多种模型的标注性能，标注者间一致性达78%。

    

    本文介绍了一项关于Auto-AdvER开发的案例研究，这是一种专门面向汽车文本广告领域的跨度标注模式及其对应数据集。Auto-AdvER以行业需求为导向开发，旨在增强该领域的文本挖掘分析能力，并提供了一个语言学上独特的序列标注数据集。我们提出了一个由三个标签组成的标注模式：“车况”、“历史”和“销售选项”。我们阐述了标注的指导原则，描述了标注模式的开发方法，并展示了一项标注研究的结果，标注者间一致性达到78%。此外，我们比较了多种模型的性能，包括Transformer编码器模型（BERT、DeBERTaV3）、基于Transformer解码器的开源和闭源大语言模型（Llama、Qwen、GPT和Gemini），以及条件随机场。我们的结果表明，经过微调的……

    arXiv:2412.05655v2 Announce Type: replace  Abstract: This paper presents a case study on the development of Auto-AdvER, a specialised span-labelling schema, along with its corresponding dataset for the textual car advertisement domain. Developed with industry needs in mind, Auto-AdvER is designed to enhance text mining analytics in this domain and contributes a linguistically unique sequence-labelling dataset. We introduce a schema consisting of three labels: 'Condition', 'Historic', and 'Sales Options'. We outline the guiding principles for annotation, describe the methodology for schema development, and present the results of an annotation study demonstrating an inter-annotator agreement of 78%. Furthermore, we compare the performance of various models, including transformer encoder models (BERT, DeBERTaV3) and transformer decoder-based open- and closed-source large language models: Llama, Qwen, GPT, and Gemini, as well as conditional random fields. Our results show that a fine-tuned
    
[^373]: 人工利维坦：透过霍布斯社会契约论的视角探索大语言模型智能体的社会演化

    Artificial Leviathan: Exploring Social Evolution of LLM Agents Through the Lens of Hobbesian Social Contract Theory

    [https://arxiv.org/abs/2406.14373](https://arxiv.org/abs/2406.14373)

    该论文构建了一个赋予心理驱力的LLM智能体沙盒模拟社会，并以霍布斯社会契约论为分析框架，实验发现智能体从最初类似“自然状态”的无序冲突逐步演化出社会契约，为计算社会科学验证社会理论提供了新范式。

    

    大语言模型（LLM）的出现和人工智能（AI）的进步为大规模计算社会科学研究提供了机遇。在先前LLM智能体设计探索的基础上，我们的工作引入了一个模拟的智能体社会，其中复杂的社会关系会随时间动态地形成和演化。智能体被赋予心理驱力，并被置于一个沙盒式的生存环境中。我们通过托马斯·霍布斯的开创性社会契约论（SCT）的视角对智能体社会进行评估，分析是否如该理论所假设的那样，智能体会试图通过将权利让渡给一个绝对的君主以换取秩序与安全，从而逃离野蛮的“自然状态”。我们的实验揭示了两者的契合：起初，智能体陷入不受约束的冲突之中，这与霍布斯对自然状态的描绘如出一辙；然而，随着模拟的推进，社会契约逐渐涌现，从而导致……

    arXiv:2406.14373v4 Announce Type: replace  Abstract: The emergence of Large Language Models (LLMs) and advancements in Artificial Intelligence (AI) offer an opportunity for computational social science research at scale. Building upon prior explorations of LLM agent design, our work introduces a simulated agent society where complex social relationships dynamically form and evolve over time. Agents are imbued with psychological drives and placed in a sandbox survival environment. We conduct an evaluation of the agent society through the lens of Thomas Hobbes's seminal Social Contract Theory (SCT). We analyze whether, as the theory postulates, agents seek to escape a brutish "state of nature" by surrendering rights to an absolute sovereign in exchange for order and security. Our experiments unveil an alignment: Initially, agents engage in unrestrained conflict, mirroring Hobbes's depiction of the state of nature. However, as the simulation progresses, social contracts emerge, leading to
    
[^374]: 基于仿真优化的预算约束下语言模型提示词集合扩展与精炼

    Budgeted Prompt-Set Expansion and Refinement for Language Models via Simulation Optimization

    [https://arxiv.org/abs/2404.08164](https://arxiv.org/abs/2404.08164)

    本文提出了一种基于仿真优化的两阶段框架，在有限评估预算下将提示词表示为中等维度向量并构建得分代理模型，从而实现语言模型提示词集合的高效扩展与精炼以最大化任务得分。

    

    提示词是用户提供的指令或描述，用于引导生成式语言模型产生满足应用需求的输出。当这些需求发生变化时，现有提示词可能需要被调整和改进。评估候选提示词需要耗费时间和计算资源。因此，我们研究如何在有限的评估预算下，通过仿真优化方法对初始提示词集合进行扩展与精炼，目标是在固定语言模型上最大化预先定义的任务得分。具体而言，我们提出了一个两阶段框架：在第一阶段，将初始提示词集合扩展为有限的候选集合，其中每个提示词由一个中等维度的向量表示；在随后的评估与选择阶段，为表示提示词的中等维度向量构建得分的代理模型。我们提出基于该代理模型依次选择提示词进行评估（原文摘要在此处截断）。

    arXiv:2404.08164v3 Announce Type: replace-cross  Abstract: A prompt is an instruction or description provided by the user to guide a generative language model toward outputs that meet application requirements. When these requirements change, existing prompts may need to be adapted and improved. Evaluating candidate prompts requires time and computational resources. We therefore study how to expand and refine an initial set of prompts under a limited evaluation budget through simulation optimization, aiming to maximize a pre-defined task score for a fixed language model. Specifically, we propose a two-stage framework. In the first stage, we expand the initial set of prompts into a finite candidate set, where each prompt is represented by a moderate-dimensional vector. In the subsequent stage of evaluation and selection, we construct a surrogate model of the score for moderate-dimensional vectors representing the prompts. We propose sequentially selecting prompts for evaluation using the
    

