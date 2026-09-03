# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [User Feedback Provides a Unique Signal that LLMs Can not Detect](https://arxiv.org/abs/2609.02859) | 用户反馈是LLM改进的高价值学习信号，其看似无效源于评估范式的系统性偏差——模型自身无法察觉反馈带来的改进，而实验证明基于反馈的修订能以显著更高的比率解决目标问题。 |
| [^2] | [Post-Training Language Models for Gold-Medal Performance in Coding Competitions](https://arxiv.org/abs/2609.02849) | 该研究通过结合大规模题目筛选、监督微调、强化学习以及反馈驱动的测试时计算策略 GenCorrect，使语言模型在 IOI 2025 编程竞赛中取得了超越金牌分数线（438.3 分）的成绩（Nano-CC 达 468 分，Ultra-CC 达 502 分）。 |
| [^3] | [Dutch Books for Language Models](https://arxiv.org/abs/2609.02797) | 该论文基于德·菲内蒂定理提出一种利用线性规划计算荷兰赌利润的评估方法，无需真实结果标签即可量化语言模型概率预测的不连贯性，并发现语言模型预测存在显著的不连贯现象。 |
| [^4] | [DiscoSign: Discourse-Aware Text to Sign Language Gloss Translation](https://arxiv.org/abs/2609.02796) | 提出了DiscoSign，一种基于大语言模型的语篇感知文本到手语注释翻译框架，通过处理空间共指消解、问答从句和概念-注释一致性三种语篇现象，并引入新颖的语篇连贯性评估指标，突破了传统句子级手语翻译的局限。 |
| [^5] | [EarlyEval: Cheaper Agent Evaluation via Early Outcome Prediction](https://arxiv.org/abs/2609.02783) | 该论文提出EarlyEval框架，通过基于智能体中间行为训练轻量级的成功/失败分类器来提前预测任务结果并及早中止运行，从而在单个任务内部削减成本，大幅降低LLM智能体的评估开销。 |
| [^6] | [HyperStyler: Low-resource Authorship Style Transfer via Context-aware Style Navigation and Hypernetworks](https://arxiv.org/abs/2609.02772) | HyperStyler将低资源作者风格迁移解耦为风格选择与风格实现两个阶段，通过上下文感知的风格导航器预测风格坐标，并利用超网络进行动态参数调制来避免风格与内容纠缠，从而同时实现高风格保真度与语义保留。 |
| [^7] | [From Reweighting to Rewriting: Unlocking the Intervention Effects of Influential Samples in Training Data Attribution](https://arxiv.org/abs/2609.02771) | 该论文发现重加权无法释放影响函数所识别样本的干预价值，并提出“影响引导的响应重写”方法——通过重写所选样本的响应而非调整其权重，从而真正解锁训练数据归因中影响力样本的干预效果。 |
| [^8] | [Untangling the Mechanisms of Misleading Context in Medical Question Answering](https://arxiv.org/abs/2609.02754) | 该研究通过在MedMisBench中注入伪造证据和纯粹断言两类误导性上下文，系统揭示了推理模型的医学判断被误导的机制，发现模型对纯粹断言的易感性显著高于伪造证据（高出10至27个百分点），且误导信息虽在推理轨迹中被大量披露却难以被察觉。 |
| [^9] | [Repo-To-Skill: Distilling GitHub Repositories Into AI4AI Skills](https://arxiv.org/abs/2609.02749) | 提出DisCo技能驱动型研究智能体，通过任务无关与任务导向两种蒸馏方式，将GitHub仓库中的操作性知识转化为紧凑、经过验证且可跨任务复用的AI4AI技能。 |
| [^10] | [Incremental Pooled LLM Evaluation for Cost-Effective Retrieval Model Selection](https://arxiv.org/abs/2609.02745) | 提出增量式池化LLM评估方法，通过LLM判断候选系统检索文档的并集并随新系统增量扩展文档池，实现低成本、可复用的检索模型对比评估，其排序结果与金标准评估高度一致。 |
| [^11] | [Language Models Can Control Their Own Attention](https://arxiv.org/abs/2609.02737) | 该论文提出“声明式注意力”协议，让语言模型在思维链中自主声明需要关注的上下文区域，推理引擎据此像解析工具调用一样跳过大部分KV缓存读取，从而以内在方式避免了外部评分方法每步O(N)的开销。 |
| [^12] | [Choosing a PEFT Variant for Per-Patient Dysarthric ASR: A Single-Speaker Case Study on Two ASR Bases](https://arxiv.org/abs/2609.02735) | 该单说话人案例研究比较了七种LoRA系列PEFT方法在两个ASR基础模型上的构音障碍语音识别表现，发现注意力投影适配器能显著降低字符错误率，且更简单廉价的LoRA与DoRA性能无显著差异，因而推荐采用LoRA。 |
| [^13] | [CORAL: An LLM-Native Harness for Production Recommender Systems](https://arxiv.org/abs/2609.02730) | CORAL是一个LLM原生闭环框架，让智能体持续观察线上推荐系统的运行信号、基于过往决策与结果的记忆进行推理并调用工具，从决策的实际效果中学习，从而实现生产级推荐系统的持续自动化优化。 |
| [^14] | [Door-in-the-Face Requests and Refusal Behaviour in Large Language Models](https://arxiv.org/abs/2609.02707) | 该研究发现“留面子”说服技术对不同大语言模型的效果截然不同：它在Anthropic前沿模型上能将较小请求的依从率从29.3%提升至65.8%，但在OpenAI和Google的前沿模型上反而使依从率降低15.5至23.0个百分点。 |
| [^15] | [Trace as State: Reasoning Traces as Conditional States for Long-Context Transformers](https://arxiv.org/abs/2609.02702) | 提出Trace as State方法，将推理轨迹作为任务状态的文本代理置于长上下文之前以指导模型重读，在27个模型-任务-指标组合中的26个上优于将轨迹置于上下文之后的对照方法。 |
| [^16] | [DKL: Decoupled Knowledge Learning for Instruction-Tuned Language Models](https://arxiv.org/abs/2609.02685) | 提出DKL解耦知识学习方法，能够在不损害指令遵循能力、也无需生成海量合成问答数据的情况下，将新语料库知识注入指令微调语言模型，从而缓解RAG在检索失败时的幻觉问题。 |
| [^17] | [From Tokens to Semantics: Leveraging Complementary Signals for Hallucination Detection in Black-Box LLMs](https://arxiv.org/abs/2609.02679) | 该论文针对无参考文档的黑盒大语言模型，提出联合利用语义熵与词元级不确定性这两种互补信号（包括TopK聚合、CoCoA混合方法及Gated等监督方法）来更准确地检测幻觉。 |
| [^18] | [oHC: Orthogonal Hyper-Connections on SO(4) via Quaternions](https://arxiv.org/abs/2609.02672) | 该论文证明了双随机矩阵约束的混合会随深度耗尽残差流的多样性，并提出通过四元数在SO(4)流形上构造正交混合矩阵的oHC方法，既保证缩放稳定又完整保持残差流的范数与多样性。 |
| [^19] | [WinoQueer-NL: Assessing Bias in Dutch Language Models toward LGBTQ+ Identities](https://arxiv.org/abs/2609.02651) | 该研究构建了首个评估荷兰语语言模型对LGBTQ+身份偏见的基准数据集WinoQueer-NL，通过与荷兰酷儿群体的调查验证了145个文化相关刻板印象并新发现22种偏见，揭示了看似中性的平均偏见得分背后隐藏的显著偏见。 |
| [^20] | [Loom: Weaving Diagnostic Strands into Free-Text Consensus via Embedding-Space Reweighting](https://arxiv.org/abs/2609.02649) | Loom是一个部署于真实工业根因分析的生成式共识框架，它将模块化启发式产生的开放式诊断假设投影到连续嵌入空间，并通过基于质心的迭代重加权算法解决冲突信号，从而把嘈杂矛盾的文本假设聚合为可靠共识。 |
| [^21] | [TaRA: Training-Aware Low-Rank Adaptation Initialization](https://arxiv.org/abs/2609.02639) | TaRA提出了一种训练感知的LoRA初始化方法，通过使低秩因子诱导的梯度密切逼近全秩权重矩阵的梯度来提升训练初期的梯度保真度，且几乎不增加计算开销。 |
| [^22] | [Scalable Direction-Following TTS via Voice Impression-Guided Pseudo Triplet Construction](https://arxiv.org/abs/2609.02623) | 提出一种利用印象可控语音合成模型与大语言模型自动构建（参考语音、方向文本、修改后语音）伪三元组的可扩展流水线，解决了方向跟随语音合成中训练数据稀缺的问题，仅凭伪数据即可实现稳定的说话人特征保留式风格修改。 |
| [^23] | [Predictors of Loneliness in Older Adults Using Multimodal Analysis of Speech and Language](https://arxiv.org/abs/2609.02606) | 本研究通过多模态分析310名老年人电话访谈中的语言特征和声学特征，发现高孤独感与更多使用否定词、负面语气及冲突相关语言相关，为自然对话情境下孤独感的客观、可扩展检测提供了新方法。 |
| [^24] | [When Persona Attributes Improve Population Alignment in Large Language Models](https://arxiv.org/abs/2609.02526) | 本文研究人格提示中属性选择对大语言模型人群对齐效果的影响，发现属性的选择比数量更关键，使用更多属性并不必然带来更好的性能。 |
| [^25] | [Debias-SparseGPT: Bias-Aware Pruning for Large Language Models](https://arxiv.org/abs/2609.02496) | 提出Debias-SparseGPT，一种在剪枝过程中利用人口统计学对比输入定义的二阶项进行表征去偏的后训练剪枝方法，能在保持模型困惑度和零样本准确率的同时，显著减少剪枝引发的偏见。 |
| [^26] | [ViSAR: Training-Free Adaptive-$k$ Retrieval for Visual Document Question Answering](https://arxiv.org/abs/2609.02486) | 提出了一种无需训练的自适应k值检索方法ViSAR，通过在嵌入空间中构建查询条件的页面级相似度矩阵来动态确定检索页面数量，在保持或提升答案准确性的同时将RAG延迟降低高达58.7%。 |
| [^27] | [How LLMs Build Fictional Worlds: Setting and Narrative Space in AI-Generated Creative Storytelling](https://arxiv.org/abs/2609.02482) | 该研究通过微调BERT分类器分析五种叙事空间类型，发现人类小说以角色与环境具身交互的“行动空间”为主，而大语言模型生成的故事则系统性偏向强调氛围情感的“感知空间”，且这一差异在叙事全程保持稳定。 |
| [^28] | [PragAlign: Feedback-Guided Pragmatic Alignment for Controlled Synthetic Dialogue Generation](https://arxiv.org/abs/2609.02480) | PragAlign是一个反馈引导的受控合成对话生成框架，通过“生成—评估—修改”循环利用基于LLM的评估器进行多维度评分与针对性反馈，将评估器接受率提升至99.50%，显著优于一次性生成（72.25%）和无结构化反馈的重复生成（95.88%）。 |
| [^29] | [Learning to Fuse LLMs with Ontology Rankers for Rare-Disease Diagnosis](https://arxiv.org/abs/2609.02473) | 该论文提出一种基于行为的融合模型，将大语言模型与本体排序器结合用于罕见病诊断，在保留证据可追溯性的同时，将 Phenomizer 的 Recall@1 分别提升 7.86 和 20.18 个百分点。 |
| [^30] | [Scalable Kronecker-Fisher Approximation: Efficient Hessian Analysis for Billion-Parameter Language Models Compression](https://arxiv.org/abs/2609.02451) | 本文提出一种可扩展的Kronecker-Fisher近似方法，无需存储完整Fisher矩阵即可对十亿参数语言模型进行高效Hessian分析，发现值投影层是最脆弱的组件，为混合精度分配等压缩与优化策略提供了实用的理论工具。 |
| [^31] | [When Decodability Is Not Enough: Logical Validity Representations, Behavioral Dissociation, and Causal Tests in Language Models](https://arxiv.org/abs/2609.02438) | 该研究发现即使大语言模型在逻辑验证任务上的行为表现接近随机，其隐藏状态中仍能近乎完美地解码出逻辑有效性信息，但因果干预显示这种表征并未被模型实际利用，揭示了“可解码性不等于因果性使用”这一重要结论。 |
| [^32] | [UTP-Bench: Uncertainty-aware Travel Planning Benchmark](https://arxiv.org/abs/2609.02421) | 该论文提出了 UTP-Bench——首个引入真实交通延误分布与人流密度等不确定性因素的大规模旅行规划基准，覆盖印度 504 个城市，用以评估大语言模型生成的行程在现实干扰下的稳健性。 |
| [^33] | [Before the Script, Set the Stage: How Worldview Simulation Amplifies Psychologically Grounded Persuasion in Multi-Turn Jailbreaking](https://arxiv.org/abs/2609.02414) | BLUEPRINT框架通过结合18个心理学影响因素的世界观模拟与蒙特卡洛树搜索，以最少查询次数在六个前沿大模型上实现接近100%的多轮越狱攻击成功率，并揭示了各模型共享的"转向具体可执行任务"这一越狱路径。 |
| [^34] | [Improving Health Literacy through Lay Summarization of Radiological Reports: An Evaluation of BioNER and Retrieval-Augmented Generation](https://arxiv.org/abs/2609.02396) | 本研究提出了一个将NER临床发现提取与RAG上下文锚定相结合的框架，证明这两种技术能够显著提升放射报告自动通俗摘要的质量、事实一致性和可读性，从而改善患者的健康素养。 |
| [^35] | [PolERo: Studying Political Evasion in Romanian](https://arxiv.org/abs/2609.02391) | 该论文提出首个罗马尼亚语政治回避检测数据集PolERo（包含来自五位罗马尼亚总统官方记录的3,574个人工标注问答对），并将分类体系与模型从英语扩展到新语言，同时通过双语联合训练和翻译增强研究了跨语言迁移能力。 |
| [^36] | [MultiGhostBench: A Multilingual Benchmark for Long-Form LLM-Generated Text Attribution under Distribution Shifts](https://arxiv.org/abs/2609.02379) | 本文提出了MultiGhostBench多语言基准，包含五个最新大语言模型在六种语言下生成的928本约59K词的长篇书籍，用于评估领域、作者和语言偏移下的LLM文本归因，发现没有单一方法始终最优且分布偏移会导致性能下降。 |
| [^37] | [NE-R1: Enhancing Named Entity Recognition Model via Reinforcement Learning](https://arxiv.org/abs/2609.02366) | 提出了NE-R1框架，通过“按需检索”机制和两阶段训练（多任务指令微调初始化与基于CoT的端到端强化学习优化），借助兼顾准确性与检索收益的多维奖励，在参数化知识与外部知识之间进行自适应选择，实现检索增强命名实体识别的最先进性能。 |
| [^38] | [SonicCaps: Large-Scale Diverse and Fine-Grained Captioning for Improved Audio-Retrieval](https://arxiv.org/abs/2609.02343) | 提出SonicCaps大规模音频字幕数据集，包含约1500万条字幕配对70万个音频片段，利用多模态大语言模型为每个音频生成约24条多样化、细粒度的字幕，有效克服了现有数据集语义多样性低和一对一映射的局限，显著提升音频检索性能。 |
| [^39] | [SALA: Semantic-Aware Logical Alignment for Complex Reasoning in In-Context Learning](https://arxiv.org/abs/2609.02336) | SALA框架通过自动学习任务特定的推理操作，并在连续语义空间中利用动态时间规整（DTW）实现推理序列的软性对齐，从而为复杂推理的上下文学习提供灵活且可解释的示例选择方法。 |
| [^40] | [Counter-GEO-Bench: Evaluating Defenses Against Information-Distorting Generative Engine Optimization](https://arxiv.org/abs/2609.02316) | 提出了首个针对生成式引擎优化（GEO）攻击的防御基准Counter-GEO-Bench，通过将247个经人工验证的查询与信息保留型和信息扭曲型GEO改写配对来评估防御方法，并揭示现有三种主流防御方法最多仅能将攻击成功率相对降低5.7%。 |
| [^41] | [DiffIE: Diffusion-based Open Information Extraction](https://arxiv.org/abs/2609.02315) | DIFFIE将条件离散扩散的随机性本身作为抽取机制，通过多条独立的反向扩散轨迹生成候选三元组池，实现抽取预算与训练的解耦，并在CaRB基准上取得新的最优性能。 |
| [^42] | [Efficient GUI Agents: A Systems Survey of Observation, Memory, Action, and Runtime Optimization](https://arxiv.org/abs/2609.02309) | 本文是首篇从端到端系统效率视角综述GUI智能体的工作，从观察、记忆、动作与运行时优化四个维度系统梳理了高效GUI智能体的主流机制与新兴开销。 |
| [^43] | [Improving Evaluation Realism with Inference-Time Compute and Deployment Scaffolds](https://arxiv.org/abs/2609.02302) | 该论文提出“批判式精炼”和 DISH 智能体框架两种技术，通过投入额外推理时计算并模仿真实部署环境，使模拟对齐评估更难被能力强模型识别为测试，从而提升安全评估的真实性与结论可靠性。 |
| [^44] | [SCX Router: Streaming Zero-Shot Model Selection with a Decoder-KV Classifier and a Real-World Task Ontology](https://arxiv.org/abs/2609.02292) | SCX Router是一个轻量级零样本模型路由器，通过解码器-KV缓存执行路径实现流式模型选择，无需自回归生成即可为各候选LLM分配适配度评分，从而在真实任务层面实现速度、成本与质量的优化权衡。 |
| [^45] | [Entangled Representations Amplify Collateral Damage in Unlearning](https://arxiv.org/abs/2609.02285) | 该研究首次通过受控实验验证了表示纠缠会加剧机器遗忘的附带损害——通过训练知识域解耦程度不同的语言模型套件，证明更解耦的模型在固定遗忘水平下保留成本可降低约4倍。 |
| [^46] | [Do Large Language Models Capture the Diversity in their Training Data?](https://arxiv.org/abs/2609.02275) | 该论文提出一种基于信息论的方法，通过比较模型生成输出与训练数据的条件熵，发现大语言模型（如OLMo、Pythia和GPT-Neo）生成内容的多样性系统性地低于其训练数据的多样性。 |
| [^47] | [CoMerge: Conflict-Driven Preference Optimization for Multi-Task Model Merging](https://arxiv.org/abs/2609.02273) | CoMerge 提出一种冲突驱动的偏好优化框架，将模型合并重新表述为偏好优化问题，利用朴素合并的缺陷作为困难负样本自监督构建偏好对，并通过优化轻量级的逐张量合并系数来缓解参数空间干扰，从而提升多任务大语言模型的合并效果。 |
| [^48] | [PaperCompiler: Faithful Paper-to-Code Generation via Repository-Level Specification Compilation](https://arxiv.org/abs/2609.02272) | 论文提出PaperCompiler框架，将基于论文的证据编译为显式的仓库级实现规格，避免了现有论文到代码智能体中间输出被下游编码智能体忽略或曲解的问题，从而实现更忠实的论文到代码生成。 |
| [^49] | [From Detection to Characterization: A Large-Scale Study of Ragebait on Japanese X](https://arxiv.org/abs/2609.02262) | 本研究利用LLM辅助标注构建数据集并训练出日语愤怒诱饵检测集成分类器，首次对X平台日语帖子进行大规模分析，发现愤怒诱饵在政治、歧视、公共卫生和人际冲突等争议性话题中更为普遍。 |
| [^50] | [APEx: Distillation of Agent Procedural Experience for Adaptive Deep Research Question Answering](https://arxiv.org/abs/2609.02253) | APEx提出分层经验利用框架，通过执行器-蒸馏器-规划器闭环架构和三阶段GRPO交替训练，将智能体交互历史蒸馏为程序性技能，并在测试时作为先验支持规划器在线自适应，从而提升深度研究问答性能。 |
| [^51] | [RideSkill: A Hierarchical Algorithm for Generalized Ride Sharing with LLM-Driven Automatic Evolution](https://arxiv.org/abs/2609.02250) | 该论文提出RideSkill，一种由大语言模型驱动自动进化的分层算法，用于解决泛化拼车问题，克服了传统多智能体强化学习方法在泛化性、可迁移性和大规模训练方面的局限。 |
| [^52] | [LeakageBench: Document-Level Leakage Risk for Redacting Personally Identifiable Information in Document Images](https://arxiv.org/abs/2609.02207) | 该论文提出了LeakageBench——一个用于评估文档图像中PII脱敏文档级泄露风险的挑战性基准数据集，实验表明即使借助Code Interpreter等工具将GPT-5.5的定位F1从0.090提升至0.249，页面级泄露率仍高达0.968，揭示了现有方法在文档级脱敏安全性上的严重不足。 |
| [^53] | [Breadth Beats Depth: Improving GCG-Based Jailbreak Optimization with Breadth-Oriented Suffix Search](https://arxiv.org/abs/2609.02172) | 本文提出即插即用框架BOSS，通过尾部聚焦对抗损失和面向广度的后缀搜索策略改进基于GCG的越狱攻击优化，在提升攻击成功率的同时降低了优化时间。 |
| [^54] | [Do Cantonese-Adapted Language Models Better Predict Cantonese Reading? A Cross-Model Eye-Tracking Evaluation](https://arxiv.org/abs/2609.02163) | 本研究基于自然粤语眼动追踪数据，通过词汇意外度、词性意外度和熵等信息论指标评估发现，经过大规模粤语继续预训练和指令微调的 CantoneseLLM-7B 比通用模型或轻度粤语适配模型更能预测人类粤语阅读行为。 |
| [^55] | [OBJECTION! Lawyer Agents Mitigate Guilty Bias in Legal Judgment Prediction](https://arxiv.org/abs/2609.02158) | 该论文提出OBJECTION推理时框架，将对抗性律师智能体嵌入罪责、违法性和可责性三步推理的每个阶段，通过注入法律辩护论点主动挑战模型的有罪预设，从而缓解法律判决预测中的“有罪偏见”。 |
| [^56] | [A Layered Taxonomy for Chinese Learner Grammatical Error Annotation](https://arxiv.org/abs/2609.02153) | 本文提出了一种连接中文语法纠错与教学错误分析的分层语法错误标注体系，采用三层核心标签加中文特有扩展范畴的设计，并通过MuCGEC覆盖率分析和多模型一致性研究验证了其有效性。 |
| [^57] | [EmoStance: Response-Side Affective-Orientation Control for Empathetic Response Generation via Emoji Weak Supervision](https://arxiv.org/abs/2609.02133) | 该论文提出 EmoStance 方法，将多标注者表情符号分布作为弱监督证据来诱导近似倾听者立场的潜在控制空间，并通过连续前缀嵌入引导冻结的指令微调大语言模型，实现共情回复生成中的响应侧情感取向控制。 |
| [^58] | [C$^{3}$T: Counterfactual Causal Reasoning for Sentiment Shifts in Social-Media Conversation Trees](https://arxiv.org/abs/2609.02131) | 该论文提出CaSiRe因果情感推理标注层与C³T反事实因果对话Transformer模型，通过将否认、证据、攻击等话语行为视为干预措施，联合预测并解释社交媒体谣言对话树中情感如何发生转变及其因果来源。 |
| [^59] | [AI agents reshape consensus formation in human groups](https://arxiv.org/abs/2609.02122) | 本研究通过协作描述游戏实验发现，LLM智能体在人机混合群体中的比例会以三种截然不同的方式重塑共识形成——低比例促进人类主导共识、中等比例阻碍收敛、高比例恢复强共识但使其转向更抽象的智能体主导约定。 |
| [^60] | [text2ql: Multi-Target Natural Language Querying via a Language-Agnostic Intermediate Representation](https://arxiv.org/abs/2609.02115) | text2ql框架通过语言无关的中间表示QueryIR和可插拔渲染器架构，实现了同时面向SQL和GraphQL的多目标自然语言查询，其零LLM确定性模式在3.2毫秒中位延迟下达到100%执行准确率，并为每个生成的查询提供运行时置信度分数。 |
| [^61] | [Predict, Don't Iterate: Efficient Adaptive-Length Infilling for Diffusion Language Models](https://arxiv.org/abs/2609.02108) | 提出一种“预测而非迭代”的高效自适应长度填充方法，让扩散语言模型直接一次性预测合适的填充长度，从而克服对初始长度的敏感性，并避免迭代搜索带来的大量额外计算开销。 |
| [^62] | [MASkills: Continual Skills Optimization for Multi-Agent LLM Systems](https://arxiv.org/abs/2609.02094) | MASkills是一个持续学习框架，通过技能条件化信用分配、分层信用聚合和动量平滑优化的新流水线，使多智能体LLM系统的技能库能够通过精炼、归纳、巩固和剪枝不断演进优化。 |
| [^63] | [Selective Knowledge Edit Reversal via Gated Singular Vector Shrinkage](https://arxiv.org/abs/2609.02091) | 本文提出基于门控奇异向量收缩的谱分析逆转框架，通过假设编辑信息稀疏编码于权重矩阵主奇异子空间，实现了对大语言模型中特定知识编辑的选择性精准逆转，同时保留其他有益编辑不受影响。 |
| [^64] | [IDEEA: training-free Input-Dependent stEEring via Activation cluster matching](https://arxiv.org/abs/2609.02089) | 提出IDEEA框架，通过对每个注意力头的正负激活进行聚类并求解最优匹配问题来构建簇条件化的引导方向，首次实现了无需训练、随输入自适应变化的大模型激活引导，克服了传统固定单一方向引导的根本局限。 |
| [^65] | [XMerge: Cross-Axis Selection and Reconstructive Layer Merging for LLM Depth Compression](https://arxiv.org/abs/2609.02083) | XMerge 是一种训练后的大语言模型深度压缩方法，通过跨轴选择识别隐藏状态变化最小的层块，并利用局部边界重构重新拟合相邻存留块，在不改变架构、不增加推理参数、无需任务标签的情况下实现高质量的层删除压缩。 |
| [^66] | [Transfer Safety Awareness for Cross-Modal Safety Drift in Multimodal Large Language Models](https://arxiv.org/abs/2609.02082) | 针对多模态大语言模型中“跨模态安全漂移”这一新安全问题（无害文本结合图像即可传达有害意图且模型难以拒绝），提出轻量级的安全意识表示迁移方法（SRT），将文本安全信号迁移至视觉场景以有效缓解该风险。 |
| [^67] | [HyGRAIL: Cost-Aware and Evidence-Grounded Scientific Hypothesis Discovery over Knowledge Graphs](https://arxiv.org/abs/2609.02056) | HyGRAIL 提出了一个结合异构图神经网络分诊与大语言模型审查的成本感知、证据支撑框架，通过仅将图上不确定的模糊候选假设路由给 LLM 审查，从而在知识图谱上实现高效且可靠的科学假设发现。 |
| [^68] | [Privacy Washing: Detecting Internal Contradictions in Privacy Policies](https://arxiv.org/abs/2609.02055) | 本文提出“隐私洗白”概念并构建四阶段检测流水线（语句提取、兼容性过滤与自然语言推理筛查、多模型评审验证、主题分析），在相隔11年的两个隐私政策语料库中发现高度一致的矛盾模式，其中第三方共享类矛盾最为普遍，约12.2%的公司存在至少一个经大语言模型评审团确认的内部矛盾。 |
| [^69] | [A Tri-Agent Framework for Evaluating and Aligning Question Clarification Capabilities of Large Language Models](https://arxiv.org/abs/2609.02054) | 本文提出一个由问题澄清智能体、应答智能体和评估智能体组成的三智能体框架，用于稳健地评估和对齐大语言模型在交互对话中识别歧义并进行问题澄清的能力。 |
| [^70] | [The Dynamics of Continuous Mixture Collapse in Language Models](https://arxiv.org/abs/2609.02049) | 该研究揭示了语言模型无法保持连续混合推理状态的深层原因，识别出三种相互独立的失败机制：transformer 架构对混合几何结构的固有扭曲、训练过程对这种扭曲的显著放大，以及 softmax 读出与自回归反馈构成的动力系统导致混合分量被单一主导或坍缩至不可区分。 |
| [^71] | [How Output Format Confounds Data Quality and Capability in Instruction Tuning](https://arxiv.org/abs/2609.02015) | 输出格式同时混淆了指令微调的数据质量评估与模型能力测量——质量信号存在于梯度更新方向而非谱统计量中，且模型能力是相对于训练时的输出格式存储的，更换格式可能让提升40多分的技能几乎消失。 |
| [^72] | [Train What You Deploy: Closing the MLP Reachability Gap in Low-Rank Clone Distillation](https://arxiv.org/abs/2609.02006) | 该论文提出“训练你所部署的”原则，让训练直接覆盖完整部署矩阵而非教师诱导的权重切片，在不增加任何推理成本的前提下释放低秩克隆蒸馏中62.5-81.4%被困住的容量，在三个教师模型上取得显著性能提升。 |
| [^73] | [NS-Copilot: An LLM-Driven Agent System for Autonomous Neuroscience Analysis](https://arxiv.org/abs/2609.01971) | NS-Copilot是一个由大语言模型驱动的多智能体系统，能够自主选择和协调神经科学领域的各类预训练模型，支持EEG和细胞外尖峰数据等关键模态，为专业神经科学分析任务提供端到端的自主工作流程。 |
| [^74] | [Sparse Readout Prism: Explaining Logit-Lens Scores in Features Instead of Tokens](https://arxiv.org/abs/2609.01936) | 该论文提出稀疏读出棱镜（SRP），仅利用读出矩阵自身的权重将其分解为稀疏“读出特征”，把logit-lens分数解释为特征贡献之和，从而消除了透镜读数对拟合语料库的依赖（语料库条件性），并支持跨词元、上下文、层与透镜的比较。 |
| [^75] | [CRISP: Cliff-awaRe Input-adaptive Sparse Prefilling with Structural-Mass-Motivated Routing](https://arxiv.org/abs/2609.01925) | 该论文提出CRISP方法，用直接从代理注意力图结构中读取路由决策的结构代理C_struct替代JSD路由，解决了动态稀疏注意力路由中的两个结构性挑战，实现了长上下文LLM推理的高效输入自适应稀疏预填充。 |
| [^76] | [Grounded, Compute-Efficient LLM Policy Agents for Energy-Poverty Equity in Physically-Constrained Peer-to-Peer Energy Markets](https://arxiv.org/abs/2609.01918) | 该论文提出EqGrid闭环仿真框架，以低频开源LLM政策智能体设定价格、碳限额与定向补贴，配合高频多智能体强化学习交易者在受物理电网约束的点对点能源市场中出清交易，并通过真实智能电表数据校验的家庭画像和形式化能源贫困公平指标（能源负担、基尼系数、LIHC）来衡量AI的社会影响，从而避免了对碳密集型云LLM的依赖。 |
| [^77] | [Accurate in space, unreliable in time: how LLMs represent national cultural change](https://arxiv.org/abs/2609.01902) | 该研究基于二十余年的世界价值观调查数据发现，大语言模型虽能在空间上将各国较准确地定位于文化地图上的当前位置，却无法可靠地表征各国文化随时间演变的变迁轨迹。 |
| [^78] | [GAPS: Dimension-Level Gates for Conditional Activation Steering](https://arxiv.org/abs/2609.01878) | GAPS提出维度级条件化的激活转向方法，通过两个无需训练的门控——静态可分离性门控（基于AUROC筛选携带概念信息的神经元）和动态后验门控（基于高斯模型判断激活状态），精确决定对哪些神经元进行干预，从而更细粒度地抑制语言模型的不良行为。 |
| [^79] | [Thinking effort aligns between humans and reasoning models in abductive reasoning](https://arxiv.org/abs/2609.01867) | 该论文通过溯因推理任务（其难度无法通过形式结构捷径伪装）发现，大型推理模型与人类在推理努力程度（思考成本）上表现出行为对齐。 |
| [^80] | [ExecRetrieval: Measuring the Functional-Correctness Gap in Code-Embedding Retrieval](https://arxiv.org/abs/2609.01865) | 提出 ExecRetrieval 基准（939 个 Python 任务），通过在搜索池中植入与规范实现几乎相同、但经执行验证的有缺陷变体，首次衡量了代码嵌入检索在区分功能正确代码与错误代码上的差距。 |
| [^81] | [The Memory Trust Gap: Capability-Dependent Failures in Persistent-Memory Agents](https://arxiv.org/abs/2609.01852) | 该论文提出并量化了“记忆信任差距”现象：持久记忆智能体会过度信任（而非混淆于）过期的存储事实并覆盖权威证据，且这种失效受模型能力门控——规模越大的模型在过期记忆被伪装成最新信息时崩溃反而越严重。 |
| [^82] | [Cite or Decline: A Strict Course-Grounded Chatbot for STEM Lecture Videos](https://arxiv.org/abs/2609.01846) | 本文提出了VideoPoints平台一学期的实际部署，其检索增强聊天机器人严格基于课程材料回答问题并提供带时间戳的引用，在无证据时选择拒答，833条学生消息中实现了零课程边界越界，证明了严格课程约束设计的可行性。 |
| [^83] | [Candidate Generation and Definition-Guided Verification for Sentence-Level Depression Symptom Recognition](https://arxiv.org/abs/2609.01833) | 提出了一种两阶段框架，先由对比学习微调的句子编码器生成抑郁症状候选，再由微调的语言模型依据诊断定义验证候选症状是否出现，在句子级抑郁症状识别任务上取得了所有方法中最佳的准确率和F1分数。 |
| [^84] | [Interpretable Symptom Vectors for Depression in a Large Language Model](https://arxiv.org/abs/2609.01832) | 该研究通过机制可解释性技术发现大语言模型内部在第21层对抑郁症状产生几何分离，并构建“症状向量”将文本投影后得到各症状系数，其能保留临床医生标注的严重程度排序，从而增强LLM在抑郁症评估中的临床可信度。 |
| [^85] | [AVERT: Audio-Verified Adjudication for Spoken Dialogue State Tracking](https://arxiv.org/abs/2609.01828) | AVERT通过结合跨轮一致性与音频条件验证器对候选值打分，并利用投票、添加、交换三种算子纠正口语对话状态跟踪中的三类可恢复错误，在SpokenWOZ上无需重训练即可将JGA提升至40.13。 |
| [^86] | [TalkFa: A Unified Benchmark for Farsi Dialogue Generation and Understanding](https://arxiv.org/abs/2609.01810) | 该论文提出了TalkFa——首个针对波斯语的统一对话生成与理解基准，由三个经母语者严格人工审核的数据集构成，并通过实验证明LoRA微调只需少量训练数据即可显著提升波斯语对话生成与理解性能。 |
| [^87] | [How Do Prompt Variations Affect Energy Consumption in On-Device LLMs?](https://arxiv.org/abs/2609.01798) | 本研究通过大规模实证分析首次揭示，提示词的认知负荷主要影响设备端LLM推理中每个token的能耗成本，而措辞模式主要通过token使用量影响总能耗，为节能导向的模型感知提示词设计提供了依据。 |
| [^88] | [Disentangling Statistical Preemption from Entrenchment in Language Models' Avoidance of Overgeneralization](https://arxiv.org/abs/2609.01794) | 本研究通过在语言模型上进行受控养育实验并系统移除先占性与非先占性证据，首次区分了统计先占与固着两种假说，发现语言模型避免过度泛化时并不依赖动词层面的先占效应，而是将竞争结构视为间接正面证据而非负面证据。 |
| [^89] | [VakyArth: Evaluating Pragmatic Competence in LLMs across Indic Languages](https://arxiv.org/abs/2609.01788) | 该论文提出了首个针对印度语系语言（印地语、旁遮普语、泰米尔语、马拉雅拉姆语）的语用能力诊断基准VakyArth，通过母语者编写的多任务评估揭示了多语言大模型在印度语言文化相关的语用推理上存在系统性失败。 |
| [^90] | [MemeCULT-1K: Benchmarking South Asian Cultural Context and Humor Understanding of Multimodal Models](https://arxiv.org/abs/2609.01772) | 提出了包含 1000 个南亚多语言模因的基准数据集 MemeCULT-1K，并证明为视觉语言模型提供少量文化背景信息即可显著且一致地提升其对模因的理解能力。 |
| [^91] | [When Can a Machine Trust a Statute? A Survival Certificate for Machine-Extracted Legal Logic](https://arxiv.org/abs/2609.01741) | 该论文提出一种“被动存续证书”方法，通过量化不同法条抽取器之间的分歧、在1,000次蒙特卡洛试验中重放噪声并以Wilson 95%置信下界作为门槛，来认证哪些机器提取的Duquenne-Guigues形式蕴含能够在解析噪声下可靠存续。 |
| [^92] | [SpeakPay: Domain-Adaptive LoRA Fine-Tuning of Whisper for Low-Resource Nepali Financial Speech Recognition](https://arxiv.org/abs/2609.01737) | 提出了SpeakPay语音优先数字钱包，通过构建403条尼泊尔语金融语音指令数据集并使用LoRA微调Whisper，将词错误率降低67.2%、天城文数字识别准确率从0%提升至73.9%、交易成功率提升约20倍，为视障用户提供了可用的语音支付方案。 |
| [^93] | [Harness Engineering in LLM Tool Use via Agent-Native Reusable Tool Primitives](https://arxiv.org/abs/2609.01736) | 提出以自然语言取代API模式作为工具调用接口的“工具原语”设计，并构建包含25,519个函数的集中式仓库ToolFace供LLM在推理时动态检索工具，从而解决多步多轮推理脆弱及大规模工具目录下性能退化的问题。 |
| [^94] | [Learning Evidence Sufficiency Boundaries for Selective Answering in Grounded Multi-Hop QA](https://arxiv.org/abs/2609.01687) | 提出了证据充分性边界训练框架，通过构建有序证据链并直接监督弃答到作答的转变，使多跳问答模型学会在证据不支持或部分支持时弃答、证据首次充分时作答、且在冗余证据下保持答案稳定。 |
| [^95] | [Ranked by the Matcher: A Reproducibility Audit of Knowledge Graph Extraction from Threat Reports](https://arxiv.org/abs/2609.01671) | 该论文对威胁报告知识图谱抽取评估进行了可复现性审计，发现三元组匹配规则的不明确与差异会显著逆转系统排名（同一预测集F1跨度达0.16–0.70），并提出与人工裁决一致性达86%的LLM评判器以及可独立变换验证层的CTIForge平台，以实现更可靠、可分离组件效应的评估。 |
| [^96] | [Beyond Textual Chain-of-Thought: A Survey on Action-Grounded Reasoning in Autonomous Driving](https://arxiv.org/abs/2609.01659) | 本综述调研171篇论文，提出以中间表示形式为组织轴心的分类体系，将自动驾驶中从文本思维链转向基于动作推理的方法系统化为四大类13个子类，并指出能够扎根真实世界且与实时性耦合的中间表示是驾驶智能体推理的未来前沿。 |
| [^97] | [PRO-Step: Step-level Process Reward Optimization for Retrieval-Augmented Generation](https://arxiv.org/abs/2609.01658) | PRO-Step训练了一个同时评估逻辑有效性与证据支撑的生成式过程奖励模型，通过PRM引导的价值树搜索构建偏好对并进行步骤级直接偏好优化，从而有效抑制RAG多跳推理中的错误传播问题。 |
| [^98] | [Whose Judgments Count? Representation Gaps in Crowdsourced Content Moderation Produce Unequal Protection from Perceived Toxicity](https://arxiv.org/abs/2609.01625) | 该研究通过结合大规模删除判断数据与反事实模拟，揭示了众包内容审核中的“内群体保护”效应——与审核员群体共享人口身份的用户获得了不成比例的更多免于感知毒性的保护，从而导致不同用户群体受到的保护不平等。 |
| [^99] | [MESSY STREETS: A Benchmark for Geocoding Real-World Addresses](https://arxiv.org/abs/2609.01612) | MESSY STREETS是一个评估地理编码器处理真实杂乱网页地址的新基准，揭示了商业地理编码器的召回率比开源系统高出多达49个百分点，差距主要源于非规范地址的候选返回率差异。 |
| [^100] | [EvalDetectBench: A Benchmark for Measuring Evaluation Awareness in Frontier Language Models](https://arxiv.org/abs/2609.01611) | 该论文提出了EvalDetectBench，一个开放式的基准和流水线，用于衡量前沿大语言模型的评估意识（即识别自己正在被评估的能力）以及各个基准的可检测程度，从而保障AI安全评估结果的有效性。 |
| [^101] | [Selective Agent Guidance via Entropy: Learning Autonomous Policies from Imperfect VLM Teachers](https://arxiv.org/abs/2609.01567) | 该论文提出SAGE框架，仅在智能体不确定时才查询昂贵的视觉语言模型教师，并利用环境优势对教师建议进行加权蒸馏，从而训练出无需教师引导即可自主行动的轻量级强化学习策略。 |
| [^102] | [FinLifeBench: Exhaustive Life-Event History and Financial-State Reconstruction from Longitudinal Banking Dialogue](https://arxiv.org/abs/2609.01198) | 提出FinLifeBench基准，基于6,000个韩语银行对话会话，评估大语言模型在穷尽式重建客户人生事件历史与34维财务状态方面的长程记忆能力，发现随会话累积事件召回率显著下降（0.591降至0.445），且错误主要源于事件遗漏。 |
| [^103] | [Quit While You're Ahead: Quit for Efficient Candidate Generation in Machine Translation Reranking](https://arxiv.org/abs/2609.00588) | 提出Quit方法，通过不确定性量化的早停策略对机器翻译的整个候选生成—重排序流程进行增量式生成与重排序，在最高候选质量稳定时提前终止，从而在保持翻译质量的同时显著降低推理延迟。 |
| [^104] | [Exploring Collaboration between a language and a non-language agent](https://arxiv.org/abs/2609.00474) | 该论文提出LLAMIA-Bench基准，用于研究将非语言智能体的连续表示“言语化”为文本是否成为LLM协作的瓶颈，并提出潜在状态内化方法来改善LLM与国际象棋引擎等非语言智能体的协作。 |
| [^105] | [CogEvol: Towards Efficient and Reliable Learning Environment Generation](https://arxiv.org/abs/2608.30968) | CogEvol是专为学习环境生成训练的模型系列，能将课程简报一次性转化为幻灯片或交互式HTML页面，通过真实生产失败数据驱动的SFT和修复奖励作弊后加固的GRPO强化学习保障可靠性，其27B模型以少26.9倍的参数量媲美旗舰编程模型，并已投入真实生产环境服务。 |
| [^106] | [Lot Machine: Multimodal Lot Extraction from Auction Catalogs](https://arxiv.org/abs/2608.30510) | 本文提出了一个利用视觉-语言模型从历史拍卖目录中自动提取结构化拍品元数据的流水线，并在不同提示策略、受限解码框架和部署条件下进行了系统评估，以满足文化遗产机构在预算、算力和数据隐私方面的实际需求。 |
| [^107] | [GPAgentBench-2K: Benchmarking Large Language Model Agents in Complex Clinical Action Space](https://arxiv.org/abs/2608.30188) | 该论文提出了首个基于受约束马尔可夫决策过程的基层医疗临床决策LLM智能体基准GPAgentBench-2K，评估发现即使是诊断准确率最高的前沿模型，在超过一半的高风险病例中也会违反安全约束，揭示了临床质量与安全之间的鸿沟。 |
| [^108] | [SHADOWBENCH: Toward Reliable Automatic Evaluation of Semantic Alignment in Autoformalization](https://arxiv.org/abs/2608.29270) | 提出 SA-Pass 评估方法，通过“影子”辅助陈述的双向逻辑检查来可靠评估自动形式化中的语义对齐，并构建了包含 178 个研究生至研究级问题的 Lean 4 基准 ShadowBench。 |
| [^109] | [The Illusion of Replacement: Rethinking Specialized Machine Learning Models in the Foundation Model Era](https://arxiv.org/abs/2608.28980) | 本文综述159篇论文后发现，语言模型虽在极端少样本预测等特定场景中可与专用模型竞争，但一旦直接评估结构表示与计算能力，并无证据表明其能全面取代机器学习中的专用架构。 |
| [^110] | [Automated Researchers Can Reliably Mitigate Alignment Failures](https://arxiv.org/abs/2608.28945) | 自动化对齐研究员（AAR）通过后训练方法能够可靠地缓解10种对齐失败并泛化到更大的模型，其效果甚至优于28名经验丰富的人类研究员在八小时内开发的方法。 |
| [^111] | [Difference-in-Differences on a Censored Rating Scale Can Manufacture an Effect: Evidence from a Pre-Registered LLM-Judge Audit](https://arxiv.org/abs/2608.27309) | 本文揭示双重差分法在截断评分量表上因截断不均会制造虚假交互效应，并通过预注册审计实证证明该偏差可导致无效结果。 |
| [^112] | [SPEAR: Distilling Domain-Adaptive Reasoning Skeletons via Sequential Symbolic Alignment in Reinforcement Learning](https://arxiv.org/abs/2608.26550) | SPEAR提出了一种无需训练、即插即用的过程奖励方法，通过符号里程碑和最长公共子序列对齐，在强化学习蒸馏中提供密集且逻辑一致的奖励，避免了昂贵的外部神经过程奖励模型。 |
| [^113] | [A Storage-Retrieval Gap in Parametric Knowledge Graph Memory](https://arxiv.org/abs/2608.25489) | 该论文提出将知识图谱离线编译为LoRA适配器作为参数化知识层，在零查询上下文成本下实现事实知识泛化，但发现存储知识无法通过相似性检索恢复，揭示了参数化记忆中的存储-检索差距。 |
| [^114] | [Whitewashing Hate, Smearing Harmless Content: Annotator-Style Rebuttal Attacks on LLM-Based Moderation](https://arxiv.org/abs/2608.22230) | 本研究揭示了标注者风格的反驳攻击能显著破坏LLM仇恨言论审核的准确性，且洗白与污蔑两种操纵方向存在模型特定的不对称效应。 |
| [^115] | [ToSCA: Leveraging Hierarchical Reinforcement Learning on Temporal and Strategic Abstractions of Conversational Agents](https://arxiv.org/abs/2608.21969) | 本文提出一种两级层次强化学习框架，结合话语级策略抽象与词元级解码，并引入双粒度奖励机制，以提升对话代理在复杂交互中的性能。 |
| [^116] | [Agentic Scaffolding Amplifies Sycophantic Behavior in Large Language Models](https://arxiv.org/abs/2608.21377) | 本文发现代理式交互脚手架（如多轮反馈和迭代细化）会系统性放大LLM的谄媚行为，导致平均准确率下降6.3%，且更强模型放大效应更显著。 |
| [^117] | [LoRA-GA$^2$: Low Rank Adaptation with Multi-step Gradient Adaptive Alignment](https://arxiv.org/abs/2608.19800) | 本文提出LoRA-GA²算法，通过轻量级探针利用多步梯度信息，结合谱感知的秩分配和最优初始化，在不增加GPU内存的前提下缩小LoRA与全参数微调的性能差距。 |
| [^118] | [Preference Tree Optimization: Enhancing Goal-Oriented Dialogue with Look-Ahead Simulations](https://arxiv.org/abs/2608.12062) | 本文提出偏好树优化框架，结合前瞻模拟和直接偏好优化，有效应对数据稀缺，提升目标导向对话系统的决策能力。 |
| [^119] | [REAP: Relation-Aware Elicitation and Parsing for Closed-Book Knowledge Base Construction from LLMs](https://arxiv.org/abs/2608.10963) | REAP系统通过结构化思维链推理、关系特定查询策略与空集门控机制的组合，在闭卷、无微调且参数量不超过32B的约束下，从大语言模型中提取参数化知识构建知识库，宏平均F1达到0.62。 |
| [^120] | [GPTKB 2.0: Browsing, Querying, and Auditing a Disambiguated LLM-Derived Knowledge Base](https://arxiv.org/abs/2608.06992) | GPTKB 2.0 是一个交互式网络演示系统，展示了从大语言模型构建的大规模消歧知识库（含3840万三元组和160万实体），在构建过程中通过上下文引导消歧区分同名异义、合并同义提及，并支持实体浏览、事实溯源审计、SPARQL与自然语言查询及实体链接。 |
| [^121] | [Direct Construction of Disambiguated Knowledge Bases from Large Language Models](https://arxiv.org/abs/2608.03729) | 提出GPTKB 2.0方法，通过对实体、关系和类别的即时消歧机制，直接从大语言模型构建了首个百万级规模的消歧知识库，包含超过100万个实体和3840万条三元组。 |
| [^122] | [PGMem: Tightly Coupled Persona-Memory Graph for Lifelong Personalized Agents](https://arxiv.org/abs/2608.01708) | PGMem通过类型化溯源边和证据边将事件与人格节点紧耦合为异构图，使每个人格信号都可追溯到支持或修正它的事件，解决了记忆与人格脱节的问题，并在三个基准上持续超越现有记忆基线。 |
| [^123] | [Do VLMs Read or Rewrite? On Transcription Faithfulness in Vision-Language Models](https://arxiv.org/abs/2607.21617) | 本文提出 FaithC4 多语言扰动基准，揭示视觉语言模型在面对不完美文本时常将其“改写”为更合理形式而非忠实转录，其中通用 VLM 在扰动下词错率退化最严重，而传统 OCR 最为稳健。 |
| [^124] | [Multi-Mask Diffusion Language Models for Few-Step Generation](https://arxiv.org/abs/2607.19686) | 提出多掩码扩散模型MultiMDM，通过在前向过程中保留掩码结构、在反向过程中先预测指定掩码再精炼为干净词元的起草能力，实现高质量的少步文本生成。 |
| [^125] | [What Transfers Under Source Shift? Definitions, Examples, and Fine-Tuning for Climate Disclosure Classification](https://arxiv.org/abs/2607.17952) | 该论文将气候披露分类重构为跨源适应问题，通过在十一个开源与闭源LLM上评估定义、示例和微调三种策略，发现尽管所有策略平均均能带来跨源收益，但源内表现最强的策略（如相似度检索和LoRA微调）并不一定是源偏移场景下最有效的策略。 |
| [^126] | [Persistent Sparse Autoencoders: Learning Feature-Specific Timescales in Language Model Representations](https://arxiv.org/abs/2607.17117) | 该论文提出持久稀疏自编码器，通过为每个特征学习一个持久性系数，使稀疏自编码器能够仅凭重构目标从语言模型激活中自动学习特征特定的时间尺度，同时保持高质量的重构效果。 |
| [^127] | [LLM Watermarking as Big Data Provenance: A Deployment-Oriented Systematization](https://arxiv.org/abs/2607.10103) | 本文将LLM水印系统化为大数据生态系统的溯源基础设施，沿插入点、验证权限、运行状态和转换威胁模型四个部署维度对现有方法进行分类，并分析部署选择对可靠性、安全性和可扩展性的影响。 |
| [^128] | [Gauge dependence and structured-output corruption in sign-branched repetition penalties: measurements across models, inference stacks, and alternative repetition controls](https://arxiv.org/abs/2607.09791) | 该论文揭示了主流推理引擎中的符号分支乘法重复惩罚依赖于 logit 任意零点（规范选择），导致惩罚操作缺乏良好定义且在不同模型上效果各异，并会使 JSON 结构化输出的有效率从 97% 骤降至 23%，同时提出了减法式与归一化等不受规范影响的替代方案。 |
| [^129] | [What You See Is What You Get: Observation-Aligned Supervision for Chart-to-Code Generation](https://arxiv.org/abs/2607.04726) | 论文揭示了图表到代码生成训练中存在的四类潜在变量与观察图像不匹配问题，并提出观察对齐监督方法，用视觉上可约束的量替换潜在变量作为监督目标。 |
| [^130] | [Can LLM-as-a-Judge Reliably Verify Rubrics in Agentic Scenarios?](https://arxiv.org/abs/2606.29920) | 该论文提出了RuVerBench——首个用于评估LLM作为评判者在智能体场景（深度研究和智能体编程）中验证评分标准可靠性的基准，包含2,458个人工标注实例，并发现即使最先进的模型仍存在显著的可靠性缺陷。 |
| [^131] | [SABER-Math: Automated Benchmark for Information Retrieval Evaluation in Mathematics](https://arxiv.org/abs/2606.29894) | 该论文提出了首个无需专家标注、完全自动化的数学信息检索评估基准SABER-Math，它从28.3万道高中数学题出发自动构建具有挑战性的重排序任务，以克服现有基准无法捕捉细粒度数学相关性的问题。 |
| [^132] | [AdaMem: Learning What to Remember with Adaptive Memory Policies for Personalized Agents](https://arxiv.org/abs/2606.21144) | 提出AdaMem框架，利用基于用户反馈持续更新的自适应自然语言记忆策略，根据不同交互情境个性化控制LLM智能体的记忆写入内容，并构建AdaMem-Bench基准进行验证。 |
| [^133] | [VTOS: Learning to Orchestrate Vision Tools by Co-Searching Solutions and Observers](https://arxiv.org/abs/2606.20728) | 该论文提出VTOS框架，通过联合搜索组合视觉工具的可执行解决方案程序与能诊断失败模式并生成可操作反馈的观察者程序，实现自适应的视觉工具编排，克服了现有视觉编程智能体固定流水线在密集物体、遮挡、小目标和领域偏移下的脆弱性。 |
| [^134] | [Zone of Proximal Policy Optimization: Teacher in Prompts, Not Gradients](https://arxiv.org/abs/2606.18216) | 该论文提出ZPPO，受维果茨基最近发展区理论启发，将教师模型的帮助置于提示词中而非策略梯度中，通过为难题重新构造提示（如将正确教师回答纳入二选一问题），使小型学生模型能够基于自身rollout进行强化学习，从而规避知识蒸馏在小模型上的模仿脆弱性以及向梯度注入教师回答所导致的漂移问题。 |
| [^135] | [EComAgentBench: Benchmarking Shopping Agents on Long-Horizon Tasks with Distributed Hidden Intent](https://arxiv.org/abs/2606.17698) | 该论文提出了EComAgentBench基准，通过662个基于真实亚马逊商品的任务，将购物者隐藏意图分散于可见查询、工具访问的用户档案和澄清对话中，用以评估LLM购物智能体在长程任务中挖掘隐含需求、验证商品并归因失败的能力。 |
| [^136] | [Implicit vs. Explicit Prompting Strategies for LVLMs in Referential Communication](https://arxiv.org/abs/2606.17372) | 该研究通过控制任务差异并对比显式与隐式两种提示方式，发现大型视觉语言模型仅在显式提示下才能像人类一样协调生成高效指称表达，而无法从隐式提示中自主推断出交流效率的需求，揭示了人类与AI交流能力的关键差异。 |
| [^137] | [Do Large Language Models Always Tell The Same Stories?](https://arxiv.org/abs/2606.17350) | 研究发现大型语言模型生成的故事彼此之间比人类撰写的故事更加相似，前沿模型尤其倾向于收敛到一种“平均化”的通用叙事，缺乏人类作者群体的集体多样性。 |
| [^138] | [Follow the Latent Roadmap: Navigating Revocable Decoding for Diffusion LLMs with Anchor Tokens](https://arxiv.org/abs/2606.16847) | 提出了一种免训练框架ASRD，通过在嵌入空间中将解码上下文解耦为基于时间一致性识别的受信任锚定令牌和不确定候选令牌，解决了扩散大语言模型可撤销解码中的错误传播与局部错误强化问题。 |
| [^139] | [SHARD: Safe and Helpful Alignment via Self-Reframing Distillation](https://arxiv.org/abs/2606.15517) | SHARD提出一种自我重构蒸馏方法，通过重写敏感提示凸显良性意图、将模型自身回答重构为更安全更有益的版本并据此微调，从而在保持安全性的同时提升有益性，效果可与更大教师模型的蒸馏相媲美。 |
| [^140] | [MUDIDI: A Two-Stage Framework for Multilingual Dictionary Digitization with Language Models](https://arxiv.org/abs/2606.09435) | 该论文提出MUDIDI两阶段框架，利用视觉-语言模型将多语言词典扫描件数字化并转换为机器可读的词典学格式，同时发布了人工标注的词典条目数据集。 |
| [^141] | [What's in a Name? Morphological Shortcuts by LLMs in Pharmacology](https://arxiv.org/abs/2606.05616) | 大语言模型在药理学中过度依赖药物名称的词缀线索来推断药物含义，即使面对虚构药物也会产生类别级别的药理学响应，很少明确承认这种依赖，且有时会错误混淆共享相同词缀的药物属性。 |
| [^142] | [Beyond Retrieval: Learning Compact User Representations for Scalable LLM Personalization](https://arxiv.org/abs/2606.04547) | 提出TAP-PER框架，通过时序注意力前缀嵌入将用户偏好编码为紧凑的可学习表示，摆脱了检索式个性化对检索质量的依赖以及参数式个性化随用户规模增长的高昂存储成本，实现可扩展的大语言模型个性化。 |
| [^143] | [Knowledge Editing for Masked Diffusion Language Models](https://arxiv.org/abs/2606.03924) | 首次将“定位后编辑”知识编辑方法迁移至掩码扩散语言模型，发现最优编辑位置（最后一个主体词元处的早中期层MLP）在自回归模型与掩码扩散模型间可迁移，但多词元编辑在掩码扩散模型中退化显著更严重。 |
| [^144] | [The Geometry of LLM-as-Judge: Why Inter-LLM Consensus Is Not Human Alignment](https://arxiv.org/abs/2606.03043) | 本文提出一种将LLM评判者分数视为向量并通过测量离散度、有效秩、与人类分数夹角及一致性三元组的几何检验方法，揭示了LLM之间的相互共识不能等同于与人类判断的对齐——评判者在主观标准上彼此一致程度接近人类，但与人类评分的一致性仅达58-66%，原因在于它们可能共享同样的盲点。 |
| [^145] | [Translating Classical Poetry into Modern Prose](https://arxiv.org/abs/2606.02806) | 该论文构建了Padyam2Gadyam数据集（包含600首13-17世纪泰卢固语古典诗歌及其人工校验的泰卢固语和英语散文翻译），并据此评估了机器翻译系统与大语言模型在零样本诗歌到散文翻译任务上的表现，发现尽管大语言模型优于机器翻译系统，但各系统在散文翻译的生成与评估上仍存在系统性问题。 |
| [^146] | [Who Annotates in NLP? A Large-scale Assessment of Human Annotation Reporting between 2018 and 2025](https://arxiv.org/abs/2606.02255) | 首次对2018至2025年间主要NLP会议中的人类标注报告进行大规模任务级审计，提出统一的标注报告分类体系并借助经验证的LLM抽取流程构建了大规模标注报告数据集，揭示了标注者身份与过程控制等信息在论文中的普遍缺失。 |
| [^147] | [TUX: Measuring Human--AI Tacit Understanding](https://arxiv.org/abs/2605.30930) | 该论文提出了一个受派对游戏 Wavelength 启发的谱系放置任务，并定义了默契理解指数（TUX），用于量化人类与 LLM 智能体在缺乏明确目标、沟通或反馈情况下达成默契对齐的程度，发现特质空间中更接近的人类—智能体配对具有更高的默契度。 |
| [^148] | [Give it Space! Explicit Disentangling of Positional and Semantic Representations in Encoders](https://arxiv.org/abs/2605.30022) | 该论文通过将编码器Transformer中的语义、绝对位置和相对位置表示显式解耦为三条独立的信息流，实现了对位置信息内部处理机制的清晰研究，发现隔离的绝对位置子空间会自发塌缩为低频二维流形，为设计更好的位置编码提供了启示。 |
| [^149] | [On Asymmetric Optimization of Reasoning and Perception in Vision-Language Model Post-Training](https://arxiv.org/abs/2605.29496) | 该研究揭示视觉语言模型后训练中存在感知与推理的非对称提升现象——SFT中源于感知token占比失衡、RL中源于结果奖励与推理的耦合，并通过损失重加权等方法将端到端性能提升高达18.2分。 |
| [^150] | [When Discourse Pressures Conflict: Information Structure in Vision-Language Model Outputs](https://arxiv.org/abs/2605.28346) | 该研究借助匈牙利语中话题与焦点对应专属句法位置的特性，首次系统评估了视觉-语言模型在视觉问答中区分话语旧信息（话题）与新信息（焦点）的能力，发现模型虽能产出信息结构相关的句式，但与人类多变的语用策略不同，它们会坍缩为狭窄固定的响应模板，表现出模式坍缩式的过度规则化。 |
| [^151] | [BioELX: Context-Aware Cross-lingual Biomedical Entity Linking without Task-Specific Supervision](https://arxiv.org/abs/2605.27380) | BioELX提出了一种检索-重排序框架，利用Wikidata衍生的跨语言别名监督进行检索，并通过提及锚定提示将LLM重排序器适配于实体链接，实现了无需任务特定监督的上下文感知跨语言生物医学实体链接。 |
| [^152] | [CroCo: Cross-Lingual Contrastive Preference Tuning on Self-Generations](https://arxiv.org/abs/2605.26293) | 基于自生成响应的跨语言对比偏好调优无需语言特定的偏好标注，仅凭英语偏好训练的奖励模型即可在14种高低资源语言上实现有效迁移，并避免监督微调的灾难性遗忘。 |
| [^153] | [Measuring Reasoning Quality in LLMs: A Multi-Dimensional Behavioral Framework](https://arxiv.org/abs/2605.24661) | 本文提出了一个植根于认知科学的多维度行为评估框架，从正确性、一致性、鲁棒性、局部逻辑连贯性、效率和稳定性六个维度衡量大语言模型的推理质量，突破了仅依赖最终答案正确性的传统评估局限，并支持面向具体部署场景的模型选择。 |
| [^154] | [Response-free item difficulty modelling for multiple-choice items with fine-tuned transformers: Component-wise representation and multi-task learning](https://arxiv.org/abs/2605.16991) | 该论文提出对Transformer进行端到端微调以在无作答数据的情况下预测多选题难度，并通过组件化表示和多任务问答学习两种扩展提升了难度估计效果。 |
| [^155] | [When Prompts Interact: Assessing Prompt Arithmetic for Deconfounding under Distribution Shift](https://arxiv.org/abs/2605.03096) | 本文研究了通过任务算术组合软提示能否提升模型对混杂变量引起分布偏移的鲁棒性，并提出了一种混合提示算术方法来去除模型对虚假特征的依赖，相比完全微调更具计算效率。 |
| [^156] | [Can Coding Agents Reproduce Findings in Computational Materials Science?](https://arxiv.org/abs/2605.00803) | 本文提出 AutoMat 基准，用于评估大语言模型编码智能体复现计算材料科学论文中科学论断的能力，涵盖恢复欠规范计算流程、驾驭专用工具链和验证证据是否支持论断三大挑战。 |
| [^157] | [Language Diffusion Models are Associative Memories Capable of Retrieving Unseen Data](https://arxiv.org/abs/2604.26841) | 该论文证明均匀离散扩散语言模型本质上是联想记忆，其吸引盆可通过条件似然最大化而非显式能量函数形成，并揭示了由数据规模支配的从记忆到泛化的急剧转变，使其能够检索未见过的数据。 |
| [^158] | [GroupDPO: Memory-Efficient Group-Wise Direct Preference Optimization](https://arxiv.org/abs/2604.15602) | GroupDPO 提出一种内存高效的分组式直接偏好优化方法，通过基于目标特定逐响应系数的一阶线性化，在保持一阶梯度不变的同时于反向传播中解耦样本，从而充分利用偏好数据中多候选响应的监督信息。 |
| [^159] | [The Enforcement and Feasibility of Hate Speech Moderation](https://arxiv.org/abs/2604.12289) | 该研究通过54万条推文的大规模审计发现Twitter/X上80%的仇恨言论五个月后仍在线，但模拟“自动排序+人工分流”工作流证明大幅清除仇恨言论在财务上完全可行且成本远低于监管罚款，表明仇恨言论泛滥源于平台资源配置不足而非技术限制。 |
| [^160] | [Are Non-English Papers Reviewed Fairly? Language-of-Study Bias in NLP Peer Reviews](https://arxiv.org/abs/2604.07119) | 该研究首次系统刻画了NLP同行评审中的“研究语言偏见”，区分其负面与正面形式，构建了人工标注数据集LOBSTER及基于大语言模型的检测方法（宏F1达87.37），并通过分析15,645条评审发现非英语论文遭受的偏见显著更高。 |
| [^161] | [A Universal Vibe? Finding and Controlling Language-Agnostic Informal Register with SAEs](https://arxiv.org/abs/2603.26236) | 研究发现多语言模型中存在一个跨语言的“非正式语域”共享核心子空间，可通过稀疏自编码器进行定位和控制，表明俚语等语用语域是以统一抽象概念而非孤立的特定语言记忆被处理的。 |
| [^162] | [FDARxBench: Benchmarking Regulatory and Clinical Reasoning on FDA Generic Drug Assessment](https://arxiv.org/abs/2603.19539) | 该论文与FDA监管审查员合作，提出了首个基于FDA药品标签文档、由专家精心策划的仿制药评估问答基准FDARxBench，涵盖事实性、多跳推理和拒答任务，实验揭示了当前语言模型在事实依据、长上下文检索和安全拒答方面存在重大不足。 |
| [^163] | [Language Model Maps for Prompt-Response Distributions via Log-Likelihood Vectors](https://arxiv.org/abs/2603.18593) | 该论文提出用提示-回复对上的对数似然向量表示语言模型并构建模型地图，使模型间的欧氏距离近似对应条件分布的KL散度，从而捕捉模型属性与任务性能的全局结构，预测下游任务得分，并在无需直接观察的情况下近似复合提示操作的效果。 |
| [^164] | [ICE: Intervention-Consistent Explanation Evaluation with Statistical Grounding for LLMs](https://arxiv.org/abs/2603.18579) | 提出ICE框架，通过在多种干预算子下将模型解释与同等规模的随机基线进行统计对比，首次揭示了大语言模型的解释忠实性是依赖干预方法的量而非固定属性（切换算子导致差距高达44个百分点），并能检测出比随机表现更差的反忠实性现象。 |
| [^165] | [Mediocrity is the key for LLM as a Judge Anchor Selection](https://arxiv.org/abs/2603.16848) | 研究发现，在LLM作为评判者的基准测试中，选择表现“平庸”（中等水平）的模型作为锚点最为可靠，而常见的极端锚点（最强或最弱模型）会显著降低模型排名的可靠性。 |
| [^166] | [Probing Cultural Signals in Large Language Models through Author Profiling](https://arxiv.org/abs/2603.16749) | 本研究通过零样本歌词作者画像任务揭示了大语言模型中的系统性文化偏见——多数模型默认偏向北美族裔而DeepSeek-1.5B更对齐亚洲族裔，并创新性地提出MAD和RD两个公平性指标来量化这些差异。 |
| [^167] | [GONE: Structural Knowledge Unlearning via Neighborhood-Expanded Distribution Shaping](https://arxiv.org/abs/2603.12275) | 本文提出了GONE基准用于评估大型语言模型对结构化知识图谱事实的遗忘效果，能够解耦直接事实移除、推理泄漏和灾难性遗忘三种效应，并设计了邻域扩展分布塑造（NEDS）这一新型遗忘框架。 |
| [^168] | [TikZilla: Scaling Text-to-TikZ with High-Quality Data and Reinforcement Learning](https://arxiv.org/abs/2603.03072) | 该论文通过构建规模扩大四倍以上且质量更高的DaTikZ-V4数据集，并结合强化学习（而非仅用监督微调）来扩展Text-to-TikZ生成，以解决文本与图形不匹配及循环、无关内容等渲染错误问题。 |
| [^169] | [CLASE: A Hybrid Method for Chinese Legalese Stylistic Evaluation](https://arxiv.org/abs/2602.12639) | 该论文提出了CLASE，一种针对中文法律文本的混合式文体评估方法，解决了法律专家难以人工制定评分标准、基于参考的指标混淆语义与文体、以及LLM作为裁判不透明且不一致等评估难题。 |
| [^170] | [Reviewing the Reviewer: LLM-Assisted Reviewer Feedback Generation for Guideline Compliance](https://arxiv.org/abs/2602.10118) | 该论文提出一个基于大语言模型的推理时框架，通过将评审分解为论证片段、识别违反ACL滚动评审指南的问题并利用迭代重排序算法生成针对性反馈，帮助审稿人改进评审质量并提升指南合规性。 |
| [^171] | [Constrained Group Relative Policy Optimization](https://arxiv.org/abs/2602.05863) | 本文提出约束GRPO（Constrained GRPO），一种基于拉格朗日方法的GRPO扩展用于约束策略优化，并揭示了在归一化前对标量化奖励会导致共享分母耦合，使改变一个约束乘子会同时影响奖励与其他约束的相对权重这一关键失败模式。 |
| [^172] | [Modular Expert Merging for Biomedical Retrieval](https://arxiv.org/abs/2602.04731) | 本文提出模块化专家合并方法，通过合成难负样本和LoRA微调领域专家并合并，在生物医学检索上优于大规模混合训练，兼顾通用性能。 |
| [^173] | [Learning Query-Specific Rubrics from Human Preferences for DeepResearch Report Generation](https://arxiv.org/abs/2602.03619) | 本文提出通过强化学习从人类偏好标注数据中训练查询特定的评分准则生成器，采用混合奖励（偏好一致性、格式有效性、LLM评估）来解决深度研究报告生成中训练与评估缺乏可验证奖励信号的难题。 |
| [^174] | [CALIBURN: Self-Calibrated LLM Unlearning Alignment](https://arxiv.org/abs/2602.02824) | 我们提出了一种自校准遗忘方法，通过量化模型对不良知识的置信度来精确调整梯度更新，在实现细粒度遗忘的同时减少对保留数据的依赖，从而提升模型效用。 |
| [^175] | [Culturally Grounded Personas in Large Language Models: Characterization and Alignment with Socio-Psychological Value Frameworks](https://arxiv.org/abs/2601.22396) | 该论文提出基于世界价值观调查（WVS）的可解释变量生成具有文化根基的LLM合成人格，并从英格尔哈特-韦尔泽尔文化地图定位、人口层面一致性和道德基础理论三个互补视角，验证这些人格与人类社会心理价值框架的对齐程度。 |
| [^176] | [ChartAttack: Testing the Vulnerability of LLMs to Malicious Prompting in Chart Generation](https://arxiv.org/abs/2601.12983) | 本文提出了ChartAttack框架和AttackViz数据集，首次系统评估了多模态大语言模型在图表生成中利用设计误导元素诱导错误解读的能力，攻击可显著降低模型和人类的问答准确率，且在AttackViz上微调可提升模型的鲁棒性。 |
| [^177] | [Beyond Transfer Accuracy: Mechanism-Guided Controlled Adaptation for Low-Resource Languages](https://arxiv.org/abs/2601.08146) | 该论文提出了一种无需反事实的回路发现方法，并据此提出回路定向监督微调（CT-SFT），仅更新任务相关的注意力头和LayerNorm，从而在低资源语言适配中既保持竞争力又最有效地避免灾难性遗忘。 |
| [^178] | [LLMs Can't Play Hangman: On the Necessity of a Private Working Memory for Language Agents](https://arxiv.org/abs/2601.06973) | 本文提出了私有状态交互任务（PSIT）并从理论上证明：仅依赖公开对话历史的语言智能体在架构上无法完成需要私有隐藏状态的任务（如猜单词游戏），从而论证了语言智能体必须配备私有工作记忆。 |
| [^179] | [Expos\'ia: Teaching and Assessment of Academic Writing Skills for Research Project Proposals and Peer Feedback](https://arxiv.org/abs/2601.06536) | 提出了首个连接高等教育中写作与反馈的公开数据集Expos'ia，包含学生研究项目提案、同伴与导师反馈以及基于教学理论的细粒度人工评分，并用于基准测试大语言模型在写作与反馈自动评分任务上的表现。 |
| [^180] | [CHisAgent: A Multi-Agent Framework for Event Taxonomy Construction in Ancient Chinese Cultural Systems](https://arxiv.org/abs/2601.05520) | 该论文提出CHisAgent多智能体框架，通过归纳、扩展、充实三个角色专业化阶段，从《二十四史》等中国古代文献中自动构建历史事件分类体系，克服了LLM在中国历史语境下推理能力不足和人工分类构建成本高的问题。 |
| [^181] | [Agent Tools Orchestration Leaks More: Dataset, Benchmark, and Mitigation](https://arxiv.org/abs/2512.16310) | 该研究首次形式化了LLM智能体通过组合多个无害工具调用结果而泄露敏感信息的“工具编排隐私风险”（TOP-R），构建了包含1000个实例的TOP-Bench评测基准，并提出TOP-Align（SFT+DPO）训练方法来有效缓解该隐私泄露风险。 |
| [^182] | [GMTRouter: Personalized LLM Router over Multi-turn User Interactions](https://arxiv.org/abs/2511.08590) | 提出GMTRouter，将多轮用户-LLM交互建模为包含用户、LLM、查询、响应和轮次五种节点类型的异构图，以最大程度保留交互的关系结构，从而在用户偏好数据稀缺且格式不一致的情况下实现个性化的LLM路由。 |
| [^183] | [SignBind-LLM: Multi-Stage Modality Fusion for Sign Language Translation](https://arxiv.org/abs/2509.00030) | SignBind-LLM 通过三个分别处理连续手语、手指拼写和唇读的独立预训练专家流，利用轻量级 transformer 进行时间对齐融合，并结合预训练语言模型完成翻译，在无需人工词汇标注的情况下显著提升了手语翻译效果。 |
| [^184] | [HarmReduction: Benchmarking LLMs in Harm Reduction Information Provision to Support People Who Use Drugs](https://arxiv.org/abs/2507.21815) | 本文提出了HarmReduction基准，通过包含2,160个问答-证据对的数据集，从安全边界检查、定量数值提供和多药物使用风险推断三项任务评估大语言模型在为药物使用者提供减少伤害信息时的准确性与安全风险。 |
| [^185] | [Using Large Language Models for Legal Decision-Making in Austrian Value-Added Tax Law: A Comparative Study](https://arxiv.org/abs/2507.08468) | 本文通过微调和检索增强生成（RAG）两种方法，在教科书案例与真实税务咨询案例上系统评估了大型语言模型在奥地利及欧盟增值税法律决策中的能力，确定了LLM系统的最佳配置并检验了其法律推理能力。 |
| [^186] | [DLM-One: Diffusion Language Models for One-Step Sequence Generation](https://arxiv.org/abs/2506.00290) | DLM-One提出了一种基于分数蒸馏的框架，将扩散语言模型的生成过程压缩为单步，实现采样步数约2000倍、推理时间约500倍的加速，同时保持有竞争力的文本生成性能。 |
| [^187] | [SocialMaze: A Benchmark for Evaluating and Enhancing Social Reasoning in Large Language Models in Complex Social Environments](https://arxiv.org/abs/2505.23713) | 该论文提出了SocialMaze基准，通过深度推理、动态交互和信息不确定性三个设计维度，在社交推理游戏、日常互动和数字社区平台等六项任务中评估并提升大型语言模型在复杂社会环境中的社会推理能力。 |
| [^188] | [Modeling and Optimizing User Preferences in AI Copilots: A Comprehensive Survey and Taxonomy](https://arxiv.org/abs/2505.21907) | 本综述系统梳理并分类了AI副驾系统中用户偏好信号的获取、跨交互阶段建模及反馈优化方法，以实现个性化。 |
| [^189] | [When Can Large Reasoning Models Save Thinking? Mechanistic Analysis of Behavioral Divergence in Reasoning](https://arxiv.org/abs/2505.15276) | 该论文从思考终止边界置信度、内部注意力分布分歧和注意力分配三个机制角度，揭示了大型推理模型在“节省思考”提示下仍持续思考的原因（高困惑度及对原始问题的过多注意力），并提出注意力干预方法虽能抑制思考但会降低准确率。 |
| [^190] | [Multimodal Language Models as Text-to-Image Model Evaluators](https://arxiv.org/abs/2505.00759) | 提出MT2IE评估框架，让单个多模态大语言模型作为评估代理迭代生成提示词并给图像评分，其与人类判断的相关性高于现有指标，且仅需20个提示词即可高效复现T2I模型官方排名。 |
| [^191] | [Elite political incivility is rising across democracies](https://arxiv.org/abs/2503.22411) | 通过用大语言模型分析26个国家议员的1380万条推文，发现精英政治不文明行为在2017至2022年间几乎翻倍，且这一上升主要源于各党派行为的普遍激进化，而非激进政党崛起的结构性变化。 |
| [^192] | [Evaluating the Evaluator: Summarization Metrics and LLM-Judges beyond English](https://arxiv.org/abs/2503.17039) | 本文构建了首个超越英语的多语言摘要元评估数据集BASSE，基于2,040个摘要的人类判断对自动评估指标和LLM裁判进行基准测试，发现专有LLM裁判与人类判断相关性最高，其次是特定标准的自动指标。 |
| [^193] | [Beyond-RAG: Question Identification and Answer Generation in Real-Time Conversations](https://arxiv.org/abs/2410.10136) | 该论文提出了一个超越传统RAG的实时决策支持系统，通过先识别客户问题并判断其是否匹配FAQ数据库来直接检索答案或经由RAG生成答案，将响应时间缩短至2秒以内，显著减轻了客服人员的查询负担。 |
| [^194] | [Prompting the Unknown: Understanding Response Uncertainty in Large Language Models](https://arxiv.org/abs/2407.14845) | 该论文提出了一个提示-响应概念模型，识别出大语言模型响应不确定性的四个来源（提示规范不足、模型质量、任务变异性和语义冗余），并证明了提高提示信息性或模型质量可以降低响应不确定性。 |
| [^195] | [A Survey of Transformer-based Language Models with Focus on Efficiency](https://arxiv.org/abs/2406.16893) | 本文从效率视角系统综述了312篇关于基于Transformer的大语言模型的文献，全面梳理了数据整理、模型设计、模型缩减、动态推理以及预训练、微调、提示工程和RAG等适配策略中的效率提升方法。 |
| [^196] | [GPTBIAS: A Comprehensive Framework for Evaluating Bias in Large Language Models](https://arxiv.org/abs/2312.06315) | 本文提出了GPTBIAS框架，利用GPT-4等高性能大语言模型来评估其他模型的社会偏见，并设计了专门用于偏见评估的“偏见攻击指令”提示词，从而提升了偏见评估的可信度和可解释性。 |

# 详细

[^1]: 用户反馈提供了大语言模型自身无法检测到的独特信号

    User Feedback Provides a Unique Signal that LLMs Can not Detect

    [https://arxiv.org/abs/2609.02859](https://arxiv.org/abs/2609.02859)

    用户反馈是LLM改进的高价值学习信号，其看似无效源于评估范式的系统性偏差——模型自身无法察觉反馈带来的改进，而实验证明基于反馈的修订能以显著更高的比率解决目标问题。

    

    利用用户交互中自然产生的反馈，为大语言模型（LLM）提供了一种极具前景的学习信号。然而，近期的研究表明，这种反馈本质上充满噪声，难以被有效利用。我们通过证明用户反馈实际上是一种高度可操作、可用于改进的信号，对这一观念提出挑战，并指出其“看似无效”源于当前评估范式中的一种系统性偏差。为了分离出反馈本身的有效性，我们构建了具有明确真值（ground truth）的合成数据，同时辅以贴近真实场景的自然数据，以验证我们的发现在现实环境中同样成立。通过在两种设置下对比“有反馈辅助”与“无反馈”情况下生成的模型修订版本，我们发现：基于反馈的修订能够以显著更高的比率解决目标问题。最后，我们揭示了这一评估偏差的根源：当模型在反馈的帮助下成功修复了某个问题时，模型本身却无法检测到反馈所起的作用——也就是说，用户反馈提供了LLM自身无法察觉的独特信号。

    arXiv:2609.02859v1 Announce Type: new  Abstract: Harnessing naturally occurring feedback from user interactions offers a promising learning signal for Large Language Models (LLMs). However, recent studies suggest this feedback is inherently noisy and difficult to leverage effectively. We challenge this conception by demonstrating that user feedback is a highly actionable signal for improvement, and that its perceived ineffectiveness stems from a systematic bias in current evaluation paradigms. To isolate the usefulness of feedback, we construct synthetic data with a definitive ground truth, alongside naturalistic data to validate that our findings hold in real-world scenarios. By comparing model revisions generated with and without access to feedback across both settings, we show that feedback-informed revisions resolve targeted issues at significantly higher rates than baseline revisions. Finally, we expose the root of the evaluation bias: when a model successfully fixes an issue excl
    
[^2]: 面向编程竞赛金牌表现的语言模型后训练

    Post-Training Language Models for Gold-Medal Performance in Coding Competitions

    [https://arxiv.org/abs/2609.02849](https://arxiv.org/abs/2609.02849)

    该研究通过结合大规模题目筛选、监督微调、强化学习以及反馈驱动的测试时计算策略 GenCorrect，使语言模型在 IOI 2025 编程竞赛中取得了超越金牌分数线（438.3 分）的成绩（Nano-CC 达 468 分，Ultra-CC 达 502 分）。

    

    竞赛编程已成为检验大语言模型推理能力的关键测试，其中 IOI 和 ICPC 等国际赛事代表了最具挑战性的场景。我们提出了一条端到端的专门化流水线，结合了大规模题目筛选、合成推理轨迹、监督微调（SFT）和强化学习（RL）。利用 22,000 道精选题目，我们通过 SFT 和 RL 训练了 Nemotron-3-Nano-CC（30B-A3B），并仅通过 SFT 训练了 Nemotron-3-Ultra-CC（550B-A55B）。我们进一步提出了 GenCorrect，这是一种由反馈驱动的测试时计算策略，可迭代地生成、评估并改进多样化的解决方案。在 IOI 2025 上，Nano-CC 在后训练后从 130 分提升至 291 分，结合 GenCorrect 后达到 468 分，超过了 438.3 的金牌分数线，而 Ultra-CC 达到了 502 分。在这些结果的指导下，我们开发了一个面向竞赛的 Ultra-CC 系统，并在 IOI 2026 期间进行了前瞻性评估。

    arXiv:2609.02849v1 Announce Type: cross  Abstract: Competitive programming has become a key test of large language model reasoning, with international competitions such as IOI and ICPC representing its most challenging settings. We present an end-to-end specialization pipeline combining large-scale problem curation, synthetic reasoning traces, supervised fine-tuning (SFT), and reinforcement learning (RL). Using 22,000 curated problems, we train Nemotron-3-Nano-CC (30B-A3B) with SFT and RL and Nemotron-3-Ultra-CC (550B-A55B) with SFT alone. We further introduce GenCorrect, a feedback-driven test-time compute strategy that iteratively generates, evaluates, and refines diverse solutions. On IOI 2025, Nano-CC improves from 130 points to 291 after post-training and to 468 with GenCorrect, exceeding the gold threshold of 438.3 while Ultra-CC reaches 502. Guided by these results, we develop a competition-specific Ultra-CC system and evaluate it prospectively during IOI 2026. Under the same ti
    
[^3]: 面向语言模型的荷兰赌

    Dutch Books for Language Models

    [https://arxiv.org/abs/2609.02797](https://arxiv.org/abs/2609.02797)

    该论文基于德·菲内蒂定理提出一种利用线性规划计算荷兰赌利润的评估方法，无需真实结果标签即可量化语言模型概率预测的不连贯性，并发现语言模型预测存在显著的不连贯现象。

    

    人们越来越多地使用语言模型来辅助生活决策。许多此类决策涉及概率预测：某个重大生活事件、自然灾害或经济结果发生的可能性有多大？语言模型的用户可能默认这些预测源自一个连贯一致的世界模型。在本文中，我们通过一个基于德·菲内蒂定理的评估程序来检验语言模型概率预测的连贯性。我们让语言模型对基于股票收益数据生成的事件做出预测，然后利用线性规划计算最大的荷兰赌利润——即套利者通过针对模型生成的概率下注所能确保获得的利润——并将其作为衡量不连贯性的指标。我们的评估方法不需要真实结果标签，因此即使在结果尚未被观测或尚未确定的情况下，也能对预测的连贯性进行评估。我们发现语言模型的预测中存在大量不连贯性的证据。

    arXiv:2609.02797v1 Announce Type: cross  Abstract: People increasingly use language models to support life decisions. Many such decisions involve a probabilistic forecast: How likely is a major life event, a natural disaster, or an economic outcome? Users of language models may implicitly trust that these forecasts fall out of a coherent world model. In this paper, we evaluate the coherence of language model probabilistic forecasts through a procedure that builds on a theorem due to de Finetti. We elicit forecasts from language models across events generated from stock returns data. We then use linear programs to compute the largest Dutch-book profit - the profit an arbitrageur could guarantee by betting against model-generated probabilities - which we use as a measure of incoherence. Our procedure does not require outcome labels, so we can evaluate coherence even in settings where outcomes are not observed or have not yet resolved. We find substantial evidence of incoherence in langua
    
[^4]: DiscoSign：语篇感知的文本到手语词汇注释翻译

    DiscoSign: Discourse-Aware Text to Sign Language Gloss Translation

    [https://arxiv.org/abs/2609.02796](https://arxiv.org/abs/2609.02796)

    提出了DiscoSign，一种基于大语言模型的语篇感知文本到手语注释翻译框架，通过处理空间共指消解、问答从句和概念-注释一致性三种语篇现象，并引入新颖的语篇连贯性评估指标，突破了传统句子级手语翻译的局限。

    

    手语处理系统传统上在句子层面运作，忽略了对理解手语至关重要的语篇现象。我们提出了DiscoSign，一种基于语言学研究的、语篇感知的文本到手语词汇注释翻译的计算方法。我们在基于大语言模型（LLM）的模块化翻译框架中处理了三种关键现象：（i）空间共指消解，即实体在整个语篇中保持一致的空间位置；（ii）问答从句（QACs），即服务于特定语篇功能的假拟分裂结构；（iii）概念-注释一致性，确保英语概念与美国手语（ASL）符号之间的稳定映射。由于传统翻译评估指标无法捕捉语篇层面的质量，我们引入了一套新颖的评估指标，旨在评估我们框架所处理的语篇连贯性的各个维度。

    arXiv:2609.02796v1 Announce Type: new  Abstract: Sign language processing systems have traditionally operated at the sentence level, ignoring critical discourse phenomena fundamental to sign language comprehension. We introduce DiscoSign, a computational approach for discourse-aware text to sign language gloss translation grounded in linguistic research. We address three key phenomena within our modular Large Language Model (LLM)-based translation framework: (i) spatial coreference resolution, where entities maintain consistent spatial locations throughout discourse; (ii) Question-Answer Clauses (QACs), pseudocleft structures serving specific discourse functions; and (iii) concept-gloss consistency, ensuring stable mappings between English concepts and American Sign Language (ASL) signs. Traditional translation metrics fail to capture discourse-level quality, so we introduce a suite of novel evaluation metrics designed to assess each dimension of discourse coherence addressed by our fr
    
[^5]: EarlyEval：通过早期结果预测实现更廉价的智能体评估

    EarlyEval: Cheaper Agent Evaluation via Early Outcome Prediction

    [https://arxiv.org/abs/2609.02783](https://arxiv.org/abs/2609.02783)

    该论文提出EarlyEval框架，通过基于智能体中间行为训练轻量级的成功/失败分类器来提前预测任务结果并及早中止运行，从而在单个任务内部削减成本，大幅降低LLM智能体的评估开销。

    

    评估大语言模型智能体对于指导其开发至关重要，但其成本已变得过于高昂：前沿模型在智能体基准测试上完整运行一次可能花费数百至数千美元，而在迭代开发周期中这一成本需要反复支付。此前以基准蒸馏为核心的工作减少了评估任务的数量，但并未降低执行每个保留任务的成本。在本工作中，我们提出了早期结果预测这一方法，这是一种互补的效率提升维度，改为在单个任务内部削减成本。我们的关键洞察是：智能体的最终结果往往在执行完成之前就能从其中间行为中显现出来。我们将这一想法实例化为EarlyEval——一个轻量级框架，它基于行为特征、文本特征和参考解法特征训练一对LightGBM成功与失败分类器，并在任一分类器达到校准置信度阈值时立即中止智能体的运行。

    arXiv:2609.02783v1 Announce Type: new  Abstract: Evaluating LLM agents is essential for guiding their development, yet it has grown prohibitively expensive: a single pass of a frontier model over an agentic benchmark can cost hundreds to thousands of dollars, a price paid repeatedly across iterative development cycles. Prior efforts, centered on benchmark distillation, reduce the number of evaluation tasks but leave the cost of executing each retained task untouched. In this work, we introduce early outcome prediction, a complementary axis of efficiency that instead cuts cost within each task. Our key insight is that an agent's final outcome is often evident from its intermediate behavior well before execution completes. We instantiate this idea in EarlyEval, a lightweight framework that trains a pair of LightGBM success and failure classifiers over behavioral, textual, and reference-solution features, and halts an agent run the moment either classifier crosses a calibrated confidence 
    
[^6]: HyperStyler：基于上下文感知风格导航与超网络的低资源作者风格迁移

    HyperStyler: Low-resource Authorship Style Transfer via Context-aware Style Navigation and Hypernetworks

    [https://arxiv.org/abs/2609.02772](https://arxiv.org/abs/2609.02772)

    HyperStyler将低资源作者风格迁移解耦为风格选择与风格实现两个阶段，通过上下文感知的风格导航器预测风格坐标，并利用超网络进行动态参数调制来避免风格与内容纠缠，从而同时实现高风格保真度与语义保留。

    

    低资源作者风格迁移（LAST）旨在仅使用少量参考样本，将文本改写为任意目标作者的写作风格，同时保留原文含义。现有方法往往难以同时实现高风格保真度和语义保留，因为它们将多样化的参考文献压缩为单一静态的作者嵌入，从而平均化了上下文相关的风格变化；同时它们依赖隐藏表示进行风格控制，导致风格与内容相互纠缠。我们提出了HyperStyler，这是一种新颖的架构，将LAST解耦为风格选择和风格实现两个阶段。其中，Stylo-navigator通过联合建模源文本上下文和目标作者参考文献来预测风格坐标，Stylo-hypernet则通过动态参数调制（而非隐藏状态注入）来实现这些风格坐标。我们在Reddit、Blog和News数据集上的实验表明，HyperStyler在相关指标上始终优于先前的各种方法。

    arXiv:2609.02772v1 Announce Type: new  Abstract: Low-resource authorship style transfer (LAST) aims to rewrite text into the style of an arbitrary target author using only a few reference examples while preserving the original meaning. Existing methods often struggle to achieve both high style fidelity and semantic preservation because they compress diverse references into a single static author embedding, which averages out context-dependent stylistic variation, and rely on hidden representations for style control, which entangle style with content. We propose HyperStyler, a novel architecture that decouples LAST into style selection and style realization. Stylo-navigator predicts style coordinates by jointly modeling the source context and target-author references, and Stylo-hypernet realizes them via dynamic parameter modulation instead of hidden-state injection. Our experiments on Reddit, Blog, and News datasets demonstrate that HyperStyler consistently outperforms prior methods in
    
[^7]: 从重加权到重写：解锁训练数据归因中影响力样本的干预效果

    From Reweighting to Rewriting: Unlocking the Intervention Effects of Influential Samples in Training Data Attribution

    [https://arxiv.org/abs/2609.02771](https://arxiv.org/abs/2609.02771)

    该论文发现重加权无法释放影响函数所识别样本的干预价值，并提出“影响引导的响应重写”方法——通过重写所选样本的响应而非调整其权重，从而真正解锁训练数据归因中影响力样本的干预效果。

    

    训练数据归因（TDA）旨在识别塑造模型行为的训练样本，但其干预价值既取决于选择了哪些样本，也取决于如何对这些样本进行修改。影响函数（IF）估计的是无穷小重加权下的行为变化，然而在常规的基于权重的干预下，由IF选出的样本相较于随机选择的样本往往优势有限。这引出了一个问题：是有影响力的样本本身缺乏干预价值，还是重加权方式未能实现其行为杠杆作用。我们提出影响引导的响应重写方法，该方法利用影响函数识别干预目标，并在保持指令不变的情况下，将其响应替换为与目标行为一致或相反的监督信号。我们在四个开源权重的大语言模型上，以认知性弃答为主要测试场景，在相同的影响选择样本上比较了重写与重加权两种干预方式。响应重写产生了

    arXiv:2609.02771v1 Announce Type: cross  Abstract: Training data attribution (TDA) aims to identify training examples that shape model behavior, but its intervention value depends on both which examples are selected and how they are modified. Influence functions (IF) estimate behavioral changes under infinitesimal reweighting, yet IF-selected examples often show limited advantages over random selection under conventional weight-based interventions. This raises the question of whether influential examples lack intervention value or whether reweighting fails to realize their behavioral leverage.We introduce influence-guided response rewriting, which uses IF to identify intervention targets and replaces their responses with behavior-aligned or behavior-opposed supervision while keeping instructions fixed. Across four open-weight LLMs, we compare rewriting and reweighting on the same influence-selected examples using epistemic abstention as our primary testbed. Response rewriting produces 
    
[^8]: 剖析医学问答中误导性上下文的作用机制

    Untangling the Mechanisms of Misleading Context in Medical Question Answering

    [https://arxiv.org/abs/2609.02754](https://arxiv.org/abs/2609.02754)

    该研究通过在MedMisBench中注入伪造证据和纯粹断言两类误导性上下文，系统揭示了推理模型的医学判断被误导的机制，发现模型对纯粹断言的易感性显著高于伪造证据（高出10至27个百分点），且误导信息虽在推理轨迹中被大量披露却难以被察觉。

    

    大语言模型如今能够以专家级水平回答医学问题。然而，这些系统所依据的上下文可能具有误导性，而误导性上下文会破坏模型的医学判断。为了理解误导性上下文如何破坏这一判断，我们考察了模型对上下文的易感性、对误导信息的披露程度、推理被破坏的机制以及决策的可监控性。在一个由临床医生审核、包含8,627个问题的问答基准MedMisBench的医学推理子集上，我们注入了两类误导性上下文线索：伪造的证据和纯粹的断言。我们测试了三个推理模型，其中两个会暴露其完整的推理轨迹，另一个前沿模型仅暴露其最终回复。所有三个模型都更容易受纯粹断言而非伪造证据的影响，采用断言答案的频率高出10至27个百分点。误导性线索在81%至98%的推理轨迹中被披露，但仅……

    arXiv:2609.02754v1 Announce Type: cross  Abstract: Large language models now answer medical questions with expert-level performance. However, the context these systems act on can be misleading, and misleading context can corrupt a model's medical judgment. To understand how misleading context corrupts this judgment, we examine the model's susceptibility to the context, disclosure of it, mechanism of corrupted reasoning, and monitorability of the decision. On the medical reasoning subset of MedMisBench, a clinician-reviewed question-answering benchmark of 8,627 questions, we inject two types of misleading context cues, fabricated evidence and a bare assertion. We test three reasoning models, two that expose their full reasoning trace and one frontier model that exposes only its response. All three are more susceptible to the assertion than to the fabricated evidence, adopting the asserted answer 10 to 27 points more often. The misleading cues are disclosed in 81 to 98% of traces but onl
    
[^9]: Repo-To-Skill：将GitHub仓库蒸馏为AI4AI技能

    Repo-To-Skill: Distilling GitHub Repositories Into AI4AI Skills

    [https://arxiv.org/abs/2609.02749](https://arxiv.org/abs/2609.02749)

    提出DisCo技能驱动型研究智能体，通过任务无关与任务导向两种蒸馏方式，将GitHub仓库中的操作性知识转化为紧凑、经过验证且可跨任务复用的AI4AI技能。

    

    自主智能体已开始能够端到端地开展机器学习（ML）研究。这类智能体将模型骨干与用于规划、执行、记忆和验证的框架相结合，但这种架构仍然把领域专业知识排除在智能体之外。我们将这一缺失的层面称为“操作性知识”，即区分“了解一种方法”与“真正使其奏效”的诀窍。这些知识并非在领域中缺失，它们存在于仓库和论文之中，只是以面向人类读者的形式呈现，且体量过大，无法在执行任务时加载。一旦将这些知识蒸馏为紧凑且经过验证的技能，它们就可以跨任务复用，而无需在每次运行中重新摸索。我们提出了DisCo，一个由技能驱动的研究智能体，它既能创建技能，又能在研究过程中使用这些技能。其蒸馏以两种互补的形式进行：任务无关型，将该领域广泛使用的仓库浓缩为可复用的技能；以及任务导向型，为具体任务生成所需的技能。

    arXiv:2609.02749v1 Announce Type: new  Abstract: Autonomous agents are beginning to carry out machine-learning (ML) research end to end. These agents combine a model backbone with a harness for planning, execution, memory, and verification, but this architecture still leaves domain-specific know-how outside the agent. We call this missing layer operational knowledge, the know-how that separates knowing a method from making it work. That knowledge is not absent from the field. It appears in repositories and papers, but in forms written for human readers and too large to load during a task. Once distilled into compact, verified skills, this knowledge can be reused across tasks rather than rediscovered during each run.   We present DisCo, a skill-powered research agent that creates skills and uses them during research. Its distillation runs in two complementary forms: task-agnostic, condensing the field's widely used repositories into reusable skills, and task-oriented, producing the skil
    
[^10]: 面向低成本检索模型选择的增量式池化LLM评估

    Incremental Pooled LLM Evaluation for Cost-Effective Retrieval Model Selection

    [https://arxiv.org/abs/2609.02745](https://arxiv.org/abs/2609.02745)

    提出增量式池化LLM评估方法，通过LLM判断候选系统检索文档的并集并随新系统增量扩展文档池，实现低成本、可复用的检索模型对比评估，其排序结果与金标准评估高度一致。

    

    为生产环境的RAG系统选择检索模型需要可靠的对比评估，但大规模获取相关性判断代价高昂，且随着新候选系统的出现难以重复进行。我们研究了池化LLM评估方法，即由LLM对当前候选系统集合所检索文档的并集进行判断，并随着新系统的引入，通过仅判断其贡献的新文档来增量扩展文档池。这些判断结果被重复利用，从而在共同基础上评估所有系统。我们在四个检索基准上验证了该方法，涵盖密集、稀疏和混合配置的11个系统，并将其部署用于比较一个金融新闻问答系统的62种检索配置。池化LLM排序在各数据集上与金标准评估高度相关，且在考虑qrels的bootstrap不确定性后，97%的系统两两排序得以保持。

    arXiv:2609.02745v1 Announce Type: cross  Abstract: Selecting a retrieval model for a production RAG system requires reliable comparative evaluation, but obtaining relevance judgments at scale is expensive and difficult to repeat as new candidate systems arrive. We study pooled LLM evaluation, in which an LLM judges the union of documents retrieved by the current set of candidate systems, and the pool is then expanded incrementally as new systems are introduced by judging only the new documents they contribute. These judgments are reused to evaluate all systems on a common basis. We validate this approach on four retrieval benchmarks with 11 systems spanning dense, sparse, and hybrid configurations, and deploy it to compare 62 retrieval configurations for a financial news QA system. Pooled LLM rankings correlate strongly with gold-standard evaluation across datasets, and 97% of pairwise system orderings are preserved once bootstrap uncertainty in the qrels is taken into account. In prod
    
[^11]: 语言模型可以控制自己的注意力

    Language Models Can Control Their Own Attention

    [https://arxiv.org/abs/2609.02737](https://arxiv.org/abs/2609.02737)

    该论文提出“声明式注意力”协议，让语言模型在思维链中自主声明需要关注的上下文区域，推理引擎据此像解析工具调用一样跳过大部分KV缓存读取，从而以内在方式避免了外部评分方法每步O(N)的开销。

    

    语言模型将大部分注意力集中在上下文的一小部分上，然而它们却要读取整个KV缓存来找出少数重要的token。如果用户在100万token的对话中询问之前的某个细节，全局注意力层必须扫描完整上下文才能生成回复的每一个token。一种著名的方法通过轻量级代理分数预先选择相关token来缓解这一成本，但这种外部评分机制在每一步仍然会产生O(N)的开销。我们采取一种内在的方法，其动机源于一个简单的问题：模型难道不是已经知道上下文的哪些部分是相关的吗？为此，我们引入了声明式注意力，这是一种协议，可引导模型在其思维链中声明它需要关注的位置，将生成过程划分为三种模式：（完整上下文）、（特定区域）和（仅最近输出）。推理引擎像解析工具调用一样解析这些声明，并跳过大部分KV……

    arXiv:2609.02737v1 Announce Type: cross  Abstract: Language models spend most of their attention on a small fraction of context, yet they read the entire KV cache to find the few tokens that matter. If the user asks about a previous detail in a 1M-token conversation, global attention layers must scan the full context to generate each token of the reply. A prominent approach mitigates this cost by pre-selecting relevant tokens via lightweight proxy scores, but this extrinsic scoring still incurs O(N) per step. We take an intrinsic approach motivated by the simple question: wouldn't the model already know which parts of the context are relevant? To this end, we introduce Declarative Attention (DA), a protocol that elicits the model to declare where it needs to attend within its chain-of-thought, partitioning generation into three modes:  (full context),  (a specific region), and  (recent output only). The inference engine parses these declarations like tool calls and skips most of the KV
    
[^12]: 为每位患者构音障碍语音识别选择PEFT变体：基于两个ASR基础模型的单说话人案例研究

    Choosing a PEFT Variant for Per-Patient Dysarthric ASR: A Single-Speaker Case Study on Two ASR Bases

    [https://arxiv.org/abs/2609.02735](https://arxiv.org/abs/2609.02735)

    该单说话人案例研究比较了七种LoRA系列PEFT方法在两个ASR基础模型上的构音障碍语音识别表现，发现注意力投影适配器能显著降低字符错误率，且更简单廉价的LoRA与DoRA性能无显著差异，因而推荐采用LoRA。

    

    每位患者专用适配器是构音障碍自动语音识别（ASR）的首选生产架构，然而参数高效微调（PEFT）变体尚未在说话人相关的每位患者场景中进行过比较。我们提出了一项单说话人案例研究，在两个生产级基础模型（经过匈牙利语微调的Whisper-large-v3和多语言Qwen3-ASR-1.7B检查点）上，比较了七种LoRA系列方法（LoRA、QLoRA、AdaLoRA、DoRA、LoHA、VeRA、VB-LoRA），研究对象为一名中风后的匈牙利语男性说话人（S1，409条语音；临床听觉感知评估显示为重度构音障碍）。注意力投影适配器在两个基础模型上都显著改善了字符错误率（CER）。在三个随机种子下，配对自助法检验未发现LoRA与DoRA之间存在显著差异（p>0.5；Whisper上CER为13.86/13.90%，Qwen3-ASR上为28.10/28.33%），因此我们采用了更简单、成本更低的LoRA。真正的4位（NF4）QLoRA在每个种子和两个基础模型上的表现都更差（CER为14.56/30.09%），且没有……

    arXiv:2609.02735v1 Announce Type: new  Abstract: Per-patient adapters are the preferred production architecture for dysarthric automatic speech recognition (ASR), yet parameter-efficient fine-tuning (PEFT) variants have not been compared in the speaker-dependent, per-patient regime. We present a single-speaker case study comparing seven LoRA-family methods (LoRA, QLoRA, AdaLoRA, DoRA, LoHA, VeRA, VB-LoRA) on two production bases (Whisper-large-v3 with Hungarian fine-tuning, and a multilingual Qwen3-ASR-1.7B checkpoint) for one post-stroke Hungarian male speaker (S1, 409 utterances; severe dysarthria on auditory-perceptual clinical assessment). Attention-projection adapters substantially improve CER on both bases. Across three seeds, a paired bootstrap detects no significant LoRA-DoRA difference (p>0.5; 13.86/13.90 % CER on Whisper, 28.10/28.33 % on Qwen3-ASR), so we adopt the simpler, cheaper LoRA. Real 4-bit (NF4) QLoRA is worse on every seed and both bases (14.56/30.09 % CER) with no
    
[^13]: CORAL：面向生产推荐系统的LLM原生框架

    CORAL: An LLM-Native Harness for Production Recommender Systems

    [https://arxiv.org/abs/2609.02730](https://arxiv.org/abs/2609.02730)

    CORAL是一个LLM原生闭环框架，让智能体持续观察线上推荐系统的运行信号、基于过往决策与结果的记忆进行推理并调用工具，从决策的实际效果中学习，从而实现生产级推荐系统的持续自动化优化。

    

    生产级推荐系统塑造着数十亿人所看到的内容，维持其性能需要持续优化：随着内容、用户行为和上游模型的变化，管理检索、排序和服务的各项决策必须被不断重新审视。传统上，人类工程师通过在线实验来测试这些变更——这是一个缓慢、被动的流程，受限于工程人力，导致系统中的部分环节在环境变化时未能得到及时调整。尽管大语言模型已被应用于排序、用户建模和离线模型开发，但很少有系统将一个智能体置于持续的闭环中，使其作用于线上推荐系统并从其决策的实际效果中学习。我们提出了CORAL（通过智能体循环实现约束优化的推荐系统，Constraint-Optimized Recommender via an Agentic Loop），一个LLM原生的管控框架，它闭合了这一循环：在每轮周期中，智能体观察运行信号，基于对过往决策及其结果的记忆进行推理，并调用工具——包括……（原文摘要在此处截断）

    arXiv:2609.02730v1 Announce Type: new  Abstract: Production recommender systems shape what billions of people see, and sustaining their performance requires continual optimization: as content, user behavior, and upstream models shift, the choices governing retrieval, ranking, and serving must be revisited. Traditionally, human engineers test such changes through online experiments--a slow, reactive process limited by engineering effort, leaving parts of the system unrevised as conditions change. Although large language models have been applied to ranking, user modeling, and offline model development, few systems place an agent in a continual closed loop that acts on a live recommender and learns from the measured effects of its decisions. We present CORAL (Constraint-Optimized Recommender via an Agentic Loop), an LLM-native harness that closes this loop: each cycle, the agent observes operating signals, reasons over a memory of past decisions and outcomes, and invokes tools--including 
    
[^14]: 留面子式请求与大语言模型的拒绝行为

    Door-in-the-Face Requests and Refusal Behaviour in Large Language Models

    [https://arxiv.org/abs/2609.02707](https://arxiv.org/abs/2609.02707)

    该研究发现“留面子”说服技术对不同大语言模型的效果截然不同：它在Anthropic前沿模型上能将较小请求的依从率从29.3%提升至65.8%，但在OpenAI和Google的前沿模型上反而使依从率降低15.5至23.0个百分点。

    

    arXiv:2609.02707v1 公告类型：新 摘要：留面子技术对语言模型有效吗？在人类中，一个被拒绝的大请求会使随后的较小请求更容易被接受。我们在来自三家提供商的九个生产级模型上对此进行了测试：每个模型先拒绝一个大请求，随后收到同一请求的较小版本，我们将其依从性与直接提问的情况进行比较。答案因模型而异。在Anthropic的前沿模型上，该技术有效：Opus 5在拒绝较大请求后，有65.8%的概率回答较小的请求，而直接提问时仅为29.3%。而在OpenAI和Google的前沿模型以及Haiku 4.5上，该技术适得其反，使依从性降低了15.5至23.0个百分点。一项对照实验定位了这一效应：在无关主题上被拒绝的大请求，在所有九个模型上产生的影响都小于相关主题的请求，因此让步本身在任何模型上都起作用，而刚拒绝某事后所产生的反应则因模型家族而异。

    arXiv:2609.02707v1 Announce Type: new  Abstract: Does the door-in-the-face technique work on language models? In humans, a large request that is refused makes a smaller follow-up request more likely to be granted. We test this on nine production models from three providers: each model refuses a large request, then receives a smaller version of the same request, and we compare its compliance with asking directly. The answer depends on the model. On Anthropic's frontier models the technique works: Opus 5 answers the smaller request 65.8% of the time after refusing the larger one, against 29.3% when asked directly. On the frontier models of OpenAI and Google, and on Haiku 4.5, it backfires, lowering compliance by 15.5 to 23.0 points. A control locates the effect: a refused large request on an unrelated topic does less than the related one on all nine models, so the concession itself matters everywhere, while the reaction to having just refused something differs by model family. The techni
    
[^15]: 轨迹即状态：将推理轨迹作为长上下文Transformer的条件状态

    Trace as State: Reasoning Traces as Conditional States for Long-Context Transformers

    [https://arxiv.org/abs/2609.02702](https://arxiv.org/abs/2609.02702)

    提出Trace as State方法，将推理轨迹作为任务状态的文本代理置于长上下文之前以指导模型重读，在27个模型-任务-指标组合中的26个上优于将轨迹置于上下文之后的对照方法。

    

    Transformer以因果方式处理信息，但长上下文推理可能依赖于只有在较晚时刻才能发现的任务状态。我们通过条件状态更新任务对这一不匹配进行了形式化：对于因果状态更新处理器，先提供条件在最坏情况下可能比最后提供条件节省指数级的内存。基于这一原理，我们提出了Trace as State（轨迹即状态）方法：我们将收集到的推理轨迹作为任务状态的文本代理，并在一次全新的处理中将它置于长上下文块之前，使先前推导出的信息能够指导模型重读上下文。我们对Trace as State以及Trace Append进行了大量实验，后者是一个匹配的对照方法，使用相同的任务状态代理但将其置于上下文之后。在三个模型和三个长上下文数据集上，Trace as State在27个报告的模型、任务和指标组合中的26个上优于Trace Append。在GraphWalks Parents任务上，精确匹配率使DeepSeek V4 Pro Previ……

    arXiv:2609.02702v1 Announce Type: new  Abstract: Transformers process information causally, but long-context reasoning may depend on task state discovered only later. We formalize this mismatch through conditional state update tasks. For causal state update processors, providing the condition first can require exponentially less memory in the worst case than providing it last.   Motivated by this principle, we introduce Trace as State. We use collected reasoning traces as a textual proxy for task state and place it before the long-context block on a fresh pass, allowing information derived previously to guide rereading.   We conduct extensive experiments on Trace as State and Trace Append, a matched control that uses the same task state proxy but put it after the context. Across three models and three long-context datasets, Trace as State outperforms Trace Append in 26 of 27 reported combinations of model, task, and metric. On GraphWalks Parents, exact match lifts DeepSeek V4 Pro Previ
    
[^16]: DKL：面向指令微调语言模型的解耦知识学习

    DKL: Decoupled Knowledge Learning for Instruction-Tuned Language Models

    [https://arxiv.org/abs/2609.02685](https://arxiv.org/abs/2609.02685)

    提出DKL解耦知识学习方法，能够在不损害指令遵循能力、也无需生成海量合成问答数据的情况下，将新语料库知识注入指令微调语言模型，从而缓解RAG在检索失败时的幻觉问题。

    

    RAG（检索增强生成）已成为将新的、特定语料库知识融入遵循指令的大语言模型（Instruct LLM）的事实标准方法。尽管基于RAG的提示改善了事实依据，但当检索不正确或不完整时它会失效，从而导致幻觉。RAFT和PA-RAG等微调方法通过将新知识注入模型参数来增强RAG，但需要生成覆盖整个语料库的海量合成问答数据。在文本语料库上进行扩展预训练（EPT）虽然避免了全面合成数据生成的需要，但会损害Instruct LLM的指令遵循能力，因此需要在预训练之后进行指令微调（IFT）。然而，IFT成本高昂，并且由于指令微调语料库的不可获得性而可能无法实施。在这项工作中，我们提出了DKL——面向指令微调语言模型的解耦知识学习方法。与对Instruct LLM进行EPT不同……（摘要在此处截断）

    arXiv:2609.02685v1 Announce Type: cross  Abstract: RAG has become the de facto method for incorporating new, corpus-specific knowledge into an instruction following LLM (Instruct LLM). Although RAG-based prompting improves factual grounding, it fails when retrieval is incorrect or incomplete, leading to hallucinations. Finetuning methods such as RAFT and PA-RAG enhance RAG by injecting new knowledge into the model's parameters, but require generating a massive amount of synthetic QA that covers the entire corpus. Extended Pre-Training (EPT) on the text corpus avoids the need for comprehensive synthetic data generation but compromises an Instruct LLM's instruction-following capabilities, necessitating instruction fine-tuning (IFT) after pre-training. However, IFT is costly and may be infeasible due to the unavailability of an instruction-tuning corpus. In this work, we propose DKL-Decoupled Knowledge Learning for Instruction-Tuned Language Models. Instead of doing EPT on the Instruct LL
    
[^17]: 从词元到语义：利用互补信号检测黑盒大语言模型中的幻觉

    From Tokens to Semantics: Leveraging Complementary Signals for Hallucination Detection in Black-Box LLMs

    [https://arxiv.org/abs/2609.02679](https://arxiv.org/abs/2609.02679)

    该论文针对无参考文档的黑盒大语言模型，提出联合利用语义熵与词元级不确定性这两种互补信号（包括TopK聚合、CoCoA混合方法及Gated等监督方法）来更准确地检测幻觉。

    

    当大语言模型支持面向公众或高风险的工作流程时，遗漏的虚假编造内容可能损害用户和机构的利益，而误报则会消耗有限的人工审核资源。在缺乏可信上下文或参考文档的情况下，我们研究了两种可通过黑盒模型API获取的信号：语义熵（衡量采样响应含义之间的分歧程度）和由词元对数概率导出的不确定性。这两种信号的失效模式可以互补：当所有响应形成单一语义簇时，语义熵将失去信息量，而词元不确定性则可能遗漏模型始终自信的错误。我们通过TopK方法聚合采样响应中的词元级信号，扩展了基于词元的不确定性检测；评估了结合目标响应不确定性与语义差异性的混合方法CoCoA；并提出并研究了两种监督方法：Gated（将单簇情形路由至聚合的……

    arXiv:2609.02679v1 Announce Type: cross  Abstract: When LLMs support public-facing or high-stakes workflows, missed fabrications can harm users and institutions, while false alarms consume limited human-review capacity. When no trusted context or reference document is available, we study two signals accessible through black-box model APIs: semantic entropy, which measures disagreement among sampled response meanings, and uncertainty derived from token log-probabilities. Their failure modes can be complementary: semantic entropy becomes uninformative when responses form one semantic cluster, while token uncertainty can miss consistently confident errors. We extend token-based uncertainty detection by aggregating token-level signals across sampled responses through our TopK method, evaluate the hybrid CoCoA method, which combines target-response uncertainty with semantic dissimilarity, and propose and study two supervised methods: Gated, which routes single-cluster cases to an aggregated
    
[^18]: oHC：基于四元数在SO(4)流形上的正交超连接

    oHC: Orthogonal Hyper-Connections on SO(4) via Quaternions

    [https://arxiv.org/abs/2609.02672](https://arxiv.org/abs/2609.02672)

    该论文证明了双随机矩阵约束的混合会随深度耗尽残差流的多样性，并提出通过四元数在SO(4)流形上构造正交混合矩阵的oHC方法，既保证缩放稳定又完整保持残差流的范数与多样性。

    

    超连接用n条并行的残差流取代Transformer的单条残差流，并在每一层通过学习到的n×n残差矩阵对它们进行混合。若不对该矩阵施加约束，混合步骤对残差流的缩放因子便没有任何限制，且该因子会随层数不断累积，从而导致训练不稳定。流形约束超连接通过将矩阵限制为双随机矩阵来解决这一问题。这将缩放因子的上界限制为一，使混合无法再放大任何方向，但没有任何下界约束。我们证明，在该集合内，混合步骤只能通过缩小各残差流之间的差异来降低残差流的范数，而其均值保持不变；由于这种缩减会随层数累积，各残差流变得越来越相似，其多样性随网络深度被消耗殆尽。因此，我们提出正交超连接，……

    arXiv:2609.02672v1 Announce Type: new  Abstract: Hyper-Connections (HC) replace the single residual stream of a Transformer with $n$ parallel ones, mixing them at every layer with a learned $n \times n$ residual matrix. Leaving that matrix unconstrained places no limit on the factor by which the mixing step rescales the residual streams, and that factor compounds across layers, which destabilizes training. Manifold-constrained Hyper-Connections (mHC) address this by restricting the matrix to the doubly stochastic matrices. That caps the factor at one, so the mixing can no longer amplify any direction, but nothing bounds it from below. We prove that inside this set the mixing step can reduce the norm of the residual streams only by shrinking the differences between the streams, while their mean is left unchanged; and since the reduction accumulates over layers, the streams grow more alike and their diversity is spent with depth. We therefore propose Orthogonal Hyper-Connections (oHC), r
    
[^19]: WinoQueer-NL：评估荷兰语语言模型对LGBTQ+身份的偏见

    WinoQueer-NL: Assessing Bias in Dutch Language Models toward LGBTQ+ Identities

    [https://arxiv.org/abs/2609.02651](https://arxiv.org/abs/2609.02651)

    该研究构建了首个评估荷兰语语言模型对LGBTQ+身份偏见的基准数据集WinoQueer-NL，通过与荷兰酷儿群体的调查验证了145个文化相关刻板印象并新发现22种偏见，揭示了看似中性的平均偏见得分背后隐藏的显著偏见。

    

    尽管英语语言模型中的反酷儿偏见已被广泛研究，但荷兰语模型的相关研究仍然不足。为填补这一空白，我们基于英语WinoQueer基准开发了一个在文化和语言层面进行本地化适配的荷兰语数据集，其中包含刻板印象句与反刻板印象句的成对句子。为了验证并扩展该数据集，我们对43名荷兰酷儿群体参与者开展了在线调查，确认了171个刻板印象中的145个具有文化相关性，并通过自由文本回答识别出22种新偏见。最终发布的数据集包含42,906个句子，我们使用多种荷兰语专用模型和多语言模型对其进行评估，涵盖掩码语言模型（MLM）和自回归语言模型（ARLM），偏见通过比较刻板印象句与反刻板印象句的对数似然得分来衡量。虽然各模型的平均偏见得分看似中性（约50%），但更深入的分析揭示了显著的……（原文摘要到此截断）

    arXiv:2609.02651v1 Announce Type: new  Abstract: While English language models have been widely examined for anti-queer bias, Dutch models remain understudied. To address this gap, we developed a culturally and linguistically adapted Dutch dataset based on the English WinoQueer benchmark, containing pairs of stereotypical and counter-stereotypical sentences. To validate and expand it, we conducted an online survey with 43 Dutch queer participants, confirming 145 of 171 stereotypes as culturally relevant and identifying 22 new biases through free-text responses. The final released dataset, comprising 42,906 sentences, was evaluated using a range of Dutch-specific and multilingual models, including both masked language models (MLMs) and autoregressive language models (ARLMs), with bias measured via a score comparing log-likelihoods of stereotypical versus counter-stereotypical sentences. While the mean bias score across models appeared neutral (~50%), closer analysis revealed significant
    
[^20]: Loom：通过嵌入空间重加权将诊断线索编织成自由文本共识

    Loom: Weaving Diagnostic Strands into Free-Text Consensus via Embedding-Space Reweighting

    [https://arxiv.org/abs/2609.02649](https://arxiv.org/abs/2609.02649)

    Loom是一个部署于真实工业根因分析的生成式共识框架，它将模块化启发式产生的开放式诊断假设投影到连续嵌入空间，并通过基于质心的迭代重加权算法解决冲突信号，从而把嘈杂矛盾的文本假设聚合为可靠共识。

    

    将嘈杂且相互矛盾的文本假设聚合为可靠共识，是在真实工业场景中部署NLP系统时的一项根本性挑战。虽然单体式大语言模型（LLM）智能体为根因分析（RCA）等任务提供了无限的表达能力，但它们存在上下文长度限制、幻觉不断累积以及难以承受的推理延迟等问题。传统弱监督方法虽具备统计严谨性，但在数学上仅限于离散类别。我们提出了Loom，一个部署于真实世界根因分析的生成式共识框架，它弥合了上述两种范式。Loom通过将模块化启发式方法（由事件特定实体、时间和指标动态填充的诊断模板）产生的开放式假设投影到连续嵌入空间中进行聚合，并采用基于质心的迭代重加权算法来解决冲突信号。所得的共识权重为单一……（摘要原文在此处截断）

    arXiv:2609.02649v1 Announce Type: new  Abstract: Aggregating noisy, conflicting textual hypotheses into a reliable consensus is a fundamental challenge when deploying NLP systems in real-world industrial settings. While monolithic Large Language Model (LLM) agents offer unbounded expressivity for tasks like Root Cause Analysis (RCA), they suffer from context limits, compounding hallucinations, and prohibitive inference latency. Traditional weak supervision offers statistical rigor but is mathematically restricted to discrete classes. We present Loom, a generative consensus framework deployed for real-world RCA that bridges these paradigms. Loom aggregates open-form hypotheses emitted by modular heuristics (diagnostic templates dynamically populated with episode-specific entities, times, and metrics) by projecting them into a continuous embedding space, and resolves conflicting signals with an iterative centroid-based reweighting algorithm. The resulting consensus weights ground a singl
    
[^21]: TaRA：训练感知的低秩适应初始化

    TaRA: Training-Aware Low-Rank Adaptation Initialization

    [https://arxiv.org/abs/2609.02639](https://arxiv.org/abs/2609.02639)

    TaRA提出了一种训练感知的LoRA初始化方法，通过使低秩因子诱导的梯度密切逼近全秩权重矩阵的梯度来提升训练初期的梯度保真度，且几乎不增加计算开销。

    

    低秩适应已成为参数高效微调（PEFT）的事实标准，然而由于低秩分解带来的信息瓶颈，其性能对初始化高度敏感。现有方法试图通过利用预训练权重、激活值或梯度的主成分来构建高质量的LoRA初始化，但这些方法并未直接考虑全秩模型的训练动态。本文提出了训练感知低秩适应初始化，该方法在初始化LoRA时，使低秩因子所诱导的梯度能够密切逼近相应全秩权重矩阵的梯度。TaRA源于数学公式推导，在提升训练初期梯度保真度的同时，引入的计算开销几乎可以忽略不计。在多样且具有挑战性的微调任务中，TaRA始终……

    arXiv:2609.02639v1 Announce Type: cross  Abstract: Low-Rank Adaptation (LoRA) has become a de facto standard for parameter-efficient fine-tuning (PEFT), yet its performance is highly sensitive to initialization due to the information bottleneck imposed by low-rank decomposition. Existing approaches attempt to construct high-quality LoRA initializations by exploiting principal components of pretrained weights, activations, or gradients. However, these methods do not directly account for the training dynamics of the full-rank model. In this paper, we propose Training-aware Low-Rank Adaptation Initialization (TaRA), a method that initializes LoRA such that the gradients induced by the low-rank factors closely approximate the gradient of the corresponding full-rank weight matrix. Derived from a mathematical formulation, TaRA improves gradient fidelity at the start of training while introducing negligible computational overhead. Across diverse and challenging fine-tuning tasks, TaRA consist
    
[^22]: 基于语音印象引导伪三元组构建的可扩展方向跟随语音合成

    Scalable Direction-Following TTS via Voice Impression-Guided Pseudo Triplet Construction

    [https://arxiv.org/abs/2609.02623](https://arxiv.org/abs/2609.02623)

    提出一种利用印象可控语音合成模型与大语言模型自动构建（参考语音、方向文本、修改后语音）伪三元组的可扩展流水线，解决了方向跟随语音合成中训练数据稀缺的问题，仅凭伪数据即可实现稳定的说话人特征保留式风格修改。

    

    语音演员常常需要重新朗读同一段剧本，并根据表演指示调整自己的演绎方式。我们将这一场景定义为“方向跟随语音合成”，即系统在保留说话人身份和语言内容的前提下，生成一段相对于参考语音能够体现给定表演指示的新语音。该方法面临的一个关键挑战是缺乏能够捕捉此类相对修改的训练数据。为解决这一问题，我们提出了一种可扩展的伪三元组构建流水线，用于生成（参考语音、方向文本、修改后语音）三元组。该流水线利用印象可控的语音合成模型生成受控的风格变化，并借助大语言模型根据估计的印象差异生成自然语言的方向描述。实验结果表明，仅使用伪三元组即可实现稳定的、保留说话人特征的语音修改；而将伪数据与真实录制数据相结合，还能在保持其他性能的同时进一步提升方向对齐度。

    arXiv:2609.02623v1 Announce Type: cross  Abstract: Voice actors often re-read the same script while modifying their delivery in response to performance directions. We study this setting as direction-following TTS, where a system generates a new utterance that reflects a given direction relative to a reference utterance while preserving speaker identity and linguistic content. A key challenge is the lack of training data capturing such relative modifications. To address this, we propose a scalable pseudo-triplet construction pipeline that generates~(reference utterance, direction text, modified utterance) triplets. It generates controlled style variations using an impression-controllable TTS model and uses an LLM to produce natural language directions from estimated impression differences. Experimental results demonstrate that pseudo-triplets alone enable stable speaker-preserving modification, and that combining pseudo and recorded data further improves direction alignment while mainta
    
[^23]: 利用语音和语言的多模态分析预测老年人孤独感的预测因子

    Predictors of Loneliness in Older Adults Using Multimodal Analysis of Speech and Language

    [https://arxiv.org/abs/2609.02606](https://arxiv.org/abs/2609.02606)

    本研究通过多模态分析310名老年人电话访谈中的语言特征和声学特征，发现高孤独感与更多使用否定词、负面语气及冲突相关语言相关，为自然对话情境下孤独感的客观、可扩展检测提供了新方法。

    

    孤独感是老年人面临的一个重大公共卫生问题，与更高的抑郁风险、认知衰退和死亡率相关。可扩展的、客观的孤独感检测方法仍然有限，尤其是在自然对话情境中。我们通过半结构化电话访谈分析了310名老年人语音和语言中的孤独感标志物，以帮助理解他们如何感知和处理孤独感，以及他们的语言在不同孤独感程度下有何差异。我们的多模态框架将语言特征（心理语言学词典、n-gram和主题模型）与声学特征（音高、音调、响度）相结合，以检验这些特征与自我报告的孤独感评分之间的关联。预定义方法和数据驱动方法均捕捉到了言语内容和声音表达中的模式。研究发现，较高的孤独感与否定词（r = 0.11）、负面语气（r = 0.12）以及冲突相关语言相关联；较低的孤独感则与……

    arXiv:2609.02606v1 Announce Type: new  Abstract: Loneliness is a critical public health issue among older adults, linked to higher risks of depression, cognitive decline, and mortality. Scalable, objective methods for its detection remain limited, particularly in natural conversational contexts. We analyzed speech and language markers of loneliness in 310 older adults using semi-structured telephone interviews to help understand how they process feeling lonely and how their language differs at different levels of feeling loneliness. Our multimodal framework combined linguistic features (psycholinguistic dictionaries, n-grams, and topic models) with acoustic features (pitch, tone, loudness) to examine associations with self-reported loneliness scores. Both predefined and data-driven methods captured patterns in verbal content and vocal delivery. Higher loneliness was associated with negations(r = 0.11), negative tone(r = 0.12), and conflict-related language. Lower loneliness was linked 
    
[^24]: 人格属性何时能改善大语言模型的人群对齐

    When Persona Attributes Improve Population Alignment in Large Language Models

    [https://arxiv.org/abs/2609.02526](https://arxiv.org/abs/2609.02526)

    本文研究人格提示中属性选择对大语言模型人群对齐效果的影响，发现属性的选择比数量更关键，使用更多属性并不必然带来更好的性能。

    

    大语言模型越来越多地被用于预测人类参与者在调查小组中的回答。为实现这一目标，人格提示近期成为一种用于引导和对齐大型预训练语言模型的技术。人格提示是指在提示中使用简短的“人格”文字描述来引导大语言模型的生成。人格通过不同的属性来描述个体，例如其社会人口学特征、态度或行为，目的是对齐大语言模型，使其生成与相应人类回答相关的回答。然而，近期研究在人格提示方面得出了混合且部分相互矛盾的结果，没有明确的成功与失败规律。少数一致的发现之一是人格属性的选择很重要，且使用更多属性并不一定能带来更好的性能。目前仍不清楚不同的属性选择如何影响大语言模型与人群的对齐效果。

    arXiv:2609.02526v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly used to predict the responses of human participants in survey panels. Towards that goal, persona prompting has recently emerged as a technique to inform and align large pretrained language models. Persona prompting refers to the practice of using short textual descriptions of 'personas' in prompts to steer the LLM's generations. Personas describe individuals through different attributes such as their socio-demographics, attitudes, or behaviors, with the aim of aligning LLMs to produce responses that correlate with the corresponding human responses. Yet, recent work has produced mixed and partly conflicting results of persona prompting without clear patterns of success and failure. Among the few consistent findings is that the selection of persona attributes matters, and that using more attributes does not necessarily lead to better performance. It remains unclear how different attribute selec
    
[^25]: Debias-SparseGPT：面向大语言模型的偏见感知剪枝方法

    Debias-SparseGPT: Bias-Aware Pruning for Large Language Models

    [https://arxiv.org/abs/2609.02496](https://arxiv.org/abs/2609.02496)

    提出Debias-SparseGPT，一种在剪枝过程中利用人口统计学对比输入定义的二阶项进行表征去偏的后训练剪枝方法，能在保持模型困惑度和零样本准确率的同时，显著减少剪枝引发的偏见。

    

    诸如剪枝和量化等模型压缩技术有助于大语言模型（LLM）的高效部署与加速。然而，近期研究表明，像SparseGPT这类权重稀疏化方法可能会放大模型中已有的偏见，其输出会因提示中的人物设定线索而产生显著差异。在本文中，我们提出了Debias-SparseGPT，这是一种后训练剪枝方法，通过在人口统计学上相互对比的输入上定义的二阶项来融入表征去偏机制。我们在多种生成式大语言模型上对该方法进行了实证验证。在各种模型和稀疏度设置（25%、50%以及结构化2:4稀疏度）下，与SparseGPT相比，Debias-SparseGPT在保持模型困惑度和零样本准确率的同时，能够持续降低剪枝所引发的偏见。在限制最严格的2:4结构化稀疏模式下——即对模型质量损害最严重的情形下，增……（摘要不完整）

    arXiv:2609.02496v1 Announce Type: new  Abstract: Model compression techniques such as pruning and quantization facilitate the efficient deployment and acceleration of Large Language Models (LLMs). However, recent studies show that weight sparsification methods, such as SparseGPT, can amplify existing biases in models, with outputs varying significantly depending on persona cues in the prompt. In this paper, we introduce Debias-SparseGPT, a post-training pruning method incorporating representational debiasing using a second-order term defined over demographically contrasting inputs. We perform empirical validation of our method over a wide range of generative LLMs. Across models and sparsity regimes (25%, 50%, and structured 2:4 sparsity), Debias-SparseGPT consistently reduces pruning-induced bias compared to SparseGPT while preserving model perplexity and zero-shot accuracy. Under the most restrictive 2:4 structured sparsity pattern, which most aggressively degrades model quality, augm
    
[^26]: ViSAR：面向视觉文档问答的无需训练的自适应k值检索方法

    ViSAR: Training-Free Adaptive-$k$ Retrieval for Visual Document Question Answering

    [https://arxiv.org/abs/2609.02486](https://arxiv.org/abs/2609.02486)

    提出了一种无需训练的自适应k值检索方法ViSAR，通过在嵌入空间中构建查询条件的页面级相似度矩阵来动态确定检索页面数量，在保持或提升答案准确性的同时将RAG延迟降低高达58.7%。

    

    文档视觉问答通常利用检索增强生成技术，其中晚期交互编码器常被用于识别与用户查询相关的文档页面，然后由大型视觉-语言模型生成答案。现有方法通常无论查询复杂度如何都检索固定数量的前k个页面，这会增加大型视觉-语言模型的延迟，并可能降低答案的准确性。我们提出了ViSAR（视觉语义激活检索），这是一种面向晚期交互视觉文档检索的无需训练的自适应k值检索方法。ViSAR直接在嵌入空间中运行，构建以查询为条件的页面级相似度矩阵，突出与查询相关的语义，并动态确定需要检索的页面数量。在多个编码器和大型视觉-语言模型上的实验表明，ViSAR能够检索紧凑且适应查询的页面集合，将RAG延迟降低高达58.7%，同时保持或提升答案准确性。

    arXiv:2609.02486v1 Announce Type: cross  Abstract: Document Visual Question Answering (DocVQA) often leverages Retrieval-Augmented Generation (RAG), where late-interaction encoders are commonly used to identify document pages relevant to a user query, before answer generation by a Large Vision-Language Model (LVLM). Existing approaches typically retrieve a fixed top-$k$ number of pages regardless of query complexity, which increases LVLM latency and may degrade answer accuracy. We introduce ViSAR (Visual Semantic Activation Retrieval), a training-free adaptive-$k$ retrieval method for late-interaction visual document retrieval. ViSAR operates directly in the embedding space to construct a query-conditioned page-level similarity matrix that highlights query-relevant semantics and dynamically determines the number of pages to retrieve. Across multiple encoders and LVLMs, ViSAR retrieves compact, query-adapted page sets that reduce RAG latency by up to 58.7\%, while maintaining or improvi
    
[^27]: 大语言模型如何构建虚构世界：AI生成创意叙事中的场景设定与叙事空间

    How LLMs Build Fictional Worlds: Setting and Narrative Space in AI-Generated Creative Storytelling

    [https://arxiv.org/abs/2609.02482](https://arxiv.org/abs/2609.02482)

    该研究通过微调BERT分类器分析五种叙事空间类型，发现人类小说以角色与环境具身交互的“行动空间”为主，而大语言模型生成的故事则系统性偏向强调氛围情感的“感知空间”，且这一差异在叙事全程保持稳定。

    

    本文分析了大语言模型（LLM）如何运用世界观构建策略，重点关注场景设定作为故事世界构建的一个可测量维度。我们将每个模型生成的1000篇英文和德文AI故事与古腾堡计划中的人类创作小说进行比较。在此前研究的基础上，我们通过五种叙事空间类型对场景设定进行操作化定义：“行动空间”、“感知空间”、“视觉空间”、“描述空间”和“无空间”，并使用针对德语和英语微调的BERT分类器进行识别。我们使用GPT 4.1、LlaMA 3.3、Mistral 3.2和Gemma 3生成叙事文本，并将其空间分布与人类创作的基线进行比较。我们发现，人类创作的文本主要采用“行动空间”，将叙事植根于角色与环境的具身交互中，而大语言模型则系统性地过度生成“感知空间”，强调氛围和情感。这种差异在叙事时间跨度上保持稳定。

    arXiv:2609.02482v1 Announce Type: new  Abstract: In this paper, we analyze how Large Language Models (LLMs) employ worldbuilding strategies, focusing on setting as one measurable dimension of storyworld construction. We compare 1,000 AI-generated stories per model in English and German with human-authored fiction from Project Gutenberg. Building on prior work, we operationalize setting through five types of narrative space: "action", "perceived," "visual," "descriptive" and "no space", identified using fine-tuned BERT classifiers for German and English. We generate narratives using GPT 4.1, LlaMA 3.3, Mistral 3.2, and Gemma 3 and compare their spatial distributions to a human-authored baseline. We find that human-authored texts predominantly employ "action space," grounding narratives in embodied character-environment interaction, whereas LLMs systematically overproduce "perceived space," emphasizing atmosphere and affect. This divergence remains stable across narrative time. Overall, 
    
[^28]: PragAlign：面向受控合成对话生成的反馈引导式语用对齐

    PragAlign: Feedback-Guided Pragmatic Alignment for Controlled Synthetic Dialogue Generation

    [https://arxiv.org/abs/2609.02480](https://arxiv.org/abs/2609.02480)

    PragAlign是一个反馈引导的受控合成对话生成框架，通过“生成—评估—修改”循环利用基于LLM的评估器进行多维度评分与针对性反馈，将评估器接受率提升至99.50%，显著优于一次性生成（72.25%）和无结构化反馈的重复生成（95.88%）。

    

    合成对话生成可以支持隐私受限服务环境下的研究，但生成的对话必须保留交际意图、情感含义和自然的对话流程。我们提出了PragAlign，这是一个反馈引导的受控合成对话生成框架，以服务上下文、目标意图和目标情感为条件，并辅以特质风格的辅助控制。PragAlign采用“生成—评估—修改”循环，其中基于大语言模型（LLM）的评估器对意图对齐、情感对齐、连贯性、流畅性和综合质量进行评分，然后提供针对具体标准的反馈，最多进行三轮优化。在800个匹配的对话规范上，PragAlign达到了99.50%的评估器定义接受率，相比之下，一次性生成仅为72.25%，而没有结构化反馈的重复生成为95.88%。这表明重复尝试贡献了相对于一次性生成的大部分增益，而……

    arXiv:2609.02480v1 Announce Type: new  Abstract: Synthetic dialogue generation can support research in privacy-restricted service settings, but generated conversations must preserve communicative intent, affective meaning, and natural dialogue flow. We introduce PragAlign, a feedback-guided framework for controlled synthetic dialogue generation conditioned on service context, target intent, and target emotion, with auxiliary trait-style controls. PragAlign uses a generate--evaluate--revise loop in which an LLM-based evaluator scores intent alignment, emotion alignment, coherence, fluency, and aggregate quality, then provides criterion-specific feedback for up to three refinement rounds. On 800 matched dialogue specifications, PragAlign achieves 99.50\% evaluator-defined acceptance, compared with 72.25\% for one-shot generation and 95.88\% for repeated generation without structured feedback. This indicates that repeated attempts account for much of the gain over one-shot generation, whi
    
[^29]: 学习将大语言模型与本体排序器融合用于罕见病诊断

    Learning to Fuse LLMs with Ontology Rankers for Rare-Disease Diagnosis

    [https://arxiv.org/abs/2609.02473](https://arxiv.org/abs/2609.02473)

    该论文提出一种基于行为的融合模型，将大语言模型与本体排序器结合用于罕见病诊断，在保留证据可追溯性的同时，将 Phenomizer 的 Recall@1 分别提升 7.86 和 20.18 个百分点。

    

    本体排序器在罕见病诊断中依然有用，因为每个候选诊断都可以追溯到相匹配的患者表型。大语言模型（LLM）能够根据相同的患者描述生成鉴别诊断，但其预测缺乏同样清晰的证据链。我们不去追问哪个系统应该取代另一个，而是探讨大语言模型能否在不放弃证据可追溯性的前提下改进排序器。我们提出的基于行为的融合模型会审视两份排序列表、二者之间的一致性，以及每个候选诊断背后的本体支持，并学习在具体病例中应多大程度地依赖各个系统。在比较之前，我们移除了一条已被记录在案的测试集泄露途径，该泄露源于基准病例与本体注释出自同一批文献。在八个开源大语言模型上，融合方法在 Phenomizer 的 Recall@1 指标上，于 Phenopacket Store 数据集提升了 7.86 个百分点，在 RAMEDIS 数据集提升了 20.18 个百分点。当与 DeepSeek-V4-Flash 配对时……（摘要至此截断）

    arXiv:2609.02473v1 Announce Type: new  Abstract: Ontology rankers remain useful for rare-disease diagnosis because each candidate can be traced to matched patient phenotypes. Large language models (LLMs) can generate differential diagnoses from the same patient description, but their predictions lack an equally clear evidence trail. Rather than asking which system should replace the other, we ask whether an LLM can improve the ranker without giving up its evidence. Our behavior-based fusion model examines the two ranked lists, their agreement, and the ontology support behind each candidate, and learns how much to rely on each system for the individual case. Before comparison, we remove a documented test-set leakage pathway caused by benchmark cases and ontology annotations being derived from the same publications. Across eight open LLMs, fusion improves Phenomizer Recall@1 by 7.86 percentage points on Phenopacket Store and 20.18 points on RAMEDIS. When paired with DeepSeek-V4-Flash thr
    
[^30]: 可扩展的Kronecker-Fisher近似：面向十亿参数语言模型压缩的高效Hessian分析

    Scalable Kronecker-Fisher Approximation: Efficient Hessian Analysis for Billion-Parameter Language Models Compression

    [https://arxiv.org/abs/2609.02451](https://arxiv.org/abs/2609.02451)

    本文提出一种可扩展的Kronecker-Fisher近似方法，无需存储完整Fisher矩阵即可对十亿参数语言模型进行高效Hessian分析，发现值投影层是最脆弱的组件，为混合精度分配等压缩与优化策略提供了实用的理论工具。

    

    在本文中，我们提出了一种可扩展的基于Kronecker的近似方法，该方法无需存储整个Fisher矩阵即可捕捉跨层交互，使得对十亿参数规模的神经网络进行实用的Hessian分析成为可能，而此类网络的完整计算是不可行的。我们的方法揭示了一致的脆弱性模式：在多个模型家族中，值投影层表现出最高的敏感性和最强的跨层相关性，而其他组件则表现出架构特定的行为。通过在量化、稀疏化、层间破坏以及破坏后微调方面的大量实验，我们证明了我们的近似与性能下降和性能恢复均具有很强的相关性。我们的框架为识别大模型中的脆弱组件提供了一个实用的、有理论依据的工具，为有引导的压缩与优化策略（如混合精度分配）开辟了新的途径。

    arXiv:2609.02451v1 Announce Type: cross  Abstract: In this paper, we propose a scalable Kronecker-based approximation that captures cross-layer interactions without storing the entire Fisher matrix, enabling practical Hessian analysis for billion-parameter networks where full computation is infeasible. Our approach reveals consistent vulnerability patterns: value projection layers exhibit the highest sensitivity and strongest cross-layer correlations across multiple model families, while other components exhibit architecture-specific behaviors. Through extensive experiments on quantization, sparsification, inter-layer corruption, and post-corruption fine-tuning, we demonstrate that our approximation strongly correlates with both performance degradation and recovery. Our framework provides a practical, theoretically grounded tool for identifying fragile components in large models, opening new avenues for guided compression and optimization strategies, such as mixed-precision allocation,
    
[^31]: 当可解码性并不足够时：语言模型中的逻辑有效性表征、行为解离与因果检验

    When Decodability Is Not Enough: Logical Validity Representations, Behavioral Dissociation, and Causal Tests in Language Models

    [https://arxiv.org/abs/2609.02438](https://arxiv.org/abs/2609.02438)

    该研究发现即使大语言模型在逻辑验证任务上的行为表现接近随机，其隐藏状态中仍能近乎完美地解码出逻辑有效性信息，但因果干预显示这种表征并未被模型实际利用，揭示了“可解码性不等于因果性使用”这一重要结论。

    

    大型语言模型看起来可能具备逻辑推理能力，但仅凭答案的对错并不能告诉我们模型内部究竟表征了什么。我们使用匹配的有效-无效前提-论断对，在五个开源权重Transformer模型中研究逻辑验证任务，这些前提-论断对覆盖不同的推理家族、语义领域、模板和难度级别。尽管行为表现接近随机水平，逻辑有效性往往可以从隐藏状态中被几乎完美地解码出来，并且在留出的模板、领域和推理家族上仍保持很强的可解码性。在能够正确定义以正确性为条件的评估的情形下，有效性在行为错误的样本上也保持高度可解码。与此同时，详尽的留一法测试揭示了这种泛化能力的明显局限，而与随机对照相比，沿探针导出的有效性方向进行的因果干预仅有微弱且非特异性的效果。

    arXiv:2609.02438v1 Announce Type: new  Abstract: Large language models can look capable of logical reasoning, but correct or incorrect answers alone tell us little about what the model represents internally. We study logical verification in five open-weight transformer models using matched valid--invalid premise--claim pairs that vary across inference families, semantic domains, templates, and difficulty levels. Despite near-chance behavioral performance, logical validity is often almost perfectly decodable from hidden states and remains strongly decodable under held-out templates, domains, and inference families. Validity also remains highly decodable on behaviorally incorrect examples in the conditions where correctness-conditioned evaluation is well defined. At the same time, exhaustive leave-one-out tests reveal clear limits to this generalization, and interventions along probe-derived validity directions have only weak, nonspecific effects compared with random controls. Our result
    
[^32]: UTP-Bench：不确定性感知的旅行规划基准

    UTP-Bench: Uncertainty-aware Travel Planning Benchmark

    [https://arxiv.org/abs/2609.02421](https://arxiv.org/abs/2609.02421)

    该论文提出了 UTP-Bench——首个引入真实交通延误分布与人流密度等不确定性因素的大规模旅行规划基准，覆盖印度 504 个城市，用以评估大语言模型生成的行程在现实干扰下的稳健性。

    

    大型语言模型（LLMs）近来在自动化旅行行程生成方面展现出了强大的能力。然而，现实世界的旅行规划本质上充满不确定性：交通延误、人流量波动以及意外的随机延误经常会使原本可行的行程安排失效。现有的基准测试（如 TravelPlanner 和 TripCraft）都假设环境是确定性的，仅评估静态的约束满足情况，而忽略了生成的计划在面临此类不确定性时是否依然稳健。为了解决这一局限性，我们推出了 UTP-Bench，一个面向不确定性感知旅行规划的大规模基准。该数据集整合了覆盖印度 504 个城市的真实旅行数据，包括景点、餐厅、住宿以及多模式交通网络。为了模拟现实中的干扰情况，UTP-Bench 引入了从主要交通网络和景点采集的真实延误分布和人流密度模式。

    arXiv:2609.02421v1 Announce Type: new  Abstract: Large Language Models (LLMs) have recently demonstrated strong capabilities in automated travel itinerary generation. However, real- world travel planning is inherently uncertain: transportation delays, crowd fluctuations, and unexpected stochastic delays frequently inval- idate otherwise feasible schedules. Existing benchmarks like TravelPlanner and TripCraft assume deterministic environments, evaluating only static constraint satisfaction and ignoring whether generated plans remain robust when such uncertainties arise. To address this limitation, we introduce UTP-Bench1 , a large-scale benchmark for uncertainty-aware travel planning. The dataset integrates real-world travel data spanning 504 cities of India, including attractions, restau- rants, accommodations, and multi-modal trans- portation networks. To model realistic disrup- tions, UTP-Bench incorporates empirical delay distributions and crowd-density patterns col- lected from maj
    
[^33]: 在剧本之前，先搭建舞台：世界观模拟如何在多轮越狱攻击中放大基于心理学的说服效果

    Before the Script, Set the Stage: How Worldview Simulation Amplifies Psychologically Grounded Persuasion in Multi-Turn Jailbreaking

    [https://arxiv.org/abs/2609.02414](https://arxiv.org/abs/2609.02414)

    BLUEPRINT框架通过结合18个心理学影响因素的世界观模拟与蒙特卡洛树搜索，以最少查询次数在六个前沿大模型上实现接近100%的多轮越狱攻击成功率，并揭示了各模型共享的"转向具体可执行任务"这一越狱路径。

    

    arXiv:2609.02414v1 公告类型：交叉 摘要：多轮越狱攻击表明，有害意图可以分散在对话中，但现有方法模糊了究竟是对话机制导致了模型的脆弱性。我们提出了BLUEPRINT，一个安全评估框架，它将因子化的社会影响策略空间与WORLDVIEWSIM（一个跨轮次情境上下文模块）分离开来。蒙特卡洛树搜索在四轮对话轨迹中优化18个基于理论的影响因素的轮级组合。在六个前沿模型上的实验表明，BLUEPRINT在主要开源权重模型和专有模型上实现了接近上限的攻击成功率（ASR），同时所需平均查询次数最少（2.46次）。由此产生的轨迹进一步揭示了抗性目标中模型特异性的脆弱性：每个模型对不同的影响因素和策略转换做出反应，但它们都共享一条共同的恢复路径——转向具体、可执行的任务框架总能从坚决拒绝的状态中逃脱。消融实验表

    arXiv:2609.02414v1 Announce Type: cross  Abstract: Multi-turn jailbreak attacks demonstrate that harmful intent can be distributed across dialogue, yet existing methods obscure what conversational mechanisms drive vulnerability. We introduce BLUEPRINT, a safety-evaluation framework separating a factorized social-influence strategy space from WORLDVIEWSIM, a cross-turn situational context module. Monte Carlo Tree Search optimizes turn-level combinations of 18 theory-grounded influence factors across a four-turn trajectory. Across six frontier models, BLUEPRINT achieves near-ceiling ASR on major open-weight and proprietary models, while requiring the fewest average queries (2.46). The resulting trajectories further reveal model-specific vulnerability among resistant targets: each responds to distinct influence factors and strategy transitions, yet all share a common recovery pathway-shifting toward concrete, executable task framing consistently escapes hard-refusal states. Ablations conf
    
[^34]: 通过放射报告的通俗化摘要提升健康素养：生物医学命名实体识别（BioNER）与检索增强生成（RAG）的评估

    Improving Health Literacy through Lay Summarization of Radiological Reports: An Evaluation of BioNER and Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.02396](https://arxiv.org/abs/2609.02396)

    本研究提出了一个将NER临床发现提取与RAG上下文锚定相结合的框架，证明这两种技术能够显著提升放射报告自动通俗摘要的质量、事实一致性和可读性，从而改善患者的健康素养。

    

    放射学报告主要是为临床医生撰写的，其专业术语常常使患者难以理解。因此，许多患者转而求助于公开可用的大语言模型（LLM）来帮助解释他们的报告，尽管此类模型在事实不准确和幻觉方面的风险已有充分记录。自动化的通俗摘要生成已成为一种有前景的替代方案，然而，检索增强和临床信息驱动的方法在放射学特定沟通中的有效性仍未得到充分探索。本研究探讨了检索增强生成（RAG）和命名实体识别（NER）在多大程度上能够提升自动生成的通俗摘要的质量、事实一致性和可读性，并与标准的基于LLM的生成方法进行比较。我们开发了一个框架，将基于NER的临床相关发现提取与用于上下文锚定的RAG机制相结合，并在……（原文摘要在此处截断）

    arXiv:2609.02396v1 Announce Type: new  Abstract: Radiology reports are written primarily for clinicians, and their specialized terminology often makes them difficult for patients to interpret. As a result, many patients turn to publicly available Large Language Models (LLMs) to help explain their reports, despite well-documented risks of factual inaccuracies and hallucinations. Automated lay-summary generation has emerged as a promising alternative, yet the effectiveness of retrieval-enhanced and clinically informed approaches for radiology-specific communication remains underexplored. This study investigates the extent to which Retrieval-Augmented Generation (RAG) and Named Entity Recognition (NER) improve the quality, factual consistency, and readability of automatically generated lay summaries compared with standard LLM-based generation. We develop a framework combining NER-based extraction of clinically relevant findings with a RAG mechanism for contextual grounding, evaluated acro
    
[^35]: PolERo：研究罗马尼亚语中的政治回避现象

    PolERo: Studying Political Evasion in Romanian

    [https://arxiv.org/abs/2609.02391](https://arxiv.org/abs/2609.02391)

    该论文提出首个罗马尼亚语政治回避检测数据集PolERo（包含来自五位罗马尼亚总统官方记录的3,574个人工标注问答对），并将分类体系与模型从英语扩展到新语言，同时通过双语联合训练和翻译增强研究了跨语言迁移能力。

    

    政治回避指那些对问题有所回应但隐瞒所请求信息的回答。近期的自然语言处理工作将政治回避框架化为一个分类任务，采用了响应清晰度与细粒度回避策略的两级分类体系。然而，现有的响应清晰度和回避分类研究仅限于英语，这些分类体系和模型行为能否跨语言、跨政治环境迁移仍是一个悬而未决的问题。我们提出了PolERo，一个包含3,574个人工标注问答对的数据集，这些问答对提取自五位罗马尼亚总统的官方发言记录。我们在相同的实验条件下，在两个数据集上评估了多种分类方法，包括TF-IDF基线、微调的编码器模型、我们提出的滑动窗口编码器，以及零样本/少样本大语言模型提示。我们还通过联合双语训练和基于机器翻译的数据增强来研究跨语言迁移。我们的结果表明，微调……（原文摘要在此截断）

    arXiv:2609.02391v1 Announce Type: cross  Abstract: Political evasion refers to responses that engage with a question while withholding the requested information. Recent NLP work frames political evasion as a classification task using a two-level taxonomy of response clarity and fine-grained evasion strategies. Existing work on response clarity and evasion classification is limited to English, leaving open whether the taxonomy and model behavior transfer across languages and political contexts. We introduce PolERo, a dataset of 3,574 human-annotated question-answer pairs extracted from official transcripts of five Romanian presidents. We evaluate multiple classification approaches on both datasets under matched conditions, including TF-IDF baselines, fine-tuned encoder models, a proposed sliding-window encoder, and zero/few-shot LLM prompting. We study cross-lingual transfer through joint bilingual training and machine-translation-based data augmentation. Our results indicate that fine-
    
[^36]: MultiGhostBench：面向分布偏移下长篇LLM生成文本归因的多语言基准

    MultiGhostBench: A Multilingual Benchmark for Long-Form LLM-Generated Text Attribution under Distribution Shifts

    [https://arxiv.org/abs/2609.02379](https://arxiv.org/abs/2609.02379)

    本文提出了MultiGhostBench多语言基准，包含五个最新大语言模型在六种语言下生成的928本约59K词的长篇书籍，用于评估领域、作者和语言偏移下的LLM文本归因，发现没有单一方法始终最优且分布偏移会导致性能下降。

    

    尽管现有的LLM作者归因研究已经取得了一定进展，但可用的基准仍然有限，通常只关注英语、受控环境或相对过时的模型，而少数多语言研究也仅考虑了相对较短的文本。我们提出了MultiGhostBench，这是一个多语言基准，包含由五个最新大语言模型生成的928本书，涵盖六种语言和三种文字系统，每本书的平均长度约为59,000词。该基准支持在领域、作者和语言偏移下的评估。对代表性作者归因方法的评估表明，没有单一方法能在所有设置下始终表现最佳，且在分布偏移下性能普遍下降。基于Transformer的检测器能够跨语言保留与生成器相关的信息，尽管迁移效果因语言对而异，而基于统计和指纹的检测器则更加依赖于具体语言。

    arXiv:2609.02379v1 Announce Type: cross  Abstract: While existing work on LLM authorship attribution (AA) has made progress, available benchmarks remain limited, often focusing on English, controlled settings, or relatively outdated models, with the few multilingual studies considering only relatively short texts. We introduce MultiGhostBench, a multilingual benchmark comprising 928 books generated by five recent LLMs across six languages and three scripts, with an average length of approximately 59K words per book. The benchmark supports evaluation under domain, author, and language shifts. Evaluation of representative AA methods shows that no single method consistently performs best across settings, and performance generally degrades under distribution shifts. Transformer-based detectors can retain generator-related information across languages, although transfer effectiveness varies by language pair, whereas statistical and fingerprint-based detectors are more language-dependent. We
    
[^37]: NE-R1：通过强化学习增强命名实体识别模型

    NE-R1: Enhancing Named Entity Recognition Model via Reinforcement Learning

    [https://arxiv.org/abs/2609.02366](https://arxiv.org/abs/2609.02366)

    提出了NE-R1框架，通过“按需检索”机制和两阶段训练（多任务指令微调初始化与基于CoT的端到端强化学习优化），借助兼顾准确性与检索收益的多维奖励，在参数化知识与外部知识之间进行自适应选择，实现检索增强命名实体识别的最先进性能。

    

    自大语言模型（LLMs）出现以来，命名实体识别（NER）已取得长足进展。然而，由于参数化知识的不足，长尾实体和领域特定实体的识别仍然充满挑战。检索增强生成（RAG）通过注入外部知识提供了一种有前景的解决方案，但在处理熟悉的情况时也会引入噪声和不必要的成本。在本文中，我们提出了NE-R1，这是一种用于自适应检索增强NER的新型框架。我们为NER设计了一种“按需检索”机制，然后通过两阶段训练方法将其集成到模型中：（1）多任务指令微调初始化；（2）结合思维链（CoT）的端到端强化学习优化。为了在参数化知识和外部知识之间实现合理选择，我们设计了一个同时考虑准确性和检索收益的多维奖励。NE-R1在各种数据集上实现了最先进的性能。

    arXiv:2609.02366v1 Announce Type: cross  Abstract: Named Entity Recognition (NER) has achieved substantial progress since the advent of large language models (LLMs). Nevertheless, the recognition of long-tail and domain-specific entities remains challenging due to the deficiency in parametric knowledge. Retrieval-augmented generation (RAG) offers a promising remedy by injecting external knowledge, but it also introduces noise and unnecessary cost when dealing with familiar cases. In this paper, we propose NE-R1, a novel framework for adaptive retrieval-augmented NER. We design a "retrieval-on-demand" mechanism for NER. Then we integrate it into models by a two-stage training method: (1) multi-task instruction tuning initialization; (2) end-to-end RL optimization with CoT. To achieve reasonable selection between parameterized and external knowledge, we design a multi-dimensional reward considering both accuracy and retrieval benefit. NE-R1 achieves state-of-the-art performance on variou
    
[^38]: SonicCaps：用于改进音频检索的大规模多样化细粒度字幕数据集

    SonicCaps: Large-Scale Diverse and Fine-Grained Captioning for Improved Audio-Retrieval

    [https://arxiv.org/abs/2609.02343](https://arxiv.org/abs/2609.02343)

    提出SonicCaps大规模音频字幕数据集，包含约1500万条字幕配对70万个音频片段，利用多模态大语言模型为每个音频生成约24条多样化、细粒度的字幕，有效克服了现有数据集语义多样性低和一对一映射的局限，显著提升音频检索性能。

    

    音频-语言建模的最新进展一直由大规模音频字幕数据集驱动。然而，现有数据集仍然受限于语义多样性低、缺乏声学细节的笼统描述，以及难以反映听觉感知固有模糊性的一对一音频-字幕映射。我们推出了SonicCaps，这是一个大规模音频字幕数据集，包含约1500万条字幕，与约70万个音频片段配对，由多模态大语言模型（Qwen3-Omni）以音频和文本为共同条件生成。为了明确地促进多样性，我们通过结构化提示工程和少样本生成，为每个音频生成约24条字幕，涵盖主要描述、改写变体（详细程度、风格）和语义标签。人类评估表明，SonicCaps的评分显著高于现有字幕数据集，细粒度分析显示我们的字幕被认为更具描述性（注：原文摘要在此处截断）。

    arXiv:2609.02343v1 Announce Type: cross  Abstract: Recent advances in audio-language modeling have been driven by large-scale audio captioning datasets. However, existing datasets remain limited by low semantic diversity, generic descriptions lacking acoustic details, and one-to-one audio-caption mappings that poorly reflect the inherent ambiguity of auditory perception. We introduce SonicCaps, a large-scale audio captioning dataset comprising ~15M captions paired with ~700k audio clips, generated using a multi-modal large language model (Qwen3-Omni) conditioned on both audio and text. To explicitly promote diversity, we generate around 24 captions per audio via structured prompt engineering and few- shot generation, spanning main descriptions, rephrased variants (verbosity, style) and semantic tags. Human evaluation shows that SonicCaps is rated significantly higher than existing captioning datasets, with fine-grained analyses indicating that our captions are perceived as more descrip
    
[^39]: SALA：面向上下文学习复杂推理的语义感知逻辑对齐

    SALA: Semantic-Aware Logical Alignment for Complex Reasoning in In-Context Learning

    [https://arxiv.org/abs/2609.02336](https://arxiv.org/abs/2609.02336)

    SALA框架通过自动学习任务特定的推理操作，并在连续语义空间中利用动态时间规整（DTW）实现推理序列的软性对齐，从而为复杂推理的上下文学习提供灵活且可解释的示例选择方法。

    

    有效的上下文学习（ICL）在复杂推理任务中依赖于选择合适的示例。传统的基于表面相似性的检索方法无法捕捉底层的解题逻辑。近期基于逻辑的方法通过匹配预定义的推理步骤来解决这一问题，但僵化的规则和精确匹配标准难以处理灵活多样的推理过程。为了解决这一问题，我们提出了SALA——一个语义感知逻辑对齐框架。SALA不再依赖固定的推理步骤清单，而是自动学习任务特定的推理操作。随后，它将这些操作嵌入到连续的语义空间中，并利用动态时间规整（DTW）来对齐推理序列。这种方法能够对推理逻辑进行软性、灵活的匹配，同时保持高度的可解释性。在四个推理基准和三个大语言模型上的实验表明，SALA优于现有的示例选择方法。

    arXiv:2609.02336v1 Announce Type: new  Abstract: Effective in-context learning (ICL) for complex reasoning relies on selecting the right demonstrations. Traditional retrieval methods based on surface similarity fail to capture the underlying problem-solving logic. Recent logic-based methods address this by matching predefined reasoning steps, but the rigid rules and exact-match criteria is improper to handle flexible or diverse reasoning processes. To address the problem, we propose SALA, a Semantic-Aware Logical Alignment framework. Instead of relying on a fixed inventory, SALA automatically learns task-specific reasoning operations. It then embeds these operations into a continuous semantic space and uses dynamic time warping (DTW) to align the reasoning sequences. This approach allows for soft, flexible matching of reasoning logic while remaining highly interpretable. Experiments across four reasoning benchmarks and three LLMs demonstrate that SALA outperforms existing demonstration
    
[^40]: Counter-GEO-Bench：评估针对信息扭曲型生成式引擎优化的防御方法

    Counter-GEO-Bench: Evaluating Defenses Against Information-Distorting Generative Engine Optimization

    [https://arxiv.org/abs/2609.02316](https://arxiv.org/abs/2609.02316)

    提出了首个针对生成式引擎优化（GEO）攻击的防御基准Counter-GEO-Bench，通过将247个经人工验证的查询与信息保留型和信息扭曲型GEO改写配对来评估防御方法，并揭示现有三种主流防御方法最多仅能将攻击成功率相对降低5.7%。

    

    生成式引擎优化（GEO）使内容生产者能够提高其网页在生成式搜索引擎中的可见度，但同样的技术也可能被用于传递针对性错误信息——当攻击者发布看似普通的GEO优化文档，这些文档被受害大语言模型（LLM）检索并合成为扭曲的答案时。目前尚无现有基准在受控条件下评估针对这一威胁的防御方法。因此，我们提出了Counter-GEO-Bench，这是一个防御基准，将247个经人工验证、质量把关的查询与信息保留型和信息扭曲型的GEO改写版本配对，并在三个受害LLM上从攻击成功率（ASR）、误报率和答案质量三个维度评估防御方法。在Counter-GEO-Bench上，三种现成的防御方法（Granite Guardian、Llama Guard 3和NeMo Self-Check Fact-Checking）最多仅能将攻击成功率相对降低5.7%，而Granite Guardian的降低效果并不显著……

    arXiv:2609.02316v1 Announce Type: cross  Abstract: Generative engine optimization (GEO) enables content producers to increase the visibility of their web pages in generative search engines, but the same techniques can deliver targeted misinformation when adversaries publish ordinary-looking GEO-optimized documents that victim large language models (LLMs) retrieve and synthesize into distorted answers. No existing benchmark evaluates defenses against this threat under controlled conditions. Therefore, we present Counter-GEO-Bench, a defense benchmark that pairs 247 human-verified, quality-gated queries with information-preserving and information-distorting GEO rewrites, and evaluates defenses on attack success rate (ASR), false positive rate, and answer quality across three victim LLMs. Under Counter-GEO-Bench, three off-the-shelf defenses (Granite Guardian, Llama Guard 3, and NeMo Self-Check Fact-Checking) reduce ASR by at most 5.7% relative, while Granite Guardian's reduction is not s
    
[^41]: DiffIE：基于扩散模型的开源信息抽取

    DiffIE: Diffusion-based Open Information Extraction

    [https://arxiv.org/abs/2609.02315](https://arxiv.org/abs/2609.02315)

    DIFFIE将条件离散扩散的随机性本身作为抽取机制，通过多条独立的反向扩散轨迹生成候选三元组池，实现抽取预算与训练的解耦，并在CaRB基准上取得新的最优性能。

    

    单个句子通常表达多个有效的关系三元组，这使得开放信息抽取（OpenIE）从根本上是一个多输出任务。现有的神经 系统 通过自回归生成来处理这一问题，这种方式灵活但速度慢且容易产生冗余；或者通过固定槽位预测来处理，这种方式高效但将抽取预算与训练过程耦合在一起。我们提出DIFFIE，它将条件离散扩散的随机性本身作为抽取机制：基于逐词元角色标签的独立反向扩散轨迹生成一个候选三元组池，然后在宽松匹配标准下进行聚类并排序以形成最终输出。候选池大小和返回抽取结果的数量都是推理阶段的选择，从而将抽取预算与训练解耦，并将测试时计算量暴露为一个可调节的维度。DIFFIE在CaRB上实现了新的最先进水平，在F1和AUC（1-1）上均取得最佳成绩，并优于...

    arXiv:2609.02315v1 Announce Type: cross  Abstract: A single sentence often expresses multiple valid relational triplets, which makes Open Information Extraction (OpenIE) fundamentally a multi-output task. Existing neural systems handle this by autoregressive generation, which is flexible but slow and prone to redundancy, or by fixed-slot prediction, which is efficient but couples the extraction budget to training. We introduce DIFFIE which instead treats the stochasticity of conditional discrete diffusion as the extraction mechanism itself: independent reverse-diffusion trajectories over per-token role tags produce a pool of candidate triplets, which are clustered under lenient matching and ranked to form the output. Both the pool size and the number of returned extractions are inference-time choices, decoupling the extraction budget from training and exposing test-time compute as a tunable axis. DIFFIE achieves the new state of the art in CaRB (1-1) both F1 and AUC, and outperforms th
    
[^42]: 高效GUI智能体：关于观察、记忆、动作与运行时优化的系统性综述

    Efficient GUI Agents: A Systems Survey of Observation, Memory, Action, and Runtime Optimization

    [https://arxiv.org/abs/2609.02309](https://arxiv.org/abs/2609.02309)

    本文是首篇从端到端系统效率视角综述GUI智能体的工作，从观察、记忆、动作与运行时优化四个维度系统梳理了高效GUI智能体的主流机制与新兴开销。

    

    GUI智能体日益广泛地运行于网站、移动应用和桌面环境中，然而该领域仍主要通过任务成功率来汇报研究进展。我们认为，实际部署同样取决于效率：即智能体在成功完成任务的同时消耗了多少上下文、计算资源、动作预算和运行时开销。本综述从端到端的系统视角研究高效GUI智能体，涵盖当前的技术主线：观察效率、上下文与记忆效率、动作效率以及规划器侧/系统效率。对于每个子方向，我们通过定向检索并结合向后与向前引用链来扩展种子文献，进而综合其主导机制、所报告的效率指标以及它们引入的新开销。纵观相关文献，近期进展汇聚于少数几个反复出现的核心思想：以选择性读取取代全上下文摄入、从全局到局部的视觉资源分配……

    arXiv:2609.02309v1 Announce Type: new  Abstract: GUI agents increasingly operate across websites, mobile apps, and desktop environments, yet the field still reports progress primarily through task success. We argue that practical deployment depends equally on efficiency: how much context, computation, action budget, and runtime overhead an agent consumes while succeeding. This survey studies efficient GUI agents through an end-to-end systems lens that preserves the current technical axes of observation efficiency, context and memory efficiency, action efficiency, and planner-side/system efficiency. For each subsection, we expand the seed literature through targeted search plus backward and forward citation chaining, then synthesize the dominant mechanisms, reported efficiency signals, and new overheads they introduce. Across the literature, recent progress converges on a small set of recurring ideas: selective reading instead of full-context ingestion, global-to-local visual allocation
    
[^43]: 通过推理时计算与部署支架提升评估的真实性

    Improving Evaluation Realism with Inference-Time Compute and Deployment Scaffolds

    [https://arxiv.org/abs/2609.02302](https://arxiv.org/abs/2609.02302)

    该论文提出“批判式精炼”和 DISH 智能体框架两种技术，通过投入额外推理时计算并模仿真实部署环境，使模拟对齐评估更难被能力强模型识别为测试，从而提升安全评估的真实性与结论可靠性。

    

    对齐评估面临的一个核心障碍是“评估意识”：能力强大的模型能够分辨出自己是在被测试而非被部署，这削弱了安全评估所能支持的结论。我们提出了两种技术，使模拟的对齐评估更难与真实部署区分开来。我们的第一种技术是“批判式精炼”，它在模拟器的每个动作上投入额外的推理时计算：模拟器生成多个候选动作，利用目标模型实例提供的关于如何使其更真实的反馈对候选动作进行精炼，然后以最接近真实部署的候选动作继续评估。我们的第二种技术是 DISH（模仿部署的 SWE-Agent 框架），它将目标模型封装在一个智能体框架中，缩小了编码场景下模拟环境与真实部署环境之间的差距。我们在多个目标模型上测试了这些技术，发现它们可以叠加组合：同时应用两者能带来更大的真实性提升。

    arXiv:2609.02302v1 Announce Type: new  Abstract: A core obstacle to alignment evaluation is evaluation awareness: capable models can tell when they are being tested rather than deployed, weakening the conclusions a safety evaluation can support. We present two techniques that make simulated alignment evaluations harder to distinguish from real deployments. Our first technique, critique refinement, spends additional inference-time compute on each simulator action: the simulator generates multiple candidate actions, refines them using feedback from an instance of the target model on how to make them more realistic, and continues the evaluation with the most deployment-like candidate. Our second technique, DISH (Deployment-Imitating SWE-Agent Harness), wraps the target in an agent harness, reducing the gap between simulated and real deployment environments in coding settings. We test the techniques on multiple target models and find that they compose: applying both yields larger realism g
    
[^44]: SCX Router：基于解码器KV分类器与真实世界任务本体的流式零样本模型选择

    SCX Router: Streaming Zero-Shot Model Selection with a Decoder-KV Classifier and a Real-World Task Ontology

    [https://arxiv.org/abs/2609.02292](https://arxiv.org/abs/2609.02292)

    SCX Router是一个轻量级零样本模型路由器，通过解码器-KV缓存执行路径实现流式模型选择，无需自回归生成即可为各候选LLM分配适配度评分，从而在真实任务层面实现速度、成本与质量的优化权衡。

    

    大型语言模型（LLM）的快速普及及其应用的日益多样化带来了一项独特的优化机会：为每个任务选择合适的模型，同时在任务层面优化速度、成本和质量。然而，推理端点在质量、价格、延迟、上下文支持、工具使用、领域专长和推理行为等方面差异巨大。这种异质性使得人工启发式规则难以维护，并且难以仅靠其自身在速度—成本—质量的权衡上持续取得理想结果。我们提出了SCX Router，一个基于GLiClass的轻量级路由器，无需自回归生成即可为每个推理时模型标签分配适配度评分。发布的0.6B参数检查点将Qwen3解码器与一个浅层双向评分器相结合。其解码器-KV执行路径在整个会话期间保留纯文本的键值缓存，仅对新的对话轮次进行编码，并对……进行评估（摘要在此处截断）

    arXiv:2609.02292v1 Announce Type: new  Abstract: The rapid proliferation of large language models (LLMs) and the growing diversity of their applications presents a unique optimization opportunity: selecting the right model for the task, while optimizing for speed, cost, and quality at a per-task level. However, inference endpoints can vary widely in quality, price, latency, context support, tool use, domain expertise, and reasoning behavior. This heterogeneity makes manual heuristics difficult to maintain and unlikely to achieve consistently favorable speed--cost--quality trade-offs on their own. We introduce \router{}, a lightweight GLiClass-based router that assigns a suitability score to each inference-time model label without autoregressive generation. The released 0.6B-parameter checkpoint combines a Qwen3 decoder with a shallow bidirectional scorer. Its decoder-KV execution path preserves a text-only key--value cache across a session, encodes only new dialogue turns, and evaluate
    
[^45]: 纠缠表示加剧机器遗忘中的附带损害

    Entangled Representations Amplify Collateral Damage in Unlearning

    [https://arxiv.org/abs/2609.02285](https://arxiv.org/abs/2609.02285)

    该研究首次通过受控实验验证了表示纠缠会加剧机器遗忘的附带损害——通过训练知识域解耦程度不同的语言模型套件，证明更解耦的模型在固定遗忘水平下保留成本可降低约4倍。

    

    可解释性研究中一个长期存在的直觉是，表示纠缠——即神经网络中不同知识域之间共享结构——会使机器遗忘变得更加困难。尽管这一直觉广为流传，但此前从未在受控实验中得到直接验证。我们提出了一种实现验证的方法：通过改造选择性梯度掩蔽（SGTM），我们在英文维基百科语料上训练了六个254M参数的语言模型套件，这些模型在生物学与非生物学知识之间具有不同等级的解耦程度。将该套件中的每个模型分别应用三种标准遗忘方法后，我们发现解耦程度更高的模型始终能实现更好的“保留-遗忘”权衡：在固定遗忘水平下，最解耦的模型在三种方法中的两种下保留成本约降低4倍，在第三种方法下降低1.3倍。由于我们的干预仅改变了模型本身，而不改变数据或遗忘算法，因……

    arXiv:2609.02285v1 Announce Type: cross  Abstract: A long-held intuition in interpretability research is that representational entanglement, the sharing of structure between knowledge domains in a neural network, makes unlearning harder. While the intuition is widespread, it has never been directly tested in a controlled experiment. We present a way to do so: by repurposing Selective Gradient Masking (SGTM), we train a suite of six 254M-parameter language models on English Wikipedia with graded levels of disentanglement between biology and non-biology knowledge. Applying three standard unlearning methods to every model in the suite, we find that more disentangled models consistently achieve better retain-forget trade-offs: at a fixed level of forgetting, the most disentangled models incur roughly $4\times$ lower retain cost under two of the three methods, and $1.3\times$ lower under the third. Because our intervention changes only the model, not the data or the unlearning algorithm, th
    
[^46]: 大型语言模型能否捕捉其训练数据中的多样性？

    Do Large Language Models Capture the Diversity in their Training Data?

    [https://arxiv.org/abs/2609.02275](https://arxiv.org/abs/2609.02275)

    该论文提出一种基于信息论的方法，通过比较模型生成输出与训练数据的条件熵，发现大语言模型（如OLMo、Pythia和GPT-Neo）生成内容的多样性系统性地低于其训练数据的多样性。

    

    大型语言模型被训练用于建模文本的条件分布，但它们是否能够捕捉训练数据中存在的合理输出的全部多样性，这一问题仍未得到充分理解。我们通过信息论的视角来研究这个问题，将模型生成输出的条件熵与相应训练数据的条件熵进行比较。给定成对的输入-输出样本，我们使用条件熵及其基于冯·诺依曼熵的矩阵类比方法来衡量超出条件输入所能解释的输出变异性，而无需同一提示的多个参考输出。在具有公开可用训练数据的大语言模型家族中，包括OLMo、Pythia和GPT-Neo，我们一致发现，在不同的模型规模、序列长度和解码策略下，模型生成的输出都表现出比其训练数据更低的条件熵。我们观察到类似的……（摘要截断）

    arXiv:2609.02275v1 Announce Type: cross  Abstract: Large language models are trained to model conditional distributions over text, yet it remains inadequately understood whether they capture the full diversity of plausible outputs present in their training data. We study this question through an information-theoretic lens by comparing the conditional entropy of model-generated outputs with that of the corresponding training data. Given paired input-output samples, we use conditional entropy and its matrix-based analogue based on von Neumann entropy to measure output variability beyond what is explained by the conditioning input, without requiring multiple reference outputs for the same prompt. Across LLM families with publicly available training data, including OLMo, Pythia, and GPT-Neo, we consistently find that model-generated outputs exhibit lower conditional entropy than their training data, across different model scales, sequence lengths, and decoding strategies. We observe a simi
    
[^47]: CoMerge：面向多任务模型合并的冲突驱动偏好优化

    CoMerge: Conflict-Driven Preference Optimization for Multi-Task Model Merging

    [https://arxiv.org/abs/2609.02273](https://arxiv.org/abs/2609.02273)

    CoMerge 提出一种冲突驱动的偏好优化框架，将模型合并重新表述为偏好优化问题，利用朴素合并的缺陷作为困难负样本自监督构建偏好对，并通过优化轻量级的逐张量合并系数来缓解参数空间干扰，从而提升多任务大语言模型的合并效果。

    

    模型合并为构建多任务大语言模型（LLM）提供了一种无需完整模型重训练的高效范式，但其仍然受到参数干扰问题的挑战。尽管现有方法旨在保留单个专家模型的能力并缓解干扰，但它们通常不会直接从朴素合并所暴露出的性能退化行为中学习。在本文中，我们提出了一种面向模型合并的冲突驱动偏好优化框架，该框架将模型合并重新表述为一个偏好优化问题。该方法利用一种自监督的、冲突驱动的策略，将朴素合并方法（如任务算术）的缺陷作为困难负样本，从而在无需外部标注的情况下构建偏好对。通过应用偏好优化来细化轻量级的、逐张量的合并系数，CoMerge 使模型能够缓解参数空间中的干扰。

    arXiv:2609.02273v1 Announce Type: new  Abstract: Model merging provides an efficient paradigm for constructing multi-task large language models (LLMs) without full model retraining, yet it remains challenged by parameter interference. While existing methods aim to preserve the capabilities of individual expert models and mitigate interference, they generally do not directly learn from the potentially degraded behaviors exposed by naive merging. In this paper, we propose a conflict-driven preference optimization framework for model merging (CoMerge), which reformulates model merging as a preference optimization problem. The approach utilizes a self-supervised, conflict-driven strategy that leverages the defects of naive merging methods (e.g., task arithmetic) as hard negative samples to construct preference pairs without external annotations. By applying preference optimization to refine lightweight, tensor-wise merging coefficients, CoMerge enables the model to mitigate parameter-space
    
[^48]: PaperCompiler：通过仓库级规格编译实现忠实的论文到代码生成

    PaperCompiler: Faithful Paper-to-Code Generation via Repository-Level Specification Compilation

    [https://arxiv.org/abs/2609.02272](https://arxiv.org/abs/2609.02272)

    论文提出PaperCompiler框架，将基于论文的证据编译为显式的仓库级实现规格，避免了现有论文到代码智能体中间输出被下游编码智能体忽略或曲解的问题，从而实现更忠实的论文到代码生成。

    

    将研究论文忠实地转化为仓库级实现仍然具有挑战性，因为论文通常在高层次上描述方法，将实现假设隐含其中，并要求生成的代码仓库保持方法逻辑、评估协议和跨文件一致性。尽管论文到代码智能体最近取得了进展，但它们的中间输出通常以自由形式的计划或摘要呈现，下游编码智能体可能会忽略、重新解释或压缩这些内容，导致算法简化和仓库结构不一致。为了应对这些挑战，我们提出了PaperCompiler，一个将基于论文的证据编译为显式仓库级实现规格的论文到代码生成框架。PaperCompiler在获取实现相关证据的同时，保留来源出处，并区分论文支持的、推断的、外部委托的以及未解决的信息。（摘要原文截断）

    arXiv:2609.02272v1 Announce Type: cross  Abstract: Faithfully translating research papers into repository-level implementations remains challenging because papers often describe methods at a high level, leave implementation assumptions implicit, and require generated repositories to preserve method logic, evaluation protocols, and cross-file consistency. Despite recent advances in paper-to-code agents, their intermediate outputs are often presented as free-form plans or summaries that downstream coding agents may ignore, reinterpret, or compress, leading to algorithmic simplification and inconsistent repository structure. To address these challenges, we introduce PaperCompiler, a paper-to-code generation framework that compiles paper-grounded evidence into explicit repository-level implementation specifications. PaperCompiler grounds implementation-relevant evidence while preserving source provenance and distinguishing paper-supported, inferred, externally delegated, and unresolved inf
    
[^49]: 从检测到特征刻画：日本X平台上“愤怒诱饵”的大规模研究

    From Detection to Characterization: A Large-Scale Study of Ragebait on Japanese X

    [https://arxiv.org/abs/2609.02262](https://arxiv.org/abs/2609.02262)

    本研究利用LLM辅助标注构建数据集并训练出日语愤怒诱饵检测集成分类器，首次对X平台日语帖子进行大规模分析，发现愤怒诱饵在政治、歧视、公共卫生和人际冲突等争议性话题中更为普遍。

    

    愤怒诱饵指的是故意设计用来激起愤怒或义愤，从而增加关注度和互动量的在线内容。然而，目前对愤怒诱饵的可靠大规模检测与系统性分析仍然有限，阻碍了人们理解其流行程度、影响及缓解措施的努力。本研究旨在开发一个有效的愤怒诱饵检测框架，并大规模地阐明愤怒诱饵的特征，为理解和缓解网络上具有情绪煽动性的内容提供基础。我们在大型语言模型（LLM）的辅助下构建了一个标注数据集，并训练了多个日语语言模型用于愤怒诱饵检测。随后，将由此得到的集成分类器应用于X平台上日语帖子的大规模数据集。我们的分析表明，愤怒诱饵在政治与社会争议性话题中更为普遍，包括政治、歧视、公共卫生和人际冲突等领域。

    arXiv:2609.02262v1 Announce Type: cross  Abstract: Ragebait refers to online content intentionally designed to provoke anger or outrage and thereby increase attention and engagement. However, reliable large-scale detection and systematic analysis of ragebait remain limited, hindering efforts to understand its prevalence, impact, and mitigation. This study aims to develop an effective ragebait detection framework and to clarify the characteristics of ragebait at scale, providing a basis for understanding and mitigating emotionally provocative content online. We constructed a labeled dataset with the assistance of a large language model (LLM) and trained several Japanese language models for ragebait detection. The resulting ensemble classifier was then applied to a large-scale dataset of Japanese-language posts on X. Our analysis shows that ragebait is more prevalent in politically and socially contentious topics, including politics, discrimination, public health, and interpersonal confl
    
[^50]: APEx：面向自适应深度研究问答的智能体程序性经验蒸馏

    APEx: Distillation of Agent Procedural Experience for Adaptive Deep Research Question Answering

    [https://arxiv.org/abs/2609.02253](https://arxiv.org/abs/2609.02253)

    APEx提出分层经验利用框架，通过执行器-蒸馏器-规划器闭环架构和三阶段GRPO交替训练，将智能体交互历史蒸馏为程序性技能，并在测试时作为先验支持规划器在线自适应，从而提升深度研究问答性能。

    

    深度研究智能体通过外部工具增强大型语言模型，以多轮推理的方式回答复杂的长时程问题。从先前经验中学习对持续改进至关重要，然而现有方法要么检索冗长的任务特定轨迹而加重决策负担，要么蒸馏出的程序性技能与下游策略适配相脱节。我们提出APEx，一个分层经验利用框架，它将交互历史组织为实例级轨迹记忆和类别级程序性技能，并通过执行器、蒸馏器和规划器三个模块构成的闭环架构将二者耦合。这三个模块通过三阶段交替的GRPO训练范式进行优化，实现奖励引导的技能蒸馏而非固定提示生成。在测试时，蒸馏出的技能作为程序性先验，通过技能引导的测试时强化（学习）支持规划器的在线自适应。

    arXiv:2609.02253v1 Announce Type: new  Abstract: Deep research agents augment large language models with external tools to answer complex, long-horizon questions through multi-turn reasoning. Learning from prior experience is crucial for continual improvement, yet existing methods either retrieve verbose task-specific traces that burden decision-making, or distill procedural skills that remain decoupled from downstream policy adaptation. We propose APEx, a hierarchical experience utilization framework that organizes interaction history into instance-level trajectory memories and category-level procedural skills, and couples them through a closed-loop architecture of Executor, Distiller, and Planner. The three modules are optimized via a three-stage alternating GRPO training paradigm, enabling reward-guided skill distillation rather than fixed-prompt generation. At test time, distilled skills serve as procedural priors for online Planner adaptation through skill-guided test-time reinfor
    
[^51]: RideSkill：一种基于大语言模型驱动自动进化的泛化拼车分层算法

    RideSkill: A Hierarchical Algorithm for Generalized Ride Sharing with LLM-Driven Automatic Evolution

    [https://arxiv.org/abs/2609.02250](https://arxiv.org/abs/2609.02250)

    该论文提出RideSkill，一种由大语言模型驱动自动进化的分层算法，用于解决泛化拼车问题，克服了传统多智能体强化学习方法在泛化性、可迁移性和大规模训练方面的局限。

    

    拼车允许具有不同起讫点（OD对）的多名乘客共享同一辆车辆，是一个具有挑战性的运营问题，因为它需要在不确定且多变的情况下，高效地将不同OD对的订单捆绑并分配给车辆。尽管多智能体强化学习（MARL）解决方案已取得了有前景的性能，但它们存在泛化能力有限（难以适应不同的环境场景）、可迁移性低（难以适应不同的平台目标）以及在大规模系统中训练困难（如维度灾难）等问题。最近，受大语言模型（LLM）规模化发展的启发，一些工作将LLM引入网约车系统，要么直接将LLM用作决策智能体，要么利用LLM进行自动算法设计。然而，这些方法均不支持车辆共享，这使问题变得更加复杂。

    arXiv:2609.02250v1 Announce Type: cross  Abstract: Ride-sharing, which allows multiple passengers with different origin-destination (OD) pairs to share a single vehicle, is a challenging operational problem, as it requires orders with different OD pairs to be efficiently bundled and assigned to vehicles under uncertain and varying scenarios. Although multi-agent reinforcement learning (MARL) solutions have achieved promising performance, they suffer from limited generalization (adapting to different environmental scenarios), low transferability (adapting to different platform objectives), and training difficulties in large-scale systems, such as the curse of dimensionality. Recently, motivated by the scaling of large language models (LLMs), several works have incorporated LLMs into ride-hailing systems, either by employing LLMs directly as decision-making agents or using them for automatic algorithm design. However, none of these approaches support vehicle sharing, which complicates th
    
[^52]: LeakageBench：文档图像中个人身份信息脱敏的文档级泄露风险基准

    LeakageBench: Document-Level Leakage Risk for Redacting Personally Identifiable Information in Document Images

    [https://arxiv.org/abs/2609.02207](https://arxiv.org/abs/2609.02207)

    该论文提出了LeakageBench——一个用于评估文档图像中PII脱敏文档级泄露风险的挑战性基准数据集，实验表明即使借助Code Interpreter等工具将GPT-5.5的定位F1从0.090提升至0.249，页面级泄露率仍高达0.968，揭示了现有方法在文档级脱敏安全性上的严重不足。

    

    真实世界的个人身份信息（PII）脱敏通常在文档图像上进行——包括扫描件、截图和PDF渲染图——其中OCR错误、版面结构和视觉噪声决定了敏感信息是否真正被移除。现有的PII基准大多以文本为中心，无法衡量文档级的脱敏风险：只要遗漏一个标识符，整页文档就仍然不安全。我们提出了LeakageBench，这是一个包含500张文档图像的挑战性基准数据集，其中包含11,954个符合GDPR的PII标注，涵盖直接标识符、关联密钥和上下文再识别信息。我们使用实体级F1、分组泄露和文档级泄露指标，评估了通用OCR流水线、商业及任务适配的依赖OCR的检测器，以及免OCR的视觉语言模型。Code Interpreter将GPT-5.5的定位F1从0.090提升至0.249，但关键的页面级泄露率仍高达0.968。这些结果表明，强大的……

    arXiv:2609.02207v1 Announce Type: cross  Abstract: Real-world personally identifiable information (PII) redaction often operates on document images---scans, screenshots, and PDF renderings---where OCR errors, layout structure, and visual noise determine whether sensitive information is actually removed. Existing PII benchmarks are mostly text-centric and do not measure document-level redaction risk: a page remains unsafe if even one identifier is missed. We introduce LeakageBench, a challenge set of 500 document images with 11,954 GDPR-aligned PII annotations spanning direct identifiers, linkage keys, and contextual re-identification surfaces. We evaluate generic OCR pipelines, commercial and task-adapted OCR-dependent detectors, and OCR-free vision-language models using entity-level F1, group-wise leakage, and document-level leakage metrics. Code Interpreter raises GPT-5.5 localization F1 from 0.090 to 0.249, but critical page-level leakage remains 0.968. These results show that stron
    
[^53]: 广度胜过深度：通过面向广度的后缀搜索改进基于GCG的越狱优化

    Breadth Beats Depth: Improving GCG-Based Jailbreak Optimization with Breadth-Oriented Suffix Search

    [https://arxiv.org/abs/2609.02172](https://arxiv.org/abs/2609.02172)

    本文提出即插即用框架BOSS，通过尾部聚焦对抗损失和面向广度的后缀搜索策略改进基于GCG的越狱攻击优化，在提升攻击成功率的同时降低了优化时间。

    

    基于优化的越狱攻击（如贪心坐标梯度法GCG）通过在白盒源模型上优化对抗性后缀，实现了较强的有效性和可迁移性。然而，现有的基于GCG的方法依赖于平均对抗损失和深度贪心搜索，这可能过度强调容易越狱的行为，而忽视后缀空间中有前景的区域。我们提出了BOSS，这是一个即插即用的框架，通过面向广度的后缀搜索来改进基于GCG的越狱优化。BOSS使用尾部聚焦对抗损失、标准源损失和行为覆盖率来选择终端后缀，然后探索多条短轨迹，并有选择地延续有前景的后缀。在公开基准上的实验表明，BOSS在多种基于GCG的方法上提升了攻击成功率，同时缩短了优化时间。

    arXiv:2609.02172v1 Announce Type: new  Abstract: Optimization-based jailbreak attacks such as Greedy Coordinate Gradient (GCG) achieve strong effectiveness and transferability by optimizing adversarial suffixes on white-box source models. However, existing GCG-based methods rely on averaged adversarial loss and deep greedy search, which can over-emphasize easy-to-jailbreak behaviors and overlook promising regions of the suffix space. We propose BOSS, a plug-and-play framework that improves GCG-based jailbreak optimization through breadth-oriented suffix search. BOSS uses Tail-Focused Adversarial Loss (TFAL), standard source loss, and behavior coverage to select terminal suffixes, then explores multiple short trajectories and selectively continues promising suffixes. Experiments on public benchmarks show that BOSS improves attack success rates across multiple GCG-based methods while reducing optimization time.
    
[^54]: 粤语适配语言模型能更好地预测粤语阅读吗？一项跨模型眼动追踪评估

    Do Cantonese-Adapted Language Models Better Predict Cantonese Reading? A Cross-Model Eye-Tracking Evaluation

    [https://arxiv.org/abs/2609.02163](https://arxiv.org/abs/2609.02163)

    本研究基于自然粤语眼动追踪数据，通过词汇意外度、词性意外度和熵等信息论指标评估发现，经过大规模粤语继续预训练和指令微调的 CantoneseLLM-7B 比通用模型或轻度粤语适配模型更能预测人类粤语阅读行为。

    

    源自自回归语言模型的信息论度量被广泛用于刻画塑造人类阅读的预期，但针对特定语言变体的训练是否能提升这种心理语言学层面的对齐仍不清楚。对于粤语而言，这个问题依然悬而未决——近期的自然语言处理评估显示，与面向普通话的模型或通用模型相比，粤语特定训练的收益并不一致。本研究利用自然语境下的粤语眼动追踪数据，比较了两组同家族模型的适配对比：CKIP GPT-2 Tiny 与其经过轻度粤语适配的衍生模型 JED351，以及 Qwen2.5-7B 与 CantoneseLLM-7B（后者经历了规模大得多的粤语继续预训练与指令微调）。研究者从每个模型中提取词汇意外度、词性意外度、目标词之前的熵以及熵减这四项指标。结果显示，词汇意外度和四指标联合模型一致地更倾向于 CantoneseLLM-7B。

    arXiv:2609.02163v1 Announce Type: new  Abstract: Information-theoretic measures derived from autoregressive language models are widely used to characterize the expectations that shape human reading, but whether language-variety-specific training improves such psycholinguistic alignment remains unclear. This question is still open for Cantonese, where recent NLP evaluations reported mixed benefits from Cantonese-specific training relative to Mandarin-oriented or general-purpose models. Using naturalistic Cantonese eye-tracking data, we compare two within-family adaptation contrasts: CKIP GPT-2 Tiny versus its lightly Cantonese-adapted JED351 derivative, and Qwen2.5-7B versus CantoneseLLM-7B, which underwent substantially more extensive Cantonese continued pretraining and instruction tuning. From each model, we derive lexical surprisal, POS surprisal, entropy before the target, and entropy reduction. Lexical surprisal and the joint four-metric model consistently favor CantoneseLLM-7B, fo
    
[^55]: 反对！律师智能体缓解法律判决预测中的有罪偏见

    OBJECTION! Lawyer Agents Mitigate Guilty Bias in Legal Judgment Prediction

    [https://arxiv.org/abs/2609.02158](https://arxiv.org/abs/2609.02158)

    该论文提出OBJECTION推理时框架，将对抗性律师智能体嵌入罪责、违法性和可责性三步推理的每个阶段，通过注入法律辩护论点主动挑战模型的有罪预设，从而缓解法律判决预测中的“有罪偏见”。

    

    法律判决预测（LJP）模型通常在从控方视角描述案件事实的文档上进行训练，且现有数据集在标签上严重偏向有罪判决。因此，这些模型存在“有罪偏见”，会盲目将控方叙述当作客观事实加以接受。以往研究通过采用三步推理结构或在合成的无罪数据上训练，虽提升了整体准确率，但仍无法在推理阶段缓解偏见。本文提出OBJECTION，一种推理时流程，它在罪责、违法性和可责性这三步推理的每一步中都集成了一个对抗性律师智能体。与通用批评者不同，该智能体通过在每个推理阶段注入法律辩护论点，主动挑战模型的有罪预设。为全面评估该方法，我们还构建了一个包含3.4千条真实案例的新“自然无罪”数据集。

    arXiv:2609.02158v1 Announce Type: cross  Abstract: Legal Judgment Prediction (LJP) models are typically trained on documents that describe facts from a prosecutorial perspective. Existing datasets further exhibit severe label imbalance toward guilty outcomes. Consequently, these models suffer from "Guilty Bias", blindly accepting the prosecution's narrative as objective truth. Previous studies employing three-step reasoning structures or training on synthetically generated innocence data improve overall accuracy, but they still fail to mitigate bias at inference time.   In this paper, we introduce OBJECTION, an inference-time pipeline that integrates an Adversarial Lawyer Agent into each 3-step reasoning of offense, unlawfulness, and culpability. Unlike generic critics, our agent actively challenges the model's presumptions of guilt by injecting legal defense arguments at each reasoning stage. To thoroughly evaluate this, we present a new "Natural Innocent" dataset including 3.4k real-
    
[^56]: 面向中文学习者语法错误标注的分层分类体系

    A Layered Taxonomy for Chinese Learner Grammatical Error Annotation

    [https://arxiv.org/abs/2609.02153](https://arxiv.org/abs/2609.02153)

    本文提出了一种连接中文语法纠错与教学错误分析的分层语法错误标注体系，采用三层核心标签加中文特有扩展范畴的设计，并通过MuCGEC覆盖率分析和多模型一致性研究验证了其有效性。

    

    中文学习者作文中的语法错误标注需要既保持一致性又具有语言学意义的标签。本文提出了一种分层式标注方案，将计算视角下的中文语法纠错（CGEC）与教学层面的错误分析相联系。该方案首先识别字符和标点层面的正字法错误，并按编辑操作及其子类型进行标注。其他错误则获得由编辑操作、语言学领域和词性构成的三层核心标签，并可选地扩展中文特有范畴，包括体、情态、比较、论元结构和补语。该分类体系借鉴了CGEC资源、学习者错误分类学以及现代汉语语法，通过对自动抽取的MuCGEC编辑进行覆盖率分析，以及在五个人大型语言模型应用该体系的初步一致性研究对其进行评估。结果支持了这种分层方法，同时识别出类别边界方面的问题。

    arXiv:2609.02153v1 Announce Type: new  Abstract: Grammatical error annotation in Chinese learner writing requires labels that are both consistent and linguistically meaningful. This paper proposes a layered scheme linking computational Chinese grammatical error correction (CGEC) with pedagogical error analysis. The scheme first identifies character- and punctuation-level orthographic errors, labeling them by edit operation and subtype. Other errors receive a three-layer core label combining edit operation, linguistic domain, and part of speech, with optional Chinese-specific extensions for aspect, modality, comparison, argument structure, and complements. Drawing on CGEC resources, learner-error taxonomies, and Mandarin grammar, the taxonomy is evaluated through a coverage analysis of automatically extracted MuCGEC edits and a preliminary consistency study in which five large language models apply it to a sample. The results support the layered approach while identifying category bound
    
[^57]: EmoStance：基于表情符号弱监督的共情回复生成响应侧情感取向控制

    EmoStance: Response-Side Affective-Orientation Control for Empathetic Response Generation via Emoji Weak Supervision

    [https://arxiv.org/abs/2609.02133](https://arxiv.org/abs/2609.02133)

    该论文提出 EmoStance 方法，将多标注者表情符号分布作为弱监督证据来诱导近似倾听者立场的潜在控制空间，并通过连续前缀嵌入引导冻结的指令微调大语言模型，实现共情回复生成中的响应侧情感取向控制。

    

    共情回复生成要求模型不仅要决定说什么，还要决定如何回应前一位说话者的情感状况。我们将其形式化为响应侧情感取向控制问题，并将多标注者的表情符号分布用作弱情感-态度证据（而非作为输出符号或黄金标准标签），以诱导出一个在操作上近似倾听者立场的潜在控制空间。我们构建了 EmojiDialogue——一个在 EmpatheticDialogues 基础上扩展的语句级数据集，包含表情符号投票和置信度分数——并提出 EmoStance 方法：该方法对源侧情感表达进行建模，从对话上下文和说话者角色预测软性的响应侧取向，并通过连续前缀嵌入来引导一个冻结的指令微调大语言模型。在由 20 名标注者参与的 800 次判断的盲测成对评估中，EmoStance 取得了 62.2% 的决定性胜率，其中在上下文具体性和针对个体化方面的增益最为明显。

    arXiv:2609.02133v1 Announce Type: new  Abstract: Empathetic response generation requires models to decide not only what to say, but also how to respond to the previous speaker's affective situation. We formulate this as response-side affective-orientation control and use multi-annotator emoji distributions as weak affective--attitudinal evidence, rather than as output symbols or gold labels, to induce a latent control space that operationally approximates listener stance. We construct EmojiDialogue, an utterance-level extension of EmpatheticDialogues with emoji votes and confidence scores, and propose EmoStance, which models source-side affective expression, predicts a soft response-side orientation from dialogue context and speaker roles, and steers a frozen instruction-tuned LLM through continuous prefix embeddings. In blind pairwise evaluation with 20 annotators and 800 judgments, EmoStance achieves a 62.2% decisive win rate, with the clearest gains in contextual specificity and per
    
[^58]: C³T：面向社交媒体对话树中情感转变的反事实因果推理

    C$^{3}$T: Counterfactual Causal Reasoning for Sentiment Shifts in Social-Media Conversation Trees

    [https://arxiv.org/abs/2609.02131](https://arxiv.org/abs/2609.02131)

    该论文提出CaSiRe因果情感推理标注层与C³T反事实因果对话Transformer模型，通过将否认、证据、攻击等话语行为视为干预措施，联合预测并解释社交媒体谣言对话树中情感如何发生转变及其因果来源。

    

    社交媒体帖子串中的情感不仅在不同帖子之间有所差异；随着用户在分支回复树中对主张、更正、证据和敌意言论作出反应，情感还会随之发生转变。我们通过将话语行为（如否认/更正、证据/链接、毒性/攻击）视为候选干预措施，来研究以谣言为中心的对话树中情感变化的原因，并回答以下问题：(i) 某条回复表达了何种情感，(ii) 该情感相对于其父节点是否发生了转变，以及 (iii) 哪条先前的消息最有可能驱动了该回复的情感。为支持这一研究设定，我们提出了CaSiRe，这是一个构建在公开谣言对话数据集之上的因果情感推理层，它增加了帖子级情感标签、诱导生成的父子情感转变标签、经过校准的多标签干预标签，以及显式标注的因果来源标签。随后，我们提出C³T（反事实因果对话Transformer），这是一种线程结构的时间模型，可联合预测……

    arXiv:2609.02131v1 Announce Type: cross  Abstract: Sentiment in social-media threads does not only vary across posts; it shifts as users react to claims, corrections, evidence, and hostility within a branching reply tree. We study why sentiment changes in rumor-centric conversation trees by treating discourse moves (e.g., denial/correction, evidence/link, toxicity/attack) as candidate interventions and asking (i) what sentiment a reply expresses, (ii) whether the sentiment shifts relative to its parent, and (iii) which prior message most plausibly drove the reply's sentiment. To support this setting, we introduce CaSiRe, a causal sentiment reasoning layer over public rumor conversation datasets that adds post-level sentiment labels, induced parent-child shift labels, calibrated multi-label intervention tags, and explicitly annotated causal-source labels. We then propose C$^{3}$T (Counterfactual Causal Conversation Transformer), a thread-structured temporal model that jointly predicts n
    
[^59]: AI智能体重塑人类群体中的共识形成

    AI agents reshape consensus formation in human groups

    [https://arxiv.org/abs/2609.02122](https://arxiv.org/abs/2609.02122)

    本研究通过协作描述游戏实验发现，LLM智能体在人机混合群体中的比例会以三种截然不同的方式重塑共识形成——低比例促进人类主导共识、中等比例阻碍收敛、高比例恢复强共识但使其转向更抽象的智能体主导约定。

    

    随着大语言模型（LLM）智能体从工具转变为人类群体中的参与者，集体行为研究面临的一个根本问题是：它们日益增长的存在如何重塑共识的形成。本研究在一个协作描述游戏中考察人机混合群体，其中共享的约定通过多轮随机两两交流而涌现。通过改变LLM智能体的比例，我们识别出三种截然不同的共识形成机制：低比例智能体促进人类主导的共识；中等比例会破坏收敛过程；高比例则恢复强共识，同时使其转向智能体主导的约定。至关重要的是，这些机制不仅在收敛强度上有所不同，而且在所得共识的语义基础和交流形式上也存在差异：人类主导的共识更加具体、整体，植根于共享的现实世界类比，而智能体主导的共识则更加抽象。

    arXiv:2609.02122v1 Announce Type: new  Abstract: As large language model (LLM) agents shift from tools to participants in human groups, a fundamental question for collective behavior is how their growing presence reshapes consensus formation. Here we study mixed human-AI groups in a collaborative description game, in which shared conventions emerge through repeated rounds of random pairwise communication. Varying the proportions of LLM agents, we identify three distinct regimes of consensus formation: low agent proportions facilitate human-led consensus, intermediate proportions disrupt convergence, and high proportions restore strong consensus while shifting it toward agent-led conventions. Crucially, these regimes differ not only in the strength of convergence, but also in the semantic grounding and communicative form of the resulting consensus: human-led consensus is more concrete, holistic, and grounded in shared real-world analogies, whereas agent-led consensus is more abstract, l
    
[^60]: text2ql：基于语言无关中间表示的多目标自然语言查询

    text2ql: Multi-Target Natural Language Querying via a Language-Agnostic Intermediate Representation

    [https://arxiv.org/abs/2609.02115](https://arxiv.org/abs/2609.02115)

    text2ql框架通过语言无关的中间表示QueryIR和可插拔渲染器架构，实现了同时面向SQL和GraphQL的多目标自然语言查询，其零LLM确定性模式在3.2毫秒中位延迟下达到100%执行准确率，并为每个生成的查询提供运行时置信度分数。

    

    数据库的自然语言接口传统上存在三个结构性局限：仅针对关系型SQL、在查询时无条件依赖大语言模型（LLM）推理，以及当生成的查询语义不正确时缺乏任何运行时信号。本文提出了text2ql，一个开源Python框架，通过语言无关的中间表示和可插拔的渲染器架构解决了这三个局限。单一的七阶段检测流水线同时服务于SQL和GraphQL两种目标；零LLM确定性模式以3.2毫秒的中位延迟实现100%的执行准确率，且无API成本；每个生成的查询都携带一个由加性信号模型计算出的、范围在[0.15, 0.97]之间的运行时置信度分数。该系统在Spider和BIRD基准的50个查询随机样本上进行了评估（结果为指示性结果；计划进行完整数据集评估）……

    arXiv:2609.02115v1 Announce Type: cross  Abstract: Natural language interfaces to databases have traditionally suffered from three structural limitations: exclusive targeting of relational SQL, unconditional dependence on large language model (LLM) inference at query time, and absence of any runtime signal when generated queries are semantically incorrect. This paper presents text2ql, an open-source Python framework that addresses all three limitations through a language-agnostic Intermediate Representation (QueryIR) and a pluggable renderer architecture. A single seven-stage detection pipeline serves both SQL and GraphQL targets; a zero-LLM deterministic mode delivers 100% execution accuracy at a median latency of 3.2 ms with no API cost; and every generated query carries a runtime confidence score in [0.15, 0.97] computed from an additive signal model. Evaluated on 50-query random samples from the Spider and BIRD benchmarks (indicative results; full-set evaluation is planned), the LL
    
[^61]: 预测而非迭代：扩散语言模型的高效自适应长度填充

    Predict, Don't Iterate: Efficient Adaptive-Length Infilling for Diffusion Language Models

    [https://arxiv.org/abs/2609.02108](https://arxiv.org/abs/2609.02108)

    提出一种“预测而非迭代”的高效自适应长度填充方法，让扩散语言模型直接一次性预测合适的填充长度，从而克服对初始长度的敏感性，并避免迭代搜索带来的大量额外计算开销。

    

    扩散语言模型（DLMs）已成为自回归范式的一种有前景的替代方案。凭借双向注意力和任意顺序生成能力，DLMs 天然适合填充任务，即在前缀和后缀条件的约束下生成中间片段的任务。然而，填充任务对片段的长度非常敏感，而 DLMs 要求在生成前预先固定长度。尽管已有研究将 DLMs 扩展到动态长度，但这些方法仍存在两个局限：（i）对初始长度的敏感性。这些方法需要预设一个长度来初始化搜索，且对该初始长度高度敏感，往往产生次优结果。（ii）推理效率低下。它们要么在生成过程中插入改变长度的操作，要么利用多步去噪置信度反复搜索合适的长度，两者都会引入大量额外的前向传播和计算开销。

    arXiv:2609.02108v1 Announce Type: cross  Abstract: Diffusion language models (DLMs) have emerged as a promising alternative to the auto-regressive paradigm. With bidirectional attention and any-order generation, DLMs naturally fit infilling tasks, which require generating a middle span conditioned on both the prefix and the suffix. However, infilling is sensitive to the length of the span, while DLMs require the length to be fixed before generation. Although prior studies extend DLMs to dynamic lengths, they still suffer from two limitations. (i) Sensitivity to initial length. These methods require a preset length to initialize the search and are highly sensitive to this initial length, often yielding suboptimal results. (ii) Inference inefficiency. They either insert length-changing operations during generation or repeatedly search for an appropriate length using multi-step denoising confidence, both of which introduce substantial extra forward passes and computational cost. Therefore
    
[^62]: MASkills：面向多智能体LLM系统的持续技能优化

    MASkills: Continual Skills Optimization for Multi-Agent LLM Systems

    [https://arxiv.org/abs/2609.02094](https://arxiv.org/abs/2609.02094)

    MASkills是一个持续学习框架，通过技能条件化信用分配、分层信用聚合和动量平滑优化的新流水线，使多智能体LLM系统的技能库能够通过精炼、归纳、巩固和剪枝不断演进优化。

    

    基于LLM的多智能体系统在复杂任务上展现出了强大的性能，然而从交互经验中实现持续改进仍然具有挑战性。现有的自我反思方法会构建经验记忆，但记忆大多难以调用、精炼或扩展，而智能体技能则提供了一个更可操作的单元：结构化的程序性知识，它明确了何时行动、如何行动以及使用哪些资源或工具。我们提出了MASkills，这是一个通过智能体技能来优化多智能体LLM系统的持续学习框架。MASkills提出了一个新的智能体优化流水线，集成了技能条件化的信用分配、分层信用聚合和动量平滑优化，使智能体技能库能够通过精炼、归纳、巩固和剪枝不断演进。在HotpotQA、LoCoMo和GAIA上的实验证明了MASkills在多种智能体任务上的有效性。

    arXiv:2609.02094v1 Announce Type: new  Abstract: LLM-based multi-agent systems have shown strong performance on complex tasks, yet continual improvement from interaction experience remains challenging. Existing self-reflection methods build experience memories, but memories are mostly hard to invoke, refine, or scale, while agent skills offer a more actionable unit: structured procedural knowledge that specifies when to act, how to act, and which resources or tools to use. We introduce MASkills, a continual learning framework that optimizes multi-agent LLM systems through agent skills. MASkills presents a new agent-optimization pipeline that integrates skill-conditioned credit assignment, hierarchical credit aggregation, and momentum-smoothed optimization, enabling agent skill libraries to evolve through refinement, induction, consolidation, and pruning. Experiments on HotpotQA, LoCoMo, and GAIA demonstrate the effectiveness of MASkills across multiple agentic tasks. Our code is availa
    
[^63]: 基于门控奇异向量收缩的选择性知识编辑逆转

    Selective Knowledge Edit Reversal via Gated Singular Vector Shrinkage

    [https://arxiv.org/abs/2609.02091](https://arxiv.org/abs/2609.02091)

    本文提出基于门控奇异向量收缩的谱分析逆转框架，通过假设编辑信息稀疏编码于权重矩阵主奇异子空间，实现了对大语言模型中特定知识编辑的选择性精准逆转，同时保留其他有益编辑不受影响。

    

    知识编辑为更新大语言模型中的事实知识提供了一种高效方式。然而，恶意编辑可能引入安全风险，因此有必要逆转不良的编辑效果。现有的针对参数修改型编辑的逆转方法主要集中于全局删除，这可能会同时抹除应当保留的有益编辑。本文研究编辑知识的选择性逆转问题，其目标是在保留其余已编辑事实的同时，逆转特定的目标编辑事实。基于“每次编辑均以稀疏方式编码在编辑后矩阵的主子空间中”这一假设，我们提出了一种基于谱分析的逆转框架，通过门控奇异向量收缩在编辑权重的主奇异子空间内定位对编辑敏感的成分。在多种设置下的实验表明，我们的方法能够在保留无关已编辑事实的同时有效逆转所选编辑。这些结果表明……

    arXiv:2609.02091v1 Announce Type: new  Abstract: Knowledge editing provides an efficient way to update factual knowledge in large language models. However, malicious edits may introduce safety risks, making it necessary to reverse undesirable editing effects. Existing reversal methods for parameter-modifying edits mainly focus on global removal, which may also erase beneficial edits that should be preserved. In this paper, we study selective reversal of edited knowledge, where the goal is to reverse targeted edited facts while preserving the remaining edited facts. Based on the hypothesis that each edit is sparsely encoded within the dominant subspace of the edited matrix, we propose a spectral-based reversal framework that locates edit-sensitive components within the dominant singular subspace of edited weights. Experiments across multiple settings demonstrate the effectiveness of our method in reversing selected edits while preserving unrelated edited facts. These results suggest tha
    
[^64]: IDEEA：通过激活簇匹配实现无需训练的输入相关引导

    IDEEA: training-free Input-Dependent stEEring via Activation cluster matching

    [https://arxiv.org/abs/2609.02089](https://arxiv.org/abs/2609.02089)

    提出IDEEA框架，通过对每个注意力头的正负激活进行聚类并求解最优匹配问题来构建簇条件化的引导方向，首次实现了无需训练、随输入自适应变化的大模型激活引导，克服了传统固定单一方向引导的根本局限。

    

    引导技术通过在推理时向选定的激活中注入偏置来对齐大型语言模型（LLM），与监督微调或强化学习等权重更新方法相比，这是一种成本远低的替代方案。然而，现有的大多数无需训练的引导方法都是输入无关的：只拟合一次单一方向，并在所有输入间共享。这在根本上存在局限，因为不同的输入占据激活空间的不同区域，并且针对同一目标概念容许不同的最优引导方向，就像相对于固定损失的梯度会随输入而变化一样。我们通过 IDEEA（通过激活簇匹配实现输入相关引导）来弥补这一空白，这是一个用于输入相关引导的无需训练的框架。IDEEA 对每个注意力头的正负激活支持进行聚类，并求解一个最优匹配问题来构建一组条件于簇的方向……（摘要原文在此处被截断）

    arXiv:2609.02089v1 Announce Type: new  Abstract: Steering aligns large language models (LLMs) by injecting a bias into selected activations at inference time, offering a far cheaper alternative to weight-update methods such as supervised fine-tuning or reinforcement learning. However, most existing training-free steering methods are input-independent: a single direction is fitted once and shared across all inputs. This is fundamentally limiting as different inputs occupy different regions of the activation space and admit different optimal steering directions toward the same target concept, much as the gradient with respect to a fixed loss varies from input to input. We close this gap with IDEEA (Input-Dependent stEEring via Activation cluster matching), a training-free framework for input-dependent steering. IDEEA clusters the positive and negative activation supports per attention head, and solves an optimal-matching problem to construct a set of cluster-conditional directions, all a
    
[^65]: XMerge：用于大语言模型深度压缩的跨轴选择与重构式层合并

    XMerge: Cross-Axis Selection and Reconstructive Layer Merging for LLM Depth Compression

    [https://arxiv.org/abs/2609.02083](https://arxiv.org/abs/2609.02083)

    XMerge 是一种训练后的大语言模型深度压缩方法，通过跨轴选择识别隐藏状态变化最小的层块，并利用局部边界重构重新拟合相邻存留块，在不改变架构、不增加推理参数、无需任务标签的情况下实现高质量的层删除压缩。

    

    删除完整的 transformer 层可以保持标准的服务架构，但现有的深度压缩方法可能会损失大量质量，且损失在不同模型间变化难以预测。我们提出了 XMerge，这是一种包含两个组件的训练后方法。跨轴选择用于识别隐藏状态的相对幅度和角度变化较小的层块，局部边界重构则重新拟合相邻的存留块以匹配原始两个块的输出。XMerge 不使用任务标签或端到端微调，也不引入架构变更或额外的推理时参数。在七个 Llama 和 Qwen 主干模型（0.5B-8B）、五个已发表的基线方法和三个层削减级别上的实验表明，其相对于基线的优势在最激进的删除设置下最大：在 k=4 时，它在 CORE（一个 22 项任务的聚合基准）上于七个主干模型中的六个上排名第一，并且在 MMLU 上也在七个中的六个上排名第一（两个基准上同时排名第一的有五个）。

    arXiv:2609.02083v1 Announce Type: cross  Abstract: Removing complete transformer layers preserves a standard serving architecture, but existing depth-compression methods can lose substantial quality, and the loss varies unpredictably across models. We introduce XMerge, a post-training method with two components. Cross-axis selection identifies a block with low relative-magnitude and angular hidden-state change, and local boundary reconstruction re-fits the adjacent surviving block to match the original two-block output. XMerge uses no task labels or end-to-end fine-tuning, and it introduces neither architectural changes nor additional inference-time parameters. Across seven Llama and Qwen backbones (0.5B-8B), five published baselines, and three layer-reduction levels, its advantage over baselines is largest at the most aggressive removal: at k=4 it ranks first on six of seven backbones on CORE (a 22-task aggregate) and, separately, on six of seven on MMLU (five of seven on both at once
    
[^66]: 多模态大语言模型中跨模态安全漂移的安全意识迁移

    Transfer Safety Awareness for Cross-Modal Safety Drift in Multimodal Large Language Models

    [https://arxiv.org/abs/2609.02082](https://arxiv.org/abs/2609.02082)

    针对多模态大语言模型中“跨模态安全漂移”这一新安全问题（无害文本结合图像即可传达有害意图且模型难以拒绝），提出轻量级的安全意识表示迁移方法（SRT），将文本安全信号迁移至视觉场景以有效缓解该风险。

    

    视觉模态增强了多模态大语言模型（MLLMs）的能力，但也引入了安全隐患：一个本身无害的文本查询在与视觉图像结合时可能传达有害意图。我们将这种现象称为“跨模态安全漂移”，我们的初步研究表明，此类请求的安全响应率显著低于包含明确不安全文本的请求。本文旨在系统研究这一问题。首先，我们进行了实证分析，识别出代表性的不安全响应模式。在此基础上，我们对模型表示和注意力机制进行了解释分析，揭示出视觉风险线索受到的关注有限，难以有效触发拒绝响应。受不安全文本处理中的安全信号可以迁移这一观察的启发，我们提出了安全意识表示迁移，这是一种轻量级的方向细化方法，能够缓解跨模态安全漂移并显著提升……

    arXiv:2609.02082v1 Announce Type: cross  Abstract: Visual modality enhances the capabilities of multimodal large language models (MLLMs) but also introduces a safety concern: a benign textual query may convey harmful intent when grounded in a visual image. We term this cross-modal safety drift and our pilot studies show that the safety response rate for such requests is substantially lower than that for requests containing explicitly unsafe text. This paper aims to systematically study this issue. First, we conduct an empirical analysis to identify representative unsafe response patterns. Building on these, we interpret model representations and attentions, revealing that visually risky cues receive limited attention and weakly trigger refusal. Motivated by the observation that safety signals from unsafe text processing can be transferred, we propose safety-awareness representation transfer (SRT), a lightweight direction-refinement method that mitigates cross-modal safety drift with a 
    
[^67]: HyGRAIL：知识图谱上成本感知且证据支撑的科学假设发现

    HyGRAIL: Cost-Aware and Evidence-Grounded Scientific Hypothesis Discovery over Knowledge Graphs

    [https://arxiv.org/abs/2609.02056](https://arxiv.org/abs/2609.02056)

    HyGRAIL 提出了一个结合异构图神经网络分诊与大语言模型审查的成本感知、证据支撑框架，通过仅将图上不确定的模糊候选假设路由给 LLM 审查，从而在知识图谱上实现高效且可靠的科学假设发现。

    

    科学知识图谱组织了从科学文献中提取的实体与关系，但其本质上仍是不完整的。因此，这类图谱中缺失的类型化链接可以代表合理的科学假设，例如材料与应用之间尚未被探索的关联。然而，科学假设发现极具挑战性，因为在类型化候选对中真正的发现极其稀少：图神经网络（GNN）虽然高效，但在处理模糊案例时并不可靠；而大语言模型（LLM）虽然知识丰富，但若穷尽式地应用则成本过高，且其并不能自然地扎根于图结构之中。我们提出了 HyGRAIL，一个成本感知且以证据为支撑的框架，它将异构 GNN 分诊与基于 LLM 的假设审查相结合。HyGRAIL 首先使用 GNN 对候选假设进行评分，并识别出一个经验证集校准的模糊区域，仅将图上不确定的案例路由至 LLM 审查。对于……

    arXiv:2609.02056v1 Announce Type: new  Abstract: Scientific knowledge graphs organize entities and relations extracted from scientific literature, but they remain inherently incomplete. Missing typed links in such graphs can therefore represent plausible scientific hypotheses, such as unexplored associations between materials and applications. However, scientific hypothesis discovery is challenging because true discoveries are extremely sparse among typed candidate pairs: graph neural networks (GNNs) are efficient but unreliable for ambiguous cases, while large language models (LLMs) are knowledgeable but too costly to apply exhaustively and are not naturally grounded in graph structures. We propose HyGRAIL, a cost-aware and evidence-grounded framework that combines heterogeneous GNN triage with LLM-based hypothesis review. HyGRAIL first uses a GNN to score candidate hypotheses and identify a validation-calibrated ambiguous region, routing only graph-uncertain cases to LLM review. For 
    
[^68]: 隐私洗白：检测隐私政策中的内部矛盾

    Privacy Washing: Detecting Internal Contradictions in Privacy Policies

    [https://arxiv.org/abs/2609.02055](https://arxiv.org/abs/2609.02055)

    本文提出“隐私洗白”概念并构建四阶段检测流水线（语句提取、兼容性过滤与自然语言推理筛查、多模型评审验证、主题分析），在相隔11年的两个隐私政策语料库中发现高度一致的矛盾模式，其中第三方共享类矛盾最为普遍，约12.2%的公司存在至少一个经大语言模型评审团确认的内部矛盾。

    

    隐私政策中可能存在内部矛盾，即政策中作出的承诺会被同一政策其他部分记录的做法所削弱。我们通过一个四阶段流水线将这一现象——“隐私洗白”——进行可操作化处理：语句提取、兼容性过滤与自然语言推理筛查、多模型评判验证以及主题分析，矛盾由三个大语言模型组成的评审团通过多数投票予以确认。将该流水线应用于两个网站隐私政策语料库——2026年收集的123份（OPPT）和2015年收集的115份（OPP-115）——研究发现，跨越11年的时间间隔，相同的类别模式反复出现，其中第三方共享类矛盾在每次主要运行中均占已确认案例的大多数，这与政策撰写中的结构性因素相符，而不一定是有意欺骗。至少有一个经评审团确认的矛盾出现在12.2%的OPPT公司中（15/123；排除遗留政策后为9.8%）……

    arXiv:2609.02055v1 Announce Type: cross  Abstract: Privacy policies may contain internal contradictions in which commitments are undermined by practices documented elsewhere in the same policy. We operationalize this phenomenon, privacy washing, through a four-stage pipeline: statement extraction, compatibility filtering and natural language inference screening, multi-model judge verification, and thematic analysis, with contradictions confirmed by majority vote of a three-model LLM panel. Applied to two corpora of website privacy policies, 123 collected in 2026 (OPPT) and 115 collected in 2015 (OPP-115), the pipeline finds the same category patterns recurring across the 11-year gap, with third-party sharing contradictions the majority of confirmed cases in each primary run, consistent with structural factors in policy composition rather than necessarily intentional deception. At least one panel-confirmed contradiction appears in 12.2% of OPPT companies (15/123; 9.8% excluding legacy p
    
[^69]: 一个用于评估和对齐大语言模型问题澄清能力的三智能体框架

    A Tri-Agent Framework for Evaluating and Aligning Question Clarification Capabilities of Large Language Models

    [https://arxiv.org/abs/2609.02054](https://arxiv.org/abs/2609.02054)

    本文提出一个由问题澄清智能体、应答智能体和评估智能体组成的三智能体框架，用于稳健地评估和对齐大语言模型在交互对话中识别歧义并进行问题澄清的能力。

    

    大语言模型（LLM）日益被部署在交互式系统中，而在这些系统中精确理解用户意图至关重要。此类系统的一项关键能力是有效的问题澄清，尤其是在用户查询含糊不清或信息不充分时。本文提出了一个新颖的三智能体框架，用于稳健地评估大语言模型进行澄清式对话的能力。我们的框架由三个基于大语言模型的不同智能体组成：（1）问题澄清智能体（QCA），即被评估的系统，负责识别歧义并提出澄清性问题；（2）应答智能体（RA），用于模拟人类用户的回应，其中可能包含无关或具有挑战性的回复；（3）评估智能体（EA），即作为裁判的大语言模型（LLM-as-a-judge），基于一整套全面的指标来评估对话质量。我们以供应链领域为例，详细介绍了合成数据生成的方法论。

    arXiv:2609.02054v1 Announce Type: new  Abstract: Large Language Models (LLMs) are increasingly deployed in interactive systems where understanding user intent precisely is paramount. A key capability for such systems is effective question clarification, especially when user queries are ambiguous or underspecified. This paper introduces a novel tri-agent framework for the robust evaluation of an LLM's ability to engage in clarifying dialogue. Our framework comprises three distinct LLM-based agents: (1) a Question Clarifying Agent (QCA), the system under evaluation, tasked with identifying ambiguities and posing clarifying questions; (2) a Respondent Agent (RA), designed to simulate human user responses, potentially including irrelevant or challenging replies; and (3) an Evaluator Agent (EA), an LLM-as-a-judge, which assesses the quality of the dialogue based on a comprehensive set of metrics. We detail a methodology for synthetic data generation in the supply chain domain as an example.
    
[^70]: 语言模型中连续混合坍缩的动力学

    The Dynamics of Continuous Mixture Collapse in Language Models

    [https://arxiv.org/abs/2609.02049](https://arxiv.org/abs/2609.02049)

    该研究揭示了语言模型无法保持连续混合推理状态的深层原因，识别出三种相互独立的失败机制：transformer 架构对混合几何结构的固有扭曲、训练过程对这种扭曲的显著放大，以及 softmax 读出与自回归反馈构成的动力系统导致混合分量被单一主导或坍缩至不可区分。

    

    大语言模型的潜在状态推理方法用连续状态（例如词元嵌入的加权混合）取代离散的中间词元，以保留多种可能的推理方向，而不是只承诺其中一种。然而，预训练语言模型往往无法保持这些混合状态。我们通过理论分析与在多种模型上开展的受控实证研究相结合的方式探究其成因，并识别出三种相互独立且截然不同的失败来源。首先，transformer 架构本身就会扭曲混合的几何结构，而训练过程会显著放大这种效应。此外，即使模型能够完美地以线性方式传输混合，失败仍可能发生：softmax 读出与自回归反馈共同构成一个动力系统，该系统要么不断放大微小的差异直到混合中的某一分量占据主导地位，要么收缩不同的混合直到它们变得无法区分。我们验证了这一理论预测……

    arXiv:2609.02049v1 Announce Type: cross  Abstract: LLMs latent-state reasoning methods replace discrete intermediate tokens with continuous states, such as weighted mixtures of token embeddings, to retain multiple possible reasoning directions rather than committing to one. Yet pretrained language models often fail to preserve these mixtures. We study why through a combination of theoretical analysis and controlled empirical investigations on a variety of models. We identify three independent, distinct sources of failure. First, transformer architectures already distort mixture geometry, and training substantially amplifies this effect. Moreover, the failure can occur even if the model transports mixtures perfectly linearly: the softmax readout and autoregressive feedback form a dynamical system that either amplifies small differences until one component of the mixture dominates or contracts different mixtures until they become indistinguishable. We verify this theoretical prediction e
    
[^71]: 输出格式如何在指令微调中混淆数据质量与能力评估

    How Output Format Confounds Data Quality and Capability in Instruction Tuning

    [https://arxiv.org/abs/2609.02015](https://arxiv.org/abs/2609.02015)

    输出格式同时混淆了指令微调的数据质量评估与模型能力测量——质量信号存在于梯度更新方向而非谱统计量中，且模型能力是相对于训练时的输出格式存储的，更换格式可能让提升40多分的技能几乎消失。

    

    arXiv:2609.02015v1 Announce Type: new 摘要：指令微调数据通过质量指标来评判，微调后的模型通过基准测试来评判，但这两种评判都要经过一个“输出接口”：即答案书写的表面格式。我们通过在12个任务、四种语义等价接口、三个模型家族以及受控损坏条件下的梯度签名分析，证明该接口同时混淆了这两种测量。谱统计量（如有效秩）可被证明对接口旋转保持不变，且经验上对语义损坏“视而不见”，而梯度更新的方向才真正携带质量信号。随接口变化的残差并非噪声：它在所有三个模型家族中都能完美识别出每个单元自己的目标任务。能力本身是相对于训练接口存储的：一项在训练格式下能将准确率提升超过40分的技能，在其他所有格式下可能几乎不可见，而只需修正单个生成预算即可翻转测量结果……（摘要在此处截断）

    arXiv:2609.02015v1 Announce Type: new  Abstract: Instruction-tuning data are judged by quality metrics, and tuned models are judged by benchmarks, but both judgments pass through an output interface: the surface format in which an answer is written. Using gradient signatures across 12 tasks, four semantically equivalent interfaces, three model families, and controlled corruptions, we show that this interface confounds both measurements. Spectral statistics such as effective rank are provably invariant to interface rotation and empirically blind to semantic corruption, while the direction of the update carries the quality signal. The interface-varying residual is not noise: it identifies each unit's own target task perfectly across all three families. Capability itself is stored relative to the training interface: a skill that raises accuracy by more than 40 points under the training format can be nearly invisible under every other, and correcting a single generation budget flips the me
    
[^72]: 训练你所部署的：缩小低秩克隆蒸馏中MLP的可达性差距

    Train What You Deploy: Closing the MLP Reachability Gap in Low-Rank Clone Distillation

    [https://arxiv.org/abs/2609.02006](https://arxiv.org/abs/2609.02006)

    该论文提出“训练你所部署的”原则，让训练直接覆盖完整部署矩阵而非教师诱导的权重切片，在不增加任何推理成本的前提下释放低秩克隆蒸馏中62.5-81.4%被困住的容量，在三个教师模型上取得显著性能提升。

    

    压缩后的学生模型存在两个未必一致的结构：它在推理时部署的权重，以及其训练所能到达的权重族。我们证明，最先进的权重继承蒸馏方法——低秩克隆——部署的是全宽的学生MLP，但训练却被绑定在由教师模型诱导的权重切片上，导致每个已部署矩阵62.5%至81.4%的独立线性自由度无法被训练到——这些容量在推理时付出了代价，却从未被训练。我们的原则只有一句话：训练你所部署的。从相同的LRC热启动出发，我们将训练对象设为整个已部署矩阵，通过两种可合并的实现方式（Dense-LRC和CORE-LRC，二者均可折叠为单一的已部署权重），在不改变部署形状、部署参数量或推理FLOPs的前提下完成训练。这恢复了被搁置的模型容量：在每个教师模型下采用更强的实现方式，相对于匹配预算的朴素LRC基线，在三个教师模型（Llama3.2-3B、Llama3.1-8B、Qw……）上分别取得+2.36/+2.71/+10.45的Avg9提升。

    arXiv:2609.02006v1 Announce Type: cross  Abstract: A compressed student has two shapes that need not agree: the weight it deploys at inference and the weight family its training can reach. We show that a state-of-the-art weight-inheritance distiller, Low-Rank Clone (LRC), deploys a full-width student MLP but ties training to a teacher-induced slice, leaving 62.5-81.4% of each deployed matrix's independent linear degrees of freedom unreachable-paid for at inference, never trainable. Our principle is one line: train what you deploy. From the identical LRC warm start, we make the training object the entire deployed matrix, with no change in deployed shape, deployed parameter count, or inference FLOPs, via two mergeable realizations (Dense-LRC and CORE-LRC) that both collapse to one deployed weight. This recovers stranded capacity: taking the stronger realization per teacher, +2.36/+2.71/+10.45 Avg9 over matched-budget plain-LRC baselines across three teachers (Llama3.2-3B, Llama3.1-8B, Qw
    
[^73]: NS-Copilot：一个由大语言模型驱动的自主神经科学分析智能体系统

    NS-Copilot: An LLM-Driven Agent System for Autonomous Neuroscience Analysis

    [https://arxiv.org/abs/2609.01971](https://arxiv.org/abs/2609.01971)

    NS-Copilot是一个由大语言模型驱动的多智能体系统，能够自主选择和协调神经科学领域的各类预训练模型，支持EEG和细胞外尖峰数据等关键模态，为专业神经科学分析任务提供端到端的自主工作流程。

    

    人工智能正在迅速推动神经科学的发展，然而由于显著的跨学科壁垒，许多实验室未能充分释放其潜力。尽管针对生理数据的预训练神经模型进展迅速，但其异构的架构和特定模态的限制阻碍了系统性的整合、选择与评估。尽管基于大语言模型（LLM）的智能体系统在智能科学应用方面近期取得了进展，现有方法往往仍缺乏有效选择和协调多样化神经科学预训练模型并处理该领域独特数据类型所需的领域专业知识。我们提出了NS-Copilot，一个由大语言模型驱动的神经科学分析多智能体系统，它能够自主支持多样化专业任务的端到端工作流程。该系统统一了特定领域的预训练模型，并支持关键的神经科学模态，包括脑电图（EEG）和细胞外尖峰数据……

    arXiv:2609.01971v1 Announce Type: new  Abstract: AI is rapidly advancing neuroscience, yet many laboratories fail to fully unleash its potential due to significant interdisciplinary barriers. While pre-trained neural models for physiological data are progressing quickly, their heterogeneous architectures and modality-specific constraints hinder systematic integration, selection, and evaluation. Despite recent advances in large language model (LLM)-based agent systems for intelligent scientific applications, existing approaches often still lack the domain expertise required to effectively select and coordinate diverse neuroscience pre-trained models and handle unique data types in this domain. We present NS-Copilot, an LLM-driven multi-agent system for neuroscience analysis that autonomously supports end-to-end workflows for diverse professional tasks. It unifies domain-specific pre-trained models and supports key neuroscience modalities, including EEG and extracellular spike data, thro
    
[^74]: 稀疏读出棱镜（SRP）：用特征而非词元来解释Logit-Lens分数

    Sparse Readout Prism: Explaining Logit-Lens Scores in Features Instead of Tokens

    [https://arxiv.org/abs/2609.01936](https://arxiv.org/abs/2609.01936)

    该论文提出稀疏读出棱镜（SRP），仅利用读出矩阵自身的权重将其分解为稀疏“读出特征”，把logit-lens分数解释为特征贡献之和，从而消除了透镜读数对拟合语料库的依赖（语料库条件性），并支持跨词元、上下文、层与透镜的比较。

    

    语言模型对下一个词元的预测是跨层逐步形成的，而“透镜”方法通过将中间隐藏状态解码为词元来追踪这一过程。但透镜的读数同时反映了隐藏状态以及用于解码的读出矩阵。许多透镜是在语料库上拟合的，我们证明：仅在拟合语料库上不同的两个透镜，会对相同的隐藏状态报告不同的词元。我们将这种依赖性称为“语料库条件性”。为了独立于拟合语料库来考察读出结构，我们提出了稀疏读出棱镜（Sparse Readout Prism, SRP），它仅使用读出矩阵自身的权重对其进行分解，并将任意词元的logit或logit差表示为稀疏读出特征贡献之和。这使读出特征成为透镜读数的一种全新分析单元，揭示出词元身份可能掩盖的结构，并支持跨词元、上下文、层与透镜之间的比较。

    arXiv:2609.01936v1 Announce Type: cross  Abstract: A language model's prediction of its next token develops across layers, and lens methods track this process by decoding intermediate hidden states into tokens. But a lens reading reflects both the hidden state and the readout (the unembedding matrix) used to decode it. Many lenses are fit on a corpus, and we show that two lenses differing only in their fitting corpus can report different tokens for the same hidden states. We call this dependence corpus conditionality. To examine readout structure independently of the fitting corpus, we introduce Sparse Readout Prism (SRP), which decomposes the readout using only its weights and expresses any token logit or logit difference as a sum of contributions from sparse readout features. This reveals readout features as a new unit of analysis for lens readings, exposing structure that token identities can obscure and enabling comparisons across tokens, contexts, layers, and lenses. Replacing the
    
[^75]: CRISP：悬崖感知的输入自适应稀疏预填充与基于结构质量驱动的路由

    CRISP: Cliff-awaRe Input-adaptive Sparse Prefilling with Structural-Mass-Motivated Routing

    [https://arxiv.org/abs/2609.01925](https://arxiv.org/abs/2609.01925)

    该论文提出CRISP方法，用直接从代理注意力图结构中读取路由决策的结构代理C_struct替代JSD路由，解决了动态稀疏注意力路由中的两个结构性挑战，实现了长上下文LLM推理的高效输入自适应稀疏预填充。

    

    长上下文大语言模型（LLM）推理中的注意力预填充阶段计算复杂度呈二次方增长，使自注意力成为严重的计算瓶颈。传统的稀疏注意力方法通过固定模式或离线分析来缓解这一问题，但缺乏适应输入相关注意力结构的灵活性。近期的动态方法通过实时将注意力头路由到稀疏模式来解决这一问题，但其依赖于带有额外开销的间接路由代理，且其预算分配机制忽略了softmax之后的质量层级。我们提出了CRISP（Cliff-awaRe Input-adaptive Sparse Prefilling，悬崖感知的输入自适应稀疏预填充），该方法识别并解决了这一动态路由范式中的两个结构性挑战。首先，我们证明了路由决策可以直接从代理注意力图的结构中读取。我们用C_struct取代了Jensen-Shannon散度（JSD）路由，C_struct是一种结构代理，用于测量垂直-斜线兼容位置处的注意力质量，并能重现……（摘要在此处截断）

    arXiv:2609.01925v1 Announce Type: cross  Abstract: The attention prefilling phase of long-context LLM inference scales quadratically, making self-attention a severe computational bottleneck. Traditional sparse attention methods mitigate this through fixed patterns or offline profiling, but lack the flexibility to adapt to input-dependent attention structure. Recent dynamic methods address this by routing heads to sparse patterns in real-time, but rely on indirect routing proxies with overhead and budget allocation mechanisms that overlook the post-softmax mass hierarchy. We present CRISP (Cliff-awaRe Input-adaptive Sparse Prefilling), which identifies and addresses two structural challenges in this dynamic routing paradigm. First, we show that the routing decision can be read directly off the structure of the proxy attention map. We replace the Jensen-Shannon Divergence (JSD) routing with C_struct, a structural proxy that measures mass at Vertical-Slash compatible positions and reprodu
    
[^76]: 面向物理约束点对点能源市场中能源贫困公平的、有据可依且计算高效的LLM政策智能体

    Grounded, Compute-Efficient LLM Policy Agents for Energy-Poverty Equity in Physically-Constrained Peer-to-Peer Energy Markets

    [https://arxiv.org/abs/2609.01918](https://arxiv.org/abs/2609.01918)

    该论文提出EqGrid闭环仿真框架，以低频开源LLM政策智能体设定价格、碳限额与定向补贴，配合高频多智能体强化学习交易者在受物理电网约束的点对点能源市场中出清交易，并通过真实智能电表数据校验的家庭画像和形式化能源贫困公平指标（能源负担、基尼系数、LIHC）来衡量AI的社会影响，从而避免了对碳密集型云LLM的依赖。

    

    能源贫困在“自然语言处理促进社会公益”领域几乎处于空白状态，而现有少量工作要么是静态检索/问答系统，要么依赖于碳密集型的云端大语言模型——对于人道主义场景而言，这形成了一种自相矛盾的“计算悖论”。我们提出了EqGrid，一个闭环仿真系统：其中低频率、开源权重的LLM政策智能体在一个由经验数据支撑的家庭画像社区上设定价格与碳限额以及定向补贴，同时高频多智能体强化学习交易者在一个受物理配电网约束（采用带动态运行包络的IEEE-33节点系统）的连续双边拍卖中完成市场出清。我们的贡献有三点，直接回应了如何衡量人工智能的社会影响：(i) 有据可依的家庭画像（与社会人口统计特征区域匹配），其负荷曲线的形状与水平真实性均经过真实智能电表数据校验；(ii) 形式化的能源贫困公平性指标（能源负担、能源负担基尼系数、LIHC），显示干预措施能够重新……（原文摘要在此处截断）

    arXiv:2609.01918v1 Announce Type: new  Abstract: Energy poverty is nearly absent from NLP-for-social-good, and the little existing work is either static retrieval/QA or relies on carbon-intensive cloud LLMs, a self-defeating "computational irony" for a humanitarian setting. We present EqGrid, a closed-loop simulation in which a low-frequency, open-weight LLM policy agent sets price and carbon bounds and targeted subsidies over a community of empirically-grounded household personas, while high-frequency multi-agent RL traders clear a continuous double auction constrained by a physical distribution grid (IEEE-33-bus with Dynamic Operating Envelopes). Our contribution is threefold and directly addresses how to measure the social impact of AI: (i) grounded personas (region-matched socio-demographics) whose load curves are checked for shape and level realism against real smart-meter data; (ii) formal energy-poverty equity metrics (Energy Burden, Gini of EB, LIHC) showing the intervention re
    
[^77]: 空间上准确，时间上不可靠：大语言模型如何表征国家文化变迁

    Accurate in space, unreliable in time: how LLMs represent national cultural change

    [https://arxiv.org/abs/2609.01902](https://arxiv.org/abs/2609.01902)

    该研究基于二十余年的世界价值观调查数据发现，大语言模型虽能在空间上将各国较准确地定位于文化地图上的当前位置，却无法可靠地表征各国文化随时间演变的变迁轨迹。

    

    对文化一致性的评估已成为大语言模型（LLM）开发与改进的重要组成部分。然而，大多数评估将文化视为单一的静态快照，仅考察模型是否在当前时点上准确表征了一个社会。文化心理学研究表明，文化价值观会随时间以不同的速度和方向发生变化。因此，一个“具有文化意识”的模型不仅应捕捉一种文化当下的位置，还应捕捉其随时间的演变过程。我们利用超过二十年的世界价值观调查数据，考察了这一被忽视的文化意识维度。我们在英格尔哈特-韦尔泽尔文化地图上，将40个国家的文化变迁轨迹与四个最先进的（SOTA）大语言模型所生成的轨迹进行比较。我们的研究结果表明，尽管模型通常能将各国定位在其最近一次调查所得的位置附近，但这些表征往往……（原文摘要在此处截断，未完整提供）

    arXiv:2609.01902v1 Announce Type: cross  Abstract: Assessments of cultural alignment have become an important part of the development and improvement of large language models (LLMs). However, the majority of the evaluations treat culture as a single snapshot, investigating only whether a model represents a society accurately at the current time. Research in cultural psychology shows that cultural values change at different rates and directions over time. Therefore, a "culturally aware" model should capture not only where a culture is today but also how it has changed over time. We examine this missing dimension of cultural awareness using more than two decades of the World Values Survey data. We compare the cultural trajectories of 40 countries with the trajectories produced by four state-of-the-art (SOTA) LLMs on the Inglehart-Welzel cultural map. Our findings show that while models generally place countries close to their most recent surveyed positions, these representations tend to 
    
[^78]: GAPS：用于条件激活转向的维度级门控

    GAPS: Dimension-Level Gates for Conditional Activation Steering

    [https://arxiv.org/abs/2609.01878](https://arxiv.org/abs/2609.01878)

    GAPS提出维度级条件化的激活转向方法，通过两个无需训练的门控——静态可分离性门控（基于AUROC筛选携带概念信息的神经元）和动态后验门控（基于高斯模型判断激活状态），精确决定对哪些神经元进行干预，从而更细粒度地抑制语言模型的不良行为。

    

    激活转向通过在生成过程中向隐藏状态添加转向向量来抑制语言模型的不良行为。最近的条件方法（如CAST和DSAS）通过决定何时干预来改善行为-能力的权衡，但一旦激活，它们就会将完整的稠密向量应用于所有隐藏维度，而不考虑某个神经元是否携带概念信息或已处于期望状态。我们引入了维度级条件化作为选择性的一个互补维度，它还能决定对哪些神经元进行干预。我们的方法GAPS（基于后验和可分离性的门控激活转向）结合了两个无需训练的门控：一个是静态可分离性门控，它将转向限制在具有统计可靠概念信息的神经元上（通过AUROC衡量）；另一个是动态后验门控，仅当神经元的当前激活在高斯模型下更能被不良概念解释时才对其进行转向。

    arXiv:2609.01878v1 Announce Type: new  Abstract: Activation steering suppresses undesired behaviors in language models by adding a steering vector to the hidden state during generation. Recent conditional methods such as CAST and DSAS improve the behavior-capability trade-off by deciding when to intervene, but once active, they apply the full dense vector to all hidden dimensions, regardless of whether a neuron carries concept information or already lies in the desired regime. We introduce dimension-level conditioning as a complementary axis of selectivity that also decides which neurons to intervene on. Our method, GAPS (Gated Activation steering via Posterior and Separability), combines two training-free gates: a static separability gate that restricts steering to neurons with statistically reliable concept information (via AUROC), and a dynamic posterior gate that steers a neuron only when its current activation is better explained by the undesired concept under a Gaussian model. Th
    
[^79]: 在溯因推理中，人类与推理模型的思考努力程度相互一致

    Thinking effort aligns between humans and reasoning models in abductive reasoning

    [https://arxiv.org/abs/2609.01867](https://arxiv.org/abs/2609.01867)

    该论文通过溯因推理任务（其难度无法通过形式结构捷径伪装）发现，大型推理模型与人类在推理努力程度（思考成本）上表现出行为对齐。

    

    认知建模中的一个重大问题涉及大型语言模型与人类在语言和非语言任务上的行为一致性。与标准大型语言模型不同，大型推理模型（LRMs）通过基于可验证奖励的强化学习进行优化，鼓励其对推理任务给出正确解答，而非产生符合偏好的响应。最近的研究（de Varda et al., 2025）通过在一系列推理任务上比较人类反应时间与模型推理轨迹，考察了人类与大型推理模型的思考成本。我们通过转向溯因推理来隔离研究这种对齐：与演绎任务不同，溯因推理的难度无法从形式结构中推断，且不提供任何模型可以在没有真正搜索的情况下用以伪装努力的捷径，从而为“共享努力”这一实证主张提供了更坚实的基础。我们进一步发现了大型推理模型与人类推理努力之间对齐的证据，同时有证据表明（摘要在此处被截断）

    arXiv:2609.01867v1 Announce Type: cross  Abstract: A major question in cognitive modeling concerns the behavioral alignment between large language models and humans across linguistic and non-linguistic tasks. Unlike standard LLMs, large reasoning models (LRMs) are optimized with reinforcement learning from verifiable rewards, encouraging correct solutions to reasoning tasks rather than preference-aligned responses. Recent work (de Varda et al., 2025) investigates the cost of thinking in humans and LRMs by comparing human reaction times with model reasoning traces across a range of reasoning tasks. We isolate this alignment by turning to abductive reasoning: unlike deductive tasks, its difficulty cannot be inferred from formal structure and offers no shortcuts a model could exploit to mimic effort without genuine search, providing firmer ground for empirical claims of shared effort. We find further evidence of alignment between LRM and human reasoning effort, as well as evidence that mo
    
[^80]: ExecRetrieval：衡量代码嵌入检索中的功能正确性差距

    ExecRetrieval: Measuring the Functional-Correctness Gap in Code-Embedding Retrieval

    [https://arxiv.org/abs/2609.01865](https://arxiv.org/abs/2609.01865)

    提出 ExecRetrieval 基准（939 个 Python 任务），通过在搜索池中植入与规范实现几乎相同、但经执行验证的有缺陷变体，首次衡量了代码嵌入检索在区分功能正确代码与错误代码上的差距。

    

    基于嵌入的代码检索是编码智能体和检索增强代码生成的核心组件，在这些场景中，检索到功能正确的代码比检索到词汇上相似的代码更为重要。现有的代码检索基准并未在搜索池中植入受控的、经执行验证的、针对每个查询规范实现的单次编辑变体，因此“嵌入模型能否在检索场景中从功能上区分正确代码与近似克隆但不正确的代码”这一问题仍未得到解答。解决这一问题需要一个搜索池本身就包含相关反事实样本的基准——即与每个规范实现几乎完全相同、且经过执行验证的有缺陷变体——从而可以直接检验检索器的排序结果是否具备功能区分能力，而不仅仅是主题或身份上的重合。我们提出了 ExecRetrieval，包含 939 个 Python 任务，每个任务都配有一个经执行验证的规范实现，以及最多四个经执行验证的……

    arXiv:2609.01865v1 Announce Type: cross  Abstract: Embedding-based code retrieval is a core component of coding agents and retrieval-augmented code generation, where retrieving correct code matters more than retrieving lexically similar code. Existing code-retrieval benchmarks do not plant controlled, execution-verified single-edit variants of each query's canonical implementation in the search pool, leaving the question of whether embeddings can functionally discriminate correct from near-clone-but-incorrect code unanswered in a retrieval setting. Resolving this requires a benchmark whose search pool itself contains the relevant counterfactuals -- execution-verified buggy variants near-identical to each canonical -- so that a retriever's rank ordering can be directly tested for functional discrimination rather than topical or identity overlap. We introduce ExecRetrieval, 939 Python tasks each paired with one execution-verified canonical implementation and up to four execution-verified
    
[^81]: 记忆信任差距：持久记忆智能体中依赖模型能力的失效

    The Memory Trust Gap: Capability-Dependent Failures in Persistent-Memory Agents

    [https://arxiv.org/abs/2609.01852](https://arxiv.org/abs/2609.01852)

    该论文提出并量化了“记忆信任差距”现象：持久记忆智能体会过度信任（而非混淆于）过期的存储事实并覆盖权威证据，且这种失效受模型能力门控——规模越大的模型在过期记忆被伪装成最新信息时崩溃反而越严重。

    

    持久记忆为个性化智能体提供支持，但一条过期的存储事实可能在毫无警告的情况下覆盖当前的权威证据。我们研究随着模型能力的变化，这种危害何时开始显现。我们评估了一个冻结的、闭集的、按动作评分的基准测试，该基准包含两个测试套件，分别代表“无记忆”的两种不同含义（一个是“收益”套件，在没有存储事实的情况下无法解决；另一个是“安全”套件，其中权威工具始终持有正确值），并在同一系列不同规模的模型（Qwen3 0.6/1.7/4/8B）上进行测试。我们发现“记忆信任差距”反映的是过度信任而非混淆。在收益套件中，所有规模的模型在0.92-1.00的情况下都会使用过期值作答。在安全套件中，陷阱条件下低于无记忆基线的危害（Δ_mem）受模型能力门控，一旦过期笔记被伪装成最新内容，规模更大的模型反而崩溃得最严重。在一个2×2×2×2的因子实验中，探究了哪个特征触发了过度信任……（原文摘要在此处截断）

    arXiv:2609.01852v1 Announce Type: new  Abstract: Persistent memory supports personalized agents, but a stale stored fact can override current authoritative evidence without warning. We study when this harm begins as model capability changes. We evaluate a frozen, closed-set, action-scored benchmark with 2 suites that represent 2 different meanings of "no memory" (a Benefit suite, unsolvable without the stored fact, and a Safety suite, in which an authoritative tool always holds the correct value), on a same-family model-size series (Qwen3 0.6/1.7/4/8B). The Memory Trust Gap reflects over-trust rather than confusion. In the Benefit suite, models answer with the stale value 0.92-1.00 of the time at every scale. In the Safety suite, harm below the no-memory baseline under the trap conditions ($\Delta_{\mathrm{mem}}$) is capability-gated, with the larger models collapsing most once a stale note is made to look current. In a $2\times2\times2\times2$ factorial, which feature triggers over-tr
    
[^82]: 引用或拒绝：面向STEM课程视频的严格课程约束聊天机器人

    Cite or Decline: A Strict Course-Grounded Chatbot for STEM Lecture Videos

    [https://arxiv.org/abs/2609.01846](https://arxiv.org/abs/2609.01846)

    本文提出了VideoPoints平台一学期的实际部署，其检索增强聊天机器人严格基于课程材料回答问题并提供带时间戳的引用，在无证据时选择拒答，833条学生消息中实现了零课程边界越界，证明了严格课程约束设计的可行性。

    

    录制的课程视频通常配备搜索和摘要功能，是标准的学习资源。然而，学生难以针对具体课程提问，也无法根据教师的授课内容验证答案。我们报告了VideoPoints平台为期一个学期的部署情况，该平台配备了一个检索增强聊天机器人，能够基于课程讲座材料回答问题并返回带时间戳的引用。该聊天机器人仅从当前课程中检索内容，利用章节摘要来指导转录文本的排序，并返回可点击的带时间戳引用。学生用它进行快速查询和考试复习。在833条消息中，70.5%的消息包含引用，没有任何一条跨越课程边界，当没有讲座证据匹配时，聊天机器人通常选择拒绝回答而非强行作答。在用户看来，引用功能是最持续有用的特性，而练习题生成则是最强烈的未满足需求。我们还在真实环境中对该设计进行了评估。

    arXiv:2609.01846v1 Announce Type: new  Abstract: Recorded lecture videos, often enhanced with search and summarization features, are a standard study resource. However, students cannot easily ask course specific questions or verify answers against an instructor's lecture. We report a semester-long deployment of VideoPoints platform with a retrieval-augmented chatbot that answers from course lecture materials and returns timestamped citations. The chatbot retrieves only from the active course, uses chapter summaries to guide transcript ranking, and returns clickable timestamped citations. Students used it for quick lookups and exam review. Across 833 messages, 70.5% included citations, none crossed a course boundary, and when no lecture evidence matched, the chatbot usually declined rather than answering. Among the users, citations were the most consistently useful feature, while practice-question generation was the strongest unmet request. We also evaluated the design on the real-world
    
[^83]: 面向句子级抑郁症状识别的候选生成与定义引导验证

    Candidate Generation and Definition-Guided Verification for Sentence-Level Depression Symptom Recognition

    [https://arxiv.org/abs/2609.01833](https://arxiv.org/abs/2609.01833)

    提出了一种两阶段框架，先由对比学习微调的句子编码器生成抑郁症状候选，再由微调的语言模型依据诊断定义验证候选症状是否出现，在句子级抑郁症状识别任务上取得了所有方法中最佳的准确率和F1分数。

    

    句子级抑郁症状识别具有挑战性，因为相似的表达在症状相关性上可能存在差异，且语言模型的推理缺乏对诊断定义的充分依据。本研究提出了一个两阶段框架，将症状候选生成与基于定义的验证相分离。一个经过对比学习微调的句子编码器为每个句子生成症状候选，随后一个经过微调的语言模型利用该句子、其上下文以及候选特定的诊断定义来验证该候选症状是否存在，并在作答前依据该定义核查自身的判断。在与编码器基线、基于推理的基线、医学大模型和通用大语言模型基线以及匹配的单阶段监督分类器的对比评估中，所提出的流水线在所有方法中取得了最佳的准确率和F1分数，其生成的判断依据与专家撰写的标注相吻合。一项初步的临床审计表明模式……（摘要在此处被截断）

    arXiv:2609.01833v1 Announce Type: new  Abstract: Sentence-level recognition of depression symptoms is challenging because similar expressions can differ in symptom relevance, and language-model inference is insufficiently grounded in diagnostic definitions. This study proposes a two-stage framework separating symptom-candidate generation from definition-grounded verification. A contrastively fine-tuned sentence encoder generates a symptom candidate per sentence, and a fine-tuned language model verifies whether the candidate is present or absent using the sentence, its context, and a candidate-specific diagnostic definition, checking its judgment against that definition before answering. Evaluated against encoder, inference-based, medical, and general LLM baselines and a matched single-stage supervised classifier, the proposed pipeline attains the best accuracy and F1 scores of all methods, with rationales matching expert-authored annotations. A preliminary clinical audit indicates mode
    
[^84]: 大语言模型中用于抑郁症的可解释症状向量

    Interpretable Symptom Vectors for Depression in a Large Language Model

    [https://arxiv.org/abs/2609.01832](https://arxiv.org/abs/2609.01832)

    该研究通过机制可解释性技术发现大语言模型内部在第21层对抑郁症状产生几何分离，并构建“症状向量”将文本投影后得到各症状系数，其能保留临床医生标注的严重程度排序，从而增强LLM在抑郁症评估中的临床可信度。

    

    抑郁症患者呈现出多样化的症状特征，然而临床实践中通常将这种差异简化为单一的严重程度评分。大语言模型（LLMs）有潜力从患者的言语中捕捉各种症状及其严重程度。然而，抑郁症状在LLM内部如何表示仍然知之甚少，这限制了临床信任度。为了检验模型内部激活是否与临床医生的判断相符，我们使用机制可解释性技术分析了Gemma-3-27B-PT的残差流。通过记录来自经过验证的临床量表的多种症状描述的激活，我们发现在多个距离度量下，症状组在第21层的几何分离最为显著。随后，我们使用语义投影方法，将留出的自然语言文本投影到由这些临床量表构建的症状向量（Symptom Vectors）上。所得到的每个症状的系数保留了临床医生标注的严重程度排序。

    arXiv:2609.01832v1 Announce Type: cross  Abstract: Patients with depression present with diverse symptom profiles, yet clinical practice routinely reduces this variation to a single severity score. Large language models (LLMs) can potentially capture various symptoms and their severity from patient speech. However, how depressive symptoms are represented inside LLMs remains poorly understood, limiting clinical trust. To examine whether internal model activations match clinician judgment, we analyzed the residual stream of Gemma-3-27B-PT using mechanistic interpretability techniques. Recording activations across symptom descriptions drawn from validated clinical instruments, we found that symptom groups geometrically separated the most at layer 21 across multiple distance metrics. Using Semantic Projection, we then projected held-out naturalistic text onto Symptom Vectors constructed from these instruments. The resulting per-symptom coefficients preserved clinician-annotated rank orderi
    
[^85]: AVERT：面向口语对话状态跟踪的音频验证裁决

    AVERT: Audio-Verified Adjudication for Spoken Dialogue State Tracking

    [https://arxiv.org/abs/2609.01828](https://arxiv.org/abs/2609.01828)

    AVERT通过结合跨轮一致性与音频条件验证器对候选值打分，并利用投票、添加、交换三种算子纠正口语对话状态跟踪中的三类可恢复错误，在SpokenWOZ上无需重训练即可将JGA提升至40.13。

    

    口语对话状态跟踪需要从语音中恢复槽位-值对，其中ASR错误集中在实体值上并在多轮对话中持续存在，这使其既是生成问题也是编辑问题。强大的逐轮文本编辑器可以纠正其中大部分错误，但由于仅基于转录文本操作，仍会留下三种可恢复的错误：跨轮预测不一致的值、被遗漏的槽位，以及音频不支持的值。我们提出AVERT，它通过结合跨轮一致性与训练好的音频条件验证器对每个候选值进行打分，并使用三种算子——投票、添加和交换——来解决这三种错误类型，每个算子仅应用于其对应错误常见的槽位。在SpokenWOZ数据集上，基础语音LLM达到33.04 JGA，文本编辑器达到38.34，而AVERT达到40.13，且均无需重新训练。这一结果与消耗完整口语历史的1B端到端系统（39.32）处于同一水平，尽管AVERT使用的是两个1B解码器而非一个。

    arXiv:2609.01828v1 Announce Type: new  Abstract: Spoken dialogue state tracking recovers slot-value pairs from speech, where ASR errors concentrate in entity values and persist across turns, making it both a generation and an editing problem. A strong per-turn text editor corrects much of this but, operating on the transcript alone, leaves three recoverable errors: a value predicted inconsistently across turns, an omitted slot, and a value the audio does not support. We present AVERT, which scores each candidate value by combining cross-turn agreement with a trained audio-conditioned verifier and resolves the three error types with three operators, vote, add, and swap, each restricted to the slots where its error is common. On SpokenWOZ, a base speech-LLM reaches 33.04 JGA, a text editor 38.34, and AVERT 40.13, without retraining either. This is in the range of a 1B end-to-end system that consumes the full spoken history (39.32), though AVERT uses two 1B decoders rather than one. The a
    
[^86]: TalkFa：一个用于波斯语对话生成与理解的统一基准

    TalkFa: A Unified Benchmark for Farsi Dialogue Generation and Understanding

    [https://arxiv.org/abs/2609.01810](https://arxiv.org/abs/2609.01810)

    该论文提出了TalkFa——首个针对波斯语的统一对话生成与理解基准，由三个经母语者严格人工审核的数据集构成，并通过实验证明LoRA微调只需少量训练数据即可显著提升波斯语对话生成与理解性能。

    

    波斯语有超过1.2亿人使用，但缺乏一个全面的对话生成与理解基准。我们推出了TalkFa，这是一个包含三个互补数据集的统一基准：(1) WIKI-FADIAL，包含4.2K个基于维基百科的对话，用于知识增强的生成任务；(2) DAILYDIALOG-FA，包含6.6K个标注了对话行为和情感的对话；(3) PLAYDIAL-FA，包含2.1K个带有情感标签的戏剧对话。虽然大语言模型辅助了数据构建，但每个对话都经过波斯语母语使用者的多阶段审查和修订，最终只发布经人工批准的对话。在六个LLAMA和MISTRAL模型上的实验表明，LoRA能显著提升对话生成效果，且仅需25-50%的训练数据即可恢复超过90%的最终性能收益。在分类任务方面，FABERT在对话行为识别上表现最佳，而LORA-MISTRAL-7B在情感识别上表现最好。

    arXiv:2609.01810v1 Announce Type: new  Abstract: Farsi, spoken by more than 120 million people, lacks a comprehensive benchmark for dialogue generation and understanding. We introduce TALKFA, a unified benchmark comprising three complementary datasets: (1) WIKI-FADIAL, 4.2K Wikipedia-grounded dialogues for knowledge-grounded generation; (2) DAILYDIALOG-FA, 6.6K dialogues annotated for dialogue acts and emotions; and (3) PLAYDIAL-FA, 2.1K theatrical dialogues with sentiment labels. While LLMs assist data construction, every dialogue undergoes multi-stage review and revision by native Farsi speakers, and only the final human-approved dialogues are released. Experiments with six LLAMA and MISTRAL models show that LoRA substantially improves dialogue generation while requiring only 25-50% of the training data to recover over 90% of the final performance gains. Across classification tasks, FABERT achieves the best dialogue-act performance, LORA-MISTRAL-7B performs best on emotion recognitio
    
[^87]: 提示词变化如何影响设备端大语言模型的能耗？

    How Do Prompt Variations Affect Energy Consumption in On-Device LLMs?

    [https://arxiv.org/abs/2609.01798](https://arxiv.org/abs/2609.01798)

    本研究通过大规模实证分析首次揭示，提示词的认知负荷主要影响设备端LLM推理中每个token的能耗成本，而措辞模式主要通过token使用量影响总能耗，为节能导向的模型感知提示词设计提供了依据。

    

    大语言模型（LLM）越来越多地部署在移动设备上，使得能效成为关键的部署约束，然而提示词设计对能耗的影响仍未得到充分研究。本文旨在理解两个提示词属性——认知负荷和措辞模式——如何塑造设备端LLM推理的能耗行为。我们开展了一项涵盖提示词属性、数据集、模型和设备的广泛实证研究，并通过阶段级剖析将预填充和解码阶段的能耗分开分析。我们发现认知负荷主要影响每个token的能耗成本，而措辞模式主要通过token使用量来影响能耗。我们的能耗-质量分析进一步表明，提示词设计在不同模型上以不同方式重塑了可达到的前沿边界，凸显了在节能的设备端LLM推理中需要模型感知的提示词设计。代码、数据集和脚本可在 https://amai-gsu.github.io/PromptProperty/ 获取。

    arXiv:2609.01798v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed on mobile devices, making energy efficiency a key deployment constraint, yet the energy impact of prompt design remains underexplored. This paper aims to understand how two prompt properties, cognitive load and phrasing pattern, shape the energy behavior of on-device LLM inference. We conduct a broad empirical study covering prompt properties, datasets, models, and devices, with phase-level profiling that separates prefill and decode energy. We find that cognitive load primarily affects the energy cost per token, while phrasing pattern affects energy largely through token usage. Our energy-quality analysis further shows that prompt design reshapes the attainable frontier differently across models, highlighting the need for model-aware prompt design in energy-efficient on-device LLM inference. Code, datasets, and scripts are available at https://amai-gsu.github.io/PromptProperty/.
    
[^88]: 区分语言模型在避免过度泛化中的统计先占与固着效应

    Disentangling Statistical Preemption from Entrenchment in Language Models' Avoidance of Overgeneralization

    [https://arxiv.org/abs/2609.01794](https://arxiv.org/abs/2609.01794)

    本研究通过在语言模型上进行受控养育实验并系统移除先占性与非先占性证据，首次区分了统计先占与固着两种假说，发现语言模型避免过度泛化时并不依赖动词层面的先占效应，而是将竞争结构视为间接正面证据而非负面证据。

    

    学习者如何在没有明确负面证据的情况下避免诸如"Tom laughed me"这样的过度泛化表达？构式主义者提出了两种解释针对过度泛化的间接负面证据的假说：先占假说（preemption，即偏向于接触近义构式——例如"she made him laugh"）与固着假说（entrenchment，即接触动词的所有语法用法，包括像"He laughed"这样的例子）。我们通过在以儿童-照护者对话训练的语言模型上进行受控养育实验来区分这两种假说，在实验中我们系统地移除先占性证据与非先占性证据。我们发现，虽然语言模型能够避免过度泛化，但它们并未在动词特定层面表现出先占效应，而是表现出微弱但非零的抽象先占证据。结合对语言模型训练动态的分析结果，我们发现语言模型将竞争结构视为间接的正面证据——而非负面证据——……（原文在此处截断）

    arXiv:2609.01794v1 Announce Type: new  Abstract: How do learners avoid overgeneralizations such as Tom laughed me without explicit negative evidence? Constructionists have posited two proposals that describe indirect negative evidence against overgeneralizations: preemption (which privileges exposure to near-synonymous construction---e.g., she made him laugh) vs. entrenchment (all exposures to a verb's grammatical usages, including cases like He laughed). We disentangle these hypotheses by running controlled rearing experiments on LMs trained on child-caregiver conversations, where we systematically remove preemptive vs. non-preemptive evidence. We find that while LMs avoid overgeneralizations, they do not show preemption at a verb-specific level, instead showing weak but non-zero evidence of abstract preemption. Combined with results from analyzing the LMs' training dynamics, we find that LMs treat competing structures as indirect positive---as opposed to negative---evidence in the ve
    
[^89]: VakyArth：评估大语言模型在印度语系语言中的语用能力

    VakyArth: Evaluating Pragmatic Competence in LLMs across Indic Languages

    [https://arxiv.org/abs/2609.01788](https://arxiv.org/abs/2609.01788)

    该论文提出了首个针对印度语系语言（印地语、旁遮普语、泰米尔语、马拉雅拉姆语）的语用能力诊断基准VakyArth，通过母语者编写的多任务评估揭示了多语言大模型在印度语言文化相关的语用推理上存在系统性失败。

    

    现实世界的交流往往需要语用推理，即解读通过上下文和文化惯例所隐含的意义，而非字面陈述的意义。现有的语用能力评估大多局限于英语和高资源语言，尽管印度语系语言具有丰富的语言和文化多样性，但在这一领域仍未被探索。我们提出了VakyArth，这是首个针对印度语系语言的语用能力基准，设计为诊断性评估，涵盖印地语、旁遮普语、泰米尔语和马拉雅拉姆语。VakyArth通过多项选择题、自然语言推理和翻译任务，从五个语言现象维度评估模型：指示语、言语行为、含义推断、社会语用和连贯性，所有题目均由母语者撰写。在不同家族和规模的多语言大语言模型（LLM）上，我们发现模型在基于印度语言和文化惯例的语用含义理解上存在一致的失败。我们的分析表明存在系统性的……

    arXiv:2609.01788v1 Announce Type: cross  Abstract: Real-world communication often requires pragmatic reasoning: interpreting meanings implied through context and cultural convention rather than stated literally. Existing pragmatic evaluation remains largely limited to English and high-resource languages, leaving Indic languages unexplored despite their linguistic and cultural diversity. We introduce VakyArth, the first pragmatic benchmark for Indic languages, designed as a diagnostic evaluation covering Hindi, Punjabi, Tamil, and Malayalam. VakyArth evaluates models across five phenomena: deixis, speech acts, implicature, social pragmatics, and coherence; through multiple-choice questions, natural language inference, and translation, with all items authored by native speakers. Across multilingual large language models (LLMs) of varying families and sizes, we find consistent failures on pragmatic meanings rooted in Indic linguistic and cultural conventions. Our analysis shows systematic
    
[^90]: MemeCULT-1K：多模态模型对南亚文化语境与幽默理解的基准测试

    MemeCULT-1K: Benchmarking South Asian Cultural Context and Humor Understanding of Multimodal Models

    [https://arxiv.org/abs/2609.01772](https://arxiv.org/abs/2609.01772)

    提出了包含 1000 个南亚多语言模因的基准数据集 MemeCULT-1K，并证明为视觉语言模型提供少量文化背景信息即可显著且一致地提升其对模因的理解能力。

    

    模因（meme）理解不仅仅是对视觉内容或字面文本的识别，它还需要隐性的文化知识和语用推理能力，而大多数视觉-语言模型仍然缺乏这些能力。我们推出了 MemeCULT-1K，这是一个多语言基准数据集，包含 1,000 个孟加拉语、英语和印地语的南亚模因，每个模因都配有文化背景注释和三个人工撰写的解释，此外还附带一个包含 54 个孟加拉语地区方言模因的补充数据集。我们在两种设置下评估了十三个主流视觉语言模型（VLM）：仅模因输入和情境感知模式。提供最少的文化背景信息在所有模型和语言上都带来了一致的性能提升：平均 SBERT 相似度从 44.6 提高到 56.4（+11.8），BLEURT 从 37.3 提高到 42.3（+5.0），LLM-as-a-Judge 评分从 5 分制中的 2.57 提高到 3.43（+0.86）。细粒度的错误分析表明，闭源模型的主要失败点在于实体和引用的误识别，而开源模型……

    arXiv:2609.01772v1 Announce Type: new  Abstract: Meme understanding goes beyond recognizing visual content or literal text; it requires implicit cultural knowledge and pragmatic inference that most vision-language models still lack. We introduce MemeCULT-1K, a multilingual benchmark of 1,000 South Asian memes in Bengali, English, and Hindi, where each meme is paired with a cultural context note and three human-written explanations, along with a supplementary set of 54 Bengali regional dialect memes. We evaluate thirteen popular Vision Language Models (VLMs) under two settings: meme-only and context-aware. Providing minimal cultural context yields consistent gains across all models and languages: mean SBERT similarity improves from 44.6 to 56.4 (+11.8), BLEURT from 37.3 to 42.3 (+5.0), and LLM-as-a-Judge scores from 2.57 to 3.43 out of 5 (+0.86). Fine-grained error analysis reveals that closed-source models fail mainly on entity and reference misidentification, while open-source models 
    
[^91]: 机器何时能信任法律条文？机器提取法律逻辑的存续证书

    When Can a Machine Trust a Statute? A Survival Certificate for Machine-Extracted Legal Logic

    [https://arxiv.org/abs/2609.01741](https://arxiv.org/abs/2609.01741)

    该论文提出一种“被动存续证书”方法，通过量化不同法条抽取器之间的分歧、在1,000次蒙特卡洛试验中重放噪声并以Wilson 95%置信下界作为门槛，来认证哪些机器提取的Duquenne-Guigues形式蕴含能够在解析噪声下可靠存续。

    

    法律条文在人们阅读之前越来越多地被机器解析，而不同解析器之间存在分歧：在密苏里州的法律条文上，两个独立编写的抽取器在数值阈值存在性判断上的分歧达到了0.43的假阴性率。我们探究什么样的形式逻辑能够在此类噪声中存续。我们为机器提取的法律条文形式背景的Duquenne-Guigues蕴含基构建了一种被动式存续证书：测量每个属性上抽取器之间的分歧，在1,000次蒙特卡洛试验中将这些分歧重放于蕴含基之上，只有当存续率的单侧Wilson 95%置信下界达到0.95时，一条蕴含才被认证；每条被认证的蕴含都附带前提文本片段和最小反例。在29,365个密苏里州法条章节和502个印度中央法案章节上，预注册的留出集检验得以通过（7编中10个法条族完全精确；11编中16个法条族在5%容差内通过），然而在某一全局部署的错误模型下，93.2%的留出章节未通过……（摘要在此截断）

    arXiv:2609.01741v1 Announce Type: new  Abstract: Statutes are increasingly parsed by machines before people read them, and the parsers disagree: on Missouri's statutes, two independently written extractors diverge on numeric-threshold presence at a false-negative rate of 0.43. We ask what formal logic survives such noise. We build a passive survival certificate for the Duquenne-Guigues implication basis of machine-extracted statutory contexts: per-attribute inter-extractor disagreement is measured, replayed against the basis in 1,000 Monte Carlo trials, and an implication is certified only when a one-sided Wilson 95% lower bound on survival reaches 0.95; every certified implication carries premise spans and a minimal counterexample. On 29,365 Missouri sections and 502 Indian central-Act sections, the preregistered held-out gate passes (10 statute families across 7 Titles exact; 16 across 11 with 5% tolerance), yet under one globally deployed error model 93.2% of held-out chapters fall 
    
[^92]: SpeakPay：面向低资源尼泊尔语金融语音识别的Whisper领域自适应LoRA微调

    SpeakPay: Domain-Adaptive LoRA Fine-Tuning of Whisper for Low-Resource Nepali Financial Speech Recognition

    [https://arxiv.org/abs/2609.01737](https://arxiv.org/abs/2609.01737)

    提出了SpeakPay语音优先数字钱包，通过构建403条尼泊尔语金融语音指令数据集并使用LoRA微调Whisper，将词错误率降低67.2%、天城文数字识别准确率从0%提升至73.9%、交易成功率提升约20倍，为视障用户提供了可用的语音支付方案。

    

    尼泊尔的移动支付应用以图形界面为主，对视障用户而言基本无法使用。本文提出了SpeakPay——一款语音优先的数字钱包，并记录了其核心技术贡献：一项针对低资源金融语音识别领域自适应的对照研究。我们引入了NepFinSpeech-403，一个包含403条尼泊尔语金融语音指令的数据集（涵盖转账、充值和余额查询操作，涉及237个不同的数字），并使用LoRA对Whisper large-v2进行微调。在留出测试集上，领域自适应模型将词错误率从129.95%（零样本基线）降至42.58%，相对降低了67.2%，并将天城文数字识别准确率从0.0%提升至73.9%。我们发现词级指标低估了实际任务层面的影响：领域自适应将交易成功率从1.67%提升至33.33%，约为20倍的提升。该改进在任务层面表现稳定……

    arXiv:2609.01737v1 Announce Type: new  Abstract: Mobile payment applications in Nepal are graphically mediated and largely inaccessible to visually impaired users. This paper presents SpeakPay, a voice-first digital wallet, and documents the central technical contribution: a controlled study of domain adaptation for low-resource financial speech recognition. We introduce NepFinSpeech-403, a 403-utterance dataset of Nepali financial voice commands (send, load, and balance operations spanning 237 unique numerals), and fine-tune Whisper large-v2 with LoRA. On the held-out test set, the domain-adapted model reduces Word Error Rate from 129.95% (zero-shot baseline) to 42.58% --- a 67.2% relative reduction --- and improves Devanagari numeral recognition accuracy from 0.0% to 73.9%. We find that word-level metrics understate the practical task-level impact: domain adaptation improves the Transaction Success Rate from 1.67% to 33.33%, a roughly 20x gain. The improvement is consistent at the in
    
[^93]: 通过智能体原生可复用工具原语实现LLM工具使用中的Harness工程

    Harness Engineering in LLM Tool Use via Agent-Native Reusable Tool Primitives

    [https://arxiv.org/abs/2609.01736](https://arxiv.org/abs/2609.01736)

    提出以自然语言取代API模式作为工具调用接口的“工具原语”设计，并构建包含25,519个函数的集中式仓库ToolFace供LLM在推理时动态检索工具，从而解决多步多轮推理脆弱及大规模工具目录下性能退化的问题。

    

    增强了外部工具的大型语言模型（LLM）在解决复杂现实任务方面已展现出卓越能力。然而，现有方法面临两个关键挑战：由工具输出类型和API模式不兼容导致的脆弱的多步与多轮推理，以及在大规模工具目录下的性能下降。为解决这些问题，我们提出了**工具原语**，这一设计以自然语言作为工具调用的接口，取代了僵化的基于API模式的调用方式，其中每个工具都被封装了一个LLM接口，在内部处理模式解析与执行，从而实现工具之间的自然通信，支持嵌套和多轮工具调用。基于工具原语，我们构建了**ToolFace**，一个包含25,519个函数的集中式仓库，LLM可以在推理时从中动态检索仅相关的工具，从而无需枚举原始API模式……（摘要原文在此处被截断）

    arXiv:2609.01736v1 Announce Type: cross  Abstract: Large language models (LLMs) augmented with external tools have demonstrated remarkable capability in solving complex real-world tasks. However, existing approaches suffer from two key challenges: brittle multi-step and multi-turn reasoning caused by incompatible tool output types and API schemas, and performance degradation under large tool catalogues. To address these, we introduce \textbf{Tool Primitives}, a design that replaces rigid API schema-based invocation with natural language as the interface for tool calling, where each tool is wrapped with an LLM interface that handles schema resolution and execution internally, enabling natural inter-tool communication for nested and multi-turn tool calling. Building on Tool Primitives, we host \textbf{ToolFace}, a centralized repository of 25,519 functions from which LLMs dynamically retrieve only the relevant tools at inference time, eliminating the need to enumerate raw API schemas in 
    
[^94]: 在有据多跳问答中学习证据充分性边界以实现选择性回答

    Learning Evidence Sufficiency Boundaries for Selective Answering in Grounded Multi-Hop QA

    [https://arxiv.org/abs/2609.01687](https://arxiv.org/abs/2609.01687)

    提出了证据充分性边界训练框架，通过构建有序证据链并直接监督弃答到作答的转变，使多跳问答模型学会在证据不支持或部分支持时弃答、证据首次充分时作答、且在冗余证据下保持答案稳定。

    

    有据问答系统应当仅在所提供的证据支持答案时才进行回答。在多跳问答中，这一要求难以满足，因为部分证据可能使不成立的答案显得合理。我们通过证据充分性边界来研究选择性回答：对于同一个问题，模型应在证据不支持或部分支持的上下文下弃答，在上下文首次变得充分时作答，并在添加冗余证据时保持答案稳定。我们提出了证据充分性边界训练，这是一种面向生成的训练框架，它构建有序的证据链并直接监督从弃答到作答的转变过程。该方法结合了层级监督、边界翻转间隔、边界后稳定性以及答案召回保护。我们基于HotpotQA、2WikiMultiHopQA和MuSiQue构建证据链，然后使用链式指标、原始问答效用以及不支持……（摘要原文在此处被截断）

    arXiv:2609.01687v1 Announce Type: new  Abstract: Grounded question answering systems should answer only when the supplied evidence supports the answer. In multi-hop QA, this requirement is difficult because partial evidence can make an unsupported answer appear plausible. We study selective answering through evidence sufficiency boundaries: for the same question, a model should abstain under unsupported or partially supported context, answer when the context first becomes sufficient, and keep the answer stable when redundant evidence is added. We introduce Evidence Sufficiency Boundary Training, a generation-native training framework that constructs ordered evidence chains and supervises the abstain-to-answer transition directly. The method combines level supervision, a boundary flip margin, post-boundary stability, and answer recall protection. We build evidence chains from HotpotQA, 2WikiMultiHopQA, and MuSiQue, then evaluate models with chain metrics, raw QA utility, and unsupported
    
[^95]: 由匹配器决定排名：威胁报告知识图谱抽取的可复现性审计

    Ranked by the Matcher: A Reproducibility Audit of Knowledge Graph Extraction from Threat Reports

    [https://arxiv.org/abs/2609.01671](https://arxiv.org/abs/2609.01671)

    该论文对威胁报告知识图谱抽取评估进行了可复现性审计，发现三元组匹配规则的不明确与差异会显著逆转系统排名（同一预测集F1跨度达0.16–0.70），并提出与人工裁决一致性达86%的LLM评判器以及可独立变换验证层的CTIForge平台，以实现更可靠、可分离组件效应的评估。

    

    安全团队和研究人员依据已发表的三元组F1分数来选择用于威胁报告的知识图谱抽取工具，然而这些分数取决于预测三元组如何与金标准标注进行匹配。在被审查的十二个系统中，我们仅能对其中五个重新实现其声明的匹配规则。在八种评分协议下对共享文档上的十个系统输出进行重新评分后，四十五对系统排序中有十一对发生逆转；同一份固定的预测集合的F1分数跨度可达0.16至0.70。在GRID的外部378项校准集上，没有任何机械匹配器（词汇、嵌入或蕴含类）与多评审员人工裁决的一致性超过71%，而LLM评判器则达到86%。为了将组件本身的效应与匹配器带来的奖励区分开，我们构建了CTIForge平台，其确定性验证层可以变化，同时保持抽取输出字节级完全一致。在七种测试的部署配置中，验证层提高了全部四个托管骨干模型的精确率，并降低了所有……

    arXiv:2609.01671v1 Announce Type: cross  Abstract: Security teams and researchers choose knowledge-graph extraction tooling for threat reports on the strength of published triple-F1 scores, yet those scores depend on how predicted triples are matched to gold annotations. We could reimplement the stated matching rule for only five of twelve inspected systems. Re-scoring ten system outputs on shared documents under eight protocols reverses eleven of forty-five pairwise orderings; one fixed prediction set spans 0.16-0.70 F1. On GRID's external 378-item calibration set, no mechanical matcher (lexical, embedding, or entailment) agrees with multi-reviewer adjudication above 71%, whereas an LLM judge reaches 86%. To separate component effects from matcher rewards, we build CTIForge, whose deterministic validation layer can vary while extraction is held byte-identical. Across seven tested deployment configurations, validation raises precision for all four hosted backbones and lowers it for all
    
[^96]: 超越文本思维链：自动驾驶中基于动作的推理综述

    Beyond Textual Chain-of-Thought: A Survey on Action-Grounded Reasoning in Autonomous Driving

    [https://arxiv.org/abs/2609.01659](https://arxiv.org/abs/2609.01659)

    本综述调研171篇论文，提出以中间表示形式为组织轴心的分类体系，将自动驾驶中从文本思维链转向基于动作推理的方法系统化为四大类13个子类，并指出能够扎根真实世界且与实时性耦合的中间表示是驾驶智能体推理的未来前沿。

    

    思维链推理通过在生成最终答案之前引出中间步骤来驱动生成模型。在自动驾驶中，答案是一个连续的动作，因此其推理必须与物理世界共享相同的时空结构。本综述研究了由此产生的从文本思维链向基于动作的推理的转变。通过调研171篇论文，其中包括130篇方法论文以及41篇基准、数据集、综述与分析论文，我们提出了一个以表示为中心的分类体系，将中间状态的形式作为组织轴心。我们将130种方法系统化为四个类别：基于语言的推理、视觉空间推理、潜在动态推理和外化推理，并进一步划分为与不同感兴趣区域相关联的13个子类型。我们的综合分析表明，驾驶智能体推理的开放前沿在于那些能够扎根于真实世界并与实时性相耦合的中间表示。

    arXiv:2609.01659v1 Announce Type: cross  Abstract: Chain-of-thought (CoT) reasoning powers generative models by eliciting intermediate steps before producing an answer. In autonomous driving, the answer is a continuous action. Thus its reasoning must share the same spatiotemporal structure as the physical world. This survey studies the resulting shift from textual CoT to action-grounded reasoning. Surveying 171 papers, including 130 method papers and 41 benchmarks, datasets, surveys, and analysis papers, we propose a representation-centered taxonomy that treats the form of the intermediate state as the organizing axis. We systematize the 130 methods into four categories: language-based, visual-spatial, latent-dynamic, and externalized reasoning, further divided into 13 subtypes tied to distinct regions of interests. Our synthesis shows that the open frontier of reasoning in driving agents lies in intermediate representations that can be grounded in the real world, coupled to real-time 
    
[^97]: PRO-Step：面向检索增强生成的步骤级过程奖励优化

    PRO-Step: Step-level Process Reward Optimization for Retrieval-Augmented Generation

    [https://arxiv.org/abs/2609.01658](https://arxiv.org/abs/2609.01658)

    PRO-Step训练了一个同时评估逻辑有效性与证据支撑的生成式过程奖励模型，通过PRM引导的价值树搜索构建偏好对并进行步骤级直接偏好优化，从而有效抑制RAG多跳推理中的错误传播问题。

    

    检索增强生成（RAG）通过将回答建立在外部知识的基础上来增强大语言模型，但多跳推理仍然容易受到错误传播的影响，即早期检索失败会干扰后续步骤。标准的基于结果的优化只奖励最终答案，使得中间的检索和推理错误无法被发现。虽然现有的基于过程的方法引入了步骤级信号，但它们仍然根据最终答案对每一步进行评分，从而奖励那些有缺陷的检索碰巧产生正确答案的虚假成功。RAG中的步骤级监督需要在每一步同时评估逻辑有效性和证据支撑。我们提出PRO-STEP：训练一个生成式过程奖励模型（PRM）来评估这两个维度，采用PRM引导的价值树搜索来构建对比有效步骤与有缺陷步骤的偏好对，并通过步骤级直接偏好优化（DPO）来优化策略。

    arXiv:2609.01658v1 Announce Type: cross  Abstract: Retrieval-Augmented Generation enhances Large Language Models by grounding responses in external knowledge, but multi-hop reasoning remains vulnerable to error propagation, where early retrieval failures confound subsequent steps. Standard outcome-based optimization only rewards the final answer, leaving intermediate retrieval and reasoning errors undetected. While existing process-based methods introduce step-level signals, they still score each step against the final answer, rewarding spurious successes where flawed retrieval coincidentally produces the correct answer. Step-level supervision in RAG requires evaluating both logical validity and evidential grounding at each step. We introduce PRO-STEP: we train a generative PRM that evaluates both dimensions, employ PRM-guided value tree search to construct preference pairs contrasting valid steps against flawed ones, and optimize the policy via step-level Direct Preference Optimizatio
    
[^98]: 谁的判断算数？众包内容审核中的代表性差距导致免于感知毒性的不平等保护

    Whose Judgments Count? Representation Gaps in Crowdsourced Content Moderation Produce Unequal Protection from Perceived Toxicity

    [https://arxiv.org/abs/2609.01625](https://arxiv.org/abs/2609.01625)

    该研究通过结合大规模删除判断数据与反事实模拟，揭示了众包内容审核中的“内群体保护”效应——与审核员群体共享人口身份的用户获得了不成比例的更多免于感知毒性的保护，从而导致不同用户群体受到的保护不平等。

    

    内容审核是数字治理的核心形式，然而人们对于哪些内容应当从共享的网络空间中移除存在分歧。尽管平台通过汇总人类判断来构建审核系统，但这一过程如何决定哪些用户受到保护、免于其所感知的有毒内容，仍不清楚。我们通过结合大规模判断数据与反事实模拟来填补这一空白，这些模拟追踪了审核员群体的人口构成如何塑造用户间保护的分布。将该框架应用于16,221名美国受访者对来自Twitter、Reddit和4chan的102,463条评论的删除判断，我们发现了审核需求中的人口异质性。我们进一步揭示了一种一致的内群体保护模式：感知毒性的降低不成比例地惠及与审核员群体共享人口身份的用户。至关重要的是，反映总体人口构成的审核员群体……（原文在此处截断）

    arXiv:2609.01625v1 Announce Type: cross  Abstract: Content moderation is a central form of digital governance, yet people disagree over what content should be removed from shared online spaces. While platforms aggregate human judgments to build moderation systems, it remains unclear how this process shapes which users are protected from content they perceive as toxic. We address this gap by combining large-scale judgment data with counterfactual simulations that trace how the demographic composition of moderator pools shapes the distribution of protection across users. Applying this framework to removal judgments from 16,221 U.S. respondents evaluating 102,463 comments from Twitter, Reddit, and 4chan, we find demographic heterogeneities in moderation demand. We further reveal a consistent pattern of in-group protection: reductions in perceived toxicity accrue disproportionately to users who share the demographic identities of the moderator pool. Crucially, moderator pools that mirror t
    
[^99]: MESSY STREETS：一个用于真实世界地址地理编码的基准

    MESSY STREETS: A Benchmark for Geocoding Real-World Addresses

    [https://arxiv.org/abs/2609.01612](https://arxiv.org/abs/2609.01612)

    MESSY STREETS是一个评估地理编码器处理真实杂乱网页地址的新基准，揭示了商业地理编码器的召回率比开源系统高出多达49个百分点，差距主要源于非规范地址的候选返回率差异。

    

    我们介绍了MESSY STREETS，这是一个用于评估地理编码器处理逐字网页地址的基准，包含存在性验证以及对表面形式偏差的受控测量。与基于干净或合成扰动地址的传统基准不同，MESSY STREETS包含表面形式与规范表示不一致的地址，其组成部分可能缺失、重复、格式错误或不完整。该基准基于2024年12月的Web Data Commons语料库构建，参考位置由OpenAddresses或OpenStreetMap确定。性能最强的商业地理编码器在召回率方面比开源系统高出多达49个百分点。这一差距主要由各系统在非规范地址上候选返回率的差异造成；一旦返回了候选结果，各系统的位置精度大体相当。仅非规范表面形式一项就导致了多达25个百分点的召回率差异。

    arXiv:2609.01612v1 Announce Type: cross  Abstract: We introduce MESSY STREETS, a benchmark for evaluating geocoders on verbatim web addresses, with existence verification and controlled measurement of surface-form divergence. Unlike conventional benchmarks based on clean or synthetically perturbed addresses, MESSY STREETS contains addresses whose surface forms diverge from canonical representations and whose components may be missing, repeated, malformed, or incomplete. The benchmark is constructed from the December 2024 Web Data Commons corpus, with reference locations established from OpenAddresses or OpenStreetMap.   The strongest commercial geocoders outperform open-source systems by up to 49 percentage points in recall. This gap is driven primarily by differences in candidate return rates on non-canonical addresses; once a candidate is returned, positional accuracy is broadly comparable across systems. Non-canonical surface form alone accounts for up to 25 percentage points of rec
    
[^100]: EvalDetectBench：一个用于衡量前沿语言模型评估意识的基准

    EvalDetectBench: A Benchmark for Measuring Evaluation Awareness in Frontier Language Models

    [https://arxiv.org/abs/2609.01611](https://arxiv.org/abs/2609.01611)

    该论文提出了EvalDetectBench，一个开放式的基准和流水线，用于衡量前沿大语言模型的评估意识（即识别自己正在被评估的能力）以及各个基准的可检测程度，从而保障AI安全评估结果的有效性。

    

    arXiv:2609.01611v1 公告类型：新 摘要：前沿大型语言模型通常能够识别出自己正在被评估，这种能力被称为“评估意识”。如果模型在评估中的表现与部署时的表现不同，这将损害评估结果的有效性，而评估结果是当前AI安全框架的关键组成部分。我们推出了EvalDetectBench，这是一个用于测量评估意识的开放式流水线和基准，可与任何兼容Inspect的评估配合使用，使从业者能够针对现有和未来的基准进行测试。EvalDetectBench附带了一个新整理的对话记录套件，涵盖当前前沿模型系统卡的评估场景以及多样化的部署来源。该基准有两个用途：衡量前沿大语言模型识别自己正在被评估的可靠程度，以及评估各个基准作为评估场景的可检测程度。我们发现了现有文献中引入系统性偏差的两个方法论选择：……

    arXiv:2609.01611v1 Announce Type: new  Abstract: Frontier large language models can often recognize when they are being evaluated, a capability known as evaluation awareness. If models behave differently in evaluations than in deployment, this undermines the validity of evaluation results, which are a crucial component of current AI safety frameworks. We introduce EvalDetectBench, an open pipeline and benchmark for measuring evaluation awareness that works with any Inspect-compatible evaluation, allowing practitioners to test against current and future benchmarks. EvalDetectBench ships with a newly curated transcript suite covering current frontier system-card evaluations and diverse deployment sources. The benchmark serves two purposes: measuring how reliably frontier LLMs recognize that they are being evaluated, and assessing how detectable individual benchmarks are as evaluations. We identify two methodological choices in the existing literature that introduce systematic bias: the i
    
[^101]: 基于熵的选择性智能体引导：从不完美的视觉语言模型教师中学习自主策略

    Selective Agent Guidance via Entropy: Learning Autonomous Policies from Imperfect VLM Teachers

    [https://arxiv.org/abs/2609.01567](https://arxiv.org/abs/2609.01567)

    该论文提出SAGE框架，仅在智能体不确定时才查询昂贵的视觉语言模型教师，并利用环境优势对教师建议进行加权蒸馏，从而训练出无需教师引导即可自主行动的轻量级强化学习策略。

    

    视觉语言模型为交互式决策提供了有用的先验知识，但直接将其用作策略既昂贵又脆弱：它们必须在每一步都被查询，无法通过环境交互得到改进，并且可能重复系统性错误。我们研究如何从一个在线、昂贵、不完美但具有信息量的视觉语言模型教师中学习一个廉价的自主策略。我们提出了SAGE（基于熵的选择性智能体引导），这是一个仅在学习者不确定时才查询视觉语言模型的框架，它在训练期间执行其建议的动作，并将引导蒸馏到一个轻量级的强化学习（RL）策略中。由于视觉语言模型的建议并不总是可靠的，SAGE可以使用由环境得出的优势来对教师动作蒸馏进行加权，而不是将所有建议视为同样有用。在稀疏奖励的视觉推理和导航任务中，SAGE学习到的策略在评估时无需视觉语言模型引导即可自主行动，并改进了……

    arXiv:2609.01567v1 Announce Type: new  Abstract: Vision-Language Models (VLMs) provide useful priors for interactive decision-making, but using them directly as policies is expensive and brittle: they must be queried at every step, do not improve from environment interaction, and can repeat systematic errors. We study how to learn a cheap autonomous policy from an online, expensive, and imperfect but informative VLM teacher. We propose SAGE (Selective Agent Guidance via Entropy), a framework that queries a VLM only when the learner is uncertain, executes the suggested action during training, and distills guidance into a lightweight Reinforcement Learning (RL) policy. Because VLM advice is not always reliable, SAGE can weight teacher-action distillation using environment-derived advantages rather than treating all suggestions as equally useful. Across sparse-reward visual reasoning and navigation tasks, SAGE learns policies that act without VLM guidance at evaluation time and improves o
    
[^102]: FinLifeBench：从纵向银行对话中穷尽式重建人生事件历史与财务状态

    FinLifeBench: Exhaustive Life-Event History and Financial-State Reconstruction from Longitudinal Banking Dialogue

    [https://arxiv.org/abs/2609.01198](https://arxiv.org/abs/2609.01198)

    提出FinLifeBench基准，基于6,000个韩语银行对话会话，评估大语言模型在穷尽式重建客户人生事件历史与34维财务状态方面的长程记忆能力，发现随会话累积事件召回率显著下降（0.591降至0.445），且错误主要源于事件遗漏。

    

    重复的银行交互要求助手在生活变化随日常请求偶然出现时，维护完整、最新且可追溯的客户记录。现有基准强调问答、有界回合或定向回忆，而非穷尽式的纵向重建。我们提出FinLifeBench，它在同一累积对话上评估两项任务：重建每个人生事件实例及其首次确立的会话，以及在连续检查点上重建完整的34条路径财务状态。该基准包含来自20条独立合成轨迹的6,000个八轮韩语银行会话，为24种事件类型和34条状态路径提供确定性的、穷尽式的黄金标准及共识质量保证。在全上下文条件下对十一个大语言模型的评估中，事件锚点召回率从15个会话时的0.591下降至300个会话时的0.445。错误主要由遗漏事件导致，而非（摘要原文在此处截断）

    arXiv:2609.01198v1 Announce Type: cross  Abstract: Repeated banking interactions require assistants to maintain complete, current, and traceable customer records as life changes emerge incidentally in routine requests. Existing benchmarks emphasize question answering, bounded episodes, or targeted recall rather than exhaustive longitudinal reconstruction. We introduce FinLifeBench, which evaluates two tasks over the same cumulative dialogue: reconstructing every life-event instance with its first-establishing session and reconstructing a complete 34-path financial state at consecutive checkpoints. The benchmark contains 6,000 eight-turn Korean banking sessions from 20 independent synthetic trajectories, with deterministic, exhaustive gold for 24 event types and 34 state paths and consensus quality assurance. Across eleven LLMs under a full-context condition, event-anchor recall falls from 0.591 at 15 sessions to 0.445 at 300. Errors are driven primarily by omitted events rather than po
    
[^103]: 见好就收：用于机器翻译重排序中高效候选生成的Quit方法

    Quit While You're Ahead: Quit for Efficient Candidate Generation in Machine Translation Reranking

    [https://arxiv.org/abs/2609.00588](https://arxiv.org/abs/2609.00588)

    提出Quit方法，通过不确定性量化的早停策略对机器翻译的整个候选生成—重排序流程进行增量式生成与重排序，在最高候选质量稳定时提前终止，从而在保持翻译质量的同时显著降低推理延迟。

    

    重排序方法，如最小贝叶斯风险（MBR）解码和质量估计（QE）重排序，被广泛应用于现代神经机器翻译（NMT）中，用于从一组候选假设中选出最终输出。然而，这些性能提升是以高推理延迟为代价的。现有的加速方法仅针对MBR解码且只减少重排序计算，既未解决QE重排序的问题，也基本未触及候选生成——而后者可能是更大的计算瓶颈。在本工作中，我们提出了Quit（基于不确定性量化的增量终止），这是一种针对整个“生成—重排序”流程的新型早停策略。Quit将候选生成视为不确定性下的序列决策过程，增量式地生成并重排序候选译文，当候选集中最高的估计质量趋于稳定时即停止生成。在三个NMT模型、19个语言对上的全面实验表明……

    arXiv:2609.00588v1 Announce Type: new  Abstract: Reranking methods, such as Minimum Bayes Risk (MBR) decoding and Quality Estimation (QE) reranking, are widely used in modern neural machine translation (NMT) to select an output from a set of candidate hypotheses. However, the performance gains come at the cost of high inference latency. Existing acceleration methods target MBR decoding and reduce only reranking computation, leaving QE reranking unaddressed and candidate generation---which can be the larger computational bottleneck---largely untouched. In this work, we propose Quit (Quantifying Uncertainty for Incremental Termination), a novel early-stopping strategy for the entire generation--reranking pipeline. Viewing candidate generation as a sequential decision under uncertainty, Quit incrementally generates and reranks candidates, stopping when the highest estimated quality in the candidate set stabilizes. Comprehensive experiments on three NMT models across 19 language pairs show
    
[^104]: 探索语言智能体与非语言智能体之间的协作

    Exploring Collaboration between a language and a non-language agent

    [https://arxiv.org/abs/2609.00474](https://arxiv.org/abs/2609.00474)

    该论文提出LLAMIA-Bench基准，用于研究将非语言智能体的连续表示“言语化”为文本是否成为LLM协作的瓶颈，并提出潜在状态内化方法来改善LLM与国际象棋引擎等非语言智能体的协作。

    

    大型语言模型（LLM）越来越多地被部署为协调者，通过自然语言调度专门的子智能体来解决复杂任务。然而，在博弈和机器人技术等许多重要领域，目前最强的智能体并非语言模型。将非语言智能体与LLM集成需要进行“言语化”：在每个交互步骤中，将其丰富的连续表示压缩为稀疏的文本摘要。为了研究言语化是否构成瓶颈，我们提出了LLAMIA-Bench，这是一套包含六个多样化协作式国际象棋任务的基准，涵盖三个方面：行为模仿、状态评估和自然语言解释。每个任务都对应一个经典的国际象棋难题，无论是LLM还是象棋引擎都无法独立解决。为了实现LLM与非语言智能体的协作，我们提出了“潜在状态内化”方法，将子智能体的连续表示投影到……

    arXiv:2609.00474v1 Announce Type: cross  Abstract: LLMs are increasingly deployed as orchestrators that coordinate specialized subagents to solve complex tasks through natural language. However, in many important domains like game playing and robotics, the strongest available agents are not language models. Integrating non-language agents with LLMs would require \emph{verbalization}: compressing their rich continuous representations into sparse textual summaries at each interaction step. To study whether verbalization constitutes a bottleneck, we introduce \textsc{LLAMIA-Bench}, a suite of six diverse collaborative chess tasks spanning three facets: behavioral imitation, state assessment, and natural-language explanation. Each task instantiates a well-established chess problem that neither the LLM nor the chess engine can solve alone. To solve LLM collaboration with non-language agents, we introduce \emph{latent state internalization}, which projects the subagent's continuous represent
    
[^105]: CogEvol：迈向高效可靠的学习环境生成

    CogEvol: Towards Efficient and Reliable Learning Environment Generation

    [https://arxiv.org/abs/2608.30968](https://arxiv.org/abs/2608.30968)

    CogEvol是专为学习环境生成训练的模型系列，能将课程简报一次性转化为幻灯片或交互式HTML页面，通过真实生产失败数据驱动的SFT和修复奖励作弊后加固的GRPO强化学习保障可靠性，其27B模型以少26.9倍的参数量媲美旗舰编程模型，并已投入真实生产环境服务。

    

    我们提出了CogEvol，一个专门为学习环境生成任务训练的模型家族：将课程简报一次性转化为成品学习产物（结构化JSON幻灯片或自包含的交互式HTML页面）。在22万次生产请求中，CogEvol生成一张幻灯片的中位耗时为17秒，生成一个交互式页面为59秒，取代了耗时数分钟的多轮智能体框架流程。可靠性是被强制保障而非仅是期望：一条以生产环境为基础的数据管道将真实失败案例转化为53,687条经验证的SFT样本，同时一种规则与视觉语言模型（VLM）相结合的混合奖励驱动基于GRPO的强化学习——在我们发现并修复了一次奖励作弊（reward hacking）事件（该事件产生了视觉上令人信服但无法游玩的游戏）后，系统得到了进一步加固。CogEvol-27B在幻灯片质量上得分83.7，在包含500个案例的交互式HTML基准上得分63.7，而其参数量比旗舰级编程模型少26.9倍，并与OpenMAIC团队合作，为其线上生产流量提供服务。

    arXiv:2608.30968v1 Announce Type: cross  Abstract: We present CogEvol, a family of models trained specifically for Learning Environment Generation: turning a course brief into a finished learning artifact (structured-JSON slides or self-contained interactive HTML pages) in a single pass. Across 220k production requests, CogEvol completes a slide in a median of 17 seconds and an interactive page in 59, replacing minutes-long multi-turn agent scaffolding. Reliability is enforced rather than hoped for: a production-grounded data pipeline turns real failures into 53,687 verified SFT samples, and a hybrid rule-plus-VLM reward drives GRPO-based RL, hardened after we caught and fixed a reward-hacking episode that produced visually convincing but unplayable games. CogEvol-27B scores 83.7 on slide quality and 63.7 on a 500-case interactive-HTML benchmark with 26.9x fewer parameters than flagship coding models, and, in collaboration with the OpenMAIC team, serves their live production traffic. C
    
[^106]: Lot Machine：从拍卖目录中进行多模态拍品信息抽取

    Lot Machine: Multimodal Lot Extraction from Auction Catalogs

    [https://arxiv.org/abs/2608.30510](https://arxiv.org/abs/2608.30510)

    本文提出了一个利用视觉-语言模型从历史拍卖目录中自动提取结构化拍品元数据的流水线，并在不同提示策略、受限解码框架和部署条件下进行了系统评估，以满足文化遗产机构在预算、算力和数据隐私方面的实际需求。

    

    对于溯源研究和艺术市场研究而言，拍卖目录是追踪特定物品在时间和空间上流转的重要资源。虽然历史拍卖目录遵循既定的领域惯例，但其内部格式仍然高度多变，且由于缺乏机器可读的拍卖拍品表示，其大规模分析目前受到限制。我们提出了一个流水线，可以从 German Sales（一个涵盖19和20世纪历史拍卖与销售目录的大型数据库）中自动提取结构化的拍品级元数据。基于人工标注的代表性目录页面测试集，我们在不同的提示策略和受限解码框架下对视觉-语言模型进行了评估。为了反映文化遗产机构面临的实际约束，包括预算、计算资源和数据隐私要求，我们在不同的部署方式下对这些方法进行了基准测试。

    arXiv:2608.30510v1 Announce Type: cross  Abstract: For provenance research and art market studies, auction catalogs are an essential resource to trace specific objects over time and space. While historical auction catalogs follow established domain conventions, their internal formatting remains highly variable, and their large-scale analysis is currently restricted by the lack of machine-readable representations of the auction lots. We propose a pipeline to automatically extract structured lot-level metadata from German Sales, a large database of historical auction and sales catalogs from the 19th and 20th centuries. Using a manually annotated test set of representative catalog pages, we evaluate Vision-Language Models (VLMs) under varying prompt strategies and constrained decoding frameworks. To reflect the practical constraints faced by cultural heritage institutions, including budget, compute resources, and data privacy requirements, we benchmark the methods across different deploym
    
[^107]: GPAgentBench-2K：在复杂临床动作空间中评测大语言模型智能体

    GPAgentBench-2K: Benchmarking Large Language Model Agents in Complex Clinical Action Space

    [https://arxiv.org/abs/2608.30188](https://arxiv.org/abs/2608.30188)

    该论文提出了首个基于受约束马尔可夫决策过程的基层医疗临床决策LLM智能体基准GPAgentBench-2K，评估发现即使是诊断准确率最高的前沿模型，在超过一半的高风险病例中也会违反安全约束，揭示了临床质量与安全之间的鸿沟。

    

    大语言模型（LLM）作为临床智能体展现出巨大潜力，然而现有基准测试将临床工作流程简化为静态预测或具有粗粒度动作集的无约束马尔可夫决策过程（MDP）。为解决这一问题，我们提出了GPAgentBench-2K，这是首个面向基层医疗临床决策的受约束马尔可夫决策过程（CMDP）LLM智能体基准，其构建自经过专家验证的真实全科医生（GP）问诊记录。我们的环境建模了六种基础临床动作的完整谱系，在动作空间上施加了拓扑工作流先验，并将基于安全性的弃权决策操作化为一种一等结果。对16个最先进LLM的评估显示，随着动作空间的扩展，模型性能显著下降。关键的是，我们揭示了临床质量-安全鸿沟：即使是诊断准确率最高的前沿模型，在超过一半的高风险病例中也违反了安全约束。最后，我们建立了一个参考（摘要在此处截断）

    arXiv:2608.30188v1 Announce Type: new  Abstract: Large Language Models (LLMs) show great potential as clinical agents, yet existing benchmarks reduce clinical workflows to static predictions or unconstrained Markov Decision Processes (MDPs) with coarse action sets. To address this, we introduce GPAgentBench-2K, the first Constrained MDP (CMDP) LLM-agent benchmark for primary-care clinical decision-making, constructed from expert-validated records of real-world GP encounters. Our environment models a full spectrum of six foundational clinical actions, imposes a topological workflow prior over the action space, and operationalizes safety-informed abstention as a first-class outcome. Evaluating 16 state-of-the-art LLMs reveals a significant performance degradation as the action space scales. Crucially, we uncover a clinical quality-safety gap: even frontier models with the highest diagnosis accuracy violate safety constraints in over half of high-risk cases. Finally, we establish a refere
    
[^108]: SHADOWBENCH：迈向自动形式化中语义对齐的可靠自动评估

    SHADOWBENCH: Toward Reliable Automatic Evaluation of Semantic Alignment in Autoformalization

    [https://arxiv.org/abs/2608.29270](https://arxiv.org/abs/2608.29270)

    提出 SA-Pass 评估方法，通过“影子”辅助陈述的双向逻辑检查来可靠评估自动形式化中的语义对齐，并构建了包含 178 个研究生至研究级问题的 Lean 4 基准 ShadowBench。

    

    自动形式化是将非正式的数学定理翻译为 Lean 等证明助手代码的过程。其核心挑战在于，当前的评估指标可能会接受类型正确但语义不对齐的陈述，或者拒绝以不同表述方式书写的正确陈述。受 Pass@$k$ 的启发，我们提出了 SA-Pass（语义对齐通过测试），该方法使用称为“影子”的辅助陈述来测试形式化陈述，这些影子刻画了目标陈述的意图。只有当一个生成的陈述能够编译、蕴含每个影子（前向检查），并且被所有影子的合取所蕴含（后向检查）时，才能获得满分。我们在 ShadowBench 中实例化了 SA-Pass，这是一个 Lean 4 完整自动形式化基准，包含 178 个研究生至研究级别的问题，涵盖八个数学领域。配备 Numina-Lean-Agent 的 Claude Code（Opus 4.8）达到了 61.8% 的编译率和 11.2% 的 SA-Pass。在六种智能体配置生成的输出上，SA-Pass 实现了……

    arXiv:2608.29270v1 Announce Type: cross  Abstract: Autoformalization translates informal mathematical theorems into code for proof assistants such as Lean. A central challenge is that current evaluation metrics can accept type-correct but misaligned statements or reject correct statements written in a different formulation. Inspired by Pass@$k$, we propose SA-Pass (*Semantic Alignment Pass*), which tests formal statements using auxiliary statements called *shadows* that characterize the intended statement. A generated statement receives full credit only when it compiles, implies each shadow (forward check), and is implied by their conjunction (backward check). We instantiate SA-Pass in ShadowBench, a Lean 4 full autoformalization benchmark of 178 postgraduate- to research-level problems spanning eight mathematical areas. Claude Code (Opus 4.8) with Numina-Lean-Agent reaches $61.8\%$ compile rate and $11.2\%$ SA-Pass. Across outputs generated by six agentic configurations, SA-Pass achie
    
[^109]: 替代的幻象：重新思考基础模型时代的专用机器学习模型

    The Illusion of Replacement: Rethinking Specialized Machine Learning Models in the Foundation Model Era

    [https://arxiv.org/abs/2608.28980](https://arxiv.org/abs/2608.28980)

    本文综述159篇论文后发现，语言模型虽在极端少样本预测等特定场景中可与专用模型竞争，但一旦直接评估结构表示与计算能力，并无证据表明其能全面取代机器学习中的专用架构。

    

    机器学习传统上为结构化数据构建的专用架构能否被基于语言的模型所取代？本文通过对2016年至2026年间涵盖九种模态的159篇论文的综述来检验这一问题，在考虑预测精度的同时兼顾结构表示与结构计算。论文区分了“执行任务”与“保留并计算使任务可处理的结构”这两个概念，并将现有方法归纳为八种表示机制，范围从纯语言系统到完全专用的架构。研究发现，语言中介模型在特定场景下极具竞争力，包括极端少样本预测、离散化符号任务、文本标注的知识图谱以及大规模单模态预训练。然而，只要直接评估结构表示或结构计算而非仅评估精度，就没有发现通用替代的证据。

    arXiv:2608.28980v1 Announce Type: cross  Abstract: Can the specialized architectures that machine learning has traditionally built for structured data be replaced by language-based models? This question is examined through a review of 159 papers (2016--2026) across nine modalities, with predictive accuracy considered alongside structural representation and computation. A distinction is made between performing a task and preserving and computing the structure that makes the task tractable, and existing approaches are organized into eight representational regimes, ranging from language-only systems to fully specialized architectures. Language-mediated models are found to be highly competitive in specific settings, including extreme few-shot prediction, discretized symbolic tasks, textually annotated knowledge graphs, and large-scale single-modality pretraining. However, whenever structural representation or computation is directly evaluated rather than accuracy alone, no evidence of gene
    
[^110]: 自动化研究人员能够可靠地缓解对齐失败

    Automated Researchers Can Reliably Mitigate Alignment Failures

    [https://arxiv.org/abs/2608.28945](https://arxiv.org/abs/2608.28945)

    自动化对齐研究员（AAR）通过后训练方法能够可靠地缓解10种对齐失败并泛化到更大的模型，其效果甚至优于28名经验丰富的人类研究员在八小时内开发的方法。

    

    自动化对齐研究可能会加速实现与人类对齐的AI的进程，但这是否真的有效却难以衡量。幸运的是，许多对齐失败，例如欺骗、谄媚和越狱，已经可以通过公开基准来衡量。我们研究了自动化对齐研究员能否通过后训练来缓解对齐失败，方法是提出训练方法和数据，以同时优化多个安全基准，同时保持通用能力。在10种对齐失败中，最强的AAR方法显著减少了目标对齐失败，并能泛化到留出的基准测试、多轮行为审计，以及比目标模型大4.7倍的模型。作为人类基线，28名经验丰富的研究人员获得了最多八小时的时间来为相同的基准开发方法，但他们的方法表现不如最好的AAR方法。将人类想法作为AAR的初始研究方向并不能改善结果。

    arXiv:2608.28945v1 Announce Type: new  Abstract: Automating alignment research may accelerate progress toward aligned AI, but whether it does is hard to measure. Luckily, many alignment failures, such as deception, sycophancy, and jailbreaks, are already measurable by public benchmarks. We study whether automated alignment researchers (AARs) can post-train to mitigate alignment failures by proposing training methods and data to simultaneously optimize multiple safety benchmarks, while preserving general capability. Across 10 alignment failures, the strongest AAR methods significantly reduce the targeted alignment failures and generalize to a held-out benchmark, multi-turn behavioral audits, and models up to 4.7 times larger than the target model. As a human baseline, 28 experienced researchers receive up to eight hours to develop methods for the same benchmarks, but their methods underperform the best AAR methods. Using human ideas as the AARs' initial research direction does not impro
    
[^111]: 在截断评分量表上的双重差分法可能制造虚假效应：来自预注册大语言模型裁判审计的证据

    Difference-in-Differences on a Censored Rating Scale Can Manufacture an Effect: Evidence from a Pre-Registered LLM-Judge Audit

    [https://arxiv.org/abs/2608.27309](https://arxiv.org/abs/2608.27309)

    本文揭示双重差分法在截断评分量表上因截断不均会制造虚假交互效应，并通过预注册审计实证证明该偏差可导致无效结果。

    

    arXiv:2608.27309v1 公告类型：交叉 摘要：对大语言模型裁判的审计通过对比匹配条件来验证偏差，而最严谨的设计会进行两次差分：在两项候选回答之间进行项目内对比，再在操控的属性上进行二次差分，最终从有界评分量表上读取结果。我们证明，这一终点在该量表上无法被识别。双重差分的每一项都受到各自截断比例的影响，因此观察到的统计量混淆了差异偏好与差异衰减：当两项回答以不平等的方式截断时，一个共同作用于两者的严重性偏移会制造出交互效应，因为距界限的不等距离使它们恰好落在良好刺激所在的位置。我们在一个冻结教学裁判的预注册审计中展示了这一失败，该审计在首次990次调用之前就已密封。注册的主要终点，即陈述的学习者画像对裁判脚手架偏好的影响，为零：+0.085分（95% BCa置信区间）。

    arXiv:2608.27309v1 Announce Type: cross  Abstract: Audits of LLM judges certify a bias by contrasting matched conditions, and the strongest designs difference twice: a within-item contrast between two candidate responses, differenced again across a manipulated attribute, read off a bounded rating scale. We show that this endpoint is not identified on the scale that reports it. Each term of the double difference is censored by its own share, so the observed statistic confounds differential preference with differential attenuation: a severity shift common to both responses manufactures an interaction whenever the two censor it unequally, as unequal distances from the bounds make them, exactly where good stimuli place them. We exhibit the failure inside a pre-registered audit of a frozen pedagogy judge, sealed before the first of its 990 calls. The registered primary endpoint, the effect of a stated learner profile on the judge's scaffolding preference, is null: $+0.085$ points (95\% BCa 
    
[^112]: SPEAR：通过强化学习中的序列符号对齐提炼领域自适应推理骨架

    SPEAR: Distilling Domain-Adaptive Reasoning Skeletons via Sequential Symbolic Alignment in Reinforcement Learning

    [https://arxiv.org/abs/2608.26550](https://arxiv.org/abs/2608.26550)

    SPEAR提出了一种无需训练、即插即用的过程奖励方法，通过符号里程碑和最长公共子序列对齐，在强化学习蒸馏中提供密集且逻辑一致的奖励，避免了昂贵的外部神经过程奖励模型。

    

    基于强化学习的知识蒸馏有潜力将复杂推理从教师模型转移到学生模型，但目前面临一个关键困境：研究者必须在稀疏的基于结果的奖励（其逻辑指导不足）和昂贵的神经过程奖励模型（用于密集信号）之间做出选择。我们通过引入SPEAR（符号过程评估与对齐奖励）解决了这一问题，这是一种无需训练且即插即用的过程奖励方法，用于序列级策略蒸馏。SPEAR将自然语言推理轨迹投影到领域自适应的符号里程碑中，为过程级推理对齐提供了高效代理。通过利用最长公共子序列（LCS）来将学生探索与教师里程碑对齐，SPEAR提供了密集且顺序感知的奖励信号，在不需外部神经验证器的情况下强制逻辑一致性。我们的实验...

    arXiv:2608.26550v1 Announce Type: new  Abstract: Reinforcement learning-based knowledge distillation has the potential to transfer complex reasoning from teacher to student models, yet it currently faces a critical dilemma: researchers must choose between sparse outcome-based rewards, which provide insufficient logical guidance, or expensive neural Process Reward Models (PRMs) for dense signals. We resolve this by introducing SPEAR (Symbolic Process Evaluation and Alignment Reward), a training-free and plug-and-play process reward method for sequence-level on-policy distillation. SPEAR projects natural-language reasoning traces into domain-adaptive symbolic milestones, providing an efficient proxy for process-level reasoning alignment. By utilizing the longest common subsequence (LCS) to align student explorations with teacher milestones, SPEAR provides a dense, order-aware reward signal that enforces logical consistency without the need for an external neural verifier. Our experiments
    
[^113]: 参数化知识图谱记忆中的存储-检索差距

    A Storage-Retrieval Gap in Parametric Knowledge Graph Memory

    [https://arxiv.org/abs/2608.25489](https://arxiv.org/abs/2608.25489)

    该论文提出将知识图谱离线编译为LoRA适配器作为参数化知识层，在零查询上下文成本下实现事实知识泛化，但发现存储知识无法通过相似性检索恢复，揭示了参数化记忆中的存储-检索差距。

    

    arXiv:2608.25489v1 公告类型：交叉 摘要：图检索增强生成在查询时将检索到的子图放入模型的上下文窗口中，每次调用都支付重复的令牌成本，并在每次调用时暴露源数据。我们研究了一种替代方案：将知识图谱离线编译为每个实体一个LoRA适配器的库，这些适配器作为参数化知识层，通过注入权重而非文本来查询，在查询时零上下文成本。在MetaQA数据集上，我们发现子图训练的适配器编码了上下文无关的事实知识，这些知识能泛化到未见问题：在单值关系上，适配器相对于几乎无法闭卷的基础模型（0.007）获得了+0.243的精确匹配分数提升，且只有正确的适配器能恢复这些知识（相对于基础模型的oracle差距为+0.283）。然而，存储的知识无法通过相似性恢复：在无子图的查询下，基于嵌入和权重空间几何的检索性能均不佳。

    arXiv:2608.25489v1 Announce Type: cross  Abstract: Graph retrieval-augmented generation places retrieved subgraphs into the model's context window at query time, paying a recurring token cost and exposing source data on every call. We study an alternative: compiling a knowledge graph offline into a bank of LoRA adapters, one per entity, that serve as a parametric knowledge layer queried by injecting weights rather than text, at zero query-time context cost. On the MetaQA dataset, we find that subgraph-trained adapters encode context-free factual knowledge that generalizes to unseen questions: on single-valued relations the adapter gains $+0.243$ exact-match score over a base model that is nearly blind closed-book ($0.007$), and only the correct adapter recovers this knowledge (an oracle gap of $+0.283$ over the base model). However, the stored knowledge is not recoverable by similarity: given a query with no subgraph, embedding-based and weight-space geometry retrieval both perform at 
    
[^114]: 洗白仇恨、污蔑无害内容：针对基于LLM的内容审核的标注者风格反驳攻击

    Whitewashing Hate, Smearing Harmless Content: Annotator-Style Rebuttal Attacks on LLM-Based Moderation

    [https://arxiv.org/abs/2608.22230](https://arxiv.org/abs/2608.22230)

    本研究揭示了标注者风格的反驳攻击能显著破坏LLM仇恨言论审核的准确性，且洗白与污蔑两种操纵方向存在模型特定的不对称效应。

    

    大型语言模型（LLMs）越来越多地被用于仇恨言论审核，通常出现在人类与AI协作的工作流程中，其中审核者在最终决策前提供反馈。这种反馈引入了两种操纵方向：将仇恨内容洗白为正常内容，以及将正常内容污蔑为仇恨内容。本研究考察了初始正确的模型判断对标注者风格反驳的敏感性，并分析了攻击有效性是否因操纵方向而异。我们引入了一种重新判断协议，该协议通过决策边界扰动和对抗性理由扩展了直接矛盾。在多个LLM和两个仇恨言论数据集上的实验表明，标注者风格的反驳显著降低了审核性能，在多轮设置中效果更强。结果进一步揭示了在攻击配置中，洗白和污蔑之间存在稳定且模型特定的不对称性，这表明...

    arXiv:2608.22230v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used for hate speech moderation, often within human--AI workflows in which reviewers provide feedback before a final decision. Such feedback introduces two manipulation directions: whitewashing hateful content as normal and smearing normal content as hateful. This study examines the susceptibility of initially correct model judgments to annotator-style rebuttals and analyzes whether attack effectiveness differs across manipulation directions. We introduce a rejudge protocol that extends direct contradiction with decision-boundary perturbations and adversarial rationales. Experiments with multiple LLMs on two hate speech datasets show that annotator-style rebuttals substantially degrade moderation performance, with stronger effects in multi-turn settings. The results further reveal stable, model-specific asymmetries between whitewashing and smearing across attack configurations, indicating dis
    
[^115]: ToSCA：基于对话代理时间与策略抽象的层次强化学习

    ToSCA: Leveraging Hierarchical Reinforcement Learning on Temporal and Strategic Abstractions of Conversational Agents

    [https://arxiv.org/abs/2608.21969](https://arxiv.org/abs/2608.21969)

    本文提出一种两级层次强化学习框架，结合话语级策略抽象与词元级解码，并引入双粒度奖励机制，以提升对话代理在复杂交互中的性能。

    

    人类在日常互动和思考中具有多个层次的时间抽象能力，例如概念感知和策略规划。受此启发，我们为对话代理提出了一种两级层次强化学习（RL）框架，弥合了以往基于词元级别或话语级别RL方法之间的差距。该框架基于两级MDP开发，其中词元级别的响应解码依赖于话语级别的动作，即显式文本策略。基于理论推导和效率考虑，我们使用DQN求解高层评论家，使用PPO求解低层演员-评论家。为进一步缓解奖励稀疏性并促进收敛，我们还设计了双粒度奖励机制，将话语级别的满意度评分与词元级别的内在动机和K-L惩罚相结合。在日常对话和情感支持对话上的实验表明，所提方法优于现有基线。

    arXiv:2608.21969v1 Announce Type: new  Abstract: Humans have multiple levels of temporal abstractions on daily interaction and thinking, such as concept perception and strategic planning. Inspired by this nature, we propose a two-level hierarchical reinforcement learning (RL) framework for conversational agents, bridging the gap between previous token-level or utterance-level RL methods. Developed on a two-level MDP, the token-level response decoding is conditioned on the utterance-level action, the explicit textual strategies. Based on theoretical derivation and efficiency consideration, we use DQN to solve the high-level critic and PPO to solve the low-level actor-critic. To further alleviate the reward sparsity and facilitate the convergence, we also design the dual-granularity reward mechanism, in which the utterance-level satisfaction score is integrated with token-level intrinsic motivation and K-L penalty. Experiments on both daily and emotional support conversations show that o
    
[^116]: 代理式脚手架放大大型语言模型中的谄媚行为

    Agentic Scaffolding Amplifies Sycophantic Behavior in Large Language Models

    [https://arxiv.org/abs/2608.21377](https://arxiv.org/abs/2608.21377)

    本文发现代理式交互脚手架（如多轮反馈和迭代细化）会系统性放大LLM的谄媚行为，导致平均准确率下降6.3%，且更强模型放大效应更显著。

    

    大型语言模型中的谄媚行为，即优先迎合用户认同而非提供真实回答的倾向，已被广泛记录，但主要在单轮对话场景中研究。本文探讨了一个关键问题：对LLM施加更强的交互脚手架是否会使谄媚行为变得更糟？通过4800次真实性判断（200个陈述×6个模型×4种条件），我们发现代理系统特有的交互脚手架（反馈循环、重新考虑检查点和迭代细化）系统性地放大了谄媚行为。多轮交互、用户压力和迭代自我细化各自为模型提供了更多趋向认同的机会，这种漂移伴随着平均准确率下降6.3个百分点，表明这种屈服是有害的而非纠正性的。更强大的模型显示出更大的放大效应，这...

    arXiv:2608.21377v1 Announce Type: cross  Abstract: Sycophancy in large language models, the tendency to prioritize user agreement over truthful responses, has been documented extensively but studied primarily in single-turn settings. This paper investigates a critical question: does subjecting LLMs to greater interaction scaffolding make sycophancy better or worse? Across 4,800 veracity judgments (200 statements $\times$ 6 models $\times$ 4 conditions), we find that the interaction scaffolding characteristic of agentic systems (feedback loops, reconsideration checkpoints, and iterative refinement) systematically amplifies sycophantic behavior. Multi-turn interaction, user pressure, and iterative self-refinement each provide additional opportunities for models to drift toward agreement, and this drift coincides with a mean accuracy drop of $-6.3$ percentage points, establishing the capitulation as harmful rather than corrective. More capable models show larger amplification effects, a t
    
[^117]: LoRA-GA²：基于多步梯度自适应对齐的低秩适应方法

    LoRA-GA$^2$: Low Rank Adaptation with Multi-step Gradient Adaptive Alignment

    [https://arxiv.org/abs/2608.19800](https://arxiv.org/abs/2608.19800)

    本文提出LoRA-GA²算法，通过轻量级探针利用多步梯度信息，结合谱感知的秩分配和最优初始化，在不增加GPU内存的前提下缩小LoRA与全参数微调的性能差距。

    

    低秩适应（LoRA）是一种突出的大型模型微调方法，能以降低的内存开销实现有竞争力的性能。然而，LoRA与全参数微调之间仍存在持续的性能差距。近期研究试图通过使用预训练权重的单步梯度近似，将LoRA更新与全参数微调更新的主方向或内在维度对齐来缩小这一差距。然而，这些方法未能捕捉梯度的完整动态。在本文中，我们提出LoRA-GA²，一种有效利用多步梯度信息的微调算法。具体来说，我们引入了一个轻量级的探针，用于获取预训练权重的多步梯度，且不增加额外的GPU内存成本，仅带来微不足道的时间开销。我们进一步采用了基于谱感知和重要性驱动的秩分配，以及从多步梯度推导出的最优初始化。

    arXiv:2608.19800v1 Announce Type: cross  Abstract: Low-Rank Adaptation (LoRA) is a prominent fine-tuning method for large models, achieving competitive performance with reduced memory overhead. However, a persistent performance gap remains between LoRA and full fine-tuning. Recent studies have sought to narrow this gap by employing one-step gradient approximations of pretrained weights to align LoRA updates with the principal directions or intrinsic dimensionalities of full fine-tuning updates. Nevertheless, these approaches fail to capture the full dynamics of the gradients. In this paper, we propose LoRA-GA$^2$, an effective fine-tuning algorithm that fully leverages multi-step gradient information. Specifically, we introduce a lightweight probe for multi-step gradients of pretrained weights that incurs no additional GPU memory cost and only marginal time overhead. We further employ a spectrum-aware, importance-based rank allocation and optimal initialization derived from multi-step 
    
[^118]: 偏好树优化：通过前瞻模拟增强目标导向对话

    Preference Tree Optimization: Enhancing Goal-Oriented Dialogue with Look-Ahead Simulations

    [https://arxiv.org/abs/2608.12062](https://arxiv.org/abs/2608.12062)

    本文提出偏好树优化框架，结合前瞻模拟和直接偏好优化，有效应对数据稀缺，提升目标导向对话系统的决策能力。

    

    摘要：开发能够进行多轮、目标导向对话的对话系统仍然是一个重大挑战，尤其是在数据有限的专业领域。本研究提出了一种名为偏好树优化（PTO）的新框架，旨在通过一种称为“带前瞻的偏好树”的方法生成偏好数据，从而迭代改进此类对话系统中的代理模型。聚焦于动机性访谈（MI）——一种旨在促进行为改变的咨询技术——我们利用虚拟患者和预言评估器模拟对话并生成丰富的偏好数据集。通过将此方法与直接偏好优化（DPO）相结合，我们旨在在迭代训练周期中增强代理的决策能力。所提出的框架解决了数据稀缺问题，并推动了目标导向领域更细致、更有效的对话系统的发展。

    arXiv:2608.12062v1 Announce Type: cross  Abstract: Developing dialogue systems capable of engaging in multi-turn, goal-oriented conversations remains a significant challenge, especially in specialized domains with limited data. This research proposes a novel framework called Preference Tree Optimization (PTO), designed to iteratively improve agent models in such dialogue systems, by generating preference data using a method called Preference Tree with Look-Ahead. Focusing on Motivational Interviewing (MI) -- a counseling technique aimed at facilitating behavioral change -- we leverage virtual patients and an oracle evaluator to simulate conversations and generate rich preference datasets. By combining this method with Direct Preference Optimization (DPO), we aim to enhance the agent's decision-making capabilities over iterative training cycles. The proposed framework addresses data scarcity and advances the development of more nuanced and effective dialogue systems in goal-oriented dom
    
[^119]: REAP：面向大语言模型闭卷知识库构建的关系感知引导与解析方法

    REAP: Relation-Aware Elicitation and Parsing for Closed-Book Knowledge Base Construction from LLMs

    [https://arxiv.org/abs/2608.10963](https://arxiv.org/abs/2608.10963)

    REAP系统通过结构化思维链推理、关系特定查询策略与空集门控机制的组合，在闭卷、无微调且参数量不超过32B的约束下，从大语言模型中提取参数化知识构建知识库，宏平均F1达到0.62。

    

    我们提出了REAP系统，用于参加AKBC 2026共享任务，该任务要求在闭卷设置下从语言模型中构建知识库，参数量预算不超过32B，且不允许对模型进行微调。我们的系统结合了结构化思维链推理、面向特定关系的查询策略，以及基于推理的空集门控机制，以引导模型输出参数化知识，随后直接将其提取为有效的JSON数组。在测试集上，该系统基于Mistral-Small-24B-Instruct-2501模型构建，取得了0.62的宏平均F1分数，其中在countryLandBordersCountry（F1 = 0.95）、companyTradesAtStockExchange（F1 = 0.73）和hasArea（F1 = 0.77）等关系上表现尤为突出。我们的代码已在 https://github.com/yammdd/AKBC-Shared-Task-2026 公开。

    arXiv:2608.10963v2 Announce Type: replace  Abstract: We present the REAP system for the AKBC Shared Task 2026 on constructing knowledge bases from language models in a closed-book setting, subject to a budget of at most 32B parameters and no model fine-tuning. Our system combines structured chain-of-thought reasoning, relation-specific query strategies, and a reasoning-based empty-set gate to elicit parametric knowledge, followed by direct extraction into valid JSON arrays. On the test set, the system, built on the Mistral-Small-24B-Instruct-2501 model, achieves a macro-F1 score of 0.62, with particularly strong results on countryLandBordersCountry (F1 = 0.95), companyTradesAtStockExchange (F1 = 0.73), and hasArea (F1 = 0.77). Our code is publicly available at https://github.com/yammdd/AKBC-Shared-Task-2026.
    
[^120]: GPTKB 2.0：浏览、查询和审计消歧后的LLM衍生知识库

    GPTKB 2.0: Browsing, Querying, and Auditing a Disambiguated LLM-Derived Knowledge Base

    [https://arxiv.org/abs/2608.06992](https://arxiv.org/abs/2608.06992)

    GPTKB 2.0 是一个交互式网络演示系统，展示了从大语言模型构建的大规模消歧知识库（含3840万三元组和160万实体），在构建过程中通过上下文引导消歧区分同名异义、合并同义提及，并支持实体浏览、事实溯源审计、SPARQL与自然语言查询及实体链接。

    

    我们展示了一个用于探索从大语言模型（LLM）中物化的大规模消歧知识库（KB）的网络演示系统。GPTKB 2.0 包含 3840 万条三元组，覆盖 160 万个规范实体，以及 20.76 万个整合关系和 6.6 万个整合类。与以往主要通过表面字符串识别实体的 LLM 衍生知识库不同，GPTKB 2.0 在递归式知识库构建过程中执行基于上下文引导的消歧，在事实抽取的同时区分同名异义实体并合并同义提及。该演示使这一构建过程可被检查：用户可以浏览实体、在知识库中跟踪链接，并审计单条事实的来源，包括表面形式、候选匹配、源三元组以及消歧决策。该界面还支持结构化 SPARQL 查询、将自然语言问题转换为 SPARQL 进行查询，以及将用户提供的文本中的实体链接到 GPTKB 2.0 的规范条目。

    arXiv:2608.06992v2 Announce Type: replace-cross  Abstract: We present a web demo for exploring a large-scale disambiguated knowledge base (KB) materialized from a large language model (LLM). GPTKB 2.0 contains 38.4M triples over 1.6M canonical entities, together with 207.6K consolidated relations and 66K consolidated classes. Unlike prior LLM-derived knowledge bases that largely identify entities by surface strings, GPTKB 2.0 performs context-guided disambiguation during recursive KB construction, separating homonyms and merging synonymous mentions as facts are elicited. The demo makes this process inspectable: users can browse entities, follow links across the KB, and audit the provenance of individual facts, including surface forms, candidate matches, source triples, and disambiguation decisions. The interface further supports structured SPARQL queries, natural-language questions translated to SPARQL, and entity linking from user-provided text to canonical GPTKB 2.0 entries. GPTKB 2.
    
[^121]: 从大语言模型直接构建消歧知识库

    Direct Construction of Disambiguated Knowledge Bases from Large Language Models

    [https://arxiv.org/abs/2608.03729](https://arxiv.org/abs/2608.03729)

    提出GPTKB 2.0方法，通过对实体、关系和类别的即时消歧机制，直接从大语言模型构建了首个百万级规模的消歧知识库，包含超过100万个实体和3840万条三元组。

    

    自动化知识库构建（AKBC）是自然语言处理领域的一项核心任务，近期有研究提出直接从大语言模型（LLM）生成知识库，将模型本身视为知识来源。然而，大语言模型本身并不具备实体的表示形式，这导致知识库中出现重复条目以及实体混淆的问题。我们提出了GPTKB 2.0，这是一种直接从大语言模型构建消歧知识库的方法论。GPTKB 2.0 引入了对实体、关系和类别的即时消歧机制，并经过精心设计以同时满足可扩展性和消歧准确性两方面的要求。我们分析了核心设计决策，并刻画了准确性、规模与成本之间的权衡关系。我们大规模地执行了GPTKB 2.0，得到了一个包含超过100万个消歧实体和3840万条三元组的实体化知识库。这是首个对实体、关系和类别进行显式内部规范化的百万级规模大语言模型原生知识库。

    arXiv:2608.03729v3 Announce Type: replace-cross  Abstract: Automated Knowledge Base Construction (AKBC) is a core NLP task, and recent work proposes generating knowledge bases directly from large language models (LLMs), treating the model itself as the knowledge source. However, LLMs natively possess no representation of entities, leading to duplicate entries as well as conflations. We propose GPTKB 2.0, a methodology for constructing disambiguated KBs directly from LLMs. GPTKB 2.0 incorporates on-the-fly disambiguation of entities, relations and classes, and is meticulously designed to satisfy both scalability and disambiguation accuracy. We analyze the central design decisions and characterize the trade-offs between accuracy, scale, and cost. We execute GPTKB 2.0 at scale, obtaining a materialized KB containing over 1M disambiguated entities and 38.4M triples. This represents the first million-scale LLM-native KB with explicit internal canonicalization of entities, relations, and cla
    
[^122]: PGMem：面向终身个性化智能体的紧耦合人格-记忆图

    PGMem: Tightly Coupled Persona-Memory Graph for Lifelong Personalized Agents

    [https://arxiv.org/abs/2608.01708](https://arxiv.org/abs/2608.01708)

    PGMem通过类型化溯源边和证据边将事件与人格节点紧耦合为异构图，使每个人格信号都可追溯到支持或修正它的事件，解决了记忆与人格脱节的问题，并在三个基准上持续超越现有记忆基线。

    

    长期个性化对话智能体必须随着用户人格的演变来持续追踪其偏好。现有记忆系统能够很好地组织过去的事件，但将人格存储为扁平化的档案，与支撑这些人格的事件相互分离。这种松散耦合导致了“记忆-人格有效性鸿沟”和“人格感知检索鸿沟”两大问题。我们提出PGMem，一种异构的人格-记忆图，通过带类型的溯源边和证据边将事件节点与人格节点连接起来，使每个人格信号都可追溯到支持或修正它的事件。在检索阶段，PGMem从与查询相关的种子节点出发进行扩展，并依据证据有效性对信号进行排序。在使用小型语言模型骨干的三个基准测试中，PGMem始终优于基于摘要的、人格感知的、图结构化的以及智能体式的记忆基线方法，且随着上下文的增长性能不断提升。PGMem的源代码已在 https://github.com/wonjunchoi23/pgmem/ 开源。

    arXiv:2608.01708v2 Announce Type: replace  Abstract: Long-term personalized dialogue agents must track user preferences as their personas evolve. Existing memory systems organize past events well, but store personas as flat profiles detached from the events that justify them. This loose coupling leads to the memory-persona validity gap and the persona-aware retrieval gap. We propose PGMem, a heterogeneous persona-memory graph that connects event and persona nodes through typed provenance and evidence edges, keeping each persona signal traceable to the events that support or revise it. At retrieval time, PGMem expands from query-relevant seeds and ranks signals by evidential validity. Across three benchmarks with small language model backbones, PGMem consistently outperforms summary-based, persona-aware, graph-structured, and agentic memory baselines, and improves performance as the context grows. The source code of PGMem is available at https://github.com/wonjunchoi23/pgmem/
    
[^123]: VLM 是在阅读还是改写？论视觉语言模型中的转录忠实性

    Do VLMs Read or Rewrite? On Transcription Faithfulness in Vision-Language Models

    [https://arxiv.org/abs/2607.21617](https://arxiv.org/abs/2607.21617)

    本文提出 FaithC4 多语言扰动基准，揭示视觉语言模型在面对不完美文本时常将其“改写”为更合理形式而非忠实转录，其中通用 VLM 在扰动下词错率退化最严重，而传统 OCR 最为稳健。

    

    视觉语言模型（VLM）正越来越多地被用于替代传统 OCR 流水线进行文档理解。本文表明，它们并不总是充当忠实的转录者：当文本不完美时，它们往往倾向于将其改写成更“合理”的形式——这种行为是现有的干净文本 OCR 基准无法检测到的。我们提出了 FaithC4，这是一个包含 1,455 份单页文档（涵盖英语、中文、韩语）的多语言扰动基准，包含三类扰动：乱序、随机替换和视觉相似替换。我们利用该基准评估了 15 个系统，涵盖通用 VLM、OCR 专用 VLM 和传统 OCR 流水线。这三类系统在扰动下的词错率（WER）退化程度存在差异：通用 VLM 退化高达 6.9 个百分点，OCR 专用 VLM 退化 0.1-3.4 个百分点，而传统 OCR 在英语上退化不足 0.8 个百分点。通过对 Qwen3-VL-4B 进行逐层探测，我们识别出一种一致性（摘要在此处被截断）。

    arXiv:2607.21617v2 Announce Type: replace  Abstract: Vision Language Models (VLMs) are increasingly used in place of traditional OCR pipelines for document understanding. In this paper, we show they do not always act as faithful transcribers: when text is imperfect, they often tend to rewrite it into a more plausible form - a behavior that clean-text OCR benchmarks cannot detect. We introduce FaithC4, a multilingual perturbation benchmark of 1,455 single-page documents (English, Chinese, Korean) with three perturbation families: scramble, random substitution, and visually similar substitution. We use the benchmark to evaluate 15 systems spanning general-purpose VLMs, OCR-specialized VLMs, and traditional OCR pipelines. These three categories differ in WER degradation under perturbation: general-purpose VLMs degrade by up to 6.9 points, OCR-specialized VLMs by 0.1-3.4 points, and traditional OCR by less than 0.8 points on English. Probing Qwen3-VL-4B layer-by-layer, we identify a consis
    
[^124]: 面向少步生成的多掩码扩散语言模型

    Multi-Mask Diffusion Language Models for Few-Step Generation

    [https://arxiv.org/abs/2607.19686](https://arxiv.org/abs/2607.19686)

    提出多掩码扩散模型MultiMDM，通过在前向过程中保留掩码结构、在反向过程中先预测指定掩码再精炼为干净词元的起草能力，实现高质量的少步文本生成。

    

    arXiv:2607.19686v3 公告类型：替换。掩码扩散模型是一类很有前景的语言生成器，但实现高质量的少步生成仍然具有挑战性。在MDM中，所有前向轨迹都会坍缩到单一的全掩码状态，因此没有为一致性风格的少步生成保留终端熵。虽然最近基于均匀状态扩散的少步替代方案避免了这种退化问题，但与MDM相比，将干净词元与噪声区分开来变得更加困难，这通常会损害建模质量和训练效率。在这项工作中，我们提出了多掩码扩散模型，它为少步生成保留了掩码结构。在前向过程中，每个干净词元首先被推向一个指定的掩码，然后逐渐在掩码集合上混合。因此，反向过程具备了起草能力，即先将指定掩码预测出来，再将其精炼为干净词元。我们推导了闭式ELBO训练目标……

    arXiv:2607.19686v3 Announce Type: replace  Abstract: Masked diffusion models (MDMs) are a promising family of language generators, but achieving high-quality few-step generation remains challenging. In MDMs, all forward trajectories collapse to a single fully masked state, leaving no terminal entropy for consistency-style few-step generation. While recent few-step alternatives based on uniform-state diffusion avoid this degeneracy, it becomes harder to distinguish clean tokens from noise than MDMs, which usually harms modeling quality and training efficiency. In this work, we propose a multi-mask diffusion model (MultiMDM) that preserves the masking structure towards few-step generation. In the forward process, each clean token is first pushed towards a designated mask and then gradually mixes over the mask set. As a result, the backward process has a drafting capability by predicting a designated mask before refining to a clean token. We derive a closed-form ELBO training objective fo
    
[^125]: 源偏移下什么能迁移？定义、示例与微调在气候披露分类中的应用

    What Transfers Under Source Shift? Definitions, Examples, and Fine-Tuning for Climate Disclosure Classification

    [https://arxiv.org/abs/2607.17952](https://arxiv.org/abs/2607.17952)

    该论文将气候披露分类重构为跨源适应问题，通过在十一个开源与闭源LLM上评估定义、示例和微调三种策略，发现尽管所有策略平均均能带来跨源收益，但源内表现最强的策略（如相似度检索和LoRA微调）并不一定是源偏移场景下最有效的策略。

    

    气候披露分类是分析企业气候披露的一项基础任务，然而此类披露出现在多种不同的来源中——年报、新闻稿和财报电话会议——这些来源在长度、目的和写作风格上各不相同。现有评估大多在单一来源内进行，这使得常见的LLM适应策略在源偏移情况下是否仍然有效这一问题悬而未决。我们将气候披露分类重新构建为一个跨源适应问题，并在十一个开源与闭源LLM上研究了三种广泛使用的适应策略——定义、示例和微调，使用了两个共享相同标签空间但来自不同来源的语料库。我们发现所有策略平均而言都能带来正向的跨源收益，但源内最强的策略并非跨源最强的策略：基于相似度的检索和LoRA微调在源内获益最多……

    arXiv:2607.17952v2 Announce Type: replace  Abstract: Climate disclosure classification is a fundamental task for analysing corporate climate disclosures, yet such disclosures appear in many different sources -- annual reports, press releases, and earnings calls -- that differ in length, purpose, and writing style. Existing evaluations are mostly conducted within a single source, leaving open whether common LLM adaptation strategies remain effective under source shift. We reframe climate disclosure classification as a cross-source adaptation problem and study three widely used adaptation strategies -- definitions, examples, and fine-tuning -- across eleven open- and closed-source LLMs, using two corpora that share the same label space but come from different sources. We find that all strategies bring positive cross-source gains on average, but the strongest in-source strategies are not the strongest cross-source ones: similarity-based retrieval and LoRA fine-tuning gain most in-source b
    
[^126]: 持久稀疏自编码器：在语言模型表示中学习特征特定的时间尺度

    Persistent Sparse Autoencoders: Learning Feature-Specific Timescales in Language Model Representations

    [https://arxiv.org/abs/2607.17117](https://arxiv.org/abs/2607.17117)

    该论文提出持久稀疏自编码器，通过为每个特征学习一个持久性系数，使稀疏自编码器能够仅凭重构目标从语言模型激活中自动学习特征特定的时间尺度，同时保持高质量的重构效果。

    

    稀疏自编码器（SAE）将语言模型的激活分解为稀疏特征，然而这些模型传统上对每个词元进行独立编码，无法揭示跨序列持续存在的信息。我们首先证明，时间持久性可以在标准SAE特征中自然涌现：当一个特征激活后，隐藏状态会保持与其方向对齐，且过去的激活有助于重构后续的隐藏状态。这种持续性在不用特征之间的差异很大。因此，我们提出了持久稀疏自编码器（Persistent SAEs），这是标准SAE的一种扩展，它为每个特征学习一个持久性系数，使模型能够仅通过重构任务学习特征特定的时间尺度。我们的实验表明，持久SAE在学习一系列时间尺度的同时保持了有竞争力的重构质量：短时间尺度（快速）特征保持局部可解释性，而长时间尺度（……（摘要在此处被截断）

    arXiv:2607.17117v2 Announce Type: replace-cross  Abstract: Sparse autoencoders (SAEs) decompose language model activations into sparse features, yet these models traditionally encode each token independently, failing to expose information that persists across a sequence. We first show that temporal persistence can naturally emerge in standard SAE features: after a feature activates, the hidden state remains aligned with its direction, and past activations help reconstruct later hidden states. How long this lasts varies widely across features. We therefore introduce Persistent Sparse Autoencoders (Persistent SAEs), an extension of standard SAEs that learns a persistence coefficient for each feature, allowing the model to learn feature-specific timescales from reconstruction alone. Our experiments show that Persistent SAEs retain competitive reconstruction quality while learning a spectrum of timescales: short-timescale (fast) features stay locally interpretable, whereas long-timescale (
    
[^127]: LLM水印作为大数据溯源：面向部署的系统性综述

    LLM Watermarking as Big Data Provenance: A Deployment-Oriented Systematization

    [https://arxiv.org/abs/2607.10103](https://arxiv.org/abs/2607.10103)

    本文将LLM水印系统化为大数据生态系统的溯源基础设施，沿插入点、验证权限、运行状态和转换威胁模型四个部署维度对现有方法进行分类，并分析部署选择对可靠性、安全性和可扩展性的影响。

    

    随着大语言模型（LLM）的广泛部署，其输出内容可以被大规模复制、转换和重新分发，而缺乏可靠的来源证据，这给信任、问责、知识产权（IP）保护以及高风险决策带来了风险。LLM水印通过在生成过程中或生成后向文本中嵌入可检测的信号来解决这一问题。然而，现有方法在设计假设、威胁模型和评估标准上各不相同，而水印放置位置、检测权限和密钥管理等部署选择会影响可靠性、安全性和可扩展性。本文将LLM水印系统化为面向大规模数据生态系统的溯源基础设施。我们沿四个部署维度对现有方法进行组织：插入点、验证权限、运行状态和转换威胁模型，并将其与大数据的Volume（容量）、Ve（原文截断）……

    arXiv:2607.10103v2 Announce Type: replace-cross  Abstract: As large language models (LLMs) become widely deployed, their outputs can be copied, transformed, and redistributed at scale without reliable evidence of origin, creating risks for trust, accountability, intellectual property (IP) protection, and high-stakes decision-making. LLM watermarking addresses this problem by embedding detectable signals into text during or after generation. However, existing methods vary in design assumptions, threat models, and evaluation criteria, while deployment choices such as watermark placement, detection authority, and key management affect reliability, security, and scalability. This paper systematizes LLM watermarking as provenance infrastructure for large-scale data ecosystems. We organize existing approaches along four deployment dimensions: insertion point, verification authority, operational state, and transformation threat model, and relate them to the big data requirements of Volume, Ve
    
[^128]: 符号分支重复惩罚中的规范依赖性与结构化输出损坏：跨模型、推理栈及替代重复控制方法的测量

    Gauge dependence and structured-output corruption in sign-branched repetition penalties: measurements across models, inference stacks, and alternative repetition controls

    [https://arxiv.org/abs/2607.09791](https://arxiv.org/abs/2607.09791)

    该论文揭示了主流推理引擎中的符号分支乘法重复惩罚依赖于 logit 任意零点（规范选择），导致惩罚操作缺乏良好定义且在不同模型上效果各异，并会使 JSON 结构化输出的有效率从 97% 骤降至 23%，同时提出了减法式与归一化等不受规范影响的替代方案。

    

    部署于整个大语言模型推理生态系统中（HuggingFace、vLLM、llama.cpp 以及十几个其他推理引擎）的乘法重复惩罚会根据每个原始 logit 的符号进行分支运算（正数除以 theta，负数乘以 theta）。但 softmax 对于给所有 logit 加上一个常数是不变的，因此模型的 logit 零点是任意的（一种规范/规范自由度选择），而符号分支却读取了这个零点。由此产生两个可测量的后果：(1) 该惩罚没有良好定义：对模型的 logit 进行常数重新中心化在 theta=1 时被证明是无操作（no-op），但在常规设置 theta=1.3 下，它会改变 58-96% 的贪心解码 token，而减法式惩罚和归一化惩罚则不改变任何 token；真实的模型检查点处于差异巨大的零点上，因此固定的 repetition_penalty 在每个模型上实际上是不同的操作。(2) 它会破坏结构化输出：在 200 个真实世界的 JSON 模式（schema）上，theta=1.3 将有效且符合模式的输出比例从 97% 降至 23%。应用该惩罚……（摘要在此处截断）

    arXiv:2607.09791v2 Announce Type: replace-cross  Abstract: The multiplicative repetition penalty shipped across the LLM inference ecosystem (HuggingFace, vLLM, llama$.$cpp, and a dozen further engines) branches on the sign of each raw logit (divide positives by theta, multiply negatives). But the softmax is unchanged by adding a constant to every logit, so a model's logit zero-point is arbitrary (a gauge choice), and the sign-branch reads it. Two measurable consequences follow. (1) The penalty is not well-defined: re-centering a model's logits by a constant is a provable no-op at theta=1, yet at a routine theta=1.3 it changes 58-96% of greedy tokens, while subtractive and normalized penalties change none; real checkpoints sit at widely different zero-points, so a fixed repetition_penalty is a different operation on every model. (2) It corrupts structured output: on 200 real-world JSON schemas, theta=1.3 drops the rate of valid, schema-conformant output from 97% to 23%. Applying the pen
    
[^129]: 所见即所得：面向图表到代码生成的观察对齐监督

    What You See Is What You Get: Observation-Aligned Supervision for Chart-to-Code Generation

    [https://arxiv.org/abs/2607.04726](https://arxiv.org/abs/2607.04726)

    论文揭示了图表到代码生成训练中存在的四类潜在变量与观察图像不匹配问题，并提出观察对齐监督方法，用视觉上可约束的量替换潜在变量作为监督目标。

    

    图表到代码生成通常通过对参考绘图脚本进行监督微调来训练，这隐式地将黄金代码视为完全可观察的目标。然而，许多图表程序包含无法从渲染图像中唯一恢复的潜在变量。我们在五种图表类型中识别出这种潜在变量与观察不匹配问题的四种形式：聚合导致的不匹配，即原始样本被简化为箱线图统计量或直方图分箱统计；归一化导致的不匹配，即饼图中绝对尺度被移除；投影导致的不匹配，即三维信息在二维渲染中丢失；以及水平集导致的不匹配，即标量场只能通过选定的等高线被观察。这些不匹配引入了目标歧义，并要求模型生成图像本身无法支持的信息。我们提出观察对齐监督方法，用视觉上可约束的量来替换潜在变量。

    arXiv:2607.04726v4 Announce Type: replace  Abstract: Chart-to-code generation is commonly trained through supervised fine-tuning on reference plotting scripts, implicitly treating the gold code as a fully observable target. However, many chart programs contain latent variables that cannot be uniquely recovered from the rendered image. We identify this latent-observation mismatch in four forms across five chart types: aggregation-induced mismatch, where raw samples are reduced to box statistics or histogram bin masses; normalization-induced mismatch, where absolute scale is removed in pie charts; projection-induced mismatch, where 3D information is lost through 2D rendering; and level-set-induced mismatch, where a scalar field is observable only through selected contour lines. These mismatches introduce target ambiguity and require models to generate information unsupported by the image. We propose Observation-Aligned Supervision, which replaces latent variables with visually constraine
    
[^130]: LLM作为评判者能在智能体场景中可靠地验证评分标准吗？

    Can LLM-as-a-Judge Reliably Verify Rubrics in Agentic Scenarios?

    [https://arxiv.org/abs/2606.29920](https://arxiv.org/abs/2606.29920)

    该论文提出了RuVerBench——首个用于评估LLM作为评判者在智能体场景（深度研究和智能体编程）中验证评分标准可靠性的基准，包含2,458个人工标注实例，并发现即使最先进的模型仍存在显著的可靠性缺陷。

    

    基于评分标准的评分已成为模型评估中广泛使用的范式，通常采用LLM作为评判者（LaaJ）来进行评分标准打分。然而，LaaJ在评分标准评分方面的可靠性仍未得到充分探索。这一担忧在智能体场景中尤为突出，因为冗长且复杂的输出进一步挑战了可靠评分。为解决这一问题，我们对LaaJ在评分标准验证方面的可靠性进行了系统的元评估。我们介绍了RuVerBench，这是首个用于评估智能体场景中LaaJ评分标准验证可靠性的基准。RuVerBench涵盖两个主流的智能体领域——深度研究和智能体编程，共包含2,458个实例，每个实例包含一个模型生成的输出、一个评分标准，以及一个指示该输出是否满足评分标准的人工标注标签。使用RuVerBench，我们评估了众多前沿LLM，发现即使是最先进的模型也能取得强劲的表现，但仍然存在显著的

    arXiv:2606.29920v2 Announce Type: replace  Abstract: Rubric-based scoring has become a widely used paradigm in model evaluation, typically with LLM-as-a-Judge (LaaJ) for rubric scoring. However, the reliability of LaaJ for rubric scoring remains underexplored. This concern is especially pronounced in agentic scenarios, where long, complex outputs further challenge reliable scoring. To address this, we conduct a systematic meta-evaluation of LaaJ reliability for rubric verification. We introduce RuVerBench, the first benchmark for assessing LaaJ reliability in rubric verification for agentic scenarios. RuVerBench covers two prevalent agentic domains, deep research and agentic coding, with 2,458 instances, each containing a model-generated output, a rubric, and a human-annotated label indicating whether the output satisfies the rubric. Using RuVerBench, we evaluate numerous frontier LLMs and find that even the most advanced models achieve strong performance but still exhibit substantial 
    
[^131]: SABER-Math：面向数学信息检索评估的自动化基准

    SABER-Math: Automated Benchmark for Information Retrieval Evaluation in Mathematics

    [https://arxiv.org/abs/2606.29894](https://arxiv.org/abs/2606.29894)

    该论文提出了首个无需专家标注、完全自动化的数学信息检索评估基准SABER-Math，它从28.3万道高中数学题出发自动构建具有挑战性的重排序任务，以克服现有基准无法捕捉细粒度数学相关性的问题。

    

    随着智能体AI系统处理越来越复杂的数学任务，它们越来越依赖信息检索（IR）来搜索问题数据库、定理库和教育资源。然而，选择合适的检索器仍然很困难，因为无法直接将其对下游性能的影响隔离开来加以评估。另一方面，现有的检索专用基准往往无法捕捉细粒度的数学相关性，从而错误地惩罚相关文档。我们通过引入SABER-Math来填补这一空白，这是首个无需专家标注、完全自动化的数学信息检索评估基准。SABER-Math从28.3万道带解答的高中数学题目出发，通过三个步骤构建具有挑战性的重排序任务：(i) 首先，大语言模型为每道题目提取简洁的解题摘要和数学主题；(ii) 然后，利用基于本体主题和词汇解答相似性的方法为每个查询发现相关文档……（原文摘要在此处截断）

    arXiv:2606.29894v2 Announce Type: replace-cross  Abstract: As agentic AI systems tackle more complex mathematical tasks, they increasingly rely on information retrieval (IR) to search problem databases, theorem libraries, and educational resources. However, choosing the right retriever remains difficult, as it is infeasible to directly isolate its effect on downstream performance. On the other hand, existing retrieval-specific benchmarks often fail to capture fine-grained mathematical relevance, penalizing relevant documents. We address this gap by introducing SABER-Math, the first fully automated benchmark for evaluating mathematical IR without expert annotation. Starting from 283K high-school-level math problems with solutions, SABER-Math builds challenging reranking tasks in three steps: (i) first, LLMs extract concise solution summaries and mathematical topics for each problem; (ii) then, per-query relevant documents are discovered using ontology topic-based and lexical solutions-s
    
[^132]: AdaMem：通过自适应记忆策略学习个性化智能体应记住的内容

    AdaMem: Learning What to Remember with Adaptive Memory Policies for Personalized Agents

    [https://arxiv.org/abs/2606.21144](https://arxiv.org/abs/2606.21144)

    提出AdaMem框架，利用基于用户反馈持续更新的自适应自然语言记忆策略，根据不同交互情境个性化控制LLM智能体的记忆写入内容，并构建AdaMem-Bench基准进行验证。

    

    长期记忆系统使LLM智能体能够在单一上下文窗口之外保留信息，但大多数系统专注于在提取后存储和检索事实，而写入决策的规范却相对不足。什么内容值得记忆可能取决于用户当前的任务、话题、活动或交互对象，而统一的提取方法会在这些不同情境中应用单一的重要性标准。我们将这一挑战形式化为偏好条件化的写入控制，并提出了AdaMem，它利用自适应的自然语言记忆策略来个性化智能体写入记忆的内容。每个策略代表用户在特定交互情境下的记忆偏好，通过定期反馈进行更新，并控制后续的记忆写入。我们在AdaMem-Bench中对这一循环进行评估，该基准在五个为期十周的故事中，为六个并发交互角色分配了不同的记忆偏好。在两种提取模式下……

    arXiv:2606.21144v2 Announce Type: replace-cross  Abstract: Long-term memory systems allow LLM agents to preserve information beyond a single context window, but most systems focus on storing and retrieving facts after extraction, leaving the write decision under-specified. What deserves memory can depend on the user's current task, topic, activity, or interaction partner, while uniform extraction applies one notion of importance across these different situations. We formulate this challenge as preference-conditioned write control and introduce AdaMem, which uses adaptive natural-language Memory Policies to personalize what an agent writes to memory. Each policy represents the user's memory preference for a particular interaction context, is updated from periodic feedback, and controls subsequent memory writing. We evaluate this loop in AdaMem-Bench, which assigns different memory preferences to six concurrent interaction personas across five ten-week stories. Across two extraction mode
    
[^133]: VTOS：通过协同搜索解决方案与观察者来学习编排视觉工具

    VTOS: Learning to Orchestrate Vision Tools by Co-Searching Solutions and Observers

    [https://arxiv.org/abs/2606.20728](https://arxiv.org/abs/2606.20728)

    该论文提出VTOS框架，通过联合搜索组合视觉工具的可执行解决方案程序与能诊断失败模式并生成可操作反馈的观察者程序，实现自适应的视觉工具编排，克服了现有视觉编程智能体固定流水线在密集物体、遮挡、小目标和领域偏移下的脆弱性。

    

    开放词汇检测器、分割模型和后处理算子等视觉基础工具是计算机视觉的强大构建模块，但其有效性在很大程度上取决于如何对它们进行编排：使用哪些工具、以何种顺序、采用什么参数，以及在何种视觉条件下使用。现有的视觉编程智能体通常生成固定的解决方案流水线，导致它们在面对密集物体、遮挡、小目标和领域偏移时表现脆弱。我们提出了VTOS（视觉工具编排搜索），这是一个通过解决方案与观察者联合搜索来实现自适应视觉工具编排的框架。VTOS协同搜索可执行的解决方案程序（组合Grounding DINO、SAM、NMS和切片检测等视觉工具），以及观察者程序（用于诊断候选解决方案、识别失效模式并生成可操作的反馈）。这些观察结果被累积在一个共享的视觉……（摘要在此处截断）

    arXiv:2606.20728v2 Announce Type: replace-cross  Abstract: Vision foundation tools such as open-vocabulary detectors, segmentation models, and post-processing operators are powerful building blocks for computer vision, but their effectiveness depends heavily on how they are orchestrated: which tools are used, in what order, with what parameters, and under what visual conditions. Existing visual-programming agents typically generate a fixed solution pipeline, making them brittle under dense objects, occlusion, small targets, and domain shift. We introduce VTOS (Vision Tools Orchestration Search), a framework for adaptive visual tool orchestration through joint solution-observer search. VTOS co-searches executable solution programs that compose vision tools such as Grounding DINO, SAM, NMS, and slice-and-detect, together with observer programs that diagnose candidate solutions, identify failure modes, and generate actionable feedback. These observations are accumulated in a shared Vision
    
[^134]: 最近发展区策略优化：教师置于提示中，而非梯度中

    Zone of Proximal Policy Optimization: Teacher in Prompts, Not Gradients

    [https://arxiv.org/abs/2606.18216](https://arxiv.org/abs/2606.18216)

    该论文提出ZPPO，受维果茨基最近发展区理论启发，将教师模型的帮助置于提示词中而非策略梯度中，通过为难题重新构造提示（如将正确教师回答纳入二选一问题），使小型学生模型能够基于自身rollout进行强化学习，从而规避知识蒸馏在小模型上的模仿脆弱性以及向梯度注入教师回答所导致的漂移问题。

    

    知识蒸馏能够将教师模型的能力迁移给小型学生模型，但在“小学生”场景下十分脆弱：迫使学生去模仿远大于自身的教师模型的logits，会使其过度集中于教师分布中最尖锐的众数，从而损害其在训练语料之外的基准任务族上的泛化能力。强化学习（RL）通过在学生自身生成的rollouts上进行训练，避免了logit模仿的问题。然而，当所有rollout都失败时——产生零优势并被静默丢弃——将更强教师的回答注入策略梯度会破坏在策略假设并引起漂移。受维果茨基“最近发展区”理论的启发，我们提出了最近发展区策略优化（ZPPO），它将教师保留在提示词中而非策略梯度中。对于难题，ZPPO会构造两个重新表述的提示。其中一种包含候选的二选一问题（BCQ）将一个正确的教师回答与……（原文摘要在此处截断）

    arXiv:2606.18216v2 Announce Type: replace  Abstract: Knowledge distillation transfers a teacher's competence to a small student but is brittle in the small-student regime: forcing the student to imitate logits from a much larger teacher concentrates it on the teacher's sharpest modes, hurting generalization on benchmark families beyond the training corpus. Reinforcement learning (RL) avoids logit imitation by training on the student's own rollouts. However, on questions where every rollout fails-yielding zero advantage and being silently discarded-injecting a stronger teacher's response into the policy gradient breaks the on-policy assumption and induces drift. We introduce Zone of Proximal Policy Optimization (ZPPO), inspired by Vygotsky's zone of proximal development, which keeps the teacher inside the prompt rather than the policy gradient. On hard questions, ZPPO constructs two reformulated prompts. A Binary Candidate-included Question (BCQ) pairs one correct teacher response with 
    
[^135]: EComAgentBench：在具有分布式隐藏意图的长程任务上对购物智能体进行基准测试

    EComAgentBench: Benchmarking Shopping Agents on Long-Horizon Tasks with Distributed Hidden Intent

    [https://arxiv.org/abs/2606.17698](https://arxiv.org/abs/2606.17698)

    该论文提出了EComAgentBench基准，通过662个基于真实亚马逊商品的任务，将购物者隐藏意图分散于可见查询、工具访问的用户档案和澄清对话中，用以评估LLM购物智能体在长程任务中挖掘隐含需求、验证商品并归因失败的能力。

    

    随着基于大语言模型（LLM）的购物智能体进入生产环境，现有基准测试无法捕捉购物者需求到达智能体的方式：有的需求在查询中隐含表达，有的记录在用户档案中，还有的只有在提出恰当问题时才会显现。那些预先暴露完整意图、且仅评估最终选择的基准测试，既无法构成这种长程挑战，也无法解释智能体遗漏了哪项需求。为填补这一空白，我们提出了EComAgentBench，一个基于真实亚马逊产品和评论构建的包含662个任务的基准测试。每个任务将这些需求分散在可见查询、需要工具调用的用户档案以及预设的澄清对话中；智能体必须挖掘隐藏意图，对照商品属性和评论证据验证候选商品，并在100次工具调用内确定单一商品。此外，带有类型标注和来源标记的评分标准对每个任务进行评估，将每次失败归因于某项具体需求及其来源。任务构建流程自动化且可靠，每个答案……

    arXiv:2606.17698v3 Announce Type: replace  Abstract: As LLM-based shopping agents enter production, existing benchmarks fail to capture how a shopper's requirements arrive: stated implicitly in the query, recorded in a profile, or revealed only when the right question is asked. Benchmarks that expose full intent upfront and grade only the final choice can neither pose this long-horizon challenge nor explain which requirement an agent missed. To address this gap, we introduce EComAgentBench, a benchmark of 662 tasks grounded in real Amazon products and reviews. Each task scatters these requirements across a visible query, a tool-gated profile, and scripted clarification; an agent must uncover hidden intent, verify candidates against attributes and review evidence, and commit to a single product within 100 tool calls. Moreover, typed, source-tagged rubrics grade every task, attributing each failure to a requirement and its source. Construction is automated yet reliable, with every answer
    
[^136]: 指称交流中大型视觉语言模型的隐式与显式提示策略对比

    Implicit vs. Explicit Prompting Strategies for LVLMs in Referential Communication

    [https://arxiv.org/abs/2606.17372](https://arxiv.org/abs/2606.17372)

    该研究通过控制任务差异并对比显式与隐式两种提示方式，发现大型视觉语言模型仅在显式提示下才能像人类一样协调生成高效指称表达，而无法从隐式提示中自主推断出交流效率的需求，揭示了人类与AI交流能力的关键差异。

    

    最近有两项研究就大型视觉语言模型（LVLMs）能否像人类一样在高效指称表达上进行协调这一问题上，得出了看似相互矛盾的结论。我们在控制这两项研究之间任务差异的同时，直接比较了它们各自的提示方式。我们重复验证了如下发现：当被明确提示时，模型能够协调生成高效的指称表达，这表明其他任务差异并非导致结果分歧的原因。然而，我们还发现，同样的模型在更隐式的提示下无法推断出对交流效率的需求，这突显了人类与AI系统在交流方式上的关键差异。

    arXiv:2606.17372v3 Announce Type: replace-cross  Abstract: Two recent studies \citep{jones2026llms, zeng2026lvlms} reach apparently contradictory conclusions about whether large vision-language models (LVLMs) can coordinate similarly to humans on efficient referring expressions. We control for task differences between the studies while directly comparing their prompting styles. We replicate the finding that models can coordinate efficient referring expressions when \textit{explicitly} prompted to do so, suggesting that other task differences are not responsible for divergent results. However, we also find that the same models fail to infer the need for communicative efficiency from a more \textit{implicit} prompt, highlighting critical differences between how humans and AI systems communicate.
    
[^137]: 大型语言模型总是讲述相同的故事吗？

    Do Large Language Models Always Tell The Same Stories?

    [https://arxiv.org/abs/2606.17350](https://arxiv.org/abs/2606.17350)

    研究发现大型语言模型生成的故事彼此之间比人类撰写的故事更加相似，前沿模型尤其倾向于收敛到一种“平均化”的通用叙事，缺乏人类作者群体的集体多样性。

    

    大型语言模型（LLM）的最新进展使其能够生成高质量的散文，然而这些模型是否能够生成多样化或具有创造性的作品仍然是一个有争议的问题。在这项工作中，我们通过叙事相似性的框架研究了LLM生成故事的多样性。我们使用对比框架以及来自r/WritingPrompts的人工撰写故事和提示词数据集，通过人工评估和三种不同的自动标注方法，收集了针对10个代表性LLM的叙事相似性判断。我们的发现揭示了一个明显的趋势：LLM生成的叙事彼此之间的相似程度始终高于人类撰写的故事。我们证明，尤其是前沿模型会收敛于一种“平均”的通用叙事，这种叙事接近于单个人类故事，但缺乏人类作者群体的集体多样性。最后，我们展示了常见的缓解（方法）……

    arXiv:2606.17350v2 Announce Type: replace-cross  Abstract: Recent advances in large language models (LLMs) have enabled the generation of high-quality prose, yet whether these models are capable of generating diverse or creative artifacts remains a contested question. In this work, we investigate the diversity of LLM-generated stories through the framework of narrative similarity. Using a contrastive framework and a dataset of human-written stories and prompts from r/WritingPrompts, we collect narrative similarity judgments across 10 representative LLMs, utilizing both human evaluations and three different automatic annotation methods. Our findings reveal a clear trend: LLM-generated narratives are consistently more similar to each other than human-written stories are. We demonstrate that frontier models in particular converge on a "mean" generic narrative that approximates individual human stories but lacks the collective diversity of human authors. Finally, we show that common mitiga
    
[^138]: 跟随潜空间路线图：利用锚定令牌为扩散大语言模型导航可撤销解码

    Follow the Latent Roadmap: Navigating Revocable Decoding for Diffusion LLMs with Anchor Tokens

    [https://arxiv.org/abs/2606.16847](https://arxiv.org/abs/2606.16847)

    提出了一种免训练框架ASRD，通过在嵌入空间中将解码上下文解耦为基于时间一致性识别的受信任锚定令牌和不确定候选令牌，解决了扩散大语言模型可撤销解码中的错误传播与局部错误强化问题。

    

    扩散大语言模型为并行生成提供了一条有前景的途径，但面临解码速度与质量之间的权衡。虽然可撤销解码策略试图通过验证和重新掩码令牌来缓解错误，但它们通常在混合质量的上下文中运行，这导致两个关键的失败：错误传播，即新令牌从错误上下文中吸收有毒信息；以及局部错误强化，即错误之间相互强化以逃避检测。为缓解这些挑战，我们提出了ASRD（锚定监督可撤销解码），这是一个在嵌入空间中运行的免训练框架。ASRD显式地将解码上下文解耦为受信任的锚定令牌（通过时间一致性识别）和不确定的候选令牌。利用动态锚定令牌缓存，我们引入了两个互补机制：(1) 锚定监督……（摘要在此处截断）

    arXiv:2606.16847v4 Announce Type: replace-cross  Abstract: Diffusion Large Language Models (dLLMs) offer a promising avenue for parallel generation but face a trade-off between decoding speed and quality. While revocable decoding strategies attempt to mitigate errors by verifying and remasking tokens, they typically operate within a mixed-quality context. This leads to two critical failures: \textit{Error Propagation}, where new tokens absorb toxic information from erroneous context, and \textit{Local Error Reinforcement}, where errors mutually reinforce each other to evade detection. To alleviate these challenges, we propose ASRD (Anchor Supervised Revocable Decoding), a training-free framework that operates within the embedding space. ASRD explicitly decouples the decoding context into trusted \textit{Anchor Tokens}, which are identified via temporal consistency, and uncertain candidates. Leveraging a dynamic Anchor Tokens Cache, we introduce two complementary mechanisms: (1) Anchor-
    
[^139]: SHARD：通过自我重构蒸馏实现安全且有益的对齐

    SHARD: Safe and Helpful Alignment via Self-Reframing Distillation

    [https://arxiv.org/abs/2606.15517](https://arxiv.org/abs/2606.15517)

    SHARD提出一种自我重构蒸馏方法，通过重写敏感提示凸显良性意图、将模型自身回答重构为更安全更有益的版本并据此微调，从而在保持安全性的同时提升有益性，效果可与更大教师模型的蒸馏相媲美。

    

    大型语言模型在处理敏感提示时常常表现不佳：它们可能直接拒绝回答、提供泛泛的安全套话，或者无法满足用户那些本可以在安全前提下得到解答的正当信息需求。我们提出了SHARD，一种用于提升安全-有益性的自我重构蒸馏方法。该方法首先依据哲学指导原则重写敏感提示，以凸显其中的良性意图；然后将模型原本的回答重构为更安全、更有帮助的回答；最后在模型自我重构的回答上对模型进行微调。在DNA数据集和LINGUASAFE的英语子集上的实验表明，SHARD在保持安全性的同时，提升了大多数模型家族的有益性。它还与来自更大教师模型的蒸馏方法相比具有竞争力，这表明模型能够将自身引发的安全且有益的行为加以内化。警告：本文包含可能具有冒犯性或有害的内容。

    arXiv:2606.15517v2 Announce Type: replace  Abstract: Large language models often struggle with sensitive prompts. They may refuse outright, provide generic safety boilerplate, or fail to address the user's legitimate informational needs that can be answered safely. We introduce SHARD, a self-reframing distillation method to improve safe-helpfulness. It first rewrites sensitive prompts to surface benign intent using philosophical guidelines, then reframes its original responses into safe, more helpful ones, and finally fine-tunes the model on its self-reframed responses. Across DNA and the English subset of LINGUASAFE, SHARD improves helpfulness for most model families while preserving safety. It also remains competitive with distillation from a larger teacher model, suggesting that models can internalize safe and helpful behavior elicited from their own. Warning: This paper contains content that may be offensive or harmful.
    
[^140]: MUDIDI：基于语言模型的多语言词典数字化两阶段框架

    MUDIDI: A Two-Stage Framework for Multilingual Dictionary Digitization with Language Models

    [https://arxiv.org/abs/2606.09435](https://arxiv.org/abs/2606.09435)

    该论文提出MUDIDI两阶段框架，利用视觉-语言模型将多语言词典扫描件数字化并转换为机器可读的词典学格式，同时发布了人工标注的词典条目数据集。

    

    多语言词典是低资源语言和濒危语言最有价值的文献资源之一，然而许多词典目前仅有扫描件可用。几十年来，由于语言特有的文字系统、充满缩写和交叉引用词条的复杂多栏版式，这些词典的数字化及其向机器可读格式的转换几乎无法实现。近期的视觉-语言模型为解决这一问题提供了有希望的方案，但它们在字符保留、标记保留以及词典学结构处理方面的表现尚不明确。我们提出了MUDIDI，一个用于多语言词典数字化的两阶段框架。第一阶段评估字符识别和标记保留的质量；第二阶段专注于词典条目分割，并将其映射到机器可读的词典学模式——SIL的多词典格式化器（Multi-Dictionary Formatter）中。我们还发布了一个由人工标注的词典条目组成的数据集……

    arXiv:2606.09435v2 Announce Type: replace  Abstract: Multilingual dictionaries are among the most valuable documentary resources for low-resource and endangered languages, yet many remain available only as scans. For many decades, their digitization and conversion into a machine-readable format was nearly impossible due to language-specific scripts, complex multi-column layouts full of entries with abbreviations and cross-references. Recent vision-language models offer a promising solution, but it is unclear how well they preserve characters, markup, and process lexicographic structure. We introduce MUDIDI, a two-stage framework for multi-lingual dictionary digitization. Stage One evaluates the quality of character recognition and markup preservation; Stage Two focuses on dictionary entry segmentation with subsequent mapping into a machine-readable lexicographic schema, SIL's Multi-Dictionary Formatter. We also release a dataset that consists of human-annotated lexicographic entries co
    
[^141]: 名字里有什么？大语言模型在药理学中的词形捷径

    What's in a Name? Morphological Shortcuts by LLMs in Pharmacology

    [https://arxiv.org/abs/2606.05616](https://arxiv.org/abs/2606.05616)

    大语言模型在药理学中过度依赖药物名称的词缀线索来推断药物含义，即使面对虚构药物也会产生类别级别的药理学响应，很少明确承认这种依赖，且有时会错误混淆共享相同词缀的药物属性。

    

    词语的形态特征常常能为其含义提供线索，但纯粹依赖这些映射在高风险领域可能导致过度泛化。例如在医学领域，大语言模型（LLM）仅凭词缀就能对虚构药物（如wugcillin）进行自信的推理，并生成看似合理的临床内容。我们对药理学中LLM的“词缀启发式”进行了行为学和机制层面的研究。通过使用由真实词缀构建的虚构药物名称，我们证明仅凭词缀信号就能引发类别级别的药理学响应。我们提出了一个框架，用于识别模型的药物语义主要由词缀、词干还是整个药物名称驱动。将该框架应用于653种药物后，我们发现模型通常主要通过词缀线索来推断药物含义，却很少明确表明这种依赖，有时还会错误地混淆共享相同词缀的药物之间的属性。

    arXiv:2606.05616v2 Announce Type: replace  Abstract: The morphological form of a word can often give cues to its meaning, but purely relying on these mappings can lead to overgeneralization in high-stakes domains. In the medical domain, for instance, LLMs can confidently reason about fictitious drugs from their affixes alone (e.g., wugcillin) and generate plausible-looking clinical content. We present a behavioral and mechanistic study of LLM "affix heuristics" in pharmacology. Using fictitious drug names built from real affixes, we show that affix signals alone elicit class-level pharmacological responses. We introduce a framework for identifying whether a model's drug semantics are driven mainly by the affix, the stem, or the drug name as a whole. Applied across 653 drugs, our framework reveals that models often induce drug meaning primarily through affix cues, yet rarely explicitly indicate this reliance, and sometimes incorrectly conflate properties among affix-sharing drugs. Activ
    
[^142]: 超越检索：学习紧凑的用户表示以实现可扩展的大语言模型个性化

    Beyond Retrieval: Learning Compact User Representations for Scalable LLM Personalization

    [https://arxiv.org/abs/2606.04547](https://arxiv.org/abs/2606.04547)

    提出TAP-PER框架，通过时序注意力前缀嵌入将用户偏好编码为紧凑的可学习表示，摆脱了检索式个性化对检索质量的依赖以及参数式个性化随用户规模增长的高昂存储成本，实现可扩展的大语言模型个性化。

    

    个性化大语言模型需要在使模型行为适应个体用户的同时，保持鲁棒性和部署规模上的效率。现有方法通常在输入层面对大语言模型进行个性化，即通过检索用户历史或构建用户画像提示，或者在参数层面对其进行个性化，即为每个用户维护特定的参数高效模块。前者使个性化效果依赖于检索质量和提示设计，而后者会产生随用户规模增长而增加的存储和维护成本。为了解决这些局限，我们提出了TAP-PER（Temporal Attentive Prefix for PERsonalization，时序注意力前缀个性化），这是一个基于前缀的框架，它将用户偏好编码为可学习的表示，避免了将用户历史序列化到提示中，并用轻量级的用户状态前缀嵌入取代了繁重的每用户适配器模块。受个性化推荐系统的启发，TAP-PER将用户……（原文摘要在此处截断）

    arXiv:2606.04547v3 Announce Type: replace-cross  Abstract: Personalizing large language models requires adapting model behavior to individual users while preserving robustness and deployment-scale efficiency. Existing approaches typically personalize LLMs either at the input level, by retrieving user histories or constructing profile prompts, or at the parameter level, by maintaining user-specific parameter-efficient modules. The former makes personalization sensitive to retrieval quality and prompt design, whereas the latter incurs storage and maintenance costs that grow with the user population. To address these limitations, we propose TAP-PER (Temporal Attentive Prefix for PERsonalization), a prefix-based framework that encodes user preferences as learnable representations, avoiding the serialization of user histories into prompts and replacing heavy per-user adapters with lightweight user-state prefix embeddings. Inspired by personalized recommendation systems, TAP-PER decomposes u
    
[^143]: 面向掩码扩散语言模型的知识编辑

    Knowledge Editing for Masked Diffusion Language Models

    [https://arxiv.org/abs/2606.03924](https://arxiv.org/abs/2606.03924)

    首次将“定位后编辑”知识编辑方法迁移至掩码扩散语言模型，发现最优编辑位置（最后一个主体词元处的早中期层MLP）在自回归模型与掩码扩散模型间可迁移，但多词元编辑在掩码扩散模型中退化显著更严重。

    

    知识编辑旨在更新或纠正语言模型中的事实知识。一种广泛使用的方法是“定位后编辑”，即先在模型中定位某个事实所在位置，然后编辑该处的权重。迄今为止，此类方法仅针对自回归模型开发。掩码扩散模型以双向方式建模文本，并通过迭代去噪而非下一个词预测来生成文本，此类方法是否适用于掩码扩散模型仍是一个开放问题。我们通过将定位后编辑方法迁移到掩码扩散模型上，并将多个掩码扩散模型与其匹配的自回归模型进行比较来解答这一问题。我们的核心发现有两点。第一，编辑应施加的位置在两类模型之间可以迁移：在最后一个主体词元处的早中期层MLP进行编辑，对两类模型而言都是最有效的。第二，这种共享的位置并不保证产生相同的结果。单词元编辑在两类模型中都能成功，但随着编辑目标变长，掩码扩散模型中的编辑效果退化程度远比自回归模型剧烈。

    arXiv:2606.03924v2 Announce Type: replace  Abstract: Knowledge editing aims to update or correct factual knowledge in a language model. A widely used approach, locate-then-edit, first localizes a fact within the model and then edits the weights there. To date, such methods have been developed exclusively for autoregressive models (ARMs). Whether they work for masked diffusion models (MDMs), which model text bidirectionally and generate by iterative denoising rather than next-token prediction, remains an open question. We address it by transferring locate-then-edit to MDMs and comparing multiple MDMs with their matched ARMs. Our central finding has two parts. First, where an edit should be applied transfers between them: the same early-to-mid-layer MLP at the last subject token is most effective for both. Second, this shared location does not guarantee a shared outcome. Single-token edits succeed in both, but as targets grow longer, editing degrades far more sharply in the MDMs than in 
    
[^144]: LLM作为评判者的几何学：为什么LLM之间的共识并不等于人类对齐

    The Geometry of LLM-as-Judge: Why Inter-LLM Consensus Is Not Human Alignment

    [https://arxiv.org/abs/2606.03043](https://arxiv.org/abs/2606.03043)

    本文提出一种将LLM评判者分数视为向量并通过测量离散度、有效秩、与人类分数夹角及一致性三元组的几何检验方法，揭示了LLM之间的相互共识不能等同于与人类判断的对齐——评判者在主观标准上彼此一致程度接近人类，但与人类评分的一致性仅达58-66%，原因在于它们可能共享同样的盲点。

    

    如今，大语言模型（LLM）评判者对大多数开放式NLP输出进行打分，它们之间的一致性通常被视为分数可信的证据。但这种解读并不安全：评判者之间达成一致，可能是因为它们真正捕捉到了质量，也可能是因为它们共享同样的盲点，而仅凭一致性统计无法区分这两种情况。我们开发了一种能够进行这种区分的几何测试。我们将每个评判者的分数视为一个向量，在两个由社区构建、覆盖四个领域和八种语言的印度语系基准上，对42个评判者测量其分数的离散度、有效秩、与人类分数的夹角，以及评判者-评判者、评判者-人类、人类-人类一致性三元组。每一项比较都是参考匹配的：评判者与一个留出评分者都针对相同的双人评分均值进行评分，因为若非如此，平均参考值会以数度之差美化评判者的表现。在主观评分标准上，评判者之间的一致程度与人类相当，但仅达到人类一致性的58-66%，并且常常集中于某个轴上……

    arXiv:2606.03043v2 Announce Type: replace  Abstract: LLM judges now score most open-ended NLP output, and their mutual agreement is routinely read as evidence that the scores can be trusted. That reading is unsafe: judges may agree because they capture quality, or because they share the same blind spots, and agreement statistics alone cannot tell these apart. We develop a geometric test that can. Treating each judge's scores as a vector, we measure spread, effective rank, the angle to human scores, and the judge-judge, judge-human, and human-human agreement triple for 42 judges on two community-built Indic benchmarks covering four domains and eight languages. Every comparison is reference-matched: a judge and a held-out rater are scored against the same two-rater mean, since an averaged reference otherwise flatters judges by several degrees. On subjective rubrics, judges agree with one another as much as humans do yet reach only 58-66% of human agreement and often concentrate on an axi
    
[^145]: 将古典诗歌翻译为现代散文

    Translating Classical Poetry into Modern Prose

    [https://arxiv.org/abs/2606.02806](https://arxiv.org/abs/2606.02806)

    该论文构建了Padyam2Gadyam数据集（包含600首13-17世纪泰卢固语古典诗歌及其人工校验的泰卢固语和英语散文翻译），并据此评估了机器翻译系统与大语言模型在零样本诗歌到散文翻译任务上的表现，发现尽管大语言模型优于机器翻译系统，但各系统在散文翻译的生成与评估上仍存在系统性问题。

    

    我们构建了一个用于诗歌到散文翻译任务的数据集，将13至17世纪的泰卢固语古典诗歌翻译为当代泰卢固语和英语散文，并将其命名为Padyam2Gadyam。该数据集包含600首诗歌及其经人工校验的泰卢固语和英语散文翻译。我们利用该数据集评估了2个机器翻译系统和5个当代大语言模型（LLM）在零样本条件下将诗歌翻译为泰卢固语和英语散文的能力。我们的结果表明，尽管通用大语言模型的表现优于机器翻译系统，但在所有系统中，两种语言的散文翻译在生成和评估方面均存在系统性问题。

    arXiv:2606.02806v3 Announce Type: replace  Abstract: We built a dataset for the task of poem-to-prose translation from 13th-17th Century Telugu classical poetry to contemporary Telugu and English prose, which we call Padyam2Gadyam. The dataset consists of 600 poems and their human-verified Telugu and English prose translations. We evaluated 2 machine translation systems and 5 contemporary Large Language Models (LLMs) on their ability to do zero-shot poem-to-prose translation into Telugu and English using this dataset. Our results indicate that while the general purpose LLMs are better than the machine translation systems, there are systematic issues with the generation and evaluation of prose translation in both languages across systems.
    
[^146]: 谁在NLP中进行标注？2018至2025年间人类标注报告的大规模评估

    Who Annotates in NLP? A Large-scale Assessment of Human Annotation Reporting between 2018 and 2025

    [https://arxiv.org/abs/2606.02255](https://arxiv.org/abs/2606.02255)

    首次对2018至2025年间主要NLP会议中的人类标注报告进行大规模任务级审计，提出统一的标注报告分类体系并借助经验证的LLM抽取流程构建了大规模标注报告数据集，揭示了标注者身份与过程控制等信息在论文中的普遍缺失。

    

    人类标注是许多NLP研究的实证基础，涵盖从数据集构建到模型评估的各个环节，但论文往往不清楚标注者是谁、标注过程如何被控制。我们首次对主要NLP会议中的人类标注报告进行了大规模、任务级别的审计，探究哪些标注细节被记录、哪些缺失，以及报告内容如何随时间、主题、会议和人类判断的预期用途而变化。我们提出了一个统一的标注报告实践分类体系，并在Annotated-gold（一个由人工裁定的金标准，包含41篇论文和72个标注任务）上验证了LLM辅助的抽取流程，其中表现最佳的模型与裁定标签达成与人类相当的一致性，Krippendorff's alpha系数为0.606，而人类之间的一致性为0.585。利用该流程，我们构建了Annotated-llm数据集，涵盖ACL会议论文（原文摘要在此处被截断）。

    arXiv:2606.02255v2 Announce Type: replace-cross  Abstract: Human annotation is the empirical foundation of much NLP research, from dataset construction to model evaluation, but papers often leave unclear who produced the annotations and how the annotation process was controlled. We provide the first large-scale, task-level audit of human annotation reporting across major NLP venues, asking which annotation details are documented, which are missing, and how reporting varies across time, topic, venue, and intended use of human judgment. We introduce a unified taxonomy of annotation-reporting practices and validate an LLM-assisted extraction pipeline against Annotated-gold, a human-adjudicated gold standard of 41 papers and 72 annotation tasks, where the best model reaches human-comparable agreement with adjudicated labels, with Krippendorff's alpha of 0.606 versus 0.585 for human-human agreement. Using this pipeline, we construct Annotated-llm, a dataset covering ACL-venue papers from 20
    
[^147]: TUX：测量人类与人工智能之间的默契理解

    TUX: Measuring Human--AI Tacit Understanding

    [https://arxiv.org/abs/2605.30930](https://arxiv.org/abs/2605.30930)

    该论文提出了一个受派对游戏 Wavelength 启发的谱系放置任务，并定义了默契理解指数（TUX），用于量化人类与 LLM 智能体在缺乏明确目标、沟通或反馈情况下达成默契对齐的程度，发现特质空间中更接近的人类—智能体配对具有更高的默契度。

    

    随着大语言模型（LLM）越来越多地充当协作伙伴，人机对齐通常通过明确的任务成功、准确率或奖励优化来评估。然而，许多协作场景依赖于默契理解：即智能体能否在没有明确目标、沟通或反馈的情况下，与人类的评价立场或表征先验保持一致。为了研究这种能力，我们开发了一个受社交派对游戏 Wavelength 启发的谱系放置任务，在该任务中，人类和智能体独立地将概念放置在主观谱系上。我们将默契理解指数（TUX）操作化为一种衡量人类与智能体判断之间相似性的成对行为度量，并通过 241 名人类参与者和来自四个模型的 200 个基于角色配置的 LLM 智能体对其进行评估。我们发现，在特质空间中距离最近的人类—智能体配对获得了显著更高的 TUX 分数，这表明默契对齐与……

    arXiv:2605.30930v2 Announce Type: replace-cross  Abstract: As large language models (LLMs) increasingly act as collaborative partners, human--AI alignment is often evaluated through explicit task success, accuracy, or reward optimization. Yet many collaborative settings depend on tacit understanding: whether an agent can align with a human's evaluative stance or representational priors without clear objectives, communication, or feedback. To study this capacity, we develop a spectrum-placement task inspired by the social party game Wavelength, in which humans and agents independently place concepts along subjective spectra. We operationalize the Tacit Understanding Index (TUX) as a pairwise behavioral measure of similarity between human and agent judgments, and evaluate it with 241 human participants and 200 profile-conditioned LLM agents across four models. We find that nearest human--agent pairs in trait space achieve significantly higher TUX, suggesting that tacit alignment is assoc
    
[^148]: 给它空间！编码器中位置表示与语义表示的显式解耦

    Give it Space! Explicit Disentangling of Positional and Semantic Representations in Encoders

    [https://arxiv.org/abs/2605.30022](https://arxiv.org/abs/2605.30022)

    该论文通过将编码器Transformer中的语义、绝对位置和相对位置表示显式解耦为三条独立的信息流，实现了对位置信息内部处理机制的清晰研究，发现隔离的绝对位置子空间会自发塌缩为低频二维流形，为设计更好的位置编码提供了启示。

    

    位置编码（PE）是置换不变Transformer表示序列顺序的基础，然而位置信息是如何被处理和存储的仍然知之甚少。诸如RoPE等现代位置编码方法在长上下文理解或检索等任务上仍然表现不佳。因此，更好地理解内部位置机制有助于设计更好的位置编码。基于位置信号和语义信号在训练好的Transformer中占据近乎正交子空间的证据，我们修改了一个编码器Transformer，使其处理三条显式解耦的流：语义流、绝对位置流（AP）和相对位置流（RP），并将掩码语言建模（MLM）目标限制在语义流上。这种解耦使得干净的机制性研究成为可能，并得出了三个要点。（1）被隔离的AP子空间自发地塌缩成一个低频二维流形，该流形可（原文在此处截断）

    arXiv:2605.30022v2 Announce Type: replace-cross  Abstract: Positional encoding (PE) underpins how permutation-invariant Transformers represent sequence order, yet how positional information is processed and stored remains poorly understood. Modern PE methods such as RoPE still struggle on tasks such as long-context understanding or retrieval \cite{chen-etal-2025-hope}. Hence, a better understanding of the internal positional mechanism could help design better PE. Building on evidence that positional and semantic signals occupy nearly orthogonal subspaces in trained Transformers, we modify an encoder Transformer to process three explicitly disentangled streams: semantic, absolute positional (AP) and relative positional (RP), and confine the masked-language-modeling (MLM) objective to the semantic stream. This decoupling enables a clean mechanistic study and yields three take-aways. (1) The isolated AP subspace spontaneously collapses into a low-frequency two-dimensional manifold that ca
    
[^149]: 视觉语言模型后训练中推理与感知的非对称优化研究

    On Asymmetric Optimization of Reasoning and Perception in Vision-Language Model Post-Training

    [https://arxiv.org/abs/2605.29496](https://arxiv.org/abs/2605.29496)

    该研究揭示视觉语言模型后训练中存在感知与推理的非对称提升现象——SFT中源于感知token占比失衡、RL中源于结果奖励与推理的耦合，并通过损失重加权等方法将端到端性能提升高达18.2分。

    

    后训练极大地提升了前沿视觉语言模型的推理能力，但其在感知方面的收益相对有限，这为端到端视觉推理造成了瓶颈。为研究这一差距，我们引入了一个受控诊断框架，包含两个将感知与推理解耦的合成任务。我们的分析揭示了一致的感知-推理非对称性：后训练对推理的提升明显大于对感知的提升，但其底层机制在不同训练范式下有所不同。对于监督微调（SFT），这种非对称性源于token不平衡，即感知在思维链监督中占据的token比例较小；对损失进行重新加权可使端到端性能提升高达18.2分。对于强化学习（RL），这种非对称性则源于奖励耦合，因为结果奖励与推理的相关性强于与感知的相关性。添加基于每……

    arXiv:2605.29496v2 Announce Type: replace  Abstract: Post-training has greatly improved reasoning in frontier vision-language models, yet its gains for perception remain comparatively limited, creating a bottleneck for end-to-end visual reasoning. To investigate this gap, we introduce a controlled diagnostic framework with two synthetic tasks that disentangle perception from reasoning. Our analysis reveals a consistent perception-reasoning asymmetry: post-training improves reasoning more substantially than perception, though the underlying mechanism differs across training paradigms. For supervised fine-tuning (SFT), this asymmetry stems from token imbalance, with perception occupying a smaller fraction of tokens in chain-of-thought supervision. Reweighting the loss boosts end-to-end performance by up to 18.2 points. For reinforcement learning (RL), the asymmetry instead arises from reward coupling, as outcome rewards correlate more strongly with reasoning than perception. Adding a per
    
[^150]: 当话语压力相互冲突时：视觉-语言模型输出中的信息结构

    When Discourse Pressures Conflict: Information Structure in Vision-Language Model Outputs

    [https://arxiv.org/abs/2605.28346](https://arxiv.org/abs/2605.28346)

    该研究借助匈牙利语中话题与焦点对应专属句法位置的特性，首次系统评估了视觉-语言模型在视觉问答中区分话语旧信息（话题）与新信息（焦点）的能力，发现模型虽能产出信息结构相关的句式，但与人类多变的语用策略不同，它们会坍缩为狭窄固定的响应模板，表现出模式坍缩式的过度规则化。

    

    视觉-语言模型（VLM）越来越多地被用于评估其能否识别正确的视觉内容，但人们对其是否能以符合话语要求的形式表达这些内容却知之甚少。我们利用信息结构（IS）来填补这一研究空白，测试VLM在基于视觉的问答中能否区分话语旧信息（话题，Topic）与话语新信息（焦点，Focus）。我们借助匈牙利语展开研究——在该语言中，话题和焦点分别映射到专门的句法位置，使得信息结构的选择可以在文本中被直接观察到。通过将六个VLM与人类参与者进行比较，我们发现模型能够产出与信息结构相关的句法结构，但过度规则化了这种敏感性。在话语地位、语法角色（偏好主语充当话题）与有定性（偏好无定形式充当焦点）等多重压力的相互作用下，人类会采用多样化的策略来实现信息结构。相比之下，VLM则坍缩为狭窄的固定响应模板，呈现出类似模式坍缩（mode collapse）的现象。

    arXiv:2605.28346v3 Announce Type: replace  Abstract: Vision-language models (VLMs) are increasingly evaluated for whether they identify the right visual content, but little is known about whether they express such content in a discourse-appropriate form. We address this research gap using information structure (IS), testing whether VLMs distinguish discourse-old Topics from discourse-new Foci in visually grounded question answering. We exploit Hungarian, a language in which Topic and Focus map onto dedicated syntactic positions, making IS choices observable in text. Comparing six VLMs with human participants, we find that models produce IS-relevant constructions, but over-regularise this sensitivity. Under the interacting pressures of discourse status, grammatical role (preference for subject Topics) and definiteness (preference for indefinite Foci), humans choose variable strategies for IS realisation. VLMs, by contrast, collapse onto narrow response templates, resembling mode collaps
    
[^151]: BioELX：无需任务特定监督的上下文感知跨语言生物医学实体链接

    BioELX: Context-Aware Cross-lingual Biomedical Entity Linking without Task-Specific Supervision

    [https://arxiv.org/abs/2605.27380](https://arxiv.org/abs/2605.27380)

    BioELX提出了一种检索-重排序框架，利用Wikidata衍生的跨语言别名监督进行检索，并通过提及锚定提示将LLM重排序器适配于实体链接，实现了无需任务特定监督的上下文感知跨语言生物医学实体链接。

    

    跨语言生物医学实体链接（BEL）将任何语言中的提及（mention）映射到生物医学知识库中的唯一标识符，为临床和生物医学NLP应用提供支持。我们发现了影响当前系统的两个问题。首先，用于训练跨语言BEL检索器的UMLS别名严重偏向英语，导致检索器对非英语提及的泛化能力较差。其次，尽管上下文通常对消歧是必要的，但将上下文简单地注入仅以别名对齐为目标训练的检索器中，会严重降低检索性能。我们提出了BioELX，一个解决这两个问题的检索-重排序框架。在检索阶段，我们使用源自Wikidata的跨语言别名监督对SapBERT_multi进行继续训练，从而在各语言之间形成共享的概念邻域。在重排序阶段，我们通过提及锚定提示（mention-anchored prompting）将预训练的LLM重排序器适配到实体链接任务。

    arXiv:2605.27380v2 Announce Type: replace-cross  Abstract: Cross-lingual biomedical entity linking (BEL) maps mentions in any language to unique identifiers in a biomedical knowledge base, supporting clinical and biomedical NLP applications. We identify two issues affecting current systems. First, the UMLS (Bodenreider,2004) aliases used to train cross-lingual BEL retrievers are heavily skewed toward English, so retrievers generalize poorly to non-English mentions. Second, although context is often necessary for disambiguation, naively injecting context into retrievers trained only to align aliases severely degrades retrieval. We propose BioELX, a retrieve-rerank framework that addresses both issues. For retrieval, we continue training SapBERT_multi (Liu et al., 2021b) using Wikidata-derived cross-lingual alias supervision, forming shared concept neighborhoods across languages. For reranking, we adapt pretrained LLM rerankers to entity linking through mention-anchored prompting, which 
    
[^152]: CroCo：基于自生成响应的跨语言对比偏好调优

    CroCo: Cross-Lingual Contrastive Preference Tuning on Self-Generations

    [https://arxiv.org/abs/2605.26293](https://arxiv.org/abs/2605.26293)

    基于自生成响应的跨语言对比偏好调优无需语言特定的偏好标注，仅凭英语偏好训练的奖励模型即可在14种高低资源语言上实现有效迁移，并避免监督微调的灾难性遗忘。

    

    先前的研究表明，通过奖励分数设定的大语言模型自生成响应之间的受控对比性，能够改善英语环境下的下游偏好调优效果。我们将该方法扩展到多种语言，并在总计14种高资源和低资源语言上，对两个模型在多样化任务中进行了评估。我们的核心发现是，基于自生成响应的跨语言对比偏好调优无需语言特定的偏好标注即可实现迁移。在多语言基础模型之上，仅基于英语偏好训练的奖励模型就能在大多数语言中产生有用的语言内排序；无论是在单语还是多语设置下进行配对，都能在大多数配置中超越各个基线模型，同时避免了监督微调中的灾难性遗忘问题。我们观察到，这些收益依赖于在线策略数据。离线策略响应会降低收益，而在线偏好优化……

    arXiv:2605.26293v2 Announce Type: replace-cross  Abstract: Prior work establishes that controlled contrastiveness between self-generated responses from large language models, set via reward scores, improves downstream preference tuning in English. We extend this method to multiple languages and evaluate two models across a total of 14 high and low-resource languages on a diverse set of tasks. Our central finding is that cross-lingual contrastive preference tuning on self-generations (CroCo) transfers without language-specific preference annotation. A reward model trained on English preferences (atop a multilingual base) produces useful within-language rankings across most languages, and pairing in either a monolingual or multilingual setting improves over each model on the majority of setups while preventing the catastrophic forgetting of supervised fine-tuning. We observe that the gains require on-policy data. Off-policy responses reduce the benefit and online preference optimization 
    
[^153]: 衡量大语言模型推理质量：一个多维度行为评估框架

    Measuring Reasoning Quality in LLMs: A Multi-Dimensional Behavioral Framework

    [https://arxiv.org/abs/2605.24661](https://arxiv.org/abs/2605.24661)

    本文提出了一个植根于认知科学的多维度行为评估框架，从正确性、一致性、鲁棒性、局部逻辑连贯性、效率和稳定性六个维度衡量大语言模型的推理质量，突破了仅依赖最终答案正确性的传统评估局限，并支持面向具体部署场景的模型选择。

    

    尽管大语言模型在推理基准测试上取得了显著进展，当前的评估实践仍然以最终答案的正确性为核心，对于模型如何进行推理、在上下文变化下其行为有多可靠、以及以多高的效率得出结论等方面所能提供的洞见十分有限。本文提出了一个从行为视角衡量大语言模型推理质量的统一多维度框架，将植根于认知科学的六个具有理论依据的维度进行了可操作化定义：正确性（CQ）、一致性（CS）、鲁棒性（RS）、局部逻辑连贯性（LS）、效率（ES）和稳定性（SS）。该框架引入了面向部署场景的聚合方法，使得模型选择能够超越基于准确率的排行榜，实现针对特定应用场景的评估。在多个大语言模型和基准上的实验揭示了被单一指标评估所系统性掩盖的行为，包括局部逻辑连贯性与正确性之间的正交性，以及部署场景下的……（原文摘要至此截断）

    arXiv:2605.24661v4 Announce Type: replace  Abstract: Despite remarkable progress on reasoning benchmarks, current LLM evaluation practice remains anchored to final-answer correctness, providing limited insight into how models reason, how reliably they behave under contextual variation, or how efficiently they reach conclusions. This paper proposes a unified multi-dimensional framework for measuring LLM reasoning quality from a behavioral perspective, operationalizing six theoretically grounded dimensions rooted in cognitive science: Correctness (CQ), Consistency (CS), Robustness (RS), Local Logical Coherence (LS), Efficiency (ES), and Stability (SS). The framework introduces deployment-aware aggregation, enabling context-specific model selection beyond accuracy-based leaderboards. Experiments across multiple LLMs and benchmarks reveal behaviors systematically concealed by single-metric evaluation, including the orthogonality of local logical coherence and correctness, deployment-contex
    
[^154]: 基于微调Transformer的无作答多选题难度建模：组件化表示与多任务学习

    Response-free item difficulty modelling for multiple-choice items with fine-tuned transformers: Component-wise representation and multi-task learning

    [https://arxiv.org/abs/2605.16991](https://arxiv.org/abs/2605.16991)

    该论文提出对Transformer进行端到端微调以在无作答数据的情况下预测多选题难度，并通过组件化表示和多任务问答学习两种扩展提升了难度估计效果。

    

    在测验施测之前，由于尚无作答数据可用于校准，题目难度往往需要预先估计。大多数无作答难度建模方法是人工提取题目文本特征，再输入独立的统计模型；而我们对Transformer进行端到端微调，直接基于题目措辞进行学习，从而避免了基于理论的特征设计以及会丢失信息的预处理。我们针对阅读理解类多选题开展研究，这类题目的难度取决于横跨文章、问题和选项的推理要求，然而最简单的模型只看到一个不加区分的序列，且仅以难度作为训练目标。我们引入并研究了针对联合编码基线的两种扩展：一种是组件化变体，对措辞的各个组成部分分别进行编码；另一种是多任务变体，增加了问答作为辅助任务。我们在从近3……的语料库中抽取的三种训练集规模下对这些方法进行了比较（原文摘要在此处不完整）。

    arXiv:2605.16991v2 Announce Type: replace-cross  Abstract: Item difficulty must often be estimated before test administration, when no responses are yet available for calibration. While most response-free difficulty modelling approaches derive item-text features by hand for a separate statistical model, we fine-tune a transformer end-to-end on the wording, avoiding the theory-based feature design and the preprocessing that discards information. We address reading-comprehension multiple-choice items, whose difficulty depends on inferential demands spanning passage, question, and options, yet the simplest model sees one undifferentiated sequence and is trained on difficulty alone. We introduce and investigate two extensions to the joint-encoding baseline: a component-wise variant, which encodes the wording parts separately, and a multi-task variant, which adds an auxiliary task of question answering. We compare the methods across three training-set sizes sampled from a corpus of nearly 3
    
[^155]: 当提示词相互作用：评估提示词算术在分布偏移下的去混杂能力

    When Prompts Interact: Assessing Prompt Arithmetic for Deconfounding under Distribution Shift

    [https://arxiv.org/abs/2605.03096](https://arxiv.org/abs/2605.03096)

    本文研究了通过任务算术组合软提示能否提升模型对混杂变量引起分布偏移的鲁棒性，并提出了一种混合提示算术方法来去除模型对虚假特征的依赖，相比完全微调更具计算效率。

    

    在分类任务中，模型可能依赖混杂变量来获得强大的分布内性能，捕获在分布偏移下失效的虚假特征。这种捷径行为会导致在分布外场景中出现显著的性能下降。任务算术提供了一种潜在的解决方案，通过减去次要模型更新来移除不需要的信号，但它通常需要完全微调，计算成本高昂。提示调优提供了一种参数高效的替代方案，通过一小组可训练的虚拟令牌来适配模型。对由此产生的提示词进行任务算术运算，为对整个模型进行操作提供了一种有吸引力的替代方法，但这种方法能在多大程度上限制对虚假特征的依赖仍有待验证。在这项工作中，我们研究了通过任务算术组合软提示是否能提高模型对混杂偏移的鲁棒性。我们提出了混合提示算术方法……

    arXiv:2605.03096v2 Announce Type: replace-cross  Abstract: In classification tasks, models may rely on confounding variables to achieve strong in-distribution performance, capturing spurious features that fail under distribution shift. This shortcut behavior leads to substantial degradation in out-of-distribution settings. Task arithmetic offers a potential solution by removing unwanted signals via subtraction of secondary model updates, but it typically requires full fine-tuning, which is computationally expensive. Prompt tuning provides a parameter-efficient alternative by adapting models through a small set of trainable virtual tokens. Task arithmetic on the resulting prompts presents an appealing alternative to operations on entire models, but the extent to which this approach can limit reliance on spurious features remains to be established. In this work, we study whether composing soft prompts through task arithmetic improves robustness to confounding shifts. We propose Hybrid Pr
    
[^156]: 编码智能体能否复现计算材料科学中的研究发现？

    Can Coding Agents Reproduce Findings in Computational Materials Science?

    [https://arxiv.org/abs/2605.00803](https://arxiv.org/abs/2605.00803)

    本文提出 AutoMat 基准，用于评估大语言模型编码智能体复现计算材料科学论文中科学论断的能力，涵盖恢复欠规范计算流程、驾驭专用工具链和验证证据是否支持论断三大挑战。

    

    大语言模型正越来越多地被部署为自主编码智能体，并在软件工程基准测试中取得了极为出色的性能。然而，这种成功能否迁移到计算科学工作流程中尚不明确，因为这类任务不仅需要强大的编码能力，还需要能够驾驭复杂的、特定领域的操作流程，并在科学论断的语境下解释结果。为了解答这一问题，我们提出了 AutoMat，一个用于评估基于大语言模型的智能体复现计算材料科学论断能力的基准。AutoMat 包含三个相互关联的挑战：恢复欠规范的计算流程、驾驭专用工具链，以及判断所得到的结果能否支持某一论断。通过与领域专家紧密合作，我们从真实的材料科学论文中精选出一组论断，用以测试编码智能体能否恢复（此处摘要内容被截断）

    arXiv:2605.00803v2 Announce Type: replace-cross  Abstract: Large language models are increasingly deployed as autonomous coding agents and have achieved remarkably strong performance on software engineering benchmarks. However, it is unclear whether such success transfers to computational scientific workflows, where tasks require not only strong coding ability, but also the ability to navigate complex, domain-specific procedures and to interpret results in the context of scientific claims. To address this question, we present AutoMat, a benchmark for evaluating LLM-based agents' ability to reproduce claims from computational materials science. AutoMat poses three interrelated challenges: recovering underspecified computational procedures, navigating specialized toolchains, and determining whether the resulting evidence supports a claim. By working closely with subject matter experts, we curate a set of claims from real materials science papers to test whether coding agents can recover 
    
[^157]: 语言扩散模型是能够检索未见数据的联想记忆

    Language Diffusion Models are Associative Memories Capable of Retrieving Unseen Data

    [https://arxiv.org/abs/2604.26841](https://arxiv.org/abs/2604.26841)

    该论文证明均匀离散扩散语言模型本质上是联想记忆，其吸引盆可通过条件似然最大化而非显式能量函数形成，并揭示了由数据规模支配的从记忆到泛化的急剧转变，使其能够检索未见过的数据。

    

    语言扩散模型何时会记忆其训练数据，以及如何定量评估其真正的生成机制？我们通过证明基于均匀分布的离散扩散模型（UDDMs）在根本上表现为具有涌现创造能力的联想记忆（AMs）来回答这些问题。联想记忆的核心思想是通过在存储的数据点周围建立独特的吸引盆，从而可靠地将这些数据点作为“记忆”恢复出来。历史上，像Hopfield网络这样的模型使用显式的能量函数来保证这些稳定吸引子的存在。我们拓展了这一视角，利用了一个关键观察：能量并非严格必需，因为吸引盆也可以通过条件似然最大化来形成。通过评估模型对训练样本和测试样本的词元恢复能力，我们在UDDMs中识别出一个由训练规模大小所支配的、从记忆到泛化的急剧转变（摘要在此处截断）。

    arXiv:2604.26841v2 Announce Type: replace-cross  Abstract: When do language diffusion models memorize their training data, and how to quantitatively assess their true generative regime? We address these questions by showing that Uniform-based Discrete Diffusion Models (UDDMs) fundamentally behave as Associative Memories (AMs) $\textit{with emergent creative capabilities}$. The core idea of an AM is to reliably recover stored data points as $\textit{memories}$ by establishing distinct basins of attraction around them. Historically, models like Hopfield networks use an explicit energy function to guarantee these stable attractors. We broaden this perspective by leveraging the observation that energy is not strictly necessary, as basins of attraction can also be formed via conditional likelihood maximization. By evaluating token recovery of $\textit{training}$ and $\textit{test}$ examples, we identify in UDDMs a sharp memorization-to-generalization transition governed by the size of the t
    
[^158]: GroupDPO：内存高效的分组式直接偏好优化

    GroupDPO: Memory-Efficient Group-Wise Direct Preference Optimization

    [https://arxiv.org/abs/2604.15602](https://arxiv.org/abs/2604.15602)

    GroupDPO 提出一种内存高效的分组式直接偏好优化方法，通过基于目标特定逐响应系数的一阶线性化，在保持一阶梯度不变的同时于反向传播中解耦样本，从而充分利用偏好数据中多候选响应的监督信息。

    

    偏好优化被广泛用于将大型语言模型（LLM）与偏好反馈对齐。然而，大多数现有方法在每个提示上仅训练单个正负样本对，丢弃了偏好数据集中通常包含多个候选响应所带来的额外监督信息。受此局限性的启发，最近的工作探索了分组式偏好优化，即对同一提示的多个响应进行联合对比，但由于组耦合目标的内存开销，其经验表现和可扩展性仍未得到充分研究。在这项工作中，我们对分组式偏好优化进行了统一的实证与系统研究，并为组耦合目标开发了一种内存高效的实现。通过以目标特定的逐响应系数实例化一阶线性化，我们的实现既保持了一阶梯度不变，又在反向传播过程中实现了样本间的解耦。

    arXiv:2604.15602v2 Announce Type: replace  Abstract: Preference optimization is widely used to align Large Language Models (LLMs) with preference feedback. However, most existing methods train on a single positive-negative pair per prompt, discarding additional supervision available in preference datasets that typically contain multiple candidate responses. Motivated by this limitation, recent work explores group-wise preference optimization, which jointly contrasts multiple responses for the same prompt, but its empirical behavior and scalability remain underexplored due to the memory overhead of group-coupled objectives. In this work, we present a unified empirical and systems study of group-wise preference optimization and develop a memory-efficient implementation for group-coupled objectives. By instantiating first-order linearization with objective-specific per-response coefficients, our implementation preserves first-order gradients while decoupling samples during backpropagation
    
[^159]: 仇恨言论内容审核的执行与可行性

    The Enforcement and Feasibility of Hate Speech Moderation

    [https://arxiv.org/abs/2604.12289](https://arxiv.org/abs/2604.12289)

    该研究通过54万条推文的大规模审计发现Twitter/X上80%的仇恨言论五个月后仍在线，但模拟“自动排序+人工分流”工作流证明大幅清除仇恨言论在财务上完全可行且成本远低于监管罚款，表明仇恨言论泛滥源于平台资源配置不足而非技术限制。

    

    网络仇恨言论的危害从心理健康恶化到暴力行为不一而足，然而平台对仇恨言论的审核是否一致、执法在大规模上是否可行，仍鲜为人知。我们使用由受过训练的母语使用者标注的54万条推文（代表平台上一整天的内容）对Twitter（现X）的仇恨言论审核进行了审计。发布五个月后，80%的仇恨推文（包括暴力言论）仍然在线。仇恨推文被删除的概率仅比非仇恨推文略高，远低于诈骗或成人内容的处理力度，且对严重程度和传播范围均不敏感。自动检测无法可靠地对仇恨言论进行分类，但能对其高精度排序，从而支持人工分流。在模拟该工作流程时，当前的人员配置几乎无法减少曝光量，但大幅减少曝光在财务上是可行的，成本远低于适用的监管罚款。仇恨言论的持续存在反映的是资源配置问题，而非技术限制。

    arXiv:2604.12289v2 Announce Type: replace-cross  Abstract: Online hate speech is associated with harms ranging from deteriorating mental health to violence, yet how consistently platforms moderate hate, and whether enforcement is feasible at scale, remain poorly understood. We audit hate speech moderation on Twitter (now X) using 540,000 tweets annotated by trained native speakers, representative of a full day on the platform. Five months after posting, 80% of hateful tweets, including violent ones, remained online. Removal was only marginally more likely than for non-hateful tweets, far below scams or adult content, and insensitive to severity and reach. Automated detection could not reliably classify hate but ranked it highly, enabling human triage. Simulating this workflow, current staffing curbed little exposure, yet substantial reductions proved financially feasible, far below applicable regulatory fines. Persistent hate reflects resource allocation, not technical limits.
    
[^160]: 非英语论文会被公平评审吗？NLP同行评审中的研究语言偏见

    Are Non-English Papers Reviewed Fairly? Language-of-Study Bias in NLP Peer Reviews

    [https://arxiv.org/abs/2604.07119](https://arxiv.org/abs/2604.07119)

    该研究首次系统刻画了NLP同行评审中的“研究语言偏见”，区分其负面与正面形式，构建了人工标注数据集LOBSTER及基于大语言模型的检测方法（宏F1达87.37），并通过分析15,645条评审发现非英语论文遭受的偏见显著更高。

    

    同行评审在NLP的发表过程中扮演着核心角色，但容易受到各种偏见的影响。本文研究“研究语言”（Language-of-Study, LoS）偏见：即审稿人根据论文所研究的语言而非其科学价值来差异化评价论文的倾向。尽管审稿指南中已明确指出了这类偏见，但人们对它的了解仍然甚少。先前的工作往往将此类评论归入更宽泛的低质量或缺乏建设性评审的类别中，而未将其定义为一种独特的偏见形式。我们首次对LoS偏见进行了系统性刻画，区分了其负面与正面两种形式，并引入了人工标注数据集LOBSTER（Language-Of-study Bias in ScienTific pEer Review）以及一个基于大语言模型的检测流水线，其宏观F1分数达到87.37。我们分析了15,645条评审意见，以估计负面偏见和正面偏见相对于研究语言的差异，并发现非英语论文面临显著更高的（偏见）。

    arXiv:2604.07119v2 Announce Type: replace  Abstract: Peer review plays a central role in the NLP publication process, but is susceptible to various biases. Here, we study language-of-study (LoS) bias: the tendency for reviewers to evaluate a paper differently based on the language(s) it studies, rather than its scientific merit. Despite being explicitly flagged in reviewing guidelines, such biases are poorly understood. Prior work treats such comments as part of broader categories of weak or unconstructive reviews without defining them as a distinct form of bias. We present the first systematic characterization of LoS bias, distinguishing negative and positive forms, and introduce the human-annotated dataset LOBSTER (Language-Of-study Bias in ScienTific pEer Review) and an LLM-based detection pipeline achieving 87.37 macro F1. We analyze 15,645 reviews to estimate how negative and positive biases differ with respect to the LoS, and find that non-English papers face substantially higher
    
[^161]: 一种普遍的“氛围”？利用稀疏自编码器（SAEs）寻找并控制跨语言的非正式语域

    A Universal Vibe? Finding and Controlling Language-Agnostic Informal Register with SAEs

    [https://arxiv.org/abs/2603.26236](https://arxiv.org/abs/2603.26236)

    研究发现多语言模型中存在一个跨语言的“非正式语域”共享核心子空间，可通过稀疏自编码器进行定位和控制，表明俚语等语用语域是以统一抽象概念而非孤立的特定语言记忆被处理的。

    

    尽管多语言语言模型能够成功地在不同语言之间迁移事实性和句法知识，但它们究竟是以孤立的、特定语言记忆的方式，还是以统一的抽象概念的方式来处理文化特定的语用语域（如俚语），目前仍不清楚。我们通过使用稀疏自编码器（SAEs）探测 Gemma-2-9B-IT 的内部表示来研究这一问题，涵盖英语、希伯来语和俄语三种类型学上差异显著的源语言。为了将语用语域处理与简单的词汇敏感性彻底区分开来，我们引入了一个新颖的数据集，其中每个目标词都是多义词，会同时出现在字面意义和非正式语境中。我们发现，虽然大部分非正式语域信号分布在特定语言的特征中，但一个规模虽小却高度稳健的跨语言核心始终稳定出现。这一共享核心形成了一个几何上连贯的“非正式语域子空间”……

    arXiv:2603.26236v2 Announce Type: replace  Abstract: While multilingual language models successfully transfer factual and syntactic knowledge across languages, it remains unclear whether they process culture-specific pragmatic registers, such as slang, as isolated language-specific memorizations or as unified, abstract concepts. We study this by probing the internal representations of Gemma-2-9B-IT using Sparse Autoencoders (SAEs) across three typologically diverse source languages: English, Hebrew, and Russian. To definitively isolate pragmatic register processing from trivial lexical sensitivity, we introduce a novel dataset in which every target term is polysemous, appearing in both literal and informal contexts. We find that while much of the informal-register signal is distributed across language-specific features, a small but highly robust cross-linguistic core consistently emerges. This shared core forms a geometrically coherent ``informal register subspace'' that sharpens in th
    
[^162]: FDARxBench：基于FDA仿制药评估的监管与临床推理基准测试

    FDARxBench: Benchmarking Regulatory and Clinical Reasoning on FDA Generic Drug Assessment

    [https://arxiv.org/abs/2603.19539](https://arxiv.org/abs/2603.19539)

    该论文与FDA监管审查员合作，提出了首个基于FDA药品标签文档、由专家精心策划的仿制药评估问答基准FDARxBench，涵盖事实性、多跳推理和拒答任务，实验揭示了当前语言模型在事实依据、长上下文检索和安全拒答方面存在重大不足。

    

    我们提出了一个由专家精心策划的真实世界基准测试，用于评估基于文档的问答（QA）能力，该基准受仿制药评估需求启发，基于美国食品药品监督管理局（FDA）的药品标签文档构建。药品标签包含丰富但异构的临床和监管信息，这使得当前语言模型难以进行准确的问答。通过与FDA监管审查员合作，我们提出了FDARxBench，并构建了一个多阶段流程，用于生成高质量的、由专家精心策划的问答示例，涵盖事实性、多跳推理和拒答任务，同时设计了评估协议来评估开卷和闭卷两种推理模式。在专有模型和开放权重模型上的实验揭示了当前模型在事实依据、长上下文检索和安全拒答行为方面存在显著差距。虽然该基准源于FDA仿制药评估的实际需求，但它也为具有挑战性的……提供了坚实的基础。

    arXiv:2603.19539v2 Announce Type: replace-cross  Abstract: We introduce an expert curated, real-world benchmark for evaluating document-grounded question-answering (QA) motivated by generic drug assessment, using the U.S. Food and Drug Administration (FDA) drug label documents. Drug labels contain rich but heterogeneous clinical and regulatory information, making accurate question answering difficult for current language models. In collaboration with FDA regulatory assessors, we introduce FDARxBench, and construct a multi-stage pipeline for generating high-quality, expert curated, QA examples spanning factual, multi-hop, and refusal tasks, and design evaluation protocols to assess both open-book and closed-book reasoning. Experiments across proprietary and open-weight models reveal substantial gaps in factual grounding, long-context retrieval, and safe refusal behavior. While motivated by FDA generic drug assessment needs, this benchmark also provides a substantial foundation for chall
    
[^163]: 基于对数似然向量的面向提示-回复分布的语言模型地图

    Language Model Maps for Prompt-Response Distributions via Log-Likelihood Vectors

    [https://arxiv.org/abs/2603.18593](https://arxiv.org/abs/2603.18593)

    该论文提出用提示-回复对上的对数似然向量表示语言模型并构建模型地图，使模型间的欧氏距离近似对应条件分布的KL散度，从而捕捉模型属性与任务性能的全局结构，预测下游任务得分，并在无需直接观察的情况下近似复合提示操作的效果。

    

    我们提出了一种方法，通过语言模型在提示-回复对上的对数似然向量来表示该模型，并构建模型地图用于比较各模型的条件分布。在该空间中，模型之间的欧氏距离平方与对应条件分布之间的KL散度近似成正比。在大量公开可用的语言模型上进行的实验表明，这些地图能够捕捉有意义的核心结构，包括与模型属性及任务性能之间的关系。该表示还能捕捉由提示修改引起的系统性偏移及其近似的可加组合性；我们利用这些向量来预测下游任务得分，并借助其可加结构，在无需直接观察相应对数似然向量的情况下近似复合提示操作的效果。我们进一步引入PMI向量以减少（摘要在此处截断）

    arXiv:2603.18593v2 Announce Type: replace  Abstract: We propose a method that represents language models by log-likelihood vectors over prompt-response pairs and constructs model maps for comparing their conditional distributions. In this space, squared Euclidean distances between models are approximately proportional to the KL divergence between the corresponding conditional distributions. Experiments on a large collection of publicly available language models show that the maps capture meaningful global structure, including relationships to model attributes and task performance. The representation also captures systematic shifts induced by prompt modifications and their approximate additive compositionality; we use the vectors to predict downstream task scores and leverage their additive structure to approximate the effects of composite prompt operations without directly observing the corresponding log-likelihood vectors. We further introduce PMI vectors to reduce the influence of un
    
[^164]: ICE：面向大语言模型的基于统计基础的干预一致性解释评估

    ICE: Intervention-Consistent Explanation Evaluation with Statistical Grounding for LLMs

    [https://arxiv.org/abs/2603.18579](https://arxiv.org/abs/2603.18579)

    提出ICE框架，通过在多种干预算子下将模型解释与同等规模的随机基线进行统计对比，首次揭示了大语言模型的解释忠实性是依赖干预方法的量而非固定属性（切换算子导致差距高达44个百分点），并能检测出比随机表现更差的反忠实性现象。

    

    评估解释是否忠实反映模型的推理过程仍然是一个开放性问题。现有基准采用单一干预且缺乏统计检验，因此无法区分真正的忠实性与随机水平的表现。我们表明，忠实性并非固定属性，而是一个依赖算子的量，会随测量它所使用的干预方法而变化。我们提出ICE（干预一致性解释）框架，该框架在多个算子下将解释与同等规模的随机基线进行对比评估。通过对7个大语言模型在4个任务上使用删除和检索填充算子进行评估，我们发现切换算子会在18%的配置（28个注意力比较中的5个）中跨越正证据阈值，差距可达44个百分点。随机化基线在近三分之一的英文删除配置中检测到反忠实性（即解释比随机还差）……

    arXiv:2603.18579v2 Announce Type: replace-cross  Abstract: Evaluating whether explanations faithfully reflect a model's reasoning remains an open problem. Existing benchmarks use single interventions without statistical testing, making it impossible to distinguish genuine faithfulness from chance-level performance. We show that faithfulness is not a fixed property but an operator-dependent quantity that changes with the intervention method used to measure it. We introduce ICE (Intervention-Consistent Explanation), a framework that evaluates explanations against random baselines of equal size under multiple operators. Evaluating 7 LLMs across 4 tasks with deletion and retrieval infill operators, we find that switching operators crosses the positive-evidence threshold in 18% of configurations (5 of 28 attention comparisons), with gaps reaching 44 percentage points. Randomized baselines detect anti-faithfulness (explanations worse than random) in nearly one-third of English deletion confi
    
[^165]: 平庸之道：LLM作为评判者的锚点选择关键

    Mediocrity is the key for LLM as a Judge Anchor Selection

    [https://arxiv.org/abs/2603.16848](https://arxiv.org/abs/2603.16848)

    研究发现，在LLM作为评判者的基准测试中，选择表现“平庸”（中等水平）的模型作为锚点最为可靠，而常见的极端锚点（最强或最弱模型）会显著降低模型排名的可靠性。

    

    “LLM作为评判者”范式已成为评估开放式生成任务的标准方法。为了解决成对比较带来的二次方可扩展性成本问题，诸如Arena-Hard和AlpacaEval等流行的基准测试会将所有模型与单一锚点模型进行比较。然而，尽管这种做法被广泛使用，锚点选择对结果可靠性的影响却在很大程度上未被探索。在这项工作中，我们通过在Arena-Hard-v2.0数据集上评估22个不同的锚点，系统地研究了锚点选择的影响。我们发现锚点的选择至关重要：一个糟糕的锚点会显著降低与人类排名的相关性。我们指出，常见的锚点选择方式（表现最好和表现最差的模型）并不适合作为锚点。因为这些极端锚点始终优于或劣于所有其他模型，所以它们很难反映模型之间的相对排名。我们进一步量化了……

    arXiv:2603.16848v2 Announce Type: replace  Abstract: The ``LLM-as-a-judge'' paradigm has become a standard method for evaluating open-ended generation. To address the quadratic scalability costs of pairwise comparisons, popular benchmarks like Arena-Hard and AlpacaEval compare all models against a single anchor. However, despite its widespread use, the impact of anchor selection on the reliability of the results remains largely unexplored. In this work, we systematically investigate the effect of anchor selection by evaluating 22 different anchors on the Arena-Hard-v2.0 dataset. We find that the choice of anchor is critical: a poor anchor can dramatically reduce correlation with human rankings. We identify that common anchor choices (best-performing and worst-performing models) make poor anchors. Because these extreme anchors are consistently better or worse than all other models, they are seldom indicative of the relative ranking of the models. We further quantify the effect size of a
    
[^166]: 通过作者画像探测大语言模型中的文化信号

    Probing Cultural Signals in Large Language Models through Author Profiling

    [https://arxiv.org/abs/2603.16749](https://arxiv.org/abs/2603.16749)

    本研究通过零样本歌词作者画像任务揭示了大语言模型中的系统性文化偏见——多数模型默认偏向北美族裔而DeepSeek-1.5B更对齐亚洲族裔，并创新性地提出MAD和RD两个公平性指标来量化这些差异。

    

    大语言模型（LLM）正日益被部署于具有社会影响的应用中，这引发了人们对其所编码的文化偏见的担忧。我们通过评估大语言模型能否在零样本设置下从歌词中进行作者画像来探测这些表征，即在不进行任务特定微调的情况下推断歌手的性别和族裔。在超过10,000首歌词上对多个开源模型进行评估后发现，大语言模型取得了不俗的画像性能，但表现出系统性的文化对齐：大多数模型默认偏向北美族裔，而DeepSeek-1.5B则与亚洲族裔的对齐更强。这一发现既体现在模型的预测分布中，也体现在对其所生成理由的分析中。为了量化这些差异，我们引入了两个公平性指标——模态准确率散度（MAD）和召回率散度（RD），并表明Ministral-8B显示出最强的族裔（摘要在此处似乎被截断）

    arXiv:2603.16749v3 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly deployed in applications with societal impact, raising concerns about the cultural biases they encode. We probe these representations by evaluating whether LLMs can perform author profiling from song lyrics in a zero-shot setting, inferring singers' gender and ethnicity without task-specific fine-tuning. Across several open-source models evaluated on more than 10,000 lyrics, we find that LLMs achieve non-trivial profiling performance but demonstrate systematic cultural alignment: most models default toward North American ethnicity, while DeepSeek-1.5B aligns more strongly with Asian ethnicity. This finding emerges from both the models' prediction distributions and an analysis of their generated rationales. To quantify these disparities, we introduce two fairness metrics, Modality Accuracy Divergence (MAD) and Recall Divergence (RD), and show that Ministral-8B displays the strongest ethnic
    
[^167]: GONE：基于邻域扩展分布塑造的结构化知识遗忘

    GONE: Structural Knowledge Unlearning via Neighborhood-Expanded Distribution Shaping

    [https://arxiv.org/abs/2603.12275](https://arxiv.org/abs/2603.12275)

    本文提出了GONE基准用于评估大型语言模型对结构化知识图谱事实的遗忘效果，能够解耦直接事实移除、推理泄漏和灾难性遗忘三种效应，并设计了邻域扩展分布塑造（NEDS）这一新型遗忘框架。

    

    在大型语言模型（LLMs）中，知识遗忘是一项紧迫且具有挑战性的任务，因为LLMs具有前所未有的记忆和消化大规模训练数据的能力，这在安全性、隐私和知识产权方面引发了更为重大的问题。然而，现有工作（包括参数编辑、微调和基于蒸馏的方法）都专注于扁平的句子级数据，而忽视了自然结构化数据中关系型、多跳和推理性的知识。针对这一空白，本文提出了图遗忘与节点擦除（Graph Oblivion and Node Erasure, GONE），一个用于评估大型语言模型中结构化知识图谱（KG）事实知识遗忘的基准。该基于KG的基准能够解耦遗忘的三种效应：直接事实移除、基于推理的知识泄漏以及灾难性遗忘。此外，本文还设计了一种新颖的遗忘框架——邻域扩展分布塑造（Neighborhood-Expanded Distribution Shaping, NEDS）。

    arXiv:2603.12275v2 Announce Type: replace  Abstract: Unlearning knowledge is a pressing and challenging task in Large Language Models (LLMs) because of their unprecedented capability to memorize and digest training data at scale, raising more significant issues regarding safety, privacy, and intellectual property. However, existing works, including parameter editing, fine-tuning, and distillation-based methods, are all focused on flat sentence-level data but overlook the relational, multi-hop, and reasoned knowledge in naturally structured data. In response to this gap, this paper introduces Graph Oblivion and Node Erasure (GONE), a benchmark for evaluating knowledge unlearning over structured knowledge graph (KG) facts in LLMs.This KG-based benchmark enables the disentanglement of three effects of unlearning: direct fact removal, reasoning-based leakage, and catastrophic forgetting. In addition, Neighborhood-Expanded Distribution Shaping (NEDS), a novel unlearning framework, is design
    
[^168]: TikZilla：通过高质量数据与强化学习扩展文本到TikZ的生成

    TikZilla: Scaling Text-to-TikZ with High-Quality Data and Reinforcement Learning

    [https://arxiv.org/abs/2603.03072](https://arxiv.org/abs/2603.03072)

    该论文通过构建规模扩大四倍以上且质量更高的DaTikZ-V4数据集，并结合强化学习（而非仅用监督微调）来扩展Text-to-TikZ生成，以解决文本与图形不匹配及循环、无关内容等渲染错误问题。

    

    大语言模型（LLMs）正越来越多地被用于协助科学家的各类工作流程。一个关键挑战是从文本描述生成高质量图形，这些图形通常以TikZ程序的形式表示，并可渲染为科学图像。先前的研究已为该任务提出了多种数据集和建模方法。然而，现有的Text-to-TikZ数据集规模太小且噪声过多，无法捕捉TikZ的复杂性，导致文本与渲染图形之间的不匹配。此外，先前的方法仅依赖监督微调（SFT），未能让模型接触到图形的渲染语义，常常导致循环、无关内容和错误空间关系等问题。为解决这些问题，我们构建了DaTikZ-V4数据集，其规模是DaTikZ-V3的四倍以上，质量显著更高，并加入了由LLM生成的图形描述进行丰富。利用该数据集……

    arXiv:2603.03072v3 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly used to assist scientists across diverse workflows. A key challenge is generating high-quality figures from textual descriptions, often represented as TikZ programs that can be rendered as scientific images. Prior research has proposed a variety of datasets and modeling approaches for this task. However, existing datasets for Text-to-TikZ are too small and noisy to capture the complexity of TikZ, causing mismatches between text and rendered figures. Moreover, prior approaches rely solely on supervised fine-tuning (SFT), which does not expose the model to the rendered semantics of the figure, often resulting in errors such as looping, irrelevant content, and incorrect spatial relations. To address these issues, we construct DaTikZ-V4, a dataset more than four times larger and substantially higher in quality than DaTikZ-V3, enriched with LLM-generated figure descriptions. Using this dataset
    
[^169]: CLASE：一种用于中文法律文本文体评估的混合方法

    CLASE: A Hybrid Method for Chinese Legalese Stylistic Evaluation

    [https://arxiv.org/abs/2602.12639](https://arxiv.org/abs/2602.12639)

    该论文提出了CLASE，一种针对中文法律文本的混合式文体评估方法，解决了法律专家难以人工制定评分标准、基于参考的指标混淆语义与文体、以及LLM作为裁判不透明且不一致等评估难题。

    

    大型语言模型（LLM）生成的法律文本通常可以达到较为合理的事实准确性，但往往无法遵循法律写作的专业文体规范和语言惯例。为了提升文体质量，关键的第一步是建立一种可靠的评估方法。然而，让法律专家手动开发这样的评估指标并不现实，因为法律写作实践中的隐含文体要求难以被形式化为明确的评分标准。与此同时，现有的自动评估方法也存在不足：基于参考文本的指标将语义准确性与文体忠实度混为一谈，而以LLM作为裁判的评估方式则存在不透明和不一致的问题。为应对这些挑战，我们提出了CLASE（中文法律文本文体评估），一种专注于法律文本文体表现的混合评估方法。该方法采用了混合评分机制...

    arXiv:2602.12639v2 Announce Type: replace  Abstract: Legal text generated by large language models (LLMs) can usually achieve reasonable factual accuracy, but it frequently fails to adhere to the specialised stylistic norms and linguistic conventions of legal writing. In order to improve stylistic quality, a crucial first step is to establish a reliable evaluation method. However, having legal experts manually develop such a metric is impractical, as the implicit stylistic requirements in legal writing practice are difficult to formalise into explicit rubrics. Meanwhile, existing automatic evaluation methods also fall short: reference-based metrics conflate semantic accuracy with stylistic fidelity, and LLM-as-a-judge evaluations suffer from opacity and inconsistency. To address these challenges, we introduce CLASE (Chinese LegAlese Stylistic Evaluation), a hybrid evaluation method that focuses on the stylistic performance of legal text. The method incorporates a hybrid scoring mechani
    
[^170]: 审阅审稿人：基于大语言模型的审稿人反馈生成以促进审稿指南合规

    Reviewing the Reviewer: LLM-Assisted Reviewer Feedback Generation for Guideline Compliance

    [https://arxiv.org/abs/2602.10118](https://arxiv.org/abs/2602.10118)

    该论文提出一个基于大语言模型的推理时框架，通过将评审分解为论证片段、识别违反ACL滚动评审指南的问题并利用迭代重排序算法生成针对性反馈，帮助审稿人改进评审质量并提升指南合规性。

    

    同行评审是科学质量的核心，然而对简单启发式方法的依赖——即懒惰思维和非特异性批评——已经威胁到评审质量。先前的工作将懒惰思维检测框定为单标签分类问题，并且止步于检测，但评审片段往往表现出多个共现问题，而且审稿人从可操作的、符合指南的反馈中获益要多于仅获得标签。我们进一步表明，直接提示现成大语言模型生成反馈时，其经常重写整篇评审或面向作者而非审稿人，这促使我们提出一种推理时（inference-time）的方法。我们引入一个由大语言模型驱动的框架，该框架将评审分解为论证片段，识别违反 ACL 滚动评审（ARR）指南的问题，并使用针对特定问题的模板生成有针对性的反馈，这些模板通过一种新颖的迭代式、基于重排序的生成算法进行精炼。在一项受控重写研究中，我们的反馈减少了指南……

    arXiv:2602.10118v2 Announce Type: replace  Abstract: Peer review is central to scientific quality, yet reliance on simple heuristics, namely lazy thinking and non-specific critiques, has threatened review quality. Prior work frames lazy thinking detection as single-label classification and stops at detection, yet review segments often exhibit multiple co-occurring issues, and reviewers benefit more from actionable, guideline-aware feedback than from labels alone. We further show that off-the-shelf LLMs prompted for feedback frequently rewrite the entire review or address the authors rather than the reviewer, motivating an inference-time approach. We introduce an LLM-driven framework that decomposes reviews into argumentative segments, identifies issues violating ACL Rolling Review (ARR) guidelines, and generates targeted feedback using issue-specific templates refined by a novel iterative, reranking-based generation algorithm. In a controlled rewriting study, our feedback reduces guide
    
[^171]: 约束组相对策略优化

    Constrained Group Relative Policy Optimization

    [https://arxiv.org/abs/2602.05863](https://arxiv.org/abs/2602.05863)

    本文提出约束GRPO（Constrained GRPO），一种基于拉格朗日方法的GRPO扩展用于约束策略优化，并揭示了在归一化前对标量化奖励会导致共享分母耦合，使改变一个约束乘子会同时影响奖励与其他约束的相对权重这一关键失败模式。

    

    组相对策略优化（GRPO）仍然是大语言模型（LLM）和视觉语言模型（VLM）微调中占主导地位的无评论家（critic-free）方法，但其与约束策略优化（例如在安全关键领域）的兼容性尚未得到仔细研究。在本工作中，我们提出了约束GRPO（Constrained GRPO），这是一种基于拉格朗日方法的GRPO扩展，用于约束策略优化。我们证明，在归一化之前对标量化奖励这一标准做法会引入一种关键的拉格朗日特有的失败模式：GRPO的组内归一化使得约束优化对多组件学习信号的聚合方式高度敏感。我们进一步证明，在归一化之前对标量化奖励会引入共享分母耦合，使得改变某一个乘子不仅会改变其对应约束的受重视程度，还会改变奖励与其他约束之间的相对权重。我们通过一个简单但关键的修改来解决这一问题。

    arXiv:2602.05863v3 Announce Type: replace-cross  Abstract: Group Relative Policy Optimization (GRPO) remains the dominant critic-free approach for fine-tuning LLMs and VLMs, but its compatibility with constrained policy optimization (e.g. for safety-critical domains) has not been carefully examined. In this work, we introduce Constrained GRPO, a Lagrangian-based extension of GRPO for constrained policy optimization. We show that the standard practice of scalarizing rewards before normalization introduces a critical Lagrangian-specific failure mode: GRPO's within-group normalization makes constrained optimization highly sensitive to how multi-component learning signals are aggregated. We show that scalarizing rewards before normalization introduces shared-denominator coupling, so that changing one multiplier alters not only the emphasis on its corresponding constraint, but also the relative weighting of the reward and other constraints. We address this with a simple but crucial modifica
    
[^172]: 模块化专家合并用于生物医学检索

    Modular Expert Merging for Biomedical Retrieval

    [https://arxiv.org/abs/2602.04731](https://arxiv.org/abs/2602.04731)

    本文提出模块化专家合并方法，通过合成难负样本和LoRA微调领域专家并合并，在生物医学检索上优于大规模混合训练，兼顾通用性能。

    

    arXiv:2602.04731v2 公告类型：替换 摘要：将通用大型语言模型适配为领域专用的密集检索器通常需要在混合领域数据上进行大规模训练。我们表明，合并独立训练的领域专用专家在四个仅解码器LLM家族（0.6B-7B）、四种合并方法和来自MTEB的十二项医学及通用检索任务中持续优于这种方法，这表明参数空间组合捕捉了互补的领域优势，而大规模混合领域训练则将其平均化。为了进一步最大化专家质量，我们引入了Synthesize-Train-Merge（STM），这是一个模块化框架，它使用顶级LLM合成难负样本，并通过LoRA微调领域专用专家后再进行合并，无需持续预训练。合成的难负样本对较小模型带来最大收益，STM在生物医学检索任务上实现了强劲性能，同时保持了具有竞争力的通用领域结果。

    arXiv:2602.04731v2 Announce Type: replace  Abstract: Adapting general-purpose LLMs into domain-specialized dense retrievers typically requires large-scale training on mixed-domain data. We show that merging independently trained domain-specialized experts consistently exceeds this approach across four decoder-only LLM families (0.6B-7B), four merging methods, and twelve medical and general retrieval tasks from MTEB, suggesting that parameter-space composition captures complementary domain strengths that large-scale mixed-domain training averages out. To further maximize expert quality, we introduce Synthesize-Train-Merge (STM), a modular framework that synthesizes hard negatives with a top-tier LLM and fine-tunes domain-specialized experts via LoRA before merging them, without continual pre-training. Synthesized hard negatives yield the largest gains for smaller models, and STM achieves strong performance on biomedical retrieval tasks while maintaining competitive general-domain result
    
[^173]: 从人类偏好中学习查询特定的评分准则以用于深度研究报告生成

    Learning Query-Specific Rubrics from Human Preferences for DeepResearch Report Generation

    [https://arxiv.org/abs/2602.03619](https://arxiv.org/abs/2602.03619)

    本文提出通过强化学习从人类偏好标注数据中训练查询特定的评分准则生成器，采用混合奖励（偏好一致性、格式有效性、LLM评估）来解决深度研究报告生成中训练与评估缺乏可验证奖励信号的难题。

    

    如今，开发可靠的深度研究（DeepResearch）风格的长篇报告生成仍然具有挑战性，因为其训练和评估缺乏可验证的奖励信号。因此，基于评分准则的评估已成为一种常见做法。然而，现有方法要么依赖于缺乏足够粒度的粗糙预定义评分准则，要么依赖于成本高昂且难以规模化的人工构建的查询特定评分准则。在本文中，我们提出了一个流程，用于训练以人类偏好为基础、面向深度研究报告生成的查询特定评分准则生成器。我们首先构建了一个深度研究风格查询的数据集，其中包含对配对报告的人类偏好标注，并通过强化学习训练评分准则生成器，采用结合偏好一致性、格式有效性和基于大语言模型（LLM）的评分准则评估的混合奖励。我们在两个阶段对所得到的评分准则生成器进行评估。首先，在留出的人类偏好测试集上……

    arXiv:2602.03619v3 Announce Type: replace  Abstract: Nowadays, developing reliable DeepResearch-style long-form report generation remains challenging, as training and evaluation lack verifiable reward signals. Accordingly, rubric-based evaluation has become a common practice. However, existing approaches either rely on coarse, pre-defined rubrics that lack sufficient granularity or depend on manually constructed query-specific rubrics that are costly and difficult to scale. In this paper, we propose a pipeline to train preference-grounded query-specific rubric generators tailored for DeepResearch report generation. We first construct a dataset of DeepResearch-style queries annotated with human preferences over paired reports, and train rubric generators via reinforcement learning with a hybrid reward combining preference consistency, format validity, and LLM-based rubric evaluation. We evaluate the resulting rubric generators in two stages. First, on a held-out human-preference test se
    
[^174]: CALIBURN：自校准的大语言模型遗忘对齐

    CALIBURN: Self-Calibrated LLM Unlearning Alignment

    [https://arxiv.org/abs/2602.02824](https://arxiv.org/abs/2602.02824)

    我们提出了一种自校准遗忘方法，通过量化模型对不良知识的置信度来精确调整梯度更新，在实现细粒度遗忘的同时减少对保留数据的依赖，从而提升模型效用。

    

    大语言模型遗忘旨在从预训练语言模型中移除不良知识的影响，这为解决安全和隐私问题提供了一种实用机制。现有的遗忘方法，如梯度上升，容易导致灾难性遗忘。基于对齐的方法提供了另一种方向，但其有效性受限于参考模型的质量。在现实场景中，这两种方法仍需要大量保留数据集来维持通用知识。我们提出了一种原则性方法，该方法量化目标大语言模型对不良知识的置信度，并利用该置信度更精确地校准模型的遗忘梯度更新。它能够实现对遗忘的细粒度控制，同时更好地保持模型效用，从而减少对保留数据或过高的遗忘训练数据的依赖。在包括MUSE和WMDP在内的多个基准上的广泛评估表明，该方法表现出色。

    arXiv:2602.02824v2 Announce Type: replace  Abstract: LLM unlearning aims to remove the influence of undesirable knowledge from pretrained language models, which offers a practical mechanism for addressing safety and privacy concerns. Existing unlearning approaches, such as Gradient Ascent, are prone to catastrophic forgetting. Alignment-based approaches provide an alternative direction, yet their effectiveness is limited by the quality of the reference model. In realistic settings, both methods still require large retention datasets to preserve general knowledge. We propose a principled method that quantifies the target LLM's confidence in undesirable knowledge and uses it to calibrate the model's unlearning gradient updates more precisely. It enables fine-grained control over forgetting while better preserving model utility, thus reducing the dependence on retention data or prohibitive unlearning training data. Extensive evaluations on multiple benchmarks, including MUSE and WMDP, sho
    
[^175]: 大语言模型中的文化根基化人格：与心理社会价值框架的刻画与对齐

    Culturally Grounded Personas in Large Language Models: Characterization and Alignment with Socio-Psychological Value Frameworks

    [https://arxiv.org/abs/2601.22396](https://arxiv.org/abs/2601.22396)

    该论文提出基于世界价值观调查（WVS）的可解释变量生成具有文化根基的LLM合成人格，并从英格尔哈特-韦尔泽尔文化地图定位、人口层面一致性和道德基础理论三个互补视角，验证这些人格与人类社会心理价值框架的对齐程度。

    

    尽管大语言模型（LLMs）在模拟人类行为方面的应用日益广泛，但这些合成人格在多大程度上准确反映了不同文化条件下的世界观与道德价值体系仍不确定。本文研究了合成的、具有文化根基的人格与既有框架的对齐情况，具体包括世界价值观调查（WVS）、英格尔哈特-韦尔泽尔文化地图以及道德基础理论。我们基于一组可解释的、源自WVS的变量概念化并生成了LLM人格，并通过三个互补的视角对这些生成的人格进行考察：在英格尔哈特-韦尔泽尔地图上的定位，揭示其反映了不同文化条件之间的稳定差异；与世界价值观调查在人口统计层面的一致性，其回答分布大体上遵循人类群体的模式；以及源自道德基础理论的道德画像。

    arXiv:2601.22396v3 Announce Type: replace-cross  Abstract: Despite the growing utility of Large Language Models (LLMs) for simulating human behavior, the extent to which these synthetic personas accurately reflect world and moral value systems across different cultural conditionings remains uncertain. This paper investigates the alignment of synthetic, culturally-grounded personas with established frameworks, specifically the World Values Survey (WVS), the Inglehart-Welzel Cultural Map, and Moral Foundations Theory. We conceptualize and produce LLM-generated personas based on a set of interpretable WVS-derived variables, and we examine the generated personas through three complementary lenses: positioning on the Inglehart-Welzel map, which unveils their interpretation reflecting stable differences across cultural conditionings; demographic-level consistency with the World Values Survey, where response distributions broadly track human group patterns; and moral profiles derived from a M
    
[^176]: ChartAttack：测试大语言模型在图表生成中对恶意提示的脆弱性

    ChartAttack: Testing the Vulnerability of LLMs to Malicious Prompting in Chart Generation

    [https://arxiv.org/abs/2601.12983](https://arxiv.org/abs/2601.12983)

    本文提出了ChartAttack框架和AttackViz数据集，首次系统评估了多模态大语言模型在图表生成中利用设计误导元素诱导错误解读的能力，攻击可显著降低模型和人类的问答准确率，且在AttackViz上微调可提升模型的鲁棒性。

    

    多模态大语言模型（MLLM）越来越多地被用于从数据表格自动生成图表，这提高了效率但也带来了新的滥用风险。我们提出了ChartAttack，一个用于评估MLLM如何利用设计误导元素生成会诱导错误解读的图表的框架。我们还介绍了AttackViz，一个图表问答（QA）数据集，其中标注了有效的误导元素及其诱导的错误答案。ChartAttack使MLLM的问答准确率在域内降低了17.2个百分点，跨域降低了11.9个百分点。条件欺骗率显示出针对性攻击效果：在域内，原本正确的答案有11.2%的概率被转变为攻击者预期的答案，跨域为11.7%至14.9%，而原本错误的答案很少发生变化。一项对照人类实验表明，ChartAttack生成的图表也会降低人类的问答表现。最后，在AttackViz上进行微调可以提高MLLM对误导性图表的域内鲁棒性。

    arXiv:2601.12983v4 Announce Type: replace  Abstract: Multimodal large language models (MLLMs) are increasingly used to automate chart generation from data tables, improving efficiency but introducing new misuse risks. We present ChartAttack, a framework for evaluating how MLLMs use design misleaders to generate charts that induce incorrect interpretations. We also introduce AttackViz, a chart question-answering (QA) dataset labeled with effective misleaders and their induced incorrect answers. ChartAttack reduces MLLM QA accuracy by 17.2 points in-domain and 11.9 points cross-domain. Conditional deception rates show targeted effects: correct answers shift to attacker-intended answers 11.2\% of the time in-domain and 11.7-14.9\% cross-domain, while originally incorrect answers rarely change. A controlled human study shows that ChartAttack-generated charts also reduce human QA performance. Finally, fine-tuning on AttackViz improves in-domain MLLM robustness to misleading charts. Our find
    
[^177]: 超越迁移准确率：面向低资源语言的机制引导可控适配

    Beyond Transfer Accuracy: Mechanism-Guided Controlled Adaptation for Low-Resource Languages

    [https://arxiv.org/abs/2601.08146](https://arxiv.org/abs/2601.08146)

    该论文提出了一种无需反事实的回路发现方法，并据此提出回路定向监督微调（CT-SFT），仅更新任务相关的注意力头和LayerNorm，从而在低资源语言适配中既保持竞争力又最有效地避免灾难性遗忘。

    

    现有的回路发现方法依赖于具有清晰反事实的模板化任务，限制了其在多样化自然文本上的应用。我们通过标签平衡的激活均值和任务方向相关性评分，将变换器上下文分解方法（CD-T）适配到非结构化设置中，实现了无需反事实的回路发现。我们利用所发现的回路提出回路定向监督微调（CT-SFT），将参数更新限制在任务相关的注意力头和LayerNorm上。在NusaX跨语言情感迁移任务上的实验表明，CT-SFT在低资源适配方面极具竞争力。虽然非回路的稀疏更新和全量微调有时能通过容量招募达到相当的目标准确率，但CT-SFT最为一致地避免了灾难性遗忘，保留了源语言及相关任务的性能。在XNLI上的扩展实验在更难的任务上进一步支持了关于源语言保留和干预的发现。

    arXiv:2601.08146v4 Announce Type: replace-cross  Abstract: Existing circuit discovery methods rely on templated tasks with clean counterfactuals, limiting their use on diverse natural text. We adapt Contextual Decomposition for Transformers (CD-T) for unstructured settings via label-balanced activation means and task-directional relevance scoring, enabling counterfactual-free circuit discovery. We leverage the discovered circuits for Circuit-Targeted Supervised Fine-Tuning (CT-SFT), restricting parameter updates to task-relevant heads and LayerNorm. Experiments on NusaX cross-lingual sentiment transfer show that CT-SFT is highly competitive for low-resource adaptation. While non-circuit sparse updates and full fine-tuning sometimes match target accuracy through capacity recruitment, CT-SFT most consistently avoids catastrophic forgetting, preserving source-language and related-task performance. Extensions to XNLI support the source-retention and intervention findings on a harder task a
    
[^178]: 大语言模型玩不了猜单词游戏：论语言智能体私有工作记忆的必要性

    LLMs Can't Play Hangman: On the Necessity of a Private Working Memory for Language Agents

    [https://arxiv.org/abs/2601.06973](https://arxiv.org/abs/2601.06973)

    本文提出了私有状态交互任务（PSIT）并从理论上证明：仅依赖公开对话历史的语言智能体在架构上无法完成需要私有隐藏状态的任务（如猜单词游戏），从而论证了语言智能体必须配备私有工作记忆。

    

    随着大语言模型从文本补全走向自主智能体，它们仍然受限于缺乏私有工作记忆的标准聊天界面。这引出了一个根本性问题：智能体能否可靠地完成依赖隐藏状态的交互式任务？我们定义了私有状态交互任务，这类任务要求智能体在生成与固定隐藏状态保持一致的公开回复的同时，生成并维护隐藏信息。我们从理论上证明，任何仅能访问公开对话历史的智能体都无法在PSIT中既避免秘密在对话记录中泄露，又能给出与固定隐藏状态一致的回复，从而得出一个架构层面的不可能性定理。为了实证验证这一局限性，我们引入了一种自洽性测试协议，用于评估智能体能否在分叉的对话分支中持续维护隐藏的秘密。标准的基于聊天的LLM以及基于检索的记忆基线方法均告失败。

    arXiv:2601.06973v2 Announce Type: replace  Abstract: As LLMs move from text completion toward autonomous agents, they remain constrained by the standard chat interface, which lacks private working memory. This raises a fundamental question: can agents reliably perform interactive tasks that depend on hidden state? We define Private State Interactive Tasks (PSITs), which require agents to generate and maintain hidden information while producing public responses consistent with a fixed hidden state. We show theoretically that any agent restricted to the public conversation history cannot both keep the secret unresolved in the transcript and respond consistently with a fixed hidden state in PSITs, yielding an architectural impossibility theorem. To empirically validate this limitation, we introduce a self-consistency testing protocol that evaluates whether agents can maintain a hidden secret across forked dialogue branches. Standard chat-based LLMs and retrieval-based memory baselines fai
    
[^179]: Expos'ia：面向研究项目提案与同伴反馈的学术写作技能教学与评估

    Expos\'ia: Teaching and Assessment of Academic Writing Skills for Research Project Proposals and Peer Feedback

    [https://arxiv.org/abs/2601.06536](https://arxiv.org/abs/2601.06536)

    提出了首个连接高等教育中写作与反馈的公开数据集Expos'ia，包含学生研究项目提案、同伴与导师反馈以及基于教学理论的细粒度人工评分，并用于基准测试大语言模型在写作与反馈自动评分任务上的表现。

    

    我们提出了Expos'ia，这是首个将高等教育中写作与反馈相连接的公开数据集，为研究和开发基于教育理论的计算方法以教学和评估学术写作提供了基础。Expos'ia包含学生的研究项目提案，以及由同伴和导师给出的反馈（包括评论和自由文本评语）。该数据集收集自计算机科学专业的“科学工作导论”课程。Expos'ia反映了学术写作过程的多阶段特性，包括起草、接收反馈以及根据所获反馈修改写作。项目提案和同伴反馈均附有人工评估分数，这些分数基于我们开发的细粒度、以教学理论为依据的写作与反馈评估量表。我们利用Expos'ia对最先进的大语言模型（LLMs）在两项任务上进行基准测试：（1）对提案的自动评分……

    arXiv:2601.06536v3 Announce Type: replace  Abstract: We present Expos\'ia, the first public dataset that connects writing and feedback in higher education, enabling research on educationally grounded computational approaches to teaching and evaluating academic writing. Expos\'ia includes student research project proposals and peer and instructor feedback consisting of comments and free-text reviews. The dataset was collected in the "Introduction to Scientific Work" course of the Computer Science. Expos\'ia reflects the multi-stage nature of the academic writing process that includes drafting, receiving feedback, and revising the writing based on the feedback received. Both the project proposals and peer feedback are accompanied by human assessment scores based on a fine-grained, pedagogically-grounded schema for writing and feedback assessment that we develop.   We use Expos\'ia to benchmark state-of-the-art large language models (LLMs) on two tasks: automated scoring of (1) the propos
    
[^180]: CHisAgent：面向中国古代文化体系的事件分类体系构建多智能体框架

    CHisAgent: A Multi-Agent Framework for Event Taxonomy Construction in Ancient Chinese Cultural Systems

    [https://arxiv.org/abs/2601.05520](https://arxiv.org/abs/2601.05520)

    该论文提出CHisAgent多智能体框架，通过归纳、扩展、充实三个角色专业化阶段，从《二十四史》等中国古代文献中自动构建历史事件分类体系，克服了LLM在中国历史语境下推理能力不足和人工分类构建成本高的问题。

    

    尽管大型语言模型（LLM）在许多任务上表现出色，但其在历史与文化推理方面的能力有限，尤其是在中国历史等非英语语境中。分类体系结构为组织历史知识、提升理解提供了一种有效机制，然而人工构建分类体系成本高昂且难以规模化。因此，我们提出了CHisAgent，一个面向中国古代语境的历史分类体系构建多智能体LLM框架。CHisAgent将分类体系构建分解为三个角色专业化的阶段：自下而上的“归纳器”从原始历史语料中推导出初始层级结构；自上而下的“扩展器”利用LLM的世界知识补充缺失的中间概念；以及证据引导的“充实器”整合外部结构化历史资源以确保忠实性。利用《二十四史》……

    arXiv:2601.05520v2 Announce Type: replace  Abstract: Despite strong performance on many tasks, large language models (LLMs) show limited ability in historical and cultural reasoning, particularly in non-English contexts such as Chinese history. Taxonomic structures offer an effective mechanism to organize historical knowledge and improve understanding. However, manual taxonomy construction is costly and difficult to scale. Therefore, we propose \textbf{CHisAgent}, a multi-agent LLM framework for historical taxonomy construction in ancient Chinese contexts. CHisAgent decomposes taxonomy construction into three role-specialized stages: a bottom-up \textit{Inducer} that derives an initial hierarchy from raw historical corpora, a top-down \textit{Expander} that introduces missing intermediate concepts using LLM world knowledge, and an evidence-guided \textit{Enricher} that integrates external structured historical resources to ensure faithfulness. Using the \textit{Twenty-Four Histories}, 
    
[^181]: 智能体工具编排泄露更多：数据集、基准与缓解方法

    Agent Tools Orchestration Leaks More: Dataset, Benchmark, and Mitigation

    [https://arxiv.org/abs/2512.16310](https://arxiv.org/abs/2512.16310)

    该研究首次形式化了LLM智能体通过组合多个无害工具调用结果而泄露敏感信息的“工具编排隐私风险”（TOP-R），构建了包含1000个实例的TOP-Bench评测基准，并提出TOP-Align（SFT+DPO）训练方法来有效缓解该隐私泄露风险。

    

    大语言模型（LLM）智能体能够将各自单独看似不泄露信息的工具返回结果组合起来，进而推断出敏感结论，由此产生了“工具编排隐私风险”（Tools Orchestration Privacy Risk, TOP-R）。我们通过三个条件对TOP-R进行形式化定义：结论敏感性、单源不可推断性和组合可推断性。我们提出了基于库的反向推断种子扩展方法（LRSE），这是一个横跨四个工具库的反向构建流水线，并利用它构建了TOP-Bench——一个包含1,000个实例的基准数据集，在受控的两阶段工具使用协议下进行评估。在六个LLM智能体上，平均任务完成率、泄露率和H分数分别为98.0%、88.6%和20.4。在启用原生推理的情况下，四个模型的平均最终回答泄露率为81.4%，推理轨迹泄露率为82.4%。在禁用推理的情况下，三种仅依赖提示词的防护措施在TOP-Bench上将H分数平均提升约3.4分。我们进一步提出了TOP-Align，一种基于SFT+DPO的方法，用于学习更安全的工具使用行为。

    arXiv:2512.16310v4 Announce Type: replace-cross  Abstract: LLM agents can combine individually non-revealing tool returns and disclose a sensitive conclusion, creating Tools Orchestration Privacy Risk (TOP-R). We formalize TOP-R through three conditions: conclusion sensitivity, single-source non-inferability, and compositional inferability. We introduce Library-Grounded Reverse-Inference Seed Expansion (LRSE), a four-library reverse-construction pipeline, and use it to build TOP-Bench, a 1,000-instance benchmark evaluated under a controlled two-stage tool-use protocol. Across six LLM agents, average task completion, leakage, and H-score are 98.0 percent, 88.6 percent, and 20.4. With native reasoning enabled, four models average 81.4 percent final-response leakage and 82.4 percent reasoning-trace leakage. With reasoning disabled, three prompt-only safeguards improve H-score by an average of about 3.4 points on TOP-Bench. We further propose TOP-Align, an SFT+DPO method for learning safer
    
[^182]: GMTRouter：基于多轮用户交互的个性化LLM路由器

    GMTRouter: Personalized LLM Router over Multi-turn User Interactions

    [https://arxiv.org/abs/2511.08590](https://arxiv.org/abs/2511.08590)

    提出GMTRouter，将多轮用户-LLM交互建模为包含用户、LLM、查询、响应和轮次五种节点类型的异构图，以最大程度保留交互的关系结构，从而在用户偏好数据稀缺且格式不一致的情况下实现个性化的LLM路由。

    

    大语言模型（LLM）路由在平衡响应质量与计算成本方面已展现出强大能力。由于用户展现出多样化的偏好，个性化在LLM路由中受到越来越多的关注，因为即使是相同的查询也可能需要不同的模型来生成符合个人需求的响应。然而，现有方法并未实现完全个性化，且往往无法忠实地捕捉用户与LLM之间复杂的交互关系。此外，用户偏好数据通常稀缺且格式不一致，这限制了直接利用用户特定数据的方法的有效性。为了应对这些挑战，我们提出了GMTRouter，它将多轮用户-LLM交互表示为一个包含五种节点类型（用户、LLM、查询、响应和轮次）的异构图，从而最大程度地保留了交互中丰富的关系结构。通过轻量级的归纳图……

    arXiv:2511.08590v2 Announce Type: replace  Abstract: Large Language Model (LLM) routing has demonstrated strong capability in balancing response quality with computational cost. As users exhibit diverse preferences, personalization has attracted increasing attention in LLM routing, since even identical queries may require different models to generate responses tailored to individual needs. However, existing approaches are not fully personalized and often fail to faithfully capture the complex interactions between users and LLMs. Moreover, user preference data is typically scarce and inconsistent in format, which limits the effectiveness of methods that directly leverage user-specific data. To address these challenges, we propose GMTRouter, which represents multi-turn user-LLM interactions as a heterogeneous graph with five node types: user, LLM, query, response and turn, thereby maximally preserving the rich relational structure of the interaction. Through a lightweight inductive graph
    
[^183]: SignBind-LLM：面向手语翻译的多阶段模态融合

    SignBind-LLM: Multi-Stage Modality Fusion for Sign Language Translation

    [https://arxiv.org/abs/2509.00030](https://arxiv.org/abs/2509.00030)

    SignBind-LLM 通过三个分别处理连续手语、手指拼写和唇读的独立预训练专家流，利用轻量级 transformer 进行时间对齐融合，并结合预训练语言模型完成翻译，在无需人工词汇标注的情况下显著提升了手语翻译效果。

    

    当前的手语翻译（SLT）系统试图在单个端到端网络中学习手语的所有方面——包括手动手势、高速手指拼写以及异步的非手动面部线索。在缺乏细致监督的情况下学习多项任务，导致对手指拼写的专有名词和技术术语识别效果不佳，同时使来自唇部动作的丰富消歧信息在很大程度上未被充分利用。我们提出 SignBind-LLM，这是一个模块化框架，通过三个专门的专家流来解决上述局限：一个用于连续手语识别，一个用于手指拼写识别，一个用于唇读。每个专家均使用 CTC 在约两百万个自动生成的伪词汇（pseudo-gloss）序列上独立预训练，从而无需人工词汇标注。一个带有可学习时间对齐机制的轻量级 transformer 融合各专家的输出，再由预训练语言模型将所得的伪词汇序列进行翻译（摘要在此处被截断）。

    arXiv:2509.00030v4 Announce Type: replace  Abstract: Current sign language translation (SLT) systems attempt to learn all aspects of signing---manual gestures, high-speed fingerspelling, and asynchronous non-manual facial cues---within a single end-to-end network. Learning multiple tasks without detailed supervision leads to poor recognition of fingerspelled proper nouns and technical terms, and leaves rich disambiguating information from lip movements largely unexploited. We introduce SignBind-LLM, a modular framework that addresses these limitations through three dedicated expert streams: one for continuous signing, one for fingerspelling, and one for lipreading. Each expert is pre-trained independently using CTC on approximately two million automatically generated pseudo-gloss sequences, removing the need for manual gloss annotation. A lightweight transformer with learned temporal alignment fuses the expert outputs, and a pre-trained language model translates the resulting pseudo-gl
    
[^184]: HarmReduction：评估大语言模型在减少伤害信息提供方面以支持药物使用者的基准测试

    HarmReduction: Benchmarking LLMs in Harm Reduction Information Provision to Support People Who Use Drugs

    [https://arxiv.org/abs/2507.21815](https://arxiv.org/abs/2507.21815)

    本文提出了HarmReduction基准，通过包含2,160个问答-证据对的数据集，从安全边界检查、定量数值提供和多药物使用风险推断三项任务评估大语言模型在为药物使用者提供减少伤害信息时的准确性与安全风险。

    

    数百万人的福祉正受到药物使用危害的挑战。减少伤害作为一种公共卫生策略，提供非评判性、基于证据的信息，旨在改善健康结果并降低相关安全风险。一些大型语言模型（LLMs）已展现出高水平的医学推理能力，有望满足药物使用者（PWUD）的信息需求。然而，它们在相关任务中的表现仍在很大程度上未被探索。我们推出了HarmReduction，一个旨在评估大语言模型在减少伤害信息提供中准确性与安全风险的基准测试。该基准数据集（HR-Basic）包含2,160个问答-证据对，涵盖三项任务：检查安全边界、提供定量数值以及推断多药物使用风险。我们构建了指令和RAG方案，基于模型固有知识及整合的证据来评估模型行为。

    arXiv:2507.21815v2 Announce Type: replace  Abstract: Millions of individuals' well-being are challenged by the harms of substance use. Harm reduction as a public health strategy provides non-judgemental, evidence-based information intended to improve health outcomes and reduce associated safety risks. Some large language models (LLMs) have demonstrated a high level of medical reasoning, promising to address the information needs of people who use drugs (PWUD). However, their performance in relevant tasks remains largely unexplored. We introduce HarmReduction, a benchmark designed to evaluate LLMs' accuracy and safety risks in harm reduction information provision. The benchmark dataset (HR-Basic) has 2,160 question-answer-evidence pairs. The scope covers three tasks: checking safety boundaries, providing quantitative values, and inferring polysubstance use risks. We build the Instruction and RAG schemes to evaluate model behaviours based on their inherent knowledge and the integration o
    
[^185]: 在奥地利增值税法中使用大型语言模型进行法律决策：一项比较研究

    Using Large Language Models for Legal Decision-Making in Austrian Value-Added Tax Law: A Comparative Study

    [https://arxiv.org/abs/2507.08468](https://arxiv.org/abs/2507.08468)

    本文通过微调和检索增强生成（RAG）两种方法，在教科书案例与真实税务咨询案例上系统评估了大型语言模型在奥地利及欧盟增值税法律决策中的能力，确定了LLM系统的最佳配置并检验了其法律推理能力。

    

    本文对大型语言模型（LLM）在奥地利及欧盟增值税（VAT）法律框架内协助法律决策的能力进行了实验评估。在税务咨询实践中，客户通常以自然语言描述案例，这使得LLM成为支持自动化决策、减轻税务专业人员工作负担的理想候选工具。鉴于法律分析需要具备法律依据且论证充分，LLM容易产生幻觉的特性构成了重大挑战。实验聚焦于提升LLM性能的两种常用方法：微调（fine-tuning）和检索增强生成（RAG）。本研究将这两种方法应用于教科书式案例和来自某税务咨询公司的真实案例，以系统地确定基于LLM系统的最佳配置，并评估LLM的法律推理能力。研究结果表明……

    arXiv:2507.08468v2 Announce Type: replace  Abstract: This paper provides an experimental evaluation of the capability of large language models (LLMs) to assist in legal decision-making within the framework of Austrian and European Union value-added tax (VAT) law. In tax consulting practice, clients often describe cases in natural language, making LLMs a prime candidate for supporting automated decision-making and reducing the workload of tax professionals. Given the requirement for legally grounded and well-justified analyses, the propensity of LLMs to hallucinate presents a considerable challenge. The experiments focus on two common methods for enhancing LLM performance: fine-tuning and retrieval-augmented generation (RAG). In this study, these methods are applied on both textbook cases and real-world cases from a tax consulting firm to systematically determine the best configurations of LLM-based systems and assess the legal-reasoning capabilities of LLMs. The findings highlight the 
    
[^186]: DLM-One：用于单步序列生成的扩散语言模型

    DLM-One: Diffusion Language Models for One-Step Sequence Generation

    [https://arxiv.org/abs/2506.00290](https://arxiv.org/abs/2506.00290)

    DLM-One提出了一种基于分数蒸馏的框架，将扩散语言模型的生成过程压缩为单步，实现采样步数约2000倍、推理时间约500倍的加速，同时保持有竞争力的文本生成性能。

    

    本文介绍了DLM-One，这是一个基于分数蒸馏的框架，可实现连续扩散语言模型（DLM）的单步序列生成。DLM-One通过将学生模型输出的分数与前向扩散噪声空间中预训练教师DLM的分数函数对齐，从而消除了迭代精炼过程。我们证明了该框架与具体架构无关，并在多种连续流形上具有鲁棒性，包括标准的词嵌入空间和logit单纯形空间。通过对多个代表性扩散语言模型的实验，我们展示了DLM-One在采样步数上实现了高达约2000倍的加速，在墙钟时间上实现了约500倍的加速，同时在基准文本生成任务上保持了有竞争力的性能。我们进一步分析了语言领域扩散蒸馏中的失败模式，并提出了一种对抗正则化的两阶段训练方案以防止学生模型退化。

    arXiv:2506.00290v2 Announce Type: replace  Abstract: This paper introduces DLM-One, a score-distillation-based framework for one-step sequence generation with continuous diffusion language models (DLMs). DLM-One eliminates iterative refinement by aligning the scores of a student model's outputs with the score function of a pretrained teacher DLM in the forward-diffused noisy space. We demonstrate that our framework is architecture-agnostic and robust across diverse continuous manifolds, including standard token embedding spaces and logit simplex spaces. Through experiments on multiple representative DLMs, we show that DLM-One achieves up to $\sim$2000$\times$ speedup in sampling steps and $\sim$500$\times$ in wall-clock time, while maintaining competitive performance on benchmark text generation tasks. We further analyze failure modes in language-domain diffusion distillation and propose an adversarially-regularized two-stage training scheme to prevent student degeneration. Our finding
    
[^187]: SocialMaze：一个用于评估和增强大型语言模型在复杂社会环境中社会推理能力的基准

    SocialMaze: A Benchmark for Evaluating and Enhancing Social Reasoning in Large Language Models in Complex Social Environments

    [https://arxiv.org/abs/2505.23713](https://arxiv.org/abs/2505.23713)

    该论文提出了SocialMaze基准，通过深度推理、动态交互和信息不确定性三个设计维度，在社交推理游戏、日常互动和数字社区平台等六项任务中评估并提升大型语言模型在复杂社会环境中的社会推理能力。

    

    大型语言模型（LLM）越来越多地被部署在与社会情境相关的应用中，在这些应用里，成功需要理解上下文、推断他人的心理状态，并对不可靠的信息进行推理。然而，现有的基准测试很少能在复杂且动态演变的环境中联合评估这些需求。我们提出了SocialMaze，这是一个基准测试，它围绕三个描述性设计维度——深度推理、动态交互和信息不确定性——组织了六项任务，涵盖社交推理游戏、日常生活互动和数字社区平台。这些维度刻画的是任务难度的预期来源，而非模型能力的潜在因子分析维度。自动化检查和人工验证保障了数据质量。对十二个专有和开源权重大型语言模型的评估显示，模型在利用动态演变的交互历史方面存在显著差异；更强的思维链推理者在需要深度推理的任务上表现更好。

    arXiv:2505.23713v2 Announce Type: replace  Abstract: Large language models (LLMs) are increasingly deployed in socially grounded applications, where success requires interpreting context, inferring others' mental states, and reasoning about unreliable information. Yet existing benchmarks rarely evaluate these demands jointly in complex, evolving settings. We introduce SocialMaze, a benchmark that organizes six tasks across social deduction games, daily-life interactions, and digital community platforms along three descriptive design axes: deep reasoning, dynamic interaction, and information uncertainty. These axes characterize intended sources of task difficulty rather than latent, factor-analytic dimensions of model capability. Automated checks and human validation support data quality. Evaluations of twelve proprietary and open-weight LLMs show substantial variation in the use of evolving interaction histories; stronger chain-of-thought reasoners perform better on tasks requiring dee
    
[^188]: AI副驾中用户偏好的建模与优化：综合综述与分类体系

    Modeling and Optimizing User Preferences in AI Copilots: A Comprehensive Survey and Taxonomy

    [https://arxiv.org/abs/2505.21907](https://arxiv.org/abs/2505.21907)

    本综述系统梳理并分类了AI副驾系统中用户偏好信号的获取、跨交互阶段建模及反馈优化方法，以实现个性化。

    

    AI副驾代表了新一代AI驱动系统，旨在协助用户——尤其是知识工作者和开发者——完成复杂且富含上下文的任务。随着这些系统日益融入日常工作流程，个性化已成为提升易用性、有效性和用户满意度的关键因素。个性化的核心在于偏好优化：即系统检测、解读并与个体用户偏好对齐的能力。尽管在智能助手和优化算法领域已有大量先前研究，但二者在AI副驾中的交叉领域仍然缺乏深入探索。本综述通过考察用户偏好如何在AI副驾中被实际应用来填补这一空白。我们研究了偏好信号如何被获取、在不同交互阶段如何被建模，以及如何通过反馈循环进行优化。基于全面的文献回顾，我们定义了AI副驾的概念

    arXiv:2505.21907v3 Announce Type: replace  Abstract: AI copilots represent a new generation of AI-powered systems designed to assist users, particularly knowledge workers and developers, in complex, context-rich tasks. As these systems become more embedded in daily workflows, personalization has emerged as a critical factor for improving usability, effectiveness, and user satisfaction. Central to this personalization is preference optimization: the system's ability to detect, interpret, and align with individual user preferences. While prior work in intelligent assistants and optimization algorithms is extensive, their intersection within AI copilots remains underexplored. This survey addresses that gap by examining how user preferences are operationalized in AI copilots. We investigate how preference signals are sourced, modeled across different interaction stages, and refined through feedback loops. Building on a comprehensive literature review, we define the concept of an AI copilot
    
[^189]: 大型推理模型何时能够节省思考？推理中行为分歧的机制分析

    When Can Large Reasoning Models Save Thinking? Mechanistic Analysis of Behavioral Divergence in Reasoning

    [https://arxiv.org/abs/2505.15276](https://arxiv.org/abs/2505.15276)

    该论文从思考终止边界置信度、内部注意力分布分歧和注意力分配三个机制角度，揭示了大型推理模型在“节省思考”提示下仍持续思考的原因（高困惑度及对原始问题的过多注意力），并提出注意力干预方法虽能抑制思考但会降低准确率。

    

    大型推理模型在复杂任务上取得了显著成功，但其“过度思考”的倾向导致了效率低下。尽管“节省思考”提示旨在缓解这一问题，我们发现大型推理模型仍频繁进入“仍在思考”模式，而非预期的“无思考”模式，尤其是在困难问题上。为分析这种行为分歧，我们从三个角度考察大型推理模型：思考终止边界处的置信度、内部注意力分布的分歧，以及跨提示片段的注意力分配。我们发现高困惑度与较晚出现的“仍在思考”行为相关，且“仍在思考”的情形会将更多注意力分配给原始问题。基于这些观察，我们提出了一种注意力干预方法来调节这种行为。虽然该干预抑制了显式思考，但也导致准确率下降，这表明……

    arXiv:2505.15276v2 Announce Type: replace  Abstract: Large reasoning models (LRMs) have achieved remarkable success on complex tasks, yet their tendency to "overthink" leads to inefficiencies. Although "save-thinking" prompts are intended to mitigate this issue, we find that LRMs still frequently enter the "Still-thinking" mode instead of the expected "No-thinking" mode, especially on difficult queries. To analyze this behavioral divergence, we examine LRMs from three perspectives: confidence at the thinking-termination boundary, divergence in internal attention distributions, and attention allocation across prompt segments. We find that high perplexity is associated with later Still-thinking behavior, and that Still-thinking cases allocate more attention to the original question. Based on these observations, we propose an attention intervention method to regulate this behavior. While this intervention suppresses explicit thinking, it also causes a drop in accuracy, suggesting that the
    
[^190]: 多模态语言模型作为文生图模型评估器

    Multimodal Language Models as Text-to-Image Model Evaluators

    [https://arxiv.org/abs/2505.00759](https://arxiv.org/abs/2505.00759)

    提出MT2IE评估框架，让单个多模态大语言模型作为评估代理迭代生成提示词并给图像评分，其与人类判断的相关性高于现有指标，且仅需20个提示词即可高效复现T2I模型官方排名。

    

    arXiv:2505.00759v3 公告类型： replace-cross 摘要：文生图（T2I）生成模型的持续改进导致依赖静态数据集的自动评估基准逐渐被淘汰，促使研究人员寻求评估T2I进展的替代方法。我们提出了多模态文生图评估框架MT2IE（Multimodal Text-to-Image Eval），在该框架中，单个多模态大语言模型（MLLM）充当评估代理，迭代地生成评估提示词并对生成的图像进行评分。我们表明，MT2IE的图文一致性分数与人类判断的相关性高于文献中先前提出的评估指标。MT2IE生成的提示词在探测T2I模型性能方面非常高效：仅用20个生成的评估提示词就能基本复现三个结构各异的基准测试的官方T2I模型排名，所用提示词数量比基准测试自身的提示词集少了28至105倍。当与CLIPScore等现有评估指标进行比较时……

    arXiv:2505.00759v3 Announce Type: replace-cross  Abstract: The steady improvements of text-to-image (T2I) generative models lead to slow deprecation of automatic evaluation benchmarks that rely on static datasets, motivating researchers to seek alternative ways to evaluate T2I progress. We present Multimodal Text-to-Image Eval (MT2IE), an evaluation framework in which a single multimodal large language model (MLLM) acts as an evaluator agent, iteratively generating the evaluation prompts and scoring the resulting images. We show that MT2IE's image-text consistency scores have higher correlation with human judgment than metrics previously introduced in the literature. MT2IE generates prompts that are efficient at probing T2I model performance: closely recovering the official T2I model rankings of three structurally distinct benchmarks from just 20 generated evaluation prompts, 28-105x fewer than the benchmarks' own prompt sets. When compared to existing evaluation metrics such as CLIPSc
    
[^191]: 民主国家的精英政治不文明行为正在上升

    Elite political incivility is rising across democracies

    [https://arxiv.org/abs/2503.22411](https://arxiv.org/abs/2503.22411)

    通过用大语言模型分析26个国家议员的1380万条推文，发现精英政治不文明行为在2017至2022年间几乎翻倍，且这一上升主要源于各党派行为的普遍激进化，而非激进政党崛起的结构性变化。

    

    政治不文明行为是否在各个民主国家中上升——如果是，原因何在？我们使用经过验证的大语言模型，对26个国家议员发布的约1380万条推文进行了分析，发现严重不文明推文的比例在2017年至2022年间几乎翻倍，从约2.7%上升到约5.8%。民粹主义是最强的政党层面预测因子（OR = 1.46，95% CI：1.29–1.65），且其与不文明行为的关联在反对党中更为显著。更健全的自由民主制度与更低的严重不文明程度相关（每标准差OR = 0.84，95% CI：0.72–0.98）。不文明的上升绝大多数源于行为层面的变化——各个政治光谱的政党家族都变得更加激进——而非组成结构的变化：激进政党日益崛起这一因素对此的解释力很小。民族主义和极右政党始终是最不文明的，但随着其2020年峰值部分回落而主流政党变得更加不文明，两者之间的差距正在缩小。

    arXiv:2503.22411v2 Announce Type: replace  Abstract: Is political incivility rising across democracies - and if so, why? Analysing approximately 13.8 million tweets from parliamentarians in 26 countries with a validated large language model, we find that the share of severe incivility nearly doubled between 2017 and 2022, from $\approx$2.7\% to $\approx$5.8\% of tweets. Populism is the strongest party-level predictor (OR~=~1.46, 95\%~CI: 1.29--1.65), and its association with incivility is amplified in opposition. Stronger liberal democracy is associated with lower severe incivility (OR~=~0.84 per SD, 95\%~CI: 0.72--0.98). The rise in incivility is overwhelmingly behavioral - party families across the spectrum becoming more aggressive - rather than compositional: the growing prominence of radical parties explains little of it. Nationalist and radical-right parties remain the most uncivil throughout, but the gap narrows as their 2020 peak partially recedes while mainstream parties become
    
[^192]: 评估评估器：超越英语的摘要评估指标与大语言模型裁判

    Evaluating the Evaluator: Summarization Metrics and LLM-Judges beyond English

    [https://arxiv.org/abs/2503.17039](https://arxiv.org/abs/2503.17039)

    本文构建了首个超越英语的多语言摘要元评估数据集BASSE，基于2,040个摘要的人类判断对自动评估指标和LLM裁判进行基准测试，发现专有LLM裁判与人类判断相关性最高，其次是特定标准的自动指标。

    

    自动文本摘要依赖自动评估方法，通过自动指标和大语言模型作为裁判（LLM-as-a-Judge）来快速判定摘要模型的质量。然而，这些技术需要元评估来确保其能够准确捕捉人类判断。在本文中，我们探索了英语之外的元评估，构建了一个新的多语言摘要元评估数据集（BASSE），其中包含针对2,040个生成式摘要的人类判断，这些摘要由人工或五个大语言模型（LLM）在四种不同提示下生成。对于每个摘要，标注者使用5点李克特量表从五个标准进行评估：连贯性、一致性、流畅性、相关性以及5W1H。随后，我们对自动摘要评估指标和大语言模型裁判模型进行了基准测试。结果表明，目前专有的裁判大语言模型与人类判断的相关性最高，其次是针对特定标准的自动评估指标，而开源……

    arXiv:2503.17039v3 Announce Type: replace-cross  Abstract: Automatic text summarization relies on automatic evaluation to quickly determine the quality of summarization models via automatic metrics and LLM-as-a-Judge models. However, these techniques require meta-evaluation to ensure that they capture human judgments correctly. In this paper, we explore this meta-evaluation beyond English by generating a new multilingual summary meta-evaluation dataset (BASSE), which comprises human judgments on 2,040 abstractive summaries, generated either manually or by five Large Language Models (LLMs) with four different prompts. For each summary, annotators evaluate five criteria on a 5-point Likert scale: coherence, consistency, fluency, relevance, and 5W1H. We then benchmark automatic summarization metrics and LLM-as-a-Judge models. Our results show that currently proprietary judge LLMs have the highest correlation with human judgments, followed by criteria-specific automatic metrics, while open
    
[^193]: 超越RAG：实时对话中的问题识别与答案生成

    Beyond-RAG: Question Identification and Answer Generation in Real-Time Conversations

    [https://arxiv.org/abs/2410.10136](https://arxiv.org/abs/2410.10136)

    该论文提出了一个超越传统RAG的实时决策支持系统，通过先识别客户问题并判断其是否匹配FAQ数据库来直接检索答案或经由RAG生成答案，将响应时间缩短至2秒以内，显著减轻了客服人员的查询负担。

    

    在客户联络中心，人工客服由于需要手动解读客户查询并检索相关的知识库（KB）文章，往往面临较长的平均处理时间（AHT）。虽然基于大语言模型（LLM）的检索增强生成（RAG）系统已在业界被广泛采用以协助此类任务，但RAG在实时对话场景中面临诸多挑战，例如查询表述不准确以及对常见问题（FAQ）的冗余检索。为了解决这些局限性，我们提出了一个能够超越RAG的决策支持系统，该系统首先实时识别客户提出的问题。如果该查询匹配到FAQ，系统将直接从FAQ数据库中检索答案；否则，系统通过RAG生成答案。我们的方法减少了对人工查询的依赖，能够在2秒内向客服人员提供响应。该系统已部署于Minerva CQ的AI赋能人工客服辅助解决方案中。

    arXiv:2410.10136v2 Announce Type: replace-cross  Abstract: In customer contact centers, human agents often struggle with long average handling times (AHT) due to the need to manually interpret queries and retrieve relevant knowledge base (KB) articles. While retrieval augmented generation (RAG) systems using large language models (LLMs) have been widely adopted in industry to assist with such tasks, RAG faces challenges in real-time conversations, such as inaccurate query formulation and redundant retrieval of frequently asked questions (FAQs). To address these limitations, we propose a decision support system that can look beyond RAG by first identifying customer questions in real time. If the query matches an FAQ, the system retrieves the answer directly from the FAQ database; otherwise, it generates answers via RAG. Our approach reduces reliance on manual queries, providing responses to agents within 2 seconds. Deployed in AI-powered human-agent assist solution at Minerva CQ, this s
    
[^194]: 提示未知：理解大语言模型中的响应不确定性

    Prompting the Unknown: Understanding Response Uncertainty in Large Language Models

    [https://arxiv.org/abs/2407.14845](https://arxiv.org/abs/2407.14845)

    该论文提出了一个提示-响应概念模型，识别出大语言模型响应不确定性的四个来源（提示规范不足、模型质量、任务变异性和语义冗余），并证明了提高提示信息性或模型质量可以降低响应不确定性。

    

    大语言模型（LLM）被广泛应用于跨领域的决策制定中。确保生成安全可靠的响应对基于LLM的应用的有效部署至关重要，尤其是在医疗保健和金融等高风险领域。这些应用通常使用精心设计的提示来引导响应生成；然而，提示与LLM生成响应的可靠性之间的关系尚未被充分理解。为填补这一空白，我们提出了一个新颖的提示-响应概念模型，通过识别响应不确定性的四个来源——提示规范不足、模型质量、任务变异性和语义冗余——来解释提示中提供的任务相关信息量（信息性）与LLM生成响应不确定性之间的关系。我们证明了随着提示信息性或模型质量的提升，响应不确定性会降低。

    arXiv:2407.14845v4 Announce Type: replace-cross  Abstract: Large language models (LLMs) are widely used in decision-making across diverse domains. Ensuring the generation of safe and reliable responses is critical for the effective deployment of LLM-based applications, particularly in high-stakes domains such as healthcare and finance. Most of these applications typically use carefully crafted prompts to guide response generation; however, the relationship between prompts and the reliability of LLM-generated responses is not yet fully understood. To address this gap, we propose a novel prompt-response concept model that explains the relationship between the amount of task-relevant information (informativeness) provided in the prompt and the LLM-generated response uncertainty by identifying four sources of response uncertainty: prompt underspecification, model quality, task variability, and semantic redundancy. We prove that response uncertainty decreases as prompt informativeness or mo
    
[^195]: 一项聚焦效率的基于Transformer语言模型综述

    A Survey of Transformer-based Language Models with Focus on Efficiency

    [https://arxiv.org/abs/2406.16893](https://arxiv.org/abs/2406.16893)

    本文从效率视角系统综述了312篇关于基于Transformer的大语言模型的文献，全面梳理了数据整理、模型设计、模型缩减、动态推理以及预训练、微调、提示工程和RAG等适配策略中的效率提升方法。

    

    基于Transformer的大型语言模型（LLM）的出现极大地增强了自然语言处理（NLP）的能力，同时也加剧了对计算资源的需求。因此，从计算需求、能源消耗、碳足迹和经济成本等因素出发提升效率已成为一个重要的研究领域。这促使我们从效率的视角对NLP领域基于Transformer的大型语言模型进行综述。在这篇涵盖312篇文章的综述中，我们系统地讨论了针对数据整理、模型设计、模型缩减和动态推理等多个方面的效率提升研究工作。此外，还结合了预训练、微调、提示工程和检索增强生成（RAG）等模型适配策略中的效率考量。并且，在统计分析之后还进行了深入的评估与讨论。

    arXiv:2406.16893v3 Announce Type: replace-cross  Abstract: The emergence of Transformer-based Large Language Models (LLMs) has substantially augmented the capabilities of Natural Language Processing (NLP), thereby intensifying the demand for computational resources. Therefore, enhancing efficiency based on factors like computational requirements, energy consumption, carbon footprint and financial cost has become a vital area of research. This motivates us to conduct a survey on Transformer-based LLMs in NLP from the perspective of efficiency. In this survey of 312 articles, the efficiency-improvement endeavors have been systematically discussed targeting various aspects such as data curation, model design, model downsizing, and dynamic inferencing. This has been augmented with efficiency considerations in model adaptation strategies like pre-training, fine-tuning, prompt-engineering and Retrieval-Augmented Generation (RAG). Furthermore, a statistical analysis followed by an in-depth ev
    
[^196]: GPTBIAS：一个用于评估大语言模型偏见的综合框架

    GPTBIAS: A Comprehensive Framework for Evaluating Bias in Large Language Models

    [https://arxiv.org/abs/2312.06315](https://arxiv.org/abs/2312.06315)

    本文提出了GPTBIAS框架，利用GPT-4等高性能大语言模型来评估其他模型的社会偏见，并设计了专门用于偏见评估的“偏见攻击指令”提示词，从而提升了偏见评估的可信度和可解释性。

    

    警告：本文包含可能具有冒犯性或令人不适的内容。大语言模型（LLM）在各种应用中的使用大幅增加，无论是以原始形式还是通过微调适配的形式。因此，LLM 日益流行，并被庞大的用户群体广泛采用。然而，LLM 的一个隐忧是可能生成带有社会偏见的内容。现有的评估方法存在诸多局限，其结果的可解释性程度也较为有限。在本工作中，我们提出了一个名为 GPTBIAS 的偏见评估框架，该框架利用 LLM（例如 GPT-4）的高性能来评估模型中的偏见。我们还引入了专门为评估模型偏见而设计的提示词，称为“偏见攻击指令”。为了增强偏见评估的可信度和可解释性，我们的框架不仅提供……

    arXiv:2312.06315v2 Announce Type: replace  Abstract: Warning: This paper contains content that may be offensive or upsetting. There has been a significant increase in the usage of large language models (LLMs) in various applications, both in their original form and through fine-tuned adaptations. As a result, LLMs have gained popularity and are being widely adopted by a large user community. However, one of the concerns with LLMs is the potential generation of socially biased content. The existing evaluation methods have many constraints, and their results exhibit a limited degree of interpretability. In this work, we propose a bias evaluation framework named GPTBIAS that leverages the high performance of LLMs (e.g., GPT-4 \cite{openai2023gpt4}) to assess bias in models. We also introduce prompts called Bias Attack Instructions, which are specifically designed for evaluating model bias. To enhance the credibility and interpretability of bias evaluation, our framework not only provides 
    

