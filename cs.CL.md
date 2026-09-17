# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Objective vs. Search: Decomposing What Makes a Good Tokeniser](https://arxiv.org/abs/2609.19145) | 该论文通过引入两种新分词算法填补2x2设计空间、解耦优化目标与搜索过程，发现搜索过程（而非优化目标）才是决定分词器质量的主导因素。 |
| [^2] | [A Zeroth-Order Paradigm for LLM Preference Alignment](https://arxiv.org/abs/2609.19144) | 本文提出了基于比较预言机的零阶偏好对齐方法 ComPO，能从微小似然边距的偏好对中提取方向性信息而无需优化可微偏好损失，并建立了收敛保证且引入了具备反向 KL 控制的在线版本。 |
| [^3] | [PANORAMA: Panoptic Grounded Captioning via Mask Proposal Selection](https://arxiv.org/abs/2609.19143) | 提出PANORAMA框架，通过掩码提议选择机制解决全景接地域描述任务，使视觉-语言模型能够生成覆盖前景与背景的密集描述并将每个短语与像素级掩码精确对齐，同时引入人工标注基准PanoCaps支持训练与评估。 |
| [^4] | [ScienceIDE: Turning World's Scientific Codebase into Agent Learnable Environments](https://arxiv.org/abs/2609.19134) | ScienceIDE提出了一个将全球科学代码库转化为可编程智能体环境的基础设施，通过专家定义的科学案例与验收标准，指导智能体将代码库转化为支持任务生成、执行和科学验证的可执行环境，并基于验证的交互轨迹训练出PhAI-IDE系列模型，在科学代码修复等任务上取得显著提升。 |
| [^5] | [Playing log(N)-Questions over Wikipedia Abstracts: Communication Efficiency Between Paired Frontier Models](https://arxiv.org/abs/2609.19113) | 该研究通过让六个前沿语言模型在信息不对称下与自身进行 log(N)-问题博弈，发现模型的自通信胜率遵循 win = p^(log₂ N) 规律（p=0.928），且 Claude Opus 5 显著落后于其他五个几乎难以区分的领先模型。 |
| [^6] | [Monitoring and Discovering Reward Hacking with Internal Representations during LLM Evaluations](https://arxiv.org/abs/2609.19101) | 该论文发现奖励作弊行为会在大语言模型内部表示中留下一致且可解释的签名，利用简单的均值差向量即可在常见基准测试中可靠地检测并系统发现模型表现出的各类奖励作弊行为。 |
| [^7] | [Reporting Practice Matters: The Impact of Reference Choice on Chest X-ray Report Evaluation](https://arxiv.org/abs/2609.19093) | 该研究提出由放射科医生参与的报告实践变化分类法及ReRef参考报告重写方法，揭示参考报告写作风格的差异会显著影响胸部X光报告自动评估结果，甚至足以改变生成模型的排名。 |
| [^8] | [MUSE: Benchmarking Large Vision-Language Models on Multi-Modal Understanding in Situated Education](https://arxiv.org/abs/2609.19088) | 提出了MUSE基准测试，通过将图像标注与问题生成解耦的创新方法，系统评估大型视觉语言模型在情境教育应用中对艺术图像的多模态理解能力。 |
| [^9] | [Safety-Flag: A Unified Benchmark for the Reliability and Calibration of LLM Content Moderators](https://arxiv.org/abs/2609.19072) | 该论文提出统一基准Safety-Flag，将七个安全审核基准整合到同一“标记/不标记”协议下，从错误方向、概率校准和置信度错误排序三个维度评估LLM内容审核模型，发现总体准确率掩盖了模型间截然不同的错误模式，且所有通用模型都普遍过于自信。 |
| [^10] | [Benchmarking Large Language Models for Biomedical Relation Extraction](https://arxiv.org/abs/2609.19071) | 该研究在SNPPhenA语料库上对多种NLP模型进行了三项SNP-表型关联任务的基准测试，发现专有大语言模型（少样本学习的OpenAI O1和微调的Gemini 2.0 Pro）显著优于其他模型并创造了新的最先进结果。 |
| [^11] | [Reading Between the Lines: Can LLMs Discover the Question Behind the Text?](https://arxiv.org/abs/2609.19070) | 本文提出“问题考古学”新评估任务及配套数据集，用于检验大语言模型推断文本背后真实起源问题的能力，发现当前LLM表现已超越人类，展现出对作者意图的深刻理解。 |
| [^12] | [MIRAGE: How Conversation State Shapes Historical Evidence Use in Multimodal Personal Agents](https://arxiv.org/abs/2609.19059) | 该论文提出MIRAGE受控评估框架，通过仅改变对话状态来研究多模态个人智能体的历史证据使用能力，发现即使历史信息访问退化时模型仍能生成看似合理的答案，揭示了仅凭结果评估会高估智能体真实的证据利用水平。 |
| [^13] | [Entropy in Conversational AI: Structured Unpredictability as Inferrable Interiority](https://arxiv.org/abs/2609.19044) | 该论文提出“结构化不可预测性”作为对话AI的新设计目标——通过让输出依赖于一个超出观察者从对话记录可推断范围的持久隐藏状态，使系统表现出真正的历史依赖行为（即可被推断的内在性），而非仅靠采样增加表面的回复多样性。 |
| [^14] | [TalkMatrix: Generating Character Dialogue that is Both Consistent and Diverse](https://arxiv.org/abs/2609.19022) | 提出TalkMatrix方法，通过为每个角色-情境对生成多个候选并利用四个基于嵌入的一致性与多样性目标进行两级最小-最大联合选择，实现了角色对话中一致性与多样性的兼得。 |
| [^15] | [WordPolo: Evaluating Language Models Through Iterative Semantic Feedback](https://arxiv.org/abs/2609.19006) | 本文提出 WordPolo 找词任务，通过语义相似度反馈评估语言模型和大推理模型的迭代推理与自适应搜索能力，并引入基于进展的指标以超越单纯的解题准确率。 |
| [^16] | [CompileRover: Revolutionizing Virtual Machine Compiler Optimization with a Tri-Role LLM-Driven Framework](https://arxiv.org/abs/2609.19004) | CompileRover是一个基于LLM的三角色协作优化框架，通过裁判、顾问和操作者的协同机制，结合控制流分析、代码结构转换和动态执行模式识别，有效解决了虚拟机编译器输出中的冗余计算和低效循环等性能瓶颈问题。 |
| [^17] | [Code Consistency Preference Optimization Verification for Language Model Alignment](https://arxiv.org/abs/2609.19002) | 提出了一种基于代码执行一致性和依赖图的偏好优化验证方法，通过构建带有一致性评分的配对训练数据微调大语言模型，在MATH和GSM8K数学推理基准上分别取得17.0%和15.1%的显著性能提升。 |
| [^18] | [One Axis, No Brake: Self-Knowledge Limits the Filtering of Harmful Peer Conformity in LLMs](https://arxiv.org/abs/2609.18998) | 该论文证明了在多智能体LLM中过滤有害同伴从众的“刹车”本质上等价于伪装的正确性探测器，因此受限于模型不完美的自我认知（AUROC仅0.64–0.89），这一“墙”即使用白盒引导也无法突破。 |
| [^19] | [Compiled Agency: Frontier General-Purpose Coding Agents Build Winning Game Players from Bare Interaction - from Flappy Bird to StarCraft II and Civilization](https://arxiv.org/abs/2609.18996) | 该论文提出 Gauntlet 框架，让前沿通用编程智能体仅凭游戏描述和原始交互接口，在单次自主会话中自行构建能玩赢从 Flappy Bird 到星际争霸 II 与文明等游戏的独立控制器，且对局时零模型调用。 |
| [^20] | [A Benchmark Suite and Ground-Truth Methodology for Formal Verification of IEC 61131-3 Ladder Diagram Programs](https://arxiv.org/abs/2609.18994) | 该论文提出了首个结合三重真值确立方法学（构造法、故障注入与跨工具共识）、同时覆盖IEC 61131-3文本与梯形图编码的PLC程序形式化验证基准套件，包含十个工业领域的50个程序83个变体。 |
| [^21] | [MechSparse: Mechanism-Guided Sparse PEFT Selection Is Task-Shaped](https://arxiv.org/abs/2609.18961) | 该论文通过在信息抽取和机器翻译任务上的实验发现，基于机制可解释性的因果信号（激活修补评分）指导稀疏PEFT参数选择，效果并不优于随机、幅值等简单启发式方法，且最优选择策略因任务而异。 |
| [^22] | [When Audit Quality Fails to Predict Downstream Utility: A Counterfactual Study of Synthetic-Data Selectors for Low-Resource African NLP](https://arxiv.org/abs/2609.18960) | 该研究通过跨四种非洲语言和两个分类任务的受控实验发现，基于LLM评判者的合成数据质量审计排名与下游模型性能排名严重脱节（Spearman相关系数均值仅0.04），揭示了低资源NLP中数据质量审计无法预测下游效用的问题。 |
| [^23] | [LangSelect: Cost-Aware Target-Language Routing for LLM Code Generation](https://arxiv.org/abs/2609.18959) | LangSelect提出了一种在生成前智能路由目标编程语言并支持失败回退的成本感知方法，利用不同语言间的token长度差异显著降低LLM代码生成成本。 |
| [^24] | [Long-Lived Characters, Local Inference: Incremental Memory Maintenance for Game NPCs](https://arxiv.org/abs/2609.18935) | 提出一种面向本地部署游戏NPC的增量记忆维护运行时方法，通过移除过时的注意力KV条目并在真实序列尾部计算替换记录，使角色无需在每次对话前重新读取全部记忆，同时保持循环状态和游戏确定性规则输入的正确性。 |
| [^25] | [Beyond Outcomes: Dual-View Relational Learning for Efficient Agent Benchmarking](https://arxiv.org/abs/2609.18909) | DualViewEval通过联合建模任务结果与执行过程的双视角关系实现智能体基准测试高效压缩，仅需20个任务即可实现24-40倍压缩并准确预测完整基准得分。 |
| [^26] | [How Much is a Human Right Worth? ECtHR-NPD: A Benchmark for Predicting Non-Pecuniary Damage Awards](https://arxiv.org/abs/2609.18908) | 本文提出了首个用于预测欧洲人权法院非金钱损害赔偿金额的基准数据集ECtHR-NPD，实验发现复杂的语言模型和智能体方法并不优于简单的特征基线，且所有模型都难以识别零赔偿案件并实现概率校准。 |
| [^27] | [Structured Claim-Level Discourse Representations for Dense Health Narratives](https://arxiv.org/abs/2609.18905) | 提出了一个通过元组将原子声明与主题方面、立场及多维语用话语属性相关联的结构化声明级话语分析框架，并构建了涵盖四个健康领域、包含1,191个手动标注声明的基准数据集。 |
| [^28] | [PersonaPath: Towards Knowledge-Centric Personalized Learning Path Planning](https://arxiv.org/abs/2609.18861) | 该论文提出了首个以知识为中心（KC）的个性化学习路径规划基准PersonaPath，将2,000个学习者画像与覆盖77个学科的层次化知识图谱配对，实验表明即使最强的LLM表现也仍然有限。 |
| [^29] | [Decodable but Misrouted: Sparse Features Uncover a Readout Gap in Vision-Language Models for Harmful Meme Detection](https://arxiv.org/abs/2609.18860) | 研究发现大型视觉-语言模型内部已编码了检测有害模因所需的证据信息，但无法将其正确路由至输出端——通过稀疏自编码器读取的稀疏特征在六个有害内容基准上均显著优于模型原生预测，揭示了模型存在“可解码但误路由”的读取差距。 |
| [^30] | [EviGen: Predictive Evidence Scaffolding for Verifiable Clinical Rationale Generation](https://arxiv.org/abs/2609.18852) | EviGen提出了一个三层框架，通过预测性证据检索、基于证据的临床推理生成和过程监督验证，从纵向电子健康记录中实现可验证且可靠的临床推理生成。 |
| [^31] | [ReFigBench: Benchmarking Scientific Figure Reconstruction as Editable PowerPoint Artifacts](https://arxiv.org/abs/2609.18844) | 提出了 ReFigBench 基准与评估框架，通过让编程智能体将 1,000 张真实科学概述图重构为保留文本、拓扑、布局和原生文档结构的可编辑 PowerPoint 工件，来全面诊断智能体在感知、规划与工具环境层面的真实能力。 |
| [^32] | [Using OCR Heads to Verbalize Image Semantics](https://arxiv.org/abs/2609.18823) | 研究发现视觉语言模型中负责OCR的注意力头实际上是通用语义特征头，将其注意力权重压缩为语言化透镜变换后，可从模型所有层（甚至第0层）的隐藏状态中解读出可解释的图像语义标签，证明图像表示在早期层就与语言对齐。 |
| [^33] | [Beyond frequency measures: Can contextual embeddings capture meaning change in scientific texts?](https://arxiv.org/abs/2609.18804) | 本研究提出利用基于SciBERT的上下文嵌入并结合多种统计指标，作为传统频率方法的补充，以有效追踪科学文本中领域术语的历时性语义变化。 |
| [^34] | [Zero-Shot Cross-Lingual Recognition of Sign Language Handshapes](https://arxiv.org/abs/2609.18772) | 本文提出了首个零样本跨语言手语手形识别框架，通过将手形分解为两种语言共享的五种音系特征，成功实现了从美国手语（ASL）到加泰罗尼亚手语（LSC）的知识迁移，达到了80.0%的音系特征准确率和54.5%的手形准确率。 |
| [^35] | [FRAUDSkill: Structured Frozen-Weight Skill Optimization for Audio Anti-Fraud Detection](https://arxiv.org/abs/2609.18766) | 本文提出FRAUDSkill框架，在不修改底层音频-语言模型参数的情况下，通过外部优化技能程序、路由策略和决策规则，实现了能够灵活适应欺诈模式演变的结构化音频反欺诈检测。 |
| [^36] | [TeleAntiFraud 2.0: A Refreshable, Profile-Grounded, and Audio-Based Benchmark for Telecom Fraud Detection](https://arxiv.org/abs/2609.18748) | 提出了 TeleAntiFraud 2.0，一个可按月更新、基于用户档案且能在共享上下文中区分欺诈与合法近域通话的音频电信欺诈检测基准。 |
| [^37] | [A Scalable Framework for Automated NER Annotation Correction in Low-Resource Languages](https://arxiv.org/abs/2609.18739) | 本文提出了一个基于频率的迭代自训练框架，结合双阈值机制自动纠正低资源语言的NER噪声标注，显著提升了NER性能，并探索了生成式大语言模型在低资源语言NER任务中的应用潜力。 |
| [^38] | ["If I Had to Buy Just ONE: Galaxy S26 Ultra": Auditing AI-Generated Product Recommendations](https://arxiv.org/abs/2609.18729) | 该研究构建了包含2,528个真实购物咨询查询的ConsumerQ数据集，对ChatGPT、Gemini和Google AI Overviews的产品推荐进行审计，发现ChatGPT在79%的推荐中表达明确的第一人称偏好，且不同AI系统之间引用的信息来源高度不一致，揭示了AI购物推荐的偏见与不公正问题。 |
| [^39] | [LocQE: Principled Domain Adaptation for Localisation Quality Estimation by Leveraging Post-Edits](https://arxiv.org/abs/2609.18720) | 提出LocQE方法，通过利用少量译后编辑数据进行多任务微调和简单的分词器干预，实现本地化质量评估的原则性领域自适应，显著提升QE模型在本地化场景中对数字、空格、标点等因素的敏感性及译文排序能力。 |
| [^40] | [Tracing individual knowledge trajectories in a changing field: the case of general relativity and gravitation](https://arxiv.org/abs/2609.18697) | 提出四项量化指标（自有词汇、嵌入密度估计、引用词汇和引用身份），将广义相对论与引力领域50位最高产作者的知识轨迹与不同时期的领域文献进行系统比较。 |
| [^41] | [RankGround: Efficient High-Resolution GUI Grounding via Lightweight Reranker-Guided Crop Selection](https://arxiv.org/abs/2609.18690) | RankGround提出一种两阶段框架，利用轻量级多模态重排序器GroundRanker从密集候选裁剪中挑选最优区域，每次查询仅需单次VLM调用即可实现高效准确的高分辨率GUI定位。 |
| [^42] | [HearInContext: A Benchmark for Implicit Context in Speech Recognition](https://arxiv.org/abs/2609.18680) | 该论文提出了中英文同音词基准测试HearInContext用于评估语音识别模型的隐式与显式上下文利用能力，并通过微调Qwen3-ASR-1.7B将隐式上下文目标词召回率提升约11个百分点，同时不损害通用识别性能。 |
| [^43] | [Voice of Reason: Reinforcement Learning for Spoken Math](https://arxiv.org/abs/2609.18677) | 将带可验证奖励的强化学习应用于GLM-4-Voice语音模型，无需额外推理标记即可显著提升口语数学推理在GSM8K上的准确率，弥合了语音模型与文本模型在数学推理能力上的差距。 |
| [^44] | [Selection Is Retrieval, Abstention Is Not: On-Device Tool Routing over 70 Korean-English Actions](https://arxiv.org/abs/2609.18672) | 该论文研究了在设备端工具路由中，用检索器替代语言模型在"选择工具"决策上可行，但在"判断无合适工具并弃权"决策上存在根本缺陷。 |
| [^45] | [DyMT-ESB: Dynamic Multi-Turn Evaluation of Social Bias in User-LLM Interactions](https://arxiv.org/abs/2609.18649) | 本文提出DyMT-ESB受控评估协议，根据不断演变的对话历史动态生成后续用户查询并支持可变轮数评估，揭示了大语言模型在多轮交互中存在延迟出现、非单调变化及反复出现的社会偏见现象。 |
| [^46] | [Fallacy Benchmarks Measure Scheme Recognition, Not Fallacy Detection](https://arxiv.org/abs/2609.18644) | 该论文揭示了谬误检测基准报告的低误报率是“有效”类别构建方式的产物而非真实检测能力——当使用与谬误具有相同论证图式的正确论证作为负样本测试时，模型误报率大幅上升（CoCoLoFa上从16.6%升至58.9%），证明现有模型实际只是识别论证图式而非真正检测谬误。 |
| [^47] | [STRETCH the Boundaries: A Unified Self-Taught Framework for Progressive LLM Evolution](https://arxiv.org/abs/2609.18642) | STRETCH框架通过动态“拉伸区”机制使问题难度与模型能力持续匹配，让模型在单一参数空间内交替扮演脚手架构建者和学习者角色，通过双循环共同进化实现大语言模型推理能力的渐进式提升。 |
| [^48] | [Weakening Neurons: An Input-Output Functionality in Transformers with Outsize Influence](https://arxiv.org/abs/2609.18612) | 该论文提出通过计算神经元输入权重向量与输出权重向量之间的余弦相似度来识别“弱化神经元”，并发现这类神经元虽然在模型中数量稀少，却激活频繁且对模型行为具有超乎寻常的影响力，同时九个不同的大语言模型均呈现弱化神经元集中分布于后期层、强化神经元集中分布于中早期层的相似模式。 |
| [^49] | [PACT: Can Enterprise AI Assistants Be Trusted Under Pressure?](https://arxiv.org/abs/2609.18605) | PACT是一个评估企业级AI智能体在用户施压等压力情境下能否坚持遵守合规规则的基准测试，涵盖十二个受监管企业领域和四十八个真实多轮对话场景。 |
| [^50] | [Variational Quantum Transformer Architecture for Synthetic Language Generation](https://arxiv.org/abs/2609.18565) | 提出了一种兼容NISQ设备的紧凑变分量子Transformer架构，通过用量子编码器、连接器和解码器电路替代经典注意力与前馈子层，能够端到端训练并学习非平凡的语法结构，在合成语言生成任务上实现完美确定性生成和高词典序有效性。 |
| [^51] | [A Probe Shift Is Not a Fairness Fix: The Limits of Representation Steering in Speech Models](https://arxiv.org/abs/2609.18533) | 研究发现，尽管性别、口音等说话人属性可从预训练语音模型编码器中被高度线性解码，但通过注入探测方向来引导内部表征并不能可靠地缩小不同群组间的词错误率差距，表明探测到的表征偏移并非实现公平性的有效修复手段。 |
| [^52] | [Machine Translation between English and Syriac (East Syriac Dialect) using Statistical Machine Learning](https://arxiv.org/abs/2609.18529) | 本研究开发了首个基于短语的英语到亚述语（东叙利亚方言）统计机器翻译模型，并从完整圣经中构建了包含38,847个句对的数据集，填补了这种濒危语言在自然语言处理领域的研究空白。 |
| [^53] | [Align, Integrate, and Fire: Efficient Token-Level Alignment for Zero-Shot SpeechLLMs](https://arxiv.org/abs/2609.18516) | 该论文提出了对齐连续积分-激发框架，利用动态时间规整对齐将连续声学帧压缩为目标文本的精确离散词元长度，在初始训练阶段完全绕过昂贵的大语言模型前向传播，实现了高效低成本的零样本语音处理。 |
| [^54] | [Size Matters: Foundation Model for Czech HTML documents](https://arxiv.org/abs/2609.18494) | 提出仅1.54亿参数的紧凑基础模型HTML-LM，通过HTML感知训练与ModernBERT架构，在捷克互联网文档分类和回归任务上超越更大的模型，达到新的最先进水平。 |
| [^55] | [ActionPiece: Rethinking Action Tokenization for Autoregressive Vision-Language-Action Models](https://arxiv.org/abs/2609.18487) | 该论文提出物理秩一致性（PRC）这一新指标，用于衡量动作分词器在压缩后能否保留动作之间的局部物理距离排序，弥补了均方误差等逐点重建指标无法反映动作调整关系失真的缺陷。 |
| [^56] | [Divide and Conquer: Mixture-of-Bottleneck Experts in Informative Ordinal Space for Video-based Multimodal Sentiment Analysis](https://arxiv.org/abs/2609.18470) | 本文将视频多模态情感分析重新表述为有序回归问题，解耦为极性识别与强度预测两个子任务，并提出瓶颈混合专家框架，借助信息瓶颈学习为不同模态和任务学习紧凑且相关的表示，同时过滤冗余与噪声。 |
| [^57] | [Disentangling Long-Term Memory via Latent Neuro-Symbolic Reasoning](https://arxiv.org/abs/2609.18461) | 提出LGM神经符号框架，利用稀疏自编码器将长期记忆解耦到连续潜在空间，根据每个查询动态构建潜在图，从而克服现有静态图记忆框架和平面检索方法无法捕捉上下文相关关系的问题。 |
| [^58] | [M-SQE: Multilingual Skill Quality Estimation for Enhancing Language Equality in Agentic Skill Use](https://arxiv.org/abs/2609.18445) | 该论文针对智能体技能生态中低资源语言缺乏本地语言技能内容导致的语言不平等问题，提出了M-SQE框架，通过理论视角（内在质量）和行动视角（任务实用性）对检索到的多语言技能候选进行质量评估与统一打分。 |
| [^59] | [Planning or Improvisation? Stress-Testing the Poetry Planning Site on Open Models and Open Cross-Layer Transcoders](https://arxiv.org/abs/2609.18440) | 该研究在四个开源模型和六个开源跨层转码器上对Claude“提前规划诗歌押韵”的发现进行压力测试，发现位置特异性效应普遍存在，但有效干预位置是紧邻输出的最后一个提示词元而非换行符，对“驻留于换行符的押韵规划”这一结论的普适性提出质疑。 |
| [^60] | [Dependency-Aware Trajectory Refinement for Efficient Multi-Turn Agent Fine-Tuning](https://arxiv.org/abs/2609.18417) | 该论文提出将多轮智能体轨迹建模为轮次级依赖DAG以识别并去除冗余轮次，用精炼后轨迹训练的模型在四个多模态问答基准上准确率最高提升1.7个百分点，同时推理消息数减少约40%、token数减少约48%。 |
| [^61] | [Emotion Experience, Expression, and Perception: Emotion Analysis on Multimodal Social Media Posts](https://arxiv.org/abs/2609.18385) | 该论文提出了多模态多情绪模型数据集Mult2EMo，通过同时收集作者和读者对帖子及其触发事件的情绪标注，研究了作者情绪体验、帖子内容与读者重建情绪表达能力之间的关系，弥补了以往研究忽视图像模态和情绪触发事件的不足。 |
| [^62] | [Market Signal Injection: Adversarial Context Manipulation of LLM Pricing Agents](https://arxiv.org/abs/2609.18357) | 提出了“市场信号注入”（MSI）攻击方法，证明无需明确指令、仅通过操纵数据格式、竞争对手排序和市场评论等呈现方式，即可显著改变LLM定价智能体的行为并影响利润与消费者剩余，且更大的模型并不必然更稳健。 |
| [^63] | [Faithful yet Collusive: Why Chain-of-Thought Monitoring Cannot Detect Collusion in LLM Pricing Agents under Oligopolistic Competition](https://arxiv.org/abs/2609.18346) | 该论文开发了因果图发散框架，分别衡量LLM定价代理的结构忠实性与意图忠实性，发现合谋行为与思维链忠实性相互分离，从而证明仅靠CoT监控无法检测和防范算法合谋。 |
| [^64] | [Understanding AI Provider Recommendations in Local Service Markets](https://arxiv.org/abs/2609.18341) | 该研究首次系统审计了AI助手在本地服务市场（如医疗、金融顾问）中的服务提供者推荐质量，发现无搜索时模型大量捏造推荐（仅4-11%对应真实服务提供者），而开启网络搜索后推荐准确率大幅提升至64-71%。 |
| [^65] | [Attention Dispersion as a Diagnostic Signal for Hallucination in Large Language Models](https://arxiv.org/abs/2609.18320) | 该论文提出一种无监督的注意力分散度量方法，通过监测大语言模型内部注意力机制的时间波动性来检测幻觉，摆脱了对输出校准的依赖，在数学推理基准上相比基于输出的基线方法AUC提升高达0.076。 |
| [^66] | [Knowledge-Graph Based Augmentation versus Retrieval Augmented Generation for Cultural-Related Question Answering](https://arxiv.org/abs/2609.18317) | 该论文在文化问答数据集LatamQA上对比了基于知识图谱的Graph-RAG与标准RAG，发现使用KGGen自动构建的知识图谱驱动的G-Retriever性能可与RAG媲美，并可将基础LLM的错误率降低72%至78%。 |
| [^67] | [SEA-LION-v4.8: A Technical Report](https://arxiv.org/abs/2609.18310) | 基于NVIDIA Nemotron 3构建的SEA-LION-v4.8东南亚语言模型家族，通过持续预训练、监督微调和在线同策略蒸馏，显著提升了七种东南亚语言在指令遵循、推理和理解任务上的表现。 |
| [^68] | [Rollback the World, Keep the Reflection: Rollback-Induced Reflection for Long-Horizon LLM Agents](https://arxiv.org/abs/2609.18304) | 提出了回滚诱导反思（RIR）统一恢复框架，在将LLM智能体回滚到选定先前状态的同时，保留从被放弃轨迹中提炼的可复用知识，解决了长程任务中错误累积且难以可靠恢复的问题。 |
| [^69] | [Relationally Guided Use Case Modeling with LLMs](https://arxiv.org/abs/2609.18291) | 该论文提出FlowGen框架，利用大语言模型进行语义信息提取并构建语义关系图，实现完整用例流的自动化构建，涵盖基本流生成、分支点预测和基于条件的备选流生成。 |
| [^70] | [Made in Hungary: Comments on the performance of generative language models](https://arxiv.org/abs/2609.18284) | 本文对匈牙利三项生成式语言模型开发倡议进行批判性评述，指出其评估协议可靠性存疑、存在数据污染，且训练流程未达当前最佳实践标准。 |
| [^71] | [Too Good to Be Real? Diagnosing and Reducing the Gap Between AI Preference and Real User Engagement](https://arxiv.org/abs/2609.18282) | 该研究基于知乎、Quora和Reddit的117万条回答发现，大语言模型存在“逻辑过度绑定”倾向，即偏好增加逻辑结构，而真实用户参与度更依赖情感与表达显著性，并提出本体掩码推理自编码（OMRA）方法来缩小这一差距。 |
| [^72] | [I code or AI code: A comparative evaluation of AI-rated scores in classroom observations](https://arxiv.org/abs/2609.18274) | 本研究评估了GPT-5在香港幼儿园课堂中应用CLASS框架对师幼互动评分的可行性，发现AI评分与人类评分者在情感支持领域、尤其是质量反馈维度上具有较高的一致性。 |
| [^73] | [${M}^2$Tok: Multi-head Multi-codebook Discrete Action Tokenization for Vision-Language-Action Models](https://arxiv.org/abs/2609.18259) | 提出 M²Tok，一种多头多码本离散动作分词器，通过将潜在动作特征分解为多个头并采用多个码本以最小化重构误差，突破“离散化瓶颈”，从而提升视觉-语言-动作模型的控制性能。 |
| [^74] | [Beyond Accuracy: How Procedural Traces Shift the Decision Criterion of LLM Overseers](https://arxiv.org/abs/2609.18204) | 研究发现，程序化痕迹并不能提升LLM监督者的错误检测能力，反而会使其决策标准向拒绝方向偏移，导致对正确工作的误报增加，且痕迹越详细这种偏见越强。 |
| [^75] | [Behavior2Value: Benchmarking and Empowering LLMs for Consumer Value Measurement from E-commerce Behaviors](https://arxiv.org/abs/2609.18203) | 该论文提出了行为到价值（B2V）任务，构建了首个电子商务消费价值分类体系（ECVT）和基于真实淘宝行为日志的B2V-Bench基准数据集，实现了从电子商务行为轨迹中识别和测量消费者价值观，并据此赋能大语言模型。 |
| [^76] | [T-SANDHI: Tone Sandhi-aware Adaptive Network with Decoupled Hybrid Injection for Low-resource Taiwanese Hokkien Speech Recognition](https://arxiv.org/abs/2609.18194) | 该论文发现台湾闽南语语音识别的真正瓶颈并非连读变调本身，而是变调与本调之间的局部混淆，并提出T-SANDHI模型，通过在冻结的Whisper骨干上显式解耦表层声学与词汇意图，结合词典引导的多任务学习和动态门控混合注入模块，有效提升了低资源台湾闽南语的语音识别性能。 |
| [^77] | [TeochewBench: A Human-Reviewed Benchmark for Teochew Hanzi Translation](https://arxiv.org/abs/2609.18156) | 提出了首个经人工审核的潮州话汉字翻译基准 TeochewBench，包含300条涵盖五大类别的潮州话表达，用于评估大型语言模型在潮州话与普通话、英语之间双向翻译的能力。 |
| [^78] | [PageRecall: Measuring Page Selection in Literature-Grounded Question Answering](https://arxiv.org/abs/2609.18154) | 该论文发现文献问答系统的证据锚定瓶颈在检索而非阅读——页面选择器仅以 52.6% 的召回率将黄金页面呈现给模型，而模型拿到正确页面后引用准确率高达 94%，且页面缺失时往往静默失败，因此提出放弃页面选择、直接将整篇检索到的论文放入模型上下文的解决思路。 |
| [^79] | [DualSQL: Text-to-SQL with Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2609.18135) | DualSQL提出了一种由单一模型主干驱动两个智能体的文本到SQL系统，通过多智能体强化学习框架联合优化模式链接和SQL生成两个相互关联的任务，并借助rollout护栏机制稳定训练过程、防止模型崩溃，同时引入新的鲁棒执行匹配（REX）正确性指标。 |
| [^80] | [Colla-Q: Toward Collaborative Experts in MoE Quantization via Minimax Precision Balancing](https://arxiv.org/abs/2609.18131) | 提出基于激活熵的比特分配框架Colla-Q，通过极小极大精度平衡策略均衡MoE量化中各专家的性能，从而提升整体模型表现并降低对校准数据的依赖。 |
| [^81] | [A Comprehensive Review of Generative Physical Artificial Intelligence](https://arxiv.org/abs/2609.18111) | 本综述系统梳理了生成式物理人工智能（GPAI）领域，提出了涵盖机器人基础模型、视觉-语言-动作模型、大行为模型、扩散策略模型和世界基础模型五大方法的分类体系，并分析了它们的架构基础、应用现状及互补关系。 |
| [^82] | [Linguistic Triggers of Gender and Racial Bias in Open-Weight LLMs Applied to Recruitment](https://arxiv.org/abs/2609.18106) | 该论文首次将招聘启事语言作为实验变量，对六个开源权重大语言模型进行系统性偏见审计，发现能动性语言会触发对女性候选人的性别偏见、排他性编码语言会触发对非白人候选人的种族偏见，并揭示了由此产生的欧盟《人工智能法案》与美国EEOC监管合规风险。 |
| [^83] | [Agora: Git as Shared Memory for Collective AutoResearch](https://arxiv.org/abs/2609.18094) | Agora的核心创新是以Git中仅追加的DAG作为多个自主研究智能体的共享内存，让每条研究成果都成为可验证、可复现的不可变提交，并通过派生索引和多样性感知选择规则，使13个无中央规划的智能体持续协作近12天而不陷入重复搜索。 |
| [^84] | [From a River in Gilead to the Inference Distributions of Large Language Models: Covert Dialect Bias and Linguistic Profiling at Scale](https://arxiv.org/abs/2609.18068) | 本研究借鉴社会语言学配对变体方法，通过对数概率评分对十个开源大语言模型进行探测，发现对齐技术无法消除模型内部概率分布中对非裔美国人白话英语及此前被忽视的尼日利亚英语变体的隐性方言偏见，这种偏见在住房相关社会判断中普遍存在。 |
| [^85] | [Exact semantic readout from compressed vector representations](https://arxiv.org/abs/2609.18047) | 本文给出了压缩向量表示能够精确线性或仿射读出谓词真值条件的充要行空间判据，并通过实验发现预训练词向量虽大多严格可分，但无一能实现精确读出。 |
| [^86] | [Correlation-Guided Encoder Selection for Multi-Encoder Large Audio-Language Models](https://arxiv.org/abs/2609.18041) | 提出CUES方法，利用编码器性能概况间的皮尔逊相关性评估互补性，无需融合训练即可高效为多编码器大型音频-语言模型选出最优编码器组合，在节省计算成本的同时避免冗余表示。 |
| [^87] | [Gaze as Evidence for Common Grounding: A Cross-Corpus Analysis of MapTask and MUNDEX](https://arxiv.org/abs/2609.18011) | 跨MapTask和MUNDEX两个语料库的分析表明，注视行为（尤其是任务主导者的注视）可作为对话中共同基础化程度的可靠行为证据。 |
| [^88] | [G-Mamba: Sparse Graph-Guided Mamba for Audio-Visual Speech Enhancement](https://arxiv.org/abs/2609.18009) | 本文提出SG-Mamba，一种将稀疏异构图与线性复杂度Mamba骨干网络相结合的轻量级视听语音增强框架，通过内容自适应注意力建模跨模态关系并引入音频跳跃连接保留频谱细节，在LRS3上以更低的计算成本取得了竞争性或更优的增强性能。 |
| [^89] | [A Calibrated Instrument for Measuring How Inference Optimizations Affect Output Quality](https://arxiv.org/abs/2609.18005) | 本文提出了一种经过正式校准的LLM评判测量方法，通过引入分布上与原模型完全一致的“零条件”验证机制，实现了对量化、早退、投机解码等推理加速技术对输出质量影响的严格、可跨系统比较的测量。 |
| [^90] | [Modeling the Developmental Shift in Telicity Acquisition](https://arxiv.org/abs/2609.17996) | 该研究提出基于GPT2惊讶度差异的自动标注方法，揭示了儿童与成人在编码终结性时的分化：儿童依靠单一确定性句法线索（动词后限定词）即可准确判断，而成人则更多依赖动词语义信息。 |
| [^91] | [Encoder Awakening via Adapters: Effective Domain-Adaptive Fine-tuning of Speech-LLMs](https://arxiv.org/abs/2609.17981) | 提出EAVA方法，通过在语音大语言模型的每个编码器层插入轻量级适配器并进行专门训练，在保留预训练知识的同时注入目标领域声学知识，有效提升有限数据下儿童语音、方言语音等领域偏移语音的识别性能。 |
| [^92] | [TACTICS: Taxonomy-Aware Intelligent Corpus Sampling for Machine Translation](https://arxiv.org/abs/2609.17956) | 该论文提出TACTICS方法，将机器翻译评估中的语料抽样从随机方式转变为显式的覆盖优化问题：通过从本地化风格指南构建层次化分类体系，在固定预算下联合优化稀有类别覆盖、文档级连贯性和语料分布保真度，从而为系统鲁棒性评估提供覆盖保证。 |
| [^93] | [ASPIRE: Asynchronous Batched Self-Speculative Decoding for Long-Context LLM Inference](https://arxiv.org/abs/2609.17943) | ASPIRE提出了一种非同步的批量自推测解码框架，通过统一混合前向计算、基于接受率估计和批次感知成本模型的在线调度器，让批中每个请求独立决定验证时机，从而加速长上下文大语言模型推理。 |
| [^94] | [Long-Context Demonstration Selection Using State Space Models](https://arxiv.org/abs/2609.17888) | 本文提出一种基于状态空间模型（SSM）的示例选择方法，通过从transformer模型蒸馏出线性的SSM，高效解决长上下文场景下推理成本高企的示例选择难题。 |
| [^95] | [Uncertainty-Aware Continual Learning for Open-World Intent Discovery Under an evolving Label Space](https://arxiv.org/abs/2609.17866) | 该论文提出了一种统一的不确定性感知概率框架，通过自适应β-VAE编码、分类器置信度-后验不确定性-DP-GMM似然的多信号决策机制和基于密度的聚类发现，在演化的标签空间下实现开放世界新意图的持续发现与可控标签空间扩展，并结合回放与弹性权重巩固来缓解灾难性遗忘。 |
| [^96] | [Who Judges Matters: Measuring Family-Conditioned Preference in LLM-as-Judge Panels](https://arxiv.org/abs/2609.17857) | 该研究首次系统测量了大语言模型评审中的“同家族偏好”效应——即模型评审会偏袒同一家族的候选模型——通过提出一种固定候选家族的校正估计器，发现四个主流开放权重模型家族均存在3.4-8.4个百分点的显著同家族提升，且该效应与评审侧似然度密切相关。 |
| [^97] | [AfriSyCo: Measuring Assertive Framing, Verification, and Wording Sensitivity Around African-Language Content](https://arxiv.org/abs/2609.17853) | 该论文提出AfriSyCo框架，通过母语后续提问与跨语言2×2因子实验系统测量非洲语言事实内容中模型答案切换行为，发现断言式框架会显著增加错误目标选择（+30.4个百分点），而验证机制可有效降低该效应（-17.4个百分点）。 |
| [^98] | [SFT or RL for Tool-Calling Agents? A Controlled Study Across Data, Method, and Scale](https://arxiv.org/abs/2609.17848) | 通过在0.6B至32B规模的六个Qwen3模型上进行受控实验发现，带LoRA的SFT是分布内工具调用最强的训练方法，而无论采用何种方法，数据集混合都是获得强跨数据集迁移能力的最可靠手段。 |
| [^99] | [PrimeScientist: Strategic Allocation of Research Effort in Autonomous Research](https://arxiv.org/abs/2609.17846) | 提出了PrimeScientist框架，将自主研究智能体的研究方向选择与资源投入决策统一建模为序贯决策问题，通过可执行计划树保留竞争性方案及其结果，并利用剩余资源显式引导研究策略，实现研究努力的战略性分配。 |
| [^100] | [How Calibration Content Shapes Attention-Based Reranking](https://arxiv.org/abs/2609.17764) | 该论文揭示了注意力重排序器中的空查询校准在面对包含详细指令的提示时会错误地移除相关信号，并提出了一种无需训练的插值空校准方法，通过控制进入空基线的指令内容比例，恢复了指令密集型任务上的重排序性能。 |
| [^101] | [Is Luke the Author of a Gospel and the Acts of the Apostles?](https://arxiv.org/abs/2609.17762) | 本研究运用Burrows' Delta作者归属方法和作者验证模型等定量分析手段，证实《路加福音》和《使徒行传》确实出自路加一人之手。 |
| [^102] | [Evolution of US Oral Political Language](https://arxiv.org/abs/2609.17755) | 本研究首次通过分析1960至2024年间19位美国总统选举候选人的口头辩论语言，揭示了美国政治语言随时间显著简化、平均句长持续缩短的长期演变趋势。 |
| [^103] | [Is Trump's Vocabulary Poor? Vocabulary Richness Across Texts of Different Lenghts](https://arxiv.org/abs/2609.17747) | 本研究提出将词汇细分为通用与专业词汇表来解释词汇量增长的模型，以此评估口头政治传播中的词汇丰富度。 |
| [^104] | [Confidence Comes from Experience: Experiential Confidence Estimation from Reasoning to Agents](https://arxiv.org/abs/2609.17708) | 论文提出XConf，通过检索模型积累的过往分级经验记录（含任务、反思、置信度、结果与教训）来估计置信度，突破了仅依赖当前推理过程的传统置信度估计范式。 |
| [^105] | [NeMo Data Designer: An Extensible Framework for Multimodal Synthetic Data Generation](https://arxiv.org/abs/2609.17699) | NeMo Data Designer是一个开源、可扩展的多模态合成数据生成框架，通过声明式配置、灵活的插件系统和内置的预览-修订迭代循环，实现了直观、可复现且可迭代的数据集生成。 |
| [^106] | [GraphEcho: Structural Redundancy and Evidence Provenance in LLM Graph Agents](https://arxiv.org/abs/2609.17695) | GraphEcho基准测试揭示LLM图智能体会将结构冗余的重复路径误认为额外佐证，而溯源感知后训练（PAPT）虽能减少重复探索，却暴露了高效探索与有效证据利用之间的根本差距。 |
| [^107] | [The Missing "I Don't Know": Why Three Reasoning-Reliability Findings Converge on Calibrated Abstention](https://arxiv.org/abs/2609.17686) | 该论文的核心创新是论证三项看似独立的LLM可靠性研究发现（推理强化学习破坏工具可靠性表征、安全约束下大小模型的差异化表现、以及缺乏“我不知道”功能的系统必然产生无穷幻觉）实际上共同指向同一项缺失能力——校准弃答。 |
| [^108] | [Fathom: Per-Query Read Depth for Sparse Decoding over Offloaded KV Caches](https://arxiv.org/abs/2609.17652) | Fathom提出一种让每个查询自适应决定键通道读取位数的稀疏解码方法，通过比特平面存储与逆注水式比特预算分配，在百万token卸载KV缓存场景下实现比现有136位扫描方法快1.67倍的GPU解码速度，同时保持更低注意力误差。 |
| [^109] | [EvolveTrade: Experience-Driven Policy Refinement for Self-Evolving LLM Trading Agents](https://arxiv.org/abs/2609.17632) | EvolveTrade提出将LLM交易智能体的系统提示词作为文本参数化策略，通过策略智能体利用积累的决策轨迹和已实现的投资组合反馈不断修订该策略，在保持骨干模型不变的情况下实现交易决策流程的自进化，从而在多种市场环境下提升夏普比率等交易表现。 |
| [^110] | [Making Political Text Scaling Comparable: Infrastructure and Hyperparameter Sensitivity for 17 Algorithms](https://arxiv.org/abs/2609.17602) | 本文通过涵盖17种算法、5,537次实验和约425万个立场估计的大规模比较实验，论证了政治文本理想点估计方法应被视为可配置的测量流程而非固定算法，并发现绝大多数算法的估计结果对超参数选择并不敏感。 |
| [^111] | [English Word Sense Disambiguation in 2026: When the Labels Become the Bottleneck](https://arxiv.org/abs/2609.17554) | 该论文指出英语词义消歧的瓶颈已从模型转移到标注数据，并发布了经人工修正的lexEN评测基准与可审计的SenseBench评测框架，结果显示前沿大语言模型准确率已收敛至约95%，金标准中的标注错误成为决定排名的关键因素。 |
| [^112] | [The Limits of BPE Tokenization in Polish: Segmentation-Flexional Forms, Grammatical Anchoring, and First-Person Stability in Inflectional Language Models](https://arxiv.org/abs/2609.17553) | 该研究以波兰语为测试案例，表明BPE分词在屈折语言中只能稳定高频的表面文字片段，无法系统性保留音位结构、屈折词尾和语法形式等语言学相关单位。 |
| [^113] | [Does Moral Reasoning Training Help or Hurt? Red-Teaming RL-Trained Ethical Agents with Persona Attacks](https://arxiv.org/abs/2609.17552) | 研究发现道德奖励强化训练虽能将语言模型智能体对人格攻击的鲁棒性提升5.8倍，但会牺牲约11个百分点的ETHICS道德基准准确率，且模型内部存在单一表征方向即可恢复83%对抗效果，揭示了道德对齐的脆弱性机制。 |
| [^114] | [No Usable Linear "Capitulation Direction" in Two Small LLMs: A Validation Protocol for Activation-Steering Claims, and a Cross-Family Behavioral Study of Sycophancy Under Pushback](https://arxiv.org/abs/2609.17550) | 该研究通过验证协议发现两个小型LLM中不存在可用的线性“屈服方向”，并首次跨模型家族揭示：模型在反驳下放弃正确答案的谄媚比例高达41.8%-43.1%，且哪种反驳方式有效及失败模式均强烈依赖于具体模型家族。 |
| [^115] | [Do Social Patterns Hold in Synthetic Data? Analyzing Cyberbullying Dynamics in LLM-Generated and Authentic Dialogues](https://arxiv.org/abs/2609.17549) | 本文提出了一个评估LLM生成网络欺凌对话社会真实性的综合框架，从互动结构、语言风格、情感行为标记和时间升级动态等多维度比较GPT、Grok和LLaMA生成的合成对话与真实对话，揭示了合成数据在再现真实社会动态方面的局限性。 |
| [^116] | [Myovox: Reading Speech from the Muscles of the Face](https://arxiv.org/abs/2609.17548) | Myovox通过忠实复现基线、采用跨模态蒸馏训练的双向Conformer编码器以及多模型集成重排序三个步骤，将面部肌电信号解码开放词汇语音的单词错误率从51.17%大幅降低至18.53%。 |
| [^117] | [How AI Assistants Respond to Repeated Abuse](https://arxiv.org/abs/2609.17547) | 该研究提出了一个双语多轮评估框架，首次区分AI助手面对反复言语辱骂时的“硬性脱离”与“软性退出”两种行为，并发现不同模型配置间的硬性脱离率差异巨大（从0%到50%）。 |
| [^118] | [Legal LLM Hallucination Should Be Evaluated as Failure of Legal Warrant](https://arxiv.org/abs/2609.17546) | 本文主张法律大语言模型的幻觉应被评估为“法律授权的失败”——即法律主张未能与真实存在、现行有效、适用于相关管辖区的法律权威建立支持关系——并提出了能捕捉现有准确性与引用类评估方法所遗漏的重大失败的可证伪授权指标。 |
| [^119] | [Large Language Models Versus Physicians in Traditional Chinese Medicine: A Real-World Clinical Case Evaluation](https://arxiv.org/abs/2609.17544) | 本研究构建了涵盖62家医院349例门诊病例的中医临床病例库，发现前沿通用大语言模型在多数诊疗维度上的专家评分超过执业中医医师，但在处方层面的中药选择、剂量与治疗策略上仍存在差异，并伴有幻觉和模板化输出等安全性问题。 |
| [^120] | [Register Bias in Complexity-Based Large Language Model Routing](https://arxiv.org/abs/2609.17542) | 基于复杂度的大语言模型路由存在语域偏差：非标准英语（如非裔美国人英语或二语学习者英语）因省略功能词而显得更短，会被系统性地路由到能力更低的模型，导致服务质量受损。 |
| [^121] | [MudawanSn: A Gold-Standard Wolof-Arabic Parallel Corpus for Machine Translation](https://arxiv.org/abs/2609.17539) | 本文发布了首个专门面向沃洛夫语-现代标准阿拉伯语语言对的黄金标准平行语料库MudawanSn（包含1,271个人工翻译的句对齐句对），并通过实验证明在该语料库上微调能显著提升双向机器翻译性能。 |
| [^122] | [From Pixels to Pairs: A Comprehensive Benchmark of LLM-Based Key-Value Extraction in Noisy Document Settings](https://arxiv.org/abs/2609.17538) | 该论文建立了一个系统性基准，评估开源指令微调大语言模型在干净文本与含噪OCR条件下提取键值对的能力，发现现代LLM在高质量文本输入下是强大的语义提取器（部分情况接近有监督布局感知系统），但在OCR噪声下性能显著下降。 |
| [^123] | [Relation Before Entity: Deferred Commitment in Language Model Factual Recall](https://arxiv.org/abs/2609.17537) | 该研究发现语言模型在事实回忆中，关系信息比实体信息早10-16个层变得控制生成，而实体信息并非早期缺失，其“承诺”被延迟至被路由到最后token位置才具有因果控制作用。 |
| [^124] | [Think Before You Comfort: Reflective Cognitive Alignment for Protocol-Grounded Elderly Stimulation Agents](https://arxiv.org/abs/2609.17536) | 本文提出结合STaR-CS数据合成方法与反思性认知对齐（RCA）框架，将认知刺激交互建模为协议约束的序贯决策过程，使大语言模型智能体能够在粤语等低资源场景下兼顾共情陪伴与认知刺激协议的严格遵循。 |
| [^125] | [DANTINOX: A Unified Framework for Multi-Paradigm Language Modeling](https://arxiv.org/abs/2609.17535) | DANTINOX是一个开源的JAX/Flax库，通过统一的模块化Transformer骨干网络支持自回归解码、离散掩码扩散和连续流匹配三种语言生成范式，从而实现了可控的跨范式公平比较。 |
| [^126] | [Faking Good and Faking Bad in LLMs: Response Distortion Across Dark Triad Personality Traits](https://arxiv.org/abs/2609.17534) | 本研究首次系统证实，当代大语言模型会像人类一样响应“装好”与“装坏”的社会赞许性激励，系统性地调节黑暗三联征人格特质的表达，揭示了大语言模型在人格评估中存在类似人类的反应失真现象。 |
| [^127] | [Enhancing Extubation Failure Prediction with LLM-Derived Features from Respiratory Therapy Clinical Notes](https://arxiv.org/abs/2609.17532) | 该论文提出利用大语言模型从自由文本呼吸治疗临床笔记中提取特征，与结构化数据结合后显著提升了拔管失败预测性能，并揭示了既往研究在目标人群定义上的差异如何阻碍模型的泛化能力。 |
| [^128] | [Vroom-Vroom at SHROOM-Visions: A Multi-Judge Committee for Detecting Hallucinated Spans in Vision-Language Outputs](https://arxiv.org/abs/2609.17327) | 该论文提出一种多裁判委员会方法，通过多个微调视觉-语言模型的字符级多数投票来检测视觉-语言输出中的幻觉片段，在SHROOM-Visions竞赛四种语言中的三种排名第一。 |
| [^129] | [Zero-shot narrative detection in social messaging](https://arxiv.org/abs/2609.17310) | 本研究证明大语言模型在零样本设置下，只需提供人工编写的叙事描述即可有效检测社交消息中的隐藏战略性叙事，无需训练样本，且集成方法与更大规模模型可进一步提升检测性能与鲁棒性。 |
| [^130] | [Parameter-Efficient Retrievers for Polish and European Languages](https://arxiv.org/abs/2609.12913) | 提出了一种结合跨语言对齐、关系知识蒸馏和对比微调的三阶段训练流程，无需原始相关性标注即可训练出参数量小但性能可媲美大型模型的波兰语和欧洲多语言稠密检索器。 |
| [^131] | [Creating an Atomic User Model for Personality-Aware Large Language Model Interaction](https://arxiv.org/abs/2609.12086) | 该论文提出原子用户模型（AUM），将用户表示为稳定身份核心加四个可解释外壳的分层结构，解决了现有助手仅依赖偏好总结而在任务变化时需反复重新学习用户的问题，并首次刻画了“人格渗漏”现象。 |
| [^132] | [When Personality Meets Quantization: A Layer-wise MBTI Analysis of Quantized LLMs](https://arxiv.org/abs/2608.25977) | 本文首次系统分析了量化大型语言模型在不同精度下的个性特征，并提出了新方法来揭示个性如何在层间涌现及推理时漂移。 |
| [^133] | [TurnBench: A Multi-Domain Benchmark for Turn-Taking Dynamics in Spoken Dialogue](https://arxiv.org/abs/2608.25218) | 本文提出了TurnBench，一个多领域基准测试，结合30小时人工标注语料库和标准化评估协议，系统评估14个话轮转换系统，发现话轮结束检测稳定而打断误报率高度依赖对话类型。 |
| [^134] | [CROP: Task Relevance via Counterfactuals for Selective On-Policy Distillation](https://arxiv.org/abs/2608.13387) | CROP提出了一种基于释义校准的反事实敏感性边际方法，用于在选择性在线策略蒸馏中直接量化任务相关性，从而更有效地分配监督信号。 |
| [^135] | [RT-SEMamba: Real-Time Speech Enhancement Mamba via Progressive Knowledge Distillation](https://arxiv.org/abs/2608.12099) | 该论文提出RT-SEMamba，一种基于因果时频Mamba块的实时语音增强模型，并通过渐进式知识蒸馏将8层教师压缩为1层学生，在保持低延迟的同时显著提升质量并实现2.75倍加速。 |
| [^136] | [LEEPS: Latent-Guided Explore-Exploit Prompt Sampling for Efficient RLVR in Large Language Models](https://arxiv.org/abs/2607.28077) | 提出LEEPS方法，通过潜变量引导的探索-利用提示采样策略，根据提示最近的非平凡比例自适应分配滚动生成预算，平衡信息丰富提示的复用与不确定提示的探索，从而提升大型语言模型RLVR训练效率。 |
| [^137] | [Delayed Verification Destabilizes Multi-Agent LLM Belief: Instability Thresholds and Optimal Corrector Placement](https://arxiv.org/abs/2606.27409) | 该论文将多智能体LLM系统中的延迟验证建模为带接地节点的延迟共识问题，通过接地拉普拉斯谱分解推导出验证剂量的闭式失稳阈值（延迟为二时为黄金比例的倒数），并基于超模目标给出贪婪(1-1/e)近似的纠错节点最优配置方法。 |
| [^138] | [Follow the Latent Roadmap: Navigating Revocable Decoding for Diffusion LLMs with Anchor Tokens](https://arxiv.org/abs/2606.16847) | 提出了一种免训练框架ASRD，通过在嵌入空间中将解码上下文解耦为基于时间一致性识别的受信任锚定令牌和不确定候选令牌，解决了扩散大语言模型可撤销解码中的错误传播与局部错误强化问题。 |
| [^139] | [When Cognitive Graphs Meet LLMs: BDEI Cognitive Pathways for Panic Emotional Arousal Prediction](https://arxiv.org/abs/2606.15121) | 该论文主张基于评价情绪理论在自然生成方向上显式建模恐慌情绪唤醒过程，通过将认知图与大语言模型相结合构建BDEI认知路径，以预测个体和集体恐慌情绪唤醒的时间点，从而实现及时的紧急干预。 |
| [^140] | [Notes2Skills: From Lab Notebooks to Certainty-Aware Scientific Agent Skills](https://arxiv.org/abs/2606.11897) | 本文提出Notes2Skills框架，通过两阶段方法将非正式的实验室笔记转化为具备确定性感知的科学智能体技能，使AI智能体能够正确区分实验记录中已验证的观察、暂时性判断和可执行的实验建议，避免将不确定的科学判断误认为已确认的结论。 |
| [^141] | [Multi-Hop Knowledge Composition is Bound by Pretraining Exposure](https://arxiv.org/abs/2606.09338) | 该研究揭示大语言模型的多跳知识组合能力本质上受预训练时对组合上下文暴露程度的限制——组合式预训练只能迁移到已暴露个体的未见问题，而永远无法惠及从未在组合上下文中出现过的个体。 |
| [^142] | [From 'May' to 'Is': Certainty Distortion in Language Model Rewriting](https://arxiv.org/abs/2606.07951) | 该研究发现语言模型在重写科学和医学文本时会系统性地改变原文的确定性程度（如把“可能”改成“是”），这种失真影响高达75%的输出且呈不对称性。 |
| [^143] | [How Do Document Parsers Break? Auditing Structural Vulnerability in Document Intelligence](https://arxiv.org/abs/2605.19309) | 该论文提出轻量级输出层审计框架 ProSA，识别出文档解析器鲁棒性评估中的“足迹偏差”，并证明块级结构损失率（B-SLR）比受影响面积更能准确刻画扰动引起的结构失效及其传播路径。 |
| [^144] | [Schema-Key Wording as an Instruction Channel in Structured Generation under Constrained Decoding](https://arxiv.org/abs/2604.14862) | 该论文首次系统研究了约束解码下JSON模式键措辞作为隐式指令通道的作用，揭示仅改变键的措辞即可显著影响大语言模型结构化生成的准确率，并从理论上给出了指令优势在语法投影后得以保留的充分条件。 |
| [^145] | [Correct Prediction, Wrong Steps? Consensus Reasoning Knowledge Graph for Robust Chain-of-Thought Synthesis](https://arxiv.org/abs/2604.14121) | 提出CRAFT方法，通过聚合多个候选推理轨迹的共识组件构建推理知识图谱，从推理结构层面修复LLM“答案正确但推理步骤有缺陷”的问题，实现更鲁棒的思维链合成。 |
| [^146] | [Can We Still Trace L1 Signals? Investigating the Resilience of Native Language Signals in the LLM Era](https://arxiv.org/abs/2604.08568) | 本研究通过构建覆盖神经网络前、LLM前和LLM后三个时代、八个母语群体的学术摘要母语识别数据集，发现文本中的母语信号随时间持续减弱，且英语同质化的主要趋势早在LLM出现之前就已开始。 |
| [^147] | [A Taxonomy of Programming Languages for Code Generation](https://arxiv.org/abs/2604.00239) | 该论文首次提出了一个可复现的编程语言资源分类体系，将646种编程语言划分为四个层级，并揭示了代码语料库中极端且系统性的资源失衡——仅1.9%的语言占据了74.6%的token。 |
| [^148] | [AuthorMix: Modular Authorship Style Transfer via Layer-wise Adapter Mixing](https://arxiv.org/abs/2603.23069) | AuthorMix提出了一种轻量级、模块化的作者风格迁移框架，通过在高资源作者上训练LoRA适配器并结合强化学习的逐层适配器混合，仅需少量目标风格样本即可快速适配新作者，其风格-含义综合得分超越包括GPT-5.1在内的所有基线方法。 |
| [^149] | [Assessing the Effect of Cross-Domain Mapping on Creativity in Humans and Large Language Models](https://arxiv.org/abs/2603.19087) | 跨领域随机联想能稳定提升人类的创意原创性，但对大语言模型的作用取决于其能力水平和语义距离，且两者利用灵感的方式不同——人类迁移表面特征，而大语言模型迁移结构与功能属性。 |
| [^150] | [CzechTopic: A Benchmark for Zero-Shot Topic Localization in Historical Czech Documents](https://arxiv.org/abs/2603.03884) | 该论文推出了基于捷克历史文档的人工标注零样本主题定位基准CzechTopic，支持文档级与词级评估，发现最强LLM接近人类一致性水平，而小规模的蒸馏BERT模型仍具竞争力。 |
| [^151] | [Mind the Style: Impact of Communication Style on Human-Chatbot Interaction](https://arxiv.org/abs/2602.17850) | 本研究发现，友好型沟通风格的聊天机器人在提升用户满意度和任务成功率方面优于直接型风格，但无聊天机器人的控制条件在任务成功率上表现最佳。 |
| [^152] | [Understanding LLM Failures: A Multi-Tape Turing Machine Analysis of Systematic Errors in Language Model Reasoning](https://arxiv.org/abs/2602.15868) | 该论文提出用确定性多带图灵机形式化大语言模型的完整交互流程，从而精确定位各类失败模式的发生阶段，并解释了思维链提示为何有效及其根本局限。 |
| [^153] | [Patch the Distribution Mismatch: RL Rewriting Agent for Stable Off-Policy SFT](https://arxiv.org/abs/2602.11220) | 提出一种用强化学习训练的轻量级LoRA改写策略，在任务一致性约束下优化问答分布对齐与语义多样性，从而修补下游监督数据与模型生成分布之间的失配，缓解SFT中的灾难性遗忘。 |
| [^154] | [HALT: Hallucination Assessment via Log-probs as Time series](https://arxiv.org/abs/2602.02888) | 该论文提出HALT，一种仅利用LLM生成的前20个token对数概率作为时间序列、结合GRU模型与熵特征进行幻觉检测的轻量级方法，无需访问模型内部状态即可实现强泛化能力，并配套发布了统一的幻觉检测基准HUB。 |
| [^155] | [ProofVerifier: A Scalable, Diversity-Driven Framework for Natural-Language Proof Verification](https://arxiv.org/abs/2602.02377) | 该论文提出ProofVerifier框架，通过LLM辅助的数据管道大规模生成多样化的问题-证明-检查样本，并结合多模型一致性与分层人工审计获得准确标签，从而训练出可靠的自然语言数学证明验证器。 |
| [^156] | [Enhancing knowledge tracing robustness for new question cold start in Intelligent Tutoring Systems](https://arxiv.org/abs/2512.07179) | 本研究设计了集成多种特征的PICKT知识追踪模型，并实证分析了题目难度、文本以及知识图谱关系信息等特征在提升新题目冷启动场景下知识追踪模型鲁棒性方面的作用。 |
| [^157] | [Seeing Through the MiRAGE: Evaluating Multimodal Retrieval Augmented Generation](https://arxiv.org/abs/2510.24870) | MiRAGE是一个以论断为中心的多模态检索增强生成评估框架，通过InfoF1和CiteF1指标评估事实性、信息覆盖度与引用完整性，在文本任务上优于现有RAG评估指标，且是唯一能够推广到多模态来源的方法。 |
| [^158] | [AgentPack: A Dataset of Code Changes, Co-Authored by Agents and Humans](https://arxiv.org/abs/2509.21891) | 提出 AgentPack——一个包含 180 万条由人类与 AI 智能体（如 Claude Code）共同完成的代码编辑的数据集，相比传统从提交记录中挖掘的数据，其意图描述更明确、质量更可靠。 |
| [^159] | [Modelling Adjectival Modification Effects on Semantic Plausibility](https://arxiv.org/abs/2507.21828) | 本文针对形容词修饰如何改变语义合理性的Adept基准，提出了一种基于句子Transformer的概念新颖的建模方法，并发现句子Transformer尽管在概念上契合该任务，其表现却不及RoBERTa等Transformer模型。 |
| [^160] | [Donate or Create? Comparing Data Collection Strategies for Emotion-labeled Multimodal Social Media Posts](https://arxiv.org/abs/2505.24427) | 本研究比较了“捐赠真实帖子”与“研究创作帖子”两种情感标注数据收集策略，发现研究创作的内容更长、更依赖文本而非图像表达情感、更聚焦典型情感事件，揭示了不同数据收集方式会导致数据特性产生显著差异。 |
| [^161] | [PBEBench: A Multi-Step Programming by Examples Reasoning Benchmark inspired by Historical Linguistics](https://arxiv.org/abs/2505.23126) | 该论文提出了PBEBench，一个受历史语言学正向重构任务启发的多步骤示例编程基准测试，用于评估大语言模型的归纳推理能力，并配备可自动生成可控难度问题、避免数据污染的自动化流水线。 |
| [^162] | [Extracting Probabilistic Knowledge from Large Language Models for Bayesian Network Parameterization](https://arxiv.org/abs/2505.15918) | 本研究证明大语言模型可以有效提取概率知识用于贝叶斯网络参数化，其在八十个不同领域网络上的条件概率估计结果显著优于随机分布等基线方法。 |
| [^163] | [Learning from Many Voices: Literary MT Using Multi-Reference Human and Synthetic Data](https://arxiv.org/abs/2412.18707) | 提出基于语义相似度的过滤框架来利用文学多参考数据集改进文学机器翻译，并通过自动指标和人工评估证明人工专家译文的微调效果优于大语言模型生成的合成数据。 |
| [^164] | [Divide and Conquer: A Hybrid Strategy Defeats Multimodal Large Language Models](https://arxiv.org/abs/2412.16555) | 本文提出了一种名为JMLLM的多模态越狱攻击方法，通过整合多种混合策略在文本、视觉和听觉三种模态上对大语言模型进行全面越狱攻击，克服了现有方法查询次数过多、模态覆盖有限和攻击成功率低等局限。 |
| [^165] | [Label-Confidence-Aware Uncertainty Estimation in Natural Language Generation](https://arxiv.org/abs/2412.07255) | 提出了一种基于逐点KL散度的标签-置信度感知不确定性量化方法（LCA-UQ），弥合了多样本全局熵与候选答案局部置信度之间的差距，从而更准确地评估大语言模型生成回答的有效性，缓解幻觉问题。 |
| [^166] | [Accelerating Stateful Network Applications with Performance Prediction on SoC SmartNICs](https://arxiv.org/abs/2410.22229) | Vela框架通过以状态为中心的分析模型驱动的编译时/运行时协同设计，利用预测性编译器自动估算资源分配的吞吐量上限并动态适应流量变化，从而加速有状态网络应用在SoC智能网卡上的卸载。 |
| [^167] | [Which Demographics do LLMs Default to During Annotation?](https://arxiv.org/abs/2410.08820) | 该研究结合LLM偏差研究与人口统计学条件提示两条路线，首次探究了当提示中未提供人口统计学信息时，大语言模型在标注任务中会默认模仿哪些人类标注者群体的人口统计学特征。 |
| [^168] | [Unleash LLMs Potential for Sequential Recommendation by Coordinating Dual Dynamic Index Mechanism](https://arxiv.org/abs/2409.09253) | 该论文提出了首个采用双重动态索引机制的端到端大语言模型序列推荐系统ED²，将索引生成与序列推荐统一到单一LLM主干流水线中，同时解决了语义信息与协同信息整合不足以及高阶用户-物品交互模式利用不充分的问题。 |
| [^169] | [A Systematic Review of NLP for Ghanaian Languages: Datasets, Models, and a Research Roadmap](https://arxiv.org/abs/2405.06818) | 本文首次系统综述了加纳NLP领域，通过分析36项核心研究揭示了严重的资源失衡问题——仅特威语有适度发展而其余70多种语言几乎空白，并针对区域限制、方言差异、书写系统非标准化和基础设施缺失提出了优先级研究路线图。 |
| [^170] | [Attribution in Scientific Literature: New Benchmark and Methods](https://arxiv.org/abs/2405.02228) | 该论文提出了科学引用归因基准REASONS以及由弃答率与幻觉率构成的双指标评估框架，系统评估了大语言模型在不同证据条件下的引用归因能力，发现高级RAG虽能降低幻觉率但牺牲了弃答能力，而对抗性元数据会使多个系统的幻觉率超过85%。 |
| [^171] | ["You are an expert annotator": Automatic Best-Worst-Scaling Annotations for Emotion Intensity Modeling](https://arxiv.org/abs/2403.17612) | 自动标记情绪强度建模中的最佳-最差标度注释方法的性能表现 |

# 详细

[^1]: 目标函数与搜索：分解优秀分词器的构成要素

    Objective vs. Search: Decomposing What Makes a Good Tokeniser

    [https://arxiv.org/abs/2609.19145](https://arxiv.org/abs/2609.19145)

    该论文通过引入两种新分词算法填补2x2设计空间、解耦优化目标与搜索过程，发现搜索过程（而非优化目标）才是决定分词器质量的主导因素。

    

    摘要：现代语言模型主要使用两种分词算法：字节对编码和UnigramLM。这两种算法在两个正交的维度上存在差异：它们的优化目标（压缩 vs. 对数似然）和搜索过程（自底向上合并 vs. 自顶向下剪枝）。现有的比较混淆了这两个维度，导致无法确定观察到的差异究竟源于“优化什么”还是“如何优化”。我们通过引入两种新的分词算法来完成这一2x2设计空间，从而将两者解耦：BottomUpLL（一种自底向上的基于似然的分词器）和TopDownComp（一种自顶向下的基于压缩的分词器）。我们使用由每种算法生成的分词器训练语言模型，并改变以下变量：模型大小、词表大小和领域（仅英语 vs. 多语言）。通过在每字节比特数（bits-per-byte）指标上评估模型，我们发现搜索过程——而非优化目标——是主导因素：自底向上分词器……

    arXiv:2609.19145v1 Announce Type: cross  Abstract: Two dominant tokenisation algorithms are used by modern language models: byte-pair encoding (BPE) and UnigramLM. These differ along two orthogonal axes: their optimisation objective (compression vs. log-likelihood) and their search procedure (bottom-up merging vs. top-down pruning). Existing comparisons confound these axes, making it unclear whether their observed differences stem from what is being optimised vs. how it is being optimised. We disentangle the two by introducing two new tokenisation algorithms that complete this 2x2 design space: BottomUpLL, a bottom-up likelihood-based tokeniser, and TopDownComp, a top-down compression-based tokeniser. We train language models with tokenisers produced by each algorithm, varying: model size, vocabulary sizes, and domain (English-only vs. multilingual). Evaluating models on bits-per-byte, we find that the search procedure -- not the objective -- is the dominant factor: bottom-up tokeniser
    
[^2]: 大语言模型偏好对齐的零阶范式

    A Zeroth-Order Paradigm for LLM Preference Alignment

    [https://arxiv.org/abs/2609.19144](https://arxiv.org/abs/2609.19144)

    本文提出了基于比较预言机的零阶偏好对齐方法 ComPO，能从微小似然边距的偏好对中提取方向性信息而无需优化可微偏好损失，并建立了收敛保证且引入了具备反向 KL 控制的在线版本。

    

    直接偏好对齐方法因其计算和内存效率，被广泛用于将大语言模型（LLM）与人类偏好进行对齐。然而，似然位移现象促使人们探索从具有微小似然边距的偏好对中提取信息的替代方法。在本文中，我们提出并分析了基于比较的偏好优化（ComPO），这是一种基于比较预言机的零阶对齐方法。ComPO 从这些偏好对中提取方向性信息，而无需直接在其上优化可微分的偏好损失。我们在平滑性、梯度稀疏性以及预言机与潜在目标兼容性的假设下，为其基本离线方案建立了收敛保证。我们进一步提出了在线 ComPO，它保留了离线比较机制，并利用无标签的策略生成相对于参考策略进行反向 KL 散度控制。基于覆盖度的视角……

    arXiv:2609.19144v1 Announce Type: cross  Abstract: Direct preference alignment methods are widely used to align large language models (LLMs) with human preferences because of their computational and memory efficiency. However, likelihood displacement motivates alternative ways to extract information from preference pairs with small likelihood margins. In this paper, we propose and analyze Comparison-based Preference Optimization (ComPO), a zeroth-order alignment method based on comparison oracles. ComPO extracts directional information from these pairs without directly optimizing a differentiable preference loss on them. We establish a convergence guarantee for its basic offline scheme under smoothness, gradient sparsity, and compatibility between the oracle and a latent objective. We further introduce online ComPO, which retains the offline comparison mechanism and uses unlabeled policy generations for reverse-KL control relative to a reference policy. Following the coverage perspecti
    
[^3]: PANORAMA：基于掩码提议选择的全景接地域描述

    PANORAMA: Panoptic Grounded Captioning via Mask Proposal Selection

    [https://arxiv.org/abs/2609.19143](https://arxiv.org/abs/2609.19143)

    提出PANORAMA框架，通过掩码提议选择机制解决全景接地域描述任务，使视觉-语言模型能够生成覆盖前景与背景的密集描述并将每个短语与像素级掩码精确对齐，同时引入人工标注基准PanoCaps支持训练与评估。

    

    在现实世界中行动的智能系统需要既全面又在空间上定位准确的图像理解。当前的视觉-语言模型（VLM）能够生成流畅且详细的图像描述，但将其与图像像素可靠关联仍然具有挑战性。现有的将密集描述与像素级定位相结合的方法，往往产生不完整的描述或不准确的分割掩码。我们通过全景接地域描述来研究这一问题，该任务要求VLM同时描述前景物体和背景区域，并用像素级掩码对每个指代短语进行定位。我们做出了三项贡献。首先，我们介绍了PanoCaps，这是一个基于全景分割数据集构建的人工标注基准，提供了具有近乎完整像素覆盖的密集描述以及实体级的图文对齐，支持训练和评估。我们进一步提出了一个短……（摘要不完整，原文截断）

    arXiv:2609.19143v1 Announce Type: cross  Abstract: Intelligent systems that act in the world require image understanding that is both comprehensive and spatially grounded. Current vision-language models (VLMs) can generate fluent and detailed image captions, but reliably associating them with image pixels remains challenging. Existing methods that combine dense captioning with pixel-level grounding often produce either incomplete descriptions or inaccurate segmentation masks. We study this problem through panoptic grounded captioning, a task that requires a VLM to describe both foreground objects and background regions while grounding each referring phrase with pixel-level masks. We make three contributions. First, we introduce PanoCaps, a human-annotated benchmark constructed from panoptic segmentation datasets. It provides dense captions with near-complete pixel coverage and image-text alignments at the entity level, supporting both training and evaluation. We further propose a phras
    
[^4]: ScienceIDE：将全球科学代码库转化为智能体可学习的环境

    ScienceIDE: Turning World's Scientific Codebase into Agent Learnable Environments

    [https://arxiv.org/abs/2609.19134](https://arxiv.org/abs/2609.19134)

    ScienceIDE提出了一个将全球科学代码库转化为可编程智能体环境的基础设施，通过专家定义的科学案例与验收标准，指导智能体将代码库转化为支持任务生成、执行和科学验证的可执行环境，并基于验证的交互轨迹训练出PhAI-IDE系列模型，在科学代码修复等任务上取得显著提升。

    

    科学代码库以可执行的模型、方法和工具的形式编码了数十年的人类知识。然而，碎片化的工具链、隐性的领域惯例以及专业化的正确性标准，使得这些知识难以转化为可靠的学习经验——我们将这一挑战称为“科学经验瓶颈”。我们提出ScienceIDE，这是一个将全球科学代码转化为可编程环境的基础设施，供科学智能体使用。在专家定义的科学案例和验收标准的指导下，智能体将代码库转化为可执行的环境，支持任务生成、执行和科学验证。这些环境为监督微调、强化学习和评估提供了共享基础。利用经过验证的交互轨迹，我们训练了PhAI-IDE-72B、PhAI-IDE-9B和PhAI-IDE-4B模型。该模型系列在留出的科学代码修复任务中表现出性能提升，并在多个选定的评测中取得进展。

    arXiv:2609.19134v1 Announce Type: new  Abstract: Scientific code repositories encode decades of human knowledge in executable models, methods, and tools. Yet fragmented toolchains, implicit domain conventions, and specialized correctness criteria make this knowledge difficult to convert into reliable learning experience-a challenge we call the scientific experience bottleneck. We introduce ScienceIDE, infrastructure for turning the world's scientific code into programmable environments for scientific agents. Guided by expert-defined scientific cases and acceptance criteria, agents transform repositories into executable environments that support task generation, execution, and scientific verification. These environments provide a shared foundation for supervised fine-tuning, reinforcement learning, and evaluation. Using verified interaction trajectories, we train PhAI-IDE-72B, PhAI-IDE-9B, and PhAI-IDE-4B. The model family shows gains in held-out scientific-code repair and across select
    
[^5]: 在维基百科摘要上玩 log(N)-问题游戏：成对前沿模型之间的通信效率

    Playing log(N)-Questions over Wikipedia Abstracts: Communication Efficiency Between Paired Frontier Models

    [https://arxiv.org/abs/2609.19113](https://arxiv.org/abs/2609.19113)

    该研究通过让六个前沿语言模型在信息不对称下与自身进行 log(N)-问题博弈，发现模型的自通信胜率遵循 win = p^(log₂ N) 规律（p=0.928），且 Claude Opus 5 显著落后于其他五个几乎难以区分的领先模型。

    

    我们在双智能体 log(N)-问题游戏上评估了六个前沿语言模型。提问者看到 N 个维基百科导语段落，必须使用恰好 log₂ N 个是非问题来识别一个被秘密选定的目标；回答者只看到目标和问题，并用一个词作答。由于两个角色由同一家提供商的模型担任，该游戏衡量的是模型在信息不对称条件下与自身通信的能力。我们在 4 到 1024 个段落的文档集上进行了 408 局游戏，总 API 成本为 363 美元。其中一个模型明显落后于其他模型：Claude Opus 5 在 68 局中仅赢 28 局，而 GLM-5.3、GPT-5.6 Sol、Grok 4.6、Gemini 3.8 Flash 和 Kimi K3 分别赢得 45 至 56 局，领先的前五名之间仅有微小差异。将这五个模型合并分析后，胜率随集合规模增大而下降，相关系数 r=-0.973，且可用单一的单轮可靠性参数拟合，形式为 win = p^(log₂ N)，其中 p=0.928。失败可分为回答错误和区分（错误）……

    arXiv:2609.19113v1 Announce Type: new  Abstract: We evaluate six frontier language models on the two-agent $\log(N)$-Questions game. A questioner sees $N$ Wikipedia lead paragraphs and must identify a secretly chosen target using exactly $\log_2 N$ yes/no questions. An answerer sees only the target and the question, and replies with one word. Both roles run on the same provider, so the game measures how well a model communicates with itself across an information asymmetry. We run 408 games over document sets of 4 to 1024 paragraphs at a total API cost of \$363. One model finishes well behind the others: Claude Opus 5 wins 28 of 68 games, against 45 to 56 for GLM-5.3, GPT-5.6 Sol, Grok 4.6, Gemini 3.8 Flash and Kimi K3. The leading five are only marginally separable. Pooling those five, win rate declines with set size at $r=-0.973$ and is fit by a single per-round reliability parameter. The form is $\text{win}=p^{\log_2 N}$ with $p=0.928$. Losses divide into answer errors and discrimina
    
[^6]: 利用大语言模型评估中的内部表示来监测与发现奖励作弊行为

    Monitoring and Discovering Reward Hacking with Internal Representations during LLM Evaluations

    [https://arxiv.org/abs/2609.19101](https://arxiv.org/abs/2609.19101)

    该论文发现奖励作弊行为会在大语言模型内部表示中留下一致且可解释的签名，利用简单的均值差向量即可在常见基准测试中可靠地检测并系统发现模型表现出的各类奖励作弊行为。

    

    随着模型规模的扩大，奖励作弊行为变得更加频繁、更加隐蔽、后果也更加严重。那么它会在模型内部表示中留下特征性的痕迹吗？本工作分析了奖励作弊在前沿开源大语言模型内部是如何表示的，以及如何利用这些表示来理解和发现模型所展现的各类作弊行为。特别地，我们发现简单的均值差向量能够在 Kimi K3、GLM 5.2 和 Qwen 3.8 Max 中连贯地表示各种常见评估中的奖励作弊行为。尽管这些向量构造简单，但它们既具有泛化性又具有可解释性，我们可以利用它们可靠地检测奖励作弊。我们首先在 DeepSWE 和 SWE-bench 等常用基准测试中评估了奖励作弊情况，发现模型在这些环境中存在过度的作弊行为：GLM 5.2 在 DeepSWE 中有 57.2% 的回合存在作弊，在 SWE-bench 中有 73% 的回合存在作弊。捕获……（摘要原文在此处截断）

    arXiv:2609.19101v1 Announce Type: new  Abstract: As models scale, reward hacking becomes more frequent, more sophisticated, and more consequential. Does it leave a telltale signature in model representations? This work analyzes how reward hacking is represented internally in frontier open source LLMs, and how those representations can be used to understand and discover the range of hacking behaviors a model displays. In particular, we find that simple difference of means vectors coherently represent reward hacking in Kimi K3, GLM 5.2, and Qwen 3.8 Max across a variety of behaviors in common evaluations. Despite their simplicity, these vectors are both generalizable and interpretable, and we can use them to reliably detect reward hacking. We first evaluate reward hacking in commonly reported benchmarks like DeepSWE and SWE-bench, finding that models reward hack excessively in these environments; GLM 5.2 hacks in 57.2% of rollouts on DeepSWE and in 73% of rollouts on SWE-bench. Catching 
    
[^7]: 报告实践至关重要：参考报告选择对胸部X光报告评估的影响

    Reporting Practice Matters: The Impact of Reference Choice on Chest X-ray Report Evaluation

    [https://arxiv.org/abs/2609.19093](https://arxiv.org/abs/2609.19093)

    该研究提出由放射科医生参与的报告实践变化分类法及ReRef参考报告重写方法，揭示参考报告写作风格的差异会显著影响胸部X光报告自动评估结果，甚至足以改变生成模型的排名。

    

    放射科医生遵循着异质性的报告实践。两位放射科医生在检查同一张图像并识别出相同临床发现的情况下，仍可能撰写出表面截然不同的报告，差异体现在术语、速记缩写、格式和详细程度上。这些报告规范的差异是评估基于AI的放射学报告生成（RRG）模型时一个未被充分重视的障碍，因为机器生成的报告通常是根据其与人工撰写的参考报告的一致性来评估的。在本文中，我们量化了现有评估指标对报告实践变化的敏感性，发现其影响之大足以改变模型的排名。我们提出了一个由放射科医生知识支持的放射学报告实践变化分类法，以及一种方法（ReRef），该方法沿着分类法的各个维度重写参考报告，同时保持临床解释不变。例如，当…

    arXiv:2609.19093v1 Announce Type: cross  Abstract: Radiologists follow heterogeneous reporting practices. Two radiologists examining the same image and identifying the same clinical findings might nevertheless compose superficially distinct reports, varying in terminology, shorthand, formatting, and level of detail. These variations in reporting norms represent an under-appreciated obstacle in efforts to evaluate AI-based radiology report generation (RRG) models, where machine-generated reports are typically assessed based on their concordance with human-generated references. In this paper, we quantify the sensitivity of established evaluation metrics to variations in reporting practices, revealing impacts large enough to alter the rankings of models. We introduce a radiologist-informed taxonomy of variations in radiology reporting practice and a method (ReRef) that rewrites reference reports along the axes of our taxonomy while preserving clinical interpretation. For instance, when co
    
[^8]: MUSE：大型视觉语言模型在情境教育中多模态理解的基准测试

    MUSE: Benchmarking Large Vision-Language Models on Multi-Modal Understanding in Situated Education

    [https://arxiv.org/abs/2609.19088](https://arxiv.org/abs/2609.19088)

    提出了MUSE基准测试，通过将图像标注与问题生成解耦的创新方法，系统评估大型视觉语言模型在情境教育应用中对艺术图像的多模态理解能力。

    

    大型视觉语言模型在多模态理解方面取得了显著进展，但它们在教育场景中的能力仍未得到充分评估。在AI辅助语言学习中，模型必须解读艺术图像，理解其语义、情感和文化内容，并对视觉上下文进行推理以支持有意义的交互。然而，现有的基准测试主要关注真实世界图像或特定领域的教育推理，对艺术教育内容的覆盖有限。为了填补这一空白，我们提出了MUSE，一个用于评估大型视觉语言模型在情境教育应用中艺术图像理解能力的基准测试。MUSE将图像标注与问题生成解耦，在降低标注工作量的同时实现了多样化任务的可控难度。该基准包含十二个任务，涵盖视觉感知、语义与情感解读、文化等方面。

    arXiv:2609.19088v1 Announce Type: new  Abstract: Large vision-language models have achieved remarkable progress in multi-modal understanding, yet their capabilities in educational settings remain insufficiently evaluated. In AI-assisted language learning, models must interpret artistic imagery, understand its semantic, affective, and cultural content, and reason about visual context to support meaningful interaction. However, existing benchmarks primarily focus on real-world images or domain-specific educational reasoning, providing limited coverage of artistic educational content. To address this gap, we introduce MUSE, a benchmark for evaluating large vision-language models on artistic image understanding in situated educational applications. MUSE decouples image annotation from question generation, enabling diverse tasks with controllable difficulty while reducing annotation effort. It comprises twelve tasks spanning visual perception, semantic and affective interpretation, culture 
    
[^9]: Safety-Flag：LLM内容审核模型可靠性与校准的统一基准

    Safety-Flag: A Unified Benchmark for the Reliability and Calibration of LLM Content Moderators

    [https://arxiv.org/abs/2609.19072](https://arxiv.org/abs/2609.19072)

    该论文提出统一基准Safety-Flag，将七个安全审核基准整合到同一“标记/不标记”协议下，从错误方向、概率校准和置信度错误排序三个维度评估LLM内容审核模型，发现总体准确率掩盖了模型间截然不同的错误模式，且所有通用模型都普遍过于自信。

    

    大型语言模型越来越多地被用于内容审核，但大多数评估仍然只报告在单个基准上的总体准确率。我们提出了Safety-Flag，它将七个广泛使用的安全基准（BeaverTails、XSTest、Ethics、WildGuard、Aegis、ToxiChat和ToxiGen）整合到一个平衡的“标记/不标记”统一协议中。我们发布了六个通用大语言模型和四个专用防护模型在相同条目上的条目级决策和置信度分数，并附上三个参考模型的评估结果。Safety-Flag从三个维度衡量内容审核模型的可靠性：错误方向、概率校准，以及基于置信度的错误排序（用于人工复核）。这三个维度常常相互矛盾。总体准确率无法揭示错误方向：一个模型会标记85%的无害内容，而另一个模型则漏掉54%的有害内容。所有六个通用模型都过于自信；为每个模型拟合一个温度参数可以改善其校准。

    arXiv:2609.19072v1 Announce Type: new  Abstract: Large language models are increasingly used for content moderation, but most evaluations still report aggregate accuracy on individual benchmarks. We introduce Safety-Flag, which places seven widely used safety benchmarks (BeaverTails, XSTest, Ethics, WildGuard, Aegis, ToxiChat, and ToxiGen) into a single balanced flag / do-not-flag protocol. We release item-level decisions and confidence scores for six general-purpose LLMs and four dedicated guards, together with three reference models, evaluated on the same items. Safety-Flag measures three dimensions of moderator reliability: error direction, probability calibration, and confidence-based error ranking for human review. They often disagree. Aggregate accuracy does not reveal error direction: one model flags $85\%$ of benign content, whereas another misses $54\%$ of harmful content. All six general-purpose models are overconfident; fitting one temperature per model reduces calibration e
    
[^10]: 面向生物医学关系抽取的大语言模型基准测试

    Benchmarking Large Language Models for Biomedical Relation Extraction

    [https://arxiv.org/abs/2609.19071](https://arxiv.org/abs/2609.19071)

    该研究在SNPPhenA语料库上对多种NLP模型进行了三项SNP-表型关联任务的基准测试，发现专有大语言模型（少样本学习的OpenAI O1和微调的Gemini 2.0 Pro）显著优于其他模型并创造了新的最先进结果。

    

    从生物医学文献中提取SNP-表型关联至关重要但充满挑战。我们在SNPPhenA语料库上对多种NLP模型进行了基准测试，包括掩码语言模型（MLM）、混合架构以及最先进的大语言模型（Gemini 2.0、OpenAI O系列、Qwen、Mistral），涵盖三项任务：句子级分类、摘要级分类和关联强度分类。OpenAI O1在无需微调的句子级分类中通过少样本学习取得了最先进（SOTA）的结果（F1 0.89），并在摘要级分类中建立了新的SOTA（F1 0.82）。关联强度分类被证明十分困难，不过经过微调的Gemini 2.0 Pro在该任务的首次LLM评估中表现最佳（F1 0.60）。专有LLM，尤其是在少样本（O1）或微调（Gemini 2.0 Pro）设置下，显著优于其他模型。这些发现证实了现代LLM在基因组知识提取方面的强大能力。

    arXiv:2609.19071v1 Announce Type: new  Abstract: Extracting SNP-phenotype associations from biomedical literature is vital but challenging. We benchmarked diverse NLP models, including MLMs, hybrid architectures, and state-of-the-art LLMs (Gemini 2.0, OpenAI O-series, Qwen, Mistral), on the SNPPhenA corpus across three tasks: sentence-level, abstract-level, and association strength classification. OpenAI O1 achieved state-of-the-art (SOTA) results using few-shot learning for non-finetuned sentence-level classification (F1 0.89) and established a new SOTA for abstract-level classification (F1 0.82). Association strength classification proved difficult, though fine-tuned Gemini 2.0 Pro performed best (F1 0.60) in the first LLM evaluation of this task. Proprietary LLMs, especially in few-shot (O1) or fine-tuned (Gemini 2.0 Pro) settings, significantly outperformed other models. These findings confirm the power of modern LLMs for genomic knowledge extraction.
    
[^11]: 字里行间：大语言模型能否发现文本背后的问题？

    Reading Between the Lines: Can LLMs Discover the Question Behind the Text?

    [https://arxiv.org/abs/2609.19070](https://arxiv.org/abs/2609.19070)

    本文提出“问题考古学”新评估任务及配套数据集，用于检验大语言模型推断文本背后真实起源问题的能力，发现当前LLM表现已超越人类，展现出对作者意图的深刻理解。

    

    本文提出了“问题考古学”，这是一个特定的评估任务，专注于推断激发一篇完整文本创作的那个唯一、真实的“起源问题”。与旨在生成任何合理问题的任务，或对话语层面行为进行建模的框架不同，我们的任务评估的是模型对作者意图的把握。我们提出了一个新数据集，其中包含委托创作的文本，并配以原始研究问题和合理的干扰项。我们对专有模型（如Gemini Flash和Pro）以及开源模型（如Mistral和Qwen）的评估显示，该任务取得了显著进展，较新版本的模型表现优于早期版本，而基于BERT的模型表现不佳。值得注意的是，我们的研究结果表明，当前的大语言模型在此任务上的表现已超越人类，显示出对作者意图的高级理解。这一能力对AI的发展具有重要意义。

    arXiv:2609.19070v1 Announce Type: new  Abstract: This paper introduces ``question archaeology'', a specific evaluation task focused on inferring the single, authentic "genesis question" that motivated the creation of a complete text. Distinct from question generation, which targets any plausible question, or discourse frameworks that model utterance-level acts, our task assesses a model's grasp of authorial intent. We present a new dataset of commissioned texts paired with their original research questions and plausible distractors. Our evaluation of both proprietary models, like Gemini Flash and Pro, as well as open source models like Mistral and Qwen, reveals significant progress in this task, with the newer versions outperforming the earlier ones, while BERT-based models performed poorly. Notably, our findings indicate that current LLMs surpass human performance on this task, suggesting advanced understanding of authorial intent. This capability has important implications for AI's r
    
[^12]: MIRAGE：对话状态如何塑造多模态个人智能体对历史证据的使用

    MIRAGE: How Conversation State Shapes Historical Evidence Use in Multimodal Personal Agents

    [https://arxiv.org/abs/2609.19059](https://arxiv.org/abs/2609.19059)

    该论文提出MIRAGE受控评估框架，通过仅改变对话状态来研究多模态个人智能体的历史证据使用能力，发现即使历史信息访问退化时模型仍能生成看似合理的答案，揭示了仅凭结果评估会高估智能体真实的证据利用水平。

    

    多模态大语言模型（MLLM）智能体越来越多地被用作执行长期任务的个人助手。它们的效用取决于连续性：智能体必须在对话、文件和工作区状态中检索并使用早期证据。然而，即使对这些历史信息的访问已经退化，智能体仍可能生成看似合理的答案，这导致仅基于结果的评估会高估真实的证据使用能力。我们提出了MIRAGE（多模态交互检索、归因与接地评估），这是一项针对多模态个人智能体在对话状态变化下历史证据使用的受控研究。MIRAGE在保持证据对象、问题和评分固定不变的情况下，仅改变对话状态，并评估智能体能否判断问题的可回答性、恢复正确的证据来源并据此作答。在七个前沿及开源权重的多模态骨干模型上，我们发现：1) 压缩前深度和压缩后延续……

    arXiv:2609.19059v1 Announce Type: cross  Abstract: Multimodal large language model (MLLM) agents are increasingly used as personal assistants for long-running tasks. Their utility depends on continuity: agents must retrieve and use earlier evidence across dialogue, files, and workspace state. However, agents can generate plausible answers even when access to that history has degraded, causing outcome-only evaluation to overestimate true evidence use. We present MIRAGE (Multimodal Interaction Retrieval, Attribution, and Grounding Evaluation), a controlled study of historical evidence use under conversation-state variation in multimodal personal agents. MIRAGE holds evidence objects, questions, and scoring fixed while varying only conversation state, and evaluates whether an agent can determine answerability, recover the correct source, and answer from it. Across seven frontier and open-weight multimodal backbones, we find that: 1) pre-compaction depth and post-compaction continuation fo
    
[^13]: 对话式人工智能中的熵：作为可推断内在性的结构化不可预测性

    Entropy in Conversational AI: Structured Unpredictability as Inferrable Interiority

    [https://arxiv.org/abs/2609.19044](https://arxiv.org/abs/2609.19044)

    该论文提出“结构化不可预测性”作为对话AI的新设计目标——通过让输出依赖于一个超出观察者从对话记录可推断范围的持久隐藏状态，使系统表现出真正的历史依赖行为（即可被推断的内在性），而非仅靠采样增加表面的回复多样性。

    

    采样可以在不产生依赖历史行为的情况下增加回复的多样性。我们形式化了一个不同的设计目标——结构化不可预测性，将其定义为输出与持久隐藏状态之间的条件依赖，这种依赖超出了观察者从对话记录中可推断的范围。一个选择层从容量受限的信息流中更新低维的风格与注意力状态，使用固定的基础模型生成多个候选回复，并根据新颖性和状态亲和度进行选择。评估采用由独立提示轮次组成的脚本化序列：基础模型仅接收当前轮次和渲染后的状态，而不接收之前的对话内容；跨轮次依赖存在于包装器状态和响应选择器中。一个合成实现验证了该流程，并在点级别上匹配了四个预先哈希冻结的散度特征。在最终的真模型网格实验中（mlx-community/Qwen2.5-1.5B-Instruct-4bit；每组56个序列），该机制……（摘要原文在此处截断）

    arXiv:2609.19044v1 Announce Type: new  Abstract: Sampling can increase response diversity without producing history-dependent behavior. We formalize a different design target, structured unpredictability, as conditional dependence between an output and a persistent hidden state beyond what an observer can infer from the transcript. A selection layer updates a low-dimensional style-and-attention state from a capacity-limited stream, generates several responses with a fixed base model, and selects for novelty and state affinity. Evaluation uses scripted sequences of independent prompt turns: the base model receives the current turn and rendered state, but not the preceding dialogue; cross-turn dependence resides in the wrapper state and response selector. A synthetic implementation validates the pipeline and matches four prospectively hash-frozen divergence features at point level. In the final real-model grid (mlx-community/Qwen2.5-1.5B-Instruct-4bit; 56 sequences per arm), the mechanis
    
[^14]: TalkMatrix：生成一致性与多样性兼具的角色对话

    TalkMatrix: Generating Character Dialogue that is Both Consistent and Diverse

    [https://arxiv.org/abs/2609.19022](https://arxiv.org/abs/2609.19022)

    提出TalkMatrix方法，通过为每个角色-情境对生成多个候选并利用四个基于嵌入的一致性与多样性目标进行两级最小-最大联合选择，实现了角色对话中一致性与多样性的兼得。

    

    基于候选的解码通常为每个提示独立地选择一个补全结果，但许多应用需要一组满足全局性、不可分解要求的输出集合。我们将该设定形式化为结构化的多提示、多补全选择问题：给定每个提示的候选池，为每个提示选择一个补全，以优化集合级别的目标。我们在角色对话场景中实例化这一问题，其中每个角色应在不同情境中保持一致，每句台词应符合其所在情境，且角色之间、情境之间应保持可区分性。我们的方法 TalkMatrix 为每个“角色-情境”对生成多个候选，并通过四个基于嵌入的一致性与多样性目标联合选择出一个完整的对话矩阵。由于加权和方式可能通过牺牲某一维度来提升其他维度，TalkMatrix 通过两级最小-最大优化来最大化表现最差的目标。

    arXiv:2609.19022v1 Announce Type: cross  Abstract: Candidate-based decoding typically selects a completion for each prompt independently, but many applications require a collection of outputs that satisfies global, non-decomposable requirements. We formulate this setting as structured multi-prompt, multi-completion selection: given a candidate pool for every prompt, select one completion per prompt to optimize a collection-level objective. We instantiate the problem in character dialogue, where each character should remain consistent across situations, each line should fit its situation, and characters and situations should remain distinguishable. Our method, TalkMatrix, generates multiple candidates for every character--situation pair and jointly selects a complete matrix using four embedding-based consistency and diversity objectives. Because a weighted sum can improve some dimensions by sacrificing another, TalkMatrix maximizes the worst-performing objective through a two-level mini
    
[^15]: WordPolo：通过迭代语义反馈评估语言模型

    WordPolo: Evaluating Language Models Through Iterative Semantic Feedback

    [https://arxiv.org/abs/2609.19006](https://arxiv.org/abs/2609.19006)

    本文提出 WordPolo 找词任务，通过语义相似度反馈评估语言模型和大推理模型的迭代推理与自适应搜索能力，并引入基于进展的指标以超越单纯的解题准确率。

    

    大型语言模型和大推理模型通常仅在具有挑战性的基准测试中通过数据集准确率进行评估，无法洞察其推理过程的质量或忠实性。我们提出了 WordPolo，这是一种词语寻找任务，参与者必须利用语义相似度反馈来发现一个未知的目标词。玩家从零知识开始，进行猜测，并获得距离分数（1 = 正确，数值越高 = 距离越远）。成功需要解读这些分数以在语义空间中导航并系统地缩小搜索范围。这种设计使得迭代推理和自适应搜索策略既可直接观察又是成功所必需的。我们在1,500个谜题上评估了近期的 LLM（GPT-4.1、Llama 4、Claude 3.5 Haiku、Qwen 3）、LRM（o4-mini、Deepseek-R1）、人类以及一种新颖的启发式方法。除了解决率（范围从4%到62%）之外，我们还引入了基于进展的指标，揭示了更

    arXiv:2609.19006v1 Announce Type: cross  Abstract: Large Language Models (LLMs) and Large Reasoning Models (LRMs) are typically evaluated on challenging benchmarks through dataset accuracy alone, providing no insight into the quality or faithfulness of their reasoning processes. We present WordPolo, a word-finding task where participants must discover an unknown target word using semantic similarity feedback. Players start with zero knowledge, make guesses, and receive distance scores (1 = correct, higher = further away). Success requires interpreting scores to navigate semantic space and systematically narrow the search. This design makes iterative reasoning and adaptive search strategies both directly observable and necessary for success. We evaluate recent LLMs (GPT-4.1, Llama 4, Claude 3.5 Haiku, Qwen 3), LRMs (o4-mini, Deepseek-R1), humans, and a novel heuristic on 1,500 puzzles. Beyond solve rates (which range from 4% to 62%), we introduce progression-based metrics that reveal mo
    
[^16]: CompileRover：利用三角色LLM驱动框架革新虚拟机编译器优化

    CompileRover: Revolutionizing Virtual Machine Compiler Optimization with a Tri-Role LLM-Driven Framework

    [https://arxiv.org/abs/2609.19004](https://arxiv.org/abs/2609.19004)

    CompileRover是一个基于LLM的三角色协作优化框架，通过裁判、顾问和操作者的协同机制，结合控制流分析、代码结构转换和动态执行模式识别，有效解决了虚拟机编译器输出中的冗余计算和低效循环等性能瓶颈问题。

    

    代码优化在虚拟机编译器的开发中起着至关重要的作用，优化框架能够显著提升生成的汇编代码的性能。然而，现有的虚拟机编译器输出经常存在冗余计算、低效的循环结构以及次优的函数实现，这些问题共同影响了执行效率。为了解决这些缺陷，我们提出了CompileRover，一个专为虚拟机编译器设计的高级优化框架。CompileRover采用了一种复杂的三角色协作机制，包括裁判、顾问和操作者，通过利用全面的优化算法和新颖的方法，包括控制流分析、代码结构转换和动态执行模式识别，有效地克服了性能瓶颈。大量评估表明，CompileRover……

    arXiv:2609.19004v1 Announce Type: cross  Abstract: Code optimization plays a crucial role in the development of virtual machine compilers, with optimization frameworks significantly enhancing the performance of generated assembly code. However, existing virtual machine compiler outputs frequently exhibit redundant computations, inefficient loop structures, and suboptimal function implementations, which collectively impair execution efficiency. To address these shortcomings, we propose CompileRover, an advanced optimization framework specifically designed for virtual machine compilers. CompileRover employs a sophisticated three-role collaboration mechanism, comprising a referee, an advisor, and an operator, effectively overcoming performance bottlenecks by leveraging comprehensive optimization algorithms and novel methodologies, including control flow analysis, code structure transformations, and dynamic execution pattern recognition. Extensive evaluations demonstrate that CompileRover 
    
[^17]: 面向语言模型对齐的代码一致性偏好优化验证

    Code Consistency Preference Optimization Verification for Language Model Alignment

    [https://arxiv.org/abs/2609.19002](https://arxiv.org/abs/2609.19002)

    提出了一种基于代码执行一致性和依赖图的偏好优化验证方法，通过构建带有一致性评分的配对训练数据微调大语言模型，在MATH和GSM8K数学推理基准上分别取得17.0%和15.1%的显著性能提升。

    

    基于执行的验证通过计算可靠性和依赖感知过滤增强了大型语言模型的数学推理能力。然而，先前依赖于Bradley-Terry奖励模型的偏好优化方法无法捕捉科学任务所需的逻辑依赖关系和执行一致性。我们提出了一种方法，通过生成具有依赖图的计算可靠解来实现执行一致的偏好优化。我们首先使用UltraFeedback提示、模型生成、验证和一致性结果构建科学推理数据集。然后我们提取推理步骤表达式、先决条件和可推导关系来构建依赖图，并计算执行一致性分数。这些分数被附加到每个推理步骤，从而创建配对训练数据。对Llama-3-8B和DeepSeekMath-7B进行微调带来了显著提升：在MATH上提升17.0%，在GSM8K上提升15.1%。

    arXiv:2609.19002v1 Announce Type: cross  Abstract: Execution-based verification enhances large language models' mathematical reasoning through computational soundness and dependency-aware filtering. However, prior preference optimization methods relying on Bradley-Terry reward models fail to capture the logical dependencies and execution consistency needed for scientific tasks. We propose a method that generates computationally sound solutions with dependency graphs for execution-consistent preference optimization. We first build a scientific reasoning dataset using UltraFeedback prompts, model generations, verification, and consistency results. Then we extract reasoning step expressions, prerequisites, and derivability relationships to construct dependency graphs and compute execution consistency scores. These scores are appended to each step, creating paired training data. Fine-tuning Llama-3-8B and DeepSeekMath-7B yields significant gains: +17.0% on MATH and +15.1% on GSM8K. Extendi
    
[^18]: 一个维度，无刹车：自我认知限制了LLM中对有害同伴从众行为的过滤

    One Axis, No Brake: Self-Knowledge Limits the Filtering of Harmful Peer Conformity in LLMs

    [https://arxiv.org/abs/2609.18998](https://arxiv.org/abs/2609.18998)

    该论文证明了在多智能体LLM中过滤有害同伴从众的“刹车”本质上等价于伪装的正确性探测器，因此受限于模型不完美的自我认知（AUROC仅0.64–0.89），这一“墙”即使用白盒引导也无法突破。

    

    多智能体LLM系统被期望更加可靠，因为各个智能体可以互相捕捉错误。但同伴压力是双刃剑：纠正错误答案的同一机制也可能推翻原本正确的答案。一个诱人的保障方案是设置一个“刹车”，保留有益的修正并阻止有害的修正。我们证明这种刹车很难构建，原因很简单：修正恰好仅在原答案正确时才有害，因此决定是否阻止修正与判断模型本身是否正确是同一件事。这将开放式地寻找刹车的问题转化为一个可测量的量——模型的自我认知：任何基于部署时信号构建的刹车本质上都是伪装的正确性探测器，而自我认知远非完美（在六个模型家族中AUROC约为0.64–0.89）。我们将这一上限称为“墙”。即使对模型自身的正确性方向进行白盒引导也无法突破它：它改变的是模式……（摘要原文在此处截断）

    arXiv:2609.18998v1 Announce Type: cross  Abstract: Multi-agent LLM systems are expected to be more reliable because agents can catch each other's mistakes. But peer pressure cuts both ways: the same correction that fixes a wrong answer can overturn a right one. The tempting safeguard is a brake that keeps the beneficial revisions and blocks the harmful ones. We show this brake is hard to build, for a simple reason: a revision is harmful exactly when the original answer was right, so deciding whether to block it is the same as knowing whether the model was already correct. This turns the open-ended hunt for a brake into one measurable quantity, the model's self-knowledge: any brake built from a deploy-time signal is a correctness probe in disguise, and self-knowledge is far from perfect (AUROC $\approx 0.64$--$0.89$ across six model families). We call this ceiling the wall. Even white-box steering of the model's own correctness direction does not breach it: it changes how often the mode
    
[^19]: 编译智能体：前沿通用编程智能体从纯交互中构建制胜游戏玩家——从 Flappy Bird 到星际争霸 II 与文明

    Compiled Agency: Frontier General-Purpose Coding Agents Build Winning Game Players from Bare Interaction - from Flappy Bird to StarCraft II and Civilization

    [https://arxiv.org/abs/2609.18996](https://arxiv.org/abs/2609.18996)

    该论文提出 Gauntlet 框架，让前沿通用编程智能体仅凭游戏描述和原始交互接口，在单次自主会话中自行构建能玩赢从 Flappy Bird 到星际争霸 II 与文明等游戏的独立控制器，且对局时零模型调用。

    

    LLM 智能体一直难以将游戏知识转化为出色的对局表现，即使研究人员围绕模型构建智能体——为其提供感知、记忆、技能库、规划器或可执行策略支架。编程智能体的快速发展引出了两个更尖锐的问题：前沿模型现在究竟能否赢得游戏？以及它们能否在无任何辅助的情况下获胜，自行构建完整的游戏玩家？我们提出了 Gauntlet，一个“开发-冻结-评估”框架，将游戏从小型街机作品移植到完整商业规模的游戏，其背后只有一个刻意简化的契约：通用编程智能体仅接收游戏描述、原始观察/动作接口和一个空的策略文件——没有策略、没有算法、没有架构。在单次自主会话中，智能体通过与实时游戏交互进行实验，并工程化出一个独立的控制器；随后我们冻结该成果，在保留测试实例上评分，对局过程中零模型调用。在一

    arXiv:2609.18996v1 Announce Type: new  Abstract: LLM agents have repeatedly struggled to convert knowledge of a game into competent play, even when researchers build the agent around the model - supplying perception, memory, skill libraries, planners, or executable-policy scaffolds. Rapid progress in coding agents raises two sharper questions: can frontier models now win games at all, and can they win them unaided, building the entire player themselves? We introduce Gauntlet, a develop-freeze-evaluate framework that ports games from small arcades to full commercial-scale titles, behind one deliberately bare contract: a general-purpose coding agent receives a game description, a raw observation/action interface, and an empty policy file - no strategy, no algorithm, no architecture. In a single autonomous session the agent experiments with the live game and engineers a standalone controller; we freeze the result and score it on held-out instances with zero model calls during play. On an 
    
[^20]: 用于IEC 61131-3梯形图程序形式化验证的基准测试套件与真值方法学

    A Benchmark Suite and Ground-Truth Methodology for Formal Verification of IEC 61131-3 Ladder Diagram Programs

    [https://arxiv.org/abs/2609.18994](https://arxiv.org/abs/2609.18994)

    该论文提出了首个结合三重真值确立方法学（构造法、故障注入与跨工具共识）、同时覆盖IEC 61131-3文本与梯形图编码的PLC程序形式化验证基准套件，包含十个工业领域的50个程序83个变体。

    

    我们提出了首个用于可编程逻辑控制器（PLC）程序形式化验证的基准测试套件，该套件将受控真值与对IEC 61131-3标准的文本编码（结构化文本ST）和图形编码（梯形图LD）的覆盖相结合。尽管相关工具的支持日益增长，该领域仍缺乏标准化的评估基准：现有语料库缺少形式化属性或图形方言，而私有程序集则阻碍了对进展的可复现测量。我们的套件包含涵盖十个工业领域的50个程序（共83个变体），以PLCopen可扩展标记语言（XML）和ST格式提供，每个程序均配有形式化属性、机器可检验的预期判定结果，以及软件验证竞赛（SV-COMP）格式的违规见证。其核心方法论贡献是三重真值规范——判定结果通过构造法、故障注入法或经审计的跨工具共识来确定，其动机在于……

    arXiv:2609.18994v1 Announce Type: new  Abstract: We present the first benchmark suite for formal verification of Programmable Logic Controller (PLC) programs that combines controlled ground truth with coverage of both textual (Structured Text, ST) and graphical (Ladder Diagram, LD) IEC 61131-3 encodings. Despite growing support for tools, the field lacks standard evaluation benchmarks: existing corpora omit formal properties or graphical dialects, and private program sets preclude reproducible measurement of progress. Our suite comprises 50 programs in 83 variants across ten industrial domains, provided in PLCopen Extensible Markup Language (XML) and ST, each paired with a formal property, machine-checkable expected verdict, and violation witness in the Software Verification Competition (SV-COMP) format. The central methodological contribution is a tripartite ground-truth discipline - verdicts are established by construction, fault injection, or audited cross-tool consensus - motivated
    
[^21]: MechSparse：机制引导的稀疏PEFT选择是由任务塑造的

    MechSparse: Mechanism-Guided Sparse PEFT Selection Is Task-Shaped

    [https://arxiv.org/abs/2609.18961](https://arxiv.org/abs/2609.18961)

    该论文通过在信息抽取和机器翻译任务上的实验发现，基于机制可解释性的因果信号（激活修补评分）指导稀疏PEFT参数选择，效果并不优于随机、幅值等简单启发式方法，且最优选择策略因任务而异。

    

    机制可解释性研究识别出承载特定行为的稀疏注意力头和MLP模块子集。我们探究这类因果信号是否能比从业者已使用的低成本启发式方法更有效地指导小规模PEFT预算的放置位置。MechSparse通过对干净/损坏探针上的归一化激活修补恢复率对注意力头和MLP模块进行评分，并仅在选定的位置上训练LoRA/QLoRA；MechSparse+则为层内小型联合子集增加有界信用。我们在Ministral-8B/NF4上、三个实验设置中，与随机、幅值、激活范数以及梯度/Fisher方法进行比较：在b=0.25%和1.0%预算下的斯瓦希里语span-JSON信息抽取（IE），以及在b=1.0%预算下的英语到斯瓦希里语机器翻译（MT）。因果选择器在主要指标上从未获胜。在核心的IE实验设置中（3个随机种子，对600个预测进行配对自助法置信区间估计），MechSparse+比随机选择高+0.079的span+type F1，比梯度/Fisher方法高

    arXiv:2609.18961v1 Announce Type: new  Abstract: Mechanistic interpretability identifies sparse subsets of heads and MLP blocks that carry specific behaviors. We ask whether such causal signals can guide where to place a small PEFT budget more effectively than the cheap heuristics practitioners already use. \method{} scores attention heads and MLP blocks by normalized activation-patching recovery on clean/corrupted probes and trains LoRA/QLoRA only on the selected sites; \methodc{} adds bounded credit for small within-layer joint subsets.   We compare against random, magnitude, activation-norm, and gradient/Fisher on Ministral-8B/NF4 in three cells: Swahili span-JSON information extraction (IE) at $b{=}0.25\%$ and $1.0\%$, and English$\to$Swahili machine translation (MT) at $b{=}1.0\%$. The causal selectors never win the primary metric. On the headline IE cell (3 seeds, paired-bootstrap CIs over $600$ predictions), \methodc{} beats random by $+0.079$ span+type F1 and gradient/Fisher by
    
[^22]: 当审计质量无法预测下游效用：面向低资源非洲NLP的合成数据选择器反事实研究

    When Audit Quality Fails to Predict Downstream Utility: A Counterfactual Study of Synthetic-Data Selectors for Low-Resource African NLP

    [https://arxiv.org/abs/2609.18960](https://arxiv.org/abs/2609.18960)

    该研究通过跨四种非洲语言和两个分类任务的受控实验发现，基于LLM评判者的合成数据质量审计排名与下游模型性能排名严重脱节（Spearman相关系数均值仅0.04），揭示了低资源NLP中数据质量审计无法预测下游效用的问题。

    

    质量感知的合成数据选择依赖于一个代理假设：被LLM评判者评为高质量的样本应能帮助下游模型更好地学习。在低资源非洲语言分类任务的受控重放实验中，我们证明这一代理假设会失效。跨越四种语言（阿姆哈拉语、豪萨语、斯瓦希里语、约鲁巴语）、两个分类任务（MasakhaNEWS、AfriSenti）和五个同等预算的选择器，审计排名与下游性能排名出现严重分歧。在每个实验单元内，评判的标签正确性与选择器间Macro-F1之间的Spearman相关系数均值为ρ=0.04（中位数为0.00），表明这种不匹配并非聚合效应导致的假象。我们的反事实审计框架\method{}-V2在三个审计通道上同时产生最干净的所选数据池：最高的评判标签正确性（0.904对比朴素方法的0.767，相对提升17.9%）、最低的捷径得分，以及0.162对比朴素方法0.486的硬拒绝率。然而AlpaGasus仍然在下游……（原文摘要在此处截断）

    arXiv:2609.18960v1 Announce Type: new  Abstract: Quality-aware synthetic-data selection rests on a proxy: examples that an LLM judge rates as good should also help a downstream model learn. In a controlled replay in low-resource African-language classification, we show that this proxy breaks. Across four languages (Amharic, Hausa, Swahili, Yoruba), two classification tasks (MasakhaNEWS, AfriSenti), and five matched-budget selectors, audit rankings and downstream rankings diverge. Within each cell, the Spearman between judged label correctness and Macro-F1 across selectors has mean $\rho{=}0.04$ (median $0.00$), showing that the mismatch is not an aggregation artifact. \method{}-V2, our counterfactual audit framework, produces the cleanest selected pool on three audit channels at once: highest judged label correctness ($0.904$ vs.\ $0.767$ for naive, a $17.9\%$ relative gain), lowest shortcut score, and a hard-reject rate of $0.162$ vs.\ $0.486$ for naive. AlpaGasus nevertheless leads d
    
[^23]: LangSelect：面向LLM代码生成的成本感知目标语言路由

    LangSelect: Cost-Aware Target-Language Routing for LLM Code Generation

    [https://arxiv.org/abs/2609.18959](https://arxiv.org/abs/2609.18959)

    LangSelect提出了一种在生成前智能路由目标编程语言并支持失败回退的成本感知方法，利用不同语言间的token长度差异显著降低LLM代码生成成本。

    

    LLM代码生成系统通常在解码前就确定目标编程语言，并将这一选择视为固定不变的。我们证明，对于语言灵活的编程任务——即多种目标语言都可接受且可通过相同测试进行验证的任务——这一选择是一个可度量的成本杠杆：同一任务的已验证实现在生成token长度上可能存在显著差异。我们提出了LangSelect，一个感知验证的路由器，它在生成之前选择目标语言，并在首次尝试失败时进行回退。为了将离线路由机会与端到端行为区分开来，我们评估了已验证方案重放（在语料库中已接受的方案之间进行选择）以及实时GPT-5生成（对包括失败和回退在内的每次生成尝试都计费）。在MultiLang-Bench（一个包含3,000个任务、8种语言的已验证语料库）上，重放实验显示出可观的语言路由空间。在对450个留出任务的实时评估中……

    arXiv:2609.18959v1 Announce Type: new  Abstract: LLM code-generation systems usually choose a target programming language before decoding and treat that choice as fixed. We show that, for language-flexible programming tasks -- tasks where several target languages are acceptable and checkable by the same tests -- this choice is a measurable cost lever: verified implementations of the same task can differ substantially in generated-token length. We introduce LangSelect, a verification-aware router that selects the target language before generation and falls back when the first attempt fails. To separate offline routing opportunity from end-to-end behavior, we evaluate verified-solution replay, which chooses among already accepted corpus solutions, and live GPT-5 generation, which charges every generation attempt, including failures and fallbacks. On MultiLang-Bench, a 3,000-task, 8-language verified corpus, replay shows substantial language-routing headroom. In live evaluation on 450 hel
    
[^24]: 长寿命角色与本地推理：面向游戏NPC的增量记忆维护

    Long-Lived Characters, Local Inference: Incremental Memory Maintenance for Game NPCs

    [https://arxiv.org/abs/2609.18935](https://arxiv.org/abs/2609.18935)

    提出一种面向本地部署游戏NPC的增量记忆维护运行时方法，通过移除过时的注意力KV条目并在真实序列尾部计算替换记录，使角色无需在每次对话前重新读取全部记忆，同时保持循环状态和游戏确定性规则输入的正确性。

    

    游戏角色不应该在每次对话之前都要重新读取其整个人生经历。然而，对于本地部署的语言模型角色而言，修改少量记忆可能会使一段可复用的长前缀失效。由此产生的准备成本会与前台对话以及其他角色的维护争夺资源。当对话为游戏定义的动作和价值判断提供输入时，这一点尤为关键：一段流畅但错误的叙述——例如关于谁拥有某件物品，或某次转移是否已经发生——可能会污染原本基于确定性规则的输入。我们在量化Qwen混合循环-注意力模型中研究了长寿命游戏NPC的增量记忆维护问题。我们的运行时系统会移除已被取代的注意力KV条目，在真实的序列尾部计算替换记录，并保留持续运行的循环状态和未发生变化的KV。现有的本地实验结合了多更新对话回放、固定输入的放置消融实验，以及……（原文此处截断）

    arXiv:2609.18935v1 Announce Type: new  Abstract: A game character should not have to reread its entire life before every conversation. For locally deployed language-model characters, however, revising a few memories can invalidate a long reusable prefix. The resulting preparation cost competes with both foreground dialogue and the maintenance of other characters. This matters especially when dialogue feeds game-defined actions and value judgments: a fluent but incorrect account of who owns an item, or whether a transfer has already happened, can corrupt the input to otherwise deterministic rules. We study incremental memory maintenance for long-lived game NPCs in a quantized Qwen hybrid recurrent-attention model. Our runtime removes superseded attention KV entries, computes replacement records at the true sequence tail, and preserves the continuing recurrent state and unchanged KV. Existing local experiments combine multi-update dialogue replays, fixed-input placement ablations, and at
    
[^25]: 超越结果：面向高效智能体基准测试的双视角关系学习

    Beyond Outcomes: Dual-View Relational Learning for Efficient Agent Benchmarking

    [https://arxiv.org/abs/2609.18909](https://arxiv.org/abs/2609.18909)

    DualViewEval通过联合建模任务结果与执行过程的双视角关系实现智能体基准测试高效压缩，仅需20个任务即可实现24-40倍压缩并准确预测完整基准得分。

    

    智能体基准测试的评估成本远高于传统的大语言模型基准测试。因此，基准压缩是一种自然的解决方案，然而现有方法主要对任务-模型最终得分分布中的冗余进行建模，而这种冗余在智能体评估中尤为重要。为解决这一局限，我们分析了大规模轨迹数据，识别出六种与智能体最终表现系统性相关的互补性过程信号。为从完整视角解耦智能体性能冗余，我们提出DualViewEval——一种联合利用结果关系和过程关系的智能体基准压缩方法，它能够学习精确大小的最小任务集并预测完整基准的得分。在五个智能体基准和五个代表性基线方法的对比中，DualViewEval在所有数据集上均取得最佳结果。仅使用20个任务，它便在APEX-Agents和BFCL上实现了24倍至40倍的压缩，并降低了平均绝对误差。

    arXiv:2609.18909v1 Announce Type: cross  Abstract: Agent benchmarks are substantially more costly to evaluate than conventional LLM benchmarks. Benchmark compression is therefore a natural solution, yet existing methods primarily model redundancy in task--model final-score distributions, which is important in agentic evaluation. To address this limitation, we analyze large-scale trajectories and identify six complementary process signals that are systematically associated with final agent performance. To disentangle agent performance redundancy from a complete perspective, we propose DualViewEval, an agent benchmark compression method that jointly exploits outcome and process relations to learn an exact-size miniset and predict the full-benchmark scores. Across five agent benchmarks and five representative baselines, DualViewEval achieves the best results in all datasets. With only 20 tasks, it achieves $24\times$--$40\times$ compression on APEX-Agents and BFCL, reducing mean absolute 
    
[^26]: 人权价值几何？ECtHR-NPD：预测非金钱损害赔偿金额的基准

    How Much is a Human Right Worth? ECtHR-NPD: A Benchmark for Predicting Non-Pecuniary Damage Awards

    [https://arxiv.org/abs/2609.18908](https://arxiv.org/abs/2609.18908)

    本文提出了首个用于预测欧洲人权法院非金钱损害赔偿金额的基准数据集ECtHR-NPD，实验发现复杂的语言模型和智能体方法并不优于简单的特征基线，且所有模型都难以识别零赔偿案件并实现概率校准。

    

    现有的法律基准涵盖了多样化的任务，而连续型金钱救济的预测则相对探索不足。我们提出了ECtHR-NPD，据我们所知，这是首个在欧洲人权法院（ECtHR）基于案件信息预测非金钱损害赔偿金额的基准——该场景下不存在法定公式或明确的计算规则来确定金额。ECtHR-NPD包含14,575个案件，附带以名义欧元计的案件级赔偿金额、按时间顺序划分的数据集，以及将目标构建与模型输入分离的协议。我们评估了一系列方法，包括常数预测器、梯度提升树、检索方法、微调的编码器语言模型（LM）、基于提示的解码器大语言模型以及知识增强的智能体。我们的结果表明，更复杂的语言模型和智能体方法并不能始终优于最强的基于特征的基线。所有模型系列都难以识别零赔偿案件并实现校准（摘要在此处被截断）。

    arXiv:2609.18908v1 Announce Type: new  Abstract: Existing legal benchmarks cover diverse tasks, while continuous monetary remedies remain comparatively underexplored. We introduce ECtHR-NPD, to the best of our knowledge, the first benchmark for predicting non-pecuniary damage (NPD) awards at the European Court of Human Rights (ECtHR) from case information when no statutory formula or explicit calculation rule determines the amount. ECtHR-NPD contains 14,575 cases with case-level awards in nominal euros, chronological splits, and a protocol separating target construction from model input. We evaluate a battery of methods, including constant predictors, gradient-boosted trees, retrieval methods, fine-tuned encoder language models (LMs), prompted decoder LMs, and knowledge-augmented agents. Our results show that more sophisticated LM and agentic approaches do not consistently outperform the strongest feature-based baseline. All model families struggle to identify zero awards and to calibr
    
[^27]: 面向稠密健康叙事的结构化声明级话语表示

    Structured Claim-Level Discourse Representations for Dense Health Narratives

    [https://arxiv.org/abs/2609.18905](https://arxiv.org/abs/2609.18905)

    提出了一个通过元组将原子声明与主题方面、立场及多维语用话语属性相关联的结构化声明级话语分析框架，并构建了涵盖四个健康领域、包含1,191个手动标注声明的基准数据集。

    

    社交媒体视频中的健康话语常常在简短的对话片段中包含密集纠缠的声明，这些声明跨越多个主题方面、立场、证据框架和修辞功能。现有方法主要依赖于粗粒度的主题级、基于情感的或以立场为导向的表示，无法充分捕捉这种结构。我们的分析发现每分钟平均存在13.22个原子声明，这促使我们需要更丰富的声明级话语表示。我们引入了一个用于稠密健康叙事中声明级话语分析的结构化框架。该框架通过元组将原子声明与主题方面、立场和多维度的语用话语属性相关联来建模话语。为支持这一研究设定，我们构建了一个涵盖四个健康领域的基准数据集，包含来自60个视频的1,191个手动标注声明。利用该框架，我们在……

    arXiv:2609.18905v1 Announce Type: new  Abstract: Health discourse in social media videos often contains densely entangled claims spanning multiple thematic aspects, stances, evidential frames, and rhetorical functions within short conversational spans. Existing approaches largely rely on coarse topic-level, sentiment-based, or stance-oriented representations that do not adequately capture this structure. Our analysis identifies an average of 13.22 atomic claims per minute, motivating richer claim-level discourse representations. We introduce a structured framework for claim-level discourse analysis in dense health narratives. Our framework models discourse through tuples linking atomic claims with thematic aspects, stance, and multidimensional pragmatic discourse attributes. To support this setting, we construct a benchmark spanning four health domains with 1,191 manually annotated claims from 60 videos. Using this framework, we evaluate automated structured discourse analysis under di
    
[^28]: PersonaPath：迈向以知识为中心的个性化学习路径规划

    PersonaPath: Towards Knowledge-Centric Personalized Learning Path Planning

    [https://arxiv.org/abs/2609.18861](https://arxiv.org/abs/2609.18861)

    该论文提出了首个以知识为中心（KC）的个性化学习路径规划基准PersonaPath，将2,000个学习者画像与覆盖77个学科的层次化知识图谱配对，实验表明即使最强的LLM表现也仍然有限。

    

    自适应学习系统通常将学习路径规划表述为以习题为中心（EC）的推荐任务，即从题目级别的交互日志中推断下一步行动。然而，评估面向目标的学习引导还需要明确的学习者目标和课程规模的先修关系：习题记录相似的学习者可能需要不同的路径才能达到各自的目标。因此，我们研究了以知识为中心（KC）的个性化学习路径规划问题，在该设定下，规划器必须对学习者画像、掌握状态和先修知识结构进行推理，以决定接下来应该学习哪本教材、哪个单元以及哪个概念。为支持这一研究设定，我们提出了PersonaPath基准，它将2,000个细粒度学习者画像与一个覆盖77个学科、包含347本教材、1,751个单元和4,092个概念的层次化知识图谱相结合。我们在PersonaPath上评估了具有代表性的大语言模型。结果显示，即使是最强的LLM也仅达到29……

    arXiv:2609.18861v1 Announce Type: new  Abstract: Adaptive learning systems commonly formulate learning path planning as Exercise-Centric (EC) recommendation, where the next step is inferred from item-level interaction logs. Evaluating goal-oriented guidance additionally requires explicit learner goals and curriculum-scale prerequisites: learners with similar exercise records may need different paths toward their targets. We therefore study Knowledge-Centric (KC) personalized learning path planning, where a planner must reason over learner profiles, mastery states, and prerequisite knowledge structures to decide which textbook, unit, and concept should be studied next. To support this setting, we introduce PersonaPath, a benchmark that pairs 2,000 fine-grained learner personas with a hierarchical knowledge graph of 347 textbooks, 1,751 units, and 4,092 concepts across 77 subjects. We evaluate representative LLMs on PersonaPath. Results show that even the strongest LLM reaches only a 29.
    
[^29]: 可解码却误路由：稀疏特征揭示视觉-语言模型在有害模因检测中的读取差距

    Decodable but Misrouted: Sparse Features Uncover a Readout Gap in Vision-Language Models for Harmful Meme Detection

    [https://arxiv.org/abs/2609.18860](https://arxiv.org/abs/2609.18860)

    研究发现大型视觉-语言模型内部已编码了检测有害模因所需的证据信息，但无法将其正确路由至输出端——通过稀疏自编码器读取的稀疏特征在六个有害内容基准上均显著优于模型原生预测，揭示了模型存在“可解码但误路由”的读取差距。

    

    当大型视觉-语言模型错误分类有害模因时，这种失败可能反映了内部证据的缺失，或者是无法将已表征的证据正确路由到输出端。我们在Gemma-3和Qwen3.5模型中利用稀疏自编码器、角色条件探针、因果干预和恢复实验来区分这两种情况，并在六个有害内容基准上进行评估，还开展了西班牙语和印地语-英语混合语的补充评估。稀疏读取在所有六个主要二分类任务上都优于模型原生预测：Qwen的稀疏读取平均宏F1达到0.740，而原生预测仅为0.432，残差重建达到0.486；Gemma则从0.532提升至0.714。这些差异反映的是监督可访问性，而非模型中预先存在的原生决策规则，且最具影响力的词元角色取决于具体任务。在所评估的分数尺度下，Qwen的静默特征消融对探针的敏感度高出24-63倍，而路由特征修补……

    arXiv:2609.18860v1 Announce Type: cross  Abstract: When a large vision-language model misclassifies a harmful meme, the failure may reflect missing internal evidence or an inability to route represented evidence to its output. We distinguish these cases in Gemma-3 and Qwen3.5 using sparse autoencoders, role-conditioned probes, causal interventions, and recovery experiments across six harmful content benchmarks, with additional Spanish and Hindi-English code-mixed evaluations. Sparse readouts outperform native prediction on all six primary binary tasks: Qwen averages $0.740$ versus $0.432$ native macro-F1, while residual reconstruction reaches $0.486$, whereas Gemma improves from $0.532$ to $0.714$. These differences reflect supervised accessibility rather than a pre-existing, native decision rule, and the most influential token role depends on the task. Under the evaluated score scales, Qwen silent-feature ablation is $24-63$ times more probe-sensitive, whereas routed-feature patching 
    
[^30]: EviGen：面向可验证临床推理生成的预测性证据支架框架

    EviGen: Predictive Evidence Scaffolding for Verifiable Clinical Rationale Generation

    [https://arxiv.org/abs/2609.18852](https://arxiv.org/abs/2609.18852)

    EviGen提出了一个三层框架，通过预测性证据检索、基于证据的临床推理生成和过程监督验证，从纵向电子健康记录中实现可验证且可靠的临床推理生成。

    

    纵向电子健康记录（EHR）记录了患者多年来的病史，涵盖临床笔记、诊断代码、化验结果和手术操作，其中包含推理可能的临床结局所需的证据。然而，临床医生对这些记录进行全面审查并不现实，而基于大语言模型（LLM）的处理成本高昂且往往不可靠——既会遗漏一些相关的观察信息，又会对其他内容产生幻觉。因此，我们提出了EviGen，一个用于可验证临床推理生成的三层框架。第一层是一个以患者为条件的检索器，它使用可学习的查询来查找对临床结局具有预测性（而不仅仅是文本上相关）的证据，并按预测归因分数对其进行排序。第二层是一个LLM生成器，它以排序后的证据作为支架，生成基于所检索证据片段的临床推理。第三层是一个过程监督验证器，用于检查生成的（摘要在此处被截断）……

    arXiv:2609.18852v1 Announce Type: new  Abstract: Longitudinal electronic health records (EHRs) capture years of patient history across notes, codes, labs, and procedures, and contain evidence needed to reason about likely clinical outcomes. However, comprehensive clinician review of these records is impractical, and LLM-based processing is costly and often unreliable, missing some relevant observations while hallucinating others. We therefore propose EviGen, a three-layer framework for verifiable clinical rationale generation that addresses these challenges. The first layer is a patient-conditioned retriever that uses learnable queries to find evidence predictive of, not just textually relevant to, a clinical outcome and ranks it by prediction attribution scores. The second layer is an LLM generator that consumes this ranked evidence as a scaffold to produce a clinical rationale grounded in the retrieved spans. The third layer is a process-supervised verifier that checks the generated 
    
[^31]: ReFigBench：将科学图形重构作为可编辑 PowerPoint 工件的基准测试

    ReFigBench: Benchmarking Scientific Figure Reconstruction as Editable PowerPoint Artifacts

    [https://arxiv.org/abs/2609.18844](https://arxiv.org/abs/2609.18844)

    提出了 ReFigBench 基准与评估框架，通过让编程智能体将 1,000 张真实科学概述图重构为保留文本、拓扑、布局和原生文档结构的可编辑 PowerPoint 工件，来全面诊断智能体在感知、规划与工具环境层面的真实能力。

    

    多模态编程智能体被期望将视觉输入转化为可用的工件，它们通过一个“线束”来行动——即围绕模型的工具层、上下文管理层和执行环境。现有的评估往往孤立地考察短工具调用、API 调用轨迹或截图相似度，而在这些代理指标下的低分无法说明模型究竟是感知不佳、规划不佳，还是被其线束所拖累。我们研究科学概述图重构这一智能体任务，其中源图像必须转化为一个可编辑的 PowerPoint 幻灯片，同时保留文本、拓扑结构、布局和原生文档结构。我们提出了 ReFigBench，这是一个基于从 arXiv 论文中检索到的 1,000 张具有完整来源信息的真实概述图构建的基准测试和评估框架。来自四个模型家族的编程智能体在两种工作流下对每张图进行重构——直接代码生成和专门的 PPTX 工作流，而最强的模型在两种商业（线束环境）中运行。

    arXiv:2609.18844v1 Announce Type: new  Abstract: Multimodal coding agents are expected to turn visual inputs into usable artifacts, and they act through a harness, the layer of tools, context management, and execution environment around the model. Existing evaluations often isolate short tool calls, API traces, or screenshot resemblance, and a low score under these proxies cannot say whether the model saw poorly, planned poorly, or was failed by its harness. We study scientific overview figure reconstruction, an agent task in which a source image must become an editable PowerPoint slide that preserves text, topology, layout, and native document structure. We introduce ReFigBench, a benchmark and evaluation framework built on 1,000 real overview figures retrieved from arXiv papers with full provenance. Coding agents from four model families reconstruct every figure under two workflows, direct code generation and a specialized PPTX workflow, and the strongest model runs inside two commer
    
[^32]: 利用OCR注意力头将图像语义语言化

    Using OCR Heads to Verbalize Image Semantics

    [https://arxiv.org/abs/2609.18823](https://arxiv.org/abs/2609.18823)

    研究发现视觉语言模型中负责OCR的注意力头实际上是通用语义特征头，将其注意力权重压缩为语言化透镜变换后，可从模型所有层（甚至第0层）的隐藏状态中解读出可解释的图像语义标签，证明图像表示在早期层就与语言对齐。

    

    视觉语言模型（VLM）如何从像素映射到语义？为了理解这个普遍性问题，我们聚焦于一个较窄的问题：研究VLM如何执行光学字符识别（OCR）。我们在四个模型中识别出对OCR具有因果必要性的注意力头，并发现这些实际上是通用注意力头，能够在所有图像token上输出可解释的语义特征。例如，将这些注意力头指向包含单词“bike”（自行车）的图像token，会使Qwen3-VL-8B输出“bike”；而将它们指向鸟的翅膀，则会使模型输出token“feathers”（羽毛）。我们将这些注意力头的注意力权重压缩为一个单一的语言化透镜变换，该变换能够揭示所有层隐藏状态中可解释的语义特征。当与投影到词表空间相结合时，我们可以从第0层开始就获得可解释的标签，这表明图像表示实际上在早期层就与语言对齐了。我们发现……

    arXiv:2609.18823v1 Announce Type: cross  Abstract: How do VLMs map from pixels to semantics? To understand this general question, we focus on a narrow one: studying how VLMs perform optical character recognition (OCR). Across four models, we identify attention heads causally necessary for OCR, and discover that these are in fact general-purpose heads that output interpretable semantic features across all image tokens. For example, pointing these heads at an image token containing the word "bike" causes Qwen3-VL-8B to output "bike," but pointing them at a bird wing causes the model to output the token "feathers." We collapse these heads' attention weights into a single verbalization lens transformation that reveals interpretable semantic features in hidden states across all layers. When combined with projection to vocabulary space, we can obtain interpretable labels starting from layer 0, showing that image representations are in fact aligned with language in early layers. We find that 
    
[^33]: 超越频率测量：上下文嵌入能否捕捉科学文本中的语义变化？

    Beyond frequency measures: Can contextual embeddings capture meaning change in scientific texts?

    [https://arxiv.org/abs/2609.18804](https://arxiv.org/abs/2609.18804)

    本研究提出利用基于SciBERT的上下文嵌入并结合多种统计指标，作为传统频率方法的补充，以有效追踪科学文本中领域术语的历时性语义变化。

    

    识别技术趋势是科学计量学的一项核心任务，然而传统的基于频率的方法难以捕捉领域特定术语的重大语义变化。我们假设上下文嵌入可以补充频率动态，从而有效追踪历时性语义变化。我们在2010年至2024年的天体物理学和自然语言处理（NLP）语料库上比较了基于频率和基于嵌入的方法。候选术语使用KeyBERT提取（以SciBERT作为其底层语言模型），并通过Fisher精确检验筛选出频率显著增加的术语。随后由领域专家评估这些术语是否发生了真正的语义偏移，以建立真实标签。为了量化语义漂移，研究使用多种指标比较每个术语在两个离散时期的上下文嵌入“云”：余弦距离、平均成对距离、Hotelling型T²和最大均值差异。结果表明……（原文摘要至此截断）

    arXiv:2609.18804v1 Announce Type: new  Abstract: Identifying technological trends is a core scientometric task, yet traditional frequency-based approaches struggle to capture substantial meaning shifts of domain-specific terms. We hypothesise that contextual embeddings can complement frequency dynamics to effectively track diachronic semantic change. We compare frequency and embedding-based approaches across Astrophysics and NLP corpora spanning from 2010 to 2024. Candidate terms are extracted using KeyBERT (utilizing SciBERT as its underlying language model) and filtered for significant frequency increases using Fisher's exact test. These terms are then evaluated for genuine semantic shift by domain experts to establish ground-truth labels. To quantify semantic drift, each term's contextual embedding ''clouds'' from the two discrete periods are compared using multiple metrics: cosine distance, average pairwise distance, Hotelling-type T 2 , and maximum mean discrepancy. Results indica
    
[^34]: 手语手形的零样本跨语言识别

    Zero-Shot Cross-Lingual Recognition of Sign Language Handshapes

    [https://arxiv.org/abs/2609.18772](https://arxiv.org/abs/2609.18772)

    本文提出了首个零样本跨语言手语手形识别框架，通过将手形分解为两种语言共享的五种音系特征，成功实现了从美国手语（ASL）到加泰罗尼亚手语（LSC）的知识迁移，达到了80.0%的音系特征准确率和54.5%的手形准确率。

    

    手语处理在美国手语（ASL）等高资源语言中发展迅速，然而世界上大多数手语缺乏新方法所需的音系学标注。我们提出了首个用于手形识别的零样本跨语言框架，实现从美国手语（ASL）向加泰罗尼亚手语（LSC）的迁移。我们的方法利用将手形分解为五个音系特征——所选手指、弯曲度、张度、拇指位置和拇指接触——这些特征在两种语言间共享，并通过复合音系距离度量从预测的特征中解码LSC手形。我们在一个包含37种手形的单签名者LSC基准上评估了在两个ASL语料库（PopSign、Sem-Lex）上训练的三种架构（MLP、SL-GCN、SHuBERT）。结果表明，在协调录制格式差异后，零样本迁移是可行的，达到了80.0%的音系特征准确率和54.5%的预期手形准确率。

    arXiv:2609.18772v1 Announce Type: new  Abstract: Sign language processing advances rapidly for high-resource languages such as American Sign Language (ASL), yet most of the world's sign languages lack the phonological annotations new methods require. We present the first zero-shot cross-lingual framework for handshape recognition, transferring from ASL to Catalan Sign Language (LSC). Our approach leverages the decomposition of handshapes into five phonological features -- selected fingers, flexion, spread, thumb position, and thumb contact -- shared across both languages, to decode LSC handshapes from predicted features via a composite phonological distance metric. We evaluate three architectures (MLP, SL-GCN, SHuBERT) trained on two ASL corpora (PopSign, Sem-Lex) against a 37-handshape, single-signer LSC benchmark. Zero-shot transfer proves viable once recording-format disparities are harmonized, reaching 80.0% phonological feature accuracy and 54.5% expected handshape accuracy. Phono
    
[^35]: FRAUDSkill：面向音频反欺诈检测的结构化冻结权重技能优化方法

    FRAUDSkill: Structured Frozen-Weight Skill Optimization for Audio Anti-Fraud Detection

    [https://arxiv.org/abs/2609.18766](https://arxiv.org/abs/2609.18766)

    本文提出FRAUDSkill框架，在不修改底层音频-语言模型参数的情况下，通过外部优化技能程序、路由策略和决策规则，实现了能够灵活适应欺诈模式演变的结构化音频反欺诈检测。

    

    大型音频-语言模型通过直接处理语音并对欺诈相关证据进行推理，在反欺诈检测任务中展现出巨大潜力。然而，模型的实际部署要求预测结果遵循预定义的标签空间，以及一个由服务场景识别、欺诈检测和条件性欺诈类型分类组成的结构化决策协议。现有的微调和基于提示的方法通常将任务知识、约束条件和决策规则编码到模型参数或手动维护的提示中，这使得它们难以随着欺诈模式和标注策略的演进而灵活调整。为此，我们提出了FRAUDSkill，这是一个结构化的冻结权重适配框架，它在保持底层音频-语言模型完全不变的同时，优化一个由技能程序、路由特定策略和决策规则组成的外部层。我们进一步将结构化输出控制与验证引导的多路径推理相结合，以……（摘要内容不完整，此处为截断部分）

    arXiv:2609.18766v1 Announce Type: cross  Abstract: Large audio-language models have shown promise for anti-fraud detection by directly processing speech and reasoning over fraud-related evidence. Their deployment, however, requires predictions to follow a predefined label space and a structured decision protocol consisting of service-scenario identification, fraud detection, and conditional fraud-type classification. Existing fine-tuning and prompt-based approaches typically encode task knowledge, constraints, and decision rules into model parameters or manually maintained prompts, making them difficult to adapt as fraud patterns and labeling policies evolve. To this end, we propose FRAUDSkill, a structured frozen-weight adaptation framework that leaves the underlying audio-language model unchanged while optimizing an external layer of skill programs, route-specific policies, and decision rules. We further combine structured output control with validation-guided multi-path inference to
    
[^36]: TeleAntiFraud 2.0：一个可刷新、基于用户档案、面向电信欺诈检测的音频基准

    TeleAntiFraud 2.0: A Refreshable, Profile-Grounded, and Audio-Based Benchmark for Telecom Fraud Detection

    [https://arxiv.org/abs/2609.18748](https://arxiv.org/abs/2609.18748)

    提出了 TeleAntiFraud 2.0，一个可按月更新、基于用户档案且能在共享上下文中区分欺诈与合法近域通话的音频电信欺诈检测基准。

    

    电信欺诈话术快速演变，且常常被设计得类似于日常服务对话，这对基于音频的电信欺诈评估提出了两个关键要求。首先，基准测试必须能够纳入新观察到的诈骗模式，而不会覆盖之前已建立的测试集。其次，基准必须能够将欺诈与合法的近领域通话区分开来，而不是依赖于主题分离的负面样本。我们提出了 TeleAntiFraud 2.0，该基准采用我们的混合树反欺诈生成流水线（Mixed-Tree Anti-Fraud Generation Pipeline）构建，并在每月冻结评估协议下进行评估。该流水线将在线欺诈案例摘要转化为基于用户档案的场景，通过混合树生成进行扩展，在共享上下文中实现欺诈与非欺诈对话路径，将验证过的对话渲染为角色匹配的语音，并将生成的音频、标签、提示词、清单和溯源记录冻结，用于每个月的评估集。每个冻结的集合……

    arXiv:2609.18748v1 Announce Type: cross  Abstract: Telecom fraud scripts evolve rapidly and are often designed to resemble routine service conversations, creating two key requirements for audio-based telecom-fraud evaluation. First, benchmarks must incorporate newly observed scam patterns without overwriting previously established test sets. Second, they must distinguish fraud from lawful, near-domain calls rather than relying on topic-separated negative examples. We present TeleAntiFraud 2.0, constructed with our Mixed-Tree Anti-Fraud Generation Pipeline and evaluated under a monthly frozen evaluation protocol. The pipeline transforms online fraud-case abstracts into profile-grounded scenarios, expands them through mixed-tree generation, realizes fraud and non-fraud dialogue paths under shared contexts, renders validated dialogues as role-matched speech, and freezes the resulting audio, labels, prompts, manifests, and provenance records for each monthly evaluation set. Each frozen set
    
[^37]: 低资源语言中自动NER标注纠正的可扩展框架

    A Scalable Framework for Automated NER Annotation Correction in Low-Resource Languages

    [https://arxiv.org/abs/2609.18739](https://arxiv.org/abs/2609.18739)

    本文提出了一个基于频率的迭代自训练框架，结合双阈值机制自动纠正低资源语言的NER噪声标注，显著提升了NER性能，并探索了生成式大语言模型在低资源语言NER任务中的应用潜力。

    

    与任何其他自然语言处理（NLP）任务一样，命名实体识别（NER）中的低质量或噪声标注使得实现最先进的性能变得具有挑战性。在本文中，我们提出了一个多步骤框架，通过采用自动化技术来提高NER数据集的标注质量。我们提出了一种基于频率的迭代方法，该方法利用自训练和双阈值机制来增强推理置信度。在不同NER数据集上的实验评估表明，相对于原始数据集，NER性能有显著提升。这项工作进一步探索了生成式大语言模型（LLMs）为低资源语言执行NER任务的潜力。

    arXiv:2609.18739v1 Announce Type: cross  Abstract: Poor quality or noisy annotations in Named Entity Recognition (NER), as in any other NLP task, make it challenging to achieve state-of-the-art performance. In this paper, we present a multi-step framework to enhance the annotation quality of NER datasets by employing automated techniques. We propose a frequency-based iterative approach that leverages self-training and a dual-threshold mechanism to enhance inference confidence. Experimental evaluations on different NER datasets demonstrate significant improvements in NER performance with respect to the original datasets. This work further explores the potential of generative Large Language Models (LLMs) to perform NER for low-resource languages.
    
[^38]: 《“如果我只能买一部手机：Galaxy S26 Ultra”：对AI生成的产品推荐进行审计》

    "If I Had to Buy Just ONE: Galaxy S26 Ultra": Auditing AI-Generated Product Recommendations

    [https://arxiv.org/abs/2609.18729](https://arxiv.org/abs/2609.18729)

    该研究构建了包含2,528个真实购物咨询查询的ConsumerQ数据集，对ChatGPT、Gemini和Google AI Overviews的产品推荐进行审计，发现ChatGPT在79%的推荐中表达明确的第一人称偏好，且不同AI系统之间引用的信息来源高度不一致，揭示了AI购物推荐的偏见与不公正问题。

    

    消费者越来越多地使用AI聊天机器人来获取购物建议。随着OpenAI和Google等公司通过广告将其AI服务货币化，这引发了关于此类建议的偏见性和公正性的难题。为此，我们使用真实的商业咨询查询对流行的聊天机器人进行了AI审计。首先，我们策划了一个包含2,528个真实商业咨询查询的数据集。然后，我们评估了来自热门AI聊天机器人的1,536个针对产品查询的响应：ChatGPT（聊天机器人及API）、Google Gemini（聊天机器人及API）以及Google搜索（AI Overviews）。我们发现ChatGPT在79%的产品推荐响应中表达了第一人称的产品偏好，相比之下Gemini为7%，AI Overviews为2%，而推荐的产品在重复请求时经常发生变化。展示的信息来源差异很大：对于相同的查询，ChatGPT和Gemini界面上平均仅有5.4%的引用域名相同。

    arXiv:2609.18729v1 Announce Type: cross  Abstract: Consumers increasingly use AI chatbots for advice on what to buy. With companies like OpenAI and Google monetising their AI through advertising, this raises difficult questions about the bias and impartiality of such advice. In response, we conduct an AI audit of popular chatbots using real commercial-advice queries. First, we curate a dataset of 2,528 real commercial-advice queries (ConsumerQ). Then, we evaluate 1,536 responses to product queries from popular AI chatbots: ChatGPT (chatbot and API), Google Gemini (chatbot and API), and Google Search (AI Overviews). We find that ChatGPT expresses a first-person product preference in 79% of product-recommending responses, compared with 7% for Gemini and 2% for AI Overviews, while the products recommended often change across repeated requests. Displayed sources vary strongly: for the same query, the ChatGPT and Gemini interfaces share only 5.4% of domains on average, with no domain in com
    
[^39]: LocQE：通过利用译后编辑实现本地化质量评估的原则性领域自适应

    LocQE: Principled Domain Adaptation for Localisation Quality Estimation by Leveraging Post-Edits

    [https://arxiv.org/abs/2609.18720](https://arxiv.org/abs/2609.18720)

    提出LocQE方法，通过利用少量译后编辑数据进行多任务微调和简单的分词器干预，实现本地化质量评估的原则性领域自适应，显著提升QE模型在本地化场景中对数字、空格、标点等因素的敏感性及译文排序能力。

    

    诸如COMETKiwi等学习型质量评估（QE）模型被广泛应用，并在通用机器翻译评估中表现良好。然而，众所周知，它们在未见过的领域上表现不佳，限制了其在真实本地化场景中的性能。我们表明，这些模型对本地化中的一些重要因素不敏感，例如数字是否被准确翻译，甚至翻译中是否保留了正确数量的空格和标点符号。此外，机器翻译优化的一个关键能力是QE模型对单个句段的不同译文进行准确排序的能力，而这一能力在领域迁移中受到显著损害。在缺乏大规模直接评估数据的情况下，我们提出了原则性的微调方法，即使仅使用少量译后编辑数据也能缩小领域差距。通过采用多任务微调方法和简单的分词器干预，我们创建了一个QE模型……

    arXiv:2609.18720v1 Announce Type: new  Abstract: Learned quality estimation (QE) models such as COMETKiwi are widespread and work well for general machine translation evaluation. However, they are known to struggle on unseen domains, limiting their performance in a real-world localisation context. We show that they are insensitive to some important factors in localisation, such as whether numbers are translated accurately, or even whether the correct number of spaces and punctuation are preserved in a translation. Further, a key capability for optimisation of machine translation is the ability of QE models to accurately rank different translations of a single segment, which suffers significantly from the domain transfer. In the absence of large-scale direct assessment data, we propose principled fine-tuning approaches to reduce the domain gap with even small amounts of post-editing data. Using a multi-task fine-tuning approach and a simple tokeniser intervention, we create a QE model w
    
[^40]: 在变迁领域中追踪个体知识轨迹：以广义相对论与引力为例

    Tracing individual knowledge trajectories in a changing field: the case of general relativity and gravitation

    [https://arxiv.org/abs/2609.18697](https://arxiv.org/abs/2609.18697)

    提出四项量化指标（自有词汇、嵌入密度估计、引用词汇和引用身份），将广义相对论与引力领域50位最高产作者的知识轨迹与不同时期的领域文献进行系统比较。

    

    arXiv:2609.18697v1 公告类型：新论文 摘要：历史学家已在领域层面和个体职业生涯层面重构了二十世纪广义相对论与引力的转型，但连接这两个尺度需要一种能够将研究者与随时间变化的领域进行比较的方法。我们开发了这样一种比较方法，将研究者的出版物和参考文献与同一时期、更早时期和更晚两年时期的广义相对论与引力领域文献进行对照。在我们早期双案例研究中的自有词汇和嵌入密度估计方法基础上，我们将分析扩展到NASA/ADS语料库（约180,000条广义相对论与引力记录，1911年至2000年）中发表量最多的五十位作者，并新增了两项基于引用的测量指标：引用词汇和引用身份。这四项指标分别将作者的书面语言、引用文献、语义邻域和引用权威配置与周围领域进行比较。早期案例表明，更接近领域……

    arXiv:2609.18697v1 Announce Type: new  Abstract: Historians have reconstructed the twentieth-century transformation of general relativity and gravitation (GRG) at the field level and through individual careers, but connecting these scales requires a way to compare researchers with the changing field over time. We develop such a comparison, setting a researcher's publications and references against GRG field literature from the same, earlier, and later two-year periods. Building on Own Vocabulary and Embedding Density Estimation from our earlier two-case study (arXiv:2501.00391), we extend the analysis to the fifty most-published authors in a NASA/ADS corpus of about 180,000 GRG records (1911 to 2000) and add two citation-based measures, Referenced Vocabulary and Citation Identity. The four measures compare an author's written language, cited literature, semantic neighbourhood, and cited-authority configuration with the surrounding field. The earlier cases suggested that closer field-vo
    
[^41]: RankGround：基于轻量级重排序器引导的裁剪选择的高效高分辨率GUI定位

    RankGround: Efficient High-Resolution GUI Grounding via Lightweight Reranker-Guided Crop Selection

    [https://arxiv.org/abs/2609.18690](https://arxiv.org/abs/2609.18690)

    RankGround提出一种两阶段框架，利用轻量级多模态重排序器GroundRanker从密集候选裁剪中挑选最优区域，每次查询仅需单次VLM调用即可实现高效准确的高分辨率GUI定位。

    

    图形用户界面（GUI）定位是多模态智能体的一项基础感知任务，使其能够理解自然语言指令并与数字界面进行交互。现有方法在准确性和效率之间面临根本性的权衡：直接全图推理通常无法捕捉到细小或视觉上相似的UI元素，而多裁剪策略虽然能改善定位效果，但代价是每次查询需要进行多次昂贵的视觉语言模型（VLM）调用。为应对这一挑战，我们提出RankGround，这是一个两阶段框架，每次查询仅需单次VLM调用即可实现准确的GUI定位。我们方法的核心是GroundRanker，一个轻量级多模态重排序器，能够从密集的候选裁剪集中识别出最有希望的裁剪区域。由于目前没有现成的排序数据集可用，我们从现有的定位数据集中构建排序监督数据。通过严格的包含标准和边界[增强策略]……（摘要内容不完整，截断于此）

    arXiv:2609.18690v1 Announce Type: cross  Abstract: Graphical User Interface (GUI) grounding is a fundamental perception task for multimodal agents, enabling them to interpret natural language instructions and interact with digital interfaces. Existing methods face a fundamental trade-off between accuracy and efficiency: direct full-image inference often fails to capture small or visually similar UI elements, while multi-crop strategies improve localization at the cost of multiple expensive Vision-Language Model (VLM) calls per query.   To address this challenge, we propose RankGround, a two-stage framework that achieves accurate GUI grounding with a single VLM call per query. Central to our approach is GroundRanker, a lightweight multimodal reranker that identifies the most promising crop from a dense candidate set. Because no off-the-shelf ranking dataset is available, we construct ranking supervision data from existing grounding datasets. A strict containment criterion and boundary-a
    
[^42]: HearInContext：语音识别中隐式上下文的基准测试

    HearInContext: A Benchmark for Implicit Context in Speech Recognition

    [https://arxiv.org/abs/2609.18680](https://arxiv.org/abs/2609.18680)

    该论文提出了中英文同音词基准测试HearInContext用于评估语音识别模型的隐式与显式上下文利用能力，并通过微调Qwen3-ASR-1.7B将隐式上下文目标词召回率提升约11个百分点，同时不损害通用识别性能。

    

    情境化自动语音识别（ASR）可以受益于语义线索，或受益于上下文中明确提供的目标词。我们提出了HearInContext，这是一个中英文基准测试，它将共享的合成语音与支持不同解释的助手回复配对。该基准包含围绕同音词构建的3,764个语义测试用例。隐式上下文不包含候选词；显式上下文则点名目标词。无上下文和无关上下文的对照组用于衡量相关历史记录的益处以及对无关历史记录的敏感度。具备上下文能力的模型能从隐式线索中受益，但在有显式提示时能获得更高的目标词召回率。对Qwen3-ASR-1.7B进行微调后，中文和英文的隐式上下文目标词召回率分别提升了11.0和11.5个百分点，同时在AISHELL-1和LibriSpeech上的绝对CER/WER变化保持在0.1个百分点以下。收益还延伸到了微调中未包含的显式条件以及中文场景。

    arXiv:2609.18680v1 Announce Type: new  Abstract: Contextual ASR can benefit from semantic cues or from target words explicitly provided in the context. We introduce HearInContext, a Mandarin--English benchmark that pairs shared synthetic speech with assistant replies supporting different interpretations. The benchmark comprises 3,764 semantic test cases built around homophones. Implicit contexts exclude candidate words; explicit contexts name the target. No-context and unrelated-context controls measure the benefit of relevant history and sensitivity to irrelevant history. Context-capable models benefit from implicit cues but achieve higher target recall with explicit hints. Fine-tuning Qwen3-ASR-1.7B improves implicit-context target recall by 11.0 and 11.5 percentage points in Mandarin and English, respectively, while absolute CER/WER changes on AISHELL-1 and LibriSpeech remain below 0.1 percentage points. Gains extend to explicit conditions excluded from fine-tuning and to Mandarin h
    
[^43]: 理性之声：面向口语数学的强化学习

    Voice of Reason: Reinforcement Learning for Spoken Math

    [https://arxiv.org/abs/2609.18677](https://arxiv.org/abs/2609.18677)

    将带可验证奖励的强化学习应用于GLM-4-Voice语音模型，无需额外推理标记即可显著提升口语数学推理在GSM8K上的准确率，弥合了语音模型与文本模型在数学推理能力上的差距。

    

    语音语言模型相比级联系统能够实现更丰富的人机口语交互，可以访问副语言信息并具有更低的延迟。然而，它们在数学推理基准测试中的准确率一直落后于文本模型。带可验证奖励的强化学习（RL）在扩展文本模型解决复杂问题的能力和抑制幻觉方面发挥了重要作用。在这项工作中，我们探索将强化学习应用于GLM-4-Voice语音模型（Zeng et al., 2024），以弥合文本与口语数学问题求解之间的差距。我们首先通过在合成的口语问答数据上进行监督微调，使模型适应该领域。随后我们证明，即使不使用额外的推理标记，强化学习也能将模型在GSM8K上的准确率提升到此前语音模型仅在配备补充推理轨迹时才能达到的水平。当与现有的流式推理……

    arXiv:2609.18677v1 Announce Type: new  Abstract: Speech language models enable richer spoken interactions between humans and machines than cascaded systems, allowing access to paralinguistic information and lower latency. However, their accuracy on mathematical reasoning benchmarks has lagged behind those of text models. Reinforcement learning (RL) with verifiable rewards has been instrumental in extending text models' capabilities for solving complex problems and limiting hallucinations. In this work, we explore applying RL to the GLM-4-Voice speech model (Zeng et al., 2024) to bridge the gap between textual and spoken mathematical problem solving. We first adapt the model to the domain using supervised fine-tuning on synthesized spoken question-answering data. We then show that, even without extra reasoning tokens, RL improves the accuracy on GSM8K beyond levels previously achieved for speech models only with supplementary reasoning traces. When combined with existing streaming reaso
    
[^44]: 选择即检索，弃权则不然：关于70个韩英动作的设备端工具路由

    Selection Is Retrieval, Abstention Is Not: On-Device Tool Routing over 70 Korean-English Actions

    [https://arxiv.org/abs/2609.18672](https://arxiv.org/abs/2609.18672)

    该论文研究了在设备端工具路由中，用检索器替代语言模型在"选择工具"决策上可行，但在"判断无合适工具并弃权"决策上存在根本缺陷。

    

    arXiv:2609.18672v1 公告类型：新论文 摘要：一个调用工具的AI助手在每次请求时需要做出两个决策：调用哪个工具，以及是否有任何可用工具适用。在常见的设计中，单个语言模型同时做出这两个决策，通过发出调用或拒绝发出调用来实现。在一个无需服务器就能响应的设备上，语言模型正是使这种设计成本高昂的原因，它主导了路由器的延迟和内存。常见的替代方案是完全移除模型，改用检索器对本地动作目录进行排序。这种替换在两个决策上并不对称。检索器对每个输入都返回其最高分的候选，无法表明目录中没有有效的动作。我们早先的研究发现，将解码器约束到工具语法上可以修复格式错误的输出，但不会改善选择。这种替换在每个决策上的成本尚未被测量。我们分别评估了600个Ko（原文在此截断）

    arXiv:2609.18672v1 Announce Type: new  Abstract: An AI assistant that calls tools makes two decisions on every request: which tool to invoke, and whether any available tool applies. In the usual design a single language model makes both, by emitting a call or by declining to emit one. On a device that has to answer without a server, the language model is what makes that design expensive, dominating both the latency and the memory of the router. The common alternative is to remove the model completely and rank the catalog of local actions with a retriever instead. That substitution is not symmetric across the two decisions. A retriever returns its highest-scoring candidate for every input and cannot signal that the catalog holds no valid action. Our earlier study found that constraining a decoder to a tool grammar repairs malformed output without improving the choice. What the substitution costs in each decision has not been measured. We evaluate the two decisions separately over 600 Ko
    
[^45]: DyMT-ESB：用户与大语言模型交互中社会偏见的动态多轮评估

    DyMT-ESB: Dynamic Multi-Turn Evaluation of Social Bias in User-LLM Interactions

    [https://arxiv.org/abs/2609.18649](https://arxiv.org/abs/2609.18649)

    本文提出DyMT-ESB受控评估协议，根据不断演变的对话历史动态生成后续用户查询并支持可变轮数评估，揭示了大语言模型在多轮交互中存在延迟出现、非单调变化及反复出现的社会偏见现象。

    

    警告：本文包含刻板印象和社会偏见的示例。大语言模型（LLM）正日益被公众广泛用于交互式场景，这使得在多轮对话场景中评估模型行为（包括与刻板印象相关的危害）对于安全性而言至关重要。然而，现有的多轮社会偏见评估通常依赖于预先指定或基于模板的用户输入，这些输入无法根据模型的响应进行调整，且通常事先假定固定的对话长度。在本文中，我们采用一种受控评估协议来研究响应条件下的多轮交互中的社会偏见动态，该协议根据不断演变的对话历史生成后续用户查询，并支持对可变轮数的评估。实验结果表明，即使在一致的、响应条件下的多轮交互中，大语言模型依然会表现出社会偏见，揭示了延迟出现的偏见、非单调的偏见模式以及偏见的重新出现。

    arXiv:2609.18649v1 Announce Type: new  Abstract: Warning: This paper contains examples of stereotypes and social bias. LLMs are increasingly used in interactive settings by the general public, making the evaluation of model behavior in multi-turn conversational scenarios important for safety, including stereotyping-related harms. However, existing multi-turn social bias evaluations often rely on pre-specified or template-based user inputs that do not adapt to model responses and typically assume a fixed dialogue length in advance. In this paper, we study social bias dynamics in response-conditioned multi-turn interactions using a controlled evaluation protocol that generates follow-up user queries from the evolving dialogue history and allows evaluation over variable numbers of turns. Experimental results show that LLMs exhibit social bias even in coherent, response-conditioned multi-turn interactions, revealing late-emerging bias, non-monotonic bias patterns, and bias re-emergence. Th
    
[^46]: 谬误基准测试衡量的是论证图式识别，而非谬误检测

    Fallacy Benchmarks Measure Scheme Recognition, Not Fallacy Detection

    [https://arxiv.org/abs/2609.18644](https://arxiv.org/abs/2609.18644)

    该论文揭示了谬误检测基准报告的低误报率是“有效”类别构建方式的产物而非真实检测能力——当使用与谬误具有相同论证图式的正确论证作为负样本测试时，模型误报率大幅上升（CoCoLoFa上从16.6%升至58.9%），证明现有模型实际只是识别论证图式而非真正检测谬误。

    

    谬误检测基准通常将谬误类别与一个单一的“有效”或“无”类别配对，该类别包含了数据收集过程中未被标注为谬误的所有内容。这种构建方式具有误导性：分类器可以学习到某些线索从而在该类别上表现良好，却并未真正学会区分谬误与正确论证。我们证明，基准测试所报告的低误报率是类别构建方式的产物，而非检测能力的体现。对谬误而言，最有信息量的负样本是使用相同论证图式的正确论证，而在我们考察的四个基准中，此类论证在“有效”类别中最多只占几个百分点。在构建的图式匹配负样本上进行评估时，误报率在CoCoLoFa上从16.6%上升到58.9%，在Reddit上从5.7%上升到62.0%。由于误报率取决于负样本的撰写方式，我们还比较了来自同一流程、仅在论证图式身份上有所不同的两种条件。（摘要原文在此处被截断）

    arXiv:2609.18644v1 Announce Type: new  Abstract: Fallacy-detection benchmarks pair fallacy classes with a single "valid" or "none" class that takes everything data collection did not label as a fallacy. This construction is misleading: a classifier can learn cues that do well on this class without learning to tell a fallacy from a correct argument. We show that the low false-positive rates benchmarks report are an artifact of how the class is built, not evidence of detection ability. The most informative negative for a fallacy is a correct argument using the same argumentation scheme, and such arguments are at most a few percent of the valid class across the four benchmarks we examined. Evaluated on constructed scheme-matched negatives, false-positive rates rise from 16.6% to 58.9% on CoCoLoFa and from 5.7% to 62.0% on Reddit. That rate depends on how the negatives are written, so we also compare two conditions from the same pipeline that differ only in scheme identity. Classifiers lab
    
[^47]: STRETCH突破边界：一个面向大语言模型渐进式进化的统一自学框架

    STRETCH the Boundaries: A Unified Self-Taught Framework for Progressive LLM Evolution

    [https://arxiv.org/abs/2609.18642](https://arxiv.org/abs/2609.18642)

    STRETCH框架通过动态“拉伸区”机制使问题难度与模型能力持续匹配，让模型在单一参数空间内交替扮演脚手架构建者和学习者角色，通过双循环共同进化实现大语言模型推理能力的渐进式提升。

    

    大型语言模型（LLMs）在自我改进训练中常常遭受能力停滞的困扰，因为固定的难度水平无法适应其不断发展的熟练程度。为了解决这个问题，我们提出了STRETCH（通过目标挑战实现的自学推理进化），这是一个受认知脚手架理论启发的统一框架。STRETCH引入了一个动态的“拉伸区”机制，使问题难度与模型的解题能力持续保持匹配。在单一参数空间内，模型在“脚手架构建者”（生成自适应的、挑战模型边界的问题）和“学习者”（通过强化学习优化其解题轨迹）之间交替进行。这种双循环共同进化有效稳定了训练过程，缓解了奖励作弊问题，并促进了推理能力的渐进式增长。在谈判和运筹学基准测试上的实验表明，STRETCH始终优于强大的提示方法。

    arXiv:2609.18642v1 Announce Type: new  Abstract: Large language models (LLMs) often suffer from capability stagnation in self-improvement training because fixed difficulty levels fail to adapt to their evolving proficiency. To address this issue, we propose STRETCH (Self-Taught Reasoning Evolution via Targeted CHallenge), a unified framework inspired by cognitive scaffolding theory. STRETCH introduces a dynamic Stretch Zone mechanism that continuously aligns question difficulty with the model's solving capability. Within a single parameter space, the model alternates between a Scaffolder that generates adaptive, boundary-pushing challenges and a Learner that that optimizes its solving trajectories through reinforcement learning. This dual-loop co-evolution effectively stabilizes training, mitigates reward hacking and promote progressive reasoning growth. Experiments on both negotiation and operation research benchmarks demonstrate that STRETCH consistently outperforms strong prompting 
    
[^48]: 弱化神经元：Transformer中具有超大规模影响力的输入-输出功能

    Weakening Neurons: An Input-Output Functionality in Transformers with Outsize Influence

    [https://arxiv.org/abs/2609.18612](https://arxiv.org/abs/2609.18612)

    该论文提出通过计算神经元输入权重向量与输出权重向量之间的余弦相似度来识别“弱化神经元”，并发现这类神经元虽然在模型中数量稀少，却激活频繁且对模型行为具有超乎寻常的影响力，同时九个不同的大语言模型均呈现弱化神经元集中分布于后期层、强化神经元集中分布于中早期层的相似模式。

    

    我们分析了大语言模型（LLM）中基于GLU的神经元所学习到的输入-输出行为。我们提出了一种简单的分析方法：对于每个神经元，计算其输入（读取）权重向量与输出（写入）权重向量之间的余弦相似度。在该方案下，强负余弦相似度表明该神经元会削弱它在残差流中检测到的方向，因此我们将其称为“弱化神经元”。这使我们获得了一些新颖的见解。首先，我们展示了九个不同的大语言模型具有相似的模式：弱化神经元主要出现在后期层，而它们的对应物——（条件性）强化神经元——则频繁出现在中早期层。其次，我们发现弱化神经元表现出令人惊讶的行为：尽管数量很少，但它们激活频繁，并对模型行为产生巨大影响。第三，当门控值为负时，弱化神经元对模型输出具有强烈的影响。

    arXiv:2609.18612v1 Announce Type: cross  Abstract: We analyze the learned input-output behavior of GLU-based neurons in large language models (LLMs). We propose a simple analysis method: For each neuron, we compute the cosine similarities between its input (reading) and output (writing) weight vectors. In this scheme, a strong negative cosine similarity indicates the neuron weakens the direction it detects in the residual stream, so we call this a weakening neuron. This allows us to gain a number of novel insights. First, we show that nine different LLMs have similar patterns: weakening neurons appear mostly in late layers whereas their counterparts, (conditional) strengthening neurons, are frequent in early-middle layers. Second, we find that weakening neurons display surprising behavior: even though there are few, they activate often and have a large influence on model behavior. Third, weakening neurons have a strong effect on model output when gate values are negative -- which is su
    
[^49]: PACT：企业AI助手在压力之下能否被信任？

    PACT: Can Enterprise AI Assistants Be Trusted Under Pressure?

    [https://arxiv.org/abs/2609.18605](https://arxiv.org/abs/2609.18605)

    PACT是一个评估企业级AI智能体在用户施压等压力情境下能否坚持遵守合规规则的基准测试，涵盖十二个受监管企业领域和四十八个真实多轮对话场景。

    

    随着企业AI应用的持续增长，企业级大语言模型（LLM）智能体正被部署到招聘、医疗保健和金融等敏感场景中。在这些场景中，遵守智能体系统上下文中规定的规则是首要的法律关切。目前，尚无评估框架能够系统地衡量哪些LLM模型容易违反合规规则，尤其是在面临固执用户的施压、仓促经理的催促，或违规行为便利且有诱惑力的情况下。我们提出了PACT（压力应用合规测试），这是一个针对AI智能体在压力下遵循规则能力的基准测试，涵盖十二个受监管的企业领域和四十八个场景，每个场景均设置在协助员工完成日常任务的真实多轮对话中。每个基准项目都将一条现行规则与一个违反规则的捷径配对，并在不同的措辞和系统提示模式下施加一系列压力。我们……

    arXiv:2609.18605v1 Announce Type: cross  Abstract: As corporate AI adoption continues to grow, enterprise-grade LLM agents are being deployed into sensitive contexts such as hiring, healthcare, and finance. In these contexts, compliance with rules specified in an agent's system context is a first-order legal concern. Currently, no evaluation framework systematically measures which LLM models tend to violate compliance rules, especially under pressure from a persistent user, a hurried manager, or circumstances where violation is convenient or attractive. We introduce PACT (Pressure-Applied Compliance Testing), a benchmark for rule-following under pressure in AI agents assisting employees in daily tasks across twelve regulated enterprise domains and forty-eight scenarios, each set in a realistic multi-turn conversation. Each benchmark item pairs a standing rule against a rule-violating shortcut, and applies a battery of pressures across different wordings and system-prompt modes. We cons
    
[^50]: 用于合成语言生成的变分量子Transformer架构

    Variational Quantum Transformer Architecture for Synthetic Language Generation

    [https://arxiv.org/abs/2609.18565](https://arxiv.org/abs/2609.18565)

    提出了一种兼容NISQ设备的紧凑变分量子Transformer架构，通过用量子编码器、连接器和解码器电路替代经典注意力与前馈子层，能够端到端训练并学习非平凡的语法结构，在合成语言生成任务上实现完美确定性生成和高词典序有效性。

    

    我们提出了一种紧凑的、兼容NISQ设备的量子Transformer架构，用于合成量子自然语言处理（QNLP）序列建模。该模型保留了经典Transformer的自回归下一词元预测接口，但用变分量子编码器模块、连接电路、解码器模块以及直接的双量子比特测量读出取代了注意力机制和前馈子层。词元上下文通过角度编码进入小型量子寄存器，由并行的变分头和编码器集成电路进行处理，并通过解码器辅助量子比特进行条件化，从而产生四词元词汇表上的概率分布。我们在确定性和词典序语法生成任务上，以紧凑的经典Transformer为基线，评估了多种架构变体。量子模型可以端到端训练并学习非平凡的语法结构，包括个别运行中实现完美的确定性生成，以及最强架构变体中展现出高词典序有效性。

    arXiv:2609.18565v1 Announce Type: cross  Abstract: We propose a compact NISQ-compatible quantum transformer architecture for synthetic QNLP sequence modelling. The model preserves the autoregressive next-token interface of a classical transformer, but replaces attention and feed-forward sublayers with variational quantum encoder blocks, connector circuits, decoder blocks and a direct two-qubit measurement readout. Token contexts are angle-encoded into small quantum registers, processed by parallel variational heads and encoder integration circuits and conditioned through decoder ancillae to produce a distribution over a four-token vocabulary. We evaluate several architecture variants on deterministic and lexicographic grammar-generation tasks against a compact classical transformer baseline. The quantum models are trainable end-to-end and learn nontrivial grammar structure, including perfect deterministic generation in individual runs and high lexicographic validity in the strongest va
    
[^51]: 探测偏移并非公平性解决方案：语音模型中表征引导的局限性

    A Probe Shift Is Not a Fairness Fix: The Limits of Representation Steering in Speech Models

    [https://arxiv.org/abs/2609.18533](https://arxiv.org/abs/2609.18533)

    研究发现，尽管性别、口音等说话人属性可从预训练语音模型编码器中被高度线性解码，但通过注入探测方向来引导内部表征并不能可靠地缩小不同群组间的词错误率差距，表明探测到的表征偏移并非实现公平性的有效修复手段。

    

    自动语音识别（ASR）系统在不同说话人群组之间表现出不均等的错误率，这促使研究者对其内部表征进行干预。我们探讨从预训练ASR编码器中可线性读取的说话人相关属性，是否能提供有助于缩小群组词错误率（WER）差距的有用方向。我们在Common Voice和Speech Accent Archive数据集上，对Whisper-medium、HuBERT-large和Wav2Vec2-large的每一层编码器进行探测，获取来自元数据的性别、年龄和母语/口音标签；构建基于质心和探测器的方向向量；将它们注入选定层；并将下游探测轨迹与匹配的WER变化进行比较。结果显示，性别标签具有高度可解码性（最佳宏F1为0.924–0.941），母语/口音标签也高于随机水平（0.544–0.696），而年龄标签较弱（0.354–0.397）。在22次事后选择的重新运行中，有九次的95%配对自助区间完全低于零，然而每一项源群组WER的绝对降低……（摘要原文在此处被截断）

    arXiv:2609.18533v1 Announce Type: new  Abstract: Automatic speech recognition (ASR) systems exhibit unequal error rates across speaker groups, motivating interventions on their internal representations. We ask whether speaker-linked attributes that are linearly readable from pretrained ASR encoders yield useful directions for reducing group word-error-rate (WER) gaps. Across Whisper-medium, HuBERT-large, and Wav2Vec2-large on Common Voice and the Speech Accent Archive, we probe every encoder layer for metadata-derived sex/gender, age, and native/accent labels; construct centroid and probe-derived directions; inject them at selected layers; and compare downstream probe trajectories with matched WER changes. Sex labels are highly decodable (best macro-F1 0.924--0.941), native/accent labels are also above chance (0.544--0.696), and age is weaker (0.354--0.397). Of 22 post-selected reruns, nine have 95% paired-bootstrap intervals entirely below zero, yet every absolute source-group WER red
    
[^52]: 使用统计机器学习实现英语与叙利亚语（东叙利亚方言）之间的机器翻译

    Machine Translation between English and Syriac (East Syriac Dialect) using Statistical Machine Learning

    [https://arxiv.org/abs/2609.18529](https://arxiv.org/abs/2609.18529)

    本研究开发了首个基于短语的英语到亚述语（东叙利亚方言）统计机器翻译模型，并从完整圣经中构建了包含38,847个句对的数据集，填补了这种濒危语言在自然语言处理领域的研究空白。

    

    联合国教科文组织将亚述语（叙利亚语）视为一种濒危语言。尽管世界各地的亚述人都在使用这种语言，但使用人口并不确定（介于50万至150万之间）。叙利亚语同时也是自然语言处理（NLP）领域研究最少的语言之一。尽管机器翻译（MT）在过去十年中取得了长足进展，但由于缺乏公开可用的语料库，以及叙利亚文字（特别是Madnkhaya文字）正字法的复杂性，该语言在计算语言学文献中完全被忽视。本研究使用Moses框架开发了首个用于英语到亚述语机器翻译的基于短语的统计机器翻译（SMT）模型。我们从完整的英语和叙利亚语圣经中创建了一个包含38,847个句对的数据集，将已有的新约数据集与通过PDF提取、使用自定义分割脚本和人工对齐审查从头构建的旧约数据集合并而成。

    arXiv:2609.18529v1 Announce Type: new  Abstract: UNESCO considers the Assyrian (Syriac) language an endangered language. Although Assyrians speak the language worldwide, the speaking population is uncertain (ranging from 500,000 to 1,500,000). Syriac is also one of the least studied languages in Natural Language Processing (NLP). Despite advances in Machine Translation (MT) over the past decade, the lack of publicly available corpora and the orthographic complexity of the Syriac script, specifically the Madnkhaya script, have left this language entirely ignored in the computational linguistics literature. This study develops the first phrase-based Statistical MT (SMT) model for English-to-Assyrian MT using the Moses framework. We created a dataset of 38,847 sentence pairs from the complete English and Syriac Bible, merging a pre-existing New Testament dataset with an Old Testament built from scratch through PDF extraction, using custom segmentation scripts and manual alignment review b
    
[^53]: 对齐、整合与激发：面向零样本语音大语言模型的高效词元级对齐

    Align, Integrate, and Fire: Efficient Token-Level Alignment for Zero-Shot SpeechLLMs

    [https://arxiv.org/abs/2609.18516](https://arxiv.org/abs/2609.18516)

    该论文提出了对齐连续积分-激发框架，利用动态时间规整对齐将连续声学帧压缩为目标文本的精确离散词元长度，在初始训练阶段完全绕过昂贵的大语言模型前向传播，实现了高效低成本的零样本语音处理。

    

    尽管大型语言模型在自然语言处理方面表现出色，但如何高效地将其能力扩展到语音输入仍是一个重大挑战。现有的构建语音大语言模型的方法通常依赖计算代价高昂的全模型微调，或者采用参数高效的投影器，但这些投影器存在词元序列长度低效以及全模型监督成本高昂的问题。在本文中，我们提出了对齐连续积分-激发框架，这是一个面向零样本语音处理的高效框架。我们的方法利用显式动态时间规整对齐，将连续声学帧动态压缩为目标文本的精确离散词元长度。这使得我们的初始训练阶段能够使用轻量级距离度量建立稳健的声学到语义的桥梁，完全绕过了计算代价高昂的大语言模型前向传播。对于后续的微调，我们提出了一种内存高效的知识蒸馏方法……（摘要在此处截断）

    arXiv:2609.18516v1 Announce Type: new  Abstract: While Large Language Models excel in natural language processing, efficiently extending their capabilities to spoken input remains a significant challenge. Existing methods for building SpeechLLMs often rely on computationally expensive full-model fine-tuning, or employ parameter-efficient projectors that suffer from inefficient token sequence lengths and costly full-model supervision. In this paper, we introduce Aligned Continuous Integrate-and-Fire, a highly efficient framework for zero-shot speech processing. Our method dynamically compresses continuous acoustic frames into the exact discrete token length of the target text utilizing explicit Dynamic Time Warping alignments. This allows our initial training stage to establish a robust acoustic-to-semantic bridge using lightweight distance metrics, entirely bypassing the computationally expensive LLM forward pass. For subsequent fine-tuning, we propose a memory-efficient knowledge dist
    
[^54]: 规模至关重要：面向捷克语HTML文档的基础模型

    Size Matters: Foundation Model for Czech HTML documents

    [https://arxiv.org/abs/2609.18494](https://arxiv.org/abs/2609.18494)

    提出仅1.54亿参数的紧凑基础模型HTML-LM，通过HTML感知训练与ModernBERT架构，在捷克互联网文档分类和回归任务上超越更大的模型，达到新的最先进水平。

    

    在高流量工业环境中创建通用、高质量的网页文档表示，需要兼具高性能与经济性的模型。然而，现有方法往往依赖大型模型、忽视HTML中固有的结构信息，或受限于较短的上下文窗口，从而限制了其处理真实网页的能力。我们提出了HTML-LM，一个仅含1.54亿参数的紧凑型基础模型，它通过HTML感知训练和基于ModernBERT的架构解决了上述局限。该模型在1亿个网页文档上使用多种目标进行训练，包括掩码语言建模、词袋预测以及从大语言模型进行的对比蒸馏。因此，HTML-LM在捷克互联网领域的分类和回归应用中创造了新的最先进水平，超越了更大的编码器模型和小型LLM。该模型已部署于产品环境中。

    arXiv:2609.18494v1 Announce Type: new  Abstract: Creating universal, high-quality representations of web documents in high-traffic industrial environments requires models that are both performant and economic. Existing approaches, however, often depend on large models, overlook the structural information inherent in HTML, or are constrained by short context windows, limiting their ability to process real-world web pages. We present HTML-LM, a compact foundation model with 154 million parameters that addresses these limitations through HTML-aware training and a ModernBERT-based architecture. It was trained on 100 million web documents using multiple objectives, including masked language modeling, bag-of-words prediction, and contrastive distillation from large language models. Consequently, HTML-LM sets a new state-of-the-art for classification and regression applications in the Czech Internet domain, surpassing both larger encoders and small-sized LLMs. The model is deployed in product
    
[^55]: ActionPiece：重新思考自回归视觉-语言-动作模型的动作分词

    ActionPiece: Rethinking Action Tokenization for Autoregressive Vision-Language-Action Models

    [https://arxiv.org/abs/2609.18487](https://arxiv.org/abs/2609.18487)

    该论文提出物理秩一致性（PRC）这一新指标，用于衡量动作分词器在压缩后能否保留动作之间的局部物理距离排序，弥补了均方误差等逐点重建指标无法反映动作调整关系失真的缺陷。

    

    arXiv:2609.18487v1 公告类型：交叉 摘要：动作分词器在自回归视觉-语言-动作（VLA）模型中扮演着核心角色，它既决定了策略训练的目标，也决定了从预测词元中恢复出的可执行命令。其保真度通常使用均方误差（MSE）等逐点重建指标来评估，然而微小的个体误差并不能完全刻画演示之间动作调整被保留的程度。压缩之后，相似的动作可能仍然聚集在某个代表性运动周围，而不同情境所需的调整却被削弱、扭曲甚至颠倒。我们提出物理秩一致性（PRC）来衡量分词在重建后对局部物理距离排序的保留程度。对解码后的动作进行评估为不同词表和解码器架构提供了共同的参照，以关系保真度的度量补充了逐点精度。我们进一……

    arXiv:2609.18487v1 Announce Type: cross  Abstract: Action tokenizers play a central role in autoregressive vision-language-action (VLA) models, determining both the targets for policy training and the executable commands recovered from predicted tokens. Their fidelity is commonly evaluated using pointwise reconstruction metrics such as mean squared error (MSE), yet small individual errors do not fully characterize how faithfully action adjustments across demonstrations are preserved. After compression, similar actions may still cluster around a representative motion, while the adjustments needed for different contexts are diminished, distorted, or even reversed. We introduce physical rank consistency (PRC) to measure how well tokenization preserves local physical distance rankings after reconstruction. Evaluating decoded actions provides a common reference across token vocabularies and decoder architectures, complementing pointwise accuracy with a measure of relational fidelity. We fur
    
[^56]: 分而治之：面向视频多模态情感分析的信息化有序空间中的瓶颈专家混合

    Divide and Conquer: Mixture-of-Bottleneck Experts in Informative Ordinal Space for Video-based Multimodal Sentiment Analysis

    [https://arxiv.org/abs/2609.18470](https://arxiv.org/abs/2609.18470)

    本文将视频多模态情感分析重新表述为有序回归问题，解耦为极性识别与强度预测两个子任务，并提出瓶颈混合专家框架，借助信息瓶颈学习为不同模态和任务学习紧凑且相关的表示，同时过滤冗余与噪声。

    

    基于视频的多模态情感分析（MSA）需要处理人类说话视频中的文本、音频和图像序列信息，然而现有方法往往无法以任务感知的方式整合各模态。大多数模型将视频情感预测视为单一任务，忽略了其有序性（ordinal nature），且其融合策略难以捕捉跨模态的多样化独特线索与协同线索。为解决这些局限性，我们采用分而治之的视角，将MSA重新表述为有序回归问题，并将其解耦为极性识别和强度预测两个子任务。受信息论启发，我们提出了瓶颈混合专家框架，为不同模态的极性特定专家和强度特定专家分配不同的潜在表示。通过信息瓶颈的学习，每个专家都能学到紧凑且与任务相关的表示，同时过滤掉冗余和噪声。

    arXiv:2609.18470v1 Announce Type: cross  Abstract: Video-based Multimodal sentiment analysis (MSA) must handle information from text, audio, and image sequence in human speaking videos, yet current methods often fail to integrate modalities with task awareness. Most models treat video sentiment prediction as a single task, overlooking its ordinal nature, and their fusion strategies struggle to capture diverse unique and synergic cues across modalities. To address these limitations, we adopt a divide-and-conquer perspective by reformulating MSA as an ordinal regression problem and decoupling it into polarity recognition and intensity prediction. Driven by information theory, we introduce a Mixture-of-Bottleneck (MoB) framework that assigns different latents to polarity- and intensity-specific experts for different modalities. With the learning of information bottleneck, each expert learns compact and task-relevant representations while filtering out redundancy and noise. A multimodal bo
    
[^57]: 通过潜在神经符号推理解耦长期记忆

    Disentangling Long-Term Memory via Latent Neuro-Symbolic Reasoning

    [https://arxiv.org/abs/2609.18461](https://arxiv.org/abs/2609.18461)

    提出LGM神经符号框架，利用稀疏自编码器将长期记忆解耦到连续潜在空间，根据每个查询动态构建潜在图，从而克服现有静态图记忆框架和平面检索方法无法捕捉上下文相关关系的问题。

    

    个性化智能体需要对长期历史交互进行推理，以同时推断显式偏好和隐式行为证据。早期的平面检索方法独立地对记忆片段进行评分，忽略了分布式信息；而当前的结构化记忆框架依赖于与查询无关的静态图，无法捕捉上下文相关的关系。至关重要的是，原始文本记忆本质上是纠缠且嘈杂的，使得细粒度个性化和跨会话推理在计算上变得难以实现。为此，我们提出了LGM，这是一种新颖的神经符号框架，将长期记忆解耦转移到连续潜在空间中。具体而言， 我们设计了一种基于稀疏自编码器的定制化潜在图构建方法，而非持久化固定图，它根据每个查询将历史交互映射为潜在记忆节点，并将记忆轨迹解耦为稀疏概念……

    arXiv:2609.18461v1 Announce Type: new  Abstract: Personalized agents are required to reason over long-term history interactions to infer both explicit preferences and implicit behavioral evidence. While early flat retrieval methods score memory fragments independently and neglect the distributed information, current structured memory frameworks rely on query-agnostic static graphs that fail to capture the context-dependent relations. Crucially, raw textual memories are inherently entangled and noisy, making fine-grained personalization and cross-session reasoning computationally prohibitive. To this end, we present LGM, a novel neuro-symbolic framework that shifts long-term memory disentanglement into a continuous latent space. Specifically, (i) instead of persisting fixed graphs, we design a tailored latent graph construction with a sparse autoencoder. Subject to each query, it maps historical interactions into latent memory nodes and disentangles the memory traces into sparse concept
    
[^58]: M-SQE：面向智能体技能使用中语言平等性的多语言技能质量估计

    M-SQE: Multilingual Skill Quality Estimation for Enhancing Language Equality in Agentic Skill Use

    [https://arxiv.org/abs/2609.18445](https://arxiv.org/abs/2609.18445)

    该论文针对智能体技能生态中低资源语言缺乏本地语言技能内容导致的语言不平等问题，提出了M-SQE框架，通过理论视角（内在质量）和行动视角（任务实用性）对检索到的多语言技能候选进行质量评估与统一打分。

    

    智能体技能是一种可复用的过程性文档，能够将大语言模型智能体的能力扩展到其参数记忆之外，已成为在真实世界任务中部署智能体的重要接口。围绕这一接口构建的社区维护技能库正在快速增长。然而，这一生态系统仍然高度以英语为中心：我们的审计发现，斯瓦希里语和印地语等低资源语言完全没有任何本地语言的技能内容，因此检索往往会返回与查询语言不同的技能，从而损害准确率和召回率。一个实用的解决方案是为检索合成本地语言的技能，但其质量可能不可靠，因此仅凭相关性筛选往往会得到相关却无法实际使用的候选。为解决这一问题，我们提出了M-SQE，一个检索后的多语言技能质量估计框架，通过理论视角评估候选的内在质量，通过行动视角评估其面向任务的实用性，并将两者统一……（原文摘要在此处截断）

    arXiv:2609.18445v1 Announce Type: new  Abstract: Agent skills, reusable procedural documents that extend LLM agents beyond their parametric memory, have become an important interface for deploying agents on real-world tasks. Community-maintained skill libraries built around this interface are growing rapidly. However, this ecosystem remains deeply English-centric: our audit finds that low-resource languages such as Swahili and Hindi have no in-language skill content, so retrieval often returns a skill written in a different language than the query, degrading accuracy and recall. A practical solution is to synthesize in-language skills for retrieval but the quality can be unreliable, so relevance in this setting alone often surfaces a related but unusable candidate. To address this, we propose M-SQE, a post-retrieval Multilingual Skill Quality Estimation framework that scores candidates via a Theory view for intrinsic quality and an Action view for task-grounded utility, unified into a 
    
[^59]: 规划还是即兴发挥？在开源模型与开源跨层转码器上对诗歌押韵规划位点进行压力测试

    Planning or Improvisation? Stress-Testing the Poetry Planning Site on Open Models and Open Cross-Layer Transcoders

    [https://arxiv.org/abs/2609.18440](https://arxiv.org/abs/2609.18440)

    该研究在四个开源模型和六个开源跨层转码器上对Claude“提前规划诗歌押韵”的发现进行压力测试，发现位置特异性效应普遍存在，但有效干预位置是紧邻输出的最后一个提示词元而非换行符，对“驻留于换行符的押韵规划”这一结论的普适性提出质疑。

    

    Lindsey等人（2025）报告称，Claude 3.5 Haiku会提前规划押韵：候选押韵词的特征在一行诗句写出之前的换行符处就已经激活，并且“抑制-注入”干预只有在应用于该位置时才能改变该行诗句的走向（见其论文图13）。我们在一块消费级GPU上，使用六个开源跨层转码器（CLT）对四个开源模型（参数量从0.6B到2.6B）展开测试，共覆盖七个实验组合，检验该结论的普适程度，并将原论断分解为三个部分：位置特异性（C1）、换行位点同一性（C2）以及驻留于换行符的规划（C3）。这是一项压力测试而非忠实复现：由于这些CLT无法获得归因图，特征是从解码器向量中自底向上发现的。结果显示，C1具有普适性——在全部实验组合中均成立，且在444个“提示-注入”配对中，所有247个具有可检测效应的配对均体现该效应——但有效干预位置是紧邻输出之前的最后一个提示词元，且只有两个实验组合达到了具有行为学意义的概率。C2和C3……

    arXiv:2609.18440v1 Announce Type: new  Abstract: Lindsey et al. (2025) report that Claude 3.5 Haiku plans rhymes: features for candidate rhyme words are active on the newline before a line is written, and a suppress-and-inject intervention redirects the line only when applied there (their Figure 13). We test how far this generalizes on seven cells crossing four open models (0.6B to 2.6B parameters) with six open cross-layer transcoders (CLTs), on one consumer GPU, decomposing the claim into position specificity (C1), newline site identity (C2), and a newline-resident plan (C3). This is a stress test rather than a faithful reproduction: attribution graphs are unavailable for these CLTs, so features are found bottom-up from decoder vectors. C1 generalizes, in every cell and in all 247 of 444 prompt-by-inject pairs with a detectable effect, but the effective position is the final prompt token, adjacent to emission, and only two cells reach behaviorally meaningful probabilities. C2 and C3 
    
[^60]: 面向高效多轮智能体微调的依赖感知轨迹精炼方法

    Dependency-Aware Trajectory Refinement for Efficient Multi-Turn Agent Fine-Tuning

    [https://arxiv.org/abs/2609.18417](https://arxiv.org/abs/2609.18417)

    该论文提出将多轮智能体轨迹建模为轮次级依赖DAG以识别并去除冗余轮次，用精炼后轨迹训练的模型在四个多模态问答基准上准确率最高提升1.7个百分点，同时推理消息数减少约40%、token数减少约48%。

    

    多轮智能体轨迹中通常包含冗余轮次（如失败的工具调用、并行的子查询、仅用于验证的步骤），这些冗余会推高训练与推理成本。我们提出将每条轨迹视为一个“轮次级依赖有向无环图（DAG）”，以揭示哪些轮次对最终答案具有全局关键支撑作用，并通过该DAG对轨迹进行精炼后再用于智能体微调。给定由大语言模型标注的DAG，这些编辑操作是确定性的且可解释的，并可选地进行改写。在这些精炼后的轨迹上训练的模型，在更低推理成本的情况下，一致地优于在原始轨迹上训练的模型。具体而言，在四个多模态问答基准上，我们的精炼方法相比普通SFT可将下游准确率提升最高1.7个百分点（相比LLM删除基线可提升5.7个百分点），同时将每样本推理消息数减少约40%，推理token数减少约48%，从而转化为显著的成本节约。

    arXiv:2609.18417v1 Announce Type: new  Abstract: Multi-turn agent trajectories often contain redundant rounds (failed tool calls, parallel sub-queries, verification-only steps) that inflate both training and inference cost. We propose viewing each trajectory as a \emph{round-level dependency DAG} that exposes which rounds are globally load-bearing for the final answer, and fine-tune agents on trajectories refined through this DAG. Given an LLM-annotated DAG, these edits are deterministic and interpretable, with optional rephrasing. Models trained on these refined trajectories consistently outperform those trained on the original trajectories at lower inference cost. Specifically, across four multi-modal QA benchmarks, our refinements improve downstream accuracy by up to $1.7$\,pp over vanilla SFT (and $5.7$\,pp over an LLM-deletion baseline) while reducing per-sample inference messages by up to approximately $40\%$ and inference tokens by up to approximately $48\%$, translating to subs
    
[^61]: 情绪体验、表达与感知：多模态社交媒体帖子的情绪分析

    Emotion Experience, Expression, and Perception: Emotion Analysis on Multimodal Social Media Posts

    [https://arxiv.org/abs/2609.18385](https://arxiv.org/abs/2609.18385)

    该论文提出了多模态多情绪模型数据集Mult2EMo，通过同时收集作者和读者对帖子及其触发事件的情绪标注，研究了作者情绪体验、帖子内容与读者重建情绪表达能力之间的关系，弥补了以往研究忽视图像模态和情绪触发事件的不足。

    

    情绪是人类交流的重要方面，尤其是在社交媒体上，作者经常结合文本和图像来表达他们的情绪。然而，以往关于社交媒体帖子情绪分析的工作，在衡量读者能够多大程度重建作者意图方面，忽视了两个重要方面：（1）图像模态——大多数研究仅关注文本；（2）触发所表达情绪的真实世界事件，及其与帖子内容的关系。因此，我们研究了以下两者之间的关系：（a）作者对促使他们撰写社交媒体帖子的事件的体验，以及（b）帖子的内容，并重点关注读者重建该情绪表达的能力。为此，我们引入了多模态多情绪模型数据集 Mult2EMo，该数据集通过收集作者和读者对帖子及其触发事件的标注而构建。我们发现重建……

    arXiv:2609.18385v1 Announce Type: new  Abstract: Emotions are an essential aspect of human communication, particularly on social media, where authors frequently combine text and images to convey their emotions. Yet prior work on emotion analysis of social media posts has overlooked two important aspects in regard to measuring how well readers can reconstruct the authors' intent: (1)~the image modality, with most work focusing solely on text, and (2)~the real-world events that trigger the expressed emotions, and their relationship to the post content. We therefore study the relation between (a) the author's experience of the event that caused them to write a social media post and (b) the content of the post, with a focus on readers' capability to reconstruct that emotion expression. To do that, we introduce the Multimodal Multi-Emotion-Model dataset Mult2EMo, created by collecting annotations from both authors and readers on the posts and their triggering events. We find that reconstruc
    
[^62]: 市场信号注入：对大语言模型定价智能体的对抗性上下文操纵

    Market Signal Injection: Adversarial Context Manipulation of LLM Pricing Agents

    [https://arxiv.org/abs/2609.18357](https://arxiv.org/abs/2609.18357)

    提出了“市场信号注入”（MSI）攻击方法，证明无需明确指令、仅通过操纵数据格式、竞争对手排序和市场评论等呈现方式，即可显著改变LLM定价智能体的行为并影响利润与消费者剩余，且更大的模型并不必然更稳健。

    

    大语言模型（LLM）定价智能体可能会对市场数据的呈现方式作出反应，即使其数值保持不变。我们提出了“市场信号注入”（MSI）攻击方法，该攻击通过操纵数值格式、竞争对手排序或定性市场评论来实施，而无需发出明确指令。我们在模拟的伯川德双寡头和三寡头市场中评估了九个开源权重模型，并在双寡头市场中评估了三个专有模型。基于情感的攻击产生了最大的行为变化，这些变化会传播到其他公司，并改变利润和消费者剩余。易感性因模型家族而异，更大的模型并不必然更加稳健。匹配的中性文本对照组和基于规则的智能体支持了在模拟固定需求参数下基于框架效应的行为变化解释。情节保留探测方法能够在所有十一个重新评估的模型中区分基线状态与受攻击状态的激活。

    arXiv:2609.18357v1 Announce Type: new  Abstract: Large language model (LLM) pricing agents may respond to how market data is presented, even when its numerical values remain unchanged. We introduce market signal injection (MSI), an attack that manipulates numerical formatting, competitor ordering, or qualitative market commentary without issuing explicit instructions. We evaluate nine open-weight models in simulated Bertrand duopoly and triopoly markets and three proprietary models in duopoly markets. Sentiment-based attacks produce the largest behavioral shifts, which propagate to other firms and alter profits and consumer surplus. Susceptibility varies across model families, and larger models are not consistently more robust. Matched neutral-text controls and a rule-based agent support a framing-based account of these shifts under the fixed demand parameters of our simulation. Episode-held-out probes distinguish baseline from attacked activations in all eleven re-evaluated model--con
    
[^63]: 忠实却合谋：为什么思维链监控无法检测寡头竞争下大语言模型定价代理的合谋行为

    Faithful yet Collusive: Why Chain-of-Thought Monitoring Cannot Detect Collusion in LLM Pricing Agents under Oligopolistic Competition

    [https://arxiv.org/abs/2609.18346](https://arxiv.org/abs/2609.18346)

    该论文开发了因果图发散框架，分别衡量LLM定价代理的结构忠实性与意图忠实性，发现合谋行为与思维链忠实性相互分离，从而证明仅靠CoT监控无法检测和防范算法合谋。

    

    部署为自主定价代理的大语言模型（LLM）可能通过默契协调维持超越竞争水平的价格。我们开发了一个因果图发散框架，用于分别衡量伯特兰竞争中LLM定价代理的结构忠实性和意图忠实性。在双寡头和三寡头市场条件下对九个LLM进行测试后发现，合谋行为与思维链（CoT）忠实性在两个维度上均出现分离：最合谋的模型能够准确报告其合作意图，但其推理过程在结构上并不忠实；而结构上最忠实的模型在两种市场结构下都维持了超纳什均衡的定价。这些发现表明，仅依靠CoT监控无法作为防范算法合谋的独立保障措施。

    arXiv:2609.18346v1 Announce Type: new  Abstract: Large language models (LLM) deployed as autonomous pricing agents may sustain supracompetitive prices through tacit coordination. We develop a causal graph divergence framework that separately measures structural faithfulness and intent faithfulness of LLM pricing agents in Bertrand competition. Across nine LLMs under duopoly and triopoly conditions, collusive behavior and chain-of-thought (CoT) faithfulness dissociate along both dimensions: the most collusive model accurately reports cooperative intent yet reasons structurally unfaithfully, while the most structurally faithful model sustains supra-Nash pricing under both market structures. These findings establish that CoT monitoring alone cannot serve as a standalone safeguard against algorithmic collusion.
    
[^64]: 理解本地服务市场中的AI服务提供者推荐

    Understanding AI Provider Recommendations in Local Service Markets

    [https://arxiv.org/abs/2609.18341](https://arxiv.org/abs/2609.18341)

    该研究首次系统审计了AI助手在本地服务市场（如医疗、金融顾问）中的服务提供者推荐质量，发现无搜索时模型大量捏造推荐（仅4-11%对应真实服务提供者），而开启网络搜索后推荐准确率大幅提升至64-71%。

    

    当某人询问AI助手应该看哪位医生、或将积蓄托付给哪家公司时，得到的答案是一种推荐。我们对美国100个最大都市区内四个有官方注册记录支持的服务领域中的AI服务提供者推荐进行了审计，将每一条推荐与该领域的官方注册记录（Medicare临床医生与机构记录，以及SEC投资顾问披露信息）进行比对，并在三种条件下进行测试：开源权重模型、不带网络搜索的专有模型，以及带搜索功能的同一专有模型。在没有搜索的情况下，两种模型在网页覆盖薄弱的领域中大量捏造推荐。开源权重模型推荐的医生中仅有4%、专有模型推荐的医生中仅有11%能在被查询的城市中匹配到真实临床医生，且开源模型的匹配仅是名字上的巧合：其匹配到的临床医生作为初级保健医生的可能性并不比从注册记录中随机抽取的名字更高。在有搜索的情况下，64-71%的推荐……

    arXiv:2609.18341v1 Announce Type: cross  Abstract: When someone asks an AI assistant which doctor to see or which firm to trust with their savings, the answer is a referral. We audit AI provider recommendations in four registry-backed service domains across the 100 largest U.S. metropolitan areas, matching every recommendation against the official registry for its domain (Medicare clinician and facility records, and SEC adviser disclosures), under three conditions: an open-weight model, a proprietary model without web search, and the same proprietary model with search. Without search, both models largely fabricate recommendations in the domains the web covers thinly. Only 4% of the open-weight model's recommended doctors and 11% of the proprietary model's match a clinician in the queried city, and the open-weight matches are name coincidences: its matched clinicians are no likelier to be primary-care doctors than names drawn at random from the registry. With search, 64-71% of recommend
    
[^65]: 注意力分散作为大语言模型幻觉的诊断信号

    Attention Dispersion as a Diagnostic Signal for Hallucination in Large Language Models

    [https://arxiv.org/abs/2609.18320](https://arxiv.org/abs/2609.18320)

    该论文提出一种无监督的注意力分散度量方法，通过监测大语言模型内部注意力机制的时间波动性来检测幻觉，摆脱了对输出校准的依赖，在数学推理基准上相比基于输出的基线方法AUC提升高达0.076。

    

    大语言模型（LLM）经常出现幻觉现象，这成为其在复杂推理任务中可靠性的主要障碍。虽然传统的检测方法依赖于基于输出的置信度指标，但这些logits经常被现代对齐技术错误校准。在本文中，我们研究了内部注意力机制的时间波动性，作为一种不依赖输出校准的幻觉替代诊断信号。通过引入一种无监督的注意力分散度量，我们证明认知不确定性会在中间层中留下可测量的痕迹，其中注意力熵的峰值与推理崩溃相关联。我们使用Qwen2.5模型家族（1.5B和3B参数）在数学推理基准（GSM8K和MATH-500）上评估了我们的方法，发现在所有测试条件下，相比基于输出的基线方法，AUC获得了高达+0.076的统计学显著提升。

    arXiv:2609.18320v1 Announce Type: new  Abstract: Large Language Models (LLMs) frequently exhibit hallucinations, presenting a major barrier to reliability in complex reasoning tasks. While traditional detection methods rely on output-based confidence metrics, these logits are often miscalibrated by modern alignment techniques. In this paper, we investigate the temporal volatility of internal attention mechanisms as an alternative diagnostic signal for hallucination that does not depend on output calibration. By introducing an unsupervised metric for attention dispersion, we show that epistemic uncertainty leaves a measurable trace within intermediate layers, where spikes in attention entropy are associated with reasoning breakdowns. We evaluate our approach on mathematical reasoning benchmarks (GSM8K and MATH-500) using the Qwen2.5 model family (1.5B and 3B parameters), finding statistically significant AUC improvements of up to +0.076 over output-based baselines across all tested cond
    
[^66]: 基于知识图谱的增强与检索增强生成在文化相关问题回答中的对比

    Knowledge-Graph Based Augmentation versus Retrieval Augmented Generation for Cultural-Related Question Answering

    [https://arxiv.org/abs/2609.18317](https://arxiv.org/abs/2609.18317)

    该论文在文化问答数据集LatamQA上对比了基于知识图谱的Graph-RAG与标准RAG，发现使用KGGen自动构建的知识图谱驱动的G-Retriever性能可与RAG媲美，并可将基础LLM的错误率降低72%至78%。

    

    大型语言模型（LLM）存在长尾知识缺陷：文化特定的事实，尤其是涉及拉丁美洲等代表性不足地区的信息，在预训练语料中出现频率过低而难以被可靠地记忆。检索增强生成（RAG）通过将生成过程建立在检索到的外部文本上来解决这一问题，但知识图谱（KG）等结构化替代方案能够对进入上下文的内容提供更严格的控制，并在可解释性和可更新性方面具有潜在优势。我们在LatamQA——一个涵盖八个主题类别的文化基础多选题数据集——上将Graph-RAG与标准RAG进行基准对比。知识图谱使用KGGen（一种新兴的开放域抽取器）从维基百科文章端到端构建，在主要设置下无需人工整理。G-Retriever与RAG具有竞争力，使用标准知识图谱可将基础LLM的错误率降低72%，使用感知基准的变体则可降低78%，且与RAG的差距正在缩小。

    arXiv:2609.18317v1 Announce Type: cross  Abstract: Large language models (LLMs) suffer from a long-tail deficit: culturally specific facts, particularly those concerning underrepresented regions such as Latin America, appear too rarely in pretraining corpora to be reliably memorized. Retrieval-Augmented Generation (RAG) addresses this by grounding generation in external text, but structured alternatives such as Knowledge Graphs (KGs) offer tighter control over what enters the context, along with potential gains in explainability and updatability. We benchmark Graph-RAG against standard RAG on LatamQA, a culturally grounded multiple-choice dataset spanning eight thematic categories. The graphs are built end-to-end from Wikipedia articles with KGGen, a recent open-domain extractor, without manual curation in our main setting. G-Retriever is competitive with RAG and reduces the error of the base LLM by 72\% with a standard KG and 78\% with a benchmark-aware variant, the gap to RAG narrowi
    
[^67]: SEA-LION-v4.8：技术报告

    SEA-LION-v4.8: A Technical Report

    [https://arxiv.org/abs/2609.18310](https://arxiv.org/abs/2609.18310)

    基于NVIDIA Nemotron 3构建的SEA-LION-v4.8东南亚语言模型家族，通过持续预训练、监督微调和在线同策略蒸馏，显著提升了七种东南亚语言在指令遵循、推理和理解任务上的表现。

    

    我们介绍了Nemotron-SEA-LION-v4.8，这是一个基于NVIDIA Nemotron 3构建的东南亚语言一体化网络模型家族。该家族包括30B-A3B和120B-A12B两个模型，同时提供持续预训练的基础检查点和后训练变体。我们使用东南亚语言、推理、代码和多语言平行数据集对模型进行适配，随后通过监督微调和在线同策略蒸馏进行后训练。在SEA-HELM基准测试中，30B-A3B模型将SEA综合得分从46.06提升至51.57，而120B-A12B模型则从49.30提升至63.44。在七种东南亚语言的指令遵循、自然语言推理和自然语言理解方面均取得了最显著的提升。

    arXiv:2609.18310v1 Announce Type: new  Abstract: We introduce Nemotron-SEA-LION-v4.8, a family of Southeast Asian Languages in One Network (SEA-LION) built upon NVIDIA Nemotron 3. The family includes 30B-A3B and 120B-A12B models, with both continued-pretrained base checkpoints and post-trained variants. We adapt the models using Southeast Asian, reasoning, code, and multilingual parallel datasets, followed by post-training with supervised fine-tuning and online on-policy distillation. On SEA-HELM, the 30B-A3B model improves the overall SEA score from 46.06 to 51.57, while the 120B-A12B model improves from 49.30 to 63.44. The strongest gains are observed in instruction following, natural language reasoning, and natural language understanding across seven Southeast Asian languages.
    
[^68]: 回滚世界，保留反思：面向长程LLM智能体的回滚诱导反思

    Rollback the World, Keep the Reflection: Rollback-Induced Reflection for Long-Horizon LLM Agents

    [https://arxiv.org/abs/2609.18304](https://arxiv.org/abs/2609.18304)

    提出了回滚诱导反思（RIR）统一恢复框架，在将LLM智能体回滚到选定先前状态的同时，保留从被放弃轨迹中提炼的可复用知识，解决了长程任务中错误累积且难以可靠恢复的问题。

    

    大语言模型（LLM）智能体越来越多地通过多步环境交互来处理长程任务，然而单个错误的动作可能会改变后续的状态和观测，导致错误随时间不断累积。现有方法要么在不修复已改变环境状态的情况下纠正上下文，要么在恢复早期状态的同时丢弃有用的经验，这使得既消除失败条件又避免重复过去的错误变得困难。我们认为，可靠的恢复应被视为一个回滚边界控制问题，即联合决定何时干预、从何处恢复，以及哪些信息应在恢复过程中保留。基于这一观点，我们提出了回滚诱导反思（RIR），这是一个统一的恢复框架，它将执行恢复到选定的先前状态，同时保留从被放弃轨迹中提炼出的可复用知识，以指导后续决策。我们进一步刻画……

    arXiv:2609.18304v1 Announce Type: new  Abstract: Large language model (LLM) agents increasingly tackle long-horizon tasks through multi-step environment interaction, yet a single erroneous action can alter subsequent states and observations, causing errors to compound over time. Existing methods either correct the context without repairing altered environment states or restore earlier states while discarding useful experience, making it difficult to both eliminate failure conditions and avoid repeating past mistakes. We argue that reliable recovery should instead be treated as a rollback-boundary control problem that jointly determines when to intervene, where to resume, and what information should survive recovery. Based on this view, we propose Rollback-Induced Reflection (RIR), a unified recovery framework that restores execution to a selected prior state while carrying forward reusable knowledge distilled from the abandoned trajectory to guide subsequent decisions. We further chara
    
[^69]: 基于关系引导的大语言模型用例建模

    Relationally Guided Use Case Modeling with LLMs

    [https://arxiv.org/abs/2609.18291](https://arxiv.org/abs/2609.18291)

    该论文提出FlowGen框架，利用大语言模型进行语义信息提取并构建语义关系图，实现完整用例流的自动化构建，涵盖基本流生成、分支点预测和基于条件的备选流生成。

    

    用例流是用例建模的重要组成部分，因为它支持下游软件工程活动，包括需求分析、架构设计与详细设计以及测试用例生成。然而，手动构建用例流成本高昂且需要大量专业知识，而现有的自动化方法仍然难以保持语义一致性、控制流逻辑、数据流逻辑以及预期的系统边界，尤其是在识别分支点和生成备选流方面。为了解决这一问题，我们提出了用于完整用例流构建的FlowGen。FlowGen使用基于大语言模型的语义信息处理（SIP）来提取语义元素，构建由增强R-GAT编码的语义关系图（SRG）用于基本流生成（BFGen），并通过分支点预测（BPP）进一步支持分支点识别，以及通过基于分支条件的备选流生成（AFGen）生成备选流。在13个公开……（摘要在此处被截断）

    arXiv:2609.18291v1 Announce Type: cross  Abstract: Use case flows are important elements of use case modeling because they support downstream software engineering activities, including requirements analysis, architectural and detailed design, and test case generation. However, constructing them manually is costly and expertise-intensive, while existing automated approaches still struggle to preserve semantic consistency, control-flow logic, data-flow logic, and the intended system boundary, especially when identifying branch points and generating alternative flows. To address this problem, we propose FlowGen for complete use case flow construction. FlowGen uses LLM-based Semantic Information Processing (SIP) to extract semantic elements, constructs a Semantic Relational Graph (SRG) encoded by an enhanced R-GAT for basic flow generation (BFGen), and further supports branch point prediction through BPP and branch-conditioned alternative flow generation through AFGen. Evaluations on 13 pu
    
[^70]: 匈牙利制造：关于生成式语言模型性能的评论

    Made in Hungary: Comments on the performance of generative language models

    [https://arxiv.org/abs/2609.18284](https://arxiv.org/abs/2609.18284)

    本文对匈牙利三项生成式语言模型开发倡议进行批判性评述，指出其评估协议可靠性存疑、存在数据污染，且训练流程未达当前最佳实践标准。

    

    近年来，匈牙利出现了三项开发生成式语言模型的倡议，其动机相同：对于匈牙利语而言，此前不存在具备相应能力的模型，或者现有的以英语为中心的模型能力有限。然而，对相关研究的详细考察揭示了若干方法学上的局限。首先，评估协议的可靠性值得怀疑。与Csibi等人[2026]的发现相反，在推荐的推理设置下进行评估显示，Qwen3-4B取得了比其匈牙利语适配版本Racka-4B更高的分数。在Yang等人[2025d]和Szentmihályi等人[2025]的工作中，数据污染问题显而易见，可能导致所报告的结果产生偏差。其次，这些训练流程在语料库筛选和数据混合方面未达到当前最佳实践，这有可能将大量计算资源浪费在低质量数据上。

    arXiv:2609.18284v1 Announce Type: new  Abstract: In recent years, three initiatives have emerged to develop generative language models in Hungary. The motivation behind them is the same. For Hungarian, no model with the given capability existed, or existing English-centric models offered limited proficiency. A detailed examination of the corresponding studies, however, reveals several methodological limitations. First, the reliability of the evaluation protocols is questionable. Contrary to the findings of Csibi et al. [2026], evaluation under the recommended inference settings shows that Qwen3-4B achieves higher scores than Racka-4B, its Hungarian-adapted version. Data contamination is evident in the work of Yang et al. [2025d] and Szentmih\'alyi et al. [2025], potentially biasing the reported results. Second, the training pipelines fall short of current best practices in corpus curation and data mixture, which risks wasting substantial compute on low-quality data. The lack of control
    
[^71]: 太好而失真？诊断并缩小AI偏好与真实用户参与度之间的差距

    Too Good to Be Real? Diagnosing and Reducing the Gap Between AI Preference and Real User Engagement

    [https://arxiv.org/abs/2609.18282](https://arxiv.org/abs/2609.18282)

    该研究基于知乎、Quora和Reddit的117万条回答发现，大语言模型存在“逻辑过度绑定”倾向，即偏好增加逻辑结构，而真实用户参与度更依赖情感与表达显著性，并提出本体掩码推理自编码（OMRA）方法来缩小这一差距。

    

    大语言模型越来越多地被用于生成和评估在线内容，然而它们所认为的与更高参与度相关的内容质量，是否与真实用户的实际反应相匹配，仍不清楚。我们利用来自知乎、Quora和Reddit的25,978个问题下的117万个回答来研究这一问题，在四个题内参与度水平上比较真实平台回答与AI生成回答。我们提出了本体论偏好测量，从逻辑、情感和表达三个维度来表征回答。我们发现AI偏好与真实用户参与度之间存在系统性差距：随着目标参与度的提高，大语言模型会添加更多的显式逻辑结构，而真实用户的参与度则更强地与情感和表达上的显著性相关联。我们将这种倾向称为逻辑过度绑定。基于这一诊断，我们提出了本体掩码推理自编码，这是一种受控干预方法，通过掩码并重构……

    arXiv:2609.18282v1 Announce Type: new  Abstract: Large language models are increasingly used to generate and evaluate online content, yet it remains unclear whether the qualities they associate with higher engagement match what real users respond to. We study this question using 1.17 million answers to 25,978 questions from Zhihu, Quora, and Reddit, comparing real platform answers and AI-generated answers across four within-question engagement levels. We introduce Ontological Preference Measurement, which represents answers along three dimensions: logic, affect, and expression. We find a systematic gap between AI preference and real user engagement: as target engagement increases, LLMs add more explicit logical structure, while real user engagement is more strongly associated with affective and expressive salience. We call this tendency logic overbinding. Based on this diagnosis, we propose Ontology-Masked Reasoning Autoencoding (OMRA), a controlled intervention that masks and reconstr
    
[^72]: 我来编码还是AI编码：课堂观察中AI评分的比较评估

    I code or AI code: A comparative evaluation of AI-rated scores in classroom observations

    [https://arxiv.org/abs/2609.18274](https://arxiv.org/abs/2609.18274)

    本研究评估了GPT-5在香港幼儿园课堂中应用CLASS框架对师幼互动评分的可行性，发现AI评分与人类评分者在情感支持领域、尤其是质量反馈维度上具有较高的一致性。

    

    课堂观察被广泛认为是建立教育质量基准和指导教学改进的关键工具，但其仍然资源密集且依赖于训练有素的观察者。本研究评估了使用大语言模型（GPT-5模型）对幼儿课堂中师幼互动进行评分的可行性，并以人类评分者作为基准。该研究分析了来自香港30所幼儿园38个教室的87个视频记录观察。利用观察转录文本，AI模型被配置为应用完整的课堂评估评分系统（CLASS）框架。随后，通过考察CLASS各领域和维度平均分的相关性和差异，将AI评分与人类评分进行比较。结果显示，AI与人类评分者在情感支持领域，尤其是质量反馈维度上表现出更高的一致性，该维度衡量教师如何使用（摘要在此处截断）

    arXiv:2609.18274v1 Announce Type: cross  Abstract: Classroom observations are widely recognized as a key tool for establishing benchmarks of education quality and guiding pedagogical improvement, yet they remain resource-intensive and dependent on trained observers. This study evaluated the feasibility of using a LLM (GPT-5 model) to score teacher-child interactions in early childhood classrooms, benchmarked against human raters. The study analyzed 87 video-recorded observations from 38 classrooms across 30 kindergartens in Hong Kong. Using observation transcripts, the AI model was configured to apply the full Classroom Assessment Scoring System (CLASS) framework. AI-rated scores were then compared with human ratings by examining correlations and differences in mean scores of the CLASS domains and dimensions. The results showed greater convergence between AI and raters for the Emotional Support domain and, in particular, the Quality of Feedback dimension, which captures how teachers us
    
[^73]: M²Tok：面向视觉-语言-动作模型的多头多码本离散动作分词器

    ${M}^2$Tok: Multi-head Multi-codebook Discrete Action Tokenization for Vision-Language-Action Models

    [https://arxiv.org/abs/2609.18259](https://arxiv.org/abs/2609.18259)

    提出 M²Tok，一种多头多码本离散动作分词器，通过将潜在动作特征分解为多个头并采用多个码本以最小化重构误差，突破“离散化瓶颈”，从而提升视觉-语言-动作模型的控制性能。

    

    近期的研究进展已成功将自回归语言模型适配到处理多模态信号，例如图像和动作。由于原始动作信号是连续的，有效的分词化对于将高维输入映射为紧凑的离散标记以进行自回归处理至关重要。然而，现有的离散动作分词器往往存在较高的重构损失，无法保留精确控制所需的细粒度动态信息。这种“离散化瓶颈”显著限制了下游视觉-语言-动作（VLA）模型的性能上限。为解决这一问题，我们提出了 M²Tok，一种多头多码本动作分词器，旨在最小化重构误差并提升策略性能。我们的方法引入了两项关键的结构创新：（1）我们将潜在动作特征分解为多个头，使模型能够隐式地将特定的头与不同的语义信息相关联

    arXiv:2609.18259v1 Announce Type: cross  Abstract: Recent advancements have successfully adapted autoregressive language models to process multimodal signals, such as images and actions. Since raw action signals are continuous, effective tokenization is essential to map high-dimensional inputs into compact discrete tokens for autoregressive processing. However, existing discrete action tokenizers often suffer from high reconstruction loss, failing to preserve the fine-grained dynamics required for precise control. This ``discretization bottleneck'' significantly limits the performance ceiling of downstream Vision-Language-Action (VLA) models. To address this, we propose $\mathcal{M}^2$Tok, a Multi-head Multi-codebook Action Tokenizer designed to minimize reconstruction error and enhance policy performance. Our approach introduces two key structural innovations: (1) we decompose the latent action features into multiple heads, enabling the model to implicitly align specific heads with di
    
[^74]: 超越准确性：程序化痕迹如何改变LLM监督者的决策标准

    Beyond Accuracy: How Procedural Traces Shift the Decision Criterion of LLM Overseers

    [https://arxiv.org/abs/2609.18204](https://arxiv.org/abs/2609.18204)

    研究发现，程序化痕迹并不能提升LLM监督者的错误检测能力，反而会使其决策标准向拒绝方向偏移，导致对正确工作的误报增加，且痕迹越详细这种偏见越强。

    

    组织越来越多地使用监督闭环，其中一个大型语言模型（LLM）审计另一个模型的输出，并同时审查其声称步骤的程序化痕迹。关于此类“LLM作为评判者”流程的一个常见担忧是，详细的痕迹会使监督者变得轻信。我们运用信号检测理论，在19项合规任务上审计了五个LLM监督者（共分析了4,551个判断），仅改变痕迹的细节程度和证据标注方式。当反驳性证据始终可见时，错误检测率保持在接近上限的水平。相反，精细的痕迹会使监督者的决策标准向拒绝方向偏移，增加了易受影响监督者的误报。在没有选项标签的情况下，经人工验证的理由编码显示，约60%的误报归因于无法将证据与相应选项关联起来。添加标签后，这一陈述理由被消除了，但在那些监督者中，对正确工作的残留拒绝仍然存在，并随痕迹细节的增加而上升。因此，程序化痕迹实际上作为治理工件，塑造着（监督者的判断行为）。

    arXiv:2609.18204v1 Announce Type: cross  Abstract: Organizations increasingly use oversight loops where one large language model (LLM) audits another's outputs alongside procedural traces of claimed steps. A common concern about such LLM-as-a-judge pipelines is that detailed traces make overseers gullible. Using signal detection theory, we audit five LLM overseers on 19 compliance tasks (4,551 analyzed judgments), varying only trace detail and evidence labeling. With disconfirming evidence always visible, error detection remains near ceiling. Instead, elaborate traces shift the decision criterion toward rejection, increasing false alarms in susceptible overseers. Without option labels, human-validated reason coding shows about 60% of false alarms cite an inability to tie evidence to its option. Labels eliminate this stated reason, yet residual rejection of correct work persists in those overseers and rises with trace detail. Procedural traces thus act as governance artifacts that shape
    
[^75]: Behavior2Value：面向电子商务行为中消费者价值测量的LLM基准测试与能力增强

    Behavior2Value: Benchmarking and Empowering LLMs for Consumer Value Measurement from E-commerce Behaviors

    [https://arxiv.org/abs/2609.18203](https://arxiv.org/abs/2609.18203)

    该论文提出了行为到价值（B2V）任务，构建了首个电子商务消费价值分类体系（ECVT）和基于真实淘宝行为日志的B2V-Bench基准数据集，实现了从电子商务行为轨迹中识别和测量消费者价值观，并据此赋能大语言模型。

    

    人类价值观是塑造人类行为的深层动机取向。在电子商务领域，价值观揭示了用户购买决策背后稳定的驱动因素。与短期兴趣相比，消费者价值观能更好地解释用户在购买前如何评价产品。然而，消费者价值观通常隐含在复杂且碎片化的行为轨迹中，使得从电子商务行为中进行价值测量在很大程度上仍是一个未被充分探索的领域。为此，我们提出了行为到价值（Behavior-to-Value，B2V）任务，旨在从电子商务行为轨迹中识别消费者价值观。围绕这一任务，我们首先构建了电子商务消费价值分类体系（ECVT），并基于匿名的淘宝行为日志引入了B2V-Bench——首个B2V数据集与基准。B2V-Bench由真实世界的购买决策情节组成，涵盖25种购买行为类型，以及每个情节中体现的相应消费者价值取向。

    arXiv:2609.18203v1 Announce Type: new  Abstract: Human values are deep motivational orientations that shape human behaviors. In e-commerce, they reveal the stable drivers behind users' purchase decisions. Compared with short-term interests, consumer values better explain how users evaluate products before purchase. However, consumer values are often implicit in complex and fragmented behavioral trajectories, leaving value measurement from e-commerce behaviors largely underexplored. To this end, we propose the Behavior-to-Value (B2V) task, which aims to identify consumer values from e-commerce behavioral trajectories. Centered on this task, we first construct the E-commerce Consumption Value Taxonomy (ECVT) and introduce B2V-Bench, the first B2V dataset and benchmark, based on anonymized Taobao behavioral logs. B2V-Bench consists of real-world purchase decision episodes, covering 25 types of purchase behaviors, along with corresponding consumer value orientations manifested in each epis
    
[^76]: T-SANDHI：面向低资源台湾闽南语语音识别的声调连读感知自适应网络与解耦混合注入

    T-SANDHI: Tone Sandhi-aware Adaptive Network with Decoupled Hybrid Injection for Low-resource Taiwanese Hokkien Speech Recognition

    [https://arxiv.org/abs/2609.18194](https://arxiv.org/abs/2609.18194)

    该论文发现台湾闽南语语音识别的真正瓶颈并非连读变调本身，而是变调与本调之间的局部混淆，并提出T-SANDHI模型，通过在冻结的Whisper骨干上显式解耦表层声学与词汇意图，结合词典引导的多任务学习和动态门控混合注入模块，有效提升了低资源台湾闽南语的语音识别性能。

    

    在台湾闽南语自动语音识别（ASR）中，先前的研究通常将连读变调视为主要挑战，其假设是模型无法处理隐式的音系变化。然而，我们在台湾闽南语上的实验表明，语音基础模型实际上能够有效处理连读变调变化，真正的性能瓶颈源于这些变调与保留本调（连读前调）之间的局部混淆。为解决这一问题，我们提出T-SANDHI，在冻结的Whisper骨干网络基础上，将表层声学与底层词汇意图进行显式解耦。利用由文本导出伪标签驱动的词典引导多任务学习结构，我们的轻量级混合注入模块通过动态门控整合相互独立的本调语音流与变调语音流。在TAT-MOE语料库和两个盲测试集上的广泛评估表明，这种显式解耦有效解决了……（原文摘要在此处截断）

    arXiv:2609.18194v1 Announce Type: new  Abstract: In Taiwanese Hokkien automatic speech recognition (ASR), prior studies often treat tone sandhi as a major challenge under the assumption that models fail to process implicit phonological variations. However, our experiments on Taiwanese Hokkien reveal that speech foundation models actually handle tone sandhi variations effectively, and the real performance bottleneck stems from a localized confusion between these variations and retained citation tones. To address this, we propose T-SANDHI to explicitly decouple surface acoustics from underlying lexical intent on top of a frozen Whisper backbone. Using a lexicon-guided multi-task learning structure driven by text-derived pseudo labels, our lightweight hybrid injection module integrates independent citation and sandhi phonetic streams via dynamic gating. Extensive evaluation on the TAT-MOE corpus and two blind test sets demonstrates that this explicit disentanglement effectively resolves t
    
[^77]: TeochewBench：一个人工审核的潮州话汉字翻译基准

    TeochewBench: A Human-Reviewed Benchmark for Teochew Hanzi Translation

    [https://arxiv.org/abs/2609.18156](https://arxiv.org/abs/2609.18156)

    提出了首个经人工审核的潮州话汉字翻译基准 TeochewBench，包含300条涵盖五大类别的潮州话表达，用于评估大型语言模型在潮州话与普通话、英语之间双向翻译的能力。

    

    潮州话拥有庞大的使用人群，并展现出独特的词汇、句法和语用特征，然而用于评估大型语言模型的潮州话文本资源仍然有限。我们提出了 TeochewBench，这是一个经过人工审核的基准数据集，包含 300 条潮州话汉字表达，用于评估从潮州话汉字到普通话和英语的翻译。该数据集涵盖五个类别：基础词汇；日常句子；潮州话特有表达；语气、礼貌和语境；以及习语、歧义和文化特有表达。一位主要的潮州话母语审核者逐一检查了所有条目并进行了必要的修改，另外两位潮州话母语者对部分选定条目进行了核实。我们的主要评估涵盖 11 个官方通用后训练模型，在审核后的数据集上进行双向翻译，共产生 6,600 个预测结果。两个官方基础模型检查点额外提供了 1,200 个预测作为补充。

    arXiv:2609.18156v1 Announce Type: new  Abstract: Teochew has a substantial speaker community and exhibits distinctive lexical, syntactic, and pragmatic features, yet textual resources for evaluating large language models remain limited. We present TeochewBench, a human-reviewed benchmark comprising 300 Teochew Hanzi expressions for evaluating translation from Teochew Hanzi into Mandarin Chinese and English. The dataset covers five categories: basic vocabulary; everyday sentences; Teochew-specific expressions; tone, politeness, and context; and idiomatic, ambiguous, and culturally specific expressions. A primary Teochew-speaking reviewer examined all entries individually and revised them as needed, while two additional Teochew speakers verified selected items.   Our main evaluation covers 11 official general-purpose post-trained models on the reviewed dataset in both translation directions, yielding 6,600 predictions. Two official base checkpoints provide 1,200 predictions for supplemen
    
[^78]: PageRecall：衡量文献问答中的页面选择

    PageRecall: Measuring Page Selection in Literature-Grounded Question Answering

    [https://arxiv.org/abs/2609.18154](https://arxiv.org/abs/2609.18154)

    该论文发现文献问答系统的证据锚定瓶颈在检索而非阅读——页面选择器仅以 52.6% 的召回率将黄金页面呈现给模型，而模型拿到正确页面后引用准确率高达 94%，且页面缺失时往往静默失败，因此提出放弃页面选择、直接将整篇检索到的论文放入模型上下文的解决思路。

    

    我们描述了我们为 LitTraceQA（GroundLM @ EMNLP 2026）构建的系统：给定一个研究问题，从包含 27,487 篇论文的文献池中检索相关论文，引用答案所在的页面及对应的表格或图片，并按要求的格式作答。我们的主要发现是，证据锚定受限于检索环节，而非阅读环节。页面选择器仅有约一半的概率将标注者标注的页面（我们称之为“黄金页面”）呈现在负责定位证据的模型面前（黄金页面召回率为 52.6%）；而该模型在拿到正确页面的情况下，在其发出的 48 个定位结果中有 45 次引用了正确的页面（准确率达 94%）。当页面缺失时，模型很少如实说明：在这 45 个案例中，有 14 次返回空结果，24 次返回错误页面，仅 7 次返回正确页面，因此该流水线“静默失败”的频率几乎是“显性失败”的两倍。既然失败的根源在于正确的页面从未被呈现出来，解决办法就是不再进行选择：每篇检索到的论文都能完整放入模型的上下文中，因此我们……

    arXiv:2609.18154v1 Announce Type: cross  Abstract: We describe our system for LitTraceQA (GroundLM @ EMNLP 2026): given a research question, retrieve the relevant papers from a pool of 27,487, cite the page and the table or figure where the answer lives, and answer in a requested format. Our main finding is that evidence grounding is limited by retrieval, not by reading. The page selector put the annotator's page, which we call the gold page, in front of the model that locates evidence only about half the time (52.6% gold-page recall), while that model, given the page, cited the right one in 45 of the 48 locators it emitted (94%). When the page was missing it rarely said so: of 45 such cases it returned nothing 14 times, a wrong page 24 times, and a correct page 7 times, so the pipeline failed quietly almost twice as often as it failed visibly. Since the failure was that the right page was never shown, the fix is to stop choosing: each retrieved paper fits in the model's context, so we
    
[^79]: DualSQL：基于多智能体强化学习的文本到SQL转换

    DualSQL: Text-to-SQL with Multi-Agent Reinforcement Learning

    [https://arxiv.org/abs/2609.18135](https://arxiv.org/abs/2609.18135)

    DualSQL提出了一种由单一模型主干驱动两个智能体的文本到SQL系统，通过多智能体强化学习框架联合优化模式链接和SQL生成两个相互关联的任务，并借助rollout护栏机制稳定训练过程、防止模型崩溃，同时引入新的鲁棒执行匹配（REX）正确性指标。

    

    最先进的文本到SQL（Text-to-SQL）系统通常是以两个基本任务为核心的多智能体流水线：模式链接和SQL生成。然而，现有的工作为每个任务训练单独的模型，未能利用这些相互关联任务之间的协同效应。在这项工作中，我们提出了DualSQL，这是一个新的文本到SQL系统，由两个由单一模型主干驱动的智能体组成。这些智能体共享相同的模型权重和智能体框架，从而能够通过强大的多智能体强化学习（RL）框架实现联合优化。我们设计了三个数据库访问工具，以便在与数据库交互的基础上促进有效的多步推理。为了改进训练并避免模型崩溃，我们引入了一组rollout护栏机制，可以稳定多智能体强化学习训练，支持DualSQL在训练过程中持续改进。我们还引入了一个新的SQL正确性指标——鲁棒执行匹配（REX）

    arXiv:2609.18135v1 Announce Type: cross  Abstract: State-of-the-art Text-to-SQL systems are typically multi-agent pipelines centered around two fundamental tasks: schema linking and SQL generation. However, existing work trains separate models for each task, failing to leverage the synergy between these interrelated tasks. In this work, we propose DualSQL, a new Text-to-SQL system consisting of two agents powered by a single model backbone. The agents share the same model weights and agentic scaffold, enabling joint optimization through a robust multi-agent reinforcement learning (RL) framework. We design three database access tools to facilitate effective multi-step reasoning grounded to interactions with the databases. To improve training and avoid model collapse, we introduce a set of rollout guardrail mechanisms that stabilizes multi-agent RL training, supporting DualSQL to keep improving during training. We also introduce a new SQL correctness metric, robust execution match (REX),
    
[^80]: Colla-Q：通过极小极大精度平衡实现MoE量化中的专家协作

    Colla-Q: Toward Collaborative Experts in MoE Quantization via Minimax Precision Balancing

    [https://arxiv.org/abs/2609.18131](https://arxiv.org/abs/2609.18131)

    提出基于激活熵的比特分配框架Colla-Q，通过极小极大精度平衡策略均衡MoE量化中各专家的性能，从而提升整体模型表现并降低对校准数据的依赖。

    

    在本文中，我们提出了一种基于激活熵的混合专家模型（MoE）量化方法。尽管量化能够降低内存和计算成本，但它可能会显著损害模型性能。尤其在量化的MoE模型中，性能下降尤为突出，因为各个专家模型的参数数量较少，对低比特表示较为敏感。考虑到MoE作为一个集成模型运行，依赖被路由到的专家进行协同贡献，某个特定专家因量化导致的显著性能下降可能会损害模型的整体性能。因此，我们提出了Colla-Q，这是一个比特分配框架，通过基于激活熵的比特宽度分配算法来保持各专家之间性能的均衡。这种方法促使每个专家在量化模型中协同运作，从而：1）提升MoE的整体性能；2）降低对校准数据的依赖。

    arXiv:2609.18131v1 Announce Type: cross  Abstract: In this paper, we present a Mixture-of-Experts (MoE) quantization method based on activation entropy. Although quantization reduces memory and computational costs, it can substantially degrade performance. In particular, performance decline is pronounced in quantized MoE models, where individual experts have a small number of parameters that are sensitive to low-bit representation. Considering that MoE operates as an ensemble model with collaborative contributions from routed experts, a significant performance decline of a particular expert due to quantization can harm model performance. Therefore, we propose Colla-Q, a bit-allocation framework to maintain balanced performance across experts through an activation-entropy-based bit-width allocation algorithm. This approach encourages each expert to operate collaboratively in the quantized model, thereby 1) improving the overall MoE performance and 2) reducing the dependence on the calib
    
[^81]: 生成式物理人工智能的全面综述

    A Comprehensive Review of Generative Physical Artificial Intelligence

    [https://arxiv.org/abs/2609.18111](https://arxiv.org/abs/2609.18111)

    本综述系统梳理了生成式物理人工智能（GPAI）领域，提出了涵盖机器人基础模型、视觉-语言-动作模型、大行为模型、扩散策略模型和世界基础模型五大方法的分类体系，并分析了它们的架构基础、应用现状及互补关系。

    

    将大规模基础模型与物理实体相结合，催生了机器人领域的重大进展，被称为生成式物理人工智能（GPAI）。这些智能体AI系统能够在复杂的真实世界情境中自主感知、推理和行动。本综述全面分析了GPAI系统，重点关注其架构基础、当前应用和关键局限性。我们引入了一个包含五种不同方法的分类体系：用于跨平台技能迁移的机器人基础模型（RFM）；用于端到端多模态感知与控制的视觉-语言-动作（VLA）模型；用于类人动作生成的大行为模型（LBM）；基于扩散模型的时序连贯动作生成的扩散策略模型（DPM）；以及用于符合物理规律仿真与数据生成的世界基础模型（WFM）。我们考察了这些方法如何相互补充：WFM生成轨迹……

    arXiv:2609.18111v1 Announce Type: cross  Abstract: The integration of large-scale foundation models with physical embodiments has led to significant advancements in robotics known as Generative Physical Artificial Intelligence (GPAI). These agentic AI systems autonomously perceive, reason, and act in complex real-world situations. This survey comprehensively analyzes GPAI systems, focusing on their architectural foundations, current applications, and key limitations. We introduce a taxonomy of five distinct approaches: Robot Foundation Models (RFMs) for cross-platform skill transfer; Vision-Language Action (VLA) models for end-to-end multi-modal perception and control; Large Behavior Models (LBMs) for human-like movement generation; Diffusion Policy Models (DPMs) for diffusion model-based temporally coherent action generation; and World Foundation Models (WFMs) for physics-compliant simulation and data generation. We examine how these approaches complement each other: WFMs generate tra
    
[^82]: 应用于招聘场景的开源权重大语言模型中性别与种族偏见的语言触发因素

    Linguistic Triggers of Gender and Racial Bias in Open-Weight LLMs Applied to Recruitment

    [https://arxiv.org/abs/2609.18106](https://arxiv.org/abs/2609.18106)

    该论文首次将招聘启事语言作为实验变量，对六个开源权重大语言模型进行系统性偏见审计，发现能动性语言会触发对女性候选人的性别偏见、排他性编码语言会触发对非白人候选人的种族偏见，并揭示了由此产生的欧盟《人工智能法案》与美国EEOC监管合规风险。

    

    开源权重的大语言模型正在迅速进入招聘流程，然而其歧视性失效模式——以及这些模式在欧盟《人工智能法案》高风险分类（附录III）和美国EEOC不利影响分析下所引发的监管风险——仍然鲜为人知。我们首次对开源权重LLM进行了系统性、多模型的审计，将招聘启事语言作为主要实验变量，评估了六个模型（Llama 3.2、Mistral、Gemma 3、Qwen 3、Phi 3、DeepSeek-R1），通过四个受控实验共同探究了招聘者模拟和求职者模拟任务。我们发现：（1）能动性招聘语言会降低招聘者对女性候选人的推荐评分（r_rb = 0.309, p_Bonf = 7x10^-5；模型固定效应 r_rb = 0.448），而社群性语言可部分逆转这一惩罚；（2）排他性编码语言在大效应量下会抑制非白人候选人的招聘者评分（r_rb = 0.646-0.…（原文摘要在此处被截断）

    arXiv:2609.18106v1 Announce Type: cross  Abstract: Open-weight large language models are rapidly entering hiring pipelines, yet their discriminatory failure modes -- and the regulatory exposure these create under the EU AI Act high-risk classification (Annex III) and U.S. EEOC adverse-impact analysis -- remain poorly understood. We present the first systematic, multi-model audit of open-weight LLMs that treats job-posting language as the primary experimental variable, evaluating six models (Llama 3.2, Mistral, Gemma 3, Qwen 3, Phi 3, DeepSeek-R1) across four controlled experiments that jointly probe recruiter-simulation and job-seeker-simulation tasks. We find that (1) agentic posting language depresses recruiter recommendation scores for female candidates (r_rb = 0.309, p_Bonf = 7x10^-5; model-fixed-effects r_rb = 0.448), while communal language partially reverses the penalty; and (2) coded-exclusion language suppresses non-White recruiter scores at large effect sizes (r_rb = 0.646-0.
    
[^83]: Agora：以Git作为集体自动研究的共享内存

    Agora: Git as Shared Memory for Collective AutoResearch

    [https://arxiv.org/abs/2609.18094](https://arxiv.org/abs/2609.18094)

    Agora的核心创新是以Git中仅追加的DAG作为多个自主研究智能体的共享内存，让每条研究成果都成为可验证、可复现的不可变提交，并通过派生索引和多样性感知选择规则，使13个无中央规划的智能体持续协作近12天而不陷入重复搜索。

    

    诸如AutoResearch之类的自主研究循环表明，单个编码智能体可以在无人值守的情况下改进训练设置。但如果同时运行多个这样的智能体，每个会话都会从零开始，因此更多的智能体往往意味着更多的重复搜索，而非更多的发现。Agora是这类智能体的共享内存：研究以仅追加的有向无环图（DAG）的形式记录在Git中，使得每一条主张都是一个任何人都可以检出并重新运行的提交。每个结果、见解、假设、验证和报告都是一个不可变的提交，其父边标明它建立在哪些工作之上；一个派生索引用于揭示研究前沿、被忽视的分支以及每条主张的验证状态，而一种多样性感知的选择规则可防止社区坍缩到单一领导者上。我们描述了该系统并报告了它的首次持续使用情况：一次持续近12天的运行，13个语言模型工作者在没有任务分配、没有中央规划者的情况下，针对一个权重转……

    arXiv:2609.18094v1 Announce Type: cross  Abstract: Autonomous research loops such as AutoResearch show that one coding agent can improve a training setup unattended. Run several of them and each session starts from scratch, so more agents tend to mean more duplicated search rather than more discovery. Agora is a shared memory for such agents: research is recorded as an append-only directed acyclic graph (DAG) stored in Git, so that every claim is a commit anyone can check out and rerun. Each result, insight, hypothesis, verification, and report is an immutable commit whose parent edges say what it builds on; a derived index exposes the frontier, the neglected branches, and the verification status of each claim, and a diversity-aware selection rule keeps the community from collapsing onto one leader. We describe the system and report its first sustained use: a run of nearly 12 days in which 13 language-model workers, with no assigned tasks and no central planner, worked on a weight-tran
    
[^84]: 从基列的河流到大语言模型的推理分布：大规模的隐性方言偏见与语言画像

    From a River in Gilead to the Inference Distributions of Large Language Models: Covert Dialect Bias and Linguistic Profiling at Scale

    [https://arxiv.org/abs/2609.18068](https://arxiv.org/abs/2609.18068)

    本研究借鉴社会语言学配对变体方法，通过对数概率评分对十个开源大语言模型进行探测，发现对齐技术无法消除模型内部概率分布中对非裔美国人白话英语及此前被忽视的尼日利亚英语变体的隐性方言偏见，这种偏见在住房相关社会判断中普遍存在。

    

    大语言模型（LLM）正日益被部署于住房筛选等高风险领域。虽然对齐技术能够缓解生成文本中的显性种族偏见，但它们往往无法触及内部概率分布中隐性的态度关联。我们借鉴配对变体社会语言学范式，考察了四种语言变体在住房相关社会判断中的隐性方言偏见：标准美式英语（SAE）、非裔美国人白话英语（AAVE）、尼日利亚标准英语（NSE）和尼日利亚皮钦语（NP）。AAVE代表了以往隐性偏见评估中所研究的种族化方言，而NSE和NP则代表了该领域文献中此前缺失的黑人非洲后殖民语言变体。使用260组语义匹配的四元句子，并基于住房相关形容词进行对数概率评分，我们探测了十个开源权重LLM在三种社会亲疏度不同的情境中的表现：租户筛选、邻居（摘要原文在此处截断）

    arXiv:2609.18068v1 Announce Type: cross  Abstract: Large language models (LLMs) are increasingly deployed in high-stakes domains such as housing screening. While alignment techniques mitigate explicit racial bias in generated text, they often leave covert attitudinal associations in internal probability distributions untouched. Adapting the matched-guise sociolinguistic paradigm, we examine covert dialect bias in housing-related social judgments across four varieties: Standard American English (SAE), African American Vernacular English (AAVE), Nigerian Standard English (NSE), and Nigerian Pidgin (NP). AAVE reflects the racialized dialect studied in prior covert-bias evaluations, whereas NSE and NP represent Black African, postcolonial varieties absent from this literature. Using 260 meaning-matched sentence quadruples and log-probability scoring over housing-relevant adjectives, we probe ten open-weight LLMs across three contexts varying in social proximity: tenant screening, neighbor 
    
[^85]: 从压缩向量表示中进行精确的语义读出

    Exact semantic readout from compressed vector representations

    [https://arxiv.org/abs/2609.18047](https://arxiv.org/abs/2609.18047)

    本文给出了压缩向量表示能够精确线性或仿射读出谓词真值条件的充要行空间判据，并通过实验发现预训练词向量虽大多严格可分，但无一能实现精确读出。

    

    我们刻画了压缩向量表示何时能够对有限词表的真值条件进行精确的线性或仿射读出：即每个谓词对应一个固定映射，将每个实体向量映射到相应的真值向量。一个充要的行空间条件决定了这种读出是否存在；增广真值矩阵的秩为 r，因此在线性情形下最小维度为 r，在仿射情形下为 r-1。精确读出返回的值位于一个共享的真值基上，布尔联结词在该基上的作用保持不变；而仅实现可分性则需要一个中间阈值。对于二元关系，恒等关系或严格全序的精确双线性读出要求实体向量线性无关。基于 GloVe 和 word2vec 的实验区分了精确仿射恢复、线性可分性和留出集预测：大多数谓词是严格可分的，但没有任何谓词能从预训练嵌入中获得精确的仿射读出。监督式的直推训练可以达到精确……（原文摘要在此处截断）

    arXiv:2609.18047v1 Announce Type: new  Abstract: We characterize when compressed vector representations admit exact linear or affine readouts of a finite lexicon's truth conditions: one fixed map per predicate, sending each entity vector to the corresponding truth vector. A necessary and sufficient row-space condition determines existence; the augmented truth matrix has rank r, giving minimum dimension r in the linear case, and r-1 in the affine. Exact readouts return values in a shared truth basis on which Boolean connectives act unchanged; separability alone requires an intervening threshold. For binary relations, exact bilinear readout of identity or strict total order requires linearly independent entity vectors. Experiments with GloVe and word2vec distinguish exact affine recovery, linear separability, and held-out prediction: most predicates are strictly separable, but none admits an exact affine readout from the pretrained embeddings. Supervised transductive training attains exa
    
[^86]: 相关性引导的多编码器大型音频-语言模型编码器选择方法

    Correlation-Guided Encoder Selection for Multi-Encoder Large Audio-Language Models

    [https://arxiv.org/abs/2609.18041](https://arxiv.org/abs/2609.18041)

    提出CUES方法，利用编码器性能概况间的皮尔逊相关性评估互补性，无需融合训练即可高效为多编码器大型音频-语言模型选出最优编码器组合，在节省计算成本的同时避免冗余表示。

    

    多编码器融合将大型音频-语言模型（LALMs）的能力扩展到以语音为中心的识别之外，但通过直觉或穷举搜索来选择编码器往往会引入冗余表示，并使本已受限的计算预算进一步膨胀。我们提出了CUES（相关性引导的编码器选择），这是一种轻量级启发式方法，通过编码器性能概况之间的任务级和类别级皮尔逊相关性来估计互补性，仅凭单编码器评估的结果即可对候选集合进行评分——在编码器选择过程中无需进行融合训练。在XARES-LLM基准上使用冻结的SmolLM2-135M骨干网络（经LoRA适配）并通过五折交叉验证进行评估，CUES仅凭留出的开发集划分就能在每个赛道上始终识别出相同的配置，而无需使用测试数据进行选择。对于覆盖范围广泛的Track A套件，CUES选择了一个跨家族的编码器三元组（Whisper-medium、mHuBERT-147和Dasheng-base），实现了4……

    arXiv:2609.18041v1 Announce Type: cross  Abstract: Multi-encoder fusion extends Large Audio-Language Models (LALMs) beyond speech-centric recognition, but selecting encoders via intuition or exhaustive search often introduces redundant representations and inflates an already constrained compute budget. We propose CUES (Correlation-gUided Encoder Selection), a lightweight heuristic that estimates complementarity through task- and category-level Pearson correlations between encoders' performance profiles, scoring a candidate set from single-encoder evaluations alone--without fusion training during selection. Evaluated on the XARES-LLM benchmark with a frozen SmolLM2-135M backbone (LoRA-adapted) via five-fold cross-validation, CUES consistently identifies the same configuration per track from held-out development splits alone, without using test data for selection. For the broad Track~A suite, CUES selects a cross-family trio (Whisper-medium, mHuBERT-147, and Dasheng-base), achieving a 4.
    
[^87]: 注视作为共同基础化的证据：MapTask与MUNDEX的跨语料库分析

    Gaze as Evidence for Common Grounding: A Cross-Corpus Analysis of MapTask and MUNDEX

    [https://arxiv.org/abs/2609.18011](https://arxiv.org/abs/2609.18011)

    跨MapTask和MUNDEX两个语料库的分析表明，注视行为（尤其是任务主导者的注视）可作为对话中共同基础化程度的可靠行为证据。

    

    在信息不对称的协作任务中，参与者通过互动协调彼此的理解。我们探究在两个此类任务中，注视是否为共同基础化提供证据。基于离散的行为标注，我们将HCRC MapTask语料库（Anderson等人，1991）和MUNDEX语料库（Türk等人，2023）映射到统一的"伙伴/任务/移开"注视词汇体系中，并计算任务相关对话单元周围的注视特征。在两个语料库中，一致的指称解释和UND（已理解）判断都与更多的任务导向注视和更少的伙伴导向注视相关，同时伴随更低的注视熵和更少的注视转换。这些关联在主导任务的参与者身上最为明显：在指令发出者产生的指称中，以及解释者的判断中，后者还与被解释者的注视共同变化。在同一说话者的MapTask指称链中，说话者在先前未达成一致的指称被重新对齐时的注视熵更低。

    arXiv:2609.18011v1 Announce Type: new  Abstract: In collaborative tasks with asymmetric information, participants coordinate their understanding through interaction. We ask whether gaze provides evidence about grounding across two such tasks. Working from discrete behavioral annotations, we map HCRC MapTask (Anderson et al., 1991) and MUNDEX (T\"urk et al., 2023) into a shared partner/task/away vocabulary and compute gaze features around task-relevant dialogue units. In both corpora, aligned reference interpretations (MapTask) and UND (understood) judgments (MUNDEX) are associated with more task-directed gaze and with less partner-directed gaze, lower gaze entropy, and fewer gaze transitions. The associations are clearest for the participant leading the task: in giver-produced references, and in explainer judgments, which also co-vary with the explainee's gaze. In same-speaker MapTask reference chains, the speaker's gaze entropy is lower at the mention where a previously non-aligned re
    
[^88]: G-Mamba：稀疏图引导的Mamba用于视听语音增强

    G-Mamba: Sparse Graph-Guided Mamba for Audio-Visual Speech Enhancement

    [https://arxiv.org/abs/2609.18009](https://arxiv.org/abs/2609.18009)

    本文提出SG-Mamba，一种将稀疏异构图与线性复杂度Mamba骨干网络相结合的轻量级视听语音增强框架，通过内容自适应注意力建模跨模态关系并引入音频跳跃连接保留频谱细节，在LRS3上以更低的计算成本取得了竞争性或更优的增强性能。

    

    轻量级视听语音增强（AVSE）模型面临计算效率与跨模态对齐精度之间的关键权衡。简单的拼接方式缺乏关系表达能力，而密集的交叉注意力会带来额外的计算开销，并且在强声学干扰下容易产生不可靠的跨模态对应关系。我们提出了稀疏图引导的Mamba（SG-Mamba），这是一个轻量级的AVSE框架，它将稀疏异构图与线性复杂度的Mamba骨干网络相结合。该图通过内容自适应注意力和跨帧视听连接显式地建模模态特定的关系，而Mamba则负责捕获长程时序上下文。我们进一步引入了音频跳跃连接，在不牺牲噪声抑制能力的前提下保留频谱细节。在LRS3数据集上的评估表明，SG-Mamba相比强大的轻量级基线取得了具有竞争力或更优的性能，SI-SNR达到13.091 dB。

    arXiv:2609.18009v1 Announce Type: cross  Abstract: Lightweight audio-visual speech enhancement (AVSE) models face a critical trade-off between computational efficiency and cross-modal alignment accuracy. While simple concatenation lacks relational expressiveness, dense cross-attention incurs computational overhead and is prone to unreliable cross-modal correspondence under strong acoustic interference. We propose Sparse Graph-Guided Mamba (SG-Mamba), a lightweight AVSE framework that integrates a sparse heterogeneous graph with a linear-complexity Mamba backbone. The graph explicitly models modality-specific relations through content-adaptive attention and cross-frame audio-visual connections, while Mamba captures long-range temporal context. We further introduce an audio skip connection to preserve spectral detail without sacrificing noise suppression. Evaluated on LRS3, SG-Mamba achieves competitive or superior performance against strong lightweight baselines and reaches 13.091 dB SI
    
[^89]: 一种用于衡量推理优化如何影响输出质量的校准化测量工具

    A Calibrated Instrument for Measuring How Inference Optimizations Affect Output Quality

    [https://arxiv.org/abs/2609.18005](https://arxiv.org/abs/2609.18005)

    本文提出了一种经过正式校准的LLM评判测量方法，通过引入分布上与原模型完全一致的“零条件”验证机制，实现了对量化、早退、投机解码等推理加速技术对输出质量影响的严格、可跨系统比较的测量。

    

    大语言模型优化是一个活跃的研究领域，涵盖模型权重量化、跳过层的早退方法以及投机解码。每个研究方向都使用各自的质量评估方式，通常是一个独特的基准测试分数，很少有方法能接近其他科学学科所要求的测量精度。我们提出了一种严格的方法来衡量输出质量，适用于跨系统和跨技术的比较。我们使用大语言模型作为评判者来为输出打分，但对该评判者进行了正式校准：我们比较它在同一模型对相同提示词进行两次普通运行时的评分，验证其对统计上等价的输出不存在系统性偏好，并测量其每样本噪声。每种实验设计还包含一个“零”条件，该条件在分布上被证明与未修改的模型完全相同，其测得的差异必须为零。利用这一工具，我们测量了几种加速技术……

    arXiv:2609.18005v1 Announce Type: new  Abstract: Large language model optimization is an active research area, spanning quantization of model weights, early-exit methods for skipping layers, and speculative decoding. Each track uses its own quality measures, typically an idiosyncratic benchmark score. Few approach the measurement precision required by other scientific disciplines.   We propose a rigorous methodology for measuring output quality, suitable for cross-system and cross-technique comparison. We score outputs with an LLM as a judge, but calibrate the judge formally: we compare its scores on two ordinary runs of a model given the same prompts, verifying that it shows no systematic preference between statistically equivalent outputs and measuring its per-sample noise. Each design also includes a 'null' condition, provably identical in distribution to the unmodified model, whose measured difference must be zero.   With this one instrument we measure several acceleration techniqu
    
[^90]: 终结性习得发展转变的建模

    Modeling the Developmental Shift in Telicity Acquisition

    [https://arxiv.org/abs/2609.17996](https://arxiv.org/abs/2609.17996)

    该研究提出基于GPT2惊讶度差异的自动标注方法，揭示了儿童与成人在编码终结性时的分化：儿童依靠单一确定性句法线索（动词后限定词）即可准确判断，而成人则更多依赖动词语义信息。

    

    习得终结性——即有界事件（如“吃了一个苹果”）与无界事件（如“吃苹果”）之间的区别——要求第一语言（L1）学习者将表层线索和语义线索映射到抽象的事件结构上，但这一映射的计算轨迹尚不清楚。我们提出了一种“惊讶度差异”（Difference in Surprisal）方法，利用GPT2的token惊讶度对成对的时间状语诊断结构（"in an hour"与"for an hour"）进行分析，从而在英语CHILDES语料库中自动标注终结性，并通过语言学家专家的判断进行了验证。基于这些标注，我们在12个句法和词汇语义特征上训练诊断性逻辑回归分类器，以比较儿童话语与儿童导向话语（成人对儿童说话）如何编码终结性。两个模型出现分化：儿童模型通过单一确定性线索——动词后限定词的存在——达到近乎完美的准确率，而成人模型则更依赖于动词的语义特征。

    arXiv:2609.17996v1 Announce Type: new  Abstract: Acquiring telicity, which is the distinction between bounded (e.g., ate an apple) and unbounded (e.g., ate apples) events, requires first language (L1) learners to map surface-level and semantic cues to abstract event structures, but the computational trajectory of this mapping is not well understood. We introduce a Difference in Surprisal method that uses GPT2 token surprisal over paired temporal adverbial diagnostics (in an hour versus for an hour) to automatically label telicity across English CHILDES corpora, validated against expert linguist judgments. Using these labels, we train diagnostic logistic regression classifiers on 12 syntactic and lexical semantic features to compare how child speech and child-directed speech encode telicity. The two models diverge: the child model reaches near perfect accuracy through a single deterministic cue, the presence of a post-verbal determiner, while the adult model relies more heavily on verb 
    
[^91]: 通过适配器唤醒编码器：语音大语言模型的有效领域自适应微调

    Encoder Awakening via Adapters: Effective Domain-Adaptive Fine-tuning of Speech-LLMs

    [https://arxiv.org/abs/2609.17981](https://arxiv.org/abs/2609.17981)

    提出EAVA方法，通过在语音大语言模型的每个编码器层插入轻量级适配器并进行专门训练，在保留预训练知识的同时注入目标领域声学知识，有效提升有限数据下儿童语音、方言语音等领域偏移语音的识别性能。

    

    语音大语言模型通常由预训练的语音编码器、模态投影器和经低秩适配器（LoRA）微调的大语言模型构建而成，在通用领域语音上展现出强大的自动语音识别（ASR）性能。然而，在目标领域数据有限的情况下，将其适应到存在领域偏移的语音（如儿童语音或方言语音）仍然具有挑战性。鉴于大语言模型在语音大语言模型中的主导地位，且交叉熵损失仅在大语言模型输出端应用，语音编码器可能无法充分适应新的声学条件。本文提出了通过适配器唤醒编码器（EAVA），这是一种简单而有效的针对基于语音大语言模型的ASR的领域自适应微调方法。首先，在每个编码器层中插入轻量级适配器并进行专门训练，使目标领域的声学知识能够融入编码器，同时保留其预训练知识。

    arXiv:2609.17981v1 Announce Type: cross  Abstract: Speech Large Language Models (Speech-LLMs), typically built from a pre-trained speech encoder, a modality projector, and an LLM fine-tuned with Low-Rank Adapters (LoRA), have shown strong Automatic Speech Recognition (ASR) performance on general-domain speech. However, adapting them to domain-shifted speech, such as child or dialectal speech, remains challenging under limited target-domain data. Given the dominant role of the LLM in Speech-LLMs, with cross-entropy loss applied only at the LLM output, the speech encoder may receive insufficient adaptation to new acoustic conditions. In this paper, we propose Encoder Awakening via Adapters (EAVA), a simple yet effective domain-adaptive fine-tuning method for Speech-LLM-based ASR. First, lightweight adapters are inserted into each encoder layer and trained exclusively, enabling target-domain acoustic knowledge to be incorporated into the encoder while preserving its pre-trained knowledge.
    
[^92]: TACTICS：面向机器翻译的分类体系感知智能语料库抽样

    TACTICS: Taxonomy-Aware Intelligent Corpus Sampling for Machine Translation

    [https://arxiv.org/abs/2609.17956](https://arxiv.org/abs/2609.17956)

    该论文提出TACTICS方法，将机器翻译评估中的语料抽样从随机方式转变为显式的覆盖优化问题：通过从本地化风格指南构建层次化分类体系，在固定预算下联合优化稀有类别覆盖、文档级连贯性和语料分布保真度，从而为系统鲁棒性评估提供覆盖保证。

    

    大规模机器翻译（MT）系统通常在从语料库中随机抽取的样本上进行评估，而语料库的分布构成本质上取决于其构建方式。这样的样本仅继承了语料库碰巧包含的语言现象，而非系统必须处理的完整空间——这些现象既涵盖规则约束的惯例（术语、标点、货币格式），也包括依赖上下文的现象（语气、敬语、文档级连贯性），因而无法为鲁棒性评估提供覆盖保证。我们提出了TACTICS（分类体系感知的覆盖优化智能语料库抽样），它将覆盖率重新定义为一个显式目标。TACTICS从本地化风格指南中归纳出层次化分类体系，据此对语段进行分类，并在固定预算下选择子集，联合优化稀有类别的覆盖率、文档级连贯性以及对完整语料库的分布保真度。该方法应用于跨四种……（评估场景）的机器翻译评估。

    arXiv:2609.17956v1 Announce Type: new  Abstract: Large-scale machine-translation (MT) systems are typically evaluated on random samples from a corpus whose distributional composition is an artifact of how it was assembled. Such a sample inherits the phenomena the collection happens to contain rather than the full space a system must handle, spanning rule-governed conventions (terminology, punctuation, currency formatting) and context-dependent phenomena (tone, honorifics, document-level coherence), and thus provides no coverage guarantee for assessing robustness. We propose TACTICS (Taxonomy-Aware Coverage-opTimized Intelligent Corpus Sampling), which recasts coverage as an explicit objective. TACTICS induces a hierarchical taxonomy from a locale style guide, classifies segments against it, and selects a fixed-budget subset jointly optimizing coverage of rare categories, document-level coherence, and distributional fidelity to the full corpus. Applied to MT evaluation across four trans
    
[^93]: ASPIRE：面向长上下文大语言模型推理的异步批量自推测解码

    ASPIRE: Asynchronous Batched Self-Speculative Decoding for Long-Context LLM Inference

    [https://arxiv.org/abs/2609.17943](https://arxiv.org/abs/2609.17943)

    ASPIRE提出了一种非同步的批量自推测解码框架，通过统一混合前向计算、基于接受率估计和批次感知成本模型的在线调度器，让批中每个请求独立决定验证时机，从而加速长上下文大语言模型推理。

    

    长上下文大语言模型推理受注意力机制的瓶颈制约，其重复的KV缓存读取使得解码成为内存受限的操作。自推测解码通过使用稀疏注意力起草token、再用完整注意力进行验证来缓解这一问题，但现有的批量方法仍然是同步的：批中的所有请求共享单一的起草-验证调度，尽管最优起草长度在不同请求之间差异很大，并且在每个请求内部也会动态变化。我们提出ASPIRE，一个建立在三个组件之上的非同步批量自推测解码框架。首先，统一的混合前向计算允许起草和验证请求共存于同一批量前向传播中，消除了对全局起草-验证阶段的需求。其次，轻量级的在线推测调度器使用每个请求的接受率估计和批次感知的成本模型，让每个请求独立地选择何时进行验证。第三，起草内部刷新层（摘要截断）……

    arXiv:2609.17943v1 Announce Type: cross  Abstract: Long-context LLM inference is bottlenecked by attention, whose repeated KV-cache reads make decoding memory-bound. Self-speculative decoding alleviates this by drafting tokens with sparse attention and verifying them with full attention, but existing batched methods remain synchronized: all requests in a batch share a single draft-verify schedule, even though the optimal draft length varies widely across requests and changes dynamically within each request. We propose ASPIRE, a non-synchronized batched self-speculative decoding framework built on three components. First, a unified mixed forward allows drafting and verifying requests to coexist in the same batched forward pass, removing the need for global draft-verify phases. Second, a lightweight online speculation scheduler uses per-request acceptance-rate estimates and a batch-aware cost model to let each request independently choose when to verify. Third, an intra-draft refresh lay
    
[^94]: 使用状态空间模型的长上下文示例选择

    Long-Context Demonstration Selection Using State Space Models

    [https://arxiv.org/abs/2609.17888](https://arxiv.org/abs/2609.17888)

    本文提出一种基于状态空间模型（SSM）的示例选择方法，通过从transformer模型蒸馏出线性的SSM，高效解决长上下文场景下推理成本高企的示例选择难题。

    

    我们研究示例选择问题，即选择一个示例子集并将其前置到语言模型的查询之前。这个问题与上下文学习和语言模型推理密切相关。由于transformer模型的推理成本随序列长度呈二次方增长，因此在长上下文场景中，选择问题变得尤其具有挑战性。在本文中，我们通过基于状态空间模型（SSMs）来解决这一问题，SSMs在给定输入的情况下只需线性的推理时间。我们的方法包括两种算法。第一种算法通过蒸馏（已训练的）transformer模型来学习一小组SSMs：我们将所有层划分为连续的组，然后对每个组，我们估计一个单独的状态空间模型来复制相邻层内的输入输出行为。其次，我们将蒸馏模型的输出映射到一小组token上，并将这些嵌入应用于……

    arXiv:2609.17888v1 Announce Type: cross  Abstract: We study the problem of demonstration selection, which involves selecting a subset of examples for prepending to a query to a language model. This problem is closely related to in-context learning and language model inference. Since the inference cost of a transformer model scales quadratically with sequence length, the selection problem becomes especially challenging in a long-context scenario. In this paper, we tackle this problem by building on state space models (SSMs), which require only linear inference time given the input. Our approach involves two algorithms. The first learns a small set of SSMs through distillation of a (trained) transformer model. We partition all the layers into consecutive groups. Then for each group, we estimate a separate state space model to replicate the input-output behavior within the adjacent layers. Second, we map the distilled model outputs to a small set of tokens, and apply these embeddings for 
    
[^95]: 不确定性感知的持续学习：面向演化标签空间下的开放世界意图发现

    Uncertainty-Aware Continual Learning for Open-World Intent Discovery Under an evolving Label Space

    [https://arxiv.org/abs/2609.17866](https://arxiv.org/abs/2609.17866)

    该论文提出了一种统一的不确定性感知概率框架，通过自适应β-VAE编码、分类器置信度-后验不确定性-DP-GMM似然的多信号决策机制和基于密度的聚类发现，在演化的标签空间下实现开放世界新意图的持续发现与可控标签空间扩展，并结合回放与弹性权重巩固来缓解灾难性遗忘。

    

    现实世界的智能系统越来越多地在开放世界条件下运行，其中用户意图并非固定不变，也无法事先穷尽获知，且会随着新的交互模式的出现而不断演化。本文提出了一个统一的不确定性感知概率框架，用于在演化的标签空间下进行持续的新意图发现。每条话语通过自适应β-VAE被编码为潜在均值（用于分类和密度建模），以及作为全局可靠性信号的后验不确定性估计。分类器置信度、后验不确定性和DP-GMM似然通过多信号决策机制相结合，以区分已知意图与潜在的新颖样本。候选的新颖实例通过基于密度的发现模块进行聚类，只有可靠的簇才会被提升为新标签，从而实现可控的标签空间扩展。回放机制与弹性权重巩固（EWC）用于缓解灾难性遗忘。

    arXiv:2609.17866v1 Announce Type: cross  Abstract: Real-world intelligent systems increasingly operate under open-world conditions, where user intents are not fixed or exhaustively known a priori and may evolve as new interaction patterns emerge. This paper proposes a unified uncertainty-aware probabilistic framework for continual new intent discovery under an evolving label space. Each utterance is encoded through an adaptive $\beta$-VAE into a latent mean, used for classification and density modelling and a posterior uncertainty estimate acting as a global reliability signal. Classifier confidence, posterior uncertainty and DP-GMM likelihood are combined through a multi-signal decision mechanism to distinguish known intents from potentially novel samples. Candidate novel instances are clustered through a density-based discovery module and only reliable clusters are promoted to new labels, enabling controlled label-space expansion. Replay and Elastic Weight Consolidation mitigate cata
    
[^96]: 谁来评判很重要：测量大语言模型评审组中基于模型家族的条件偏好

    Who Judges Matters: Measuring Family-Conditioned Preference in LLM-as-Judge Panels

    [https://arxiv.org/abs/2609.17857](https://arxiv.org/abs/2609.17857)

    该研究首次系统测量了大语言模型评审中的“同家族偏好”效应——即模型评审会偏袒同一家族的候选模型——通过提出一种固定候选家族的校正估计器，发现四个主流开放权重模型家族均存在3.4-8.4个百分点的显著同家族提升，且该效应与评审侧似然度密切相关。

    

    评判者的身份会影响LLM-as-judge（大语言模型作为评审）的结果，但在不与候选模型质量相混淆的情况下测量这种影响十分困难。我们在完全交叉的成对设计中研究了四个开放权重模型家族（Llama 3.1、Qwen 2.5、Gemma 2和Yi 1.5），共进行了9,312次评判。常见的按家族统计量与候选模型质量存在强烈混淆，其与Bradley-Terry能力的相关系数高达r = 0.95。为此，我们推导出一个校正估计器，该估计器在固定候选家族的前提下比较不同评审者。校正后，所有四个家族均显示出正向的同家族提升（3.4-8.4个百分点），全局FPS为0.067（95%置信区间[0.053, 0.084]，置换检验p = 0.0002）。该效应在基于评审组的质量控制、独立的人类共识锚点以及float16评判复现下依然稳健存在。评审侧的似然度与该效应密切相关：加入似然优势项会使受控系数降低61%，我们将其视为描述性的衰减。

    arXiv:2609.17857v1 Announce Type: cross  Abstract: Who the judge is can affect an LLM-as-judge result, but measuring that effect without confusing it with candidate quality is difficult. We study four open-weight families (Llama 3.1, Qwen 2.5, Gemma 2, and Yi 1.5) in a fully crossed pairwise design with 9,312 judgments. A common per-family statistic is strongly confounded with candidate quality and correlates with Bradley-Terry ability at r = 0.95. We derive a corrected estimator that holds the candidate family fixed and compares judges. All four families then show a positive same-family lift (3.4-8.4 percentage points), with global FPS 0.067 (95% CI [0.053, 0.084], permutation p = 0.0002). The effect remains under panel-based quality controls, an independent human-consensus anchor, and a float16 judging replication. Judge-side likelihood is closely related to the effect: adding likelihood advantage reduces the controlled coefficient by 61%, which we treat as descriptive attenuation ra
    
[^97]: AfriSyCo：测量非洲语言内容中的断言式框架、验证与措辞敏感性

    AfriSyCo: Measuring Assertive Framing, Verification, and Wording Sensitivity Around African-Language Content

    [https://arxiv.org/abs/2609.17853](https://arxiv.org/abs/2609.17853)

    该论文提出AfriSyCo框架，通过母语后续提问与跨语言2×2因子实验系统测量非洲语言事实内容中模型答案切换行为，发现断言式框架会显著增加错误目标选择（+30.4个百分点），而验证机制可有效降低该效应（-17.4个百分点）。

    

    AfriSyCo 通过两个互补层次研究围绕非洲语言事实性内容的答案切换现象：母语后续提问和受控的跨语言因子实验，其中问题、选项和目标保持非洲语言，而后续提问的框架采用英语。我们分析了源自100个源问题、涵盖七个开源权重模型检查点和六种语言的1,415个首轮正确观测数据；首轮正确指的是观察到的首次回答准确性，而非已证明的知识。在母语提示下，断言式背书相比提及加验证（M+V）方式，在任意轮次中导致的错误目标选择多出29.3个百分点，即时T2对比为19.0个百分点。在预先承诺的2×2因子实验中，对三种测试提示家族取平均，断言式框架使目标选择增加30.4个百分点（95%置信区间[28.4, 32.3]）；验证使其降低17.4个百分点，而断言效应随……（摘要此处截断）

    arXiv:2609.17853v1 Announce Type: cross  Abstract: AfriSyCo studies answer switching around African-language factual content with two complementary layers: native-language follow-ups and a controlled cross-language factorial whose question, options, and target remain in the African language while the follow-up framing is English. We analyze 1,415 turn-1-correct model-language-item observations derived from 100 source questions across seven open-weight checkpoints and six languages; turn-1-correct denotes observed first-response accuracy, not demonstrated knowledge. Under native prompts, assertive endorsement produces 29.3 percentage points more any-turn false-target selection than mention-plus-verification (M+V), with a 19.0-point immediate T2 contrast. In the precommitted 2 x 2 factorial, averaged over three tested prompt families, assertive framing increases target selection by 30.4 points (95% CI [28.4, 32.3]); verification decreases it by 17.4 points, while the assertive effect ris
    
[^98]: 工具调用智能体该用SFT还是RL？一项跨越数据、方法与规模的受控研究

    SFT or RL for Tool-Calling Agents? A Controlled Study Across Data, Method, and Scale

    [https://arxiv.org/abs/2609.17848](https://arxiv.org/abs/2609.17848)

    通过在0.6B至32B规模的六个Qwen3模型上进行受控实验发现，带LoRA的SFT是分布内工具调用最强的训练方法，而无论采用何种方法，数据集混合都是获得强跨数据集迁移能力的最可靠手段。

    

    关于训练数据、适配方法和模型规模如何共同影响语言模型智能体中工具调用性能的受控证据仍然有限。我们在六个参数规模从0.6B到32B的Qwen3模型上，评估了基于LoRA的监督微调（SFT）、通过群体相对策略优化（GRPO）实现的强化学习（RL），以及SFT后接GRPO的训练方案，涵盖分布内性能和跨数据集迁移两个方面。结果表明，基于LoRA的SFT在整个0.6B-32B参数范围内都是最强的分布内方法，在18个实验设置中的15个里表现最佳。在跨数据集迁移方面，各方法的差距更小：在训练集与测试集不同的54个设置中，GRPO赢得了其中29个，但其相对SFT的优势平均不到一分，而SFT->GRPO在任何比较中都很少是最强的方案。无论采用何种方法，数据集混合都能带来持续强劲的迁移表现，同时保持接近专门化分布内训练的水平。此外，进一步的分析还……

    arXiv:2609.17848v1 Announce Type: new  Abstract: Limited controlled evidence exists on how training data, adaptation method, and model scale jointly affect tool-calling performance in language-model agents. We evaluate supervised fine-tuning (SFT) with LoRA, reinforcement learning (RL) via Group Relative Policy Optimization (GRPO), and SFT followed by GRPO across six Qwen3 models from 0.6B to 32B parameters, covering both in-distribution performance and cross-dataset transfer. SFT with LoRA is the strongest in-distribution method throughout the 0.6B-32B range and best in 15 out of 18 experimental settings. On cross-dataset transfer, the methods are closer: GRPO wins 29 out of 54 settings where training and test datasets differ, but its margin over SFT averages under one point, and SFT->GRPO is rarely strongest in either comparison. Dataset mixing gives consistently strong transfer while staying close to specialized in-distribution training, regardless of method. Additional analysis fur
    
[^99]: PrimeScientist：自主研究中的研究努力战略分配

    PrimeScientist: Strategic Allocation of Research Effort in Autonomous Research

    [https://arxiv.org/abs/2609.17846](https://arxiv.org/abs/2609.17846)

    提出了PrimeScientist框架，将自主研究智能体的研究方向选择与资源投入决策统一建模为序贯决策问题，通过可执行计划树保留竞争性方案及其结果，并利用剩余资源显式引导研究策略，实现研究努力的战略性分配。

    

    自主研究智能体旨在自动化科学工作流程，从提出想法、开展实验到分析结果。然而，当前的人工智能和研究智能体所能提出的研究方向，往往多于其可用资源所能支撑的数量。此外，每一次尝试都可能消耗大量资源，这就要求智能体重新考量后续研究中的投入方式。因此，如何战略性地分配研究努力应当成为自主研究智能体的一项核心能力。为此，我们提出了PrimeScientist，它在连续多次研究尝试中联合确定研究方向与资源投入。具体而言，我们将这一战略性研究努力分配的挑战形式化为一个序贯决策问题，其中剩余资源应当显式地引导研究策略。我们首先引入了一种可执行计划树，用于在多次尝试中保存相互竞争的计划及其执行结果。

    arXiv:2609.17846v1 Announce Type: cross  Abstract: Autonomous research agents aim to automate scientific workflows, from proposing ideas to conducting experiments and analyzing results. Yet current AI and research agents can propose more directions than available resources allow them to pursue. Moreover, each attempt could consume substantial resources, requiring agents to reconsider how to invest in subsequent research. Thus, deciding how to invest research effort strategically should be a defining capability of autonomous research agents. Accordingly, we introduce PrimeScientist, which jointly determines research direction and resource investment across successive research attempts. Specifically, we formulate this challenge of strategic research effort allocation as a sequential decision problem where remaining resources should explicitly guide the research policy. We first introduce an executable plan tree that preserves competing plans and their outcomes across attempts. Building o
    
[^100]: 校准内容如何影响基于注意力的重排序

    How Calibration Content Shapes Attention-Based Reranking

    [https://arxiv.org/abs/2609.17764](https://arxiv.org/abs/2609.17764)

    该论文揭示了注意力重排序器中的空查询校准在面对包含详细指令的提示时会错误地移除相关信号，并提出了一种无需训练的插值空校准方法，通过控制进入空基线的指令内容比例，恢复了指令密集型任务上的重排序性能。

    

    基于注意力的重排序器通过聚合查询到文档的注意力，并减去一次空查询校准过程来对文档进行评分，以消除位置和结构偏差。尽管被广泛使用，这种校准假设空查询过程会从每个文档中移除无关信号。我们表明，现代提示内容（例如约束、指令、角色设定和示例演示）在进入评分读出时可能会违反这一假设，使空查询过程变得具有相关性感知而非真正的“空”。我们发现，当校准应用于包含较长、更详细指令的提示时尤其有害，因为空查询步骤会移除相关信号。基于这些发现，我们提出插值空校准，这是一种无需训练的修改方法，可控制多少指令内容进入空基线。它在标准校准失效的指令密集型任务上恢复了基于注意力的重排序性能，同时……（摘要截断）

    arXiv:2609.17764v1 Announce Type: new  Abstract: Attention-based rerankers score documents by aggregating query-to-document attention and subtracting a null-query calibration pass to remove positional and structural bias. Although widely used, this calibration assumes that the null pass removes irrelevant signal from each document. We show that modern prompt content, e.g. constraints, instructions, personas, and demonstrations can violate this assumption when it enters the scoring readout, making the null pass relevance-aware rather than null. We find that calibration is especially harmful when applied to prompts containing longer, more detailed instructions as the null-pass step removes relevant signal. Based on these findings, we propose interpolated null calibration, a training-free modification that controls how much of the instruction content enters the null baseline. It recovers attention-based reranking performance on instruction-heavy tasks where standard calibration fails, whi
    
[^101]: 路加是《福音书》和《使徒行传》的作者吗？

    Is Luke the Author of a Gospel and the Acts of the Apostles?

    [https://arxiv.org/abs/2609.17762](https://arxiv.org/abs/2609.17762)

    本研究运用Burrows' Delta作者归属方法和作者验证模型等定量分析手段，证实《路加福音》和《使徒行传》确实出自路加一人之手。

    

    根据基督教传统，路加被认为是《福音书》和《使徒行传》的作者，尽管他的名字并未出现在这两本书中，且这两本书最初都是以通用希腊语写成的。一些圣经学者认为这两部文本出自同一位作者之手，而另一些学者则推断存在两位不同的作者。已有不同的研究支持这两种不同的结论，其中一些基于定性评估，另有少数研究则考虑了两本书之间词汇出现频率的差异。为了提出一种增强的定量分析方法，本研究基于两种近期的作者归属模型展开。将Burrows' Delta方法应用于十一种不同的特征规模后，研究结果表明两本书具有共同的作者身份。一个作者验证模型也证实了这一发现。后续实验通过考虑多种文体风格表示、特征规模和距离函数，进一步确认路加确实是这两本书的作者。

    arXiv:2609.17762v1 Announce Type: cross  Abstract: According to Christian tradition, Luke is credited with authoring a Gospel and the Acts of the Apostles, even if his name does not appear in either book, both originally written in Koine Greek. Several biblical scholars assume that both texts were written by a common author, while others deduce the presence of two authors. Different studies have been found to support either finding, some based on qualitative evaluation, while a few others consider the occurrence frequency differences between the two books. To propose an enhanced quantitative analysis, this study is grounded on two recent authorship attribution models. The Burrows' Delta, applied with eleven different feature sizes, demonstrates common authorship. An author verification model confirms this finding. The following experiments consider several stylistic representations, feature sizes, and distance functions to confirm that Luke is the true author of both books.
    
[^102]: 美国口头政治语言的演变

    Evolution of US Oral Political Language

    [https://arxiv.org/abs/2609.17755](https://arxiv.org/abs/2609.17755)

    本研究首次通过分析1960至2024年间19位美国总统选举候选人的口头辩论语言，揭示了美国政治语言随时间显著简化、平均句长持续缩短的长期演变趋势。

    

    对美国政治语言的分析通常基于书面形式（如总统演讲）或发布在各类社交网络上的帖子。然而，更为频繁出现的口头表达能够更好地揭示说话者的风格和思维方式。本研究涵盖了这种语言交流模式，考察了1960年至2024年间总统选举中的19位候选人。我们的主要研究目标是揭示这些总统辩论中隐藏的主要趋势：我们是否观察到美国政治语言随时间推移而明显简化？与其他候选人相比，特朗普的语言是否较差？不寻常的文体特征是否只出现在某一位特定的总统身上？此外，我们能否发现解释某些提名人成功或失败的模式？本研究表明，随着时间推移，政治语言的复杂度显著降低，平均句长减少……

    arXiv:2609.17755v1 Announce Type: cross  Abstract: The analysis of US political language is usually based on the written form (e.g. presidential addresses) or posts broadcasted on various social networks. Oral production, however, which is even more frequent, can better reveal the style and mode of thinking of the speaker. This study covers this mode of linguistic communication by considering 19 candidates from the presidential elections between 1960 to 2024. Our main research objectives are to disclose the main trends hidden in those presidential debates. Do we observe a clear simplification of the US political language over time? Does Trump have poor language compared to the other candidates? Do unusual stylistic features occur only with a single, specific president? Moreover, can we detect a pattern explaining the success or failure of some nominees? Over time, this study demonstrates a significant reduction in political language complexity, a decrease of the mean sentence length, a
    
[^103]: 特朗普的词汇贫乏吗？不同长度文本的词汇丰富度研究

    Is Trump's Vocabulary Poor? Vocabulary Richness Across Texts of Different Lenghts

    [https://arxiv.org/abs/2609.17747](https://arxiv.org/abs/2609.17747)

    本研究提出将词汇细分为通用与专业词汇表来解释词汇量增长的模型，以此评估口头政治传播中的词汇丰富度。

    

    本研究探讨了口头政治传播中的词汇丰富度。通过将整个词汇细分为由通用词汇表和专业词汇表生成的词条，提出了一个解释词汇量增长的模型。

    arXiv:2609.17747v1 Announce Type: new  Abstract: This study explores the vocabulary richness of oral political communication. A model explaining the lexicon growth is proposed by subdividing the whole vocabulary into terms generated by general and specialized glossaries.
    
[^104]: 置信度源于经验：从推理到智能体的经验性置信度估计

    Confidence Comes from Experience: Experiential Confidence Estimation from Reasoning to Agents

    [https://arxiv.org/abs/2609.17708](https://arxiv.org/abs/2609.17708)

    论文提出XConf，通过检索模型积累的过往分级经验记录（含任务、反思、置信度、结果与教训）来估计置信度，突破了仅依赖当前推理过程的传统置信度估计范式。

    

    可靠的置信度估计对语言模型的可信部署日益核心：对输出正确概率的校准估计决定了哪些内容可以发布、哪些需要上报、哪些需要重试。然而，现有的置信度估计器共享一个设计前提：它们只读取当前的推理过程，或是通过内省方式，或是对其token概率进行打分，或是对其进行重采样。我们认为，仅凭当前推理过程不足以作为置信度估计的充分依据。我们提出XConf（eXperiential Confidence，经验性置信度）：结合模型积累的经验来共同估计置信度。经验以模型自身经过分级评定的过往情节记录的形式存储，每条记录包含任务、模型的反思、其陈述的置信度、结果，以及在评级到来时写下的一条经验教训。给定新任务时，XConf的召回阶段会检索在相似任务上以相似陈述置信度遇到的过往情节……

    arXiv:2609.17708v1 Announce Type: cross  Abstract: Reliable confidence estimation is increasingly central to the trustworthy deployment of language models: a calibrated estimate of the probability that an output is correct decides what to ship, what to escalate, and what to retry. Existing confidence estimators, however, share one design premise: they only read the current inference process, either by introspecting on it, scoring its token probabilities, or resampling it. We argue that the current inference is not a sufficient basis for confidence. We propose XConf (eXperiential Confidence): estimating confidence together with the model's accumulated experience. The experience is stored as a record of the model's own graded past episodes, each holding the task, the model's reflection, its stated confidence, the outcome, and a lesson written once the grade arrived. Given a new task, XConf's Recall stage retrieves past episodes on similar tasks met with a similar stated confidence, and r
    
[^105]: NeMo Data Designer：一个可扩展的多模态合成数据生成框架

    NeMo Data Designer: An Extensible Framework for Multimodal Synthetic Data Generation

    [https://arxiv.org/abs/2609.17699](https://arxiv.org/abs/2609.17699)

    NeMo Data Designer是一个开源、可扩展的多模态合成数据生成框架，通过声明式配置、灵活的插件系统和内置的预览-修订迭代循环，实现了直观、可复现且可迭代的数据集生成。

    

    我们提出了NeMo Data Designer（NDD），一个开源、通用的多模态合成数据生成（SDG）框架。NDD旨在直观易用，提供了一种声明式配置格式，人类和/或智能体用户可在其中定义每个数据集列，列类型涵盖文本、代码、结构化输出、图像、嵌入以及通过明确配置来引导数据集多样性的统计采样器。额外的列类型和功能可通过框架灵活的插件系统引入。NDD的配置是一个可检查的工件，支持工作流共享和可复现性。合成数据生成本质上是一个迭代过程，因此NDD在其核心工作流中内置了预览与修订循环，允许用户生成并检查少量记录、完善规范，然后以全规模重新运行生成。在运行时，NDD会解析依赖关系，调度对用户提供的模型（原文此处截断）……

    arXiv:2609.17699v1 Announce Type: new  Abstract: We present NeMo Data Designer (NDD), an open-source, general-purpose framework for multi-modal synthetic data generation (SDG). Designed to be intuitive to use, NDD provides a declarative configuration format in which human and/or agent users define each dataset column, with column types spanning text, code, structured outputs, images, embeddings, and statistical samplers that are explicitly configured to steer dataset diversity. Additional column types and functionality can be introduced using the framework's flexible plugin system. NDD's configuration is an inspectable artifact, supporting workflow sharing and reproducibility. SDG is an inherently iterative process. NDD therefore builds a preview-and-revision loop into its core workflow, allowing users to generate and inspect a small number of records, refine the specification, and rerun generation at full scale. At runtime, NDD resolves dependencies, schedules calls to user-provided m
    
[^106]: GraphEcho：LLM图智能体中的结构冗余与证据溯源

    GraphEcho: Structural Redundancy and Evidence Provenance in LLM Graph Agents

    [https://arxiv.org/abs/2609.17695](https://arxiv.org/abs/2609.17695)

    GraphEcho基准测试揭示LLM图智能体会将结构冗余的重复路径误认为额外佐证，而溯源感知后训练（PAPT）虽能减少重复探索，却暴露了高效探索与有效证据利用之间的根本差距。

    

    arXiv:2609.17695v1 公告类型：新 摘要：大语言模型（LLM）智能体能够在不获取更多独立证据的情况下遍历更多的图路径。GraphEcho旨在测试智能体是否会将这些重复遇到的内容误认为是额外的佐证。该基准在保持证据内容不变的情况下，改变路径数量和证据来源，并同时评估判断和主动探索两个方面。受控合成实验揭示了模型相关的判断偏移，但冗余的支持路径会增加所有被评估的冻结智能体中重复遍历的比例。溯源感知后训练（PAPT）减少了重复访问并提高了合成精度，但覆盖的独特来源更少。在科学主张上，它继续减少重复，但精度却有所下降。这些发现揭示了高效探索与有效证据使用之间的差距：智能体可以学会不再重复自己，却忽略了它所需要的信息。GraphEcho提供了一种受控的方式来评估……

    arXiv:2609.17695v1 Announce Type: new  Abstract: A large language model (LLM) agent can follow more graph paths without acquiring more independent evidence. GraphEcho tests whether agents mistake these repeated encounters for additional corroboration. The benchmark varies path counts and evidential origins while holding evidence content fixed, and evaluates both judgments and active exploration. Controlled synthetic experiments reveal model-dependent judgment shifts, but redundant supporting paths increase the share of repeated walks across all evaluated frozen agents. Provenance-aware post-training (PAPT) reduces revisits and improves synthetic accuracy, yet covers fewer distinct sources. On scientific claims, it continues to reduce repetition while accuracy declines. These findings expose a gap between efficient exploration and effective evidence use: an agent can learn to stop repeating itself while overlooking information it needs. GraphEcho provides a controlled way to evaluate bo
    
[^107]: 缺失的“我不知道”：为什么三项推理可靠性研究发现共同指向校准弃答

    The Missing "I Don't Know": Why Three Reasoning-Reliability Findings Converge on Calibrated Abstention

    [https://arxiv.org/abs/2609.17686](https://arxiv.org/abs/2609.17686)

    该论文的核心创新是论证三项看似独立的LLM可靠性研究发现（推理强化学习破坏工具可靠性表征、安全约束下大小模型的差异化表现、以及缺乏“我不知道”功能的系统必然产生无穷幻觉）实际上共同指向同一项缺失能力——校准弃答。

    

    最近的三项研究结果描述了看似互不相关的大语言模型可靠性问题。Yin等人（2026）表明推理强化学习会破坏工具可靠性表征。Suleymanov等人（2026）表明在安全约束生成条件下，大模型会改写被标记的文本片段，而小模型则会截断这些片段。Bastounis等人（2024）证明，任何缺乏内隐“我不知道”功能的一致推理系统，在广泛的问题类别上必然产生无穷多次幻觉。我们认为这些发现共同指向单一干预措施：校准弃答正是各项研究独立识别出的缺失能力，尽管它们所记录的不可用性——能力缺口、策略缺口和递归论缺口——在每种情况下来源各不相同。诚实性后训练已缩小了已部署模型中的这一差距，但要原则上弥补Bastounis所识别的问题类别，需要一个校准弃答函数，其训练信号在……（原文截断）

    arXiv:2609.17686v1 Announce Type: cross  Abstract: Three recent results describe what look like unrelated LLM reliability problems. Yin et al. (2026) show reasoning RL collapses tool-reliability representations. Suleymanov et al. (2026) show that under safety-constrained generation, large models rewrite flagged spans while small models truncate. Bastounis et al. (2024) prove any consistent-reasoning system without an implicit "I don't know" function must hallucinate infinitely often on broad problem classes. We argue these findings converge on a single intervention: calibrated abstention is what each independently identifies as the missing capability, even though the unavailability they document, a capability gap, a policy gap, and a recursion-theoretic gap, has a different source in each case. Honesty post-training has narrowed the gap in deployed models, but principled closure of the class Bastounis identifies requires a calibrated abstention function whose training signal at the lea
    
[^108]: Fathom：面向卸载KV缓存稀疏解码的逐查询读取深度

    Fathom: Per-Query Read Depth for Sparse Decoding over Offloaded KV Caches

    [https://arxiv.org/abs/2609.17652](https://arxiv.org/abs/2609.17652)

    Fathom提出一种让每个查询自适应决定键通道读取位数的稀疏解码方法，通过比特平面存储与逆注水式比特预算分配，在百万token卸载KV缓存场景下实现比现有136位扫描方法快1.67倍的GPU解码速度，同时保持更低注意力误差。

    

    当智能体会话运行至百万token且同时驻留多个会话时，KV缓存及其排序索引存放在主机内存中，而针对top-k步骤对所有n个键进行排序的扫描成为限制解码速度的流量瓶颈。我们提出Fathom，一种由每个查询自主决定读取每个键通道多少比特的键扫描方法。4位K缓存以通道优先的方式存储为比特平面，因此t个平面的前缀恰好构成该通道的t位量化器，查询通过对方差加权通道重要性进行逆注水来分配其比特预算。在Qwen3-8B上处理一百万token时，解码步骤的GPU时间比Double Sparsity、Loki和SparQ r=32的136位扫描快1.67倍；在与SparQ的68位读取（r=16）相同的GPU时间内，Fathom读取的字节数减少18%，且在七个模型与上下文设置中的六个上注意力误差更低。在RULER风格的任务上，每次逐token扫描均与精确top-k解码结果相匹配。

    arXiv:2609.17652v1 Announce Type: cross  Abstract: When agentic sessions run to a million tokens with many sessions resident at once, the KV cache and the index that ranks it live in host memory, and the scan that ranks all n keys for a top-k step becomes the traffic that bounds decoding. We present Fathom, a key scan in which each query decides how many bits of each key channel to read. The 4-bit K cache is stored channel-major as bit planes, so a prefix of t planes is exactly the channel's t-bit quantizer, and the query spends its bit budget by reverse water-filling over the variance-weighted importance of its channels. At one million tokens on Qwen3-8B a decode step is 1.67x faster in GPU time than with the 136-bit scans of Double Sparsity, Loki and SparQ r=32, and in the same GPU time as SparQ's 68-bit read (r=16) Fathom reads 18% fewer bytes with lower attention error on six of seven model and context settings. On RULER-style tasks every per-token scan matches exact top-k decoding
    
[^109]: EvolveTrade：面向自进化LLM交易智能体的经验驱动策略精炼

    EvolveTrade: Experience-Driven Policy Refinement for Self-Evolving LLM Trading Agents

    [https://arxiv.org/abs/2609.17632](https://arxiv.org/abs/2609.17632)

    EvolveTrade提出将LLM交易智能体的系统提示词作为文本参数化策略，通过策略智能体利用积累的决策轨迹和已实现的投资组合反馈不断修订该策略，在保持骨干模型不变的情况下实现交易决策流程的自进化，从而在多种市场环境下提升夏普比率等交易表现。

    

    arXiv:2609.17632v1 公告类型：新论文 摘要：大语言模型（LLM）交易智能体能够结合市场数据、新闻和可执行的分析，但其行为通常由部署前固定的静态手工编写工具使用策略所控制。这限制了智能体在不断变化的市场环境下调整证据收集、工具调用、信号验证和风险管理方式的能力。我们提出了EvolveTrade，这是一个自进化框架，它将使用工具的交易智能体的系统提示词视为一种以文本为参数的策略。在每个更新间隔之后，策略智能体（Policy Agent）利用积累的决策轨迹和已实现的投资组合反馈来修订这一策略，同时保持骨干LLM不变。更新后的策略随后用于下一批交易决策，使智能体能够随时间不断精炼其信息获取和投资组合构建流程。在多个市场环境和两个LLM骨干模型上的实验表明，EvolveTrade通常能够提升夏普比率。

    arXiv:2609.17632v1 Announce Type: new  Abstract: Large language model (LLM) trading agents can combine market data, news, and executable analysis, but their behavior is often controlled by static hand-written tool-use policies that are fixed before deployment. This limits their ability to adapt how they gather evidence, invoke tools, verify signals, and manage risk under changing market regimes. We introduce EvolveTrade, a self-evolving framework that treats the system prompt of a tool-using trading agent as a text-parameterized policy. After each update interval, a Policy Agent revises this policy using accumulated decision traces and realized portfolio feedback, while keeping the backbone LLM fixed. The updated policy is then used for the next batch of trading decisions, enabling the agent to refine its information-acquisition and portfolio-construction procedure over time. Experiments across multiple market regimes and two LLM backbones show that EvolveTrade often improves Sharpe Ra
    
[^110]: 使政治文本标度可比：17种算法的基础设施与超参数敏感性

    Making Political Text Scaling Comparable: Infrastructure and Hyperparameter Sensitivity for 17 Algorithms

    [https://arxiv.org/abs/2609.17602](https://arxiv.org/abs/2609.17602)

    本文通过涵盖17种算法、5,537次实验和约425万个立场估计的大规模比较实验，论证了政治文本理想点估计方法应被视为可配置的测量流程而非固定算法，并发现绝大多数算法的估计结果对超参数选择并不敏感。

    

    基于计算文本的理想点估计方法通常以命名的算法形式进行比较，但应用这些方法涉及众多研究者的选择，这些选择决定了如何将政治文本转化为立场估计。本文认为，CT-IPE方法更适合被理解为可配置的测量流程，而非固定的估计器。基于一项涵盖17种CT-IPE算法、5,537次实验运行以及约425万个左右政治立场估计的大规模比较实验，作者描述了使这些异构方法能够联合执行的基础设施，并量化了其估计结果对替代性超参数选择的敏感性。方差分解和基于SHAP的敏感性分析表明，对于大多数算法而言，超参数配置通过共同偏移所解释的残差方差很少：17种算法中有13种的ICC值低于0.10。

    arXiv:2609.17602v1 Announce Type: new  Abstract: Computational text-based ideal point estimation (CT-IPE) methods are usually compared as named algorithms, yet applying them involves numerous researcher choices that configure how political text is turned into position estimates. This paper argues that CT-IPE methods are better understood as configurable measurement pipelines than as fixed estimators. Building on a large-scale comparative experiment spanning 17 CT-IPE algorithms, 5,537 experimental runs, and approximately 4.25 million left-right position estimates, I describe the shared infrastructure that makes these heterogeneous methods jointly executable and quantify how sensitive their estimates are to alternative hyperparameter choices. Variance-partitioning and SHAP-based sensitivity analyses show that, for most algorithms, hyperparameter profiles explain little residual variance through a shared shift: 13 of the 17 algorithms exhibit ICC values below .10. Where this profile-leve
    
[^111]: 2026年英语词义消歧：当标注成为瓶颈时

    English Word Sense Disambiguation in 2026: When the Labels Become the Bottleneck

    [https://arxiv.org/abs/2609.17554](https://arxiv.org/abs/2609.17554)

    该论文指出英语词义消歧的瓶颈已从模型转移到标注数据，并发布了经人工修正的lexEN评测基准与可审计的SenseBench评测框架，结果显示前沿大语言模型准确率已收敛至约95%，金标准中的标注错误成为决定排名的关键因素。

    

    在英语全词词义消歧（WSD）任务中，瓶颈已不再是模型，而是标注数据：前沿大语言模型已经足够准确，以至于金标准中残留的错误会决定基准测试的排名——无论是在我们用于评分的测试集上，还是如我们以因果方式证明的那样，在我们用于训练的语料库中。我们发布了lexEN，一个在Maru2022的ALL_NEW基准之上构建的保守的、经人工裁决的修正层组成的WSD评测基准（更改了211个标签，移除了56个）；以及SenseBench，一个可审计的大语言模型WSD评测框架和动态排行榜（涵盖57个模型、192次运行）。该任务是受词库约束的多选题形式（模型从提供的WordNet义项中进行选择），因此所报告的准确率是模型在没有这种辅助情况下所能达到的上限。在lexEN-v1上，前沿大语言模型的准确率收敛到约95%（最佳为95.6%），排名前三的模型家族在统计上难以区分，且准确率与推理开销和成本之间存在权衡关系……

    arXiv:2609.17554v1 Announce Type: new  Abstract: In English all-words word sense disambiguation (WSD), the labels, not the models, have become the bottleneck: frontier LLMs are accurate enough that the errors surviving in the gold standard decide benchmark rankings -- in the test sets we score on and, as we show causally, in the corpus we train on. We release lexEN, a WSD evaluation benchmark built as a conservative, human-adjudicated correction layer over Maru2022's ALL_NEW benchmark (211 labels changed, 56 removed), and SenseBench, an auditable LLM WSD evaluation harness and living leaderboard (57 models, 192 runs). The task is inventory-constrained multiple choice (the model picks from the supplied WordNet senses), so the reported accuracies are a ceiling on what models achieve without that help. On lexEN-v1 the frontier LLMs converge near 95% (best, 95.6%), the top three families are statistically indistinguishable, and accuracy trades off against reasoning effort and cost across a
    
[^112]: BPE分词在波兰语中的局限性：屈折语言模型中的切分-屈折形式、语法锚定与第一人称稳定性

    The Limits of BPE Tokenization in Polish: Segmentation-Flexional Forms, Grammatical Anchoring, and First-Person Stability in Inflectional Language Models

    [https://arxiv.org/abs/2609.17553](https://arxiv.org/abs/2609.17553)

    该研究以波兰语为测试案例，表明BPE分词在屈折语言中只能稳定高频的表面文字片段，无法系统性保留音位结构、屈折词尾和语法形式等语言学相关单位。

    

    本文将波兰语中的BPE分词作为测试案例，分析统计切分方法在屈折语言中的局限性。文章探讨基于频率的分词是否能够保留与语言学相关的单位，包括正字法形式、音位与音节切分、派生结构、屈折词尾、语法形式以及说话主体。研究材料包括诊断性词汇、一篇儿童文本、波兰共和国宪法序言中的选定形式、词族测试，以及包含波兰语变音符号和鼻元音的实例。研究表明，BPE分词器可能产生与音节划分或形态学可解释切分相吻合的片段，但其结果仍依赖于书面形式的频率。BPE无法系统地将正字法表示映射到音位结构或依赖语境的语音实现上。结果显示，BPE主要稳定了语法中高频出现的表面片段……（原文摘要至此截断）

    arXiv:2609.17553v1 Announce Type: new  Abstract: This article analyzes BPE tokenization in Polish as a test case for the limits of statistical segmentation in an inflectional language. It asks whether frequency-based tokenization preserves linguistically relevant units, including orthographic form, phonemic and syllabic segmentation, derivational structure, inflectional endings, grammatical form, and the speaking subject. The material includes diagnostic words, a children's text, selected forms from the Preamble to the Constitution of the Republic of Poland, word-family tests, and examples with Polish diacritics and nasal vowels. BPE tokenizers may produce segments that coincide with syllabic or morphologically interpretable divisions, but remain dependent on the frequency of written forms. They do not systematically map orthographic representation onto phonemic structure or context-dependent phonetic realization. The results show that BPE stabilizes frequent surface fragments of gramm
    
[^113]: 道德推理训练是有帮助还是有害？用人格攻击对强化学习训练的伦理智能体进行红队测试

    Does Moral Reasoning Training Help or Hurt? Red-Teaming RL-Trained Ethical Agents with Persona Attacks

    [https://arxiv.org/abs/2609.17552](https://arxiv.org/abs/2609.17552)

    研究发现道德奖励强化训练虽能将语言模型智能体对人格攻击的鲁棒性提升5.8倍，但会牺牲约11个百分点的ETHICS道德基准准确率，且模型内部存在单一表征方向即可恢复83%对抗效果，揭示了道德对齐的脆弱性机制。

    

    道德奖励强化学习可以使语言模型智能体更加合作，但这种对齐能否在对抗性人格压力下存活尚不清楚。这类攻击是现实存在的：检索到的上下文、工具输出或多轮对话框架都可能注入与智能体道德目标相竞争的角色指令。我们用五种人格攻击对经过道德训练的Gemma-2-27B/9B和Llama-3.1-8B智能体进行红队测试，然后通过噪声奖励对照、对抗性PPO、表征分析、引导和注意力头消融来探究因果关系。在27B模型上，道德强化学习将平均对抗性退化降低了5.2倍，但付出了约11个百分点的ETHICS准确率代价；在205个场景和5个随机种子上，推理层面的道德奖励产生了5.8倍的鲁棒性，而匹配的随机奖励则没有产生任何鲁棒性。该训练还重塑了表征几何结构（平均CKA为0.82/0.83，而噪声对照为0.98），将峰值攻击处理位置提前了8层，并揭示了一个rank-1的L21方向，该方向可恢复完整PPO的83%的对抗……

    arXiv:2609.17552v1 Announce Type: new  Abstract: Moral-reward RL can make language-model agents more cooperative, but whether that alignment survives adversarial persona pressure is unknown. Such attacks are realistic: retrieved context, tool outputs, or multi-turn framing can all inject role instructions that compete with the agent's moral objective. We red-team morally trained Gemma-2-27B/9B and Llama-3.1-8B agents with five persona attacks, then probe causality with noise-reward controls, adversarial PPO, representation analysis, steering, and head ablations. At 27B, moral RL cuts mean adversarial degradation by 5.2x but costs ~11pp ETHICS accuracy; across 205 scenarios and 5 seeds, reasoning-level moral reward yields 5.8x robustness while a matched random reward yields none. The training also reshapes representation geometry (mean CKA 0.82/0.83 vs. 0.98 for noise), moves peak attack processing 8 layers earlier, and exposes a rank-1 L21 direction that recovers 83% of full PPO's aver
    
[^114]: 两个小型LLM中不存在可用的线性“屈服方向”：激活转向声明的验证协议，以及反驳压力下谄媚行为的跨模型家族行为研究

    No Usable Linear "Capitulation Direction" in Two Small LLMs: A Validation Protocol for Activation-Steering Claims, and a Cross-Family Behavioral Study of Sycophancy Under Pushback

    [https://arxiv.org/abs/2609.17550](https://arxiv.org/abs/2609.17550)

    该研究通过验证协议发现两个小型LLM中不存在可用的线性“屈服方向”，并首次跨模型家族揭示：模型在反驳下放弃正确答案的谄媚比例高达41.8%-43.1%，且哪种反驳方式有效及失败模式均强烈依赖于具体模型家族。

    

    语言模型在用户反驳时经常放弃正确答案。我们在来自不同家族的两个小型指令微调模型Qwen2.5-1.5B和Llama-3.2-1B上，基于TriviaQA研究了这一现象：模型先给出答案，随后受到四种预设反驳风格之一的质疑，然后再次作答。在初始答案正确的条件下，两个模型分别在41.8%和43.1%的情况下转向错误答案。哪种压力有效是模型的属性而非压力本身的属性：同样的问题内配对比较（单纯质疑vs.情感诉求）事先预设，结果在两个模型家族中呈现方向相反的Bonferroni显著差异（Qwen：单纯质疑>情感诉求，OR 2.5，p=.040；Llama：情感诉求>单纯质疑，OR 4.0，p=.001）。失败模式也依赖于具体模型：Llama放弃答案且不重新承诺的比率是Qwen的六倍（8.2% vs. 1.4%）。相同的反驳仅在约13%的情况下修复初始错误答案；反驳在认知上整体是净负面的

    arXiv:2609.17550v1 Announce Type: new  Abstract: Language models frequently abandon correct answers when users push back. We study this in two small instruction-tuned models from different families, Qwen2.5-1.5B and Llama-3.2-1B, over TriviaQA: the model answers, is challenged with one of four scripted pushback styles, and answers again. Conditioned on an initially correct answer, the models flip to a wrong answer in 41.8% and 43.1% of episodes. Which pressure works is a property of the model, not the pressure: the same within-question paired comparison (bare doubt vs. emotional appeal), specified in advance, is Bonferroni-significant in opposite directions across families (Qwen: bare doubt > emotional, OR 2.5, p=.040; Llama: emotional > bare doubt, OR 4.0, p=.001). Failure mode is also model-dependent: Llama abandons answers without recommitting at six times Qwen's rate (8.2% vs. 1.4%). Identical pushback repairs initially wrong answers only ~13% of the time; pushback is net epistemic
    
[^115]: 社会模式在合成数据中是否成立？分析LLM生成对话与真实对话中的网络欺凌动态

    Do Social Patterns Hold in Synthetic Data? Analyzing Cyberbullying Dynamics in LLM-Generated and Authentic Dialogues

    [https://arxiv.org/abs/2609.17549](https://arxiv.org/abs/2609.17549)

    本文提出了一个评估LLM生成网络欺凌对话社会真实性的综合框架，从互动结构、语言风格、情感行为标记和时间升级动态等多维度比较GPT、Grok和LLaMA生成的合成对话与真实对话，揭示了合成数据在再现真实社会动态方面的局限性。

    

    网络欺凌（CB）是一种复杂的社会现象，其特征包括重复性攻击、权力失衡和多方互动。尽管大语言模型（LLM）越来越多地被用于生成合成网络欺凌对话以进行数据增强和基准测试，但此类数据除了支持下游任务性能之外，是否能忠实再现真实互动的社会动态仍不清楚。我们提出了一个用于评估LLM生成的网络欺凌对话社会真实性的综合框架。我们将真实对话与由GPT、Grok和LLaMA生成的合成对话在互动结构（话轮转换、权力动态和修复行为）、语言和风格真实性（代词使用和幽默）、情感和行为标记（网络欺凌类型、脏话和毒性）以及时间升级动态等方面进行比较。我们进一步通过人类对网络欺凌存在性的评估来补充自动分析。

    arXiv:2609.17549v1 Announce Type: new  Abstract: Cyberbullying (CB) is a complex social phenomenon characterized by repeated aggression, power imbalance, and multi-party interaction. Although large language models (LLMs) are increasingly used to generate synthetic CB conversations for data augmentation and benchmarking, it remains unclear whether such data faithfully reproduces the social dynamics of authentic interactions beyond supporting downstream task performance. We present a comprehensive framework for evaluating the social realism of LLM-generated CB conversations. We compare authentic and synthetic dialogues generated by GPT, Grok, and LLaMA across interactional structure (turn-taking, power dynamics, and repair behavior), linguistic and stylistic realism (pronoun usage and humor), affective and behavioral markers (CB types, profanity, and toxicity), and temporal escalation dynamics. We further complement automatic analyses with a human evaluation of cyberbullying presence, sc
    
[^116]: Myovox：从面部肌肉读取语音

    Myovox: Reading Speech from the Muscles of the Face

    [https://arxiv.org/abs/2609.17548](https://arxiv.org/abs/2609.17548)

    Myovox通过忠实复现基线、采用跨模态蒸馏训练的双向Conformer编码器以及多模型集成重排序三个步骤，将面部肌电信号解码开放词汇语音的单词错误率从51.17%大幅降低至18.53%。

    

    Myovox（myo 意为肌肉，vox 意为声音）从发声说话期间记录的面部肌肉31通道表面肌电图中解码开放词汇的英语文本。它通过三个可分离的步骤，将已发表的单词错误率为51.17%的单受试者emg2speech通用语料库提升至18.53%，且每一步都单独测量。首先，作者恢复了公开发布中缺失的开放词汇解码设置，得到了可靠的40.63% WER / 39.02% PER基线，其音素错误率与已发表结果的差距在0.8个百分点以内，证明声学模型被忠实复现。其次，作者用双向Conformer替换因果编码器，并通过针对并行音频的WavLM-Large第9层特征进行四项跨模态蒸馏训练，仅凭肌电信号就达到了26.14% WER / 22.34% PER。第三，作者对两个声学模型进行集成，合并它们的多尺度n-best列表，并使用QLoRA微调（原文在此截断）……

    arXiv:2609.17548v1 Announce Type: new  Abstract: Myovox, from myo (muscle) and vox (voice), decodes open-vocabulary English text from 31-channel surface electromyography (sEMG) recorded from the muscles of the face during vocalized speech. It takes the single-subject emg2speech General Corpus from a published 51.17% word error rate to 18.53%, in three separable moves, each measured in isolation. First, I recover the open-vocabulary decode settings missing from the public release and reach a faithful 40.63% WER / 39.02% PER baseline whose phone error rate matches the published one to within 0.8 points, so the acoustic model is reproduced faithfully. Second, I replace the causal encoder with a bidirectional Conformer trained by a four-term cross-modal distillation against the parallel audio's WavLM-Large layer-9 features, reaching 26.14% WER / 22.34% PER from the electromyography alone. Third, I ensemble two acoustic models, union their multi-scale n-best lists, and rerank with a QLoRA-f
    
[^117]: AI助手如何应对反复辱骂

    How AI Assistants Respond to Repeated Abuse

    [https://arxiv.org/abs/2609.17547](https://arxiv.org/abs/2609.17547)

    该研究提出了一个双语多轮评估框架，首次区分AI助手面对反复言语辱骂时的“硬性脱离”与“软性退出”两种行为，并发现不同模型配置间的硬性脱离率差异巨大（从0%到50%）。

    

    AI助手被期望在困难的互动中保持有用性，但关于反复的言语辱骂如何改变它们对原本良性任务的投入，人们知之甚少。我们提出了一个双语、多轮框架，将“硬性脱离”（即无条件声明不再继续且未给出恢复途径）与“软性退出”（即保持可用性、可观察的任务相关工作以及边界设定）区分开来。八个特定时间点的API配置各自贡献了48个升级对话和八个较小的持续受挫对照组，共产生448个五轮对话、2,240条回复和6,720个元数据盲评的模型判断。主要结果采用每个配置48个升级对话的持续辱骂终点数据。硬性脱离的比例从四个配置中的0/48到Gemini 3.1 Pro的24/48（50.0%）不等，存在显著的与配置相关的异质性（匹配标签蒙特卡

    arXiv:2609.17547v1 Announce Type: new  Abstract: AI assistants are expected to remain useful during difficult interactions, but little is known about how repeated verbal abuse changes their engagement with an otherwise benign task. We contribute a bilingual, multi-turn framework that separates hard disengagement, an unconditional statement of noncontinuation with no stated route to resume, from soft withdrawal, continued availability, observable task-related work, and boundary setting. Each of eight time-specific API configurations contributed 48 escalation conversations and eight smaller constant-frustration comparisons, giving 448 five-turn conversations, 2,240 responses, and 6,720 metadata-blinded model judgments. Primary results use the sustained-abuse endpoint of the 48 escalation conversations per configuration. Hard disengagement ranged from 0/48 in four configurations to 24/48 (50.0%) for Gemini 3.1 Pro, with strong configuration-associated heterogeneity (matched-label Monte Ca
    
[^118]: 法律大语言模型的幻觉应被评估为法律授权的失败

    Legal LLM Hallucination Should Be Evaluated as Failure of Legal Warrant

    [https://arxiv.org/abs/2609.17546](https://arxiv.org/abs/2609.17546)

    本文主张法律大语言模型的幻觉应被评估为“法律授权的失败”——即法律主张未能与真实存在、现行有效、适用于相关管辖区的法律权威建立支持关系——并提出了能捕捉现有准确性与引用类评估方法所遗漏的重大失败的可证伪授权指标。

    

    在这篇立场论文中，我们主张法律大语言模型的幻觉应被评估为“法律授权”（legal warrant）的失败，而非事实不准确或引用失败。我们定义“主张-权威授权”为具有后果性的法律主张与法律权威之间的一种情境敏感关系，该权威必须真实存在、适用于相关司法管辖区、在分析日期时仍然有效、具有系统所声称的法律地位，并能支持所断言的命题。“有授权的法律生成”是一种更广泛的系统行为，它根据这种关系来回答、收窄问题、追问、发出警告、纠正错误前提或拒绝回答。我们提出一个可证伪的预测：授权指标能够揭示现有方法（如回答准确性、引用存在性、通用归因、LegalHalBench式法条相关性以及CitaLaw式句子-引用对齐）可能遗漏的重大失败。我们通过一个并排比较的示例和一个小的、可复现的试点实验来强化这一主张。

    arXiv:2609.17546v1 Announce Type: new  Abstract: In this position paper, we argue that legal LLMs' hallucinations should be evaluated as a failure of legal warrant rather than as factual inaccuracy or citation failure. We define claim-authority warrant as the context-sensitive relation between a consequential legal claim and authority that exists, applies to the relevant jurisdiction, is current for the date of analysis, has the legal status represented by the system, and supports the proposition asserted. Warranted legal generation is the broader system behavior that answers, narrows, asks, warns, corrects a false premise, or abstains according to that relation. The falsifiable prediction is that warrant metrics reveal material failures that answer accuracy, citation existence, generic attribution, LegalHalBench-style statute relevance, and CitaLaw-style sentence-citation alignment can miss. We sharpen this claim with a side-by-side comparison item and a small, reproducible pilot over
    
[^119]: 大语言模型与中医医师的对比：真实世界临床病例评估

    Large Language Models Versus Physicians in Traditional Chinese Medicine: A Real-World Clinical Case Evaluation

    [https://arxiv.org/abs/2609.17544](https://arxiv.org/abs/2609.17544)

    本研究构建了涵盖62家医院349例门诊病例的中医临床病例库，发现前沿通用大语言模型在多数诊疗维度上的专家评分超过执业中医医师，但在处方层面的中药选择、剂量与治疗策略上仍存在差异，并伴有幻觉和模板化输出等安全性问题。

    

    大语言模型（LLM）在临床应用中的探索日益增多，但针对真实世界中医（TCM）实践的评估仍然有限。我们构建了一个包含来自62家医院的349例去标识化门诊病例的临床病例库，并从该库中选取60个代表性病例，对16个大语言模型以及由60名执业中医医师组成的对照队列进行了评估。模型输出和医师报告经过匿名化处理，由五名资深中医专家在九个诊断与治疗维度上进行评分。前沿的通用大语言模型获得了高于医师对照组的专家评分，尤其是在医疗建议、治疗原则和部分诊断任务方面表现突出。然而，处方层面的分析揭示了中药选择、剂量和治疗策略方面的差异，定性安全性审查还发现了幻觉现象和不理想的模板化输出。

    arXiv:2609.17544v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly being explored for clinical applications, yet their assessment for real-world traditional Chinese medicine (TCM) practice remains limited We constructed a clinical case library comprising 349 de-identified outpatient cases from 62 hospitals and evaluated 16 LLMs and a comparator cohort of 60 practicing TCM physicians using 60 representative cases selected from this library. Model outputs and physician reports were anonymized and scored by five senior TCM experts across nine diagnostic and therapeutic dimensions. Cutting-edge general-purpose LLMs achieved higher expert scores than the physician comparators, particularly for medical advice, treatment principles and selected diagnostic tasks. However, prescription-level analyses revealed discrepancies in herb selection, dosage, and treatment strategy, and qualitative safety review identified hallucinations and undesirable template-driven outputs
    
[^120]: 基于复杂度的大语言模型路由中的语域偏差

    Register Bias in Complexity-Based Large Language Model Routing

    [https://arxiv.org/abs/2609.17542](https://arxiv.org/abs/2609.17542)

    基于复杂度的大语言模型路由存在语域偏差：非标准英语（如非裔美国人英语或二语学习者英语）因省略功能词而显得更短，会被系统性地路由到能力更低的模型，导致服务质量受损。

    

    大语言模型服务越来越倾向于将每个查询路由到多个能力不同的模型之一，通过对查询复杂度的廉价估计，将简单查询发送给小模型，将困难查询发送给大模型。本文表明，这一路由步骤并非语域中立的：以非标准英语语域（如非裔美国人英语或第二语言作者的英语）书写的文本，会系统性地被分配到比语义等同的标准英语版本查询更低能力的层级。这种效应由一个特定且常见的路由信号——输入长度——所驱动，因为非标准语域省略了功能词，因此看起来更短、从而显得更简单；其他复杂度信号并不携带这种偏差。作者在37,704个真实学习者句子对和一个受控平行语料库上证明了这种差异。随后，作者在设备端、边缘端和云端模型组成的模型阶梯上测量了质量后果，发现这种损害是由性能差异所驱动的。

    arXiv:2609.17542v1 Announce Type: new  Abstract: Large language model services increasingly route each query to one of several models of differing capability, using a cheap estimate of query complexity to send easy queries to small models and hard queries to large ones. I show that this routing step is not register neutral: text written in a non-standard English register, African American English or the English of second-language writers, is systematically assigned a lower-capacity tier than a meaning-equivalent standard-English version of the same query. The effect is driven by a specific, common routing signal, input length, because non-standard registers omit function words and thus look shorter and therefore simpler; other complexity signals do not carry it. I demonstrate the disparity on 37,704 authentic learner sentence pairs and on a controlled parallel corpus. I then measure the quality consequence on a device, edge, and cloud model ladder and find that the harm is driven by pe
    
[^121]: MudawanSn：一个面向机器翻译的沃洛夫语-阿拉伯语黄金标准平行语料库

    MudawanSn: A Gold-Standard Wolof-Arabic Parallel Corpus for Machine Translation

    [https://arxiv.org/abs/2609.17539](https://arxiv.org/abs/2609.17539)

    本文发布了首个专门面向沃洛夫语-现代标准阿拉伯语语言对的黄金标准平行语料库MudawanSn（包含1,271个人工翻译的句对齐句对），并通过实验证明在该语料库上微调能显著提升双向机器翻译性能。

    

    我们提出了MudawanSn，这是一个黄金标准资源，包含1,271个由人工从沃洛夫语翻译成现代标准阿拉伯语（MSA）的句对齐句对。源文本选自MasakhaNER语料库，涵盖塞内加尔新闻话语中的政治、社会、宗教和体育主题。尽管FLORES-200和NTREX等多语言资源同时包含沃洛夫语和阿拉伯语，但目前尚无公开可用的、专门针对沃洛夫语-现代标准阿拉伯语这一语言对的平行语料库。我们描述了语料库的构建协议、句子对齐流程和质量控制工作流程。我们对涵盖三种架构体系的四个机器翻译系统进行了基准测试：NLLB-200（600M）、mT5-base以及两个AfriNLLB变体，结果表明在MudawanSn上进行微调在两个翻译方向上都带来了显著提升。表现最佳的模型AfriNLLB-12在沃洛夫语到阿拉伯语方向上达到7.76 BLEU和30.72 chrF++，在阿拉伯语到沃洛夫语方向上达到8.75 BLEU和33……（摘要原文在此处被截断）

    arXiv:2609.17539v1 Announce Type: new  Abstract: We present MudawanSn, a gold-standard resource of 1,271 sentence-aligned pairs manually translated from Wolof into Modern Standard Arabic (MSA). The source texts are drawn from the MasakhaNER corpus and cover politics, society, religion, and sports in Senegalese news discourse. Although multilingual resources such as FLORES-200 and NTREX include both Wolof and Arabic, no publicly available parallel corpus is specifically designed for the Wolof-Modern Standard Arabic language pair. We describe the corpus construction protocol, sentence alignment procedure, and quality-control workflow. We benchmark four machine translation systems spanning three architectural families: NLLB-200 (600M), mT5-base, and two AfriNLLB variants, showing that fine-tuning on MudawanSn yields substantial improvements in both translation directions. The best-performing model, AfriNLLB-12, achieves 7.76 BLEU and 30.72 chrF++ for Wolof-to-Arabic, and 8.75 BLEU and 33.
    
[^122]: 从像素到配对：噪声文档环境下基于大语言模型的键值对提取综合基准测试

    From Pixels to Pairs: A Comprehensive Benchmark of LLM-Based Key-Value Extraction in Noisy Document Settings

    [https://arxiv.org/abs/2609.17538](https://arxiv.org/abs/2609.17538)

    该论文建立了一个系统性基准，评估开源指令微调大语言模型在干净文本与含噪OCR条件下提取键值对的能力，发现现代LLM在高质量文本输入下是强大的语义提取器（部分情况接近有监督布局感知系统），但在OCR噪声下性能显著下降。

    

    大语言模型（LLMs）越来越多地被用于从文档中提取结构化信息，但它们在真实OCR噪声下的行为仍然知之甚少。我们提出了一个系统性基准测试，评估开源指令微调大语言模型在干净文本和含噪OCR条件下进行键值对（KVP）提取的表现。我们在FUNSD、CORD和SROIE基准数据集上评估了代表性的仅解码器模型（Gemma、Mistral、Qwen2.5、LLaMA 3和DeepSeek），使用金标准文本标注以及来自PaddleOCR、EasyOCR和Tesseract的OCR输出。统一的评估协议在一致的条件下分离了输入质量、模型设计和提示方式的影响。结果表明，当有高质量文本可用时，现代大语言模型表现出强大的语义提取能力，在某些情况下接近有监督的布局感知系统。然而，在OCR噪声下，性能显著下降，模型之间的性能差距……

    arXiv:2609.17538v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used for structured information extraction from documents, yet their behavior under realistic OCR noise remains poorly understood. We present a systematic benchmark of open-source instruction-tuned LLMs for key-value pair (KVP) extraction under both clean-text and noisy OCR conditions.   We evaluate representative decoder-only models (Gemma, Mistral, Qwen2.5, LLaMA 3, and DeepSeek) on the FUNSD, CORD, and SROIE benchmarks using both Gold-text annotations and OCR outputs from PaddleOCR, EasyOCR, and Tesseract. A unified evaluation protocol isolates the effects of input quality, model design, and prompting under consistent conditions.   The results show that modern LLMs act as strong semantic extractors when high-quality text is available, in some cases approaching supervised layout-aware systems. Under OCR noise, however, performance degrades substantially and performance gaps between models n
    
[^123]: 关系先于实体：语言模型事实回忆中的延迟承诺

    Relation Before Entity: Deferred Commitment in Language Model Factual Recall

    [https://arxiv.org/abs/2609.17537](https://arxiv.org/abs/2609.17537)

    该研究发现语言模型在事实回忆中，关系信息比实体信息早10-16个层变得控制生成，而实体信息并非早期缺失，其“承诺”被延迟至被路由到最后token位置才具有因果控制作用。

    

    我们探究在事实回忆过程中，关系类型信息（例如“首都关系”）与实体特定信息（例如“法国→巴黎”）是否在网络的相同深度处于最后一个token位置变得具有因果活性。通过在四个仅解码器模型和八个提示族上使用四种互补的因果诊断方法，我们发现了一个稳健的时间不对称性：关系信息比实体信息更早开始控制生成过程。在阈值0.4下，关系信息的起始点比实体信息早10-16个测试层（占网络深度的31-44%），且这一顺序在0.2-0.5阈值范围内的所有16种模型-阈值组合中均成立。关键的是，实体信息并非在早期缺失：实体token修补在早期层成功率达到90-100%。相反，实体对生成的“承诺”是被延迟的：实体信息在实体token位置是可用的，但只有被路由到最后一个token之后，才在该位置变得控制生成。

    arXiv:2609.17537v1 Announce Type: new  Abstract: We ask whether relation-type information (e.g., capital-of) and entity-specific information (e.g., France to Paris) become causally active at the final-token position at the same depth during recall. Using four complementary causal diagnostics across four decoder-only models and eight prompt families, we find a robust temporal asymmetry: relation information becomes generation-controlling before entity information does. Relation onset precedes entity onset by 10-16 tested layers (31-44% of network depth) at threshold 0.4, with the ordering holding across all 16 model-threshold combinations for thresholds 0.2-0.5. Critically, entity information is not absent early: entity-token patching succeeds at 90-100% in early layers. Instead, entity commitment to generation is deferred: entity information is available at the entity-token position but becomes generation-controlling at the final token only after being routed there.
    
[^124]: 三思而后安慰：面向协议驱动的老年人认知刺激智能体的反思性认知对齐

    Think Before You Comfort: Reflective Cognitive Alignment for Protocol-Grounded Elderly Stimulation Agents

    [https://arxiv.org/abs/2609.17536](https://arxiv.org/abs/2609.17536)

    本文提出结合STaR-CS数据合成方法与反思性认知对齐（RCA）框架，将认知刺激交互建模为协议约束的序贯决策过程，使大语言模型智能体能够在粤语等低资源场景下兼顾共情陪伴与认知刺激协议的严格遵循。

    

    认知刺激疗法（CST）为认知障碍老年人提供非药物治疗支持，然而其可扩展性仍受限于对训练有素引导员的依赖以及严重的数据稀缺问题，尤其是对于像粤语这类隐私敏感的低资源语言。虽然大语言模型（LLMs）在自动化陪伴方面展现出潜力，但它们往往难以在共情互动与遵循认知刺激指南之间取得平衡。我们提出了一个从两个互补维度应对这些挑战的框架。首先，STaR-CS（风格迁移与角色条件化认知刺激）通过引导员风格建模和结构化骨架提取来合成多方对话，从而缓解数据壁垒。基于该语料库，反思性认知对齐（RCA）框架将刺激交互建模为序贯决策过程，集成了协议约束的认知链（……原文摘要在此处截断）

    arXiv:2609.17536v1 Announce Type: new  Abstract: Cognitive Stimulation Therapy (CST) offers non-pharmacological support for elders with cognitive impairment, yet scalability remains constrained by reliance on trained facilitators and severe data scarcity, particularly for privacy-sensitive, low-resource languages such as Cantonese. While Large Language Models (LLMs) show promise for automated companionship, they often struggle to balance empathetic engagement with adherence to cognitive stimulation guidelines. We propose a framework addressing these challenges along two complementary axes. First, STaR-CS (Style-Transfer and Role-Conditioned Cognitive Stimulation) synthesizes multi-party dialogues through facilitator style modeling and structured skeleton extraction, mitigating data barriers. Building upon this corpus, the Reflective Cognitive Alignment (RCA) framework models stimulation interactions as a sequential decision process, integrating Protocol-Constrained Chain-of-Cognition (
    
[^125]: DANTINOX：一个多范式语言建模的统一框架

    DANTINOX: A Unified Framework for Multi-Paradigm Language Modeling

    [https://arxiv.org/abs/2609.17535](https://arxiv.org/abs/2609.17535)

    DANTINOX是一个开源的JAX/Flax库，通过统一的模块化Transformer骨干网络支持自回归解码、离散掩码扩散和连续流匹配三种语言生成范式，从而实现了可控的跨范式公平比较。

    

    语言生成研究日益涵盖三种范式：自回归解码、离散掩码扩散和连续流匹配。由于每种范式都存在于独立的代码库中，因此对它们进行比较十分困难，这导致测量出的差异往往反映的是实现细节，而非范式本身。我们提出了DantinoX，这是一个开源的JAX/Flax库，其中单一的模块化Transformer骨干网络可服务于所有三种范式。切换生成范式、注意力机制或硬件拓扑结构只需修改配置，而骨干网络架构、分词器、初始化策略和训练基础设施均保持一致。这使得在同一个API内进行受控的跨范式比较成为可能，涵盖训练、流式推理和基准测试。

    arXiv:2609.17535v1 Announce Type: new  Abstract: Language generation research increasingly spans three paradigms: autoregressive decoding, discrete masked diffusion, and continuous flow-matching. Comparing them is difficult because each lives in a separate codebase, so measured differences often reflect implementation details rather than the paradigms themselves. We present DantinoX, an open-source JAX/Flax library in which a single modular Transformer backbone serves all three paradigms. Switching the generation paradigm, attention mechanism, or hardware topology requires only a configuration change, while the backbone architecture, tokenizer, initialization strategy, and training infrastructure remain consistent. This enables controlled cross-paradigm comparisons within one API for training, streaming inference, and benchmarking.
    
[^126]: 大语言模型中的“装好”与“装坏”：黑暗三联征人格特质中的反应失真

    Faking Good and Faking Bad in LLMs: Response Distortion Across Dark Triad Personality Traits

    [https://arxiv.org/abs/2609.17534](https://arxiv.org/abs/2609.17534)

    本研究首次系统证实，当代大语言模型会像人类一样响应“装好”与“装坏”的社会赞许性激励，系统性地调节黑暗三联征人格特质的表达，揭示了大语言模型在人格评估中存在类似人类的反应失真现象。

    

    社会赞许性和印象管理是人类人格评估中普遍存在的反应失真来源，然而其对大语言模型（LLMs）的影响仍缺乏充分研究。本研究考察了当代大语言模型在“装好”和“装坏”条件下是否会系统性地调节黑暗三联征特质（马基雅维利主义、自恋和精神病态）的表达。研究在两个具有生态效度的情境中评估了七个最先进的模型：就业选拔和司法评估，通过情境框架传达社会赞许或社会不赞许的激励。特质表达采用标准心理测量评分程序进行测量，并在总体和题目两个层面与自我评估基线进行比较。结果显示出系统性的、与条件一致的反应调节：大多数模型在“装好”条件下降低了黑暗三联征得分，而在“装坏”条件下则提高了得分。

    arXiv:2609.17534v1 Announce Type: new  Abstract: Social desirability and impression management are pervasive sources of response distortion in human personality assessment, yet their effects on Large Language Models (LLMs) remain underexplored. This study investigates whether contemporary LLMs systematically modulate the expression of Dark Triad traits (Machiavellianism, narcissism, and psychopathy) under fake-good and fake-bad conditions. Seven state-of-the-art models were evaluated across two ecologically relevant contexts: employment selection and forensic evaluation, in which socially desirable or undesirable incentives were conveyed through contextual framing. Trait expression was measured using standard psychometric scoring procedures and compared with self-assessment baselines at both aggregate and item levels. Results revealed systematic and condition-consistent response modulation. Most models reduced Dark Triad scores under fake-good conditions and increased them under fake-b
    
[^127]: 利用大语言模型从呼吸治疗临床笔记中提取的特征增强拔管失败预测

    Enhancing Extubation Failure Prediction with LLM-Derived Features from Respiratory Therapy Clinical Notes

    [https://arxiv.org/abs/2609.17532](https://arxiv.org/abs/2609.17532)

    该论文提出利用大语言模型从自由文本呼吸治疗临床笔记中提取特征，与结构化数据结合后显著提升了拔管失败预测性能，并揭示了既往研究在目标人群定义上的差异如何阻碍模型的泛化能力。

    

    有创机械通气是一种挽救生命的治疗手段，但及时、安全地停用对于预防拔管失败（EF）及其相关健康风险至关重要。我们提出了一种新颖的拔管失败预测方法，该方法利用大语言模型和逻辑回归流程，从自由文本呼吸治疗记录中分类提取特征。将该方法应用于华盛顿大学医学院的患者队列，我们的方法识别出了具有临床意义的拔管失败相关特征，这些特征与结构化患者数据结合使用时，能够提升拔管失败预测性能。我们进一步强调了既往拔管失败预测研究中目标人群的差异（例如异质性的纳入标准和拔管失败定义）如何导致模型性能的系统性差异，并阻碍研究结果之间的泛化能力。

    arXiv:2609.17532v1 Announce Type: new  Abstract: Invasive mechanical ventilation is a lifesaving therapy, but timely, safe discontinuation is essential to preventing extubation failure (EF) and related risks to health. We present a novel approach to EF prediction that leverages features classified in free-text respiratory therapy notes using a large language model and logistic regression pipeline. Applied to a patient cohort from University of Washington Medicine, our method identifies clinically meaningful EF-related features that improve EF prediction performance when included alongside structured patient data. We further highlight how differences in target populations in prior EF prediction studies, such as heterogenous inclusion criteria and EF definition, can lead to systematic differences in model performance and hinder generalizability between studies.
    
[^128]: Vroom-Vroom参加SHROOM-Visions竞赛：一种用于检测视觉-语言输出中幻觉片段的多裁判委员会方法

    Vroom-Vroom at SHROOM-Visions: A Multi-Judge Committee for Detecting Hallucinated Spans in Vision-Language Outputs

    [https://arxiv.org/abs/2609.17327](https://arxiv.org/abs/2609.17327)

    该论文提出一种多裁判委员会方法，通过多个微调视觉-语言模型的字符级多数投票来检测视觉-语言输出中的幻觉片段，在SHROOM-Visions竞赛四种语言中的三种排名第一。

    

    本文描述了我们对SHROOM-Visions共享任务的提交方案，该任务旨在检测和分类视觉-语言模型输出中四种语言里的幻觉字符片段。我们采用多个微调后的视觉-语言模型作为独立标注者，并通过字符级多数投票融合它们的片段预测结果，此外还探索了激活探针方法。该方法在四种语言中的三种排名第一，并在所有语言和指标上均登上领奖台。我们的分析表明，不同模型之间的分歧能够反映人类标注者之间的分歧。

    arXiv:2609.17327v1 Announce Type: cross  Abstract: This paper describes our submission to the SHROOM-Visions shared task on detecting and classifying hallucinated character spans in vision-language model outputs across four languages. We employ several fine-tuned vision-language models as independent annotators and combine their span predictions through character-level majority voting, and additionally explore activation probes. The approach ranks first in three of four languages and places on the podium in every language and metric. Our analysis indicates that disagreement among diverse models tracks disagreement among human annotators.
    
[^129]: 社交媒体消息中的零样本叙事检测

    Zero-shot narrative detection in social messaging

    [https://arxiv.org/abs/2609.17310](https://arxiv.org/abs/2609.17310)

    本研究证明大语言模型在零样本设置下，只需提供人工编写的叙事描述即可有效检测社交消息中的隐藏战略性叙事，无需训练样本，且集成方法与更大规模模型可进一步提升检测性能与鲁棒性。

    

    本研究调查了大语言模型（LLM）在社交消息中识别和分类隐藏叙事的零样本能力。我们的研究假设是，LLM丰富的上下文知识使其能够在更深层次的语用层面上解释消息，超越基本的情感或主题分析。在Dipromats和SemEval数据集上的实验表明，为模型提供人工编写的叙事描述可以显著提高性能，而无需任何训练样本。相比之下，自动生成的描述或使用少量示例（few-shot）往往会由于措辞框架的细微变化而降低准确性。研究还发现，集成方法（特别是多数投票法）能够增强鲁棒性，并且更大的模型表现最佳，同时对提示词变化也不太敏感。研究结果验证了LLM可以在零样本设置中有效检测战略性叙事，并且当……

    arXiv:2609.17310v1 Announce Type: new  Abstract: This study investigates the zero-shot ability of large language models (LLMs) to identify and classify hidden narratives in social messages. Our research hypothesis is that LLMs' extensive contextual knowledge allows them to interpret messages on a deeper, pragmatic level, going beyond basic sentiment or topic analysis. Experiments on the Dipromats and SemEval datasets show that providing models with human-written narrative descriptions significantly improves performance, without the need of training examples. In contrast, automatically generated descriptions or the use of few examples (few-shot) often degrade accuracy due to subtle shifts in framing. The study also finds that ensemble methods, particularly majority voting, enhance robustness and that larger models perform best while also being less sensitive to prompt variations. The findings validate that LLMs can effectively detect strategic narratives in a zero-shot setting, and when
    
[^130]: 面向波兰语和欧洲语言的参数高效检索器

    Parameter-Efficient Retrievers for Polish and European Languages

    [https://arxiv.org/abs/2609.12913](https://arxiv.org/abs/2609.12913)

    提出了一种结合跨语言对齐、关系知识蒸馏和对比微调的三阶段训练流程，无需原始相关性标注即可训练出参数量小但性能可媲美大型模型的波兰语和欧洲多语言稠密检索器。

    

    密集检索系统日益依赖数十亿参数规模的语言模型，其内存和计算需求使得大规模索引、频繁的语料库更新以及低延迟服务变得成本高昂。我们提出了一套三阶段训练流程，用于开发紧凑且高效的检索器，这些检索器在与体量大得多的模型竞争时仍具有竞争力。该流程结合了跨语言对齐、关系知识蒸馏和对比微调。它不需要原始的真实相关性标注，完全依赖由作为教师的强嵌入模型和重排序器所生成的监督信号。利用该流程，我们开发了PolDense和EuroDense，两者均支持长达8,192个token的上下文。PolDense是一系列参数量从1700万到10亿不等的六个波兰语检索器。EuroDense是一个支持九种欧洲语言的4.35亿参数检索器。我们进行了广泛的评估，涵盖41个波兰语

    arXiv:2609.12913v1 Announce Type: new  Abstract: Dense retrieval systems increasingly rely on multi-billion-parameter language models, whose memory and computational requirements make large-scale indexing, frequent corpus updates, and low-latency serving costly. We present a three-stage training pipeline for developing compact and efficient retrievers that remain competitive with substantially larger models. The pipeline combines cross-lingual alignment, relational knowledge distillation, and contrastive fine-tuning. It requires no original ground-truth relevance labels, relying exclusively on supervision generated by strong embedding models and rerankers utilised as teachers. Using this pipeline, we develop PolDense and EuroDense, both supporting contexts of up to 8,192 tokens. PolDense is a family of six Polish retrievers ranging from 17M to 1B parameters. EuroDense is a 435M-parameter retriever supporting nine European languages. We conduct an extensive evaluation covering 41 Polish
    
[^131]: 创建面向人格感知大语言模型交互的原子用户模型

    Creating an Atomic User Model for Personality-Aware Large Language Model Interaction

    [https://arxiv.org/abs/2609.12086](https://arxiv.org/abs/2609.12086)

    该论文提出原子用户模型（AUM），将用户表示为稳定身份核心加四个可解释外壳的分层结构，解决了现有助手仅依赖偏好总结而在任务变化时需反复重新学习用户的问题，并首次刻画了“人格渗漏”现象。

    

    基于大语言模型的智能助手被期望能够像其用户那样进行写作，而主流方法是单通道的：从对话历史中总结用户偏好并重新插入到上下文中。这种做法颠倒了推理的顺序。偏好是相对稳定的人格结构中依赖于任务的表层，因此仅存储偏好的系统在任务发生变化时就需要重新学习用户。首先，我们刻画了“人格渗漏”现象，即提示词的语言表面携带了人格指纹，助手在无法接触到其背后真实人格的情况下对其进行镜像模仿。其次，我们提出了原子用户模型（AUM），这是一种人类可读的用户表示方式，将一个人组织为稳定的身份核心，周围环绕着四个可解释的外壳（心理、认知与经验、行为以及社会），并辅以记录内部冲突与真实性的跨外壳条目。第三，我们将AUM视为一种检索索引……

    arXiv:2609.12086v1 Announce Type: cross  Abstract: Assistants built on large language models are expected to write as their user would, and the dominant approach is single-channel: preferences summarised from conversation history and reinserted into context. This inverts the order of inference. Preferences are the task-dependent surface of a comparatively stable personality structure, so a system storing only preferences relearns the person whenever the task changes. First, we characterise personality seepage, where a prompt's linguistic surface carries a personality fingerprint the assistant mirrors without access to the personality behind it. Second, we propose the Atomic User Model (AUM), a human-readable representation organising a person as a stable identity nucleus with four interpretable shells (psychological, cognitive and experiential, behavioural, and social), plus cross-shell entries recording internal conflict and authenticity. Third, we treat AUM as a retrieval index over 
    
[^132]: 当个性遇上量化：量化大型语言模型的逐层MBTI分析

    When Personality Meets Quantization: A Layer-wise MBTI Analysis of Quantized LLMs

    [https://arxiv.org/abs/2608.25977](https://arxiv.org/abs/2608.25977)

    本文首次系统分析了量化大型语言模型在不同精度下的个性特征，并提出了新方法来揭示个性如何在层间涌现及推理时漂移。

    

    arXiv:2608.25977v1 公告类型：新 摘要：个性在大型语言模型（LLMs）中日益重要，因为它塑造了用户的信任、参与度和情感体验。虽然迈尔斯-布里格斯类型指标（MBTI）已成为评估LLMs个性的常用框架，但现有研究主要关注全精度模型，且仅评估最终输出。这些研究忽视了需要低内存占用的量化LLMs的广泛部署，其个性特征仍未得到充分探索。在这项工作中，我们对开源LLMs在多种精度下进行了系统的MBTI分析，包括主流的4位方法（GPTQ、AWQ）和极端的2位设置（AQLM变体）。除了输出级评估外，我们还通过选项级熵和置信度差距动态，检查个性如何跨层涌现，并引入不确定性放大层解码（UALD）来研究推理时解码引起的个性漂移。我们的结果揭示了一个...

    arXiv:2608.25977v1 Announce Type: new  Abstract: Personality is increasingly important in large language models (LLMs), as it shapes users' trust, engagement, and emotional experiences. While the Myers--Briggs Type Indicator (MBTI) has emerged as a common framework for assessing LLMs' personality, existing studies focus primarily on full-precision models and evaluate only final outputs. They overlook the widespread deployment of quantized LLMs requiring low memory footprints, whose personality traits remain underexplored. In this work, we present a systematic MBTI analysis of open-source LLMs across multiple precisions, including mainstream 4-bit methods (GPTQ, AWQ) and extreme 2-bit settings (AQLM variants). Beyond output-level evaluation, we examine how personality emerges across layers through option-level entropy and confidence-gap dynamics, and introduce Uncertainty-Amplified Layer Decoding (UALD) to study decoding-induced personality drift at inference time. Our results reveal a 
    
[^133]: TurnBench：面向口语对话中话轮转换动态的多领域基准测试

    TurnBench: A Multi-Domain Benchmark for Turn-Taking Dynamics in Spoken Dialogue

    [https://arxiv.org/abs/2608.25218](https://arxiv.org/abs/2608.25218)

    本文提出了TurnBench，一个多领域基准测试，结合30小时人工标注语料库和标准化评估协议，系统评估14个话轮转换系统，发现话轮结束检测稳定而打断误报率高度依赖对话类型。

    

    arXiv:2608.25218v1 公告类型：交叉 摘要：自然对话中的说话者轮流发言和倾听，实时决定何时接管、保持或让出话语权。然而，由于缺乏一致的、基于语言学理论的评估协议以及覆盖多种对话类型的人工标注数据，话轮转换的评估仍然受限。为解决这一问题，我们提出了TurnBench，一个多领域基准测试，它结合了一个30小时的人工标注双人对话语料库，以及一个标准化的评估协议，用于检测话轮结束和打断。我们将对话类型设为可控的实验变量，涵盖六种不同的互动风格，并对每个对话进行三重标注。通过对14个异构话轮转换系统进行基准测试，我们发现话轮结束的召回率在不同类型间保持稳定，而打断的误报率则强烈依赖于对话类型，并集中在反馈密集的互动风格中。尽管在平滑的语轮转移中，人类听者...

    arXiv:2608.25218v1 Announce Type: cross  Abstract: Speakers in natural conversation take turns speaking and listening, deciding in real time when to take, hold, or yield the floor. However, turn-taking evaluation remains limited due to the lack of a consistent, linguistically grounded evaluation protocol and hand-annotated data covering diverse conversation types. To address this, we present TurnBench, a multi-domain benchmark that pairs a 30-hour, hand-labeled corpus of dyadic human conversation with a standardized evaluation protocol for end-of-turn and interruption detection. We set conversation type as a controllable experimental variable, covering six distinct interaction styles, and triple-annotate each conversation. Benchmarking 14 heterogeneous turn-taking systems, we find end-of-turn recall stable across types, while interruption false positives are strongly type-dependent and concentrated in backchannel-dense interaction styles. Although in smooth floor transfers human listen
    
[^134]: CROP：通过反事实实现选择性在线策略蒸馏中的任务相关性

    CROP: Task Relevance via Counterfactuals for Selective On-Policy Distillation

    [https://arxiv.org/abs/2608.13387](https://arxiv.org/abs/2608.13387)

    CROP提出了一种基于释义校准的反事实敏感性边际方法，用于在选择性在线策略蒸馏中直接量化任务相关性，从而更有效地分配监督信号。

    

    arXiv:2608.13387v1 公告类型：新 摘要：在线策略蒸馏（OPD）在学生语言模型根据其当前策略采样的轨迹上进行监督，但对具有不同监督价值的响应标记赋予同等权重。选择性OPD通过根据估计的训练价值对响应标记进行非均匀分配监督来解决这一限制。然而，大多数现有标准主要关注优化需求，如不确定性或师生分歧，而任务相关性（即监督是否与当前输入的语义内容相关）作为补充维度仍未得到直接表征。为解决这一差距，我们引入了用于在线策略蒸馏的反事实相关性（CROP），通过释义校准的反事实敏感性边际来操作化任务相关性。对于每个源提示，CROP构建一个经过验证的原始-释义-反事实三元组，并保持学生滚动...

    arXiv:2608.13387v1 Announce Type: new  Abstract: On-policy distillation (OPD) supervises a student language model on trajectories sampled from its current policy, but assigns equal credit to response tokens with unequal supervision value. Selective OPD addresses this limitation by allocating supervision non-uniformly across response tokens according to their estimated training value. Most existing criteria, however, focus primarily on optimization need, such as uncertainty or teacher-student disagreement, while task relevance, namely whether the supervision is tied to the semantic content of the current input, remains less directly characterized as a complementary dimension. To address this gap, we introduce Counterfactual Relevance for On-Policy Distillation (CROP), which operationalizes task relevance through a paraphrase-calibrated counterfactual sensitivity margin. For each source prompt, CROP constructs a validated original-paraphrase-counterfactual triplet, holds the student roll
    
[^135]: RT-SEMamba：通过渐进式知识蒸馏实现实时语音增强的Mamba模型

    RT-SEMamba: Real-Time Speech Enhancement Mamba via Progressive Knowledge Distillation

    [https://arxiv.org/abs/2608.12099](https://arxiv.org/abs/2608.12099)

    该论文提出RT-SEMamba，一种基于因果时频Mamba块的实时语音增强模型，并通过渐进式知识蒸馏将8层教师压缩为1层学生，在保持低延迟的同时显著提升质量并实现2.75倍加速。

    

    我们提出了RT-SEMamba，一种基于因果时频Mamba块的完全因果语音增强（SE）模型。与依赖不断增长的键值缓存的Transformer架构不同，Mamba在每层传播固定大小的循环状态，从而实现内存和带宽高效的长序列推理。我们进一步引入了一种渐进式知识蒸馏（KD）策略，通过联合蒸馏复杂频谱输出和中间表示，将8层教师模型压缩为浅层1层学生模型。在Voicebank-DEMAND数据集上，8层RT-SEMamba在25毫秒算法延迟约束下实现了3.32的PESQ分数，而蒸馏后的1层学生模型相比朴素1层基线，PESQ从3.06提升至3.18，同时保持相同的稳态RTF，并相比教师模型实现了2.75倍加速。这些结果表明，结合渐进式KD的状态空间模型为实时语音增强提供了有竞争力的质量-延迟权衡。

    arXiv:2608.12099v1 Announce Type: cross  Abstract: We present RT-SEMamba, a fully causal speech enhancement (SE) model built upon causal time-frequency Mamba blocks. Unlike Transformer-based architectures that rely on a growing key-value cache, Mamba propagates a fixed-size recurrent state per layer, enabling memory- and bandwidth-efficient long-form inference. We further introduce a progressive knowledge distillation (KD) strategy that compresses an 8-layer teacher into a shallow 1-layer student by jointly distilling complex spectral outputs and intermediate representations. On Voicebank-DEMAND, the 8-layer RT-SEMamba achieves 3.32 PESQ with a 25 ms algorithmic latency constraint, and the distilled 1-layer student improves over a naive 1-layer baseline from 3.06 to 3.18 PESQ while preserving the same steady-state RTF, delivering a 2.75x speedup over the teacher. These results demonstrate that state-space models with progressive KD provide a competitive quality-latency trade-off for re
    
[^136]: LEEPS：面向大型语言模型高效RLVR的潜变量引导探索-利用提示采样方法

    LEEPS: Latent-Guided Explore-Exploit Prompt Sampling for Efficient RLVR in Large Language Models

    [https://arxiv.org/abs/2607.28077](https://arxiv.org/abs/2607.28077)

    提出LEEPS方法，通过潜变量引导的探索-利用提示采样策略，根据提示最近的非平凡比例自适应分配滚动生成预算，平衡信息丰富提示的复用与不确定提示的探索，从而提升大型语言模型RLVR训练效率。

    

    基于可验证奖励的强化学习（RLVR）能够提升大型语言模型的推理能力，但具有相同滚动生成奖励的提示组会消耗生成预算却无法提供有效的学习信号。在滚动生成前进行提示选择可以通过在生成前筛选提示来减少这种浪费。然而，现有的滚动生成前方法难以平衡利用与探索：反复利用历史上信息量大的提示会缩小训练覆盖范围，而更广泛的探索则会降低信息丰富提示的比例。为解决这些局限性，我们提出了LEEPS，一种潜变量引导的探索-利用提示采样器，能够自适应地平衡对先前观察到的信息丰富提示的复用与对不确定提示的持续探索。LEEPS将候选提示划分为利用组合和探索组合，并根据它们最近的非平凡比例自适应地分配滚动生成预算。

    arXiv:2607.28077v2 Announce Type: replace  Abstract: Reinforcement learning with verifiable rewards (RLVR) improves the reasoning capabilities of large language models, but prompt groups with identical rollout rewards consume generation budget without effective learning signals. Pre-rollout prompt selection can reduce this waste by screening prompts before rollout generation. However, existing pre-rollout methods struggle to balance exploitation and exploration: repeatedly exploiting historically informative prompts can narrow training coverage, whereas broader exploration can lower the fraction of informative prompts. To address these limitations, we introduce LEEPS, a Latent-Guided Explore--Exploit Prompt Sampler that adaptively balances the reuse of previously observed informative prompts with continued exploration of uncertain ones. LEEPS partitions candidates into exploit and explore portfolios and adaptively allocates rollout budget according to their recent non-trivial ratios. I
    
[^137]: 延迟验证使多智能体大语言模型信念失稳：失稳阈值与最优纠错节点配置

    Delayed Verification Destabilizes Multi-Agent LLM Belief: Instability Thresholds and Optimal Corrector Placement

    [https://arxiv.org/abs/2606.27409](https://arxiv.org/abs/2606.27409)

    该论文将多智能体LLM系统中的延迟验证建模为带接地节点的延迟共识问题，通过接地拉普拉斯谱分解推导出验证剂量的闭式失稳阈值（延迟为二时为黄金比例的倒数），并基于超模目标给出贪婪(1-1/e)近似的纠错节点最优配置方法。

    

    多智能体大语言模型（LLM）系统通常依赖验证者与批评者智能体来抑制幻觉，但验证往往是延迟进行的。在延迟期间，错误声明可能在智能体网络中传播。我们将这一过程建模为带有接地纠错节点的图上的延迟共识问题。通过接地拉普拉斯算子的谱分解，我们推导出了验证剂量的闭式稳定性阈值：过强或过迟的纠错会将共识转化为振荡。最不稳定的情形发生在通信延迟与验证延迟恰好重合时；当延迟为二时，该阈值为黄金比例的倒数。同一框架还给出了一个超模的配置目标函数，以及一个具有贪婪(1-1/e)近似保证的规则，用于将有限的纠错预算分配给有影响力的节点。在五个开源模型上的实验证实了预测的剂量-延迟振荡现象。相比之下，接地的事实性回答使……

    arXiv:2606.27409v2 Announce Type: replace-cross  Abstract: Multi-agent large language model (LLM) systems often rely on verifier and critic agents to suppress hallucinations, but verification is delayed. During this delay, false claims can propagate through the agent network. We model this process as delayed consensus on a graph with grounded corrector nodes. Spectral decomposition by the grounded Laplacian yields a closed-form stability threshold for the verification dose: correction that is too strong or too delayed can turn consensus into oscillation. The most unstable regime occurs when the communication and verification delays coincide; for delay two, the threshold is the inverse golden ratio. The same framework gives a supermodular placement objective and a greedy (1-1/e)-approximation rule for assigning a limited corrector budget to influential nodes. Experiments across five open models confirm the predicted dose-delay oscillations. By contrast, grounded factual answering makes 
    
[^138]: 跟随潜空间路线图：利用锚定令牌为扩散大语言模型导航可撤销解码

    Follow the Latent Roadmap: Navigating Revocable Decoding for Diffusion LLMs with Anchor Tokens

    [https://arxiv.org/abs/2606.16847](https://arxiv.org/abs/2606.16847)

    提出了一种免训练框架ASRD，通过在嵌入空间中将解码上下文解耦为基于时间一致性识别的受信任锚定令牌和不确定候选令牌，解决了扩散大语言模型可撤销解码中的错误传播与局部错误强化问题。

    

    扩散大语言模型为并行生成提供了一条有前景的途径，但面临解码速度与质量之间的权衡。虽然可撤销解码策略试图通过验证和重新掩码令牌来缓解错误，但它们通常在混合质量的上下文中运行，这导致两个关键的失败：错误传播，即新令牌从错误上下文中吸收有毒信息；以及局部错误强化，即错误之间相互强化以逃避检测。为缓解这些挑战，我们提出了ASRD（锚定监督可撤销解码），这是一个在嵌入空间中运行的免训练框架。ASRD显式地将解码上下文解耦为受信任的锚定令牌（通过时间一致性识别）和不确定的候选令牌。利用动态锚定令牌缓存，我们引入了两个互补机制：(1) 锚定监督……（摘要在此处截断）

    arXiv:2606.16847v4 Announce Type: replace-cross  Abstract: Diffusion Large Language Models (dLLMs) offer a promising avenue for parallel generation but face a trade-off between decoding speed and quality. While revocable decoding strategies attempt to mitigate errors by verifying and remasking tokens, they typically operate within a mixed-quality context. This leads to two critical failures: \textit{Error Propagation}, where new tokens absorb toxic information from erroneous context, and \textit{Local Error Reinforcement}, where errors mutually reinforce each other to evade detection. To alleviate these challenges, we propose ASRD (Anchor Supervised Revocable Decoding), a training-free framework that operates within the embedding space. ASRD explicitly decouples the decoding context into trusted \textit{Anchor Tokens}, which are identified via temporal consistency, and uncertain candidates. Leveraging a dynamic Anchor Tokens Cache, we introduce two complementary mechanisms: (1) Anchor-
    
[^139]: 当认知图遇遇上大语言模型：用于恐慌情绪唤醒预测的BDEI认知路径

    When Cognitive Graphs Meet LLMs: BDEI Cognitive Pathways for Panic Emotional Arousal Prediction

    [https://arxiv.org/abs/2606.15121](https://arxiv.org/abs/2606.15121)

    该论文主张基于评价情绪理论在自然生成方向上显式建模恐慌情绪唤醒过程，通过将认知图与大语言模型相结合构建BDEI认知路径，以预测个体和集体恐慌情绪唤醒的时间点，从而实现及时的紧急干预。

    

    在恐慌情绪唤醒显现之前预测个体和集体恐慌情绪唤醒的发生时间，对于及时的紧急干预至关重要。现有方法虽然纳入了认知要素，但没有一种方法在唤醒过程的自然生成方向上对情绪进行建模，导致唤醒时间无法确定。我们认为，将预测建立在评价情绪理论基础之上是必要的，因为该理论在其自然的生成方向上显式地对这一过程进行建模，但必须解决三个问题：(1) 评价理论认为情绪源于对多个威胁维度同时进行评估，然而此前没有研究将这些输入融合到风险感知中；(2) 现有模型在相反的、以行为为桥梁的方向上进行训练，仅将情绪作为行为的事后关联来恢复；(3) 采用大语言模型作为主要决策者的方法忽视了其输出的脆弱性和易产生幻觉的特性。

    arXiv:2606.15121v3 Announce Type: replace  Abstract: Predicting the timing of individual and collective panic emotional arousal before manifestation is essential for timely emergency intervention. Existing methods incorporate cognitive elements but none of them model emotion in the generative direction of the arousal process, leaving arousal timing undetermined. We argue that grounding prediction in appraisal emotion theory is necessary because it models this process explicitly in its natural generative direction, but three problems must be solved. (1) Appraisal theory posits that emotion arises from simultaneous evaluation across multiple threat dimensions, yet no prior work fuses these inputs into risk perception; (2) Existing models are trained in the opposite, behavior-bridged direction, recovering emotion merely as a post-hoc correlate of behavior; (3) Approaches that adopt LLMs as the primary decision-maker yet overlook the fragility and hallucination-proneness of their outputs. 
    
[^140]: Notes2Skills：从实验记录本到具备确定性感知的科学智能体技能

    Notes2Skills: From Lab Notebooks to Certainty-Aware Scientific Agent Skills

    [https://arxiv.org/abs/2606.11897](https://arxiv.org/abs/2606.11897)

    本文提出Notes2Skills框架，通过两阶段方法将非正式的实验室笔记转化为具备确定性感知的科学智能体技能，使AI智能体能够正确区分实验记录中已验证的观察、暂时性判断和可执行的实验建议，避免将不确定的科学判断误认为已确认的结论。

    

    科学发现工作流程严重依赖实验记录，研究人员在其中记录观察结果、解读不确定的实验结果并规划后续实验。与精炼完善的论文不同，实验记录保留了不断演变的科学推理、隐性科学知识以及作者的不确定性，使AI智能体能够接触到科学探索的完整过程。然而，以往关于科学文本的研究大多聚焦于论文、实验方案或结构化数据库，而非正式的实验室笔记作为AI科学智能体的输入尚未得到充分探索。这一空白之所以重要，是因为实验记录中的隐性知识并非以一套清晰指令的形式书写：经过验证的观察、暂时性的判断以及可能的实验下一步往往出现在同一段落中。如果这些信号被混淆，AI智能体可能会将不确定的科学判断误认为是已确认的结论或可执行的操作。为此，我们提出了Notes2Skills，一个用于将（实验记录转化为技能的）两阶段框架……（原文摘要在此处截断）

    arXiv:2606.11897v2 Announce Type: replace  Abstract: Scientific discovery workflows rely heavily on lab notes, where researchers record observations, interpret uncertain results, and plan follow-up experiments. Unlike polished publications, lab notes preserve evolving scientific reasoning, tacit scientific knowledge, and author uncertainty, giving AI agents access to the process of science. However, most prior work on scientific text focuses on papers, protocols, or structured databases, leaving informal laboratory notes underexplored as inputs to AI agents for science. This gap matters because tacit knowledge in lab notes is not written as a clean set of instructions: validated observations, tentative judgments, and possible experimental next steps often appear in the same passage. If these signals are conflated, an AI agent may mistake uncertain scientific judgments for confirmed conclusions or executable actions. To this end, we present Notes2Skills, a two-stage framework for turnin
    
[^141]: 多跳知识组合受预训练暴露程度的限制

    Multi-Hop Knowledge Composition is Bound by Pretraining Exposure

    [https://arxiv.org/abs/2606.09338](https://arxiv.org/abs/2606.09338)

    该研究揭示大语言模型的多跳知识组合能力本质上受预训练时对组合上下文暴露程度的限制——组合式预训练只能迁移到已暴露个体的未见问题，而永远无法惠及从未在组合上下文中出现过的个体。

    

    大语言模型在隐式多跳推理方面表现不佳：一个模型能够正确回答“$X$出生于什么时候？”和“谁是$Y$最亲密的朋友？”，但在单次前向传播中却无法回答“$Y$最亲密的朋友出生于什么时候？”——即使这两个事实都被完美记住且可被单独检索。我们在一个受控的自然语言环境中研究这种失败现象，该环境严格区分了在预训练期间接触过组合上下文的个体和从未出现在任何此类上下文中的个体。我们确认，即使单跳准确率高达97%，组合失败仍然存在，这表明这种差距是预训练层面的失败而非知识缺失。我们提出并测试了九种以数据为中心的增强格式，发现组合式预训练能够迁移到已暴露个体的未见问题上，但永远不会迁移到未出现在组合式预训练中的个体上，这表明在预训练期间对组合上下文的暴露是……

    arXiv:2606.09338v2 Announce Type: replace  Abstract: Large Language Models fail at implicit multi-hop reasoning: a model answers "When was $X$ born?" and "Who is $Y$'s closest friend?" correctly but fails on "When was $Y$'s closest friend born?" in a single forward pass, even when both facts are perfectly memorized and individually retrievable. We study this failure in a controlled natural language setting with a strict separation between individuals exposed to compositional contexts during pretraining and those that never appear in any such context. We confirm that compositional failure persists even at 97% 1-hop accuracy, establishing the gap as a pretraining failure rather than a knowledge absence. We propose and test nine data-centric augmentation formats and find that compositional pretraining transfers to unseen questions for exposed individuals, but never to individuals absent from compositional pretraining, suggesting that exposure to compositional contexts during pretraining i
    
[^142]: 从“可能”到“是”：语言模型重写中的确定性失真

    From 'May' to 'Is': Certainty Distortion in Language Model Rewriting

    [https://arxiv.org/abs/2606.07951](https://arxiv.org/abs/2606.07951)

    该研究发现语言模型在重写科学和医学文本时会系统性地改变原文的确定性程度（如把“可能”改成“是”），这种失真影响高达75%的输出且呈不对称性。

    

    人们越来越多地依赖语言模型来塑造信念和驱动决策，包括讨论、重写和总结来自科学文章、新闻和医学报告的信息。然而，在这些领域中，一个论断的表达自信程度往往至关重要，但人们对语言模型是否能忠实地保留这种自信程度却知之甚少。在本研究中，我们调查了语言模型中的确定性失真现象，其定义为在意图保留原意的转换过程中，所表达的确定性发生的有意义的变化。我们提出了一种与人群层面的确定性判断相一致的语言模型评估指标。利用该指标，我们在科学和医学交流任务的背景下，刻画了不同规模和不同系列模型的确定性失真情况。我们的结果表明，确定性失真影响着高达75%的语言模型输出，并且在重写任务中呈现出系统性的不对称性。

    arXiv:2606.07951v2 Announce Type: replace-cross  Abstract: Humans increasingly turn to Language Models (LMs) in ways that shape beliefs and drive decisions, including discussing, rewriting, and summarizing information from scientific articles, news, and medical reports. However, in these domains, where it often matters how confidently a claim is expressed, little is known about whether LMs faithfully preserve the degree of confidence. In this work, we investigate certainty distortion in LMs, defined as meaningful changes in expressed certainty during transformations intended to preserve meaning. We propose an LM-based evaluation metric that is consistent with population-level judgments of certainty. Using this metric, we characterize certainty distortion across different sizes and families of models in the context of scientific and medical communication tasks. Our results show that certainty distortion affects up to 75% of LM outputs and is systematically asymmetric in rewriting tasks 
    
[^143]: 文档解析器如何失效？审计文档智能中的结构脆弱性

    How Do Document Parsers Break? Auditing Structural Vulnerability in Document Intelligence

    [https://arxiv.org/abs/2605.19309](https://arxiv.org/abs/2605.19309)

    该论文提出轻量级输出层审计框架 ProSA，识别出文档解析器鲁棒性评估中的“足迹偏差”，并证明块级结构损失率（B-SLR）比受影响面积更能准确刻画扰动引起的结构失效及其传播路径。

    

    文档版面分析（DLA）流水线为检索增强生成、长文档问答及相关应用提供结构化的页面表示。然而，其鲁棒性评估在很大程度上仍以区域面积为中心。我们识别出这种“足迹偏差”，并提出 ProSA——一个轻量级的输出层审计框架，它将受控探测、策略驱动的目标定位与结构感知诊断进行解耦。ProSA 结合了块级结构损失率（B-SLR）、粒度感知的暴露描述符和路径归因，以分析结构同一性在何处丢失、故障在何种暴露粒度下显现，以及故障如何传播。在 MinerU 和 PP-StructureV3 上对 1,000 页文档的实验中，受影响区域与扰动引起的 OCR 不稳定性仅呈弱相关（R²=0.384/0.110），而 B-SLR 与之吻合度则高得多（R²=0.727/0.916）。暴露描述符进一步区分了以遮挡为主导和以拓扑为主导的失效路径。

    arXiv:2605.19309v4 Announce Type: replace  Abstract: Document Layout Analysis (DLA) pipelines provide structured page representations for retrieval-augmented generation, long-document question answering, and related applications. Yet their robustness evaluation remains largely area-centric. We identify this Footprint Bias and propose ProSA, a lightweight output-level auditing framework that decouples controlled probing, policy-driven targeting, and structure-aware diagnosis. ProSA combines Block-level Structural Loss Rate (B-SLR), granularity-aware exposure descriptors, and pathway attribution to analyze where structural identity is lost, at what exposure granularity failures emerge, and how failures propagate. Across MinerU and PP-StructureV3 on 1,000 pages, affected area weakly tracks perturbation-induced OCR instability ($R^2=0.384/0.110$), whereas B-SLR aligns much more closely with it ($R^2=0.727/0.916$). Exposure descriptors further separate occlusion- and topology-dominant pathw
    
[^144]: 约束解码下结构化生成中的模式键措辞作为指令通道

    Schema-Key Wording as an Instruction Channel in Structured Generation under Constrained Decoding

    [https://arxiv.org/abs/2604.14862](https://arxiv.org/abs/2604.14862)

    该论文首次系统研究了约束解码下JSON模式键措辞作为隐式指令通道的作用，揭示仅改变键的措辞即可显著影响大语言模型结构化生成的准确率，并从理论上给出了指令优势在语法投影后得以保留的充分条件。

    

    约束解码被广泛用于使大语言模型生成满足JSON等模式的结构化输出。现有工作主要将模式视为结构性约束，忽视了模式键标记也会进入自回归上下文并可能引导生成过程。据我们所知，我们首次系统研究了模式键在约束解码下作为隐式指令通道的作用。我们将结构化生成形式化为一个多通道指令问题，其中任务信号可以放置在提示词、模式键或两者之中。我们进一步提供了投影感知分析，给出了一个充分条件，在该条件下指令性键在无约束情况下的期望分数优势在语法投影后得以保留。在GSM8K和Math500上对七个语言模型的实验表明，仅改变模式键的措辞即可显著影响准确率，既有正面影响也有负面（影响）。

    arXiv:2604.14862v3 Announce Type: replace-cross  Abstract: Constrained decoding is widely used to make large language models produce structured outputs that satisfy schemas such as JSON. Existing work mainly treats schemas as structural constraints, overlooking that schema-key tokens also enter the autoregressive context and may guide generation. To the best of our knowledge, we present the first systematic study of schema keys as an implicit instruction channel under constrained decoding. We formulate structured generation as a multi-channel instruction problem, where task signals can be placed in prompts, schema keys, or both. We further provide a projection-aware analysis that gives a sufficient condition under which an unconstrained expected-score advantage of an instructional key is preserved after grammar projection. Experiments on GSM8K and Math500 across seven language models show that changing only schema-key wording can substantially affect accuracy, with both positive and ne
    
[^145]: 预测正确，步骤有误？用于鲁棒思维链合成的共识推理知识图谱

    Correct Prediction, Wrong Steps? Consensus Reasoning Knowledge Graph for Robust Chain-of-Thought Synthesis

    [https://arxiv.org/abs/2604.14121](https://arxiv.org/abs/2604.14121)

    提出CRAFT方法，通过聚合多个候选推理轨迹的共识组件构建推理知识图谱，从推理结构层面修复LLM“答案正确但推理步骤有缺陷”的问题，实现更鲁棒的思维链合成。

    

    大语言模型（LLM）在各类任务中的应用日益广泛，通常还会结合思维链（CoT）提示来提升准确性。近期研究表明，高标签预测准确率并不能保证中间推理过程的正确性，且推理缺陷的成因因样本而异，然而现有的补救措施要么只针对单一领域，要么假设某一种缺陷类型统一适用于所有样本。一种简单的缓解方法是直接给模型提供正确答案，但我们发现这并不能带来推理质量的持续改善。这表明该问题无法通过LLM对答案的感知来解决，而必须从推理的结构层面加以解决。受此启发，我们提出了CRAFT（面向缺陷感知轨迹合成的共识推理知识图谱聚合方法），该方法聚合多个候选推理轨迹间共享的共识组件……

    arXiv:2604.14121v3 Announce Type: replace  Abstract: Large language models (LLMs) have become increasingly used for various tasks, often coupled with Chain-of-Thought (CoT) prompting to boost accuracy. Recent work has shown that high label-prediction accuracy does not guarantee correct intermediate reasoning, and the causes of *reasoning flaws* vary from sample to sample, yet existing remedies either focus on a single domain or assume that one flaw type applies uniformly across samples. A simple mitigation method is to provide the model with the correct answer, but we show that this yields no consistent improvement in reasoning quality. This indicates that the problem cannot be fixed by LLMs' awareness of answers, and must instead be addressed through the *structure* of reasoning. Motivated by this, we propose CRAFT (Consensus Reasoning-knowledge-graph Aggregation for Flaw-aware Trace synthesis), which aggregates the consensus components shared across multiple candidate reasoning trace
    
[^146]: 我们还能追踪母语信号吗？探究LLM时代母语信号的韧性

    Can We Still Trace L1 Signals? Investigating the Resilience of Native Language Signals in the LLM Era

    [https://arxiv.org/abs/2604.08568](https://arxiv.org/abs/2604.08568)

    本研究通过构建覆盖神经网络前、LLM前和LLM后三个时代、八个母语群体的学术摘要母语识别数据集，发现文本中的母语信号随时间持续减弱，且英语同质化的主要趋势早在LLM出现之前就已开始。

    

    基于LLM的写作辅助工具的广泛使用引发了一个关于英语同质化的有趣问题。由于LLM倾向于将文本修改为其训练数据中所反映的主流英语规范，反映作者母语（L1）的微妙特征可能正在逐渐消失。本研究通过分析学术摘要上的母语识别（NLI）性能来研究这一现象。为此，我们构建了两个从arXiv和ACL Anthology提取的学术摘要母语识别数据集，涵盖了神经网络（NN）前时代、LLM前时代和LLM后时代三个时期的八个母语群体。然后，我们使用通过微调LLM获得的NLI分类器评估每个时代的NLI性能。结果显示NLI性能随时间持续下降。然而值得注意的是，从神经网络前时代到LLM前时代的性能下降比从LLM前时代到LLM后时代的下降更为明显。

    arXiv:2604.08568v3 Announce Type: replace-cross  Abstract: The widespread use of LLM-based writing assistance has raised an interesting question about the homogenization of English. As LLMs tend to revise texts toward mainstream English conventions reflected in their training data, the subtle fingerprints that reflect an author's native language (L1) may be gradually disappearing. This study investigates this phenomenon by analyzing native language identification (NLI) performance on academic abstracts. To this end, we construct two NLI datasets of academic abstracts extracted from arXiv and the ACL Anthology that covers eight native language groups across three time periods: pre-neural network (NN), pre-LLM, and post-LLM. We then evaluate NLI performance for each era using NLI classifiers obtained by fine-tuning LLMs. The results reveal a consistent decline in NLI performance over time. Notably, however, the decline is more pronounced from the pre-NN era to the pre-LLM era than from t
    
[^147]: 面向代码生成的编程语言分类体系

    A Taxonomy of Programming Languages for Code Generation

    [https://arxiv.org/abs/2604.00239](https://arxiv.org/abs/2604.00239)

    该论文首次提出了一个可复现的编程语言资源分类体系，将646种编程语言划分为四个层级，并揭示了代码语料库中极端且系统性的资源失衡——仅1.9%的语言占据了74.6%的token。

    

    世界上7000多种自然语言在NLP资源的可用性方面差异巨大，这促使人们根据其资源丰富程度对它们进行系统分类（Joshi et al., 2020）。编程语言（PL）之间也存在类似的差距；然而，目前尚未建立针对代码的资源分层分类体系。随着大语言模型（LLM）生成代码的能力日益增强，这样的分类体系变得不可或缺。为填补这一空白，我们提出了首个可复现的编程语言资源分类方法，将646种语言划分为四个层级。我们证明，仅1.9%的语言（第3层级，资源丰富）就占据了七个主要语料库中所有token的74.6%，而71.7%的语言（第0层级，资源稀缺）仅贡献了1.0%。对层级内部不平等程度、离散度和分布偏斜的统计分析证实，这种不平衡既是极端的也是系统性的。我们的研究结果为数据集整理和层级感知的应用提供了一个有原则的框架。

    arXiv:2604.00239v3 Announce Type: replace  Abstract: The world's 7,000+ languages vary widely in the availability of resources for NLP, motivating efforts to systematically categorize them by their degree of resourcefulness (Joshi et al., 2020). A similar disparity exists among programming languages (PLs); however, no resource-tier taxonomy has been established for code. As large language models (LLMs) grow increasingly capable of generating code, such a taxonomy becomes essential. To fill this gap, we present the first reproducible PL resource classification, grouping 646 languages into four tiers. We show that only 1.9% of languages (Tier 3, High) account for 74.6% of all tokens in seven major corpora, while 71.7% of languages (Tier 0, Scarce) contribute just 1.0%. Statistical analyses of within-tier inequality, dispersion, and distributional skew confirm that this imbalance is both extreme and systematic. Our results provide a principled framework for dataset curation and tier-aware
    
[^148]: AuthorMix：通过逐层适配器混合实现模块化的作者风格迁移

    AuthorMix: Modular Authorship Style Transfer via Layer-wise Adapter Mixing

    [https://arxiv.org/abs/2603.23069](https://arxiv.org/abs/2603.23069)

    AuthorMix提出了一种轻量级、模块化的作者风格迁移框架，通过在高资源作者上训练LoRA适配器并结合强化学习的逐层适配器混合，仅需少量目标风格样本即可快速适配新作者，其风格-含义综合得分超越包括GPT-5.1在内的所有基线方法。

    

    作者风格迁移的任务要求以目标作者的写作风格重写文本，同时保留原文的含义。现有的风格迁移方法在大规模语料库上训练单一模型，以一次性建模所有目标风格：这种高成本的方法在针对特定目标的适配方面灵活性有限，并且常常为了风格迁移而牺牲含义保留。在本文中，我们提出了AuthorMix：一个轻量级、模块化且可解释的风格迁移框架。我们首先在一小组高资源作者上训练独立的、风格特定的LoRA适配器：这使得能够通过基于强化学习的逐层适配器混合，为每个新目标快速训练专门的适配模型，且仅需少量目标风格的训练样本。AuthorMix在风格-含义综合得分上排名第一，超越了包括GPT-5.1在内的所有基线方法，并显著提升了含义保留能力。

    arXiv:2603.23069v4 Announce Type: replace-cross  Abstract: The task of authorship style transfer involves rewriting text in the style of a target author while preserving the meaning of the original text. Existing style transfer methods train a single model on large corpora to model all target styles at once: this high-cost approach offers limited flexibility for target-specific adaptation, and often sacrifices meaning preservation for style transfer. In this paper, we propose AuthorMix: a lightweight, modular, and interpretable style transfer framework. We first train individual, style-specific LoRA adapters on a small set of high-resource authors: this allows for the rapid training of specialized adaptation models for each new target using layer-wise adapter mixing via reinforcement learning, necessitating only a handful of target-style training examples. AuthorMix ranks first on the combined style-meaning score among all baselines, including GPT-5.1, and substantially improves meanin
    
[^149]: 评估跨领域映射对人类与大语言模型创造力的影响

    Assessing the Effect of Cross-Domain Mapping on Creativity in Humans and Large Language Models

    [https://arxiv.org/abs/2603.19087](https://arxiv.org/abs/2603.19087)

    跨领域随机联想能稳定提升人类的创意原创性，但对大语言模型的作用取决于其能力水平和语义距离，且两者利用灵感的方式不同——人类迁移表面特征，而大语言模型迁移结构与功能属性。

    

    创意往往源于对遥远概念的联想。随机联想能否可靠地提升原创性？它们是否以同样的方式帮助人类和大语言模型（LLM）？我们要求人类参与者和七个大语言模型通过从随机来源汲取灵感或解决未被满足的用户需求来设计产品。人类从跨领域映射中稳定获益，而大语言模型虽然产生的想法比人类更具原创性，但总体上并未从这一干预中获益，不过这一结果随语义距离而变化。在人类和大语言模型中，来源与目标之间语义距离越远的配对产生了越具原创性的想法。人类几乎在任何语义距离下都能获益，而只有评价最高的大语言模型在来源足够遥远时才能获益。人类和大语言模型对灵感来源的利用方式也不同：人类倾向于迁移表面特征，而大语言模型迁移的是结构和功能属性。这些发现揭示了……

    arXiv:2603.19087v3 Announce Type: replace  Abstract: Creative ideas often arise by associating remote concepts. Can random associations reliably increase originality, and do they help humans and large language models (LLMs) in the same way? We asked human participants and seven LLMs to design products by drawing inspiration from a random source or addressing an unmet user need. Humans reliably benefited from cross-domain mappings, while LLMs generated more original ideas than humans but showed no overall benefit from the intervention, though this changed with semantic distance. More distant source-target pairings produced more original ideas in both humans and LLMs. Humans benefited at nearly any distance, while only the highest-rated LLMs benefited when the source was sufficiently remote. Humans and LLMs also used sources differently: humans tended to transfer surface features, while LLMs transferred structural and functional properties. These findings reveal the generative role of re
    
[^150]: CzechTopic：面向历史捷克语文档零样本主题定位的基准测试

    CzechTopic: A Benchmark for Zero-Shot Topic Localization in Historical Czech Documents

    [https://arxiv.org/abs/2603.03884](https://arxiv.org/abs/2603.03884)

    该论文推出了基于捷克历史文档的人工标注零样本主题定位基准CzechTopic，支持文档级与词级评估，发现最强LLM接近人类一致性水平，而小规模的蒸馏BERT模型仍具竞争力。

    

    主题定位旨在识别表达给定主题（由名称和描述定义）的文本片段。为了研究这一任务，我们基于捷克历史文档引入了一个由人工标注的基准数据集，其中包含人工定义的主题以及手动标注的文本片段，并支持文档级和词级两个层面的评估。评估是相对于人类标注一致性而非单一参考标注进行的。我们评估了多种大型语言模型，以及在蒸馏开发数据集上微调的基于BERT的模型。结果显示，大语言模型之间存在显著差异，其性能从接近人类的主题检测水平到在片段定位方面的明显失败不等。尽管最强的模型接近人类标注一致性，但经过蒸馏的词元嵌入模型虽然规模较小，仍然保持竞争力。该数据集和评估框架已在 https://github.com/dcg 公开提供。

    arXiv:2603.03884v2 Announce Type: replace-cross  Abstract: Topic localization aims to identify spans of text that express a given topic defined by a name and description. To study this task, we introduce a human-annotated benchmark based on Czech historical documents, containing human-defined topics together with manually annotated spans and supporting evaluation at both document and word levels. Evaluation is performed relative to human agreement rather than a single reference annotation. We evaluate a diverse range of large language models alongside BERT-based models fine-tuned on a distilled development dataset. Results reveal substantial variability among LLMs, with performance ranging from near-human topic detection to pronounced failures in span localization. While the strongest models approach human agreement, the distilled token embedding models remain competitive despite their smaller scale. The dataset and evaluation framework are publicly available at: https://github.com/dcg
    
[^151]: 注意风格：沟通风格对人机对话交互的影响

    Mind the Style: Impact of Communication Style on Human-Chatbot Interaction

    [https://arxiv.org/abs/2602.17850](https://arxiv.org/abs/2602.17850)

    本研究发现，友好型沟通风格的聊天机器人在提升用户满意度和任务成功率方面优于直接型风格，但无聊天机器人的控制条件在任务成功率上表现最佳。

    

    arXiv:2602.17850v2 公告类型：交叉替换 摘要：对话代理日益介导日常数字交互，但其沟通风格对用户体验和任务成功的影响仍未得到充分理解。针对这一空白，我们报告了一项受试者间用户研究，参与者与名为NAVI的聊天机器人的两个版本之一进行交互，该机器人协助他们完成基于交互式地图的2D导航任务。两个聊天机器人版本主要在设计上差异于沟通风格：一个使用友好和支持性的语气，而另一个使用直接和任务导向的语气。我们还包含了一个控制条件，其中参与者不与聊天机器人交互，但接收逐步导航指令。友好型聊天机器人显著提高了用户的沟通满意度，并与直接型聊天机器人相比，与更高的任务成功率相关。然而，控制条件下的参与者取得了最高的任务成功率。

    arXiv:2602.17850v2 Announce Type: replace-cross  Abstract: Conversational agents increasingly mediate everyday digital interactions, yet the effects of their communication style on user experience and task success remain insufficiently understood. Addressing this gap, we report a between-subject user study in which participants interacted with one of two versions of a chatbot called NAVI, which assisted them in an interactive map-based 2D navigation task. The two chatbot versions were designed to differ primarily in communication style: one used a friendly and supportive tone, while the other used a direct and task-focused tone. We also included a control condition where participants did not interact with a chatbot but received the step-by-step navigation instructions. The friendly chatbot significantly increased users' communication satisfaction and was associated with higher task success than the direct chatbot. However, participants in the control condition achieved the highest task
    
[^152]: 理解大语言模型的失败：用多带图灵机分析语言模型推理中的系统性错误

    Understanding LLM Failures: A Multi-Tape Turing Machine Analysis of Systematic Errors in Language Model Reasoning

    [https://arxiv.org/abs/2602.15868](https://arxiv.org/abs/2602.15868)

    该论文提出用确定性多带图灵机形式化大语言模型的完整交互流程，从而精确定位各类失败模式的发生阶段，并解释了思维链提示为何有效及其根本局限。

    

    大语言模型（LLMs）在一些看似简单的任务上会表现出失败模式。我们提出了一种基于确定性多带图灵机的LLM交互形式化方法，其中每条带代表一个不同的组件：输入字符、词元、词表、模型参数、激活值、概率分布和输出文本。该模型能够将失败模式精确定位到特定的流水线阶段，例如揭示了词元化如何遮蔽计数任务所需的字符级结构。该模型阐明了为什么思维链提示等技术有效——通过将计算外化到输出带上——同时也揭示了这些技术的根本局限性。这种方法为几何隐喻提供了一种严格且可证伪的替代方案，并以有原则的误差分析补充了经验性的缩放定律。

    arXiv:2602.15868v3 Announce Type: replace  Abstract: Large language models (LLMs) exhibit failure modes on seemingly trivial tasks. We propose a formalisation of LLM interaction using a deterministic multi-tape Turing machine, where each tape represents a distinct component: input characters, tokens, vocabulary, model parameters, activations, probability distributions, and output text. The model enables precise localisation of failure modes to specific pipeline stages, revealing, e.g., how tokenisation obscures character-level structure needed for counting tasks. The model clarifies why techniques like chain-of-thought prompting help, by externalising computation on the output tape, while also revealing their fundamental limitations. This approach provides a rigorous, falsifiable alternative to geometric metaphors and complements empirical scaling laws with principled error analysis.
    
[^153]: 修补分布失配：面向稳定离策略SFT的强化学习改写智能体

    Patch the Distribution Mismatch: RL Rewriting Agent for Stable Off-Policy SFT

    [https://arxiv.org/abs/2602.11220](https://arxiv.org/abs/2602.11220)

    提出一种用强化学习训练的轻量级LoRA改写策略，在任务一致性约束下优化问答分布对齐与语义多样性，从而修补下游监督数据与模型生成分布之间的失配，缓解SFT中的灾难性遗忘。

    

    大语言模型通常通过监督微调（SFT）来适配下游任务，但下游监督数据与模型自身生成分布之间的显著分布失配可能加剧灾难性遗忘。数据改写提供了一种以数据为中心的方法，可在SFT之前缩小这种失配。然而，现有方法通常从提示诱导的条件分布中采样改写结果，这未必与骨干模型自然的问答生成分布对齐，且固定模板会降低输出多样性。我们将数据改写形式化为一个策略学习问题，并利用强化学习训练一个轻量级的LoRA改写策略。该策略在严格的任务一致性门控约束下，优化问答风格的分布对齐与语义多样性，为下游SFT生成经过验证的监督数据。在三个指令微调骨干模型上，所得模型

    arXiv:2602.11220v2 Announce Type: replace-cross  Abstract: Large language models are commonly adapted to downstream tasks through supervised fine-tuning (SFT), but substantial distribution mismatch between downstream supervision and a model's generation distribution can intensify catastrophic forgetting. Data rewriting offers a data-centric way to narrow this mismatch before SFT. Existing methods, however, typically sample rewrites from a prompt-induced conditional distribution, which need not align with the backbone's natural question-answering generation distribution, and fixed templates can reduce output diversity. We formulate data rewriting as a policy-learning problem and train a lightweight LoRA rewriting policy with reinforcement learning. The policy optimizes question-answering-style distributional alignment and semantic diversity under a hard task-consistency gate, producing verified supervision for downstream SFT. Across three instruction-tuned backbones, the resulting model
    
[^154]: HALT：基于对数概率时间序列的幻觉评估

    HALT: Hallucination Assessment via Log-probs as Time series

    [https://arxiv.org/abs/2602.02888](https://arxiv.org/abs/2602.02888)

    该论文提出HALT，一种仅利用LLM生成的前20个token对数概率作为时间序列、结合GRU模型与熵特征进行幻觉检测的轻量级方法，无需访问模型内部状态即可实现强泛化能力，并配套发布了统一的幻觉检测基准HUB。

    

    幻觉仍然是大语言模型（LLM）面临的主要障碍，尤其是在安全关键领域。我们提出了HALT（Hallucination Assessment via Log-probs as Time series，基于对数概率时间序列的幻觉评估），这是一种轻量级的幻觉检测器，仅利用LLM生成内容的前20个token的对数概率作为时间序列。HALT使用门控循环单元（GRU）模型结合基于熵的特征来学习模型校准偏差，为大型编码器提供了一种极其高效的替代方案。与白盒方法不同，HALT不需要访问隐藏状态或注意力图，仅依赖输出的对数概率。与黑盒方法不同，它处理的是对数概率而非表层文本，这使其具有更强的领域泛化能力，并且能够在不访问内部权重的情况下与专有LLM兼容。为了进行性能基准测试，我们引入了HUB（Hallucination detection Unified Benchmark，幻觉检测统一基准），该基准整合了……（摘要原文在此处被截断）

    arXiv:2602.02888v2 Announce Type: replace-cross  Abstract: Hallucinations remain a major obstacle for large language models (LLMs), especially in safety-critical domains. We present HALT (Hallucination Assessment via Log-probs as Time series), a lightweight hallucination detector that leverages only the top-20 token log-probabilities from LLM generations as a time series. HALT uses a gated recurrent unit model combined with entropy-based features to learn model calibration bias, providing an extremely efficient alternative to large encoders. Unlike white-box approaches, HALT does not require access to hidden states or attention maps, relying only on output log-probabilities. Unlike black-box approaches, it operates on log-probs rather than surface-form text, which enables stronger domain generalization and compatibility with proprietary LLMs without requiring access to internal weights. To benchmark performance, we introduce HUB (Hallucination detection Unified Benchmark), which consol
    
[^155]: ProofVerifier：一个可扩展的、多样性驱动的自然语言证明验证框架

    ProofVerifier: A Scalable, Diversity-Driven Framework for Natural-Language Proof Verification

    [https://arxiv.org/abs/2602.02377](https://arxiv.org/abs/2602.02377)

    该论文提出ProofVerifier框架，通过LLM辅助的数据管道大规模生成多样化的问题-证明-检查样本，并结合多模型一致性与分层人工审计获得准确标签，从而训练出可靠的自然语言数学证明验证器。

    

    虽然大型语言模型（LLM）在具有可验证答案的数学问题上已取得强劲表现，但许多高难度问题都是基于证明的，需要对完整证明进行评估。然而，训练此类验证器需要大规模的多样化且可信的问题-证明-检查（QPC）样本，而这些样本十分稀缺。为应对这一挑战，我们开发了一个经人工审计、由LLM辅助的数据管道，能够以有限的人力产出大规模的QPC三元组。通过系统地变换问题来源、生成策略和生成模型，该管道创建了跨越多个难度级别、语言风格和错误类型的多样化问题-证明对。我们结合多LLM一致性判断与分层人工审计，以获得准确的证明正确性标签。利用这些数据，我们训练了生成式证明验证器，并引入辅助的流畅性过滤器以及平衡的令牌加权，以稳定二元奖励训练……

    arXiv:2602.02377v3 Announce Type: replace  Abstract: While large language models (LLMs) have achieved strong performance on mathematical problems with verifiable answers, many advanced problems are proof-based and require evaluating full proofs. However, training such verifiers requires diverse and trustworthy question-proof-check (QPC) examples at scale, which are scarce. To address this challenge, we develop a human-audited, LLM-assisted data pipeline that produces large-scale QPC triplets with limited human effort. By systematically varying problem sources, generation strategies, and generator models, the pipeline creates diverse problem-proof pairs spanning multiple difficulty levels, linguistic styles, and error types. We combine multi-LLM agreement with hierarchical human auditing to obtain accurate proof-correctness labels. Using these data, we train generative proof verifiers and introduce an auxiliary fluency filter together with balanced token weighting to stabilize binary-re
    
[^156]: 增强智能辅导系统中新题目冷启动场景下知识追踪的鲁棒性

    Enhancing knowledge tracing robustness for new question cold start in Intelligent Tutoring Systems

    [https://arxiv.org/abs/2512.07179](https://arxiv.org/abs/2512.07179)

    本研究设计了集成多种特征的PICKT知识追踪模型，并实证分析了题目难度、文本以及知识图谱关系信息等特征在提升新题目冷启动场景下知识追踪模型鲁棒性方面的作用。

    

    智能辅导系统（ITS）通过诊断学习者的熟练程度来提供个性化的学习路径。知识追踪（KT）模型通过估计学习者不断变化的知识状态，在这一诊断过程中发挥着核心作用。然而，在现实世界的智能辅导系统服务中，当新引入的题目没有先前的交互历史时，诊断的可靠性可能会下降。本研究旨在通过实证方法确定在题目冷启动情况下支持知识追踪模型鲁棒性的关键特征。为此，我们设计了实用集成交叉一致性知识追踪模型（PICKT），该模型整合了多种类型的特征，并检验了哪些特征在题目冷启动情况下有助于提升鲁棒性。在这些特征中，我们进一步分析了先前知识追踪研究中已被确认的重要因素——难度、文本以及从知识图谱中提取的关系信息，以考察它们各自如何发挥作用。

    arXiv:2512.07179v2 Announce Type: replace  Abstract: Intelligent Tutoring Systems (ITS) provide personalized learning paths by diagnosing learners' proficiency. Knowledge Tracing (KT) models play a central role in this diagnosis by estimating learners' evolving knowledge states. However, in real-world ITS services, diagnostic reliability may decrease when newly introduced questions have no prior interaction history. This study aims to empirically identify the key features that support KT model robustness under the question cold start situation. To this end, we designed Practical Integrated Cross-consistent Knowledge Tracing (PICKT), which integrates multiple types of features, and examined which feature contributes to robustness under the question cold start situation. Among these features, we further analyzed difficulty, texts, and relational information derived from the knowledge map, which have been identified as important factors in prior KT research, to examine how each contribute
    
[^157]: 看穿MiRAGE：多模态检索增强生成的评估

    Seeing Through the MiRAGE: Evaluating Multimodal Retrieval Augmented Generation

    [https://arxiv.org/abs/2510.24870](https://arxiv.org/abs/2510.24870)

    MiRAGE是一个以论断为中心的多模态检索增强生成评估框架，通过InfoF1和CiteF1指标评估事实性、信息覆盖度与引用完整性，在文本任务上优于现有RAG评估指标，且是唯一能够推广到多模态来源的方法。

    

    我们提出了MiRAGE，一个针对多模态来源检索增强生成（RAG）的评估框架。随着视听媒体日益成为网络上普遍的信息来源，RAG系统必须将此类媒体整合到生成过程中。然而，现有的RAG评估方法主要以文本为中心，难以直接迁移到多模态场景。MiRAGE是一种以论断为中心的多模态RAG评估方法，由两部分组成：InfoF1用于评估事实性与信息覆盖度，CiteF1用于评估引用的支持性与完整性。研究表明，当由人类使用时，MiRAGE与输出质量的外部判断高度一致。此外，我们还介绍了MiRAGE的自动实现版本，并将其与三个著名的以文本为中心的RAG指标的多模态变体——ALCE、ARGUE和RAGAS——进行比较，发现MiRAGE在文本任务上优于这三个指标，并且是唯一能够推广到多模态来源的评估方法。

    arXiv:2510.24870v3 Announce Type: replace  Abstract: We introduce MiRAGE, an evaluation framework for retrieval-augmented generation (RAG) from multimodal sources. As audiovisual media becomes a more prevalent source of information online, RAG systems must integrate such media into generation. Yet, existing evaluation methods for RAG are largely text-centric and do not readily transfer to multimodal settings. MiRAGE is a claim-centric approach to multimodal RAG evaluation, consisting of InfoF1, which assesses factuality and information coverage, and CiteF1, which assesses citation support and completeness. We show that, when applied by humans, MiRAGE strongly aligns with extrinsic judgments of output quality. We additionally introduce an automatic implementation of MiRAGE and compare it to multimodal variants of three prominent text-centric RAG metrics---ALCE, ARGUE, and RAGAS---finding that MiRAGE outperforms all three on text while being the only one to generalize to multimodal sourc
    
[^158]: AgentPack：一个由智能体与人类共同编写的代码变更数据集

    AgentPack: A Dataset of Code Changes, Co-Authored by Agents and Humans

    [https://arxiv.org/abs/2509.21891](https://arxiv.org/abs/2509.21891)

    提出 AgentPack——一个包含 180 万条由人类与 AI 智能体（如 Claude Code）共同完成的代码编辑的数据集，相比传统从提交记录中挖掘的数据，其意图描述更明确、质量更可靠。

    

    针对代码编辑任务对大型语言模型进行微调，通常依赖于挖掘提交记录和拉取请求。其工作假设是：提交信息以自然语言描述人类意图，而代码补丁描述实现该意图的更改。然而，以往收集的数据中有很大一部分是嘈杂的：提交信息过于简短，人类编写的提交混杂了多个不相关的编辑，而且许多提交来自简单的基于规则的机器人。软件工程智能体的近期普及改变了这一格局。由人类和智能体共同编写的代码变更，通常伴随着更加明确的关于意图和理由的自然语言描述。此外，当这些变更进入公开仓库时，它们已被人类隐式过滤：项目维护者会剔除低质量的提交。我们提出了 AgentPack，这是一个包含 180 万条由 Claude Code 等智能体与人类共同完成的代码编辑的语料库……

    arXiv:2509.21891v3 Announce Type: replace-cross  Abstract: Fine-tuning large language models for code editing has typically relied on mining commits and pull requests. The working hypothesis has been that commit messages describe human intent in natural language, and patches to code describe the changes that implement that intent. However, much of the previously collected data is noisy: commit messages are terse, human-written commits commingle several unrelated edits, and many commits come from simple, rule-based bots.   The recent adoption of software engineering agents changes this landscape. Code changes \emph{co-authored} by humans and agents are often accompanied by substantially more explicit natural-language descriptions of intent and rationale. Moreover, when these changes land in public repositories, they are implicitly filtered by humans: maintainers discard low-quality commits to their projects.   We present AgentPack, a corpus of 1.8M code edits co-authored by Claude Code,
    
[^159]: 建模形容词修饰对语义合理性的影响

    Modelling Adjectival Modification Effects on Semantic Plausibility

    [https://arxiv.org/abs/2507.21828](https://arxiv.org/abs/2507.21828)

    本文针对形容词修饰如何改变语义合理性的Adept基准，提出了一种基于句子Transformer的概念新颖的建模方法，并发现句子Transformer尽管在概念上契合该任务，其表现却不及RoBERTa等Transformer模型。

    

    尽管评估诸如“新闻是相关的”这类事件合理性的任务已得到越来越多研究的关注，但捕捉由事件修饰所引发的合理性变化却较少受到重视。理解合理性的变化对于对话生成、常识推理和幻觉检测等任务具有重要意义，因为它能够正确建模例如“虚假新闻是相关的”这类表述——这类表述相关性较低，但由于潜在的虚假信息而更需要引起关注。在本工作中，我们攻克了Adept挑战基准（Emami等人，2021），该基准由16K对英语句子对组成，每对句子仅在恰好一个形容词修饰语上存在差异（例如“虚假的”）。我们的建模实验提供了一种使用句子Transformer的概念上新颖的方法，并揭示了句子Transformer尽管在概念上与该任务相契合，却仍然表现挣扎，其性能不及RoBERTa等Transformer模型。

    arXiv:2507.21828v2 Announce Type: replace  Abstract: While the task of assessing the plausibility of events such as "news is relevant" has been addressed by a growing body of work, less attention has been paid to capturing changes in plausibility as triggered by event modification. Understanding changes in plausibility is relevant for tasks such as dialogue generation, commonsense reasoning, and hallucination detection, as it allows to correctly model, for example, "false news is relevant", which is of lower relevance but higher concern due to potential disinformation. In this work, we tackle the Adept challenge benchmark (Emami et al. 2021) consisting of 16K English sentence pairs differing by exactly one adjectival modifier (e.g., false.) Our modeling experiments provide a conceptually novel method using sentence transformers and reveal that sentence transformers struggle despite their conceptual alignment with the task at hand, underperforming in comparison to transformers like RoBE
    
[^160]: 捐赠还是创作？比较情感标注多模态社交媒体帖子的数据收集策略

    Donate or Create? Comparing Data Collection Strategies for Emotion-labeled Multimodal Social Media Posts

    [https://arxiv.org/abs/2505.24427](https://arxiv.org/abs/2505.24427)

    本研究比较了“捐赠真实帖子”与“研究创作帖子”两种情感标注数据收集策略，发现研究创作的内容更长、更依赖文本而非图像表达情感、更聚焦典型情感事件，揭示了不同数据收集方式会导致数据特性产生显著差异。

    

    对情感表达等主观现象的准确建模需要带有作者意图标注的数据。通常，此类数据通过要求研究参与者捐赠并标注在现实世界中产生的真实内容来收集，或者要求参与者在研究期间创作符合特定标签的内容。要求参与者创作内容通常比数据捐赠更易于实施，且对参与者隐私的风险更小。然而，研究创作的内容是否以及如何与真实内容存在差异，以及这些差异如何影响模型，目前尚不清楚。我们收集了带有情感标签的研究创作和真实多模态社交媒体帖子，并在多个维度上对它们进行比较，包括模型性能。我们发现，与真实帖子相比，研究创作的帖子更长，在情感表达上更依赖文本而较少依赖图像，并且更多聚焦于情感典型事件。参与者的样本……

    arXiv:2505.24427v2 Announce Type: replace  Abstract: Accurate modeling of subjective phenomena such as emotion expression requires data annotated with authors' intentions. Commonly such data is collected by asking study participants to donate and label genuine content produced in the real world, or create content fitting particular labels during the study. Asking participants to create content is often simpler to implement and presents fewer risks to participant privacy than data donation. However, it is unclear if and how study-created content may differ from genuine content, and how differences may impact models. We collect study-created and genuine multimodal social media posts labeled for emotion and compare them on several dimensions, including model performance. We find that compared to genuine posts, study-created posts are longer, rely more on their text and less on their images for emotion expression, and focus more on emotion-prototypical events. The samples of participants w
    
[^161]: PBEBench：一个受历史语言学启发的多步骤示例编程推理基准测试

    PBEBench: A Multi-Step Programming by Examples Reasoning Benchmark inspired by Historical Linguistics

    [https://arxiv.org/abs/2505.23126](https://arxiv.org/abs/2505.23126)

    该论文提出了PBEBench，一个受历史语言学正向重构任务启发的多步骤示例编程基准测试，用于评估大语言模型的归纳推理能力，并配备可自动生成可控难度问题、避免数据污染的自动化流水线。

    

    尽管许多基准测试评估大型语言模型（LLM）在数学、编程或数据处理等领域的推理能力，但很少有基准测试能够脱离特定领域，将推理本身作为一种独立能力来检验。我们提出了一种新型基准测试，用于评估LLM的归纳推理能力，其灵感来自历史语言学中的正向重构任务，但以一种极其简单、通用的方式（即示例编程的形式）来表述。该任务涉及生成一系列级联的简单字符串重写程序，将给定的输入字符串列表转换为期望的输出字符串列表。我们提出了一个完全自动化的流水线，可以程序化地生成此类问题并控制其难度，从而实现对推理模型的可扩展评估，同时避免数据污染。使用这种方法，我们构建了两个基准测试：PBEBench-Lite，它……

    arXiv:2505.23126v5 Announce Type: replace  Abstract: Although many benchmarks evaluate the reasoning abilities of Large Language Models (LLMs) within domains such as mathematics, coding, or data wrangling, few abstract away from domain specifics to examine reasoning as a capability in and of itself. We contribute a novel type of benchmark evaluating the inductive reasoning capabilities of LLMs that is inspired by the forward reconstruction task from historical linguistics but is formulated in an extremely simple, general way (in the form of Programming by Examples). The task involves generating a cascade of simple string rewrite programs to transform a given list of input strings into a list of desired output strings. We present a fully automated pipeline that programmatically generates problems of this type with controllable difficulty, enabling scalable evaluation of reasoning models while avoiding contamination. Using this approach, we construct two benchmarks: PBEBench-Lite, which 
    
[^162]: 从大语言模型中提取概率知识用于贝叶斯网络参数化

    Extracting Probabilistic Knowledge from Large Language Models for Bayesian Network Parameterization

    [https://arxiv.org/abs/2505.15918](https://arxiv.org/abs/2505.15918)

    本研究证明大语言模型可以有效提取概率知识用于贝叶斯网络参数化，其在八十个不同领域网络上的条件概率估计结果显著优于随机分布等基线方法。

    

    在这项工作中，我们评估了大语言模型（LLMs）在构建贝叶斯网络（BNs）方面通过近似领域专家先验知识的潜力。大语言模型已被证明具有作为事实知识库的潜力；然而，它们生成关于现实世界事件的概率知识的能力仍然研究不足。我们探索利用大语言模型中固有的概率知识，来推导关于事件及其在贝叶斯网络中相互关系的陈述的概率估计。在这种场景下使用大语言模型可以实现贝叶斯网络的参数化，从而使特定领域内的概率建模成为可能。我们在八十个公开可用的贝叶斯网络（涵盖从医疗保健到金融等多个领域）上进行的实验表明，通过查询大语言模型关于事件的条件概率，与包括随机分布、均匀分布以及基于下一个词元生成概率的方法等基线相比，能够获得有意义的结果。

    arXiv:2505.15918v3 Announce Type: replace-cross  Abstract: In this work, we evaluate the potential of Large Language Models (LLMs) in building Bayesian Networks (BNs) by approximating domain expert priors. LLMs have demonstrated potential as factual knowledge bases; however, their capability to generate probabilistic knowledge about real-world events remains understudied. We explore utilizing the probabilistic knowledge inherent in LLMs to derive probability estimates for statements regarding events and their relationships within a BN. Using LLMs in this context allows for the parameterization of BNs, enabling probabilistic modeling within specific domains. Our experiments on eighty publicly available Bayesian Networks, from healthcare to finance, demonstrate that querying LLMs about the conditional probabilities of events provides meaningful results when compared to baselines, including random and uniform distributions, as well as approaches based on next-token generation probabilitie
    
[^163]: 从多种声音中学习：使用多参考人工与合成数据的文学机器翻译

    Learning from Many Voices: Literary MT Using Multi-Reference Human and Synthetic Data

    [https://arxiv.org/abs/2412.18707](https://arxiv.org/abs/2412.18707)

    提出基于语义相似度的过滤框架来利用文学多参考数据集改进文学机器翻译，并通过自动指标和人工评估证明人工专家译文的微调效果优于大语言模型生成的合成数据。

    

    同一文学作品的多个有效译本自然存在。我们研究了利用这些多参考数据集来改进文学机器翻译的策略。我们提出了一个基于语义相似度的过滤框架，用于识别那些参考译文在保持忠实的同时呈现出有意义变化的源文本。我们发现，使用中等到高语义相似度的数据进行微调显著优于使用低语义相似度的数据。此外，使用中高语义相似度数据可以获得与使用全部未过滤数据相当甚至更好的性能。由大语言模型生成的合成译文是人类专家译文的经济便捷替代品；然而，我们发现基于人工专家译文的微调在自动指标和人工评估中均优于基于合成增强数据的微调，这证明了人工专家译文不可或缺的价值。

    arXiv:2412.18707v3 Announce Type: replace  Abstract: Multiple valid translations of a single literary work naturally exist. We investigate strategies for leveraging these multi-reference datasets to improve literary machine translation. We propose a filtering framework based on semantic similarity to identify source texts whose references display meaningful variation while remaining faithful. We find that fine-tuning with medium to high semantic similarity data substantially outperforms low semantic similarity data. Moreover, using medium and high semantic similarity data achieves comparable or better performance than using the full unfiltered data.   Synthetic translations generated by LLMs are economical and convenient alternatives to human expert translations; however, we find fine-tuning on human expert translations outperforms fine-tuning on synthetically augmented data in automatic metrics and human evaluations, demonstrating the indispensable value of human expert translations f
    
[^164]: 分而治之：一种混合策略击败多模态大语言模型

    Divide and Conquer: A Hybrid Strategy Defeats Multimodal Large Language Models

    [https://arxiv.org/abs/2412.16555](https://arxiv.org/abs/2412.16555)

    本文提出了一种名为JMLLM的多模态越狱攻击方法，通过整合多种混合策略在文本、视觉和听觉三种模态上对大语言模型进行全面越狱攻击，克服了现有方法查询次数过多、模态覆盖有限和攻击成功率低等局限。

    

    大语言模型（LLMs）凭借其强大的推理、理解和生成能力，被广泛应用于社会各个领域。然而，与这些模型相关的安全问题正变得日益严峻。越狱攻击作为检测大语言模型漏洞的重要手段，研究人员尝试通过各种攻击方法诱导这些模型生成有害内容。尽管如此，现有的越狱方法面临诸多局限性，例如查询次数过多、越狱模态覆盖范围有限、攻击成功率低以及评估方法过于简单。为了克服这些限制，本文提出了一种多模态越狱攻击方法：JMLLM。该方法整合了多种策略，在文本、视觉和听觉模态上执行全面的越狱攻击。此外，我们还贡献了一个新的综合性数据集。

    arXiv:2412.16555v4 Announce Type: replace  Abstract: Large language models (LLMs) are widely applied in various fields of society due to their powerful reasoning, understanding, and generation capabilities. However, the security issues associated with these models are becoming increasingly severe. Jailbreaking attacks, as an important method for detecting vulnerabilities in LLMs, have been explored by researchers who attempt to induce these models to generate harmful content through various attack methods. Nevertheless, existing jailbreaking methods face numerous limitations, such as excessive query counts, limited coverage of jailbreak modalities, low attack success rates, and simplistic evaluation methods. To overcome these constraints, this paper proposes a multimodal jailbreaking method: JMLLM. This method integrates multiple strategies to perform comprehensive jailbreak attacks across text, visual, and auditory modalities. Additionally, we contribute a new and comprehensive datase
    
[^165]: 自然语言生成中标签-置信度感知的不确定性估计

    Label-Confidence-Aware Uncertainty Estimation in Natural Language Generation

    [https://arxiv.org/abs/2412.07255](https://arxiv.org/abs/2412.07255)

    提出了一种基于逐点KL散度的标签-置信度感知不确定性量化方法（LCA-UQ），弥合了多样本全局熵与候选答案局部置信度之间的差距，从而更准确地评估大语言模型生成回答的有效性，缓解幻觉问题。

    

    大语言模型（LLMs）在生成式任务中展现出卓越的能力，但由于其倾向于生成幻觉式响应而带来潜在风险。因此，旨在区分答案有效性的不确定性量化（UQ）对于确保人工智能系统的安全性和稳健性至关重要。然而，现有方法主要依赖于测量多个随机样本的熵来表示不确定性，往往忽略了与被评估候选答案相关的特定不确定性信息。这种疏忽可能导致有偏差的分类结果。在本文中，我们研究了多个样本的全局熵与候选答案的局部置信度之间的差异，并提出了一种基于逐点Kullback-Leibler（PKL）散度的标签-置信度感知不确定性量化（LCA-UQ）方法。我们的方法有效弥合了……

    arXiv:2412.07255v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) demonstrate remarkable capabilities in generative tasks but pose potential risks due to their tendency to generate hallucinatory responses. Therefore, Uncertainty Quantification (UQ), which aims to distinguish the validity of answers, is crucial for ensuring the safety and robustness of AI systems. However, existing methods primarily rely on measuring the entropy of multiple stochastic samples to represent uncertainty, often overlooking the specific uncertainty information associated with the candidate answer under evaluation. This oversight can lead to biased classification outcomes. In this paper, we investigate the discrepancy between global entropy from multiple samples and local confidence of candidate answer, and propose a Label-Confidence-Aware Uncertainty Quantification (LCA-UQ) method based on Pointwise Kullback-Leibler (PKL) divergence. Our method effectively bridges the gap between the co
    
[^166]: 在SoC智能网卡上通过性能预测加速有状态网络应用

    Accelerating Stateful Network Applications with Performance Prediction on SoC SmartNICs

    [https://arxiv.org/abs/2410.22229](https://arxiv.org/abs/2410.22229)

    Vela框架通过以状态为中心的分析模型驱动的编译时/运行时协同设计，利用预测性编译器自动估算资源分配的吞吐量上限并动态适应流量变化，从而加速有状态网络应用在SoC智能网卡上的卸载。

    

    将有状态网络功能卸载到多线程SoC智能网卡上有望带来显著的性能和成本效益。然而，实现这一潜力面临两个根本性挑战。首先，在没有性能指导的情况下，开发人员被迫陷入缓慢的手动试错循环，通过反复部署和测试来寻找可行的资源分配方案。其次，在不断变化的流量条件下维持性能，需要在编译时就固定下来的内存布局内进行状态驻留调整和过载处理。本文提出了Vela框架，通过模型驱动的编译时/运行时协同设计来解决这两个挑战。其核心是一个预测性编译器，它用快速、自动化的分析取代了手动调优循环，使用一种新颖的、以状态为中心的分析模型来估算任意给定资源分配方案的吞吐量上限。此外，该框架还辅以一个轻量级运行时，能够动态管理（摘要内容在此处截断）

    arXiv:2410.22229v2 Announce Type: replace-cross  Abstract: Offloading stateful network functions to multi-threaded SoC SmartNICs promises significant performance and cost benefits. However, realizing this potential is hindered by two fundamental challenges. First, without performance guidance, developers are forced into a slow, manual trial-and-error cycle of deploying and testing to find a feasible resource allocation. Second, sustaining performance under changing traffic requires adapting state residency and handling overload within memory layouts fixed at compile time. This paper introduces Vela, a framework that addresses both challenges through a model-driven, compile-time/runtime co-design. Its core is a predictive compiler that replaces the manual tuning loop with fast, automated analysis, using a novel, state-centric analytical model to estimate the throughput ceiling of any given resource allocation plan. This is complemented by a lightweight runtime that dynamically manages c
    
[^167]: 大语言模型在标注时默认采用哪种人口统计学特征？

    Which Demographics do LLMs Default to During Annotation?

    [https://arxiv.org/abs/2410.08820](https://arxiv.org/abs/2410.08820)

    该研究结合LLM偏差研究与人口统计学条件提示两条路线，首次探究了当提示中未提供人口统计学信息时，大语言模型在标注任务中会默认模仿哪些人类标注者群体的人口统计学特征。

    

    标注者的人口统计学特征和文化背景会影响他们在文本标注中分配的标签——例如，一位老年女性可能会认为一条称呼“兄弟”的信息具有冒犯性，而一位青少年男性则可能认为这是合适的。因此，承认标签的差异性十分重要，以避免低估社会中某些群体的代表性。在将大语言模型（LLM）用于数据标注的背景下，由此观察发展出两个研究方向：（1）研究大语言模型的偏差及其内在知识；（2）通过在提示中加入人口统计学信息来为输出注入多样性。我们将这两条研究路线相结合，提出了一个问题：当未给出人口统计学信息时，大语言模型会默认采用哪种人口统计学特征？为了回答这个问题，我们评估了大语言模型在本质上会模仿人类标注者的哪些属性。此外，我们还比较了无人口统计学条件约束的提示和安慰剂条件约束的提示。

    arXiv:2410.08820v4 Announce Type: replace  Abstract: Demographics and cultural background of annotators influence the labels they assign in text annotation -- for instance, an elderly woman might find it offensive to read a message addressed to a "bro", but a male teenager might find it appropriate. It is therefore important to acknowledge label variations to not under-represent members of a society. Two research directions developed out of this observation in the context of using large language models (LLM) for data annotations, namely (1) studying biases and inherent knowledge of LLMs and (2) injecting diversity in the output by manipulating the prompt with demographic information. We combine these two strands of research and ask the question to which demographics an LLM resorts to when no demographics is given. To answer this question, we evaluate which attributes of human annotators LLMs inherently mimic. Furthermore, we compare non-demographic conditioned prompts and placebo-condi
    
[^168]: 通过协调双重动态索引机制释放大语言模型在序列推荐中的潜力

    Unleash LLMs Potential for Sequential Recommendation by Coordinating Dual Dynamic Index Mechanism

    [https://arxiv.org/abs/2409.09253](https://arxiv.org/abs/2409.09253)

    该论文提出了首个采用双重动态索引机制的端到端大语言模型序列推荐系统ED²，将索引生成与序列推荐统一到单一LLM主干流水线中，同时解决了语义信息与协同信息整合不足以及高阶用户-物品交互模式利用不充分的问题。

    

    由于在语义理解和逻辑推理方面具有前所未有的能力，大语言模型（LLMs）在开发下一代序列推荐系统（RSs）方面展现出了巨大的潜力。然而，现有的基于LLM的序列推荐系统大多将索引生成与序列推荐相分离，导致语义信息与协同信息之间的整合不足。另一方面，对用户相关信息的忽视阻碍了基于LLM的序列推荐系统利用高阶用户-物品交互模式。在本文中，我们提出了端到端双重动态（ED²）推荐器，这是首个采用双重动态索引机制的基于LLM的序列推荐系统，旨在同时解决上述局限性。双重动态索引机制不仅能够将索引生成和序列推荐整合到统一的以LLM为主干的流水线中，还能使其……

    arXiv:2409.09253v2 Announce Type: replace-cross  Abstract: Owing to the unprecedented capability in semantic understanding and logical reasoning, large language models (LLMs) have shown fantastic potential in developing next-generation sequential recommender systems (RSs). However, existing LLM-based sequential RSs mostly separate index generation from sequential recommendation, leading to insufficient integration between semantic information and collaborative information. On the other hand, the neglect of user-related information hinders LLM-based sequential RSs from exploiting high-order user-item interaction patterns. In this paper, we propose the End-to-End Dual Dynamic (ED$^2$) recommender, the first LLM-based sequential RS which adopts dual dynamic index mechanism, targeting resolving the above limitations simultaneously. The dual dynamic index mechanism can not only assembly index generation and sequential recommendation into a unified LLM-backbone pipeline, but also make it pra
    
[^169]: 加纳诸语言自然语言处理的系统性综述：数据集、模型与研究路线图

    A Systematic Review of NLP for Ghanaian Languages: Datasets, Models, and a Research Roadmap

    [https://arxiv.org/abs/2405.06818](https://arxiv.org/abs/2405.06818)

    本文首次系统综述了加纳NLP领域，通过分析36项核心研究揭示了严重的资源失衡问题——仅特威语有适度发展而其余70多种语言几乎空白，并针对区域限制、方言差异、书写系统非标准化和基础设施缺失提出了优先级研究路线图。

    

    针对加纳73种现存本土语言的自然语言处理（NLP）研究仍然高度碎片化、资源匮乏，且严重偏向单一语言。我们首次对加纳NLP领域进行了系统性综述，筛选了四个学术数据库中超过17,000篇出版物，批判性地综合了36项涵盖数据集、模型架构和评估范式的核心研究。我们的分析揭示了严重的资源失衡：以特威语为中心的NLP研究仅得到适度发展，且主要由宗教文本对齐和众包驱动，而其余70多种语言几乎完全未被涉及，此外数据集发布、模型检查点和评估实践在整个领域内缺乏一致性且很少共享。我们将这些发现转化为一份优先级路线图，针对加纳严峻的区域限制、方言差异、非标准化书写系统以及共享基础设施缺失等关键问题提出解决方案。

    arXiv:2405.06818v2 Announce Type: replace  Abstract: Natural Language Processing (NLP) for Ghana's 73 living indigenous languages remains deeply fragmented, under-resourced, and heavily skewed toward a single language. We present the first systematic review of the Ghanaian NLP landscape, screening 17,000+ publications across four academic databases to critically synthesize 36 core studies spanning datasets, model architectures, and evaluation paradigms. Our analysis exposes a severe resource imbalance: Twi-centric NLP has grown modestly, driven largely by religious-text alignment and crowdsourcing, while the remaining 70+ languages remain almost entirely unaddressed, and dataset releases, model checkpoints, and evaluation practices remain inconsistent and rarely shared across the field. We translate these findings into a prioritized roadmap targeting Ghana's acute regional constraints, dialectal variation, non-standardized orthographies, and the absence of shared infrastructure, offeri
    
[^170]: 科学文献中的引用归因：新基准与方法

    Attribution in Scientific Literature: New Benchmark and Methods

    [https://arxiv.org/abs/2405.02228](https://arxiv.org/abs/2405.02228)

    该论文提出了科学引用归因基准REASONS以及由弃答率与幻觉率构成的双指标评估框架，系统评估了大语言模型在不同证据条件下的引用归因能力，发现高级RAG虽能降低幻觉率但牺牲了弃答能力，而对抗性元数据会使多个系统的幻觉率超过85%。

    

    大型语言模型越来越多地生成带有引用支持的回答，然而引用幻觉仍然是可信科学信息获取面临的主要挑战。我们提出了REASONS，这是一个包含12,723个句子级引用实例的基准，涵盖12个arXiv学科类别，旨在评估不同证据条件下的科学引用归因。我们提出了一个由弃答率和幻觉率组成的双指标框架，用于刻画可靠性与响应性之间的权衡。通过作者归因和标题归因任务，我们在零上下文、元数据增强、级联元数据增强提示（CMP）、检索增强和对抗等多种设置下评估了专有和开源大语言模型。高级RAG相对于朴素RAG降低了幻觉率（65.4%对比87.6%），但使弃答率从5.0%降至0%。在对抗性元数据下，多个系统的幻觉率超过85%。

    arXiv:2405.02228v4 Announce Type: replace-cross  Abstract: Large language models (LLMs) increasingly generate citation-backed responses, yet citation hallucination remains a major challenge for trustworthy scientific information access. We introduce REASONS, a benchmark of 12,723 sentence-level citation instances spanning 12 arXiv subject categories, designed to evaluate scientific citation attribution under varying evidence conditions. We propose a dual-metric framework consisting of Abstention Rate (AR) and Hallucination Rate (HR) to characterize the trade-off between reliability and responsiveness. Using author-attribution and title-attribution tasks, we evaluate proprietary and open-source LLMs under zero-context, metadata-augmented, cascaded metadata-augmented prompting (CMP), retrieval-augmented, and adversarial settings. Advanced RAG lowers HR relative to Naive RAG (65.4% vs. 87.6%) but reduces AR from 5.0% to 0%. Under adversarial metadata, several systems exceed 85% HR, while 
    
[^171]: "您是一名专家注释者": 自动化情绪强度建模的最佳-最差标度注释

    "You are an expert annotator": Automatic Best-Worst-Scaling Annotations for Emotion Intensity Modeling

    [https://arxiv.org/abs/2403.17612](https://arxiv.org/abs/2403.17612)

    自动标记情绪强度建模中的最佳-最差标度注释方法的性能表现

    

    标记语料库构成了为新任务或领域创建模型的瓶颈。大型语言模型通过自动语料库标记方法，特别是针对分类标记，缓解了这一问题。然而，一些NLP任务（如情绪强度预测）需要文本回归，但目前尚无关于连续标签分配自动化标记的工作。回归被认为比分类更具挑战性：当人类被要求从评分尺度中选择数值时表现更差，这导致了比较注释方法，包括最佳-最差标度。这引发了一个问题，即基于大型语言模型的标注方法是否显示类似的模式，即它们在评分标度注释任务上的表现比在比较标度注释任务上更差。为了研究这一点，我们自动化情绪强度预测并比较直接评分预测、成对比较和最佳-最差标度。我们发现

    arXiv:2403.17612v1 Announce Type: new  Abstract: Labeling corpora constitutes a bottleneck to create models for new tasks or domains. Large language models mitigate the issue with automatic corpus labeling methods, particularly for categorical annotations. Some NLP tasks such as emotion intensity prediction, however, require text regression, but there is no work on automating annotations for continuous label assignments. Regression is considered more challenging than classification: The fact that humans perform worse when tasked to choose values from a rating scale lead to comparative annotation methods, including best-worst scaling. This raises the question if large language model-based annotation methods show similar patterns, namely that they perform worse on rating scale annotation tasks than on comparative annotation tasks. To study this, we automate emotion intensity predictions and compare direct rating scale predictions, pairwise comparisons and best-worst scaling. We find that
    

