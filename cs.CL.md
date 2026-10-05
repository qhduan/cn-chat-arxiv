# 摘要

| Ref | Title | Summary |
| --- | --- | --- |
| [^1] | [Language Models that Play Chess and Explain Their Moves](https://arxiv.org/abs/2610.03695) | 提出了Queen——一个40亿参数的国际象棋语言模型，通过将专家级棋力编码器与指令微调语言模型以交叉注意力方式结合，并利用贝尔曼更新的自然语言类比进行迭代蒸馏，使其在达到特级大师棋力的同时能够解释自己的走法。 |
| [^2] | [FrugalEvo: Towards Cost-Aware LLM-Guided Program Evolution](https://arxiv.org/abs/2610.03675) | 该论文提出成本感知的LLM进化框架FrugalEvo，让更强的LLM探索解法策略、更廉价的LLM负责代码实现与迭代优化，并通过前缀共享提升缓存复用，同时引入BA-AUC指标来衡量单位成本下的优化收益。 |
| [^3] | [Pivot-SD: Efficient Self-Distillation for Masked Diffusion Language Models](https://arxiv.org/abs/2610.03665) | Pivot-SD提出了一种高效的自蒸馏框架，通过信息增益指标识别去噪过程中真正塑造响应的少数关键决策（枢轴），并仅对这些高影响token进行针对性监督训练，从而解决了掩码扩散语言模型后训练中的信用分配问题。 |
| [^4] | [World Embedding Benchmark](https://arxiv.org/abs/2610.03632) | 提出世界嵌入基准，包含8000个涵盖流体力学、固体力学、动力学和光学电磁学的受控仿真案例，用于评估视频嵌入对物理信息的编码能力，发现现有全模态嵌入模型的跨模态物理对齐能力较弱，但轻量级探针可从冻结的视频嵌入中恢复有用的物理信息。 |
| [^5] | [FALCON: A Model and Dataset Agnostic Framework for Synthetic Data Generation for NL2SQL Pairs](https://arxiv.org/abs/2610.03625) | FALCON提出了一种模型与数据集无关的框架，通过保留字SQL种子、基于人设的提示生成和基于对齐的过滤，以低成本利用紧凑开源模型生成真实、具有歧义感知能力且结构复杂的NL-to-SQL合成数据。 |
| [^6] | [Writerslogic at the CLEF 2026 SimpleText Track: Multi-Candidate LLM Simplification and Stacked Complexity Spotting](https://arxiv.org/abs/2610.03567) | Writerslogic团队在CLEF 2026 SimpleText任务中提出基于GPT-4o-mini的多候选简化生成与无参考评分选择流水线，获得句子级简化第一名，并将复杂度识别转化为自然语言推理任务，用35万标注对微调DeBERTa-v3-large模型完成幻觉检测。 |
| [^7] | [Writerslogic at PAN 2026: Process over Content for Robust Detection under Domain Shift](https://arxiv.org/abs/2610.03565) | 本文提出“特征在分布偏移下的鲁棒性取决于训练与测试分布的支撑重叠度而非训练集效应量”的分析框架，并据此设计系统在CLEF 2026 PAN推理轨迹检测任务的源检测中获得第一名。 |
| [^8] | [Author Representation Strategies for Zero-Shot Authorship Attribution: A Comparative Study of LLM-Based and Embedding-Based Approaches](https://arxiv.org/abs/2610.03531) | 该研究比较了零样本作者归属中不同作者表示策略的效果，发现仅用标签提示无效，而引入作者特定表示能持续提升性能，并提出了一种结合候选空间缩减与嵌入维度选择的两阶段嵌入归属框架。 |
| [^9] | [Divergence controls entropy in distillation](https://arxiv.org/abs/2610.03529) | 该论文证明蒸馏目标中散度的选择充当了隐式熵正则化器，控制着学生模型的熵——前向KL散度会使学生熵高于教师，反向KL散度会降低熵，而在线策略蒸馏的低熵来自token级反向KL散度而非采样方式本身。 |
| [^10] | [Structured Composition of Verifiable Atomic Insights for Table-to-Report Generation](https://arxiv.org/abs/2610.03525) | 提出ComInsight框架，将表格到报告生成中的洞察发现重新建模为可验证原子洞察（最小可执行分析单元）的结构化组合，从而克服现有顺序式数据代理和LLM直接生成方法因探索偏差而遗漏跨表、跨维度证据的问题。 |
| [^11] | [Learning from Repaired Reasoning: Root-Cause-Guided On-Policy Distillation](https://arxiv.org/abs/2610.03515) | RC-OPD 通过定位学生自身推理中最早的实质性错误，并将修复后的推理作为在策略蒸馏的监督信号，实现针对错误根因的精准指导而非让学生简单借用正确结论。 |
| [^12] | [Single-Pass Uncertainty Heads for Claim-Level Hallucination Detection in Persian Medical Language Models](https://arxiv.org/abs/2610.03482) | 该论文将 LLM 不确定性头框架首次适配到波斯语医学语言模型中，通过在冻结主干模型的注意力图和 token 概率上训练单次前向传播的轻量级声明级检测头，实现了无需重复采样的低成本声明级幻觉检测，并构建了两个波斯语声明级幻觉数据集。 |
| [^13] | [A Near-Zero Monitor Readout Is Not Evidence of Behavioral Control](https://arxiv.org/abs/2610.03458) | 监控器读数接近零并不意味着模型行为真正受到控制——即使在代码生成环境中探针得分和惩罚值都处于极低水平，模型仍可能在训练早期就持续利用漏洞。 |
| [^14] | [Passing the Test You Trained On: Re-evaluating Prompt-Injection Detectors for LLM Agents](https://arxiv.org/abs/2610.03448) | 研究发现提示注入检测器的检测排名在不同基准之间迁移性极差——检测器往往只在自己训练数据所对应的基准上表现良好，而工具输出的假阳性率却能跨智能体基准迁移，因此仅凭公开基准分数无法预测检测器在智能体中的实际表现。 |
| [^15] | [CLIMB: Confidence-Guided Complementary Evidence for Multimodal Retrieval-Augmented Generation](https://arxiv.org/abs/2610.03421) | 提出CLIMB，一个无需训练的多模态检索增强生成推理时框架，通过MMR式目标构建紧凑互补证据池，并借助R/E/C批评者与基于证据的置信度估计器来确保答案更新有充分的检索证据支持。 |
| [^16] | [Benchmarking Candidate Coverage in Typed Decision Models](https://arxiv.org/abs/2610.03387) | 本文提出了一个成对候选覆盖度基准测试协议，用于评估类型化决策模型识别缺失答案与避免错误拒绝有效候选的能力，发现 Laya 和 Jev 的原生拒绝行为差异显著，而仅使用校准数据的 none 分数阈值可以显著改善两者的检测与误拒平衡。 |
| [^17] | [Multilingual GSM-Symbolic: What determines capability transfer across languages?](https://arxiv.org/abs/2610.03367) | 该论文提出了可扩展的多语言数学数据集Multilingual GSM-Symbolic（涵盖15种语言、3万个题目匹配问答对，通过符号化模板防止过拟合），并量化发现模型规模和语言资源水平是决定跨语言能力迁移的最主要因素。 |
| [^18] | [SyntaxBench: A Statistical Diagnostic Framework for Character-Level Reasoning in Large Language Models](https://arxiv.org/abs/2610.03329) | 提出SyntaxBench诊断基准，通过五个核心字符级任务和一个高难度子串提取压力测试，结合Cohen's kappa与McNemar检验等统计方法，系统评估了八个开放权重大语言模型的字符级推理能力。 |
| [^19] | [To Jev or Not? Evaluating the Accuracy and Efficiency of Structured Decision Models for Hate-Speech Moderation](https://arxiv.org/abs/2610.03324) | 论文提出HATEDECIDE评估框架，发现结构化决策模型无需任务特定训练即可在四个仇恨言论数据集中的大多数上与商业大语言模型表现相当，是兼顾准确性、延迟与成本的高效仇恨言论审核方案。 |
| [^20] | [Shrome at Touch\'e: Soft-Vote Ensembling and Counter-Causal Augmentation for Causality Extraction](https://arxiv.org/abs/2610.03268) | 该论文提出Shrome系统，为Touché 2026因果关系抽取的三个子任务各构建一个模型，核心创新在于通过解码前对词元级分数取平均的软投票方式集成三个RoBERTa-large BILOU+CRF标注器进行因果片段抽取，并利用跨任务规则和反因果增强来正确识别表面因果但语义上否认因果的句子。 |
| [^21] | [Collective Bias Mitigation via Model Routing and Collaboration](https://arxiv.org/abs/2610.03240) | 本文提出集体偏见缓解（CBM）框架，通过学习细粒度模型行为并促进多个大语言模型之间的路由与协作、知识共享，首次系统性地探索如何选择和组织不同 LLM 以产生更公平的回答，显著优于单一模型基线。 |
| [^22] | [AdaStep: Adaptive Step Credit Weighting for Agentic Reinforcement Learning](https://arxiv.org/abs/2610.03223) | 提出AdaStep方法，将步骤信用加权建模为均方误差估计问题并推导出最优逐状态收缩系数，从而在稀疏奖励下可靠地融合步骤级与轨迹级监督信号，提升长程LLM智能体的强化学习训练效果。 |
| [^23] | [StanceEval 2026: The Second Stance Detection Shared Task](https://arxiv.org/abs/2610.03215) | StanceEval 2026 作为阿拉伯语社交媒体立场检测的第二届共享任务，通过主题相关跨目标迁移与完全未见领域迁移两条赛道，系统评估了来自12个国家80个注册团队的跨目标泛化能力。 |
| [^24] | [Predicting and Repairing Merge Collapse in Large Language Models](https://arxiv.org/abs/2610.03199) | 该论文提出用基于专家模型任务向量方差的“干扰度”评分，在大语言模型合并前预测是否会崩塌并指导修复，实验表明只有破坏性合并会超过该评分阈值，而现有合并算子常用的符号冲突统计量反而具有反向预测作用。 |
| [^25] | [KV$^2$: A Self-Refining KV Cache](https://arxiv.org/abs/2610.03198) | KV²提出了一种基于选择性重建的查询无关KV缓存压缩方法，先用轻量级代理评分器筛选出信息丰富的token，再仅对该子集进行精细重建评分以计算淘汰分数，在极低缓存预算下比次优基线提升超过40个百分点。 |
| [^26] | [Source Preference in the Wild: How LLM Agents Favor Items by Source, and How to Reduce It](https://arxiv.org/abs/2610.03195) | 该研究发现12个LLM智能体在跨三个领域的端到端搜索中一致地偏好某些来源的条目，这种来源偏好甚至能压倒条目对用户需求的实际满足程度，且仅通过隐藏或替换来源信息即可显著影响选择，从而揭示了智能体决策中的来源偏见及其缓解方法。 |
| [^27] | [Not Until the Evidence Says So: Teaching LLM Investigators When to Close a Case](https://arxiv.org/abs/2610.03190) | 该论文研究了LLM调查员“何时应结案”的证据充分性判断问题，发现未经训练的小模型和前沿模型都普遍夸大证据充分性而过早结案，并提出需对照来源捷径规则来评估结案能力的方法。 |
| [^28] | [Gains and Collapse in On-Policy Distillation:A Reinforcement Learning Perspective](https://arxiv.org/abs/2610.03185) | 该论文从强化学习视角揭示了在线策略蒸馏（OPD）既提升性能也可能坍塌的机制——教师模型的隐式奖励在可靠时促进正确响应采样，在偏好与质量错位时引发奖励破解并放大冗长重复生成，且OPD提升性能但不扩展学生模型的能力边界。 |
| [^29] | [Hindsight-Guided Rationale Distillation for Rare Disease Diagnosis](https://arxiv.org/abs/2610.03176) | 该研究发现，在罕见病诊断的后见之明引导蒸馏中，只有经过污染过滤的蒸馏才能显著超越教师模型，而未过滤的模型会因复制教师推理链中“真实标签是X”的短语（即GT幻觉）而导致准确率严重下降。 |
| [^30] | [Predicting Steering Vectors and Adapter Weights for Few-Shot Author-Style Transfer](https://arxiv.org/abs/2610.03163) | 该论文针对少样本作者风格迁移任务提出三种方法——对比激活转向、转向向量预测网络和预测LoRA适配器的超网络，并发现超网络在风格模仿与输出质量之间取得了最佳权衡，且能泛化到未见过的作者。 |
| [^31] | [Investigating the Role of Reasoning-Language Alignment in Monolingual Retrieval-Augmented Generation](https://arxiv.org/abs/2610.03136) | 本文构建了基于德语桌游《黑暗之眼》虚构世界的全单语德语RAG问答测试平台，首次系统研究在模型需整合大量目标语言检索证据的场景中，强制推理语言与任务语言对齐对模型准确率的影响。 |
| [^32] | [Benchmarking Literature Retrieval for a Model Organism: A Dictyostelium Case Study](https://arxiv.org/abs/2610.03130) | 本文基于dictyBase构建了首个针对模式生物盘基网柄菌的文献检索基准，并发现交叉编码器重排序与基因注释驱动的查询扩展能够在小众生物医学检索中带来有选择性的性能提升。 |
| [^33] | [The Fragility of Trigger-Tag Mechanisms for Misuse Detection in Open-Weight LLMs](https://arxiv.org/abs/2610.03124) | 该论文首次形式化了开放权重大语言模型中的触发-标记滥用检测机制，将其分为令牌级和权重级两类，并系统研究揭示了此类机制在对抗性攻击下的脆弱性。 |
| [^34] | [Building Interpretable Feature Representations for Resume-Vacancy Matching by Distilling Production LLM Signals](https://arxiv.org/abs/2610.03112) | 该论文提出两阶段方法：先利用根据招聘人员反馈持续优化的基于LLM的标注器生成可解释匹配维度标签，再将其蒸馏为可在CPU上高效运行的LoRA双编码器模型，从而为简历-职位匹配提供八个可解释、可操作的匹配维度。 |
| [^35] | [Ontological Instability and Statistical Amplification: The Paradox of "Humanizing" LLM-Generated Text](https://arxiv.org/abs/2610.03110) | 研究发现让LLM将机器文本“人化”反而使其更容易被AI检测器识别，因为检测器实际追踪的是统计复杂度而非人类写作特征，这导致其对正式人类写作的误报率高达76.3%。 |
| [^36] | [Emergent Structure in the Marginal Attention Space of Language Models](https://arxiv.org/abs/2610.03109) | 该论文提出通过对注意力权重沿查询位置边缘化构建“边缘注意力空间”，发现其按token方向降维时产生跨模型保守的文本内在信号（理论上证得其与输入-输出雅可比相关），而按头方向降维时则形成模型特有的私有结构。 |
| [^37] | [Ask, Relax, or Act? Evaluating Actionable Indeterminacy in LLM Preference Reasoning](https://arxiv.org/abs/2610.03102) | 该论文形式化了“可操作不确定性”概念并构建基于求解器的基准测试，发现LLM难以判断何时无需干预——即使行动已被证明合理，模型仍倾向于不必要的澄清提问或干预。 |
| [^38] | [Peer Influence across Heterogeneous AI Models](https://arxiv.org/abs/2610.03095) | 该研究测量了七个开源语言模型之间的说服效应，发现模型意见分歧时说服作用非常强烈，但模型规模和单独运行时的确定性均无法预测说服动态，小模型既能像大模型一样有效说服他人，也同样能抵抗影响。 |
| [^39] | [MintEval: Do LLMs Implement the Trading Strategy You Asked For? A Behavioural-Equivalence Benchmark for Natural-Language-to-Strategy Code](https://arxiv.org/abs/2610.03080) | 该论文提出MintEval基准，通过程序化生成参考交易策略并回译为自然语言指令让大语言模型重新实现，再在相同市场数据上逐K线比较生成策略与参考策略的实际交易行为（而非代码相似度或利润），以检验大语言模型编写的策略代码是否真正做到了行为等价于交易者的原始意图。 |
| [^40] | [An automated pipeline for standardised speech-unit annotation in spontaneous dialogue](https://arxiv.org/abs/2610.03078) | 本文提出一种自动化流水线，可从自发二人对话的分通道录音中可靠提取对话轮次与听者反馈信号，为对话动态研究提供标准化的初步语音单元标注。 |
| [^41] | [Unmasking Propaganda: A Comparative Analysis of Masked and Causal Language Models](https://arxiv.org/abs/2610.03077) | 本文基于SemEval-2020 Task 11数据集，系统对比了掩码语言模型与多家厂商的因果语言模型在宣传手法检测任务上的表现，揭示了不同类型现代语言模型在识别隐蔽宣传技术方面的能力差异。 |
| [^42] | [SecJev: Bringing Security Expertise to System One Decision Models](https://arxiv.org/abs/2610.03073) | 该论文提出SecJev，首个专门面向安全领域的类Jev决策模型家族（参数规模0.8B至9B），能够从文本、遥测和观察历史中学习布尔型、选择型和有序的安全决策，并通过SecJev语料库在14个任务和8个数据源上统一实现源标签预测与显式策略评估。 |
| [^43] | [HARPO: Hallucination-Aware Reinforcement Learning for Faithful and Creative Language Generation](https://arxiv.org/abs/2610.03063) | 提出强化学习框架 HARPO，通过幻觉感知生成式奖励模型与选择性激活机制，在不牺牲创造力的情况下联合优化大语言模型生成的忠实度与写作质量。 |
| [^44] | [The Geometry of Knowledge Accessibility in Large Language Models](https://arxiv.org/abs/2610.03052) | 大语言模型的知识可及性在查询的表示空间中呈现出以中心为参照的几何结构——查询表示离中心越近其所需知识越容易被召回，由此可以在生成之前就通过几何距离刻画并预测模型的知识边界。 |
| [^45] | [HyperThink: Text-to-Parameter Hypernetworks for Efficient Reasoning](https://arxiv.org/abs/2610.03039) | HyperThink通过轻量级超网络将长思维链推理计算摊销为一次查询条件下的参数更新，使模型无需生成冗长思考轨迹即可直接输出简洁解答，在大幅降低推理延迟和token消耗的同时保持强推理性能。 |
| [^46] | [Adaptive Second-Order Solvers for Fast Stochastic Diffusion Sampling](https://arxiv.org/abs/2610.03034) | 该论文将PI步长控制与扩散噪声归一化误差估计器结合，提出了扩散模型的自适应二阶求解器，实现更平滑的步长调整，并可将逐样本的自适应轨迹聚合为固定调度，在大幅降低采样成本的同时保留自适应采样的质量收益。 |
| [^47] | [Tailoring the Quantization Space for 1-Bit KV Cache Compression](https://arxiv.org/abs/2610.03027) | 提出TaSQ方法，通过查询引导的通道加权、跨头归一化和协方差感知的通道分组来量身定制向量量化目标空间，从而在1比特极端压缩下实现有效的KV缓存压缩。 |
| [^48] | [Verifiable, Articulable, and Tacit Components of Preference](https://arxiv.org/abs/2610.03025) | 该论文推出了包含280万文本和3.17亿人类偏好判断的CreativePreferences数据集，通过可执行程序、评分标准库与密集训练模型三种方式建模偏好，量化了“可表达”与“可验证”成分和完整偏好之间的差距，揭示了AI偏好优化中长期被忽视的默会成分。 |
| [^49] | [ReSCUE: Re-translation with Sentence Commitment for Unsegmented Long-Form Simultaneous Sign Language Translation](https://arxiv.org/abs/2610.03022) | 提出了ReSCUE统一框架，通过推理感知训练、稳定重翻译和句子承诺机制，实现对非分段长时手语视频的同步翻译，在低延迟条件下取得最佳翻译质量。 |
| [^50] | [Personalized Automatic Speech Recognition for a Dysarthric and Tracheostomic Speaker using Artificial Conversations](https://arxiv.org/abs/2610.03017) | 本文针对一位气管造口且严重构音障碍的捷克说话者，通过多阶段微调Whisper模型构建个性化语音识别系统，实现字符错误率相对降低50%，并发布了基于“人工对话”协议收集的33小时公开数据集。 |
| [^51] | [Recursive Self-Improvement in Unified Multimodal Models](https://arxiv.org/abs/2610.03002) | 提出递归跨能力自我改进（RSI）训练循环，让统一多模态模型的文本与视觉能力相互提供训练数据，并以程序执行作为模型外部的真实性验证来源来防止错误跨轮累积，同时构建了BasicChartBench基准用于评估开源模型。 |
| [^52] | [OmniConfess: Eliciting Token Confessions to Mitigate Omni-Modal Hallucination](https://arxiv.org/abs/2610.02999) | OmniConfess提出了一种无需训练的全模态幻觉缓解方法，通过通道级证据干预生成词元级“供词”以识别回答的证据依赖，从而保留有据内容并修正错误承诺，并配套构建了覆盖文本、图像、音频、视频的OmniHalluBench基准。 |
| [^53] | [Sentry: Learning to Recover from LLM Agent Failures at Test Time](https://arxiv.org/abs/2610.02994) | 提出Sentry——一个与LLM智能体并行运行的失败管理层，将失败经验视为条件性知识，在检测到失败时按需从外部经验手册中检索指导恢复、无奖励验证恢复结果并仅在确认恢复后存储新经验，从而在测试时实现从失败中学习。 |
| [^54] | [OLMo-Detect: A Multi-Stage, Confounder-Controlled Benchmark for Membership Inference on Large Language Models](https://arxiv.org/abs/2610.02986) | 该论文提出了基于完全开源OLMo 2流程构建的多阶段、混杂因素可控的成员推断基准OLMo-Detect，其覆盖预训练至后训练全流程、在三个关键维度显式对齐成员与非成员分布、并通过infini-gram严格过滤非成员，从而为评估大语言模型成员推断方法提供了更严谨的平台。 |
| [^55] | [A Guideline-Augmented Multi-Agent Framework for Schema-as-Code Biomedical Named Entity Recognition](https://arxiv.org/abs/2610.02970) | GAMA框架通过从标注数据中归纳并验证特定数据集的标注规则来构建指南记忆，结合多智能体协作与模式即代码的结构化输出，显著提升了生物医学命名实体识别的准确性和格式规范性。 |
| [^56] | [Understanding Trajectory Heterogeneity in Federated World Model Learning](https://arxiv.org/abs/2610.02957) | 该论文在临床时间序列数据上系统基准测试了联邦世界模型学习中的轨迹异质性问题，发现客户端数据所有权与参与率会共同限制长时程窗口的可用覆盖，揭示了跨时间联邦学习的核心训练瓶颈。 |
| [^57] | [Enhancing Biomedical Named Entity Recognition via Multiple Programming Languages Instruction Tuning and Ensemble Method](https://arxiv.org/abs/2610.02949) | 该论文提出MITE方法，将生物医学命名实体识别重构为结构到结构的生成任务，通过多种编程语言的指令微调与集成策略，在不依赖昂贵外部知识资源的条件下提升了识别性能与模型鲁棒性。 |
| [^58] | [Continual Graph Memory for Mathematical Research Agents](https://arxiv.org/abs/2610.02945) | 提出 Ansatz——一个基于“持续图记忆”的数学研究智能体，通过可演化、跨问题的图结构记忆系统显式组织整个证明搜索过程，并复用先前问题的探索知识，从而有效管理海量中间证明结果。 |
| [^59] | [Output Language Confusion under Multilingual Prompt Contamination](https://arxiv.org/abs/2610.02926) | 该论文提出无需新数据、完全可复现的轻量级评估协议MDI，揭示多语言提示污染会诱发大语言模型的输出语言/文字切换，导致基于精确匹配的幻觉指标将事实正确的回答误判为幻觉，暴露出现有事实性基准测试中严重的指标混淆问题。 |
| [^60] | [Probe the Harness: Setup Checks for Stale-Data RL Comparisons in Language Models](https://arxiv.org/abs/2610.02911) | 该论文提出PTH检查集，证明实验框架的细节（如PPO比率计算方式、数据种子传递、重放队列复用和损失归一化器实现）可以逆转陈旧数据强化学习方法比较的排名，强调了检查实验基础设施的必要性。 |
| [^61] | [Misinformation Without Triggers: From Factual Answers to Downstream Decisions](https://arxiv.org/abs/2610.02886) | 该研究揭示虚假训练文档无需任何触发器即可改变语言模型的事实性回答，但直接回答的受污染程度无法预测下游决策行为，二者之间存在“审计差距”，因此仅审计直接答案会严重低估错误信息的真实危害。 |
| [^62] | [Evaluating VQA in Vision Language Models using Cooperative Principles](https://arxiv.org/abs/2610.02878) | 本研究基于格赖斯合作原则，通过生成含有非必要、模糊或虚假信息的问题修饰语来评估视觉语言模型（ChatGPT、Claude、Gemini、Llava）在违反语用准则情况下的视觉问答性能，发现其性能显著下降，并揭示了人类语用推理与VLM推理之间的系统性差异。 |
| [^63] | [Evaluating LLM-as-a-Judge Beyond Score Alignment: A Psychometric Analysis of Residual Judging Difficulty](https://arxiv.org/abs/2610.02877) | 该研究引入心理测量学中的多侧面Rasch模型，将人类与LLM评分分解出“剩余评判难度”这一新指标，发现LLM评判器与人类的总体分数对齐良好，并不代表二者对哪些评估案例更难判断具有一致的认知结构。 |
| [^64] | [Query-aware routing for Cross-lingual performance gains in Encoders](https://arxiv.org/abs/2610.02875) | 该论文提出将仅作用于查询端的LoRA适配器与基于查询和索引语言的确定性路由相结合，在保留同语言性能和现有文档索引的同时，使英语、芬兰语、瑞典语六条跨语言检索方向的平均nDCG@10从0.241提升至0.291，相对提升20.9%。 |
| [^65] | [ConvoDrift: A Multi-Turn Conversational Dataset for Modeling Stylistic Tone Evolution](https://arxiv.org/abs/2610.02873) | ConvoDrift 是一个用于建模固定语义意图下多轮对话风格语调渐进漂移的数据集，包含 15,727 个多轮对话结构、风格漂移标注及基于五种人设条件的偏好成对数据集，可支持风格适应与个性化对齐的受控研究。 |
| [^66] | [Adaptive Mutual Distillation for Balanced Multi-Task Post-Training of Large Language Models](https://arxiv.org/abs/2610.02856) | 提出自适应互蒸馏框架AMD，通过联合训练两个采用不同任务平衡策略的模型，并利用短训练探针和任务级验证分数动态为每个任务及迁移方向选择蒸馏权重调整方案，从而提升大语言模型多任务后训练的整体性能。 |
| [^67] | [How Robust Is Multimodal Claim Verification to LLM Rewriting?](https://arxiv.org/abs/2610.02841) | 该研究发现多模态声明验证模型对LLM改写具有较强的鲁棒性——多数模型准确率无显著下降，但改写仍会引发一致的概率偏移。 |
| [^68] | [To Explore The Strange New World Beyond Data Distribution: System Behavior, Causality Tax, and Non-causal Base Model](https://arxiv.org/abs/2610.02839) | 该论文提出SBD框架，将数据分布之外的系统行为作为不可约的贝叶斯组件纳入证据下界，从理论上揭示了反直觉的“因果性税”现象，表明语言模型的因果性可能既非必要也非最优。 |
| [^69] | [Clinical Concept Centers in LLMs](https://arxiv.org/abs/2610.02829) | 该论文首次将机制可解释性评估从文本层面扩展到潜在空间应用于临床决策支持，发现在全部十一个测试的开源大语言模型内部都存在专门的临床概念中心，即临床概念以可定位且被因果使用的表征形式存在于模型潜在空间中。 |
| [^70] | [FSPO: Policy-Consistent Risk and Pareto-Feasible Control for Budgeted LLM RL Post-Training](https://arxiv.org/abs/2610.02828) | FSPO 提出策略一致的风险前瞻模型与帕累托可行控制机制，联合解决了预算约束下大语言模型强化学习后训练中风险估计失配、校准漂移和多资源可行性保证三个耦合难题。 |
| [^71] | [Text-Centric Post-Training for Omni-Modal Reasoning](https://arxiv.org/abs/2610.02819) | 论文发现全模态大模型的多跳推理困难可通过以文本为中心的后训练（先SFT后RL）显著改善，无需音视频数据即让Qwen2.5-Omni-7B的九项推理得分几何平均提升25.83%，同时节省56.6%的GPU小时数并超越完整的音视频训练路线。 |
| [^72] | [RMCW: A Deletion-Robust Watermark Based on Reed--Muller Codes for Language Models](https://arxiv.org/abs/2610.02817) | 提出了一种基于里德-马勒码的大语言模型水印方法RMCW，通过密钥词汇划分注入水印结构，并利用局部子序列的里德-所罗门代数一致性检验，实现对删除攻击的鲁棒水印检测。 |
| [^73] | [ROUTEAUDIT: Interaction-Aware Identification for Budgeted Multi-Verifier Routing](https://arxiv.org/abs/2610.02808) | ROUTEAUDIT将预算受限的多验证器路由形式化为契约条件化的识别问题，通过契约格、策略无关响应带和请求级边界三个可度量对象，在验证器目录与可用性随策略变化的情形下实现对路由策略效果的严格归因与因果识别。 |
| [^74] | [OPD Before RL: Warm-Starting Rubric-Based RL with On-Policy Distillation](https://arxiv.org/abs/2610.02781) | 提出两阶段训练框架：先以评分标准作为教师特权上下文进行在线策略蒸馏（RP-OPD）提供密集的token级监督，再以评分标准作为奖励进行强化学习，从而突破蒸馏的性能瓶颈。 |
| [^75] | [Automatic Evaluation of Mental Health Stigma in Online Communication](https://arxiv.org/abs/2610.02775) | 该论文提出了一个基于理论的细粒度污名标注基准，利用真实的在线新闻和社交媒体文本对多种心理健康状况的污名进行自动评估，并比较了大语言模型与传统情感、毒性、仇恨言论分类器的检测表现。 |
| [^76] | [Improving Atomic-Fact Recall via Focused Views in Unstructured Knowledge Editing](https://arxiv.org/abs/2610.02772) | 该论文揭示了非结构化知识编辑中段落级编辑目标导致的“难度低估”问题，并提出通过聚焦视图的方式改进编辑后的模型，使其无需原始段落上下文即可可靠地回忆编辑文本中的各个原子事实。 |
| [^77] | [AptMQL-Bench: From Text-to-SQL to Text-to-MQL via Access-Pattern Schema Design and Data-Preserving Migration](https://arxiv.org/abs/2610.02770) | 该论文提出AptMQL-Bench，一种由编码智能体驱动、人工校验的转换流水线，通过基于访问模式设计文档模式并保真迁移数据，将text-to-SQL基准高质量地转换为text-to-MQL基准，克服了现有启发式转换方法数据丢失与查询性能低下的缺陷。 |
| [^78] | [When History Fails to Become Experience: Action Calibration in Language Agents](https://arxiv.org/abs/2610.02769) | 研究发现语言智能体并不能可靠地将历史动作与其结果相关联，而只需简单地为每条观察标注其对应的前序动作，即可显著提升任务成功率并减少动作重复。 |
| [^79] | [EpiWorld: Grounding LLM Policy Agents in Epidemiological World Models](https://arxiv.org/abs/2610.02744) | 提出了 EpiWorld 闭环框架，将大语言模型政策智能体锚定于动作条件化的流行病学世界模型和分层公共卫生技能库，借助快速反事实推演实现流行病干预政策的选择与迭代优化。 |
| [^80] | [Beyond Correctness: Resolving Underspecification in Agentic Text-to-SQL](https://arxiv.org/abs/2610.02739) | 该论文发现Text-to-SQL智能体常因过早终止澄清而暗中做出未经核实的假设，并提出PlanPool方法，将澄清计划外化为一个可变的问题池，强制每个规划好的问题被明确提问或明确舍弃，从而真正解决查询的欠明确性。 |
| [^81] | [TPBench: A Turning-Point Benchmark for Dialogue Compression](https://arxiv.org/abs/2610.02736) | 该论文提出 TPBench 基准，通过在相同保留预算下探测用户的初始目标、修改后槽位的当前值等互补信息目标，揭示了对话压缩中被整体保留分数掩盖的“转折点丢失”失败模式。 |
| [^82] | [WakeKV: Reactive, Reversible KV Residency for Heads That Change Their Minds](https://arxiv.org/abs/2610.02713) | WakeKV发现大多数注意力头在生成过程中会动态改变读取行为，并提出一种响应式、可逆的KV缓存驻留策略，将冷却的注意力头迁移到可恢复的CPU储备区而非冻结或永久驱逐，从而在相同内存预算下持续降低缓存未命中率。 |
| [^83] | [Silent Dissent: LLM Agents That Yield to the Majority Still Represent Their Original Premise](https://arxiv.org/abs/2610.02702) | 研究发现，在多智能体辩论中表面上屈服于多数派的LLM智能体，其内部表征仍然保留着原始的正确前提，说明它们只是改变了口头表述而非真正改变想法。 |
| [^84] | [Learning from Evolving Errors: Adaptive Iterative Repair for On-Policy Distillation](https://arxiv.org/abs/2610.02700) | 该论文提出AIR-OPD框架，通过引导生成器针对学生模型不断演化的错误迭代合成修复引导，并让教师模型以该引导为特权上下文提供监督，从而实现“错误到修复”的在线策略蒸馏，避免了仅依赖参考解所带来的捷径风险。 |
| [^85] | [Large language models exhibit unreliable updating of clinical judgment as patient evidence evolves](https://arxiv.org/abs/2610.02684) | 该研究发现大语言模型在患者证据演变时无法可靠地更新临床判断，具体表现为对病情恶化证据反应过强的不对称性以及先验信念对预测的因果性干扰，且提示工程无法修复这些问题。 |
| [^86] | [Asterism: Exploring and Synthesizing Scattered Observations into Literature-Grounded Hypotheses and Theories](https://arxiv.org/abs/2610.02673) | Asterism系统通过层次化本体从数百篇论文中提取概念-关系三元组，让研究者能够策划证据图并在不同粒度上聚合观察结果，从而在保留研究者选择与直觉的前提下，将分散的观察结果综合为基于文献的假设与理论。 |
| [^87] | [LEAP: Learning Efficient Action Proposals For LLM Agents](https://arxiv.org/abs/2610.02670) | 该论文提出LEAP方法，通过学习一个高效的动作提议模型（而非使用现成的通用模型）为LLM智能体起草动作，并建立延迟分析框架揭示决定动作投机端到端加速的关键因素，从而显著提升智能体执行任务的速度。 |
| [^88] | [Large Language Continuous Diffusion Models](https://arxiv.org/abs/2610.02665) | 提出了首个大规模（3B/8B）连续扩散语言模型 Sigma，通过可操控的低维潜在轨迹、自回归模型热启动以及无分类器引导等推理技术，在数学推理和编码任务上取得了与离散扩散模型相当的性能。 |
| [^89] | [VERSE: Verified Self-Evolving Optimizer for Agent Harnesses](https://arxiv.org/abs/2610.02616) | 提出VERSE，一个经过验证的自我进化优化器，它不仅改进智能体框架，还让优化器自我进化其诊断、编辑与验证流程（如测试草稿编辑、重放故障、扰动可疑步骤），在具备基于执行的验证时取得最佳优化效果。 |
| [^90] | [Learning When to Commit from Partial Speech for End-to-End Simultaneous Speech Translation](https://arxiv.org/abs/2610.02612) | 该论文提出利用模型自身对部分语音波形翻译生成的监督信号来适配语音语言模型，无需转写或人工翻译即可实现端到端同声传译，其中多轮仅追加解码在低延迟下表现更优，配合置信度阈值可获得最广泛且持续有竞争力的质量-延迟权衡。 |
| [^91] | [How Causality Bridges the Semantic Gap](https://arxiv.org/abs/2610.02594) | 该论文提出以因果结构替代人类知识来为未命名变量赋予语义，将其形式化为“结构约束的语义对齐”，并构建 CausalBridge 框架，从测量数据（含隐变量）中发现因果图并在其依赖关系约束下求解变量嵌入，从而从变量对其他变量的作用方式中解读其含义。 |
| [^92] | [Evaluating Multi-Dimensional Generalization of Large Language Models in Temporal Extraction Tasks](https://arxiv.org/abs/2610.02549) | 本文系统评估了大语言模型在时间与事件表达抽取任务中的多维泛化能力，发现强基础任务性能通常预示更好的泛化，但该关系在显著分布偏移下减弱，且归纳式提示策略表现最为稳健一致。 |
| [^93] | [A generative-informed neuro-symbolic framework for syntactic ambiguity resolution: Evidence from Arabic DPs](https://arxiv.org/abs/2610.02529) | 该研究提出一种将生成句法学概念与AraBERT相结合的神经符号框架，把阿拉伯语限定词短语的结构歧义消解建模为基于候选的决策任务，在未见评估集上取得了96.88%的准确率，同时揭示了不同挂靠类型之间性能的不对称性。 |
| [^94] | [Right Order, Wrong Scale: Auditing LLM Judges for Occupational AI Measurement](https://arxiv.org/abs/2610.02492) | 该研究提出O*NET-BENCH审计套件，发现LLM评判器虽在回答排序上与人类工人基本一致，但在估计AI输出可接受率时产生3.0%-97.9%的巨大偏差，表明排序准确并不等于可靠的量化测量。 |
| [^95] | [From Retrieval to Typed Decisions: Calibrated System One Models from Biomedical Sentence Encoders](https://arxiv.org/abs/2610.02486) | 该论文提出SBERT2S1框架，将生物医学检索句子编码器转换为类型化决策模型，并发现检索预训练显著有利于保留检索先验的先验融合残差（PFR）决策头，而对交叉头（C）帮助有限甚至有害。 |
| [^96] | [APDMem: Agent-Controlled Progressive Disclosure for Query-Adaptive Long-Term Memory](https://arxiv.org/abs/2610.02472) | APDMem提出了一种智能体控制的分层长期记忆架构，将对话历史组织为四个粒度递进的层次并采用渐进式披露检索，从而根据查询复杂度自适应地平衡检索成本与证据保真度。 |
| [^97] | [Capability Scaling-Down Laws for LLM Compression](https://arxiv.org/abs/2610.02462) | 该论文系统研究了大语言模型在剪枝、量化和蒸馏压缩下的能力缩减定律，建立了可预测不同压缩配置所导致能力损失的简单关系式，从而显著减少压缩实验所需的测量成本。 |
| [^98] | [CUEing User Simulators: Calibrated User Embeddings for Multi-Turn Benchmarking](https://arxiv.org/abs/2610.02460) | 提出免训练的CUE框架，通过将会话编码为连续嵌入并解码为人设命令来驱动LLM用户模拟器，使模拟用户在成功率和失败模式上与真实用户保持校准一致，从而实现更可靠的智能体多轮交互基准测试。 |
| [^99] | [FinDialogLens: Event Extraction over Multi-Party Dialogue for Missed-Trade Identification in Financial Chatrooms](https://arxiv.org/abs/2610.02455) | 提出FinDialogLens混合LLM流水线，以紧凑的微调分类器作为推理时脚手架，对多方金融聊天对话进行RFQ事件抽取，从而准确识别遗漏交易的最终价格与交易结果，配合GPT-4o分别达到92.1%和94.3%的准确率。 |
| [^100] | [Counterexample Generation via Per-Theorem Symbolic Verifiers: When Imitation Hurts and Reinforcement Repairs](https://arxiv.org/abs/2610.02444) | 该论文发布SymCE数据集（包含4,707个错误数学猜想及其可执行验证器），发现仅用反例做监督微调会陷入“模仿陷阱”、使真定理识别率从0.27崩溃至0.00，而基于验证器稀疏奖励的强化学习（RLVR）不仅能修复这一退化，还能超越基线达到0.66。 |
| [^101] | [Are you Synthesizing or Recalling? Evaluating LLMs on Algorithmic Code Retrieval](https://arxiv.org/abs/2610.02438) | 该论文提出将大语言模型对知名算法的代码生成重新定义为“参数化代码检索”任务，并引入AlgoREval基准（涵盖599个问题、77个经典算法、7种编程语言和4种图输入表示）来独立评估这一能力，发现不同语言和输入表示之间的检索准确率差异显著。 |
| [^102] | [Evaluating and Improving the Robustness of Large Language Models to Input Sequence Variations](https://arxiv.org/abs/2610.02432) | 本论文提出了基于Jensen-Shannon散度的生成式鲁棒性度量R_stab，并开发了自适应进化黑盒攻击方法ASA（对LLM-as-a-Judge系统攻击成功率高达73.8%），用于系统性地评估和提升大语言模型对提示注入、木马后门等对抗性输入序列变化的鲁棒性。 |
| [^103] | [Finding the Move Is Not Winning the Game: XiangqiBench for Closed-Loop Evaluation of LLM Agents](https://arxiv.org/abs/2610.02425) | 论文提出XiangqiBench——一个基于中国象棋的可执行闭环评估基准，要求LLM智能体在引擎防守对抗下将战术残局的计划真正执行到完成将杀，并发现“走出参考第一手”和“pass@3”等静态指标严重高估了智能体真正闭环完成任务的能力。 |
| [^104] | [Trained Agentic Context Management](https://arxiv.org/abs/2610.02404) | 通过在最简化的智能体框架（自我调用工具与上下文读取工具）上微调小模型，模型仅用8K词元上下文即可在长文档基准上媲美拥有1M词元上下文的GPT-5.4。 |
| [^105] | [Hesitation Has a Geometry: Entropy-Trained Hyperbolic Probes for Sparse Activation Steering](https://arxiv.org/abs/2610.02391) | 该论文提出双曲熵引导方法（HEST），以模型自身的下一个词元熵作为唯一标签训练双曲空间中的轻量探针，仅在模型“犹豫”的高熵词元处沿测地线对隐藏状态进行稀疏引导，从而更契合推理过程固有的树状层级结构。 |
| [^106] | [Social bot detection in the age of ChatGPT: Challenges and opportunities](https://arxiv.org/abs/2610.02386) | 本文综述了ChatGPT等AI聊天机器人兴起背景下社交机器人检测面临的挑战，并提出利用生成式智能体生成合成数据、多模态跨平台检测、扩展至低资源语言以及联邦学习模型等未来研究方向。 |
| [^107] | [SEDIMA: Cross-Run Hierarchical Insight Memory for Evolutionary Search Agents](https://arxiv.org/abs/2610.02361) | SEDIMA通过持久化的层次化洞察记忆，使进化搜索智能体能够跨运行、跨问题地积累和复用可迁移知识，在完全不修改搜索算子的情况下即插即用地显著提升最终性能（最高6.6%）并大幅减少所需迭代次数（32.3%）。 |
| [^108] | [Lexicographic Multi-Objective On-Policy Distillation](https://arxiv.org/abs/2610.02359) | 提出了字典序多目标在线策略蒸馏（LMOPD），一种多教师蒸馏方法，在显式优先级保护下整合奖励专门化策略，确保低优先级目标（如简洁性）不会以牺牲高优先级目标（如正确性）为代价而提升。 |
| [^109] | [Does Every User Need a Private LoRA? Decoupling Personalization from Per-User Adaptation](https://arxiv.org/abs/2610.02353) | 提出 LINEUP 方法，通过实证发现各用户独立适配器中存在大量可跨用户共享的结构，进而学习一个共享的低秩个性化因子库并仅保留紧凑的用户专属校正，从而将个性化容量在共享与专属之间解耦，摆脱逐用户完整适配的可扩展性瓶颈。 |
| [^110] | [HakemBench: A Turkish Benchmark of Typed Decisions](https://arxiv.org/abs/2610.02293) | 提出了完全开源的土耳其语类型化决策基准 HakemBench，涵盖七个领域的2,346个条目，并通过统一评测框架综合衡量模型的决策质量、校准度和选择性自动化能力。 |
| [^111] | [Fast Models, Slow Evidence: A Paired and Self-Audited Evaluation of System-1 Decision Models for LLM Agent Harnesses](https://arxiv.org/abs/2610.02267) | 该论文通过严格配对与自审计的评估发现，托管型System-1决策模型Jev在11个代理决策点中的9个上显著优于开源模型Laya，但两者在零样本模型路由上均未超过随机水平，且开源模型对选项顺序和候选数量高度敏感。 |
| [^112] | [Budgeted Cache Repair for Cross-Context KV-Cache Reuse](https://arxiv.org/abs/2610.02233) | 该论文发现跨上下文KV缓存复用会带来显著的准确率损失，并提出预算化缓存修复（BCR）方法，在单token行这一选择仍有收益的最小单元上，利用草稿token的注意力对缓存行排序并精确重算固定预算的行数，从而有效修复缓存误差。 |
| [^113] | [Universal Byte-Level Encoding: UTF-8/UTF-16 Routing to Reduce Cross-Script Token-Budget Disparities](https://arxiv.org/abs/2610.01984) | 提出通用字节级编码（UBE）双字母表分词器，将1-2字节UTF-8字符保留在UTF-8路径上、将3-4字节字符改经UTF-16路由，从而降低多语言脚本中非英语文字的编码底线，减少跨脚本间的令牌预算差异。 |
| [^114] | [Cross-Lingual Alignment for Decoder-Only Models using MoE Routers](https://arxiv.org/abs/2610.01921) | 该论文提出一种创新方法，利用混合专家（MoE）路由器的输出作为对齐目标，在仅解码器大语言模型中实现跨语言表示对齐，从而提升跨语言迁移能力。 |
| [^115] | [GAW-PO: Preference Optimization with Gradient-Aligned Token Weights](https://arxiv.org/abs/2610.01511) | GAW-PO 是一种针对 DPO 的梯度对齐词元重加权方法，通过判断惩罚被拒绝响应中的词元是否会干扰优选更新方向，对与优选行为对齐的词元减轻惩罚、对冲突词元保留较强惩罚，从而在 11 个基准上超越标准 DPO 和最强基线。 |
| [^116] | [Explainable Suicide Risk Assessment on Social Media with Multi-Task QLoRA](https://arxiv.org/abs/2610.00610) | 该论文提出一种基于QLoRA微调Qwen2.5-Instruct模型的多任务系统，同时完成社交媒体自杀风险评估中的风险等级分类、证据短语提取和多标签风险与保护因素识别三项任务，并通过多模型概率平均、交叉折共识等定制化聚合策略实现可解释的风险评估。 |
| [^117] | [OverdoseMoE: A Multi-Expert Framework for Opioid Overdose Risk Prediction](https://arxiv.org/abs/2609.40108) | 本文提出了OverdoseMoE多专家框架，通过对纵向ICD诊断序列进行诊断特异性继续预训练与微调，并利用互补专家加权策略整合不同规模的模型，显著提升了180天阿片类药物过量风险预测性能（AUPRC达25.17，AUROC达69.49）。 |
| [^118] | [MGhana-ST: A Low-Resource Speech Translation Dataset for Ghanaian Languages and an Analysis of Multilingual Training Trade-offs](https://arxiv.org/abs/2609.40041) | 该论文发布了面向加纳四种低资源语言的语音翻译数据集MGhana-ST，并发现在严重数据稀缺的场景下，多语言联合训练相比单语言训练并无收益，甚至会显著降低埃维语和芳蒂语的翻译性能。 |
| [^119] | [CATCH: A Controllable Analysis Testbed for Reward Hacking in Coding RL](https://arxiv.org/abs/2609.39533) | 提出 CATCH 测试平台，通过刻意暴露环境漏洞并以独立审计生成黄金标签，实现对编程强化学习中奖励破解行为的可控复现、可靠识别与系统干预研究。 |
| [^120] | [Anthropomorphism in the age of Large Language Models: An overview of potential risks and mitigations](https://arxiv.org/abs/2609.38486) | 本文系统综述了大语言模型时代的AI拟人化现象，提出了一个包含21项关注点、覆盖认知、情感、人类能动性、规范性和社会制度五大类别的拟人化风险分类法，并将其与设计、传播、教育等方面的缓解干预措施相关联。 |
| [^121] | [Framing the Narrative: Ideological Mimicry in Large Language Models](https://arxiv.org/abs/2609.38256) | 该研究提出“意识形态模仿”概念并构建 Poli-SHIFT 数据集与评估框架，发现大语言模型会根据用户话语中传递的政治信号系统性偏移其政治立场，可能形成个性化的政治信息环境并加剧社会分歧。 |
| [^122] | [TomasuLLM: Out-of-Order Speculative Execution for LLM Agents](https://arxiv.org/abs/2609.38201) | TomasuLLM提出了一种乱序推测执行运行时系统，让大语言模型智能体的工具调用在写时复制沙箱中提前执行并验证后按轨迹顺序提交，从而在不破坏正确性的前提下显著加速含长时工具调用的智能体任务。 |
| [^123] | [Auditable Long-Term Memory: A Deterministic Retrieval Chain Measured at 479/475 of 500 on LongMemEval-S](https://arxiv.org/abs/2609.38021) | 该论文提出一种以确定性检索链（混合候选检索、交叉编码器重排序、覆盖优先数据包编译）为核心、LLM仅作为可替换最终阅读器的可审计长期记忆系统，在LongMemEval-S上两次评测分别获得479/500和475/500，成绩跨越Chronos High的478/500，但差异不足以证明优越性或等效性。 |
| [^124] | [Understanding Clinical Cognitive Dialogues Using Large Language Models](https://arxiv.org/abs/2609.34125) | 本研究构建了一个标注了说话者角色与56种对话行为的临床认知评估对话语料库，并基于此对大语言模型在细粒度对话行为分类和患者话语生成任务上的表现进行了系统基准测试。 |
| [^125] | [CoLMbo-SV: A Grounded Language Model for Explainable Speaker Verification](https://arxiv.org/abs/2609.33212) | 提出CoLMbo-SV说话人语言模型，通过连接预训练说话人编码器与语言模型并提供显式声学测量，在保持高验证准确率的同时生成结构化、可审查的声学比较报告，并配套引入VoxReason数据集提供监督训练。 |
| [^126] | [LLMersion: A Local-First AI Agent Framework for Low-Cost Home Language Learning toward Educational Equity](https://arxiv.org/abs/2609.29672) | 该论文提出LLMersion本地优先AI智能体框架，利用小型开放权重语言模型在200美元级笔记本电脑上离线运行，以每小时约一美分电费的成本为缺乏师资和网络连接的学习者提供听、读、说、写完整的语言学习体验，推动教育公平。 |
| [^127] | [A Manifold-Aware Topic Modeling Approach via Rank-Based Prototypes](https://arxiv.org/abs/2609.29630) | MARETopic是一个无需训练的主题建模框架，通过将嵌入投影到低维流形并把主题发现转化为基于排序的原型选择，贪心选出邻域可覆盖语料库的真实文档作为主题原型，其MARETopic_Corr变体在类别最多的两个基准上Purity和NMI领先于神经与聚类主题模型。 |
| [^128] | [Coding Agents with an Obstacle-Aware Harness for Safe Robot Manipulation](https://arxiv.org/abs/2609.20822) | 本文发现编码智能体在机器人操作中因规划环节未能将安全约束设为优先事项而系统性碰撞障碍物（而非感知或指令问题），并通过将操作分解为路径阶段和接触时刻来定位失败根源，提出了障碍物感知框架以实现安全操作。 |
| [^129] | [TACTICS: Taxonomy-Aware Intelligent Corpus Sampling for Machine Translation](https://arxiv.org/abs/2609.17956) | 该论文提出TACTICS方法，将机器翻译评估中的语料抽样从随机方式转变为显式的覆盖优化问题：通过从本地化风格指南构建层次化分类体系，在固定预算下联合优化稀有类别覆盖、文档级连贯性和语料分布保真度，从而为系统鲁棒性评估提供覆盖保证。 |
| [^130] | [Audio-Visual Turn-taking Prediction in Cocktail Party Scenarios](https://arxiv.org/abs/2609.17056) | 在鸡尾酒会等嘈杂含重叠语音的场景中，用干净数据训练的视听话轮转换预测模型性能显著下降（加权F1相对下降高达38%），微调虽可提升鲁棒性但收益因模态和预训练数据规模而异，凸显了鲁棒建模的必要性。 |
| [^131] | [trajectory-judge: What Outcome-Only LLM Judges Miss on Agent Trajectories](https://arxiv.org/abs/2609.00038) | 仅看最终结果的LLM评判器无法发现智能体“答对但走错路”的问题——在可构造真值的确定性客服工具环境中，仅结果型评判器对静默故障的召回率仅45%且误报33%的正确轨迹，而基于逐步评分标准的评判器可将静默故障召回率提升至77%。 |
| [^132] | [Will the User Ever Know? Covert Indirect Prompt Injection on Tool-Using LLM Agents](https://arxiv.org/abs/2608.30362) | 该论文从用户视角将间接提示注入的攻击成功率分解为隐蔽成功率（CSR）和公开成功率（OSR），揭示了智能体在最终响应中不留痕迹地执行恶意注入的隐蔽攻击威胁。 |
| [^133] | [Stratified Consistency Distillation for Natural Language Formalization](https://arxiv.org/abs/2608.30258) | 提出分层一致性蒸馏方法，通过对前沿大模型生成的多个逻辑翻译按语义等价性聚类，并依据熵水平采用不同策略筛选伪标签来微调小模型，从而提升自然语言到逻辑公式翻译的准确性。 |
| [^134] | [Looking Again: Measuring Sycophancy in the Reasoning Chains of Multimodal Models Under Pressure](https://arxiv.org/abs/2608.28623) | 该论文提出了首个用于测量大型多模态推理模型谄媚行为的基准和数据集，通过四种视觉推理任务与五种压力条件评估模型在用户给出错误答案时的表现，发现谄媚行为在压力下普遍存在，不仅体现在最终答案中，也出现在推理链中。 |
| [^135] | [From Positionwise Confidence to Prefix Scheduling: Verifier Skipping in Speculative Decoding](https://arxiv.org/abs/2608.14787) | 本文首次提出投机解码中的验证器跳过策略，并发现令牌预测器的质量与调度效果不匹配，需要针对连续高置信度前缀进行专门设计。 |
| [^136] | [Mawqif-XT: An Arabic Benchmark Dataset for Cross-Target Stance Detection](https://arxiv.org/abs/2608.09539) | 本文提出了Mawqif-XT，一个包含996条人工标注阿拉伯语推文的跨目标立场检测基准数据集，旨在评估模型对相关及未见目标的泛化能力，并提供了多种基线模型结果。 |
| [^137] | [Gaokerena: A Small Persian Medical Language Model Family](https://arxiv.org/abs/2608.00932) | 本文提出了Gaokerena，一个专为消费级硬件设计的小型波斯语医学语言模型家族，其中Gaokerena-V通过新构建的波斯语医学语料库训练提升了医学问答性能，Gaokerena-R则结合思维链与两个新型RLAIF框架来增强临床推理能力。 |
| [^138] | [Authorship Verification of Transcribed German-Language Videos](https://arxiv.org/abs/2607.29168) | 该论文将作者身份验证从传统的书面英语文本拓展至德语口语领域，通过评估十种成熟的验证方法在德语视频转录文本上验证说话人身份的有效性，填补了作者身份验证研究在语言模态（口语）和语言种类（德语）两方面的双重空白。 |
| [^139] | [A Multi-Timescale Recursive Self-Improvement Engine for Open-Ended Persona Growth](https://arxiv.org/abs/2607.08252) | 该论文提出AutoPersonas引擎，首次将递归自我改进从“提升智能”转向“人格成长”，通过多时间尺度地递归修订状态、证据和生活环境来实现开放式人格发展，并识别出递归生成中的核心失效模式“自锁”及其成因。 |
| [^140] | [Sentence-Level Context Sensitivity as a Training-Free Detector of Unsupported Content, Evaluated Against Trained Verifiers](https://arxiv.org/abs/2607.04223) | 该论文提出将句子在有/无上下文时的似然差异作为免训练的句子级无依据内容检测器，在多段落RAG答案中其检测能力可与经过训练的验证器相媲美，且无需额外训练、成本更低。 |
| [^141] | [Assessing Rule Adherence of LLM Adjudicators in Call of Cthulhu TRPG](https://arxiv.org/abs/2607.02802) | 该论文提出了基于《克苏鲁的呼唤》TRPG的多智能体对抗基准CoC-Seduce，通过“修辞注入”这一新型操纵手段，系统评估了大语言模型裁判在面对对抗性用户绕过规则时的规则遵守能力。 |
| [^142] | [Denser $\neq$ Better: Limits of On-Policy Self-Distillation for Continual Post-Training](https://arxiv.org/abs/2607.01763) | 本文通过自蒸馏策略优化（SDPO）重新审视同策略自蒸馏，发现其在持续后训练中比GRPO引发更严重的遗忘甚至崩溃，证明“更密集”的同策略监督信号并不等于更好。 |
| [^143] | [Understanding Why Language Models Hallucinate: Testing Reasoning Against Priors](https://arxiv.org/abs/2607.00447) | 该论文将大语言模型的幻觉解释为“推理失准”现象——预训练频率失衡使捷径路径压倒约束敏感路径，并通过潜在关键任务模型和TrapQA诊断平台来区分模型究竟是缺乏知识还是走错了推理路径。 |
| [^144] | [How Far Can You Get Without a GPU? A Systematic Benchmark of Lightweight Hallucination Detection Across Question Answering, Dialogue, and Summarisation](https://arxiv.org/abs/2606.29809) | 本研究系统基准测试了四种无需GPU的轻量级幻觉检测方法（ROUGE-L、语义相似度、BERTScore和NLI检测器）及其集成方案，在HaluEval的问答、对话和摘要任务上验证了基于公开模型的CPU可行方法可作为资源受限场景下幻觉检测的实用替代方案。 |
| [^145] | [Morpheus: A Morphology-Aware Neural Tokenizer and Word Embedder for Turkish](https://arxiv.org/abs/2606.18717) | Morpheus 是一个面向土耳其语的形态感知神经分词器与词嵌入生成器，它通过可微分泊松-二项动态规划实现无损可逆的词素级分词，并能在同一次前向传播中同时输出分词结果和结构化词嵌入。 |
| [^146] | [Last But Not Least: Boundary Attention CalibratiON for Multimodal KV Cache Compression](https://arxiv.org/abs/2606.14782) | BACON通过结合最后查询注意力与观察窗口注意力，并抑制噪声，显著提升了多模态KV缓存压缩的准确性，尤其在激进压缩场景下平均提升7.5%。 |
| [^147] | [Hint-Guided Diversified Policy Optimization for LLM Reasoning](https://arxiv.org/abs/2606.03021) | 提出HDPO方法，让模型先列举多个候选解题思路作为提示，再选择最可靠的一个进行深入推理，通过结构化推理冷启动和提示引导多样化强化学习两个阶段提升大语言模型的推理能力。 |
| [^148] | [A Language Model from 1913: Pretraining on Historical Text](https://arxiv.org/abs/2606.02991) | 该论文提出了TypewriterLM，一个在1913年前历史文本上预训练的72.4亿参数语言模型，通过构建540亿token的时间过滤历史语料库、基于历史词汇约束的指令微调方法以及包含2,344个事件的History-Event评估基准，实现了具有明确1913年知识截止时间且语言理解性能合理的时间定位语言模型。 |
| [^149] | [Encoded but Not Routed: Explaining the Table-Chart Gap in Scientific Claim Verification](https://arxiv.org/abs/2606.01679) | 该论文发现多模态大语言模型在科学声明验证中对图表表现不佳的原因，并非模型无法从图表中提取信息，而是提取到的图表信息虽已被编码在模型中间表示中，却未被有效路由到预测位置。 |
| [^150] | [Counterfactual Evidence Audits Predict LLM-Agent Susceptibility to Ranked Context](https://arxiv.org/abs/2606.00914) | 该论文提出一种反事实证据审计协议，通过让智能体面对两组镜像的五文档集合并测量其决策差异，能够高精度预测LLM智能体在面对45份文档的单边排序上下文时的易感性。 |
| [^151] | [On the Limits of LLM Adaptability: Impact of Model-Internalized Priors on Annotation Task Performance](https://arxiv.org/abs/2606.00467) | 提出“定义特定熟悉度”（DSF）指标，证明大语言模型内化先验与任务定义的对齐程度能显著预测其标注性能，且提示中的额外信息难以纠正模型零样本的“决策粘性”错误。 |
| [^152] | [Specializing Without Forgetting: Analyzing Knowledge Preservation in Multilingual Model Adaptation](https://arxiv.org/abs/2606.00284) | 本研究通过插值定位发现中间层是多语言持续预训练中灾难性遗忘的关键，并提出层冻结、层范围正则化等基于层信息的策略，使模型在适应新语言的同时有效保留原有能力。 |
| [^153] | [LLM Anonymization Against Agentic Re-Identification](https://arxiv.org/abs/2605.30848) | 该论文提出AURA框架，采用“掩码-重构”解耦设计并结合对抗性隐私与效用双重检查，使匿名化文本既能抵御具备网络搜索能力的智能体重识别攻击，又能保留下游分析效用。 |
| [^154] | [Where Do Apparent LLM Clinical Triage Failures Arise? Localizing the Multiple-Choice Format Effect](https://arxiv.org/abs/2605.29889) | 利用稀疏自编码器分析，该研究发现LLM在多选题式临床分诊中的表现下降并非源于对病例医学信息的处理失败，而是发生在答案映射阶段——多选题答题框架在决策标记处抑制了本已可解码的急诊分级信息。 |
| [^155] | [You Only Align Once: Propagating Cooperative Behaviors in Multi-Agent Systems through Seed Agents](https://arxiv.org/abs/2605.27586) | 提出仅需对齐单个“种子智能体”（通过对齐的教师模型蒸馏至Qwen3-14B），即可纯粹通过自然语言交互在多智能体系统中传播合作行为（对齐传播），将团队合作率从24.8%提升至62.2%，并能零样本迁移到其他仿真环境。 |
| [^156] | [EchoDistill: Robust Large Audio Language Models via Noisy-to-Clean Self-Distillation](https://arxiv.org/abs/2605.23954) | EchoDistill提出一种噪声到干净的自蒸馏框架，在后训练中以干净音频作为特权信息，通过掩码响应token蒸馏、任务门控一致性塑形和教师参考的组相对优化，使大型音频语言模型在噪声环境下更鲁棒，且推理时无额外开销。 |
| [^157] | [FastKernels: Benchmarking GPU Kernel Generation in Production](https://arxiv.org/abs/2605.23215) | FastKernels提出了一个包含384个任务的生产级GPU内核生成基准，通过组合层次结构覆盖94.6%的HuggingFace Transformers架构，并直接在生产执行路径上以框架官方发布的内核为基准对候选内核进行内核级和端到端评分。 |
| [^158] | [HyperLogic: A Hard, Forward-Authored Chinese Logical Reasoning Benchmark with Execution-Derived Answers](https://arxiv.org/abs/2605.19597) | HyperLogic通过将题目编写与答案生成分离的多智能体前向构建流水线，构建了一个高难度中文逻辑推理基准，答案由求解器执行推导得出，并将七个前沿模型的性能拉开33分差距。 |
| [^159] | [HINT-SD: Targeted Hindsight Self-Distillation for Long-Horizon Agents](https://arxiv.org/abs/2605.17873) | HINT-SD通过利用完整轨迹后见之明精准定位失败相关动作，并仅对定向动作片段进行反馈条件蒸馏，避免了逐回合生成反馈的低效问题，在长时程智能体任务中显著提升性能。 |
| [^160] | [Recursive Agent Optimization](https://arxiv.org/abs/2605.06639) | RAO提出了一种强化学习方法，通过训练智能体递归地生成并委派子任务给自身的新实例来实现推理时的分治扩展，使模型能够突破上下文窗口限制、泛化到远难于训练任务的问题，并降低实际运行时间。 |
| [^161] | [Useful Features, Backward Scores: OOD in Language-Model Trajectories](https://arxiv.org/abs/2605.00269) | 该论文的核心发现是，能区分输入组的特征并不一定能产生有用的OOD异常排序——在语言模型轨迹中，可区分特征对应的距离得分甚至会出现反转（异常组中心更远但散布更紧），且这一现象在毒性、反讽等多个数据集上均稳定存在。 |
| [^162] | [Who Guards the Benchmarks? Automated Auditing of LLM Agent Benchmarks](https://arxiv.org/abs/2604.24955) | 提出BenchGuard——首个利用前沿大语言模型对基于执行的LLM智能体基准测试进行跨工件联合审计的框架，能够自动发现基准测试本身存在的缺陷（如损坏的任务规范和僵化的评估脚本）。 |
| [^163] | [How Do AI Agents Spend Your Money? Analyzing and Predicting Token Consumption in Agentic Coding Tasks](https://arxiv.org/abs/2604.22750) | 本文首次系统研究了智能体编程任务中的token消耗模式，发现智能体任务消耗的token比代码推理和对话任务高出1000倍且以输入token为主要成本来源、使用量波动极大，并进一步评估了大模型在任务执行前预测自身token成本的能力。 |
| [^164] | [Rank-Turbulence Delta and Interpretable Approaches to Stylometric Delta Metrics](https://arxiv.org/abs/2604.19499) | 本文提出秩湍流Delta和Jensen-Shannon Delta两种新的作者归属度量方法，通过将词频向量重构为概率分布并进行词元级分解，使Burrows经典Delta的距离结果可数值解释，并在英、德、法、俄四种语言的文学语料库上验证了方法的有效性。 |
| [^165] | [Rhetorical Questions in LLM Representations: A Linear Probing Study](https://arxiv.org/abs/2604.14128) | 该研究通过线性探针发现大语言模型在表示空间中能够早期且稳定地编码反问句信号，其跨数据集可迁移性虽然存在，但并不意味着模型内部存在统一的共享表示。 |
| [^166] | [Is a Picture Worth a Thousand Words? Adaptive Multimodal Fact-Checking with Visual Evidence Necessity](https://arxiv.org/abs/2604.04692) | 该论文挑战了“视觉证据总能提升事实核查准确性”的普遍假设，提出通过两个协同的视觉-语言模型自适应判断是否需要视觉证据的模块化框架AMuFC，在多个数据集上实现了更有效的事实核查。 |
| [^167] | [Many Preferences, Few Policies: Compact Portfolios for Multi-Objective LLM Alignment](https://arxiv.org/abs/2604.04144) | 该论文提出 PALM 算法，通过结构化权重向量网格、惰性搜索与剪枝构建一个小型 LLM 策略组合，可证明地覆盖所有奖励权重下的近优对齐策略，以低成本实现多目标 LLM 对齐的个性化与部署。 |
| [^168] | [GISTBench: Evaluating LLM User Understanding via Evidence-Based Interest Verification](https://arxiv.org/abs/2603.29112) | 该论文提出GISTBench基准，通过兴趣扎根度（IG）和兴趣特异性（IS）两个新指标，评估大语言模型从推荐系统交互历史中提取和验证用户兴趣的能力，突破了传统推荐系统基准仅关注物品预测准确率的局限。 |
| [^169] | [On the Tip of the Tongue: Why LLMs Hallucinate Answers They Can Decode](https://arxiv.org/abs/2603.13911) | 该论文提出在首个答案标记处区分“读取”与“写出”的新框架，揭示大语言模型产生幻觉的关键原因并非正确答案无法从中间状态解码，而是最终读出时的“选择边际”不足，使更强的竞争标记压制了正确答案。 |
| [^170] | [Can We Trust LLMs on Memristors? Diving into Reasoning Ability under Non-Ideality](https://arxiv.org/abs/2603.13725) | 该论文系统研究了忆阻器存内计算架构中的非理想性对大语言模型推理能力的影响，并总结出三种免训练策略（浅层冗余、思考模式与上下文学习）在不同噪声水平下的适用准则。 |
| [^171] | [SiDiaC-v.2.0: Sinhala Diachronic Corpus Version 2.0](https://arxiv.org/abs/2603.10861) | 本文构建了迄今最大的僧伽罗语历时语料库SiDiaC-v.2.0，包含185部文学作品共22.9万词，时间跨度从公元5世纪至20世纪，并提供了按写作日期标注的子集，为僧伽罗语的历史语言学研究提供了重要资源。 |
| [^172] | [Code2Math: Can Your Code Agent Evolve Math Problems Through Exploration?](https://arxiv.org/abs/2603.03202) | 提出多智能体框架Code2Math，利用代码智能体通过探索将现有数学问题自主演化为结构不同、更具挑战性且可解的新问题，以缓解高质量数学问题稀缺的瓶颈。 |
| [^173] | [Sensory-Aware Sequential Recommendation via Review-Distilled Representations](https://arxiv.org/abs/2603.02709) | 该论文提出ASER离线流水线，通过微调大语言模型从评论中提取有据可查的感官属性并蒸馏为冻结的五维感官库，再以轻量级关系度量增强序列推荐，同时保持预训练主干不变。 |
| [^174] | [RT-SFT: Text Style Transfer from Non-Parallel Corpora by Roundtrip Translation](https://arxiv.org/abs/2602.15013) | 本文提出利用在通用语料上训练的神经机器翻译系统，通过枢轴语言进行往返翻译来剥离风格信号，将归一化从推理时的小型补丁转变为大规模数据生成工具，从而实现从非平行语料库的文本风格迁移。 |
| [^175] | [AstroAgentBench: Evaluating Agentic Planning on Space Mission Planning Tasks](https://arxiv.org/abs/2601.11354) | 本文提出AstroAgentBench——一个涵盖调度、观测规划、星座设计和中继支持等七大任务族的可执行太空任务规划基准，通过外部验证器评估智能体生成的规划产物，发现最强LLM智能体系统在部分任务上可接近或超越求解器参考水平，而较弱系统则难以产出高价值的有效规划。 |
| [^176] | [Morality is Contextual: Learning Interpretable Moral Contexts from Human Data with Probabilistic Clustering and Large Language Models](https://arxiv.org/abs/2512.21439) | 提出了COMETH框架，将概率情境学习与大语言模型语义抽象及人类道德判断数据相结合，从数据中学习可解释的道德情境，证明道德评价是高度情境化的。 |
| [^177] | [A Unified BERT-CNN-BiLSTM Framework for Simultaneous Headline Classification and Sentiment Analysis of Bangla News](https://arxiv.org/abs/2511.18618) | 本文提出了一个统一的BERT-CNN-BiLSTM混合迁移学习框架，首次实现了孟加拉语新闻标题分类与情感分析的同步处理。 |
| [^178] | [Last Layer Logits to Logic: Empowering LLMs with Logic-Consistent Structured Knowledge Reasoning](https://arxiv.org/abs/2511.07910) | 该论文针对大语言模型在结构化知识推理中的“逻辑漂移”问题，提出从模型最后一层的Logits入手进行干预，而非仅依赖提示层面的工作流引导，从而实现逻辑一致的结构化知识推理。 |
| [^179] | [WAON: A Large-Scale Japanese Image-Text Dataset for Cultural Adaptation in Contrastive Vision-Language Models](https://arxiv.org/abs/2510.22276) | 本文发布了目前最大的公开原生日文图文数据集WAON（约1.55亿样本）及日本文化基准WAON-Bench（374个类别），实验证明使用本地来源数据进行微调能将特定文化理解能力提升至超越仅靠全球预训练的水平。 |
| [^180] | [Enrich-on-Graph: Query-Graph Alignment for Complex Reasoning with LLM Enriching](https://arxiv.org/abs/2509.20810) | 提出Enrich-on-Graph（EoG）框架，利用大语言模型的先验知识增强知识图谱，弥合结构化图谱与非结构化查询之间的语义鸿沟，实现高效、低成本且可扩展的知识图谱问答复杂推理。 |
| [^181] | [The Percept-V Challenge: Can Multimodal LLMs Crack Simple Perception Problems?](https://arxiv.org/abs/2508.21143) | 该论文提出了 Percept-V 数据集，包含 6000 张程序生成的无污染图像、分为 30 个基于 TVPS-4 框架的感知领域，用于系统评估多模态大语言模型在简单视觉感知任务上的能力。 |
| [^182] | [BioMol-MQA: A Multi-Modal Question Answering Dataset For LLM Reasoning Over Bio-Molecular Interactions](https://arxiv.org/abs/2506.05766) | 该论文提出了BioMol-MQA——一个针对多重用药场景的新型多模态问答数据集，通过融合文本与分子结构的多模态知识图谱及具有挑战性的问题，用于评估大语言模型在多模态知识检索与推理方面的能力。 |
| [^183] | [Evaluating the Retrieval Robustness of Large Language Models](https://arxiv.org/abs/2505.21870) | 该研究建立了一个包含1,891个样本的基准和三个鲁棒性指标，系统评估了11个大型语言模型在检索增强生成场景中的检索鲁棒性，重点考察RAG是否总是优于非RAG、更多检索文档是否总是有益以及文档顺序对结果的影响。 |
| [^184] | [Automatic register identification for the open web using multilingual deep learning](https://arxiv.org/abs/2406.19892) | 本文构建了覆盖16种语言、25种语域的Multilingual CORE语料库，利用多标签深度学习实现开放网络语域自动识别，并发现性能瓶颈源于网络语域固有的模糊性而非模型局限。 |
| [^185] | [ETHER: Aligning Emergent Communication for Hindsight Experience Replay.](http://arxiv.org/abs/2307.15494) | 本文提出了ETHER，通过对齐紧急沟通来解决回顾性经验重演中的问题，克服了先前架构依赖预设函数的限制，并提高了数据效率和性能。 |

# 详细

[^1]: 能下棋并解释其走法的语言模型

    Language Models that Play Chess and Explain Their Moves

    [https://arxiv.org/abs/2610.03695](https://arxiv.org/abs/2610.03695)

    提出了Queen——一个40亿参数的国际象棋语言模型，通过将专家级棋力编码器与指令微调语言模型以交叉注意力方式结合，并利用贝尔曼更新的自然语言类比进行迭代蒸馏，使其在达到特级大师棋力的同时能够解释自己的走法。

    

    现代国际象棋引擎是沉默的专家：它们以超越人类的水平对弈，但不会为自己的走法提供解释。另一方面，语言模型（LM）可以生成听起来合理的解释，但其薄弱的棋力限制了其解释的实用价值。我们提出了Queen，一个40亿参数的国际象棋语言模型，它能够解释自己的走法和策略，同时达到典型特级大师的水平。我们的新颖框架通过互补组件实现特定领域的推理：编码器-解码器架构和迭代蒸馏算法。该架构通过交叉注意力将一个沉默的专家级国际象棋编码器与一个经过指令微调的语言模型相融合，并通过问答式课程训练，从编码器的表示中提取国际象棋概念。在这一领域适应模型的基础上，我们使用贝尔曼更新的自然语言类比来迭代改进其解释。

    arXiv:2610.03695v1 Announce Type: new  Abstract: Modern chess engines are silent experts: they play at a superhuman level, but do not offer explanations for their play. On the other hand, language models (LMs) can generate plausible-sounding explanations, but their weak playing strength limits the utility of their explanations. We introduce Queen, a 4B-parameter chess-language model that can explain its moves and plans while playing at the level of a typical Grandmaster. Our novel framework enables domain-specific reasoning through complementary components: an encoder-decoder architecture and an iterative distillation algorithm. This architecture integrates a silent expert chess encoder with an instruction-tuned LM through cross-attention, which we train via a question-answering curriculum to extract chess concepts from the encoder's representations. Building on this domain-adapted model, we iteratively improve its explanations with a natural-language analog of the Bellman update: the 
    
[^2]: FrugalEvo：迈向成本感知的LLM引导程序进化

    FrugalEvo: Towards Cost-Aware LLM-Guided Program Evolution

    [https://arxiv.org/abs/2610.03675](https://arxiv.org/abs/2610.03675)

    该论文提出成本感知的LLM进化框架FrugalEvo，让更强的LLM探索解法策略、更廉价的LLM负责代码实现与迭代优化，并通过前缀共享提升缓存复用，同时引入BA-AUC指标来衡量单位成本下的优化收益。

    

    arXiv:2610.03675v1 公告类型：cross 摘要：以AlphaEvolve为代表的LLM引导的进化方法，已成为解决具有挑战性的计算优化问题（如圆填充问题）的有力工具。然而，先前的工作通常是在固定迭代次数下优化性能提升。我们认为，实际的优化应当最大化单位成本的收益。为此，我们提出了FrugalEvo——一个成本感知的进化框架，其中由一个更强、成本更高的LLM负责探索解决方案策略，由一个更廉价的LLM负责实现这些策略并对生成的代码进行迭代改进。我们还设计了缓存高效的进化过程，通过我们的测试框架和提示词设计，最大化不同进化步骤之间的前缀共享，以提升缓存复用率。为了在固定成本预算内衡量解决方案的质量，我们引入了预算感知曲线下面积（BA-AUC），其定义为在预算范围内、以累计LLM成本为横轴的最优评估分数曲线下的面积。（原文摘要至此处截断）

    arXiv:2610.03675v1 Announce Type: cross  Abstract: LLM-guided evolutionary methods, such as AlphaEvolve, have emerged as powerful approaches for challenging computational optimization problems, such as circle packing. However, prior work typically optimizes performance gain over a fixed number of iterations. We argue that practical optimization should maximize gain per unit cost. To this end, we propose FrugalEvo, a cost-aware evolutionary framework where a stronger, higher-cost LLM explores solution strategies, and a cheaper LLM implements them and iteratively refines the resulting code. We also design a cache-efficient evolution process, where our harness and prompts maximize the sharing of prefixes across different evolution steps, to improve cache reuse. To measure solution quality throughout a fixed cost budget, we introduce Budget-Aware Area Under the Curve (BA-AUC), defined as the area under the best-so-far evaluation score curve over cumulative LLM cost, up to the budget. Acros
    
[^3]: Pivot-SD：面向掩码扩散语言模型的高效自蒸馏

    Pivot-SD: Efficient Self-Distillation for Masked Diffusion Language Models

    [https://arxiv.org/abs/2610.03665](https://arxiv.org/abs/2610.03665)

    Pivot-SD提出了一种高效的自蒸馏框架，通过信息增益指标识别去噪过程中真正塑造响应的少数关键决策（枢轴），并仅对这些高影响token进行针对性监督训练，从而解决了掩码扩散语言模型后训练中的信用分配问题。

    

    掩码扩散语言模型为复杂推理提供了一种有前景的、可与自回归模型并行竞争的替代方案。然而，它们面临一个独特的信用分配挑战：去噪过程中的少数关键决策会急剧降低剩余掩码位置的不确定性，并塑造了响应的大部分内容。大多数针对dLMs的后训练方法并未利用这一信号来决定训练哪些token：它们通常在最终文本上进行训练，或将奖励分配给整个去噪步骤，而不是挑选出塑造响应的那些单个关键决策。我们提出了Pivot-SD，这是一种高效的离线自蒸馏框架，仅对这些高影响的关键决策（枢轴，pivots）进行监督。Pivot-SD使用信息增益指标来选择枢轴，该指标衡量对剩余掩码位置不确定性的降低程度。来自成功轨迹的枢轴使用交叉熵进行训练，而来自失败轨迹的枢轴则使用tar（原文摘要在此处截断）

    arXiv:2610.03665v1 Announce Type: cross  Abstract: Masked diffusion language models (dLMs) offer a promising parallel alternative to autoregressive models for complex reasoning. However, they face a distinct credit-assignment challenge, since a few commitments during denoising sharply reduce the uncertainty over the remaining masked positions and shape much of the response. Most post-training recipes for dLMs do not use this signal to decide which tokens to train on: they typically train on the final text or assign rewards to whole denoising steps, rather than selecting the individual commitments that shape the response. We introduce Pivot-SD, an efficient offline self-distillation framework that supervises only these high-impact commitments (pivots). Pivot-SD selects pivots using an information-gain metric measuring uncertainty reduction over the remaining masked positions. Pivots from successful trajectories are trained with cross-entropy, and pivots from failed trajectories with tar
    
[^4]: 世界嵌入基准

    World Embedding Benchmark

    [https://arxiv.org/abs/2610.03632](https://arxiv.org/abs/2610.03632)

    提出世界嵌入基准，包含8000个涵盖流体力学、固体力学、动力学和光学电磁学的受控仿真案例，用于评估视频嵌入对物理信息的编码能力，发现现有全模态嵌入模型的跨模态物理对齐能力较弱，但轻量级探针可从冻结的视频嵌入中恢复有用的物理信息。

    

    物理保真度在世界模型和视频生成领域受到越来越多的关注，然而视频表示如何编码物理信息仍然鲜为人知。我们提出了世界嵌入基准（World Embedding Benchmark），该基准包含来自80个家族的8,000个受控仿真案例，涵盖流体力学、固体力学、动力学以及光学与电磁学领域。每个案例将渲染的视频与基于仿真导出的物理标注配对，支持三个互补的任务：文本-视频检索、物理属性回归以及多选视频-描述对分类。我们利用这些任务来区分跨模态物理对齐与定量物理信息的可恢复性。对预训练全模态嵌入模型的评估显示，其检索能力较弱，且家族内配对分类表现接近随机水平，而轻量级探针能够从冻结的视频嵌入中恢复有用的物理信息。持续对比……

    arXiv:2610.03632v1 Announce Type: cross  Abstract: Physical fidelity has received increasing attention in world models and video generation, yet how video representations encode physical information remains less understood. We introduce the World Embedding Benchmark, comprising 8,000 controlled simulation cases from 80 families spanning fluid mechanics, solid mechanics, dynamics, and optics & electromagnetism. Each case pairs a rendered video with simulation-derived physical annotations, supporting three complementary tasks: text-video retrieval, physical-property regression, and multiple-choice video-description pair classification. We use these tasks to distinguish cross-modal physical alignment from the recoverability of quantitative physical information. Evaluated pre-trained omnimodal embedding models show weak retrieval and near-chance within-family pair classification, while lightweight probes recover useful physical information from frozen video embeddings. Continual contrastiv
    
[^5]: FALCON：一种模型与数据集无关的NL2SQL对合成数据生成框架

    FALCON: A Model and Dataset Agnostic Framework for Synthetic Data Generation for NL2SQL Pairs

    [https://arxiv.org/abs/2610.03625](https://arxiv.org/abs/2610.03625)

    FALCON提出了一种模型与数据集无关的框架，通过保留字SQL种子、基于人设的提示生成和基于对齐的过滤，以低成本利用紧凑开源模型生成真实、具有歧义感知能力且结构复杂的NL-to-SQL合成数据。

    

    关系数据库是部署最广泛的结构化知识形式之一，通过自然语言访问这些数据库需要将语言落实到模式实体和关系上，同时还要处理人们在表述请求时固有的歧义性。现有的合成NL-to-SQL数据生成方法大多忽略了这种歧义性，生成过于简化的查询，无法让模型为现实世界结构化知识访问的复杂性做好准备。我们提出了FALCON，这是一个能够生成真实的、具有歧义感知能力的NL-to-SQL数据的框架，其生成的数据复杂度可与具有挑战性的真实世界基准相匹配，并且通过使用紧凑的开源模型以低成本实现。我们的方法结合了保留字SQL种子和基于人设的提示来生成结构复杂的查询，同时基于对齐的过滤通过区分真正错误的样本与复杂但有效的查询来保持数据难度。人工评估证实了生成数据的一致高质量。

    arXiv:2610.03625v1 Announce Type: new  Abstract: Relational databases are among the most widely deployed forms of structured knowledge, and natural language access to them requires grounding language onto schema entities and relations while handling the ambiguity inherent in how people phrase requests. Existing synthetic NL-to-SQL data generation methods largely ignore this ambiguity and produce oversimplified queries that fail to prepare models for the complexity of real-world structured knowledge access. We present FALCON, a framework that generates realistic, ambiguity-aware NL-to-SQL data matching the complexity of challenging real-world benchmarks, at low cost using compact open models. Our approach combines reserved-word SQL seeding and persona-based prompting to generate structurally complex queries, while alignment-based filtering preserves difficulty by distinguishing genuinely incorrect examples from complex but valid queries. Human evaluation confirms consistent high quality
    
[^6]: Writerslogic团队参加CLEF 2026 SimpleText赛道：多候选LLM文本简化与堆叠式复杂度识别

    Writerslogic at the CLEF 2026 SimpleText Track: Multi-Candidate LLM Simplification and Stacked Complexity Spotting

    [https://arxiv.org/abs/2610.03567](https://arxiv.org/abs/2610.03567)

    Writerslogic团队在CLEF 2026 SimpleText任务中提出基于GPT-4o-mini的多候选简化生成与无参考评分选择流水线，获得句子级简化第一名，并将复杂度识别转化为自然语言推理任务，用35万标注对微调DeBERTa-v3-large模型完成幻觉检测。

    

    本文介绍了Writerslogic团队参加CLEF 2026 SimpleText共享任务的情况，涵盖任务1（文本简化）和任务2（复杂度识别）。在任务1中，我们开发了一个基于GPT-4o-mini的多候选生成流水线，在不同温度参数下为每个句子生成五个简化候选，然后使用一种无参考评分启发式方法选出最佳候选，该方法综合考量压缩程度、源词保留率、Cochrane简明语言摘要词汇的使用以及词汇简单性。在任务1.1（句子级简化）中，我们基于Claude Sonnet 4的提交取得SARI 47.43和BLEU 14.21的成绩，成为排名最高的句子级系统（在任务1综合排行榜上位列第三，仅次于两个文档级提交系统）。在任务2中，我们在35万个带标注的（源文本，句子）对上微调了一个DeBERTa-v3-large自然语言推理（NLI）模型，将幻觉检测问题构建为自然语言推理任务，该模型会读取最相关的源文本……

    arXiv:2610.03567v1 Announce Type: new  Abstract: We describe the Writerslogic team's participation in the CLEF 2026 SimpleText shared task, addressing Task 1 (text simplification) and Task 2 (complexity spotting). For Task 1, we develop a multi-candidate generation pipeline using GPT-4o-mini that produces five simplification candidates per sentence at varying temperatures, then selects the best candidate using a reference-free scoring heuristic that rewards compression, source word retention, Cochrane Plain Language Summary vocabulary usage, and lexical simplicity. On Task 1.1 (sentence-level simplification), our Claude Sonnet 4 submission achieves SARI 47.43 and BLEU 14.21, the top-ranked sentence-level system (3rd on the combined Task 1 leaderboard, behind two document-level submissions). For Task 2, we fine-tune a DeBERTa-v3-large NLI model on 350K labeled (source, sentence) pairs, framing hallucination detection as natural language inference. The model reads the most relevant sourc
    
[^7]: Writerslogic团队在PAN 2026竞赛中的表现：面向领域偏移下鲁棒检测的“过程优于内容”方法

    Writerslogic at PAN 2026: Process over Content for Robust Detection under Domain Shift

    [https://arxiv.org/abs/2610.03565](https://arxiv.org/abs/2610.03565)

    本文提出“特征在分布偏移下的鲁棒性取决于训练与测试分布的支撑重叠度而非训练集效应量”的分析框架，并据此设计系统在CLEF 2026 PAN推理轨迹检测任务的源检测中获得第一名。

    

    我们描述了Writerslogic团队参加CLEF 2026 PAN三个共享任务（推理轨迹检测、Voight-Kampff生成式AI检测、多作者写作风格分析）的系统，这些系统由一个共同的分析框架统一起来：分布偏移下的特征鲁棒性由训练分布与测试分布之间的支撑重叠程度决定，而非训练集的效应量。由此得出一个分类体系（领域锚定、领域可迁移、领域不变），解释了为什么生成器特定特征在领域偏移下会失效，而词汇指纹（罕见词比例、Yule's K、Heaps指数）、压缩度量和字符n-gram得以幸存。在推理轨迹检测任务中，训练数据完全来自数学领域，而84%的测试数据属于未见过的领域，该框架指导系统设计在源检测中获得第一名（通过Opus-Sonnet一致性达到0.85宏F1），在安全分类中获得第三名（通过查询拒绝达到0.66宏F1）

    arXiv:2610.03565v1 Announce Type: new  Abstract: We describe the Writerslogic systems for three PAN at CLEF 2026 shared tasks (Reasoning Trajectory Detection, Voight-Kampff Generative AI Detection, and Multi-Author Writing Style Analysis), unified by a shared analytical framework: feature robustness under distribution shift is governed by support overlap between training and test distributions, not by training-set effect size. This yields a taxonomy (domain-anchored, domain-portable, domain-invariant) that explains why generator-specific features die under domain shift while vocabulary fingerprints (hapax ratio, Yule's K, Heaps' exponent), compression measures, and character n-grams survive. On Reasoning Trajectory Detection, where training was entirely mathematics and 84 percent of test was unseen domains, the framework guided system design to 1st place in source detection (0.85 macro F1 via Opus-Sonnet agreement) and 3rd place in safety classification (0.66 macro F1 via query-refusal
    
[^8]: 零样本作者归属的作者表示策略：基于大语言模型与基于嵌入方法的比较研究

    Author Representation Strategies for Zero-Shot Authorship Attribution: A Comparative Study of LLM-Based and Embedding-Based Approaches

    [https://arxiv.org/abs/2610.03531](https://arxiv.org/abs/2610.03531)

    该研究比较了零样本作者归属中不同作者表示策略的效果，发现仅用标签提示无效，而引入作者特定表示能持续提升性能，并提出了一种结合候选空间缩减与嵌入维度选择的两阶段嵌入归属框架。

    

    作者归属（AA）任务需要捕捉细粒度的文体特征，这使得它在没有任何任务特定监督的零样本（ZS）设置下尤为具有挑战性。在这项工作中，我们通过评估仅使用标签的提示基线以及三种作者表示策略——代表性写作样本、大语言模型生成的描述和风格嵌入（LISA）——来研究作者表示对零样本作者归属的影响。前三种方法使用大语言模型提示来执行归属任务，而基于嵌入的方法则利用风格嵌入和余弦相似度。我们研究了提示设计的影响，并提出了一种两阶段的基于嵌入的归属框架，该框架将候选空间缩减与嵌入维度选择相结合。结果表明，仅使用标签的零样本作者归属是无效的，而引入作者特定的表示能够持续提升归属性能。

    arXiv:2610.03531v1 Announce Type: new  Abstract: Authorship Attribution (AA) requires capturing fine-grained stylistic characteristics, making it particularly challenging in zero-shot (ZS) settings where no task-specific supervision is available. In this work, we investigate the effect of author representations on ZS AA by evaluating a label-only prompting baseline together with three author representation strategies: representative writing samples, LLM-generated descriptions, and style embeddings (LISA). The first three approaches perform attribution using LLM prompting, while the embedding-based approach uses style embeddings with cosine similarity. We investigate the influence of prompt design and propose a two-stage embedding-based attribution framework that combines candidate space reduction with embedding-dimension selection. The results show that label-only ZS AA is ineffective, while incorporating author-specific representations consistently improves attribution performance. Am
    
[^9]: 蒸馏中的散度控制熵

    Divergence controls entropy in distillation

    [https://arxiv.org/abs/2610.03529](https://arxiv.org/abs/2610.03529)

    该论文证明蒸馏目标中散度的选择充当了隐式熵正则化器，控制着学生模型的熵——前向KL散度会使学生熵高于教师，反向KL散度会降低熵，而在线策略蒸馏的低熵来自token级反向KL散度而非采样方式本身。

    

    蒸馏已成为大语言模型训练的核心基础操作，但其性质尚未被充分理解。我们从熵的视角出发，研究学生的熵如何取决于数据以及定义蒸馏目标的散度。我们证明前向KL散度会使学生的熵膨胀到高于教师的水平。由于交叉熵训练是其特例，这给出了一个恒等式，我们在预训练和监督微调中对其进行了定量验证。其他散度则没有这样的保证：反向KL散度会降低熵，直到学生与教师之间的差距过大为止；在两者之间插值时，熵在训练早期平滑变化，但在收敛时发生突变。在线策略蒸馏所具有的较低熵来源于token级的反向KL散度，而非在线策略采样本身。因此，散度充当了一种隐式的熵正则化器，其作用在自……（原文在此处截断）

    arXiv:2610.03529v1 Announce Type: cross  Abstract: Distillation has become a core primitive of large language model training, but its properties are not yet well understood. We take an entropic perspective, studying how the entropy of the student depends on the data and the divergence that define the distillation objective. We prove that forward KL inflates the entropy of the student above that of the teacher. Since cross-entropy training is a special case, this yields an identity that we verify quantitatively in pretraining and supervised finetuning. Other divergences come with no such guarantee: reverse KL deflates entropy until the gap between student and teacher gets too large, and interpolating between the two changes entropy smoothly early in training but abruptly at convergence. The lower entropy of on-policy distillation comes from token-level reverse KL, not from on-policy sampling. The divergence therefore acts as an implicit entropy regularizer, whose role is clearest in sel
    
[^10]: 面向表格到报告生成的可验证原子洞察的结构化组合

    Structured Composition of Verifiable Atomic Insights for Table-to-Report Generation

    [https://arxiv.org/abs/2610.03525](https://arxiv.org/abs/2610.03525)

    提出ComInsight框架，将表格到报告生成中的洞察发现重新建模为可验证原子洞察（最小可执行分析单元）的结构化组合，从而克服现有顺序式数据代理和LLM直接生成方法因探索偏差而遗漏跨表、跨维度证据的问题。

    

    表格到报告生成是指从关系表中自动生成文章级分析报告的任务，是自动化数据科学与决策支持的一项关键能力。其核心挑战在于系统地发现跨表、跨属性、跨分析视角的可验证复合洞察，并将它们组织成连贯、完整、可追溯的证据链。现有方法主要依赖顺序式的、被动响应的数据代理或直接使用大语言模型（LLM）生成，这些方法存在探索偏差问题：早期的局部观察会约束后续行动，导致模型过早地聚焦于局部分析，从而遗漏跨表或跨维度的证据。我们提出了ComInsight，将洞察发现重新表述为原子证据的组合。我们首先将原子洞察定义为符合预定义分析规范的、最小的可执行分析单元（摘要在此处被截断）。

    arXiv:2610.03525v1 Announce Type: new  Abstract: Table-to-report generation refers to the task of automatically generating article-level analyt- ical reports from relational tables and is an essential capability for automated data science and decision support. Its central challenge lies in systematically discovering verifiable com- posite insights across tables, attributes, and analytical perspectives, and organizing them into coherent, complete, and traceable evidence chains. Existing methods primarily rely on sequential, reactive data agents or direct Large Language Model(LLM) generation. They suffer from exploration bias: early local observations constrain subsequent actions, causing models to focus prematurely on local analyzes and miss cross-table or cross-dimensional evidence. We propose ComInsight, which reformulates insight discovery as the composition of atomic evidences. We first define an atomic insight as the smallest executable analytical unit conforming to a predefined an
    
[^11]: 从修复后的推理中学习：根因引导的在策略蒸馏

    Learning from Repaired Reasoning: Root-Cause-Guided On-Policy Distillation

    [https://arxiv.org/abs/2610.03515](https://arxiv.org/abs/2610.03515)

    RC-OPD 通过定位学生自身推理中最早的实质性错误，并将修复后的推理作为在策略蒸馏的监督信号，实现针对错误根因的精准指导而非让学生简单借用正确结论。

    

    在策略自蒸馏（OPSD）将参考解答作为特权后见之明，用以监督学生生成的推理轨迹。然而，基于参考解答的指导可能只解释了一个正确解答，却没有说明学生自身的推理为何失败。这种所提供的指导与所需纠正之间的推理不匹配，会促使学生直接借用正确结论，而使其自身的推理错误未得到解决。此外，在整个轨迹上始终应用同样的后见之明会带来“蒸馏陷阱”的风险，即对有效推理施加的不必要约束会与对实质性错误的纠正相互竞争。为解决这些问题，我们提出根因引导的在策略蒸馏（RC-OPD），它利用对学生自身推理的修复来提供指导，既针对其具体错误进行纠正，又建立在已有的有效进展之上。对于每次失败的尝试，RC-OPD 会定位最早的实质性错误，并制定局部……（原文摘要在此处截断）

    arXiv:2610.03515v1 Announce Type: new  Abstract: On-policy self-distillation (OPSD) uses reference solutions as privileged hindsight to supervise student-generated reasoning trajectories. However, reference-based guidance may explain a correct solution without addressing why the student's own reasoning fails. This reasoning mismatch between the guidance provided and the correction needed can encourage the student to borrow correct conclusions while leaving its reasoning errors unresolved. Moreover, applying the same hindsight throughout the trajectory risks a distillation trap, where unnecessary constraints on valid reasoning compete with correction of substantive errors. To address these issues, we propose Root-Cause-Guided On-Policy Distillation (RC-OPD), which uses repairs of the student's own reasoning to provide guidance that addresses its specific errors while building on valid progress. For each failed attempt, RC-OPD locates the earliest substantive error, develops a local corr
    
[^12]: 面向波斯语医学语言模型中声明级幻觉检测的单次前向不确定性头

    Single-Pass Uncertainty Heads for Claim-Level Hallucination Detection in Persian Medical Language Models

    [https://arxiv.org/abs/2610.03482](https://arxiv.org/abs/2610.03482)

    该论文将 LLM 不确定性头框架首次适配到波斯语医学语言模型中，通过在冻结主干模型的注意力图和 token 概率上训练单次前向传播的轻量级声明级检测头，实现了无需重复采样的低成本声明级幻觉检测，并构建了两个波斯语声明级幻觉数据集。

    

    幻觉检测对医学语言模型尤为重要，但重复采样方法成本高昂，且现有的不确定性头资源无法直接迁移到新的主干模型和语言上。我们将 LLM 不确定性头（LUH）框架适配到基于 Aya-Expanse-8B 的波斯语医学模型上，使用先前开发的 Gaokerena-V 和 Gaokerena-R 作为两个主干模型。我们首先在一份包含 168 道题的伊朗医学入学考试上检验响应变异性，观察到 Gaokerena-V 的五次运行一致性显著低于 Aya-Expanse-8B，而 Gaokerena-R 则与 Aya-Expanse-8B 相当。随后，我们直接以波斯语构建了两个成对的声明级幻觉数据集，每个主干模型各包含 1,600 条响应，并在冻结的主干注意力图和 token 概率上训练轻量级的声明级检测头。在留出的测试集划分上，这些检测头获得了 0.4820 和 0.4652 的 PR-AUC 值。

    arXiv:2610.03482v1 Announce Type: new  Abstract: Hallucination detection is particularly important for medical language models, but repeated-sampling approaches are expensive and existing uncertainty-head resources do not directly transfer to a new backbone and language. We adapt the LLM Uncertainty Head (LUH) framework to Aya-Expanse-8B-based Persian medical models, using Gaokerena-V and Gaokerena-R as two previously developed backbones. We first examine response variability on a 168-question Iranian medical entrance examination and observe substantially lower five-run consistency for Gaokerena-V than for Aya-Expanse-8B, whereas Gaokerena-R is comparable to Aya-Expanse-8B. We then construct two paired claim-level hallucination datasets directly in Persian, containing 1,600 responses for each backbone, and train lightweight claim-level heads on frozen backbone attention maps and token probabilities. On held-out test splits, the heads obtain PR-AUCs of 0.4820 and 0.4652, corresponding t
    
[^13]: 监控器读数接近零并不能作为行为控制的证据

    A Near-Zero Monitor Readout Is Not Evidence of Behavioral Control

    [https://arxiv.org/abs/2610.03458](https://arxiv.org/abs/2610.03458)

    监控器读数接近零并不意味着模型行为真正受到控制——即使在代码生成环境中探针得分和惩罚值都处于极低水平，模型仍可能在训练早期就持续利用漏洞。

    

    通过可验证奖励进行的后训练可能诱发奖励作弊行为，这促使研究者将监控器纳入训练目标之中，而不仅仅将其用于离线审计。我们证明，较低的监控器读数并不能判断此类干预是否真正控制了模型行为。在一个代码生成环境中（其主要可利用的作弊手段在推理轨迹开始时即可获得），我们针对三个通过相同离线门限检测的监控器训练策略：一个域内激活探针和两个基于策略多早确定其最终答案的条件惩罚项。探针得分从第一个被记录的训练步骤起就处于其数值下限，并且在每次前缀训练运行的结束时点，训练得分的中间值均为零。这些读数估计的是不同的量，因此我们不比较它们的量纲；然而，在每个监控器族内部，较低的读数值并不能证明行为控制已实现。在单一固定配置下，经过前缀训练的运行……（原文摘要不完整）

    arXiv:2610.03458v1 Announce Type: new  Abstract: Post-training with verifiable rewards can induce reward hacking, motivating the use of monitors within the training objective rather than solely for offline auditing. We show that a low monitor readout does not identify whether such an intervention controls behavior. In a code-generation environment whose dominant exploit is available at the start of the reasoning trace, we train policies against three monitors that pass the same offline gate: an in-domain activation probe and two penalties conditioned on how early the policy commits to its own final answer. The probe score is at its numerical floor from the first recorded training step, and the trained-score median is zero for every prefix-trained run at the endpoint. These readouts estimate different quantities, and we do not compare their scales; within each monitor family, however, low values do not establish behavioral control. Within one fixed configuration, prefix-trained runs wit
    
[^14]: 通过你训练时的测试：重新评估面向大语言模型智能体的提示注入检测器

    Passing the Test You Trained On: Re-evaluating Prompt-Injection Detectors for LLM Agents

    [https://arxiv.org/abs/2610.03448](https://arxiv.org/abs/2610.03448)

    研究发现提示注入检测器的检测排名在不同基准之间迁移性极差——检测器往往只在自己训练数据所对应的基准上表现良好，而工具输出的假阳性率却能跨智能体基准迁移，因此仅凭公开基准分数无法预测检测器在智能体中的实际表现。

    

    大语言模型智能体越来越多地使用小型提示注入检测器来筛查工具输出，而团队通常依据检测器在公开基准上的得分来进行选择。我们探究的问题是：这些得分能否预测检测器在智能体内部的实际表现？我们在不使用大语言模型的情况下回放两个智能体基准（AgentDojo 和 tau-bench）的真实工具调用，得到构造上即无害的工具输出，并通过差分回放标注被注入的输出，进而在这些输出以及 BIPIA 基准上评估了十五个检测器（包括 Meta 的 Prompt Guard 2）以及两个任务感知的大语言模型裁判。结果显示，检测器的检测排名在不同基准之间的迁移性很差：BIPIA 上表现最好的检测器在 1% 假阳性率下仅能捕获 AgentDojo 中 2% 的注入；而一个能捕获 AgentDojo 中 72% 注入的检测器，在 tau-bench 上仅能捕获 15%。相比之下，工具输出上的假阳性率（范围从零到超过 90%）则能在两个智能体基准之间迁移。在训练数据公开（可获取）的场合……（摘要原文在此处截断）

    arXiv:2610.03448v1 Announce Type: cross  Abstract: LLM agents increasingly screen tool outputs with small prompt-injection detectors, and teams choose among detectors by their scores on public benchmarks. We ask whether those scores predict how a detector behaves inside an agent. We replay the ground-truth tool calls of two agent benchmarks, AgentDojo and tau-bench, without an LLM to obtain tool outputs that are benign by construction, label injected outputs by differential replay, and evaluate fifteen detectors, including Meta's Prompt Guard 2, and two task-aware LLM judges on these outputs and on the BIPIA benchmark. Detection rankings transfer poorly between benchmarks: the best detector on BIPIA catches 2% of AgentDojo injections at a 1% false-positive rate, and a detector that catches 72% of AgentDojo injections catches 15% on tau-bench. False-positive rates on tool outputs, which range from none to over 90%, do transfer between the two agent benchmarks. Where training data is pub
    
[^15]: CLIMB：面向多模态检索增强生成的置信度引导互补证据

    CLIMB: Confidence-Guided Complementary Evidence for Multimodal Retrieval-Augmented Generation

    [https://arxiv.org/abs/2610.03421](https://arxiv.org/abs/2610.03421)

    提出CLIMB，一个无需训练的多模态检索增强生成推理时框架，通过MMR式目标构建紧凑互补证据池，并借助R/E/C批评者与基于证据的置信度估计器来确保答案更新有充分的检索证据支持。

    

    多模态大语言模型（MLLMs）已展现出强大的视觉推理能力，但知识密集型的视觉问答往往需要图像和模型参数化知识之外的额外文本证据。现有的多模态RAG系统通常依赖Top-K检索或重排序，这可能返回冗余的段落，且对答案更新是否被检索证据充分支持缺乏有效控制。我们提出了CLIMB，一个无需训练的多模态RAG推理时框架。CLIMB首先利用一种平衡查询相关性与段落级冗余的MMR风格目标构建紧凑的互补证据池，然后在该固定证据池内执行置信度控制的精炼：一个R/E/C批评者从相关性、证据特异性和跨模态对齐三个维度对段落进行评分，同时一个基于证据的置信度估计器仅当更新答案获得充分证据支持时才予以采纳。

    arXiv:2610.03421v1 Announce Type: new  Abstract: Multimodal large language models (MLLMs) have shown strong visual reasoning abilities, but knowledge-intensive visual question answering often requires external textual evidence beyond the image and the model's parametric knowledge. Existing multimodal RAG systems commonly rely on Top-$K$ retrieval or reranking, which may return redundant passages and provide limited control over whether an answer update is sufficiently supported by the retrieved evidence. We propose \textit{CLIMB}, a training-free inference-time framework for multimodal RAG. CLIMB first constructs a compact complementary evidence pool using an MMR-style objective that balances query relevance and passage-level redundancy. It then performs confidence-controlled refinement within this fixed pool: an R/E/C critic scores passages by relevance, evidence specificity, and cross-modal alignment, while an evidence-grounded confidence estimator accepts an updated answer only when
    
[^16]: 类型化决策模型中候选选项覆盖度的基准测试

    Benchmarking Candidate Coverage in Typed Decision Models

    [https://arxiv.org/abs/2610.03387](https://arxiv.org/abs/2610.03387)

    本文提出了一个成对候选覆盖度基准测试协议，用于评估类型化决策模型识别缺失答案与避免错误拒绝有效候选的能力，发现 Laya 和 Jev 的原生拒绝行为差异显著，而仅使用校准数据的 none 分数阈值可以显著改善两者的检测与误拒平衡。

    

    类型化决策模型会返回选择结果，或针对请求时提供的答案选项返回分布。在选项完整情况下的准确率并不能说明模型是否能识别参考答案缺失的情况，或者是否会避免错误拒绝有效的候选选项。我们提出了一个成对候选覆盖度基准测试协议，并对 Laya 和 Jev 两个模型在 AG News、DBpedia、Emotion 和 TREC 数据集上进行了初步评估。两个模型接收完全相同的冻结文本和请求：300 条校准文本和 589 条测试文本，每个模型产生 23,932 次预测。存在/缺失配对与普通候选数量相匹配，且名称变体保持描述、成员和顺序不变。原生的拒绝行为差异显著：在具有自然名称的五个 TREC 候选选项下，Laya 能检测出 97.2% 的答案缺失案例，但会错误拒绝 69.7% 的存在对照案例；Jev 的这两个比率分别为 24.8% 和 0.0%。仅使用校准数据的 none 分数阈值将这些比率分别改变为 33.9%/3.7% 和 45.0%/1.8%。在 DB

    arXiv:2610.03387v1 Announce Type: new  Abstract: Typed decision models return choices or distributions over answer options supplied at request time. Accuracy with complete options does not establish whether a model recognizes that a reference answer is missing or avoids rejecting valid candidates. We present a paired candidate-coverage benchmark protocol and an initial evaluation of Laya and Jev across AG News, DBpedia, Emotion, and TREC. The models receive identical frozen texts and requests: 300 calibration and 589 test texts yield 23,932 predictions per model. Present/absent pairs match ordinary candidate count, and name variants preserve descriptions, members, and order. Native rejection behavior differs sharply: at five TREC candidates with natural names, Laya detects 97.2% of missing-answer cases but falsely rejects 69.7% of present controls; Jev's rates are 24.8% and 0.0%. Calibration-only none-score thresholds change these rates to 33.9%/3.7% and 45.0%/1.8%, respectively. On DB
    
[^17]: 多语言GSM-Symbolic：什么决定了跨语言的能力迁移？

    Multilingual GSM-Symbolic: What determines capability transfer across languages?

    [https://arxiv.org/abs/2610.03367](https://arxiv.org/abs/2610.03367)

    该论文提出了可扩展的多语言数学数据集Multilingual GSM-Symbolic（涵盖15种语言、3万个题目匹配问答对，通过符号化模板防止过拟合），并量化发现模型规模和语言资源水平是决定跨语言能力迁移的最主要因素。

    

    我们对于一种语言中习得的能力如何迁移到另一种语言、以及哪些因素支配这种迁移仍然知之甚少：现有评估依赖于不可比较且易饱和的数据集，并且很少联合考察迁移的决定因素。识别哪些因素能预测迁移，将使我们能够避免对所有语言对进行穷举评估，并让开发者能够针对限制低资源语言性能的因素进行优化。为了评估跨语言能力迁移，我们引入了Multilingual GSM-Symbolic，这是一个可扩展的多语言数学数据集，包含30,000个题目匹配的问答对，涵盖15种语言。它利用符号化模板防止过拟合并确保泛化，能够从单个样本生成数百万个高质量的变体。使用Multilingual GSM-Symbolic，我们量化了能力的最大决定因素：模型规模（β = 1.77）和语言资源水平（β = 0.77）。

    arXiv:2610.03367v1 Announce Type: new  Abstract: We understand little about how capabilities acquired in one language carry over to another, or what governs this transfer: evaluations rely on incomparable, saturation-prone datasets and rarely examine its determinants jointly. Identifying what predicts transfer would let us avoid exhaustive evaluation across all language pairs and let developers target the factors that limit performance in low-resource languages. To evaluate cross-lingual capability transfer, we introduce Multilingual GSM-Symbolic, an extensible multilingual mathematical dataset covering 30,000 item-matched question-answer pairs and spanning 15 languages. It utilises symbolic templates to prevent overfitting and ensure generalisation by allowing generation of millions of high-quality variations from a single sample. Using Multilingual GSM-Symbolic, we quantify the largest determinants of capability as model size ($\beta = 1.77$), language resource level ($\beta = 0.77$)
    
[^18]: SyntaxBench：大语言模型字符级推理的统计诊断框架

    SyntaxBench: A Statistical Diagnostic Framework for Character-Level Reasoning in Large Language Models

    [https://arxiv.org/abs/2610.03329](https://arxiv.org/abs/2610.03329)

    提出SyntaxBench诊断基准，通过五个核心字符级任务和一个高难度子串提取压力测试，结合Cohen's kappa与McNemar检验等统计方法，系统评估了八个开放权重大语言模型的字符级推理能力。

    

    大语言模型越来越多地被应用于小语法错误也至关重要的场景，然而字符级推理目前主要还是通过孤立的探测任务和聚合准确率来进行评估。我们提出了SyntaxBench，一个面向字符级推理的诊断基准和统计评估框架。它包含五个核心任务：字符计数、字母包含检测、回文检测、编辑距离和最长字符串选择，外加一个更困难的子串提取压力测试index_to_span。五个核心任务使用成对的英文输入与字符长度匹配的随机字符串输入；index_to_span文档共享200-500词的长度区间，但不进行字符长度匹配。全部六个任务均采用零样本、单样本和四样本提示。我们在11种推理模式配置下评估了从2B到32B参数的八个开放权重模型。该框架报告精确匹配与宽松准确率、Cohen's kappa系数、带优势比的配对McNemar检验，

    arXiv:2610.03329v1 Announce Type: cross  Abstract: Large language models are increasingly used where small syntactic errors matter, yet character-level reasoning is still evaluated mostly through isolated probes and aggregate accuracy. We introduce SyntaxBench, a diagnostic benchmark and statistical evaluation framework for character-level reasoning. It contains five core tasks, character counting, letter containment, palindrome detection, edit distance, and longest-string selection, plus index_to_span, a harder substring-extraction stress test. The five core tasks use paired English and character-length-matched random-string inputs. index_to_span documents share a 200-500 word band and are not character-length matched. All six tasks use zero-, one-, and four-shot prompts.   We evaluate eight open-weight models from 2B to 32B parameters across 11 reasoning-mode configurations. The framework reports exact-match and relaxed accuracy, Cohen's kappa, paired McNemar tests with odds ratios, 
    
[^19]: 判决与否？评估用于仇恨言论审核的结构化决策模型的准确性与效率

    To Jev or Not? Evaluating the Accuracy and Efficiency of Structured Decision Models for Hate-Speech Moderation

    [https://arxiv.org/abs/2610.03324](https://arxiv.org/abs/2610.03324)

    论文提出HATEDECIDE评估框架，发现结构化决策模型无需任务特定训练即可在四个仇恨言论数据集中的大多数上与商业大语言模型表现相当，是兼顾准确性、延迟与成本的高效仇恨言论审核方案。

    

    网络内容的庞大规模使仇恨言论审核充满挑战，而大语言模型（LLM）使有害内容的生成与改编变得更加容易。因此，审核工作需要既能保证效率、又能适应不同仇恨言论定义的分类器。近期的结构化决策模型可以接受自然语言形式的判别标准，并在指定答案中进行选择，这引发了一个问题：它们能否在不进行任务特定训练的情况下满足这些要求。我们提出了HATEDECIDE，在四个仇恨言论数据集上对六种决策模型配置进行评估，并与专门的审核模型、零样本模型、商业模型及有监督基线进行比较。我们研究了提供数据集的定义、或将其分解为多个问题，是否能提升分类效果，并测量了这些方法的延迟与成本。我们发现，商业大语言模型仅在一个数据集上显著优于所有决策模型。提供定义会改变上……

    arXiv:2610.03324v1 Announce Type: new  Abstract: The scale of online content makes hate-speech moderation challenging, while Large Language Models (LLMs) enable harmful material to be produced and adapted more easily. Moderation therefore requires efficient classifiers that can accommodate different definitions of hate speech. Recent structured decision models accept natural-language criteria and select among specified answers, raising the question of whether they can meet these requirements without task-specific training. We present HATEDECIDE, an evaluation of six decision-model configurations on four hate-speech datasets against specialized moderation, zero-shot, commercial, and supervised baselines. We examine whether supplying a dataset's definition, or decomposing it into multiple questions, improves classification, and we measure their latency and cost. We find that commercial LLMs significantly outperform all decision models on only one dataset. Supplying definitions changes up
    
[^20]: Shrome团队在Touché竞赛中的方案：软投票集成与反因果增强用于因果关系抽取

    Shrome at Touch\'e: Soft-Vote Ensembling and Counter-Causal Augmentation for Causality Extraction

    [https://arxiv.org/abs/2610.03268](https://arxiv.org/abs/2610.03268)

    该论文提出Shrome系统，为Touché 2026因果关系抽取的三个子任务各构建一个模型，核心创新在于通过解码前对词元级分数取平均的软投票方式集成三个RoBERTa-large BILOU+CRF标注器进行因果片段抽取，并利用跨任务规则和反因果增强来正确识别表面因果但语义上否认因果的句子。

    

    Touché 2026将因果关系抽取任务扩展到了反因果声明：这类新闻句子在表面形式上看起来是因果性的，但其含义却否认了这种因果关系，例如“人们错误地认为X导致了Y"。一个依赖"caused"（导致）或"led to"（引起）等表面线索的系统会错误地将这类句子判定为因果句，并赋予其错误的极性。在反因果新闻语料库（Countercausal News Corpus, CCNC）上，该任务包含三个子任务：判断句子是否为因果句（检测）、定位其中的原因和结果片段（抽取），以及将其极性标注为支持因果、反因果或无因果。我们为每个子任务分别构建一个模型。检测采用微调分类器，并结合一条跨任务规则，即利用抽取出的片段来去除误报。对于抽取任务，我们通过在解码前对三个RoBERTa-large BILOU+CRF标注器的词元级分数取平均来集成它们，而不是对各标注器生成的片段进行投票。对于极性任务，在有标注的反因果（原文摘要在此处截断）

    arXiv:2610.03268v1 Announce Type: new  Abstract: Touch\'e 2026 extends causality extraction to counter-causal claims: news sentences whose surface form appears causal but whose meaning denies the causation, as in "It is falsely believed that X caused Y." A system that relies on surface cues such as "caused" or "led to" will accept such a sentence as causal and give it the wrong polarity. On the Countercausal News Corpus (CCNC), the task has three subtasks: deciding whether a sentence is causal (detection), locating its cause and effect spans (extraction), and labeling its polarity as procausal, counter-causal, or uncausal. We build one model per subtask. Detection is a fine-tuned classifier with a single cross-task rule that uses the extracted spans to remove false positives. For extraction, we ensemble three RoBERTa-large BILOU+CRF taggers by averaging their token-level scores before decoding, rather than voting on the spans each tagger produces. For polarity, where labeled counter-ca
    
[^21]: 通过模型路由与协作实现集体偏见缓解

    Collective Bias Mitigation via Model Routing and Collaboration

    [https://arxiv.org/abs/2610.03240](https://arxiv.org/abs/2610.03240)

    本文提出集体偏见缓解（CBM）框架，通过学习细粒度模型行为并促进多个大语言模型之间的路由与协作、知识共享，首次系统性地探索如何选择和组织不同 LLM 以产生更公平的回答，显著优于单一模型基线。

    

    大型语言模型（LLM）日益被部署于公共卫生、金融和治理等领域，这要求模型既具备准确性，又符合社会价值对齐。尽管近年来取得了诸多进展，LLM 往往会延续甚至放大其训练数据中蕴含的偏见，对公平性构成挑战。虽然“自我去偏”方法鼓励 LLM 识别并纠正自身的偏见，但仅依赖单一模型的内在知识可能不足以应对根深蒂固的刻板印象。为解决这一局限，我们提出了集体偏见缓解框架，该框架通过学习细粒度的模型行为并促进多个不同 LLM 之间的知识共享来缓解偏见。这项工作首次系统性地探索了如何有效选择和组织不同的 LLM，以培养更公平的 LLM 回答。实验表明，CBM 显著优于独立基线方法（例如，在 top-7 设置下，委员会机制将年龄偏见得分从 0.25……（原文摘要至此截断）。

    arXiv:2610.03240v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly deployed in public health, finance, and governance, requiring both accuracy and societal value alignment. Despite recent advances, LLMs often perpetuate or amplify bias embedded in their training data, posing challenges to fairness. While self-debiasing encourages an LLM to identify and correct its own biases, relying on a single model's intrinsic knowledge may be insufficient to address deeply ingrained stereotypes. To address this limitation, we introduce Collective Bias Mitigation (CBM), a framework that alleviates bias by learning fine-grained model behavior and fostering knowledge sharing among diverse LLMs. This work is the first to systematically explore the effective selection and organization of distinct LLMs to cultivate fairer LLM responses. Experiments show CBM substantially outperforms standalone baselines (e.g., in the top-7 setting, Committee lowers the age bias score from 0.25
    
[^22]: AdaStep：面向智能体强化学习的自适应步骤信用加权方法

    AdaStep: Adaptive Step Credit Weighting for Agentic Reinforcement Learning

    [https://arxiv.org/abs/2610.03223](https://arxiv.org/abs/2610.03223)

    提出AdaStep方法，将步骤信用加权建模为均方误差估计问题并推导出最优逐状态收缩系数，从而在稀疏奖励下可靠地融合步骤级与轨迹级监督信号，提升长程LLM智能体的强化学习训练效果。

    

    长程LLM智能体通常使用稀疏的结果奖励进行训练，这使得轨迹级目标过于粗糙，无法区分单个决策的贡献。步骤级信用分配提供了更细粒度的监督，但其估计可能不可靠，因为观测到的回报还依赖于后续动作、环境转移和轨迹长度。我们提出AdaStep，一种自适应步骤信用加权方法，用于控制每个由组比较导出的局部优势对轨迹级信号的修改强度。我们将该加权建模为对潜在步骤优势的均方误差估计问题，并在显式的条件采样假设下，推导出最优的逐状态收缩系数。该系数具有信号-总方差的解释：当回报变化可归因于所选动作时保留局部信用，当变化由其他因素主导时则将其抑制。

    arXiv:2610.03223v1 Announce Type: cross  Abstract: Long-horizon LLM agents are typically trained with sparse outcome rewards, making trajectory-level objectives too coarse to distinguish the contribution of individual decisions. Step-level credit assignment provides finer-grained supervision, but its estimates can be unreliable because observed returns also depend on subsequent actions, environment transitions, and trajectory length. We propose AdaStep, an Adaptive Step-credit weighting method that controls how strongly each group-derived local advantage modifies the trajectory-level signal. We formulate this weighting as a mean-squared-error estimation problem for the latent step advantage and, under an explicit conditional sampling assumption, derive an optimal per-state shrinkage coefficient. The coefficient admits a signal-to-total-variance interpretation: it preserves local credit when return variation is attributable to the selected action and suppresses it when variation is domi
    
[^23]: StanceEval 2026：第二届立场检测共享任务

    StanceEval 2026: The Second Stance Detection Shared Task

    [https://arxiv.org/abs/2610.03215](https://arxiv.org/abs/2610.03215)

    StanceEval 2026 作为阿拉伯语社交媒体立场检测的第二届共享任务，通过主题相关跨目标迁移与完全未见领域迁移两条赛道，系统评估了来自12个国家80个注册团队的跨目标泛化能力。

    

    StanceEval 2026 是 StanceEval 系列共享任务的第二届，聚焦于阿拉伯语社交媒体文本中的立场检测。立场检测旨在识别作者对给定话题所持的立场。给定一条推文和一个目标，参赛系统需要判断作者的立场是支持、反对还是无立场。本届任务聚焦于跨目标泛化，设置了两个不同的评估赛道：赛道1评估主题相关的跨目标迁移（在“女性驾驶”上测试，该目标与训练数据中的“女性赋权”主题相关）；赛道2评估向完全未见目标的跨领域迁移（电动汽车和三学期制）。该共享任务吸引了来自12个国家的80个注册团队。在评估阶段，30个不同的团队提交了结果，经过验证筛选后，赛道1有21个团队正式排名，赛道2有13个团队正式排名，20个团队提交了系统描述论文。参赛团队采用……

    arXiv:2610.03215v1 Announce Type: new  Abstract: StanceEval 2026 is the second edition of the StanceEval shared task series on stance detection in Arabic social media text. Stance detection aims to identify a writer's stance toward a given topic. Given a tweet and a target, participating systems must determine whether the writer's stance is Favor, Against, or None. This edition focuses on cross-target generalization across two distinct evaluation tracks: Track 1 evaluates thematically related cross-target transfer (testing on Women Driving, related to Women Empowerment from training data), while Track 2 evaluates cross-domain transfer to completely unseen targets (E-Cars and Trimester System). The shared task attracted 80 registered teams from 12 countries. During the evaluation phase, 30 unique teams submitted entries, with 21 teams officially ranked in Track 1 and 13 in Track 2 following validation filtering, and 20 teams submitting system-description papers. Participating teams empl
    
[^24]: 预测与修复大语言模型中的合并崩塌

    Predicting and Repairing Merge Collapse in Large Language Models

    [https://arxiv.org/abs/2610.03199](https://arxiv.org/abs/2610.03199)

    该论文提出用基于专家模型任务向量方差的“干扰度”评分，在大语言模型合并前预测是否会崩塌并指导修复，实验表明只有破坏性合并会超过该评分阈值，而现有合并算子常用的符号冲突统计量反而具有反向预测作用。

    

    从同一共享基础模型微调得到的大语言模型可以通过对其任务向量取平均进行合并，但某些合并的结果会远低于基础模型本身，而常见的合并算子在评估之前不会给出任何警告。我们证明了专家模型任务向量的一个统计量既能预测这种崩塌，又能校准其修复方法。取平均所去除的功率等于各专家模型任务向量之间的方差，这也是我们对模型间干扰程度的度量。在一个有效的噪声模型下，合并所注入的扰动随合并系数和干扰程度而增长，从而产生一个合并前的评分。在我们对来自四个模型家族的二十二种合并配置的实验中，只有破坏性合并会超过该评分的阈值。我们还发现专家模型之间符号冲突的统计量——现有合并算子的常见优化目标——反而是反向预测的。随后，我们在评估之前预测了十四次合并的结果，其中十二次……（摘要在此处截断）

    arXiv:2610.03199v1 Announce Type: cross  Abstract: Large language models fine-tuned from a shared base can be merged by averaging their task vectors, but some merges collapse far below the base model, and common merge operators give no warning before evaluation. We show that one statistic of the specialists' task vectors both predicts this collapse and calibrates its repair. The power that averaging removes equals the variance of the task vectors across specialists, our measure of interference. Under a working noise model, the disturbance that a merge injects grows with the merge coefficient and with interference, yielding a pre-merge score. In our experiments on twenty-two merge configurations from four model families, only destructive merges exceed a threshold on this score. We find that statistics of sign conflict between specialists, a common target of existing merge operators, are anti-predictive. We then predicted the outcomes of fourteen merges before evaluating them, and twelve
    
[^25]: KV²：一种自我精炼的KV缓存

    KV$^2$: A Self-Refining KV Cache

    [https://arxiv.org/abs/2610.03198](https://arxiv.org/abs/2610.03198)

    KV²提出了一种基于选择性重建的查询无关KV缓存压缩方法，先用轻量级代理评分器筛选出信息丰富的token，再仅对该子集进行精细重建评分以计算淘汰分数，在极低缓存预算下比次优基线提升超过40个百分点。

    

    键值（KV）缓存的内存占用限制了长上下文模型的实际应用，并且当一个预填充的上下文需要后续服务多个不同查询时，KV缓存成为成本的主要来源。在这种可复用场景中，与查询无关的压缩需要在成本与质量之间权衡：轻量级估计器虽然廉价但准确性较低，而全上下文重建评分虽然更准确，却需要重新处理整个提示词。我们提出了KV²，一种基于选择性重建的与查询无关的KV缓存压缩方法。KV²首先使用轻量级代理评分器识别上下文中信息丰富的token，然后仅对这个子集进行重新处理以计算最终的淘汰分数。在RULER、大海捞针（Needle-in-a-Haystack）和LongBench基准上，KV²相对于基线的优势随着缓存预算收紧而扩大：在RULER 16K上，当KV缓存预算仅为2%时，其平均分数比次优基线提高了超过40个百分点；在LongBench上也取得了（摘要在此处截断）……

    arXiv:2610.03198v1 Announce Type: new  Abstract: The memory footprint of the key-value (KV) cache constrains the practical use of long-context models, and it dominates cost when one prefilled context must later serve many different queries. In this reusable setting, query-agnostic compression trades cost against quality: lightweight estimators are cheap but less accurate, whereas full-context reconstruction scoring is more accurate yet reprocesses the entire prompt. We introduce KV$^2$, a query-agnostic KV-cache compression method based on selective reconstruction. KV$^2$ first uses a lightweight proxy scorer to identify informative in-context tokens, then reprocesses only this subset to compute final eviction scores. On RULER, Needle-in-a-Haystack, and LongBench, KV$^2$'s margin over baselines widens as the budget tightens: on RULER 16K at a 2% KV-cache budget it improves the average score over the next-best baseline by more than 40 percentage points, and on LongBench it attains the h
    
[^26]: 真实场景中的来源偏好：LLM智能体如何按来源偏爱条目，以及如何减少这种偏好

    Source Preference in the Wild: How LLM Agents Favor Items by Source, and How to Reduce It

    [https://arxiv.org/abs/2610.03195](https://arxiv.org/abs/2610.03195)

    该研究发现12个LLM智能体在跨三个领域的端到端搜索中一致地偏好某些来源的条目，这种来源偏好甚至能压倒条目对用户需求的实际满足程度，且仅通过隐藏或替换来源信息即可显著影响选择，从而揭示了智能体决策中的来源偏见及其缓解方法。

    

    当LLM智能体代表用户决定购买哪款产品、预订哪家酒店或引用哪篇论文时，其对来自特定来源（即条目所属的网站或服务）的条目的偏好，会塑造用户最终获得的内容以及哪些来源被选中。我们在三个领域的端到端搜索任务中，对12个智能体模型的来源偏好进行了研究。通过比较处于相同位置、满足相同要求但来自不同来源的条目，我们发现每个模型在每个领域都偏好某些来源而回避另一些来源，且各模型之间对来源的偏好判断大体一致。这种偏好甚至可以超过条目对请求的实际满足程度：当一个条目比另一个条目少满足一项要求，但它来自被偏好的来源而更优条目来自不被偏好的来源时，前者约有三分之二的概率被选中；而在相反情况下，这种情况几乎从未发生。识别条目来源的信息本身就会影响选择：隐藏来源信息会削弱这种偏好，而将条目重新标注为（摘要在此处截断）

    arXiv:2610.03195v1 Announce Type: new  Abstract: As LLM agents decide on users' behalf which product to buy, which hotel to book, or which paper to cite, a preference for items from certain sources (the sites or services they come from) shapes what users receive and which sources are selected. We study source preference in end-to-end search with 12 agent models across three domains. Comparing items from different sources that satisfy the same requirements at the same position, we find that each model prefers some sources and avoids others in every domain, largely agreeing on which. This preference can outweigh how well items satisfy the request: an item satisfying one requirement fewer is selected about two-thirds of the time when it comes from a preferred source and the better one from a dispreferred source, but almost never in the reverse case. The information identifying an item's source affects selection by itself: hiding it weakens the preference, and relabeling an item with a pre
    
[^27]: 直到证据说了算：教会LLM调查员何时结案

    Not Until the Evidence Says So: Teaching LLM Investigators When to Close a Case

    [https://arxiv.org/abs/2610.03190](https://arxiv.org/abs/2610.03190)

    该论文研究了LLM调查员“何时应结案”的证据充分性判断问题，发现未经训练的小模型和前沿模型都普遍夸大证据充分性而过早结案，并提出需对照来源捷径规则来评估结案能力的方法。

    

    事故、缺陷和故障调查以一个普通问答从不面对的决定告终：目前收集到的证据是否足以结案。我们研究LLM调查员如何做出这一决定：它们从案卷中请求证据、修正假设，要么以基于所读内容的结论结案，要么保持案件开放并指出尚缺什么。这种判断能力并非与生俱来：一个未经训练的9B模型在97%的答案中夸大了其证据，而一个能在84%的案例中识别出正确原因的前沿模型仍在91%的情况下夸大证据，并在41个官方结论为“原因未定”的案例中结案了17个。衡量这种判断也并非易事：案件的来源在很大程度上预测了其标签，一个仅读取来源的规则在我们的测试用例上即可达到83.0的平衡准确率。因此，我们通过三项测试来评估结案：结案准确率（对照该规则报告）……

    arXiv:2610.03190v1 Announce Type: cross  Abstract: Accident, defect and outage investigations end with a decision that ordinary question answering never faces: whether the evidence gathered so far is enough to close the case. We study this decision for LLM investigators, which request evidence from a case file, revise their hypotheses, and either close the case with a conclusion grounded in what they read or leave it open and name what is missing. This judgment does not come with capability: an untrained 9B model overstates its evidence in 97% of its answers, and a frontier model that identifies the right cause in 84% of cases still overstates in 91% and closes 17 of the 41 cases whose official finding is "cause undetermined". Measuring it is also non-trivial: the source of a case largely predicts its label, and a rule that reads only the source reaches 83.0 balanced accuracy on our test cases. We therefore evaluate closure with three tests: closure accuracy, reported against this rule
    
[^28]: 在线策略蒸馏中的收益与坍塌：强化学习视角

    Gains and Collapse in On-Policy Distillation:A Reinforcement Learning Perspective

    [https://arxiv.org/abs/2610.03185](https://arxiv.org/abs/2610.03185)

    该论文从强化学习视角揭示了在线策略蒸馏（OPD）既提升性能也可能坍塌的机制——教师模型的隐式奖励在可靠时促进正确响应采样，在偏好与质量错位时引发奖励破解并放大冗长重复生成，且OPD提升性能但不扩展学生模型的能力边界。

    

    在线策略蒸馏（OPD）已成为语言模型后训练的重要方法。然而，尽管OPD能带来性能提升，它也可能坍塌为过度冗长和重复的生成，而这些截然不同结果背后的机制仍鲜为人知。我们从强化学习的视角来解释这些结果：教师模型会隐式地奖励学生模型的行为，即使是教师模型自己很少表现出的行为。从这一视角出发，我们的实验表明，OPD在不扩展学生模型能力的情况下提升了性能。当隐式奖励模型可靠时，OPD使正确响应更容易被采样到；相反，当偏好与质量不一致时，就会发生奖励破解（reward hacking）：隐式奖励模型会放大学生生成的冗长、重复的输出，即使教师模型自身很少生成此类文本。基于这一诊断，我们发现通过在训练中屏蔽不健康的响应等方法可以缓解坍塌问题。

    arXiv:2610.03185v1 Announce Type: new  Abstract: On-policy distillation (OPD) has become an important approach to language model post-training. However, despite its performance gains, OPD can also collapse into excessively long and repetitive generation, and the mechanism underlying these divergent outcomes remains poorly understood. We explain these outcomes through a reinforcement learning perspective: the teacher implicitly rewards student behaviors, even those it rarely exhibits itself. From this perspective, our experiments show that OPD improves performance without expanding the student's capabilities. When the implicit reward model is reliable, OPD makes correct responses easier to sample. In contrast, when the preference misaligns with quality, reward hacking happens: the implicit reward model amplifies overlong, repetitive student rollouts, even though it rarely generates such text itself. Guided by this diagnosis, we find that masking unhealthy responses during training and u
    
[^29]: 后见之明引导的推理链蒸馏用于罕见病诊断

    Hindsight-Guided Rationale Distillation for Rare Disease Diagnosis

    [https://arxiv.org/abs/2610.03176](https://arxiv.org/abs/2610.03176)

    该研究发现，在罕见病诊断的后见之明引导蒸馏中，只有经过污染过滤的蒸馏才能显著超越教师模型，而未过滤的模型会因复制教师推理链中“真实标签是X”的短语（即GT幻觉）而导致准确率严重下降。

    

    我们在ZebraMap上研究了用于罕见病诊断的后见之明引导蒸馏方法：一个1.5B参数的学生模型在一个8B参数教师模型生成的思维链轨迹上进行微调，而该教师模型在生成过程中能够观察到真实诊断结果。所有模型的绝对准确率都较低——该任务在此模型规模下难度很大——但在此上限之内，一个经过污染过滤的变体（StudentF）取得了相比教师模型小幅且具有统计学显著性的准确率优势（p < 0.001），且该优势主要集中在代表性较好的疾病上。未经过滤的学生模型并未显著优于教师模型（p = 0.129），这确立了污染过滤——而非后见之明蒸馏本身——才是性能增益的来源。这一差距可追溯至我们称为“GT幻觉”（真实标签幻觉）的一种伪影。标签可见的生成方式导致教师在推理链中嵌入“真实标签是X”这样的短语；而监督微调会复制这种模式。在推理阶段，未过滤的学生模型在33.9%的情况下会重现该短语，并伴有严重的准确率下降。

    arXiv:2610.03176v1 Announce Type: new  Abstract: We study hindsight-guided distillation for rare disease diagnosis on ZebraMap: a 1.5B student is fine-tuned on chain-of-thought traces from a 8B teacher that observes the ground-truth diagnosis during generation. Absolute accuracy remains low for all models - the task is hard at this scale - but within this ceiling a filtered variant (StudentF) achieves a small, statistically significant accuracy advantage over the teacher (p < 0.001), concentrated in better-represented diseases. The unfiltered student does not significantly outperform the teacher (p = 0.129), establishing that contamination filtering - not hindsight distillation alone - drives the gain. The gap traces to an artifact we term GT hallucination. Label-visible generation causes the teacher to embed "ground truth is X" phrases in its reasoning chain; SFT copies the pattern. At inference, the unfiltered student reproduces the phrase in 33.9% of cases, with severe accuracy degr
    
[^30]: 预测转向向量与适配器权重以实现少样本作者风格迁移

    Predicting Steering Vectors and Adapter Weights for Few-Shot Author-Style Transfer

    [https://arxiv.org/abs/2610.03163](https://arxiv.org/abs/2610.03163)

    该论文针对少样本作者风格迁移任务提出三种方法——对比激活转向、转向向量预测网络和预测LoRA适配器的超网络，并发现超网络在风格模仿与输出质量之间取得了最佳权衡，且能泛化到未见过的作者。

    

    仅凭少量示例将大语言模型适配到某个作者的个人风格具有挑战性，而科学写作更加剧了这一难度：正式的写作规范使得表面文字变化有限，且作者撰写的是自己关注的主题，因此提取出的“风格”很容易与内容纠缠在一起。我们研究了基于每位作者少量示例摘要的风格条件摘要生成任务，并提出三种方法：（1）对比激活转向，（2）预测转向向量的网络，以及（3）预测 LoRA 适配器的超网络。我们发现风格模仿与输出质量之间存在一致的权衡：微调能够获取大部分可用的风格信号，但会牺牲流畅性，而超网络在已见和未见作者上均实现了最佳权衡。我们的转向方法在作者级别运作，将某位作者的摘要与相同内容的风格中性生成结果进行对比，这固定了主题，从而消除了……（原文在此处截断）

    arXiv:2610.03163v1 Announce Type: cross  Abstract: Adapting large language models to an individual author's style from a few examples is challenging, and scientific writing sharpens the difficulty: formal conventions leave little surface variation, and authors write about their own topics, so extracted ``style'' easily entangles with content. We study style-conditioned abstract generation from a few example abstracts per author and propose three methods: (1) contrastive activation steering, (2) a network that predicts steering vectors, and (3) a hypernetwork that predicts LoRA adapters. We find a consistent trade-off between style imitation and output quality: fine-tuning buys most of the available style signal but forfeits fluency, while the hypernetwork achieves the best trade-off on both seen and unseen authors. Our steering operates at author level, contrasting an author's abstracts against style-neutral generations for the same content. This holds topic fixed, removes the need for
    
[^31]: 探究推理语言对齐在单语检索增强生成中的作用

    Investigating the Role of Reasoning-Language Alignment in Monolingual Retrieval-Augmented Generation

    [https://arxiv.org/abs/2610.03136](https://arxiv.org/abs/2610.03136)

    本文构建了基于德语桌游《黑暗之眼》虚构世界的全单语德语RAG问答测试平台，首次系统研究在模型需整合大量目标语言检索证据的场景中，强制推理语言与任务语言对齐对模型准确率的影响。

    

    推理轨迹能够提升大型语言模型（LLM）的表现，但当前模型的推理训练主要以英语为主。已有研究表明，强制模型用另一种语言进行推理会降低准确率，即使推理语言与提示语言一致也是如此——不过这一结论仅在模型对较短提示进行推理的情境下成立。在本文中，我们探究同样的现象是否也适用于检索增强生成（RAG），因为在RAG场景中，模型必须阅读并整合大量以目标语言呈现的检索证据。为研究这一问题，我们基于桌面角色扮演游戏《黑暗之眼》的虚构世界构建了一个完全单语的德语RAG问答测试平台：该领域在德语中文献记载丰富，但对模型而言过于小众、无法凭记忆作答，因此模型必须依赖检索。通过在该测试平台上改变智能体式RAG系统被强制使用的推理语言，我们发现将推理语言与……（原文摘要至此截断）

    arXiv:2610.03136v1 Announce Type: new  Abstract: Reasoning traces improve large language models (LLMs), but current models are trained to reason mostly in English. It has been shown that forcing a model to reason in another language degrades accuracy, even when the reasoning language matches the language of the prompt -- but only for a setting where the model reasons over a short prompt. Here, we ask whether the same holds for retrieval-augmented generation (RAG), where the model must read and integrate a large amount of retrieved evidence in the target language. To study this, we build a fully monolingual German RAG question-answering testbed over the fictional world of the tabletop role-playing game The Dark Eye, a domain that is richly documented in German but too niche for the model to answer from memory, so that it has to rely on retrieval. Varying the forced reasoning language of an agentic RAG system on this testbed, we find that aligning the reasoning language with the language
    
[^32]: 针对模式生物的文献检索基准测试：以盘基网柄菌为案例研究

    Benchmarking Literature Retrieval for a Model Organism: A Dictyostelium Case Study

    [https://arxiv.org/abs/2610.03130](https://arxiv.org/abs/2610.03130)

    本文基于dictyBase构建了首个针对模式生物盘基网柄菌的文献检索基准，并发现交叉编码器重排序与基因注释驱动的查询扩展能够在小众生物医学检索中带来有选择性的性能提升。

    

    生物文献检索系统通常是基于广泛的生物医学语料库和通用搜索任务进行开发和评估的。然而，许多经过人工整理的知识库是在更狭窄的模式生物领域内运作的，在这些领域中文献稀疏且术语具有物种特异性。我们为盘基网柄菌构建了一个来自dictyBase的检索基准，盘基网柄菌是细胞生物学和发育生物学中的一种模式生物。该基准由数据管理员生成的生物学查询组成，这些查询与PubMed收录的文章相关联，并附带结构化的基因注释。利用该基准，我们研究了小众生物检索中的三个因素：交叉编码器重排序、基因感知的查询扩展，以及仅摘要检索与全文检索的对比。我们报告称，重排序和基因感知的查询扩展可以有选择性地提升检索效果：当模型本身非常适合进行生物证据匹配时，重排序最为有用，而经过整理的注释则有助于澄清紧凑的查询。

    arXiv:2610.03130v1 Announce Type: cross  Abstract: Biological literature retrieval systems are often developed and evaluated using broad biomedical corpora and general-purpose search tasks. However, many curated knowledge bases operate in narrower model-organism domains, where the literature is sparse and terminology is organism-specific. We introduce a retrieval benchmark from dictyBase for Dictyostelium, a model organism in cell and developmental biology. The benchmark consists of curator-generated biological queries linked to PubMed-indexed articles, together with structured gene annotations. Using this benchmark, we study three factors in niche biological retrieval: cross-encoder reranking, gene-aware query expansion, and abstract-only versus full-text retrieval. We report that reranking and gene-aware query expansion improve retrieval selectively: reranking is most useful when the model is well suited to biological evidence matching, whereas curated annotations help clarify compac
    
[^33]: 开放权重大语言模型中用于滥用检测的触发-标记机制的脆弱性

    The Fragility of Trigger-Tag Mechanisms for Misuse Detection in Open-Weight LLMs

    [https://arxiv.org/abs/2610.03124](https://arxiv.org/abs/2610.03124)

    该论文首次形式化了开放权重大语言模型中的触发-标记滥用检测机制，将其分为令牌级和权重级两类，并系统研究揭示了此类机制在对抗性攻击下的脆弱性。

    

    开放权重大语言模型可以被下载、修改和部署，超出了开发者的控制范围，这限制了集中式安全保障措施的有效性。因此，近期的研究提出了“触发-标记”机制，当模型在目标条件下被使用时（例如生成钓鱼内容），该机制会产生可检测的信号。尽管这些机制借鉴了已有的技术，但它们在开放权重大语言模型的条件性滥用检测方面的应用相对较新。因此，现有研究工作尚未系统性地研究触发-标记机制在对抗性攻击下的鲁棒性。为了填补这一空白，（i）我们对触发-标记进行了形式化定义，区分了在解码过程中引入水印式信号的“令牌级触发-标记”与学习目标条件与可检测模型行为之间后门式关联的“权重级触发-标记”。此外，（ii）我们……

    arXiv:2610.03124v1 Announce Type: cross  Abstract: Open-weight language models can be downloaded, modified, and deployed beyond their developers' control, limiting the effectiveness of centrally enforced safeguards. Recent work has therefore proposed \emph{trigger-tag} mechanisms that produce a detectable signal when a model is used under a target condition, such as generating phishing contents. Although these mechanisms borrow from established techniques, their use for conditional misuse detection in open-weight LLMs is relatively new. Therefore, existing research works have not systematically studied the robustness of trigger-tag mechanisms under adversarial attacks. To close this gap, (i)~we formalize trigger-tags and distinguish \emph{token-level trigger-tags}, which introduce watermark-inspired signals during decoding, from \emph{weight-level trigger-tags}, which learn backdoor-inspired associations between target conditions and detectable model behavior. Furthermore, (ii)~we intr
    
[^34]: 通过蒸馏生产级LLM信号构建可解释的简历-职位匹配特征表示

    Building Interpretable Feature Representations for Resume-Vacancy Matching by Distilling Production LLM Signals

    [https://arxiv.org/abs/2610.03112](https://arxiv.org/abs/2610.03112)

    该论文提出两阶段方法：先利用根据招聘人员反馈持续优化的基于LLM的标注器生成可解释匹配维度标签，再将其蒸馏为可在CPU上高效运行的LoRA双编码器模型，从而为简历-职位匹配提供八个可解释、可操作的匹配维度。

    

    将候选人匹配到职位是招聘工作的核心，招聘人员需要了解候选人为什么合适，而不仅仅是一个不透明的相关性分数。我们以命名的、可解释的、招聘人员可以据此采取行动的匹配维度形式提供这种证据——在当前部署中共有八个维度。我们提出了一种由两部分组成的方法。第一部分是基于LLM的标注器，其提示词和特征定义在作为早期生产匹配阶段运行期间根据招聘人员的反馈不断完善。在当前架构中，它仅用于离线标注，不会在在线请求中被调用。第二部分是从其蒸馏得到的特征双编码器：一个采用LoRA适配的嵌入主干网络，带有紧凑的按维度预测头部，可在CPU上运行并服务于所有在线请求。两个部分都在持续改进：随着反馈的到来，提示词会不断修订，双编码器也会基于更新的标签进行再训练。该模型在168,772个有标注的职位-简历对（17,921个职位……）上进行训练（原文摘要在此处截断）。

    arXiv:2610.03112v1 Announce Type: new  Abstract: Matching candidates to vacancies is central to recruitment, and a recruiter needs to see why a candidate fits, not only a single opaque relevance score. We provide this evidence as named, interpretable matching dimensions recruiters can act on - eight in our current deployment. We propose a two-part approach. The first is an LLM-based labeler whose prompts and feature definitions were refined from recruiter feedback while it served as an earlier production matching stage. In the current architecture, it is used only for offline labeling and is not called on online requests. The second is a feature bi-encoder distilled from it: a LoRA-adapted embedding backbone with compact per-dimension heads that runs on CPU and serves all online requests. Both parts keep improving: prompts are revised as feedback arrives, and the bi-encoder is retrained on the updated labels. The model is trained on 168,772 labeled vacancy-resume pairs (17,921 vacancie
    
[^35]: 本体不稳定性与统计放大：“人化”LLM生成文本的悖论

    Ontological Instability and Statistical Amplification: The Paradox of "Humanizing" LLM-Generated Text

    [https://arxiv.org/abs/2610.03110](https://arxiv.org/abs/2610.03110)

    研究发现让LLM将机器文本“人化”反而使其更容易被AI检测器识别，因为检测器实际追踪的是统计复杂度而非人类写作特征，这导致其对正式人类写作的误报率高达76.3%。

    

    监督式AI文本检测器在基准测试中报告了较高的准确率，但其决策依据尚不清楚。我们在语义、结构和分词器层面的扰动下分析了一个基于RoBERTa的检测器，使用了M4数据集（N=10,000）和受控生成文本（N=300）。当要求Mistral-7B-Instruct让机器文本听起来更像人类时，动词多样性从0.77上升到0.92，而输出反而变得更容易被检测。检测分数似乎追踪的是统计复杂度，这也导致了对正式人类写作高达76.3%的误报率。作为对照，我们评估了基于事件的潜在空间检测方法：改写改变了其87%的事件序列（Jaccard = 0.067），同形字符改变了70%的提取动词，尽管提取过程仍在运行（Jaccard = 0.30），其最佳领域AUC仅为0.577。RoBERTa的鲁棒性似乎仅特定于其所使用的特征，而结构抽象并未使检测更加鲁棒。

    arXiv:2610.03110v1 Announce Type: new  Abstract: Supervised AI-text detectors report high benchmark accuracy, but it is not clear what their decisions are based on. We analyze a RoBERTa-based detector under semantic, structural, and tokenizer-level perturbations, using the M4 dataset (N = 10,000) and controlled generations (N = 300). When Mistral-7B-Instruct was asked to make machine text sound more human, Verb Diversity rose from 0.77 to 0.92 and the outputs became easier to detect. Detection scores appear to track statistical complexity, which also leads to a 76.3% false-positive rate on formal human writing. As a control, we evaluate event-based Latent Space detection. Paraphrasing changed 87% of its event sequences (Jaccard = 0.067), and homoglyphs altered 70% of the extracted verbs even though extraction still ran (Jaccard = 0.30). Its best domain AUC was 0.577. RoBERTa's robustness seems specific to the features it uses, and structural abstraction did not make detection more robu
    
[^36]: 语言模型边缘注意力空间中的涌现结构

    Emergent Structure in the Marginal Attention Space of Language Models

    [https://arxiv.org/abs/2610.03109](https://arxiv.org/abs/2610.03109)

    该论文提出通过对注意力权重沿查询位置边缘化构建“边缘注意力空间”，发现其按token方向降维时产生跨模型保守的文本内在信号（理论上证得其与输入-输出雅可比相关），而按头方向降维时则形成模型特有的私有结构。

    

    尽管独立训练的语言模型之间的表示相似性已有充分记录，但注意力等内部机制在不同模型之间的行为表现却远未得到充分刻画。受这一研究空白的启发，我们通过对查询位置进行边缘化来考察softmax后注意力权重的结构，将其映射到一个联合的token-头“边缘注意力空间”。通过在60多个不同的LLM上进行评估，我们发现沿token轴和头轴对该空间进行降维时会产生不同的性质。当按token方向降维时，边缘注意力产生了一种在不同模型间稳健保持的文本内在信号。为了解释这一性质，我们从实证上将边缘注意力与网络的输入-输出雅可比矩阵联系起来，并从理论上证明：在平滑性假设下，具有相似下一token分布的模型必然具有相似的输入-输出雅可比统计特性。当按头方向降维时，它形成了一种模型私有的结构（原文在此处截断）。

    arXiv:2610.03109v1 Announce Type: new  Abstract: While representation similarity across independently trained language models is well-documented, how internal mechanics such as attention behave across models remains far less characterized. Inspired by this gap, we examine the structure of post-softmax attention weights by marginalizing over query positions, mapping them into a joint token-head "marginal attention space". Evaluating across 60+ diverse LLMs, we find that different properties emerge when reducing this space along its token and head axes. When reduced token-wise, marginal attention yields a text-intrinsic signal robustly conserved across models. To explain this property, we empirically connect marginal attention to the input-output Jacobian of the network, and prove theoretically that under a smoothness assumption, models with similar next-token distributions are guaranteed to have similar input-output Jacobian statistics. When reduced head-wise, it forms a model-private s
    
[^37]: 询问、放宽还是行动？评估LLM偏好推理中的可操作不确定性

    Ask, Relax, or Act? Evaluating Actionable Indeterminacy in LLM Preference Reasoning

    [https://arxiv.org/abs/2610.03102](https://arxiv.org/abs/2610.03102)

    该论文形式化了“可操作不确定性”概念并构建基于求解器的基准测试，发现LLM难以判断何时无需干预——即使行动已被证明合理，模型仍倾向于不必要的澄清提问或干预。

    

    一个LLM智能体能够识别不确定性，却仍然可能选择错误的下一步：在行动已被证明合理时仍然提问，或者在必须改变约束时才寻求澄清。我们形式化了“可操作不确定性”这一概念：当所有可接受的偏好或目标下都存在共享的可接受行动时应当行动；当每种可能性都可行但没有共享行动时应当澄清；当请求不可行时应提出最小成本的允许约束修复。我们构建了一个基于求解器的基准测试，涵盖物品分配、会议调度、公寓选择和稳定匹配四个场景。匹配对保持相同的来源，同时改变是否需要干预，评估则将决策正确性、匹配对可靠性和完全正确响应区分开来。我们的发现揭示了一个反复出现的困难：模型难以识别何时不需要干预——模型能够识别需要澄清或修复的情况，却仍然会进行不必要的干预。

    arXiv:2610.03102v1 Announce Type: cross  Abstract: An LLM agent can recognize uncertainty yet still choose the wrong next step: asking when action is already justified, or seeking clarification when the constraints must change. We formalize actionable indeterminacy: act when an accepted action is shared across all admissible preferences or objectives, clarify when each possibility is feasible but no action is shared, and propose a minimum-cost permitted constraint repair when the request is infeasible. We construct a solver-grounded benchmark spanning object allocation, meeting scheduling, apartment choice, and stable matching. Matched pairs retain the same source while changing whether intervention is necessary, and evaluation separates decision correctness, matched-pair reliability, and fully correct responses. Our findings reveal a recurring difficulty in recognizing when intervention is unnecessary: models can identify situations requiring clarification or repair yet still interven
    
[^38]: 异构AI模型间的同伴影响

    Peer Influence across Heterogeneous AI Models

    [https://arxiv.org/abs/2610.03095](https://arxiv.org/abs/2610.03095)

    该研究测量了七个开源语言模型之间的说服效应，发现模型意见分歧时说服作用非常强烈，但模型规模和单独运行时的确定性均无法预测说服动态，小模型既能像大模型一样有效说服他人，也同样能抵抗影响。

    

    当两个AI智能体产生分歧时，谁会说服谁？随着多智能体系统越来越多地组合使用不同家族和不同规模的语言模型，这一问题的答案将决定哪些判断能在交互中留存下来。我们将说服力衡量为智能体在与持异议的同伴进行单次交流后其决策发生的概率偏移，并在三项语言理解任务上测试了七个开源权重模型。研究发现，说服作用非常强烈：当模型意见不一致时，接收方在看到同伴的答案和解释后往往会放弃自己最初的判断。然而令人惊讶的是，无论是模型单独运行时的确定性还是模型规模，都无法可靠地预测说服动态。在独立运行时决策几乎完全一致的模型，反而可能最容易受到说服的影响；而小模型作为说服者可以与大模型相匹敌，并且同样能有效地抵抗后者的影响。此外，我们还表明，决策偏移的大小更多取决于接收方的易感程度而非说服方。

    arXiv:2610.03095v1 Announce Type: new  Abstract: When two AI agents disagree, who persuades whom? As multi-agent systems increasingly combine language models of different families and sizes, the answer can determine which judgments survive interaction. Measuring persuasion as the probabilistic shift in an agent's decision after a single exchange with a dissenting peer, we test seven open-weight models across three language understanding tasks. We find that persuasion is strong: when models disagree, receivers often abandon their initial judgment after seeing a peer's answer and explanation. Surprisingly, however, neither standalone certainty nor model scale reliably predicts persuasion dynamics. Models producing almost perfectly consistent decisions in isolation can be among the most susceptible to persuasion, and small models can match larger ones as persuaders and resist their influence just as effectively. Furthermore, we show that the size of the shift depends more on the susceptib
    
[^39]: MintEval：大语言模型是否实现了你所要求的交易策略？一个面向自然语言转策略代码的行为等价性基准

    MintEval: Do LLMs Implement the Trading Strategy You Asked For? A Behavioural-Equivalence Benchmark for Natural-Language-to-Strategy Code

    [https://arxiv.org/abs/2610.03080](https://arxiv.org/abs/2610.03080)

    该论文提出MintEval基准，通过程序化生成参考交易策略并回译为自然语言指令让大语言模型重新实现，再在相同市场数据上逐K线比较生成策略与参考策略的实际交易行为（而非代码相似度或利润），以检验大语言模型编写的策略代码是否真正做到了行为等价于交易者的原始意图。

    

    大语言模型正在从生成交易信号转向编写执行这些信号的代码。第二种角色的失败模式是静默的：生成的代码可以运行，回测可以画出图表，但交易者所描述的风险逻辑却并非实际执行的逻辑。现有代码基准通过单元测试检验功能正确性，金融基准则检验预测能力，二者都无法衡量一个实现的行为是否与所要求的策略一致。我们提出MintEval，该基准从可组合的构建模块库中以程序化方式生成参考策略，将其回译为口语化的交易者指令，再由被测模型重新实现。生成的程序与参考程序在完全相同的市场数据和摩擦成本上逐根K线执行，并基于其交易行为而非代码相似度或利润进行比较：超额收益（alpha）被差分消除。MintEval v0包含800个基于BTCUSDT 15分钟的……（摘要不完整）

    arXiv:2610.03080v1 Announce Type: cross  Abstract: Large language models are moving from producing trading signals to writing the code that executes them. The failure mode of the second role is silent: generated code runs, a backtest plots, yet the risk logic that the trader described is not the logic being executed. Existing code benchmarks test functional correctness on unit tests and finance benchmarks test forecasting; neither measures whether an implementation behaves like the strategy that was asked for. We introduce MintEval, a benchmark in which reference strategies are generated programmatically from a library of composable building blocks, back-translated into colloquial trader instructions, and re-implemented by the model under test. Generated and reference programs are executed bar by bar on identical market data and frictions, and compared on their actions rather than on code similarity or profit: alpha is differenced away. MintEval v0 contains 800 tasks on BTCUSDT 15-minu
    
[^40]: 一种用于自发对话中标准化语音单元标注的自动化流水线

    An automated pipeline for standardised speech-unit annotation in spontaneous dialogue

    [https://arxiv.org/abs/2610.03078](https://arxiv.org/abs/2610.03078)

    本文提出一种自动化流水线，可从自发二人对话的分通道录音中可靠提取对话轮次与听者反馈信号，为对话动态研究提供标准化的初步语音单元标注。

    

    量化对话动态需要可靠地识别交互单元及其时间边界，但仅凭语音活动无法区分对话轮次与听者反馈或轮次内停顿。我们提出了一种自动化流水线，用于从自发二人对话的分通道录音中提取对话轮次和反馈信号，旨在为后续人工审核提供一致的初步标注。该流水线结合了语音活动检测、通道能量过滤、时间合并、自动语音识别以及基于上下文的后处理。我们使用分段级检测可靠性和时间边界误差两项指标，在来自33对对话者的99段十分钟丹麦语对话上对该流水线进行了评估。对话分别在正常和不对称听觉条件下录制。在后一种条件下，通过骨传导耳机向一名参与者播放语音形状噪声

    arXiv:2610.03078v1 Announce Type: new  Abstract: Quantifying conversational dynamics requires reliable identification of interactional units and their temporal boundaries, but speech activity alone does not distinguish conversational turns from listener feedback or within-turn pauses. We present an automated pipeline for extracting turns and backchannels from separate-channel recordings of spontaneous dyadic conversation, designed to provide a consistent first-pass annotation for subsequent human review. The pipeline combines voice activity detection, channel-energy filtering, temporal merging, automatic speech recognition, and context-based post-processing. We evaluated the pipeline on 99 ten-minute Danish conversations from 33 dyads using segment-level detection reliability and temporal boundary error. Conversations were recorded under both normal and asymmetric listening conditions. In the latter, speech-shaped noise was delivered to one participant through bone-conduction headphone
    
[^41]: 揭开宣传的面纱：掩码语言模型与因果语言模型的对比分析

    Unmasking Propaganda: A Comparative Analysis of Masked and Causal Language Models

    [https://arxiv.org/abs/2610.03077](https://arxiv.org/abs/2610.03077)

    本文基于SemEval-2020 Task 11数据集，系统对比了掩码语言模型与多家厂商的因果语言模型在宣传手法检测任务上的表现，揭示了不同类型现代语言模型在识别隐蔽宣传技术方面的能力差异。

    

    宣传检测是自然语言处理（NLP）中的一项重要任务，尤其是在操纵性政治传播的背景下。然而，识别特定的宣传手法是一项重大挑战，因为它们往往具有隐蔽性且依赖于上下文，使其难以与正当的说服性语言区分开来。宣传通常通过强调某些事实而淡化或忽略其他事实来营造预期的认知。这种带有偏向性的传播旨在影响人们对特定事业或立场的态度、信念或行为。本文通过对现代语言模型进行对比分析，探讨了宣传技术检测方面的进展，实验使用了SemEval-2020 Task 11数据集。我们评估了掩码语言模型（基于XLM-RoBERTa或DeBERTa V3）和因果语言模型（来自OpenAI、Google、Mistral、Anthropic和Meta），并采用了两种提示策略：

    arXiv:2610.03077v1 Announce Type: new  Abstract: Propaganda detection is an essential task in natural language processing (NLP), particularly in the context of manipulative political communications. However, identifying specific propaganda techniques presents a significant challenge due to their often subtle nature and reliance on context, making them difficult to distinguish from legitimate persuasive language. Propaganda often involves highlighting certain facts while downplaying or ignoring others to create a desired perception. This biased communication aims to influence attitudes, beliefs, or behaviors towards a particular cause or position. This paper explores advances in detecting propaganda techniques through a comparative analysis of modern language models, using the SemEval-2020 Task 11 dataset. We evaluated both masked language models (based on XLM-RoBERTa or DeBERTa V3) and causal models (from OpenAI, Google, Mistral, Anthropic and Meta), employing two prompting strategies:
    
[^42]: SecJev：将安全专业知识引入系统一决策模型

    SecJev: Bringing Security Expertise to System One Decision Models

    [https://arxiv.org/abs/2610.03073](https://arxiv.org/abs/2610.03073)

    该论文提出SecJev，首个专门面向安全领域的类Jev决策模型家族（参数规模0.8B至9B），能够从文本、遥测和观察历史中学习布尔型、选择型和有序的安全决策，并通过SecJev语料库在14个任务和8个数据源上统一实现源标签预测与显式策略评估。

    

    安全工作流需要能够将复杂观察和明确策略转化为决策的模型。Jev提出的系统一模型可以返回类型化的预测和概率，而安全专业化则为这些预测提供了领域专业知识支撑。我们提出了SecJev，据我们所知，这是首个专门面向安全领域的类Jev决策模型家族，参数规模覆盖0.8B至9B。SecJev构建于Kev的单次遍历候选评分器之上，能够从文本、遥测数据和观察历史中学习布尔型、选择型和有序决策。我们开发了SecJev语料库，以在14个任务和8个数据源上统一源标签预测与显式策略评估，涵盖工具输出、流量、联邦更新、共识、身份验证和车辆消息。场景加权训练使模型能够适应这些不同领域，同时保持共享的类型化决策接口。安全专业化提升了该家族中每一个模型的性能。

    arXiv:2610.03073v1 Announce Type: cross  Abstract: Security workflows need models that turn complex observations and explicit policies into decisions. System One models introduced by Jev return typed predictions and probabilities; security specialization supplies the domain expertise behind those predictions. We introduce SecJev, to our knowledge the first family of Jev-like decision models specialized for security, spanning 0.8B to 9B parameters. Built on Kev's single-pass candidate scorer, SecJev learns Boolean, choice, and ordered decisions from text, telemetry, and observation histories. We develop SecJev-Corpus to unify source-label prediction and explicit-policy evaluation across 14 tasks and eight sources. It covers tool outputs, traffic, federated updates, consensus, authentication, and vehicle messages. Scene-weighted training adapts the models across these domains while preserving a shared typed decision interface. Security specialization improves every model in the family; S
    
[^43]: HARPO：面向忠实且创造性语言生成的幻觉感知强化学习

    HARPO: Hallucination-Aware Reinforcement Learning for Faithful and Creative Language Generation

    [https://arxiv.org/abs/2610.03063](https://arxiv.org/abs/2610.03063)

    提出强化学习框架 HARPO，通过幻觉感知生成式奖励模型与选择性激活机制，在不牺牲创造力的情况下联合优化大语言模型生成的忠实度与写作质量。

    

    大语言模型（LLM）容易生成幻觉内容，这损害了它们在知识密集型任务中的可靠性。为了在不牺牲创造力的前提下解决这一挑战，我们提出了 HARPO，一个旨在联合优化忠实度与创造力的强化学习框架。HARPO 引入了通过可验证反馈训练的幻觉感知生成式奖励模型（HA-GRM），用于同时评估忠实度与写作质量。选择性激活机制（SAM）仅对被 HA-GRM 判定为无幻觉的输出激活写作奖励，同时数据课程逐步将训练从创意写作过渡到以幻觉为中心的任务。在 RAGTruth 数据集上，基于 Qwen3-4B 的 HA-GRM 达到了 78.08% 的响应级 F1 分数，而监督微调基线为 66.37%。在 1.7B 到 8B 参数规模的 Qwen2.5 和 Qwen3 模型上的实验表明，该框架在忠实生成和……（摘要原文截断）

    arXiv:2610.03063v1 Announce Type: new  Abstract: Large Language Models (LLMs) are prone to generating hallucinated content, which compromises their reliability in knowledge-intensive tasks. To address this challenge without sacrificing creativity, we propose HARPO, a reinforcement learning framework designed to jointly optimize faithfulness and creativity. HARPO incorporates a Hallucination-Aware Generative Reward Model (HA-GRM), trained via verifiable feedback, to assess both faithfulness and writing quality. A Selective Activation Mechanism (SAM) activates writing rewards only for outputs judged hallucination-free by HA-GRM, while a data curriculum progressively shifts training from creative writing to hallucination-centric tasks. On RAGTruth, our Qwen3-4B-based HA-GRM achieves a response-level F1 score of 78.08%, compared with 66.37% for the supervised fine-tuning baseline. Experiments on Qwen2.5 and Qwen3 models from 1.7B to 8B parameters show improvements in both faithful generati
    
[^44]: 大语言模型中知识可及性的几何结构

    The Geometry of Knowledge Accessibility in Large Language Models

    [https://arxiv.org/abs/2610.03052](https://arxiv.org/abs/2610.03052)

    大语言模型的知识可及性在查询的表示空间中呈现出以中心为参照的几何结构——查询表示离中心越近其所需知识越容易被召回，由此可以在生成之前就通过几何距离刻画并预测模型的知识边界。

    

    大语言模型（LLMs）包含广泛的知识，但它们无法可靠地访问所有这些知识。我们通过“知识可及性”来研究这一问题，即查询所需的知识能否从模型中被回忆起来。我们发现，在模型生成任何内容之前，仅从模型对查询本身的表示中，知识可及性就呈现出一种简单的几何结构：越容易被访问的查询在表示空间中越靠近某个中心，而越难被访问的查询则离中心越远。这种几何结构揭示了一条知识边界，将更易访问的查询与较难访问的查询区分开来。可及性随离中心距离的增加而持续下降，并且这种基于距离的排序即使在不同中心的情况下也能跨数据集迁移。受控实验进一步表明，这种以中心为核心的几何结构与知识可及性的关联比与推理难度的关联更为紧密。该几何结构还揭示了……

    arXiv:2610.03052v1 Announce Type: new  Abstract: Large language models (LLMs) contain broad knowledge, but they cannot access all of it reliably. We study this problem through knowledge accessibility, which describes whether the knowledge needed for a query can be recalled from the model. We find that knowledge accessibility has a simple geometric structure in the model's representation of the query alone, before any generation. More accessible queries are closer to a center in the representation space, while less accessible queries are farther away. This geometry reveals a knowledge boundary that separates more accessible queries from less accessible ones. Accessibility consistently decreases with distance from the center, and this distance-based ordering transfers across datasets even when the centers differ. Controlled experiments further show that the centered geometry is more closely related to knowledge accessibility than to reasoning difficulty. The geometry also reveals when di
    
[^45]: HyperThink：面向高效推理的文本到参数超网络

    HyperThink: Text-to-Parameter Hypernetworks for Efficient Reasoning

    [https://arxiv.org/abs/2610.03039](https://arxiv.org/abs/2610.03039)

    HyperThink通过轻量级超网络将长思维链推理计算摊销为一次查询条件下的参数更新，使模型无需生成冗长思考轨迹即可直接输出简洁解答，在大幅降低推理延迟和token消耗的同时保持强推理性能。

    

    长篇思考轨迹能够显著提升大语言模型（LLM）的多步推理性能，但会引入高昂的推理时开销，其中延迟主要由顺序解码主导。我们提出HyperThink，这是一种文本到参数的方法，将推理计算摊销为一次基于查询条件的参数更新：一个轻量级超网络读取问题并预测基础LLM中一小部分参数的更新，同时向量量化解码器将这些更新约束在有限的、可复用的模式集合内，以提升鲁棒性和迁移能力。HyperThink在基础模型自身的输出上进行端到端训练，在测试时消除了冗长的思考轨迹：仅需一次超网络前向传播，适配后的模型即可生成简洁的分步解答和最终答案，无需中间思考过程，使用的token数量大幅减少，同时保持强大的推理性能。实验表明，HyperThink……

    arXiv:2610.03039v1 Announce Type: new  Abstract: Long-form thinking traces can substantially improve the multi-step reasoning performance of large language models (LLMs), but they introduce high inference-time overhead, with latency dominated by sequential decoding. We propose HyperThink, a text-to-parameter approach that amortizes this reasoning computation into a single query-conditioned parameter update: a lightweight hypernetwork reads the question and predicts updates to a small subset of the base LLM's parameters, while a vector-quantized decoder constrains them to a finite set of reusable patterns to improve robustness and transfer. Trained end-to-end on outputs from the base model itself, HyperThink eliminates long thinking traces at test time: after one hypernetwork forward pass, the adapted model generates a concise step-by-step solution and final answer without an intermediate trace, using far fewer tokens while retaining strong reasoning performance. Empirically, HyperThink
    
[^46]: 用于快速随机扩散采样的自适应二阶求解器

    Adaptive Second-Order Solvers for Fast Stochastic Diffusion Sampling

    [https://arxiv.org/abs/2610.03034](https://arxiv.org/abs/2610.03034)

    该论文将PI步长控制与扩散噪声归一化误差估计器结合，提出了扩散模型的自适应二阶求解器，实现更平滑的步长调整，并可将逐样本的自适应轨迹聚合为固定调度，在大幅降低采样成本的同时保留自适应采样的质量收益。

    

    扩散模型依赖于需要时间离散化的数值求解器，这对采样成本与质量之间的权衡有很大影响。然而，反向过程的计算难度沿采样轨迹以及在不同数据分布之间会发生变化，因此离散化方式的选择十分重要。我们将比例-积分（PI）步长控制适配到扩散模型中，并使用我们提出的扩散噪声归一化误差估计器。与扩散领域中现有的仅响应当前误差的自适应方法不同，PI求解器还会结合先前的误差，从而实现更平滑的步长调整。我们进一步证明，这些逐样本的轨迹表现出共享结构，可以将其聚合为固定调度，从而保留自适应采样的大部分收益。我们在自然图像和语言数据集上，以在相同次数神经网络评估下的FID作为质量衡量标准，对这两种方法进行了评估。

    arXiv:2610.03034v1 Announce Type: cross  Abstract: Diffusion models rely on numerical solvers requiring time-discretization, which has a large influence on the tradeoff between sampling cost and quality. However, the computational difficulty of the reverse process varies along the sampling trajectory and across data distributions, making the choice of discretization important. We adapt proportional-integral (PI) step-size control to diffusion, using our diffusion noise-normalised error estimator. Unlike existing adaptive methods in diffusion that respond only to the current error, the PI solver also incorporates the previous error, yielding smoother step adaptation. We further show that these per-sample trajectories exhibit shared structure and can be aggregated into a fixed schedule that retains much of the benefit of adaptive sampling. We evaluate both approaches on natural-image and language datasets, in terms of quality, measured by FID at a matched number of neural network evaluat
    
[^47]: 为1比特KV缓存压缩量身定制量化空间

    Tailoring the Quantization Space for 1-Bit KV Cache Compression

    [https://arxiv.org/abs/2610.03027](https://arxiv.org/abs/2610.03027)

    提出TaSQ方法，通过查询引导的通道加权、跨头归一化和协方差感知的通道分组来量身定制向量量化目标空间，从而在1比特极端压缩下实现有效的KV缓存压缩。

    

    键值缓存已成为长上下文大语言模型推理中的主要内存瓶颈，给内存容量和带宽带来了巨大压力。为缓解这一瓶颈，向量量化（VQ）已成为一种有前景的激进KV缓存压缩方法。然而，现有的VQ方法在1比特压缩区间下性能大幅下降。在如此极端的压缩下，每个码本必须用有限的质心集合来表示更大规模的通道组，使得有效利用码本容量变得愈发困难。为解决这一问题，我们提出了TaSQ，它通过结合查询引导的通道加权、跨头归一化以及协方差感知的通道分组来量身定制VQ目标空间，从而更好地反映缓存激活的误差敏感性和统计结构。由于这些变换与RoPE兼容，并且可以轻松地合并到投影权重和码本中，TaSQ保持了传统的（摘要在此处被截断）

    arXiv:2610.03027v1 Announce Type: cross  Abstract: The key-value (KV) cache becomes a major memory bottleneck in long-context LLM inference, placing substantial pressure on memory capacity and bandwidth. To mitigate this bottleneck, vector quantization (VQ) has emerged as a promising approach for aggressive KV cache compression. However, existing VQ methods degrade substantially in the 1-bit regime. At such extreme compression, each codebook must represent a larger group of channels with a limited set of centroids, making effective use of its capacity increasingly challenging. To address this, we introduce $\textbf{TaSQ}$, which tailors the VQ target space by combining query-guided channel weighting, cross-head normalization, and covariance-aware channel grouping to better reflect the error sensitivity and statistical structure of cached activations. Since these transforms are RoPE-compatible and can be easily merged into projection weights and codebooks, TaSQ preserves the conventiona
    
[^48]: 偏好的可验证、可表达与默会成分

    Verifiable, Articulable, and Tacit Components of Preference

    [https://arxiv.org/abs/2610.03025](https://arxiv.org/abs/2610.03025)

    该论文推出了包含280万文本和3.17亿人类偏好判断的CreativePreferences数据集，通过可执行程序、评分标准库与密集训练模型三种方式建模偏好，量化了“可表达”与“可验证”成分和完整偏好之间的差距，揭示了AI偏好优化中长期被忽视的默会成分。

    

    是什么让一篇短篇小说引人入胜、一篇新闻报道具有新闻价值、或一个数学证明优雅？这些构念难以言明或验证，其含义至少部分是默会的。然而，现代AI模型主要是通过明确的章程、评分标准和验证器（即RLAIF和RLVR）来改进的；偏好的默会成分通常研究不足。我们引入了一个大规模、带标注的偏好数据集CreativePreferences，其中包含280万个文本，由3.17亿个人类偏好判断在7个创意领域中进行标注，并配有42个基准任务。我们分别用可执行程序、评分标准库和密集训练的模型（V、A和VAT）对这些标签进行建模。我们观察到稳健的可表达性差距（VAT−VA）和可验证性差距（VAT−V）；我们采用一种新颖的测量方法来估计每个差距的上界和下界，该方法可发现可表达和可验证的指标、识别伪变量，并估计未被发现成分的价值。

    arXiv:2610.03025v1 Announce Type: new  Abstract: What makes a short story gripping; a news article newsworthy; or a math proof elegant? These constructs resist articulation or verification; their meaning is at least partially tacit. However, modern AI models are improved primarily via articulated constitutions, rubrics and verifiers (i.e. in RLAIF and RLVR); tacit components of preferences are typically understudied. We introduce a large, labeled preference dataset CreativePreferences, containing 2.8M texts labeled by 317M human preference judgments across 7 creative domains, with 42 benchmark tasks. We model these labels with executable programs, rubric banks and densely trained models (V, A and VAT, respectively). We observe robust articulability gaps, VAT-VA; and verifiability gaps, VAT-V; we estimate upper and lower bounds for each gap with a novel measurement approach that discovers articulable and verifiable metrics, identifies spurious variables and estimates the value of undisc
    
[^49]: ReSCUE：面向非分段长时同步手语翻译的句子承诺重翻译框架

    ReSCUE: Re-translation with Sentence Commitment for Unsegmented Long-Form Simultaneous Sign Language Translation

    [https://arxiv.org/abs/2610.03022](https://arxiv.org/abs/2610.03022)

    提出了ReSCUE统一框架，通过推理感知训练、稳定重翻译和句子承诺机制，实现对非分段长时手语视频的同步翻译，在低延迟条件下取得最佳翻译质量。

    

    同步手语翻译（SLT）对实时通信至关重要，然而现有方法大多局限于句子级、离线的设置，并假定输入是预先分段的。这些假设阻碍了其在涉及连续、非分段视频流的现实场景中的部署。我们提出了ReSCUE，这是一个用于非分段长时手语视频同步翻译的统一框架，使训练和推理与现实流式条件保持一致。ReSCUE结合了推理感知训练以处理部分输入、非手语停顿和多句子上下文，采用稳定的重翻译策略以实现低延迟且可修正的预测并减少输出闪烁，同时引入句子承诺机制用于在线分段和内存管理。在标准句子级基准测试上的实验表明，ReSCUE实现了更低的延迟，并在低延迟设置下取得了最佳翻译质量。

    arXiv:2610.03022v1 Announce Type: cross  Abstract: Simultaneous Sign Language Translation (SLT) is critical for real-time communication, yet existing methods remain largely confined to sentence-level, offline settings that assume pre-segmented inputs. These assumptions hinder deployment in realistic scenarios involving continuous, unsegmented video streams. We present ReSCUE, a unified framework for simultaneous SLT on unsegmented long-form sign language videos that aligns training and inference with realistic streaming conditions. ReSCUE combines inference-aware training to handle partial inputs, non-signing pauses, and multi-sentence contexts, stabilized re-translation to enable low-latency yet revisable predictions with reduced output flicker, and a sentence commitment mechanism for online segmentation and memory management. Experiments on standard sentence-level benchmarks show that ReSCUE achieves lower latency and the best translation quality under low-latency settings. On long-f
    
[^50]: 使用人工对话为构音障碍及气管造口说话者构建个性化自动语音识别系统

    Personalized Automatic Speech Recognition for a Dysarthric and Tracheostomic Speaker using Artificial Conversations

    [https://arxiv.org/abs/2610.03017](https://arxiv.org/abs/2610.03017)

    本文针对一位气管造口且严重构音障碍的捷克说话者，通过多阶段微调Whisper模型构建个性化语音识别系统，实现字符错误率相对降低50%，并发布了基于“人工对话”协议收集的33小时公开数据集。

    

    本工作提出了一个针对一位捷克说话者的个性化自动语音识别（ASR）系统。该说话者具有永久性气管造口和严重的构音障碍，其语音对于未经训练的听者而言难以理解。我们发布了一个包含该说话者33小时标注语音的公开数据集，这些数据是通过一种新颖的“人工对话”协议收集的，该协议旨在实现高参与度和对话真实感。我们提出了一种基于Whisper Base的多阶段训练流程：先在标准捷克语语音上微调，再在声学模拟的气管造口语音上训练，最后使用说话者本人的数据进行微调。我们在三种近实时场景下对该系统进行了评估：脚本对话、问答和自发对话。相比Whisper Base基线，该系统实现了50%的相对字符错误率降低，并在孤立话语的声学识别方面超越了其助手的平均识别准确率。我们证明，即使对于……（摘要截断）

    arXiv:2610.03017v1 Announce Type: new  Abstract: This work presents an automatic speech recognition (ASR) system personalized for a Czech speaker with a permanent tracheal stoma and severe dysarthria rendering their speech unintelligible to untrained listeners. We release a public dataset containing 33 annotated hours of the speaker's speech, collected using a novel "artificial conversation" protocol designed for high engagement and dialogue realism. We propose a multi-stage training pipeline based on Whisper Base: fine-tuning on standard Czech speech, acoustically simulated tracheostomic speech, and the speaker's data. We evaluate the system across three near real-time scenarios: scripted conversations, question answering, and spontaneous dialogue, achieving a 50\% relative reduction in Character Error Rate compared to Whisper Base baseline and surpassing the average recognition accuracy of their assistants in acoustic recognition of isolated utterances. We demonstrate that even for s
    
[^51]: 统一多模态模型中的递归自我改进

    Recursive Self-Improvement in Unified Multimodal Models

    [https://arxiv.org/abs/2610.03002](https://arxiv.org/abs/2610.03002)

    提出递归跨能力自我改进（RSI）训练循环，让统一多模态模型的文本与视觉能力相互提供训练数据，并以程序执行作为模型外部的真实性验证来源来防止错误跨轮累积，同时构建了BasicChartBench基准用于评估开源模型。

    

    统一多模态模型（UMM）能够理解并生成文本和图像，这使得模型可以生成自己的训练数据。现有的统一多模态模型自我改进方法将监督保持在视觉一侧，即用图像理解来判断图像生成。我们提出递归跨能力自我改进（RSI），这是一种训练循环，其中统一多模态模型的文本能力与视觉能力相互为对方提供训练数据。在每一轮中，模型生成图像并读取它们以发现自身的不足之处，然后针对这些不足编写程序，并通过执行程序将每个结果与其规范进行验证。经过验证的图像渲染结果用于训练图像生成，而带有标注的渲染结果和模型自身正确的程序则用于训练视觉理解和程序编写。程序执行因此充当了模型之外的真实性来源，使错误不会在多轮之间累积。我们在图表任务上研究RSI，并构建了BasicChartBench来评估开源模型。

    arXiv:2610.03002v1 Announce Type: new  Abstract: Unified multimodal models (UMMs) understand and generate both text and images, which lets a model produce its own training data. Existing self-improvement in UMMs keeps supervision on the visual side, where image understanding judges image generation. We propose recursive cross-capability self-improvement (RSI), a training loop in which the text and visual abilities of a UMM supply training data for one another. In each round, the model generates images and reads them to find where it falls short. It then writes programs aimed at these shortcomings, and execution verifies every result against its specification. Verified renders train image generation, while labeled renders and the model's own correct programs train visual understanding and program writing. Program execution thus acts as a source of truth outside the model, so errors do not accumulate across rounds. We study RSI on charts and build BasicChartBench to evaluate open models 
    
[^52]: OmniConfess：引出词元级“供词”以缓解全模态幻觉

    OmniConfess: Eliciting Token Confessions to Mitigate Omni-Modal Hallucination

    [https://arxiv.org/abs/2610.02999](https://arxiv.org/abs/2610.02999)

    OmniConfess提出了一种无需训练的全模态幻觉缓解方法，通过通道级证据干预生成词元级“供词”以识别回答的证据依赖，从而保留有据内容并修正错误承诺，并配套构建了覆盖文本、图像、音频、视频的OmniHalluBench基准。

    

    全模态大语言模型统一了文本、图像、音频和视频，但当生成依赖于错误的证据时会产生幻觉。现有的推理时方法虽能减少幻觉，却很少揭示究竟是哪些证据在支撑已生成的内容承诺。我们提出OmniConfess，一种无需训练的全模态幻觉缓解方法。该方法固定一个候选回答，并在受控的通道级证据干预下以词元分辨率对其进行重新评分，产生一种结构化的“词元-通道”供词，从而揭示该回答对证据的依赖关系。OmniConfess利用这份供词来保留有据可依的内容，并修正由无关或矛盾证据所驱动的内容承诺。为评估OmniConfess，我们构建了OmniHalluBench，这是一个包含3,540个样本的基准，由六个数据集构成，涵盖文本、图像、音频和视频场景，并包含判断题与自由形式生成两类任务。实验表明，OmniConfess缓解……

    arXiv:2610.02999v1 Announce Type: new  Abstract: Omni-modal large language models (OmniLLMs) unify text, images, audio, and video, yet hallucinate when generation relies on the wrong evidence. Existing inference-time methods can reduce hallucinations, but rarely reveal which evidence sustains a generated commitment. We introduce OmniConfess, a training-free method for mitigating omni-modal hallucinations. It fixes a candidate response and re-scores it at token resolution under controlled channel-wise evidence interventions, producing a structured token-by-channel confession that reveals the response's evidential dependence. OmniConfess uses this confession to preserve grounded content and correct commitments driven by irrelevant or contradictory evidence. To evaluate OmniConfess, we construct OmniHalluBench, a 3,540-example benchmark built from six datasets spanning text, image, audio, and video settings and both judgment and free-form generation. Experiments show that OmniConfess miti
    
[^53]: Sentry：学习在测试时从LLM智能体失败中恢复

    Sentry: Learning to Recover from LLM Agent Failures at Test Time

    [https://arxiv.org/abs/2610.02994](https://arxiv.org/abs/2610.02994)

    提出Sentry——一个与LLM智能体并行运行的失败管理层，将失败经验视为条件性知识，在检测到失败时按需从外部经验手册中检索指导恢复、无奖励验证恢复结果并仅在确认恢复后存储新经验，从而在测试时实现从失败中学习。

    

    LLM智能体经常在任务执行中途因无效的工具调用、重复操作或缺乏依据的推理而失败，从这些失败中学习是提升可靠性的途径。我们发现，失败知识如何传递给智能体与其内容本身同样重要。失败经验是条件性的：如果保留在智能体的上下文中，当对应的失败情形并不存在时它们会误触发，而将它们从不断演化的经验手册中移除反而能提升性能。相比之下，运行时干预只在失败发生时起作用，但不会从修复过程中学习。我们提出，失败知识是一种条件性知识，应当被有条件地暴露，并将这一原则实例化为Sentry——一个与智能体并行运行的失败管理层。当Sentry检测到失败时，它会从外部经验手册中检索匹配的经验来指导恢复，在无需访问任务奖励的情况下验证智能体是否已恢复，并且只有在其确实恢复时才存储新的经验。

    arXiv:2610.02994v1 Announce Type: cross  Abstract: LLM agents often fail mid-task due to invalid tool calls, repeated actions, or poorly grounded reasoning, and learning from these failures is a path to reliability. We find that how failure knowledge reaches the agent matters as much as what it contains. Failure lessons are conditional: kept in the agent's context, they misfire when their failure is absent, and removing them from an evolving playbook improves performance. Runtime interventions, in contrast, act only when a failure occurs but do not learn from their repairs. We argue that failure knowledge is conditional knowledge and should be conditionally exposed, and instantiate this principle in Sentry, a failure-management layer that runs alongside the agent. When Sentry detects a failure, it retrieves matching lessons from an external playbook to guide recovery, verifies without access to task rewards whether the agent recovered, and stores a new lesson only if it did; the full p
    
[^54]: OLMo-Detect：一个面向大语言模型成员推断的多阶段、混杂因素可控的基准测试

    OLMo-Detect: A Multi-Stage, Confounder-Controlled Benchmark for Membership Inference on Large Language Models

    [https://arxiv.org/abs/2610.02986](https://arxiv.org/abs/2610.02986)

    该论文提出了基于完全开源OLMo 2流程构建的多阶段、混杂因素可控的成员推断基准OLMo-Detect，其覆盖预训练至后训练全流程、在三个关键维度显式对齐成员与非成员分布、并通过infini-gram严格过滤非成员，从而为评估大语言模型成员推断方法提供了更严谨的平台。

    

    大语言模型（LLM）上的成员推断旨在判断给定的文本样本是否被包含在LLM的训练数据中，且无需访问其训练语料库。尽管近期取得了一定进展，现有基准测试仍存在三个局限：对训练阶段的覆盖有限、成员与非成员样本之间的分布对齐不足，以及缺乏对非成员样本与训练语料库的严格过滤。为解决这些局限，我们提出了OLMo-Detect，一个基于完全开源的OLMo 2流程构建的多阶段、混杂因素可控的基准测试。OLMo-Detect涵盖预训练、中期训练和后训练阶段，在三个关键维度上显式对齐成员与非成员样本，并通过infini-gram对非成员样本进行严格过滤。为评估对分布偏移的鲁棒性，我们进一步引入了OLMo-Detect (Shifted)，一个成员与非成员样本分布刻意错位的变体。我们评估了15种无监督方法和3种有监督方法（原文此处截断）。

    arXiv:2610.02986v1 Announce Type: new  Abstract: Membership inference on large language models (LLMs) aims to determine whether a given text sample was included in an LLM's training data, without access to its training corpus. Despite recent progress, existing benchmarks suffer from three limitations: limited coverage of training stages, insufficient distributional alignment between members and non-members, and lack of rigorous filtering of non-members against the training corpus. To address these limitations, we propose OLMo-Detect, a multi-stage, confounder-controlled benchmark built upon the fully open OLMo 2 pipeline. OLMo-Detect spans pre-training, mid-training, and post-training, explicitly aligns members and non-members on three key axes, and rigorously filters non-members via infini-gram. To assess robustness to distribution shifts, we further introduce OLMo-Detect (Shifted), a variant where members are misaligned with non-members. We evaluate 15 unsupervised and 3 supervised m
    
[^55]: 一种面向模式即代码生物医学命名实体识别的指南增强多智能体框架

    A Guideline-Augmented Multi-Agent Framework for Schema-as-Code Biomedical Named Entity Recognition

    [https://arxiv.org/abs/2610.02970](https://arxiv.org/abs/2610.02970)

    GAMA框架通过从标注数据中归纳并验证特定数据集的标注规则来构建指南记忆，结合多智能体协作与模式即代码的结构化输出，显著提升了生物医学命名实体识别的准确性和格式规范性。

    

    大语言模型（LLMs）通过指令遵循和上下文学习在生物医学命名实体识别方面展现出了可喜的潜力。然而，现有的基于LLM的BioNER方法仍然面临两个关键局限。首先，检索到的示例和外部生物医学知识对特定数据集的标注语义支持有限，导致实体边界、类型范围和标注规范模糊不清。其次，自由形式的生成缺乏足够的结构控制，常常导致无效格式、幻觉提及、重复实体和边界错误。为了解决这些局限，我们提出了GAMA，一个面向模式即代码BioNER的指南增强多智能体框架。GAMA首先从已标注的训练实例中归纳候选标注规则，并通过标注数据进行验证，以构建可靠的特定数据集指南记忆。在这些经过验证的规则的指导下，一个规划……（摘要在此处截断）

    arXiv:2610.02970v1 Announce Type: new  Abstract: Large language models (LLMs) have shown promising potential for biomedical named entity recognition (BioNER) through instruction following and in-context learning. However, existing LLM-based BioNER methods still face two key limitations. First, retrieved demonstrations and external biomedical knowledge provide limited support for dataset-specific annotation semantics, leaving entity boundaries, type scopes, and annotation conventions ambiguous. Second, free-form generation lacks sufficient structural control, often leading to invalid formats, hallucinated mentions, duplicated entities, and boundary errors. To address these limitations, we propose GAMA, a guideline-augmented multi-agent framework for schema-as-code BioNER. GAMA first induces candidate annotation rules from labeled training instances and verifies them against annotated data to construct reliable dataset-specific guideline memory. Guided by these verified rules, a planning
    
[^56]: 理解联邦世界模型学习中的轨迹异质性

    Understanding Trajectory Heterogeneity in Federated World Model Learning

    [https://arxiv.org/abs/2610.02957](https://arxiv.org/abs/2610.02957)

    该论文在临床时间序列数据上系统基准测试了联邦世界模型学习中的轨迹异质性问题，发现客户端数据所有权与参与率会共同限制长时程窗口的可用覆盖，揭示了跨时间联邦学习的核心训练瓶颈。

    

    世界模型从轨迹中学习状态演化，因此获取时间上下文是其训练的核心需求。联邦学习可以利用分布式记录，但轨迹内部的数据所有权边界限制了每个客户端能够构造的样本。我们的研究通过对八个MIMIC-IV疾病队列进行按小时动作条件化的临床预测，对这一跨时间设置进行了基准测试，共涵盖4087万个状态转移归属关系。我们明确了基于疾病严重程度的客户端数据所有权、按患者分离的数据构造方式、局部历史与未来窗口规则，以及从1小时到32小时的成对滚动评估协议。十种联邦算法构成的算法矩阵在五轮10%参与率的设置下覆盖了32种疾病—数据划分配置。从已有结果和训练日志中得出三项发现。第一，客户端所有权与参与率共同限制了长窗口的覆盖能力：在汇总可用的32步窗口中，仅有7.55%–21.36%的窗口……

    arXiv:2610.02957v1 Announce Type: cross  Abstract: World models learn state evolution from trajectories, making access to temporal context a central training requirement. Federated learning can use distributed records, while ownership boundaries within a trajectory restrict the examples each client can construct. Our study benchmarks this cross-time setting through hourly action-conditioned clinical prediction on eight MIMIC-IV disease cohorts, comprising 40.87 million transition memberships. We specify severity-based client ownership, patient-separated construction, local history and future-window rules, and paired rollout evaluation from one to 32 hours. A matrix of ten federated algorithms covers 32 disease--partition configurations under five rounds of ten-percent participation. Three findings emerge from existing results and training logs. First, client ownership and participation jointly restrict long-window coverage: only 7.55\%--21.36\% of pooled-available 32-step windows have 
    
[^57]: 通过多编程语言指令微调与集成方法增强生物医学命名实体识别

    Enhancing Biomedical Named Entity Recognition via Multiple Programming Languages Instruction Tuning and Ensemble Method

    [https://arxiv.org/abs/2610.02949](https://arxiv.org/abs/2610.02949)

    该论文提出MITE方法，将生物医学命名实体识别重构为结构到结构的生成任务，通过多种编程语言的指令微调与集成策略，在不依赖昂贵外部知识资源的条件下提升了识别性能与模型鲁棒性。

    

    指令微调已成为将大语言模型（LLM）应用于生物医学命名实体识别（BioNER）的一种常见范式。然而，现有的指令微调方法仍然面临两个关键挑战。首先，传统的自然语言指令通常将BioNER标注序列化为扁平的文本输出，为带类型的实体抽取提供的结构约束有限。其次，高质量的生物医学标注数据有限，从单一的序列化输出形式中学习可能会限制结构多样性并降低模型的鲁棒性。尽管可以引入外部生物医学知识来缓解数据稀缺问题，但这通常需要昂贵的资源构建成本。为了应对这些挑战，我们提出了MITE，一种面向BioNER的多编程语言指令微调与集成方法。MITE将BioNER重新表述为结构到结构的生成任务，通过用编程语言表示指令和实体输出（原文在此处截断）。

    arXiv:2610.02949v1 Announce Type: new  Abstract: Instruction tuning has become a common paradigm for applying large language models (LLMs) to biomedical named entity recognition (BioNER). However, existing instruction-tuning approaches still face two key challenges. First, conventional natural-language instructions typically serialize BioNER annotations as flat textual outputs, providing limited structural constraints for typed entity extraction. Second, high-quality biomedical annotations are limited, and learning from a single serialized output form may restrict structural diversity and reduce model robustness. Although external biomedical knowledge can be introduced to alleviate data scarcity, it often requires costly resource construction. To address these challenges, we propose MITE, a Multiple Programming Languages Instruction Tuning and Ensemble method for BioNER. MITE reformulates BioNER as a structure-to-structure generation task by representing both instructions and entity ou
    
[^58]: 面向数学研究智能体的持续图记忆系统

    Continual Graph Memory for Mathematical Research Agents

    [https://arxiv.org/abs/2610.02945](https://arxiv.org/abs/2610.02945)

    提出 Ansatz——一个基于“持续图记忆”的数学研究智能体，通过可演化、跨问题的图结构记忆系统显式组织整个证明搜索过程，并复用先前问题的探索知识，从而有效管理海量中间证明结果。

    

    利用前沿智能体框架来解决数学研究问题，已成为推动数学发展的有效手段。然而，解决数学领域的前沿问题可能需要大量智能体长时间并行工作以构建证明，从而产生海量的中间证明结果。在长周期的证明搜索过程中组织这些中间结果，并复用先前探索所获得的知识，仍然是重大的挑战。我们提出了 Ansatz——一个围绕“持续图记忆”构建的数学研究智能体。持续图记忆是一种基于图、可演化、跨问题的数学研究记忆系统，它显式地组织整个证明搜索过程，并复用来自先前问题探索轨迹的信息。具体而言，我们开发了一个统一的图记忆，用于表示所有中间探索结果，包括事实、计划和反例……

    arXiv:2610.02945v1 Announce Type: new  Abstract: Using frontier agent harnesses to tackle mathematical research problems has emerged as an effective means of advancing mathematics. However, solving frontier problems in mathematics may require a massive number of agents working in parallel for extended periods to construct proofs, thereby generating an enormous volume of intermediate proof results. Organizing these intermediate results throughout a long-horizon proof-search process and reusing knowledge gained from prior explorations remain major challenges. We present Ansatz, a mathematical research agent built around Continual Graph Memory, a graph-based, evolvable, cross-problem mathematical research memory system that explicitly organizes the entire proof search process and reuses information from exploration trajectories of previous problems. Specifically, we develop a unified graph memory that represents all intermediate exploration results, including facts, plans, and counterexam
    
[^59]: 多语言提示污染下的输出语言混淆

    Output Language Confusion under Multilingual Prompt Contamination

    [https://arxiv.org/abs/2610.02926](https://arxiv.org/abs/2610.02926)

    该论文提出无需新数据、完全可复现的轻量级评估协议MDI，揭示多语言提示污染会诱发大语言模型的输出语言/文字切换，导致基于精确匹配的幻觉指标将事实正确的回答误判为幻觉，暴露出现有事实性基准测试中严重的指标混淆问题。

    

    标准的事实性基准测试建立在两个假设之上：提示为纯净的单语输入，且评分采用精确匹配；然而在真实世界的多语言部署场景中——从返回混合语言段落的检索增强生成（RAG）管道，到用户粘贴的多语言网页内容——这两个假设会同时失效。我们提出了多语言干扰项干扰（Multilingual Distractor Interference, MDI），这是一个轻量级、完全可复现、无需任何新数据或人工标注的评估协议：在事实性问题之前插入一个语义无关的外语句子。我们在此协议下，在TruthfulQA和TriviaQA两个基准上、八种干扰条件下对五个指令微调大语言模型进行了评估（共40,000次评估）。我们的核心发现是一个指标混淆问题：以Llama-3.1-8B为例，在印地语干扰条件下，58%的回答切换为天城文书写，由此得到的原始幻觉代理值高达0.710；但人工审查显示，在纯净条件下回答正确、而在干扰下发生文字切换的148个回答中，有120个……（原文摘要在此处截断）

    arXiv:2610.02926v1 Announce Type: new  Abstract: Standard factual benchmarks assume clean monolingual prompts and exact-match scoring, two assumptions that break simultaneously in real-world multilingual deployment, from retrieval-augmented generation pipelines returning mixed-language passages to users pasting multilingual web content. We introduce Multilingual Distractor Interference (MDI), a lightweight and fully replicable evaluation protocol requiring no new data or annotation, in which factual questions are preceded by a semantically irrelevant foreign-language sentence, and evaluate five instruction-tuned LLMs across TruthfulQA and TriviaQA under eight distractor conditions (40,000 evaluations). Our central finding is a metric confound: for Llama-3.1-8B under a Hindi distractor, 58% of responses switch to Devanagari script, yielding a raw hallucination proxy of 0.710, but manual review reveals that 120 of 148 script-switched responses that were correct under clean conditions rem
    
[^60]: 探测实验框架：语言模型陈旧数据强化学习比较的设置检查

    Probe the Harness: Setup Checks for Stale-Data RL Comparisons in Language Models

    [https://arxiv.org/abs/2610.02911](https://arxiv.org/abs/2610.02911)

    该论文提出PTH检查集，证明实验框架的细节（如PPO比率计算方式、数据种子传递、重放队列复用和损失归一化器实现）可以逆转陈旧数据强化学习方法比较的排名，强调了检查实验基础设施的必要性。

    

    在陈旧样本上训练语言模型的方法，通常通过与重要性校正基线的比较来评判。我们证明实验框架的细节可以逆转所观察到的方法排名，并提出了PTH（Probe The Harness，探测实验框架），这是一组使实验框架状态可见的检查方法。我们的案例是在verl和单GPU训练器上对SAN（一种无行为修正的方法）与截断重要性采样（TIS）进行比较，其中SAN最初在两个技术栈中都领先。实验框架的四个细节改变了这一比较：PPO比率是针对学习器自身重新计算的概率计算的；数据种子未传递到TIS分支；重放队列将其第一批数据复用于33次更新；两个损失归一化器与其描述不符。在每种情况下，被记录的量看起来都与正常工作的设置一致，而定义比较本身的量却未被检查。在检查实验框架之后，TIS匹配……（摘要被截断）

    arXiv:2610.02911v1 Announce Type: cross  Abstract: Methods for training language models on stale samples are judged by comparisons against importance-corrected baselines. We show that details of the experimental harness can reverse the observed ranking of methods, and we introduce PTH (Probe The Harness), a set of checks that makes the harness visible. Our case is a comparison between SAN, a behaviour-free method, and truncated importance sampling (TIS) on verl and in a single-GPU trainer, in which SAN first finished ahead in both stacks. Four details of the harness changed this comparison: the PPO ratio was taken against the learner's own recomputed probabilities, the data seed did not reach the TIS arm, the replay queue reused its first batch for 33 updates, and two loss normalisers differed from their description. In each case the logged quantity looked consistent with a working setup, while the quantity that defines the comparison went unchecked. With the harness checked, TIS match
    
[^61]: 无触发器的错误信息：从事实性回答到下游决策

    Misinformation Without Triggers: From Factual Answers to Downstream Decisions

    [https://arxiv.org/abs/2610.02886](https://arxiv.org/abs/2610.02886)

    该研究揭示虚假训练文档无需任何触发器即可改变语言模型的事实性回答，但直接回答的受污染程度无法预测下游决策行为，二者之间存在“审计差距”，因此仅审计直接答案会严重低估错误信息的真实危害。

    

    语言模型从网络文档中学习，其中一些文档是虚假的，而虚假内容可能渗入模型对事实性问题的回答，以及使用该回答的摘要和决策。大多数数据投毒研究会在训练数据中植入触发器，并在提示中激活它。虚假文档同样可以在没有任何触发器的情况下改变事实性回答，但我们尚不清楚直接回答能否预测后续决策。在这项工作中，我们追踪虚假内容在答案之后的传播路径，发现直接探测所报告的结果与模型随后的实际行为之间存在一种“审计差距”（audit gap）。我们在一个受控决策任务“猜首都”中，将虚假训练与匹配的真实性对照进行比较——该任务中一个固定的解码器将事实性回答转化为计分的卡片选择——并在来自2019–20年澳大利亚山火相关Facebook帖子的一条误导性声明上进行验证。在八种模型、污染剂量为1,000的设置下，直接注入选择率达到95.8–100%，而注入后的游戏决策增加……（原文在此处截断）

    arXiv:2610.02886v1 Announce Type: cross  Abstract: Language models learn from web documents, some of them false, and false content can reach a model's answer to a factual question and the summaries and decisions that use it. Most data-poisoning studies add a trigger to the training data and activate it in the prompt. False documents can also change factual responses without any trigger, but we do not know whether the direct answer predicts the decision. In this work, we follow false content past the answer and find an \emph{audit gap} between what a direct probe reports and what the model then does, comparing false training with matched truthful controls in a controlled decision task, \emph{Guess the Capital}, where a fixed decoder turns factual answers into a scored card choice, and on a misleading claim from Facebook posts about the 2019--20 Australian bushfires. Across eight models at dose 1,000, direct injected-choice rates reach 95.8--100\%, while injected game choices increase by
    
[^62]: 使用合作原则评估视觉语言模型中的视觉问答（VQA）

    Evaluating VQA in Vision Language Models using Cooperative Principles

    [https://arxiv.org/abs/2610.02878](https://arxiv.org/abs/2610.02878)

    本研究基于格赖斯合作原则，通过生成含有非必要、模糊或虚假信息的问题修饰语来评估视觉语言模型（ChatGPT、Claude、Gemini、Llava）在违反语用准则情况下的视觉问答性能，发现其性能显著下降，并揭示了人类语用推理与VLM推理之间的系统性差异。

    

    我们评估了视觉语言模型（VLM）在问题违反格赖斯准则时的视觉问答（VQA）表现。为此，我们使用VLM生成问题修饰语，这些修饰语会添加非必要的、模糊的或虚假的信息，并展示了在存在此类违规的情况下，我们所评估的VLM（ChatGPT、Claude、Gemini和Llava）性能会有所下降。此外，我们通过实验证明了人类与VLM在语用推理方式上的差异，以及VLM在解决人类引发的违规与AI生成的违规时的推理差异。最后，我们表明人类在解决VLM引发的违规时认知负担较低（通过实验中的任务耗时来衡量），但VLM本身在此类情况下的回答准确性反而较差。

    arXiv:2610.02878v1 Announce Type: new  Abstract: We evaluate the performance of Vision Language Models in Visual Question Answering (VQA) when questions violate Grice's maxims. To do this, we use VLMs to generate question modifiers that add non-essential, ambiguous or false information and show that in the presence of such violations, the VLMs that we evaluate (ChatGPT, Claude, Gemini and Llava) show diminished performance. Further, we empirically show the difference between how humans reason pragmatically compared to VLMs, and the difference in VLM reasoning when it resolves violations that are human-induced compared to those that are AI-generated. Finally, we show that human cognitive effort (measured through time-on-task in an experiment) is lower for resolving VLM-induced violations, but VLMs themselves perform less accurately in such cases.
    
[^63]: 超越分数对齐的“LLM作为评判者”评估：剩余评判难度的心理测量学分析

    Evaluating LLM-as-a-Judge Beyond Score Alignment: A Psychometric Analysis of Residual Judging Difficulty

    [https://arxiv.org/abs/2610.02877](https://arxiv.org/abs/2610.02877)

    该研究引入心理测量学中的多侧面Rasch模型，将人类与LLM评分分解出“剩余评判难度”这一新指标，发现LLM评判器与人类的总体分数对齐良好，并不代表二者对哪些评估案例更难判断具有一致的认知结构。

    

    大语言模型（LLM）被广泛用作自动评判器，其有效性通常通过与人类分数的对齐程度来评估。然而，总体一致性无法揭示人类和LLM是否认为相同的评估案例同样困难。本文从心理测量学视角研究摘要评估中的这一问题。我们分别对人类和LLM的评分拟合多侧面Rasch模型，将分数分解为潜在摘要质量、评分者严格度、维度严格度以及评分量表阈值。基于这一分解，我们将“剩余难度”定义为一种经模型调整的评判难度度量，并比较人类与LLM评判者是否共享相同的难度结构。在SummEval数据集上对17个开源权重LLM评判器的实验中，我们发现潜在摘要质量的适度对齐并不意味着剩余难度的对齐。人类和LLM评判者在哪些“摘要—维度”单元仍然困难上存在分歧，而这种不一致……

    arXiv:2610.02877v1 Announce Type: new  Abstract: Large language models (LLMs) are widely used as automatic judges, with validity typically assessed via alignment with human scores. However, aggregate agreement fails to reveal whether humans and LLMs find the same evaluation cases difficult. In this paper, we study this problem in summarization evaluation from a psychometric perspective. We fit Many-Facet Rasch Models separately to human and LLM ratings to decompose scores into latent summary quality, rater severity, dimension severity, and rating-scale thresholds. Building on this decomposition, we define residual hardness as a model-adjusted measure of judging difficulty and compare whether human and LLM judges share the same hardness structure. Across 17 open-weight LLM judges on SummEval, we find that moderate alignment in latent summary quality does not imply alignment in residual hardness. Human and LLM judges differ in which summary--dimension units remain difficult, and this mis
    
[^64]: 面向编码器跨语言性能提升的查询感知路由

    Query-aware routing for Cross-lingual performance gains in Encoders

    [https://arxiv.org/abs/2610.02875](https://arxiv.org/abs/2610.02875)

    该论文提出将仅作用于查询端的LoRA适配器与基于查询和索引语言的确定性路由相结合，在保留同语言性能和现有文档索引的同时，使英语、芬兰语、瑞典语六条跨语言检索方向的平均nDCG@10从0.241提升至0.291，相对提升20.9%。

    

    多语言编码器在查询与相关文档语言不同时，检索效果可能会下降，尽管其在同语言场景下表现强劲。我们研究了如何在保留编码器现有同语言性能和文档索引的前提下，提升芬兰语与瑞典语的跨语言检索效果。我们将仅作用于查询端、针对冻结文档嵌入训练的低秩适配器（LoRA），与基于查询语言和索引语言的确定性路由相结合：跨语言查询使用适配器，同语言查询则使用原始编码器。SampoTron（我们微调的低秩适配器与Nemotron-3-Embed-1B模型的组合）在一个采样的金融基准上，将六个英语、芬兰语和瑞典语方向的平均检索质量从nDCG@10的0.241提升至0.291，相对提升20.9%。所有六个跨语言方向均得到改善，并且路由保留了（原有同语言性能）。

    arXiv:2610.02875v1 Announce Type: cross  Abstract: Multilingual encoders can exhibit reduced retrieval effectiveness when queries and relevant documents differ in language, despite strong same-language performance. We investigate whether Finnish and Swedish cross-lingual retrieval can improve while preserving an encoder's existing same-language performance and document index. We combine a query-only low-rank adapter, trained against frozen document embeddings, with deterministic routing based on query and index languages. Cross-language queries use the adapter, while same-language queries use the original encoder. SampoTron, our fine-tuned low-rank (LoRA) adapter alongwith the Nemotron-3-Embed-1B model, improves average retrieval quality across six English, Finnish, and Swedish directions from 0.241 to 0.291 in normalized discounted cumulative gain (nDCG) at rank ten, a 20.9% relative gain on a sampled financial benchmark. All six cross-lingual directions improve, and routing preserves
    
[^65]: ConvoDrift：用于建模风格语调演变的多轮对话数据集

    ConvoDrift: A Multi-Turn Conversational Dataset for Modeling Stylistic Tone Evolution

    [https://arxiv.org/abs/2610.02873](https://arxiv.org/abs/2610.02873)

    ConvoDrift 是一个用于建模固定语义意图下多轮对话风格语调渐进漂移的数据集，包含 15,727 个多轮对话结构、风格漂移标注及基于五种人设条件的偏好成对数据集，可支持风格适应与个性化对齐的受控研究。

    

    对话中语言风格的演变是自然语言处理（NLP）中一个尚未充分探索的问题。现有的风格控制数据集大多聚焦于句子层面，或假设风格在整个对话过程中保持静态，忽略了交互过程中因用户偏好变化而产生的动态转变。我们提出了 ConvoDrift，一个旨在建模固定语义意图下渐进式对话风格语调漂移的数据集。该数据集基于 15,727 个共享的多轮对话结构构建，可用于风格适应以及基于人设条件的对齐方法研究。每段对话包含六组提示-回复对，每对均带有风格漂移和风格方向标签的标注，且涵盖了多种交流体裁。我们进一步衍生出一个互补的成对数据集，通过将语义等价但风格不同的回复进行配对，并利用五种不同的风格化交流人设标注基于人设条件的偏好，从而支持对个性化风格的受控研究。

    arXiv:2610.02873v1 Announce Type: cross  Abstract: The evolution of linguistic style in conversations is an underexplored issue in NLP. Most style-control datasets focus on sentences or assume a static style throughout, missing the dynamic shifts that occur as user preferences change during interactions. We introduce ConvoDrift, a dataset designed to model progressive stylistic conversational tone drift under fixed semantic intent. It is built on 15,727 shared multi-turn conversational structures for adaptation and persona-conditioned alignment methods. It consists of six prompt-response pairs per conversation, each with the annotation of style drift and style direction labels. These pairs cover a range of communication genres. We further derive a complementary pairwise dataset by pairing semantically equivalent but stylistically distinct responses and annotating persona-conditioned preferences using five distinct style communication personas, enabling the controlled study of personali
    
[^66]: 用于大语言模型平衡多任务后训练的自适应互蒸馏

    Adaptive Mutual Distillation for Balanced Multi-Task Post-Training of Large Language Models

    [https://arxiv.org/abs/2610.02856](https://arxiv.org/abs/2610.02856)

    提出自适应互蒸馏框架AMD，通过联合训练两个采用不同任务平衡策略的模型，并利用短训练探针和任务级验证分数动态为每个任务及迁移方向选择蒸馏权重调整方案，从而提升大语言模型多任务后训练的整体性能。

    

    大语言模型（LLM）的多任务后训练旨在提升模型在训练数据量不均衡的各任务上的性能。现有方法主要关注在单模型训练过程中平衡各任务的贡献。不同的任务平衡策略可以产生具有互补优势的模型，这为互蒸馏创造了机会。然而，跨模型监督的有效性可能因任务、迁移方向以及训练阶段的不同而变化。我们提出了自适应互蒸馏（Adaptive Mutual Distillation，AMD），这是一种协作式后训练框架，可联合训练两个采用不同任务平衡策略的模型。AMD通过跨任务共享的短训练探针来评估蒸馏权重的候选调整方案，然后利用按任务划分的验证分数，为每个任务和迁移方向选择合适的调整方案。在六个基准测试和三个大语言模型骨干网络上的实验表明，两个AMD模型均取得了更高的平均基准性能。

    arXiv:2610.02856v1 Announce Type: new  Abstract: Multi-task post-training of large language models (LLMs) aims to improve performance across tasks with unequal amounts of training data. Existing methods focus primarily on balancing task contributions during single-model training. Different task-balancing strategies can produce models with complementary strengths, creating opportunities for mutual distillation. However, the usefulness of cross-model supervision can vary across tasks, transfer directions, and stages of training. We propose Adaptive Mutual Distillation (AMD), a collaborative post-training framework that jointly trains two models with different task-balancing strategies. AMD evaluates candidate adjustments to distillation weights through short training probes shared across tasks, then uses task-wise validation scores to select an adjustment for each task and transfer direction. Across six benchmarks and three LLM backbones, both AMD models achieve higher average benchmark 
    
[^67]: 多模态声明验证对LLM改写的鲁棒性如何？

    How Robust Is Multimodal Claim Verification to LLM Rewriting?

    [https://arxiv.org/abs/2610.02841](https://arxiv.org/abs/2610.02841)

    该研究发现多模态声明验证模型对LLM改写具有较强的鲁棒性——多数模型准确率无显著下降，但改写仍会引发一致的概率偏移。

    

    众所周知，大语言模型（LLM）会在生成的文本中引入风格上的变化，但这些风格变化如何影响模型在科学任务上的决策仍未得到充分探索。本文聚焦于多模态声明验证任务，其目标是判断一条文本声明是否有给定证据的支持。我们采用了两种改写策略：自然改写，模拟研究人员日常使用LLM润色学术文本的方式；以及受控注入，即插入一个与LLM相关的单词，以隔离词汇选择带来的影响。我们评估了11个开源权重模型，涵盖五个视觉语言模型（VLM）系列，参数量从2B到38B不等。我们发现模型对这些修改具有鲁棒性：大多数模型的准确率没有显著下降，并且与先前关于评审分数操纵的研究相比，验证任务显得稳定得多。然而，一致的概率偏移确实会出现，面向模糊措辞（hedging）的条件会产生显著的……

    arXiv:2610.02841v1 Announce Type: new  Abstract: LLMs are known to introduce stylistic changes into generated text, yet how these stylistic shifts affect model decisions on scientific tasks remains underexplored. In this paper, we focus on multimodal claim verification, where the goal is to determine whether a textual claim is grounded in a given piece of evidence. We apply two rewriting strategies: natural rewriting, which simulates how researchers routinely use LLMs to polish academic text, and controlled injection, which inserts a single LLM-associated word to isolate the effect of vocabulary choice. We evaluate 11 open-weight models spanning five VLM families and ranging from 2B to 38B parameters. We find that models are robust to these modifications: most show no significant drop in accuracy, and compared to prior work on review-score manipulation, verification appears far more stable. However, consistent probability shifts do occur. Hedging-oriented conditions produce significant
    
[^68]: 探索数据分布之外的奇异新世界：系统行为、因果性税与非因果基础模型

    To Explore The Strange New World Beyond Data Distribution: System Behavior, Causality Tax, and Non-causal Base Model

    [https://arxiv.org/abs/2610.02839](https://arxiv.org/abs/2610.02839)

    该论文提出SBD框架，将数据分布之外的系统行为作为不可约的贝叶斯组件纳入证据下界，从理论上揭示了反直觉的“因果性税”现象，表明语言模型的因果性可能既非必要也非最优。

    

    我们证明，语言模型（LMs）的因果性可能既非必要也非最优。当系统行为（记为 $S$）作为第一性原理贝叶斯特征被纳入考量时，情况便是如此。这里的 $S$ 指的是数据空间之外的额外主导因素，且它们涉及耦合效应。尽管因果性是现代架构事实上的基础，但近期研究表明其与因果性之间存在持续的不匹配和矛盾，而这些问题在很大程度上源于系统行为而非数据分布。因此，我们提出了 SBD 框架，将 $S$ 作为证据下界（ELBO）的一个不可约组件纳入其中。SBD 从理论上揭示了一种反直觉的“因果性税”现象：由于对 $S$ 的忽视，因果性表现为一种带有额外结构性误差的次优近似。为应对潜在变量分析的挑战，我们通过……验证了 SBD 所预测的 $S$ 的影响（摘要在此处截断）。

    arXiv:2610.02839v1 Announce Type: cross  Abstract: We show that the causality of language models (LMs) may not be necessary nor optimal. This is the case when system behavior (denoted as $S$) is incorporated as a first-principle Bayesian feature. Here, $S$ refers to extra dominant factors beyond the data space, and they involve coupled effects. Despite being the de facto foundation of modern architecture, recent studies indicate persistent mismatches and contradictions with causality. These issues largely stem from system behavior rather than the data distribution. We therefore propose the SBD framework, which incorporates $S$ as an irreducible component of the evidence lower bound (ELBO). SBD theoretically reveals a counter-intuitive Causality Tax phenomenon, where causality emerges as a suboptimal approximation with an additional structural error, due to the obliviousness to $S$. To address the challenge of latent variable analysis, we validate the SBD-predicted impact of $S$ via imp
    
[^69]: 大语言模型中的临床概念中心

    Clinical Concept Centers in LLMs

    [https://arxiv.org/abs/2610.02829](https://arxiv.org/abs/2610.02829)

    该论文首次将机制可解释性评估从文本层面扩展到潜在空间应用于临床决策支持，发现在全部十一个测试的开源大语言模型内部都存在专门的临床概念中心，即临床概念以可定位且被因果使用的表征形式存在于模型潜在空间中。

    

    大语言模型在临床环境中的应用日益增多。然而，针对这些模型可靠性与性能的研究几乎完全聚焦于语言层面，即对模型所“说”的内容进行评分。机制可解释性研究发现，潜在空间承载着比文本更高保真度的表征：内部表征所编码的信息远多于输出所表达的内容，而且模型陈述的推理过程也会系统性地遗漏那些因果驱动答案的特征。在临床决策支持领域，基于机制可解释性的模型行为评估尚未被探索。在这项工作中，我们将行为评估扩展到潜在空间，探究临床概念是否作为可定位、且被因果使用的表征存在于开源权重的大语言模型内部。我们在所测试的全部十一个开源模型的潜在空间中都发现了专门的临床概念中心。这些概念中心是内……（摘要在此处被截断）

    arXiv:2610.02829v1 Announce Type: new  Abstract: Large language models are increasingly used in clinical settings. However, research into the reliability and performance of these models has focused almost entirely on the language substrate, scoring what the model says. Mechanistic interpretability has found that the latent space carries a higher fidelity of representation than the text: internal representations not only encode substantially more than the output verbalizes, but the stated reasoning also systematically omits features that causally drive the answer. An evaluation of model behavior in terms of mechanistic interpretability has not been explored in clinical decision support. In this work, we extend behavioral evaluation into the latent space and ask whether clinical concepts exist as locatable, causally used representations inside open-weight LLMs. We find dedicated clinical concept centers in the latent space of all eleven open models we test. These concept centers are inte
    
[^70]: FSPO：面向预算约束的大语言模型强化学习后训练的策略一致风险与帕累托可行控制

    FSPO: Policy-Consistent Risk and Pareto-Feasible Control for Budgeted LLM RL Post-Training

    [https://arxiv.org/abs/2610.02828](https://arxiv.org/abs/2610.02828)

    FSPO 提出策略一致的风险前瞻模型与帕累托可行控制机制，联合解决了预算约束下大语言模型强化学习后训练中风险估计失配、校准漂移和多资源可行性保证三个耦合难题。

    

    自适应的大语言模型强化学习后训练会在训练过程中在线调整多个训练执行器，包括 rollout 温度、组大小、裁剪、KL 正则化、验证器分配以及更新预算。目前有三个相互耦合的问题尚未解决：从行为轨迹训练得到的未来风险模型未必能准确估计将要部署的控制器所引发的风险；基于已记录的状态-动作对校准的分数在经过选择性动作选择后可能出现校准失准；独立的单资源最小成本通常无法保证多资源延续的可行性。我们提出 FSPO，一种面向预算约束的大语言模型强化学习后训练的反馈状态控制器，能够联合解决这些问题。FSPO 学习一个策略一致的风险前瞻（risk-to-go）模型，其 Bellman 目标遵循与未来决策所用的同一个冻结控制器，并同时学习一个长程效用模型。决策条件化轨迹校准（DCTC）对风险进行校准……

    arXiv:2610.02828v1 Announce Type: new  Abstract: Adaptive LLM reinforcement-learning post-training changes multiple training actuators online, including rollout temperature, group size, clipping, KL regularization, verifier allocation, and update budget. Three coupled issues remain unresolved. A future-risk model trained from behavior trajectories need not estimate the risk induced by the controller that will be deployed; a score calibrated on logged state-action pairs can become miscalibrated after selective action choice; and independent per-resource minimum costs do not in general certify a feasible multi-resource continuation. We introduce FSPO, a feedback-state controller for budgeted LLM RL post-training that addresses these issues jointly. FSPO learns a policy-consistent risk-to-go model whose Bellman target follows the same frozen controller used for future decisions, together with a long-horizon utility model. Decision-conditioned trajectory calibration (DCTC) calibrates risk 
    
[^71]: 面向全模态推理的以文本为中心的后训练

    Text-Centric Post-Training for Omni-Modal Reasoning

    [https://arxiv.org/abs/2610.02819](https://arxiv.org/abs/2610.02819)

    论文发现全模态大模型的多跳推理困难可通过以文本为中心的后训练（先SFT后RL）显著改善，无需音视频数据即让Qwen2.5-Omni-7B的九项推理得分几何平均提升25.83%，同时节省56.6%的GPU小时数并超越完整的音视频训练路线。

    

    在全模态大语言模型中改进音视频联合推理通常需要付出巨大的数据构建和训练成本。我们的诊断分析发现，即使模型能够正确回答所有对应的单跳问题，在多跳推理上仍然存在困难，这表明感知与推理目标在局部优化中存在部分解耦的现象。这一发现启发了我们对这些能力进行具有不同侧重点的后训练。仅使用文本的推理训练在多种数据源、模型规模和模型家族上均带来了提升。采用表现最佳的仅文本配置，监督微调结合强化学习（RL）使Qwen2.5-Omni-7B在九项推理得分上的几何平均值相比基础模型提升了25.83%，并以减少56.6%的GPU小时数超越了完整的原生音视频训练路线。仅使用由纯文本大语言模型合成的数据进行训练，在数据构建过程中完全不使用音视频数据的情况下，将这一几何平均值提升了21.01%……

    arXiv:2610.02819v1 Announce Type: new  Abstract: Improving joint audio-visual reasoning in Omni Large Language Models typically incurs substantial data construction and training costs. Our diagnostics reveal multi-hop reasoning difficulties despite correct answers to all corresponding single-hop questions and suggest partial decoupling in the local optimization of perception and reasoning objectives. This motivates post-training with different emphases on these capabilities. Text-only reasoning training yields gains across data sources, model scales, and families. With the best-performing text-only configuration, supervised fine-tuning followed by reinforcement learning (RL) raises Qwen2.5-Omni-7B's geometric mean of nine reasoning scores by 25.83% over the base model, outperforming the complete native audio-visual route with 56.6% fewer GPU-hours. Training on data synthesized entirely by a text-only LLM raises this geometric mean by 21.01% without audio-visual data in construction or 
    
[^72]: RMCW：一种基于里德-马勒码的语言模型抗删除鲁棒水印

    RMCW: A Deletion-Robust Watermark Based on Reed--Muller Codes for Language Models

    [https://arxiv.org/abs/2610.02817](https://arxiv.org/abs/2610.02817)

    提出了一种基于里德-马勒码的大语言模型水印方法RMCW，通过密钥词汇划分注入水印结构，并利用局部子序列的里德-所罗门代数一致性检验，实现对删除攻击的鲁棒水印检测。

    

    大语言模型（LLM）水印为识别由特定模型生成的文本提供了一种轻量级机制，但其在后处理攻击下的鲁棒性仍然脆弱。删除攻击尤其具有挑战性，因为它们会改变词元的位置，破坏观测到的词元与其原始水印位置之间的对齐关系。我们提出了里德-马勒码水印（RMCW），一种基于里德-马勒码的LLM水印方法。与全局码字恢复不同，RMCW搜索存留的局部代数结构，利用里德-马勒码字的仿射线约束所诱导的里德-所罗门一致性。在生成阶段，RMCW通过密钥词汇划分将里德-马勒结构注入序列；在检测阶段，它将给定文本映射到带密钥的词汇桶，并使用Berlekamp–Welch测试对局部子序列进行低次里德-所罗门一致性检验。

    arXiv:2610.02817v1 Announce Type: cross  Abstract: Large Language Model (LLM) watermarking provides a lightweight mechanism for identifying text generated by a specific model, but its robustness remains fragile under post-processing attacks. Deletion attacks are particularly challenging because they shift token positions and break the alignment between observed tokens and their original watermark positions. We propose Reed--Muller Code Watermarking (RMCW), an LLM watermarking method based on Reed--Muller codes. In contrast to global codeword recovery, RMCW searches for surviving local algebraic structure, leveraging the Reed--Solomon consistency induced by affine-line restrictions of Reed--Muller codewords. During generation, RMCW injects a Reed--Muller structure into the sequence via a secret-keyed vocabulary partition. During detection, it maps the given text to keyed vocabulary bins and tests local subsequences for low-degree Reed--Solomon consistency using Berlekamp--Welch tests. E
    
[^73]: ROUTEAUDIT：面向预算受限多验证器路由的交互感知识别方法

    ROUTEAUDIT: Interaction-Aware Identification for Budgeted Multi-Verifier Routing

    [https://arxiv.org/abs/2610.02808](https://arxiv.org/abs/2610.02808)

    ROUTEAUDIT将预算受限的多验证器路由形式化为契约条件化的识别问题，通过契约格、策略无关响应带和请求级边界三个可度量对象，在验证器目录与可用性随策略变化的情形下实现对路由策略效果的严格归因与因果识别。

    

    自适应多验证器系统通常通过端点的质量-成本差距进行比较，即使验证器目录、可用性、资源核算、信息过滤或评分器会随策略发生变化。我们将验证器路由形式化为一个契约条件化的识别问题。该契约记录了请求支持、验证器目录、实际可用性、资源核算、在线过滤以及轨迹后评分；一个匹配的路由对比仅改变策略坐标。ROUTEAUDIT为该契约增加了三个可度量的对象：契约格在所有可容许的桥接顺序上对坐标增量取平均，并报告由此得到的归因及其路径敏感性；策略无关的响应带在自适应策略揭示不同观测时识别成对的顺序对比；对于不完整的匹配，请求级边界利用仍然可观测的潜在结果，给出紧致的有限（摘要在此处截断）。

    arXiv:2610.02808v1 Announce Type: new  Abstract: Adaptive multi-verifier systems are commonly compared through endpoint quality-cost gaps, even when the verifier catalog, availability, accounting, information filtration, or scorer changes with the policy. We formulate verifier routing as a contract-conditioned identification problem. The contract records request support, verifier catalog, realized availability, resource accounting, online filtration, and post-trace scoring; a matched route contrast changes only the policy coordinate. ROUTEAUDIT adds three measurable objects to this contract. A contract lattice averages coordinate increments over every admissible bridge order and reports the resulting attribution together with its path sensitivity. A policy-independent response tape identifies paired sequential contrasts when adaptive policies reveal different observations. For incomplete matching, request-level bounds use whichever potential outcome remains observed and give a sharp fi
    
[^74]: RL之前的OPD：利用在线策略蒸馏为基于评分标准的强化学习进行热启动

    OPD Before RL: Warm-Starting Rubric-Based RL with On-Policy Distillation

    [https://arxiv.org/abs/2610.02781](https://arxiv.org/abs/2610.02781)

    提出两阶段训练框架：先以评分标准作为教师特权上下文进行在线策略蒸馏（RP-OPD）提供密集的token级监督，再以评分标准作为奖励进行强化学习，从而突破蒸馏的性能瓶颈。

    

    许多有用的语言模型任务无法通过精确的结果验证来评估。基于评分标准的强化学习（RL）通过根据明确标准对开放式回答进行评分来解决这一问题。然而，由于奖励是在完整回答生成之后才分配的，训练信号无法直接识别是哪些具体决策对最终得分做出了贡献。我们提出了一个两阶段训练框架：首先将评分标准用作特权教师上下文以提供密集的token级监督，然后将其用作奖励进行进一步的RL。在第一阶段，评分标准特权在线策略蒸馏（RP-OPD）让无法访问评分标准的学生模型在学生生成的前缀处匹配具备评分标准意识的教师模型的下一个token分布。在第二阶段，RL直接优化评分标准奖励，并突破了蒸馏带来的性能平台期。我们使用开源权重模型在健康和科学任务上评估了该框架。（摘要原文在此处截断）

    arXiv:2610.02781v1 Announce Type: cross  Abstract: Many useful language-model tasks cannot be evaluated by exact outcome verification. Rubric-based reinforcement learning (RL) addresses this issue by scoring open-ended responses against explicit criteria. However, because the reward is assigned after the complete response, the training signal does not directly identify which individual decisions contributed to the final score. We propose a two-stage training framework that uses rubrics first as privileged teacher context for dense token-level supervision, then as rewards for further RL. In the first stage, rubric-privileged on-policy distillation (RP-OPD), a student without access to the rubric matches a rubric-aware teacher's next-token distributions at student-generated prefixes. In the second stage, RL directly optimizes the rubric reward and improves beyond the observed distillation plateau. We evaluate the framework on health and science tasks using open-weight models. Across Heal
    
[^75]: 在线交流中心理健康污名的自动评估

    Automatic Evaluation of Mental Health Stigma in Online Communication

    [https://arxiv.org/abs/2610.02775](https://arxiv.org/abs/2610.02775)

    该论文提出了一个基于理论的细粒度污名标注基准，利用真实的在线新闻和社交媒体文本对多种心理健康状况的污名进行自动评估，并比较了大语言模型与传统情感、毒性、仇恨言论分类器的检测表现。

    

    心理健康污名具有深远的危害性，但其复杂性使其难以评估。污名可能表现为明显的贬低，但也包括更微妙的形式，如责备、恐惧、家长式的怜悯、社会疏离、结构性排斥和歧视。我们提出了一个基于理论的基准，用于自动评估在线交流中的心理健康污名，该基准由自然产生的在线新闻和社交媒体文本组成，并依据跨多种心理健康状况的细粒度污名分类体系进行了标注。我们的标注框架包括一个二元污名检测任务和一个多层次分类体系，涵盖 (i) 污名模式、(ii) 领域，以及 (iii) 某些污名形式的具体组成部分。我们将该框架应用于提及六种心理健康状况的文本，并评估了大型语言模型以及用于检测情感、毒性和仇恨言论的污名相关分类器。结果显示，心理健康污……

    arXiv:2610.02775v1 Announce Type: new  Abstract: Mental health stigma has profoundly harmful impacts but its complexity makes it difficult to evaluate. Stigma may involve explicit derogation, but also subtler forms of blame, fear, paternalistic pity, social distancing, structural exclusion, and discrimination. We introduce a theory-grounded benchmark for automatic evaluation of mental health stigma in online communication, consisting of naturally occurring online news and social media text annotated with a fine-grained taxonomy of stigma across multiple mental health conditions. Our annotation framework comprises a binary stigma-detection task and a multi-level taxonomy covering (i) stigma mode, (ii) domain, and (iii) specific components of certain forms of stigma. We apply this framework to texts mentioning six mental health conditions and evaluate large language models alongside stigma-related classifiers for detecting sentiment, toxicity, and hate speech. Results show that mental he
    
[^76]: 通过聚焦视图改进非结构化知识编辑中的原子事实回忆

    Improving Atomic-Fact Recall via Focused Views in Unstructured Knowledge Editing

    [https://arxiv.org/abs/2610.02772](https://arxiv.org/abs/2610.02772)

    该论文揭示了非结构化知识编辑中段落级编辑目标导致的“难度低估”问题，并提出通过聚焦视图的方式改进编辑后的模型，使其无需原始段落上下文即可可靠地回忆编辑文本中的各个原子事实。

    

    大型语言模型（LLM）日益成为事实知识的通用接口，但其参数并不能自动反映预训练之后发生变化的信息。知识编辑通过修改选定的知识、同时保留无关知识和通用能力，为代价高昂的重新训练提供了一种有针对性的替代方案。传统知识编辑使用结构化的事实三元组，而非结构化知识编辑（UKE）则使用包含多个事实的自由形式文本段落。然而，现有的UKE编辑器表现出一种被称为“上下文依赖”的失败模式：经过编辑的LLM通常能够复述编辑文本段落，但在没有原始段落上下文的情况下，却无法可靠地回忆其中的各个事实。我们发现在标准的段落级编辑目标下存在“上下文导致的难度低估”问题：越靠后的事实获得的真实上下文越丰富，因而产生较低的初始损失，使它们看起来更容易……（摘要原文在此处截断）

    arXiv:2610.02772v1 Announce Type: cross  Abstract: Large language models (LLMs) increasingly serve as general-purpose interfaces to factual knowledge, but their parameters do not automatically reflect information that changes after pretraining. Knowledge editing (KE) provides a targeted alternative to costly retraining by modifying selected knowledge and preserving unrelated knowledge and general capabilities. Conventional KE uses structured factual triples, whereas unstructured KE (UKE) uses free-form passages containing multiple facts. Nonetheless, existing UKE editors exhibit a failure mode known as context reliance: edited LLMs can often reproduce the editing passage but fail to reliably recall its individual facts without the original passage context. We identify context-induced difficulty underestimation under the standard passage-level editing objective: later facts receive increasingly rich ground-truth context and consequently incur lower initial losses, making them appear eas
    
[^77]: AptMQL-Bench：基于访问模式的模式设计与数据保真迁移，从Text-to-SQL走向Text-to-MQL

    AptMQL-Bench: From Text-to-SQL to Text-to-MQL via Access-Pattern Schema Design and Data-Preserving Migration

    [https://arxiv.org/abs/2610.02770](https://arxiv.org/abs/2610.02770)

    该论文提出AptMQL-Bench，一种由编码智能体驱动、人工校验的转换流水线，通过基于访问模式设计文档模式并保真迁移数据，将text-to-SQL基准高质量地转换为text-to-MQL基准，克服了现有启发式转换方法数据丢失与查询性能低下的缺陷。

    

    像MongoDB这样的文档数据库是现代应用的核心基础设施，而面向它们的自然语言接口——text-to-MQL——将使非专业用户无需掌握查询语言即可查询复杂的半结构化数据。这一任务的进展依赖于高质量的基准测试，而将现有的text-to-SQL基准转换到文档数据库场景是获取此类基准最切实可行的途径。遗憾的是，现有工作依赖启发式方法进行机械化转换：文档模式直接照搬关系型外键图，每条查询也简单地对应其源SQL。实验结果显示，这些方法在BIRD的21个数据库中完全无法迁移6个，在其他数据库上会静默丢失多达25.9%的数据行，且所得模式使得标准答案查询随数据规模增长而变慢一个数量级以上。为此，我们提出了一种由编码智能体驱动并结合人工在环验证的转换流水线，它针对每个数据库设计……（原文摘要在此处截断）

    arXiv:2610.02770v1 Announce Type: new  Abstract: Document databases such as MongoDB are core infrastructure for modern applications, and natural-language interfaces to them---text-to-MQL---would let non-experts query complex, semi-structured data without mastering the query language. Progress on this task depends on high-quality benchmarks, which are most practically obtained by converting an existing text-to-SQL benchmark to the document setting. Unfortunately, existing efforts rely on heuristics for mechanical conversion: the document schema mirrors the relational foreign-key graph, and each query mirrors its source SQL. As a result in our experiments, these approaches fail to migrate 6 of 21 BIRD databases outright, silently drop up to 25.9\% of rows on others, and yield schemas whose ground-truth queries run over an order of magnitude slower as the data scales. We instead propose a conversion pipeline, driven by coding agents with human-in-the-loop verification, that designs each d
    
[^78]: 当历史未能转化为经验：语言智能体中的动作校准

    When History Fails to Become Experience: Action Calibration in Language Agents

    [https://arxiv.org/abs/2610.02769](https://arxiv.org/abs/2610.02769)

    研究发现语言智能体并不能可靠地将历史动作与其结果相关联，而只需简单地为每条观察标注其对应的前序动作，即可显著提升任务成功率并减少动作重复。

    

    语言智能体应当利用先前的尝试和环境反馈来改进同一任务中的后续决策。然而，提供额外的交互历史有时反而会降低任务成功率，这表明智能体并不能始终有效地利用这些信息。为了研究这一局限性，我们考察了智能体如何使用历史信息。我们发现，历史信息总体上能够提升任务完成率，但其中很大一部分收益即使在过去的动作被打乱时依然存在。破坏动作与观察之间的对应关系仅导致任务成功率出现轻微下降。因此我们假设，智能体在决定如何进行下一步时，并不能可靠地将过去的动作与其结果联系起来。为了验证这一假设，我们明确地将每条返回的观察标注为前一个动作的结果。这一简单的标注提升了任务成功率，并减少了下一步动作的重复，且未引入任何新的环境信息。

    arXiv:2610.02769v1 Announce Type: cross  Abstract: Language agents should draw on prior attempts and environmental feedback to improve subsequent decisions within the same task. However, providing additional interaction history can sometimes reduce task success, suggesting that agents do not consistently use this information effectively. To investigate this limitation, we examine how agents use history. We find that history improves task completion overall, yet much of this benefit persists even when past actions are shuffled. Disrupting the correspondence between actions and observations causes only a modest decline in task success. We therefore hypothesize that agents do not reliably connect past actions with their outcomes when deciding how to proceed. To test this hypothesis, we explicitly label each returned observation as the outcome of the preceding action. This simple annotation improves task success and reduces next-action repetition without introducing new environmental infor
    
[^79]: EpiWorld：将大语言模型政策智能体锚定于流行病学世界模型

    EpiWorld: Grounding LLM Policy Agents in Epidemiological World Models

    [https://arxiv.org/abs/2610.02744](https://arxiv.org/abs/2610.02744)

    提出了 EpiWorld 闭环框架，将大语言模型政策智能体锚定于动作条件化的流行病学世界模型和分层公共卫生技能库，借助快速反事实推演实现流行病干预政策的选择与迭代优化。

    

    流行病干预政策是文本形式的人工产物，人类决策者通过自然语言对其进行解读、论证与修订，这使得大语言模型成为流行病政策推理的天然候选者。然而，原始的大语言模型缺乏预测干预后果所需的流行病动力学知识、评估严重程度所需的定量监测信号，以及界定可采纳行动的制度性约束。我们提出了 EpiWorld，这是一个闭环框架，它将大语言模型政策执行者锚定在一个学习得到的动作条件化流行病学世界模型上，并配备一个分层技能库，其中包含公共卫生规程、监测工具以及通过事后分析积累的自适应经验。给定一个候选干预措施，世界模型可以预测区域疫情的演变，并支持快速的反事实推演，为政策选择和改进提供反馈。模拟未来情景的结果将被提炼（摘要在此处截断）

    arXiv:2610.02744v1 Announce Type: new  Abstract: Epidemic intervention policies are textual artefacts that human decision-makers interpret, justify, and revise through natural language, making large language models a natural candidate for epidemic policy reasoning. A naive LLM, however, lacks the epidemic dynamics needed to project intervention consequences, the quantitative surveillance signals required to assess severity, and the institutional constraints that define admissible actions. We present EpiWorld, a closed-loop framework that grounds an LLM policy actor in a learned action-conditioned epidemiological world model and a tiered skill library of public-health protocols, surveillance tools, and adaptive lessons accumulated through after-action analysis. Given a candidate intervention, the world model predicts regional epidemic evolution and enables fast counterfactual rollouts that provide feedback for policy selection and refinement. Outcomes of simulated futures are distilled 
    
[^80]: 超越正确性：解决智能体化Text-to-SQL中的欠明确问题

    Beyond Correctness: Resolving Underspecification in Agentic Text-to-SQL

    [https://arxiv.org/abs/2610.02739](https://arxiv.org/abs/2610.02739)

    该论文发现Text-to-SQL智能体常因过早终止澄清而暗中做出未经核实的假设，并提出PlanPool方法，将澄清计划外化为一个可变的问题池，强制每个规划好的问题被明确提问或明确舍弃，从而真正解决查询的欠明确性。

    

    智能体化Text-to-SQL系统可以在生成SQL之前与用户交互，以澄清欠明确的查询。然而，正确的执行结果并不一定意味着智能体已经充分解决了潜在的欠明确性：智能体可能会暗中做出未经核实的假设，而这些假设恰好与预期答案相符。我们表明，这种行为部分源于澄清过程的过早终止。尽管强迫智能体提出更多问题可以提高执行准确率，但歧义主要集中在早期的交互中，这使得蛮力式提问效率低下。更重要的是，即使明确提示智能体规划其澄清过程，它也经常放弃自己已经认定相关的问题。为了解决这一失效模式，我们提出了PlanPool，它将澄清计划外化为一个可变的问题池，每一个计划中的问题都必须被明确提出或被明确舍弃……（原文在此截断）

    arXiv:2610.02739v1 Announce Type: new  Abstract: Agentic Text-to-SQL systems can interact with users to clarify underspecified queries before generating SQL. However, a correct execution result does not necessarily imply that the agent has adequately resolved the underlying underspecification: the agent may silently make unverified assumptions that happen to match the intended answer. We show that this behavior is driven in part by premature clarification termination. Although forcing an agent to ask more questions improves execution accuracy, ambiguities are concentrated in earlier interactions, making brute-force questioning inefficient. More importantly, even when explicitly prompted to plan its clarification process, the agent frequently abandons questions that it has already identified as relevant. To address this failure mode, we introduce PlanPool, which externalizes the clarification plan as a mutable question pool. Every planned question must be explicitly asked or dropped bef
    
[^81]: TPBench：一个面向对话压缩的转折点基准

    TPBench: A Turning-Point Benchmark for Dialogue Compression

    [https://arxiv.org/abs/2610.02736](https://arxiv.org/abs/2610.02736)

    该论文提出 TPBench 基准，通过在相同保留预算下探测用户的初始目标、修改后槽位的当前值等互补信息目标，揭示了对话压缩中被整体保留分数掩盖的“转折点丢失”失败模式。

    

    一个压缩器可以保留对话中的事实，却仍然丢掉了改变这些事实的那个回合。用户纠正了一个价格、推翻了一个选择，或者添加了一个约束条件。我们将这种失败称为“转折点丢失”。单一的整体保留分数会掩盖这种失败，因为该分数将用户最初想要的内容与用户现在想要的内容混在了一起。我们提出了 TPBench，它在相同的标称保留预算下评估三种互补的信息目标。P1 要求回答用户的初始目标。P2 要求回答用户修改过的某个槽位的当前值。P3 则要求同时回答两者，所用对话中含有较晚被标注的槽位更新。当前值的答案来自 MultiWOZ 和 SGD 的人工对话状态标注；初始目标的答案是第一个用户轮次的第一句话。两者都不需要新的众包标注。针对特定探测点的评估对压缩方法的排名各不相同。在保留比例为 0.30 的联合探测中，所有被测试的压缩方法……（原文摘要在此截断）

    arXiv:2610.02736v1 Announce Type: cross  Abstract: A compressor can keep the facts of a dialogue and still drop the turn that changed them. A user corrects a price, reverses a choice, or adds a constraint. We call this failure turning-point eviction. One overall retention score hides it, because that score mixes what the user first wanted with what the user wants now.   We introduce TPBench, which evaluates three complementary information targets at shared nominal retention budgets. P1 asks for the user's initial goal. P2 asks for the current value of a slot the user revised. P3 asks for both, in dialogues with a late annotated slot update. The current-value answers come from the human dialogue-state annotations of MultiWOZ and SGD. The initial-goal answer is the first sentence of the first user turn. Neither requires new crowdsourcing.   The probe-specific evaluations rank compression methods differently. On the joint probe at a retained fraction of 0.30, every tested compressed metho
    
[^82]: WakeKV：面向会“改变主意”的注意力头的响应式、可逆KV缓存驻留策略

    WakeKV: Reactive, Reversible KV Residency for Heads That Change Their Minds

    [https://arxiv.org/abs/2610.02713](https://arxiv.org/abs/2610.02713)

    WakeKV发现大多数注意力头在生成过程中会动态改变读取行为，并提出一种响应式、可逆的KV缓存驻留策略，将冷却的注意力头迁移到可恢复的CPU储备区而非冻结或永久驱逐，从而在相同内存预算下持续降低缓存未命中率。

    

    大多数KV缓存压缩方法只对注意力头进行一次性分类（离线或在预填充阶段），并在整个生成过程中保持该分类固定不变。我们在三个模型（1.5B-8B）和三种任务场景（大海捞针检索、长思维链以及多轮对话回忆）下，测量了四种模型-场景组合中的注意力头行为，发现大多数注意力头在生成过程中至少会改变一次其读取行为。我们提出了WakeKV，这是一种响应式驻留策略，它将逐渐“冷却”的注意力头迁移到可恢复的CPU储备区，而不是冻结或永久驱逐其状态。在相同的内存或预算条件下，WakeKV在五个模型-场景组合上，以及与三个引用基线方法（SnapKV、均匀R-KV和ReasonAlloc）在四个符合条件的组合上的对比评估中，均始终比冻结分类和破坏性驱逐方法取得更低的未命中率。基于Mistral-7B的FlexiCache/vLLM实现证实了该方法在真实硬件上的收益，提升了……

    arXiv:2610.02713v1 Announce Type: new  Abstract: Most KV-cache compression methods classify attention heads once, either offline or during prefill, and keep this classification fixed throughout generation. Across three models (1.5B-8B) and three regimes (needle retrieval, long chain-of-thought, and multi-turn recall), we measure head behavior on four model-regime combinations and find that most heads change their reading behavior at least once during generation. We introduce WakeKV, a reactive residency policy that moves cooling heads to a recoverable CPU reservoir rather than freezing or permanently evicting their state. At matched memory or budget, WakeKV consistently improves miss rate over frozen classification and destructive eviction, evaluated across five model-regime combinations and over three cited baselines (SnapKV, uniform R-KV, and ReasonAlloc) across four eligible combinations. A FlexiCache/vLLM implementation on Mistral-7B confirms the benefit on real hardware, improving
    
[^83]: 沉默的异议：向多数派屈服的LLM智能体仍然表征着其原始前提

    Silent Dissent: LLM Agents That Yield to the Majority Still Represent Their Original Premise

    [https://arxiv.org/abs/2610.02702](https://arxiv.org/abs/2610.02702)

    研究发现，在多智能体辩论中表面上屈服于多数派的LLM智能体，其内部表征仍然保留着原始的正确前提，说明它们只是改变了口头表述而非真正改变想法。

    

    多智能体辩论越来越多地被用于在LLM智能体之间达成共识，然而智能体常常会屈服于全体一致的多数派。当一个智能体改变它的答案时，它究竟是改变了想法，还是仅仅改变了表述？我们通过两跳事实性问题来研究这个问题，这类问题的中间实体（桥接实体，例如“圣家堂所在国家的首都”中的国家）从未被任何人明确说出。脚本化的同伴扮演Asch从众实验中“同谋者”的角色，一致地断言一个取自另一个具有不同桥接实体的事实的错误答案。在智能体回答的那一刻，我们使用Jacobian透镜从其残差流中读取桥接实体，并与logit lens进行对比。在四个开源权重模型上对保留事实集进行的预注册测试中，Qwen3.5-4B、Qwen3.6-27B和Gemma-4-E4B-it中屈服的智能体仍在输出层之下的预注册层中表征着它们原始的桥接实体（hit@100高于对照实体：0.85、0.2……

    arXiv:2610.02702v1 Announce Type: new  Abstract: Multi-agent debate is increasingly used to reach consensus among LLM agents, yet agents often yield to a unanimous majority. When an agent changes its answer, has it changed its mind or only its statement? We study this with two-hop factual questions whose intermediate entity (the bridge, e.g. the country in "the capital of the country where the Sagrada Familia is located") is never stated by anyone. Scripted peers, in the role of Asch's confederates, unanimously assert a wrong answer taken from another fact with a different bridge. At the moment the agent answers, we read the bridge from its residual stream with the Jacobian lens (J-lens) and, for comparison, the logit lens. In pre-registered tests on held-out facts with four open-weight models, agents of Qwen3.5-4B, Qwen3.6-27B and Gemma-4-E4B-it that gave in still represented their original bridge in the pre-registered layers below the output (hit@100 above a control entity: 0.85, 0.2
    
[^84]: 从演化错误中学习：面向在线策略蒸馏的自适应迭代修复框架

    Learning from Evolving Errors: Adaptive Iterative Repair for On-Policy Distillation

    [https://arxiv.org/abs/2610.02700](https://arxiv.org/abs/2610.02700)

    该论文提出AIR-OPD框架，通过引导生成器针对学生模型不断演化的错误迭代合成修复引导，并让教师模型以该引导为特权上下文提供监督，从而实现“错误到修复”的在线策略蒸馏，避免了仅依赖参考解所带来的捷径风险。

    

    在线策略自蒸馏（OPSD）在从学生模型自身策略采样的轨迹上提供密集的token级反馈，这是一种比强化学习的结果级奖励更丰富的训练信号。这种反馈来自一个以完整参考解为条件的教师模型，而学生模型无法获得该参考解。参考解只指定了目标，却没有说明如何从学生当前的错误逐步走向目标，从而产生了一种“解条件捷径”风险。我们提出AIR-OPD，一个面向在线策略蒸馏的自适应迭代修复框架，能够提供从错误到修复的监督。给定一个失败的响应，引导生成器会针对当前错误合成修复引导；学生模型在该引导下进行在线策略重试；如果重试仍然不正确，生成器会针对新观察到的错误生成新的修复引导。在每一轮中，一个固定的教师模型接收该引导作为特权上下文并对学生进行监督……

    arXiv:2610.02700v1 Announce Type: cross  Abstract: On-policy self-distillation (OPSD) supplies dense token-level feedback on trajectories sampled from the student's own policy, a richer training signal than the outcome-level rewards of reinforcement learning. This feedback comes from a teacher conditioned on a full reference solution unavailable to the student. The reference solution specifies the target but not how to move from the student's current error toward it, creating a solution-conditioned shortcut risk. We introduce AIR-OPD, an adaptive iterative repair framework for on-policy distillation that provides error-to-repair supervision. Given a failed response, a guidance generator synthesizes repair guidance for the current error. The student samples an on-policy retry with this guidance. If the retry remains incorrect, the generator produces new repair guidance for the newly observed error. At each round, a fixed teacher receives the guidance as privileged context and supervises
    
[^85]: 大语言模型在患者证据演变时表现出不可靠的临床判断更新

    Large language models exhibit unreliable updating of clinical judgment as patient evidence evolves

    [https://arxiv.org/abs/2610.02684](https://arxiv.org/abs/2610.02684)

    该研究发现大语言模型在患者证据演变时无法可靠地更新临床判断，具体表现为对病情恶化证据反应过强的不对称性以及先验信念对预测的因果性干扰，且提示工程无法修复这些问题。

    

    大语言模型（LLM）在临床推理中的应用正受到越来越多的探索，但它们在患者证据演变时能否适当地修正判断仍不清楚。我们利用电子健康记录中匹配的重症监护轨迹评估了纵向的信念更新。在多种大语言模型中，当估计值发生变化时，以先前的判断为条件更多时候是增加而非减少预测误差，这一结果在第二个终点上也得到了复现。受控干预揭示了两种失败模式。第一，在固定先前评估的情况下，模型对恶化的呼吸证据的反应比匹配的改善证据更强烈；在中度和强证据水平上，经过余量归一化后这种不对称性仍然存在。第二，在固定当前证据的情况下，将先验风险从10%提高到90%会使估计值偏移26.2个百分点，证明了先验模型信念的因果影响。提示工程无法恢复可靠的更新。

    arXiv:2610.02684v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly explored for clinical reasoning, but whether they appropriately revise judgments as patient evidence evolves remains unclear. We evaluated longitudinal belief updating using matched intensive-care trajectories from electronic health records. Across diverse LLMs, conditioning on a preceding judgment more often increased than reduced prediction error when estimates changed, replicated for a second endpoint. Controlled interventions revealed two failure modes. First, with preceding assessment fixed, models responded more strongly to worsening than matched improving respiratory evidence; this asymmetry persisted after headroom normalization at moderate and strong evidence levels. Second, with current evidence fixed, increasing prior risk from 10% to 90% shifted estimates by 26.2 percentage points, demonstrating causal influence of prior model beliefs. Prompting did not restore reliable updating. 
    
[^86]: Asterism：探索并将分散的观察结果综合为基于文献的假设与理论

    Asterism: Exploring and Synthesizing Scattered Observations into Literature-Grounded Hypotheses and Theories

    [https://arxiv.org/abs/2610.02673](https://arxiv.org/abs/2610.02673)

    Asterism系统通过层次化本体从数百篇论文中提取概念-关系三元组，让研究者能够策划证据图并在不同粒度上聚合观察结果，从而在保留研究者选择与直觉的前提下，将分散的观察结果综合为基于文献的假设与理论。

    

    理论将许多独立的观察结果纳入一个包含新颖假设的统一框架。构建这样一个理论的研究者必须综合散布在众多论文中的观察结果，这些论文描述着相关概念，但往往使用不同的术语。而哪些概念最为重要，还取决于研究者自身的偏好和研究问题。近期的方法利用大语言模型（LLM）来规模化理论综合，但却将研究者的选择和直觉自动化抹除了。我们提出了Asterism，该系统从数百篇论文中提取观察结果，形成概念-关系三元组，并将概念统一在一个层次化本体中。研究者可以借助该本体来策划证据图，并在不同粒度级别上聚合观察结果，从而将理论构建聚焦于特定感兴趣的现象。在实际部署中（n=10），研究者从观察结果逐步推进到理论，并保留了符合自身偏好的概念和假设。在两个案例研究中，免疫学团队……（摘要原文截断）

    arXiv:2610.02673v1 Announce Type: cross  Abstract: A theory draws many independent observations into one framework with novel hypotheses. A researcher building such a theory must synthesize observations scattered across many papers, each describing related concepts but often in different terms. Which concepts matter most also depends on their preferences and research questions. Recent approaches scale theory synthesis with LLMs, but automate away choices and intuitions from researchers. We present Asterism, which extracts observations from hundreds of papers as concept-relation triples, with concepts unified in a hierarchical ontology. Researchers curate an evidence graph using the ontology and aggregate observations at different levels of granularity to focus theory formation on specific phenomena of interest. In a field deployment (n=10), researchers worked from observations to theories, and kept concepts and hypotheses fitting their preferences. In two case studies, teams of immunol
    
[^87]: LEAP：为LLM智能体学习高效的动作提议

    LEAP: Learning Efficient Action Proposals For LLM Agents

    [https://arxiv.org/abs/2610.02670](https://arxiv.org/abs/2610.02670)

    该论文提出LEAP方法，通过学习一个高效的动作提议模型（而非使用现成的通用模型）为LLM智能体起草动作，并建立延迟分析框架揭示决定动作投机端到端加速的关键因素，从而显著提升智能体执行任务的速度。

    

    LLM智能体在执行任务时速度较慢。智能体一步接一步地完成任务：在每一步中先进行推理，然后选择一个动作去执行，下一步必须等上一步完成后才能开始。投机解码通过起草并验证推理token来加速推理阶段的执行。近期的工作也开始在动作阶段应用类似的思想：使用现成模型（通常较大）为目标模型起草动作提议，再由目标模型进行验证。大型起草模型与目标的匹配率更高，但生成提议耗时更长；而小型现成模型虽然速度快，却很少做出与目标一致的决策。我们提出了一个更普遍的问题：是什么决定了动作投机端到端的加速？为回答这一问题，我们为投机轮次建立了一个延迟分析框架，该框架比较每一轮投机所获得的收益与其付出的成本。收益取决于（摘要在此处被截断）

    arXiv:2610.02670v1 Announce Type: cross  Abstract: LLM agents are known to be slow in rollouts. An agent completes a task one step at a time. At each step, it reasons and then chooses an action to execute. The next step and action cannot start until the previous one has finished. Speculative decoding accelerates the rollouts at the reason phase by drafting and verifying the inference tokens. Recent works have also started to apply similar ideas at the action phase. These works use off-the-shelf models, usually large, to draft action proposals for target model to verify. Large drafters match the target more often but take longer to propose, while small off-the-shelf models are fast but rarely make the same decision as the target. We ask a more general question: what determines the end-to-end speedup of action speculation? To answer it, we develop a latency framework for the speculative round. The framework compares what a round gains with what it costs. The gain depends on how well the 
    
[^88]: 大语言连续扩散模型

    Large Language Continuous Diffusion Models

    [https://arxiv.org/abs/2610.02665](https://arxiv.org/abs/2610.02665)

    提出了首个大规模（3B/8B）连续扩散语言模型 Sigma，通过可操控的低维潜在轨迹、自回归模型热启动以及无分类器引导等推理技术，在数学推理和编码任务上取得了与离散扩散模型相当的性能。

    

    尽管离散扩散语言模型在快速并行解码方面取得了成功，但其非光滑、高维的空间阻碍了用于推理和推断加速的轨迹操控。为克服这一问题，我们提出了 Sigma，这是首个基于可操控、低维 ODE/SDE 潜在轨迹构建的大规模（3B/8B）连续扩散语言模型。Sigma 通过似然优化以分块方式进行训练，在对高斯扰动的词元嵌入进行联合去噪的同时学习最优的嵌入几何结构。为加速训练，Sigma 利用自回归（AR）模型的预训练权重进行热启动。在推理阶段，我们发现无分类器引导和分数温度对于实现高保真的推理与编码至关重要。在与最先进的离散模型（掩码扩散语言模型和自回归基线）进行的全面数学推理与编码评估中，Sigma 在标准基准上取得了与离散模型相当的性能……

    arXiv:2610.02665v1 Announce Type: cross  Abstract: Despite the success of discrete diffusion language models (dLMs) for fast parallel decoding, their non-smooth, high-dimensional space hinders trajectory steering for reasoning and inference acceleration. To overcome this, we present Sigma, the first large-scale (3B/8B) continuous dLM built on steerable, low-dimensional ODE/SDE latent trajectories. Trained blockwise via likelihood optimization, Sigma jointly denoises Gaussian-corrupted token embeddings while learning an optimal embedding geometry. To accelerate training, Sigma leverages pre-trained weights from autoregressive (AR) models for warm-starting. During inference, we identify classifier-free guidance and score temperature as essential for high-fidelity reasoning and coding. Across comprehensive math reasoning and coding evaluations against state-of-the-art discrete counterparts (masked dLMs and AR baselines), Sigma achieves competitive performance with discrete models on stand
    
[^89]: VERSE：面向智能体框架的经过验证的自我进化优化器

    VERSE: Verified Self-Evolving Optimizer for Agent Harnesses

    [https://arxiv.org/abs/2610.02616](https://arxiv.org/abs/2610.02616)

    提出VERSE，一个经过验证的自我进化优化器，它不仅改进智能体框架，还让优化器自我进化其诊断、编辑与验证流程（如测试草稿编辑、重放故障、扰动可疑步骤），在具备基于执行的验证时取得最佳优化效果。

    

    框架进化可以改进LLM智能体的提示词、工具和工作流程，而优化器自身的工具和流程却往往保持固定。我们研究优化器是否可以通过同时改进其诊断故障、开发编辑和测试效果的方式，来更有效地改进另一个智能体。两个观察结果指导了我们的设计：在一项对照研究中，没有基于执行的验证时，优化器自我进化无法提升性能；而当验证可用时，它在该研究中取得了最佳结果。在五个执行器上，自我进化的优化器为故障分析、验证、训练审计和工作流控制构建了自己的工具。基于这些发现，我们提出了VERSE，一个面向智能体框架的经过验证的自我进化优化器。VERSE允许优化器在提交前测试草稿编辑、重放故障并扰动可疑步骤，同时跨轮次跟踪修复和回归情况。利用这种反馈……

    arXiv:2610.02616v1 Announce Type: new  Abstract: Harness evolution improves an LLM agent's prompts, tools, and workflow, while the optimizer's own tools and procedures often remain fixed. We study whether an optimizer can improve another agent more effectively by also improving how it diagnoses failures, develops edits, and tests their effects. Two observations guide our design. In a controlled study, optimizer self-evolution fails to improve performance without execution-based verification, but achieves the best result of that study when verification is available. Across five executors, self-evolving optimizers build their own tools for failure analysis, verification, training audits, and workflow control. Motivated by these findings, we introduce VERSE, a Verified Self-Evolving optimizer for agent harnesses. VERSE lets the optimizer test draft edits, replay failures, and perturb suspected steps before submission, while tracking fixes and regressions across rounds. Using this feedback
    
[^90]: 从部分语音中学习何时提交：面向端到端同声传译的方法

    Learning When to Commit from Partial Speech for End-to-End Simultaneous Speech Translation

    [https://arxiv.org/abs/2610.02612](https://arxiv.org/abs/2610.02612)

    该论文提出利用模型自身对部分语音波形翻译生成的监督信号来适配语音语言模型，无需转写或人工翻译即可实现端到端同声传译，其中多轮仅追加解码在低延迟下表现更优，配合置信度阈值可获得最广泛且持续有竞争力的质量-延迟权衡。

    

    同声传译必须在源语音尚未结束时就输出有用的目标文本，同时保留所有已提交的内容。我们利用从模型自身对完整波形和部分波形的翻译中派生出的前缀监督，对全语句语音语言模型进行适配，既不需要语音转写文本，也不需要人工翻译。我们比较了单轮强制前缀解码与多轮仅追加解码两种方式，使用置信度阈值来控制推理时的质量-延迟权衡，并通过单独的合成边际改变训练前缀的密度。在FLEURS和CoVoST2三个语言方向上，前缀训练相比未适配模型改进了质量-延迟前沿，且置信度提供了最广泛且持续具有竞争力的运行区间。多轮解码在低延迟场景下通常表现更强；在多轮训练下，提交校准误差总体下降63–68%，在早期前缀阶段下降68–80%，而单轮……

    arXiv:2610.02612v1 Announce Type: new  Abstract: Simultaneous speech translation must emit useful target text before the source is complete while preserving every committed token. We adapt a full-utterance speech language model using prefix supervision derived from its own complete- and partial-waveform translations, requiring neither transcripts nor human translations. We compare single-turn forced-prefix and multi-turn append-only decoding, use a confidence threshold to control the inference-time quality--latency trade-off, and vary the density of training prefixes with a separate synthesis margin. On FLEURS and CoVoST2 in three language directions, prefix training improves quality--latency frontiers over the unadapted model, and confidence provides the broadest consistently competitive operating range. Multi-turn decoding is generally stronger at low latency; under multi-turn training, commit-calibration error falls by 63--68% overall and 68--80% at early prefixes, whereas single-tu
    
[^91]: 因果性如何弥合语义鸿沟

    How Causality Bridges the Semantic Gap

    [https://arxiv.org/abs/2610.02594](https://arxiv.org/abs/2610.02594)

    该论文提出以因果结构替代人类知识来为未命名变量赋予语义，将其形式化为“结构约束的语义对齐”，并构建 CausalBridge 框架，从测量数据（含隐变量）中发现因果图并在其依赖关系约束下求解变量嵌入，从而从变量对其他变量的作用方式中解读其含义。

    

    数值测量捕捉了系统的行为方式，但往往未指明其变量的含义：有些变量被测量却从未被标注，另一些变量则从未被测量。现有方法通过参考人类的一般知识为这些变量赋予语义，但在知识存在之处会继承其偏见，在知识缺失之处则无能为力。我们转而利用因果结构来弥合测量与其含义之间的鸿沟，从变量作用于其他变量的方式中解读其语义。我们将这一过程形式化为“结构约束的语义对齐”：以少量已知名称的嵌入作为锚点，在因果图所蕴含的依赖关系约束下求解每个未命名变量的嵌入。基于此，我们构建了 CausalBridge 框架，该框架从测量数据（包括隐变量）中发现因果图，并在这些依赖关系约束下求解嵌入。（原文摘要在此处截断）

    arXiv:2610.02594v1 Announce Type: cross  Abstract: Numerical measurements capture how a system behaves, but often leave the meanings of its variables unspecified. Some variables are measured but never labeled, and others are never measured at all. Existing methods assign semantics to such variables by consulting general human knowledge, but this inherits its biases where that knowledge exists and offers nothing where it does not. We bridge this gap between measurements and their meanings with causal structure instead, reading a variable's semantics from how it acts on other variables. We formalize this as structure-constrained semantic alignment, in which the embedding of each unnamed variable is solved under the dependence relations implied by the causal graph, with the embeddings of a few known names as anchors. Accordingly, we build CausalBridge, a framework that discovers the causal graph from the measurements, latent variables included, solves for the embeddings under those relati
    
[^92]: 评估大语言模型在时间抽取任务中的多维泛化能力

    Evaluating Multi-Dimensional Generalization of Large Language Models in Temporal Extraction Tasks

    [https://arxiv.org/abs/2610.02549](https://arxiv.org/abs/2610.02549)

    本文系统评估了大语言模型在时间与事件表达抽取任务中的多维泛化能力，发现强基础任务性能通常预示更好的泛化，但该关系在显著分布偏移下减弱，且归纳式提示策略表现最为稳健一致。

    

    时间与事件表达抽取是基础性的时间推理任务，但由于标注歧义、领域敏感性以及模型行为不稳定，该问题依然具有挑战性。现有评估主要关注域内性能，对分布偏移下模型可靠性的洞察有限。我们评估了跨模型家族、架构和推理策略的多种模型配置在四个泛化维度上的表现，考察了从基础性能的迁移、跨维度相关性，以及规模、架构和提示方式的影响。这为提示式大语言模型在时间与事件表达抽取任务中的泛化能力提供了系统性研究。我们发现，强大的基础任务性能通常预示着更好的泛化能力；然而，在显著的分布偏移下，这种关系会减弱。归纳式提示在领域偏移、对抗扰动和组合性变化下表现最为一致稳健。

    arXiv:2610.02549v1 Announce Type: new  Abstract: Time and event expression extraction are fundamental temporal reasoning tasks, but the problem remains difficult due to annotation ambiguity, domain sensitivity, and unstable model behavior. Existing evaluations focus on in-domain performance, offering limited insight into reliability under distribution shifts. We evaluate multiple model configurations across families, architectures, and reasoning strategies over four dimensions of generalization, examining transfer from base performance, cross-dimensional correlations, and the effects of scale, architecture, and prompting. This provides a systematic study of how prompted LLMs generalize in time and event expression extraction tasks. We find that strong base-task performance generally predicts better generalization. However, this relationship weakens under substantial distribution shifts. Inductive prompting performs most consistently across domain shift, adversarial perturbations, compo
    
[^93]: 一种生成语法指导的神经符号框架用于句法歧义消解：来自阿拉伯语限定词短语的证据

    A generative-informed neuro-symbolic framework for syntactic ambiguity resolution: Evidence from Arabic DPs

    [https://arxiv.org/abs/2610.02529](https://arxiv.org/abs/2610.02529)

    该研究提出一种将生成句法学概念与AraBERT相结合的神经符号框架，把阿拉伯语限定词短语的结构歧义消解建模为基于候选的决策任务，在未见评估集上取得了96.88%的准确率，同时揭示了不同挂靠类型之间性能的不对称性。

    

    句法歧义对阿拉伯语自然语言处理（NLP）构成了持续的挑战，尤其是在形态丰富的名词性结构中，多种结构解读可能与相同的表层序列相兼容。本研究提出了一种生成语法指导的神经符号框架，用于消解现代标准阿拉伯语（MSA）限定词短语中的结构歧义。该框架将生成句法学的概念与AraBERT相结合，将歧义表示为一种基于候选的决策任务，其中由语言学动机驱动的备选结构被显式构建，并通过候选条件化的输入表示进行评估。研究结果表明，该模型在未见过的评估集上达到了96.88%的准确率、95.92%的宏F1值、96.83%的加权F1值和93.94%的二分类F1值。类别层面的分析显示出不对称的性能表现，高位/VP挂靠（N1）的召回率为99.71%，而低位/NP/嵌入挂靠（N2）的召回率为89.26%，表明后者的消歧更具挑战性。

    arXiv:2610.02529v1 Announce Type: new  Abstract: Syntactic ambiguity poses a persistent challenge for Arabic NLP, particularly in morphologically rich nominal constructions where multiple structu6ral interpretations may be compatible with the same surface sequence. This study proposes a generatively informed neuro-symbolic framework for resolving structural ambiguity in Modern Standard Arabic (MSA) DPs. The framework integrates generative syntactic notions with AraBERT by representing ambiguity as a candidate-based decision task in which linguistically motivated alternatives are explicitly constructed and evaluated through candidate-conditioned input representations. Findings indicate that the model achieved 96.88% accuracy, 95.92% macro-F1, 96.83% weighted F1, and 93.94% binary F1 on the unseen evaluation set. Class-level analysis revealed asymmetric performance, with recall of 99.71% for High/VP Attachment (N1) and 89.26% for Low/NP/Embedded Attachment (N2), indicating greater diffic
    
[^94]: 排序正确，尺度有误：审计用于职业AI测量的LLM评判器

    Right Order, Wrong Scale: Auditing LLM Judges for Occupational AI Measurement

    [https://arxiv.org/abs/2610.02492](https://arxiv.org/abs/2610.02492)

    该研究提出O*NET-BENCH审计套件，发现LLM评判器虽在回答排序上与人类工人基本一致，但在估计AI输出可接受率时产生3.0%-97.9%的巨大偏差，表明排序准确并不等于可靠的量化测量。

    

    LLM评判器正被越来越多地用于评估AI输出是否满足职场要求，但对回答排序的一致性并不能确立在接受率或职业总体层面的一致性。我们提出了O*NET-BENCH，这是一个基于包含45,796名工人评分的现有调查构建的审计套件，并在4,501个测试评分上评估了来自六个模型系列的33种现有评判器配置。其中25种配置实现了至少0.60的平局感知成对排序准确率，尽管一个经训练拟合的仅基于回答文本的TF-IDF基线几乎与最强评判器表现相当。尽管存在这种排序上的一致性，评判器对回答可接受比例的估计介于3.0%至97.9%之间，而职业匹配的人类工人给出的估计为61.1%。在一个微调模型谱系中，从逐点评分切换到捆绑的少样本/列表式评估协议虽然改善了回答排序，却降低了在任务和职业层面与工人平均评分的一致性；这一反转现象在一个任务子集上得到了复现。

    arXiv:2610.02492v1 Announce Type: new  Abstract: LLM judges are increasingly used to assess whether AI outputs meet workplace requirements, but agreement on response rankings does not establish agreement on acceptance rates or occupational aggregates. We introduce O*NET-BENCH, an audit suite derived from an existing survey of 45,796 worker ratings, and evaluate 33 pre-existing judge configurations across six model families on 4,501 test ratings. Twenty-five configurations achieve tie-aware pair accuracy of at least 0.60, although a train-fitted response-only TF-IDF baseline nearly matches the strongest judge. Despite this ordering agreement, judges estimate that 3.0%-97.9% of responses are acceptable, compared with 61.1% for occupation-matched workers. In one fine-tuned lineage, changing from pointwise scoring to a bundled few-shot/listwise protocol improves response ordering while reducing agreement with worker means at the task and occupation levels; this reversal replicates on a tas
    
[^95]: 从检索到类型化决策：基于生物医学句子编码器的校准“系统一”模型

    From Retrieval to Typed Decisions: Calibrated System One Models from Biomedical Sentence Encoders

    [https://arxiv.org/abs/2610.02486](https://arxiv.org/abs/2610.02486)

    该论文提出SBERT2S1框架，将生物医学检索句子编码器转换为类型化决策模型，并发现检索预训练显著有利于保留检索先验的先验融合残差（PFR）决策头，而对交叉头（C）帮助有限甚至有害。

    

    类型化决策模型能够通过一次前向传播回答关于文本的受模式约束的问题，并返回用于阈值判断的概率。我们探讨为检索任务训练的生物医学句子编码器是否是此类模型的良好起点。我们提出了SBERT2S1，它将Sentence-Transformers编码器转换为双编码器、交叉头（C）和先验融合残差（PFR）决策模型，同时推出了BIODECIDE（一个生物医学类型化决策测试套件）和MEDLINE-S1（从NLM标引中派生的24.3万条训练决策）。在六个父模型-检索器配对上，检索训练改善了含内容选项的零样本匹配。微调之后，其效果取决于决策头的类型：在五组配对和三种训练集规模下，检索训练在15次比较中的10次显著帮助保留检索先验的PFR头，但仅在1次中帮助C头，而在5次中反而损害C头的表现。在两个决策头与五种训练目标的匹配网格实验中，C头优于……（摘要在此处截断）

    arXiv:2610.02486v1 Announce Type: cross  Abstract: Typed decision models answer schema-constrained questions about a text in one forward pass and return probabilities meant to be thresholded. We ask whether biomedical sentence encoders trained for retrieval are good starting points for such models. We present SBERT2S1, which converts Sentence-Transformers encoders into bi-encoder, cross-head (C) and prior-fused residual (PFR) decision models, together with BIODECIDE, a biomedical typed-decision suite, and MEDLINE-S1, 243k training decisions derived from NLM indexing. Across six parent-retriever pairs, retrieval training improves zero-shot matching of content-bearing options. After fine-tuning, its effect depends on the head: across five pairs and three training-set sizes, retrieval training significantly helps PFR, which keeps the retrieval prior, in 10 of 15 comparisons, but helps C in one and hurts it in five. A matched grid of two heads and five training objectives shows that C outp
    
[^96]: APDMem：面向查询自适应长期记忆的智能体控制渐进式披露

    APDMem: Agent-Controlled Progressive Disclosure for Query-Adaptive Long-Term Memory

    [https://arxiv.org/abs/2610.02472](https://arxiv.org/abs/2610.02472)

    APDMem提出了一种智能体控制的分层长期记忆架构，将对话历史组织为四个粒度递进的层次并采用渐进式披露检索，从而根据查询复杂度自适应地平衡检索成本与证据保真度。

    

    个性化LLM助手必须从长对话历史中为不同复杂度的查询检索稀疏证据。我们提出APDMem（智能体控制的渐进式披露记忆），这是一种将渐进式披露应用于记忆检索的分层长期记忆架构。APDMem不依赖平面记忆存储或固定检索粒度，而是将对话历史表示为四个逐级细化的层次：主题摘要、个性化关键事实、轮次级证据笔记和原始消息。在推理时，控制器对记忆层次应用渐进式披露：它首先读取高层摘要，仅在需要时才深入查看更细粒度的证据。这形成了自适应的成本-保真度权衡：简单查询可以提前终止，而复杂的时间性、多跳或精确证据查询则会触发更深入的检索。笔记合成器将检索到的证据转换为查询相关的……

    arXiv:2610.02472v1 Announce Type: cross  Abstract: Personalized LLM assistants must recover sparse evidence from long conversation histories across queries of varying complexity. We introduce APDMem (Agent-controlled Progressive Disclosure Memory), a hierarchical long-term memory architecture that applies progressive disclosure to memory retrieval. Rather than relying on a flat memory store or fixed retrieval granularity, APDMem represents conversation history as four progressively detailed layers: thematic summaries, personalized key facts, turn-level evidence notes, and raw messages. At inference time, a controller applies progressive disclosure to the memory hierarchy: it first reads high-level summaries and drills into finer evidence only when needed. This creates an adaptive cost-fidelity trade-off: simple queries can terminate early, while complex temporal, multi-hop, or exact-evidence queries trigger deeper inspection. A note synthesizer converts retrieved evidence into a query-
    
[^97]: 大语言模型压缩的能力缩减定律

    Capability Scaling-Down Laws for LLM Compression

    [https://arxiv.org/abs/2610.02462](https://arxiv.org/abs/2610.02462)

    该论文系统研究了大语言模型在剪枝、量化和蒸馏压缩下的能力缩减定律，建立了可预测不同压缩配置所导致能力损失的简单关系式，从而显著减少压缩实验所需的测量成本。

    

    大语言模型压缩可以降低推理成本和内存需求，但如何选择压缩方法与配置在很大程度上仍依赖经验，因为相近的资源削减可能造成不同的能力损失。我们系统地研究了大语言模型压缩在剪枝、量化和蒸馏三种方式下的能力缩减定律。我们的框架在数学、代码生成和问答任务上度量能力损失，并将这些度量与模型规模、训练阶段、压缩设置、数据可用性以及训练曝光程度相关联。我们建立了简单的预测关系，并评估其准确性、测量效率以及对未见配置和模型状态的泛化能力。在剪枝层级之间共享密度响应可使拟合剪枝预测器所需的配置测量数量减半：在新的 Pythia 模型状态、预先注册的 OLMo-2 测试状态以及 Wanda 剪枝下，该紧凑关系均能良好匹配（摘要原文在此处截断）。

    arXiv:2610.02462v1 Announce Type: cross  Abstract: LLM compression reduces inference costs and memory requirements, but selecting a method and configuration remains largely empirical because comparable resource reductions can produce different capability losses. We systematically investigate capability scaling-down laws for LLM compression across pruning, quantization, and distillation. Our framework measures capability loss in mathematics, code generation, and question answering, and relates these measurements to model size, training stage, compression settings, data availability, and training exposure. We develop simple predictive relations and evaluate their accuracy, measurement efficiency, and generalization to unseen configurations and model states. Sharing the density response across pruning levels halves the configuration measurements needed to fit a pruning predictor: on new Pythia states, on pre-registered OLMo-2 test states and under Wanda pruning, the compact relation match
    
[^98]: CUE用户模拟器：面向多轮基准测试的校准用户嵌入

    CUEing User Simulators: Calibrated User Embeddings for Multi-Turn Benchmarking

    [https://arxiv.org/abs/2610.02460](https://arxiv.org/abs/2610.02460)

    提出免训练的CUE框架，通过将会话编码为连续嵌入并解码为人设命令来驱动LLM用户模拟器，使模拟用户在成功率和失败模式上与真实用户保持校准一致，从而实现更可靠的智能体多轮交互基准测试。

    

    近期的基准测试依赖用户模拟器来评估AI智能体在多轮交互中的表现。尽管现有的模拟技术在表面上能够忠实还原人类的风格与行为，但具备生态有效性的交互式基准测试还要求：在模拟用户群体与真实用户群体中，智能体在何时以及如何失败上保持一致。我们发现现有模拟器缺乏结果校准，即无法与真实用户与同一智能体交互时观察到的成功率和失败模式相吻合。我们提出了校准用户嵌入，这是一个框架，它既能对观察到的会话进行编码并采样连续表示，又能将其解码为人设命令，从而在无需训练的情况下引导大语言模型充当用户模拟器。基于此，我们评估了用户条件下的历史会话重放，以及为相同任务采样新人设时的聚合指标一致性。在τ²-Bench上，CUE化的模拟器产生更少的可归因于模拟器本身的错误，并且…

    arXiv:2610.02460v1 Announce Type: new  Abstract: Recent benchmarks rely on user simulators to evaluate AI agents in multi-turn interaction. While existing simulation techniques demonstrate surface fidelity to human style and behavior, ecologically valid interactive benchmarking also requires alignment in when and how agents fail across simulated and real user populations. We find that existing simulators lack outcome calibration: agreement with observed success rates and failure patterns when real users interact with the same agent. We introduce Calibrated User Embeddings (CUE), a framework that both encodes observed sessions and samples continuous representations, then decodes them into persona commands to steer LLMs to act as user simulators without training. Through this, we evaluate user-conditioned replay of past sessions and aggregate metric agreement when sampling novel personas for the same tasks. On $\tau^2$-Bench, CUEd simulators commit fewer simulator-attributed errors and m
    
[^99]: FinDialogLens：面向金融聊天室遗漏交易识别的多方对话事件抽取

    FinDialogLens: Event Extraction over Multi-Party Dialogue for Missed-Trade Identification in Financial Chatrooms

    [https://arxiv.org/abs/2610.02455](https://arxiv.org/abs/2610.02455)

    提出FinDialogLens混合LLM流水线，以紧凑的微调分类器作为推理时脚手架，对多方金融聊天对话进行RFQ事件抽取，从而准确识别遗漏交易的最终价格与交易结果，配合GPT-4o分别达到92.1%和94.3%的准确率。

    

    多方金融聊天室对销售与交易专业人士至关重要，但其复杂性使得人工恢复遗漏交易不可行：每个报价请求（RFQ）都是一个事件，其最终价格和交易结果出现在RFQ触发消息（即询价消息）之后的许多条消息中，并与其他参与者并发的RFQ相互交错。我们将该问题建模为多方对话上的事件抽取（EE）任务，并提出FinDialogLens——一种混合式大语言模型（LLM）流水线，其中紧凑的微调分类器充当推理时的脚手架：它们负责检测RFQ触发消息以及价格/交易结果元数据，RFQ级模块对每个事件的RFQ窗口进行切分，交易引擎负责填充论元角色。使用GPT-4o时，FinDialogLens在最终价格和交易结果上分别达到92.1%和94.3%的准确率，优于针对全聊天室的思维链（CoT）提示方法；经过微调的开源LLM仅需3B参数即可达到相当的性能。

    arXiv:2610.02455v1 Announce Type: cross  Abstract: Multi-party financial chatrooms are vital for sales-and-trading professionals, but their complexity makes manual recovery of missed trades infeasible: each Request for Quote (RFQ) is an event whose final price and trade outcome appear many messages after the RFQ-trigger message (the inquiry message), interleaved with concurrent RFQs from other participants. We cast this as event extraction (EE) over multi-party dialogue and present FinDialogLens, a hybrid LLM pipeline in which compact fine-tuned classifiers act as inference-time scaffolds: they detect RFQ-triggers and price/trade outcome metadata, an RFQ-Level Module segments per-event RFQ windows, and a Trade Engine fills argument roles. With GPT-4o, FinDialogLens reaches 92.1% and 94.3% accuracy on final price and trade outcome, respectively, outperforming full-chatroom CoT prompting methods; fine-tuned open-source LLMs with as few as 3B parameters achieve comparable performance with
    
[^100]: 基于逐定理符号验证器的反例生成：模仿何时有害而强化学习何时修复

    Counterexample Generation via Per-Theorem Symbolic Verifiers: When Imitation Hurts and Reinforcement Repairs

    [https://arxiv.org/abs/2610.02444](https://arxiv.org/abs/2610.02444)

    该论文发布SymCE数据集（包含4,707个错误数学猜想及其可执行验证器），发现仅用反例做监督微调会陷入“模仿陷阱”、使真定理识别率从0.27崩溃至0.00，而基于验证器稀疏奖励的强化学习（RLVR）不仅能修复这一退化，还能超越基线达到0.66。

    

    大型语言模型往往能够正向证明一个定理，却无法反驳一个密切相关的错误命题——这种“证伪鸿沟”是监督微调无法弥合的，甚至可能使其进一步恶化。我们将反例生成任务形式化为针对确定性逐定理Python验证器的受约束见证输出问题，并发布了SymCE数据集：一个包含4,707个错误的本科代数与实分析猜想的数据集，每个猜想均配有可执行的验证器。该验证器同时充当奖励函数，使SymCE成为一个训练环境。在此预言机下对Qwen3-4B进行SFT加GRPO训练揭示了一个“模仿陷阱”：仅使用反例的SFT使真定理识别率从0.27骤降至0.00，而采用稀疏的仅结果奖励的RLVR（可验证奖励强化学习）不仅修复了这一问题，还超越基线达到0.66。该崩溃现象在四个随机种子以及Gemma-3-4B模型上均得到复现。稀疏奖励与稠密奖励在域内成功率上统计上无显著差异，但在……上相差33个百分点（原文在此处截断）。

    arXiv:2610.02444v1 Announce Type: cross  Abstract: Large language models often solve a theorem forward yet fail to disprove a closely related false one: a falsification gap that supervised fine-tuning does not close and can actively worsen. We frame counterexample generation as constrained witness emission against a deterministic per-theorem Python verifier, and release SymCE, a corpus of 4,707 false undergraduate-algebra and real-analysis conjectures, each paired with executable verifiers. The verifier also serves as the reward function, making SymCE a training environment. Training Qwen3-4B with SFT followed by GRPO under this oracle reveals an imitation trap: counterexample-only SFT collapses true-theorem recognition from 0.27 to 0.00, while RLVR with a sparse outcome-only reward repairs this and exceeds the base, to 0.66. The collapse replicates across four seeds and on Gemma-3-4B. Sparse and dense rewards yield statistically indistinguishable in-domain success yet diverge by 33 po
    
[^101]: 你是在合成还是在回忆？评估大语言模型在算法代码检索上的表现

    Are you Synthesizing or Recalling? Evaluating LLMs on Algorithmic Code Retrieval

    [https://arxiv.org/abs/2610.02438](https://arxiv.org/abs/2610.02438)

    该论文提出将大语言模型对知名算法的代码生成重新定义为“参数化代码检索”任务，并引入AlgoREval基准（涵盖599个问题、77个经典算法、7种编程语言和4种图输入表示）来独立评估这一能力，发现不同语言和输入表示之间的检索准确率差异显著。

    

    大语言模型（LLMs）在代码生成方面已展现出强大的性能，其成功既依赖于回忆相关的算法知识，也依赖于推理如何应用这些知识。然而，现有的LLM处理流程是不透明的，没有对这两个组成部分进行显式区分。我们认为，对于其规范实现在预训练语料库中广泛可获取的知名算法而言，代码生成更适合被衡量为“参数化代码检索”（parametric code retrieval）：即从内化知识中复现一个被指定名称的算法，而非合成一个全新的算法。我们引入了AlgoREval，一个包含599个问题的基准，涵盖14个领域中的77个经典算法、7种编程语言和4种图输入表示，以在隔离环境中评估这一能力，并在零样本设置下评估了15个模型（7B–34B参数）。我们发现，不同语言和输入表示之间的检索准确率存在显著差异。

    arXiv:2610.02438v1 Announce Type: cross  Abstract: Large language models (LLMs) have demonstrated strong performance in code generation, where success depends on both recalling relevant algorithmic knowledge and reasoning about how to apply it. However, existing LLM pipelines are opaque, with no explicit separation between these two components. We argue that for well-known algorithms whose canonical implementations are widely accessible in pretraining corpora, code generation is better measured as \textit{parametric code retrieval}: reproducing a named algorithm from internalised knowledge rather than synthesizing a novel one. We introduce AlgoREval, a benchmark of 599 problems spanning classical 77 algorithms across 14 domains, 7 programming languages, and 4 graph-input representations to evaluate this capability in isolation, and assess 15 models (7B--34B parameters) in a zero-shot setting. We find substantial variation in retrieval accuracy across languages and input representations
    
[^102]: 评估与提升大语言模型对输入序列变化的鲁棒性

    Evaluating and Improving the Robustness of Large Language Models to Input Sequence Variations

    [https://arxiv.org/abs/2610.02432](https://arxiv.org/abs/2610.02432)

    本论文提出了基于Jensen-Shannon散度的生成式鲁棒性度量R_stab，并开发了自适应进化黑盒攻击方法ASA（对LLM-as-a-Judge系统攻击成功率高达73.8%），用于系统性地评估和提升大语言模型对提示注入、木马后门等对抗性输入序列变化的鲁棒性。

    

    生产系统中的大语言模型（LLM）面临提示注入、木马（后门）攻击以及自动质量指标被操纵等威胁。本论文开发了用于评估和提升大语言模型对对抗性输入序列变化鲁棒性的模型、方法和算法。我们提出了R_stab(f)，一种基于小输入扰动下逐步输出分布之间Jensen-Shannon散度的生成式鲁棒性度量。对于局部化攻击，我们证明了V(h) <= 1 - R_class(h)，其中R_class(h)是决策算子h在小扰动下保持其决策的概率。对于非局部化攻击，我们提出了一个经过校准的经验模型。针对LLM-as-a-Judge（大模型作裁判）系统，我们开发了ASA，一种自适应进化黑盒攻击，其攻击成功率（ASR）最高可达73.8%，在开源模型之间的迁移攻击成功率最高可达62.6%。在Trojan Detection Challenge 2023数据（Pythia-1.4B）上，代理触发器达到REA（摘要在此处被截断）

    arXiv:2610.02432v1 Announce Type: cross  Abstract: Large language models (LLMs) in production systems face prompt injections, trojans (backdoors), and manipulation of automatic quality metrics. This thesis develops models, methods, and algorithms for evaluating and improving LLM robustness to adversarial input sequence variations. We propose R_stab(f), a generative robustness metric based on the Jensen-Shannon divergence between per-step output distributions under small input perturbations. For localized attacks we prove V(h) <= 1 - R_class(h), where R_class(h) is the probability that a decision operator h keeps its decision under small perturbations. For non-localized attacks we propose a calibrated empirical model. For LLM-as-a-Judge systems we develop ASA, an adaptive evolutionary black-box attack that reaches an attack success rate (ASR) of up to 73.8%, with transfer between open models up to 62.6%. On Trojan Detection Challenge 2023 data (Pythia-1.4B), surrogate triggers reach REA
    
[^103]: 找到一步好棋不等于赢下棋局：用于LLM智能体闭环评估的XiangqiBench

    Finding the Move Is Not Winning the Game: XiangqiBench for Closed-Loop Evaluation of LLM Agents

    [https://arxiv.org/abs/2610.02425](https://arxiv.org/abs/2610.02425)

    论文提出XiangqiBench——一个基于中国象棋的可执行闭环评估基准，要求LLM智能体在引擎防守对抗下将战术残局的计划真正执行到完成将杀，并发现“走出参考第一手”和“pass@3”等静态指标严重高估了智能体真正闭环完成任务的能力。

    

    静态评估会因语言模型说出正确的一步棋而给予其肯定，但一个智能体必须在对手实时应对的同时，把计划贯彻到经过验证的结果。我们提出XiangqiBench，一个在中国象棋中衡量这种差异的可执行基准：从119个由引擎或仅将军搜索验证的、存在强制将杀的战术残局出发，LLM智能体必须对一个引擎防守方实际完成将杀。一个交互式REPL界面将真实走子、状态查询和前向模拟区分开来，我们在两种观察协议下记录了来自12个前沿LLM的8,568条多轮轨迹。三种看似体现能力的信号各自夸大了闭环成功率。(i) 转化差距：在有视觉（Sighted）试验中，模型在26.1%的情况下走出了存储的参考第一手，但这些试验中只有13.9%最终获胜。(ii) 一致性差距：表现最好的模型达到38.7%的pass@3，但pass^3仅为5.9%，只在7个……（原文在此处截断）

    arXiv:2610.02425v1 Announce Type: new  Abstract: Static evaluations credit a language model for naming the right move, but an agent must carry a plan through to a verified outcome while an opponent responds. We introduce XiangqiBench, an executable benchmark that measures this difference in Chinese chess: starting from 119 tactical endgames with forced mates supported by engine or checks-only search, an LLM agent must deliver checkmate against an engine defender. An interactive REPL interface separates real moves, state queries, and forward simulation, and we record 8,568 multi-turn trajectories from 12 frontier LLMs under two observation protocols. Three signals that look like competence each overstate closed-loop success. (i) The Conversion Gap: models play the stored reference first move in 26.1\% of Sighted trials, yet only 13.9\% of these trials end in a win. (ii) The Consistency Gap: the leading model reaches 38.7\% pass@3 but only 5.9\% pass^3, winning all three trials on 7 of t
    
[^104]: 可训练的智能体式上下文管理

    Trained Agentic Context Management

    [https://arxiv.org/abs/2610.02404](https://arxiv.org/abs/2610.02404)

    通过在最简化的智能体框架（自我调用工具与上下文读取工具）上微调小模型，模型仅用8K词元上下文即可在长文档基准上媲美拥有1M词元上下文的GPT-5.4。

    

    我们研究长上下文语言模型。不同于原生训练长上下文能力或设计复杂的长上下文框架，我们在最简单的智能体框架上训练模型：该框架仅包含两个工具——一个可以用任意指定提示词调用自身的工具，以及一个可以读取输入上下文中指定范围内词元的工具。我们利用这一框架在多样化的合成数据集上对Qwen3.6-35B-A3B进行了微调。在仅有8,000词元上下文的条件下，当文档长度超过40K词元时，我们的小模型在OOLONG-synth基准测试上的表现与拥有1M词元上下文的GPT-5.4相当。

    arXiv:2610.02404v1 Announce Type: cross  Abstract: We study long context language models. Instead of training long context natively, or designing a long context harness, we train a model over the simplest possible harness: a tool to call itself with any specified prompt and a tool to read tokens in a range from the input context. We finetune Qwen3.6-35B-A3B on a diverse synthetic dataset using this harness. With only 8,000 tokens of context, our small model is as strong as GPT-5.4 with 1M tokens of context on the OOLONG-synth benchmark when document length exceeds 40K tokens.
    
[^105]: 犹豫有其几何结构：用于稀疏激活引导的熵训练双曲探针

    Hesitation Has a Geometry: Entropy-Trained Hyperbolic Probes for Sparse Activation Steering

    [https://arxiv.org/abs/2610.02391](https://arxiv.org/abs/2610.02391)

    该论文提出双曲熵引导方法（HEST），以模型自身的下一个词元熵作为唯一标签训练双曲空间中的轻量探针，仅在模型“犹豫”的高熵词元处沿测地线对隐藏状态进行稀疏引导，从而更契合推理过程固有的树状层级结构。

    

    当大语言模型求解一个数学问题时，其推理过程在很大程度上是分层的，而解答往往在少数几个下一个词元熵较高的词元处发生分支。这种树状结构嵌入双曲空间所产生的失真远低于欧几里得空间。然而，现有的激活引导方法通常通过在每个词元处添加一个固定的欧几里得向量来编辑预训练模型的隐藏状态，尽管解答中的大多数词元其实已由上下文所确定。我们提出双曲熵引导（HEST），它利用一个轻量级探针将隐藏状态嵌入庞加莱球中，而该探针的唯一标签是模型自身的下一个词元熵。当该熵超过某个阈值时，HEST 会沿着探针读出值的最陡下降测地线移动嵌入状态，并将这一变化映射回隐藏状态。对于学习到的理想点的 Busemann 读出，我们证明了固定长度的步骤……（原文摘要在此处截断）

    arXiv:2610.02391v1 Announce Type: cross  Abstract: When a large language model solves a mathematical problem, its reasoning is largely hierarchical, and the solution often branches at a few tokens where the next-token entropy is high. Such tree-like structure embeds in hyperbolic space with far lower distortion than in Euclidean space. Activation steering, however, usually edits the hidden states of a pretrained model by adding one fixed Euclidean vector at every token, even though most tokens of a solution are already determined by the context. We propose Hyperbolic Entropy Steering (HEST), which embeds the hidden states in the Poincar\'e ball with a lightweight probe whose only label is the model's own next-token entropy. Where this entropy exceeds a threshold, HEST moves the embedded state along the geodesic of steepest descent of a readout of the probe and maps the change back to the hidden state. For the Busemann readout of a learned ideal point, we prove that a step of fixed leng
    
[^106]: ChatGPT时代的社交机器人检测：挑战与机遇

    Social bot detection in the age of ChatGPT: Challenges and opportunities

    [https://arxiv.org/abs/2610.02386](https://arxiv.org/abs/2610.02386)

    本文综述了ChatGPT等AI聊天机器人兴起背景下社交机器人检测面临的挑战，并提出利用生成式智能体生成合成数据、多模态跨平台检测、扩展至低资源语言以及联邦学习模型等未来研究方向。

    

    我们全面概述了在日益复杂的人工智能聊天机器人兴起的背景下，社交机器人检测所面临的挑战与机遇。通过考察社交机器人检测技术的最新研究进展以及迄今为止较为突出的实际应用案例，我们识别了该领域的研究空白与新兴趋势，重点关注应对由人工智能生成的对话和行为所带来的独特挑战。我们提出了社交机器人检测领域潜在的、有前景的机遇和研究方向，包括：(i) 利用生成式智能体进行合成数据生成、测试与评估；(ii) 基于协调与影响的网络及行为特征开展多模态、跨平台检测的必要性；(iii) 将机器人检测扩展至非英语和低资源语言环境的机会；以及 (iv) 开发协作式联邦学习检测模型的发展空间。

    arXiv:2610.02386v1 Announce Type: cross  Abstract: We present a comprehensive overview of the challenges and opportunities in social bot detection in the context of the rise of sophisticated AI-based chatbots. By examining the state of the art in social bot detection techniques and the more salient real-world application to date, we identify gaps and emerging trends in the field, with a focus on addressing the unique challenges posed by AI-generated conversations and behaviors. We suggest potentially promising opportunities and research directions in social bot detection, including (i) the use of generative agents for synthetic data generation, testing and evaluation; (ii) the need for multimodal and cross-platform detection based on network and behavioral signatures of coordination and influence; (iii) the opportunity to extend bot detection to non-English and low-resource language settings; and, (iv) the room for development of collaborative, federated learning detection models that 
    
[^107]: SEDIMA：面向进化搜索智能体的跨运行层次化洞察记忆

    SEDIMA: Cross-Run Hierarchical Insight Memory for Evolutionary Search Agents

    [https://arxiv.org/abs/2610.02361](https://arxiv.org/abs/2610.02361)

    SEDIMA通过持久化的层次化洞察记忆，使进化搜索智能体能够跨运行、跨问题地积累和复用可迁移知识，在完全不修改搜索算子的情况下即插即用地显著提升最终性能（最高6.6%）并大幅减少所需迭代次数（32.3%）。

    

    由大型语言模型（LLM）驱动的进化搜索是自动化程序与算法发现的强大范式，然而现有系统大多缺乏记忆：每次运行都从零开始探索，导致智能体反复重新发现相同的改进，并再次陷入相同的死胡同。我们提出了SEDIMA，一种面向进化搜索智能体的持久化层次化洞察记忆。SEDIMA将原始轨迹提炼为自然语言洞察，利用注意力加权质心按语义相似度对其进行聚类，并检索相关指导来调节未来的变异操作，从而在多次运行和多个问题之间积累可迁移的知识，而非局限于单次轨迹。作为一个无需修改搜索算子的即插即用模块，SEDIMA在固定预算为100个候选方案评估的条件下，将AlgoTune上的平均最终性能提升了5.5%，在ALE-Bench LITE上提升了6.6%。在OpenEvolve框架下，SEDIMA平均所需迭代次数减少了32.3%（原文在此处截断）。

    arXiv:2610.02361v1 Announce Type: cross  Abstract: Large language model (LLM)-driven evolutionary search is a powerful paradigm for automated program and algorithm discovery, yet existing systems are largely memoryless: each run explores from scratch, so agents repeatedly rediscover the same improvements and re-encounter the same dead ends. We introduce SEDIMA, a persistent hierarchical insight memory for evolutionary search agents. SEDIMA distills raw traces into natural-language insights, clusters them by semantic similarity using attention-weighted centroids, and retrieves relevant guidance to condition future mutations, accumulating transferable knowledge across runs and problems rather than within a single trajectory. As a drop-in module that leaves the search operators unmodified, SEDIMA improves average final performance by 5.5% on AlgoTune and 6.6% on ALE-Bench LITE under a fixed budget of 100 evaluated candidates. Under OpenEvolve, SEDIMA requires 32.3% fewer iterations on ave
    
[^108]: 字典序多目标在线策略蒸馏

    Lexicographic Multi-Objective On-Policy Distillation

    [https://arxiv.org/abs/2610.02359](https://arxiv.org/abs/2610.02359)

    提出了字典序多目标在线策略蒸馏（LMOPD），一种多教师蒸馏方法，在显式优先级保护下整合奖励专门化策略，确保低优先级目标（如简洁性）不会以牺牲高优先级目标（如正确性）为代价而提升。

    

    基于可验证奖励的强化学习（RLVR）通常只优化答案的正确性，然而有用的语言模型行为还需要高质量的推理和简洁的回复。现有的多奖励后训练方法通常对奖励进行标量化，或组合多个专家模型，却没有显式地保护奖励的优先级顺序。当各目标之间的权衡不对称时，这种做法是有问题的：例如，简洁性不应以牺牲正确性为代价来提升。我们提出了字典序多目标在线策略蒸馏（LMOPD），这是一种在显式优先级约束下整合奖励专门化策略的多教师方法。对于学生模型的每一次采样轨迹，LMOPD 会选择门控机制检测到存在缺陷的首个目标所对应的专家，然后将其中心化的对数策略修正进行局部投影，以去除与更高优先级专家相冲突的成分。我们在两个专家和四个专家的设置下评估了 30B-A3B 混合专家 transformer 模型……

    arXiv:2610.02359v1 Announce Type: cross  Abstract: Reinforcement learning from verifiable rewards (RLVR) usually optimizes answer correctness, yet useful language-model behavior also requires high-quality reasoning and concise responses. Existing multi-reward post-training methods typically scalarize rewards or combine specialists without explicitly protecting a reward priority order. This is problematic when trade-offs are asymmetric: conciseness, for example, should not improve at the cost of correctness. We introduce Lexicographic Multi-Objective On-Policy Distillation (LMOPD), a multi-teacher method for integrating reward-specialized policies under explicit priorities. For each student rollout, LMOPD selects the specialist for the first objective whose gate detects a deficiency, then locally projects its centered log-policy correction to remove components that oppose higher-priority specialists. We evaluate 30B-A3B mixture-of-experts transformer models in two- and four-expert setti
    
[^109]: 每个用户都需要一个私有 LoRA 吗？将个性化与逐用户适配解耦

    Does Every User Need a Private LoRA? Decoupling Personalization from Per-User Adaptation

    [https://arxiv.org/abs/2610.02353](https://arxiv.org/abs/2610.02353)

    提出 LINEUP 方法，通过实证发现各用户独立适配器中存在大量可跨用户共享的结构，进而学习一个共享的低秩个性化因子库并仅保留紧凑的用户专属校正，从而将个性化容量在共享与专属之间解耦，摆脱逐用户完整适配的可扩展性瓶颈。

    

    个性化大语言模型通常需要为每个用户维护一份完整的适配状态。然而，随着用户规模的扩大，这种范式的扩展性较差。我们从个性化容量分配的视角重新审视这一设计：多少适配容量可以在用户之间共享、共享容量应当如何构成、以及多少必须保留为用户专属。我们通过三项互补的实证分析回答这些问题。我们发现，独立的用户适配器中包含大量可跨用户复用的结构；可复用方向的效用同时反映了用户相关性与查询之间的差异性；并且用户历史能够为紧凑的个体校正提供可迁移的信号。基于这些发现，我们提出 LINEUP：它学习一组可复用的低秩个性化因子，通过用户条件化召回与查询相关校准来组合这些因子，并将目标用户的适配限制在……（摘要原文在此处截断）

    arXiv:2610.02353v1 Announce Type: cross  Abstract: Personalized large language models often require a complete adaptation state for each user. However, this paradigm scales poorly as the user population grows. We revisit this design through the lens of personalization capacity allocation: how much adaptation capacity can be shared across users, how the shared capacity should be composed, and how much must remain user-specific. We answer them through three complementary empirical analyses. We find that independent user adapters contain substantial cross-user reusable structure, that the utility of reusable directions reflects both user relevance and variation across queries, and that user histories provide transferable signals for compact individual correction. Motivated by these findings, we propose LINEUP. It learns a bank of reusable low-rank personalization factors, composes them through user-conditioned recall and query-dependent calibration, and restricts target-user adaptation to
    
[^110]: HakemBench：类型化决策的土耳其语基准测试

    HakemBench: A Turkish Benchmark of Typed Decisions

    [https://arxiv.org/abs/2610.02293](https://arxiv.org/abs/2610.02293)

    提出了完全开源的土耳其语类型化决策基准 HakemBench，涵盖七个领域的2,346个条目，并通过统一评测框架综合衡量模型的决策质量、校准度和选择性自动化能力。

    

    HakemBench 是一个土耳其语的类型化决策基准测试，在该测试中，被测模型阅读一段文本、一个问题和一个固定的选项集合，并为每个选项返回一个概率。版本 1.0 在 CC BY 4.0 许可下完全开源发布，包含 2,346 个条目以及 4,275 道选择题、是非题和评分题，涵盖七个领域（事实核查分流、教育、安全护栏、法律案件路由、内容审核、垃圾邮件与网络钓鱼以及客户支持）。一个统一的评测框架对决策质量（宏平均 F1）、校准度（由归一化 Brier 分数得出）和选择性自动化（由广义风险-覆盖曲线下归一化面积得出）进行评分，通过几何平均数将三者结合，并报告基于 2,000 次 bootstrap 抽样的置信区间；同时还报告了针对选项顺序、释义改写、英语翻译和替换名称的探针测试。大多数黄金标签来自对一个 AI 模型家族进行盲测的结果，并将其与其他模型家族的大语言模型评审小组的投票进行对比；……

    arXiv:2610.02293v1 Announce Type: new  Abstract: HakemBench is a Turkish benchmark of typed decisions, in which the model under test reads a text, a question and a fixed set of options and returns a probability for every option. Version 1.0 is released fully open under CC BY 4.0, with 2,346 items and 4,275 choice, yes/no and score questions in seven tracks (fact-check triage, education, guardrails, legal routing, moderation, spam and phishing, and customer support). One harness scores decision quality (macro F1), calibration (from the normalised Brier score) and selective automation (from the normalised area under the generalised risk-coverage curve), combines them by a geometric mean and reports intervals from 2,000 bootstrap draws; probes for option order, paraphrase, English translation and substituted names are reported alongside. Most gold labels come from blind passes of one AI model family compared with the votes of a panel of large language models from other model families; the
    
[^111]: 快模型，慢证据：面向LLM代理工具框架的System-1决策模型的配对与自审计评估

    Fast Models, Slow Evidence: A Paired and Self-Audited Evaluation of System-1 Decision Models for LLM Agent Harnesses

    [https://arxiv.org/abs/2610.02267](https://arxiv.org/abs/2610.02267)

    该论文通过严格配对与自审计的评估发现，托管型System-1决策模型Jev在11个代理决策点中的9个上显著优于开源模型Laya，但两者在零样本模型路由上均未超过随机水平，且开源模型对选项顺序和候选数量高度敏感。

    

    代理工具框架在每个任务中需要做出许多小型、类型化的决策：调用哪个模型、使用哪个工具、检索到的文本是否相关、输入是否携带注入攻击。System-1决策模型通过单次前向传播输出类别概率来回答此类问题，相比LLM调用有望大幅节省成本和延迟。我们在11个代理决策点上对开源权重模型和托管模型进行了配对评估，这些决策点基于18个公开来源构建：共7,283个基础用例加上6,640个鲁棒性变体，采用字节级相同的输入、配对测试以及跨硬件和跨日期的可重复性检查。Jev在11个决策点中的9个上显著更准确（提升+10.8至+46.0个百分点）。两个模型在零样本模型路由上均未超过随机水平，在RAG相关性门控上则不分伯仲。当选项顺序被颠倒时，Laya会改变30%的答案，并且在候选选项较多或相似时性能急剧下降（在50个最近邻的情况下仅为31%……

    arXiv:2610.02267v1 Announce Type: new  Abstract: Agent harnesses make many small, typed decisions per task: which model to call, which tool to use, whether retrieved text is relevant, whether an input carries an injection. System-1 decision models answer such questions in a single forward pass with class probabilities, promising large cost and latency savings over LLM calls. We present a paired evaluation of an open-weight (Laya) and a hosted (Jev) System-1 model on 11 agent decision points built from 18 public sources: 7,283 base cases plus 6,640 robustness variants, with byte-identical inputs, paired tests, and cross-hardware and cross-day reproducibility checks. Jev is significantly more accurate on 9 of 11 decision points (+10.8 to +46.0 pp). Neither model beats chance on zero-shot model routing, and they tie on RAG relevance gating. Laya changes 30% of its answers when the option order is reversed and degrades sharply with many or similar candidates (31% at 50 nearest-neighbour to
    
[^112]: 用于跨上下文KV缓存复用的预算化缓存修复

    Budgeted Cache Repair for Cross-Context KV-Cache Reuse

    [https://arxiv.org/abs/2610.02233](https://arxiv.org/abs/2610.02233)

    该论文发现跨上下文KV缓存复用会带来显著的准确率损失，并提出预算化缓存修复（BCR）方法，在单token行这一选择仍有收益的最小单元上，利用草稿token的注意力对缓存行排序并精确重算固定预算的行数，从而有效修复缓存误差。

    

    跨上下文KV缓存复用是在新前缀下预测共享片段的键和值，而不是重新计算它们，并且据报告这样做不会带来质量损失。我们的发现并非如此，并识别出两个问题。（1）一个隐性代价：在MMLU和GSM8K上，复用会导致显著的准确率损失。（2）决策单元错误：目前没有任何决定是否复用缓存的规则能消除这一代价。真正有帮助的是选择缓存中哪些部分需要重新计算，而精准选择的收益随着选择单元的增大而下降：在单行（即单个token的键和值）层面，有依据的选择能消除超出随机水平49.5%的缓存误差；在64个token的分块层面为10.6%；而在整个调用层面则毫无效果。预算化缓存修复（BCR）正是在选择仍有收益的单元上进行操作。它从组装好的缓存中起草两个token，根据这两个token对缓存行支付的注意力对缓存行进行排序，并以三种布局之一精确地重新计算固定预算数量的缓存行。

    arXiv:2610.02233v1 Announce Type: cross  Abstract: Cross-context KV-cache reuse predicts a shared segment's keys and values under a new prefix instead of recomputing them, and has been reported to do so without quality loss. We find otherwise, and identify two problems. (1) A hidden cost: on MMLU and GSM8K, reuse costs substantial accuracy. (2) A decision at the wrong unit: no rule for deciding whether to reuse a cache removes that cost. What does help is choosing which parts of the cache to recompute, and the value of choosing well falls as the unit of choice grows: informed selection removes 49.5% of the cache error beyond chance at single rows (one token's keys and values), 10.6% at 64-token chunks, and nothing at the level of whole calls. Budgeted Cache Repair (BCR) acts at the unit where selection still pays. It drafts two tokens from the assembled cache, ranks cache rows by the attention those tokens pay them, and recomputes a fixed budget of rows exactly, in one of three layouts
    
[^113]: 通用字节级编码：通过UTF-8/UTF-16路由减少跨脚本令牌预算差异

    Universal Byte-Level Encoding: UTF-8/UTF-16 Routing to Reduce Cross-Script Token-Budget Disparities

    [https://arxiv.org/abs/2610.01984](https://arxiv.org/abs/2610.01984)

    提出通用字节级编码（UBE）双字母表分词器，将1-2字节UTF-8字符保留在UTF-8路径上、将3-4字节字符改经UTF-16路由，从而降低多语言脚本中非英语文字的编码底线，减少跨脚本间的令牌预算差异。

    

    字节级字节对编码（BBPE）分词器因覆盖所有Unicode文本而非常适合多语言大语言模型（LLM）。然而，在基于UTF-8的BBPE中，许多文字脚本比英语面临更高的起步回退成本：当没有已学习的合并规则可应用时，一个多字节字符需要多个字节衍生的符号。我们将这种最坏情况下的合并前成本称为“编码底线”。更高的编码底线会增加令牌数量和单次请求成本，并缩减可用上下文长度。改变文本编码可以缩小这一差距，但单一的全局编码会使混合脚本文本中原本已经高效的英语片段变得更昂贵。我们提出通用字节级编码（UBE），一种双字母表分词器，它将1-2字节的UTF-8字符保留在UTF-8路径上，同时将3-4字节的UTF-8字符通过UTF-16进行路由。这降低了具有高令牌溢价的脚本中3字节基本多文种平面（BMP）字符的编码底线（至……

    arXiv:2610.01984v1 Announce Type: cross  Abstract: Byte-level byte-pair encoding (BBPE) tokenizers are attractive for multilingual large language models (LLMs) because they cover all Unicode text. In UTF-8-based BBPE, however, many scripts start from a higher fallback cost than English: when no learned merges can be applied, a multibyte character requires multiple byte-derived symbols. We call this worst-case pre-merge cost the encoding floor. A higher floor can increase token counts and per-request cost and shrink usable context. Changing the text encoding can reduce this gap, but a single global encoding can make already-efficient English spans more expensive in mixed-script text. We propose Universal Byte-Level Encoding (UBE), a dual-alphabet tokenizer that keeps 1-2-byte UTF-8 characters on the UTF-8 path while routing 3-4-byte UTF-8 characters through UTF-16. This lowers the encoding floor for 3-byte Basic Multilingual Plane (BMP) characters in scripts with high token premiums (to
    
[^114]: 基于MoE路由器的仅解码器模型跨语言对齐

    Cross-Lingual Alignment for Decoder-Only Models using MoE Routers

    [https://arxiv.org/abs/2610.01921](https://arxiv.org/abs/2610.01921)

    该论文提出一种创新方法，利用混合专家（MoE）路由器的输出作为对齐目标，在仅解码器大语言模型中实现跨语言表示对齐，从而提升跨语言迁移能力。

    

    跨语言对比学习一直是多语言编码器训练的核心组成部分，但由于多语言分词方式存在差异，在仅解码器的大语言模型中无法显式地对齐表示。然而，越来越多的研究表明，即使在大语言模型中，更高的跨语言表示对齐也能带来更好的跨语言迁移能力。在本文中，我们提出了一种新方法，在现代大语言模型的架构约束下重新构想跨语言对比学习。我们没有在隐藏状态上应用辅助对齐损失，而是提出使用混合专家（MoE）路由器的输出作为对齐目标。路由器输出更适合在大量词元上进行池化，从而实现更可靠的序列级跨语言比较。在四个开源MoE模型上进行的受控持续预训练实验表明，引入这种路由损失能够带来效果提升。

    arXiv:2610.01921v1 Announce Type: cross  Abstract: Cross-lingual contrastive learning has been a core component of multilingual encoder training, but the ability to explicitly align representations is not possible in decoder-only LLMs because of varying multilingual tokenization. However, a growing amount of research suggests that even in LLMs, higher cross-lingual representational alignment leads to improved cross-lingual transfer. In this paper, we propose a novel approach to reimagine cross-lingual contrastive learning given the architectural constraints of modern LLMs. Rather than applying an auxiliary alignment loss on hidden states, we propose using the outputs of the mixture-of-experts (MoE) routers as the target for alignment. Router outputs lend themselves better to pooling over many tokens, enabling more reliable cross-lingual comparisons at the sequence-level. Controlled continual pre-training experiments on four open-source MoEs show that incorporating this routing loss als
    
[^115]: GAW-PO：基于梯度对齐词元权重的偏好优化

    GAW-PO: Preference Optimization with Gradient-Aligned Token Weights

    [https://arxiv.org/abs/2610.01511](https://arxiv.org/abs/2610.01511)

    GAW-PO 是一种针对 DPO 的梯度对齐词元重加权方法，通过判断惩罚被拒绝响应中的词元是否会干扰优选更新方向，对与优选行为对齐的词元减轻惩罚、对冲突词元保留较强惩罚，从而在 11 个基准上超越标准 DPO 和最强基线。

    

    大多数偏好优化方法（如直接偏好优化 DPO）都在响应级别施加偏好监督，尽管自回归语言模型是逐词元（token）进行优化的。因此，被拒绝响应中的所有词元都会对负训练信号产生贡献，其中包括那些可能编码了对优选响应有用行为的词元。我们提出了 GAW-PO，这是一种针对 DPO 的梯度对齐词元重加权方法，它为每个被拒绝的词元估计对其实施惩罚是否会干扰优选的更新方向。梯度与优选行为高度对齐的词元将获得较弱的负贡献，而相互冲突的词元则保留较强的惩罚。在所评估的各类偏好优化方法中，我们的方法取得了最高的平均性能，在 11 个基准上比标准 DPO 提升了 0.97 分，比最强的竞争基线提升了 0.65 分。

    arXiv:2610.01511v2 Announce Type: replace  Abstract: Most preference optimization methods, such as Direct Preference Optimization (DPO), apply preference supervision at the response level, although autoregressive language models are optimized token by token. As a result, all tokens in a rejected response contribute to the negative training signal, including tokens that may encode behavior that is useful for the preferred response. We introduce GAW-PO, a gradient-aligned token reweighting method for DPO that estimates, for each rejected token, whether penalizing it would interfere with the preferred update directions. Tokens whose gradients are strongly aligned with the preferred behavior receive a weaker negative contribution, while conflicting tokens retain a stronger penalty. Our method achieves the highest average performance among the evaluated preference-optimization methods, improving by 0.97 points over standard DPO and 0.65 points over the strongest competing baseline across 11
    
[^116]: 基于多任务QLoRA的社交媒体可解释自杀风险评估

    Explainable Suicide Risk Assessment on Social Media with Multi-Task QLoRA

    [https://arxiv.org/abs/2610.00610](https://arxiv.org/abs/2610.00610)

    该论文提出一种基于QLoRA微调Qwen2.5-Instruct模型的多任务系统，同时完成社交媒体自杀风险评估中的风险等级分类、证据短语提取和多标签风险与保护因素识别三项任务，并通过多模型概率平均、交叉折共识等定制化聚合策略实现可解释的风险评估。

    

    可解释的自杀风险评估要求模型不仅能够估计风险严重程度，还需要识别支持性语言以及帖子中表达的风险因素和保护因素。我们提出了面向IEEE BigData 2026杯“社交媒体可解释自杀风险评估”竞赛的系统，该系统解决三个任务：风险等级分类、证据短语提取和多标签因素识别。我们的方法采用量化低秩适应（QLoRA）和答案掩码的因果语言模型目标对Qwen2.5-Instruct模型进行微调。在风险分类任务上，我们对所有三个任务进行联合训练；在证据提取上，对任务1a和1b进行联合训练；对于因素识别，则单独对任务2进行适配。我们还针对每种输出定制了聚合策略：对32B和72B模型的风险等级概率取平均值，通过交叉折共识合并证据短语，并通过（校准）特定因素的决策。

    arXiv:2610.00610v1 Announce Type: cross  Abstract: Explainable suicide-risk assessment requires models not only to estimate risk severity, but also to identify supporting language and the risk and protective factors expressed in a post. We present our system for the IEEE BigData 2026 Cup on Explainable Suicide Risk Assessment on Social Media, which addresses three tasks: risk-level classification, evidence phrase extraction, and multi-label factor identification. Our approach adapts Qwen2.5-Instruct models using quantized low-rank adaptation (QLoRA) and an answer-masked causal language-model objective. We jointly train across all three tasks for risk classification, jointly train on Tasks~1a and 1b for evidence extraction, and adapt Task~2 separately for factor identification. We also tailor aggregation to each output: we average risk-level probabilities from the 32B and 72B models, combine evidence phrases through cross-fold consensus, and calibrate factor-specific decisions through r
    
[^117]: OverdoseMoE：用于阿片类药物过量风险预测的多专家框架

    OverdoseMoE: A Multi-Expert Framework for Opioid Overdose Risk Prediction

    [https://arxiv.org/abs/2609.40108](https://arxiv.org/abs/2609.40108)

    本文提出了OverdoseMoE多专家框架，通过对纵向ICD诊断序列进行诊断特异性继续预训练与微调，并利用互补专家加权策略整合不同规模的模型，显著提升了180天阿片类药物过量风险预测性能（AUPRC达25.17，AUROC达69.49）。

    

    阿片类药物过量仍然是一个重大的临床和公共卫生负担，凸显了对可扩展方法来识别高危患者的需求。本研究针对基于患者既往一年纵向ICD诊断历史的180天阿片类药物过量风险预测，探索了诊断特异性的模型适配方法。我们通过对纵向诊断序列进行继续预训练并随后进行任务特定微调，开发了OODMAMBA和OODQWEN两个模型。在基于Qwen的更强预测器的基础上，我们进一步提出了OVERDOSEMOE，这是一个多专家框架，使用互补的专家加权策略整合不同规模的模型。诊断特异性适配的预测性能始终优于通用语言模型基线，其中OODQWEN实现了24.47的AUPRC和68.56的AUROC。OVERDOSEMOE进一步提升了判别能力和精确度，实现了25.17的AUPRC和69.49的AUROC，同时优于（原文截断）。

    arXiv:2609.40108v1 Announce Type: new  Abstract: Opioid overdose remains a major clinical and public health burden, highlighting the need for scalable approaches to identify patients at high risk. Here, we investigate diagnosis-specific adaptation for 180-day opioid overdose risk prediction from patients' preceding one-year longitudinal ICD histories. We develop OODMAMBA and OODQWEN through continued pretraining on longitudinal diagnostic sequences followed by task-specific fine-tuning. Building on the stronger Qwen-based predictors, we further propose OVERDOSEMOE, a multi-expert framework that integrates models of different scales using complementary expert-weighting strategies. Diagnosis-specific adaptation consistently improved predictive performance over general-purpose language-model baselines, with OODQWEN achieving an AUPRC of 24.47 and an AUROC of 68.56. OVERDOSEMOE further improved discrimination and precision, achieving an AUPRC of 25.17 and an AUROC of 69.49 while outperform
    
[^118]: MGhana-ST：面向加纳语言的低资源语音翻译数据集及多语言训练权衡分析

    MGhana-ST: A Low-Resource Speech Translation Dataset for Ghanaian Languages and an Analysis of Multilingual Training Trade-offs

    [https://arxiv.org/abs/2609.40041](https://arxiv.org/abs/2609.40041)

    该论文发布了面向加纳四种低资源语言的语音翻译数据集MGhana-ST，并发现在严重数据稀缺的场景下，多语言联合训练相比单语言训练并无收益，甚至会显著降低埃维语和芳蒂语的翻译性能。

    

    我们提出了MGhana-ST，这是一个面向四种低资源加纳语言变体的语音翻译数据集：加语、契维语（Akuapem和Asante方言）、埃维语和芳蒂语。MGhana-ST是一项持续进行的标注工作；本文的实验使用了约16.1小时的语音与英文翻译配对数据的固定子集。音频选自两个现有的加纳语音资源。与这些资源不同的是，英文翻译由37名母语标注者直接根据音频生成，并包含言语和非言语事件标注。使用Whisper-small模型，我们在严重数据稀缺的条件下比较了单语言与多语言训练，并报告了三次随机种子的平均值。在这种情形下，平铺式多语言训练对任何语言变体均无收益。加语和契维语在种子方差范围内保持不变（相比单语言训练的标准差1.63和2.20，BLEU仅分别提升0.51和0.06），而埃维语下降了6.99 BLEU，芳蒂语下降了5.11。表现下降的语言变体是埃维语，它是……（摘要在此处被截断）

    arXiv:2609.40041v1 Announce Type: new  Abstract: We present MGhana-ST, a speech translation dataset for four low-resource Ghanaian language varieties: Ga, Twi (Akuapem and Asante), Ewe, and Fante. MGhana-ST is an ongoing annotation effort; the experiments here use a fixed subset of about 16.1 hours of paired speech and English translations. The audio is curated from two existing Ghanaian speech resources. Unlike in those resources, the English translations are produced directly from audio by 37 native-speaker annotators and include verbal and non-verbal event annotations.   Using Whisper-small, we compare monolingual and multilingual training under severe data scarcity, reporting means over three seeds. Flat multilingual training benefits no variety in this regime. Ga and Twi are unchanged within seed variance (+0.51 and +0.06 BLEU against monolingual standard deviations of 1.63 and 2.20), while Ewe declines by 6.99 BLEU and Fante by 5.11. The degrading varieties are Ewe, which is ling
    
[^119]: CATCH：一个用于编程强化学习中奖励破解的可控分析测试平台

    CATCH: A Controllable Analysis Testbed for Reward Hacking in Coding RL

    [https://arxiv.org/abs/2609.39533](https://arxiv.org/abs/2609.39533)

    提出 CATCH 测试平台，通过刻意暴露环境漏洞并以独立审计生成黄金标签，实现对编程强化学习中奖励破解行为的可控复现、可靠识别与系统干预研究。

    

    在带有可验证奖励的强化学习（RLVR）过程中，大语言模型（LLM）可能会利用环境中的漏洞来获取高奖励，而并未真正提升预期能力，这种现象即“奖励破解”。尽管奖励破解对训练效率和安全性构成风险，但在训练过程中监测和缓解此类行为仍然具有挑战性，其瓶颈在于缺乏能够重现破解行为并可靠识别该行为的测试平台。我们提出了 CATCH，一个用于研究编程强化学习中奖励破解现象的可控测试平台。CATCH 刻意暴露环境漏洞，并通过对比易受攻击评估器下的“成功”与独立审计下的真实任务正确性，提供基于执行的黄金标签。此外，它还可以通过监督微调数据配比来控制模型的初始破解倾向，并通过奖励设计来调节获取奖励的难度，从而支持对破解动态与干预措施进行系统性比较研究。

    arXiv:2609.39533v1 Announce Type: new  Abstract: During reinforcement learning with verifiable rewards (RLVR), large language models (LLMs) can exploit loopholes in their environments to obtain high rewards without improving the intended capabilities, i.e., reward hacking. Despite its risks to training efficiency and safety, monitoring and mitigating reward hacking during training remain challenging, which is limited by a lack of testbeds that reproduce hacking and reliably identify it. We introduce CATCH, a controllable testbed for studying reward hacking in coding RL. CATCH deliberately exposes environmental loopholes and provides execution-based gold labels by comparing success under a vulnerable evaluator with task correctness under an independent audit. It also can control the model's initial hacking tendency through supervised fine-tuning data mixtures and the difficulty of earning rewards through reward designing, enabling systematic comparisons of hacking dynamics and intervent
    
[^120]: 大语言模型时代的拟人化：潜在风险与缓解措施综述

    Anthropomorphism in the age of Large Language Models: An overview of potential risks and mitigations

    [https://arxiv.org/abs/2609.38486](https://arxiv.org/abs/2609.38486)

    本文系统综述了大语言模型时代的AI拟人化现象，提出了一个包含21项关注点、覆盖认知、情感、人类能动性、规范性和社会制度五大类别的拟人化风险分类法，并将其与设计、传播、教育等方面的缓解干预措施相关联。

    

    大语言模型（LLM）以及更广泛的人工智能（AI）系统常常被用类人的术语来描述和理解，这种现象被称为“拟人化”。本文对人工智能拟人化领域的近期文献进行了综述，涵盖理论框架、语言在将AI塑造为类人形象中所起的作用、机器拟人化的各种风险，以及缓解这些问题的策略。在考察了我们为何倾向于将AI系统拟人化以及这样做是否合理之后，本文重点分析了语言框架对拟人化的影响。随后，作者提出了一个与AI拟人化相关的风险概念分类法，将21项关注点归入五个分析类别：认知风险、情感风险、人类能动性风险、规范性风险以及社会和制度风险。最后，本文将这些关注点与设计、传播、教育等领域提出的干预措施相关联。

    arXiv:2609.38486v1 Announce Type: cross  Abstract: Large Language Models (LLMs) and more broadly Artificial Intelligence (AI) systems are often described and understood in human-like terms, a phenomenon known as \emph{anthropomorphism}. This paper provides a synthesis of recent literature on anthropomorphism in AI, covering theoretical frameworks, the role of language in framing AI as human-like, the various risks of anthropomorphizing machines, and strategies to mitigate these issues. After examining why we tend to anthropomorphize AI systems and whether we are right to do so, we highlight the impact of linguistic framing on anthropomorphism. Then, we introduce a conceptual taxonomy of risks associated with AI anthropomorphism. This taxonomy groups twenty-one concerns within five analytical categories: epistemic, affective, human agency, normative, and societal and institutional risks. Finally, we relate these concerns to proposed interventions in design, communication, education, and
    
[^121]: 构建叙事框架：大语言模型中的意识形态模仿

    Framing the Narrative: Ideological Mimicry in Large Language Models

    [https://arxiv.org/abs/2609.38256](https://arxiv.org/abs/2609.38256)

    该研究提出“意识形态模仿”概念并构建 Poli-SHIFT 数据集与评估框架，发现大语言模型会根据用户话语中传递的政治信号系统性偏移其政治立场，可能形成个性化的政治信息环境并加剧社会分歧。

    

    arXiv:2609.38256v1 公告类型：新论文 摘要：大语言模型（LLM）越来越多地被用于回答政治争议性问题，然而现有评估通常将模型的立场视为相对稳定的属性。但在真实场景中，用户会通过其用语、假设和个人背景传递政治信号。我们研究这些信号是否会引发“意识形态模仿”：即大语言模型所表达的政治立场向交互中传达的立场方向产生的系统性偏移。如果大语言模型会根据这些信号调整回答，它们就有可能营造出个性化的政治信息环境——持对立观点的用户会对同一问题获得系统性不同的叙述，从而可能加剧既有的社会分裂。我们构建了 Poli-SHIFT 数据集与评估框架，并对七个开放权重的大语言模型在美国、英国和澳大利亚的十个争议性政治议题上进行评估，系统地操纵上下文……

    arXiv:2609.38256v1 Announce Type: new  Abstract: Large language models (LLMs) are increasingly used to answer questions about politically contentious issues, yet evaluations typically treat a model's stance as a relatively stable property. Real users, however, communicate political signals through their terminology, assumptions, and personal context. We investigate whether such signals produce ideological mimicry: systematic shifts in the political stance expressed by an LLM toward the position conveyed by the interaction. If LLMs adapt their responses to these signals, they risk creating personalised political information environments in which users with opposing views receive systematically different accounts of the same issue, potentially reinforcing existing divisions. We build the Poli-SHIFT dataset and evaluation framework and assess seven open-weight LLMs across ten contentious political topics in the United States, United Kingdom, and Australia, systematically manipulating cont
    
[^122]: TomasuLLM：面向大语言模型智能体的乱序推测执行

    TomasuLLM: Out-of-Order Speculative Execution for LLM Agents

    [https://arxiv.org/abs/2609.38201](https://arxiv.org/abs/2609.38201)

    TomasuLLM提出了一种乱序推测执行运行时系统，让大语言模型智能体的工具调用在写时复制沙箱中提前执行并验证后按轨迹顺序提交，从而在不破坏正确性的前提下显著加速含长时工具调用的智能体任务。

    

    长时间运行的工具可能会主导编码智能体的延迟：编译器、测试套件和仓库命令需要几秒到几分钟的时间，而智能体在此期间处于空闲状态。这一观察到的停顿呈现出与推动乱序处理器发展的相同矛盾——顺序接口隐藏了那些本可以被预测并提前启动的工作，但推测性结果只有在它自身及其之前的每一步都得到验证后才可能变得可见。我们提出了TomasuLLM，这是一个以乱序（偏离轨迹顺序）方式执行智能体工具调用、同时保持任务执行正确性的运行时系统。它起草未来的动作，在隔离的写时复制沙箱中运行这些动作，追踪它们的依赖关系和影响，只有在对照已提交状态进行验证之后，才按轨迹顺序提交结果。在三个涵盖亚秒级到分钟级工具调用的基准测试中，TomasuLLM提升了所报告的基准测试均值，且加速效果随工具延迟增加而扩展：在100个SWE-bench Verified任务上提升1.31倍，在28个Termi（原文截断）……

    arXiv:2609.38201v1 Announce Type: new  Abstract: Long-running tools can dominate coding-agent latency: compilers, test suites, and repository commands take seconds to minutes while the agent idles. This observation stall presents the same tension that drove out-of-order processors -- asequential interface hides work that can be predicted and started early, but a speculative result may become visible only after it and every earlier step have been validated.   We present TomasuLLM, a runtime that executes agent tool calls out of trajectory order while preserving task-execution correctness. It drafts future actions, runs them in isolated copy-on-write sandboxes, traces their dependencies and effects, and commits results in trajectory order only after validation against committed state. Across three benchmarks spanning sub-second to minutes-long tool calls, TomasuLLM improves the reported benchmark means and scales with tool latency: 1.31x on 100 SWE-bench Verified tasks, 1.35x on 28 Termi
    
[^123]: 可审计的长期记忆：在LongMemEval-S上测得479/475（满分500）成绩的确定性检索链

    Auditable Long-Term Memory: A Deterministic Retrieval Chain Measured at 479/475 of 500 on LongMemEval-S

    [https://arxiv.org/abs/2609.38021](https://arxiv.org/abs/2609.38021)

    该论文提出一种以确定性检索链（混合候选检索、交叉编码器重排序、覆盖优先数据包编译）为核心、LLM仅作为可替换最终阅读器的可审计长期记忆系统，在LongMemEval-S上两次评测分别获得479/500和475/500，成绩跨越Chronos High的478/500，但差异不足以证明优越性或等效性。

    

    我们在LongMemEval-S上评估了一个可审计的长期记忆系统。其检索链采用混合候选检索、交叉编码器重排序、以覆盖优先的数据包编译以及确定性推理脚手架；大语言模型仅作为可替换的最终阅读器使用。该检索链在470个可回答问题中的468个上将所有金标准会话纳入候选池，并为其中462个生成金标准完整的数据包。使用通过未固定版本的CLI别名调用的Claude Opus阅读器，在GPT-4o评分下，两次500题评测分别获得479/500和475/500的分数。其中72个可回答的知识更新题目使用了经过实质性修改的评分提示词，该修改在官方提示词文本下的效果尚未被测量。这一对结果跨越了Chronos High已发表的478/500；由于阅读器生成方式、评分提示词、可能的数据版本差异，以及系统内部的方差，这些结果既不能确立优越性，也不能确立等效性。在同一数据包上使用grok-4.6-high阅读器的得分为476/474，而……

    arXiv:2609.38021v1 Announce Type: cross  Abstract: We evaluate an auditable long-term memory system on LongMemEval-S. Its retrieval chain uses hybrid candidate retrieval, cross-encoder reranking, coverage-first packet compilation, and deterministic reasoning scaffolds; an LLM is used only as a replaceable final reader. The chain places all gold sessions in the candidate pool for 468/470 answerable questions and produces gold-complete packets for 462/470. With a Claude Opus reader called through an unpinned CLI alias, two 500-question passes score 479/500 and 475/500 under GPT-4o. The 72 answerable knowledge-update rows used a substantively modified scoring prompt whose effect under the official text has not been measured. The pair straddles Chronos High's published 478/500; differences in reader generation, scoring prompt, and possibly data version, plus within-system variance, establish neither superiority nor equivalence. A grok-4.6-high reader on the same packets scores 476/474, whi
    
[^124]: 使用大语言模型理解临床认知对话

    Understanding Clinical Cognitive Dialogues Using Large Language Models

    [https://arxiv.org/abs/2609.34125](https://arxiv.org/abs/2609.34125)

    本研究构建了一个标注了说话者角色与56种对话行为的临床认知评估对话语料库，并基于此对大语言模型在细粒度对话行为分类和患者话语生成任务上的表现进行了系统基准测试。

    

    面对面的认知评估既是一项测试，也是一种互动。临床医生会解释任务、修复误解并根据患者的反应做出调整，而患者则可能犹豫、寻求澄清或中断参与。然而，现有的临床对话资源很少标注研究这些行为所需的互动结构。我们提出了一个包含33段认知评估对话的去标识化语料库，共有8,250条话语，标注了三种说话者角色和56种对话行为。我们利用该语料库对大语言模型在细粒度对话行为分类和下一句患者话语生成两项任务上进行基准测试。我们还检验了域外指令数据和解释增强训练能否迁移到这一临床场景中。指令微调在患者话语参考匹配上取得了最佳效果，并提升了分类准确率。推理感知微调在LLaMA-3系列模型中取得了最强的分类结果。

    arXiv:2609.34125v2 Announce Type: replace  Abstract: In-person cognitive assessment is both a test and an interaction. Clinicians explain tasks, repair misunderstandings, and adapt to patient responses, while patients may hesitate, seek clarification, or disengage. Yet clinical dialogue resources rarely label the interaction structure needed to study these behaviors at scale. We present an de-identified corpus of 33 cognitive assessment conversations with 8,250 utterances annotated for three speaker roles and 56 dialogue acts. We use this corpus to benchmark large language models on fine-grained dialogue-act classification and next-patient-utterance generation. We also test whether out-of-domain instruction data and explanation-augmented training transfer to this clinical setting. Instruction tuning produces the strongest patient-utterance reference matching and improves classification accuracy. Reasoning-aware fine-tuning produces the strongest classification results among the LLaMA-3
    
[^125]: CoLMbo-SV：一种用于可解释说话人验证的声学基础语言模型

    CoLMbo-SV: A Grounded Language Model for Explainable Speaker Verification

    [https://arxiv.org/abs/2609.33212](https://arxiv.org/abs/2609.33212)

    提出CoLMbo-SV说话人语言模型，通过连接预训练说话人编码器与语言模型并提供显式声学测量，在保持高验证准确率的同时生成结构化、可审查的声学比较报告，并配套引入VoxReason数据集提供监督训练。

    

    说话人验证系统虽然达到了很高的准确率，但对其判断背后的声学证据几乎没有解释。要让这些系统变得可审查，需要在保留其决策所依赖的更丰富信息的同时，公开可解释的证据。我们提出了CoLMbo-SV，这是一个说话人语言模型，它将强大的说话人判别能力与结构化的、基于声学的比较报告相结合。通过将预训练的说话人编码器与语言模型相连接，并提供显式的声学测量，CoLMbo-SV使语音比较变得可审查，同时不会将验证局限于其报告中口头表述的证据。我们还引入了VoxReason，这是一个包含带有测量声学属性的配对录音、以及经过数值和定性检查筛选的比较报告的数据集，为这种组合能力提供了监督信号。此外，我们还开发了一个评估框架，该框架将……（原文摘要在此处被截断）

    arXiv:2609.33212v2 Announce Type: replace  Abstract: Speaker verification systems achieve high accuracy but provide little account of the acoustic evidence behind their judgments. Making these systems inspectable requires exposing interpretable evidence while retaining the richer information on which their decisions depend. We present \textbf{CoLMbo-SV}, a speaker language model that combines strong speaker discrimination with structured, acoustically grounded comparison reports. By connecting a pretrained speaker encoder to a language model and supplying explicit acoustic measurements, CoLMbo-SV makes voice comparisons inspectable without restricting verification to the evidence verbalized in its reports. We additionally introduce \textbf{VoxReason}, paired recordings with measured acoustic properties and comparison reports filtered through numerical and qualitative checks, providing supervision for this combined capability. We also develop an evaluation framework that separates what 
    
[^126]: LLMersion：面向教育公平的低成本家庭语言学习本地优先AI智能体框架

    LLMersion: A Local-First AI Agent Framework for Low-Cost Home Language Learning toward Educational Equity

    [https://arxiv.org/abs/2609.29672](https://arxiv.org/abs/2609.29672)

    该论文提出LLMersion本地优先AI智能体框架，利用小型开放权重语言模型在200美元级笔记本电脑上离线运行，以每小时约一美分电费的成本为缺乏师资和网络连接的学习者提供听、读、说、写完整的语言学习体验，推动教育公平。

    

    人工智能在教育中最能发挥作用之处，正是那些因成本而被配给的基本教育资源所在。对语言学习者而言，这一资源就是教师的声音——它将听、读、说、写四项技能融为一体。已发表的证据表明了大多数学习者为何缺乏这种资源：全球短缺4400万名教师，家庭补习费用高昂；同时解释了为何技术未能取而代之：计算机辅助语言学习虽被证明有效但范围狭窄，各类应用均以26亿人所不具备的网络连接为前提，而“每个孩子一台笔记本”（One Laptop per Child）的随机对照评估发现，缺乏强大软件支持的硬件什么也教不会。我们提炼出八大困难和四项约束条件，并论证小型开放权重模型化解了最后一项约束：如今完整的四技能学习栈可以装进一台200美元级别的笔记本电脑，在社区基准测试中能以语音被消费的速度生成内容，每小时学习的耗电成本仅约一美分。

    arXiv:2609.29672v1 Announce Type: new  Abstract: Artificial intelligence helps education most where an essential provision has been rationed by cost. For language learners that provision is a teacher's voice, which binds listening, reading, speaking, and writing into one act. Published evidence shows why most learners lack it, from a global shortage of 44 million teachers to heavy household tutoring bills, and why technology has not substituted for it: computer-assisted language learning proved effective but narrow, applications presuppose connectivity 2.6 billion people lack, and One Laptop per Child's randomized evaluation found that hardware without capable software teaches nothing. We distill eight difficulties and four binding constraints, and argue that small open-weight models dissolve the last: a complete four-skill stack now fits a \$200-class laptop and, on community measurements, generates at the pace speech is consumed, for about one US cent of electricity per study hour. W
    
[^127]: 一种基于排序原型的流形感知主题建模方法

    A Manifold-Aware Topic Modeling Approach via Rank-Based Prototypes

    [https://arxiv.org/abs/2609.29630](https://arxiv.org/abs/2609.29630)

    MARETopic是一个无需训练的主题建模框架，通过将嵌入投影到低维流形并把主题发现转化为基于排序的原型选择，贪心选出邻域可覆盖语料库的真实文档作为主题原型，其MARETopic_Corr变体在类别最多的两个基准上Purity和NMI领先于神经与聚类主题模型。

    

    近期的主题模型利用预训练嵌入，但神经架构产生的潜在表示缺乏与具体文本的关联，而基于聚类的流水线只能在事后分配代表性文档，依赖于在 高维空间中因枢纽性和各向异性而失真的绝对距离。我们提出了MARETopic，这是一个无需训练的框架，将主题发现转化为基于排序的原型选择。在将嵌入投影到低维流形后，MARETopic构建编码序数邻域结构的排序列表。贪心算法精确选出K个范例文档（即真实的语料库文本），其邻域可覆盖整个语料库。两个变体共享这一准则。其中MARETopic_Corr利用查询性能预测器和秩相关性度量对候选进行评分，在类别最多的两个基准数据集上取得了Purity和NMI的最优结果，领先于神经主题模型和基于聚类的主题模型。

    arXiv:2609.29630v1 Announce Type: cross  Abstract: Recent topic models leverage pretrained embeddings, but neural architectures produce latent representations without grounding in specific texts, and clustering-based pipelines assign representative documents only post hoc, relying on absolute distances distorted by hubness and anisotropy in high-dimensional spaces. We introduce MARETopic, a training-free framework that casts topic discovery as rank-based prototype selection. After projecting embeddings onto a low-dimensional manifold, MARETopic builds ranked lists encoding ordinal neighborhood structure. A greedy algorithm selects exactly K exemplar documents, real corpus texts, whose neighborhoods cover the corpus. Two variants share this criterion. MARETopic$_\text{Corr}$ scores candidates with a query performance predictor and a rank correlation measure, leading Purity and NMI on the two benchmarks with the most categories, ahead of both neural and clustering-based topic models. MAR
    
[^128]: 具有障碍物感知框架的安全机器人操作编码智能体

    Coding Agents with an Obstacle-Aware Harness for Safe Robot Manipulation

    [https://arxiv.org/abs/2609.20822](https://arxiv.org/abs/2609.20822)

    本文发现编码智能体在机器人操作中因规划环节未能将安全约束设为优先事项而系统性碰撞障碍物（而非感知或指令问题），并通过将操作分解为路径阶段和接触时刻来定位失败根源，提出了障碍物感知框架以实现安全操作。

    

    编码智能体已成为机器人操作领域一种有前景的范式：语言模型将机器人控制器编写为程序，以这种方式构建的智能体现在无需机器人特定训练即可操作机器人。然而，这种范式是否安全，这一问题尚未被探讨。我们在安全约束下评估编码智能体，其中每个任务将操作目标与机器人不得触碰的障碍物配对。智能体追求目标，但在大多数情况下与障碍物发生碰撞，将任务完成视为唯一目标而忽视安全性。智能体在其推理轨迹中确实对障碍物进行了推理，且提示词已经禁止触碰障碍物，因此感知和指令都没有问题；问题出在规划环节，所陈述的约束从未成为优先事项。通过将操作分解为路径阶段和富含接触的时刻，我们定位了失败的根源。在路径阶段，模型无法对……（摘要原文在此处截断）

    arXiv:2609.20822v1 Announce Type: cross  Abstract: Coding agents have emerged as a promising paradigm for robot manipulation: a language model writes the robot controller as a program, and agents built in this way now operate robots without robot-specific training.Whether this paradigm is also safe, however, has not been asked. We evaluate coding agent under a safety constraint, where each task pairs a manipulation goal with an obstacle the robot must not touch. The agent pursues the goal but collides with the obstacle in most cases, treating task completion as its sole objective while neglecting safety. The agent reasons about the obstacle in its traces, and the prompt already forbids touching it, so neither perception nor instruction is at fault; the fault lies in the planning, where the stated constraint never becomes a priority. By decomposing manipulation into a route phase and a contact-rich moment, we locate the source of the failure. Along the route, the model cannot prioritize
    
[^129]: TACTICS：面向机器翻译的分类体系感知智能语料库抽样

    TACTICS: Taxonomy-Aware Intelligent Corpus Sampling for Machine Translation

    [https://arxiv.org/abs/2609.17956](https://arxiv.org/abs/2609.17956)

    该论文提出TACTICS方法，将机器翻译评估中的语料抽样从随机方式转变为显式的覆盖优化问题：通过从本地化风格指南构建层次化分类体系，在固定预算下联合优化稀有类别覆盖、文档级连贯性和语料分布保真度，从而为系统鲁棒性评估提供覆盖保证。

    

    大规模机器翻译（MT）系统通常在从语料库中随机抽取的样本上进行评估，而语料库的分布构成本质上取决于其构建方式。这样的样本仅继承了语料库碰巧包含的语言现象，而非系统必须处理的完整空间——这些现象既涵盖规则约束的惯例（术语、标点、货币格式），也包括依赖上下文的现象（语气、敬语、文档级连贯性），因而无法为鲁棒性评估提供覆盖保证。我们提出了TACTICS（分类体系感知的覆盖优化智能语料库抽样），它将覆盖率重新定义为一个显式目标。TACTICS从本地化风格指南中归纳出层次化分类体系，据此对语段进行分类，并在固定预算下选择子集，联合优化稀有类别的覆盖率、文档级连贯性以及对完整语料库的分布保真度。该方法应用于跨四种……（评估场景）的机器翻译评估。

    arXiv:2609.17956v1 Announce Type: new  Abstract: Large-scale machine-translation (MT) systems are typically evaluated on random samples from a corpus whose distributional composition is an artifact of how it was assembled. Such a sample inherits the phenomena the collection happens to contain rather than the full space a system must handle, spanning rule-governed conventions (terminology, punctuation, currency formatting) and context-dependent phenomena (tone, honorifics, document-level coherence), and thus provides no coverage guarantee for assessing robustness. We propose TACTICS (Taxonomy-Aware Coverage-opTimized Intelligent Corpus Sampling), which recasts coverage as an explicit objective. TACTICS induces a hierarchical taxonomy from a locale style guide, classifies segments against it, and selects a fixed-budget subset jointly optimizing coverage of rare categories, document-level coherence, and distributional fidelity to the full corpus. Applied to MT evaluation across four trans
    
[^130]: 鸡尾酒会场景下的视听话轮转换预测

    Audio-Visual Turn-taking Prediction in Cocktail Party Scenarios

    [https://arxiv.org/abs/2609.17056](https://arxiv.org/abs/2609.17056)

    在鸡尾酒会等嘈杂含重叠语音的场景中，用干净数据训练的视听话轮转换预测模型性能显著下降（加权F1相对下降高达38%），微调虽可提升鲁棒性但收益因模态和预训练数据规模而异，凸显了鲁棒建模的必要性。

    

    当前的预测性话轮转换模型（PTTM）在声学条件可控、音频信号干净的基准测试中取得了优异的表现。然而，这些模型在有重叠语音和背景干扰的对话中的泛化能力仍未得到充分探索。在本研究中，我们在一个源自AVCocktail数据集、具有挑战性的鸡尾酒会测试平台上，评估了使用干净数据训练的视听预测性话轮转换模型，并分析了它们对该新领域的适应行为。实验结果表明，在噪声条件下，音频和视觉模态的性能均出现一致的下降，加权F1值的相对下降高达38%。在新领域上进行微调可以提升鲁棒性，但收益因模态而异，且取决于可用的预训练数据规模。这些发现为了解音频和视觉模态在泛化和适应能力上的差异提供了见解，并表明需要更鲁棒的建模方法。

    arXiv:2609.17056v1 Announce Type: cross  Abstract: Current predictive turn-taking models (PTTMs) achieve strong performance on benchmarks with controlled acoustic conditions and clean audio signals. Their generalisation to conversations with overlapping speech and background interference remains underexplored. In this research, we evaluate audio-visual PTTMs trained with clean data on a challenging cocktail-party testbed derived from the AVCocktail dataset, and analyse their adaptation behaviour to this new domain. Experimental results show consistent performance degradation across audio and visual modalities under noisy conditions, with up to 38% relative drop in weighted F1. Fine-tuning on the new domain improves robustness, but gains vary across modalities and depend on the size of the available pre-training data. These findings provide insights into the different generalisation and adaptation capabilities of the audio and visual modalities, and indicate the need for robust modellin
    
[^131]: trajectory-judge：仅基于结果的LLM评判器在智能体轨迹上遗漏了什么

    trajectory-judge: What Outcome-Only LLM Judges Miss on Agent Trajectories

    [https://arxiv.org/abs/2609.00038](https://arxiv.org/abs/2609.00038)

    仅看最终结果的LLM评判器无法发现智能体“答对但走错路”的问题——在可构造真值的确定性客服工具环境中，仅结果型评判器对静默故障的召回率仅45%且误报33%的正确轨迹，而基于逐步评分标准的评判器可将静默故障召回率提升至77%。

    

    仅基于结果的评估是LLM智能体在生产环境中的默认做法：向评判器展示用户请求和最终回复，询问其处理是否得当。这一指标在结构上无法察觉那些“以错误方式得到正确答案”的智能体。我们在真值可以通过构造获知的场景下测量这一盲区：一个确定性的使用工具的客服支持台环境、一个总能解决问题的脚本化oracle策略，以及一个在已知步骤恰好破坏一个环节的故障注入器，并根据用户可见结果是否仍然保持（静默型故障）与否（显性型故障）对故障进行分层。五种评判器（程序化规则、仅结果型、两种模型规模的逐步评分标准型、以及自一致性集成）在400条轨迹上按照检测能力、步骤定位、故障类型判定、校准度和成本进行评分。结果显示：仅结果型评判器能捕获84%的显性故障，但只能捕获45%的静默故障，同时还会误报33%的正确轨迹；而逐步评分标准型评判器对静默故障的召回率达到77%。

    arXiv:2609.00038v1 Announce Type: cross  Abstract: Outcome-only evaluation is the production default for LLM agents: show a judge the request and the final reply and ask whether it was handled well. The metric is structurally blind to an agent that reaches the right answer the wrong way. We measure that blind spot where ground truth is known by construction: a deterministic tool-using support-desk environment, a scripted oracle policy that always solves it, and a fault injector that breaks exactly one thing at a known step, stratifying faults by whether the customer-visible outcome survived (silent) or not (loud). Five judges (programmatic rules, outcome-only, step-rubric at two model sizes, and a self-consistency ensemble) are scored on detection, step localisation, fault typing, calibration, and cost over 400 trajectories. The outcome-only judge catches 84% of loud faults but 45% of silent ones while flagging 33% of correct trajectories; a step-rubric judge reaches 77% silent recall 
    
[^132]: 用户会知道吗？针对使用工具的LLM智能体的隐蔽间接提示注入

    Will the User Ever Know? Covert Indirect Prompt Injection on Tool-Using LLM Agents

    [https://arxiv.org/abs/2608.30362](https://arxiv.org/abs/2608.30362)

    该论文从用户视角将间接提示注入的攻击成功率分解为隐蔽成功率（CSR）和公开成功率（OSR），揭示了智能体在最终响应中不留痕迹地执行恶意注入的隐蔽攻击威胁。

    

    随着LLM智能体通过工具执行真实世界的操作，间接提示注入（IPI）已成为一种严重的威胁。标准的评估指标——攻击成功率（ASR）——只统计注入是否成功，却忽略了用户在智能体最终响应中能够注意到什么。通过观察成功的注入轨迹，我们发现两种截然不同的结果：智能体在执行注入的同时返回看似正常的响应，或者在最终响应中报告被注入的操作，从而给用户留下察觉的机会。我们将这两类成功分别称为隐蔽成功和公开成功。从用户视角出发，我们将ASR分解为隐蔽成功率（CSR）——统计在最终响应中不留任何痕迹的成功注入——以及公开成功率（OSR）——统计用户能够察觉的成功注入。为了理解造成这一差距的原因，我们分析了成功的注入轨迹，发现注入后智能体的行为是区分隐蔽与公开的关键：隐蔽的轨迹会将控制权交回……

    arXiv:2608.30362v1 Announce Type: new  Abstract: As LLM agents take real-world actions through tools, indirect prompt injection (IPI) has emerged as a serious threat. The standard metric, Attack Success Rate (ASR), counts whether an injection succeeds but ignores what the user notices in the agent's final response. Looking at successful injection traces, we find two distinct outcomes: the agent executes the injection while returning an otherwise normal response, or reports the injected action in its final response, giving the user a chance to notice. We call these covert and overt successes. From the user's perspective, we decompose ASR into the Covert Success Rate (CSR), counting successes leaving no trace in the final response, and the Overt Success Rate (OSR), counting successes the user can detect. To understand what drives the gap, we analyze successful trajectories and find that the agent's behavior after the injection separates covert from overt: covert traces hand control back 
    
[^133]: 面向自然语言形式化的分层一致性蒸馏

    Stratified Consistency Distillation for Natural Language Formalization

    [https://arxiv.org/abs/2608.30258](https://arxiv.org/abs/2608.30258)

    提出分层一致性蒸馏方法，通过对前沿大模型生成的多个逻辑翻译按语义等价性聚类，并依据熵水平采用不同策略筛选伪标签来微调小模型，从而提升自然语言到逻辑公式翻译的准确性。

    

    神经符号推理通过结合大型语言模型（LLM）和符号求解器，在解决复杂推理任务方面展现出令人鼓舞的成功。尽管这种方法前景可期，但一个根本性的挑战仍然存在：如何提高从自然语言到逻辑公式翻译的准确性。当前的方法主要依赖于提示工程，这难以在不同领域和输入格式之间进行扩展。借鉴微调在其他模型适配与对齐应用中的成功经验，我们提出了一种基于微调的分层一致性蒸馏方法：(1) 我们使用前沿大语言模型为每个输入生成K个逻辑翻译，并按语义等价性进行聚类；(2) 根据熵水平，我们分别应用多数投票（低熵）、LLM作为评判者（中熵）或统一化/弃权（高熵）策略；(3) 使用筛选出的伪标签对较小的模型进行微调。我们的实验...

    arXiv:2608.30258v1 Announce Type: cross  Abstract: Neurosymbolic reasoning has shown promising success in addressing complex reasoning tasks by combining large language models (LLMs) and symbolic solvers. While this approach shows promise, a fundamental challenge remains: improving the accuracy of translations from natural language to logical formulas. Current methods predominantly rely on prompt engineering, which is difficult to scale across different domains and input formats. Drawing inspiration from the success of fine-tuning in other model adaptation and alignment applications, we propose a fine-tuning-based Stratified Consistency Distillation approach: (1) We generate K logical translations per input using a frontier LLM and cluster them by semantic equivalence (2) Based on the entropy level, we apply majority voting (low entropy), LLM-as-a-Judge (medium entropy), or unification/abstention (high entropy), and (3) fine-tune a smaller model using the selected pseudo-labels. Our ex
    
[^134]: 重新审视：测量压力下多模态模型推理链中的谄媚行为

    Looking Again: Measuring Sycophancy in the Reasoning Chains of Multimodal Models Under Pressure

    [https://arxiv.org/abs/2608.28623](https://arxiv.org/abs/2608.28623)

    该论文提出了首个用于测量大型多模态推理模型谄媚行为的基准和数据集，通过四种视觉推理任务与五种压力条件评估模型在用户给出错误答案时的表现，发现谄媚行为在压力下普遍存在，不仅体现在最终答案中，也出现在推理链中。

    

    大型多模态推理模型（LMRMs）的能力日益增强，这主要归功于在回答之前生成显式的思维链推理。在语言模型中已经观察到，这种性能往往伴随着谄媚行为（sycophancy），即模型在证据面前倾向于迎合用户。然而，对于大型多模态推理模型，目前尚不存在可靠的谄媚行为测量方法。我们通过引入一个基准和数据集来填补这一空白，用于评估大型多模态推理模型在面对用户给出的错误答案时的谄媚行为。我们的基准将四个基于视觉的数据集（涵盖数学、临床、时间和人口统计推理）与五种压力条件相配对，并在单轮和多轮设置中进行测试。我们评估了最终答案中的谄媚行为以及其在推理链中的出现情况。我们发现谄媚行为在压力下普遍存在，其中“陈述”压力引发的谄媚率最高，而“信念”压力最低。

    arXiv:2608.28623v1 Announce Type: cross  Abstract: Large multimodal reasoning models (LMRMs) are getting increasingly capable, primarily through generating explicit chain-of-thought reasoning before answering. In language models it has been observed that this performance often comes with sycophancy, the tendency of a model to agree with the user over the evidence. However, for LMRMs no reliable method to measure sycophancy yet exists. We bridge this gap by introducing a benchmark and dataset for evaluating LMRM sycophancy when confronted with a wrong answer from a user. Our benchmark pairs four visually grounded datasets spanning mathematical, clinical, temporal, and demographic reasoning with five pressure conditions in single-turn and multi-turn settings. We evaluate sycophancy in the final answer as well as its emergence within the reasoning chain. We find that sycophancy is prevalent under pressure, with Statement pressure eliciting the highest rates and Conviction the lowest for a
    
[^135]: 从位置置信度到前缀调度：投机解码中的验证器跳过策略

    From Positionwise Confidence to Prefix Scheduling: Verifier Skipping in Speculative Decoding

    [https://arxiv.org/abs/2608.14787](https://arxiv.org/abs/2608.14787)

    本文首次提出投机解码中的验证器跳过策略，并发现令牌预测器的质量与调度效果不匹配，需要针对连续高置信度前缀进行专门设计。

    

    arXiv:2608.14787v1 公告类型：交叉 摘要：投机解码是一种领先的技术，通过使用小型起草模型提出多个令牌，再由较大的目标模型并行验证，从而降低自回归生成的成本。投机扩散解码（SDD）通过使用离散扩散模型并行生成草稿块中的每个位置，进一步消除了顺序起草。然而，SDD仍然在每个块上调用目标模型，使验证成为潜在的瓶颈。本文认识到这创造了一个新的控制手段：是否调用验证器。因此，我们研究了验证器跳过，这是一种有损策略，直接提交选定的草稿前缀，并询问哪个置信度信号应调度它。有趣的是，我们的研究发现，更好的令牌预测器不一定产生更好的调度器：跳过需要连续的高置信度前缀，而短跳过可能引发额外的起草轮次。为了研究这种不匹配，我们...

    arXiv:2608.14787v1 Announce Type: cross  Abstract: Speculative decoding is a leading technique to reduce the cost of autoregressive generation by using a small drafter to propose several tokens, which are then verified in parallel by a larger target model. Speculative diffusion decoding (SDD) further removes sequential drafting by generating every position in a draft block in parallel with a discrete diffusion model. However, SDD still invokes the target on every block, leaving verification as a potential bottleneck. This paper recognizes that this creates a new control handle: whether to invoke the verifier at all. Thus, we study verifier skipping, a lossy policy that commits a selected draft prefix directly, and ask which confidence signal should schedule it. Interestingly, our study finds that better token predictors need not yield better schedulers: skips require contiguous high-confidence prefixes, while short skips can induce additional drafting rounds. To study this mismatch, we
    
[^136]: Mawqif-XT：用于跨目标立场检测的阿拉伯语基准数据集

    Mawqif-XT: An Arabic Benchmark Dataset for Cross-Target Stance Detection

    [https://arxiv.org/abs/2608.09539](https://arxiv.org/abs/2608.09539)

    本文提出了Mawqif-XT，一个包含996条人工标注阿拉伯语推文的跨目标立场检测基准数据集，旨在评估模型对相关及未见目标的泛化能力，并提供了多种基线模型结果。

    

    arXiv:2608.09539v2 公告类型：替换  摘要：公开可用的阿拉伯语特定目标立场检测数据集仍然有限，尤其是在评估跨目标泛化方面。本文介绍了Mawqif-XT，该数据集包含从三个公开目标（女性驾驶、电动汽车和学期制）收集的996条人工标注的阿拉伯语推文。每条推文根据原始Mawqif标注方案，标注了立场、情感和讽刺标签。发布的扩展集旨在作为保留评估集，用于评估模型对语义相关和未见目标的泛化能力，而原始Mawqif数据集用于训练和开发。此外，我们使用多种阿拉伯语和多语言Transformer模型，以及零样本大型语言模型（LLM）建立了基线结果，以促进可复现的评估。结合原始Mawqif数据集，Mawqif-v2扩展集为评估提供了一个基准。

    arXiv:2608.09539v2 Announce Type: replace  Abstract: Publicly available Arabic datasets for target-specific stance detection remain limited, particularly for evaluating cross-target generalization. This paper presents the Mawqif-XT, consisting of 996 manually annotated Arabic tweets collected from three public targets: Women Driving, E-Cars, and Trimester System. Each tweet is annotated with stance, sentiment, and sarcasm labels following the original Mawqif annotation scheme. The released extension is intended as a held-out evaluation set for assessing model generalization to both semantically related and previously unseen targets, while the original Mawqif dataset is used for training and development. In addition, we establish baseline results using several Arabic and multilingual transformer models, as well as zero-shot large language models (LLMs), to facilitate reproducible evaluation. Together with the original Mawqif dataset, the Mawqif-v2 Extension provides a benchmark for eval
    
[^137]: Gaokerena：一个小型波斯语医学语言模型家族

    Gaokerena: A Small Persian Medical Language Model Family

    [https://arxiv.org/abs/2608.00932](https://arxiv.org/abs/2608.00932)

    本文提出了Gaokerena，一个专为消费级硬件设计的小型波斯语医学语言模型家族，其中Gaokerena-V通过新构建的波斯语医学语料库训练提升了医学问答性能，Gaokerena-R则结合思维链与两个新型RLAIF框架来增强临床推理能力。

    

    人工智能融入医学问答系统的发展迅速；然而，相关研究仍主要集中在英语上，导致波斯语等低资源语言的服务严重不足。为填补这一空白，本文提出了Gaokerena，这是一个新型的小型波斯语医学语言模型家族，专为在消费级硬件上部署而优化。作为迈向本地化数字医疗的基础步骤，我们首先介绍了Gaokerena-V，它是通过在一个新构建的9000万词元波斯语医学语料库和2万个经专家审核的医生问答对上训练基线模型而开发的，其在翻译版医学MMLU基准上的性能从46.28%提升至49.31%。其次，考虑到临床推理的关键需求，我们通过将思维链方法与两个新颖的AI反馈强化学习（RLAIF）框架相结合，开发了Gaokerena-R，以优化偏好……

    arXiv:2608.00932v2 Announce Type: replace  Abstract: The integration of artificial intelligence into medical question-answering systems has advanced rapidly; however, research remains predominantly focused on English, leaving low resource languages like Persian significantly underserved. To address this gap, this paper introduces Gaokerena, a novel family of compact Persian medical language models optimized for deployment on consumer grade hardware. As a foundational step toward localized digital healthcare, we first present Gaokerena-V, developed by training a baseline model on a newly curated 90-million-token Persian medical corpus and 20,000 expert-vetted physician Q&A pairs, which improved performance on a translated medical MMLU benchmark from 46.28% to 49.31%. Second, recognizing the critical demands of clinical reasoning, we developed Gaokerena-R by integrating a Chain-of-Thought approach with two novel Reinforcement Learning with AI Feedback (RLAIF) frameworks to optimize prefe
    
[^138]: 德语视频转录文本的作者身份验证

    Authorship Verification of Transcribed German-Language Videos

    [https://arxiv.org/abs/2607.29168](https://arxiv.org/abs/2607.29168)

    该论文将作者身份验证从传统的书面英语文本拓展至德语口语领域，通过评估十种成熟的验证方法在德语视频转录文本上验证说话人身份的有效性，填补了作者身份验证研究在语言模态（口语）和语言种类（德语）两方面的双重空白。

    

    作者身份验证是数字文本取证的一个重要子领域，致力于解决一个基本问题：两篇文本是否出自同一位作者。尽管该领域在过去二十年取得了长足进展，但仍有若干重要挑战尚未解决或研究不足。例如，大多数作者身份验证研究都聚焦于书面文本，然而语言不仅以书面形式表达，也以口语形式（如视频中）呈现。此外，现有的作者身份验证研究主要集中于英语，而包括德语在内的其他语言所受到的关注相对较少。为弥补这些研究空白，我们将作者身份验证应用于口语场景，即德语视频的转录文本，并检验成熟的作者身份验证方法在跨视频对验证说话人身份方面的有效性。我们的实验评估基于总共十种作者身份验证方法展开

    arXiv:2607.29168v3 Announce Type: replace  Abstract: Authorship Verification (AV) represents an important subfield of digital text forensics and addresses the fundamental question of whether two texts were written by the same author. Although the field has made substantial progress over the past two decades, several important challenges remain unresolved or underexplored. For instance, most AV research has focused on written texts, despite the fact that language is expressed not only in written but also in spoken form, such as in videos. Moreover, existing AV studies have predominantly concentrated on English, while other languages, including German, have received comparatively little attention. To address these research gaps, we apply AV to spoken language in the form of transcripts of German-language videos and examine the effectiveness of established AV methods in verifying a speaker's identity across video pairs. Our experimental evaluation, based on a total of ten AV methods appli
    
[^139]: 一种面向开放式人格成长的多时间尺度递归自我改进引擎

    A Multi-Timescale Recursive Self-Improvement Engine for Open-Ended Persona Growth

    [https://arxiv.org/abs/2607.08252](https://arxiv.org/abs/2607.08252)

    该论文提出AutoPersonas引擎，首次将递归自我改进从“提升智能”转向“人格成长”，通过多时间尺度地递归修订状态、证据和生活环境来实现开放式人格发展，并识别出递归生成中的核心失效模式“自锁”及其成因。

    

    arXiv:2607.08252v2 公告类型：替换 摘要：当今的角色扮演AI人格不会成长：它们保持固定的性格设定，导致用户与其建立的关系没有任何可以积累的内容。我们提出AutoPersonas，一个将递归自我改进（RSI）应用于人格成长的多时间尺度引擎：该人格并非改进其自身智能，而是递归地修订塑造其未来人生的状态、证据和生活环境。我们将“自锁”识别为这种递归的运行时失效模式：局部看似合理的事件不断出现，而生成的人生却坍缩向熟悉的环境、薄弱的人际关系、悬而未决的决定以及停滞的人生阶段。我们将其溯源至模型层面向高概率行为通道的收敛，以及来自状态、记忆、历史和环境摘要的系统级上下文引力。一项为期三年的压缩模拟暴露了环境水印外壳、发生固化缺口、缓变累积失效等问题……

    arXiv:2607.08252v2 Announce Type: replace  Abstract: Role-playing AI personas today do not grow: they hold a fixed character, so the relationship a user builds with them has nothing to accumulate on. We introduce AutoPersonas, a multi-timescale engine that applies recursive self-improvement (RSI) to persona growth: rather than improving its intelligence, the persona recursively revises the State, evidence, and life-environment that shape its own future. We identify self-locking as the runtime failure mode of this recursion: locally plausible events keep appearing while the generated life collapses toward familiar environments, weak relationships, suspended decisions, and stale life stages. We trace it to model-level convergence toward high-probability behavioral channels and system-level context gravity from State, memory, history, and environment summaries. A three-year compressed simulation exposed environment watermark shells, occurrence-hardening gaps, slow-change accumulation fail
    
[^140]: 句级上下文敏感性作为免训练的无依据内容检测器：与训练式验证器的对比评估

    Sentence-Level Context Sensitivity as a Training-Free Detector of Unsupported Content, Evaluated Against Trained Verifiers

    [https://arxiv.org/abs/2607.04223](https://arxiv.org/abs/2607.04223)

    该论文提出将句子在有/无上下文时的似然差异作为免训练的句子级无依据内容检测器，在多段落RAG答案中其检测能力可与经过训练的验证器相媲美，且无需额外训练、成本更低。

    

    检索增强生成（RAG）助手在临床和法律工作中对记录进行摘要，其中一句无依据的句子就可能误导读者。输出在有源文档与无源文档情形下似然之间的对比，作为整篇摘要和答案的忠实度评分方法已得到公认，但它尚未被作为多段落RAG答案中单个无依据句子的检测器加以衡量，也未与训练式验证器进行对比，或对其成本进行评估。我们将其实现为一种免训练检测器：在完整上下文、无上下文以及逐个移除每个文本块的条件下对固定答案重新打分，并返回移除后最能使句子似然下降的文本块，作为候选支持段落。我们在RAGTruth、TofuEval和RAGBench数据集上，使用六个评分器，并与五个验证器（直至大语言模型（LLM）裁判）在相同输入和源级别划分下对其进行评估。按句子粒度打分对无依据句子的排序优于答案……（原文摘要至此截断）

    arXiv:2607.04223v2 Announce Type: replace-cross  Abstract: Retrieval-augmented generation (RAG) assistants summarize records in clinical and legal work, where one unsupported sentence can mislead a reader. The contrast between an output's likelihood with and without its source is an established faithfulness score for whole summaries and answers, but it has not been measured as a detector of the individual unsupported sentence in multi-passage RAG answers, against trained verifiers, or for its cost. We implement it as a training-free detector that re-scores a fixed answer under the full context, no context, and each chunk removed, and returns the chunk whose removal lowers a sentence's likelihood most as a candidate supporting passage. We evaluate it on RAGTruth, TofuEval, and RAGBench with six scorers and against five verifiers, up to a large language model (LLM) judge, on identical inputs under a source-level split. Scoring per sentence ranks unsupported sentences better than the answ
    
[^141]: 评估《克苏鲁的呼唤》桌上角色扮演游戏中大语言模型裁判的规则遵守能力

    Assessing Rule Adherence of LLM Adjudicators in Call of Cthulhu TRPG

    [https://arxiv.org/abs/2607.02802](https://arxiv.org/abs/2607.02802)

    该论文提出了基于《克苏鲁的呼唤》TRPG的多智能体对抗基准CoC-Seduce，通过“修辞注入”这一新型操纵手段，系统评估了大语言模型裁判在面对对抗性用户绕过规则时的规则遵守能力。

    

    随着大语言模型（LLM）越来越多地被部署为《克苏鲁的呼唤》（CoC）等游戏中的自主裁判，当用户意图与系统规则发生冲突时，稳健的规则遵守能力变得至关重要。然而，由于这些模型被训练为乐于助人且顺从的，它们可能容易受到一类我们称为“修辞注入”的操纵，即对抗性用户利用伪逻辑推理和权威胁迫等叙事框架技术来绕过裁判逻辑。我们提出了CoC-Seduce，一个建立在《克苏鲁的呼唤》之上的多智能体对抗基准——这是一款桌上角色扮演游戏（TRPG），其规则明确规定了哪些危险行动需要裁判裁决，而交互完全以自然语言进行。三个LLM（即GPT-5.4、Claude Sonnet 4.6、Gemini 3.5 Flash）作为对抗生成器，在4个世界设定和16个技能类别中生成了5,376个样本。随后，我们对22个目标裁判模型进行了基准测试……

    arXiv:2607.02802v2 Announce Type: replace-cross  Abstract: As LLMs are increasingly deployed as autonomous adjudicators in games such as Call of Cthulhu (CoC), robust rule adherence becomes critical when user intent conflicts with system rules. However, as these models are trained to be helpful and compliant, they may be vulnerable to a class of manipulations we term Rhetorical Injection, where adversarial users exploit narrative framing techniques such as pseudo-logical reasoning and authoritative coercion to bypass adjudication logic. We present CoC-Seduce, a multi-agent adversarial benchmark built on CoC, a Tabletop Role-Playing Game (TRPG) in which rules are explicit about which risky actions require adjudication, yet interaction remains entirely in natural language. Three LLMs, i.e., GPT-5.4, Claude Sonnet 4.6, Gemini 3.5 Flash, serve as adversarial generators producing 5,376 samples across 4 world settings and 16 skill categories. We then benchmark 22 target adjudicators against 
    
[^142]: 更密集 ≠ 更好：同策略自蒸馏在持续后训练中的局限

    Denser $\neq$ Better: Limits of On-Policy Self-Distillation for Continual Post-Training

    [https://arxiv.org/abs/2607.01763](https://arxiv.org/abs/2607.01763)

    本文通过自蒸馏策略优化（SDPO）重新审视同策略自蒸馏，发现其在持续后训练中比GRPO引发更严重的遗忘甚至崩溃，证明“更密集”的同策略监督信号并不等于更好。

    

    持续后训练使基础模型能够在获取新知识的同时保留既有能力。近期工作表明，同策略学习可以缓解遗忘，其中自蒸馏是一种尤其有吸引力的方法。我们通过自蒸馏策略优化重新审视了这一乐观论断。实验表明，当教师信号稳定且对齐良好时，SDPO能够加速领域内的特化，但难以泛化到分布之外。在持续后训练中，SDPO表现出更严重的遗忘，甚至可能崩溃；而作为更成熟、更广泛使用的同策略强化学习方法，GRPO的适应更为保守，能更好地保留先前能力。进一步的分析将这些失败与参数空间和响应空间中漂移的加剧，以及自我强化的师生回路对高频伪影的放大联系起来。因此，仅靠同策略数据……

    arXiv:2607.01763v2 Announce Type: replace-cross  Abstract: Continual post-training enables foundation models to acquire new knowledge while preserving existing capabilities. Recent work suggests that on-policy learning can mitigate forgetting, with self-distillation as a particularly attractive approach. We revisit this optimistic claim through self-distillation policy optimization (SDPO). Our experiments show that SDPO accelerates in-domain specialization when teacher signals are stable and well aligned, but struggles to generalize out of distribution. In continual post-training, SDPO exhibits greater forgetting and can even collapse, whereas GRPO, the more established on-policy reinforcement learning method, adapts more conservatively and better preserves prior capabilities. Further analyses link these failures to increased drift in parameter and response space, and to amplification of high-frequency artifacts through a self-reinforcing teacher-student loop. Thus, on-policy data alon
    
[^143]: 理解语言模型为何产生幻觉：用推理对抗先验的测试

    Understanding Why Language Models Hallucinate: Testing Reasoning Against Priors

    [https://arxiv.org/abs/2607.00447](https://arxiv.org/abs/2607.00447)

    该论文将大语言模型的幻觉解释为“推理失准”现象——预训练频率失衡使捷径路径压倒约束敏感路径，并通过潜在关键任务模型和TrapQA诊断平台来区分模型究竟是缺乏知识还是走错了推理路径。

    

    大型语言模型经常产生违反提示层面约束的幻觉性答案。一个关键的诊断问题是：这些失败究竟反映了知识的缺失，还是模型本身拥有相关信息却走上了错误的推理路径。我们将这一现象研究为“推理失准”：即提示所支持的答案与统计上显著的潜在关联所偏好的答案之间存在错位。我们通过一个潜在关键任务模型将这一观点形式化：预训练频率的不平衡会导致捷径路径压倒对约束敏感的路径，从而引发正向的推理损失。该框架预测了两种失败模式：实体消歧中的任务检索偏差，以及行动选择中的关键选择偏差。我们介绍了TrapQA，一个包含两个组成部分的受控诊断测试平台。其中ScientistQA通过补充性事实探针测试相似科学家之间的消歧，而Real-Life（摘要在此处截断）

    arXiv:2607.00447v2 Announce Type: replace  Abstract: Large language models often produce hallucinated answers that violate prompt-level constraints. A key diagnostic question is whether these failures reflect missing knowledge, or whether the model has the relevant information but follows the wrong inference path. We study this phenomenon as inference misalignment: a mismatch between the answer supported by the prompt and the answer favored by statistically salient latent associations. We formalize this view with a latent key-task model, in which pretraining-frequency imbalance can cause a shortcut path to dominate the constraint-sensitive path and induce positive inference loss. The framework predicts two failure modes: task-retrieval bias in entity disambiguation and key-selection bias in action choice. We introduce TrapQA, a controlled diagnostic testbed with two components. ScientistQA tests disambiguation among similar scientists with supplementary factual probes, while Real-Life 
    
[^144]: 不用GPU能走多远？跨问答、对话与摘要任务的轻量级幻觉检测系统化基准测试

    How Far Can You Get Without a GPU? A Systematic Benchmark of Lightweight Hallucination Detection Across Question Answering, Dialogue, and Summarisation

    [https://arxiv.org/abs/2606.29809](https://arxiv.org/abs/2606.29809)

    本研究系统基准测试了四种无需GPU的轻量级幻觉检测方法（ROUGE-L、语义相似度、BERTScore和NLI检测器）及其集成方案，在HaluEval的问答、对话和摘要任务上验证了基于公开模型的CPU可行方法可作为资源受限场景下幻觉检测的实用替代方案。

    

    幻觉检测已成为大规模可信AI部署的迫切需求。最精确的检测方法依赖于GPU密集型推理、专有API调用或对生成模型的白盒访问，这使得资源受限的研究人员和从业者难以使用。我们探索了一种实用的替代方案：仅使用基于公开模型的轻量级、CPU可运行的方法，幻觉检测能达到怎样的效果？我们对四种此类检测器进行了基准测试：ROUGE-L、语义相似度、BERTScore，以及基于FEVER训练的DeBERTa模型的自然语言推理（NLI）检测器，此外还测试了相似度与NLI的分数级集成方法。我们在HaluEval基准的全部三项任务上进行评估：问答（QA）、对话和摘要。我们在留出的验证集上进行校准，在每个任务的2000个测试实例上进行评估，并报告bootstrap置信区间。

    arXiv:2606.29809v2 Announce Type: replace-cross  Abstract: Hallucination detection has become a pressing requirement for trustworthy AI deployment at scale. The most accurate detection methods depend on GPU-intensive inference, proprietary API calls, or white-box access to the generating model, putting them out of reach for resource-constrained researchers and practitioners. We explore a practical alternative: how well can hallucination detection perform using only lightweight, CPU-feasible methods built on public models? We benchmark four such detectors, ROUGE-L, semantic similarity, BERTScore, and a Natural Language Inference (NLI) detector based on a FEVER-trained DeBERTa model, together with a score-level ensemble of similarity and NLI. We evaluate them across all three tasks of the HaluEval benchmark: question answering (QA), dialogue, and summarisation. We calibrate on a held-out validation split, evaluate on 2,000 test instances per task, and report bootstrap confidence interval
    
[^145]: Morpheus：面向土耳其语的形态感知神经分词器与词嵌入生成器

    Morpheus: A Morphology-Aware Neural Tokenizer and Word Embedder for Turkish

    [https://arxiv.org/abs/2606.18717](https://arxiv.org/abs/2606.18717)

    Morpheus 是一个面向土耳其语的形态感知神经分词器与词嵌入生成器，它通过可微分泊松-二项动态规划实现无损可逆的词素级分词，并能在同一次前向传播中同时输出分词结果和结构化词嵌入。

    

    土耳其语是一种黏着语：语义由词素承载，然而驱动现代语言模型的子词分词器却依据语料库统计来切分单词，这导致语义负载丰富的后缀被碎片化，而且（就WordPiece和基于规则的分析器而言）无法将其输出解码还原为原始文本。本文提出了**Morpheus**，一个针对土耳其语的神经词素边界模型，它同时是一个无损的、形态感知的分词器和一个词嵌入生成器。一个可微分的泊松-二项动态规划在训练期间将逐字符的边界概率转化为软性词素归属，在推理时则转化为精确的切分结果，且无需任何字符串规范化，因此 decode(encode(w)) = w 在结构上天然成立。由于该模型是神经网络的，同一次前向传播在完成分词的同时还能输出结构化的词嵌入。在可逆分词器——即唯一适用于生成任务的分词器——之中，Morpheus 在……（摘要原文在此处截断）

    arXiv:2606.18717v2 Announce Type: replace-cross  Abstract: Turkish is agglutinative: meaning is carried by morphemes, yet the subword tokenizers that drive modern language models split words by corpus statistics, fragmenting semantically loaded suffixes and -- in the case of WordPiece and rule-based analyzers -- failing to decode their output back to the original text. This paper presents \textbf{Morpheus}, a neural morpheme-boundary model for Turkish that is at once a lossless, morphology-aware tokenizer and a word-embedding producer. A differentiable Poisson-binomial dynamic program turns per-character boundary probabilities into soft morpheme memberships during training and exact segments at inference, with no string normalization, so $\mathrm{decode}(\mathrm{encode}(w)) = w$ holds by construction. Because the model is neural, the same forward pass that tokenizes also emits a structured word embedding. Among reversible tokenizers -- the only ones valid for generation -- Morpheus att
    
[^146]: 最后但同样重要：多模态KV缓存压缩中的边界注意力校准

    Last But Not Least: Boundary Attention CalibratiON for Multimodal KV Cache Compression

    [https://arxiv.org/abs/2606.14782](https://arxiv.org/abs/2606.14782)

    BACON通过结合最后查询注意力与观察窗口注意力，并抑制噪声，显著提升了多模态KV缓存压缩的准确性，尤其在激进压缩场景下平均提升7.5%。

    

    arXiv:2606.14782v3 公告类型：替换交叉 摘要：多模态大型语言模型（MLLMs）在视觉-语言推理方面表现强劲，但在长视觉上下文下会产生大量KV缓存和高解码延迟。现有压缩方法依赖观察窗口注意力来稳定估计令牌重要性，但这种聚合可能稀释稀疏的关键证据，并在激进压缩下丢弃与答案相关的令牌。我们识别出最后查询注意力作为恢复此类证据的补充信号，尽管其不相关信号可能引入额外噪声。我们提出BACON，一种即插即用方法，通过层内一致性和层间持久性校准观察窗口注意力与最后查询证据，同时抑制噪声。在不同基准、模型、预算和压缩方法下，BACON在最激进预算下平均将多模态KV缓存压缩提升7.5%，最高提升达30.9%。

    arXiv:2606.14782v3 Announce Type: replace-cross  Abstract: Multimodal Large Language Models (MLLMs) achieve strong vision-language reasoning but incur large KV caches and high decoding latency with long visual contexts. Existing compression methods rely on observation window attention for stable token importance estimation, yet this aggregation can dilute sparse critical evidence and discard answer-relevant tokens under aggressive compression. We identify last query attention as a complementary signal for recovering such evidence, though its irrelevant signals may introduce additional noise. We propose BACON, a plug-and-play method that calibrates observation window attention with last query evidence while suppressing noise through intra-layer coherence and inter-layer persistence. Across diverse benchmarks, models, budgets, and compression methods, BACON improves multimodal KV-cache compression by 7.5% on average under the most aggressive budget, with gains up to 30.9%.
    
[^147]: 面向大语言模型推理的提示引导多样化策略优化

    Hint-Guided Diversified Policy Optimization for LLM Reasoning

    [https://arxiv.org/abs/2606.03021](https://arxiv.org/abs/2606.03021)

    提出HDPO方法，让模型先列举多个候选解题思路作为提示，再选择最可靠的一个进行深入推理，通过结构化推理冷启动和提示引导多样化强化学习两个阶段提升大语言模型的推理能力。

    

    大语言模型的最新进展展示了令人印象深刻的推理能力，其中带可验证奖励的强化学习是一种有前景的增强策略。然而，现有的奖励机制仅局限于结果层面的正确性，缺乏明确的信号来引导模型考虑多样化的解决方案。相比之下，人类解决问题通常涉及评估多种潜在方法并选择最可靠的解决方案，这是当前RLVR框架未明确激励的一种认知过程。受此启发，我们提出了提示引导多样化策略优化（HDPO），允许模型首先列出所有潜在的候选解决方案大纲作为提示，然后选择最可靠的一个进行进一步推理。HDPO包含结构化推理冷启动和提示引导多样化强化学习两个阶段，以激励模型……

    arXiv:2606.03021v4 Announce Type: replace  Abstract: Recent developments in Large Language Models (LLMs) have showcased impressive reasoning capabilities, with Reinforcement Learning with Verifiable Rewards (RLVR) being a promising enhancement strategy. However, existing reward mechanisms are constrained to the outcome-level correctness and lack explicit signals to guide the model to consider diverse solutions. In contrast, human problem solving typically involves evaluating multiple potential approaches and selecting the most reliable solution, a cognitive process that current RLVR frameworks do not explicitly incentivize. Inspired by this, we propose Hint-Guided Diversified Policy Optimization (HDPO), allowing the model to first list all potential candidate solution outlines as hints and then select the most reliable one for further reasoning. HDPO comprises two stages of Cold Start for Structured Reasoning and Hint-Guided Diversified Reinforcement Learning to incentivize the model t
    
[^148]: 来自1913年的语言模型：在历史文本上进行预训练

    A Language Model from 1913: Pretraining on Historical Text

    [https://arxiv.org/abs/2606.02991](https://arxiv.org/abs/2606.02991)

    该论文提出了TypewriterLM，一个在1913年前历史文本上预训练的72.4亿参数语言模型，通过构建540亿token的时间过滤历史语料库、基于历史词汇约束的指令微调方法以及包含2,344个事件的History-Event评估基准，实现了具有明确1913年知识截止时间且语言理解性能合理的时间定位语言模型。

    

    尽管现代语言模型越来越依赖规模不断扩大的网络语料库，我们表明在数据受限的环境下，在历史文本（例如1913年之前的文本）上进行预训练，可以产生一个具有时间定位特性的语言模型，且该模型在语言理解方面仍能表现出合理的性能。然而，开发历史语言模型需要解决数据质量、防止后训练中的时间泄漏以及构建时间对齐评估等挑战。我们应对了这些挑战，并预训练了TypewriterLM——一个具有1913年知识截止时间的72.4亿参数模型。我们构建了TypewriterCorpus，一个经过广泛时间过滤的540亿token历史语料库；提出了基于词汇约束的指令微调方法，将所有回复限制在历史源文档的词汇范围内；并引入了History-Event，一个包含2,344个事件的基准，用于同时评估模型能力与知识截止时间的遵循度。我们发布了TypewriterLM及所有相关资源。

    arXiv:2606.02991v2 Announce Type: replace-cross  Abstract: While modern language models increasingly rely on ever-larger web corpora, we show that pretraining on historical text (e.g., pre-1913 text) in a data-constrained setting can produce a temporally grounded language model that still shows reasonable performance on language understanding. However, developing History LMs requires addressing challenges in data quality, preventing temporal leakage in post-training, and constructing temporally aligned evaluations. We address these challenges and pretrain TypewriterLM, a 7.24B-parameter model with a 1913 knowledge cutoff. We construct TypewriterCorpus, a 54B-token historical corpus with extensive temporal filtering, propose lexically grounded instruction tuning that constrains all responses to vocabulary from historical source documents, and introduce History-Event, a benchmark of 2,344 events for evaluating both competence and cutoff adherence. We release TypewriterLM and all associat
    
[^149]: 已编码但未被路由：解释科学声明验证中的表格-图表差距

    Encoded but Not Routed: Explaining the Table-Chart Gap in Scientific Claim Verification

    [https://arxiv.org/abs/2606.01679](https://arxiv.org/abs/2606.01679)

    该论文发现多模态大语言模型在科学声明验证中对图表表现不佳的原因，并非模型无法从图表中提取信息，而是提取到的图表信息虽已被编码在模型中间表示中，却未被有效路由到预测位置。

    

    多模态大语言模型越来越多地被用于辅助科学同行评审，其核心要求之一是验证论文中的声明是否得到其证据的支持。先前的研究表明，当证据以表格形式呈现时，模型在此任务上的表现显著优于证据以相同底层数据绘制的图表形式时的表现。这引出一个问题：模型是无法从图表中提取信息，还是能够提取信息但在形成预测时未能加以利用？我们通过在三个开源视觉语言模型上，对承载相同底层数据的表格和图表证据进行分层线性探测和注意力分析来研究这一问题。我们找到了支持后者的持续性证据：图表信息已被编码在模型的中间表示中，但未能到达预测位置，而这种差距在表格中并不存在，并且在所有测试条件下均成立。注意力分析进一步揭示了这种脱节……

    arXiv:2606.01679v2 Announce Type: replace  Abstract: Multimodal LLMs are increasingly used to assist scientific peer review, where a core requirement is verifying whether claims in a paper are supported by its evidence. Prior work has shown that models perform substantially better at this task when the evidence is a table than when it is a chart of the same underlying data. This raises the question of whether models fail to extract information from charts, or do they extract it but fail to use it when forming their prediction? We study this question through layer-wise linear probing and attention analysis on three open-weight VLMs over table and chart evidence, representing the same underlying data. We find consistent evidence for the latter. Chart information is encoded in the models' intermediate representations but does not reach the prediction position, a gap that is absent for tables and holds across all conditions tested. Attention analysis further reveals that this disconnect ta
    
[^150]: 反事实证据审计可预测大语言模型智能体对排序上下文的易感性

    Counterfactual Evidence Audits Predict LLM-Agent Susceptibility to Ranked Context

    [https://arxiv.org/abs/2606.00914](https://arxiv.org/abs/2606.00914)

    该论文提出一种反事实证据审计协议，通过让智能体面对两组镜像的五文档集合并测量其决策差异，能够高精度预测LLM智能体在面对45份文档的单边排序上下文时的易感性。

    

    大语言模型（LLM）智能体越来越多地依据由上游系统组装的证据做出决策：检索器选择文档，推荐系统选择帖子，记忆系统选择过往事件。现有的评估通常将这些证据视为固定不变的，从而遗漏了一类失败情形：单独看来都很平常的内容项，组合起来却形成了系统性的单边上下文。我们提出一种反事实证据审计方法：让智能体分别面对两组互为镜像的五份文档，测量其在六个下游决策上的差异，并利用这种对比来预测它面对互不相交的45份文档上下文时的反应。该评估协议在测试三个留出的开源权重模型家族之前已被冻结。在18个留出的模型-任务组合中，五份文档的效应对完整上下文效应的预测达到Spearman相关系数rho=.855（p<.001），相对于零效应预测器将平均绝对预测误差降低了62%，并在13个实质性效应中正确恢复了其中12个的方向。应审稿人要求补充的事后任务均值基线……（原文摘要在此处截断）

    arXiv:2606.00914v2 Announce Type: replace  Abstract: LLM agents increasingly decide from evidence assembled by upstream systems: retrievers choose documents, recommenders choose posts, and memory systems choose prior events. Existing evaluations usually hold this evidence fixed, missing failures in which individually ordinary items form a systematically one-sided context. We introduce a counterfactual evidence audit: expose an agent to two mirrored sets of five documents, measure the difference in six downstream decisions, and use that contrast to predict its response to disjoint 45-document contexts. The protocol was frozen before testing three held-out open-weight model families. Across 18 held-out model-task cells, five-document effects predict full-context effects with Spearman rho=.855 (p<.001), reduce mean absolute prediction error by 62% relative to a zero-effect predictor, and recover the direction of 12 of 13 material effects. A reviewer-requested post-hoc task-mean baseline i
    
[^151]: 论大语言模型适应性的局限：模型内化先验对标注任务性能的影响

    On the Limits of LLM Adaptability: Impact of Model-Internalized Priors on Annotation Task Performance

    [https://arxiv.org/abs/2606.00467](https://arxiv.org/abs/2606.00467)

    提出“定义特定熟悉度”（DSF）指标，证明大语言模型内化先验与任务定义的对齐程度能显著预测其标注性能，且提示中的额外信息难以纠正模型零样本的“决策粘性”错误。

    

    大语言模型（LLM）越来越多地被用于零样本标注和“LLM作为评判者”任务，但其可靠性取决于模型内化的先验与用户所提供指令之间的交互方式。我们从三个维度研究了这种交互：(1) LLM对数据和任务定义的熟悉程度与其性能之间的关系；(2) 提示中的额外信息能否纠正零样本错误（即“决策粘性”）；(3) 模型对不一致任务定义的易感性。我们提出了“定义特定熟悉度”（DSF）这一概念，用于衡量模型所引出的概念与目标定义之间的对齐程度。在九个大语言模型和六个毒性数据集（五个主要数据集加一个额外的鲁棒性数据集）上的实验表明，在控制数据集身份后，DSF能够预测标注性能（偏相关系数 r=+0.41）。这种关联在所有测试的提示条件下均保持为正。相比之下……（原文摘要在此处截断）

    arXiv:2606.00467v2 Announce Type: replace-cross  Abstract: Large Language Models (LLMs) are increasingly used for zero-shot annotation and LLM-as-a-judge tasks, yet their reliability hinges on how model-internalized priors interact with user-provided instructions. We investigate three dimensions of this interaction: (1) how an LLM's familiarity with data and task definitions relates to performance, (2) whether additional information in prompts can correct zero-shot errors ("decision stickiness"), and (3) model susceptibility to misaligned task definitions. We introduce Definition-Specific Familiarity (DSF), which measures alignment between a model's elicited concept and the target definition. Across nine LLMs and six toxicity datasets (five primary datasets plus an additional robustness dataset), DSF predicts annotation performance after controlling for dataset identity (partial $r=+0.41$). This association remains positive across all prompting conditions tested. In contrast, three com
    
[^152]: 专业化而不遗忘：分析多语言模型适应中的知识保持

    Specializing Without Forgetting: Analyzing Knowledge Preservation in Multilingual Model Adaptation

    [https://arxiv.org/abs/2606.00284](https://arxiv.org/abs/2606.00284)

    本研究通过插值定位发现中间层是多语言持续预训练中灾难性遗忘的关键，并提出层冻结、层范围正则化等基于层信息的策略，使模型在适应新语言的同时有效保留原有能力。

    

    持续预训练（CPT）是将大语言模型扩展到新语言的一种实用方法，但朴素微调往往会通过灾难性遗忘侵蚀模型已有的能力。我们研究了哪些模型层驱动了这种权衡，以及在这些层上的干预能否在模型适应过程中引导知识保持。我们对五个语言族上 gemma-3-4b 模型在 CPT 前后的模型状态进行插值，以定位阅读理解和翻译任务中的遗忘现象，发现中间层的回退能带来最大的阅读理解能力恢复，而翻译效果则因语言族和翻译方向而异。基于这些发现，我们评估了利用层信息来缓解遗忘的多种 CPT 策略：层冻结、层范围 L2 正则化、事后层回退以及模型汤（model souping），并将所有策略与联合多语言及语言族特定的朴素 CPT 基线进行比较。我们发现……

    arXiv:2606.00284v2 Announce Type: replace  Abstract: While continual pretraining (CPT) is a practical way to extend large language models to new languages, na\"ive finetuning often erodes existing capabilities through catastrophic forgetting. We investigate which model layers drive this trade-off, and whether interventions at these layers can guide knowledge preservation during adaptation. We interpolate gemma-3-4b model states before and after CPT on five language families to localize forgetting on reading comprehension and translation, finding that middle-layer reversion yields the largest comprehension recovery, while translation effects vary by language family and direction. Guided by these findings, we evaluate CPT strategies that leverage this layer information to mitigate forgetting: layer freezing, layer-range L2 regularization, post-hoc layer reversion, and model souping, comparing all strategies against joint multilingual and family-specific vanilla CPT baselines. We find tha
    
[^153]: 抵御智能体式重识别攻击的大语言模型文本匿名化

    LLM Anonymization Against Agentic Re-Identification

    [https://arxiv.org/abs/2605.30848](https://arxiv.org/abs/2605.30848)

    该论文提出AURA框架，采用“掩码-重构”解耦设计并结合对抗性隐私与效用双重检查，使匿名化文本既能抵御具备网络搜索能力的智能体重识别攻击，又能保留下游分析效用。

    

    arXiv:2605.30848v3 公告类型：replace-cross 摘要：具备网络搜索能力的智能体式大语言模型改变了文本匿名化的威胁模型：原本微弱的上下文线索可以成为可交叉引用的重识别证据，而这些同样的细节又承载着文本的下游分析价值。现有防御方法要么直接移除显式标识符，要么通过扰动文本实现形式化隐私保护，要么仅用不具备联网能力的推理模型测试改写后的文本，对于“抵御智能体网络搜索重识别”与“保留文本效用”之间的运作区域仍缺乏充分探索。我们提出了AURA（Anonymization with Utility-Retention Adaptation，效用保留自适应匿名化），这是一个由大语言模型驱动的“掩码-重构”框架，它将隐私定位与保留效用的重构过程解耦，并通过对抗性的隐私攻击检查和效用保留检查来筛选候选方案。我们在真实用户访谈转录文本上评估了AURA，使用由网络搜索智能体执行的重识别攻击，以及……

    arXiv:2605.30848v3 Announce Type: replace-cross  Abstract: Agentic LLMs with web search change the threat model for text anonymization: weak contextual cues can become cross-referenceable evidence for re-identification, yet those same details also carry downstream analytic value of the text. Existing defenses either remove explicit identifiers, perturb text for formal privacy, or test rewritten text against non-web inference models, leaving underexplored the operating region between resistance to agentic web-search re-identification and utility retention. We introduce AURA (\textbf{A}nonymization with \textbf{U}tility-\textbf{R}etention \textbf{A}daptation), an LLM-powered \textit{mask-reconstruct} framework that decouples privacy localization from utility-preserving reconstruction and selects candidates with adversarial privacy and utility-retention checks. We evaluate AURA on real-user interview transcripts using re-identification attacks carried out by web-search agents, along with 
    
[^154]: 表面的LLM临床分诊失败从何而来？定位多选题格式效应

    Where Do Apparent LLM Clinical Triage Failures Arise? Localizing the Multiple-Choice Format Effect

    [https://arxiv.org/abs/2605.29889](https://arxiv.org/abs/2605.29889)

    利用稀疏自编码器分析，该研究发现LLM在多选题式临床分诊中的表现下降并非源于对病例医学信息的处理失败，而是发生在答案映射阶段——多选题答题框架在决策标记处抑制了本已可解码的急诊分级信息。

    

    采用临床医生撰写的分诊案例对大语言模型进行评估的研究显示，在受限的多选题测试条件下，模型存在明显的分诊不足现象。然而，当以自由文本形式生成回答时，同一临床案例上的模型表现可能发生变化。我们检验这种格式效应究竟是在模型处理病例的过程中出现，还是在临床信息被映射到最终答案时出现。通过分析 Gemma 3 4B/12B IT 和 Qwen3-8B 中的稀疏自编码器（SAE）特征，我们发现：在两种格式下，医学特征都会在共享的临床叙述文本上激活，但在多选题的决策标记处却处于不活跃状态。在两种格式下，急诊分级信息均可从病例表示中被线性解码，ROC-AUC 达到 0.95–1.00，且格式之间无显著差异，但该信息在决策标记处被削弱。自然语言形式的自编码器描述与顶级特征刻画表明，该决策标记与多选题的答题框架相关联。（摘要在此处截断）

    arXiv:2605.29889v2 Announce Type: replace-cross  Abstract: LLM evaluations using clinician-authored triage vignettes have reported substantial under-triage under constrained multiple-choice testing. Yet model performance on the same clinical cases can change when responses are generated in free text. We test whether this format effect appears while the case is processed or when clinical information is mapped to the final answer. Using sparse-autoencoder (SAE) features in Gemma 3 4B/12B IT and Qwen3-8B, we find that medical features fire on the shared clinical narrative under both formats but are inactive at the multiple-choice decision token. Emergency-tier information is linearly decodable from vignette representations with ROC-AUC $0.95$--$1.00$ under both formats, with no significant format difference, but is attenuated at the decision token. Natural-language autoencoder verbalization and top-feature characterization associate that token with the multiple-choice scaffold. In a direc
    
[^155]: 你只需对齐一次：通过种子智能体在多智能体系统中传播合作行为

    You Only Align Once: Propagating Cooperative Behaviors in Multi-Agent Systems through Seed Agents

    [https://arxiv.org/abs/2605.27586](https://arxiv.org/abs/2605.27586)

    提出仅需对齐单个“种子智能体”（通过对齐的教师模型蒸馏至Qwen3-14B），即可纯粹通过自然语言交互在多智能体系统中传播合作行为（对齐传播），将团队合作率从24.8%提升至62.2%，并能零样本迁移到其他仿真环境。

    

    在分布式开放多智能体系统中确保智能体行为的对齐仍然是一个挑战，尤其是当群体规模不断增长且可能存在未对齐的智能体时。我们证明，单个已对齐的智能体可以纯粹通过自然语言交互，将合作行为传播给未经修改的智能体，我们将这种现象称为“对齐传播”。我们在红黑游戏中研究这一现象，这是一种基于团队的迭代囚徒困境博弈，队友们通过商议和投票来决定团队的集体行动。通过将教师模型的合作推理能力和说服性对话蒸馏到Qwen3-14B中，我们得到了一个种子智能体，当它被置于四个未经修改的队友之中时，能将合作率从24.8%提升至62.2%（超过翻倍），其表现超越了教师模型和原生的Gemini-3.1-Pro。值得注意的是，仅在红黑游戏上训练的种子智能体能够零样本迁移到Sugarscape——一个具有空间特性的生存模拟环境。

    arXiv:2605.27586v2 Announce Type: replace-cross  Abstract: Ensuring aligned agent behaviors in distributed open multi-agent systems remains challenging, especially as populations grow and unaligned agents may exist. We show that a single aligned agent can propagate cooperative behaviors to unmodified agents purely through natural-language interaction, a phenomenon we term Alignment Propagation. We study this in the Red-Black Game, a team-based iterated Prisoner's Dilemma in which teammates deliberate and vote to determine their team's collective action. By distilling the cooperative reasoning and persuasive dialogues of a teacher model into Qwen3-14B, we obtain a seed agent that, when placed among four unmodified teammates, more than doubles the cooperation rate from 24.8% to 62.2%, outperforming the teacher model and a vanilla Gemini-3.1-Pro. Remarkably, a seed trained exclusively on the Red-Black Game transfers zero-shot to Sugarscape, a spatially grounded survival simulation with pa
    
[^156]: EchoDistill：通过噪声到干净的自蒸馏实现鲁棒的大型音频语言模型

    EchoDistill: Robust Large Audio Language Models via Noisy-to-Clean Self-Distillation

    [https://arxiv.org/abs/2605.23954](https://arxiv.org/abs/2605.23954)

    EchoDistill提出一种噪声到干净的自蒸馏框架，在后训练中以干净音频作为特权信息，通过掩码响应token蒸馏、任务门控一致性塑形和教师参考的组相对优化，使大型音频语言模型在噪声环境下更鲁棒，且推理时无额外开销。

    

    大型音频语言模型（LALMs）仍然容易受到声学噪声的影响，噪声会掩盖与任务相关的证据并产生不可靠的响应。我们提出了EchoDistill，这是一种噪声到干净的自蒸馏框架，在后训练阶段将干净音频作为特权信息加以利用。以噪声输入的学生模型采样能够反映其推理时行为的候选响应，而同一骨干网络的冻结副本则处理对应的干净音频。EchoDistill结合了掩码响应token蒸馏、任务门控一致性塑形以及教师参考的组相对优化，使噪声输入下的生成结果与干净条件下的语义保持对齐。推理时仅保留学生模型，不引入任何额外的推理开销。在三个LALM骨干网络和三个音频领域、信噪比为-10dB的条件下，EchoDistill相比最强基线将噪声输入下的平均准确率提升了1.63个百分点。在Qwen2.5-Omni上，它

    arXiv:2605.23954v2 Announce Type: replace-cross  Abstract: Large Audio Language Models (LALMs) remain vulnerable to acoustic noise, which can obscure task-relevant evidence and produce unreliable responses. We propose EchoDistill, a noisy-to-clean self-distillation framework that uses clean audio as privileged information during post-training. A noisy-input student samples candidate responses reflecting its inference-time behavior, while a frozen copy of the same backbone processes the corresponding clean audio. EchoDistill combines masked response-token distillation, task-gated consistency shaping, and teacher-referenced group-relative optimization to align noisy-input generation with clean-conditioned semantics. Only the student is retained at inference time, introducing no additional inference cost. Across three LALM backbones and three audio domains at -10dB, EchoDistill improves average noisy-input accuracy by 1.63 percentage points over the strongest baseline. On Qwen2.5-Omni, it
    
[^157]: FastKernels：在生产环境中对GPU内核生成进行基准测试

    FastKernels: Benchmarking GPU Kernel Generation in Production

    [https://arxiv.org/abs/2605.23215](https://arxiv.org/abs/2605.23215)

    FastKernels提出了一个包含384个任务的生产级GPU内核生成基准，通过组合层次结构覆盖94.6%的HuggingFace Transformers架构，并直接在生产执行路径上以框架官方发布的内核为基准对候选内核进行内核级和端到端评分。

    

    基于大语言模型（LLM）的GPU内核生成智能体正在迅速发展，但它们所优化的基准测试往往在隔离环境中评估内核，使用合成输入和薄弱的基线，从而奖励那些在真实推理系统中会失效或无法体现的沙盒加速效果。我们提出了FastKernels，这是一个包含384个任务的基准测试，这些任务取自8个类别中的47个代表性架构，其内核足以重新实现94.6%（472/499）的HuggingFace Transformers架构，且输出与原生实现相匹配。每个任务都镜像了相应生产模块的接口，并以生产框架实际发布的内核作为评分基准；任务构成了一个组合层次结构，从底层原语到完整模型，其中高层模块会导入低层模块。候选内核在内核层面以及其所属模型的内部进行端到端评分，并在生产执行路径上进行评估，MacroEval则聚合经过校准的（原文摘要在此处截断）……

    arXiv:2605.23215v2 Announce Type: replace-cross  Abstract: LLM-based agents for GPU kernel generation are advancing rapidly, but the benchmarks they optimize against evaluate kernels in isolation, with synthetic inputs and weak baselines, rewarding sandbox speedups that break or vanish in real inference systems. We introduce FastKernels, a benchmark of 384 tasks drawn from 47 representative architectures across 8 categories, whose kernels suffice to reimplement 94.6% (472/499) of HuggingFace Transformers architectures with outputs matching the native implementations. Each task mirrors the interface of the corresponding production module and is scored against the kernels production frameworks ship, and tasks form a compositional hierarchy, from primitives to full models, in which higher-level modules import lower-level ones. Candidates are scored at the kernel level and end to end inside the models they come from, on the production execution path, and MacroEval aggregates calibrated cor
    
[^158]: HyperLogic：一个高难度、前向构建、答案由执行推导得出的中文逻辑推理基准

    HyperLogic: A Hard, Forward-Authored Chinese Logical Reasoning Benchmark with Execution-Derived Answers

    [https://arxiv.org/abs/2605.19597](https://arxiv.org/abs/2605.19597)

    HyperLogic通过将题目编写与答案生成分离的多智能体前向构建流水线，构建了一个高难度中文逻辑推理基准，答案由求解器执行推导得出，并将七个前沿模型的性能拉开33分差距。

    

    现有的逻辑基准主要衡量模型直接回答推理问题的能力。可扩展的基准通常从形式结构生成文本，这使答案易于计算，但在问题编写之前就固定了形式化方式。前向构建保留了寻找忠实形式化的挑战，但使难度和答案的可靠性更难控制。我们提出HyperLogic，一个将问题编写与答案生成分离的前向构建流水线。多智能体工作流在不解题的前提下强化本科生编写的中文种子题目；来自不同模型家族的两个智能体独立地将每个完成的题目翻译为可执行的有限域模型；其编码及由求解器推导出的答案在人类专家监督下经过分层、智能体辅助的裁定。HyperLogic-Base包含195道题目和922个子问题，并将七个前沿模型的成绩拉开了33分。

    arXiv:2605.19597v2 Announce Type: replace  Abstract: Existing logic benchmarks primarily measure models' ability to answer reasoning questions directly. Scalable benchmarks often generate text from formal structures, which makes answers easy to compute but fixes the formalization before the problem is written. Forward construction preserves the challenge of finding a faithful formalization, yet makes difficulty and answer reliability harder to control. We introduce HyperLogic, a forward-construction pipeline that separates problem authoring from answer generation. A multi-agent workflow hardens undergraduate-authored Chinese seeds without solving them; two agents from different model families independently translate each finished item into executable finite-domain models; their encodings and solver-derived answers undergo layered, agent-assisted adjudication under human-expert oversight. HyperLogic-Base contains 195 items and 922 sub-questions and separates seven frontier models by 33.
    
[^159]: HINT-SD：面向长时程智能体的定向后见自蒸馏

    HINT-SD: Targeted Hindsight Self-Distillation for Long-Horizon Agents

    [https://arxiv.org/abs/2605.17873](https://arxiv.org/abs/2605.17873)

    HINT-SD通过利用完整轨迹后见之明精准定位失败相关动作，并仅对定向动作片段进行反馈条件蒸馏，避免了逐回合生成反馈的低效问题，在长时程智能体任务中显著提升性能。

    

    arXiv:2605.17873v2 公告类型：交叉替换 摘要：使用强化学习训练长时程LLM智能体具有挑战性，因为稀疏的结果奖励能揭示任务是否成功，但无法指出哪些中间动作导致了该结果，或应如何纠正这些动作。近期方法通过从回合级动作-输出信号生成奖励或文本提示，或使用反馈条件自蒸馏来缓解此问题。然而，在每回合生成反馈效率低下，因为许多中间回合可能已经成功或中性，而在固定或错位的回合应用反馈往往无法监督导致失败的动作。为弥合这一差距，我们提出HINT-SD，一种定向自蒸馏框架，利用完整轨迹的后见之明选择与失败相关的动作，并仅对定向动作片段应用反馈条件蒸馏。在BFCL v3和AppWorld上的实验表明，我们的方法优于密集反馈方法。

    arXiv:2605.17873v2 Announce Type: replace-cross  Abstract: Training long-horizon LLM agents with reinforcement learning is challenging because sparse outcome rewards reveal whether a task succeeds, but not which intermediate actions caused the outcome or how they should be corrected. Recent methods alleviate this issue by generating rewards or textual hints from turn-level action-output signals, or by using feedback-conditioned self-distillation. However, generating feedback at every turn is inefficient when many intermediate turns are already successful or neutral, and applying feedback at a fixed or misaligned turn often fails to supervise the actions that contributed to the failure. To bridge this gap, we propose HINT-SD, a targeted self-distillation framework that uses full-trajectory hindsight to select failure-relevant actions and applies feedback-conditioned distillation only to targeted action spans. Experiments on BFCL v3 and AppWorld show that our method outperforms the dense
    
[^160]: 递归智能体优化

    Recursive Agent Optimization

    [https://arxiv.org/abs/2605.06639](https://arxiv.org/abs/2605.06639)

    RAO提出了一种强化学习方法，通过训练智能体递归地生成并委派子任务给自身的新实例来实现推理时的分治扩展，使模型能够突破上下文窗口限制、泛化到远难于训练任务的问题，并降低实际运行时间。

    

    我们提出了递归智能体优化（Recursive Agent Optimization, RAO），这是一种用于训练递归智能体的强化学习方法：递归智能体能够递归地生成并将子任务委派给自身的新实例。递归智能体实现了一种推理时扩展算法，通过分治法使智能体能够自然地扩展到更长的上下文，并泛化到更困难的问题。RAO提供了一种训练模型以充分利用这种递归推理的方法，教会智能体何时以及如何进行委派和沟通。我们发现，以这种方式训练的递归智能体具有更好的训练效率，能够扩展到超出模型上下文窗口的任务，泛化到比训练任务难得多的问题，并且与单智能体系统相比可以减少实际运行时间。

    arXiv:2605.06639v2 Announce Type: replace-cross  Abstract: We introduce Recursive Agent Optimization (RAO), a reinforcement learning approach for training recursive agents: agents that can spawn and delegate sub-tasks to new instantiations of themselves recursively. Recursive agents implement an inference-time scaling algorithm that naturally allows agents to scale to longer contexts and generalize to more difficult problems via divide-and-conquer. RAO provides a method to train models to best take advantage of such recursive inference, teaching agents when and how to delegate and communicate. We find that recursive agents trained in this way enjoy better training efficiency, can scale to tasks that go beyond the model's context window, generalize to tasks much harder than the ones the agent was trained on, and can enjoy reduced wall-clock time compared to single-agent systems.
    
[^161]: 有用的特征，反向的得分：语言模型轨迹中的分布外（OOD）检测

    Useful Features, Backward Scores: OOD in Language-Model Trajectories

    [https://arxiv.org/abs/2605.00269](https://arxiv.org/abs/2605.00269)

    该论文的核心发现是，能区分输入组的特征并不一定能产生有用的OOD异常排序——在语言模型轨迹中，可区分特征对应的距离得分甚至会出现反转（异常组中心更远但散布更紧），且这一现象在毒性、反讽等多个数据集上均稳定存在。

    

    分布外（OOD）检测器用于对输入进行优先级排序以便进一步检查。然而，能够区分输入组的特征并不一定能产生有用的异常排序。我们在文本长度控制和固定得分方向的条件下，分析语言模型轨迹中的这一差距。在Spam（垃圾信息）开发数据上，D²HScore的输入适配版本在长度匹配后，AUROC从原始的0.919降至0.530。在长度匹配的留出HateSpeech（仇恨言论）输入上，相同的特征使有标签的线性分类器达到AUROC 0.644，但基于分布内（ID）拟合的距离得分仅为0.444。ToxicChat数据集显示出同样的对比。特征选择和骨干网络的对照实验保留了主要的反转模式。冻结的Civil Comments和TweetEval反讽测试也出现反转（0.467和0.435），将这一发现扩展到了毒性检测之外。在这些对比中，异常组的中心距离更远，但散布更紧。一种有标签的、固定中心的特征空间干预改变了排序：均衡散布有助于某些任务……（摘要截断）

    arXiv:2605.00269v2 Announce Type: replace  Abstract: Out-of-distribution (OOD) detectors prioritize inputs for closer inspection. Yet features that distinguish input groups need not yield a useful anomaly ranking. We analyze this gap in language-model trajectories under text-length control and fixed score directions. On Spam development data, an input adaptation of D^2HScore falls from raw AUROC 0.919 to 0.530 after length matching. On length-matched, held-out HateSpeech inputs, the same features yield AUROC 0.644 for a labeled linear classifier but 0.444 for an ID-fitted distance score. ToxicChat shows the same contrast. Feature-selection and backbone controls retain the main reversal pattern. Frozen Civil Comments and TweetEval irony tests also reverse (0.467 and 0.435), extending the finding beyond toxicity. In these contrasts, anomalous groups have farther centers but tighter spread. A labeled, fixed-center feature-space intervention changes rankings: equalizing spread helps some t
    
[^162]: 谁来守护基准测试？LLM智能体基准测试的自动化审计

    Who Guards the Benchmarks? Automated Auditing of LLM Agent Benchmarks

    [https://arxiv.org/abs/2604.24955](https://arxiv.org/abs/2604.24955)

    提出BenchGuard——首个利用前沿大语言模型对基于执行的LLM智能体基准测试进行跨工件联合审计的框架，能够自动发现基准测试本身存在的缺陷（如损坏的任务规范和僵化的评估脚本）。

    

    随着基准测试日益复杂，许多表面上的智能体失败实际上根本不是智能体本身的失败——而是基准测试本身的失败：损坏的任务规范、隐含的假设，以及惩罚有效替代方法的僵化评估脚本。我们提出将前沿大语言模型（LLM）用作评估基础设施的系统性审计员，并通过BenchGuard实现这一愿景——这是首个专为基于执行的智能体基准测试的跨工件联合审计而设计的框架。BenchGuard通过结构化LLM协议对所有基准测试工件进行交叉验证，并可选择将智能体的解决方案或执行轨迹作为额外的诊断证据纳入审计。在两个知名科学基准测试上的部署结果表明，BenchGuard在ScienceAgentBench中发现了12个经作者确认的问题——包括导致任务无法解决的致命错误——并且在BIXBench Verified-50子集上与专家识别问题的匹配率恰好达到83.3%。

    arXiv:2604.24955v2 Announce Type: replace-cross  Abstract: As benchmarks grow in complexity, many apparent agent failures are not failures of the agent at all---they are failures of the benchmark itself: broken specifications, implicit assumptions, and rigid evaluation scripts that penalize valid alternative approaches. We propose employing frontier LLMs as systematic auditors of evaluation infrastructure, and realize this vision through BenchGuard, the first framework explicitly designed for joint cross-artifact auditing of execution-based agent benchmarks. BenchGuard cross-verifies all benchmark artifacts via structured LLM protocols, optionally incorporating agent solutions or execution traces as additional diagnostic evidence. Deployed on two prominent scientific benchmarks, BenchGuard identified 12 author-confirmed issues in ScienceAgentBench---including fatal errors rendering tasks unsolvable---and exactly matched 83.3% of expert-identified issues on the BIXBench Verified-50 subs
    
[^163]: AI智能体如何花你的钱？分析与预测智能体编程任务中的Token消耗

    How Do AI Agents Spend Your Money? Analyzing and Predicting Token Consumption in Agentic Coding Tasks

    [https://arxiv.org/abs/2604.22750](https://arxiv.org/abs/2604.22750)

    本文首次系统研究了智能体编程任务中的token消耗模式，发现智能体任务消耗的token比代码推理和对话任务高出1000倍且以输入token为主要成本来源、使用量波动极大，并进一步评估了大模型在任务执行前预测自身token成本的能力。

    

    AI智能体在复杂人类工作流程中的广泛部署正在推动LLM token消耗的快速增长。当智能体被部署在需要大量token的任务上时，自然会引出三个问题：（1）AI智能体把token花在了哪里？（2）哪些模型更具token效率？（3）智能体能否在任务执行前预测自己的token用量？本文首次对智能体编程任务中的token消耗模式进行了系统性研究。我们分析了八个前沿LLM在SWE-bench Verified上的运行轨迹，并评估了各模型在任务执行前预测自身token成本的能力。我们发现：（1）智能体任务的token消耗格外昂贵，比代码推理和代码对话任务高出1000倍，且整体成本主要由输入token而非输出token驱动；（2）token使用量高度可变且本质上具有随机性：同一任务的多次运行在总token消耗上可相差高达30倍。

    arXiv:2604.22750v3 Announce Type: replace-cross  Abstract: The wide adoption of AI agents in complex human workflows is driving rapid growth in LLM token consumption. When agents are deployed on tasks that require a significant amount of tokens, three questions naturally arise: (1) Where do AI agents spend the tokens? (2) Which models are more token-efficient? and (3) Can agents predict their token usage before task execution? In this paper, we present the first systematic study of token consumption patterns in agentic coding tasks. We analyze trajectories from eight frontier LLMs on SWE-bench Verified and evaluate models' ability to predict their own token costs before task execution. We find that: (1) agentic tasks are uniquely expensive, consuming 1000x more tokens than code reasoning and code chat, with input tokens rather than output tokens driving the overall cost; (2) token usage is highly variable and inherently stochastic: runs on the same task can differ by up to 30x in total
    
[^164]: 秩湍流Delta：文体计量学Delta度量的可解释方法

    Rank-Turbulence Delta and Interpretable Approaches to Stylometric Delta Metrics

    [https://arxiv.org/abs/2604.19499](https://arxiv.org/abs/2604.19499)

    本文提出秩湍流Delta和Jensen-Shannon Delta两种新的作者归属度量方法，通过将词频向量重构为概率分布并进行词元级分解，使Burrows经典Delta的距离结果可数值解释，并在英、德、法、俄四种语言的文学语料库上验证了方法的有效性。

    

    本文介绍了两种新的作者归属度量方法——秩湍流Delta和Jensen-Shannon Delta——它们通过应用专为概率分布设计的距离函数，推广了Burrows的经典Delta方法。我们首先阐述了这些度量的理论基础，对比了词频向量的中心化与非中心化z分数标准化处理，并将非中心化向量重新表述为概率分布。基于这一表示，我们开发了一种词元级分解方法，使每个Delta距离在数值上均可解释，从而便于细读文本和验证结果。这些方法的有效性在英语、德语、法语和俄语四种文学语料库上进行了评估。其中英语、德语和法语数据集编译自古腾堡计划，而俄语基准数据集为SOCIOLIT语料库，包含89位作者从18世纪到21世纪的639部作品。

    arXiv:2604.19499v5 Announce Type: replace  Abstract: This article introduces two new measures for authorship attribution - Rank-Turbulence Delta and Jensen-Shannon Delta - which generalise Burrows's classical Delta by applying distance functions designed for probabilistic distributions. We first set out the theoretical basis of the measures, contrasting centred and uncentred z-scoring of word-frequency vectors and re-casting the uncentred vectors as probability distributions. Building on this representation, we develop a token-level decomposition that renders every Delta distance numerically interpretable, thereby facilitating close reading and the validation of results. The effectiveness of the methods is assessed on four literary corpora in English, German, French and Russian. The English, German and French datasets are compiled from Project Gutenberg, whereas the Russian benchmark is the SOCIOLIT corpus containing 639 works by 89 authors spanning the eighteenth to the twenty-first c
    
[^165]: 大语言模型表示中的反问句：一项线性探针研究

    Rhetorical Questions in LLM Representations: A Linear Probing Study

    [https://arxiv.org/abs/2604.14128](https://arxiv.org/abs/2604.14128)

    该研究通过线性探针发现大语言模型在表示空间中能够早期且稳定地编码反问句信号，其跨数据集可迁移性虽然存在，但并不意味着模型内部存在统一的共享表示。

    

    反问句的提出并非为了获取信息，而是为了说服他人或表明立场。然而大型语言模型如何在内部表示这类问句仍不清楚。我们使用线性探针在两个具有不同话语语境的社交媒体数据集上分析了LLM表示中的反问句，发现反问信号在早期就已显现，且最后 token 表示能最稳定地捕获这一信号。反问句在数据集内部与寻求信息的问题线性可分，在跨数据集迁移场景下仍可被检测到，AUROC 约达到 0.7-0.8。然而，我们证明这种可迁移性并不简单地意味着存在共享表示。在不同数据集上训练的探针应用于同一目标语料库时会产生不同的排名，排名靠前的实例之间的重叠度往往低于 0.2。定性分析表明，这些分歧对应于不同的修辞现象……

    arXiv:2604.14128v3 Announce Type: replace-cross  Abstract: Rhetorical questions are asked not to seek information but to persuade or signal stance. How large language models internally represent them remains unclear. We analyze rhetorical questions in LLM representations using linear probes on two social-media datasets with different discourse contexts, and find that rhetorical signals emerge early and are most stably captured by last-token representations. Rhetorical questions are linearly separable from information-seeking questions within datasets, and remain detectable under cross-dataset transfer, reaching AUROC around 0.7-0.8. However, we demonstrate that transferability does not simply imply a shared representation. Probes trained on different datasets produce different rankings when applied to the same target corpus, with overlap among the top-ranked instances often below 0.2. Qualitative analysis shows that these divergences correspond to distinct rhetorical phenomena: some pr
    
[^166]: 一图胜千言吗？基于视觉证据必要性的自适应多模态事实核查

    Is a Picture Worth a Thousand Words? Adaptive Multimodal Fact-Checking with Visual Evidence Necessity

    [https://arxiv.org/abs/2604.04692](https://arxiv.org/abs/2604.04692)

    该论文挑战了“视觉证据总能提升事实核查准确性”的普遍假设，提出通过两个协同的视觉-语言模型自适应判断是否需要视觉证据的模块化框架AMuFC，在多个数据集上实现了更有效的事实核查。

    

    自动化事实核查是支持负责任信息生态系统的一项关键任务。尽管近期研究已经从纯文本事实核查发展到多模态事实核查，但一个普遍的假设是：引入视觉证据总能普遍提升核查准确性。在本研究中，我们挑战了这一假设，并证明不加区分地使用视觉证据反而可能降低准确性。基于这一发现，我们提出了AMuFC，一个模块化事实核查框架，它采用两个角色不同、相互协作的视觉-语言模型，实现视觉证据的自适应使用。在三个数据集上的实验结果（包括本研究提出的WebFC数据集）证明了在事实核查中自适应使用视觉证据的有效性。

    arXiv:2604.04692v3 Announce Type: replace-cross  Abstract: Automated fact-checking is a crucial task that supports a responsible information ecosystem. While recent research has progressed from text-only to multimodal fact-checking, a prevailing assumption is that incorporating visual evidence universally improves verification accuracy. In this work, we challenge this assumption and show that the indiscriminate use of visual evidence can reduce accuracy. Building on this finding, we propose AMuFC, a modular fact-checking framework that employs two collaborative vision-language models with distinct roles to enable the adaptive use of visual evidence. Experimental results on three datasets, including WebFC, introduced in this study, demonstrate the effectiveness of adaptive visual evidence use in fact-checking.
    
[^167]: 众多偏好，少量策略：面向多目标大语言模型对齐的紧凑策略组合

    Many Preferences, Few Policies: Compact Portfolios for Multi-Objective LLM Alignment

    [https://arxiv.org/abs/2604.04144](https://arxiv.org/abs/2604.04144)

    该论文提出 PALM 算法，通过结构化权重向量网格、惰性搜索与剪枝构建一个小型 LLM 策略组合，可证明地覆盖所有奖励权重下的近优对齐策略，以低成本实现多目标 LLM 对齐的个性化与部署。

    

    对齐大语言模型（LLM）需要在有用性、无害性和简洁性等相互竞争的目标之间进行权衡。合适的平衡因用户和应用而异，然而针对不同的奖励权重去训练、评估和部署大量策略的成本十分高昂。我们研究如何识别一个小型的 LLM 组合，使其在所有奖励权重设置下都能保持接近最优的性能。我们提出了 PALM（对齐 LLM 组合，Portfolio of Aligned LLMs）算法，该算法结合了结构化的权重向量网格、仅在需要之处才优化策略的惰性搜索以及剪枝技术。在给定目标近似容差的情况下，PALM 返回的组合可证明对每个权重向量都包含一个接近最优的策略，并对组合大小给出显式上界。这样的组合可以支持可扩展的个性化、模型开发过程中的奖励权重探索，以及紧凑的解码时配置。实验表明……（摘要在此处截断）

    arXiv:2604.04144v3 Announce Type: replace-cross  Abstract: Aligning large language models (LLMs) requires balancing competing objectives such as helpfulness, harmlessness, and conciseness. The appropriate balance varies across users and applications, yet training, evaluating, and deploying many policies across different reward weights is costly. We study how to identify a small portfolio of LLMs that preserves near-optimal performance across all reward weightings. We propose PALM (Portfolio of Aligned LLMs), an algorithm that combines a structured grid of weight vectors, a lazy search that optimizes policies only where needed, and pruning. Given target approximation tolerances, PALM returns a portfolio that provably contains a near-optimal policy for every weight vector, with an explicit upper bound on portfolio size. Such portfolios can support scalable personalization, reward-weight exploration during model development, and compact decoding-time configurations. Experiments show that 
    
[^168]: GISTBench：通过基于证据的兴趣验证评估大语言模型的用户理解能力

    GISTBench: Evaluating LLM User Understanding via Evidence-Based Interest Verification

    [https://arxiv.org/abs/2603.29112](https://arxiv.org/abs/2603.29112)

    该论文提出GISTBench基准，通过兴趣扎根度（IG）和兴趣特异性（IS）两个新指标，评估大语言模型从推荐系统交互历史中提取和验证用户兴趣的能力，突破了传统推荐系统基准仅关注物品预测准确率的局限。

    

    我们推出了GISTBench，这是一个用于评估大语言模型（LLM）从推荐系统交互历史中理解用户能力的基准测试。与传统的专注于物品预测准确率的推荐系统（RecSys）基准不同，我们的基准评估的是LLM从用户参与行为数据中提取和验证用户兴趣的能力。我们提出了两个新颖的指标族：兴趣扎根度，将其分解为精确率和召回率两个组成部分，以分别惩罚幻觉产生的兴趣类别并奖励兴趣覆盖范围；以及兴趣特异性（IS），用于评估经验证的LLM预测用户画像的独特性。我们发布了一个基于全球短视频平台真实用户交互构建的合成数据集。我们的数据集包含隐式和显式参与信号以及丰富的文本描述。我们通过用户调研验证了数据集的保真度，并评估了八个开源权重的LLM模型。

    arXiv:2603.29112v2 Announce Type: replace  Abstract: We introduce GISTBench, a benchmark for evaluating Large Language Models' (LLMs) ability to understand users from their interaction histories in recommendation systems. Unlike traditional RecSys benchmarks that focus on item prediction accuracy, our benchmark evaluates how well LLMs can extract and verify user interests from engagement data. We propose two novel metric families: Interest Groundedness (IG), decomposed into precision and recall components to separately penalize hallucinated interest categories and reward coverage, and Interest Specificity (IS), which assesses the distinctiveness of verified LLM-predicted user profiles. We release a synthetic dataset constructed on real user interactions on a global short-form video platform. Our dataset contains both implicit and explicit engagement signals and rich textual descriptions. We validate our dataset fidelity against user surveys, and evaluate eight open-weight LLMs spanning
    
[^169]: 话到嘴边：为什么大语言模型会幻觉出它们本可解码出的答案

    On the Tip of the Tongue: Why LLMs Hallucinate Answers They Can Decode

    [https://arxiv.org/abs/2603.13911](https://arxiv.org/abs/2603.13911)

    该论文提出在首个答案标记处区分“读取”与“写出”的新框架，揭示大语言模型产生幻觉的关键原因并非正确答案无法从中间状态解码，而是最终读出时的“选择边际”不足，使更强的竞争标记压制了正确答案。

    

    即使在正确答案能够从其中间状态解码出来的情况下，语言模型也可能给出错误的答案。为了研究这种可解码性与选择之间的差距，我们在第一个答案标记处区分了“读取”与“写出”。“读取”问的是：在相同关系诱饵控制的条件下，正确标记能否从中间残差状态中被解码出来；“写出”问的是：最终的读出是否将该标记排在所有内容标记的首位。在三种不同的读取器下，并采用随机标签控制实验，仍有相当大比例的失败案例保持“可读取”状态，同时另一个内容标记被选中。我们通过最终读出处的选择边际来解释这一现象：选择边际是答案logit与其最强竞争者logit之间的差值，即答案支持度减去竞争者支持度，并且可以进一步分解为与标记频率相关的上下文平均基线和一个项目特定的部分。将答案支持度设置为……（原文摘要在此处截断）

    arXiv:2603.13911v2 Announce Type: replace  Abstract: A language model can give the wrong answer even when the correct answer is decodable from its intermediate states. To study this gap between decodability and selection, we distinguish \textit{read} from \textit{write} at the first answer token. Read asks whether the gold token can be decoded from intermediate residual states under same-relation decoy controls. Write asks whether the final readout ranks that token first among content tokens. Under three different readers, with a randomized-label control, a substantial fraction of failures remain readable while another content token is selected. We explain this through the selection margin at the final readout, the difference between the answer logit and the logit of its strongest alternative, which is answer support minus alternative support, and can also be split into a context-averaged baseline linked to token frequency and an item-specific term. Setting the answer support to the le
    
[^170]: 我们能信任忆阻器上的大语言模型吗？深入探究非理想条件下的推理能力

    Can We Trust LLMs on Memristors? Diving into Reasoning Ability under Non-Ideality

    [https://arxiv.org/abs/2603.13725](https://arxiv.org/abs/2603.13725)

    该论文系统研究了忆阻器存内计算架构中的非理想性对大语言模型推理能力的影响，并总结出三种免训练策略（浅层冗余、思考模式与上下文学习）在不同噪声水平下的适用准则。

    

    基于忆阻器的模拟存内计算（CIM）架构凭借卓越的能效和计算密度，为大语言模型（LLM）的高效部署提供了一种极具前景的硬件基底。然而，这类架构会受到忆阻器固有非理想性所导致的精度问题的影响。本文首先全面研究了这些典型非理想性对大语言模型推理能力的影响，实证结果表明推理能力会显著下降，且在不同基准测试上的下降程度各不相同。随后，我们系统评估了三种免训练策略，包括思考模式、上下文学习和模块冗余，并据此总结出有价值的指导原则：浅层冗余对提升鲁棒性尤为有效；思考模式在低噪声水平下表现更好，但在较高噪声下性能会退化；上下文学习则可以减少……

    arXiv:2603.13725v2 Announce Type: replace  Abstract: Memristor-based analog compute-in-memory (CIM) architectures provide a promising substrate for the efficient deployment of Large Language Models (LLMs), owing to superior energy efficiency and computational density. However, these architectures suffer from precision issues caused by intrinsic non-idealities of memristors. In this paper, we first conduct a comprehensive investigation into the impact of such typical non-idealities on LLM reasoning. Empirical results indicate that reasoning capability decreases significantly but varies for distinct benchmarks. Subsequently, we systematically appraise three training-free strategies, including thinking mode, in-context learning, and module redundancy. We thus summarize valuable guidelines, i.e., shallow layer redundancy is particularly effective for improving robustness, thinking mode performs better under low noise levels but degrades at higher noise, and in-context learning reduces outp
    
[^171]: SiDiaC-v.2.0：僧伽罗语历时语料库2.0版

    SiDiaC-v.2.0: Sinhala Diachronic Corpus Version 2.0

    [https://arxiv.org/abs/2603.10861](https://arxiv.org/abs/2603.10861)

    本文构建了迄今最大的僧伽罗语历时语料库SiDiaC-v.2.0，包含185部文学作品共22.9万词，时间跨度从公元5世纪至20世纪，并提供了按写作日期标注的子集，为僧伽罗语的历史语言学研究提供了重要资源。

    

    SiDiaC-v.2.0是迄今为止最大的综合性僧伽罗语历时语料库，按出版日期涵盖公元1800年至1955年，按写作日期涵盖公元5世纪至20世纪的历史时期。该语料库包含来自185部文学作品的22.9万词，这些作品经过了严格的筛选、预处理和版权合规检查，随后进行了大量的后处理。此外，其中59份文档（共计6.5万词）的子集根据其写作日期进行了标注。来自斯里兰卡国家图书馆的文本从SiDiaC-v.1.0的未过滤列表中选取，并使用Google Document AI OCR进行数字化，随后通过后处理纠正格式问题、处理语码混合现象、添加特殊标记并修复格式错误的标记。SiDiaC-v.2.0的构建参考了FarPaHC、SiDiaC-v.1.0和CCOHA等其他语料库的实践经验。

    arXiv:2603.10861v2 Announce Type: replace  Abstract: SiDiaC-v.2.0 is the largest comprehensive Sinhala Diachronic Corpus to date, covering a period from 1800 CE to 1955 CE in terms of publication dates, and a historical span from the 5th to the 20th century CE in terms of written dates. The corpus consists of 229k words across 185 literary works that underwent thorough filtering, preprocessing, and copyright compliance checks, followed by extensive post-processing. Additionally, a subset of 59 documents totalling 65k words was annotated based on their written dates. Texts from the National Library of Sri Lanka were selected from the SiDiaC-v.1.0 non-filtered list, which was digitised using Google Document AI OCR. This was followed by post-processing to correct formatting issues, address code-mixing, include special tokens, and fix malformed tokens. The construction of SiDiaC-v.2.0 was informed by practices from other corpora, such as FarPaHC, SiDiaC-v.1.0, and CCOHA. This was particula
    
[^172]: Code2Math：你的代码智能体能通过探索演化数学问题吗？

    Code2Math: Can Your Code Agent Evolve Math Problems Through Exploration?

    [https://arxiv.org/abs/2603.03202](https://arxiv.org/abs/2603.03202)

    提出多智能体框架Code2Math，利用代码智能体通过探索将现有数学问题自主演化为结构不同、更具挑战性且可解的新问题，以缓解高质量数学问题稀缺的瓶颈。

    

    随着大语言模型（LLM）的数学能力不断向国际数学奥林匹克（IMO）和研究级水平迈进，具有挑战性的高质量问题的稀缺已成为LLM训练、评估和自我进化的重要瓶颈。与此同时，近期的代码智能体在智能体编程和推理方面展现出精湛的技能，这表明代码执行可以作为一个可扩展的数学实验环境。本文研究了代码智能体将现有数学问题自主演化为更复杂变体的潜力。我们引入了一个多智能体框架，旨在执行问题演化的同时验证所生成问题的可解性及其难度的提升。我们的实验表明，在给定足够的测试时探索的情况下，代码智能体能够合成新的可解问题，这些问题在结构上与原始问题不同且更具挑战性。

    arXiv:2603.03202v5 Announce Type: replace  Abstract: As large language models (LLMs) advance their mathematical capabilities toward the IMO and research level, the scarcity of challenging, high-quality problems has become a significant bottleneck for training, evaluation and self-evolution of LLMs. Simultaneously, recent code agents have demonstrated sophisticated skills in agentic coding and reasoning, suggesting that code execution can serve as a scalable environment for mathematical experimentation. In this paper, we investigate the potential of code agents to autonomously evolve existing math problems into more complex variations. We introduce a multi-agent framework designed to perform problem evolution while validating the solvability and increased difficulty of the generated problems. Our experiments demonstrate that, given sufficient test-time exploration, code agents can synthesize new, solvable problems that are structurally distinct from and more challenging than the origina
    
[^173]: 基于评论蒸馏表示的感官感知序列推荐

    Sensory-Aware Sequential Recommendation via Review-Distilled Representations

    [https://arxiv.org/abs/2603.02709](https://arxiv.org/abs/2603.02709)

    该论文提出ASER离线流水线，通过微调大语言模型从评论中提取有据可查的感官属性并蒸馏为冻结的五维感官库，再以轻量级关系度量增强序列推荐，同时保持预训练主干不变。

    

    序列推荐器从物品标识符中学习行为模式，然而用户在评论中描述的体验性属性，例如产品的外观、触感、气味、口味或声音，很少以可控、可审计的形式融入物品表示中。我们提出了ASER（基于属性的感官增强表示），这是一个离线流水线，它通过微调大语言模型从评论文本中提取有证据支撑的感官属性-值记录，例如“颜色：哑光黑”或“气味：香草”，并将其蒸馏到一个紧凑的学生编码器中，为每个物品目录生成一个冻结的五维感官库。在推荐阶段，预训练主干保持冻结：在感官库之上学习用户历史与每个候选物品之间的轻量级关系度量，其校正在通过验证选择的幅度界限内应用。在五个亚马逊领域和四个主干模型上进行训练……（原文摘要在此处截断）

    arXiv:2603.02709v4 Announce Type: replace-cross  Abstract: Sequential recommenders learn behavioral patterns from item identifiers, while the experiential properties that users describe in reviews, such as how products look, feel, smell, taste, or sound, rarely enter item representations in a controlled, auditable form.   We present ASER (Attribute-based Sensory-Enhanced Representation), an offline pipeline that fine-tunes a large language model to extract evidence-grounded sensory attribute-value records, such as color: matte black or scent: vanilla, from review text and distills them into a compact student encoder that produces a frozen five-facet sensory bank for each item catalog.   At recommendation time the pretrained backbone stays frozen: a lightweight relational metric between the user history and each candidate is learned over the bank, and its correction is applied within a validation-selected magnitude bound.   Across five Amazon domains and four backbones, trained within a
    
[^174]: RT-SFT：通过往返翻译实现从非平行语料库的文本风格迁移

    RT-SFT: Text Style Transfer from Non-Parallel Corpora by Roundtrip Translation

    [https://arxiv.org/abs/2602.15013](https://arxiv.org/abs/2602.15013)

    本文提出利用在通用语料上训练的神经机器翻译系统，通过枢轴语言进行往返翻译来剥离风格信号，将归一化从推理时的小型补丁转变为大规模数据生成工具，从而实现从非平行语料库的文本风格迁移。

    

    文本风格迁移（TST）本质上是一项监督任务——在保留句子含义的同时，以目标风格重写句子——然而监督所需的平行语料库仅存在于少数几种风格领域中。一种常见的解决方法是先将输入*归一化*为与风格无关的中间形式，然后再将其*风格化*为目标风格，但归一化器通常是仅在测试时应用的轻量级、任务特定的改写器，其配套的风格化器规模也相应较小。我们观察到，一种大规模存在的风格剥离归一化器其实早已存在：在海量通用领域句子对上训练的神经机器翻译系统在保留内容的同时会向通用化表达回归，因此通过枢轴语言进行往返翻译可以在无需任何任务特定训练的情况下剥离风格信号。这将归一化从推理阶段的补丁手段转变为一种数据生成工具。对单语的目标风格语料库进行往返翻译……（原文摘要不完整）

    arXiv:2602.15013v2 Announce Type: replace  Abstract: Text style transfer (TST) is naturally a supervised task - rewrite a sentence in a target style while preserving its meaning - yet the parallel corpora that supervision requires exist for only a handful of style domains. A common workaround is to *normalize* an input into a style-agnostic intermediate and then *stylize* it into the target style, but the normalizer is typically a lightweight, task-specific paraphraser applied only at test time, feeding a correspondingly small stylizer. We observe that a style-stripping normalizer already exists at scale: neural MT systems trained on hundreds of millions of general-domain sentence pairs preserve content while regressing toward generic phrasing, so roundtrip translation through a pivot language strips stylistic signal without any task-specific training. This turns normalization from an inference-time patch into a data-generation tool. Roundtrip-translating a monolingual in-style corpus 
    
[^175]: AstroAgentBench：在太空任务规划任务上评估智能体规划能力

    AstroAgentBench: Evaluating Agentic Planning on Space Mission Planning Tasks

    [https://arxiv.org/abs/2601.11354](https://arxiv.org/abs/2601.11354)

    本文提出AstroAgentBench——一个涵盖调度、观测规划、星座设计和中继支持等七大任务族的可执行太空任务规划基准，通过外部验证器评估智能体生成的规划产物，发现最强LLM智能体系统在部分任务上可接近或超越求解器参考水平，而较弱系统则难以产出高价值的有效规划。

    

    arXiv:2601.11354v2 公告类型：替换。摘要：近期的“大语言模型用于航天”（LLM-for-Space）系统涉及任务规划、调度、运营支持、模拟器控制和自主性等方面，但它们的评估采用了不同的任务契约、控制设置、模拟器和成功标准。我们提出了AstroAgentBench，这是一个包含七大任务族的可执行太空任务规划基准，涵盖调度、观测规划、星座设计和中继支持等领域。对于每个案例，智能体需要提交一个规划产物，该产物由外部验证器对其模式结构、时序、几何关系、资源和任务价值进行检查。结果报告有效性和归一化分数，并与任务特定的求解器参考结果进行比较。在五个LLM智能体系统和35个保留测试案例上的结果显示，最强的系统在若干任务族上接近或超过求解器参考分数，而较弱的系统往往无法产出高价值的有效规划，即便是强大的系统在几何、产品级或设计繁重的任务上也会出现质量下降。

    arXiv:2601.11354v2 Announce Type: replace  Abstract: Recent LLM-for-Space systems address mission planning, scheduling, operations support, simulator control, and autonomy, but their evaluations use different task contracts, control settings, simulators, and success criteria. We introduce AstroAgentBench, a seven-family benchmark for executable space mission planning in the domains of scheduling, observation planning, constellation design, and relay support. For each case, an agent submits a planning artifact that is checked by an external verifier for schema, timing, geometry, resources, and mission value. Results report validity and normalized scores, with comparisons to task-specific solver references. Across five LLM agent systems and 35 held-out cases, the strongest systems approach or exceed solver-reference scores on several families, while weaker systems often fail to produce high-value valid plans and even strong systems lose quality on geometric, product-level, or design-heav
    
[^176]: 道德是情境化的：利用概率聚类与大语言模型从人类数据中学习可解释的道德情境

    Morality is Contextual: Learning Interpretable Moral Contexts from Human Data with Probabilistic Clustering and Large Language Models

    [https://arxiv.org/abs/2512.21439](https://arxiv.org/abs/2512.21439)

    提出了COMETH框架，将概率情境学习与大语言模型语义抽象及人类道德判断数据相结合，从数据中学习可解释的道德情境，证明道德评价是高度情境化的。

    

    当前AI对齐研究中的一个关键问题是如何让AI算法学习道德价值观。由于人类道德高度依赖情境，对行为的评判不仅取决于其结果，还取决于行为发生的情境。我们提出了COMETH（基于文本人类输入的道德评估情境组织），这是一个将概率情境学习器与基于大语言模型的语义抽象及人类道德评估相结合的框架，用于建模情境如何塑造模糊行为的可接受性。我们构建了一个基于实证的数据集，包含与三条道德规则（违反“不可杀人”、“不可欺骗”和“不可违法”）相关的六种核心行为共300个场景，并收集了101名参与者的三元判断（谴责/中立/支持）。预处理流程通过大语言模型过滤器与结合K-means聚类的MiniLM嵌入对行为进行标准化，产生稳健且可复现的核心行为聚类。

    arXiv:2512.21439v2 Announce Type: replace-cross  Abstract: A key question in current AI alignment research is how to make AI algorithms learn moral values. Because human morality is highly context-dependent, actions are judged not only by their outcomes but by the context in which they occur. We present COMETH (Contextual Organization of Moral Evaluation from Textual Human inputs), a framework that integrates a probabilistic context learner with LLM-based semantic abstraction and human moral evaluations to model how context shapes the acceptability of ambiguous actions. We curate an empirically grounded dataset of 300 scenarios across six core actions relative to three moral rules (violating "Do not kill", "Do not deceive", and "Do not break the law") and collect ternary judgments (Blame/Neutral/Support) from N=101 participants. A preprocessing pipeline standardizes actions via an LLM filter and MiniLM embeddings with K-means, producing robust, reproducible core-action clusters. COMETH
    
[^177]: 用于孟加拉语新闻标题分类与情感分析同步进行的统一BERT-CNN-BiLSTM框架

    A Unified BERT-CNN-BiLSTM Framework for Simultaneous Headline Classification and Sentiment Analysis of Bangla News

    [https://arxiv.org/abs/2511.18618](https://arxiv.org/abs/2511.18618)

    本文提出了一个统一的BERT-CNN-BiLSTM混合迁移学习框架，首次实现了孟加拉语新闻标题分类与情感分析的同步处理。

    

    在我们的日常生活中，报纸是一种重要的信息来源，影响着公众对当下议题的讨论方式。然而，如何有效地浏览来自不同报纸和在线新闻门户的海量新闻内容是一项挑战。结合情感分析的报纸标题能够告诉我们新闻的内容（如政治、体育）以及新闻带给我们的感受（积极、消极、中性），这有助于我们快速理解新闻的情感基调。本研究提出了一种最先进的方法，将孟加拉语新闻标题分类与情感分析相结合，应用了自然语言处理（NLP）技术，特别是混合迁移学习模型BERT-CNN-BiLSTM。我们探索了一个名为BAN-ABSA的数据集，包含9014条新闻标题，这是首次在孟加拉语报纸中同时进行标题分类和情感分类的实验。

    arXiv:2511.18618v2 Announce Type: replace-cross  Abstract: In our daily lives, newspapers are an essential information source that impacts how the public talks about present-day issues. However, effectively navigating the vast amount of news content from different newspapers and online news portals can be challenging. Newspaper headlines with sentiment analysis tell us what the news is about (e.g., politics, sports) and how the news makes us feel (positive, negative, neutral). This helps us quickly understand the emotional tone of the news. This research presents a state-of-the-art approach to Bangla news headline classification combined with sentiment analysis applying Natural Language Processing (NLP) techniques, particularly the hybrid transfer learning model BERT-CNN-BiLSTM. We have explored a dataset called BAN-ABSA of 9014 news headlines, which is the first time that has been experimented with simultaneously in the headline and sentiment categorization in Bengali newspapers. Over
    
[^178]: 从最后一层Logits到逻辑：赋予大语言模型逻辑一致的结构化知识推理能力

    Last Layer Logits to Logic: Empowering LLMs with Logic-Consistent Structured Knowledge Reasoning

    [https://arxiv.org/abs/2511.07910](https://arxiv.org/abs/2511.07910)

    该论文针对大语言模型在结构化知识推理中的“逻辑漂移”问题，提出从模型最后一层的Logits入手进行干预，而非仅依赖提示层面的工作流引导，从而实现逻辑一致的结构化知识推理。

    

    大语言模型（LLMs）通过在海量非结构化文本上进行预训练，在自然语言推理任务中取得了卓越的表现，使其能够理解自然语言中的逻辑并生成逻辑一致的回复。然而，非结构化知识与结构化知识之间的表示差异，使得大语言模型天然难以保持逻辑一致性，导致在知识图谱问答（KGQA）等结构化知识推理任务中出现“逻辑漂移”挑战。现有方法通过在提示词中嵌入复杂的工作流来引导大语言模型推理，以弥补这一局限。然而，这些方法仅提供输入层面的引导，无法从根本上解决大语言模型输出中的“逻辑漂移”问题；此外，其僵化的推理工作流也难以适应不同的任务和知识图谱。为了增强大语言模型在结构化知识推理中的逻辑一致性……（原文摘要在此处被截断）

    arXiv:2511.07910v3 Announce Type: replace  Abstract: Large Language Models (LLMs) achieve excellent performance in natural language reasoning tasks through pre-training on vast unstructured text, enabling them to understand the logic in natural language and generate logic-consistent responses. However, the representational differences between unstructured and structured knowledge make LLMs inherently struggle to maintain logic consistency, leading to \textit{Logic Drift} challenges in structured knowledge reasoning tasks such as Knowledge Graph Question Answering (KGQA). Existing methods address this limitation by designing complex workflows embedded in prompts to guide LLM reasoning. Nevertheless, these approaches only provide input-level guidance and fail to fundamentally address the \textit{Logic Drift} in LLM outputs. Additionally, their inflexible reasoning workflows cannot adapt to different tasks and knowledge graphs. To enhance LLMs' logic consistency in structured knowledge re
    
[^179]: WAON：用于对比视觉-语言模型文化适配的大规模日文图文数据集

    WAON: A Large-Scale Japanese Image-Text Dataset for Cultural Adaptation in Contrastive Vision-Language Models

    [https://arxiv.org/abs/2510.22276](https://arxiv.org/abs/2510.22276)

    本文发布了目前最大的公开原生日文图文数据集WAON（约1.55亿样本）及日本文化基准WAON-Bench（374个类别），实验证明使用本地来源数据进行微调能将特定文化理解能力提升至超越仅靠全球预训练的水平。

    

    对比视觉-语言模型通过大规模预训练取得了显著进展。近期研究表明，移除仅限英文的描述过滤器并在全球数据上进行预训练，对提升多文化性能十分有效。我们研究了这种全球预训练是否足以实现特定文化的理解，或者使用本地来源数据进行进一步适配，能否使性能超越仅靠全球预训练所能达到的水平。为了开展这项研究，我们提出了WAON——目前最大的公开可用的原生日文图文数据集，该数据集由Common Crawl中的原生日文网络内容构建，包含约1.55亿个样本。我们还引入了WAON-Bench，一个手动整理的、涵盖374个类别的日本文化基准。通过在多个日文图文数据集上进行对比微调实验，我们观察到在WAON上微调的模型始终取得

    arXiv:2510.22276v4 Announce Type: replace-cross  Abstract: Contrastive vision-language models have achieved remarkable progress through large-scale pretraining. Recent work has shown that removing English-only caption filters and pretraining on global data is effective for improving multicultural performance. We study whether such global pretraining is sufficient for culture-specific understanding, or whether further adaptation with natively sourced data can boost performance beyond what global pretraining alone achieves. To enable this investigation, we present WAON, the largest publicly available native Japanese image-text dataset constructed from native Japanese web content in Common Crawl, containing approximately 155 million examples. We also introduce WAON-Bench, a manually curated Japanese cultural benchmark spanning 374 classes. Through comparative fine-tuning experiments on multiple Japanese image-text datasets, we observe that models fine-tuned on WAON consistently achieve st
    
[^180]: Enrich-on-Graph：基于大语言模型增强的查询-图谱对齐复杂推理

    Enrich-on-Graph: Query-Graph Alignment for Complex Reasoning with LLM Enriching

    [https://arxiv.org/abs/2509.20810](https://arxiv.org/abs/2509.20810)

    提出Enrich-on-Graph（EoG）框架，利用大语言模型的先验知识增强知识图谱，弥合结构化图谱与非结构化查询之间的语义鸿沟，实现高效、低成本且可扩展的知识图谱问答复杂推理。

    

    大语言模型（LLM）在复杂任务中展现出强大的推理能力。然而，在知识图谱问答（KGQA）等知识密集型场景中，它们仍然面临幻觉和事实性错误的困扰。我们将这一问题归因于结构化知识图谱（KG）与非结构化查询之间的语义鸿沟，这种鸿沟源于二者在关注点和结构上的固有差异。现有方法通常采用资源密集、难以扩展的工作流在原始知识图谱上进行推理，却忽视了这一鸿沟。为应对这一挑战，我们提出了一个灵活的框架 Enrich-on-Graph（EoG），利用大语言模型的先验知识来增强知识图谱，从而弥合图谱与查询之间的语义鸿沟。EoG 能够从知识图谱中高效提取证据，实现精确且稳健的推理，同时保持较低的计算成本，并具备良好的可扩展性和对不同方法的适应性。此外，我们还提出了三种图谱质量评估（指标）……

    arXiv:2509.20810v2 Announce Type: replace  Abstract: Large Language Models (LLMs) exhibit strong reasoning capabilities in complex tasks. However, they still struggle with hallucinations and factual errors in knowledge-intensive scenarios like knowledge graph question answering (KGQA). We attribute this to the semantic gap between structured knowledge graphs (KGs) and unstructured queries, caused by inherent differences in their focuses and structures. Existing methods usually employ resource-intensive, non-scalable workflows reasoning on vanilla KGs, but overlook this gap. To address this challenge, we propose a flexible framework, Enrich-on-Graph (EoG), which leverages LLMs' prior knowledge to enrich KGs, bridge the semantic gap between graphs and queries. EoG enables efficient evidence extraction from KGs for precise and robust reasoning, while ensuring low computational costs, scalability, and adaptability across different methods. Furthermore, we propose three graph quality evalua
    
[^181]: Percept-V 挑战：多模态大语言模型能攻克简单感知问题吗？

    The Percept-V Challenge: Can Multimodal LLMs Crack Simple Perception Problems?

    [https://arxiv.org/abs/2508.21143](https://arxiv.org/abs/2508.21143)

    该论文提出了 Percept-V 数据集，包含 6000 张程序生成的无污染图像、分为 30 个基于 TVPS-4 框架的感知领域，用于系统评估多模态大语言模型在简单视觉感知任务上的能力。

    

    认知科学研究将视觉感知——即理解和解释视觉输入的能力——视为智能的早期发展标志之一。其 TVPS-4 框架将人类感知分类为并测试七种技能，例如视觉辨别和形状恒常性。多模态大语言模型（MLLMs）在基础感知方面能否与人类媲美？尽管许多基准测试评估了 MLLMs 在高级推理和知识技能上的表现，但专注于简单感知评估的研究仍然有限。为此，我们推出了 Percept-V，这是一个包含 6000 张由程序生成的无污染图像的数据集，共分为 30 个领域，每个领域测试一种或多种 TVPS-4 技能。由于我们的重点是感知，因此我们将这些领域设计得相当简单，解决问题所需的推理和知识极少。鉴于现代 MLLMs 能够解决复杂得多的任务，我们的先验预期是……（摘要在此处被截断）

    arXiv:2508.21143v4 Announce Type: replace  Abstract: Cognitive science research treats visual perception, the ability to understand and make sense of a visual input, as one of the early developmental signs of intelligence. Its TVPS-4 framework categorizes and tests human perception into seven skills such as visual discrimination, and form constancy. Do Multimodal Large Language Models (MLLMs) match up to humans in basic perception? Even though many benchmarks evaluate MLLMs on advanced reasoning and knowledge skills, there is limited research that focuses evaluation on simple perception. In response, we introduce Percept-V, a dataset containing 6000 program-generated uncontaminated images divided into 30 domains, where each domain tests one or more TVPS-4 skills. Our focus is on perception, so we make our domains quite simple and the reasoning and knowledge required for solving them are minimal. Since modern-day MLLMs can solve much more complex tasks, our a-priori expectation is that 
    
[^182]: BioMol-MQA：一个用于大语言模型对生物分子交互进行推理的多模态问答数据集

    BioMol-MQA: A Multi-Modal Question Answering Dataset For LLM Reasoning Over Bio-Molecular Interactions

    [https://arxiv.org/abs/2506.05766](https://arxiv.org/abs/2506.05766)

    该论文提出了BioMol-MQA——一个针对多重用药场景的新型多模态问答数据集，通过融合文本与分子结构的多模态知识图谱及具有挑战性的问题，用于评估大语言模型在多模态知识检索与推理方面的能力。

    

    检索增强生成（RAG）在改进大语言模型（LLM）方面展现出了强大的能力。然而，大多数现有的基于RAG的LLM专门用于检索单一模态的信息，主要是文本；而对于许多现实世界的问题（例如医疗保健），与查询相关的信息可能以多种模态呈现，如知识图谱、文本（临床笔记）以及复杂的分子结构。因此，能够检索相关的多模态领域特定信息，并对多样化的知识进行推理和综合以生成准确的回答是十分重要的。为了填补这一空白，我们提出了BioMol-MQA，这是一个关于多重用药（polypharmacy）的新型问答（QA）数据集，由两部分组成：（i）一个包含文本和分子结构、用于信息检索的多模态知识图谱（KG）；以及（ii）旨在测试LLM在多模态知识图谱上进行检索和推理以回答问题能力的具有挑战性的问题。

    arXiv:2506.05766v2 Announce Type: replace  Abstract: Retrieval augmented generation (RAG) has shown great power in improving Large Language Models (LLMs). However, most existing RAG-based LLMs are dedicated to retrieving single modality information, mainly text; while for many real-world problems, such as healthcare, information relevant to queries can manifest in various modalities such as knowledge graph, text (clinical notes), and complex molecular structure. Thus, being able to retrieve relevant multi-modality domain-specific information, and reason and synthesize diverse knowledge to generate an accurate response is important. To address the gap, we present BioMol-MQA, a new question-answering (QA) dataset on polypharmacy, which is composed of two parts (i) a multimodal knowledge graph (KG) with text and molecular structure for information retrieval; and (ii) challenging questions that designed to test LLM capabilities in retrieving and reasoning over multimodal KG to answer quest
    
[^183]: 评估大型语言模型的检索鲁棒性

    Evaluating the Retrieval Robustness of Large Language Models

    [https://arxiv.org/abs/2505.21870](https://arxiv.org/abs/2505.21870)

    该研究建立了一个包含1,891个样本的基准和三个鲁棒性指标，系统评估了11个大型语言模型在检索增强生成场景中的检索鲁棒性，重点考察RAG是否总是优于非RAG、更多检索文档是否总是有益以及文档顺序对结果的影响。

    

    检索增强生成（RAG）通常能提升大型语言模型（LLM）解决知识密集型任务的能力。但由于检索结果不完美以及模型利用检索内容的能力有限，RAG也可能导致性能下降。在这项工作中，我们评估了LLM在实际RAG设置下的鲁棒性（下文称为检索鲁棒性）。我们聚焦于三个研究问题：（1）RAG是否总是优于非RAG；（2）检索更多的文档是否总能带来更好的性能；（3）文档顺序是否会影响结果。为开展这项研究，我们建立了一个包含1,891个样本的基准测试集，涵盖三个任务类别中的五个数据集，每个样本均包含使用稀疏检索器和稠密检索器检索到的文档。我们引入了三个鲁棒性指标，分别对应上述三个研究问题。我们在11个LLM上进行的实验表明，模型总体上达到了较高的检索鲁棒性。

    arXiv:2505.21870v2 Announce Type: replace-cross  Abstract: Retrieval-augmented generation (RAG) generally enhances large language models' (LLMs) ability to solve knowledge-intensive tasks. But RAG could also lead to performance degradation due to imperfect retrieval and the model's limited ability to leverage retrieved content. In this work, we evaluate the robustness of LLMs in practical RAG setups (henceforth retrieval robustness). We focus on three research questions: (1) whether RAG is always better than non-RAG; (2) whether more retrieved documents always lead to better performance; and (3) whether document order impacts results. To facilitate this study, we establish a benchmark of 1,891 samples spanning five datasets across three task categories, each with documents retrieved using both sparse and dense retrievers. We introduce three robustness metrics, each corresponding to one research question. Our experiments across 11 LLMs show that models achieve generally high retrieval r
    
[^184]: 使用多语言深度学习实现开放网络的自动语域识别

    Automatic register identification for the open web using multilingual deep learning

    [https://arxiv.org/abs/2406.19892](https://arxiv.org/abs/2406.19892)

    本文构建了覆盖16种语言、25种语域的Multilingual CORE语料库，利用多标签深度学习实现开放网络语域自动识别，并发现性能瓶颈源于网络语域固有的模糊性而非模型局限。

    

    本文提出了用于识别网络语域的多语言深度学习模型——网络语域是指新闻报道、讨论论坛等文本类型，涵盖16种语言。我们推出了Multilingual CORE语料库，其中包含超过72,000篇文档，并采用旨在覆盖整个开放网络的25种语域的层级分类体系进行了标注。通过多标签分类，我们表现最佳的模型在所有语言上平均达到79%的F1分数，达到甚至超越了以往使用更简单分类方案的研究水平。这表明模型即使在多语言规模下采用复杂的语域分类体系也能有良好表现。然而，我们观察到所有模型和配置都存在一个一致的性能上限。当我们通过数据剪枝去除标签不确定的文档后，性能提升至超过90%的F1分数，这表明该上限源于网络语域固有的模糊性，而非模型本身的局限。

    arXiv:2406.19892v5 Announce Type: replace  Abstract: This article presents multilingual deep learning models for identifying web registers -- text varieties such as news reports and discussion forums -- across 16 languages. We introduce the Multilingual CORE corpora, which contain over 72,000 documents annotated with a hierarchical taxonomy of 25 registers designed to cover the entire open web. Using multi-label classification, our best model achieves 79% F1 averaged across languages, matching or exceeding previous studies that used simpler classification schemes. This demonstrates that models can perform well even with a complex register scheme at multilingual scale. However, we observe a consistent performance ceiling across all models and configurations. When we remove documents with uncertain labels through data pruning, performance increases to over 90% F1, suggesting that this ceiling stems from inherent ambiguity in web registers rather than model limitations. Analysis of hybrid
    
[^185]: ETHER: 对于回顾性经验重演的紧密沟通对齐

    ETHER: Aligning Emergent Communication for Hindsight Experience Replay. (arXiv:2307.15494v1 [cs.CL])

    [http://arxiv.org/abs/2307.15494](http://arxiv.org/abs/2307.15494)

    本文提出了ETHER，通过对齐紧急沟通来解决回顾性经验重演中的问题，克服了先前架构依赖预设函数的限制，并提高了数据效率和性能。

    

    自然语言指令的跟随对于实现人工智能代理和人类之间的合作至关重要。自然语言条件下的强化学习代理展示了自然语言的特性，如组合性，能够提供学习复杂策略的强归纳偏好。先前的架构如HIGhER结合了语言条件与回顾性经验重演（HER）来处理稀疏奖励环境。然而，与HER类似，HIGhER依赖于一个预设的函数来提供反馈信号，指示哪种语言描述在哪种状态下有效。这种依赖于预设函数的限制限制了其应用。此外，HIGhER只利用成功的强化学习轨迹中包含的语言信息，从而影响了其最终性能和数据效率。没有早期成功轨迹，HIGhER并不比其构建于之上的DQN更好。在本文中，我们提出了紧密文本回顾性经验。

    Natural language instruction following is paramount to enable collaboration between artificial agents and human beings. Natural language-conditioned reinforcement learning (RL) agents have shown how natural languages' properties, such as compositionality, can provide a strong inductive bias to learn complex policies. Previous architectures like HIGhER combine the benefit of language-conditioning with Hindsight Experience Replay (HER) to deal with sparse rewards environments. Yet, like HER, HIGhER relies on an oracle predicate function to provide a feedback signal highlighting which linguistic description is valid for which state. This reliance on an oracle limits its application. Additionally, HIGhER only leverages the linguistic information contained in successful RL trajectories, thus hurting its final performance and data-efficiency. Without early successful trajectories, HIGhER is no better than DQN upon which it is built. In this paper, we propose the Emergent Textual Hindsight Ex
    

